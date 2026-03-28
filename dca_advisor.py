#!/usr/bin/env python3
"""
DCA Advisor — suggests Dollar Cost Averaging levels for long-term positions
that are currently underwater.

Eligibility: strategy == "long_term" AND pnl_pct < -5%.

Support levels are derived from three sources:
  1. Recent swing lows in daily price data (rolling-window local minima)
  2. Round-number price levels below current price
  3. Fibonacci retracements from the all-time high (ATH) to the current price

DCA share amounts are sized at 0.5% of total portfolio value per level.

Usage:
    python3 dca_advisor.py                # Full portfolio DCA report
    python3 dca_advisor.py --ticker META  # Single ticker
    python3 dca_advisor.py --json         # Machine-readable output
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

PORTFOLIO_FILE = Path(__file__).parent / "portfolio.json"

# Fibonacci retracement ratios (from ATH down)
FIB_RATIOS = [0.236, 0.382, 0.500, 0.618, 0.786]

# DCA level labels keyed by descending priority
DCA_LABELS = ["Strong support", "Major support", "Key support", "Deep support"]

# Portfolio percentage allocated per DCA buy (0.5%)
DCA_PCT_PER_LEVEL = 0.005


# ---------------------------------------------------------------------------
# Portfolio helpers
# ---------------------------------------------------------------------------

def _load_portfolio() -> dict:
    """Load portfolio.json."""
    if not PORTFOLIO_FILE.exists():
        return {"holdings": [], "total_portfolio_value": 0}
    with open(PORTFOLIO_FILE) as f:
        return json.load(f)


def _total_portfolio_value() -> float:
    """Return the total portfolio value from portfolio.json."""
    return float(_load_portfolio().get("total_portfolio_value", 0))


# ---------------------------------------------------------------------------
# Support level detection
# ---------------------------------------------------------------------------

def _swing_lows(df: pd.DataFrame, window: int = 10, max_levels: int = 5) -> List[float]:
    """Identify significant swing lows from the close-price series.

    A swing low is a candle whose low is the minimum over a symmetric window
    of *window* bars on each side.  Only levels below the most-recent close
    are returned, sorted ascending.
    """
    if df is None or df.empty or len(df) < window * 2 + 1:
        return []

    lows = df["Low"] if "Low" in df.columns else df["Close"]
    current_price = float(df["Close"].iloc[-1])

    swing_low_prices: List[float] = []
    for i in range(window, len(lows) - window):
        segment = lows.iloc[i - window : i + window + 1]
        local_min = float(segment.min())
        if float(lows.iloc[i]) == local_min:
            swing_low_prices.append(local_min)

    # Keep unique values below current price, deduplicated within 1%
    below = sorted(set(p for p in swing_low_prices if p < current_price), reverse=True)
    deduped: List[float] = []
    for price in below:
        if not deduped or abs(price - deduped[-1]) / deduped[-1] > 0.01:
            deduped.append(price)
        if len(deduped) >= max_levels:
            break

    return sorted(deduped)


def _round_number_levels(current_price: float, count: int = 4) -> List[float]:
    """Generate round-number price levels below the current price.

    For prices above $100 the round interval is $10; for $10-$100 it is $5;
    for prices below $10 it is $1.
    """
    if current_price >= 100:
        interval = 10.0
    elif current_price >= 10:
        interval = 5.0
    else:
        interval = 1.0

    # Start just below current price and step down
    start = (current_price // interval) * interval
    levels = [start - interval * i for i in range(1, count + 3)]
    return sorted([l for l in levels if l > 0])[:count]


def _fibonacci_levels(df: pd.DataFrame, current_price: float) -> List[float]:
    """Compute Fibonacci retracement levels from the ATH to the current price.

    Returns levels below the current price sorted ascending.
    """
    if df is None or df.empty:
        return []

    highs = df["High"] if "High" in df.columns else df["Close"]
    ath = float(highs.max())
    if ath <= current_price:
        return []

    drop = ath - current_price
    levels = [round(ath - ratio * drop, 2) for ratio in FIB_RATIOS]
    below = sorted(set(l for l in levels if l < current_price))
    return below


def _combine_support_levels(
    swing: List[float],
    round_nums: List[float],
    fibs: List[float],
    current_price: float,
    max_levels: int = 5,
) -> List[float]:
    """Merge the three support sources, deduplicate within 1%, sort descending.

    Levels are sorted from closest-to-current down so the first DCA is the
    nearest actionable support.
    """
    combined = swing + round_nums + fibs
    combined = sorted(set(p for p in combined if 0 < p < current_price), reverse=True)

    deduped: List[float] = []
    for price in combined:
        if not deduped or abs(price - deduped[-1]) / deduped[-1] > 0.01:
            deduped.append(round(price, 2))
        if len(deduped) >= max_levels:
            break

    return deduped  # highest (nearest current) first


# ---------------------------------------------------------------------------
# DCA math
# ---------------------------------------------------------------------------

def _new_average_cost(
    current_shares: int,
    current_avg: float,
    add_shares: int,
    buy_price: float,
) -> float:
    """Compute the blended average cost after adding shares."""
    total_shares = current_shares + add_shares
    if total_shares == 0:
        return 0.0
    total_cost = current_shares * current_avg + add_shares * buy_price
    return round(total_cost / total_shares, 2)


def _shares_to_add(portfolio_value: float, buy_price: float) -> int:
    """Return the number of whole shares to buy using 0.5% of portfolio value."""
    if buy_price <= 0:
        return 0
    dollar_amount = portfolio_value * DCA_PCT_PER_LEVEL
    return max(1, int(dollar_amount / buy_price))


# ---------------------------------------------------------------------------
# Core analysis function
# ---------------------------------------------------------------------------

def analyze_dca(holding: dict, df_daily: Optional[pd.DataFrame]) -> dict:
    """Analyse a single holding and return DCA suggestion levels.

    Parameters
    ----------
    holding:
        One entry from portfolio.json["holdings"].
    df_daily:
        Daily OHLCV DataFrame for the ticker (must have Close and ideally
        High/Low columns).  If None or empty, technical support is skipped
        and only round-number levels are used.

    Returns
    -------
    dict with keys: ticker, current_price, avg_cost, pnl_pct, strategy,
    support_levels, dca_levels, recommendation.
    """
    ticker = holding["ticker"]
    current_price = float(holding.get("current_price", 0))
    avg_cost = float(holding.get("avg_cost", 0))
    shares = int(holding.get("shares", 0))
    pnl_pct = float(holding.get("pnl_pct", 0))
    strategy = holding.get("strategy", "trade")

    portfolio_value = _total_portfolio_value()

    # --- Detect support levels ---
    swing = _swing_lows(df_daily) if df_daily is not None and not df_daily.empty else []
    round_nums = _round_number_levels(current_price)
    fibs = _fibonacci_levels(df_daily, current_price) if df_daily is not None and not df_daily.empty else []
    support_levels = _combine_support_levels(swing, round_nums, fibs, current_price)

    # --- Build DCA level entries ---
    dca_levels: List[dict] = []
    for i, price in enumerate(support_levels[:4]):
        add = _shares_to_add(portfolio_value, price)
        new_avg = _new_average_cost(shares, avg_cost, add, price)
        label = DCA_LABELS[i] if i < len(DCA_LABELS) else "Support"
        dollar_cost = round(add * price, 2)
        dca_levels.append(
            {
                "price": price,
                "shares_to_add": add,
                "dollar_cost": dollar_cost,
                "new_avg_cost": new_avg,
                "label": label,
            }
        )

    # --- Recommendation text ---
    if dca_levels:
        first = dca_levels[0]
        if len(dca_levels) >= 2:
            second = dca_levels[1]
            rec = (
                "Next DCA zone: ${:.0f}–${:.0f} ({})".format(
                    second["price"], first["price"], first["label"].lower()
                )
            )
        else:
            rec = "Next DCA zone: ${:.2f} ({})".format(first["price"], first["label"].lower())
    else:
        rec = "No clear support levels identified below current price"

    return {
        "ticker": ticker,
        "current_price": round(current_price, 2),
        "avg_cost": round(avg_cost, 2),
        "pnl_pct": round(pnl_pct, 2),
        "strategy": strategy,
        "support_levels": support_levels,
        "dca_levels": dca_levels,
        "recommendation": rec,
    }


# ---------------------------------------------------------------------------
# Portfolio-wide analysis
# ---------------------------------------------------------------------------

def analyze_portfolio_dca() -> List[dict]:
    """Run DCA analysis on all eligible long-term holdings.

    Eligibility: strategy == "long_term" AND pnl_pct < -5%.

    Returns a list of analysis dicts sorted by pnl_pct ascending (worst first).
    """
    portfolio = _load_portfolio()
    holdings = portfolio.get("holdings", [])

    eligible = [
        h for h in holdings
        if h.get("strategy") == "long_term" and float(h.get("pnl_pct", 0)) < -5.0
    ]

    if not eligible:
        return []

    try:
        import data_layer
    except ImportError:
        data_layer = None

    results: List[dict] = []
    for h in eligible:
        df = None
        if data_layer is not None:
            try:
                df = data_layer.fetch_daily_ohlcv(h["ticker"])
            except Exception as exc:
                logger.warning("%s: could not fetch price data — %s", h["ticker"], exc)

        result = analyze_dca(h, df)
        results.append(result)

    results.sort(key=lambda x: x["pnl_pct"])
    return results


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def format_report(dca_list: List[dict]) -> str:
    """Render a readable DCA advisory report."""
    lines: List[str] = []
    now = datetime.now().strftime("%Y-%m-%d %H:%M")

    lines.append("=" * 65)
    lines.append("  DCA ADVISOR — %s" % now)
    lines.append("=" * 65)

    if not dca_list:
        lines.append("")
        lines.append("  No eligible positions (need long_term strategy with pnl < -5%).")
        lines.append("")
        lines.append("=" * 65)
        return "\n".join(lines)

    for item in dca_list:
        ticker = item["ticker"]
        curr = item["current_price"]
        avg = item["avg_cost"]
        pnl = item["pnl_pct"]
        support = item["support_levels"]
        levels = item["dca_levels"]
        rec = item["recommendation"]

        pnl_str = "{:+.1f}%".format(pnl)
        lines.append("")
        lines.append("  %-6s  $%-8.2f  avg cost $%-8.2f  P&L %s" % (ticker, curr, avg, pnl_str))
        lines.append("  " + "─" * 61)

        if support:
            sup_str = ", ".join("${:.2f}".format(s) for s in support[:5])
            lines.append("  Support levels : %s" % sup_str)
        else:
            lines.append("  Support levels : (none detected)")

        if levels:
            lines.append("  DCA plan:")
            for lvl in levels:
                lines.append(
                    "    [%s]  buy %d sh @ $%.2f = $%.0f  →  new avg $%.2f"
                    % (
                        lvl["label"],
                        lvl["shares_to_add"],
                        lvl["price"],
                        lvl["dollar_cost"],
                        lvl["new_avg_cost"],
                    )
                )
        else:
            lines.append("  DCA plan : No levels found")

        lines.append("  Rec: %s" % rec)

    lines.append("")
    lines.append("  %d position(s) analysed — DCA sizing = 0.5%% of portfolio per level" % len(dca_list))
    lines.append("=" * 65)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="DCA Advisor — suggests averaging-down levels")
    parser.add_argument("--ticker", type=str, help="Analyse a single ticker")
    parser.add_argument("--json", action="store_true", dest="as_json", help="Output as JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")

    if args.ticker:
        portfolio = _load_portfolio()
        holdings = portfolio.get("holdings", [])
        matches = [h for h in holdings if h["ticker"].upper() == args.ticker.upper()]
        if not matches:
            print("Ticker %s not found in portfolio." % args.ticker.upper())
            return
        holding = matches[0]

        df = None
        try:
            import data_layer
            df = data_layer.fetch_daily_ohlcv(holding["ticker"])
        except Exception:
            pass

        result = analyze_dca(holding, df)
        dca_list = [result]
    else:
        dca_list = analyze_portfolio_dca()

    if args.as_json:
        print(json.dumps(dca_list, indent=2, default=str))
    else:
        print(format_report(dca_list))


if __name__ == "__main__":
    main()
