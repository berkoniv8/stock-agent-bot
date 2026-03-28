#!/usr/bin/env python3
"""
Sector Rotation Analysis — detects money flow between sectors using ETF
performance comparison.

Tracks 11 sector ETFs, computes multi-timeframe returns and momentum scores,
identifies rotation signals, and cross-references against portfolio holdings.

Sector ETFs tracked:
    XLK  Technology          XLF  Financials         XLV  Healthcare
    XLE  Energy              XLY  Consumer Disc.      XLP  Consumer Staples
    XLI  Industrials         XLB  Materials           XLU  Utilities
    XLRE Real Estate         XLC  Communication

Usage:
    python3 sector_rotation.py           # Full rotation report
    python3 sector_rotation.py --json    # Output as JSON
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import yfinance as yf
import pandas as pd

logger = logging.getLogger(__name__)

PORTFOLIO_FILE = Path("portfolio.json")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ETF symbol → human-readable sector name
SECTOR_ETFS: Dict[str, str] = {
    "XLK": "Technology",
    "XLF": "Financials",
    "XLV": "Healthcare",
    "XLE": "Energy",
    "XLY": "Consumer Discretionary",
    "XLP": "Consumer Staples",
    "XLI": "Industrials",
    "XLB": "Materials",
    "XLU": "Utilities",
    "XLRE": "Real Estate",
    "XLC": "Communication",
}

# Normalised sector name aliases used in portfolio.json / yfinance info
SECTOR_ALIASES: Dict[str, str] = {
    "Technology": "Technology",
    "Information Technology": "Technology",
    "Financials": "Financials",
    "Financial Services": "Financials",
    "Healthcare": "Healthcare",
    "Health Care": "Healthcare",
    "Energy": "Energy",
    "Consumer Discretionary": "Consumer Discretionary",
    "Consumer Cyclical": "Consumer Discretionary",
    "Consumer Staples": "Consumer Staples",
    "Consumer Defensive": "Consumer Staples",
    "Industrials": "Industrials",
    "Materials": "Materials",
    "Basic Materials": "Materials",
    "Utilities": "Utilities",
    "Real Estate": "Real Estate",
    "Communication": "Communication",
    "Communication Services": "Communication",
}

# Reverse map: canonical sector → ETF
SECTOR_TO_ETF: Dict[str, str] = {v: k for k, v in SECTOR_ETFS.items()}


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _compute_return(close: "pd.Series", n_bars: int) -> float:
    """Return percentage change over the last n_bars bars, or 0 if insufficient data."""
    if len(close) <= n_bars:
        return 0.0
    prev = float(close.iloc[-n_bars - 1])
    curr = float(close.iloc[-1])
    if prev <= 0:
        return 0.0
    return round((curr - prev) / prev * 100, 4)


def _momentum_score(ret_1w: float, ret_1m: float, ret_3m: float) -> float:
    """Weighted momentum: 1w × 0.5 + 1m × 0.3 + 3m × 0.2."""
    return round(ret_1w * 0.5 + ret_1m * 0.3 + ret_3m * 0.2, 4)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_sector_performance(period: str = "1mo") -> Dict[str, dict]:
    """Fetch price data for all 11 sector ETFs and compute performance metrics.

    Returns a dict keyed by canonical sector name:
        {
            "Technology": {
                "etf": "XLK",
                "return_1w": 2.3,
                "return_1m": -5.1,
                "return_3m": 8.2,
                "momentum_score": 7.2,
                "rank": 1,
            },
            ...
        }
    """
    # Always fetch at least 3 months so we can compute 3m returns regardless
    # of the requested `period` (which mainly influences recency of 1w/1m).
    fetch_period = "6mo"

    raw: Dict[str, dict] = {}

    for etf, sector in SECTOR_ETFS.items():
        try:
            tk = yf.Ticker(etf)
            df = tk.history(period=fetch_period, interval="1d")
            if df is None or df.empty:
                logger.warning("No data returned for %s", etf)
                continue

            close = df["Close"].dropna()

            ret_1w = _compute_return(close, 5)
            ret_1m = _compute_return(close, 21)
            ret_3m = _compute_return(close, 63)
            mom = _momentum_score(ret_1w, ret_1m, ret_3m)

            raw[sector] = {
                "etf": etf,
                "return_1w": ret_1w,
                "return_1m": ret_1m,
                "return_3m": ret_3m,
                "momentum_score": mom,
                "rank": 0,  # filled in below
            }
        except Exception as e:
            logger.error("Failed to fetch %s (%s): %s", etf, sector, e)

    # Rank by momentum score descending
    sorted_sectors = sorted(raw.keys(), key=lambda s: raw[s]["momentum_score"], reverse=True)
    for rank, sector in enumerate(sorted_sectors, start=1):
        raw[sector]["rank"] = rank

    return raw


def detect_rotation(performance: Dict[str, dict]) -> Dict[str, list]:
    """Detect sector rotation by comparing short-term vs medium-term momentum.

    A sector is "rotating into" when its 1-week return strongly exceeds its
    1-month return (recent acceleration).  It is "rotating out of" when the
    opposite is true (recent deceleration / reversal).

    Returns:
        {
            "rotating_into":    ["Energy", "Healthcare"],
            "rotating_out_of":  ["Technology", "Consumer Discretionary"],
            "momentum_leaders": ["Energy", "Healthcare", "Industrials"],
            "momentum_laggards":["Technology", "Real Estate"],
        }
    """
    rotating_into: List[str] = []
    rotating_out_of: List[str] = []

    # Acceleration threshold: 1w return must exceed 1m return by this margin (pp)
    ACCEL_THRESHOLD = 1.5

    for sector, data in performance.items():
        accel = data["return_1w"] - data["return_1m"]
        if accel > ACCEL_THRESHOLD:
            rotating_into.append(sector)
        elif accel < -ACCEL_THRESHOLD:
            rotating_out_of.append(sector)

    # Leaders/laggards by momentum score (top/bottom third)
    sorted_by_mom = sorted(
        performance.keys(),
        key=lambda s: performance[s]["momentum_score"],
        reverse=True,
    )
    n = max(1, len(sorted_by_mom) // 3)
    momentum_leaders = sorted_by_mom[:n]
    momentum_laggards = sorted_by_mom[-n:]

    return {
        "rotating_into": rotating_into,
        "rotating_out_of": rotating_out_of,
        "momentum_leaders": momentum_leaders,
        "momentum_laggards": momentum_laggards,
    }


def get_portfolio_exposure(rotation: Dict[str, list], portfolio: dict) -> Dict[str, list]:
    """Compare portfolio sector weights against rotation signals.

    Accepts a portfolio dict (same shape as portfolio.json) or a pre-loaded dict.

    Returns:
        {
            "aligned":     [{"ticker": "XLE", "sector": "Energy", "signal": "rotating_into"}],
            "misaligned":  [...],
            "suggestions": ["Consider reducing Technology weight", ...],
        }
    """
    holdings = portfolio.get("holdings", [])
    total_value = sum(h.get("current_value", 0) for h in holdings)
    if total_value <= 0:
        total_value = 1.0  # avoid division by zero

    # Compute weight by sector
    sector_weights: Dict[str, float] = {}
    sector_tickers: Dict[str, List[str]] = {}
    for h in holdings:
        raw_sector = h.get("sector", "")
        canonical = SECTOR_ALIASES.get(raw_sector, raw_sector)
        if canonical not in SECTOR_ETFS.values():
            continue
        weight = h.get("current_value", 0) / total_value * 100
        sector_weights[canonical] = sector_weights.get(canonical, 0) + weight
        sector_tickers.setdefault(canonical, []).append(h["ticker"])

    rotating_into = set(rotation.get("rotating_into", []))
    rotating_out_of = set(rotation.get("rotating_out_of", []))
    leaders = set(rotation.get("momentum_leaders", []))
    laggards = set(rotation.get("momentum_laggards", []))

    aligned: List[dict] = []
    misaligned: List[dict] = []
    suggestions: List[str] = []

    for sector, weight in sector_weights.items():
        tickers = sector_tickers.get(sector, [])
        entry = {"sector": sector, "tickers": tickers, "weight_pct": round(weight, 2)}

        if sector in rotating_into or sector in leaders:
            entry["signal"] = "rotating_into" if sector in rotating_into else "momentum_leader"
            aligned.append(entry)
        elif sector in rotating_out_of or sector in laggards:
            entry["signal"] = "rotating_out_of" if sector in rotating_out_of else "momentum_laggard"
            misaligned.append(entry)

    # Suggestions
    for item in misaligned:
        s = item["sector"]
        signal = item.get("signal", "")
        if signal == "rotating_out_of":
            suggestions.append(
                f"Consider reducing {s} weight — capital rotating out"
            )
        elif signal == "momentum_laggard":
            suggestions.append(
                f"Consider reducing {s} weight — momentum lagging peers"
            )

    for s in rotating_into:
        if s not in sector_weights:
            suggestions.append(
                f"{s} showing relative strength — consider adding exposure"
            )

    for s in leaders:
        if s not in sector_weights:
            suggestions.append(
                f"{s} is a momentum leader — no current exposure"
            )

    return {
        "aligned": aligned,
        "misaligned": misaligned,
        "suggestions": suggestions,
    }


def format_report(
    performance: Dict[str, dict],
    rotation: Dict[str, list],
    exposure: Dict[str, list],
) -> str:
    """Build a complete, human-readable sector rotation report."""
    lines = []
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines.append("=" * 72)
    lines.append("  SECTOR ROTATION ANALYSIS")
    lines.append(f"  Generated: {now_str}")
    lines.append("=" * 72)

    # ---- Performance table ----
    lines.append("\n  SECTOR PERFORMANCE RANKINGS")
    lines.append("  " + "-" * 65)
    header = f"  {'#':>2}  {'Sector':<25}  {'ETF':<5}  {'1w%':>6}  {'1m%':>6}  {'3m%':>6}  {'Mom':>6}"
    lines.append(header)
    lines.append("  " + "-" * 65)

    sorted_sectors = sorted(performance.keys(), key=lambda s: performance[s]["rank"])
    for sector in sorted_sectors:
        d = performance[sector]
        lines.append(
            f"  {d['rank']:>2}  {sector:<25}  {d['etf']:<5}"
            f"  {d['return_1w']:>+5.1f}%"
            f"  {d['return_1m']:>+5.1f}%"
            f"  {d['return_3m']:>+5.1f}%"
            f"  {d['momentum_score']:>+5.1f}"
        )

    # ---- Rotation signals ----
    lines.append("\n  ROTATION SIGNALS")
    lines.append("  " + "-" * 65)

    rotating_into = rotation.get("rotating_into", [])
    rotating_out_of = rotation.get("rotating_out_of", [])
    leaders = rotation.get("momentum_leaders", [])
    laggards = rotation.get("momentum_laggards", [])

    lines.append(
        f"  Rotating INTO:    {', '.join(rotating_into) if rotating_into else 'None detected'}"
    )
    lines.append(
        f"  Rotating OUT OF:  {', '.join(rotating_out_of) if rotating_out_of else 'None detected'}"
    )
    lines.append(f"  Momentum Leaders: {', '.join(leaders) if leaders else '—'}")
    lines.append(f"  Momentum Laggards:{' ' + ', '.join(laggards) if laggards else ' —'}")

    # ---- Portfolio exposure ----
    aligned = exposure.get("aligned", [])
    misaligned = exposure.get("misaligned", [])
    suggestions = exposure.get("suggestions", [])

    if aligned or misaligned:
        lines.append("\n  PORTFOLIO ALIGNMENT")
        lines.append("  " + "-" * 65)

        if aligned:
            lines.append("  Aligned with rotation:")
            for item in aligned:
                tickers_str = ", ".join(item.get("tickers", []))
                lines.append(
                    f"    + {item['sector']:<25} {item['weight_pct']:>5.1f}%"
                    f"  [{item.get('signal', '')}]  ({tickers_str})"
                )

        if misaligned:
            lines.append("  Misaligned with rotation:")
            for item in misaligned:
                tickers_str = ", ".join(item.get("tickers", []))
                lines.append(
                    f"    - {item['sector']:<25} {item['weight_pct']:>5.1f}%"
                    f"  [{item.get('signal', '')}]  ({tickers_str})"
                )

    if suggestions:
        lines.append("\n  SUGGESTIONS")
        lines.append("  " + "-" * 65)
        for s in suggestions:
            lines.append(f"  * {s}")

    lines.append("\n" + "=" * 72)
    return "\n".join(lines)


def send_alert(rotation: Dict[str, list], exposure: Dict[str, list]) -> None:
    """Send an email alert when significant rotation is detected.

    Significant = at least 2 sectors rotating in or out, OR portfolio has
    misaligned holdings.
    """
    import notifications

    rotating_into = rotation.get("rotating_into", [])
    rotating_out_of = rotation.get("rotating_out_of", [])
    misaligned = exposure.get("misaligned", [])

    significant = len(rotating_into) >= 2 or len(rotating_out_of) >= 2 or len(misaligned) >= 1

    if not significant:
        logger.info("No significant rotation detected — skipping alert")
        return

    suggestions = exposure.get("suggestions", [])
    lines = [
        "SECTOR ROTATION ALERT",
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Rotating INTO:  {', '.join(rotating_into) if rotating_into else 'None'}",
        f"Rotating OUT OF: {', '.join(rotating_out_of) if rotating_out_of else 'None'}",
    ]
    if misaligned:
        lines.append("")
        lines.append("Portfolio misaligned with rotation:")
        for item in misaligned:
            lines.append(f"  - {item['sector']} ({item['weight_pct']:.1f}%) [{item.get('signal','')}]")
    if suggestions:
        lines.append("")
        lines.append("Suggestions:")
        for s in suggestions:
            lines.append(f"  * {s}")

    text = "\n".join(lines)
    subject = "Sector Rotation Alert — " + datetime.now().strftime("%Y-%m-%d")
    sent = notifications.send_email_text(text, subject=subject)
    if sent:
        logger.info("Sector rotation alert email sent")
    else:
        logger.info("Sector rotation alert email not sent (not configured)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _load_portfolio() -> dict:
    if not PORTFOLIO_FILE.exists():
        logger.warning("portfolio.json not found — exposure analysis skipped")
        return {}
    try:
        with open(PORTFOLIO_FILE, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.error("Failed to load portfolio.json: %s", e)
        return {}


def main():
    parser = argparse.ArgumentParser(description="Sector Rotation Analysis")
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw data as JSON instead of formatted report",
    )
    parser.add_argument(
        "--send",
        action="store_true",
        help="Send email alert if significant rotation detected",
    )
    parser.add_argument(
        "--period",
        default="1mo",
        help="yfinance period string used as context label (default: 1mo)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    for lib in ("urllib3", "yfinance", "peewee"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    performance = fetch_sector_performance(period=args.period)
    rotation = detect_rotation(performance)
    portfolio = _load_portfolio()
    exposure = get_portfolio_exposure(rotation, portfolio)

    if args.json:
        output = {
            "generated_at": datetime.now().isoformat(),
            "performance": performance,
            "rotation": rotation,
            "exposure": exposure,
        }
        print(json.dumps(output, indent=2))
    else:
        print(format_report(performance, rotation, exposure))

    if args.send:
        send_alert(rotation, exposure)


if __name__ == "__main__":
    main()
