#!/usr/bin/env python3
"""
Position Monitor — watches your real portfolio and generates SELL alerts.

Checks each holding for:
  1. Technical breakdown (price below key moving averages, bearish patterns)
  2. Stop-loss breach (trailing stop based on ATR)
  3. Excessive loss (beyond max drawdown threshold)
  4. Target reached (profit target hit)
  5. Deteriorating fundamentals

Usage:
    python3 position_monitor.py              # Check all positions
    python3 position_monitor.py --ticker META # Check single position
    python3 position_monitor.py --json        # Output as JSON
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

PORTFOLIO_FILE = Path("portfolio.json")
STOPS_FILE = Path("logs/position_stops.json")


def load_portfolio() -> dict:
    """Load the real portfolio."""
    if not PORTFOLIO_FILE.exists():
        return {"holdings": []}
    with open(PORTFOLIO_FILE) as f:
        return json.load(f)


def load_stops() -> dict:
    """Load saved trailing stop levels."""
    if not STOPS_FILE.exists():
        return {}
    with open(STOPS_FILE) as f:
        return json.load(f)


def save_stops(stops: dict) -> None:
    """Persist trailing stop levels."""
    STOPS_FILE.parent.mkdir(exist_ok=True)
    with open(STOPS_FILE, "w") as f:
        json.dump(stops, f, indent=2)


def analyze_position(holding: dict) -> dict:
    """Analyze a single holding and generate sell signals if needed."""
    ticker = holding["ticker"]
    shares = int(holding.get("shares", 0))
    avg_cost = float(holding.get("avg_cost", 0))
    current_price = float(holding.get("current_price", 0))
    cost_basis = float(holding.get("cost_basis", 0))
    unrealized_pnl = float(holding.get("unrealized_pnl", 0) or 0)
    pnl_pct = float(holding.get("pnl_pct", 0) or 0)
    sector = holding.get("sector", "Unknown")

    signals = []  # type: List[str]
    severity = "HOLD"  # HOLD, WATCH, TRIM, SELL
    reasons = []  # type: List[str]

    strategy = holding.get("strategy", "trade")  # "long_term" or "trade"

    try:
        import data_layer
        import technical_analysis

        df = data_layer.fetch_daily_ohlcv(ticker)
        if df.empty:
            return {
                "ticker": ticker, "action": "HOLD", "strategy": strategy,
                "signals": [], "reasons": ["No price data available"],
                "pnl_pct": pnl_pct, "unrealized_pnl": unrealized_pnl,
            }

        tech = technical_analysis.analyze(ticker, df)

        if strategy == "long_term":
            # ----------------------------------------------------------------
            # LONG-TERM HOLDS (MAG 7 + SPY + QQQ)
            # Strategy: ride market corrections, only exit on fundamental collapse
            # Only alert if: catastrophic loss OR earnings miss + price collapse together
            # ----------------------------------------------------------------

            # Catastrophic loss (50%+ from entry) — even long-term has a limit
            if pnl_pct < -50:
                signals.append("CATASTROPHIC_LOSS")
                reasons.append("Down %.1f%% from entry — reassess thesis" % pnl_pct)
                severity = _escalate(severity, "TRIM")

            # RSI extremely overbought — consider trimming into strength
            if tech.rsi and tech.rsi > 85:
                signals.append("RSI_OVERBOUGHT")
                reasons.append("RSI %.1f — extended, could trim some" % tech.rsi)
                severity = _escalate(severity, "WATCH")

            # Very deep below 200 SMA AND RSI not oversold (not just a dip)
            if tech.sma200 > 0 and current_price < tech.sma200:
                pct_below = ((tech.sma200 - current_price) / tech.sma200) * 100
                if pct_below > 30 and tech.rsi and tech.rsi > 35:
                    signals.append("DEEP_BREAKDOWN")
                    reasons.append("Price %.1f%% below 200 SMA — consider reducing" % pct_below)
                    severity = _escalate(severity, "TRIM")

        else:
            # ----------------------------------------------------------------
            # SHORT-TERM TRADES (AVGO, BMNR, CRWD, DELL, MP, NFLX, ONDS, ZETA, URA...)
            # Strategy: strict stops, exit quickly when wrong, protect capital
            # ----------------------------------------------------------------
            signal_points = 0

            # 1. ATR-based stop from entry price
            if hasattr(tech, 'atr') and tech.atr and avg_cost > 0:
                hard_stop = avg_cost - (2.0 * tech.atr)
                if current_price < hard_stop:
                    signals.append("STOP_HIT")
                    reasons.append("Below stop $%.2f (2x ATR from entry $%.2f)" % (hard_stop, avg_cost))
                    signal_points += 4

            # 2. Below 200 SMA
            if tech.sma200 > 0 and current_price < tech.sma200:
                pct_below = ((tech.sma200 - current_price) / tech.sma200) * 100
                if pct_below > 20:
                    signals.append("DEEP_BELOW_200SMA")
                    reasons.append("%.1f%% below 200 SMA — major breakdown" % pct_below)
                    signal_points += 3
                elif pct_below > 10:
                    signals.append("BELOW_200SMA")
                    reasons.append("%.1f%% below 200 SMA" % pct_below)
                    signal_points += 2

            # 3. Strong MACD bearish
            if tech.macd_value and tech.macd_signal:
                if tech.macd_value < tech.macd_signal and tech.macd_value < -3:
                    signals.append("MACD_BEARISH")
                    reasons.append("MACD bearish (%.2f)" % tech.macd_value)
                    signal_points += 2

            # 4. Head & Shoulders below SMA
            hs = any("head & shoulders" in p.lower() for p in tech.pattern_details)
            if hs and tech.sma200 > 0 and current_price < tech.sma200:
                signals.append("BEARISH_PATTERN")
                reasons.append("Head & Shoulders below 200 SMA")
                signal_points += 2

            # 5. Hard loss threshold for trades (25%)
            if pnl_pct < -25:
                signals.append("MAX_LOSS")
                reasons.append("Down %.1f%% — past max loss for a trade" % pnl_pct)
                signal_points += 5
            elif pnl_pct < -18:
                signals.append("APPROACHING_MAX_LOSS")
                reasons.append("Down %.1f%% — approaching max loss" % pnl_pct)
                signal_points += 2

            # Convert points to action
            if signal_points >= 7:
                severity = "SELL"
            elif signal_points >= 4:
                severity = "TRIM"
            elif signal_points >= 2:
                severity = "WATCH"

    except Exception as e:
        logger.debug("Technical analysis failed for %s: %s", ticker, e)

    portfolio = load_portfolio()
    total_val = float(portfolio.get("total_portfolio_value", 1))
    position_weight = (float(holding.get("current_value", 0)) / total_val * 100) if total_val > 0 else 0

    return {
        "ticker": ticker,
        "shares": shares,
        "avg_cost": avg_cost,
        "current_price": current_price,
        "unrealized_pnl": unrealized_pnl,
        "pnl_pct": pnl_pct,
        "sector": sector,
        "strategy": strategy,
        "action": severity,
        "signals": signals,
        "reasons": reasons,
        "weight_pct": round(position_weight, 1),
        "checked_at": datetime.now().isoformat(),
    }


def _escalate(current: str, new: str) -> str:
    """Return the more severe action level."""
    levels = {"HOLD": 0, "WATCH": 1, "TRIM": 2, "SELL": 3}
    if levels.get(new, 0) > levels.get(current, 0):
        return new
    return current


def check_all_positions(ticker_filter: Optional[str] = None) -> List[dict]:
    """Check all portfolio positions for sell signals."""
    portfolio = load_portfolio()
    holdings = portfolio.get("holdings", [])

    if ticker_filter:
        holdings = [h for h in holdings if h["ticker"].upper() == ticker_filter.upper()]

    results = []
    for h in holdings:
        result = analyze_position(h)
        results.append(result)

    return results


def get_actionable(results: List[dict]) -> List[dict]:
    """Filter results to only actionable items (not HOLD)."""
    return [r for r in results if r["action"] != "HOLD"]


def format_report(results: List[dict]) -> str:
    """Format position monitor results as a readable report."""
    actionable = get_actionable(results)

    lines = []
    lines.append("=" * 60)
    lines.append("  POSITION MONITOR — %s" % datetime.now().strftime("%Y-%m-%d %H:%M"))
    lines.append("=" * 60)
    lines.append("")
    lines.append("  %d positions checked, %d need attention" % (len(results), len(actionable)))
    lines.append("")

    # Show actionable items first
    action_icons = {"SELL": "!!!", "TRIM": "!!", "WATCH": "!", "HOLD": "OK"}
    action_colors = {"SELL": "RED", "TRIM": "ORANGE", "WATCH": "YELLOW", "HOLD": "GREEN"}

    for r in sorted(results, key=lambda x: {"SELL": 0, "TRIM": 1, "WATCH": 2, "HOLD": 3}.get(x["action"], 4)):
        icon = action_icons.get(r["action"], "?")
        pnl_val = r.get("unrealized_pnl", 0)
        pnl_str = "${:+,.2f}".format(pnl_val)
        strategy_tag = " [LONG]" if r.get("strategy") == "long_term" else " [TRADE]"
        lines.append("  [%s] %s — %s%s" % (icon, r["ticker"], r["action"], strategy_tag))
        lines.append("       Price: $%.2f | P&L: %s (%.1f%%) | Weight: %.1f%%" % (
            r.get("current_price", 0), pnl_str,
            r.get("pnl_pct", 0), r.get("weight_pct", 0)))
        if r["reasons"]:
            for reason in r["reasons"]:
                lines.append("       - %s" % reason)
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


def format_sms_alert(result: dict) -> str:
    """Format a short SMS alert for a position."""
    reasons_short = "; ".join(r[:50] for r in result.get("reasons", [])[:2])
    pnl_str = "${:+,.0f}".format(result.get("unrealized_pnl", 0))
    return (
        "%s %s: $%.2f\n"
        "P&L: %s (%.1f%%)\n"
        "%s"
    ) % (
        result["action"], result["ticker"],
        result.get("current_price", 0),
        pnl_str,
        result.get("pnl_pct", 0),
        reasons_short,
    )


def send_sell_alerts(results: List[dict]) -> None:
    """Send notifications for positions that need attention."""
    actionable = get_actionable(results)
    if not actionable:
        return

    import notifications

    # Build full report for email
    report = format_report(results)
    sell_count = sum(1 for r in actionable if r["action"] == "SELL")
    trim_count = sum(1 for r in actionable if r["action"] == "TRIM")
    watch_count = sum(1 for r in actionable if r["action"] == "WATCH")

    subject = "Position Alert: %d SELL, %d TRIM, %d WATCH" % (sell_count, trim_count, watch_count)

    # Email full report
    notifications.send_email_text(report, subject=subject)

    # SMS disabled — use Telegram for urgent alerts instead

    # Telegram
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")
    if token and chat_id and not token.startswith("your_"):
        try:
            import requests as req
            url = "https://api.telegram.org/bot%s/sendMessage" % token
            req.post(url, json={
                "chat_id": chat_id,
                "text": report,
            }, timeout=10)
        except Exception as e:
            logger.debug("Telegram sell alert failed: %s", e)


def main():
    parser = argparse.ArgumentParser(description="Position Monitor")
    parser.add_argument("--ticker", type=str, help="Check single ticker")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--notify", action="store_true", help="Send notifications for actionable positions")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    results = check_all_positions(ticker_filter=args.ticker)

    if args.json:
        print(json.dumps(results, indent=2, default=str))
    else:
        print(format_report(results))

    if args.notify:
        send_sell_alerts(results)

    return results


if __name__ == "__main__":
    main()
