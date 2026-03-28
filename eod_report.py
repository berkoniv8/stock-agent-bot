#!/usr/bin/env python3
"""
End-of-Day Report Generator — produces a daily P&L summary after market close.

Report includes:
- Daily P&L (realized + unrealized)
- Position changes (entries, exits, partial fills)
- Alert activity
- Win/loss breakdown for the day
- Portfolio snapshot

Usage:
    python3 eod_report.py               # Generate today's report
    python3 eod_report.py --date 2026-03-25  # Specific date
    python3 eod_report.py --json         # Output as JSON
    python3 eod_report.py --send         # Generate and send via notifications
"""

import argparse
import json
import logging
import os
from datetime import datetime, date
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

REPORTS_DIR = Path("logs/eod_reports")


def load_paper_state():
    # type: () -> Dict
    """Load paper trading state."""
    state_file = Path("logs/paper_state.json")
    if not state_file.exists():
        return {}
    try:
        with open(state_file) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def get_todays_trades(state, report_date):
    # type: (Dict, str) -> Dict
    """Extract trades that occurred on the given date.

    Returns dict with 'entries' and 'exits' lists.
    """
    entries = []
    exits = []

    for pos in state.get("open_positions", []):
        entry_date = str(pos.get("entry_date", ""))[:10]
        if entry_date == report_date:
            entries.append({
                "ticker": pos.get("ticker", ""),
                "direction": pos.get("direction", "BUY"),
                "shares": pos.get("shares", 0),
                "entry_price": pos.get("entry_price", 0),
                "stop_loss": pos.get("stop_loss", 0),
            })

    for trade in state.get("closed_trades", []):
        exit_date = str(trade.get("exit_date", ""))[:10]
        if exit_date == report_date:
            exits.append({
                "ticker": trade.get("ticker", ""),
                "direction": trade.get("direction", "BUY"),
                "pnl": float(trade.get("pnl", 0)),
                "pnl_pct": float(trade.get("pnl_pct", 0)),
                "exit_reason": trade.get("exit_reason", "unknown"),
                "bars_held": trade.get("bars_held", 0),
            })

    return {"entries": entries, "exits": exits}


def compute_unrealized_pnl(state):
    # type: (Dict) -> List[Dict]
    """Compute unrealized P&L for open positions.

    Uses entry price as reference (no live price fetch in report generation).
    """
    positions = []
    for pos in state.get("open_positions", []):
        ticker = pos.get("ticker", "")
        entry = float(pos.get("entry_price", 0))
        current_stop = float(pos.get("current_stop", pos.get("stop_loss", 0)))
        shares = int(pos.get("shares", 0))

        # Risk at current stop
        if pos.get("direction", "BUY") == "BUY":
            risk_per_share = entry - current_stop if current_stop > 0 else 0
        else:
            risk_per_share = current_stop - entry if current_stop > 0 else 0

        positions.append({
            "ticker": ticker,
            "direction": pos.get("direction", "BUY"),
            "shares": shares,
            "entry_price": entry,
            "current_stop": current_stop,
            "risk_at_stop": round(risk_per_share * shares, 2),
            "targets_hit": sum(1 for k in ["t1_hit", "t2_hit", "t3_hit"]
                               if pos.get(k)),
        })
    return positions


def get_alert_activity(report_date):
    # type: (str) -> List[Dict]
    """Get alerts that triggered on the given date."""
    alerts_file = Path("logs/price_alerts.json")
    if not alerts_file.exists():
        return []

    try:
        with open(alerts_file) as f:
            alerts = json.load(f)
    except (json.JSONDecodeError, IOError):
        return []

    triggered = []
    for a in alerts:
        triggered_at = str(a.get("triggered_at", ""))[:10]
        if triggered_at == report_date and a.get("triggered"):
            triggered.append({
                "ticker": a.get("ticker", ""),
                "type": a.get("type", "CUSTOM"),
                "condition": a.get("condition", ""),
                "price": a.get("price", 0),
                "triggered_price": a.get("triggered_price", 0),
            })
    return triggered


def generate_report(report_date=None):
    # type: (Optional[str]) -> Dict
    """Generate end-of-day report.

    Args:
        report_date: Date string (YYYY-MM-DD). Defaults to today.

    Returns dict with all report sections.
    """
    if report_date is None:
        report_date = date.today().isoformat()

    state = load_paper_state()
    trades = get_todays_trades(state, report_date)
    positions = compute_unrealized_pnl(state)
    alerts = get_alert_activity(report_date)

    # Daily realized P&L
    realized_pnl = sum(t["pnl"] for t in trades["exits"])
    wins = [t for t in trades["exits"] if t["pnl"] > 0]
    losses = [t for t in trades["exits"] if t["pnl"] <= 0]

    # Portfolio snapshot
    capital = float(state.get("capital", 0))
    initial = float(os.getenv("INITIAL_CAPITAL", "100000"))
    total_realized = sum(float(t.get("pnl", 0)) for t in state.get("closed_trades", []))

    report = {
        "date": report_date,
        "generated_at": datetime.now().isoformat(),
        "portfolio": {
            "capital": round(capital, 2),
            "initial_capital": round(initial, 2),
            "total_realized_pnl": round(total_realized, 2),
            "return_pct": round((capital - initial) / initial * 100, 2) if initial > 0 else 0,
            "open_positions": len(positions),
        },
        "daily_activity": {
            "entries": len(trades["entries"]),
            "exits": len(trades["exits"]),
            "realized_pnl": round(realized_pnl, 2),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / len(trades["exits"]) * 100, 1) if trades["exits"] else 0,
            "best_trade": max((t["pnl"] for t in trades["exits"]), default=0),
            "worst_trade": min((t["pnl"] for t in trades["exits"]), default=0),
        },
        "new_entries": trades["entries"],
        "closed_trades": trades["exits"],
        "open_positions": positions,
        "alerts_triggered": alerts,
    }

    return report


def format_report(report):
    # type: (Dict) -> str
    """Format report as readable text."""
    lines = []
    d = report["date"]
    p = report["portfolio"]
    da = report["daily_activity"]

    lines.append("")
    lines.append("=" * 60)
    lines.append("  END-OF-DAY REPORT — %s" % d)
    lines.append("=" * 60)

    # Portfolio
    lines.append("")
    lines.append("  PORTFOLIO")
    lines.append("  " + "-" * 50)
    lines.append("  Capital:     $%s" % "{:,.2f}".format(p["capital"]))
    lines.append("  Total P&L:   $%s (%.2f%%)" % (
        "{:,.2f}".format(p["total_realized_pnl"]), p["return_pct"]))
    lines.append("  Open:        %d positions" % p["open_positions"])

    # Daily activity
    lines.append("")
    lines.append("  TODAY'S ACTIVITY")
    lines.append("  " + "-" * 50)
    if da["entries"] > 0 or da["exits"] > 0:
        lines.append("  Entries:     %d" % da["entries"])
        lines.append("  Exits:       %d  (W: %d / L: %d)" % (
            da["exits"], da["wins"], da["losses"]))
        if da["exits"] > 0:
            lines.append("  Realized:    $%s  WR: %.1f%%" % (
                "{:,.2f}".format(da["realized_pnl"]), da["win_rate"]))
            lines.append("  Best:        $%s" % "{:,.2f}".format(da["best_trade"]))
            lines.append("  Worst:       $%s" % "{:,.2f}".format(da["worst_trade"]))
    else:
        lines.append("  No trades today.")

    # New entries
    if report["new_entries"]:
        lines.append("")
        lines.append("  NEW ENTRIES")
        lines.append("  " + "-" * 50)
        for e in report["new_entries"]:
            lines.append("  %-6s %4s  %d shares @ $%.2f  stop: $%.2f" % (
                e["ticker"], e["direction"], e["shares"],
                e["entry_price"], e["stop_loss"]))

    # Closed trades
    if report["closed_trades"]:
        lines.append("")
        lines.append("  CLOSED TRADES")
        lines.append("  " + "-" * 50)
        for t in report["closed_trades"]:
            lines.append("  %-6s %4s  P&L: $%+.2f (%+.1f%%)  %s  %d bars" % (
                t["ticker"], t["direction"], t["pnl"], t["pnl_pct"],
                t["exit_reason"], t["bars_held"]))

    # Open positions
    if report["open_positions"]:
        lines.append("")
        lines.append("  OPEN POSITIONS")
        lines.append("  " + "-" * 50)
        for pos in report["open_positions"]:
            targets = "T%d hit" % pos["targets_hit"] if pos["targets_hit"] > 0 else "no targets"
            lines.append("  %-6s %4s  %d shares @ $%.2f  stop: $%.2f  %s" % (
                pos["ticker"], pos["direction"], pos["shares"],
                pos["entry_price"], pos["current_stop"], targets))

    # Alerts
    if report["alerts_triggered"]:
        lines.append("")
        lines.append("  ALERTS TRIGGERED")
        lines.append("  " + "-" * 50)
        for a in report["alerts_triggered"]:
            lines.append("  %-6s %s %s $%.2f (hit at $%.2f)" % (
                a["ticker"], a["type"], a["condition"],
                a["price"], a["triggered_price"]))

    lines.append("")
    lines.append("=" * 60)
    lines.append("")

    return "\n".join(lines)


def save_report(report):
    # type: (Dict) -> str
    """Save report to disk. Returns file path."""
    os.makedirs(str(REPORTS_DIR), exist_ok=True)
    filename = "eod_%s.json" % report["date"]
    filepath = REPORTS_DIR / filename
    with open(str(filepath), "w") as f:
        json.dump(report, f, indent=2, default=str)
    return str(filepath)


def send_report(report):
    # type: (Dict) -> None
    """Send report via configured notification channels."""
    text = format_report(report)

    try:
        import notifications
        notifications.notify(
            "EOD Report — %s" % report["date"],
            text,
            level="info",
        )
    except Exception as e:
        logger.debug("Failed to send EOD report via notifications: %s", e)

    # Telegram
    try:
        import telegram_bot
        telegram_bot.send_message(text)
        logger.info("EOD report sent to Telegram")
    except Exception as e:
        logger.debug("Telegram EOD report failed: %s", e)

    print(text)


def list_reports():
    # type: () -> List[str]
    """List available EOD reports."""
    if not REPORTS_DIR.exists():
        return []
    files = sorted(REPORTS_DIR.glob("eod_*.json"), reverse=True)
    return [f.stem.replace("eod_", "") for f in files]


def load_report(report_date):
    # type: (str) -> Optional[Dict]
    """Load a saved EOD report."""
    filepath = REPORTS_DIR / ("eod_%s.json" % report_date)
    if not filepath.exists():
        return None
    try:
        with open(str(filepath)) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="End-of-Day Report Generator")
    parser.add_argument("--date", type=str, help="Report date (YYYY-MM-DD)")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--send", action="store_true", help="Send via notifications")
    parser.add_argument("--save", action="store_true", help="Save report to disk")
    parser.add_argument("--list", action="store_true", help="List saved reports")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")

    if args.list:
        reports = list_reports()
        if reports:
            print("\n  Saved EOD reports:")
            for r in reports[:20]:
                print("    %s" % r)
        else:
            print("\n  No saved reports.")
        print()
        return

    report = generate_report(args.date)

    if args.save:
        path = save_report(report)
        logger.info("Report saved to %s", path)

    if args.json:
        print(json.dumps(report, indent=2, default=str))
        return

    if args.send:
        send_report(report)
        return

    print(format_report(report))


if __name__ == "__main__":
    main()
