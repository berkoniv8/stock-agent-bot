#!/usr/bin/env python3
"""
Performance Attribution — breaks down P&L contribution by sector, signal,
time period, and trade direction to identify what's driving returns.

Answers:
- Which sectors contributed most to P&L?
- Which signals had the best/worst outcomes?
- Are long or short trades performing better?
- How does performance vary by day-of-week or holding period?
- What's the win rate trend over time?

Usage:
    python3 performance_attribution.py              # Full attribution report
    python3 performance_attribution.py --by sector  # Sector breakdown only
    python3 performance_attribution.py --by signal  # Signal breakdown only
    python3 performance_attribution.py --json       # Output as JSON
"""

import argparse
import json
import logging
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def load_closed_trades():
    # type: () -> List[Dict]
    """Load all closed trades from paper trading and backtest data."""
    trades = []

    # Paper trading
    try:
        import paper_trader
        state = paper_trader.load_state()
        for t in state.get("closed_trades", []):
            t["_source"] = "paper"
            trades.append(t)
    except Exception:
        pass

    # Backtest CSVs
    try:
        import signal_analytics
        bt_trades = signal_analytics.load_backtest_trades()
        for t in bt_trades:
            t["_source"] = "backtest"
            # Normalize field names
            t.setdefault("pnl", float(t.get("pnl", 0)))
            t.setdefault("pnl_pct", float(t.get("pnl_pct", 0)))
            t.setdefault("direction", t.get("direction", "BUY"))
            t.setdefault("triggered_signals",
                         t.get("signals", "").split("|") if t.get("signals") else [])
            trades.append(t)
    except Exception:
        pass

    return trades


def attribute_by_sector(trades, watchlist_map=None):
    # type: (List[Dict], Optional[Dict[str, str]]) -> Dict[str, Dict]
    """Break down P&L by sector.

    Args:
        trades: List of closed trade dicts.
        watchlist_map: Optional dict mapping ticker -> sector.

    Returns dict mapping sector -> performance metrics.
    """
    if watchlist_map is None:
        watchlist_map = _load_sector_map()

    sectors = defaultdict(lambda: {
        "trades": 0, "wins": 0, "losses": 0,
        "total_pnl": 0.0, "tickers": set(),
    })

    for t in trades:
        ticker = t.get("ticker", "")
        sector = watchlist_map.get(ticker, "Unknown")
        pnl = float(t.get("pnl", 0))

        s = sectors[sector]
        s["trades"] += 1
        s["total_pnl"] += pnl
        s["tickers"].add(ticker)
        if pnl > 0:
            s["wins"] += 1
        else:
            s["losses"] += 1

    # Derive metrics
    result = {}
    for sector, s in sectors.items():
        n = s["trades"]
        result[sector] = {
            "trades": n,
            "wins": s["wins"],
            "losses": s["losses"],
            "win_rate": round(s["wins"] / n * 100, 1) if n > 0 else 0,
            "total_pnl": round(s["total_pnl"], 2),
            "avg_pnl": round(s["total_pnl"] / n, 2) if n > 0 else 0,
            "pnl_contribution_pct": 0,  # Filled below
            "tickers": sorted(s["tickers"]),
        }

    # Contribution %
    total_pnl = sum(v["total_pnl"] for v in result.values())
    if total_pnl != 0:
        for sector in result:
            result[sector]["pnl_contribution_pct"] = round(
                result[sector]["total_pnl"] / abs(total_pnl) * 100, 1
            )

    return dict(sorted(result.items(), key=lambda x: -x[1]["total_pnl"]))


def attribute_by_signal(trades):
    # type: (List[Dict]) -> Dict[str, Dict]
    """Break down P&L by individual signal that triggered the trade."""
    signals = defaultdict(lambda: {
        "trades": 0, "wins": 0, "losses": 0,
        "total_pnl": 0.0,
    })

    for t in trades:
        sigs = t.get("triggered_signals", [])
        if isinstance(sigs, str):
            sigs = sigs.split("|")
        pnl = float(t.get("pnl", 0))

        for sig in sigs:
            sig = sig.strip()
            if not sig:
                continue
            s = signals[sig]
            s["trades"] += 1
            s["total_pnl"] += pnl
            if pnl > 0:
                s["wins"] += 1
            else:
                s["losses"] += 1

    result = {}
    for sig, s in signals.items():
        n = s["trades"]
        result[sig] = {
            "trades": n,
            "wins": s["wins"],
            "losses": s["losses"],
            "win_rate": round(s["wins"] / n * 100, 1) if n > 0 else 0,
            "total_pnl": round(s["total_pnl"], 2),
            "avg_pnl": round(s["total_pnl"] / n, 2) if n > 0 else 0,
        }

    return dict(sorted(result.items(), key=lambda x: -x[1]["total_pnl"]))


def attribute_by_direction(trades):
    # type: (List[Dict]) -> Dict[str, Dict]
    """Break down P&L by trade direction (BUY vs SELL)."""
    dirs = defaultdict(lambda: {
        "trades": 0, "wins": 0, "losses": 0, "total_pnl": 0.0,
    })

    for t in trades:
        direction = t.get("direction", "BUY")
        pnl = float(t.get("pnl", 0))
        d = dirs[direction]
        d["trades"] += 1
        d["total_pnl"] += pnl
        if pnl > 0:
            d["wins"] += 1
        else:
            d["losses"] += 1

    result = {}
    for direction, d in dirs.items():
        n = d["trades"]
        result[direction] = {
            "trades": n,
            "wins": d["wins"],
            "losses": d["losses"],
            "win_rate": round(d["wins"] / n * 100, 1) if n > 0 else 0,
            "total_pnl": round(d["total_pnl"], 2),
            "avg_pnl": round(d["total_pnl"] / n, 2) if n > 0 else 0,
        }

    return result


def attribute_by_holding_period(trades):
    # type: (List[Dict]) -> Dict[str, Dict]
    """Break down P&L by holding period buckets."""
    buckets = {
        "1-3 days": (1, 3),
        "4-7 days": (4, 7),
        "8-14 days": (8, 14),
        "15-30 days": (15, 30),
        "30+ days": (31, 9999),
    }

    result = {name: {"trades": 0, "wins": 0, "total_pnl": 0.0}
              for name in buckets}

    for t in trades:
        bars = int(t.get("bars_held", 0))
        pnl = float(t.get("pnl", 0))

        for name, (lo, hi) in buckets.items():
            if lo <= bars <= hi:
                result[name]["trades"] += 1
                result[name]["total_pnl"] += pnl
                if pnl > 0:
                    result[name]["wins"] += 1
                break

    for name, d in result.items():
        n = d["trades"]
        d["win_rate"] = round(d["wins"] / n * 100, 1) if n > 0 else 0
        d["avg_pnl"] = round(d["total_pnl"] / n, 2) if n > 0 else 0
        d["total_pnl"] = round(d["total_pnl"], 2)

    return result


def attribute_by_exit_reason(trades):
    # type: (List[Dict]) -> Dict[str, Dict]
    """Break down P&L by exit reason."""
    reasons = defaultdict(lambda: {
        "trades": 0, "wins": 0, "total_pnl": 0.0,
    })

    for t in trades:
        reason = t.get("exit_reason", "unknown")
        pnl = float(t.get("pnl", 0))
        r = reasons[reason]
        r["trades"] += 1
        r["total_pnl"] += pnl
        if pnl > 0:
            r["wins"] += 1

    result = {}
    for reason, r in reasons.items():
        n = r["trades"]
        result[reason] = {
            "trades": n,
            "wins": r["wins"],
            "win_rate": round(r["wins"] / n * 100, 1) if n > 0 else 0,
            "total_pnl": round(r["total_pnl"], 2),
            "avg_pnl": round(r["total_pnl"] / n, 2) if n > 0 else 0,
        }

    return dict(sorted(result.items(), key=lambda x: -x[1]["total_pnl"]))


def compute_rolling_win_rate(trades, window=10):
    # type: (List[Dict], int) -> List[Dict]
    """Compute rolling win rate over a sliding window of trades.

    Returns list of {trade_index, rolling_win_rate, rolling_pnl} dicts.
    """
    if len(trades) < window:
        return []

    # Sort by exit date
    sorted_trades = sorted(trades, key=lambda t: t.get("exit_date", ""))

    result = []
    for i in range(window - 1, len(sorted_trades)):
        chunk = sorted_trades[i - window + 1:i + 1]
        wins = sum(1 for t in chunk if float(t.get("pnl", 0)) > 0)
        total_pnl = sum(float(t.get("pnl", 0)) for t in chunk)
        result.append({
            "trade_index": i + 1,
            "date": sorted_trades[i].get("exit_date", "")[:10],
            "rolling_win_rate": round(wins / window * 100, 1),
            "rolling_pnl": round(total_pnl, 2),
        })

    return result


def full_attribution():
    # type: () -> Dict
    """Run complete performance attribution.

    Returns dict with all attribution breakdowns.
    """
    trades = load_closed_trades()
    if not trades:
        return {"error": "No closed trades found", "total_trades": 0}

    total_pnl = sum(float(t.get("pnl", 0)) for t in trades)
    wins = sum(1 for t in trades if float(t.get("pnl", 0)) > 0)

    return {
        "total_trades": len(trades),
        "total_pnl": round(total_pnl, 2),
        "win_rate": round(wins / len(trades) * 100, 1) if trades else 0,
        "by_sector": attribute_by_sector(trades),
        "by_signal": attribute_by_signal(trades),
        "by_direction": attribute_by_direction(trades),
        "by_holding_period": attribute_by_holding_period(trades),
        "by_exit_reason": attribute_by_exit_reason(trades),
        "rolling_win_rate": compute_rolling_win_rate(trades),
    }


def _load_sector_map():
    # type: () -> Dict[str, str]
    """Build ticker -> sector mapping from watchlist."""
    try:
        import data_layer
        wl = data_layer.load_watchlist()
        return {e["ticker"]: e.get("sector", "Unknown") for e in wl}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_attribution(result):
    # type: (Dict) -> None
    """Print formatted attribution report."""
    if "error" in result:
        print("\n  %s\n" % result["error"])
        return

    print("\n" + "=" * 65)
    print("  PERFORMANCE ATTRIBUTION")
    print("  %d trades | Total P&L: $%s | Win rate: %.1f%%" % (
        result["total_trades"],
        "{:,.2f}".format(result["total_pnl"]),
        result["win_rate"]))
    print("=" * 65)

    # By sector
    by_sector = result["by_sector"]
    if by_sector:
        print("\n  BY SECTOR")
        print("  " + "-" * 60)
        print("  %-20s %6s %6s %10s %8s %6s" % (
            "Sector", "Trades", "Win%", "Total P&L", "Avg P&L", "Contr%"))
        for sector, s in by_sector.items():
            print("  %-20s %6d %5.1f%% $%9s $%7s %+5.1f%%" % (
                sector[:20], s["trades"], s["win_rate"],
                "{:,.2f}".format(s["total_pnl"]),
                "{:,.2f}".format(s["avg_pnl"]),
                s["pnl_contribution_pct"]))

    # By signal
    by_signal = result["by_signal"]
    if by_signal:
        print("\n  BY SIGNAL (top 10)")
        print("  " + "-" * 60)
        print("  %-28s %6s %6s %10s %8s" % (
            "Signal", "Trades", "Win%", "Total P&L", "Avg P&L"))
        for sig, s in list(by_signal.items())[:10]:
            print("  %-28s %6d %5.1f%% $%9s $%7s" % (
                sig[:28], s["trades"], s["win_rate"],
                "{:,.2f}".format(s["total_pnl"]),
                "{:,.2f}".format(s["avg_pnl"])))

    # By direction
    by_dir = result["by_direction"]
    if by_dir:
        print("\n  BY DIRECTION")
        print("  " + "-" * 60)
        for direction, d in by_dir.items():
            print("  %-6s  %d trades  WR: %.1f%%  P&L: $%s  Avg: $%s" % (
                direction, d["trades"], d["win_rate"],
                "{:,.2f}".format(d["total_pnl"]),
                "{:,.2f}".format(d["avg_pnl"])))

    # By holding period
    by_hold = result["by_holding_period"]
    if by_hold:
        print("\n  BY HOLDING PERIOD")
        print("  " + "-" * 60)
        for period, h in by_hold.items():
            if h["trades"] > 0:
                print("  %-12s  %d trades  WR: %.1f%%  Avg P&L: $%s" % (
                    period, h["trades"], h["win_rate"],
                    "{:,.2f}".format(h["avg_pnl"])))

    # By exit reason
    by_exit = result["by_exit_reason"]
    if by_exit:
        print("\n  BY EXIT REASON")
        print("  " + "-" * 60)
        for reason, e in by_exit.items():
            print("  %-16s  %d trades  WR: %.1f%%  Avg P&L: $%s" % (
                reason, e["trades"], e["win_rate"],
                "{:,.2f}".format(e["avg_pnl"])))

    # Rolling win rate trend
    rolling = result.get("rolling_win_rate", [])
    if rolling:
        recent = rolling[-5:]
        print("\n  WIN RATE TREND (rolling 10-trade)")
        print("  " + "-" * 60)
        for r in recent:
            bar = "#" * int(r["rolling_win_rate"] / 2)
            print("  Trade #%d (%s): %.1f%% %s" % (
                r["trade_index"], r["date"], r["rolling_win_rate"], bar))

    print("\n" + "=" * 65 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Performance Attribution")
    parser.add_argument("--by", type=str, choices=["sector", "signal", "direction", "holding", "exit"],
                        help="Show only one breakdown")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

    result = full_attribution()

    if args.json:
        print(json.dumps(result, indent=2, default=str))
        return

    if args.by:
        key = "by_%s" % args.by
        if args.by == "holding":
            key = "by_holding_period"
        elif args.by == "exit":
            key = "by_exit_reason"

        if key in result:
            subset = result[key]
            print("\n  ATTRIBUTION BY %s" % args.by.upper())
            print("  " + "-" * 50)
            print(json.dumps(subset, indent=2, default=str))
            print()
        return

    print_attribution(result)


if __name__ == "__main__":
    main()
