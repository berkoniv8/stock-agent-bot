#!/usr/bin/env python3
"""
Signal Analytics — analyzes historical backtest and alert data to determine
which signals have the best win rates, profit factors, and reliability.

Usage:
    python3 signal_analytics.py                     # Analyze all backtest CSVs
    python3 signal_analytics.py --file logs/backtest_20260325_231756.csv
"""

import argparse
import csv
import glob
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)

LOGS_DIR = Path("logs")


def load_backtest_trades(filepath: str = None) -> List[Dict]:
    """Load trades from backtest CSV files."""
    trades = []

    if filepath:
        files = [filepath]
    else:
        files = sorted(glob.glob(str(LOGS_DIR / "backtest_*.csv")))

    for f in files:
        try:
            with open(f, newline="") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    row["_source"] = f
                    trades.append(row)
        except Exception as e:
            logger.error("Error reading %s: %s", f, e)

    return trades


def analyze_by_signal(trades: List[Dict]) -> Dict[str, dict]:
    """Break down performance by individual signal type.

    Returns dict mapping signal_name -> stats.
    """
    signal_stats = defaultdict(lambda: {
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "total_pnl": 0.0,
        "total_win_pnl": 0.0,
        "total_loss_pnl": 0.0,
        "pnl_values": [],
    })

    for trade in trades:
        signals = trade.get("signals", "").split("|")
        pnl = float(trade.get("pnl", 0))

        for sig in signals:
            sig = sig.strip()
            if not sig:
                continue

            stats = signal_stats[sig]
            stats["total_trades"] += 1
            stats["pnl_values"].append(pnl)
            stats["total_pnl"] += pnl

            if pnl > 0:
                stats["wins"] += 1
                stats["total_win_pnl"] += pnl
            else:
                stats["losses"] += 1
                stats["total_loss_pnl"] += abs(pnl)

    # Compute derived metrics
    for sig, stats in signal_stats.items():
        n = stats["total_trades"]
        stats["win_rate"] = (stats["wins"] / n * 100) if n > 0 else 0
        stats["avg_pnl"] = stats["total_pnl"] / n if n > 0 else 0
        stats["profit_factor"] = (
            stats["total_win_pnl"] / stats["total_loss_pnl"]
            if stats["total_loss_pnl"] > 0
            else float("inf") if stats["total_win_pnl"] > 0 else 0
        )
        stats["avg_win"] = stats["total_win_pnl"] / stats["wins"] if stats["wins"] > 0 else 0
        stats["avg_loss"] = stats["total_loss_pnl"] / stats["losses"] if stats["losses"] > 0 else 0
        # Expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
        win_rate_dec = stats["wins"] / n if n > 0 else 0
        loss_rate_dec = stats["losses"] / n if n > 0 else 0
        stats["expectancy"] = (win_rate_dec * stats["avg_win"]) - (loss_rate_dec * stats["avg_loss"])

    return dict(signal_stats)


def analyze_by_combination(trades: List[Dict]) -> Dict[str, dict]:
    """Analyze performance by signal combinations (the full combo that triggered)."""
    combo_stats = defaultdict(lambda: {
        "total_trades": 0,
        "wins": 0,
        "losses": 0,
        "total_pnl": 0.0,
    })

    for trade in trades:
        combo_key = trade.get("signals", "").replace("|", " + ")
        if not combo_key:
            continue

        pnl = float(trade.get("pnl", 0))
        stats = combo_stats[combo_key]
        stats["total_trades"] += 1
        stats["total_pnl"] += pnl
        if pnl > 0:
            stats["wins"] += 1
        else:
            stats["losses"] += 1

    # Compute win rate
    for combo, stats in combo_stats.items():
        n = stats["total_trades"]
        stats["win_rate"] = (stats["wins"] / n * 100) if n > 0 else 0
        stats["avg_pnl"] = stats["total_pnl"] / n if n > 0 else 0

    return dict(combo_stats)


def analyze_by_exit(trades: List[Dict]) -> Dict[str, dict]:
    """Analyze performance by exit reason."""
    exit_stats = defaultdict(lambda: {
        "total_trades": 0,
        "total_pnl": 0.0,
        "avg_bars": 0.0,
        "bars_list": [],
    })

    for trade in trades:
        reason = trade.get("exit_reason", "unknown")
        pnl = float(trade.get("pnl", 0))
        bars = int(trade.get("bars_held", 0))

        stats = exit_stats[reason]
        stats["total_trades"] += 1
        stats["total_pnl"] += pnl
        stats["bars_list"].append(bars)

    for reason, stats in exit_stats.items():
        n = stats["total_trades"]
        stats["avg_pnl"] = stats["total_pnl"] / n if n > 0 else 0
        stats["avg_bars"] = sum(stats["bars_list"]) / n if n > 0 else 0

    return dict(exit_stats)


def print_analytics(trades: List[Dict]) -> None:
    """Print comprehensive signal analytics report."""
    if not trades:
        print("No trade data found. Run the backtester first.")
        return

    print(f"\n{'=' * 70}")
    print(f"  SIGNAL ANALYTICS — {len(trades)} trades analyzed")
    print(f"{'=' * 70}")

    # Overall stats
    pnls = [float(t.get("pnl", 0)) for t in trades]
    wins = sum(1 for p in pnls if p > 0)
    total_pnl = sum(pnls)
    print(f"\n  OVERALL: {wins}/{len(trades)} wins ({wins / len(trades) * 100:.0f}%), total P&L: ${total_pnl:,.2f}")

    # By individual signal
    sig_stats = analyze_by_signal(trades)
    print(f"\n  PERFORMANCE BY SIGNAL")
    print(f"  {'─' * 62}")
    print(f"  {'Signal':<30} {'Trades':>6} {'Win%':>6} {'Avg P&L':>9} {'PF':>5} {'Expect':>8}")
    print(f"  {'─' * 62}")

    sorted_sigs = sorted(sig_stats.items(), key=lambda x: x[1]["expectancy"], reverse=True)
    for sig, stats in sorted_sigs:
        pf_str = f"{stats['profit_factor']:.2f}" if stats['profit_factor'] != float("inf") else "inf"
        print(
            f"  {sig:<30} {stats['total_trades']:>6} {stats['win_rate']:>5.0f}% "
            f"${stats['avg_pnl']:>8,.2f} {pf_str:>5} ${stats['expectancy']:>7,.2f}"
        )

    # Best and worst signals
    if sorted_sigs:
        best = sorted_sigs[0]
        worst = sorted_sigs[-1]
        print(f"\n  Best signal:  {best[0]} (expectancy ${best[1]['expectancy']:,.2f}/trade)")
        print(f"  Worst signal: {worst[0]} (expectancy ${worst[1]['expectancy']:,.2f}/trade)")

    # By signal combination
    combo_stats = analyze_by_combination(trades)
    if combo_stats:
        print(f"\n  TOP SIGNAL COMBINATIONS")
        print(f"  {'─' * 62}")
        sorted_combos = sorted(combo_stats.items(), key=lambda x: x[1]["win_rate"], reverse=True)
        for combo, stats in sorted_combos[:10]:
            if stats["total_trades"] >= 2:  # Only show combos with 2+ trades
                print(f"  {combo[:50]:<50} {stats['total_trades']:>3}x {stats['win_rate']:>5.0f}% ${stats['avg_pnl']:>7,.2f}")

    # By exit reason
    exit_stats = analyze_by_exit(trades)
    if exit_stats:
        print(f"\n  EXIT ANALYSIS")
        print(f"  {'─' * 62}")
        for reason, stats in sorted(exit_stats.items()):
            print(f"  {reason:<20} {stats['total_trades']:>5} trades  avg P&L: ${stats['avg_pnl']:>7,.2f}  avg bars: {stats['avg_bars']:>.0f}")

    # Recommendations
    print(f"\n  RECOMMENDATIONS")
    print(f"  {'─' * 62}")
    for sig, stats in sorted_sigs:
        if stats["total_trades"] >= 3:
            if stats["win_rate"] >= 60 and stats["expectancy"] > 0:
                print(f"  KEEP   {sig} — {stats['win_rate']:.0f}% win rate, positive expectancy")
            elif stats["win_rate"] < 40 and stats["expectancy"] < 0:
                print(f"  REVIEW {sig} — {stats['win_rate']:.0f}% win rate, negative expectancy")

    print(f"\n{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(description="Signal Analytics")
    parser.add_argument("--file", type=str, help="Specific backtest CSV to analyze")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

    trades = load_backtest_trades(args.file)
    print_analytics(trades)


if __name__ == "__main__":
    main()
