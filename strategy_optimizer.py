#!/usr/bin/env python3
"""
Strategy Optimizer — analyzes backtest and paper trading results to suggest
signal weight adjustments for better performance.

Approach:
1. Load historical trade data (backtest CSVs + paper trading)
2. Compute per-signal win rate, expectancy, and profit factor
3. Suggest weight increases for high-performing signals
4. Suggest weight decreases for underperforming signals
5. Optionally write optimized weights to a config file

Usage:
    python3 strategy_optimizer.py                  # Show recommendations
    python3 strategy_optimizer.py --apply          # Apply suggested weights
    python3 strategy_optimizer.py --simulate       # Simulate with new weights
"""

import argparse
import csv
import glob
import json
import logging
import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

LOGS_DIR = Path("logs")
OPTIMIZED_WEIGHTS_FILE = Path("logs/optimized_weights.json")

# Minimum trades for a signal to be considered for optimization
MIN_TRADES_FOR_OPTIMIZATION = 5


def load_all_trades():
    # type: () -> List[Dict]
    """Load trades from all available sources: backtest CSVs + paper trading.

    Returns list of trade dicts with at least: signals, pnl, pnl_pct, exit_reason.
    """
    trades = []

    # 1. Backtest CSVs
    for filepath in sorted(glob.glob(str(LOGS_DIR / "backtest_*.csv"))):
        try:
            with open(filepath, newline="") as f:
                for row in csv.DictReader(f):
                    row["_source"] = "backtest"
                    trades.append(row)
        except Exception as e:
            logger.debug("Error reading %s: %s", filepath, e)

    # 2. Paper trading closed trades
    try:
        import paper_trader
        state = paper_trader.load_state()
        for t in state.get("closed_trades", []):
            trades.append({
                "signals": "|".join(t.get("triggered_signals", [])),
                "pnl": t.get("pnl", 0),
                "pnl_pct": t.get("pnl_pct", 0),
                "exit_reason": t.get("exit_reason", ""),
                "signal_score": t.get("signal_score", 0),
                "_source": "paper",
            })
    except Exception:
        pass

    return trades


def compute_signal_performance(trades):
    # type: (List[Dict]) -> Dict[str, Dict]
    """Compute detailed performance metrics per signal.

    Returns dict mapping signal_name -> metrics dict.
    """
    stats = defaultdict(lambda: {
        "trades": 0,
        "wins": 0,
        "losses": 0,
        "total_pnl": 0.0,
        "win_pnl": 0.0,
        "loss_pnl": 0.0,
        "pnl_values": [],
    })

    for trade in trades:
        signals = trade.get("signals", "").split("|")
        pnl = float(trade.get("pnl", 0))

        for sig in signals:
            sig = sig.strip()
            if not sig:
                continue

            s = stats[sig]
            s["trades"] += 1
            s["total_pnl"] += pnl
            s["pnl_values"].append(pnl)

            if pnl > 0:
                s["wins"] += 1
                s["win_pnl"] += pnl
            else:
                s["losses"] += 1
                s["loss_pnl"] += abs(pnl)

    # Derive metrics
    for sig, s in stats.items():
        n = s["trades"]
        s["win_rate"] = (s["wins"] / n * 100) if n > 0 else 0
        s["avg_pnl"] = s["total_pnl"] / n if n > 0 else 0
        s["profit_factor"] = (
            s["win_pnl"] / s["loss_pnl"]
            if s["loss_pnl"] > 0
            else float("inf") if s["win_pnl"] > 0 else 0
        )
        avg_win = s["win_pnl"] / s["wins"] if s["wins"] > 0 else 0
        avg_loss = s["loss_pnl"] / s["losses"] if s["losses"] > 0 else 0
        wr = s["wins"] / n if n > 0 else 0
        lr = s["losses"] / n if n > 0 else 0
        s["expectancy"] = (wr * avg_win) - (lr * avg_loss)
        s["avg_win"] = avg_win
        s["avg_loss"] = avg_loss

    return dict(stats)


def get_current_weights():
    # type: () -> Dict[str, int]
    """Get current signal weights from signal_engine."""
    try:
        import signal_engine
        return dict(signal_engine.SIGNAL_WEIGHTS)
    except Exception:
        return {}


def suggest_weight_adjustments(signal_perf, current_weights):
    # type: (Dict[str, Dict], Dict[str, int]) -> List[Dict]
    """Generate weight adjustment suggestions based on performance data.

    Rules:
    - Signals with win_rate > 60% and positive expectancy: suggest +1
    - Signals with win_rate > 70% and PF > 2.0: suggest +2
    - Signals with win_rate < 40% and negative expectancy: suggest -1
    - Signals with win_rate < 30%: suggest remove (weight = 0)
    - Never suggest weight below 0 or above 5

    Returns list of suggestion dicts.
    """
    suggestions = []

    for sig, perf in signal_perf.items():
        if perf["trades"] < MIN_TRADES_FOR_OPTIMIZATION:
            continue

        current_w = current_weights.get(sig, 0)
        new_w = current_w
        reason = ""

        wr = perf["win_rate"]
        pf = perf["profit_factor"]
        exp = perf["expectancy"]

        if wr >= 70 and pf > 2.0 and exp > 0:
            new_w = min(5, current_w + 2)
            reason = "Excellent: %.0f%% WR, %.1fx PF" % (wr, pf)
        elif wr >= 60 and exp > 0:
            new_w = min(5, current_w + 1)
            reason = "Strong: %.0f%% WR, positive expectancy" % wr
        elif wr < 30:
            new_w = 0
            reason = "Poor: %.0f%% WR — consider removing" % wr
        elif wr < 40 and exp < 0:
            new_w = max(0, current_w - 1)
            reason = "Weak: %.0f%% WR, negative expectancy" % wr

        if new_w != current_w:
            suggestions.append({
                "signal": sig,
                "current_weight": current_w,
                "suggested_weight": new_w,
                "change": new_w - current_w,
                "reason": reason,
                "win_rate": round(wr, 1),
                "profit_factor": round(pf, 2) if pf != float("inf") else "inf",
                "expectancy": round(exp, 2),
                "trades": perf["trades"],
            })

    # Sort by impact (biggest positive changes first, then negative)
    suggestions.sort(key=lambda x: -x["change"])
    return suggestions


def compute_optimized_weights(current_weights, suggestions):
    # type: (Dict[str, int], List[Dict]) -> Dict[str, int]
    """Apply suggestions to current weights and return optimized set."""
    optimized = dict(current_weights)
    for s in suggestions:
        optimized[s["signal"]] = s["suggested_weight"]
    return optimized


def save_optimized_weights(weights):
    # type: (Dict[str, int]) -> None
    """Save optimized weights to file."""
    os.makedirs(os.path.dirname(OPTIMIZED_WEIGHTS_FILE) or ".", exist_ok=True)
    data = {
        "weights": weights,
        "timestamp": __import__("datetime").datetime.now().isoformat(),
    }
    with open(OPTIMIZED_WEIGHTS_FILE, "w") as f:
        json.dump(data, f, indent=2)
    logger.info("Optimized weights saved to %s", OPTIMIZED_WEIGHTS_FILE)


def load_optimized_weights():
    # type: () -> Optional[Dict[str, int]]
    """Load previously saved optimized weights."""
    if not OPTIMIZED_WEIGHTS_FILE.exists():
        return None
    try:
        with open(OPTIMIZED_WEIGHTS_FILE) as f:
            data = json.load(f)
        return data.get("weights")
    except (json.JSONDecodeError, IOError):
        return None


def simulate_with_weights(trades, weights):
    # type: (List[Dict], Dict[str, int]) -> Dict
    """Simulate trade selection with different weights.

    Re-scores each trade using the new weights and computes
    hypothetical performance at various thresholds.
    """
    results = {}

    for threshold in [4, 5, 6, 7]:
        selected = []
        for trade in trades:
            signals = trade.get("signals", "").split("|")
            score = sum(weights.get(s.strip(), 0) for s in signals if s.strip())
            if score >= threshold:
                selected.append(trade)

        n = len(selected)
        if n == 0:
            results[threshold] = {"trades": 0, "win_rate": 0, "total_pnl": 0, "avg_pnl": 0}
            continue

        pnls = [float(t.get("pnl", 0)) for t in selected]
        wins = sum(1 for p in pnls if p > 0)
        total_pnl = sum(pnls)

        results[threshold] = {
            "trades": n,
            "win_rate": round(wins / n * 100, 1),
            "total_pnl": round(total_pnl, 2),
            "avg_pnl": round(total_pnl / n, 2),
        }

    return results


def full_optimization():
    # type: () -> Dict
    """Run full optimization pipeline.

    Returns dict with performance data, suggestions, and simulation results.
    """
    trades = load_all_trades()
    if not trades:
        return {"error": "No trade data found. Run backtester first."}

    signal_perf = compute_signal_performance(trades)
    current_weights = get_current_weights()
    suggestions = suggest_weight_adjustments(signal_perf, current_weights)
    optimized = compute_optimized_weights(current_weights, suggestions)

    # Simulate both
    sim_current = simulate_with_weights(trades, current_weights)
    sim_optimized = simulate_with_weights(trades, optimized)

    return {
        "total_trades": len(trades),
        "signal_performance": signal_perf,
        "current_weights": current_weights,
        "suggestions": suggestions,
        "optimized_weights": optimized,
        "simulation": {
            "current": sim_current,
            "optimized": sim_optimized,
        },
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_optimization(result):
    # type: (Dict) -> None
    """Print formatted optimization report."""
    if "error" in result:
        print("\n  %s\n" % result["error"])
        return

    print("\n" + "=" * 70)
    print("  STRATEGY OPTIMIZATION REPORT")
    print("  Based on %d historical trades" % result["total_trades"])
    print("=" * 70)

    # Signal performance table
    perf = result["signal_performance"]
    print("\n  SIGNAL PERFORMANCE")
    print("  " + "-" * 65)
    print("  %-28s %6s %6s %8s %6s %8s" % (
        "Signal", "Trades", "Win%", "Avg P&L", "PF", "Expect"))
    print("  " + "-" * 65)

    for sig, stats in sorted(perf.items(), key=lambda x: -x[1]["expectancy"]):
        pf = "%.2f" % stats["profit_factor"] if stats["profit_factor"] != float("inf") else "inf"
        print("  %-28s %6d %5.0f%% $%7.2f %6s $%7.2f" % (
            sig[:28], stats["trades"], stats["win_rate"],
            stats["avg_pnl"], pf, stats["expectancy"]))

    # Suggestions
    suggestions = result["suggestions"]
    if suggestions:
        print("\n  WEIGHT ADJUSTMENT SUGGESTIONS")
        print("  " + "-" * 65)
        for s in suggestions:
            arrow = "+" if s["change"] > 0 else ""
            print("  %-28s  %d -> %d  (%s%d)  %s" % (
                s["signal"][:28], s["current_weight"], s["suggested_weight"],
                arrow, s["change"], s["reason"]))
    else:
        print("\n  No weight adjustments suggested (need more trade data).")

    # Simulation comparison
    sim = result["simulation"]
    print("\n  SIMULATION COMPARISON")
    print("  " + "-" * 65)
    print("  %-12s %-8s %8s %8s %8s" % ("Weights", "Thresh", "Trades", "Win%", "Avg P&L"))
    print("  " + "-" * 65)
    for threshold in sorted(sim["current"].keys()):
        cur = sim["current"][threshold]
        opt = sim["optimized"][threshold]
        print("  Current      T=%d     %6d  %6.1f%%  $%7.2f" % (
            threshold, cur["trades"], cur["win_rate"], cur["avg_pnl"]))
        print("  Optimized    T=%d     %6d  %6.1f%%  $%7.2f" % (
            threshold, opt["trades"], opt["win_rate"], opt["avg_pnl"]))

    print("\n" + "=" * 70 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Strategy Optimizer")
    parser.add_argument("--apply", action="store_true",
                        help="Save optimized weights to file")
    parser.add_argument("--simulate", action="store_true",
                        help="Show simulation with current vs optimized weights")
    parser.add_argument("--json", action="store_true",
                        help="Output as JSON")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

    result = full_optimization()

    if args.json:
        # Clean up non-serializable values
        import copy
        out = copy.deepcopy(result)
        for sig, perf in out.get("signal_performance", {}).items():
            if "pnl_values" in perf:
                del perf["pnl_values"]
            if perf.get("profit_factor") == float("inf"):
                perf["profit_factor"] = "inf"
        print(json.dumps(out, indent=2, default=str))
        return

    print_optimization(result)

    if args.apply and "optimized_weights" in result:
        save_optimized_weights(result["optimized_weights"])
        print("  Optimized weights saved to %s" % OPTIMIZED_WEIGHTS_FILE)
        print("  To use them, set OPTIMIZED_WEIGHTS=1 in .env\n")


if __name__ == "__main__":
    main()
