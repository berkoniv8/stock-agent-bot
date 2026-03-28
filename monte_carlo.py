#!/usr/bin/env python3
"""
Monte Carlo Simulation — projects future equity curves by randomly sampling
from historical trade P&L distributions.

Provides:
- Probability of hitting equity targets
- Drawdown risk at various confidence levels
- Ruin probability (dropping below a threshold)
- Confidence bands for future equity

Usage:
    python3 monte_carlo.py                          # 1000 simulations, 100 trades
    python3 monte_carlo.py --sims 5000 --trades 200 # Custom
    python3 monte_carlo.py --ruin-level 80000       # Custom ruin threshold
"""

import argparse
import json
import logging
import math
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_SIMULATIONS = 1000
DEFAULT_FUTURE_TRADES = 100


def load_historical_pnls():
    # type: () -> Tuple[List[float], float]
    """Load historical P&L values from paper trading or backtest.

    Returns:
        (pnl_list, starting_capital)
    """
    # Try paper trading first
    try:
        import paper_trader
        state = paper_trader.load_state()
        pnls = [t["pnl"] for t in state.get("closed_trades", [])]
        capital = state.get("starting_capital", 100000)
        if pnls:
            return pnls, capital
    except Exception:
        pass

    # Try backtest CSVs
    try:
        import risk_metrics
        pnls, capital = risk_metrics.load_backtest_pnls()
        if pnls:
            return pnls, capital
    except Exception:
        pass

    return [], 100000


def run_simulation(
    pnls,               # type: List[float]
    starting_capital,   # type: float
    n_simulations=None, # type: Optional[int]
    n_future_trades=None,  # type: Optional[int]
    ruin_level=None,    # type: Optional[float]
    seed=None,          # type: Optional[int]
):
    # type: (...) -> Dict
    """Run Monte Carlo simulation by resampling historical P&L values.

    Args:
        pnls: Historical per-trade P&L values.
        starting_capital: Starting equity.
        n_simulations: Number of simulation paths (default 1000).
        n_future_trades: Number of future trades to simulate (default 100).
        ruin_level: Equity level considered "ruin" (default: 50% of starting).
        seed: Random seed for reproducibility.

    Returns:
        Dict with simulation results, percentiles, and statistics.
    """
    if not pnls:
        return {"error": "No historical P&L data available"}

    n_sims = n_simulations or DEFAULT_SIMULATIONS
    n_trades = n_future_trades or DEFAULT_FUTURE_TRADES
    ruin = ruin_level if ruin_level is not None else starting_capital * 0.5

    if seed is not None:
        np.random.seed(seed)

    pnl_array = np.array(pnls)

    # Run simulations
    # Each row is a simulation, each column is a trade
    random_indices = np.random.randint(0, len(pnl_array), size=(n_sims, n_trades))
    sampled_pnls = pnl_array[random_indices]

    # Build equity curves: cumulative sum + starting capital
    equity_curves = starting_capital + np.cumsum(sampled_pnls, axis=1)

    # Prepend starting capital
    start_col = np.full((n_sims, 1), starting_capital)
    equity_curves = np.hstack([start_col, equity_curves])

    # Final equity for each simulation
    final_equities = equity_curves[:, -1]

    # Max drawdown for each simulation
    max_drawdowns = np.zeros(n_sims)
    max_drawdown_pcts = np.zeros(n_sims)
    for i in range(n_sims):
        curve = equity_curves[i]
        peak = np.maximum.accumulate(curve)
        dd = peak - curve
        dd_pct = dd / peak
        max_drawdowns[i] = np.max(dd)
        max_drawdown_pcts[i] = np.max(dd_pct) * 100

    # Ruin probability
    min_equities = np.min(equity_curves, axis=1)
    ruin_count = np.sum(min_equities <= ruin)
    ruin_probability = ruin_count / n_sims * 100

    # Equity targets
    targets = {}
    target_levels = [1.1, 1.25, 1.5, 2.0]  # 10%, 25%, 50%, 100% gain
    for mult in target_levels:
        target = starting_capital * mult
        hit_count = np.sum(final_equities >= target)
        targets["%.0f%% gain" % ((mult - 1) * 100)] = round(hit_count / n_sims * 100, 1)

    # Loss probability
    loss_count = np.sum(final_equities < starting_capital)
    loss_probability = loss_count / n_sims * 100

    # Percentile bands for equity curve
    percentiles = [5, 10, 25, 50, 75, 90, 95]
    bands = {}
    for p in percentiles:
        band = np.percentile(equity_curves, p, axis=0)
        bands["p%d" % p] = [round(float(v), 2) for v in band]

    # Summary statistics
    mean_final = float(np.mean(final_equities))
    median_final = float(np.median(final_equities))
    std_final = float(np.std(final_equities))
    best_case = float(np.max(final_equities))
    worst_case = float(np.min(final_equities))

    return {
        "n_simulations": n_sims,
        "n_future_trades": n_trades,
        "starting_capital": starting_capital,
        "historical_trades": len(pnls),
        "historical_avg_pnl": round(float(np.mean(pnl_array)), 2),
        "historical_std_pnl": round(float(np.std(pnl_array)), 2),
        "historical_win_rate": round(float(np.sum(pnl_array > 0) / len(pnl_array) * 100), 1),
        # Final equity stats
        "mean_final_equity": round(mean_final, 2),
        "median_final_equity": round(median_final, 2),
        "std_final_equity": round(std_final, 2),
        "best_case": round(best_case, 2),
        "worst_case": round(worst_case, 2),
        "expected_return_pct": round((mean_final - starting_capital) / starting_capital * 100, 2),
        # Risk metrics
        "loss_probability": round(loss_probability, 1),
        "ruin_probability": round(ruin_probability, 1),
        "ruin_level": ruin,
        "avg_max_drawdown": round(float(np.mean(max_drawdowns)), 2),
        "avg_max_drawdown_pct": round(float(np.mean(max_drawdown_pcts)), 2),
        "p95_max_drawdown": round(float(np.percentile(max_drawdowns, 95)), 2),
        "p95_max_drawdown_pct": round(float(np.percentile(max_drawdown_pcts, 95)), 2),
        # Targets
        "target_probabilities": targets,
        # Equity bands (sampled for display — every 10th trade)
        "equity_bands": {k: v[::max(1, n_trades // 20)] for k, v in bands.items()},
        # Full percentile data
        "percentile_final": {
            "p5": round(float(np.percentile(final_equities, 5)), 2),
            "p10": round(float(np.percentile(final_equities, 10)), 2),
            "p25": round(float(np.percentile(final_equities, 25)), 2),
            "p50": round(float(np.percentile(final_equities, 50)), 2),
            "p75": round(float(np.percentile(final_equities, 75)), 2),
            "p90": round(float(np.percentile(final_equities, 90)), 2),
            "p95": round(float(np.percentile(final_equities, 95)), 2),
        },
    }


def compute_kelly_optimal(pnls):
    # type: (List[float]) -> Dict
    """Compute Kelly criterion and optimal position sizing from P&L data.

    Returns dict with kelly fraction, half-kelly, and risk of ruin estimates.
    """
    if not pnls:
        return {"kelly_fraction": 0, "half_kelly": 0}

    arr = np.array(pnls)
    wins = arr[arr > 0]
    losses = arr[arr <= 0]

    if len(wins) == 0 or len(losses) == 0:
        return {"kelly_fraction": 0, "half_kelly": 0}

    win_rate = len(wins) / len(arr)
    avg_win = float(np.mean(wins))
    avg_loss = float(np.mean(np.abs(losses)))

    if avg_loss == 0:
        return {"kelly_fraction": 0, "half_kelly": 0}

    payoff_ratio = avg_win / avg_loss
    kelly = win_rate - ((1 - win_rate) / payoff_ratio)
    kelly = max(0, kelly)

    return {
        "kelly_fraction": round(kelly, 4),
        "half_kelly": round(kelly / 2, 4),
        "kelly_pct": round(kelly * 100, 1),
        "half_kelly_pct": round(kelly / 2 * 100, 1),
        "win_rate": round(win_rate * 100, 1),
        "payoff_ratio": round(payoff_ratio, 2),
    }


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_simulation(result):
    # type: (Dict) -> None
    """Print formatted Monte Carlo report."""
    if "error" in result:
        print("\n  %s\n" % result["error"])
        return

    print("\n" + "=" * 60)
    print("  MONTE CARLO SIMULATION")
    print("=" * 60)

    print("\n  INPUT DATA")
    print("  " + "-" * 50)
    print("  Historical trades:  %d" % result["historical_trades"])
    print("  Avg P&L per trade:  $%.2f" % result["historical_avg_pnl"])
    print("  Std P&L:            $%.2f" % result["historical_std_pnl"])
    print("  Historical win rate: %.1f%%" % result["historical_win_rate"])
    print("  Simulations:        %d" % result["n_simulations"])
    print("  Future trades:      %d" % result["n_future_trades"])

    print("\n  PROJECTED OUTCOMES (after %d trades)" % result["n_future_trades"])
    print("  " + "-" * 50)
    print("  Starting capital:   $%s" % "{:,.2f}".format(result["starting_capital"]))
    print("  Mean final equity:  $%s" % "{:,.2f}".format(result["mean_final_equity"]))
    print("  Median final equity: $%s" % "{:,.2f}".format(result["median_final_equity"]))
    print("  Expected return:    %+.2f%%" % result["expected_return_pct"])
    print("  Best case:          $%s" % "{:,.2f}".format(result["best_case"]))
    print("  Worst case:         $%s" % "{:,.2f}".format(result["worst_case"]))

    print("\n  RISK ANALYSIS")
    print("  " + "-" * 50)
    print("  Loss probability:     %.1f%%" % result["loss_probability"])
    print("  Ruin probability:     %.1f%%  (below $%s)" % (
        result["ruin_probability"], "{:,.0f}".format(result["ruin_level"])))
    print("  Avg max drawdown:     $%s (%.1f%%)" % (
        "{:,.2f}".format(result["avg_max_drawdown"]),
        result["avg_max_drawdown_pct"]))
    print("  95th %% max drawdown: $%s (%.1f%%)" % (
        "{:,.2f}".format(result["p95_max_drawdown"]),
        result["p95_max_drawdown_pct"]))

    print("\n  TARGET PROBABILITIES")
    print("  " + "-" * 50)
    for target, prob in result["target_probabilities"].items():
        bar = "#" * int(prob / 2)
        print("  %-12s  %5.1f%%  %s" % (target, prob, bar))

    # Percentile final equity
    pf = result["percentile_final"]
    print("\n  EQUITY DISTRIBUTION")
    print("  " + "-" * 50)
    for label, value in pf.items():
        pct = label.replace("p", "")
        bar = "#" * max(1, int((value - result["starting_capital"]) / result["starting_capital"] * 40))
        if value < result["starting_capital"]:
            bar = "-" * max(1, int((result["starting_capital"] - value) / result["starting_capital"] * 40))
        print("  %sth %%ile:  $%s  %s" % (pct.rjust(3), "{:>12,.2f}".format(value), bar))

    # Sparkline of median curve
    bands = result.get("equity_bands", {})
    median_band = bands.get("p50", [])
    if len(median_band) > 2:
        mn = min(median_band)
        mx = max(median_band)
        span = mx - mn if mx != mn else 1
        bars = "▁▂▃▄▅▆▇█"
        sparkline = ""
        for v in median_band:
            idx = int((v - mn) / span * (len(bars) - 1))
            sparkline += bars[idx]
        print("\n  MEDIAN EQUITY PATH")
        print("  " + "-" * 50)
        print("  %s" % sparkline)
        print("  $%s — $%s" % ("{:,.0f}".format(mn), "{:,.0f}".format(mx)))

    print("\n" + "=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Monte Carlo Simulation")
    parser.add_argument("--sims", type=int, default=1000, help="Number of simulations")
    parser.add_argument("--trades", type=int, default=100, help="Future trades to simulate")
    parser.add_argument("--ruin-level", type=float, default=None, help="Ruin equity level")
    parser.add_argument("--seed", type=int, default=None, help="Random seed")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--kelly", action="store_true", help="Show Kelly criterion analysis")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

    pnls, capital = load_historical_pnls()

    if not pnls:
        print("\n  No trade data found. Run backtester or paper trader first.\n")
        return

    result = run_simulation(
        pnls, capital,
        n_simulations=args.sims,
        n_future_trades=args.trades,
        ruin_level=args.ruin_level,
        seed=args.seed,
    )

    if args.json:
        print(json.dumps(result, indent=2, default=str))
    else:
        print_simulation(result)

    if args.kelly:
        kelly = compute_kelly_optimal(pnls)
        print("  KELLY CRITERION")
        print("  " + "-" * 40)
        print("  Full Kelly:   %.1f%% of capital per trade" % kelly["kelly_pct"])
        print("  Half Kelly:   %.1f%% (recommended)" % kelly["half_kelly_pct"])
        print("  Win rate:     %.1f%%" % kelly["win_rate"])
        print("  Payoff ratio: %.2f" % kelly["payoff_ratio"])
        print()


if __name__ == "__main__":
    main()
