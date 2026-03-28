#!/usr/bin/env python3
"""
Risk-Adjusted Performance Metrics — computes Sharpe ratio, Sortino ratio,
Calmar ratio, max drawdown, and other risk analytics from trade history.

Works with both paper trading and backtest data.

Usage:
    python3 risk_metrics.py --paper           # Analyze paper trading history
    python3 risk_metrics.py --backtest        # Analyze latest backtest
    python3 risk_metrics.py --equity-curve     # Show equity curve with drawdowns
"""

import argparse
import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

# Risk-free rate assumption (annualized)
RISK_FREE_RATE = 0.045  # 4.5% (T-bill proxy)


def compute_metrics(
    pnls: List[float],
    starting_capital: float = 100000,
    periods_per_year: float = 252,
) -> Dict:
    """Compute comprehensive risk-adjusted metrics from a sequence of P&L values.

    Args:
        pnls: List of per-trade P&L values (dollars).
        starting_capital: Initial capital for return calculations.
        periods_per_year: Annualization factor (252 for daily, ~50 for weekly trades).

    Returns dict with all metrics.
    """
    if not pnls:
        return {"error": "No trade data"}

    n = len(pnls)

    # Convert P&L to returns
    equity = [starting_capital]
    for p in pnls:
        equity.append(equity[-1] + p)
    equity = np.array(equity)

    returns = np.diff(equity) / equity[:-1]
    returns = returns[np.isfinite(returns)]

    if len(returns) == 0:
        return {"error": "No valid returns"}

    # Basic stats
    total_pnl = sum(pnls)
    total_return = (equity[-1] - starting_capital) / starting_capital
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    win_rate = len(wins) / n if n > 0 else 0

    # Mean and std of returns
    mean_return = np.mean(returns)
    std_return = np.std(returns, ddof=1) if len(returns) > 1 else 0

    # Annualized returns and volatility
    # Estimate trades per year based on trade frequency
    ann_factor = min(periods_per_year, n)  # Don't over-annualize sparse data
    ann_return = mean_return * ann_factor
    ann_volatility = std_return * math.sqrt(ann_factor)

    # --- Sharpe Ratio ---
    # (annualized return - risk-free rate) / annualized volatility
    rf_per_period = RISK_FREE_RATE / periods_per_year
    excess_returns = returns - rf_per_period
    sharpe = 0
    if std_return > 0:
        sharpe = (np.mean(excess_returns) * math.sqrt(ann_factor)) / (std_return * math.sqrt(ann_factor))
        # Simplified: sharpe = (mean_return - rf_per_period) / std_return * sqrt(ann_factor)
        sharpe = (mean_return - rf_per_period) / std_return * math.sqrt(ann_factor)

    # --- Sortino Ratio ---
    # Uses downside deviation instead of total std
    downside_returns = returns[returns < 0]
    downside_std = np.std(downside_returns, ddof=1) if len(downside_returns) > 1 else 0
    sortino = 0
    if downside_std > 0:
        sortino = (mean_return - rf_per_period) / downside_std * math.sqrt(ann_factor)

    # --- Max Drawdown ---
    peak = equity[0]
    max_dd = 0
    max_dd_pct = 0
    dd_start = 0
    dd_end = 0
    current_dd_start = 0

    for i in range(len(equity)):
        if equity[i] > peak:
            peak = equity[i]
            current_dd_start = i
        dd = peak - equity[i]
        dd_pct = dd / peak if peak > 0 else 0
        if dd_pct > max_dd_pct:
            max_dd = dd
            max_dd_pct = dd_pct
            dd_start = current_dd_start
            dd_end = i

    # --- Calmar Ratio ---
    # Annualized return / max drawdown
    calmar = 0
    if max_dd_pct > 0:
        calmar = ann_return / max_dd_pct

    # --- Profit Factor ---
    gross_wins = sum(wins) if wins else 0
    gross_losses = sum(abs(l) for l in losses) if losses else 0
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    # --- Expectancy ---
    avg_win = np.mean(wins) if wins else 0
    avg_loss = np.mean([abs(l) for l in losses]) if losses else 0
    expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

    # --- Payoff Ratio ---
    payoff_ratio = avg_win / avg_loss if avg_loss > 0 else float("inf")

    # --- Recovery Factor ---
    recovery_factor = total_pnl / max_dd if max_dd > 0 else float("inf")

    # --- Ulcer Index (measure of drawdown severity) ---
    drawdowns_pct = []
    peak_val = equity[0]
    for e in equity:
        if e > peak_val:
            peak_val = e
        dd_pct_val = (peak_val - e) / peak_val if peak_val > 0 else 0
        drawdowns_pct.append(dd_pct_val ** 2)
    ulcer_index = math.sqrt(np.mean(drawdowns_pct)) if drawdowns_pct else 0

    # --- Win/Loss streaks ---
    max_win_streak = 0
    max_loss_streak = 0
    curr_streak = 0
    streak_type = None
    for p in pnls:
        if p > 0:
            if streak_type == "win":
                curr_streak += 1
            else:
                curr_streak = 1
                streak_type = "win"
            max_win_streak = max(max_win_streak, curr_streak)
        else:
            if streak_type == "loss":
                curr_streak += 1
            else:
                curr_streak = 1
                streak_type = "loss"
            max_loss_streak = max(max_loss_streak, curr_streak)

    # --- Kelly Criterion ---
    # Optimal fraction of capital to risk
    kelly = 0
    if payoff_ratio != float("inf") and avg_loss > 0:
        kelly = win_rate - ((1 - win_rate) / payoff_ratio)
        kelly = max(0, kelly)  # Don't recommend negative sizing

    return {
        "total_trades": n,
        "total_pnl": round(total_pnl, 2),
        "total_return_pct": round(total_return * 100, 2),
        "win_rate": round(win_rate * 100, 1),
        "avg_win": round(float(avg_win), 2),
        "avg_loss": round(float(avg_loss), 2),
        "max_win": round(max(pnls), 2),
        "max_loss": round(min(pnls), 2),
        "profit_factor": round(profit_factor, 2) if profit_factor != float("inf") else "inf",
        "expectancy": round(expectancy, 2),
        "payoff_ratio": round(payoff_ratio, 2) if payoff_ratio != float("inf") else "inf",
        # Risk-adjusted
        "sharpe_ratio": round(sharpe, 2),
        "sortino_ratio": round(sortino, 2),
        "calmar_ratio": round(calmar, 2),
        # Drawdown
        "max_drawdown": round(max_dd, 2),
        "max_drawdown_pct": round(max_dd_pct * 100, 2),
        "recovery_factor": round(recovery_factor, 2) if recovery_factor != float("inf") else "inf",
        "ulcer_index": round(ulcer_index * 100, 2),
        # Volatility
        "annualized_return_pct": round(ann_return * 100, 2),
        "annualized_volatility_pct": round(ann_volatility * 100, 2),
        # Sizing
        "kelly_criterion": round(kelly * 100, 1),
        # Streaks
        "max_win_streak": max_win_streak,
        "max_loss_streak": max_loss_streak,
        # Equity
        "starting_capital": starting_capital,
        "ending_capital": round(equity[-1], 2),
        "equity_curve": [round(e, 2) for e in equity.tolist()],
    }


def load_paper_pnls() -> tuple:
    """Load P&L values from paper trading history.

    Returns (pnls, starting_capital).
    """
    try:
        import paper_trader
        state = paper_trader.load_state()
        pnls = [t["pnl"] for t in state.get("closed_trades", [])]
        capital = state.get("starting_capital", 100000)
        return pnls, capital
    except Exception as e:
        logger.error("Failed to load paper trades: %s", e)
        return [], 100000


def load_backtest_pnls() -> tuple:
    """Load P&L values from latest backtest.

    Returns (pnls, starting_capital).
    """
    import glob
    files = sorted(glob.glob("logs/backtest_*.csv"))
    if not files:
        return [], 100000

    import csv
    pnls = []
    with open(files[-1], newline="") as f:
        for row in csv.DictReader(f):
            try:
                pnls.append(float(row.get("pnl", 0)))
            except (ValueError, TypeError):
                pass

    return pnls, 100000


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_risk_report(metrics: Dict) -> None:
    """Print formatted risk-adjusted performance report."""
    if "error" in metrics:
        print(f"\n  {metrics['error']}")
        return

    print(f"\n{'=' * 60}")
    print(f"  RISK-ADJUSTED PERFORMANCE REPORT")
    print(f"{'=' * 60}")

    print(f"\n  RETURNS")
    print(f"  {'─' * 50}")
    print(f"  Total Trades:         {metrics['total_trades']}")
    print(f"  Total P&L:            ${metrics['total_pnl']:>12,.2f}")
    print(f"  Total Return:         {metrics['total_return_pct']:>+8.2f}%")
    print(f"  Annualized Return:    {metrics['annualized_return_pct']:>+8.2f}%")
    print(f"  Win Rate:             {metrics['win_rate']:>8.1f}%")

    print(f"\n  RISK-ADJUSTED RATIOS")
    print(f"  {'─' * 50}")
    print(f"  Sharpe Ratio:         {metrics['sharpe_ratio']:>8.2f}")
    print(f"  Sortino Ratio:        {metrics['sortino_ratio']:>8.2f}")
    print(f"  Calmar Ratio:         {metrics['calmar_ratio']:>8.2f}")

    # Interpret Sharpe
    sr = metrics["sharpe_ratio"]
    if sr >= 2.0:
        quality = "Excellent"
    elif sr >= 1.0:
        quality = "Good"
    elif sr >= 0.5:
        quality = "Moderate"
    elif sr >= 0:
        quality = "Below average"
    else:
        quality = "Poor"
    print(f"  Sharpe Quality:       {quality}")

    print(f"\n  DRAWDOWN")
    print(f"  {'─' * 50}")
    print(f"  Max Drawdown:         ${metrics['max_drawdown']:>12,.2f}")
    print(f"  Max Drawdown %:       {metrics['max_drawdown_pct']:>8.2f}%")
    print(f"  Recovery Factor:      {metrics['recovery_factor']}")
    print(f"  Ulcer Index:          {metrics['ulcer_index']:>8.2f}%")

    print(f"\n  TRADE QUALITY")
    print(f"  {'─' * 50}")
    print(f"  Profit Factor:        {metrics['profit_factor']}")
    print(f"  Expectancy:           ${metrics['expectancy']:>10,.2f} / trade")
    print(f"  Payoff Ratio:         {metrics['payoff_ratio']}")
    print(f"  Avg Win:              ${metrics['avg_win']:>10,.2f}")
    print(f"  Avg Loss:             ${metrics['avg_loss']:>10,.2f}")
    print(f"  Max Win:              ${metrics['max_win']:>10,.2f}")
    print(f"  Max Loss:             ${metrics['max_loss']:>10,.2f}")

    print(f"\n  VOLATILITY & SIZING")
    print(f"  {'─' * 50}")
    print(f"  Annualized Vol:       {metrics['annualized_volatility_pct']:>8.2f}%")
    print(f"  Kelly Criterion:      {metrics['kelly_criterion']:>8.1f}%")
    print(f"  Win Streak:           {metrics['max_win_streak']}")
    print(f"  Loss Streak:          {metrics['max_loss_streak']}")

    # Equity sparkline
    eq = metrics.get("equity_curve", [])
    if len(eq) > 2:
        min_eq = min(eq)
        max_eq = max(eq)
        span = max_eq - min_eq if max_eq != min_eq else 1
        bars = "▁▂▃▄▅▆▇█"
        sparkline = ""
        step = max(1, len(eq) // 50)
        for i in range(0, len(eq), step):
            idx = int((eq[i] - min_eq) / span * (len(bars) - 1))
            sparkline += bars[idx]
        print(f"\n  EQUITY CURVE")
        print(f"  {'─' * 50}")
        print(f"  {sparkline}")
        print(f"  ${min_eq:,.0f} {'─' * 25} ${max_eq:,.0f}")

    print(f"\n{'=' * 60}\n")


def main():
    parser = argparse.ArgumentParser(description="Risk-Adjusted Performance Metrics")
    parser.add_argument("--paper", action="store_true", help="Analyze paper trading history")
    parser.add_argument("--backtest", action="store_true", help="Analyze latest backtest")
    parser.add_argument("--capital", type=float, default=100000, help="Starting capital")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

    if args.paper:
        pnls, capital = load_paper_pnls()
    elif args.backtest:
        pnls, capital = load_backtest_pnls()
    else:
        # Try paper first, then backtest
        pnls, capital = load_paper_pnls()
        if not pnls:
            pnls, capital = load_backtest_pnls()

    if args.capital:
        capital = args.capital

    if not pnls:
        print("\n  No trade data found. Run backtester or paper trader first.\n")
        return

    metrics = compute_metrics(pnls, starting_capital=capital)
    print_risk_report(metrics)


if __name__ == "__main__":
    main()
