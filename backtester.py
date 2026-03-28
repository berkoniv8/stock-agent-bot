#!/usr/bin/env python3
"""
Backtesting Engine — replays historical data bar-by-bar,
evaluates signals at each step, and tracks hypothetical P&L.

Usage:
    python3 backtester.py                         # Backtest full watchlist
    python3 backtester.py --ticker AAPL           # Single ticker
    python3 backtester.py --period 2y             # Custom lookback
    python3 backtester.py --threshold 4           # Lower signal threshold
"""

import argparse
import csv
import logging
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import pandas as pd
from dotenv import load_dotenv

import data_layer
import technical_analysis
import fundamental_analysis
import signal_engine

load_dotenv()
logger = logging.getLogger(__name__)

RESULTS_DIR = Path("logs")
RESULTS_DIR.mkdir(exist_ok=True)


@dataclass
class BacktestTrade:
    """A single simulated trade."""
    ticker: str
    direction: str
    entry_date: str
    entry_price: float
    stop_loss: float
    target_1: float
    target_2: float
    exit_date: str = ""
    exit_price: float = 0.0
    exit_reason: str = ""
    pnl: float = 0.0
    pnl_pct: float = 0.0
    bars_held: int = 0
    signal_score: int = 0
    signals: str = ""


@dataclass
class BacktestResult:
    """Aggregate results of a backtest run."""
    ticker: str
    period: str
    total_bars: int = 0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    total_pnl: float = 0.0
    avg_pnl: float = 0.0
    max_win: float = 0.0
    max_loss: float = 0.0
    avg_bars_held: float = 0.0
    profit_factor: float = 0.0
    trades: List[BacktestTrade] = field(default_factory=list)


def _compute_position(entry_price: float, stop_loss: float, direction: str) -> dict:
    """Compute simplified position targets."""
    risk = abs(entry_price - stop_loss)
    if direction == "BUY":
        return {
            "target_1": entry_price + 1.5 * risk,
            "target_2": entry_price + 3.0 * risk,
        }
    else:
        return {
            "target_1": entry_price - 1.5 * risk,
            "target_2": entry_price - 3.0 * risk,
        }


def backtest_ticker(
    ticker: str,
    period: str = "2y",
    threshold: int = 5,
    min_bars_for_signal: int = 200,
    sector: str = "Technology",
) -> BacktestResult:
    """Run a walk-forward backtest on a single ticker.

    Slides a window through historical data, computing signals at each bar.
    When a signal fires, simulates a trade and tracks outcome.
    """
    logger.info("Backtesting %s over %s (threshold=%d)", ticker, period, threshold)

    df = data_layer.fetch_daily_ohlcv(ticker, period=period)
    if df.empty or len(df) < min_bars_for_signal + 30:
        logger.warning("Insufficient data for %s backtest (%d bars)", ticker, len(df))
        return BacktestResult(ticker=ticker, period=period, total_bars=len(df))

    result = BacktestResult(ticker=ticker, period=period, total_bars=len(df))

    # Pre-fetch fundamentals once (they don't change bar-by-bar)
    fund = fundamental_analysis.FundamentalSignals(ticker=ticker, fundamental_score=3)

    open_trade: Optional[BacktestTrade] = None
    cooldown = 0  # bars to wait after closing a trade

    for i in range(min_bars_for_signal, len(df)):
        bar = df.iloc[i]
        bar_date = str(df.index[i].date())
        window = df.iloc[:i + 1]

        # If we have an open trade, check for exit
        if open_trade is not None:
            open_trade.bars_held += 1
            hit_stop = False
            hit_target = False

            if open_trade.direction == "BUY":
                hit_stop = bar["Low"] <= open_trade.stop_loss
                hit_target = bar["High"] >= open_trade.target_1
            else:
                hit_stop = bar["High"] >= open_trade.stop_loss
                hit_target = bar["Low"] <= open_trade.target_1

            if hit_stop:
                open_trade.exit_date = bar_date
                open_trade.exit_price = open_trade.stop_loss
                open_trade.exit_reason = "stop_loss"
            elif hit_target:
                open_trade.exit_date = bar_date
                open_trade.exit_price = open_trade.target_1
                open_trade.exit_reason = "target_1"
            elif open_trade.bars_held >= 30:
                # Time-based exit after 30 bars
                open_trade.exit_date = bar_date
                open_trade.exit_price = bar["Close"]
                open_trade.exit_reason = "time_exit"

            if open_trade.exit_date:
                # Close the trade
                if open_trade.direction == "BUY":
                    open_trade.pnl = open_trade.exit_price - open_trade.entry_price
                else:
                    open_trade.pnl = open_trade.entry_price - open_trade.exit_price
                open_trade.pnl_pct = (open_trade.pnl / open_trade.entry_price) * 100

                result.trades.append(open_trade)
                open_trade = None
                cooldown = 5  # wait 5 bars before next trade
            continue

        # Cooldown period
        if cooldown > 0:
            cooldown -= 1
            continue

        # Evaluate signals on the window up to this bar
        tech = technical_analysis.analyze(ticker, window)
        alert = signal_engine.evaluate(tech, fund, threshold=threshold)

        if alert is not None:
            entry_price = bar["Close"]

            # Set stop-loss
            if alert.direction == "BUY":
                stop_candidates = [
                    tech.ema21 * 0.995 if tech.ema21 > 0 and tech.ema21 < entry_price else 0,
                    tech.support_level * 0.995 if tech.support_level > 0 and tech.support_level < entry_price else 0,
                ]
                stop_candidates = [s for s in stop_candidates if s > 0]
                stop_loss = max(stop_candidates) if stop_candidates else entry_price * 0.97
            else:
                stop_candidates = [
                    tech.ema21 * 1.005 if tech.ema21 > 0 and tech.ema21 > entry_price else 0,
                    tech.resistance_level * 1.005 if tech.resistance_level > 0 and tech.resistance_level > entry_price else 0,
                ]
                stop_candidates = [s for s in stop_candidates if s > 0]
                stop_loss = min(stop_candidates) if stop_candidates else entry_price * 1.03

            targets = _compute_position(entry_price, stop_loss, alert.direction)

            open_trade = BacktestTrade(
                ticker=ticker,
                direction=alert.direction,
                entry_date=bar_date,
                entry_price=round(entry_price, 2),
                stop_loss=round(stop_loss, 2),
                target_1=round(targets["target_1"], 2),
                target_2=round(targets["target_2"], 2),
                signal_score=alert.signal_score,
                signals="|".join(s[0] for s in alert.triggered_signals),
            )

    # Close any open trade at the end
    if open_trade is not None:
        open_trade.exit_date = str(df.index[-1].date())
        open_trade.exit_price = df["Close"].iloc[-1]
        open_trade.exit_reason = "end_of_data"
        if open_trade.direction == "BUY":
            open_trade.pnl = open_trade.exit_price - open_trade.entry_price
        else:
            open_trade.pnl = open_trade.entry_price - open_trade.exit_price
        open_trade.pnl_pct = (open_trade.pnl / open_trade.entry_price) * 100
        result.trades.append(open_trade)

    # Compute aggregate stats
    result.total_trades = len(result.trades)
    if result.total_trades > 0:
        pnls = [t.pnl for t in result.trades]
        result.winning_trades = sum(1 for p in pnls if p > 0)
        result.losing_trades = sum(1 for p in pnls if p <= 0)
        result.win_rate = result.winning_trades / result.total_trades * 100
        result.total_pnl = sum(pnls)
        result.avg_pnl = result.total_pnl / result.total_trades
        result.max_win = max(pnls) if pnls else 0
        result.max_loss = min(pnls) if pnls else 0
        result.avg_bars_held = sum(t.bars_held for t in result.trades) / result.total_trades

        gross_wins = sum(p for p in pnls if p > 0)
        gross_losses = abs(sum(p for p in pnls if p < 0))
        result.profit_factor = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    return result


def print_report(result: BacktestResult) -> None:
    """Print a formatted backtest report."""
    print(f"\n{'=' * 60}")
    print(f"  BACKTEST REPORT: {result.ticker}")
    print(f"{'=' * 60}")
    print(f"  Period:           {result.period} ({result.total_bars} bars)")
    print(f"  Total trades:     {result.total_trades}")
    print(f"  Winning:          {result.winning_trades}")
    print(f"  Losing:           {result.losing_trades}")
    print(f"  Win rate:         {result.win_rate:.1f}%")
    print(f"  Total P&L:        ${result.total_pnl:,.2f}")
    print(f"  Avg P&L/trade:    ${result.avg_pnl:,.2f}")
    print(f"  Max win:          ${result.max_win:,.2f}")
    print(f"  Max loss:         ${result.max_loss:,.2f}")
    print(f"  Profit factor:    {result.profit_factor:.2f}")
    print(f"  Avg bars held:    {result.avg_bars_held:.1f}")

    if result.trades:
        print(f"\n  {'─' * 56}")
        print(f"  {'Date':>12} {'Dir':>4} {'Entry':>8} {'Exit':>8} {'P&L':>8} {'%':>6} {'Reason':>10} {'Bars':>4}")
        print(f"  {'─' * 56}")
        for t in result.trades:
            pnl_str = f"${t.pnl:,.2f}"
            pct_str = f"{t.pnl_pct:+.1f}%"
            print(f"  {t.entry_date:>12} {t.direction:>4} ${t.entry_price:>7,.2f} ${t.exit_price:>7,.2f} {pnl_str:>8} {pct_str:>6} {t.exit_reason:>10} {t.bars_held:>4}")

    print(f"{'=' * 60}\n")


def save_results_csv(results: List[BacktestResult]) -> str:
    """Save backtest results to CSV."""
    path = RESULTS_DIR / f"backtest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"

    rows = []
    for r in results:
        for t in r.trades:
            rows.append({
                "ticker": t.ticker,
                "direction": t.direction,
                "entry_date": t.entry_date,
                "entry_price": t.entry_price,
                "stop_loss": t.stop_loss,
                "target_1": t.target_1,
                "exit_date": t.exit_date,
                "exit_price": t.exit_price,
                "exit_reason": t.exit_reason,
                "pnl": round(t.pnl, 2),
                "pnl_pct": round(t.pnl_pct, 2),
                "bars_held": t.bars_held,
                "signal_score": t.signal_score,
                "signals": t.signals,
            })

    if rows:
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        logger.info("Backtest results saved to %s", path)

    return str(path)


def main():
    parser = argparse.ArgumentParser(description="Stock Agent Backtester")
    parser.add_argument("--ticker", type=str, help="Single ticker to backtest")
    parser.add_argument("--period", type=str, default="2y", help="Lookback period (default: 2y)")
    parser.add_argument("--threshold", type=int, default=5, help="Signal threshold (default: 5)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
    for lib in ("urllib3", "yfinance", "peewee"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    if args.ticker:
        tickers = [{"ticker": args.ticker, "sector": "Technology"}]
    else:
        tickers = data_layer.load_watchlist()

    results = []
    for entry in tickers:
        try:
            r = backtest_ticker(
                entry["ticker"],
                period=args.period,
                threshold=args.threshold,
                sector=entry.get("sector", "Technology"),
            )
            results.append(r)
            print_report(r)
        except Exception as e:
            logger.error("Backtest error for %s: %s", entry["ticker"], e, exc_info=True)

    # Summary across all tickers
    if len(results) > 1:
        total_trades = sum(r.total_trades for r in results)
        total_wins = sum(r.winning_trades for r in results)
        total_pnl = sum(r.total_pnl for r in results)
        print(f"\n{'=' * 60}")
        print(f"  PORTFOLIO BACKTEST SUMMARY")
        print(f"{'=' * 60}")
        print(f"  Tickers tested:   {len(results)}")
        print(f"  Total trades:     {total_trades}")
        print(f"  Win rate:         {total_wins / total_trades * 100:.1f}%" if total_trades > 0 else "  Win rate:         N/A")
        print(f"  Total P&L:        ${total_pnl:,.2f}")
        print(f"{'=' * 60}\n")

    if results:
        csv_path = save_results_csv(results)
        print(f"Results saved to: {csv_path}")


if __name__ == "__main__":
    main()
