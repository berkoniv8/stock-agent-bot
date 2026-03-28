#!/usr/bin/env python3
"""
Portfolio Risk Monitor — tracks total exposure, sector concentration,
correlation between holdings, and overall portfolio risk.

Usage:
    python3 risk_monitor.py                # Full risk report
    python3 risk_monitor.py --update       # Update portfolio prices and save
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

PORTFOLIO_FILE = Path("portfolio.json")
WATCHLIST_FILE = Path("watchlist.csv")


def load_portfolio() -> dict:
    with open(PORTFOLIO_FILE) as f:
        return json.load(f)


def load_sector_map() -> Dict[str, str]:
    """Build ticker -> sector map from watchlist."""
    import csv
    sectors = {}
    if WATCHLIST_FILE.exists():
        with open(WATCHLIST_FILE, newline="") as f:
            for row in csv.DictReader(f):
                sectors[row["ticker"].upper()] = row.get("sector", "Unknown")
    return sectors


def fetch_current_prices(tickers: List[str]) -> Dict[str, float]:
    """Fetch current prices for a list of tickers."""
    prices = {}
    for ticker in tickers:
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(period="1d")
            if not hist.empty:
                prices[ticker] = float(hist["Close"].iloc[-1])
        except Exception as e:
            logger.error("Error fetching %s: %s", ticker, e)
    return prices


def fetch_returns(tickers: List[str], period: str = "3mo") -> pd.DataFrame:
    """Fetch daily returns for correlation analysis."""
    frames = {}
    for ticker in tickers:
        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(period=period)
            if not hist.empty:
                frames[ticker] = hist["Close"].pct_change().dropna()
        except Exception:
            pass

    if not frames:
        return pd.DataFrame()

    return pd.DataFrame(frames).dropna()


def compute_correlation_matrix(returns_df: pd.DataFrame) -> pd.DataFrame:
    """Compute pairwise correlation matrix."""
    if returns_df.empty or len(returns_df.columns) < 2:
        return pd.DataFrame()
    return returns_df.corr()


def compute_portfolio_volatility(returns_df: pd.DataFrame, weights: Dict[str, float]) -> float:
    """Compute portfolio-level annualized volatility using correlation-weighted approach."""
    if returns_df.empty:
        return 0.0

    tickers = [t for t in returns_df.columns if t in weights]
    if len(tickers) < 2:
        return 0.0

    w = np.array([weights.get(t, 0) for t in tickers])
    w = w / w.sum() if w.sum() > 0 else w

    cov_matrix = returns_df[tickers].cov() * 252  # annualize
    port_var = np.dot(w, np.dot(cov_matrix, w))
    return float(np.sqrt(port_var)) * 100  # as percentage


def analyze_risk(update_prices: bool = False) -> dict:
    """Run full portfolio risk analysis.

    Returns a dict with all risk metrics.
    """
    portfolio = load_portfolio()
    holdings = portfolio.get("holdings", [])
    total_value = portfolio.get("total_portfolio_value", 0)
    cash = portfolio.get("available_cash", 0)
    sector_map = load_sector_map()

    if not holdings:
        return {"error": "No holdings configured"}

    tickers = [h["ticker"] for h in holdings]

    # Fetch current prices
    prices = fetch_current_prices(tickers)

    # Update holdings with current values
    for h in holdings:
        ticker = h["ticker"]
        if ticker in prices:
            h["current_price"] = prices[ticker]
            h["current_value"] = prices[ticker] * h.get("shares", 0)
            h["cost_basis"] = h.get("avg_cost", 0) * h.get("shares", 0)
            h["unrealized_pnl"] = h["current_value"] - h["cost_basis"]
            h["pnl_pct"] = (h["unrealized_pnl"] / h["cost_basis"] * 100) if h["cost_basis"] else 0
            h["sector"] = sector_map.get(ticker, "Unknown")

    # Recalculate total invested
    total_invested = sum(h.get("current_value", 0) for h in holdings)
    actual_total = total_invested + cash

    # Sector concentration
    sector_exposure = defaultdict(float)
    for h in holdings:
        sector = h.get("sector", "Unknown")
        sector_exposure[sector] += h.get("current_value", 0)

    sector_pcts = {s: (v / actual_total * 100) if actual_total else 0 for s, v in sector_exposure.items()}

    # Position concentration (largest position as % of portfolio)
    position_pcts = {}
    for h in holdings:
        position_pcts[h["ticker"]] = (h.get("current_value", 0) / actual_total * 100) if actual_total else 0

    max_position = max(position_pcts.values()) if position_pcts else 0
    max_position_ticker = max(position_pcts, key=position_pcts.get) if position_pcts else ""

    # Correlation analysis
    returns_df = fetch_returns(tickers)
    corr_matrix = compute_correlation_matrix(returns_df)

    # Portfolio weights for volatility calc
    weights = {h["ticker"]: h.get("current_value", 0) for h in holdings}
    port_volatility = compute_portfolio_volatility(returns_df, weights)

    # Individual stock volatility
    individual_vol = {}
    if not returns_df.empty:
        for ticker in returns_df.columns:
            individual_vol[ticker] = float(returns_df[ticker].std() * np.sqrt(252) * 100)

    # Cash ratio
    cash_pct = (cash / actual_total * 100) if actual_total else 0

    # Total unrealized P&L
    total_pnl = sum(h.get("unrealized_pnl", 0) for h in holdings)
    total_pnl_pct = (total_pnl / sum(h.get("cost_basis", 0) for h in holdings) * 100) if any(h.get("cost_basis") for h in holdings) else 0

    # Highest correlated pair
    high_corr_pairs = []
    if not corr_matrix.empty:
        for i in range(len(corr_matrix)):
            for j in range(i + 1, len(corr_matrix)):
                t1, t2 = corr_matrix.index[i], corr_matrix.columns[j]
                c = corr_matrix.iloc[i, j]
                if abs(c) > 0.7:
                    high_corr_pairs.append((t1, t2, round(c, 2)))

    # Update portfolio.json if requested
    if update_prices:
        portfolio["holdings"] = holdings
        portfolio["total_portfolio_value"] = round(actual_total, 2)
        portfolio["_last_updated"] = datetime.now().isoformat()
        with open(PORTFOLIO_FILE, "w") as f:
            json.dump(portfolio, f, indent=4)
        logger.info("Portfolio prices updated in %s", PORTFOLIO_FILE)

    return {
        "total_value": round(actual_total, 2),
        "total_invested": round(total_invested, 2),
        "cash": cash,
        "cash_pct": round(cash_pct, 1),
        "holdings": holdings,
        "sector_exposure": dict(sector_pcts),
        "position_concentration": position_pcts,
        "max_position_pct": round(max_position, 1),
        "max_position_ticker": max_position_ticker,
        "portfolio_volatility": round(port_volatility, 1),
        "individual_volatility": individual_vol,
        "correlation_matrix": corr_matrix.to_dict() if not corr_matrix.empty else {},
        "high_correlation_pairs": high_corr_pairs,
        "total_unrealized_pnl": round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl_pct, 1),
    }


def print_risk_report(risk: dict) -> None:
    """Print formatted risk report."""
    if "error" in risk:
        print(f"Error: {risk['error']}")
        return

    print(f"\n{'=' * 65}")
    print(f"  PORTFOLIO RISK MONITOR")
    print(f"{'=' * 65}")

    # Overview
    print(f"\n  OVERVIEW")
    print(f"  {'─' * 55}")
    print(f"  Total value:          ${risk['total_value']:>12,.2f}")
    print(f"  Invested:             ${risk['total_invested']:>12,.2f}")
    print(f"  Cash:                 ${risk['cash']:>12,.2f}  ({risk['cash_pct']}%)")
    print(f"  Unrealized P&L:       ${risk['total_unrealized_pnl']:>12,.2f}  ({risk['total_pnl_pct']:+.1f}%)")
    print(f"  Portfolio volatility: {risk['portfolio_volatility']:>12.1f}% (annualized)")

    # Holdings
    print(f"\n  HOLDINGS")
    print(f"  {'─' * 55}")
    print(f"  {'Ticker':<6} {'Shares':>6} {'Price':>9} {'Value':>10} {'P&L':>9} {'%':>7} {'Vol':>6}")
    for h in risk.get("holdings", []):
        vol = risk.get("individual_volatility", {}).get(h["ticker"], 0)
        print(f"  {h['ticker']:<6} {h.get('shares', 0):>6} ${h.get('current_price', 0):>8,.2f} ${h.get('current_value', 0):>9,.2f} ${h.get('unrealized_pnl', 0):>8,.2f} {h.get('pnl_pct', 0):>+6.1f}% {vol:>5.1f}%")

    # Sector exposure
    print(f"\n  SECTOR EXPOSURE")
    print(f"  {'─' * 55}")
    for sector, pct in sorted(risk.get("sector_exposure", {}).items(), key=lambda x: -x[1]):
        bar = "#" * int(pct / 2)
        print(f"  {sector:<28} {pct:>5.1f}%  {bar}")

    # Concentration risk
    print(f"\n  CONCENTRATION RISK")
    print(f"  {'─' * 55}")
    print(f"  Largest position: {risk['max_position_ticker']} ({risk['max_position_pct']}%)")
    max_pos_limit = float(os.getenv("MAX_POSITION_SIZE_PCT", "10"))
    if risk["max_position_pct"] > max_pos_limit:
        print(f"  WARNING: Exceeds {max_pos_limit}% position limit!")

    # Correlation warnings
    pairs = risk.get("high_correlation_pairs", [])
    if pairs:
        print(f"\n  HIGH CORRELATION PAIRS (>0.70)")
        print(f"  {'─' * 55}")
        for t1, t2, corr in pairs:
            print(f"  {t1} / {t2}: {corr:.2f}")
    else:
        print(f"\n  No highly correlated pairs (>0.70) detected.")

    print(f"\n{'=' * 65}\n")


def main():
    parser = argparse.ArgumentParser(description="Portfolio Risk Monitor")
    parser.add_argument("--update", action="store_true", help="Update portfolio.json with current prices")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
    for lib in ("urllib3", "yfinance", "peewee"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    risk = analyze_risk(update_prices=args.update)
    print_risk_report(risk)


if __name__ == "__main__":
    main()
