#!/usr/bin/env python3
"""
Trailing Stop Manager — tracks open positions and adjusts stop-losses
upward as price moves in the trade's favor.

Supports:
- ATR-based trailing (2x ATR below highest close)
- Percentage-based trailing (configurable %)
- Chandelier exit (highest high - N*ATR)

Tracks open positions in logs/open_positions.json.

Usage:
    python3 trailing_stop.py              # Check all open positions
    python3 trailing_stop.py --add AAPL BUY 252.62 245.00 20
    python3 trailing_stop.py --close AAPL
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

POSITIONS_FILE = Path("logs/open_positions.json")


def _load_positions() -> List[Dict]:
    if not POSITIONS_FILE.exists():
        return []
    try:
        with open(POSITIONS_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def _save_positions(positions: List[Dict]) -> None:
    POSITIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(POSITIONS_FILE, "w") as f:
        json.dump(positions, f, indent=2)


def add_position(
    ticker: str,
    direction: str,
    entry_price: float,
    initial_stop: float,
    shares: int,
    target_1: float = 0,
    target_2: float = 0,
    target_3: float = 0,
) -> None:
    """Add a new open position to track."""
    positions = _load_positions()

    # Check for duplicate
    for p in positions:
        if p["ticker"] == ticker and p["status"] == "open":
            logger.warning("%s already has an open position", ticker)
            return

    position = {
        "ticker": ticker,
        "direction": direction,
        "entry_price": entry_price,
        "entry_date": datetime.now().isoformat(),
        "initial_stop": initial_stop,
        "current_stop": initial_stop,
        "trailing_stop": initial_stop,
        "highest_price": entry_price,
        "lowest_price": entry_price,
        "shares": shares,
        "target_1": target_1,
        "target_2": target_2,
        "target_3": target_3,
        "t1_hit": False,
        "t2_hit": False,
        "t3_hit": False,
        "status": "open",
        "updates": [],
    }
    positions.append(position)
    _save_positions(positions)
    logger.info("Added position: %s %s @ %.2f, stop %.2f, %d shares",
                direction, ticker, entry_price, initial_stop, shares)


def close_position(ticker: str, exit_price: float = 0, reason: str = "manual") -> Optional[Dict]:
    """Close an open position."""
    positions = _load_positions()
    closed = None

    for p in positions:
        if p["ticker"] == ticker and p["status"] == "open":
            if exit_price == 0:
                try:
                    tk = yf.Ticker(ticker)
                    hist = tk.history(period="1d")
                    if not hist.empty:
                        exit_price = float(hist["Close"].iloc[-1])
                except Exception:
                    pass

            p["status"] = "closed"
            p["exit_price"] = exit_price
            p["exit_date"] = datetime.now().isoformat()
            p["exit_reason"] = reason

            if p["direction"] == "BUY":
                p["pnl"] = (exit_price - p["entry_price"]) * p["shares"]
            else:
                p["pnl"] = (p["entry_price"] - exit_price) * p["shares"]

            p["pnl_pct"] = (p["pnl"] / (p["entry_price"] * p["shares"])) * 100
            closed = p
            break

    _save_positions(positions)
    if closed:
        logger.info("Closed %s: P&L $%.2f (%.1f%%)", ticker, closed["pnl"], closed["pnl_pct"])
    return closed


def compute_trailing_stop(
    current_price: float,
    highest_price: float,
    lowest_price: float,
    direction: str,
    atr: float,
    method: str = "atr",
    trail_pct: float = 3.0,
    atr_mult: float = 2.0,
) -> float:
    """Compute the trailing stop level.

    Methods:
    - 'atr':     Chandelier exit — highest_high - atr_mult * ATR (for longs)
    - 'percent': Fixed percentage trail from highest price
    - 'hybrid':  Max of ATR and percentage methods (tightest stop)
    """
    if direction == "BUY":
        atr_stop = highest_price - atr_mult * atr if atr > 0 else 0
        pct_stop = highest_price * (1 - trail_pct / 100)

        if method == "atr":
            return atr_stop
        elif method == "percent":
            return pct_stop
        else:  # hybrid
            return max(atr_stop, pct_stop)
    else:
        atr_stop = lowest_price + atr_mult * atr if atr > 0 else float("inf")
        pct_stop = lowest_price * (1 + trail_pct / 100)

        if method == "atr":
            return atr_stop
        elif method == "percent":
            return pct_stop
        else:  # hybrid
            return min(atr_stop, pct_stop)


def update_positions() -> List[Dict]:
    """Update all open positions with current prices and trailing stops.

    Returns list of position updates with actions taken.
    """
    positions = _load_positions()
    updates = []

    open_positions = [p for p in positions if p["status"] == "open"]
    if not open_positions:
        logger.info("No open positions to update")
        return updates

    for pos in open_positions:
        ticker = pos["ticker"]
        direction = pos["direction"]

        try:
            tk = yf.Ticker(ticker)
            hist = tk.history(period="5d", interval="1d")
            if hist.empty:
                continue

            current_price = float(hist["Close"].iloc[-1])
            current_high = float(hist["High"].iloc[-1])
            current_low = float(hist["Low"].iloc[-1])

            # Compute ATR from recent data
            if len(hist) >= 2:
                tr_vals = []
                for i in range(1, len(hist)):
                    hl = hist["High"].iloc[i] - hist["Low"].iloc[i]
                    hc = abs(hist["High"].iloc[i] - hist["Close"].iloc[i - 1])
                    lc = abs(hist["Low"].iloc[i] - hist["Close"].iloc[i - 1])
                    tr_vals.append(max(hl, hc, lc))
                atr = float(np.mean(tr_vals)) if tr_vals else 0
            else:
                atr = 0

            # Update high/low watermarks
            if direction == "BUY":
                if current_high > pos["highest_price"]:
                    pos["highest_price"] = current_high
            else:
                if current_low < pos["lowest_price"]:
                    pos["lowest_price"] = current_low

            # Compute new trailing stop
            new_trailing = compute_trailing_stop(
                current_price=current_price,
                highest_price=pos["highest_price"],
                lowest_price=pos["lowest_price"],
                direction=direction,
                atr=atr,
                method="hybrid",
            )

            # Only ratchet the stop in the favorable direction
            old_stop = pos["current_stop"]
            if direction == "BUY":
                if new_trailing > old_stop:
                    pos["current_stop"] = round(new_trailing, 2)
                    pos["trailing_stop"] = round(new_trailing, 2)
            else:
                if new_trailing < old_stop:
                    pos["current_stop"] = round(new_trailing, 2)
                    pos["trailing_stop"] = round(new_trailing, 2)

            # Check if stop was hit
            stop_hit = False
            if direction == "BUY" and current_low <= pos["current_stop"]:
                stop_hit = True
            elif direction == "SELL" and current_high >= pos["current_stop"]:
                stop_hit = True

            # Check targets
            if direction == "BUY":
                if pos.get("target_1") and current_high >= pos["target_1"]:
                    pos["t1_hit"] = True
                if pos.get("target_2") and current_high >= pos["target_2"]:
                    pos["t2_hit"] = True
                if pos.get("target_3") and current_high >= pos["target_3"]:
                    pos["t3_hit"] = True
            else:
                if pos.get("target_1") and current_low <= pos["target_1"]:
                    pos["t1_hit"] = True
                if pos.get("target_2") and current_low <= pos["target_2"]:
                    pos["t2_hit"] = True
                if pos.get("target_3") and current_low <= pos["target_3"]:
                    pos["t3_hit"] = True

            # Compute unrealized P&L
            if direction == "BUY":
                unrealized = (current_price - pos["entry_price"]) * pos["shares"]
            else:
                unrealized = (pos["entry_price"] - current_price) * pos["shares"]

            update = {
                "ticker": ticker,
                "direction": direction,
                "current_price": round(current_price, 2),
                "entry_price": pos["entry_price"],
                "old_stop": round(old_stop, 2),
                "new_stop": pos["current_stop"],
                "stop_moved": pos["current_stop"] != round(old_stop, 2),
                "stop_hit": stop_hit,
                "highest_price": round(pos["highest_price"], 2),
                "unrealized_pnl": round(unrealized, 2),
                "t1_hit": pos["t1_hit"],
                "t2_hit": pos["t2_hit"],
                "atr": round(atr, 2),
            }
            updates.append(update)

            # Log the update
            pos["updates"].append({
                "date": datetime.now().isoformat(),
                "price": current_price,
                "stop": pos["current_stop"],
                "atr": round(atr, 2),
            })

            if stop_hit:
                logger.warning(
                    "%s STOP HIT: price %.2f crossed stop %.2f",
                    ticker, current_price, pos["current_stop"],
                )
                close_position(ticker, pos["current_stop"], "trailing_stop")
            elif update["stop_moved"]:
                logger.info(
                    "%s stop ratcheted: %.2f → %.2f (price: %.2f, high: %.2f)",
                    ticker, old_stop, pos["current_stop"], current_price, pos["highest_price"],
                )

        except Exception as e:
            logger.error("Error updating %s: %s", ticker, e)

    _save_positions(positions)
    return updates


def print_positions() -> None:
    """Print a formatted view of all positions."""
    positions = _load_positions()
    open_pos = [p for p in positions if p["status"] == "open"]
    closed_pos = [p for p in positions if p["status"] == "closed"]

    print(f"\n{'=' * 70}")
    print(f"  TRAILING STOP MANAGER")
    print(f"{'=' * 70}")

    if open_pos:
        print(f"\n  OPEN POSITIONS ({len(open_pos)})")
        print(f"  {'─' * 62}")
        print(f"  {'Ticker':<6} {'Dir':>4} {'Entry':>8} {'Price':>8} {'Stop':>8} {'High':>8} {'P&L':>9} {'T1':>3} {'T2':>3}")
        print(f"  {'─' * 62}")

        for p in open_pos:
            # Fetch current price
            try:
                tk = yf.Ticker(p["ticker"])
                hist = tk.history(period="1d")
                curr = float(hist["Close"].iloc[-1]) if not hist.empty else 0
            except Exception:
                curr = 0

            if p["direction"] == "BUY":
                pnl = (curr - p["entry_price"]) * p["shares"]
            else:
                pnl = (p["entry_price"] - curr) * p["shares"]

            t1 = "Y" if p.get("t1_hit") else "-"
            t2 = "Y" if p.get("t2_hit") else "-"

            print(f"  {p['ticker']:<6} {p['direction']:>4} ${p['entry_price']:>7,.2f} ${curr:>7,.2f} ${p['current_stop']:>7,.2f} ${p['highest_price']:>7,.2f} ${pnl:>8,.2f} {t1:>3} {t2:>3}")
    else:
        print(f"\n  No open positions.")

    if closed_pos:
        print(f"\n  CLOSED POSITIONS ({len(closed_pos)})")
        print(f"  {'─' * 62}")
        for p in closed_pos[-5:]:  # Last 5
            pnl = p.get("pnl", 0)
            print(f"  {p['ticker']:<6} {p['direction']:>4} entry=${p['entry_price']:.2f} exit=${p.get('exit_price', 0):.2f} P&L=${pnl:,.2f} ({p.get('exit_reason', '')})")

    print(f"\n{'=' * 70}\n")


def main():
    parser = argparse.ArgumentParser(description="Trailing Stop Manager")
    parser.add_argument("--add", nargs=5, metavar=("TICKER", "DIR", "ENTRY", "STOP", "SHARES"),
                        help="Add position: TICKER BUY|SELL ENTRY_PRICE STOP_PRICE SHARES")
    parser.add_argument("--close", type=str, help="Close position for ticker")
    parser.add_argument("--update", action="store_true", help="Update all trailing stops")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
    for lib in ("urllib3", "yfinance", "peewee"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    if args.add:
        ticker, direction, entry, stop, shares = args.add
        add_position(ticker.upper(), direction.upper(), float(entry), float(stop), int(shares))
    elif args.close:
        close_position(args.close.upper())
    elif args.update:
        updates = update_positions()
        for u in updates:
            status = "STOP MOVED" if u["stop_moved"] else "unchanged"
            print(f"  {u['ticker']}: ${u['current_price']:.2f} | stop: ${u['new_stop']:.2f} ({status}) | P&L: ${u['unrealized_pnl']:.2f}")
    else:
        print_positions()


if __name__ == "__main__":
    main()
