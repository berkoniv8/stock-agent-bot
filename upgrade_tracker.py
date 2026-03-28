#!/usr/bin/env python3
"""
Upgrade Tracker — monitors analyst rating changes (upgrades/downgrades)
for portfolio holdings and watchlist tickers.

Usage:
    python3 upgrade_tracker.py              # Check last 7 days of rating changes
    python3 upgrade_tracker.py --days 14    # Check last 14 days
    python3 upgrade_tracker.py --notify     # Send alerts via email + Telegram
"""

import argparse
import json
import logging
import os
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def _get_tickers(tickers: Optional[List[str]] = None) -> List[str]:
    """Return list of ticker strings from provided list or watchlist + portfolio."""
    if tickers:
        return [t.upper() for t in tickers]

    all_tickers = set()

    # Load watchlist
    try:
        import data_layer
        watchlist = data_layer.load_watchlist()
        for entry in watchlist:
            all_tickers.add(entry["ticker"].upper())
    except Exception as e:
        logger.warning("Could not load watchlist: %s", e)

    # Load portfolio holdings
    try:
        with open("portfolio.json") as f:
            portfolio = json.load(f)
        for h in portfolio.get("holdings", []):
            all_tickers.add(h["ticker"].upper())
    except Exception as e:
        logger.warning("Could not load portfolio: %s", e)

    return sorted(all_tickers)


def _get_held_tickers() -> set:
    """Return set of tickers currently held in portfolio."""
    try:
        with open("portfolio.json") as f:
            portfolio = json.load(f)
        return set(h["ticker"].upper() for h in portfolio.get("holdings", []))
    except Exception:
        return set()


def _classify_action(row: dict) -> str:
    """Classify the recommendation action."""
    # yfinance recommendations may have different column names across versions
    # Common fields: Firm, To Grade, From Grade, Action
    action = str(row.get("Action", row.get("action", ""))).strip().lower()

    if "up" in action:
        return "Upgrade"
    elif "down" in action:
        return "Downgrade"
    elif "init" in action:
        return "Initiated"
    elif "reit" in action or "main" in action:
        return "Reiterated"
    elif action:
        return action.title()
    else:
        # Infer from grade changes
        to_grade = str(row.get("To Grade", row.get("to_grade", ""))).strip().lower()
        from_grade = str(row.get("From Grade", row.get("from_grade", ""))).strip().lower()

        buy_words = {"buy", "overweight", "outperform", "strong buy", "positive", "accumulate"}
        sell_words = {"sell", "underweight", "underperform", "strong sell", "negative", "reduce"}
        hold_words = {"hold", "neutral", "equal-weight", "market perform", "sector perform", "peer perform", "in-line"}

        def grade_rank(g: str) -> int:
            g = g.lower()
            if any(w in g for w in buy_words):
                return 3
            if any(w in g for w in hold_words):
                return 2
            if any(w in g for w in sell_words):
                return 1
            return 0

        to_rank = grade_rank(to_grade)
        from_rank = grade_rank(from_grade)

        if not from_grade:
            return "Initiated"
        if to_rank > from_rank:
            return "Upgrade"
        if to_rank < from_rank:
            return "Downgrade"
        return "Reiterated"


def fetch_recommendations(ticker: str) -> Optional[pd.DataFrame]:
    """
    Fetch analyst recommendations for a single ticker via yfinance.

    Returns a DataFrame or None on failure.
    """
    try:
        yf_ticker = yf.Ticker(ticker)
        recs = yf_ticker.recommendations
        if recs is None or (hasattr(recs, "empty") and recs.empty):
            return None
        return recs
    except Exception as e:
        logger.debug("No recommendations for %s: %s", ticker, e)
        return None


def scan_all_recommendations(
    tickers: Optional[List[str]] = None,
    days: int = 7,
) -> List[Dict]:
    """
    Check all tickers for analyst rating changes in the last N days.

    Parameters
    ----------
    tickers : list of str, optional
        Tickers to check. If None, loads from watchlist + portfolio.
    days : int
        Look-back window in days (default 7).

    Returns
    -------
    list of dict
        Each dict has: ticker, firm, old_rating, new_rating, action, date, held.
    """
    ticker_list = _get_tickers(tickers)
    held = _get_held_tickers()
    cutoff = datetime.now() - timedelta(days=days)
    changes = []

    logger.info("Scanning %d tickers for rating changes (last %d days)...", len(ticker_list), days)

    for ticker in ticker_list:
        try:
            recs = fetch_recommendations(ticker)
            if recs is None:
                continue

            # The DataFrame may have a DatetimeIndex or a 'Date' column
            df = recs.copy()
            if not isinstance(df.index, pd.DatetimeIndex):
                if "Date" in df.columns:
                    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
                    df = df.set_index("Date")
                elif "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"], errors="coerce")
                    df = df.set_index("date")

            # Filter to recent entries
            if isinstance(df.index, pd.DatetimeIndex):
                # Remove timezone if present
                if df.index.tz is not None:
                    df.index = df.index.tz_localize(None)
                recent = df[df.index >= cutoff]
            else:
                recent = df

            if recent.empty:
                continue

            for idx, row in recent.iterrows():
                row_dict = row.to_dict()

                firm = str(row_dict.get("Firm", row_dict.get("firm", "Unknown")))
                to_grade = str(row_dict.get("To Grade", row_dict.get("to_grade", "")))
                from_grade = str(row_dict.get("From Grade", row_dict.get("from_grade", "")))
                action = _classify_action(row_dict)

                # Determine date string
                if isinstance(idx, (datetime, pd.Timestamp)):
                    date_str = idx.strftime("%Y-%m-%d")
                else:
                    date_str = str(idx)

                changes.append({
                    "ticker": ticker,
                    "firm": firm,
                    "old_rating": from_grade if from_grade and from_grade != "nan" else "",
                    "new_rating": to_grade if to_grade and to_grade != "nan" else "",
                    "action": action,
                    "date": date_str,
                    "held": ticker in held,
                })

        except Exception as e:
            logger.warning("Error fetching recommendations for %s: %s", ticker, e)
            continue

    # Sort by date descending, upgrades/downgrades first
    action_priority = {"Upgrade": 0, "Downgrade": 1, "Initiated": 2, "Reiterated": 3}
    changes.sort(key=lambda x: (action_priority.get(x["action"], 9), x["date"]))
    changes.reverse()
    # Re-sort: most recent first, then by action priority
    changes.sort(key=lambda x: x["date"], reverse=True)

    logger.info("Found %d rating changes.", len(changes))
    return changes


def format_report(changes: List[Dict]) -> str:
    """Format analyst rating changes into a human-readable report."""
    if not changes:
        return "No analyst rating changes found in the period."

    lines = []
    lines.append("Analyst Rating Changes")
    lines.append("=" * 40)
    lines.append("%d rating change(s) found\n" % len(changes))

    # Group upgrades/downgrades first
    upgrades = [c for c in changes if c["action"] == "Upgrade"]
    downgrades = [c for c in changes if c["action"] == "Downgrade"]
    others = [c for c in changes if c["action"] not in ("Upgrade", "Downgrade")]

    if upgrades:
        lines.append("UPGRADES:")
        for c in upgrades:
            held_tag = " [HELD]" if c.get("held") else ""
            arrow = "%s -> %s" % (c["old_rating"], c["new_rating"]) if c["old_rating"] else c["new_rating"]
            lines.append("  %s%s — %s (%s) [%s]" % (
                c["ticker"], held_tag, arrow, c["firm"], c["date"]))
        lines.append("")

    if downgrades:
        lines.append("DOWNGRADES:")
        for c in downgrades:
            held_tag = " [HELD]" if c.get("held") else ""
            arrow = "%s -> %s" % (c["old_rating"], c["new_rating"]) if c["old_rating"] else c["new_rating"]
            lines.append("  %s%s — %s (%s) [%s]" % (
                c["ticker"], held_tag, arrow, c["firm"], c["date"]))
        lines.append("")

    if others:
        lines.append("OTHER CHANGES:")
        for c in others:
            held_tag = " [HELD]" if c.get("held") else ""
            grade_info = c["new_rating"] or "N/A"
            if c["old_rating"]:
                grade_info = "%s -> %s" % (c["old_rating"], c["new_rating"])
            lines.append("  %s%s — %s: %s (%s) [%s]" % (
                c["ticker"], held_tag, c["action"], grade_info, c["firm"], c["date"]))
        lines.append("")

    return "\n".join(lines)


def send_upgrade_alerts(changes: List[Dict]) -> None:
    """Send analyst rating alerts via email and Telegram for upgrades/downgrades."""
    if not changes:
        return

    # Only alert on upgrades and downgrades (not reiterations)
    notable = [c for c in changes if c["action"] in ("Upgrade", "Downgrade")]
    if not notable:
        return

    report = format_report(notable)

    # Telegram
    try:
        import telegram_bot
        telegram_bot.send_message(report)
    except Exception:
        pass

    # Email
    try:
        import notifications
        notifications.send_email_text(report, subject="Analyst Rating Alert — %d change(s)" % len(notable))
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Track analyst upgrades/downgrades")
    parser.add_argument("--days", type=int, default=7,
                        help="Look-back window in days (default: 7)")
    parser.add_argument("--notify", action="store_true",
                        help="Send alerts via email + Telegram")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers to check")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    changes = scan_all_recommendations(tickers=args.tickers, days=args.days)
    print(format_report(changes))

    if args.notify:
        send_upgrade_alerts(changes)
        print("\nAlerts sent." if changes else "\nNo alerts to send.")


if __name__ == "__main__":
    main()
