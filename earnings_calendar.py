#!/usr/bin/env python3
"""
Earnings Calendar — fetches upcoming earnings dates for all tickers in
portfolio.json using yfinance and provides alerting for near-term earnings.

Usage:
    python3 earnings_calendar.py              # Print report (default 7 days)
    python3 earnings_calendar.py --days 14    # Look ahead N days
    python3 earnings_calendar.py --json       # Output as JSON
    python3 earnings_calendar.py --send       # Send via email/SMS
"""

import argparse
import json
import logging
import os
from datetime import date, datetime
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

PORTFOLIO_FILE = Path(__file__).parent / "portfolio.json"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def _parse_earnings_date(raw) -> Optional[date]:
    """Coerce yfinance calendar date value to a Python date.

    The calendar dict value for 'Earnings Date' may be:
      - a list of Timestamp / datetime / date objects
      - a single Timestamp / datetime / date object
      - a string like "2026-04-15"
      - None / missing
    We always take the *first* (nearest) date when a list is returned.
    """
    if raw is None:
        return None

    # If it's a list, grab the first element
    if isinstance(raw, list):
        if not raw:
            return None
        raw = raw[0]

    # pandas Timestamp / datetime — both have .date()
    if hasattr(raw, "date"):
        try:
            return raw.date()
        except Exception:
            pass

    # Already a date
    if isinstance(raw, date):
        return raw

    # String fallback
    if isinstance(raw, str):
        for fmt in ("%Y-%m-%d", "%Y-%m-%dT%H:%M:%S", "%m/%d/%Y"):
            try:
                return datetime.strptime(raw[:10], fmt).date()
            except ValueError:
                continue

    return None


def _parse_estimate_eps(calendar: dict) -> Optional[float]:
    """Extract EPS estimate from the calendar dict."""
    for key in ("EPS Estimate", "Earnings Estimate", "epsEstimate"):
        val = calendar.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    return None


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def get_earnings_dates(tickers: List[str]) -> Dict[str, dict]:
    """Fetch upcoming earnings dates for a list of tickers.

    Returns a dict keyed by ticker:
        {
            "AAPL": {
                "date": "2026-04-15",
                "days_away": 5,
                "estimate_eps": 2.34,
            }
        }

    Tickers with no earnings data or past earnings dates are omitted.
    """
    import yfinance as yf

    today = date.today()
    results: Dict[str, dict] = {}

    for ticker in tickers:
        try:
            t = yf.Ticker(ticker)
            cal = t.calendar

            if not cal:
                logger.debug("%s: no calendar data", ticker)
                continue

            # calendar can be a dict or a DataFrame (older yfinance versions)
            if hasattr(cal, "to_dict"):
                # DataFrame — transpose so columns become keys
                cal = cal.T.to_dict().get(0, {})

            earnings_raw = cal.get("Earnings Date")
            earnings_dt = _parse_earnings_date(earnings_raw)

            if earnings_dt is None:
                logger.debug("%s: could not parse earnings date from %r", ticker, earnings_raw)
                continue

            days_away = (earnings_dt - today).days

            results[ticker] = {
                "date": earnings_dt.isoformat(),
                "days_away": days_away,
                "estimate_eps": _parse_estimate_eps(cal),
            }

        except Exception as e:
            logger.warning("%s: error fetching calendar — %s", ticker, e)

    return results


def get_portfolio_earnings() -> List[dict]:
    """Load portfolio.json holdings and return earnings info sorted by date.

    Each item in the returned list is the holding dict merged with earnings
    data:
        {
            "ticker": "AAPL",
            "shares": 10,
            "strategy": "trade",
            ...holding fields...,
            "date": "2026-04-15",
            "days_away": 5,
            "estimate_eps": 2.34,
        }

    Holdings with no upcoming earnings data are included with null date fields
    so callers have a complete picture.
    """
    if not PORTFOLIO_FILE.exists():
        logger.error("portfolio.json not found at %s", PORTFOLIO_FILE)
        return []

    with open(PORTFOLIO_FILE) as f:
        portfolio = json.load(f)

    holdings = portfolio.get("holdings", [])
    tickers = [h["ticker"] for h in holdings]

    earnings_map = get_earnings_dates(tickers)

    enriched: List[dict] = []
    for holding in holdings:
        ticker = holding["ticker"]
        info = earnings_map.get(ticker, {})
        row = dict(holding)
        row["date"] = info.get("date")
        row["days_away"] = info.get("days_away")
        row["estimate_eps"] = info.get("estimate_eps")
        enriched.append(row)

    # Sort: items with a date first (ascending), then None dates at the end
    def _sort_key(item):
        d = item.get("days_away")
        if d is None:
            return (1, 0)
        return (0, d)

    enriched.sort(key=_sort_key)
    return enriched


def check_earnings_warnings(days_ahead: int = 7) -> List[dict]:
    """Return holdings whose earnings are within *days_ahead* calendar days.

    Includes earnings happening today (days_away == 0) and future earnings
    up to *days_ahead* days out. Past earnings (negative days_away) are
    excluded.
    """
    all_earnings = get_portfolio_earnings()
    warnings = [
        item for item in all_earnings
        if item.get("days_away") is not None and 0 <= item["days_away"] <= days_ahead
    ]
    return warnings


def format_earnings_report(earnings: List[dict]) -> str:
    """Return a human-readable earnings report string."""
    if not earnings:
        return "No upcoming earnings data found for portfolio holdings.\n"

    lines = [
        "=" * 60,
        "  PORTFOLIO EARNINGS CALENDAR",
        "=" * 60,
    ]

    for item in earnings:
        ticker = item["ticker"]
        d = item.get("date")
        days_away = item.get("days_away")
        eps = item.get("estimate_eps")
        strategy = item.get("strategy", "—")
        shares = item.get("shares", 0)
        pnl = item.get("unrealized_pnl", 0.0)
        pnl_pct = item.get("pnl_pct", 0.0)

        if d is None:
            date_str = "No date available"
            days_str = ""
        elif days_away == 0:
            date_str = d
            days_str = "  *** TODAY ***"
        elif days_away == 1:
            date_str = d
            days_str = "  (tomorrow)"
        else:
            date_str = d
            days_str = "  (%d days)" % days_away

        eps_str = "EPS est: $%.2f" % eps if eps is not None else "EPS est: N/A"
        pnl_sign = "+" if pnl >= 0 else ""
        pnl_str = "%s$%.0f (%s%.1f%%)" % (pnl_sign, pnl, pnl_sign, pnl_pct)

        lines.append(
            "  %-6s  %s%s" % (ticker, date_str, days_str)
        )
        lines.append(
            "         Shares: %d | Strategy: %-10s | P&L: %-22s | %s"
            % (shares, strategy, pnl_str, eps_str)
        )
        lines.append("  " + "─" * 56)

    lines.append("=" * 60)
    lines.append("  Generated: %s" % datetime.now().strftime("%Y-%m-%d %H:%M"))
    lines.append("=" * 60)
    return "\n".join(lines) + "\n"


def send_earnings_alert(warnings: List[dict]) -> None:
    """Send earnings warning via email and SMS using notifications module."""
    import notifications

    if not warnings:
        logger.info("No earnings warnings to send")
        return

    report = format_earnings_report(warnings)
    subject = "Earnings Alert: %d holding(s) report soon" % len(warnings)

    # Build a compact SMS version (≤160 chars per holding)
    sms_lines = ["Earnings alert:"]
    for item in warnings:
        d = item.get("days_away")
        day_label = "today" if d == 0 else ("tomorrow" if d == 1 else "in %d days" % d)
        sms_lines.append("%s reports %s" % (item["ticker"], day_label))
    sms_text = " | ".join(sms_lines)

    notifications.send_email_text(report, subject=subject)
    # SMS disabled — email queued for daily digest instead

    # Telegram
    try:
        import telegram_bot
        telegram_bot.send_message(report)
        logger.info("Earnings alert sent to Telegram")
    except Exception as e:
        logger.debug("Telegram earnings alert failed: %s", e)

    logger.info("Earnings alerts sent for %d holdings", len(warnings))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Portfolio earnings calendar")
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Look-ahead window in days for warnings (default: 7)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="as_json",
        help="Output as JSON",
    )
    parser.add_argument(
        "--send",
        action="store_true",
        help="Send alerts via email/SMS for holdings within --days",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        dest="show_all",
        help="Show all holdings (not just upcoming within --days)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if args.show_all:
        earnings = get_portfolio_earnings()
    else:
        earnings = check_earnings_warnings(days_ahead=args.days)

    if args.as_json:
        print(json.dumps(earnings, indent=2, default=str))
        return

    report = format_earnings_report(earnings)
    print(report)

    if args.send:
        warnings = check_earnings_warnings(days_ahead=args.days)
        if warnings:
            send_earnings_alert(warnings)
            print("Alerts sent for %d holding(s)." % len(warnings))
        else:
            print("No earnings within %d days — nothing to send." % args.days)


if __name__ == "__main__":
    main()
