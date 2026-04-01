#!/usr/bin/env python3
"""
News Monitor — fetches and filters news for portfolio holdings using yfinance.

Sends alerts for significant news events (earnings, M&A, regulatory, etc.)
and deduplicates via a local sent-cache to avoid repeated notifications.

Usage:
    python3 news_monitor.py               # Print significant news (last 24h)
    python3 news_monitor.py --all         # Show all news, not just significant
    python3 news_monitor.py --hours 48    # Look back 48 hours instead of 24
"""

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Set

import yfinance as yf
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)
NEWS_CACHE_FILE = LOGS_DIR / "news_sent.json"

PORTFOLIO_FILE = Path("portfolio.json")

# Keywords that signal a potentially market-moving news item
SIGNIFICANT_KEYWORDS = [
    "earnings", "revenue", "beat", "miss", "lawsuit", "fda",
    "merger", "acquisition", "downgrade", "upgrade", "recall",
    "investigation", "guidance", "layoff", "ceo",
]


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

def load_sent_cache() -> Set[str]:
    """Load set of already-sent article URLs from logs/news_sent.json."""
    if not NEWS_CACHE_FILE.exists():
        return set()
    try:
        with open(NEWS_CACHE_FILE, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return set(data)
        return set()
    except Exception as e:
        logger.warning("Could not load news cache: %s", e)
        return set()


def save_sent_cache(cache: Set[str]) -> None:
    """Persist sent article URL cache to logs/news_sent.json."""
    try:
        with open(NEWS_CACHE_FILE, "w") as f:
            json.dump(sorted(cache), f, indent=2)
    except Exception as e:
        logger.warning("Could not save news cache: %s", e)


# ---------------------------------------------------------------------------
# Fetching
# ---------------------------------------------------------------------------

def fetch_news(ticker: str, max_items: int = 5) -> List[dict]:
    """Fetch recent news for a single ticker via yfinance.

    Returns list of dicts with keys:
        ticker, title, publisher, url, published_at (ISO string), age_hours (float)
    """
    results = []
    try:
        tk = yf.Ticker(ticker)
        raw_news = tk.news or []
        now_ts = datetime.now(timezone.utc).timestamp()

        for item in raw_news[:max_items]:
            try:
                pub_ts = item.get("providerPublishTime", 0)
                age_hours = (now_ts - pub_ts) / 3600.0 if pub_ts else 0.0

                pub_dt = (
                    datetime.fromtimestamp(pub_ts, tz=timezone.utc).isoformat()
                    if pub_ts
                    else ""
                )

                # yfinance >=0.2.x wraps content under a nested dict in some versions
                content = item.get("content", item)
                title = (
                    content.get("title", "")
                    if isinstance(content, dict)
                    else item.get("title", "")
                )
                publisher = (
                    content.get("provider", {}).get("displayName", "")
                    if isinstance(content, dict)
                    else item.get("publisher", "")
                )
                url = item.get("link", item.get("url", ""))
                if not url and isinstance(content, dict):
                    url = content.get("canonicalUrl", {}).get("url", "")

                if not title:
                    continue

                results.append({
                    "ticker": ticker,
                    "title": title,
                    "publisher": publisher,
                    "url": url,
                    "published_at": pub_dt,
                    "age_hours": round(age_hours, 2),
                })
            except Exception as e:
                logger.debug("Skipping malformed news item for %s: %s", ticker, e)
                continue

    except Exception as e:
        logger.error("Failed to fetch news for %s: %s", ticker, e)

    return results


def fetch_portfolio_news(max_age_hours: float = 24.0) -> List[dict]:
    """Fetch news for all current portfolio holdings.

    Loads portfolio.json, fetches news for every holding, filters to items
    published within the last `max_age_hours`, and returns them sorted
    newest-first.
    """
    if not PORTFOLIO_FILE.exists():
        logger.error("portfolio.json not found")
        return []

    try:
        with open(PORTFOLIO_FILE, "r") as f:
            portfolio = json.load(f)
    except Exception as e:
        logger.error("Failed to load portfolio.json: %s", e)
        return []

    holdings = portfolio.get("holdings", [])
    tickers = [h["ticker"] for h in holdings if "ticker" in h]

    all_news = []
    for ticker in tickers:
        items = fetch_news(ticker, max_items=10)
        for item in items:
            if item["age_hours"] <= max_age_hours:
                all_news.append(item)

    # Sort newest first (smallest age_hours = most recent)
    all_news.sort(key=lambda x: x["age_hours"])
    return all_news


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def filter_significant(news_items: List[dict]) -> List[dict]:
    """Return only items whose title contains a significant keyword.

    Matching is case-insensitive.
    """
    significant = []
    for item in news_items:
        title_lower = item.get("title", "").lower()
        if any(kw in title_lower for kw in SIGNIFICANT_KEYWORDS):
            significant.append(item)
    return significant


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def format_news_report(items: List[dict]) -> str:
    """Build a readable news report grouped by ticker."""
    if not items:
        return "No news items to display.\n"

    # Group by ticker
    by_ticker = {}  # type: dict
    for item in items:
        ticker = item["ticker"]
        by_ticker.setdefault(ticker, []).append(item)

    lines = []
    lines.append("=" * 70)
    lines.append("  PORTFOLIO NEWS MONITOR")
    lines.append("  Generated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    lines.append("=" * 70)

    for ticker in sorted(by_ticker.keys()):
        ticker_items = by_ticker[ticker]
        lines.append(f"\n  {ticker}  ({len(ticker_items)} item{'s' if len(ticker_items) != 1 else ''})")
        lines.append("  " + "-" * 60)
        for item in ticker_items:
            age = item["age_hours"]
            if age < 1:
                age_str = f"{int(age * 60)}m ago"
            elif age < 24:
                age_str = f"{age:.1f}h ago"
            else:
                age_str = f"{age / 24:.1f}d ago"

            lines.append(f"  [{age_str}] {item['title']}")
            publisher = item.get("publisher", "")
            if publisher:
                lines.append(f"           {publisher}")
            url = item.get("url", "")
            if url:
                lines.append(f"           {url}")

    lines.append("\n" + "=" * 70)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Alerts
# ---------------------------------------------------------------------------

def send_news_alerts(items: List[dict]) -> None:
    """Send news alerts via email and SMS (SMS only for significant items, max 3).

    Skips articles already in the sent cache and updates the cache afterwards.
    """
    import notifications

    if not items:
        return

    cache = load_sent_cache()
    new_items = [i for i in items if i.get("url") not in cache]

    if not new_items:
        logger.info("All news items already sent — nothing to alert")
        return

    # Build full email report
    report_text = format_news_report(new_items)
    subject = f"Portfolio News Alert — {len(new_items)} new item{'s' if len(new_items) != 1 else ''}"

    email_sent = notifications.send_email_text(report_text, subject=subject)
    if email_sent:
        logger.info("News email sent: %s items", len(new_items))

    # SMS disabled — news alerts go via Telegram and daily digest email

    # Telegram
    try:
        import telegram_bot
        telegram_bot.send_message(report_text)
        logger.info("News alert sent to Telegram")
    except Exception as e:
        logger.debug("Telegram news alert failed: %s", e)

    # Update cache with all items we just processed
    for item in new_items:
        url = item.get("url", "")
        if url:
            cache.add(url)

    save_sent_cache(cache)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Portfolio News Monitor")
    parser.add_argument(
        "--all",
        action="store_true",
        help="Show all recent news, not just significant items",
    )
    parser.add_argument(
        "--hours",
        type=float,
        default=24.0,
        metavar="N",
        help="Look-back window in hours (default: 24)",
    )
    parser.add_argument(
        "--send",
        action="store_true",
        help="Send alerts via configured notification channels",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
    )
    for lib in ("urllib3", "yfinance", "peewee"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    news = fetch_portfolio_news(max_age_hours=args.hours)

    if not args.all:
        news = filter_significant(news)

    print(format_news_report(news))

    if args.send:
        send_news_alerts(news)


if __name__ == "__main__":
    main()
