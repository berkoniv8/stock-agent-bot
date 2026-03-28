#!/usr/bin/env python3
"""
Trade Journal — structured logging of trade decisions with notes, tags,
screenshots, and post-trade review.

Stores entries as JSON in logs/trade_journal.json (or SQLite when enabled).
Each entry captures the full context: why the trade was taken, what signals
fired, the emotional state, and post-exit review.

Usage:
    python3 trade_journal.py --list                     # Show recent entries
    python3 trade_journal.py --add AAPL --note "..."    # Add a note to latest AAPL entry
    python3 trade_journal.py --review AAPL              # Add post-trade review
    python3 trade_journal.py --tags momentum breakout    # Filter by tags
    python3 trade_journal.py --stats                    # Show journal statistics
    python3 trade_journal.py --export journal.csv       # Export to CSV
"""

import argparse
import csv
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

JOURNAL_PATH = os.getenv("JOURNAL_PATH", "logs/trade_journal.json")

# Pre-defined tags for quick classification
VALID_TAGS = {
    "momentum", "breakout", "reversal", "mean-reversion", "earnings-play",
    "sector-rotation", "trend-follow", "scalp", "swing", "position",
    "high-conviction", "speculative", "hedged", "oversold-bounce",
    "gap-fill", "accumulation", "distribution",
}


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def _load_journal():
    # type: () -> List[Dict]
    """Load journal entries from disk."""
    if not os.path.exists(JOURNAL_PATH):
        return []
    try:
        with open(JOURNAL_PATH) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def _save_journal(entries):
    # type: (List[Dict]) -> None
    """Save journal entries to disk."""
    os.makedirs(os.path.dirname(JOURNAL_PATH) or ".", exist_ok=True)
    with open(JOURNAL_PATH, "w") as f:
        json.dump(entries, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------

def create_entry(
    ticker,             # type: str
    direction,          # type: str
    entry_price,        # type: float
    shares,             # type: int
    stop_loss,          # type: float
    target_1,           # type: float
    signal_score,       # type: int
    triggered_signals,  # type: List[str]
    setup_note="",      # type: str
    tags=None,          # type: Optional[List[str]]
    sector="",          # type: str
):
    # type: (...) -> Dict
    """Create a new journal entry when a trade is opened.

    Args:
        ticker: Stock symbol.
        direction: "BUY" or "SELL".
        entry_price: Entry price.
        shares: Number of shares.
        stop_loss: Initial stop-loss price.
        target_1: First take-profit target.
        signal_score: Confluence score at entry.
        triggered_signals: List of signal names that triggered entry.
        setup_note: Free-text note about the trade setup.
        tags: List of classification tags.
        sector: Stock sector.

    Returns:
        The created journal entry dict.
    """
    entry_id = "%s_%s" % (ticker, datetime.now().strftime("%Y%m%d_%H%M%S"))

    entry = {
        "id": entry_id,
        "ticker": ticker,
        "sector": sector,
        "direction": direction,
        "entry_price": entry_price,
        "shares": shares,
        "stop_loss": stop_loss,
        "target_1": target_1,
        "signal_score": signal_score,
        "triggered_signals": triggered_signals,
        "tags": tags or [],
        "setup_note": setup_note,
        "entry_date": datetime.now().isoformat(),
        "exit_date": None,
        "exit_price": None,
        "exit_reason": None,
        "pnl": None,
        "pnl_pct": None,
        # Post-trade review
        "review_note": None,
        "review_rating": None,  # 1-5 self-assessment
        "lessons": None,
        "mistakes": [],
        "what_went_well": [],
        # Metadata
        "notes": [setup_note] if setup_note else [],
        "created_at": datetime.now().isoformat(),
        "updated_at": datetime.now().isoformat(),
    }

    entries = _load_journal()
    entries.append(entry)
    _save_journal(entries)

    logger.info("JOURNAL — Created entry %s for %s %s @ $%.2f",
                entry_id, direction, ticker, entry_price)

    return entry


def close_entry(
    ticker,         # type: str
    exit_price,     # type: float
    exit_reason,    # type: str
    pnl,            # type: float
    pnl_pct,        # type: float
):
    # type: (...) -> Optional[Dict]
    """Close the most recent open journal entry for a ticker.

    Args:
        ticker: Stock symbol.
        exit_price: Exit price.
        exit_reason: Why the trade was closed.
        pnl: Profit/loss in dollars.
        pnl_pct: Profit/loss as percentage.

    Returns:
        The updated entry, or None if not found.
    """
    entries = _load_journal()

    # Find the most recent open entry for this ticker
    target = None
    for entry in reversed(entries):
        if entry["ticker"] == ticker and entry["exit_date"] is None:
            target = entry
            break

    if target is None:
        logger.warning("JOURNAL — No open entry found for %s", ticker)
        return None

    target["exit_date"] = datetime.now().isoformat()
    target["exit_price"] = exit_price
    target["exit_reason"] = exit_reason
    target["pnl"] = round(pnl, 2)
    target["pnl_pct"] = round(pnl_pct, 2)
    target["updated_at"] = datetime.now().isoformat()

    _save_journal(entries)

    emoji = "+" if pnl >= 0 else ""
    logger.info("JOURNAL — Closed %s: %s$%.2f (%.1f%%) — %s",
                ticker, emoji, pnl, pnl_pct, exit_reason)

    return target


def add_note(ticker, note):
    # type: (str, str) -> Optional[Dict]
    """Add a note to the most recent entry for a ticker.

    Returns the updated entry, or None if not found.
    """
    entries = _load_journal()

    target = None
    for entry in reversed(entries):
        if entry["ticker"] == ticker:
            target = entry
            break

    if target is None:
        return None

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    target["notes"].append("[%s] %s" % (timestamp, note))
    target["updated_at"] = datetime.now().isoformat()
    _save_journal(entries)

    return target


def add_review(
    ticker,         # type: str
    review_note,    # type: str
    rating=None,    # type: Optional[int]
    lessons=None,   # type: Optional[str]
    mistakes=None,  # type: Optional[List[str]]
    what_went_well=None,  # type: Optional[List[str]]
):
    # type: (...) -> Optional[Dict]
    """Add a post-trade review to the most recent closed entry.

    Args:
        ticker: Stock symbol.
        review_note: Free-text review.
        rating: Self-assessment 1-5 (1=terrible, 5=perfect execution).
        lessons: Key lesson learned.
        mistakes: List of mistakes made.
        what_went_well: List of things done right.

    Returns:
        The updated entry, or None.
    """
    entries = _load_journal()

    target = None
    for entry in reversed(entries):
        if entry["ticker"] == ticker and entry["exit_date"] is not None:
            target = entry
            break

    if target is None:
        logger.warning("JOURNAL — No closed entry found for %s to review", ticker)
        return None

    target["review_note"] = review_note
    if rating is not None:
        target["review_rating"] = max(1, min(5, rating))
    if lessons:
        target["lessons"] = lessons
    if mistakes:
        target["mistakes"] = mistakes
    if what_went_well:
        target["what_went_well"] = what_went_well
    target["updated_at"] = datetime.now().isoformat()

    _save_journal(entries)
    logger.info("JOURNAL — Review added for %s (rating: %s)", ticker, rating)

    return target


def add_tags(ticker, tags):
    # type: (str, List[str]) -> Optional[Dict]
    """Add tags to the most recent entry for a ticker."""
    entries = _load_journal()

    target = None
    for entry in reversed(entries):
        if entry["ticker"] == ticker:
            target = entry
            break

    if target is None:
        return None

    existing = set(target.get("tags", []))
    existing.update(tags)
    target["tags"] = sorted(existing)
    target["updated_at"] = datetime.now().isoformat()
    _save_journal(entries)

    return target


# ---------------------------------------------------------------------------
# Queries
# ---------------------------------------------------------------------------

def get_entries(
    ticker=None,    # type: Optional[str]
    tags=None,      # type: Optional[List[str]]
    open_only=False,  # type: bool
    closed_only=False,  # type: bool
    limit=50,       # type: int
):
    # type: (...) -> List[Dict]
    """Query journal entries with optional filters.

    Args:
        ticker: Filter by ticker symbol.
        tags: Filter by tags (entries must have ALL specified tags).
        open_only: Only return entries without exit_date.
        closed_only: Only return entries with exit_date.
        limit: Max entries to return.

    Returns:
        List of matching entries, most recent first.
    """
    entries = _load_journal()

    if ticker:
        entries = [e for e in entries if e["ticker"] == ticker.upper()]
    if open_only:
        entries = [e for e in entries if e.get("exit_date") is None]
    if closed_only:
        entries = [e for e in entries if e.get("exit_date") is not None]
    if tags:
        tag_set = set(tags)
        entries = [e for e in entries if tag_set.issubset(set(e.get("tags", [])))]

    # Most recent first
    entries = list(reversed(entries))
    return entries[:limit]


def get_stats():
    # type: () -> Dict
    """Compute journal-level statistics.

    Returns dict with counts, win rate, avg rating, common tags, etc.
    """
    entries = _load_journal()
    if not entries:
        return {"total_entries": 0}

    closed = [e for e in entries if e.get("exit_date") is not None]
    open_entries = [e for e in entries if e.get("exit_date") is None]

    wins = [e for e in closed if (e.get("pnl") or 0) > 0]
    losses = [e for e in closed if (e.get("pnl") or 0) <= 0]

    # Tag frequency
    tag_counts = {}
    for e in entries:
        for t in e.get("tags", []):
            tag_counts[t] = tag_counts.get(t, 0) + 1

    # Review ratings
    ratings = [e["review_rating"] for e in closed if e.get("review_rating")]

    # Win rate by tag
    tag_performance = {}
    for e in closed:
        pnl = e.get("pnl") or 0
        for t in e.get("tags", []):
            if t not in tag_performance:
                tag_performance[t] = {"wins": 0, "losses": 0, "total_pnl": 0}
            if pnl > 0:
                tag_performance[t]["wins"] += 1
            else:
                tag_performance[t]["losses"] += 1
            tag_performance[t]["total_pnl"] += pnl

    # Common mistakes
    all_mistakes = []
    for e in closed:
        all_mistakes.extend(e.get("mistakes", []))

    return {
        "total_entries": len(entries),
        "open_trades": len(open_entries),
        "closed_trades": len(closed),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": round(len(wins) / len(closed) * 100, 1) if closed else 0,
        "total_pnl": round(sum(e.get("pnl") or 0 for e in closed), 2),
        "avg_pnl": round(sum(e.get("pnl") or 0 for e in closed) / len(closed), 2) if closed else 0,
        "avg_rating": round(sum(ratings) / len(ratings), 1) if ratings else 0,
        "reviewed_count": len(ratings),
        "unreviewed_count": len(closed) - len(ratings),
        "tag_counts": dict(sorted(tag_counts.items(), key=lambda x: -x[1])),
        "tag_performance": tag_performance,
        "common_mistakes": all_mistakes,
    }


def export_csv(filepath):
    # type: (str) -> int
    """Export journal entries to CSV.

    Returns the number of rows exported.
    """
    entries = _load_journal()
    if not entries:
        return 0

    fields = [
        "id", "ticker", "sector", "direction", "entry_date", "entry_price",
        "shares", "stop_loss", "target_1", "signal_score", "exit_date",
        "exit_price", "exit_reason", "pnl", "pnl_pct", "tags",
        "setup_note", "review_note", "review_rating", "lessons",
    ]

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for e in entries:
            row = dict(e)
            row["tags"] = ",".join(e.get("tags", []))
            row["setup_note"] = (e.get("notes", []) or [""])[0] if e.get("notes") else ""
            writer.writerow(row)

    return len(entries)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_entries(entries):
    # type: (List[Dict]) -> None
    """Print journal entries in a readable format."""
    if not entries:
        print("\n  No journal entries found.\n")
        return

    for e in entries:
        status = "OPEN" if e.get("exit_date") is None else "CLOSED"
        pnl_str = ""
        if e.get("pnl") is not None:
            sign = "+" if e["pnl"] >= 0 else ""
            pnl_str = " | P&L: %s$%.2f (%.1f%%)" % (sign, e["pnl"], e.get("pnl_pct", 0))

        tags_str = " [%s]" % ", ".join(e.get("tags", [])) if e.get("tags") else ""

        print("\n  %s %s %s @ $%.2f  (%s)%s%s" % (
            e["direction"], e["ticker"], e.get("entry_date", "")[:10],
            e["entry_price"], status, pnl_str, tags_str,
        ))
        print("    Score: %d | Signals: %s" % (
            e["signal_score"], ", ".join(e.get("triggered_signals", [])[:5]),
        ))

        if e.get("notes"):
            for n in e["notes"][-3:]:
                print("    Note: %s" % n)

        if e.get("review_note"):
            rating_str = " (%d/5)" % e["review_rating"] if e.get("review_rating") else ""
            print("    Review%s: %s" % (rating_str, e["review_note"]))
        elif status == "CLOSED":
            print("    (No review yet — run --review %s)" % e["ticker"])

    print()


def print_stats(stats):
    # type: (Dict) -> None
    """Print journal statistics."""
    if stats.get("total_entries", 0) == 0:
        print("\n  No journal entries yet.\n")
        return

    print("\n  TRADE JOURNAL STATISTICS")
    print("  " + "=" * 50)
    print("  Total Entries:    %d" % stats["total_entries"])
    print("  Open Trades:      %d" % stats["open_trades"])
    print("  Closed Trades:    %d" % stats["closed_trades"])
    print("  Win Rate:         %.1f%%" % stats["win_rate"])
    print("  Total P&L:        $%.2f" % stats["total_pnl"])
    print("  Avg P&L:          $%.2f" % stats["avg_pnl"])
    print("  Avg Review:       %.1f/5" % stats["avg_rating"])
    print("  Reviewed:         %d/%d" % (stats["reviewed_count"], stats["closed_trades"]))

    if stats.get("tag_counts"):
        print("\n  TOP TAGS")
        print("  " + "-" * 40)
        for tag, count in list(stats["tag_counts"].items())[:10]:
            print("    %-20s %d" % (tag, count))

    if stats.get("tag_performance"):
        print("\n  PERFORMANCE BY TAG")
        print("  " + "-" * 40)
        for tag, perf in stats["tag_performance"].items():
            total = perf["wins"] + perf["losses"]
            wr = perf["wins"] / total * 100 if total > 0 else 0
            print("    %-18s  WR: %5.1f%%  P&L: $%.2f" % (tag, wr, perf["total_pnl"]))

    if stats.get("common_mistakes"):
        print("\n  COMMON MISTAKES")
        print("  " + "-" * 40)
        for m in stats["common_mistakes"][:5]:
            print("    - %s" % m)

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Trade Journal")
    parser.add_argument("--list", action="store_true", help="List recent entries")
    parser.add_argument("--ticker", type=str, help="Filter by ticker")
    parser.add_argument("--add", type=str, metavar="TICKER", help="Add note to a ticker")
    parser.add_argument("--note", type=str, help="Note text (used with --add)")
    parser.add_argument("--review", type=str, metavar="TICKER", help="Add review to closed trade")
    parser.add_argument("--review-note", type=str, help="Review text")
    parser.add_argument("--rating", type=int, help="Rating 1-5")
    parser.add_argument("--tags", nargs="+", help="Filter by or add tags")
    parser.add_argument("--add-tags", type=str, metavar="TICKER", help="Add tags to a ticker")
    parser.add_argument("--stats", action="store_true", help="Show journal statistics")
    parser.add_argument("--export", type=str, metavar="FILE", help="Export to CSV")
    parser.add_argument("--open", action="store_true", help="Show only open entries")
    parser.add_argument("--closed", action="store_true", help="Show only closed entries")
    parser.add_argument("--limit", type=int, default=20, help="Max entries to show")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")

    if args.stats:
        stats = get_stats()
        print_stats(stats)
        return

    if args.export:
        count = export_csv(args.export)
        print("\n  Exported %d entries to %s\n" % (count, args.export))
        return

    if args.add and args.note:
        entry = add_note(args.add.upper(), args.note)
        if entry:
            print("\n  Note added to %s\n" % args.add.upper())
        else:
            print("\n  No entry found for %s\n" % args.add.upper())
        return

    if args.add_tags and args.tags:
        entry = add_tags(args.add_tags.upper(), args.tags)
        if entry:
            print("\n  Tags added to %s: %s\n" % (args.add_tags.upper(), ", ".join(args.tags)))
        else:
            print("\n  No entry found for %s\n" % args.add_tags.upper())
        return

    if args.review:
        note = args.review_note or ""
        entry = add_review(
            args.review.upper(), note,
            rating=args.rating,
        )
        if entry:
            print("\n  Review added to %s\n" % args.review.upper())
        else:
            print("\n  No closed entry found for %s\n" % args.review.upper())
        return

    # Default: list entries
    entries = get_entries(
        ticker=args.ticker,
        tags=args.tags,
        open_only=args.open,
        closed_only=args.closed,
        limit=args.limit,
    )
    print_entries(entries)


if __name__ == "__main__":
    main()
