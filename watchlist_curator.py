#!/usr/bin/env python3
"""
Watchlist Auto-Curator — automatically suggests additions and removals for the
watchlist based on screener results, sector rotation, and position performance.

Addition criteria:
- Appears in 2+ screener screens (momentum, breakout, oversold, accumulation)
- In a LEADING or IMPROVING sector (from sector rotation)
- Not already on the watchlist

Removal criteria:
- Consecutive losing trades (3+ losses in paper trading)
- In a LAGGING or WEAKENING sector with negative momentum
- No signal activity in 30+ days

Usage:
    python3 watchlist_curator.py                    # Show suggestions
    python3 watchlist_curator.py --apply            # Apply suggestions to watchlist
    python3 watchlist_curator.py --add-only         # Only show addition suggestions
    python3 watchlist_curator.py --remove-only      # Only show removal suggestions
"""

import argparse
import csv
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

WATCHLIST_PATH = Path("watchlist.csv")
MIN_SCREENS_FOR_ADD = int(os.getenv("CURATOR_MIN_SCREENS", "2"))
MAX_CONSECUTIVE_LOSSES = int(os.getenv("CURATOR_MAX_LOSSES", "3"))
STALE_DAYS = int(os.getenv("CURATOR_STALE_DAYS", "30"))


def load_watchlist():
    # type: () -> List[Dict]
    """Load current watchlist."""
    if not WATCHLIST_PATH.exists():
        return []
    with open(WATCHLIST_PATH, newline="") as f:
        return list(csv.DictReader(f))


def save_watchlist(entries):
    # type: (List[Dict]) -> None
    """Save watchlist to CSV."""
    fieldnames = ["ticker", "sector", "notes"]
    with open(WATCHLIST_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(entries)


def get_screener_candidates():
    # type: () -> Dict[str, Dict]
    """Run screener and find tickers that appear in multiple screens.

    Returns dict mapping ticker -> {screens: [...], sector: str, score: float}.
    """
    candidates = {}

    try:
        import screener
        results = screener.run_screen("all", universe="default", top_n=20)

        for screen_name, hits in results.items():
            for hit in hits:
                ticker = hit.get("ticker", "")
                if not ticker:
                    continue
                if ticker not in candidates:
                    candidates[ticker] = {
                        "screens": [],
                        "sector": hit.get("sector", ""),
                        "price": hit.get("price", 0),
                        "rsi": hit.get("rsi", 0),
                        "volume_ratio": hit.get("volume_ratio", 0),
                    }
                candidates[ticker]["screens"].append(screen_name)
    except Exception as e:
        logger.debug("Screener candidates failed: %s", e)

    return candidates


def get_sector_phases():
    # type: () -> Dict[str, str]
    """Get current sector phases from rotation analysis.

    Returns dict mapping sector_name -> phase (LEADING, IMPROVING, etc).
    """
    try:
        import sector_rotation
        # We need data — try to load from a cached rotation result
        history = sector_rotation.get_history(1) if hasattr(sector_rotation, "get_history") else []
        if not history:
            return {}
        return {r.get("sector", ""): r.get("phase", "NEUTRAL") for r in history}
    except Exception:
        return {}


def get_trade_history_by_ticker():
    # type: () -> Dict[str, List[Dict]]
    """Get paper trading history grouped by ticker.

    Returns dict mapping ticker -> list of closed trades (most recent first).
    """
    try:
        import paper_trader
        state = paper_trader.load_state()
        closed = state.get("closed_trades", [])

        by_ticker = {}  # type: Dict[str, List[Dict]]
        for t in closed:
            ticker = t.get("ticker", "")
            if ticker not in by_ticker:
                by_ticker[ticker] = []
            by_ticker[ticker].append(t)

        # Sort each by date descending
        for ticker in by_ticker:
            by_ticker[ticker].sort(
                key=lambda x: x.get("exit_date", ""), reverse=True
            )

        return by_ticker
    except Exception:
        return {}


def get_alert_history_by_ticker():
    # type: () -> Dict[str, str]
    """Get most recent alert date per ticker.

    Returns dict mapping ticker -> last_alert_date_iso.
    """
    try:
        import alert_tracker
        history = alert_tracker.load_history()
        latest = {}  # type: Dict[str, str]
        for entry in history:
            ticker = entry.get("ticker", "")
            date = entry.get("timestamp", entry.get("date", ""))
            if ticker and (ticker not in latest or date > latest[ticker]):
                latest[ticker] = date
        return latest
    except Exception:
        return {}


def suggest_additions(
    watchlist_tickers,  # type: List[str]
    screener_candidates,  # type: Dict[str, Dict]
    sector_phases,      # type: Dict[str, str]
    min_screens=None,   # type: Optional[int]
):
    # type: (...) -> List[Dict]
    """Suggest tickers to add to the watchlist.

    Returns list of suggestion dicts sorted by strength.
    """
    min_s = min_screens if min_screens is not None else MIN_SCREENS_FOR_ADD
    wl_set = set(t.upper() for t in watchlist_tickers)

    suggestions = []
    for ticker, info in screener_candidates.items():
        if ticker.upper() in wl_set:
            continue

        n_screens = len(info["screens"])
        if n_screens < min_s:
            continue

        # Bonus for leading sector
        sector = info.get("sector", "")
        phase = sector_phases.get(sector, "NEUTRAL")
        sector_bonus = 0
        if phase in ("LEADING", "IMPROVING"):
            sector_bonus = 1

        score = n_screens + sector_bonus

        suggestions.append({
            "ticker": ticker,
            "sector": sector,
            "screens": info["screens"],
            "n_screens": n_screens,
            "sector_phase": phase,
            "score": score,
            "reason": "Appears in %d screens: %s%s" % (
                n_screens, ", ".join(info["screens"]),
                " (sector: %s)" % phase if sector_bonus else "",
            ),
        })

    suggestions.sort(key=lambda x: -x["score"])
    return suggestions


def suggest_removals(
    watchlist,          # type: List[Dict]
    trade_history,      # type: Dict[str, List[Dict]]
    alert_dates,        # type: Dict[str, str]
    sector_phases,      # type: Dict[str, str]
    max_losses=None,    # type: Optional[int]
    stale_days=None,    # type: Optional[int]
):
    # type: (...) -> List[Dict]
    """Suggest tickers to remove from the watchlist.

    Returns list of suggestion dicts.
    """
    max_l = max_losses if max_losses is not None else MAX_CONSECUTIVE_LOSSES
    stale = stale_days if stale_days is not None else STALE_DAYS
    cutoff = (datetime.now() - timedelta(days=stale)).isoformat()

    suggestions = []
    for entry in watchlist:
        ticker = entry["ticker"]
        reasons = []

        # Check consecutive losses
        trades = trade_history.get(ticker, [])
        if len(trades) >= max_l:
            consecutive_losses = 0
            for t in trades:
                if t.get("pnl", 0) <= 0:
                    consecutive_losses += 1
                else:
                    break
            if consecutive_losses >= max_l:
                reasons.append("%d consecutive losing trades" % consecutive_losses)

        # Check stale (no alert activity)
        last_alert = alert_dates.get(ticker, "")
        if last_alert and last_alert < cutoff:
            days_since = (datetime.now() - datetime.fromisoformat(last_alert)).days
            reasons.append("No signal in %d days" % days_since)
        elif not last_alert and not trades:
            reasons.append("Never generated a signal")

        # Check sector weakness
        sector = entry.get("sector", "")
        phase = sector_phases.get(sector, "")
        if phase == "LAGGING":
            reasons.append("Sector is LAGGING")

        if reasons:
            suggestions.append({
                "ticker": ticker,
                "sector": sector,
                "reasons": reasons,
                "reason": "; ".join(reasons),
            })

    return suggestions


def curate(min_screens=None, max_losses=None, stale_days=None):
    # type: (Optional[int], Optional[int], Optional[int]) -> Dict
    """Run full curation analysis.

    Returns dict with additions and removals suggestions.
    """
    watchlist = load_watchlist()
    watchlist_tickers = [e["ticker"] for e in watchlist]

    screener_candidates = get_screener_candidates()
    sector_phases = get_sector_phases()
    trade_history = get_trade_history_by_ticker()
    alert_dates = get_alert_history_by_ticker()

    additions = suggest_additions(
        watchlist_tickers, screener_candidates, sector_phases, min_screens
    )
    removals = suggest_removals(
        watchlist, trade_history, alert_dates, sector_phases, max_losses, stale_days
    )

    return {
        "current_count": len(watchlist),
        "additions": additions,
        "removals": removals,
        "screener_scanned": len(screener_candidates),
    }


def apply_suggestions(additions, removals):
    # type: (List[Dict], List[Dict]) -> Tuple[int, int]
    """Apply curation suggestions to the watchlist.

    Returns (added_count, removed_count).
    """
    watchlist = load_watchlist()

    # Remove
    remove_tickers = set(r["ticker"].upper() for r in removals)
    original_len = len(watchlist)
    watchlist = [e for e in watchlist if e["ticker"].upper() not in remove_tickers]
    removed = original_len - len(watchlist)

    # Add
    existing = set(e["ticker"].upper() for e in watchlist)
    added = 0
    for a in additions:
        if a["ticker"].upper() not in existing:
            watchlist.append({
                "ticker": a["ticker"],
                "sector": a.get("sector", ""),
                "notes": "Auto-added: %s" % ", ".join(a.get("screens", [])),
            })
            added += 1
            existing.add(a["ticker"].upper())

    save_watchlist(watchlist)
    return added, removed


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_suggestions(result):
    # type: (Dict) -> None
    """Print curation suggestions."""
    print("\n" + "=" * 60)
    print("  WATCHLIST CURATION")
    print("=" * 60)
    print("  Current watchlist: %d tickers" % result["current_count"])
    print("  Screener scanned:  %d candidates" % result["screener_scanned"])

    additions = result["additions"]
    if additions:
        print("\n  SUGGESTED ADDITIONS (%d)" % len(additions))
        print("  " + "-" * 50)
        for a in additions:
            print("  + %-6s  %-20s  %s" % (a["ticker"], a.get("sector", ""), a["reason"]))
    else:
        print("\n  No additions suggested.")

    removals = result["removals"]
    if removals:
        print("\n  SUGGESTED REMOVALS (%d)" % len(removals))
        print("  " + "-" * 50)
        for r in removals:
            print("  - %-6s  %-20s  %s" % (r["ticker"], r.get("sector", ""), r["reason"]))
    else:
        print("\n  No removals suggested.")

    print("\n" + "=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Watchlist Auto-Curator")
    parser.add_argument("--apply", action="store_true", help="Apply suggestions")
    parser.add_argument("--add-only", action="store_true", help="Only show additions")
    parser.add_argument("--remove-only", action="store_true", help="Only show removals")
    parser.add_argument("--min-screens", type=int, default=None,
                        help="Min screens for addition (default %d)" % MIN_SCREENS_FOR_ADD)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
    for lib in ("urllib3", "yfinance"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    result = curate(min_screens=args.min_screens)

    if args.add_only:
        result["removals"] = []
    if args.remove_only:
        result["additions"] = []

    print_suggestions(result)

    if args.apply:
        added, removed = apply_suggestions(result["additions"], result["removals"])
        print("  Applied: +%d added, -%d removed\n" % (added, removed))


if __name__ == "__main__":
    main()
