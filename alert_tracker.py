"""
Alert Tracker — deduplication and cooldown logic.

Prevents the same signal from firing repeatedly within a configurable window.
Supports both JSON file and SQLite backends (SQLite preferred when available).
"""

import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

HISTORY_FILE = Path("logs/alert_history.json")
COOLDOWN_HOURS = int(os.getenv("ALERT_COOLDOWN_HOURS", "24"))
USE_DB = os.getenv("USE_SQLITE", "0") == "1"


def _load_history() -> dict:
    """Load alert history from disk."""
    if not HISTORY_FILE.exists():
        return {}
    try:
        with open(HISTORY_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return {}


def _save_history(history: dict) -> None:
    """Persist alert history to disk."""
    HISTORY_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=2)


def _make_key(ticker: str, direction: str, signals: list) -> str:
    """Create a deduplication key from ticker + direction + signal combo."""
    sig_names = sorted(s[0] if isinstance(s, (list, tuple)) else str(s) for s in signals)
    return f"{ticker}:{direction}:{','.join(sig_names)}"


def is_duplicate(ticker: str, direction: str, signals: list) -> bool:
    """Check if an alert with the same ticker/direction/signals was sent
    within the cooldown window.

    Returns True if this is a duplicate (should be suppressed).
    """
    key = _make_key(ticker, direction, signals)

    if USE_DB:
        try:
            import database as db
            return db.check_alert_duplicate(key, COOLDOWN_HOURS)
        except Exception as e:
            logger.debug("DB fallback for is_duplicate: %s", e)

    # File-based fallback
    history = _load_history()
    last_fired = history.get(key)

    if last_fired is None:
        return False

    try:
        last_dt = datetime.fromisoformat(last_fired)
        cutoff = datetime.now() - timedelta(hours=COOLDOWN_HOURS)
        if last_dt > cutoff:
            logger.info(
                "Suppressing duplicate alert for %s %s (last fired %s, cooldown %dh)",
                direction, ticker, last_fired, COOLDOWN_HOURS,
            )
            return True
    except ValueError:
        pass

    return False


def record_alert(ticker: str, direction: str, signals: list) -> None:
    """Record that an alert was sent, for future dedup checks."""
    key = _make_key(ticker, direction, signals)

    if USE_DB:
        try:
            import database as db
            db.record_alert_dedup(key)
            db.prune_alert_history(7)
            logger.info("Recorded alert (db): %s", key)
            return
        except Exception as e:
            logger.debug("DB fallback for record_alert: %s", e)

    # File-based fallback
    history = _load_history()
    history[key] = datetime.now().isoformat()

    # Prune entries older than 7 days
    cutoff = datetime.now() - timedelta(days=7)
    pruned = {}
    for k, v in history.items():
        try:
            if datetime.fromisoformat(v) > cutoff:
                pruned[k] = v
        except ValueError:
            pass

    _save_history(pruned)
    logger.info("Recorded alert: %s", key)


def clear_history() -> None:
    """Clear all alert history (useful for testing)."""
    if USE_DB:
        try:
            import database as db
            with db.get_connection() as conn:
                conn.execute("DELETE FROM alert_history")
            logger.info("Alert history cleared (db)")
            return
        except Exception:
            pass

    _save_history({})
    logger.info("Alert history cleared")
