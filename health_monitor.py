#!/usr/bin/env python3
"""
System Health Monitor — checks API connectivity, data freshness, rate limits,
and disk usage to ensure the trading agent is operating correctly.

Checks:
- YAHOO_FINANCE   yfinance API reachable and returning data
- DATA_FRESHNESS  OHLCV data is recent (within 2 trading days)
- DISK_SPACE      Logs directory not exceeding size limit
- CONFIG_VALID    Required .env variables present
- POSITIONS_SYNC  Paper trading state file valid and consistent

Usage:
    python3 health_monitor.py           # Run all checks
    python3 health_monitor.py --json    # Output as JSON
    python3 health_monitor.py --check yahoo  # Single check
"""

import argparse
import json
import logging
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

MAX_LOG_SIZE_MB = float(os.getenv("MAX_LOG_SIZE_MB", "100"))
REQUIRED_ENV_VARS = [
    "INITIAL_CAPITAL",
    "RISK_PER_TRADE_PCT",
    "MAX_OPEN_POSITIONS",
]


def check_yahoo_finance():
    # type: () -> Dict
    """Check if yfinance API is reachable."""
    try:
        import yfinance as yf
        tk = yf.Ticker("SPY")
        data = tk.history(period="1d")
        if data.empty:
            return {
                "name": "YAHOO_FINANCE",
                "status": "WARN",
                "message": "API returned empty data for SPY",
            }
        return {
            "name": "YAHOO_FINANCE",
            "status": "OK",
            "message": "API reachable, SPY last close: $%.2f" % float(data["Close"].iloc[-1]),
        }
    except Exception as e:
        return {
            "name": "YAHOO_FINANCE",
            "status": "FAIL",
            "message": "API unreachable: %s" % str(e)[:100],
        }


def check_data_freshness():
    # type: () -> Dict
    """Check if cached OHLCV data is recent."""
    data_dir = Path("data")
    if not data_dir.exists():
        return {
            "name": "DATA_FRESHNESS",
            "status": "WARN",
            "message": "No data directory found",
        }

    csv_files = list(data_dir.glob("*.csv"))
    if not csv_files:
        return {
            "name": "DATA_FRESHNESS",
            "status": "WARN",
            "message": "No CSV data files found",
        }

    now = datetime.now()
    stale = []
    for f in csv_files:
        mtime = datetime.fromtimestamp(f.stat().st_mtime)
        age_hours = (now - mtime).total_seconds() / 3600
        # More than 48 hours old on a weekday is stale
        if age_hours > 48:
            stale.append(f.name)

    if stale:
        return {
            "name": "DATA_FRESHNESS",
            "status": "WARN",
            "message": "%d stale files (>48h): %s" % (len(stale), ", ".join(stale[:5])),
        }

    return {
        "name": "DATA_FRESHNESS",
        "status": "OK",
        "message": "%d data files, all recent" % len(csv_files),
    }


def check_disk_space():
    # type: () -> Dict
    """Check logs directory size."""
    logs_dir = Path("logs")
    if not logs_dir.exists():
        return {
            "name": "DISK_SPACE",
            "status": "OK",
            "message": "No logs directory yet",
        }

    total_bytes = 0
    file_count = 0
    for f in logs_dir.rglob("*"):
        if f.is_file():
            total_bytes += f.stat().st_size
            file_count += 1

    total_mb = total_bytes / (1024 * 1024)
    status = "OK"
    if total_mb > MAX_LOG_SIZE_MB:
        status = "WARN"
    elif total_mb > MAX_LOG_SIZE_MB * 2:
        status = "FAIL"

    return {
        "name": "DISK_SPACE",
        "status": status,
        "message": "Logs: %.1f MB across %d files (limit: %.0f MB)" % (
            total_mb, file_count, MAX_LOG_SIZE_MB),
    }


def check_config():
    # type: () -> Dict
    """Check that required environment variables are set."""
    missing = [v for v in REQUIRED_ENV_VARS if not os.getenv(v)]
    if missing:
        return {
            "name": "CONFIG_VALID",
            "status": "WARN",
            "message": "Missing env vars: %s" % ", ".join(missing),
        }
    return {
        "name": "CONFIG_VALID",
        "status": "OK",
        "message": "All %d required env vars present" % len(REQUIRED_ENV_VARS),
    }


def check_positions_sync():
    # type: () -> Dict
    """Check paper trading state file integrity."""
    state_file = Path("logs/paper_state.json")
    if not state_file.exists():
        return {
            "name": "POSITIONS_SYNC",
            "status": "OK",
            "message": "No paper trading state (not started yet)",
        }

    try:
        with open(state_file) as f:
            state = json.load(f)

        open_pos = state.get("open_positions", [])
        closed = state.get("closed_trades", [])
        capital = state.get("capital", 0)

        issues = []
        if capital < 0:
            issues.append("negative capital ($%.2f)" % capital)

        for p in open_pos:
            if "ticker" not in p:
                issues.append("position missing ticker")
            if p.get("shares", 0) <= 0:
                issues.append("%s has invalid shares" % p.get("ticker", "?"))

        if issues:
            return {
                "name": "POSITIONS_SYNC",
                "status": "WARN",
                "message": "Issues: %s" % "; ".join(issues[:3]),
            }

        return {
            "name": "POSITIONS_SYNC",
            "status": "OK",
            "message": "%d open positions, %d closed trades, capital: $%s" % (
                len(open_pos), len(closed), "{:,.2f}".format(capital)),
        }
    except json.JSONDecodeError:
        return {
            "name": "POSITIONS_SYNC",
            "status": "FAIL",
            "message": "Corrupt state file — cannot parse JSON",
        }
    except Exception as e:
        return {
            "name": "POSITIONS_SYNC",
            "status": "FAIL",
            "message": "Error reading state: %s" % str(e)[:80],
        }


# All checks registry
ALL_CHECKS = {
    "yahoo": check_yahoo_finance,
    "freshness": check_data_freshness,
    "disk": check_disk_space,
    "config": check_config,
    "positions": check_positions_sync,
}


def run_all_checks():
    # type: () -> List[Dict]
    """Run all health checks and return results."""
    results = []
    for name, fn in ALL_CHECKS.items():
        try:
            result = fn()
            result["checked_at"] = datetime.now().isoformat()
            results.append(result)
        except Exception as e:
            results.append({
                "name": name.upper(),
                "status": "FAIL",
                "message": "Check crashed: %s" % str(e)[:80],
                "checked_at": datetime.now().isoformat(),
            })
    return results


def run_single_check(check_name):
    # type: (str) -> Optional[Dict]
    """Run a single named health check."""
    fn = ALL_CHECKS.get(check_name)
    if fn is None:
        return None
    result = fn()
    result["checked_at"] = datetime.now().isoformat()
    return result


def get_overall_status(results):
    # type: (List[Dict]) -> str
    """Determine overall system status from check results."""
    statuses = [r["status"] for r in results]
    if "FAIL" in statuses:
        return "FAIL"
    if "WARN" in statuses:
        return "WARN"
    return "OK"


def save_health_report(results):
    # type: (List[Dict]) -> None
    """Save health check results to disk."""
    os.makedirs("logs", exist_ok=True)
    report = {
        "timestamp": datetime.now().isoformat(),
        "overall": get_overall_status(results),
        "checks": results,
    }
    with open("logs/health_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_health_report(results):
    # type: (List[Dict]) -> None
    """Print formatted health report."""
    overall = get_overall_status(results)
    status_icon = {"OK": "+", "WARN": "!", "FAIL": "X"}

    print("\n  SYSTEM HEALTH CHECK")
    print("  " + "-" * 55)

    for r in results:
        icon = status_icon.get(r["status"], "?")
        print("  [%s] %-18s %s" % (icon, r["name"], r["message"]))

    print("  " + "-" * 55)
    print("  Overall: %s" % overall)
    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="System Health Monitor")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--check", type=str, choices=list(ALL_CHECKS.keys()),
                        help="Run single check")
    parser.add_argument("--save", action="store_true", help="Save report to disk")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-8s  %(message)s")

    if args.check:
        result = run_single_check(args.check)
        if result:
            if args.json:
                print(json.dumps(result, indent=2, default=str))
            else:
                print("\n  [%s] %s: %s\n" % (
                    result["status"], result["name"], result["message"]))
        return

    results = run_all_checks()

    if args.save:
        save_health_report(results)

    if args.json:
        print(json.dumps({
            "overall": get_overall_status(results),
            "checks": results,
        }, indent=2, default=str))
    else:
        print_health_report(results)


if __name__ == "__main__":
    main()
