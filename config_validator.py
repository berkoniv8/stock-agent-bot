"""
Config Validator — startup health check for the stock agent.

Validates .env settings, portfolio.json, watchlist.csv,
and API connectivity before running the agent.
"""

import csv
import json
import logging
import os
from pathlib import Path
from typing import List, Tuple

from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


def validate_env() -> List[Tuple[str, str, str]]:
    """Check .env configuration.

    Returns list of (level, key, message) tuples.
    level is 'ok', 'warn', or 'error'.
    """
    checks = []

    # Required numeric values
    for key, default in [
        ("TOTAL_PORTFOLIO_VALUE", "50000"),
        ("AVAILABLE_CASH", "10000"),
        ("MAX_RISK_PER_TRADE_PCT", "1.0"),
        ("MAX_POSITION_SIZE_PCT", "10.0"),
        ("SIGNAL_THRESHOLD", "5"),
        ("RUN_INTERVAL_MINUTES", "15"),
    ]:
        val = os.getenv(key, default)
        try:
            float(val)
            checks.append(("ok", key, f"{val}"))
        except ValueError:
            checks.append(("error", key, f"Invalid numeric value: '{val}'"))

    # Validate risk parameters make sense
    try:
        portfolio = float(os.getenv("TOTAL_PORTFOLIO_VALUE", "50000"))
        cash = float(os.getenv("AVAILABLE_CASH", "10000"))
        risk_pct = float(os.getenv("MAX_RISK_PER_TRADE_PCT", "1.0"))
        pos_pct = float(os.getenv("MAX_POSITION_SIZE_PCT", "10.0"))

        if cash > portfolio:
            checks.append(("warn", "AVAILABLE_CASH", "Cash exceeds total portfolio value"))
        if risk_pct > 5:
            checks.append(("warn", "MAX_RISK_PER_TRADE_PCT", f"Risk per trade {risk_pct}% is aggressive (>5%)"))
        if pos_pct > 25:
            checks.append(("warn", "MAX_POSITION_SIZE_PCT", f"Max position {pos_pct}% is concentrated (>25%)"))
    except ValueError:
        pass

    # API keys — warn if missing, not error (system works without them)
    api_keys = {
        "ALPHA_VANTAGE_API_KEY": "Alpha Vantage (price data fallback)",
        "FMP_API_KEY": "Financial Modeling Prep (fundamentals)",
        "NEWSAPI_KEY": "NewsAPI (news sentiment)",
        "FINNHUB_API_KEY": "Finnhub (news fallback)",
    }
    for key, desc in api_keys.items():
        val = os.getenv(key, "")
        if not val or val.startswith("your_"):
            checks.append(("warn", key, f"Not configured — {desc} disabled"))
        else:
            checks.append(("ok", key, f"Configured ({len(val)} chars)"))

    # Notification channels
    notif_keys = {
        "SMTP_USER": "Email notifications",
        "SLACK_WEBHOOK_URL": "Slack notifications",
        "TELEGRAM_BOT_TOKEN": "Telegram notifications",
    }
    configured_notifs = 0
    for key, desc in notif_keys.items():
        val = os.getenv(key, "")
        if val and not val.startswith("your_") and not val.startswith("https://hooks.slack.com/services/YOUR"):
            checks.append(("ok", key, f"{desc} enabled"))
            configured_notifs += 1
        else:
            checks.append(("warn", key, f"{desc} not configured"))

    if configured_notifs == 0:
        checks.append(("warn", "NOTIFICATIONS", "No notification channels configured — alerts will only print to console and log to CSV"))

    return checks


def validate_portfolio() -> List[Tuple[str, str, str]]:
    """Check portfolio.json."""
    checks = []
    path = Path("portfolio.json")

    if not path.exists():
        checks.append(("warn", "portfolio.json", "File not found — using .env defaults"))
        return checks

    try:
        with open(path) as f:
            data = json.load(f)

        required = ["total_portfolio_value", "available_cash"]
        for key in required:
            if key in data:
                checks.append(("ok", f"portfolio.{key}", str(data[key])))
            else:
                checks.append(("warn", f"portfolio.{key}", "Not set in portfolio.json"))

        holdings = data.get("holdings", [])
        checks.append(("ok", "portfolio.holdings", f"{len(holdings)} holdings configured"))

        for h in holdings:
            if not h.get("ticker"):
                checks.append(("error", "portfolio.holdings", "Holding missing 'ticker' field"))
            if not h.get("shares"):
                checks.append(("warn", f"portfolio.{h.get('ticker', '?')}", "Shares is 0 or missing"))

    except json.JSONDecodeError as e:
        checks.append(("error", "portfolio.json", f"Invalid JSON: {e}"))
    except Exception as e:
        checks.append(("error", "portfolio.json", f"Read error: {e}"))

    return checks


def validate_watchlist() -> List[Tuple[str, str, str]]:
    """Check watchlist.csv."""
    checks = []
    path = Path("watchlist.csv")

    if not path.exists():
        checks.append(("error", "watchlist.csv", "File not found — agent has nothing to scan"))
        return checks

    try:
        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        if not rows:
            checks.append(("error", "watchlist.csv", "File is empty"))
            return checks

        # Check required columns
        if "ticker" not in reader.fieldnames:
            checks.append(("error", "watchlist.csv", "Missing 'ticker' column"))
            return checks

        tickers = [r["ticker"].strip().upper() for r in rows if r.get("ticker")]
        checks.append(("ok", "watchlist.csv", f"{len(tickers)} tickers: {', '.join(tickers[:10])}{'...' if len(tickers) > 10 else ''}"))

        # Check for duplicates
        seen = set()
        dupes = set()
        for t in tickers:
            if t in seen:
                dupes.add(t)
            seen.add(t)
        if dupes:
            checks.append(("warn", "watchlist.csv", f"Duplicate tickers: {', '.join(dupes)}"))

    except Exception as e:
        checks.append(("error", "watchlist.csv", f"Read error: {e}"))

    return checks


def validate_directories() -> List[Tuple[str, str, str]]:
    """Check required directories exist."""
    checks = []
    logs_dir = Path("logs")
    if logs_dir.exists():
        checks.append(("ok", "logs/", "Directory exists"))
    else:
        logs_dir.mkdir(exist_ok=True)
        checks.append(("ok", "logs/", "Directory created"))
    return checks


def run_all() -> bool:
    """Run all validations and print report.

    Returns True if no errors were found.
    """
    sections = [
        ("Environment (.env)", validate_env()),
        ("Portfolio (portfolio.json)", validate_portfolio()),
        ("Watchlist (watchlist.csv)", validate_watchlist()),
        ("Directories", validate_directories()),
    ]

    print(f"\n{'=' * 60}")
    print(f"  STOCK AGENT — STARTUP HEALTH CHECK")
    print(f"{'=' * 60}")

    total_errors = 0
    total_warnings = 0

    for section_name, checks in sections:
        print(f"\n  {section_name}")
        print(f"  {'─' * 50}")

        for level, key, msg in checks:
            if level == "ok":
                icon = "[OK]"
            elif level == "warn":
                icon = "[!!]"
                total_warnings += 1
            else:
                icon = "[XX]"
                total_errors += 1

            print(f"    {icon} {key}: {msg}")

    print(f"\n{'─' * 60}")
    if total_errors > 0:
        print(f"  {total_errors} error(s), {total_warnings} warning(s)")
        print(f"  Fix errors before running the agent.")
    elif total_warnings > 0:
        print(f"  No errors, {total_warnings} warning(s) — agent will run with reduced functionality.")
    else:
        print(f"  All checks passed.")
    print(f"{'=' * 60}\n")

    return total_errors == 0


if __name__ == "__main__":
    run_all()
