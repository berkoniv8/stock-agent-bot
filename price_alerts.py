#!/usr/bin/env python3
"""
Price Alert System — monitors tickers for key price level breaches
between scan cycles. Triggers alerts when support/resistance, round
numbers, or custom levels are hit.

Alert types:
- SUPPORT_BREAK    Price dropped below support level
- RESISTANCE_BREAK Price broke above resistance level
- STOP_WARNING     Position approaching stop-loss (within 1%)
- TARGET_HIT       Position reached a take-profit target
- CUSTOM           User-defined price level crossed

Persists alerts in logs/price_alerts.json.

Usage:
    python3 price_alerts.py --check              # Check all alerts
    python3 price_alerts.py --add AAPL above 180 # Add custom alert
    python3 price_alerts.py --add AAPL below 150 # Add custom alert
    python3 price_alerts.py --list               # List active alerts
    python3 price_alerts.py --remove 0           # Remove alert by index
"""

import argparse
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yfinance as yf
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

ALERTS_FILE = Path("logs/price_alerts.json")
STOP_WARNING_PCT = float(os.getenv("STOP_WARNING_PCT", "1.0"))


def _load_alerts():
    # type: () -> List[Dict]
    """Load saved price alerts."""
    if not ALERTS_FILE.exists():
        return []
    try:
        with open(ALERTS_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def _save_alerts(alerts):
    # type: (List[Dict]) -> None
    """Save price alerts to disk."""
    os.makedirs(os.path.dirname(ALERTS_FILE) or ".", exist_ok=True)
    with open(ALERTS_FILE, "w") as f:
        json.dump(alerts, f, indent=2, default=str)


def add_alert(ticker, condition, price, note=""):
    # type: (str, str, float, str) -> Dict
    """Add a custom price alert.

    Args:
        ticker: Stock symbol.
        condition: "above" or "below".
        price: Target price level.
        note: Optional note.

    Returns the created alert dict.
    """
    alert = {
        "ticker": ticker.upper(),
        "condition": condition,
        "price": price,
        "note": note,
        "type": "CUSTOM",
        "created_at": datetime.now().isoformat(),
        "triggered": False,
        "triggered_at": None,
    }
    alerts = _load_alerts()
    alerts.append(alert)
    _save_alerts(alerts)
    logger.info("Price alert added: %s %s $%.2f", ticker, condition, price)
    return alert


def remove_alert(index):
    # type: (int) -> bool
    """Remove alert by index. Returns True if removed."""
    alerts = _load_alerts()
    if 0 <= index < len(alerts):
        removed = alerts.pop(index)
        _save_alerts(alerts)
        logger.info("Removed alert: %s %s $%.2f",
                     removed["ticker"], removed["condition"], removed["price"])
        return True
    return False


def get_active_alerts():
    # type: () -> List[Dict]
    """Get all non-triggered alerts."""
    return [a for a in _load_alerts() if not a.get("triggered")]


def generate_position_alerts():
    # type: () -> List[Dict]
    """Auto-generate alerts from open paper positions.

    Creates stop warning and target alerts for each position.
    """
    alerts = []
    try:
        import paper_trader
        state = paper_trader.load_state()
        for pos in state.get("open_positions", []):
            ticker = pos["ticker"]
            stop = pos.get("current_stop", pos.get("stop_loss", 0))
            entry = pos["entry_price"]

            # Stop warning alert
            if pos["direction"] == "BUY" and stop > 0:
                warning_level = stop * (1 + STOP_WARNING_PCT / 100)
                alerts.append({
                    "ticker": ticker,
                    "condition": "below",
                    "price": round(warning_level, 2),
                    "type": "STOP_WARNING",
                    "note": "Stop at $%.2f" % stop,
                    "triggered": False,
                })
            elif pos["direction"] == "SELL" and stop > 0:
                warning_level = stop * (1 - STOP_WARNING_PCT / 100)
                alerts.append({
                    "ticker": ticker,
                    "condition": "above",
                    "price": round(warning_level, 2),
                    "type": "STOP_WARNING",
                    "note": "Stop at $%.2f" % stop,
                    "triggered": False,
                })

            # Target alerts
            for target_key, label in [("target_1", "T1"), ("target_2", "T2"), ("target_3", "T3")]:
                hit_key = target_key[0] + target_key[-1] + "_hit"
                if pos.get(hit_key):
                    continue
                target = pos.get(target_key, 0)
                if target <= 0:
                    continue
                condition = "above" if pos["direction"] == "BUY" else "below"
                alerts.append({
                    "ticker": ticker,
                    "condition": condition,
                    "price": round(target, 2),
                    "type": "TARGET_HIT",
                    "note": label,
                    "triggered": False,
                })

    except Exception as e:
        logger.debug("Position alert generation failed: %s", e)

    return alerts


def fetch_current_price(ticker):
    # type: (str) -> Optional[float]
    """Fetch current price for a ticker."""
    try:
        tk = yf.Ticker(ticker)
        data = tk.history(period="1d")
        if not data.empty:
            return float(data["Close"].iloc[-1])
    except Exception as e:
        logger.debug("Price fetch failed for %s: %s", ticker, e)
    return None


def check_alert(alert, current_price):
    # type: (Dict, float) -> bool
    """Check if a single alert condition is met.

    Returns True if the alert should trigger.
    """
    if alert.get("triggered"):
        return False
    if current_price is None:
        return False

    condition = alert["condition"]
    target = alert["price"]

    if condition == "above" and current_price >= target:
        return True
    if condition == "below" and current_price <= target:
        return True
    return False


def check_all_alerts(include_position_alerts=True):
    # type: (bool) -> List[Dict]
    """Check all active alerts against current prices.

    Args:
        include_position_alerts: Also check auto-generated position alerts.

    Returns list of triggered alert dicts.
    """
    custom_alerts = _load_alerts()
    active_custom = [a for a in custom_alerts if not a.get("triggered")]

    position_alerts = generate_position_alerts() if include_position_alerts else []

    all_alerts = active_custom + position_alerts

    # Group by ticker to minimize API calls
    by_ticker = {}  # type: Dict[str, List[Tuple[int, Dict]]]
    for i, alert in enumerate(all_alerts):
        ticker = alert["ticker"]
        if ticker not in by_ticker:
            by_ticker[ticker] = []
        by_ticker[ticker].append((i, alert))

    triggered = []
    for ticker, alert_list in by_ticker.items():
        price = fetch_current_price(ticker)
        if price is None:
            continue

        for idx, alert in alert_list:
            if check_alert(alert, price):
                alert["triggered"] = True
                alert["triggered_at"] = datetime.now().isoformat()
                alert["triggered_price"] = price
                triggered.append(alert)

                logger.info("PRICE ALERT: %s %s $%.2f (current: $%.2f) [%s]",
                            ticker, alert["condition"], alert["price"],
                            price, alert.get("type", "CUSTOM"))

    # Update custom alerts on disk
    if any(a.get("triggered") for a in custom_alerts):
        _save_alerts(custom_alerts)

    return triggered


def send_triggered_alerts(triggered):
    # type: (List[Dict]) -> None
    """Send triggered alerts via notification channels."""
    if not triggered:
        return

    import requests

    for alert in triggered:
        msg = "PRICE ALERT: %s hit $%.2f (%s $%.2f) — %s%s" % (
            alert["ticker"],
            alert.get("triggered_price", 0),
            alert["condition"],
            alert["price"],
            alert.get("type", "CUSTOM"),
            " — %s" % alert["note"] if alert.get("note") else "",
        )

        # Slack
        webhook = os.getenv("SLACK_WEBHOOK_URL", "")
        if webhook and not webhook.startswith("https://hooks.slack.com/services/YOUR"):
            try:
                requests.post(webhook, json={"text": msg}, timeout=10)
            except Exception:
                pass

        # Discord
        discord = os.getenv("DISCORD_WEBHOOK_URL", "")
        if discord and "discord.com/api/webhooks" in discord:
            try:
                requests.post(discord, json={"content": msg}, timeout=10)
            except Exception:
                pass

        print("  %s" % msg)


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------

def print_alerts():
    # type: () -> None
    """Print all alerts with status."""
    alerts = _load_alerts()
    position_alerts = generate_position_alerts()

    if not alerts and not position_alerts:
        print("\n  No price alerts configured.\n")
        return

    if alerts:
        print("\n  CUSTOM ALERTS")
        print("  " + "-" * 55)
        for i, a in enumerate(alerts):
            status = "TRIGGERED" if a.get("triggered") else "active"
            print("  [%d] %-6s %5s $%10.2f  %-10s  %s  %s" % (
                i, a["ticker"], a["condition"], a["price"],
                status, a.get("note", ""),
                a.get("triggered_at", "")[:16] if a.get("triggered") else ""))

    if position_alerts:
        print("\n  POSITION ALERTS (auto-generated)")
        print("  " + "-" * 55)
        for a in position_alerts:
            print("  %-6s %5s $%10.2f  %-14s  %s" % (
                a["ticker"], a["condition"], a["price"],
                a["type"], a.get("note", "")))

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Price Alert System")
    parser.add_argument("--check", action="store_true", help="Check all alerts")
    parser.add_argument("--add", nargs=3, metavar=("TICKER", "CONDITION", "PRICE"),
                        help="Add alert: TICKER above/below PRICE")
    parser.add_argument("--note", type=str, default="", help="Note for --add")
    parser.add_argument("--remove", type=int, metavar="INDEX", help="Remove alert by index")
    parser.add_argument("--list", action="store_true", help="List all alerts")
    parser.add_argument("--send", action="store_true", help="Send triggered alerts")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
    for lib in ("urllib3", "yfinance"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    if args.add:
        ticker, condition, price = args.add
        if condition not in ("above", "below"):
            print("Condition must be 'above' or 'below'")
            return
        add_alert(ticker, condition, float(price), args.note)
        print("  Alert added: %s %s $%s" % (ticker.upper(), condition, price))
        return

    if args.remove is not None:
        if remove_alert(args.remove):
            print("  Alert removed.")
        else:
            print("  Invalid alert index.")
        return

    if args.list:
        print_alerts()
        return

    if args.check:
        print("\n  Checking price alerts...")
        triggered = check_all_alerts()
        if triggered:
            print("  %d alerts triggered:" % len(triggered))
            if args.send:
                send_triggered_alerts(triggered)
            else:
                for t in triggered:
                    print("  %s %s $%.2f [%s]" % (
                        t["ticker"], t["condition"], t["price"], t.get("type", "")))
        else:
            print("  No alerts triggered.")
        print()
        return

    # Default: list
    print_alerts()


if __name__ == "__main__":
    main()
