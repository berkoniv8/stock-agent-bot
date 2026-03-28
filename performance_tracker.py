#!/usr/bin/env python3
"""
Performance Tracker — monitors how past alerts performed against live prices,
and tracks daily portfolio P&L snapshots over time.

Alert tracking:
    Reads alerts from dashboard.csv, fetches current prices, and computes
    whether each alert hit its targets, stop-loss, or is still open.

Daily P&L Snapshots:
    Reads portfolio.json, computes total value / unrealized / realized P&L,
    and saves a daily snapshot to logs/performance_history.json.

Usage:
    python3 performance_tracker.py              # Take snapshot + show 30-day stats
    python3 performance_tracker.py --days 60    # Show last 60 days
    python3 performance_tracker.py --alerts     # Legacy alert tracking mode
    python3 performance_tracker.py --ticker AAPL  # Filter alerts by ticker
"""

import argparse
import csv
import json
import logging
import math
import os
import sys
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

HISTORY_FILE = Path("logs/performance_history.json")
PORTFOLIO_FILE = Path("portfolio.json")

DASHBOARD_CSV = Path("logs/dashboard.csv")
PERFORMANCE_CSV = Path("logs/performance.csv")


# ---------------------------------------------------------------------------
# Daily P&L Snapshots
# ---------------------------------------------------------------------------

def take_snapshot() -> Optional[Dict[str, Any]]:
    """Read portfolio.json, compute totals, and save a daily snapshot.

    Saves to logs/performance_history.json (JSON array of daily entries).
    Only one snapshot per calendar date — overwrites if same date exists.
    Returns the snapshot dict, or None on error.
    """
    if not PORTFOLIO_FILE.exists():
        logger.warning("portfolio.json not found — cannot take snapshot")
        return None

    try:
        with open(PORTFOLIO_FILE, "r") as f:
            portfolio = json.load(f)
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("Failed to read portfolio.json: %s", exc)
        return None

    holdings = portfolio.get("holdings", [])
    net_liq = float(portfolio.get("total_portfolio_value", 0))
    cash = float(portfolio.get("available_cash", 0))
    realized_ytd = float(portfolio.get("realized_pnl_ytd", 0))
    unrealized_pnl = sum(
        float(h.get("unrealized_pnl", 0) or 0) for h in holdings
    )
    total_pnl = unrealized_pnl + realized_ytd

    # Identify best and worst holdings by unrealized P&L
    best_ticker = ""
    worst_ticker = ""
    if holdings:
        sorted_h = sorted(
            holdings,
            key=lambda h: float(h.get("unrealized_pnl", 0) or 0),
        )
        worst_ticker = sorted_h[0].get("ticker", "")
        best_ticker = sorted_h[-1].get("ticker", "")

    today_str = date.today().isoformat()

    snapshot = {
        "date": today_str,
        "net_liq": round(net_liq, 2),
        "unrealized_pnl": round(unrealized_pnl, 2),
        "realized_ytd": round(realized_ytd, 2),
        "total_pnl": round(total_pnl, 2),
        "num_positions": len(holdings),
        "cash": round(cash, 2),
        "best_ticker": best_ticker,
        "worst_ticker": worst_ticker,
    }

    # Load existing history
    history = _load_history_file()

    # Overwrite entry for today if it already exists
    history = [entry for entry in history if entry.get("date") != today_str]
    history.append(snapshot)

    # Sort by date
    history.sort(key=lambda e: e.get("date", ""))

    # Save
    os.makedirs("logs", exist_ok=True)
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history, f, indent=2)
        logger.info(
            "Performance snapshot saved: %s net_liq=$%.0f total_pnl=$%.0f",
            today_str, net_liq, total_pnl,
        )
    except OSError as exc:
        logger.error("Failed to write performance history: %s", exc)
        return None

    return snapshot


def _load_history_file() -> List[Dict[str, Any]]:
    """Load the raw history file, returning an empty list on error."""
    if not HISTORY_FILE.exists():
        return []
    try:
        with open(HISTORY_FILE, "r") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        return []
    except (json.JSONDecodeError, OSError):
        return []


def get_history(days: int = 30) -> List[Dict[str, Any]]:
    """Load and return the last *days* performance snapshots."""
    history = _load_history_file()
    if not history:
        return []
    # Already sorted by date in take_snapshot; just return the tail
    return history[-days:]


def compute_stats(history: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute performance statistics from a list of daily snapshots.

    Returns a dict with:
        total_return_pct, best_day, worst_day, current_streak,
        max_drawdown_pct, avg_daily_change, sharpe_estimate
    """
    stats = {
        "total_return_pct": 0.0,
        "best_day": {"date": "", "change": 0.0},
        "worst_day": {"date": "", "change": 0.0},
        "current_streak": {"days": 0, "direction": "flat"},
        "max_drawdown_pct": 0.0,
        "avg_daily_change": 0.0,
        "sharpe_estimate": 0.0,
    }  # type: Dict[str, Any]

    if len(history) < 2:
        return stats

    net_liqs = [float(h.get("net_liq", 0)) for h in history]

    # Total return
    first_val = net_liqs[0]
    last_val = net_liqs[-1]
    if first_val > 0:
        stats["total_return_pct"] = round(
            ((last_val - first_val) / first_val) * 100, 2
        )

    # Daily changes
    daily_changes = []  # type: List[float]
    best_change = 0.0
    best_date = ""
    worst_change = 0.0
    worst_date = ""

    for i in range(1, len(history)):
        prev_val = net_liqs[i - 1]
        curr_val = net_liqs[i]
        change = curr_val - prev_val
        daily_changes.append(change)

        if change > best_change:
            best_change = change
            best_date = history[i].get("date", "")
        if change < worst_change:
            worst_change = change
            worst_date = history[i].get("date", "")

    stats["best_day"] = {"date": best_date, "change": round(best_change, 2)}
    stats["worst_day"] = {"date": worst_date, "change": round(worst_change, 2)}

    # Average daily change
    if daily_changes:
        stats["avg_daily_change"] = round(
            sum(daily_changes) / len(daily_changes), 2
        )

    # Current streak (consecutive up or down days)
    streak_dir = "flat"
    streak_count = 0
    for change in reversed(daily_changes):
        if change > 0:
            if streak_dir == "up" or streak_dir == "flat":
                streak_dir = "up"
                streak_count += 1
            else:
                break
        elif change < 0:
            if streak_dir == "down" or streak_dir == "flat":
                streak_dir = "down"
                streak_count += 1
            else:
                break
        else:
            break
    stats["current_streak"] = {"days": streak_count, "direction": streak_dir}

    # Max drawdown from peak
    peak = net_liqs[0]
    max_dd = 0.0
    for val in net_liqs[1:]:
        if val > peak:
            peak = val
        if peak > 0:
            dd = ((peak - val) / peak) * 100
            if dd > max_dd:
                max_dd = dd
    stats["max_drawdown_pct"] = round(max_dd, 2)

    # Sharpe estimate: daily_return_mean / daily_return_stdev * sqrt(252)
    daily_returns = []  # type: List[float]
    for i in range(1, len(net_liqs)):
        prev = net_liqs[i - 1]
        if prev > 0:
            daily_returns.append((net_liqs[i] - prev) / prev)

    if len(daily_returns) >= 2:
        mean_ret = sum(daily_returns) / len(daily_returns)
        variance = sum((r - mean_ret) ** 2 for r in daily_returns) / (
            len(daily_returns) - 1
        )
        stdev = math.sqrt(variance) if variance > 0 else 0
        if stdev > 0:
            stats["sharpe_estimate"] = round(
                (mean_ret / stdev) * math.sqrt(252), 2
            )

    return stats


def format_report(
    stats: Dict[str, Any],
    history: List[Dict[str, Any]],
) -> str:
    """Human-readable performance report with key stats + last 7 days."""
    lines = []  # type: List[str]
    lines.append("Portfolio Performance Report")
    lines.append("=" * 35)

    total_ret = stats.get("total_return_pct", 0)
    lines.append("Total Return: %+.2f%%" % total_ret)

    best = stats.get("best_day", {})
    worst = stats.get("worst_day", {})
    lines.append(
        "Best Day:  $%+,.0f  (%s)" % (best.get("change", 0), best.get("date", "N/A"))
    )
    lines.append(
        "Worst Day: $%+,.0f  (%s)" % (worst.get("change", 0), worst.get("date", "N/A"))
    )

    streak = stats.get("current_streak", {})
    streak_days = streak.get("days", 0)
    streak_dir = streak.get("direction", "flat")
    if streak_days > 0:
        lines.append("Streak: %d %s day%s" % (
            streak_days, streak_dir, "s" if streak_days != 1 else ""))
    else:
        lines.append("Streak: flat")

    lines.append("Max Drawdown: %.2f%%" % stats.get("max_drawdown_pct", 0))
    lines.append("Avg Daily Change: $%+,.0f" % stats.get("avg_daily_change", 0))
    sharpe = stats.get("sharpe_estimate", 0)
    lines.append("Sharpe (est): %.2f" % sharpe)

    # Last 7 days table
    recent = history[-7:] if len(history) >= 7 else history
    if recent:
        lines.append("")
        lines.append("Last %d Days:" % len(recent))
        lines.append("%-12s %12s %12s" % ("Date", "Net Liq", "Total P&L"))
        lines.append("-" * 38)
        for entry in recent:
            lines.append(
                "%-12s $%11s $%+11s"
                % (
                    entry.get("date", ""),
                    "{:,.0f}".format(entry.get("net_liq", 0)),
                    "{:,.0f}".format(entry.get("total_pnl", 0)),
                )
            )

    return "\n".join(lines)


def format_chart(history: List[Dict[str, Any]]) -> str:
    """ASCII bar chart of portfolio net liquidation value over time."""
    if not history:
        return "No performance history available."

    net_liqs = [float(h.get("net_liq", 0)) for h in history]
    dates = [h.get("date", "")[5:] for h in history]  # MM-DD only

    min_val = min(net_liqs) if net_liqs else 0
    max_val = max(net_liqs) if net_liqs else 0
    val_range = max_val - min_val

    chart_width = 40
    lines = []  # type: List[str]
    lines.append("Portfolio Value Chart")
    lines.append("=" * (chart_width + 18))

    for i, (d, val) in enumerate(zip(dates, net_liqs)):
        if val_range > 0:
            bar_len = int(((val - min_val) / val_range) * chart_width)
        else:
            bar_len = chart_width // 2
        bar_len = max(bar_len, 1)
        bar = "#" * bar_len
        lines.append("%s | %s $%s" % (d, bar, "{:,.0f}".format(val)))

    lines.append("")
    lines.append("Range: $%s - $%s" % (
        "{:,.0f}".format(min_val), "{:,.0f}".format(max_val)))

    return "\n".join(lines)


def load_alerts() -> List[Dict]:
    """Load past alerts from the dashboard log."""
    if not DASHBOARD_CSV.exists():
        return []
    with open(DASHBOARD_CSV, newline="") as f:
        return list(csv.DictReader(f))


def evaluate_alert(alert: dict) -> dict:
    """Evaluate a single past alert against subsequent price action.

    Fetches price data from alert date to now and checks if:
    - Stop-loss was hit
    - Target 1 was hit
    - Target 2 was hit
    - Target 3 was hit
    """
    ticker = alert.get("ticker", "")
    direction = alert.get("direction", "BUY")

    try:
        entry_price = float(alert.get("entry_price", 0))
        stop_loss = float(alert.get("stop_loss", 0))
        target_1 = float(alert.get("target_1", 0))
        target_2 = float(alert.get("target_2", 0))
        target_3 = float(alert.get("target_3", 0))
    except (ValueError, TypeError):
        return {**alert, "status": "error", "current_price": 0, "unrealized_pnl": 0}

    # Parse alert timestamp
    timestamp = alert.get("timestamp", "")
    try:
        alert_date = datetime.fromisoformat(timestamp).strftime("%Y-%m-%d")
    except ValueError:
        alert_date = timestamp[:10] if len(timestamp) >= 10 else None

    if not alert_date or entry_price == 0:
        return {**alert, "status": "error", "current_price": 0, "unrealized_pnl": 0}

    # Fetch price history since alert
    try:
        tk = yf.Ticker(ticker)
        df = tk.history(start=alert_date, interval="1d")
        if df.empty:
            return {**alert, "status": "no_data", "current_price": 0, "unrealized_pnl": 0}
        df.index = pd.to_datetime(df.index).tz_localize(None)
    except Exception as e:
        logger.error("Error fetching %s: %s", ticker, e)
        return {**alert, "status": "error", "current_price": 0, "unrealized_pnl": 0}

    current_price = float(df["Close"].iloc[-1])
    highs = df["High"].values
    lows = df["Low"].values

    # Determine outcome
    stop_hit = False
    t1_hit = False
    t2_hit = False
    t3_hit = False
    stop_date = ""
    t1_date = ""
    t2_date = ""
    t3_date = ""

    for idx in range(len(df)):
        bar_date = str(df.index[idx].date())

        if direction == "BUY":
            if not stop_hit and lows[idx] <= stop_loss:
                stop_hit = True
                stop_date = bar_date
            if not t1_hit and highs[idx] >= target_1:
                t1_hit = True
                t1_date = bar_date
            if not t2_hit and highs[idx] >= target_2:
                t2_hit = True
                t2_date = bar_date
            if not t3_hit and highs[idx] >= target_3:
                t3_hit = True
                t3_date = bar_date
        else:
            if not stop_hit and highs[idx] >= stop_loss:
                stop_hit = True
                stop_date = bar_date
            if not t1_hit and lows[idx] <= target_1:
                t1_hit = True
                t1_date = bar_date
            if not t2_hit and lows[idx] <= target_2:
                t2_hit = True
                t2_date = bar_date
            if not t3_hit and lows[idx] <= target_3:
                t3_hit = True
                t3_date = bar_date

    # Determine status
    if stop_hit and not t1_hit:
        status = "STOPPED OUT"
    elif t3_hit:
        status = "TARGET 3 HIT"
    elif t2_hit:
        status = "TARGET 2 HIT"
    elif t1_hit:
        status = "TARGET 1 HIT"
    elif stop_hit and t1_hit:
        status = "MIXED (stop + target)"
    else:
        status = "OPEN"

    # Compute unrealized P&L
    if direction == "BUY":
        unrealized_pnl = current_price - entry_price
    else:
        unrealized_pnl = entry_price - current_price

    unrealized_pct = (unrealized_pnl / entry_price) * 100 if entry_price else 0
    shares = int(alert.get("shares", 1))
    dollar_pnl = unrealized_pnl * shares

    days_held = (df.index[-1] - df.index[0]).days

    return {
        **alert,
        "status": status,
        "current_price": round(current_price, 2),
        "unrealized_pnl": round(unrealized_pnl, 2),
        "unrealized_pct": round(unrealized_pct, 2),
        "dollar_pnl": round(dollar_pnl, 2),
        "days_held": days_held,
        "stop_hit": stop_hit,
        "stop_date": stop_date,
        "t1_hit": t1_hit,
        "t1_date": t1_date,
        "t2_hit": t2_hit,
        "t2_date": t2_date,
        "t3_hit": t3_hit,
        "t3_date": t3_date,
    }


def print_performance(results: List[Dict]) -> None:
    """Print a formatted performance report."""
    if not results:
        print("No alerts to evaluate.")
        return

    print(f"\n{'=' * 80}")
    print(f"  PERFORMANCE TRACKER — {len(results)} alerts evaluated")
    print(f"{'=' * 80}")

    # Summary stats
    statuses = [r.get("status", "") for r in results]
    t1_count = sum(1 for s in statuses if "TARGET" in s)
    stop_count = sum(1 for s in statuses if "STOPPED" in s)
    open_count = sum(1 for s in statuses if s == "OPEN")
    total_dollar = sum(r.get("dollar_pnl", 0) for r in results)

    print(f"  Targets hit:      {t1_count}")
    print(f"  Stopped out:      {stop_count}")
    print(f"  Still open:       {open_count}")
    print(f"  Total $ P&L:      ${total_dollar:,.2f}")
    print()

    # Individual alerts
    print(f"  {'Ticker':>6} {'Dir':>4} {'Entry':>8} {'Now':>8} {'P&L':>8} {'%':>7} {'Status':>18} {'Days':>5}")
    print(f"  {'─' * 70}")

    for r in results:
        if r.get("status") == "error":
            continue
        ticker = r.get("ticker", "")
        direction = r.get("direction", "")
        entry = float(r.get("entry_price", 0))
        current = r.get("current_price", 0)
        pnl = r.get("unrealized_pnl", 0)
        pct = r.get("unrealized_pct", 0)
        status = r.get("status", "")
        days = r.get("days_held", 0)

        color_pnl = f"${pnl:+,.2f}"
        color_pct = f"{pct:+.1f}%"

        print(f"  {ticker:>6} {direction:>4} ${entry:>7,.2f} ${current:>7,.2f} {color_pnl:>8} {color_pct:>7} {status:>18} {days:>5}")

    print(f"{'=' * 80}\n")


def save_performance_csv(results: List[Dict]) -> None:
    """Save performance evaluation to CSV."""
    if not results:
        return

    fields = [
        "timestamp", "ticker", "direction", "signal_score", "entry_price",
        "stop_loss", "current_price", "status", "unrealized_pnl", "unrealized_pct",
        "dollar_pnl", "days_held", "t1_hit", "t1_date", "t2_hit", "t2_date",
        "t3_hit", "t3_date", "stop_hit", "stop_date",
    ]

    with open(PERFORMANCE_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    print(f"Performance data saved to {PERFORMANCE_CSV}")


def main():
    parser = argparse.ArgumentParser(description="Stock Agent Performance Tracker")
    parser.add_argument(
        "--days", type=int, default=30,
        help="Number of days of history to show (default: 30)",
    )
    parser.add_argument(
        "--alerts", action="store_true",
        help="Legacy mode: evaluate past alert performance",
    )
    parser.add_argument("--ticker", type=str, help="Filter alerts by ticker (use with --alerts)")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s")
    for lib in ("urllib3", "yfinance", "peewee"):
        logging.getLogger(lib).setLevel(logging.WARNING)

    if args.alerts:
        # Legacy alert tracking mode
        alerts = load_alerts()
        if args.ticker:
            alerts = [a for a in alerts if a.get("ticker", "").upper() == args.ticker.upper()]
        if not alerts:
            print("No alerts found in dashboard.csv")
            return
        print("Evaluating %d alerts..." % len(alerts))
        results = []
        for alert in alerts:
            result = evaluate_alert(alert)
            results.append(result)
        print_performance(results)
        save_performance_csv(results)
        return

    # Daily P&L snapshot mode (default)
    snapshot = take_snapshot()
    if snapshot:
        print("Snapshot saved: net_liq=$%s total_pnl=$%s" % (
            "{:,.0f}".format(snapshot["net_liq"]),
            "{:+,.0f}".format(snapshot["total_pnl"]),
        ))

    history = get_history(days=args.days)
    if not history:
        print("No performance history yet. Run again after taking a snapshot.")
        return

    stats = compute_stats(history)
    print()
    print(format_report(stats, history))
    print()
    print(format_chart(history))


if __name__ == "__main__":
    main()
