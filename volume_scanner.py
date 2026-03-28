#!/usr/bin/env python3
"""
Volume Scanner — detects stocks with abnormal trading volume.

Unusual volume often precedes big price moves. This module scans the watchlist
(and/or portfolio holdings) for volume spikes relative to a 20-day average.

Usage:
    python3 volume_scanner.py                  # Scan watchlist with default 2x threshold
    python3 volume_scanner.py --threshold 3.0  # Use 3x average volume threshold
"""

import argparse
import json
import logging
import os
from datetime import datetime
from typing import Dict, List, Optional

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


def scan_volume_spikes(
    tickers: Optional[List[str]] = None,
    threshold: float = 2.0,
) -> List[Dict]:
    """
    Scan tickers for unusual volume relative to a 20-day average.

    Parameters
    ----------
    tickers : list of str, optional
        Tickers to scan. If None, loads from watchlist + portfolio.
    threshold : float
        Volume ratio above which a spike is flagged (default 2.0 = 200% of average).

    Returns
    -------
    list of dict
        Each dict has: ticker, current_volume, avg_volume, volume_ratio,
        price_change_pct, signal, held.
    """
    ticker_list = _get_tickers(tickers)
    held = _get_held_tickers()
    spikes = []

    logger.info("Scanning %d tickers for volume spikes (threshold=%.1fx)...", len(ticker_list), threshold)

    for ticker in ticker_list:
        try:
            yf_ticker = yf.Ticker(ticker)
            # Fetch ~30 trading days to compute 20-day average
            hist = yf_ticker.history(period="1mo")
            if hist is None or hist.empty or len(hist) < 5:
                logger.debug("Skipping %s — insufficient data (%d rows)", ticker, len(hist) if hist is not None else 0)
                continue

            # Current day is the last row
            current_volume = int(hist["Volume"].iloc[-1])
            current_close = float(hist["Close"].iloc[-1])
            prev_close = float(hist["Close"].iloc[-2])

            # 20-day average volume (exclude today)
            vol_window = hist["Volume"].iloc[:-1]
            if len(vol_window) < 5:
                continue
            avg_volume = int(vol_window.tail(20).mean())

            if avg_volume == 0:
                continue

            volume_ratio = current_volume / avg_volume
            price_change_pct = ((current_close - prev_close) / prev_close) * 100

            if volume_ratio >= threshold:
                # Determine signal type
                if price_change_pct > 0.5:
                    signal = "ACCUMULATION"
                elif price_change_pct < -0.5:
                    signal = "DISTRIBUTION"
                else:
                    signal = "UNUSUAL"

                spikes.append({
                    "ticker": ticker,
                    "current_volume": current_volume,
                    "avg_volume": avg_volume,
                    "volume_ratio": round(volume_ratio, 2),
                    "price_change_pct": round(price_change_pct, 2),
                    "signal": signal,
                    "held": ticker in held,
                })

        except Exception as e:
            logger.warning("Error scanning %s: %s", ticker, e)
            continue

    # Sort by volume ratio descending
    spikes.sort(key=lambda x: x["volume_ratio"], reverse=True)
    logger.info("Found %d volume spikes.", len(spikes))
    return spikes


def format_report(spikes: List[Dict]) -> str:
    """Format volume spikes into a human-readable report."""
    if not spikes:
        return "No unusual volume detected."

    lines = []
    lines.append("Volume Spike Report")
    lines.append("=" * 40)
    lines.append("Threshold exceeded — %d ticker(s)\n" % len(spikes))

    for s in spikes:
        held_tag = " [HELD]" if s.get("held") else ""
        lines.append("%s%s — %s" % (s["ticker"], held_tag, s["signal"]))
        lines.append("  Volume: {:,} (avg {:,})".format(s["current_volume"], s["avg_volume"]))
        lines.append("  Ratio: %.1fx average" % s["volume_ratio"])
        lines.append("  Price Change: %+.2f%%" % s["price_change_pct"])
        lines.append("")

    lines.append("Signals:")
    lines.append("  ACCUMULATION = price up + volume spike (bullish)")
    lines.append("  DISTRIBUTION = price down + volume spike (bearish)")
    lines.append("  UNUSUAL = flat price + volume spike (watch closely)")

    return "\n".join(lines)


def send_volume_alerts(spikes: List[Dict]) -> None:
    """Send volume spike alerts via email and Telegram."""
    if not spikes:
        return

    report = format_report(spikes)

    # Telegram
    try:
        import telegram_bot
        telegram_bot.send_message(report)
    except Exception:
        pass

    # Email
    try:
        import notifications
        notifications.send_email_text(report, subject="Volume Spike Alert — %d tickers" % len(spikes))
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Scan for unusual volume spikes")
    parser.add_argument("--threshold", type=float, default=2.0,
                        help="Volume ratio threshold (default: 2.0)")
    parser.add_argument("--notify", action="store_true",
                        help="Send alerts via email + Telegram")
    parser.add_argument("--tickers", nargs="+", help="Specific tickers to scan")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    spikes = scan_volume_spikes(tickers=args.tickers, threshold=args.threshold)
    print(format_report(spikes))

    if args.notify:
        send_volume_alerts(spikes)
        print("\nAlerts sent." if spikes else "\nNo alerts to send.")


if __name__ == "__main__":
    main()
