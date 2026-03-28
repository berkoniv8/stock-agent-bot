#!/usr/bin/env python3
"""
Telegram Bot — two-way communication with the Stock Agent from your phone.

Commands:
    /buy        — BUY opportunities from full watchlist scan
    /sell       — Which positions should I sell?
    /check TSLA — Analyze any ticker (held or new)
    /scan       — Quick scan (holdings + top buy opportunities)
    /status     — Portfolio summary + P&L
    /positions  — List all open positions
    /risk       — Portfolio risk summary
    /regime     — Current market regime
    /briefing   — Morning briefing
    /volume     — Unusual volume spikes
    /upgrades   — Analyst upgrades/downgrades
    /alerts     — Active price alerts
    /help       — List commands

    Buy alerts also push automatically when the scan engine fires a signal.

Setup:
    1. Message @BotFather on Telegram, send /newbot
    2. Copy the token to .env as TELEGRAM_BOT_TOKEN
    3. Start a chat with your new bot
    4. Run: python3 telegram_bot.py --get-chat-id
    5. Copy the chat ID to .env as TELEGRAM_CHAT_ID
    6. Run: python3 telegram_bot.py

Usage:
    python3 telegram_bot.py              # Start the bot
    python3 telegram_bot.py --get-chat-id # Find your chat ID
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")
API_BASE = "https://api.telegram.org/bot%s" % TOKEN


def send_message(text, chat_id=None, parse_mode=None):
    """Send a message to the Telegram chat."""
    cid = chat_id or CHAT_ID
    if not TOKEN or not cid:
        return False

    # Telegram max message length is 4096
    chunks = [text[i:i+4000] for i in range(0, len(text), 4000)]
    for chunk in chunks:
        payload = {"chat_id": cid, "text": chunk}
        if parse_mode:
            payload["parse_mode"] = parse_mode
        try:
            resp = requests.post(API_BASE + "/sendMessage", json=payload, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            logger.error("Telegram send failed: %s", e)
            return False
    return True


def get_updates(offset=None, timeout=30):
    """Long-poll for new messages."""
    params = {"timeout": timeout}
    if offset:
        params["offset"] = offset
    try:
        resp = requests.get(API_BASE + "/getUpdates", params=params, timeout=timeout + 5)
        resp.raise_for_status()
        return resp.json().get("result", [])
    except Exception as e:
        logger.error("getUpdates failed: %s", e)
        return []


def get_chat_id():
    """Get your chat ID by reading the first message sent to the bot."""
    print("Send any message to your bot on Telegram, then press Enter here...")
    input()
    updates = get_updates(timeout=5)
    if not updates:
        print("No messages found. Make sure you sent a message to your bot.")
        return
    for u in updates:
        msg = u.get("message", {})
        chat = msg.get("chat", {})
        print("\nFound chat:")
        print("  Chat ID: %s" % chat.get("id"))
        print("  Name: %s %s" % (chat.get("first_name", ""), chat.get("last_name", "")))
        print("\nAdd this to your .env file:")
        print("  TELEGRAM_CHAT_ID=%s" % chat.get("id"))


# ---------------------------------------------------------------------------
# Command handlers
# ---------------------------------------------------------------------------

def cmd_status(args=""):
    """Portfolio summary."""
    try:
        portfolio = json.load(open("portfolio.json"))
        total = portfolio.get("total_portfolio_value", 0)
        cash = portfolio.get("available_cash", 0)
        holdings = portfolio.get("holdings", [])
        unrealized = sum(float(h.get("unrealized_pnl", 0) or 0) for h in holdings)
        realized = float(portfolio.get("realized_pnl_ytd", 0) or 0)

        lines = ["Portfolio Summary"]
        lines.append("=" * 30)
        lines.append("Net Liq: $%s" % f"{total:,.0f}")
        lines.append("Cash: $%s" % f"{cash:,.0f}")
        lines.append("Positions: %d" % len(holdings))
        lines.append("Unrealized: $%s" % f"{unrealized:+,.0f}")
        lines.append("Realized YTD: $%s" % f"{realized:+,.0f}")
        lines.append("Total P&L: $%s" % f"{unrealized + realized:+,.0f}")

        # Top winners/losers
        sorted_h = sorted(holdings, key=lambda h: float(h.get("unrealized_pnl", 0) or 0))
        if sorted_h:
            worst = sorted_h[0]
            best = sorted_h[-1]
            lines.append("")
            lines.append("Best: %s $%s (%.1f%%)" % (
                best["ticker"], f"{float(best.get('unrealized_pnl', 0)):+,.0f}",
                float(best.get("pnl_pct", 0))))
            lines.append("Worst: %s $%s (%.1f%%)" % (
                worst["ticker"], f"{float(worst.get('unrealized_pnl', 0)):+,.0f}",
                float(worst.get("pnl_pct", 0))))

        return "\n".join(lines)
    except Exception as e:
        return "Error loading portfolio: %s" % e


def cmd_positions(args=""):
    """List all open positions."""
    try:
        portfolio = json.load(open("portfolio.json"))
        holdings = portfolio.get("holdings", [])
        if not holdings:
            return "No open positions."

        lines = ["Open Positions (%d)" % len(holdings)]
        lines.append("-" * 35)
        for h in sorted(holdings, key=lambda x: x.get("ticker", "")):
            pnl = float(h.get("unrealized_pnl", 0) or 0)
            pct = float(h.get("pnl_pct", 0) or 0)
            pnl_str = "${:+,.0f}".format(pnl)
            lines.append("%s: %d @ $%.2f | %s (%.1f%%)" % (
                h["ticker"], h.get("shares", 0),
                float(h.get("current_price", 0)), pnl_str, pct))

        return "\n".join(lines)
    except Exception as e:
        return "Error: %s" % e


def cmd_sell(args=""):
    """Check which positions should be sold."""
    try:
        import position_monitor
        results = position_monitor.check_all_positions()
        return position_monitor.format_report(results)
    except Exception as e:
        return "Error running position monitor: %s" % e


def cmd_check(args=""):
    """Check a single ticker."""
    ticker = args.strip().upper() if args else ""
    if not ticker:
        try:
            portfolio = json.load(open("portfolio.json"))
            tickers = sorted(h["ticker"] for h in portfolio.get("holdings", []))
            return "Usage: /check TSLA\n\nYour holdings:\n%s\n\nOr just type a ticker name (e.g. NVDA)" % " ".join(tickers)
        except Exception:
            return "Usage: /check TSLA\n\nOr just type a ticker name (e.g. NVDA)"

    try:
        import position_monitor
        results = position_monitor.check_all_positions(ticker_filter=ticker)
        if results:
            return position_monitor.format_report(results)

        # Not in portfolio — run a scan
        import data_layer
        import technical_analysis
        import fundamental_analysis
        import signal_engine

        df = data_layer.fetch_daily_ohlcv(ticker)
        if df.empty:
            return "No data for %s" % ticker

        tech = technical_analysis.analyze(ticker, df)
        fund = fundamental_analysis.analyze(ticker, "Technology")
        alert = signal_engine.evaluate(tech, fund)

        lines = ["%s Analysis" % ticker]
        lines.append("-" * 30)
        lines.append("Price: $%.2f" % tech.current_price)
        lines.append("RSI: %.1f" % tech.rsi)
        lines.append("MACD: %.2f" % tech.macd_value)
        lines.append("Above 200 SMA: %s" % tech.price_above_200sma)
        lines.append("Fundamental: %d/6" % fund.fundamental_score)

        if alert:
            lines.append("")
            lines.append("SIGNAL: %s (score %d)" % (alert.direction, alert.signal_score))
            signals = ", ".join(s[0] for s in alert.triggered_signals)
            lines.append("Signals: %s" % signals)
        else:
            lines.append("\nNo signal (below threshold)")

        if tech.pattern_details:
            lines.append("\nPatterns: %s" % ", ".join(tech.pattern_details[:3]))

        return "\n".join(lines)
    except Exception as e:
        return "Error checking %s: %s" % (ticker, e)


def cmd_scan(args=""):
    """Quick scan — holdings signals + top buy opportunities from full watchlist."""
    try:
        import data_layer
        import signal_engine
        import technical_analysis
        import fundamental_analysis

        watchlist = data_layer.load_watchlist()
        portfolio = json.load(open("portfolio.json"))
        held_tickers = set(h["ticker"] for h in portfolio.get("holdings", []))

        hold_signals = []
        buy_signals = []

        for entry in watchlist:
            ticker = entry["ticker"]
            try:
                df = data_layer.fetch_daily_ohlcv(ticker)
                if df.empty:
                    continue
                tech = technical_analysis.analyze(ticker, df)
                fund = fundamental_analysis.analyze(ticker, entry.get("sector", ""))
                alert = signal_engine.evaluate(tech, fund)
                if alert:
                    line = "%s %s (score %d)" % (alert.direction, ticker, alert.signal_score)
                    if ticker in held_tickers:
                        hold_signals.append(line)
                    elif alert.direction == "BUY":
                        buy_signals.append((alert.signal_score, line))
            except Exception:
                continue

        lines = []
        if hold_signals:
            lines.append("Holdings Signals:")
            lines.extend(hold_signals)
        if buy_signals:
            if lines:
                lines.append("")
            lines.append("New Buy Opportunities:")
            for _, l in sorted(buy_signals, reverse=True)[:5]:
                lines.append(l)
        if not lines:
            return "No signals right now. Use /buy for a full watchlist BUY scan."
        return "\n".join(lines)
    except Exception as e:
        return "Scan error: %s" % e


def cmd_buy(args=""):
    """Scan full watchlist for BUY opportunities (new positions to enter)."""
    try:
        import data_layer
        import signal_engine
        import technical_analysis
        import fundamental_analysis
        import position_sizing

        watchlist = data_layer.load_watchlist()
        portfolio = json.load(open("portfolio.json"))
        held_tickers = set(h["ticker"] for h in portfolio.get("holdings", []))

        # Optional: filter by sector or ticker prefix
        filter_text = args.strip().upper() if args.strip() else ""

        candidates = []
        for entry in watchlist:
            ticker = entry["ticker"]
            if filter_text and filter_text not in ticker and filter_text not in entry.get("sector", "").upper():
                continue
            try:
                df = data_layer.fetch_daily_ohlcv(ticker)
                if df.empty:
                    continue
                tech = technical_analysis.analyze(ticker, df)
                fund = fundamental_analysis.analyze(ticker, entry.get("sector", ""))
                alert = signal_engine.evaluate(tech, fund)
                if alert and alert.direction == "BUY":
                    plan = position_sizing.compute(alert)
                    in_portfolio = " [HELD]" if ticker in held_tickers else ""
                    if plan:
                        candidates.append((
                            alert.signal_score,
                            ticker,
                            tech.current_price,
                            plan.stop_loss,
                            plan.target_1,
                            plan.target_2,
                            alert.signal_score,
                            in_portfolio,
                        ))
                    else:
                        candidates.append((alert.signal_score, ticker, tech.current_price,
                                           0, 0, 0, alert.signal_score, in_portfolio))
            except Exception:
                continue

        if not candidates:
            return "No BUY signals on the watchlist right now."

        candidates.sort(reverse=True)
        lines = ["BUY Opportunities (%d found)" % len(candidates)]
        lines.append("-" * 38)
        for score, ticker, price, stop, t1, t2, _, held in candidates[:8]:
            lines.append("%s%s  $%.2f  score:%d" % (ticker, held, price, score))
            if stop:
                lines.append("  Stop: $%.2f | T1: $%.2f | T2: $%.2f" % (stop, t1, t2))
        lines.append("")
        lines.append("Use /check TICKER for full details.")
        return "\n".join(lines)
    except Exception as e:
        return "Buy scan error: %s" % e


def cmd_risk(args=""):
    """Portfolio risk summary."""
    try:
        import risk_monitor
        report = risk_monitor.analyze_risk()
        lines = ["Risk Summary"]
        lines.append("-" * 30)
        vol = report.get("portfolio_volatility_pct", 0) or report.get("portfolio_volatility", 0)
        lines.append("Volatility: %.1f%%" % float(vol or 0))
        cash_pct = report.get("cash_pct", 0)
        lines.append("Cash: %.1f%%" % float(cash_pct or 0))
        total = report.get("total_value", 0)
        unrealized = report.get("total_unrealized_pnl", 0)
        lines.append("Portfolio: $%s" % "{:,.0f}".format(float(total or 0)))
        lines.append("Unrealized: $%s" % "{:+,.0f}".format(float(unrealized or 0)))

        sector = report.get("sector_exposure", {})
        if sector:
            lines.append("\nSector Weights:")
            for s, pct in sorted(sector.items(), key=lambda x: x[1], reverse=True)[:5]:
                lines.append("  %s: %.1f%%" % (s, float(pct or 0)))

        high_corr = report.get("high_correlation_pairs", [])
        if high_corr:
            lines.append("\nHigh Correlations:")
            for pair in high_corr[:3]:
                lines.append("  %s / %s: %.2f" % (pair[0], pair[1], pair[2]))

        warnings = report.get("warnings", [])
        if warnings:
            lines.append("\nWarnings:")
            for w in warnings[:3]:
                lines.append("  ! %s" % w)

        return "\n".join(lines)
    except Exception as e:
        return "Risk error: %s" % e


def cmd_regime(args=""):
    """Current market regime."""
    try:
        regime_file = Path("logs/market_regime.json")
        if regime_file.exists():
            data = json.load(open(regime_file))
            if isinstance(data, list) and data:
                latest = data[-1]
                return "Market Regime: %s\nConfidence: %d%%\nScore: %d\nUpdated: %s" % (
                    latest.get("regime", "N/A"),
                    latest.get("confidence", 0),
                    latest.get("raw_score", 0),
                    str(latest.get("timestamp", ""))[:16],
                )
        # Run fresh
        import market_regime
        result = market_regime.detect_regime()
        return "Market Regime: %s\nConfidence: %d%%\nDescription: %s" % (
            result["regime"], result["confidence"],
            result.get("params", {}).get("description", ""))
    except Exception as e:
        return "Regime error: %s" % e


def cmd_briefing(args=""):
    """Generate morning briefing."""
    try:
        import daily_briefing
        briefing = daily_briefing.generate_briefing()
        return daily_briefing.format_briefing(briefing)
    except Exception as e:
        return "Briefing error: %s" % e


def cmd_alerts(args=""):
    """Show active price alerts."""
    try:
        import price_alerts
        alerts = price_alerts.get_active_alerts()
        if not alerts:
            return "No active price alerts set.\n\nTo add one:\n/alert TSLA above 400"

        lines = ["Active Price Alerts (%d)" % len(alerts)]
        lines.append("-" * 30)
        for a in alerts:
            note = a.get("note", "")
            note_str = " (%s)" % note if note else ""
            lines.append("%s %s $%.2f%s" % (
                a.get("ticker", ""), a.get("condition", ""),
                float(a.get("price", 0)), note_str))
        return "\n".join(lines)
    except Exception as e:
        return "Alerts error: %s" % e


def cmd_earnings(args=""):
    """Upcoming earnings for your holdings."""
    try:
        import earnings_calendar
        earnings = earnings_calendar.get_portfolio_earnings()
        return earnings_calendar.format_earnings_report(earnings)
    except Exception as e:
        return "Earnings error: %s" % e


def cmd_news(args=""):
    """Latest significant news for your holdings."""
    try:
        import news_monitor
        hours = 24
        try:
            hours = int(args.strip()) if args.strip() else 24
        except Exception:
            pass
        items = news_monitor.fetch_portfolio_news(max_age_hours=hours)
        significant = news_monitor.filter_significant(items)
        if not significant:
            all_items = items[:10]
            if not all_items:
                return "No news in the last %d hours." % hours
            return news_monitor.format_news_report(all_items)
        return news_monitor.format_news_report(significant)
    except Exception as e:
        return "News error: %s" % e


def cmd_dca(args=""):
    """DCA suggestions for long-term positions that are down."""
    try:
        import dca_advisor
        ticker = args.strip().upper() if args.strip() else None
        if ticker:
            portfolio = json.load(open("portfolio.json"))
            holdings = [h for h in portfolio.get("holdings", []) if h["ticker"] == ticker]
            if not holdings:
                return "%s not in your portfolio." % ticker
            import data_layer
            df = data_layer.fetch_daily_ohlcv(ticker)
            result = dca_advisor.analyze_dca(holdings[0], df)
            return dca_advisor.format_report([result])
        else:
            dca_list = dca_advisor.analyze_portfolio_dca()
            return dca_advisor.format_report(dca_list)
    except Exception as e:
        return "DCA error: %s" % e


def cmd_tax(args=""):
    """Tax loss harvesting opportunities."""
    try:
        import tax_harvesting
        portfolio = json.load(open("portfolio.json"))
        analysis = tax_harvesting.analyze_harvesting(portfolio)
        return tax_harvesting.format_report(analysis)
    except Exception as e:
        return "Tax error: %s" % e


def cmd_rotation(args=""):
    """Sector rotation — where is money flowing?"""
    try:
        import sector_rotation
        perf = sector_rotation.fetch_sector_performance()
        rotation = sector_rotation.detect_rotation(perf)
        portfolio = json.load(open("portfolio.json"))
        exposure = sector_rotation.get_portfolio_exposure(rotation, portfolio)
        return sector_rotation.format_report(perf, rotation, exposure)
    except Exception as e:
        return "Rotation error: %s" % e


def cmd_weekly(args=""):
    """Generate weekly portfolio report."""
    try:
        import weekly_report
        report = weekly_report.generate_weekly_report()
        return weekly_report.format_report(report)
    except Exception as e:
        return "Weekly report error: %s" % e


def cmd_ibsync(args=""):
    """Sync portfolio from Interactive Brokers."""
    try:
        import ib_sync
        if not ib_sync.is_available():
            return "IB sync not available. Make sure TWS or IB Gateway is running and ib_insync is installed."
        result = ib_sync.sync_portfolio(dry_run="--dry-run" in args)
        if result.get("synced"):
            return "IB Sync complete!\nPositions: %d\nNet Liq: $%s\nCash: $%s" % (
                result.get("positions", 0),
                "{:,.0f}".format(result.get("net_liq", 0)),
                "{:,.0f}".format(result.get("cash", 0)),
            )
        return "IB Sync failed: %s" % result.get("error", "Unknown error")
    except Exception as e:
        return "IB sync error: %s" % e


def cmd_volume(args=""):
    """Scan for unusual volume spikes."""
    try:
        import volume_scanner
        threshold = 2.0
        if args.strip():
            try:
                threshold = float(args.strip())
            except ValueError:
                pass
        spikes = volume_scanner.scan_volume_spikes(threshold=threshold)
        return volume_scanner.format_report(spikes)
    except Exception as e:
        return "Volume scan error: %s" % e


def cmd_upgrades(args=""):
    """Check analyst upgrades/downgrades."""
    try:
        import upgrade_tracker
        days = 7
        if args.strip():
            try:
                days = int(args.strip())
            except ValueError:
                pass
        changes = upgrade_tracker.scan_all_recommendations(days=days)
        return upgrade_tracker.format_report(changes)
    except Exception as e:
        return "Upgrade tracker error: %s" % e


def cmd_performance(args=""):
    """Portfolio performance history and stats."""
    try:
        import performance_tracker
        days = 30
        if args.strip():
            try:
                days = int(args.strip())
            except ValueError:
                pass
        history = performance_tracker.get_history(days=days)
        if not history:
            # Try taking a snapshot first
            performance_tracker.take_snapshot()
            history = performance_tracker.get_history(days=days)
        if not history:
            return "No performance history yet. Run /perf again after market close."
        stats = performance_tracker.compute_stats(history)
        report = performance_tracker.format_report(stats, history)
        chart = performance_tracker.format_chart(history)
        return report + "\n\n" + chart
    except Exception as e:
        return "Performance error: %s" % e


def cmd_options(args=""):
    """Options strategy analysis for a ticker."""
    ticker = args.strip().upper() if args.strip() else ""
    if not ticker:
        return "Usage: /options TSLA\n\nAnalyzes the options chain and suggests a strategy."
    try:
        import options_analyzer
        result = options_analyzer.analyze_ticker_options(ticker, "BUY")
        return options_analyzer.format_options_report(result)
    except Exception as e:
        return "Options error: %s" % e


def cmd_grade(args=""):
    """Grade a ticker — full analysis with grade, logic, risk, and options."""
    ticker = args.strip().upper() if args.strip() else ""
    if not ticker:
        return "Usage: /grade TSLA\n\nFull graded analysis with logic, risks, and options strategy."
    try:
        import data_layer
        import technical_analysis
        import fundamental_analysis
        import signal_engine
        import position_sizing
        import trade_grader
        import options_analyzer

        df = data_layer.fetch_daily_ohlcv(ticker)
        if df.empty:
            return "No data for %s" % ticker

        tech = technical_analysis.analyze(ticker, df)
        fund = fundamental_analysis.analyze(ticker, "Technology")
        alert = signal_engine.evaluate(tech, fund, threshold=1)  # Low threshold to always get a grade

        if not alert:
            # Still generate a basic analysis even without a signal
            lines = ["%s — Grade: N/A (no signal)" % ticker]
            lines.append("Price: $%.2f" % tech.current_price)
            lines.append("RSI: %.1f | MACD: %.2f" % (tech.rsi, tech.macd_value))
            lines.append("Above 200 SMA: %s" % tech.price_above_200sma)
            lines.append("Fundamental: %d/15" % fund.fundamental_score)
            lines.append("\nNo confluence signal — waiting for setup.")
            return "\n".join(lines)

        plan = position_sizing.compute(alert)
        if not plan:
            return "%s signal detected but could not size position." % ticker

        grade_info = trade_grader.grade_trade(alert, plan)
        options_suggestion = None
        try:
            options_suggestion = options_analyzer.analyze_ticker_options(
                ticker, alert.direction, alert.signal_score
            )
        except Exception:
            pass

        return trade_grader.format_graded_alert(alert, plan, grade_info, options_suggestion)
    except Exception as e:
        return "Grade error: %s" % e


def cmd_help(args=""):
    """List available commands."""
    return (
        "Stock Agent Commands:\n"
        "─────────────────────\n"
        "/buy       — BUY opportunities (watchlist scan)\n"
        "/sell      — Which holdings to sell?\n"
        "/grade TSLA — Full graded analysis + options\n"
        "/options TSLA — Options strategy suggestion\n"
        "/check TSLA — Analyze any ticker\n"
        "/scan      — Quick scan (holdings + top buys)\n"
        "/status    — Portfolio summary\n"
        "/positions — All open positions\n"
        "/perf      — Performance history + stats\n"
        "/risk      — Risk summary\n"
        "/regime    — Market regime\n"
        "/briefing  — Morning briefing\n"
        "/earnings  — Upcoming earnings\n"
        "/news      — Portfolio news\n"
        "/volume    — Unusual volume spikes\n"
        "/upgrades  — Analyst upgrades/downgrades\n"
        "/dca       — DCA suggestions\n"
        "/tax       — Tax harvesting\n"
        "/rotation  — Sector rotation\n"
        "/weekly    — Weekly report\n"
        "/ibsync    — Sync from IB\n"
        "/alerts    — Price alerts\n"
        "/help      — This message\n"
        "\n"
        "Tip: Just type a ticker (e.g. NVDA) to check it.\n"
        "All alerts include grade, logic, risk, and options play."
    )


COMMANDS = {
    "/status": cmd_status,
    "/positions": cmd_positions,
    "/pos": cmd_positions,
    "/sell": cmd_sell,
    "/buy": cmd_buy,
    "/check": cmd_check,
    "/scan": cmd_scan,
    "/risk": cmd_risk,
    "/regime": cmd_regime,
    "/briefing": cmd_briefing,
    "/alerts": cmd_alerts,
    "/earnings": cmd_earnings,
    "/news": cmd_news,
    "/dca": cmd_dca,
    "/tax": cmd_tax,
    "/rotation": cmd_rotation,
    "/weekly": cmd_weekly,
    "/ibsync": cmd_ibsync,
    "/volume": cmd_volume,
    "/upgrades": cmd_upgrades,
    "/performance": cmd_performance,
    "/perf": cmd_performance,
    "/options": cmd_options,
    "/grade": cmd_grade,
    "/help": cmd_help,
    "/start": cmd_help,
}


def handle_message(text, chat_id):
    """Process an incoming message and respond."""
    text = text.strip()
    if not text.startswith("/"):
        # Natural language — single ticker treated as check
        words = text.upper().split()
        if len(words) == 1 and words[0].isalpha() and 1 <= len(words[0]) <= 5:
            return cmd_check(words[0])
        # Route to AI trade advisor for natural language questions
        try:
            import trade_advisor
            answer = trade_advisor.answer_question(text)
            if answer:
                return answer
        except Exception as e:
            return "Sorry, I couldn't process that question.\nError: %s\n\nTry /help for commands." % str(e)
        return cmd_help()

    parts = text.split(maxsplit=1)
    command = parts[0].lower().split("@")[0]  # Remove @botname suffix
    args = parts[1] if len(parts) > 1 else ""

    handler = COMMANDS.get(command)
    if handler:
        return handler(args)
    return "Unknown command: %s\n\nType /help for available commands." % command


def run_bot():
    """Main bot loop — long polls for messages."""
    if not TOKEN:
        print("ERROR: TELEGRAM_BOT_TOKEN not set in .env")
        print("1. Message @BotFather on Telegram")
        print("2. Send /newbot and follow instructions")
        print("3. Copy the token to .env")
        sys.exit(1)

    if not CHAT_ID:
        print("WARNING: TELEGRAM_CHAT_ID not set — bot will respond to anyone")
        print("Run: python3 telegram_bot.py --get-chat-id")

    print("Stock Agent Telegram Bot starting...")
    print("Listening for commands...")

    offset = None
    while True:
        try:
            updates = get_updates(offset=offset, timeout=30)
            for update in updates:
                offset = update["update_id"] + 1
                msg = update.get("message", {})
                text = msg.get("text", "")
                chat_id = msg.get("chat", {}).get("id")

                if not text or not chat_id:
                    continue

                # Security: only respond to authorized chat
                if CHAT_ID and str(chat_id) != str(CHAT_ID):
                    send_message("Unauthorized. Your chat ID: %s" % chat_id, chat_id=chat_id)
                    continue

                logger.info("Received: %s", text)

                try:
                    response = handle_message(text, chat_id)
                    send_message(response, chat_id=chat_id)
                except Exception as e:
                    send_message("Error: %s" % str(e)[:200], chat_id=chat_id)
                    logger.error("Handler error: %s", traceback.format_exc())

        except KeyboardInterrupt:
            print("\nBot stopped.")
            break
        except Exception as e:
            logger.error("Bot loop error: %s", e)
            # Exponential backoff for network issues (e.g. after sleep/wake)
            import random
            wait = min(5 + random.randint(0, 5), 30)
            logger.info("Retrying in %d seconds...", wait)
            time.sleep(wait)


def main():
    parser = argparse.ArgumentParser(description="Stock Agent Telegram Bot")
    parser.add_argument("--get-chat-id", action="store_true", help="Find your Telegram chat ID")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    # Change to project directory
    os.chdir(Path(__file__).parent)

    if args.get_chat_id:
        get_chat_id()
    else:
        run_bot()


if __name__ == "__main__":
    main()
