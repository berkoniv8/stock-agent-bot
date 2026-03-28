"""
Notification Layer — sends trade alerts via Email, Slack, Telegram,
and logs to a local dashboard CSV.
"""

import csv
import json
import logging
import os
import smtplib
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from pathlib import Path

import requests
from dotenv import load_dotenv

from position_sizing import PositionPlan
from signal_engine import TradeAlert

load_dotenv()
logger = logging.getLogger(__name__)

LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)
DASHBOARD_LOG = LOGS_DIR / "dashboard.csv"


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def format_alert_text(alert: TradeAlert, plan: PositionPlan) -> str:
    """Build a human-readable alert message."""
    signals_str = ", ".join(f"{s[0]} (+{s[1]})" for s in alert.triggered_signals)
    fund_details = ""
    if alert.fundamental:
        fund_details = (
            f"\n  Fundamental score: {alert.fundamental.fundamental_score}/6"
            f"\n  Details: {', '.join(alert.fundamental.score_details) or 'N/A'}"
        )

    tech_details = ""
    if alert.technical:
        t = alert.technical
        tech_details = (
            f"\n  EMA9: {t.ema9:.2f}  |  EMA21: {t.ema21:.2f}  |  SMA200: {t.sma200:.2f}"
            f"\n  RSI: {t.rsi:.1f}  |  MACD: {t.macd_value:.2f}  |  BB: [{t.bb_lower:.2f}–{t.bb_upper:.2f}]"
            f"\n  Patterns: {', '.join(t.pattern_details) or 'none'}"
        )

    text = (
        f"{'=' * 55}\n"
        f"  TRADE ALERT: {alert.direction} {alert.ticker}\n"
        f"{'=' * 55}\n"
        f"  Signal score: {alert.signal_score} (threshold met)\n"
        f"  Signals: {signals_str}\n"
        f"{tech_details}"
        f"{fund_details}\n"
        f"\n"
        f"  POSITION PLAN\n"
        f"  {'─' * 40}\n"
        f"  Entry price:    ${plan.entry_price:,.2f}\n"
        f"  Stop-loss:      ${plan.stop_loss:,.2f}\n"
        f"  Risk/share:     ${plan.risk_per_share:,.2f}\n"
        f"  Shares:         {plan.shares}\n"
        f"  Position value: ${plan.position_value:,.2f}\n"
        f"  Max loss:       ${plan.max_loss:,.2f}\n"
        f"\n"
        f"  TARGETS\n"
        f"  T1 (1.5:1 R/R): ${plan.target_1:,.2f}\n"
        f"  T2 (3.0:1 R/R): ${plan.target_2:,.2f}\n"
        f"  T3 (Fib ext):   ${plan.target_3:,.2f}\n"
        f"{'=' * 55}\n"
    )
    return text


def format_alert_html(alert: TradeAlert, plan: PositionPlan) -> str:
    """Build an HTML version for email."""
    color = "#2e7d32" if alert.direction == "BUY" else "#c62828"
    signals_str = ", ".join(f"{s[0]} (+{s[1]})" for s in alert.triggered_signals)

    html = f"""
    <div style="font-family: Arial, sans-serif; max-width: 500px; margin: 0 auto;">
      <div style="background: {color}; color: white; padding: 16px; border-radius: 8px 8px 0 0;">
        <h2 style="margin: 0;">{alert.direction} {alert.ticker}</h2>
        <p style="margin: 4px 0 0;">Signal score: {alert.signal_score}</p>
      </div>
      <div style="border: 1px solid #ddd; padding: 16px; border-radius: 0 0 8px 8px;">
        <p><strong>Signals:</strong> {signals_str}</p>
        <table style="width: 100%; border-collapse: collapse;">
          <tr><td style="padding: 4px 0;"><strong>Entry</strong></td><td>${plan.entry_price:,.2f}</td></tr>
          <tr><td style="padding: 4px 0;"><strong>Stop-loss</strong></td><td>${plan.stop_loss:,.2f}</td></tr>
          <tr><td style="padding: 4px 0;"><strong>Shares</strong></td><td>{plan.shares}</td></tr>
          <tr><td style="padding: 4px 0;"><strong>Position</strong></td><td>${plan.position_value:,.2f}</td></tr>
          <tr><td style="padding: 4px 0;"><strong>Max loss</strong></td><td>${plan.max_loss:,.2f}</td></tr>
          <tr style="border-top: 1px solid #eee;"><td style="padding: 4px 0;"><strong>T1 (1.5:1)</strong></td><td>${plan.target_1:,.2f}</td></tr>
          <tr><td style="padding: 4px 0;"><strong>T2 (3:1)</strong></td><td>${plan.target_2:,.2f}</td></tr>
          <tr><td style="padding: 4px 0;"><strong>T3 (Fib)</strong></td><td>${plan.target_3:,.2f}</td></tr>
        </table>
      </div>
    </div>
    """
    return html


# ---------------------------------------------------------------------------
# Email notification
# ---------------------------------------------------------------------------

def send_email(alert: TradeAlert, plan: PositionPlan) -> bool:
    """Send alert via SMTP email."""
    host = os.getenv("SMTP_HOST", "")
    port = int(os.getenv("SMTP_PORT", "587"))
    user = os.getenv("SMTP_USER", "")
    password = os.getenv("SMTP_PASSWORD", "")
    to_addr = os.getenv("ALERT_EMAIL_TO", "")

    if not all([host, user, password, to_addr]) or user.startswith("your_"):
        logger.info("Email not configured — skipping")
        return False

    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = f"Trade Alert: {alert.direction} {alert.ticker} (score {alert.signal_score})"
        msg["From"] = user
        msg["To"] = to_addr

        msg.attach(MIMEText(format_alert_text(alert, plan), "plain"))
        msg.attach(MIMEText(format_alert_html(alert, plan), "html"))

        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(user, password)
            server.sendmail(user, [to_addr], msg.as_string())

        logger.info("Email sent for %s to %s", alert.ticker, to_addr)
        return True
    except Exception as e:
        logger.error("Email send failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# Slack notification
# ---------------------------------------------------------------------------

def send_slack(alert: TradeAlert, plan: PositionPlan) -> bool:
    """Send alert to Slack via webhook."""
    webhook_url = os.getenv("SLACK_WEBHOOK_URL", "")
    if not webhook_url or webhook_url.startswith("https://hooks.slack.com/services/YOUR"):
        logger.info("Slack webhook not configured — skipping")
        return False

    text = format_alert_text(alert, plan)
    payload = {
        "channel": os.getenv("SLACK_CHANNEL", "#stock-alerts"),
        "username": "Stock Agent",
        "text": f"```{text}```",
        "icon_emoji": ":chart_with_upwards_trend:",
    }

    try:
        resp = requests.post(webhook_url, json=payload, timeout=10)
        resp.raise_for_status()
        logger.info("Slack notification sent for %s", alert.ticker)
        return True
    except Exception as e:
        logger.error("Slack send failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# Telegram notification
# ---------------------------------------------------------------------------

def send_telegram(alert: TradeAlert, plan: PositionPlan) -> bool:
    """Send alert to Telegram."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "")
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "")

    if not token or not chat_id or token.startswith("your_"):
        logger.info("Telegram not configured — skipping")
        return False

    text = format_alert_text(alert, plan)
    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "Markdown",
    }

    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        logger.info("Telegram notification sent for %s", alert.ticker)
        return True
    except Exception as e:
        logger.error("Telegram send failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# Discord notification
# ---------------------------------------------------------------------------

def send_discord(alert: TradeAlert, plan: PositionPlan) -> bool:
    """Send alert to Discord via webhook."""
    webhook_url = os.getenv("DISCORD_WEBHOOK_URL", "")
    if not webhook_url or "discord.com/api/webhooks" not in webhook_url:
        logger.info("Discord webhook not configured — skipping")
        return False

    color = 0x2E7D32 if alert.direction == "BUY" else 0xC62828
    signals_str = ", ".join(s[0] for s in alert.triggered_signals)

    embed = {
        "title": "%s %s — Score %d" % (alert.direction, alert.ticker, alert.signal_score),
        "color": color,
        "fields": [
            {"name": "Signals", "value": signals_str, "inline": False},
            {"name": "Entry", "value": "$%.2f" % plan.entry_price, "inline": True},
            {"name": "Stop", "value": "$%.2f" % plan.stop_loss, "inline": True},
            {"name": "Shares", "value": str(plan.shares), "inline": True},
            {"name": "T1 (1.5:1)", "value": "$%.2f" % plan.target_1, "inline": True},
            {"name": "T2 (3:1)", "value": "$%.2f" % plan.target_2, "inline": True},
            {"name": "Max Loss", "value": "$%.2f" % plan.max_loss, "inline": True},
        ],
        "footer": {"text": "Stock Agent"},
        "timestamp": datetime.utcnow().isoformat(),
    }

    payload = {
        "username": "Stock Agent",
        "embeds": [embed],
    }

    try:
        resp = requests.post(webhook_url, json=payload, timeout=10)
        resp.raise_for_status()
        logger.info("Discord notification sent for %s", alert.ticker)
        return True
    except Exception as e:
        logger.error("Discord send failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# Generic webhook notification
# ---------------------------------------------------------------------------

def send_sms(alert: TradeAlert, plan: PositionPlan) -> bool:
    """Send alert via SMS using email-to-SMS gateway."""
    gateway = os.getenv("SMS_GATEWAY", "")
    host = os.getenv("SMTP_HOST", "")
    user = os.getenv("SMTP_USER", "")
    password = os.getenv("SMTP_PASSWORD", "")

    if not gateway or not all([host, user, password]) or user.startswith("your_") or password.startswith("your_"):
        logger.info("SMS not configured — skipping")
        return False

    # SMS is short — compact format
    signals = ", ".join(s[0] for s in alert.triggered_signals[:3])
    text = (
        f"{alert.direction} {alert.ticker} (score {alert.signal_score})\n"
        f"Entry: ${plan.entry_price:,.2f}\n"
        f"Stop: ${plan.stop_loss:,.2f}\n"
        f"Shares: {plan.shares}\n"
        f"T1: ${plan.target_1:,.2f}\n"
        f"Signals: {signals}"
    )

    try:
        port = int(os.getenv("SMTP_PORT", "587"))
        msg = MIMEText(text)
        msg["From"] = user
        msg["To"] = gateway
        msg["Subject"] = f"{alert.direction} {alert.ticker}"

        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(user, password)
            server.sendmail(user, [gateway], msg.as_string())

        logger.info("SMS sent for %s to %s", alert.ticker, gateway)
        return True
    except Exception as e:
        logger.error("SMS send failed: %s", e)
        return False


def send_sms_text(text: str, subject: str = "Stock Agent") -> bool:
    """Send a plain text SMS message (for sell alerts, briefings, etc.)."""
    gateway = os.getenv("SMS_GATEWAY", "")
    host = os.getenv("SMTP_HOST", "")
    user = os.getenv("SMTP_USER", "")
    password = os.getenv("SMTP_PASSWORD", "")

    if not gateway or not all([host, user, password]) or user.startswith("your_") or password.startswith("your_"):
        return False

    try:
        port = int(os.getenv("SMTP_PORT", "587"))
        msg = MIMEText(text[:160])  # SMS limit
        msg["From"] = user
        msg["To"] = gateway
        msg["Subject"] = subject

        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(user, password)
            server.sendmail(user, [gateway], msg.as_string())

        logger.info("SMS text sent: %s", subject)
        return True
    except Exception as e:
        logger.error("SMS text send failed: %s", e)
        return False


def send_email_text(text: str, subject: str = "Stock Agent", html: str = None) -> bool:
    """Send a plain text/HTML email (for sell alerts, briefings, etc.)."""
    host = os.getenv("SMTP_HOST", "")
    user = os.getenv("SMTP_USER", "")
    password = os.getenv("SMTP_PASSWORD", "")
    to_addr = os.getenv("ALERT_EMAIL_TO", "")

    if not all([host, user, password, to_addr]) or user.startswith("your_") or password.startswith("your_"):
        return False

    try:
        port = int(os.getenv("SMTP_PORT", "587"))
        msg = MIMEMultipart("alternative")
        msg["Subject"] = subject
        msg["From"] = user
        msg["To"] = to_addr

        msg.attach(MIMEText(text, "plain"))
        if html:
            msg.attach(MIMEText(html, "html"))

        with smtplib.SMTP(host, port) as server:
            server.starttls()
            server.login(user, password)
            server.sendmail(user, [to_addr], msg.as_string())

        logger.info("Email sent: %s", subject)
        return True
    except Exception as e:
        logger.error("Email send failed: %s", e)
        return False


def send_webhook(alert: TradeAlert, plan: PositionPlan) -> bool:
    """Send alert to a generic webhook URL as JSON.

    The payload includes all alert and plan details so any downstream
    system (Zapier, IFTTT, n8n, custom server) can consume it.
    """
    webhook_url = os.getenv("CUSTOM_WEBHOOK_URL", "")
    if not webhook_url:
        return False

    signals_str = [s[0] for s in alert.triggered_signals]
    fund_score = alert.fundamental.fundamental_score if alert.fundamental else 0

    payload = {
        "event": "trade_alert",
        "timestamp": datetime.utcnow().isoformat(),
        "ticker": alert.ticker,
        "direction": alert.direction,
        "signal_score": alert.signal_score,
        "triggered_signals": signals_str,
        "fundamental_score": fund_score,
        "plan": {
            "entry_price": plan.entry_price,
            "stop_loss": plan.stop_loss,
            "risk_per_share": plan.risk_per_share,
            "shares": plan.shares,
            "position_value": plan.position_value,
            "max_loss": plan.max_loss,
            "target_1": plan.target_1,
            "target_2": plan.target_2,
            "target_3": plan.target_3,
        },
    }

    # Support custom headers (e.g., API keys)
    headers = {"Content-Type": "application/json"}
    auth_header = os.getenv("CUSTOM_WEBHOOK_AUTH", "")
    if auth_header:
        headers["Authorization"] = auth_header

    try:
        resp = requests.post(webhook_url, json=payload, headers=headers, timeout=15)
        resp.raise_for_status()
        logger.info("Webhook notification sent for %s to %s", alert.ticker, webhook_url)
        return True
    except Exception as e:
        logger.error("Webhook send failed: %s", e)
        return False


# ---------------------------------------------------------------------------
# Dashboard log (CSV)
# ---------------------------------------------------------------------------

def log_to_dashboard(alert: TradeAlert, plan: PositionPlan) -> None:
    """Append alert to the local dashboard CSV log and optionally to SQLite."""
    signals_str = "|".join(s[0] for s in alert.triggered_signals)
    fund_score = alert.fundamental.fundamental_score if alert.fundamental else 0
    news_sent = alert.fundamental.news_sentiment_score if alert.fundamental else 0

    # Log to SQLite if enabled
    if os.getenv("USE_SQLITE", "0") == "1":
        try:
            import database as db
            db.insert_alert(
                ticker=alert.ticker, direction=alert.direction,
                signal_score=alert.signal_score, signals=signals_str,
                entry_price=plan.entry_price, stop_loss=plan.stop_loss,
                shares=plan.shares, position_value=plan.position_value,
                max_loss=plan.max_loss, target_1=plan.target_1,
                target_2=plan.target_2, target_3=plan.target_3,
                fundamental_score=fund_score, news_sentiment=news_sent,
            )
        except Exception as e:
            logger.debug("DB alert insert fallback: %s", e)

    # Always write CSV too (backward compat)
    file_exists = DASHBOARD_LOG.exists()
    row = {
        "timestamp": datetime.now().isoformat(),
        "ticker": alert.ticker,
        "direction": alert.direction,
        "signal_score": alert.signal_score,
        "signals": signals_str,
        "entry_price": plan.entry_price,
        "stop_loss": plan.stop_loss,
        "shares": plan.shares,
        "position_value": plan.position_value,
        "max_loss": plan.max_loss,
        "target_1": plan.target_1,
        "target_2": plan.target_2,
        "target_3": plan.target_3,
        "fundamental_score": fund_score,
        "news_sentiment": news_sent,
    }

    fieldnames = list(row.keys())
    with open(DASHBOARD_LOG, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    logger.info("Dashboard log updated for %s", alert.ticker)


# ---------------------------------------------------------------------------
# Send all notifications
# ---------------------------------------------------------------------------

def notify(alert: TradeAlert, plan: PositionPlan) -> None:
    """Dispatch alert through all configured channels with grade + options."""
    # Always log to dashboard
    log_to_dashboard(alert, plan)

    # Grade the trade and generate options suggestion
    graded_text = None
    try:
        import trade_grader
        import options_analyzer

        grade_info = trade_grader.grade_trade(alert, plan)
        options_suggestion = None
        try:
            options_suggestion = options_analyzer.analyze_ticker_options(
                alert.ticker, alert.direction, alert.signal_score
            )
        except Exception as e:
            logger.debug("Options analysis failed for %s: %s", alert.ticker, e)

        graded_text = trade_grader.format_graded_alert(
            alert, plan, grade_info, options_suggestion
        )
    except Exception as e:
        logger.debug("Trade grading failed for %s: %s", alert.ticker, e)

    # Print to console (graded version if available)
    if graded_text:
        print(graded_text)
    else:
        print(format_alert_text(alert, plan))

    # Send graded alert via Telegram (richer format)
    if graded_text:
        try:
            import telegram_bot
            telegram_bot.send_message(graded_text)
        except Exception as e:
            logger.debug("Telegram graded alert failed: %s", e)
            send_telegram(alert, plan)
    else:
        send_telegram(alert, plan)

    # Send through other configured channels (standard format)
    send_email(alert, plan)
    send_sms(alert, plan)
    send_slack(alert, plan)
    send_discord(alert, plan)
    send_webhook(alert, plan)
