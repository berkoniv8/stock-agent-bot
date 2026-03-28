"""Tests for notification layer."""

import csv
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

from technical_analysis import TechnicalSignals
from fundamental_analysis import FundamentalSignals
from signal_engine import TradeAlert
from position_sizing import PositionPlan
import notifications


def _make_alert():
    tech = TechnicalSignals(
        ticker="AAPL", current_price=170.0, ema9=168.0, ema21=165.0,
        sma200=150.0, rsi=55.0, macd_value=1.2,
        bb_lower=160.0, bb_upper=180.0, pattern_details=["breakout"],
    )
    fund = FundamentalSignals(
        ticker="AAPL", fundamental_score=4,
        score_details=["P/E 25 below sector median 30"],
    )
    return TradeAlert(
        ticker="AAPL", signal_score=7, direction="BUY",
        triggered_signals=[("ema_cross_bullish", 2), ("breakout_with_volume", 3)],
        technical=tech, fundamental=fund,
    )


def _make_plan():
    return PositionPlan(
        ticker="AAPL", direction="BUY", entry_price=170.0,
        stop_loss=165.0, risk_per_share=5.0, shares=20,
        position_value=3400.0, target_1=177.5, target_2=185.0,
        target_3=190.0, max_loss=100.0, risk_reward_t1=1.5,
        risk_reward_t2=3.0,
    )


class TestFormatAlertText(unittest.TestCase):
    def test_contains_key_info(self):
        text = notifications.format_alert_text(_make_alert(), _make_plan())
        self.assertIn("BUY", text)
        self.assertIn("AAPL", text)
        self.assertIn("170.00", text)
        self.assertIn("165.00", text)
        self.assertIn("ema_cross_bullish", text)

    def test_sell_alert(self):
        alert = _make_alert()
        alert.direction = "SELL"
        plan = _make_plan()
        plan.direction = "SELL"
        text = notifications.format_alert_text(alert, plan)
        self.assertIn("SELL", text)


class TestFormatAlertHtml(unittest.TestCase):
    def test_contains_html_tags(self):
        html = notifications.format_alert_html(_make_alert(), _make_plan())
        self.assertIn("<div", html)
        self.assertIn("AAPL", html)
        self.assertIn("170.00", html)


class TestSendEmail(unittest.TestCase):
    @patch.dict(os.environ, {"SMTP_HOST": "", "SMTP_USER": ""})
    def test_skips_when_not_configured(self):
        result = notifications.send_email(_make_alert(), _make_plan())
        self.assertFalse(result)

    @patch.dict(os.environ, {
        "SMTP_HOST": "smtp.test.com", "SMTP_PORT": "587",
        "SMTP_USER": "user@test.com", "SMTP_PASSWORD": "pass",
        "ALERT_EMAIL_TO": "alert@test.com",
    })
    @patch("notifications.smtplib.SMTP")
    def test_sends_when_configured(self, mock_smtp):
        mock_server = MagicMock()
        mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_server)
        mock_smtp.return_value.__exit__ = MagicMock(return_value=False)
        result = notifications.send_email(_make_alert(), _make_plan())
        self.assertTrue(result)


class TestSendSlack(unittest.TestCase):
    @patch.dict(os.environ, {"SLACK_WEBHOOK_URL": ""})
    def test_skips_when_not_configured(self):
        result = notifications.send_slack(_make_alert(), _make_plan())
        self.assertFalse(result)

    @patch.dict(os.environ, {"SLACK_WEBHOOK_URL": "https://hooks.slack.com/services/REAL/TOKEN"})
    @patch("notifications.requests.post")
    def test_sends_when_configured(self, mock_post):
        mock_post.return_value.raise_for_status = MagicMock()
        result = notifications.send_slack(_make_alert(), _make_plan())
        self.assertTrue(result)
        mock_post.assert_called_once()


class TestSendTelegram(unittest.TestCase):
    @patch.dict(os.environ, {"TELEGRAM_BOT_TOKEN": "", "TELEGRAM_CHAT_ID": ""})
    def test_skips_when_not_configured(self):
        result = notifications.send_telegram(_make_alert(), _make_plan())
        self.assertFalse(result)


class TestSendDiscord(unittest.TestCase):
    @patch.dict(os.environ, {"DISCORD_WEBHOOK_URL": ""})
    def test_skips_when_not_configured(self):
        result = notifications.send_discord(_make_alert(), _make_plan())
        self.assertFalse(result)

    @patch.dict(os.environ, {"DISCORD_WEBHOOK_URL": "https://discord.com/api/webhooks/123/abc"})
    @patch("notifications.requests.post")
    def test_sends_with_embed(self, mock_post):
        mock_post.return_value.raise_for_status = MagicMock()
        result = notifications.send_discord(_make_alert(), _make_plan())
        self.assertTrue(result)
        call_args = mock_post.call_args
        payload = call_args[1]["json"] if "json" in call_args[1] else call_args[0][1]
        self.assertIn("embeds", payload)


class TestSendWebhook(unittest.TestCase):
    @patch.dict(os.environ, {"CUSTOM_WEBHOOK_URL": ""})
    def test_skips_when_not_configured(self):
        result = notifications.send_webhook(_make_alert(), _make_plan())
        self.assertFalse(result)

    @patch.dict(os.environ, {
        "CUSTOM_WEBHOOK_URL": "https://example.com/hook",
        "CUSTOM_WEBHOOK_AUTH": "Bearer token123",
    })
    @patch("notifications.requests.post")
    def test_sends_with_auth_header(self, mock_post):
        mock_post.return_value.raise_for_status = MagicMock()
        result = notifications.send_webhook(_make_alert(), _make_plan())
        self.assertTrue(result)
        call_args = mock_post.call_args
        headers = call_args[1].get("headers", {})
        self.assertEqual(headers.get("Authorization"), "Bearer token123")


class TestLogToDashboard(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
        self._tmp.close()
        self._orig = notifications.DASHBOARD_LOG
        notifications.DASHBOARD_LOG = Path(self._tmp.name)
        # Remove so it creates fresh
        os.unlink(self._tmp.name)

    def tearDown(self):
        notifications.DASHBOARD_LOG = self._orig
        try:
            os.unlink(self._tmp.name)
        except OSError:
            pass

    @patch.dict(os.environ, {"USE_SQLITE": "0"})
    def test_creates_csv(self):
        notifications.log_to_dashboard(_make_alert(), _make_plan())
        self.assertTrue(Path(self._tmp.name).exists())
        with open(self._tmp.name, newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["ticker"], "AAPL")
        self.assertEqual(rows[0]["direction"], "BUY")

    @patch.dict(os.environ, {"USE_SQLITE": "0"})
    def test_appends_multiple(self):
        notifications.log_to_dashboard(_make_alert(), _make_plan())
        notifications.log_to_dashboard(_make_alert(), _make_plan())
        with open(self._tmp.name, newline="") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 2)


if __name__ == "__main__":
    unittest.main()
