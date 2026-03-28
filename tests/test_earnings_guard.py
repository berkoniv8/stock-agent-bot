"""Tests for earnings calendar guard."""

import unittest
from datetime import datetime, timedelta
from unittest.mock import patch

import earnings_guard


class TestParseDate(unittest.TestCase):
    def test_datetime_passthrough(self):
        """datetime should pass through."""
        dt = datetime(2026, 4, 15, 10, 0)
        result = earnings_guard._parse_date(dt)
        self.assertEqual(result, dt)

    def test_string_ymd(self):
        """YYYY-MM-DD string should parse."""
        result = earnings_guard._parse_date("2026-04-15")
        self.assertEqual(result, datetime(2026, 4, 15))

    def test_none_returns_none(self):
        """None should return None."""
        self.assertIsNone(earnings_guard._parse_date(None))

    def test_invalid_string(self):
        """Invalid string should return None."""
        self.assertIsNone(earnings_guard._parse_date("not a date"))


class TestCheckEarningsSafe(unittest.TestCase):
    @patch("earnings_guard.get_next_earnings")
    def test_no_earnings_date_allows_entry(self, mock_earnings):
        """No earnings date found should allow entry."""
        mock_earnings.return_value = None
        safe, info = earnings_guard.check_earnings_safe("TEST")
        self.assertTrue(safe)

    @patch("earnings_guard.get_next_earnings")
    def test_distant_earnings_allows_entry(self, mock_earnings):
        """Earnings far away should allow entry."""
        mock_earnings.return_value = datetime.now() + timedelta(days=30)
        safe, info = earnings_guard.check_earnings_safe("TEST", blackout_days=3)
        self.assertTrue(safe)
        self.assertGreater(info["days_until"], 3)

    @patch("earnings_guard.get_next_earnings")
    def test_near_earnings_blocks_entry(self, mock_earnings):
        """Earnings within blackout should block entry."""
        mock_earnings.return_value = datetime.now() + timedelta(days=1)
        safe, info = earnings_guard.check_earnings_safe("TEST", blackout_days=3)
        self.assertFalse(safe)
        self.assertIn("blackout", info["reason"])

    @patch("earnings_guard.get_next_earnings")
    def test_earnings_today_blocks(self, mock_earnings):
        """Earnings today should block."""
        mock_earnings.return_value = datetime.now() + timedelta(hours=4)
        safe, info = earnings_guard.check_earnings_safe("TEST", blackout_days=3)
        self.assertFalse(safe)

    @patch("earnings_guard.get_next_earnings")
    def test_custom_blackout_window(self, mock_earnings):
        """Custom blackout window should be respected."""
        # 8 calendar days = ~5 trading days
        mock_earnings.return_value = datetime.now() + timedelta(days=8)
        # With 3-day blackout, should be safe
        safe_3, _ = earnings_guard.check_earnings_safe("TEST", blackout_days=3)
        self.assertTrue(safe_3)
        # With 7-day blackout, should be blocked
        safe_7, _ = earnings_guard.check_earnings_safe("TEST", blackout_days=7)
        self.assertFalse(safe_7)

    @patch("earnings_guard.get_next_earnings")
    def test_info_dict_structure(self, mock_earnings):
        """Info dict should have all expected keys."""
        mock_earnings.return_value = datetime.now() + timedelta(days=15)
        _, info = earnings_guard.check_earnings_safe("TEST")
        self.assertIn("earnings_date", info)
        self.assertIn("days_until", info)
        self.assertIn("reason", info)
        self.assertIn("blackout_days", info)


if __name__ == "__main__":
    unittest.main()
