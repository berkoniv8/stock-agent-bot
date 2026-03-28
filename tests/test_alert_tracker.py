"""
Unit tests for alert deduplication tracker.
"""

import sys
import os
import unittest
import json
import tempfile
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import alert_tracker


class TestAlertTracker(unittest.TestCase):

    def setUp(self):
        """Use a temp file for history during tests."""
        self.original_file = alert_tracker.HISTORY_FILE
        self.tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self.tmp.close()
        alert_tracker.HISTORY_FILE = Path(self.tmp.name)
        alert_tracker.clear_history()

    def tearDown(self):
        alert_tracker.HISTORY_FILE = self.original_file
        os.unlink(self.tmp.name)

    def test_first_alert_not_duplicate(self):
        signals = [("breakout_with_volume", 3), ("ema_cross_bullish", 2)]
        self.assertFalse(alert_tracker.is_duplicate("AAPL", "BUY", signals))

    def test_recorded_alert_is_duplicate(self):
        signals = [("breakout_with_volume", 3)]
        alert_tracker.record_alert("AAPL", "BUY", signals)
        self.assertTrue(alert_tracker.is_duplicate("AAPL", "BUY", signals))

    def test_different_ticker_not_duplicate(self):
        signals = [("breakout_with_volume", 3)]
        alert_tracker.record_alert("AAPL", "BUY", signals)
        self.assertFalse(alert_tracker.is_duplicate("MSFT", "BUY", signals))

    def test_different_direction_not_duplicate(self):
        signals = [("breakout_with_volume", 3)]
        alert_tracker.record_alert("AAPL", "BUY", signals)
        self.assertFalse(alert_tracker.is_duplicate("AAPL", "SELL", signals))

    def test_expired_cooldown_not_duplicate(self):
        signals = [("breakout_with_volume", 3)]
        # Manually insert an old timestamp
        key = alert_tracker._make_key("AAPL", "BUY", signals)
        old_time = (datetime.now() - timedelta(hours=25)).isoformat()
        alert_tracker._save_history({key: old_time})
        self.assertFalse(alert_tracker.is_duplicate("AAPL", "BUY", signals))

    def test_clear_history(self):
        signals = [("breakout_with_volume", 3)]
        alert_tracker.record_alert("AAPL", "BUY", signals)
        alert_tracker.clear_history()
        self.assertFalse(alert_tracker.is_duplicate("AAPL", "BUY", signals))


if __name__ == "__main__":
    unittest.main()
