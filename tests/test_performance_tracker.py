"""Tests for performance tracker module."""

import unittest

import performance_tracker


class TestLoadAlerts(unittest.TestCase):
    def test_missing_file(self):
        from unittest.mock import patch, MagicMock
        from pathlib import Path
        with patch.object(performance_tracker, "DASHBOARD_CSV", Path("/nonexistent.csv")):
            result = performance_tracker.load_alerts()
        self.assertEqual(result, [])


class TestEvaluateAlert(unittest.TestCase):
    def test_missing_entry_price(self):
        alert = {"ticker": "AAPL", "direction": "BUY", "entry_price": "0",
                 "stop_loss": "165", "target_1": "177", "target_2": "185",
                 "target_3": "190", "timestamp": "2026-01-01", "shares": "10"}
        result = performance_tracker.evaluate_alert(alert)
        self.assertEqual(result["status"], "error")

    def test_invalid_price_values(self):
        alert = {"ticker": "AAPL", "direction": "BUY", "entry_price": "bad",
                 "stop_loss": "165", "target_1": "177", "target_2": "185",
                 "target_3": "190", "timestamp": "2026-01-01", "shares": "10"}
        result = performance_tracker.evaluate_alert(alert)
        self.assertEqual(result["status"], "error")

    def test_missing_timestamp(self):
        alert = {"ticker": "AAPL", "direction": "BUY", "entry_price": "170",
                 "stop_loss": "165", "target_1": "177", "target_2": "185",
                 "target_3": "190", "timestamp": "", "shares": "10"}
        result = performance_tracker.evaluate_alert(alert)
        self.assertEqual(result["status"], "error")


class TestPrintPerformance(unittest.TestCase):
    def test_empty_results(self):
        # Should not raise
        performance_tracker.print_performance([])

    def test_with_results(self):
        results = [{
            "ticker": "AAPL", "direction": "BUY", "entry_price": 170,
            "current_price": 180, "unrealized_pnl": 10, "unrealized_pct": 5.9,
            "dollar_pnl": 100, "days_held": 10, "status": "TARGET 1 HIT",
        }]
        # Should not raise
        performance_tracker.print_performance(results)


class TestSavePerformanceCsv(unittest.TestCase):
    def test_empty_results(self):
        # Should not raise
        performance_tracker.save_performance_csv([])


if __name__ == "__main__":
    unittest.main()
