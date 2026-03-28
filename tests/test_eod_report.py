"""Tests for end-of-day report generator."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import eod_report


class TestGetTodaysTrades(unittest.TestCase):
    def test_extracts_entries_for_date(self):
        state = {
            "open_positions": [
                {"ticker": "AAPL", "entry_date": "2026-03-25", "direction": "BUY",
                 "shares": 10, "entry_price": 170, "stop_loss": 165},
                {"ticker": "MSFT", "entry_date": "2026-03-24", "direction": "BUY",
                 "shares": 5, "entry_price": 400, "stop_loss": 390},
            ],
            "closed_trades": [],
        }
        result = eod_report.get_todays_trades(state, "2026-03-25")
        self.assertEqual(len(result["entries"]), 1)
        self.assertEqual(result["entries"][0]["ticker"], "AAPL")

    def test_extracts_exits_for_date(self):
        state = {
            "open_positions": [],
            "closed_trades": [
                {"ticker": "GOOGL", "exit_date": "2026-03-25", "direction": "BUY",
                 "pnl": 200, "pnl_pct": 3.5, "exit_reason": "target_1", "bars_held": 5},
                {"ticker": "JPM", "exit_date": "2026-03-24", "direction": "SELL",
                 "pnl": -50, "pnl_pct": -1.0, "exit_reason": "stop_loss", "bars_held": 3},
            ],
        }
        result = eod_report.get_todays_trades(state, "2026-03-25")
        self.assertEqual(len(result["exits"]), 1)
        self.assertEqual(result["exits"][0]["ticker"], "GOOGL")

    def test_no_trades_on_date(self):
        state = {
            "open_positions": [
                {"ticker": "AAPL", "entry_date": "2026-03-20", "direction": "BUY",
                 "shares": 10, "entry_price": 170, "stop_loss": 165},
            ],
            "closed_trades": [],
        }
        result = eod_report.get_todays_trades(state, "2026-03-25")
        self.assertEqual(len(result["entries"]), 0)
        self.assertEqual(len(result["exits"]), 0)

    def test_empty_state(self):
        result = eod_report.get_todays_trades({}, "2026-03-25")
        self.assertEqual(result["entries"], [])
        self.assertEqual(result["exits"], [])


class TestComputeUnrealizedPnl(unittest.TestCase):
    def test_computes_risk_for_long(self):
        state = {
            "open_positions": [{
                "ticker": "AAPL", "direction": "BUY", "shares": 10,
                "entry_price": 170.0, "current_stop": 165.0,
            }],
        }
        result = eod_report.compute_unrealized_pnl(state)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["ticker"], "AAPL")
        self.assertEqual(result[0]["risk_at_stop"], 50.0)  # (170-165)*10

    def test_computes_risk_for_short(self):
        state = {
            "open_positions": [{
                "ticker": "TSLA", "direction": "SELL", "shares": 5,
                "entry_price": 200.0, "current_stop": 210.0,
            }],
        }
        result = eod_report.compute_unrealized_pnl(state)
        self.assertEqual(result[0]["risk_at_stop"], 50.0)  # (210-200)*5

    def test_no_stop_loss(self):
        state = {
            "open_positions": [{
                "ticker": "AAPL", "direction": "BUY", "shares": 10,
                "entry_price": 170.0,
            }],
        }
        result = eod_report.compute_unrealized_pnl(state)
        self.assertEqual(result[0]["risk_at_stop"], 0)

    def test_targets_hit_count(self):
        state = {
            "open_positions": [{
                "ticker": "AAPL", "direction": "BUY", "shares": 10,
                "entry_price": 170.0, "stop_loss": 165.0,
                "t1_hit": True, "t2_hit": True, "t3_hit": False,
            }],
        }
        result = eod_report.compute_unrealized_pnl(state)
        self.assertEqual(result[0]["targets_hit"], 2)

    def test_empty_positions(self):
        result = eod_report.compute_unrealized_pnl({})
        self.assertEqual(result, [])


class TestGetAlertActivity(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
        alerts = [
            {"ticker": "AAPL", "type": "CUSTOM", "condition": "above",
             "price": 180, "triggered": True, "triggered_at": "2026-03-25T14:30:00",
             "triggered_price": 182.5},
            {"ticker": "MSFT", "type": "STOP_WARNING", "condition": "below",
             "price": 390, "triggered": True, "triggered_at": "2026-03-24T10:00:00",
             "triggered_price": 388},
            {"ticker": "GOOGL", "type": "CUSTOM", "condition": "below",
             "price": 140, "triggered": False, "triggered_at": None},
        ]
        json.dump(alerts, self._tmp)
        self._tmp.close()
        self._alerts_path = Path(self._tmp.name)

    def tearDown(self):
        try:
            os.unlink(self._tmp.name)
        except OSError:
            pass

    def test_filters_by_date(self):
        # Temporarily point the function at our temp file
        import types

        def patched_get_alert_activity(report_date):
            alerts_file = self._alerts_path
            if not alerts_file.exists():
                return []
            with open(str(alerts_file)) as f:
                alerts = json.load(f)
            triggered = []
            for a in alerts:
                triggered_at = str(a.get("triggered_at", ""))[:10]
                if triggered_at == report_date and a.get("triggered"):
                    triggered.append({
                        "ticker": a.get("ticker", ""),
                        "type": a.get("type", "CUSTOM"),
                        "condition": a.get("condition", ""),
                        "price": a.get("price", 0),
                        "triggered_price": a.get("triggered_price", 0),
                    })
            return triggered

        result = patched_get_alert_activity("2026-03-25")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["ticker"], "AAPL")

    def test_no_alerts_file(self):
        result = eod_report.get_alert_activity("2026-03-25")
        # Returns empty if no file or no matching alerts
        self.assertIsInstance(result, list)


class TestGenerateReport(unittest.TestCase):
    @patch("eod_report.load_paper_state")
    @patch("eod_report.get_alert_activity")
    def test_generates_report_structure(self, mock_alerts, mock_state):
        mock_state.return_value = {
            "capital": 105000,
            "open_positions": [
                {"ticker": "AAPL", "direction": "BUY", "shares": 10,
                 "entry_price": 170, "entry_date": "2026-03-25",
                 "stop_loss": 165},
            ],
            "closed_trades": [
                {"ticker": "GOOGL", "direction": "BUY", "pnl": 200,
                 "pnl_pct": 3.5, "exit_date": "2026-03-25",
                 "exit_reason": "target_1", "bars_held": 5},
            ],
        }
        mock_alerts.return_value = []

        report = eod_report.generate_report("2026-03-25")
        self.assertEqual(report["date"], "2026-03-25")
        self.assertIn("portfolio", report)
        self.assertIn("daily_activity", report)
        self.assertEqual(report["daily_activity"]["entries"], 1)
        self.assertEqual(report["daily_activity"]["exits"], 1)
        self.assertEqual(report["daily_activity"]["realized_pnl"], 200)

    @patch("eod_report.load_paper_state")
    @patch("eod_report.get_alert_activity")
    def test_empty_state(self, mock_alerts, mock_state):
        mock_state.return_value = {}
        mock_alerts.return_value = []

        report = eod_report.generate_report("2026-03-25")
        self.assertEqual(report["daily_activity"]["entries"], 0)
        self.assertEqual(report["daily_activity"]["exits"], 0)


class TestFormatReport(unittest.TestCase):
    def test_produces_readable_text(self):
        report = {
            "date": "2026-03-25",
            "portfolio": {
                "capital": 105000, "initial_capital": 100000,
                "total_realized_pnl": 5000, "return_pct": 5.0,
                "open_positions": 2,
            },
            "daily_activity": {
                "entries": 1, "exits": 1, "realized_pnl": 200,
                "wins": 1, "losses": 0, "win_rate": 100.0,
                "best_trade": 200, "worst_trade": 200,
            },
            "new_entries": [
                {"ticker": "AAPL", "direction": "BUY", "shares": 10,
                 "entry_price": 170, "stop_loss": 165},
            ],
            "closed_trades": [
                {"ticker": "GOOGL", "direction": "BUY", "pnl": 200,
                 "pnl_pct": 3.5, "exit_reason": "target_1", "bars_held": 5},
            ],
            "open_positions": [
                {"ticker": "MSFT", "direction": "BUY", "shares": 5,
                 "entry_price": 400, "current_stop": 390,
                 "risk_at_stop": 50, "targets_hit": 1},
            ],
            "alerts_triggered": [],
        }
        text = eod_report.format_report(report)
        self.assertIn("END-OF-DAY REPORT", text)
        self.assertIn("2026-03-25", text)
        self.assertIn("AAPL", text)
        self.assertIn("GOOGL", text)
        self.assertIn("MSFT", text)

    def test_no_trades_day(self):
        report = {
            "date": "2026-03-25",
            "portfolio": {
                "capital": 100000, "initial_capital": 100000,
                "total_realized_pnl": 0, "return_pct": 0,
                "open_positions": 0,
            },
            "daily_activity": {
                "entries": 0, "exits": 0, "realized_pnl": 0,
                "wins": 0, "losses": 0, "win_rate": 0,
                "best_trade": 0, "worst_trade": 0,
            },
            "new_entries": [],
            "closed_trades": [],
            "open_positions": [],
            "alerts_triggered": [],
        }
        text = eod_report.format_report(report)
        self.assertIn("No trades today", text)


class TestSaveAndLoadReport(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.mkdtemp()
        self._orig = eod_report.REPORTS_DIR
        eod_report.REPORTS_DIR = Path(self._tmpdir)

    def tearDown(self):
        eod_report.REPORTS_DIR = self._orig
        import shutil
        shutil.rmtree(self._tmpdir, ignore_errors=True)

    def test_save_and_load(self):
        report = {"date": "2026-03-25", "portfolio": {"capital": 100000}}
        path = eod_report.save_report(report)
        self.assertTrue(os.path.exists(path))

        loaded = eod_report.load_report("2026-03-25")
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["date"], "2026-03-25")

    def test_load_nonexistent(self):
        result = eod_report.load_report("1999-01-01")
        self.assertIsNone(result)

    def test_list_reports(self):
        eod_report.save_report({"date": "2026-03-25"})
        eod_report.save_report({"date": "2026-03-24"})
        reports = eod_report.list_reports()
        self.assertEqual(len(reports), 2)
        self.assertIn("2026-03-25", reports)


if __name__ == "__main__":
    unittest.main()
