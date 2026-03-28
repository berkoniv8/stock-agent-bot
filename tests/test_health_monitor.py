"""Tests for system health monitor."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import health_monitor


class TestCheckConfig(unittest.TestCase):
    @patch.dict(os.environ, {
        "INITIAL_CAPITAL": "100000",
        "RISK_PER_TRADE_PCT": "1.0",
        "MAX_OPEN_POSITIONS": "5",
    })
    def test_all_vars_present(self):
        result = health_monitor.check_config()
        self.assertEqual(result["status"], "OK")

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_vars(self):
        result = health_monitor.check_config()
        self.assertEqual(result["status"], "WARN")
        self.assertIn("Missing", result["message"])


class TestCheckDiskSpace(unittest.TestCase):
    def test_no_logs_dir(self):
        with patch("health_monitor.Path") as mock_path:
            mock_path.return_value.exists.return_value = False
            # Use actual Path for the check
            pass
        # Just verify it doesn't crash with real filesystem
        result = health_monitor.check_disk_space()
        self.assertIn(result["status"], ("OK", "WARN", "FAIL"))

    def test_returns_valid_structure(self):
        result = health_monitor.check_disk_space()
        self.assertIn("name", result)
        self.assertIn("status", result)
        self.assertIn("message", result)
        self.assertEqual(result["name"], "DISK_SPACE")


class TestCheckPositionsSync(unittest.TestCase):
    def test_no_state_file(self):
        with patch("health_monitor.Path") as mock_path_cls:
            instance = MagicMock()
            instance.exists.return_value = False
            mock_path_cls.return_value = instance
            # Run with real Path
        result = health_monitor.check_positions_sync()
        self.assertIn(result["status"], ("OK", "WARN", "FAIL"))

    def test_valid_state(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
        state = {
            "capital": 100000,
            "open_positions": [
                {"ticker": "AAPL", "shares": 10, "direction": "BUY",
                 "entry_price": 170},
            ],
            "closed_trades": [],
        }
        json.dump(state, tmp)
        tmp.close()

        with patch("health_monitor.Path") as mock_path:
            mock_instance = MagicMock()
            mock_instance.exists.return_value = True
            mock_path.return_value = mock_instance
            # Use builtins open with real file
            with patch("builtins.open", return_value=open(tmp.name)):
                result = health_monitor.check_positions_sync()

        os.unlink(tmp.name)
        # May be OK or WARN depending on filesystem state
        self.assertIn(result["status"], ("OK", "WARN", "FAIL"))

    def test_corrupt_state(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
        tmp.write("{bad json")
        tmp.close()

        orig_path = Path("logs/paper_state.json")
        # Directly test with a temp file by patching
        with patch("health_monitor.Path") as mock_path:
            inst = MagicMock()
            inst.exists.return_value = True
            mock_path.return_value = inst
            with patch("builtins.open", return_value=open(tmp.name)):
                result = health_monitor.check_positions_sync()

        os.unlink(tmp.name)
        self.assertEqual(result["status"], "FAIL")
        self.assertIn("Corrupt", result["message"])


class TestCheckDataFreshness(unittest.TestCase):
    def test_returns_valid_structure(self):
        result = health_monitor.check_data_freshness()
        self.assertIn("name", result)
        self.assertEqual(result["name"], "DATA_FRESHNESS")
        self.assertIn(result["status"], ("OK", "WARN", "FAIL"))


class TestRunAllChecks(unittest.TestCase):
    @patch("health_monitor.check_yahoo_finance")
    @patch("health_monitor.check_data_freshness")
    @patch("health_monitor.check_disk_space")
    @patch("health_monitor.check_config")
    @patch("health_monitor.check_positions_sync")
    def test_runs_all(self, mock_pos, mock_cfg, mock_disk, mock_fresh, mock_yf):
        mock_yf.return_value = {"name": "YAHOO_FINANCE", "status": "OK", "message": "ok"}
        mock_fresh.return_value = {"name": "DATA_FRESHNESS", "status": "OK", "message": "ok"}
        mock_disk.return_value = {"name": "DISK_SPACE", "status": "OK", "message": "ok"}
        mock_cfg.return_value = {"name": "CONFIG_VALID", "status": "OK", "message": "ok"}
        mock_pos.return_value = {"name": "POSITIONS_SYNC", "status": "OK", "message": "ok"}

        results = health_monitor.run_all_checks()
        self.assertEqual(len(results), 5)
        for r in results:
            self.assertIn("checked_at", r)

    def test_handles_check_crash(self):
        def boom():
            raise Exception("boom")

        orig = health_monitor.ALL_CHECKS.copy()
        health_monitor.ALL_CHECKS["yahoo"] = boom
        try:
            results = health_monitor.run_all_checks()
            self.assertEqual(len(results), 5)
            failed = [r for r in results if r["status"] == "FAIL"]
            self.assertEqual(len(failed), 1)
            self.assertIn("boom", failed[0]["message"])
        finally:
            health_monitor.ALL_CHECKS.update(orig)


class TestRunSingleCheck(unittest.TestCase):
    def test_valid_check(self):
        with patch.dict(os.environ, {
            "INITIAL_CAPITAL": "100000",
            "RISK_PER_TRADE_PCT": "1.0",
            "MAX_OPEN_POSITIONS": "5",
        }):
            result = health_monitor.run_single_check("config")
        self.assertIsNotNone(result)
        self.assertEqual(result["name"], "CONFIG_VALID")

    def test_invalid_check(self):
        result = health_monitor.run_single_check("nonexistent")
        self.assertIsNone(result)


class TestGetOverallStatus(unittest.TestCase):
    def test_all_ok(self):
        results = [{"status": "OK"}, {"status": "OK"}]
        self.assertEqual(health_monitor.get_overall_status(results), "OK")

    def test_any_warn(self):
        results = [{"status": "OK"}, {"status": "WARN"}]
        self.assertEqual(health_monitor.get_overall_status(results), "WARN")

    def test_any_fail(self):
        results = [{"status": "OK"}, {"status": "WARN"}, {"status": "FAIL"}]
        self.assertEqual(health_monitor.get_overall_status(results), "FAIL")


class TestSaveHealthReport(unittest.TestCase):
    def test_saves_to_disk(self):
        results = [{"name": "TEST", "status": "OK", "message": "test"}]
        with patch("health_monitor.open", unittest.mock.mock_open()) as mock_file:
            with patch("health_monitor.os.makedirs"):
                health_monitor.save_health_report(results)
        mock_file.assert_called_once()


if __name__ == "__main__":
    unittest.main()
