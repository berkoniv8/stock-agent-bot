"""Tests for config validator module."""

import csv
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import config_validator


class TestValidateEnv(unittest.TestCase):
    @patch.dict(os.environ, {
        "TOTAL_PORTFOLIO_VALUE": "100000",
        "AVAILABLE_CASH": "50000",
        "MAX_RISK_PER_TRADE_PCT": "1.0",
        "MAX_POSITION_SIZE_PCT": "10.0",
        "SIGNAL_THRESHOLD": "5",
        "RUN_INTERVAL_MINUTES": "15",
    })
    def test_valid_env(self):
        checks = config_validator.validate_env()
        errors = [c for c in checks if c[0] == "error"]
        self.assertEqual(len(errors), 0)

    @patch.dict(os.environ, {"TOTAL_PORTFOLIO_VALUE": "not_a_number"})
    def test_invalid_numeric(self):
        checks = config_validator.validate_env()
        errors = [c for c in checks if c[0] == "error" and c[1] == "TOTAL_PORTFOLIO_VALUE"]
        self.assertGreater(len(errors), 0)

    @patch.dict(os.environ, {
        "TOTAL_PORTFOLIO_VALUE": "100000",
        "AVAILABLE_CASH": "200000",  # Cash > portfolio
        "MAX_RISK_PER_TRADE_PCT": "1.0",
        "MAX_POSITION_SIZE_PCT": "10.0",
    })
    def test_warns_cash_exceeds_portfolio(self):
        checks = config_validator.validate_env()
        warns = [c for c in checks if c[0] == "warn" and "Cash exceeds" in c[2]]
        self.assertGreater(len(warns), 0)

    @patch.dict(os.environ, {
        "TOTAL_PORTFOLIO_VALUE": "100000",
        "AVAILABLE_CASH": "50000",
        "MAX_RISK_PER_TRADE_PCT": "8.0",  # Aggressive
        "MAX_POSITION_SIZE_PCT": "10.0",
    })
    def test_warns_aggressive_risk(self):
        checks = config_validator.validate_env()
        warns = [c for c in checks if c[0] == "warn" and "aggressive" in c[2]]
        self.assertGreater(len(warns), 0)


class TestValidatePortfolio(unittest.TestCase):
    def test_missing_file(self):
        with patch("config_validator.Path") as mock_path:
            mock_path.return_value.exists.return_value = False
            checks = config_validator.validate_portfolio()
        warns = [c for c in checks if c[0] == "warn"]
        self.assertGreater(len(warns), 0)

    def test_valid_portfolio(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
        json.dump({
            "total_portfolio_value": 100000,
            "available_cash": 50000,
            "holdings": [{"ticker": "AAPL", "shares": 10}],
        }, tmp)
        tmp.close()

        with patch("config_validator.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            with patch("builtins.open", return_value=open(tmp.name)):
                checks = config_validator.validate_portfolio()
        os.unlink(tmp.name)
        errors = [c for c in checks if c[0] == "error"]
        self.assertEqual(len(errors), 0)


class TestValidateWatchlist(unittest.TestCase):
    def test_valid_watchlist(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
        writer = csv.DictWriter(tmp, fieldnames=["ticker", "sector"])
        writer.writeheader()
        writer.writerow({"ticker": "AAPL", "sector": "Technology"})
        writer.writerow({"ticker": "MSFT", "sector": "Technology"})
        tmp.close()

        with patch("config_validator.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            with patch("builtins.open", return_value=open(tmp.name, newline="")):
                checks = config_validator.validate_watchlist()
        os.unlink(tmp.name)
        ok_checks = [c for c in checks if c[0] == "ok"]
        self.assertGreater(len(ok_checks), 0)

    def test_duplicate_tickers_warned(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
        writer = csv.DictWriter(tmp, fieldnames=["ticker", "sector"])
        writer.writeheader()
        writer.writerow({"ticker": "AAPL", "sector": "Technology"})
        writer.writerow({"ticker": "AAPL", "sector": "Technology"})
        tmp.close()

        with patch("config_validator.Path") as mock_path:
            mock_path.return_value.exists.return_value = True
            with patch("builtins.open", return_value=open(tmp.name, newline="")):
                checks = config_validator.validate_watchlist()
        os.unlink(tmp.name)
        warns = [c for c in checks if c[0] == "warn" and "Duplicate" in c[2]]
        self.assertGreater(len(warns), 0)


class TestValidateDirectories(unittest.TestCase):
    def test_creates_or_confirms_logs(self):
        checks = config_validator.validate_directories()
        ok_checks = [c for c in checks if c[0] == "ok"]
        self.assertGreater(len(ok_checks), 0)


class TestRunAll(unittest.TestCase):
    @patch("config_validator.validate_env", return_value=[("ok", "TEST", "ok")])
    @patch("config_validator.validate_portfolio", return_value=[("ok", "TEST", "ok")])
    @patch("config_validator.validate_watchlist", return_value=[("ok", "TEST", "ok")])
    @patch("config_validator.validate_directories", return_value=[("ok", "TEST", "ok")])
    def test_returns_true_when_healthy(self, *mocks):
        result = config_validator.run_all()
        self.assertTrue(result)

    @patch("config_validator.validate_env", return_value=[("error", "BAD", "broken")])
    @patch("config_validator.validate_portfolio", return_value=[])
    @patch("config_validator.validate_watchlist", return_value=[])
    @patch("config_validator.validate_directories", return_value=[])
    def test_returns_false_on_errors(self, *mocks):
        result = config_validator.run_all()
        self.assertFalse(result)


if __name__ == "__main__":
    unittest.main()
