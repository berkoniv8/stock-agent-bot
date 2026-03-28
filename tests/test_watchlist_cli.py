"""Tests for watchlist CLI module."""

import csv
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import watchlist_cli


class TestLoadWatchlist(unittest.TestCase):
    def test_missing_file(self):
        orig = watchlist_cli.WATCHLIST_PATH
        watchlist_cli.WATCHLIST_PATH = Path("/nonexistent_watchlist.csv")
        result = watchlist_cli.load_watchlist()
        watchlist_cli.WATCHLIST_PATH = orig
        self.assertEqual(result, [])

    def test_load_existing(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
        writer = csv.DictWriter(tmp, fieldnames=["ticker", "sector", "notes"])
        writer.writeheader()
        writer.writerow({"ticker": "AAPL", "sector": "Technology", "notes": ""})
        tmp.close()

        orig = watchlist_cli.WATCHLIST_PATH
        watchlist_cli.WATCHLIST_PATH = Path(tmp.name)
        result = watchlist_cli.load_watchlist()
        watchlist_cli.WATCHLIST_PATH = orig
        os.unlink(tmp.name)

        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["ticker"], "AAPL")


class TestSaveWatchlist(unittest.TestCase):
    def test_save_and_reload(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
        tmp.close()

        orig = watchlist_cli.WATCHLIST_PATH
        watchlist_cli.WATCHLIST_PATH = Path(tmp.name)

        entries = [
            {"ticker": "AAPL", "sector": "Technology", "notes": "test"},
            {"ticker": "MSFT", "sector": "Technology", "notes": ""},
        ]
        watchlist_cli.save_watchlist(entries)
        loaded = watchlist_cli.load_watchlist()

        watchlist_cli.WATCHLIST_PATH = orig
        os.unlink(tmp.name)

        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0]["ticker"], "AAPL")


class TestCmdAdd(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
        writer = csv.DictWriter(self._tmp, fieldnames=["ticker", "sector", "notes"])
        writer.writeheader()
        self._tmp.close()
        self._orig = watchlist_cli.WATCHLIST_PATH
        watchlist_cli.WATCHLIST_PATH = Path(self._tmp.name)

    def tearDown(self):
        watchlist_cli.WATCHLIST_PATH = self._orig
        try:
            os.unlink(self._tmp.name)
        except OSError:
            pass

    @patch("watchlist_cli.yf")
    def test_add_new_ticker(self, mock_yf):
        mock_ticker = MagicMock()
        mock_ticker.info = {"shortName": "Apple Inc."}
        mock_yf.Ticker.return_value = mock_ticker

        args = MagicMock()
        args.ticker = "aapl"
        args.sector = "Technology"
        args.notes = "test add"

        watchlist_cli.cmd_add(args)
        entries = watchlist_cli.load_watchlist()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["ticker"], "AAPL")

    @patch("watchlist_cli.yf")
    def test_add_duplicate_rejected(self, mock_yf):
        # First add
        mock_ticker = MagicMock()
        mock_ticker.info = {"shortName": "Apple"}
        mock_yf.Ticker.return_value = mock_ticker

        args = MagicMock()
        args.ticker = "AAPL"
        args.sector = "Technology"
        args.notes = ""
        watchlist_cli.cmd_add(args)

        # Second add of same ticker
        watchlist_cli.cmd_add(args)
        entries = watchlist_cli.load_watchlist()
        self.assertEqual(len(entries), 1)


class TestCmdRemove(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
        writer = csv.DictWriter(self._tmp, fieldnames=["ticker", "sector", "notes"])
        writer.writeheader()
        writer.writerow({"ticker": "AAPL", "sector": "Technology", "notes": ""})
        writer.writerow({"ticker": "MSFT", "sector": "Technology", "notes": ""})
        self._tmp.close()
        self._orig = watchlist_cli.WATCHLIST_PATH
        watchlist_cli.WATCHLIST_PATH = Path(self._tmp.name)

    def tearDown(self):
        watchlist_cli.WATCHLIST_PATH = self._orig
        try:
            os.unlink(self._tmp.name)
        except OSError:
            pass

    def test_remove_existing(self):
        args = MagicMock()
        args.ticker = "AAPL"
        watchlist_cli.cmd_remove(args)
        entries = watchlist_cli.load_watchlist()
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["ticker"], "MSFT")

    def test_remove_nonexistent(self):
        args = MagicMock()
        args.ticker = "GOOGL"
        watchlist_cli.cmd_remove(args)
        entries = watchlist_cli.load_watchlist()
        self.assertEqual(len(entries), 2)  # Unchanged


class TestCmdList(unittest.TestCase):
    def test_empty_watchlist(self):
        orig = watchlist_cli.WATCHLIST_PATH
        watchlist_cli.WATCHLIST_PATH = Path("/nonexistent.csv")
        # Should not raise
        watchlist_cli.cmd_list(MagicMock())
        watchlist_cli.WATCHLIST_PATH = orig


if __name__ == "__main__":
    unittest.main()
