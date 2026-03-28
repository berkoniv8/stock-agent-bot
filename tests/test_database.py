"""Tests for the SQLite database layer."""

import os
import tempfile
import unittest
from pathlib import Path

import database as db


class TestDatabaseSetup(unittest.TestCase):
    def setUp(self):
        self._orig_path = db.DB_PATH
        self._tmpdir = tempfile.mkdtemp()
        db.DB_PATH = Path(os.path.join(self._tmpdir, "test.db"))
        db.init()

    def tearDown(self):
        db.DB_PATH = self._orig_path

    def test_init_creates_db(self):
        """init() should create the database file."""
        self.assertTrue(db.DB_PATH.exists())

    def test_schema_version(self):
        """Schema version should be set after init."""
        self.assertEqual(db.get_schema_version(), db.SCHEMA_VERSION)

    def test_double_init_safe(self):
        """Calling init() twice should not error."""
        db.init()
        self.assertEqual(db.get_schema_version(), db.SCHEMA_VERSION)


class TestAlerts(unittest.TestCase):
    def setUp(self):
        self._orig_path = db.DB_PATH
        self._tmpdir = tempfile.mkdtemp()
        db.DB_PATH = Path(os.path.join(self._tmpdir, "test.db"))
        db.init()

    def tearDown(self):
        db.DB_PATH = self._orig_path

    def test_insert_and_get(self):
        """Should insert and retrieve alerts."""
        row_id = db.insert_alert(
            ticker="AAPL", direction="BUY", signal_score=7,
            signals="ema_cross|breakout", entry_price=150.0,
            stop_loss=145.0, shares=10,
        )
        self.assertGreater(row_id, 0)

        alerts = db.get_alerts(limit=10)
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["ticker"], "AAPL")
        self.assertEqual(alerts[0]["signal_score"], 7)

    def test_filter_by_ticker(self):
        """Should filter alerts by ticker."""
        db.insert_alert(ticker="AAPL", direction="BUY", signal_score=5, signals="")
        db.insert_alert(ticker="MSFT", direction="BUY", signal_score=6, signals="")

        aapl = db.get_alerts(ticker="AAPL")
        self.assertEqual(len(aapl), 1)
        self.assertEqual(aapl[0]["ticker"], "AAPL")


class TestPositions(unittest.TestCase):
    def setUp(self):
        self._orig_path = db.DB_PATH
        self._tmpdir = tempfile.mkdtemp()
        db.DB_PATH = Path(os.path.join(self._tmpdir, "test.db"))
        db.init()

    def tearDown(self):
        db.DB_PATH = self._orig_path

    def test_insert_and_get_open(self):
        """Should insert and get open positions."""
        db.insert_position(
            ticker="AAPL", direction="BUY", entry_price=150.0,
            entry_date="2026-03-26T10:00:00", initial_stop=145.0, shares=10,
        )
        positions = db.get_open_positions()
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0]["ticker"], "AAPL")
        self.assertEqual(positions[0]["status"], "open")

    def test_update_and_close(self):
        """Should update position fields and close."""
        pos_id = db.insert_position(
            ticker="AAPL", direction="BUY", entry_price=150.0,
            entry_date="2026-03-26", initial_stop=145.0, shares=10,
        )
        db.update_position(pos_id, current_stop=147.0, highest_price=155.0)

        positions = db.get_open_positions()
        self.assertAlmostEqual(positions[0]["current_stop"], 147.0)

        db.update_position(pos_id, status="closed", exit_price=160.0,
                           exit_reason="target_1", pnl=100.0)
        self.assertEqual(len(db.get_open_positions()), 0)
        self.assertEqual(len(db.get_closed_positions()), 1)

    def test_has_open_position(self):
        """Should detect open position for ticker."""
        self.assertFalse(db.has_open_position("AAPL"))
        db.insert_position("AAPL", "BUY", 150.0, "2026-03-26", 145.0, 10)
        self.assertTrue(db.has_open_position("AAPL"))


class TestPaperTrading(unittest.TestCase):
    def setUp(self):
        self._orig_path = db.DB_PATH
        self._tmpdir = tempfile.mkdtemp()
        db.DB_PATH = Path(os.path.join(self._tmpdir, "test.db"))
        db.init()

    def tearDown(self):
        db.DB_PATH = self._orig_path

    def test_default_paper_state(self):
        """Should create default paper state."""
        state = db.get_paper_state()
        self.assertEqual(state["cash"], 100000)
        self.assertEqual(state["open_positions"], [])

    def test_reset_paper_state(self):
        """Should reset paper state with custom capital."""
        db.reset_paper_state(50000)
        state = db.get_paper_state()
        self.assertEqual(state["cash"], 50000)

    def test_paper_position_lifecycle(self):
        """Should insert, update, and delete paper positions."""
        pos_id = db.insert_paper_position(
            ticker="AAPL", direction="BUY", entry_price=150.0,
            entry_date="2026-03-26", shares=10, cost_basis=1500.0,
            stop_loss=145.0, current_stop=145.0,
        )
        positions = db.get_paper_positions()
        self.assertEqual(len(positions), 1)

        db.update_paper_position(pos_id, current_stop=147.0, t1_hit=1)
        positions = db.get_paper_positions()
        self.assertAlmostEqual(positions[0]["current_stop"], 147.0)
        self.assertTrue(positions[0]["t1_hit"])

        db.delete_paper_position(pos_id)
        self.assertEqual(len(db.get_paper_positions()), 0)

    def test_paper_trade_record(self):
        """Should record closed paper trades."""
        db.insert_paper_trade(
            ticker="AAPL", direction="BUY", entry_price=150.0,
            entry_date="2026-03-20", exit_price=160.0, exit_date="2026-03-25",
            exit_reason="target_1", shares=10, pnl=100.0, pnl_pct=6.67,
        )
        trades = db.get_paper_trades()
        self.assertEqual(len(trades), 1)
        self.assertAlmostEqual(trades[0]["pnl"], 100.0)


class TestAlertDedup(unittest.TestCase):
    def setUp(self):
        self._orig_path = db.DB_PATH
        self._tmpdir = tempfile.mkdtemp()
        db.DB_PATH = Path(os.path.join(self._tmpdir, "test.db"))
        db.init()

    def tearDown(self):
        db.DB_PATH = self._orig_path

    def test_not_duplicate_first_time(self):
        """First alert should not be duplicate."""
        self.assertFalse(db.check_alert_duplicate("AAPL:BUY:ema_cross"))

    def test_duplicate_within_cooldown(self):
        """Same key within cooldown should be duplicate."""
        db.record_alert_dedup("AAPL:BUY:ema_cross")
        self.assertTrue(db.check_alert_duplicate("AAPL:BUY:ema_cross", cooldown_hours=24))

    def test_prune_old_entries(self):
        """Pruning should remove old entries."""
        db.record_alert_dedup("OLD:BUY:test")
        # Manually age the entry
        with db.get_connection() as conn:
            conn.execute(
                "UPDATE alert_history SET timestamp = datetime('now', '-8 days') WHERE dedup_key = ?",
                ("OLD:BUY:test",),
            )
        removed = db.prune_alert_history(7)
        self.assertEqual(removed, 1)


class TestWatchlist(unittest.TestCase):
    def setUp(self):
        self._orig_path = db.DB_PATH
        self._tmpdir = tempfile.mkdtemp()
        db.DB_PATH = Path(os.path.join(self._tmpdir, "test.db"))
        db.init()

    def tearDown(self):
        db.DB_PATH = self._orig_path

    def test_add_and_get(self):
        """Should add and retrieve watchlist entries."""
        db.add_to_watchlist("AAPL", "Technology", "Core holding")
        db.add_to_watchlist("MSFT", "Technology")
        wl = db.get_watchlist()
        self.assertEqual(len(wl), 2)
        self.assertEqual(wl[0]["ticker"], "AAPL")

    def test_no_duplicates(self):
        """Should not add duplicate tickers."""
        self.assertTrue(db.add_to_watchlist("AAPL"))
        self.assertFalse(db.add_to_watchlist("AAPL"))

    def test_remove(self):
        """Should remove ticker."""
        db.add_to_watchlist("AAPL")
        self.assertTrue(db.remove_from_watchlist("AAPL"))
        self.assertEqual(len(db.get_watchlist()), 0)


class TestPortfolio(unittest.TestCase):
    def setUp(self):
        self._orig_path = db.DB_PATH
        self._tmpdir = tempfile.mkdtemp()
        db.DB_PATH = Path(os.path.join(self._tmpdir, "test.db"))
        db.init()

    def tearDown(self):
        db.DB_PATH = self._orig_path

    def test_default_config(self):
        """Should return default portfolio config."""
        config = db.get_portfolio_config()
        self.assertEqual(config["total_portfolio_value"], 50000)

    def test_update_config(self):
        """Should update portfolio config."""
        db.get_portfolio_config()  # ensure row exists
        db.update_portfolio_config(total_portfolio_value=75000, available_cash=20000)
        config = db.get_portfolio_config()
        self.assertEqual(config["total_portfolio_value"], 75000)
        self.assertEqual(config["available_cash"], 20000)

    def test_upsert_holding(self):
        """Should insert and update holdings."""
        db.upsert_holding("AAPL", 20, 150.0, "Technology", 160.0)
        holdings = db.get_holdings()
        self.assertEqual(len(holdings), 1)
        self.assertEqual(holdings[0]["shares"], 20)

        # Update
        db.upsert_holding("AAPL", 30, 155.0, "Technology", 165.0)
        holdings = db.get_holdings()
        self.assertEqual(len(holdings), 1)
        self.assertEqual(holdings[0]["shares"], 30)


class TestBacktestTrades(unittest.TestCase):
    def setUp(self):
        self._orig_path = db.DB_PATH
        self._tmpdir = tempfile.mkdtemp()
        db.DB_PATH = Path(os.path.join(self._tmpdir, "test.db"))
        db.init()

    def tearDown(self):
        db.DB_PATH = self._orig_path

    def test_insert_and_get(self):
        """Should insert and retrieve backtest trades."""
        db.insert_backtest_trade("run_001", {
            "ticker": "AAPL", "direction": "BUY",
            "entry_price": 150, "exit_price": 160,
            "pnl": 100, "signals": "ema_cross|breakout",
        })
        trades = db.get_backtest_trades("run_001")
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["ticker"], "AAPL")

    def test_run_ids(self):
        """Should list distinct run IDs."""
        db.insert_backtest_trade("run_001", {"ticker": "AAPL"})
        db.insert_backtest_trade("run_002", {"ticker": "MSFT"})
        runs = db.get_backtest_runs()
        self.assertEqual(len(runs), 2)


if __name__ == "__main__":
    unittest.main()
