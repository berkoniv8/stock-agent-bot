"""Tests for the paper trading engine."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import paper_trader
from position_sizing import PositionPlan
from signal_engine import TradeAlert


class TestPaperTraderState(unittest.TestCase):
    def setUp(self):
        self._orig_file = paper_trader.PAPER_FILE
        self._tmpdir = tempfile.mkdtemp()
        paper_trader.PAPER_FILE = Path(os.path.join(self._tmpdir, "test_paper.json"))

    def tearDown(self):
        paper_trader.PAPER_FILE = self._orig_file

    def test_default_state(self):
        """Default state should have starting capital and empty positions."""
        state = paper_trader._default_state()
        self.assertEqual(state["cash"], paper_trader.DEFAULT_CAPITAL)
        self.assertEqual(state["open_positions"], [])
        self.assertEqual(state["closed_trades"], [])

    def test_save_and_load(self):
        """State should persist across save/load."""
        state = paper_trader._default_state()
        state["cash"] = 50000
        paper_trader.save_state(state)
        loaded = paper_trader.load_state()
        self.assertEqual(loaded["cash"], 50000)

    def test_reset_state(self):
        """Reset should clear positions and restore capital."""
        state = paper_trader._default_state()
        state["cash"] = 1000
        state["open_positions"] = [{"ticker": "TEST"}]
        paper_trader.save_state(state)

        reset = paper_trader.reset_state(75000)
        self.assertEqual(reset["cash"], 75000)
        self.assertEqual(reset["open_positions"], [])

    def test_load_missing_file(self):
        """Loading nonexistent file should return default state."""
        paper_trader.PAPER_FILE = Path("/nonexistent/paper.json")
        state = paper_trader.load_state()
        self.assertEqual(state["cash"], paper_trader.DEFAULT_CAPITAL)


def _make_alert_and_plan(ticker="AAPL", direction="BUY", price=150.0):
    """Create a test TradeAlert and PositionPlan."""
    alert = TradeAlert(
        ticker=ticker,
        signal_score=7,
        direction=direction,
        triggered_signals=[("ema_cross_bullish", 2), ("breakout_with_volume", 3)],
    )
    plan = PositionPlan(
        ticker=ticker,
        direction=direction,
        entry_price=price,
        stop_loss=price * 0.97 if direction == "BUY" else price * 1.03,
        risk_per_share=price * 0.03,
        shares=10,
        position_value=price * 10,
        target_1=price * 1.05,
        target_2=price * 1.10,
        target_3=price * 1.15,
        max_loss=price * 0.03 * 10,
        risk_reward_t1=1.5,
        risk_reward_t2=3.0,
    )
    return alert, plan


class TestExecuteEntry(unittest.TestCase):
    def setUp(self):
        self._orig_file = paper_trader.PAPER_FILE
        self._tmpdir = tempfile.mkdtemp()
        paper_trader.PAPER_FILE = Path(os.path.join(self._tmpdir, "test_paper.json"))
        paper_trader.reset_state(100000)

    def tearDown(self):
        paper_trader.PAPER_FILE = self._orig_file

    def test_basic_entry(self):
        """Should open a position and deduct cash."""
        state = paper_trader.load_state()
        alert, plan = _make_alert_and_plan()
        pos = paper_trader.execute_entry(state, alert, plan)

        self.assertIsNotNone(pos)
        self.assertEqual(pos["ticker"], "AAPL")
        self.assertEqual(pos["shares"], 10)

        state = paper_trader.load_state()
        self.assertEqual(len(state["open_positions"]), 1)
        self.assertAlmostEqual(state["cash"], 100000 - 150.0 * 10)

    def test_duplicate_ticker_blocked(self):
        """Should not open two positions on same ticker."""
        state = paper_trader.load_state()
        alert, plan = _make_alert_and_plan()
        paper_trader.execute_entry(state, alert, plan)

        state = paper_trader.load_state()
        pos2 = paper_trader.execute_entry(state, alert, plan)
        self.assertIsNone(pos2)

    def test_insufficient_cash(self):
        """Should block entry when cash is too low."""
        state = paper_trader.load_state()
        state["cash"] = 100  # Not enough
        paper_trader.save_state(state)

        state = paper_trader.load_state()
        alert, plan = _make_alert_and_plan()
        pos = paper_trader.execute_entry(state, alert, plan)
        self.assertIsNone(pos)


class TestExecuteExit(unittest.TestCase):
    def setUp(self):
        self._orig_file = paper_trader.PAPER_FILE
        self._tmpdir = tempfile.mkdtemp()
        paper_trader.PAPER_FILE = Path(os.path.join(self._tmpdir, "test_paper.json"))
        paper_trader.reset_state(100000)
        # Enter a position
        state = paper_trader.load_state()
        alert, plan = _make_alert_and_plan()
        paper_trader.execute_entry(state, alert, plan)

    def tearDown(self):
        paper_trader.PAPER_FILE = self._orig_file

    def test_full_exit_win(self):
        """Full exit at profit should compute positive P&L."""
        state = paper_trader.load_state()
        trade = paper_trader.execute_exit(state, "AAPL", 160.0, "target_1")

        self.assertIsNotNone(trade)
        self.assertAlmostEqual(trade["pnl"], (160 - 150) * 10)
        self.assertEqual(trade["exit_reason"], "target_1")

        state = paper_trader.load_state()
        self.assertEqual(len(state["open_positions"]), 0)
        self.assertEqual(len(state["closed_trades"]), 1)

    def test_full_exit_loss(self):
        """Full exit at loss should compute negative P&L."""
        state = paper_trader.load_state()
        trade = paper_trader.execute_exit(state, "AAPL", 140.0, "trailing_stop")

        self.assertAlmostEqual(trade["pnl"], (140 - 150) * 10)

    def test_partial_exit(self):
        """Partial exit should reduce shares but keep position open."""
        state = paper_trader.load_state()
        trade = paper_trader.execute_exit(state, "AAPL", 155.0, "target_1", partial_pct=33)

        self.assertIsNotNone(trade)
        self.assertEqual(trade["shares"], 3)  # 33% of 10

        state = paper_trader.load_state()
        self.assertEqual(len(state["open_positions"]), 1)
        self.assertEqual(state["open_positions"][0]["shares"], 7)  # 10 - 3

    def test_exit_nonexistent_ticker(self):
        """Exit on unknown ticker should return None."""
        state = paper_trader.load_state()
        trade = paper_trader.execute_exit(state, "UNKNOWN", 100.0, "manual")
        self.assertIsNone(trade)

    def test_cash_restored_on_exit(self):
        """Cash should increase by exit proceeds."""
        state = paper_trader.load_state()
        cash_before = state["cash"]
        paper_trader.execute_exit(state, "AAPL", 155.0, "target_1")
        state = paper_trader.load_state()
        expected_cash = cash_before + 155.0 * 10
        self.assertAlmostEqual(state["cash"], expected_cash)


class TestPerformance(unittest.TestCase):
    def setUp(self):
        self._orig_file = paper_trader.PAPER_FILE
        self._tmpdir = tempfile.mkdtemp()
        paper_trader.PAPER_FILE = Path(os.path.join(self._tmpdir, "test_paper.json"))
        paper_trader.reset_state(100000)

    def tearDown(self):
        paper_trader.PAPER_FILE = self._orig_file

    def test_no_trades_performance(self):
        """Performance with no trades should indicate zero."""
        state = paper_trader.load_state()
        perf = paper_trader.compute_performance(state)
        self.assertEqual(perf["total_trades"], 0)

    def test_performance_after_trades(self):
        """Performance should compute win rate and P&L after trades."""
        state = paper_trader.load_state()

        # Simulate 3 closed trades
        state["closed_trades"] = [
            {"ticker": "AAPL", "pnl": 500, "pnl_pct": 3.3, "bars_held": 5,
             "triggered_signals": ["ema_cross_bullish"], "entry_price": 150,
             "exit_price": 155, "direction": "BUY", "entry_date": "2026-03-01",
             "exit_date": "2026-03-05", "exit_reason": "target_1", "shares": 10,
             "signal_score": 7},
            {"ticker": "MSFT", "pnl": -200, "pnl_pct": -1.3, "bars_held": 3,
             "triggered_signals": ["breakout_with_volume"], "entry_price": 380,
             "exit_price": 375, "direction": "BUY", "entry_date": "2026-03-02",
             "exit_date": "2026-03-04", "exit_reason": "trailing_stop", "shares": 4,
             "signal_score": 6},
            {"ticker": "GOOGL", "pnl": 300, "pnl_pct": 2.0, "bars_held": 7,
             "triggered_signals": ["ema_cross_bullish", "price_above_vwap"],
             "entry_price": 170, "exit_price": 173, "direction": "BUY",
             "entry_date": "2026-03-03", "exit_date": "2026-03-10",
             "exit_reason": "target_1", "shares": 10, "signal_score": 8},
        ]
        paper_trader.save_state(state)

        state = paper_trader.load_state()
        perf = paper_trader.compute_performance(state)

        self.assertEqual(perf["total_trades"], 3)
        self.assertEqual(perf["wins"], 2)
        self.assertEqual(perf["losses"], 1)
        self.assertAlmostEqual(perf["win_rate"], 66.7, places=0)
        self.assertAlmostEqual(perf["total_pnl"], 600)
        self.assertGreater(perf["profit_factor"], 1.0)
        self.assertGreater(perf["expectancy"], 0)
        self.assertEqual(perf["max_win_streak"], 1)  # win, loss, win
        self.assertEqual(perf["max_loss_streak"], 1)


class TestCanOpenPosition(unittest.TestCase):
    def setUp(self):
        self._orig_file = paper_trader.PAPER_FILE
        self._tmpdir = tempfile.mkdtemp()
        paper_trader.PAPER_FILE = Path(os.path.join(self._tmpdir, "test_paper.json"))
        paper_trader.reset_state(100000)

    def tearDown(self):
        paper_trader.PAPER_FILE = self._orig_file

    def test_can_open_normal(self):
        """Should allow opening with sufficient cash and no conflicts."""
        state = paper_trader.load_state()
        _, plan = _make_alert_and_plan()
        can, reason = paper_trader.can_open_position(state, plan)
        self.assertTrue(can)

    def test_blocked_by_duplicate(self):
        """Should block when same ticker already open."""
        state = paper_trader.load_state()
        alert, plan = _make_alert_and_plan()
        paper_trader.execute_entry(state, alert, plan)

        state = paper_trader.load_state()
        can, reason = paper_trader.can_open_position(state, plan)
        self.assertFalse(can)
        self.assertIn("Already have", reason)


if __name__ == "__main__":
    unittest.main()
