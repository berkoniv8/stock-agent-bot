"""Tests for trailing stop manager."""

import json
import os
import tempfile
import unittest

import trailing_stop


class TestComputeTrailingStop(unittest.TestCase):
    def test_atr_long(self):
        """ATR trailing stop for long position."""
        stop = trailing_stop.compute_trailing_stop(
            current_price=105, highest_price=110, lowest_price=95,
            direction="BUY", atr=3.0, method="atr", atr_mult=2.0,
        )
        # Should be highest_price - 2*ATR = 110 - 6 = 104
        self.assertAlmostEqual(stop, 104.0)

    def test_atr_short(self):
        """ATR trailing stop for short position."""
        stop = trailing_stop.compute_trailing_stop(
            current_price=95, highest_price=110, lowest_price=90,
            direction="SELL", atr=3.0, method="atr", atr_mult=2.0,
        )
        # Should be lowest_price + 2*ATR = 90 + 6 = 96
        self.assertAlmostEqual(stop, 96.0)

    def test_percent_long(self):
        """Percentage trailing stop for long position."""
        stop = trailing_stop.compute_trailing_stop(
            current_price=105, highest_price=110, lowest_price=95,
            direction="BUY", atr=3.0, method="percent", trail_pct=3.0,
        )
        # Should be highest_price * (1 - 0.03) = 110 * 0.97 = 106.7
        self.assertAlmostEqual(stop, 106.7)

    def test_hybrid_uses_tighter_stop(self):
        """Hybrid should use the tighter (higher for longs) stop."""
        stop = trailing_stop.compute_trailing_stop(
            current_price=105, highest_price=110, lowest_price=95,
            direction="BUY", atr=3.0, method="hybrid", trail_pct=3.0, atr_mult=2.0,
        )
        atr_stop = 110 - 2 * 3.0  # 104
        pct_stop = 110 * 0.97      # 106.7
        self.assertAlmostEqual(stop, max(atr_stop, pct_stop))

    def test_zero_atr_long(self):
        """Zero ATR should produce 0 stop for ATR method."""
        stop = trailing_stop.compute_trailing_stop(
            current_price=100, highest_price=100, lowest_price=95,
            direction="BUY", atr=0, method="atr",
        )
        self.assertEqual(stop, 0)


class TestPositionManagement(unittest.TestCase):
    def setUp(self):
        """Use a temp file for positions."""
        self._orig_file = trailing_stop.POSITIONS_FILE
        self._tmpdir = tempfile.mkdtemp()
        trailing_stop.POSITIONS_FILE = type(trailing_stop.POSITIONS_FILE)(
            os.path.join(self._tmpdir, "test_positions.json")
        )

    def tearDown(self):
        trailing_stop.POSITIONS_FILE = self._orig_file

    def test_add_position(self):
        """Adding a position should persist to file."""
        trailing_stop.add_position("TEST", "BUY", 100.0, 95.0, 10)
        positions = trailing_stop._load_positions()
        self.assertEqual(len(positions), 1)
        self.assertEqual(positions[0]["ticker"], "TEST")
        self.assertEqual(positions[0]["status"], "open")

    def test_duplicate_prevention(self):
        """Should not add duplicate open positions for same ticker."""
        trailing_stop.add_position("TEST", "BUY", 100.0, 95.0, 10)
        trailing_stop.add_position("TEST", "BUY", 105.0, 100.0, 5)
        positions = trailing_stop._load_positions()
        self.assertEqual(len(positions), 1)

    def test_close_position(self):
        """Closing a position should update status and compute P&L."""
        trailing_stop.add_position("TEST", "BUY", 100.0, 95.0, 10)
        closed = trailing_stop.close_position("TEST", exit_price=110.0, reason="target")
        self.assertIsNotNone(closed)
        self.assertEqual(closed["status"], "closed")
        self.assertAlmostEqual(closed["pnl"], 100.0)  # (110-100)*10

    def test_close_sell_position(self):
        """Short position P&L should be entry - exit."""
        trailing_stop.add_position("TEST", "SELL", 100.0, 105.0, 10)
        closed = trailing_stop.close_position("TEST", exit_price=90.0, reason="target")
        self.assertAlmostEqual(closed["pnl"], 100.0)  # (100-90)*10


if __name__ == "__main__":
    unittest.main()
