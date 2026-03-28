"""Tests for backtester module."""

import unittest

import backtester


class TestComputePosition(unittest.TestCase):
    def test_buy_targets(self):
        result = backtester._compute_position(100.0, 95.0, "BUY")
        # risk = 5, T1 = 100 + 7.5 = 107.5, T2 = 100 + 15 = 115
        self.assertAlmostEqual(result["target_1"], 107.5, places=1)
        self.assertAlmostEqual(result["target_2"], 115.0, places=1)

    def test_sell_targets(self):
        result = backtester._compute_position(100.0, 105.0, "SELL")
        # risk = 5, T1 = 100 - 7.5 = 92.5, T2 = 100 - 15 = 85
        self.assertAlmostEqual(result["target_1"], 92.5, places=1)
        self.assertAlmostEqual(result["target_2"], 85.0, places=1)


class TestBacktestTradeDataclass(unittest.TestCase):
    def test_defaults(self):
        trade = backtester.BacktestTrade(
            ticker="AAPL", direction="BUY", entry_date="2026-01-01",
            entry_price=170.0, stop_loss=165.0, target_1=177.5, target_2=185.0,
        )
        self.assertEqual(trade.exit_date, "")
        self.assertEqual(trade.pnl, 0.0)
        self.assertEqual(trade.bars_held, 0)


class TestBacktestResultDataclass(unittest.TestCase):
    def test_defaults(self):
        result = backtester.BacktestResult(ticker="AAPL", period="2y")
        self.assertEqual(result.total_trades, 0)
        self.assertEqual(result.win_rate, 0.0)
        self.assertEqual(result.trades, [])

    def test_with_trades(self):
        trade = backtester.BacktestTrade(
            ticker="AAPL", direction="BUY", entry_date="2026-01-01",
            entry_price=170.0, stop_loss=165.0, target_1=177.5, target_2=185.0,
            exit_date="2026-01-10", exit_price=177.5, pnl=7.5, pnl_pct=4.4,
        )
        result = backtester.BacktestResult(
            ticker="AAPL", period="2y", total_trades=1,
            winning_trades=1, win_rate=100.0, trades=[trade],
        )
        self.assertEqual(result.winning_trades, 1)


class TestSaveResultsCsv(unittest.TestCase):
    def test_empty_results(self):
        result = backtester.BacktestResult(ticker="AAPL", period="2y")
        path = backtester.save_results_csv([result])
        self.assertIsInstance(path, str)


if __name__ == "__main__":
    unittest.main()
