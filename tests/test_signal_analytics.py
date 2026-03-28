"""Tests for signal analytics module."""

import unittest

import signal_analytics


def _make_trades():
    return [
        {"signals": "ema_cross_bullish|breakout_with_volume", "pnl": "200",
         "exit_reason": "target_1", "bars_held": "5"},
        {"signals": "ema_cross_bullish", "pnl": "-80",
         "exit_reason": "stop_loss", "bars_held": "12"},
        {"signals": "breakout_with_volume|macd_bullish_cross", "pnl": "150",
         "exit_reason": "target_1", "bars_held": "8"},
        {"signals": "rsi_oversold_bounce", "pnl": "-50",
         "exit_reason": "stop_loss", "bars_held": "3"},
        {"signals": "ema_cross_bullish|rsi_oversold_bounce", "pnl": "300",
         "exit_reason": "target_1", "bars_held": "7"},
    ]


class TestAnalyzeBySignal(unittest.TestCase):
    def test_signal_counts(self):
        result = signal_analytics.analyze_by_signal(_make_trades())
        self.assertIn("ema_cross_bullish", result)
        self.assertEqual(result["ema_cross_bullish"]["total_trades"], 3)

    def test_win_rate(self):
        result = signal_analytics.analyze_by_signal(_make_trades())
        ema = result["ema_cross_bullish"]
        # 2 wins (200, 300), 1 loss (-80) => 66.7%
        self.assertAlmostEqual(ema["win_rate"], 66.7, places=0)

    def test_profit_factor(self):
        result = signal_analytics.analyze_by_signal(_make_trades())
        ema = result["ema_cross_bullish"]
        # wins: 200+300=500, losses: 80 => PF = 500/80 = 6.25
        self.assertAlmostEqual(ema["profit_factor"], 6.25, places=1)

    def test_expectancy(self):
        result = signal_analytics.analyze_by_signal(_make_trades())
        ema = result["ema_cross_bullish"]
        self.assertGreater(ema["expectancy"], 0)

    def test_empty_trades(self):
        result = signal_analytics.analyze_by_signal([])
        self.assertEqual(result, {})

    def test_empty_signals_skipped(self):
        trades = [{"signals": "", "pnl": "100", "exit_reason": "target_1", "bars_held": "5"}]
        result = signal_analytics.analyze_by_signal(trades)
        self.assertEqual(result, {})


class TestAnalyzeByCombination(unittest.TestCase):
    def test_combo_tracking(self):
        result = signal_analytics.analyze_by_combination(_make_trades())
        self.assertGreater(len(result), 0)
        # Check a known combo
        key = "ema_cross_bullish + breakout_with_volume"
        self.assertIn(key, result)
        self.assertEqual(result[key]["total_trades"], 1)

    def test_empty(self):
        result = signal_analytics.analyze_by_combination([])
        self.assertEqual(result, {})


class TestAnalyzeByExit(unittest.TestCase):
    def test_exit_counts(self):
        result = signal_analytics.analyze_by_exit(_make_trades())
        self.assertIn("target_1", result)
        self.assertEqual(result["target_1"]["total_trades"], 3)
        self.assertIn("stop_loss", result)
        self.assertEqual(result["stop_loss"]["total_trades"], 2)

    def test_avg_bars(self):
        result = signal_analytics.analyze_by_exit(_make_trades())
        stop = result["stop_loss"]
        # bars: 12, 3 => avg = 7.5
        self.assertAlmostEqual(stop["avg_bars"], 7.5, places=1)


class TestLoadBacktestTrades(unittest.TestCase):
    def test_no_files(self):
        trades = signal_analytics.load_backtest_trades("/nonexistent/path.csv")
        self.assertEqual(trades, [])


if __name__ == "__main__":
    unittest.main()
