"""Tests for strategy optimizer module."""

import json
import os
import tempfile
import unittest

import strategy_optimizer


def _make_trades(win_signals, lose_signals, n_wins=10, n_losses=5):
    """Generate synthetic trade data.

    Args:
        win_signals: pipe-separated signal string for winning trades.
        lose_signals: pipe-separated signal string for losing trades.
        n_wins: number of winning trades.
        n_losses: number of losing trades.
    """
    trades = []
    for _ in range(n_wins):
        trades.append({
            "signals": win_signals,
            "pnl": 100,
            "pnl_pct": 2.0,
            "exit_reason": "target_1",
        })
    for _ in range(n_losses):
        trades.append({
            "signals": lose_signals,
            "pnl": -80,
            "pnl_pct": -1.5,
            "exit_reason": "stop_loss",
        })
    return trades


class TestComputeSignalPerformance(unittest.TestCase):
    def test_basic_performance(self):
        trades = _make_trades("ema_cross|rsi_bounce", "ema_cross|bb_squeeze", 10, 5)
        perf = strategy_optimizer.compute_signal_performance(trades)

        self.assertIn("ema_cross", perf)
        self.assertIn("rsi_bounce", perf)
        self.assertIn("bb_squeeze", perf)

        # ema_cross appears in all 15 trades
        self.assertEqual(perf["ema_cross"]["trades"], 15)
        # rsi_bounce appears only in wins
        self.assertEqual(perf["rsi_bounce"]["trades"], 10)
        self.assertEqual(perf["rsi_bounce"]["win_rate"], 100.0)

    def test_win_rate_calculation(self):
        trades = _make_trades("sig_a", "sig_a", 6, 4)
        perf = strategy_optimizer.compute_signal_performance(trades)
        self.assertAlmostEqual(perf["sig_a"]["win_rate"], 60.0)

    def test_empty_trades(self):
        perf = strategy_optimizer.compute_signal_performance([])
        self.assertEqual(perf, {})

    def test_expectancy(self):
        trades = _make_trades("sig_a", "sig_a", 7, 3)
        perf = strategy_optimizer.compute_signal_performance(trades)
        # 7 wins at 100, 3 losses at -80
        # WR = 0.7, avg_win = 100, avg_loss = 80
        # Expectancy = 0.7 * 100 - 0.3 * 80 = 46
        self.assertAlmostEqual(perf["sig_a"]["expectancy"], 46.0)


class TestSuggestWeightAdjustments(unittest.TestCase):
    def test_suggest_increase_for_strong_signal(self):
        perf = {
            "strong_signal": {
                "trades": 20, "wins": 15, "losses": 5,
                "win_rate": 75.0, "profit_factor": 3.0,
                "expectancy": 50.0, "avg_pnl": 40.0,
            },
        }
        weights = {"strong_signal": 2}
        suggestions = strategy_optimizer.suggest_weight_adjustments(perf, weights)
        self.assertEqual(len(suggestions), 1)
        self.assertGreater(suggestions[0]["suggested_weight"], 2)

    def test_suggest_decrease_for_weak_signal(self):
        perf = {
            "weak_signal": {
                "trades": 20, "wins": 5, "losses": 15,
                "win_rate": 25.0, "profit_factor": 0.3,
                "expectancy": -40.0, "avg_pnl": -30.0,
            },
        }
        weights = {"weak_signal": 2}
        suggestions = strategy_optimizer.suggest_weight_adjustments(perf, weights)
        self.assertEqual(len(suggestions), 1)
        self.assertEqual(suggestions[0]["suggested_weight"], 0)

    def test_no_suggestion_for_insufficient_data(self):
        perf = {
            "rare_signal": {
                "trades": 2, "wins": 2, "losses": 0,
                "win_rate": 100.0, "profit_factor": float("inf"),
                "expectancy": 100.0, "avg_pnl": 100.0,
            },
        }
        weights = {"rare_signal": 2}
        suggestions = strategy_optimizer.suggest_weight_adjustments(perf, weights)
        self.assertEqual(len(suggestions), 0)

    def test_weight_capped_at_5(self):
        perf = {
            "excellent": {
                "trades": 30, "wins": 25, "losses": 5,
                "win_rate": 83.0, "profit_factor": 5.0,
                "expectancy": 80.0, "avg_pnl": 70.0,
            },
        }
        weights = {"excellent": 4}
        suggestions = strategy_optimizer.suggest_weight_adjustments(perf, weights)
        if suggestions:
            self.assertLessEqual(suggestions[0]["suggested_weight"], 5)

    def test_weight_not_below_zero(self):
        perf = {
            "bad": {
                "trades": 15, "wins": 4, "losses": 11,
                "win_rate": 26.0, "profit_factor": 0.2,
                "expectancy": -50.0, "avg_pnl": -40.0,
            },
        }
        weights = {"bad": 1}
        suggestions = strategy_optimizer.suggest_weight_adjustments(perf, weights)
        if suggestions:
            self.assertGreaterEqual(suggestions[0]["suggested_weight"], 0)


class TestComputeOptimizedWeights(unittest.TestCase):
    def test_applies_suggestions(self):
        current = {"sig_a": 2, "sig_b": 3, "sig_c": 1}
        suggestions = [
            {"signal": "sig_a", "suggested_weight": 4, "current_weight": 2, "change": 2},
            {"signal": "sig_c", "suggested_weight": 0, "current_weight": 1, "change": -1},
        ]
        optimized = strategy_optimizer.compute_optimized_weights(current, suggestions)
        self.assertEqual(optimized["sig_a"], 4)
        self.assertEqual(optimized["sig_b"], 3)  # Unchanged
        self.assertEqual(optimized["sig_c"], 0)


class TestSimulateWithWeights(unittest.TestCase):
    def test_simulation(self):
        trades = [
            {"signals": "sig_a|sig_b", "pnl": 100},
            {"signals": "sig_a", "pnl": -50},
            {"signals": "sig_b|sig_c", "pnl": 80},
        ]
        weights = {"sig_a": 3, "sig_b": 2, "sig_c": 1}
        results = strategy_optimizer.simulate_with_weights(trades, weights)

        # At threshold 4: sig_a|sig_b (5), sig_b|sig_c would not qualify (3)
        self.assertIn(4, results)
        self.assertIn(5, results)

    def test_empty_trades(self):
        results = strategy_optimizer.simulate_with_weights([], {"sig_a": 2})
        for threshold in [4, 5, 6, 7]:
            self.assertEqual(results[threshold]["trades"], 0)


class TestSaveLoadWeights(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._tmp.close()
        self._orig = strategy_optimizer.OPTIMIZED_WEIGHTS_FILE
        strategy_optimizer.OPTIMIZED_WEIGHTS_FILE = __import__("pathlib").Path(self._tmp.name)

    def tearDown(self):
        strategy_optimizer.OPTIMIZED_WEIGHTS_FILE = self._orig
        try:
            os.unlink(self._tmp.name)
        except OSError:
            pass

    def test_save_and_load(self):
        weights = {"sig_a": 3, "sig_b": 2}
        strategy_optimizer.save_optimized_weights(weights)
        loaded = strategy_optimizer.load_optimized_weights()
        self.assertEqual(loaded, weights)

    def test_load_nonexistent(self):
        os.unlink(self._tmp.name)
        strategy_optimizer.OPTIMIZED_WEIGHTS_FILE = __import__("pathlib").Path("/tmp/nonexistent_opt.json")
        result = strategy_optimizer.load_optimized_weights()
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
