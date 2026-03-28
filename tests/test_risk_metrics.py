"""Tests for risk-adjusted performance metrics."""

import math
import unittest

import risk_metrics


class TestComputeMetrics(unittest.TestCase):
    def test_no_data(self):
        """Empty P&L list should return error."""
        result = risk_metrics.compute_metrics([])
        self.assertIn("error", result)

    def test_all_wins(self):
        """All winning trades should produce positive metrics."""
        pnls = [100, 200, 150, 300, 250]
        metrics = risk_metrics.compute_metrics(pnls, starting_capital=10000)

        self.assertEqual(metrics["total_trades"], 5)
        self.assertAlmostEqual(metrics["total_pnl"], 1000)
        self.assertEqual(metrics["win_rate"], 100.0)
        self.assertGreater(metrics["sharpe_ratio"], 0)
        # Sortino = 0 when no downside (no losses to compute downside std)
        self.assertGreaterEqual(metrics["sortino_ratio"], 0)
        self.assertEqual(metrics["max_drawdown"], 0)  # No drawdown with all wins
        self.assertEqual(metrics["max_win_streak"], 5)

    def test_all_losses(self):
        """All losing trades should produce negative metrics."""
        pnls = [-50, -100, -75, -60, -80]
        metrics = risk_metrics.compute_metrics(pnls, starting_capital=10000)

        self.assertLess(metrics["total_pnl"], 0)
        self.assertEqual(metrics["win_rate"], 0)
        self.assertLess(metrics["sharpe_ratio"], 0)
        self.assertGreater(metrics["max_drawdown"], 0)
        self.assertEqual(metrics["max_loss_streak"], 5)

    def test_mixed_trades(self):
        """Mixed wins/losses should produce reasonable metrics."""
        pnls = [200, -50, 150, -30, 100, -80, 250, -40, 180, -60]
        metrics = risk_metrics.compute_metrics(pnls, starting_capital=50000)

        self.assertEqual(metrics["total_trades"], 10)
        self.assertEqual(metrics["win_rate"], 50.0)
        self.assertGreater(metrics["profit_factor"], 1)
        self.assertGreater(metrics["expectancy"], 0)
        self.assertGreater(metrics["max_drawdown"], 0)

    def test_sharpe_ratio_positive_for_good_strategy(self):
        """Consistent profits with low variance should give good Sharpe."""
        pnls = [100, 110, 95, 105, 102, 98, 107, 103, 99, 101] * 5
        metrics = risk_metrics.compute_metrics(pnls, starting_capital=50000)
        self.assertGreater(metrics["sharpe_ratio"], 0)

    def test_sortino_higher_than_sharpe_for_upside_skew(self):
        """Strategy with big wins and small losses should have Sortino > Sharpe."""
        pnls = [500, -50, 400, -30, 600, -40, 300, -20, 450, -35]
        metrics = risk_metrics.compute_metrics(pnls, starting_capital=50000)
        # Sortino should be higher because downside vol is lower than total vol
        self.assertGreater(metrics["sortino_ratio"], metrics["sharpe_ratio"])

    def test_max_drawdown_calculation(self):
        """Should correctly identify max drawdown."""
        # Equity: 10000 -> 10500 -> 10200 -> 9800 -> 10100
        # Max DD from 10500 to 9800 = 700 (~6.7%)
        pnls = [500, -300, -400, 300]
        metrics = risk_metrics.compute_metrics(pnls, starting_capital=10000)
        self.assertAlmostEqual(metrics["max_drawdown"], 700, places=0)
        self.assertAlmostEqual(metrics["max_drawdown_pct"], 6.67, places=0)

    def test_kelly_criterion(self):
        """Kelly should be between 0 and 100%."""
        pnls = [200, -100, 150, -80, 200, -90, 180, -70]
        metrics = risk_metrics.compute_metrics(pnls, starting_capital=50000)
        self.assertGreaterEqual(metrics["kelly_criterion"], 0)
        self.assertLessEqual(metrics["kelly_criterion"], 100)

    def test_equity_curve_length(self):
        """Equity curve should have n+1 points (starting + each trade)."""
        pnls = [100, -50, 200]
        metrics = risk_metrics.compute_metrics(pnls, starting_capital=10000)
        self.assertEqual(len(metrics["equity_curve"]), 4)
        self.assertEqual(metrics["equity_curve"][0], 10000)

    def test_recovery_factor(self):
        """Recovery factor = total P&L / max drawdown."""
        pnls = [500, -300, -200, 400, 300]
        metrics = risk_metrics.compute_metrics(pnls, starting_capital=10000)
        expected_rf = metrics["total_pnl"] / metrics["max_drawdown"]
        self.assertAlmostEqual(metrics["recovery_factor"], round(expected_rf, 2))

    def test_streaks(self):
        """Should correctly identify win and loss streaks."""
        pnls = [100, 200, 300, -50, -60, 100]  # 3 wins, 2 losses, 1 win
        metrics = risk_metrics.compute_metrics(pnls, starting_capital=10000)
        self.assertEqual(metrics["max_win_streak"], 3)
        self.assertEqual(metrics["max_loss_streak"], 2)

    def test_single_trade(self):
        """Should handle single trade without crashing."""
        metrics = risk_metrics.compute_metrics([150], starting_capital=10000)
        self.assertEqual(metrics["total_trades"], 1)
        self.assertAlmostEqual(metrics["total_pnl"], 150)


if __name__ == "__main__":
    unittest.main()
