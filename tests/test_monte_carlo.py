"""Tests for Monte Carlo simulation module."""

import unittest

import monte_carlo


class TestRunSimulation(unittest.TestCase):
    def test_basic_simulation(self):
        pnls = [100, -50, 200, -30, 150, -80, 120, -40, 180, -60]
        result = monte_carlo.run_simulation(
            pnls, starting_capital=100000,
            n_simulations=100, n_future_trades=50, seed=42,
        )
        self.assertNotIn("error", result)
        self.assertEqual(result["n_simulations"], 100)
        self.assertEqual(result["n_future_trades"], 50)
        self.assertEqual(result["starting_capital"], 100000)
        self.assertEqual(result["historical_trades"], 10)
        self.assertGreater(result["mean_final_equity"], 0)

    def test_empty_pnls(self):
        result = monte_carlo.run_simulation([], 100000)
        self.assertIn("error", result)

    def test_all_wins(self):
        pnls = [100, 200, 150, 250, 180]
        result = monte_carlo.run_simulation(
            pnls, 100000, n_simulations=50, n_future_trades=20, seed=1,
        )
        self.assertGreater(result["mean_final_equity"], 100000)
        self.assertEqual(result["loss_probability"], 0.0)
        self.assertEqual(result["historical_win_rate"], 100.0)

    def test_all_losses(self):
        pnls = [-100, -200, -150, -250, -180]
        result = monte_carlo.run_simulation(
            pnls, 100000, n_simulations=50, n_future_trades=20, seed=1,
        )
        self.assertLess(result["mean_final_equity"], 100000)
        self.assertEqual(result["loss_probability"], 100.0)

    def test_target_probabilities(self):
        pnls = [100, -50, 200, -30, 150, -80, 120, -40, 180, -60]
        result = monte_carlo.run_simulation(
            pnls, 100000, n_simulations=200, n_future_trades=50, seed=42,
        )
        targets = result["target_probabilities"]
        self.assertIn("10% gain", targets)
        self.assertIn("50% gain", targets)
        # 10% gain should be more likely than 50% gain
        self.assertGreaterEqual(targets["10% gain"], targets["50% gain"])

    def test_percentile_ordering(self):
        pnls = [100, -50, 200, -30, 150, -80, 120, -40, 180, -60]
        result = monte_carlo.run_simulation(
            pnls, 100000, n_simulations=200, n_future_trades=50, seed=42,
        )
        pf = result["percentile_final"]
        self.assertLessEqual(pf["p5"], pf["p25"])
        self.assertLessEqual(pf["p25"], pf["p50"])
        self.assertLessEqual(pf["p50"], pf["p75"])
        self.assertLessEqual(pf["p75"], pf["p95"])

    def test_drawdown_positive(self):
        pnls = [100, -200, 50, -150, 300]
        result = monte_carlo.run_simulation(
            pnls, 100000, n_simulations=50, n_future_trades=20, seed=42,
        )
        self.assertGreaterEqual(result["avg_max_drawdown"], 0)
        self.assertGreaterEqual(result["avg_max_drawdown_pct"], 0)

    def test_reproducibility_with_seed(self):
        pnls = [100, -50, 200, -30, 150]
        r1 = monte_carlo.run_simulation(pnls, 100000, n_simulations=50, seed=42)
        r2 = monte_carlo.run_simulation(pnls, 100000, n_simulations=50, seed=42)
        self.assertEqual(r1["mean_final_equity"], r2["mean_final_equity"])

    def test_custom_ruin_level(self):
        pnls = [-500, -300, -400, 100, -200]
        result = monte_carlo.run_simulation(
            pnls, 100000, n_simulations=100, n_future_trades=50,
            ruin_level=90000, seed=42,
        )
        self.assertEqual(result["ruin_level"], 90000)
        self.assertGreater(result["ruin_probability"], 0)

    def test_equity_bands_present(self):
        pnls = [100, -50, 200, -30]
        result = monte_carlo.run_simulation(
            pnls, 100000, n_simulations=50, n_future_trades=30, seed=42,
        )
        bands = result["equity_bands"]
        self.assertIn("p50", bands)
        self.assertIn("p5", bands)
        self.assertIn("p95", bands)


class TestKellyOptimal(unittest.TestCase):
    def test_positive_kelly(self):
        pnls = [200, -100, 150, -80, 200, -90, 180, -70]
        kelly = monte_carlo.compute_kelly_optimal(pnls)
        self.assertGreater(kelly["kelly_fraction"], 0)
        self.assertGreater(kelly["half_kelly"], 0)
        self.assertLess(kelly["half_kelly"], kelly["kelly_fraction"])

    def test_all_wins(self):
        pnls = [100, 200, 150]
        kelly = monte_carlo.compute_kelly_optimal(pnls)
        self.assertEqual(kelly["kelly_fraction"], 0)  # Can't compute without losses

    def test_all_losses(self):
        pnls = [-100, -200, -150]
        kelly = monte_carlo.compute_kelly_optimal(pnls)
        self.assertEqual(kelly["kelly_fraction"], 0)

    def test_empty(self):
        kelly = monte_carlo.compute_kelly_optimal([])
        self.assertEqual(kelly["kelly_fraction"], 0)


if __name__ == "__main__":
    unittest.main()
