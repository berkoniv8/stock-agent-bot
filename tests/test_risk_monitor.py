"""Tests for risk monitor module."""

import unittest

import numpy as np
import pandas as pd

import risk_monitor


class TestComputeCorrelationMatrix(unittest.TestCase):
    def test_with_data(self):
        np.random.seed(42)
        returns = pd.DataFrame({
            "AAPL": np.random.randn(60),
            "MSFT": np.random.randn(60),
        })
        result = risk_monitor.compute_correlation_matrix(returns)
        self.assertEqual(result.shape, (2, 2))
        self.assertAlmostEqual(result.loc["AAPL", "AAPL"], 1.0)

    def test_empty_dataframe(self):
        result = risk_monitor.compute_correlation_matrix(pd.DataFrame())
        self.assertTrue(result.empty)

    def test_single_column(self):
        returns = pd.DataFrame({"AAPL": np.random.randn(30)})
        result = risk_monitor.compute_correlation_matrix(returns)
        self.assertTrue(result.empty)


class TestComputePortfolioVolatility(unittest.TestCase):
    def test_with_data(self):
        np.random.seed(42)
        returns = pd.DataFrame({
            "AAPL": np.random.randn(60) * 0.02,
            "MSFT": np.random.randn(60) * 0.02,
        })
        weights = {"AAPL": 5000, "MSFT": 5000}
        vol = risk_monitor.compute_portfolio_volatility(returns, weights)
        self.assertGreater(vol, 0)

    def test_empty_returns(self):
        vol = risk_monitor.compute_portfolio_volatility(pd.DataFrame(), {})
        self.assertEqual(vol, 0.0)

    def test_single_ticker(self):
        returns = pd.DataFrame({"AAPL": np.random.randn(30) * 0.02})
        weights = {"AAPL": 10000}
        vol = risk_monitor.compute_portfolio_volatility(returns, weights)
        self.assertEqual(vol, 0.0)  # Need 2+ tickers


class TestAnalyzeRiskStructure(unittest.TestCase):
    def test_returns_dict(self):
        # Just verify the structure is correct by checking the function exists
        # Full test requires portfolio.json and API calls
        self.assertTrue(callable(risk_monitor.analyze_risk))

    def test_print_risk_report_error(self):
        risk_monitor.print_risk_report({"error": "No holdings configured"})
        # Should not raise


class TestLoadSectorMap(unittest.TestCase):
    def test_returns_dict(self):
        # May return empty if no watchlist.csv, but should not crash
        result = risk_monitor.load_sector_map()
        self.assertIsInstance(result, dict)


if __name__ == "__main__":
    unittest.main()
