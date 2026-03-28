"""Tests for correlation guard module."""

import unittest

import numpy as np
import pandas as pd

import correlation_guard


def _make_returns(n=60, seed=42, trend=0.0):
    """Generate synthetic close prices and return as DataFrame."""
    np.random.seed(seed)
    close = 100 + np.cumsum(np.random.randn(n) * 0.5 + trend)
    dates = pd.date_range("2025-09-01", periods=n, freq="B")
    return pd.DataFrame({"Close": close}, index=dates)


class TestNormalizeSector(unittest.TestCase):
    def test_canonical(self):
        self.assertEqual(correlation_guard.normalize_sector("Technology"), "Technology")

    def test_alias(self):
        self.assertEqual(
            correlation_guard.normalize_sector("Financial Services"), "Financials"
        )
        self.assertEqual(
            correlation_guard.normalize_sector("Consumer Cyclical"),
            "Consumer Discretionary",
        )

    def test_empty(self):
        self.assertEqual(correlation_guard.normalize_sector(""), "Unknown")
        self.assertEqual(correlation_guard.normalize_sector(None), "Unknown")


class TestSectorConcentration(unittest.TestCase):
    def test_no_positions_allows_entry(self):
        safe, info = correlation_guard.check_sector_concentration("Technology", [])
        self.assertTrue(safe)
        self.assertEqual(info["current_count"], 0)

    def test_under_limit(self):
        positions = [
            {"ticker": "AAPL", "sector": "Technology"},
        ]
        safe, info = correlation_guard.check_sector_concentration(
            "Technology", positions, max_per_sector=2
        )
        self.assertTrue(safe)
        self.assertEqual(info["current_count"], 1)

    def test_at_limit_blocks(self):
        positions = [
            {"ticker": "AAPL", "sector": "Technology"},
            {"ticker": "MSFT", "sector": "Technology"},
        ]
        safe, info = correlation_guard.check_sector_concentration(
            "Technology", positions, max_per_sector=2
        )
        self.assertFalse(safe)
        self.assertIn("concentration", info["reason"])

    def test_different_sector_ok(self):
        positions = [
            {"ticker": "AAPL", "sector": "Technology"},
            {"ticker": "MSFT", "sector": "Technology"},
        ]
        safe, info = correlation_guard.check_sector_concentration(
            "Healthcare", positions, max_per_sector=2
        )
        self.assertTrue(safe)
        self.assertEqual(info["current_count"], 0)

    def test_alias_matching(self):
        """Financial Services and Financials should be treated as same sector."""
        positions = [
            {"ticker": "JPM", "sector": "Financial Services"},
            {"ticker": "GS", "sector": "Financials"},
        ]
        safe, info = correlation_guard.check_sector_concentration(
            "Financial Services", positions, max_per_sector=2
        )
        self.assertFalse(safe)


class TestPairwiseCorrelation(unittest.TestCase):
    def test_identical_series(self):
        """Same series should correlate at 1.0."""
        df = _make_returns(60, seed=1)
        returns = correlation_guard.get_returns_from_data(df)
        corr = correlation_guard.compute_pairwise_correlation(returns, returns)
        self.assertAlmostEqual(corr, 1.0, places=2)

    def test_independent_series(self):
        """Independent random series should have low correlation."""
        df_a = _make_returns(60, seed=1)
        df_b = _make_returns(60, seed=99)
        r_a = correlation_guard.get_returns_from_data(df_a)
        r_b = correlation_guard.get_returns_from_data(df_b)
        corr = correlation_guard.compute_pairwise_correlation(r_a, r_b)
        self.assertLess(abs(corr), 0.5)

    def test_none_returns_zero(self):
        corr = correlation_guard.compute_pairwise_correlation(None, None)
        self.assertEqual(corr, 0.0)

    def test_short_series(self):
        df = _make_returns(5)
        returns = correlation_guard.get_returns_from_data(df)
        self.assertIsNone(returns)


class TestPriceCorrelation(unittest.TestCase):
    def test_no_positions_is_safe(self):
        safe, info = correlation_guard.check_price_correlation(
            "AAPL", _make_returns(), [], {}
        )
        self.assertTrue(safe)
        self.assertEqual(info["avg_correlation"], 0.0)

    def test_high_correlation_blocks(self):
        """Two very similar series should be blocked."""
        np.random.seed(42)
        base_close = 100 + np.cumsum(np.random.randn(60) * 0.5)
        dates = pd.date_range("2025-09-01", periods=60, freq="B")
        # Make near-identical series
        df_a = pd.DataFrame({"Close": base_close}, index=dates)
        df_b = pd.DataFrame({"Close": base_close + np.random.randn(60) * 0.01}, index=dates)

        positions = [{"ticker": "EXISTING"}]
        data_cache = {"EXISTING": df_b}

        safe, info = correlation_guard.check_price_correlation(
            "NEW", df_a, positions, data_cache, max_correlation=0.5
        )
        self.assertFalse(safe)
        self.assertGreater(info["avg_correlation"], 0.5)

    def test_low_correlation_passes(self):
        """Independent series should pass."""
        df_a = _make_returns(60, seed=1)
        df_b = _make_returns(60, seed=99)

        positions = [{"ticker": "EXISTING"}]
        data_cache = {"EXISTING": df_b}

        safe, info = correlation_guard.check_price_correlation(
            "NEW", df_a, positions, data_cache, max_correlation=0.75
        )
        self.assertTrue(safe)

    def test_missing_data_allows_entry(self):
        """No candidate data should allow entry with warning."""
        positions = [{"ticker": "EXISTING"}]
        safe, info = correlation_guard.check_price_correlation(
            "NEW", None, positions, {}
        )
        self.assertTrue(safe)
        self.assertIn("Insufficient", info["reason"])


class TestCombinedCheck(unittest.TestCase):
    def test_safe_on_empty_portfolio(self):
        safe, info = correlation_guard.check_correlation_safe(
            "AAPL", "Technology", []
        )
        self.assertTrue(safe)

    def test_sector_block_checked_first(self):
        """Sector check should fire before price correlation."""
        positions = [
            {"ticker": "AAPL", "sector": "Technology"},
            {"ticker": "MSFT", "sector": "Technology"},
        ]
        safe, info = correlation_guard.check_correlation_safe(
            "GOOGL", "Technology", positions,
            data_cache=None, candidate_data=None,
        )
        # Default limit is 2, so this should block
        self.assertFalse(safe)
        self.assertEqual(info["check"], "sector_concentration")

    def test_passes_both_checks(self):
        df_cand = _make_returns(60, seed=1)
        df_exist = _make_returns(60, seed=99)
        positions = [{"ticker": "JPM", "sector": "Financials"}]
        data_cache = {"JPM": df_exist}

        safe, info = correlation_guard.check_correlation_safe(
            "AAPL", "Technology", positions,
            data_cache=data_cache, candidate_data=df_cand,
        )
        self.assertTrue(safe)
        self.assertEqual(info["check"], "passed")


if __name__ == "__main__":
    unittest.main()
