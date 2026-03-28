"""Tests for sector rotation analysis — matches new sector_rotation.py API."""

import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np

import sector_rotation


def _make_close_series(n=130, trend=0.1, seed=42):
    """Return a synthetic Close price pd.Series."""
    np.random.seed(seed)
    dates = pd.date_range("2025-09-01", periods=n, freq="B")
    prices = 100 + np.cumsum(np.random.randn(n) * 0.5 + trend)
    return pd.Series(prices, index=dates, name="Close")


def _make_performance_dict():
    """Return a minimal but valid performance dict as returned by fetch_sector_performance."""
    sectors = list(sector_rotation.SECTOR_ETFS.values())
    perf = {}
    for i, sector in enumerate(sectors):
        etf = list(sector_rotation.SECTOR_ETFS.keys())[i]
        score = float(i - 5)  # ranges from -5 to +5
        perf[sector] = {
            "etf": etf,
            "return_1w": score * 0.5,
            "return_1m": score * 0.4,
            "return_3m": score * 0.3,
            "momentum_score": score,
            "rank": i + 1,
        }
    return perf


# ---------------------------------------------------------------------------
# _compute_return helper
# ---------------------------------------------------------------------------

class TestComputeReturn(unittest.TestCase):
    def test_positive_trend(self):
        """Uptrending close series should produce a positive return."""
        close = _make_close_series(130, trend=0.3)
        ret = sector_rotation._compute_return(close, 21)
        self.assertGreater(ret, 0)

    def test_insufficient_data_returns_zero(self):
        """Fewer bars than requested lookback should return 0."""
        close = _make_close_series(5, trend=0.1)
        ret = sector_rotation._compute_return(close, 21)
        self.assertEqual(ret, 0.0)

    def test_flat_returns_near_zero(self):
        """Flat (constant) price should return ~0."""
        close = pd.Series([100.0] * 50)
        ret = sector_rotation._compute_return(close, 21)
        self.assertAlmostEqual(ret, 0.0, places=4)


# ---------------------------------------------------------------------------
# _momentum_score helper
# ---------------------------------------------------------------------------

class TestMomentumScore(unittest.TestCase):
    def test_all_positive(self):
        """All positive returns → positive momentum score."""
        score = sector_rotation._momentum_score(2.0, 3.0, 5.0)
        self.assertGreater(score, 0)

    def test_all_negative(self):
        """All negative returns → negative momentum score."""
        score = sector_rotation._momentum_score(-2.0, -3.0, -5.0)
        self.assertLess(score, 0)

    def test_weighting_short_term_matters_more(self):
        """1w return has the highest weight (0.5)."""
        score_1w_high = sector_rotation._momentum_score(10.0, 0.0, 0.0)
        score_3m_high = sector_rotation._momentum_score(0.0, 0.0, 10.0)
        self.assertGreater(score_1w_high, score_3m_high)

    def test_known_values(self):
        """Check formula: 1w×0.5 + 1m×0.3 + 3m×0.2."""
        expected = 1.0 * 0.5 + 2.0 * 0.3 + 4.0 * 0.2
        result = sector_rotation._momentum_score(1.0, 2.0, 4.0)
        self.assertAlmostEqual(result, expected, places=4)


# ---------------------------------------------------------------------------
# detect_rotation
# ---------------------------------------------------------------------------

class TestDetectRotation(unittest.TestCase):
    def test_returns_required_keys(self):
        """detect_rotation should return dict with 4 expected keys."""
        perf = _make_performance_dict()
        rotation = sector_rotation.detect_rotation(perf)
        for key in ("rotating_into", "rotating_out_of", "momentum_leaders", "momentum_laggards"):
            self.assertIn(key, rotation)

    def test_leaders_laggards_are_lists(self):
        """Leaders and laggards should be non-empty lists."""
        perf = _make_performance_dict()
        rotation = sector_rotation.detect_rotation(perf)
        self.assertIsInstance(rotation["momentum_leaders"], list)
        self.assertIsInstance(rotation["momentum_laggards"], list)
        self.assertGreater(len(rotation["momentum_leaders"]), 0)
        self.assertGreater(len(rotation["momentum_laggards"]), 0)

    def test_rotation_into_detected(self):
        """Sector with 1w >> 1m should appear in rotating_into."""
        perf = {
            "Technology": {"etf": "XLK", "return_1w": 5.0, "return_1m": -1.0,
                           "return_3m": 0.0, "momentum_score": 2.2, "rank": 1},
            "Energy":     {"etf": "XLE", "return_1w": 0.0, "return_1m": 0.1,
                           "return_3m": 0.0, "momentum_score": 0.03, "rank": 2},
        }
        rotation = sector_rotation.detect_rotation(perf)
        self.assertIn("Technology", rotation["rotating_into"])

    def test_rotation_out_detected(self):
        """Sector with 1w << 1m should appear in rotating_out_of."""
        perf = {
            "Technology": {"etf": "XLK", "return_1w": -5.0, "return_1m": 2.0,
                           "return_3m": 0.0, "momentum_score": -1.9, "rank": 2},
            "Energy":     {"etf": "XLE", "return_1w": 0.0, "return_1m": 0.1,
                           "return_3m": 0.0, "momentum_score": 0.03, "rank": 1},
        }
        rotation = sector_rotation.detect_rotation(perf)
        self.assertIn("Technology", rotation["rotating_out_of"])

    def test_empty_performance(self):
        """Empty performance dict should return empty lists without error."""
        rotation = sector_rotation.detect_rotation({})
        for key in ("rotating_into", "rotating_out_of", "momentum_leaders", "momentum_laggards"):
            self.assertIsInstance(rotation[key], list)


# ---------------------------------------------------------------------------
# get_portfolio_exposure
# ---------------------------------------------------------------------------

class TestGetPortfolioExposure(unittest.TestCase):
    def _make_rotation(self, into=None, out_of=None, leaders=None, laggards=None):
        return {
            "rotating_into":    into or [],
            "rotating_out_of":  out_of or [],
            "momentum_leaders": leaders or [],
            "momentum_laggards": laggards or [],
        }

    def _make_portfolio(self, sectors):
        """Build a minimal portfolio dict with one holding per sector."""
        holdings = []
        for ticker, sector, value in sectors:
            holdings.append({
                "ticker": ticker,
                "sector": sector,
                "current_value": value,
            })
        return {"holdings": holdings}

    def test_aligned_holding_detected(self):
        """Holding in a rotating-into sector should appear in aligned."""
        rotation = self._make_rotation(into=["Technology"])
        portfolio = self._make_portfolio([("AAPL", "Technology", 10000)])
        exposure = sector_rotation.get_portfolio_exposure(rotation, portfolio)
        aligned_sectors = [a["sector"] for a in exposure["aligned"]]
        self.assertIn("Technology", aligned_sectors)

    def test_misaligned_holding_detected(self):
        """Holding in a rotating-out-of sector should appear in misaligned."""
        rotation = self._make_rotation(out_of=["Energy"])
        portfolio = self._make_portfolio([("XLE", "Energy", 5000)])
        exposure = sector_rotation.get_portfolio_exposure(rotation, portfolio)
        misaligned_sectors = [m["sector"] for m in exposure["misaligned"]]
        self.assertIn("Energy", misaligned_sectors)

    def test_returns_required_keys(self):
        rotation = self._make_rotation()
        exposure = sector_rotation.get_portfolio_exposure(rotation, {})
        for key in ("aligned", "misaligned", "suggestions"):
            self.assertIn(key, exposure)

    def test_suggestions_generated_for_misaligned(self):
        """Misaligned holdings should produce at least one suggestion."""
        rotation = self._make_rotation(out_of=["Energy"])
        portfolio = self._make_portfolio([("XLE", "Energy", 5000)])
        exposure = sector_rotation.get_portfolio_exposure(rotation, portfolio)
        self.assertGreater(len(exposure["suggestions"]), 0)

    def test_empty_portfolio(self):
        """Empty portfolio should return empty lists without error."""
        rotation = self._make_rotation(into=["Technology"])
        exposure = sector_rotation.get_portfolio_exposure(rotation, {})
        self.assertEqual(exposure["aligned"], [])
        self.assertEqual(exposure["misaligned"], [])


# ---------------------------------------------------------------------------
# format_report
# ---------------------------------------------------------------------------

class TestFormatReport(unittest.TestCase):
    def test_report_contains_headers(self):
        """Report should contain key section headers."""
        perf = _make_performance_dict()
        rotation = sector_rotation.detect_rotation(perf)
        exposure = sector_rotation.get_portfolio_exposure(rotation, {})
        report = sector_rotation.format_report(perf, rotation, exposure)
        self.assertIn("SECTOR ROTATION ANALYSIS", report)
        self.assertIn("SECTOR PERFORMANCE RANKINGS", report)
        self.assertIn("ROTATION SIGNALS", report)

    def test_report_is_string(self):
        perf = _make_performance_dict()
        rotation = sector_rotation.detect_rotation(perf)
        exposure = sector_rotation.get_portfolio_exposure(rotation, {})
        report = sector_rotation.format_report(perf, rotation, exposure)
        self.assertIsInstance(report, str)
        self.assertGreater(len(report), 100)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestSectorMapping(unittest.TestCase):
    def test_known_sectors(self):
        """Known sector names should map to ETFs via SECTOR_TO_ETF."""
        self.assertEqual(sector_rotation.SECTOR_TO_ETF.get("Technology"), "XLK")
        self.assertEqual(sector_rotation.SECTOR_TO_ETF.get("Healthcare"), "XLV")
        self.assertEqual(sector_rotation.SECTOR_TO_ETF.get("Energy"), "XLE")

    def test_unknown_sector_returns_none(self):
        """Unknown sector should return None."""
        self.assertIsNone(sector_rotation.SECTOR_TO_ETF.get("Alien Industries"))

    def test_sector_etfs_has_11_entries(self):
        """SECTOR_ETFS should track exactly 11 sectors."""
        self.assertEqual(len(sector_rotation.SECTOR_ETFS), 11)

    def test_sector_aliases_covers_yfinance_names(self):
        """SECTOR_ALIASES should handle common yfinance sector name variants."""
        aliases = sector_rotation.SECTOR_ALIASES
        self.assertEqual(aliases.get("Information Technology"), "Technology")
        self.assertEqual(aliases.get("Consumer Cyclical"), "Consumer Discretionary")
        self.assertEqual(aliases.get("Communication Services"), "Communication")


if __name__ == "__main__":
    unittest.main()
