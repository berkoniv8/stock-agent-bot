"""Tests for market regime detector."""

import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import market_regime


def _make_spy_data(n=250, trend=0.05):
    """Generate synthetic SPY-like OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    close = 400 + np.cumsum(np.random.randn(n) * 1.5 + trend)
    high = close + np.abs(np.random.randn(n) * 0.5)
    low = close - np.abs(np.random.randn(n) * 0.5)
    volume = np.random.randint(50e6, 150e6, size=n).astype(float)
    return pd.DataFrame({
        "Open": close + np.random.randn(n) * 0.2,
        "High": high, "Low": low, "Close": close, "Volume": volume,
    }, index=dates)


def _make_vix_data(n=120, level=18.0):
    """Generate synthetic VIX data."""
    np.random.seed(99)
    dates = pd.date_range("2025-06-01", periods=n, freq="B")
    close = level + np.random.randn(n) * 2
    close = np.clip(close, 10, 50)
    return pd.DataFrame({"Close": close}, index=dates)


class TestAnalyzeTrend(unittest.TestCase):
    def test_uptrend(self):
        df = _make_spy_data(250, trend=0.2)
        trend = market_regime.analyze_trend(df)
        self.assertTrue(trend["above_50sma"])
        self.assertTrue(trend["above_200sma"])
        self.assertGreater(trend["sma50_slope"], 0)

    def test_downtrend(self):
        df = _make_spy_data(250, trend=-0.2)
        trend = market_regime.analyze_trend(df)
        self.assertFalse(trend["above_50sma"])
        self.assertLess(trend["sma50_slope"], 0)

    def test_short_data(self):
        df = _make_spy_data(20)
        trend = market_regime.analyze_trend(df)
        self.assertIn("above_50sma", trend)

    def test_empty_data(self):
        trend = market_regime.analyze_trend(pd.DataFrame())
        self.assertFalse(trend["above_200sma"])
        self.assertEqual(trend["sma50_slope"], 0)

    def test_none_data(self):
        trend = market_regime.analyze_trend(None)
        self.assertEqual(trend["rsi"], 50.0)


class TestAnalyzeVolatility(unittest.TestCase):
    def test_normal_vix(self):
        df = _make_vix_data(120, level=16.0)
        vol = market_regime.analyze_volatility(df)
        self.assertFalse(vol["vix_elevated"])
        self.assertFalse(vol["vix_extreme"])

    def test_elevated_vix(self):
        df = _make_vix_data(120, level=25.0)
        vol = market_regime.analyze_volatility(df)
        self.assertTrue(vol["vix_elevated"])

    def test_extreme_vix(self):
        df = _make_vix_data(120, level=35.0)
        vol = market_regime.analyze_volatility(df)
        self.assertTrue(vol["vix_extreme"])

    def test_empty_data(self):
        vol = market_regime.analyze_volatility(pd.DataFrame())
        self.assertEqual(vol["vix_current"], 20.0)


class TestAnalyzeBreadth(unittest.TestCase):
    def test_normal_breadth(self):
        df = _make_spy_data(60)
        breadth = market_regime.analyze_breadth(df)
        self.assertGreater(breadth["up_day_pct_20d"], 0)
        self.assertLess(breadth["up_day_pct_20d"], 100)

    def test_empty_data(self):
        breadth = market_regime.analyze_breadth(pd.DataFrame())
        self.assertEqual(breadth["up_day_pct_20d"], 50.0)


class TestClassifyRegime(unittest.TestCase):
    def test_strong_bull(self):
        trend = {
            "above_50sma": True, "above_200sma": True,
            "sma50_slope": 0.3, "sma200_slope": 0.1,
            "sma50_above_200": True, "price_vs_200sma_pct": 5.0,
            "rsi": 65.0,
        }
        vol = {"vix_current": 14, "vix_sma20": 15, "vix_elevated": False,
               "vix_extreme": False, "vix_trend": "falling"}
        breadth = {"up_day_pct_20d": 65, "up_day_pct_5d": 70}

        regime, conf, _ = market_regime.classify_regime(trend, vol, breadth)
        self.assertEqual(regime, market_regime.BULL_STRONG)
        self.assertGreater(conf, 0)

    def test_strong_bear(self):
        trend = {
            "above_50sma": False, "above_200sma": False,
            "sma50_slope": -0.3, "sma200_slope": -0.1,
            "sma50_above_200": False, "price_vs_200sma_pct": -8.0,
            "rsi": 35.0,
        }
        vol = {"vix_current": 32, "vix_sma20": 28, "vix_elevated": True,
               "vix_extreme": True, "vix_trend": "rising"}
        breadth = {"up_day_pct_20d": 30, "up_day_pct_5d": 20}

        regime, conf, _ = market_regime.classify_regime(trend, vol, breadth)
        self.assertEqual(regime, market_regime.BEAR_STRONG)

    def test_neutral(self):
        trend = {
            "above_50sma": False, "above_200sma": True,
            "sma50_slope": 0.01, "sma200_slope": -0.01,
            "sma50_above_200": False, "price_vs_200sma_pct": 0.5,
            "rsi": 50.0,
        }
        vol = {"vix_current": 18, "vix_sma20": 18, "vix_elevated": False,
               "vix_extreme": False, "vix_trend": "stable"}
        breadth = {"up_day_pct_20d": 50, "up_day_pct_5d": 50}

        regime, _, _ = market_regime.classify_regime(trend, vol, breadth)
        self.assertEqual(regime, market_regime.NEUTRAL)


class TestDetectRegime(unittest.TestCase):
    def test_with_synthetic_data(self):
        spy = _make_spy_data(250, trend=0.1)
        vix = _make_vix_data(120, level=16)
        result = market_regime.detect_regime(spy_data=spy, vix_data=vix)

        self.assertIn("regime", result)
        self.assertIn("confidence", result)
        self.assertIn("params", result)
        self.assertIn(result["regime"], market_regime.REGIME_PARAMS)

    def test_params_structure(self):
        spy = _make_spy_data(250)
        vix = _make_vix_data(120)
        result = market_regime.detect_regime(spy_data=spy, vix_data=vix)
        params = result["params"]
        self.assertIn("threshold_adjustment", params)
        self.assertIn("position_size_mult", params)
        self.assertIn("long_bias", params)


class TestRegimeParams(unittest.TestCase):
    def test_all_regimes_have_params(self):
        for regime in [market_regime.BULL_STRONG, market_regime.BULL_WEAK,
                       market_regime.NEUTRAL, market_regime.BEAR_WEAK,
                       market_regime.BEAR_STRONG]:
            self.assertIn(regime, market_regime.REGIME_PARAMS)
            params = market_regime.REGIME_PARAMS[regime]
            self.assertIn("threshold_adjustment", params)
            self.assertIn("position_size_mult", params)
            self.assertIn("description", params)

    def test_bear_more_selective_than_bull(self):
        bull = market_regime.REGIME_PARAMS[market_regime.BULL_STRONG]
        bear = market_regime.REGIME_PARAMS[market_regime.BEAR_STRONG]
        self.assertGreater(bear["threshold_adjustment"], bull["threshold_adjustment"])
        self.assertLess(bear["position_size_mult"], bull["position_size_mult"])


class TestRegimeHistory(unittest.TestCase):
    def setUp(self):
        self._orig = market_regime.REGIME_FILE
        self._tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._tmp.close()
        market_regime.REGIME_FILE = Path(self._tmp.name)

    def tearDown(self):
        market_regime.REGIME_FILE = self._orig
        try:
            os.unlink(self._tmp.name)
        except OSError:
            pass

    def test_save_and_load(self):
        spy = _make_spy_data(250)
        vix = _make_vix_data(120)
        market_regime.detect_regime(spy_data=spy, vix_data=vix)

        history = market_regime.get_history(10)
        self.assertEqual(len(history), 1)
        self.assertIn("regime", history[0])
        self.assertIn("confidence", history[0])


if __name__ == "__main__":
    unittest.main()
