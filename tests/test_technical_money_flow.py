"""Tests for money flow indicators: MFI, A/D line, OBV."""

import math
import unittest

import numpy as np
import pandas as pd

import technical_analysis


def _make_ohlcv(n=100):
    """Generate synthetic OHLCV data with volume patterns."""
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.1
    volume = np.random.randint(500000, 2000000, size=n).astype(float)
    df = pd.DataFrame({
        "Open": open_, "High": high, "Low": low,
        "Close": close, "Volume": volume,
    }, index=dates)
    return df


class TestMFI(unittest.TestCase):
    def test_mfi_computed(self):
        """MFI should be computed on sufficient data."""
        df = _make_ohlcv(100)
        df = technical_analysis.compute_indicators(df)
        self.assertIn("MFI", df.columns)
        mfi_val = df["MFI"].iloc[-1]
        self.assertFalse(pd.isna(mfi_val))
        self.assertGreaterEqual(mfi_val, 0)
        self.assertLessEqual(mfi_val, 100)

    def test_detect_mfi(self):
        """detect_mfi should return value and oversold/overbought flags."""
        df = _make_ohlcv(100)
        df = technical_analysis.compute_indicators(df)
        mfi, oversold, overbought = technical_analysis.detect_mfi(df)
        self.assertGreater(mfi, 0)
        self.assertIsInstance(oversold, (bool, np.bool_))
        self.assertIsInstance(overbought, (bool, np.bool_))

    def test_mfi_missing_column(self):
        """detect_mfi should handle missing MFI column gracefully."""
        df = pd.DataFrame({"Close": [100, 101]})
        mfi, oversold, overbought = technical_analysis.detect_mfi(df)
        self.assertEqual(mfi, 0.0)
        self.assertFalse(oversold)
        self.assertFalse(overbought)


class TestADLine(unittest.TestCase):
    def test_ad_computed(self):
        """A/D line should be computed."""
        df = _make_ohlcv(100)
        df = technical_analysis.compute_indicators(df)
        self.assertIn("AD", df.columns)
        self.assertFalse(pd.isna(df["AD"].iloc[-1]))

    def test_detect_ad_trend(self):
        """detect_ad_trend should return trend direction."""
        df = _make_ohlcv(100)
        df = technical_analysis.compute_indicators(df)
        ad_val, bullish, bearish = technical_analysis.detect_ad_trend(df)
        self.assertIsInstance(ad_val, float)
        # One of bullish/bearish should be True (unlikely exactly flat)
        self.assertNotEqual(bullish, bearish)

    def test_ad_insufficient_data(self):
        """detect_ad_trend should handle insufficient data."""
        df = _make_ohlcv(5)
        df = technical_analysis.compute_indicators(df)
        ad_val, bullish, bearish = technical_analysis.detect_ad_trend(df, lookback=10)
        self.assertEqual(ad_val, 0.0)


class TestOBV(unittest.TestCase):
    def test_obv_computed(self):
        """OBV should be computed."""
        df = _make_ohlcv(100)
        df = technical_analysis.compute_indicators(df)
        self.assertIn("OBV", df.columns)
        self.assertFalse(pd.isna(df["OBV"].iloc[-1]))

    def test_detect_obv(self):
        """detect_obv should return trend and divergence flags."""
        df = _make_ohlcv(100)
        df = technical_analysis.compute_indicators(df)
        obv_val, bull_trend, bear_trend, bull_div, bear_div = technical_analysis.detect_obv(df)
        self.assertIsInstance(obv_val, float)
        self.assertIsInstance(bull_div, (bool, np.bool_))
        self.assertIsInstance(bear_div, (bool, np.bool_))

    def test_obv_in_full_analysis(self):
        """Full analysis should populate OBV fields."""
        df = _make_ohlcv(210)
        signals = technical_analysis.analyze("TEST", df)
        self.assertIsInstance(signals.obv, float)
        self.assertIn(signals.obv_trend_bullish, (True, False))
        self.assertIn(signals.obv_trend_bearish, (True, False))


class TestMoneyFlowInAnalysis(unittest.TestCase):
    def test_all_money_flow_fields_populated(self):
        """Full analysis should populate all money flow fields."""
        df = _make_ohlcv(210)
        signals = technical_analysis.analyze("TEST", df)
        self.assertIsInstance(signals.mfi, float)
        self.assertIsInstance(signals.ad_line, float)
        self.assertIsInstance(signals.obv, float)
        self.assertIn(signals.mfi_oversold, (True, False))
        self.assertIn(signals.mfi_overbought, (True, False))
        self.assertIn(signals.ad_trend_bullish, (True, False))
        self.assertIn(signals.ad_trend_bearish, (True, False))


if __name__ == "__main__":
    unittest.main()
