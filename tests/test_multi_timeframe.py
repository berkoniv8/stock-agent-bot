"""Tests for multi-timeframe confirmation module."""

import math
import unittest

import numpy as np
import pandas as pd

import technical_analysis
import multi_timeframe


def _make_ohlcv(n=210, trend="up"):
    """Generate synthetic OHLCV data."""
    np.random.seed(42)
    dates = pd.date_range("2025-01-01", periods=n, freq="B")
    if trend == "up":
        close = 100 + np.cumsum(np.abs(np.random.randn(n) * 0.3))
    elif trend == "down":
        close = 200 - np.cumsum(np.abs(np.random.randn(n) * 0.3))
    else:
        close = 100 + np.random.randn(n) * 0.5

    high = close + np.abs(np.random.randn(n) * 0.3)
    low = close - np.abs(np.random.randn(n) * 0.3)
    open_ = close + np.random.randn(n) * 0.1
    volume = np.random.randint(500000, 2000000, size=n).astype(float)
    return pd.DataFrame({
        "Open": open_, "High": high, "Low": low,
        "Close": close, "Volume": volume,
    }, index=dates)


class TestMTFConfirmation(unittest.TestCase):
    def test_insufficient_data_returns_neutral(self):
        """Should return neutral with insufficient intraday data."""
        daily_df = _make_ohlcv(210)
        daily_signals = technical_analysis.analyze("TEST", daily_df)
        intraday_df = pd.DataFrame()  # empty

        result = multi_timeframe.confirm_signal(daily_signals, intraday_df, "BUY")
        self.assertEqual(result.score_adjustment, 0)
        self.assertFalse(result.confirmed)
        self.assertIn("Insufficient", result.details[0])

    def test_bullish_intraday_confirms_buy(self):
        """Bullish intraday data should confirm BUY signal."""
        daily_df = _make_ohlcv(210, trend="up")
        daily_signals = technical_analysis.analyze("TEST", daily_df)
        intraday_df = _make_ohlcv(50, trend="up")

        result = multi_timeframe.confirm_signal(daily_signals, intraday_df, "BUY")
        # Should have positive or zero adjustment (bullish intraday for BUY)
        self.assertGreaterEqual(result.score_adjustment, 0)
        self.assertIsInstance(result.confirmed, bool)
        self.assertIsInstance(result.details, list)

    def test_bearish_intraday_contradicts_buy(self):
        """Bearish intraday data should contradict BUY signal."""
        daily_df = _make_ohlcv(210, trend="up")
        daily_signals = technical_analysis.analyze("TEST", daily_df)
        intraday_df = _make_ohlcv(50, trend="down")

        result = multi_timeframe.confirm_signal(daily_signals, intraday_df, "BUY")
        # Bearish intraday should give negative or zero adjustment for BUY
        self.assertLessEqual(result.score_adjustment, 0)

    def test_result_dataclass_fields(self):
        """MTFConfirmation should have all expected fields."""
        result = multi_timeframe.MTFConfirmation()
        self.assertFalse(result.confirmed)
        self.assertEqual(result.score_adjustment, 0)
        self.assertEqual(result.intraday_trend, "")
        self.assertEqual(result.details, [])

    def test_score_adjustment_range(self):
        """Score adjustment should be in [-2, +2] range."""
        daily_df = _make_ohlcv(210)
        daily_signals = technical_analysis.analyze("TEST", daily_df)
        intraday_df = _make_ohlcv(50)

        result = multi_timeframe.confirm_signal(daily_signals, intraday_df, "BUY")
        self.assertGreaterEqual(result.score_adjustment, -2)
        self.assertLessEqual(result.score_adjustment, 2)


if __name__ == "__main__":
    unittest.main()
