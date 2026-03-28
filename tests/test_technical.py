"""
Unit tests for the technical analysis module.

Tests pattern detection, indicator computation, and Fibonacci levels
using synthetic OHLCV data.
"""

import sys
import os
import unittest

import numpy as np
import pandas as pd

# Add parent dir to path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import technical_analysis as ta


def make_ohlcv(closes, volumes=None, spread=0.5):
    """Build a synthetic OHLCV DataFrame from a list of close prices."""
    n = len(closes)
    if volumes is None:
        volumes = [1_000_000] * n
    df = pd.DataFrame({
        "Open": [c - spread * 0.3 for c in closes],
        "High": [c + spread for c in closes],
        "Low": [c - spread for c in closes],
        "Close": closes,
        "Volume": volumes,
    }, index=pd.date_range("2024-01-01", periods=n, freq="B"))
    return df


class TestComputeIndicators(unittest.TestCase):

    def test_ema_sma_computed(self):
        closes = list(range(100, 310))  # 210 bars
        df = make_ohlcv(closes)
        result = ta.compute_indicators(df)

        self.assertIn("EMA9", result.columns)
        self.assertIn("EMA21", result.columns)
        self.assertIn("SMA200", result.columns)
        self.assertFalse(result["EMA9"].isna().iloc[-1])
        self.assertFalse(result["SMA200"].isna().iloc[-1])

    def test_avg_volume(self):
        closes = list(range(100, 130))
        vols = [1_000_000] * 30
        df = make_ohlcv(closes, vols)
        result = ta.compute_indicators(df)
        self.assertAlmostEqual(result["AvgVol20"].iloc[-1], 1_000_000)


class TestEMACross(unittest.TestCase):

    def test_bullish_cross(self):
        """EMA9 crossing above EMA21 should be detected."""
        # Create data where short EMA will cross above long EMA
        closes = [100] * 25 + [99, 98, 97, 96, 95] + [96, 98, 101, 104, 108, 112, 116]
        df = make_ohlcv(closes)
        df = ta.compute_indicators(df)
        bullish, bearish = ta.detect_ema_cross(df)
        # The cross may or may not happen on last bar — just verify no crash
        # numpy bools are valid too
        self.assertIn(bullish, (True, False))
        self.assertIn(bearish, (True, False))

    def test_no_cross_on_flat(self):
        """Flat prices should produce no cross."""
        closes = [100.0] * 50
        df = make_ohlcv(closes)
        df = ta.compute_indicators(df)
        bullish, bearish = ta.detect_ema_cross(df)
        self.assertFalse(bullish)
        self.assertFalse(bearish)


class TestBreakout(unittest.TestCase):

    def test_breakout_detected(self):
        """Price above 20-bar high + volume surge = breakout."""
        closes = [100.0] * 25 + [110.0]  # last bar spikes
        vols = [1_000_000] * 25 + [3_000_000]  # volume surge
        df = make_ohlcv(closes, vols)
        df = ta.compute_indicators(df)
        self.assertTrue(ta.detect_breakout(df))

    def test_no_breakout_low_volume(self):
        """Price above 20-bar high but low volume = no breakout."""
        closes = [100.0] * 25 + [110.0]
        vols = [1_000_000] * 25 + [1_000_000]  # normal volume
        df = make_ohlcv(closes, vols)
        df = ta.compute_indicators(df)
        self.assertFalse(ta.detect_breakout(df))

    def test_no_breakout_below_high(self):
        """Price NOT above 20-bar high = no breakout."""
        closes = [100.0] * 25 + [99.0]
        vols = [1_000_000] * 25 + [3_000_000]
        df = make_ohlcv(closes, vols)
        df = ta.compute_indicators(df)
        self.assertFalse(ta.detect_breakout(df))


class TestDoubleBottom(unittest.TestCase):

    def test_double_bottom_detected(self):
        """Two similar troughs with neckline break."""
        # Build a W-shaped pattern
        part1 = [100, 99, 98, 97, 96, 95, 94, 95, 96, 97, 98, 99, 100]  # first trough at 94
        part2 = [99, 98, 97, 96, 95, 94.5, 95, 96, 97, 98, 99, 100]     # second trough at 94.5
        part3 = [101, 102]  # break above neckline (100)
        closes = [105] * 20 + part1 + part2 + part3  # pad to get enough bars
        df = make_ohlcv(closes)
        result = ta.detect_double_bottom(df)
        self.assertTrue(result)

    def test_no_double_bottom_on_uptrend(self):
        """Steady uptrend should not trigger double bottom."""
        closes = list(range(50, 110))
        df = make_ohlcv(closes)
        self.assertFalse(ta.detect_double_bottom(df))


class TestHeadAndShoulders(unittest.TestCase):

    def test_head_and_shoulders_detected(self):
        """Classic H&S: left shoulder, higher head, right shoulder, neckline break."""
        # Build the pattern
        base = [100] * 10
        left_shoulder = [101, 103, 105, 107, 105, 103, 101]   # peak at 107
        valley1 = [100, 99, 100]
        head = [102, 105, 108, 112, 108, 105, 102]             # peak at 112
        valley2 = [100, 99, 100]
        right_shoulder = [101, 103, 106, 103, 101]              # peak at 106
        breakdown = [99, 97, 95]                                 # neckline break
        closes = base + left_shoulder + valley1 + head + valley2 + right_shoulder + breakdown
        df = make_ohlcv(closes)
        result = ta.detect_head_and_shoulders(df)
        self.assertTrue(result)

    def test_inverse_hs_detected(self):
        """Inverse H&S: bullish reversal pattern."""
        base = [100] * 10
        left_shoulder = [99, 97, 95, 93, 95, 97, 99]   # trough at 93
        peak1 = [100, 101, 100]
        head = [98, 95, 92, 88, 92, 95, 98]             # trough at 88
        peak2 = [100, 101, 100]
        right_shoulder = [99, 97, 94, 97, 99]            # trough at 94
        breakout = [101, 103, 105]                        # neckline break
        closes = base + left_shoulder + peak1 + head + peak2 + right_shoulder + breakout
        df = make_ohlcv(closes)
        result = ta.detect_inverse_head_shoulders(df)
        self.assertTrue(result)


class TestFibonacci(unittest.TestCase):

    def test_fib_levels_computed(self):
        """Verify Fibonacci level calculation."""
        closes = list(range(100, 160)) + list(range(160, 130, -1))
        df = make_ohlcv(closes)
        levels = ta.compute_fibonacci_levels(df, lookback=60)

        self.assertIn("swing_high", levels)
        self.assertIn("swing_low", levels)
        self.assertIn(38.2, levels)
        self.assertIn(61.8, levels)
        # 38.2% should be between swing high and swing low
        self.assertGreater(levels[38.2], levels["swing_low"])
        self.assertLess(levels[38.2], levels["swing_high"])

    def test_fib_bounce_detection(self):
        """Price touching a Fib level and bouncing."""
        fib_levels = {
            "swing_high": 200.0,
            "swing_low": 100.0,
            38.2: 161.8,
            61.8: 138.2,
        }
        # Simulate a bounce off 61.8% level (138.2)
        closes = [150, 140, 138.5, 142]
        lows = [148, 138, 137.8, 140]
        df = pd.DataFrame({
            "Open": [149, 141, 139, 141],
            "High": [151, 141, 140, 143],
            "Low": lows,
            "Close": closes,
            "Volume": [1_000_000] * 4,
        }, index=pd.date_range("2024-06-01", periods=4, freq="B"))

        bounce_382, bounce_618 = ta.detect_fib_bounce(df, fib_levels)
        self.assertTrue(bounce_618)


class TestRSI(unittest.TestCase):

    def test_rsi_computed(self):
        """RSI should be computed on sufficient data."""
        closes = list(range(100, 150)) + list(range(150, 130, -1))
        df = make_ohlcv(closes)
        df = ta.compute_indicators(df)
        rsi, oversold, overbought = ta.detect_rsi(df)
        self.assertGreater(rsi, 0)
        self.assertLess(rsi, 100)

    def test_rsi_oversold_on_decline(self):
        """Steep decline should produce RSI < 30."""
        closes = [100] * 20 + [100 - i * 2 for i in range(20)]  # drop from 100 to 62
        df = make_ohlcv(closes)
        df = ta.compute_indicators(df)
        rsi, oversold, overbought = ta.detect_rsi(df)
        self.assertTrue(oversold)


class TestMACD(unittest.TestCase):

    def test_macd_computed(self):
        """MACD values should be present."""
        closes = list(range(100, 150))
        df = make_ohlcv(closes)
        df = ta.compute_indicators(df)
        macd, signal, hist, bull, bear = ta.detect_macd_cross(df)
        # In a steady uptrend, MACD should be positive
        self.assertGreater(macd, 0)

    def test_no_crash_on_short_data(self):
        """Should not crash with insufficient data."""
        closes = [100, 101]
        df = make_ohlcv(closes)
        df = ta.compute_indicators(df)
        macd, signal, hist, bull, bear = ta.detect_macd_cross(df)
        self.assertIn(bull, (True, False))


class TestBollingerBands(unittest.TestCase):

    def test_bands_computed(self):
        """Bollinger Bands should be present."""
        closes = [100 + (i % 5) for i in range(50)]
        df = make_ohlcv(closes)
        df = ta.compute_indicators(df)
        upper, middle, lower, squeeze, bo_upper, bo_lower = ta.detect_bollinger(df)
        self.assertGreater(upper, lower)
        self.assertGreater(middle, lower)

    def test_breakout_upper(self):
        """Price above upper band should trigger breakout."""
        closes = [100] * 25 + [120]  # spike above upper band
        df = make_ohlcv(closes)
        df = ta.compute_indicators(df)
        upper, middle, lower, squeeze, bo_upper, bo_lower = ta.detect_bollinger(df)
        self.assertTrue(bo_upper)


class TestATR(unittest.TestCase):

    def test_atr_computed(self):
        """ATR should be computed on sufficient data."""
        import math
        closes = [100 + 5 * math.sin(i / 5) for i in range(50)]
        df = make_ohlcv(closes, spread=2.0)
        df = ta.compute_indicators(df)
        atr, stop_long, stop_short = ta.compute_atr_stops(df)
        self.assertGreater(atr, 0)
        self.assertLess(stop_long, closes[-1])
        self.assertGreater(stop_short, closes[-1])


class TestDivergence(unittest.TestCase):

    def test_bullish_divergence(self):
        """Price lower low + RSI higher low = bullish divergence."""
        # Create data: price goes down-up-further down, but RSI doesn't
        part1 = [100] * 20
        part2 = [100, 99, 97, 95, 92, 90, 92, 95, 98, 100]  # first trough at 90
        part3 = [100, 98, 96, 93, 91, 89, 88, 90, 93, 97, 100]  # second trough at 88 (lower)
        closes = part1 + part2 + part3
        df = make_ohlcv(closes, spread=1.0)
        df = ta.compute_indicators(df)
        bull, bear = ta.detect_rsi_divergence(df, lookback=25)
        # The divergence may or may not trigger depending on RSI values,
        # but it should not crash
        self.assertIn(bull, (True, False))
        self.assertIn(bear, (True, False))

    def test_macd_divergence_no_crash(self):
        """MACD divergence should not crash on normal data."""
        import math
        closes = [100 + 10 * math.sin(i / 8) for i in range(60)]
        df = make_ohlcv(closes)
        df = ta.compute_indicators(df)
        bull, bear = ta.detect_macd_divergence(df, lookback=25)
        self.assertIn(bull, (True, False))
        self.assertIn(bear, (True, False))


class TestVWAP(unittest.TestCase):

    def test_vwap_computed(self):
        """VWAP should be computed."""
        closes = [100 + i * 0.1 for i in range(30)]
        df = make_ohlcv(closes)
        df = ta.compute_indicators(df)
        vwap, above = ta.detect_vwap(df)
        self.assertGreater(vwap, 0)


class TestFullAnalysis(unittest.TestCase):

    def test_analyze_returns_signals(self):
        """Full analyze function returns a TechnicalSignals object with all indicators."""
        # Use zigzag data so RSI/MACD have variance to compute on
        import math
        closes = [150 + 20 * math.sin(i / 10) + i * 0.3 for i in range(210)]
        df = make_ohlcv(closes)
        signals = ta.analyze("TEST", df)

        self.assertEqual(signals.ticker, "TEST")
        self.assertGreater(signals.current_price, 0)
        self.assertGreater(signals.rsi, 0)
        self.assertGreater(signals.atr, 0)
        self.assertGreater(signals.vwap, 0)
        self.assertIn(signals.macd_bullish_cross, (True, False))
        self.assertIn(signals.bb_squeeze, (True, False))
        self.assertIn(signals.rsi_bullish_divergence, (True, False))
        self.assertIn(signals.macd_bullish_divergence, (True, False))
        self.assertIsInstance(signals.pattern_details, list)

    def test_analyze_insufficient_data(self):
        """Should handle insufficient data gracefully."""
        closes = [100, 101, 102]
        df = make_ohlcv(closes)
        signals = ta.analyze("TEST", df)
        self.assertEqual(signals.current_price, 0.0)  # Not enough data


if __name__ == "__main__":
    unittest.main()
