"""Tests for performance attribution module."""

import unittest

import performance_attribution


def _make_trades():
    """Generate sample closed trades."""
    return [
        {
            "ticker": "AAPL", "direction": "BUY", "pnl": 200, "pnl_pct": 4.0,
            "exit_reason": "target_1", "bars_held": 5,
            "triggered_signals": ["ema_cross_bullish", "rsi_oversold_bounce"],
            "exit_date": "2026-01-10",
        },
        {
            "ticker": "MSFT", "direction": "BUY", "pnl": -80, "pnl_pct": -1.5,
            "exit_reason": "stop_loss", "bars_held": 12,
            "triggered_signals": ["breakout_with_volume"],
            "exit_date": "2026-01-15",
        },
        {
            "ticker": "GOOGL", "direction": "BUY", "pnl": 350, "pnl_pct": 5.0,
            "exit_reason": "target_1", "bars_held": 8,
            "triggered_signals": ["ema_cross_bullish", "macd_bullish_cross"],
            "exit_date": "2026-01-20",
        },
        {
            "ticker": "JPM", "direction": "SELL", "pnl": -50, "pnl_pct": -1.0,
            "exit_reason": "stop_loss", "bars_held": 3,
            "triggered_signals": ["rsi_oversold_bounce"],
            "exit_date": "2026-01-25",
        },
        {
            "ticker": "AAPL", "direction": "BUY", "pnl": 150, "pnl_pct": 3.0,
            "exit_reason": "time_exit", "bars_held": 28,
            "triggered_signals": ["ema_cross_bullish"],
            "exit_date": "2026-02-05",
        },
    ]


class TestAttributeBySector(unittest.TestCase):
    def test_basic_sector_attribution(self):
        trades = _make_trades()
        sector_map = {
            "AAPL": "Technology", "MSFT": "Technology",
            "GOOGL": "Communication", "JPM": "Financials",
        }
        result = performance_attribution.attribute_by_sector(trades, sector_map)
        self.assertIn("Technology", result)
        self.assertEqual(result["Technology"]["trades"], 3)  # AAPL x2 + MSFT
        self.assertIn("Communication", result)
        self.assertIn("Financials", result)

    def test_contribution_sums_roughly(self):
        trades = _make_trades()
        sector_map = {"AAPL": "Tech", "MSFT": "Tech", "GOOGL": "Tech", "JPM": "Fin"}
        result = performance_attribution.attribute_by_sector(trades, sector_map)
        total = sum(abs(s["pnl_contribution_pct"]) for s in result.values())
        self.assertGreater(total, 0)

    def test_empty_trades(self):
        result = performance_attribution.attribute_by_sector([], {})
        self.assertEqual(result, {})


class TestAttributeBySignal(unittest.TestCase):
    def test_signal_counts(self):
        trades = _make_trades()
        result = performance_attribution.attribute_by_signal(trades)
        self.assertIn("ema_cross_bullish", result)
        # ema_cross_bullish appears in 3 trades: AAPL, GOOGL, AAPL
        self.assertEqual(result["ema_cross_bullish"]["trades"], 3)

    def test_signal_win_rates(self):
        trades = _make_trades()
        result = performance_attribution.attribute_by_signal(trades)
        # rsi_oversold_bounce: 1 win (AAPL +200), 1 loss (JPM -50) = 50% WR
        rsi = result["rsi_oversold_bounce"]
        self.assertEqual(rsi["trades"], 2)
        self.assertEqual(rsi["win_rate"], 50.0)


class TestAttributeByDirection(unittest.TestCase):
    def test_direction_breakdown(self):
        trades = _make_trades()
        result = performance_attribution.attribute_by_direction(trades)
        self.assertIn("BUY", result)
        self.assertEqual(result["BUY"]["trades"], 4)
        self.assertIn("SELL", result)
        self.assertEqual(result["SELL"]["trades"], 1)

    def test_empty(self):
        result = performance_attribution.attribute_by_direction([])
        self.assertEqual(result, {})


class TestAttributeByHoldingPeriod(unittest.TestCase):
    def test_period_buckets(self):
        trades = _make_trades()
        result = performance_attribution.attribute_by_holding_period(trades)
        self.assertIn("1-3 days", result)
        self.assertIn("4-7 days", result)
        self.assertIn("15-30 days", result)
        # bars_held=3 → "1-3 days", bars_held=5 → "4-7 days", bars_held=8 → "8-14 days"
        self.assertEqual(result["1-3 days"]["trades"], 1)  # JPM (3)
        self.assertEqual(result["4-7 days"]["trades"], 1)  # AAPL (5)
        self.assertEqual(result["8-14 days"]["trades"], 2)  # GOOGL (8), MSFT (12)

    def test_empty(self):
        result = performance_attribution.attribute_by_holding_period([])
        for bucket in result.values():
            self.assertEqual(bucket["trades"], 0)


class TestAttributeByExitReason(unittest.TestCase):
    def test_exit_reasons(self):
        trades = _make_trades()
        result = performance_attribution.attribute_by_exit_reason(trades)
        self.assertIn("target_1", result)
        self.assertEqual(result["target_1"]["trades"], 2)
        self.assertIn("stop_loss", result)
        self.assertEqual(result["stop_loss"]["trades"], 2)
        self.assertIn("time_exit", result)


class TestRollingWinRate(unittest.TestCase):
    def test_rolling_computation(self):
        trades = _make_trades() * 3  # 15 trades
        result = performance_attribution.compute_rolling_win_rate(trades, window=5)
        self.assertGreater(len(result), 0)
        for r in result:
            self.assertIn("rolling_win_rate", r)
            self.assertGreaterEqual(r["rolling_win_rate"], 0)
            self.assertLessEqual(r["rolling_win_rate"], 100)

    def test_insufficient_trades(self):
        result = performance_attribution.compute_rolling_win_rate(
            _make_trades()[:3], window=10
        )
        self.assertEqual(result, [])


class TestFullAttribution(unittest.TestCase):
    def test_returns_dict_structure(self):
        result = performance_attribution.full_attribution()
        self.assertIn("total_trades", result)
        if result["total_trades"] > 0:
            self.assertIn("by_sector", result)
            self.assertIn("by_signal", result)
            self.assertIn("by_direction", result)
            self.assertIn("by_holding_period", result)
            self.assertIn("by_exit_reason", result)


if __name__ == "__main__":
    unittest.main()
