"""
Unit tests for the signal confluence engine and position sizing.
"""

import sys
import os
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from technical_analysis import TechnicalSignals
from fundamental_analysis import FundamentalSignals
import signal_engine
import position_sizing


class TestConfluenceScoring(unittest.TestCase):

    def _make_signals(self, **tech_kwargs):
        tech = TechnicalSignals(ticker="TEST", current_price=100.0, ema21=98.0, sma200=90.0)
        for k, v in tech_kwargs.items():
            setattr(tech, k, v)
        fund = FundamentalSignals(ticker="TEST")
        return tech, fund

    def test_no_signals_no_alert(self):
        tech, fund = self._make_signals()
        alert = signal_engine.evaluate(tech, fund, threshold=5)
        self.assertIsNone(alert)

    def test_threshold_met(self):
        tech, fund = self._make_signals(
            breakout_with_volume=True,   # +3
            ema_cross_bullish=True,      # +2
            price_above_200sma=True,     # +1
        )
        alert = signal_engine.evaluate(tech, fund, threshold=5)
        self.assertIsNotNone(alert)
        self.assertEqual(alert.signal_score, 6)
        self.assertEqual(alert.direction, "BUY")

    def test_below_threshold(self):
        tech, fund = self._make_signals(
            price_above_200sma=True,     # +1
        )
        alert = signal_engine.evaluate(tech, fund, threshold=5)
        self.assertIsNone(alert)

    def test_sell_direction_on_bearish(self):
        tech, fund = self._make_signals(
            head_and_shoulders=True,
            breakout_with_volume=True,   # +3
            price_above_200sma=True,     # +1
            fib_bounce_618=True,         # +2
        )
        alert = signal_engine.evaluate(tech, fund, threshold=5)
        self.assertIsNotNone(alert)
        self.assertEqual(alert.direction, "SELL")

    def test_fundamental_bonus(self):
        tech, fund = self._make_signals(
            breakout_with_volume=True,   # +3
        )
        fund.fundamental_score = 9       # +2 for 8+ (new 15-point scale)
        fund.news_sentiment_positive = True  # +1
        alert = signal_engine.evaluate(tech, fund, threshold=5)
        self.assertIsNotNone(alert)
        self.assertEqual(alert.signal_score, 6)

    def test_perfect_fundamental(self):
        tech, fund = self._make_signals(
            breakout_with_volume=True,   # +3
        )
        fund.fundamental_score = 13      # +2 (8+) +2 (12+ bonus) on 15-point scale
        alert = signal_engine.evaluate(tech, fund, threshold=5)
        self.assertIsNotNone(alert)
        self.assertEqual(alert.signal_score, 7)


class TestPositionSizing(unittest.TestCase):

    def test_basic_sizing(self):
        tech = TechnicalSignals(
            ticker="TEST",
            current_price=100.0,
            ema21=97.0,
            sma200=90.0,
            support_level=96.0,
            resistance_level=105.0,
            fib_levels={127.2: 112.0, 161.8: 120.0},
        )
        fund = FundamentalSignals(ticker="TEST", fundamental_score=4)
        alert = signal_engine.TradeAlert(
            ticker="TEST",
            signal_score=6,
            direction="BUY",
            triggered_signals=[("breakout_with_volume", 3), ("ema_cross_bullish", 2)],
            technical=tech,
            fundamental=fund,
        )
        plan = position_sizing.compute(alert)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.ticker, "TEST")
        self.assertEqual(plan.direction, "BUY")
        self.assertGreater(plan.shares, 0)
        self.assertGreater(plan.target_1, plan.entry_price)
        self.assertGreater(plan.target_2, plan.target_1)
        self.assertLess(plan.stop_loss, plan.entry_price)

    def test_sell_sizing(self):
        tech = TechnicalSignals(
            ticker="TEST",
            current_price=100.0,
            ema21=103.0,
            sma200=110.0,
            support_level=95.0,
            resistance_level=105.0,
        )
        fund = FundamentalSignals(ticker="TEST")
        alert = signal_engine.TradeAlert(
            ticker="TEST",
            signal_score=6,
            direction="SELL",
            triggered_signals=[("head_and_shoulders", 3)],
            technical=tech,
            fundamental=fund,
        )
        plan = position_sizing.compute(alert)
        self.assertIsNotNone(plan)
        self.assertGreater(plan.stop_loss, plan.entry_price)
        self.assertLess(plan.target_1, plan.entry_price)


if __name__ == "__main__":
    unittest.main()
