"""Tests for position sizing module."""

import os
import unittest
from unittest.mock import patch, MagicMock

from technical_analysis import TechnicalSignals
from fundamental_analysis import FundamentalSignals
from signal_engine import TradeAlert
import position_sizing


def _make_tech(**kwargs):
    defaults = {
        "ticker": "AAPL", "current_price": 170.0, "ema9": 168.0,
        "ema21": 165.0, "sma200": 150.0, "support_level": 160.0,
        "resistance_level": 180.0, "atr_stop_long": 163.0,
        "atr_stop_short": 177.0, "fib_levels": {},
    }
    defaults.update(kwargs)
    return TechnicalSignals(**defaults)


def _make_alert(direction="BUY", tech=None):
    if tech is None:
        tech = _make_tech()
    fund = FundamentalSignals(ticker="AAPL", fundamental_score=4)
    return TradeAlert(
        ticker="AAPL", signal_score=6, direction=direction,
        triggered_signals=[("ema_cross_bullish", 2)],
        technical=tech, fundamental=fund,
    )


class TestCompute(unittest.TestCase):
    @patch.dict(os.environ, {
        "TOTAL_PORTFOLIO_VALUE": "100000",
        "AVAILABLE_CASH": "50000",
        "MAX_RISK_PER_TRADE_PCT": "1.0",
        "MAX_POSITION_SIZE_PCT": "10.0",
    })
    @patch("position_sizing.load_portfolio_config")
    def test_buy_plan(self, mock_config):
        mock_config.return_value = {
            "total_portfolio_value": 100000,
            "available_cash": 50000,
            "max_risk_per_trade_pct": 1.0,
            "max_position_size_pct": 10.0,
        }
        alert = _make_alert("BUY")
        plan = position_sizing.compute(alert)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.ticker, "AAPL")
        self.assertEqual(plan.direction, "BUY")
        self.assertGreater(plan.shares, 0)
        self.assertGreater(plan.entry_price, 0)
        self.assertLess(plan.stop_loss, plan.entry_price)
        self.assertGreater(plan.target_1, plan.entry_price)
        self.assertGreater(plan.target_2, plan.target_1)

    @patch("position_sizing.load_portfolio_config")
    def test_sell_plan(self, mock_config):
        mock_config.return_value = {
            "total_portfolio_value": 100000,
            "available_cash": 50000,
            "max_risk_per_trade_pct": 1.0,
            "max_position_size_pct": 10.0,
        }
        tech = _make_tech(current_price=170.0, ema21=175.0,
                          resistance_level=180.0, atr_stop_short=177.0)
        alert = _make_alert("SELL", tech=tech)
        plan = position_sizing.compute(alert)
        self.assertIsNotNone(plan)
        self.assertEqual(plan.direction, "SELL")
        self.assertGreater(plan.stop_loss, plan.entry_price)
        self.assertLess(plan.target_1, plan.entry_price)

    @patch("position_sizing.load_portfolio_config")
    def test_no_technical_returns_none(self, mock_config):
        mock_config.return_value = {
            "total_portfolio_value": 100000,
            "available_cash": 50000,
            "max_risk_per_trade_pct": 1.0,
            "max_position_size_pct": 10.0,
        }
        alert = TradeAlert(
            ticker="AAPL", signal_score=6, direction="BUY",
            triggered_signals=[], technical=None, fundamental=None,
        )
        plan = position_sizing.compute(alert)
        self.assertIsNone(plan)

    @patch("position_sizing.load_portfolio_config")
    def test_default_stop_when_no_candidates(self, mock_config):
        mock_config.return_value = {
            "total_portfolio_value": 100000,
            "available_cash": 50000,
            "max_risk_per_trade_pct": 1.0,
            "max_position_size_pct": 10.0,
        }
        tech = _make_tech(ema21=0, support_level=0, atr_stop_long=0)
        alert = _make_alert("BUY", tech=tech)
        plan = position_sizing.compute(alert)
        self.assertIsNotNone(plan)
        # Default stop is 3% below entry
        expected_stop = 170.0 * 0.97
        self.assertAlmostEqual(plan.stop_loss, round(expected_stop, 2), places=1)

    @patch("position_sizing.load_portfolio_config")
    def test_shares_limited_by_cash(self, mock_config):
        mock_config.return_value = {
            "total_portfolio_value": 100000,
            "available_cash": 500,  # Very limited cash
            "max_risk_per_trade_pct": 1.0,
            "max_position_size_pct": 10.0,
        }
        alert = _make_alert("BUY")
        plan = position_sizing.compute(alert)
        self.assertIsNotNone(plan)
        self.assertLessEqual(plan.position_value, 500)

    @patch("position_sizing.load_portfolio_config")
    def test_risk_reward_ratios(self, mock_config):
        mock_config.return_value = {
            "total_portfolio_value": 100000,
            "available_cash": 50000,
            "max_risk_per_trade_pct": 1.0,
            "max_position_size_pct": 10.0,
        }
        alert = _make_alert("BUY")
        plan = position_sizing.compute(alert)
        self.assertEqual(plan.risk_reward_t1, 1.5)
        self.assertEqual(plan.risk_reward_t2, 3.0)


class TestLoadPortfolioConfig(unittest.TestCase):
    @patch.dict(os.environ, {
        "TOTAL_PORTFOLIO_VALUE": "200000",
        "AVAILABLE_CASH": "75000",
    })
    @patch("builtins.open", side_effect=FileNotFoundError)
    def test_uses_env_defaults(self, mock_open):
        config = position_sizing.load_portfolio_config()
        self.assertEqual(config["total_portfolio_value"], 200000)
        self.assertEqual(config["available_cash"], 75000)


class TestPositionPlanDataclass(unittest.TestCase):
    def test_creation(self):
        plan = position_sizing.PositionPlan(
            ticker="AAPL", direction="BUY", entry_price=170.0,
            stop_loss=165.0, risk_per_share=5.0, shares=10,
            position_value=1700.0, target_1=177.5, target_2=185.0,
            target_3=190.0, max_loss=50.0, risk_reward_t1=1.5,
            risk_reward_t2=3.0,
        )
        self.assertEqual(plan.ticker, "AAPL")
        self.assertEqual(plan.max_loss, 50.0)


if __name__ == "__main__":
    unittest.main()
