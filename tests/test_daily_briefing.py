"""Tests for daily briefing generator."""

import unittest

import daily_briefing


class TestGenerateBriefing(unittest.TestCase):
    def test_returns_dict(self):
        briefing = daily_briefing.generate_briefing()
        self.assertIn("date", briefing)
        self.assertIn("time", briefing)
        self.assertIn("portfolio", briefing)
        self.assertIn("positions", briefing)
        self.assertIn("regime", briefing)
        self.assertIn("risk", briefing)
        self.assertIn("pending_reviews", briefing)

    def test_date_format(self):
        briefing = daily_briefing.generate_briefing()
        # Should be YYYY-MM-DD
        self.assertEqual(len(briefing["date"]), 10)
        self.assertEqual(briefing["date"][4], "-")


class TestFormatBriefing(unittest.TestCase):
    def test_format_empty_briefing(self):
        briefing = {
            "date": "2026-03-26",
            "time": "09:00",
            "portfolio": {},
            "positions": [],
            "regime": {},
            "risk": {},
            "pending_reviews": [],
        }
        text = daily_briefing.format_briefing(briefing)
        self.assertIn("DAILY BRIEFING", text)
        self.assertIn("2026-03-26", text)

    def test_format_with_portfolio(self):
        briefing = {
            "date": "2026-03-26",
            "time": "09:00",
            "portfolio": {
                "total_equity": 105000.0,
                "cash": 50000.0,
                "invested": 55000.0,
                "total_pnl": 5000.0,
                "total_return_pct": 5.0,
                "unrealized_pnl": 3000.0,
                "realized_pnl": 2000.0,
                "open_positions": 3,
                "total_trades": 10,
                "broker": "Interactive Brokers",
                "last_sync": "2026-03-26T09:00:00",
            },
            "positions": [
                {
                    "ticker": "AAPL",
                    "direction": "LONG",
                    "shares": 10,
                    "entry_price": 150.0,
                    "current_price": 155.0,
                    "current_value": 1550.0,
                    "cost_basis": 1500.0,
                    "unrealized_pnl": 50.0,
                    "unrealized_pct": 3.3,
                    "sector": "Technology",
                    "strategy": "long_term",
                },
            ],
            "top_movers": [],
            "regime": {
                "regime": "BULL_WEAK",
                "confidence": 45,
                "description": "Weakening uptrend",
                "threshold_adj": 0,
                "position_mult": 1.0,
            },
            "risk": {
                "sharpe_ratio": 1.5,
                "sortino_ratio": 2.1,
                "max_drawdown_pct": 3.5,
                "win_rate": 60.0,
                "profit_factor": 1.8,
                "kelly_criterion": 15.0,
            },
            "pending_reviews": [],
        }
        text = daily_briefing.format_briefing(briefing)
        self.assertIn("PORTFOLIO", text)
        self.assertIn("105,000", text)
        self.assertIn("AAPL", text)
        self.assertIn("BULL_WEAK", text)
        self.assertIn("Sharpe", text)
        self.assertIn("Interactive Brokers", text)

    def test_format_with_regime_adjustment(self):
        briefing = {
            "date": "2026-03-26",
            "time": "09:00",
            "portfolio": {},
            "positions": [],
            "top_movers": [],
            "regime": {
                "regime": "BEAR_STRONG",
                "confidence": 80,
                "description": "Strong downtrend",
                "threshold_adj": 2,
                "position_mult": 0.5,
            },
            "risk": {},
            "pending_reviews": [],
        }
        text = daily_briefing.format_briefing(briefing)
        self.assertIn("BEAR_STRONG", text)
        self.assertIn("80%", text)
        self.assertIn("Strong downtrend", text)


class TestGetters(unittest.TestCase):
    def test_portfolio_summary_returns_dict(self):
        result = daily_briefing.get_portfolio_summary()
        self.assertIsInstance(result, dict)

    def test_position_details_returns_list(self):
        result = daily_briefing.get_position_details()
        self.assertIsInstance(result, list)

    def test_regime_summary_returns_dict(self):
        result = daily_briefing.get_regime_summary()
        self.assertIsInstance(result, dict)

    def test_risk_snapshot_returns_dict(self):
        result = daily_briefing.get_risk_snapshot()
        self.assertIsInstance(result, dict)

    def test_journal_pending_returns_list(self):
        result = daily_briefing.get_journal_pending()
        self.assertIsInstance(result, list)


if __name__ == "__main__":
    unittest.main()
