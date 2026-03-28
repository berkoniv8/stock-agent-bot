"""Tests for portfolio rebalancer module."""

import unittest

import rebalancer


class TestAnalyzeWeights(unittest.TestCase):
    def test_empty_portfolio(self):
        result = rebalancer.analyze_weights([], 100000)
        self.assertEqual(result["positions"], [])
        self.assertEqual(result["overweight"], [])

    def test_equal_weights(self):
        """Equally weighted positions should have no overweight flags."""
        positions = [
            {"ticker": "AAPL", "weight_pct": 10.0, "market_value": 10000, "current_price": 150},
            {"ticker": "MSFT", "weight_pct": 10.0, "market_value": 10000, "current_price": 300},
        ]
        result = rebalancer.analyze_weights(positions, 100000, max_position_pct=15)
        self.assertEqual(len(result["overweight"]), 0)

    def test_overweight_detection(self):
        positions = [
            {"ticker": "AAPL", "weight_pct": 25.0, "market_value": 25000, "current_price": 150},
            {"ticker": "MSFT", "weight_pct": 5.0, "market_value": 5000, "current_price": 300},
        ]
        result = rebalancer.analyze_weights(positions, 100000, max_position_pct=10)
        self.assertEqual(len(result["overweight"]), 1)
        self.assertEqual(result["overweight"][0]["ticker"], "AAPL")

    def test_equal_target_capped_by_max(self):
        """Equal target per position should not exceed max_position_pct."""
        positions = [
            {"ticker": "A", "weight_pct": 50, "market_value": 50000, "current_price": 100},
            {"ticker": "B", "weight_pct": 50, "market_value": 50000, "current_price": 100},
        ]
        result = rebalancer.analyze_weights(positions, 100000, max_position_pct=10)
        self.assertEqual(result["equal_target"], 10.0)


class TestAnalyzeSectors(unittest.TestCase):
    def test_empty(self):
        result = rebalancer.analyze_sectors([], 100000)
        self.assertEqual(result["sectors"], [])
        self.assertEqual(result["warnings"], [])

    def test_sector_weights(self):
        positions = [
            {"ticker": "AAPL", "sector": "Technology", "market_value": 30000},
            {"ticker": "MSFT", "sector": "Technology", "market_value": 20000},
            {"ticker": "JPM", "sector": "Financials", "market_value": 10000},
        ]
        result = rebalancer.analyze_sectors(positions, 100000, max_sector_pct=30)
        tech = [s for s in result["sectors"] if s["sector"] == "Technology"][0]
        self.assertEqual(tech["weight_pct"], 50.0)
        self.assertEqual(tech["positions"], 2)
        # 50% > 30% limit
        self.assertTrue(len(result["warnings"]) > 0)

    def test_no_warnings_within_limits(self):
        positions = [
            {"ticker": "AAPL", "sector": "Technology", "market_value": 10000},
            {"ticker": "JPM", "sector": "Financials", "market_value": 10000},
        ]
        result = rebalancer.analyze_sectors(positions, 100000, max_sector_pct=30)
        self.assertEqual(len(result["warnings"]), 0)


class TestAnalyzeRisk(unittest.TestCase):
    def test_risk_calculation(self):
        positions = [
            {
                "ticker": "AAPL",
                "direction": "BUY",
                "current_price": 150.0,
                "stop_loss": 140.0,
                "shares": 10,
            },
        ]
        result = rebalancer.analyze_risk(positions, 100000)
        # Risk = (150 - 140) * 10 = 100
        self.assertEqual(result["total_risk"], 100.0)
        self.assertAlmostEqual(result["total_risk_pct"], 0.1)

    def test_sell_direction_risk(self):
        positions = [
            {
                "ticker": "AAPL",
                "direction": "SELL",
                "current_price": 150.0,
                "stop_loss": 160.0,
                "shares": 10,
            },
        ]
        result = rebalancer.analyze_risk(positions, 100000)
        # Risk = (160 - 150) * 10 = 100
        self.assertEqual(result["total_risk"], 100.0)

    def test_empty_positions(self):
        result = rebalancer.analyze_risk([], 100000)
        self.assertEqual(result["total_risk"], 0)
        self.assertEqual(result["position_risks"], [])


class TestGenerateSuggestions(unittest.TestCase):
    def test_no_positions(self):
        suggestions = rebalancer.generate_suggestions([], 100000, 50000)
        self.assertEqual(suggestions, [])

    def test_trim_overweight(self):
        positions = [
            {
                "ticker": "AAPL",
                "weight_pct": 25.0,
                "market_value": 25000,
                "current_price": 150.0,
                "shares": 166,
            },
            {
                "ticker": "MSFT",
                "weight_pct": 5.0,
                "market_value": 5000,
                "current_price": 300.0,
                "shares": 16,
            },
        ]
        suggestions = rebalancer.generate_suggestions(
            positions, 100000, 70000, max_position_pct=10
        )
        trims = [s for s in suggestions if s["action"] == "TRIM"]
        self.assertTrue(len(trims) > 0)
        self.assertEqual(trims[0]["ticker"], "AAPL")

    def test_balanced_no_suggestions(self):
        """Balanced portfolio should generate no suggestions."""
        positions = [
            {
                "ticker": "AAPL",
                "weight_pct": 9.0,
                "market_value": 9000,
                "current_price": 150.0,
                "shares": 60,
            },
            {
                "ticker": "MSFT",
                "weight_pct": 9.0,
                "market_value": 9000,
                "current_price": 300.0,
                "shares": 30,
            },
        ]
        suggestions = rebalancer.generate_suggestions(
            positions, 100000, 82000, max_position_pct=10
        )
        self.assertEqual(suggestions, [])


class TestFullAnalysis(unittest.TestCase):
    def test_returns_dict(self):
        """Full analysis should return a dict with expected keys."""
        analysis = rebalancer.full_analysis()
        self.assertIn("total_equity", analysis)
        self.assertIn("cash", analysis)
        self.assertIn("weights", analysis)
        self.assertIn("sectors", analysis)
        self.assertIn("risk", analysis)
        self.assertIn("suggestions", analysis)


if __name__ == "__main__":
    unittest.main()
