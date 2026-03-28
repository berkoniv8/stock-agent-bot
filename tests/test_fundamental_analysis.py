"""Tests for fundamental analysis module."""

import unittest
from unittest.mock import patch

import fundamental_analysis


class TestScoreNewsSentiment(unittest.TestCase):
    def test_positive_articles(self):
        articles = [
            {"title": "Amazing growth and record profits", "description": "Company beats all expectations"},
            {"title": "Stock surges on great earnings", "description": "Revenue up 50%"},
        ]
        score = fundamental_analysis.score_news_sentiment(articles)
        self.assertGreater(score, 0)

    def test_negative_articles(self):
        articles = [
            {"title": "Terrible losses and declining revenue", "description": "Company misses badly"},
            {"title": "Stock crashes on fraud investigation", "description": "SEC probe announced"},
        ]
        score = fundamental_analysis.score_news_sentiment(articles)
        self.assertLess(score, 0)

    def test_empty_articles(self):
        score = fundamental_analysis.score_news_sentiment([])
        self.assertEqual(score, 0.0)

    def test_empty_text_articles(self):
        articles = [{"title": "", "description": ""}]
        score = fundamental_analysis.score_news_sentiment(articles)
        self.assertEqual(score, 0.0)

    def test_mixed_sentiment(self):
        articles = [
            {"title": "Great earnings", "description": "Record profits"},
            {"title": "Terrible guidance", "description": "Revenue declining"},
        ]
        score = fundamental_analysis.score_news_sentiment(articles)
        # Mixed should be close to 0
        self.assertGreater(score, -1)
        self.assertLess(score, 1)


class TestAnalyze(unittest.TestCase):
    @patch("fundamental_analysis.fetch_news")
    @patch("fundamental_analysis.fetch_fundamentals")
    def test_perfect_score(self, mock_fund, mock_news):
        mock_fund.return_value = {
            "pe_ratio": 15.0,  # Below Tech median of 30
            "eps_growth_yoy": 25.0,  # >10%
            "revenue_growth_qoq": 12.0,  # >5%
            "debt_to_equity": 0.5,  # <1.0
            "analyst_consensus": "strong_buy",
        }
        mock_news.return_value = [
            {"title": "Amazing breakthrough revenue record", "description": "Best quarter ever recorded"},
        ]

        result = fundamental_analysis.analyze("AAPL", "Technology")
        self.assertEqual(result.ticker, "AAPL")
        self.assertTrue(result.pe_below_sector_median)
        self.assertTrue(result.eps_growth_above_10)
        self.assertTrue(result.revenue_growth_above_5)
        self.assertTrue(result.debt_to_equity_healthy)
        self.assertTrue(result.analyst_buy_or_better)
        # Score should be 5 or 6 (news sentiment may vary)
        self.assertGreaterEqual(result.fundamental_score, 5)

    @patch("fundamental_analysis.fetch_news")
    @patch("fundamental_analysis.fetch_fundamentals")
    def test_zero_score(self, mock_fund, mock_news):
        mock_fund.return_value = {
            "pe_ratio": 50.0,  # Above median
            "eps_growth_yoy": -5.0,  # Negative
            "revenue_growth_qoq": 2.0,  # <5%
            "debt_to_equity": 2.5,  # >1.0
            "analyst_consensus": "sell",
        }
        mock_news.return_value = [
            {"title": "Terrible losses", "description": "Company in crisis"},
        ]

        result = fundamental_analysis.analyze("BAD", "Technology")
        self.assertFalse(result.pe_below_sector_median)
        self.assertFalse(result.eps_growth_above_10)
        self.assertFalse(result.revenue_growth_above_5)
        self.assertFalse(result.debt_to_equity_healthy)
        self.assertFalse(result.analyst_buy_or_better)
        self.assertEqual(result.fundamental_score, 0)

    @patch("fundamental_analysis.fetch_news")
    @patch("fundamental_analysis.fetch_fundamentals")
    def test_missing_data(self, mock_fund, mock_news):
        mock_fund.return_value = {}
        mock_news.return_value = []
        result = fundamental_analysis.analyze("UNKNOWN", "Technology")
        self.assertEqual(result.fundamental_score, 0)
        self.assertIsNone(result.pe_ratio)
        self.assertIsNone(result.eps_growth_yoy)

    @patch("fundamental_analysis.fetch_news")
    @patch("fundamental_analysis.fetch_fundamentals")
    def test_sector_median_pe(self, mock_fund, mock_news):
        mock_fund.return_value = {"pe_ratio": 13.0}
        mock_news.return_value = []
        # Financials median is 14, so 13 is below
        result = fundamental_analysis.analyze("JPM", "Financials")
        self.assertTrue(result.pe_below_sector_median)


class TestFundamentalSignalsDataclass(unittest.TestCase):
    def test_defaults(self):
        s = fundamental_analysis.FundamentalSignals()
        self.assertEqual(s.ticker, "")
        self.assertEqual(s.fundamental_score, 0)
        self.assertIsNone(s.pe_ratio)
        self.assertEqual(s.score_details, [])


class TestSectorMedianPE(unittest.TestCase):
    def test_known_sectors(self):
        self.assertEqual(fundamental_analysis.SECTOR_MEDIAN_PE["Technology"], 30)
        self.assertEqual(fundamental_analysis.SECTOR_MEDIAN_PE["Energy"], 12)
        self.assertIn("Healthcare", fundamental_analysis.SECTOR_MEDIAN_PE)


if __name__ == "__main__":
    unittest.main()
