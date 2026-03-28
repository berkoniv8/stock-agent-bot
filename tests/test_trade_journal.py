"""Tests for trade journal module."""

import json
import os
import tempfile
import unittest

import trade_journal


class TestJournalBase(unittest.TestCase):
    """Base class that redirects journal to a temp file."""

    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._tmp.close()
        self._orig_path = trade_journal.JOURNAL_PATH
        trade_journal.JOURNAL_PATH = self._tmp.name
        # Start with empty journal
        with open(self._tmp.name, "w") as f:
            json.dump([], f)

    def tearDown(self):
        trade_journal.JOURNAL_PATH = self._orig_path
        try:
            os.unlink(self._tmp.name)
        except OSError:
            pass


class TestCreateEntry(TestJournalBase):
    def test_basic_create(self):
        entry = trade_journal.create_entry(
            ticker="AAPL", direction="BUY", entry_price=150.0,
            shares=10, stop_loss=145.0, target_1=160.0,
            signal_score=8, triggered_signals=["ema_cross", "rsi_oversold"],
        )
        self.assertEqual(entry["ticker"], "AAPL")
        self.assertEqual(entry["direction"], "BUY")
        self.assertEqual(entry["entry_price"], 150.0)
        self.assertIsNone(entry["exit_date"])
        self.assertTrue(entry["id"].startswith("AAPL_"))

    def test_with_tags_and_note(self):
        entry = trade_journal.create_entry(
            ticker="TSLA", direction="BUY", entry_price=200.0,
            shares=5, stop_loss=190.0, target_1=220.0,
            signal_score=7, triggered_signals=["breakout"],
            setup_note="Breaking out of consolidation",
            tags=["breakout", "momentum"],
            sector="Consumer Discretionary",
        )
        self.assertIn("breakout", entry["tags"])
        self.assertIn("momentum", entry["tags"])
        self.assertEqual(entry["sector"], "Consumer Discretionary")
        self.assertEqual(len(entry["notes"]), 1)

    def test_persisted_to_disk(self):
        trade_journal.create_entry(
            ticker="MSFT", direction="BUY", entry_price=300.0,
            shares=3, stop_loss=290.0, target_1=320.0,
            signal_score=6, triggered_signals=["trend"],
        )
        with open(trade_journal.JOURNAL_PATH) as f:
            data = json.load(f)
        self.assertEqual(len(data), 1)
        self.assertEqual(data[0]["ticker"], "MSFT")


class TestCloseEntry(TestJournalBase):
    def test_close_open_entry(self):
        trade_journal.create_entry(
            ticker="AAPL", direction="BUY", entry_price=150.0,
            shares=10, stop_loss=145.0, target_1=160.0,
            signal_score=8, triggered_signals=["ema_cross"],
        )
        closed = trade_journal.close_entry("AAPL", 162.0, "target_1", 120.0, 8.0)
        self.assertIsNotNone(closed)
        self.assertEqual(closed["exit_price"], 162.0)
        self.assertEqual(closed["pnl"], 120.0)
        self.assertEqual(closed["exit_reason"], "target_1")
        self.assertIsNotNone(closed["exit_date"])

    def test_close_nonexistent(self):
        result = trade_journal.close_entry("ZZZZ", 100.0, "stop", -50, -5.0)
        self.assertIsNone(result)

    def test_close_most_recent(self):
        """Should close the most recent open entry for the ticker."""
        trade_journal.create_entry(
            ticker="AAPL", direction="BUY", entry_price=140.0,
            shares=5, stop_loss=135.0, target_1=150.0,
            signal_score=6, triggered_signals=["a"],
        )
        trade_journal.close_entry("AAPL", 150.0, "target_1", 50.0, 7.1)

        trade_journal.create_entry(
            ticker="AAPL", direction="BUY", entry_price=155.0,
            shares=5, stop_loss=150.0, target_1=165.0,
            signal_score=7, triggered_signals=["b"],
        )
        closed = trade_journal.close_entry("AAPL", 160.0, "target_1", 25.0, 3.2)
        self.assertEqual(closed["entry_price"], 155.0)


class TestAddNote(TestJournalBase):
    def test_add_note(self):
        trade_journal.create_entry(
            ticker="AAPL", direction="BUY", entry_price=150.0,
            shares=10, stop_loss=145.0, target_1=160.0,
            signal_score=8, triggered_signals=["ema_cross"],
        )
        result = trade_journal.add_note("AAPL", "Price approaching resistance")
        self.assertIsNotNone(result)
        self.assertEqual(len(result["notes"]), 1)
        self.assertIn("resistance", result["notes"][0])

    def test_add_note_nonexistent(self):
        result = trade_journal.add_note("ZZZZ", "Test")
        self.assertIsNone(result)


class TestAddReview(TestJournalBase):
    def test_review_closed_trade(self):
        trade_journal.create_entry(
            ticker="AAPL", direction="BUY", entry_price=150.0,
            shares=10, stop_loss=145.0, target_1=160.0,
            signal_score=8, triggered_signals=["ema_cross"],
        )
        trade_journal.close_entry("AAPL", 162.0, "target_1", 120.0, 8.0)

        result = trade_journal.add_review(
            "AAPL", "Good execution, held through pullback",
            rating=4,
            lessons="Patience pays off",
            mistakes=["Sized too small"],
            what_went_well=["Followed the plan"],
        )
        self.assertIsNotNone(result)
        self.assertEqual(result["review_rating"], 4)
        self.assertEqual(result["lessons"], "Patience pays off")
        self.assertEqual(len(result["mistakes"]), 1)

    def test_review_open_trade_fails(self):
        trade_journal.create_entry(
            ticker="AAPL", direction="BUY", entry_price=150.0,
            shares=10, stop_loss=145.0, target_1=160.0,
            signal_score=8, triggered_signals=["ema_cross"],
        )
        result = trade_journal.add_review("AAPL", "Too early to review")
        self.assertIsNone(result)

    def test_rating_clamped(self):
        trade_journal.create_entry(
            ticker="AAPL", direction="BUY", entry_price=150.0,
            shares=10, stop_loss=145.0, target_1=160.0,
            signal_score=8, triggered_signals=["ema_cross"],
        )
        trade_journal.close_entry("AAPL", 162.0, "target_1", 120.0, 8.0)

        result = trade_journal.add_review("AAPL", "Test", rating=10)
        self.assertEqual(result["review_rating"], 5)

        result = trade_journal.add_review("AAPL", "Test", rating=-1)
        self.assertEqual(result["review_rating"], 1)


class TestAddTags(TestJournalBase):
    def test_add_tags(self):
        trade_journal.create_entry(
            ticker="AAPL", direction="BUY", entry_price=150.0,
            shares=10, stop_loss=145.0, target_1=160.0,
            signal_score=8, triggered_signals=["ema_cross"],
            tags=["momentum"],
        )
        result = trade_journal.add_tags("AAPL", ["breakout", "high-conviction"])
        self.assertIn("momentum", result["tags"])
        self.assertIn("breakout", result["tags"])
        self.assertIn("high-conviction", result["tags"])

    def test_no_duplicate_tags(self):
        trade_journal.create_entry(
            ticker="AAPL", direction="BUY", entry_price=150.0,
            shares=10, stop_loss=145.0, target_1=160.0,
            signal_score=8, triggered_signals=["ema_cross"],
            tags=["momentum"],
        )
        result = trade_journal.add_tags("AAPL", ["momentum", "breakout"])
        self.assertEqual(result["tags"].count("momentum"), 1)


class TestGetEntries(TestJournalBase):
    def _seed_entries(self):
        trade_journal.create_entry(
            ticker="AAPL", direction="BUY", entry_price=150.0,
            shares=10, stop_loss=145.0, target_1=160.0,
            signal_score=8, triggered_signals=["ema_cross"],
            tags=["momentum"],
        )
        trade_journal.create_entry(
            ticker="MSFT", direction="BUY", entry_price=300.0,
            shares=5, stop_loss=290.0, target_1=320.0,
            signal_score=7, triggered_signals=["breakout"],
            tags=["breakout"],
        )
        trade_journal.close_entry("AAPL", 162.0, "target_1", 120.0, 8.0)

    def test_get_all(self):
        self._seed_entries()
        entries = trade_journal.get_entries()
        self.assertEqual(len(entries), 2)

    def test_filter_by_ticker(self):
        self._seed_entries()
        entries = trade_journal.get_entries(ticker="MSFT")
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["ticker"], "MSFT")

    def test_filter_open_only(self):
        self._seed_entries()
        entries = trade_journal.get_entries(open_only=True)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["ticker"], "MSFT")

    def test_filter_closed_only(self):
        self._seed_entries()
        entries = trade_journal.get_entries(closed_only=True)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["ticker"], "AAPL")

    def test_filter_by_tags(self):
        self._seed_entries()
        entries = trade_journal.get_entries(tags=["momentum"])
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["ticker"], "AAPL")

    def test_limit(self):
        self._seed_entries()
        entries = trade_journal.get_entries(limit=1)
        self.assertEqual(len(entries), 1)

    def test_most_recent_first(self):
        self._seed_entries()
        entries = trade_journal.get_entries()
        # MSFT was created second, should be first
        self.assertEqual(entries[0]["ticker"], "MSFT")


class TestGetStats(TestJournalBase):
    def test_empty_journal(self):
        stats = trade_journal.get_stats()
        self.assertEqual(stats["total_entries"], 0)

    def test_stats_with_trades(self):
        trade_journal.create_entry(
            ticker="AAPL", direction="BUY", entry_price=150.0,
            shares=10, stop_loss=145.0, target_1=160.0,
            signal_score=8, triggered_signals=["ema_cross"],
            tags=["momentum"],
        )
        trade_journal.close_entry("AAPL", 162.0, "target_1", 120.0, 8.0)

        trade_journal.create_entry(
            ticker="MSFT", direction="BUY", entry_price=300.0,
            shares=5, stop_loss=290.0, target_1=320.0,
            signal_score=7, triggered_signals=["breakout"],
            tags=["breakout"],
        )
        trade_journal.close_entry("MSFT", 295.0, "stop_loss", -25.0, -1.7)

        stats = trade_journal.get_stats()
        self.assertEqual(stats["total_entries"], 2)
        self.assertEqual(stats["closed_trades"], 2)
        self.assertEqual(stats["wins"], 1)
        self.assertEqual(stats["losses"], 1)
        self.assertEqual(stats["win_rate"], 50.0)
        self.assertIn("momentum", stats["tag_counts"])
        self.assertIn("breakout", stats["tag_counts"])


class TestExportCSV(TestJournalBase):
    def test_export(self):
        trade_journal.create_entry(
            ticker="AAPL", direction="BUY", entry_price=150.0,
            shares=10, stop_loss=145.0, target_1=160.0,
            signal_score=8, triggered_signals=["ema_cross"],
            tags=["momentum", "swing"],
        )
        csv_path = self._tmp.name + ".csv"
        try:
            count = trade_journal.export_csv(csv_path)
            self.assertEqual(count, 1)
            self.assertTrue(os.path.exists(csv_path))
        finally:
            try:
                os.unlink(csv_path)
            except OSError:
                pass

    def test_export_empty(self):
        count = trade_journal.export_csv(self._tmp.name + ".csv")
        self.assertEqual(count, 0)


class TestLoadJournalEdgeCases(TestJournalBase):
    def test_corrupt_file(self):
        with open(trade_journal.JOURNAL_PATH, "w") as f:
            f.write("not json{{{")
        entries = trade_journal._load_journal()
        self.assertEqual(entries, [])

    def test_missing_file(self):
        os.unlink(trade_journal.JOURNAL_PATH)
        trade_journal.JOURNAL_PATH = "/tmp/nonexistent_journal_test.json"
        entries = trade_journal._load_journal()
        self.assertEqual(entries, [])


if __name__ == "__main__":
    unittest.main()
