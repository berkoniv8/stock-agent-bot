"""Tests for price alert system."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock

import price_alerts


class TestAddAlert(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._tmp.close()
        self._orig = price_alerts.ALERTS_FILE
        price_alerts.ALERTS_FILE = Path(self._tmp.name)
        # Start with empty alerts
        with open(self._tmp.name, "w") as f:
            json.dump([], f)

    def tearDown(self):
        price_alerts.ALERTS_FILE = self._orig
        try:
            os.unlink(self._tmp.name)
        except OSError:
            pass

    def test_add_above_alert(self):
        alert = price_alerts.add_alert("AAPL", "above", 180.0)
        self.assertEqual(alert["ticker"], "AAPL")
        self.assertEqual(alert["condition"], "above")
        self.assertEqual(alert["price"], 180.0)
        self.assertFalse(alert["triggered"])
        self.assertEqual(alert["type"], "CUSTOM")

    def test_add_below_alert(self):
        alert = price_alerts.add_alert("msft", "below", 300.0, note="support level")
        self.assertEqual(alert["ticker"], "MSFT")
        self.assertEqual(alert["condition"], "below")
        self.assertEqual(alert["note"], "support level")

    def test_alerts_persist(self):
        price_alerts.add_alert("AAPL", "above", 180.0)
        price_alerts.add_alert("GOOGL", "below", 140.0)
        with open(price_alerts.ALERTS_FILE) as f:
            data = json.load(f)
        self.assertEqual(len(data), 2)

    def test_add_creates_directory(self):
        nested = Path(self._tmp.name).parent / "sub" / "alerts.json"
        price_alerts.ALERTS_FILE = nested
        try:
            price_alerts.add_alert("AAPL", "above", 200.0)
            self.assertTrue(nested.exists())
        finally:
            try:
                os.unlink(str(nested))
                os.rmdir(str(nested.parent))
            except OSError:
                pass


class TestRemoveAlert(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._tmp.close()
        self._orig = price_alerts.ALERTS_FILE
        price_alerts.ALERTS_FILE = Path(self._tmp.name)
        with open(self._tmp.name, "w") as f:
            json.dump([], f)

    def tearDown(self):
        price_alerts.ALERTS_FILE = self._orig
        try:
            os.unlink(self._tmp.name)
        except OSError:
            pass

    def test_remove_valid_index(self):
        price_alerts.add_alert("AAPL", "above", 180.0)
        price_alerts.add_alert("GOOGL", "below", 140.0)
        result = price_alerts.remove_alert(0)
        self.assertTrue(result)
        alerts = price_alerts.get_active_alerts()
        self.assertEqual(len(alerts), 1)
        self.assertEqual(alerts[0]["ticker"], "GOOGL")

    def test_remove_invalid_index(self):
        self.assertFalse(price_alerts.remove_alert(5))
        self.assertFalse(price_alerts.remove_alert(-1))

    def test_remove_from_empty(self):
        self.assertFalse(price_alerts.remove_alert(0))


class TestGetActiveAlerts(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._tmp.close()
        self._orig = price_alerts.ALERTS_FILE
        price_alerts.ALERTS_FILE = Path(self._tmp.name)
        with open(self._tmp.name, "w") as f:
            json.dump([], f)

    def tearDown(self):
        price_alerts.ALERTS_FILE = self._orig
        try:
            os.unlink(self._tmp.name)
        except OSError:
            pass

    def test_excludes_triggered(self):
        price_alerts.add_alert("AAPL", "above", 180.0)
        price_alerts.add_alert("MSFT", "below", 300.0)
        # Manually trigger one
        alerts = price_alerts._load_alerts()
        alerts[0]["triggered"] = True
        price_alerts._save_alerts(alerts)

        active = price_alerts.get_active_alerts()
        self.assertEqual(len(active), 1)
        self.assertEqual(active[0]["ticker"], "MSFT")

    def test_empty_file(self):
        self.assertEqual(price_alerts.get_active_alerts(), [])


class TestCheckAlert(unittest.TestCase):
    def test_above_triggered(self):
        alert = {"condition": "above", "price": 150.0, "triggered": False}
        self.assertTrue(price_alerts.check_alert(alert, 155.0))

    def test_above_not_triggered(self):
        alert = {"condition": "above", "price": 150.0, "triggered": False}
        self.assertFalse(price_alerts.check_alert(alert, 145.0))

    def test_below_triggered(self):
        alert = {"condition": "below", "price": 150.0, "triggered": False}
        self.assertTrue(price_alerts.check_alert(alert, 148.0))

    def test_below_not_triggered(self):
        alert = {"condition": "below", "price": 150.0, "triggered": False}
        self.assertFalse(price_alerts.check_alert(alert, 155.0))

    def test_already_triggered(self):
        alert = {"condition": "above", "price": 150.0, "triggered": True}
        self.assertFalse(price_alerts.check_alert(alert, 200.0))

    def test_none_price(self):
        alert = {"condition": "above", "price": 150.0, "triggered": False}
        self.assertFalse(price_alerts.check_alert(alert, None))

    def test_exact_price_triggers(self):
        alert = {"condition": "above", "price": 150.0, "triggered": False}
        self.assertTrue(price_alerts.check_alert(alert, 150.0))


class TestCheckAllAlerts(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self._tmp.close()
        self._orig = price_alerts.ALERTS_FILE
        price_alerts.ALERTS_FILE = Path(self._tmp.name)
        with open(self._tmp.name, "w") as f:
            json.dump([], f)

    def tearDown(self):
        price_alerts.ALERTS_FILE = self._orig
        try:
            os.unlink(self._tmp.name)
        except OSError:
            pass

    @patch("price_alerts.fetch_current_price")
    def test_triggers_matching_alerts(self, mock_price):
        mock_price.return_value = 185.0
        price_alerts.add_alert("AAPL", "above", 180.0)
        price_alerts.add_alert("AAPL", "below", 150.0)

        triggered = price_alerts.check_all_alerts(include_position_alerts=False)
        self.assertEqual(len(triggered), 1)
        self.assertEqual(triggered[0]["condition"], "above")
        self.assertTrue(triggered[0]["triggered"])
        self.assertIsNotNone(triggered[0]["triggered_at"])

    @patch("price_alerts.fetch_current_price")
    def test_no_triggers(self, mock_price):
        mock_price.return_value = 170.0
        price_alerts.add_alert("AAPL", "above", 180.0)
        price_alerts.add_alert("AAPL", "below", 150.0)

        triggered = price_alerts.check_all_alerts(include_position_alerts=False)
        self.assertEqual(len(triggered), 0)

    @patch("price_alerts.fetch_current_price")
    def test_persists_triggered_state(self, mock_price):
        mock_price.return_value = 185.0
        price_alerts.add_alert("AAPL", "above", 180.0)
        price_alerts.check_all_alerts(include_position_alerts=False)

        # Load again — should be marked triggered
        alerts = price_alerts._load_alerts()
        self.assertTrue(alerts[0]["triggered"])

    @patch("price_alerts.fetch_current_price")
    def test_price_fetch_failure(self, mock_price):
        mock_price.return_value = None
        price_alerts.add_alert("AAPL", "above", 180.0)
        triggered = price_alerts.check_all_alerts(include_position_alerts=False)
        self.assertEqual(len(triggered), 0)


class TestGeneratePositionAlerts(unittest.TestCase):
    def test_generates_stop_warnings(self):
        import sys
        mock_pt = MagicMock()
        mock_pt.load_state.return_value = {
            "open_positions": [{
                "ticker": "AAPL",
                "direction": "BUY",
                "entry_price": 170.0,
                "current_stop": 160.0,
            }]
        }
        sys.modules["paper_trader"] = mock_pt
        try:
            alerts = price_alerts.generate_position_alerts()
            stop_warnings = [a for a in alerts if a["type"] == "STOP_WARNING"]
            self.assertGreater(len(stop_warnings), 0)
            self.assertEqual(stop_warnings[0]["condition"], "below")
        finally:
            del sys.modules["paper_trader"]

    def test_handles_missing_paper_trader(self):
        # Should not raise even if paper_trader is unavailable
        alerts = price_alerts.generate_position_alerts()
        self.assertIsInstance(alerts, list)


class TestLoadAlerts(unittest.TestCase):
    def setUp(self):
        self._orig = price_alerts.ALERTS_FILE

    def tearDown(self):
        price_alerts.ALERTS_FILE = self._orig

    def test_missing_file(self):
        price_alerts.ALERTS_FILE = Path("/tmp/nonexistent_alerts_xyz.json")
        self.assertEqual(price_alerts._load_alerts(), [])

    def test_corrupt_json(self):
        tmp = tempfile.NamedTemporaryFile(suffix=".json", delete=False, mode="w")
        tmp.write("{bad json")
        tmp.close()
        price_alerts.ALERTS_FILE = Path(tmp.name)
        self.assertEqual(price_alerts._load_alerts(), [])
        os.unlink(tmp.name)


if __name__ == "__main__":
    unittest.main()
