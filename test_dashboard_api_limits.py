"""Limits, preflight, and corrupt-history contracts."""

import os
import tempfile
import unittest
from io import BytesIO
from unittest.mock import MagicMock

import dashboard_server


def _handler():
    handler = dashboard_server.DashboardHandler.__new__(dashboard_server.DashboardHandler)
    handler.wfile = BytesIO()
    handler.headers = {}
    handler.send_response = MagicMock()
    handler.send_header = MagicMock()
    handler.end_headers = MagicMock()
    handler.send_error_response = MagicMock()
    return handler


class TestApiLimits(unittest.TestCase):
    def test_events_non_integer_limit_is_400(self):
        handler = _handler()
        handler.handle_api_events({"limit": ["abc"]})
        handler.send_error_response.assert_called_once_with(400, "limit must be an integer")

    def test_options_preflight_does_not_require_a_bearer(self):
        dashboard_server.DASHBOARD_TOKEN = "secret"
        dashboard_server.DASHBOARD_CORS_ORIGIN = "https://example.test"
        handler = _handler()
        handler.headers = {}
        handler.do_OPTIONS()
        handler.send_response.assert_called_with(204)
        handler.send_error_response.assert_not_called()

    def test_zero_model_scope_limit_is_empty(self):
        self.assertEqual(dashboard_server._nonnegative_limit("0"), 0)
        self.assertEqual(dashboard_server._nonnegative_limit("-3"), 0)
        self.assertEqual(dashboard_server._nonnegative_limit("9000"), dashboard_server._MAX_API_LIMIT)
        self.assertEqual(dashboard_server._tail_paths([1, 2, 3], 0), [])

    def test_bad_campaign_file_raises(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "campaign_report.json")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("{")
            with self.assertRaises(ValueError):
                dashboard_server.load_campaign_history(tmpdir, limit=10)

    def test_missing_campaign_root_is_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = os.path.join(tmpdir, "absent")
            result = dashboard_server.load_campaign_history(missing, limit=10)
        self.assertEqual(result["summary"]["total_reports"], 0)


if __name__ == "__main__":
    unittest.main()
