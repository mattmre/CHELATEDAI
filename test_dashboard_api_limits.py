"""Limits, preflight, and corrupt-history contracts."""

import json
import os
import sys
import tempfile
import types
import unittest
from io import BytesIO
from unittest.mock import MagicMock, patch

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

    def test_model_scope_non_integer_limit_keeps_default_json(self):
        handler = _handler()
        params = {"limit": ["abc"]}
        methods = (
            handler.handle_api_model_scope_events,
            handler.handle_api_model_scope_features,
            handler.handle_api_model_scope_interventions,
        )
        # The handler imports ArtifactStore locally. Stub that module so this
        # limit path does not load torch.
        artifacts = types.ModuleType("model_scope_artifacts")

        class ArtifactStore:
            def __init__(self, base_dir=None):
                self.base_dir = base_dir

            def list_artifacts(self, pattern="feature_event_*.json"):
                return []

        artifacts.ArtifactStore = ArtifactStore
        artifacts.load_model_scope_artifact = lambda path: {}
        artifacts.summarize_model_scope_artifact = lambda artifact: {}
        with patch.dict(sys.modules, {"model_scope_artifacts": artifacts}):
            with patch.object(ArtifactStore, "list_artifacts", return_value=[]) as listed:
                for method in methods:
                    method(params)
        self.assertEqual(listed.call_count, 3)
        handler.send_error_response.assert_not_called()
        self.assertEqual(handler.send_response.call_count, 3)
        handler.send_response.assert_called_with(200)
        body = handler.wfile.getvalue().decode("utf-8")
        self.assertNotIn("invalid literal", body)
        self.assertNotIn("ValueError", body)
        self.assertNotIn("Error reading", body)
        self.assertEqual(body.count('"status": "not_generated"'), 3)

    def test_cleanup_non_integer_limit_keeps_default_json(self):
        handler = _handler()
        with patch(
            "dashboard_server.load_evidence_cleanup_plan",
            return_value={"candidates": [], "dry_run": True},
        ) as load_plan:
            handler.handle_api_evidence_cleanup_plan({"limit": ["abc"]})
        handler.send_error_response.assert_not_called()
        load_plan.assert_called_once_with(
            dashboard_server.EVIDENCE_CLEANUP_ROOT,
            keep_latest=1,
            candidate_limit=25,
        )
        handler.send_response.assert_called_with(200)
        body = handler.wfile.getvalue().decode("utf-8")
        payload = json.loads(body)
        self.assertEqual(payload["candidates"], [])
        self.assertNotIn("invalid literal", body)
        self.assertNotIn("Error reading", body)

    def test_cleanup_non_integer_keep_latest_keeps_default(self):
        handler = _handler()
        with patch(
            "dashboard_server.load_evidence_cleanup_plan",
            return_value={"candidates": [], "dry_run": True},
        ) as load_plan:
            handler.handle_api_evidence_cleanup_plan({"keep_latest": ["abc"]})
        handler.send_error_response.assert_not_called()
        load_plan.assert_called_once_with(
            dashboard_server.EVIDENCE_CLEANUP_ROOT,
            keep_latest=1,
            candidate_limit=25,
        )

    def test_options_preflight_does_not_require_a_bearer(self):
        old_token = dashboard_server.DASHBOARD_TOKEN
        old_origin = dashboard_server.DASHBOARD_CORS_ORIGIN
        dashboard_server.DASHBOARD_TOKEN = "secret"
        dashboard_server.DASHBOARD_CORS_ORIGIN = "https://example.test"
        try:
            handler = _handler()
            handler.headers = {}
            handler.do_OPTIONS()
            handler.send_response.assert_called_with(204)
            handler.send_error_response.assert_not_called()
        finally:
            dashboard_server.DASHBOARD_TOKEN = old_token
            dashboard_server.DASHBOARD_CORS_ORIGIN = old_origin

    def test_zero_model_scope_limit_is_empty(self):
        self.assertEqual(dashboard_server._nonnegative_limit("0"), 0)
        self.assertEqual(dashboard_server._nonnegative_limit("-3"), 0)
        self.assertEqual(dashboard_server._nonnegative_limit("9000"), dashboard_server._MAX_API_LIMIT)
        self.assertEqual(dashboard_server._tail_paths([1, 2, 3], 0), [])
        self.assertEqual(dashboard_server._limit_or_default({"limit": ["abc"]}, 20), 20)
        self.assertEqual(dashboard_server._limit_or_default({"limit": ["abc"]}, 25), 25)
        self.assertEqual(dashboard_server._limit_or_default({"limit": ["0"]}, 20), 0)
        self.assertEqual(dashboard_server._limit_or_default({"limit": ["-3"]}, 25), 0)
        self.assertEqual(
            dashboard_server._limit_or_default({"limit": ["9000"]}, 25),
            dashboard_server._MAX_API_LIMIT,
        )

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
