"""The v1 HTTP contract matches the handlers on this commit.

The document is not a substitute for test_api_contract.py. These checks
call the shipped functions and read the route table in dashboard_server.py.
"""

from __future__ import annotations

import subprocess
import unittest
from io import BytesIO
from pathlib import Path

import dashboard_server

REPO = Path(__file__).resolve().parent.parent
SPEC = REPO / "docs" / "dashboard-http-contract-v1.md"

INTEGER_LIMIT_ROUTES = (
    "/api/events",
    "/api/campaign_history",
    "/api/validation_history",
    "/api/preflight_history",
    "/api/evidence_chain_history",
    "/api/model_scope/events",
    "/api/model_scope/features",
    "/api/model_scope/interventions",
    "/api/evidence_cleanup_plan",
)


def _handler():
    handler = dashboard_server.DashboardHandler.__new__(dashboard_server.DashboardHandler)
    handler.wfile = BytesIO()
    handler.headers = {}
    handler.send_response = lambda *args, **kwargs: None
    handler.send_header = lambda *args, **kwargs: None
    handler.end_headers = lambda *args, **kwargs: None
    handler._errors = []

    def send_error_response(status_code, message):
        handler._errors.append((status_code, message))

    handler.send_error_response = send_error_response
    return handler


class TestDashboardHttpContractV1(unittest.TestCase):
    def test_spec_names_this_commit_and_not_the_old_tip(self):
        text = SPEC.read_text(encoding="utf-8")
        marker = "handler_sha: "
        line = next(item for item in text.splitlines() if item.startswith(marker))
        handler_sha = line[len(marker):].strip()
        subprocess.check_call(
            ["git", "merge-base", "--is-ancestor", handler_sha, "HEAD"],
            cwd=REPO,
        )
        handler_diff = subprocess.check_output(
            ["git", "diff", handler_sha, "HEAD", "--", "dashboard_server.py"],
            cwd=REPO,
            text=True,
        )
        self.assertEqual(handler_diff, "")
        self.assertIn(f"handler_sha: {handler_sha}", text)
        self.assertIn("applies_to_origin_main_2bac5d3: no", text)
        self.assertIn("schema: dashboard-http-v1", text)
        source = (REPO / "dashboard_server.py").read_text(encoding="utf-8")
        for path in INTEGER_LIMIT_ROUTES:
            self.assertIn(f'"{path}"', source)
            self.assertIn(path, text)

    def test_non_integer_limit_helper_and_the_nine_routes(self):
        self.assertTrue(dashboard_server._non_integer_limit({"limit": ["abc"]}))
        self.assertFalse(dashboard_server._non_integer_limit({"limit": ["3"]}))
        self.assertEqual(dashboard_server._limit_or_default({"limit": ["abc"]}, 20), 20)
        handler = _handler()
        calls = (
            handler.handle_api_events,
            handler.handle_api_campaign_history,
            handler.handle_api_validation_history,
            handler.handle_api_preflight_history,
            handler.handle_api_evidence_chain_history,
            handler.handle_api_model_scope_events,
            handler.handle_api_model_scope_features,
            handler.handle_api_model_scope_interventions,
            handler.handle_api_evidence_cleanup_plan,
        )
        for method in calls:
            handler._errors.clear()
            method({"limit": ["abc"]})
            self.assertEqual(handler._errors, [(400, "limit must be an integer")], method.__name__)

    def test_head_control_page_is_empty_200_and_api_is_401(self):
        old = dashboard_server.DASHBOARD_TOKEN
        dashboard_server.DASHBOARD_TOKEN = "secret"
        try:
            page = _handler()
            page.path = "/dashboard/"
            page.do_HEAD()
            self.assertEqual(page._errors, [])
            self.assertEqual(page.wfile.getvalue(), b"")
            api = _handler()
            api.path = "/api/summary?token=secret"
            api.do_HEAD()
            self.assertEqual(api._errors, [(401, "Unauthorized")])
        finally:
            dashboard_server.DASHBOARD_TOKEN = old


if __name__ == "__main__":
    unittest.main()
