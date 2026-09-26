"""The v1 HTTP contract matches the handlers on this commit.

The document is not a substitute for test_api_contract.py. These checks
call the shipped functions and read the route table in dashboard_server.py.
"""

from __future__ import annotations

import os
import subprocess
import sys
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

    def test_unauthorized_paths_are_401_before_the_static_404(self):
        text = SPEC.read_text(encoding="utf-8")
        self.assertIn("text/html; charset=utf-8", text)
        self.assertIn("before the static check", text)
        self.assertIn("only after the bearer matches", text)
        old_token = dashboard_server.DASHBOARD_TOKEN
        old_open = dashboard_server.DASHBOARD_ALLOW_UNAUTHENTICATED
        dashboard_server.DASHBOARD_TOKEN = ""
        dashboard_server.DASHBOARD_ALLOW_UNAUTHENTICATED = False
        try:
            unknown = _handler()
            unknown.path = "/nope"
            unknown.do_GET()
            self.assertEqual(unknown._errors, [(401, "Unauthorized")])
            dotted = _handler()
            dotted.path = "/dashboard/../secret"
            dotted.do_GET()
            self.assertEqual(dotted._errors, [(401, "Unauthorized")])
            posted = _handler()
            posted.path = "/api/events"
            posted.headers = {"Authorization": "Bearer x"}
            posted.do_POST()
            self.assertEqual(posted._errors, [(401, "Unauthorized")])
        finally:
            dashboard_server.DASHBOARD_TOKEN = old_token
            dashboard_server.DASHBOARD_ALLOW_UNAUTHENTICATED = old_open

        dashboard_server.DASHBOARD_TOKEN = "secret"
        try:
            mismatch = _handler()
            mismatch.path = "/api/events"
            mismatch.headers = {"Authorization": "Bearer wrong"}
            mismatch.do_POST()
            self.assertEqual(mismatch._errors, [(401, "Unauthorized")])
            matched = _handler()
            matched.path = "/api/events"
            matched.headers = {"Authorization": "Bearer secret"}
            matched.do_POST()
            self.assertEqual(matched._errors, [(405, "Method not allowed")])
        finally:
            dashboard_server.DASHBOARD_TOKEN = old_token
            dashboard_server.DASHBOARD_ALLOW_UNAUTHENTICATED = old_open

    def test_open_mode_is_four_words_and_skips_the_bearer_compare(self):
        text = SPEC.read_text(encoding="utf-8")
        source = (REPO / "dashboard_server.py").read_text(encoding="utf-8")
        self.assertIn('{"1", "true", "yes", "on"}', source)
        self.assertIn("no bearer is compared", text)
        probe = (
            "import dashboard_server; "
            "print(dashboard_server.DASHBOARD_ALLOW_UNAUTHENTICATED)"
        )
        for value, expected in (
            ("0", "False"),
            ("false", "False"),
            ("", "False"),
            ("yes", "True"),
            ("ON", "True"),
            ("1", "True"),
        ):
            env = os.environ.copy()
            if value == "":
                env.pop("CHELATED_DASHBOARD_ALLOW_UNAUTHENTICATED", None)
            else:
                env["CHELATED_DASHBOARD_ALLOW_UNAUTHENTICATED"] = value
            env["CHELATED_DASHBOARD_TOKEN"] = ""
            out = subprocess.check_output(
                [sys.executable, "-c", probe], cwd=REPO, env=env, text=True
            )
            self.assertIn(expected, out.splitlines()[-1])

        old_token = dashboard_server.DASHBOARD_TOKEN
        old_open = dashboard_server.DASHBOARD_ALLOW_UNAUTHENTICATED
        dashboard_server.DASHBOARD_TOKEN = ""
        dashboard_server.DASHBOARD_ALLOW_UNAUTHENTICATED = True
        try:
            for headers in ({}, {"Authorization": "Bearer x"}):
                posted = _handler()
                posted.path = "/api/events"
                posted.headers = headers
                posted.do_POST()
                self.assertEqual(posted._errors, [(405, "Method not allowed")])
        finally:
            dashboard_server.DASHBOARD_TOKEN = old_token
            dashboard_server.DASHBOARD_ALLOW_UNAUTHENTICATED = old_open

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
