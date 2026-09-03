"""Contract tests for the dashboard FE<->BE boundary (CONTRACT-01, slim scope).

Locks the API surface the frontend depends on: HTTP method inventory,
upper-bound limit clamp, and the JSON error envelope. Runtime header behavior
(ACAO scoping, hardening headers) is covered by live probes, not here.
"""

import unittest
from io import BytesIO

import dashboard_server


class TestMethodInventory(unittest.TestCase):
    """Every method the FE or scanners may send has an explicit handler."""

    def test_read_and_write_methods_covered(self):
        for method in ("do_GET", "do_HEAD", "do_OPTIONS", "do_POST",
                       "do_PUT", "do_DELETE", "do_PATCH"):
            self.assertTrue(
                callable(getattr(dashboard_server.DashboardHandler, method, None)),
                f"DashboardHandler.{method} must exist (P2-01)",
            )

    def test_deny_helper_exists(self):
        self.assertTrue(callable(dashboard_server.DashboardHandler._deny_unsupported_method))

    def test_hardening_headers_override_present(self):
        self.assertIn("end_headers", dashboard_server.DashboardHandler.__dict__)


class TestLimitClamp(unittest.TestCase):
    """?limit= is bounded above (LIMIT-01); 0/negative still mean no rows."""

    def _events(self, n):
        return [{"event_type": "query", "timestamp": i} for i in range(n)]

    def test_huge_limit_clamped(self):
        self.assertEqual(
            len(dashboard_server.filter_events(self._events(6000), limit=999999999)),
            dashboard_server._MAX_API_LIMIT,
        )

    def test_negative_limit_empty(self):
        self.assertEqual(
            dashboard_server.filter_events(self._events(10), limit=-5), []
        )

    def test_zero_limit_empty(self):
        self.assertEqual(
            dashboard_server.filter_events(self._events(10), limit=0), []
        )

    def test_normal_limit_passthrough(self):
        self.assertEqual(
            len(dashboard_server.filter_events(self._events(100), limit=10)), 10
        )

    def test_max_constant_sane(self):
        self.assertGreaterEqual(dashboard_server._MAX_API_LIMIT, 1000)


class TestJsonErrorEnvelope(unittest.TestCase):
    """Stdlib-raised errors (incl. unknown verbs) use the JSON contract (F2)."""

    def _handler(self):
        handler = dashboard_server.DashboardHandler.__new__(
            dashboard_server.DashboardHandler
        )
        handler.requestline = "TRACE /api/events HTTP/1.1"
        handler.request_version = "HTTP/1.1"
        handler.command = "TRACE"
        handler.client_address = ("127.0.0.1", 1)
        handler.wfile = BytesIO()
        handler.headers = {}
        return handler

    def test_send_error_is_json_without_banner(self):
        import json

        handler = self._handler()
        handler.send_error(501, "Unsupported method ('TRACE')")
        raw = handler.wfile.getvalue().decode("utf-8", "replace")
        body = raw.split("\r\n\r\n", 1)[1]
        payload = json.loads(body)
        self.assertIn("error", payload)
        self.assertNotIn("SimpleHTTP", raw)

    def test_neutral_server_banner(self):
        self.assertEqual(dashboard_server.DashboardHandler.server_version, "Dashboard")
        self.assertEqual(dashboard_server.DashboardHandler.sys_version, "")


if __name__ == "__main__":
    unittest.main()
