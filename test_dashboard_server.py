"""
Unit tests for dashboard_server.py

Tests the helper functions and API handler behavior using mocks and temporary files.
"""

import json
import os
import tempfile
import unittest
from io import BytesIO
from unittest.mock import MagicMock

# Import the module under test
import dashboard_server

# Normalize auth defaults for deterministic tests
dashboard_server.DASHBOARD_TOKEN = ""
dashboard_server.DASHBOARD_CORS_ORIGIN = ""


class TestLoadEvents(unittest.TestCase):
    """Test the load_events function."""

    def setUp(self):
        """Create a temporary log file for testing."""
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
        self.temp_file_path = self.temp_file.name

    def tearDown(self):
        """Clean up temporary files."""
        self.temp_file.close()
        if os.path.exists(self.temp_file_path):
            os.unlink(self.temp_file_path)

    def test_load_events_valid_file(self):
        """Test loading events from a valid JSONL file."""
        events = [
            {"timestamp": 1234567890.0, "query_snippet": "test query", "action": "FAST"},
            {"timestamp": 1234567891.0, "query_snippet": "another query", "action": "ADAPT"},
        ]
        
        # Write events to temp file
        for event in events:
            self.temp_file.write(json.dumps(event) + '\n')
        self.temp_file.flush()
        
        # Load events
        loaded = dashboard_server.load_events(self.temp_file_path)
        
        self.assertEqual(len(loaded), 2)
        self.assertEqual(loaded[0]["query_snippet"], "test query")
        self.assertEqual(loaded[1]["action"], "ADAPT")

    def test_load_events_empty_file(self):
        """Test loading events from an empty file."""
        self.temp_file.flush()
        loaded = dashboard_server.load_events(self.temp_file_path)
        self.assertEqual(loaded, [])

    def test_load_events_file_not_found(self):
        """Test loading events from a non-existent file."""
        with self.assertRaises(FileNotFoundError):
            dashboard_server.load_events("nonexistent_file.jsonl")

    def test_load_events_invalid_json(self):
        """Test loading events from a file with invalid JSON."""
        self.temp_file.write("not valid json\n")
        self.temp_file.flush()
        
        with self.assertRaises(json.JSONDecodeError):
            dashboard_server.load_events(self.temp_file_path)

    def test_load_events_skip_empty_lines(self):
        """Test that empty lines are skipped."""
        events = [
            {"timestamp": 1234567890.0, "query_snippet": "test"},
        ]
        
        self.temp_file.write(json.dumps(events[0]) + '\n')
        self.temp_file.write('\n')  # Empty line
        self.temp_file.write('\n')  # Another empty line
        self.temp_file.flush()
        
        loaded = dashboard_server.load_events(self.temp_file_path)
        self.assertEqual(len(loaded), 1)


class TestSummarizeEvents(unittest.TestCase):
    """Test the summarize_events function."""

    def test_summarize_empty_events(self):
        """Test summarizing an empty list of events."""
        summary = dashboard_server.summarize_events([])
        
        self.assertEqual(summary["total_events"], 0)
        self.assertEqual(summary["query_count"], 0)
        self.assertEqual(summary["error_count"], 0)
        self.assertEqual(summary["action_breakdown"], {})
        self.assertIsNone(summary["time_range"]["earliest"])
        self.assertIsNone(summary["time_range"]["latest"])

    def test_summarize_query_events(self):
        """Test summarizing query events."""
        events = [
            {"timestamp": 1234567890.0, "query_snippet": "test 1", "action": "FAST"},
            {"timestamp": 1234567891.0, "query_snippet": "test 2", "action": "FAST"},
            {"timestamp": 1234567892.0, "query_snippet": "test 3", "action": "ADAPT"},
        ]
        
        summary = dashboard_server.summarize_events(events)
        
        self.assertEqual(summary["total_events"], 3)
        self.assertEqual(summary["query_count"], 3)
        self.assertEqual(summary["error_count"], 0)
        self.assertEqual(summary["action_breakdown"], {"FAST": 2, "ADAPT": 1})
        self.assertEqual(summary["time_range"]["earliest"], 1234567890.0)
        self.assertEqual(summary["time_range"]["latest"], 1234567892.0)

    def test_summarize_error_events(self):
        """Test summarizing events with errors."""
        events = [
            {"timestamp": 1234567890.0, "event_type": "error", "error": "test error"},
            {"timestamp": 1234567891.0, "query_snippet": "test", "action": "FAST"},
        ]
        
        summary = dashboard_server.summarize_events(events)
        
        self.assertEqual(summary["total_events"], 2)
        self.assertEqual(summary["query_count"], 1)
        self.assertEqual(summary["error_count"], 1)

    def test_summarize_mixed_events(self):
        """Test summarizing a mix of different event types."""
        events = [
            {"timestamp": 1234567890.0, "query_snippet": "query 1", "action": "FAST"},
            {"timestamp": 1234567891.0, "query_snippet": "query 2", "action": "DEEP"},
            {"timestamp": 1234567892.0, "event_type": "error", "error": "error msg"},
            {"timestamp": 1234567893.0, "action": "ADAPT"},
        ]
        
        summary = dashboard_server.summarize_events(events)
        
        self.assertEqual(summary["total_events"], 4)
        self.assertEqual(summary["query_count"], 2)
        self.assertEqual(summary["error_count"], 1)
        self.assertEqual(summary["action_breakdown"], {"FAST": 1, "DEEP": 1, "ADAPT": 1})


class TestFilterEvents(unittest.TestCase):
    """Test the filter_events function."""

    def setUp(self):
        """Create sample events for testing."""
        self.events = [
            {"timestamp": 1234567890.0, "query_snippet": "query 1", "action": "FAST"},
            {"timestamp": 1234567891.0, "query_snippet": "query 2", "action": "ADAPT"},
            {"timestamp": 1234567892.0, "event_type": "error", "error": "error msg"},
            {"timestamp": 1234567893.0, "query_snippet": "query 3", "action": "DEEP"},
            {"timestamp": 1234567894.0, "query_snippet": "query 4", "action": "FAST"},
        ]

    def test_filter_no_filters(self):
        """Test filtering with no filters (should return all, sorted by timestamp desc)."""
        filtered = dashboard_server.filter_events(self.events)
        
        self.assertEqual(len(filtered), 5)
        # Should be sorted by timestamp descending
        self.assertEqual(filtered[0]["timestamp"], 1234567894.0)
        self.assertEqual(filtered[-1]["timestamp"], 1234567890.0)

    def test_filter_by_query_type(self):
        """Test filtering by query event type."""
        filtered = dashboard_server.filter_events(self.events, event_type="query")
        
        self.assertEqual(len(filtered), 4)
        # All should have query_snippet
        for event in filtered:
            self.assertIn("query_snippet", event)

    def test_filter_by_error_type(self):
        """Test filtering by error event type."""
        filtered = dashboard_server.filter_events(self.events, event_type="error")
        
        self.assertEqual(len(filtered), 1)
        self.assertEqual(filtered[0]["event_type"], "error")

    def test_filter_with_limit(self):
        """Test filtering with a limit."""
        filtered = dashboard_server.filter_events(self.events, limit=2)
        
        self.assertEqual(len(filtered), 2)
        # Should return the 2 most recent
        self.assertEqual(filtered[0]["timestamp"], 1234567894.0)
        self.assertEqual(filtered[1]["timestamp"], 1234567893.0)

    def test_filter_query_type_with_limit(self):
        """Test filtering by type with limit."""
        filtered = dashboard_server.filter_events(self.events, event_type="query", limit=2)
        
        self.assertEqual(len(filtered), 2)
        # Should have query_snippet and be most recent
        self.assertIn("query_snippet", filtered[0])
        self.assertIn("query_snippet", filtered[1])

    def test_filter_empty_events(self):
        """Test filtering an empty list."""
        filtered = dashboard_server.filter_events([])
        self.assertEqual(filtered, [])

    def test_filter_limit_zero(self):
        """Test with limit=0 should return empty list."""
        filtered = dashboard_server.filter_events(self.events, limit=0)
        self.assertEqual(filtered, [])


class TestDashboardHandler(unittest.TestCase):
    """Test the DashboardHandler class."""

    def setUp(self):
        """Set up test fixtures."""
        self._old_dashboard_token = dashboard_server.DASHBOARD_TOKEN
        self._old_dashboard_cors = dashboard_server.DASHBOARD_CORS_ORIGIN
        dashboard_server.DASHBOARD_TOKEN = ""
        dashboard_server.DASHBOARD_CORS_ORIGIN = ""

        # Create temporary log file
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
        self.temp_file_path = self.temp_file.name
        
        # Write sample events
        events = [
            {"timestamp": 1234567890.0, "query_snippet": "test query", "action": "FAST"},
            {"timestamp": 1234567891.0, "query_snippet": "another query", "action": "ADAPT"},
        ]
        for event in events:
            self.temp_file.write(json.dumps(event) + '\n')
        self.temp_file.flush()
        self.temp_file.close()
        
        # Set the global log file path
        dashboard_server.LOG_FILE_PATH = self.temp_file_path

    def tearDown(self):
        """Clean up temporary files."""
        if os.path.exists(self.temp_file_path):
            os.unlink(self.temp_file_path)
        dashboard_server.DASHBOARD_TOKEN = self._old_dashboard_token
        dashboard_server.DASHBOARD_CORS_ORIGIN = self._old_dashboard_cors

    def _make_handler(self):
        """Create a handler instance without socketserver initialization."""
        handler = dashboard_server.DashboardHandler.__new__(dashboard_server.DashboardHandler)
        handler.wfile = BytesIO()
        handler.headers = {}
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        return handler

    def test_handle_api_events_no_params(self):
        """Test /api/events endpoint without parameters."""
        handler = self._make_handler()
        
        handler.handle_api_events({})
        
        # Check that response was sent
        handler.send_response.assert_called_once()
        output = handler.wfile.getvalue()
        response_data = json.loads(output.decode('utf-8'))
        
        self.assertIn("events", response_data)
        self.assertEqual(len(response_data["events"]), 2)

    def test_handle_api_events_with_limit(self):
        """Test /api/events endpoint with limit parameter."""
        handler = self._make_handler()
        
        handler.handle_api_events({"limit": ["1"]})
        
        output = handler.wfile.getvalue()
        response_data = json.loads(output.decode('utf-8'))
        
        self.assertEqual(len(response_data["events"]), 1)

    def test_handle_api_events_with_event_type(self):
        """Test /api/events endpoint with event_type parameter."""
        handler = self._make_handler()
        
        handler.handle_api_events({"event_type": ["query"]})
        
        output = handler.wfile.getvalue()
        response_data = json.loads(output.decode('utf-8'))
        
        self.assertEqual(len(response_data["events"]), 2)

    def test_handle_api_summary(self):
        """Test /api/summary endpoint."""
        handler = self._make_handler()
        
        handler.handle_api_summary()
        
        output = handler.wfile.getvalue()
        response_data = json.loads(output.decode('utf-8'))
        
        self.assertIn("total_events", response_data)
        self.assertIn("query_count", response_data)
        self.assertIn("error_count", response_data)
        self.assertIn("action_breakdown", response_data)
        self.assertEqual(response_data["total_events"], 2)

    def test_handle_api_events_file_not_found(self):
        """Test API endpoint when log file is not found."""
        # Set invalid log file path
        dashboard_server.LOG_FILE_PATH = "nonexistent_file.jsonl"

        handler = self._make_handler()
        
        handler.handle_api_events({})
        
        # Should call send_response with 404
        calls = [call[0][0] for call in handler.send_response.call_args_list]
        self.assertIn(404, calls)

    def test_send_json_response(self):
        """Test sending a JSON response."""
        handler = self._make_handler()
        
        test_data = {"key": "value", "number": 42}
        handler.send_json_response(test_data)
        
        handler.send_response.assert_called_once_with(200)
        output = handler.wfile.getvalue()
        response_data = json.loads(output.decode('utf-8'))
        
        self.assertEqual(response_data, test_data)

    def test_send_error_response(self):
        """Test sending an error response."""
        handler = self._make_handler()
        
        handler.send_error_response(404, "Not found")
        
        handler.send_response.assert_called_once_with(404)
        output = handler.wfile.getvalue()
        response_data = json.loads(output.decode('utf-8'))
        
        self.assertIn("error", response_data)
        self.assertEqual(response_data["error"], "Not found")


class TestIntegration(unittest.TestCase):
    """Integration tests for the full workflow."""

    def test_full_workflow(self):
        """Test loading, summarizing, and filtering events."""
        # Create temp file with events
        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.jsonl')
        
        events = [
            {"timestamp": 1234567890.0, "query_snippet": "query 1", "action": "FAST", "global_variance": 0.001},
            {"timestamp": 1234567891.0, "query_snippet": "query 2", "action": "ADAPT", "global_variance": 0.002},
            {"timestamp": 1234567892.0, "event_type": "error", "error": "test error"},
            {"timestamp": 1234567893.0, "query_snippet": "query 3", "action": "DEEP", "global_variance": 0.003},
        ]
        
        for event in events:
            temp_file.write(json.dumps(event) + '\n')
        temp_file.flush()
        temp_file.close()
        
        try:
            # Load events
            loaded = dashboard_server.load_events(temp_file.name)
            self.assertEqual(len(loaded), 4)
            
            # Summarize
            summary = dashboard_server.summarize_events(loaded)
            self.assertEqual(summary["total_events"], 4)
            self.assertEqual(summary["query_count"], 3)
            self.assertEqual(summary["error_count"], 1)
            self.assertEqual(summary["action_breakdown"]["FAST"], 1)
            self.assertEqual(summary["action_breakdown"]["ADAPT"], 1)
            self.assertEqual(summary["action_breakdown"]["DEEP"], 1)
            
            # Filter queries only
            queries = dashboard_server.filter_events(loaded, event_type="query")
            self.assertEqual(len(queries), 3)
            
            # Filter with limit
            limited = dashboard_server.filter_events(loaded, limit=2)
            self.assertEqual(len(limited), 2)
            
            # Filter queries with limit
            limited_queries = dashboard_server.filter_events(loaded, event_type="query", limit=2)
            self.assertEqual(len(limited_queries), 2)
            
        finally:
            os.unlink(temp_file.name)


class TestDashboardSecurity(unittest.TestCase):
    """Security-oriented behavior tests."""

    def setUp(self):
        self._old_dashboard_token = dashboard_server.DASHBOARD_TOKEN
        self._old_dashboard_cors = dashboard_server.DASHBOARD_CORS_ORIGIN
        dashboard_server.DASHBOARD_TOKEN = ""
        dashboard_server.DASHBOARD_CORS_ORIGIN = ""

    def tearDown(self):
        dashboard_server.DASHBOARD_TOKEN = self._old_dashboard_token
        dashboard_server.DASHBOARD_CORS_ORIGIN = self._old_dashboard_cors

    def _make_handler(self):
        handler = dashboard_server.DashboardHandler.__new__(dashboard_server.DashboardHandler)
        handler.wfile = BytesIO()
        handler.headers = {}
        handler.send_response = MagicMock()
        handler.send_header = MagicMock()
        handler.end_headers = MagicMock()
        return handler

    def test_do_get_blocks_api_without_token_header(self):
        """API requests should be blocked when token auth is enabled and header is missing."""
        dashboard_server.DASHBOARD_TOKEN = "secret-token"
        handler = self._make_handler()
        handler.path = "/api/summary"
        handler.handle_api_summary = MagicMock()
        handler.send_error_response = MagicMock()

        handler.do_GET()

        handler.send_error_response.assert_called_once_with(401, "Unauthorized")
        handler.handle_api_summary.assert_not_called()

    def test_do_get_allows_api_with_valid_token_header(self):
        """API requests should pass through when a valid bearer token is provided."""
        dashboard_server.DASHBOARD_TOKEN = "secret-token"
        handler = self._make_handler()
        handler.headers = {"Authorization": "Bearer secret-token"}
        handler.path = "/api/summary"
        handler.handle_api_summary = MagicMock()
        handler.send_error_response = MagicMock()

        handler.do_GET()

        handler.handle_api_summary.assert_called_once()
        handler.send_error_response.assert_not_called()

    def test_run_server_rejects_non_local_bind_without_token(self):
        """Server should refuse non-local host bind unless token auth is configured."""
        dashboard_server.DASHBOARD_TOKEN = ""
        with self.assertRaises(ValueError):
            dashboard_server.run_server(host="0.0.0.0", port=8080, log_file="chelation_events.jsonl")


if __name__ == "__main__":
    unittest.main()
