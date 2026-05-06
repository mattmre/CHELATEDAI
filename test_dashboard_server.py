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
            {"timestamp": 1234567892.0, "event_type": "info", "error": None},
        ]
        
        summary = dashboard_server.summarize_events(events)
        
        self.assertEqual(summary["total_events"], 3)
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

    def test_summarize_adaptive_runtime_events(self):
        """Test summarizing adaptive diagnostics and route telemetry."""
        events = [
            {
                "timestamp": 1.0,
                "event_type": "runtime_diagnostics",
                "runtime": {"latency_ms": 10.0},
                "route": {"key": "route-a"},
            },
            {
                "timestamp": 2.0,
                "event_type": "adaptive_gate_evaluated",
                "actions": ["prefer_global_scout", "normalize_runtime_vectors"],
            },
            {
                "timestamp": 3.0,
                "event_type": "adapter_route_selected",
                "route_key": "route-a",
            },
            {
                "timestamp": 4.0,
                "adaptive_gate": {"actions": ["prefer_global_scout"]},
            },
        ]

        summary = dashboard_server.summarize_events(events)

        self.assertEqual(summary["runtime_diagnostics_count"], 1)
        self.assertEqual(summary["adapter_route_breakdown"], {"route-a": 2})
        self.assertEqual(summary["adaptive_gate_actions"]["prefer_global_scout"], 2)
        self.assertEqual(summary["latency_ms"]["mean"], 10.0)


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


class TestCampaignHistory(unittest.TestCase):
    """Test campaign-history discovery helpers."""

    def test_load_campaign_history_empty_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = dashboard_server.load_campaign_history(os.path.join(tmpdir, "missing"))

        self.assertEqual(result["reports"], [])
        self.assertEqual(result["summary"]["total_reports"], 0)

    def test_load_campaign_history_normalizes_model_scope_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = os.path.join(tmpdir, "experiment_runs")
            report_dir = os.path.join(root, "model-scope-overlay-smoke", "latest", "campaign")
            os.makedirs(report_dir)
            report_path = os.path.join(report_dir, "campaign_report.json")
            with open(report_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "record_type": "model_scope_campaign_report",
                        "run_label": "smoke",
                        "task": "SciFact",
                        "promotion_decision": {
                            "decision": "hold",
                            "default_change_allowed": False,
                        },
                        "adaptive_overlay_summary": {
                            "ready_for_broader_validation": True,
                            "next_action": "broaden_validation",
                        },
                        "adaptive_overlay_artifact_card": {
                            "artifact_card_id": "overlay-card-smoke",
                        },
                    },
                    handle,
                )

            result = dashboard_server.load_campaign_history(root)

        self.assertEqual(result["summary"]["total_reports"], 1)
        self.assertEqual(result["summary"]["overlay_ready"], 1)
        self.assertEqual(result["summary"]["promotion_allowed"], 0)
        report = result["reports"][0]
        self.assertEqual(report["run_label"], "smoke")
        self.assertEqual(report["decision"], "hold")
        self.assertEqual(report["artifact_card_id"], "overlay-card-smoke")


class TestValidationHistory(unittest.TestCase):
    """Test validation-history discovery helpers."""

    def test_load_validation_history_empty_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = dashboard_server.load_validation_history(os.path.join(tmpdir, "missing"))

        self.assertEqual(result["reports"], [])
        self.assertEqual(result["summary"]["latest_passed"], None)

    def test_load_validation_history_normalizes_bundle_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = os.path.join(tmpdir, "experiment_runs")
            report_dir = os.path.join(root, "overlay-model-scope-validation", "latest")
            os.makedirs(report_dir)
            report_path = os.path.join(report_dir, "validation_summary.json")
            with open(report_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "record_type": "overlay_model_scope_validation_bundle",
                        "output_dir": report_dir,
                        "passed": False,
                        "command_count": 2,
                        "failed_commands": ["model_scope_overlay_smoke"],
                        "results": [],
                    },
                    handle,
                )

            result = dashboard_server.load_validation_history(root)

        self.assertEqual(result["summary"]["total_reports"], 1)
        self.assertEqual(result["summary"]["passed"], 0)
        self.assertEqual(result["summary"]["failed"], 1)
        self.assertFalse(result["summary"]["latest_passed"])
        report = result["reports"][0]
        self.assertEqual(report["record_type"], "overlay_model_scope_validation_bundle")
        self.assertEqual(report["failed_commands"], ["model_scope_overlay_smoke"])


class TestPreflightHistory(unittest.TestCase):
    """Test default-promotion preflight discovery helpers."""

    def test_load_preflight_history_empty_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = dashboard_server.load_preflight_history(os.path.join(tmpdir, "missing"))

        self.assertEqual(result["reports"], [])
        self.assertEqual(result["summary"]["latest_review_allowed"], None)

    def test_load_preflight_history_normalizes_blocked_preflight(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = os.path.join(tmpdir, "experiment_runs")
            report_dir = os.path.join(root, "default-promotion-preflight", "latest")
            os.makedirs(report_dir)
            report_path = os.path.join(report_dir, "preflight.json")
            with open(report_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "record_type": "default_promotion_preflight",
                        "review_allowed": False,
                        "default_change_allowed": False,
                        "blockers": ["repeat_seed_evidence_does_not_support_default_promotion"],
                        "artifacts": {
                            "validation_summary": {"passed": True},
                            "promotion_linkage_audit": {"passed": True},
                            "repeat_seed_decision": {"passed": False},
                        },
                    },
                    handle,
                )

            result = dashboard_server.load_preflight_history(root)

        self.assertEqual(result["summary"]["total_reports"], 1)
        self.assertEqual(result["summary"]["review_allowed"], 0)
        self.assertEqual(result["summary"]["blocked"], 1)
        self.assertFalse(result["summary"]["latest_review_allowed"])
        self.assertEqual(
            result["summary"]["latest_blockers"],
            ["repeat_seed_evidence_does_not_support_default_promotion"],
        )
        report = result["reports"][0]
        self.assertEqual(report["record_type"], "default_promotion_preflight")
        self.assertEqual(report["artifact_count"], 3)

    def test_load_preflight_history_accepts_default_promotion_filename(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = os.path.join(tmpdir, "experiment_runs")
            report_dir = os.path.join(root, "default-promotion-evidence-chain", "latest")
            os.makedirs(report_dir)
            report_path = os.path.join(report_dir, "default-promotion-preflight.json")
            with open(report_path, "w", encoding="utf-8") as handle:
                json.dump({"review_allowed": False, "blockers": ["blocked"]}, handle)

            result = dashboard_server.load_preflight_history(root)

        self.assertEqual(result["summary"]["total_reports"], 1)
        self.assertEqual(
            result["reports"][0]["path"],
            "experiment_runs/default-promotion-evidence-chain/latest/default-promotion-preflight.json",
        )


class TestEvidenceIndex(unittest.TestCase):
    """Test evidence-index dashboard helper."""

    def test_load_evidence_index_missing_file(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = dashboard_server.load_evidence_index(os.path.join(tmpdir, "missing.json"))

        self.assertFalse(result["present"])
        self.assertEqual(result["summary"]["artifact_counts"], {})

    def test_load_evidence_index_normalizes_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "evidence_index.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "record_type": "cross_artifact_evidence_index",
                        "root": "experiment_runs",
                        "summary": {
                            "artifact_counts": {"validation_summaries": 2, "campaign_reports": 1},
                            "latest_review_allowed": False,
                            "latest_preflight_blockers": ["repeat_seed_evidence_does_not_support_default_promotion"],
                            "latest_chain_passed": True,
                        },
                        "artifacts": {
                            "validation_summaries": [{"path": "validation_summary.json"}],
                        },
                    },
                    handle,
                )

            result = dashboard_server.load_evidence_index(path)

        self.assertTrue(result["present"])
        self.assertEqual(result["record_type"], "cross_artifact_evidence_index")
        self.assertEqual(result["summary"]["artifact_counts"]["validation_summaries"], 2)
        self.assertFalse(result["summary"]["latest_review_allowed"])
        self.assertTrue(result["summary"]["latest_chain_passed"])

    def test_load_evidence_index_returns_empty_for_malformed_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "evidence_index.json")
            with open(path, "w", encoding="utf-8") as handle:
                handle.write("[not-an-object]")

            result = dashboard_server.load_evidence_index(path)

        self.assertTrue(result["present"])
        self.assertEqual(result["summary"]["artifact_counts"], {})


class TestEvidenceChainHistory(unittest.TestCase):
    """Test evidence-chain history discovery helpers."""

    def test_load_evidence_chain_history_empty_root(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            result = dashboard_server.load_evidence_chain_history(os.path.join(tmpdir, "missing"))

        self.assertEqual(result["reports"], [])
        self.assertIsNone(result["summary"]["latest_chain_passed"])

    def test_load_evidence_chain_history_normalizes_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = os.path.join(tmpdir, "experiment_runs")
            report_dir = os.path.join(root, "default-promotion-evidence-chain", "latest")
            os.makedirs(report_dir)
            report_path = os.path.join(report_dir, "evidence_chain_summary.json")
            with open(report_path, "w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "record_type": "default_promotion_evidence_chain",
                        "chain_passed": True,
                        "review_allowed": False,
                        "preflight_blockers": ["repeat_seed_evidence_does_not_support_default_promotion"],
                        "artifacts": {"validation_summary": "validation_summary.json"},
                    },
                    handle,
                )

            result = dashboard_server.load_evidence_chain_history(root)

        self.assertEqual(result["summary"]["total_reports"], 1)
        self.assertEqual(result["summary"]["loaded_reports"], 1)
        self.assertEqual(result["summary"]["passed"], 1)
        self.assertTrue(result["summary"]["latest_chain_passed"])
        self.assertFalse(result["summary"]["latest_review_allowed"])
        self.assertEqual(result["reports"][0]["artifact_count"], 1)

    def test_load_evidence_chain_history_skips_non_object_json(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = os.path.join(tmpdir, "experiment_runs")
            report_dir = os.path.join(root, "default-promotion-evidence-chain", "latest")
            os.makedirs(report_dir)
            report_path = os.path.join(report_dir, "evidence_chain_summary.json")
            with open(report_path, "w", encoding="utf-8") as handle:
                json.dump(["not", "object"], handle)

            result = dashboard_server.load_evidence_chain_history(root)

        self.assertEqual(result["summary"]["total_reports"], 1)
        self.assertEqual(result["summary"]["loaded_reports"], 0)
        self.assertEqual(result["reports"], [])


class TestEvidenceCleanupPlan(unittest.TestCase):
    """Test evidence cleanup dry-run dashboard helper."""

    def test_load_evidence_cleanup_plan_summarizes_candidates(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = os.path.join(tmpdir, "experiment_runs")
            old_dir = os.path.join(root, "evidence-index", "old")
            latest_dir = os.path.join(root, "evidence-index", "latest")
            os.makedirs(old_dir)
            os.makedirs(latest_dir)
            with open(os.path.join(old_dir, "evidence_index.json"), "w", encoding="utf-8") as handle:
                json.dump({"record_type": "old"}, handle)
            with open(os.path.join(latest_dir, "evidence_index.json"), "w", encoding="utf-8") as handle:
                json.dump({"record_type": "latest"}, handle)

            result = dashboard_server.load_evidence_cleanup_plan(root, keep_latest=1, candidate_limit=10)

        self.assertTrue(result["dry_run"])
        self.assertIn("source_artifacts", result)
        self.assertIn("source_status", result)
        self.assertIn("present", result["source_status"]["evidence_index"])
        self.assertIn("present", result["source_status"]["freshness_audit"])
        self.assertEqual(result["summary"]["candidate_count"], 1)
        self.assertEqual(result["summary"]["retained_count"], 1)
        self.assertEqual(result["summary"]["candidate_types"], ["evidence_indexes"])
        self.assertEqual(result["candidates"][0]["path"], "experiment_runs/evidence-index/old/evidence_index.json")

    def test_load_evidence_cleanup_plan_surfaces_source_artifacts(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = os.path.join(tmpdir, "experiment_runs")
            evidence_index = os.path.join(root, "evidence-index", "missing", "evidence_index.json")
            freshness_audit = os.path.join(root, "evidence-index", "missing", "freshness_audit.json")

            result = dashboard_server.load_evidence_cleanup_plan(
                root,
                evidence_index=evidence_index,
                freshness_audit=freshness_audit,
            )

        self.assertEqual(
            result["source_artifacts"]["evidence_index"],
            evidence_index.replace(os.sep, "/"),
        )
        self.assertEqual(
            result["source_artifacts"]["freshness_audit"],
            freshness_audit.replace(os.sep, "/"),
        )
        self.assertFalse(result["source_status"]["evidence_index"]["present"])
        self.assertFalse(result["source_status"]["freshness_audit"]["present"])

    def test_load_evidence_cleanup_plan_ignores_empty_source_overrides(self):
        result = dashboard_server.load_evidence_cleanup_plan(
            "missing-experiment-runs",
            evidence_index="",
            freshness_audit="",
        )

        self.assertEqual(
            result["source_artifacts"]["evidence_index"],
            "experiment_runs/evidence-index/latest/evidence_index.json",
        )
        self.assertEqual(
            result["source_artifacts"]["freshness_audit"],
            "experiment_runs/evidence-index/latest/freshness_audit.json",
        )

    def test_load_evidence_cleanup_plan_limits_candidate_rows(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = os.path.join(tmpdir, "experiment_runs")
            for index in range(3):
                report_dir = os.path.join(root, "validation", str(index))
                os.makedirs(report_dir)
                with open(os.path.join(report_dir, "validation_summary.json"), "w", encoding="utf-8") as handle:
                    json.dump({"passed": True, "index": index}, handle)

            result = dashboard_server.load_evidence_cleanup_plan(root, keep_latest=0, candidate_limit=2)

        self.assertEqual(result["summary"]["candidate_count"], 3)
        self.assertEqual(len(result["candidates"]), 2)


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
