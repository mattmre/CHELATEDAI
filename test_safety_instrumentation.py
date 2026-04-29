"""Stage 0 safety-testbed instrumentation preflight tests.

These tests verify the gauges before the larger project-car safety testbed
starts changing knobs. They intentionally focus on diagnostics trustworthiness:
normal, empty, error, reformulated, routed, dashboard, and JSON-safety paths.
"""

import json
import unittest

import httpx
import numpy as np

from adapter_router import AdapterRouter
from dashboard_server import summarize_events
from integrated_diagnostics_report import IntegratedDiagnosticsReport
from qdrant_client.http.exceptions import UnexpectedResponse
from query_reformulator import QueryReformulator
from run_live_fire_diagnostics import EventCollector, FakeQdrant, _dataset, _make_engine
from stability_tracker import StabilityTracker


def _ingested_engine():
    logger = EventCollector()
    engine = _make_engine(logger)
    corpus, payloads, _queries, _qrels = _dataset()
    engine.ingest(corpus, payloads)
    return engine, logger


def _assert_json_safe(testcase, payload):
    encoded = json.dumps(payload)
    forbidden = ("tensor(", "Parameter containing", "Identity()", "array(")
    for marker in forbidden:
        testcase.assertNotIn(marker, encoded)


class TestStage0InstrumentationPreflight(unittest.TestCase):
    def test_normal_path_runtime_diagnostics_are_json_safe_and_sanitized(self):
        engine, logger = _ingested_engine()

        result = engine.run_inference("adaptive chelation retrieval")
        diagnostics = engine.get_last_runtime_diagnostics()
        telemetry = engine.get_runtime_telemetry()

        self.assertEqual(len(result), 4)
        self.assertEqual(diagnostics["runtime"]["status"], "ok")
        self.assertEqual(diagnostics["runtime"]["action"], "CHELATE")
        self.assertEqual(diagnostics["retrieval_policy"]["policy"], "local_chelation")
        self.assertIn("query_hash", diagnostics["query_summary"])
        self.assertNotIn("adaptive chelation retrieval", json.dumps(diagnostics))
        self.assertGreaterEqual(telemetry["total_inferences"], 1)
        self.assertTrue(any(event["event_type"] == "runtime_diagnostics" for event in logger.events))
        _assert_json_safe(self, diagnostics)

    def test_empty_result_path_reports_empty_status_and_counter(self):
        engine, logger = _ingested_engine()
        engine.qdrant = FakeQdrant()
        engine._vector_store = engine.qdrant

        std_top, final_top, mask, jaccard = engine.run_inference("adaptive chelation retrieval")
        diagnostics = engine.get_last_runtime_diagnostics()
        telemetry = engine.get_runtime_telemetry()
        summary = summarize_events(logger.events)

        self.assertEqual(std_top, [])
        self.assertEqual(final_top, [])
        self.assertEqual(mask.shape, (engine.vector_size,))
        self.assertEqual(jaccard, 0.0)
        self.assertEqual(diagnostics["runtime"]["status"], "empty_results")
        self.assertEqual(telemetry["empty_result_count"], 1)
        self.assertGreaterEqual(summary["runtime_diagnostics_count"], 1)
        _assert_json_safe(self, diagnostics)

    def test_qdrant_error_path_reports_error_status_and_counter(self):
        class ErrorQdrant(FakeQdrant):
            def query_points(self, *args, **kwargs):
                raise UnexpectedResponse(
                    status_code=500,
                    reason_phrase="boom",
                    content=b"query failed",
                    headers=httpx.Headers({}),
                )

        engine, logger = _ingested_engine()
        engine.qdrant = ErrorQdrant()
        engine._vector_store = engine.qdrant

        std_top, final_top, mask, jaccard = engine.run_inference("adaptive chelation retrieval")
        diagnostics = engine.get_last_runtime_diagnostics()
        telemetry = engine.get_runtime_telemetry()

        self.assertEqual(std_top, [])
        self.assertEqual(final_top, [])
        self.assertEqual(mask.shape, (engine.vector_size,))
        self.assertEqual(jaccard, 0.0)
        self.assertEqual(diagnostics["runtime"]["status"], "qdrant_error")
        self.assertEqual(diagnostics["runtime"]["error_type"], "UnexpectedResponse")
        self.assertEqual(telemetry["qdrant_error_count"], 1)
        self.assertTrue(any(event["level"] == "ERROR" for event in logger.events))
        _assert_json_safe(self, diagnostics)

    def test_reformulated_path_preserves_parent_and_variant_metadata(self):
        engine, _logger = _ingested_engine()
        engine._query_reformulator = QueryReformulator(logger=engine.logger)
        engine._query_reformulator_max_variants = 3

        _std_top, final_top, _mask, _jaccard = engine.run_inference("the adaptive chelation retrieval")
        diagnostics = engine.get_last_runtime_diagnostics()
        reformulation = diagnostics["query_reformulation"]

        self.assertGreater(len(final_top), 0)
        self.assertEqual(diagnostics["runtime"]["action"], "REFORMULATE")
        self.assertEqual(diagnostics["retrieval_policy"]["policy"], "multi_variant_merge")
        self.assertGreaterEqual(reformulation["variant_count"], 2)
        self.assertEqual(reformulation["variants"][0]["strategy"], "original")
        self.assertIn("stopword_removed", {variant["strategy"] for variant in reformulation["variants"]})
        _assert_json_safe(self, diagnostics)

    def test_routed_path_records_route_outcomes_and_effectiveness(self):
        engine, _logger = _ingested_engine()
        engine._stability_tracker = StabilityTracker()
        engine._stability_tracker.logger = engine.logger
        router = AdapterRouter(logger=engine.logger)
        router.register("adaptive", [1.0, 0.3, 0.05, 0.05], engine.adapter)
        router.register("storage", [0.05, 1.0, 0.25, 0.05], engine.adapter)
        engine._adapter_router = router

        engine.run_inference("adaptive chelation retrieval")
        diagnostics = engine.get_last_runtime_diagnostics()
        effectiveness = router.get_route_effectiveness()

        self.assertEqual(diagnostics["route"]["key"], "adaptive")
        self.assertEqual(diagnostics["route_outcome"]["route_key"], "adaptive")
        self.assertEqual(effectiveness["routes"]["adaptive"]["count"], 1)
        self.assertIn("norm_drift", diagnostics)
        self.assertEqual(diagnostics["norm_drift"]["adapter_norm_ratio_latest"], 1.0)
        _assert_json_safe(self, diagnostics)

    def test_dashboard_telemetry_summary_aggregates_runtime_routes_actions_and_latency(self):
        events = [
            {
                "timestamp": "2026-04-27T00:00:00Z",
                "event_type": "runtime_diagnostics",
                "runtime": {"latency_ms": 10.0},
                "route": {"key": "adaptive"},
            },
            {
                "timestamp": "2026-04-27T00:00:01Z",
                "event_type": "runtime_diagnostics",
                "runtime": {"latency_ms": 20.0},
                "route": {"key": "storage"},
            },
            {
                "timestamp": "2026-04-27T00:00:02Z",
                "event_type": "adaptive_gate_evaluated",
                "actions": ["prefer_global_scout", "normalize_runtime_vectors"],
            },
            {
                "timestamp": "2026-04-27T00:00:03Z",
                "adaptive_gate": {"actions": ["prefer_global_scout"]},
            },
        ]

        summary = summarize_events(events)

        self.assertEqual(summary["runtime_diagnostics_count"], 2)
        self.assertEqual(summary["adapter_route_breakdown"], {"adaptive": 1, "storage": 1})
        self.assertEqual(summary["adaptive_gate_actions"]["prefer_global_scout"], 2)
        self.assertEqual(summary["adaptive_gate_actions"]["normalize_runtime_vectors"], 1)
        self.assertEqual(summary["latency_ms"]["mean"], 15.0)
        self.assertEqual(summary["latency_ms"]["p50"], 10.0)
        self.assertEqual(summary["latency_ms"]["p95"], 20.0)

    def test_integrated_report_serializes_numpy_and_unexpected_objects_without_raw_runtime_objects(self):
        class UnexpectedObject:
            def __str__(self):
                return "unexpected-object"

        report = IntegratedDiagnosticsReport(
            phase="stage0",
            retrieval_metrics={"ndcg": np.float32(0.5), "values": np.array([1, 2, 3])},
            metadata={"tuple": (np.float64(1.5),), "set": {"a", "b"}, "object": UnexpectedObject()},
            runtime={"latency_ms": np.float64(4.2)},
            telemetry={"array": np.array([0.1, 0.2])},
        ).to_dict()

        encoded = json.dumps(report)

        self.assertIn("unexpected-object", encoded)
        self.assertEqual(report["retrieval_metrics"]["values"], [1, 2, 3])
        self.assertEqual(report["telemetry"]["array"], [0.1, 0.2])
        _assert_json_safe(self, report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
