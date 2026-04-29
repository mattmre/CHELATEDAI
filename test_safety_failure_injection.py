"""Stage 5 failure-injection ravine tests."""

import json
import unittest

import numpy as np
import torch

from adapter_router import AdapterRouter
from adaptive_gate_orchestrator import AdaptiveGateOrchestrator
from computational_storage_poc.mock_array import ArraySimulation
from device_profiles import DeviceClass, get_profile
from evolution_strategies_optimizer import EvolutionStrategiesConfig, EvolutionaryOnlineUpdater
from integrated_diagnostics_report import IntegratedDiagnosticsReport
from quantization_promotion_gate import QuantizationPromotionGate
from run_live_fire_diagnostics import EventCollector, _make_engine
from run_safety_testbed import build_closed_course_engine


class _BadEmbeddingBackend:
    vector_size = 4

    def __init__(self, output):
        self.output = output

    def embed_raw(self, texts):
        del texts
        return np.asarray(self.output, dtype=np.float32)


class _CountingOnlineUpdater:
    def __init__(self):
        self.calls = 0

    def update(self, query_vec, positive_vecs, negative_vecs):
        del query_vec, positive_vecs, negative_vecs
        self.calls += 1


class TestStage5FailureInjection(unittest.TestCase):
    def test_bad_inputs_return_safe_api_shape_and_sanitized_diagnostics(self):
        engine, _logger, _fixture = build_closed_course_engine()

        for bad_query in ["", 12345]:
            std, final, mask, jaccard = engine.run_inference(bad_query)
            diagnostics = engine.get_last_runtime_diagnostics()
            payload = json.dumps(diagnostics)

            self.assertIsInstance(std, list)
            self.assertIsInstance(final, list)
            self.assertEqual(mask.shape, (engine.vector_size,))
            self.assertGreaterEqual(jaccard, 0.0)
            if str(bad_query):
                self.assertNotIn(str(bad_query), payload)
            self.assertIn("query_hash", diagnostics["query_summary"])

    def test_embedding_failures_report_embedding_error_not_uncaught_exception(self):
        logger = EventCollector()
        engine = _make_engine(logger)
        engine.embedding_backend = _BadEmbeddingBackend([[np.nan, 0.0, 0.0, 0.0]])

        result = engine.run_inference("adaptive retrieval")
        diagnostics = engine.get_last_runtime_diagnostics()
        telemetry = engine.get_runtime_telemetry()

        self.assertEqual(result[0], [])
        self.assertEqual(result[1], [])
        self.assertEqual(diagnostics["runtime"]["status"], "embedding_error")
        self.assertEqual(diagnostics["runtime"]["action"], "ERROR")
        self.assertEqual(telemetry["embedding_error_count"], 1)
        self.assertTrue(any(event["event_type"] == "embed_validation" for event in logger.events))

    def test_vector_store_failures_are_explicit_for_missing_payload_text(self):
        engine, logger, _fixture = build_closed_course_engine()
        engine.qdrant.points[0].payload.pop("text")

        with self.assertRaises(ValueError):
            engine.refresh_corpus_vectors(batch_size=2)

        self.assertTrue(any(event["event_type"] == "corpus_vector_refresh_failed" for event in logger.events))

    def test_routing_failures_fallback_and_bad_query_vectors_are_explicit(self):
        router = AdapterRouter(logger=EventCollector())

        fallback = router.select([1.0, 0.0, 0.0, 0.0], fallback=lambda: "default")
        outcome = router.record_outcome(fallback.key, jaccard=0.0, latency_ms=1.0)
        decision = AdaptiveGateOrchestrator(logger=EventCollector()).evaluate({
            "route_effectiveness": router.get_route_effectiveness()
        })

        self.assertEqual(fallback.key, "fallback")
        self.assertEqual(outcome["route_key"], "fallback")
        self.assertIn("disable_low_effectiveness_route", decision.actions)
        with self.assertRaises(ValueError):
            router.select([0.0, 0.0, 0.0, 0.0])

    def test_quantization_failure_loses_gain_and_fails_closed(self):
        result = QuantizationPromotionGate(0.8, logger=EventCollector()).evaluate(
            fp32_fitness=0.75,
            quantized_fitness=0.50,
            baseline_fitness=0.60,
        )
        decision = AdaptiveGateOrchestrator(require_quantization_gate=True, logger=EventCollector()).evaluate({
            "quantization_gate": result.to_dict()
        })

        self.assertFalse(result.passed)
        self.assertIn("retained_gain_below_threshold", result.reasons)
        self.assertFalse(decision.passed)
        self.assertIn("reject_quantized_candidate", decision.actions)

    def test_structural_health_failure_warns_without_mutating_engine(self):
        decision = AdaptiveGateOrchestrator(min_structural_health=0.7, logger=EventCollector()).evaluate({
            "structural_health": {"score": 0.2}
        })

        self.assertTrue(decision.passed)
        self.assertEqual(decision.status, "warning")
        self.assertIn("reduce_optimization_aggression", decision.actions)
        self.assertIn("enable_query_reformulation", decision.actions)

    def test_storage_latency_breach_emits_penalty_recommendation(self):
        array = ArraySimulation(num_drives=1, device_profile=get_profile(DeviceClass.RP2040))
        storage = array.sharded_population_evaluation([
            {"candidate_id": f"c{i}", "fitness": float(i)}
            for i in range(5)
        ])
        decision = AdaptiveGateOrchestrator(storage_latency_sla_ms=0.05, logger=EventCollector()).evaluate({
            "storage": storage
        })

        self.assertGreater(storage["storage_latency_ms"], 0.05)
        self.assertIn("apply_storage_latency_penalty", decision.actions)

    def test_diagnostics_serialization_handles_unexpected_objects_without_raw_runtime_leak(self):
        class Unexpected:
            def __repr__(self):
                return "<Unexpected object>"

        report = IntegratedDiagnosticsReport(
            cycle=99,
            phase="ravine",
            retrieval_fitness=np.float32(0.1),
            final_fitness=np.float64(0.2),
            runtime={"unexpected": Unexpected(), "tensor": torch.tensor([1.0, 2.0])},
            telemetry={"array": np.array([1, 2, 3])},
        ).to_dict()
        payload = json.dumps(report)

        self.assertIn("Unexpected object", payload)
        self.assertNotIn("tensor(", payload)
        self.assertEqual(report["telemetry"]["array"], [1, 2, 3])

    def test_online_update_failures_skip_when_unready_and_raise_on_bad_feedback(self):
        adapter = torch.nn.Linear(2, 2, bias=False)
        updater = EvolutionaryOnlineUpdater(
            adapter,
            config=EvolutionStrategiesConfig(population_size=4, generations=1, seed=5),
            update_interval=2,
            logger=EventCollector(),
        )

        skipped = updater.update(np.array([1.0, 0.0]), np.array([[1.0, 0.0]]), np.array([[0.0, 1.0]]))
        with self.assertRaises(ValueError):
            updater.update(np.array([1.0, 0.0]), np.array([]), np.array([[0.0, 1.0]]))

        engine, _logger, _fixture = build_closed_course_engine()
        engine.qdrant.points = engine.qdrant.points[:3]
        counting = _CountingOnlineUpdater()
        engine._online_updater = counting
        engine.run_inference("adaptive retrieval")

        self.assertIsNone(skipped)
        self.assertEqual(counting.calls, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
