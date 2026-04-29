"""Stage 1 safety-testbed component bench tests."""

import json
import unittest

import numpy as np
import torch

from adapter_router import AdapterRouter
from adaptive_gate_orchestrator import AdaptiveGateOrchestrator
from computational_storage_poc.mock_array import ArraySimulation
from dashboard_server import summarize_events
from device_profiles import DeviceClass, get_profile
from distributed_fitness_evaluator import LocalFitnessEvaluator, MockStorageFitnessEvaluator
from fitness_composition_orchestrator import FitnessCompositionOrchestrator
from integrated_diagnostics_report import IntegratedDiagnosticsReport
from quantization_promotion_gate import QuantizationPromotionGate
from query_reformulator import QueryReformulator
from retrieval_fitness_evaluator import RetrievalFitnessEvaluator
from run_live_fire_diagnostics import EventCollector, _dataset, _make_engine
from stability_tracker import StabilityTracker
from structural_health_score import StructuralHealthScore


def _ingested_engine():
    logger = EventCollector()
    engine = _make_engine(logger)
    corpus, payloads, _queries, _qrels = _dataset()
    engine.ingest(corpus, payloads)
    return engine, logger, corpus, payloads


class TestStage1ComponentBench(unittest.TestCase):
    def test_embedding_backend_bench_validates_shape_dtype_and_empty_input(self):
        engine, _logger, _corpus, _payloads = _ingested_engine()

        vectors = engine.embed(["adaptive retrieval", "ssd storage"])
        empty = engine.embed([])

        self.assertEqual(vectors.shape, (2, engine.vector_size))
        self.assertEqual(vectors.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(vectors)))
        self.assertEqual(empty.shape, (0,))
        self.assertEqual(engine._last_embedding_norms["batch_size"], 2)

    def test_vector_store_bench_ingest_retrieve_refresh_and_missing_text_failure(self):
        engine, _logger, corpus, _payloads = _ingested_engine()

        retrieved = engine.qdrant.retrieve(engine.collection_name, ids=[0, 1])
        refresh = engine.refresh_corpus_vectors(batch_size=3)

        self.assertEqual([point.payload["text"] for point in retrieved], corpus[:2])
        self.assertEqual(refresh, {"updated": len(corpus), "failed": 0})

        engine.qdrant.points[0].payload.pop("text")
        with self.assertRaises(ValueError):
            engine.refresh_corpus_vectors(batch_size=3)

    def test_chelation_rerank_bench_exercises_actions_and_noise_logging(self):
        engine, _logger, _corpus, _payloads = _ingested_engine()

        engine.chelation_threshold = 999.0
        fast = engine.run_inference("adaptive retrieval")
        fast_diag = engine.get_last_runtime_diagnostics()

        engine.chelation_threshold = 0.0
        chelated = engine.run_inference("adaptive retrieval")
        chelated_diag = engine.get_last_runtime_diagnostics()

        engine.use_quantization = False
        engine.use_centering = True
        engine.run_inference("adaptive retrieval")
        always_diag = engine.get_last_runtime_diagnostics()

        self.assertEqual(fast_diag["runtime"]["action"], "FAST")
        self.assertEqual(chelated_diag["runtime"]["action"], "CHELATE")
        self.assertEqual(always_diag["runtime"]["action"], "CHELATE_ALWAYS")
        self.assertEqual(fast[2].shape, (engine.vector_size,))
        self.assertEqual(chelated[2].shape, (engine.vector_size,))
        self.assertGreater(len(engine.chelation_log), 0)

    def test_adaptive_threshold_bench_updates_clamps_and_reports_stats(self):
        engine, _logger, _corpus, _payloads = _ingested_engine()
        engine.enable_adaptive_threshold(min_samples=2, min_bound=0.0001, max_bound=0.001)

        engine._update_adaptive_threshold(0.0002)
        engine._update_adaptive_threshold(0.005)
        stats = engine.get_threshold_stats()

        self.assertTrue(stats["enabled"])
        self.assertEqual(stats["variance_samples_count"], 2)
        self.assertGreaterEqual(stats["current_threshold"], 0.0001)
        self.assertLessEqual(stats["current_threshold"], 0.001)

        engine.disable_adaptive_threshold()
        self.assertFalse(engine.get_threshold_stats()["enabled"])

    def test_query_reformulator_bench_variants_bounds_and_logging(self):
        logger = EventCollector()
        reformulator = QueryReformulator(logger=logger)

        variants = reformulator.reformulate("The adaptive retrieval chelation system", max_variants=3)

        self.assertEqual([variant.strategy for variant in variants], ["original", "stopword_removed", "focused_prefix"])
        with self.assertRaises(ValueError):
            reformulator.reformulate("!!!")
        with self.assertRaises(ValueError):
            reformulator.reformulate("valid query", max_variants=0)
        self.assertTrue(any(event["event_type"] == "query_reformulated" for event in logger.events))

    def test_adapter_router_bench_selection_fallback_outcomes_and_bad_inputs(self):
        logger = EventCollector()
        router = AdapterRouter(logger=logger)

        fallback = router.select([1.0, 0.0], fallback=lambda: "default")
        router.register("x", [1.0, 0.0], "adapter-x")
        router.register("y", [0.0, 1.0], "adapter-y")
        route = router.select([0.9, 0.1])
        outcome = router.record_outcome(route.key, jaccard=0.7, latency_ms=3.0)

        self.assertEqual(fallback.key, "fallback")
        self.assertEqual(route.key, "x")
        self.assertEqual(outcome["route_key"], "x")
        self.assertEqual(router.get_route_effectiveness()["routes"]["x"]["count"], 1)
        with self.assertRaises(ValueError):
            router.select([0.0, 0.0])
        with self.assertRaises(ValueError):
            router.register("bad", [], "adapter")

    def test_stability_tracker_bench_all_metrics_and_norm_bands(self):
        tracker = StabilityTracker()
        tracker.logger = EventCollector()
        adapter = torch.nn.Linear(2, 2, bias=False)

        tracker.record_mask([1, 0, 1])
        tracker.record_mask([1, 1, 0])
        tracker.record_variance_distribution([0.1, 0.2, 0.3])
        tracker.record_variance_distribution([0.1, 0.25, 0.35])
        tracker.record_collapse_set(["a", "b"])
        tracker.record_collapse_set(["a"])
        tracker.record_threshold(0.0002)
        tracker.record_threshold(0.0008)
        tracker.record_adapter_snapshot(adapter)
        with torch.no_grad():
            adapter.weight.add_(0.1)
        tracker.record_adapter_snapshot(adapter)
        tracker.record_norms(query_norm=1.0, result_norms=[1.0, 1.1], adapter_input_norm=1.0, adapter_output_norm=1.2)

        report = tracker.get_stability_report()

        self.assertEqual(report["mask_stability"]["count"], 2)
        self.assertEqual(report["variance_convergence"]["count"], 2)
        self.assertGreater(report["persistent_collapse_ratio"], 0.0)
        self.assertGreater(report["threshold_oscillation"], 0.0)
        self.assertGreater(report["adapter_drift"]["total"], 0.0)
        self.assertEqual(report["norm_drift"]["adapter_norm_ratio_latest"], 1.2)

    def test_retrieval_fitness_bench_non_saturated_and_no_relevance_paths(self):
        evaluator = RetrievalFitnessEvaluator(qrels={"q1": {"d1": 1}, "q2": {"d4": 1}}, k=2, logger=EventCollector())
        result = evaluator.evaluate_rankings({"q1": ["d2", "d1"], "q2": ["d5", "d6"]}, candidate_id="bench")
        empty = RetrievalFitnessEvaluator(qrels={"q3": {}}, logger=EventCollector()).evaluate_rankings({"q3": []})

        self.assertGreater(result.fitness, 0.0)
        self.assertLess(result.fitness, 1.0)
        self.assertEqual(result.candidate_id, "bench")
        self.assertEqual(empty.fitness, 0.0)
        self.assertEqual(empty.evaluated_queries, 0)

    def test_fitness_composition_bench_health_quantization_storage_and_bounds(self):
        evaluator = RetrievalFitnessEvaluator(qrels={"q1": {"d1": 1}}, k=2, logger=EventCollector())
        composition = FitnessCompositionOrchestrator(
            evaluator,
            health_weight=0.5,
            quantization_gate=QuantizationPromotionGate(0.8, logger=EventCollector()),
            logger=EventCollector(),
        ).compose_rankings(
            {"q1": ["d1"]},
            health_result=StructuralHealthScore(logger=EventCollector()).evaluate(0.4, 0.0, 0.0),
            quantized_rankings={"q1": ["d2"]},
            baseline_fitness=0.0,
            storage_metadata={"storage_latency_ms": 12.0},
        )

        payload = composition.to_dict()

        self.assertGreaterEqual(composition.final_fitness, 0.0)
        self.assertLessEqual(composition.final_fitness, 1.0)
        self.assertFalse(composition.quantization_gate.passed)
        self.assertEqual(payload["storage_metadata"]["storage_latency_ms"], 12.0)

    def test_quantization_gate_bench_retention_min_gain_and_degradation(self):
        gate = QuantizationPromotionGate(retained_gain_threshold=0.8, minimum_fp32_gain=0.01, logger=EventCollector())

        passed = gate.evaluate(fp32_fitness=0.7, quantized_fitness=0.66, baseline_fitness=0.5)
        failed = gate.evaluate(fp32_fitness=0.7, quantized_fitness=0.55, baseline_fitness=0.5)
        no_gain = gate.evaluate(fp32_fitness=0.505, quantized_fitness=0.5, baseline_fitness=0.5)

        self.assertTrue(passed.passed)
        self.assertFalse(failed.passed)
        self.assertFalse(no_gain.passed)

    def test_quantization_gate_rejects_candidates_without_fp32_lift(self):
        gate = QuantizationPromotionGate(retained_gain_threshold=0.8, logger=EventCollector())

        result = gate.evaluate(fp32_fitness=0.5, quantized_fitness=0.51, baseline_fitness=0.5)

        self.assertFalse(result.passed)
        self.assertIn("fp32_gain_below_minimum", result.reasons)

    def test_structural_health_bench_component_ramps_and_penalty(self):
        scorer = StructuralHealthScore(logger=EventCollector())

        healthy = scorer.evaluate(0.0, 0.0, 0.0)
        degraded = scorer.evaluate(0.5, 0.5, 0.5, metadata={"observed_queries": 4})

        self.assertGreater(healthy.score, degraded.score)
        self.assertLess(degraded.penalty_multiplier(0.5), healthy.penalty_multiplier(0.5))
        self.assertEqual(degraded.components["metadata_observed_queries"], 4)

    def test_adaptive_gate_bench_all_advisory_controls(self):
        decision = AdaptiveGateOrchestrator(
            min_structural_health=0.7,
            storage_latency_sla_ms=5.0,
            min_final_fitness=0.4,
            logger=EventCollector(),
        ).evaluate({
            "final_fitness": 0.5,
            "structural_health": {"score": 0.4},
            "quantization_gate": {"passed": False},
            "storage": {"storage_latency_ms": 12.0},
            "norm_drift": {"adapter_norm_ratio_latest": 3.0},
            "route_effectiveness": {
                "routes": {"bad": {"count": 3, "mean_jaccard": 0.1}},
                "last_route_outcome": {"route_key": "bad"},
            },
            "retrieval_policy": {"policy": "global_scout", "high_variance_fast_path": True},
            "runtime": {"status": "ok", "latency_ms": 4.0},
        })

        for action in {
            "reduce_optimization_aggression",
            "enable_query_reformulation",
            "reject_quantized_candidate",
            "apply_storage_latency_penalty",
            "normalize_runtime_vectors",
            "disable_low_effectiveness_route",
            "prefer_global_scout",
        }:
            self.assertIn(action, decision.actions)
        self.assertTrue(all(recommendation["apply_mode"] == "advisory" for recommendation in decision.recommendations))

    def test_distributed_storage_bench_local_mock_storage_and_ann_merge(self):
        local = LocalFitnessEvaluator(lambda candidate: float(candidate) * 2.0, logger=EventCollector())
        storage = MockStorageFitnessEvaluator(
            ArraySimulation(num_drives=2, device_profile=get_profile(DeviceClass.CONSUMER_NVME)),
            lambda candidate: float(candidate),
            logger=EventCollector(),
        )
        ann = ArraySimulation(num_drives=2).storage_resident_ann_query(
            np.array([1.0, 0.0]),
            {
                0: {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])},
                1: {"c": np.array([0.9, 0.1])},
            },
            top_k=2,
        )

        self.assertEqual(local.evaluate_candidate(0.4).fitness, 0.8)
        self.assertEqual(storage.evaluate_population([0.1, 0.4])["best_candidate_id"], "candidate_1")
        self.assertEqual([hit["doc_id"] for hit in ann["top_hits"]], ["a", "c"])

    def test_reporting_bench_integrated_dashboard_and_verification_shapes(self):
        report = IntegratedDiagnosticsReport(
            cycle=1,
            phase="bench",
            retrieval_fitness=0.5,
            final_fitness=0.45,
            runtime={"latency_ms": np.float32(2.0), "status": "ok"},
            adaptive_gate={"actions": ["prefer_global_scout"]},
            telemetry={"samples": np.array([1, 2])},
        ).to_dict()
        events = [
            {"timestamp": "1", "event_type": "runtime_diagnostics", "runtime": report["runtime"], "route": {"key": "x"}},
            {"timestamp": "2", "adaptive_gate": report["adaptive_gate"]},
        ]
        summary = summarize_events(events)

        json.dumps(report)
        self.assertEqual(report["telemetry"]["samples"], [1, 2])
        self.assertEqual(summary["runtime_diagnostics_count"], 1)
        self.assertEqual(summary["adaptive_gate_actions"], {"prefer_global_scout": 1})


if __name__ == "__main__":
    unittest.main(verbosity=2)
