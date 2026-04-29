"""Stage 2 dyno tests for ChelatedAI control surfaces."""

import unittest

import numpy as np
import torch

from adapter_router import AdapterRouter
from adaptive_gate_orchestrator import AdaptiveGateOrchestrator
from antigravity_engine import AntigravityEngine
from computational_storage_poc.mock_array import ArraySimulation
from device_profiles import DeviceClass, get_profile
from evolution_strategies_optimizer import EvolutionStrategiesConfig, EvolutionaryOnlineUpdater
from quantization_promotion_gate import QuantizationPromotionGate
from run_live_fire_diagnostics import EventCollector, _dataset, _make_engine
from stability_tracker import StabilityTracker
from structural_health_score import StructuralHealthScore


def _ingested_engine(chelation_p=50, use_quantization=True):
    logger = EventCollector()
    engine = _make_engine(logger)
    engine.chelation_p = chelation_p
    engine.use_quantization = use_quantization
    corpus, payloads, _queries, _qrels = _dataset()
    engine.ingest(corpus, payloads)
    return engine, logger


class TestStage2ControlSurfaceDyno(unittest.TestCase):
    def test_threshold_sweep_changes_chelation_action_curve(self):
        actions = {}

        for threshold in [0.0, 0.0002, 0.0004, 0.0008, 0.001, 999.0]:
            engine, _logger = _ingested_engine()
            engine.chelation_threshold = threshold
            _std, _chel, _mask, jaccard = engine.run_inference("adaptive retrieval")
            diag = engine.get_last_runtime_diagnostics()
            actions[threshold] = diag["runtime"]["action"]
            self.assertGreaterEqual(jaccard, 0.0)
            self.assertLessEqual(jaccard, 1.0)
            self.assertGreaterEqual(diag["runtime"]["latency_ms"], 0.0)

        self.assertEqual(actions[0.0], "CHELATE")
        self.assertEqual(actions[999.0], "FAST")
        self.assertTrue(any(action == "CHELATE" for action in actions.values()))
        self.assertTrue(any(action == "FAST" for action in actions.values()))

    def test_percentile_sweep_has_monotonic_mask_density(self):
        engine, _logger = _ingested_engine(use_quantization=False)
        cluster = np.array([
            [0.00, 0.00, 0.00, 0.00, 0.0, 0.0, 0.0, 0.0],
            [0.01, 0.02, 0.03, 0.04, 0.1, 0.2, 0.3, 0.4],
            [0.02, 0.04, 0.06, 0.08, 0.2, 0.4, 0.6, 0.8],
            [0.03, 0.06, 0.09, 0.12, 0.3, 0.6, 0.9, 1.2],
        ])

        densities = []
        for percentile in [25, 50, 75]:
            engine.chelation_p = percentile
            mask = engine._chelate_toxicity(cluster)
            densities.append(float(mask.mean()))

        self.assertLessEqual(densities[0], densities[1])
        self.assertLessEqual(densities[1], densities[2])

    def test_inference_chelation_path_returns_percentile_mask(self):
        densities = []

        for percentile in [25, 75]:
            engine, _logger = _ingested_engine(chelation_p=percentile)
            engine.chelation_threshold = 0.0
            _std, _chel, mask, _jaccard = engine.run_inference("adaptive retrieval")
            diag = engine.get_last_runtime_diagnostics()
            densities.append(float(mask.mean()))
            self.assertEqual(diag["runtime"]["action"], "CHELATE")
            self.assertLess(float(mask.mean()), 1.0)

        self.assertLessEqual(densities[0], densities[1])

    def test_reformulation_rank_fusion_can_change_order(self):
        fused = AntigravityEngine._fuse_reformulated_rankings([
            ["a", "b", "c", "d"],
            ["b", "a", "e", "f"],
            ["b", "g", "a", "h"],
        ], limit=5)

        self.assertEqual(fused[0], "b")
        self.assertIn("e", fused)

    def test_adaptive_threshold_dyno_moves_with_variance_percentile_and_clamps(self):
        engine, _logger = _ingested_engine()
        engine.enable_adaptive_threshold(percentile=90, min_samples=2, min_bound=0.0001, max_bound=0.001)

        for variance in [0.0001, 0.0002, 0.0009, 0.003]:
            engine._update_adaptive_threshold(variance)

        stats = engine.get_threshold_stats()

        self.assertTrue(stats["enabled"])
        self.assertEqual(stats["percentile"], 90)
        self.assertGreater(stats["current_threshold"], 0.0002)
        self.assertLessEqual(stats["current_threshold"], 0.001)

    def test_quantization_dyno_retention_thresholds_reject_lossy_candidates(self):
        baseline = 0.50
        fp32 = 0.70
        quantized = 0.66
        gate_80 = QuantizationPromotionGate(0.8, logger=EventCollector()).evaluate(fp32, quantized, baseline)
        gate_90 = QuantizationPromotionGate(0.9, logger=EventCollector()).evaluate(fp32, quantized, baseline)
        severe_loss = QuantizationPromotionGate(0.8, logger=EventCollector()).evaluate(fp32, 0.50, baseline)

        self.assertTrue(gate_80.passed)
        self.assertFalse(gate_90.passed)
        self.assertFalse(severe_loss.passed)

    def test_norm_drift_dyno_recommends_only_outside_hard_band(self):
        gate = AdaptiveGateOrchestrator(logger=EventCollector())

        safe = gate.evaluate({"norm_drift": {"adapter_norm_ratio_latest": 1.2}})
        watch = gate.evaluate({"norm_drift": {"adapter_norm_ratio_latest": 1.4}})
        hard = gate.evaluate({"norm_drift": {"adapter_norm_ratio_latest": 2.2}})

        self.assertNotIn("normalize_runtime_vectors", safe.actions)
        self.assertIn("inspect_norm_drift", watch.actions)
        self.assertIn("normalize_runtime_vectors", hard.actions)

    def test_route_effectiveness_dyno_separates_good_neutral_bad_routes(self):
        logger = EventCollector()
        router = AdapterRouter(logger=logger)
        router.register("good", [1.0, 0.0], object())
        router.register("bad", [0.0, 1.0], object())

        for score in [0.8, 0.9, 0.85]:
            router.record_outcome("good", score, latency_ms=2.0)
        for score in [0.1, 0.2, 0.15]:
            router.record_outcome("bad", score, latency_ms=2.0)

        effectiveness = router.get_route_effectiveness()
        decision = AdaptiveGateOrchestrator(logger=EventCollector()).evaluate({"route_effectiveness": effectiveness})

        self.assertGreater(effectiveness["routes"]["good"]["mean_jaccard"], 0.8)
        self.assertLess(effectiveness["routes"]["bad"]["mean_jaccard"], 0.25)
        self.assertIn("disable_low_effectiveness_route", decision.actions)

    def test_structural_health_dyno_penalty_tracks_collapse_ramp(self):
        scorer = StructuralHealthScore(logger=EventCollector())

        scores = [
            scorer.evaluate(persistent_collapse_ratio=ratio, isomer_ratio=0.0, topology_drift=0.0).score
            for ratio in [0.0, 0.25, 0.5, 0.75]
        ]

        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[1], scores[2])
        self.assertGreater(scores[2], scores[3])

    def test_storage_latency_dyno_penalizes_only_sla_breach(self):
        gate = AdaptiveGateOrchestrator(storage_latency_sla_ms=0.15, logger=EventCollector())
        array = ArraySimulation(num_drives=2, device_profile=get_profile(DeviceClass.CONSUMER_NVME))

        below = array.sharded_population_evaluation([{"candidate_id": "a", "fitness": 1.0}])
        above = array.sharded_population_evaluation([
            {"candidate_id": f"c{i}", "fitness": float(i)}
            for i in range(10)
        ])

        below_decision = gate.evaluate({"storage": below})
        above_decision = gate.evaluate({"storage": above})

        self.assertLessEqual(below["storage_latency_ms"], 0.15)
        self.assertGreater(above["storage_latency_ms"], 0.15)
        self.assertNotIn("apply_storage_latency_penalty", below_decision.actions)
        self.assertIn("apply_storage_latency_penalty", above_decision.actions)

    def test_online_es_update_dyno_changes_adapter_while_staying_norm_bounded(self):
        adapter = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            adapter.weight.copy_(torch.eye(2))
        before = adapter.weight.detach().clone()
        config = EvolutionStrategiesConfig(
            population_size=4,
            rank=1,
            sigma=0.01,
            learning_rate=0.05,
            generations=1,
            seed=123,
            normalize_fitness=True,
        )
        updater = EvolutionaryOnlineUpdater(adapter, config=config, logger=EventCollector())

        result = updater.update(
            np.array([1.0, 0.0]),
            np.array([[1.0, 0.1], [0.9, 0.0]]),
            np.array([[-1.0, 0.0], [0.0, -1.0]]),
        )
        with torch.no_grad():
            query = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
            ratio = float(torch.norm(adapter(query)).item() / torch.norm(query).item())
        tracker = StabilityTracker()
        tracker.record_norms(adapter_input_norm=1.0, adapter_output_norm=ratio)

        self.assertIsNotNone(result)
        self.assertGreater(float(torch.norm(adapter.weight - before).item()), 0.0)
        self.assertGreaterEqual(ratio, 0.5)
        self.assertLessEqual(ratio, 2.0)
        self.assertEqual(tracker.compute_norm_drift_report()["adapter_norm_ratio_latest"], ratio)


if __name__ == "__main__":
    unittest.main(verbosity=2)
