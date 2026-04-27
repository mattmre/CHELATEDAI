import unittest
from threading import Lock

import numpy as np
import torch


class TestRetrievalFitnessEvaluator(unittest.TestCase):
    def test_evaluate_rankings_computes_weighted_retrieval_fitness(self):
        from retrieval_fitness_evaluator import RetrievalFitnessEvaluator

        evaluator = RetrievalFitnessEvaluator(
            qrels={"q1": {"d1": 1, "d2": 1}, "q2": {"d4": 1}},
            k=3,
        )
        result = evaluator.evaluate_rankings({
            "q1": ["d1", "d3", "d2"],
            "q2": ["d5", "d4", "d6"],
        })

        self.assertGreater(result.fitness, 0.0)
        self.assertLessEqual(result.fitness, 1.0)
        self.assertEqual(result.evaluated_queries, 2)
        self.assertGreater(result.ndcg_at_k, 0.0)
        self.assertGreater(result.mrr, 0.0)
        self.assertGreater(result.recall_at_k, 0.0)

    def test_composer_applies_structural_penalty(self):
        from retrieval_fitness_evaluator import RetrievalFitnessComposer, RetrievalFitnessEvaluator

        evaluator = RetrievalFitnessEvaluator(qrels={"q1": {"d1": 1}}, k=2)
        composer = RetrievalFitnessComposer(evaluator, penalty_weight=0.5)
        unpenalized = evaluator.evaluate_rankings({"q1": ["d1", "d2"]})
        penalized = composer.compose({"q1": ["d1", "d2"]}, penalty_score=0.0)

        self.assertLess(penalized.fitness, unpenalized.fitness)


class TestEliteArchiveAndESOptions(unittest.TestCase):
    def test_elite_archive_restores_best_parameters(self):
        from elite_archive import EliteArchive

        param = torch.nn.Parameter(torch.tensor([0.0]))
        archive = EliteArchive(max_size=1)
        archive.add_candidate("winner", 1.0, 1, [torch.tensor([3.0])])

        restored = archive.restore_best([param])

        self.assertTrue(restored)
        self.assertAlmostEqual(float(param.item()), 3.0)

    def test_es_result_includes_elite_and_reproducibility_metadata(self):
        from evolution_strategies_optimizer import EvolutionStrategiesConfig, LowRankEvolutionStrategyOptimizer

        torch.manual_seed(1)
        module = torch.nn.Linear(2, 2, bias=False)
        optimizer = LowRankEvolutionStrategyOptimizer(
            module,
            EvolutionStrategiesConfig(
                population_size=4,
                generations=2,
                sigma=0.01,
                learning_rate=0.01,
                seed=99,
                antithetic_sampling=True,
                fitness_shaping="linear_rank",
                elite_pool_size=2,
            ),
            logger=unittest.mock.MagicMock(),
        )

        def fitness_fn():
            with torch.no_grad():
                return float(module.weight.sum().item())

        result = optimizer.optimize(fitness_fn)

        self.assertEqual(result["generations"], 2)
        self.assertIn("elite_candidates", result)
        self.assertGreaterEqual(len(result["elite_candidates"]), 1)
        self.assertEqual(result["reproducibility"]["seed"], 99)


class TestQuantizationAndReproducibilityGates(unittest.TestCase):
    def test_quantization_gate_rejects_collapsed_gain(self):
        from quantization_promotion_gate import QuantizationPromotionGate

        gate = QuantizationPromotionGate(retained_gain_threshold=0.8)
        result = gate.evaluate(fp32_fitness=0.8, quantized_fitness=0.2, baseline_fitness=0.0)

        self.assertFalse(result.passed)
        self.assertLess(result.retained_gain_ratio, 0.8)

    def test_seed_matrix_and_gate_summary(self):
        from reproducibility_context import build_seed_matrix, evaluate_seed_scores

        seeds = build_seed_matrix(42, 3)
        summary = evaluate_seed_scores([0.50, 0.51, 0.49], tolerance=0.03)

        self.assertEqual(seeds, [42, 1051, 2060])
        self.assertTrue(summary.passed)
        self.assertAlmostEqual(summary.mean_score, 0.5)


class TestStructuralAndDistributedFitness(unittest.TestCase):
    def test_structural_health_score_bounded(self):
        from structural_health_score import StructuralHealthScore

        scorer = StructuralHealthScore()
        healthy = scorer.evaluate(persistent_collapse_ratio=0.0, isomer_ratio=0.0, topology_drift=0.0)
        unhealthy = scorer.evaluate(persistent_collapse_ratio=1.0, isomer_ratio=1.0, topology_drift=1.0)

        self.assertAlmostEqual(healthy.score, 1.0)
        self.assertAlmostEqual(unhealthy.score, 0.0)
        self.assertLess(unhealthy.penalty_multiplier(0.5), healthy.penalty_multiplier(0.5))

    def test_mock_storage_fitness_evaluator_reports_latency(self):
        from computational_storage_poc.mock_array import ArraySimulation
        from distributed_fitness_evaluator import MockStorageFitnessEvaluator

        evaluator = MockStorageFitnessEvaluator(ArraySimulation(num_drives=2), lambda candidate: float(candidate), logger=unittest.mock.MagicMock())
        result = evaluator.evaluate_population([0.1, 0.4, 0.2])

        self.assertEqual(result["backend"], "mock_storage")
        self.assertEqual(result["best_candidate_id"], "candidate_1")
        self.assertGreater(result["storage_latency_ms"], 0.0)

    def test_local_fitness_evaluator_returns_interface_batch(self):
        from distributed_fitness_evaluator import LocalFitnessEvaluator
        from fitness_interfaces import FitnessEvaluation

        evaluator = LocalFitnessEvaluator(lambda candidate: float(candidate) * 2.0, logger=unittest.mock.MagicMock())
        evaluations = evaluator.batch_evaluate([0.1, 0.4])
        population = evaluator.evaluate_population([0.1, 0.4])

        self.assertIsInstance(evaluations[0], FitnessEvaluation)
        self.assertEqual(evaluations[1].fitness, 0.8)
        self.assertEqual(population["backend"], "local")
        self.assertEqual(population["best_candidate_id"], "candidate_1")

    def test_device_profile_changes_latency(self):
        from computational_storage_poc.mock_array import ArraySimulation
        from device_profiles import DeviceClass, get_profile

        rp2040 = ArraySimulation(num_drives=1, device_profile=get_profile(DeviceClass.RP2040))
        smartssd = ArraySimulation(num_drives=1, device_profile=get_profile(DeviceClass.SMARTSSD))

        path = [1, 2, 3]
        self.assertGreater(rp2040.single_thread_execution(path), smartssd.single_thread_execution(path))

    def test_storage_resident_ann_query_merges_shard_hits(self):
        from computational_storage_poc.mock_array import ArraySimulation

        simulation = ArraySimulation(num_drives=2)
        result = simulation.storage_resident_ann_query(
            np.array([1.0, 0.0]),
            {
                0: {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])},
                1: {"c": np.array([0.9, 0.1]), "d": np.array([-1.0, 0.0])},
            },
            top_k=2,
        )

        self.assertEqual([hit["doc_id"] for hit in result["top_hits"]], ["a", "c"])
        self.assertEqual(result["scope"], "simulation_only")


class TestP3ResearchScaffolding(unittest.TestCase):
    def test_query_reformulator_generates_deterministic_variants(self):
        from query_reformulator import QueryReformulator

        variants = QueryReformulator().reformulate("The future of storage AI runtime", max_variants=3)

        self.assertEqual(variants[0].strategy, "original")
        self.assertIn("future", variants[1].text)
        self.assertNotIn("the", variants[1].text)

    def test_adapter_router_selects_nearest_centroid(self):
        from adapter_router import AdapterRouter

        router = AdapterRouter(logger=unittest.mock.MagicMock())
        router.register("x", [1.0, 0.0], adapter="adapter-x")
        router.register("y", [0.0, 1.0], adapter="adapter-y")

        route = router.select([0.9, 0.1])

        self.assertEqual(route.key, "x")
        self.assertEqual(route.adapter, "adapter-x")

    def test_query_reformulator_edge_cases(self):
        from query_reformulator import QueryReformulator

        reformulator = QueryReformulator(logger=unittest.mock.MagicMock())

        with self.assertRaises(ValueError):
            reformulator.reformulate("   !!!   ")
        with self.assertRaises(ValueError):
            reformulator.reformulate("valid query", max_variants=0)
        self.assertEqual(len(reformulator.reformulate("the", max_variants=3)), 1)

    def test_engine_adapter_routing_guard_prevents_recursive_reroute(self):
        from adapter_router import AdapterRoute
        from antigravity_engine import AntigravityEngine

        class FakeQdrant:
            def query_points(self, **kwargs):
                return type("QueryResult", (), {
                    "points": [
                        type("Point", (), {"id": 1, "vector": [1.0, 0.0]})()
                    ]
                })()

        class AlwaysRouteElsewhere:
            def __init__(self):
                self.calls = 0

            def select(self, query_vector, fallback=None):
                self.calls += 1
                return AdapterRoute(key="alternate", score=1.0, adapter="alternate-adapter")

        engine = object.__new__(AntigravityEngine)
        engine.vector_size = 2
        engine.collection_name = "test"
        engine.qdrant = FakeQdrant()
        engine.use_quantization = False
        engine.use_centering = False
        engine.chelation_threshold = 1.0
        engine._adaptive_threshold_enabled = False
        engine._adaptive_threshold_lock = Lock()
        engine._variance_history = []
        engine.logger = unittest.mock.MagicMock()
        engine.adapter = "default-adapter"
        engine.embed = lambda _query: np.array([[1.0, 0.0]])
        engine._adapter_router = AlwaysRouteElsewhere()

        std_top, final_top, _mask, _jaccard = engine.run_inference("route me")

        self.assertEqual(std_top, [1])
        self.assertEqual(final_top, [1])
        self.assertEqual(engine.adapter, "default-adapter")
        self.assertEqual(engine._adapter_router.calls, 1)


class TestAdaptiveWorkflowOrchestration(unittest.TestCase):
    def test_fitness_composition_unifies_health_quantization_and_storage(self):
        from fitness_composition_orchestrator import FitnessCompositionOrchestrator
        from quantization_promotion_gate import QuantizationPromotionGate
        from retrieval_fitness_evaluator import RetrievalFitnessEvaluator

        evaluator = RetrievalFitnessEvaluator(qrels={"q1": {"d1": 1}}, k=2, logger=unittest.mock.MagicMock())
        orchestrator = FitnessCompositionOrchestrator(
            retrieval_evaluator=evaluator,
            health_weight=0.5,
            quantization_gate=QuantizationPromotionGate(retained_gain_threshold=0.8, logger=unittest.mock.MagicMock()),
            logger=unittest.mock.MagicMock(),
        )

        result = orchestrator.compose_rankings(
            {"q1": ["d1", "d2"]},
            candidate_id="candidate",
            structural_health_score=0.2,
            quantized_fitness=0.4,
            baseline_fitness=0.0,
            storage_metadata={"storage_latency_ms": 3.5},
        )

        self.assertAlmostEqual(result.structural_health_multiplier, 0.6)
        self.assertAlmostEqual(result.final_fitness, 0.6)
        self.assertFalse(result.quantization_gate.passed)
        self.assertEqual(result.storage_metadata["storage_latency_ms"], 3.5)
        self.assertEqual(result.to_dict()["retrieval_metrics"]["evaluated_queries"], 1.0)

    def test_integrated_diagnostics_and_adaptive_gate_emit_actions(self):
        from adaptive_gate_orchestrator import AdaptiveGateOrchestrator
        from fitness_composition_orchestrator import FitnessCompositionOrchestrator
        from integrated_diagnostics_report import IntegratedDiagnosticsReport
        from quantization_promotion_gate import QuantizationPromotionGate
        from retrieval_fitness_evaluator import RetrievalFitnessEvaluator

        evaluator = RetrievalFitnessEvaluator(qrels={"q1": {"d1": 1}}, k=2, logger=unittest.mock.MagicMock())
        composition = FitnessCompositionOrchestrator(
            evaluator,
            health_weight=0.5,
            quantization_gate=QuantizationPromotionGate(retained_gain_threshold=0.9, logger=unittest.mock.MagicMock()),
            logger=unittest.mock.MagicMock(),
        ).compose_rankings(
            {"q1": ["d1"]},
            structural_health_score=0.4,
            quantized_fitness=0.2,
            baseline_fitness=0.0,
            storage_metadata={"storage_latency_ms": 12.0},
        )
        diagnostics = IntegratedDiagnosticsReport.from_composition(
            composition,
            cycle=1,
            phase="test",
            baseline_fitness=0.0,
            es_result={"generations": 1},
        )

        decision = AdaptiveGateOrchestrator(
            min_structural_health=0.6,
            storage_latency_sla_ms=5.0,
            logger=unittest.mock.MagicMock(),
        ).evaluate(diagnostics.to_dict())
        diagnostics.adaptive_gate = decision.to_dict()

        self.assertTrue(decision.passed)
        self.assertEqual(decision.status, "warning")
        self.assertIn("enable_query_reformulation", decision.actions)
        self.assertIn("reject_quantized_candidate", decision.actions)
        self.assertIn("apply_storage_latency_penalty", decision.actions)
        self.assertEqual(diagnostics.to_dict()["adaptive_gate"]["status"], "warning")


if __name__ == "__main__":
    unittest.main(verbosity=2)
