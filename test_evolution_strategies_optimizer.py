import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch
import torch.nn as nn

from chelation_adapter import LowRankAffineAdapter
from config import ChelationConfig


class TestEvolutionStrategiesOptimizer(unittest.TestCase):
    def setUp(self):
        patcher = patch("evolution_strategies_optimizer.get_logger", return_value=MagicMock())
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_quantization_simulation_preserves_shape_and_bounds(self):
        from evolution_strategies_optimizer import simulate_int8_quantization

        tensor = torch.tensor([[0.0, 0.1, -0.2], [0.3, -0.4, 0.5]])
        quantized = simulate_int8_quantization(tensor, levels=127)

        self.assertEqual(quantized.shape, tensor.shape)
        self.assertFalse(torch.isnan(quantized).any().item())
        self.assertLessEqual(float(quantized.abs().max()), float(tensor.abs().max()) + 1e-6)

    def test_es_training_improves_adapter_loss(self):
        from evolution_strategies_optimizer import EvolutionStrategiesConfig, train_adapter_with_es

        torch.manual_seed(7)
        adapter = LowRankAffineAdapter(input_dim=4, rank=1)
        inputs = torch.eye(4)
        targets = torch.roll(inputs, shifts=1, dims=1)
        targets = torch.nn.functional.normalize(targets, p=2, dim=1)
        criterion = nn.MSELoss()

        with torch.no_grad():
            before = float(criterion(adapter(inputs), targets).item())

        result = train_adapter_with_es(
            adapter,
            inputs,
            targets,
            criterion,
            config=EvolutionStrategiesConfig(
                population_size=12,
                rank=1,
                sigma=0.05,
                learning_rate=0.05,
                generations=6,
                seed=123,
                normalize_fitness=True,
            ),
            logger=MagicMock(),
        )

        self.assertLess(result["final_loss"], before)
        self.assertEqual(result["generations"], 6)

    def test_kalman_sigma_scheduler_lowers_sigma_for_noisy_fitness(self):
        from evolution_strategies_optimizer import KalmanSigmaScheduler

        scheduler = KalmanSigmaScheduler(base_sigma=0.02, process_noise=0.001, min_sigma_ratio=0.1)
        sigma = scheduler.step([0.0, 10.0, -10.0, 5.0])

        self.assertLess(sigma, 0.02)
        self.assertGreaterEqual(sigma, 0.002)

    def test_online_updater_runs_micro_population_update(self):
        from evolution_strategies_optimizer import EvolutionStrategiesConfig, EvolutionaryOnlineUpdater

        torch.manual_seed(5)
        adapter = LowRankAffineAdapter(input_dim=3, rank=1)
        updater = EvolutionaryOnlineUpdater(
            adapter,
            config=EvolutionStrategiesConfig(
                population_size=6,
                rank=1,
                sigma=0.01,
                learning_rate=0.01,
                generations=2,
                seed=42,
            ),
            logger=MagicMock(),
        )

        result = updater.update(
            np.array([1.0, 0.0, 0.0]),
            np.array([[1.0, 0.1, 0.0], [0.9, 0.0, 0.1]]),
            np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]),
        )

        self.assertIsNotNone(result)
        self.assertEqual(result["generations"], 2)

    def test_optimizer_accepts_shared_fitness_interface(self):
        from evolution_strategies_optimizer import EvolutionStrategiesConfig, LowRankEvolutionStrategyOptimizer
        from fitness_interfaces import FitnessEvaluation, FitnessFunctionInterface

        class SumFitness(FitnessFunctionInterface):
            def evaluate_candidate(self, candidate, candidate_id="candidate"):
                return FitnessEvaluation(candidate_id=candidate_id, fitness=float(candidate.weight.sum().item()))

        module = nn.Linear(2, 2, bias=False)
        optimizer = LowRankEvolutionStrategyOptimizer(
            module,
            EvolutionStrategiesConfig(population_size=4, generations=1, seed=3),
            logger=MagicMock(),
        )

        result = optimizer.optimize(SumFitness())

        self.assertEqual(result["generations"], 1)
        self.assertIn("elite_candidates", result)


class TestEvolutionStrategiesConfig(unittest.TestCase):
    def test_es_optimizer_preset_available(self):
        preset = ChelationConfig.get_preset("balanced", "es_optimizer")

        self.assertEqual(preset["optimizer"], "eggroll_es")
        self.assertTrue(preset["quantization_aware"])

    def test_invalid_es_optimizer_config_rejected(self):
        from evolution_strategies_optimizer import EvolutionStrategiesConfig

        with self.assertRaises(ValueError):
            EvolutionStrategiesConfig(population_size=1)


class TestEngineEvolutionaryWiring(unittest.TestCase):
    def test_set_sedimentation_optimizer_records_choice(self):
        from antigravity_engine import AntigravityEngine

        engine = object.__new__(AntigravityEngine)
        engine.logger = MagicMock()

        engine.set_sedimentation_optimizer(
            "eggroll_es",
            population_size=4,
            generations=2,
            quantization_aware=True,
        )

        self.assertEqual(engine._sedimentation_optimizer_type, "eggroll_es")
        self.assertEqual(engine._es_optimizer_kwargs["population_size"], 4)
        self.assertTrue(engine._es_optimizer_kwargs["quantization_aware"])

    def test_invalid_sedimentation_optimizer_rejected(self):
        from antigravity_engine import AntigravityEngine

        engine = object.__new__(AntigravityEngine)
        engine.logger = MagicMock()

        with self.assertRaises(ValueError):
            engine.set_sedimentation_optimizer("not-real")


if __name__ == "__main__":
    unittest.main(verbosity=2)
