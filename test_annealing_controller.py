from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import torch

from annealing_controller import AnnealingController
from antigravity_engine import AntigravityEngine


class _IdentityAdapter(torch.nn.Module):
    def load(self, _path):
        return False

    def forward(self, tensor):
        return tensor


class _TinyBackend:
    vector_size = 4

    def embed_raw(self, texts):
        vectors = []
        for index, _text in enumerate(texts):
            vector = np.zeros(self.vector_size, dtype=np.float32)
            vector[index % self.vector_size] = 1.0
            vectors.append(vector)
        return np.vstack(vectors)


class TestAnnealingController(unittest.TestCase):
    def test_observe_drift_triggers_above_threshold_only(self):
        controller = AnnealingController(trigger_threshold=0.2)
        controller.observe_drift(0.19)
        self.assertFalse(controller.should_correct())

        controller.observe_drift(0.2)
        self.assertFalse(controller.should_correct())

        controller.observe_drift(0.4)
        self.assertTrue(controller.should_correct())
        self.assertAlmostEqual(controller.temperature, 0.4)

    def test_cooling_reduces_temperature_until_zero(self):
        controller = AnnealingController(initial_temperature=0.5, cooling_rate=0.5)
        controller.end_cycle()
        self.assertAlmostEqual(controller.temperature, 0.25)

        controller.temperature = 1e-7
        controller.end_cycle()
        self.assertEqual(controller.temperature, 0.0)

    def test_cycle_settings_are_monotone_with_temperature(self):
        low = AnnealingController(initial_temperature=0.1, max_temperature=1.0).cycle_settings()
        high = AnnealingController(initial_temperature=1.0, max_temperature=1.0).cycle_settings()

        self.assertLess(low["learning_rate_scale"], high["learning_rate_scale"])
        self.assertLessEqual(low["epochs"], high["epochs"])
        self.assertLess(low["online_intensity"], high["online_intensity"])
        self.assertEqual(high["learning_rate_scale"], 1.0)
        self.assertEqual(high["epochs"], 3)

    def test_invalid_values_raise(self):
        with self.assertRaises(ValueError):
            AnnealingController(initial_temperature=-0.1)
        with self.assertRaises(ValueError):
            AnnealingController(cooling_rate=0.0)
        with self.assertRaises(ValueError):
            AnnealingController(trigger_threshold=-0.1)
        with self.assertRaises(ValueError):
            AnnealingController(max_temperature=0.0)
        with self.assertRaises(ValueError):
            AnnealingController().observe_drift(-0.1)


class TestAnnealingEngineIntegration(unittest.TestCase):
    def _make_engine(self):
        patchers = [
            patch("antigravity_engine.get_logger", return_value=MagicMock()),
            patch("antigravity_engine.create_adapter", return_value=_IdentityAdapter()),
            patch("antigravity_engine.create_embedding_backend", return_value=_TinyBackend()),
        ]
        for patcher in patchers:
            patcher.start()
            self.addCleanup(patcher.stop)
        return AntigravityEngine(
            qdrant_location=":memory:",
            model_name="all-MiniLM-L6-v2",
            store_full_text_payload=True,
        )

    def test_enable_annealing_controller_observes_explicit_drift(self):
        engine = self._make_engine()
        controller = engine.enable_annealing_controller(trigger_threshold=0.2, max_temperature=1.0)

        observation = engine.observe_annealing_drift(0.5)

        self.assertIs(observation["should_correct"], True)
        self.assertAlmostEqual(observation["temperature"], 0.5)
        self.assertIs(controller, engine._annealing_controller)
        self.assertAlmostEqual(engine._temperature, 0.5)

    def test_sedimentation_cycle_uses_scaled_annealing_settings_on_real_engine_path(self):
        engine = self._make_engine()
        engine.ingest(["doc a", "doc b", "doc c", "doc d"])
        engine.enable_annealing_controller(initial_temperature=1.0, trigger_threshold=0.2)

        engine.run_sedimentation_cycle(threshold=1, learning_rate=0.2, epochs=5)

        settings = engine._last_annealing_settings
        self.assertIsNotNone(settings)
        self.assertEqual(settings["original_epochs"], 5)
        self.assertEqual(settings["effective_epochs"], 3)
        self.assertAlmostEqual(settings["effective_learning_rate"], 0.2)
        self.assertAlmostEqual(engine._annealing_controller.temperature, 0.7)
        self.assertAlmostEqual(engine._temperature, 0.7)
        engine.logger.log_event.assert_any_call(
            "sedimentation_start",
            "Running sedimentation cycle (Mode=baseline, Threshold=1, LR=0.2)",
            threshold=1,
            learning_rate=0.2,
            epochs=3,
            training_mode="baseline",
            noise_injection=None,
        )

    def test_computed_drift_uses_existing_structural_signals(self):
        engine = self._make_engine()
        engine.enable_annealing_controller(trigger_threshold=0.2)
        engine._stability_tracker = MagicMock()
        engine._stability_tracker.get_stability_report.return_value = {
            "persistent_collapse_ratio": 0.25,
            "threshold_oscillation": 0.10,
        }
        engine._isomer_detector = MagicMock()
        engine._isomer_detector.get_isomer_report.return_value = {
            "cumulative_mean_strength": 0.35,
        }
        engine._last_runtime_diagnostics = {"runtime": {"global_variance": 0.05}}

        observation = engine.observe_annealing_drift()

        self.assertAlmostEqual(observation["drift_magnitude"], 0.35)
        self.assertAlmostEqual(observation["temperature"], 0.35)


if __name__ == "__main__":
    unittest.main()
