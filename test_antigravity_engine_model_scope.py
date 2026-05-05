import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

try:
    import torch  # noqa: F401
    from antigravity_engine import AntigravityEngine

    HAS_ENGINE_DEPS = True
except ImportError:
    HAS_ENGINE_DEPS = False


@unittest.skipUnless(HAS_ENGINE_DEPS, "Requires AntigravityEngine dependencies")
class TestAntigravityEngineModelScope(unittest.TestCase):
    def setUp(self):
        self.logger_patcher = patch("antigravity_engine.get_logger")
        self.adapter_patcher = patch("antigravity_engine.create_adapter")
        self.backend_patcher = patch("antigravity_engine.create_embedding_backend")
        self.vector_store_patcher = patch("antigravity_engine.create_vector_store")

        self.mock_get_logger = self.logger_patcher.start()
        self.mock_adapter_cls = self.adapter_patcher.start()
        self.mock_backend_factory = self.backend_patcher.start()
        self.mock_vector_store_factory = self.vector_store_patcher.start()

        self.mock_logger = MagicMock()
        self.mock_get_logger.return_value = self.mock_logger

        self.mock_adapter = MagicMock()
        self.mock_adapter.load.return_value = False
        self.mock_adapter.side_effect = lambda tensor: tensor
        self.mock_adapter_cls.return_value = self.mock_adapter

        self.mock_backend = MagicMock()
        self.mock_backend.vector_size = 768
        self.mock_backend.embed_raw.side_effect = (
            lambda texts: np.random.randn(len(texts), 768).astype(np.float32)
        )
        self.mock_backend_factory.return_value = self.mock_backend

        self.mock_vector_store = MagicMock()
        self.mock_vector_store.collection_exists.return_value = False
        self.mock_vector_store_factory.return_value = self.mock_vector_store

    def tearDown(self):
        self.vector_store_patcher.stop()
        self.backend_patcher.stop()
        self.adapter_patcher.stop()
        self.logger_patcher.stop()

    def _make_engine(self):
        return AntigravityEngine(
            qdrant_location=":memory:",
            model_name="all-MiniLM-L6-v2",
            use_quantization=True,
            use_centering=False,
        )

    def test_run_inference_records_model_scope_summary(self):
        engine = self._make_engine()
        runtime = MagicMock()
        runtime.describe_runtime.return_value = {"model_name": "Qwen/Qwen3.5-2B"}
        runtime.observe_text.return_value = {
            "runtime": {"model_name": "Qwen/Qwen3.5-2B"},
            "capture": {
                "token_count": 4,
                "captured_layer_count": 2,
                "layer_indices": [0, 1],
            },
            "steering": {
                "policy_name": "shadow_test",
                "matched_rule_count": 1,
                "runtime_applied": False,
            },
            "memory": {
                "episode_entry_id": "episode_q1",
                "segment_sizes": {"working": 1, "episode": 1, "expectation": 0, "persistent": 0},
            },
            "expectation_comparison": {
                "passed": True,
                "score": 1.0,
            },
            "output_path": "artifact.json",
        }

        engine.enable_model_scope_observation(runtime=runtime)
        qvec = np.random.randn(768)
        engine.embed = MagicMock(return_value=np.array([qvec]))
        points = [SimpleNamespace(id=i, vector=np.ones(768).tolist(), score=0.9) for i in range(10)]
        self.mock_vector_store.query_points.return_value = SimpleNamespace(points=points)

        _, chel_top, _, _ = engine.run_inference("alpha beta gamma")

        self.assertEqual(chel_top, list(range(10)))
        runtime.observe_text.assert_called_once()
        diagnostics = engine.get_last_runtime_diagnostics()
        self.assertEqual(diagnostics["model_scope"]["status"], "observed")
        self.assertEqual(diagnostics["model_scope"]["captured_layer_count"], 2)
        self.assertEqual(diagnostics["model_scope"]["steering"]["matched_rule_count"], 1)
        self.assertEqual(diagnostics["model_scope"]["memory"]["episode_entry_id"], "episode_q1")
        self.assertTrue(diagnostics["model_scope"]["expectation_comparison"]["passed"])
        self.assertEqual(engine.get_runtime_telemetry()["model_scope_observation_count"], 1)
        self.assertEqual(engine.get_last_model_scope_artifact()["output_path"], "artifact.json")

    def test_run_inference_fails_closed_when_model_scope_errors(self):
        engine = self._make_engine()
        runtime = MagicMock()
        runtime.describe_runtime.return_value = {"model_name": "Qwen/Qwen3.5-2B"}
        runtime.observe_text.side_effect = RuntimeError("hook failure")

        engine.enable_model_scope_observation(runtime=runtime)
        qvec = np.random.randn(768)
        engine.embed = MagicMock(return_value=np.array([qvec]))
        points = [SimpleNamespace(id=i, vector=np.ones(768).tolist(), score=0.9) for i in range(10)]
        self.mock_vector_store.query_points.return_value = SimpleNamespace(points=points)

        _, chel_top, _, _ = engine.run_inference("alpha beta gamma")

        self.assertEqual(chel_top, list(range(10)))
        diagnostics = engine.get_last_runtime_diagnostics()
        self.assertEqual(diagnostics["model_scope"]["status"], "error")
        self.assertEqual(engine.get_runtime_telemetry()["model_scope_error_count"], 1)

    def test_close_releases_model_scope_runtime(self):
        engine = self._make_engine()
        runtime = MagicMock()
        runtime.describe_runtime.return_value = {"model_name": "Qwen/Qwen3.5-2B"}

        engine.enable_model_scope_observation(runtime=runtime)
        engine.close()

        runtime.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
