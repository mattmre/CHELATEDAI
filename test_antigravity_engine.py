"""
Focused unit tests for AntigravityEngine (F-013).
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import requests
import torch

from antigravity_engine import AntigravityEngine


class TestAntigravityEngine(unittest.TestCase):
    def setUp(self):
        self.qdrant_patcher = patch("antigravity_engine.QdrantClient")
        self.logger_patcher = patch("antigravity_engine.get_logger")
        self.adapter_patcher = patch("antigravity_engine.ChelationAdapter")
        self.st_patcher = patch("sentence_transformers.SentenceTransformer")

        self.mock_qdrant_cls = self.qdrant_patcher.start()
        self.mock_get_logger = self.logger_patcher.start()
        self.mock_adapter_cls = self.adapter_patcher.start()
        self.mock_st_cls = self.st_patcher.start()

        self.mock_logger = MagicMock()
        self.mock_get_logger.return_value = self.mock_logger

        self.mock_qdrant = MagicMock()
        self.mock_qdrant.collection_exists.return_value = False
        self.mock_qdrant_cls.return_value = self.mock_qdrant

        self.mock_model = MagicMock()
        self.mock_model.get_sentence_embedding_dimension.return_value = 768
        self.mock_model.device = "cpu"
        self.mock_model.encode.side_effect = lambda texts, **_: np.random.randn(len(texts), 768).astype(np.float32)
        self.mock_st_cls.return_value = self.mock_model

        self.mock_adapter = MagicMock()
        self.mock_adapter.load.return_value = False
        self.mock_adapter.side_effect = lambda x: SimpleNamespace(numpy=lambda: x.numpy())
        self.mock_adapter_cls.return_value = self.mock_adapter

    def tearDown(self):
        self.st_patcher.stop()
        self.adapter_patcher.stop()
        self.logger_patcher.stop()
        self.qdrant_patcher.stop()

    def _make_engine(self, **kwargs):
        return AntigravityEngine(
            qdrant_location=":memory:",
            model_name="all-MiniLM-L6-v2",
            **kwargs,
        )

    def test_embed_local_mode_shape(self):
        engine = self._make_engine()
        result = engine.embed(["a", "b", "c"])
        self.assertEqual(result.shape, (3, 768))
        self.assertIsInstance(result, np.ndarray)

    def test_embed_ollama_mode_timeout_fallback(self):
        self.st_patcher.stop()
        with patch("antigravity_engine.requests") as mock_requests, patch("antigravity_engine.REQUESTS_AVAILABLE", True):
            mock_requests.exceptions = requests.exceptions
            mock_requests.post.side_effect = requests.exceptions.Timeout("timeout")
            engine = AntigravityEngine(qdrant_location=":memory:", model_name="ollama:nomic-embed-text")
            result = engine.embed(["query"])
            self.assertEqual(result.shape, (1, 768))
            self.assertEqual(result.dtype, np.float32)
            self.assertTrue(np.allclose(result, np.zeros((1, 768))))
        self.st_patcher.start()

    def test_embed_ollama_mode_returns_float32(self):
        """F-035: Verify Ollama mode returns float32 dtype, not object dtype."""
        self.st_patcher.stop()
        with patch("antigravity_engine.requests") as mock_requests, patch("antigravity_engine.REQUESTS_AVAILABLE", True):
            mock_requests.exceptions = requests.exceptions
            # Simulate Ollama returning a Python list
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"embedding": [0.1] * 768}
            mock_requests.post.return_value = mock_response
            
            engine = AntigravityEngine(qdrant_location=":memory:", model_name="ollama:nomic-embed-text")
            result = engine.embed(["test query"])
            
            self.assertEqual(result.shape, (1, 768))
            self.assertEqual(result.dtype, np.float32)
            self.assertIsInstance(result, np.ndarray)
        self.st_patcher.start()

    def test_embed_ollama_mode_mixed_success_failure_consistent_dtype(self):
        """F-035: Verify mixed success/fallback returns consistent shape/dtype."""
        self.st_patcher.stop()
        with patch("antigravity_engine.requests") as mock_requests, patch("antigravity_engine.REQUESTS_AVAILABLE", True):
            mock_requests.exceptions = requests.exceptions

            def side_effect_fn(*args, **kwargs):
                prompt = kwargs.get("json", {}).get("prompt", "")
                if prompt in ("test", "success query"):
                    mock_response = MagicMock()
                    mock_response.status_code = 200
                    mock_response.json.return_value = {"embedding": [0.5] * 768}
                    return mock_response
                raise requests.exceptions.Timeout("timeout")
            
            mock_requests.post.side_effect = side_effect_fn
            
            engine = AntigravityEngine(qdrant_location=":memory:", model_name="ollama:nomic-embed-text")
            result = engine.embed(["success query", "timeout query"])
            
            # Should be consistent shape and dtype
            self.assertEqual(result.shape, (2, 768))
            self.assertEqual(result.dtype, np.float32)
            
            # First should be non-zero, second should be zero fallback
            self.assertFalse(np.allclose(result[0], 0))
            self.assertTrue(np.allclose(result[1], 0))
        self.st_patcher.start()

    def test_chelate_toxicity_mask_shape(self):
        engine = self._make_engine()
        cluster = np.random.randn(20, 768)
        mask = engine._chelate_toxicity(cluster)
        self.assertEqual(mask.shape, (768,))
        self.assertTrue(np.all((mask == 0) | (mask == 1)))

    def test_run_inference_fast_path(self):
        engine = self._make_engine(use_quantization=True, use_centering=False)
        qvec = np.random.randn(768)
        engine.embed = MagicMock(return_value=np.array([qvec]))
        points = [SimpleNamespace(id=i, vector=np.ones(768).tolist(), score=0.9) for i in range(10)]
        self.mock_qdrant.query_points.return_value = SimpleNamespace(points=points)
        std_top, chel_top, mask, jaccard = engine.run_inference("x")
        self.assertEqual(std_top, list(range(10)))
        self.assertEqual(chel_top, list(range(10)))
        self.assertEqual(mask.shape, (768,))
        self.assertEqual(jaccard, 1.0)

    def test_run_inference_chelation_path_with_centering(self):
        engine = self._make_engine(use_quantization=False, use_centering=True)
        qvec = np.random.randn(768)
        engine.embed = MagicMock(return_value=np.array([qvec]))
        points = [SimpleNamespace(id=i, vector=np.random.randn(768).tolist(), score=0.9) for i in range(10)]
        self.mock_qdrant.query_points.return_value = SimpleNamespace(points=points)
        engine._spectral_chelation_ranking = MagicMock(return_value=([9, 8, 7, 6, 5, 4, 3, 2, 1, 0], np.zeros(768)))
        std_top, chel_top, _, _ = engine.run_inference("x")
        self.assertEqual(std_top, list(range(10)))
        self.assertEqual(chel_top, [9, 8, 7, 6, 5, 4, 3, 2, 1, 0])

    def test_run_inference_empty_results(self):
        engine = self._make_engine()
        engine.embed = MagicMock(return_value=np.array([np.random.randn(768)]))
        self.mock_qdrant.query_points.return_value = SimpleNamespace(points=[])
        std_top, chel_top, mask, jaccard = engine.run_inference("x")
        self.assertEqual(std_top, [])
        self.assertEqual(chel_top, [])
        self.assertEqual(mask.shape, (768,))
        self.assertEqual(jaccard, 0.0)

    def test_ingest_upserts_points(self):
        engine = self._make_engine()
        engine.embed = MagicMock(return_value=np.random.randn(3, 768))
        texts = ["d1", "d2", "d3"]
        payloads = [{"k": 1}, {"k": 2}, {"k": 3}]
        engine.ingest(texts, payloads)
        self.mock_qdrant.upsert.assert_called_once()
        points = self.mock_qdrant.upsert.call_args.kwargs["points"]
        self.assertEqual(len(points), 3)

    def test_ingest_empty_corpus_no_crash(self):
        engine = self._make_engine()
        engine.embed = MagicMock(return_value=np.array([]))
        engine.ingest([])
        self.mock_qdrant.upsert.assert_not_called()

    def test_checkpoint_manager_initialized(self):
        """F-043: Verify CheckpointManager is initialized during engine construction."""
        with patch("antigravity_engine.CheckpointManager") as mock_cm_cls:
            mock_cm = MagicMock()
            mock_cm_cls.return_value = mock_cm
            
            engine = self._make_engine()
            
            # Verify CheckpointManager was instantiated
            mock_cm_cls.assert_called_once()
            self.assertIsNotNone(engine.checkpoint_manager)
            self.assertEqual(engine.checkpoint_manager, mock_cm)

    def test_sedimentation_uses_safe_training_context(self):
        """F-043: Verify run_sedimentation_cycle uses SafeTrainingContext."""
        with patch("antigravity_engine.SafeTrainingContext") as mock_stc_cls, \
             patch("antigravity_engine.sync_vectors_to_qdrant") as mock_sync:
            
            mock_stc = MagicMock()
            mock_stc.__enter__ = MagicMock(return_value=mock_stc)
            mock_stc.__exit__ = MagicMock(return_value=False)
            mock_stc_cls.return_value = mock_stc
            
            # Mock sync to return success (no failures)
            mock_sync.return_value = (5, 0)

            engine = self._make_engine()
            train_param = torch.nn.Parameter(torch.tensor(1.0))
            self.mock_adapter.parameters.return_value = [train_param]
            self.mock_adapter.side_effect = lambda x: x * train_param

            # Simulate chelation log with targets
            engine.chelation_log = {
                "doc1": [np.random.randn(768) for _ in range(3)],
                "doc2": [np.random.randn(768) for _ in range(3)],
            }
            
            # Mock Qdrant retrieve
            mock_point1 = MagicMock()
            mock_point1.id = "doc1"
            mock_point1.vector = np.random.randn(768).tolist()
            mock_point1.payload = {"text": "test1"}
            
            mock_point2 = MagicMock()
            mock_point2.id = "doc2"
            mock_point2.vector = np.random.randn(768).tolist()
            mock_point2.payload = {"text": "test2"}
            
            self.mock_qdrant.retrieve.return_value = [mock_point1, mock_point2]
            
            # Run sedimentation
            engine.run_sedimentation_cycle(threshold=3, learning_rate=0.001, epochs=1)
            
            # Verify SafeTrainingContext was created with checkpoint manager
            mock_stc_cls.assert_called_once()
            call_args = mock_stc_cls.call_args
            self.assertEqual(call_args[0][0], engine.checkpoint_manager)
            self.assertEqual(call_args[0][1], engine.adapter_path)
            self.assertIn("sedimentation_cycle", call_args[0][2])
            
            # Verify mark_success was called (since failed_updates=0)
            mock_stc.mark_success.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
