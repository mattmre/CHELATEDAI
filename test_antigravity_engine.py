"""
Focused unit tests for AntigravityEngine (F-013).
"""

import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import requests

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
            self.assertTrue(np.allclose(result, np.zeros((1, 768))))
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
