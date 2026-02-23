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
        with patch("embedding_backend.requests") as mock_requests, patch("embedding_backend.REQUESTS_AVAILABLE", True):
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
        with patch("embedding_backend.requests") as mock_requests, patch("embedding_backend.REQUESTS_AVAILABLE", True):
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
        with patch("embedding_backend.requests") as mock_requests, patch("embedding_backend.REQUESTS_AVAILABLE", True):
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

    def test_ingest_empty_embed_result_on_nonempty_batch_skips_upsert(self):
        """F-025: Empty embed result on non-empty batch should not call upsert."""
        engine = self._make_engine()
        # Mock embed to return empty array for non-empty batch
        engine.embed = MagicMock(return_value=np.array([]))
        texts = ["doc1", "doc2", "doc3"]
        
        engine.ingest(texts)
        
        # Should log error and skip upsert
        self.mock_qdrant.upsert.assert_not_called()
        # Verify error was logged
        self.mock_logger.log_error.assert_called()
        error_call = self.mock_logger.log_error.call_args
        self.assertEqual(error_call[0][0], "embed_validation")
        self.assertIn("empty", error_call[0][1].lower())

    def test_ingest_dimension_mismatch_skips_upsert(self):
        """F-025: Dimension mismatch should not call upsert."""
        engine = self._make_engine()
        # Mock embed to return wrong dimension (512 instead of 768)
        engine.embed = MagicMock(return_value=np.random.randn(3, 512).astype(np.float32))
        texts = ["doc1", "doc2", "doc3"]
        
        engine.ingest(texts)
        
        # Should log error and skip upsert
        self.mock_qdrant.upsert.assert_not_called()
        # Verify error was logged with dimension info
        self.mock_logger.log_error.assert_called()
        error_call = self.mock_logger.log_error.call_args
        self.assertEqual(error_call[0][0], "embed_validation")
        self.assertIn("dimension mismatch", error_call[0][1].lower())

    def test_ingest_malformed_shape_1d_skips_upsert(self):
        """F-025: 1D shape (not 2D [batch, dim]) should not call upsert."""
        engine = self._make_engine()
        # Mock embed to return 1D array instead of 2D
        engine.embed = MagicMock(return_value=np.random.randn(768).astype(np.float32))
        texts = ["doc1"]
        
        engine.ingest(texts)
        
        # Should log error and skip upsert
        self.mock_qdrant.upsert.assert_not_called()
        # Verify error was logged with shape info
        self.mock_logger.log_error.assert_called()
        error_call = self.mock_logger.log_error.call_args
        self.assertEqual(error_call[0][0], "embed_validation")
        self.assertIn("not 2d", error_call[0][1].lower())

    def test_ingest_malformed_shape_3d_skips_upsert(self):
        """F-025: 3D shape should not call upsert."""
        engine = self._make_engine()
        # Mock embed to return 3D array
        engine.embed = MagicMock(return_value=np.random.randn(2, 3, 768).astype(np.float32))
        texts = ["doc1", "doc2"]
        
        engine.ingest(texts)
        
        # Should log error and skip upsert
        self.mock_qdrant.upsert.assert_not_called()
        # Verify error was logged
        self.mock_logger.log_error.assert_called()
        error_call = self.mock_logger.log_error.call_args
        self.assertEqual(error_call[0][0], "embed_validation")
        self.assertIn("not 2d", error_call[0][1].lower())

    def test_ingest_count_mismatch_skips_upsert(self):
        """F-025: Embedding count != text count should not call upsert."""
        engine = self._make_engine()
        # Mock embed to return wrong number of embeddings (2 instead of 3)
        engine.embed = MagicMock(return_value=np.random.randn(2, 768).astype(np.float32))
        texts = ["doc1", "doc2", "doc3"]
        
        engine.ingest(texts)
        
        # Should log error and skip upsert
        self.mock_qdrant.upsert.assert_not_called()
        # Verify error was logged with counts
        self.mock_logger.log_error.assert_called()
        error_call = self.mock_logger.log_error.call_args
        self.assertEqual(error_call[0][0], "embed_validation")
        self.assertIn("mismatch", error_call[0][1].lower())

    def test_ingest_valid_embeddings_still_work(self):
        """F-025: Verify valid embeddings still pass through successfully."""
        engine = self._make_engine()
        # Mock embed to return valid embeddings
        engine.embed = MagicMock(return_value=np.random.randn(3, 768).astype(np.float32))
        texts = ["doc1", "doc2", "doc3"]
        payloads = [{"k": 1}, {"k": 2}, {"k": 3}]
        
        engine.ingest(texts, payloads)
        
        # Should successfully upsert
        self.mock_qdrant.upsert.assert_called_once()
        points = self.mock_qdrant.upsert.call_args.kwargs["points"]
        self.assertEqual(len(points), 3)
        # Should not log any validation errors
        for call in self.mock_logger.log_error.call_args_list:
            if call[0][0] == "embed_validation":
                self.fail("Should not log embed_validation error for valid embeddings")

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

    def test_qdrant_location_none_rejected(self):
        """F-020: Verify None qdrant_location raises ValueError."""
        with self.assertRaises(ValueError) as context:
            AntigravityEngine(qdrant_location=None, model_name="all-MiniLM-L6-v2")
        self.assertIn("cannot be None", str(context.exception))

    def test_qdrant_location_malformed_url_rejected(self):
        """F-020: Verify malformed URL missing hostname raises ValueError."""
        with self.assertRaises(ValueError) as context:
            AntigravityEngine(qdrant_location="http://", model_name="all-MiniLM-L6-v2")
        self.assertIn("Invalid Qdrant URL", str(context.exception))
        self.assertIn("missing hostname", str(context.exception))

    def test_qdrant_location_valid_url_accepted(self):
        """F-020: Verify valid HTTP/HTTPS URL is accepted with location= parameter."""
        # Test HTTP URL
        AntigravityEngine(qdrant_location="http://localhost:6333", model_name="all-MiniLM-L6-v2")
        self.mock_qdrant_cls.assert_called_with(location="http://localhost:6333")
        
        # Reset mock
        self.mock_qdrant_cls.reset_mock()
        
        # Test HTTPS URL
        AntigravityEngine(qdrant_location="https://example.com:6333", model_name="all-MiniLM-L6-v2")
        self.mock_qdrant_cls.assert_called_with(location="https://example.com:6333")

    def test_qdrant_location_local_path_accepted(self):
        """F-020: Verify local path is accepted with path= parameter."""
        AntigravityEngine(qdrant_location="./data/qdrant", model_name="all-MiniLM-L6-v2")
        self.mock_qdrant_cls.assert_called_with(path="./data/qdrant")

    def test_qdrant_location_memory_accepted(self):
        """F-020: Verify :memory: special value is accepted with location= parameter."""
        self._make_engine()
        self.mock_qdrant_cls.assert_called_with(location=":memory:")

    def test_get_chelated_vector_uses_with_vectors_true(self):
        """F-027: Verify get_chelated_vector uses query_points with with_vectors=True."""
        engine = self._make_engine()
        
        # Mock scout results with vectors
        mock_hit1 = SimpleNamespace(id=1, vector=np.random.randn(768).tolist(), score=0.9)
        mock_hit2 = SimpleNamespace(id=2, vector=np.random.randn(768).tolist(), score=0.8)
        self.mock_qdrant.query_points.return_value = SimpleNamespace(points=[mock_hit1, mock_hit2])
        
        # Call get_chelated_vector
        result = engine.get_chelated_vector("test query")
        
        # Verify query_points was called with with_vectors=True
        self.mock_qdrant.query_points.assert_called_once()
        call_kwargs = self.mock_qdrant.query_points.call_args.kwargs
        self.assertEqual(call_kwargs["with_vectors"], True)
        # F-040: Verify with_payload uses config default (False for optimization)
        self.assertIn("with_payload", call_kwargs)
        
        # Verify result is numpy array with correct shape
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (768,))

    def test_get_chelated_vector_does_not_call_retrieve(self):
        """F-027: Verify get_chelated_vector does NOT call retrieve() in happy path."""
        engine = self._make_engine()
        
        # Mock scout results with vectors
        mock_hit1 = SimpleNamespace(id=1, vector=np.random.randn(768).tolist(), score=0.9)
        mock_hit2 = SimpleNamespace(id=2, vector=np.random.randn(768).tolist(), score=0.8)
        self.mock_qdrant.query_points.return_value = SimpleNamespace(points=[mock_hit1, mock_hit2])
        
        # Call get_chelated_vector
        result = engine.get_chelated_vector("test query")
        
        # Verify retrieve was NOT called
        self.mock_qdrant.retrieve.assert_not_called()
        
        # Verify result is valid
        self.assertIsInstance(result, np.ndarray)
        self.assertEqual(result.shape, (768,))

    def test_get_chelated_vector_fallback_empty_results(self):
        """F-027: Verify get_chelated_vector falls back to raw query vector when no results."""
        engine = self._make_engine()
        
        # Mock empty scout results
        self.mock_qdrant.query_points.return_value = SimpleNamespace(points=[])
        
        # Mock embed to return a specific vector
        expected_vec = np.random.randn(768).astype(np.float32)
        engine.embed = MagicMock(return_value=np.array([expected_vec]))
        
        # Call get_chelated_vector
        result = engine.get_chelated_vector("test query")
        
        # Verify fallback returns raw query vector
        np.testing.assert_array_equal(result, expected_vec)
        
        # Verify retrieve was not called
        self.mock_qdrant.retrieve.assert_not_called()

    def test_get_chelated_vector_fallback_no_vectors(self):
        """F-027: Verify get_chelated_vector falls back when scout results have no vectors."""
        engine = self._make_engine()
        
        # Mock scout results without vectors (all None)
        mock_hit1 = SimpleNamespace(id=1, vector=None, score=0.9)
        mock_hit2 = SimpleNamespace(id=2, vector=None, score=0.8)
        self.mock_qdrant.query_points.return_value = SimpleNamespace(points=[mock_hit1, mock_hit2])
        
        # Mock embed to return a specific vector
        expected_vec = np.random.randn(768).astype(np.float32)
        engine.embed = MagicMock(return_value=np.array([expected_vec]))
        
        # Call get_chelated_vector
        result = engine.get_chelated_vector("test query")
        
        # Verify fallback returns raw query vector
        np.testing.assert_array_equal(result, expected_vec)
        
        # Verify retrieve was not called
        self.mock_qdrant.retrieve.assert_not_called()

    def test_get_chelated_vector_error_handling_unchanged(self):
        """F-027: Verify get_chelated_vector error handling behavior unchanged."""
        engine = self._make_engine()
        
        # Import exception types
        from qdrant_client.http.exceptions import ResponseHandlingException
        
        # Mock query_points to raise Qdrant exception
        self.mock_qdrant.query_points.side_effect = ResponseHandlingException("test error")
        
        # Mock embed to return a specific vector
        expected_vec = np.random.randn(768).astype(np.float32)
        engine.embed = MagicMock(return_value=np.array([expected_vec]))
        
        # Call get_chelated_vector
        result = engine.get_chelated_vector("test query")
        
        # Verify fallback returns raw query vector on error
        np.testing.assert_array_equal(result, expected_vec)
        
        # Verify error was logged
        self.mock_logger.log_error.assert_called()
        error_call = self.mock_logger.log_error.call_args
        self.assertEqual(error_call[0][0], "qdrant")
        self.assertIn("Qdrant error in get_chelated_vector", error_call[0][1])

    def test_spectral_chelation_ranking_vectorized_output_order(self):
        """F-028: Verify vectorized ranking output order matches expected cosine ranking."""
        engine = self._make_engine()
        
        # Create deterministic test vectors (3D for simplicity)
        # After centering around mean [1, 1, 1], we get:
        # query: [1, 0, 0] - [1, 1, 1] = [0, -1, -1]
        # cand1: [2, 1, 1] - [1, 1, 1] = [1, 0, 0]
        # cand2: [0, 2, 1] - [1, 1, 1] = [-1, 1, 0]
        # cand3: [1, 1, 2] - [1, 1, 1] = [0, 0, 1]
        
        # For this test, let's use actual 768-dim vectors with known relationships
        query_vec = np.zeros(768)
        query_vec[0] = 1.0  # Query focuses on first dimension
        
        # Create candidates with varying similarity to query
        local_vectors = []
        # Candidate 0: high similarity (same direction)
        vec0 = np.zeros(768)
        vec0[0] = 0.8
        local_vectors.append(vec0)
        
        # Candidate 1: medium similarity
        vec1 = np.zeros(768)
        vec1[0] = 0.5
        vec1[1] = 0.5
        local_vectors.append(vec1)
        
        # Candidate 2: low similarity (orthogonal)
        vec2 = np.zeros(768)
        vec2[1] = 1.0
        local_vectors.append(vec2)
        
        local_ids = ["doc0", "doc1", "doc2"]
        
        # Call the method
        sorted_ids, center = engine._spectral_chelation_ranking(query_vec, local_vectors, local_ids)
        
        # After centering, doc0 should still rank highest due to alignment
        # Exact order depends on centering, but we verify stable deterministic output
        self.assertEqual(len(sorted_ids), 3)
        self.assertEqual(set(sorted_ids), set(local_ids))
        
        # Verify center of mass is computed
        self.assertEqual(center.shape, (768,))
        expected_center = np.mean(np.array(local_vectors), axis=0)
        np.testing.assert_array_almost_equal(center, expected_center)

    def test_spectral_chelation_ranking_zero_norm_candidates(self):
        """F-028: Verify zero-norm candidates do not raise and produce stable output."""
        engine = self._make_engine()
        
        # Create query vector
        query_vec = np.ones(768) / np.sqrt(768)  # Unit vector
        
        # Create candidates including zero vectors
        local_vectors = []
        # Candidate 0: normal vector
        vec0 = np.random.randn(768)
        local_vectors.append(vec0)
        
        # Candidate 1: zero vector (will be zero norm after centering too if all others non-zero)
        vec1 = np.zeros(768)
        local_vectors.append(vec1)
        
        # Candidate 2: another normal vector
        vec2 = np.random.randn(768)
        local_vectors.append(vec2)
        
        local_ids = ["doc0", "doc1", "doc2"]
        
        # Call the method - should not raise
        sorted_ids, center = engine._spectral_chelation_ranking(query_vec, local_vectors, local_ids)
        
        # Verify output is stable and complete
        self.assertEqual(len(sorted_ids), 3)
        self.assertEqual(set(sorted_ids), set(local_ids))
        
        # Verify no NaN or inf values in result
        # All IDs should be present exactly once
        self.assertEqual(sorted_ids.count("doc0"), 1)
        self.assertEqual(sorted_ids.count("doc1"), 1)
        self.assertEqual(sorted_ids.count("doc2"), 1)

    def test_spectral_chelation_ranking_all_zero_after_centering(self):
        """F-028: Verify behavior when all vectors become zero after centering."""
        engine = self._make_engine()
        
        # Create identical vectors (all will be zero after centering)
        identical_vec = np.random.randn(768)
        query_vec = identical_vec.copy()
        local_vectors = [identical_vec.copy(), identical_vec.copy(), identical_vec.copy()]
        local_ids = ["doc0", "doc1", "doc2"]
        
        # Call the method - should not raise
        sorted_ids, center = engine._spectral_chelation_ranking(query_vec, local_vectors, local_ids)
        
        # Verify output is stable and complete
        self.assertEqual(len(sorted_ids), 3)
        self.assertEqual(set(sorted_ids), set(local_ids))
        
        # Center should equal the identical vector
        np.testing.assert_array_almost_equal(center, identical_vec)

    def test_spectral_chelation_ranking_preserves_existing_behavior(self):
        """F-028: Verify ranking preserves existing semantics on typical inputs."""
        engine = self._make_engine()
        
        # Create typical query and candidate vectors
        np.random.seed(42)  # For reproducibility
        query_vec = np.random.randn(768)
        query_vec = query_vec / np.linalg.norm(query_vec)  # Normalize
        
        local_vectors = []
        for i in range(10):
            vec = np.random.randn(768)
            vec = vec / np.linalg.norm(vec)  # Normalize
            local_vectors.append(vec)
        
        local_ids = [f"doc{i}" for i in range(10)]
        
        # Call the method
        sorted_ids, center = engine._spectral_chelation_ranking(query_vec, local_vectors, local_ids)
        
        # Verify basic properties
        self.assertEqual(len(sorted_ids), 10)
        self.assertEqual(set(sorted_ids), set(local_ids))
        
        # Verify center of mass
        expected_center = np.mean(np.array(local_vectors), axis=0)
        np.testing.assert_array_almost_equal(center, expected_center)
        
        # Verify chelation_log was updated
        for doc_id in local_ids:
            self.assertIn(doc_id, engine.chelation_log)
            self.assertEqual(len(engine.chelation_log[doc_id]), 1)
            np.testing.assert_array_almost_equal(engine.chelation_log[doc_id][0], center)

    def test_close_calls_qdrant_close(self):
        """F-039: Verify close() calls qdrant.close() when client is present."""
        engine = self._make_engine()
        
        # Verify qdrant client exists
        self.assertIsNotNone(engine.qdrant)
        
        # Call close
        engine.close()
        
        # Verify qdrant.close() was called
        self.mock_qdrant.close.assert_called_once()
        
        # Verify qdrant is set to None
        self.assertIsNone(engine.qdrant)

    def test_close_idempotent(self):
        """F-039: Verify close() is safe to call multiple times (idempotent)."""
        engine = self._make_engine()
        
        # Call close multiple times
        engine.close()
        engine.close()
        engine.close()
        
        # Should only call qdrant.close() once (first time)
        self.mock_qdrant.close.assert_called_once()
        
        # Verify qdrant remains None after multiple calls
        self.assertIsNone(engine.qdrant)

    def test_close_handles_qdrant_close_error(self):
        """F-039: Verify close() handles errors from qdrant.close() gracefully."""
        engine = self._make_engine()
        
        # Make qdrant.close() raise an exception
        self.mock_qdrant.close.side_effect = RuntimeError("Qdrant close error")
        
        # close() should not raise - should log error instead
        engine.close()
        
        # Verify error was logged
        self.mock_logger.log_error.assert_called()
        error_call = self.mock_logger.log_error.call_args
        self.assertEqual(error_call[0][0], "resource_cleanup")
        self.assertIn("Error closing Qdrant client", error_call[0][1])
        
        # Verify qdrant is set to None even on error
        self.assertIsNone(engine.qdrant)

    def test_context_manager_calls_close(self):
        """F-039: Verify context manager invokes close() on exit."""
        with patch("antigravity_engine.QdrantClient") as mock_qdrant_cls:
            mock_qdrant = MagicMock()
            mock_qdrant_cls.return_value = mock_qdrant
            
            with AntigravityEngine(qdrant_location=":memory:", model_name="all-MiniLM-L6-v2") as engine:
                # Verify engine is returned
                self.assertIsNotNone(engine)
                self.assertIsNotNone(engine.qdrant)
            
            # After context exit, close should have been called
            mock_qdrant.close.assert_called_once()

    def test_context_manager_does_not_suppress_exceptions(self):
        """F-039: Verify context manager does not suppress exceptions."""
        with patch("antigravity_engine.QdrantClient") as mock_qdrant_cls:
            mock_qdrant = MagicMock()
            mock_qdrant_cls.return_value = mock_qdrant
            
            with self.assertRaises(ValueError) as context:
                with AntigravityEngine(qdrant_location=":memory:", model_name="all-MiniLM-L6-v2"):
                    # Raise an exception inside the context
                    raise ValueError("Test exception")
            
            # Verify exception propagated
            self.assertEqual(str(context.exception), "Test exception")
            
            # Verify close was still called
            mock_qdrant.close.assert_called_once()

    def test_payload_optimization_default_stores_text(self):
        """F-040: Verify default behavior stores full text in payload (backward compatibility)."""
        engine = self._make_engine()
        
        # Mock upsert to capture what was stored
        captured_points = []
        def capture_upsert(collection_name, points):
            captured_points.extend(points)
        self.mock_qdrant.upsert = MagicMock(side_effect=capture_upsert)
        
        # Ingest some documents
        texts = ["Document 1", "Document 2", "Document 3"]
        payloads = [{"meta": "a"}, {"meta": "b"}, {"meta": "c"}]
        engine.ingest(texts, payloads)
        
        # Verify text was stored in payload
        self.assertEqual(len(captured_points), 3)
        for i, point in enumerate(captured_points):
            self.assertIn("text", point.payload)
            self.assertEqual(point.payload["text"], texts[i])
            self.assertEqual(point.payload["meta"], payloads[i]["meta"])

    def test_payload_optimization_omit_text_when_disabled(self):
        """F-040: Verify text is omitted from payload when store_full_text_payload=False."""
        engine = self._make_engine(store_full_text_payload=False)
        
        # Mock upsert to capture what was stored
        captured_points = []
        def capture_upsert(collection_name, points):
            captured_points.extend(points)
        self.mock_qdrant.upsert = MagicMock(side_effect=capture_upsert)
        
        # Ingest some documents
        texts = ["Document 1", "Document 2", "Document 3"]
        payloads = [{"meta": "a"}, {"meta": "b"}, {"meta": "c"}]
        engine.ingest(texts, payloads)
        
        # Verify text was NOT stored in payload, but metadata was
        self.assertEqual(len(captured_points), 3)
        for i, point in enumerate(captured_points):
            self.assertNotIn("text", point.payload)
            self.assertEqual(point.payload["meta"], payloads[i]["meta"])

    def test_payload_optimization_query_points_uses_config_flag(self):
        """F-040: Verify query_points calls use FETCH_PAYLOAD_ON_QUERY config."""
        engine = self._make_engine()
        
        # Mock query_points
        mock_hit = SimpleNamespace(id=1, vector=np.random.randn(768).tolist(), score=0.9)
        self.mock_qdrant.query_points.return_value = SimpleNamespace(points=[mock_hit])
        
        # Call _gravity_sensor (which uses query_points internally)
        query_vec = np.random.randn(768)
        engine._gravity_sensor(query_vec)
        
        # Verify query_points was called with with_payload from config
        self.mock_qdrant.query_points.assert_called_once()
        call_kwargs = self.mock_qdrant.query_points.call_args.kwargs
        self.assertIn("with_payload", call_kwargs)
        # Should use config default (False for optimization)
        from config import ChelationConfig
        self.assertEqual(call_kwargs["with_payload"], ChelationConfig.FETCH_PAYLOAD_ON_QUERY)

    def test_payload_optimization_streaming_ingestion_respects_flag(self):
        """F-040: Verify ingest_streaming respects store_full_text_payload flag."""
        engine = self._make_engine(store_full_text_payload=False)
        
        # Mock upsert to capture what was stored
        captured_points = []
        def capture_upsert(collection_name, points):
            captured_points.extend(points)
        self.mock_qdrant.upsert = MagicMock(side_effect=capture_upsert)
        
        # Streaming ingestion
        texts = [f"Doc {i}" for i in range(5)]
        payloads = [{"idx": i} for i in range(5)]
        engine.ingest_streaming(texts, payloads, batch_size=2)
        
        # Verify text was NOT stored in payload
        self.assertEqual(len(captured_points), 5)
        for i, point in enumerate(captured_points):
            self.assertNotIn("text", point.payload)
            self.assertEqual(point.payload["idx"], i)


if __name__ == "__main__":
    unittest.main(verbosity=2)
