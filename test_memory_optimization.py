"""
Unit Tests for Memory Optimization Features (PR-A Part 1)

Tests streaming ingestion and chelation log capping without requiring external services.
"""

import unittest
import numpy as np
from collections import defaultdict
from unittest.mock import Mock, patch

try:
    import torch  # noqa: F401
    import sentence_transformers  # noqa: F401
    from config import ChelationConfig
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


@unittest.skipUnless(HAS_TORCH, "Requires torch and sentence-transformers")
class TestStreamingIngestion(unittest.TestCase):
    """Test streaming ingestion functionality."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the entire AntigravityEngine initialization to avoid Qdrant/Ollama dependencies
        with patch('antigravity_engine.QdrantClient'), \
             patch('antigravity_engine.get_logger'), \
             patch('antigravity_engine.create_adapter'), \
             patch('sentence_transformers.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 768
            mock_model.encode.side_effect = lambda texts, **kwargs: np.random.randn(len(texts), 768)
            mock_model.device = "cpu"
            mock_st.return_value = mock_model

            from antigravity_engine import AntigravityEngine
            self.engine = AntigravityEngine(
                qdrant_location=":memory:",
                model_name="all-MiniLM-L6-v2"
            )
            
            # Mock the embed method
            self.engine.embed = Mock(side_effect=self._mock_embed)
            
            # Mock qdrant upsert
            self.engine.qdrant.upsert = Mock()
            
            # Mock logger
            self.engine.logger = Mock()
            self.engine.logger.log_event = Mock()

    def _mock_embed(self, texts):
        """Mock embedding function that returns dummy vectors."""
        if isinstance(texts, str):
            texts = [texts]
        # Return random vectors of correct dimension
        return np.random.randn(len(texts), 768)

    def test_ingest_streaming_with_list(self):
        """Test streaming ingestion with list input."""
        # Prepare test data
        texts = [f"Document {i}" for i in range(250)]
        payloads = [{"index": i} for i in range(250)]
        
        # Run streaming ingestion
        result = self.engine.ingest_streaming(texts, payloads, batch_size=100)
        
        # Verify results
        self.assertEqual(result['total_docs'], 250)
        self.assertEqual(result['total_batches'], 3)
        self.assertEqual(result['start_id'], 0)
        self.assertEqual(result['end_id'], 249)
        
        # Verify embed was called for each batch
        self.assertEqual(self.engine.embed.call_count, 3)
        
        # Verify upsert was called for each batch
        self.assertEqual(self.engine.qdrant.upsert.call_count, 3)

    def test_ingest_streaming_with_generator(self):
        """Test streaming ingestion with generator input (memory-efficient)."""
        # Create generator
        def text_generator():
            for i in range(150):
                yield f"Document {i}"
        
        def payload_generator():
            for i in range(150):
                yield {"index": i}
        
        # Run streaming ingestion
        result = self.engine.ingest_streaming(
            text_generator(),
            payload_generator(),
            batch_size=50
        )
        
        # Verify results
        self.assertEqual(result['total_docs'], 150)
        self.assertEqual(result['total_batches'], 3)
        self.assertEqual(result['start_id'], 0)
        self.assertEqual(result['end_id'], 149)

    def test_ingest_streaming_no_payloads(self):
        """Test streaming ingestion without payloads."""
        texts = [f"Document {i}" for i in range(100)]
        
        result = self.engine.ingest_streaming(texts, batch_size=30)
        
        self.assertEqual(result['total_docs'], 100)
        self.assertEqual(result['total_batches'], 4)
        
        # F-040: Check that payloads respect store_full_text_payload flag (default: True)
        for call_args in self.engine.qdrant.upsert.call_args_list:
            points = call_args[1]['points']
            for point in points:
                # Default should have text in payload (backward compatibility)
                if self.engine.store_full_text_payload:
                    self.assertIn('text', point.payload)

    def test_ingest_streaming_custom_start_id(self):
        """Test streaming ingestion with custom start ID."""
        texts = [f"Document {i}" for i in range(50)]
        
        result = self.engine.ingest_streaming(texts, batch_size=20, start_id=1000)
        
        self.assertEqual(result['total_docs'], 50)
        self.assertEqual(result['start_id'], 1000)
        self.assertEqual(result['end_id'], 1049)
        
        # Verify first batch starts at correct ID
        first_call_points = self.engine.qdrant.upsert.call_args_list[0][1]['points']
        self.assertEqual(first_call_points[0].id, 1000)

    def test_ingest_streaming_mismatched_payload_length(self):
        """Test streaming ingestion when payload iterator is shorter."""
        texts = [f"Document {i}" for i in range(100)]
        payloads = [{"index": i} for i in range(50)]  # Only half
        
        result = self.engine.ingest_streaming(texts, payloads, batch_size=30)
        
        # Should still process all texts, using empty dicts for missing payloads
        self.assertEqual(result['total_docs'], 100)
        self.assertEqual(result['total_batches'], 4)

    def test_ingest_streaming_empty_input(self):
        """Test streaming ingestion with empty input."""
        texts = []
        
        result = self.engine.ingest_streaming(texts, batch_size=100)
        
        self.assertEqual(result['total_docs'], 0)
        self.assertEqual(result['total_batches'], 0)
        self.assertEqual(self.engine.qdrant.upsert.call_count, 0)

    def test_ingest_streaming_single_document(self):
        """Test streaming ingestion with single document."""
        texts = ["Single document"]
        
        result = self.engine.ingest_streaming(texts, batch_size=100)
        
        self.assertEqual(result['total_docs'], 1)
        self.assertEqual(result['total_batches'], 1)

    def test_ingest_streaming_exact_batch_size(self):
        """Test streaming ingestion when total docs is exact multiple of batch size."""
        texts = [f"Document {i}" for i in range(200)]
        
        result = self.engine.ingest_streaming(texts, batch_size=100)
        
        self.assertEqual(result['total_docs'], 200)
        self.assertEqual(result['total_batches'], 2)
        self.assertEqual(result['end_id'], 199)

    def test_ingest_streaming_uses_config_defaults(self):
        """Test that streaming ingestion uses config defaults when not specified."""
        texts = [f"Document {i}" for i in range(100)]
        
        # Call without batch_size to test default
        result = self.engine.ingest_streaming(texts)
        
        # Should use STREAMING_BATCH_SIZE from config
        expected_batches = (100 + ChelationConfig.STREAMING_BATCH_SIZE - 1) // ChelationConfig.STREAMING_BATCH_SIZE
        self.assertEqual(result['total_batches'], expected_batches)

    def test_ingest_streaming_logging(self):
        """Test that streaming ingestion logs progress correctly."""
        texts = [f"Document {i}" for i in range(500)]
        
        self.engine.ingest_streaming(texts, batch_size=50)
        
        # Check that logger was called for start and complete
        log_calls = self.engine.logger.log_event.call_args_list
        
        # Find start and complete events
        start_events = [c for c in log_calls if 'streaming_ingestion_start' in str(c)]
        complete_events = [c for c in log_calls if 'streaming_ingestion_complete' in str(c)]
        
        self.assertGreater(len(start_events), 0, "Should log start event")
        self.assertGreater(len(complete_events), 0, "Should log complete event")


@unittest.skipUnless(HAS_TORCH, "Requires torch and sentence-transformers")
class TestChelationLogCapping(unittest.TestCase):
    """Test chelation log memory management."""

    def setUp(self):
        """Set up test fixtures."""
        with patch('antigravity_engine.QdrantClient'), \
             patch('antigravity_engine.get_logger'), \
             patch('antigravity_engine.create_adapter'), \
             patch('sentence_transformers.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 768
            mock_model.encode.side_effect = lambda texts, **kwargs: np.random.randn(len(texts), 768)
            mock_model.device = "cpu"
            mock_st.return_value = mock_model

            from antigravity_engine import AntigravityEngine
            self.engine = AntigravityEngine(
                qdrant_location=":memory:",
                model_name="all-MiniLM-L6-v2"
            )
            
            # Initialize chelation log
            self.engine.chelation_log = defaultdict(list)
            
            # Mock cosine similarity
            self.engine._cosine_similarity_manual = Mock(return_value=0.9)

    def test_chelation_log_capping_enforced(self):
        """Test that chelation log entries are capped per document."""
        query_vec = np.random.randn(768)
        local_vectors = [np.random.randn(768) for _ in range(10)]
        local_ids = list(range(10))
        
        # Set a small cap for testing
        original_cap = ChelationConfig.CHELATION_LOG_MAX_ENTRIES_PER_DOC
        test_cap = 5
        
        try:
            # Temporarily set config to smaller cap
            ChelationConfig.CHELATION_LOG_MAX_ENTRIES_PER_DOC = test_cap
            
            # Call _spectral_chelation_ranking multiple times to accumulate logs
            for _ in range(10):
                self.engine._spectral_chelation_ranking(query_vec, local_vectors, local_ids)
            
            # Check that each document has at most test_cap entries
            for doc_id in local_ids:
                log_entries = self.engine.chelation_log[doc_id]
                self.assertLessEqual(
                    len(log_entries),
                    test_cap,
                    f"Document {doc_id} has {len(log_entries)} entries, expected <= {test_cap}"
                )
        finally:
            # Restore original cap
            ChelationConfig.CHELATION_LOG_MAX_ENTRIES_PER_DOC = original_cap

    def test_chelation_log_keeps_recent_entries(self):
        """Test that chelation log keeps most recent entries when capping."""
        query_vec = np.random.randn(768)
        local_ids = [0]  # Single document
        
        original_cap = ChelationConfig.CHELATION_LOG_MAX_ENTRIES_PER_DOC
        test_cap = 3
        
        try:
            ChelationConfig.CHELATION_LOG_MAX_ENTRIES_PER_DOC = test_cap
            
            # Create distinguishable center of mass vectors
            centers = []
            for i in range(10):
                # Create distinct vectors by using sequential values
                distinct_vector = np.ones(768) * i
                local_vectors = [distinct_vector + np.random.randn(768) * 0.1]
                
                sorted_ids, center = self.engine._spectral_chelation_ranking(
                    query_vec, local_vectors, local_ids
                )
                centers.append(center)
            
            # Check that log has only most recent entries
            log_entries = self.engine.chelation_log[0]
            self.assertEqual(len(log_entries), test_cap)
            
            # Verify these are the last 3 centers added
            for i, entry in enumerate(log_entries):
                expected_center = centers[-(test_cap - i)]
                np.testing.assert_array_almost_equal(
                    entry, expected_center,
                    err_msg=f"Entry {i} doesn't match expected recent center"
                )
        finally:
            ChelationConfig.CHELATION_LOG_MAX_ENTRIES_PER_DOC = original_cap

    def test_chelation_log_no_capping_under_limit(self):
        """Test that chelation log doesn't cap when under limit."""
        query_vec = np.random.randn(768)
        local_ids = [0, 1, 2]
        local_vectors = [np.random.randn(768) for _ in range(len(local_ids))]
        
        # Call a few times, staying under cap
        for _ in range(5):
            self.engine._spectral_chelation_ranking(query_vec, local_vectors, local_ids)
        
        # All entries should be present (5 calls x 3 docs = 5 entries per doc)
        for doc_id in local_ids:
            self.assertEqual(len(self.engine.chelation_log[doc_id]), 5)

    def test_chelation_log_different_docs_independent(self):
        """Test that different documents have independent log caps."""
        query_vec = np.random.randn(768)
        
        original_cap = ChelationConfig.CHELATION_LOG_MAX_ENTRIES_PER_DOC
        test_cap = 3
        
        try:
            ChelationConfig.CHELATION_LOG_MAX_ENTRIES_PER_DOC = test_cap
            
            # Add many entries for doc 0
            for _ in range(10):
                local_vectors = [np.random.randn(768) for _ in range(2)]
                self.engine._spectral_chelation_ranking(query_vec, local_vectors, [0, 1])
            
            # Add just a few for doc 2
            for _ in range(2):
                local_vectors = [np.random.randn(768) for _ in range(1)]
                self.engine._spectral_chelation_ranking(query_vec, local_vectors, [2])
            
            # Check caps are independent
            self.assertEqual(len(self.engine.chelation_log[0]), test_cap)
            self.assertEqual(len(self.engine.chelation_log[1]), test_cap)
            self.assertEqual(len(self.engine.chelation_log[2]), 2)  # Under cap
        finally:
            ChelationConfig.CHELATION_LOG_MAX_ENTRIES_PER_DOC = original_cap


@unittest.skipUnless(HAS_TORCH, "Requires torch and sentence-transformers")
class TestBackwardCompatibility(unittest.TestCase):
    """Test that new features don't break existing behavior."""

    def setUp(self):
        """Set up test fixtures."""
        with patch('antigravity_engine.QdrantClient'), \
             patch('antigravity_engine.get_logger'), \
             patch('antigravity_engine.create_adapter'), \
             patch('sentence_transformers.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 768
            mock_model.encode.side_effect = lambda texts, **kwargs: np.random.randn(len(texts), 768)
            mock_model.device = "cpu"
            mock_st.return_value = mock_model

            from antigravity_engine import AntigravityEngine
            self.engine = AntigravityEngine(
                qdrant_location=":memory:",
                model_name="all-MiniLM-L6-v2"
            )
            
            self.engine.embed = Mock(side_effect=self._mock_embed)
            self.engine.qdrant.upsert = Mock()
            self.engine.logger = Mock()

    def _mock_embed(self, texts):
        """Mock embedding function."""
        if isinstance(texts, str):
            texts = [texts]
        return np.random.randn(len(texts), 768)

    def test_original_ingest_still_works(self):
        """Test that original ingest() method still works unchanged."""
        texts = [f"Document {i}" for i in range(100)]
        payloads = [{"index": i} for i in range(100)]
        
        # Should not raise any errors
        self.engine.ingest(texts, payloads)
        
        # Verify it was called
        self.assertGreater(self.engine.qdrant.upsert.call_count, 0)

    def test_ingest_and_ingest_streaming_produce_compatible_results(self):
        """Test that both ingest methods produce compatible ID sequences."""
        # Reset mock
        self.engine.qdrant.upsert.reset_mock()
        
        # Use small dataset
        texts1 = [f"Doc {i}" for i in range(10)]
        self.engine.ingest(texts1)
        
        # Get IDs from first method
        ingest_ids = []
        for call_args in self.engine.qdrant.upsert.call_args_list:
            points = call_args[1]['points']
            ingest_ids.extend([p.id for p in points])
        
        # Reset and use streaming
        self.engine.qdrant.upsert.reset_mock()
        texts2 = [f"Doc {i}" for i in range(10)]
        self.engine.ingest_streaming(texts2, batch_size=5)
        
        # Get IDs from streaming method
        streaming_ids = []
        for call_args in self.engine.qdrant.upsert.call_args_list:
            points = call_args[1]['points']
            streaming_ids.extend([p.id for p in points])
        
        # Both should produce sequential IDs from 0
        self.assertEqual(ingest_ids, streaming_ids)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
