"""
Unit tests for sedimentation_trainer.py

Tests the shared helper functions used by both antigravity_engine.py
and recursive_decomposer.py for sedimentation-based training.
"""

import unittest
import numpy as np
from unittest.mock import Mock, MagicMock
from qdrant_client.models import PointStruct

from sedimentation_trainer import compute_homeostatic_target, sync_vectors_to_qdrant


class TestComputeHomeostaticTarget(unittest.TestCase):
    """Test suite for compute_homeostatic_target function."""
    
    def test_basic_computation(self):
        """Test basic homeostatic target computation."""
        current_vec = np.array([1.0, 0.0, 0.0])
        noise_vectors = [
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 1.0])
        ]
        push_magnitude = 0.1
        
        result = compute_homeostatic_target(current_vec, noise_vectors, push_magnitude)
        
        # Result should be normalized
        self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=6)
        
        # Result should push away from average noise
        avg_noise = np.mean(noise_vectors, axis=0)
        direction = current_vec - avg_noise
        # Result should have positive dot product with push direction
        self.assertGreater(np.dot(result, direction), 0)
    
    def test_normalized_output(self):
        """Verify output is always normalized to unit length."""
        current_vec = np.array([3.0, 4.0, 0.0])
        noise_vectors = [
            np.array([1.0, 1.0, 1.0]),
            np.array([2.0, 2.0, 2.0])
        ]
        push_magnitude = 0.5
        
        result = compute_homeostatic_target(current_vec, noise_vectors, push_magnitude)
        
        # Should be unit vector
        self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=6)
    
    def test_zero_division_handling(self):
        """Test that zero division is handled with epsilon."""
        # Edge case: current_vec equals avg_noise
        current_vec = np.array([1.0, 1.0, 1.0])
        noise_vectors = [
            np.array([1.0, 1.0, 1.0]),
            np.array([1.0, 1.0, 1.0])
        ]
        push_magnitude = 0.1
        
        # Should not raise error due to epsilon in normalization
        result = compute_homeostatic_target(current_vec, noise_vectors, push_magnitude)
        
        # Result should still be normalized (though direction is arbitrary)
        self.assertAlmostEqual(np.linalg.norm(result), 1.0, places=6)
        self.assertFalse(np.any(np.isnan(result)))
        self.assertFalse(np.any(np.isinf(result)))
    
    def test_push_magnitude_effect(self):
        """Test that larger push magnitude increases deviation from current."""
        current_vec = np.array([1.0, 0.0, 0.0])
        noise_vectors = [np.array([0.0, 1.0, 0.0])]
        
        result_small = compute_homeostatic_target(current_vec, noise_vectors, 0.01)
        result_large = compute_homeostatic_target(current_vec, noise_vectors, 1.0)
        
        # Both should be normalized
        self.assertAlmostEqual(np.linalg.norm(result_small), 1.0, places=6)
        self.assertAlmostEqual(np.linalg.norm(result_large), 1.0, places=6)
        
        # Larger push should deviate more from current direction
        dot_small = np.dot(result_small, current_vec)
        dot_large = np.dot(result_large, current_vec)
        # Both should be close to current_vec but large push slightly less aligned
        self.assertGreater(dot_small, 0.9)
        self.assertLessEqual(dot_small, 1.0)
        self.assertGreater(dot_large, 0.7)
        self.assertLess(dot_large, dot_small)
    
    def test_multiple_noise_vectors(self):
        """Test with varying numbers of noise vectors."""
        current_vec = np.array([1.0, 0.0, 0.0])
        push_magnitude = 0.1
        
        # Single noise vector
        noise_single = [np.array([0.0, 1.0, 0.0])]
        result_single = compute_homeostatic_target(current_vec, noise_single, push_magnitude)
        
        # Multiple noise vectors (same average)
        noise_multiple = [
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 1.0, 0.0]),
            np.array([0.0, 1.0, 0.0])
        ]
        result_multiple = compute_homeostatic_target(current_vec, noise_multiple, push_magnitude)
        
        # Results should be very similar (same average noise)
        np.testing.assert_allclose(result_single, result_multiple, atol=1e-6)


class TestSyncVectorsToQdrant(unittest.TestCase):
    """Test suite for sync_vectors_to_qdrant function."""
    
    def test_successful_sync(self):
        """Test successful vector synchronization."""
        # Mock Qdrant client
        mock_qdrant = Mock()
        mock_point = Mock()
        mock_point.id = "doc1"
        mock_point.payload = {"text": "test"}
        mock_qdrant.retrieve.return_value = [mock_point]
        mock_qdrant.upsert.return_value = None
        
        # Mock logger
        mock_logger = Mock()
        
        # Test data
        ordered_ids = ["doc1"]
        new_vectors = np.array([[0.1, 0.2, 0.3]])
        
        total, failed = sync_vectors_to_qdrant(
            mock_qdrant, "test_collection", ordered_ids,
            new_vectors, chunk_size=10, logger=mock_logger
        )
        
        self.assertEqual(total, 1)
        self.assertEqual(failed, 0)
        mock_qdrant.retrieve.assert_called_once()
        mock_qdrant.upsert.assert_called_once()
    
    def test_chunking(self):
        """Test that large batches are properly chunked."""
        mock_qdrant = Mock()
        mock_logger = Mock()
        
        # Create multiple docs
        n_docs = 25
        ordered_ids = [f"doc{i}" for i in range(n_docs)]
        new_vectors = np.random.rand(n_docs, 384)
        
        # Mock retrieve to return points with payloads
        def mock_retrieve(collection_name, ids, with_vectors):
            points = []
            for doc_id in ids:
                point = Mock()
                point.id = doc_id
                point.payload = {"text": f"text_{doc_id}"}
                points.append(point)
            return points
        
        mock_qdrant.retrieve.side_effect = mock_retrieve
        mock_qdrant.upsert.return_value = None
        
        chunk_size = 10
        total, failed = sync_vectors_to_qdrant(
            mock_qdrant, "test_collection", ordered_ids,
            new_vectors, chunk_size=chunk_size, logger=mock_logger
        )
        
        # Should have 3 chunks: 10, 10, 5
        self.assertEqual(mock_qdrant.retrieve.call_count, 3)
        self.assertEqual(mock_qdrant.upsert.call_count, 3)
        self.assertEqual(total, n_docs)
        self.assertEqual(failed, 0)
    
    def test_payload_preservation(self):
        """Test that existing payloads are preserved."""
        mock_qdrant = Mock()
        mock_logger = Mock()
        
        original_payload = {"text": "important", "metadata": {"key": "value"}}
        mock_point = Mock()
        mock_point.id = "doc1"
        mock_point.payload = original_payload
        mock_qdrant.retrieve.return_value = [mock_point]
        
        # Capture upsert call
        upserted_points = []
        def capture_upsert(collection_name, points):
            upserted_points.extend(points)
        mock_qdrant.upsert.side_effect = capture_upsert
        
        ordered_ids = ["doc1"]
        new_vectors = np.array([[0.1, 0.2, 0.3]])
        
        sync_vectors_to_qdrant(
            mock_qdrant, "test_collection", ordered_ids,
            new_vectors, chunk_size=10, logger=mock_logger
        )
        
        # Verify payload was preserved
        self.assertEqual(len(upserted_points), 1)
        self.assertEqual(upserted_points[0].payload, original_payload)
    
    def test_value_error_handling(self):
        """Test that ValueError is caught and logged."""
        mock_qdrant = Mock()
        mock_logger = Mock()
        
        # Simulate ValueError during retrieve
        mock_qdrant.retrieve.side_effect = ValueError("Invalid data")
        
        ordered_ids = ["doc1", "doc2"]
        new_vectors = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        
        total, failed = sync_vectors_to_qdrant(
            mock_qdrant, "test_collection", ordered_ids,
            new_vectors, chunk_size=10, logger=mock_logger
        )
        
        self.assertEqual(total, 0)
        self.assertEqual(failed, 2)
        mock_logger.log_error.assert_called_once()
        call_args = mock_logger.log_error.call_args
        self.assertEqual(call_args[0][0], "database_update")
        self.assertIn("Invalid vector data", call_args[0][1])
    
    def test_generic_exception_handling(self):
        """Test that generic exceptions are caught and logged."""
        mock_qdrant = Mock()
        mock_logger = Mock()
        
        # Simulate generic error during upsert
        mock_point = Mock()
        mock_point.id = "doc1"
        mock_point.payload = {}
        mock_qdrant.retrieve.return_value = [mock_point]
        mock_qdrant.upsert.side_effect = Exception("Connection failed")
        
        ordered_ids = ["doc1"]
        new_vectors = np.array([[0.1, 0.2, 0.3]])
        
        total, failed = sync_vectors_to_qdrant(
            mock_qdrant, "test_collection", ordered_ids,
            new_vectors, chunk_size=10, logger=mock_logger
        )
        
        self.assertEqual(total, 0)
        self.assertEqual(failed, 1)
        mock_logger.log_error.assert_called_once()
        call_args = mock_logger.log_error.call_args
        self.assertEqual(call_args[0][0], "database_update")
        self.assertIn("failed", call_args[0][1])
    
    def test_partial_failure(self):
        """Test that partial failures are handled correctly."""
        mock_qdrant = Mock()
        mock_logger = Mock()
        
        # First chunk succeeds, second fails
        call_count = [0]
        def side_effect_retrieve(*args, **kwargs):
            call_count[0] += 1
            if call_count[0] == 1:
                # First chunk succeeds
                mock_point = Mock()
                mock_point.id = kwargs['ids'][0]
                mock_point.payload = {}
                return [mock_point]
            else:
                # Second chunk fails
                raise Exception("Network error")
        
        mock_qdrant.retrieve.side_effect = side_effect_retrieve
        mock_qdrant.upsert.return_value = None
        
        ordered_ids = ["doc1", "doc2"]
        new_vectors = np.array([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])
        
        total, failed = sync_vectors_to_qdrant(
            mock_qdrant, "test_collection", ordered_ids,
            new_vectors, chunk_size=1, logger=mock_logger
        )
        
        self.assertEqual(total, 1)  # First chunk succeeded
        self.assertEqual(failed, 1)  # Second chunk failed
        self.assertEqual(mock_logger.log_error.call_count, 1)
    
    def test_empty_batch(self):
        """Test handling of empty batches."""
        mock_qdrant = Mock()
        mock_logger = Mock()
        
        ordered_ids = []
        new_vectors = np.array([]).reshape(0, 384)
        
        total, failed = sync_vectors_to_qdrant(
            mock_qdrant, "test_collection", ordered_ids,
            new_vectors, chunk_size=10, logger=mock_logger
        )
        
        self.assertEqual(total, 0)
        self.assertEqual(failed, 0)
        mock_qdrant.retrieve.assert_not_called()
        mock_qdrant.upsert.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
