"""
Unit Tests for Adaptive Threshold Tuning

Tests the adaptive threshold feature in AntigravityEngine without requiring
external services (Qdrant, Ollama). Uses mocking for service dependencies.
"""

import unittest
import numpy as np
from unittest.mock import Mock, patch
from collections import namedtuple

try:
    import torch  # noqa: F401
    import sentence_transformers  # noqa: F401
    from httpx import Headers
    from antigravity_engine import AntigravityEngine
    from config import ChelationConfig
    from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# Mock types for Qdrant responses
MockPoint = namedtuple('MockPoint', ['id', 'vector', 'score'])
MockQueryResult = namedtuple('MockQueryResult', ['points'])


@unittest.skipUnless(HAS_TORCH, "Requires torch and sentence-transformers")
class TestAdaptiveThresholdDisabled(unittest.TestCase):
    """Test that adaptive threshold is disabled by default (backward compatibility)."""

    def setUp(self):
        """Set up test fixtures with mocked dependencies."""
        with patch('antigravity_engine.QdrantClient'), \
             patch('antigravity_engine.get_logger'), \
             patch('antigravity_engine.create_adapter'), \
             patch('sentence_transformers.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 768
            mock_model.encode.side_effect = lambda texts, **kwargs: np.random.randn(len(texts), 768)
            mock_model.device = "cpu"
            mock_st.return_value = mock_model

            self.engine = AntigravityEngine(
                qdrant_location=":memory:",
                model_name="all-MiniLM-L6-v2"
            )

    def test_disabled_by_default(self):
        """Test that adaptive threshold is disabled by default."""
        self.assertFalse(self.engine._adaptive_threshold_enabled)
        self.assertEqual(
            self.engine.chelation_threshold,
            ChelationConfig.DEFAULT_CHELATION_THRESHOLD
        )
        self.assertEqual(len(self.engine._variance_history), 0)

    def test_default_configuration(self):
        """Test that default configuration is properly set."""
        self.assertEqual(
            self.engine._adaptive_threshold_percentile,
            ChelationConfig.ADAPTIVE_THRESHOLD_PERCENTILE
        )
        self.assertEqual(
            self.engine._adaptive_threshold_window,
            ChelationConfig.ADAPTIVE_THRESHOLD_WINDOW
        )
        self.assertEqual(
            self.engine._adaptive_threshold_min_samples,
            ChelationConfig.ADAPTIVE_THRESHOLD_MIN_SAMPLES
        )

    def test_get_threshold_stats_disabled(self):
        """Test get_threshold_stats when adaptive mode is disabled."""
        stats = self.engine.get_threshold_stats()
        
        self.assertFalse(stats['enabled'])
        self.assertEqual(stats['current_threshold'], ChelationConfig.DEFAULT_CHELATION_THRESHOLD)
        self.assertEqual(stats['variance_samples_count'], 0)
        self.assertNotIn('variance_min', stats)

    def test_sanitize_ollama_text_applies_length_and_control_char_rules(self):
        """Ollama text sanitization should truncate and strip control chars."""
        raw = ("a" * (ChelationConfig.OLLAMA_INPUT_MAX_CHARS + 10)) + "\x00\x01"
        cleaned = self.engine._sanitize_ollama_text(raw, doc_index=0)
        self.assertLessEqual(len(cleaned), ChelationConfig.OLLAMA_INPUT_MAX_CHARS)
        self.assertNotIn("\x00", cleaned)
        self.assertNotIn("\x01", cleaned)


@unittest.skipUnless(HAS_TORCH, "Requires torch and sentence-transformers")
class TestAdaptiveThresholdEnableDisable(unittest.TestCase):
    """Test enabling and disabling adaptive threshold mode."""

    def setUp(self):
        """Set up test fixtures with mocked dependencies."""
        with patch('antigravity_engine.QdrantClient'), \
             patch('antigravity_engine.get_logger'), \
             patch('antigravity_engine.create_adapter'), \
             patch('sentence_transformers.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 768
            mock_model.encode.side_effect = lambda texts, **kwargs: np.random.randn(len(texts), 768)
            mock_model.device = "cpu"
            mock_st.return_value = mock_model

            self.engine = AntigravityEngine(
                qdrant_location=":memory:",
                model_name="all-MiniLM-L6-v2"
            )

    def test_enable_with_defaults(self):
        """Test enabling adaptive threshold with default parameters."""
        self.engine.enable_adaptive_threshold()
        
        self.assertTrue(self.engine._adaptive_threshold_enabled)
        self.assertEqual(
            self.engine._adaptive_threshold_percentile,
            ChelationConfig.ADAPTIVE_THRESHOLD_PERCENTILE
        )

    def test_enable_with_custom_parameters(self):
        """Test enabling adaptive threshold with custom parameters."""
        self.engine.enable_adaptive_threshold(
            percentile=80,
            window=50,
            min_samples=10,
            min_bound=0.0002,
            max_bound=0.005
        )
        
        self.assertTrue(self.engine._adaptive_threshold_enabled)
        self.assertEqual(self.engine._adaptive_threshold_percentile, 80)
        self.assertEqual(self.engine._adaptive_threshold_window, 50)
        self.assertEqual(self.engine._adaptive_threshold_min_samples, 10)
        self.assertEqual(self.engine._adaptive_threshold_min, 0.0002)
        self.assertEqual(self.engine._adaptive_threshold_max, 0.005)

    def test_disable_resets_threshold(self):
        """Test that disabling resets threshold to default."""
        # Enable and modify threshold
        self.engine.enable_adaptive_threshold()
        self.engine.chelation_threshold = 0.001
        self.engine._variance_history = [0.0001, 0.0002, 0.0003]
        
        # Disable
        self.engine.disable_adaptive_threshold()
        
        self.assertFalse(self.engine._adaptive_threshold_enabled)
        self.assertEqual(
            self.engine.chelation_threshold,
            ChelationConfig.DEFAULT_CHELATION_THRESHOLD
        )
        self.assertEqual(len(self.engine._variance_history), 0)

    def test_parameter_validation(self):
        """Test that invalid parameters are validated/clamped."""
        # This should clamp percentile to valid range
        self.engine.enable_adaptive_threshold(
            percentile=150,  # Invalid, should clamp to 100
            window=0,  # Invalid, should clamp to 1
            min_samples=-5  # Invalid, should clamp to 1
        )
        
        self.assertEqual(self.engine._adaptive_threshold_percentile, 100)
        self.assertEqual(self.engine._adaptive_threshold_window, 1)
        self.assertEqual(self.engine._adaptive_threshold_min_samples, 1)


@unittest.skipUnless(HAS_TORCH, "Requires torch and sentence-transformers")
class TestAdaptiveThresholdUpdate(unittest.TestCase):
    """Test threshold update logic during inference."""

    def setUp(self):
        """Set up test fixtures with mocked dependencies."""
        with patch('antigravity_engine.QdrantClient'), \
             patch('antigravity_engine.get_logger'), \
             patch('antigravity_engine.create_adapter'), \
             patch('sentence_transformers.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 768
            mock_model.encode.side_effect = lambda texts, **kwargs: np.random.randn(len(texts), 768)
            mock_model.device = "cpu"
            mock_st.return_value = mock_model

            self.engine = AntigravityEngine(
                qdrant_location=":memory:",
                model_name="all-MiniLM-L6-v2"
            )

    def test_no_update_when_disabled(self):
        """Test that threshold doesn't update when adaptive mode is disabled."""
        initial_threshold = self.engine.chelation_threshold
        
        # Call update helper directly
        self.engine._update_adaptive_threshold(0.001)
        
        # Threshold should not change
        self.assertEqual(self.engine.chelation_threshold, initial_threshold)
        self.assertEqual(len(self.engine._variance_history), 0)

    def test_variance_history_accumulation(self):
        """Test that variance history accumulates correctly."""
        self.engine.enable_adaptive_threshold(window=5)
        
        variances = [0.0001, 0.0002, 0.0003]
        for v in variances:
            self.engine._update_adaptive_threshold(v)
        
        self.assertEqual(len(self.engine._variance_history), 3)
        self.assertEqual(self.engine._variance_history, variances)

    def test_window_trimming(self):
        """Test that variance history is trimmed to window size."""
        self.engine.enable_adaptive_threshold(window=3)
        
        # Add more than window size
        variances = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]
        for v in variances:
            self.engine._update_adaptive_threshold(v)
        
        # Should only keep last 3
        self.assertEqual(len(self.engine._variance_history), 3)
        self.assertEqual(self.engine._variance_history, variances[-3:])

    def test_threshold_update_after_min_samples(self):
        """Test that threshold updates after reaching min_samples."""
        self.engine.enable_adaptive_threshold(
            percentile=75,
            min_samples=3,
            min_bound=0.0,
            max_bound=1.0
        )
        
        # Add samples below min_samples - no update
        self.engine._update_adaptive_threshold(0.0001)
        self.engine._update_adaptive_threshold(0.0002)
        initial_threshold = self.engine.chelation_threshold
        
        # Add one more to reach min_samples - should update
        self.engine._update_adaptive_threshold(0.0003)
        
        # Threshold should be updated to 75th percentile
        expected_threshold = np.percentile([0.0001, 0.0002, 0.0003], 75)
        self.assertAlmostEqual(
            self.engine.chelation_threshold,
            expected_threshold,
            places=6
        )
        self.assertNotEqual(self.engine.chelation_threshold, initial_threshold)

    def test_threshold_clamping_to_bounds(self):
        """Test that computed threshold is clamped to safety bounds."""
        self.engine.enable_adaptive_threshold(
            percentile=90,
            min_samples=2,
            min_bound=0.0005,
            max_bound=0.001
        )
        
        # Add very small values - computed threshold would be below min_bound
        self.engine._update_adaptive_threshold(0.00001)
        self.engine._update_adaptive_threshold(0.00002)
        
        # Should be clamped to min_bound
        self.assertEqual(self.engine.chelation_threshold, 0.0005)
        
        # Now add large values - computed threshold would exceed max_bound
        for _ in range(10):
            self.engine._update_adaptive_threshold(0.01)
        
        # Should be clamped to max_bound
        self.assertEqual(self.engine.chelation_threshold, 0.001)

    def test_percentile_calculation(self):
        """Test that percentile is calculated correctly."""
        self.engine.enable_adaptive_threshold(
            percentile=50,  # Median
            min_samples=5,
            min_bound=0.0,
            max_bound=1.0
        )
        
        # Add known set of values
        variances = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]
        for v in variances:
            self.engine._update_adaptive_threshold(v)
        
        # 50th percentile should be median
        expected_threshold = np.percentile(variances, 50)
        self.assertAlmostEqual(
            self.engine.chelation_threshold,
            expected_threshold,
            places=6
        )


@unittest.skipUnless(HAS_TORCH, "Requires torch and sentence-transformers")
class TestAdaptiveThresholdInRunInference(unittest.TestCase):
    """Test that adaptive threshold update is called during run_inference."""

    def setUp(self):
        """Set up test fixtures with mocked dependencies."""
        with patch('antigravity_engine.QdrantClient') as mock_qdrant, \
             patch('antigravity_engine.get_logger'), \
             patch('antigravity_engine.create_adapter'), \
             patch('sentence_transformers.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 768
            mock_model.encode.side_effect = lambda texts, **kwargs: np.random.randn(len(texts), 768)
            mock_model.device = "cpu"
            mock_st.return_value = mock_model

            self.engine = AntigravityEngine(
                qdrant_location=":memory:",
                model_name="all-MiniLM-L6-v2"
            )
            self.mock_qdrant = mock_qdrant.return_value

    def test_adaptive_update_called_in_inference(self):
        """Test that _update_adaptive_threshold is called during run_inference."""
        # Enable adaptive mode
        self.engine.enable_adaptive_threshold(min_samples=1)
        
        # Mock embed to return a vector
        test_vector = np.random.randn(self.engine.vector_size).tolist()
        self.engine.embed = Mock(return_value=[test_vector])
        
        # Mock Qdrant query_points to return mock results
        mock_vectors = [np.random.randn(self.engine.vector_size).tolist() for _ in range(10)]
        mock_points = [
            MockPoint(id=i, vector=v, score=0.9)
            for i, v in enumerate(mock_vectors)
        ]
        mock_result = MockQueryResult(points=mock_points)
        self.mock_qdrant.query_points.return_value = mock_result
        
        # Run inference
        self.engine.run_inference("test query")
        
        # Verify variance history was updated
        self.assertGreater(len(self.engine._variance_history), 0)

    def test_variance_computed_from_local_cluster(self):
        """Test that variance is computed from local cluster vectors."""
        self.engine.enable_adaptive_threshold(min_samples=1)
        
        # Mock embed
        test_vector = np.random.randn(self.engine.vector_size).tolist()
        self.engine.embed = Mock(return_value=[test_vector])
        
        # Create mock vectors with known variance
        mock_vectors = [
            np.zeros(self.engine.vector_size).tolist(),
            np.ones(self.engine.vector_size).tolist()
        ]
        mock_points = [
            MockPoint(id=i, vector=v, score=0.9)
            for i, v in enumerate(mock_vectors)
        ]
        mock_result = MockQueryResult(points=mock_points)
        self.mock_qdrant.query_points.return_value = mock_result
        
        # Run inference
        self.engine.run_inference("test query")
        
        # Verify that a variance was recorded
        self.assertEqual(len(self.engine._variance_history), 1)
        # The variance should be non-zero since we have different vectors
        self.assertGreater(self.engine._variance_history[0], 0)


@unittest.skipUnless(HAS_TORCH, "Requires torch and sentence-transformers")
class TestAdaptiveThresholdStats(unittest.TestCase):
    """Test threshold statistics reporting."""

    def setUp(self):
        """Set up test fixtures with mocked dependencies."""
        with patch('antigravity_engine.QdrantClient'), \
             patch('antigravity_engine.get_logger'), \
             patch('antigravity_engine.create_adapter'), \
             patch('sentence_transformers.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 768
            mock_model.encode.side_effect = lambda texts, **kwargs: np.random.randn(len(texts), 768)
            mock_model.device = "cpu"
            mock_st.return_value = mock_model

            self.engine = AntigravityEngine(
                qdrant_location=":memory:",
                model_name="all-MiniLM-L6-v2"
            )

    def test_stats_with_history(self):
        """Test statistics when variance history exists."""
        self.engine.enable_adaptive_threshold(min_samples=2)
        
        variances = [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]
        for v in variances:
            self.engine._update_adaptive_threshold(v)
        
        stats = self.engine.get_threshold_stats()
        
        self.assertTrue(stats['enabled'])
        self.assertEqual(stats['variance_samples_count'], 5)
        self.assertAlmostEqual(stats['variance_min'], 0.0001, places=6)
        self.assertAlmostEqual(stats['variance_max'], 0.0005, places=6)
        self.assertAlmostEqual(stats['variance_mean'], 0.0003, places=6)
        self.assertAlmostEqual(stats['variance_median'], 0.0003, places=6)

    def test_stats_structure(self):
        """Test that stats dictionary has all expected keys."""
        self.engine.enable_adaptive_threshold()
        stats = self.engine.get_threshold_stats()
        
        expected_keys = {
            'enabled', 'current_threshold', 'percentile', 'window',
            'min_samples', 'min_bound', 'max_bound', 'variance_samples_count'
        }
        
        for key in expected_keys:
            self.assertIn(key, stats)


@unittest.skipUnless(HAS_TORCH, "Requires torch and sentence-transformers")
class TestQdrantErrorHandling(unittest.TestCase):
    """Test Qdrant error handling in inference path (Finding F-007)."""

    def setUp(self):
        """Set up test fixtures with mocked dependencies."""
        with patch('antigravity_engine.QdrantClient') as mock_qdrant, \
             patch('antigravity_engine.get_logger') as mock_logger, \
             patch('antigravity_engine.create_adapter'), \
             patch('sentence_transformers.SentenceTransformer') as mock_st:
            mock_model = Mock()
            mock_model.get_sentence_embedding_dimension.return_value = 768
            mock_model.encode.side_effect = lambda texts, **kwargs: np.random.randn(len(texts), 768)
            mock_model.device = "cpu"
            mock_st.return_value = mock_model

            self.engine = AntigravityEngine(
                qdrant_location=":memory:",
                model_name="all-MiniLM-L6-v2"
            )
            self.mock_qdrant = mock_qdrant.return_value
            self.mock_logger = mock_logger.return_value

    def test_gravity_sensor_handles_response_handling_exception(self):
        """Test _gravity_sensor returns empty array on ResponseHandlingException."""
        # Configure mock to raise ResponseHandlingException
        self.mock_qdrant.query_points.side_effect = ResponseHandlingException("Connection error")
        
        # Call _gravity_sensor
        query_vec = np.random.randn(self.engine.vector_size).tolist()
        result = self.engine._gravity_sensor(query_vec)
        
        # Assert fallback behavior: empty array
        self.assertEqual(result.shape, (0,))
        self.assertTrue(isinstance(result, np.ndarray))
        
        # Verify logger was called with correct signature
        self.mock_logger.log_error.assert_called_once()
        call_args = self.mock_logger.log_error.call_args
        self.assertEqual(call_args[0][0], "qdrant")  # error_type
        self.assertIn("Qdrant error in _gravity_sensor", call_args[0][1])  # message

    def test_gravity_sensor_handles_unexpected_response(self):
        """Test _gravity_sensor returns empty array on UnexpectedResponse."""
        # Configure mock to raise UnexpectedResponse
        self.mock_qdrant.query_points.side_effect = UnexpectedResponse(
            500, "Internal Server Error", b"{}", Headers({})
        )
        
        # Call _gravity_sensor
        query_vec = np.random.randn(self.engine.vector_size).tolist()
        result = self.engine._gravity_sensor(query_vec)
        
        # Assert fallback behavior: empty array
        self.assertEqual(result.shape, (0,))
        
        # Verify logger was called
        self.mock_logger.log_error.assert_called_once()

    def test_get_chelated_vector_handles_qdrant_error(self):
        """Test get_chelated_vector returns original query vector on Qdrant error."""
        # Mock embed to return a predictable vector
        test_vector = np.random.randn(self.engine.vector_size)
        self.engine.embed = Mock(return_value=[test_vector])
        
        # Configure mock to raise ResponseHandlingException
        self.mock_qdrant.query_points.side_effect = ResponseHandlingException("Connection error")
        
        # Call get_chelated_vector
        result = self.engine.get_chelated_vector("test query")
        
        # Assert fallback behavior: returns original query vector
        np.testing.assert_array_equal(result, test_vector)
        
        # Verify logger was called with correct signature
        self.mock_logger.log_error.assert_called_once()
        call_args = self.mock_logger.log_error.call_args
        self.assertEqual(call_args[0][0], "qdrant")  # error_type
        self.assertIn("Qdrant error in get_chelated_vector", call_args[0][1])  # message

    def test_run_inference_handles_qdrant_error(self):
        """Test run_inference returns empty results on Qdrant error."""
        # Mock embed to return a vector
        test_vector = np.random.randn(self.engine.vector_size).tolist()
        self.engine.embed = Mock(return_value=[test_vector])
        
        # Configure mock to raise UnexpectedResponse
        self.mock_qdrant.query_points.side_effect = UnexpectedResponse(
            500, "Internal Server Error", b"{}", Headers({})
        )
        
        # Call run_inference
        std_top, chel_top, mask, jaccard = self.engine.run_inference("test query")
        
        # Assert fallback behavior: empty results structure
        self.assertEqual(std_top, [])
        self.assertEqual(chel_top, [])
        self.assertEqual(jaccard, 0.0)
        # Mask should be all ones (identity)
        np.testing.assert_array_equal(mask, np.ones(self.engine.vector_size))
        
        # Verify logger was called with correct signature
        self.mock_logger.log_error.assert_called_once()
        call_args = self.mock_logger.log_error.call_args
        self.assertEqual(call_args[0][0], "qdrant")  # error_type
        self.assertIn("Qdrant error in run_inference", call_args[0][1])  # message

    def test_success_path_unchanged(self):
        """Test that success path behavior is unchanged after error handling."""
        # Mock embed
        test_vector = np.random.randn(self.engine.vector_size).tolist()
        self.engine.embed = Mock(return_value=[test_vector])
        
        # Mock successful Qdrant response
        mock_vectors = [np.random.randn(self.engine.vector_size).tolist() for _ in range(10)]
        mock_points = [
            MockPoint(id=i, vector=v, score=0.9)
            for i, v in enumerate(mock_vectors)
        ]
        mock_result = MockQueryResult(points=mock_points)
        self.mock_qdrant.query_points.return_value = mock_result
        
        # Call run_inference
        std_top, chel_top, mask, jaccard = self.engine.run_inference("test query")
        
        # Assert success behavior: non-empty results
        self.assertGreater(len(std_top), 0)
        self.assertGreater(len(chel_top), 0)
        self.assertGreaterEqual(jaccard, 0.0)
        self.assertLessEqual(jaccard, 1.0)
        
        # Verify logger error was NOT called
        self.mock_logger.log_error.assert_not_called()


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
