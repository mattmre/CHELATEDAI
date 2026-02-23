"""
Tests for DimensionMaskPredictor and EmbeddingQualityAssessor (Phase 4)

Run: python -m pytest test_dimension_mask_predictor.py -v
"""

import unittest
from unittest.mock import patch
import numpy as np
import torch

from dimension_mask_predictor import DimensionMaskPredictor, MaskPreTrainer
from embedding_quality import EmbeddingQualityAssessor


# =============================================================================
# DimensionMaskPredictor Tests
# =============================================================================

@patch('dimension_mask_predictor.get_logger')
class TestDimensionMaskPredictor(unittest.TestCase):
    """Tests for the DimensionMaskPredictor neural module."""

    def setUp(self):
        self.input_dim = 128

    def test_initialization_defaults(self, mock_logger):
        """Test default initialization."""
        pred = DimensionMaskPredictor(self.input_dim)
        self.assertEqual(pred.input_dim, self.input_dim)
        self.assertEqual(pred.threshold, 0.5)

    def test_initialization_custom(self, mock_logger):
        """Test custom initialization parameters."""
        pred = DimensionMaskPredictor(self.input_dim, hidden_ratio=0.5, threshold=0.7)
        self.assertEqual(pred.threshold, 0.7)

    def test_forward_output_shape(self, mock_logger):
        """Test that forward produces correct output shape."""
        pred = DimensionMaskPredictor(self.input_dim)
        mean_t = torch.randn(self.input_dim)
        var_t = torch.randn(self.input_dim).abs()
        scores = pred(mean_t, var_t)
        self.assertEqual(scores.shape, (self.input_dim,))

    def test_forward_output_range(self, mock_logger):
        """Test that forward output is in [0, 1] (sigmoid)."""
        pred = DimensionMaskPredictor(self.input_dim)
        mean_t = torch.randn(self.input_dim)
        var_t = torch.randn(self.input_dim).abs()
        scores = pred(mean_t, var_t)
        self.assertTrue(torch.all(scores >= 0).item())
        self.assertTrue(torch.all(scores <= 1).item())

    def test_predict_mask_returns_binary(self, mock_logger):
        """Test predict_mask returns binary numpy array."""
        pred = DimensionMaskPredictor(self.input_dim)
        cluster = np.random.randn(50, self.input_dim).astype(np.float32)
        mask = pred.predict_mask(cluster)
        self.assertEqual(mask.shape, (self.input_dim,))
        unique_vals = set(mask.tolist())
        self.assertTrue(unique_vals.issubset({0.0, 1.0}))

    def test_predict_mask_empty_cluster(self, mock_logger):
        """Test predict_mask with empty cluster returns all-ones mask."""
        pred = DimensionMaskPredictor(self.input_dim)
        mask = pred.predict_mask(np.array([]).reshape(0, self.input_dim))
        np.testing.assert_array_equal(mask, np.ones(self.input_dim))

    def test_predict_mask_deterministic(self, mock_logger):
        """Test predict_mask is deterministic for same input."""
        pred = DimensionMaskPredictor(self.input_dim)
        pred.eval()
        cluster = np.random.randn(50, self.input_dim).astype(np.float32)
        mask1 = pred.predict_mask(cluster)
        mask2 = pred.predict_mask(cluster)
        np.testing.assert_array_equal(mask1, mask2)


# =============================================================================
# MaskPreTrainer Tests
# =============================================================================

@patch('dimension_mask_predictor.get_logger')
class TestMaskPreTrainer(unittest.TestCase):
    """Tests for the MaskPreTrainer distillation class."""

    def setUp(self):
        self.input_dim = 64
        self.predictor = DimensionMaskPredictor(self.input_dim)

    def test_initialization(self, mock_logger):
        """Test pre-trainer initialization."""
        trainer = MaskPreTrainer(self.predictor, chelation_p=85)
        self.assertEqual(trainer.chelation_p, 85)
        self.assertEqual(trainer.buffer_size_current, 0)

    def test_initialization_invalid_predictor(self, mock_logger):
        """Test that non-DimensionMaskPredictor raises TypeError."""
        with self.assertRaises(TypeError):
            MaskPreTrainer("not_a_predictor")

    def test_record_example(self, mock_logger):
        """Test recording training examples."""
        trainer = MaskPreTrainer(self.predictor)
        cluster = np.random.randn(30, self.input_dim).astype(np.float32)
        trainer.record_example(cluster)
        self.assertEqual(trainer.buffer_size_current, 1)

    def test_record_example_empty_cluster(self, mock_logger):
        """Test that empty clusters are not recorded."""
        trainer = MaskPreTrainer(self.predictor)
        trainer.record_example(np.array([]).reshape(0, self.input_dim))
        self.assertEqual(trainer.buffer_size_current, 0)

    def test_buffer_trimming(self, mock_logger):
        """Test that buffer trims to max size."""
        trainer = MaskPreTrainer(self.predictor, buffer_size=5)
        for _ in range(10):
            cluster = np.random.randn(10, self.input_dim).astype(np.float32)
            trainer.record_example(cluster)
        self.assertEqual(trainer.buffer_size_current, 5)

    def test_train_empty_buffer(self, mock_logger):
        """Test training with empty buffer returns zero results."""
        trainer = MaskPreTrainer(self.predictor)
        result = trainer.train(epochs=5)
        self.assertIsNone(result["final_loss"])
        self.assertEqual(result["epochs_trained"], 0)

    def test_train_produces_loss(self, mock_logger):
        """Test that training produces a finite loss."""
        trainer = MaskPreTrainer(self.predictor, learning_rate=0.01)
        for _ in range(10):
            cluster = np.random.randn(20, self.input_dim).astype(np.float32)
            trainer.record_example(cluster)
        result = trainer.train(epochs=5)
        self.assertIsNotNone(result["final_loss"])
        self.assertTrue(np.isfinite(result["final_loss"]))
        self.assertEqual(result["epochs_trained"], 5)

    def test_clear_buffer(self, mock_logger):
        """Test clearing training buffer."""
        trainer = MaskPreTrainer(self.predictor)
        cluster = np.random.randn(10, self.input_dim).astype(np.float32)
        trainer.record_example(cluster)
        self.assertEqual(trainer.buffer_size_current, 1)
        trainer.clear_buffer()
        self.assertEqual(trainer.buffer_size_current, 0)


# =============================================================================
# EmbeddingQualityAssessor Tests
# =============================================================================

@patch('embedding_quality.get_logger')
class TestEmbeddingQualityAssessor(unittest.TestCase):
    """Tests for the EmbeddingQualityAssessor class."""

    def test_initialization_defaults(self, mock_logger):
        """Test default initialization."""
        assessor = EmbeddingQualityAssessor()
        self.assertEqual(assessor.decay_factor, 0.95)
        self.assertEqual(assessor.high_threshold, 0.8)
        self.assertEqual(assessor.low_threshold, 0.3)

    def test_initialization_invalid_decay(self, mock_logger):
        """Test invalid decay factor raises ValueError."""
        with self.assertRaises(ValueError):
            EmbeddingQualityAssessor(decay_factor=0.0)
        with self.assertRaises(ValueError):
            EmbeddingQualityAssessor(decay_factor=1.5)

    def test_initialization_invalid_thresholds(self, mock_logger):
        """Test invalid thresholds raise ValueError."""
        with self.assertRaises(ValueError):
            EmbeddingQualityAssessor(low_threshold=0.9, high_threshold=0.3)

    def test_compute_quality_empty_log(self, mock_logger):
        """Test quality scores on empty chelation log."""
        assessor = EmbeddingQualityAssessor()
        scores = assessor.compute_quality_scores({})
        self.assertEqual(len(scores), 0)

    def test_compute_quality_zero_events(self, mock_logger):
        """Test quality for document with empty event list."""
        assessor = EmbeddingQualityAssessor()
        scores = assessor.compute_quality_scores({0: []})
        self.assertEqual(scores[0], 1.0)

    def test_compute_quality_many_events_lower(self, mock_logger):
        """Test that more chelation events give lower quality."""
        assessor = EmbeddingQualityAssessor()
        vec = [np.zeros(10)]
        log = {0: vec * 1, 1: vec * 10, 2: vec * 50}
        scores = assessor.compute_quality_scores(log)
        self.assertGreater(scores[0], scores[1])
        self.assertGreater(scores[1], scores[2])

    def test_classify_document(self, mock_logger):
        """Test document classification by quality."""
        assessor = EmbeddingQualityAssessor(high_threshold=0.8, low_threshold=0.3)
        self.assertEqual(assessor.classify_document(0.9), "high")
        self.assertEqual(assessor.classify_document(0.5), "medium")
        self.assertEqual(assessor.classify_document(0.1), "low")

    def test_adaptive_threshold_high_quality(self, mock_logger):
        """Test adaptive threshold for high quality document."""
        assessor = EmbeddingQualityAssessor()
        base = 0.001
        threshold = assessor.get_adaptive_threshold(1.0, base)
        self.assertAlmostEqual(threshold, base, places=6)

    def test_adaptive_threshold_low_quality(self, mock_logger):
        """Test adaptive threshold for low quality document is higher."""
        assessor = EmbeddingQualityAssessor()
        base = 0.001
        high_t = assessor.get_adaptive_threshold(1.0, base)
        low_t = assessor.get_adaptive_threshold(0.0, base)
        self.assertGreater(low_t, high_t)

    def test_quality_report(self, mock_logger):
        """Test quality report structure."""
        assessor = EmbeddingQualityAssessor()
        vec = [np.zeros(10)]
        log = {0: vec * 1, 1: vec * 20}
        report = assessor.get_quality_report(log)
        self.assertIn("scores", report)
        self.assertIn("classification_counts", report)
        self.assertIn("mean_quality", report)
        self.assertEqual(len(report["scores"]), 2)

    def test_quality_report_empty(self, mock_logger):
        """Test quality report on empty log."""
        assessor = EmbeddingQualityAssessor()
        report = assessor.get_quality_report({})
        self.assertEqual(report["mean_quality"], 0.0)
        self.assertEqual(report["high_quality_count"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
