"""
Unit Tests for Teacher Distillation Module

Tests the TeacherDistillationHelper class with mocked models to avoid downloads.
Fast, deterministic tests for distillation logic.
"""

import unittest
import numpy as np
from unittest.mock import MagicMock, patch, Mock
from teacher_distillation import (
    TeacherDistillationHelper,
    create_distillation_helper,
    generate_hybrid_targets,
)


class TestTeacherDistillationHelper(unittest.TestCase):
    """Test TeacherDistillationHelper with mocked teacher model."""

    def setUp(self):
        """Set up test fixtures with mocked dependencies."""
        # Patch logger to avoid file I/O
        self.logger_patcher = patch("teacher_distillation.get_logger")
        self.mock_logger = self.logger_patcher.start()
        self.mock_logger.return_value = MagicMock()

        # Create helper (model not loaded yet)
        self.helper = TeacherDistillationHelper(
            teacher_model_name="test-model"
        )

    def tearDown(self):
        """Clean up patches."""
        self.logger_patcher.stop()

    def test_initialization(self):
        """Test that helper initializes correctly without loading model."""
        self.assertEqual(self.helper.teacher_model_name, "test-model")
        self.assertIsNone(self.helper.teacher_model)
        self.assertIsNone(self.helper.teacher_dim)

    @patch("teacher_distillation.SentenceTransformer")
    @patch("teacher_distillation.torch")
    def test_load_teacher_model(self, mock_torch, mock_st_class):
        """Test lazy loading of teacher model."""
        # Mock torch.cuda.is_available
        mock_torch.cuda.is_available.return_value = False

        # Mock SentenceTransformer instance
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st_class.return_value = mock_model

        # Load model
        self.helper.load_teacher_model()

        # Verify model was loaded
        self.assertIsNotNone(self.helper.teacher_model)
        self.assertEqual(self.helper.teacher_dim, 384)
        mock_st_class.assert_called_once_with(
            "test-model",
            device="cpu"
        )

        # Second call should not reload
        self.helper.load_teacher_model()
        self.assertEqual(mock_st_class.call_count, 1)

    @patch("teacher_distillation.SentenceTransformer")
    def test_load_teacher_model_import_error(self, mock_st_class):
        """Test handling of missing sentence-transformers."""
        # Simulate ImportError
        mock_st_class.side_effect = ImportError("No module named 'sentence_transformers'")

        with self.assertRaises(ImportError) as ctx:
            self.helper.load_teacher_model()

        self.assertIn("sentence-transformers required", str(ctx.exception))

    @patch("teacher_distillation.SentenceTransformer")
    @patch("teacher_distillation.torch")
    def test_get_teacher_embeddings(self, mock_torch, mock_st_class):
        """Test getting embeddings from teacher model."""
        # Setup mocks
        mock_torch.cuda.is_available.return_value = False
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        
        # Mock encode to return normalized embeddings
        mock_embeddings = np.random.randn(3, 384)
        # Normalize them
        mock_embeddings = mock_embeddings / (np.linalg.norm(mock_embeddings, axis=1, keepdims=True) + 1e-9)
        mock_model.encode.return_value = mock_embeddings
        
        mock_st_class.return_value = mock_model

        # Get embeddings
        texts = ["text one", "text two", "text three"]
        embeddings = self.helper.get_teacher_embeddings(texts)

        # Verify
        self.assertEqual(embeddings.shape, (3, 384))
        mock_model.encode.assert_called_once()
        
        # Check normalize_embeddings flag was set
        call_kwargs = mock_model.encode.call_args[1]
        self.assertTrue(call_kwargs['normalize_embeddings'])

    def test_get_teacher_embeddings_empty_input(self):
        """Test handling of empty text list."""
        embeddings = self.helper.get_teacher_embeddings([])
        
        self.assertEqual(len(embeddings), 0)
        self.assertIsNone(self.helper.teacher_model)  # Should not load model

    @patch("teacher_distillation.SentenceTransformer")
    @patch("teacher_distillation.torch")
    def test_get_teacher_embeddings_error_fallback(self, mock_torch, mock_st_class):
        """Test fallback to zero vectors on encoding error."""
        # Setup mocks
        mock_torch.cuda.is_available.return_value = False
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.side_effect = RuntimeError("Encoding failed")
        mock_st_class.return_value = mock_model

        # Get embeddings
        texts = ["text one", "text two"]
        embeddings = self.helper.get_teacher_embeddings(texts)

        # Should return zero vectors as fallback
        self.assertEqual(embeddings.shape, (2, 384))
        self.assertTrue(np.allclose(embeddings, np.zeros((2, 384))))

    @patch("teacher_distillation.SentenceTransformer")
    @patch("teacher_distillation.torch")
    def test_check_dimension_compatibility(self, mock_torch, mock_st_class):
        """Test dimension compatibility checking."""
        # Setup mocks
        mock_torch.cuda.is_available.return_value = False
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st_class.return_value = mock_model

        # Test compatible dimensions
        self.assertTrue(self.helper.check_dimension_compatibility(384))

        # Test incompatible dimensions
        self.assertFalse(self.helper.check_dimension_compatibility(768))

    @patch("teacher_distillation.SentenceTransformer")
    @patch("teacher_distillation.torch")
    def test_generate_distillation_targets_teacher_only(self, mock_torch, mock_st_class):
        """Test generating pure teacher targets (weight=1.0)."""
        # Setup mocks
        mock_torch.cuda.is_available.return_value = False
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 4
        
        # Create normalized mock embeddings
        teacher_embeds = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ])
        mock_model.encode.return_value = teacher_embeds
        mock_st_class.return_value = mock_model

        # Current embeddings (student)
        current_embeds = np.array([
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        texts = ["text1", "text2"]

        # Generate targets with teacher_weight=1.0 (pure teacher)
        targets = self.helper.generate_distillation_targets(
            texts=texts,
            current_embeddings=current_embeds,
            teacher_weight=1.0
        )

        # Targets should be identical to teacher embeddings (normalized)
        self.assertEqual(targets.shape, current_embeds.shape)
        np.testing.assert_array_almost_equal(targets, teacher_embeds, decimal=5)

    @patch("teacher_distillation.SentenceTransformer")
    @patch("teacher_distillation.torch")
    def test_generate_distillation_targets_student_only(self, mock_torch, mock_st_class):
        """Test generating student-only targets (weight=0.0)."""
        # Current embeddings (student)
        current_embeds = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ])

        texts = ["text1", "text2"]

        # Generate targets with teacher_weight=0.0 (pure student)
        targets = self.helper.generate_distillation_targets(
            texts=texts,
            current_embeddings=current_embeds,
            teacher_weight=0.0
        )

        # Targets should be identical to student embeddings (no teacher loading)
        np.testing.assert_array_equal(targets, current_embeds)
        self.assertIsNone(self.helper.teacher_model)  # Model should not load

    @patch("teacher_distillation.SentenceTransformer")
    @patch("teacher_distillation.torch")
    def test_generate_distillation_targets_blended(self, mock_torch, mock_st_class):
        """Test generating blended targets (0 < weight < 1)."""
        # Setup mocks
        mock_torch.cuda.is_available.return_value = False
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 4
        
        # Teacher embeddings
        teacher_embeds = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ])
        mock_model.encode.return_value = teacher_embeds
        mock_st_class.return_value = mock_model

        # Student embeddings
        current_embeds = np.array([
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])

        texts = ["text1", "text2"]

        # Generate blended targets (50/50)
        targets = self.helper.generate_distillation_targets(
            texts=texts,
            current_embeddings=current_embeds,
            teacher_weight=0.5
        )

        # Targets should be normalized blend
        self.assertEqual(targets.shape, current_embeds.shape)
        
        # Check normalization (all norms should be ~1.0)
        norms = np.linalg.norm(targets, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(2), decimal=5)
        
        # Targets should be between student and teacher
        # (not exactly halfway due to normalization)
        self.assertFalse(np.allclose(targets, current_embeds))
        self.assertFalse(np.allclose(targets, teacher_embeds))

    @patch("teacher_distillation.SentenceTransformer")
    @patch("teacher_distillation.torch")
    def test_generate_distillation_targets_dimension_mismatch(self, mock_torch, mock_st_class):
        """Test handling of dimension mismatch between teacher and student."""
        # Setup mocks
        mock_torch.cuda.is_available.return_value = False
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 6  # Wrong dim
        
        # Teacher embeddings with wrong dimension
        teacher_embeds = np.array([
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
        ])
        mock_model.encode.return_value = teacher_embeds
        mock_st_class.return_value = mock_model

        # Student embeddings (different dimension)
        current_embeds = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ])

        texts = ["text1", "text2"]

        # Should fallback to student embeddings
        targets = self.helper.generate_distillation_targets(
            texts=texts,
            current_embeddings=current_embeds,
            teacher_weight=0.5
        )

        # Should return student embeddings unchanged
        np.testing.assert_array_equal(targets, current_embeds)

    @patch("teacher_distillation.SentenceTransformer")
    @patch("teacher_distillation.torch")
    def test_compute_alignment_metric(self, mock_torch, mock_st_class):
        """Test alignment metric computation."""
        # Identical embeddings (perfect alignment)
        student = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        teacher = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])

        alignment = self.helper.compute_alignment_metric(student, teacher)
        self.assertAlmostEqual(alignment, 1.0, places=5)

        # Orthogonal embeddings (zero alignment)
        student = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ])
        teacher = np.array([
            [0.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ])

        alignment = self.helper.compute_alignment_metric(student, teacher)
        self.assertAlmostEqual(alignment, 0.0, places=5)

    def test_compute_alignment_metric_empty(self):
        """Test alignment with empty arrays."""
        student = np.array([])
        teacher = np.array([])

        alignment = self.helper.compute_alignment_metric(student, teacher)
        self.assertEqual(alignment, 0.0)

    def test_compute_alignment_metric_shape_mismatch(self):
        """Test alignment with shape mismatch."""
        student = np.array([[1.0, 0.0, 0.0]])
        teacher = np.array([[1.0, 0.0], [0.0, 1.0]])

        alignment = self.helper.compute_alignment_metric(student, teacher)
        self.assertEqual(alignment, 0.0)


class TestDistillationFactoryFunctions(unittest.TestCase):
    """Test factory functions for creating distillation helpers."""

    @patch("teacher_distillation.get_logger")
    @patch("teacher_distillation.ChelationConfig")
    def test_create_distillation_helper_default(self, mock_config, mock_logger):
        """Test creating helper with default config."""
        mock_config.DEFAULT_TEACHER_MODEL = "default-model"
        mock_logger.return_value = MagicMock()

        helper = create_distillation_helper()

        self.assertEqual(helper.teacher_model_name, "default-model")

    @patch("teacher_distillation.get_logger")
    def test_create_distillation_helper_custom(self, mock_logger):
        """Test creating helper with custom model."""
        mock_logger.return_value = MagicMock()

        helper = create_distillation_helper(teacher_model_name="custom-model")

        self.assertEqual(helper.teacher_model_name, "custom-model")

    @patch("teacher_distillation.get_logger")
    @patch("teacher_distillation.SentenceTransformer")
    @patch("teacher_distillation.torch")
    def test_generate_hybrid_targets(self, mock_torch, mock_st_class, mock_logger):
        """Test convenience function for hybrid target generation."""
        # Setup mocks
        mock_torch.cuda.is_available.return_value = False
        mock_logger.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 4
        
        teacher_embeds = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ])
        mock_model.encode.return_value = teacher_embeds
        mock_st_class.return_value = mock_model

        # Inputs
        texts = ["text1", "text2"]
        current_embeds = np.array([
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        homeostatic_targets = np.array([
            [0.0, 0.0, 0.8, 0.6],
            [0.0, 0.6, 0.0, 0.8],
        ])

        # Generate hybrid targets
        targets = generate_hybrid_targets(
            texts=texts,
            current_embeddings=current_embeds,
            homeostatic_targets=homeostatic_targets,
            teacher_weight=0.5
        )

        # Should get blended normalized targets
        self.assertEqual(targets.shape, (2, 4))
        norms = np.linalg.norm(targets, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(2), decimal=5)

    @patch("teacher_distillation.get_logger")
    def test_generate_hybrid_targets_homeostatic_only(self, mock_logger):
        """Test hybrid targets with teacher_weight=0 (pure homeostatic)."""
        mock_logger.return_value = MagicMock()
        
        texts = ["text1", "text2"]
        current_embeds = np.array([
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ])
        homeostatic_targets = np.array([
            [0.0, 0.0, 0.8, 0.6],
            [0.0, 0.6, 0.0, 0.8],
        ])

        # teacher_weight=0 should return homeostatic only
        targets = generate_hybrid_targets(
            texts=texts,
            current_embeddings=current_embeds,
            homeostatic_targets=homeostatic_targets,
            teacher_weight=0.0
        )

        # Should be identical to homeostatic targets
        np.testing.assert_array_equal(targets, homeostatic_targets)


if __name__ == "__main__":
    unittest.main(verbosity=2)
