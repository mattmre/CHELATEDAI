"""
Unit Tests for Teacher Distillation Module

Tests the TeacherDistillationHelper class with mocked models to avoid downloads.
Fast, deterministic tests for distillation logic.
"""

import unittest
import numpy as np
import torch
from unittest.mock import MagicMock, patch
from teacher_distillation import (
    DimensionProjection,
    EnsembleTeacherHelper,
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
        """Test handling of dimension mismatch with projection disabled."""
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

        # Disable projection to test original fallback path
        self.helper._projection_enabled = False

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


class TestDimensionProjection(unittest.TestCase):
    """Test DimensionProjection module."""

    def test_direct_projection_shape(self):
        """Test output shape for direct (no bottleneck) projection."""
        proj = DimensionProjection(teacher_dim=16, student_dim=8)
        x = torch.randn(5, 16)
        out = proj(x)
        self.assertEqual(out.shape, (5, 8))

    def test_bottleneck_projection_shape(self):
        """Test output shape for bottleneck projection."""
        proj = DimensionProjection(teacher_dim=16, student_dim=8, hidden_dim=4)
        x = torch.randn(5, 16)
        out = proj(x)
        self.assertEqual(out.shape, (5, 8))

    def test_near_identity_init(self):
        """Test that matching-dim direct projection is near-identity."""
        dim = 8
        proj = DimensionProjection(teacher_dim=dim, student_dim=dim)
        x = torch.randn(3, dim)
        with torch.no_grad():
            out = proj(x)
        # Output should be very close to input due to near-identity init
        np.testing.assert_allclose(
            out.numpy(), x.numpy(), atol=0.05,
        )

    def test_project_numpy(self):
        """Test convenience numpy projection method."""
        proj = DimensionProjection(teacher_dim=8, student_dim=4)
        data = np.random.randn(3, 8).astype(np.float32)
        result = proj.project_numpy(data)
        self.assertEqual(result.shape, (3, 4))
        self.assertIsInstance(result, np.ndarray)

    def test_projection_preserves_norms(self):
        """Test that near-identity projection approximately preserves norms."""
        dim = 16
        proj = DimensionProjection(teacher_dim=dim, student_dim=dim)
        x = torch.randn(10, dim)
        x_norms = torch.norm(x, dim=1)
        with torch.no_grad():
            out = proj(x)
        out_norms = torch.norm(out, dim=1)
        # Norms should be approximately preserved for matching dims
        np.testing.assert_allclose(
            out_norms.numpy(), x_norms.numpy(), rtol=0.1,
        )

    @patch("teacher_distillation.get_logger")
    def test_ensure_projection_creates_on_mismatch(self, mock_logger):
        """Test _ensure_projection creates projection when dims differ."""
        mock_logger.return_value = MagicMock()
        helper = TeacherDistillationHelper("test-model")
        helper.teacher_dim = 16
        helper._ensure_projection(8)
        self.assertIsNotNone(helper._projection)
        self.assertIsInstance(helper._projection, DimensionProjection)

    @patch("teacher_distillation.get_logger")
    def test_ensure_projection_skips_when_matching(self, mock_logger):
        """Test _ensure_projection does not create when dims match."""
        mock_logger.return_value = MagicMock()
        helper = TeacherDistillationHelper("test-model")
        helper.teacher_dim = 8
        helper._ensure_projection(8)
        self.assertIsNone(helper._projection)

    @patch("teacher_distillation.get_logger")
    @patch("teacher_distillation.SentenceTransformer")
    def test_projection_in_distillation_targets(self, mock_st, mock_logger):
        """Test that projection is used in generate_distillation_targets."""
        mock_logger.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 16
        teacher_embeds = np.random.randn(3, 16).astype(np.float32)
        mock_model.encode.return_value = teacher_embeds
        mock_st.return_value = mock_model

        helper = TeacherDistillationHelper(
            "test-model", projection_enabled=True,
        )
        # Manually set teacher model to skip load_teacher_model's SentenceTransformer call
        helper.teacher_model = mock_model
        helper.teacher_dim = 16
        student_embeds = np.random.randn(3, 8).astype(np.float32)
        targets = helper.generate_distillation_targets(
            texts=["a", "b", "c"],
            current_embeddings=student_embeds,
            teacher_weight=0.5,
        )
        # Should return student dim, not fall back
        self.assertEqual(targets.shape, (3, 8))
        # Projection should have been created
        self.assertIsNotNone(helper._projection)

    @patch("teacher_distillation.get_logger")
    @patch("teacher_distillation.SentenceTransformer")
    def test_projection_disabled_falls_back(self, mock_st, mock_logger):
        """Test that disabled projection falls back to student embeddings."""
        mock_logger.return_value = MagicMock()
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 16
        teacher_embeds = np.random.randn(3, 16).astype(np.float32)
        mock_model.encode.return_value = teacher_embeds
        mock_st.return_value = mock_model

        helper = TeacherDistillationHelper(
            "test-model", projection_enabled=False,
        )
        helper.teacher_model = mock_model
        helper.teacher_dim = 16
        student_embeds = np.random.randn(3, 8).astype(np.float32)
        targets = helper.generate_distillation_targets(
            texts=["a", "b", "c"],
            current_embeddings=student_embeds,
            teacher_weight=0.5,
        )
        # Should fall back to student embeddings
        np.testing.assert_array_equal(targets, student_embeds)

    @patch("teacher_distillation.get_logger")
    def test_projection_in_alignment_metric(self, mock_logger):
        """Test that projection is used in compute_alignment_metric."""
        mock_logger.return_value = MagicMock()

        helper = TeacherDistillationHelper(
            "test-model", projection_enabled=True,
        )
        helper.teacher_dim = 16

        student = np.random.randn(3, 8).astype(np.float32)
        teacher = np.random.randn(3, 16).astype(np.float32)

        alignment = helper.compute_alignment_metric(student, teacher)
        # Projection was created and used
        self.assertIsNotNone(helper._projection)
        self.assertIsInstance(alignment, float)

    def test_projection_with_different_dims(self):
        """Test projection works across various dimension pairs."""
        for t_dim, s_dim in [(32, 16), (16, 32), (64, 8), (8, 64)]:
            proj = DimensionProjection(teacher_dim=t_dim, student_dim=s_dim)
            x = torch.randn(4, t_dim)
            out = proj(x)
            self.assertEqual(out.shape, (4, s_dim))

    def test_projection_gradient_flow(self):
        """Test that gradients flow through the projection."""
        proj = DimensionProjection(teacher_dim=16, student_dim=8)
        x = torch.randn(3, 16, requires_grad=True)
        out = proj(x)
        loss = out.sum()
        loss.backward()
        self.assertIsNotNone(x.grad)
        # Check projection parameters received gradients
        for param in proj.parameters():
            self.assertIsNotNone(param.grad)


class TestEnsembleTeacherHelper(unittest.TestCase):
    """Test EnsembleTeacherHelper multi-teacher support."""

    def setUp(self):
        self.logger_patcher = patch("teacher_distillation.get_logger")
        self.mock_logger = self.logger_patcher.start()
        self.mock_logger.return_value = MagicMock()

    def tearDown(self):
        self.logger_patcher.stop()

    def _make_teacher(self, dim, embeddings):
        """Create a mock TeacherDistillationHelper."""
        teacher = MagicMock(spec=TeacherDistillationHelper)
        teacher.teacher_dim = dim
        teacher.get_teacher_embeddings.return_value = embeddings
        teacher.compute_alignment_metric.return_value = 0.8
        teacher._projection_enabled = True
        return teacher

    def test_ensemble_init_equal_weights(self):
        """Test ensemble initializes with equal weights by default."""
        t1 = self._make_teacher(8, np.zeros((2, 8)))
        t2 = self._make_teacher(8, np.zeros((2, 8)))
        ensemble = EnsembleTeacherHelper([t1, t2])
        self.assertAlmostEqual(ensemble.weights[0], 0.5)
        self.assertAlmostEqual(ensemble.weights[1], 0.5)

    def test_ensemble_init_custom_weights(self):
        """Test ensemble with custom weights."""
        t1 = self._make_teacher(8, np.zeros((2, 8)))
        t2 = self._make_teacher(8, np.zeros((2, 8)))
        ensemble = EnsembleTeacherHelper([t1, t2], weights=[0.3, 0.7])
        self.assertAlmostEqual(ensemble.weights[0], 0.3)
        self.assertAlmostEqual(ensemble.weights[1], 0.7)

    def test_ensemble_weight_normalization(self):
        """Test that weights are normalized to sum to 1."""
        t1 = self._make_teacher(8, np.zeros((2, 8)))
        t2 = self._make_teacher(8, np.zeros((2, 8)))
        ensemble = EnsembleTeacherHelper([t1, t2], weights=[2.0, 8.0])
        self.assertAlmostEqual(sum(ensemble.weights), 1.0)
        self.assertAlmostEqual(ensemble.weights[0], 0.2)
        self.assertAlmostEqual(ensemble.weights[1], 0.8)

    def test_ensemble_get_teacher_embeddings(self):
        """Test weighted average embeddings from ensemble."""
        emb1 = np.ones((3, 8), dtype=np.float32) * 2.0
        emb2 = np.ones((3, 8), dtype=np.float32) * 4.0
        t1 = self._make_teacher(8, emb1)
        t2 = self._make_teacher(8, emb2)
        ensemble = EnsembleTeacherHelper([t1, t2], weights=[0.5, 0.5])
        result = ensemble.get_teacher_embeddings(["a", "b", "c"], target_dim=8)
        # Expected: 0.5 * 2.0 + 0.5 * 4.0 = 3.0
        np.testing.assert_allclose(result, np.ones((3, 8)) * 3.0)

    def test_ensemble_with_dimension_mismatch(self):
        """Test ensemble handles dimension mismatch via projection."""
        emb1 = np.random.randn(3, 16).astype(np.float32)
        emb2 = np.random.randn(3, 8).astype(np.float32)
        t1 = self._make_teacher(16, emb1)
        t2 = self._make_teacher(8, emb2)
        ensemble = EnsembleTeacherHelper([t1, t2])
        result = ensemble.get_teacher_embeddings(["a", "b", "c"], target_dim=8)
        # Should project t1 from 16 -> 8, t2 already 8
        self.assertEqual(result.shape, (3, 8))
        # Projection should have been created for teacher 0
        self.assertIn(0, ensemble._projections)

    def test_ensemble_distillation_targets(self):
        """Test ensemble generate_distillation_targets."""
        emb1 = np.random.randn(3, 8).astype(np.float32)
        emb2 = np.random.randn(3, 8).astype(np.float32)
        t1 = self._make_teacher(8, emb1)
        t2 = self._make_teacher(8, emb2)
        ensemble = EnsembleTeacherHelper([t1, t2])
        student = np.random.randn(3, 8).astype(np.float32)
        targets = ensemble.generate_distillation_targets(
            texts=["a", "b", "c"],
            student_embeddings=student,
            teacher_weight=0.5,
        )
        self.assertEqual(targets.shape, (3, 8))
        # Should be normalized
        norms = np.linalg.norm(targets, axis=1)
        np.testing.assert_allclose(norms, np.ones(3), atol=0.01)

    def test_ensemble_alignment_metric(self):
        """Test ensemble compute_alignment_metric."""
        t1 = self._make_teacher(8, np.zeros((3, 8)))
        t1.compute_alignment_metric.return_value = 0.6
        t2 = self._make_teacher(8, np.zeros((3, 8)))
        t2.compute_alignment_metric.return_value = 0.9
        ensemble = EnsembleTeacherHelper([t1, t2], weights=[0.5, 0.5])
        student = np.random.randn(3, 8).astype(np.float32)
        alignment = ensemble.compute_alignment_metric(student)
        # 0.5 * 0.6 + 0.5 * 0.9 = 0.75
        self.assertAlmostEqual(alignment, 0.75)

    def test_ensemble_single_teacher_fallback(self):
        """Test that factory returns single helper for single teacher_models."""
        helper = create_distillation_helper(
            teacher_models=[("test-model", 1.0)],
        )
        self.assertIsInstance(helper, TeacherDistillationHelper)

    def test_factory_returns_ensemble(self):
        """Test that factory returns ensemble for multiple teacher_models."""
        helper = create_distillation_helper(
            teacher_models=[("model-a", 0.5), ("model-b", 0.5)],
        )
        self.assertIsInstance(helper, EnsembleTeacherHelper)
        self.assertEqual(len(helper.teachers), 2)

    def test_factory_returns_single(self):
        """Test that factory returns single helper for teacher_model_name."""
        helper = create_distillation_helper(
            teacher_model_name="test-model",
        )
        self.assertIsInstance(helper, TeacherDistillationHelper)
        self.assertEqual(helper.teacher_model_name, "test-model")

    def test_ensemble_duck_types_api(self):
        """Test ensemble has the same API methods as TeacherDistillationHelper."""
        t1 = self._make_teacher(8, np.zeros((2, 8)))
        ensemble = EnsembleTeacherHelper([t1])
        self.assertTrue(hasattr(ensemble, "get_teacher_embeddings"))
        self.assertTrue(hasattr(ensemble, "generate_distillation_targets"))
        self.assertTrue(hasattr(ensemble, "compute_alignment_metric"))

    def test_ensemble_with_projections(self):
        """Test ensemble projection caching."""
        emb1 = np.random.randn(2, 16).astype(np.float32)
        t1 = self._make_teacher(16, emb1)
        ensemble = EnsembleTeacherHelper([t1])
        # First call creates projection
        ensemble.get_teacher_embeddings(["a", "b"], target_dim=8)
        self.assertIn(0, ensemble._projections)
        # Second call reuses
        proj_ref = ensemble._projections[0]
        ensemble.get_teacher_embeddings(["a", "b"], target_dim=8)
        self.assertIs(ensemble._projections[0], proj_ref)

    def test_ensemble_preset_diverse(self):
        """Test diverse preset exists in config."""
        from config import ChelationConfig
        preset = ChelationConfig.get_preset("diverse", "ensemble")
        self.assertIn("models", preset)
        self.assertEqual(len(preset["models"]), 2)

    def test_ensemble_preset_multilingual(self):
        """Test multilingual preset exists in config."""
        from config import ChelationConfig
        preset = ChelationConfig.get_preset("multilingual", "ensemble")
        self.assertIn("models", preset)

    def test_ensemble_teacher_weight(self):
        """Test ensemble uses teacher_weight attribute."""
        emb1 = np.ones((2, 4), dtype=np.float32)
        t1 = self._make_teacher(4, emb1)
        ensemble = EnsembleTeacherHelper([t1])
        ensemble.teacher_weight = 0.8
        student = np.zeros((2, 4), dtype=np.float32)
        targets = ensemble.generate_distillation_targets(
            texts=["a", "b"],
            student_embeddings=student,
        )
        # With student=0 and teacher=1, blend = 0.8 * 1 + 0.2 * 0 = 0.8
        # After normalization, norms should be 1
        norms = np.linalg.norm(targets, axis=1)
        np.testing.assert_allclose(norms, np.ones(2), atol=0.01)


class TestBatchEncodingParams(unittest.TestCase):
    """Test batch encoding parameters on TeacherDistillationHelper."""

    def setUp(self):
        self.logger_patcher = patch("teacher_distillation.get_logger")
        self.mock_logger = self.logger_patcher.start()
        self.mock_logger.return_value = MagicMock()

    def tearDown(self):
        self.logger_patcher.stop()

    def test_default_batch_params(self):
        """Test that default batch params match backward-compatible values."""
        helper = TeacherDistillationHelper("test-model")
        self.assertEqual(helper.batch_size, 64)
        self.assertFalse(helper.show_progress)
        self.assertEqual(helper.max_corpus_chunk, 10000)

    def test_custom_batch_size(self):
        """Test setting custom batch_size."""
        helper = TeacherDistillationHelper("test-model", batch_size=128)
        self.assertEqual(helper.batch_size, 128)

    def test_custom_show_progress(self):
        """Test setting show_progress=True."""
        helper = TeacherDistillationHelper("test-model", show_progress=True)
        self.assertTrue(helper.show_progress)

    def test_custom_max_corpus_chunk(self):
        """Test setting custom max_corpus_chunk."""
        helper = TeacherDistillationHelper("test-model", max_corpus_chunk=500)
        self.assertEqual(helper.max_corpus_chunk, 500)

    @patch("teacher_distillation.SentenceTransformer")
    @patch("teacher_distillation.torch")
    def test_eager_load(self, mock_torch, mock_st_class):
        """Test that eager_load=True loads model immediately."""
        mock_torch.cuda.is_available.return_value = False
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st_class.return_value = mock_model

        helper = TeacherDistillationHelper("test-model", eager_load=True)
        self.assertIsNotNone(helper.teacher_model)
        self.assertEqual(helper.teacher_dim, 384)

    @patch("teacher_distillation.SentenceTransformer")
    @patch("teacher_distillation.torch")
    def test_batch_size_forwarded_to_encode(self, mock_torch, mock_st_class):
        """Test that batch_size is forwarded to model.encode()."""
        mock_torch.cuda.is_available.return_value = False
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.random.randn(3, 384)
        mock_st_class.return_value = mock_model

        helper = TeacherDistillationHelper("test-model", batch_size=32)
        helper.get_teacher_embeddings(["a", "b", "c"])

        call_kwargs = mock_model.encode.call_args[1]
        self.assertEqual(call_kwargs["batch_size"], 32)

    @patch("teacher_distillation.SentenceTransformer")
    @patch("teacher_distillation.torch")
    def test_show_progress_forwarded_to_encode(self, mock_torch, mock_st_class):
        """Test that show_progress is forwarded to model.encode()."""
        mock_torch.cuda.is_available.return_value = False
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_model.encode.return_value = np.random.randn(2, 384)
        mock_st_class.return_value = mock_model

        helper = TeacherDistillationHelper("test-model", show_progress=True)
        helper.get_teacher_embeddings(["a", "b"])

        call_kwargs = mock_model.encode.call_args[1]
        self.assertTrue(call_kwargs["show_progress_bar"])

    @patch("teacher_distillation.SentenceTransformer")
    @patch("teacher_distillation.torch")
    def test_chunked_encoding_splits_large_input(self, mock_torch, mock_st_class):
        """Test that large inputs are split into chunks."""
        mock_torch.cuda.is_available.return_value = False
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 4

        def encode_side_effect(texts, **kwargs):
            return np.ones((len(texts), 4))

        mock_model.encode.side_effect = encode_side_effect
        mock_st_class.return_value = mock_model

        helper = TeacherDistillationHelper("test-model", max_corpus_chunk=3)
        texts = [f"text_{i}" for i in range(10)]
        result = helper.get_teacher_embeddings(texts)

        self.assertEqual(result.shape, (10, 4))
        # 10 texts / chunk_size 3 = 4 chunks (3+3+3+1)
        self.assertEqual(mock_model.encode.call_count, 4)

    @patch("teacher_distillation.SentenceTransformer")
    @patch("teacher_distillation.torch")
    def test_chunked_encoding_per_chunk_error(self, mock_torch, mock_st_class):
        """Test per-chunk error handling returns zeros for failed chunks."""
        mock_torch.cuda.is_available.return_value = False
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 4

        call_count = [0]

        def encode_side_effect(texts, **kwargs):
            call_count[0] += 1
            if call_count[0] == 2:
                raise RuntimeError("Simulated chunk failure")
            return np.ones((len(texts), 4))

        mock_model.encode.side_effect = encode_side_effect
        mock_st_class.return_value = mock_model

        helper = TeacherDistillationHelper("test-model", max_corpus_chunk=3)
        texts = [f"text_{i}" for i in range(9)]
        result = helper.get_teacher_embeddings(texts)

        self.assertEqual(result.shape, (9, 4))
        # Chunk 1 (texts 0-2): ones, Chunk 2 (texts 3-5): zeros, Chunk 3 (texts 6-8): ones
        np.testing.assert_array_equal(result[0:3], np.ones((3, 4)))
        np.testing.assert_array_equal(result[3:6], np.zeros((3, 4)))
        np.testing.assert_array_equal(result[6:9], np.ones((3, 4)))

    @patch("teacher_distillation.SentenceTransformer")
    @patch("teacher_distillation.torch")
    def test_single_chunk_fast_path(self, mock_torch, mock_st_class):
        """Test that small inputs use single-chunk fast path."""
        mock_torch.cuda.is_available.return_value = False
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 4
        mock_model.encode.return_value = np.ones((3, 4))
        mock_st_class.return_value = mock_model

        helper = TeacherDistillationHelper("test-model", max_corpus_chunk=10000)
        texts = ["a", "b", "c"]
        result = helper.get_teacher_embeddings(texts)

        self.assertEqual(result.shape, (3, 4))
        self.assertEqual(mock_model.encode.call_count, 1)


class TestParallelEnsembleEncoding(unittest.TestCase):
    """Test parallel encoding in EnsembleTeacherHelper."""

    def setUp(self):
        self.logger_patcher = patch("teacher_distillation.get_logger")
        self.mock_logger = self.logger_patcher.start()
        self.mock_logger.return_value = MagicMock()

    def tearDown(self):
        self.logger_patcher.stop()

    def _make_teacher(self, dim, embeddings):
        """Create a mock TeacherDistillationHelper."""
        teacher = MagicMock(spec=TeacherDistillationHelper)
        teacher.teacher_dim = dim
        teacher.get_teacher_embeddings.return_value = embeddings
        teacher.compute_alignment_metric.return_value = 0.8
        teacher._projection_enabled = True
        return teacher

    def test_parallel_encoding_default_enabled(self):
        """Test that parallel encoding is enabled by default."""
        t1 = self._make_teacher(8, np.zeros((2, 8)))
        ensemble = EnsembleTeacherHelper([t1])
        self.assertTrue(ensemble.parallel_encoding)
        self.assertEqual(ensemble.max_workers, 4)

    def test_parallel_encoding_disabled(self):
        """Test ensemble with parallel_encoding=False."""
        t1 = self._make_teacher(8, np.ones((2, 8)))
        t2 = self._make_teacher(8, np.ones((2, 8)) * 2)
        ensemble = EnsembleTeacherHelper([t1, t2], parallel_encoding=False)
        self.assertFalse(ensemble.parallel_encoding)
        result = ensemble.get_teacher_embeddings(["a", "b"], target_dim=8)
        self.assertEqual(result.shape, (2, 8))

    def test_parallel_encoding_produces_same_result(self):
        """Test that parallel and sequential paths produce same results."""
        emb1 = np.ones((3, 8), dtype=np.float32) * 2.0
        emb2 = np.ones((3, 8), dtype=np.float32) * 4.0
        t1_par = self._make_teacher(8, emb1)
        t2_par = self._make_teacher(8, emb2)
        t1_seq = self._make_teacher(8, emb1)
        t2_seq = self._make_teacher(8, emb2)

        ensemble_par = EnsembleTeacherHelper(
            [t1_par, t2_par], weights=[0.5, 0.5], parallel_encoding=True,
        )
        ensemble_seq = EnsembleTeacherHelper(
            [t1_seq, t2_seq], weights=[0.5, 0.5], parallel_encoding=False,
        )

        result_par = ensemble_par.get_teacher_embeddings(["a", "b", "c"], target_dim=8)
        result_seq = ensemble_seq.get_teacher_embeddings(["a", "b", "c"], target_dim=8)
        np.testing.assert_allclose(result_par, result_seq)

    def test_parallel_custom_max_workers(self):
        """Test custom max_workers setting."""
        t1 = self._make_teacher(8, np.zeros((2, 8)))
        ensemble = EnsembleTeacherHelper([t1], max_workers=2)
        self.assertEqual(ensemble.max_workers, 2)

    def test_parallel_single_teacher_uses_sequential(self):
        """Test that single teacher falls back to sequential even when parallel enabled."""
        emb = np.ones((2, 8), dtype=np.float32)
        t1 = self._make_teacher(8, emb)
        ensemble = EnsembleTeacherHelper([t1], parallel_encoding=True)
        # Single teacher -> parallel path is not triggered (len(teachers) > 1 check)
        result = ensemble.get_teacher_embeddings(["a", "b"], target_dim=8)
        self.assertEqual(result.shape, (2, 8))
        t1.get_teacher_embeddings.assert_called_once()

    def test_parallel_handles_teacher_failure(self):
        """Test that parallel encoding handles individual teacher failures."""
        emb1 = np.ones((2, 8), dtype=np.float32) * 2.0
        t1 = self._make_teacher(8, emb1)
        t2 = self._make_teacher(8, np.array([]))  # Simulates failure
        ensemble = EnsembleTeacherHelper([t1, t2], weights=[0.5, 0.5])
        result = ensemble.get_teacher_embeddings(["a", "b"], target_dim=8)
        # Only t1 contributes: 0.5 * 2.0 = 1.0
        np.testing.assert_allclose(result, np.ones((2, 8)))


class TestFactoryBatchParams(unittest.TestCase):
    """Test that factory function forwards batch encoding params."""

    def setUp(self):
        self.logger_patcher = patch("teacher_distillation.get_logger")
        self.mock_logger = self.logger_patcher.start()
        self.mock_logger.return_value = MagicMock()

    def tearDown(self):
        self.logger_patcher.stop()

    def test_factory_forwards_batch_size(self):
        """Test factory forwards batch_size to single teacher."""
        helper = create_distillation_helper(
            teacher_model_name="test-model", batch_size=128,
        )
        self.assertIsInstance(helper, TeacherDistillationHelper)
        self.assertEqual(helper.batch_size, 128)

    def test_factory_forwards_show_progress(self):
        """Test factory forwards show_progress to single teacher."""
        helper = create_distillation_helper(
            teacher_model_name="test-model", show_progress=True,
        )
        self.assertTrue(helper.show_progress)

    def test_factory_forwards_max_corpus_chunk(self):
        """Test factory forwards max_corpus_chunk to single teacher."""
        helper = create_distillation_helper(
            teacher_model_name="test-model", max_corpus_chunk=500,
        )
        self.assertEqual(helper.max_corpus_chunk, 500)

    def test_factory_forwards_params_to_ensemble(self):
        """Test factory forwards batch params to ensemble teachers."""
        helper = create_distillation_helper(
            teacher_models=[("model-a", 0.5), ("model-b", 0.5)],
            batch_size=256,
            show_progress=True,
            max_corpus_chunk=2000,
            parallel_encoding=False,
            max_workers=2,
        )
        self.assertIsInstance(helper, EnsembleTeacherHelper)
        self.assertFalse(helper.parallel_encoding)
        self.assertEqual(helper.max_workers, 2)
        # Check individual teachers got batch params
        for teacher in helper.teachers:
            self.assertEqual(teacher.batch_size, 256)
            self.assertTrue(teacher.show_progress)
            self.assertEqual(teacher.max_corpus_chunk, 2000)

    def test_factory_default_params_backward_compatible(self):
        """Test factory with no batch params matches original defaults."""
        helper = create_distillation_helper(teacher_model_name="test-model")
        self.assertEqual(helper.batch_size, 64)
        self.assertFalse(helper.show_progress)
        self.assertEqual(helper.max_corpus_chunk, 10000)


class TestTeacherEncodingPresets(unittest.TestCase):
    """Test teacher_encoding preset category in ChelationConfig."""

    def test_default_preset(self):
        """Test default teacher_encoding preset."""
        from config import ChelationConfig
        preset = ChelationConfig.get_preset("default", "teacher_encoding")
        self.assertEqual(preset["batch_size"], 64)
        self.assertFalse(preset["eager_load"])
        self.assertFalse(preset["show_progress"])
        self.assertEqual(preset["max_corpus_chunk"], 10000)
        self.assertTrue(preset["parallel_encoding"])
        self.assertEqual(preset["max_workers"], 4)

    def test_large_corpus_preset(self):
        """Test large_corpus teacher_encoding preset."""
        from config import ChelationConfig
        preset = ChelationConfig.get_preset("large_corpus", "teacher_encoding")
        self.assertEqual(preset["batch_size"], 128)
        self.assertTrue(preset["eager_load"])
        self.assertTrue(preset["show_progress"])
        self.assertEqual(preset["max_corpus_chunk"], 50000)

    def test_memory_constrained_preset(self):
        """Test memory_constrained teacher_encoding preset."""
        from config import ChelationConfig
        preset = ChelationConfig.get_preset("memory_constrained", "teacher_encoding")
        self.assertEqual(preset["batch_size"], 16)
        self.assertFalse(preset["parallel_encoding"])
        self.assertEqual(preset["max_corpus_chunk"], 2000)

    def test_gpu_optimized_preset(self):
        """Test gpu_optimized teacher_encoding preset."""
        from config import ChelationConfig
        preset = ChelationConfig.get_preset("gpu_optimized", "teacher_encoding")
        self.assertEqual(preset["batch_size"], 256)
        self.assertTrue(preset["eager_load"])
        self.assertEqual(preset["max_corpus_chunk"], 100000)
        self.assertEqual(preset["max_workers"], 8)

    def test_invalid_preset_raises(self):
        """Test that invalid preset name raises ValueError."""
        from config import ChelationConfig
        with self.assertRaises(ValueError):
            ChelationConfig.get_preset("nonexistent", "teacher_encoding")

    def test_teacher_encoding_type_registered(self):
        """Test that teacher_encoding is a valid preset_type."""
        from config import ChelationConfig
        # Should not raise
        preset = ChelationConfig.get_preset("default", "teacher_encoding")
        self.assertIn("description", preset)


if __name__ == "__main__":
    unittest.main(verbosity=2)
