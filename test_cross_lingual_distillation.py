"""
Unit Tests for Cross-Lingual Distillation Module

Tests CrossLingualTeacherRouter and LanguageTeacherMapping with mocked
teacher models and language detection. No actual model loading.
"""

import unittest
import numpy as np
from unittest.mock import MagicMock, patch

from cross_lingual_distillation import (
    CrossLingualTeacherRouter,
    LanguageTeacherMapping,
    create_cross_lingual_router,
)
from language_detector import LanguageDetector
from teacher_distillation import TeacherDistillationHelper


class TestLanguageTeacherMapping(unittest.TestCase):
    """Test LanguageTeacherMapping configuration class."""

    def test_basic_mapping(self):
        """Test basic language-to-teacher mapping."""
        mapping = LanguageTeacherMapping(
            mappings={"en": "model-en", "de": "model-de"},
            default_teacher="model-fallback",
        )
        self.assertEqual(mapping.get_teacher_for_language("en"), "model-en")
        self.assertEqual(mapping.get_teacher_for_language("de"), "model-de")

    def test_fallback_to_default(self):
        """Test unmapped languages fall back to default teacher."""
        mapping = LanguageTeacherMapping(
            mappings={"en": "model-en"},
            default_teacher="model-fallback",
        )
        self.assertEqual(mapping.get_teacher_for_language("zh"), "model-fallback")

    def test_empty_mappings(self):
        """Test with no language-specific mappings (all use default)."""
        mapping = LanguageTeacherMapping(
            mappings={},
            default_teacher="universal-model",
        )
        self.assertEqual(mapping.get_teacher_for_language("en"), "universal-model")
        self.assertEqual(mapping.get_teacher_for_language("zh"), "universal-model")

    def test_get_unique_teachers(self):
        """Test listing unique teacher models."""
        mapping = LanguageTeacherMapping(
            mappings={"en": "model-en", "de": "model-de", "fr": "model-en"},
            default_teacher="model-fallback",
        )
        unique = mapping.get_unique_teachers()
        self.assertEqual(len(unique), 3)
        self.assertIn("model-en", unique)
        self.assertIn("model-de", unique)
        self.assertIn("model-fallback", unique)

    def test_has_language(self):
        """Test checking language existence."""
        mapping = LanguageTeacherMapping(
            mappings={"en": "model-en"},
            default_teacher="default",
        )
        self.assertTrue(mapping.has_language("en"))
        self.assertFalse(mapping.has_language("zh"))

    def test_add_mapping(self):
        """Test adding a new language mapping."""
        mapping = LanguageTeacherMapping(
            mappings={}, default_teacher="default",
        )
        mapping.add_mapping("ko", "model-ko")
        self.assertEqual(mapping.get_teacher_for_language("ko"), "model-ko")
        self.assertTrue(mapping.has_language("ko"))

    def test_repr(self):
        """Test string representation."""
        mapping = LanguageTeacherMapping(
            mappings={"en": "model-en"},
            default_teacher="default",
        )
        r = repr(mapping)
        self.assertIn("LanguageTeacherMapping", r)
        self.assertIn("model-en", r)

    def test_default_teacher_from_config(self):
        """Test that default_teacher falls back to config default."""
        mapping = LanguageTeacherMapping(mappings={})
        # Should use ChelationConfig.DEFAULT_TEACHER_MODEL
        from config import ChelationConfig
        self.assertEqual(mapping.default_teacher, ChelationConfig.DEFAULT_TEACHER_MODEL)


class TestCrossLingualTeacherRouter(unittest.TestCase):
    """Test CrossLingualTeacherRouter with mocked teachers and detector."""

    def setUp(self):
        self.logger_patcher = patch("cross_lingual_distillation.get_logger")
        self.mock_logger = self.logger_patcher.start()
        self.mock_logger.return_value = MagicMock()

        self.detector_patcher = patch("language_detector.get_logger")
        self.mock_det_logger = self.detector_patcher.start()
        self.mock_det_logger.return_value = MagicMock()

        # Create a mock detector that returns predictable languages
        self.mock_detector = MagicMock(spec=LanguageDetector)

        # Default mapping: en -> model-en, zh -> model-zh, fallback -> model-multi
        self.mapping = LanguageTeacherMapping(
            mappings={"en": "model-en", "zh": "model-zh"},
            default_teacher="model-multi",
        )

    def tearDown(self):
        self.logger_patcher.stop()
        self.detector_patcher.stop()

    def _make_router(self, **kwargs):
        """Create a router with mocked detector and teacher creation."""
        router = CrossLingualTeacherRouter(
            language_mapping=self.mapping,
            detector=self.mock_detector,
            **kwargs,
        )
        return router

    def _mock_teacher(self, dim, embeddings):
        """Create a mock teacher that returns given embeddings."""
        teacher = MagicMock(spec=TeacherDistillationHelper)
        teacher.teacher_dim = dim
        teacher.get_teacher_embeddings.return_value = embeddings
        teacher._projection_enabled = True
        return teacher

    def test_duck_types_api(self):
        """Test that router has same API as TeacherDistillationHelper."""
        router = self._make_router()
        self.assertTrue(hasattr(router, "get_teacher_embeddings"))
        self.assertTrue(hasattr(router, "generate_distillation_targets"))
        self.assertTrue(hasattr(router, "compute_alignment_metric"))
        self.assertTrue(hasattr(router, "teacher_weight"))

    def test_get_teacher_embeddings_empty(self):
        """Test empty texts returns empty array."""
        router = self._make_router()
        result = router.get_teacher_embeddings([])
        self.assertEqual(len(result), 0)

    def test_get_teacher_embeddings_single_language(self):
        """Test routing all texts to a single teacher."""
        self.mock_detector.detect_batch.return_value = ["en", "en", "en"]

        router = self._make_router()
        # Mock the teacher creation
        en_embs = np.random.randn(3, 8).astype(np.float32)
        mock_teacher = self._mock_teacher(8, en_embs)
        router._teachers["model-en"] = mock_teacher

        result = router.get_teacher_embeddings(
            ["hello", "world", "test"], target_dim=8,
        )
        self.assertEqual(result.shape, (3, 8))
        mock_teacher.get_teacher_embeddings.assert_called_once()

    def test_get_teacher_embeddings_multi_language(self):
        """Test routing texts to different teachers by language."""
        self.mock_detector.detect_batch.return_value = ["en", "zh", "en"]

        router = self._make_router()

        en_embs = np.array([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0]],
                           dtype=np.float32)
        zh_embs = np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32)

        en_teacher = self._mock_teacher(4, en_embs)
        zh_teacher = self._mock_teacher(4, zh_embs)
        router._teachers["model-en"] = en_teacher
        router._teachers["model-zh"] = zh_teacher

        result = router.get_teacher_embeddings(
            ["hello", "你好", "world"], target_dim=4,
        )
        self.assertEqual(result.shape, (3, 4))
        # Index 0 should be en_embs[0], index 1 should be zh_embs[0], index 2 should be en_embs[1]
        np.testing.assert_array_almost_equal(result[0], en_embs[0])
        np.testing.assert_array_almost_equal(result[1], zh_embs[0])
        np.testing.assert_array_almost_equal(result[2], en_embs[1])

    def test_get_teacher_embeddings_fallback_language(self):
        """Test unmapped language routes to default teacher."""
        self.mock_detector.detect_batch.return_value = ["ko"]

        router = self._make_router()
        ko_embs = np.random.randn(1, 8).astype(np.float32)
        mock_teacher = self._mock_teacher(8, ko_embs)
        router._teachers["model-multi"] = mock_teacher

        result = router.get_teacher_embeddings(["안녕하세요"], target_dim=8)
        self.assertEqual(result.shape, (1, 8))
        mock_teacher.get_teacher_embeddings.assert_called_once()

    def test_generate_distillation_targets_weight_zero(self):
        """Test teacher_weight=0 returns student embeddings."""
        router = self._make_router(teacher_weight=0.5)
        student = np.random.randn(3, 8).astype(np.float32)
        targets = router.generate_distillation_targets(
            texts=["a", "b", "c"],
            student_embeddings=student,
            teacher_weight=0.0,
        )
        np.testing.assert_array_equal(targets, student)

    def test_generate_distillation_targets_blended(self):
        """Test blended distillation target generation."""
        self.mock_detector.detect_batch.return_value = ["en", "en"]

        router = self._make_router(teacher_weight=0.5)

        teacher_embs = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=np.float32)
        mock_teacher = self._mock_teacher(4, teacher_embs)
        router._teachers["model-en"] = mock_teacher

        student = np.array([
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float32)

        targets = router.generate_distillation_targets(
            texts=["hello", "world"],
            student_embeddings=student,
            teacher_weight=0.5,
        )
        self.assertEqual(targets.shape, (2, 4))
        # Should be normalized
        norms = np.linalg.norm(targets, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(2), decimal=5)
        # Should not be identical to either student or teacher
        self.assertFalse(np.allclose(targets, student))
        self.assertFalse(np.allclose(targets, teacher_embs))

    def test_generate_distillation_targets_default_weight(self):
        """Test that default teacher_weight from init is used."""
        self.mock_detector.detect_batch.return_value = ["en"]

        router = self._make_router(teacher_weight=0.7)

        teacher_embs = np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32)
        mock_teacher = self._mock_teacher(4, teacher_embs)
        router._teachers["model-en"] = mock_teacher

        student = np.array([[0.0, 0.0, 1.0, 0.0]], dtype=np.float32)

        targets = router.generate_distillation_targets(
            texts=["hello"],
            student_embeddings=student,
            # No teacher_weight passed -> should use 0.7
        )
        self.assertEqual(targets.shape, (1, 4))
        # Expected: 0.3 * [0,0,1,0] + 0.7 * [1,0,0,0] = [0.7, 0, 0.3, 0], normalized
        expected_raw = np.array([[0.7, 0.0, 0.3, 0.0]])
        expected_norm = expected_raw / np.linalg.norm(expected_raw)
        np.testing.assert_array_almost_equal(targets, expected_norm, decimal=5)

    def test_compute_alignment_metric_with_texts(self):
        """Test alignment metric when texts are provided (generates teacher embeds)."""
        self.mock_detector.detect_batch.return_value = ["en", "en"]

        router = self._make_router()

        teacher_embs = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)
        mock_teacher = self._mock_teacher(3, teacher_embs)
        router._teachers["model-en"] = mock_teacher

        # Student same as teacher -> perfect alignment
        student = np.array([
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ], dtype=np.float32)

        alignment = router.compute_alignment_metric(
            student, texts=["hello", "world"],
        )
        self.assertAlmostEqual(alignment, 1.0, places=4)

    def test_compute_alignment_metric_with_teacher_embeds(self):
        """Test alignment metric with pre-computed teacher embeds."""
        router = self._make_router()

        student = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        teacher = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

        alignment = router.compute_alignment_metric(student, teacher_embeds=teacher)
        self.assertAlmostEqual(alignment, 1.0, places=5)

    def test_compute_alignment_metric_orthogonal(self):
        """Test alignment metric with orthogonal embeddings."""
        router = self._make_router()

        student = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        teacher = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float32)

        alignment = router.compute_alignment_metric(student, teacher_embeds=teacher)
        self.assertAlmostEqual(alignment, 0.0, places=5)

    def test_compute_alignment_metric_empty(self):
        """Test alignment metric with empty student embeds."""
        router = self._make_router()
        alignment = router.compute_alignment_metric(np.array([]))
        self.assertEqual(alignment, 0.0)

    def test_compute_alignment_metric_no_texts_no_teacher(self):
        """Test alignment metric with no texts and no teacher embeds."""
        router = self._make_router()
        student = np.random.randn(3, 8).astype(np.float32)
        alignment = router.compute_alignment_metric(student)
        self.assertEqual(alignment, 0.0)

    def test_get_language_stats(self):
        """Test language statistics gathering."""
        self.mock_detector.detect_batch.return_value = ["en", "zh", "en", "zh", "en"]

        router = self._make_router()
        stats = router.get_language_stats(["a", "b", "c", "d", "e"])
        self.assertEqual(stats["en"], 3)
        self.assertEqual(stats["zh"], 2)

    def test_get_active_teachers(self):
        """Test listing active (loaded) teachers."""
        router = self._make_router()
        self.assertEqual(router.get_active_teachers(), [])

        # Manually add a teacher
        router._teachers["model-en"] = MagicMock()
        self.assertEqual(router.get_active_teachers(), ["model-en"])

    def test_lazy_teacher_creation(self):
        """Test that teachers are created lazily on first use."""
        self.mock_detector.detect_batch.return_value = ["en"]

        router = self._make_router()
        self.assertEqual(len(router._teachers), 0)

        # Patch the TeacherDistillationHelper constructor
        with patch("cross_lingual_distillation.TeacherDistillationHelper") as mock_cls:
            mock_instance = MagicMock()
            mock_instance.get_teacher_embeddings.return_value = np.zeros((1, 8))
            mock_cls.return_value = mock_instance

            router.get_teacher_embeddings(["hello"], target_dim=8)
            mock_cls.assert_called_once_with(
                teacher_model_name="model-en",
                projection_enabled=True,
                projection_hidden_dim=None,
            )

    def test_projection_for_dimension_mismatch(self):
        """Test that projection is created for teacher-target dim mismatch."""
        self.mock_detector.detect_batch.return_value = ["en"]

        router = self._make_router(projection_enabled=True)

        # Teacher returns 16-dim embeddings, but target is 8
        teacher_embs = np.random.randn(1, 16).astype(np.float32)
        mock_teacher = self._mock_teacher(16, teacher_embs)
        router._teachers["model-en"] = mock_teacher

        result = router.get_teacher_embeddings(["hello"], target_dim=8)
        self.assertEqual(result.shape, (1, 8))
        # A projection should have been created
        self.assertTrue(len(router._projections) > 0)


class TestCrossLingualRouterFactory(unittest.TestCase):
    """Test create_cross_lingual_router factory function."""

    def setUp(self):
        self.logger_patcher = patch("cross_lingual_distillation.get_logger")
        self.mock_logger = self.logger_patcher.start()
        self.mock_logger.return_value = MagicMock()

        self.detector_patcher = patch("language_detector.get_logger")
        self.mock_det_logger = self.detector_patcher.start()
        self.mock_det_logger.return_value = MagicMock()

    def tearDown(self):
        self.logger_patcher.stop()
        self.detector_patcher.stop()

    def test_factory_with_preset_en_de(self):
        """Test factory using en_de preset."""
        router = create_cross_lingual_router(preset="en_de")
        self.assertIsInstance(router, CrossLingualTeacherRouter)
        self.assertTrue(router.language_mapping.has_language("en"))
        self.assertTrue(router.language_mapping.has_language("de"))

    def test_factory_with_preset_en_zh(self):
        """Test factory using en_zh preset."""
        router = create_cross_lingual_router(preset="en_zh")
        self.assertIsInstance(router, CrossLingualTeacherRouter)
        self.assertTrue(router.language_mapping.has_language("en"))
        self.assertTrue(router.language_mapping.has_language("zh"))

    def test_factory_with_preset_en_ja(self):
        """Test factory using en_ja preset."""
        router = create_cross_lingual_router(preset="en_ja")
        self.assertIsInstance(router, CrossLingualTeacherRouter)
        self.assertTrue(router.language_mapping.has_language("en"))
        self.assertTrue(router.language_mapping.has_language("ja"))

    def test_factory_with_preset_multilingual_universal(self):
        """Test factory using multilingual_universal preset."""
        router = create_cross_lingual_router(preset="multilingual_universal")
        self.assertIsInstance(router, CrossLingualTeacherRouter)
        self.assertEqual(
            router.language_mapping.default_teacher,
            "paraphrase-multilingual-mpnet-base-v2",
        )

    def test_factory_with_preset_multilingual_hybrid(self):
        """Test factory using multilingual_hybrid preset."""
        router = create_cross_lingual_router(preset="multilingual_hybrid")
        self.assertIsInstance(router, CrossLingualTeacherRouter)
        self.assertTrue(router.language_mapping.has_language("en"))
        self.assertEqual(
            router.language_mapping.default_teacher,
            "paraphrase-multilingual-mpnet-base-v2",
        )

    def test_factory_manual_config(self):
        """Test factory with manual configuration."""
        router = create_cross_lingual_router(
            language_mappings={"en": "custom-en", "fr": "custom-fr"},
            default_teacher="custom-default",
            teacher_weight=0.8,
        )
        self.assertIsInstance(router, CrossLingualTeacherRouter)
        self.assertEqual(
            router.language_mapping.get_teacher_for_language("en"), "custom-en",
        )
        self.assertEqual(
            router.language_mapping.get_teacher_for_language("fr"), "custom-fr",
        )
        self.assertEqual(
            router.language_mapping.get_teacher_for_language("zh"), "custom-default",
        )
        self.assertAlmostEqual(router.teacher_weight, 0.8)

    def test_factory_empty_manual_config(self):
        """Test factory with no mappings uses defaults."""
        router = create_cross_lingual_router()
        self.assertIsInstance(router, CrossLingualTeacherRouter)

    def test_invalid_preset_raises(self):
        """Test that invalid preset name raises ValueError."""
        with self.assertRaises(ValueError):
            create_cross_lingual_router(preset="nonexistent_preset")


class TestCrossLingualPresets(unittest.TestCase):
    """Test cross-lingual presets in ChelationConfig."""

    def test_preset_en_de_exists(self):
        """Test en_de preset is accessible."""
        from config import ChelationConfig
        preset = ChelationConfig.get_preset("en_de", "cross_lingual")
        self.assertIn("language_mappings", preset)
        self.assertIn("en", preset["language_mappings"])
        self.assertIn("de", preset["language_mappings"])

    def test_preset_en_zh_exists(self):
        """Test en_zh preset is accessible."""
        from config import ChelationConfig
        preset = ChelationConfig.get_preset("en_zh", "cross_lingual")
        self.assertIn("language_mappings", preset)
        self.assertIn("zh", preset["language_mappings"])

    def test_preset_multilingual_universal_exists(self):
        """Test multilingual_universal preset is accessible."""
        from config import ChelationConfig
        preset = ChelationConfig.get_preset("multilingual_universal", "cross_lingual")
        self.assertIn("default_teacher", preset)

    def test_all_presets_have_required_keys(self):
        """Test all cross-lingual presets have required keys."""
        from config import ChelationConfig
        for name in ["en_de", "en_zh", "en_ja",
                      "multilingual_universal", "multilingual_hybrid"]:
            preset = ChelationConfig.get_preset(name, "cross_lingual")
            self.assertIn("language_mappings", preset)
            self.assertIn("default_teacher", preset)
            self.assertIn("teacher_weight", preset)
            self.assertIn("description", preset)


if __name__ == "__main__":
    unittest.main(verbosity=2)
