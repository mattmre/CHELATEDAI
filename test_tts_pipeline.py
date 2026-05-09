"""Tests for VectorSteerer, TTSConfig, TTSPipeline (≥30 tests)."""
from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from tts_pipeline import (
    SteeringSignal,
    TTSConfig,
    TTSPipeline,
    TTSResult,
    VectorSteerer,
)
from vector_translator import TranslationConfig, VectorTranslator
from vector_transport import TransportConfig, TransportTarget, VectorTransport

_TL_PATCH = "vector_translator.get_logger"
_TP_PATCH = "vector_transport.get_logger"
_TTS_PATCH = "tts_pipeline.get_logger"


def _make_translator(dim: int = 8) -> VectorTranslator:
    with patch(_TL_PATCH, return_value=MagicMock()):
        return VectorTranslator(TranslationConfig(offset_dim=dim))


def _make_transport(**kw) -> VectorTransport:
    with patch(_TP_PATCH, return_value=MagicMock()):
        return VectorTransport(TransportConfig(**kw))


def _make_pipeline(
    dim: int = 8,
    *,
    translation_enabled: bool = True,
    transport_enabled: bool = True,
    steering_enabled: bool = True,
) -> TTSPipeline:
    translator = _make_translator(dim)
    transport = _make_transport()
    steerer = VectorSteerer(max_strength=0.3)
    config = TTSConfig(
        translation_enabled=translation_enabled,
        transport_enabled=transport_enabled,
        steering_enabled=steering_enabled,
    )
    with patch(_TTS_PATCH, return_value=MagicMock()):
        return TTSPipeline(translator, transport, steerer, config)


class TestSteeringSignal(unittest.TestCase):
    def test_construction(self):
        direction = np.array([1.0, 0.0, 0.0])
        sig = SteeringSignal(direction=direction, strength=0.2, source="test")
        np.testing.assert_array_equal(sig.direction, direction)
        self.assertAlmostEqual(sig.strength, 0.2)
        self.assertEqual(sig.source, "test")


class TestVectorSteererBasic(unittest.TestCase):
    def test_default_construction(self):
        s = VectorSteerer()
        self.assertAlmostEqual(s._max_strength, 0.3)
        self.assertTrue(s._enabled)
        self.assertEqual(len(s._signals), 0)

    def test_add_signal_stores(self):
        s = VectorSteerer()
        sig = SteeringSignal(np.array([1.0, 0.0]), 0.1, "src")
        s.add_signal(sig)
        self.assertEqual(len(s._signals), 1)

    def test_add_multiple_signals(self):
        s = VectorSteerer()
        for i in range(3):
            s.add_signal(SteeringSignal(np.array([float(i), 0.0]), 0.1, f"s{i}"))
        self.assertEqual(len(s._signals), 3)

    def test_clear_signals(self):
        s = VectorSteerer()
        s.add_signal(SteeringSignal(np.array([1.0, 0.0]), 0.1, "x"))
        s.clear_signals()
        self.assertEqual(len(s._signals), 0)


class TestVectorSteererSteer(unittest.TestCase):
    def setUp(self):
        self.dim = 8
        self.v = np.zeros(self.dim)

    def test_no_signals_passthrough(self):
        s = VectorSteerer()
        steered, meta = s.steer(self.v)
        np.testing.assert_array_almost_equal(steered, self.v)

    def test_no_signals_was_steered_false(self):
        s = VectorSteerer()
        _, meta = s.steer(self.v)
        self.assertFalse(meta["was_steered"])

    def test_no_signals_delta_norm_zero(self):
        s = VectorSteerer()
        _, meta = s.steer(self.v)
        self.assertAlmostEqual(meta["total_delta_norm"], 0.0)

    def test_no_signals_count_zero(self):
        s = VectorSteerer()
        _, meta = s.steer(self.v)
        self.assertEqual(meta["signals_applied"], 0)

    def test_with_signal_differs_from_original(self):
        s = VectorSteerer()
        direction = np.zeros(self.dim)
        direction[0] = 1.0
        s.add_signal(SteeringSignal(direction, 0.1, "src"))
        steered, _ = s.steer(self.v)
        self.assertFalse(np.allclose(steered, self.v))

    def test_was_steered_true_when_signals_present(self):
        s = VectorSteerer()
        s.add_signal(SteeringSignal(np.ones(self.dim), 0.05, "src"))
        _, meta = s.steer(self.v)
        self.assertTrue(meta["was_steered"])

    def test_signals_applied_count(self):
        s = VectorSteerer()
        for i in range(3):
            d = np.zeros(self.dim)
            d[i] = 1.0
            s.add_signal(SteeringSignal(d, 0.01, f"s{i}"))
        _, meta = s.steer(self.v)
        self.assertEqual(meta["signals_applied"], 3)

    def test_total_delta_clamped_to_max_strength(self):
        s = VectorSteerer(max_strength=0.1)
        # Single large signal
        direction = np.zeros(self.dim)
        direction[0] = 1.0
        s.add_signal(SteeringSignal(direction, 1.0, "big"))
        _, meta = s.steer(self.v)
        self.assertAlmostEqual(meta["total_delta_norm"], 0.1, places=5)

    def test_delta_not_clamped_below_max(self):
        s = VectorSteerer(max_strength=0.5)
        direction = np.zeros(self.dim)
        direction[0] = 1.0
        s.add_signal(SteeringSignal(direction, 0.1, "small"))
        _, meta = s.steer(self.v)
        self.assertLessEqual(meta["total_delta_norm"], 0.5)
        self.assertGreater(meta["total_delta_norm"], 0.0)

    def test_disabled_returns_passthrough(self):
        s = VectorSteerer(enabled=False)
        s.add_signal(SteeringSignal(np.ones(self.dim), 0.5, "src"))
        steered, meta = s.steer(self.v)
        np.testing.assert_array_almost_equal(steered, self.v)
        self.assertFalse(meta["was_steered"])

    def test_total_delta_norm_in_metadata_matches_actual(self):
        s = VectorSteerer(max_strength=1.0)
        direction = np.zeros(self.dim)
        direction[2] = 1.0
        s.add_signal(SteeringSignal(direction, 0.2, "src"))
        steered, meta = s.steer(self.v)
        actual = float(np.linalg.norm(steered - self.v))
        self.assertAlmostEqual(meta["total_delta_norm"], actual, places=5)

    def test_direction_is_normalised_in_delta(self):
        s = VectorSteerer(max_strength=1.0)
        direction = np.zeros(self.dim)
        direction[0] = 5.0  # unnormalised
        s.add_signal(SteeringSignal(direction, 0.1, "src"))
        steered, _ = s.steer(self.v)
        # delta should be 0.1 in dim 0 regardless of unnormalised direction
        self.assertAlmostEqual(steered[0], 0.1, places=5)


class TestVectorSteererFromSparseEvent(unittest.TestCase):
    def _make_event(self, n_features: int = 3, dim: int = 16):
        class Feature:
            def __init__(self, fid: str, val: float):
                self.feature_id = fid
                self.value = val

        class Event:
            pass

        evt = Event()
        evt.features = [Feature(f"f{i}", float(i + 1) * 0.5) for i in range(n_features)]
        evt.dim = dim
        return evt

    def test_returns_vector_steerer(self):
        evt = self._make_event()
        s = VectorSteerer.from_sparse_feature_event(evt)
        self.assertIsInstance(s, VectorSteerer)

    def test_signals_added_per_feature(self):
        evt = self._make_event(n_features=4)
        s = VectorSteerer.from_sparse_feature_event(evt)
        self.assertEqual(len(s._signals), 4)

    def test_no_features_no_signals(self):
        evt = self._make_event(n_features=0)
        s = VectorSteerer.from_sparse_feature_event(evt)
        self.assertEqual(len(s._signals), 0)

    def test_strength_scaled_by_strength_scale(self):
        evt = self._make_event(n_features=1, dim=8)
        evt.features[0].value = 1.0
        s = VectorSteerer.from_sparse_feature_event(evt, strength_scale=0.2)
        # strength = min(1.0 * 0.2, 0.3) = 0.2
        self.assertAlmostEqual(s._signals[0].strength, 0.2, places=5)

    def test_strength_clamped_to_max(self):
        evt = self._make_event(n_features=1, dim=8)
        evt.features[0].value = 100.0
        s = VectorSteerer.from_sparse_feature_event(evt, strength_scale=1.0)
        self.assertLessEqual(s._signals[0].strength, 0.3)


class TestTTSConfig(unittest.TestCase):
    def test_defaults(self):
        c = TTSConfig()
        self.assertTrue(c.translation_enabled)
        self.assertTrue(c.transport_enabled)
        self.assertTrue(c.steering_enabled)
        self.assertTrue(c.record_intermediates)

    def test_custom_values(self):
        c = TTSConfig(
            translation_enabled=False,
            transport_enabled=False,
            steering_enabled=True,
            record_intermediates=False,
        )
        self.assertFalse(c.translation_enabled)
        self.assertFalse(c.transport_enabled)
        self.assertTrue(c.steering_enabled)
        self.assertFalse(c.record_intermediates)


class TestTTSPipelineApply(unittest.TestCase):
    def setUp(self):
        self.dim = 8
        self.v = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    def test_all_disabled_original_unchanged(self):
        p = _make_pipeline(self.dim, translation_enabled=False, transport_enabled=False, steering_enabled=False)
        r = p.apply(self.v)
        np.testing.assert_array_almost_equal(r.after_steering, self.v)
        self.assertEqual(r.stages_applied, [])

    def test_all_disabled_total_delta_norm_zero(self):
        p = _make_pipeline(self.dim, translation_enabled=False, transport_enabled=False, steering_enabled=False)
        r = p.apply(self.v)
        self.assertAlmostEqual(r.total_delta_norm, 0.0)

    def test_result_type(self):
        p = _make_pipeline(self.dim)
        r = p.apply(self.v)
        self.assertIsInstance(r, TTSResult)

    def test_original_preserved(self):
        p = _make_pipeline(self.dim)
        r = p.apply(self.v)
        np.testing.assert_array_equal(r.original, self.v)

    def test_only_translation_fires(self):
        """Only translation enabled, with a learned offset."""
        translator = _make_translator(self.dim)
        translator.set_learned_offset(np.ones(self.dim) * 0.05)
        transport = _make_transport()
        steerer = VectorSteerer()
        cfg = TTSConfig(translation_enabled=True, transport_enabled=False, steering_enabled=False)
        with patch(_TTS_PATCH, return_value=MagicMock()):
            p = TTSPipeline(translator, transport, steerer, cfg)
        r = p.apply(self.v)
        self.assertIn("translation", r.stages_applied)
        self.assertNotIn("transport", r.stages_applied)
        self.assertNotIn("steering", r.stages_applied)
        self.assertFalse(np.allclose(r.after_translation, self.v))

    def test_only_transport_fires(self):
        """Only transport enabled, v far from target centroid."""
        translator = _make_translator(self.dim)
        transport = _make_transport(min_similarity_for_transport=1.0, default_weight=0.2)
        tc_centroid = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        transport.add_target(TransportTarget("t1", tc_centroid))
        steerer = VectorSteerer()
        cfg = TTSConfig(translation_enabled=False, transport_enabled=True, steering_enabled=False)
        with patch(_TTS_PATCH, return_value=MagicMock()):
            p = TTSPipeline(translator, transport, steerer, cfg)
        r = p.apply(self.v)
        self.assertIn("transport", r.stages_applied)
        self.assertNotIn("translation", r.stages_applied)
        self.assertNotIn("steering", r.stages_applied)

    def test_only_steering_fires(self):
        translator = _make_translator(self.dim)
        transport = _make_transport()
        steerer = VectorSteerer(max_strength=0.3)
        direction = np.zeros(self.dim)
        direction[1] = 1.0
        steerer.add_signal(SteeringSignal(direction, 0.1, "src"))
        cfg = TTSConfig(translation_enabled=False, transport_enabled=False, steering_enabled=True)
        with patch(_TTS_PATCH, return_value=MagicMock()):
            p = TTSPipeline(translator, transport, steerer, cfg)
        r = p.apply(self.v)
        self.assertIn("steering", r.stages_applied)
        self.assertNotIn("translation", r.stages_applied)
        self.assertNotIn("transport", r.stages_applied)

    def test_full_pipeline_three_stages(self):
        """All three stages must fire."""
        translator = _make_translator(self.dim)
        translator.set_learned_offset(np.ones(self.dim) * 0.05)

        # Transport: v and target orthogonal → sim=0 < 0.3
        transport = _make_transport(min_similarity_for_transport=1.0, default_weight=0.2)
        tc_centroid = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        transport.add_target(TransportTarget("t1", tc_centroid))

        steerer = VectorSteerer(max_strength=0.3)
        direction = np.zeros(self.dim)
        direction[2] = 1.0
        steerer.add_signal(SteeringSignal(direction, 0.1, "src"))

        cfg = TTSConfig()
        with patch(_TTS_PATCH, return_value=MagicMock()):
            p = TTSPipeline(translator, transport, steerer, cfg)
        r = p.apply(self.v)
        self.assertIn("translation", r.stages_applied)
        self.assertIn("transport", r.stages_applied)
        self.assertIn("steering", r.stages_applied)
        self.assertEqual(len(r.stages_applied), 3)

    def test_stages_applied_only_modified_stages(self):
        """Passthrough stages must not appear in stages_applied."""
        # No offset, no targets, no signals → all passthrough
        p = _make_pipeline(self.dim)
        r = p.apply(self.v)
        self.assertEqual(r.stages_applied, [])

    def test_total_delta_norm_matches_l2(self):
        translator = _make_translator(self.dim)
        translator.set_learned_offset(np.ones(self.dim) * 0.1)
        transport = _make_transport()
        steerer = VectorSteerer()
        cfg = TTSConfig()
        with patch(_TTS_PATCH, return_value=MagicMock()):
            p = TTSPipeline(translator, transport, steerer, cfg)
        r = p.apply(self.v)
        expected = float(np.linalg.norm(r.after_steering - r.original))
        self.assertAlmostEqual(r.total_delta_norm, expected, places=6)

    def test_after_translation_in_result(self):
        p = _make_pipeline(self.dim)
        r = p.apply(self.v)
        self.assertIsNotNone(r.after_translation)
        self.assertEqual(r.after_translation.shape, self.v.shape)

    def test_after_transport_in_result(self):
        p = _make_pipeline(self.dim)
        r = p.apply(self.v)
        self.assertIsNotNone(r.after_transport)
        self.assertEqual(r.after_transport.shape, self.v.shape)

    def test_record_intermediates_false_still_stores(self):
        """record_intermediates=False: intermediates are still stored in TTSResult."""
        translator = _make_translator(self.dim)
        transport = _make_transport()
        steerer = VectorSteerer()
        cfg = TTSConfig(record_intermediates=False)
        with patch(_TTS_PATCH, return_value=MagicMock()):
            p = TTSPipeline(translator, transport, steerer, cfg)
        r = p.apply(self.v)
        self.assertIsNotNone(r.after_translation)
        self.assertIsNotNone(r.after_transport)

    def test_apply_with_feature_event(self):
        class Feature:
            feature_id = "f1"
            value = 1.0

        class Event:
            features = [Feature()]
            dim = 8

        p = _make_pipeline(self.dim)
        r = p.apply(self.v, feature_event=Event())
        self.assertIsInstance(r, TTSResult)

    def test_translation_result_none_when_disabled(self):
        p = _make_pipeline(self.dim, translation_enabled=False)
        r = p.apply(self.v)
        self.assertIsNone(r.translation_result)

    def test_transport_result_none_when_disabled(self):
        p = _make_pipeline(self.dim, transport_enabled=False)
        r = p.apply(self.v)
        self.assertIsNone(r.transport_result)

    def test_steering_meta_none_when_disabled(self):
        p = _make_pipeline(self.dim, steering_enabled=False)
        r = p.apply(self.v)
        self.assertIsNone(r.steering_meta)


class TestTTSPipelineBuildDefault(unittest.TestCase):
    def test_build_default_returns_pipeline(self):
        with patch(_TTS_PATCH, return_value=MagicMock()):
            with patch(_TL_PATCH, return_value=MagicMock()):
                with patch(_TP_PATCH, return_value=MagicMock()):
                    p = TTSPipeline.build_default(dim=8)
        self.assertIsInstance(p, TTSPipeline)

    def test_build_default_passthrough_no_offset(self):
        with patch(_TTS_PATCH, return_value=MagicMock()):
            with patch(_TL_PATCH, return_value=MagicMock()):
                with patch(_TP_PATCH, return_value=MagicMock()):
                    p = TTSPipeline.build_default(dim=8)
        v = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        r = p.apply(v)
        # No offset, no targets, no signals → all passthrough
        self.assertEqual(r.stages_applied, [])
        np.testing.assert_array_almost_equal(r.after_steering, v)


class TestTTSPipelineDiagnostics(unittest.TestCase):
    def test_get_diagnostics_returns_dict(self):
        p = _make_pipeline()
        diag = p.get_diagnostics()
        self.assertIsInstance(diag, dict)

    def test_get_diagnostics_expected_keys(self):
        p = _make_pipeline()
        diag = p.get_diagnostics()
        for key in (
            "translation_enabled",
            "transport_enabled",
            "steering_enabled",
            "record_intermediates",
            "translator_has_offset",
            "translator_cluster_count",
            "transport_target_count",
            "steerer_signal_count",
            "steerer_max_strength",
        ):
            self.assertIn(key, diag, f"Missing key: {key}")

    def test_diagnostics_reflect_state(self):
        translator = _make_translator(8)
        translator.set_learned_offset(np.ones(8) * 0.1)
        transport = _make_transport()
        transport.add_target(TransportTarget("t1", np.ones(8)))
        steerer = VectorSteerer()
        cfg = TTSConfig()
        with patch(_TTS_PATCH, return_value=MagicMock()):
            p = TTSPipeline(translator, transport, steerer, cfg)
        diag = p.get_diagnostics()
        self.assertTrue(diag["translator_has_offset"])
        self.assertEqual(diag["transport_target_count"], 1)


if __name__ == "__main__":
    unittest.main()
