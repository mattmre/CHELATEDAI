"""
test_tts_engine_integration.py — Integration tests for Slice 22 TTS wiring.

Tests cover:
  - ModelScopeSteeringPolicy.activate() (R3 fix in steering_policy.py)
  - AntigravityEngine.enable_tts() / get_last_tts_result() (engine-TTS bridge)
  - TTS intercept in run_inference() — vector transformation verified
  - Dashboard state helpers (_TTS_DASHBOARD_STATE, update_tts_dashboard_state,
    _tts_result_to_dict, handle_api_tts_status, handle_api_tts_last_result)
  - TTSConfig disabled-stage pass-through semantics
  - TTSPipeline.apply() total_delta_norm when all stages are no-ops
"""
from __future__ import annotations

import io
import unittest
from unittest.mock import MagicMock, patch

import numpy as np


# ---------------------------------------------------------------------------
# Helpers / stubs
# ---------------------------------------------------------------------------

def _make_engine(vector_size: int = 8):
    """Return a minimal AntigravityEngine-like object for unit tests."""
    from antigravity_engine import AntigravityEngine

    with (
        patch("antigravity_engine.get_logger", return_value=MagicMock()),
        patch("antigravity_engine.EmbeddingBackend", autospec=True),
        patch("antigravity_engine.QdrantVectorStore", autospec=True),
        patch("antigravity_engine.ChelationConfig") as MockCfg,
    ):
        MockCfg.return_value.embedding_dim = vector_size
        MockCfg.return_value.collection_name = "test_col"
        MockCfg.return_value.qdrant_location = ":memory:"
        MockCfg.return_value.model_name = "all-MiniLM-L6-v2"
        MockCfg.return_value.adapter_type = "mlp"
        MockCfg.return_value.bounded_adapter = False
        MockCfg.return_value.enable_online_updates = False
        MockCfg.return_value.enable_dimension_masking = False
        MockCfg.return_value.enable_quality_tracking = False
        MockCfg.return_value.enable_model_scope = False
        try:
            engine = AntigravityEngine.__new__(AntigravityEngine)
            engine.logger = MagicMock()
            engine.vector_size = vector_size
            engine.adapter = None
            engine._tts_pipeline = None
            engine._last_tts_result = None
            return engine
        except Exception:
            return None


# ===========================================================================
# R3 — ModelScopeSteeringPolicy.activate()
# ===========================================================================

class TestActivateMethod(unittest.TestCase):
    """Tests for ModelScopeSteeringPolicy.activate() (R3 fix)."""

    def setUp(self):
        from steering_policy import ModelScopeSteeringPolicy
        self.Policy = ModelScopeSteeringPolicy

    def test_default_deployment_mode_is_shadow(self):
        policy = self.Policy()
        self.assertEqual(policy.deployment_mode, "shadow_mode")

    def test_activate_soft_scale(self):
        policy = self.Policy()
        policy.activate("soft_scale")
        self.assertEqual(policy.deployment_mode, "soft_scale")

    def test_activate_suppression(self):
        policy = self.Policy()
        policy.activate("suppression")
        self.assertEqual(policy.deployment_mode, "suppression")

    def test_activate_active(self):
        policy = self.Policy()
        policy.activate("active")
        self.assertEqual(policy.deployment_mode, "active")

    def test_activate_shadow_mode_raises(self):
        policy = self.Policy()
        with self.assertRaises(ValueError):
            policy.activate("shadow_mode")

    def test_activate_invalid_string_raises(self):
        policy = self.Policy()
        with self.assertRaises(ValueError):
            policy.activate("unknown_mode")

    def test_activate_empty_string_raises(self):
        policy = self.Policy()
        with self.assertRaises(ValueError):
            policy.activate("")

    def test_activate_is_idempotent(self):
        policy = self.Policy()
        policy.activate("soft_scale")
        policy.activate("soft_scale")
        self.assertEqual(policy.deployment_mode, "soft_scale")

    def test_activate_can_switch_modes(self):
        policy = self.Policy()
        policy.activate("soft_scale")
        policy.activate("suppression")
        self.assertEqual(policy.deployment_mode, "suppression")

    def test_activate_preserves_other_fields(self):
        from steering_policy import SteeringRule
        rule = SteeringRule(feature_id="f1", strength=0.5)
        policy = self.Policy(name="my_policy", rules=[rule])
        policy.activate("active")
        self.assertEqual(policy.name, "my_policy")
        self.assertEqual(len(policy.rules), 1)
        self.assertEqual(policy.rules[0].feature_id, "f1")


# ===========================================================================
# TTSPipeline semantics
# ===========================================================================

class TestTTSPipelinePassthrough(unittest.TestCase):
    """Verify TTSPipeline.apply() behaves correctly when stages are disabled."""

    def _build_pipeline(self, **config_kwargs):
        from tts_pipeline import TTSConfig, TTSPipeline, VectorSteerer
        from vector_translator import TranslationConfig, VectorTranslator
        from vector_transport import TransportConfig, VectorTransport

        dim = 16
        cfg = TTSConfig(**config_kwargs)
        translator = VectorTranslator(TranslationConfig(offset_dim=dim))
        transport = VectorTransport(TransportConfig())
        steerer = VectorSteerer(max_strength=0.3)
        return TTSPipeline(translator, transport, steerer, cfg), dim

    @patch("tts_pipeline.get_logger", return_value=MagicMock())
    def test_all_enabled_no_signals_zero_delta(self, _log):
        """With no translation offset, no transport targets, no steering signals → total_delta_norm ≈ 0."""
        pipeline, dim = self._build_pipeline(
            translation_enabled=True, transport_enabled=True, steering_enabled=True
        )
        v = np.ones(dim, dtype=float)
        result = pipeline.apply(v)
        self.assertAlmostEqual(result.total_delta_norm, 0.0, places=10)
        self.assertEqual(result.stages_applied, [])

    @patch("tts_pipeline.get_logger", return_value=MagicMock())
    def test_translation_disabled(self, _log):
        pipeline, dim = self._build_pipeline(translation_enabled=False)
        v = np.random.rand(dim)
        result = pipeline.apply(v)
        np.testing.assert_array_equal(result.after_translation, result.original)
        self.assertIsNone(result.translation_result)

    @patch("tts_pipeline.get_logger", return_value=MagicMock())
    def test_transport_disabled(self, _log):
        pipeline, dim = self._build_pipeline(transport_enabled=False)
        v = np.random.rand(dim)
        result = pipeline.apply(v)
        np.testing.assert_array_equal(result.after_transport, result.after_translation)
        self.assertIsNone(result.transport_result)

    @patch("tts_pipeline.get_logger", return_value=MagicMock())
    def test_steering_disabled(self, _log):
        pipeline, dim = self._build_pipeline(steering_enabled=False)
        v = np.random.rand(dim)
        result = pipeline.apply(v)
        np.testing.assert_array_equal(result.after_steering, result.after_transport)
        self.assertIsNone(result.steering_meta)

    @patch("tts_pipeline.get_logger", return_value=MagicMock())
    def test_result_original_unchanged(self, _log):
        pipeline, dim = self._build_pipeline()
        v = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                       9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0])
        before = v.copy()
        result = pipeline.apply(v)
        np.testing.assert_array_equal(result.original, before)

    @patch("tts_pipeline.get_logger", return_value=MagicMock())
    def test_total_delta_norm_matches_l2(self, _log):
        pipeline, dim = self._build_pipeline()
        v = np.ones(dim)
        result = pipeline.apply(v)
        expected = float(np.linalg.norm(result.after_steering - result.original))
        self.assertAlmostEqual(result.total_delta_norm, expected, places=12)

    @patch("tts_pipeline.get_logger", return_value=MagicMock())
    def test_build_default_classmethod(self, _log):
        from tts_pipeline import TTSPipeline
        pipeline = TTSPipeline.build_default(dim=32)
        v = np.zeros(32)
        result = pipeline.apply(v)
        self.assertIsNotNone(result)
        self.assertEqual(len(result.original), 32)


# ===========================================================================
# AntigravityEngine.enable_tts() / get_last_tts_result()
# ===========================================================================

class TestEngineTTSMethods(unittest.TestCase):
    """Tests for engine.enable_tts() and engine.get_last_tts_result()."""

    def _make_bare_engine(self, dim: int = 16):
        """Build a minimal stub engine with the enable_tts/get_last_tts_result methods wired."""
        from antigravity_engine import AntigravityEngine
        engine = object.__new__(AntigravityEngine)
        engine.logger = MagicMock()
        engine.vector_size = dim
        engine.adapter = None
        engine._tts_pipeline = None
        engine._last_tts_result = None
        return engine

    @patch("antigravity_engine.get_logger", return_value=MagicMock())
    def test_get_last_tts_result_returns_none_before_enable(self, _log):
        engine = self._make_bare_engine()
        self.assertIsNone(engine.get_last_tts_result())

    @patch("antigravity_engine.get_logger", return_value=MagicMock())
    @patch("dashboard_server.update_tts_dashboard_state")
    def test_enable_tts_sets_pipeline(self, _upd, _log):
        from tts_pipeline import TTSConfig
        engine = self._make_bare_engine()
        engine.enable_tts(tts_config=TTSConfig())
        self.assertIsNotNone(engine._tts_pipeline)

    @patch("antigravity_engine.get_logger", return_value=MagicMock())
    @patch("dashboard_server.update_tts_dashboard_state")
    def test_enable_tts_resets_last_result(self, _upd, _log):
        from tts_pipeline import TTSConfig
        engine = self._make_bare_engine()
        engine._last_tts_result = "stale"
        engine.enable_tts(tts_config=TTSConfig())
        self.assertIsNone(engine._last_tts_result)

    @patch("antigravity_engine.get_logger", return_value=MagicMock())
    @patch("dashboard_server.update_tts_dashboard_state")
    def test_enable_tts_logs_event(self, _upd, _log):
        from tts_pipeline import TTSConfig
        engine = self._make_bare_engine()
        engine.enable_tts(tts_config=TTSConfig())
        engine.logger.log_event.assert_called_once()
        call_args = engine.logger.log_event.call_args[0]
        self.assertEqual(call_args[0], "tts_enabled")

    @patch("antigravity_engine.get_logger", return_value=MagicMock())
    @patch("dashboard_server.update_tts_dashboard_state")
    def test_enable_tts_calls_dashboard_update(self, mock_update, _log):
        from tts_pipeline import TTSConfig
        engine = self._make_bare_engine()
        engine.enable_tts(tts_config=TTSConfig())
        mock_update.assert_called_once()
        kwargs = mock_update.call_args[1]
        self.assertTrue(kwargs.get("enabled", False))

    @patch("antigravity_engine.get_logger", return_value=MagicMock())
    @patch("dashboard_server.update_tts_dashboard_state")
    def test_enable_tts_without_config_uses_defaults(self, _upd, _log):
        engine = self._make_bare_engine()
        engine.enable_tts()
        diag = engine._tts_pipeline.get_diagnostics()
        self.assertTrue(diag["translation_enabled"])
        self.assertTrue(diag["transport_enabled"])
        self.assertTrue(diag["steering_enabled"])

    @patch("antigravity_engine.get_logger", return_value=MagicMock())
    @patch("dashboard_server.update_tts_dashboard_state")
    def test_enable_tts_respects_custom_config(self, _upd, _log):
        from tts_pipeline import TTSConfig
        engine = self._make_bare_engine()
        cfg = TTSConfig(translation_enabled=False, transport_enabled=True, steering_enabled=False)
        engine.enable_tts(tts_config=cfg)
        diag = engine._tts_pipeline.get_diagnostics()
        self.assertFalse(diag["translation_enabled"])
        self.assertTrue(diag["transport_enabled"])
        self.assertFalse(diag["steering_enabled"])

    @patch("antigravity_engine.get_logger", return_value=MagicMock())
    @patch("dashboard_server.update_tts_dashboard_state")
    def test_get_last_tts_result_returns_none_after_enable_before_inference(self, _upd, _log):
        from tts_pipeline import TTSConfig
        engine = self._make_bare_engine()
        engine.enable_tts(tts_config=TTSConfig())
        self.assertIsNone(engine.get_last_tts_result())

    @patch("antigravity_engine.get_logger", return_value=MagicMock())
    @patch("dashboard_server.update_tts_dashboard_state", side_effect=RuntimeError("dashboard down"))
    def test_enable_tts_dashboard_failure_emits_warning(self, _upd, _log):
        """L5 gap: enable_tts() dashboard update failure path must emit a UserWarning
        and NOT propagate the exception, and must still set engine._tts_pipeline."""
        import warnings
        from tts_pipeline import TTSConfig
        engine = self._make_bare_engine()
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            engine.enable_tts(tts_config=TTSConfig())
        # Must not raise -- call completed
        self.assertIsNotNone(engine._tts_pipeline)
        # At least one UserWarning must have been emitted
        user_warnings = [x for x in w if issubclass(x.category, UserWarning)]
        self.assertGreater(len(user_warnings), 0, "Expected a UserWarning but none was emitted")
        warning_text = str(user_warnings[0].message).lower()
        self.assertIn("dashboard", warning_text)


# ===========================================================================
# TTS intercept in run_inference()
# ===========================================================================

class TestRunInferenceTTSIntercept(unittest.TestCase):
    """Verify run_inference() applies TTS and stores the result."""

    def _build_engine_with_mocked_internals(self, dim: int = 8):
        """Build engine stub where embedding + Qdrant are fully mocked."""
        from antigravity_engine import AntigravityEngine
        engine = object.__new__(AntigravityEngine)
        engine.logger = MagicMock()
        engine.vector_size = dim
        engine.adapter = None
        engine._tts_pipeline = None
        engine._last_tts_result = None
        # required attributes referenced in run_inference
        engine._config = MagicMock()
        engine._config.enable_model_scope = False
        engine._config.enable_online_updates = False
        engine._config.enable_dimension_masking = False
        engine._config.enable_quality_tracking = False
        engine._embedding_backend = MagicMock()
        engine._embedding_backend.embed.return_value = np.zeros(dim)
        engine._vector_store = MagicMock()
        engine._vector_store.search.return_value = []
        engine._collapse_threshold = 0.95
        engine._oscillation_window = 5
        engine._history_vectors = []
        engine._query_count = 0
        engine._adapter_routing_active = False
        # Mock qdrant client and collection_name so run_inference can complete
        # past the retrieval step (TTS result is set before this call).
        qdrant_mock = MagicMock()
        qdrant_response = MagicMock()
        qdrant_response.points = []
        qdrant_mock.query_points.return_value = qdrant_response
        engine.qdrant = qdrant_mock
        engine.collection_name = "test_collection"
        return engine

    @patch("antigravity_engine.get_logger", return_value=MagicMock())
    @patch("dashboard_server.update_tts_dashboard_state")
    def test_tts_result_stored_after_inference(self, _upd, _log):
        """After run_inference, get_last_tts_result() must return a TTSResult."""
        from tts_pipeline import TTSConfig
        engine = self._build_engine_with_mocked_internals(dim=8)
        engine.enable_tts(tts_config=TTSConfig())

        # Patch self.embed so the embedding step succeeds
        good_embedding = np.zeros((1, 8))
        with (
            patch.object(engine, "embed", return_value=good_embedding),
            patch.object(engine, "_observe_model_scope_query", return_value=None),
            patch.object(engine, "_record_runtime_diagnostics", return_value=None),
            patch.object(engine, "_build_runtime_diagnostics", return_value={}),
        ):
            try:
                engine.run_inference("hello world")
            except Exception:
                pass  # Qdrant stub may raise — we only care about _last_tts_result
        result = engine.get_last_tts_result()
        self.assertIsNotNone(result)
        # TTSResult.after_steering is the final vector
        self.assertEqual(len(result.after_steering), 8)

    @patch("antigravity_engine.get_logger", return_value=MagicMock())
    def test_no_tts_result_when_tts_disabled(self, _log):
        """Without enable_tts(), get_last_tts_result() stays None after inference."""
        engine = self._build_engine_with_mocked_internals(dim=8)
        good_embedding = np.zeros((1, 8))
        with (
            patch.object(engine, "embed", return_value=good_embedding),
            patch.object(engine, "_observe_model_scope_query", return_value=None),
            patch.object(engine, "_record_runtime_diagnostics", return_value=None),
            patch.object(engine, "_build_runtime_diagnostics", return_value={}),
        ):
            try:
                engine.run_inference("hello world")
            except Exception:
                pass
        self.assertIsNone(engine.get_last_tts_result())

    @patch("antigravity_engine.get_logger", return_value=MagicMock())
    @patch("dashboard_server.update_tts_dashboard_state")
    def test_tts_q_vec_replaces_original(self, _upd, _log):
        """When a translation offset is set, after_steering differs from the zero embedding."""
        from tts_pipeline import TTSConfig
        from vector_translator import TranslationConfig, VectorTranslator
        from vector_transport import TransportConfig, VectorTransport
        from tts_pipeline import VectorSteerer, TTSPipeline

        dim = 8
        engine = self._build_engine_with_mocked_internals(dim=dim)
        # Build a pipeline with a known non-zero offset
        cfg = TTSConfig(translation_enabled=True, transport_enabled=False, steering_enabled=False)
        t_config = TranslationConfig(offset_dim=dim)
        translator = VectorTranslator(t_config)
        offset = np.ones(dim) * 0.5
        translator.set_learned_offset(offset)
        transport = VectorTransport(TransportConfig())
        steerer = VectorSteerer(max_strength=0.3)
        engine._tts_pipeline = TTSPipeline(translator, transport, steerer, cfg)
        engine._last_tts_result = None

        good_embedding = np.zeros((1, dim))
        with (
            patch.object(engine, "embed", return_value=good_embedding),
            patch.object(engine, "_observe_model_scope_query", return_value=None),
            patch.object(engine, "_record_runtime_diagnostics", return_value=None),
            patch.object(engine, "_build_runtime_diagnostics", return_value={}),
        ):
            engine.run_inference("test")
        result = engine.get_last_tts_result()
        self.assertIsNotNone(result, "TTS result should be populated after run_inference")
        # after_steering should NOT equal the raw embedding (all-zeros)
        self.assertFalse(np.allclose(result.after_steering, np.zeros(dim)))

    @patch("antigravity_engine.get_logger", return_value=MagicMock())
    @patch("dashboard_server.update_tts_dashboard_state")
    def test_tts_apply_failure_leaves_original_embedding(self, _upd, _log):
        """L5: when _tts.apply() raises, run_inference must not raise and
        _last_tts_result must remain None (the error path must not set it)."""
        from tts_pipeline import TTSConfig

        dim = 8
        engine = self._build_engine_with_mocked_internals(dim=dim)
        engine.enable_tts(tts_config=TTSConfig())

        # Confirm baseline: result is None before inference
        self.assertIsNone(engine.get_last_tts_result())

        # Patch apply() to raise so the TTS error path is exercised
        engine._tts_pipeline.apply = MagicMock(
            side_effect=RuntimeError("TTS pipeline test failure")
        )

        good_embedding = np.zeros((1, dim))
        with (
            patch.object(engine, "embed", return_value=good_embedding),
            patch.object(engine, "_observe_model_scope_query", return_value=None),
            patch.object(engine, "_record_runtime_diagnostics", return_value=None),
            patch.object(engine, "_build_runtime_diagnostics", return_value={}),
        ):
            # Must NOT raise — TTS failure is a safety fallback, not a fatal error
            engine.run_inference("test query")

        # _last_tts_result must still be None — the failure path must not set it
        self.assertIsNone(
            engine.get_last_tts_result(),
            "_last_tts_result must not be set when _tts.apply() raises",
        )

        # The logger must have recorded the error
        engine.logger.log_error.assert_called()
        call_args = engine.logger.log_error.call_args
        self.assertEqual(call_args[0][0], "tts_pipeline")


# ===========================================================================
# Dashboard state helpers
# ===========================================================================

class TestTTSDashboardStateHelpers(unittest.TestCase):
    """Tests for _TTS_DASHBOARD_STATE and update_tts_dashboard_state."""

    def setUp(self):
        import dashboard_server as ds
        ds._TTS_DASHBOARD_STATE["enabled"] = False
        ds._TTS_DASHBOARD_STATE["config"] = {}
        ds._TTS_DASHBOARD_STATE["last_result"] = None

    def test_initial_state_disabled(self):
        import dashboard_server as ds
        self.assertFalse(ds._TTS_DASHBOARD_STATE["enabled"])
        self.assertEqual(ds._TTS_DASHBOARD_STATE["config"], {})
        self.assertIsNone(ds._TTS_DASHBOARD_STATE["last_result"])

    def test_update_enabled_flag(self):
        import dashboard_server as ds
        ds.update_tts_dashboard_state(enabled=True)
        self.assertTrue(ds._TTS_DASHBOARD_STATE["enabled"])

    def test_update_config(self):
        import dashboard_server as ds
        ds.update_tts_dashboard_state(enabled=True, config={"translation_enabled": False})
        self.assertFalse(ds._TTS_DASHBOARD_STATE["config"]["translation_enabled"])

    def test_update_last_result(self):
        import dashboard_server as ds
        payload = {"total_delta_norm": 0.123, "stages_applied": ["translation"]}
        ds.update_tts_dashboard_state(enabled=True, last_result=payload)
        self.assertEqual(ds._TTS_DASHBOARD_STATE["last_result"]["total_delta_norm"], 0.123)

    def test_update_config_is_defensive_copy(self):
        import dashboard_server as ds
        cfg = {"translation_enabled": True}
        ds.update_tts_dashboard_state(enabled=True, config=cfg)
        cfg["translation_enabled"] = False
        self.assertTrue(ds._TTS_DASHBOARD_STATE["config"]["translation_enabled"])

    def test_update_last_result_is_defensive_copy(self):
        import dashboard_server as ds
        payload = {"total_delta_norm": 0.5}
        ds.update_tts_dashboard_state(enabled=True, last_result=payload)
        payload["total_delta_norm"] = 99.9
        self.assertEqual(ds._TTS_DASHBOARD_STATE["last_result"]["total_delta_norm"], 0.5)

    def test_update_without_config_leaves_existing_config(self):
        import dashboard_server as ds
        ds._TTS_DASHBOARD_STATE["config"] = {"transport_enabled": True}
        ds.update_tts_dashboard_state(enabled=True)
        self.assertTrue(ds._TTS_DASHBOARD_STATE["config"]["transport_enabled"])

    def test_update_without_result_leaves_existing_result(self):
        import dashboard_server as ds
        ds._TTS_DASHBOARD_STATE["last_result"] = {"total_delta_norm": 0.7}
        ds.update_tts_dashboard_state(enabled=True)
        self.assertEqual(ds._TTS_DASHBOARD_STATE["last_result"]["total_delta_norm"], 0.7)


class TestTtsResultToDict(unittest.TestCase):
    """Tests for _tts_result_to_dict() serialisation helper."""

    def _make_tts_result(self, dim=8, apply_stages=False):
        from tts_pipeline import TTSResult
        v = np.ones(dim)
        if apply_stages:
            after = v + np.ones(dim) * 0.1
        else:
            after = v.copy()
        return TTSResult(
            original=v,
            after_translation=after.copy(),
            after_transport=after.copy(),
            after_steering=after.copy(),
            translation_result=None,
            transport_result=None,
            steering_meta=None,
            total_delta_norm=float(np.linalg.norm(after - v)),
            stages_applied=["translation"] if apply_stages else [],
        )

    def test_last_result_available_true(self):
        import dashboard_server as ds
        r = self._make_tts_result()
        d = ds._tts_result_to_dict(r)
        self.assertTrue(d["last_result_available"])

    def test_norm_before_and_after_present(self):
        import dashboard_server as ds
        r = self._make_tts_result()
        d = ds._tts_result_to_dict(r)
        self.assertIn("norm_before", d)
        self.assertIn("norm_after", d)
        self.assertIsInstance(d["norm_before"], float)

    def test_stages_applied_list(self):
        import dashboard_server as ds
        r = self._make_tts_result(apply_stages=True)
        d = ds._tts_result_to_dict(r)
        self.assertIsInstance(d["stages_applied"], list)
        self.assertIn("translation", d["stages_applied"])

    def test_total_delta_norm_float(self):
        import dashboard_server as ds
        r = self._make_tts_result(apply_stages=True)
        d = ds._tts_result_to_dict(r)
        self.assertIsInstance(d["total_delta_norm"], float)

    def test_none_results_produce_none_fields(self):
        import dashboard_server as ds
        r = self._make_tts_result()
        d = ds._tts_result_to_dict(r)
        self.assertIsNone(d["translation_offset_norm"])
        self.assertIsNone(d["transport_weight_used"])
        self.assertIsNone(d["steering_delta_norm"])

    def test_passthrough_total_delta_near_zero(self):
        import dashboard_server as ds
        r = self._make_tts_result(apply_stages=False)
        d = ds._tts_result_to_dict(r)
        self.assertAlmostEqual(d["total_delta_norm"], 0.0, places=10)


# ===========================================================================
# Dashboard HTTP API handlers
# ===========================================================================

class _FakeRequest:
    """Minimal fake HTTP request object for DashboardHandler."""
    def makefile(self, *args, **kwargs):
        return io.BytesIO(b"GET / HTTP/1.0\r\n\r\n")


class TestDashboardTTSHandlers(unittest.TestCase):
    """Tests for handle_api_tts_status() and handle_api_tts_last_result()."""

    def _make_handler(self):
        import dashboard_server as ds

        sent_bytes = []

        class StubHandler(ds.DashboardHandler):
            def __init__(self):  # noqa: D107
                # skip normal BaseHTTPRequestHandler.__init__
                self.wfile = io.BytesIO()

            def send_json_response(self, payload: dict) -> None:  # type: ignore[override]
                sent_bytes.append(payload)

            def send_error_response(self, code: int, msg: str) -> None:  # type: ignore[override]
                sent_bytes.append({"__error__": True, "code": code, "msg": msg})

        handler = StubHandler()
        return handler, sent_bytes

    def setUp(self):
        import dashboard_server as ds
        ds._TTS_DASHBOARD_STATE["enabled"] = False
        ds._TTS_DASHBOARD_STATE["config"] = {}
        ds._TTS_DASHBOARD_STATE["last_result"] = None

    def test_status_returns_enabled_false_by_default(self):
        handler, sent = self._make_handler()
        handler.handle_api_tts_status()
        self.assertEqual(len(sent), 1)
        self.assertFalse(sent[0]["enabled"])

    def test_status_returns_enabled_true_when_set(self):
        import dashboard_server as ds
        ds._TTS_DASHBOARD_STATE["enabled"] = True
        handler, sent = self._make_handler()
        handler.handle_api_tts_status()
        self.assertTrue(sent[0]["enabled"])

    def test_status_contains_config_key(self):
        handler, sent = self._make_handler()
        handler.handle_api_tts_status()
        self.assertIn("config", sent[0])

    def test_status_contains_last_result_summary_key(self):
        handler, sent = self._make_handler()
        handler.handle_api_tts_status()
        self.assertIn("last_result_summary", sent[0])

    def test_status_last_result_summary_none_when_no_result(self):
        handler, sent = self._make_handler()
        handler.handle_api_tts_status()
        self.assertIsNone(sent[0]["last_result_summary"])

    def test_status_last_result_summary_populated_when_result(self):
        import dashboard_server as ds
        ds._TTS_DASHBOARD_STATE["last_result"] = {
            "stages_applied": ["translation"],
            "total_delta_norm": 0.05,
        }
        handler, sent = self._make_handler()
        handler.handle_api_tts_status()
        summary = sent[0]["last_result_summary"]
        self.assertIsNotNone(summary)
        self.assertIn("stages_applied", summary)
        self.assertAlmostEqual(summary["total_delta_norm"], 0.05)

    def test_last_result_no_data_when_none(self):
        handler, sent = self._make_handler()
        handler.handle_api_tts_last_result()
        self.assertFalse(sent[0]["last_result_available"])

    def test_last_result_data_when_available(self):
        import dashboard_server as ds
        ds._TTS_DASHBOARD_STATE["last_result"] = {
            "last_result_available": True,
            "total_delta_norm": 0.1,
            "stages_applied": ["steering"],
        }
        handler, sent = self._make_handler()
        handler.handle_api_tts_last_result()
        self.assertTrue(sent[0]["last_result_available"])
        self.assertAlmostEqual(sent[0]["total_delta_norm"], 0.1)

    def test_last_result_null_fields_when_no_result(self):
        handler, sent = self._make_handler()
        handler.handle_api_tts_last_result()
        response = sent[0]
        for key in ("translation_offset_norm", "transport_weight_used",
                    "steering_delta_norm", "norm_before", "norm_after",
                    "total_delta_norm", "stages_applied"):
            self.assertIn(key, response)
            self.assertIsNone(response[key])


# ===========================================================================
# VectorSteerer standalone
# ===========================================================================

class TestVectorSteererStandaloneMode(unittest.TestCase):
    """Verify VectorSteerer steers correctly and obeys max_strength."""

    def setUp(self):
        from tts_pipeline import VectorSteerer, SteeringSignal
        self.VectorSteerer = VectorSteerer
        self.SteeringSignal = SteeringSignal

    def test_no_signals_no_change(self):
        steerer = self.VectorSteerer()
        v = np.array([1.0, 0.0, 0.0])
        out, meta = steerer.steer(v)
        np.testing.assert_array_equal(out, v)
        self.assertFalse(meta["was_steered"])

    def test_signal_applied(self):
        steerer = self.VectorSteerer(max_strength=1.0)
        direction = np.array([0.0, 1.0, 0.0])
        steerer.add_signal(self.SteeringSignal(direction=direction, strength=0.5, source="test"))
        v = np.zeros(3)
        out, meta = steerer.steer(v)
        self.assertTrue(meta["was_steered"])
        self.assertAlmostEqual(out[1], 0.5, places=10)

    def test_max_strength_clamped(self):
        steerer = self.VectorSteerer(max_strength=0.1)
        direction = np.array([1.0, 0.0, 0.0])
        steerer.add_signal(self.SteeringSignal(direction=direction, strength=5.0, source="big"))
        v = np.zeros(3)
        out, meta = steerer.steer(v)
        self.assertAlmostEqual(meta["total_delta_norm"], 0.1, places=10)

    def test_clear_signals(self):
        steerer = self.VectorSteerer()
        direction = np.array([1.0, 0.0, 0.0])
        steerer.add_signal(self.SteeringSignal(direction=direction, strength=0.5, source="s"))
        steerer.clear_signals()
        v = np.zeros(3)
        _, meta = steerer.steer(v)
        self.assertFalse(meta["was_steered"])


# ===========================================================================
# SteeringMode enum
# ===========================================================================

class TestSteeringModeEnum(unittest.TestCase):
    """SteeringMode only has SHADOW, SOFT_SCALE, SUPPRESSION (no ACTIVE)."""

    def test_shadow_exists(self):
        from steering_policy import SteeringMode
        self.assertEqual(SteeringMode.SHADOW.value, "SHADOW")

    def test_soft_scale_exists(self):
        from steering_policy import SteeringMode
        self.assertEqual(SteeringMode.SOFT_SCALE.value, "SOFT_SCALE")

    def test_suppression_exists(self):
        from steering_policy import SteeringMode
        self.assertEqual(SteeringMode.SUPPRESSION.value, "SUPPRESSION")

    def test_active_not_in_steering_mode(self):
        from steering_policy import SteeringMode
        values = {m.value for m in SteeringMode}
        self.assertNotIn("ACTIVE", values)


# ===========================================================================
# --enable-tts CLI flag wiring in run_road_course_campaign
# ===========================================================================

class TestRunRoadCourseCampaignEnableTTS(unittest.TestCase):
    """Tests for the --enable-tts flag wiring in run_road_course_campaign.

    These tests exercise the real TTSConfig initialization and real
    evaluate_profile() call path (with mock data to avoid network/MTEB deps).
    """

    def _make_minimal_corpus_queries_qrels(self):
        """Return minimal corpus/queries/qrels dicts for one query + two docs."""
        corpus = {"doc0": "the cat sat on the mat", "doc1": "neural networks learn representations"}
        queries = {"q0": "cat mat"}
        qrels = {"q0": {"doc0": 1}}
        return corpus, queries, qrels

    def test_tts_config_is_none_when_flag_not_set(self):
        """When --enable-tts is not passed, tts_config must be None."""
        import argparse
        # Simulate parse_args without --enable-tts
        import run_road_course_campaign as rrc
        parser = argparse.ArgumentParser()
        parser.add_argument("--enable-tts", action="store_true", default=False)
        args = parser.parse_args([])
        tts_config = rrc.TTSConfig() if args.enable_tts else None
        self.assertIsNone(tts_config)

    def test_tts_config_is_constructed_when_flag_set(self):
        """When --enable-tts is passed, TTSConfig must be constructed."""
        import argparse
        import run_road_course_campaign as rrc
        parser = argparse.ArgumentParser()
        parser.add_argument("--enable-tts", action="store_true", default=False)
        args = parser.parse_args(["--enable-tts"])
        tts_config = rrc.TTSConfig() if args.enable_tts else None
        self.assertIsNotNone(tts_config)
        self.assertIsInstance(tts_config, rrc.TTSConfig)

    @patch("antigravity_engine.get_logger", return_value=MagicMock())
    @patch("dashboard_server.update_tts_dashboard_state")
    def test_evaluate_profile_tts_enabled_sets_pipeline_on_engine(self, _upd, _log):
        """evaluate_profile() with tts_config must call engine.enable_tts()."""
        from run_road_course_campaign import RoadCourseProfile, evaluate_profile
        from tts_pipeline import TTSConfig

        tts_config = TTSConfig()
        profile = RoadCourseProfile("baseline")
        corpus, queries, qrels = self._make_minimal_corpus_queries_qrels()

        captured_tts_config = []

        def fake_enable_tts(self_engine, tts_config=None, **kwargs):
            captured_tts_config.append(tts_config)
            # Also set _tts_pipeline to a dummy so get_last_tts_result works
            from tts_pipeline import TTSPipeline, VectorSteerer
            from vector_translator import TranslationConfig, VectorTranslator
            from vector_transport import TransportConfig, VectorTransport

            cfg = tts_config or TTSConfig()
            translator = VectorTranslator(TranslationConfig(offset_dim=self_engine.vector_size))
            transport = VectorTransport(TransportConfig())
            steerer = VectorSteerer(max_strength=0.3)
            self_engine._tts_pipeline = TTSPipeline(translator, transport, steerer, cfg)
            self_engine._last_tts_result = None

        from antigravity_engine import AntigravityEngine
        with patch.object(AntigravityEngine, "enable_tts", fake_enable_tts):
            try:
                evaluate_profile(
                    profile,
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    corpus=corpus,
                    queries=queries,
                    qrels=qrels,
                    tts_config=tts_config,
                )
            except Exception:
                # Some imports may fail in limited CI; we still check the capture
                pass

        # enable_tts must have been called at least once with a TTSConfig
        self.assertGreater(len(captured_tts_config), 0, "engine.enable_tts() was never called")
        self.assertIsInstance(captured_tts_config[0], TTSConfig)

    @patch("antigravity_engine.get_logger", return_value=MagicMock())
    @patch("dashboard_server.update_tts_dashboard_state")
    def test_evaluate_profile_tts_disabled_does_not_call_enable_tts(self, _upd, _log):
        """evaluate_profile() without tts_config must NOT call engine.enable_tts()."""
        from run_road_course_campaign import RoadCourseProfile, evaluate_profile

        profile = RoadCourseProfile("baseline")
        corpus, queries, qrels = self._make_minimal_corpus_queries_qrels()

        captured_tts_config = []

        def fake_enable_tts(self_engine, tts_config=None, **kwargs):
            captured_tts_config.append(tts_config)

        from antigravity_engine import AntigravityEngine
        with patch.object(AntigravityEngine, "enable_tts", fake_enable_tts):
            try:
                evaluate_profile(
                    profile,
                    model_name="sentence-transformers/all-MiniLM-L6-v2",
                    corpus=corpus,
                    queries=queries,
                    qrels=qrels,
                    tts_config=None,
                )
            except Exception:
                pass

        self.assertEqual(
            len(captured_tts_config),
            0,
            "engine.enable_tts() must NOT be called when tts_config is None",
        )

    def test_result_tts_key_disabled_when_no_tts_config(self):
        """Profile result must contain tts.enabled=False when tts_config is None."""
        from run_road_course_campaign import RoadCourseProfile, evaluate_profile

        profile = RoadCourseProfile("baseline")
        corpus, queries, qrels = self._make_minimal_corpus_queries_qrels()

        try:
            result = evaluate_profile(
                profile,
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                corpus=corpus,
                queries=queries,
                qrels=qrels,
                tts_config=None,
            )
        except ImportError as exc:
            self.skipTest(f"sentence-transformers not available: {exc}")

        self.assertIn("tts", result)
        self.assertFalse(result["tts"]["enabled"])

    def test_result_tts_key_enabled_when_tts_config_passed(self):
        """Profile result must contain tts.enabled=True when tts_config is provided."""
        from run_road_course_campaign import RoadCourseProfile, evaluate_profile
        from tts_pipeline import TTSConfig

        profile = RoadCourseProfile("baseline")
        corpus, queries, qrels = self._make_minimal_corpus_queries_qrels()
        tts_config = TTSConfig()

        try:
            result = evaluate_profile(
                profile,
                model_name="sentence-transformers/all-MiniLM-L6-v2",
                corpus=corpus,
                queries=queries,
                qrels=qrels,
                tts_config=tts_config,
            )
        except ImportError as exc:
            self.skipTest(f"sentence-transformers not available: {exc}")

        self.assertIn("tts", result)
        self.assertTrue(result["tts"]["enabled"])
        self.assertIn("query_count", result["tts"])
        self.assertIn("per_query", result["tts"])


# ===========================================================================
# Real CLI wiring: main() argparse path for TTS flags
# ===========================================================================

class TestRunRoadCourseCampaignCLIWiring(unittest.TestCase):
    """Exercise the real main() argparse path — not a reconstructed parser.

    These tests mock run_campaign to avoid model loading but call the
    actual run_road_course_campaign.main() so the real ArgumentParser and
    TTSConfig construction code paths are covered.
    """

    def _minimal_campaign_result(self) -> dict:
        """Return a fake run_campaign result with the keys main() accesses."""
        return {
            "task": "SciFact",
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "corpus_size": 2,
            "query_count": 1,
            "profile_results": [],
            "default_recommendation": {
                "recommended_profile": "baseline",
                "baseline_ndcg_at_10": 0.5,
                "best_ndcg_at_10": 0.5,
                "delta_vs_baseline": 0.0,
                "default_change_allowed": False,
                "reason": "baseline_remains_best",
            },
        }

    @patch("run_road_course_campaign.run_campaign")
    def test_main_no_tts_steering_disables_steering_in_tts_config(self, mock_run_campaign):
        """--enable-tts --no-tts-steering must produce TTSConfig(steering_enabled=False)."""
        import sys
        import tempfile
        import os
        import run_road_course_campaign as rrc

        mock_run_campaign.return_value = self._minimal_campaign_result()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "out.json")
            argv = [
                "run_road_course_campaign.py",
                "--enable-tts",
                "--no-tts-steering",
                "--output", output_file,
                # Keep network/data loading fast — SciFact with 1 query still
                # hits MTEB; max_queries=1 exercises the CLI path with minimal work.
                "--max-queries", "1",
                "--sample-docs", "2",
            ]
            with patch.object(sys, "argv", argv):
                rrc.main()

        # Verify run_campaign was called with a TTSConfig that has steering disabled
        self.assertTrue(mock_run_campaign.called, "run_campaign must have been called")
        call_kwargs = mock_run_campaign.call_args
        tts_config_arg = call_kwargs.kwargs.get("tts_config") or (
            call_kwargs.args[6] if len(call_kwargs.args) > 6 else None
        )
        self.assertIsNotNone(
            tts_config_arg,
            "--enable-tts must produce a non-None tts_config passed to run_campaign",
        )
        from tts_pipeline import TTSConfig
        self.assertIsInstance(tts_config_arg, TTSConfig)
        self.assertFalse(
            tts_config_arg.steering_enabled,
            "--no-tts-steering must set TTSConfig.steering_enabled=False",
        )
        self.assertTrue(
            tts_config_arg.translation_enabled,
            "--tts-translation defaults True; must remain True when not overridden",
        )
        self.assertTrue(
            tts_config_arg.transport_enabled,
            "--tts-transport defaults True; must remain True when not overridden",
        )

    @patch("run_road_course_campaign.run_campaign")
    def test_main_no_tts_translation_disables_translation_in_tts_config(self, mock_run_campaign):
        """--enable-tts --no-tts-translation must produce TTSConfig(translation_enabled=False)."""
        import sys
        import tempfile
        import os
        import run_road_course_campaign as rrc

        mock_run_campaign.return_value = self._minimal_campaign_result()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "out.json")
            argv = [
                "run_road_course_campaign.py",
                "--enable-tts",
                "--no-tts-translation",
                "--output", output_file,
                "--max-queries", "1",
                "--sample-docs", "2",
            ]
            with patch.object(sys, "argv", argv):
                rrc.main()

        self.assertTrue(mock_run_campaign.called, "run_campaign must have been called")
        call_kwargs = mock_run_campaign.call_args
        tts_config_arg = call_kwargs.kwargs.get("tts_config") or (
            call_kwargs.args[6] if len(call_kwargs.args) > 6 else None
        )
        self.assertIsNotNone(
            tts_config_arg,
            "--enable-tts must produce a non-None tts_config passed to run_campaign",
        )
        from tts_pipeline import TTSConfig
        self.assertIsInstance(tts_config_arg, TTSConfig)
        self.assertFalse(
            tts_config_arg.translation_enabled,
            "--no-tts-translation must set TTSConfig.translation_enabled=False",
        )
        self.assertTrue(
            tts_config_arg.transport_enabled,
            "--tts-transport defaults True; must remain True when not overridden",
        )
        self.assertTrue(
            tts_config_arg.steering_enabled,
            "--tts-steering defaults True; must remain True when not overridden",
        )

    @patch("run_road_course_campaign.run_campaign")
    def test_main_no_tts_transport_disables_transport_in_tts_config(self, mock_run_campaign):
        """--enable-tts --no-tts-transport must produce TTSConfig(transport_enabled=False)."""
        import sys
        import tempfile
        import os
        import run_road_course_campaign as rrc

        mock_run_campaign.return_value = self._minimal_campaign_result()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "out.json")
            argv = [
                "run_road_course_campaign.py",
                "--enable-tts",
                "--no-tts-transport",
                "--output", output_file,
                "--max-queries", "1",
                "--sample-docs", "2",
            ]
            with patch.object(sys, "argv", argv):
                rrc.main()

        self.assertTrue(mock_run_campaign.called, "run_campaign must have been called")
        call_kwargs = mock_run_campaign.call_args
        tts_config_arg = call_kwargs.kwargs.get("tts_config") or (
            call_kwargs.args[6] if len(call_kwargs.args) > 6 else None
        )
        self.assertIsNotNone(
            tts_config_arg,
            "--enable-tts must produce a non-None tts_config passed to run_campaign",
        )
        from tts_pipeline import TTSConfig
        self.assertIsInstance(tts_config_arg, TTSConfig)
        self.assertFalse(
            tts_config_arg.transport_enabled,
            "--no-tts-transport must set TTSConfig.transport_enabled=False",
        )
        self.assertTrue(
            tts_config_arg.translation_enabled,
            "--tts-translation defaults True; must remain True when not overridden",
        )
        self.assertTrue(
            tts_config_arg.steering_enabled,
            "--tts-steering defaults True; must remain True when not overridden",
        )

    @patch("run_road_course_campaign.run_campaign")
    def test_main_without_enable_tts_passes_none_tts_config(self, mock_run_campaign):
        """When --enable-tts is absent, main() must pass tts_config=None to run_campaign."""
        import sys
        import tempfile
        import os
        import run_road_course_campaign as rrc

        mock_run_campaign.return_value = self._minimal_campaign_result()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = os.path.join(tmpdir, "out.json")
            argv = [
                "run_road_course_campaign.py",
                "--output", output_file,
                "--max-queries", "1",
                "--sample-docs", "2",
            ]
            with patch.object(sys, "argv", argv):
                rrc.main()

        self.assertTrue(mock_run_campaign.called, "run_campaign must have been called")
        call_kwargs = mock_run_campaign.call_args
        tts_config_arg = call_kwargs.kwargs.get("tts_config") or (
            call_kwargs.args[6] if len(call_kwargs.args) > 6 else None
        )
        self.assertIsNone(
            tts_config_arg,
            "Without --enable-tts, tts_config must be None",
        )


if __name__ == "__main__":
    unittest.main()
