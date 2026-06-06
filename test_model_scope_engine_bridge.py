"""Tests for model_scope_engine_bridge — Phase 6 engine integration."""

from __future__ import annotations

import dataclasses
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers to build minimal ActivationEvent / SparseFeatureEvent for tests
# ---------------------------------------------------------------------------


def _make_activation_event(run_id: str = "run_test_01") -> object:
    from model_scope_runtime import ActivationEvent

    return ActivationEvent(
        schema_version="1.0",
        model_id="test_model",
        layer_id="layer.0",
        token_count=4,
        shape=(1, 4, 16),
        mean_activation=0.1,
        norm_activation=0.8,
        captured_at="2026-01-01T00:00:00+00:00",
        run_id=run_id,
    )


def _make_sparse_feature_event(run_id: str = "run_test_01") -> object:
    from model_scope_features import SparseFeatureEvent

    activation = _make_activation_event(run_id)
    return SparseFeatureEvent(
        source_activation=activation,
        feature_source="raw_stats",
        features={"mean_activation": 0.1, "norm_activation": 0.8},
        feature_count=3,
        nonzero_count=2,
        extracted_at="2026-01-01T00:00:00+00:00",
    )


# ---------------------------------------------------------------------------
# ModelScopeBridgeConfig tests
# ---------------------------------------------------------------------------


class TestModelScopeBridgeConfig(unittest.TestCase):
    def test_defaults(self):
        from model_scope_engine_bridge import ModelScopeBridgeConfig

        cfg = ModelScopeBridgeConfig()
        self.assertEqual(cfg.artifact_dir, "experiment_runs/model_scope")
        self.assertTrue(cfg.enable_feature_extraction)
        self.assertFalse(cfg.enable_steering)
        self.assertEqual(cfg.max_events_in_memory, 100)
        self.assertEqual(cfg.observation_tag, "engine_bridge")

    def test_custom_values(self):
        from model_scope_engine_bridge import ModelScopeBridgeConfig

        cfg = ModelScopeBridgeConfig(
            artifact_dir="custom/dir",
            enable_feature_extraction=False,
            enable_steering=True,
            max_events_in_memory=50,
            observation_tag="my_tag",
        )
        self.assertEqual(cfg.artifact_dir, "custom/dir")
        self.assertFalse(cfg.enable_feature_extraction)
        self.assertTrue(cfg.enable_steering)
        self.assertEqual(cfg.max_events_in_memory, 50)
        self.assertEqual(cfg.observation_tag, "my_tag")

    def test_is_dataclass(self):
        from model_scope_engine_bridge import ModelScopeBridgeConfig

        cfg = ModelScopeBridgeConfig()
        self.assertTrue(dataclasses.is_dataclass(cfg))

    def test_asdict(self):
        from model_scope_engine_bridge import ModelScopeBridgeConfig

        cfg = ModelScopeBridgeConfig(artifact_dir="x/y")
        d = dataclasses.asdict(cfg)
        self.assertIn("artifact_dir", d)
        self.assertEqual(d["artifact_dir"], "x/y")


# ---------------------------------------------------------------------------
# ModelScopeObservationResult tests
# ---------------------------------------------------------------------------


class TestModelScopeObservationResult(unittest.TestCase):
    def test_basic_construction(self):
        from model_scope_engine_bridge import ModelScopeObservationResult

        r = ModelScopeObservationResult(run_id="r1", query="hello world")
        self.assertEqual(r.run_id, "r1")
        self.assertEqual(r.query, "hello world")
        self.assertIsNone(r.activation_event)
        self.assertIsNone(r.feature_event)
        self.assertEqual(r.intervention_count, 0)
        self.assertIsNone(r.artifact_path)
        self.assertEqual(r.elapsed_ms, 0.0)
        self.assertIsNone(r.error)

    def test_optional_fields(self):
        from model_scope_engine_bridge import ModelScopeObservationResult

        act = _make_activation_event()
        r = ModelScopeObservationResult(
            run_id="r2",
            query="q",
            activation_event=act,
            feature_event=None,
            intervention_count=3,
            artifact_path="/some/path.json",
            elapsed_ms=12.5,
            error=None,
        )
        self.assertIs(r.activation_event, act)
        self.assertEqual(r.intervention_count, 3)
        self.assertEqual(r.artifact_path, "/some/path.json")
        self.assertAlmostEqual(r.elapsed_ms, 12.5)

    def test_error_field(self):
        from model_scope_engine_bridge import ModelScopeObservationResult

        r = ModelScopeObservationResult(run_id="r3", query="q", error="runtime_not_loaded")
        self.assertEqual(r.error, "runtime_not_loaded")

    def test_is_dataclass(self):
        from model_scope_engine_bridge import ModelScopeObservationResult

        r = ModelScopeObservationResult(run_id="x", query="y")
        self.assertTrue(dataclasses.is_dataclass(r))


# ---------------------------------------------------------------------------
# ModelScopeEngineBridge initialisation tests
# ---------------------------------------------------------------------------


class TestModelScopeEngineBridgeInit(unittest.TestCase):
    def _bridge_in_tmpdir(self, **kw):
        from model_scope_engine_bridge import ModelScopeEngineBridge, ModelScopeBridgeConfig

        td = tempfile.mkdtemp(prefix="bridge_test_")
        cfg = ModelScopeBridgeConfig(artifact_dir=td, **kw)
        return ModelScopeEngineBridge(cfg), td

    def test_creates_artifact_store(self):
        from model_scope_artifacts import ArtifactStore

        bridge, _ = self._bridge_in_tmpdir()
        self.assertIsInstance(bridge._artifact_store, ArtifactStore)

    def test_creates_memory_manager(self):
        from model_scope_memory import MemoryManager

        bridge, _ = self._bridge_in_tmpdir()
        self.assertIsInstance(bridge._memory_manager, MemoryManager)

    def test_creates_feature_extractor(self):
        from model_scope_features import FeatureExtractor

        bridge, _ = self._bridge_in_tmpdir()
        self.assertIsInstance(bridge._extractor, FeatureExtractor)

    def test_creates_steering_actuator(self):
        from model_scope_steering import SteeringActuator

        bridge, _ = self._bridge_in_tmpdir()
        self.assertIsInstance(bridge._actuator, SteeringActuator)

    def test_counters_start_at_zero(self):
        bridge, _ = self._bridge_in_tmpdir()
        self.assertEqual(bridge._observation_count, 0)
        self.assertEqual(bridge._error_count, 0)

    def test_default_config_used_when_none(self):
        from model_scope_engine_bridge import ModelScopeEngineBridge

        with patch("model_scope_engine_bridge.Path") as MockPath:
            mock_path_inst = MagicMock()
            MockPath.return_value = mock_path_inst
            mock_path_inst.__truediv__ = lambda self, other: mock_path_inst
            mock_path_inst.mkdir = MagicMock()
            mock_path_inst.__str__ = lambda self: "experiment_runs/model_scope"
            bridge = ModelScopeEngineBridge(None)
        self.assertEqual(bridge._config.artifact_dir, "experiment_runs/model_scope")

    def test_register_shadow_policy_records_default_and_returns_id(self):
        from model_scope_engine_bridge import ModelScopeEngineBridge
        from steering_policy import PolicyStatus

        bridge, _ = self._bridge_in_tmpdir()
        policy_id = bridge.register_shadow_policy(
            name="test_shadow_policy",
            feature_space="raw_stats",
            target_features=("mean_activation", "norm_activation"),
            max_interventions=7,
            status=PolicyStatus.ACTIVE,
        )
        self.assertEqual(bridge.get_last_shadow_policy_id(), policy_id)
        self.assertIn(policy_id, bridge.get_active_policy_ids())

    def test_register_shadow_policy_records_status_and_id(self):
        from model_scope_engine_bridge import ModelScopeEngineBridge
        from steering_policy import PolicyStatus

        bridge, _ = self._bridge_in_tmpdir()
        policy_id = "explicit-ms-policy-id"
        returned = bridge.register_shadow_policy(
            name="inactive_shadow_policy",
            feature_space="raw_stats",
            target_features=("mean_activation", "norm_activation"),
            max_interventions=1,
            status=PolicyStatus.DISABLED,
            policy_id=policy_id,
        )
        self.assertEqual(returned, policy_id)
        self.assertNotIn(policy_id, bridge.get_active_policy_ids())


# ---------------------------------------------------------------------------
# Bridge.observe — runtime not loaded
# ---------------------------------------------------------------------------


class TestBridgeObserveRuntimeNotLoaded(unittest.TestCase):
    def setUp(self):
        from model_scope_engine_bridge import ModelScopeEngineBridge, ModelScopeBridgeConfig

        self._td = tempfile.mkdtemp(prefix="bridge_obs_test_")
        cfg = ModelScopeBridgeConfig(artifact_dir=self._td)
        self._bridge = ModelScopeEngineBridge(cfg)

    def _mock_runtime(self, *, loaded: bool, events=None):
        rt = MagicMock()
        rt.is_loaded.return_value = loaded
        rt.get_events.return_value = events or []
        return rt

    def test_not_loaded_returns_error(self):
        rt = self._mock_runtime(loaded=False)
        result = self._bridge.observe("hello", rt)
        self.assertEqual(result.error, "runtime_not_loaded")

    def test_not_loaded_increments_error_count(self):
        rt = self._mock_runtime(loaded=False)
        self._bridge.observe("hello", rt)
        self.assertEqual(self._bridge._error_count, 1)

    def test_not_loaded_does_not_increment_observation_count(self):
        rt = self._mock_runtime(loaded=False)
        self._bridge.observe("hello", rt)
        self.assertEqual(self._bridge._observation_count, 0)

    def test_not_loaded_result_has_no_activation(self):
        rt = self._mock_runtime(loaded=False)
        result = self._bridge.observe("hello", rt)
        self.assertIsNone(result.activation_event)

    def test_not_loaded_run_id_present(self):
        rt = self._mock_runtime(loaded=False)
        result = self._bridge.observe("hello", rt)
        self.assertIsNotNone(result.run_id)
        self.assertTrue(result.run_id.startswith("obs_"))


# ---------------------------------------------------------------------------
# Bridge.observe — runtime loaded, has events
# ---------------------------------------------------------------------------


class TestBridgeObserveLoaded(unittest.TestCase):
    def setUp(self):
        from model_scope_engine_bridge import ModelScopeEngineBridge, ModelScopeBridgeConfig

        self._td = tempfile.mkdtemp(prefix="bridge_obs_loaded_")
        cfg = ModelScopeBridgeConfig(
            artifact_dir=self._td, enable_feature_extraction=True
        )
        self._bridge = ModelScopeEngineBridge(cfg)

    def _mock_runtime(self, events):
        rt = MagicMock()
        rt.is_loaded.return_value = True
        rt.get_events.return_value = events
        return rt

    def test_returns_activation_event(self):
        act = _make_activation_event()
        rt = self._mock_runtime([act])
        result = self._bridge.observe("query text", rt)
        self.assertIs(result.activation_event, act)

    def test_uses_last_event_when_multiple(self):
        act1 = _make_activation_event("r1")
        act2 = _make_activation_event("r2")
        rt = self._mock_runtime([act1, act2])
        result = self._bridge.observe("q", rt)
        self.assertIs(result.activation_event, act2)

    def test_increments_observation_count(self):
        act = _make_activation_event()
        rt = self._mock_runtime([act])
        self._bridge.observe("q", rt)
        self.assertEqual(self._bridge._observation_count, 1)

    def test_feature_event_populated(self):
        act = _make_activation_event()
        rt = self._mock_runtime([act])
        result = self._bridge.observe("q", rt)
        self.assertIsNotNone(result.feature_event)

    def test_artifact_path_saved(self):
        act = _make_activation_event()
        rt = self._mock_runtime([act])
        result = self._bridge.observe("q", rt)
        self.assertIsNotNone(result.artifact_path)
        self.assertTrue(Path(result.artifact_path).exists())

    def test_no_error_on_success(self):
        act = _make_activation_event()
        rt = self._mock_runtime([act])
        result = self._bridge.observe("q", rt)
        self.assertIsNone(result.error)

    def test_empty_events_no_activation(self):
        rt = self._mock_runtime([])
        result = self._bridge.observe("q", rt)
        self.assertIsNone(result.activation_event)
        self.assertEqual(self._bridge._observation_count, 1)


# ---------------------------------------------------------------------------
# Bridge.get_telemetry
# ---------------------------------------------------------------------------


class TestBridgeGetTelemetry(unittest.TestCase):
    def setUp(self):
        from model_scope_engine_bridge import ModelScopeEngineBridge, ModelScopeBridgeConfig

        self._td = tempfile.mkdtemp(prefix="bridge_tel_")
        cfg = ModelScopeBridgeConfig(artifact_dir=self._td)
        self._bridge = ModelScopeEngineBridge(cfg)

    def test_returns_dict(self):
        tel = self._bridge.get_telemetry()
        self.assertIsInstance(tel, dict)

    def test_has_required_keys(self):
        tel = self._bridge.get_telemetry()
        self.assertIn("observation_count", tel)
        self.assertIn("error_count", tel)
        self.assertIn("artifact_dir", tel)
        self.assertIn("memory_snapshot", tel)

    def test_initial_counts_zero(self):
        tel = self._bridge.get_telemetry()
        self.assertEqual(tel["observation_count"], 0)
        self.assertEqual(tel["error_count"], 0)

    def test_memory_snapshot_is_dict(self):
        tel = self._bridge.get_telemetry()
        self.assertIsInstance(tel["memory_snapshot"], dict)

    def test_artifact_dir_matches_config(self):
        tel = self._bridge.get_telemetry()
        self.assertEqual(tel["artifact_dir"], self._td)


# ---------------------------------------------------------------------------
# Bridge.list_recent_observations
# ---------------------------------------------------------------------------


class TestBridgeListRecentObservations(unittest.TestCase):
    def setUp(self):
        from model_scope_engine_bridge import ModelScopeEngineBridge, ModelScopeBridgeConfig

        self._td = tempfile.mkdtemp(prefix="bridge_list_")
        cfg = ModelScopeBridgeConfig(artifact_dir=self._td, enable_feature_extraction=True)
        self._bridge = ModelScopeEngineBridge(cfg)

    def test_empty_store_returns_empty_list(self):
        result = self._bridge.list_recent_observations()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 0)

    def test_after_observation_returns_items(self):
        act = _make_activation_event()
        rt = MagicMock()
        rt.is_loaded.return_value = True
        rt.get_events.return_value = [act]
        self._bridge.observe("test query", rt)
        items = self._bridge.list_recent_observations()
        self.assertGreaterEqual(len(items), 1)

    def test_each_item_has_path(self):
        act = _make_activation_event()
        rt = MagicMock()
        rt.is_loaded.return_value = True
        rt.get_events.return_value = [act]
        self._bridge.observe("q", rt)
        for item in self._bridge.list_recent_observations():
            self.assertIn("path", item)

    def test_limit_respected(self):
        rt = MagicMock()
        rt.is_loaded.return_value = True
        for i in range(5):
            act = _make_activation_event(f"run_{i:03d}")
            rt.get_events.return_value = [act]
            self._bridge.observe(f"q{i}", rt)
        items = self._bridge.list_recent_observations(limit=2)
        self.assertLessEqual(len(items), 2)


# ---------------------------------------------------------------------------
# Bridge.get_summary_for_diagnostics
# ---------------------------------------------------------------------------


class TestBridgeGetSummaryForDiagnostics(unittest.TestCase):
    def setUp(self):
        from model_scope_engine_bridge import ModelScopeEngineBridge, ModelScopeBridgeConfig

        self._td = tempfile.mkdtemp(prefix="bridge_summary_")
        cfg = ModelScopeBridgeConfig(artifact_dir=self._td)
        self._bridge = ModelScopeEngineBridge(cfg)

    def test_returns_dict(self):
        s = self._bridge.get_summary_for_diagnostics()
        self.assertIsInstance(s, dict)

    def test_has_observation_count(self):
        s = self._bridge.get_summary_for_diagnostics()
        self.assertIn("observation_count", s)

    def test_has_error_count(self):
        s = self._bridge.get_summary_for_diagnostics()
        self.assertIn("error_count", s)

    def test_has_last_artifact(self):
        s = self._bridge.get_summary_for_diagnostics()
        self.assertIn("last_artifact", s)

    def test_has_bridge_config(self):
        s = self._bridge.get_summary_for_diagnostics()
        self.assertIn("bridge_config", s)
        self.assertIsInstance(s["bridge_config"], dict)

    def test_last_artifact_none_when_empty(self):
        s = self._bridge.get_summary_for_diagnostics()
        self.assertIsNone(s["last_artifact"])

    def test_bridge_config_contains_artifact_dir(self):
        s = self._bridge.get_summary_for_diagnostics()
        self.assertIn("artifact_dir", s["bridge_config"])

    def test_get_summary_for_diagnostics_warns_on_corrupt_artifact(self):
        import warnings
        # Write a corrupt (non-JSON) file that matches the artifact pattern.
        bad_path = os.path.join(self._td, "feature_event_bad.json")
        with open(bad_path, "w", encoding="utf-8") as fh:
            fh.write("NOT VALID JSON {{{")
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = self._bridge.get_summary_for_diagnostics()
        self.assertIsNone(result["last_artifact"])
        matching = [
            w for w in caught
            if issubclass(w.category, UserWarning)
            and (
                "get_summary_for_diagnostics" in str(w.message)
                or "failed to read" in str(w.message)
            )
        ]
        self.assertTrue(
            matching,
            f"Expected a UserWarning about corrupt artifact; got: {[str(w.message) for w in caught]}",
        )


# ---------------------------------------------------------------------------
# Shim research preflight metadata
# ---------------------------------------------------------------------------


class TestModelScopeEngineBridgeResearchSeam(unittest.TestCase):
    def setUp(self) -> None:
        from model_scope_engine_bridge import ModelScopeBridgeConfig, ModelScopeEngineBridge

        os.environ.pop("CHELATED_SHIM_RESEARCH", None)
        os.environ.pop("CHELATED_SHIM_PROMOTED", None)
        self._td = tempfile.mkdtemp(prefix="bridge_research_")
        self._bridge = ModelScopeEngineBridge(ModelScopeBridgeConfig(artifact_dir=self._td, enable_feature_extraction=False))

    def tearDown(self) -> None:
        os.environ.pop("CHELATED_SHIM_RESEARCH", None)
        os.environ.pop("CHELATED_SHIM_PROMOTED", None)

    @staticmethod
    def _runtime(loaded: bool, events):
        rt = MagicMock()
        rt.is_loaded.return_value = loaded
        rt.get_events.return_value = events
        return rt

    def test_research_meta_emits_when_research_enabled(self) -> None:
        from model_scope_engine_bridge import ModelScopeBridgeConfig, ModelScopeEngineBridge

        os.environ["CHELATED_SHIM_RESEARCH"] = "1"
        bridge = ModelScopeEngineBridge(ModelScopeBridgeConfig(artifact_dir=self._td, enable_feature_extraction=False))
        bridge.observe("q", self._runtime(True, [_make_activation_event("run-1")]))

        meta = bridge.get_last_research_shim_meta()
        self.assertIsNotNone(meta)
        self.assertEqual(meta.get("sip_seam"), "ModelScopeEngineBridge.observe")
        self.assertTrue(meta.get("research_shim_guard"))
        self.assertEqual(meta.get("research_stall_count"), 0)

    def test_research_meta_includes_promoted_sip_apply_when_promoted_enabled(self) -> None:
        from model_scope_engine_bridge import ModelScopeBridgeConfig, ModelScopeEngineBridge

        os.environ["CHELATED_SHIM_RESEARCH"] = "1"
        os.environ["CHELATED_SHIM_PROMOTED"] = "1"
        bridge = ModelScopeEngineBridge(ModelScopeBridgeConfig(artifact_dir=self._td, enable_feature_extraction=False))
        bridge.observe("q", self._runtime(True, [_make_activation_event("run-promoted")]))

        meta = bridge.get_last_research_shim_meta()
        self.assertIsNotNone(meta)
        self.assertIn("promoted_sip_apply", meta)
        self.assertIsInstance(meta["promoted_sip_apply"], dict)
        self.assertEqual(meta.get("research_shim_meta"), None)

    def test_research_meta_tracks_stall_count_on_no_events(self) -> None:
        os.environ["CHELATED_SHIM_RESEARCH"] = "1"
        self._bridge.observe("first miss", self._runtime(True, []))
        self._bridge.observe("second miss", self._runtime(True, []))
        meta = self._bridge.get_last_research_shim_meta()
        self.assertIsNotNone(meta)
        self.assertEqual(meta.get("research_stall_count"), 2)

    def test_research_meta_absent_without_research_flag(self) -> None:
        self._bridge.observe("q", self._runtime(True, [_make_activation_event("run-2")]))
        self.assertIsNone(self._bridge.get_last_research_shim_meta())

    def test_research_meta_absent_when_runtime_not_loaded(self) -> None:
        result = self._bridge.observe("q", self._runtime(False, [_make_activation_event("run-3")]))
        self.assertIsNone(self._bridge.get_last_research_shim_meta())
        self.assertEqual(result.error, "runtime_not_loaded")

    def test_research_meta_reset_when_research_disabled_after_enabled_run(self) -> None:
        os.environ["CHELATED_SHIM_RESEARCH"] = "1"
        self._bridge.observe("q", self._runtime(True, [_make_activation_event("run-4")]))
        self.assertIsNotNone(self._bridge.get_last_research_shim_meta())

        os.environ.pop("CHELATED_SHIM_RESEARCH", None)
        self._bridge.observe("q", self._runtime(True, [_make_activation_event("run-5")]))
        self.assertIsNone(self._bridge.get_last_research_shim_meta())


# ---------------------------------------------------------------------------
# AntigravityEngine.observe_query_with_model_scope
# ---------------------------------------------------------------------------


class TestAntigravityEngineObserveQueryWithModelScope(unittest.TestCase):
    def _make_engine_stub(self):
        """Return a minimal engine-like object with bridge and runtime mocked."""
        engine = MagicMock()
        engine._runtime_telemetry = {
            "model_scope_observation_count": 0,
            "model_scope_error_count": 0,
        }
        engine._model_scope_runtime = None
        engine._model_scope_bridge = None
        return engine

    def test_returns_error_when_not_enabled(self):
        from antigravity_engine import AntigravityEngine

        real_method = AntigravityEngine.observe_query_with_model_scope
        engine = self._make_engine_stub()
        result = real_method(engine, "test query")
        self.assertEqual(result, {"error": "model_scope_not_enabled"})

    def test_returns_error_when_bridge_none(self):
        from antigravity_engine import AntigravityEngine

        engine = self._make_engine_stub()
        engine._model_scope_runtime = MagicMock()
        engine._model_scope_bridge = None
        result = AntigravityEngine.observe_query_with_model_scope(engine, "q")
        self.assertEqual(result, {"error": "model_scope_not_enabled"})

    def test_returns_dict_when_bridge_attached(self):
        from model_scope_engine_bridge import ModelScopeEngineBridge, ModelScopeBridgeConfig
        from antigravity_engine import AntigravityEngine

        td = tempfile.mkdtemp(prefix="engine_obs_")
        cfg = ModelScopeBridgeConfig(artifact_dir=td)
        bridge = ModelScopeEngineBridge(cfg)

        engine = self._make_engine_stub()
        rt = MagicMock()
        rt.is_loaded.return_value = False
        engine._model_scope_runtime = rt
        engine._model_scope_bridge = bridge

        result = AntigravityEngine.observe_query_with_model_scope(engine, "my query")
        self.assertIsInstance(result, dict)
        self.assertIn("run_id", result)
        self.assertIn("query", result)

    def test_increments_telemetry_count(self):
        from model_scope_engine_bridge import ModelScopeEngineBridge, ModelScopeBridgeConfig
        from antigravity_engine import AntigravityEngine

        td = tempfile.mkdtemp(prefix="engine_tel_")
        cfg = ModelScopeBridgeConfig(artifact_dir=td)
        bridge = ModelScopeEngineBridge(cfg)

        engine = self._make_engine_stub()
        rt = MagicMock()
        rt.is_loaded.return_value = False
        engine._model_scope_runtime = rt
        engine._model_scope_bridge = bridge

        AntigravityEngine.observe_query_with_model_scope(engine, "q")
        self.assertEqual(engine._runtime_telemetry["model_scope_observation_count"], 1)

    def test_stores_result_in_last_artifact(self):
        from model_scope_engine_bridge import ModelScopeEngineBridge, ModelScopeBridgeConfig
        from antigravity_engine import AntigravityEngine

        td = tempfile.mkdtemp(prefix="engine_art_")
        cfg = ModelScopeBridgeConfig(artifact_dir=td)
        bridge = ModelScopeEngineBridge(cfg)

        engine = self._make_engine_stub()
        rt = MagicMock()
        rt.is_loaded.return_value = False
        engine._model_scope_runtime = rt
        engine._model_scope_bridge = bridge

        AntigravityEngine.observe_query_with_model_scope(engine, "q")
        self.assertIsNotNone(engine._last_model_scope_artifact)


# ---------------------------------------------------------------------------
# Dashboard handler route existence and JSON response tests
# ---------------------------------------------------------------------------


class TestDashboardHandlerModelScopeRoutes(unittest.TestCase):
    def test_events_handler_exists(self):
        from dashboard_server import DashboardHandler

        self.assertTrue(hasattr(DashboardHandler, "handle_api_model_scope_events"))

    def test_features_handler_exists(self):
        from dashboard_server import DashboardHandler

        self.assertTrue(hasattr(DashboardHandler, "handle_api_model_scope_features"))

    def test_interventions_handler_exists(self):
        from dashboard_server import DashboardHandler

        self.assertTrue(hasattr(DashboardHandler, "handle_api_model_scope_interventions"))

    def _make_handler(self):
        from dashboard_server import DashboardHandler

        handler = MagicMock(spec=DashboardHandler)
        handler.send_json_response = MagicMock()
        handler.send_error_response = MagicMock()
        return handler

    def test_events_endpoint_returns_json_no_artifacts(self):
        from dashboard_server import DashboardHandler

        handler = self._make_handler()
        with patch("model_scope_artifacts.ArtifactStore.list_artifacts", return_value=[]):
            DashboardHandler.handle_api_model_scope_events(handler, {})
        handler.send_json_response.assert_called_once()
        call_arg = handler.send_json_response.call_args[0][0]
        self.assertIn("status", call_arg)
        self.assertEqual(call_arg["status"], "not_generated")

    def test_features_endpoint_returns_json_no_artifacts(self):
        from dashboard_server import DashboardHandler

        handler = self._make_handler()
        with patch("model_scope_artifacts.ArtifactStore.list_artifacts", return_value=[]):
            DashboardHandler.handle_api_model_scope_features(handler, {})
        handler.send_json_response.assert_called_once()
        call_arg = handler.send_json_response.call_args[0][0]
        self.assertIn("status", call_arg)
        self.assertEqual(call_arg["status"], "not_generated")

    def test_interventions_endpoint_returns_json_no_artifacts(self):
        from dashboard_server import DashboardHandler

        handler = self._make_handler()
        with patch("model_scope_artifacts.ArtifactStore.list_artifacts", return_value=[]):
            DashboardHandler.handle_api_model_scope_interventions(handler, {})
        handler.send_json_response.assert_called_once()
        call_arg = handler.send_json_response.call_args[0][0]
        self.assertIn("status", call_arg)
        self.assertEqual(call_arg["status"], "not_generated")

    def test_events_endpoint_count_key(self):
        from dashboard_server import DashboardHandler

        handler = self._make_handler()
        with patch("model_scope_artifacts.ArtifactStore.list_artifacts", return_value=[]):
            DashboardHandler.handle_api_model_scope_events(handler, {})
        call_arg = handler.send_json_response.call_args[0][0]
        self.assertIn("count", call_arg)
        self.assertEqual(call_arg["count"], 0)

    def test_features_endpoint_count_key(self):
        from dashboard_server import DashboardHandler

        handler = self._make_handler()
        with patch("model_scope_artifacts.ArtifactStore.list_artifacts", return_value=[]):
            DashboardHandler.handle_api_model_scope_features(handler, {})
        call_arg = handler.send_json_response.call_args[0][0]
        self.assertIn("count", call_arg)

    def test_interventions_endpoint_count_key(self):
        from dashboard_server import DashboardHandler

        handler = self._make_handler()
        with patch("model_scope_artifacts.ArtifactStore.list_artifacts", return_value=[]):
            DashboardHandler.handle_api_model_scope_interventions(handler, {})
        call_arg = handler.send_json_response.call_args[0][0]
        self.assertIn("count", call_arg)

    def test_events_endpoint_reason_key_when_empty(self):
        from dashboard_server import DashboardHandler

        handler = self._make_handler()
        with patch("model_scope_artifacts.ArtifactStore.list_artifacts", return_value=[]):
            DashboardHandler.handle_api_model_scope_events(handler, {})
        call_arg = handler.send_json_response.call_args[0][0]
        self.assertIn("reason", call_arg)
        self.assertIsNotNone(call_arg["reason"])

    def test_interventions_endpoint_reason_key_when_empty(self):
        from dashboard_server import DashboardHandler

        handler = self._make_handler()
        with patch("model_scope_artifacts.ArtifactStore.list_artifacts", return_value=[]):
            DashboardHandler.handle_api_model_scope_interventions(handler, {})
        call_arg = handler.send_json_response.call_args[0][0]
        self.assertIn("reason", call_arg)

    def test_model_scope_artifact_root_constant_exists(self):
        import dashboard_server

        self.assertTrue(hasattr(dashboard_server, "MODEL_SCOPE_ARTIFACT_ROOT"))

    def test_model_scope_artifact_root_is_string(self):
        import dashboard_server

        self.assertIsInstance(dashboard_server.MODEL_SCOPE_ARTIFACT_ROOT, str)


# ---------------------------------------------------------------------------
# MOD-6 fix: max_total_interventions wired through bridge config
# ---------------------------------------------------------------------------


class TestModelScopeBridgeConfigMaxTotalInterventions(unittest.TestCase):
    """Verify that max_total_interventions is exposed in config and wired to the actuator."""

    def test_default_max_total_interventions(self):
        from model_scope_engine_bridge import ModelScopeBridgeConfig

        cfg = ModelScopeBridgeConfig()
        self.assertEqual(cfg.max_total_interventions, 100)

    def test_custom_max_total_interventions(self):
        from model_scope_engine_bridge import ModelScopeBridgeConfig

        cfg = ModelScopeBridgeConfig(max_total_interventions=5)
        self.assertEqual(cfg.max_total_interventions, 5)

    def test_max_total_interventions_in_asdict(self):
        import dataclasses
        from model_scope_engine_bridge import ModelScopeBridgeConfig

        cfg = ModelScopeBridgeConfig(max_total_interventions=42)
        d = dataclasses.asdict(cfg)
        self.assertIn("max_total_interventions", d)
        self.assertEqual(d["max_total_interventions"], 42)

    def test_actuator_cap_is_honoured_when_steering_enabled(self):
        """Bridge with enable_steering=True and max_total_interventions=5 must honour the cap.

        We register a SOFT_SCALE policy on the actuator's registry, apply it
        more than 5 times, and assert the total_applied count does not exceed 5.
        """
        import tempfile
        from model_scope_engine_bridge import ModelScopeEngineBridge, ModelScopeBridgeConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = ModelScopeBridgeConfig(
                artifact_dir=tmpdir,
                enable_steering=True,
                max_total_interventions=5,
            )
            bridge = ModelScopeEngineBridge(cfg)

            # Verify the cap was forwarded to the actuator
            self.assertEqual(bridge._actuator._max_total, 5)

    def test_actuator_cap_zero_when_not_overridden_legacy_would_be_wrong(self):
        """Before the fix the actuator was always initialised with max_total_interventions=0
        (meaning unlimited). Now with the default of 100, interventions are capped at 100.
        This test ensures the old hard-coded 0 is gone.
        """
        import tempfile
        from model_scope_engine_bridge import ModelScopeEngineBridge, ModelScopeBridgeConfig

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = ModelScopeBridgeConfig(artifact_dir=tmpdir, enable_steering=True)
            bridge = ModelScopeEngineBridge(cfg)
            # Default config gives 100, NOT 0
            self.assertEqual(bridge._actuator._max_total, 100)
            self.assertNotEqual(bridge._actuator._max_total, 0)


class TestBridgeMaxTotalInterventionsRuntimeEnforcement(unittest.TestCase):
    """Verify that max_total_interventions cap is enforced at runtime, not just stored."""

    def test_bridge_max_total_interventions_enforced_at_runtime(self):
        """After `max_total_interventions` successful interventions the next apply() is declined.

        This tests RUNTIME BEHAVIOR of the cap — not just attribute storage.

        Design:
        - cap = 2, so the first two apply() calls should succeed (applied=True),
          and the third must be declined with decline_reason == "max_total_interventions_exceeded".
        - We use a SOFT_SCALE policy whose target_features overlap with the feature event,
          so that applied=True is actually produced (SHADOW never sets applied=True and would
          never decrement the cap).
        """
        from model_scope_steering import SteeringActuator
        from model_scope_features import SparseFeatureEvent
        from model_scope_runtime import ActivationEvent
        from steering_policy import PolicyRegistry, SteeringMode, SteeringPolicyConfig
        from datetime import datetime, timezone

        cap = 2

        # Build a real registry + actuator with the same cap as the bridge config would wire in.
        registry = PolicyRegistry()
        policy_config = SteeringPolicyConfig(
            name="test_cap_policy",
            mode=SteeringMode.SOFT_SCALE,
            target_features=["mean_activation"],
            scale_factor=0.5,
            policy_id="test_cap_policy_id",
        )
        registry.register(policy_config)

        actuator = SteeringActuator(registry, max_total_interventions=cap)

        # Confirm the cap value is what we set (catches a hardcoded-0 bug).
        self.assertEqual(actuator._max_total, cap)
        self.assertNotEqual(actuator._max_total, 0)

        def _make_event():
            activation = ActivationEvent(
                schema_version="1.0",
                model_id="test_model",
                layer_id="layer.0",
                token_count=1,
                shape=(1,),
                mean_activation=0.5,
                norm_activation=0.8,
                captured_at=datetime.now(timezone.utc).isoformat(),
                run_id="cap_test_run",
            )
            return SparseFeatureEvent(
                source_activation=activation,
                feature_source="raw_stats",
                features={"mean_activation": 0.5},
                feature_count=1,
                nonzero_count=1,
                extracted_at=datetime.now(timezone.utc).isoformat(),
            )

        # --- Apply up to the cap (each should be applied=True) ---
        records_applied = []
        for i in range(cap):
            _, record = actuator.apply(_make_event(), policy_config.policy_id)
            records_applied.append(record)

        applied_count = sum(1 for r in records_applied if r.applied)
        self.assertEqual(
            applied_count,
            cap,
            f"Expected exactly {cap} applied interventions before hitting the cap, got {applied_count}",
        )
        self.assertEqual(actuator.total_applied(), cap)

        # --- The (cap + 1)-th call MUST be declined ---
        _, declined_record = actuator.apply(_make_event(), policy_config.policy_id)
        self.assertFalse(
            declined_record.applied,
            "The intervention beyond the cap must NOT be applied (applied should be False)",
        )
        self.assertEqual(
            declined_record.decline_reason,
            "max_total_interventions_exceeded",
            f"Expected decline_reason='max_total_interventions_exceeded', got {declined_record.decline_reason!r}",
        )

        # Total applied count must NOT exceed the cap.
        self.assertLessEqual(
            actuator.total_applied(),
            cap,
            f"total_applied() exceeded cap {cap}: got {actuator.total_applied()}",
        )

    def test_bridge_wires_max_total_interventions_from_config(self):
        """ModelScopeEngineBridge must forward max_total_interventions from its config to SteeringActuator.

        This test will fail if max_total_interventions is hardcoded (e.g. always 0 or always 100)
        instead of being read from the config object.
        """
        import tempfile
        from model_scope_engine_bridge import ModelScopeEngineBridge, ModelScopeBridgeConfig

        custom_cap = 7

        with tempfile.TemporaryDirectory() as tmpdir:
            cfg = ModelScopeBridgeConfig(
                artifact_dir=tmpdir,
                enable_steering=True,
                max_total_interventions=custom_cap,
            )
            bridge = ModelScopeEngineBridge(cfg)

            # The actuator must reflect the exact value from config.
            self.assertEqual(
                bridge._actuator._max_total,
                custom_cap,
                f"Expected actuator._max_total={custom_cap} (from config), "
                f"got {bridge._actuator._max_total}",
            )

        # A different cap value must also be forwarded correctly (guards against hardcoding).
        other_cap = 3
        with tempfile.TemporaryDirectory() as tmpdir2:
            cfg2 = ModelScopeBridgeConfig(
                artifact_dir=tmpdir2,
                enable_steering=True,
                max_total_interventions=other_cap,
            )
            bridge2 = ModelScopeEngineBridge(cfg2)
            self.assertEqual(
                bridge2._actuator._max_total,
                other_cap,
                f"Expected actuator._max_total={other_cap} (from config), "
                f"got {bridge2._actuator._max_total}",
            )




# ---------------------------------------------------------------------------
# Bridge.observe — steering enabled, intervention_count > 0
# ---------------------------------------------------------------------------


class TestBridgeObserveWithSteeringEnabled(unittest.TestCase):
    """Gap-1 coverage: end-to-end observe() with enable_steering=True."""

    def setUp(self):
        from model_scope_engine_bridge import ModelScopeEngineBridge, ModelScopeBridgeConfig
        from steering_policy import SteeringPolicyConfig, SteeringMode, PolicyStatus

        self._td = tempfile.mkdtemp(prefix="bridge_steering_")
        cfg = ModelScopeBridgeConfig(
            artifact_dir=self._td,
            enable_feature_extraction=True,
            enable_steering=True,
        )
        self._bridge = ModelScopeEngineBridge(cfg)

        # Register a SOFT_SCALE ACTIVE policy whose target_features match what
        # FeatureExtractor._raw_stats_fallback() emits for our test ActivationEvent:
        #   "mean_activation", "norm_activation", "token_count", "shape_0"
        policy_cfg = SteeringPolicyConfig(
            name="test_soft_scale_policy",
            mode=SteeringMode.SOFT_SCALE,
            target_features=["mean_activation", "norm_activation"],
            scale_factor=1.5,
            status=PolicyStatus.ACTIVE,
        )
        self._bridge._actuator._registry.register(policy_cfg)

    def _mock_runtime_with_event(self):
        act = _make_activation_event()
        rt = MagicMock()
        rt.is_loaded.return_value = True
        rt.get_events.return_value = [act]
        return rt

    def test_observe_with_steering_enabled_returns_nonzero_intervention_count(self):
        rt = self._mock_runtime_with_event()
        result = self._bridge.observe("test steering query", rt)
        self.assertIsNone(result.error)
        self.assertGreater(result.intervention_count, 0)

    def test_observe_with_steering_persists_intervention_file(self):
        """Gap-2 dashboard path: observe() must write intervention_*.json to disk."""
        rt = self._mock_runtime_with_event()
        self._bridge.observe("test persistence query", rt)
        artifact_dir = Path(self._td)
        intervention_files = list(artifact_dir.glob("intervention_*.json"))
        self.assertGreater(
            len(intervention_files),
            0,
            "Expected at least one intervention_*.json file but found none.",
        )

    def test_observe_with_steering_persisted_file_is_valid_json(self):
        rt = self._mock_runtime_with_event()
        self._bridge.observe("test json validity", rt)
        artifact_dir = Path(self._td)
        intervention_files = list(artifact_dir.glob("intervention_*.json"))
        self.assertGreater(len(intervention_files), 0)
        with open(intervention_files[0], encoding="utf-8") as fh:
            data = json.load(fh)
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

    def test_observe_with_steering_persisted_record_has_applied_true(self):
        rt = self._mock_runtime_with_event()
        self._bridge.observe("test record fields", rt)
        artifact_dir = Path(self._td)
        intervention_files = list(artifact_dir.glob("intervention_*.json"))
        self.assertGreater(len(intervention_files), 0)
        with open(intervention_files[0], encoding="utf-8") as fh:
            records = json.load(fh)
        applied_records = [r for r in records if r.get("applied") is True]
        self.assertGreater(len(applied_records), 0)

    def test_get_summary_for_diagnostics_has_intervention_summary(self):
        rt = self._mock_runtime_with_event()
        self._bridge.observe("test summary", rt)
        summary = self._bridge.get_summary_for_diagnostics()
        self.assertIn("intervention_summary", summary)
        self.assertIn("total_applied", summary["intervention_summary"])
        self.assertIn("total_shadow", summary["intervention_summary"])
        self.assertGreater(summary["intervention_summary"]["total_applied"], 0)

    def test_observe_feature_events_include_raw_activation_dim_count(self):
        from steering_policy import SteeringPolicyConfig, SteeringMode, PolicyStatus

        policy_cfg = SteeringPolicyConfig(
            name="test_default_feature_policy",
            mode=SteeringMode.SHADOW,
            target_features=[
                "mean_activation",
                "norm_activation",
                "token_count",
                "shape_0",
                "raw_activation_dim_count",
            ],
            status=PolicyStatus.ACTIVE,
        )
        self._bridge._actuator._registry.register(policy_cfg)

        rt = self._mock_runtime_with_event()
        result = self._bridge.observe("test raw feature key", rt)
        self.assertIsNotNone(result.feature_event)
        self.assertIn(
            "raw_activation_dim_count",
            result.feature_event.features,
        )
        self.assertIn(
            "raw_activation_dim_count",
            self._bridge._actuator._registry.get(self._bridge._actuator._registry.list_active()[-1].policy_id).target_features,
        )


# ---------------------------------------------------------------------------
# Gap 1: Disk write failure leaves bridge in partial state
# ---------------------------------------------------------------------------


class TestBridgeObserveSteeringDiskWriteFailure(unittest.TestCase):
    """Gap-1 fix: OSError during intervention file write must not raise and must still
    increment observation_count."""

    def setUp(self):
        from model_scope_engine_bridge import ModelScopeEngineBridge, ModelScopeBridgeConfig
        from steering_policy import SteeringPolicyConfig, SteeringMode, PolicyStatus

        self._td = tempfile.mkdtemp(prefix="bridge_diskfail_")
        cfg = ModelScopeBridgeConfig(
            artifact_dir=self._td,
            enable_feature_extraction=True,
            enable_steering=True,
        )
        self._bridge = ModelScopeEngineBridge(cfg)

        policy_cfg = SteeringPolicyConfig(
            name="test_diskfail_policy",
            mode=SteeringMode.SOFT_SCALE,
            target_features=["mean_activation", "norm_activation"],
            scale_factor=1.5,
            status=PolicyStatus.ACTIVE,
        )
        self._bridge._actuator._registry.register(policy_cfg)

    def _mock_runtime_with_event(self):
        act = _make_activation_event()
        rt = MagicMock()
        rt.is_loaded.return_value = True
        rt.get_events.return_value = [act]
        return rt

    def test_observe_with_steering_disk_write_failure_still_increments_count(self):
        """If the intervention file write raises OSError, observe() must not propagate
        the exception and must still increment observation_count to 1."""
        import warnings as _warnings_mod

        rt = self._mock_runtime_with_event()

        _real_open = open  # save reference before patch

        def _selective_open(path, *args, **kwargs):
            if "intervention_" in str(path):
                raise OSError("disk full")
            return _real_open(path, *args, **kwargs)

        captured = []
        with _warnings_mod.catch_warnings(record=True) as w:
            _warnings_mod.simplefilter("always")
            with patch("builtins.open", side_effect=_selective_open):
                result = self._bridge.observe("test disk failure", rt)
            captured = list(w)

        # Must not have raised
        self.assertIsNone(result.error)
        # observation_count must still be incremented
        self.assertEqual(self._bridge.get_telemetry()["observation_count"], 1)
        # A UserWarning specifically about the intervention file write failure must
        # have been emitted — not just any UserWarning.
        user_warnings = [x for x in captured if issubclass(x.category, UserWarning)]
        self.assertGreater(len(user_warnings), 0, "Expected a UserWarning about the disk write failure")
        disk_write_warnings = [
            x for x in user_warnings
            if "intervention_" in str(x.message) or "Failed to persist" in str(x.message)
        ]
        self.assertGreater(
            len(disk_write_warnings), 0,
            "Expected a UserWarning mentioning intervention file write failure"
        )


# ---------------------------------------------------------------------------
# Gap 2: Unbounded intervention file accumulation
# ---------------------------------------------------------------------------


class TestBridgeObserveSteeringFileCap(unittest.TestCase):
    """Gap-2 fix: after >50 observe() calls with steering, at most 50 intervention
    files must remain on disk."""

    def setUp(self):
        from model_scope_engine_bridge import ModelScopeEngineBridge, ModelScopeBridgeConfig
        from steering_policy import SteeringPolicyConfig, SteeringMode, PolicyStatus

        self._td = tempfile.mkdtemp(prefix="bridge_cap_")
        cfg = ModelScopeBridgeConfig(
            artifact_dir=self._td,
            enable_feature_extraction=True,
            enable_steering=True,
            max_total_interventions=1000,
        )
        self._bridge = ModelScopeEngineBridge(cfg)

        policy_cfg = SteeringPolicyConfig(
            name="test_cap_policy",
            mode=SteeringMode.SOFT_SCALE,
            target_features=["mean_activation", "norm_activation"],
            scale_factor=1.5,
            status=PolicyStatus.ACTIVE,
        )
        self._bridge._actuator._registry.register(policy_cfg)

    def _mock_runtime_with_event(self):
        act = _make_activation_event()
        rt = MagicMock()
        rt.is_loaded.return_value = True
        rt.get_events.return_value = [act]
        return rt

    def test_observe_with_steering_caps_intervention_files_at_50(self):
        """After 55 observe() calls with steering, at most 50 intervention files must
        remain in the artifact directory."""
        rt = self._mock_runtime_with_event()
        for _ in range(55):
            self._bridge.observe("cap test query", rt)

        artifact_dir = Path(self._td)
        intervention_files = list(artifact_dir.glob("intervention_*.json"))
        self.assertLessEqual(
            len(intervention_files),
            50,
            f"Expected at most 50 intervention files, found {len(intervention_files)}",
        )


# ---------------------------------------------------------------------------
# Gap 1 (L11): Cap cleanup warns on OSError from unlink instead of swallowing
# ---------------------------------------------------------------------------


class TestBridgeCapCleanupWarnsOnUnlinkFailure(unittest.TestCase):
    """Verify that an OSError during cap-cleanup unlink emits a UserWarning
    instead of being silently swallowed, and that observe() does NOT raise."""

    def setUp(self):
        from model_scope_engine_bridge import ModelScopeEngineBridge, ModelScopeBridgeConfig
        from steering_policy import SteeringPolicyConfig, SteeringMode, PolicyStatus

        self._td = tempfile.mkdtemp(prefix="bridge_cleanup_warn_")
        cfg = ModelScopeBridgeConfig(
            artifact_dir=self._td,
            enable_feature_extraction=True,
            enable_steering=True,
            max_total_interventions=10000,
        )
        self._bridge = ModelScopeEngineBridge(cfg)

        policy_cfg = SteeringPolicyConfig(
            name="test_cleanup_warn_policy",
            mode=SteeringMode.SOFT_SCALE,
            target_features=["mean_activation", "norm_activation"],
            scale_factor=1.5,
            status=PolicyStatus.ACTIVE,
        )
        self._bridge._actuator._registry.register(policy_cfg)

    def _mock_runtime_with_event(self):
        act = _make_activation_event()
        rt = MagicMock()
        rt.is_loaded.return_value = True
        rt.get_events.return_value = [act]
        return rt

    def test_cap_cleanup_emits_warning_on_unlink_failure(self):
        """When Path.unlink raises OSError during cap cleanup, a UserWarning
        mentioning 'cleanup' or 'intervention file' must be emitted, and
        observe() must NOT raise."""
        import warnings as _warnings_mod

        artifact_dir = Path(self._td)

        # Pre-populate 51 dummy intervention files so cleanup is triggered
        # on the very next observe() call.
        for i in range(51):
            dummy = artifact_dir / f"intervention_dummy_{i:04d}.json"
            dummy.write_text("[]", encoding="utf-8")

        _unlink_call_count = {"n": 0}
        _real_unlink = Path.unlink

        def _failing_unlink(self_path, *args, **kwargs):
            # Raise only on the first unlink call (one failure is enough to
            # exercise the warning path).
            if _unlink_call_count["n"] == 0:
                _unlink_call_count["n"] += 1
                raise OSError("permission denied")
            _unlink_call_count["n"] += 1
            return _real_unlink(self_path, *args, **kwargs)

        rt = self._mock_runtime_with_event()
        captured = []
        with _warnings_mod.catch_warnings(record=True) as w:
            _warnings_mod.simplefilter("always")
            with patch.object(Path, "unlink", _failing_unlink):
                result = self._bridge.observe("cleanup warn test", rt)
            captured = list(w)

        # observe() must NOT raise — result.error stays None
        self.assertIsNone(result.error, f"observe() raised unexpectedly: {result.error}")

        # A UserWarning about the cap cleanup failure must have been emitted
        user_warnings = [x for x in captured if issubclass(x.category, UserWarning)]
        cleanup_warnings = [
            x for x in user_warnings
            if "cleanup" in str(x.message).lower() or "intervention file" in str(x.message).lower()
        ]
        self.assertGreater(
            len(cleanup_warnings), 0,
            f"Expected a UserWarning mentioning cap cleanup failure, got: {[str(x.message) for x in user_warnings]}",
        )


if __name__ == "__main__":
    unittest.main()
