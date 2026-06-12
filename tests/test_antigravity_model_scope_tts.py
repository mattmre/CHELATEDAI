"""Model-Scope -> TTS seam regression tests."""

from __future__ import annotations

from dataclasses import dataclass
import sys
import types
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from antigravity_engine import AntigravityEngine


@dataclass
class _FeatureEvent:
    feature_source: str
    feature_count: int
    features: dict
    source_activation: object
    extracted_at: str = "2026-01-01T00:00:00+00:00"
    nonzero_count: int = 1
    schema_version: str = "1.0"


@dataclass
class _ModelScopeObservationResult:
    run_id: str
    query: str
    activation_event: object
    feature_event: object
    intervention_count: int
    artifact_path: str | None
    elapsed_ms: float
    error: object = None


class _Result:
    def __init__(self, after_steering: np.ndarray):
        self.after_steering = after_steering
        self.after_translation = None
        self.after_transport = None
        self.steering_meta = {"was_steered": True}
        self.translation_result = None
        self.transport_result = None
        self.total_delta_norm = 0.0
        self.stages_applied = []


class TestAntigravityModelScopeTTS(unittest.TestCase):
    def _build_stub_engine(self, dim: int = 8) -> AntigravityEngine:
        engine = AntigravityEngine.__new__(AntigravityEngine)
        engine.vector_size = dim
        engine.logger = MagicMock()
        engine.adapter = None
        engine.model_name = "stub"
        engine._query_reformulation_active = False
        engine._adapter_routing_active = False
        engine._query_reformulator = None
        engine._adapter_router = None
        engine.use_quantization = False
        engine.use_centering = False
        engine.chelation_threshold = 0.5
        engine.collection_name = "stub_collection"
        engine._runtime_telemetry = {}
        engine._adaptive_threshold_lock = MagicMock()
        engine._adaptive_threshold_lock.__enter__.return_value = None
        engine._adaptive_threshold_lock.__exit__.return_value = None
        engine._research_post_embed_stall = 0
        engine._research_chelation_stall = 0
        engine._last_research_shim_meta = None

        point = MagicMock()
        point.id = "stub-id"
        point.vector = np.ones(dim, dtype=float)
        qdrant = MagicMock()
        qdrant_response = MagicMock()
        qdrant_response.points = [point]
        qdrant.query_points.return_value = qdrant_response
        engine.qdrant = qdrant
        return engine

    def test_run_inference_passes_model_scope_feature_event_to_tts(self) -> None:
        engine = self._build_stub_engine()
        feature_event = {"feature_source": "unit_test", "features": {"mean_activation": 1.0}, "feature_count": 1}
        model_scope_summary = {
            "feature_event": feature_event,
            "feature_event_present": True,
            "status": "observed",
        }

        captured = {}

        class _TtsPipeline:
            def apply(self, v, cluster_id=None, cluster_confidence=0.0, feature_event=None):
                captured["feature_event"] = feature_event
                return _Result(np.array(v, dtype=float))

        engine._tts_pipeline = _TtsPipeline()

        fake_dash = types.ModuleType("dashboard_server")
        fake_dash.update_tts_dashboard_state = MagicMock()
        fake_dash._tts_result_to_dict = lambda _r: {"ok": True}

        with patch.dict(sys.modules, {"dashboard_server": fake_dash}), patch.object(
            engine, "_observe_model_scope_query", return_value=model_scope_summary
        ), patch.object(
            engine, "_record_runtime_diagnostics", return_value=None
        ), patch.object(
            engine, "_build_runtime_diagnostics", return_value={"runtime": {"status": "ok"}}
        ), patch.object(
            engine, "_update_adaptive_threshold", return_value=None
        ), patch.object(
            engine, "_select_retrieval_policy", return_value={"policy": "FAST"}
        ):
            with patch.object(
                engine, "embed", return_value=np.ones((1, 8), dtype=float)
            ):
                engine.run_inference("model scope seam test")

        self.assertIs(captured.get("feature_event"), feature_event)

    def test_run_inference_without_model_scope_event_passes_none_to_tts(self) -> None:
        engine = self._build_stub_engine()

        captured = {}

        class _TtsPipeline:
            def apply(self, v, cluster_id=None, cluster_confidence=0.0, feature_event=None):
                captured["feature_event"] = feature_event
                return _Result(np.array(v, dtype=float))

        engine._tts_pipeline = _TtsPipeline()

        fake_dash = types.ModuleType("dashboard_server")
        fake_dash.update_tts_dashboard_state = MagicMock()
        fake_dash._tts_result_to_dict = lambda _r: {"ok": True}

        model_scope_summary = {"status": "observed", "feature_event_present": False}
        with patch.dict(sys.modules, {"dashboard_server": fake_dash}), patch.object(
            engine, "_observe_model_scope_query", return_value=model_scope_summary
        ), patch.object(
            engine, "_record_runtime_diagnostics", return_value=None
        ), patch.object(
            engine, "_build_runtime_diagnostics", return_value={"runtime": {"status": "ok"}}
        ), patch.object(
            engine, "_update_adaptive_threshold", return_value=None
        ), patch.object(
            engine, "_select_retrieval_policy", return_value={"policy": "FAST"}
        ):
            with patch.object(
                engine, "embed", return_value=np.ones((1, 8), dtype=float)
            ):
                engine.run_inference("no model scope event test")

        self.assertIsNone(captured.get("feature_event"))

    def test_observe_model_scope_query_bridge_returns_feature_event_object(self) -> None:
        engine = self._build_stub_engine()
        engine.logger.log_error = MagicMock()
        engine._model_scope_runtime = object()

        bridge_feature_event = _FeatureEvent(
            feature_source="unit_test",
            feature_count=1,
            nonzero_count=1,
            features={"mean_activation": 1.2},
            source_activation=object(),
        )
        bridge_result = _ModelScopeObservationResult(
            run_id="obs_1",
            query="bridge seam test",
            activation_event=object(),
            feature_event=bridge_feature_event,
            intervention_count=0,
            artifact_path="/tmp/obs.json",
            elapsed_ms=12.5,
        )

        class _Bridge:
            def observe(self, query, runtime):
                return bridge_result

        engine._model_scope_bridge = _Bridge()
        summary = engine._observe_model_scope_query("bridge seam test")

        self.assertEqual(summary.get("status"), "observed")
        self.assertTrue(summary.get("feature_event_present"))
        self.assertIs(engine._last_model_scope_feature_event, bridge_feature_event)

    def test_run_inference_bridge_observation_passes_raw_feature_event_to_tts(self) -> None:
        engine = self._build_stub_engine()
        feature_event = _FeatureEvent(
            feature_source="unit_test",
            feature_count=1,
            nonzero_count=1,
            features={"mean_activation": 1.2},
            source_activation=object(),
        )
        bridge_result = _ModelScopeObservationResult(
            run_id="obs_2",
            query="bridge seam run_inference",
            activation_event=object(),
            feature_event=feature_event,
            intervention_count=0,
            artifact_path="/tmp/obs.json",
            elapsed_ms=12.5,
        )

        class _Bridge:
            def observe(self, query, runtime):
                return bridge_result

        engine._model_scope_runtime = types.SimpleNamespace(
            model_name="bridge-model",
            describe_runtime=lambda: {"model_name": "bridge-model"},
        )
        engine._model_scope_bridge = _Bridge()

        captured = {}

        class _TtsPipeline:
            def apply(self, v, cluster_id=None, cluster_confidence=0.0, feature_event=None):
                captured["feature_event"] = feature_event
                return _Result(np.array(v, dtype=float))

        engine._tts_pipeline = _TtsPipeline()

        fake_dash = types.ModuleType("dashboard_server")
        fake_dash.update_tts_dashboard_state = MagicMock()
        fake_dash._tts_result_to_dict = lambda _r: {"ok": True}

        with patch.dict(sys.modules, {"dashboard_server": fake_dash}), patch.object(
            engine, "_record_runtime_diagnostics", return_value=None
        ), patch.object(
            engine, "_build_runtime_diagnostics", return_value={"runtime": {"status": "ok"}}
        ), patch.object(
            engine, "_update_adaptive_threshold", return_value=None
        ), patch.object(
            engine, "_select_retrieval_policy", return_value={"policy": "FAST"}
        ):
            with patch.object(
                engine, "embed", return_value=np.ones((1, 8), dtype=float)
            ):
                engine.run_inference("bridge seam run_inference")

        self.assertIs(captured.get("feature_event"), feature_event)


if __name__ == "__main__":
    unittest.main()
