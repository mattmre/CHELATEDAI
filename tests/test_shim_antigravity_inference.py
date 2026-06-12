"""SHIM-CD-01: run_inference pre-retrieval seam insertion test."""

from __future__ import annotations

import os
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np

from antigravity_engine import AntigravityEngine


class TestShimAntigravityRunInferenceInferenceSeam(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = AntigravityEngine.__new__(AntigravityEngine)
        self.engine.vector_size = 4
        self.engine.mode = "local"
        self.engine.model_name = "test-model"
        self.engine.chelation_threshold = 1.0
        self.engine.use_quantization = False
        self.engine.use_centering = False
        self.engine._adaptive_threshold_enabled = False
        self.engine._adaptive_threshold_lock = __import__("threading").Lock()
        self.engine._runtime_telemetry = {}
        self.engine.logger = MagicMock()
        self.engine.embed = MagicMock(return_value=np.array([[1.0, 2.0, 3.0, 4.0]]))
        self.engine._tts_pipeline = None
        self.engine._research_pre_retrieval_stall = 0
        self.engine._research_post_embed_stall = 0
        self.engine._research_chelation_stall = 0
        self.engine._last_research_shim_meta = None
        self.engine.collection_name = "test"
        self.engine._model_scope_feature_event = None
        self.engine._last_model_scope_artifact = {}
        self.engine._last_model_scope_feature_event = None
        self.engine._model_scope_scope = None
        self.engine._model_scope_observation_active = False
        self.engine._query_reformulation_active = False
        self.engine._adapter_routing_active = False
        self.engine._adapter_router = None
        self.engine._last_embedding_norms = {}
        self.engine._observe_model_scope_query = MagicMock(return_value=None)
        self.engine._last_runtime_diagnostics = None
        self.engine._runtime_json_safe = AntigravityEngine._runtime_json_safe
        self.engine._record_runtime_diagnostics = MagicMock(side_effect=AntigravityEngine._record_runtime_diagnostics.__get__(self.engine))
        self.engine._build_runtime_diagnostics = MagicMock(
            side_effect=AntigravityEngine._build_runtime_diagnostics.__get__(self.engine)
        )
        self.engine._select_retrieval_policy = MagicMock(
            side_effect=AntigravityEngine._select_retrieval_policy.__get__(self.engine)
        )
        self.engine.qdrant = MagicMock()
        hit = SimpleNamespace(
            id="doc-1",
            vector=np.array([0.5, 0.5, 0.5, 0.5]),
        )
        self.engine.qdrant.query_points.return_value = SimpleNamespace(points=[hit])

    def tearDown(self) -> None:
        os.environ.pop("CHELATED_SHIM_RESEARCH", None)
        os.environ.pop("CHELATED_SHIM_PROMOTED", None)

    @patch("antigravity_engine.research_enabled", return_value=True)
    @patch("antigravity_engine.promoted_sip_apply")
    @patch("antigravity_engine.research_preflight_metadata")
    def test_run_inference_applies_promoted_sip_when_enabled(
        self,
        mock_preflight_metadata: MagicMock,
        mock_promoted_apply: MagicMock,
        _enabled: MagicMock,
    ) -> None:
        mock_promoted_apply.return_value = (np.array([2.0, 3.0, 4.0, 5.0]), {"promoted_sip_applied": True})
        preflight_calls = []

        def _fake_preflight_metadata(*, seam: str, stall_count: int, extra=None) -> dict:
            preflight_calls.append({"seam": seam, "stall_count": stall_count, "extra": extra})
            return {"research_shim_guard": True, "research_stall_count": stall_count, "sip_seam": seam}

        mock_preflight_metadata.side_effect = _fake_preflight_metadata

        self.engine.run_inference("shim query")

        query_args = self.engine.qdrant.query_points.call_args.kwargs
        self.assertTrue(np.allclose(query_args["query"], np.array([2.0, 3.0, 4.0, 5.0])))
        seams = {call["seam"] for call in preflight_calls}
        self.assertIn("AntigravityEngine.pre_retrieval", seams)
        self.assertIn("AntigravityEngine.chelation_variance", seams)
        self.assertGreaterEqual(len(preflight_calls), 2)

        final_meta = self.engine._last_research_shim_meta
        self.assertEqual(final_meta["sip_seam"], "AntigravityEngine.chelation_variance")
        self.assertNotIn("promoted_sip_apply", final_meta)

    @patch("antigravity_engine.research_enabled", return_value=False)
    def test_run_inference_leaves_research_meta_unset_off(self, _enabled: MagicMock) -> None:
        self.engine.run_inference("shim query")
        self.assertIsNone(self.engine._last_research_shim_meta)


if __name__ == "__main__":
    unittest.main()
