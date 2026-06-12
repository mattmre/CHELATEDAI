"""Rollback/no-op guard checks for production shim seams.

These tests verify that disabled research gates leave behavior unchanged and do not
invoke promoted-shim machinery.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

import numpy as np

from model_scope_runtime import ActivationEvent, LocalModelRuntime
from model_scope_engine_bridge import ModelScopeEngineBridge
from steering_policy import ModelScopeSteeringPolicy
from self_healing_chelation import SelfHealingChelationPlanner, SelfHealingChelationConfig
from computational_storage_poc import block_graph


class _NotLoadedRuntime:
    """Runtime stub that returns the "not loaded" path."""

    def is_loaded(self) -> bool:
        return False

    def get_events(self):
        return []


class TestShimProductionSeamsRollback(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("CHELATED_SHIM_RESEARCH", None)
        os.environ.pop("CHELATED_SHIM_PROMOTED", None)
        block_graph.clear_last_research_shim_meta()

    def _fake_runtime_with_events(self, events):
        runtime = LocalModelScopeRuntimeForTest(
            events=events,
            artifact_dir=tempfile.mkdtemp(prefix="ms-runtime-"),
        )
        return runtime

    @mock.patch("model_scope_runtime.promoted_sip_apply")
    @mock.patch("model_scope_runtime.research_enabled", return_value=False)
    def test_model_scope_runtime_off_does_not_apply_promoted_sip(
        self,
        _research_enabled: mock.MagicMock,
        mock_promoted: mock.MagicMock,
    ) -> None:
        event = ActivationEvent(
            schema_version="1.0",
            model_id="test-model",
            layer_id="model.layers.0",
            token_count=2,
            shape=(1, 2, 3),
            mean_activation=0.0,
            norm_activation=1.0,
            captured_at="2026-01-01T00:00:00+00:00",
            run_id="run-rollback",
        )

        runtime = self._fake_runtime_with_events([event])
        artifact = runtime.observe_text("rollback test", metadata={"source": "rollback-test"})

        mock_promoted.assert_not_called()
        self.assertIsNone(artifact["research_shim_meta"])
        self.assertIsNone(runtime.get_last_research_shim_meta())

    @mock.patch("steering_policy.promoted_sip_apply")
    @mock.patch("steering_policy.research_enabled", return_value=False)
    def test_steering_policy_off_does_not_emit_research_meta(
        self,
        _research_enabled: mock.MagicMock,
        mock_promoted: mock.MagicMock,
    ) -> None:
        policy = ModelScopeSteeringPolicy(name="rollback-policy", rules=[])
        policy.activate("active")

        mock_promoted.assert_not_called()
        self.assertIsNone(policy.get_last_research_shim_meta())

    @mock.patch("model_scope_engine_bridge.promoted_sip_apply")
    @mock.patch("model_scope_engine_bridge.research_enabled", return_value=False)
    def test_model_scope_bridge_off_does_not_emit_meta(
        self,
        _research_enabled: mock.MagicMock,
        mock_promoted: mock.MagicMock,
    ) -> None:
        bridge = ModelScopeEngineBridge()
        result = bridge.observe("rollback query", runtime=_NotLoadedRuntime())

        mock_promoted.assert_not_called()
        self.assertEqual(result.error, "runtime_not_loaded")
        self.assertIsNone(bridge.get_last_research_shim_meta())

    @mock.patch("computational_storage_poc.block_graph.promoted_sip_apply")
    @mock.patch("computational_storage_poc.block_graph.research_enabled", return_value=False)
    def test_block_graph_off_does_not_apply_promoted_sip(
        self,
        _research_enabled: mock.MagicMock,
        mock_promoted: mock.MagicMock,
    ) -> None:
        payload = block_graph.build_graph_payload([np.eye(512, dtype=np.float32)])
        out, processed = block_graph.run_block_graph(
            payload,
            np.ones(512, dtype=np.float32),
            trigger_offset=0,
            hidden_activation="identity",
        )

        mock_promoted.assert_not_called()
        self.assertIsNone(block_graph.get_last_research_shim_meta())
        self.assertEqual(processed, 1)
        self.assertTrue(np.allclose(out, np.ones(512, dtype=np.float32)))

    @mock.patch("self_healing_chelation.promoted_sip_apply")
    @mock.patch("self_healing_chelation.research_enabled", return_value=False)
    def test_self_healing_off_does_not_apply_promoted_sip(
        self,
        _research_enabled: mock.MagicMock,
        mock_promoted: mock.MagicMock,
    ) -> None:
        planner = SelfHealingChelationPlanner(
            config=SelfHealingChelationConfig(
                baseline_fitness=0.1,
                max_directives=1,
            )
        )

        with mock.patch.object(
            planner,
            "generate_directives",
            return_value=[],
        ), mock.patch.object(
            planner,
            "evaluate_directives",
            return_value=[],
        ):
            plan = planner.build_update_plan(
                context="rollback test",
                diagnostics={"structural_health": {"score": 0.6}},
                fitness=lambda directive: 1.0,
            )

        mock_promoted.assert_not_called()
        self.assertIsNone(planner.get_last_research_shim_meta())
        self.assertIsNone(plan.get("research_shim_meta"))


class LocalModelScopeRuntimeForTest(LocalModelRuntime):
    """Fixture runtime that bypasses model loading."""

    def __init__(self, events, artifact_dir: str):
        super().__init__(
            model_name="test-model",
            artifact_dir=artifact_dir,
            eager_load=False,
        )
        self._events = list(events)
        self._model = object()

    def run_inference(self, input_ids, run_id=None):  # pragma: no cover - test hook
        return list(self._events)


if __name__ == "__main__":
    unittest.main()
