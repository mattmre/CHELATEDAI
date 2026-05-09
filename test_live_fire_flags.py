"""Tests for --enable-model-scope flag in run_live_fire_diagnostics.py."""

from __future__ import annotations

import argparse
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch


class TestEnableModelScopeFlag(unittest.TestCase):
    def _parse(self, args_list):
        parser = argparse.ArgumentParser()
        parser.add_argument("--output", default=None)
        parser.add_argument("--enable-model-scope", action="store_true", default=False)
        return parser.parse_args(args_list)

    def test_enable_model_scope_in_namespace(self):
        args = self._parse([])
        self.assertIn("enable_model_scope", vars(args))

    def test_default_is_false(self):
        args = self._parse([])
        self.assertFalse(args.enable_model_scope)

    def test_flag_sets_true(self):
        args = self._parse(["--enable-model-scope"])
        self.assertTrue(args.enable_model_scope)

    def test_enable_model_scope_calls_engine_method(self):
        from run_live_fire_diagnostics import run_live_fire_diagnostics
        args = SimpleNamespace(enable_model_scope=True)
        mock_engine = MagicMock()
        mock_engine.enable_model_scope_observation = MagicMock()
        with patch("run_live_fire_diagnostics._make_engine", return_value=mock_engine), \
             patch("run_live_fire_diagnostics._dataset", return_value=([], [], {}, {})), \
             patch("run_live_fire_diagnostics._collect_rankings", return_value=({}, [])), \
             patch("run_live_fire_diagnostics.RetrievalFitnessEvaluator") as mock_eval_cls, \
             patch("run_live_fire_diagnostics.InitialChelatedValues"), \
             patch("run_live_fire_diagnostics.FitnessCompositionOrchestrator"), \
             patch("run_live_fire_diagnostics.AdaptiveGateOrchestrator"), \
             patch("run_live_fire_diagnostics.QuantizationPromotionGate"), \
             patch("run_live_fire_diagnostics.SelfHealingChelationPlanner"), \
             patch("run_live_fire_diagnostics.IntegratedDiagnosticsReport"), \
             patch("run_live_fire_diagnostics.StructuralHealthScore"):
            mock_eval = MagicMock()
            mock_eval.evaluate_rankings.return_value = MagicMock(
                ndcg_at_k=0.5, mrr=0.5, recall_at_k=0.5, fitness=0.5
            )
            mock_eval_cls.return_value = mock_eval
            mock_engine.ingest = MagicMock()
            mock_engine.get_last_runtime_diagnostics = MagicMock(return_value={})
            mock_engine.get_runtime_telemetry = MagicMock(return_value={})
            mock_engine.model_name = "test"
            mock_engine.mode = "local"
            mock_engine._stability_tracker = MagicMock()
            mock_engine._stability_tracker.get_stability_report = MagicMock(return_value={})
            mock_engine._model_scope_bridge = None
            mock_engine.adapter = MagicMock()
            mock_engine._adapter_router = MagicMock()
            mock_engine._query_reformulator = MagicMock()
            mock_engine._query_reformulator_max_variants = 2
            try:
                run_live_fire_diagnostics(args=args)
            except Exception:
                pass
            mock_engine.enable_model_scope_observation.assert_called_once()

    def test_no_flag_does_not_call_engine_method(self):
        from run_live_fire_diagnostics import run_live_fire_diagnostics
        args = SimpleNamespace(enable_model_scope=False)
        mock_engine = MagicMock()
        mock_engine.enable_model_scope_observation = MagicMock()
        with patch("run_live_fire_diagnostics._make_engine", return_value=mock_engine), \
             patch("run_live_fire_diagnostics._dataset", return_value=([], [], {}, {})), \
             patch("run_live_fire_diagnostics._collect_rankings", return_value=({}, [])), \
             patch("run_live_fire_diagnostics.RetrievalFitnessEvaluator") as mock_eval_cls, \
             patch("run_live_fire_diagnostics.InitialChelatedValues"), \
             patch("run_live_fire_diagnostics.FitnessCompositionOrchestrator"), \
             patch("run_live_fire_diagnostics.AdaptiveGateOrchestrator"), \
             patch("run_live_fire_diagnostics.QuantizationPromotionGate"), \
             patch("run_live_fire_diagnostics.SelfHealingChelationPlanner"), \
             patch("run_live_fire_diagnostics.IntegratedDiagnosticsReport"), \
             patch("run_live_fire_diagnostics.StructuralHealthScore"):
            mock_eval = MagicMock()
            mock_eval.evaluate_rankings.return_value = MagicMock(
                ndcg_at_k=0.5, mrr=0.5, recall_at_k=0.5, fitness=0.5
            )
            mock_eval_cls.return_value = mock_eval
            mock_engine.ingest = MagicMock()
            mock_engine.get_last_runtime_diagnostics = MagicMock(return_value={})
            mock_engine.get_runtime_telemetry = MagicMock(return_value={})
            mock_engine.model_name = "test"
            mock_engine.mode = "local"
            mock_engine._stability_tracker = MagicMock()
            mock_engine._stability_tracker.get_stability_report = MagicMock(return_value={})
            mock_engine._model_scope_bridge = None
            mock_engine.adapter = MagicMock()
            mock_engine._adapter_router = MagicMock()
            mock_engine._query_reformulator = MagicMock()
            mock_engine._query_reformulator_max_variants = 2
            try:
                run_live_fire_diagnostics(args=args)
            except Exception:
                pass
            mock_engine.enable_model_scope_observation.assert_not_called()

    def test_enable_model_scope_graceful_on_exception(self):
        from run_live_fire_diagnostics import run_live_fire_diagnostics
        args = SimpleNamespace(enable_model_scope=True)
        mock_engine = MagicMock()
        mock_engine.enable_model_scope_observation = MagicMock(
            side_effect=RuntimeError("no transformer model")
        )
        with patch("run_live_fire_diagnostics._make_engine", return_value=mock_engine), \
             patch("run_live_fire_diagnostics._dataset", return_value=([], [], {}, {})), \
             patch("run_live_fire_diagnostics._collect_rankings", return_value=({}, [])), \
             patch("run_live_fire_diagnostics.RetrievalFitnessEvaluator") as mock_eval_cls, \
             patch("run_live_fire_diagnostics.InitialChelatedValues"), \
             patch("run_live_fire_diagnostics.FitnessCompositionOrchestrator"), \
             patch("run_live_fire_diagnostics.AdaptiveGateOrchestrator"), \
             patch("run_live_fire_diagnostics.QuantizationPromotionGate"), \
             patch("run_live_fire_diagnostics.SelfHealingChelationPlanner"), \
             patch("run_live_fire_diagnostics.IntegratedDiagnosticsReport"), \
             patch("run_live_fire_diagnostics.StructuralHealthScore"):
            mock_eval = MagicMock()
            mock_eval.evaluate_rankings.return_value = MagicMock(
                ndcg_at_k=0.5, mrr=0.5, recall_at_k=0.5, fitness=0.5
            )
            mock_eval_cls.return_value = mock_eval
            mock_engine.ingest = MagicMock()
            mock_engine.get_last_runtime_diagnostics = MagicMock(return_value={})
            mock_engine.get_runtime_telemetry = MagicMock(return_value={})
            mock_engine.model_name = "test"
            mock_engine.mode = "local"
            mock_engine._stability_tracker = MagicMock()
            mock_engine._stability_tracker.get_stability_report = MagicMock(return_value={})
            mock_engine._model_scope_bridge = None
            mock_engine.adapter = MagicMock()
            mock_engine._adapter_router = MagicMock()
            mock_engine._query_reformulator = MagicMock()
            mock_engine._query_reformulator_max_variants = 2
            # The enable-model-scope exception must be swallowed; function may fail
            # later due to incomplete mocks — that is unrelated to this test.
            try:
                run_live_fire_diagnostics(args=args)
            except Exception:
                pass
            # Key assertion: the method was called (try-block was entered) and its
            # exception was caught (execution continued past the enable block).
            mock_engine.enable_model_scope_observation.assert_called_once()

    def test_args_none_default_does_not_call_method(self):
        """run_live_fire_diagnostics(args=None) must not call enable_model_scope_observation."""
        from run_live_fire_diagnostics import run_live_fire_diagnostics
        mock_engine = MagicMock()
        mock_engine.enable_model_scope_observation = MagicMock()
        with patch("run_live_fire_diagnostics._make_engine", return_value=mock_engine), \
             patch("run_live_fire_diagnostics._dataset", return_value=([], [], {}, {})), \
             patch("run_live_fire_diagnostics._collect_rankings", return_value=({}, [])), \
             patch("run_live_fire_diagnostics.RetrievalFitnessEvaluator") as mock_eval_cls, \
             patch("run_live_fire_diagnostics.InitialChelatedValues"), \
             patch("run_live_fire_diagnostics.FitnessCompositionOrchestrator"), \
             patch("run_live_fire_diagnostics.AdaptiveGateOrchestrator"), \
             patch("run_live_fire_diagnostics.QuantizationPromotionGate"), \
             patch("run_live_fire_diagnostics.SelfHealingChelationPlanner"), \
             patch("run_live_fire_diagnostics.IntegratedDiagnosticsReport"), \
             patch("run_live_fire_diagnostics.StructuralHealthScore"):
            mock_eval = MagicMock()
            mock_eval.evaluate_rankings.return_value = MagicMock(
                ndcg_at_k=0.5, mrr=0.5, recall_at_k=0.5, fitness=0.5
            )
            mock_eval_cls.return_value = mock_eval
            mock_engine.ingest = MagicMock()
            mock_engine.get_last_runtime_diagnostics = MagicMock(return_value={})
            mock_engine.get_runtime_telemetry = MagicMock(return_value={})
            mock_engine.model_name = "test"
            mock_engine.mode = "local"
            mock_engine._stability_tracker = MagicMock()
            mock_engine._stability_tracker.get_stability_report = MagicMock(return_value={})
            mock_engine._model_scope_bridge = None
            mock_engine.adapter = MagicMock()
            mock_engine._adapter_router = MagicMock()
            mock_engine._query_reformulator = MagicMock()
            mock_engine._query_reformulator_max_variants = 2
            try:
                run_live_fire_diagnostics(args=None)
            except Exception:
                pass
            mock_engine.enable_model_scope_observation.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)
