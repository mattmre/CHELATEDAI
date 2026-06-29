"""Tests for the executable steering rollback plan (Phase II rung 10 — A1a).

The per-record rollback primitive (rollback_feature_event) is tested in
test_model_scope_steering.py; this covers the plan-level executable rollback that a
production control plane uses: build a plan from a SteeringActuator's applied
interventions and execute it to restore the exact pre-steering feature values."""
from __future__ import annotations

import json
import unittest
from datetime import datetime, timezone

from model_scope_features import SparseFeatureEvent
from model_scope_runtime import ActivationEvent
from model_scope_steering import (
    SteeringActuator,
    build_rollback_plan,
    execute_rollback,
)
from steering_policy import PolicyRegistry, SteeringMode, SteeringPolicyConfig


def _activation():
    return ActivationEvent(
        schema_version="1.0", model_id="m", layer_id="l", token_count=4, shape=(1, 4),
        mean_activation=0.0, norm_activation=0.0,
        captured_at=datetime.now(timezone.utc).isoformat(), run_id="r",
    )


def _event(features):
    return SparseFeatureEvent(
        schema_version="1.0", source_activation=_activation(), feature_source="s",
        features=dict(features), feature_count=len(features),
        nonzero_count=sum(1 for v in features.values() if v != 0.0),
        extracted_at=datetime.now(timezone.utc).isoformat(),
    )


def _actuator(mode, target, scale=0.5, policy_id="p"):
    cfg = SteeringPolicyConfig(name=policy_id, policy_id=policy_id, mode=mode,
                               target_features=target, scale_factor=scale)
    reg = PolicyRegistry()
    reg.register(cfg)
    return cfg, SteeringActuator(reg)


class TestRollbackPlan(unittest.TestCase):
    def test_soft_scale_round_trip_restores_original(self):
        cfg, act = _actuator(SteeringMode.SOFT_SCALE, ["f0", "f1"], scale=0.5)
        ev = _event({"f0": 2.0, "f1": 4.0, "f2": 6.0})
        out, rec = act.apply(ev, cfg.policy_id)
        self.assertTrue(rec.applied)
        self.assertEqual(out.features["f0"], 1.0)  # scaled down
        plan = build_rollback_plan(act.get_records())
        self.assertEqual(plan.step_count, 1)
        restored = execute_rollback(plan, out)
        self.assertEqual(restored.features["f0"], 2.0)  # restored
        self.assertEqual(restored.features["f1"], 4.0)
        self.assertEqual(restored.features["f2"], 6.0)  # untouched feature preserved

    def test_suppression_round_trip(self):
        cfg, act = _actuator(SteeringMode.SUPPRESSION, ["f0"])
        out, _ = act.apply(_event({"f0": 5.0, "f1": 1.0}), cfg.policy_id)
        self.assertEqual(out.features["f0"], 0.0)
        restored = execute_rollback(build_rollback_plan(act.get_records()), out)
        self.assertEqual(restored.features["f0"], 5.0)

    def test_multi_intervention_reverse_order_restores_earliest_original(self):
        cfg_a = SteeringPolicyConfig(name="A", policy_id="A", mode=SteeringMode.SOFT_SCALE,
                                     target_features=["f0"], scale_factor=0.5)
        cfg_b = SteeringPolicyConfig(name="B", policy_id="B", mode=SteeringMode.SOFT_SCALE,
                                     target_features=["f0"], scale_factor=0.5)
        reg = PolicyRegistry()
        reg.register(cfg_a)
        reg.register(cfg_b)
        act = SteeringActuator(reg)
        out1, _ = act.apply(_event({"f0": 8.0}), "A")   # f0: 8 -> 4 (original 8)
        out2, _ = act.apply(out1, "B")                  # f0: 4 -> 2 (original 4)
        self.assertEqual(out2.features["f0"], 2.0)
        plan = build_rollback_plan(act.get_records())
        self.assertEqual(plan.step_count, 2)
        # Reverse order: undo B (back to 4), then A (restores the EARLIEST original 8).
        restored = execute_rollback(plan, out2)
        self.assertEqual(restored.features["f0"], 8.0)

    def test_shadow_and_empty_plan_are_noops(self):
        cfg, act = _actuator(SteeringMode.SHADOW, ["f0"])
        ev = _event({"f0": 2.0})
        act.apply(ev, cfg.policy_id)  # shadow -> applied=False, not in the plan
        plan = build_rollback_plan(act.get_records())
        self.assertEqual(plan.step_count, 0)
        restored = execute_rollback(plan, ev)
        self.assertEqual(restored.features["f0"], 2.0)
        self.assertIsNot(restored, ev)  # an unmodified COPY, not the same object

    def test_to_dict_is_json_safe(self):
        cfg, act = _actuator(SteeringMode.SOFT_SCALE, ["f0"])
        act.apply(_event({"f0": 2.0}), cfg.policy_id)
        plan = build_rollback_plan(act.get_records())
        d = plan.to_dict()
        json.dumps(d)
        self.assertEqual(d["record_type"], "steering_rollback_plan")
        self.assertEqual(d["step_count"], 1)
        self.assertEqual(d["policy_ids"], ["p"])

    def test_rollback_yields_feature_event_never_touches_weights(self):
        # Steering operates only on SparseFeatureEvent activations; the rollback
        # returns a SparseFeatureEvent and structurally cannot mutate base weights.
        cfg, act = _actuator(SteeringMode.SOFT_SCALE, ["f0"])
        out, _ = act.apply(_event({"f0": 2.0}), cfg.policy_id)
        restored = execute_rollback(build_rollback_plan(act.get_records()), out)
        self.assertIsInstance(restored, SparseFeatureEvent)


if __name__ == "__main__":
    unittest.main()
