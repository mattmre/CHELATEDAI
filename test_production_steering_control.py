"""Tests for production_steering_control.py — rung 10 A1c production control plane
(live steering gated on the A1b promotion decision; default-safe SHADOW otherwise)."""
from __future__ import annotations

import json
import unittest
from types import SimpleNamespace

from model_scope_steering import InterventionRecord, RollbackPlan
from production_steering_control import resolve_production_mode
from steering_policy import SteeringMode
from steering_route_promotion import evaluate_steering_route_promotion


def _promo(promotable):
    return SimpleNamespace(promotable=promotable)


class TestProductionSteeringControl(unittest.TestCase):
    def test_shadow_always_allowed(self):
        d = resolve_production_mode(SteeringMode.SHADOW, _promo(False))
        self.assertEqual(d.resolved_mode, SteeringMode.SHADOW)
        self.assertFalse(d.downgraded)

    def test_live_mode_permitted_when_promotable(self):
        for mode in (SteeringMode.SOFT_SCALE, SteeringMode.SUPPRESSION):
            d = resolve_production_mode(mode, _promo(True))
            self.assertEqual(d.resolved_mode, mode)
            self.assertFalse(d.downgraded)
            self.assertTrue(d.promotable)

    def test_live_mode_downgraded_to_shadow_when_not_promotable(self):
        for mode in (SteeringMode.SOFT_SCALE, SteeringMode.SUPPRESSION):
            d = resolve_production_mode(mode, _promo(False))
            self.assertEqual(d.resolved_mode, SteeringMode.SHADOW)  # default-safe
            self.assertTrue(d.downgraded)
            self.assertFalse(d.promotable)

    def test_none_decision_downgrades_fail_closed(self):
        d = resolve_production_mode(SteeringMode.SOFT_SCALE, None)
        self.assertEqual(d.resolved_mode, SteeringMode.SHADOW)
        self.assertTrue(d.downgraded)
        self.assertFalse(d.promotable)

    def test_integration_with_real_a1b_decision(self):
        plan = RollbackPlan(records=[InterventionRecord(policy_id="p", applied=True)])
        promoted = evaluate_steering_route_promotion(
            fp32_fitness=1.0, quantized_fitness=0.9, rollback_plan=plan)
        self.assertTrue(promoted.promotable)
        # A promoted route gets its requested live mode.
        self.assertEqual(
            resolve_production_mode(SteeringMode.SOFT_SCALE, promoted).resolved_mode,
            SteeringMode.SOFT_SCALE,
        )
        # A route that fails quantization is downgraded to SHADOW.
        rejected = evaluate_steering_route_promotion(
            fp32_fitness=1.0, quantized_fitness=0.5, rollback_plan=plan)
        self.assertFalse(rejected.promotable)
        d2 = resolve_production_mode(SteeringMode.SOFT_SCALE, rejected)
        self.assertEqual(d2.resolved_mode, SteeringMode.SHADOW)
        self.assertTrue(d2.downgraded)

    def test_to_dict_is_json_safe(self):
        d = resolve_production_mode(SteeringMode.SUPPRESSION, _promo(False)).to_dict()
        json.dumps(d)
        self.assertEqual(d["record_type"], "production_steering_mode_decision")
        self.assertEqual(d["requested_mode"], SteeringMode.SUPPRESSION.value)
        self.assertEqual(d["resolved_mode"], SteeringMode.SHADOW.value)
        self.assertTrue(d["downgraded"])


if __name__ == "__main__":
    unittest.main()
