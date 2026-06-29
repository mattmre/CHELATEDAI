"""Tests for steering_route_promotion.py — rung 10 A1b route-promotion gate
(quantization-survival composed with an actionable A1a rollback plan)."""
from __future__ import annotations

import json
import unittest

from model_scope_steering import InterventionRecord, RollbackPlan
from quantization_promotion_gate import QuantizationPromotionGate
from steering_route_promotion import evaluate_steering_route_promotion


def _plan(steps=1):
    return RollbackPlan(records=[InterventionRecord(policy_id=f"p{i}", applied=True) for i in range(steps)])


class TestSteeringRoutePromotion(unittest.TestCase):
    def test_promotable_when_quant_survives_and_rollback_actionable(self):
        d = evaluate_steering_route_promotion(fp32_fitness=1.0, quantized_fitness=0.9, rollback_plan=_plan(1))
        self.assertTrue(d.promotable)
        self.assertTrue(d.quantization_passed)
        self.assertTrue(d.rollback_actionable)
        self.assertEqual(d.rollback_step_count, 1)
        self.assertEqual(d.reasons, [])

    def test_not_promotable_when_quantization_fails(self):
        # quantized gain 0.5 / fp32 gain 1.0 = 0.5 < default 0.8 threshold
        d = evaluate_steering_route_promotion(fp32_fitness=1.0, quantized_fitness=0.5, rollback_plan=_plan(1))
        self.assertFalse(d.promotable)
        self.assertFalse(d.quantization_passed)
        self.assertIn("quantization_survival_failed", d.reasons)
        self.assertIn("retained_gain_below_threshold", d.reasons)  # surfaced from the quant gate

    def test_not_promotable_when_rollback_plan_missing(self):
        d = evaluate_steering_route_promotion(fp32_fitness=1.0, quantized_fitness=0.9, rollback_plan=None)
        self.assertFalse(d.promotable)
        self.assertTrue(d.quantization_passed)   # quantization is fine...
        self.assertFalse(d.rollback_actionable)  # ...but no rollback -> fail-closed
        self.assertIn("missing_rollback_plan", d.reasons)
        self.assertEqual(d.rollback_step_count, 0)

    def test_empty_plan_requires_reversible_steps_by_default(self):
        d = evaluate_steering_route_promotion(fp32_fitness=1.0, quantized_fitness=0.9, rollback_plan=_plan(0))
        self.assertFalse(d.promotable)
        self.assertIn("rollback_plan_has_no_reversible_steps", d.reasons)

    def test_empty_plan_tolerated_when_steps_not_required(self):
        d = evaluate_steering_route_promotion(
            fp32_fitness=1.0, quantized_fitness=0.9, rollback_plan=_plan(0), require_rollback_steps=False
        )
        self.assertTrue(d.promotable)
        self.assertTrue(d.rollback_actionable)

    def test_custom_gate_threshold_changes_quant_outcome(self):
        lax = QuantizationPromotionGate(retained_gain_threshold=0.4)
        d = evaluate_steering_route_promotion(
            fp32_fitness=1.0, quantized_fitness=0.5, rollback_plan=_plan(1), gate=lax
        )
        self.assertTrue(d.quantization_passed)  # 0.5 retained >= 0.4
        self.assertTrue(d.promotable)

    def test_string_path_is_not_an_actionable_rollback_plan(self):
        # A string rollback "path" is NOT an executable RollbackPlan -> the
        # isinstance guard rejects duck-typed impostors (fail-closed). This is the
        # whole point of A1b: an EXECUTABLE plan, not a string path.
        d = evaluate_steering_route_promotion(
            fp32_fitness=1.0, quantized_fitness=0.9, rollback_plan="path/to/rollback",
        )
        self.assertFalse(d.promotable)
        self.assertFalse(d.rollback_actionable)
        self.assertIn("missing_rollback_plan", d.reasons)
        self.assertEqual(d.rollback_step_count, 0)

    def test_to_dict_is_json_safe(self):
        d = evaluate_steering_route_promotion(
            fp32_fitness=1.0, quantized_fitness=0.9, rollback_plan=_plan(2)
        ).to_dict()
        json.dumps(d)
        self.assertEqual(d["record_type"], "steering_route_promotion_decision")
        self.assertTrue(d["promotable"])
        self.assertEqual(d["rollback_step_count"], 2)
        self.assertIsInstance(d["quantization"], dict)


if __name__ == "__main__":
    unittest.main()
