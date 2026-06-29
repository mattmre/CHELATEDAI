"""Promotion gate for a Model-Scope steering route (Phase II rung 10 — A1b).

Composes two existing gates into the rung-10 promotion condition for a steering
route — "a promoted shim route must survive quantization and be rollback-backed":

1. **Quantization-survival** — the route's fitness GAIN must survive quantization
   (`QuantizationPromotionGate`: retained-gain ratio >= threshold).
2. **Actionable rollback** — the route must carry an EXECUTABLE rollback, i.e. an
   A1a `RollbackPlan` (not merely a string path). By default the plan must contain
   at least one reversible step (a live SOFT_SCALE/SUPPRESSION route that actually
   intervened); `require_rollback_steps=False` tolerates a zero-step plan (e.g. a
   route that matched nothing on the sample).

Fail-closed: a route is promotable only if BOTH hold. Pure composition over the
existing gates — no GPU, no engine, fully unit-testable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from model_scope_steering import RollbackPlan
from quantization_promotion_gate import QuantizationPromotionGate


@dataclass
class SteeringRoutePromotionDecision:
    """Fail-closed decision for promoting a steering route to live."""

    promotable: bool
    quantization_passed: bool
    rollback_actionable: bool
    rollback_step_count: int
    retained_gain_ratio: float
    reasons: List[str] = field(default_factory=list)
    quantization: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_type": "steering_route_promotion_decision",
            "promotable": self.promotable,
            "quantization_passed": self.quantization_passed,
            "rollback_actionable": self.rollback_actionable,
            "rollback_step_count": self.rollback_step_count,
            "retained_gain_ratio": self.retained_gain_ratio,
            "reasons": list(self.reasons),
            "quantization": self.quantization,
        }


def evaluate_steering_route_promotion(
    *,
    fp32_fitness: float,
    quantized_fitness: float,
    rollback_plan: Optional[RollbackPlan],
    baseline_fitness: float = 0.0,
    require_rollback_steps: bool = True,
    gate: Optional[QuantizationPromotionGate] = None,
) -> SteeringRoutePromotionDecision:
    """Return the fail-closed promotion decision for a steering route.

    ``rollback_plan`` is an A1a ``RollbackPlan`` (or None). ``gate`` lets the caller
    supply a configured ``QuantizationPromotionGate``; otherwise a default one is
    used. Promotable iff quantization-survival passes AND the rollback is actionable.
    """
    quant_gate = gate or QuantizationPromotionGate()
    quant = quant_gate.evaluate(fp32_fitness, quantized_fitness, baseline_fitness)

    has_plan = isinstance(rollback_plan, RollbackPlan)
    step_count = rollback_plan.step_count if has_plan else 0
    rollback_actionable = has_plan and (step_count > 0 if require_rollback_steps else True)

    reasons = list(quant.reasons)
    if not quant.passed:
        reasons.append("quantization_survival_failed")
    if not has_plan:
        reasons.append("missing_rollback_plan")
    elif require_rollback_steps and step_count == 0:
        reasons.append("rollback_plan_has_no_reversible_steps")

    return SteeringRoutePromotionDecision(
        promotable=bool(quant.passed and rollback_actionable),
        quantization_passed=bool(quant.passed),
        rollback_actionable=bool(rollback_actionable),
        rollback_step_count=int(step_count),
        retained_gain_ratio=float(quant.retained_gain_ratio),
        reasons=reasons,
        quantization=quant.to_dict(),
    )
