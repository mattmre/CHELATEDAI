"""Production-wired control plane for Model-Scope steering (Phase II rung 10 — A1c).

Rung 10's gap was that the steering seams went live behind an env/CLI research guard
(``run_live_fire_diagnostics.py --enable-model-scope``) — a binary "on" with no
promotion authority. This is the production-wired control plane that replaces "is the
flag on?" with "did this route earn live mode?": a LIVE steering mode
(``SOFT_SCALE`` / ``SUPPRESSION``) is permitted only if the route passed the A1b
promotion gate (``SteeringRoutePromotionDecision.promotable`` — quantization-survival
AND an actionable A1a rollback). Otherwise the route is DOWNGRADED to the default-safe
mode (``SHADOW``: observe-only, no mutation). ``SHADOW`` is always allowed.

Default-safe and fail-closed: anything that is not provably promotable runs in SHADOW.
Pure decision logic over the A1b decision — no GPU, no engine, no model weights.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from steering_policy import SteeringMode

# Live modes mutate feature activations; SHADOW only observes.
LIVE_MODES = (SteeringMode.SOFT_SCALE, SteeringMode.SUPPRESSION)


@dataclass
class ProductionModeDecision:
    """The mode a route is actually allowed to run in production, with provenance."""

    requested_mode: SteeringMode
    resolved_mode: SteeringMode
    downgraded: bool
    promotable: bool
    reason: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "record_type": "production_steering_mode_decision",
            "requested_mode": self.requested_mode.value,
            "resolved_mode": self.resolved_mode.value,
            "downgraded": self.downgraded,
            "promotable": self.promotable,
            "reason": self.reason,
        }


def resolve_production_mode(
    requested_mode: SteeringMode,
    promotion_decision: Any,
    *,
    default_safe_mode: SteeringMode = SteeringMode.SHADOW,
) -> ProductionModeDecision:
    """Resolve the mode a route may run in production.

    - ``SHADOW`` requested -> always allowed (observe-only, never mutates).
    - A LIVE mode requested -> allowed ONLY if ``promotion_decision.promotable`` is
      truthy (an A1b ``SteeringRoutePromotionDecision``). Otherwise downgraded to
      ``default_safe_mode`` (SHADOW by default).

    ``promotion_decision`` may be None (treated as not promotable -> downgrade).
    """
    requested = SteeringMode(requested_mode)
    promotable = bool(getattr(promotion_decision, "promotable", False))

    if requested not in LIVE_MODES:
        return ProductionModeDecision(
            requested_mode=requested, resolved_mode=requested, downgraded=False,
            promotable=promotable, reason="shadow_or_non_live_mode_always_allowed",
        )
    if promotable:
        return ProductionModeDecision(
            requested_mode=requested, resolved_mode=requested, downgraded=False,
            promotable=True, reason="route_promoted_live_mode_permitted",
        )
    safe = SteeringMode(default_safe_mode)
    return ProductionModeDecision(
        requested_mode=requested, resolved_mode=safe, downgraded=(safe != requested),
        promotable=False, reason="route_not_promoted_downgraded_to_safe",
    )
