"""Adaptive test-time compute budgeting for retrieval and agent campaigns."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Iterable, Mapping


COMPUTE_BUDGET_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ComputeBudgetPolicyConfig:
    """Thresholds for fail-closed adaptive compute escalation."""

    max_budget_units: int = 8
    base_budget_units: int = 1
    uncertainty_threshold: float = 0.45
    low_support_threshold: float = 0.55
    disagreement_threshold: float = 0.35
    safety_risk_threshold: float = 0.60
    hard_negative_risk_threshold: float = 0.50
    long_query_tokens: int = 18
    allow_self_consistency: bool = True
    allow_corrective_retrieval: bool = True
    allow_rerank: bool = True
    allow_abstain: bool = True

    def __post_init__(self) -> None:
        if self.max_budget_units < 1:
            raise ValueError("max_budget_units must be >= 1")
        if self.base_budget_units < 1:
            raise ValueError("base_budget_units must be >= 1")
        if self.base_budget_units > self.max_budget_units:
            raise ValueError("base_budget_units must be <= max_budget_units")
        for field_name in (
            "uncertainty_threshold",
            "low_support_threshold",
            "disagreement_threshold",
            "safety_risk_threshold",
            "hard_negative_risk_threshold",
        ):
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1]")
        if self.long_query_tokens < 1:
            raise ValueError("long_query_tokens must be >= 1")


def _score(value: Any, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def decide_compute_budget(
    signals: Mapping[str, Any],
    *,
    config: ComputeBudgetPolicyConfig | None = None,
) -> Dict[str, Any]:
    """Return a deterministic escalation plan for test-time compute."""

    cfg = config or ComputeBudgetPolicyConfig()
    uncertainty = _score(signals.get("uncertainty_score"))
    support = _score(signals.get("retrieval_support_score"), default=1.0)
    disagreement = _score(signals.get("evaluator_disagreement"))
    safety_risk = _score(signals.get("safety_risk"))
    hard_negative_risk = _score(signals.get("hard_negative_risk"))
    token_count = _int(signals.get("query_token_count"))

    actions = ["answer_direct"]
    reasons = []
    budget_units = int(cfg.base_budget_units)

    if support < cfg.low_support_threshold:
        reasons.append("low_retrieval_support")
        if cfg.allow_corrective_retrieval:
            actions.extend(["query_reformulate", "corrective_retrieve"])
            budget_units += 2

    if uncertainty >= cfg.uncertainty_threshold:
        reasons.append("high_uncertainty")
        if cfg.allow_self_consistency:
            actions.append("self_consistency_sample")
            budget_units += 2

    if disagreement >= cfg.disagreement_threshold:
        reasons.append("evaluator_disagreement")
        actions.append("multi_verifier_grade")
        budget_units += 1

    if hard_negative_risk >= cfg.hard_negative_risk_threshold:
        reasons.append("hard_negative_risk")
        if cfg.allow_rerank:
            actions.append("hard_negative_rerank")
            budget_units += 1

    if token_count >= cfg.long_query_tokens:
        reasons.append("long_query")
        if cfg.allow_rerank:
            actions.append("rerank_top_k")
            budget_units += 1

    if safety_risk >= cfg.safety_risk_threshold:
        reasons.append("safety_risk")
        actions.append("safety_verifier")
        budget_units += 1
        if cfg.allow_abstain:
            actions.append("abstain_if_unsupported")

    deduped_actions = []
    seen = set()
    for action in actions:
        if action not in seen:
            seen.add(action)
            deduped_actions.append(action)

    budget_units = min(int(cfg.max_budget_units), max(int(cfg.base_budget_units), budget_units))
    return {
        "schema_version": COMPUTE_BUDGET_SCHEMA_VERSION,
        "artifact_type": "compute_budget_decision",
        "budget_units": budget_units,
        "max_budget_units": int(cfg.max_budget_units),
        "actions": deduped_actions,
        "reasons": reasons or ["baseline_budget"],
        "escalated": bool(reasons),
        "signals": {
            "uncertainty_score": uncertainty,
            "retrieval_support_score": support,
            "evaluator_disagreement": disagreement,
            "safety_risk": safety_risk,
            "hard_negative_risk": hard_negative_risk,
            "query_token_count": token_count,
        },
        "config": asdict(cfg),
    }


def summarize_compute_budget_decisions(decisions: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    items = [dict(item) for item in decisions]
    action_counts: Dict[str, int] = {}
    reason_counts: Dict[str, int] = {}
    budget_units = []
    for item in items:
        try:
            budget_units.append(int(item.get("budget_units", 0)))
        except (TypeError, ValueError):
            budget_units.append(0)
        for action in item.get("actions", []) or []:
            action_counts[str(action)] = action_counts.get(str(action), 0) + 1
        for reason in item.get("reasons", []) or []:
            reason_counts[str(reason)] = reason_counts.get(str(reason), 0) + 1
    return {
        "schema_version": COMPUTE_BUDGET_SCHEMA_VERSION,
        "artifact_type": "compute_budget_summary",
        "decision_count": len(items),
        "escalated_count": sum(1 for item in items if bool(item.get("escalated"))),
        "mean_budget_units": (sum(budget_units) / len(budget_units)) if budget_units else 0.0,
        "max_budget_units": max(budget_units) if budget_units else 0,
        "action_counts": action_counts,
        "reason_counts": reason_counts,
    }
