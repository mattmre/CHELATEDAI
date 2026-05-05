"""Comparator-first promotion gates over replayable evidence bundles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Mapping

from evidence_contract import EVIDENCE_SCHEMA_VERSION, summarize_evidence_bundle


PROMOTION_SCHEMA_VERSION = 1


@dataclass
class PromotionGateConfig:
    min_replay_score: float = 0.0
    min_holdout_score: float = 0.0
    min_evaluator_agreement: float = 0.0
    require_replay: bool = True
    require_holdout: bool = True
    require_safety: bool = True
    require_no_hard_negative_blockers: bool = True
    require_no_reward_overoptimization: bool = True

    def __post_init__(self) -> None:
        for field_name in ("min_replay_score", "min_holdout_score", "min_evaluator_agreement"):
            value = float(getattr(self, field_name))
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{field_name} must be in [0, 1]")


def _section_passed(report: Mapping[str, Any], key: str) -> bool | None:
    section = report.get(key)
    if not isinstance(section, Mapping):
        return None
    if "passed" in section:
        return bool(section.get("passed"))
    if "promotion_ready" in section:
        return bool(section.get("promotion_ready"))
    return None


def evaluate_promotion_candidate(
    *,
    candidate_id: str,
    evidence_bundle: Mapping[str, Any],
    comparator_report: Mapping[str, Any] | None = None,
    holdout_report: Mapping[str, Any] | None = None,
    safety_report: Mapping[str, Any] | None = None,
    hard_negative_report: Mapping[str, Any] | None = None,
    evaluator_report: Mapping[str, Any] | None = None,
    reward_report: Mapping[str, Any] | None = None,
    config: PromotionGateConfig | None = None,
) -> Dict[str, Any]:
    """Return a single fail-closed promotion decision for any candidate artifact."""

    cfg = config or PromotionGateConfig()
    reasons = []
    bundle_summary = summarize_evidence_bundle(evidence_bundle)
    if int(evidence_bundle.get("schema_version", -1)) != EVIDENCE_SCHEMA_VERSION:
        reasons.append("unsupported_evidence_bundle_schema")
    if int(bundle_summary.get("event_count", 0)) == 0:
        reasons.append("empty_evidence_bundle")

    replay_passed = _section_passed(comparator_report or {}, "replay")
    if replay_passed is None and comparator_report:
        replay_passed = bool(comparator_report.get("passed", comparator_report.get("comparison_count", 0) > 0))
    replay_score = float((comparator_report or {}).get("score", (comparator_report or {}).get("mean_score", 0.0)))
    if cfg.require_replay and replay_passed is None:
        reasons.append("missing_replay_report")
    if replay_passed is False:
        reasons.append("replay_failed")
    if replay_score < cfg.min_replay_score:
        reasons.append("replay_score_below_threshold")

    holdout_passed = _section_passed({"holdout": holdout_report or {}}, "holdout")
    holdout_score = float((holdout_report or {}).get("score", (holdout_report or {}).get("mean_score", 0.0)))
    if cfg.require_holdout and holdout_passed is None:
        reasons.append("missing_holdout_report")
    if holdout_passed is False:
        reasons.append("holdout_failed")
    if holdout_report is not None and holdout_score < cfg.min_holdout_score:
        reasons.append("holdout_score_below_threshold")

    safety_passed = _section_passed({"safety": safety_report or {}}, "safety")
    if cfg.require_safety and safety_passed is None:
        reasons.append("missing_safety_report")
    if safety_passed is False:
        reasons.append("safety_failed")

    blockers = int((hard_negative_report or {}).get("blocker_count", 0))
    if cfg.require_no_hard_negative_blockers and blockers > 0:
        reasons.append("hard_negative_blockers_present")

    agreement = float((evaluator_report or {}).get("agreement_score", 0.0))
    if evaluator_report is not None and agreement < cfg.min_evaluator_agreement:
        reasons.append("evaluator_agreement_below_threshold")

    reward_diverged = bool((reward_report or {}).get("heldout_divergence_detected", False))
    if cfg.require_no_reward_overoptimization and reward_diverged:
        reasons.append("reward_overoptimization_detected")

    return {
        "schema_version": PROMOTION_SCHEMA_VERSION,
        "artifact_type": "promotion_decision",
        "candidate_id": str(candidate_id),
        "promotion_ready": len(reasons) == 0,
        "reasons": reasons,
        "evidence_summary": bundle_summary,
        "replay_score": float(replay_score),
        "holdout_score": float(holdout_score),
        "evaluator_agreement": float(agreement),
        "hard_negative_blocker_count": blockers,
        "safety_passed": safety_passed,
        "config": asdict(cfg),
    }
