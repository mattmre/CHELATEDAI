"""Small evaluator agreement and drift summaries for campaign reports."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Mapping, Sequence


EVALUATOR_FABRIC_SCHEMA_VERSION = 1


def summarize_evaluator_results(results: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    items = [dict(item) for item in results]
    passed = [bool(item.get("passed")) for item in items if "passed" in item]
    score_values = []
    for item in items:
        try:
            score_values.append(float(item.get("score")))
        except (TypeError, ValueError):
            continue
    pass_count = sum(1 for item in passed if item)
    fail_count = sum(1 for item in passed if not item)
    total_votes = pass_count + fail_count
    majority = None
    agreement = 0.0
    if total_votes:
        majority = pass_count >= fail_count
        agreement = max(pass_count, fail_count) / total_votes
    return {
        "schema_version": EVALUATOR_FABRIC_SCHEMA_VERSION,
        "artifact_type": "evaluator_fabric_summary",
        "evaluator_count": len(items),
        "vote_count": total_votes,
        "pass_count": pass_count,
        "fail_count": fail_count,
        "majority_passed": majority,
        "agreement_score": float(agreement),
        "mean_score": float(sum(score_values) / len(score_values)) if score_values else 0.0,
        "evaluator_ids": [str(item.get("evaluator_id") or item.get("name") or "unknown") for item in items],
    }


def grade_trace_events(
    events: Sequence[Mapping[str, Any]],
    *,
    required_surfaces: Iterable[str] | None = None,
    max_failure_count: int = 0,
) -> Dict[str, Any]:
    """Grade a full model/retrieval/tool trajectory at event level."""

    required = {str(item) for item in (required_surfaces or [])}
    seen_surfaces = {str(event.get("surface") or "unknown") for event in events}
    failures = []
    for index, event in enumerate(events):
        decision = str(event.get("decision") or "").lower()
        safety = event.get("safety_results")
        safety_failed = isinstance(safety, Mapping) and safety.get("passed") is False
        if decision in {"failed", "blocked", "unsafe", "unsupported"} or safety_failed:
            failures.append(
                {
                    "event_index": index,
                    "event_id": event.get("event_id"),
                    "surface": event.get("surface"),
                    "decision": event.get("decision"),
                    "failure_type": "event_failure",
                }
            )
    missing_surfaces = sorted(required - seen_surfaces)
    for surface in missing_surfaces:
        failures.append(
            {
                "event_index": None,
                "event_id": None,
                "surface": surface,
                "decision": None,
                "failure_type": "missing_required_surface",
            }
        )
    return {
        "schema_version": EVALUATOR_FABRIC_SCHEMA_VERSION,
        "artifact_type": "trace_event_grade",
        "event_count": len(events),
        "surfaces": sorted(seen_surfaces),
        "required_surfaces": sorted(required),
        "failure_count": len(failures),
        "failures": failures,
        "passed": len(failures) <= int(max_failure_count),
    }


def detect_reward_overoptimization(
    training_report: Mapping[str, Any],
    heldout_report: Mapping[str, Any],
    *,
    max_divergence: float = 0.10,
    min_training_gain: float = 0.0,
) -> Dict[str, Any]:
    """Flag train/evaluator reward gains that do not survive heldout checks."""

    train_gain = float(training_report.get("score", training_report.get("mean_score", 0.0)))
    heldout_gain = float(heldout_report.get("score", heldout_report.get("mean_score", 0.0)))
    divergence = train_gain - heldout_gain
    detected = train_gain > float(min_training_gain) and divergence > float(max_divergence)
    return {
        "schema_version": EVALUATOR_FABRIC_SCHEMA_VERSION,
        "artifact_type": "reward_overoptimization_report",
        "training_score": train_gain,
        "heldout_score": heldout_gain,
        "divergence": divergence,
        "max_divergence": float(max_divergence),
        "min_training_gain": float(min_training_gain),
        "heldout_divergence_detected": bool(detected),
        "passed": not detected,
    }
