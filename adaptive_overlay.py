"""Observation-only adaptive channel overlay records.

This module does not route or promote anything. It normalizes existing
Engine-Scope rows into channel-variation records so current gates, road-course
profiles, and future Model-Scope hooks can be compared through one vocabulary.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, Mapping, Sequence


ADAPTIVE_OVERLAY_SCHEMA_VERSION = 1

_CHANNEL_KEYWORDS = (
    ("attnres", "adapter"),
    ("reform", "query_reformulation"),
    ("mask", "mask_gate"),
    ("chelate", "chelation"),
    ("adaptive", "chelation"),
    ("guard", "guardrail"),
    ("baseline", "baseline"),
)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def stable_overlay_hash(payload: Any) -> str:
    encoded = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def infer_channel_type(row: Mapping[str, Any]) -> str:
    """Infer the broad control channel represented by an Engine-Scope row."""

    profile = str(row.get("profile") or "").lower()
    action = str(row.get("action") or "").lower()
    retrieval_intervention = str(row.get("retrieval_intervention") or "").lower()
    haystack = " ".join([profile, action, retrieval_intervention])
    for keyword, channel_type in _CHANNEL_KEYWORDS:
        if keyword in haystack:
            return channel_type
    if str(row.get("row_type") or "") == "mask_probe":
        return "mask_gate"
    return "engine_profile"


def infer_overlay_decision(row: Mapping[str, Any]) -> str:
    """Map existing row/fault state into overlay vocabulary without taking action."""

    explicit = row.get("decision") or row.get("gate_decision")
    if explicit:
        return str(explicit)
    fault_class = str(row.get("fault_class") or "")
    if fault_class == "actuator_active_positive":
        return "amplify_candidate"
    if fault_class == "actuator_active_negative":
        return "damp_candidate"
    if fault_class == "reference":
        return "protect_baseline"
    if fault_class in {"no_op_tied", "actuator_active_neutral"}:
        return "observe"
    if bool(row.get("promotion_blocker", False)):
        return "protect"
    return "observe"


def infer_protection_level(row: Mapping[str, Any]) -> str:
    profile = str(row.get("profile") or "")
    if profile == "baseline" or str(row.get("fault_class") or "") == "reference":
        return "frozen"
    if bool(row.get("promotion_blocker", False)):
        return "guarded"
    if row.get("safety_outcome") not in (None, "", "passed", "pass", True):
        return "safety_critical"
    return "mutable"


def infer_aggression_level(row: Mapping[str, Any]) -> str:
    action = str(row.get("action") or "").upper()
    if action in {"", "UNKNOWN", "FAST"}:
        return "observe"
    if action in {"CHELATE_ALWAYS"}:
        return "high"
    if action in {"CHELATE", "REFORMULATE"}:
        return "normal"
    if bool(row.get("gate_applied", False)):
        return "soft"
    return "observe"


def channel_variation_from_engine_scope_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Return an observation-only adaptive overlay record for one Engine-Scope row."""

    payload = dict(row)
    channel_type = infer_channel_type(payload)
    profile = str(payload.get("profile") or channel_type)
    query_id = None if payload.get("query_id") is None else str(payload.get("query_id"))
    metric_delta = payload.get("delta_ndcg_at_10", payload.get("selected_delta_ndcg_at_10"))
    active_negative = str(payload.get("fault_class") or "") == "actuator_active_negative"
    blocker = bool(payload.get("promotion_blocker", False))
    identity = {
        "task": payload.get("task"),
        "seed": payload.get("seed"),
        "query_id": query_id,
        "profile": profile,
        "row_type": payload.get("row_type"),
        "artifact_path": payload.get("_artifact_path"),
    }
    record = {
        "schema_version": ADAPTIVE_OVERLAY_SCHEMA_VERSION,
        "record_type": "channel_variation",
        "channel_id": profile,
        "channel_type": channel_type,
        "variation_id": f"var_{stable_overlay_hash(identity)}",
        "task": payload.get("task"),
        "seed": payload.get("seed"),
        "query_offset": payload.get("query_offset"),
        "query_id": query_id,
        "aggression_level": infer_aggression_level(payload),
        "protection_level": infer_protection_level(payload),
        "metric_delta": metric_delta,
        "active_negative_flags": ["actuator_active_negative"] if active_negative else [],
        "safety_flags": [str(payload["safety_outcome"])] if payload.get("safety_outcome") else [],
        "promotion_blocker": blocker,
        "decision": infer_overlay_decision(payload),
        "decision_reason": payload.get("fault_class") or payload.get("evaluator_outcome") or payload.get("decision"),
        "source_row_type": payload.get("row_type"),
        "source_family": payload.get("source_family"),
        "source_artifact_path": payload.get("_artifact_path"),
        "source_row": _json_safe(payload),
    }
    record["record_hash"] = f"overlay_{stable_overlay_hash(record)}"
    return record


def build_channel_variation_records(rows: Iterable[Mapping[str, Any]]) -> list[Dict[str, Any]]:
    return [channel_variation_from_engine_scope_row(row) for row in rows]


def build_adaptive_overlay_intake(
    *,
    query_id: str | None = None,
    task: str | None = None,
    channel_records: Iterable[Mapping[str, Any]] = (),
    context: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a replayable intake packet for observation-only overlay analysis."""

    records = [_json_safe(dict(record)) for record in channel_records]
    intake = {
        "schema_version": ADAPTIVE_OVERLAY_SCHEMA_VERSION,
        "record_type": "adaptive_overlay_intake",
        "query_id": query_id,
        "task": task,
        "channel_record_count": len(records),
        "channel_records": records,
        "context": _json_safe(context or {}),
    }
    intake["intake_id"] = f"intake_{stable_overlay_hash(intake)}"
    return intake


def summarize_channel_variations(records: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    records_list = [dict(record) for record in records]
    by_channel_type: Dict[str, int] = {}
    by_decision: Dict[str, int] = {}
    blockers = 0
    active_negatives = 0
    for record in records_list:
        channel_type = str(record.get("channel_type") or "unknown")
        decision = str(record.get("decision") or "unknown")
        by_channel_type[channel_type] = by_channel_type.get(channel_type, 0) + 1
        by_decision[decision] = by_decision.get(decision, 0) + 1
        blockers += int(bool(record.get("promotion_blocker", False)))
        active_negatives += int(bool(record.get("active_negative_flags")))
    return {
        "schema_version": ADAPTIVE_OVERLAY_SCHEMA_VERSION,
        "record_count": len(records_list),
        "channel_types": by_channel_type,
        "decisions": by_decision,
        "promotion_blockers": blockers,
        "active_negative_records": active_negatives,
    }


def _metric_delta(record: Mapping[str, Any]) -> float | None:
    value = record.get("metric_delta")
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _branch_group_key(record: Mapping[str, Any], keys: Sequence[str]) -> tuple[str, ...]:
    return tuple(str(record.get(key) or "") for key in keys)


def compute_branch_set_metrics(
    records: Iterable[Mapping[str, Any]],
    *,
    success_delta: float = 0.001,
    group_keys: Sequence[str] = ("task", "seed", "query_id"),
) -> Dict[str, Any]:
    """Compute HeavySkill-style branch metrics over engine channel variants.

    This is intentionally post-hoc and observation-only. It asks whether any
    non-baseline channel variant helped, how often branch search had an oracle
    win available, and how often that win was clean of blockers.
    """

    groups: Dict[tuple[str, ...], list[Dict[str, Any]]] = {}
    for record in records:
        item = dict(record)
        if _metric_delta(item) is None:
            continue
        groups.setdefault(_branch_group_key(item, group_keys), []).append(item)

    group_summaries = []
    pass_count = 0
    safe_pass_count = 0
    regression_count = 0
    total_best_delta = 0.0
    total_mean_delta = 0.0
    oracle_gap_sum = 0.0
    oracle_gap_count = 0

    for key, group_records in sorted(groups.items()):
        candidates = [
            record
            for record in group_records
            if str(record.get("channel_type") or "") != "baseline"
        ]
        if not candidates:
            candidates = group_records
        deltas = [float(_metric_delta(record) or 0.0) for record in candidates]
        best_index = max(range(len(candidates)), key=lambda index: deltas[index])
        best_record = candidates[best_index]
        best_delta = deltas[best_index]
        mean_delta = sum(deltas) / len(deltas)
        branch_passed = best_delta > success_delta
        safe_branch_passed = (
            branch_passed
            and not bool(best_record.get("promotion_blocker", False))
            and not bool(best_record.get("active_negative_flags"))
        )
        branch_regressed = any(delta < -success_delta for delta in deltas)
        selected_records = [
            record
            for record in candidates
            if str(record.get("decision") or "") in {"amplify_candidate", "route", "promote"}
        ]
        selected_delta = None
        if selected_records:
            selected_delta = max(float(_metric_delta(record) or 0.0) for record in selected_records)
            oracle_gap_sum += max(0.0, best_delta - selected_delta)
            oracle_gap_count += 1

        pass_count += int(branch_passed)
        safe_pass_count += int(safe_branch_passed)
        regression_count += int(branch_regressed)
        total_best_delta += best_delta
        total_mean_delta += mean_delta
        group_summary = {
            "group": {name: value for name, value in zip(group_keys, key)},
            "branch_count": len(candidates),
            "mean_delta": mean_delta,
            "best_delta": best_delta,
            "best_channel_id": best_record.get("channel_id"),
            "best_channel_type": best_record.get("channel_type"),
            "pass_at_k": branch_passed,
            "safe_pass_at_k": safe_branch_passed,
            "regressed_at_k": branch_regressed,
            "selected_delta": selected_delta,
            "oracle_gap": None if selected_delta is None else max(0.0, best_delta - selected_delta),
        }
        group_summaries.append(group_summary)

    group_count = len(group_summaries)
    return {
        "schema_version": ADAPTIVE_OVERLAY_SCHEMA_VERSION,
        "metric_type": "branch_set_metrics",
        "success_delta": float(success_delta),
        "group_keys": list(group_keys),
        "group_count": group_count,
        "pass_at_k_rate": pass_count / group_count if group_count else 0.0,
        "safe_pass_at_k_rate": safe_pass_count / group_count if group_count else 0.0,
        "regressed_at_k_rate": regression_count / group_count if group_count else 0.0,
        "mean_best_delta": total_best_delta / group_count if group_count else 0.0,
        "mean_branch_delta": total_mean_delta / group_count if group_count else 0.0,
        "mean_oracle_gap": oracle_gap_sum / oracle_gap_count if oracle_gap_count else 0.0,
        "oracle_gap_group_count": oracle_gap_count,
        "groups": group_summaries,
    }


def summarize_overlay_readiness(
    overlay_report: Mapping[str, Any],
    *,
    min_groups: int = 3,
    min_safe_pass_rate: float = 0.05,
    max_regressed_rate: float = 0.0,
    min_mean_best_delta: float = 0.001,
) -> Dict[str, Any]:
    """Summarize whether overlay metrics justify broader validation.

    This is not a promotion gate. It is a fail-closed readiness signal used to
    decide whether a channel family deserves more replay/holdout work.
    """

    metrics = overlay_report.get("branch_set_metrics", {}) if isinstance(overlay_report, Mapping) else {}
    summary = overlay_report.get("summary", {}) if isinstance(overlay_report, Mapping) else {}
    group_count = int(metrics.get("group_count", 0))
    safe_pass_rate = float(metrics.get("safe_pass_at_k_rate", 0.0))
    regressed_rate = float(metrics.get("regressed_at_k_rate", 1.0))
    mean_best_delta = float(metrics.get("mean_best_delta", 0.0))
    blocker_count = int(summary.get("promotion_blockers", 0))
    active_negative_count = int(summary.get("active_negative_records", 0))

    blockers = []
    if group_count < min_groups:
        blockers.append("insufficient_branch_groups")
    if safe_pass_rate < min_safe_pass_rate:
        blockers.append("safe_pass_rate_below_threshold")
    if regressed_rate > max_regressed_rate:
        blockers.append("regression_rate_above_threshold")
    if mean_best_delta < min_mean_best_delta:
        blockers.append("mean_best_delta_below_threshold")
    if blocker_count > 0:
        blockers.append("promotion_blockers_present")
    if active_negative_count > 0:
        blockers.append("active_negative_records_present")

    return {
        "schema_version": ADAPTIVE_OVERLAY_SCHEMA_VERSION,
        "record_type": "adaptive_overlay_readiness",
        "ready_for_broader_validation": len(blockers) == 0,
        "blockers": blockers,
        "group_count": group_count,
        "safe_pass_at_k_rate": safe_pass_rate,
        "regressed_at_k_rate": regressed_rate,
        "mean_best_delta": mean_best_delta,
        "promotion_blockers": blocker_count,
        "active_negative_records": active_negative_count,
        "criteria": {
            "min_groups": int(min_groups),
            "min_safe_pass_rate": float(min_safe_pass_rate),
            "max_regressed_rate": float(max_regressed_rate),
            "min_mean_best_delta": float(min_mean_best_delta),
        },
        "next_action": (
            "run broader replay and holdout validation"
            if not blockers
            else "continue observation and coverage-aware channel collection"
        ),
    }


def _record_budget_units(record: Mapping[str, Any]) -> int:
    for key in ("budget_units", "compute_budget_units"):
        value = record.get(key)
        if isinstance(value, (int, float)):
            return int(value)
    source_row = record.get("source_row", {})
    if isinstance(source_row, Mapping):
        for key in ("budget_units", "compute_budget_units"):
            value = source_row.get(key)
            if isinstance(value, (int, float)):
                return int(value)
    return 1


def summarize_overlay_trajectory_health(overlay_report: Mapping[str, Any]) -> Dict[str, Any]:
    """Summarize overlay branch-search health without changing runtime behavior."""

    records = (
        list(overlay_report.get("channel_variation_records", []))
        if isinstance(overlay_report, Mapping)
        else []
    )
    metrics = dict(overlay_report.get("branch_set_metrics", {})) if isinstance(overlay_report, Mapping) else {}
    groups = list(metrics.get("groups", []))
    blocker_recurrence: Dict[str, int] = {}
    budget_total = 0
    safe_passes = 0
    repeated_branch_groups = 0

    for record in records:
        budget_total += _record_budget_units(record)
        channel_id = str(record.get("channel_id") or "unknown")
        if bool(record.get("promotion_blocker", False)) or bool(record.get("active_negative_flags")):
            blocker_recurrence[channel_id] = blocker_recurrence.get(channel_id, 0) + 1

    oracle_gaps = []
    for group in groups:
        if int(group.get("branch_count", 0) or 0) > 1:
            repeated_branch_groups += 1
        if bool(group.get("safe_pass_at_k", False)):
            safe_passes += 1
        oracle_gap = group.get("oracle_gap")
        if isinstance(oracle_gap, (int, float)):
            oracle_gaps.append(float(oracle_gap))

    midpoint = len(oracle_gaps) // 2
    early_gap = sum(oracle_gaps[:midpoint]) / midpoint if midpoint else 0.0
    late_count = len(oracle_gaps) - midpoint
    late_gap = sum(oracle_gaps[midpoint:]) / late_count if late_count else 0.0
    if len(oracle_gaps) < 2:
        oracle_gap_trend = "insufficient_signal"
    elif late_gap < early_gap:
        oracle_gap_trend = "improving"
    elif late_gap > early_gap:
        oracle_gap_trend = "worsening"
    else:
        oracle_gap_trend = "flat"

    group_count = int(metrics.get("group_count", len(groups)) or 0)
    loop_burden = repeated_branch_groups / group_count if group_count else 0.0
    budget_per_safe_pass = budget_total / safe_passes if safe_passes else None
    warnings = []
    if blocker_recurrence:
        warnings.append("blocker_recurrence_present")
    if oracle_gap_trend == "worsening":
        warnings.append("oracle_gap_worsening")
    if budget_per_safe_pass is None:
        warnings.append("no_safe_pass_budget_signal")

    return {
        "schema_version": ADAPTIVE_OVERLAY_SCHEMA_VERSION,
        "record_type": "adaptive_overlay_trajectory_health",
        "record_count": len(records),
        "branch_group_count": group_count,
        "loop_burden_rate": loop_burden,
        "blocker_recurrence_by_channel": dict(sorted(blocker_recurrence.items())),
        "oracle_gap_trend": oracle_gap_trend,
        "early_mean_oracle_gap": early_gap,
        "late_mean_oracle_gap": late_gap,
        "budget_units": budget_total,
        "safe_pass_count": safe_passes,
        "budget_per_safe_pass": budget_per_safe_pass,
        "warnings": warnings,
    }


def decide_overlay_collection_budget(
    overlay_report: Mapping[str, Any],
    *,
    uncertainty_score: float = 0.0,
    coverage_novelty_score: float = 0.0,
    blocker_history_count: int = 0,
    base_budget_units: int = 1,
    max_budget_units: int = 8,
) -> Dict[str, Any]:
    """Choose an advisory overlay collection budget from report diagnostics."""

    readiness = dict(overlay_report.get("readiness", {})) if isinstance(overlay_report, Mapping) else {}
    trajectory = dict(overlay_report.get("trajectory_health", {})) if isinstance(overlay_report, Mapping) else {}
    blockers = set(str(blocker) for blocker in readiness.get("blockers", []))
    warnings = set(str(warning) for warning in trajectory.get("warnings", []))
    score = 0.0
    reasons = []

    if bool(readiness.get("ready_for_broader_validation", False)):
        score += 0.20
        reasons.append("overlay_ready")
    if "insufficient_branch_groups" in blockers:
        score += 0.25
        reasons.append("insufficient_branch_groups")
    if "safe_pass_rate_below_threshold" in blockers:
        score += 0.15
        reasons.append("safe_pass_rate_below_threshold")
    if "promotion_blockers_present" in blockers or "active_negative_records_present" in blockers:
        score -= 0.30
        reasons.append("blocker_history_present")
    if "blocker_recurrence_present" in warnings:
        score -= 0.20
        reasons.append("trajectory_blocker_recurrence")
    if "oracle_gap_worsening" in warnings:
        score += 0.15
        reasons.append("oracle_gap_worsening")
    if blocker_history_count > 0:
        score -= min(0.30, blocker_history_count * 0.05)
        reasons.append("external_blocker_history")

    score += max(0.0, min(1.0, float(uncertainty_score))) * 0.30
    score += max(0.0, min(1.0, float(coverage_novelty_score))) * 0.25
    if uncertainty_score:
        reasons.append("uncertainty_signal")
    if coverage_novelty_score:
        reasons.append("coverage_novelty_signal")

    if score >= 0.55:
        decision = "broaden_collection"
        budget_units = max(base_budget_units + 2, int(round(max_budget_units * 0.75)))
    elif score >= 0.20:
        decision = "standard_collection"
        budget_units = max(base_budget_units, int(round(max_budget_units * 0.40)))
    else:
        decision = "observe_only"
        budget_units = base_budget_units

    return {
        "schema_version": ADAPTIVE_OVERLAY_SCHEMA_VERSION,
        "record_type": "adaptive_overlay_collection_budget_policy",
        "decision": decision,
        "budget_units": min(int(max_budget_units), max(int(base_budget_units), int(budget_units))),
        "score": float(score),
        "reasons": sorted(set(reasons)),
        "inputs": {
            "uncertainty_score": float(uncertainty_score),
            "coverage_novelty_score": float(coverage_novelty_score),
            "blocker_history_count": int(blocker_history_count),
            "base_budget_units": int(base_budget_units),
            "max_budget_units": int(max_budget_units),
        },
        "advisory_only": True,
        "next_action": (
            "collect broader overlay branch evidence"
            if decision == "broaden_collection"
            else "continue observation without promotion"
        ),
    }


def build_overlay_report(
    rows: Iterable[Mapping[str, Any]],
    *,
    success_delta: float = 0.001,
) -> Dict[str, Any]:
    """Build normalized channel records plus summary and branch-set metrics."""

    records = build_channel_variation_records(rows)
    report = {
        "schema_version": ADAPTIVE_OVERLAY_SCHEMA_VERSION,
        "record_type": "adaptive_overlay_report",
        "channel_variation_records": records,
        "summary": summarize_channel_variations(records),
        "branch_set_metrics": compute_branch_set_metrics(records, success_delta=success_delta),
    }
    report["readiness"] = summarize_overlay_readiness(report)
    report["trajectory_health"] = summarize_overlay_trajectory_health(report)
    return report


def build_overlay_artifact_card(
    *,
    candidate_id: str,
    overlay_report: Mapping[str, Any],
    purpose: str = "adaptive overlay candidate evidence",
    source_path: str | None = None,
    promotion_decision: Mapping[str, Any] | None = None,
    validation_report: Mapping[str, Any] | None = None,
    collection_policy: Mapping[str, Any] | None = None,
    verifier_cards: Iterable[Mapping[str, Any]] | None = None,
    replay_report: Mapping[str, Any] | None = None,
    holdout_report: Mapping[str, Any] | None = None,
    hard_negative_report: Mapping[str, Any] | None = None,
    evaluator_report: Mapping[str, Any] | None = None,
    safety_report: Mapping[str, Any] | None = None,
    limitations: Iterable[str] | None = None,
    rollback_path: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a compact card for candidate overlay review and rollback tracking."""

    readiness = dict(overlay_report.get("readiness", {})) if isinstance(overlay_report, Mapping) else {}
    summary = dict(overlay_report.get("summary", {})) if isinstance(overlay_report, Mapping) else {}
    metrics = dict(overlay_report.get("branch_set_metrics", {})) if isinstance(overlay_report, Mapping) else {}
    trajectory_health = dict(overlay_report.get("trajectory_health", {})) if isinstance(overlay_report, Mapping) else {}
    decision = dict(promotion_decision or {})
    blocker_list = list(readiness.get("blockers") or [])
    limitation_list = list(limitations or [])
    if not bool(decision.get("promotion_ready", False)):
        limitation_list.append("not_default_promoted")
    limitation_list.extend(str(blocker) for blocker in blocker_list)

    card = {
        "schema_version": ADAPTIVE_OVERLAY_SCHEMA_VERSION,
        "record_type": "adaptive_overlay_artifact_card",
        "candidate_id": str(candidate_id),
        "purpose": purpose,
        "source_overlay_report_hash": f"overlay_report_{stable_overlay_hash(overlay_report)}",
        "source_overlay_report_path": source_path,
        "readiness": {
            "ready_for_broader_validation": bool(readiness.get("ready_for_broader_validation", False)),
            "blockers": blocker_list,
            "next_action": readiness.get("next_action"),
            "group_count": int(readiness.get("group_count", metrics.get("group_count", 0)) or 0),
            "safe_pass_at_k_rate": float(readiness.get("safe_pass_at_k_rate", metrics.get("safe_pass_at_k_rate", 0.0)) or 0.0),
            "regressed_at_k_rate": float(readiness.get("regressed_at_k_rate", metrics.get("regressed_at_k_rate", 0.0)) or 0.0),
            "mean_best_delta": float(readiness.get("mean_best_delta", metrics.get("mean_best_delta", 0.0)) or 0.0),
        },
        "evidence": {
            "record_count": int(summary.get("record_count", 0) or 0),
            "channel_types": _json_safe(summary.get("channel_types", {})),
            "decisions": _json_safe(summary.get("decisions", {})),
            "promotion_blockers": int(summary.get("promotion_blockers", 0) or 0),
            "active_negative_records": int(summary.get("active_negative_records", 0) or 0),
            "branch_group_count": int(metrics.get("group_count", 0) or 0),
            "pass_at_k_rate": float(metrics.get("pass_at_k_rate", 0.0) or 0.0),
            "safe_pass_at_k_rate": float(metrics.get("safe_pass_at_k_rate", 0.0) or 0.0),
            "mean_oracle_gap": float(metrics.get("mean_oracle_gap", 0.0) or 0.0),
            "trajectory_health": _json_safe(trajectory_health),
            "validation": _json_safe(validation_report or {}),
            "collection_policy": _json_safe(collection_policy or {}),
            "verifier_cards": _json_safe(list(verifier_cards or [])),
            "replay": _json_safe(replay_report or {}),
            "holdout": _json_safe(holdout_report or {}),
            "hard_negative": _json_safe(hard_negative_report or {}),
            "evaluator": _json_safe(evaluator_report or {}),
            "safety": _json_safe(safety_report or {}),
            "promotion_decision": {
                "promotion_ready": bool(decision.get("promotion_ready", False)),
                "adaptive_overlay_ready": bool(decision.get("adaptive_overlay_ready", False)),
                "reasons": list(decision.get("reasons") or []),
            },
        },
        "limitations": sorted(set(str(item) for item in limitation_list if item)),
        "rollback_path": rollback_path,
        "metadata": _json_safe(metadata or {}),
    }
    card["card_id"] = f"overlay_card_{stable_overlay_hash(card)}"
    return card


def build_verifier_evidence_card(
    *,
    verifier_id: str,
    subject_id: str,
    rubric: Mapping[str, Any],
    result: Mapping[str, Any],
    source_path: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build review-only verifier/rubric evidence for overlay artifacts."""

    passed = bool(result.get("passed", False))
    score = result.get("score")
    normalized_score = float(score) if isinstance(score, (int, float)) else None
    card = {
        "schema_version": ADAPTIVE_OVERLAY_SCHEMA_VERSION,
        "record_type": "verifier_evidence_card",
        "verifier_id": str(verifier_id),
        "subject_id": str(subject_id),
        "rubric": _json_safe(rubric),
        "result": {
            "passed": passed,
            "score": normalized_score,
            "reasons": list(result.get("reasons") or []),
            "blockers": list(result.get("blockers") or []),
        },
        "source_path": source_path,
        "metadata": _json_safe(metadata or {}),
        "advisory_only": True,
    }
    card["card_id"] = f"verifier_card_{stable_overlay_hash(card)}"
    return card


def build_overlay_validation_report(
    *,
    candidate_id: str,
    replay_overlay_report: Mapping[str, Any],
    holdout_overlay_report: Mapping[str, Any] | None = None,
    min_holdout_safe_pass_rate: float = 0.05,
    max_holdout_regressed_rate: float = 0.0,
    metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build fail-closed replay/holdout validation evidence for an overlay."""

    replay_readiness = (
        dict(replay_overlay_report.get("readiness", {}))
        if isinstance(replay_overlay_report, Mapping)
        else {}
    )
    holdout_readiness = (
        dict(holdout_overlay_report.get("readiness", {}))
        if isinstance(holdout_overlay_report, Mapping)
        else {}
    )
    holdout_metrics = (
        dict(holdout_overlay_report.get("branch_set_metrics", {}))
        if isinstance(holdout_overlay_report, Mapping)
        else {}
    )
    replay_ready = bool(replay_readiness.get("ready_for_broader_validation", False))
    holdout_ready = bool(holdout_readiness.get("ready_for_broader_validation", False))
    holdout_safe_pass_rate = float(
        holdout_readiness.get("safe_pass_at_k_rate", holdout_metrics.get("safe_pass_at_k_rate", 0.0)) or 0.0
    )
    holdout_regressed_rate = float(
        holdout_readiness.get("regressed_at_k_rate", holdout_metrics.get("regressed_at_k_rate", 1.0)) or 0.0
    )

    blockers = []
    if not replay_ready:
        blockers.append("replay_overlay_not_ready")
    if holdout_overlay_report is None:
        blockers.append("missing_holdout_overlay_report")
    elif not holdout_ready:
        blockers.append("holdout_overlay_not_ready")
    if holdout_safe_pass_rate < min_holdout_safe_pass_rate:
        blockers.append("holdout_safe_pass_rate_below_threshold")
    if holdout_regressed_rate > max_holdout_regressed_rate:
        blockers.append("holdout_regression_rate_above_threshold")
    blockers.extend(f"replay:{blocker}" for blocker in replay_readiness.get("blockers", []))
    blockers.extend(f"holdout:{blocker}" for blocker in holdout_readiness.get("blockers", []))

    report = {
        "schema_version": ADAPTIVE_OVERLAY_SCHEMA_VERSION,
        "record_type": "adaptive_overlay_validation_report",
        "candidate_id": str(candidate_id),
        "replay_overlay_report_hash": f"overlay_report_{stable_overlay_hash(replay_overlay_report)}",
        "holdout_overlay_report_hash": (
            f"overlay_report_{stable_overlay_hash(holdout_overlay_report)}"
            if holdout_overlay_report is not None
            else None
        ),
        "replay_readiness": _json_safe(replay_readiness),
        "holdout_readiness": _json_safe(holdout_readiness),
        "criteria": {
            "min_holdout_safe_pass_rate": float(min_holdout_safe_pass_rate),
            "max_holdout_regressed_rate": float(max_holdout_regressed_rate),
        },
        "validation_ready": len(blockers) == 0,
        "blockers": sorted(set(blockers)),
        "next_action": (
            "eligible for promotion review with artifact card"
            if not blockers
            else "collect broader replay and holdout overlay evidence"
        ),
        "metadata": _json_safe(metadata or {}),
    }
    report["validation_report_id"] = f"overlay_validation_{stable_overlay_hash(report)}"
    return report
