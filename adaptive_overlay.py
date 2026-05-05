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
    return report
