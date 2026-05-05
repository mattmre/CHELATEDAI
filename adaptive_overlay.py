"""Observation-only adaptive channel overlay records.

This module does not route or promote anything. It normalizes existing
Engine-Scope rows into channel-variation records so current gates, road-course
profiles, and future Model-Scope hooks can be compared through one vocabulary.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, Iterable, Mapping


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
