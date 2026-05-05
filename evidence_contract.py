"""Shared evidence events for Engine-Scope, Model-Scope, retrieval, tools, and evaluators."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence


EVIDENCE_SCHEMA_VERSION = 1

EVIDENCE_SURFACES = {
    "engine_scope",
    "model_scope",
    "retrieval",
    "tool",
    "evaluator",
    "safety",
    "memory",
    "training",
    "promotion",
}


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def build_episode_event(
    *,
    event_type: str,
    surface: str,
    payload: Mapping[str, Any] | None = None,
    run_id: str | None = None,
    campaign_id: str | None = None,
    query_id: str | None = None,
    artifact_id: str | None = None,
    action: str | None = None,
    decision: str | None = None,
    outcome_metrics: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    source_family: str | None = None,
    trace: Mapping[str, Any] | None = None,
    evaluator_results: Sequence[Mapping[str, Any]] | None = None,
    safety_results: Mapping[str, Any] | None = None,
    promotion_status: Mapping[str, Any] | None = None,
    event_id: str | None = None,
    created_at: str | None = None,
) -> Dict[str, Any]:
    """Build one normalized evidence event."""

    if surface not in EVIDENCE_SURFACES:
        raise ValueError(f"unknown evidence surface '{surface}'")
    base = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "event_type": str(event_type),
        "surface": surface,
        "run_id": run_id,
        "campaign_id": campaign_id,
        "query_id": query_id,
        "artifact_id": artifact_id,
        "action": action,
        "decision": decision,
        "outcome_metrics": _json_safe(outcome_metrics or {}),
        "provenance": _json_safe(provenance or {}),
        "source_family": source_family,
        "payload": _json_safe(payload or {}),
        "trace": _json_safe(trace or {}),
        "evaluator_results": _json_safe(list(evaluator_results or [])),
        "safety_results": _json_safe(safety_results or {}),
        "promotion_status": _json_safe(promotion_status or {}),
        "created_at": created_at or _utcnow_iso(),
    }
    base["event_id"] = event_id or f"evt_{_stable_hash(base)}"
    return base


def build_evidence_bundle(
    events: Iterable[Mapping[str, Any]],
    *,
    bundle_id: str | None = None,
    campaign_id: str | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    """Build a replayable evidence bundle."""

    event_list = []
    for event in events:
        if int(event.get("schema_version", -1)) != EVIDENCE_SCHEMA_VERSION:
            raise ValueError(f"unsupported evidence event schema: {event.get('schema_version')}")
        event_list.append(_json_safe(event))
    payload = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "artifact_type": "evidence_bundle",
        "bundle_id": bundle_id,
        "campaign_id": campaign_id,
        "metadata": _json_safe(metadata or {}),
        "event_count": len(event_list),
        "events": event_list,
    }
    payload["bundle_id"] = bundle_id or f"bundle_{_stable_hash(payload)}"
    return payload


def load_evidence_bundle(path_or_dir: str | Path) -> Dict[str, Any]:
    """Load one evidence bundle or combine all bundle JSON files in a directory."""

    path = Path(path_or_dir)
    if path.is_dir():
        events: List[Dict[str, Any]] = []
        metadata = {"source_dir": str(path)}
        for candidate in sorted(path.glob("*.json")):
            payload = load_evidence_bundle(candidate)
            events.extend(payload.get("events", []))
        return build_evidence_bundle(events, metadata=metadata)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != EVIDENCE_SCHEMA_VERSION:
        raise ValueError(f"unsupported evidence bundle schema: {payload.get('schema_version')}")
    if payload.get("artifact_type") != "evidence_bundle":
        raise ValueError(f"unsupported evidence artifact type: {payload.get('artifact_type')}")
    return payload


def write_evidence_bundle(path: str | Path, bundle: Mapping[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_json_safe(bundle), indent=2), encoding="utf-8")
    return output_path


def summarize_evidence_bundle(bundle: Mapping[str, Any]) -> Dict[str, Any]:
    events = [event for event in bundle.get("events", []) if isinstance(event, Mapping)]
    by_surface: Dict[str, int] = {}
    by_event_type: Dict[str, int] = {}
    by_source_family: Dict[str, int] = {}
    decisions: Dict[str, int] = {}
    for event in events:
        surface = str(event.get("surface") or "unknown")
        event_type = str(event.get("event_type") or "unknown")
        source_family = str(event.get("source_family") or "unknown")
        decision = str(event.get("decision") or "unknown")
        by_surface[surface] = by_surface.get(surface, 0) + 1
        by_event_type[event_type] = by_event_type.get(event_type, 0) + 1
        by_source_family[source_family] = by_source_family.get(source_family, 0) + 1
        decisions[decision] = decisions.get(decision, 0) + 1
    return {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "bundle_id": bundle.get("bundle_id"),
        "event_count": len(events),
        "surfaces": by_surface,
        "event_types": by_event_type,
        "source_families": by_source_family,
        "decisions": decisions,
    }


def evidence_event_from_engine_scope_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert an Engine-Scope row to a shared evidence event."""

    metrics = {
        key: row.get(key)
        for key in (
            "delta_ndcg_at_10",
            "delta_mrr",
            "delta_recall_at_10",
            "rank_delta",
            "selected_delta_ndcg_at_10",
        )
        if key in row
    }
    decision = row.get("decision") or row.get("fault_class") or row.get("action")
    return build_episode_event(
        event_type=str(row.get("row_type") or "engine_scope_row"),
        surface="engine_scope",
        query_id=None if row.get("query_id") is None else str(row.get("query_id")),
        artifact_id=row.get("_artifact_path"),
        action=None if row.get("action") is None else str(row.get("action")),
        decision=None if decision is None else str(decision),
        outcome_metrics=metrics,
        provenance={"artifact_path": row.get("_artifact_path"), "task": row.get("task"), "profile": row.get("profile")},
        source_family=None if row.get("source_family") is None else str(row.get("source_family")),
        payload=row,
    )


def evidence_event_from_model_scope_artifact(artifact: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert a Model-Scope artifact to a shared evidence event."""

    capture = artifact.get("capture", {}) if isinstance(artifact.get("capture"), Mapping) else {}
    metadata = capture.get("metadata", {}) if isinstance(capture.get("metadata"), Mapping) else {}
    return build_episode_event(
        event_type=str(artifact.get("artifact_type") or "model_scope_artifact"),
        surface="model_scope",
        query_id=None if metadata.get("query_id") is None else str(metadata.get("query_id")),
        artifact_id=artifact.get("output_path"),
        action="observe",
        decision=None,
        outcome_metrics={
            "token_count": capture.get("token_count"),
            "captured_layer_count": capture.get("captured_layer_count"),
        },
        provenance={"output_path": artifact.get("output_path"), "model_name": artifact.get("runtime", {}).get("model_name")},
        source_family="model_scope_runtime_observation",
        payload=artifact,
        trace=artifact.get("trace") if isinstance(artifact.get("trace"), Mapping) else {},
    )
