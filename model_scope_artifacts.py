"""Versioned artifact helpers for Model-Scope observation payloads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

from evidence_contract import evidence_event_from_model_scope_artifact


MODEL_SCOPE_ARTIFACT_SCHEMA_VERSION = 1


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def build_model_scope_artifact(
    *,
    runtime: Mapping[str, Any],
    capture: Mapping[str, Any],
    output_path: str | None = None,
    trace: Mapping[str, Any] | None = None,
    evidence: Mapping[str, Any] | None = None,
) -> Dict[str, Any]:
    artifact = {
        "schema_version": MODEL_SCOPE_ARTIFACT_SCHEMA_VERSION,
        "artifact_type": "model_scope_runtime_observation",
        "runtime": _json_safe(runtime),
        "capture": _json_safe(capture),
    }
    if output_path is not None:
        artifact["output_path"] = str(output_path)
    if trace is not None:
        artifact["trace"] = _json_safe(trace)
    if evidence is not None:
        artifact["evidence"] = _json_safe(evidence)
    return artifact


def write_model_scope_artifact(path: str | Path, artifact: Mapping[str, Any]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(_json_safe(artifact), indent=2), encoding="utf-8")
    return output_path


def load_model_scope_artifact(path: str | Path) -> Dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if int(payload.get("schema_version", -1)) != MODEL_SCOPE_ARTIFACT_SCHEMA_VERSION:
        raise ValueError(f"unsupported Model-Scope artifact schema: {payload.get('schema_version')}")
    return payload


def summarize_model_scope_artifact(artifact: Mapping[str, Any]) -> Dict[str, Any]:
    capture = artifact.get("capture", {})
    runtime = artifact.get("runtime", {})
    return {
        "schema_version": MODEL_SCOPE_ARTIFACT_SCHEMA_VERSION,
        "model_name": runtime.get("model_name"),
        "device": runtime.get("device"),
        "token_count": capture.get("token_count"),
        "captured_layer_count": capture.get("captured_layer_count", 0),
        "layer_indices": capture.get("layer_indices"),
        "output_path": artifact.get("output_path"),
        "evidence_event_id": artifact.get("evidence", {}).get("event_id") if isinstance(artifact.get("evidence"), Mapping) else None,
    }


def model_scope_artifact_to_evidence_event(artifact: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert a Model-Scope artifact into the shared evidence-event contract."""

    return evidence_event_from_model_scope_artifact(artifact)
