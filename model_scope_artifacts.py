"""Artifact persistence for Model-Scope sparse feature events and related records."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping

from evidence_contract import evidence_event_from_model_scope_artifact
from model_scope_features import SparseFeatureEvent
from model_scope_runtime import ActivationEvent


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


def feature_event_to_dict(event: SparseFeatureEvent) -> dict[str, Any]:
    """Serialise a SparseFeatureEvent to a JSON-compatible dict."""
    return {
        "schema_version": event.schema_version,
        "source_activation": event.source_activation.to_dict(),
        "feature_source": event.feature_source,
        "features": event.features,
        "feature_count": event.feature_count,
        "nonzero_count": event.nonzero_count,
        "extracted_at": event.extracted_at,
    }


def feature_event_from_dict(data: dict[str, Any]) -> SparseFeatureEvent:
    """Deserialise a SparseFeatureEvent from a dict."""
    return SparseFeatureEvent(
        schema_version=data["schema_version"],
        source_activation=ActivationEvent.from_dict(data["source_activation"]),
        feature_source=data["feature_source"],
        features=data["features"],
        feature_count=data["feature_count"],
        nonzero_count=data["nonzero_count"],
        extracted_at=data["extracted_at"],
    )


class ArtifactStore:
    """Compact versioned JSON artifact storage for SparseFeatureEvent objects."""

    def __init__(self, base_dir: str | Path = ".") -> None:
        self._base_dir = Path(base_dir)

    def save_feature_event(
        self,
        event: SparseFeatureEvent,
        *,
        name: str | None = None,
    ) -> Path:
        """Serialise event to JSON in base_dir; return the Path."""
        if name is None:
            run_id = event.source_activation.run_id
            layer_id_safe = (
                event.source_activation.layer_id.replace(".", "_").replace("/", "_")
            )
            name = f"feature_event_{run_id}_{layer_id_safe}.json"
        path = self._base_dir / name
        with path.open("w", encoding="utf-8") as fh:
            json.dump(feature_event_to_dict(event), fh, indent=2)
        return path

    def load_feature_event(self, path: str | Path) -> SparseFeatureEvent:
        """Deserialise a SparseFeatureEvent from a JSON file."""
        path = Path(path)
        with path.open("r", encoding="utf-8") as fh:
            data = json.load(fh)
        return feature_event_from_dict(data)

    def save_batch(
        self,
        events: list[SparseFeatureEvent],
        *,
        manifest_name: str = "feature_batch_manifest.json",
    ) -> Path:
        """Save each event as individual JSON and write a manifest listing their paths."""
        event_paths: list[str] = []
        for idx, event in enumerate(events):
            run_id = event.source_activation.run_id
            layer_id_safe = (
                event.source_activation.layer_id.replace(".", "_").replace("/", "_")
            )
            name = f"feature_event_{run_id}_{layer_id_safe}_{idx}.json"
            event_path = self.save_feature_event(event, name=name)
            event_paths.append(str(event_path))
        manifest: dict[str, Any] = {
            "schema_version": "1.0",
            "count": len(events),
            "artifact_paths": event_paths,
        }
        manifest_path = self._base_dir / manifest_name
        with manifest_path.open("w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2)
        return manifest_path

    def load_batch(self, manifest_path: str | Path) -> list[SparseFeatureEvent]:
        """Reload batch from manifest."""
        manifest_path = Path(manifest_path)
        with manifest_path.open("r", encoding="utf-8") as fh:
            manifest = json.load(fh)
        return [self.load_feature_event(p) for p in manifest["artifact_paths"]]

    def list_artifacts(self, pattern: str = "feature_event_*.json") -> list[Path]:
        """Glob for artifacts in base_dir matching pattern."""
        return sorted(self._base_dir.glob(pattern))
