"""Typed segmented memory for Model-Scope observations and replay bundles."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping

from model_scope_artifacts import summarize_model_scope_artifact


MODEL_SCOPE_MEMORY_SCHEMA_VERSION = 1


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


def _deep_copy(value: Any) -> Any:
    return json.loads(json.dumps(_json_safe(value)))


def _stable_hash(payload: Any) -> str:
    encoded = json.dumps(_json_safe(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


@dataclass
class MemorySegmentConfig:
    """Configuration for one typed Model-Scope memory segment."""

    name: str
    max_entries: int
    description: str = ""
    allow_promotion: bool = False

    def __post_init__(self) -> None:
        if self.max_entries < 1:
            raise ValueError("max_entries must be >= 1")


DEFAULT_SEGMENT_CONFIGS = [
    MemorySegmentConfig(
        name="working",
        max_entries=32,
        description="Short-lived recent observations used for immediate context.",
        allow_promotion=False,
    ),
    MemorySegmentConfig(
        name="episode",
        max_entries=256,
        description="Replayable observation episodes used for offline comparison and training.",
        allow_promotion=True,
    ),
    MemorySegmentConfig(
        name="expectation",
        max_entries=128,
        description="Reference profiles used to compare live observations against expected behavior.",
        allow_promotion=False,
    ),
    MemorySegmentConfig(
        name="persistent",
        max_entries=512,
        description="Promoted artifacts and replay entries retained beyond one campaign.",
        allow_promotion=False,
    ),
]


class ModelScopeMemoryStore:
    """Bounded typed memory store for Model-Scope artifacts."""

    def __init__(self, segment_configs: Iterable[MemorySegmentConfig] | None = None):
        configs = list(segment_configs or DEFAULT_SEGMENT_CONFIGS)
        self._segment_configs = {config.name: config for config in configs}
        self._segments = {config.name: [] for config in configs}

    def _require_segment(self, segment: str) -> None:
        if segment not in self._segments:
            valid = ", ".join(sorted(self._segments))
            raise ValueError(f"unknown memory segment '{segment}'. Valid segments: {valid}")

    def _append_entry(
        self,
        segment: str,
        *,
        entry_kind: str,
        payload: Mapping[str, Any],
        tags: Iterable[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
        entry_id: str | None = None,
    ) -> Dict[str, Any]:
        self._require_segment(segment)
        config = self._segment_configs[segment]
        entry = {
            "entry_id": entry_id or f"{segment}_{_stable_hash([segment, entry_kind, payload, _utcnow_iso()])}",
            "segment": segment,
            "entry_kind": entry_kind,
            "created_at": _utcnow_iso(),
            "tags": [str(item) for item in (tags or [])],
            "metadata": _json_safe(metadata or {}),
            "payload": _json_safe(payload),
        }
        self._segments[segment].append(entry)
        overflow = len(self._segments[segment]) - config.max_entries
        if overflow > 0:
            del self._segments[segment][:overflow]
        return _deep_copy(entry)

    def segment_sizes(self) -> Dict[str, int]:
        return {name: len(entries) for name, entries in self._segments.items()}

    def get_segment_entries(self, segment: str) -> List[Dict[str, Any]]:
        self._require_segment(segment)
        return [_deep_copy(entry) for entry in self._segments[segment]]

    def store_expectation_profile(
        self,
        profile: Mapping[str, Any],
        *,
        profile_id: str | None = None,
        tags: Iterable[str] | None = None,
        metadata: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        resolved_profile_id = str(profile_id or profile.get("profile_id") or _stable_hash(profile))
        payload = {
            "profile_id": resolved_profile_id,
            "profile": _json_safe(profile),
        }
        return self._append_entry(
            "expectation",
            entry_kind="expectation_profile",
            payload=payload,
            tags=["expectation_profile", *(tags or [])],
            metadata=metadata,
            entry_id=f"expectation_{resolved_profile_id}",
        )

    def get_expectation_profile(self, profile_id: str) -> Dict[str, Any] | None:
        needle = str(profile_id)
        for entry in reversed(self._segments["expectation"]):
            payload = entry.get("payload", {})
            if str(payload.get("profile_id")) == needle:
                return _deep_copy(payload.get("profile"))
        return None

    def record_observation(
        self,
        artifact: Mapping[str, Any],
        *,
        query_text: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        expectation_profile_id: str | None = None,
        overlay_id: str | None = None,
        promote: bool = False,
        evidence_event_ids: Iterable[str] | None = None,
        promotion_status: Mapping[str, Any] | None = None,
        retention_reason: str | None = None,
        synthetic_depth: int = 0,
        source_lineage: Iterable[str] | None = None,
    ) -> Dict[str, Any]:
        capture = artifact.get("capture", {})
        stored_metadata = {**dict(capture.get("metadata", {})), **dict(metadata or {})}
        query_hash = (
            capture.get("prompt_hash")
            or stored_metadata.get("query_hash")
            or stored_metadata.get("query_id")
            or _stable_hash(query_text or artifact)
        )
        artifact_payload = _deep_copy(artifact)
        artifact_payload.pop("memory", None)
        payload = {
            "query_hash": str(query_hash),
            "query_text": None if query_text is None else str(query_text),
            "model_name": artifact.get("runtime", {}).get("model_name"),
            "artifact_summary": summarize_model_scope_artifact(artifact),
            "artifact": artifact_payload,
            "steering": artifact.get("steering"),
            "expectation_comparison": artifact.get("expectation_comparison"),
            "expectation_profile_id": None if expectation_profile_id is None else str(expectation_profile_id),
            "overlay_id": None if overlay_id is None else str(overlay_id),
            "metadata": _json_safe(stored_metadata),
            "evidence_event_ids": [str(item) for item in (evidence_event_ids or [])],
            "promotion_status": _json_safe(promotion_status or {"promotion_ready": False, "reasons": ["candidate_evidence_only"]}),
            "retention_reason": retention_reason or "candidate_evidence",
            "synthetic_depth": int(synthetic_depth),
            "source_lineage": [str(item) for item in (source_lineage or [])],
        }
        tags = ["observation", f"model:{payload['model_name']}"]
        working_entry = self._append_entry(
            "working",
            entry_kind="observation",
            payload=payload,
            tags=tags,
            metadata={"query_hash": str(query_hash)},
        )
        episode_entry = self._append_entry(
            "episode",
            entry_kind="observation",
            payload=payload,
            tags=tags,
            metadata={"query_hash": str(query_hash)},
        )
        persistent_entry_id = None
        if promote:
            persistent_entry = self._append_entry(
                "persistent",
                entry_kind="promoted_observation",
                payload={
                    **payload,
                    "source_episode_entry_id": episode_entry["entry_id"],
                },
                tags=["promoted", *tags],
                metadata={"reason": "record_observation_promote", "query_hash": str(query_hash)},
            )
            persistent_entry_id = persistent_entry["entry_id"]
        return {
            "query_hash": str(query_hash),
            "working_entry_id": working_entry["entry_id"],
            "episode_entry_id": episode_entry["entry_id"],
            "persistent_entry_id": persistent_entry_id,
            "segment_sizes": self.segment_sizes(),
        }

    def annotate_entry(self, segment: str, entry_id: str, *, update: Mapping[str, Any]) -> Dict[str, Any]:
        self._require_segment(segment)
        for entry in self._segments[segment]:
            if entry["entry_id"] == entry_id:
                payload = dict(entry.get("payload", {}))
                payload.update(_json_safe(update))
                entry["payload"] = payload
                return _deep_copy(entry)
        raise KeyError(f"memory entry '{entry_id}' not found in segment '{segment}'")

    def promote_episode(
        self,
        episode_entry_id: str,
        *,
        reason: str,
        metadata: Mapping[str, Any] | None = None,
        promotion_status: Mapping[str, Any] | None = None,
    ) -> Dict[str, Any]:
        status = dict(promotion_status or {})
        if status and not bool(status.get("promotion_ready")):
            return {
                "source_episode_entry_id": episode_entry_id,
                "persistent_entry_id": None,
                "segment_sizes": self.segment_sizes(),
                "promoted": False,
                "reasons": list(status.get("reasons", ["promotion_not_ready"])),
            }
        for entry in self._segments["episode"]:
            if entry["entry_id"] != episode_entry_id:
                continue
            promoted = self._append_entry(
                "persistent",
                entry_kind="promoted_episode",
                payload={
                    **dict(entry.get("payload", {})),
                    "source_episode_entry_id": episode_entry_id,
                    "promotion_reason": str(reason),
                    "promotion_status": _json_safe(status or {"promotion_ready": True, "reasons": []}),
                },
                tags=["promoted", *entry.get("tags", [])],
                metadata=metadata,
            )
            return {
                "source_episode_entry_id": episode_entry_id,
                "persistent_entry_id": promoted["entry_id"],
                "segment_sizes": self.segment_sizes(),
                "promoted": True,
            }
        raise KeyError(f"episode entry '{episode_entry_id}' not found")

    def build_replay_bundle(
        self,
        *,
        segment: str = "episode",
        entry_ids: Iterable[str] | None = None,
        limit: int | None = None,
        include_artifacts: bool = True,
    ) -> Dict[str, Any]:
        self._require_segment(segment)
        selected = list(self._segments[segment])
        if entry_ids is not None:
            allowed = {str(entry_id) for entry_id in entry_ids}
            selected = [entry for entry in selected if entry["entry_id"] in allowed]
        selected.sort(key=lambda entry: (str(entry.get("created_at")), str(entry.get("entry_id"))))
        if limit is not None:
            selected = selected[-int(limit):]
        bundle_entries = []
        for entry in selected:
            payload = dict(entry.get("payload", {}))
            item = {
                "entry_id": entry["entry_id"],
                "created_at": entry["created_at"],
                "query_hash": payload.get("query_hash"),
                "query_text": payload.get("query_text"),
                "artifact_summary": payload.get("artifact_summary"),
                "steering": payload.get("steering"),
                "expectation_comparison": payload.get("expectation_comparison"),
                "expectation_profile_id": payload.get("expectation_profile_id"),
                "overlay_id": payload.get("overlay_id"),
                "metadata": payload.get("metadata"),
                "evidence_event_ids": payload.get("evidence_event_ids", []),
                "promotion_status": payload.get("promotion_status"),
                "retention_reason": payload.get("retention_reason"),
                "synthetic_depth": payload.get("synthetic_depth", 0),
                "source_lineage": payload.get("source_lineage", []),
            }
            if include_artifacts:
                item["artifact"] = payload.get("artifact")
            bundle_entries.append(item)
        return {
            "schema_version": MODEL_SCOPE_MEMORY_SCHEMA_VERSION,
            "artifact_type": "model_scope_replay_bundle",
            "source_segment": segment,
            "entry_count": len(bundle_entries),
            "entry_ids": [entry["entry_id"] for entry in bundle_entries],
            "segment_sizes": self.segment_sizes(),
            "entries": _json_safe(bundle_entries),
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": MODEL_SCOPE_MEMORY_SCHEMA_VERSION,
            "artifact_type": "model_scope_memory_snapshot",
            "segment_configs": [asdict(config) for config in self._segment_configs.values()],
            "segment_sizes": self.segment_sizes(),
            "segments": _deep_copy(self._segments),
        }

    def save(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return output_path

    @classmethod
    def load(cls, path: str | Path) -> "ModelScopeMemoryStore":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if int(payload.get("schema_version", -1)) != MODEL_SCOPE_MEMORY_SCHEMA_VERSION:
            raise ValueError(f"unsupported Model-Scope memory schema: {payload.get('schema_version')}")
        configs = [MemorySegmentConfig(**item) for item in payload.get("segment_configs", [])]
        store = cls(segment_configs=configs or None)
        segments = payload.get("segments", {})
        for name in store._segments:
            store._segments[name] = [_deep_copy(entry) for entry in segments.get(name, [])]
        return store
