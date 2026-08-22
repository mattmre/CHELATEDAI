"""Atomic, content-addressed Variation checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Mapping, Optional, Tuple

from ..canonical import canonical_bytes, digest_for
from .errors import VariationCheckpointError


CHECKPOINT_SCHEMA = "egv-variation-checkpoint-v1"


@dataclass(frozen=True)
class VariationCheckpoint:
    campaign_id: str
    run_id: str
    arm_id: str
    task_id: str
    seed: int
    attempt_index: int
    last_candidate_id: Optional[str]
    ledger_head_event_id: Optional[str]
    ledger_head_hash: str
    projection_generation: str
    artifact_manifest_hash: str
    protocol_digest: str
    model_digest: str
    adapter_digest: Optional[str]
    retrieval_policy_digest: str
    state_digest: str
    status: str
    schema_version: str = CHECKPOINT_SCHEMA

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "campaign_id": self.campaign_id,
            "run_id": self.run_id,
            "arm_id": self.arm_id,
            "task_id": self.task_id,
            "seed": self.seed,
            "attempt_index": self.attempt_index,
            "last_candidate_id": self.last_candidate_id,
            "ledger_head_event_id": self.ledger_head_event_id,
            "ledger_head_hash": self.ledger_head_hash,
            "projection_generation": self.projection_generation,
            "artifact_manifest_hash": self.artifact_manifest_hash,
            "protocol_digest": self.protocol_digest,
            "model_digest": self.model_digest,
            "adapter_digest": self.adapter_digest,
            "retrieval_policy_digest": self.retrieval_policy_digest,
            "state_digest": self.state_digest,
            "status": self.status,
        }

    @property
    def digest(self) -> str:
        return digest_for(self.to_dict())

    def validate(self) -> None:
        if self.schema_version != CHECKPOINT_SCHEMA:
            raise VariationCheckpointError("unsupported Variation checkpoint schema")
        if not all(isinstance(value, str) and value for value in (self.campaign_id, self.run_id, self.arm_id, self.task_id)):
            raise VariationCheckpointError("Variation checkpoint identity is incomplete")
        if (
            not isinstance(self.seed, int)
            or isinstance(self.seed, bool)
            or not isinstance(self.attempt_index, int)
            or isinstance(self.attempt_index, bool)
            or self.seed < 0
            or self.attempt_index < 0
        ):
            raise VariationCheckpointError("Variation checkpoint counters must be nonnegative")
        if self.last_candidate_id is not None and not isinstance(self.last_candidate_id, str):
            raise VariationCheckpointError("checkpoint last candidate ID must be a string or null")
        if self.ledger_head_event_id is not None and not isinstance(self.ledger_head_event_id, str):
            raise VariationCheckpointError("checkpoint ledger event ID must be a string or null")
        if self.status not in {"RUNNING", "PROMOTED", "BUDGET_EXHAUSTED", "FAILED"}:
            raise VariationCheckpointError("Variation checkpoint status is outside the closed vocabulary")
        for name in (
            "ledger_head_hash",
            "projection_generation",
            "artifact_manifest_hash",
            "protocol_digest",
            "model_digest",
            "retrieval_policy_digest",
            "state_digest",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or len(value) != 64:
                raise VariationCheckpointError("checkpoint {} must be a SHA-256 digest".format(name))
            try:
                int(value, 16)
            except ValueError as exc:
                raise VariationCheckpointError("checkpoint {} must be hexadecimal".format(name)) from exc
        if self.attempt_index == 0 or self.last_candidate_id is None:
            raise VariationCheckpointError("Variation checkpoint must bind a completed candidate attempt")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "VariationCheckpoint":
        required = {
            "schema_version",
            "campaign_id",
            "run_id",
            "arm_id",
            "task_id",
            "seed",
            "attempt_index",
            "last_candidate_id",
            "ledger_head_event_id",
            "ledger_head_hash",
            "projection_generation",
            "artifact_manifest_hash",
            "protocol_digest",
            "model_digest",
            "adapter_digest",
            "retrieval_policy_digest",
            "state_digest",
            "status",
        }
        if set(value) != required:
            raise VariationCheckpointError("Variation checkpoint has an unexpected field set")
        checkpoint = cls(**{key: value[key] for key in required})
        checkpoint.validate()
        return checkpoint


class CheckpointStore:
    """Write-once checkpoint files under one isolated arm workspace."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def save(self, checkpoint: VariationCheckpoint) -> Path:
        checkpoint.validate()
        path = self.root / "checkpoint-{}.json".format(checkpoint.digest)
        encoded = canonical_bytes(checkpoint.to_dict())
        if path.exists():
            try:
                if path.read_bytes() != encoded:
                    raise VariationCheckpointError("checkpoint content conflicts with an existing immutable file")
            except OSError as exc:
                raise VariationCheckpointError("checkpoint cannot be read") from exc
            return path
        descriptor, temporary_name = tempfile.mkstemp(prefix=".checkpoint-", dir=str(self.root))
        os.close(descriptor)
        temporary = Path(temporary_name)
        try:
            with temporary.open("wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.chmod(0o444)
            temporary.replace(path)
        finally:
            if temporary.exists():
                temporary.unlink()
        return path

    def load(self, path: Path) -> VariationCheckpoint:
        candidate_path = Path(path).resolve()
        try:
            candidate_path.relative_to(self.root)
        except ValueError as exc:
            raise VariationCheckpointError("Variation checkpoint is outside its isolated store") from exc
        try:
            value = json.loads(candidate_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise VariationCheckpointError("Variation checkpoint cannot be read") from exc
        if not isinstance(value, Mapping):
            raise VariationCheckpointError("Variation checkpoint must be a JSON object")
        checkpoint = VariationCheckpoint.from_mapping(value)
        if canonical_bytes(value) != canonical_bytes(checkpoint.to_dict()):
            raise VariationCheckpointError("Variation checkpoint is not canonical JSON")
        expected_name = "checkpoint-{}.json".format(checkpoint.digest)
        if candidate_path.name != expected_name:
            raise VariationCheckpointError("Variation checkpoint filename is not content-addressed")
        return checkpoint

    def latest(self, *, run_id: Optional[str] = None) -> Optional[Tuple[Path, VariationCheckpoint]]:
        candidates = []
        for path in sorted(self.root.glob("checkpoint-*.json")):
            checkpoint = self.load(path)
            if run_id is None or checkpoint.run_id == run_id:
                candidates.append((path, checkpoint))
        if not candidates:
            return None
        candidates.sort(key=lambda item: (item[1].attempt_index, item[1].digest))
        return candidates[-1]


__all__ = ["CHECKPOINT_SCHEMA", "CheckpointStore", "VariationCheckpoint"]
