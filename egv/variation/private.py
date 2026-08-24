"""Durable evaluator-private trajectory sidecars for training freeze."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any, Dict, Iterable, Mapping, Tuple

from ..canonical import canonical_bytes, digest_bytes
from ..evaluation.artifacts import ContentAddressedArtifactStore
from .errors import VariationCheckpointError
from .generator import CandidateContext


PRIVATE_TRAJECTORY_SCHEMA = "egv-private-trajectory-sidecar-v1"


def _context_dict(context: CandidateContext) -> Dict[str, Any]:
    return {
        "campaign_id": context.campaign_id, "run_id": context.run_id, "seed": context.seed,
        "arm_id": context.arm_id, "task_id": context.task_id, "family_id": context.family_id,
        "public_locus": context.public_locus, "public_rule_id": context.public_rule_id,
        "attempt_index": context.attempt_index, "parent_candidate_id": context.parent_candidate_id,
        "retrieval_records": [dict(item) for item in context.retrieval_records],
        "retrieval_digest": context.retrieval_digest, "model_digest": context.model_digest,
        "adapter_digest": context.adapter_digest, "prompt_digest": context.prompt_digest,
    }


def _context_from_mapping(value: Mapping[str, Any]) -> CandidateContext:
    fields = tuple(CandidateContext.__dataclass_fields__)
    legacy_fields = fields[:-3]
    if not isinstance(value, Mapping) or set(value) != set(legacy_fields):
        raise VariationCheckpointError("private candidate context is not closed")
    payload = dict(value)
    records = payload.get("retrieval_records")
    if not isinstance(records, list) or any(not isinstance(item, Mapping) for item in records):
        raise VariationCheckpointError("private candidate retrieval records are malformed")
    payload["retrieval_records"] = tuple(dict(item) for item in records)
    payload.update({"task_statement": None, "initial_source": None, "initial_source_digest": None})
    return CandidateContext(**payload)


class PrivateTrajectoryStore:
    """Write-once context/source records excluded from public projections."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root).resolve()
        self.records = self.root / "records"
        self.records.mkdir(parents=True, exist_ok=True)
        self.artifacts = ContentAddressedArtifactStore(self.root / "artifacts")

    def record(self, *, candidate_id: str, context: CandidateContext, source: bytes) -> None:
        if not isinstance(candidate_id, str) or not candidate_id or "/" in candidate_id or "\\" in candidate_id:
            raise VariationCheckpointError("private trajectory candidate ID is invalid")
        if type(context) is not CandidateContext or not isinstance(source, bytes) or not source:
            raise VariationCheckpointError("private trajectory sidecar input is not exact")
        ref = self.artifacts.put(source, media_type="text/x-python", role="private-candidate-source")
        value = {
            "schema_version": PRIVATE_TRAJECTORY_SCHEMA,
            "candidate_id": candidate_id,
            "context": _context_dict(context),
            "source_digest": ref.digest,
        }
        encoded = canonical_bytes(value)
        path = self.records / (candidate_id + ".json")
        if path.exists():
            if path.read_bytes() != encoded:
                raise VariationCheckpointError("private trajectory sidecar conflicts with durable content")
            return
        descriptor, name = tempfile.mkstemp(prefix=".private-trajectory-", dir=str(self.records))
        temporary = Path(name)
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            temporary.chmod(0o400)
            os.replace(str(temporary), str(path))
        finally:
            if temporary.exists():
                temporary.unlink()

    def load_attempts(self, candidate_ids: Iterable[str]) -> Tuple[object, ...]:
        from ..training.contracts import PrivateTrajectoryAttempt

        attempts = []
        for candidate_id in sorted(set(candidate_ids)):
            path = self.records / (candidate_id + ".json")
            try:
                raw = path.read_bytes()
                value = json.loads(raw.decode("utf-8"))
            except (OSError, UnicodeError, ValueError) as exc:
                raise VariationCheckpointError("private trajectory sidecar is missing or unreadable") from exc
            if canonical_bytes(value) != raw or set(value) != {
                "schema_version", "candidate_id", "context", "source_digest"
            }:
                raise VariationCheckpointError("private trajectory sidecar is not canonical and closed")
            if value["schema_version"] != PRIVATE_TRAJECTORY_SCHEMA or value["candidate_id"] != candidate_id:
                raise VariationCheckpointError("private trajectory sidecar identity is invalid")
            context = _context_from_mapping(value["context"])
            source = self.artifacts.read(value["source_digest"])
            if digest_bytes(source) != value["source_digest"]:
                raise VariationCheckpointError("private trajectory source digest is invalid")
            attempts.append(PrivateTrajectoryAttempt(candidate_id, context, source))
        return tuple(attempts)


__all__ = ["PRIVATE_TRAJECTORY_SCHEMA", "PrivateTrajectoryStore"]
