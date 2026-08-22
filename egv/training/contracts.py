"""Frozen private contracts for EGV trajectory training data."""

from __future__ import annotations

from dataclasses import dataclass
import json
from types import MappingProxyType
from typing import Any, Dict, Mapping, Tuple

from ..canonical import canonical_json, content_id, digest_bytes, digest_for, validate_sha256


TRAINING_DATASET_SCHEMA_VERSION = "egv-frozen-training-dataset-v1"
TRAINING_EXAMPLE_SCHEMA_VERSION = "egv-private-training-example-v1"
TRAINING_ARMS = frozenset({"B", "D"})
MAX_TRAINING_TOKENS = 4096


def _digest(value: str, field: str) -> str:
    return validate_sha256(value, field)


@dataclass(frozen=True)
class LedgerCutoff:
    campaign_id: str
    sequence: int
    event_id: str
    event_hash: str
    receipt_chain_head: str
    receipt_count: int
    evaluator_key_id: str

    @classmethod
    def capture(cls, ledger: Any, *, campaign_id: str) -> "LedgerCutoff":
        ledger.verify_integrity()
        row = ledger.connection.execute(
            "SELECT sequence,event_id,event_hash FROM events ORDER BY sequence DESC LIMIT 1"
        ).fetchone()
        if row is None:
            raise ValueError("cannot freeze an empty ledger")
        if not ledger.connection.execute(
            "SELECT 1 FROM campaigns WHERE campaign_id=?", (campaign_id,)
        ).fetchone():
            raise ValueError("ledger cutoff campaign is unknown")
        key_row = ledger.connection.execute("SELECT value FROM meta WHERE key='evaluator_key_id'").fetchone()
        if key_row is None or not str(key_row[0]):
            raise ValueError("cannot freeze a ledger without a pinned evaluator receipt key")
        receipt_count = int(ledger.connection.execute("SELECT COUNT(*) FROM receipts").fetchone()[0])
        if receipt_count < 1:
            raise ValueError("cannot freeze a training ledger without signed receipts")
        return cls(
            campaign_id,
            int(row["sequence"]),
            str(row["event_id"]),
            str(row["event_hash"]),
            str(ledger.receipt_head()),
            receipt_count,
            str(key_row[0]),
        )

    def validate(self, ledger: Any) -> None:
        if not isinstance(self.campaign_id, str) or not self.campaign_id:
            raise ValueError("ledger cutoff campaign_id must be non-empty")
        if not isinstance(self.sequence, int) or isinstance(self.sequence, bool) or self.sequence < 1:
            raise ValueError("ledger cutoff sequence must be positive")
        _digest(self.event_hash, "ledger cutoff event_hash")
        _digest(self.receipt_chain_head, "ledger cutoff receipt_chain_head")
        if not isinstance(self.receipt_count, int) or isinstance(self.receipt_count, bool) or self.receipt_count < 1:
            raise ValueError("ledger cutoff receipt_count must be positive")
        if not isinstance(self.evaluator_key_id, str) or not self.evaluator_key_id:
            raise ValueError("ledger cutoff evaluator_key_id must be non-empty")
        row = ledger.connection.execute(
            "SELECT sequence,event_id,event_hash FROM events WHERE sequence=?", (self.sequence,)
        ).fetchone()
        if row is None or str(row["event_id"]) != self.event_id or str(row["event_hash"]) != self.event_hash:
            raise ValueError("ledger cutoff does not bind the exact durable event")
        # Training is a one-time freeze operation. Building against a ledger
        # that advanced after capture risks correction/retraction races.
        if ledger.ledger_head_event_id() != self.event_id or ledger.ledger_head_hash() != self.event_hash:
            raise ValueError("ledger advanced after the frozen training cutoff")
        if ledger.receipt_head() != self.receipt_chain_head:
            raise ValueError("receipt chain advanced or changed after the frozen training cutoff")
        if int(ledger.connection.execute("SELECT COUNT(*) FROM receipts").fetchone()[0]) != self.receipt_count:
            raise ValueError("receipt count differs from the frozen training cutoff")
        key_row = ledger.connection.execute("SELECT value FROM meta WHERE key='evaluator_key_id'").fetchone()
        if key_row is None or str(key_row[0]) != self.evaluator_key_id:
            raise ValueError("evaluator receipt key differs from the frozen training cutoff")
        ledger.verify_integrity()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "campaign_id": self.campaign_id,
            "sequence": self.sequence,
            "event_id": self.event_id,
            "event_hash": self.event_hash,
            "receipt_chain_head": self.receipt_chain_head,
            "receipt_count": self.receipt_count,
            "evaluator_key_id": self.evaluator_key_id,
        }

    @property
    def digest(self) -> str:
        return digest_for(self.to_dict())


@dataclass(frozen=True)
class PrivateTrajectoryAttempt:
    """Evaluator-private bytes and prompt context omitted from public exports."""

    candidate_id: str
    context: Any
    candidate_source: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.candidate_id, str) or not self.candidate_id:
            raise ValueError("private trajectory candidate_id must be non-empty")
        if not isinstance(self.candidate_source, bytes) or not self.candidate_source:
            raise ValueError("private trajectory candidate source must be non-empty bytes")


@dataclass(frozen=True)
class TrainingExample:
    row_id: str
    campaign_id: str
    run_id: str
    task_id: str
    arm_id: str
    seed: int
    attempt_index: int
    candidate_id: str
    prompt: str
    target: str
    prompt_digest: str
    target_digest: str
    sft_row_json: str
    sft_row_digest: str
    prompt_context_digest: str
    source_digest: str
    verdict_receipt_digest: str
    effect_receipt_digest: str
    cutoff_digest: str
    input_token_count: int

    def __post_init__(self) -> None:
        if self.arm_id not in TRAINING_ARMS:
            raise ValueError("training examples are restricted to Arms B and D")
        if not self.task_id or not self.candidate_id or not self.run_id or not self.campaign_id:
            raise ValueError("training example identity fields must be non-empty")
        if not isinstance(self.prompt, str) or not self.prompt or not isinstance(self.target, str) or not self.target:
            raise ValueError("training example prompt and target must be non-empty private text")
        for field, value in (
            ("prompt_digest", self.prompt_digest),
            ("target_digest", self.target_digest),
            ("sft_row_digest", self.sft_row_digest),
            ("prompt_context_digest", self.prompt_context_digest),
            ("source_digest", self.source_digest),
            ("verdict_receipt_digest", self.verdict_receipt_digest),
            ("effect_receipt_digest", self.effect_receipt_digest),
            ("cutoff_digest", self.cutoff_digest),
        ):
            _digest(value, field)
        if digest_bytes(self.prompt.encode("utf-8")) != self.prompt_digest:
            raise ValueError("private prompt digest mismatch")
        if digest_bytes(self.target.encode("utf-8")) != self.target_digest:
            raise ValueError("private target digest mismatch")
        if not isinstance(self.sft_row_json, str) or not self.sft_row_json:
            raise ValueError("training example requires a canonical egv-sft-row-v1 envelope")
        if digest_bytes(self.sft_row_json.encode("utf-8")) != self.sft_row_digest:
            raise ValueError("SFT row digest mismatch")
        try:
            parsed_sft_row = json.loads(self.sft_row_json)
        except (TypeError, ValueError) as exc:
            raise ValueError("SFT row is not valid canonical JSON") from exc
        if canonical_json(parsed_sft_row) != self.sft_row_json:
            raise ValueError("SFT row JSON is not canonical")
        if parsed_sft_row.get("schema_version") != "egv-sft-row-v1":
            raise ValueError("training example does not carry egv-sft-row-v1")
        if parsed_sft_row.get("task_id") != self.task_id or parsed_sft_row.get("arm") != self.arm_id:
            raise ValueError("SFT row identity differs from the private training example")
        if parsed_sft_row.get("attempt_index") != self.attempt_index:
            raise ValueError("SFT row attempt differs from the private training example")
        if parsed_sft_row.get("candidate_artifact_digest") != self.source_digest:
            raise ValueError("SFT row source binding differs from the private target")
        if parsed_sft_row.get("prompt_digest") != self.prompt_context_digest:
            raise ValueError("SFT row prompt context binding differs from the private prompt")
        if parsed_sft_row.get("diagnostic_enum") != "PASS" or parsed_sft_row.get("promotion_disposition") != "PROMOTED":
            raise ValueError("only verified PASS/PROMOTED SFT rows may become positive targets")
        if not isinstance(self.input_token_count, int) or self.input_token_count < 1:
            raise ValueError("input_token_count must be positive")
        expected = content_id("trainrow", self.manifest_record(include_row_id=False))
        if self.row_id != expected:
            raise ValueError("training row ID is not content-derived")

    def manifest_record(self, *, include_row_id: bool = True) -> Dict[str, Any]:
        result = {
            "schema_version": TRAINING_EXAMPLE_SCHEMA_VERSION,
            "campaign_id": self.campaign_id,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "arm_id": self.arm_id,
            "seed": self.seed,
            "attempt_index": self.attempt_index,
            "candidate_id": self.candidate_id,
            "prompt_digest": self.prompt_digest,
            "target_digest": self.target_digest,
            "sft_row_digest": self.sft_row_digest,
            "prompt_context_digest": self.prompt_context_digest,
            "source_digest": self.source_digest,
            "verdict_receipt_digest": self.verdict_receipt_digest,
            "effect_receipt_digest": self.effect_receipt_digest,
            "cutoff_digest": self.cutoff_digest,
            "input_token_count": self.input_token_count,
        }
        if include_row_id:
            result["row_id"] = self.row_id
        return result

    def private_record(self) -> Dict[str, Any]:
        return {
            **self.manifest_record(),
            "prompt": self.prompt,
            "target": self.target,
            "sft_row": json.loads(self.sft_row_json),
        }


@dataclass(frozen=True)
class FrozenTrainingDataset:
    cutoff: LedgerCutoff
    examples: Tuple[TrainingExample, ...]
    excluded_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        ordered = tuple(sorted(self.examples, key=lambda row: (row.task_id, row.arm_id, row.seed, row.attempt_index, row.candidate_id)))
        if self.examples != ordered:
            raise ValueError("frozen training examples are not deterministically ordered")
        row_ids = [row.row_id for row in self.examples]
        if len(row_ids) != len(set(row_ids)):
            raise ValueError("frozen training rows are not unique")
        if any(row.cutoff_digest != self.cutoff.digest for row in self.examples):
            raise ValueError("training row is bound to a different ledger cutoff")
        normalized = {str(key): int(value) for key, value in self.excluded_counts.items()}
        if any(value < 0 for value in normalized.values()):
            raise ValueError("excluded counts cannot be negative")
        object.__setattr__(self, "excluded_counts", MappingProxyType(dict(sorted(normalized.items()))))

    def manifest(self) -> Dict[str, Any]:
        return {
            "schema_version": TRAINING_DATASET_SCHEMA_VERSION,
            "cutoff": self.cutoff.to_dict(),
            "sequence_packing": False,
            "max_input_tokens": MAX_TRAINING_TOKENS,
            "arms": sorted(TRAINING_ARMS),
            "row_count": len(self.examples),
            "rows": [row.manifest_record() for row in self.examples],
            "excluded_counts": dict(self.excluded_counts),
        }

    @property
    def digest(self) -> str:
        return digest_for(self.manifest())


__all__ = [
    "FrozenTrainingDataset",
    "LedgerCutoff",
    "MAX_TRAINING_TOKENS",
    "PrivateTrajectoryAttempt",
    "TRAINING_ARMS",
    "TrainingExample",
]
