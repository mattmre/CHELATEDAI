"""Model-visible contracts for the repaired static and AVO-inspired pilots."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from ..canonical import canonical_json, digest_bytes, digest_for


class RecommissioningProtocolError(ValueError):
    """A prompt, feedback record, or coordinate crossed its frozen boundary."""


PROTOCOL_IDS = ("P0", "P1")
MODEL_BUNDLE_IDS = ("LIGHTNING", "SUPER")
DIAGNOSTIC_IDS = (
    "PASS",
    "WRONG_OUTPUT",
    "SYNTAX_OR_IMPORT",
    "RUNTIME_EXCEPTION",
    "TIMEOUT",
    "RESOURCE_LIMIT",
    "MUTATION_LOCUS_VIOLATION",
    "PROTOCOL_VIOLATION",
)


@dataclass(frozen=True)
class PilotCoordinate:
    campaign_id: str
    model_bundle: str
    protocol_id: str
    task_id: str
    task_family: str
    seed: int

    def validate(self) -> None:
        for value, label in (
            (self.campaign_id, "campaign ID"),
            (self.task_id, "task ID"),
            (self.task_family, "task family"),
        ):
            if (
                not isinstance(value, str)
                or not value
                or len(value) > 128
                or any(character in value for character in ("/", "\\", "\x00"))
            ):
                raise RecommissioningProtocolError("{} is not a bounded identity".format(label))
        if self.model_bundle not in MODEL_BUNDLE_IDS:
            raise RecommissioningProtocolError("coordinate uses an unknown model bundle")
        if self.protocol_id not in PROTOCOL_IDS:
            raise RecommissioningProtocolError("coordinate uses an unknown protocol")
        if not isinstance(self.seed, int) or self.seed < 0 or self.seed > 2**31 - 1:
            raise RecommissioningProtocolError("coordinate seed is outside the frozen range")

    @property
    def coordinate_id(self) -> str:
        self.validate()
        return digest_for(
            {
                "campaign_id": self.campaign_id,
                "model_bundle": self.model_bundle,
                "protocol_id": self.protocol_id,
                "task_id": self.task_id,
                "task_family": self.task_family,
                "seed": self.seed,
            }
        )


@dataclass(frozen=True)
class VisibleTaskPackage:
    task_id: str
    task_family: str
    public_specification: str
    initial_source: str
    public_examples: Tuple[Mapping[str, Any], ...]
    public_locus: str
    manifest_digest: str

    def public_manifest(self) -> Dict[str, Any]:
        return {
            "schema_version": "egv-learnable-visible-task-v2",
            "task_id": self.task_id,
            "task_family": self.task_family,
            "public_specification": self.public_specification,
            "initial_source": self.initial_source,
            "public_examples": [dict(example) for example in self.public_examples],
            "public_locus": self.public_locus,
        }

    def validate(self, coordinate: PilotCoordinate) -> None:
        coordinate.validate()
        if self.task_id != coordinate.task_id or self.task_family != coordinate.task_family:
            raise RecommissioningProtocolError("visible task package differs from its coordinate")
        for value, label, ceiling in (
            (self.public_specification, "public specification", 32 * 1024),
            (self.initial_source, "initial source", 256 * 1024),
            (self.public_locus, "public locus", 512),
        ):
            if not isinstance(value, str) or not value.strip() or len(value.encode("utf-8")) > ceiling:
                raise RecommissioningProtocolError("{} is empty or exceeds its byte ceiling".format(label))
        if (
            not isinstance(self.public_examples, tuple)
            or len(self.public_examples) > 8
            or any(not isinstance(example, Mapping) for example in self.public_examples)
        ):
            raise RecommissioningProtocolError("public examples are outside the frozen envelope")
        if not isinstance(self.manifest_digest, str) or len(self.manifest_digest) != 64:
            raise RecommissioningProtocolError("visible task manifest digest is malformed")
        if self.manifest_digest != digest_for(self.public_manifest()):
            raise RecommissioningProtocolError("visible task manifest digest does not bind the package")
        prohibited = ("expected_output", "golden_patch", "oracle_token", "evaluator_seed", "private_key")
        visible = "\n".join(
            (
                self.public_specification,
                self.initial_source,
                canonical_json(list(self.public_examples)),
                self.public_locus,
            )
        ).lower()
        if any(token in visible for token in prohibited):
            raise RecommissioningProtocolError("visible task package contains evaluator-private vocabulary")


@dataclass(frozen=True)
class AttemptFeedback:
    coordinate_id: str
    evidence_id: str
    attempt_index: int
    diagnostic: str
    source_digest: str
    source: str
    public_summary: str

    def validate(self) -> None:
        if not isinstance(self.coordinate_id, str) or len(self.coordinate_id) != 64:
            raise RecommissioningProtocolError("feedback coordinate digest is malformed")
        if not isinstance(self.evidence_id, str) or not self.evidence_id or len(self.evidence_id) > 128:
            raise RecommissioningProtocolError("feedback evidence ID is not bounded")
        if not isinstance(self.attempt_index, int) or self.attempt_index < 1 or self.attempt_index > 6:
            raise RecommissioningProtocolError("feedback attempt index is outside Stage-A budget")
        if self.diagnostic not in DIAGNOSTIC_IDS:
            raise RecommissioningProtocolError("feedback diagnostic is outside the visible enum")
        if not isinstance(self.source_digest, str) or len(self.source_digest) != 64:
            raise RecommissioningProtocolError("feedback source digest is malformed")
        if (
            not isinstance(self.source, str)
            or not self.source.strip()
            or len(self.source.encode("utf-8")) > 64 * 1024
            or digest_bytes(self.source.encode("utf-8")) != self.source_digest
        ):
            raise RecommissioningProtocolError("feedback source bytes do not match the bounded digest")
        if not isinstance(self.public_summary, str) or len(self.public_summary.encode("utf-8")) > 1024:
            raise RecommissioningProtocolError("feedback summary exceeds the byte ceiling")
        lowered = self.public_summary.lower()
        if any(token in lowered for token in ("expected output", "golden", "oracle", "evaluator seed", "hidden path")):
            raise RecommissioningProtocolError("feedback summary contains evaluator-private material")

    def record(self) -> Dict[str, Any]:
        self.validate()
        return {
            "coordinate_id": self.coordinate_id,
            "evidence_id": self.evidence_id,
            "attempt_index": self.attempt_index,
            "diagnostic": self.diagnostic,
            "source_digest": self.source_digest,
            "source": self.source,
            "public_summary": self.public_summary,
        }


@dataclass(frozen=True)
class PromptEnvelope:
    coordinate_id: str
    protocol_id: str
    attempt_index: int
    visible_evidence_ids: Tuple[str, ...]
    messages: Tuple[Mapping[str, str], ...]
    prompt_digest: str


def _feedback_records(
    feedback: Sequence[AttemptFeedback], *, coordinate_id: str
) -> Tuple[Dict[str, Any], ...]:
    records = []
    seen = set()
    for item in feedback:
        item.validate()
        if item.coordinate_id != coordinate_id:
            raise RecommissioningProtocolError("feedback crossed an arm/task/seed coordinate boundary")
        if item.attempt_index in seen:
            raise RecommissioningProtocolError("feedback contains duplicate attempt indices")
        seen.add(item.attempt_index)
        records.append(item.record())
    records.sort(key=lambda value: value["attempt_index"])
    if [record["attempt_index"] for record in records] != list(range(1, len(records) + 1)):
        raise RecommissioningProtocolError("feedback is not a contiguous same-coordinate lineage")
    return tuple(records)


def render_prompt(
    coordinate: PilotCoordinate,
    task: VisibleTaskPackage,
    *,
    attempt_index: int,
    feedback: Sequence[AttemptFeedback] = (),
) -> PromptEnvelope:
    """Render P0 or P1 while enforcing the same task/seed/split reset boundary.

    The caller owns one fresh feedback sequence per coordinate.  This function
    accepts no campaign-global memory and therefore cannot silently carry state
    across arms, tasks, seeds, or evaluation splits.
    """

    coordinate.validate()
    task.validate(coordinate)
    if not isinstance(attempt_index, int) or attempt_index < 1 or attempt_index > 6:
        raise RecommissioningProtocolError("attempt index is outside the frozen six-candidate budget")
    records = _feedback_records(feedback, coordinate_id=coordinate.coordinate_id)
    if len(records) != attempt_index - 1:
        raise RecommissioningProtocolError("feedback lineage does not match the next attempt")
    if coordinate.protocol_id == "P0" and len(records) > 1:
        visible_records = records[-1:]
    else:
        visible_records = records
    visible_ids = tuple(sorted(record["evidence_id"] for record in visible_records))
    system = (
        "You are a bounded code-repair variation operator. The public specification is authoritative. "
        "Return one schema-constrained candidate only. Never request hidden tests, expected values, "
        "credentials, network access, filesystem discovery, or evaluator authority."
    )
    protocol = {
        "P0": (
            "Static control: propose one complete repair from the current public task and, when present, "
            "only the immediately preceding bounded diagnostic. Do not infer persistent memory."
        ),
        "P1": (
            "AVO-inspired treatment: inspect the same-coordinate lineage, avoid repeated source digests, "
            "state a short falsifiable change summary, and choose PROPOSE, EDIT, CRITIQUE, or COMMIT. "
            "The supervisor is deterministic and supplies no extra model call or private information."
        ),
    }[coordinate.protocol_id]
    user_value = {
        "schema_version": "egv-recommissioning-visible-prompt-v1",
        "coordinate_id": coordinate.coordinate_id,
        "protocol_id": coordinate.protocol_id,
        "attempt_index": attempt_index,
        "task": {
            **task.public_manifest(),
            "manifest_digest": task.manifest_digest,
        },
        "visible_feedback": list(visible_records),
        "response_contract": {
            "fields": ["action", "source", "summary", "evidence_ids"],
            "source": "complete Python module implementing the public locus",
            "evidence_ids": list(visible_ids),
        },
    }
    messages = (
        {"role": "system", "content": system},
        {"role": "user", "content": protocol + "\n" + canonical_json(user_value)},
    )
    digest = digest_for([dict(message) for message in messages])
    return PromptEnvelope(
        coordinate_id=coordinate.coordinate_id,
        protocol_id=coordinate.protocol_id,
        attempt_index=attempt_index,
        visible_evidence_ids=visible_ids,
        messages=messages,
        prompt_digest=digest,
    )


def lineage_feedback(
    *,
    coordinate: PilotCoordinate,
    attempt_index: int,
    diagnostic: str,
    source: str,
    public_summary: str,
) -> AttemptFeedback:
    coordinate.validate()
    source_digest = digest_bytes(source.encode("utf-8"))
    evidence_id = "egv-feedback-{}-{}-{}".format(
        coordinate.coordinate_id[:12], attempt_index, source_digest[:12]
    )
    result = AttemptFeedback(
        coordinate.coordinate_id,
        evidence_id,
        attempt_index,
        diagnostic,
        source_digest,
        source,
        public_summary,
    )
    result.validate()
    return result


__all__ = [
    "AttemptFeedback",
    "DIAGNOSTIC_IDS",
    "MODEL_BUNDLE_IDS",
    "PROTOCOL_IDS",
    "PilotCoordinate",
    "PromptEnvelope",
    "RecommissioningProtocolError",
    "VisibleTaskPackage",
    "lineage_feedback",
    "render_prompt",
]
