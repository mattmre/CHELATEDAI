"""Canonical commissioning generation envelopes and accepted evidence gates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Sequence, Tuple

from ..canonical import content_id, digest_for, validate_sha256
from ..identities import commissioning_run_id
from ..receipts import verify_receipt
from ..variation.arms import arm_policy


REQUEST_SCHEMA = "egv-commissioning-generation-request-v1"
RESPONSE_SCHEMA = "egv-commissioning-generation-response-v1"
ACCEPTED_EVIDENCE_SCHEMA = "egv-commissioning-accepted-evidence-v1"
COMMISSIONING_ARMS = ("B", "D")
COMMISSIONING_SEEDS = (0, 1)
_REQUEST_FIELDS = frozenset(
    {
        "schema_version", "request_id", "campaign_id", "run_id", "task_id", "task_family",
        "task_record_digest", "corpus_manifest_digest", "arm_id", "arm_policy_digest", "seed",
        "model_manifest_digest", "variation_protocol_digest", "status",
    }
)
_RESPONSE_FIELDS = frozenset(
    {
        "schema_version", "response_id", "request_id", "candidate_id",
        "candidate_artifact_digest", "model_output_digest", "output_byte_count", "disposition",
        "receipts",
    }
)


class CommissioningTrajectoryError(RuntimeError):
    """Commissioning generation evidence is incomplete, ambiguous, or untrusted."""


def _digest(value: Any, field: str) -> str:
    try:
        return validate_sha256(value, field)
    except Exception as exc:
        raise CommissioningTrajectoryError(str(exc)) from exc


def _identifier(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value or len(value) > 255 or "/" in value or "\\" in value:
        raise CommissioningTrajectoryError("{} is not a bounded identifier".format(field))
    return value


@dataclass(frozen=True)
class GenerationRequest:
    campaign_id: str
    run_id: str
    task_id: str
    task_family: str
    task_record_digest: str
    corpus_manifest_digest: str
    arm_id: str
    arm_policy_digest: str
    seed: int
    model_manifest_digest: str
    variation_protocol_digest: str
    request_id: str
    status: str = "PENDING"
    schema_version: str = REQUEST_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != REQUEST_SCHEMA or self.status != "PENDING":
            raise CommissioningTrajectoryError("generation request must be a pending commissioning envelope")
        for field in ("campaign_id", "run_id", "task_id", "task_family", "request_id"):
            _identifier(getattr(self, field), field)
        if self.arm_id not in COMMISSIONING_ARMS:
            raise CommissioningTrajectoryError("commissioning generation is restricted to arms B and D")
        if self.seed not in COMMISSIONING_SEEDS or isinstance(self.seed, bool):
            raise CommissioningTrajectoryError("commissioning generation seed is outside the frozen set")
        for field in (
            "task_record_digest", "corpus_manifest_digest", "arm_policy_digest",
            "model_manifest_digest", "variation_protocol_digest",
        ):
            _digest(getattr(self, field), field)
        if self.arm_policy_digest != arm_policy(self.arm_id).digest:
            raise CommissioningTrajectoryError("request arm policy differs from frozen Variation policy")
        unsigned = self.to_dict()
        supplied = unsigned.pop("request_id")
        if supplied != content_id("genreq", unsigned):
            raise CommissioningTrajectoryError("generation request ID is not content-derived")

    @classmethod
    def build(
        cls,
        *,
        campaign_id: str,
        task_record: Mapping[str, Any],
        corpus_manifest_digest: str,
        arm_id: str,
        seed: int,
        model_manifest_digest: str,
        variation_protocol_digest: str,
    ) -> "GenerationRequest":
        if not isinstance(task_record, Mapping) or task_record.get("split") != "train":
            raise CommissioningTrajectoryError("generation requests require an immutable train task record")
        required = {
            "template_id", "family_id", "split", "ordinal", "source_digest", "public_rule_id", "public_locus"
        }
        if set(task_record) != required:
            raise CommissioningTrajectoryError("train task record is not the closed evaluation schema")
        policy = arm_policy(arm_id)
        payload = {
            "schema_version": REQUEST_SCHEMA,
            "campaign_id": campaign_id,
            "run_id": commissioning_run_id(
                campaign_id=campaign_id,
                task_id=task_record["template_id"],
                arm_id=arm_id,
                seed=seed,
            ),
            "task_id": task_record["template_id"],
            "task_family": task_record["family_id"],
            "task_record_digest": digest_for(dict(task_record)),
            "corpus_manifest_digest": corpus_manifest_digest,
            "arm_id": arm_id,
            "arm_policy_digest": policy.digest,
            "seed": seed,
            "model_manifest_digest": model_manifest_digest,
            "variation_protocol_digest": variation_protocol_digest,
            "status": "PENDING",
        }
        payload["request_id"] = content_id("genreq", payload)
        return cls.from_mapping(payload)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GenerationRequest":
        if not isinstance(value, Mapping) or set(value) != _REQUEST_FIELDS:
            raise CommissioningTrajectoryError("generation request is not a closed schema")
        return cls(**{field: value[field] for field in _REQUEST_FIELDS})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "request_id": self.request_id,
            "campaign_id": self.campaign_id,
            "run_id": self.run_id,
            "task_id": self.task_id,
            "task_family": self.task_family,
            "task_record_digest": self.task_record_digest,
            "corpus_manifest_digest": self.corpus_manifest_digest,
            "arm_id": self.arm_id,
            "arm_policy_digest": self.arm_policy_digest,
            "seed": self.seed,
            "model_manifest_digest": self.model_manifest_digest,
            "variation_protocol_digest": self.variation_protocol_digest,
            "status": self.status,
        }

    @property
    def digest(self) -> str:
        return digest_for(self.to_dict())


@dataclass(frozen=True)
class GenerationResponse:
    request_id: str
    candidate_id: str
    candidate_artifact_digest: str
    model_output_digest: str
    output_byte_count: int
    disposition: str
    receipts: Tuple[Mapping[str, Any], ...]
    response_id: str
    schema_version: str = RESPONSE_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != RESPONSE_SCHEMA or self.disposition != "PROMOTED":
            raise CommissioningTrajectoryError("generation response is not an accepted promoted envelope")
        for field in ("request_id", "candidate_id", "response_id"):
            _identifier(getattr(self, field), field)
        if (
            not isinstance(self.output_byte_count, int)
            or isinstance(self.output_byte_count, bool)
            or self.output_byte_count <= 0
            or self.output_byte_count > 256 * 1024
        ):
            raise CommissioningTrajectoryError("model output byte count is outside the bounded source contract")
        artifact = _digest(self.candidate_artifact_digest, "candidate artifact digest")
        if _digest(self.model_output_digest, "model output digest") != artifact:
            raise CommissioningTrajectoryError("model output digest differs from evaluated candidate bytes")
        if not isinstance(self.receipts, tuple) or len(self.receipts) != 3:
            raise CommissioningTrajectoryError("accepted response requires authority, verdict, and effect receipts")
        unsigned = self.to_dict()
        supplied = unsigned.pop("response_id")
        if supplied != content_id("genresp", unsigned):
            raise CommissioningTrajectoryError("generation response ID is not content-derived")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "GenerationResponse":
        if not isinstance(value, Mapping) or set(value) != _RESPONSE_FIELDS:
            raise CommissioningTrajectoryError("generation response is not a closed schema")
        receipts = value["receipts"]
        if not isinstance(receipts, list):
            raise CommissioningTrajectoryError("generation response receipts must be an array")
        payload = dict(value)
        payload["receipts"] = tuple(dict(item) if isinstance(item, Mapping) else item for item in receipts)
        return cls(**payload)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "response_id": self.response_id,
            "request_id": self.request_id,
            "candidate_id": self.candidate_id,
            "candidate_artifact_digest": self.candidate_artifact_digest,
            "model_output_digest": self.model_output_digest,
            "output_byte_count": self.output_byte_count,
            "disposition": self.disposition,
            "receipts": [dict(receipt) for receipt in self.receipts],
        }

    @property
    def digest(self) -> str:
        return digest_for(self.to_dict())


def validate_accepted_response(
    request: GenerationRequest,
    response: GenerationResponse,
    *,
    evaluator_public_key: Any,
    evaluator_digest: str,
    expected_first_sequence: int,
    expected_previous_receipt_hash: str,
) -> Dict[str, Any]:
    """Verify real evaluator receipts and return only accepted B/D evidence."""

    if type(request) is not GenerationRequest or type(response) is not GenerationResponse:
        raise CommissioningTrajectoryError("accepted response validation requires exact envelope types")
    pinned_evaluator_digest = _digest(evaluator_digest, "expected evaluator digest")
    if (
        not isinstance(expected_first_sequence, int)
        or isinstance(expected_first_sequence, bool)
        or expected_first_sequence < 1
    ):
        raise CommissioningTrajectoryError("expected first receipt sequence is invalid")
    expected_previous = _digest(
        expected_previous_receipt_hash,
        "expected previous receipt hash",
    )
    response_content = response.to_dict()
    supplied_response_id = response_content.pop("response_id")
    if supplied_response_id != content_id("genresp", response_content):
        raise CommissioningTrajectoryError("generation response changed after envelope construction")
    if response.request_id != request.request_id:
        raise CommissioningTrajectoryError("generation response is bound to a different request")
    expected_types = ("AUTHORITY", "VERDICT", "EFFECT")
    expected_decisions = ("ALLOW", "PASS", "ALLOW")
    previous = expected_previous
    receipt_digests = []
    for index, (receipt, receipt_type, decision) in enumerate(
        zip(response.receipts, expected_types, expected_decisions),
        start=1,
    ):
        if receipt.get("receipt_type") != receipt_type or receipt.get("decision") != decision:
            raise CommissioningTrajectoryError("accepted response receipt decision chain is not ALLOW/PASS/ALLOW")
        expected_sequence = expected_first_sequence + index - 1
        try:
            verified = verify_receipt(
                receipt,
                evaluator_public_key,
                expected_sequence=expected_sequence,
                expected_previous_hash=previous,
            )
        except Exception as exc:
            raise CommissioningTrajectoryError("accepted response evaluator receipt is invalid") from exc
        if (
            receipt.get("campaign_id") != request.campaign_id
            or receipt.get("run_id") != request.run_id
            or receipt.get("task_id") != request.task_id
            or receipt.get("candidate_id") != response.candidate_id
            or receipt.get("protocol_digest") != request.variation_protocol_digest
            or receipt.get("policy_digest") != request.arm_policy_digest
            or receipt.get("evaluator_digest") != pinned_evaluator_digest
            or receipt.get("candidate_artifact_digest") != response.candidate_artifact_digest
        ):
            raise CommissioningTrajectoryError("accepted response receipt does not bind the generation request")
        previous = verified
        receipt_digests.append(verified)
    verdict = response.receipts[1]
    effect = response.receipts[2]
    if (
        verdict.get("diagnostic_enum") != "PASS"
        or verdict.get("exit_status_class") != "SUCCESS"
        or effect.get("exit_status_class") != "SUCCESS"
        or any(receipt.get("infrastructure_incident_id") is not None for receipt in response.receipts)
    ):
        raise CommissioningTrajectoryError("accepted response receipts do not prove a successful PASS execution")
    return {
        "schema_version": ACCEPTED_EVIDENCE_SCHEMA,
        "request_id": request.request_id,
        "response_id": response.response_id,
        "campaign_id": request.campaign_id,
        "task_id": request.task_id,
        "arm_id": request.arm_id,
        "seed": request.seed,
        "candidate_id": response.candidate_id,
        "candidate_artifact_digest": response.candidate_artifact_digest,
        "receipt_chain_anchor": expected_previous,
        "receipt_digests": receipt_digests,
        "disposition": "PROMOTED",
        "diagnostic_enum": "PASS",
    }


def reconcile_responses(
    requests: Sequence[GenerationRequest],
    responses: Sequence[GenerationResponse],
    *,
    evaluator_public_key: Any,
    evaluator_digest: str,
    expected_first_sequence: int,
    expected_previous_receipt_hash: str,
    require_complete: bool = False,
) -> Dict[str, Any]:
    """Idempotently reconcile actual responses without manufacturing missing evidence."""

    if any(type(request) is not GenerationRequest for request in requests):
        raise CommissioningTrajectoryError("response reconciliation requires validated generation requests")
    pinned_evaluator_digest = _digest(evaluator_digest, "expected evaluator digest")
    if (
        not isinstance(expected_first_sequence, int)
        or isinstance(expected_first_sequence, bool)
        or expected_first_sequence < 1
    ):
        raise CommissioningTrajectoryError("expected first receipt sequence is invalid")
    pinned_previous_receipt_hash = _digest(
        expected_previous_receipt_hash,
        "expected previous receipt hash",
    )
    by_request = {request.request_id: request for request in requests}
    if len(by_request) != len(requests):
        raise CommissioningTrajectoryError("generation request IDs are duplicated")
    accepted: Dict[str, Dict[str, Any]] = {}
    response_ids: Dict[str, str] = {}
    next_sequence = expected_first_sequence
    previous_receipt_hash = pinned_previous_receipt_hash
    for response in responses:
        if type(response) is not GenerationResponse:
            raise CommissioningTrajectoryError("response reconciliation requires validated response envelopes")
        request = by_request.get(response.request_id)
        if request is None:
            raise CommissioningTrajectoryError("generation response has no frozen request")
        prior = response_ids.get(response.request_id)
        if prior is not None and prior != response.response_id:
            raise CommissioningTrajectoryError("conflicting responses target one generation request")
        if prior == response.response_id:
            continue
        response_ids[response.request_id] = response.response_id
        accepted[response.request_id] = validate_accepted_response(
            request,
            response,
            evaluator_public_key=evaluator_public_key,
            evaluator_digest=pinned_evaluator_digest,
            expected_first_sequence=next_sequence,
            expected_previous_receipt_hash=previous_receipt_hash,
        )
        next_sequence += len(response.receipts)
        previous_receipt_hash = accepted[response.request_id]["receipt_digests"][-1]
    missing = sorted(set(by_request) - set(accepted))
    if require_complete and missing:
        raise CommissioningTrajectoryError("commissioning responses are incomplete; no completion may be claimed")
    ordered = [accepted[request.request_id] for request in requests if request.request_id in accepted]
    return {
        "schema_version": "egv-commissioning-response-manifest-v1",
        "request_count": len(requests),
        "accepted_response_count": len(ordered),
        "missing_request_count": len(missing),
        "status": "COMPLETE" if not missing else "PENDING",
        "accepted_evidence": ordered,
        "accepted_evidence_digest": digest_for(ordered),
        "next_receipt_sequence": next_sequence,
        "last_receipt_digest": previous_receipt_hash,
    }


__all__ = [
    "ACCEPTED_EVIDENCE_SCHEMA", "COMMISSIONING_ARMS", "COMMISSIONING_SEEDS",
    "CommissioningTrajectoryError", "GenerationRequest", "GenerationResponse", "REQUEST_SCHEMA",
    "RESPONSE_SCHEMA", "commissioning_run_id", "reconcile_responses", "validate_accepted_response",
]
