"""Pinned evaluator authority and signed Campaign gate receipts."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from ..canonical import GENESIS_HASH, canonical_bytes, content_id, digest_for, validate_sha256
from ..receipts import key_id_for_public_key, load_public_key
from .errors import CampaignError


class EvaluatorAuthorityError(CampaignError):
    """An evaluator decision is unsigned, replayed, or outside its authority."""


AUTHORITY_SCHEMA = "egv-campaign-evaluator-receipt-v1"
_FIELDS = frozenset(
    {
        "schema_version", "campaign_id", "phase", "transfer_digest", "artifact_set_digest",
        "protocol_digest", "evaluator_digest", "decision", "metrics_digest", "sequence",
        "previous_receipt_digest", "signing_key_id", "receipt_id", "signature",
    }
)


def _sha(value: Any, field: str) -> str:
    try:
        return validate_sha256(value, field)
    except Exception as exc:
        raise EvaluatorAuthorityError(str(exc)) from exc


def _decode_signature(value: Any) -> bytes:
    if not isinstance(value, str) or not value:
        raise EvaluatorAuthorityError("evaluator signature is absent")
    try:
        return base64.urlsafe_b64decode((value + "=" * (-len(value) % 4)).encode("ascii"))
    except (ValueError, UnicodeError) as exc:
        raise EvaluatorAuthorityError("evaluator signature is not base64url") from exc


@dataclass(frozen=True)
class EvaluatorReceipt:
    campaign_id: str
    phase: str
    transfer_digest: str
    artifact_set_digest: str
    protocol_digest: str
    evaluator_digest: str
    decision: str
    metrics_digest: str
    sequence: int
    previous_receipt_digest: str
    signing_key_id: str
    receipt_id: str
    signature: str
    schema_version: str = AUTHORITY_SCHEMA

    def __post_init__(self) -> None:
        if self.schema_version != AUTHORITY_SCHEMA:
            raise EvaluatorAuthorityError("unsupported evaluator receipt schema")
        for field in ("campaign_id", "phase", "signing_key_id", "receipt_id"):
            if not isinstance(getattr(self, field), str) or not getattr(self, field):
                raise EvaluatorAuthorityError("{} must be non-empty".format(field))
        if self.decision not in {"ACCEPT", "REJECT"}:
            raise EvaluatorAuthorityError("evaluator decision is outside the closed vocabulary")
        if not isinstance(self.sequence, int) or isinstance(self.sequence, bool) or self.sequence < 1:
            raise EvaluatorAuthorityError("evaluator sequence must be positive")
        for field in (
            "transfer_digest", "artifact_set_digest", "protocol_digest", "evaluator_digest",
            "metrics_digest", "previous_receipt_digest",
        ):
            _sha(getattr(self, field), field)
        _decode_signature(self.signature)

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "EvaluatorReceipt":
        if not isinstance(value, Mapping) or set(value) != _FIELDS:
            raise EvaluatorAuthorityError("evaluator receipt is not a closed schema")
        if value["schema_version"] != AUTHORITY_SCHEMA:
            raise EvaluatorAuthorityError("unsupported evaluator receipt schema")
        for field in ("campaign_id", "phase", "signing_key_id", "receipt_id"):
            if not isinstance(value[field], str) or not value[field]:
                raise EvaluatorAuthorityError("{} must be non-empty".format(field))
        if value["decision"] not in {"ACCEPT", "REJECT"}:
            raise EvaluatorAuthorityError("evaluator decision is outside the closed vocabulary")
        sequence = value["sequence"]
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 1:
            raise EvaluatorAuthorityError("evaluator sequence must be positive")
        for field in (
            "transfer_digest", "artifact_set_digest", "protocol_digest", "evaluator_digest",
            "metrics_digest", "previous_receipt_digest",
        ):
            _sha(value[field], field)
        _decode_signature(value["signature"])
        return cls(**{field: value[field] for field in _FIELDS})

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version, "campaign_id": self.campaign_id,
            "phase": self.phase, "transfer_digest": self.transfer_digest,
            "artifact_set_digest": self.artifact_set_digest, "protocol_digest": self.protocol_digest,
            "evaluator_digest": self.evaluator_digest, "decision": self.decision,
            "metrics_digest": self.metrics_digest, "sequence": self.sequence,
            "previous_receipt_digest": self.previous_receipt_digest,
            "signing_key_id": self.signing_key_id, "receipt_id": self.receipt_id,
            "signature": self.signature,
        }

    @property
    def digest(self) -> str:
        return digest_for(self.to_dict())


def sign_evaluator_receipt(fields: Mapping[str, Any], signer: Any) -> EvaluatorReceipt:
    """Evaluator-side helper; coordinator code receives only the public key."""

    payload = dict(fields)
    payload["schema_version"] = AUTHORITY_SCHEMA
    payload["signing_key_id"] = signer.key_id
    payload.pop("signature", None)
    payload.pop("receipt_id", None)
    payload["receipt_id"] = content_id("eval", payload)
    payload["signature"] = signer.sign_bytes(canonical_bytes(payload))
    return EvaluatorReceipt.from_mapping(payload)


class EvaluatorAuthority:
    def __init__(self, public_key: Any, *, campaign_id: str, protocol_digest: str, evaluator_digest: str) -> None:
        try:
            self.public_key = load_public_key(public_key)
            self.key_id = key_id_for_public_key(self.public_key)
        except Exception as exc:
            raise EvaluatorAuthorityError("evaluator public key is invalid") from exc
        if not isinstance(campaign_id, str) or not campaign_id:
            raise EvaluatorAuthorityError("authority campaign ID must be non-empty")
        self.campaign_id = campaign_id
        self.protocol_digest = _sha(protocol_digest, "authority protocol digest")
        self.evaluator_digest = _sha(evaluator_digest, "authority evaluator digest")

    def verify(
        self,
        receipt: EvaluatorReceipt,
        *,
        phase: str,
        transfer_digest: str,
        artifact_set_digest: str,
        expected_sequence: int,
        expected_previous_digest: Optional[str],
    ) -> str:
        if type(receipt) is not EvaluatorReceipt:
            raise EvaluatorAuthorityError("authority requires an exact validated evaluator receipt")
        expected_previous = expected_previous_digest or GENESIS_HASH
        bindings = (
            receipt.campaign_id == self.campaign_id,
            receipt.phase == phase,
            receipt.transfer_digest == _sha(transfer_digest, "expected transfer digest"),
            receipt.artifact_set_digest == _sha(artifact_set_digest, "expected artifact set digest"),
            receipt.protocol_digest == self.protocol_digest,
            receipt.evaluator_digest == self.evaluator_digest,
            receipt.sequence == expected_sequence,
            receipt.previous_receipt_digest == _sha(expected_previous, "expected previous receipt digest"),
            receipt.signing_key_id == self.key_id,
        )
        if not all(bindings):
            raise EvaluatorAuthorityError("evaluator receipt authority or chain binding failed")
        unsigned = receipt.to_dict()
        signature = _decode_signature(unsigned.pop("signature"))
        supplied_id = unsigned.pop("receipt_id")
        if supplied_id != content_id("eval", unsigned):
            raise EvaluatorAuthorityError("evaluator receipt ID is not content-addressed")
        unsigned["receipt_id"] = supplied_id
        try:
            self.public_key.verify(signature, canonical_bytes(unsigned))
        except Exception as exc:
            raise EvaluatorAuthorityError("evaluator receipt signature is invalid") from exc
        return receipt.digest


__all__ = [
    "AUTHORITY_SCHEMA", "EvaluatorAuthority", "EvaluatorAuthorityError", "EvaluatorReceipt",
    "sign_evaluator_receipt",
]
