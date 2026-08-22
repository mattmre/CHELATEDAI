"""Ed25519 receipts, receipt hash chains, and durable idempotent journals."""

from __future__ import annotations

import base64
from pathlib import Path
import os
from typing import Any, Dict, Mapping, Optional, Union

from .canonical import (
    GENESIS_HASH,
    canonical_bytes,
    canonical_json,
    content_id,
    digest_for,
    failure_family_root,
    validate_sha256,
)
from .errors import OptionalDependencyError, ReceiptConflictError, ReceiptVerificationError


RECEIPT_SCHEMA_VERSION = "egv-receipt-v1"
PUBLIC_RECEIPT_SCHEMA_VERSION = "egv-public-receipt-v1"
RECEIPT_TYPES = frozenset({"AUTHORITY", "VERDICT", "EFFECT"})
RECEIPT_DECISIONS = {
    "AUTHORITY": frozenset({"ALLOW", "DENY", "ERROR"}),
    "VERDICT": frozenset({"PASS", "FAIL", "ERROR"}),
    "EFFECT": frozenset({"ALLOW", "DENY", "ERROR"}),
}
PUBLIC_EXIT_STATUS_CLASSES = frozenset(
    {
        "NOT_APPLICABLE",
        "SUCCESS",
        "NONZERO",
        "TIMEOUT",
        "SIGNAL",
        "RESOURCE_LIMIT",
        "OUTPUT_LIMIT",
        "INFRASTRUCTURE_LOSS",
    }
)

_RECEIPT_REQUIRED = frozenset(
    {
        "schema_version",
        "receipt_type",
        "campaign_id",
        "run_id",
        "task_id",
        "receipt_id",
        "request_id",
        "candidate_id",
        "decision",
        "sequence",
        "previous_receipt_hash",
        "idempotency_key",
        "signing_key_id",
        "signature",
    }
)
_RECEIPT_OPTIONAL = frozenset(
    {
        "candidate_artifact_digest",
        "protocol_digest",
        "policy_digest",
        "evaluator_digest",
        "diagnostic_enum",
        "resource_bucket",
        "exit_status_class",
        "input_digest",
        "output_digest",
        "environment_diff_digest",
        "normalized_action_hash",
        "sandbox_id",
        "started_at",
        "finished_at",
        "exit_status",
        "effect_kind",
        "infrastructure_incident_id",
        "failure_family_root",
        "task_family",
        "normalized_public_locus",
        "public_rule_id",
        "public_candidate_record_digest",
        "public_dependency_set_digest",
    }
)
_RECEIPT_ALLOWED = _RECEIPT_REQUIRED | _RECEIPT_OPTIONAL


def _crypto_imports() -> tuple[Any, Any, Any, Any]:
    """Import cryptography lazily so the stdlib-only ledger still imports."""

    try:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
        from cryptography.exceptions import InvalidSignature
    except ImportError as exc:  # pragma: no cover - exercised in minimal environments
        raise OptionalDependencyError(
            "Ed25519 receipts require the 'cryptography' package; install the EGV receipt extra"
        ) from exc
    return Ed25519PrivateKey, Ed25519PublicKey, InvalidSignature, serialization


def _b64_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _b64_decode(value: Any, field: str = "signature") -> bytes:
    if not isinstance(value, str) or not value:
        raise ReceiptVerificationError(f"{field} must be a non-empty base64url string")
    padded = value + "=" * (-len(value) % 4)
    try:
        return base64.urlsafe_b64decode(padded.encode("ascii"))
    except (ValueError, UnicodeError) as exc:
        raise ReceiptVerificationError(f"{field} is not valid base64url") from exc


def public_key_bytes(public_key: Any) -> bytes:
    """Return raw Ed25519 public-key bytes from common representations."""

    Ed25519PrivateKey, Ed25519PublicKey, _InvalidSignature, serialization = _crypto_imports()
    if isinstance(public_key, Ed25519PrivateKey):
        public_key = public_key.public_key()
    if isinstance(public_key, Ed25519PublicKey):
        return public_key.public_bytes(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)
    if isinstance(public_key, str):
        public_key = public_key.encode("utf-8")
    if isinstance(public_key, bytes):
        if len(public_key) == 32:
            return public_key
        try:
            loaded = serialization.load_pem_public_key(public_key)
        except (ValueError, TypeError) as exc:
            raise ReceiptVerificationError("public key is neither raw Ed25519 bytes nor PEM") from exc
        if not isinstance(loaded, Ed25519PublicKey):
            raise ReceiptVerificationError("public key is not Ed25519")
        return loaded.public_bytes(encoding=serialization.Encoding.Raw, format=serialization.PublicFormat.Raw)
    raise ReceiptVerificationError("unsupported Ed25519 public-key representation")


def load_public_key(public_key: Any) -> Any:
    """Return a cryptography Ed25519 public-key object."""

    _Ed25519PrivateKey, Ed25519PublicKey, _InvalidSignature, serialization = _crypto_imports()
    if isinstance(public_key, Ed25519PublicKey):
        return public_key
    return Ed25519PublicKey.from_public_bytes(public_key_bytes(public_key))


def key_id_for_public_key(public_key: Any) -> str:
    """Derive a stable campaign key ID from the public key, never the secret key."""

    return content_id("key", public_key_bytes(public_key))


def receipt_hash(receipt: Mapping[str, Any]) -> str:
    """Hash a complete signed receipt, including its signature."""

    return digest_for(dict(receipt))


def _signable(receipt: Mapping[str, Any]) -> Dict[str, Any]:
    signable = dict(receipt)
    signable.pop("signature", None)
    return signable


def _validate_common_receipt(receipt: Mapping[str, Any], *, public: bool = False) -> None:
    if not isinstance(receipt, Mapping):
        raise ReceiptVerificationError("receipt must be an object")
    required = _RECEIPT_REQUIRED
    missing = sorted(required - set(receipt))
    if missing:
        raise ReceiptVerificationError(f"receipt is missing fields: {', '.join(missing)}")
    if not public and set(receipt) - _RECEIPT_ALLOWED:
        extra = sorted(set(receipt) - _RECEIPT_ALLOWED)
        raise ReceiptVerificationError(f"receipt contains forbidden fields: {', '.join(extra)}")
    expected_schema = PUBLIC_RECEIPT_SCHEMA_VERSION if public else RECEIPT_SCHEMA_VERSION
    if receipt.get("schema_version") != expected_schema:
        raise ReceiptVerificationError(f"unsupported receipt schema: {receipt.get('schema_version')!r}")
    receipt_type = receipt.get("receipt_type")
    if receipt_type not in RECEIPT_TYPES:
        raise ReceiptVerificationError(f"unsupported receipt type: {receipt_type!r}")
    decision = receipt.get("decision")
    if decision not in RECEIPT_DECISIONS[receipt_type]:
        raise ReceiptVerificationError(f"decision {decision!r} is invalid for {receipt_type}")
    if "exit_status_class" in receipt and receipt["exit_status_class"] not in PUBLIC_EXIT_STATUS_CLASSES:
        raise ReceiptVerificationError("exit_status_class is outside the closed public vocabulary")
    sequence = receipt.get("sequence")
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 1:
        raise ReceiptVerificationError("receipt sequence must be a positive integer")
    previous = receipt.get("previous_receipt_hash")
    if previous != GENESIS_HASH:
        try:
            validate_sha256(previous, "previous_receipt_hash")
        except Exception as exc:
            raise ReceiptVerificationError(str(exc)) from exc
    for field in (
        "candidate_artifact_digest",
        "protocol_digest",
        "policy_digest",
        "evaluator_digest",
        "input_digest",
        "output_digest",
        "environment_diff_digest",
        "normalized_action_hash",
        "public_candidate_record_digest",
        "public_dependency_set_digest",
        "failure_family_root",
    ):
        if field in receipt and receipt[field] is not None:
            try:
                validate_sha256(receipt[field], field)
            except Exception as exc:
                raise ReceiptVerificationError(str(exc)) from exc
    if not isinstance(receipt.get("idempotency_key"), str) or not receipt["idempotency_key"]:
        raise ReceiptVerificationError("idempotency_key must be non-empty")
    if not isinstance(receipt.get("signing_key_id"), str) or not receipt["signing_key_id"]:
        raise ReceiptVerificationError("signing_key_id must be non-empty")
    if not isinstance(receipt.get("receipt_id"), str) or not receipt["receipt_id"]:
        raise ReceiptVerificationError("receipt_id must be non-empty")
    if "infrastructure_incident_id" in receipt and (
        not isinstance(receipt["infrastructure_incident_id"], str) or not receipt["infrastructure_incident_id"]
    ):
        raise ReceiptVerificationError("infrastructure_incident_id must be a non-empty opaque ID")
    diagnostic = receipt.get("diagnostic_enum")
    incident = receipt.get("infrastructure_incident_id")
    root = receipt.get("failure_family_root")
    if diagnostic == "INTERNAL_ERROR":
        if not isinstance(incident, str) or not incident:
            raise ReceiptVerificationError("INTERNAL_ERROR receipts require infrastructure_incident_id")
        if not isinstance(root, str) or not root:
            raise ReceiptVerificationError("INTERNAL_ERROR receipts require an incident-bound failure_family_root")
        canonical_fields = ("task_family", "normalized_public_locus", "public_rule_id")
        missing_canonical = [field for field in canonical_fields if not isinstance(receipt.get(field), str) or not receipt[field]]
        if missing_canonical:
            raise ReceiptVerificationError(
                "INTERNAL_ERROR receipts require canonical fields: {}".format(", ".join(missing_canonical))
            )
        expected_root = failure_family_root(
            receipt["task_family"],
            diagnostic,
            receipt["normalized_public_locus"],
            receipt["public_rule_id"],
            infrastructure_incident_id=incident,
        )
        if root != expected_root:
            raise ReceiptVerificationError("failure_family_root is not the exact incident-bound canonical root")
    elif incident is not None or root is not None:
        raise ReceiptVerificationError("infrastructure incident and failure root are reserved for INTERNAL_ERROR")


class ReceiptSigner:
    """Create and verify campaign-scoped Ed25519 receipts."""

    def __init__(self, private_key: Any, *, key_id: Optional[str] = None) -> None:
        Ed25519PrivateKey, _Ed25519PublicKey, _InvalidSignature, _serialization = _crypto_imports()
        if isinstance(private_key, Ed25519PrivateKey):
            self._private_key = private_key
        elif isinstance(private_key, str):
            private_key = private_key.encode("utf-8")
            self._private_key = Ed25519PrivateKey.from_private_bytes(private_key)
        elif isinstance(private_key, bytes):
            try:
                self._private_key = Ed25519PrivateKey.from_private_bytes(private_key)
            except ValueError as exc:
                raise ReceiptVerificationError("Ed25519 private key must be 32 raw bytes") from exc
        else:
            raise ReceiptVerificationError("unsupported Ed25519 private-key representation")
        self.key_id = key_id or key_id_for_public_key(self.public_key)
        if self.key_id != key_id_for_public_key(self.public_key) and key_id is not None:
            raise ReceiptVerificationError("key_id does not bind to the supplied public key")

    @classmethod
    def generate(cls, *, key_id: Optional[str] = None) -> "ReceiptSigner":
        Ed25519PrivateKey, _Ed25519PublicKey, _InvalidSignature, _serialization = _crypto_imports()
        return cls(Ed25519PrivateKey.generate(), key_id=key_id)

    @property
    def public_key(self) -> Any:
        return self._private_key.public_key()

    @property
    def public_key_raw(self) -> bytes:
        return public_key_bytes(self.public_key)

    @property
    def private_key_raw(self) -> bytes:
        _Ed25519PrivateKey, _Ed25519PublicKey, _InvalidSignature, serialization = _crypto_imports()
        return self._private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )

    @property
    def public_key_pem(self) -> bytes:
        _Ed25519PrivateKey, _Ed25519PublicKey, _InvalidSignature, serialization = _crypto_imports()
        return self.public_key.public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )

    def sign_bytes(self, payload: bytes) -> str:
        return _b64_encode(self._private_key.sign(payload))

    def sign_receipt(
        self,
        fields: Mapping[str, Any],
        *,
        sequence: int,
        previous_receipt_hash: str = GENESIS_HASH,
        idempotency_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Sign one private receipt after assigning its chain position.

        ``receipt_id`` is derived from the complete unsigned receipt fields,
        including sequence and idempotency key.  A caller-provided ID must
        therefore equal that derivation; it cannot be used as mutable metadata.
        """

        payload = dict(fields)
        payload.setdefault("schema_version", RECEIPT_SCHEMA_VERSION)
        payload["sequence"] = sequence
        payload["previous_receipt_hash"] = previous_receipt_hash
        payload["idempotency_key"] = idempotency_key or payload.get("idempotency_key")
        payload["signing_key_id"] = self.key_id
        payload.pop("signature", None)
        supplied_id = payload.pop("receipt_id", None)
        derived_id = content_id("rcpt", payload)
        if supplied_id is not None and supplied_id != derived_id:
            raise ReceiptVerificationError("receipt_id is not the content address of the unsigned receipt")
        payload["receipt_id"] = derived_id
        _validate_common_receipt({**payload, "signature": "placeholder"})
        signature = self.sign_bytes(canonical_bytes(payload))
        receipt = {**payload, "signature": signature}
        verify_receipt(receipt, self.public_key, expected_key_id=self.key_id)
        return receipt

    def sign_public_receipt(
        self,
        fields: Mapping[str, Any],
        *,
        public_sequence: int,
        previous_public_receipt_digest: str = GENESIS_HASH,
    ) -> Dict[str, Any]:
        """Create a separately signed closed public receipt envelope."""

        from .public import build_public_receipt_envelope

        payload = dict(fields)
        payload["schema_version"] = PUBLIC_RECEIPT_SCHEMA_VERSION
        payload["public_sequence"] = public_sequence
        payload["previous_public_receipt_digest"] = previous_public_receipt_digest
        payload["signing_key_id"] = self.key_id
        payload.pop("signature", None)
        return build_public_receipt_envelope(payload, self)


def verify_receipt(
    receipt: Mapping[str, Any],
    public_key: Any,
    *,
    expected_key_id: Optional[str] = None,
    expected_sequence: Optional[int] = None,
    expected_previous_hash: Optional[str] = None,
) -> str:
    """Verify schema, Ed25519 signature, and optional chain position.

    Returns the complete receipt hash on success.
    """

    _validate_common_receipt(receipt)
    key = load_public_key(public_key)
    raw_public = public_key_bytes(key)
    computed_key_id = key_id_for_public_key(raw_public)
    if receipt["signing_key_id"] != computed_key_id:
        raise ReceiptVerificationError("receipt signing_key_id does not match the public key")
    if expected_key_id is not None and receipt["signing_key_id"] != expected_key_id:
        raise ReceiptVerificationError("receipt was signed by an unexpected key")
    if expected_sequence is not None and receipt["sequence"] != expected_sequence:
        raise ReceiptVerificationError("receipt sequence is not contiguous")
    if expected_previous_hash is not None and receipt["previous_receipt_hash"] != expected_previous_hash:
        raise ReceiptVerificationError("receipt previous hash does not match the journal head")
    unsigned = _signable(receipt)
    expected_id = content_id("rcpt", {key: value for key, value in unsigned.items() if key != "receipt_id"})
    if receipt["receipt_id"] != expected_id:
        raise ReceiptVerificationError("receipt_id does not match its signed content")
    signature = _b64_decode(receipt["signature"])
    _Ed25519PrivateKey, _Ed25519PublicKey, InvalidSignature, _serialization = _crypto_imports()
    try:
        key.verify(signature, canonical_bytes(unsigned))
    except InvalidSignature as exc:
        raise ReceiptVerificationError("invalid Ed25519 receipt signature") from exc
    return receipt_hash(receipt)


class ReceiptJournal:
    """Append-only evaluator receipt journal with idempotent delivery."""

    def __init__(self, path: Union[str, Path], public_key: Any) -> None:
        self.path = Path(path)
        self.public_key = load_public_key(public_key)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def receipts(self) -> list[Dict[str, Any]]:
        if not self.path.exists():
            return []
        from .canonical import parse_canonical_jsonl

        try:
            raw = parse_canonical_jsonl(self.path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise ReceiptVerificationError(f"receipt journal is not canonical JSONL: {exc}") from exc
        result: list[Dict[str, Any]] = []
        previous = GENESIS_HASH
        for index, record in enumerate(raw, start=1):
            if not isinstance(record, dict):
                raise ReceiptVerificationError(f"receipt journal line {index} is not an object")
            receipt_hash_value = verify_receipt(
                record,
                self.public_key,
                expected_sequence=index,
                expected_previous_hash=previous,
            )
            result.append(dict(record))
            previous = receipt_hash_value
        return result

    def append(self, receipt: Mapping[str, Any]) -> Dict[str, Any]:
        existing = self.receipts()
        by_key = {record["idempotency_key"]: record for record in existing}
        by_id = {record["receipt_id"]: record for record in existing}
        candidate = dict(receipt)
        if candidate.get("idempotency_key") in by_key:
            if canonical_json(by_key[candidate["idempotency_key"]]) != canonical_json(candidate):
                raise ReceiptConflictError("receipt idempotency key was delivered with conflicting content")
            return dict(by_key[candidate["idempotency_key"]])
        if candidate.get("receipt_id") in by_id:
            if canonical_json(by_id[candidate["receipt_id"]]) != canonical_json(candidate):
                raise ReceiptConflictError("receipt ID was delivered with conflicting content")
            return dict(by_id[candidate["receipt_id"]])
        previous = receipt_hash(existing[-1]) if existing else GENESIS_HASH
        verify_receipt(
            candidate,
            self.public_key,
            expected_sequence=len(existing) + 1,
            expected_previous_hash=previous,
        )
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(canonical_json(candidate) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        return candidate

    def verify(self) -> Dict[str, Any]:
        records = self.receipts()
        head = receipt_hash(records[-1]) if records else GENESIS_HASH
        # ``receipts()`` verifies every signature, sequence, and previous hash
        # before this value is returned.  Expose that fact as derived evidence
        # so callers do not synthesize a chain-valid flag.
        return {
            "count": len(records),
            "head": head,
            "receipt_ids": [r["receipt_id"] for r in records],
            "chain_valid": True,
        }

    def reconcile(self, ledger: Any) -> int:
        """Ingest all journal records into a ledger exactly once."""

        count = 0
        for receipt in self.receipts():
            before = ledger.receipt_by_id(receipt["receipt_id"])
            ledger.ingest_receipt(receipt, self.public_key)
            if before is None:
                count += 1
        return count


__all__ = [
    "GENESIS_HASH",
    "PUBLIC_RECEIPT_SCHEMA_VERSION",
    "RECEIPT_SCHEMA_VERSION",
    "ReceiptJournal",
    "ReceiptSigner",
    "key_id_for_public_key",
    "load_public_key",
    "public_key_bytes",
    "receipt_hash",
    "verify_receipt",
]
