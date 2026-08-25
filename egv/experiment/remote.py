"""Content-bound evaluator service contract for held-out campaign results.

This module deliberately does not implement a network transport.  A caller may
carry the closed command and response objects over SSH, a queue, or another
operator-controlled channel.  Admission depends only on their signed content,
not on the transport.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Dict, Mapping, Optional, Union

from ..campaign.state import _exclusive_path_lock
from ..canonical import canonical_bytes, canonical_json, canonical_value, content_id, digest_for, validate_sha256
from ..receipts import key_id_for_public_key, load_public_key, public_key_bytes
from ..variation.arms import arm_policy
from .heldout import (
    FrozenHeldoutProtocol,
    HeldoutProtocolError,
    build_signed_reconciliation,
    build_signed_result_envelope,
    validate_coordinate_operation_state,
    verify_signed_reconciliation,
    verify_signed_result_envelope,
)


SERVICE_MANIFEST_SCHEMA = "egv-heldout-verifier-service-v1"
COMMAND_SCHEMA = "egv-heldout-verifier-command-v1"
ACK_SCHEMA = "egv-heldout-verifier-ack-v1"
STATE_SCHEMA = "egv-heldout-verifier-state-v1"
ACTIONS = ("BEGIN", "VERIFY", "RECONCILE")


def _closed(value: Mapping[str, Any], keys: tuple[str, ...], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != set(keys):
        missing = sorted(set(keys) - set(value)) if isinstance(value, Mapping) else list(keys)
        extra = sorted(set(value) - set(keys)) if isinstance(value, Mapping) else []
        raise HeldoutProtocolError(f"{label} has a non-closed schema (missing={missing}, extra={extra})")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(canonical_json(value) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


@dataclass(frozen=True)
class HeldoutVerifierServiceManifest:
    """Closed evaluator identity and immutable campaign bindings."""

    campaign_id: str
    protocol_digest: str
    evaluator_digest: str
    base_model_digest: str
    adapter_digest: str
    evaluator_public_key_hex: str
    evaluator_key_id: str

    KEYS = (
        "schema_version", "campaign_id", "protocol_digest", "evaluator_digest",
        "base_model_digest", "adapter_digest", "evaluator_public_key_hex",
        "evaluator_key_id", "actions", "manifest_digest",
    )

    @classmethod
    def from_protocol(cls, protocol: FrozenHeldoutProtocol) -> "HeldoutVerifierServiceManifest":
        protocol.validate_current()
        return cls(
            campaign_id=protocol.campaign_id,
            protocol_digest=protocol.digest,
            evaluator_digest=protocol.bindings["evaluator_digest"],
            base_model_digest=protocol.bindings["base_model_digest"],
            adapter_digest=protocol.bindings["adapter_digest"],
            evaluator_public_key_hex=protocol.evaluator_public_key_hex,
            evaluator_key_id=protocol.evaluator_key_id,
        )

    @property
    def digest(self) -> str:
        return digest_for(self._unsigned())

    def _unsigned(self) -> Dict[str, Any]:
        return {
            "schema_version": SERVICE_MANIFEST_SCHEMA,
            "campaign_id": self.campaign_id,
            "protocol_digest": validate_sha256(self.protocol_digest, "protocol_digest"),
            "evaluator_digest": validate_sha256(self.evaluator_digest, "evaluator_digest"),
            "base_model_digest": validate_sha256(self.base_model_digest, "base_model_digest"),
            "adapter_digest": validate_sha256(self.adapter_digest, "adapter_digest"),
            "evaluator_public_key_hex": self.evaluator_public_key_hex,
            "evaluator_key_id": self.evaluator_key_id,
            "actions": list(ACTIONS),
        }

    def to_dict(self) -> Dict[str, Any]:
        value = self._unsigned()
        try:
            raw_key = bytes.fromhex(self.evaluator_public_key_hex)
        except (TypeError, ValueError) as exc:
            raise HeldoutProtocolError("service manifest evaluator key is not canonical hex") from exc
        if public_key_bytes(raw_key).hex() != self.evaluator_public_key_hex:
            raise HeldoutProtocolError("service manifest evaluator key is not canonical Ed25519")
        if key_id_for_public_key(raw_key) != self.evaluator_key_id:
            raise HeldoutProtocolError("service manifest evaluator key ID mismatch")
        return {**value, "manifest_digest": digest_for(value)}

    @classmethod
    def load(cls, value: Mapping[str, Any], protocol: FrozenHeldoutProtocol) -> "HeldoutVerifierServiceManifest":
        _closed(value, cls.KEYS, "heldout verifier service manifest")
        expected = cls.from_protocol(protocol)
        if canonical_value(dict(value)) != expected.to_dict():
            raise HeldoutProtocolError("heldout verifier service manifest differs from frozen protocol")
        return expected


COMMAND_KEYS = (
    "schema_version", "request_id", "action", "service_manifest_digest", "campaign_id",
    "protocol_digest", "coordinate_id", "coordinate", "coordinate_digest", "idempotency_key",
    "operation_state", "operation_state_digest", "base_model_digest", "adapter_digest",
    "evaluator_public_key_hex", "evaluator_key_id",
    "receipt_collection_root", "ledger_head_digest", "observation", "observation_digest",
)
def build_heldout_verifier_command(
    protocol: FrozenHeldoutProtocol,
    manifest: HeldoutVerifierServiceManifest,
    *,
    action: str,
    coordinate_id: str,
    operation_state: Mapping[str, Any],
    receipt_collection_root: str,
    ledger_head_digest: str,
    observation: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build one closed command suitable for an external command wrapper."""

    if action not in ACTIONS:
        raise HeldoutProtocolError("heldout verifier action is unsupported")
    HeldoutVerifierServiceManifest.load(manifest.to_dict(), protocol)
    coordinate = protocol.coordinate(coordinate_id)
    expected_idempotency = content_id("idem", {"protocol_digest": protocol.digest, "coordinate_id": coordinate_id})
    if operation_state.get("coordinate_id") != coordinate_id or operation_state.get("idempotency_key") != expected_idempotency:
        raise HeldoutProtocolError("verifier command operation state is not bound to its coordinate")
    state_digest = operation_state.get("state_digest")
    validate_sha256(state_digest, "operation_state_digest")
    exact_coordinate = coordinate.to_dict(protocol.digest, protocol.campaign_id)
    policy = arm_policy(coordinate.treatment) if coordinate.phase == "MAIN_ABLATION" else None
    active_adapter = protocol.bindings["adapter_digest"] if policy is None or policy.requires_adapter else None
    if action == "VERIFY" and observation is None:
        raise HeldoutProtocolError("VERIFY requires an opaque runner observation")
    if action != "VERIFY" and observation is not None:
        raise HeldoutProtocolError("only VERIFY may carry a runner observation")
    payload: Dict[str, Any] = {
        "schema_version": COMMAND_SCHEMA,
        "action": action,
        "service_manifest_digest": manifest.digest,
        "campaign_id": protocol.campaign_id,
        "protocol_digest": protocol.digest,
        "coordinate_id": coordinate_id,
        "coordinate": exact_coordinate,
        "coordinate_digest": digest_for(exact_coordinate),
        "idempotency_key": expected_idempotency,
        "operation_state": canonical_value(dict(operation_state)),
        "operation_state_digest": state_digest,
        "base_model_digest": protocol.bindings["base_model_digest"],
        "adapter_digest": active_adapter,
        "evaluator_public_key_hex": protocol.evaluator_public_key_hex,
        "evaluator_key_id": protocol.evaluator_key_id,
        "receipt_collection_root": validate_sha256(receipt_collection_root, "receipt_collection_root"),
        "ledger_head_digest": validate_sha256(ledger_head_digest, "ledger_head_digest"),
        "observation": canonical_value(dict(observation)) if observation is not None else None,
        "observation_digest": digest_for(observation) if observation is not None else None,
    }
    payload["request_id"] = content_id("heldout-command", payload)
    return validate_heldout_verifier_command(protocol, manifest, payload)


def validate_heldout_verifier_command(
    protocol: FrozenHeldoutProtocol,
    manifest: HeldoutVerifierServiceManifest,
    raw: Mapping[str, Any],
) -> Dict[str, Any]:
    protocol.validate_current()
    _closed(raw, COMMAND_KEYS, "heldout verifier command")
    if raw["schema_version"] != COMMAND_SCHEMA or raw["action"] not in ACTIONS:
        raise HeldoutProtocolError("heldout verifier command schema or action is unsupported")
    coordinate = protocol.coordinate(raw["coordinate_id"])
    exact_coordinate = coordinate.to_dict(protocol.digest, protocol.campaign_id)
    policy = arm_policy(coordinate.treatment) if coordinate.phase == "MAIN_ABLATION" else None
    expected_adapter = protocol.bindings["adapter_digest"] if policy is None or policy.requires_adapter else None
    expected = {
        "service_manifest_digest": manifest.digest,
        "campaign_id": protocol.campaign_id,
        "protocol_digest": protocol.digest,
        "coordinate": exact_coordinate,
        "coordinate_digest": digest_for(exact_coordinate),
        "idempotency_key": content_id("idem", {"protocol_digest": protocol.digest, "coordinate_id": coordinate.coordinate_id}),
        "base_model_digest": protocol.bindings["base_model_digest"],
        "adapter_digest": expected_adapter,
        "evaluator_public_key_hex": protocol.evaluator_public_key_hex,
        "evaluator_key_id": protocol.evaluator_key_id,
    }
    for field, value in expected.items():
        if raw[field] != value:
            raise HeldoutProtocolError(f"heldout verifier command {field} binding mismatch")
    state = validate_coordinate_operation_state(
        protocol, coordinate.coordinate_id, raw["operation_state"]
    )
    allowed_states = {
        "BEGIN": {"DISPATCHING"},
        "VERIFY": {"DISPATCHING"},
        "RECONCILE": {"DISPATCHING", "COMPLETED", "QUARANTINED"},
    }
    if state["state"] not in allowed_states[raw["action"]]:
        raise HeldoutProtocolError("heldout verifier action and operation state disagree")
    if raw["operation_state_digest"] != state["state_digest"]:
        raise HeldoutProtocolError("heldout verifier command operation state digest mismatch")
    for field in ("receipt_collection_root", "ledger_head_digest"):
        validate_sha256(raw[field], field)
    if raw["action"] == "VERIFY":
        if not isinstance(raw["observation"], Mapping) or raw["observation_digest"] != digest_for(raw["observation"]):
            raise HeldoutProtocolError("VERIFY observation binding mismatch")
    elif raw["observation"] is not None or raw["observation_digest"] is not None:
        raise HeldoutProtocolError("non-VERIFY command carries an observation")
    unsigned = dict(raw)
    request_id = unsigned.pop("request_id")
    if request_id != content_id("heldout-command", unsigned):
        raise HeldoutProtocolError("heldout verifier command request ID is not content-derived")
    return canonical_value(dict(raw))


ACK_KEYS = (
    "schema_version", "response_id", "action", "request_id", "service_manifest_digest",
    "campaign_id", "protocol_digest", "coordinate_id", "idempotency_key", "operation_state_digest",
    "base_model_digest", "adapter_digest", "evaluator_public_key_hex", "evaluator_key_id",
    "receipt_collection_root", "ledger_head_digest",
    "result_envelope_digest", "reconciliation_envelope_digest", "signing_key_id", "signature",
)


def _signed_ack(
    protocol: FrozenHeldoutProtocol,
    manifest: HeldoutVerifierServiceManifest,
    command: Mapping[str, Any],
    signer: Any,
    *,
    result_envelope: Optional[Mapping[str, Any]] = None,
    reconciliation_envelope: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "schema_version": ACK_SCHEMA,
        "action": command["action"],
        "request_id": command["request_id"],
        "service_manifest_digest": manifest.digest,
        "campaign_id": protocol.campaign_id,
        "protocol_digest": protocol.digest,
        "coordinate_id": command["coordinate_id"],
        "idempotency_key": command["idempotency_key"],
        "operation_state_digest": command["operation_state_digest"],
        "base_model_digest": command["base_model_digest"],
        "adapter_digest": command["adapter_digest"],
        "evaluator_public_key_hex": command["evaluator_public_key_hex"],
        "evaluator_key_id": command["evaluator_key_id"],
        "receipt_collection_root": command["receipt_collection_root"],
        "ledger_head_digest": command["ledger_head_digest"],
        "result_envelope_digest": digest_for(result_envelope) if result_envelope is not None else None,
        "reconciliation_envelope_digest": digest_for(reconciliation_envelope) if reconciliation_envelope is not None else None,
        "signing_key_id": protocol.evaluator_key_id,
    }
    payload["response_id"] = content_id("heldout-response", payload)
    payload["signature"] = signer.sign_bytes(canonical_bytes(payload))
    return payload


def _verify_ack(
    protocol: FrozenHeldoutProtocol,
    manifest: HeldoutVerifierServiceManifest,
    command: Mapping[str, Any],
    ack: Mapping[str, Any],
    *,
    result_envelope: Optional[Mapping[str, Any]],
    reconciliation_envelope: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    _closed(ack, ACK_KEYS, "heldout verifier acknowledgement")
    expected = {
        "schema_version": ACK_SCHEMA, "action": command["action"], "request_id": command["request_id"],
        "service_manifest_digest": manifest.digest, "campaign_id": protocol.campaign_id,
        "protocol_digest": protocol.digest, "coordinate_id": command["coordinate_id"],
        "idempotency_key": command["idempotency_key"], "operation_state_digest": command["operation_state_digest"],
        "base_model_digest": command["base_model_digest"], "adapter_digest": command["adapter_digest"],
        "evaluator_public_key_hex": command["evaluator_public_key_hex"],
        "evaluator_key_id": command["evaluator_key_id"],
        "receipt_collection_root": command["receipt_collection_root"], "ledger_head_digest": command["ledger_head_digest"],
        "result_envelope_digest": digest_for(result_envelope) if result_envelope is not None else None,
        "reconciliation_envelope_digest": digest_for(reconciliation_envelope) if reconciliation_envelope is not None else None,
        "signing_key_id": protocol.evaluator_key_id,
    }
    for field, value in expected.items():
        if ack[field] != value:
            raise HeldoutProtocolError(f"heldout verifier acknowledgement {field} mismatch")
    unsigned = dict(ack)
    signature = unsigned.pop("signature")
    response_id = unsigned.pop("response_id")
    if response_id != content_id("heldout-response", unsigned):
        raise HeldoutProtocolError("heldout verifier response ID is not content-derived")
    signature_padding = "=" * (-len(signature) % 4) if isinstance(signature, str) else ""
    try:
        signature_bytes = base64.b64decode(
            (signature + signature_padding).encode("ascii"), altchars=b"-_", validate=True
        )
        canonical_signature = base64.urlsafe_b64encode(signature_bytes).decode("ascii").rstrip("=")
        if len(signature_bytes) != 64 or canonical_signature != signature:
            raise ValueError("non-canonical signature")
        load_public_key(bytes.fromhex(protocol.evaluator_public_key_hex)).verify(
            signature_bytes,
            canonical_bytes({**unsigned, "response_id": response_id}),
        )
    except Exception as exc:
        raise HeldoutProtocolError("invalid heldout verifier acknowledgement signature") from exc
    return canonical_value(dict(ack))


@dataclass(frozen=True)
class EvaluatorVerification:
    """Facts independently derived by evaluator-owned code from opaque output."""

    result: Mapping[str, Any]
    receipt_collection_root: str
    ledger_head_digest: str


ObservationVerifier = Callable[[Mapping[str, Any], Mapping[str, Any]], EvaluatorVerification]
ReconciliationProbe = Callable[[Mapping[str, Any], Mapping[str, Any]], str]


def _state_path(root: Union[str, Path], coordinate_id: str) -> Path:
    return Path(root) / (coordinate_id + ".json")


def _load_state(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HeldoutProtocolError("evaluator state is unreadable") from exc
    keys = (
        "schema_version", "coordinate_id", "idempotency_key", "begin_request_digest",
        "begin_operation_state_digest", "effect_state", "result_envelope", "responses", "state_digest",
    )
    _closed(value, keys, "evaluator state")
    unsigned = dict(value)
    supplied = unsigned.pop("state_digest")
    if supplied != digest_for(unsigned):
        raise HeldoutProtocolError("evaluator state digest mismatch")
    return value


def _run_heldout_verifier_locked(
    protocol: FrozenHeldoutProtocol,
    manifest: HeldoutVerifierServiceManifest,
    command: Mapping[str, Any],
    *,
    signer: Any,
    state_root: Union[str, Path],
    observation_verifier: ObservationVerifier,
    reconciliation_probe: Optional[ReconciliationProbe] = None,
) -> Dict[str, Any]:
    """Process exactly one BEGIN, VERIFY, or RECONCILE evaluator command."""

    command = validate_heldout_verifier_command(protocol, manifest, command)
    if getattr(signer, "key_id", None) != protocol.evaluator_key_id:
        raise HeldoutProtocolError("heldout verifier signer differs from frozen evaluator")
    path = _state_path(state_root, command["coordinate_id"])
    state = _load_state(path)
    if state is not None:
        if state["coordinate_id"] != command["coordinate_id"] or state["idempotency_key"] != command["idempotency_key"]:
            raise HeldoutProtocolError("evaluator state coordinate binding mismatch")
        cached = state["responses"].get(command["request_id"])
        if cached is not None:
            return canonical_value(cached)

    result_envelope = None
    reconciliation_envelope = None
    action = command["action"]
    if action == "BEGIN":
        if state is not None:
            raise HeldoutProtocolError("conflicting BEGIN for an existing evaluator operation")
        state = {
            "schema_version": STATE_SCHEMA,
            "coordinate_id": command["coordinate_id"],
            "idempotency_key": command["idempotency_key"],
            "begin_request_digest": digest_for(command),
            "begin_operation_state_digest": command["operation_state_digest"],
            "effect_state": "UNKNOWN",
            "result_envelope": None,
            "responses": {},
        }
    elif state is None:
        raise HeldoutProtocolError("VERIFY or RECONCILE requires a durable evaluator BEGIN")
    elif state["begin_operation_state_digest"] != command["operation_state_digest"]:
        raise HeldoutProtocolError("evaluator command substituted the begun operation state")
    elif action == "VERIFY":
        if state["result_envelope"] is not None:
            raise HeldoutProtocolError("coordinate already has an evaluator-signed result")
        verified = observation_verifier(command["coordinate"], command["observation"])
        if not isinstance(verified, EvaluatorVerification):
            raise HeldoutProtocolError("evaluator observation verifier returned an invalid contract")
        # Only this evaluator-owned boundary can convert derived facts into an
        # admissible signed result; runner-supplied booleans are never signed directly.
        result_envelope = build_signed_result_envelope(
            protocol,
            verified.result,
            signer,
            receipt_collection_root=verified.receipt_collection_root,
            ledger_head_digest=verified.ledger_head_digest,
        )
        if result_envelope["coordinate_id"] != command["coordinate_id"]:
            raise HeldoutProtocolError("evaluator-derived result belongs to another coordinate")
        if result_envelope["receipt_collection_root"] != command["receipt_collection_root"] or result_envelope["ledger_head_digest"] != command["ledger_head_digest"]:
            raise HeldoutProtocolError("evaluator-derived receipt or ledger roots differ from command bindings")
        state["effect_state"] = "COMPLETED"
        state["result_envelope"] = result_envelope
    else:
        result_envelope = state["result_envelope"]
        if result_envelope is not None:
            decision = "COMPLETED"
        else:
            probed = reconciliation_probe(command["coordinate"], command["operation_state"]) if reconciliation_probe else "UNKNOWN"
            if probed not in {"NOT_EXECUTED", "COMPLETED", "UNKNOWN"}:
                raise HeldoutProtocolError("evaluator reconciliation probe returned an invalid decision")
            # A COMPLETED probe without the exact cached signed result is partial
            # external evidence and is therefore unknowable, never retryable.
            decision = "NOT_EXECUTED" if probed == "NOT_EXECUTED" else "UNKNOWN"
            state["effect_state"] = decision
        reconciliation_envelope = build_signed_reconciliation(
            protocol,
            command["operation_state"],
            signer,
            decision=decision,
            result_envelope=result_envelope,
            receipt_collection_root=command["receipt_collection_root"],
            ledger_head_digest=command["ledger_head_digest"],
        )

    ack = _signed_ack(
        protocol, manifest, command, signer,
        result_envelope=result_envelope,
        reconciliation_envelope=reconciliation_envelope,
    )
    response = {
        "acknowledgement": ack,
        "result_envelope": result_envelope,
        "reconciliation_envelope": reconciliation_envelope,
    }
    state["responses"][command["request_id"]] = canonical_value(response)
    unsigned_state = dict(state)
    unsigned_state.pop("state_digest", None)
    _atomic_json(path, {**unsigned_state, "state_digest": digest_for(unsigned_state)})
    return canonical_value(response)


def run_heldout_verifier_once(
    protocol: FrozenHeldoutProtocol,
    manifest: HeldoutVerifierServiceManifest,
    command: Mapping[str, Any],
    *,
    signer: Any,
    state_root: Union[str, Path],
    observation_verifier: ObservationVerifier,
    reconciliation_probe: Optional[ReconciliationProbe] = None,
) -> Dict[str, Any]:
    """Serialize one coordinate's complete evaluator state transition.

    Atomic replacement prevents torn files, while this cross-process lock
    prevents two distinct VERIFY requests from both being signed from the
    same BEGIN state.  The command is validated before deriving the lock name
    and then revalidated inside the locked transaction.
    """

    validated = validate_heldout_verifier_command(protocol, manifest, command)
    root = Path(state_root)
    lock_path = root / (validated["coordinate_id"] + ".lock")
    with _exclusive_path_lock(lock_path):
        return _run_heldout_verifier_locked(
            protocol,
            manifest,
            validated,
            signer=signer,
            state_root=root,
            observation_verifier=observation_verifier,
            reconciliation_probe=reconciliation_probe,
        )


Executor = Callable[[Mapping[str, Any]], Mapping[str, Any]]


class RemoteHeldoutResultVerifier:
    """Trainer-side verifier for a content-carried evaluator response."""

    def __init__(self, protocol: FrozenHeldoutProtocol, manifest: Mapping[str, Any], executor: Executor) -> None:
        self.protocol = protocol
        self.manifest = HeldoutVerifierServiceManifest.load(manifest, protocol)
        self.executor = executor

    def execute(self, command: Mapping[str, Any]) -> Dict[str, Any]:
        command = validate_heldout_verifier_command(self.protocol, self.manifest, command)
        raw = self.executor(command)
        _closed(raw, ("acknowledgement", "result_envelope", "reconciliation_envelope"), "heldout verifier response")
        result = raw["result_envelope"]
        reconciliation = raw["reconciliation_envelope"]
        _verify_ack(self.protocol, self.manifest, command, raw["acknowledgement"], result_envelope=result, reconciliation_envelope=reconciliation)
        if command["action"] == "VERIFY":
            if reconciliation is not None or result is None:
                raise HeldoutProtocolError("VERIFY response must contain only a signed result")
            verify_signed_result_envelope(self.protocol, result)
        elif command["action"] == "BEGIN":
            if result is not None or reconciliation is not None:
                raise HeldoutProtocolError("BEGIN response cannot contain a terminal envelope")
        else:
            if reconciliation is None:
                raise HeldoutProtocolError("RECONCILE response requires a signed reconciliation")
            verify_signed_reconciliation(
                self.protocol,
                command["operation_state"],
                reconciliation,
                result,
            )
        return canonical_value(dict(raw))


class RemoteHeldoutReconciler(RemoteHeldoutResultVerifier):
    """Trainer-side verification of signed evaluator reconciliation."""

    def execute(self, command: Mapping[str, Any]) -> Dict[str, Any]:
        raw = super().execute(command)
        if command["action"] != "RECONCILE" or raw["reconciliation_envelope"] is None:
            raise HeldoutProtocolError("reconciler requires a RECONCILE command and signed response")
        verify_signed_reconciliation(
            self.protocol,
            command["operation_state"],
            raw["reconciliation_envelope"],
            raw["result_envelope"],
        )
        return raw


__all__ = [
    "EvaluatorVerification",
    "HeldoutVerifierServiceManifest",
    "RemoteHeldoutReconciler",
    "RemoteHeldoutResultVerifier",
    "build_heldout_verifier_command",
    "run_heldout_verifier_once",
    "validate_heldout_verifier_command",
]
