"""Externally signed Gate-A freeze records for the learnable-v2 corpus.

This module freezes authorization inputs only.  It never loads evaluator
secrets, creates a signing authority, executes candidate code, or writes a
private key.  The caller supplies the exact manifests, independent blind
solver source, controller public key, and an external :class:`ReceiptSigner`.
"""

from __future__ import annotations

import base64
from typing import Any, Dict, Mapping

from ..canonical import canonical_bytes, canonical_value, digest_bytes, digest_for, validate_sha256
from ..receipts import ReceiptSigner, key_id_for_public_key, load_public_key, public_key_bytes
from .learnable import validate_learnable_manifest, validate_learnable_private_manifest


CORPUS_FREEZE_VERSION = "egv-learnable-corpus-freeze-v1"
CORPUS_VALIDATION_SCOPE = "ORACLE_AND_BLIND_SOLVER_ONLY_NO_MODEL_REQUESTS"
BLIND_SOLVER_MANIFEST_VERSION = "egv-blind-solver-source-manifest-v1"

_MAX_SOLVER_SOURCE_BYTES = 4 * 1024 * 1024
_FREEZE_FIELDS = frozenset(
    {
        "schema_version",
        "scope",
        "task_bearing_model_requests_permitted",
        "public_manifest_digest",
        "private_manifest_digest",
        "runtime_manifest_digest",
        "blind_solver_manifest_digest",
        "blind_solver_source_digest",
        "blind_solver_version",
        "evaluator_revision",
        "controller_public_key",
        "controller_key_id",
        "freeze_signer_key_id",
        "bundle_digest",
        "signature",
    }
)
_BLIND_SOLVER_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "solver_version",
        "source_digest",
        "source_size_bytes",
        "manifest_digest",
    }
)
_RUNTIME_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "backend",
        "image_ref",
        "pinned_image_id",
        "candidate_contract",
        "network",
        "read_only_root",
        "cap_drop",
        "no_new_privileges",
        "user",
        "pids_limit",
        "memory_limit",
        "tmpfs_size",
        "timeout_seconds",
        "output_limit",
        "manifest_digest",
    }
)


def _closed(record: Any, fields: frozenset[str], label: str) -> Mapping[str, Any]:
    if not isinstance(record, Mapping) or set(record) != set(fields):
        raise ValueError("{} is not a closed record".format(label))
    return record


def _nonempty_text(value: Any, label: str, *, maximum: int = 256) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ValueError("{} must be bounded non-empty text".format(label))
    return value


def _digest_without(record: Mapping[str, Any], *fields: str) -> str:
    excluded = set(fields)
    return digest_for({key: value for key, value in record.items() if key not in excluded})


def _b64url_encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _b64url_decode(value: Any, *, size: int, label: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise ValueError("{} must be canonical base64url text".format(label))
    padded = value + "=" * (-len(value) % 4)
    try:
        decoded = base64.b64decode(padded.encode("ascii"), altchars=b"-_", validate=True)
    except (UnicodeError, ValueError) as exc:
        raise ValueError("{} must be canonical base64url text".format(label)) from exc
    if len(decoded) != size or _b64url_encode(decoded) != value:
        raise ValueError("{} has an invalid canonical byte length".format(label))
    return decoded


def _validate_public_key_only(value: Any, label: str) -> bytes:
    if isinstance(value, ReceiptSigner):
        raise ValueError("{} must be a public key, not a signer".format(label))
    try:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    except ImportError:  # pragma: no cover - the receipt dependency reports this first
        Ed25519PrivateKey = ()  # type: ignore[assignment,misc]
    if isinstance(value, Ed25519PrivateKey):
        raise ValueError("{} must not contain a private key".format(label))
    try:
        raw = public_key_bytes(value)
    except Exception as exc:
        raise ValueError("{} must be an Ed25519 public key".format(label)) from exc
    if len(raw) != 32:
        raise ValueError("{} must contain exactly 32 public bytes".format(label))
    return raw


def build_blind_solver_manifest(source: bytes, *, solver_version: str) -> Dict[str, Any]:
    """Build a closed manifest for an independently supplied solver module."""

    if not isinstance(source, bytes) or not source or len(source) > _MAX_SOLVER_SOURCE_BYTES:
        raise ValueError("blind solver source has an invalid bounded byte length")
    version = _nonempty_text(solver_version, "blind solver version")
    manifest: Dict[str, Any] = {
        "schema_version": BLIND_SOLVER_MANIFEST_VERSION,
        "solver_version": version,
        "source_digest": digest_bytes(source),
        "source_size_bytes": len(source),
    }
    manifest["manifest_digest"] = digest_for(manifest)
    return manifest


def validate_blind_solver_manifest(
    manifest: Mapping[str, Any],
    *,
    solver_source: bytes,
    solver_version: str,
) -> Dict[str, Any]:
    """Validate one exact independent-solver source/version binding."""

    record = _closed(manifest, _BLIND_SOLVER_MANIFEST_FIELDS, "blind solver manifest")
    expected = build_blind_solver_manifest(solver_source, solver_version=solver_version)
    if record != expected:
        raise ValueError("blind solver manifest differs from the supplied source and version")
    return dict(canonical_value(record))


def _validate_runtime_manifest(manifest: Mapping[str, Any]) -> Dict[str, Any]:
    record = _closed(manifest, _RUNTIME_MANIFEST_FIELDS, "Gate-A runtime manifest")
    if record.get("schema_version") != "egv-learnable-production-runtime-v2":
        raise ValueError("unsupported Gate-A runtime manifest")
    if (
        record.get("backend") != "docker-enforced-v1"
        or record.get("candidate_contract") != "pure-return-v1"
        or record.get("network") != "none"
        or record.get("read_only_root") is not True
        or record.get("cap_drop") != "ALL"
        or record.get("no_new_privileges") is not True
    ):
        raise ValueError("Gate-A runtime manifest weakens the production boundary")
    for field in ("image_ref", "pinned_image_id", "user", "memory_limit", "tmpfs_size"):
        _nonempty_text(record.get(field), "runtime {}".format(field))
    for field in ("pids_limit", "output_limit"):
        value = record.get(field)
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError("runtime {} must be a positive integer".format(field))
    timeout = record.get("timeout_seconds")
    if not isinstance(timeout, (int, float)) or isinstance(timeout, bool) or timeout <= 0:
        raise ValueError("runtime timeout_seconds must be positive")
    validate_sha256(record.get("manifest_digest"), "runtime manifest digest")
    if record["manifest_digest"] != _digest_without(record, "manifest_digest"):
        raise ValueError("Gate-A runtime manifest digest mismatch")
    return dict(canonical_value(record))


def _validated_manifests(
    *,
    public_manifest: Mapping[str, Any],
    private_manifest: Mapping[str, Any],
    runtime_manifest: Mapping[str, Any],
    blind_solver_manifest: Mapping[str, Any],
    blind_solver_source: bytes,
    blind_solver_version: str,
) -> tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    public = validate_learnable_manifest(public_manifest)
    private = validate_learnable_private_manifest(private_manifest)
    runtime = _validate_runtime_manifest(runtime_manifest)
    blind = validate_blind_solver_manifest(
        blind_solver_manifest,
        solver_source=blind_solver_source,
        solver_version=blind_solver_version,
    )
    if private["public_manifest_digest"] != public["manifest_digest"]:
        raise ValueError("private corpus manifest is not bound to the public corpus manifest")
    return public, private, runtime, blind


def _freeze_core(
    *,
    public_manifest_digest: str,
    private_manifest_digest: str,
    runtime_manifest_digest: str,
    blind_solver_manifest_digest: str,
    blind_solver_source_digest: str,
    blind_solver_version: str,
    evaluator_revision: str,
    controller_public_key: bytes,
    freeze_signer_key_id: str,
) -> Dict[str, Any]:
    return {
        "schema_version": CORPUS_FREEZE_VERSION,
        "scope": CORPUS_VALIDATION_SCOPE,
        "task_bearing_model_requests_permitted": False,
        "public_manifest_digest": public_manifest_digest,
        "private_manifest_digest": private_manifest_digest,
        "runtime_manifest_digest": runtime_manifest_digest,
        "blind_solver_manifest_digest": blind_solver_manifest_digest,
        "blind_solver_source_digest": blind_solver_source_digest,
        "blind_solver_version": blind_solver_version,
        "evaluator_revision": evaluator_revision,
        "controller_public_key": _b64url_encode(controller_public_key),
        "controller_key_id": key_id_for_public_key(controller_public_key),
        "freeze_signer_key_id": freeze_signer_key_id,
    }


def _signed_payload(core: Mapping[str, Any], bundle_digest: str) -> bytes:
    return canonical_bytes({**dict(core), "bundle_digest": bundle_digest, "signature": None})


def build_corpus_freeze_bundle(
    *,
    public_manifest: Mapping[str, Any],
    private_manifest: Mapping[str, Any],
    runtime_manifest: Mapping[str, Any],
    blind_solver_manifest: Mapping[str, Any],
    blind_solver_source: bytes,
    blind_solver_version: str,
    evaluator_revision: str,
    controller_public_key: Any,
    freeze_signer: ReceiptSigner,
) -> Dict[str, Any]:
    """Build one external-authority Gate-A freeze bundle in memory.

    Only public verification material is placed in the returned record.  The
    signer's private bytes and the controller's private bytes are never read or
    serialized by this function.
    """

    if not isinstance(freeze_signer, ReceiptSigner):
        raise ValueError("Gate-A freeze requires an externally supplied ReceiptSigner")
    public, private, runtime, blind = _validated_manifests(
        public_manifest=public_manifest,
        private_manifest=private_manifest,
        runtime_manifest=runtime_manifest,
        blind_solver_manifest=blind_solver_manifest,
        blind_solver_source=blind_solver_source,
        blind_solver_version=blind_solver_version,
    )
    revision = _nonempty_text(evaluator_revision, "evaluator revision")
    validate_sha256(revision, "evaluator revision")
    controller_key = _validate_public_key_only(controller_public_key, "controller public key")
    if freeze_signer.key_id != key_id_for_public_key(freeze_signer.public_key):
        raise ValueError("freeze signer key ID is not bound to its Ed25519 public key")
    core = _freeze_core(
        public_manifest_digest=public["manifest_digest"],
        private_manifest_digest=private["manifest_digest"],
        runtime_manifest_digest=runtime["manifest_digest"],
        blind_solver_manifest_digest=blind["manifest_digest"],
        blind_solver_source_digest=blind["source_digest"],
        blind_solver_version=blind["solver_version"],
        evaluator_revision=revision,
        controller_public_key=controller_key,
        freeze_signer_key_id=freeze_signer.key_id,
    )
    bundle_digest = digest_for(core)
    signature = freeze_signer.sign_bytes(_signed_payload(core, bundle_digest))
    return dict(canonical_value({**core, "bundle_digest": bundle_digest, "signature": signature}))


def validate_corpus_freeze_bundle(
    record: Mapping[str, Any],
    *,
    public_manifest: Mapping[str, Any],
    private_manifest: Mapping[str, Any],
    runtime_manifest: Mapping[str, Any],
    blind_solver_manifest: Mapping[str, Any],
    blind_solver_source: bytes,
    blind_solver_version: str,
    evaluator_revision: str,
    controller_public_key: Any,
    freeze_public_key: Any,
) -> Dict[str, Any]:
    """Verify a bundle against every exact frozen input and external key."""

    bundle = _closed(record, _FREEZE_FIELDS, "Gate-A corpus freeze bundle")
    public, private, runtime, blind = _validated_manifests(
        public_manifest=public_manifest,
        private_manifest=private_manifest,
        runtime_manifest=runtime_manifest,
        blind_solver_manifest=blind_solver_manifest,
        blind_solver_source=blind_solver_source,
        blind_solver_version=blind_solver_version,
    )
    revision = _nonempty_text(evaluator_revision, "evaluator revision")
    validate_sha256(revision, "evaluator revision")
    controller_key = _validate_public_key_only(controller_public_key, "controller public key")
    freeze_key = _validate_public_key_only(freeze_public_key, "freeze public key")
    expected_core = _freeze_core(
        public_manifest_digest=public["manifest_digest"],
        private_manifest_digest=private["manifest_digest"],
        runtime_manifest_digest=runtime["manifest_digest"],
        blind_solver_manifest_digest=blind["manifest_digest"],
        blind_solver_source_digest=blind["source_digest"],
        blind_solver_version=blind["solver_version"],
        evaluator_revision=revision,
        controller_public_key=controller_key,
        freeze_signer_key_id=key_id_for_public_key(freeze_key),
    )
    if (
        bundle.get("scope") != CORPUS_VALIDATION_SCOPE
        or bundle.get("task_bearing_model_requests_permitted") is not False
    ):
        raise ValueError("Gate-A freeze bundle has task-bearing scope")
    actual_core = {key: bundle[key] for key in expected_core}
    if actual_core != expected_core:
        raise ValueError("Gate-A freeze bundle differs from the exact expected bindings")
    validate_sha256(bundle.get("bundle_digest"), "Gate-A bundle digest")
    expected_digest = digest_for(expected_core)
    if bundle["bundle_digest"] != expected_digest:
        raise ValueError("Gate-A freeze bundle digest mismatch")
    signature = _b64url_decode(bundle.get("signature"), size=64, label="Gate-A freeze signature")
    try:
        load_public_key(freeze_key).verify(signature, _signed_payload(expected_core, expected_digest))
    except Exception as exc:
        raise ValueError("Gate-A freeze bundle has an invalid Ed25519 signature") from exc
    return dict(canonical_value(bundle))


__all__ = [
    "BLIND_SOLVER_MANIFEST_VERSION",
    "CORPUS_FREEZE_VERSION",
    "CORPUS_VALIDATION_SCOPE",
    "build_blind_solver_manifest",
    "build_corpus_freeze_bundle",
    "validate_blind_solver_manifest",
    "validate_corpus_freeze_bundle",
]
