"""Spawn-isolated Gate-A corpus validation for the learnable-v2 corpus.

The calling process is an orchestrator only.  It passes absolute paths and a
public Docker configuration to three *sequential* ``spawn`` children:

1. an issuer freezes the exact source/corpus/runtime/controller bindings;
2. an evaluator loads the private corpus and produces signed durable evidence;
3. a fresh verifier replays that evidence and alone may publish the signed
   ``GO_CORPUS_VALIDATION`` record.

The parent never reads the evaluator seed, private keys, private manifest, or
private journals.  Gate A is explicitly non-task-bearing and makes no model
quality or cross-node transport claim.
"""

from __future__ import annotations

import base64
from dataclasses import asdict
import json
import multiprocessing as mp
import os
from pathlib import Path
import re
import secrets
import stat
import subprocess
import sys
import threading
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

from ..canonical import GENESIS_HASH, canonical_bytes, canonical_json, digest_bytes, digest_for, validate_sha256
from ..errors import EGVError, ReceiptConflictError, ReceiptVerificationError
from ..receipts import (
    ReceiptJournal,
    ReceiptSigner,
    key_id_for_public_key,
    load_public_key,
    public_key_bytes,
    receipt_hash,
    verify_receipt,
)
from .sandbox import DockerSandboxConfig


GATE_A_SOURCE_MANIFEST_VERSION = "egv-gate-a-source-manifest-v1"
GATE_A_PLAN_VERSION = "egv-gate-a-corpus-plan-v1"
GATE_A_ISSUER_VERSION = "egv-gate-a-issuer-record-v1"
GATE_A_REPORT_VERSION = "egv-gate-a-evaluator-report-v1"
GATE_A_PUBLIC_RESULT_VERSION = "egv-gate-a-public-result-v1"
GATE_A_DELIVERY_VERSION = "egv-gate-a-delivery-v1"
GATE_A_PRIVACY_VERSION = "egv-gate-a-public-privacy-scan-v1"
GATE_A_FAILURE_VERSION = "egv-gate-a-private-failure-v1"
GATE_A_BINDING_VERSION = "egv-gate-a-evaluation-bindings-v1"
GATE_A_DISPOSITION = "GO_CORPUS_VALIDATION"
GATE_A_CAMPAIGN_ID = "egv-learnable-v2-gate-a-corpus-validation"
GATE_A_STAGE_TIMEOUT_SECONDS = 4 * 60 * 60

ORACLE_TASK_COUNT = 170
ORACLE_EVALUATION_COUNT = 340
BLIND_TASK_COUNT = 18
BLIND_EVALUATION_COUNT = 36
STARTER_TASK_COUNT = 18
STARTER_EVALUATION_COUNT = 18
TOTAL_EVALUATION_COUNT = 394
EXPECTED_RECEIPT_COUNT = 1182

_SPAWN_ENVIRONMENT_LOCK = threading.Lock()

_ISSUER_CORE_FIELDS = (
    "schema_version",
    "source_manifest_digest",
    "source_commit",
    "evaluator_revision",
    "plan_digest",
    "public_manifest_digest",
    "private_manifest_digest",
    "runtime_manifest_digest",
    "blind_solver_manifest_digest",
    "blind_solver_source_digest",
    "controller_key_id",
    "freeze_signer_key_id",
    "freeze_bundle_digest",
)
_REPORT_CORE_FIELDS = (
    "schema_version",
    "evidence_scope",
    "freeze_bundle_digest",
    "plan_digest",
    "protocol_digest",
    "policy_digest",
    "public_manifest_digest",
    "private_manifest_digest",
    "runtime_manifest_digest",
    "blind_solver_manifest_digest",
    "blind_solver_source_digest",
    "source_manifest_digest",
    "source_commit",
    "candidate_source_inventory_digest",
    "evaluation_binding_digest",
    "controller_key_id",
    "freeze_signer_key_id",
    "oracle_task_count",
    "oracle_evaluation_count",
    "oracle_pass_count",
    "blind_task_count",
    "blind_evaluation_count",
    "blind_pass_count",
    "starter_task_count",
    "starter_evaluation_count",
    "starter_wrong_output_count",
    "total_evaluation_count",
    "infrastructure_loss_count",
    "expected_receipt_count",
    "receipt_count",
    "receipt_chain_head",
    "receipt_chain_valid",
    "receipt_journal_digest",
    "delivery_count",
    "delivery_chain_head",
    "delivery_chain_valid",
    "delivery_journal_digest",
    "receipt_delivery_bijection",
    "sandbox_preflight_passed",
    "sandbox_preflight_digest",
    "sandbox_negative_control_count",
    "sandbox_negative_controls_all_passed",
    "sandbox_negative_controls_digest",
    "sandbox_postflight_passed",
    "production_acceptance",
    "task_bearing_model_requests_permitted",
    "model_quality_claimed",
    "cross_node_transport_exercised",
    "hidden_values_exposed",
    "private_paths_exposed",
)
_PUBLIC_CORE_FIELDS = (
    "schema_version",
    "disposition",
    "evidence_scope",
    "production_acceptance",
    "task_bearing_model_requests_permitted",
    "separate_go_task_bearing_required",
    "model_quality_claimed",
    "cross_node_transport_exercised",
    "freeze_bundle_digest",
    "plan_digest",
    "protocol_digest",
    "policy_digest",
    "public_manifest_digest",
    "private_manifest_digest",
    "runtime_manifest_digest",
    "blind_solver_manifest_digest",
    "blind_solver_source_digest",
    "source_manifest_digest",
    "source_commit",
    "candidate_source_inventory_digest",
    "evaluation_binding_digest",
    "controller_key_id",
    "freeze_signer_key_id",
    "oracle_task_count",
    "oracle_evaluation_count",
    "oracle_pass_count",
    "blind_task_count",
    "blind_evaluation_count",
    "blind_pass_count",
    "starter_task_count",
    "starter_evaluation_count",
    "starter_wrong_output_count",
    "total_evaluation_count",
    "infrastructure_loss_count",
    "expected_receipt_count",
    "receipt_count",
    "receipt_chain_head",
    "receipt_chain_valid",
    "receipt_journal_digest",
    "delivery_count",
    "delivery_chain_head",
    "delivery_chain_valid",
    "delivery_journal_digest",
    "receipt_delivery_bijection",
    "controller_report_digest",
    "controller_report_file_digest",
    "sandbox_preflight_passed",
    "sandbox_negative_control_count",
    "sandbox_negative_controls_all_passed",
    "sandbox_negative_controls_digest",
    "sandbox_postflight_passed",
    "privacy_scan",
)

_SOURCE_MANIFEST_FIELDS = frozenset({"schema_version", "source_commit", "file_count", "files", "manifest_digest"})
_SOURCE_FILE_FIELDS = frozenset({"path", "size_bytes", "sha256"})
_PLAN_FIELDS = frozenset(
    {
        "schema_version",
        "scope",
        "task_bearing_model_requests_permitted",
        "model_quality_claimed",
        "cross_node_transport_exercised",
        "oracle_task_count",
        "oracle_evaluation_count",
        "blind_task_count",
        "blind_evaluation_count",
        "starter_task_count",
        "starter_evaluation_count",
        "total_evaluation_count",
        "expected_receipt_count",
        "plan_digest",
    }
)
_DELIVERY_FIELDS = frozenset(
    {
        "schema_version",
        "delivery_sequence",
        "previous_delivery_hash",
        "receipt_id",
        "receipt_hash",
        "delivery_hash",
    }
)
_PRIVACY_PATTERNS = (
    rb"[a-z]:\\\\",
    rb"/(?:home|users|tmp|var/tmp)/",
    rb"\\\\[a-z0-9_.-]+\\",
    rb"(?:10\.|192\.168\.|172\.(?:1[6-9]|2[0-9]|3[01])\.)[0-9]{1,3}\.[0-9]{1,3}",
    rb"-----begin (?:encrypted )?private key-----",
    rb'"(?:seed|seed_bytes|private_key|controller_private_key|freeze_private_key)"',
)
_IDENTITY_FIELDS = frozenset(
    {
        "evaluator_seed",
        "controller_private",
        "controller_public",
        "freeze_private",
        "freeze_public",
        "package_root",
        "workspace",
        "private",
        "issuer",
        "evaluator",
        "failures",
        "output_parent",
        "bytecode",
        "issuer_bytecode",
        "evaluator_bytecode",
        "verifier_bytecode",
        "evaluator_seed_state",
        "controller_private_state",
        "controller_public_state",
        "freeze_private_state",
        "freeze_public_state",
    }
)


def _closed(record: Any, fields: Iterable[str], label: str) -> Mapping[str, Any]:
    if not isinstance(record, Mapping) or set(record) != set(fields):
        raise ValueError("{} is not a closed record".format(label))
    return record


def _b64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _signature_payload(core: Mapping[str, Any], digest_field: str, digest: str, signature_field: str) -> bytes:
    return canonical_bytes({**dict(core), digest_field: digest, signature_field: None})


def _sign_closed_record(
    core: Mapping[str, Any], signer: ReceiptSigner, *, digest_field: str, signature_field: str
) -> Dict[str, Any]:
    digest = digest_for(core)
    signature = signer.sign_bytes(_signature_payload(core, digest_field, digest, signature_field))
    return {**dict(core), digest_field: digest, signature_field: signature}


def _verify_closed_signature(
    record: Mapping[str, Any],
    public_key: Any,
    *,
    core_fields: Sequence[str],
    digest_field: str,
    signature_field: str,
    label: str,
) -> Dict[str, Any]:
    expected_fields = set(core_fields) | {digest_field, signature_field}
    closed = _closed(record, expected_fields, label)
    core = {field: closed[field] for field in core_fields}
    validate_sha256(closed.get(digest_field), digest_field)
    expected_digest = digest_for(core)
    if closed[digest_field] != expected_digest:
        raise ValueError("{} digest mismatch".format(label))
    signature_text = closed.get(signature_field)
    if not isinstance(signature_text, str) or not signature_text:
        raise ValueError("{} signature is missing".format(label))
    try:
        padded = signature_text + "=" * (-len(signature_text) % 4)
        signature = base64.b64decode(padded.encode("ascii"), altchars=b"-_", validate=True)
        if len(signature) != 64 or _b64url(signature) != signature_text:
            raise ValueError("non-canonical signature")
        load_public_key(public_key).verify(
            signature,
            _signature_payload(core, digest_field, expected_digest, signature_field),
        )
    except Exception as exc:
        raise ValueError("{} has an invalid Ed25519 signature".format(label)) from exc
    return dict(closed)


def _canonical_record_bytes(value: Mapping[str, Any]) -> bytes:
    """Return the exact bytes used for one canonical public/private record."""

    return (canonical_json(value) + "\n").encode("utf-8")


def _is_linklike_stat(value: os.stat_result) -> bool:
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return bool(stat.S_ISLNK(value.st_mode) or (getattr(value, "st_file_attributes", 0) & reparse_flag))


def _assert_no_link_components(path: Path, label: str) -> None:
    """Reject symlinks/junctions in every currently existing path component."""

    candidate = Path(path)
    if not candidate.is_absolute():
        raise EGVError("{} must be absolute".format(label))
    current = Path(candidate.anchor)
    for component in candidate.parts[1:]:
        current /= component
        try:
            metadata = os.lstat(current)
        except FileNotFoundError:
            continue
        except OSError as exc:
            raise EGVError("{} path could not be inspected".format(label)) from exc
        if _is_linklike_stat(metadata):
            raise EGVError("{} must not contain symlink or reparse-point components".format(label))


def _path_identity(path: Path, label: str, *, directory: bool = False) -> Tuple[int, int]:
    candidate = Path(path)
    _assert_no_link_components(candidate, label)
    try:
        metadata = os.lstat(candidate)
    except OSError as exc:
        raise EGVError("{} is unavailable".format(label)) from exc
    expected = stat.S_ISDIR(metadata.st_mode) if directory else stat.S_ISREG(metadata.st_mode)
    if not expected or _is_linklike_stat(metadata):
        kind = "directory" if directory else "regular file"
        raise EGVError("{} must be a non-link {}".format(label, kind))
    return (int(metadata.st_dev), int(metadata.st_ino))


def _file_selection_state(path: Path, label: str) -> Tuple[int, int, int, int, int]:
    candidate = Path(path)
    _path_identity(candidate, label, directory=False)
    metadata = os.lstat(candidate)
    return (
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _require_file_selection_state(
    path: Path,
    expected: Tuple[int, int, int, int, int],
    label: str,
) -> None:
    if _file_selection_state(path, label) != expected:
        raise EGVError("{} content identity changed after parent selection".format(label))


def _require_identity(path: Path, expected: Tuple[int, int], label: str, *, directory: bool) -> None:
    if _path_identity(path, label, directory=directory) != expected:
        raise EGVError("{} identity changed during Gate-A execution".format(label))


def _write_all(descriptor: int, payload: bytes) -> None:
    offset = 0
    while offset < len(payload):
        written = os.write(descriptor, payload[offset:])
        if written <= 0:
            raise OSError("short write")
        offset += written


def _read_stable_regular_bytes(
    path: Path,
    label: str,
    *,
    expected_identity: Optional[Tuple[int, int]] = None,
    expected_state: Optional[Tuple[int, int, int, int, int]] = None,
) -> bytes:
    """Read one regular file through a descriptor bound to its checked identity."""

    source = Path(path)
    current = _path_identity(source, label, directory=False)
    if expected_identity is not None and current != expected_identity:
        raise EGVError("{} differs from the parent-selected file identity".format(label))
    expected = expected_identity or current
    if expected_state is not None:
        _require_file_selection_state(source, expected_state, label)
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_BINARY", 0)
    descriptor = os.open(str(source), flags)
    try:
        opened = os.fstat(descriptor)
        opened_identity = (int(opened.st_dev), int(opened.st_ino))
        if not stat.S_ISREG(opened.st_mode) or opened_identity != expected:
            raise EGVError("{} identity changed before read".format(label))
        opened_state = (int(opened.st_size), int(opened.st_mtime_ns), int(opened.st_ctime_ns))
        if expected_state is not None and (opened_identity != expected_state[:2] or opened_state != expected_state[2:]):
            raise EGVError("{} content changed before descriptor read".format(label))
        chunks = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        final = os.fstat(descriptor)
        final_state = (int(final.st_size), int(final.st_mtime_ns), int(final.st_ctime_ns))
        if (int(final.st_dev), int(final.st_ino)) != expected or final_state != opened_state:
            raise EGVError("{} identity changed during read".format(label))
    finally:
        os.close(descriptor)
    _require_identity(source, expected, label, directory=False)
    if expected_state is not None:
        _require_file_selection_state(source, expected_state, label)
    payload = b"".join(chunks)
    if len(payload) != opened_state[0]:
        raise EGVError("{} size changed during read".format(label))
    return payload


def _append_fsynced_line(
    path: Path,
    payload: bytes,
    *,
    parent_identity: Tuple[int, int],
    file_identity: Optional[Tuple[int, int]],
) -> Tuple[int, int]:
    """Append via a checked descriptor so a swapped path is never followed."""

    destination = Path(path)
    _require_identity(destination.parent, parent_identity, "Gate-A journal parent", directory=True)
    flags = os.O_WRONLY | os.O_APPEND
    if file_identity is None:
        flags |= os.O_CREAT | os.O_EXCL
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    binary = getattr(os, "O_BINARY", 0)
    descriptor = os.open(str(destination), flags | nofollow | binary, 0o600)
    try:
        opened = os.fstat(descriptor)
        opened_identity = (int(opened.st_dev), int(opened.st_ino))
        if not stat.S_ISREG(opened.st_mode) or (file_identity is not None and opened_identity != file_identity):
            raise EGVError("Gate-A journal identity changed during append")
        _write_all(descriptor, payload)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    _require_identity(destination.parent, parent_identity, "Gate-A journal parent", directory=True)
    _require_identity(destination, opened_identity, "Gate-A journal", directory=False)
    if file_identity is None and os.name != "nt":
        parent_descriptor = os.open(
            str(destination.parent),
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0),
        )
        try:
            opened_parent = os.fstat(parent_descriptor)
            if (int(opened_parent.st_dev), int(opened_parent.st_ino)) != parent_identity:
                raise EGVError("Gate-A journal parent identity changed before namespace fsync")
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
    return opened_identity


def _windows_directory_path(handle: int) -> Path:
    import ctypes
    from ctypes import wintypes

    get_path = ctypes.WinDLL("kernel32", use_last_error=True).GetFinalPathNameByHandleW
    get_path.argtypes = [wintypes.HANDLE, wintypes.LPWSTR, wintypes.DWORD, wintypes.DWORD]
    get_path.restype = wintypes.DWORD
    buffer = ctypes.create_unicode_buffer(32768)
    length = get_path(handle, buffer, len(buffer), 0)
    if length == 0 or length >= len(buffer):
        raise EGVError("Gate-A evidence parent handle could not be resolved")
    value = buffer.value
    if value.startswith("\\\\?\\UNC\\"):
        value = "\\\\" + value[8:]
    elif value.startswith("\\\\?\\"):
        value = value[4:]
    return Path(value)


def _open_identity_bound_parent(path: Path, expected_identity: Tuple[int, int]) -> Tuple[str, int]:
    if os.name != "nt":
        flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
        descriptor = os.open(str(path), flags)
        opened = os.fstat(descriptor)
        if not stat.S_ISDIR(opened.st_mode) or (int(opened.st_dev), int(opened.st_ino)) != expected_identity:
            os.close(descriptor)
            raise EGVError("Gate-A evidence destination parent identity changed")
        return ("posix", descriptor)

    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    handle = create_file(
        str(path),
        0,
        0x00000001 | 0x00000002,
        None,
        3,
        0x02000000,
        None,
    )
    if handle == ctypes.c_void_p(-1).value:
        raise EGVError("Gate-A evidence destination parent could not be opened")
    try:
        _require_identity(path, expected_identity, "Gate-A evidence destination parent", directory=True)
    except BaseException:
        kernel32.CloseHandle(handle)
        raise
    return ("windows", int(handle))


def _close_identity_bound_parent(kind: str, handle: int) -> None:
    if kind == "posix":
        os.close(handle)
        return
    import ctypes

    ctypes.WinDLL("kernel32", use_last_error=True).CloseHandle(handle)


def _unlink_if_exact(path: Path, expected_identity: Tuple[int, int]) -> None:
    try:
        if _path_identity(path, "Gate-A exact cleanup target", directory=False) == expected_identity:
            path.unlink()
    except (EGVError, OSError):
        return


def _atomic_write_new(
    path: Path,
    value: Mapping[str, Any],
    *,
    private: bool,
    expected_parent_identity: Optional[Tuple[int, int]] = None,
) -> None:
    """Publish one canonical record through an identity-bound parent handle."""

    destination = Path(path)
    _assert_no_link_components(destination, "Gate-A evidence destination")
    current_parent_identity = _path_identity(destination.parent, "Gate-A evidence destination parent", directory=True)
    if expected_parent_identity is not None and current_parent_identity != expected_parent_identity:
        raise EGVError("Gate-A evidence destination parent identity changed")
    parent_identity = expected_parent_identity or current_parent_identity
    if destination.exists() or destination.is_symlink():
        raise EGVError("Gate-A evidence destination already exists")
    payload = _canonical_record_bytes(value)
    parent_kind, parent_handle = _open_identity_bound_parent(destination.parent, parent_identity)
    temporary_name = ".{}.{}.tmp".format(destination.name, secrets.token_hex(16))
    temporary_identity: Optional[Tuple[int, int]] = None
    success = False
    try:
        if parent_kind == "posix":
            flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(temporary_name, flags, 0o600, dir_fd=parent_handle)
            try:
                opened = os.fstat(descriptor)
                temporary_identity = (int(opened.st_dev), int(opened.st_ino))
                _write_all(descriptor, payload)
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
            _require_identity(
                destination.parent,
                parent_identity,
                "Gate-A evidence destination parent",
                directory=True,
            )
            try:
                os.link(
                    temporary_name,
                    destination.name,
                    src_dir_fd=parent_handle,
                    dst_dir_fd=parent_handle,
                    follow_symlinks=False,
                )
            except FileExistsError as exc:
                raise EGVError("Gate-A evidence destination already exists") from exc
            published = os.stat(destination.name, dir_fd=parent_handle, follow_symlinks=False)
            if (int(published.st_dev), int(published.st_ino)) != temporary_identity:
                raise EGVError("Gate-A evidence destination identity changed")
            _require_identity(
                destination.parent,
                parent_identity,
                "Gate-A evidence destination parent",
                directory=True,
            )
            os.unlink(temporary_name, dir_fd=parent_handle)
            temporary_identity = None
            os.fsync(parent_handle)
            success = True
            return

        active_parent = _windows_directory_path(parent_handle)
        _require_identity(active_parent, parent_identity, "Gate-A evidence destination parent", directory=True)
        temporary = active_parent / temporary_name
        descriptor = os.open(
            str(temporary),
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o600,
        )
        try:
            opened = os.fstat(descriptor)
            temporary_identity = (int(opened.st_dev), int(opened.st_ino))
            _write_all(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        _require_identity(destination.parent, parent_identity, "Gate-A evidence destination parent", directory=True)
        try:
            os.link(str(temporary), str(destination))
        except FileExistsError as exc:
            raise EGVError("Gate-A evidence destination already exists") from exc
        _require_identity(destination, temporary_identity, "Gate-A evidence destination", directory=False)
        _require_identity(destination.parent, parent_identity, "Gate-A evidence destination parent", directory=True)
        if private:
            destination.chmod(0o600)
        temporary.unlink()
        temporary_identity = None
        success = True
    finally:
        if not success and temporary_identity is not None:
            _unlink_if_exact(destination, temporary_identity)
            if parent_kind == "windows":
                try:
                    active_parent = _windows_directory_path(parent_handle)
                except EGVError:
                    active_parent = None
                if active_parent is not None:
                    _unlink_if_exact(active_parent / destination.name, temporary_identity)
                    _unlink_if_exact(active_parent / temporary_name, temporary_identity)
            else:
                try:
                    current = os.stat(destination.name, dir_fd=parent_handle, follow_symlinks=False)
                    if (int(current.st_dev), int(current.st_ino)) == temporary_identity:
                        os.unlink(destination.name, dir_fd=parent_handle)
                except (FileNotFoundError, OSError):
                    pass
                try:
                    os.unlink(temporary_name, dir_fd=parent_handle)
                except FileNotFoundError:
                    pass
        _close_identity_bound_parent(parent_kind, parent_handle)


def _read_canonical_record(path: Path, label: str) -> Dict[str, Any]:
    source = Path(path)
    try:
        payload = _read_stable_regular_bytes(source, label)
        parsed = json.loads(payload.decode("utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise EGVError("{} is not canonical JSON".format(label)) from exc
    if payload != _canonical_record_bytes(parsed) or not isinstance(parsed, dict):
        raise EGVError("{} is not canonical JSON".format(label))
    return parsed


def _read_private_bytes(
    path: Union[str, Path],
    label: str,
    *,
    exact_size: int = 32,
    expected_identity: Optional[Tuple[int, int]] = None,
    expected_state: Optional[Tuple[int, int, int, int, int]] = None,
) -> bytes:
    source = Path(path)
    try:
        payload = _read_stable_regular_bytes(
            source,
            label,
            expected_identity=expected_identity,
            expected_state=expected_state,
        )
    except (OSError, EGVError) as exc:
        raise EGVError("{} is unavailable".format(label)) from exc
    if len(payload) != exact_size:
        raise EGVError("{} must contain exactly {} bytes".format(label, exact_size))
    return payload


def _read_public_key(
    path: Union[str, Path],
    label: str,
    *,
    expected_identity: Optional[Tuple[int, int]] = None,
    expected_state: Optional[Tuple[int, int, int, int, int]] = None,
) -> bytes:
    source = Path(path)
    try:
        payload = _read_stable_regular_bytes(
            source,
            label,
            expected_identity=expected_identity,
            expected_state=expected_state,
        )
    except (OSError, EGVError) as exc:
        raise EGVError("{} is unavailable".format(label)) from exc
    if not payload or len(payload) > 8192:
        raise EGVError("{} has an invalid bounded size".format(label))
    try:
        return public_key_bytes(payload)
    except Exception as exc:
        raise EGVError("{} is not an Ed25519 public key".format(label)) from exc


def _validated_external_signer(
    private_path: Union[str, Path],
    public_path: Union[str, Path],
    label: str,
    *,
    private_identity: Optional[Tuple[int, int]] = None,
    public_identity: Optional[Tuple[int, int]] = None,
    private_state: Optional[Tuple[int, int, int, int, int]] = None,
    public_state: Optional[Tuple[int, int, int, int, int]] = None,
) -> ReceiptSigner:
    signer = ReceiptSigner(
        _read_private_bytes(
            private_path,
            label + " private key",
            expected_identity=private_identity,
            expected_state=private_state,
        )
    )
    public = _read_public_key(
        public_path,
        label + " public key",
        expected_identity=public_identity,
        expected_state=public_state,
    )
    if signer.public_key_raw != public:
        raise EGVError("{} private and public keys do not match".format(label))
    return signer


def _require_distinct_authority_keys(controller_public_key: Any, freeze_public_key: Any) -> None:
    if key_id_for_public_key(controller_public_key) == key_id_for_public_key(freeze_public_key):
        raise EGVError("Gate-A controller and freeze authority must use distinct Ed25519 keys")


def _git_output(package_root: Path, *arguments: str) -> str:
    try:
        completed = subprocess.run(
            ["git", "-C", str(package_root), *arguments],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=30,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise EGVError("Gate-A source checkout could not be verified") from exc
    if completed.returncode != 0:
        raise EGVError("Gate-A source checkout could not be verified")
    return completed.stdout


def build_evaluator_source_manifest(
    package_root: Union[str, Path],
    source_commit: str,
    *,
    require_clean_git: bool = True,
    expected_root_identity: Optional[Tuple[int, int]] = None,
) -> Dict[str, Any]:
    """Inventory every ``egv/**/*.py`` byte and bind it to the exact checkout."""

    supplied_root = Path(package_root)
    current_root_identity = _path_identity(supplied_root, "Gate-A package root", directory=True)
    if expected_root_identity is not None and current_root_identity != expected_root_identity:
        raise EGVError("Gate-A package root identity changed after parent selection")
    root = supplied_root.resolve(strict=True)
    if not isinstance(source_commit, str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", source_commit):
        raise EGVError("Gate-A source commit must be a full lowercase Git object ID")
    if require_clean_git:
        actual = _git_output(root, "rev-parse", "--verify", "HEAD").strip().lower()
        if actual != source_commit:
            raise EGVError("Gate-A source commit differs from checkout HEAD")
        status = _git_output(root, "status", "--porcelain=v1", "--untracked-files=all", "--", "egv")
        if status:
            raise EGVError("Gate-A evaluator source tree is not clean at the supplied commit")
    source_root = root / "egv"
    _path_identity(source_root, "Gate-A egv source tree", directory=True)
    entries = []
    for source in sorted(source_root.rglob("*.py"), key=lambda item: item.relative_to(root).as_posix()):
        try:
            _path_identity(source, "Gate-A source inventory file", directory=False)
        except EGVError as exc:
            raise EGVError("Gate-A source inventory contains a non-regular Python file") from exc
        payload = _read_stable_regular_bytes(source, "Gate-A source inventory file")
        entries.append(
            {
                "path": source.relative_to(root).as_posix(),
                "size_bytes": len(payload),
                "sha256": digest_bytes(payload),
            }
        )
    if not entries:
        raise EGVError("Gate-A evaluator source inventory is empty")
    if require_clean_git:
        actual_after = _git_output(root, "rev-parse", "--verify", "HEAD").strip().lower()
        status_after = _git_output(root, "status", "--porcelain=v1", "--untracked-files=all", "--", "egv")
        if actual_after != source_commit or status_after:
            raise EGVError("Gate-A evaluator source tree changed during inventory")
    core: Dict[str, Any] = {
        "schema_version": GATE_A_SOURCE_MANIFEST_VERSION,
        "source_commit": source_commit,
        "file_count": len(entries),
        "files": entries,
    }
    return {**core, "manifest_digest": digest_for(core)}


def validate_evaluator_source_manifest(record: Mapping[str, Any]) -> Dict[str, Any]:
    manifest = _closed(record, _SOURCE_MANIFEST_FIELDS, "Gate-A source manifest")
    if manifest.get("schema_version") != GATE_A_SOURCE_MANIFEST_VERSION:
        raise ValueError("unsupported Gate-A source manifest")
    if not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", str(manifest.get("source_commit", ""))):
        raise ValueError("Gate-A source manifest has an invalid commit")
    files = manifest.get("files")
    if not isinstance(files, list) or not files or manifest.get("file_count") != len(files):
        raise ValueError("Gate-A source manifest has an invalid file count")
    paths = []
    for entry in files:
        file_record = _closed(entry, _SOURCE_FILE_FIELDS, "Gate-A source file")
        path = file_record.get("path")
        size = file_record.get("size_bytes")
        if (
            not isinstance(path, str)
            or not path.startswith("egv/")
            or not path.endswith(".py")
            or ".." in Path(path).parts
            or not isinstance(size, int)
            or isinstance(size, bool)
            or size < 0
        ):
            raise ValueError("Gate-A source file record is invalid")
        validate_sha256(file_record.get("sha256"), "source file digest")
        paths.append(path)
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise ValueError("Gate-A source manifest ordering or uniqueness is invalid")
    validate_sha256(manifest.get("manifest_digest"), "source manifest digest")
    if manifest["manifest_digest"] != digest_for({key: manifest[key] for key in manifest if key != "manifest_digest"}):
        raise ValueError("Gate-A source manifest digest mismatch")
    return dict(manifest)


def build_gate_a_plan() -> Dict[str, Any]:
    from .learnable_freeze import CORPUS_VALIDATION_SCOPE

    core: Dict[str, Any] = {
        "schema_version": GATE_A_PLAN_VERSION,
        "scope": CORPUS_VALIDATION_SCOPE,
        "task_bearing_model_requests_permitted": False,
        "model_quality_claimed": False,
        "cross_node_transport_exercised": False,
        "oracle_task_count": ORACLE_TASK_COUNT,
        "oracle_evaluation_count": ORACLE_EVALUATION_COUNT,
        "blind_task_count": BLIND_TASK_COUNT,
        "blind_evaluation_count": BLIND_EVALUATION_COUNT,
        "starter_task_count": STARTER_TASK_COUNT,
        "starter_evaluation_count": STARTER_EVALUATION_COUNT,
        "total_evaluation_count": TOTAL_EVALUATION_COUNT,
        "expected_receipt_count": EXPECTED_RECEIPT_COUNT,
    }
    return {**core, "plan_digest": digest_for(core)}


def validate_gate_a_plan(record: Mapping[str, Any]) -> Dict[str, Any]:
    expected = build_gate_a_plan()
    _closed(record, _PLAN_FIELDS, "Gate-A evaluation plan")
    if record != expected:
        raise ValueError("Gate-A evaluation plan differs from the frozen 394-evaluation plan")
    return dict(record)


def _fresh_receipt_replay(path: Union[str, Path], public_key: Any) -> Tuple[list[Dict[str, Any]], Dict[str, Any]]:
    """Replay canonical receipt bytes once and bind evidence to those bytes."""

    source = Path(path)
    try:
        payload = _read_stable_regular_bytes(source, "Gate-A receipt journal")
        text_payload = payload.decode("utf-8")
    except (EGVError, OSError, UnicodeError) as exc:
        raise ReceiptVerificationError("receipt journal is unavailable or non-UTF-8") from exc
    if payload and not payload.endswith(b"\n"):
        raise ReceiptVerificationError("receipt journal has an interrupted final record")
    if b"\r\n" in payload:
        raise ReceiptVerificationError("receipt journal line ending is not canonical")
    records: list[Dict[str, Any]] = []
    previous = GENESIS_HASH
    try:
        for sequence, line in enumerate(text_payload.splitlines(), start=1):
            if not line:
                raise ReceiptVerificationError("receipt journal contains a blank record")
            parsed = json.loads(line)
            if not isinstance(parsed, dict) or canonical_json(parsed) != line:
                raise ReceiptVerificationError("receipt journal contains non-canonical JSON")
            current = verify_receipt(
                parsed,
                public_key,
                expected_sequence=sequence,
                expected_previous_hash=previous,
            )
            records.append(dict(parsed))
            previous = current
    except (ValueError, TypeError) as exc:
        if isinstance(exc, ReceiptVerificationError):
            raise
        raise ReceiptVerificationError("receipt journal is not canonical JSONL") from exc
    return records, {
        "count": len(records),
        "head": previous,
        "chain_valid": True,
        "journal_digest": digest_bytes(payload),
    }


class StreamingReceiptJournal(ReceiptJournal):
    """ReceiptJournal with one startup replay and O(1) one-shot appends.

    The inherited verifier remains the independent source of truth.  This
    class changes only evaluator write performance; signatures, sequence,
    previous-hash checks, canonical JSONL, flush, and fsync remain mandatory.
    """

    def __init__(
        self,
        path: Union[str, Path],
        public_key: Any,
        *,
        expected_parent_identity: Optional[Tuple[int, int]] = None,
    ) -> None:
        candidate = Path(path)
        current_parent_identity = _path_identity(candidate.parent, "Gate-A receipt journal parent", directory=True)
        if expected_parent_identity is not None and current_parent_identity != expected_parent_identity:
            raise EGVError("Gate-A receipt journal parent identity changed")
        self._parent_identity = expected_parent_identity or current_parent_identity
        self._file_identity: Optional[Tuple[int, int]] = None
        if candidate.exists() or candidate.is_symlink():
            self._file_identity = _path_identity(candidate, "Gate-A receipt journal", directory=False)
        # Avoid ReceiptJournal's convenience mkdir here: this runtime requires
        # a pre-created, identity-checked private parent.
        self.path = candidate
        self.public_key = load_public_key(public_key)
        records = ReceiptJournal.receipts(self)
        self._records = records
        self._by_key = {record["idempotency_key"]: record for record in records}
        self._by_id = {record["receipt_id"]: record for record in records}
        self._head = receipt_hash(records[-1]) if records else GENESIS_HASH

    def receipts(self) -> list[Dict[str, Any]]:
        return [dict(record) for record in self._records]

    def append(self, receipt: Mapping[str, Any]) -> Dict[str, Any]:
        candidate = dict(receipt)
        idempotency_key = candidate.get("idempotency_key")
        receipt_id = candidate.get("receipt_id")
        if idempotency_key in self._by_key:
            existing = self._by_key[idempotency_key]
            if canonical_json(existing) != canonical_json(candidate):
                raise ReceiptConflictError("receipt idempotency key was delivered with conflicting content")
            return dict(existing)
        if receipt_id in self._by_id:
            existing = self._by_id[receipt_id]
            if canonical_json(existing) != canonical_json(candidate):
                raise ReceiptConflictError("receipt ID was delivered with conflicting content")
            return dict(existing)
        verify_receipt(
            candidate,
            self.public_key,
            expected_sequence=len(self._records) + 1,
            expected_previous_hash=self._head,
        )
        self._file_identity = _append_fsynced_line(
            self.path,
            (canonical_json(candidate) + "\n").encode("utf-8"),
            parent_identity=self._parent_identity,
            file_identity=self._file_identity,
        )
        self._records.append(candidate)
        self._by_key[str(idempotency_key)] = candidate
        self._by_id[str(receipt_id)] = candidate
        self._head = receipt_hash(candidate)
        return dict(candidate)


class DurableDeliveryJournal:
    """Separate fsynced hash chain proving every signed receipt was delivered."""

    def __init__(
        self,
        path: Union[str, Path],
        *,
        expected_parent_identity: Optional[Tuple[int, int]] = None,
    ) -> None:
        self.path = Path(path)
        current_parent_identity = _path_identity(self.path.parent, "Gate-A delivery journal parent", directory=True)
        if expected_parent_identity is not None and current_parent_identity != expected_parent_identity:
            raise EGVError("Gate-A delivery journal parent identity changed")
        self._parent_identity = expected_parent_identity or current_parent_identity
        if self.path.exists() or self.path.is_symlink():
            raise EGVError("Gate-A delivery journal must be a new path")
        self._file_identity: Optional[Tuple[int, int]] = None
        self._count = 0
        self._head = GENESIS_HASH
        self._receipt_ids: set[str] = set()

    def append(self, receipt: Mapping[str, Any]) -> Dict[str, Any]:
        receipt_id = receipt.get("receipt_id")
        if not isinstance(receipt_id, str) or not receipt_id:
            raise ReceiptVerificationError("delivered receipt has no receipt ID")
        if receipt_id in self._receipt_ids:
            raise ReceiptConflictError("receipt was delivered more than once")
        core: Dict[str, Any] = {
            "schema_version": GATE_A_DELIVERY_VERSION,
            "delivery_sequence": self._count + 1,
            "previous_delivery_hash": self._head,
            "receipt_id": receipt_id,
            "receipt_hash": receipt_hash(receipt),
        }
        record = {**core, "delivery_hash": digest_for(core)}
        self._file_identity = _append_fsynced_line(
            self.path,
            (canonical_json(record) + "\n").encode("utf-8"),
            parent_identity=self._parent_identity,
            file_identity=self._file_identity,
        )
        self._count += 1
        self._head = record["delivery_hash"]
        self._receipt_ids.add(receipt_id)
        return record

    @property
    def count(self) -> int:
        return self._count

    @property
    def head(self) -> str:
        return self._head

    @classmethod
    def replay(cls, path: Union[str, Path], receipts: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
        """Freshly replay the chain and prove an exact receipt-delivery bijection."""

        source = Path(path)
        try:
            payload = _read_stable_regular_bytes(source, "Gate-A delivery journal")
            text_payload = payload.decode("utf-8")
        except (EGVError, OSError) as exc:
            raise ReceiptVerificationError("delivery journal is unavailable") from exc
        except UnicodeError as exc:
            raise ReceiptVerificationError("delivery journal is not UTF-8") from exc
        if payload and not payload.endswith(b"\n"):
            raise ReceiptVerificationError("delivery journal has an interrupted final record")
        if b"\r\n" in payload:
            raise ReceiptVerificationError("delivery journal line ending is not canonical")
        receipt_by_id: Dict[str, Mapping[str, Any]] = {}
        for receipt in receipts:
            receipt_id = receipt.get("receipt_id")
            if not isinstance(receipt_id, str) or not receipt_id or receipt_id in receipt_by_id:
                raise ReceiptVerificationError("receipt replay has missing or duplicate receipt IDs")
            receipt_by_id[receipt_id] = receipt
        previous = GENESIS_HASH
        delivered_ids = []
        try:
            for sequence, line in enumerate(text_payload.splitlines(), start=1):
                if not line:
                    raise ReceiptVerificationError("delivery journal contains a blank record")
                parsed = json.loads(line)
                if canonical_json(parsed) != line:
                    raise ReceiptVerificationError("delivery journal contains non-canonical JSON")
                record = _closed(parsed, _DELIVERY_FIELDS, "Gate-A delivery record")
                if (
                    record.get("schema_version") != GATE_A_DELIVERY_VERSION
                    or record.get("delivery_sequence") != sequence
                    or record.get("previous_delivery_hash") != previous
                ):
                    raise ReceiptVerificationError("delivery journal sequence or chain is invalid")
                receipt_id = record.get("receipt_id")
                if receipt_id not in receipt_by_id or receipt_id in delivered_ids:
                    raise ReceiptVerificationError("delivery journal is not a receipt bijection")
                expected_receipt = receipts[sequence - 1]
                if receipt_id != expected_receipt.get("receipt_id") or record.get("receipt_hash") != receipt_hash(
                    expected_receipt
                ):
                    raise ReceiptVerificationError("delivery journal order differs from signed receipt order")
                if record.get("receipt_hash") != receipt_hash(receipt_by_id[receipt_id]):
                    raise ReceiptVerificationError("delivery journal receipt hash mismatch")
                core = {key: record[key] for key in record if key != "delivery_hash"}
                if record.get("delivery_hash") != digest_for(core):
                    raise ReceiptVerificationError("delivery journal record digest mismatch")
                previous = str(record["delivery_hash"])
                delivered_ids.append(str(receipt_id))
        except (OSError, UnicodeError, ValueError) as exc:
            if isinstance(exc, ReceiptVerificationError):
                raise
            raise ReceiptVerificationError("delivery journal is not canonical JSONL") from exc
        if len(delivered_ids) != len(receipts) or set(delivered_ids) != set(receipt_by_id):
            raise ReceiptVerificationError("delivery journal is not an exact receipt bijection")
        return {
            "count": len(delivered_ids),
            "head": previous,
            "chain_valid": True,
            "receipt_delivery_bijection": True,
            "journal_digest": digest_bytes(payload),
        }


def _config_record(config: DockerSandboxConfig) -> Dict[str, Any]:
    if type(config) is not DockerSandboxConfig:
        raise EGVError("Gate-A requires the exact DockerSandboxConfig type")
    config.validate()
    return dict(asdict(config))


def _config_from_record(record: Mapping[str, Any]) -> DockerSandboxConfig:
    expected = set(DockerSandboxConfig.__dataclass_fields__)
    _closed(record, expected, "Gate-A Docker configuration")
    config = DockerSandboxConfig(**dict(record))
    config.validate()
    return config


def _private_paths(workspace: Union[str, Path]) -> Dict[str, Path]:
    root = Path(workspace)
    private = root / "private"
    return {
        "root": root,
        "private": private,
        "issuer": private / "issuer",
        "evaluator": private / "evaluator",
        "bytecode": private / "bytecode",
        "issuer_bytecode": private / "bytecode" / "issuer",
        "evaluator_bytecode": private / "bytecode" / "evaluator",
        "verifier_bytecode": private / "bytecode" / "verifier",
        "source_manifest": private / "issuer" / "source-manifest.json",
        "plan": private / "issuer" / "evaluation-plan.json",
        "freeze": private / "issuer" / "freeze-bundle.json",
        "issuer_record": private / "issuer" / "issuer-record.json",
        "candidate_inventory": private / "evaluator" / "candidate-source-inventory.json",
        "evaluation_bindings": private / "evaluator" / "evaluation-bindings.json",
        "receipts": private / "evaluator" / "receipts.jsonl",
        "deliveries": private / "evaluator" / "deliveries.jsonl",
        "report": private / "evaluator" / "evaluator-report.json",
        "artifacts": private / "evaluator" / "artifacts",
        "sandbox": private / "evaluator" / "sandbox",
        "failures": private / "failures",
    }


def _validated_identity_record(record: Mapping[str, Any]) -> Dict[str, Tuple[int, ...]]:
    closed = _closed(record, _IDENTITY_FIELDS, "Gate-A parent identity record")
    result: Dict[str, Tuple[int, ...]] = {}
    for name, value in closed.items():
        expected_length = 5 if name.endswith("_state") else 2
        if (
            not isinstance(value, (tuple, list))
            or len(value) != expected_length
            or any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in value)
        ):
            raise EGVError("Gate-A parent identity record is invalid")
        result[name] = tuple(int(item) for item in value)
    return result


def _prepare_private_layout(workspace: Path) -> Dict[str, Tuple[int, int]]:
    """Create the fixed private layout without recursive/link-following mkdirs."""

    paths = _private_paths(workspace)
    identities = {
        "root": _path_identity(paths["root"], "Gate-A workspace", directory=True),
    }
    for name in ("private", "issuer", "evaluator", "failures", "bytecode"):
        destination = paths[name]
        parent_name = "root" if name == "private" else "private"
        _require_identity(
            destination.parent,
            identities[parent_name],
            "Gate-A private layout parent",
            directory=True,
        )
        try:
            destination.mkdir(mode=0o700, exist_ok=False)
        except OSError as exc:
            raise EGVError("Gate-A private layout could not be created safely") from exc
        identities[name] = _path_identity(destination, "Gate-A {} directory".format(name), directory=True)
    for name in ("issuer_bytecode", "evaluator_bytecode", "verifier_bytecode"):
        destination = paths[name]
        _require_identity(
            destination.parent,
            identities["bytecode"],
            "Gate-A bytecode boundary parent",
            directory=True,
        )
        try:
            destination.mkdir(mode=0o700, exist_ok=False)
        except OSError as exc:
            raise EGVError("Gate-A bytecode boundary could not be created safely") from exc
        identities[name] = _path_identity(destination, "Gate-A {} directory".format(name), directory=True)
    return identities


def _validate_private_layout(
    workspace: Union[str, Path],
    identities: Optional[Mapping[str, Tuple[int, ...]]] = None,
) -> None:
    paths = _private_paths(workspace)
    for name in (
        "root",
        "private",
        "issuer",
        "evaluator",
        "failures",
        "bytecode",
        "issuer_bytecode",
        "evaluator_bytecode",
        "verifier_bytecode",
    ):
        identity_name = "workspace" if name == "root" else name
        if identities is None:
            _path_identity(paths[name], "Gate-A {} directory".format(name), directory=True)
        else:
            _require_identity(
                paths[name],
                identities[identity_name],
                "Gate-A {} directory".format(name),
                directory=True,
            )


def _assert_spawn_bytecode_boundary(
    workspace: Union[str, Path],
    stage: str,
    identities: Mapping[str, Tuple[int, ...]],
) -> None:
    """Prove the child started source-only against its fresh empty cache root."""

    if stage not in {"issuer", "evaluator", "verifier"}:
        raise EGVError("unsupported Gate-A bytecode boundary stage")
    name = stage + "_bytecode"
    cache = _private_paths(workspace)[name]
    _require_identity(cache, identities[name], "Gate-A {} bytecode boundary".format(stage), directory=True)
    expected_prefix = os.path.normcase(str(cache.resolve(strict=True)))
    actual_prefix = getattr(sys, "pycache_prefix", None)
    if (
        not sys.dont_write_bytecode
        or not isinstance(actual_prefix, str)
        or os.path.normcase(str(Path(actual_prefix).resolve(strict=True))) != expected_prefix
        or any(cache.iterdir())
    ):
        raise EGVError("Gate-A spawned source-only bytecode boundary is not intact")


def _failure_record(
    workspace: Union[str, Path],
    stage: str,
    exc: BaseException,
    *,
    failures_identity: Optional[Tuple[int, int]] = None,
) -> None:
    try:
        core = {
            "schema_version": GATE_A_FAILURE_VERSION,
            "stage": stage,
            "exception_type": type(exc).__name__,
        }
        record = {**core, "failure_digest": digest_for(core)}
        destination = _private_paths(workspace)["failures"] / (stage + ".json")
        _atomic_write_new(
            destination,
            record,
            private=True,
            expected_parent_identity=failures_identity,
        )
    except BaseException:
        # Failure recording cannot replace or conceal the original fail-closed exit.
        return


def _assert_loaded_source_root(
    package_root: Union[str, Path],
    expected_identity: Optional[Tuple[int, int]] = None,
) -> None:
    actual = Path(__file__).resolve().parents[2]
    supplied = Path(package_root)
    if expected_identity is not None:
        _require_identity(supplied, expected_identity, "Gate-A package root", directory=True)
    expected = supplied.resolve(strict=True)
    if actual != expected:
        raise EGVError("spawned Gate-A code is not loaded from the inventoried package root")


def _protocol_digest(
    *, freeze_bundle_digest: str, plan_digest: str, source_manifest_digest: str, policy_digest: str
) -> str:
    return digest_for(
        {
            "schema_version": "egv-gate-a-protocol-binding-v1",
            "freeze_bundle_digest": freeze_bundle_digest,
            "plan_digest": plan_digest,
            "source_manifest_digest": source_manifest_digest,
            "policy_digest": policy_digest,
        }
    )


def _generate_corpus_from_validated_seed(
    evaluator_seed: Union[str, Path],
    expected_identity: Tuple[int, int],
    expected_state: Tuple[int, int, int, int, int],
) -> Any:
    """Generate the corpus from descriptor-validated seed bytes, never a reopened path."""

    from . import learnable as learnable_module

    secret = _read_private_bytes(
        evaluator_seed,
        "Gate-A evaluator seed",
        expected_identity=expected_identity,
        expected_state=expected_state,
    )
    tasks = []
    for family in learnable_module.FAMILY_SPECS_V2:
        for split in learnable_module.SPLITS_V2:
            for ordinal in range(1, family.count(split) + 1):
                tasks.append(learnable_module._build_task(secret, family, split, ordinal))
    return learnable_module.LearnableCorpusV2(tasks, secret)


def build_gate_a_runtime_manifest(config: DockerSandboxConfig) -> Dict[str, Any]:
    """Return the closed, path-free Docker contract bound by the freeze."""

    config.validate()
    manifest: Dict[str, Any] = {
        "schema_version": "egv-learnable-production-runtime-v2",
        "backend": "docker-enforced-v1",
        "image_ref": config.image_ref,
        "pinned_image_id": config.pinned_image_id,
        "candidate_contract": "pure-return-v1",
        "network": "none",
        "read_only_root": True,
        "cap_drop": "ALL",
        "no_new_privileges": True,
        "user": config.user,
        "pids_limit": config.pids_limit,
        "memory_limit": config.memory_limit,
        "tmpfs_size": config.tmpfs_size,
        "timeout_seconds": config.timeout_seconds,
        "output_limit": config.output_limit,
        "python_source_loading": "fresh-empty-pycache-prefix-v1",
        "python_bytecode_writes_disabled": True,
        "source_inventory_scope": "clean-pre-post-bytes-no-hostile-concurrent-mutation-claim",
    }
    manifest["manifest_digest"] = digest_for(manifest)
    return manifest


def _issuer_worker(
    evaluator_seed: str,
    controller_private_key: str,
    controller_public_key: str,
    freeze_private_key: str,
    freeze_public_key: str,
    workspace: str,
    package_root: str,
    source_commit: str,
    config_record: Mapping[str, Any],
    identity_record: Mapping[str, Any],
) -> None:
    identities = _validated_identity_record(identity_record)
    _assert_spawn_bytecode_boundary(workspace, "issuer", identities)

    from .blind_solver import BLIND_SOLVER_VERSION
    from .learnable_freeze import build_blind_solver_manifest, build_corpus_freeze_bundle

    _assert_loaded_source_root(package_root, identities["package_root"])
    _validate_private_layout(workspace, identities)
    paths = _private_paths(workspace)
    source_manifest = build_evaluator_source_manifest(
        package_root,
        source_commit,
        expected_root_identity=identities["package_root"],
    )
    evaluator_revision = source_manifest["manifest_digest"]
    plan = build_gate_a_plan()
    config = _config_from_record(config_record)
    runtime = build_gate_a_runtime_manifest(config)
    corpus = _generate_corpus_from_validated_seed(
        evaluator_seed,
        identities["evaluator_seed"],
        identities["evaluator_seed_state"],
    )
    public_manifest = corpus.public_manifest()
    private_manifest = corpus.private_manifest()
    blind_source_path = Path(package_root) / "egv" / "evaluation" / "blind_solver.py"
    if blind_source_path.is_symlink() or not blind_source_path.is_file():
        raise EGVError("Gate-A blind solver source is unavailable")
    blind_source = _read_stable_regular_bytes(blind_source_path, "Gate-A blind solver source")
    blind_manifest = build_blind_solver_manifest(blind_source, solver_version=BLIND_SOLVER_VERSION)
    controller_signer = _validated_external_signer(
        controller_private_key,
        controller_public_key,
        "Gate-A controller",
        private_identity=identities["controller_private"],
        public_identity=identities["controller_public"],
        private_state=identities["controller_private_state"],
        public_state=identities["controller_public_state"],
    )
    freeze_signer = _validated_external_signer(
        freeze_private_key,
        freeze_public_key,
        "Gate-A freeze authority",
        private_identity=identities["freeze_private"],
        public_identity=identities["freeze_public"],
        private_state=identities["freeze_private_state"],
        public_state=identities["freeze_public_state"],
    )
    _require_distinct_authority_keys(controller_signer.public_key, freeze_signer.public_key)
    bundle = build_corpus_freeze_bundle(
        public_manifest=public_manifest,
        private_manifest=private_manifest,
        runtime_manifest=runtime,
        blind_solver_manifest=blind_manifest,
        blind_solver_source=blind_source,
        blind_solver_version=BLIND_SOLVER_VERSION,
        evaluator_revision=evaluator_revision,
        controller_public_key=controller_signer.public_key,
        freeze_signer=freeze_signer,
    )
    issuer_core = {
        "schema_version": GATE_A_ISSUER_VERSION,
        "source_manifest_digest": source_manifest["manifest_digest"],
        "source_commit": source_commit,
        "evaluator_revision": evaluator_revision,
        "plan_digest": plan["plan_digest"],
        "public_manifest_digest": public_manifest["manifest_digest"],
        "private_manifest_digest": private_manifest["manifest_digest"],
        "runtime_manifest_digest": runtime["manifest_digest"],
        "blind_solver_manifest_digest": blind_manifest["manifest_digest"],
        "blind_solver_source_digest": blind_manifest["source_digest"],
        "controller_key_id": controller_signer.key_id,
        "freeze_signer_key_id": freeze_signer.key_id,
        "freeze_bundle_digest": bundle["bundle_digest"],
    }
    issuer_record = {**issuer_core, "issuer_record_digest": digest_for(issuer_core)}
    for destination, value in (
        (paths["source_manifest"], source_manifest),
        (paths["plan"], plan),
        (paths["freeze"], bundle),
        (paths["issuer_record"], issuer_record),
    ):
        _atomic_write_new(
            destination,
            value,
            private=True,
            expected_parent_identity=identities["issuer"],
        )


def _issuer_entry(*args: Any) -> None:
    workspace = str(args[5])
    try:
        _issuer_worker(*args)
    except BaseException as exc:
        try:
            failures_identity = _validated_identity_record(args[-1])["failures"]
        except BaseException:
            failures_identity = None
        _failure_record(
            workspace,
            "issuer",
            exc,
            failures_identity=failures_identity,
        )
        raise SystemExit(1)


def _gate_a_acceptance(
    *,
    oracle_pass_count: int,
    blind_pass_count: int,
    starter_wrong_output_count: int,
    infrastructure_loss_count: int,
    receipt_count: int,
    receipt_chain_valid: bool,
    delivery_count: int,
    delivery_chain_valid: bool,
    receipt_delivery_bijection: bool,
    sandbox_preflight_passed: bool,
    sandbox_negative_controls_all_passed: bool,
    sandbox_postflight_passed: bool,
) -> bool:
    return bool(
        oracle_pass_count == ORACLE_EVALUATION_COUNT
        and blind_pass_count == BLIND_EVALUATION_COUNT
        and starter_wrong_output_count == STARTER_EVALUATION_COUNT
        and infrastructure_loss_count == 0
        and receipt_count == EXPECTED_RECEIPT_COUNT
        and receipt_chain_valid
        and delivery_count == EXPECTED_RECEIPT_COUNT
        and delivery_chain_valid
        and receipt_delivery_bijection
        and sandbox_preflight_passed
        and sandbox_negative_controls_all_passed
        and sandbox_postflight_passed
    )


def _validate_issuer_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    expected_fields = set(_ISSUER_CORE_FIELDS) | {"issuer_record_digest"}
    issuer = _closed(record, expected_fields, "Gate-A issuer record")
    if issuer.get("schema_version") != GATE_A_ISSUER_VERSION:
        raise ValueError("unsupported Gate-A issuer record")
    for field in _ISSUER_CORE_FIELDS:
        if field.endswith("_digest") or field in {"evaluator_revision", "freeze_bundle_digest"}:
            validate_sha256(issuer.get(field), field)
    core = {field: issuer[field] for field in _ISSUER_CORE_FIELDS}
    if issuer.get("issuer_record_digest") != digest_for(core):
        raise ValueError("Gate-A issuer record digest mismatch")
    return dict(issuer)


def _candidate_sources(corpus: Any) -> Tuple[Dict[str, bytes], Dict[str, bytes], Dict[str, bytes], Dict[str, Any]]:
    """Build every candidate source before any hidden runner is constructed."""

    from .blind_solver import compile_blind_source
    from .learnable import build_reference_source, validate_model_visible_package

    oracle_sources: Dict[str, bytes] = {}
    blind_sources: Dict[str, bytes] = {}
    starter_sources: Dict[str, bytes] = {}
    entries = []
    for task in corpus.tasks:
        package = validate_model_visible_package(corpus.model_visible_package(task.task_id))
        package_bytes = canonical_bytes(package)
        source = build_reference_source(package["task"])
        oracle_sources[task.task_id] = source
        entries.append(
            {
                "kind": "oracle",
                "public_task_id": task.task_id,
                "package_digest": package["package_digest"],
                "source_digest": digest_bytes(source),
            }
        )
        if task.split == "blind":
            blind = compile_blind_source(package_bytes)
            starter = str(package["task"]["starter_source"]).encode("utf-8")
            blind_sources[task.task_id] = blind
            starter_sources[task.task_id] = starter
            entries.append(
                {
                    "kind": "blind",
                    "public_task_id": task.task_id,
                    "package_digest": package["package_digest"],
                    "source_digest": digest_bytes(blind),
                }
            )
            entries.append(
                {
                    "kind": "starter",
                    "public_task_id": task.task_id,
                    "package_digest": package["package_digest"],
                    "source_digest": digest_bytes(starter),
                }
            )
    inventory_core = {
        "schema_version": "egv-gate-a-candidate-source-inventory-v1",
        "oracle_source_count": len(oracle_sources),
        "blind_source_count": len(blind_sources),
        "starter_source_count": len(starter_sources),
        "entries": entries,
    }
    inventory = {**inventory_core, "inventory_digest": digest_for(inventory_core)}
    if (
        len(oracle_sources) != ORACLE_TASK_COUNT
        or len(blind_sources) != BLIND_TASK_COUNT
        or len(starter_sources) != STARTER_TASK_COUNT
    ):
        raise EGVError("Gate-A candidate-source inventory differs from the frozen plan")
    return oracle_sources, blind_sources, starter_sources, inventory


def _expected_evaluation_bindings(
    corpus: Any,
    addresses: Sequence[Any],
    oracle_sources: Mapping[str, bytes],
    blind_sources: Mapping[str, bytes],
    starter_sources: Mapping[str, bytes],
) -> Dict[str, Any]:
    """Derive the ordered source/task/case identity for all 394 evaluations."""

    from .learnable_runtime import build_learnable_candidate_id

    bindings = []

    def extend(kind: str, tasks: Sequence[Any], sources: Mapping[str, bytes], case_limit: Optional[int]) -> None:
        for task in tasks:
            source_digest = digest_bytes(sources[task.task_id])
            selected = sorted(
                (address for address in addresses if address.public_task_id == task.task_id),
                key=lambda address: address.case_ordinal,
            )
            if case_limit is not None:
                selected = selected[:case_limit]
            cases = {case.ordinal: case for case in task.hidden_cases}
            for address in selected:
                case = cases[address.case_ordinal]
                bindings.append(
                    {
                        "candidate_kind": kind,
                        "candidate_id": build_learnable_candidate_id(
                            candidate_kind=kind,
                            public_task_id=task.task_id,
                            source_digest=source_digest,
                            case_ordinal=address.case_ordinal,
                        ),
                        "public_task_id": task.task_id,
                        "task_id": address.controller_task_id,
                        "case_ordinal": address.case_ordinal,
                        "candidate_artifact_digest": source_digest,
                        "input_digest": digest_for(case.input_value),
                        "task_family": task.family_id,
                        "normalized_public_locus": "module:main",
                        "public_rule_id": address.public_rule_id,
                    }
                )

    extend("oracle", corpus.tasks, oracle_sources, None)
    blind_tasks = corpus.split("blind")
    extend("blind", blind_tasks, blind_sources, None)
    extend("starter", blind_tasks, starter_sources, 1)
    if (
        len(bindings) != TOTAL_EVALUATION_COUNT
        or len({binding["candidate_id"] for binding in bindings}) != len(bindings)
        or sum(binding["candidate_kind"] == "oracle" for binding in bindings) != ORACLE_EVALUATION_COUNT
        or sum(binding["candidate_kind"] == "blind" for binding in bindings) != BLIND_EVALUATION_COUNT
        or sum(binding["candidate_kind"] == "starter" for binding in bindings) != STARTER_EVALUATION_COUNT
    ):
        raise EGVError("Gate-A exact evaluation bindings differ from the frozen plan")
    core = {
        "schema_version": GATE_A_BINDING_VERSION,
        "evaluation_count": len(bindings),
        "bindings": bindings,
    }
    return {**core, "evaluation_binding_digest": digest_for(core)}


def _evaluator_worker(
    evaluator_seed: str,
    controller_private_key: str,
    controller_public_key: str,
    freeze_public_key: str,
    workspace: str,
    package_root: str,
    source_commit: str,
    config_record: Mapping[str, Any],
    identity_record: Mapping[str, Any],
) -> None:
    identities = _validated_identity_record(identity_record)
    _assert_spawn_bytecode_boundary(workspace, "evaluator", identities)

    from .boundary import mark_evaluator_process

    # This irreversible role marker precedes evaluator construction and any
    # writer-capable import in the child.
    mark_evaluator_process()

    from .artifacts import ContentAddressedArtifactStore
    from .authority import AuthorityBroker, AuthorityPolicy, DockerEnforcedRuntime
    from .blind_solver import BLIND_SOLVER_VERSION
    from .controller import EvaluatorController
    from .diagnostics import Diagnostic
    from .learnable_freeze import (
        build_blind_solver_manifest,
        validate_corpus_freeze_bundle,
    )
    from .learnable_runtime import _evaluate_task_source, build_learnable_hidden_runner
    from .sandbox import DockerCandidateSandbox

    _assert_loaded_source_root(package_root, identities["package_root"])
    _validate_private_layout(workspace, identities)
    paths = _private_paths(workspace)
    source_manifest = build_evaluator_source_manifest(
        package_root,
        source_commit,
        expected_root_identity=identities["package_root"],
    )
    issued_source = validate_evaluator_source_manifest(
        _read_canonical_record(paths["source_manifest"], "Gate-A issued source manifest")
    )
    if source_manifest != issued_source:
        raise EGVError("Gate-A evaluator source inventory changed after issuance")
    plan = validate_gate_a_plan(_read_canonical_record(paths["plan"], "Gate-A evaluation plan"))
    issuer = _validate_issuer_record(_read_canonical_record(paths["issuer_record"], "Gate-A issuer record"))
    freeze_bundle = _read_canonical_record(paths["freeze"], "Gate-A freeze bundle")

    config = _config_from_record(config_record)
    runtime = build_gate_a_runtime_manifest(config)
    corpus = _generate_corpus_from_validated_seed(
        evaluator_seed,
        identities["evaluator_seed"],
        identities["evaluator_seed_state"],
    )
    public_manifest = corpus.public_manifest()
    private_manifest = corpus.private_manifest()
    blind_source_path = Path(package_root) / "egv" / "evaluation" / "blind_solver.py"
    blind_source = _read_stable_regular_bytes(blind_source_path, "Gate-A blind solver source")
    blind_manifest = build_blind_solver_manifest(blind_source, solver_version=BLIND_SOLVER_VERSION)
    controller_signer = _validated_external_signer(
        controller_private_key,
        controller_public_key,
        "Gate-A controller",
        private_identity=identities["controller_private"],
        public_identity=identities["controller_public"],
        private_state=identities["controller_private_state"],
        public_state=identities["controller_public_state"],
    )
    freeze_public = _read_public_key(
        freeze_public_key,
        "Gate-A freeze authority public key",
        expected_identity=identities["freeze_public"],
        expected_state=identities["freeze_public_state"],
    )
    _require_distinct_authority_keys(controller_signer.public_key, freeze_public)
    validate_corpus_freeze_bundle(
        freeze_bundle,
        public_manifest=public_manifest,
        private_manifest=private_manifest,
        runtime_manifest=runtime,
        blind_solver_manifest=blind_manifest,
        blind_solver_source=blind_source,
        blind_solver_version=BLIND_SOLVER_VERSION,
        evaluator_revision=source_manifest["manifest_digest"],
        controller_public_key=controller_signer.public_key,
        freeze_public_key=freeze_public,
    )
    exact_issuer = {
        "source_manifest_digest": source_manifest["manifest_digest"],
        "source_commit": source_commit,
        "evaluator_revision": source_manifest["manifest_digest"],
        "plan_digest": plan["plan_digest"],
        "public_manifest_digest": public_manifest["manifest_digest"],
        "private_manifest_digest": private_manifest["manifest_digest"],
        "runtime_manifest_digest": runtime["manifest_digest"],
        "blind_solver_manifest_digest": blind_manifest["manifest_digest"],
        "blind_solver_source_digest": blind_manifest["source_digest"],
        "controller_key_id": controller_signer.key_id,
        "freeze_signer_key_id": key_id_for_public_key(freeze_public),
        "freeze_bundle_digest": freeze_bundle["bundle_digest"],
    }
    if any(issuer[field] != value for field, value in exact_issuer.items()):
        raise EGVError("Gate-A issuer record differs from evaluator-derived bindings")

    # All 170 oracle, 18 independent-blind, and 18 starter sources exist before
    # hidden runner construction.  The blind compiler receives canonical
    # model-visible package bytes only.
    oracle_sources, blind_sources, starter_sources, candidate_inventory = _candidate_sources(corpus)
    _atomic_write_new(
        paths["candidate_inventory"],
        candidate_inventory,
        private=True,
        expected_parent_identity=identities["evaluator"],
    )
    hidden_runner, addresses = build_learnable_hidden_runner(
        corpus.tasks, evaluator_revision=source_manifest["manifest_digest"]
    )
    evaluation_bindings = _expected_evaluation_bindings(
        corpus,
        addresses,
        oracle_sources,
        blind_sources,
        starter_sources,
    )
    _atomic_write_new(
        paths["evaluation_bindings"],
        evaluation_bindings,
        private=True,
        expected_parent_identity=identities["evaluator"],
    )

    sandbox = DockerCandidateSandbox(paths["sandbox"], config=config)
    sandbox.validate_production_integrity()
    preflight = sandbox.runtime_report()
    preflight_digest = digest_for(preflight)
    negative_controls = sandbox.negative_controls(paths["private"] / "hidden-never-mounted.json")
    negative_record = {
        "schema_version": "egv-gate-a-sandbox-negative-controls-v1",
        "results": negative_controls,
        "evidence_digest": digest_for(sandbox.last_negative_control_evidence),
    }
    negative_record["record_digest"] = digest_for(negative_record)
    _atomic_write_new(
        paths["evaluator"] / "sandbox-negative-controls.json",
        negative_record,
        private=True,
        expected_parent_identity=identities["evaluator"],
    )

    policy = AuthorityPolicy.candidate_execution()
    protocol_digest = _protocol_digest(
        freeze_bundle_digest=freeze_bundle["bundle_digest"],
        plan_digest=plan["plan_digest"],
        source_manifest_digest=source_manifest["manifest_digest"],
        policy_digest=policy.digest,
    )
    journal = StreamingReceiptJournal(
        paths["receipts"],
        controller_signer.public_key,
        expected_parent_identity=identities["evaluator"],
    )
    deliveries = DurableDeliveryJournal(paths["deliveries"], expected_parent_identity=identities["evaluator"])
    controller = EvaluatorController(
        sandbox=sandbox,
        hidden_runner=hidden_runner,
        broker=AuthorityBroker(DockerEnforcedRuntime(config), policy),
        signer=controller_signer,
        journal=journal,
        ingest=deliveries.append,
        campaign_id=GATE_A_CAMPAIGN_ID,
        protocol_digest=protocol_digest,
        policy_digest=policy.digest,
        artifact_store=ContentAddressedArtifactStore(paths["artifacts"]),
    )

    oracle_results = []
    blind_results = []
    starter_results = []
    for task in corpus.tasks:
        oracle_results.extend(
            _evaluate_task_source(
                controller,
                task,
                addresses,
                oracle_sources[task.task_id],
                candidate_kind="oracle",
            )
        )
    for task in corpus.split("blind"):
        blind_results.extend(
            _evaluate_task_source(
                controller,
                task,
                addresses,
                blind_sources[task.task_id],
                candidate_kind="blind",
            )
        )
    for task in corpus.split("blind"):
        starter_results.extend(
            _evaluate_task_source(
                controller,
                task,
                addresses,
                starter_sources[task.task_id],
                candidate_kind="starter",
                case_limit=1,
            )
        )

    sandbox.validate_production_integrity()
    receipts, journal_evidence = _fresh_receipt_replay(paths["receipts"], controller_signer.public_key)
    delivery_evidence = DurableDeliveryJournal.replay(paths["deliveries"], receipts)
    replayed_counts = _replay_receipt_outcomes(
        receipts,
        controller_public_key=controller_signer.public_key,
        protocol_digest=protocol_digest,
        policy_digest=policy.digest,
        evaluator_revision=source_manifest["manifest_digest"],
        expected_bindings=evaluation_bindings,
    )
    all_results = oracle_results + blind_results + starter_results
    oracle_pass_count = sum(result.diagnostic_enum == Diagnostic.PASS.value for result in oracle_results)
    blind_pass_count = sum(result.diagnostic_enum == Diagnostic.PASS.value for result in blind_results)
    starter_wrong_output_count = sum(
        result.diagnostic_enum == Diagnostic.WRONG_OUTPUT.value for result in starter_results
    )
    infrastructure_loss_count = sum(bool(result.infrastructure_loss) for result in all_results)
    direct_counts = {
        "oracle_evaluation_count": len(oracle_results),
        "oracle_pass_count": oracle_pass_count,
        "blind_evaluation_count": len(blind_results),
        "blind_pass_count": blind_pass_count,
        "starter_evaluation_count": len(starter_results),
        "starter_wrong_output_count": starter_wrong_output_count,
        "infrastructure_loss_count": infrastructure_loss_count,
    }
    if replayed_counts != direct_counts:
        raise EGVError("Gate-A evaluator results differ from signed receipt replay")
    production_acceptance = _gate_a_acceptance(
        oracle_pass_count=oracle_pass_count,
        blind_pass_count=blind_pass_count,
        starter_wrong_output_count=starter_wrong_output_count,
        infrastructure_loss_count=infrastructure_loss_count,
        receipt_count=journal_evidence["count"],
        receipt_chain_valid=journal_evidence["chain_valid"],
        delivery_count=delivery_evidence["count"],
        delivery_chain_valid=delivery_evidence["chain_valid"],
        receipt_delivery_bijection=delivery_evidence["receipt_delivery_bijection"],
        sandbox_preflight_passed=True,
        sandbox_negative_controls_all_passed=bool(negative_controls) and all(negative_controls.values()),
        sandbox_postflight_passed=True,
    )
    report_core = {
        "schema_version": GATE_A_REPORT_VERSION,
        "evidence_scope": "production-corpus-validation-no-model-quality-claim",
        "freeze_bundle_digest": freeze_bundle["bundle_digest"],
        "plan_digest": plan["plan_digest"],
        "protocol_digest": protocol_digest,
        "policy_digest": policy.digest,
        "public_manifest_digest": public_manifest["manifest_digest"],
        "private_manifest_digest": private_manifest["manifest_digest"],
        "runtime_manifest_digest": runtime["manifest_digest"],
        "blind_solver_manifest_digest": blind_manifest["manifest_digest"],
        "blind_solver_source_digest": blind_manifest["source_digest"],
        "source_manifest_digest": source_manifest["manifest_digest"],
        "source_commit": source_commit,
        "candidate_source_inventory_digest": candidate_inventory["inventory_digest"],
        "evaluation_binding_digest": evaluation_bindings["evaluation_binding_digest"],
        "controller_key_id": controller_signer.key_id,
        "freeze_signer_key_id": key_id_for_public_key(freeze_public),
        "oracle_task_count": len(corpus.tasks),
        "oracle_evaluation_count": len(oracle_results),
        "oracle_pass_count": oracle_pass_count,
        "blind_task_count": len(corpus.split("blind")),
        "blind_evaluation_count": len(blind_results),
        "blind_pass_count": blind_pass_count,
        "starter_task_count": len(corpus.split("blind")),
        "starter_evaluation_count": len(starter_results),
        "starter_wrong_output_count": starter_wrong_output_count,
        "total_evaluation_count": len(all_results),
        "infrastructure_loss_count": infrastructure_loss_count,
        "expected_receipt_count": EXPECTED_RECEIPT_COUNT,
        "receipt_count": journal_evidence["count"],
        "receipt_chain_head": journal_evidence["head"],
        "receipt_chain_valid": journal_evidence["chain_valid"],
        "receipt_journal_digest": journal_evidence["journal_digest"],
        "delivery_count": delivery_evidence["count"],
        "delivery_chain_head": delivery_evidence["head"],
        "delivery_chain_valid": delivery_evidence["chain_valid"],
        "delivery_journal_digest": delivery_evidence["journal_digest"],
        "receipt_delivery_bijection": delivery_evidence["receipt_delivery_bijection"],
        "sandbox_preflight_passed": True,
        "sandbox_preflight_digest": preflight_digest,
        "sandbox_negative_control_count": len(negative_controls),
        "sandbox_negative_controls_all_passed": bool(negative_controls) and all(negative_controls.values()),
        "sandbox_negative_controls_digest": digest_for(negative_controls),
        "sandbox_postflight_passed": True,
        "production_acceptance": production_acceptance,
        "task_bearing_model_requests_permitted": False,
        "model_quality_claimed": False,
        "cross_node_transport_exercised": False,
        "hidden_values_exposed": False,
        "private_paths_exposed": False,
    }
    report = _sign_closed_record(
        report_core,
        controller_signer,
        digest_field="report_digest",
        signature_field="controller_signature",
    )
    _atomic_write_new(
        paths["report"],
        report,
        private=True,
        expected_parent_identity=identities["evaluator"],
    )


def _evaluator_entry(*args: Any) -> None:
    workspace = str(args[4])
    try:
        _evaluator_worker(*args)
    except BaseException as exc:
        try:
            failures_identity = _validated_identity_record(args[-1])["failures"]
        except BaseException:
            failures_identity = None
        _failure_record(
            workspace,
            "evaluator",
            exc,
            failures_identity=failures_identity,
        )
        raise SystemExit(1)


def _validate_negative_control_record(record: Mapping[str, Any]) -> Dict[str, Any]:
    fields = {"schema_version", "results", "evidence_digest", "record_digest"}
    closed = _closed(record, fields, "Gate-A sandbox negative-control record")
    if closed.get("schema_version") != "egv-gate-a-sandbox-negative-controls-v1":
        raise ValueError("unsupported Gate-A sandbox negative-control record")
    results = closed.get("results")
    if not isinstance(results, Mapping) or not results or any(type(value) is not bool for value in results.values()):
        raise ValueError("Gate-A sandbox negative-control results are invalid")
    validate_sha256(closed.get("evidence_digest"), "sandbox negative-control evidence digest")
    core = {key: closed[key] for key in closed if key != "record_digest"}
    if closed.get("record_digest") != digest_for(core):
        raise ValueError("Gate-A sandbox negative-control record digest mismatch")
    return dict(closed)


def _replay_receipt_outcomes(
    receipts: Sequence[Mapping[str, Any]],
    *,
    controller_public_key: Any,
    protocol_digest: str,
    policy_digest: str,
    evaluator_revision: str,
    expected_bindings: Mapping[str, Any],
) -> Dict[str, Any]:
    if len(receipts) != EXPECTED_RECEIPT_COUNT or len(receipts) % 3:
        raise ReceiptVerificationError("Gate-A receipt count differs from the frozen plan")
    bindings_record = _closed(
        expected_bindings,
        {
            "schema_version",
            "evaluation_count",
            "bindings",
            "evaluation_binding_digest",
        },
        "Gate-A evaluation bindings",
    )
    bindings = bindings_record.get("bindings")
    bindings_core = {key: bindings_record[key] for key in ("schema_version", "evaluation_count", "bindings")}
    if (
        bindings_record.get("schema_version") != GATE_A_BINDING_VERSION
        or bindings_record.get("evaluation_count") != TOTAL_EVALUATION_COUNT
        or not isinstance(bindings, list)
        or len(bindings) != TOTAL_EVALUATION_COUNT
        or bindings_record.get("evaluation_binding_digest") != digest_for(bindings_core)
    ):
        raise ReceiptVerificationError("Gate-A expected evaluation bindings are invalid")
    expected_evaluator_digest = digest_for(evaluator_revision)
    expected_key_id = key_id_for_public_key(controller_public_key)
    counts = {
        "oracle_evaluation_count": 0,
        "oracle_pass_count": 0,
        "blind_evaluation_count": 0,
        "blind_pass_count": 0,
        "starter_evaluation_count": 0,
        "starter_wrong_output_count": 0,
        "infrastructure_loss_count": 0,
    }
    candidate_ids = set()
    for binding_index, offset in enumerate(range(0, len(receipts), 3)):
        authority, verdict, effect = receipts[offset : offset + 3]
        expected = bindings[binding_index]
        if not isinstance(expected, Mapping):
            raise ReceiptVerificationError("Gate-A evaluation binding is invalid")
        if [authority.get("receipt_type"), verdict.get("receipt_type"), effect.get("receipt_type")] != [
            "AUTHORITY",
            "VERDICT",
            "EFFECT",
        ]:
            raise ReceiptVerificationError("Gate-A receipt triples are not AUTHORITY/VERDICT/EFFECT")
        candidate_id = authority.get("candidate_id")
        if (
            not isinstance(candidate_id, str)
            or candidate_id != expected.get("candidate_id")
            or candidate_id in candidate_ids
            or any(receipt.get("candidate_id") != candidate_id for receipt in (verdict, effect))
            or any(receipt.get("task_id") != authority.get("task_id") for receipt in (verdict, effect))
        ):
            raise ReceiptVerificationError("Gate-A receipt candidate grouping is invalid")
        candidate_ids.add(candidate_id)
        kind = candidate_id.split("-", 1)[0]
        if kind != expected.get("candidate_kind"):
            raise ReceiptVerificationError("Gate-A receipt candidate kind differs from exact plan")
        if kind not in {"oracle", "blind", "starter"}:
            raise ReceiptVerificationError("Gate-A receipt has an unknown candidate kind")
        for receipt in (authority, verdict, effect):
            if (
                receipt.get("campaign_id") != GATE_A_CAMPAIGN_ID
                or receipt.get("run_id") != "run-evaluation"
                or receipt.get("protocol_digest") != protocol_digest
                or receipt.get("policy_digest") != policy_digest
                or receipt.get("evaluator_digest") != expected_evaluator_digest
                or receipt.get("signing_key_id") != expected_key_id
                or receipt.get("task_id") != expected.get("task_id")
                or receipt.get("candidate_artifact_digest") != expected.get("candidate_artifact_digest")
                or receipt.get("task_family") != expected.get("task_family")
                or receipt.get("normalized_public_locus") != expected.get("normalized_public_locus")
                or receipt.get("public_rule_id") != expected.get("public_rule_id")
                or receipt.get("infrastructure_incident_id") is not None
                or receipt.get("failure_family_root") is not None
            ):
                raise ReceiptVerificationError("Gate-A receipt differs from frozen campaign bindings")
        if (
            authority.get("request_id") != "request-authority-{}".format(candidate_id)
            or authority.get("idempotency_key") != "authority:{}".format(candidate_id)
            or verdict.get("request_id") != "request-verdict-{}".format(candidate_id)
            or verdict.get("idempotency_key") != "verdict:{}".format(candidate_id)
            or verdict.get("input_digest") != expected.get("input_digest")
            or effect.get("request_id") != "request-effect-{}".format(candidate_id)
            or effect.get("idempotency_key") != "effect:{}".format(candidate_id)
            or effect.get("normalized_action_hash")
            != digest_for({"action": "execute_candidate", "locus": "module:main"})
        ):
            raise ReceiptVerificationError("Gate-A receipt triple differs from exact source/task/case binding")
        if authority.get("decision") != "ALLOW" or effect.get("decision") != "ALLOW":
            raise ReceiptVerificationError("Gate-A authority/effect receipt did not allow the bounded evaluation")
        if kind in {"oracle", "blind"}:
            counts[kind + "_evaluation_count"] += 1
            if verdict.get("decision") == "PASS" and verdict.get("diagnostic_enum") == "PASS":
                counts[kind + "_pass_count"] += 1
        else:
            counts["starter_evaluation_count"] += 1
            if verdict.get("decision") == "FAIL" and verdict.get("diagnostic_enum") == "WRONG_OUTPUT":
                counts["starter_wrong_output_count"] += 1
        if verdict.get("diagnostic_enum") == "INTERNAL_ERROR" or effect.get("diagnostic_enum") == "INTERNAL_ERROR":
            counts["infrastructure_loss_count"] += 1
    if (
        counts["oracle_evaluation_count"] != ORACLE_EVALUATION_COUNT
        or counts["blind_evaluation_count"] != BLIND_EVALUATION_COUNT
        or counts["starter_evaluation_count"] != STARTER_EVALUATION_COUNT
    ):
        raise ReceiptVerificationError("Gate-A receipt candidate allocation differs from the frozen plan")
    return counts


def _privacy_scan(
    subject: Mapping[str, Any],
    *,
    forbidden_tokens: Sequence[bytes],
) -> Dict[str, Any]:
    """Scan canonical public bytes for exact private material and path/topology patterns."""

    # Include the canonical trailing newline: this is byte-for-byte the form
    # emitted by _atomic_write_new, not merely a semantic object scan.
    payload = _canonical_record_bytes(subject)
    unique_tokens = {token for token in forbidden_tokens if isinstance(token, bytes) and len(token) >= 8}
    token_findings = sum(token in payload for token in unique_tokens)
    lowered = payload.lower()
    pattern_findings = sum(re.search(pattern, lowered) is not None for pattern in _PRIVACY_PATTERNS)
    finding_count = token_findings + pattern_findings
    core = {
        "schema_version": GATE_A_PRIVACY_VERSION,
        "policy_digest": digest_for([pattern.decode("ascii") for pattern in _PRIVACY_PATTERNS]),
        "subject_digest": digest_bytes(payload),
        "prohibited_token_count": len(unique_tokens),
        "prohibited_pattern_count": len(_PRIVACY_PATTERNS),
        "finding_count": finding_count,
        "scan_passed": finding_count == 0,
        "hidden_values_exposed": token_findings > 0,
        "private_paths_or_topology_exposed": pattern_findings > 0,
        "credential_material_exposed": pattern_findings > 0 and b"private key" in lowered,
    }
    record = {**core, "privacy_receipt_digest": digest_for(core)}
    if finding_count:
        raise EGVError("Gate-A public privacy scan found prohibited material")
    return record


def _private_scan_tokens(
    corpus: Any,
    *,
    evaluator_seed: str,
    controller_private_key: Optional[str],
    freeze_private_key: str,
    workspace: str,
    package_root: str,
    identities: Mapping[str, Tuple[int, ...]],
) -> Tuple[bytes, ...]:
    tokens = []
    sensitive_paths = (
        evaluator_seed,
        freeze_private_key,
        workspace,
        package_root,
    )
    if controller_private_key is not None:
        sensitive_paths = (*sensitive_paths, controller_private_key)
    for value in sensitive_paths:
        encoded = str(Path(value).resolve(strict=False)).encode("utf-8")
        tokens.extend((encoded, canonical_json(str(Path(value).resolve(strict=False))).encode("utf-8")))
    private_sources = [
        (evaluator_seed, "evaluator_seed"),
        (freeze_private_key, "freeze_private"),
    ]
    if controller_private_key is not None:
        private_sources.append((controller_private_key, "controller_private"))
    for source, identity_name in private_sources:
        raw = _read_stable_regular_bytes(
            Path(source),
            "Gate-A private scan source",
            expected_identity=identities[identity_name],
            expected_state=identities[identity_name + "_state"],
        )
        tokens.extend((raw, raw.hex().encode("ascii"), _b64url(raw).encode("ascii")))
    for task in corpus.tasks:
        for case in task.hidden_cases:
            tokens.extend(
                (
                    canonical_bytes(case.input_value),
                    canonical_bytes(case.expected_output),
                    str(case.input_value["case_nonce"]).encode("utf-8"),
                )
            )
    return tuple(tokens)


def _resolved_path_scan_tokens(*paths: Union[str, Path]) -> Tuple[bytes, ...]:
    """Return exact path spellings that must never appear in public bytes."""

    tokens = []
    for value in paths:
        resolved = str(Path(value).resolve(strict=False))
        tokens.extend((resolved.encode("utf-8"), canonical_json(resolved).encode("utf-8")))
    return tuple(tokens)


def _verify_report(record: Mapping[str, Any], controller_public_key: Any) -> Dict[str, Any]:
    return _verify_closed_signature(
        record,
        controller_public_key,
        core_fields=_REPORT_CORE_FIELDS,
        digest_field="report_digest",
        signature_field="controller_signature",
        label="Gate-A evaluator report",
    )


def _verifier_worker(
    evaluator_seed: str,
    controller_public_key: str,
    freeze_private_key: str,
    freeze_public_key: str,
    workspace: str,
    package_root: str,
    source_commit: str,
    config_record: Mapping[str, Any],
    identity_record: Mapping[str, Any],
) -> Dict[str, Any]:
    identities = _validated_identity_record(identity_record)
    _assert_spawn_bytecode_boundary(workspace, "verifier", identities)

    from .authority import AuthorityPolicy
    from .blind_solver import BLIND_SOLVER_VERSION
    from .learnable_freeze import build_blind_solver_manifest, validate_corpus_freeze_bundle
    from .learnable_runtime import build_learnable_hidden_runner

    _assert_loaded_source_root(package_root, identities["package_root"])
    _validate_private_layout(workspace, identities)
    paths = _private_paths(workspace)
    source_manifest = build_evaluator_source_manifest(
        package_root,
        source_commit,
        expected_root_identity=identities["package_root"],
    )
    issued_source = validate_evaluator_source_manifest(
        _read_canonical_record(paths["source_manifest"], "Gate-A issued source manifest")
    )
    if source_manifest != issued_source:
        raise EGVError("Gate-A verifier source inventory changed after issuance")
    plan = validate_gate_a_plan(_read_canonical_record(paths["plan"], "Gate-A evaluation plan"))
    issuer = _validate_issuer_record(_read_canonical_record(paths["issuer_record"], "Gate-A issuer record"))
    freeze_bundle = _read_canonical_record(paths["freeze"], "Gate-A freeze bundle")
    config = _config_from_record(config_record)
    runtime = build_gate_a_runtime_manifest(config)
    corpus = _generate_corpus_from_validated_seed(
        evaluator_seed,
        identities["evaluator_seed"],
        identities["evaluator_seed_state"],
    )
    public_manifest = corpus.public_manifest()
    private_manifest = corpus.private_manifest()
    blind_source = _read_stable_regular_bytes(
        Path(package_root) / "egv" / "evaluation" / "blind_solver.py",
        "Gate-A blind solver source",
    )
    blind_manifest = build_blind_solver_manifest(blind_source, solver_version=BLIND_SOLVER_VERSION)
    controller_public = _read_public_key(
        controller_public_key,
        "Gate-A controller public key",
        expected_identity=identities["controller_public"],
        expected_state=identities["controller_public_state"],
    )
    freeze_signer = _validated_external_signer(
        freeze_private_key,
        freeze_public_key,
        "Gate-A freeze authority",
        private_identity=identities["freeze_private"],
        public_identity=identities["freeze_public"],
        private_state=identities["freeze_private_state"],
        public_state=identities["freeze_public_state"],
    )
    _require_distinct_authority_keys(controller_public, freeze_signer.public_key)
    validate_corpus_freeze_bundle(
        freeze_bundle,
        public_manifest=public_manifest,
        private_manifest=private_manifest,
        runtime_manifest=runtime,
        blind_solver_manifest=blind_manifest,
        blind_solver_source=blind_source,
        blind_solver_version=BLIND_SOLVER_VERSION,
        evaluator_revision=source_manifest["manifest_digest"],
        controller_public_key=controller_public,
        freeze_public_key=freeze_signer.public_key,
    )

    oracle_sources, blind_sources, starter_sources, candidate_inventory = _candidate_sources(corpus)
    _unused_runner, addresses = build_learnable_hidden_runner(
        corpus.tasks, evaluator_revision=source_manifest["manifest_digest"]
    )
    evaluation_bindings = _expected_evaluation_bindings(
        corpus,
        addresses,
        oracle_sources,
        blind_sources,
        starter_sources,
    )
    private_candidate_inventory = _read_canonical_record(
        paths["candidate_inventory"], "Gate-A candidate-source inventory"
    )
    if private_candidate_inventory != candidate_inventory:
        raise EGVError("Gate-A candidate-source inventory changed before fresh verification")
    private_evaluation_bindings = _read_canonical_record(
        paths["evaluation_bindings"], "Gate-A exact evaluation bindings"
    )
    if private_evaluation_bindings != evaluation_bindings:
        raise EGVError("Gate-A exact evaluation bindings changed before fresh verification")
    report = _verify_report(_read_canonical_record(paths["report"], "Gate-A evaluator report"), controller_public)
    negative = _validate_negative_control_record(
        _read_canonical_record(
            paths["evaluator"] / "sandbox-negative-controls.json",
            "Gate-A sandbox negative-control evidence",
        )
    )
    policy = AuthorityPolicy.candidate_execution()
    protocol_digest = _protocol_digest(
        freeze_bundle_digest=freeze_bundle["bundle_digest"],
        plan_digest=plan["plan_digest"],
        source_manifest_digest=source_manifest["manifest_digest"],
        policy_digest=policy.digest,
    )
    receipts, journal_evidence = _fresh_receipt_replay(paths["receipts"], controller_public)
    delivery_evidence = DurableDeliveryJournal.replay(paths["deliveries"], receipts)
    outcome_counts = _replay_receipt_outcomes(
        receipts,
        controller_public_key=controller_public,
        protocol_digest=protocol_digest,
        policy_digest=policy.digest,
        evaluator_revision=source_manifest["manifest_digest"],
        expected_bindings=evaluation_bindings,
    )
    exact_bindings = {
        "freeze_bundle_digest": freeze_bundle["bundle_digest"],
        "plan_digest": plan["plan_digest"],
        "protocol_digest": protocol_digest,
        "policy_digest": policy.digest,
        "public_manifest_digest": public_manifest["manifest_digest"],
        "private_manifest_digest": private_manifest["manifest_digest"],
        "runtime_manifest_digest": runtime["manifest_digest"],
        "blind_solver_manifest_digest": blind_manifest["manifest_digest"],
        "blind_solver_source_digest": blind_manifest["source_digest"],
        "source_manifest_digest": source_manifest["manifest_digest"],
        "source_commit": source_commit,
        "candidate_source_inventory_digest": candidate_inventory["inventory_digest"],
        "evaluation_binding_digest": evaluation_bindings["evaluation_binding_digest"],
        "controller_key_id": key_id_for_public_key(controller_public),
        "freeze_signer_key_id": freeze_signer.key_id,
        "oracle_task_count": ORACLE_TASK_COUNT,
        "blind_task_count": BLIND_TASK_COUNT,
        "starter_task_count": STARTER_TASK_COUNT,
        "total_evaluation_count": TOTAL_EVALUATION_COUNT,
        "expected_receipt_count": EXPECTED_RECEIPT_COUNT,
        "receipt_count": journal_evidence["count"],
        "receipt_chain_head": journal_evidence["head"],
        "receipt_chain_valid": journal_evidence["chain_valid"],
        "receipt_journal_digest": journal_evidence["journal_digest"],
        "delivery_count": delivery_evidence["count"],
        "delivery_chain_head": delivery_evidence["head"],
        "delivery_chain_valid": delivery_evidence["chain_valid"],
        "delivery_journal_digest": delivery_evidence["journal_digest"],
        "receipt_delivery_bijection": delivery_evidence["receipt_delivery_bijection"],
        "sandbox_negative_control_count": len(negative["results"]),
        "sandbox_negative_controls_all_passed": all(negative["results"].values()),
        "sandbox_negative_controls_digest": digest_for(negative["results"]),
    }
    exact_bindings.update(outcome_counts)
    if any(report.get(field) != value for field, value in exact_bindings.items()):
        raise EGVError("Gate-A evaluator report differs from fresh replay evidence")
    if (
        report.get("schema_version") != GATE_A_REPORT_VERSION
        or report.get("production_acceptance") is not True
        or report.get("sandbox_preflight_passed") is not True
        or report.get("sandbox_postflight_passed") is not True
        or report.get("task_bearing_model_requests_permitted") is not False
        or report.get("model_quality_claimed") is not False
        or report.get("cross_node_transport_exercised") is not False
        or report.get("hidden_values_exposed") is not False
        or report.get("private_paths_exposed") is not False
    ):
        raise EGVError("Gate-A evaluator report did not pass every frozen acceptance surface")
    if any(issuer.get(field) != value for field, value in exact_bindings.items() if field in issuer):
        raise EGVError("Gate-A issuer record differs from fresh verifier bindings")
    if not _gate_a_acceptance(
        oracle_pass_count=outcome_counts["oracle_pass_count"],
        blind_pass_count=outcome_counts["blind_pass_count"],
        starter_wrong_output_count=outcome_counts["starter_wrong_output_count"],
        infrastructure_loss_count=outcome_counts["infrastructure_loss_count"],
        receipt_count=journal_evidence["count"],
        receipt_chain_valid=journal_evidence["chain_valid"],
        delivery_count=delivery_evidence["count"],
        delivery_chain_valid=delivery_evidence["chain_valid"],
        receipt_delivery_bijection=delivery_evidence["receipt_delivery_bijection"],
        sandbox_preflight_passed=report["sandbox_preflight_passed"],
        sandbox_negative_controls_all_passed=all(negative["results"].values()),
        sandbox_postflight_passed=report["sandbox_postflight_passed"],
    ):
        raise EGVError("Gate-A fresh verifier rejected the corpus validation")

    public_preprivacy = {
        "schema_version": GATE_A_PUBLIC_RESULT_VERSION,
        "disposition": GATE_A_DISPOSITION,
        "evidence_scope": "production-corpus-validation-no-model-quality-claim",
        "production_acceptance": True,
        "task_bearing_model_requests_permitted": False,
        "separate_go_task_bearing_required": True,
        "model_quality_claimed": False,
        "cross_node_transport_exercised": False,
        **{
            field: exact_bindings[field]
            for field in (
                "freeze_bundle_digest",
                "plan_digest",
                "protocol_digest",
                "policy_digest",
                "public_manifest_digest",
                "private_manifest_digest",
                "runtime_manifest_digest",
                "blind_solver_manifest_digest",
                "blind_solver_source_digest",
                "source_manifest_digest",
                "source_commit",
                "candidate_source_inventory_digest",
                "evaluation_binding_digest",
                "controller_key_id",
                "freeze_signer_key_id",
                "oracle_task_count",
                "oracle_evaluation_count",
                "oracle_pass_count",
                "blind_task_count",
                "blind_evaluation_count",
                "blind_pass_count",
                "starter_task_count",
                "starter_evaluation_count",
                "starter_wrong_output_count",
                "total_evaluation_count",
                "infrastructure_loss_count",
                "expected_receipt_count",
                "receipt_count",
                "receipt_chain_head",
                "receipt_chain_valid",
                "receipt_journal_digest",
                "delivery_count",
                "delivery_chain_head",
                "delivery_chain_valid",
                "delivery_journal_digest",
                "receipt_delivery_bijection",
                "sandbox_negative_control_count",
                "sandbox_negative_controls_all_passed",
                "sandbox_negative_controls_digest",
            )
        },
        "controller_report_digest": report["report_digest"],
        "controller_report_file_digest": digest_bytes(_canonical_record_bytes(report)),
        "sandbox_preflight_passed": report["sandbox_preflight_passed"],
        "sandbox_postflight_passed": report["sandbox_postflight_passed"],
    }
    forbidden_tokens = _private_scan_tokens(
        corpus,
        evaluator_seed=evaluator_seed,
        controller_private_key=None,
        freeze_private_key=freeze_private_key,
        workspace=workspace,
        package_root=package_root,
        identities=identities,
    )
    # The verifier does not receive the controller private key.  Scan its
    # external public path and every seed/freeze secret it does receive; the
    # signed receipt replay proves the controller key stayed in the evaluator.
    forbidden_tokens += (
        str(Path(controller_public_key).resolve(strict=False)).encode("utf-8"),
        canonical_json(str(Path(controller_public_key).resolve(strict=False))).encode("utf-8"),
    )
    privacy = _privacy_scan(public_preprivacy, forbidden_tokens=forbidden_tokens)
    public_core = {**public_preprivacy, "privacy_scan": privacy}
    public_result = _sign_closed_record(
        public_core,
        freeze_signer,
        digest_field="result_digest",
        signature_field="freeze_authority_signature",
    )
    _verify_closed_signature(
        public_result,
        freeze_signer.public_key,
        core_fields=_PUBLIC_CORE_FIELDS,
        digest_field="result_digest",
        signature_field="freeze_authority_signature",
        label="Gate-A public result",
    )
    # A second scan covers the result digest and external signature as actually
    # written.  No result is published when either scan finds anything.
    _privacy_scan(public_result, forbidden_tokens=forbidden_tokens)
    return public_result


def _verifier_entry(*args: Any) -> None:
    worker_args = args[:-1]
    result_sender = args[-1]
    workspace = str(worker_args[4])
    try:
        result_sender.send_bytes(_canonical_record_bytes(_verifier_worker(*worker_args)))
    except BaseException as exc:
        try:
            failures_identity = _validated_identity_record(worker_args[-1])["failures"]
        except BaseException:
            failures_identity = None
        _failure_record(
            workspace,
            "verifier",
            exc,
            failures_identity=failures_identity,
        )
        raise SystemExit(1)
    finally:
        try:
            result_sender.close()
        except BaseException:
            pass


def validate_gate_a_public_result(record: Mapping[str, Any], freeze_public_key: Any) -> Dict[str, Any]:
    """Cryptographically validate a public GO against its freeze authority."""

    try:
        freeze_public = public_key_bytes(freeze_public_key)
    except Exception as exc:
        raise ValueError("Gate-A freeze public key is invalid") from exc
    result = _verify_closed_signature(
        record,
        freeze_public,
        core_fields=_PUBLIC_CORE_FIELDS,
        digest_field="result_digest",
        signature_field="freeze_authority_signature",
        label="Gate-A public result",
    )
    if (
        result.get("schema_version") != GATE_A_PUBLIC_RESULT_VERSION
        or result.get("disposition") != GATE_A_DISPOSITION
        or result.get("evidence_scope") != "production-corpus-validation-no-model-quality-claim"
        or result.get("production_acceptance") is not True
        or result.get("task_bearing_model_requests_permitted") is not False
        or result.get("separate_go_task_bearing_required") is not True
        or result.get("model_quality_claimed") is not False
        or result.get("cross_node_transport_exercised") is not False
        or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", str(result.get("source_commit", "")))
        or not re.fullmatch(r"key_[0-9a-f]{64}", str(result.get("controller_key_id", "")))
        or not re.fullmatch(r"key_[0-9a-f]{64}", str(result.get("freeze_signer_key_id", "")))
        or result.get("freeze_signer_key_id") != key_id_for_public_key(freeze_public)
        or result.get("controller_key_id") == result.get("freeze_signer_key_id")
        or not isinstance(result.get("sandbox_negative_control_count"), int)
        or isinstance(result.get("sandbox_negative_control_count"), bool)
        or result.get("sandbox_negative_control_count") <= 0
    ):
        raise ValueError("Gate-A public result has an invalid claim boundary")
    exact_counts = {
        "oracle_task_count": ORACLE_TASK_COUNT,
        "oracle_evaluation_count": ORACLE_EVALUATION_COUNT,
        "oracle_pass_count": ORACLE_EVALUATION_COUNT,
        "blind_task_count": BLIND_TASK_COUNT,
        "blind_evaluation_count": BLIND_EVALUATION_COUNT,
        "blind_pass_count": BLIND_EVALUATION_COUNT,
        "starter_task_count": STARTER_TASK_COUNT,
        "starter_evaluation_count": STARTER_EVALUATION_COUNT,
        "starter_wrong_output_count": STARTER_EVALUATION_COUNT,
        "total_evaluation_count": TOTAL_EVALUATION_COUNT,
        "infrastructure_loss_count": 0,
        "expected_receipt_count": EXPECTED_RECEIPT_COUNT,
        "receipt_count": EXPECTED_RECEIPT_COUNT,
        "delivery_count": EXPECTED_RECEIPT_COUNT,
    }
    if any(result.get(field) != value for field, value in exact_counts.items()):
        raise ValueError("Gate-A public result differs from the frozen counts")
    if (
        result.get("receipt_chain_valid") is not True
        or result.get("delivery_chain_valid") is not True
        or result.get("receipt_delivery_bijection") is not True
        or result.get("sandbox_preflight_passed") is not True
        or result.get("sandbox_negative_controls_all_passed") is not True
        or result.get("sandbox_postflight_passed") is not True
    ):
        raise ValueError("Gate-A public result has an invalid evidence disposition")
    privacy_fields = {
        "schema_version",
        "policy_digest",
        "subject_digest",
        "prohibited_token_count",
        "prohibited_pattern_count",
        "finding_count",
        "scan_passed",
        "hidden_values_exposed",
        "private_paths_or_topology_exposed",
        "credential_material_exposed",
        "privacy_receipt_digest",
    }
    privacy = _closed(result.get("privacy_scan"), privacy_fields, "Gate-A public privacy scan")
    privacy_core = {key: privacy[key] for key in privacy if key != "privacy_receipt_digest"}
    public_preprivacy = {field: result[field] for field in _PUBLIC_CORE_FIELDS if field != "privacy_scan"}
    if (
        privacy.get("schema_version") != GATE_A_PRIVACY_VERSION
        or privacy.get("policy_digest") != digest_for([pattern.decode("ascii") for pattern in _PRIVACY_PATTERNS])
        or privacy.get("subject_digest") != digest_bytes(_canonical_record_bytes(public_preprivacy))
        or not isinstance(privacy.get("prohibited_token_count"), int)
        or isinstance(privacy.get("prohibited_token_count"), bool)
        or privacy.get("prohibited_token_count") < 0
        or privacy.get("prohibited_pattern_count") != len(_PRIVACY_PATTERNS)
        or privacy.get("finding_count") != 0
        or privacy.get("scan_passed") is not True
        or privacy.get("hidden_values_exposed") is not False
        or privacy.get("private_paths_or_topology_exposed") is not False
        or privacy.get("credential_material_exposed") is not False
        or privacy.get("privacy_receipt_digest") != digest_for(privacy_core)
    ):
        raise ValueError("Gate-A public privacy scan is invalid")
    for field, value in result.items():
        if field.endswith("_digest") or field in {
            "receipt_chain_head",
            "delivery_chain_head",
            "source_manifest_digest",
        }:
            validate_sha256(value, field)
    return dict(result)


def _validate_parent_input_path(path: Path, label: str) -> Path:
    _path_identity(path, label, directory=False)
    return path.resolve(strict=True)


def _is_within(path: Path, directory: Path) -> bool:
    try:
        path.relative_to(directory)
    except ValueError:
        return False
    return True


def _run_spawn_stage(
    name: str,
    target: Any,
    args: Sequence[Any],
    *,
    timeout_seconds: float,
    expect_result: bool = False,
    bytecode_cache: Optional[Path] = None,
    expected_bytecode_cache_identity: Optional[Tuple[int, int]] = None,
) -> Optional[Dict[str, Any]]:
    if (bytecode_cache is None) != (expected_bytecode_cache_identity is None):
        raise EGVError("Gate-A spawn bytecode boundary is incomplete")
    selected_cache = None
    if bytecode_cache is not None and expected_bytecode_cache_identity is not None:
        selected_cache = Path(bytecode_cache)
        _require_identity(
            selected_cache,
            expected_bytecode_cache_identity,
            "Gate-A spawn bytecode boundary",
            directory=True,
        )
        if any(selected_cache.iterdir()):
            raise EGVError("Gate-A spawn bytecode boundary must be fresh and empty")
    context = mp.get_context("spawn")
    result_receiver = None
    result_sender = None
    process_args = tuple(args)
    if expect_result:
        result_receiver, result_sender = context.Pipe(duplex=False)
        process_args = (*process_args, result_sender)
    process = context.Process(target=target, args=process_args, name="egv-gate-a-" + name)
    try:
        if selected_cache is None:
            process.start()
        else:
            environment_updates = {
                "PYTHONPYCACHEPREFIX": str(selected_cache.resolve(strict=True)),
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            with _SPAWN_ENVIRONMENT_LOCK:
                previous_environment = {name: os.environ.get(name) for name in environment_updates}
                os.environ.update(environment_updates)
                try:
                    process.start()
                finally:
                    for variable, previous in previous_environment.items():
                        if previous is None:
                            os.environ.pop(variable, None)
                        else:
                            os.environ[variable] = previous
    except BaseException:
        if result_receiver is not None:
            result_receiver.close()
        if result_sender is not None:
            result_sender.close()
        try:
            process.close()
        except BaseException:
            pass
        raise
    if result_sender is not None:
        result_sender.close()
    process.join(timeout_seconds)
    if process.is_alive():
        process.terminate()
        process.join(30)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(30)
        still_alive = process.is_alive()
        process.close()
        if result_receiver is not None:
            result_receiver.close()
        if still_alive:
            raise EGVError("Gate-A {} stage could not be stopped after timeout".format(name))
        raise EGVError("Gate-A {} stage exceeded its frozen timeout; private evidence was preserved".format(name))
    exit_code = process.exitcode
    process.close()
    if exit_code != 0:
        if result_receiver is not None:
            result_receiver.close()
        raise EGVError("Gate-A {} stage failed; private evidence was preserved".format(name))
    if selected_cache is not None and expected_bytecode_cache_identity is not None:
        _require_identity(
            selected_cache,
            expected_bytecode_cache_identity,
            "Gate-A spawn bytecode boundary",
            directory=True,
        )
        if any(selected_cache.iterdir()):
            if result_receiver is not None:
                result_receiver.close()
            raise EGVError("Gate-A spawn bytecode boundary was not empty after child exit")
    if result_receiver is None:
        return None
    try:
        if not result_receiver.poll():
            raise EGVError("Gate-A {} stage returned no safe result".format(name))
        payload = result_receiver.recv_bytes(maxlength=1024 * 1024)
        result = json.loads(payload.decode("utf-8"))
        if not isinstance(result, dict) or payload != _canonical_record_bytes(result):
            raise EGVError("Gate-A {} stage returned a non-canonical safe result".format(name))
    except (EOFError, OSError, UnicodeError, ValueError) as exc:
        raise EGVError("Gate-A {} stage safe-result channel failed".format(name)) from exc
    finally:
        result_receiver.close()
    return dict(result)


def run_gate_a_corpus_validation(
    evaluator_seed: Union[str, Path],
    controller_private_key: Union[str, Path],
    controller_public_key: Union[str, Path],
    freeze_private_key: Union[str, Path],
    freeze_public_key: Union[str, Path],
    workspace: Union[str, Path],
    public_output: Union[str, Path],
    package_root: Union[str, Path],
    source_commit: str,
    config: DockerSandboxConfig,
    *,
    stage_timeout_seconds: float = GATE_A_STAGE_TIMEOUT_SECONDS,
) -> Dict[str, Any]:
    """Run issuer, evaluator, and verifier in three sequential spawn children."""

    if (
        not isinstance(stage_timeout_seconds, (int, float))
        or isinstance(stage_timeout_seconds, bool)
        or stage_timeout_seconds <= 0
    ):
        raise EGVError("Gate-A stage timeout must be positive")
    labeled_inputs = (
        ("Gate-A evaluator seed", Path(evaluator_seed)),
        ("Gate-A controller private key", Path(controller_private_key)),
        ("Gate-A controller public key", Path(controller_public_key)),
        ("Gate-A freeze private key", Path(freeze_private_key)),
        ("Gate-A freeze public key", Path(freeze_public_key)),
    )
    validated_inputs = [_validate_parent_input_path(path, label) for label, path in labeled_inputs]
    input_identities: Dict[Tuple[int, int], str] = {}
    selected_input_identities: Dict[str, Tuple[int, int]] = {}
    selected_input_states: Dict[str, Tuple[int, int, int, int, int]] = {}
    input_identity_names = (
        "evaluator_seed",
        "controller_private",
        "controller_public",
        "freeze_private",
        "freeze_public",
    )
    for identity_name, (label, _path), path in zip(input_identity_names, labeled_inputs, validated_inputs):
        identity = _path_identity(path, label, directory=False)
        if identity in input_identities:
            raise EGVError("{} aliases {} through the same file identity".format(label, input_identities[identity]))
        input_identities[identity] = label
        selected_input_identities[identity_name] = identity
        selected_input_states[identity_name] = _file_selection_state(path, label)
    seed, controller_private, controller_public, freeze_private, freeze_public = validated_inputs
    supplied_package_root = Path(package_root)
    _path_identity(supplied_package_root, "Gate-A package root", directory=True)
    package = supplied_package_root.resolve(strict=True)
    package_identity = _path_identity(package, "Gate-A package root", directory=True)
    if not isinstance(source_commit, str) or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", source_commit):
        raise EGVError("Gate-A source commit must be a full lowercase Git object ID")
    root = Path(workspace)
    output = Path(public_output)
    _assert_no_link_components(root, "Gate-A workspace")
    if root.exists() and not root.is_dir():
        raise EGVError("Gate-A workspace must be an absolute non-link directory")
    if root.exists() and any(root.iterdir()):
        raise EGVError("Gate-A workspace must be newly absent or empty")
    root = root.resolve(strict=False)
    _assert_no_link_components(output, "Gate-A public output")
    if output.exists() or output.is_symlink():
        raise EGVError("Gate-A public output must be a new absolute path")
    output = output.resolve(strict=False)
    if _is_within(output, root):
        raise EGVError("Gate-A public output must remain outside the evaluator-private workspace")
    if _is_within(root, package) or _is_within(output, package) or _is_within(package, root):
        raise EGVError("Gate-A workspace/output and package checkout must be disjoint")
    input_paths = set(validated_inputs)
    if output in input_paths or root in input_paths:
        raise EGVError("Gate-A output/workspace must be distinct from all private inputs")
    workspace_parent_identity = _path_identity(root.parent, "Gate-A workspace parent", directory=True)
    output_parent_identity = _path_identity(output.parent, "Gate-A public output parent", directory=True)
    if not root.exists():
        try:
            root.mkdir(mode=0o700, exist_ok=False)
        except OSError as exc:
            raise EGVError("Gate-A workspace could not be created safely") from exc
        _require_identity(
            root.parent,
            workspace_parent_identity,
            "Gate-A workspace parent",
            directory=True,
        )
    workspace_identity = _path_identity(root, "Gate-A workspace", directory=True)
    layout_identities = _prepare_private_layout(root)
    identity_record: Dict[str, Tuple[int, ...]] = {
        **selected_input_identities,
        **{"{}_state".format(name): state for name, state in selected_input_states.items()},
        "package_root": package_identity,
        "workspace": workspace_identity,
        "private": layout_identities["private"],
        "issuer": layout_identities["issuer"],
        "evaluator": layout_identities["evaluator"],
        "failures": layout_identities["failures"],
        "bytecode": layout_identities["bytecode"],
        "issuer_bytecode": layout_identities["issuer_bytecode"],
        "evaluator_bytecode": layout_identities["evaluator_bytecode"],
        "verifier_bytecode": layout_identities["verifier_bytecode"],
        "output_parent": output_parent_identity,
    }
    _validated_identity_record(identity_record)
    config_record = _config_record(config)
    common_issuer = (
        str(seed),
        str(controller_private),
        str(controller_public),
        str(freeze_private),
        str(freeze_public),
        str(root),
        str(package),
        source_commit,
        config_record,
        identity_record,
    )
    evaluator_args = (
        str(seed),
        str(controller_private),
        str(controller_public),
        str(freeze_public),
        str(root),
        str(package),
        source_commit,
        config_record,
        identity_record,
    )
    verifier_args = (
        str(seed),
        str(controller_public),
        str(freeze_private),
        str(freeze_public),
        str(root),
        str(package),
        source_commit,
        config_record,
        identity_record,
    )

    def validate_surfaces(*, output_must_be_absent: bool) -> None:
        for identity_name, (label, _original), path in zip(input_identity_names, labeled_inputs, validated_inputs):
            _require_identity(
                path,
                selected_input_identities[identity_name],
                label,
                directory=False,
            )
            _require_file_selection_state(
                path,
                selected_input_states[identity_name],
                label,
            )
        _require_identity(package, package_identity, "Gate-A package root", directory=True)
        _require_identity(root, workspace_identity, "Gate-A workspace", directory=True)
        _require_identity(
            output.parent,
            output_parent_identity,
            "Gate-A public output parent",
            directory=True,
        )
        paths = _private_paths(root)
        for name, identity in layout_identities.items():
            _require_identity(paths[name], identity, "Gate-A {} directory".format(name), directory=True)
        if output_must_be_absent and (output.exists() or output.is_symlink()):
            raise EGVError("Gate-A public output appeared before fresh verification")

    try:
        validate_surfaces(output_must_be_absent=True)
        _run_spawn_stage(
            "issuer",
            _issuer_entry,
            common_issuer,
            timeout_seconds=stage_timeout_seconds,
            bytecode_cache=_private_paths(root)["issuer_bytecode"],
            expected_bytecode_cache_identity=layout_identities["issuer_bytecode"],
        )
        validate_surfaces(output_must_be_absent=True)
        _run_spawn_stage(
            "evaluator",
            _evaluator_entry,
            evaluator_args,
            timeout_seconds=stage_timeout_seconds,
            bytecode_cache=_private_paths(root)["evaluator_bytecode"],
            expected_bytecode_cache_identity=layout_identities["evaluator_bytecode"],
        )
        validate_surfaces(output_must_be_absent=True)
        verifier_result = _run_spawn_stage(
            "verifier",
            _verifier_entry,
            verifier_args,
            timeout_seconds=stage_timeout_seconds,
            expect_result=True,
            bytecode_cache=_private_paths(root)["verifier_bytecode"],
            expected_bytecode_cache_identity=layout_identities["verifier_bytecode"],
        )
        validate_surfaces(output_must_be_absent=True)
        if verifier_result is None:
            raise EGVError("Gate-A verifier returned no safe result")
        freeze_public_bytes = _read_public_key(
            freeze_public,
            "Gate-A freeze authority public key",
            expected_identity=selected_input_identities["freeze_public"],
            expected_state=selected_input_states["freeze_public"],
        )
        validated_result = validate_gate_a_public_result(verifier_result, freeze_public_bytes)
        _privacy_scan(
            validated_result,
            forbidden_tokens=_resolved_path_scan_tokens(
                seed,
                controller_private,
                controller_public,
                freeze_private,
                freeze_public,
                root,
                output,
                package,
            ),
        )
        validate_surfaces(output_must_be_absent=True)
        _atomic_write_new(
            output,
            validated_result,
            private=False,
            expected_parent_identity=output_parent_identity,
        )
        return validated_result
    except BaseException:
        # Children never publish.  The identity-bound atomic writer removes
        # only its own partial inode, so anything else at the reserved path is
        # unrelated and must not be deleted by pathname.
        raise


__all__ = [
    "BLIND_EVALUATION_COUNT",
    "BLIND_TASK_COUNT",
    "DurableDeliveryJournal",
    "EXPECTED_RECEIPT_COUNT",
    "GATE_A_DISPOSITION",
    "GATE_A_PLAN_VERSION",
    "GATE_A_PUBLIC_RESULT_VERSION",
    "GATE_A_REPORT_VERSION",
    "GATE_A_SOURCE_MANIFEST_VERSION",
    "ORACLE_EVALUATION_COUNT",
    "ORACLE_TASK_COUNT",
    "STARTER_EVALUATION_COUNT",
    "STARTER_TASK_COUNT",
    "StreamingReceiptJournal",
    "TOTAL_EVALUATION_COUNT",
    "build_evaluator_source_manifest",
    "build_gate_a_plan",
    "build_gate_a_runtime_manifest",
    "run_gate_a_corpus_validation",
    "validate_evaluator_source_manifest",
    "validate_gate_a_plan",
    "validate_gate_a_public_result",
]
