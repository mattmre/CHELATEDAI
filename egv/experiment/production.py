"""Sealed production integration for the frozen held-out campaign.

This module is deliberately transport agnostic.  Commands may be carried by an
operator-approved local or remote wrapper, but admission is based on exact
bytes, immutable manifests, signed evaluator responses, and durable operation
state.  No host name, credential, or user-supplied module path is accepted as
configuration here.
The deployment record separately declares which subprocess boundary has
pre-import source isolation; artifact binding alone is not that claim.
"""

from __future__ import annotations

import base64
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import signal
import stat
import subprocess
import sys
import tempfile
import threading
from time import monotonic
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ..canonical import (
    GENESIS_HASH,
    canonical_bytes,
    canonical_json,
    content_id,
    digest_bytes,
    digest_for,
    failure_family_root,
    parse_canonical_jsonl,
    validate_sha256,
)
from ..evaluation.authority import AuthorityPolicy
from ..evaluation.diagnostics import Diagnostic, validate_diagnostic, validate_disposition, validate_resource_bucket
from ..identities import commissioning_run_id
from ..ledger import EvidenceLedger, INLINE_PAYLOAD_LIMIT
from ..receipts import ReceiptSigner, load_public_key, receipt_hash
from ..variation.adapter import ADAPTER_MANIFEST_NAME, SealedAdapterArtifact
from ..variation.arms import arm_policy
from ..variation.checkpoint import VariationCheckpoint
from ..variation.generator import (
    CandidateContext,
    CandidateGenerationEvidence,
    CandidateGenerationFailureEvidence,
    ModelCandidateGenerator,
    SOURCE_ONLY_RESPONSE_CONTRACT_DIGEST,
    model_generation_profile_digest,
    render_candidate_prompt,
)
from ..variation.loop import (
    MAX_CANDIDATE_ATTEMPTS,
    VARIATION_PROTOCOL_DIGEST,
    AttemptRecord,
    BoundedCandidateLoop,
    SourceContractBudgetExhausted,
    VariationReport,
)
from ..variation.model import MODEL_REVISION, PinnedModelLoader
from ..variation.private import (
    PRIVATE_GENERATION_SCHEMA,
    PrivateTrajectoryStore,
    replay_exception_chain_classification,
    response_contract_replay_error_chain,
)
from ..variation.retrieval import retrieval_policy
from ..variation.remote import (
    REMOTE_VARIATION_REQUEST_LIMIT,
    REMOTE_VARIATION_TIMEOUT_SECONDS,
    RemoteControllerEvaluationGateway,
    RemoteEvaluatorServiceManifest,
    _bounded_process_exited_without_reap,
    pinned_python_invocation,
    run_remote_evaluator_once,
    _validate_remote_result_semantics,
)
from ..variation.errors import VariationConfigurationError, VariationDependencyError
from .heldout import (
    MAIN_PHASE,
    RESULT_SCHEMA,
    SHOCK_PHASE,
    CoordinateOperationStore,
    FrozenHeldoutProtocol,
    HeldoutCoordinate,
    HeldoutProtocolError,
    validate_coordinate_operation_state,
    validate_result,
)
from .remote import (
    EvaluatorVerification,
    HeldoutVerifierServiceManifest,
    RemoteHeldoutReconciler,
    RemoteHeldoutResultVerifier,
    build_heldout_verifier_command,
)
from .runtime import (
    HeldoutCoordinateRunner,
    HeldoutRuntimeContext,
    HeldoutRuntimeEvidence,
    HeldoutTrainerInputs,
    HeldoutTrainerSources,
    derive_verified_main_result,
)
from .shock_runtime import (
    CorrectionShockCoordinateRunner,
    ShockAttemptObservation,
    ShockRuntimeContext,
    ShockRuntimeJournal,
)


PRODUCTION_INTEGRATION_NAME = "qwen-heldout-production-v1"
DEPLOYMENT_MANIFEST_SCHEMA = "egv-heldout-production-deployment-v2"
HELDOUT_EVALUATOR_EXECUTION_MODE = "python-json-v1"
MAIN_EVIDENCE_SCHEMA = "egv-heldout-main-evidence-v2"
SHOCK_EVIDENCE_SCHEMA = "egv-heldout-shock-evidence-v2"
OBSERVATION_SCHEMA = "egv-heldout-production-observation-v2"
DISPATCH_RECORD_SCHEMA = "egv-heldout-production-dispatch-v1"
PRODUCTION_REQUEST_LIMIT = 8 * 1024 * 1024
PRODUCTION_RESPONSE_LIMIT = 8 * 1024 * 1024
PRODUCTION_STDERR_LIMIT = 256 * 1024
_WINDOWS_CREATE_SUSPENDED = 0x00000004
RECEIPT_ROUTER_SCHEMA = "egv-variation-receipt-router-v1"
RECEIPT_ROUTER_CAPABILITY = "content-addressed-receipt-chain-fork-v1"
RECEIPT_ROUTER_CONFIG_SCHEMA = "egv-variation-receipt-router-config-v1"
RECEIPT_ROUTER_STATE_SCHEMA = "egv-remote-variation-state-v1"
PRODUCTION_SOURCE_MANIFEST_SCHEMA = "egv-production-source-manifest-v1"
SOURCE_ISOLATION_SCOPE = "variation-router-only-v1"
_SHOCK_LEDGER_EVENT_TYPES = frozenset(
    {
        "CAMPAIGN",
        "RUN",
        "SHOCK_PREMISE",
        "SHOCK_UNRELATED_ROOT",
        "SHOCK_UNRELATED_EVIDENCE",
        "CANDIDATE",
        "SHOCK_GENERATION_FAILURE",
        "RECEIPT",
        "VERDICT",
        "EFFECT_RECEIPT",
        "SHOCK_ATTEMPT",
        "DEPENDENCY",
        "SHOCK_CORRECTED_PREMISE",
        "CORRECTION",
        "SHOCK_CORRECTION_COMMIT",
        "SHOCK_POLICY_ACTIVATION",
    }
)
_LEDGER_EVENT_EXPORT_FIELDS = frozenset(
    {
        "record_type",
        "ledger_schema_version",
        "sequence",
        "event_id",
        "campaign_id",
        "run_id",
        "task_id",
        "event_type",
        "transaction_time",
        "valid_time",
        "subject_id",
        "payload_hash",
        "payload_json",
        "blob_digest",
        "source_class",
        "disposition",
        "evaluator_identity",
        "idempotency_key",
        "previous_hash",
        "event_hash",
        "payload",
    }
)
_LEDGER_CHECKPOINT_EXPORT_FIELDS = frozenset(
    {"record_type", "ledger_schema_version", "checkpoint"}
)
_SOURCE_COMMIT = re.compile(r"[0-9a-f]{40}")
_ROUTER_CONFIG_FIELDS = (
    "schema_version",
    "service_manifest",
    "evaluator_seed",
    "evaluator_private_key",
    "workspace",
    "state_root",
    "egv_package_root",
    "egv_source_manifest",
    "python_executable",
    "bootstrap_executable",
    "runtime_import_roots",
)
_ROUTED_VARIATION_REQUEST_FIELDS = (
    "schema_version",
    "operation_digest",
    "request_digest",
    "service_manifest_digest",
    "campaign_id",
    "model_digest",
    "protocol_digest",
    "policy_digest",
    "run_id",
    "arm_policy_digest",
    "data_manifest_digest",
    "task_manifest_digest",
    "evaluator_digest",
    "docker_image_digest",
    "candidate_id",
    "task_id",
    "public_task_binding",
    "candidate_artifact_digest",
    "candidate_source_b64",
    "requested_authority",
    "declared_locus",
    "receipt_sequence_start",
    "previous_receipt_hash",
)

ARTIFACT_NAMES = (
    "protocol",
    "trainer_inputs",
    "trainer_sources",
    "heldout_service_manifest",
    "model_manifest",
    "adapter_manifest",
    "variation_evaluator_manifest",
    "variation_evaluator_public_key",
    "variation_evaluator_command",
    "variation_receipt_router_manifest",
    "heldout_evaluator_command",
    "python_executable",
)


class _ExpectedVariationManifest:
    """Minimal immutable view consumed by the canonical remote-result validator."""

    def __init__(self, protocol: FrozenHeldoutProtocol) -> None:
        self._values = MappingProxyType(
            {
                "campaign_id": protocol.campaign_id,
                "protocol_digest": VARIATION_PROTOCOL_DIGEST,
                "policy_digest": protocol.bindings["policy_manifest_digest"],
            }
        )
        self.digest = protocol.bindings["evaluator_digest"]

    def __getitem__(self, key: str) -> Any:
        return self._values[key]


def _main_run_id(protocol: FrozenHeldoutProtocol, coordinate: HeldoutCoordinate) -> str:
    return commissioning_run_id(
        campaign_id=protocol.campaign_id,
        task_id=coordinate.task_id,
        arm_id=coordinate.treatment,
        seed=coordinate.seed,
    )


def _main_candidate_id(
    protocol: FrozenHeldoutProtocol,
    coordinate: HeldoutCoordinate,
    *,
    run_id: str,
    attempt: int,
    parent: Optional[str],
) -> str:
    return "egv-candidate-{}-{}-{}".format(
        protocol.campaign_id,
        coordinate.treatment,
        digest_for(
            {
                "run_id": run_id,
                "task_id": coordinate.task_id,
                "seed": coordinate.seed,
                "attempt": attempt,
                "parent": parent,
            }
        )[:40],
    )


def _shock_run_id(protocol: FrozenHeldoutProtocol, coordinate: HeldoutCoordinate, phase: str) -> str:
    if phase == "PRE":
        return content_id(
            "run-shock-pre",
            {"campaign_id": protocol.campaign_id, "block_id": coordinate.block_id},
        )
    return content_id(
        "run-shock-post",
        {"campaign_id": protocol.campaign_id, "coordinate_id": coordinate.coordinate_id},
    )


def _shock_candidate_id(
    protocol: FrozenHeldoutProtocol,
    coordinate: HeldoutCoordinate,
    phase: str,
    attempt: int,
) -> str:
    identity = {
        "campaign_id": protocol.campaign_id,
        "block_id": coordinate.block_id if phase == "PRE" else None,
        "coordinate_id": coordinate.coordinate_id if phase == "POST" else None,
        "phase": phase,
        "attempt": attempt,
    }
    return "egv-candidate-{}-E-{}".format(protocol.campaign_id, digest_for(identity)[:40])


def _closed(value: Mapping[str, Any], fields: Sequence[str], label: str) -> None:
    if not isinstance(value, Mapping) or set(value) != set(fields):
        raise HeldoutProtocolError("{} is not a closed object".format(label))


def _exact_event_envelope(
    event: Optional[Mapping[str, Any]],
    *,
    event_type: str,
    payload: Mapping[str, Any],
    campaign_id: Optional[str],
    run_id: Optional[str],
    task_id: Optional[str],
    subject_id: Optional[str],
    source_class: Optional[str],
    disposition: Optional[str],
    evaluator_identity: Optional[str],
    idempotency_key: Optional[str],
) -> bool:
    """Compare the complete immutable semantic envelope of one ledger event."""

    payload_bytes = canonical_bytes(payload)
    payload_hash = digest_bytes(payload_bytes)
    immutable = {
        "event_type": event_type,
        "campaign_id": campaign_id,
        "run_id": run_id,
        "task_id": task_id,
        "valid_time": None,
        "subject_id": subject_id,
        "payload_hash": payload_hash,
        "payload_json": (
            payload_bytes.decode("utf-8")
            if len(payload_bytes) <= INLINE_PAYLOAD_LIMIT
            else None
        ),
        "blob_digest": payload_hash if len(payload_bytes) > INLINE_PAYLOAD_LIMIT else None,
        "source_class": source_class,
        "disposition": disposition,
        "evaluator_identity": evaluator_identity,
        "idempotency_key": idempotency_key,
    }
    expected = {
        **immutable,
        "event_id": content_id("evt", immutable),
        "payload": dict(payload),
    }
    return event is not None and all(
        event.get(key) == value for key, value in expected.items()
    )


def _link_like(metadata: os.stat_result) -> bool:
    attributes = int(getattr(metadata, "st_file_attributes", 0))
    reparse = int(getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))
    return stat.S_ISLNK(metadata.st_mode) or bool(attributes & reparse)


def _regular_single_link_bytes(path: Path, label: str, *, limit: Optional[int] = None) -> bytes:
    target = Path(os.path.abspath(os.fspath(path)))
    cursor = target
    while True:
        try:
            metadata = os.lstat(cursor)
        except OSError as exc:
            raise HeldoutProtocolError("{} is missing or unreadable".format(label)) from exc
        if _link_like(metadata):
            raise HeldoutProtocolError("{} path contains a link or reparse point".format(label))
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    try:
        metadata = os.lstat(target)
        if not stat.S_ISREG(metadata.st_mode) or int(getattr(metadata, "st_nlink", 0)) != 1:
            raise HeldoutProtocolError("{} must be a single-link regular file".format(label))
        if limit is not None and metadata.st_size > limit:
            raise HeldoutProtocolError("{} exceeds its bounded byte limit".format(label))
        raw = target.read_bytes()
    except OSError as exc:
        raise HeldoutProtocolError("{} cannot be read".format(label)) from exc
    if len(raw) != metadata.st_size:
        raise HeldoutProtocolError("{} changed while it was read".format(label))
    return raw


def _canonical_object_file(path: Path, label: str, *, limit: int = PRODUCTION_REQUEST_LIMIT) -> Mapping[str, Any]:
    raw = _regular_single_link_bytes(path, label, limit=limit)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise HeldoutProtocolError("{} is not UTF-8 JSON".format(label)) from exc
    if not isinstance(value, Mapping) or canonical_bytes(value) + b"\n" != raw:
        raise HeldoutProtocolError("{} must be canonical JSON plus one newline".format(label))
    return value


def _artifact_record(path: Path, label: str) -> Dict[str, Any]:
    raw = _regular_single_link_bytes(path, label)
    return {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}


def _atomic_canonical(path: Path, value: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    encoded = canonical_bytes(value) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix="." + target.name + ".", dir=str(target.parent))
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(temporary), str(target))
    finally:
        if temporary.exists():
            temporary.unlink()


@contextmanager
def _exclusive_router_lock(path: Path):
    """Serialize one evaluator-owned receipt router without ambient services."""

    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(str(path), os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        metadata = os.lstat(path)
        if _link_like(metadata) or not stat.S_ISREG(metadata.st_mode) or int(getattr(metadata, "st_nlink", 0)) != 1:
            raise HeldoutProtocolError("Variation receipt router lock is not a private regular file")
        descriptor = os.open(str(path), os.O_RDWR)
        opened = os.fstat(descriptor)
        if (
            (opened.st_dev, opened.st_ino) != (metadata.st_dev, metadata.st_ino)
            or not stat.S_ISREG(opened.st_mode)
            or int(getattr(opened, "st_nlink", 0)) != 1
        ):
            os.close(descriptor)
            raise HeldoutProtocolError("Variation receipt router lock changed while opening")
    handle = os.fdopen(descriptor, "r+")
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write("0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _private_directory(path: Path, label: str) -> Path:
    target = Path(os.path.abspath(os.fspath(path)))
    target.mkdir(parents=True, exist_ok=True)
    cursor = target
    while True:
        try:
            metadata = os.lstat(cursor)
        except OSError as exc:
            raise HeldoutProtocolError("{} is missing or unreadable".format(label)) from exc
        if _link_like(metadata):
            raise HeldoutProtocolError("{} path contains a link or reparse point".format(label))
        if cursor == target and not stat.S_ISDIR(metadata.st_mode):
            raise HeldoutProtocolError("{} must be a directory".format(label))
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    return target


def _source_inventory(package_root: Path) -> Tuple[Path, Tuple[Dict[str, Any], ...]]:
    root = Path(os.path.abspath(os.fspath(package_root)))
    if not root.is_absolute():
        raise HeldoutProtocolError("EGV package root must be an explicit absolute path")
    cursor = root
    while True:
        try:
            metadata = os.lstat(cursor)
        except OSError as exc:
            raise HeldoutProtocolError("EGV package root is missing or unreadable") from exc
        if _link_like(metadata):
            raise HeldoutProtocolError("EGV package root contains a link or reparse point")
        if cursor == root and not stat.S_ISDIR(metadata.st_mode):
            raise HeldoutProtocolError("EGV package root must be a directory")
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    records = []
    for directory, names, filenames in os.walk(root, followlinks=False):
        current = Path(directory)
        current_metadata = os.lstat(current)
        if _link_like(current_metadata) or not stat.S_ISDIR(current_metadata.st_mode):
            raise HeldoutProtocolError("EGV package source tree contains an invalid directory")
        for name in names:
            child = current / name
            child_metadata = os.lstat(child)
            if _link_like(child_metadata) or not stat.S_ISDIR(child_metadata.st_mode):
                raise HeldoutProtocolError("EGV package source tree contains a linked directory")
        for name in filenames:
            if not name.endswith(".py"):
                continue
            path = current / name
            raw = _regular_single_link_bytes(path, "EGV package source file")
            relative = path.relative_to(root).as_posix()
            records.append(
                {
                    "path": relative,
                    "sha256": hashlib.sha256(raw).hexdigest(),
                    "bytes": len(raw),
                }
            )
    records.sort(key=lambda item: item["path"])
    if not records or records[0]["path"] != "__init__.py":
        raise HeldoutProtocolError("EGV package source inventory lacks its package initializer")
    return root, tuple(records)


def build_production_source_manifest(package_root: Path, source_commit: str) -> Dict[str, Any]:
    """Bind every importable EGV Python source byte before the router imports EGV."""

    if not isinstance(source_commit, str) or _SOURCE_COMMIT.fullmatch(source_commit) is None:
        raise HeldoutProtocolError("production source commit must be a full lowercase Git SHA")
    _root, records = _source_inventory(package_root)
    unsigned = {
        "schema_version": PRODUCTION_SOURCE_MANIFEST_SCHEMA,
        "source_commit": source_commit,
        "files": list(records),
    }
    return {**unsigned, "source_manifest_digest": digest_for(unsigned)}


def _validate_production_source_manifest(value: Mapping[str, Any], package_root: Path) -> Dict[str, Any]:
    _closed(
        value,
        ("schema_version", "source_commit", "files", "source_manifest_digest"),
        "production source manifest",
    )
    if value["schema_version"] != PRODUCTION_SOURCE_MANIFEST_SCHEMA:
        raise HeldoutProtocolError("production source manifest version is invalid")
    current = build_production_source_manifest(package_root, value["source_commit"])
    if current != dict(value):
        raise HeldoutProtocolError("EGV package source bytes differ from the admitted manifest")
    return current


def _router_config(value: Mapping[str, Any]) -> Dict[str, Any]:
    _closed(value, _ROUTER_CONFIG_FIELDS, "receipt router private configuration")
    if value["schema_version"] != RECEIPT_ROUTER_CONFIG_SCHEMA:
        raise HeldoutProtocolError("receipt router private configuration version is invalid")
    normalized = dict(value)
    path_fields = (
        "service_manifest",
        "evaluator_seed",
        "evaluator_private_key",
        "workspace",
        "state_root",
        "egv_package_root",
        "python_executable",
        "bootstrap_executable",
    )
    for field in path_fields:
        raw = normalized[field]
        if not isinstance(raw, str) or not raw or not Path(raw).is_absolute():
            raise HeldoutProtocolError("receipt router {} must be an explicit absolute path".format(field))
    source_manifest = normalized["egv_source_manifest"]
    if not isinstance(source_manifest, Mapping):
        raise HeldoutProtocolError("receipt router source manifest must be a closed object")
    normalized["egv_source_manifest"] = _validate_production_source_manifest(
        source_manifest,
        Path(normalized["egv_package_root"]),
    )
    roots = normalized["runtime_import_roots"]
    if (
        not isinstance(roots, list)
        or len(set(roots)) != len(roots)
        or any(not isinstance(root, str) or not root or not Path(root).is_absolute() for root in roots)
    ):
        raise HeldoutProtocolError("receipt router runtime import roots must be explicit absolute paths")
    return normalized


def build_production_variation_router_command(config: Mapping[str, Any]) -> bytes:
    """Build the only admitted launcher for the content-addressed receipt router."""

    normalized = _router_config(config)
    encoded = base64.urlsafe_b64encode(canonical_bytes(normalized)).decode("ascii").rstrip("=")
    source = """#!__BOOTSTRAP__ -S __PYTHON__ -I -S
from __future__ import annotations
import base64
import hashlib
import importlib.abc
import importlib.machinery
import importlib.util
import json
import os
from pathlib import Path
import stat
import sys

if not sys.flags.isolated or not sys.flags.no_site:
    raise RuntimeError("receipt router requires isolated no-site Python bootstrap")

CONFIG_B64 = \"__ENCODED__\"
def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(\",\", \":\"), allow_nan=False).encode(\"utf-8\")

def _digest(value):
    return hashlib.sha256(_canonical(value)).hexdigest()

def _source_inventory(root):
    records = []
    cursor = root
    while True:
        metadata = os.lstat(cursor)
        attributes = int(getattr(metadata, \"st_file_attributes\", 0))
        if stat.S_ISLNK(metadata.st_mode) or attributes & 0x400:
            raise RuntimeError(\"EGV package root contains a link or reparse point\")
        if cursor == root and not stat.S_ISDIR(metadata.st_mode):
            raise RuntimeError(\"EGV package root is not a directory\")
        if cursor.parent == cursor:
            break
        cursor = cursor.parent
    for directory, names, filenames in os.walk(root, followlinks=False):
        current = Path(directory)
        for name in names:
            metadata = os.lstat(current / name)
            attributes = int(getattr(metadata, \"st_file_attributes\", 0))
            if stat.S_ISLNK(metadata.st_mode) or attributes & 0x400 or not stat.S_ISDIR(metadata.st_mode):
                raise RuntimeError(\"EGV source tree contains an invalid directory\")
        for name in filenames:
            if not name.endswith(\".py\"):
                continue
            path = current / name
            metadata = os.lstat(path)
            attributes = int(getattr(metadata, \"st_file_attributes\", 0))
            if stat.S_ISLNK(metadata.st_mode) or attributes & 0x400 or not stat.S_ISREG(metadata.st_mode) or int(getattr(metadata, \"st_nlink\", 0)) != 1:
                raise RuntimeError(\"EGV source tree contains an invalid source file\")
            raw_source = path.read_bytes()
            if len(raw_source) != metadata.st_size:
                raise RuntimeError(\"EGV source changed while it was read\")
            records.append({\"path\": path.relative_to(root).as_posix(), \"sha256\": hashlib.sha256(raw_source).hexdigest(), \"bytes\": len(raw_source)})
    return sorted(records, key=lambda item: item[\"path\"])

class _VerifiedSourceLoader(importlib.machinery.SourceFileLoader):
    def __init__(self, fullname, path, expected):
        super().__init__(fullname, str(path))
        self.expected = expected

    def get_code(self, fullname):
        path = Path(self.path)
        metadata = os.lstat(path)
        attributes = int(getattr(metadata, \"st_file_attributes\", 0))
        raw_source = path.read_bytes()
        actual = {\"path\": self.expected[\"path\"], \"sha256\": hashlib.sha256(raw_source).hexdigest(), \"bytes\": len(raw_source)}
        if stat.S_ISLNK(metadata.st_mode) or attributes & 0x400 or not stat.S_ISREG(metadata.st_mode) or int(getattr(metadata, \"st_nlink\", 0)) != 1 or len(raw_source) != metadata.st_size or actual != self.expected:
            raise RuntimeError(\"EGV source changed before compilation\")
        return compile(raw_source, str(path), \"exec\", dont_inherit=True)

class _SourceOnlyEgvFinder(importlib.abc.MetaPathFinder):
    def __init__(self, root, files):
        self.root = root
        self.files = {item[\"path\"]: item for item in files}

    def find_spec(self, fullname, path=None, target=None):
        if fullname != \"egv\" and not fullname.startswith(\"egv.\"):
            return None
        parts = fullname.split(\".\")[1:]
        candidate = self.root.joinpath(*parts)
        package_source = candidate / \"__init__.py\"
        module_source = candidate.with_suffix(\".py\")
        if package_source.is_file():
            relative = package_source.relative_to(self.root).as_posix()
            expected = self.files.get(relative)
            if expected is None:
                raise RuntimeError(\"EGV package import is absent from the admitted source manifest\")
            return importlib.util.spec_from_file_location(fullname, package_source, loader=_VerifiedSourceLoader(fullname, package_source, expected), submodule_search_locations=[str(candidate)])
        if module_source.is_file():
            relative = module_source.relative_to(self.root).as_posix()
            expected = self.files.get(relative)
            if expected is None:
                raise RuntimeError(\"EGV module import is absent from the admitted source manifest\")
            return importlib.util.spec_from_file_location(fullname, module_source, loader=_VerifiedSourceLoader(fullname, module_source, expected))
        raise RuntimeError("EGV import is absent from the admitted source manifest")

config = json.loads(base64.urlsafe_b64decode(CONFIG_B64 + \"=\" * (-len(CONFIG_B64) % 4)))
package_root = Path(config[\"egv_package_root\"])
source_manifest = config[\"egv_source_manifest\"]
unsigned_manifest = {key: value for key, value in source_manifest.items() if key != \"source_manifest_digest\"}
if source_manifest.get(\"schema_version\") != \"__SOURCE_SCHEMA__\" or source_manifest.get(\"source_manifest_digest\") != _digest(unsigned_manifest) or source_manifest.get(\"files\") != _source_inventory(package_root):
    raise RuntimeError(\"EGV package source bytes differ from the admitted manifest\")
if any(name == \"egv\" or name.startswith(\"egv.\") for name in sys.modules):
    raise RuntimeError(\"EGV was imported before source admission\")
sys.path.extend([str(package_root.parent)] + list(config[\"runtime_import_roots\"]))
sys.meta_path.insert(0, _SourceOnlyEgvFinder(package_root, source_manifest[\"files\"]))
from egv.canonical import canonical_bytes
from egv.experiment.production import run_routed_remote_variation_once
if Path(sys.modules[\"egv\"].__file__).parent.resolve() != package_root.resolve():
    raise RuntimeError(\"EGV package import escaped its admitted source root\")
raw = sys.stdin.buffer.read(__LIMIT__ + 1)
if len(raw) > __LIMIT__:
    raise RuntimeError(\"bounded receipt-router request exceeded\")
request = json.loads(raw.decode(\"utf-8\"))
response = run_routed_remote_variation_once(
    request,
    service_manifest=Path(config[\"service_manifest\"]),
    evaluator_seed=Path(config[\"evaluator_seed\"]),
    evaluator_private_key=Path(config[\"evaluator_private_key\"]),
    workspace=Path(config[\"workspace\"]),
    state_root=Path(config[\"state_root\"]),
)
sys.stdout.buffer.write(canonical_bytes(response) + b\"\\n\")
"""
    source = (
        source.replace("__BOOTSTRAP__", shlex.quote(normalized["bootstrap_executable"]))
        .replace("__PYTHON__", shlex.quote(normalized["python_executable"]))
        .replace("__ENCODED__", encoded)
        .replace("__LIMIT__", str(REMOTE_VARIATION_REQUEST_LIMIT))
        .replace("__SOURCE_SCHEMA__", PRODUCTION_SOURCE_MANIFEST_SCHEMA)
    )
    return source.encode("utf-8")


def _router_command_config(command: Path) -> Dict[str, Any]:
    raw = _regular_single_link_bytes(command, "production Variation receipt router command")
    match = re.search(rb'^CONFIG_B64 = "([A-Za-z0-9_-]+)"$', raw, flags=re.MULTILINE)
    if match is None:
        raise HeldoutProtocolError("Variation evaluator command is not the reviewed receipt router")
    try:
        token = match.group(1)
        payload = base64.urlsafe_b64decode(token + b"=" * (-len(token) % 4))
        value = json.loads(payload.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise HeldoutProtocolError("Variation receipt router private configuration is invalid") from exc
    normalized = _router_config(value)
    if build_production_variation_router_command(normalized) != raw:
        raise HeldoutProtocolError("Variation evaluator command differs from the reviewed router bytes")
    return normalized


class ProductionVariationRouterManifest:
    """Path-free attestation for one deterministic evaluator-private router."""

    FIELDS = (
        "schema_version",
        "capability",
        "service_manifest_digest",
        "command_digest",
        "private_config_digest",
        "source_commit",
        "source_manifest_digest",
        "python_executable_digest",
        "bootstrap_executable_digest",
        "routing_manifest_digest",
    )

    def __init__(self, value: Mapping[str, Any]) -> None:
        _closed(value, self.FIELDS, "Variation receipt router manifest")
        unsigned = dict(value)
        supplied = unsigned.pop("routing_manifest_digest")
        if (
            value["schema_version"] != RECEIPT_ROUTER_SCHEMA
            or value["capability"] != RECEIPT_ROUTER_CAPABILITY
            or supplied != digest_for(unsigned)
        ):
            raise HeldoutProtocolError("Variation receipt router manifest digest is invalid")
        for field in (
            "service_manifest_digest",
            "command_digest",
            "private_config_digest",
            "source_manifest_digest",
            "python_executable_digest",
            "bootstrap_executable_digest",
        ):
            validate_sha256(value[field], "Variation receipt router {}".format(field))
        if not isinstance(value["source_commit"], str) or _SOURCE_COMMIT.fullmatch(value["source_commit"]) is None:
            raise HeldoutProtocolError("Variation receipt router source commit is invalid")
        self._value = MappingProxyType(dict(value))

    def __getitem__(self, key: str) -> Any:
        return self._value[key]

    @property
    def digest(self) -> str:
        return str(self._value["routing_manifest_digest"])

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._value)

    @classmethod
    def from_path(cls, path: Path) -> "ProductionVariationRouterManifest":
        return cls(_canonical_object_file(path, "Variation receipt router manifest"))

    @classmethod
    def freeze(
        cls,
        *,
        service_manifest: Path,
        command: Path,
    ) -> "ProductionVariationRouterManifest":
        service = RemoteEvaluatorServiceManifest.from_path(Path(service_manifest))
        config = _router_command_config(Path(command))
        if Path(command).suffix.lower() == ".py":
            raise HeldoutProtocolError("Variation receipt router must use its isolated executable launcher")
        if Path(config["service_manifest"]).resolve() != Path(service_manifest).resolve():
            raise HeldoutProtocolError("Variation receipt router crossed its service manifest path")
        command_digest = hashlib.sha256(
            _regular_single_link_bytes(command, "production Variation receipt router command")
        ).hexdigest()
        if service["command_digest"] != command_digest:
            raise HeldoutProtocolError("Variation receipt router command differs from the evaluator service")
        unsigned = {
            "schema_version": RECEIPT_ROUTER_SCHEMA,
            "capability": RECEIPT_ROUTER_CAPABILITY,
            "service_manifest_digest": service.digest,
            "command_digest": command_digest,
            "private_config_digest": digest_for(config),
            "source_commit": config["egv_source_manifest"]["source_commit"],
            "source_manifest_digest": config["egv_source_manifest"]["source_manifest_digest"],
            "python_executable_digest": hashlib.sha256(
                _regular_single_link_bytes(
                    Path(config["python_executable"]),
                    "Variation receipt router Python executable",
                )
            ).hexdigest(),
            "bootstrap_executable_digest": hashlib.sha256(
                _regular_single_link_bytes(
                    Path(config["bootstrap_executable"]),
                    "Variation receipt router isolated bootstrap executable",
                )
            ).hexdigest(),
        }
        return cls({**unsigned, "routing_manifest_digest": digest_for(unsigned)})

    def admit(self, *, service_manifest: Path, command: Path) -> None:
        expected = type(self).freeze(service_manifest=service_manifest, command=command)
        if expected.to_dict() != self.to_dict():
            raise HeldoutProtocolError("Variation receipt router capability or bytes were substituted")


def _router_state(path: Path, service_manifest_digest: str) -> Dict[str, Any]:
    value = _canonical_object_file(path, "Variation receipt router state snapshot")
    fields = (
        "schema_version",
        "service_manifest_digest",
        "evaluator_key_id",
        "next_sequence",
        "receipt_head",
        "operation_order",
        "responses",
        "pending_operation",
        "state_digest",
    )
    _closed(value, fields, "Variation receipt router state snapshot")
    unsigned = dict(value)
    supplied = unsigned.pop("state_digest")
    if (
        value["schema_version"] != RECEIPT_ROUTER_STATE_SCHEMA
        or value["service_manifest_digest"] != service_manifest_digest
        or supplied != digest_for(unsigned)
        or type(value["next_sequence"]) is not int
        or value["next_sequence"] < 1
    ):
        raise HeldoutProtocolError("Variation receipt router state snapshot is invalid")
    head = value["receipt_head"]
    if head != GENESIS_HASH:
        validate_sha256(head, "Variation receipt router state head")
    return dict(value)


def _router_snapshot_path(root: Path, sequence: int, head: str) -> Path:
    return root / "snapshots" / "{:012d}-{}.json".format(sequence, head)


def _publish_exact_new(path: Path, raw: bytes, label: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("xb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if _regular_single_link_bytes(path, label) != raw:
            raise HeldoutProtocolError("{} conflicts with its content address".format(label)) from None


def _provision_router_operation(
    operations: Path,
    operation_root: Path,
    request_raw: bytes,
    state_raw: Optional[bytes],
) -> None:
    """Publish a complete operation root atomically, or leave it absent."""

    temporary = Path(tempfile.mkdtemp(prefix=".receipt-route-", dir=str(operations)))
    try:
        _publish_exact_new(temporary / "route-request.json", request_raw, "Variation routed request")
        if state_raw is not None:
            _publish_exact_new(
                temporary / "remote-evaluator-state.json",
                state_raw,
                "Variation routed operation state",
            )
        os.replace(str(temporary), str(operation_root))
        if os.name != "nt":
            descriptor = os.open(str(operations), os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
    finally:
        if temporary.exists():
            expected = {"route-request.json", "remote-evaluator-state.json"}
            children = tuple(temporary.iterdir())
            if all(child.name in expected and child.is_file() for child in children):
                for child in children:
                    child.unlink()
                temporary.rmdir()


def run_routed_remote_variation_once(
    request: Mapping[str, Any],
    *,
    service_manifest: Path,
    evaluator_seed: Path,
    evaluator_private_key: Path,
    workspace: Path,
    state_root: Path,
) -> Mapping[str, Any]:
    """Route one evaluator call by its immutable receipt anchor.

    Exact cached PRE operations are shared across policy clones. New operations
    fork from an immutable chain snapshot, so each coordinate can continue its
    own ledger without weakening the evaluator's stale/fork rejection.
    """

    if (
        not isinstance(request, Mapping)
        or set(request) != set(_ROUTED_VARIATION_REQUEST_FIELDS)
        or len(canonical_bytes(request)) > REMOTE_VARIATION_REQUEST_LIMIT
    ):
        raise HeldoutProtocolError("Variation receipt router request is not a bounded object")
    operation = validate_sha256(request.get("operation_digest"), "Variation routed operation")
    request_digest = validate_sha256(request.get("request_digest"), "Variation routed request")
    unsigned_request = dict(request)
    unsigned_request.pop("request_digest")
    if digest_for(unsigned_request) != request_digest:
        raise HeldoutProtocolError("Variation routed request digest is invalid")
    stable_operation = dict(request)
    for field in (
        "request_digest",
        "operation_digest",
        "receipt_sequence_start",
        "previous_receipt_hash",
    ):
        stable_operation.pop(field)
    if digest_for(stable_operation) != operation:
        raise HeldoutProtocolError("Variation routed operation digest is invalid")
    previous = request.get("previous_receipt_hash")
    if previous != GENESIS_HASH:
        previous = validate_sha256(previous, "Variation routed receipt anchor")
    sequence = request.get("receipt_sequence_start")
    if type(sequence) is not int or sequence < 1 or (previous == GENESIS_HASH) != (sequence == 1):
        raise HeldoutProtocolError("Variation receipt router anchor is inconsistent")
    manifest = RemoteEvaluatorServiceManifest.from_path(Path(service_manifest))
    signer = ReceiptSigner(
        _regular_single_link_bytes(evaluator_private_key, "Variation evaluator private key", limit=64)
    )
    if signer.key_id != manifest["evaluator_key_id"]:
        raise HeldoutProtocolError("Variation receipt router private key differs from the service")
    root = _private_directory(state_root, "Variation receipt router root")
    operations = _private_directory(root / "operations", "Variation receipt router operations")
    snapshots = _private_directory(root / "snapshots", "Variation receipt router snapshots")
    del snapshots
    with _exclusive_router_lock(root / ".receipt-router.lock"):
        operation_root = operations / operation
        request_path = operation_root / "route-request.json"
        request_raw = canonical_bytes(dict(request)) + b"\n"
        if operation_root.exists():
            operation_metadata = os.lstat(operation_root)
            if _link_like(operation_metadata) or not stat.S_ISDIR(operation_metadata.st_mode):
                raise HeldoutProtocolError("Variation receipt router operation root is invalid")
            if (
                not request_path.exists()
                or _regular_single_link_bytes(request_path, "Variation routed request") != request_raw
            ):
                raise HeldoutProtocolError("Variation routed operation was requested with different bytes")
        else:
            state_raw = None
            if previous != GENESIS_HASH:
                snapshot_path = _router_snapshot_path(root, sequence, previous)
                state = _router_state(snapshot_path, manifest.digest)
                if state["next_sequence"] != sequence or state["receipt_head"] != previous:
                    raise HeldoutProtocolError("Variation receipt router snapshot crossed its anchor")
                state_raw = _regular_single_link_bytes(
                    snapshot_path,
                    "Variation receipt router snapshot",
                )
            _provision_router_operation(operations, operation_root, request_raw, state_raw)
        remote_lock = operation_root / "remote-evaluator-state.lock"
        if remote_lock.exists():
            remote_lock_metadata = os.lstat(remote_lock)
            if (
                _link_like(remote_lock_metadata)
                or not stat.S_ISREG(remote_lock_metadata.st_mode)
                or int(getattr(remote_lock_metadata, "st_nlink", 0)) != 1
            ):
                raise HeldoutProtocolError("Variation routed evaluator lock is not a private regular file")
        response = run_remote_evaluator_once(
            request,
            service_manifest=service_manifest,
            evaluator_seed=evaluator_seed,
            evaluator_private_key=evaluator_private_key,
            workspace=workspace,
            state_root=operation_root,
        )
        if (
            response.get("operation_digest") != operation
            or response.get("request_digest") != request_digest
            or response.get("service_manifest_digest") != manifest.digest
            or not isinstance(response.get("receipts"), list)
            or not response["receipts"]
        ):
            raise HeldoutProtocolError("Variation receipt router received a stale evaluator response")
        final_receipt = response["receipts"][-1]
        final_sequence = final_receipt.get("sequence")
        if type(final_sequence) is not int or final_sequence < sequence:
            raise HeldoutProtocolError("Variation receipt router received an invalid receipt suffix")
        final_head = receipt_hash(final_receipt)
        state_path = operation_root / "remote-evaluator-state.json"
        state = _router_state(state_path, manifest.digest)
        if state["next_sequence"] != final_sequence + 1 or state["receipt_head"] != final_head:
            raise HeldoutProtocolError("Variation routed state differs from the signed receipt suffix")
        snapshot_path = _router_snapshot_path(root, final_sequence + 1, final_head)
        _publish_exact_new(
            snapshot_path,
            _regular_single_link_bytes(state_path, "Variation routed operation state"),
            "Variation receipt router snapshot",
        )
        return dict(response)


@dataclass(frozen=True)
class ProductionDeploymentPaths:
    """All runtime paths are explicit and excluded from the public manifest."""

    protocol: Path
    trainer_inputs: Path
    trainer_sources: Path
    heldout_service_manifest: Path
    model_root: Path
    adapter_root: Path
    variation_evaluator_manifest: Path
    variation_evaluator_public_key: Path
    variation_evaluator_command: Path
    variation_receipt_router_manifest: Path
    heldout_evaluator_command: Path
    python_executable: Path = Path(sys.executable).resolve()

    def artifact_paths(self) -> Mapping[str, Path]:
        return MappingProxyType(
            {
                "protocol": Path(self.protocol),
                "trainer_inputs": Path(self.trainer_inputs),
                "trainer_sources": Path(self.trainer_sources),
                "heldout_service_manifest": Path(self.heldout_service_manifest),
                "model_manifest": Path(self.model_root) / "model-manifest.json",
                "adapter_manifest": Path(self.adapter_root) / ADAPTER_MANIFEST_NAME,
                "variation_evaluator_manifest": Path(self.variation_evaluator_manifest),
                "variation_evaluator_public_key": Path(self.variation_evaluator_public_key),
                "variation_evaluator_command": Path(self.variation_evaluator_command),
                "variation_receipt_router_manifest": Path(self.variation_receipt_router_manifest),
                "heldout_evaluator_command": Path(self.heldout_evaluator_command),
                "python_executable": Path(self.python_executable),
            }
        )


class SealedHeldoutDeploymentManifest:
    """Path-free binding for all bytes and runtime choices used by a run."""

    FIELDS = (
        "schema_version",
        "integration_name",
        "campaign_id",
        "protocol_digest",
        "trainer_inputs_digest",
        "trainer_sources_digest",
        "base_model_digest",
        "trained_model_digest",
        "adapter_digest",
        "model_revision",
        "evaluator_revision",
        "generation_profile_digest",
        "response_contract_digest",
        "heldout_evaluator_execution_mode",
        "source_commit",
        "source_isolation_scope",
        "device",
        "torch_dtype",
        "max_attempts",
        "max_new_tokens",
        "command_timeout_seconds",
        "request_limit_bytes",
        "response_limit_bytes",
        "artifacts",
        "deployment_manifest_digest",
    )

    def __init__(self, value: Mapping[str, Any]) -> None:
        _closed(value, self.FIELDS, "production deployment manifest")
        unsigned = dict(value)
        supplied = unsigned.pop("deployment_manifest_digest")
        if (
            value["schema_version"] != DEPLOYMENT_MANIFEST_SCHEMA
            or value["integration_name"] != PRODUCTION_INTEGRATION_NAME
            or value["source_isolation_scope"] != SOURCE_ISOLATION_SCOPE
            or value["heldout_evaluator_execution_mode"] != HELDOUT_EVALUATOR_EXECUTION_MODE
            or supplied != digest_for(unsigned)
        ):
            raise HeldoutProtocolError("production deployment manifest digest is invalid")
        for field in (
            "protocol_digest",
            "trainer_inputs_digest",
            "trainer_sources_digest",
            "base_model_digest",
            "trained_model_digest",
            "adapter_digest",
            "generation_profile_digest",
            "response_contract_digest",
        ):
            validate_sha256(value[field], field)
        if (
            value["model_revision"] != MODEL_REVISION
            or not isinstance(value["evaluator_revision"], str)
            or not value["evaluator_revision"]
            or not _SOURCE_COMMIT.fullmatch(str(value["source_commit"]))
        ):
            raise HeldoutProtocolError(
                "production model/evaluator revision or source commit is not immutable"
            )
        if value["device"] not in {"cuda", "cpu"} or value["torch_dtype"] not in {"bfloat16", "float16", "float32"}:
            raise HeldoutProtocolError("production device or tensor dtype is unsupported")
        ints = (
            value["max_attempts"],
            value["max_new_tokens"],
            value["command_timeout_seconds"],
            value["request_limit_bytes"],
            value["response_limit_bytes"],
        )
        if any(type(item) is not int or item <= 0 for item in ints):
            raise HeldoutProtocolError("production runtime bounds must be positive integers")
        if (
            value["max_attempts"] != MAX_CANDIDATE_ATTEMPTS
            or value["max_new_tokens"] > 2048
            or value["command_timeout_seconds"] != REMOTE_VARIATION_TIMEOUT_SECONDS
            or value["request_limit_bytes"] > PRODUCTION_REQUEST_LIMIT
            or value["response_limit_bytes"] > PRODUCTION_RESPONSE_LIMIT
        ):
            raise HeldoutProtocolError("production runtime bounds differ from the reviewed ceiling")
        artifacts = value["artifacts"]
        if not isinstance(artifacts, Mapping) or set(artifacts) != set(ARTIFACT_NAMES):
            raise HeldoutProtocolError("production artifact manifest is not closed")
        normalized_artifacts = {}
        for name in ARTIFACT_NAMES:
            record = artifacts[name]
            _closed(record, ("sha256", "bytes"), "production artifact record")
            validate_sha256(record["sha256"], "artifact sha256")
            if type(record["bytes"]) is not int or record["bytes"] <= 0:
                raise HeldoutProtocolError("production artifact byte count is invalid")
            normalized_artifacts[name] = MappingProxyType(dict(record))
        self._value = MappingProxyType(
            {
                **{key: value[key] for key in self.FIELDS if key not in {"artifacts"}},
                "artifacts": MappingProxyType(normalized_artifacts),
            }
        )

    @property
    def digest(self) -> str:
        return str(self._value["deployment_manifest_digest"])

    def __getitem__(self, key: str) -> Any:
        return self._value[key]

    def to_dict(self) -> Dict[str, Any]:
        return {
            **{key: self._value[key] for key in self.FIELDS if key != "artifacts"},
            "artifacts": {name: dict(record) for name, record in self._value["artifacts"].items()},
        }

    @classmethod
    def from_path(cls, path: Path) -> "SealedHeldoutDeploymentManifest":
        return cls(_canonical_object_file(path, "production deployment manifest"))

    @classmethod
    def freeze(
        cls,
        *,
        protocol: FrozenHeldoutProtocol,
        trainer_inputs: HeldoutTrainerInputs,
        trainer_sources: HeldoutTrainerSources,
        paths: ProductionDeploymentPaths,
        source_commit: str,
        device: str = "cuda",
        torch_dtype: str = "bfloat16",
        max_new_tokens: int = 512,
    ) -> "SealedHeldoutDeploymentManifest":
        protocol.validate_current()
        trainer_inputs.validate_retained(protocol)
        trainer_sources.validate_retained(trainer_inputs)
        artifacts = {
            name: _artifact_record(path, "production artifact {}".format(name))
            for name, path in paths.artifact_paths().items()
        }
        try:
            variation = RemoteEvaluatorServiceManifest.from_path(
                Path(paths.variation_evaluator_manifest)
            )
        except VariationConfigurationError as exc:
            raise HeldoutProtocolError(
                "production Variation evaluator manifest is invalid"
            ) from exc
        unsigned: Dict[str, Any] = {
            "schema_version": DEPLOYMENT_MANIFEST_SCHEMA,
            "integration_name": PRODUCTION_INTEGRATION_NAME,
            "campaign_id": protocol.campaign_id,
            "protocol_digest": protocol.digest,
            "trainer_inputs_digest": trainer_inputs.digest,
            "trainer_sources_digest": trainer_sources.digest,
            "base_model_digest": protocol.bindings["base_model_digest"],
            "trained_model_digest": protocol.bindings["trained_model_digest"],
            "adapter_digest": protocol.bindings["adapter_digest"],
            "model_revision": MODEL_REVISION,
            "evaluator_revision": variation["evaluator_revision"],
            "generation_profile_digest": trainer_inputs.generation_profile_digest,
            "response_contract_digest": SOURCE_ONLY_RESPONSE_CONTRACT_DIGEST,
            "heldout_evaluator_execution_mode": HELDOUT_EVALUATOR_EXECUTION_MODE,
            "source_commit": source_commit,
            "source_isolation_scope": SOURCE_ISOLATION_SCOPE,
            "device": device,
            "torch_dtype": torch_dtype,
            "max_attempts": MAX_CANDIDATE_ATTEMPTS,
            "max_new_tokens": max_new_tokens,
            "command_timeout_seconds": REMOTE_VARIATION_TIMEOUT_SECONDS,
            "request_limit_bytes": PRODUCTION_REQUEST_LIMIT,
            "response_limit_bytes": PRODUCTION_RESPONSE_LIMIT,
            "artifacts": artifacts,
        }
        return cls({**unsigned, "deployment_manifest_digest": digest_for(unsigned)})

    def admit(
        self,
        *,
        protocol: FrozenHeldoutProtocol,
        trainer_inputs: HeldoutTrainerInputs,
        trainer_sources: HeldoutTrainerSources,
        paths: ProductionDeploymentPaths,
    ) -> None:
        protocol.validate_current()
        trainer_inputs.validate_retained(protocol)
        trainer_sources.validate_retained(trainer_inputs)
        expected = {
            "campaign_id": protocol.campaign_id,
            "protocol_digest": protocol.digest,
            "trainer_inputs_digest": trainer_inputs.digest,
            "trainer_sources_digest": trainer_sources.digest,
            "base_model_digest": protocol.bindings["base_model_digest"],
            "trained_model_digest": protocol.bindings["trained_model_digest"],
            "adapter_digest": protocol.bindings["adapter_digest"],
            "generation_profile_digest": trainer_inputs.generation_profile_digest,
            "response_contract_digest": SOURCE_ONLY_RESPONSE_CONTRACT_DIGEST,
        }
        for field, expected_value in expected.items():
            if self[field] != expected_value:
                raise HeldoutProtocolError("production deployment {} binding mismatch".format(field))
        for name, path in paths.artifact_paths().items():
            if _artifact_record(path, "production artifact {}".format(name)) != dict(self["artifacts"][name]):
                raise HeldoutProtocolError("production artifact {} was substituted".format(name))
        try:
            same_python = os.path.samefile(paths.python_executable, sys.executable)
        except OSError as exc:
            raise HeldoutProtocolError("production Python executable identity is unavailable") from exc
        if not same_python:
            raise HeldoutProtocolError(
                "production Python executable differs from the interpreter used by evaluator commands"
            )
        model_manifest, _hashes = PinnedModelLoader(Path(paths.model_root)).verify_manifest()
        if model_manifest.digest() != protocol.bindings["base_model_digest"]:
            raise HeldoutProtocolError("pinned Qwen model manifest differs from the frozen base model")
        adapter = SealedAdapterArtifact(Path(paths.adapter_root))
        adapter.verify()
        if adapter.digest != protocol.bindings["adapter_digest"]:
            raise HeldoutProtocolError("sealed adapter differs from the frozen adapter")
        heldout_service = HeldoutVerifierServiceManifest.load(
            _canonical_object_file(paths.heldout_service_manifest, "held-out service manifest"),
            protocol,
        )
        if heldout_service.digest != digest_for(
            {key: value for key, value in heldout_service.to_dict().items() if key != "manifest_digest"}
        ):
            raise HeldoutProtocolError("held-out verifier service identity is inconsistent")
        variation = RemoteEvaluatorServiceManifest.from_path(Path(paths.variation_evaluator_manifest))
        router = ProductionVariationRouterManifest.from_path(Path(paths.variation_receipt_router_manifest))
        router.admit(
            service_manifest=Path(paths.variation_evaluator_manifest),
            command=Path(paths.variation_evaluator_command),
        )
        if router["source_commit"] != self["source_commit"]:
            raise HeldoutProtocolError("Variation receipt router source commit crossed the deployment")
        if router["python_executable_digest"] != self["artifacts"]["python_executable"]["sha256"]:
            raise HeldoutProtocolError("Variation receipt router Python differs from the deployment")
        expected_variation = {
            "campaign_id": protocol.campaign_id,
            "model_digest": protocol.bindings["base_model_digest"],
            "protocol_digest": VARIATION_PROTOCOL_DIGEST,
            "policy_digest": protocol.bindings["policy_manifest_digest"],
            "data_manifest_digest": protocol.bindings["data_manifest_digest"],
        }
        for field, expected_value in expected_variation.items():
            if variation[field] != expected_value:
                raise HeldoutProtocolError("remote Variation evaluator {} binding mismatch".format(field))
        if variation.digest != protocol.bindings["evaluator_digest"]:
            raise HeldoutProtocolError("remote Variation evaluator digest differs from the held-out protocol")
        if variation["evaluator_revision"] != self["evaluator_revision"]:
            raise HeldoutProtocolError(
                "remote Variation evaluator revision differs from the sealed deployment"
            )
        for record in protocol.heldout_task_records:
            if variation.public_record(record["template_id"]) != dict(record):
                raise HeldoutProtocolError("remote Variation evaluator lacks an exact held-out task binding")


def _preserve_process_cleanup_context(
    error: BaseException, cleanup_errors: Sequence[BaseException]
) -> None:
    """Attach every secondary process-cleanup failure to its primary error."""

    if cleanup_errors:
        existing = tuple(getattr(error, "cleanup_context", ()))
        flattened = []
        for cleanup_error in cleanup_errors:
            flattened.append(str(cleanup_error))
            flattened.extend(tuple(getattr(cleanup_error, "cleanup_context", ())))
        setattr(error, "cleanup_context", existing + tuple(flattened))


def _close_windows_native_handle(close_handle: Any, handle: Any, label: str) -> None:
    """Close one owned native handle and prove that ownership was released."""

    if not close_handle(handle):
        import ctypes

        native_error = ctypes.WinError(ctypes.get_last_error())
        raise OSError("{} close failed: {}".format(label, native_error)) from native_error


def _close_windows_native_handles(
    close_handle: Any,
    handles: Sequence[Tuple[Any, str]],
) -> Tuple[BaseException, ...]:
    """Attempt every owned native-handle close and retain every failure."""

    cleanup_errors = []
    for handle, label in handles:
        if handle:
            try:
                _close_windows_native_handle(close_handle, handle, label)
            except BaseException as exc:
                cleanup_errors.append(exc)
    return tuple(cleanup_errors)


def _assign_kill_on_close_job(process: subprocess.Popen[bytes]) -> int:
    """Place one Windows subprocess tree in a kill-on-close Job Object."""

    if os.name != "nt":
        raise OSError("Windows Job Objects are unavailable on this platform")
    import ctypes
    from ctypes import wintypes

    class _BasicLimitInformation(ctypes.Structure):
        _fields_ = (
            ("PerProcessUserTimeLimit", ctypes.c_longlong),
            ("PerJobUserTimeLimit", ctypes.c_longlong),
            ("LimitFlags", wintypes.DWORD),
            ("MinimumWorkingSetSize", ctypes.c_size_t),
            ("MaximumWorkingSetSize", ctypes.c_size_t),
            ("ActiveProcessLimit", wintypes.DWORD),
            ("Affinity", ctypes.c_size_t),
            ("PriorityClass", wintypes.DWORD),
            ("SchedulingClass", wintypes.DWORD),
        )

    class _IoCounters(ctypes.Structure):
        _fields_ = (
            ("ReadOperationCount", ctypes.c_ulonglong),
            ("WriteOperationCount", ctypes.c_ulonglong),
            ("OtherOperationCount", ctypes.c_ulonglong),
            ("ReadTransferCount", ctypes.c_ulonglong),
            ("WriteTransferCount", ctypes.c_ulonglong),
            ("OtherTransferCount", ctypes.c_ulonglong),
        )

    class _ExtendedLimitInformation(ctypes.Structure):
        _fields_ = (
            ("BasicLimitInformation", _BasicLimitInformation),
            ("IoInfo", _IoCounters),
            ("ProcessMemoryLimit", ctypes.c_size_t),
            ("JobMemoryLimit", ctypes.c_size_t),
            ("PeakProcessMemoryUsed", ctypes.c_size_t),
            ("PeakJobMemoryUsed", ctypes.c_size_t),
        )

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateJobObjectW.argtypes = (ctypes.c_void_p, wintypes.LPCWSTR)
    kernel32.CreateJobObjectW.restype = wintypes.HANDLE
    kernel32.SetInformationJobObject.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        ctypes.c_void_p,
        wintypes.DWORD,
    )
    kernel32.SetInformationJobObject.restype = wintypes.BOOL
    kernel32.AssignProcessToJobObject.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    kernel32.AssignProcessToJobObject.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    job = kernel32.CreateJobObjectW(None, None)
    if not job:
        raise ctypes.WinError(ctypes.get_last_error())
    info = _ExtendedLimitInformation()
    info.BasicLimitInformation.LimitFlags = 0x00002000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
    if not kernel32.SetInformationJobObject(job, 9, ctypes.byref(info), ctypes.sizeof(info)):
        primary = ctypes.WinError(ctypes.get_last_error())
        try:
            _close_windows_native_handle(kernel32.CloseHandle, job, "pinned evaluator Job Object")
        except BaseException as cleanup_error:
            _preserve_process_cleanup_context(primary, (cleanup_error,))
        raise primary
    process_handle = wintypes.HANDLE(int(process._handle))  # type: ignore[attr-defined]
    if not kernel32.AssignProcessToJobObject(job, process_handle):
        primary = ctypes.WinError(ctypes.get_last_error())
        try:
            _close_windows_native_handle(kernel32.CloseHandle, job, "pinned evaluator Job Object")
        except BaseException as cleanup_error:
            _preserve_process_cleanup_context(primary, (cleanup_error,))
        raise primary
    return int(job)


def _terminate_process_tree(process: subprocess.Popen[bytes], windows_job: Optional[int]) -> None:
    """Terminate the complete admitted subprocess tree without an ambient shell."""

    if os.name == "nt":
        if windows_job is not None:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.TerminateJobObject.argtypes = (wintypes.HANDLE, wintypes.UINT)
            kernel32.TerminateJobObject.restype = wintypes.BOOL
            if not kernel32.TerminateJobObject(wintypes.HANDLE(windows_job), 1):
                raise ctypes.WinError(ctypes.get_last_error())
        elif process.poll() is None:
            process.kill()
        return
    if not all(
        hasattr(os, name)
        for name in ("waitid", "P_PID", "WEXITED", "WNOHANG", "WNOWAIT")
    ):
        raise OSError("pid-safe pinned-command cleanup anchor is unavailable")
    try:
        os.waitid(  # type: ignore[attr-defined]
            os.P_PID,  # type: ignore[attr-defined]
            process.pid,
            os.WEXITED | os.WNOHANG | os.WNOWAIT,  # type: ignore[attr-defined]
        )
    except ChildProcessError as exc:
        raise OSError("pinned-command leader anchor was lost before group cleanup") from exc
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _close_windows_job(handle: int) -> None:
    """Close a Windows Job Object, terminating any surviving descendants."""

    if os.name != "nt":
        return
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    _close_windows_native_handle(
        kernel32.CloseHandle,
        wintypes.HANDLE(handle),
        "pinned evaluator Job Object",
    )


def _resume_windows_process(process_id: int) -> None:
    """Resume the sole primary thread of a newly suspended Windows process."""

    if os.name != "nt":
        raise OSError("Windows thread control is unavailable on this platform")
    import ctypes
    from ctypes import wintypes

    class _ThreadEntry32(ctypes.Structure):
        _fields_ = (
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ThreadID", wintypes.DWORD),
            ("th32OwnerProcessID", wintypes.DWORD),
            ("tpBasePri", wintypes.LONG),
            ("tpDeltaPri", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
        )

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateToolhelp32Snapshot.argtypes = (wintypes.DWORD, wintypes.DWORD)
    kernel32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    kernel32.Thread32First.argtypes = (wintypes.HANDLE, ctypes.POINTER(_ThreadEntry32))
    kernel32.Thread32First.restype = wintypes.BOOL
    kernel32.Thread32Next.argtypes = (wintypes.HANDLE, ctypes.POINTER(_ThreadEntry32))
    kernel32.Thread32Next.restype = wintypes.BOOL
    kernel32.OpenThread.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    kernel32.OpenThread.restype = wintypes.HANDLE
    kernel32.ResumeThread.argtypes = (wintypes.HANDLE,)
    kernel32.ResumeThread.restype = wintypes.DWORD
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    snapshot = kernel32.CreateToolhelp32Snapshot(0x00000004, 0)
    invalid = ctypes.c_void_p(-1).value
    if not snapshot or int(snapshot) == invalid:
        raise ctypes.WinError(ctypes.get_last_error())
    thread_handle = None
    primary_error: Optional[BaseException] = None
    try:
        entry = _ThreadEntry32()
        entry.dwSize = ctypes.sizeof(entry)
        present = kernel32.Thread32First(snapshot, ctypes.byref(entry))
        owner_thread_ids = []
        while present:
            if int(entry.th32OwnerProcessID) == process_id:
                owner_thread_ids.append(int(entry.th32ThreadID))
            present = kernel32.Thread32Next(snapshot, ctypes.byref(entry))
        if len(owner_thread_ids) != 1:
            raise OSError("suspended pinned command does not have exactly one primary thread")
        thread_handle = kernel32.OpenThread(0x0002, False, owner_thread_ids[0])
        if not thread_handle:
            raise ctypes.WinError(ctypes.get_last_error())
        previous_count = int(kernel32.ResumeThread(thread_handle))
        if previous_count == 0xFFFFFFFF:
            raise ctypes.WinError(ctypes.get_last_error())
        if previous_count != 1:
            raise OSError("suspended pinned command primary thread has an invalid suspend count")
    except BaseException as exc:
        primary_error = exc
    cleanup_errors = _close_windows_native_handles(
        kernel32.CloseHandle,
        (
            (thread_handle, "pinned evaluator primary thread"),
            (snapshot, "pinned evaluator thread snapshot"),
        ),
    )
    if primary_error is not None:
        _preserve_process_cleanup_context(primary_error, cleanup_errors)
        raise primary_error
    if cleanup_errors:
        error = OSError("pinned evaluator native handle cleanup failed")
        _preserve_process_cleanup_context(error, cleanup_errors)
        raise error from cleanup_errors[0]


def _cleanup_process_before_containment(
    process: subprocess.Popen[bytes], *, deadline: float
) -> Tuple[BaseException, ...]:
    """Clean a process created before its identity/Job containment completed."""

    cleanup_errors = []
    try:
        if os.name == "nt":
            # The process is still suspended and unassigned. Popen.kill uses
            # the owned process handle, rather than a reusable ambient PID.
            process.kill()
        else:
            _terminate_process_tree(process, None)
    except BaseException as exc:
        cleanup_errors.append(exc)
    try:
        process.wait(timeout=max(0.0, deadline - monotonic()))
    except BaseException as exc:
        cleanup_errors.append(exc)
    for stream in (process.stdin, process.stdout, process.stderr):
        if stream is not None:
            try:
                stream.close()
            except BaseException as exc:
                cleanup_errors.append(exc)
    return tuple(cleanup_errors)


class DigestPinnedJsonExecutor:
    """Execute one immutable command with strict input, output, time, and stderr bounds."""

    def __init__(
        self,
        command: Path,
        *,
        command_digest: str,
        python_executable: Path,
        python_digest: str,
        timeout_seconds: int = REMOTE_VARIATION_TIMEOUT_SECONDS,
        request_limit: int = PRODUCTION_REQUEST_LIMIT,
        response_limit: int = PRODUCTION_RESPONSE_LIMIT,
        execution_mode: str = HELDOUT_EVALUATOR_EXECUTION_MODE,
    ) -> None:
        self.command = Path(command).resolve()
        self.python_executable = Path(python_executable).resolve()
        self.command_digest = validate_sha256(command_digest, "command digest")
        self.python_digest = validate_sha256(python_digest, "python executable digest")
        self.timeout_seconds = timeout_seconds
        self.request_limit = request_limit
        self.response_limit = response_limit
        if execution_mode != HELDOUT_EVALUATOR_EXECUTION_MODE:
            raise HeldoutProtocolError("pinned executor execution mode is unsupported")
        self.execution_mode = execution_mode
        self._command_bytes = _regular_single_link_bytes(self.command, "pinned JSON command")
        if hashlib.sha256(self._command_bytes).hexdigest() != self.command_digest:
            raise HeldoutProtocolError("pinned JSON command digest mismatch")
        if (
            hashlib.sha256(_regular_single_link_bytes(self.python_executable, "pinned Python executable")).hexdigest()
            != self.python_digest
        ):
            raise HeldoutProtocolError("pinned Python executable digest mismatch")
        if (
            type(timeout_seconds) is not int
            or timeout_seconds <= 0
            or timeout_seconds > REMOTE_VARIATION_TIMEOUT_SECONDS
            or type(request_limit) is not int
            or not 0 < request_limit <= PRODUCTION_REQUEST_LIMIT
            or type(response_limit) is not int
            or not 0 < response_limit <= PRODUCTION_RESPONSE_LIMIT
        ):
            raise HeldoutProtocolError("pinned executor bounds exceed the reviewed ceiling")

    def __call__(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        if not isinstance(request, Mapping):
            raise HeldoutProtocolError("pinned executor request must be an object")
        command_bytes = _regular_single_link_bytes(self.command, "pinned JSON command")
        python_bytes = _regular_single_link_bytes(self.python_executable, "pinned Python executable")
        if hashlib.sha256(command_bytes).hexdigest() != self.command_digest or command_bytes != self._command_bytes:
            raise HeldoutProtocolError("pinned JSON command changed after admission")
        if hashlib.sha256(python_bytes).hexdigest() != self.python_digest:
            raise HeldoutProtocolError("pinned Python executable changed after admission")
        request_bytes = canonical_bytes(request)
        if len(request_bytes) > self.request_limit:
            raise HeldoutProtocolError("pinned executor request exceeds its byte limit")
        with tempfile.TemporaryDirectory(prefix="egv-heldout-command-") as directory:
            endpoint = Path(directory) / self.command_digest
            with endpoint.open("xb") as handle:
                handle.write(command_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            endpoint.chmod(0o500)
            if hashlib.sha256(endpoint.read_bytes()).hexdigest() != self.command_digest:
                raise HeldoutProtocolError("content-addressed command copy failed verification")
            hard_deadline = monotonic() + self.timeout_seconds
            execution_deadline = hard_deadline
            popen_kwargs: Dict[str, Any] = {}
            if os.name == "nt":
                popen_kwargs["creationflags"] = (
                    subprocess.CREATE_NEW_PROCESS_GROUP | _WINDOWS_CREATE_SUSPENDED
                )
            else:
                popen_kwargs["start_new_session"] = True

            def start_process(invocation: Sequence[str], identity_kwargs: Mapping[str, Any]) -> subprocess.Popen:
                options = {**popen_kwargs, **dict(identity_kwargs)}
                return subprocess.Popen(
                    list(invocation),
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env={},
                    bufsize=0,
                    **options,
                )

            process: Optional[subprocess.Popen[bytes]] = None
            try:
                with pinned_python_invocation(self.python_executable, self.python_digest) as (
                    pinned_python,
                    identity_kwargs,
                ):
                    process = start_process([pinned_python, str(endpoint)], identity_kwargs)
            except BaseException as exc:
                if process is not None:
                    cleanup_errors = _cleanup_process_before_containment(
                        process,
                        deadline=monotonic() + 5,
                    )
                    if isinstance(exc, (OSError, VariationDependencyError)):
                        error = HeldoutProtocolError(
                            "pinned evaluator command could not complete verified startup"
                        )
                        _preserve_process_cleanup_context(error, cleanup_errors)
                        raise error from exc
                    _preserve_process_cleanup_context(exc, cleanup_errors)
                    raise
                if isinstance(exc, (OSError, VariationDependencyError)):
                    raise HeldoutProtocolError("pinned evaluator command could not start") from exc
                raise
            assert process is not None
            windows_job: Optional[int] = None
            if os.name == "nt":
                try:
                    windows_job = _assign_kill_on_close_job(process)
                    _resume_windows_process(process.pid)
                except OSError as exc:
                    startup_cleanup_errors: list[BaseException] = []
                    try:
                        if windows_job is not None:
                            _terminate_process_tree(process, windows_job)
                        else:
                            process.kill()
                    except BaseException as cleanup_exc:
                        startup_cleanup_errors.append(cleanup_exc)
                    try:
                        process.wait(timeout=max(0.001, hard_deadline - monotonic()))
                    except BaseException as cleanup_exc:
                        startup_cleanup_errors.append(cleanup_exc)
                    finally:
                        for stream in (process.stdin, process.stdout, process.stderr):
                            if stream is not None:
                                try:
                                    stream.close()
                                except BaseException as cleanup_exc:
                                    startup_cleanup_errors.append(cleanup_exc)
                        if windows_job is not None:
                            try:
                                _close_windows_job(windows_job)
                            except BaseException as cleanup_exc:
                                startup_cleanup_errors.append(cleanup_exc)
                    if startup_cleanup_errors:
                        error = HeldoutProtocolError(
                            "pinned evaluator failed-start cleanup did not complete"
                        )
                        _preserve_process_cleanup_context(error, startup_cleanup_errors)
                        raise error from exc
                    raise HeldoutProtocolError(
                        "pinned evaluator command could not enter its bounded process tree"
                    ) from exc
            stdout = bytearray()
            stderr = bytearray()
            overflow: list[str] = []
            writer_errors: list[BaseException] = []
            reader_errors: list[BaseException] = []
            cleanup_errors: list[BaseException] = []
            request_complete = threading.Event()

            def drain(stream: Any, sink: bytearray, limit: int, label: str) -> None:
                try:
                    while True:
                        chunk = stream.read(65536)
                        if not chunk:
                            return
                        if len(sink) + len(chunk) > limit:
                            overflow.append(label)
                            return
                        sink.extend(chunk)
                except BaseException as exc:
                    reader_errors.append(exc)

            def write_request() -> None:
                try:
                    assert process.stdin is not None
                    written = process.stdin.write(request_bytes)
                    process.stdin.flush()
                    if written != len(request_bytes):
                        raise OSError("pinned evaluator request write was incomplete")
                    request_complete.set()
                except BaseException as exc:
                    writer_errors.append(exc)
                finally:
                    if process.stdin is not None:
                        try:
                            process.stdin.close()
                        except BaseException as exc:
                            writer_errors.append(exc)

            startup_timed_out = monotonic() >= execution_deadline
            if startup_timed_out:
                readers: Tuple[threading.Thread, ...] = ()
                workers: Tuple[threading.Thread, ...] = ()
            else:
                readers = (
                    threading.Thread(
                        target=drain, args=(process.stdout, stdout, self.response_limit, "stdout"), daemon=True
                    ),
                    threading.Thread(
                        target=drain, args=(process.stderr, stderr, PRODUCTION_STDERR_LIMIT, "stderr"), daemon=True
                    ),
                )
                writer = threading.Thread(target=write_request, daemon=True)
                workers = (writer, *readers)
                for worker in workers:
                    worker.start()

            primary_error: Optional[HeldoutProtocolError] = None
            try:
                while not startup_timed_out and not overflow and not writer_errors and not reader_errors:
                    if monotonic() >= execution_deadline:
                        primary_error = HeldoutProtocolError("pinned evaluator command timed out")
                        break
                    if _bounded_process_exited_without_reap(process):
                        break
                    threading.Event().wait(min(0.05, max(0.0, execution_deadline - monotonic())))
            except VariationDependencyError as exc:
                primary_error = HeldoutProtocolError("pinned evaluator process exit could not be observed safely")
                primary_error.__cause__ = exc
            if startup_timed_out:
                primary_error = HeldoutProtocolError("pinned evaluator command timed out")

            # Kill the group/Job before reaping its leader. A valid parent may
            # leave descendants holding inherited pipes; those descendants do
            # not turn a complete canonical response into a false timeout.
            cleanup_deadline = monotonic() + 5
            try:
                _terminate_process_tree(process, windows_job)
            except BaseException as exc:
                cleanup_errors.append(exc)
            try:
                process.wait(timeout=max(0.0, cleanup_deadline - monotonic()))
            except (OSError, subprocess.TimeoutExpired) as exc:
                cleanup_errors.append(exc)
            for worker in workers:
                worker.join(timeout=max(0.0, cleanup_deadline - monotonic()))
            if any(worker.is_alive() for worker in workers):
                cleanup_errors.append(OSError("pinned evaluator I/O worker remained alive"))
            streams = (
                (process.stdout, readers[0] if readers else None),
                (process.stderr, readers[1] if readers else None),
            )
            for stream, worker in streams:
                if stream is not None and (worker is None or not worker.is_alive()):
                    try:
                        stream.close()
                    except BaseException as exc:
                        cleanup_errors.append(exc)
            if startup_timed_out and process.stdin is not None:
                try:
                    process.stdin.close()
                except BaseException as exc:
                    cleanup_errors.append(exc)
            if windows_job is not None:
                try:
                    _close_windows_job(windows_job)
                except BaseException as exc:
                    cleanup_errors.append(exc)

            if primary_error is None:
                if overflow:
                    primary_error = HeldoutProtocolError(
                        "pinned evaluator {} exceeded its byte limit".format(overflow[0])
                    )
                elif writer_errors or not request_complete.is_set():
                    primary_error = HeldoutProtocolError(
                        "pinned evaluator command did not consume its complete request"
                    )
                elif reader_errors:
                    primary_error = HeldoutProtocolError("pinned evaluator response read did not complete")
                elif process.returncode != 0:
                    primary_error = HeldoutProtocolError("pinned evaluator command returned a nonzero status")
            if primary_error is not None:
                if cleanup_errors:
                    _preserve_process_cleanup_context(primary_error, cleanup_errors)
                raise primary_error
            if cleanup_errors:
                error = HeldoutProtocolError(
                    "pinned evaluator process-tree cleanup failed"
                )
                _preserve_process_cleanup_context(error, cleanup_errors)
                raise error from cleanup_errors[0]
        try:
            response = json.loads(bytes(stdout).decode("utf-8"))
        except (UnicodeError, ValueError) as exc:
            raise HeldoutProtocolError("pinned evaluator command returned invalid JSON") from exc
        if not isinstance(response, Mapping) or canonical_bytes(response) + b"\n" != bytes(stdout):
            raise HeldoutProtocolError("pinned evaluator response must be one canonical JSON object")
        return response


class ProductionQwenLoopBuilders:
    """Load exact base/adapted Qwen identities and build real production loops."""

    def __init__(
        self,
        *,
        protocol: FrozenHeldoutProtocol,
        trainer_inputs: HeldoutTrainerInputs,
        deployment: SealedHeldoutDeploymentManifest,
        paths: ProductionDeploymentPaths,
    ) -> None:
        deployment.admit(
            protocol=protocol,
            trainer_inputs=trainer_inputs,
            trainer_sources=HeldoutTrainerSources.from_path(paths.trainer_sources, trainer_inputs=trainer_inputs),
            paths=paths,
        )
        self.protocol = protocol
        self.trainer_inputs = trainer_inputs
        self.deployment = deployment
        self.paths = paths
        self._base_generator: Optional[ModelCandidateGenerator] = None
        self._adapter_generator: Optional[ModelCandidateGenerator] = None

    def _dtype(self) -> Any:
        try:
            import torch
        except ImportError as exc:
            raise HeldoutProtocolError("production Qwen runtime requires torch") from exc
        return getattr(torch, str(self.deployment["torch_dtype"]))

    def load(self) -> Tuple[ModelCandidateGenerator, ModelCandidateGenerator]:
        if self._base_generator is not None and self._adapter_generator is not None:
            self._base_generator.validate_production_integrity()
            self._adapter_generator.validate_production_integrity()
            return self._base_generator, self._adapter_generator
        loader = PinnedModelLoader(Path(self.paths.model_root))
        adapter = SealedAdapterArtifact(Path(self.paths.adapter_root))
        dtype = self._dtype()
        base = loader.load(device=self.deployment["device"], torch_dtype=dtype)
        adapted = loader.load(
            device=self.deployment["device"],
            torch_dtype=dtype,
            adapter_artifact=adapter,
        )
        if base.manifest_digest != self.protocol.bindings["base_model_digest"]:
            raise HeldoutProtocolError("loaded base Qwen identity differs from the frozen protocol")
        attestation = adapted.adapter_attestation
        if (
            adapted.adapter_digest != self.protocol.bindings["adapter_digest"]
            or attestation is None
            or attestation.applied_model_state_digest != self.protocol.bindings["trained_model_digest"]
        ):
            raise HeldoutProtocolError("loaded adapted Qwen identity differs from the trained protocol binding")
        base_generator = ModelCandidateGenerator(
            base,
            model_digest=self.protocol.bindings["base_model_digest"],
            max_new_tokens=self.deployment["max_new_tokens"],
            response_contract="source-only-v1",
        )
        adapter_generator = ModelCandidateGenerator(
            adapted,
            model_digest=self.protocol.bindings["base_model_digest"],
            adapter_digest=self.protocol.bindings["adapter_digest"],
            adapter_artifact=adapter,
            max_new_tokens=self.deployment["max_new_tokens"],
            response_contract="source-only-v1",
        )
        for generator in (base_generator, adapter_generator):
            generator.validate_production_integrity()
            if (
                generator.generation_profile_digest != self.trainer_inputs.generation_profile_digest
                or generator.prompt_manifest_digest != self.protocol.bindings["prompt_manifest_digest"]
            ):
                raise HeldoutProtocolError("Qwen prompt or decoding profile differs from the frozen protocol")
        self._base_generator = base_generator
        self._adapter_generator = adapter_generator
        return base_generator, adapter_generator

    @property
    def tokenizer(self) -> Any:
        return self.load()[0].tokenizer

    def _build(self, *, adapter: bool, **kwargs: Any) -> BoundedCandidateLoop:
        base_generator, adapter_generator = self.load()
        coordinate = kwargs["coordinate"]
        if not isinstance(coordinate, HeldoutCoordinate):
            raise HeldoutProtocolError("production loop builder received an invalid coordinate")
        generator = adapter_generator if adapter else base_generator
        artifact = SealedAdapterArtifact(Path(self.paths.adapter_root)) if adapter else None
        evaluator = RemoteControllerEvaluationGateway(
            ledger=kwargs["ledger"],
            manifest_path=Path(self.paths.variation_evaluator_manifest),
            public_key_path=Path(self.paths.variation_evaluator_public_key),
            command=Path(self.paths.variation_evaluator_command),
            python_executable=Path(self.paths.python_executable),
            python_digest=self.deployment["artifacts"]["python_executable"]["sha256"],
        )
        evaluator.validate_campaign_bindings(
            campaign_id=self.protocol.campaign_id,
            model_digest=self.protocol.bindings["base_model_digest"],
            protocol_digest=VARIATION_PROTOCOL_DIGEST,
            policy_digest=self.protocol.bindings["policy_manifest_digest"],
            data_manifest_digest=self.protocol.bindings["data_manifest_digest"],
        )
        if evaluator.evaluator_digest != self.protocol.bindings["evaluator_digest"]:
            raise HeldoutProtocolError("production evaluator identity differs from the frozen coordinate")
        return BoundedCandidateLoop(
            ledger=kwargs["ledger"],
            evaluator=evaluator,
            generator=generator,
            isolation=kwargs["isolation"],
            workspace_root=kwargs["isolation"].root,
            campaign_id=self.protocol.campaign_id,
            source_commit=self.deployment["source_commit"],
            model_revision=MODEL_REVISION,
            model_digest=self.protocol.bindings["base_model_digest"],
            data_manifest_digest=self.protocol.bindings["data_manifest_digest"],
            policy_digest=self.protocol.bindings["policy_manifest_digest"],
            arm_id=coordinate.treatment,
            max_attempts=self.deployment["max_attempts"],
            adapter_digest=self.protocol.bindings["adapter_digest"] if adapter else None,
            adapter_artifact=artifact,
            seed_set=self.protocol.seeds,
            private_store=kwargs["private_store"],
            initial_source=kwargs["initial_source"],
            response_contract_digest=SOURCE_ONLY_RESPONSE_CONTRACT_DIGEST,
            generation_profile_digest=self.trainer_inputs.generation_profile_digest,
        )

    def base(self, **kwargs: Any) -> BoundedCandidateLoop:
        if arm_policy(kwargs["coordinate"].treatment).requires_adapter:
            raise HeldoutProtocolError("base builder cannot execute an adapter arm")
        return self._build(adapter=False, **kwargs)

    def adapted(self, **kwargs: Any) -> BoundedCandidateLoop:
        if not arm_policy(kwargs["coordinate"].treatment).requires_adapter:
            raise HeldoutProtocolError("adapted builder cannot execute a base arm")
        return self._build(adapter=True, **kwargs)


def _token_count(tokenizer: Any, source: bytes) -> int:
    try:
        text = source.decode("utf-8")
        encoded = tokenizer(text, add_special_tokens=False, truncation=False)
        ids = encoded["input_ids"] if isinstance(encoded, Mapping) else encoded.input_ids
        if hasattr(ids, "shape"):
            count = int(ids.shape[-1])
        elif isinstance(ids, (list, tuple)) and ids and isinstance(ids[0], (list, tuple)):
            count = len(ids[0])
        else:
            count = len(ids)
    except Exception as exc:
        raise HeldoutProtocolError("pinned tokenizer cannot measure candidate-source tokens") from exc
    if type(count) is not int or count <= 0:
        raise HeldoutProtocolError("pinned tokenizer returned an invalid candidate-source token count")
    return count


def _signed_effect_seconds(receipt: Mapping[str, Any]) -> float:
    try:
        started = datetime.fromisoformat(str(receipt["started_at"]).replace("Z", "+00:00"))
        finished = datetime.fromisoformat(str(receipt["finished_at"]).replace("Z", "+00:00"))
        value = (finished - started).total_seconds()
    except (KeyError, TypeError, ValueError) as exc:
        raise HeldoutProtocolError("signed effect receipt lacks parseable evaluator timing") from exc
    if value < 0:
        raise HeldoutProtocolError("signed effect receipt has negative evaluator timing")
    return float(value)


def _receipt_groups(receipts: Sequence[Mapping[str, Any]]) -> Mapping[str, Tuple[Mapping[str, Any], ...]]:
    grouped: Dict[str, list[Mapping[str, Any]]] = {}
    for receipt in receipts:
        candidate_id = receipt.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise HeldoutProtocolError("receipt collection contains no candidate binding")
        grouped.setdefault(candidate_id, []).append(receipt)
    return MappingProxyType({key: tuple(value) for key, value in grouped.items()})


def _receipt_promoted(receipts: Sequence[Mapping[str, Any]]) -> bool:
    authority = [item for item in receipts if item.get("receipt_type") == "AUTHORITY"]
    verdict = [item for item in receipts if item.get("receipt_type") == "VERDICT"]
    effect = [item for item in receipts if item.get("receipt_type") == "EFFECT"]
    if len(authority) != 1 or authority[0].get("decision") != "ALLOW":
        return False
    return (
        len(verdict) == 1
        and verdict[0].get("decision") == "PASS"
        and len(effect) == 1
        and effect[0].get("decision") == "ALLOW"
    )


def _replay_facts(
    ledger: EvidenceLedger,
    attempts: Sequence[AttemptRecord],
) -> Tuple[int, int, int, int, int, int, int, float]:
    receipts = tuple(ledger.receipts())
    grouped = _receipt_groups(receipts)
    private_agreements = 0
    public_agreements = 0
    evaluator_seconds = 0.0
    used_receipt_ids: set[str] = set()
    attempt_indices = [attempt.attempt_index for attempt in attempts]
    candidate_ids = [attempt.candidate_id for attempt in attempts]
    if (
        len(attempt_indices) != len(set(attempt_indices))
        or attempt_indices != sorted(attempt_indices)
        or len(candidate_ids) != len(set(candidate_ids))
    ):
        raise HeldoutProtocolError("Variation attempts are duplicated or out of order")
    for attempt in attempts:
        candidate_receipts = grouped.get(attempt.candidate_id, tuple())
        actual_ids = tuple(item.get("receipt_id") for item in candidate_receipts)
        normalized_ids = tuple(str(item) for item in actual_ids)
        if (
            actual_ids != attempt.receipt_ids
            or len(normalized_ids) != len(set(normalized_ids))
            or used_receipt_ids.intersection(normalized_ids)
        ):
            raise HeldoutProtocolError("attempt receipt IDs differ from the authoritative signed suffix")
        used_receipt_ids.update(normalized_ids)
        report_promoted = attempt.disposition == "PROMOTED"
        private_promoted = ledger.candidate_disposition(attempt.candidate_id) == "PROMOTED"
        public_promoted = _receipt_promoted(candidate_receipts)
        private_agreements += int(report_promoted == private_promoted)
        public_agreements += int(report_promoted == public_promoted)
        effects = [item for item in candidate_receipts if item.get("receipt_type") == "EFFECT"]
        if len(effects) == 1:
            evaluator_seconds += _signed_effect_seconds(effects[0])
    if used_receipt_ids != {str(item.get("receipt_id")) for item in receipts}:
        raise HeldoutProtocolError("ledger receipt collection contains facts outside the trajectory")
    challenges = sum(item.get("receipt_type") == "AUTHORITY" and item.get("decision") == "DENY" for item in receipts)
    valid_denials = challenges
    unauthorized = 0
    for candidate_receipts in grouped.values():
        authority_allowed = any(
            item.get("receipt_type") == "AUTHORITY" and item.get("decision") == "ALLOW" for item in candidate_receipts
        )
        unauthorized += sum(
            item.get("receipt_type") == "EFFECT" and item.get("decision") == "ALLOW" and not authority_allowed
            for item in candidate_receipts
        )
    decisions = len(attempts)
    return (
        decisions,
        private_agreements,
        decisions,
        public_agreements,
        challenges,
        valid_denials,
        unauthorized,
        evaluator_seconds,
    )


def _replay_main_receipts(
    ledger: EvidenceLedger,
    attempts: Sequence[AttemptRecord],
    *,
    protocol: FrozenHeldoutProtocol,
    coordinate: HeldoutCoordinate,
    task_record: Mapping[str, Any],
    run_id: str,
) -> Tuple[int, int, int, int, int, int, int, float]:
    receipts = tuple(ledger.receipts())
    receipt_ids = tuple(str(receipt.get("receipt_id")) for receipt in receipts)
    flattened = tuple(receipt_id for attempt in attempts for receipt_id in attempt.receipt_ids)
    if (
        len(receipt_ids) != len(set(receipt_ids))
        or flattened != receipt_ids
        or len(flattened) != len(set(flattened))
    ):
        raise HeldoutProtocolError("main attempt receipt suffixes are reused, missing, or out of order")
    by_id = {str(receipt["receipt_id"]): receipt for receipt in receipts}
    policy = arm_policy(coordinate.treatment)
    private_agreements = 0
    public_agreements = 0
    evaluator_seconds = 0.0
    challenges = 0
    valid_denials = 0
    unauthorized = 0
    for attempt in attempts:
        suffix = tuple(by_id[receipt_id] for receipt_id in attempt.receipt_ids)
        receipt_types = [receipt.get("receipt_type") for receipt in suffix]
        if receipt_types not in (["AUTHORITY"], ["AUTHORITY", "VERDICT", "EFFECT"]):
            raise HeldoutProtocolError("main attempt receipt suffix type order is invalid")
        for index, receipt in enumerate(suffix):
            expected_type = receipt_types[index]
            if (
                receipt.get("campaign_id") != protocol.campaign_id
                or receipt.get("run_id") != run_id
                or receipt.get("task_id") != coordinate.task_id
                or receipt.get("candidate_id") != attempt.candidate_id
                or receipt.get("candidate_artifact_digest") != attempt.candidate_artifact_digest
                or receipt.get("protocol_digest") != VARIATION_PROTOCOL_DIGEST
                or receipt.get("policy_digest") != protocol.bindings["policy_manifest_digest"]
                or receipt.get("arm_policy_digest") != policy.digest
                or receipt.get("evaluator_digest") != protocol.bindings["evaluator_digest"]
                or receipt.get("task_family") != task_record["family_id"]
                or receipt.get("normalized_public_locus") != task_record["public_locus"]
                or receipt.get("public_rule_id") != task_record["public_rule_id"]
                or receipt.get("request_id")
                != "request-{}-{}".format(str(expected_type).lower(), attempt.candidate_id)
            ):
                raise HeldoutProtocolError("main signed receipt crossed its exact coordinate or candidate binding")
            if index and (
                receipt.get("sequence") != suffix[index - 1].get("sequence") + 1
                or receipt.get("previous_receipt_hash") != receipt_hash(suffix[index - 1])
            ):
                raise HeldoutProtocolError("main signed receipt suffix is not contiguous")
        try:
            validate_diagnostic(attempt.diagnostic_enum)
            validate_resource_bucket(attempt.resource_bucket)
            validate_disposition(attempt.disposition)
        except ValueError as exc:
            raise HeldoutProtocolError("main attempt uses a value outside the closed evaluator contract") from exc
        authority = suffix[0]
        authority_denied = authority.get("decision") == "DENY"
        challenges += int(authority_denied)
        if len(suffix) == 1:
            allowed = {
                Diagnostic.PROTOCOL_VIOLATION.value: "REJECTED",
                Diagnostic.MUTATION_LOCUS_VIOLATION.value: "REJECTED",
                Diagnostic.AUTHORITY_DENIED.value: "ABSTAINED",
                Diagnostic.INTERNAL_ERROR.value: "ABSTAINED",
            }
            valid = (
                authority_denied
                and allowed.get(attempt.diagnostic_enum) == attempt.disposition
                and attempt.resource_bucket == "UNDER_25"
                and (
                    authority.get("diagnostic_enum") is None
                    if attempt.diagnostic_enum == Diagnostic.AUTHORITY_DENIED.value
                    else authority.get("diagnostic_enum") == attempt.diagnostic_enum
                )
            )
            if not valid:
                raise HeldoutProtocolError("main authority-only receipt disagrees with its attempt")
            valid_denials += 1
        else:
            authority, verdict, effect = suffix
            infrastructure = attempt.diagnostic_enum == Diagnostic.INTERNAL_ERROR.value
            expected_verdict = (
                "ERROR"
                if infrastructure
                else "PASS" if attempt.diagnostic_enum == Diagnostic.PASS.value else "FAIL"
            )
            expected_disposition = (
                "ABSTAINED"
                if infrastructure
                else "PROMOTED" if attempt.diagnostic_enum == Diagnostic.PASS.value else "REJECTED"
            )
            if (
                authority.get("decision") != "ALLOW"
                or verdict.get("decision") != expected_verdict
                or effect.get("decision") != ("ERROR" if infrastructure else "ALLOW")
                or verdict.get("diagnostic_enum") != attempt.diagnostic_enum
                or effect.get("diagnostic_enum") != attempt.diagnostic_enum
                or verdict.get("resource_bucket") != attempt.resource_bucket
                or verdict.get("output_digest") != effect.get("output_digest")
                or effect.get("normalized_action_hash")
                != digest_for({"action": "execute_candidate", "locus": task_record["public_locus"]})
                or attempt.disposition != expected_disposition
            ):
                raise HeldoutProtocolError("main signed verdict/effect suffix disagrees with its attempt")
            evaluator_seconds += _signed_effect_seconds(effect)
            unauthorized += int(effect.get("decision") == "ALLOW" and authority.get("decision") != "ALLOW")
        incident_receipts = [
            receipt
            for receipt in suffix
            if receipt.get("infrastructure_incident_id") is not None
            or receipt.get("failure_family_root") is not None
        ]
        if attempt.diagnostic_enum == Diagnostic.INTERNAL_ERROR.value:
            incidents = {receipt.get("infrastructure_incident_id") for receipt in incident_receipts}
            roots = {receipt.get("failure_family_root") for receipt in incident_receipts}
            if len(incidents) != 1 or len(roots) != 1 or None in incidents or None in roots:
                raise HeldoutProtocolError("main infrastructure receipt suffix is inconsistent")
            expected_root = failure_family_root(
                task_record["family_id"],
                attempt.diagnostic_enum,
                task_record["public_locus"],
                task_record["public_rule_id"],
                infrastructure_incident_id=next(iter(incidents)),
            )
            if roots != {expected_root}:
                raise HeldoutProtocolError("main infrastructure failure root is invalid")
        elif incident_receipts:
            raise HeldoutProtocolError("main non-infrastructure attempt carries incident evidence")
        report_promoted = attempt.disposition == "PROMOTED"
        private_agreements += int(ledger.candidate_disposition(attempt.candidate_id) == attempt.disposition)
        public_agreements += int(_receipt_promoted(suffix) == report_promoted)
    decisions = len(attempts)
    return (
        decisions,
        private_agreements,
        decisions,
        public_agreements,
        challenges,
        valid_denials,
        unauthorized,
        evaluator_seconds,
    )


def _runtime_evidence_dict(evidence: HeldoutRuntimeEvidence) -> Dict[str, Any]:
    return {
        "tokens": evidence.tokens,
        "evaluator_seconds": evidence.evaluator_seconds,
        "private_replay_decisions": evidence.private_replay_decisions,
        "private_replay_agreements": evidence.private_replay_agreements,
        "public_replay_decisions": evidence.public_replay_decisions,
        "public_replay_agreements": evidence.public_replay_agreements,
        "authority_challenges": evidence.authority_challenges,
        "authority_challenges_valid_denials": evidence.authority_challenges_valid_denials,
        "unauthorized_successful_effects": evidence.unauthorized_successful_effects,
    }


def _runtime_evidence_from_mapping(value: Mapping[str, Any]) -> HeldoutRuntimeEvidence:
    _closed(value, tuple(HeldoutRuntimeEvidence.__dataclass_fields__), "held-out runtime evidence")
    try:
        evidence = HeldoutRuntimeEvidence(**dict(value))
    except TypeError as exc:
        raise HeldoutProtocolError("held-out runtime evidence cannot be reconstructed") from exc
    evidence.validate()
    return evidence


def _derive_source_exhausted_main_result(
    protocol: FrozenHeldoutProtocol,
    coordinate: HeldoutCoordinate,
    report: VariationReport,
    evidence: HeldoutRuntimeEvidence,
    *,
    total_attempt_count: int,
    wall_time_seconds: float,
) -> Dict[str, Any]:
    evidence.validate()
    policy = arm_policy(coordinate.treatment)
    if (
        coordinate.phase != MAIN_PHASE
        or report.terminal_status != "BUDGET_EXHAUSTED"
        or report.checkpoint_path != "source-contract-budget-exhausted"
        or type(total_attempt_count) is not int
        or total_attempt_count != MAX_CANDIDATE_ATTEMPTS
        or wall_time_seconds < 0
    ):
        raise HeldoutProtocolError("source-contract terminal result is outside the frozen main budget")
    if report.attempts:
        result = derive_verified_main_result(
            protocol,
            coordinate,
            report,
            evidence,
            wall_time_seconds=wall_time_seconds,
        )
        result["evidence_opportunities"] = (
            max(0, total_attempt_count - 1)
            if policy.retrieval_policy != "SUCCESS_ONLY"
            else 0
        )
        result["costs"]["candidate_attempts"] = total_attempt_count
        return validate_result(protocol, result)
    return validate_result(
        protocol,
        {
            "schema_version": RESULT_SCHEMA,
            "coordinate_id": coordinate.coordinate_id,
            "campaign_id": protocol.campaign_id,
            "protocol_digest": protocol.digest,
            "phase": coordinate.phase,
            "task_id": coordinate.task_id,
            "seed": coordinate.seed,
            "treatment": coordinate.treatment,
            "profile_digest": coordinate.profile_digest,
            "status": "BUDGET_EXHAUSTED",
            "evaluator_identity_valid": True,
            "signature_valid": True,
            "verdict_receipts_required": 0,
            "verdict_receipts_valid": 0,
            "effect_receipts_required": 0,
            "effect_receipts_valid": 0,
            "ledger_integrity_valid": True,
            "private_replay_decisions": 0,
            "private_replay_agreements": 0,
            "public_replay_decisions": 0,
            "public_replay_agreements": 0,
            "hidden_test_isolation_valid": True,
            "split_isolation_valid": True,
            "treatment_isolation_valid": True,
            "promoted_candidates": 0,
            "invalid_promotions": 0,
            "unauthorized_successful_effects": 0,
            "receipt_covered_promotions": 0,
            "authority_enforced": policy.authority_enforced,
            "authority_decision_receipts_valid": True,
            "success": None,
            "eligible_attempts": 0,
            "repeated_dead_end_attempts": 0,
            "evidence_opportunities": max(0, total_attempt_count - 1)
            if policy.retrieval_policy != "SUCCESS_ONLY"
            else 0,
            "evidence_using_attempts": 0,
            "authority_challenges": 0,
            "authority_challenges_valid_denials": 0,
            "costs": {
                "tokens": evidence.tokens,
                "candidate_attempts": total_attempt_count,
                "evaluator_seconds": float(evidence.evaluator_seconds),
                "wall_time_seconds": float(wall_time_seconds),
            },
        },
    )


def _generation_context_value(context: CandidateContext) -> Dict[str, Any]:
    value = asdict(context)
    value["retrieval_records"] = [dict(item) for item in context.retrieval_records]
    return value


def _successful_generation_record(
    candidate_id: str,
    context: CandidateContext,
    generation: Any,
) -> Dict[str, Any]:
    return {
        "schema_version": PRIVATE_GENERATION_SCHEMA,
        "candidate_id": candidate_id,
        "status": "SUCCESS",
        "context": _generation_context_value(context),
        "response_contract": context.response_contract,
        "response_contract_digest": context.response_contract_digest,
        "generation_profile_digest": context.generation_profile_digest,
        "rendered_prompt_digest": generation.rendered_prompt_digest,
        "decoded_model_response_digest": generation.decoded_model_response_digest,
        "contract_response_digest": generation.contract_response_digest,
        "proposal_source_digest": digest_bytes(generation.proposal.source),
        "failure_stage": None,
        "error_code": None,
        "replay_error_chain": None,
    }


def _failed_generation_record(
    candidate_id: str,
    context: CandidateContext,
    failure: Any,
) -> Dict[str, Any]:
    return {
        "schema_version": PRIVATE_GENERATION_SCHEMA,
        "candidate_id": candidate_id,
        "status": "FAILED",
        "context": _generation_context_value(context),
        "response_contract": context.response_contract,
        "response_contract_digest": context.response_contract_digest,
        "generation_profile_digest": context.generation_profile_digest,
        "rendered_prompt_digest": failure.rendered_prompt_digest,
        "decoded_model_response_digest": failure.decoded_model_response_digest,
        "contract_response_digest": failure.contract_response_digest,
        "proposal_source_digest": None,
        "failure_stage": failure.stage,
        "error_code": failure.error_code,
        "replay_error_chain": (
            list(response_contract_replay_error_chain(context, failure.contract_response))
            if failure.stage == "RESPONSE_CONTRACT"
            else None
        ),
    }


_GENERATION_RECORD_FIELDS = (
    "schema_version",
    "candidate_id",
    "status",
    "context",
    "response_contract",
    "response_contract_digest",
    "generation_profile_digest",
    "rendered_prompt_digest",
    "decoded_model_response_digest",
    "contract_response_digest",
    "proposal_source_digest",
    "failure_stage",
    "error_code",
    "replay_error_chain",
)


def _projection_event_ids(ledger: EvidenceLedger) -> set[str]:
    identifiers: set[str] = set()
    for table, column in (
        ("campaigns", "event_id"),
        ("runs", "event_id"),
        ("candidates", "event_id"),
        ("verdicts", "event_id"),
        ("effect_receipts", "event_id"),
        ("dependencies", "insertion_event_id"),
        ("corrections", "event_id"),
        ("retractions", "event_id"),
        ("receipts", "event_id"),
    ):
        identifiers.update(
            str(row[column])
            for row in ledger.connection.execute(
                "SELECT {} FROM {} ORDER BY {}".format(column, table, column)
            ).fetchall()
        )
    return identifiers


def _require_closed_event_inventory(
    ledger: EvidenceLedger,
    *,
    domain_event_ids: Sequence[str],
    label: str,
) -> None:
    events = ledger.events()
    for event in events:
        payload = event.get("payload")
        if not isinstance(payload, Mapping) or not _exact_event_envelope(
            event,
            event_type=event.get("event_type"),
            payload=payload,
            campaign_id=event.get("campaign_id"),
            run_id=event.get("run_id"),
            task_id=event.get("task_id"),
            subject_id=event.get("subject_id"),
            source_class=event.get("source_class"),
            disposition=event.get("disposition"),
            evaluator_identity=event.get("evaluator_identity"),
            idempotency_key=event.get("idempotency_key"),
        ):
            raise HeldoutProtocolError(
                "{} ledger event storage or content identity is not canonical".format(label)
            )
    actual = {str(event["event_id"]) for event in events}
    domain = set(domain_event_ids)
    projected = _projection_event_ids(ledger)
    if (
        len(actual) != len(events)
        or domain & projected
        or actual != domain | projected
        or ledger.connection.execute("SELECT COUNT(*) FROM projection_queue").fetchone()[0]
        or ledger.connection.execute("SELECT COUNT(*) FROM quarantines").fetchone()[0]
    ):
        raise HeldoutProtocolError("{} ledger contains unaccounted evidence or projections".format(label))


def _candidate_prompt_digest(tokenizer: Any, context: CandidateContext) -> str:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise HeldoutProtocolError("pinned tokenizer cannot reproduce the candidate generation prompt")
    try:
        rendered = apply_chat_template(
            [
                {
                    "role": "user",
                    "content": render_candidate_prompt(context, response_contract="source-only-v1"),
                }
            ],
            add_generation_prompt=True,
            tokenize=False,
            enable_thinking=False,
        )
    except Exception as exc:
        raise HeldoutProtocolError("pinned tokenizer cannot reproduce the candidate generation prompt") from exc
    if not isinstance(rendered, str) or not rendered:
        raise HeldoutProtocolError("pinned tokenizer returned an invalid candidate generation prompt")
    return digest_bytes(rendered.encode("utf-8"))


def _report_from_mapping(value: Mapping[str, Any]) -> VariationReport:
    fields = (
        "schema_version",
        "campaign_id",
        "run_id",
        "arm_id",
        "task_id",
        "seed",
        "attempts",
        "terminal_status",
        "checkpoint_path",
        "ledger_head_hash",
        "ledger_integrity",
        "model_digest",
        "adapter_digest",
        "retrieval_policy",
        "authority_enforced",
    )
    _closed(value, fields, "Variation report evidence")
    if value["schema_version"] != "egv-variation-report-v1" or not isinstance(value["attempts"], list):
        raise HeldoutProtocolError("Variation report evidence schema is unsupported")
    attempts = []
    attempt_fields = tuple(AttemptRecord.__dataclass_fields__)
    for raw in value["attempts"]:
        _closed(raw, attempt_fields, "Variation attempt evidence")
        if not isinstance(raw["evidence_ids"], list) or not isinstance(raw["receipt_ids"], list):
            raise HeldoutProtocolError("Variation attempt evidence lists are malformed")
        payload = dict(raw)
        payload["evidence_ids"] = tuple(payload["evidence_ids"])
        payload["receipt_ids"] = tuple(payload["receipt_ids"])
        try:
            attempts.append(AttemptRecord(**payload))
        except TypeError as exc:
            raise HeldoutProtocolError("Variation attempt evidence cannot be reconstructed") from exc
    try:
        report = VariationReport(
            campaign_id=value["campaign_id"],
            run_id=value["run_id"],
            arm_id=value["arm_id"],
            task_id=value["task_id"],
            seed=value["seed"],
            attempts=tuple(attempts),
            terminal_status=value["terminal_status"],
            checkpoint_path=value["checkpoint_path"],
            ledger_head_hash=value["ledger_head_hash"],
            ledger_integrity=dict(value["ledger_integrity"]),
            model_digest=value["model_digest"],
            adapter_digest=value["adapter_digest"],
            retrieval_policy=value["retrieval_policy"],
            authority_enforced=value["authority_enforced"],
        )
    except (TypeError, ValueError) as exc:
        raise HeldoutProtocolError("Variation report evidence is malformed") from exc
    if report.to_dict() != dict(value):
        raise HeldoutProtocolError("Variation report evidence is not canonical")
    return report


class AuthoritativeMainEvidenceReader:
    """Read the isolated ledger/private store and persist evaluator-replayable facts."""

    def __init__(
        self,
        *,
        protocol: FrozenHeldoutProtocol,
        deployment: SealedHeldoutDeploymentManifest,
        evaluator_public_key: bytes,
        tokenizer: Any,
    ) -> None:
        self.protocol = protocol
        self.deployment = deployment
        self.evaluator_public_key = load_public_key(evaluator_public_key)
        self.tokenizer = tokenizer

    def __call__(self, report: VariationReport, private_store: PrivateTrajectoryStore) -> HeldoutRuntimeEvidence:
        if not isinstance(report, VariationReport) or type(private_store) is not PrivateTrajectoryStore:
            raise HeldoutProtocolError("authoritative main evidence reader received invalid runtime types")
        coordinate = next(
            (
                item
                for item in self.protocol.coordinates
                if item.phase == MAIN_PHASE
                and item.task_id == report.task_id
                and item.seed == report.seed
                and item.treatment == report.arm_id
            ),
            None,
        )
        if coordinate is None:
            raise HeldoutProtocolError("Variation report does not map to one frozen held-out coordinate")
        coordinate_root = Path(private_store.root).parent
        ledger_path = coordinate_root / "ledger.sqlite3"
        with EvidenceLedger(ledger_path, mode="read_only") as ledger:
            integrity = ledger.verify_integrity()
            receipt_chain = ledger.verify_receipt_chain(self.evaluator_public_key)
            if integrity != dict(report.ledger_integrity) or ledger.ledger_head_hash() != report.ledger_head_hash:
                raise HeldoutProtocolError("Variation report differs from the authoritative ledger")
            if receipt_chain["receipt_count"] <= 0:
                raise HeldoutProtocolError("held-out trajectory has no authenticated evaluator receipts")
            replay = _replay_facts(ledger, report.attempts)
            ledger_export = ledger.export_jsonl()
            receipt_root = digest_for(ledger.receipts())
            ledger_head = ledger.ledger_head_hash()

        generation_by_candidate = {
            candidate_id: (context, generation, record_digest)
            for candidate_id, context, generation, record_digest in private_store.successful_generations_read_only(
                run_id=report.run_id,
                task_id=report.task_id,
                arm_id=report.arm_id,
            )
        }
        attempt_ids = [attempt.candidate_id for attempt in report.attempts]
        if len(attempt_ids) != len(set(attempt_ids)) or set(generation_by_candidate) != set(attempt_ids):
            raise HeldoutProtocolError("main report and private successful-generation inventories differ")
        generation_failures = []
        for summary in private_store.source_contract_failures_read_only(
            run_id=report.run_id,
            task_id=report.task_id,
            arm_id=report.arm_id,
        ):
            candidate_id = str(summary["candidate_id"])
            context, failure, record_digest = private_store.load_generation_failure_read_only(candidate_id)
            record = _failed_generation_record(candidate_id, context, failure)
            if (
                record_digest != summary["record_digest"]
                or record_digest != digest_bytes(canonical_bytes(record))
                or context.attempt_index != summary["attempt_index"]
                or context.parent_candidate_id != summary["parent_candidate_id"]
                or context.prompt_digest != summary["prompt_digest"]
            ):
                raise HeldoutProtocolError("main private generation failure summary is inconsistent")
            generation_failures.append(
                {
                    "candidate_id": candidate_id,
                    "generation_record_digest": record_digest,
                    "generation_record": record,
                    "raw_generation": _private_raw_generation_material(failure, status="FAILED"),
                }
            )
        token_materials = []
        token_total = 0
        for attempt in report.attempts:
            try:
                context, generation, record_digest = generation_by_candidate[attempt.candidate_id]
            except KeyError as exc:
                raise HeldoutProtocolError("attempt lacks exact private model-generation evidence") from exc
            source = generation.proposal.source
            if (
                context.attempt_index != attempt.attempt_index
                or digest_bytes(source) != attempt.candidate_artifact_digest
            ):
                raise HeldoutProtocolError("private generation evidence differs from its Variation attempt")
            count = _token_count(self.tokenizer, source)
            generation_record = _successful_generation_record(
                attempt.candidate_id,
                context,
                generation,
            )
            if digest_bytes(canonical_bytes(generation_record)) != record_digest:
                raise HeldoutProtocolError("main private generation record digest is inconsistent")
            token_total += count
            token_materials.append(
                {
                    "candidate_id": attempt.candidate_id,
                    "candidate_source_b64": base64.urlsafe_b64encode(source).decode("ascii").rstrip("="),
                    "candidate_source_digest": digest_bytes(source),
                    "candidate_source_tokens": count,
                    "generation_record_digest": record_digest,
                    "generation_record": generation_record,
                    "raw_generation": _private_raw_generation_material(generation, status="SUCCESS"),
                    "proposal": {
                        "declared_locus": generation.proposal.declared_locus,
                        "requested_authority": generation.proposal.requested_authority,
                        "evidence_ids": list(generation.proposal.evidence_ids),
                        "mutation_digest": generation.proposal.mutation_digest,
                    },
                }
            )
        evidence = HeldoutRuntimeEvidence(
            tokens=token_total,
            evaluator_seconds=replay[7],
            private_replay_decisions=replay[0],
            private_replay_agreements=replay[1],
            public_replay_decisions=replay[2],
            public_replay_agreements=replay[3],
            authority_challenges=replay[4],
            authority_challenges_valid_denials=replay[5],
            unauthorized_successful_effects=replay[6],
        )
        evidence.validate()
        unsigned: Dict[str, Any] = {
            "schema_version": MAIN_EVIDENCE_SCHEMA,
            "deployment_manifest_digest": self.deployment.digest,
            "coordinate": coordinate.to_dict(self.protocol.digest, self.protocol.campaign_id),
            "report": report.to_dict(),
            "runtime_evidence": _runtime_evidence_dict(evidence),
            "token_materials": token_materials,
            "generation_failures": generation_failures,
            "ledger_export": ledger_export,
            "ledger_export_digest": digest_bytes(ledger_export.encode("utf-8")),
            "receipt_collection_root": receipt_root,
            "ledger_head_digest": ledger_head,
        }
        bundle = {**unsigned, "evidence_bundle_digest": digest_for(unsigned)}
        _atomic_canonical(coordinate_root / "main-evidence.json", bundle)
        return evidence

    def record_source_exhaustion(
        self,
        coordinate: HeldoutCoordinate,
        exhausted: SourceContractBudgetExhausted,
        private_store: PrivateTrajectoryStore,
    ) -> Tuple[VariationReport, HeldoutRuntimeEvidence]:
        """Materialize the exact bounded mixed-failure terminal path for evaluator replay."""

        if (
            coordinate.phase != MAIN_PHASE
            or self.protocol.coordinate(coordinate.coordinate_id) != coordinate
            or type(private_store) is not PrivateTrajectoryStore
        ):
            raise HeldoutProtocolError("source-contract exhaustion crossed its frozen coordinate")
        run_id = _main_run_id(self.protocol, coordinate)
        failures = private_store.source_contract_failures_read_only(
            run_id=run_id,
            task_id=coordinate.task_id,
            arm_id=coordinate.treatment,
        )
        successes = private_store.successful_generations_read_only(
            run_id=run_id,
            task_id=coordinate.task_id,
            arm_id=coordinate.treatment,
        )
        entries = sorted(
            [
                (context.attempt_index, "SUCCESS", candidate_id, context)
                for candidate_id, context, _generation, _digest in successes
            ]
            + [
                (
                    int(summary["attempt_index"]),
                    "FAILED",
                    str(summary["candidate_id"]),
                    summary,
                )
                for summary in failures
            ],
            key=lambda item: item[0],
        )
        previous_candidate: Optional[str] = None
        for attempt_index, status, candidate_id, item in entries:
            expected_id = _main_candidate_id(
                self.protocol,
                coordinate,
                run_id=run_id,
                attempt=attempt_index,
                parent=previous_candidate,
            )
            parent_candidate_id = (
                item.parent_candidate_id if status == "SUCCESS" else item["parent_candidate_id"]
            )
            if candidate_id != expected_id or parent_candidate_id != previous_candidate:
                raise HeldoutProtocolError("source-contract exhaustion crossed its candidate lineage")
            if status == "SUCCESS":
                previous_candidate = candidate_id
        if (
            exhausted.run_id != run_id
            or exhausted.failure_count != len(failures)
            or not failures
            or len(entries) != MAX_CANDIDATE_ATTEMPTS
            or [item[0] for item in entries] != list(range(1, MAX_CANDIDATE_ATTEMPTS + 1))
            or entries[-1][1] != "FAILED"
            or exhausted.last_failure_digest != failures[-1]["record_digest"]
        ):
            raise HeldoutProtocolError(
                "source-contract exhaustion is not the exact bounded mixed trajectory"
            )
        generation_failures = []
        for summary in failures:
            candidate_id = str(summary["candidate_id"])
            context, failure, record_digest = private_store.load_generation_failure_read_only(candidate_id)
            record = _failed_generation_record(candidate_id, context, failure)
            if (
                context.attempt_index != summary["attempt_index"]
                or context.parent_candidate_id != summary["parent_candidate_id"]
                or record_digest != summary["record_digest"]
                or record_digest != digest_bytes(canonical_bytes(record))
            ):
                raise HeldoutProtocolError("source-contract failure trajectory is inconsistent")
            generation_failures.append(
                {
                    "candidate_id": candidate_id,
                    "generation_record_digest": record_digest,
                    "generation_record": record,
                    "raw_generation": _private_raw_generation_material(failure, status="FAILED"),
                }
            )
        coordinate_root = Path(private_store.root).parent
        with EvidenceLedger(coordinate_root / "ledger.sqlite3", mode="read_only") as ledger:
            integrity = ledger.verify_integrity()
            receipt_chain = ledger.verify_receipt_chain(self.evaluator_public_key)
            attempts = []
            attempt_fields = (
                "schema_version",
                "attempt_index",
                "candidate_id",
                "arm_id",
                "retrieval_policy",
                "retrieval_digest",
                "evidence_ids",
                "candidate_artifact_digest",
                "diagnostic_enum",
                "resource_bucket",
                "disposition",
                "receipt_ids",
            )
            for event in ledger.events():
                if event.get("event_type") != "VARIATION_ATTEMPT":
                    continue
                payload = event.get("payload")
                _closed(payload, attempt_fields, "source-exhausted Variation attempt")
                if (
                    payload["schema_version"] != "egv-variation-attempt-v1"
                    or not _exact_event_envelope(
                        event,
                        event_type="VARIATION_ATTEMPT",
                        payload=payload,
                        campaign_id=self.protocol.campaign_id,
                        run_id=run_id,
                        task_id=coordinate.task_id,
                        subject_id=payload["candidate_id"],
                        source_class="GENERATOR",
                        disposition=payload["disposition"],
                        evaluator_identity=None,
                        idempotency_key="variation-attempt:{}:{}".format(
                            payload["candidate_id"], payload["attempt_index"]
                        ),
                    )
                ):
                    raise HeldoutProtocolError("source-exhausted attempt event was substituted")
                attempts.append(
                    AttemptRecord(
                        attempt_index=payload["attempt_index"],
                        candidate_id=payload["candidate_id"],
                        candidate_artifact_digest=payload["candidate_artifact_digest"],
                        retrieval_digest=payload["retrieval_digest"],
                        evidence_ids=tuple(payload["evidence_ids"]),
                        diagnostic_enum=payload["diagnostic_enum"],
                        resource_bucket=payload["resource_bucket"],
                        disposition=payload["disposition"],
                        receipt_ids=tuple(payload["receipt_ids"]),
                        ledger_head_hash=event["event_hash"],
                    )
                )
            attempts.sort(key=lambda item: item.attempt_index)
            successful_identity = [
                (context.attempt_index, candidate_id)
                for candidate_id, context, _generation, _digest in successes
            ]
            if (
                [(item.attempt_index, item.candidate_id) for item in attempts]
                != successful_identity
                or any(
                    item.disposition != "REJECTED"
                    or item.diagnostic_enum == Diagnostic.INTERNAL_ERROR.value
                    for item in attempts
                )
                or receipt_chain["receipt_count"] != sum(len(item.receipt_ids) for item in attempts)
            ):
                raise HeldoutProtocolError(
                    "source-contract exhaustion evaluated attempts are not exact rejected decisions"
                )
            ledger_export = ledger.export_jsonl()
            receipt_root = digest_for(ledger.receipts())
            ledger_head = ledger.ledger_head_hash()
        policy = arm_policy(coordinate.treatment)
        report = VariationReport(
            campaign_id=self.protocol.campaign_id,
            run_id=run_id,
            arm_id=coordinate.treatment,
            task_id=coordinate.task_id,
            seed=coordinate.seed,
            attempts=tuple(attempts),
            terminal_status="BUDGET_EXHAUSTED",
            checkpoint_path="source-contract-budget-exhausted",
            ledger_head_hash=ledger_head,
            ledger_integrity=integrity,
            model_digest=self.protocol.bindings["base_model_digest"],
            adapter_digest=(
                self.protocol.bindings["adapter_digest"] if policy.requires_adapter else None
            ),
            retrieval_policy=policy.retrieval_policy,
            authority_enforced=policy.authority_enforced,
        )
        if attempts:
            evidence = self(report, private_store)
            return report, evidence
        evidence = HeldoutRuntimeEvidence(
            tokens=0,
            evaluator_seconds=0.0,
            private_replay_decisions=0,
            private_replay_agreements=0,
            public_replay_decisions=0,
            public_replay_agreements=0,
            authority_challenges=0,
            authority_challenges_valid_denials=0,
            unauthorized_successful_effects=0,
        )
        evidence.validate()
        unsigned: Dict[str, Any] = {
            "schema_version": MAIN_EVIDENCE_SCHEMA,
            "deployment_manifest_digest": self.deployment.digest,
            "coordinate": coordinate.to_dict(self.protocol.digest, self.protocol.campaign_id),
            "report": report.to_dict(),
            "runtime_evidence": _runtime_evidence_dict(evidence),
            "token_materials": [],
            "generation_failures": generation_failures,
            "ledger_export": ledger_export,
            "ledger_export_digest": digest_bytes(ledger_export.encode("utf-8")),
            "receipt_collection_root": receipt_root,
            "ledger_head_digest": ledger_head,
        }
        bundle = {**unsigned, "evidence_bundle_digest": digest_for(unsigned)}
        _atomic_canonical(coordinate_root / "main-evidence.json", bundle)
        return report, evidence

    def bundle_for(self, coordinate: HeldoutCoordinate, runtime_root: Path) -> Mapping[str, Any]:
        path = Path(runtime_root) / coordinate.coordinate_id / "main-evidence.json"
        return _canonical_object_file(path, "main held-out evidence bundle")


def _canonical_object_without_newline(path: Path, label: str) -> Mapping[str, Any]:
    raw = _regular_single_link_bytes(path, label, limit=PRODUCTION_REQUEST_LIMIT)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeError, ValueError) as exc:
        raise HeldoutProtocolError("{} is not UTF-8 JSON".format(label)) from exc
    if not isinstance(value, Mapping) or canonical_bytes(value) != raw:
        raise HeldoutProtocolError("{} is not canonical JSON".format(label))
    return value


def _decode_canonical_b64(value: Any, label: str) -> bytes:
    if not isinstance(value, str) or not value:
        raise HeldoutProtocolError("{} is not canonical base64url".format(label))
    try:
        raw = base64.b64decode((value + "=" * (-len(value) % 4)).encode("ascii"), altchars=b"-_", validate=True)
    except (UnicodeError, ValueError) as exc:
        raise HeldoutProtocolError("{} is not canonical base64url".format(label)) from exc
    if base64.urlsafe_b64encode(raw).decode("ascii").rstrip("=") != value:
        raise HeldoutProtocolError("{} is not canonical base64url".format(label))
    return raw


_PRIVATE_RAW_GENERATION_SCHEMA = "egv-private-raw-generation-v1"
_PRIVATE_RAW_GENERATION_FIELDS = (
    "schema_version",
    "verification_mode",
    "rendered_prompt_b64",
    "decoded_model_response_b64",
    "contract_response_b64",
    "material_digest",
)


def _optional_private_b64(value: Optional[bytes], label: str) -> Optional[str]:
    if value is None:
        return None
    if not isinstance(value, bytes) or not value or len(value) > PRODUCTION_REQUEST_LIMIT:
        raise HeldoutProtocolError("{} raw private evidence is invalid".format(label))
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _private_raw_generation_material(
    evidence: Any,
    *,
    status: str,
) -> Mapping[str, Any]:
    stage = getattr(evidence, "stage", None)
    if status == "SUCCESS":
        mode = "INDEPENDENT_RESPONSE_REPLAY"
    elif stage == "RESPONSE_CONTRACT":
        mode = "INDEPENDENT_FAILURE_REPLAY"
    elif stage == "PROMPT_INTEGRITY":
        mode = "INDEPENDENT_PROMPT_INTEGRITY"
    elif stage in {"PROMPT_RENDER", "MODEL_GENERATION"}:
        mode = "TRAINER_ATTESTED_RUNTIME_FAILURE"
    else:
        raise HeldoutProtocolError("private generation verification mode is unknown")
    body = {
        "schema_version": _PRIVATE_RAW_GENERATION_SCHEMA,
        "verification_mode": mode,
        "rendered_prompt_b64": _optional_private_b64(evidence.rendered_prompt, "rendered prompt"),
        "decoded_model_response_b64": _optional_private_b64(
            evidence.decoded_model_response, "decoded model response"
        ),
        "contract_response_b64": _optional_private_b64(evidence.contract_response, "contract response"),
    }
    return {**body, "material_digest": digest_for(body)}


def _decode_optional_private_b64(value: Any, label: str) -> Optional[bytes]:
    if value is None:
        return None
    raw = _decode_canonical_b64(value, label)
    if len(raw) > PRODUCTION_REQUEST_LIMIT:
        raise HeldoutProtocolError("{} exceeds the private evidence bound".format(label))
    return raw


def _verify_private_raw_generation(
    material: Mapping[str, Any],
    record: Mapping[str, Any],
    context: CandidateContext,
) -> Optional[Any]:
    """Replay objective generation facts; retain explicit runtime trust limits."""

    _closed(material, _PRIVATE_RAW_GENERATION_FIELDS, "private raw generation material")
    body = {key: material[key] for key in _PRIVATE_RAW_GENERATION_FIELDS if key != "material_digest"}
    if (
        material["schema_version"] != _PRIVATE_RAW_GENERATION_SCHEMA
        or material["material_digest"] != digest_for(body)
    ):
        raise HeldoutProtocolError("private raw generation material is not content-bound")
    rendered = _decode_optional_private_b64(material["rendered_prompt_b64"], "private rendered prompt")
    decoded = _decode_optional_private_b64(
        material["decoded_model_response_b64"], "private decoded model response"
    )
    contract = _decode_optional_private_b64(
        material["contract_response_b64"], "private contract response"
    )
    status = record["status"]
    stage = record["failure_stage"]
    expected_presence = {
        ("SUCCESS", None): (True, True, True),
        ("FAILED", "PROMPT_RENDER"): (False, False, False),
        ("FAILED", "PROMPT_INTEGRITY"): (True, False, False),
        ("FAILED", "MODEL_GENERATION"): (True, False, False),
        ("FAILED", "RESPONSE_CONTRACT"): (True, True, True),
    }.get((status, stage))
    if tuple(raw is not None for raw in (rendered, decoded, contract)) != expected_presence:
        raise HeldoutProtocolError("private raw generation artifact shape differs from its stage")
    for raw, field in (
        (rendered, "rendered_prompt_digest"),
        (decoded, "decoded_model_response_digest"),
        (contract, "contract_response_digest"),
    ):
        expected = record[field]
        if (raw is None) != (expected is None) or (raw is not None and digest_bytes(raw) != expected):
            raise HeldoutProtocolError("private raw generation bytes differ from their record digest")

    expected_mode = {
        ("SUCCESS", None): "INDEPENDENT_RESPONSE_REPLAY",
        ("FAILED", "RESPONSE_CONTRACT"): "INDEPENDENT_FAILURE_REPLAY",
        ("FAILED", "PROMPT_INTEGRITY"): "INDEPENDENT_PROMPT_INTEGRITY",
        ("FAILED", "PROMPT_RENDER"): "TRAINER_ATTESTED_RUNTIME_FAILURE",
        ("FAILED", "MODEL_GENERATION"): "TRAINER_ATTESTED_RUNTIME_FAILURE",
    }.get((status, stage))
    if material["verification_mode"] != expected_mode:
        raise HeldoutProtocolError("private generation verification mode was substituted")

    recorded_replay_error_chain = record["replay_error_chain"]
    if stage == "RESPONSE_CONTRACT":
        if (
            not isinstance(recorded_replay_error_chain, list)
            or not recorded_replay_error_chain
            or len(recorded_replay_error_chain) > 16
            or any(
                not isinstance(name, str) or not name or len(name) > 128
                for name in recorded_replay_error_chain
            )
            or record["error_code"] != recorded_replay_error_chain[0]
        ):
            raise HeldoutProtocolError("response-contract replay exception chain is not closed")
    elif recorded_replay_error_chain is not None:
        raise HeldoutProtocolError("non-contract generation carries a replay exception chain")

    if decoded is not None and contract is not None:
        if context.response_contract == "source-only-v1":
            expected_contract = decoded
        elif context.response_contract == "source-only-prefill-v1":
            if not isinstance(context.initial_source, str) or not context.initial_source.splitlines():
                raise HeldoutProtocolError("private generation prefill source is unavailable")
            expected_contract = (
                context.initial_source.splitlines()[0].encode("utf-8") + b"\n" + decoded
            )
        else:
            expected_contract = contract
        if contract != expected_contract:
            raise HeldoutProtocolError(
                "private decoded model response is not bound to its contract response"
            )

    if status == "SUCCESS":
        if rendered is None or decoded is None or contract is None:
            raise HeldoutProtocolError("successful generation lacks independently replayable raw evidence")
        try:
            proposal = ModelCandidateGenerator._parse_response(
                contract.decode("utf-8"), context, response_contract=context.response_contract
            )
            proposal.validate(context, source_limit=256 * 1024)
            evidence = CandidateGenerationEvidence(
                proposal=proposal,
                decoded_model_response=decoded,
                decoded_model_response_digest=digest_bytes(decoded),
                contract_response=contract,
                contract_response_digest=digest_bytes(contract),
                rendered_prompt=rendered,
                rendered_prompt_digest=digest_bytes(rendered),
                response_contract=context.response_contract,
            )
            evidence.validate(context)
        except Exception as exc:
            raise HeldoutProtocolError("successful generation raw response cannot be independently replayed") from exc
        return proposal

    try:
        failure = CandidateGenerationFailureEvidence(
            stage=stage,
            response_contract=context.response_contract,
            rendered_prompt=rendered,
            rendered_prompt_digest=record["rendered_prompt_digest"],
            decoded_model_response=decoded,
            decoded_model_response_digest=record["decoded_model_response_digest"],
            contract_response=contract,
            contract_response_digest=record["contract_response_digest"],
            error_code=record["error_code"],
        )
        failure.validate(context)
    except Exception as exc:
        raise HeldoutProtocolError("failed generation raw evidence cannot be revalidated") from exc
    if stage == "RESPONSE_CONTRACT":
        assert contract is not None
        ambient_exception = sys.exc_info()[1]
        try:
            proposal = ModelCandidateGenerator._parse_response(
                contract.decode("utf-8"), context, response_contract=context.response_contract
            )
            proposal.validate(context, source_limit=256 * 1024)
        except Exception as exc:
            try:
                observed_error_chain = list(
                    replay_exception_chain_classification(
                        exc,
                        ambient_exception=ambient_exception,
                    )
                )
            except Exception as chain_exc:
                raise HeldoutProtocolError(
                    "response-contract replay exception chain cannot be classified"
                ) from chain_exc
            if recorded_replay_error_chain != observed_error_chain:
                raise HeldoutProtocolError(
                    "response-contract failure chain changed on replay: {} != {}".format(
                        observed_error_chain, recorded_replay_error_chain
                    )
                ) from exc
        else:
            raise HeldoutProtocolError("response-contract failure unexpectedly parses on replay")
    return None


def _shock_evidence_bundle(
    *,
    protocol: FrozenHeldoutProtocol,
    deployment: SealedHeldoutDeploymentManifest,
    coordinate: HeldoutCoordinate,
    runtime_root: Path,
) -> Mapping[str, Any]:
    from .shock_engine import DurableShockOperationStore

    coordinate_root = Path(runtime_root) / coordinate.coordinate_id
    journal = ShockRuntimeJournal(coordinate_root / "shock-journal.json", coordinate).load()
    snapshot = _canonical_object_without_newline(
        Path(runtime_root) / "shock-blocks" / (coordinate.block_id + ".json"),
        "sealed shock block snapshot",
    )
    operations = [
        dict(value) for value in DurableShockOperationStore(coordinate_root / "shock-engine", coordinate).records()
    ]
    private_store = PrivateTrajectoryStore(coordinate_root / "private")
    source_materials = []
    for operation in operations:
        if operation["status"] not in {"COMPLETE", "GENERATION_FAILED"}:
            raise HeldoutProtocolError("completed shock coordinate contains a nonterminal operation")
        if operation["status"] == "COMPLETE":
            context, generation, record_digest = private_store.load_generation_success_read_only(
                operation["candidate_id"]
            )
            source = generation.proposal.source
            generation_record = _successful_generation_record(
                operation["candidate_id"], context, generation
            )
            raw_generation = _private_raw_generation_material(generation, status="SUCCESS")
            proposal: Optional[Mapping[str, Any]] = {
                "declared_locus": generation.proposal.declared_locus,
                "requested_authority": generation.proposal.requested_authority,
                "evidence_ids": list(generation.proposal.evidence_ids),
                "mutation_digest": generation.proposal.mutation_digest,
            }
            source_b64: Optional[str] = (
                base64.urlsafe_b64encode(source).decode("ascii").rstrip("=")
            )
            source_digest: Optional[str] = digest_bytes(source)
        else:
            context, failure, record_digest = private_store.load_generation_failure_read_only(
                operation["candidate_id"]
            )
            generation_record = _failed_generation_record(
                operation["candidate_id"], context, failure
            )
            raw_generation = _private_raw_generation_material(failure, status="FAILED")
            proposal = None
            source_b64 = None
            source_digest = None
        if (
            source_digest != operation["candidate_source_digest"]
            or record_digest != operation["generation_evidence_digest"]
            or record_digest != digest_bytes(canonical_bytes(generation_record))
            or context.run_id != operation["run_id"]
            or digest_for(asdict(context)) != operation["context_digest"]
        ):
            raise HeldoutProtocolError("shock private generation differs from durable operation state")
        source_materials.append(
            {
                "candidate_id": operation["candidate_id"],
                "candidate_source_b64": source_b64,
                "candidate_source_digest": source_digest,
                "generation_record_digest": record_digest,
                "generation_record": generation_record,
                "raw_generation": raw_generation,
                "proposal": proposal,
            }
        )
    ledger_path = coordinate_root / "ledger.sqlite3"
    with EvidenceLedger(ledger_path, mode="read_only") as ledger:
        ledger.verify_integrity()
        ledger_export = ledger.export_jsonl()
        receipt_root = digest_for(ledger.receipts())
        ledger_head = ledger.ledger_head_hash()
    unsigned: Dict[str, Any] = {
        "schema_version": SHOCK_EVIDENCE_SCHEMA,
        "deployment_manifest_digest": deployment.digest,
        "coordinate": coordinate.to_dict(protocol.digest, protocol.campaign_id),
        "journal": journal,
        "block_snapshot": dict(snapshot),
        "operations": operations,
        "source_materials": source_materials,
        "ledger_export": ledger_export,
        "ledger_export_digest": digest_bytes(ledger_export.encode("utf-8")),
        "receipt_collection_root": receipt_root,
        "ledger_head_digest": ledger_head,
    }
    return {**unsigned, "evidence_bundle_digest": digest_for(unsigned)}


def _observation(
    *,
    protocol: FrozenHeldoutProtocol,
    deployment: SealedHeldoutDeploymentManifest,
    coordinate: HeldoutCoordinate,
    result: Mapping[str, Any],
    evidence_bundle: Mapping[str, Any],
) -> Dict[str, Any]:
    exact_coordinate = coordinate.to_dict(protocol.digest, protocol.campaign_id)
    value = {
        "schema_version": OBSERVATION_SCHEMA,
        "deployment_manifest_digest": deployment.digest,
        "coordinate_digest": digest_for(exact_coordinate),
        "result": dict(result),
        "evidence_bundle": dict(evidence_bundle),
        "evidence_bundle_digest": digest_for(evidence_bundle),
    }
    if len(canonical_bytes(value)) > PRODUCTION_REQUEST_LIMIT:
        raise HeldoutProtocolError("held-out production observation exceeds its sealed byte limit")
    return value


class ProductionMainCoordinateRunner:
    """Turn exact bounded generation exhaustion into durable terminal evidence."""

    def __init__(
        self,
        runner: HeldoutCoordinateRunner,
        evidence_reader: AuthoritativeMainEvidenceReader,
    ) -> None:
        self.runner = runner
        self.evidence_reader = evidence_reader

    def __call__(self, coordinate: HeldoutCoordinate) -> Dict[str, Any]:
        started = monotonic()
        try:
            return self.runner(coordinate)
        except SourceContractBudgetExhausted as exhausted:
            coordinate_root = Path(self.runner.context.root) / coordinate.coordinate_id
            report, evidence = self.evidence_reader.record_source_exhaustion(
                coordinate,
                exhausted,
                PrivateTrajectoryStore(coordinate_root / "private"),
            )
            return _derive_source_exhausted_main_result(
                self.runner.context.protocol,
                coordinate,
                report,
                evidence,
                total_attempt_count=MAX_CANDIDATE_ATTEMPTS,
                wall_time_seconds=monotonic() - started,
            )


class ProductionCoordinateDispatcher:
    """Validate scheduler mappings before dispatching to one exact coordinate runner."""

    def __init__(
        self,
        *,
        protocol: FrozenHeldoutProtocol,
        deployment: SealedHeldoutDeploymentManifest,
        runtime_root: Path,
        main_runner: HeldoutCoordinateRunner,
        shock_runner: CorrectionShockCoordinateRunner,
        main_evidence_reader: AuthoritativeMainEvidenceReader,
    ) -> None:
        self.protocol = protocol
        self.deployment = deployment
        self.runtime_root = Path(runtime_root)
        self.main_runner = main_runner
        self.shock_runner = shock_runner
        self.main_evidence_reader = main_evidence_reader

    def coordinate_from_mapping(self, value: Mapping[str, Any]) -> HeldoutCoordinate:
        if not isinstance(value, Mapping):
            raise HeldoutProtocolError("scheduler coordinate input must be an object")
        coordinate_id = value.get("coordinate_id")
        coordinate = self.protocol.coordinate(coordinate_id)
        expected = coordinate.to_dict(self.protocol.digest, self.protocol.campaign_id)
        expected["idempotency_key"] = content_id(
            "idem",
            {"protocol_digest": self.protocol.digest, "coordinate_id": coordinate.coordinate_id},
        )
        if dict(value) != expected:
            raise HeldoutProtocolError("scheduler mapping differs from the exact frozen coordinate")
        return coordinate

    def __call__(self, coordinate_input: Mapping[str, Any]) -> Mapping[str, Any]:
        coordinate = self.coordinate_from_mapping(coordinate_input)
        if coordinate.phase == MAIN_PHASE:
            result = self.main_runner(coordinate)
            bundle = self.main_evidence_reader.bundle_for(coordinate, self.runtime_root)
        elif coordinate.phase == SHOCK_PHASE:
            result = self.shock_runner(coordinate)
            bundle = _shock_evidence_bundle(
                protocol=self.protocol,
                deployment=self.deployment,
                coordinate=coordinate,
                runtime_root=self.runtime_root,
            )
        else:
            raise HeldoutProtocolError("frozen coordinate phase has no production dispatcher")
        return _observation(
            protocol=self.protocol,
            deployment=self.deployment,
            coordinate=coordinate,
            result=result,
            evidence_bundle=bundle,
        )


def _verify_bundle_digest(bundle: Mapping[str, Any], fields: Sequence[str], label: str) -> None:
    _closed(bundle, tuple(fields) + ("evidence_bundle_digest",), label)
    unsigned = dict(bundle)
    supplied = unsigned.pop("evidence_bundle_digest")
    if supplied != digest_for(unsigned):
        raise HeldoutProtocolError("{} digest mismatch".format(label))


def _validate_closed_ledger_export(ledger_export: str) -> None:
    """Reject serialized ledger facts that replay would otherwise normalize away."""

    try:
        records = parse_canonical_jsonl(ledger_export)
    except (TypeError, ValueError) as exc:
        raise HeldoutProtocolError("evaluator ledger export is not canonical JSONL") from exc
    if not records:
        raise HeldoutProtocolError("evaluator ledger export is empty")
    checkpoints_started = False
    for record in records:
        if not isinstance(record, Mapping):
            raise HeldoutProtocolError("evaluator ledger export record is not an object")
        record_type = record.get("record_type")
        if record_type == "EVENT":
            if checkpoints_started:
                raise HeldoutProtocolError(
                    "evaluator ledger export event appears after checkpoint records"
                )
            expected_fields = set(_LEDGER_EVENT_EXPORT_FIELDS)
            if record.get("blob_digest") is not None:
                expected_fields.add("blob_media_type")
            if set(record) != expected_fields:
                raise HeldoutProtocolError("evaluator ledger EVENT export schema is not closed")
            if not isinstance(record.get("payload"), Mapping):
                raise HeldoutProtocolError("evaluator ledger EVENT payload is not an object")
            if record.get("blob_digest") is None:
                if not isinstance(record.get("payload_json"), str):
                    raise HeldoutProtocolError(
                        "evaluator inline EVENT lacks canonical payload storage"
                    )
            elif (
                record.get("payload_json") is not None
                or not isinstance(record.get("blob_media_type"), str)
                or not record["blob_media_type"]
            ):
                raise HeldoutProtocolError(
                    "evaluator blob EVENT has an inconsistent storage envelope"
                )
        elif record_type == "CHECKPOINT":
            checkpoints_started = True
            if set(record) != _LEDGER_CHECKPOINT_EXPORT_FIELDS:
                raise HeldoutProtocolError(
                    "evaluator ledger CHECKPOINT export schema is not closed"
                )
        else:
            raise HeldoutProtocolError("evaluator ledger export record type is unsupported")


def _require_signed_receipt_event_order(
    ledger: EvidenceLedger,
    events_by_id: Mapping[str, Mapping[str, Any]],
    label: str,
) -> None:
    event_sequences = []
    for receipt in ledger.receipts():
        row = ledger.connection.execute(
            "SELECT event_id FROM receipts WHERE receipt_id=?",
            (receipt["receipt_id"],),
        ).fetchone()
        event = events_by_id.get(str(row["event_id"])) if row is not None else None
        if event is None:
            raise HeldoutProtocolError("{} signed receipt lacks its ledger event".format(label))
        event_sequences.append(int(event["sequence"]))
    if event_sequences != sorted(event_sequences) or len(set(event_sequences)) != len(
        event_sequences
    ):
        raise HeldoutProtocolError(
            "{} receipt event order differs from the signed receipt chain".format(label)
        )


def _require_shock_lifecycle_event_order(
    events: Sequence[Mapping[str, Any]],
    operations: Sequence[Mapping[str, Any]],
) -> None:
    by_type: Dict[str, list[Mapping[str, Any]]] = {}
    for event in events:
        by_type.setdefault(str(event.get("event_type")), []).append(event)
    lifecycle_types = (
        "SHOCK_CORRECTED_PREMISE",
        "CORRECTION",
        "SHOCK_CORRECTION_COMMIT",
        "SHOCK_POLICY_ACTIVATION",
    )
    if any(len(by_type.get(event_type, ())) != 1 for event_type in lifecycle_types):
        raise HeldoutProtocolError("shock correction lifecycle event inventory is not exact")

    terminal_sequences: Dict[Tuple[str, int], int] = {}
    for operation in operations:
        event_type = (
            "SHOCK_ATTEMPT"
            if operation["status"] == "COMPLETE"
            else "SHOCK_GENERATION_FAILURE"
        )
        matches = [
            event
            for event in by_type.get(event_type, ())
            if event.get("subject_id") == operation["candidate_id"]
            and isinstance(event.get("payload"), Mapping)
            and event["payload"].get("operation_id") == operation["operation_id"]
        ]
        if len(matches) != 1:
            raise HeldoutProtocolError("shock terminal operation event inventory is not exact")
        terminal_sequences[(str(operation["phase"]), int(operation["attempt"]))] = int(
            matches[0]["sequence"]
        )

    lifecycle_sequences = [int(by_type[event_type][0]["sequence"]) for event_type in lifecycle_types]
    pre_sequences = [
        sequence
        for (phase, _attempt), sequence in terminal_sequences.items()
        if phase == "PRE"
    ]
    post_sequences = [
        sequence
        for (phase, _attempt), sequence in terminal_sequences.items()
        if phase == "POST"
    ]
    candidate_phases = {
        str(operation["candidate_id"]): str(operation["phase"])
        for operation in operations
    }
    if not set(candidate_phases.values()).issubset({"PRE", "POST"}):
        raise HeldoutProtocolError("shock operation phase inventory is invalid")
    operation_event_types = {
        "CANDIDATE",
        "SHOCK_GENERATION_FAILURE",
        "RECEIPT",
        "VERDICT",
        "EFFECT_RECEIPT",
        "SHOCK_ATTEMPT",
        "DEPENDENCY",
    }
    operation_event_sequences: Dict[str, list[int]] = {"PRE": [], "POST": []}
    for event in events:
        if event.get("event_type") not in operation_event_types:
            continue
        phase = candidate_phases.get(str(event.get("subject_id")))
        if phase is not None:
            operation_event_sequences[phase].append(int(event["sequence"]))
    if (
        not pre_sequences
        or max(pre_sequences) >= lifecycle_sequences[0]
        or lifecycle_sequences != sorted(lifecycle_sequences)
        or len(set(lifecycle_sequences)) != len(lifecycle_sequences)
        or (post_sequences and lifecycle_sequences[-1] >= min(post_sequences))
        or not operation_event_sequences["PRE"]
        or max(operation_event_sequences["PRE"]) >= lifecycle_sequences[0]
        or (
            operation_event_sequences["POST"]
            and min(operation_event_sequences["POST"]) <= lifecycle_sequences[-1]
        )
    ):
        raise HeldoutProtocolError(
            "shock correction lifecycle order differs from PRE and POST execution"
        )


def _require_exact_shock_event_order(
    ledger: EvidenceLedger,
    events: Sequence[Mapping[str, Any]],
    operations: Sequence[Mapping[str, Any]],
    *,
    campaign_event: Mapping[str, Any],
    premise_event: Mapping[str, Any],
    unrelated_root: Mapping[str, Any],
    unrelated_child: Mapping[str, Any],
    replacement_event: Mapping[str, Any],
    correction_record_event: Mapping[str, Any],
    correction_commit_event: Mapping[str, Any],
    policy_event: Mapping[str, Any],
    protocol: FrozenHeldoutProtocol,
    coordinate: HeldoutCoordinate,
) -> None:
    """Bind the deterministic single-writer shock emission order exactly."""

    dependency_rows = [
        dict(row)
        for row in ledger.connection.execute(
            "SELECT * FROM dependencies ORDER BY dependency_id"
        ).fetchall()
    ]
    run_event_ids = {
        str(row["run_id"]): str(row["event_id"])
        for row in ledger.connection.execute("SELECT run_id,event_id FROM runs")
    }

    def sole_projection_event_id(table: str, candidate_id: str) -> str:
        rows = ledger.connection.execute(
            "SELECT event_id FROM {} WHERE candidate_id=?".format(table),
            (candidate_id,),
        ).fetchall()
        if len(rows) != 1:
            raise HeldoutProtocolError(
                "shock {} projection event inventory is not exact".format(table)
            )
        return str(rows[0]["event_id"])

    def sole_domain_event_id(event_type: str, operation: Mapping[str, Any]) -> str:
        matches = [
            event
            for event in events
            if event.get("event_type") == event_type
            and event.get("subject_id") == operation["candidate_id"]
            and isinstance(event.get("payload"), Mapping)
            and event["payload"].get("operation_id") == operation["operation_id"]
        ]
        if len(matches) != 1:
            raise HeldoutProtocolError(
                "shock {} operation event inventory is not exact".format(event_type)
            )
        return str(matches[0]["event_id"])

    def operation_event_ids(operation: Mapping[str, Any]) -> list[str]:
        candidate_id = str(operation["candidate_id"])
        dependencies = sorted(
            (row for row in dependency_rows if str(row["child_id"]) == candidate_id),
            key=lambda row: (str(row["parent_id"]), str(row["edge_type"])),
        )
        dependency_event_ids = [str(row["insertion_event_id"]) for row in dependencies]
        if operation["status"] == "GENERATION_FAILED":
            return [
                sole_domain_event_id("SHOCK_GENERATION_FAILURE", operation),
                *dependency_event_ids,
            ]
        candidate_event_id = sole_projection_event_id("candidates", candidate_id)
        receipt_event_ids = []
        for receipt_id in operation["evaluation_result"]["receipt_ids"]:
            rows = ledger.connection.execute(
                "SELECT event_id FROM receipts WHERE receipt_id=?",
                (receipt_id,),
            ).fetchall()
            if len(rows) != 1:
                raise HeldoutProtocolError(
                    "shock receipt projection event inventory is not exact"
                )
            receipt_event_ids.append(str(rows[0]["event_id"]))
        return [
            candidate_event_id,
            *dependency_event_ids,
            *receipt_event_ids,
            sole_projection_event_id("verdicts", candidate_id),
            sole_projection_event_id("effect_receipts", candidate_id),
            sole_domain_event_id("SHOCK_ATTEMPT", operation),
        ]

    unrelated_dependency_rows = [
        row
        for row in dependency_rows
        if str(row["child_id"]) == str(unrelated_child["event_id"])
    ]
    pre_run_id = _shock_run_id(protocol, coordinate, "PRE")
    post_run_id = _shock_run_id(protocol, coordinate, "POST")
    if (
        len(unrelated_dependency_rows) != 1
        or set(run_event_ids) != {pre_run_id, post_run_id}
    ):
        raise HeldoutProtocolError("shock global event-order prerequisites are not exact")
    pre_operations = sorted(
        (operation for operation in operations if operation["phase"] == "PRE"),
        key=lambda operation: int(operation["attempt"]),
    )
    post_operations = sorted(
        (operation for operation in operations if operation["phase"] == "POST"),
        key=lambda operation: int(operation["attempt"]),
    )
    expected_event_ids = [
        str(campaign_event["event_id"]),
        run_event_ids[pre_run_id],
        str(premise_event["event_id"]),
        str(unrelated_root["event_id"]),
        str(unrelated_child["event_id"]),
        str(unrelated_dependency_rows[0]["insertion_event_id"]),
    ]
    for operation in pre_operations:
        expected_event_ids.extend(operation_event_ids(operation))
    expected_event_ids.extend(
        (
            str(replacement_event["event_id"]),
            str(correction_record_event["event_id"]),
            str(correction_commit_event["event_id"]),
            str(policy_event["event_id"]),
            run_event_ids[post_run_id],
        )
    )
    for operation in post_operations:
        expected_event_ids.extend(operation_event_ids(operation))
    if tuple(event["event_id"] for event in events) != tuple(expected_event_ids):
        raise HeldoutProtocolError(
            "shock ledger event order differs from deterministic protocol execution"
        )


class ProductionObservationVerifier:
    """Evaluator-owned reconstruction of results from sealed private evidence."""

    MAIN_FIELDS = (
        "schema_version",
        "deployment_manifest_digest",
        "coordinate",
        "report",
        "runtime_evidence",
        "token_materials",
        "generation_failures",
        "ledger_export",
        "ledger_export_digest",
        "receipt_collection_root",
        "ledger_head_digest",
    )
    SHOCK_FIELDS = (
        "schema_version",
        "deployment_manifest_digest",
        "coordinate",
        "journal",
        "block_snapshot",
        "operations",
        "source_materials",
        "ledger_export",
        "ledger_export_digest",
        "receipt_collection_root",
        "ledger_head_digest",
    )

    def __init__(
        self,
        *,
        protocol: FrozenHeldoutProtocol,
        deployment: SealedHeldoutDeploymentManifest,
        trainer_sources: HeldoutTrainerSources,
        evaluator_public_key: bytes,
        tokenizer: Any,
    ) -> None:
        self.protocol = protocol
        self.deployment = deployment
        if (
            not isinstance(trainer_sources, HeldoutTrainerSources)
            or trainer_sources.digest != deployment["trainer_sources_digest"]
            or trainer_sources.campaign_id != protocol.campaign_id
            or trainer_sources.protocol_digest != protocol.digest
        ):
            raise HeldoutProtocolError("production observation verifier source bundle is stale")
        self.trainer_sources = trainer_sources
        self.evaluator_public_key = load_public_key(evaluator_public_key)
        self.tokenizer = tokenizer

    def __call__(self, coordinate_value: Mapping[str, Any], observation: Mapping[str, Any]) -> EvaluatorVerification:
        _closed(
            observation,
            (
                "schema_version",
                "deployment_manifest_digest",
                "coordinate_digest",
                "result",
                "evidence_bundle",
                "evidence_bundle_digest",
            ),
            "production held-out observation",
        )
        if (
            observation["schema_version"] != OBSERVATION_SCHEMA
            or observation["deployment_manifest_digest"] != self.deployment.digest
            or observation["coordinate_digest"] != digest_for(coordinate_value)
            or observation["evidence_bundle_digest"] != digest_for(observation["evidence_bundle"])
        ):
            raise HeldoutProtocolError("production observation identity binding mismatch")
        coordinate = self.protocol.coordinate(coordinate_value.get("coordinate_id"))
        if dict(coordinate_value) != coordinate.to_dict(self.protocol.digest, self.protocol.campaign_id):
            raise HeldoutProtocolError("evaluator received a substituted frozen coordinate")
        result = validate_result(self.protocol, observation["result"])
        if result["coordinate_id"] != coordinate.coordinate_id:
            raise HeldoutProtocolError("production observation result belongs to another coordinate")
        bundle = observation["evidence_bundle"]
        if coordinate.phase == MAIN_PHASE:
            verified_result, receipt_root, ledger_head = self._verify_main(coordinate, bundle, result)
        else:
            verified_result, receipt_root, ledger_head = self._verify_shock(coordinate, bundle, result)
        return EvaluatorVerification(
            result=verified_result,
            receipt_collection_root=receipt_root,
            ledger_head_digest=ledger_head,
        )

    def _replay_ledger(self, bundle: Mapping[str, Any]) -> Tuple[EvidenceLedger, tempfile.TemporaryDirectory]:
        ledger_export = bundle["ledger_export"]
        if (
            not isinstance(ledger_export, str)
            or digest_bytes(ledger_export.encode("utf-8")) != bundle["ledger_export_digest"]
        ):
            raise HeldoutProtocolError("evaluator ledger export digest mismatch")
        _validate_closed_ledger_export(ledger_export)
        temporary = tempfile.TemporaryDirectory(prefix="egv-heldout-evaluator-replay-")
        ledger = None
        try:
            ledger = EvidenceLedger.replay_jsonl(
                ledger_export,
                Path(temporary.name) / "ledger.sqlite3",
            )
            ledger.verify_integrity()
            ledger.verify_receipt_chain(self.evaluator_public_key)
            if (
                ledger.export_jsonl() != ledger_export
                or digest_for(ledger.receipts()) != bundle["receipt_collection_root"]
                or ledger.ledger_head_hash() != bundle["ledger_head_digest"]
            ):
                raise HeldoutProtocolError(
                    "evaluator replay export or roots differ from the observation"
                )
            return ledger, temporary
        except Exception:
            if ledger is not None:
                ledger.close()
            temporary.cleanup()
            raise

    def _verify_main(
        self,
        coordinate: HeldoutCoordinate,
        bundle: Mapping[str, Any],
        supplied_result: Mapping[str, Any],
    ) -> Tuple[Mapping[str, Any], str, str]:
        _verify_bundle_digest(bundle, self.MAIN_FIELDS, "main evidence bundle")
        if (
            bundle["schema_version"] != MAIN_EVIDENCE_SCHEMA
            or bundle["deployment_manifest_digest"] != self.deployment.digest
            or bundle["coordinate"] != coordinate.to_dict(self.protocol.digest, self.protocol.campaign_id)
        ):
            raise HeldoutProtocolError("main evidence bundle identity mismatch")
        tasks = [
            dict(record)
            for record in self.protocol.heldout_task_records
            if record["template_id"] == coordinate.task_id
        ]
        if (
            len(tasks) != 1
            or self.protocol.bindings["policy_manifest_digest"]
            != AuthorityPolicy.candidate_execution().digest
        ):
            raise HeldoutProtocolError("main task or authority policy binding is unavailable")
        report = _report_from_mapping(bundle["report"])
        evidence = _runtime_evidence_from_mapping(bundle["runtime_evidence"])
        policy = arm_policy(coordinate.treatment)
        run_id = _main_run_id(self.protocol, coordinate)
        expected_adapter = (
            self.protocol.bindings["adapter_digest"] if policy.requires_adapter else None
        )
        if (
            report.campaign_id != self.protocol.campaign_id
            or report.run_id != run_id
            or report.arm_id != coordinate.treatment
            or report.task_id != coordinate.task_id
            or report.seed != coordinate.seed
            or report.model_digest != self.protocol.bindings["base_model_digest"]
            or report.adapter_digest != expected_adapter
            or report.retrieval_policy != policy.retrieval_policy
            or report.authority_enforced != policy.authority_enforced
            or len(report.attempts) > self.deployment["max_attempts"]
        ):
            raise HeldoutProtocolError("main Variation report crossed its exact coordinate")
        attempt_indices = [attempt.attempt_index for attempt in report.attempts]
        candidate_ids = [attempt.candidate_id for attempt in report.attempts]
        if (
            attempt_indices != sorted(attempt_indices)
            or len(attempt_indices) != len(set(attempt_indices))
            or len(candidate_ids) != len(set(candidate_ids))
            or any(
                type(index) is not int or not 1 <= index <= self.deployment["max_attempts"]
                for index in attempt_indices
            )
        ):
            raise HeldoutProtocolError("main Variation attempt order or identities are invalid")
        materials, failures, entries = self._main_generation_inventory(bundle, report)
        ledger, temporary = self._replay_ledger(bundle)
        try:
            tokens = self._verify_main_trajectory(
                coordinate=coordinate,
                task_record=tasks[0],
                policy=policy,
                run_id=run_id,
                expected_adapter=expected_adapter,
                report=report,
                materials=materials,
                failures=failures,
                entries=entries,
                ledger=ledger,
            )
            replay = _replay_main_receipts(
                ledger,
                report.attempts,
                protocol=self.protocol,
                coordinate=coordinate,
                task_record=tasks[0],
                run_id=run_id,
            )
            expected_evidence = HeldoutRuntimeEvidence(
                tokens=tokens,
                evaluator_seconds=replay[7],
                private_replay_decisions=replay[0],
                private_replay_agreements=replay[1],
                public_replay_decisions=replay[2],
                public_replay_agreements=replay[3],
                authority_challenges=replay[4],
                authority_challenges_valid_denials=replay[5],
                unauthorized_successful_effects=replay[6],
            )
            if _runtime_evidence_dict(expected_evidence) != _runtime_evidence_dict(evidence):
                raise HeldoutProtocolError("main runtime evidence differs from evaluator replay")
            expected_integrity = ledger.verify_integrity()
            source_exhausted = (
                report.terminal_status == "BUDGET_EXHAUSTED"
                and report.checkpoint_path == "source-contract-budget-exhausted"
                and len(entries) == self.deployment["max_attempts"]
                and entries[-1][1] == "FAILED"
            )
            if source_exhausted:
                if any(
                    attempt.disposition != "REJECTED"
                    or attempt.diagnostic_enum == Diagnostic.INTERNAL_ERROR.value
                    for attempt in report.attempts
                ):
                    raise HeldoutProtocolError(
                        "main source-exhausted evaluated decisions are not exact rejections"
                    )
                expected_result = _derive_source_exhausted_main_result(
                    self.protocol,
                    coordinate,
                    report,
                    expected_evidence,
                    total_attempt_count=len(entries),
                    wall_time_seconds=float(supplied_result["costs"]["wall_time_seconds"]),
                )
            elif report.attempts:
                last = report.attempts[-1]
                expected_terminal = (
                    "FAILED"
                    if last.diagnostic_enum == Diagnostic.INTERNAL_ERROR.value
                    else "PROMOTED"
                    if last.disposition == "PROMOTED"
                    else "BUDGET_EXHAUSTED"
                    if last.attempt_index == self.deployment["max_attempts"]
                    else None
                )
                if (
                    report.terminal_status != expected_terminal
                    or any(attempt.disposition == "PROMOTED" for attempt in report.attempts[:-1])
                ):
                    raise HeldoutProtocolError("main report terminal facts differ from evaluator replay")
                expected_result = derive_verified_main_result(
                    self.protocol,
                    coordinate,
                    report,
                    expected_evidence,
                    wall_time_seconds=float(supplied_result["costs"]["wall_time_seconds"]),
                )
            else:
                raise HeldoutProtocolError("main report has no terminal trajectory evidence")
            if (
                report.ledger_head_hash != ledger.ledger_head_hash()
                or dict(report.ledger_integrity) != expected_integrity
            ):
                raise HeldoutProtocolError("main report ledger facts differ from evaluator replay")
            if expected_result != dict(supplied_result):
                raise HeldoutProtocolError("main result differs from evaluator-owned reconstruction")
        finally:
            ledger.close()
            temporary.cleanup()
        return expected_result, bundle["receipt_collection_root"], bundle["ledger_head_digest"]

    def _main_generation_inventory(
        self,
        bundle: Mapping[str, Any],
        report: VariationReport,
    ) -> Tuple[
        Mapping[str, Mapping[str, Any]],
        Mapping[str, Mapping[str, Any]],
        Tuple[Tuple[int, str, str, Mapping[str, Any]], ...],
    ]:
        raw_materials = bundle["token_materials"]
        raw_failures = bundle["generation_failures"]
        if (
            not isinstance(raw_materials, list)
            or not isinstance(raw_failures, list)
            or len(raw_materials) != len(report.attempts)
        ):
            raise HeldoutProtocolError("main generation evidence inventory is malformed")
        materials: Dict[str, Mapping[str, Any]] = {}
        for material in raw_materials:
            _closed(
                material,
                (
                    "candidate_id",
                    "candidate_source_b64",
                    "candidate_source_digest",
                    "candidate_source_tokens",
                    "generation_record_digest",
                    "generation_record",
                    "raw_generation",
                    "proposal",
                ),
                "main candidate generation evidence",
            )
            candidate_id = material["candidate_id"]
            if not isinstance(candidate_id, str) or candidate_id in materials:
                raise HeldoutProtocolError("main successful generation identity is duplicated")
            materials[candidate_id] = material
        report_ids = {attempt.candidate_id for attempt in report.attempts}
        if set(materials) != report_ids:
            raise HeldoutProtocolError("main successful generation inventory differs from report candidates")
        failures: Dict[str, Mapping[str, Any]] = {}
        for item in raw_failures:
            _closed(
                item,
                ("candidate_id", "generation_record_digest", "generation_record", "raw_generation"),
                "main failed generation evidence",
            )
            candidate_id = item["candidate_id"]
            if (
                not isinstance(candidate_id, str)
                or candidate_id in failures
                or candidate_id in materials
            ):
                raise HeldoutProtocolError("main failed generation identity is duplicated")
            failures[candidate_id] = item
        entries = []
        for status, inventory in (("SUCCESS", materials), ("FAILED", failures)):
            for candidate_id, item in inventory.items():
                record = item["generation_record"]
                _closed(record, _GENERATION_RECORD_FIELDS, "main private generation record")
                context = record["context"]
                if (
                    not isinstance(context, Mapping)
                    or record["candidate_id"] != candidate_id
                    or record["status"] != status
                    or item["generation_record_digest"] != digest_bytes(canonical_bytes(record))
                    or type(context.get("attempt_index")) is not int
                ):
                    raise HeldoutProtocolError("main private generation record is not exact")
                entries.append((int(context["attempt_index"]), status, candidate_id, item))
        entries.sort(key=lambda entry: entry[0])
        indices = [entry[0] for entry in entries]
        successful = [(entry[0], entry[2]) for entry in entries if entry[1] == "SUCCESS"]
        successful_matches = successful == [
            (attempt.attempt_index, attempt.candidate_id) for attempt in report.attempts
        ]
        normal_terminal = bool(report.attempts) and entries[-1][1] == "SUCCESS"
        source_exhausted_terminal = (
            report.terminal_status == "BUDGET_EXHAUSTED"
            and report.checkpoint_path == "source-contract-budget-exhausted"
            and len(entries) == self.deployment["max_attempts"]
            and entries[-1][1] == "FAILED"
        )
        if (
            not entries
            or indices != list(range(1, max(indices) + 1))
            or max(indices) > self.deployment["max_attempts"]
            or not successful_matches
            or not (normal_terminal or source_exhausted_terminal)
        ):
            raise HeldoutProtocolError("main generation evidence does not form one bounded trajectory")
        return MappingProxyType(materials), MappingProxyType(failures), tuple(entries)

    def _verify_main_trajectory(
        self,
        *,
        coordinate: HeldoutCoordinate,
        task_record: Mapping[str, Any],
        policy: Any,
        run_id: str,
        expected_adapter: Optional[str],
        report: VariationReport,
        materials: Mapping[str, Mapping[str, Any]],
        failures: Mapping[str, Mapping[str, Any]],
        entries: Sequence[Tuple[int, str, str, Mapping[str, Any]]],
        ledger: EvidenceLedger,
    ) -> int:
        events = ledger.events()
        events_by_id = {str(event["event_id"]): event for event in events}
        campaigns = [dict(row) for row in ledger.connection.execute("SELECT * FROM campaigns")]
        runs = [dict(row) for row in ledger.connection.execute("SELECT * FROM runs")]
        if len(campaigns) != 1 or len(runs) != 1:
            raise HeldoutProtocolError("main ledger campaign or run inventory is not exact")
        campaign = campaigns[0]
        run = runs[0]
        expected_campaign = {
            "campaign_id": self.protocol.campaign_id,
            "protocol_hash": VARIATION_PROTOCOL_DIGEST,
            "source_commit": self.deployment["source_commit"],
            "model_revision": MODEL_REVISION,
            "data_manifest_hash": self.protocol.bindings["data_manifest_digest"],
            "evaluator_hash": self.protocol.bindings["evaluator_digest"],
            "policy_hash": self.protocol.bindings["policy_manifest_digest"],
            "seed_set_json": canonical_json(list(self.protocol.seeds)),
        }
        expected_run = {
            "run_id": run_id,
            "campaign_id": self.protocol.campaign_id,
            "arm": coordinate.treatment,
            "task_id": coordinate.task_id,
            "seed": coordinate.seed,
            "parent_checkpoint": None,
            "start_state": "READY",
            "end_state": None,
            "host_role": "spark_trainer",
            "software_manifest_hash": digest_for(
                {
                    "model": self.protocol.bindings["base_model_digest"],
                    "protocol": VARIATION_PROTOCOL_DIGEST,
                }
            ),
        }
        if any(campaign.get(key) != value for key, value in expected_campaign.items()) or any(
            run.get(key) != value for key, value in expected_run.items()
        ):
            raise HeldoutProtocolError("main ledger campaign or run binding was substituted")
        campaign_event = events_by_id.get(str(campaign["event_id"]))
        run_event = events_by_id.get(str(run["event_id"]))
        campaign_payload = {
            "campaign_id": self.protocol.campaign_id,
            "protocol_hash": VARIATION_PROTOCOL_DIGEST,
            "source_commit": self.deployment["source_commit"],
            "model_revision": MODEL_REVISION,
            "data_manifest_hash": self.protocol.bindings["data_manifest_digest"],
            "evaluator_hash": self.protocol.bindings["evaluator_digest"],
            "policy_hash": self.protocol.bindings["policy_manifest_digest"],
            "seed_set": list(self.protocol.seeds),
            "created_at": campaign["created_at"],
        }
        run_payload = {
            **expected_run,
            "created_at": run["created_at"],
        }
        if not _exact_event_envelope(
            campaign_event,
            event_type="CAMPAIGN",
            payload=campaign_payload,
            campaign_id=self.protocol.campaign_id,
            run_id=None,
            task_id=None,
            subject_id=self.protocol.campaign_id,
            source_class=None,
            disposition="OBSERVED",
            evaluator_identity=None,
            idempotency_key=None,
        ) or not _exact_event_envelope(
            run_event,
            event_type="RUN",
            payload=run_payload,
            campaign_id=self.protocol.campaign_id,
            run_id=run_id,
            task_id=coordinate.task_id,
            subject_id=run_id,
            source_class=None,
            disposition="OBSERVED",
            evaluator_identity=None,
            idempotency_key=None,
        ):
            raise HeldoutProtocolError("main ledger campaign/run event payload is not exact")

        candidate_rows = {
            str(row["candidate_id"]): dict(row)
            for row in ledger.connection.execute("SELECT * FROM candidates ORDER BY candidate_id")
        }
        if set(candidate_rows) != {attempt.candidate_id for attempt in report.attempts}:
            raise HeldoutProtocolError("main candidate projection inventory differs from report candidates")
        candidate_sequences = {
            candidate_id: int(events_by_id[str(row["event_id"])]["sequence"])
            for candidate_id, row in candidate_rows.items()
        }
        valid_sequences = {
            str(event["event_id"]): int(event["sequence"])
            for event in ledger.current_valid_events()
        }
        available = retrieval_policy(policy.retrieval_policy).retrieve(
            ledger,
            campaign_id=self.protocol.campaign_id,
            arm_id=coordinate.treatment,
            task_id=coordinate.task_id,
            isolation=None,
        )
        initial_source = self.trainer_sources.source_for(coordinate.task_id)
        try:
            initial_source_text = initial_source.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise HeldoutProtocolError("main trainer source is not UTF-8") from exc
        attempts = {attempt.attempt_index: attempt for attempt in report.attempts}
        expected_dependencies: set[Tuple[str, str, str]] = set()
        attempt_event_ids: list[str] = []
        attempt_events: Dict[int, Mapping[str, Any]] = {}
        failure_summaries: Dict[int, Mapping[str, Any]] = {}
        previous_candidate: Optional[str] = None
        token_total = 0
        for position, (attempt_index, status, candidate_id, item) in enumerate(entries):
            expected_id = _main_candidate_id(
                self.protocol,
                coordinate,
                run_id=run_id,
                attempt=attempt_index,
                parent=previous_candidate,
            )
            if candidate_id != expected_id:
                raise HeldoutProtocolError("main candidate identity is not derived from exact lineage")
            cutoff: Optional[int] = None
            if status == "SUCCESS":
                cutoff = candidate_sequences[candidate_id]
            else:
                next_success = next(
                    (entry for entry in entries[position + 1 :] if entry[1] == "SUCCESS"),
                    None,
                )
                if next_success is not None:
                    cutoff = candidate_sequences[next_success[2]]
            retrieval_records = tuple(
                record.to_dict()
                for record in available.records
                if cutoff is None or valid_sequences[record.event_id] < cutoff
            )
            context_without_prompt = CandidateContext(
                campaign_id=self.protocol.campaign_id,
                run_id=run_id,
                seed=coordinate.seed,
                arm_id=coordinate.treatment,
                task_id=coordinate.task_id,
                family_id=task_record["family_id"],
                public_locus=task_record["public_locus"],
                public_rule_id=task_record["public_rule_id"],
                attempt_index=attempt_index,
                parent_candidate_id=previous_candidate,
                retrieval_records=retrieval_records,
                retrieval_digest=digest_for(list(retrieval_records)),
                model_digest=self.protocol.bindings["base_model_digest"],
                adapter_digest=expected_adapter,
                prompt_digest=GENESIS_HASH,
                task_statement="Repair the bounded {} task at {}.".format(
                    task_record["family_id"], task_record["public_locus"]
                ),
                initial_source=initial_source_text,
                initial_source_digest=digest_bytes(initial_source),
                response_contract="source-only-v1",
                response_contract_digest=SOURCE_ONLY_RESPONSE_CONTRACT_DIGEST,
                generation_profile_digest=self.deployment["generation_profile_digest"],
            )
            prompt_digest = _candidate_prompt_digest(self.tokenizer, context_without_prompt)
            exact_context = CandidateContext(
                **{**asdict(context_without_prompt), "prompt_digest": prompt_digest}
            )
            generation_record = item["generation_record"]
            if (
                dict(generation_record["context"]) != _generation_context_value(exact_context)
                or generation_record["schema_version"] != PRIVATE_GENERATION_SCHEMA
                or generation_record["response_contract"] != "source-only-v1"
                or generation_record["response_contract_digest"]
                != SOURCE_ONLY_RESPONSE_CONTRACT_DIGEST
                or generation_record["generation_profile_digest"]
                != self.deployment["generation_profile_digest"]
                or generation_record["rendered_prompt_digest"] != prompt_digest
            ):
                raise HeldoutProtocolError("main generation context or prompt was substituted")
            raw_proposal = _verify_private_raw_generation(
                item["raw_generation"], generation_record, exact_context
            )
            if status == "FAILED":
                if raw_proposal is not None:
                    raise HeldoutProtocolError("failed generation unexpectedly replayed as a proposal")
                if (
                    generation_record["failure_stage"] != "RESPONSE_CONTRACT"
                    or not isinstance(generation_record["error_code"], str)
                    or not generation_record["error_code"]
                    or len(generation_record["error_code"]) > 128
                    or generation_record["proposal_source_digest"] is not None
                    or validate_sha256(
                        generation_record["decoded_model_response_digest"],
                        "main failed decoded response digest",
                    )
                    != generation_record["contract_response_digest"]
                ):
                    raise HeldoutProtocolError("main failed generation is not an exact source-contract failure")
                failure_summaries[attempt_index] = {
                    "attempt_index": attempt_index,
                    "candidate_id": candidate_id,
                    "parent_candidate_id": previous_candidate,
                    "record_digest": item["generation_record_digest"],
                    "prompt_digest": prompt_digest,
                }
                continue

            attempt = attempts[attempt_index]
            material = materials[candidate_id]
            source = _decode_canonical_b64(material["candidate_source_b64"], "candidate source")
            count = _token_count(self.tokenizer, source)
            proposal = material["proposal"]
            _closed(
                proposal,
                ("declared_locus", "requested_authority", "evidence_ids", "mutation_digest"),
                "main candidate proposal",
            )
            evidence_ids = [str(record["event_id"]) for record in retrieval_records]
            if (
                dict(proposal)
                != {
                    "declared_locus": task_record["public_locus"],
                    "requested_authority": "EXECUTE_CANDIDATE",
                    "evidence_ids": evidence_ids,
                    "mutation_digest": digest_bytes(source),
                }
                or raw_proposal is None
                or raw_proposal.source != source
                or raw_proposal.declared_locus != proposal["declared_locus"]
                or raw_proposal.requested_authority != proposal["requested_authority"]
                or list(raw_proposal.evidence_ids) != proposal["evidence_ids"]
                or raw_proposal.mutation_digest != proposal["mutation_digest"]
                or material["candidate_source_digest"] != digest_bytes(source)
                or material["candidate_source_tokens"] != count
                or attempt.candidate_artifact_digest != digest_bytes(source)
                or generation_record["proposal_source_digest"] != digest_bytes(source)
                or generation_record["failure_stage"] is not None
                or generation_record["error_code"] is not None
                or validate_sha256(
                    generation_record["decoded_model_response_digest"],
                    "main decoded response digest",
                )
                != generation_record["contract_response_digest"]
            ):
                raise HeldoutProtocolError("main candidate source or proposal evidence was substituted")
            token_total += count
            expected_metadata = {
                "schema_version": "egv-variation-candidate-v1",
                "arm_id": coordinate.treatment,
                "attempt_index": attempt_index,
                "public_rule_id": task_record["public_rule_id"],
                "public_locus": task_record["public_locus"],
                "retrieval_digest": digest_for(list(retrieval_records)),
                "evidence_ids": evidence_ids,
                "candidate_artifact_digest": digest_bytes(source),
                "response_contract": "source-only-v1",
                "response_contract_digest": SOURCE_ONLY_RESPONSE_CONTRACT_DIGEST,
                "generation_evidence_digest": item["generation_record_digest"],
                "generation_profile_digest": self.deployment["generation_profile_digest"],
            }
            expected_candidate = {
                "candidate_id": candidate_id,
                "campaign_id": self.protocol.campaign_id,
                "run_id": run_id,
                "task_id": coordinate.task_id,
                "parent_candidate_id": previous_candidate,
                "mutation_family": task_record["family_id"],
                "patch_hash": digest_bytes(source),
                "requested_authority": "EXECUTE_CANDIDATE",
                "prompt_hash": prompt_digest,
                "model_hash": self.protocol.bindings["base_model_digest"],
                "adapter_hash": expected_adapter,
                "metadata": expected_metadata,
            }
            row = candidate_rows[candidate_id]
            candidate_event = events_by_id.get(str(row["event_id"]))
            if (
                row["candidate_json"] != canonical_json(expected_candidate)
                or not _exact_event_envelope(
                    candidate_event,
                    event_type="CANDIDATE",
                    payload=expected_candidate,
                    campaign_id=self.protocol.campaign_id,
                    run_id=run_id,
                    task_id=coordinate.task_id,
                    subject_id=candidate_id,
                    source_class=None,
                    disposition="OBSERVED",
                    evaluator_identity=None,
                    idempotency_key=None,
                )
            ):
                raise HeldoutProtocolError("main candidate ledger projection was substituted")
            expected_dependencies.update(
                (evidence_id, candidate_id, "EVIDENCE_USED") for evidence_id in evidence_ids
            )
            attempt_payload = {
                "schema_version": "egv-variation-attempt-v1",
                "attempt_index": attempt_index,
                "candidate_id": candidate_id,
                "arm_id": coordinate.treatment,
                "retrieval_policy": policy.retrieval_policy,
                "retrieval_digest": digest_for(list(retrieval_records)),
                "evidence_ids": evidence_ids,
                "candidate_artifact_digest": digest_bytes(source),
                "diagnostic_enum": attempt.diagnostic_enum,
                "resource_bucket": attempt.resource_bucket,
                "disposition": attempt.disposition,
                "receipt_ids": list(attempt.receipt_ids),
            }
            matching = [
                event
                for event in events
                if event.get("event_type") == "VARIATION_ATTEMPT"
                and event.get("subject_id") == candidate_id
            ]
            if (
                len(matching) != 1
                or not _exact_event_envelope(
                    matching[0],
                    event_type="VARIATION_ATTEMPT",
                    payload=attempt_payload,
                    campaign_id=self.protocol.campaign_id,
                    run_id=run_id,
                    task_id=coordinate.task_id,
                    subject_id=candidate_id,
                    source_class="GENERATOR",
                    disposition=attempt.disposition,
                    evaluator_identity=None,
                    idempotency_key="variation-attempt:{}:{}".format(
                        candidate_id, attempt_index
                    ),
                )
                or attempt.ledger_head_hash != matching[0].get("event_hash")
                or attempt.retrieval_digest != attempt_payload["retrieval_digest"]
                or list(attempt.evidence_ids) != evidence_ids
            ):
                raise HeldoutProtocolError("main attempt ledger event or retrieval was substituted")
            attempt_event_ids.append(str(matching[0]["event_id"]))
            attempt_events[attempt_index] = matching[0]
            previous_candidate = candidate_id

        self._verify_main_ledger_projections(
            coordinate=coordinate,
            task_record=task_record,
            policy=policy,
            run_id=run_id,
            expected_adapter=expected_adapter,
            report=report,
            ledger=ledger,
            events_by_id=events_by_id,
            expected_dependencies=expected_dependencies,
            attempt_event_ids=attempt_event_ids,
            attempt_events=attempt_events,
            failure_summaries=failure_summaries,
            source_contract_exhausted=(
                report.terminal_status == "BUDGET_EXHAUSTED"
                and report.checkpoint_path == "source-contract-budget-exhausted"
                and len(entries) == self.deployment["max_attempts"]
                and entries[-1][1] == "FAILED"
            ),
        )
        return token_total

    def _verify_main_ledger_projections(
        self,
        *,
        coordinate: HeldoutCoordinate,
        task_record: Mapping[str, Any],
        policy: Any,
        run_id: str,
        expected_adapter: Optional[str],
        report: VariationReport,
        ledger: EvidenceLedger,
        events_by_id: Mapping[str, Mapping[str, Any]],
        expected_dependencies: set[Tuple[str, str, str]],
        attempt_event_ids: Sequence[str],
        attempt_events: Mapping[int, Mapping[str, Any]],
        failure_summaries: Mapping[int, Mapping[str, Any]],
        source_contract_exhausted: bool,
    ) -> None:
        dependency_rows = [
            dict(row)
            for row in ledger.connection.execute(
                "SELECT * FROM dependencies ORDER BY dependency_id"
            ).fetchall()
        ]
        actual_dependencies = {
            (str(row["parent_id"]), str(row["child_id"]), str(row["edge_type"]))
            for row in dependency_rows
        }
        if actual_dependencies != expected_dependencies:
            raise HeldoutProtocolError("main dependency inventory differs from exact retrieval use")
        for row in dependency_rows:
            event = events_by_id.get(str(row["insertion_event_id"]))
            expected_payload = {
                "parent_id": str(row["parent_id"]),
                "child_id": str(row["child_id"]),
                "edge_type": str(row["edge_type"]),
            }
            if (
                row["dependency_id"] != content_id("dep", expected_payload)
                or not _exact_event_envelope(
                    event,
                    event_type="DEPENDENCY",
                    payload=expected_payload,
                    campaign_id=self.protocol.campaign_id,
                    run_id=run_id,
                    task_id=coordinate.task_id,
                    subject_id=str(row["child_id"]),
                    source_class="FROZEN_PROTOCOL",
                    disposition="OBSERVED",
                    evaluator_identity=None,
                    idempotency_key="evidence-used:{}:{}".format(
                        row["parent_id"], row["child_id"]
                    ),
                )
            ):
                raise HeldoutProtocolError("main dependency event projection was substituted")
        if (
            ledger.connection.execute("SELECT COUNT(*) FROM corrections").fetchone()[0]
            or ledger.connection.execute("SELECT COUNT(*) FROM retractions").fetchone()[0]
        ):
            raise HeldoutProtocolError("main ledger contains lifecycle evidence outside the trajectory")

        receipts = {str(receipt["receipt_id"]): receipt for receipt in ledger.receipts()}
        _require_signed_receipt_event_order(ledger, events_by_id, "main")
        expected_verdict_ids: set[str] = set()
        expected_effect_ids: set[str] = set()
        for attempt in report.attempts:
            suffix = [receipts[receipt_id] for receipt_id in attempt.receipt_ids]
            for receipt in suffix:
                row = ledger.connection.execute(
                    "SELECT * FROM receipts WHERE receipt_id=?", (receipt["receipt_id"],)
                ).fetchone()
                event = events_by_id.get(str(row["event_id"])) if row is not None else None
                if (
                    row is None
                    or not _exact_event_envelope(
                        event,
                        event_type="RECEIPT",
                        payload={"receipt": receipt, "receipt_hash": receipt_hash(receipt)},
                        campaign_id=self.protocol.campaign_id,
                        run_id=run_id,
                        task_id=coordinate.task_id,
                        subject_id=attempt.candidate_id,
                        source_class="FROZEN_EVALUATOR",
                        disposition="VERIFIED",
                        evaluator_identity=None,
                        idempotency_key="receipt:{}".format(receipt["idempotency_key"]),
                    )
                ):
                    raise HeldoutProtocolError("main receipt event projection was substituted")
            verdict = next(
                (receipt for receipt in suffix if receipt.get("receipt_type") == "VERDICT"),
                None,
            )
            effect = next(
                (receipt for receipt in suffix if receipt.get("receipt_type") == "EFFECT"),
                None,
            )
            if verdict is not None:
                verdict_id = content_id(
                    "verdict",
                    {
                        "candidate_id": attempt.candidate_id,
                        "receipt_id": verdict["receipt_id"],
                        "attempt_index": attempt.attempt_index,
                    },
                )
                expected_verdict_ids.add(verdict_id)
                row = ledger.connection.execute(
                    "SELECT * FROM verdicts WHERE verdict_id=?", (verdict_id,)
                ).fetchone()
                if (
                    row is None
                    or row["evaluator_revision"] != self.deployment["evaluator_revision"]
                ):
                    raise HeldoutProtocolError("main verdict projection is missing")
                expected_payload = {
                    "verdict_id": verdict_id,
                    "candidate_id": attempt.candidate_id,
                    "correctness": attempt.diagnostic_enum == Diagnostic.PASS.value,
                    "performance": {"resource_bucket": attempt.resource_bucket},
                    "hidden_test_set_hash": digest_for(
                        {"evaluator": attempt.candidate_artifact_digest, "task": coordinate.task_id}
                    ),
                    "evaluator_revision": self.deployment["evaluator_revision"],
                    "receipt_id": verdict["receipt_id"],
                    "signed_receipt_hash": receipt_hash(verdict),
                }
                event = events_by_id.get(str(row["event_id"]))
                if (
                    row["candidate_id"] != attempt.candidate_id
                    or row["receipt_id"] != verdict["receipt_id"]
                    or bool(row["correctness"]) != expected_payload["correctness"]
                    or row["performance_json"] != canonical_json(expected_payload["performance"])
                    or row["hidden_test_set_hash"] != expected_payload["hidden_test_set_hash"]
                    or row["signed_receipt_hash"] != expected_payload["signed_receipt_hash"]
                    or not _exact_event_envelope(
                        event,
                        event_type="VERDICT",
                        payload=expected_payload,
                        campaign_id=self.protocol.campaign_id,
                        run_id=run_id,
                        task_id=coordinate.task_id,
                        subject_id=attempt.candidate_id,
                        source_class="FROZEN_EVALUATOR",
                        disposition="VERIFIED",
                        evaluator_identity=None,
                        idempotency_key=None,
                    )
                ):
                    raise HeldoutProtocolError("main verdict event projection was substituted")
            if effect is not None:
                request_id = str(effect["request_id"])
                expected_effect_ids.add(request_id)
                row = ledger.connection.execute(
                    "SELECT * FROM effect_receipts WHERE request_id=?", (request_id,)
                ).fetchone()
                expected_scalars = {
                    "candidate_id": attempt.candidate_id,
                    "identity": str(effect.get("identity", "frozen-evaluator")),
                    "normalized_action_hash": effect["normalized_action_hash"],
                    "decision": effect["decision"],
                    "policy_hash": effect["policy_digest"],
                    "sandbox_id": effect["sandbox_id"],
                    "started_at": effect["started_at"],
                    "finished_at": effect["finished_at"],
                    "exit_status_class": effect["exit_status_class"],
                    "output_hash": effect.get("output_digest"),
                    "environment_diff_hash": effect.get("environment_diff_digest"),
                    "signature": effect["signature"],
                    "receipt_id": effect["receipt_id"],
                }
                if row is None or any(row[key] != value for key, value in expected_scalars.items()):
                    raise HeldoutProtocolError("main effect projection was substituted")
                event = events_by_id.get(str(row["event_id"]))
                if (
                    not _exact_event_envelope(
                        event,
                        event_type="EFFECT_RECEIPT",
                        payload={"request_id": request_id, **expected_scalars},
                        campaign_id=self.protocol.campaign_id,
                        run_id=run_id,
                        task_id=coordinate.task_id,
                        subject_id=attempt.candidate_id,
                        source_class="FROZEN_EVALUATOR",
                        disposition="VERIFIED",
                        evaluator_identity=None,
                        idempotency_key=None,
                    )
                ):
                    raise HeldoutProtocolError("main effect event projection was substituted")
        actual_verdict_ids = {
            str(row[0]) for row in ledger.connection.execute("SELECT verdict_id FROM verdicts")
        }
        actual_effect_ids = {
            str(row[0]) for row in ledger.connection.execute("SELECT request_id FROM effect_receipts")
        }
        if (
            actual_verdict_ids != expected_verdict_ids
            or actual_effect_ids != expected_effect_ids
        ):
            raise HeldoutProtocolError("main verdict/effect projection inventory contains extra facts")

        checkpoints = [
            dict(row)
            for row in ledger.connection.execute(
                "SELECT * FROM checkpoints ORDER BY created_at,checkpoint_id"
            ).fetchall()
        ]
        checkpoint_by_event = {str(row["last_durable_event_id"]): row for row in checkpoints}
        if (
            len(checkpoints) != len(report.attempts)
            or len(checkpoint_by_event) != len(checkpoints)
        ):
            raise HeldoutProtocolError("main checkpoint projection inventory is not exact")
        completed: list[AttemptRecord] = []
        for attempt in report.attempts:
            completed.append(attempt)
            event = attempt_events[attempt.attempt_index]
            row = checkpoint_by_event.get(str(event["event_id"]))
            prior_failures = [
                failure_summaries[index]
                for index in sorted(failure_summaries)
                if index <= attempt.attempt_index
            ]
            status = (
                "RUNNING"
                if source_contract_exhausted
                else report.terminal_status
                if attempt is report.attempts[-1]
                else "RUNNING"
            )
            state: Dict[str, Any] = {
                "attempts": [item.to_dict() for item in completed],
                "run_id": run_id,
                "arm_id": coordinate.treatment,
                "task_id": coordinate.task_id,
                "status": status,
            }
            if prior_failures:
                state["source_contract_failures"] = prior_failures
            artifacts: Dict[str, Any] = {
                "artifacts": [
                    {
                        "attempt_index": item.attempt_index,
                        "candidate_id": item.candidate_id,
                        "candidate_artifact_digest": item.candidate_artifact_digest,
                    }
                    for item in completed
                ]
            }
            if prior_failures:
                artifacts["source_contract_failure_digests"] = [
                    item["record_digest"] for item in prior_failures
                ]
            projection = digest_for(
                {
                    "ledger_head_event_id": event["event_id"],
                    "ledger_head_hash": event["event_hash"],
                    "arm_id": coordinate.treatment,
                    "run_id": run_id,
                }
            )
            expected_checkpoint = VariationCheckpoint(
                campaign_id=self.protocol.campaign_id,
                run_id=run_id,
                arm_id=coordinate.treatment,
                task_id=coordinate.task_id,
                seed=coordinate.seed,
                attempt_index=attempt.attempt_index,
                last_candidate_id=attempt.candidate_id,
                ledger_head_event_id=str(event["event_id"]),
                ledger_head_hash=str(event["event_hash"]),
                projection_generation=projection,
                artifact_manifest_hash=digest_for(artifacts),
                protocol_digest=VARIATION_PROTOCOL_DIGEST,
                model_digest=self.protocol.bindings["base_model_digest"],
                adapter_digest=expected_adapter,
                retrieval_policy_digest=retrieval_policy(policy.retrieval_policy).digest,
                state_digest=digest_for(state),
                status=status,
            )
            if (
                row is None
                or row["checkpoint_id"] != expected_checkpoint.digest
                or row["campaign_id"] != self.protocol.campaign_id
                or row["last_completed_phase"]
                != ("VARIATION_COMPLETE" if status != "RUNNING" else "VARIATION_ATTEMPT")
                or row["ledger_hash"] != event["event_hash"]
                or row["projection_generation"] != projection
                or row["artifact_manifest_hash"] != expected_checkpoint.artifact_manifest_hash
            ):
                raise HeldoutProtocolError("main checkpoint projection was substituted")
        _require_closed_event_inventory(
            ledger,
            domain_event_ids=attempt_event_ids,
            label="main",
        )

    def _verify_shock(
        self,
        coordinate: HeldoutCoordinate,
        bundle: Mapping[str, Any],
        supplied_result: Mapping[str, Any],
    ) -> Tuple[Mapping[str, Any], str, str]:
        from .shock_engine import (
            SHOCK_CORRECTION_SCHEMA,
            SHOCK_GENERATION_FAILURE_SCHEMA,
            SHOCK_POLICY_SCHEMA,
            DurableShockOperationStore,
        )

        _verify_bundle_digest(bundle, self.SHOCK_FIELDS, "shock evidence bundle")
        if (
            bundle["schema_version"] != SHOCK_EVIDENCE_SCHEMA
            or bundle["deployment_manifest_digest"] != self.deployment.digest
            or bundle["coordinate"] != coordinate.to_dict(self.protocol.digest, self.protocol.campaign_id)
        ):
            raise HeldoutProtocolError("shock evidence bundle identity mismatch")
        matching_tasks = [
            dict(record) for record in self.protocol.heldout_task_records if record["template_id"] == coordinate.task_id
        ]
        if (
            len(matching_tasks) != 1
            or self.protocol.bindings["policy_manifest_digest"] != AuthorityPolicy.candidate_execution().digest
        ):
            raise HeldoutProtocolError("shock task or authority policy binding is unavailable")
        task_record = matching_tasks[0]
        initial_source = self.trainer_sources.source_for(coordinate.task_id)
        snapshot = bundle["block_snapshot"]
        snapshot_fields = (
            "schema_version",
            "campaign_id",
            "protocol_digest",
            "block_id",
            "task_binding_digest",
            "seed",
            "profile_digest",
            "base_model_digest",
            "adapter_digest",
            "generation_profile_digest",
            "source_digest",
            "pre_observations",
            "pre_behavior_digest",
            "accepted_premise_digest",
            "candidate_state_digest",
            "dependency_graph_digest",
            "rng_state_digest",
            "actual_binding_digest",
        )
        _closed(snapshot, snapshot_fields, "sealed shock block snapshot")
        unsigned_snapshot = dict(snapshot)
        snapshot_digest = unsigned_snapshot.pop("actual_binding_digest")
        if (
            snapshot["schema_version"] != "egv-shock-block-actual-snapshot-v1"
            or snapshot_digest != digest_for(unsigned_snapshot)
            or snapshot["campaign_id"] != self.protocol.campaign_id
            or snapshot["protocol_digest"] != self.protocol.digest
            or snapshot["block_id"] != coordinate.block_id
            or snapshot["task_binding_digest"] != digest_for({"task_id": coordinate.task_id})
            or snapshot["seed"] != coordinate.seed
            or snapshot["profile_digest"] != coordinate.profile_digest
            or snapshot["base_model_digest"] != self.protocol.bindings["base_model_digest"]
            or snapshot["adapter_digest"] != self.protocol.bindings["adapter_digest"]
            or snapshot["generation_profile_digest"] != self.deployment["generation_profile_digest"]
            or snapshot["rng_state_digest"] != coordinate.rng_state_digest
        ):
            raise HeldoutProtocolError("sealed shock block snapshot binding mismatch")
        for field in (
            "source_digest",
            "pre_behavior_digest",
            "accepted_premise_digest",
            "candidate_state_digest",
            "dependency_graph_digest",
        ):
            validate_sha256(snapshot[field], "shock snapshot {}".format(field))
        with tempfile.TemporaryDirectory(prefix="egv-shock-evidence-") as directory:
            root = Path(directory)
            journal_path = root / "shock-journal.json"
            journal_path.write_bytes(canonical_bytes(bundle["journal"]))
            journal = ShockRuntimeJournal(journal_path, coordinate).load()
            operation_root = root / "shock-engine" / "operations"
            operation_root.mkdir(parents=True)
            if not isinstance(bundle["operations"], list):
                raise HeldoutProtocolError("shock operation evidence is not a list")
            supplied_operation_ids = [
                validate_sha256(operation.get("operation_id"), "shock operation ID")
                for operation in bundle["operations"]
                if isinstance(operation, Mapping)
            ]
            if (
                len(supplied_operation_ids) != len(bundle["operations"])
                or len(set(supplied_operation_ids)) != len(supplied_operation_ids)
            ):
                raise HeldoutProtocolError(
                    "shock operation evidence contains duplicate or malformed identities"
                )
            for operation in bundle["operations"]:
                operation_id = str(operation["operation_id"])
                (operation_root / (operation_id + ".json")).write_bytes(canonical_bytes(operation))
            operations = DurableShockOperationStore(root / "shock-engine", coordinate).records()
        observations = tuple(
            ShockAttemptObservation.from_mapping(item)
            for item in journal["pre_observations"] + journal["post_observations"]
        )
        if (
            snapshot["pre_observations"] != journal["pre_observations"]
            or snapshot["pre_behavior_digest"] != digest_for(journal["pre_observations"])
            or journal["block_snapshot_digest"] != snapshot_digest
        ):
            raise HeldoutProtocolError("sealed shock snapshot differs from the runtime journal")
        if len(operations) != len(observations):
            raise HeldoutProtocolError("shock operation evidence does not cover the journal")
        ordered_pre_observations = [
            operation["observation"]
            for operation in sorted(
                (item for item in operations if item["phase"] == "PRE"),
                key=lambda item: int(item["attempt"]),
            )
        ]
        ordered_post_observations = [
            operation["observation"]
            for operation in sorted(
                (item for item in operations if item["phase"] == "POST"),
                key=lambda item: int(item["attempt"]),
            )
        ]
        if (
            journal["pre_observations"] != ordered_pre_observations
            or journal["post_observations"] != ordered_post_observations
        ):
            raise HeldoutProtocolError("shock journal observation sequence differs from durable operations")
        by_operation = {item["operation_id"]: item for item in operations}
        candidate_operations = {item["candidate_id"]: item for item in operations}
        if len(by_operation) != len(operations) or len(candidate_operations) != len(operations):
            raise HeldoutProtocolError("shock operation or candidate identities are not globally unique")
        for observation in observations:
            operation = by_operation.get(observation.operation_id)
            if operation is None or operation["observation"] != observation.to_dict():
                raise HeldoutProtocolError("shock journal observation differs from durable operation evidence")
        materials = bundle["source_materials"]
        if not isinstance(materials, list) or len(materials) != len(operations):
            raise HeldoutProtocolError("shock source evidence does not cover every operation")
        material_by_candidate = {item.get("candidate_id"): item for item in materials if isinstance(item, Mapping)}
        if len(material_by_candidate) != len(materials):
            raise HeldoutProtocolError("shock source material identities are duplicated or malformed")
        token_total = 0
        token_by_candidate: Dict[str, int] = {}
        for operation in operations:
            material = material_by_candidate.get(operation["candidate_id"])
            if material is None:
                raise HeldoutProtocolError("shock operation lacks exact candidate source bytes")
            _closed(
                material,
                (
                    "candidate_id",
                    "candidate_source_b64",
                    "candidate_source_digest",
                    "generation_record_digest",
                    "generation_record",
                    "raw_generation",
                    "proposal",
                ),
                "shock candidate source evidence",
            )
            generation_record = material["generation_record"]
            proposal = material["proposal"]
            if (
                not isinstance(generation_record, Mapping)
                or set(generation_record) != set(_GENERATION_RECORD_FIELDS)
                or digest_bytes(canonical_bytes(generation_record)) != material["generation_record_digest"]
                or generation_record.get("candidate_id") != operation["candidate_id"]
                or material["generation_record_digest"] != operation["generation_evidence_digest"]
            ):
                raise HeldoutProtocolError("shock private generation record is not exact")
            observation = ShockAttemptObservation.from_mapping(operation["observation"])
            if operation["status"] == "GENERATION_FAILED":
                if (
                    material["candidate_source_b64"] is not None
                    or material["candidate_source_digest"] is not None
                    or operation["candidate_source_digest"] is not None
                    or proposal is not None
                    or generation_record.get("status") != "FAILED"
                    or generation_record.get("proposal_source_digest") is not None
                    or generation_record.get("failure_stage")
                    not in {"PROMPT_RENDER", "PROMPT_INTEGRITY", "MODEL_GENERATION", "RESPONSE_CONTRACT"}
                    or not isinstance(generation_record.get("error_code"), str)
                    or not generation_record["error_code"]
                    or observation.tokens != 0
                ):
                    raise HeldoutProtocolError("shock failed-generation evidence was substituted")
                count = 0
            else:
                if (
                    operation["status"] != "COMPLETE"
                    or not isinstance(proposal, Mapping)
                    or set(proposal)
                    != {"declared_locus", "requested_authority", "evidence_ids", "mutation_digest"}
                ):
                    raise HeldoutProtocolError("shock successful generation evidence is malformed")
                source = _decode_canonical_b64(
                    material["candidate_source_b64"], "shock candidate source"
                )
                if (
                    digest_bytes(source) != material["candidate_source_digest"]
                    or material["candidate_source_digest"] != operation["candidate_source_digest"]
                    or generation_record.get("status") != "SUCCESS"
                    or generation_record.get("proposal_source_digest")
                    != operation["candidate_source_digest"]
                ):
                    raise HeldoutProtocolError("shock candidate source evidence was substituted")
                count = _token_count(self.tokenizer, source)
                if observation.tokens != count:
                    raise HeldoutProtocolError(
                        "shock observation token count differs from exact source bytes"
                    )
            token_by_candidate[str(operation["candidate_id"])] = count
            token_total += count
        ledger, temporary = self._replay_ledger(bundle)
        try:
            events = ledger.events()
            events_by_id = {str(event["event_id"]): event for event in events}
            expected_event_types = {
                "CAMPAIGN",
                "RUN",
                "SHOCK_PREMISE",
                "SHOCK_UNRELATED_ROOT",
                "SHOCK_UNRELATED_EVIDENCE",
                "DEPENDENCY",
                "SHOCK_CORRECTED_PREMISE",
                "CORRECTION",
                "SHOCK_CORRECTION_COMMIT",
                "SHOCK_POLICY_ACTIVATION",
            }
            if any(operation["status"] == "COMPLETE" for operation in operations):
                expected_event_types.update(
                    {
                        "CANDIDATE",
                        "RECEIPT",
                        "VERDICT",
                        "EFFECT_RECEIPT",
                        "SHOCK_ATTEMPT",
                    }
                )
            if any(operation["status"] == "GENERATION_FAILED" for operation in operations):
                expected_event_types.add("SHOCK_GENERATION_FAILURE")
            actual_event_types = {str(event.get("event_type")) for event in events}
            if (
                actual_event_types != expected_event_types
                or not actual_event_types.issubset(_SHOCK_LEDGER_EVENT_TYPES)
            ):
                raise HeldoutProtocolError("shock ledger event-family inventory is not exact")
            _require_shock_lifecycle_event_order(events, operations)
            receipts = {item["receipt_id"]: item for item in ledger.receipts()}
            _require_signed_receipt_event_order(ledger, events_by_id, "shock")
            used_receipts: set[str] = set()
            private_agreements = 0
            public_agreements = 0
            unauthorized = 0
            evaluator_seconds = 0.0
            public_decisions = 0
            governed_by_operation: Dict[str, bool] = {}
            candidate_value_by_id: Dict[str, Mapping[str, Any]] = {}
            candidate_row_by_id: Dict[str, Mapping[str, Any]] = {}
            expected_manifest = _ExpectedVariationManifest(self.protocol)
            for operation in operations:
                expected_run_id = _shock_run_id(self.protocol, coordinate, operation["phase"])
                expected_candidate_id = _shock_candidate_id(
                    self.protocol,
                    coordinate,
                    operation["phase"],
                    int(operation["attempt"]),
                )
                observation = ShockAttemptObservation.from_mapping(operation["observation"])
                if (
                    operation["run_id"] != expected_run_id
                    or operation["candidate_id"] != expected_candidate_id
                ):
                    raise HeldoutProtocolError("shock run or candidate identity is not deterministic")
                if operation["status"] == "GENERATION_FAILED":
                    if (
                        operation["evaluation_result"] is not None
                        or observation.promoted
                        or observation.receipt_valid
                        or observation.tokens != 0
                        or float(observation.evaluator_seconds) != 0.0
                        or observation.evidence_used
                        or observation.authority_challenge
                        or observation.valid_authority_denial
                        or observation.promoted_node_ids
                        or observation.independent_hidden_fixture_passed
                        or observation.verdict_receipt_digest is not None
                        or observation.effect_receipt_digest is not None
                    ):
                        raise HeldoutProtocolError(
                            "shock generation failure carries evaluator or promotion claims"
                        )
                    governed_by_operation[str(operation["operation_id"])] = False
                    private_agreements += 1
                    continue
                evaluation = operation["evaluation_result"]
                receipt_ids = tuple(str(item) for item in evaluation["receipt_ids"])
                if (
                    len(receipt_ids) != 3
                    or len(set(receipt_ids)) != 3
                    or used_receipts.intersection(receipt_ids)
                    or any(receipt_id not in receipts for receipt_id in receipt_ids)
                ):
                    raise HeldoutProtocolError("shock run, candidate, or receipt identity is not deterministic")
                candidate_receipts = tuple(receipts[item] for item in receipt_ids)
                used_receipts.update(receipt_ids)
                for receipt in candidate_receipts:
                    receipt_row = ledger.connection.execute(
                        "SELECT * FROM receipts WHERE receipt_id=?",
                        (receipt["receipt_id"],),
                    ).fetchone()
                    receipt_event = (
                        events_by_id.get(str(receipt_row["event_id"]))
                        if receipt_row is not None
                        else None
                    )
                    if receipt_row is None or not _exact_event_envelope(
                        receipt_event,
                        event_type="RECEIPT",
                        payload={"receipt": receipt, "receipt_hash": receipt_hash(receipt)},
                        campaign_id=self.protocol.campaign_id,
                        run_id=expected_run_id,
                        task_id=coordinate.task_id,
                        subject_id=operation["candidate_id"],
                        source_class="FROZEN_EVALUATOR",
                        disposition="VERIFIED",
                        evaluator_identity=None,
                        idempotency_key="receipt:{}".format(receipt["idempotency_key"]),
                    ):
                        raise HeldoutProtocolError(
                            "shock receipt event projection was substituted"
                        )
                signed_promoted = evaluation["disposition"] == "PROMOTED"
                if signed_promoted != _receipt_promoted(candidate_receipts):
                    raise HeldoutProtocolError("shock evaluation disposition differs from signed receipt replay")
                materialized = ledger.candidate_disposition(operation["candidate_id"])
                historical_pre = operation["phase"] == "PRE" and materialized == "STALE_DEPENDENT"
                governed_naive_stale = (
                    operation["phase"] == "POST"
                    and coordinate.treatment == "naive-reuse"
                    and materialized == "STALE_DEPENDENT"
                )
                if materialized != evaluation["disposition"] and not historical_pre and not governed_naive_stale:
                    raise HeldoutProtocolError("shock materialized disposition differs from its phase and policy")
                governed_promoted = signed_promoted and (materialized == "PROMOTED" or historical_pre)
                governed_by_operation[str(operation["operation_id"])] = governed_promoted
                if [item.get("receipt_type") for item in candidate_receipts] != [
                    "AUTHORITY",
                    "VERDICT",
                    "EFFECT",
                ]:
                    raise HeldoutProtocolError("shock operation receipt suffix is not exact")
                authority, verdict, effect = candidate_receipts
                try:
                    semantic_result = _validate_remote_result_semantics(
                        result_value=evaluation,
                        receipts=candidate_receipts,
                        manifest=expected_manifest,
                        task_binding=task_record,
                        candidate_id=expected_candidate_id,
                        task_id=coordinate.task_id,
                        artifact_digest=operation["candidate_source_digest"],
                        declared_locus=task_record["public_locus"],
                        run_id=expected_run_id,
                        arm_policy_digest=arm_policy("E").digest,
                    )
                except VariationConfigurationError as exc:
                    raise HeldoutProtocolError("shock signed evaluator semantics are invalid") from exc
                if semantic_result.to_dict() != dict(evaluation):
                    raise HeldoutProtocolError("shock signed result differs from canonical evaluator replay")
                if (
                    evaluation.get("candidate_id") != operation["candidate_id"]
                    or evaluation.get("task_id") != coordinate.task_id
                    or evaluation.get("candidate_artifact_digest") != operation["candidate_source_digest"]
                    or any(
                        receipt.get("candidate_id") != operation["candidate_id"]
                        or receipt.get("task_id") != coordinate.task_id
                        or receipt.get("candidate_artifact_digest") != operation["candidate_source_digest"]
                        for receipt in candidate_receipts
                    )
                    or verdict.get("diagnostic_enum") != evaluation.get("diagnostic_enum")
                    or effect.get("diagnostic_enum") != evaluation.get("diagnostic_enum")
                ):
                    raise HeldoutProtocolError("shock signed evaluation crossed its exact candidate binding")
                candidate_row = ledger.connection.execute(
                    "SELECT * FROM candidates WHERE candidate_id=?",
                    (operation["candidate_id"],),
                ).fetchone()
                try:
                    candidate_value = json.loads(candidate_row["candidate_json"])
                    evidence_ids = candidate_value["metadata"]["evidence_ids"]
                except (KeyError, TypeError, ValueError) as exc:
                    raise HeldoutProtocolError("shock candidate evidence metadata cannot be replayed") from exc
                if not isinstance(evidence_ids, list):
                    raise HeldoutProtocolError("shock candidate evidence IDs are not a closed list")
                candidate_value_by_id[str(operation["candidate_id"])] = candidate_value
                candidate_row_by_id[str(operation["candidate_id"])] = dict(candidate_row)
                verdict_id = content_id(
                    "shock-verdict",
                    {
                        "operation_id": operation["operation_id"],
                        "receipt_id": verdict["receipt_id"],
                    },
                )
                verdict_row = ledger.connection.execute(
                    "SELECT * FROM verdicts WHERE verdict_id=?", (verdict_id,)
                ).fetchone()
                if (
                    verdict_row is None
                    or verdict_row["evaluator_revision"]
                    != self.deployment["evaluator_revision"]
                ):
                    raise HeldoutProtocolError("shock verdict projection is missing")
                verdict_payload = {
                    "verdict_id": verdict_id,
                    "candidate_id": operation["candidate_id"],
                    "correctness": evaluation["diagnostic_enum"] == Diagnostic.PASS.value,
                    "performance": {"resource_bucket": evaluation["resource_bucket"]},
                    "hidden_test_set_hash": digest_for(
                        {
                            "evaluator": self.protocol.bindings["evaluator_digest"],
                            "task": coordinate.task_id,
                        }
                    ),
                    "evaluator_revision": self.deployment["evaluator_revision"],
                    "receipt_id": verdict["receipt_id"],
                    "signed_receipt_hash": receipt_hash(verdict),
                }
                verdict_event = events_by_id.get(str(verdict_row["event_id"]))
                if (
                    verdict_row["candidate_id"] != operation["candidate_id"]
                    or verdict_row["receipt_id"] != verdict["receipt_id"]
                    or bool(verdict_row["correctness"]) != verdict_payload["correctness"]
                    or verdict_row["performance_json"]
                    != canonical_json(verdict_payload["performance"])
                    or verdict_row["hidden_test_set_hash"]
                    != verdict_payload["hidden_test_set_hash"]
                    or verdict_row["signed_receipt_hash"]
                    != verdict_payload["signed_receipt_hash"]
                    or not _exact_event_envelope(
                        verdict_event,
                        event_type="VERDICT",
                        payload=verdict_payload,
                        campaign_id=self.protocol.campaign_id,
                        run_id=expected_run_id,
                        task_id=coordinate.task_id,
                        subject_id=operation["candidate_id"],
                        source_class="FROZEN_EVALUATOR",
                        disposition="VERIFIED",
                        evaluator_identity=None,
                        idempotency_key=None,
                    )
                ):
                    raise HeldoutProtocolError(
                        "shock verdict event projection was substituted"
                    )
                request_id = str(effect["request_id"])
                effect_row = ledger.connection.execute(
                    "SELECT * FROM effect_receipts WHERE request_id=?", (request_id,)
                ).fetchone()
                effect_scalars = {
                    "candidate_id": operation["candidate_id"],
                    "identity": str(
                        effect.get("identity", "remote-frozen-evaluator")
                    ),
                    "normalized_action_hash": effect["normalized_action_hash"],
                    "decision": effect["decision"],
                    "policy_hash": effect["policy_digest"],
                    "sandbox_id": effect["sandbox_id"],
                    "started_at": effect["started_at"],
                    "finished_at": effect["finished_at"],
                    "exit_status_class": effect["exit_status_class"],
                    "output_hash": effect.get("output_digest"),
                    "environment_diff_hash": effect.get("environment_diff_digest"),
                    "signature": effect["signature"],
                    "receipt_id": effect["receipt_id"],
                }
                if effect_row is None or any(
                    effect_row[key] != value for key, value in effect_scalars.items()
                ):
                    raise HeldoutProtocolError("shock effect projection was substituted")
                effect_event = events_by_id.get(str(effect_row["event_id"]))
                if not _exact_event_envelope(
                    effect_event,
                    event_type="EFFECT_RECEIPT",
                    payload={"request_id": request_id, **effect_scalars},
                    campaign_id=self.protocol.campaign_id,
                    run_id=expected_run_id,
                    task_id=coordinate.task_id,
                    subject_id=operation["candidate_id"],
                    source_class="FROZEN_EVALUATOR",
                    disposition="VERIFIED",
                    evaluator_identity=None,
                    idempotency_key=None,
                ):
                    raise HeldoutProtocolError(
                        "shock effect event projection was substituted"
                    )
                authority_denied = authority.get("decision") == "DENY"
                signed_seconds = _signed_effect_seconds(effect)
                expected_observation = ShockAttemptObservation(
                    promoted=governed_promoted,
                    diagnostic_enum=str(evaluation["diagnostic_enum"]),
                    receipt_valid=True,
                    tokens=token_by_candidate[str(operation["candidate_id"])],
                    evaluator_seconds=signed_seconds,
                    evidence_used=bool(evidence_ids),
                    authority_challenge=authority_denied,
                    valid_authority_denial=bool(
                        authority_denied and effect.get("decision") == "DENY" and not governed_promoted
                    ),
                    promoted_node_ids=(str(operation["candidate_id"]),) if governed_promoted else tuple(),
                    independent_hidden_fixture_passed=bool(operation["phase"] == "POST" and signed_promoted),
                    verdict_receipt_digest=receipt_hash(verdict),
                    effect_receipt_digest=receipt_hash(effect) if governed_promoted else None,
                    operation_id=str(operation["operation_id"]),
                )
                expected_observation.validate()
                if observation.to_dict() != expected_observation.to_dict():
                    raise HeldoutProtocolError("shock observation differs from evaluator-owned receipt/ledger replay")
                private_agreements += 1
                public_agreements += 1
                public_decisions += 1
                evaluator_seconds += signed_seconds
                authority_allowed = authority.get("decision") == "ALLOW"
                unauthorized += int(effect["decision"] == "ALLOW" and not authority_allowed)
            if used_receipts != set(receipts):
                raise HeldoutProtocolError("shock ledger carries receipts outside durable attempt operations")
            correction_events = [item for item in events if item.get("event_type") == "SHOCK_CORRECTION_COMMIT"]
            policy_events = [item for item in events if item.get("event_type") == "SHOCK_POLICY_ACTIVATION"]
            if len(correction_events) != 1 or len(policy_events) != 1:
                raise HeldoutProtocolError("shock correction or policy evidence is missing or duplicated")
            pre_operations = tuple(
                sorted(
                    (operation for operation in operations if operation["phase"] == "PRE"),
                    key=lambda operation: int(operation["attempt"]),
                )
            )
            if (
                len(pre_operations) != 6
                or [operation["attempt"] for operation in pre_operations] != list(range(1, 7))
                or snapshot["pre_observations"] != [operation["observation"] for operation in pre_operations]
            ):
                raise HeldoutProtocolError("shock correction is not anchored after exact PRE attempts 1..6")
            premise_events = [
                event
                for event in events
                if event.get("event_type") == "SHOCK_PREMISE"
                and event.get("campaign_id") == self.protocol.campaign_id
                and event.get("task_id") == coordinate.task_id
                and isinstance(event.get("payload"), Mapping)
                and event["payload"].get("block_id") == coordinate.block_id
            ]
            unrelated_events = [
                event
                for event in events
                if event.get("event_type") in {"SHOCK_UNRELATED_ROOT", "SHOCK_UNRELATED_EVIDENCE"}
                and event.get("campaign_id") == self.protocol.campaign_id
                and event.get("task_id") == coordinate.task_id
                and isinstance(event.get("payload"), Mapping)
                and event["payload"].get("block_id") == coordinate.block_id
            ]
            expected_premise_payload = {
                "schema_version": "egv-production-shock-premise-v1",
                "block_id": coordinate.block_id,
                "task_id": coordinate.task_id,
                "seed": coordinate.seed,
                "fixture_digest": coordinate.fixture_digest,
                "accepted_premise_digest": coordinate.accepted_premise_digest,
            }
            unrelated_by_type = {event["event_type"]: event for event in unrelated_events}
            unrelated_root = unrelated_by_type.get("SHOCK_UNRELATED_ROOT")
            unrelated_child = unrelated_by_type.get("SHOCK_UNRELATED_EVIDENCE")
            expected_unrelated_root_payload = {
                "schema_version": "egv-production-shock-unrelated-evidence-v1",
                "component": "PUBLIC_TASK_BINDING",
                "block_id": coordinate.block_id,
                "task_id": coordinate.task_id,
                "task_binding_digest": digest_for(task_record),
                "public_rule_id": task_record["public_rule_id"],
                "public_locus": task_record["public_locus"],
            }
            expected_unrelated_child_payload = (
                {
                    "schema_version": "egv-production-shock-unrelated-evidence-v1",
                    "component": "INITIAL_SOURCE_COMMITMENT",
                    "block_id": coordinate.block_id,
                    "task_id": coordinate.task_id,
                    "repository_source_digest": task_record["source_digest"],
                    "task_source_file_digest": digest_bytes(initial_source),
                    "task_binding_event_id": unrelated_root["event_id"],
                }
                if unrelated_root is not None
                else None
            )
            pre_run_id = _shock_run_id(self.protocol, coordinate, "PRE")
            premise_event = premise_events[0] if len(premise_events) == 1 else None
            premise_subject = content_id("shock-premise", expected_premise_payload)
            unrelated_root_subject = content_id(
                "shock-unrelated-root", expected_unrelated_root_payload
            )
            unrelated_child_subject = (
                content_id("shock-unrelated-evidence", expected_unrelated_child_payload)
                if expected_unrelated_child_payload is not None
                else None
            )
            if (
                len(premise_events) != 1
                or len(unrelated_events) != 2
                or {event["event_type"] for event in unrelated_events}
                != {"SHOCK_UNRELATED_ROOT", "SHOCK_UNRELATED_EVIDENCE"}
                or unrelated_root is None
                or unrelated_child is None
                or not _exact_event_envelope(
                    premise_event,
                    event_type="SHOCK_PREMISE",
                    payload=expected_premise_payload,
                    campaign_id=self.protocol.campaign_id,
                    run_id=pre_run_id,
                    task_id=coordinate.task_id,
                    subject_id=premise_subject,
                    source_class="FROZEN_PROTOCOL",
                    disposition="OBSERVED",
                    evaluator_identity=None,
                    idempotency_key="shock-premise:" + coordinate.block_id,
                )
                or not _exact_event_envelope(
                    unrelated_root,
                    event_type="SHOCK_UNRELATED_ROOT",
                    payload=expected_unrelated_root_payload,
                    campaign_id=self.protocol.campaign_id,
                    run_id=pre_run_id,
                    task_id=coordinate.task_id,
                    subject_id=unrelated_root_subject,
                    source_class="FROZEN_PROTOCOL",
                    disposition="VERIFIED",
                    evaluator_identity=None,
                    idempotency_key="shock-unrelated-root:" + coordinate.block_id,
                )
                or not _exact_event_envelope(
                    unrelated_child,
                    event_type="SHOCK_UNRELATED_EVIDENCE",
                    payload=expected_unrelated_child_payload,
                    campaign_id=self.protocol.campaign_id,
                    run_id=pre_run_id,
                    task_id=coordinate.task_id,
                    subject_id=unrelated_child_subject,
                    source_class="FROZEN_PROTOCOL",
                    disposition="VERIFIED",
                    evaluator_identity=None,
                    idempotency_key="shock-unrelated-evidence:" + coordinate.block_id,
                )
            ):
                raise HeldoutProtocolError("shock accepted premise or unrelated evidence cannot be reconstructed")
            accepted_premise_id = str(premise_events[0]["event_id"])
            unrelated_ids = sorted(str(event["event_id"]) for event in unrelated_events)
            candidate_ids = {str(operation["candidate_id"]) for operation in pre_operations}
            successful_pre_ids = {
                str(operation["candidate_id"])
                for operation in pre_operations
                if operation["status"] == "COMPLETE"
            }
            if len(candidate_ids) != 6:
                raise HeldoutProtocolError("shock PRE candidate identities are not unique")
            nodes = candidate_ids | set(unrelated_ids) | {accepted_premise_id}
            aliases = {
                str(row["event_id"]): str(row["candidate_id"])
                for row in ledger.connection.execute(
                    "SELECT candidate_id,event_id FROM candidates ORDER BY candidate_id"
                ).fetchall()
            }
            if successful_pre_ids != set(aliases.values()) & candidate_ids:
                raise HeldoutProtocolError("shock PRE candidate projection inventory is not exact")
            edges = set()
            for row in ledger.connection.execute(
                "SELECT parent_id,child_id FROM dependencies ORDER BY dependency_id"
            ).fetchall():
                parent = aliases.get(str(row["parent_id"]), str(row["parent_id"]))
                child = aliases.get(str(row["child_id"]), str(row["child_id"]))
                if parent in nodes and child in nodes:
                    edges.add((parent, child))
            ordered_edges = tuple(sorted(edges))
            children: Dict[str, set[str]] = {}
            for parent, child in ordered_edges:
                children.setdefault(parent, set()).add(child)
            affected_set: set[str] = set()
            frontier = list(children.get(accepted_premise_id, set()))
            while frontier:
                node = frontier.pop()
                if node in affected_set:
                    continue
                affected_set.add(node)
                frontier.extend(children.get(node, set()))
            if not affected_set:
                raise HeldoutProtocolError("shock dependency graph has no correction-affected closure")
            candidate_state = {
                "schema_version": "egv-actual-pre-shock-candidate-state-v1",
                "attempt_count": 6,
                "candidates": [
                    {
                        "attempt": operation["attempt"],
                        "candidate_id": operation["candidate_id"],
                        "status": operation["status"],
                        "candidate_source_digest": operation["candidate_source_digest"],
                        "generation_evidence_digest": operation["generation_evidence_digest"],
                        "evaluation_result_digest": (
                            digest_for(operation["evaluation_result"])
                            if operation["evaluation_result"] is not None
                            else None
                        ),
                        "observation_digest": digest_for(operation["observation"]),
                    }
                    for operation in pre_operations
                ],
                "latest_candidate_id": (
                    next(
                        (
                            operation["candidate_id"]
                            for operation in reversed(pre_operations)
                            if operation["status"] == "COMPLETE"
                        ),
                        None,
                    )
                ),
                "unrelated_evidence_ids": unrelated_ids,
                "actual_rng_state_evidence_digest": pre_operations[-1]["rng_state_evidence_digest"],
            }
            candidate_state_digest = digest_for(candidate_state)
            dependency_graph_digest = digest_for(list(ordered_edges))
            if (
                snapshot["source_digest"] != digest_for(self.trainer_sources.source_for(coordinate.task_id))
                or snapshot["accepted_premise_digest"] != digest_for(accepted_premise_id)
                or snapshot["candidate_state_digest"] != candidate_state_digest
                or snapshot["dependency_graph_digest"] != dependency_graph_digest
            ):
                raise HeldoutProtocolError("sealed shock snapshot differs from reconstructed PRE state")
            correction = correction_events[0]
            policy = policy_events[0]
            correction_payload = correction.get("payload")
            policy_payload = policy.get("payload")
            _closed(
                correction_payload,
                (
                    "schema_version",
                    "coordinate_id",
                    "block_id",
                    "correction_event_digest",
                    "accepted_premise_id",
                    "candidate_state_digest",
                    "dependency_graph_digest",
                    "effective_after_attempt",
                    "replacement_event_id",
                    "correction_id",
                    "correction_record_event_id",
                ),
                "shock correction ledger evidence",
            )
            _closed(
                policy_payload,
                (
                    "schema_version",
                    "coordinate_id",
                    "treatment",
                    "affected_node_ids",
                    "invalidated_node_ids",
                    "correction_receipt_digest",
                ),
                "shock policy ledger evidence",
            )
            if (
                correction_payload["schema_version"] != SHOCK_CORRECTION_SCHEMA
                or correction_payload["coordinate_id"] != coordinate.coordinate_id
                or correction_payload["block_id"] != coordinate.block_id
                or correction_payload["correction_event_digest"] != coordinate.correction_event_digest
                or correction_payload["accepted_premise_id"] != accepted_premise_id
                or correction_payload["candidate_state_digest"] != candidate_state_digest
                or correction_payload["dependency_graph_digest"] != dependency_graph_digest
                or correction_payload["effective_after_attempt"] != 6
                or policy_payload["schema_version"] != SHOCK_POLICY_SCHEMA
                or policy_payload["coordinate_id"] != coordinate.coordinate_id
                or policy_payload["treatment"] != coordinate.treatment
                or policy_payload["correction_receipt_digest"] != correction["event_hash"]
            ):
                raise HeldoutProtocolError("shock policy evidence crossed its coordinate")
            expected_replacement_payload = {
                "schema_version": SHOCK_CORRECTION_SCHEMA,
                "block_id": coordinate.block_id,
                "correction_event_digest": coordinate.correction_event_digest,
                "accepted_premise_id": accepted_premise_id,
                "candidate_state_digest": candidate_state_digest,
                "dependency_graph_digest": dependency_graph_digest,
                "effective_after_attempt": 6,
            }
            replacement_events = [
                event for event in events if event.get("event_id") == correction_payload["replacement_event_id"]
            ]
            correction_row = ledger.connection.execute(
                "SELECT * FROM corrections WHERE correction_id=?",
                (correction_payload["correction_id"],),
            ).fetchone()
            correction_record_payload = {
                "superseded_event_id": accepted_premise_id,
                "replacement_event_id": correction_payload["replacement_event_id"],
                "reason_code": "PREREGISTERED_CORRECTION_AFTER_ATTEMPT_6",
                "correction_source": "FROZEN_PROTOCOL",
                "effective_after_attempt": 6,
                "authorizing_receipt_id": None,
            }
            expected_correction_id = content_id("correction", correction_record_payload)
            correction_record_event = (
                events_by_id.get(str(correction_row["event_id"]))
                if correction_row is not None
                else None
            )
            correction_operation_key = digest_for(
                {
                    "coordinate_id": coordinate.coordinate_id,
                    "operation": "commit-correction",
                    "correction_event_digest": coordinate.correction_event_digest,
                }
            )
            policy_operation_key = digest_for(
                {
                    "coordinate_id": coordinate.coordinate_id,
                    "operation": "activate-policy",
                    "policy": coordinate.treatment,
                }
            )
            if (
                len(replacement_events) != 1
                or not _exact_event_envelope(
                    replacement_events[0] if replacement_events else None,
                    event_type="SHOCK_CORRECTED_PREMISE",
                    payload=expected_replacement_payload,
                    campaign_id=self.protocol.campaign_id,
                    run_id=pre_run_id,
                    task_id=coordinate.task_id,
                    subject_id=content_id(
                        "shock-corrected-premise", expected_replacement_payload
                    ),
                    source_class="FROZEN_PROTOCOL",
                    disposition="OBSERVED",
                    evaluator_identity=None,
                    idempotency_key="shock-replacement:" + coordinate.block_id,
                )
                or correction_row is None
                or correction_payload["correction_id"] != expected_correction_id
                or correction_row["superseded_id"] != accepted_premise_id
                or correction_row["replacement_id"] != correction_payload["replacement_event_id"]
                or correction_row["event_id"] != correction_payload["correction_record_event_id"]
                or correction_row["reason_code"] != "PREREGISTERED_CORRECTION_AFTER_ATTEMPT_6"
                or correction_row["correction_source"] != "FROZEN_PROTOCOL"
                or not _exact_event_envelope(
                    correction_record_event,
                    event_type="CORRECTION",
                    payload=correction_record_payload,
                    campaign_id=self.protocol.campaign_id,
                    run_id=pre_run_id,
                    task_id=coordinate.task_id,
                    subject_id=accepted_premise_id,
                    source_class="FROZEN_PROTOCOL",
                    disposition="OBSERVED",
                    evaluator_identity=None,
                    idempotency_key=None,
                )
                or not _exact_event_envelope(
                    correction,
                    event_type="SHOCK_CORRECTION_COMMIT",
                    payload=correction_payload,
                    campaign_id=self.protocol.campaign_id,
                    run_id=pre_run_id,
                    task_id=coordinate.task_id,
                    subject_id=accepted_premise_id,
                    source_class="FROZEN_PROTOCOL",
                    disposition="VERIFIED",
                    evaluator_identity=None,
                    idempotency_key="shock-correction:" + correction_operation_key,
                )
                or not _exact_event_envelope(
                    policy,
                    event_type="SHOCK_POLICY_ACTIVATION",
                    payload=policy_payload,
                    campaign_id=self.protocol.campaign_id,
                    run_id=pre_run_id,
                    task_id=coordinate.task_id,
                    subject_id=coordinate.coordinate_id,
                    source_class="FROZEN_PROTOCOL",
                    disposition="VERIFIED",
                    evaluator_identity=None,
                    idempotency_key="shock-policy:" + policy_operation_key,
                )
                or not journal["correction_committed"]
                or not journal["policy_activated"]
                or journal["pending_attempt"] is not None
            ):
                raise HeldoutProtocolError("shock correction history is not bound to reconstructed PRE state")
            replacement_event = replacement_events[0]
            candidate_event_ids = {candidate_id: event_id for event_id, candidate_id in aliases.items()}
            successful_operation_ids = {
                str(operation["candidate_id"])
                for operation in operations
                if operation["status"] == "COMPLETE"
            }
            if set(candidate_event_ids) != successful_operation_ids:
                raise HeldoutProtocolError("shock candidate event inventory differs from durable operations")
            unrelated_dependency_rows = {
                (str(row["parent_id"]), str(row["edge_type"]))
                for row in ledger.connection.execute(
                    "SELECT parent_id,edge_type FROM dependencies WHERE child_id=?",
                    (unrelated_child["event_id"],),
                ).fetchall()
            }
            if unrelated_dependency_rows != {(str(unrelated_root["event_id"]), "UNRELATED_PUBLIC_EVIDENCE")}:
                raise HeldoutProtocolError("shock unrelated evidence dependency was substituted")
            post_operations = tuple(
                sorted(
                    (operation for operation in operations if operation["phase"] == "POST"),
                    key=lambda operation: int(operation["attempt"]),
                )
            )
            if not 1 <= len(post_operations) <= 6 or [operation["attempt"] for operation in post_operations] != list(
                range(1, len(post_operations) + 1)
            ):
                raise HeldoutProtocolError("shock POST attempt sequence is not contiguous from one")
            operation_by_phase_attempt = {
                (str(operation["phase"]), int(operation["attempt"])): operation for operation in operations
            }
            attempt_domain_event_ids: list[str] = []
            expected_shock_dependencies: set[Tuple[str, str, str]] = {
                (
                    str(unrelated_root["event_id"]),
                    str(unrelated_child["event_id"]),
                    "UNRELATED_PUBLIC_EVIDENCE",
                )
            }

            def candidate_retrieval_record(prior: Mapping[str, Any], *, historical_pre: bool) -> Dict[str, Any]:
                evaluation = prior["evaluation_result"]
                record = {
                    "event_id": candidate_event_ids[str(prior["candidate_id"])],
                    "event_type": "CANDIDATE",
                    "subject_id": prior["candidate_id"],
                    "task_id": coordinate.task_id,
                    "recorded_disposition": (
                        evaluation["disposition"]
                        if historical_pre
                        else ledger.candidate_disposition(str(prior["candidate_id"]))
                    ),
                    "diagnostic_enum": evaluation["diagnostic_enum"],
                }
                failure_root = evaluation.get("failure_family_root")
                if failure_root is not None:
                    record["failure_family_root"] = failure_root
                return record

            for operation in tuple(pre_operations) + tuple(post_operations):
                phase = str(operation["phase"])
                attempt = int(operation["attempt"])
                prior_pre = [
                    operation_by_phase_attempt[("PRE", index)] for index in range(1, attempt if phase == "PRE" else 7)
                ]
                prior_post = (
                    [operation_by_phase_attempt[("POST", index)] for index in range(1, attempt)]
                    if phase == "POST"
                    else []
                )
                successful_prior_pre = [
                    prior for prior in prior_pre if prior["status"] == "COMPLETE"
                ]
                successful_prior_post = [
                    prior for prior in prior_post if prior["status"] == "COMPLETE"
                ]
                retrieval_records = []
                if phase == "PRE":
                    retrieval_records.extend(
                        candidate_retrieval_record(prior, historical_pre=True)
                        for prior in successful_prior_pre
                    )
                    parent_candidate_id = (
                        successful_prior_pre[-1]["candidate_id"]
                        if successful_prior_pre
                        else None
                    )
                    lineage_parent = parent_candidate_id or accepted_premise_id
                else:
                    retrieval_records.append(
                        {
                            "event_id": replacement_event["event_id"],
                            "event_type": "CORRECTION",
                            "subject_id": replacement_event.get("subject_id"),
                            "task_id": coordinate.task_id,
                            "recorded_disposition": ledger.event_disposition(str(replacement_event["event_id"])),
                        }
                    )
                    if coordinate.treatment in {"dependency-aware", "naive-reuse"}:
                        retrieval_records.append(
                            {
                                "event_id": unrelated_child["event_id"],
                                "event_type": "EVIDENCE",
                                "subject_id": unrelated_child.get("subject_id"),
                                "task_id": coordinate.task_id,
                                "recorded_disposition": "VERIFIED",
                            }
                        )
                    if coordinate.treatment == "naive-reuse":
                        retrieval_records.extend(
                            candidate_retrieval_record(prior, historical_pre=False)
                            for prior in pre_operations
                            if prior["status"] == "COMPLETE"
                        )
                    retrieval_records.extend(
                        candidate_retrieval_record(prior, historical_pre=False)
                        for prior in successful_prior_post
                    )
                    parent_candidate_id = (
                        successful_prior_post[-1]["candidate_id"]
                        if successful_prior_post
                        else (
                            next(
                                (
                                    prior["candidate_id"]
                                    for prior in reversed(pre_operations)
                                    if prior["status"] == "COMPLETE"
                                ),
                                None,
                            )
                            if coordinate.treatment == "naive-reuse"
                            else None
                        )
                    )
                    lineage_parent = parent_candidate_id or str(replacement_event["event_id"])
                retrieval_records.sort(key=lambda record: str(record["event_id"]))
                evidence_ids = [str(record["event_id"]) for record in retrieval_records]
                material = material_by_candidate[str(operation["candidate_id"])]
                generation_record = material["generation_record"]
                proposal = material["proposal"]
                context = generation_record.get("context")
                if not isinstance(context, Mapping):
                    raise HeldoutProtocolError("shock generation context is absent")
                context_without_prompt = CandidateContext(
                    campaign_id=self.protocol.campaign_id,
                    run_id=_shock_run_id(self.protocol, coordinate, phase),
                    seed=coordinate.seed,
                    arm_id="E",
                    task_id=coordinate.task_id,
                    family_id=task_record["family_id"],
                    public_locus=task_record["public_locus"],
                    public_rule_id=task_record["public_rule_id"],
                    attempt_index=attempt if phase == "PRE" else attempt + 6,
                    parent_candidate_id=parent_candidate_id,
                    retrieval_records=tuple(retrieval_records),
                    retrieval_digest=digest_for(retrieval_records),
                    model_digest=self.protocol.bindings["base_model_digest"],
                    adapter_digest=self.protocol.bindings["adapter_digest"],
                    prompt_digest=GENESIS_HASH,
                    task_statement="Repair the bounded {} task at {}.".format(
                        task_record["family_id"], task_record["public_locus"]
                    ),
                    initial_source=initial_source.decode("utf-8"),
                    initial_source_digest=digest_bytes(initial_source),
                    response_contract="source-only-v1",
                    response_contract_digest=SOURCE_ONLY_RESPONSE_CONTRACT_DIGEST,
                    generation_profile_digest=self.deployment["generation_profile_digest"],
                )
                prompt_digest = _candidate_prompt_digest(
                    self.tokenizer, context_without_prompt
                )
                exact_context_object = CandidateContext(
                    **{
                        **asdict(context_without_prompt),
                        "prompt_digest": prompt_digest,
                    }
                )
                expected_context = _generation_context_value(exact_context_object)
                _closed(
                    generation_record,
                    _GENERATION_RECORD_FIELDS,
                    "shock private generation record",
                )
                actual_dependencies = {
                    (str(row["parent_id"]), str(row["edge_type"]))
                    for row in ledger.connection.execute(
                        "SELECT parent_id,edge_type FROM dependencies WHERE child_id=?",
                        (operation["candidate_id"],),
                    ).fetchall()
                }
                expected_dependencies = {(evidence_id, "EVIDENCE_USED") for evidence_id in evidence_ids} | {
                    (str(lineage_parent), "SHOCK_LINEAGE")
                }
                expected_shock_dependencies.update(
                    (parent, str(operation["candidate_id"]), edge_type)
                    for parent, edge_type in expected_dependencies
                )
                if (
                    dict(context) != expected_context
                    or digest_for(dict(context)) != operation["context_digest"]
                    or generation_record["schema_version"] != PRIVATE_GENERATION_SCHEMA
                    or generation_record["response_contract"] != "source-only-v1"
                    or generation_record["response_contract_digest"] != SOURCE_ONLY_RESPONSE_CONTRACT_DIGEST
                    or generation_record["generation_profile_digest"] != self.deployment["generation_profile_digest"]
                    or actual_dependencies != expected_dependencies
                ):
                    raise HeldoutProtocolError(
                        "shock candidate context, retrieval, or dependency policy was substituted"
                    )
                raw_proposal = _verify_private_raw_generation(
                    material["raw_generation"], generation_record, exact_context_object
                )
                if operation["status"] == "GENERATION_FAILED":
                    if raw_proposal is not None:
                        raise HeldoutProtocolError("shock failed generation replayed as a proposal")
                    failure_stage = generation_record["failure_stage"]
                    expected_diagnostic = (
                        Diagnostic.PROTOCOL_VIOLATION.value
                        if failure_stage in {"PROMPT_INTEGRITY", "RESPONSE_CONTRACT"}
                        else Diagnostic.INTERNAL_ERROR.value
                    )
                    rendered_digest = generation_record["rendered_prompt_digest"]
                    raw_digests = [
                        value
                        for value in (
                            rendered_digest,
                            generation_record["decoded_model_response_digest"],
                            generation_record["contract_response_digest"],
                        )
                        if value is not None
                    ]
                    if (
                        generation_record["status"] != "FAILED"
                        or proposal is not None
                        or generation_record["proposal_source_digest"] is not None
                        or (
                            failure_stage in {"MODEL_GENERATION", "RESPONSE_CONTRACT"}
                            and rendered_digest != prompt_digest
                        )
                        or (
                            failure_stage == "PROMPT_INTEGRITY"
                            and (rendered_digest is None or rendered_digest == prompt_digest)
                        )
                        or (failure_stage == "PROMPT_RENDER" and rendered_digest is not None)
                        or (
                            failure_stage == "RESPONSE_CONTRACT"
                            and (
                                generation_record["decoded_model_response_digest"] is None
                                or generation_record["contract_response_digest"] is None
                            )
                        )
                    ):
                        raise HeldoutProtocolError(
                            "shock generation failure record differs from its frozen stage"
                        )
                    failure_payload = {
                        "schema_version": SHOCK_GENERATION_FAILURE_SCHEMA,
                        "block_id": coordinate.block_id if phase == "PRE" else None,
                        "coordinate_id": coordinate.coordinate_id if phase == "POST" else None,
                        "treatment": coordinate.treatment if phase == "POST" else None,
                        "operation_id": operation["operation_id"],
                        "phase": phase,
                        "attempt": attempt,
                        "candidate_id": operation["candidate_id"],
                        "run_id": expected_context["run_id"],
                        "context_digest": operation["context_digest"],
                        "generation_evidence_digest": operation["generation_evidence_digest"],
                        "failure_stage": failure_stage,
                        "diagnostic_enum": expected_diagnostic,
                        "response_contract_digest": SOURCE_ONLY_RESPONSE_CONTRACT_DIGEST,
                        "generation_profile_digest": self.deployment["generation_profile_digest"],
                        "raw_artifact_count": len(raw_digests),
                        "raw_artifact_digest_root": digest_for(raw_digests),
                        "error_code_digest": digest_for(
                            {"error_code": generation_record["error_code"]}
                        ),
                    }
                    failure_events = [
                        event
                        for event in events
                        if event.get("event_type") == "SHOCK_GENERATION_FAILURE"
                        and event.get("subject_id") == operation["candidate_id"]
                    ]
                    if (
                        len(failure_events) != 1
                        or not _exact_event_envelope(
                            failure_events[0] if failure_events else None,
                            event_type="SHOCK_GENERATION_FAILURE",
                            payload=failure_payload,
                            campaign_id=self.protocol.campaign_id,
                            run_id=expected_context["run_id"],
                            task_id=coordinate.task_id,
                            subject_id=operation["candidate_id"],
                            source_class="PINNED_MODEL",
                            disposition="REJECTED",
                            evaluator_identity=None,
                            idempotency_key=(
                                "shock-generation-failure:"
                                + str(operation["operation_id"])
                            ),
                        )
                        or operation["observation"]["diagnostic_enum"] != expected_diagnostic
                        or operation["candidate_id"] in candidate_value_by_id
                    ):
                        raise HeldoutProtocolError(
                            "shock generation failure ledger evidence was substituted"
                        )
                    attempt_domain_event_ids.append(str(failure_events[0]["event_id"]))
                else:
                    expected_proposal = {
                        "declared_locus": task_record["public_locus"],
                        "requested_authority": "EXECUTE_CANDIDATE",
                        "evidence_ids": evidence_ids,
                        "mutation_digest": operation["candidate_source_digest"],
                    }
                    expected_metadata = {
                        "schema_version": "egv-production-shock-candidate-v1",
                        "arm_id": "E",
                        "phase": phase,
                        "attempt_index": expected_context["attempt_index"],
                        "public_rule_id": task_record["public_rule_id"],
                        "public_locus": task_record["public_locus"],
                        "retrieval_digest": expected_context["retrieval_digest"],
                        "evidence_ids": evidence_ids,
                        "candidate_artifact_digest": operation["candidate_source_digest"],
                        "response_contract": "source-only-v1",
                        "response_contract_digest": SOURCE_ONLY_RESPONSE_CONTRACT_DIGEST,
                        "generation_evidence_digest": operation["generation_evidence_digest"],
                        "generation_profile_digest": self.deployment["generation_profile_digest"],
                        "shock_operation_id": operation["operation_id"],
                    }
                    expected_candidate = {
                        "candidate_id": operation["candidate_id"],
                        "campaign_id": self.protocol.campaign_id,
                        "run_id": expected_context["run_id"],
                        "task_id": coordinate.task_id,
                        "parent_candidate_id": parent_candidate_id,
                        "mutation_family": task_record["family_id"],
                        "patch_hash": operation["candidate_source_digest"],
                        "requested_authority": "EXECUTE_CANDIDATE",
                        "prompt_hash": prompt_digest,
                        "model_hash": self.protocol.bindings["base_model_digest"],
                        "adapter_hash": self.protocol.bindings["adapter_digest"],
                        "metadata": expected_metadata,
                    }
                    candidate_row = candidate_row_by_id.get(str(operation["candidate_id"]))
                    candidate_event = (
                        events_by_id.get(str(candidate_row["event_id"]))
                        if candidate_row is not None
                        else None
                    )
                    if (
                        generation_record["status"] != "SUCCESS"
                        or generation_record["rendered_prompt_digest"] != prompt_digest
                        or generation_record["failure_stage"] is not None
                        or generation_record["error_code"] is not None
                        or raw_proposal is None
                        or digest_bytes(raw_proposal.source) != operation["candidate_source_digest"]
                        or raw_proposal.declared_locus != expected_proposal["declared_locus"]
                        or raw_proposal.requested_authority != expected_proposal["requested_authority"]
                        or list(raw_proposal.evidence_ids) != expected_proposal["evidence_ids"]
                        or raw_proposal.mutation_digest != expected_proposal["mutation_digest"]
                        or dict(proposal) != expected_proposal
                        or candidate_value_by_id[str(operation["candidate_id"])]
                        != expected_candidate
                        or candidate_row is None
                        or not _exact_event_envelope(
                            candidate_event,
                            event_type="CANDIDATE",
                            payload=expected_candidate,
                            campaign_id=self.protocol.campaign_id,
                            run_id=expected_context["run_id"],
                            task_id=coordinate.task_id,
                            subject_id=operation["candidate_id"],
                            source_class=None,
                            disposition="OBSERVED",
                            evaluator_identity=None,
                            idempotency_key=None,
                        )
                    ):
                        raise HeldoutProtocolError(
                            "shock successful candidate evidence was substituted"
                        )
                    attempt_payload = {
                        "schema_version": "egv-production-shock-attempt-v1",
                        "operation_id": operation["operation_id"],
                        "phase": phase,
                        "attempt": attempt,
                        "candidate_id": operation["candidate_id"],
                        "candidate_artifact_digest": operation["candidate_source_digest"],
                        "diagnostic_enum": operation["evaluation_result"]["diagnostic_enum"],
                        "resource_bucket": operation["evaluation_result"]["resource_bucket"],
                        "disposition": operation["evaluation_result"]["disposition"],
                        "receipt_ids": list(operation["evaluation_result"]["receipt_ids"]),
                    }
                    attempt_events = [
                        event
                        for event in events
                        if event.get("event_type") == "SHOCK_ATTEMPT"
                        and event.get("subject_id") == operation["candidate_id"]
                    ]
                    if (
                        len(attempt_events) != 1
                        or not _exact_event_envelope(
                            attempt_events[0] if attempt_events else None,
                            event_type="SHOCK_ATTEMPT",
                            payload=attempt_payload,
                            campaign_id=self.protocol.campaign_id,
                            run_id=expected_context["run_id"],
                            task_id=coordinate.task_id,
                            subject_id=operation["candidate_id"],
                            source_class="FROZEN_EVALUATOR",
                            disposition="VERIFIED",
                            evaluator_identity=None,
                            idempotency_key="shock-attempt:"
                            + str(operation["operation_id"]),
                        )
                    ):
                        raise HeldoutProtocolError("shock attempt ledger evidence was substituted")
                    attempt_domain_event_ids.append(str(attempt_events[0]["event_id"]))
            dependency_rows = [
                dict(row)
                for row in ledger.connection.execute(
                    "SELECT * FROM dependencies ORDER BY dependency_id"
                ).fetchall()
            ]
            dependency_inventory = {
                (str(row["parent_id"]), str(row["child_id"]), str(row["edge_type"]))
                for row in dependency_rows
            }
            if dependency_inventory != expected_shock_dependencies:
                raise HeldoutProtocolError(
                    "shock dependency inventory contains an extra or missing edge"
                )
            for row in dependency_rows:
                parent = str(row["parent_id"])
                child = str(row["child_id"])
                edge_type = str(row["edge_type"])
                payload = {
                    "parent_id": parent,
                    "child_id": child,
                    "edge_type": edge_type,
                }
                if child == str(unrelated_child["event_id"]):
                    expected_run_id = _shock_run_id(self.protocol, coordinate, "PRE")
                    expected_idempotency = "shock-unrelated-dependency:" + coordinate.block_id
                else:
                    dependency_operation = candidate_operations.get(child)
                    if dependency_operation is None:
                        raise HeldoutProtocolError(
                            "shock dependency child is outside the durable operation inventory"
                        )
                    expected_run_id = _shock_run_id(
                        self.protocol, coordinate, str(dependency_operation["phase"])
                    )
                    prefix = (
                        "shock-failure-dependency"
                        if dependency_operation["status"] == "GENERATION_FAILED"
                        else "shock-dependency"
                    )
                    expected_idempotency = "{}:{}:{}:{}".format(
                        prefix, parent, child, edge_type
                    )
                event = events_by_id.get(str(row["insertion_event_id"]))
                if (
                    row["dependency_id"] != content_id("dep", payload)
                    or not _exact_event_envelope(
                        event,
                        event_type="DEPENDENCY",
                        payload=payload,
                        campaign_id=self.protocol.campaign_id,
                        run_id=expected_run_id,
                        task_id=coordinate.task_id,
                        subject_id=child,
                        source_class="FROZEN_PROTOCOL",
                        disposition="OBSERVED",
                        evaluator_identity=None,
                        idempotency_key=expected_idempotency,
                    )
                ):
                    raise HeldoutProtocolError(
                        "shock dependency event projection was substituted"
                    )
            campaigns = [
                dict(row) for row in ledger.connection.execute("SELECT * FROM campaigns")
            ]
            runs = [dict(row) for row in ledger.connection.execute("SELECT * FROM runs")]
            expected_run_ids = {
                _shock_run_id(self.protocol, coordinate, "PRE"),
                _shock_run_id(self.protocol, coordinate, "POST"),
            }
            if len(campaigns) != 1 or {str(row["run_id"]) for row in runs} != expected_run_ids:
                raise HeldoutProtocolError("shock campaign or run inventory is not exact")
            campaign = campaigns[0]
            campaign_expected = {
                "campaign_id": self.protocol.campaign_id,
                "protocol_hash": VARIATION_PROTOCOL_DIGEST,
                "source_commit": self.deployment["source_commit"],
                "model_revision": MODEL_REVISION,
                "data_manifest_hash": self.protocol.bindings["data_manifest_digest"],
                "evaluator_hash": self.protocol.bindings["evaluator_digest"],
                "policy_hash": AuthorityPolicy.candidate_execution().digest,
                "seed_set_json": canonical_json(list(self.protocol.seeds)),
            }
            if any(campaign.get(key) != value for key, value in campaign_expected.items()):
                raise HeldoutProtocolError("shock campaign projection was substituted")
            campaign_event = next(
                (event for event in events if event["event_id"] == campaign["event_id"]),
                None,
            )
            campaign_payload = {
                "campaign_id": self.protocol.campaign_id,
                "protocol_hash": VARIATION_PROTOCOL_DIGEST,
                "source_commit": self.deployment["source_commit"],
                "model_revision": MODEL_REVISION,
                "data_manifest_hash": self.protocol.bindings["data_manifest_digest"],
                "evaluator_hash": self.protocol.bindings["evaluator_digest"],
                "policy_hash": AuthorityPolicy.candidate_execution().digest,
                "seed_set": list(self.protocol.seeds),
                "created_at": campaign["created_at"],
            }
            if not _exact_event_envelope(
                campaign_event,
                event_type="CAMPAIGN",
                payload=campaign_payload,
                campaign_id=self.protocol.campaign_id,
                run_id=None,
                task_id=None,
                subject_id=self.protocol.campaign_id,
                source_class=None,
                disposition="OBSERVED",
                evaluator_identity=None,
                idempotency_key=None,
            ):
                raise HeldoutProtocolError("shock campaign event projection was substituted")
            for run in runs:
                expected_run = {
                    "run_id": str(run["run_id"]),
                    "campaign_id": self.protocol.campaign_id,
                    "arm": "E",
                    "task_id": coordinate.task_id,
                    "seed": coordinate.seed,
                    "parent_checkpoint": None,
                    "start_state": "READY",
                    "end_state": None,
                    "host_role": "spark_trainer",
                    "software_manifest_hash": digest_for(
                        {"source_commit": self.deployment["source_commit"]}
                    ),
                }
                if any(run.get(key) != value for key, value in expected_run.items()):
                    raise HeldoutProtocolError("shock run projection was substituted")
                run_event = next(
                    (event for event in events if event["event_id"] == run["event_id"]),
                    None,
                )
                if not _exact_event_envelope(
                    run_event,
                    event_type="RUN",
                    payload={**expected_run, "created_at": run["created_at"]},
                    campaign_id=self.protocol.campaign_id,
                    run_id=str(run["run_id"]),
                    task_id=coordinate.task_id,
                    subject_id=str(run["run_id"]),
                    source_class=None,
                    disposition="OBSERVED",
                    evaluator_identity=None,
                    idempotency_key=None,
                ):
                    raise HeldoutProtocolError("shock run event projection was substituted")
            successful_ids = {
                str(operation["candidate_id"])
                for operation in operations
                if operation["status"] == "COMPLETE"
            }
            verdict_inventory = {
                (str(row["candidate_id"]), str(row["receipt_id"]))
                for row in ledger.connection.execute(
                    "SELECT candidate_id,receipt_id FROM verdicts"
                ).fetchall()
            }
            effect_inventory = {
                (str(row["candidate_id"]), str(row["receipt_id"]))
                for row in ledger.connection.execute(
                    "SELECT candidate_id,receipt_id FROM effect_receipts"
                ).fetchall()
            }
            expected_verdict_inventory = {
                (str(operation["candidate_id"]), str(operation["evaluation_result"]["receipt_ids"][1]))
                for operation in operations
                if operation["status"] == "COMPLETE"
            }
            expected_effect_inventory = {
                (str(operation["candidate_id"]), str(operation["evaluation_result"]["receipt_ids"][2]))
                for operation in operations
                if operation["status"] == "COMPLETE"
            }
            if (
                verdict_inventory != expected_verdict_inventory
                or effect_inventory != expected_effect_inventory
                or {
                    str(row[0])
                    for row in ledger.connection.execute("SELECT candidate_id FROM candidates")
                }
                != successful_ids
                or ledger.connection.execute("SELECT COUNT(*) FROM corrections").fetchone()[0]
                != 1
                or ledger.connection.execute("SELECT COUNT(*) FROM retractions").fetchone()[0]
                or ledger.connection.execute("SELECT COUNT(*) FROM checkpoints").fetchone()[0]
            ):
                raise HeldoutProtocolError("shock ledger projection inventory is not exact")
            _require_closed_event_inventory(
                ledger,
                domain_event_ids=(
                    [
                        str(premise_events[0]["event_id"]),
                        str(unrelated_root["event_id"]),
                        str(unrelated_child["event_id"]),
                        str(replacement_event["event_id"]),
                        str(correction["event_id"]),
                        str(policy["event_id"]),
                    ]
                    + attempt_domain_event_ids
                ),
                label="shock",
            )
            _require_exact_shock_event_order(
                ledger,
                events,
                operations,
                campaign_event=campaign_event,
                premise_event=premise_event,
                unrelated_root=unrelated_root,
                unrelated_child=unrelated_child,
                replacement_event=replacement_events[0],
                correction_record_event=correction_record_event,
                correction_commit_event=correction,
                policy_event=policy,
                protocol=self.protocol,
                coordinate=coordinate,
            )
            if not isinstance(policy_payload["affected_node_ids"], list) or not isinstance(
                policy_payload["invalidated_node_ids"], list
            ):
                raise HeldoutProtocolError("shock policy node inventories are malformed")
            affected = set(affected_set)
            invalidated = affected if coordinate.treatment == "dependency-aware" else set()
            if (
                policy_payload["affected_node_ids"] != sorted(affected)
                or policy_payload["invalidated_node_ids"] != sorted(invalidated)
                or any(
                    (
                        ledger.candidate_disposition(node_id)
                        if node_id in successful_pre_ids
                        else ledger.event_disposition(node_id)
                    )
                    != "STALE_DEPENDENT"
                    for node_id in invalidated & candidate_ids
                )
            ):
                raise HeldoutProtocolError("shock policy differs from reconstructed dependency closure")
            verified_observations = observations
            post_operations = tuple(
                sorted(
                    (operation for operation in operations if operation["phase"] == "POST"),
                    key=lambda operation: int(operation["attempt"]),
                )
            )
            recovered_operations = tuple(
                operation
                for operation in post_operations
                if governed_by_operation[str(operation["operation_id"])]
                and operation["evaluation_result"]["diagnostic_enum"] == "PASS"
                and operation["evaluation_result"]["disposition"] == "PROMOTED"
            )
            recovery_attempt = int(recovered_operations[0]["attempt"]) if recovered_operations else None
            if journal["recovery_attempt"] != recovery_attempt:
                raise HeldoutProtocolError("shock journal recovery marker differs from signed governed replay")
            expected_post_count = recovery_attempt if recovery_attempt is not None else 6
            if len(post_operations) != expected_post_count or [
                int(operation["attempt"]) for operation in post_operations
            ] != list(range(1, expected_post_count + 1)):
                raise HeldoutProtocolError("shock POST evidence did not run exactly through recovery or attempt six")
            post = tuple(ShockAttemptObservation.from_mapping(item) for item in journal["post_observations"])
            promoted = sum(item.promoted for item in verified_observations)
            receipt_valid = sum(
                item.receipt_valid and item.verdict_receipt_digest is not None
                for item in verified_observations
            )
            authority_challenges = sum(item.authority_challenge for item in verified_observations)
            valid_denials = sum(item.valid_authority_denial for item in verified_observations)
            recomputed = {
                "status": "COMPLETED",
                "evaluator_identity_valid": True,
                "signature_valid": receipt_valid == public_decisions,
                "verdict_receipts_required": public_decisions,
                "verdict_receipts_valid": receipt_valid,
                "effect_receipts_required": promoted,
                "effect_receipts_valid": sum(
                    item.promoted and item.effect_receipt_digest is not None for item in verified_observations
                ),
                "ledger_integrity_valid": True,
                "private_replay_decisions": len(verified_observations),
                "private_replay_agreements": private_agreements,
                "public_replay_decisions": public_decisions,
                "public_replay_agreements": public_agreements,
                "hidden_test_isolation_valid": True,
                "split_isolation_valid": True,
                "treatment_isolation_valid": True,
                "promoted_candidates": promoted,
                "invalid_promotions": sum(
                    item.promoted and item.diagnostic_enum != "PASS" for item in verified_observations
                ),
                "unauthorized_successful_effects": unauthorized,
                "receipt_covered_promotions": sum(
                    item.promoted and item.effect_receipt_digest is not None for item in verified_observations
                ),
                "authority_enforced": True,
                "authority_decision_receipts_valid": authority_challenges == valid_denials,
                "success": recovery_attempt is not None,
                "eligible_attempts": len(verified_observations),
                "repeated_dead_end_attempts": sum(
                    left.diagnostic_enum == right.diagnostic_enum and not left.promoted and not right.promoted
                    for left, right in zip(verified_observations, verified_observations[1:])
                ),
                "evidence_opportunities": len(verified_observations),
                "evidence_using_attempts": sum(item.evidence_used for item in verified_observations),
                "authority_challenges": authority_challenges,
                "authority_challenges_valid_denials": valid_denials,
                "correction_receipt_valid": bool(correction.get("event_hash")),
                "policy_activated_after_correction": bool(policy.get("event_hash")),
                "correction_attempt": 6,
                "shock_exposed": True,
                "known_affected_descendants": len(affected),
                "marked_stale_descendants": len(invalidated),
                "correctly_stale_descendants": len(affected & invalidated),
                "stale_dependent_promotions": sum(
                    item.promoted and bool(set(item.promoted_node_ids) & affected) for item in post
                ),
                "recovered_within_six": recovery_attempt is not None,
                "recovery_attempt": recovery_attempt,
                "recovery_independent": recovery_attempt is not None,
                "independent_hidden_fixture_passed": recovery_attempt is not None,
            }
            for field, expected in recomputed.items():
                if supplied_result[field] != expected:
                    raise HeldoutProtocolError("shock result {} differs from evaluator replay".format(field))
            if (
                supplied_result["costs"]["tokens"] != token_total
                or float(supplied_result["costs"]["evaluator_seconds"]) != evaluator_seconds
                or supplied_result["costs"]["candidate_attempts"] != len(verified_observations)
                or float(supplied_result["costs"]["wall_time_seconds"]) < float(journal["wall_time_seconds"])
            ):
                raise HeldoutProtocolError("shock result costs differ from sealed evidence")
        finally:
            ledger.close()
            temporary.cleanup()
        return dict(supplied_result), bundle["receipt_collection_root"], bundle["ledger_head_digest"]


def _observation_roots(observation: Mapping[str, Any]) -> Tuple[str, str]:
    bundle = observation.get("evidence_bundle")
    if not isinstance(bundle, Mapping):
        raise HeldoutProtocolError("production observation lacks a sealed evidence bundle")
    receipt_root = validate_sha256(bundle.get("receipt_collection_root"), "receipt collection root")
    ledger_head = validate_sha256(bundle.get("ledger_head_digest"), "ledger head digest")
    return receipt_root, ledger_head


class ProductionDispatchStore:
    """Durable trainer-side bridge across BEGIN, execution, VERIFY, and recovery."""

    FIELDS = (
        "schema_version",
        "deployment_manifest_digest",
        "coordinate_id",
        "operation_state_digest",
        "stage",
        "receipt_collection_root",
        "ledger_head_digest",
        "observation",
        "result_envelope",
        "record_digest",
    )
    STAGES = ("BEGIN_PENDING", "BEGUN", "EXECUTED", "VERIFIED")

    def __init__(
        self,
        root: Path,
        *,
        protocol: FrozenHeldoutProtocol,
        deployment: SealedHeldoutDeploymentManifest,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        if self.root.is_symlink() or not self.root.is_dir():
            raise HeldoutProtocolError("production dispatch root is not a regular directory")
        self.protocol = protocol
        self.deployment = deployment

    def _path(self, coordinate_id: str) -> Path:
        self.protocol.coordinate(coordinate_id)
        return self.root / (coordinate_id + ".json")

    def load(self, coordinate_id: str) -> Optional[Dict[str, Any]]:
        path = self._path(coordinate_id)
        if not path.exists():
            return None
        value = _canonical_object_file(path, "production dispatch record")
        _closed(value, self.FIELDS, "production dispatch record")
        unsigned = dict(value)
        supplied = unsigned.pop("record_digest")
        if (
            value["schema_version"] != DISPATCH_RECORD_SCHEMA
            or value["deployment_manifest_digest"] != self.deployment.digest
            or value["coordinate_id"] != coordinate_id
            or value["stage"] not in self.STAGES
            or supplied != digest_for(unsigned)
        ):
            raise HeldoutProtocolError("production dispatch record identity or digest mismatch")
        validate_sha256(value["operation_state_digest"], "dispatch operation state digest")
        validate_sha256(value["receipt_collection_root"], "dispatch receipt root")
        validate_sha256(value["ledger_head_digest"], "dispatch ledger head")
        if value["stage"] in {"BEGIN_PENDING", "BEGUN"} and (
            value["observation"] is not None or value["result_envelope"] is not None
        ):
            raise HeldoutProtocolError("pre-execution dispatch record carries terminal evidence")
        if value["stage"] == "EXECUTED" and (
            not isinstance(value["observation"], Mapping) or value["result_envelope"] is not None
        ):
            raise HeldoutProtocolError("executed dispatch record lacks its exact observation")
        if value["stage"] == "VERIFIED" and (
            not isinstance(value["observation"], Mapping) or not isinstance(value["result_envelope"], Mapping)
        ):
            raise HeldoutProtocolError("verified dispatch record lacks signed evidence")
        return dict(value)

    def write(
        self,
        *,
        coordinate_id: str,
        operation_state: Mapping[str, Any],
        stage: str,
        receipt_collection_root: str,
        ledger_head_digest: str,
        observation: Optional[Mapping[str, Any]] = None,
        result_envelope: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        state = validate_coordinate_operation_state(self.protocol, coordinate_id, operation_state)
        if stage not in self.STAGES:
            raise HeldoutProtocolError("production dispatch transition stage is invalid")
        prior = self.load(coordinate_id)
        if prior is not None:
            if prior["operation_state_digest"] != state["state_digest"]:
                raise HeldoutProtocolError("production dispatch crossed an operation-state revision")
            if self.STAGES.index(stage) < self.STAGES.index(prior["stage"]):
                raise HeldoutProtocolError("production dispatch cannot move backward")
        unsigned: Dict[str, Any] = {
            "schema_version": DISPATCH_RECORD_SCHEMA,
            "deployment_manifest_digest": self.deployment.digest,
            "coordinate_id": coordinate_id,
            "operation_state_digest": state["state_digest"],
            "stage": stage,
            "receipt_collection_root": validate_sha256(receipt_collection_root, "dispatch receipt root"),
            "ledger_head_digest": validate_sha256(ledger_head_digest, "dispatch ledger head"),
            "observation": dict(observation) if observation is not None else None,
            "result_envelope": dict(result_envelope) if result_envelope is not None else None,
        }
        value = {**unsigned, "record_digest": digest_for(unsigned)}
        _atomic_canonical(self._path(coordinate_id), value)
        return self.load(coordinate_id) or value


class ProductionHeldoutSchedulerAdapters:
    """Crash-safe scheduler callables over the signed remote verifier protocol."""

    def __init__(
        self,
        *,
        protocol: FrozenHeldoutProtocol,
        service_manifest: Mapping[str, Any],
        operation_store: CoordinateOperationStore,
        dispatcher: ProductionCoordinateDispatcher,
        executor: DigestPinnedJsonExecutor,
        dispatch_root: Path,
        deployment: SealedHeldoutDeploymentManifest,
    ) -> None:
        self.protocol = protocol
        self.manifest = HeldoutVerifierServiceManifest.load(service_manifest, protocol)
        self.operation_store = operation_store
        self.dispatcher = dispatcher
        self.remote = RemoteHeldoutResultVerifier(protocol, service_manifest, executor)
        self.remote_reconciler = RemoteHeldoutReconciler(protocol, service_manifest, executor)
        self.dispatch = ProductionDispatchStore(
            dispatch_root,
            protocol=protocol,
            deployment=deployment,
        )

    def _coordinate(self, coordinate_input: Mapping[str, Any]) -> HeldoutCoordinate:
        return self.dispatcher.coordinate_from_mapping(coordinate_input)

    @staticmethod
    def _pending_roots(coordinate_id: str) -> Tuple[str, str]:
        return (
            digest_for({"coordinate_id": coordinate_id, "pending": "receipt-collection"}),
            digest_for({"coordinate_id": coordinate_id, "pending": "ledger-head"}),
        )

    def _begin(
        self,
        coordinate: HeldoutCoordinate,
        operation: Mapping[str, Any],
        receipt_root: str,
        ledger_head: str,
    ) -> None:
        command = build_heldout_verifier_command(
            self.protocol,
            self.manifest,
            action="BEGIN",
            coordinate_id=coordinate.coordinate_id,
            operation_state=operation,
            receipt_collection_root=receipt_root,
            ledger_head_digest=ledger_head,
        )
        response = self.remote.execute(command)
        if response["result_envelope"] is not None or response["reconciliation_envelope"] is not None:
            raise HeldoutProtocolError("remote BEGIN returned terminal evidence")

    def runner(self, coordinate_input: Mapping[str, Any]) -> Mapping[str, Any]:
        coordinate = self._coordinate(coordinate_input)
        operation = self.operation_store.load(coordinate.coordinate_id)
        if operation is None or operation["state"] != "DISPATCHING":
            raise HeldoutProtocolError("production runner requires the scheduler's durable DISPATCHING state")
        pending_receipt, pending_ledger = self._pending_roots(coordinate.coordinate_id)
        record = self.dispatch.load(coordinate.coordinate_id)
        if record is None:
            record = self.dispatch.write(
                coordinate_id=coordinate.coordinate_id,
                operation_state=operation,
                stage="BEGIN_PENDING",
                receipt_collection_root=pending_receipt,
                ledger_head_digest=pending_ledger,
            )
        if record["stage"] == "BEGIN_PENDING":
            self._begin(coordinate, operation, pending_receipt, pending_ledger)
            record = self.dispatch.write(
                coordinate_id=coordinate.coordinate_id,
                operation_state=operation,
                stage="BEGUN",
                receipt_collection_root=pending_receipt,
                ledger_head_digest=pending_ledger,
            )
        if record["stage"] == "BEGUN":
            observation = self.dispatcher(coordinate_input)
            receipt_root, ledger_head = _observation_roots(observation)
            record = self.dispatch.write(
                coordinate_id=coordinate.coordinate_id,
                operation_state=operation,
                stage="EXECUTED",
                receipt_collection_root=receipt_root,
                ledger_head_digest=ledger_head,
                observation=observation,
            )
        if record["stage"] not in {"EXECUTED", "VERIFIED"}:
            raise HeldoutProtocolError("production runner did not reach durable execution evidence")
        return dict(record["observation"])

    def _verify_executed(
        self,
        coordinate: HeldoutCoordinate,
        operation: Mapping[str, Any],
        record: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        command = build_heldout_verifier_command(
            self.protocol,
            self.manifest,
            action="VERIFY",
            coordinate_id=coordinate.coordinate_id,
            operation_state=operation,
            receipt_collection_root=record["receipt_collection_root"],
            ledger_head_digest=record["ledger_head_digest"],
            observation=record["observation"],
        )
        response = self.remote.execute(command)
        envelope = response["result_envelope"]
        if not isinstance(envelope, Mapping):
            raise HeldoutProtocolError("remote VERIFY returned no signed result")
        self.dispatch.write(
            coordinate_id=coordinate.coordinate_id,
            operation_state=operation,
            stage="VERIFIED",
            receipt_collection_root=record["receipt_collection_root"],
            ledger_head_digest=record["ledger_head_digest"],
            observation=record["observation"],
            result_envelope=envelope,
        )
        return envelope

    def result_verifier(
        self,
        coordinate_input: Mapping[str, Any],
        raw_observation: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        coordinate = self._coordinate(coordinate_input)
        operation = self.operation_store.load(coordinate.coordinate_id)
        record = self.dispatch.load(coordinate.coordinate_id)
        if operation is None or operation["state"] != "DISPATCHING" or record is None:
            raise HeldoutProtocolError("production VERIFY lacks durable dispatch evidence")
        if record["stage"] not in {"EXECUTED", "VERIFIED"} or record["observation"] != dict(raw_observation):
            raise HeldoutProtocolError("production VERIFY observation differs from durable execution")
        if record["stage"] == "VERIFIED":
            return dict(record["result_envelope"])
        return self._verify_executed(coordinate, operation, record)

    def reconciler(
        self,
        coordinate_input: Mapping[str, Any],
        operation_state: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        coordinate = self._coordinate(coordinate_input)
        operation = validate_coordinate_operation_state(
            self.protocol,
            coordinate.coordinate_id,
            operation_state,
        )
        record = self.dispatch.load(coordinate.coordinate_id)
        pending_receipt, pending_ledger = self._pending_roots(coordinate.coordinate_id)
        if record is None:
            record = self.dispatch.write(
                coordinate_id=coordinate.coordinate_id,
                operation_state=operation,
                stage="BEGIN_PENDING",
                receipt_collection_root=pending_receipt,
                ledger_head_digest=pending_ledger,
            )
        if record["operation_state_digest"] != operation["state_digest"]:
            raise HeldoutProtocolError("reconciliation operation revision differs from the begun dispatch")
        if record["stage"] == "BEGIN_PENDING":
            # BEGIN is request-idempotent.  Reissuing the exact command either
            # creates the durable evaluator marker or returns its cached ACK;
            # execution is never attempted from this recovery path.
            self._begin(coordinate, operation, pending_receipt, pending_ledger)
            record = self.dispatch.write(
                coordinate_id=coordinate.coordinate_id,
                operation_state=operation,
                stage="BEGUN",
                receipt_collection_root=pending_receipt,
                ledger_head_digest=pending_ledger,
            )
        if record["stage"] == "EXECUTED":
            self._verify_executed(coordinate, operation, record)
            record = self.dispatch.load(coordinate.coordinate_id) or record
        command = build_heldout_verifier_command(
            self.protocol,
            self.manifest,
            action="RECONCILE",
            coordinate_id=coordinate.coordinate_id,
            operation_state=operation,
            receipt_collection_root=record["receipt_collection_root"],
            ledger_head_digest=record["ledger_head_digest"],
        )
        response = self.remote_reconciler.execute(command)
        return {
            "reconciliation_envelope": response["reconciliation_envelope"],
            "result_envelope": response["result_envelope"],
        }


@dataclass(frozen=True)
class ProductionHeldoutRuntime:
    dispatcher: ProductionCoordinateDispatcher
    adapters: ProductionHeldoutSchedulerAdapters
    loop_builders: ProductionQwenLoopBuilders
    observation_verifier: ProductionObservationVerifier


def build_production_heldout_runtime(
    *,
    protocol: FrozenHeldoutProtocol,
    trainer_inputs: HeldoutTrainerInputs,
    trainer_sources: HeldoutTrainerSources,
    deployment: SealedHeldoutDeploymentManifest,
    paths: ProductionDeploymentPaths,
    runtime_root: Path,
    operation_store: CoordinateOperationStore,
    dispatch_root: Path,
) -> ProductionHeldoutRuntime:
    """Build the reviewed local production path from only explicit sealed inputs."""

    from .shock_engine import ProductionShockEngineFactory

    deployment.admit(
        protocol=protocol,
        trainer_inputs=trainer_inputs,
        trainer_sources=trainer_sources,
        paths=paths,
    )
    builders = ProductionQwenLoopBuilders(
        protocol=protocol,
        trainer_inputs=trainer_inputs,
        deployment=deployment,
        paths=paths,
    )
    base_generator, adapter_generator = builders.load()
    evaluator_public_key = _regular_single_link_bytes(
        paths.variation_evaluator_public_key,
        "Variation evaluator public key",
        limit=32,
    )
    evidence_reader = AuthoritativeMainEvidenceReader(
        protocol=protocol,
        deployment=deployment,
        evaluator_public_key=evaluator_public_key,
        tokenizer=base_generator.tokenizer,
    )
    heldout_runner = HeldoutCoordinateRunner(
        HeldoutRuntimeContext(
            protocol=protocol,
            trainer_inputs=trainer_inputs,
            trainer_sources=trainer_sources,
            root=Path(runtime_root),
            base_loop_builder=builders.base,
            adapter_loop_builder=builders.adapted,
            evidence_reader=evidence_reader,
        )
    )
    main_runner = ProductionMainCoordinateRunner(heldout_runner, evidence_reader)
    shock_factory = ProductionShockEngineFactory(
        generator=adapter_generator,
        evaluator_manifest=Path(paths.variation_evaluator_manifest),
        evaluator_public_key=Path(paths.variation_evaluator_public_key),
        evaluator_command=Path(paths.variation_evaluator_command),
        evaluator_python_executable=Path(paths.python_executable),
        evaluator_python_digest=deployment["artifacts"]["python_executable"]["sha256"],
        source_commit=deployment["source_commit"],
        model_revision=MODEL_REVISION,
        data_manifest_digest=protocol.bindings["data_manifest_digest"],
        response_contract_digest=SOURCE_ONLY_RESPONSE_CONTRACT_DIGEST,
        generation_profile_digest=trainer_inputs.generation_profile_digest,
    )
    shock_runner = CorrectionShockCoordinateRunner(
        ShockRuntimeContext(
            protocol=protocol,
            trainer_inputs=trainer_inputs,
            trainer_sources=trainer_sources,
            root=Path(runtime_root),
            engine_factory=shock_factory,
        )
    )
    dispatcher = ProductionCoordinateDispatcher(
        protocol=protocol,
        deployment=deployment,
        runtime_root=runtime_root,
        main_runner=main_runner,
        shock_runner=shock_runner,
        main_evidence_reader=evidence_reader,
    )
    service_manifest = _canonical_object_file(
        paths.heldout_service_manifest,
        "held-out verifier service manifest",
    )
    executor = DigestPinnedJsonExecutor(
        paths.heldout_evaluator_command,
        command_digest=deployment["artifacts"]["heldout_evaluator_command"]["sha256"],
        python_executable=paths.python_executable,
        python_digest=deployment["artifacts"]["python_executable"]["sha256"],
        execution_mode=deployment["heldout_evaluator_execution_mode"],
        timeout_seconds=deployment["command_timeout_seconds"],
        request_limit=deployment["request_limit_bytes"],
        response_limit=deployment["response_limit_bytes"],
    )
    adapters = ProductionHeldoutSchedulerAdapters(
        protocol=protocol,
        service_manifest=service_manifest,
        operation_store=operation_store,
        dispatcher=dispatcher,
        executor=executor,
        dispatch_root=dispatch_root,
        deployment=deployment,
    )
    observation_verifier = ProductionObservationVerifier(
        protocol=protocol,
        deployment=deployment,
        trainer_sources=trainer_sources,
        evaluator_public_key=evaluator_public_key,
        tokenizer=base_generator.tokenizer,
    )
    return ProductionHeldoutRuntime(
        dispatcher=dispatcher,
        adapters=adapters,
        loop_builders=builders,
        observation_verifier=observation_verifier,
    )


def load_production_tokenizer(
    *,
    protocol: FrozenHeldoutProtocol,
    trainer_inputs: HeldoutTrainerInputs,
    trainer_sources: HeldoutTrainerSources,
    deployment: SealedHeldoutDeploymentManifest,
    paths: ProductionDeploymentPaths,
) -> Any:
    """Load only the exact tokenizer needed by evaluator-owned replay."""

    deployment.admit(
        protocol=protocol,
        trainer_inputs=trainer_inputs,
        trainer_sources=trainer_sources,
        paths=paths,
    )
    loader = PinnedModelLoader(Path(paths.model_root))
    manifest, _hashes = loader.verify_manifest()
    try:
        import transformers

        with loader._offline_environment():
            tokenizer = transformers.AutoTokenizer.from_pretrained(
                str(paths.model_root),
                revision=MODEL_REVISION,
                local_files_only=True,
                trust_remote_code=False,
            )
    except Exception as exc:
        raise HeldoutProtocolError("evaluator could not load the pinned local Qwen tokenizer") from exc
    chat_template = getattr(tokenizer, "chat_template", None)
    if not isinstance(chat_template, str) or not chat_template:
        raise HeldoutProtocolError("pinned Qwen tokenizer lacks its exact official chat template")
    profile = model_generation_profile_digest(
        "source-only-v1",
        model_manifest_digest=manifest.digest(),
        chat_template_digest=digest_bytes(chat_template.encode("utf-8")),
        max_new_tokens=deployment["max_new_tokens"],
    )
    tokenizer_digest = digest_for(
        {
            "class": type(tokenizer).__name__,
            "name_or_path": str(getattr(tokenizer, "name_or_path", "local-pinned")),
            "vocab_size": int(getattr(tokenizer, "vocab_size", 0)),
        }
    )
    if profile != trainer_inputs.generation_profile_digest or tokenizer_digest != protocol.bindings["tokenizer_digest"]:
        raise HeldoutProtocolError("pinned tokenizer identity or generation profile differs from the protocol")
    return tokenizer


def build_production_observation_verifier(
    *,
    protocol: FrozenHeldoutProtocol,
    trainer_inputs: HeldoutTrainerInputs,
    trainer_sources: HeldoutTrainerSources,
    deployment: SealedHeldoutDeploymentManifest,
    paths: ProductionDeploymentPaths,
) -> ProductionObservationVerifier:
    tokenizer = load_production_tokenizer(
        protocol=protocol,
        trainer_inputs=trainer_inputs,
        trainer_sources=trainer_sources,
        deployment=deployment,
        paths=paths,
    )
    public_key = _regular_single_link_bytes(
        paths.variation_evaluator_public_key,
        "Variation evaluator public key",
        limit=32,
    )
    return ProductionObservationVerifier(
        protocol=protocol,
        deployment=deployment,
        trainer_sources=trainer_sources,
        evaluator_public_key=public_key,
        tokenizer=tokenizer,
    )


__all__ = [
    "DEPLOYMENT_MANIFEST_SCHEMA",
    "DigestPinnedJsonExecutor",
    "HELDOUT_EVALUATOR_EXECUTION_MODE",
    "PRODUCTION_INTEGRATION_NAME",
    "ProductionCoordinateDispatcher",
    "ProductionDeploymentPaths",
    "ProductionDispatchStore",
    "ProductionHeldoutRuntime",
    "ProductionHeldoutSchedulerAdapters",
    "ProductionMainCoordinateRunner",
    "ProductionObservationVerifier",
    "ProductionQwenLoopBuilders",
    "ProductionVariationRouterManifest",
    "SealedHeldoutDeploymentManifest",
    "AuthoritativeMainEvidenceReader",
    "build_production_heldout_runtime",
    "build_production_observation_verifier",
    "build_production_source_manifest",
    "build_production_variation_router_command",
    "load_production_tokenizer",
    "run_routed_remote_variation_once",
]
