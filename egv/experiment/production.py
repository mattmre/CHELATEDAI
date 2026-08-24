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
import stat
import subprocess
import sys
import tempfile
import threading
from types import MappingProxyType
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ..canonical import (
    GENESIS_HASH,
    canonical_bytes,
    content_id,
    digest_bytes,
    digest_for,
    validate_sha256,
)
from ..evaluation.authority import AuthorityPolicy
from ..ledger import EvidenceLedger
from ..receipts import ReceiptSigner, load_public_key, receipt_hash
from ..variation.adapter import ADAPTER_MANIFEST_NAME, SealedAdapterArtifact
from ..variation.arms import arm_policy
from ..variation.generator import ModelCandidateGenerator, SOURCE_ONLY_RESPONSE_CONTRACT_DIGEST
from ..variation.generator import model_generation_profile_digest
from ..variation.loop import (
    MAX_CANDIDATE_ATTEMPTS,
    VARIATION_PROTOCOL_DIGEST,
    AttemptRecord,
    BoundedCandidateLoop,
    VariationReport,
)
from ..variation.model import MODEL_REVISION, PinnedModelLoader
from ..variation.private import PrivateTrajectoryStore
from ..variation.remote import (
    REMOTE_VARIATION_REQUEST_LIMIT,
    REMOTE_VARIATION_TIMEOUT_SECONDS,
    RemoteControllerEvaluationGateway,
    RemoteEvaluatorServiceManifest,
    run_remote_evaluator_once,
    _validate_remote_result_semantics,
)
from ..variation.errors import VariationConfigurationError
from .heldout import (
    MAIN_PHASE,
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
DEPLOYMENT_MANIFEST_SCHEMA = "egv-heldout-production-deployment-v1"
MAIN_EVIDENCE_SCHEMA = "egv-heldout-main-evidence-v1"
SHOCK_EVIDENCE_SCHEMA = "egv-heldout-shock-evidence-v1"
OBSERVATION_SCHEMA = "egv-heldout-production-observation-v1"
DISPATCH_RECORD_SCHEMA = "egv-heldout-production-dispatch-v1"
PRODUCTION_REQUEST_LIMIT = 8 * 1024 * 1024
PRODUCTION_RESPONSE_LIMIT = 8 * 1024 * 1024
PRODUCTION_STDERR_LIMIT = 256 * 1024
RECEIPT_ROUTER_SCHEMA = "egv-variation-receipt-router-v1"
RECEIPT_ROUTER_CAPABILITY = "content-addressed-receipt-chain-fork-v1"
RECEIPT_ROUTER_CONFIG_SCHEMA = "egv-variation-receipt-router-config-v1"
RECEIPT_ROUTER_STATE_SCHEMA = "egv-remote-variation-state-v1"
PRODUCTION_SOURCE_MANIFEST_SCHEMA = "egv-production-source-manifest-v1"
SOURCE_ISOLATION_SCOPE = "variation-router-only-v1"
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
        return None

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
    python_executable: Path = Path(sys.executable)

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
        "generation_profile_digest",
        "response_contract_digest",
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
        if value["model_revision"] != MODEL_REVISION or not _SOURCE_COMMIT.fullmatch(str(value["source_commit"])):
            raise HeldoutProtocolError("production model revision or source commit is not immutable")
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
            "generation_profile_digest": trainer_inputs.generation_profile_digest,
            "response_contract_digest": SOURCE_ONLY_RESPONSE_CONTRACT_DIGEST,
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
        for record in protocol.heldout_task_records:
            if variation.public_record(record["template_id"]) != dict(record):
                raise HeldoutProtocolError("remote Variation evaluator lacks an exact held-out task binding")


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
    ) -> None:
        self.command = Path(command).resolve()
        self.python_executable = Path(python_executable).resolve()
        self.command_digest = validate_sha256(command_digest, "command digest")
        self.python_digest = validate_sha256(python_digest, "python executable digest")
        self.timeout_seconds = timeout_seconds
        self.request_limit = request_limit
        self.response_limit = response_limit
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
            endpoint = Path(directory) / (self.command_digest + self.command.suffix.lower())
            with endpoint.open("xb") as handle:
                handle.write(command_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            endpoint.chmod(0o500)
            if hashlib.sha256(endpoint.read_bytes()).hexdigest() != self.command_digest:
                raise HeldoutProtocolError("content-addressed command copy failed verification")
            invocation = (
                [str(self.python_executable), str(endpoint)]
                if self.command.suffix.lower() == ".py"
                else [str(endpoint)]
            )
            try:
                process = subprocess.Popen(
                    invocation,
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    env={},
                )
            except OSError as exc:
                raise HeldoutProtocolError("pinned evaluator command could not start") from exc
            stdout = bytearray()
            stderr = bytearray()
            overflow = []

            def drain(stream: Any, sink: bytearray, limit: int, label: str) -> None:
                while True:
                    chunk = stream.read(65536)
                    if not chunk:
                        return
                    if len(sink) + len(chunk) > limit:
                        overflow.append(label)
                        process.kill()
                        return
                    sink.extend(chunk)

            readers = (
                threading.Thread(
                    target=drain, args=(process.stdout, stdout, self.response_limit, "stdout"), daemon=True
                ),
                threading.Thread(
                    target=drain, args=(process.stderr, stderr, PRODUCTION_STDERR_LIMIT, "stderr"), daemon=True
                ),
            )
            for reader in readers:
                reader.start()
            try:
                assert process.stdin is not None
                process.stdin.write(request_bytes)
                process.stdin.close()
                process.wait(timeout=self.timeout_seconds)
            except subprocess.TimeoutExpired as exc:
                process.kill()
                process.wait()
                raise HeldoutProtocolError("pinned evaluator command timed out") from exc
            finally:
                for reader in readers:
                    reader.join(timeout=5)
                for stream in (process.stdout, process.stderr):
                    if stream is not None:
                        stream.close()
            if overflow:
                raise HeldoutProtocolError("pinned evaluator {} exceeded its byte limit".format(overflow[0]))
            if process.returncode != 0:
                raise HeldoutProtocolError("pinned evaluator command returned a nonzero status")
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
    for attempt in attempts:
        candidate_receipts = grouped.get(attempt.candidate_id, tuple())
        actual_ids = tuple(item.get("receipt_id") for item in candidate_receipts)
        if actual_ids != attempt.receipt_ids:
            raise HeldoutProtocolError("attempt receipt IDs differ from the authoritative signed suffix")
        used_receipt_ids.update(str(item) for item in actual_ids)
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
            for candidate_id, context, generation, record_digest in private_store.successful_generations(
                run_id=report.run_id,
                task_id=report.task_id,
                arm_id=report.arm_id,
            )
        }
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
            token_total += count
            token_materials.append(
                {
                    "candidate_id": attempt.candidate_id,
                    "candidate_source_b64": base64.urlsafe_b64encode(source).decode("ascii").rstrip("="),
                    "candidate_source_digest": digest_bytes(source),
                    "candidate_source_tokens": count,
                    "generation_record_digest": record_digest,
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
            "ledger_export": ledger_export,
            "ledger_export_digest": digest_bytes(ledger_export.encode("utf-8")),
            "receipt_collection_root": receipt_root,
            "ledger_head_digest": ledger_head,
        }
        bundle = {**unsigned, "evidence_bundle_digest": digest_for(unsigned)}
        _atomic_canonical(coordinate_root / "main-evidence.json", bundle)
        return evidence

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
        if operation["status"] != "COMPLETE":
            raise HeldoutProtocolError("completed shock coordinate contains a nonterminal operation")
        context, generation, record_digest = private_store.load_generation_success(operation["candidate_id"])
        source = generation.proposal.source
        context_value = asdict(context)
        context_value["retrieval_records"] = [dict(item) for item in context.retrieval_records]
        generation_record = {
            "schema_version": "egv-private-generation-evidence-v1",
            "candidate_id": operation["candidate_id"],
            "status": "SUCCESS",
            "context": context_value,
            "response_contract": context.response_contract,
            "response_contract_digest": context.response_contract_digest,
            "generation_profile_digest": context.generation_profile_digest,
            "rendered_prompt_digest": generation.rendered_prompt_digest,
            "decoded_model_response_digest": generation.decoded_model_response_digest,
            "contract_response_digest": generation.contract_response_digest,
            "proposal_source_digest": digest_bytes(source),
            "failure_stage": None,
            "error_code": None,
        }
        if (
            digest_bytes(source) != operation["candidate_source_digest"]
            or record_digest != operation["generation_evidence_digest"]
            or context.run_id != operation["run_id"]
        ):
            raise HeldoutProtocolError("shock private generation differs from durable operation state")
        source_materials.append(
            {
                "candidate_id": operation["candidate_id"],
                "candidate_source_b64": base64.urlsafe_b64encode(source).decode("ascii").rstrip("="),
                "candidate_source_digest": digest_bytes(source),
                "generation_record_digest": record_digest,
                "generation_record": generation_record,
                "proposal": {
                    "declared_locus": generation.proposal.declared_locus,
                    "requested_authority": generation.proposal.requested_authority,
                    "evidence_ids": list(generation.proposal.evidence_ids),
                    "mutation_digest": generation.proposal.mutation_digest,
                },
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


class ProductionObservationVerifier:
    """Evaluator-owned reconstruction of results from sealed private evidence."""

    MAIN_FIELDS = (
        "schema_version",
        "deployment_manifest_digest",
        "coordinate",
        "report",
        "runtime_evidence",
        "token_materials",
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
        temporary = tempfile.TemporaryDirectory(prefix="egv-heldout-evaluator-replay-")
        try:
            ledger = EvidenceLedger.replay_jsonl(
                ledger_export,
                Path(temporary.name) / "ledger.sqlite3",
            )
            ledger.verify_integrity()
            ledger.verify_receipt_chain(self.evaluator_public_key)
            if (
                digest_for(ledger.receipts()) != bundle["receipt_collection_root"]
                or ledger.ledger_head_hash() != bundle["ledger_head_digest"]
            ):
                raise HeldoutProtocolError("evaluator replay roots differ from the observation")
            return ledger, temporary
        except Exception:
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
        report = _report_from_mapping(bundle["report"])
        evidence = _runtime_evidence_from_mapping(bundle["runtime_evidence"])
        materials = bundle["token_materials"]
        if not isinstance(materials, list) or len(materials) != len(report.attempts):
            raise HeldoutProtocolError("main token evidence does not cover every attempt")
        tokens = 0
        for attempt, material in zip(report.attempts, materials):
            _closed(
                material,
                (
                    "candidate_id",
                    "candidate_source_b64",
                    "candidate_source_digest",
                    "candidate_source_tokens",
                    "generation_record_digest",
                ),
                "main candidate token evidence",
            )
            source = _decode_canonical_b64(material["candidate_source_b64"], "candidate source")
            count = _token_count(self.tokenizer, source)
            if (
                material["candidate_id"] != attempt.candidate_id
                or material["candidate_source_digest"] != digest_bytes(source)
                or attempt.candidate_artifact_digest != digest_bytes(source)
                or material["candidate_source_tokens"] != count
            ):
                raise HeldoutProtocolError("main candidate token evidence was substituted")
            validate_sha256(material["generation_record_digest"], "generation record digest")
            tokens += count
        ledger, temporary = self._replay_ledger(bundle)
        try:
            replay = _replay_facts(ledger, report.attempts)
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
            if report.ledger_head_hash != ledger.ledger_head_hash() or not report.ledger_integrity.get("chain_valid"):
                raise HeldoutProtocolError("main report ledger facts differ from evaluator replay")
            expected_result = derive_verified_main_result(
                self.protocol,
                coordinate,
                report,
                expected_evidence,
                wall_time_seconds=float(supplied_result["costs"]["wall_time_seconds"]),
            )
            if expected_result != dict(supplied_result):
                raise HeldoutProtocolError("main result differs from evaluator-owned reconstruction")
        finally:
            ledger.close()
            temporary.cleanup()
        return expected_result, bundle["receipt_collection_root"], bundle["ledger_head_digest"]

    def _verify_shock(
        self,
        coordinate: HeldoutCoordinate,
        bundle: Mapping[str, Any],
        supplied_result: Mapping[str, Any],
    ) -> Tuple[Mapping[str, Any], str, str]:
        from .shock_engine import (
            SHOCK_CORRECTION_SCHEMA,
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
            for operation in bundle["operations"]:
                operation_id = validate_sha256(operation.get("operation_id"), "shock operation ID")
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
                    "proposal",
                ),
                "shock candidate source evidence",
            )
            source = _decode_canonical_b64(material["candidate_source_b64"], "shock candidate source")
            if (
                digest_bytes(source) != material["candidate_source_digest"]
                or material["candidate_source_digest"] != operation["candidate_source_digest"]
                or material["generation_record_digest"] != operation["generation_evidence_digest"]
            ):
                raise HeldoutProtocolError("shock candidate source evidence was substituted")
            generation_record = material["generation_record"]
            proposal = material["proposal"]
            if (
                not isinstance(generation_record, Mapping)
                or not isinstance(proposal, Mapping)
                or set(proposal) != {"declared_locus", "requested_authority", "evidence_ids", "mutation_digest"}
                or digest_bytes(canonical_bytes(generation_record)) != material["generation_record_digest"]
                or generation_record.get("candidate_id") != operation["candidate_id"]
                or generation_record.get("status") != "SUCCESS"
                or generation_record.get("proposal_source_digest") != operation["candidate_source_digest"]
            ):
                raise HeldoutProtocolError("shock private generation record is not exact")
            count = _token_count(self.tokenizer, source)
            if ShockAttemptObservation.from_mapping(operation["observation"]).tokens != count:
                raise HeldoutProtocolError("shock observation token count differs from exact source bytes")
            token_by_candidate[str(operation["candidate_id"])] = count
            token_total += count
        ledger, temporary = self._replay_ledger(bundle)
        try:
            receipts = {item["receipt_id"]: item for item in ledger.receipts()}
            used_receipts: set[str] = set()
            private_agreements = 0
            public_agreements = 0
            unauthorized = 0
            evaluator_seconds = 0.0
            governed_by_operation: Dict[str, bool] = {}
            candidate_value_by_id: Dict[str, Mapping[str, Any]] = {}
            expected_manifest = _ExpectedVariationManifest(self.protocol)
            for operation in operations:
                evaluation = operation["evaluation_result"]
                expected_run_id = _shock_run_id(self.protocol, coordinate, operation["phase"])
                expected_candidate_id = _shock_candidate_id(
                    self.protocol,
                    coordinate,
                    operation["phase"],
                    int(operation["attempt"]),
                )
                receipt_ids = tuple(str(item) for item in evaluation["receipt_ids"])
                if (
                    operation["run_id"] != expected_run_id
                    or operation["candidate_id"] != expected_candidate_id
                    or len(receipt_ids) != 3
                    or len(set(receipt_ids)) != 3
                    or used_receipts.intersection(receipt_ids)
                    or any(receipt_id not in receipts for receipt_id in receipt_ids)
                ):
                    raise HeldoutProtocolError("shock run, candidate, or receipt identity is not deterministic")
                candidate_receipts = tuple(receipts[item] for item in receipt_ids)
                used_receipts.update(receipt_ids)
                observation = ShockAttemptObservation.from_mapping(operation["observation"])
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
                    "SELECT candidate_json FROM candidates WHERE candidate_id=?",
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
                evaluator_seconds += signed_seconds
                authority_allowed = authority.get("decision") == "ALLOW"
                unauthorized += int(effect["decision"] == "ALLOW" and not authority_allowed)
            if used_receipts != set(receipts):
                raise HeldoutProtocolError("shock ledger carries receipts outside durable attempt operations")
            events = ledger.events()
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
            if (
                len(premise_events) != 1
                or len(unrelated_events) != 2
                or {event["event_type"] for event in unrelated_events}
                != {"SHOCK_UNRELATED_ROOT", "SHOCK_UNRELATED_EVIDENCE"}
                or premise_events[0]["payload"] != expected_premise_payload
                or premise_events[0].get("run_id") != pre_run_id
                or premise_events[0].get("subject_id") != content_id("shock-premise", expected_premise_payload)
                or unrelated_root is None
                or unrelated_child is None
                or unrelated_root.get("payload") != expected_unrelated_root_payload
                or unrelated_child.get("payload") != expected_unrelated_child_payload
                or unrelated_root.get("run_id") != pre_run_id
                or unrelated_child.get("run_id") != pre_run_id
                or unrelated_root.get("subject_id")
                != content_id("shock-unrelated-root", expected_unrelated_root_payload)
                or unrelated_child.get("subject_id")
                != content_id("shock-unrelated-evidence", expected_unrelated_child_payload)
                or unrelated_root.get("disposition") != "VERIFIED"
                or unrelated_child.get("disposition") != "VERIFIED"
            ):
                raise HeldoutProtocolError("shock accepted premise or unrelated evidence cannot be reconstructed")
            accepted_premise_id = str(premise_events[0]["event_id"])
            unrelated_ids = sorted(str(event["event_id"]) for event in unrelated_events)
            candidate_ids = {str(operation["candidate_id"]) for operation in pre_operations}
            if len(candidate_ids) != 6:
                raise HeldoutProtocolError("shock PRE candidate identities are not unique")
            nodes = candidate_ids | set(unrelated_ids) | {accepted_premise_id}
            aliases = {
                str(row["event_id"]): str(row["candidate_id"])
                for row in ledger.connection.execute(
                    "SELECT candidate_id,event_id FROM candidates ORDER BY candidate_id"
                ).fetchall()
            }
            if not candidate_ids.issubset(set(aliases.values())):
                raise HeldoutProtocolError("shock PRE candidates are absent from the replayed ledger")
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
                        "candidate_source_digest": operation["candidate_source_digest"],
                        "generation_evidence_digest": operation["generation_evidence_digest"],
                        "evaluation_result_digest": digest_for(operation["evaluation_result"]),
                        "observation_digest": digest_for(operation["observation"]),
                    }
                    for operation in pre_operations
                ],
                "latest_candidate_id": pre_operations[-1]["candidate_id"],
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
            if (
                len(replacement_events) != 1
                or replacement_events[0].get("event_type") != "SHOCK_CORRECTED_PREMISE"
                or replacement_events[0].get("campaign_id") != self.protocol.campaign_id
                or replacement_events[0].get("task_id") != coordinate.task_id
                or replacement_events[0].get("payload") != expected_replacement_payload
                or correction_row is None
                or correction_row["superseded_id"] != accepted_premise_id
                or correction_row["replacement_id"] != correction_payload["replacement_event_id"]
                or correction_row["event_id"] != correction_payload["correction_record_event_id"]
                or correction_row["reason_code"] != "PREREGISTERED_CORRECTION_AFTER_ATTEMPT_6"
                or correction_row["correction_source"] != "FROZEN_PROTOCOL"
                or correction.get("campaign_id") != self.protocol.campaign_id
                or correction.get("task_id") != coordinate.task_id
                or correction.get("subject_id") != accepted_premise_id
                or policy.get("campaign_id") != self.protocol.campaign_id
                or policy.get("task_id") != coordinate.task_id
                or policy.get("subject_id") != coordinate.coordinate_id
                or not journal["correction_committed"]
                or not journal["policy_activated"]
                or journal["pending_attempt"] is not None
            ):
                raise HeldoutProtocolError("shock correction history is not bound to reconstructed PRE state")
            replacement_event = replacement_events[0]
            candidate_event_ids = {candidate_id: event_id for event_id, candidate_id in aliases.items()}
            if set(candidate_event_ids) != set(candidate_operations):
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
                retrieval_records = []
                if phase == "PRE":
                    retrieval_records.extend(
                        candidate_retrieval_record(prior, historical_pre=True) for prior in prior_pre
                    )
                    parent_candidate_id = prior_pre[-1]["candidate_id"] if prior_pre else None
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
                            candidate_retrieval_record(prior, historical_pre=False) for prior in pre_operations
                        )
                    retrieval_records.extend(
                        candidate_retrieval_record(prior, historical_pre=False) for prior in prior_post
                    )
                    parent_candidate_id = (
                        prior_post[-1]["candidate_id"]
                        if prior_post
                        else (pre_operations[-1]["candidate_id"] if coordinate.treatment == "naive-reuse" else None)
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
                prompt_digest = validate_sha256(
                    context.get("prompt_digest"),
                    "shock generation prompt digest",
                )
                expected_context = {
                    "campaign_id": self.protocol.campaign_id,
                    "run_id": _shock_run_id(self.protocol, coordinate, phase),
                    "seed": coordinate.seed,
                    "arm_id": "E",
                    "task_id": coordinate.task_id,
                    "family_id": task_record["family_id"],
                    "public_locus": task_record["public_locus"],
                    "public_rule_id": task_record["public_rule_id"],
                    "attempt_index": attempt if phase == "PRE" else attempt + 6,
                    "parent_candidate_id": parent_candidate_id,
                    "retrieval_records": retrieval_records,
                    "retrieval_digest": digest_for(retrieval_records),
                    "model_digest": self.protocol.bindings["base_model_digest"],
                    "adapter_digest": self.protocol.bindings["adapter_digest"],
                    "prompt_digest": prompt_digest,
                    "task_statement": "Repair the bounded {} task at {}.".format(
                        task_record["family_id"], task_record["public_locus"]
                    ),
                    "initial_source": initial_source.decode("utf-8"),
                    "initial_source_digest": digest_bytes(initial_source),
                    "response_contract": "source-only-v1",
                    "response_contract_digest": SOURCE_ONLY_RESPONSE_CONTRACT_DIGEST,
                    "generation_profile_digest": self.deployment["generation_profile_digest"],
                }
                _closed(
                    generation_record,
                    (
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
                    ),
                    "shock private generation record",
                )
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
                if (
                    dict(context) != expected_context
                    or digest_for(dict(context)) != operation["context_digest"]
                    or generation_record["schema_version"] != "egv-private-generation-evidence-v1"
                    or generation_record["response_contract"] != "source-only-v1"
                    or generation_record["response_contract_digest"] != SOURCE_ONLY_RESPONSE_CONTRACT_DIGEST
                    or generation_record["generation_profile_digest"] != self.deployment["generation_profile_digest"]
                    or generation_record["rendered_prompt_digest"] != prompt_digest
                    or generation_record["failure_stage"] is not None
                    or generation_record["error_code"] is not None
                    or dict(proposal) != expected_proposal
                    or candidate_value_by_id[str(operation["candidate_id"])] != expected_candidate
                    or actual_dependencies != expected_dependencies
                ):
                    raise HeldoutProtocolError(
                        "shock candidate context, retrieval, or dependency policy was substituted"
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
                    ledger.candidate_disposition(candidate_id) != "STALE_DEPENDENT"
                    for candidate_id in invalidated & candidate_ids
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
            receipt_valid = len(verified_observations)
            authority_challenges = sum(item.authority_challenge for item in verified_observations)
            valid_denials = sum(item.valid_authority_denial for item in verified_observations)
            recomputed = {
                "status": "COMPLETED",
                "evaluator_identity_valid": True,
                "signature_valid": receipt_valid == len(verified_observations),
                "verdict_receipts_required": len(verified_observations),
                "verdict_receipts_valid": receipt_valid,
                "effect_receipts_required": promoted,
                "effect_receipts_valid": sum(
                    item.promoted and item.effect_receipt_digest is not None for item in verified_observations
                ),
                "ledger_integrity_valid": True,
                "private_replay_decisions": len(verified_observations),
                "private_replay_agreements": private_agreements,
                "public_replay_decisions": len(verified_observations),
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
    main_runner = HeldoutCoordinateRunner(
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
    shock_factory = ProductionShockEngineFactory(
        generator=adapter_generator,
        evaluator_manifest=Path(paths.variation_evaluator_manifest),
        evaluator_public_key=Path(paths.variation_evaluator_public_key),
        evaluator_command=Path(paths.variation_evaluator_command),
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
    "PRODUCTION_INTEGRATION_NAME",
    "ProductionCoordinateDispatcher",
    "ProductionDeploymentPaths",
    "ProductionDispatchStore",
    "ProductionHeldoutRuntime",
    "ProductionHeldoutSchedulerAdapters",
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
