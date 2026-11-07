"""Atomic, closed artifacts for deterministic RHPC Stage-A observations."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import stat
import tempfile
from typing import Any, Dict, Mapping, Optional

import numpy as np

from egv.canonical import canonical_bytes, digest_for, validate_sha256

from .errors import RHPCResultError
from .stage_a import PROTOCOL_ID, StageAConfig, validate_stage_a_result


ARTIFACT_SCHEMA = "chelatedai-rhpc-stage-a-artifact-v1"
RESULT_NAME = "rhpc-stage-a.json"
MANIFEST_NAME = "manifest.json"
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "protocol_id",
        "protocol_sha256",
        "run_id",
        "execution_mode",
        "source_git_sha",
        "source_tree_clean",
        "environment",
        "predecessor_completion",
        "result",
    }
)
_ENVIRONMENT_FIELDS = frozenset(
    {"python", "platform", "numpy", "compute_device", "network_access", "gpu_context_created"}
)
_RESULT_BINDING_FIELDS = frozenset({"name", "size_bytes", "sha256", "canonical_digest"})
_PREDECESSOR_FIELDS = frozenset(
    {
        "schema_version",
        "projection_tree_sha256",
        "public_key_sha256",
        "protocol_digest",
        "evaluator_digest",
        "verification",
        "verification_digest",
    }
)
_PREDECESSOR_REPORT_FIELDS = frozenset(
    {
        "valid",
        "replay_type",
        "decisions",
        "recorded_decisions",
        "receipt_count",
        "event_count",
        "candidate_count",
        "dependency_count",
        "public_receipt_head_digest",
        "public_event_head_digest",
        "terminal_seal_verified",
        "limitations",
    }
)


def _git_object_id(value: Any) -> str:
    if not isinstance(value, str) or len(value) not in (40, 64) or value != value.lower():
        raise RHPCResultError("source Git object ID is malformed")
    try:
        int(value, 16)
    except ValueError as exc:
        raise RHPCResultError("source Git object ID is malformed") from exc
    return value


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _regular_file(path: Path, label: str) -> os.stat_result:
    try:
        info = path.lstat()
    except OSError as exc:
        raise RHPCResultError(f"{label} is unavailable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode):
        raise RHPCResultError(f"{label} must be a regular non-link file")
    if info.st_nlink != 1:
        raise RHPCResultError(f"{label} must not be hard-linked")
    return info


def _decode_canonical(path: Path, label: str) -> Mapping[str, Any]:
    _regular_file(path, label)
    try:
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
    except (OSError, UnicodeError, ValueError) as exc:
        raise RHPCResultError(f"{label} cannot be decoded") from exc
    if not isinstance(value, Mapping) or raw != canonical_bytes(value):
        raise RHPCResultError(f"{label} is not canonical JSON")
    return value


def _closed_manifest(value: Mapping[str, Any]) -> Dict[str, Any]:
    if set(value) != _MANIFEST_FIELDS:
        raise RHPCResultError("RHPC artifact manifest is not a closed schema")
    if value.get("schema_version") != ARTIFACT_SCHEMA or value.get("protocol_id") != PROTOCOL_ID:
        raise RHPCResultError("RHPC artifact manifest schema or protocol differs")
    try:
        validate_sha256(value.get("protocol_sha256"), "protocol SHA-256")
    except Exception as exc:
        raise RHPCResultError("RHPC artifact contains a malformed digest") from exc
    _git_object_id(value.get("source_git_sha"))
    if value.get("execution_mode") not in ("DEVELOPMENT", "SPARK_OFFICIAL"):
        raise RHPCResultError("RHPC execution mode is invalid")
    if not isinstance(value.get("run_id"), int) or isinstance(value.get("run_id"), bool):
        raise RHPCResultError("RHPC run ID is invalid")
    if not isinstance(value.get("source_tree_clean"), bool):
        raise RHPCResultError("RHPC source cleanliness value is invalid")
    environment = value.get("environment")
    if not isinstance(environment, Mapping) or set(environment) != _ENVIRONMENT_FIELDS:
        raise RHPCResultError("RHPC environment is not a closed schema")
    if environment.get("compute_device") != "cpu":
        raise RHPCResultError("RHPC Stage A must be CPU-only")
    if environment.get("network_access") is not False or environment.get("gpu_context_created") is not False:
        raise RHPCResultError("RHPC environment contradicts the isolation contract")
    binding = value.get("result")
    if not isinstance(binding, Mapping) or set(binding) != _RESULT_BINDING_FIELDS:
        raise RHPCResultError("RHPC result binding is not a closed schema")
    if binding.get("name") != RESULT_NAME:
        raise RHPCResultError("RHPC result file name differs")
    if not isinstance(binding.get("size_bytes"), int) or isinstance(binding.get("size_bytes"), bool):
        raise RHPCResultError("RHPC result size is invalid")
    try:
        validate_sha256(binding.get("sha256"), "result SHA-256")
        validate_sha256(binding.get("canonical_digest"), "result canonical digest")
    except Exception as exc:
        raise RHPCResultError("RHPC result binding digest is malformed") from exc
    predecessor = value.get("predecessor_completion")
    if value.get("execution_mode") == "SPARK_OFFICIAL":
        if value.get("source_tree_clean") is not True:
            raise RHPCResultError("official RHPC artifact records a dirty source tree")
        if not isinstance(predecessor, Mapping) or set(predecessor) != _PREDECESSOR_FIELDS:
            raise RHPCResultError("official RHPC artifact lacks predecessor completion")
        if predecessor.get("schema_version") != "rhpc-predecessor-completion-v1":
            raise RHPCResultError("official RHPC predecessor completion schema differs")
        try:
            validate_sha256(predecessor.get("projection_tree_sha256"), "predecessor projection tree")
            validate_sha256(predecessor.get("public_key_sha256"), "predecessor public key")
            validate_sha256(predecessor.get("protocol_digest"), "predecessor protocol")
            if predecessor.get("evaluator_digest") is not None:
                validate_sha256(predecessor.get("evaluator_digest"), "predecessor evaluator")
        except Exception as exc:
            raise RHPCResultError("official RHPC predecessor identity is malformed") from exc
        verification = predecessor.get("verification")
        if not isinstance(verification, Mapping) or set(verification) != _PREDECESSOR_REPORT_FIELDS:
            raise RHPCResultError("official RHPC predecessor report is not a closed schema")
        if verification.get("terminal_seal_verified") is not True:
            raise RHPCResultError("official RHPC artifact lacks a verified predecessor terminal seal")
        if verification.get("valid") is not True or verification.get("replay_type") != "public cryptographic decision replay":
            raise RHPCResultError("official RHPC predecessor verification is invalid")
        for count_name in ("receipt_count", "event_count", "candidate_count", "dependency_count"):
            count = verification.get(count_name)
            if not isinstance(count, int) or isinstance(count, bool) or count < 0:
                raise RHPCResultError("official RHPC predecessor count is invalid")
        if not isinstance(verification.get("decisions"), Mapping) or not isinstance(
            verification.get("recorded_decisions"), Mapping
        ):
            raise RHPCResultError("official RHPC predecessor decisions are malformed")
        if not isinstance(verification.get("limitations"), list) or not all(
            isinstance(item, str) for item in verification["limitations"]
        ):
            raise RHPCResultError("official RHPC predecessor limitations are malformed")
        try:
            validate_sha256(verification.get("public_receipt_head_digest"), "predecessor receipt head")
            validate_sha256(verification.get("public_event_head_digest"), "predecessor event head")
        except Exception as exc:
            raise RHPCResultError("official RHPC predecessor head digest is malformed") from exc
        if predecessor.get("verification_digest") != digest_for(verification):
            raise RHPCResultError("predecessor verification digest differs")
    elif predecessor is not None:
        raise RHPCResultError("development artifact may not claim predecessor completion")
    return dict(value)


def _manifest(
    result: Mapping[str, Any],
    result_bytes: bytes,
    *,
    execution_mode: str,
    protocol_sha256: str,
    source_git_sha: str,
    source_tree_clean: bool,
    predecessor_completion: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    return {
        "schema_version": ARTIFACT_SCHEMA,
        "protocol_id": PROTOCOL_ID,
        "protocol_sha256": validate_sha256(protocol_sha256, "protocol SHA-256"),
        "run_id": result["run_id"],
        "execution_mode": execution_mode,
        "source_git_sha": _git_object_id(source_git_sha),
        "source_tree_clean": source_tree_clean,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "compute_device": "cpu",
            "network_access": False,
            "gpu_context_created": False,
        },
        "predecessor_completion": predecessor_completion,
        "result": {
            "name": RESULT_NAME,
            "size_bytes": len(result_bytes),
            "sha256": hashlib.sha256(result_bytes).hexdigest(),
            "canonical_digest": digest_for(result),
        },
    }


def verify_stage_a_artifact(path: Path, *, exact_regeneration: bool = True) -> Dict[str, Any]:
    root = Path(path)
    if root.is_symlink() or not root.is_dir():
        raise RHPCResultError("RHPC artifact directory must be an available non-link directory")
    actual = {child.name for child in root.iterdir()}
    if actual != {RESULT_NAME, MANIFEST_NAME}:
        raise RHPCResultError("RHPC artifact file inventory is not exhaustive")
    manifest = _closed_manifest(_decode_canonical(root / MANIFEST_NAME, "RHPC manifest"))
    result_path = root / RESULT_NAME
    result = _decode_canonical(result_path, "RHPC result")
    result_bytes = result_path.read_bytes()
    binding = manifest["result"]
    if len(result_bytes) != binding["size_bytes"]:
        raise RHPCResultError("RHPC result size changed")
    if hashlib.sha256(result_bytes).hexdigest() != binding["sha256"]:
        raise RHPCResultError("RHPC result SHA-256 changed")
    if digest_for(result) != binding["canonical_digest"]:
        raise RHPCResultError("RHPC result canonical digest changed")
    validate_stage_a_result(result, regenerate=exact_regeneration)
    if result["run_id"] != manifest["run_id"]:
        raise RHPCResultError("RHPC result and manifest run IDs differ")
    if manifest["execution_mode"] == "SPARK_OFFICIAL" and result["official_config"] is not True:
        raise RHPCResultError("official artifact did not use the frozen official configuration")
    return {"manifest": manifest, "result": result}


def publish_stage_a_artifact(
    path: Path,
    result: Mapping[str, Any],
    *,
    execution_mode: str,
    protocol_sha256: str,
    source_git_sha: str,
    source_tree_clean: bool,
    predecessor_completion: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Publish into an absent directory through a verified sibling stage."""

    destination = Path(path).resolve()
    if destination.exists():
        raise RHPCResultError("RHPC artifact destination must be absent")
    validate_stage_a_result(result, regenerate=True)
    if execution_mode == "SPARK_OFFICIAL":
        config = StageAConfig.from_mapping(result["config"])
        if not config.official or source_tree_clean is not True or predecessor_completion is None:
            raise RHPCResultError("official RHPC artifact admission is incomplete")
    elif execution_mode != "DEVELOPMENT":
        raise RHPCResultError("RHPC artifact execution mode is invalid")
    elif predecessor_completion is not None:
        raise RHPCResultError("development artifact cannot bind predecessor completion")
    destination.parent.mkdir(parents=True, exist_ok=True)
    result_bytes = canonical_bytes(result)
    manifest = _manifest(
        result,
        result_bytes,
        execution_mode=execution_mode,
        protocol_sha256=protocol_sha256,
        source_git_sha=source_git_sha,
        source_tree_clean=source_tree_clean,
        predecessor_completion=predecessor_completion,
    )
    staging = Path(tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=str(destination.parent)))
    published = False
    try:
        for name, payload in (
            (RESULT_NAME, result_bytes),
            (MANIFEST_NAME, canonical_bytes(manifest)),
        ):
            with (staging / name).open("xb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        _fsync_directory(staging)
        verify_stage_a_artifact(staging, exact_regeneration=True)
        staging.replace(destination)
        published = True
        _fsync_directory(destination.parent)
    finally:
        if not published and staging.exists():
            shutil.rmtree(staging)
    return verify_stage_a_artifact(destination, exact_regeneration=True)


__all__ = [
    "ARTIFACT_SCHEMA",
    "MANIFEST_NAME",
    "RESULT_NAME",
    "publish_stage_a_artifact",
    "verify_stage_a_artifact",
]
