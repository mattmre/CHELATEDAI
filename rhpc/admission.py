"""Fail-closed admission checks for an official RHPC Spark observation."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import stat
import subprocess
from typing import Any, Dict, Optional

from egv.canonical import digest_for, validate_sha256
from egv.public import verify_public_projection

from .errors import RHPCAdmissionError


def _git_object_id(value: Any) -> str:
    if not isinstance(value, str) or len(value) not in (40, 64) or value != value.lower():
        raise RHPCAdmissionError("source Git object ID is malformed")
    try:
        int(value, 16)
    except ValueError as exc:
        raise RHPCAdmissionError("source Git object ID is malformed") from exc
    return value


def _sha256_file(path: Path) -> str:
    try:
        info = path.lstat()
    except OSError as exc:
        raise RHPCAdmissionError("predecessor evidence member is unavailable") from exc
    if stat.S_ISLNK(info.st_mode) or not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
        raise RHPCAdmissionError("predecessor evidence members must be regular single-link files")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _projection_tree_digest(root: Path) -> str:
    projection = Path(root)
    if projection.is_symlink() or not projection.is_dir():
        raise RHPCAdmissionError("predecessor projection must be a non-link directory")
    inventory = []
    for directory, directories, files in os.walk(str(projection), followlinks=False):
        base = Path(directory)
        for name in directories:
            if (base / name).is_symlink():
                raise RHPCAdmissionError("predecessor projection may not contain linked directories")
        for name in files:
            member = base / name
            relative = member.relative_to(projection).as_posix()
            inventory.append(
                {
                    "path": relative,
                    "size_bytes": member.lstat().st_size,
                    "sha256": _sha256_file(member),
                }
            )
    if not inventory:
        raise RHPCAdmissionError("predecessor projection is empty")
    return digest_for(sorted(inventory, key=lambda item: item["path"]))


def source_checkout_state(root: Path) -> Dict[str, Any]:
    """Return the exact commit and cleanliness of a Git checkout."""

    checkout = Path(root).resolve()
    try:
        top = subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            cwd=str(checkout),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(checkout),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        porcelain = subprocess.run(
            ["git", "status", "--porcelain=v1", "--untracked-files=all"],
            cwd=str(checkout),
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RHPCAdmissionError("source checkout state cannot be established") from exc
    if Path(top).resolve() != checkout:
        raise RHPCAdmissionError("source root must be the Git toplevel")
    normalized_sha = _git_object_id(sha)
    return {
        "source_root": str(checkout),
        "source_git_sha": normalized_sha,
        "source_tree_clean": porcelain == "",
    }


def verify_predecessor_completion(
    projection: Path,
    public_key: Path,
    *,
    protocol_digest: str,
    evaluator_digest: Optional[str] = None,
) -> Dict[str, Any]:
    """Verify the prior EGV terminal seal and its bound restoration receipt."""

    projection_path = Path(projection)
    public_key_path = Path(public_key)
    try:
        normalized_protocol = validate_sha256(protocol_digest, "predecessor protocol digest")
        normalized_evaluator = (
            validate_sha256(evaluator_digest, "predecessor evaluator digest")
            if evaluator_digest is not None
            else None
        )
        projection_digest = _projection_tree_digest(projection_path)
        public_key_digest = _sha256_file(public_key_path)
        report = verify_public_projection(
            projection_path,
            public_key_path.read_bytes(),
            protocol_digest=normalized_protocol,
            evaluator_digest=normalized_evaluator,
            require_terminal_seal=True,
        )
    except Exception as exc:
        raise RHPCAdmissionError("predecessor public projection did not verify closed") from exc
    if not report.terminal_seal_verified:
        raise RHPCAdmissionError("predecessor terminal seal is absent")
    value = report.to_dict()
    if value.get("terminal_seal_verified") is not True or value.get("valid") is not True:
        raise RHPCAdmissionError("predecessor completion report is contradictory")
    return {
        "schema_version": "rhpc-predecessor-completion-v1",
        "projection_tree_sha256": projection_digest,
        "public_key_sha256": public_key_digest,
        "protocol_digest": normalized_protocol,
        "evaluator_digest": normalized_evaluator,
        "verification": value,
        "verification_digest": digest_for(value),
    }


def admit_official_spark_run(
    source_root: Path,
    projection: Path,
    public_key: Path,
    *,
    protocol_digest: str,
    evaluator_digest: Optional[str] = None,
) -> Dict[str, Any]:
    """Admit an official run only from a clean source and sealed predecessor."""

    source = source_checkout_state(source_root)
    if source["source_tree_clean"] is not True:
        raise RHPCAdmissionError("official Spark execution requires a clean source tree")
    predecessor = verify_predecessor_completion(
        projection,
        public_key,
        protocol_digest=protocol_digest,
        evaluator_digest=evaluator_digest,
    )
    return {"source": source, "predecessor": predecessor}


__all__ = [
    "admit_official_spark_run",
    "source_checkout_state",
    "verify_predecessor_completion",
]
