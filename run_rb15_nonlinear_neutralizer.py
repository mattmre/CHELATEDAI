"""Run one frozen RB-15 Stage-A nonlinear-neutraliser fixture.

The runner writes a canonical JSON result and a small integrity manifest.  A
scientific or numerical failure is retained in the result and is not converted
into a process failure; malformed input or an artifact-integrity failure is.
"""

from __future__ import annotations

import argparse
import hashlib
import hmac
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Mapping, Tuple

from nonlinear_neutralizer_experiments import (
    NOVELTY_CLAIM_STATUS,
    PROTOCOL_ID,
    RUN_IDS,
    SCIENTIFIC_CLAIM_STATUS,
    run_stage_a,
)


MANIFEST_SCHEMA = "CHELATEDAI-RB15-STAGE-A-MANIFEST-v1"
STAGE_FILENAME = "stage_a.json"
MANIFEST_FILENAME = "manifest.json"


class ArtifactIntegrityError(ValueError):
    """Raised when an RB-15 artifact set fails its integrity contract."""


def _canonical_json_bytes(payload: Mapping[str, object]) -> bytes:
    return (
        json.dumps(
            payload,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=str(path.parent),
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            try:
                temporary_path.unlink()
            except FileNotFoundError:
                pass


def _checked_output_directory(output_directory: Path, *, create: bool) -> Path:
    directory = Path(output_directory)
    if directory.exists() and directory.is_symlink():
        raise ArtifactIntegrityError("output directory must not be a symlink")
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    if not directory.is_dir():
        raise ArtifactIntegrityError("output directory is not a directory")
    return directory


def _checked_member(output_directory: Path, member: object) -> Path:
    if not isinstance(member, str) or not member:
        raise ArtifactIntegrityError("artifact member path must be a nonempty string")
    relative = Path(member)
    if (
        relative.is_absolute()
        or member in (".", "..")
        or len(relative.parts) != 1
        or relative.name != member
    ):
        raise ArtifactIntegrityError("artifact member path must be one relative filename")
    candidate = output_directory / relative
    if candidate.is_symlink():
        raise ArtifactIntegrityError("artifact member must not be a symlink")
    if candidate.parent.resolve() != output_directory.resolve():
        raise ArtifactIntegrityError("artifact member escapes the output directory")
    return candidate


def _resource_gates(payload: Mapping[str, object]) -> Dict[str, object]:
    resource = payload.get("resource_usage")
    if not isinstance(resource, dict):
        raise ArtifactIntegrityError("stage payload is missing resource_usage")
    gates = {
        "rss_gate_passed": resource.get("rss_gate_passed"),
        "wall_gate_passed": resource.get("wall_gate_passed"),
    }
    if any(type(value) is not bool for value in gates.values()):
        raise ArtifactIntegrityError("stage payload has invalid resource gates")
    return gates


def _build_manifest(payload: Mapping[str, object], stage_bytes: bytes) -> Dict[str, object]:
    required = (
        "protocol_id",
        "run_id",
        "status",
        "scientific_claim_status",
        "novelty_claim_status",
        "failure_count",
    )
    missing = [name for name in required if name not in payload]
    if missing:
        raise ArtifactIntegrityError(
            "stage payload is missing manifest fields: " + ", ".join(missing)
        )
    return {
        "manifest_schema": MANIFEST_SCHEMA,
        "protocol_id": payload["protocol_id"],
        "run_id": payload["run_id"],
        "status": payload["status"],
        "scientific_claim_status": payload["scientific_claim_status"],
        "novelty_claim_status": payload["novelty_claim_status"],
        "failure_count": payload["failure_count"],
        "resource_gates": _resource_gates(payload),
        "stage_artifact": {
            "path": STAGE_FILENAME,
            "sha256": _sha256(stage_bytes),
            "bytes": len(stage_bytes),
        },
    }


def write_run(output_directory: Path, run_id: int) -> Dict[str, object]:
    """Execute and atomically persist one frozen run, including failures."""

    if type(run_id) is not int or run_id not in RUN_IDS:
        raise ArtifactIntegrityError("run_id must be exactly 7 or 11")
    directory = _checked_output_directory(Path(output_directory), create=True)
    stage_path = _checked_member(directory, STAGE_FILENAME)
    manifest_path = _checked_member(directory, MANIFEST_FILENAME)

    payload = run_stage_a(run_id)
    stage_bytes = _canonical_json_bytes(payload)
    manifest = _build_manifest(payload, stage_bytes)
    manifest_bytes = _canonical_json_bytes(manifest)

    _atomic_write(stage_path, stage_bytes)
    _atomic_write(manifest_path, manifest_bytes)
    verified, _ = verify_manifest(directory)
    return verified


def _load_canonical_mapping(path: Path, label: str) -> Tuple[Dict[str, object], bytes]:
    if not path.is_file():
        raise ArtifactIntegrityError(f"missing {label} file")
    raw = path.read_bytes()
    try:
        decoded = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError(f"invalid {label} JSON") from exc
    if not isinstance(decoded, dict):
        raise ArtifactIntegrityError(f"{label} JSON must be an object")
    try:
        canonical = _canonical_json_bytes(decoded)
    except (TypeError, ValueError) as exc:
        raise ArtifactIntegrityError(f"{label} JSON is not canonicalizable") from exc
    if raw != canonical:
        raise ArtifactIntegrityError(f"{label} JSON is not canonical")
    return decoded, raw


def verify_manifest(output_directory: Path) -> Tuple[Dict[str, object], Dict[str, object]]:
    """Verify canonical bytes, paths, digest, size, and duplicated key fields."""

    directory = _checked_output_directory(Path(output_directory), create=False)
    manifest_path = _checked_member(directory, MANIFEST_FILENAME)
    manifest, _ = _load_canonical_mapping(manifest_path, "manifest")
    if manifest.get("manifest_schema") != MANIFEST_SCHEMA:
        raise ArtifactIntegrityError("unexpected manifest schema")

    stage_entry = manifest.get("stage_artifact")
    if not isinstance(stage_entry, dict):
        raise ArtifactIntegrityError("manifest is missing stage_artifact")
    for field in ("path", "sha256", "bytes"):
        if field not in stage_entry:
            raise ArtifactIntegrityError(f"stage_artifact is missing {field}")
    if stage_entry["path"] != STAGE_FILENAME:
        raise ArtifactIntegrityError("manifest names an unexpected stage artifact")
    stage_path = _checked_member(directory, stage_entry["path"])
    stage, stage_bytes = _load_canonical_mapping(stage_path, "stage artifact")

    expected_size = stage_entry["bytes"]
    if type(expected_size) is not int or expected_size < 1:
        raise ArtifactIntegrityError("stage artifact byte count must be a positive integer")
    if len(stage_bytes) != expected_size:
        raise ArtifactIntegrityError("stage artifact byte count mismatch")
    expected_digest = stage_entry["sha256"]
    if not isinstance(expected_digest, str) or len(expected_digest) != 64:
        raise ArtifactIntegrityError("stage artifact digest must be a SHA-256 hex string")
    if not hmac.compare_digest(_sha256(stage_bytes), expected_digest.lower()):
        raise ArtifactIntegrityError("stage artifact digest mismatch")

    duplicated_fields = (
        "protocol_id",
        "run_id",
        "status",
        "scientific_claim_status",
        "novelty_claim_status",
        "failure_count",
    )
    for field in duplicated_fields:
        if field not in manifest or field not in stage:
            raise ArtifactIntegrityError(f"missing duplicated field {field}")
        if manifest[field] != stage[field]:
            raise ArtifactIntegrityError(f"manifest/stage mismatch for {field}")
    if stage["protocol_id"] != PROTOCOL_ID:
        raise ArtifactIntegrityError("unexpected stage protocol")
    if type(stage["run_id"]) is not int or stage["run_id"] not in RUN_IDS:
        raise ArtifactIntegrityError("unexpected stage run_id")
    if stage["status"] not in (
        "FROZEN_FAILURES_RETAINED",
        "EXECUTION_CONSISTENT_ON_FROZEN_FIXTURES",
    ):
        raise ArtifactIntegrityError("unexpected stage status")
    if stage["scientific_claim_status"] != SCIENTIFIC_CLAIM_STATUS:
        raise ArtifactIntegrityError("unexpected scientific claim status")
    if stage["novelty_claim_status"] != NOVELTY_CLAIM_STATUS:
        raise ArtifactIntegrityError("unexpected novelty claim status")
    if manifest.get("resource_gates") != _resource_gates(stage):
        raise ArtifactIntegrityError("manifest/stage resource-gate mismatch")
    return manifest, stage


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", type=int, choices=RUN_IDS, required=True)
    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
        help="Dedicated directory for this run's stage_a.json and manifest.json",
    )
    args = parser.parse_args()
    manifest = write_run(args.output_directory, args.run_id)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
