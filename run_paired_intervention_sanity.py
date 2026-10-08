"""Run and verify the bounded paired chelation intervention sanity v3 protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import uuid
from pathlib import Path
from typing import Any, Dict

from paired_intervention_experiments import (
    PAIRED_INTERVENTION_PROTOCOL_ID,
    PAIRED_INTERVENTION_PROTOCOL_PATH,
    PAIRED_INTERVENTION_PROTOCOL_SHA256,
    PAIRED_INTERVENTION_STAGE_ID,
    PAIRED_INTERVENTION_V1_PROTOCOL_SHA256,
    PAIRED_INTERVENTION_V2_PROTOCOL_SHA256,
    PairedInterventionValidationError,
    make_paired_intervention_artifact,
)


ARTIFACT_FILENAME = "paired_intervention_sanity.json"
MANIFEST_FILENAME = "manifest.json"


def _canonical_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _strict_json(raw: bytes, *, label: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise PairedInterventionValidationError(
                    f"{label} contains duplicate key {key!r}"
                )
            result[key] = value
        return result

    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PairedInterventionValidationError(
            f"{label} is not strict UTF-8 JSON: {exc}"
        ) from exc


def _write_atomic(path: Path, payload: Dict[str, object]) -> None:
    encoded = _canonical_bytes(payload)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        try:
            temporary_path.unlink()
        except FileNotFoundError:
            pass


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _sha256_file(path: Path) -> str:
    return _sha256_bytes(path.read_bytes())


def _is_reparse_point(path: Path) -> bool:
    """Reject Windows junctions and other reparse aliases on Python 3.11."""

    try:
        attributes = int(getattr(os.lstat(path), "st_file_attributes", 0))
    except (FileNotFoundError, OSError):
        return False
    return bool(attributes & 0x400)


def _is_link_like(path: Path) -> bool:
    return path.is_symlink() or _is_reparse_point(path)


def _file_record(path: Path) -> Dict[str, object]:
    content = path.read_bytes()
    return {
        "path": path.name,
        "byte_count": len(content),
        "sha256": _sha256_bytes(content),
    }


def _verify_protocol_files() -> None:
    protocol = Path(PAIRED_INTERVENTION_PROTOCOL_PATH)
    if not protocol.is_file() or _is_link_like(protocol):
        raise PairedInterventionValidationError("frozen v3 protocol file is unavailable")
    if _sha256_file(protocol) != PAIRED_INTERVENTION_PROTOCOL_SHA256:
        raise PairedInterventionValidationError("frozen v3 protocol digest mismatch")
    v1 = Path("docs/research/paired-chelation-intervention-sanity-protocol-2026-08.md")
    if not v1.is_file() or _is_link_like(v1):
        raise PairedInterventionValidationError("frozen v1 predecessor protocol is unavailable")
    if _sha256_file(v1) != PAIRED_INTERVENTION_V1_PROTOCOL_SHA256:
        raise PairedInterventionValidationError("frozen v1 predecessor protocol digest mismatch")
    v2 = Path("docs/research/paired-chelation-intervention-sanity-protocol-v2-2026-08.md")
    if not v2.is_file() or _is_link_like(v2):
        raise PairedInterventionValidationError("frozen v2 predecessor protocol is unavailable")
    if _sha256_file(v2) != PAIRED_INTERVENTION_V2_PROTOCOL_SHA256:
        raise PairedInterventionValidationError("frozen v2 predecessor protocol digest mismatch")


def _expected_manifest(artifact_payload: Dict[str, object], artifact_path: Path) -> Dict[str, object]:
    return {
        "protocol_id": PAIRED_INTERVENTION_PROTOCOL_ID,
        "protocol_file_sha256": PAIRED_INTERVENTION_PROTOCOL_SHA256,
        "v1_protocol_sha256": PAIRED_INTERVENTION_V1_PROTOCOL_SHA256,
        "v2_protocol_sha256": PAIRED_INTERVENTION_V2_PROTOCOL_SHA256,
        "stage_id": PAIRED_INTERVENTION_STAGE_ID,
        "status": "VALIDATED_SYNTHETIC_SANITY_ONLY",
        "execution_mode": "bounded_cpu_synthetic_sanity",
        "production_path_changed": False,
        "model_or_corpus_loaded": False,
        "evidence_state": artifact_payload["evidence_state"],
        "scientific_claim_status": "UNCONFIRMED",
        "novelty_claim_status": "UNCONFIRMED",
        "artifact_digest": artifact_payload["artifact_digest"],
        "entries": [_file_record(artifact_path)],
    }


def _verify_result_dir(result_dir: Path, *, allow_staged: bool) -> Dict[str, object]:
    root = Path(result_dir)
    if _is_link_like(root) or not root.is_dir():
        raise PairedInterventionValidationError("result path must be an ordinary directory")
    if root.name.startswith(".") and ".stage-" in root.name and not allow_staged:
        raise PairedInterventionValidationError("staging directories are not public evidence")
    resolved_root = root.resolve(strict=True)
    names = sorted(path.name for path in root.iterdir())
    expected_names = sorted((ARTIFACT_FILENAME, MANIFEST_FILENAME))
    if names != expected_names:
        raise PairedInterventionValidationError(
            f"result file set differs from frozen v3 set: {names}"
        )
    artifact_path = root / ARTIFACT_FILENAME
    manifest_path = root / MANIFEST_FILENAME
    for path in (artifact_path, manifest_path):
        if _is_link_like(path) or not path.is_file():
            raise PairedInterventionValidationError("result members must be ordinary files")
        resolved = path.resolve(strict=True)
        if resolved.parent != resolved_root:
            raise PairedInterventionValidationError("result member escapes its resolved root")
    _verify_protocol_files()
    artifact_raw = artifact_path.read_bytes()
    manifest_raw = manifest_path.read_bytes()
    artifact = _strict_json(artifact_raw, label="artifact")
    manifest = _strict_json(manifest_raw, label="manifest")
    if artifact_raw != _canonical_bytes(artifact):
        raise PairedInterventionValidationError("artifact bytes are not canonical v3 JSON")
    if manifest_raw != _canonical_bytes(manifest):
        raise PairedInterventionValidationError("manifest bytes are not canonical v3 JSON")
    expected_artifact = make_paired_intervention_artifact().as_dict()
    if artifact != expected_artifact:
        raise PairedInterventionValidationError(
            "artifact differs from independently regenerated frozen v3 evidence"
        )
    payload_without_digest = dict(artifact)
    retained_digest = payload_without_digest.pop("artifact_digest", None)
    recomputed_digest = _sha256_bytes(
        json.dumps(
            payload_without_digest,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )
    if retained_digest != recomputed_digest:
        raise PairedInterventionValidationError("artifact digest mismatch")
    expected_manifest = _expected_manifest(artifact, artifact_path)
    if manifest != expected_manifest:
        raise PairedInterventionValidationError("manifest differs from frozen v3 evidence")
    return manifest


def verify_result_dir(result_dir: Path) -> Dict[str, object]:
    """Verify one publicly promoted v3 result directory."""

    return _verify_result_dir(Path(result_dir), allow_staged=False)


def _promote_directory_noreplace(stage_dir: Path, final_dir: Path) -> None:
    if os.name == "posix":
        import ctypes
        import errno

        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise PairedInterventionValidationError(
                "renameat2(RENAME_NOREPLACE) is required for fail-closed publication"
            )
        renameat2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        renameat2.restype = ctypes.c_int
        result = renameat2(-100, os.fsencode(stage_dir), -100, os.fsencode(final_dir), 1)
        if result != 0:
            error = ctypes.get_errno()
            if error in {errno.EEXIST, errno.ENOTEMPTY}:
                raise FileExistsError(str(final_dir))
            raise OSError(error, os.strerror(error), str(final_dir))
        return
    os.rename(stage_dir, final_dir)


def run(output_directory: Path) -> Dict[str, object]:
    output_directory = Path(output_directory)
    if _is_link_like(output_directory) or output_directory.exists():
        raise FileExistsError(f"output path already exists: {output_directory}")
    parent = output_directory.parent.resolve()
    parent.mkdir(parents=True, exist_ok=True)
    if output_directory.exists() or _is_link_like(output_directory):
        raise FileExistsError(f"output path appeared before staging: {output_directory}")
    final = parent / output_directory.name
    stage = parent / f".{output_directory.name}.stage-{uuid.uuid4().hex}"
    stage.mkdir()
    committed = False
    try:
        artifact = make_paired_intervention_artifact()
        artifact_path = artifact.write_json(stage / ARTIFACT_FILENAME)
        manifest = _expected_manifest(artifact.as_dict(), artifact_path)
        _write_atomic(stage / MANIFEST_FILENAME, manifest)
        _verify_result_dir(stage, allow_staged=True)
        _promote_directory_noreplace(stage, final)
        committed = True
        return manifest
    finally:
        if not committed:
            shutil.rmtree(stage, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--output-directory", type=Path)
    target.add_argument("--verify-result-dir", type=Path)
    args = parser.parse_args()
    try:
        if args.verify_result_dir is not None:
            result = verify_result_dir(args.verify_result_dir)
        else:
            result = run(args.output_directory)
    except (FileExistsError, OSError, PairedInterventionValidationError, ValueError) as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}")
        return 1
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
