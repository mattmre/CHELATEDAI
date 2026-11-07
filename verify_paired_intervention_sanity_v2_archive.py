"""Verify the exact retained PRW-ISI1 v2 archive without promoting it."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict

from paired_intervention_experiments import PairedInterventionValidationError


ARTIFACT_FILENAME = "paired_intervention_sanity.json"
MANIFEST_FILENAME = "manifest.json"
V2_PROTOCOL_PATH = Path(
    "docs/research/paired-chelation-intervention-sanity-protocol-v2-2026-08.md"
)
V2_PROTOCOL_SHA256 = "0e6938ec6b6b3f5ade0ea1e82cf38d60aece8850d3a2121339de5c28ce28bda1"
V2_ARTIFACT_SHA256 = "08e7afad2dfa0b6c2b190858500d0682bab74a597527f61623a1b56ed5f3f622"
V2_MANIFEST_SHA256 = "addd1158f815cd2c36834dcffe85b7d27e400a4e58e7b5271a248212a6f46085"
V2_ARTIFACT_DIGEST = "5bb3f8ca9bc51ddd61ce855199397beb7f6e476d34928c65cd367e8839baebc2"


def _sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _is_reparse(path: Path) -> bool:
    try:
        attributes = int(getattr(os.lstat(path), "st_file_attributes", 0))
    except (FileNotFoundError, OSError):
        return False
    return bool(attributes & 0x400)


def _ordinary(path: Path, *, directory: bool) -> bool:
    if path.is_symlink() or _is_reparse(path):
        return False
    return path.is_dir() if directory else path.is_file()


def _strict_canonical(raw: bytes, *, label: str) -> Any:
    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise PairedInterventionValidationError(
                    f"{label} contains duplicate key {key!r}"
                )
            value[key] = item
        return value

    try:
        payload = json.loads(raw.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PairedInterventionValidationError(f"{label} is not strict JSON") from exc
    canonical = (
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")
    if raw != canonical:
        raise PairedInterventionValidationError(f"{label} is not canonical JSON")
    return payload


def verify_v2_archive(result_directory: Path) -> Dict[str, object]:
    """Verify exact v2 custody bytes and internal digest cross-links."""

    root = Path(result_directory)
    if not _ordinary(root, directory=True):
        raise PairedInterventionValidationError("v2 archive root must be ordinary")
    resolved_root = root.resolve(strict=True)
    expected_names = sorted((ARTIFACT_FILENAME, MANIFEST_FILENAME))
    if sorted(path.name for path in root.iterdir()) != expected_names:
        raise PairedInterventionValidationError("v2 archive file set mismatch")
    artifact_path = root / ARTIFACT_FILENAME
    manifest_path = root / MANIFEST_FILENAME
    for path in (artifact_path, manifest_path):
        if not _ordinary(path, directory=False):
            raise PairedInterventionValidationError("v2 archive member must be ordinary")
        if path.resolve(strict=True).parent != resolved_root:
            raise PairedInterventionValidationError("v2 archive member escapes root")
    if not _ordinary(V2_PROTOCOL_PATH, directory=False):
        raise PairedInterventionValidationError("v2 protocol file unavailable")
    if _sha256(V2_PROTOCOL_PATH.read_bytes()) != V2_PROTOCOL_SHA256:
        raise PairedInterventionValidationError("v2 protocol custody mismatch")
    artifact_raw = artifact_path.read_bytes()
    manifest_raw = manifest_path.read_bytes()
    if _sha256(artifact_raw) != V2_ARTIFACT_SHA256:
        raise PairedInterventionValidationError("v2 artifact custody mismatch")
    if _sha256(manifest_raw) != V2_MANIFEST_SHA256:
        raise PairedInterventionValidationError("v2 manifest custody mismatch")
    artifact = _strict_canonical(artifact_raw, label="v2 artifact")
    manifest = _strict_canonical(manifest_raw, label="v2 manifest")
    payload = dict(artifact)
    retained_digest = payload.pop("artifact_digest", None)
    recomputed_digest = _sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode(
            "utf-8"
        )
    )
    if retained_digest != V2_ARTIFACT_DIGEST or retained_digest != recomputed_digest:
        raise PairedInterventionValidationError("v2 artifact digest mismatch")
    entry = manifest.get("entries")
    expected_entry = {
        "byte_count": len(artifact_raw),
        "path": ARTIFACT_FILENAME,
        "sha256": V2_ARTIFACT_SHA256,
    }
    if entry != [expected_entry] or manifest.get("artifact_digest") != retained_digest:
        raise PairedInterventionValidationError("v2 manifest cross-link mismatch")
    return {
        "status": "ARCHIVED_V2_VERIFIED",
        "scientific_claim_status": "UNCONFIRMED",
        "novelty_claim_status": "UNCONFIRMED",
        "protocol_sha256": V2_PROTOCOL_SHA256,
        "artifact_sha256": V2_ARTIFACT_SHA256,
        "manifest_sha256": V2_MANIFEST_SHA256,
        "artifact_digest": V2_ARTIFACT_DIGEST,
        "semantic_regeneration": False,
        "custody_only": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-v2-archive", required=True, type=Path)
    args = parser.parse_args()
    try:
        result = verify_v2_archive(args.verify_v2_archive)
    except (OSError, ValueError, PairedInterventionValidationError) as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}")
        return 1
    print(json.dumps(result, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
