"""Run the bounded paired chelation intervention sanity protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Dict

from paired_intervention_experiments import make_paired_intervention_artifact


def _canonical_bytes(payload: Dict[str, object]) -> bytes:
    return (json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode(
        "utf-8"
    )


def _write_atomic(path: Path, payload: Dict[str, object]) -> None:
    encoded = _canonical_bytes(payload)
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
            handle.write(encoded)
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


def _file_record(path: Path) -> Dict[str, object]:
    content = path.read_bytes()
    return {
        "path": path.name,
        "byte_count": len(content),
        "sha256": hashlib.sha256(content).hexdigest(),
    }


def run(output_directory: Path) -> Dict[str, object]:
    output_directory.mkdir(parents=True, exist_ok=True)
    artifact = make_paired_intervention_artifact()
    artifact_path = artifact.write_json(output_directory / "paired_intervention_sanity.json")
    artifact_record = _file_record(artifact_path)
    manifest = {
        "protocol_id": artifact.protocol_id,
        "stage_id": artifact.stage_id,
        "execution_mode": "bounded_cpu_synthetic_sanity",
        "production_path_changed": False,
        "model_or_corpus_loaded": False,
        "evidence_state": artifact.evidence_state,
        "scientific_claim_status": artifact.scientific_claim_status,
        "artifact_digest": artifact.artifact_digest,
        "entries": [artifact_record],
    }
    _write_atomic(output_directory / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("artifacts/method-dev/isi1-paired-intervention-sanity"),
    )
    args = parser.parse_args()
    manifest = run(args.output_directory)
    print(json.dumps(manifest, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
