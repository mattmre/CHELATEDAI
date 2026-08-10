"""Run the bounded RB-14 synthetic observability suites.

The command intentionally performs no model, corpus, GPU, or production-path
work.  Each stage writes an atomic JSON envelope whose digest is retained in a
small manifest for later evidence reconciliation.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Dict

from observability_experiments import (
    ObservabilityArtifact,
    make_coa1_artifact,
    make_coalition_rag_artifact,
    make_ctx1_artifact,
    make_obs1_artifact,
    make_transport_artifact,
)


def _write_manifest(path: Path, payload: Dict[str, object]) -> None:
    encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode(
        "utf-8"
    )
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


def run(output_directory: Path, seed: int) -> Dict[str, object]:
    output_directory.mkdir(parents=True, exist_ok=True)
    artifacts = (
        ("obs1", make_obs1_artifact(seed=seed)),
        ("coa1", make_coa1_artifact(seed=seed)),
        ("ctx1", make_ctx1_artifact(seed=seed)),
        ("transport", make_transport_artifact(seed=seed)),
        ("coalition_rag", make_coalition_rag_artifact(seed=seed)),
    )
    entries = []
    for name, artifact in artifacts:
        path = artifact.write_json(output_directory / f"{name}.json")
        entries.append(
            {
                "name": name,
                "stage_id": artifact.stage_id,
                "path": str(path),
                "artifact_digest": artifact.artifact_digest,
                "status": artifact.status,
                "scientific_claim_status": artifact.result["scientific_claim_status"],
            }
        )
    manifest = {
        "protocol_id": artifacts[0][1].protocol_id,
        "seed": seed,
        "execution_mode": "bounded_cpu_synthetic_sanity",
        "production_path_changed": False,
        "model_or_corpus_loaded": False,
        "entries": entries,
    }
    _write_manifest(output_directory / "manifest.json", manifest)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-directory",
        type=Path,
        default=Path("artifacts/method-dev/rb14-observability"),
    )
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    manifest = run(args.output_directory, args.seed)
    print(json.dumps(manifest, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
