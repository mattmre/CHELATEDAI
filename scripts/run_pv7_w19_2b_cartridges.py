"""CLI runner for Wave 19 / Bundle #57 / Item 2B: 250-payload / 1,000-catalog cartridge array.

``--mode local-drycheck`` (default, CPU-only, ``CUDA_VISIBLE_DEVICES=""``) materializes 250
real ``.cspg`` payload files inside a 1,000-slot catalog, times first usable inference
against the physical ``payload_bytes / measured_bw`` bound, and races concurrent EK5
revocations against queued and in-flight requests.

``--mode run-live`` is fail-closed: it requires an operator-supplied cgroup-enforced host
and is not wired on this runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chelated_cartridge_array import (  # noqa: E402
    BUNDLE_ID,
    MODES,
    run_cartridge_array_experiment,
)
from chelated_pv7_spine.claim_registry import claim_registry_sha256  # noqa: E402

MANIFEST_NAME = "pv7_w19_2b_manifest.json"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_receipt_bound_bundle(output_dir: Path, manifest: Dict[str, Any]) -> Tuple[Path, Path]:
    """Write the manifest and a SHA-256-bound ``report_receipt.json``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / MANIFEST_NAME
    manifest_bytes = (json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True, default=str) + "\n").encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)
    receipt = {
        "schema_version": "pv7_report_receipt_v1",
        "bundle_id": BUNDLE_ID,
        "item": "2B",
        "mode": manifest["mode"],
        "tier": manifest["tier"],
        "spine_digest": claim_registry_sha256(),
        "manifest_file": MANIFEST_NAME,
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "deterministic_sha256": manifest["deterministic_sha256"],
        "overall_pass": bool(manifest["overall_pass"]),
        "gates": manifest["gates"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
        "source_file_sha256": {
            "chelated_cartridge_array.py": _sha256_file(REPO_ROOT / "chelated_cartridge_array.py"),
            "scripts/run_pv7_w19_2b_cartridges.py": _sha256_file(Path(__file__).resolve()),
        },
    }
    receipt_path = output_dir / "report_receipt.json"
    receipt_path.write_text(json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    return manifest_path, receipt_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Wave 19 / Bundle #57 cartridge array + EK5 revocation.")
    parser.add_argument("--mode", choices=MODES, default="local-drycheck")
    parser.add_argument("--payload-bytes", type=int, default=256 * 1024)
    parser.add_argument("--materialized-count", type=int, default=250)
    parser.add_argument("--catalog-count", type=int, default=1000)
    parser.add_argument("--revocation-trials", type=int, default=300)
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--workdir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--json-summary", action="store_true")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    manifest = run_cartridge_array_experiment(
        mode=args.mode,
        workdir=args.workdir,
        payload_bytes=args.payload_bytes,
        materialized_count=args.materialized_count,
        catalog_count=args.catalog_count,
        revocation_trials=args.revocation_trials,
        threads=args.threads,
    )
    written = None
    if args.output_dir is not None:
        manifest_path, receipt_path = write_receipt_bound_bundle(args.output_dir, manifest)
        written = {"manifest": str(manifest_path), "receipt": str(receipt_path)}
    summary = {
        "bundle_id": BUNDLE_ID,
        "mode": manifest["mode"],
        "tier": manifest["tier"],
        "overall_pass": manifest["overall_pass"],
        "failed_gates": sorted(k for k, v in manifest["gates"].items() if not v),
        "deterministic_sha256": manifest["deterministic_sha256"],
        "written": written,
    }
    print(json.dumps(summary if args.json_summary else manifest, indent=2, sort_keys=True, default=str))
    return 0 if manifest["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
