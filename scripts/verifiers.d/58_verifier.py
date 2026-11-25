"""Compile and replay Bundle 58's deterministic file-backed withdrawal traces."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

BUNDLE_ID = 58
SPINE_DIGEST = "32693f54e0ce7ad3d401ceb752ae4723b0eee251f6d49b95cdfb26a6201971e9"
DEPENDS_ON = ()

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "artifacts" / "pv7_w19_2c_cold_osd_9m" / "pv7_w19_2c_drycheck_manifest.json"
FRAGMENTS = ("docs/research/registry.d/58-t2c-cold-osd-9m.json", "scripts/verifiers.d/58_verifier.py")
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chelated_cold_osd_scanner import drycheck  # noqa: E402
from chelated_pv7_spine.fragment_compiler import compile_fragments  # noqa: E402


def verify_offline(receipt=None):
    compiled = compile_fragments(ROOT, SPINE_DIGEST, {name: (ROOT / name).read_bytes() for name in FRAGMENTS})
    with tempfile.TemporaryDirectory(prefix=".pv7-58-", dir=ROOT) as directory:
        report = drycheck(directory)
    if receipt is None and MANIFEST.is_file():
        # The committed drycheck manifest is a receipt too: replay must reproduce it exactly.
        receipt = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if not report["passed"] or (receipt is not None and receipt != report):
        raise ValueError("Bundle 58 drycheck failed or receipt differs from replay")
    # The gate is the clustered layout under the shared journal codec and the challenger's no-durability floor.
    gate = report["gate"]
    if not gate["passed"] or report["bundle_outcome"] not in {"verified-win", "verified-mixed"}:
        raise ValueError("Bundle 58 clustered-layout gate did not pass")
    rows = report["codec_symmetry"]["rows"]
    scattered = report["scattered_layout_disclosure"]["per_codec_row"]
    return {
        "bundle_id": BUNDLE_ID, "status": report["bundle_outcome"], "evidence_tier": "local-drycheck-floor",
        "compiled_sha256": compiled["compiled_sha256"], "fixture_sha256": report["immutable_extent_sha256"],
        "clustered_write_bytes_saved": {name: row["clustered"]["write_bytes_saved"] for name, row in rows.items()},
        "scattered_write_bytes_saved": {name: cell["write_bytes_saved"] for name, cell in scattered.items()},
        "scattered_gated": report["scattered_layout_disclosure"]["gated"],
        "sector_quantization_unit": report["sector_quantization_disclosure"]["unit"],
        "live_measurement": False,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path)
    args = parser.parse_args(argv)
    try:
        receipt = json.loads(args.receipt.read_text(encoding="utf-8")) if args.receipt else None
        print(json.dumps(verify_offline(receipt), sort_keys=True))
    except (KeyError, OSError, TypeError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
