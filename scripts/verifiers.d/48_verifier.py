
"""Offline replay verifier for Bundle 48's synthetic experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BUNDLE_ID = 48
SPINE_DIGEST = "32693f54e0ce7ad3d401ceb752ae4723b0eee251f6d49b95cdfb26a6201971e9"
DEPENDS_ON = ()

REPO_ROOT = Path(__file__).resolve().parents[2]
FRAGMENT_PATH = "scripts/verifiers.d/48_verifier.py"
REGISTRY_PATH = "docs/research/registry.d/48-t6a-engine-closure.json"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chelated_pv7_spine.fragment_compiler import compile_fragments  # noqa: E402
from scripts.run_pv7_w17_6a_engine_closure import execute  # noqa: E402


def verify_offline(receipt=None):
    sources = {path: (REPO_ROOT / path).read_bytes() for path in (REGISTRY_PATH, FRAGMENT_PATH)}
    compiled = compile_fragments(REPO_ROOT, SPINE_DIGEST, sources)
    report = execute("local-drycheck")
    if not report["passed"]:
        raise ValueError("Bundle 48 offline replay failed")
    if receipt is not None:
        if receipt["mode"] != "local-drycheck":
            raise ValueError("Offline receipt replay requires a local-drycheck receipt")
        # Latency is machine- and run-dependent, so compare the replayable fields.
        def replayable(source):
            return {
                **source, "engines": [
                    {**engine, "cells": [
                        {key: value for key, value in cell.items() if key not in {"ann_latency", "exact_latency"}}
                        for cell in engine["cells"]
                    ]}
                    for engine in source["engines"]
                ],
            }

        if replayable(receipt) != replayable(report):
            raise ValueError("Bundle 48 drycheck receipt differs from the offline replay")
    return {
        "bundle_id": BUNDLE_ID, "compiled_sha256": compiled["compiled_sha256"],
        "fixture_sha256": report["fixture"]["digest"],
        "canonical_trace_sha256": report["ek5"]["canonical_trace_sha256"],
        "status": "local-drycheck-pass",
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipt", type=Path, help="Optional local-drycheck receipt to replay")
    args = parser.parse_args(argv)
    try:
        receipt = json.loads(args.receipt.read_text(encoding="utf-8")) if args.receipt else None
        result = verify_offline(receipt)
    except (KeyError, OSError, TypeError, ValueError) as exc:
        parser.exit(1, f"{exc}\n")
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
