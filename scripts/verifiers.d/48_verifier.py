
"""Offline replay verifier for Bundle 48's synthetic experiment."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

BUNDLE_ID = 48
SPINE_DIGEST = "32693f54e0ce7ad3d401ceb752ae4723b0eee251f6d49b95cdfb26a6201971e9"
DEPENDS_ON = ()

REPO_ROOT = Path(__file__).resolve().parents[2]
FRAGMENT_PATH = "scripts/verifiers.d/48_verifier.py"
REGISTRY_PATH = "docs/research/registry.d/48-t6a-engine-closure.json"
LIVE_RECEIPT_REL = "artifacts/pv7_w17_6a_engine_closure_live/48-t6a-engine-closure.json"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chelated_pv7_spine.fragment_compiler import compile_fragments  # noqa: E402
from chelated_pv7_spine import _json_sha256  # noqa: E402
from scripts import run_pv7_w17_6a_engine_closure as runner  # noqa: E402
from scripts.run_pv7_w17_6a_engine_closure import execute  # noqa: E402


def verify_live_receipt(path, drycheck_report):
    """Structurally and cryptographically verify a committed ``run-live`` receipt.

    No server is contacted. The fixture digest is recomputed from the deterministic synthetic fixture, the EK5
    parity block is compared with an offline replay, and every cell's pass flag is recomputed from its own
    counters. Latency/recall values come from the recorded run and are only range-checked, never reproduced.
    """
    receipt = json.loads(path.read_text(encoding="utf-8"))
    problems = []

    def require(ok, message):
        if not ok:
            problems.append(message)

    require(receipt.get("bundle_id") == BUNDLE_ID and receipt.get("schema") == 1, "bundle_id/schema mismatch")
    require(receipt.get("mode") == "run-live", "receipt mode is not run-live")
    require(receipt.get("spine_digest") == SPINE_DIGEST, "spine_digest mismatch")
    require(receipt.get("spine_commit") == runner.SPINE_COMMIT, "spine_commit mismatch")
    require(receipt.get("fixture_origin") == runner.FIXTURE_ORIGIN and receipt.get("is_wild_corpus") is False,
            "synthetic/non-wild stamp missing")
    fixture = receipt.get("fixture", {})
    shape = (runner.LIVE_POPULATION, runner.LIVE_DIMENSION, runner.LIVE_REQUESTS)
    require((fixture.get("population"), fixture.get("dimension"), fixture.get("queries")) == shape,
            "live fixture shape differs from the preregistered 12000x128 / 10000 queries")
    require(fixture.get("seed") == runner.FIXTURE_SEED and fixture.get("synthetic") is True, "fixture seed/synthetic")
    vectors, queries, epochs, cells = runner.build_fixture(*shape)
    recomputed = _json_sha256({
        "vectors": hashlib.sha256(vectors.astype("<f4", copy=False).tobytes()).hexdigest(),
        "queries": hashlib.sha256(queries.astype("<f4", copy=False).tobytes()).hexdigest(),
        "epochs": epochs.tolist(), "counts": {density: cell.admitted_count for density, cell in cells.items()},
    })
    require(fixture.get("digest") == recomputed, "fixture digest does not match the recomputed deterministic fixture")
    require(receipt.get("ek5") == drycheck_report["ek5"], "ek5 parity block differs from the offline replay")
    require(receipt.get("ek5", {}).get("parity_passed") is True, "ek5 parity did not pass")
    engines = receipt.get("engines", [])
    require([e.get("engine") for e in engines] == ["qdrant", "opensearch", "pgvector"], "engine set/order mismatch")
    cells_passed = True
    for engine in engines:
        name = engine.get("engine")
        require("error" not in engine and bool(engine.get("server")), f"{name}: error recorded or no server identity")
        rows = engine.get("cells", [])
        require([c.get("retraction_percent") for c in rows] == list(runner.RETRACTION_PERCENTAGES),
                f"{name}: retraction cells mismatch")
        for cell in rows:
            density = cell.get("retraction_percent")
            tag = f"{name}@{density}%"
            admission = cells.get(density)
            require(admission is not None and cell.get("admitted_epoch") == admission.epoch
                    and cell.get("admitted_points") == admission.admitted_count, f"{tag}: admission snapshot mismatch")
            require(cell.get("paired_requests") == runner.LIVE_REQUESTS, f"{tag}: paired_requests")
            counters = ("timeouts", "exact_timeouts", "short_responses", "exact_short_responses",
                        "invalid_hits", "invalid_exact_hits")
            require(all(cell.get(key) == 0 for key in counters), f"{tag}: non-zero failure counters")
            require(cell.get("recall_at_10", 0) >= 1 - runner.RECALL_GAP_LIMIT, f"{tag}: recall below preregistered gap")
            require(cell.get("ann_latency", {}).get("p95_ms", 1e9) < runner.P95_LIMIT_MS, f"{tag}: p95 over limit")
            require(cell.get("route_evidence_passed") is True and cell.get("telemetry_passed") is True, f"{tag}: route")
            if name == "qdrant":
                route = cell.get("qdrant_measured_route", {})
                require(route.get("route") == "filtered_hnsw" and route.get("exact_fallback_triggered") is False,
                        f"{tag}: qdrant measured route is not filtered_hnsw / fallback seen")
            recomputed_pass = (
                not any(cell.get(k) for k in ("exact_timeouts", "exact_short_responses", "invalid_hits", "invalid_exact_hits"))
                and cell.get("telemetry_passed") is True and cell.get("performance_passed") is True
            )
            require(cell.get("passed") is True and recomputed_pass, f"{tag}: stored pass flag != recomputed")
            cells_passed = cells_passed and recomputed_pass
    require(receipt.get("passed") is True and cells_passed, "top-level passed flag is not supported by the cells")
    if problems:
        raise ValueError("Bundle 48 live receipt verification failed: " + "; ".join(problems))
    return {
        "status": "live-receipt-verified", "path": LIVE_RECEIPT_REL,
        "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "fixture_sha256": recomputed, "engines": [e["engine"] for e in engines],
        "servers": {e["engine"]: e["server"] for e in engines},
    }


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
    live_path = REPO_ROOT / LIVE_RECEIPT_REL
    live = verify_live_receipt(live_path, report) if live_path.is_file() else {"status": "live-receipt-absent"}
    return {
        "bundle_id": BUNDLE_ID, "compiled_sha256": compiled["compiled_sha256"],
        "fixture_sha256": report["fixture"]["digest"],
        "canonical_trace_sha256": report["ek5"]["canonical_trace_sha256"],
        "status": "local-drycheck-pass", "live_receipt": live,
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
