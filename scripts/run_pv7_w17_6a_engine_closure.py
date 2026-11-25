
"""Run Bundle 48's CPU-only drycheck or its paired live-engine benchmark."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import Counter
from contextlib import ExitStack
from pathlib import Path
from uuid import uuid4

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import httpx  # noqa: E402
import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chelated_pv7_spine import _json_sha256  # noqa: E402
from chelated_pv7_spine.bridge_merge_harness import l2_normalize  # noqa: E402
from chelated_pv7_spine.burn_ledger import HISTORICAL_DATASETS, normalize_dataset_id  # noqa: E402
from chelated_pv7_spine.challenger_ledger import (  # noqa: E402
    ChallengerBitemporalLedger,
    LedgerRecordEnvelope,
    OpKind,
)
from chelated_pv7_spine.ek5_reason_canonicalizer import (  # noqa: E402
    compare_with_fresh_rebuild,
    export_canonical_ek5_audit_trace,
)
from chelated_pv7_spine.hnsw_prefilter_pushdown import (  # noqa: E402
    TOP_K,
    Admission,
    OpenSearchPrefilter,
    PgvectorPrefilter,
    QdrantPrefilter,
    SimulatedHnsw,
    qdrant_route_evidence,
)
from ek5_v3_stratified import EdgeAttestation, StratifiedEvent, StratifiedEvidenceStore  # noqa: E402

BUNDLE_ID = 48
SPINE_COMMIT = "d4454e2d9fca146ccdc3d18db22ac6c0b5d3e802"
SPINE_DIGEST = "32693f54e0ce7ad3d401ceb752ae4723b0eee251f6d49b95cdfb26a6201971e9"
RECEIPT_NAME = "48-t6a-engine-closure.json"
FIXTURE_ID = "pv7-b48-synthetic-gaussian-v1"
FIXTURE_SEED = 480017
# The fixture is a synthetic Gaussian route probe, never a wild corpus; stamped at the
# receipt top level in both modes so it cannot be read as wild-corpus evidence.
FIXTURE_ORIGIN = "synthetic_engine_route_probe"
IS_WILD_CORPUS = False
RETRACTION_PERCENTAGES = (0, 10, 50, 90)
LIVE_REQUESTS = 10_000
LIVE_POPULATION = 12_000
LIVE_DIMENSION = 128
DRY_REQUESTS = 24
DRY_POPULATION = 240
DRY_DIMENSION = 32
P95_LIMIT_MS = 35.0
RECALL_GAP_LIMIT = 0.01
SEQUENCES = 1000
MUTATIONS = 12
EDGE_TYPES = (
    "DERIVES_SNAPSHOT", "DERIVES_CURRENT", "COPY_OF", "REFUTE",
    "SUPERSEDE", "VERSION_LINEAGE_SUPERSEDE", "AMENDS", "CITES_AUDIT",
)


def build_fixture(population, dimension, request_count):
    if normalize_dataset_id(FIXTURE_ID) in HISTORICAL_DATASETS:
        raise RuntimeError("Synthetic fixture ID collides with a burned dataset")
    rng = np.random.default_rng(FIXTURE_SEED)
    vectors = l2_normalize(rng.standard_normal((population, dimension))).astype(np.float32)
    queries = l2_normalize(rng.standard_normal((request_count, dimension))).astype(np.float32)
    order = rng.permutation(population)
    epochs = np.empty(population, dtype=np.int32)
    previous = 0
    # Cohorts enter the synthetic visibility window at increasing policy epochs.
    with ChallengerBitemporalLedger() as ledger:
        for epoch, density in enumerate(reversed(RETRACTION_PERCENTAGES), 1):
            cutoff = population * (100 - density) // 100
            cohort = f"cohort-{epoch}"
            ledger.append(LedgerRecordEnvelope(
                tenant_id=FIXTURE_ID, record_id=cohort, entity_key=cohort,
                authority_rank=1, valid_from=0, valid_to=None, tx_time=1,
                op_kind=OpKind.ASSERT, policy_epoch=epoch, claim_value=str(cutoff - previous),
            ))
            epochs[order[previous:cutoff]] = epoch
            previous = cutoff
        cells = {}
        for epoch, density in enumerate(reversed(RETRACTION_PERCENTAGES), 1):
            count = 0
            for cohort_epoch in range(1, len(RETRACTION_PERCENTAGES) + 1):
                view = ledger.query_as_of(FIXTURE_ID, f"cohort-{cohort_epoch}", 0, 1, policy_epoch=epoch)
                if view.status == "RESOLVED":
                    count += int(view.winner_claim_value)
            if count != np.count_nonzero(epochs <= epoch):
                raise RuntimeError("Synthetic admission snapshot differs from the challenger ledger")
            cells[density] = Admission(epoch, count, population)
    return vectors, queries, epochs, cells


def _mutation_steps(seed):
    rng = random.Random(seed)
    strict = seed % 2 == 0
    store = StratifiedEvidenceStore(strict_dag_supersede=strict)
    active = []
    for step in range(MUTATIONS):
        events = []
        if step < 8:
            events.append(StratifiedEvent(
                event_id=f"e{step}", entity="entity", slot="slot",
                claim_value=rng.choice(("yes", "no")), authority_tier=rng.randrange(4),
                tenant="synthetic", domain=rng.choice(("legal", "medical", "general")),
            ))
        event_ids = [*store.events, *(event.event_id for event in events)]
        edits = []
        if len(event_ids) >= 2:
            for index in range(rng.randint(1, 3)):
                source, target = rng.sample(event_ids, 2)
                kind = rng.choice(EDGE_TYPES)
                if strict and kind in {"SUPERSEDE", "VERSION_LINEAGE_SUPERSEDE"}:
                    source, target = sorted((source, target), reverse=True)
                edits.append(EdgeAttestation(
                    f"a{step}-{index}", source, target, kind,
                    lineage_id="lineage" if kind == "VERSION_LINEAGE_SUPERSEDE" else None,
                ))
        removed = []
        if active and (step >= 8 or rng.random() < 0.25):
            removed.append(active.pop(rng.randrange(len(active))))
        active.extend(edge.attestation_id for edge in edits)
        store.apply_mutation(events, edits, removed)
        yield store


def run_audit_parity():
    counts = Counter(semantic_diffs=0, reason_export_diffs=0, pos_cyc_export_diffs=0)
    causes = Counter()
    first_diffs = []
    digest = hashlib.sha256()
    node_count = 0
    step_count = 0
    for seed in range(SEQUENCES):
        for step, store in enumerate(_mutation_steps(seed), 1):
            trace = export_canonical_ek5_audit_trace(store)
            comparison = compare_with_fresh_rebuild(store, trace)
            semantic = comparison.semantic_diffs + comparison.canonical_semantic_mismatches
            reason = comparison.canonical_reason_diffs
            pos_cyc = comparison.canonical_pos_cyc_diffs
            counts.update(semantic_diffs=len(semantic), reason_export_diffs=len(reason),
                          pos_cyc_export_diffs=len(pos_cyc))
            if (semantic or reason or pos_cyc) and len(first_diffs) < 10:
                first_diffs.append({
                    "seed": seed, "step": step, "semantic": semantic,
                    "reason": reason, "pos_cyc": pos_cyc,
                })
            causes.update(node.cause for node in trace.nodes)
            digest.update(bytes.fromhex(trace.digest()))
            node_count += comparison.node_count
            step_count += 1
    return {
        "sequences": SEQUENCES, "steps_per_sequence": MUTATIONS, "snapshots_compared": step_count,
        "nodes_compared": node_count, **counts, "canonical_cause_counts": dict(sorted(causes.items())),
        "canonical_trace_sha256": digest.hexdigest(), "first_diffs": first_diffs,
        "parity_passed": step_count == SEQUENCES * MUTATIONS and not any(counts.values()),
    }


def _timed_search(backend, query, admission, *, exact):
    start = time.perf_counter()
    try:
        result = backend.search(query, admission, exact=exact)
    except (TimeoutError, httpx.TimeoutException):
        result = None
    except Exception as exc:
        # PostgreSQL statement_timeout cancels the statement with SQLSTATE 57014.
        if getattr(exc, "sqlstate", None) != "57014":
            raise
        result = None
    return result, (time.perf_counter() - start) * 1000


def _valid_hits(result, valid_ids, limit):
    if result is None:
        return (), 0
    valid = tuple(point for point in dict.fromkeys(result.ids) if point in valid_ids)
    invalid = len(result.ids) - len(valid) + max(0, len(valid) - limit)
    return valid[:limit], invalid


def _percentiles(samples):
    ordered = sorted(samples)
    return {f"p{p}_ms": ordered[math.ceil(p / 100 * len(ordered)) - 1] for p in (50, 95, 99)}


def _profiles(backend, queries, admission):
    samples = []
    for i in sorted({0, len(queries) // 2, len(queries) - 1}):
        try:
            evidence = backend.profile(queries[i], admission)
        except (TimeoutError, httpx.TimeoutException):
            evidence = {"route": "unknown", "exact_fallback_triggered": None, "timed_out": True}
        except httpx.HTTPStatusError as exc:
            evidence = {"route": "unknown", "exact_fallback_triggered": None, "http_status": exc.response.status_code}
        samples.append({"query_index": i, **evidence})
    return samples


def run_cell(backend, queries, epochs, density, admission, *, live):
    valid = set(map(int, np.flatnonzero(epochs <= admission.epoch)))
    expected = min(TOP_K, admission.admitted_count)
    if expected != TOP_K or len(valid) != admission.admitted_count:
        raise ValueError("Fixture cell must have at least ten verified admitted points")
    qdrant = isinstance(backend, QdrantPrefilter)
    before = backend.route_snapshot() if qdrant else None
    ann_ms, exact_ms, recalls, visits = [], [], [], []
    counters = Counter(
        timeouts=0, exact_timeouts=0, short_responses=0, exact_short_responses=0,
        invalid_hits=0, invalid_exact_hits=0,
    )
    for index, query in enumerate(queries):
        # Counterbalance order while preserving a one-to-one pair for every query.
        arms = {}
        for exact in (False, True) if index % 2 == 0 else (True, False):
            arms[exact] = _timed_search(backend, query, admission, exact=exact)
        prediction, elapsed = arms[False]
        oracle, exact_elapsed = arms[True]
        ann_ms.append(elapsed)
        exact_ms.append(exact_elapsed)
        ann_ids, invalid_ann = _valid_hits(prediction, valid, expected)
        exact_ids, invalid_exact = _valid_hits(oracle, valid, expected)
        counters.update(
            timeouts=int(prediction is None), exact_timeouts=int(oracle is None),
            short_responses=int(prediction is not None and len(ann_ids) < expected),
            exact_short_responses=int(oracle is not None and len(exact_ids) < expected),
            invalid_hits=invalid_ann, invalid_exact_hits=invalid_exact,
        )
        # Missing or short responses stay in the denominator.
        recalls.append(len(set(ann_ids).intersection(exact_ids)))
        if prediction is not None and prediction.visited_graph_candidates is not None:
            visits.append(prediction.visited_graph_candidates)
    qdrant_evidence = (
        qdrant_route_evidence(before, backend.route_snapshot(), len(queries), exact_controls=len(queries))
        if qdrant else None
    )
    profiles = _profiles(backend, queries, admission)
    fallback_flags = [profile["exact_fallback_triggered"] for profile in profiles]
    fallback_count = sum(fallback_flags) if all(flag is not None for flag in fallback_flags) else None
    candidate_latency = _percentiles(ann_ms)
    recall = sum(recalls) / (expected * len(queries))
    invalid = (
        counters["exact_timeouts"] + counters["exact_short_responses"]
        + counters["invalid_hits"] + counters["invalid_exact_hits"]
    )
    route_passed = all(flag is False for flag in fallback_flags)
    if qdrant:
        route_passed = route_passed and qdrant_evidence["exact_fallback_triggered"] is False
    if isinstance(backend, PgvectorPrefilter):
        route_passed = route_passed and all(profile["route"] == "iterative_hnsw" for profile in profiles)
    per_query_visits = len(visits) == len(queries)
    graph_telemetry_basis = "per_query_visit_counts" if per_query_visits else None
    if qdrant and not per_query_visits:
        # Stock Qdrant (e.g. 1.20.2) /telemetry exposes the filtered-HNSW route and the
        # exact-fallback counters but no per-query visited_graph_candidates. The measured
        # route/fallback delta over every paired request plus every plan probe is the
        # graph-telemetry evidence there; visit counts are recorded as unavailable.
        stock_route_verified = (
            qdrant_evidence["route"] == "filtered_hnsw"
            and qdrant_evidence["exact_fallback_triggered"] is False
            and all(profile.get("route") == "filtered_hnsw" and profile["exact_fallback_triggered"] is False
                    for profile in profiles)
        )
        graph_telemetry_basis = "qdrant_route_and_fallback_counters" if stock_route_verified else None
    if qdrant or not live:
        graph_telemetry_complete = graph_telemetry_basis is not None
    else:
        graph_telemetry_complete = None
    telemetry_passed = route_passed and graph_telemetry_complete is not False
    recall_passed = sum(recalls) >= math.ceil((1 - RECALL_GAP_LIMIT) * expected * len(queries))
    performance_passed = candidate_latency["p95_ms"] < P95_LIMIT_MS and recall_passed if live else None
    return {
        "retraction_percent": density, "admitted_epoch": admission.epoch,
        "admitted_points": admission.admitted_count, "hnsw_ef": admission.ef,
        "paired_requests": len(queries), "ann_latency": candidate_latency,
        "exact_latency": _percentiles(exact_ms), "recall_at_10": recall,
        "max_recall_gap": RECALL_GAP_LIMIT, **counters,
        "plan_probes": profiles, "probed_exact_fallback_count": fallback_count,
        "route_evidence_passed": route_passed, "qdrant_measured_route": qdrant_evidence,
        "graph_candidate_sample_count": len(visits), "graph_telemetry_complete": graph_telemetry_complete,
        "graph_telemetry_basis": graph_telemetry_basis, "per_query_visit_counts_available": bool(visits),
        "visited_graph_candidates": sum(visits) if visits else None,
        "telemetry_passed": telemetry_passed, "performance_passed": performance_passed,
        "passed": not invalid and telemetry_passed and (performance_passed if live else counters["timeouts"] == 0),
    }


def _run_live(fixture, args):
    import psycopg

    vectors, queries, epochs, cells = fixture
    name = "pv7_b48_" + uuid4().hex[:16]
    results = []
    for engine, factory in (
        ("qdrant", lambda: QdrantPrefilter(name, args.qdrant_url, api_key=os.environ.get("QDRANT_API_KEY"))),
        ("opensearch", lambda: OpenSearchPrefilter(name, vectors.shape[1], args.opensearch_url)),
        ("pgvector", lambda: PgvectorPrefilter(name, args.pg_dsn)),
    ):
        result = {"engine": engine, "server": None, "cells": []}
        stage = "connect"
        try:
            with ExitStack() as cleanup:
                backend = factory()
                cleanup.callback(backend.close)
                stage = "server_version"
                result["server"] = backend.server_info()
                stage = "create_fixture"
                if engine == "opensearch":
                    backend.create()
                    cleanup.callback(backend.drop)
                    backend.ingest_fixture(vectors, epochs)
                else:
                    backend.create(vectors.shape[1])
                    cleanup.callback(backend.drop)
                    backend.ingest(vectors, epochs)
                for density in RETRACTION_PERCENTAGES:
                    stage = f"measure_{density}_percent_retraction"
                    result["cells"].append(run_cell(backend, queries, epochs, density, cells[density], live=True))
                stage = "cleanup"
        except (OSError, RuntimeError, httpx.HTTPError, psycopg.Error) as exc:
            # Transport error strings can contain credentials or remote addresses.
            result["error"] = {"stage": stage, "type": type(exc).__name__}
            if isinstance(exc, httpx.HTTPStatusError):
                result["error"]["http_status"] = exc.response.status_code
            elif isinstance(exc, psycopg.Error):
                result["error"]["sqlstate"] = exc.sqlstate
        results.append(result)
    return results


def execute(mode, args=None):
    live = mode == "run-live"
    if not live and mode != "local-drycheck":
        raise ValueError(f"Unknown run mode {mode!r}")
    shape = (LIVE_POPULATION, LIVE_DIMENSION, LIVE_REQUESTS) if live else (DRY_POPULATION, DRY_DIMENSION, DRY_REQUESTS)
    fixture = build_fixture(*shape)
    vectors, queries, epochs, cells = fixture
    audit = run_audit_parity()
    if live:
        engines = _run_live(fixture, args)
    else:
        simulator = SimulatedHnsw(vectors, epochs)
        engines = [{
            "engine": simulator.name, "server": None,
            "cells": [run_cell(simulator, queries, epochs, density, cells[density], live=False)
                      for density in RETRACTION_PERCENTAGES],
        }]
    return {
        "schema": 1, "bundle_id": BUNDLE_ID, "mode": mode,
        "fixture_origin": FIXTURE_ORIGIN, "is_wild_corpus": IS_WILD_CORPUS,
        "spine_commit": SPINE_COMMIT, "spine_digest": SPINE_DIGEST,
        "fixture": {
            "id": FIXTURE_ID, "population": shape[0], "dimension": shape[1], "seed": FIXTURE_SEED,
            "queries": len(queries), "synthetic": True, "origin": FIXTURE_ORIGIN,
            "is_wild_corpus": IS_WILD_CORPUS, "digest": _json_sha256({
                "vectors": hashlib.sha256(vectors.astype("<f4", copy=False).tobytes()).hexdigest(),
                "queries": hashlib.sha256(queries.astype("<f4", copy=False).tobytes()).hexdigest(),
                "epochs": epochs.tolist(), "counts": {density: cell.admitted_count for density, cell in cells.items()},
            }),
        },
        "ek5": audit, "engines": engines,
        "passed": audit["parity_passed"] and all(
            "error" not in engine and len(engine["cells"]) == len(RETRACTION_PERCENTAGES)
            and all(cell["passed"] for cell in engine["cells"]) for engine in engines
        ),
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("local-drycheck", "run-live"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--qdrant-url")
    parser.add_argument("--opensearch-url")
    parser.add_argument("--pg-dsn")
    args = parser.parse_args(argv)
    if args.mode == "run-live" and not all((args.qdrant_url, args.opensearch_url, args.pg_dsn)):
        parser.error("run-live requires --qdrant-url, --opensearch-url and --pg-dsn")
    path = args.output_dir / RECEIPT_NAME
    if path.exists():
        parser.error("the output directory already contains a Bundle 48 receipt")
    report = execute(args.mode, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as output:
        json.dump(report, output, indent=2, sort_keys=True, allow_nan=False)
        output.write("\n")
    print(f"{args.mode}: {'PASS' if report['passed'] else 'FAIL'} ({path})")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
