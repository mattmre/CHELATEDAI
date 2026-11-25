"""Run the D6 offline checks or run the live benchmark on Spark #1 (10.0.0.49)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from collections import Counter
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import httpx
import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from benchmark_utils import canonicalize_id, ndcg_at_k  # noqa: E402
from chelated_grpc_gateway import (  # noqa: E402
    EpochPinnedEvidenceFence,
    GatewayRouteRegistration,
    build_shard_sidecar,
    build_tra1_route_bundle,
)
from chelated_multibackend_appliance import (  # noqa: E402
    CSLS,
    EF_MAX,
    MERGES,
    OS_URL,
    PG_DSN,
    PG_EXACT_4096,
    ChelatedControlPlane,
    LiveOpenSearchKnnBackend,
    LivePgvectorBackend,
    LiveQdrantBackend,
    RouteClaims,
    VectorShard,
    shortlist_policy,
    verify_backend_dot_inversion,
)
from ek5_v3_stratified import EdgeAttestation  # noqa: E402
from m1_baseline_ladder import (  # noqa: E402
    compute_fastfill_backfill_order,
    select_qrels_independent_anchors,
    split_query_ids,
)
from spherical_migration_bridge import CenteredProcrustesBridge  # noqa: E402
from scripts.night1_common import current_git_sha  # noqa: E402
from wave6_common import paired_bootstrap_wave6  # noqa: E402

SPARK1_IP = "10.0.0.49"
CACHE = Path("/home/mattmre/chelated-night1/cache")
BUNDLE = "wave11_d6_opensearch_pgvector_qdrant"
MANIFEST = "wave11_d6_multibackend_manifest.json"
OUTPUT = REPO / "artifacts" / BUNDLE
DOC_COUNTS = {"SciFact": 5183, "FiQA2018": 57638}
RETRACTION_DENSITIES = (0.0, 0.1, 0.5, 0.9)
OS_DIGEST = "sha256:3bf4fe26aeb9f7b03c0ebcb1302bb24e855ee6b89fd926d26a71a83ef6f7e076"
IMAGES = {
    "opensearch": "opensearchproject/opensearch:3.9.0",
    "pgvector": "pgvector/pgvector:pg17",
    "qdrant": "qdrant/qdrant:latest",
}
CONTAINER_ARGS = {
    "opensearch": (
        "-p", "127.0.0.1:19200:9200", "-e", "discovery.type=single-node",
        "-e", "DISABLE_INSTALL_DEMO_CONFIG=true", "-e", "DISABLE_SECURITY_PLUGIN=true",
        "-e", "OPENSEARCH_JAVA_OPTS=-Xms2g -Xmx2g", "-e", "knn.algo_param.index_thread_qty=4",
    ),
    "pgvector": ("-p", "127.0.0.1:15432:5432", "--shm-size=1g", "-e", "POSTGRES_PASSWORD=chelated_bench_2026"),
    "qdrant": ("-p", "127.0.0.1:16333:6333", "-p", "127.0.0.1:16334:6334"),
}
DRIVERS = {"opensearch": LiveOpenSearchKnnBackend, "pgvector": LivePgvectorBackend, "qdrant": LiveQdrantBackend}



def sha256_file(path):
    with path.open("rb") as file:
        return hashlib.file_digest(file, "sha256").hexdigest()


def write_json_atomic(path, payload):
    data = (json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n").encode("utf-8")
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, delete=False) as file:
            tmp = Path(file.name)
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        os.replace(tmp, path)
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
    return hashlib.sha256(data).hexdigest()


def write_bundle(output, manifest):
    output.mkdir(parents=True, exist_ok=True)
    digest = write_json_atomic(output / MANIFEST, manifest)
    receipt = {
        "schema_version": "report_receipt_v4_0",
        "bundle_id": BUNDLE,
        "manifest_file": MANIFEST,
        "manifest_sha256": digest,
        "artifact_hashes": {MANIFEST: digest},
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": manifest["git_sha"],
        "status": manifest["status"],
    }
    write_json_atomic(output / "report_receipt.json", receipt)
    return receipt


def _docker(*args, timeout=600):
    proc = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)
    if proc.returncode:
        # Print the operation only. Container setup arguments can contain credentials.
        raise RuntimeError(f"Docker {args[0]} exited {proc.returncode}: {proc.stderr.strip()}")
    return proc.stdout.strip()


@contextmanager
def spark_services():
    token = uuid.uuid4().hex
    created = []
    provenance = {}
    try:
        for engine, image in IMAGES.items():
            _docker("pull", image)
            (details,) = json.loads(_docker("image", "inspect", image))
            digests = details.get("RepoDigests", [])
            if not digests:
                raise RuntimeError(f"Docker returned no RepoDigests for {image}")
            if engine == "opensearch" and not any(d.endswith("@" + OS_DIGEST) for d in digests):
                raise RuntimeError(f"OpenSearch 3.9.0 digest differs from the L1 pin: {digests}")
            name = f"wave11-d6-{engine}"
            ident = _docker(
                "create", "--name", name, "--label", f"org.chelated.wave11-d6={token}", "--memory=8g",
                *CONTAINER_ARGS[engine], details["Id"],
            )
            created.append((name, ident))
            _docker("start", ident)
            provenance[engine] = {
                "image": image, "repo_digests": digests, "image_id": details["Id"],
                "architecture": details["Architecture"], "container_name": name, "memory_limit": "8g",
            }
        wait_for_services()
        yield provenance
    finally:
        failures = []
        for name, ident in reversed(created):
            try:
                (container,) = json.loads(_docker("container", "inspect", ident, timeout=20))
                if (
                    container["Name"] != "/" + name or name not in {f"wave11-d6-{e}" for e in IMAGES}
                    or container["Config"]["Labels"].get("org.chelated.wave11-d6") != token
                ):
                    raise RuntimeError(f"Container ownership changed for {name}")
                _docker("rm", "-f", "-v", ident, timeout=45)
            except (RuntimeError, subprocess.SubprocessError) as exc:
                failures.append(str(exc))
        if failures:
            raise RuntimeError("D6 container cleanup failed: " + "; ".join(failures))


def wait_for_services():
    import psycopg2

    pending = {"opensearch", "pgvector", "qdrant"}
    errors = {}
    deadline = time.monotonic() + 240
    with httpx.Client(timeout=5, trust_env=False) as client:
        while pending and time.monotonic() < deadline:
            for engine in sorted(pending):
                try:
                    if engine == "pgvector":
                        psycopg2.connect(PG_DSN, connect_timeout=3).close()
                    elif engine == "opensearch":
                        response = client.get(OS_URL + "/_cluster/health", params={"wait_for_status": "yellow"})
                        response.raise_for_status()
                        if response.json()["status"] not in ("yellow", "green"):
                            raise RuntimeError("OpenSearch cluster is still starting")
                    else:
                        client.get("http://127.0.0.1:16333/readyz").raise_for_status()
                    pending.remove(engine)
                except (httpx.HTTPError, psycopg2.OperationalError, RuntimeError) as exc:
                    errors[engine] = str(exc)
            if pending:
                time.sleep(2)
    if pending:
        raise TimeoutError(f"D6 service readiness failed: { {e: errors[e] for e in sorted(pending)} }")


@dataclass(frozen=True)
class Corpus:
    doc_ids: tuple[str, ...]
    qrels: dict
    qrels_sha: str
    dev_ids: tuple[str, ...]
    report_ids: tuple[str, ...]
    dev: tuple[np.ndarray, np.ndarray]
    report: tuple[np.ndarray, np.ndarray]
    legacy_docs: np.ndarray
    native_docs: np.ndarray
    provenance: dict


def cache_paths(cache_dir, domain, native_dim):
    return {dim: cache_dir / f"ood_{domain}_D{dim}.npz" for dim in (384, native_dim)}


def load_corpus(cache_dir, domain, native_dim):
    from scripts.night1_common import load_beir_dataset

    doc_ids, _, qids, _, qrels, qrels_sha = load_beir_dataset(domain)
    doc_ids, qids = tuple(map(canonicalize_id, doc_ids)), tuple(map(canonicalize_id, qids))
    if len(doc_ids) != DOC_COUNTS[domain] or len(set(doc_ids)) != len(doc_ids) or len(set(qids)) != len(qids):
        raise RuntimeError(f"Unexpected BEIR IDs for {domain}")
    arrays, provenance = {}, {}
    for dim, path in cache_paths(cache_dir, domain, native_dim).items():
        with np.load(path, allow_pickle=False) as blob:
            docs = blob["docs"].astype(np.float32, copy=False)
            queries = blob["queries"].astype(np.float32, copy=False)
            for key, ids in (("doc_ids", doc_ids), ("query_ids", qids)):
                if key in blob and tuple(map(canonicalize_id, blob[key].tolist())) != ids:
                    raise RuntimeError(f"Embedding order disagrees with BEIR {key}: {path}")
            provenance[str(dim)] = {
                "path": str(path), "sha256": sha256_file(path), "cache_lineage": f"Wave-6 D{dim} ordered cache",
                "doc_ids_in_cache": "doc_ids" in blob, "query_ids_in_cache": "query_ids" in blob,
            }
        if docs.shape != (len(doc_ids), dim) or queries.shape != (len(qids), dim):
            raise RuntimeError(f"Embedding cache shape disagrees with full BEIR corpus: {path}")
        if not (np.isfinite(docs).all() and np.isfinite(queries).all()):
            raise RuntimeError(f"Non-finite embedding cache: {path}")
        # Wave-6 D4096 caches were saved in fp16/bf16 (~2e-3 norm rounding); verify near-unit and re-normalize in fp32.
        if any(np.max(np.abs(np.linalg.norm(rows, axis=1) - 1.0)) > 2e-2 for rows in (docs, queries)):
            raise RuntimeError(f"Expected L2-normalized embeddings as produced by the Wave-6 cache: {path}")
        docs = _unit(docs.astype(np.float64)).astype(np.float32)
        queries = _unit(queries.astype(np.float64)).astype(np.float32)
        arrays[dim] = (docs, queries)

    legacy_docs, legacy_queries = arrays[384]
    positions = {qid: pos for pos, qid in enumerate(qids)}
    split = split_query_ids(qids)
    dev_ids = tuple(split["DEVELOPMENT"])
    report_ids = tuple(qid for qid in split["REPORT"] if qrels.get(qid))
    if len(dev_ids) < 16 or not report_ids:
        raise RuntimeError(f"Missing required DEV or REPORT rows for {domain}")
    query_arrays = (legacy_queries, arrays[native_dim][1])
    dev_idx = [positions[qid] for qid in dev_ids]
    report_idx = [positions[qid] for qid in report_ids]
    return Corpus(
        doc_ids, qrels, qrels_sha, dev_ids, report_ids,
        tuple(q[dev_idx] for q in query_arrays), tuple(q[report_idx] for q in query_arrays),
        legacy_docs, arrays[native_dim][0], provenance,
    )



def _unit(rows):
    return rows / np.linalg.norm(rows, axis=1, keepdims=True)


def _bind_control(backends, ids, docs, bridge, corpus, source, fence, snapshot_id):
    target = 1 - source
    dev = [None, None]
    dev[source] = _unit(corpus.dev[source])
    dev[target] = bridge.transform(corpus.dev[source], normalize=True, drop_mu_y=True).astype(np.float32)
    shards = []
    for backend, shard_ids, vectors, refs in zip(backends, ids, docs, dev, strict=True):
        sidecar = build_shard_sidecar(
            shard_tag=backend.name, collection_name=backend.name, doc_ids=shard_ids, docs_raw=vectors,
            dev_queries_in_shard_space=refs,
        )
        raw = refs @ vectors.T
        shards.append(VectorShard(backend, sidecar, float(raw.mean(dtype=np.float64)),
                                  max(float(raw.std(dtype=np.float64)), 1e-6)))
    retained = np.linalg.norm((corpus.dev[source] - bridge.mu_x) @ bridge.Q, axis=1)
    original_norm = np.linalg.norm(corpus.dev[source] - bridge.mu_x, axis=1)
    coverage = float(np.mean(retained >= 0.05 * original_norm))
    q_space, index, contract, evidence = build_tra1_route_bundle(
        index_id=backends[target].name, source_encoder_id=f"wave6-cache-d{backends[source].dim}",
        source_dim=backends[source].dim, target_encoder_id=f"wave6-cache-d{backends[target].dim}",
        target_dim=backends[target].dim, bridge=bridge, corpus_snapshot_id=snapshot_id,
        document_count=len(ids[target]), validation_count=len(corpus.dev_ids), validation_confidence=coverage,
    )
    generation = hashlib.sha256(
        (snapshot_id + "|" + "|".join(backend.name for backend in backends)).encode()
    ).hexdigest()
    registration = GatewayRouteRegistration(
        route_key=backends[target].name, index_generation=generation, query_space=q_space,
        resident_index=index, bridge_contract=contract, bridge=bridge,
        trusted_evidence_sha256s=frozenset({evidence}), trusted_bridge_contract_ids=frozenset({contract.stable_id}),
        primary_sidecar=shards[target].sidecar, secondary_sidecar=shards[source].sidecar, ema_alpha=0.0,
    )
    claims = RouteClaims(q_space.stable_id, contract.stable_id, bridge.weights_sha256, generation)
    control = ChelatedControlPlane(*shards, registration, fence)
    route = {"query_dim": backends[source].dim, "bridge_weights_sha256": claims.weights_sha256,
             "bridge_contract_id": claims.bridge_contract_id, "index_generation": generation,
             "dev_retained_energy_pass_fraction": coverage,
             "sidecar_bytes_per_doc": 8, "sidecar_payload_bytes": sum(s.sidecar.total_sidecar_bytes for s in shards),
             "centroid_bytes": sum(s.sidecar.shared_centroid_bytes for s in shards)}
    return control, claims, route


def epochs_with_lag(doc_ids, fence):
    perm = np.random.default_rng(1106).permutation(len(doc_ids))
    head = doc_ids[int(perm[-1])]
    prev_count = 0
    for epoch, rho in enumerate(RETRACTION_DENSITIES, 1):
        count = round(len(doc_ids) * rho)
        if count:
            snapshot = fence.commit_mutation(
                add_attestations=[
                    EdgeAttestation(attestation_id=f"d6_supersede_{i}", src_id=head, dst_id=doc_ids[int(perm[i])],
                                    edge_type="SUPERSEDE", tx_time=epoch)
                    for i in range(prev_count, count)
                ],
                target_epoch=epoch,
            )
        else:
            snapshot = fence.wait_and_get_snapshot(epoch)
        if len(snapshot.admitted_doc_ids) != len(doc_ids) - count:
            raise RuntimeError(f"EK5 retraction snapshot does not match schedule at epoch {epoch}")
        lag_count = min(count, round(len(doc_ids) * 0.05))
        lag = frozenset(doc_ids[int(i)] for i in perm[count - lag_count:count])
        yield rho, snapshot, lag
        prev_count = count


def _recall(ranked, expected):
    return len(set(ranked).intersection(expected)) / len(expected) if expected else 1.0


def quality_metrics(rankings, qrels, qids):
    report = {}
    for metric in ("ndcg_at_10", "recall_at_10"):
        scores = {}
        for mode in MERGES:
            scores[mode] = {
                qid: (
                    ndcg_at_k(rankings[qid][mode], qrels[qid], k=10)
                    if metric == "ndcg_at_10" else
                    _recall(rankings[qid][mode], {did for did, rel in qrels[qid].items() if rel > 0})
                )
                for qid in qids
            }
        report[metric] = {
            "means": {mode: float(np.mean(list(values.values()))) for mode, values in scores.items()},
            "paired_bootstrap_wave6": {
                f"{CSLS}_minus_{mode}": paired_bootstrap_wave6(scores[CSLS], scores[mode], expected_ids=qids)
                for mode in MERGES if mode != CSLS
            },
        }
    return report


def _rtt(values):
    if not values:
        return None
    return dict(zip(("p50_ms", "p95_ms", "p99_ms"), np.percentile(values, [50, 95, 99]).tolist(), strict=True))


def _exact_oracle(control, docs):
    return tuple(
        (rows.astype(np.float16) if shard.backend.fp16 else rows).astype(np.float64)
        for shard, rows in zip(control.shards, docs, strict=True)
    )


def _exact_top_10(control, queries, vectors, wire_allowed):
    # Independent exhaustive NumPy oracle matched to the indexed operand precision.
    scores_by_id = {}
    for shard, query, rows, allowed in zip(control.shards, queries, vectors, wire_allowed, strict=True):
        sidecar = shard.sidecar
        idx = np.asarray(allowed, dtype=np.intp) if allowed is not None else np.arange(sidecar.n_docs)
        if not idx.size:
            continue
        operand = (query.astype(np.float16) if shard.backend.fp16 else query).astype(np.float64)
        dots = rows[idx] @ operand if allowed is not None else rows @ operand
        centered = (dots - query.astype(np.float64) @ sidecar.mu_d) / sidecar.doc_centered_norms[idx]
        top = min(sidecar.csls_k, centered.size)
        penalty = np.partition(centered, -top)[-top:].mean()
        scores = 2 * centered - sidecar.csls_r_d[idx] - penalty
        scores_by_id.update((sidecar.doc_ids[i], float(score)) for i, score in zip(idx, scores, strict=True))
    return sorted(scores_by_id, key=lambda did: (-scores_by_id[did], did))[:10]


def _point_filter(ids, mask):
    allowed = tuple(i for i, did in enumerate(ids) if did in mask)
    return None if len(allowed) == len(ids) else allowed


def _capture_pg_plans(control, queries, snapshot, allowed):
    plans = {}
    for shard, q, wire in zip(control.shards, queries, allowed, strict=True):
        backend = shard.backend
        if not isinstance(backend, LivePgvectorBackend):
            continue
        total = shard.sidecar.n_docs if wire is None else len(wire)
        if not total:
            continue
        depth, ef, use_exact = shortlist_policy(64, total / shard.sidecar.n_docs)
        for exact in (False, True):
            if not exact and (use_exact or backend.route == PG_EXACT_4096):
                continue
            limit = total if use_exact else min(total, depth)
            plan = backend.explain(q, allowed=wire, epoch=snapshot.epoch, limit=limit, ef=ef, exact=exact)
            used_hnsw = backend.name + "_hnsw" in plan
            if used_hnsw == exact:
                raise RuntimeError(f"Unexpected pgvector query plan on {backend.name} with {exact=}: {plan[:800]}")
            routes = plans.setdefault(backend.name, {})
            # Keep the EXPLAIN shape and measured work, omitting literal vector and allowlist values.
            routes["exact" if exact else "ann"] = {
                "requested_limit": limit, "hnsw_used": used_hnsw,
                "plan": re.sub(r"'(?:\[[-+0-9eE., ]+\]|\{[0-9, ]*\})'", "'[operands elided]'", plan),
            }
    return plans


def evaluate(control, claims, corpus, query_source, snapshot, lag, *, oracle=None):
    qids = corpus.report_ids
    queries = corpus.report[query_source]
    served, initial = {}, {}
    service_rtt, initial_rtt, recalls, initial_recalls = [], [], [], []
    final_routes, initial_routes, raw = Counter(), Counter(), Counter()
    egress_stale = egress_count = replica_stale = replica_hits = 0
    admitted = snapshot.admitted_doc_ids
    ids = tuple(shard.sidecar.doc_ids for shard in control.shards)
    allowed = tuple(_point_filter(sids, admitted) for sids in ids)
    replica_allowed = tuple(_point_filter(sids, admitted | lag) for sids in ids)
    first_projected = control.prepare_query(queries[0], claims)
    plans = _capture_pg_plans(control, first_projected, snapshot, allowed)
    for qid, q in zip(qids, queries, strict=True):
        t0 = time.perf_counter()
        result = control.search(q, claims, epoch=snapshot.epoch)
        service_rtt.append((time.perf_counter() - t0) * 1000)
        served[qid] = result.merges
        if all(shard.initial_route for shard in result.shards):
            initial[qid] = result.initial_merges
            initial_rtt.append(sum(shard.ann_request_rtt_ms for shard in result.shards))
        for shard in result.shards:
            final_routes[shard.route] += 1
            if shard.initial_route:
                initial_routes[shard.initial_route] += 1
            raw.update({"hits": shard.raw_hits, "stale_hits": shard.raw_stale_hits,
                        "exact_fallbacks": int(shard.exact_fallback), "verified_exhaustions": int(shard.exhausted),
                        "refills": shard.refills})
        ranked = result.merges[CSLS]
        egress_stale += sum(did not in admitted for did in ranked)
        egress_count += len(ranked)
        if oracle is not None:
            projected = control.prepare_query(q, claims)
            expected = _exact_top_10(control, projected, oracle, allowed)
            recalls.append(_recall(ranked, expected))
            if qid in initial:
                initial_recalls.append(_recall(result.initial_merges[CSLS], expected))
            for shard, operand, wire in zip(control.shards, projected, replica_allowed, strict=True):
                page = shard.backend.search(operand, allowed=wire, epoch=snapshot.epoch, limit=10, ef=128, exact=False)
                replica_hits += len(page.hits)
                replica_stale += sum(hit.doc_id not in admitted for hit in page.hits)
    return {
        "report_queries": len(qids), "epoch": snapshot.epoch, "admitted_documents": len(admitted),
        "replica_lagged_documents": len(lag), "service_rtt": _rtt(service_rtt),
        "initial_backend_requests_rtt_sum": _rtt(initial_rtt),
        "initial_routes": dict(initial_routes), "served_routes": dict(final_routes),
        "exact_fallback_fraction_of_shard_queries": raw["exact_fallbacks"] / (len(qids) * len(control.shards)),
        "counts": dict(raw), "pgvector_explain_analyze": plans,
        "admitted_recall_at_10": float(np.mean(recalls)) if recalls else None,
        "initial_admitted_recall_at_10": float(np.mean(initial_recalls)) if initial_recalls else None,
        "stale_resurrection_rate": egress_stale / max(1, egress_count),
        "raw_engine_stale_hit_rate": raw["stale_hits"] / max(1, raw["hits"]),
        "lagged_replica_only_stale_hit_rate": replica_stale / replica_hits if replica_hits else None,
        "lagged_replica_only_raw_hits": replica_hits,
        "quality": quality_metrics(served, corpus.qrels, qids) if snapshot.epoch == 1 else None,
        "initial_shortlist_quality": quality_metrics(initial, corpus.qrels, qids)
        if snapshot.epoch == 1 and len(initial) == len(qids) else None,
    }



def _backends_for_cell(stack, engine, prefix, dim, pg_exact):
    driver = DRIVERS[engine]
    legacy = driver(prefix + "_old", 384)
    stack.callback(legacy.close)
    native = driver(prefix + "_new", dim, **({"route": PG_EXACT_4096} if pg_exact else {}))
    stack.callback(native.close)
    return legacy, native


def _route_variant(engine, dim, pg_exact):
    return ("pgvector_exact_vector4096" if pg_exact else
            "pgvector_halfvec4000_hnsw_cte" if engine == "pgvector" and dim == 4096 else engine)


def benchmark_corpus(corpus, domain, dim, provenance, canaries):
    priority = compute_fastfill_backfill_order(corpus.dev[0], corpus.legacy_docs, top_m=20)
    native_mask = np.zeros(len(corpus.doc_ids), dtype=bool)
    native_mask[priority[:round(0.2 * len(native_mask))]] = True
    positions = (np.flatnonzero(~native_mask), np.flatnonzero(native_mask))
    ids = tuple(tuple(corpus.doc_ids[i] for i in shard) for shard in positions)
    all_docs = (corpus.legacy_docs, corpus.native_docs)
    docs = tuple(vectors[idx] for vectors, idx in zip(all_docs, positions, strict=True))
    anchors = select_qrels_independent_anchors(corpus.doc_ids, anchor_frac=1.0, max_anchors=500, seed=202606)
    bridges = tuple(
        CenteredProcrustesBridge.fit(
            all_docs[source], all_docs[1 - source], fit_idx=anchors, scale_mode="unit", strict_energy_check=True,
        ) for source in (0, 1)
    )
    snapshot_hash = hashlib.sha256(json.dumps({
        "qrels": corpus.qrels_sha, "caches": {dim: p["sha256"] for dim, p in corpus.provenance.items()},
        "legacy_ids": ids[0], "native_ids": ids[1],
    }, sort_keys=True).encode()).hexdigest()
    result = {
        "full_corpus_count": len(corpus.doc_ids), "shard_counts": [len(shard) for shard in ids],
        "target_backfill_fraction": 0.20, "actual_backfill_fraction": len(ids[1]) / len(corpus.doc_ids),
        "ordered_corpus_snapshot_sha256": snapshot_hash, "cache_provenance": corpus.provenance,
        "qrels_sha256": corpus.qrels_sha, "dev_query_ids": corpus.dev_ids, "report_query_ids": corpus.report_ids,
        "qrels_blind_bridge_anchors": len(anchors), "cases": {},
    }
    fence = EpochPinnedEvidenceFence(corpus.doc_ids)
    epochs = list(epochs_with_lag(corpus.doc_ids, fence)) if domain == "SciFact" else [
        (0.0, fence.wait_and_get_snapshot(1), frozenset()),
    ]
    for engine in DRIVERS:
        cases = (False, True) if dim == 4096 and engine == "pgvector" else (False,)
        for pg_exact in cases:
            variant = _route_variant(engine, dim, pg_exact)
            print(f"D6 {domain} 384<->{dim} {variant}", flush=True)
            prefix = f"d6_{domain.lower()}_{dim}_{engine}" + ("_e" if pg_exact else "")
            with ExitStack() as stack:
                backends = _backends_for_cell(stack, engine, prefix, dim, pg_exact)
                counts = {}
                for backend, shard_ids, rows in zip(backends, ids, docs, strict=True):
                    canary_variant = backend.route if isinstance(backend, LivePgvectorBackend) else engine
                    canary_key = f"{canary_variant}_d{backend.dim}"
                    if canary_key not in canaries:
                        extras = {"route": backend.route} if isinstance(backend, LivePgvectorBackend) else {}
                        canary = DRIVERS[engine](f"canary_{len(canaries)}_{backend.dim}", backend.dim, **extras)
                        try:
                            modes = (True,) if pg_exact and backend.dim == 4096 else (False, True)
                            canaries[canary_key] = verify_backend_dot_inversion(canary, exact_modes=modes)
                        finally:
                            canary.close()
                    backend.create()
                    info = backend.server_info()
                    if engine == "opensearch" and info["number"] != "3.9.0":
                        raise RuntimeError(f"L1 requires OpenSearch server 3.9.0, got {info['number']}")
                    if engine == "pgvector" and not info["server"].startswith("PostgreSQL 17."):
                        raise RuntimeError(f"L1 requires PostgreSQL server 17, got {info['server']}")
                    provenance[engine]["server"] = info
                    backend.ingest(shard_ids, rows, epoch=1)
                    backend.finalize()
                    count = backend.count()
                    if count != len(shard_ids):
                        raise RuntimeError(f"Ingest count mismatch on {backend.name}: {count} != {len(shard_ids)}")
                    counts[backend.name] = {"engine_count": count, "expected": len(shard_ids), "dim": backend.dim,
                                            "transport": backend.transport, "vector_egress_bytes_per_hit": 0,
                                            "stored_precision": "fp16" if backend.fp16 else "fp32"}
                    if isinstance(backend, LiveQdrantBackend):
                        collection = backend.index_info()
                        if collection["points_count"] != count or collection["indexed_vectors_count"] != count:
                            raise RuntimeError(f"Qdrant telemetry count disagrees with its exact count: {collection}")
                        counts[backend.name].update(collection)
                case = {"indexes": counts, "directions": {}}
                for source, bridge in enumerate(bridges):
                    control, claims, route = _bind_control(backends, ids, docs, bridge, corpus, source, fence, snapshot_hash)
                    oracle = _exact_oracle(control, docs) if domain == "SciFact" else None
                    cells = {}
                    for rho, snapshot, lag in epochs:
                        cells[str(round(rho * 100))] = evaluate(
                            control, claims, corpus, source, snapshot, lag, oracle=oracle,
                        )
                    case["directions"][f"d{backends[source].dim}_to_d{backends[1 - source].dim}"] = {
                        "route": route, "retraction_percent": cells,
                    }
                    del oracle
                result["cases"][variant] = case
    return result


def _spark_only():
    # Bind in-process to ensure this Docker command targets the authorized Spark host.
    if os.environ.get("DOCKER_HOST") or os.environ.get("DOCKER_CONTEXT"):
        raise RuntimeError("D6 expects the local Docker socket on Spark #1")
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        try:
            probe.bind((SPARK1_IP, 0))
        except OSError as exc:
            raise RuntimeError(f"Run spark1-live on mattmre@{SPARK1_IP} from the staged checkout") from exc
    endpoint = _docker("context", "inspect", "--format", "{{.Endpoints.docker.Host}}", timeout=20)
    if not endpoint.startswith("unix://"):
        raise RuntimeError(f"D6 requires a local Unix Docker endpoint, found {endpoint}")


def run_live(cache_dir, output):
    missing = sorted({
        str(path) for domain in DOC_COUNTS for dim in (1024, 4096)
        for path in cache_paths(cache_dir, domain, dim).values() if not path.is_file()
    })
    if missing:
        raise FileNotFoundError("Required Wave-6 BEIR embedding cache files are missing: " + ", ".join(missing))
    if (output / "report_receipt.json").exists():
        raise FileExistsError(f"An existing D6 receipt is present in {output}; use --output for a new attempt")
    import importlib

    for dependency in ("psycopg2", "pgvector.psycopg2", "qdrant_client"):
        importlib.import_module(dependency)
    _spark_only()
    git_sha = current_git_sha(REPO)
    implementation_hashes = {
        str(path.relative_to(REPO).as_posix()): sha256_file(path)
        for path in (Path(__file__).resolve(), REPO / "chelated_multibackend_appliance.py")
    }
    started = time.perf_counter()
    canaries, benchmark = {}, {}
    with spark_services() as provenance:
        for domain in DOC_COUNTS:
            benchmark[domain] = {}
            for dim in (1024, 4096):
                corpus = load_corpus(cache_dir, domain, dim)
                benchmark[domain][str(dim)] = benchmark_corpus(corpus, domain, dim, provenance, canaries)
                del corpus
    gate_failures = []
    for dim, domain in benchmark["SciFact"].items():
        for variant, case in domain["cases"].items():
            for direction, document in case["directions"].items():
                for rho, cell in document["retraction_percent"].items():
                    if (cell["admitted_recall_at_10"] < 0.995 or cell["stale_resurrection_rate"] != 0.0
                        or cell["raw_engine_stale_hit_rate"] != 0.0):
                        gate_failures.append(f"SciFact/{dim}/{variant}/{direction}/{rho}")
    status = "FAILED" if gate_failures else "COMPLETED"
    receipt = write_bundle(output, {
        "schema_version": "wave11_d6_multibackend_v1", "bundle_id": BUNDLE, "status": status,
        "git_sha": git_sha, "implementation_sha256": implementation_hashes, "spark1_ip": SPARK1_IP,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": time.perf_counter() - started, "engine_provenance": provenance,
        "score_inversion_canaries": canaries, "benchmarks": benchmark, "failed_gates": gate_failures,
        "shortlist_policy": {
            "selection": "frozen L1 policy", "k_os": 64, "k_os_eff": "min(1024, ceil(4*k_os/max(p_admit,0.05)))",
            "ef_max": EF_MAX, "exact_exhaustion_at_p_admit_lte": 0.15,
            "certificate": "verified exhaustion or exact dot-prefix bound for CSLS and r_q",
        },
        "methodology": {
            "queries": "DEVELOPMENT / REPORT by m1-query-split-v1; no REPORT tuning",
            "shortlist_quality": "initial live candidate responses; separately reported from served certified path",
            "retraction_oracle": "exhaustive matched-epoch NumPy float64 over stored-precision operands",
            "replica_lag": "simulated stale replica allowlist over live backend; up to 5% of corpus",
            "pgvector_plans": "EXPLAIN (ANALYZE, BUFFERS) for first REPORT query per route/density; literals elided",
            "rtt": "in-process control-plane request plus local database wire roundtrips",
        },
    })
    print(f"{status} {output / MANIFEST} SHA-256 {receipt['manifest_sha256']}", flush=True)
    if gate_failures:
        raise RuntimeError("D6 retrieval gates failed: " + ", ".join(gate_failures))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("local-smoke", "spark1-live"))
    parser.add_argument("--cache-dir", type=Path, default=CACHE)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args(argv)
    if args.mode == "local-smoke":
        subprocess.run(
            [sys.executable, "-B", "-m", "pytest", "-p", "no:cacheprovider", "-q", "tests/test_wave11_d6_multibackend.py"],
            cwd=REPO, check=True,
        )
        return

    def terminate(signum, _frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    run_live(args.cache_dir, args.output)


if __name__ == "__main__":
    main()

