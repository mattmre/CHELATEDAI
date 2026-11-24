"""Offline wire, score, exhaustive-certificate and receipt checks for Wave 11 D6."""
from __future__ import annotations

import hashlib
import io
import json
import threading
from dataclasses import replace
from types import SimpleNamespace

import httpx
import numpy as np
import pytest

from chelated_grpc_gateway import (
    EpochPinnedEvidenceFence,
    GatewayRouteRegistration,
    TRA1AuthorizationError,
    build_shard_sidecar,
    build_tra1_route_bundle,
)
from chelated_multibackend_appliance import (
    CSLS,
    MERGES,
    PG_EXACT_4096,
    PG_HALF_CTE,
    PG_HNSW,
    ChelatedControlPlane,
    LiveOpenSearchKnnBackend,
    LivePgvectorBackend,
    RouteClaims,
    SearchPage,
    VectorHit,
    VectorShard,
    invert_opensearch_score,
    pgvector_index_sql,
    pgvector_search_sql,
    shortlist_policy,
    verify_backend_dot_inversion,
)
from scripts import run_wave11_d6_opensearch_pgvector_qdrant as runner
from spherical_migration_bridge import CenteredProcrustesBridge


def test_opensearch_piecewise_fp32_wire_score():
    dots = np.r_[np.linspace(-1, 1, 1003), np.linspace(0.3275, 0.327505, 50), -1e-7, 0.0, 1e-7]
    wire_scores = (1 + dots).astype(np.float32)
    negative = dots < 0
    wire_scores[negative] = 1 / (1 - dots[negative])
    recovered = np.array([invert_opensearch_score(s) for s in wire_scores])
    assert np.max(np.abs(dots - recovered)) < 1e-6
    np.testing.assert_array_equal([invert_opensearch_score(s) for s in (0.5, 1, 2)], [-1, 0, 1])
    for invalid in [0.0, -1.0, float("inf"), float("nan")]:
        with pytest.raises(ValueError):
            invert_opensearch_score(invalid)


def test_opensearch_wire_filters_scores_and_source_projection():
    requests = []

    def wire(request):
        body = json.loads(request.content)
        requests.append(body)
        return httpx.Response(200, json={
            "timed_out": False, "_shards": {"failed": 0}, "hits": {"hits": [
                {"_source": {"doc_id": "first", "point_id": 7, "admitted_epoch": 1}, "_score": 0.8, "sort": [0.8, 7]},
            ]},
        })

    backend = LiveOpenSearchKnnBackend("wire_test", 3)
    backend.client.close()
    backend.client = httpx.Client(base_url="http://offline", transport=httpx.MockTransport(wire))
    for exact in (False, True):
        page = backend.search(np.array([1, 0, 0]), allowed=(7,), epoch=2, limit=1, ef=256, exact=exact)
        assert page.hits == (VectorHit("first", 7, 1, -0.25),)
        assert page.exact_prefix is exact
        assert requests[-1]["_source"] == {"excludes": ["embedding"]}
    knn = requests[0]["query"]["knn"]["embedding"]
    exact = requests[1]["query"]["script_score"]
    assert knn["k"] == 1 and knn["method_parameters"]["ef_search"] == 256
    assert knn["filter"] == exact["query"] == {"bool": {"filter": [
        {"range": {"admitted_epoch": {"lte": 2}}}, {"terms": {"point_id": [7]}},
    ]}}
    assert exact["script"]["params"]["space_type"] == "innerproduct"
    backend.close()


@pytest.mark.parametrize("dim", [384, 1024])
def test_pgvector_native_hnsw_sql(dim):
    ddl = pgvector_index_sql("demo", dim, PG_HNSW)
    assert "USING hnsw (embedding vector_ip_ops) WITH (m = 16, ef_construction = 128)" in ddl
    sql, params = pgvector_search_sql(
        "demo", PG_HNSW, query=np.arange(dim), allowed=(1, 3), epoch=9, limit=20, exact=False,
    )
    assert "point_id = ANY(%(allowed)s)" in sql
    assert "ORDER BY embedding <#> %(q)s::vector LIMIT %(limit)s" in sql
    assert "-1.0 * (embedding <#> %(q)s::vector) AS raw_dot" in sql
    assert sql.endswith("SELECT doc_id, point_id, admitted_epoch, raw_dot FROM shortlist ORDER BY raw_dot DESC, point_id")
    assert params["epoch"] == 9 and params["allowed"] == [1, 3]


def test_pgvector_halfvec_cte_and_exact_vector_4096_sql():
    ddl = pgvector_index_sql("demo", 4096, PG_HALF_CTE)
    assert "USING hnsw ((subvector(embedding, 1, 4000)::halfvec(4000)) halfvec_ip_ops)" in ddl
    assert pgvector_index_sql("demo", 4096, PG_EXACT_4096) is None
    with pytest.raises(ValueError):
        pgvector_index_sql("demo", 4096, PG_HNSW)
    with pytest.raises(ValueError):
        pgvector_index_sql("wrong;name", 384, PG_HNSW)

    kwargs = {"query": np.zeros(4096), "allowed": (6, 9), "epoch": 3, "limit": 10, "exact": False}
    cte, params = pgvector_search_sql("demo", PG_HALF_CTE, k_cte=77, **kwargs)
    assert cte.startswith("WITH shortlist AS MATERIALIZED")
    assert "ORDER BY subvector(embedding, 1, 4000)::halfvec(4000) <#>" in cte
    assert "LIMIT %(k_cte)s) SELECT doc_id, point_id, admitted_epoch, -1.0 * " in cte
    assert "embedding <#> %(q)s::halfvec(4096)) AS raw_dot" in cte
    assert params["k_cte"] == 77
    exact, _ = pgvector_search_sql("demo", PG_EXACT_4096, **kwargs)
    assert exact.startswith("SELECT doc_id, point_id, admitted_epoch, -1.0 * (embedding <#> %(q)s::vector)")
    assert "ORDER BY raw_dot DESC, point_id" in exact
    assert "WITH shortlist" not in exact


@pytest.mark.parametrize("iterative", [True, False])
def test_pgvector_transaction_settings_and_projected_scores(iterative):
    statements = []

    class Cursor:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def execute(self, sql, params=None):
            statements.append((sql, params))

        def fetchall(self):
            return [("d1", 1, 1, 0.5), ("d2", 2, 1, -0.125)]

    cursor = Cursor()

    class Conn:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def cursor(self):
            return cursor

    backend = LivePgvectorBackend.__new__(LivePgvectorBackend)
    backend.name, backend.dim, backend.route = "demo", 384, PG_HNSW
    backend.fp16, backend.iterative_scan, backend.conn = False, iterative, Conn()
    backend._lock = threading.RLock()
    params = {"allowed": (1, 2), "epoch": 2, "limit": 2, "ef": 512}
    ann = backend.search(np.zeros(384), exact=False, **params)
    [(statement, bindings)] = statements
    assert "SET LOCAL hnsw.ef_search = %(ef_search)s" in statement
    assert bindings["ef_search"] == 512
    assert ("SET LOCAL hnsw.iterative_scan = 'relaxed_order'" in statement) is iterative
    assert [hit.raw_dot for hit in ann.hits] == [0.5, -0.125]
    statements.clear()
    exact = backend.search(np.zeros(384), exact=True, **params)
    assert exact.exact_prefix is True
    [(statement, _)] = statements
    assert statement.split(";\n")[:3] == list(backend._EXACT_SETTINGS)


class OfflineBackend:
    transport = "offline fixture"

    def __init__(self, name, dim, *, fp16=False):
        self.name, self.dim, self.fp16 = name, dim, fp16
        self.requests = []
        self.short_exact_once = False
        self.omit_exact = False
        self.bad_dot = False
        self.inject = None

    def create(self):
        pass

    def finalize(self):
        pass

    def ingest(self, doc_ids, embeddings, epoch):
        self.ids = doc_ids
        dtype = np.float16 if self.fp16 else np.float32
        self.vectors = embeddings.astype(dtype).astype(np.float64)
        self.generation = epoch

    def count(self):
        return len(self.ids)

    def search(self, query, *, allowed, epoch, limit, ef, exact):
        self.requests.append((allowed, epoch, limit, ef, exact))
        idx = np.arange(len(self.ids)) if allowed is None else np.asarray(allowed, dtype=np.intp)
        operand = query.astype(np.float16 if self.fp16 else np.float32).astype(np.float64)
        dots = self.vectors[idx] @ operand
        ordered = np.argsort(-dots, kind="stable")
        hits = tuple(
            VectorHit(self.ids[int(idx[row])], int(idx[row]), self.generation, float(dots[row]) + int(self.bad_dot))
            for row in ordered[:limit]
        )
        if exact and (self.short_exact_once or self.omit_exact):
            self.short_exact_once = False
            hits = hits[:-1]
        if not exact and self.inject is not None:
            hits = (self.inject, *hits[:-1])
        return SearchPage(hits, "exact" if exact else "ann", exact)


@pytest.mark.parametrize("fp16", [False, True])
def test_mandatory_canary_gates_coverage_and_rounding(fp16):
    backend = OfflineBackend("canary", 32, fp16=fp16)
    canary = verify_backend_dot_inversion(backend)
    assert canary["post_ingest_count"] == 1040
    for mode in canary["modes"].values():
        assert mode["hits_checked"] == 1040
        assert mode["max_abs_err"] < (2e-3 if fp16 else 1e-5)
        assert set(mode["groups"]) == {"negative", "zero", "positive", "near_tie"}
        rounding_err = max(group["max_abs_fp16_rounding_err"] for group in mode["groups"].values())
        if fp16:
            assert rounding_err > 0
        else:
            assert rounding_err == 0 and mode["max_abs_err"] < 1e-12
    backend.bad_dot = True
    with pytest.raises(RuntimeError, match="Dot canary failed"):
        verify_backend_dot_inversion(backend)



@pytest.fixture
def control_fixture():
    rng = np.random.default_rng(202611)
    total = 2200
    doc_ids = tuple(f"d{i:05d}" for i in range(total))
    old = rng.normal(size=(total, 8))
    old = (old / np.linalg.norm(old, axis=1, keepdims=True)).astype(np.float32)
    rotation, _ = np.linalg.qr(rng.normal(size=(12, 8)))
    new = (old @ rotation.T).astype(np.float32)
    bridge = CenteredProcrustesBridge.fit(old, new, fit_idx=np.arange(80), scale_mode="unit")
    vectors = (old[:1100], new[1100:])
    ids = (doc_ids[:1100], doc_ids[1100:])
    sides = []
    for dim, shard_ids, rows in zip((8, 12), ids, vectors, strict=True):
        backend = OfflineBackend(f"dim_{dim}", dim)
        backend.ingest(shard_ids, rows, 1)
        sidecar = build_shard_sidecar(shard_tag=backend.name, collection_name=backend.name, doc_ids=shard_ids,
                                     docs_raw=rows, dev_queries_in_shard_space=rows[:20])
        assert sidecar.total_sidecar_bytes == 8 * len(shard_ids)
        sides.append(VectorShard(backend, sidecar, 0.0, 1.0))
    source, index, contract, evidence = build_tra1_route_bundle(
        index_id=sides[1].backend.name, source_encoder_id="source", source_dim=8,
        target_encoder_id="native", target_dim=12, bridge=bridge, corpus_snapshot_id="offline",
        document_count=len(ids[1]),
    )
    reg = GatewayRouteRegistration(
        route_key="offline", index_generation="generation", query_space=source, resident_index=index,
        bridge_contract=contract, bridge=bridge, trusted_evidence_sha256s=frozenset({evidence}),
        trusted_bridge_contract_ids=frozenset({contract.stable_id}), primary_sidecar=sides[1].sidecar,
        secondary_sidecar=sides[0].sidecar,
    )
    fence = EpochPinnedEvidenceFence(doc_ids)
    control = ChelatedControlPlane(*sides, reg, fence)
    claims = RouteClaims(source.stable_id, contract.stable_id, bridge.weights_sha256, "generation")
    return control, claims, old[1101], vectors, doc_ids


def test_shortlist_policy_and_verified_50_percent_recall(control_fixture):
    control, claims, query, docs, doc_ids = control_fixture
    assert shortlist_policy(64, 1.0) == (256, 128, False)
    assert shortlist_policy(64, 0.5) == (512, 256, False)
    assert shortlist_policy(64, 0.15) == (1024, 512, True)
    assert shortlist_policy(64, 0.10) == (1024, 512, True)
    snapshots = list(runner.epochs_with_lag(doc_ids, control.fence))
    _, pinned, _ = snapshots[2]
    old = control.shards[0].backend
    old.short_exact_once = True
    result = control.search(query, claims, epoch=pinned.epoch)
    allowed = tuple(runner._point_filter(s.sidecar.doc_ids, pinned.admitted_doc_ids) for s in control.shards)
    expected = runner._exact_top_10(
        control, control.prepare_query(query, claims), runner._exact_oracle(control, docs), allowed,
    )
    assert runner._recall(result.merges[CSLS], expected) >= 0.995
    assert set(result.merges) == set(MERGES)
    assert result.shards[0].refills >= 1
    assert not old.requests[0][-1]
    assert old.requests[0][0] == allowed[0]
    assert old.requests[0][3] <= 512
    assert any(request[-1] for request in old.requests[1:])
    assert sum(s.raw_stale_hits for s in result.shards) == 0
    # The 90% retraction path skips ANN and verifies complete admitted point coverage.
    _, low, _ = snapshots[3]
    result = control.search(query, claims, epoch=low.epoch)
    assert all(r.exhausted and r.exact_fallback and r.initial_route is None for r in result.shards)
    assert sum(len(r.candidates) for r in result.shards) == len(low.admitted_doc_ids)


def test_epoch_pin_drops_stale_identity_and_future_generation(control_fixture):
    control, claims, query, _, doc_ids = control_fixture
    baseline = control.search(query, claims, epoch=1)
    _, pinned, lag = list(runner.epochs_with_lag(doc_ids, control.fence))[1]
    assert lag
    old, new = control.shards
    idx = next(i for i, did in enumerate(old.sidecar.doc_ids) if did in lag)
    old.backend.inject = VectorHit(old.sidecar.doc_ids[idx], idx, 1, 10.0)
    nidx = next(i for i, did in enumerate(new.sidecar.doc_ids) if did in pinned.admitted_doc_ids)
    new.backend.inject = VectorHit(new.sidecar.doc_ids[nidx], nidx, pinned.epoch + 1, 10.0)
    fenced = control.search(query, claims, epoch=pinned.epoch)
    assert fenced.epoch == pinned.epoch
    assert sum(s.raw_stale_hits for s in fenced.shards) == 2
    assert all(set(ranking) <= pinned.admitted_doc_ids for ranking in fenced.merges.values())
    assert all(hit.admitted_epoch <= pinned.epoch for shard in fenced.shards for hit in shard.candidates)
    old.backend.inject = new.backend.inject = None
    assert control.search(query, claims, epoch=1).merges == baseline.merges
    with pytest.raises(TRA1AuthorizationError, match="EPOCH_NOT_YET_VISIBLE"):
        control.search(query, claims, epoch=5, wait_for_epoch=False)


def test_tra1_claim_mismatch_precedes_every_wire_request(control_fixture):
    control, claims, query, _, _ = control_fixture
    for field in ("query_space_id", "bridge_contract_id", "weights_sha256", "index_generation"):
        with pytest.raises(TRA1AuthorizationError):
            control.search(query, replace(claims, **{field: "invalid"}), epoch=1)
    assert all(not shard.backend.requests for shard in control.shards)


def test_short_exact_response_never_certifies_exhaustion(control_fixture):
    control, claims, query, _, doc_ids = control_fixture
    _, low, _ = list(runner.epochs_with_lag(doc_ids, control.fence))[-1]
    control.shards[0].backend.omit_exact = True
    with pytest.raises(RuntimeError, match="Cannot verify admitted exhaustion"):
        control.search(query, claims, epoch=low.epoch)


def test_atomic_manifest_and_receipt_sha256_roundtrip(monkeypatch):
    files = {}
    renames = []

    class MemoryPath:
        def __init__(self, name):
            self.name = name

        def __truediv__(self, name):
            return MemoryPath(f"{self.name}/{name}")

        @property
        def parent(self):
            return MemoryPath(self.name.rsplit("/", 1)[0])

        def mkdir(self, **_):
            pass

        def unlink(self, *, missing_ok):
            assert missing_ok
            files.pop(self.name, None)

        def open(self, mode):
            assert mode == "rb"
            return io.BytesIO(files[self.name])

    class MemoryTemp(io.BytesIO):
        def __init__(self, mode, dir, delete):
            super().__init__()
            assert (mode, delete) == ("wb", False)
            self.name = (dir / "atomic-temp").name

        def fileno(self):
            return 0

        def __exit__(self, *_):
            files[self.name] = self.getvalue()
            self.close()

    def replace_file(src, dst):
        assert src.parent.name == dst.parent.name
        files[dst.name] = files.pop(src.name)
        renames.append(dst.name)

    monkeypatch.setattr(runner, "Path", MemoryPath)
    monkeypatch.setattr(runner.tempfile, "NamedTemporaryFile", MemoryTemp)
    monkeypatch.setattr(runner.os, "replace", replace_file)
    monkeypatch.setattr(runner.os, "fsync", lambda fd: None)
    out = MemoryPath("artifacts/d6")
    manifest = {"status": "COMPLETED", "git_sha": "0123456789", "failed_gates": []}
    receipt = runner.write_bundle(out, manifest)
    payload = files[(out / runner.MANIFEST).name]
    expected = hashlib.sha256(payload).hexdigest()
    assert payload.endswith(b"\n") and b"\r" not in payload and json.loads(payload) == manifest
    receipt_bytes = files[(out / "report_receipt.json").name]
    assert receipt_bytes.endswith(b"\n") and b"\r" not in receipt_bytes
    assert json.loads(receipt_bytes) == receipt
    assert receipt["manifest_sha256"] == runner.sha256_file(out / runner.MANIFEST) == expected
    assert receipt["artifact_hashes"] == {runner.MANIFEST: expected}
    assert renames == [(out / runner.MANIFEST).name, (out / "report_receipt.json").name]
    assert set(files) == set(renames)


def test_docker_cleanup_checks_exact_ownership_on_failure(monkeypatch):
    names, labels, removed = {}, {}, []

    def docker(*args, timeout=0):
        if args[0] == "pull":
            return ""
        if args[:2] == ("image", "inspect"):
            return json.dumps([{"Id": args[2], "Architecture": "arm64",
                                "RepoDigests": [args[2] + "@" + runner.OS_DIGEST]}])
        if args[0] == "create":
            name, label = args[2], args[4]
            ident = f"id-{name}"
            names[ident] = name
            labels[ident] = dict([label.split("=", 1)])
            return ident
        if args[0] == "start":
            if "qdrant" in args[1]:
                raise RuntimeError("simulated start failure")
            return ""
        if args[:2] == ("container", "inspect"):
            return json.dumps([{"Name": "/" + names[args[2]], "Config": {"Labels": labels[args[2]]}}])
        if args[:3] == ("rm", "-f", "-v"):
            removed.append(names[args[3]])
            return ""
        raise AssertionError(args)

    monkeypatch.setattr(runner, "_docker", docker)
    monkeypatch.setattr(runner, "uuid", SimpleNamespace(uuid4=lambda: SimpleNamespace(hex="test")))
    with pytest.raises(RuntimeError, match="simulated start failure"):
        with runner.spark_services():
            pytest.fail("All targets must be ready before benchmark execution")
    assert removed == ["wave11-d6-qdrant", "wave11-d6-pgvector", "wave11-d6-opensearch"]



def test_report_evaluation_records_quality_and_raw_replica_control(control_fixture):
    control, claims, query, docs, doc_ids = control_fixture
    qids = ("q1", "q2")
    report = np.stack((query, docs[0][0]))
    corpus = SimpleNamespace(
        report_ids=qids, report=(report, None), qrels={"q1": {doc_ids[0]: 1}, "q2": {doc_ids[-1]: 2}},
    )
    schedule = list(runner.epochs_with_lag(doc_ids, control.fence))
    _, pinned, lag = schedule[0]
    oracle = runner._exact_oracle(control, docs)
    fresh = runner.evaluate(control, claims, corpus, 0, pinned, lag, oracle=oracle)
    _, pinned, lag = schedule[2]
    retracted = runner.evaluate(control, claims, corpus, 0, pinned, lag, oracle=oracle)
    assert fresh["report_queries"] == 2
    assert set(fresh["quality"]["ndcg_at_10"]["means"]) == set(MERGES)
    assert len(fresh["quality"]["recall_at_10"]["paired_bootstrap_wave6"]) == 3
    assert fresh["initial_shortlist_quality"]
    assert set(retracted["service_rtt"]) == {"p50_ms", "p95_ms", "p99_ms"}
    assert retracted["admitted_recall_at_10"] == 1.0
    assert retracted["stale_resurrection_rate"] == retracted["raw_engine_stale_hit_rate"] == 0.0
    assert retracted["lagged_replica_only_raw_hits"] == 40


def test_reverse_tra1_route_projects_to_legacy_shard(control_fixture):
    forward, _, _, docs, _ = control_fixture
    legacy, native = forward.shards
    rng = np.random.default_rng(42)
    paired_old = rng.normal(size=(70, 8)).astype(np.float32)
    paired_native = np.pad(paired_old, ((0, 0), (0, 4)))
    bridge = CenteredProcrustesBridge.fit(paired_native, paired_old, scale_mode="unit")
    source, index, contract, evidence = build_tra1_route_bundle(
        index_id=legacy.backend.name, source_encoder_id="native", source_dim=12,
        target_encoder_id="legacy", target_dim=8, bridge=bridge, corpus_snapshot_id="offline",
        document_count=legacy.sidecar.n_docs,
    )
    reg = GatewayRouteRegistration(
        route_key="reverse", index_generation="reverse_gen", query_space=source, resident_index=index,
        bridge_contract=contract, bridge=bridge, trusted_evidence_sha256s=frozenset({evidence}),
        trusted_bridge_contract_ids=frozenset({contract.stable_id}), primary_sidecar=legacy.sidecar,
        secondary_sidecar=native.sidecar,
    )
    control = ChelatedControlPlane(legacy, native, reg, forward.fence)
    claims = RouteClaims(source.stable_id, contract.stable_id, bridge.weights_sha256, "reverse_gen")
    raw = docs[1][42]
    old_query, native_query = control.prepare_query(raw, claims)
    assert old_query.shape == (8,) and native_query.shape == (12,)
    np.testing.assert_allclose(native_query, raw, atol=1e-6)
    assert all(len(rows) == 10 for rows in control.search(raw, claims, epoch=1).merges.values())



def test_opensearch_exact_search_after_pages_without_vector_egress(monkeypatch):
    backend = LiveOpenSearchKnnBackend("exact_pages", 3)
    calls = []

    def request(method, path, *, json):
        assert method == "POST" and path == "/exact_pages/_search"
        assert json["_source"]["excludes"] == ["embedding"]
        calls.append((json["size"], json.get("search_after")))
        start = 0 if "search_after" not in json else json["search_after"][1] + 1
        return {"timed_out": False, "_shards": {"failed": 0}, "hits": {"hits": [
            {"_source": {"doc_id": str(i), "point_id": i, "admitted_epoch": 1},
             "_score": 1.0, "sort": [1.0, i]}
            for i in range(start, start + json["size"])
        ]}}

    monkeypatch.setattr(backend, "_request", request)
    page = backend.search(np.ones(3), allowed=None, epoch=1, limit=10_003, ef=512, exact=True)
    assert page.exact_prefix and len(page.hits) == 10_003
    assert calls == [(10_000, None), (3, [1.0, 9999])]
    assert [h.point_id for h in page.hits[-3:]] == [10_000, 10_001, 10_002]
    backend.close()


def test_qdrant_admits_point_ids_with_no_vectors_or_payload():
    from qdrant_client import models

    from chelated_multibackend_appliance import LiveQdrantBackend

    calls = []

    def query_points(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(points=[SimpleNamespace(id=4, vector=None, payload=None, score=-0.25)])

    backend = LiveQdrantBackend.__new__(LiveQdrantBackend)
    backend.name, backend.dim, backend.models = "qdrant_wire", 3, models
    backend.doc_ids, backend.epoch = ("a", "b", "c", "d", "e"), 2
    backend.client = SimpleNamespace(query_points=query_points)
    q = np.array([1.0, 0.0, 0.0])
    for exact in (False, True):
        page = backend.search(q, allowed=(2, 4), epoch=2, limit=2, ef=128, exact=exact)
        assert page.exact_prefix is exact and page.hits == (VectorHit("e", 4, 2, -0.25),)
        call = calls[-1]
        assert call["query_filter"].must[0].has_id == [2, 4]
        assert not call["with_vectors"] and not call["with_payload"]
        assert call["search_params"].exact is exact
        if exact:
            assert call["search_params"].quantization.ignore
    assert not backend.search(q, allowed=(4,), epoch=1, limit=1, ef=128, exact=True).hits
    assert len(calls) == 2

