"""Unit & live-gRPC integration tests for `chelated_grpc_gateway` (Wave-6 Track E / Demo D5, Guardrail K1)."""

from __future__ import annotations

import json
import threading
import time
from typing import Tuple

import grpc
import numpy as np
import pytest

from chelated_grpc_gateway import (
    BACKEND_EXECUTION_LABEL,
    EP_ADMITTED,
    SCORING_BIDIR_ZEROMU,
    SCORING_BIDIR_ZEROMU_CSLS,
    SCORING_NATIVE_DOT,
    ChelatedGatewayEngine,
    EpochPinnedEvidenceFence,
    GatewayRouteRegistration,
    NumpyInMemoryAnnBackend,
    TRA1AuthorizationError,
    build_shard_sidecar,
    build_tra1_route_bundle,
    compile_opensearch_knn_proto,
    compute_adaptive_ef,
    compute_unseen_score_upper_bound,
    pack_admitted_bitset,
    run_open_loop_dispatch,
    start_chelated_grpc_server,
    unpack_admitted_bitset,
)
from m1_baseline_ladder import fit_closed_form_centered_ridge, fit_closed_form_svd_procrustes
from spherical_migration_bridge import CenteredProcrustesBridge, l2_normalize


def _make_anisotropic_pair(
    n_docs: int = 240,
    n_queries: int = 48,
    d_src: int = 64,
    d_tgt: int = 256,
    seed: int = 17,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate paired source/target embeddings on a strong anisotropic cone with variable centered norms."""
    rng = np.random.default_rng(seed)
    latent_docs = rng.normal(size=(n_docs, d_src))
    latent_queries = latent_docs[:n_queries] + 0.18 * rng.normal(size=(n_queries, d_src))

    # Orthonormal embedding from d_src into d_tgt
    rand_mat = rng.normal(size=(d_tgt, d_src))
    q_orth, _ = np.linalg.qr(rand_mat)  # [d_tgt, d_src]
    rot_src, _ = np.linalg.qr(rng.normal(size=(d_src, d_src)))

    mu_src = np.zeros(d_src, dtype=np.float64)
    mu_src[0] = 1.15
    docs_src = l2_normalize(0.55 * l2_normalize(latent_docs @ rot_src) + mu_src).astype(np.float32)
    queries_src = l2_normalize(0.55 * l2_normalize(latent_queries @ rot_src) + mu_src).astype(np.float32)

    # Target cone with strong shared mean + heterogeneous per-doc radius -> dispersion in ||d_j - mu_d||_2
    cone_pole = np.zeros(d_tgt, dtype=np.float64)
    cone_pole[0] = 1.0
    radii = rng.uniform(0.35, 1.35, size=(n_docs, 1))
    docs_tgt_unnorm = cone_pole[None, :] * 1.30 + radii * (l2_normalize(latent_docs) @ q_orth.T)
    docs_tgt = l2_normalize(docs_tgt_unnorm).astype(np.float32)
    queries_tgt = l2_normalize(cone_pole[None, :] * 1.30 + 0.75 * (l2_normalize(latent_queries) @ q_orth.T)).astype(
        np.float32
    )
    return docs_src, queries_src, docs_tgt, queries_tgt


def test_protoc_compiles_binary_opensearch_knn_contract() -> None:
    bundle = compile_opensearch_knn_proto()
    assert bundle.protoc_exit_code == 0
    assert len(bundle.proto_sha256) == 64

    vec = np.linspace(-0.5, 0.5, num=4096, dtype=np.float32)
    bitset = pack_admitted_bitset(np.ones(5183, dtype=bool))
    req = bundle.pb2.SearchRequest(
        indices=["d5_scifact_4096"],
        knn=bundle.pb2.KnnQuery(
            field="embedding",
            vector=vec.tolist(),
            k=10,
            k_os=64,
            ef_search=128,
            admitted_bitset=bitset,
            epoch_pin=7,
            query_id="q_001",
        ),
        tenant_id="tenant-a",
        query_space_id="rep:v1:sha256:" + "a" * 64,
        bridge_contract_id="bridge:v1:sha256:" + "b" * 64,
        bridge_weights_sha256="c" * 64,
        index_generation="gen-1",
        epoch_pin=7,
        wait_for_epoch=True,
        scoring_mode=SCORING_BIDIR_ZEROMU_CSLS,
        enable_certified_stop=True,
    )
    wire = req.SerializeToString()
    json_wire = json.dumps({"vector": vec.tolist(), "k": 10, "ef_search": 128}).encode("utf-8")
    # Packed float32 is 4 * 4096 = 16,384 B + small headers + 648 B bitset (< 18 KB vs > 80 KB JSON)
    assert 16384 < len(wire) < 18000
    assert len(json_wire) > 75000

    decoded = bundle.pb2.SearchRequest.FromString(wire)
    assert decoded.knn.k == 10
    assert decoded.knn.ef_search == 128
    assert decoded.knn.epoch_pin == 7
    unpacked = unpack_admitted_bitset(decoded.knn.admitted_bitset, 5183)
    assert unpacked.shape == (5183,) and bool(np.all(unpacked))


def test_sidecar_exact_8_bytes_per_doc_and_algebraic_reconciliation() -> None:
    docs_src, queries_src, docs_tgt, _queries_tgt = _make_anisotropic_pair()
    doc_ids = [f"doc_{i:04d}" for i in range(docs_tgt.shape[0])]
    fit_idx = np.arange(80, 220, dtype=np.int64)

    bridge = CenteredProcrustesBridge.fit(docs_src, docs_tgt, fit_idx=fit_idx, scale_mode="unit")
    dev_q_hat = bridge.transform(queries_src[:16], normalize=True, drop_mu_y=True).astype(np.float32)
    sidecar = build_shard_sidecar(
        shard_tag="bridged_256d",
        collection_name="col_256",
        doc_ids=doc_ids,
        docs_raw=docs_tgt,
        dev_queries_in_shard_space=dev_q_hat,
        csls_k=10,
        zero_mu_active=True,
    )
    assert sidecar.sidecar_bytes_per_doc == 8
    assert sidecar.total_sidecar_bytes == 8 * docs_tgt.shape[0]
    assert not hasattr(sidecar, "docs_raw")

    # Verify exact float32 ZeroMu identity vs dense centered cosine
    eval_q = queries_src[16:48]
    q_hats = bridge.transform(eval_q, normalize=True, drop_mu_y=True).astype(np.float32)
    mu_d = docs_tgt.mean(axis=0, dtype=np.float64).astype(np.float32)
    dense_centered_docs = (docs_tgt - mu_d[None, :]) / np.linalg.norm(docs_tgt - mu_d[None, :], axis=1, keepdims=True)
    exact_zeromu = q_hats @ dense_centered_docs.T

    all_pids = np.arange(docs_tgt.shape[0], dtype=np.int64)
    for row in range(q_hats.shape[0]):
        raw_dots = docs_tgt @ q_hats[row]
        rescored, _ = sidecar.rescore_from_raw_dots(all_pids, raw_dots, q_hats[row])
        assert float(np.max(np.abs(rescored - exact_zeromu[row]))) < 1e-5

    # Algebraic reconciliation: RequestOnly_RawDOT (no +mu_y) vs Procrustes vs QueryAffine (+mu_y) vs ZeroMu
    stiefel_w = fit_closed_form_svd_procrustes(docs_src, docs_tgt, fit_idx)
    mu_x_a, mu_y_a, w_aff = fit_closed_form_centered_ridge(docs_src, docs_tgt, fit_idx, regularization=1e-2)

    scores_req_only = q_hats @ docs_tgt.T
    # Adding constant q_hat·mu_d does NOT change per-query ranking at all!
    scores_req_only_shifted = scores_req_only - (q_hats @ mu_d)[:, None]
    for row in range(q_hats.shape[0]):
        assert np.array_equal(np.argsort(-scores_req_only[row]), np.argsort(-scores_req_only_shifted[row]))

    scores_proc = l2_normalize(eval_q @ stiefel_w).astype(np.float32) @ docs_tgt.T
    scores_aff = l2_normalize((eval_q - mu_x_a) @ w_aff + mu_y_a).astype(np.float32) @ docs_tgt.T

    gold = np.arange(16, 48, dtype=np.int64)
    rec_zeromu = float(np.mean([gold[i] in np.argsort(-exact_zeromu[i])[:5] for i in range(len(gold))]))
    rec_req_only = float(np.mean([gold[i] in np.argsort(-scores_req_only[i])[:5] for i in range(len(gold))]))
    rec_proc = float(np.mean([gold[i] in np.argsort(-scores_proc[i])[:5] for i in range(len(gold))]))
    rec_aff = float(np.mean([gold[i] in np.argsort(-scores_aff[i])[:5] for i in range(len(gold))]))

    assert rec_zeromu >= rec_req_only >= rec_aff
    assert abs(rec_req_only - rec_proc) < 0.25


def test_tra1_ingress_fail_closed_and_causal_ema() -> None:
    docs_src, queries_src, docs_tgt, _ = _make_anisotropic_pair(d_src=32, d_tgt=64)
    doc_ids = [f"d{i}" for i in range(docs_tgt.shape[0])]

    # 1. Fit count < 32 must fail closed in TRA-1
    tiny_bridge = CenteredProcrustesBridge.fit(docs_src, docs_tgt, fit_idx=np.arange(16), scale_mode="unit")
    q_space_t, res_idx_t, contract_t, ev_sha_t = build_tra1_route_bundle(
        index_id="idx-tiny",
        source_encoder_id="enc-src",
        source_dim=32,
        target_encoder_id="enc-tgt",
        target_dim=64,
        bridge=tiny_bridge,
        corpus_snapshot_id="snap-1",
        document_count=len(doc_ids),
        validation_count=20,
    )
    sidecar = build_shard_sidecar(
        shard_tag="s64",
        collection_name="c64",
        doc_ids=doc_ids,
        docs_raw=docs_tgt,
    )
    bad_route = GatewayRouteRegistration(
        route_key="idx-tiny",
        index_generation="g1",
        query_space=q_space_t,
        resident_index=res_idx_t,
        bridge_contract=contract_t,
        bridge=tiny_bridge,
        trusted_evidence_sha256s=frozenset({ev_sha_t}),
        trusted_bridge_contract_ids=frozenset({contract_t.stable_id}),
        primary_sidecar=sidecar,
    )
    with pytest.raises(TRA1AuthorizationError, match="TRA1_ABSTAIN_FAIL_CLOSED"):
        bad_route.verify_tra1_ingress(
            claimed_query_space_id=q_space_t.stable_id,
            claimed_bridge_contract_id=contract_t.stable_id,
            claimed_weights_sha256=tiny_bridge.weights_sha256,
            claimed_index_generation="g1",
        )

    # 2. Valid route passes TRA-1, rejects tampered SHA / generation / nullspace vector, and runs causal EMA
    # Build rank-deficient Q to test retained-subspace energy guard before normalization
    valid_bridge = CenteredProcrustesBridge.fit(
        docs_src,
        docs_tgt,
        fit_idx=np.arange(64),
        scale_mode="unit",
        rank_cap=16,
    )
    q_space, res_idx, contract, ev_sha = build_tra1_route_bundle(
        index_id="idx-valid",
        source_encoder_id="enc-src",
        source_dim=32,
        target_encoder_id="enc-tgt",
        target_dim=64,
        bridge=valid_bridge,
        corpus_snapshot_id="snap-1",
        document_count=len(doc_ids),
        validation_count=24,
    )
    route = GatewayRouteRegistration(
        route_key="idx-valid",
        index_generation="gen-42",
        query_space=q_space,
        resident_index=res_idx,
        bridge_contract=contract,
        bridge=valid_bridge,
        trusted_evidence_sha256s=frozenset({ev_sha}),
        trusted_bridge_contract_ids=frozenset({contract.stable_id}),
        primary_sidecar=sidecar,
        ema_alpha=0.10,
        min_retained_energy_ratio=0.15,
    )
    decision = route.verify_tra1_ingress(
        claimed_query_space_id=q_space.stable_id,
        claimed_bridge_contract_id=contract.stable_id,
        claimed_weights_sha256=valid_bridge.weights_sha256,
        claimed_index_generation="gen-42",
    )
    assert decision.mode.value == "reverse"

    with pytest.raises(TRA1AuthorizationError, match="BRIDGE_WEIGHTS_SHA256_MISMATCH"):
        route.verify_tra1_ingress(
            claimed_query_space_id=q_space.stable_id,
            claimed_bridge_contract_id=contract.stable_id,
            claimed_weights_sha256="f" * 64,
            claimed_index_generation="gen-42",
        )

    # Vector purely in the nullspace of Q must fail retained-subspace energy guard BEFORE normalization
    _u, _sv, vt = np.linalg.svd(valid_bridge.Q.T, full_matrices=True)
    null_dir = vt[-1].astype(np.float32)  # orthogonal to column space of Q.T == row space of Q
    null_query = valid_bridge.mu_x.reshape(-1).astype(np.float32) + null_dir
    with pytest.raises(TRA1AuthorizationError, match="LOW_RETAINED_SUBSPACE_ENERGY"):
        route.project_query_causal(null_query, scoring_mode=SCORING_BIDIR_ZEROMU, update_ema=False)

    # Strictly causal EMA: projecting q0 first must NOT depend on q0 in mu_q, then updates mu_q for q1
    route.reset_causal_ema()
    q0 = queries_src[0]
    expected_q0 = l2_normalize((q0.astype(np.float64) - valid_bridge.mu_x) @ valid_bridge.Q).reshape(-1).astype(np.float32)
    got_q0, ratio0 = route.project_query_causal(q0, scoring_mode=SCORING_BIDIR_ZEROMU, update_ema=True)
    assert ratio0 >= 0.15
    assert float(np.max(np.abs(got_q0 - expected_q0))) < 1e-6

    q1 = queries_src[1]
    mu_after_q0 = 0.90 * valid_bridge.mu_x.reshape(-1).astype(np.float32) + 0.10 * q0
    expected_q1 = l2_normalize((q1 - mu_after_q0).astype(np.float64) @ valid_bridge.Q).reshape(-1).astype(np.float32)
    got_q1, _ = route.project_query_causal(q1, scoring_mode=SCORING_BIDIR_ZEROMU, update_ema=True)
    assert float(np.max(np.abs(got_q1 - expected_q1))) < 1e-6


def test_certified_stop_both_norm_endpoints_and_refill() -> None:
    # Negative numerator (dot_bound < c_shift) must maximize at n_max (less negative), NOT n_min!
    ub_neg = compute_unseen_score_upper_bound(
        dot_upper_bound=0.10,
        c_shift=0.30,
        n_min=0.50,
        n_max=1.00,
        use_csls=False,
    )
    assert abs(ub_neg - (-0.20)) < 1e-9  # (-0.20 / 1.00) == -0.20 > (-0.20 / 0.50 == -0.40)

    ub_pos = compute_unseen_score_upper_bound(
        dot_upper_bound=0.50,
        c_shift=0.30,
        n_min=0.50,
        n_max=1.00,
        use_csls=True,
        r_d_min=0.12,
        r_q_lower_bound=0.25,
    )
    # z_ub = 0.20 / 0.50 = 0.40 -> csls_ub = 2*0.40 - 0.12 - 0.25 = 0.43
    assert abs(ub_pos - 0.43) < 1e-9


def test_ek5_v3_epoch_fence_wait_true_contested_and_replica_lag() -> None:
    # Adaptive ef formula check
    ef_90_admit, ex_90 = compute_adaptive_ef(64, 64, 0.90, ef_max=256)
    ef_20_admit, ex_20 = compute_adaptive_ef(64, 64, 0.20, ef_max=256)
    ef_08_admit, ex_08 = compute_adaptive_ef(64, 64, 0.08, ef_max=256)
    assert ef_90_admit == 72 and ex_90 is False
    assert ef_20_admit == 256 and ex_20 is False
    assert ef_08_admit == 256 and ex_08 is True

    docs_src, queries_src, docs_tgt, _ = _make_anisotropic_pair(n_docs=100, n_queries=16, d_src=32, d_tgt=64)
    doc_ids = [f"doc_{i:03d}" for i in range(100)]
    bridge = CenteredProcrustesBridge.fit(docs_src, docs_tgt, fit_idx=np.arange(40, 95), scale_mode="unit")
    q_space, res_idx, contract, ev_sha = build_tra1_route_bundle(
        index_id="idx-fence",
        source_encoder_id="enc-32",
        source_dim=32,
        target_encoder_id="enc-64",
        target_dim=64,
        bridge=bridge,
        corpus_snapshot_id="snap-fence",
        document_count=100,
        validation_count=20,
    )
    sidecar = build_shard_sidecar(
        shard_tag="s64",
        collection_name="col_fence",
        doc_ids=doc_ids,
        docs_raw=docs_tgt,
    )
    backend = NumpyInMemoryAnnBackend()
    # Replica lags behind: replica thinks ALL 100 docs are still admitted!
    backend.register_collection("col_fence", docs_tgt, replica_admitted=np.ones(100, dtype=bool))

    fence = EpochPinnedEvidenceFence(doc_ids, initial_epoch=1)
    engine = ChelatedGatewayEngine(
        ann_backend=backend,
        evidence_fence=fence,
        use_replica_payload_prefilter=True,
    )
    route = GatewayRouteRegistration(
        route_key="idx-fence",
        index_generation="g1",
        query_space=q_space,
        resident_index=res_idx,
        bridge_contract=contract,
        bridge=bridge,
        trusted_evidence_sha256s=frozenset({ev_sha}),
        trusted_bridge_contract_ids=frozenset({contract.stable_id}),
        primary_sidecar=sidecar,
    )
    engine.register_route(route)

    # Commit epoch 2 asynchronously while reader blocks with wait_for_epoch=True on epoch_pin=2:
    # - doc_099 supersedes doc_000..doc_009 (AUDIT_ONLY)
    # - doc_010 <-> doc_011 mutual peer REFUTE cycle (CONTESTED)
    def _delayed_writer() -> None:
        time.sleep(0.05)
        atts = [("doc_099", f"doc_{i:03d}", "SUPERSEDE") for i in range(10)]
        atts.extend([("doc_010", "doc_011", "REFUTE"), ("doc_011", "doc_010", "REFUTE")])
        fence.commit_mutation(add_attestations=atts, target_epoch=2)

    writer_thread = threading.Thread(target=_delayed_writer)
    writer_thread.start()

    bundle = compile_opensearch_knn_proto()
    # Craft query aligned with doc_010 / doc_000 neighborhood
    knn_msg = bundle.pb2.KnnQuery(
        field="embedding",
        vector=docs_src[0].tolist(),
        k=10,
        k_os=32,
        ef_search=64,
        epoch_pin=2,
        query_id="q_fence",
    )
    res = engine.execute_single_query(
        route=route,
        knn_msg=knn_msg,
        scoring_mode=SCORING_BIDIR_ZEROMU,
        epoch_pin=2,
        wait_for_epoch=True,
        enable_certified_stop=True,
        dual_shard_fanout=False,
        adaptive_ef=True,
        allow_exact_fallback=True,
        update_causal_ema=False,
        require_exact_prefix_certificate=True,
    )
    writer_thread.join()

    assert res["fenced_epoch"] == 2
    assert res["certified_stop"] is True
    returned_ids = {h[0] for h in res["hits"]}
    retracted_ids = {f"doc_{i:03d}" for i in range(10)}
    contested_pair = {"doc_010", "doc_011"}
    # Zero stale resurrection and zero contested leakage into ADMITTED hits!
    assert len(returned_ids & retracted_ids) == 0
    assert len(returned_ids & contested_pair) == 0

    # Query directly at doc_010 to verify CONTESTED is surfaced in contested_hits
    knn_contested = bundle.pb2.KnnQuery(
        field="embedding",
        vector=docs_src[10].tolist(),
        k=10,
        k_os=32,
        ef_search=64,
        epoch_pin=2,
        query_id="q_contested",
    )
    res_c = engine.execute_single_query(
        route=route,
        knn_msg=knn_contested,
        scoring_mode=SCORING_BIDIR_ZEROMU,
        epoch_pin=2,
        wait_for_epoch=True,
        enable_certified_stop=True,
        dual_shard_fanout=False,
        adaptive_ef=True,
        allow_exact_fallback=True,
        update_causal_ema=False,
        require_exact_prefix_certificate=False,
    )
    contested_returned = {h[0] for h in res_c["contested_hits"]}
    assert "doc_010" in contested_returned or "doc_011" in contested_returned
    engine.close()


def test_live_grpcio_server_dual_shard_csls_and_batch32() -> None:
    docs_384, queries_384, docs_4096, queries_4096 = _make_anisotropic_pair(
        n_docs=160, n_queries=40, d_src=48, d_tgt=192, seed=99
    )
    doc_ids = [f"doc_{i:04d}" for i in range(160)]
    fit_idx = np.arange(50, 150, dtype=np.int64)

    # Mode B: 192-d query -> 48-d legacy shard (`legacy_384d` role) + 192-d native shard (`native_4096d` role)
    bridge_down = CenteredProcrustesBridge.fit(docs_4096, docs_384, fit_idx=fit_idx, scale_mode="unit")
    q_space, res_idx, contract, ev_sha = build_tra1_route_bundle(
        index_id="idx-dual",
        source_encoder_id="qwen-192",
        source_dim=192,
        target_encoder_id="minilm-48",
        target_dim=48,
        bridge=bridge_down,
        corpus_snapshot_id="snap-dual",
        document_count=160,
        validation_count=32,
    )

    # Partition 80% in legacy shard, 20% in native shard (disjoint physical shards!)
    legacy_mask = np.ones(160, dtype=bool)
    legacy_mask[:32] = False
    native_mask = ~legacy_mask

    dev_q_192 = queries_4096[:16]
    dev_q_48_hat = bridge_down.transform(dev_q_192, normalize=True, drop_mu_y=True).astype(np.float32)

    legacy_sidecar = build_shard_sidecar(
        shard_tag="legacy_384d",
        collection_name="col_legacy",
        doc_ids=[doc_ids[i] for i in np.flatnonzero(legacy_mask)],
        docs_raw=docs_384[legacy_mask],
        dev_queries_in_shard_space=dev_q_48_hat,
        csls_k=8,
        zero_mu_active=True,
    )
    native_sidecar = build_shard_sidecar(
        shard_tag="native_4096d",
        collection_name="col_native",
        doc_ids=[doc_ids[i] for i in np.flatnonzero(native_mask)],
        docs_raw=docs_4096[native_mask],
        dev_queries_in_shard_space=dev_q_192,
        csls_k=8,
        zero_mu_active=False,
    )

    backend = NumpyInMemoryAnnBackend()
    backend.register_collection("col_legacy", docs_384[legacy_mask])
    backend.register_collection("col_native", docs_4096[native_mask])

    fence = EpochPinnedEvidenceFence(doc_ids, initial_epoch=1)
    engine = ChelatedGatewayEngine(ann_backend=backend, evidence_fence=fence)
    route = GatewayRouteRegistration(
        route_key="idx-dual",
        index_generation="gen-dual-1",
        query_space=q_space,
        resident_index=res_idx,
        bridge_contract=contract,
        bridge=bridge_down,
        trusted_evidence_sha256s=frozenset({ev_sha}),
        trusted_bridge_contract_ids=frozenset({contract.stable_id}),
        primary_sidecar=legacy_sidecar,
        secondary_sidecar=native_sidecar,
        secondary_query_is_bridged=False,
    )
    engine.register_route(route)

    # Also register single-shard native route for batch=32 fast path test
    native_q_space = res_idx.query_space
    full_48_sidecar = build_shard_sidecar(
        shard_tag="single_48d",
        collection_name="col_legacy",
        doc_ids=legacy_sidecar.doc_ids,
        docs_raw=docs_384[legacy_mask],
        dev_queries_in_shard_space=queries_384[:16],
        csls_k=8,
        zero_mu_active=False,
    )
    single_route = GatewayRouteRegistration(
        route_key="idx-single",
        index_generation="gen-single-1",
        query_space=native_q_space,
        resident_index=res_idx,
        bridge_contract=None,
        bridge=None,
        trusted_evidence_sha256s=frozenset(),
        trusted_bridge_contract_ids=frozenset(),
        primary_sidecar=full_48_sidecar,
    )
    engine.register_route(single_route)

    server, port, bundle = start_chelated_grpc_server(engine, bind_address="127.0.0.1:0", max_workers=4)
    try:
        channel = grpc.insecure_channel(f"127.0.0.1:{port}")
        stub = bundle.pb2_grpc.SearchServiceStub(channel)

        # 1. Dual-shard HTTP/2 fanout with CSLS + Certified Stop
        req = bundle.pb2.SearchRequest(
            indices=["idx-dual"],
            knn=bundle.pb2.KnnQuery(
                field="embedding",
                vector=queries_4096[0].tolist(),
                k=10,
                k_os=32,
                ef_search=64,
                epoch_pin=1,
                query_id="q_dual_0",
            ),
            tenant_id="t1",
            query_space_id=q_space.stable_id,
            bridge_contract_id=contract.stable_id,
            bridge_weights_sha256=bridge_down.weights_sha256,
            index_generation="gen-dual-1",
            epoch_pin=1,
            wait_for_epoch=True,
            scoring_mode=SCORING_BIDIR_ZEROMU_CSLS,
            enable_certified_stop=True,
            dual_shard_fanout=True,
            adaptive_ef=True,
            allow_exact_fallback=True,
            update_causal_ema=True,
            require_exact_prefix_certificate=True,
        )
        resp = stub.Search(req, timeout=5.0)
        assert resp.backend_execution_label == BACKEND_EXECUTION_LABEL
        assert resp.certified_stop is True
        assert len(resp.hits) == 10
        assert all(h.epistemic_state == EP_ADMITTED for h in resp.hits)
        assert {h.shard_tag for h in resp.hits}.issubset({"legacy_384d", "native_4096d"})

        # 2. Tampered weights digest must fail closed with FAILED_PRECONDITION over gRPC
        bad_req = bundle.pb2.SearchRequest()
        bad_req.CopyFrom(req)
        bad_req.bridge_weights_sha256 = "0" * 64
        with pytest.raises(grpc.RpcError) as exc_info:
            stub.Search(bad_req, timeout=5.0)
        assert exc_info.value.code() == grpc.StatusCode.FAILED_PRECONDITION

        # 3. Batch=32 RPC test + Open-Loop dispatcher smoke check
        batch_knns = [
            bundle.pb2.KnnQuery(
                field="embedding",
                vector=queries_384[i % queries_384.shape[0]].tolist(),
                k=10,
                k_os=32,
                ef_search=32,
                epoch_pin=1,
                query_id=f"qb_{i}",
            )
            for i in range(32)
        ]
        breq = bundle.pb2.SearchRequest(
            indices=["idx-single"],
            knn_batch=batch_knns,
            tenant_id="t1",
            query_space_id=native_q_space.stable_id,
            index_generation="gen-single-1",
            epoch_pin=1,
            wait_for_epoch=True,
            scoring_mode=SCORING_NATIVE_DOT,
        )
        bresp = stub.Search(breq, timeout=5.0)
        assert len(bresp.batch_results) == 32
        assert all(len(r.hits) == 10 for r in bresp.batch_results)

        ol_stats = run_open_loop_dispatch(
            n_dispatches=12,
            target_rate_hz=200.0,
            worker_count=4,
            dispatch_fn=lambda i: {
                "serialize_cpu_ms": 0.05,
                "ann_rtt_ms": 0.20,
                "egress_rescore_ms": 0.02,
                "req_wire_bytes": float(len(breq.SerializeToString())),
                "resp_wire_bytes": float(len(stub.Search(breq, timeout=5.0).SerializeToString())),
            },
        )
        assert ol_stats["error_count"] == 0, ol_stats["error_samples"]
        assert ol_stats["n_dispatches"] == 12
        channel.close()
    finally:
        server.stop(grace=0.2)
        engine.close()
