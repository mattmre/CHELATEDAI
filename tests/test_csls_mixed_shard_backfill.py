"""Synthetic checks for the D2 ZeroMu sidecar, Stiefel map, and CSLS mixed-shard merge."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "run_r6_pm3_and_d2_qdrant_csls_backfill.py"
SPEC = importlib.util.spec_from_file_location("run_r6_pm3_and_d2_qdrant_csls_backfill", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def test_cone_anisotropy_grows_when_the_mean_dominates() -> None:
    rng = np.random.default_rng(0)
    isotropic = rng.normal(size=(80, 16))
    cone = np.ones((80, 16)) * 5.0 + 0.01 * rng.normal(size=(80, 16))
    iso_stats = mod.cone_anisotropy(isotropic)
    cone_stats = mod.cone_anisotropy(cone)
    assert cone_stats["mu_l2"] > iso_stats["mu_l2"]
    assert cone_stats["centered_norm_std"] < iso_stats["centered_norm_std"]
    assert cone_stats["anisotropy_ratio_mu_over_mean_radius"] > 10.0


def test_uncentered_stiefel_satisfies_w_wt_identity() -> None:
    rng = np.random.default_rng(1)
    source = rng.normal(size=(40, 6))
    target = rng.normal(size=(40, 11))
    fit_idx = np.arange(30)
    weight = mod.fit_uncentered_stiefel(source, target, fit_idx)
    assert weight.shape == (6, 11)
    gram = weight @ weight.T
    assert np.allclose(gram, np.eye(6), atol=1e-6)


def test_affine_scorer_matches_cosine_and_does_not_divide_by_centered_norm() -> None:
    rng = np.random.default_rng(2)
    source = rng.normal(size=(25, 4))
    target = rng.normal(size=(25, 5))
    target[:, 0] += 8.0
    fit_idx = np.arange(18)
    mu_x, mu_y, weight = mod.fit_query_only_affine(source, target, fit_idx)
    queries = source[18:]
    scores = mod.score_query_only_affine(queries, mu_x, mu_y, weight, target)
    mapped = (queries - mu_x) @ weight + mu_y
    assert np.allclose(scores, mod.cosine_matrix(mapped, target))
    mu_d, norms = mod.build_zeromu_sidecar(target)
    hat = mod.l2_normalize(mapped[:1])[0]
    divided = (target @ hat - float(mu_d.reshape(-1) @ hat)) / norms
    assert not np.allclose(scores[0], divided)


def test_causal_ema_excludes_the_current_query_from_mu_q() -> None:
    rng = np.random.default_rng(3)
    queries = rng.normal(size=(2, 4)).astype(np.float32)
    ortho = np.zeros((4, 5), dtype=np.float64)
    ortho[:4, :4] = np.eye(4)
    mu_x = np.zeros((1, 4), dtype=np.float64)
    _hats, mu_before = mod.causal_q_hats(queries, ortho, mu_x, ema_alpha=0.5)
    assert np.allclose(mu_before[0], mu_x.reshape(-1))
    expected_second = 0.5 * mu_x.reshape(-1) + 0.5 * queries[0]
    assert np.allclose(mu_before[1], expected_second)


def test_float32_sidecar_roundtrip_matches_brute_force() -> None:
    rng = np.random.default_rng(4)
    docs = rng.normal(size=(12, 7)).astype(np.float32)
    docs += np.array([3.0, 0, 0, 0, 0, 0, 0], dtype=np.float32)
    mu_d, norms = mod.build_zeromu_sidecar(docs)
    assert norms.dtype == np.float32
    assert norms.nbytes / norms.shape[0] == 4
    hat = mod.l2_normalize(rng.normal(size=(1, 7)))[0].astype(np.float32)
    got = mod.zeromu_scores_from_qhats(hat.reshape(1, -1), docs, mu_d, norms)[0]
    brute = (docs @ hat - float(mu_d.reshape(-1) @ hat)) / norms
    assert np.allclose(got, brute, atol=1e-5)


def test_csls_changes_cross_shard_ranking_versus_raw_merge() -> None:
    bridged = np.array([[0.95, 0.10]], dtype=np.float64)
    native = np.array([[0.80]], dtype=np.float64)
    mask = np.array([False, False, True])
    raw_bridged = np.concatenate([bridged, np.zeros((1, 1))], axis=1)
    raw_native = np.concatenate([np.zeros((1, 2)), native], axis=1)
    raw = mod.mixed_shard_scores(raw_bridged, raw_native, mask, calibrate=False)
    assert int(np.argmax(raw[0])) == 0
    bridged_ref = np.array(
        [
            [0.99, 0.01],
            [0.98, 0.02],
            [0.97, 0.00],
        ],
        dtype=np.float64,
    )
    native_ref = np.array([[0.20], [0.15], [0.10]], dtype=np.float64)
    ref_bridged = np.concatenate([bridged_ref, np.zeros((3, 1))], axis=1)
    ref_native = np.concatenate([np.zeros((3, 2)), native_ref], axis=1)
    calibrated = mod.mixed_shard_scores(
        raw_bridged,
        raw_native,
        mask,
        bridged_ref=ref_bridged,
        native_ref=ref_native,
        calibrate=True,
        csls_k=3,
    )
    assert int(np.argmax(calibrated[0])) != int(np.argmax(raw[0]))


def test_zero_and_full_backfill_identities() -> None:
    bridged = np.array([[0.2, 0.4, 0.1], [0.3, 0.1, 0.5]], dtype=np.float64)
    native = np.array([[0.9, 0.1, 0.2], [0.2, 0.8, 0.3]], dtype=np.float64)
    none = np.zeros(3, dtype=bool)
    full = np.ones(3, dtype=bool)
    assert np.allclose(mod.mixed_shard_scores(bridged, native, none, calibrate=False), bridged)
    assert np.allclose(mod.mixed_shard_scores(bridged, native, full, calibrate=False), native)


def test_recall_is_exact_when_oversample_covers_the_corpus() -> None:
    rng = np.random.default_rng(5)
    docs = rng.normal(size=(9, 4)).astype(np.float32)
    mu_d, norms = mod.build_zeromu_sidecar(docs)
    hat = mod.l2_normalize(rng.normal(size=4))[0].astype(np.float32)
    scores = mod.zeromu_scores_from_qhats(hat.reshape(1, -1), docs, mu_d, norms)[0]
    doc_ids = [f"d{i}" for i in range(9)]
    exact = mod.topk_doc_ids(scores, doc_ids, 3)
    shortlist = mod.numpy_dot_shortlist(hat, docs, k_os=9)
    rescored = mod.rescore_shortlist(shortlist, hat, docs, mu_d, norms)
    order = np.argsort(-rescored, kind="mergesort")
    approx = [doc_ids[shortlist[int(j)]] for j in order[:3]]
    assert mod.recall_of_exact_topk(exact, approx) == 1.0


def test_abstention_grade_accepts_longmemeval_did_not_mention_phrasing() -> None:
    gold = "You did not mention this information. You mentioned your cat Luna but not your hamster."
    assert mod.grade_prediction("You did not mention a hamster.", gold, abstention=True)
    assert not mod.grade_prediction("The hamster is named Luna.", gold, abstention=True)


def test_message_chunks_keep_every_raw_message() -> None:
    messages = [{"role": "user", "content": "x" * 30, "ts": float(i)} for i in range(5)]
    chunks = mod._message_chunks(messages, max_chars=80)
    flat = [row["content"] for chunk in chunks for row in chunk]
    assert flat == [row["content"] for row in messages]
    assert len(chunks) > 1


def test_qdrant_hnsw_rescore_matches_numpy_when_client_imports() -> None:
    pytest.importorskip("qdrant_client")
    rng = np.random.default_rng(6)
    docs = (rng.normal(size=(15, 8)) + np.array([2.0, 0, 0, 0, 0, 0, 0, 0])).astype(np.float32)
    client, meta = mod.index_raw_vectors_qdrant(docs, collection="unit_raw")
    assert meta["search_engine"] == "qdrant-local-exact"
    assert meta["hnsw_effective"] is False
    try:
        delta = mod.vector_identity_max_abs(client, "unit_raw", docs)
        assert delta < 1e-5
        hat = mod.l2_normalize(rng.normal(size=8))[0].astype(np.float32)
        shortlist, _dots = mod.qdrant_dot_shortlist(client, "unit_raw", hat, k_os=15)
        assert set(shortlist) == set(range(15))
        mu_d, norms = mod.build_zeromu_sidecar(docs)
        rescored = mod.rescore_shortlist(shortlist, hat, docs, mu_d, norms)
        brute = mod.zeromu_scores_from_qhats(hat.reshape(1, -1), docs, mu_d, norms)[0]
        assert np.allclose(rescored, brute[shortlist], atol=1e-5)
    finally:
        client.close()


def test_dev_zscore_and_rrf_mixed_shard_baselines() -> None:
    bridged = np.array([[0.90, 0.20, 0.0]], dtype=np.float64)
    native = np.array([[0.0, 0.0, 0.55]], dtype=np.float64)
    mask = np.array([False, False, True])
    b_ref = np.array([[0.85, 0.80, 0.0], [0.92, 0.88, 0.0]], dtype=np.float64)
    n_ref = np.array([[0.0, 0.0, 0.20], [0.0, 0.0, 0.30]], dtype=np.float64)
    zs = mod.zscore_mixed_shard_scores(bridged, native, mask, b_ref, n_ref)
    rrf = mod.rrf_mixed_shard_scores(bridged, native, mask, rrf_k=60)
    assert zs.shape == (1, 3) and rrf.shape == (1, 3)
    assert int(np.argmax(zs[0])) == 2  # native 0.55 is +6 sigma above its DEV mean 0.25
    assert np.isclose(rrf[0, 0], 1.0 / 61.0) and np.isclose(rrf[0, 2], 1.0 / 61.0)

