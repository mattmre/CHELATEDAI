"""Unit tests for Wave-6 K3 Dual-Mode (`4096->384` Mode B & `384->4096` Mode A) 3-Domain BEIR runner."""

from __future__ import annotations

import numpy as np

from scripts.run_wave6_d2_4096d_both_modes_3beir import (
    check_stiefel_orthogonality,
    evaluate_dual_shard_mode_b,
    evaluate_mode_a_upbridge,
    fit_centered_stiefel_bridge,
    l2_normalize,
)


def test_rectangular_down_and_up_stiefel_orthogonality_and_mode_b_physical_separation() -> None:
    rng = np.random.default_rng(123)
    n_docs = 120
    n_queries = 48
    latent = rng.normal(size=(n_docs + n_queries, 32))
    proj_low = rng.normal(size=(32, 64))
    proj_high = rng.normal(size=(32, 128))
    cone_high = np.zeros(128, dtype=np.float64)
    cone_high[0] = 0.85

    all_low = l2_normalize(latent @ proj_low + 0.05 * rng.normal(size=(n_docs + n_queries, 64))).astype(np.float32)
    all_high = l2_normalize(latent @ proj_high + cone_high + 0.05 * rng.normal(size=(n_docs + n_queries, 128))).astype(np.float32)

    docs_low = all_low[:n_docs]
    queries_low = all_low[n_docs:]
    docs_high = all_high[:n_docs]
    queries_high = all_high[n_docs:]

    fit_idx = np.arange(40, dtype=np.int64)
    w_down, _ = fit_centered_stiefel_bridge(docs_high, docs_low, fit_idx)
    w_up, _ = fit_centered_stiefel_bridge(docs_low, docs_high, fit_idx)
    assert w_down.shape == (128, 64)
    assert w_up.shape == (64, 128)
    assert check_stiefel_orthogonality(w_down) < 1e-12
    assert check_stiefel_orthogonality(w_up) < 1e-12

    doc_ids = [f"d{i:04d}" for i in range(n_docs)]
    query_ids = [f"q{i:04d}" for i in range(n_queries)]
    # Plant ground-truth relevant doc as nearest latent neighbor
    sim_lat = l2_normalize(latent[n_docs:]) @ l2_normalize(latent[:n_docs]).T
    qrels = {query_ids[i]: {doc_ids[int(np.argmax(sim_lat[i]))]: 1.0} for i in range(n_queries)}

    mb = evaluate_dual_shard_mode_b(
        domain="SyntheticTest",
        doc_ids=doc_ids,
        query_ids=query_ids,
        qrels=qrels,
        docs_low_384=docs_low,
        queries_low_384=queries_low,
        docs_high_4096=docs_high,
        queries_high_4096=queries_high,
        n_anchors=40,
        qdrant_host=None,
    )
    assert mb["curve"][0]["role"] == "SINGLE_SPACE_CONTROL"
    assert mb["curve"][-1]["role"] == "SINGLE_SPACE_CONTROL"
    p20 = next(r for r in mb["curve"] if abs(r["fraction"] - 0.20) < 1e-9)
    assert p20["role"] == "PHYSICAL_DUAL_SHARD_MIXED"
    assert p20["n_legacy_384d"] + p20["n_backfilled_4096d"] == n_docs
    assert "csls_minus_dev_zscore" in p20

    ma = evaluate_mode_a_upbridge(
        domain="SyntheticTest",
        doc_ids=doc_ids,
        query_ids=query_ids,
        qrels=qrels,
        docs_low_384=docs_low,
        queries_low_384=queries_low,
        docs_high_4096=docs_high,
    )
    assert "primary_qrels_blind_transductive" in ma["fits"]
    assert "sensitivity_report_positive_excluded" in ma["fits"]
