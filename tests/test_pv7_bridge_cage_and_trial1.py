"""Deterministic CPU-only unit test suite for Wave 16-Core (#47) & Trial 1 (#63).

Covers:
1. `chelated_pv7_spine.bridge_merge_harness`:
   - 5 bridge methods x 3 merge methods = 15 factorial cells (including rectangular
     384 <-> 4096 support and never denying CSLS to ordinary/centered Procrustes),
   - 20 deterministic seeded 500-anchor draws separating within-domain anchor SD
     from cross-domain N,
   - 5-encoder / 4-transition sequential composition (`v1->v2->v3->v4->v5`) vs
     direct `v1->v5` fit (`kappa`, `nonzero_kappa`, polar re-projection, `nDCG@10`, `Recall@10`),
   - progressive anchor contamination sweep (`0%, 1%, 5%, 10%, 20%`) + TRA-1
     outlier & rank diagnostics,
   - decomposed Item 6B hypothesis evaluator (`H6B_merge`, `H6B_bridge_static` with
     both `-0.010` retirement floor and `-0.005` tight equivalence + `CenteredProcrustes+CSLS`
     contrast, and `H6B_bridge_shift_and_4hop`),
   - `RequestSpanProfiler` recording microsecond spans and strictly keeping timeouts
     in the denominator as `0.0` `nDCG@10`.
2. `chelated_pv7_spine.cpu_io_cage`:
   - `systemd-run --user --scope -p MemoryMax=... -p MemorySwapMax=0` command wrapper,
   - strict sub-1 GiB enforcement rejecting `1024M == 1,073,741,824 B` and accepting `960M`,
   - `CUDA_VISIBLE_DEVICES=""` and zero-CUDA-context fail-closed checks,
   - `/proc/<pid>/io` and `/sys/block/<dev>/stat` parsers + reconciliation,
   - 3-tier storage classifier (`nvme_native`, `kernel_dm_delay_calibrated_stripe`,
     `physical_rota1_hdd_stripe`),
   - resident vLLM/Nemotron Prometheus metrics request-delta idle-lease verifier.
3. `partner_kits.trial1_search_migration_kit` & `docs/research/partner_charter_trial1_search_operator.md`:
   - mandatory charter header questions, 5 arms, migration bill worksheet, and retirement rule,
   - offline row-level scorer on all 5 open DEV fixtures (`win`, `loss`, `tie`, `timeout`,
     `exact_fallback_refill`) verifying `"status": "protocol-ready"`, timeout-in-denominator
     `0.0` `nDCG@10`, and explicit retirement of `ZeroMu` bridge premium on `tie`.
"""

from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Dict, List

import numpy as np

from chelated_pv7_spine.bridge_merge_harness import (
    BRIDGE_NAMES,
    BRIDGE_REGISTRY,
    MERGE_NAMES,
    CSLSMerge,
    CalibratedZScoreMerge,
    CenteredProcrustes,
    RawCosineMerge,
    RequestSpanProfiler,
    RidgeDriftAdapter,
    UncenteredProcrustes,
    ZeroMuPolarBridge,
    compose_bridge_chain,
    compute_tra1_anchor_diagnostics,
    evaluate_4hop_chain,
    evaluate_item6b_hypotheses,
    exact_binomial_upper_tail_p,
    holm_bonferroni_adjust,
    inject_anchor_contamination,
    l2_normalize,
    run_contamination_sweep,
    run_factorial_5x3,
    sample_anchor_draws,
    summarize_anchor_draw_variance,
)
from chelated_pv7_spine.cpu_io_cage import (
    ONE_GIB_BYTES,
    RECOMMENDED_SUB_1GIB_BYTES,
    STORAGE_TIER_DM_DELAY,
    STORAGE_TIER_NVME_NATIVE,
    STORAGE_TIER_PHYSICAL_HDD,
    CgroupBudgetViolationError,
    CgroupCageConfig,
    CudaIsolationError,
    ResidentIdleLeaseVerifier,
    assert_zero_cuda_context,
    build_cage_preflight_receipt,
    build_cpu_only_env,
    build_systemd_run_command,
    classify_storage_tier,
    diff_block_stat,
    diff_proc_io,
    parse_byte_size,
    parse_lsblk_json,
    parse_proc_io,
    parse_prometheus_metrics,
    parse_sys_block_stat,
    reconcile_proc_and_block_io,
)
from partner_kits.trial1_search_migration_kit import (
    ARM_PARTNER_INCUMBENT,
    ARM_UNCENTERED_CSLS,
    ARM_ZEROMU_CSLS,
    BUNDLE_ID,
    CLAIM_TIER,
    DEV_FIXTURE_SCENARIOS,
    MANDATORY_ARMS,
    STAGE1_STATUS,
    MigrationBillInputs,
    build_open_dev_fixture,
    compute_migration_bill,
    load_rows_jsonl,
    main as trial1_main,
    run_offline_dev_fixture_suite,
    score_trial1_rows,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _make_synthetic_domain(
    *,
    seed: int = 7,
    n_docs: int = 90,
    n_queries: int = 18,
    d_old: int = 16,
    d_new: int = 24,
    cone_shift_old: float = 2.5,
    cone_shift_new: float = 4.0,
) -> Dict[str, object]:
    """Build a deterministic two-space retrieval problem with known ground-truth qrels."""
    rng = np.random.default_rng(seed)
    # Latent semantic vectors in R^{min(d_old, d_new)}
    d_lat = min(d_old, d_new)
    docs_lat = l2_normalize(rng.normal(size=(n_docs, d_lat)))

    # Orthonormal embeddings into d_old and d_new + anisotropic cone shifts
    q_old, _ = np.linalg.qr(rng.normal(size=(d_old, d_lat)))
    q_new, _ = np.linalg.qr(rng.normal(size=(d_new, d_lat)))
    mu_old = np.zeros((1, d_old), dtype=np.float64)
    mu_old[0, 0] = cone_shift_old
    mu_new = np.zeros((1, d_new), dtype=np.float64)
    mu_new[0, 0] = cone_shift_new

    docs_old = l2_normalize(docs_lat @ q_old.T + mu_old + 0.03 * rng.normal(size=(n_docs, d_old)))
    docs_new = l2_normalize(docs_lat @ q_new.T + mu_new + 0.03 * rng.normal(size=(n_docs, d_new)))

    doc_ids = [f"d{i:04d}" for i in range(n_docs)]
    query_ids = [f"q{i:03d}" for i in range(n_queries)]

    qrels: Dict[str, Dict[str, float]] = {}
    q_lat = np.empty((n_queries, d_lat), dtype=np.float64)
    for qi, qid in enumerate(query_ids):
        target_doc = (qi * 3) % n_docs
        second_doc = (target_doc + 1) % n_docs
        q_lat[qi] = l2_normalize(docs_lat[target_doc] + 0.15 * rng.normal(size=(d_lat,)))
        qrels[qid] = {doc_ids[target_doc]: 2.0, doc_ids[second_doc]: 1.0}

    queries_old = l2_normalize(q_lat @ q_old.T + mu_old + 0.03 * rng.normal(size=(n_queries, d_old)))
    queries_new = l2_normalize(q_lat @ q_new.T + mu_new + 0.03 * rng.normal(size=(n_queries, d_new)))

    return {
        "docs_old": docs_old,
        "docs_new": docs_new,
        "queries_old": queries_old,
        "queries_new": queries_new,
        "doc_ids": doc_ids,
        "query_ids": query_ids,
        "qrels": qrels,
    }


class TestBridgeMergeHarness(unittest.TestCase):
    def test_all_5_bridges_rectangular_including_384_to_4096_and_4096_to_384(self) -> None:
        rng = np.random.default_rng(101)
        # Verify fast O(n^2 d) & thin SVD handling on real 384 <-> 4096 dimensions
        x_384 = l2_normalize(rng.normal(size=(16, 384)))
        y_4096 = l2_normalize(rng.normal(size=(16, 4096)))
        q_384 = l2_normalize(rng.normal(size=(4, 384)))
        q_4096 = l2_normalize(rng.normal(size=(4, 4096)))

        for name in BRIDGE_NAMES:
            cls = BRIDGE_REGISTRY[name]
            up = cls.fit(x_384, y_4096)
            out_up = up.transform(q_384)
            self.assertEqual(out_up.shape, (4, 4096))
            self.assertTrue(np.all(np.isfinite(out_up)))

            down = cls.fit(y_4096, x_384)
            out_down = down.transform(q_4096)
            self.assertEqual(out_down.shape, (4, 384))
            self.assertTrue(np.all(np.isfinite(out_down)))

        # UncenteredProcrustes semi-orthogonal Stiefel check W W^T = I_d_in when d_in < d_out
        up_p = UncenteredProcrustes.fit(x_384, y_4096)
        self.assertTrue(np.allclose(up_p.W @ up_p.W.T, np.eye(384), atol=1e-6))

        # RidgeDriftAdapter Woodbury check against dense formula on smaller rectangular pair
        xs = rng.normal(size=(10, 24))
        yt = rng.normal(size=(10, 32))
        ridge = RidgeDriftAdapter.fit(xs, yt, regularization=0.05)
        xc = xs - xs.mean(axis=0, keepdims=True)
        yc = yt - yt.mean(axis=0, keepdims=True)
        dense_w = np.linalg.solve(xc.T @ xc + 0.05 * np.eye(24), xc.T @ yc)
        self.assertTrue(np.allclose(ridge.W, dense_w, atol=1e-10))

    def test_factorial_5x3_gives_csls_to_every_bridge(self) -> None:
        dom = _make_synthetic_domain(seed=11, n_docs=80, n_queries=16, d_old=16, d_new=24)
        docs_old = dom["docs_old"]
        docs_new = dom["docs_new"]
        queries_old = dom["queries_old"]
        queries_new = dom["queries_new"]
        assert isinstance(docs_old, np.ndarray) and isinstance(docs_new, np.ndarray)
        assert isinstance(queries_old, np.ndarray) and isinstance(queries_new, np.ndarray)

        # Backfill 25% of corpus to test genuine dual-shard merging
        backfill_mask = np.zeros(docs_old.shape[0], dtype=bool)
        backfill_mask[:20] = True

        results = run_factorial_5x3(
            domain="synthetic_beir_a",
            src_anchors=docs_new[:40],
            tgt_anchors=docs_old[:40],
            eval_queries_src=queries_new,
            eval_queries_tgt=queries_new,
            docs_old=docs_old,
            docs_new=docs_new,
            query_ids=dom["query_ids"],  # type: ignore[arg-type]
            doc_ids=dom["doc_ids"],  # type: ignore[arg-type]
            qrels=dom["qrels"],  # type: ignore[arg-type]
            backfill_mask=backfill_mask,
            csls_k=5,
        )
        self.assertEqual(len(results), 15)
        for b_name in BRIDGE_NAMES:
            for m_name in MERGE_NAMES:
                self.assertIn((b_name, m_name), results)
                cell = results[(b_name, m_name)]
                self.assertGreaterEqual(cell.mean_ndcg_at_10, 0.0)
                self.assertLessEqual(cell.mean_ndcg_at_10, 1.0)
                self.assertEqual(len(cell.weights_sha256), 64)

        # Verify CSLS was explicitly evaluated for UncenteredProcrustes & CenteredProcrustes
        self.assertIn(("UncenteredProcrustes", "CSLSMerge"), results)
        self.assertIn(("CenteredProcrustes", "CSLSMerge"), results)
        self.assertAlmostEqual(results[("UncenteredProcrustes", "CSLSMerge")].backfill_fraction, 0.25)

    def test_three_dual_shard_mergers_direct_behavior(self) -> None:
        bridged = np.array([[0.95, 0.10, 0.0]], dtype=np.float64)
        native = np.array([[0.0, 0.0, 0.72]], dtype=np.float64)
        mask = np.array([False, False, True])
        b_ref = np.array([[0.98, 0.90, 0.0], [0.97, 0.88, 0.0], [0.96, 0.89, 0.0]], dtype=np.float64)
        n_ref = np.array([[0.0, 0.0, 0.20], [0.0, 0.0, 0.25], [0.0, 0.0, 0.30]], dtype=np.float64)

        raw = RawCosineMerge().merge(bridged, native, mask)
        zsc = CalibratedZScoreMerge().merge(bridged, native, mask, bridged_ref=b_ref, native_ref=n_ref)
        csls = CSLSMerge(csls_k=3).merge(bridged, native, mask, bridged_ref=b_ref, native_ref=n_ref)

        # Raw picks inflated bridged hub (doc 0); calibrated z-score & CSLS promote native doc 2
        self.assertEqual(int(np.argmax(raw[0])), 0)
        self.assertEqual(int(np.argmax(zsc[0])), 2)
        self.assertEqual(int(np.argmax(csls[0])), 2)

    def test_20_deterministic_seeded_anchor_draws_and_variance_separation(self) -> None:
        doc_ids = [f"doc_{i:04d}" for i in range(600)]
        draws_a = sample_anchor_draws("scifact_clean", doc_ids, n_draws=20, anchors_per_draw=500, base_seed=1600)
        draws_b = sample_anchor_draws("scifact_clean", doc_ids, n_draws=20, anchors_per_draw=500, base_seed=1600)
        self.assertEqual(len(draws_a), 20)
        self.assertEqual([d.manifest_sha256 for d in draws_a], [d.manifest_sha256 for d in draws_b])
        for d in draws_a:
            self.assertEqual(d.requested_anchors, 500)
            self.assertEqual(d.actual_anchors, 500)
            self.assertEqual(len(set(d.indices.tolist())), 500)
        self.assertGreater(len({d.manifest_sha256 for d in draws_a}), 15)

        report = summarize_anchor_draw_variance(
            {
                "dom_a": [0.70 + 0.001 * i for i in range(20)],
                "dom_b": [0.80 + 0.002 * i for i in range(20)],
                "dom_c": [0.60 - 0.001 * i for i in range(20)],
            },
            requested_anchors=500,
        )
        # Cross-domain N must strictly be 3 (NOT 60!)
        self.assertEqual(report.cross_domain_n, 3)
        self.assertGreater(report.cross_domain_sd, report.mean_within_domain_anchor_sd)
        self.assertEqual(report.per_domain["dom_a"].n_draws, 20)

    def test_4hop_v1_to_v5_sequential_vs_direct_and_polar_reprojection(self) -> None:
        rng = np.random.default_rng(202)
        n_docs = 70
        n_queries = 15
        # Peak dimension schedule: 16 -> 24 -> 32 -> 24 -> 16 (forces peak regime Jordan cosines)
        dims = [16, 24, 32, 24, 16]
        d_lat = 12
        docs_lat = l2_normalize(rng.normal(size=(n_docs, d_lat)))
        q_lat = np.empty((n_queries, d_lat), dtype=np.float64)
        doc_ids = [f"d{i:03d}" for i in range(n_docs)]
        query_ids = [f"q{i:02d}" for i in range(n_queries)]
        qrels: Dict[str, Dict[str, float]] = {}
        for qi, qid in enumerate(query_ids):
            td = (qi * 4) % n_docs
            q_lat[qi] = l2_normalize(docs_lat[td] + 0.12 * rng.normal(size=(d_lat,)))
            qrels[qid] = {doc_ids[td]: 1.0}

        doc_seq: List[np.ndarray] = []
        q_seq: List[np.ndarray] = []
        for idx, dim in enumerate(dims):
            basis, _ = np.linalg.qr(rng.normal(size=(dim, d_lat)))
            mu = np.zeros((1, dim), dtype=np.float64)
            mu[0, 0] = 1.5 + 0.4 * idx
            doc_seq.append(l2_normalize(docs_lat @ basis.T + mu + 0.02 * rng.normal(size=(n_docs, dim))))
            q_seq.append(l2_normalize(q_lat @ basis.T + mu + 0.02 * rng.normal(size=(n_queries, dim))))

        anchors = list(range(35))
        enc_names = ("v1", "v2", "v3", "v4", "v5")
        report_zm = evaluate_4hop_chain(
            bridge_name="ZeroMuPolarBridge",
            encoder_names=enc_names,
            doc_embeddings_seq=doc_seq,
            query_embeddings_seq=q_seq,
            anchor_indices=anchors,
            query_ids=query_ids,
            doc_ids=doc_ids,
            qrels=qrels,
            merge_name="CSLSMerge",
            polar_reproject=True,
        )
        self.assertEqual(len(report_zm.transitions), 4)
        self.assertEqual(report_zm.transitions[-1].transition_label, "v1->v2->v3->v4->v5")
        self.assertEqual(report_zm.transitions[-1].regime, "peak")
        # Polar re-projection restores nonzero_kappa == 1.00 on peak regime
        self.assertAlmostEqual(report_zm.transitions[-1].composed_nonzero_kappa, 1.0, places=6)
        self.assertGreater(report_zm.final_composed_ndcg_at_10, 0.70)
        self.assertGreater(report_zm.final_direct_ndcg_at_10, 0.70)

        # Verify chain composition directly for CenteredProcrustes without polar re-projection
        b1 = CenteredProcrustes.fit(doc_seq[0][anchors], doc_seq[1][anchors])
        b2 = CenteredProcrustes.fit(doc_seq[1][anchors], doc_seq[2][anchors])
        b3 = CenteredProcrustes.fit(doc_seq[2][anchors], doc_seq[3][anchors])
        comp_no_polar = compose_bridge_chain([b1, b2, b3], polar_reproject=False)
        self.assertGreaterEqual(comp_no_polar.nonzero_condition_number, 1.0)

    def test_progressive_anchor_contamination_sweep_and_tra1_diagnostics(self) -> None:
        dom = _make_synthetic_domain(seed=33, n_docs=80, n_queries=15, d_old=16, d_new=16)
        docs_old = dom["docs_old"]
        docs_new = dom["docs_new"]
        queries_new = dom["queries_new"]
        assert isinstance(docs_old, np.ndarray) and isinstance(docs_new, np.ndarray)
        assert isinstance(queries_new, np.ndarray)

        points = run_contamination_sweep(
            src_anchors=docs_new[:50],
            tgt_anchors=docs_old[:50],
            eval_queries_src=queries_new,
            docs_old=docs_old,
            query_ids=dom["query_ids"],  # type: ignore[arg-type]
            doc_ids=dom["doc_ids"],  # type: ignore[arg-type]
            qrels=dom["qrels"],  # type: ignore[arg-type]
            bridge_names=("UncenteredProcrustes", "ZeroMuPolarBridge"),
            modes=("mismatched", "duplicate", "off_domain"),
            rates=(0.0, 0.01, 0.05, 0.10, 0.20),
            seed=42,
        )
        self.assertEqual(len(points), 2 * 3 * 5)

        # Duplicate contamination at 20% must raise duplicate_pair_fraction significantly
        dup_20 = [
            p for p in points
            if p.bridge_name == "ZeroMuPolarBridge"
            and p.contamination_mode == "duplicate"
            and math.isclose(p.contamination_rate, 0.20)
        ][0]
        self.assertGreaterEqual(dup_20.tra1.duplicate_pair_fraction, 0.18)
        self.assertIn(dup_20.tra1.tra1_health_status, {"WARN_CONTAMINATED", "FAIL_CLOSED_DEGENERATE"})

        # Mismatched contamination at 20% must raise outlier_fraction relative to 0%
        mis_0 = [
            p for p in points
            if p.bridge_name == "ZeroMuPolarBridge"
            and p.contamination_mode == "mismatched"
            and math.isclose(p.contamination_rate, 0.0)
        ][0]
        mis_20 = [
            p for p in points
            if p.bridge_name == "ZeroMuPolarBridge"
            and p.contamination_mode == "mismatched"
            and math.isclose(p.contamination_rate, 0.20)
        ][0]
        self.assertGreater(mis_20.tra1.outlier_fraction, mis_0.tra1.outlier_fraction)

        # Also test direct mixed injection + TRA-1 diagnostic SHA-256 determinism
        cs, ct, bad = inject_anchor_contamination(docs_new[:50], docs_old[:50], rate=0.20, mode="mixed", seed=9)
        self.assertEqual(int(bad.sum()), 10)
        diag_a = compute_tra1_anchor_diagnostics(cs, ct, ZeroMuPolarBridge.fit(cs, ct))
        diag_b = compute_tra1_anchor_diagnostics(cs, ct, ZeroMuPolarBridge.fit(cs, ct))
        self.assertEqual(diag_a.diagnostic_sha256, diag_b.diagnostic_sha256)

    def test_item6b_decomposed_hypotheses_and_holm_exact_binomial(self) -> None:
        # Exact binomial sanity: 10/12 positive -> p = (66 + 12 + 1) / 4096 = 79 / 4096 ~= 0.019287
        self.assertAlmostEqual(exact_binomial_upper_tail_p(10, 12, 0.5), 79.0 / 4096.0, places=9)
        holm = holm_bonferroni_adjust({"a": 0.01, "b": 0.02, "c": 0.04})
        self.assertAlmostEqual(holm["a"], 0.03)
        self.assertAlmostEqual(holm["b"], 0.04)
        self.assertAlmostEqual(holm["c"], 0.04)

        domains = [f"dom_{i:02d}" for i in range(12)]
        # Case 1: All three hypotheses pass (11/12 positive merge, static clears both -0.010 and -0.005)
        rep_pass = evaluate_item6b_hypotheses(
            clean_1hop_csls_vs_zscore_by_domain={d: (0.025 if i < 11 else -0.002) for i, d in enumerate(domains)},
            static_zeromu_vs_uncentered_at_csls_by_domain={d: 0.001 for d in domains},
            static_zeromu_vs_centered_at_csls_by_domain={d: 0.0005 for d in domains},
            shift_zeromu_vs_uncentered_at_csls_by_domain={d: 0.035 for d in domains},
            fourhop_zeromu_vs_uncentered_at_csls_by_domain={d: 0.042 for d in domains},
            prefix_displacement_norms_by_domain={d: 0.18 for d in domains},
            direct_v1_v5_delta_by_domain={d: 0.005 for d in domains},
        )
        self.assertEqual(rep_pass.overall_verdict, "verified-pass")
        self.assertTrue(rep_pass.h6b_merge["passed"])
        self.assertTrue(rep_pass.h6b_bridge_static["clears_retirement_floor"])
        self.assertTrue(rep_pass.h6b_bridge_static["clears_tight_equivalence_floor"])
        self.assertAlmostEqual(rep_pass.h6b_bridge_static["retirement_floor"], -0.010)
        self.assertAlmostEqual(rep_pass.h6b_bridge_static["tight_equivalence_floor"], -0.005)
        self.assertIn("zeromu_vs_centered_csls_contrast", rep_pass.h6b_bridge_static)
        self.assertTrue(rep_pass.h6b_bridge_shift_and_4hop["passed"])

        # Case 2: CSLS passes merge, static falls between -0.010 and -0.005, shift/4-hop fail -> retire ZeroMu
        rep_retire = evaluate_item6b_hypotheses(
            clean_1hop_csls_vs_zscore_by_domain={d: 0.030 for d in domains},
            static_zeromu_vs_uncentered_at_csls_by_domain={d: -0.007 for d in domains},
            static_zeromu_vs_centered_at_csls_by_domain={d: -0.002 for d in domains},
            shift_zeromu_vs_uncentered_at_csls_by_domain={d: -0.001 for d in domains},
            fourhop_zeromu_vs_uncentered_at_csls_by_domain={d: -0.003 for d in domains},
        )
        self.assertTrue(rep_retire.h6b_bridge_static["clears_retirement_floor"])
        self.assertFalse(rep_retire.h6b_bridge_static["clears_tight_equivalence_floor"])
        self.assertEqual(rep_retire.overall_verdict, "credit-csls-retire-zeromu-premium")

    def test_request_span_profiler_keeps_timeouts_in_denominator_as_zero_ndcg(self) -> None:
        profiler = RequestSpanProfiler()
        profiler.record_span(
            request_id="req_1",
            queue_us=100.0,
            backend_knn_us=5_000.0,
            prefilter_visited=120,
            short_refill_count=0,
            exact_fallback_triggered=False,
            gemv_us=250.0,
            merge_us=150.0,
            total_us=5_500.0,
            timed_out=False,
            ndcg_at_10=0.90,
        )
        # Timed-out request supplies non-zero raw_ndcg_at_10=0.80 which MUST be forced to 0.0!
        rec_to = profiler.record_span(
            request_id="req_2_timeout",
            queue_us=500.0,
            backend_knn_us=90_000.0,
            prefilter_visited=4_096,
            short_refill_count=3,
            exact_fallback_triggered=True,
            gemv_us=0.0,
            merge_us=0.0,
            total_us=90_500.0,
            timed_out=True,
            ndcg_at_10=0.80,
        )
        self.assertEqual(rec_to.effective_ndcg_at_10, 0.0)

        with profiler.trace("req_3_ctx") as span:
            with span.measure_stage("queue"):
                _ = 1 + 1
            with span.measure_stage("backend_knn"):
                _ = 2 + 2
            span.prefilter_visited = 256
            span.short_refill_count = 1
            span.raw_ndcg_at_10 = 0.60

        summary = profiler.summarize()
        self.assertEqual(summary["total_requests"], 3)
        self.assertEqual(summary["timeout_count"], 1)
        self.assertAlmostEqual(summary["timeout_rate"], 1.0 / 3.0)
        self.assertEqual(summary["exact_fallback_count"], 1)
        # Mean including timeouts = (0.90 + 0.0 + 0.60) / 3 = 0.50 (NOT 0.75!)
        self.assertAlmostEqual(summary["mean_ndcg_at_10_including_timeouts"], 0.50, places=6)
        self.assertAlmostEqual(summary["mean_ndcg_at_10_completed_only"], 0.75, places=6)


class TestCpuIoCage(unittest.TestCase):
    def test_systemd_run_wrapper_and_sub_1gib_1024m_bug_catch(self) -> None:
        self.assertEqual(parse_byte_size("1024M"), ONE_GIB_BYTES)
        self.assertEqual(parse_byte_size("960M"), RECOMMENDED_SUB_1GIB_BYTES)

        # Catch the 1024M == 1 GiB bug strictly when claim_sub_1gib=True
        for bad_spec in ("1024M", "1G", ONE_GIB_BYTES, ONE_GIB_BYTES + 1):
            with self.assertRaises(CgroupBudgetViolationError):
                CgroupCageConfig.from_size_spec(bad_spec, claim_sub_1gib=True)

        # Reject non-zero swap when swap-free discipline is active
        with self.assertRaises(CgroupBudgetViolationError):
            CgroupCageConfig.from_size_spec("960M", memory_swap_max="64M", claim_sub_1gib=True)

        # Valid 960M sub-1 GiB config builds exact deterministic systemd-run command
        cfg = CgroupCageConfig.from_size_spec(
            "960M",
            memory_swap_max=0,
            claim_sub_1gib=True,
            allowed_cpus="0-7",
            unit_name="pv7-item2d-iq04",
        )
        cmd = build_systemd_run_command(["python3", "run_2d.py"], cfg)
        self.assertEqual(cmd[:3], ["systemd-run", "--user", "--scope"])
        self.assertIn(f"MemoryMax={RECOMMENDED_SUB_1GIB_BYTES}", cmd)
        self.assertIn("MemorySwapMax=0", cmd)
        self.assertIn("AllowedCPUs=0-7", cmd)
        self.assertEqual(cmd[-3:], ["--", "python3", "run_2d.py"])

    def test_zero_cuda_context_enforcement(self) -> None:
        clean_env = build_cpu_only_env({"PATH": "/usr/bin"})
        self.assertEqual(clean_env["CUDA_VISIBLE_DEVICES"], "")
        receipt = assert_zero_cuda_context(
            env=clean_env,
            target_pid=4242,
            proc_maps_text="7f1200000000-7f1200100000 r-xp /usr/lib/libc.so.6\n",
            nvidia_smi_csv="pid, process_name, used_memory\n9999, vllm, 20480 MiB\n",
            strict=True,
        )
        self.assertTrue(receipt.passed)
        self.assertEqual(receipt.active_cuda_pids, (9999,))

        # Fail closed if CUDA_VISIBLE_DEVICES is non-empty
        with self.assertRaises(CudaIsolationError):
            assert_zero_cuda_context(env={"CUDA_VISIBLE_DEVICES": "0"}, proc_maps_text="", strict=True)

        # Fail closed if libcuda.so is mapped in process address space
        with self.assertRaises(CudaIsolationError):
            assert_zero_cuda_context(
                env=clean_env,
                proc_maps_text="7f9900000000-7f9900200000 r-xp /usr/lib/x86_64-linux-gnu/libcuda.so.1\n",
                strict=True,
            )

        # Fail closed if target PID appears in nvidia-smi active contexts
        with self.assertRaises(CudaIsolationError):
            assert_zero_cuda_context(
                env=clean_env,
                target_pid=4242,
                proc_maps_text="",
                nvidia_smi_csv="4242, python3, 512 MiB\n",
                strict=True,
            )

    def test_proc_io_and_sys_block_stat_parsers_and_reconciliation(self) -> None:
        io_before_txt = (
            "rchar: 1000000\n"
            "wchar: 500000\n"
            "syscr: 120\n"
            "syscw: 60\n"
            "read_bytes: 4194304\n"
            "write_bytes: 1048576\n"
            "cancelled_write_bytes: 0\n"
        )
        io_after_txt = (
            "rchar: 5000000\n"
            "wchar: 1500000\n"
            "syscr: 420\n"
            "syscw: 160\n"
            "read_bytes: 25165824\n"
            "write_bytes: 5242880\n"
            "cancelled_write_bytes: 0\n"
        )
        proc_delta = diff_proc_io(parse_proc_io(io_before_txt), parse_proc_io(io_after_txt))
        self.assertEqual(proc_delta.read_bytes, 20_971_520)  # 20 MiB
        self.assertEqual(proc_delta.write_bytes, 4_194_304)  # 4 MiB
        self.assertEqual(proc_delta.syscr, 300)

        # 40960 read sectors * 512 = 20 MiB; 8192 write sectors * 512 = 4 MiB
        blk_before = parse_sys_block_stat("100 0 2000 50 50 0 1000 20 0 70 70 0 0 0 0 0 0")
        blk_after = parse_sys_block_stat("300 0 42960 250 150 0 9192 80 0 330 330 0 0 0 0 0 0")
        blk_delta = diff_block_stat(blk_before, blk_after)
        self.assertEqual(blk_delta.read_bytes, 20_971_520)
        self.assertEqual(blk_delta.write_bytes, 4_194_304)
        self.assertEqual(blk_delta.read_ios, 200)
        self.assertEqual(blk_delta.write_ios, 100)

        recon = reconcile_proc_and_block_io(proc_delta, blk_delta)
        self.assertAlmostEqual(recon.read_block_to_proc_ratio or 0.0, 1.0)
        self.assertAlmostEqual(recon.write_amplification_ratio or 0.0, 1.0)
        self.assertFalse(recon.external_background_io_suspected)

    def test_storage_tier_classifier_all_3_tiers(self) -> None:
        nvme = classify_storage_tier("nvme0n1", rota=0, transport="nvme", model="Samsung PM9A3")
        self.assertEqual(nvme.tier, STORAGE_TIER_NVME_NATIVE)
        self.assertFalse(nvme.emulated)
        self.assertFalse(nvme.allows_cylinder_order_claim)

        dm_delay = classify_storage_tier(
            "dm-0",
            rota=0,
            dm_table="0 209715200 delay /dev/nvme0n1p2 0 8",
        )
        self.assertEqual(dm_delay.tier, STORAGE_TIER_DM_DELAY)
        self.assertTrue(dm_delay.emulated)
        self.assertFalse(dm_delay.allows_cylinder_order_claim)
        self.assertEqual(dm_delay.dm_delay_ms, 8)

        hdd = classify_storage_tier("sda", rota=1, transport="sata", model="WDC WD140EDGZ")
        self.assertEqual(hdd.tier, STORAGE_TIER_PHYSICAL_HDD)
        self.assertFalse(hdd.emulated)
        self.assertTrue(hdd.allows_cylinder_order_claim)

        lsblk_rows = parse_lsblk_json(
            json.dumps(
                {
                    "blockdevices": [
                        {"name": "nvme0n1", "rota": False, "tran": "nvme", "model": "PM9A3", "type": "disk"},
                        {"name": "sda", "rota": True, "tran": "sata", "model": "Exos", "type": "disk"},
                    ]
                }
            )
        )
        self.assertEqual(lsblk_rows[0]["rota"], 0)
        self.assertEqual(lsblk_rows[1]["rota"], 1)

    def test_resident_vllm_nemotron_idle_lease_verifier(self) -> None:
        prom_idle = """
# HELP vllm:request_success_total Count of successfully processed requests.
# TYPE vllm:request_success_total counter
vllm:request_success_total{finished_reason="stop",model_name="nemotron-30b"} 412.0
vllm:num_requests_running{model_name="nemotron-30b"} 0.0
vllm:num_requests_waiting{model_name="nemotron-30b"} 0.0
vllm:prompt_tokens_total{model_name="nemotron-30b"} 128000.0
"""
        prom_incremented = prom_idle.replace("412.0", "413.0")
        prom_running = prom_idle.replace("num_requests_running{model_name=\"nemotron-30b\"} 0.0",
                                         "num_requests_running{model_name=\"nemotron-30b\"} 1.0")
        prom_reset = prom_idle.replace("412.0", "10.0")

        samples = parse_prometheus_metrics(prom_idle)
        self.assertEqual(len(samples), 4)

        verifier = ResidentIdleLeaseVerifier()
        s_before = verifier.snapshot_from_text(prom_idle, source="spark2:8000/metrics")
        s_after_clean = verifier.snapshot_from_text(prom_idle, source="spark2:8000/metrics")
        s_after_busy = verifier.snapshot_from_text(prom_incremented, source="spark2:8000/metrics")
        s_after_running = verifier.snapshot_from_text(prom_running, source="spark2:8000/metrics")
        s_after_reset = verifier.snapshot_from_text(prom_reset, source="spark2:8000/metrics")

        verdict_ok = verifier.verify_lease(s_before, s_after_clean)
        self.assertTrue(verdict_ok.valid)
        self.assertEqual(verdict_ok.lease_status, "CLEAN_IDLE_LEASE")

        verdict_busy = verifier.verify_lease(s_before, s_after_busy)
        self.assertFalse(verdict_busy.valid)
        self.assertEqual(verdict_busy.lease_status, "VOIDED_RESIDENT_ACTIVITY")
        self.assertTrue(any("RESIDENT_REQUEST_DELTA_NONZERO" in v for v in verdict_busy.violations))

        verdict_running = verifier.verify_lease(s_before, s_after_running)
        self.assertFalse(verdict_running.valid)
        self.assertTrue(any("ACTIVE_REQUESTS_AT_LEASE_END" in v for v in verdict_running.violations))

        verdict_reset = verifier.verify_lease(s_before, s_after_reset)
        self.assertFalse(verdict_reset.valid)
        self.assertTrue(any("COUNTER_RESET_DETECTED" in v for v in verdict_reset.violations))

        # End-to-end preflight receipt builder
        preflight = build_cage_preflight_receipt(
            config=CgroupCageConfig.from_size_spec("960M", claim_sub_1gib=True),
            command=["python3", "bench.py"],
            storage=classify_storage_tier("nvme0n1", rota=0, transport="nvme"),
            cuda_receipt=assert_zero_cuda_context(env=build_cpu_only_env({}), proc_maps_text="", strict=True),
            idle_lease=verdict_ok,
        )
        self.assertTrue(preflight["preflight_passed"])
        self.assertEqual(len(preflight["receipt_sha256"]), 64)


class TestTrial1SearchMigrationKitAndCharter(unittest.TestCase):
    def test_charter_contains_mandatory_sections_and_retirement_rule(self) -> None:
        charter_path = REPO_ROOT / "docs" / "research" / "partner_charter_trial1_search_operator.md"
        self.assertTrue(charter_path.exists())
        text = charter_path.read_text(encoding="utf-8")
        for needle in (
            "Bundle #63",
            "protocol-ready",
            "Three Mandatory Pre-Registration Header Questions",
            "UncenteredProcrustes+CalibratedZScore",
            "UncenteredProcrustes+CSLS",
            "ZeroMu+CalibratedZScore",
            "ZeroMu+CSLS",
            "PartnerIncumbentAlternative",
            "Migration Bill Worksheet",
            "+-0.010",
            "ZeroMu",
            "exact_fallback_refill",
        ):
            self.assertIn(needle, text)

    def test_migration_bill_worksheet_arithmetic(self) -> None:
        bill = compute_migration_bill(
            MigrationBillInputs(
                arm_id=ARM_ZEROMU_CSLS,
                corpus_docs=10_000_000,
                migration_window_hours=168.0,
                dual_index_ram_gb=100.0,
                dual_index_disk_gb=720.0,
                backfill_gpu_hours=20.0,
                fallback_cpu_hours=10.0,
                engineering_hours=10.0,
                zeromu_sidecar_bytes_per_doc=4,
            )
        )
        self.assertGreater(bill.sidecar_ram_gb, 0.03)
        self.assertGreater(bill.total_bill_usd, 1_500.0)
        self.assertAlmostEqual(bill.cost_per_million_docs_usd, bill.total_bill_usd / 10.0, places=3)

    def test_open_dev_fixtures_all_5_scenarios_and_retirement_logic(self) -> None:
        suite = run_offline_dev_fixture_suite(n_queries=24)
        self.assertEqual(suite["bundle_id"], BUNDLE_ID)
        self.assertEqual(suite["claim_tier"], CLAIM_TIER)
        self.assertEqual(suite["status"], STAGE1_STATUS)
        self.assertEqual(tuple(suite["mandatory_arms"]), MANDATORY_ARMS)
        self.assertEqual(set(suite["scenarios"].keys()), set(DEV_FIXTURE_SCENARIOS))

        # 1. win -> ZeroMu+CSLS wins while UncenteredProcrustes+CSLS misses
        win = suite["scenarios"]["win"]
        self.assertEqual(win["status"], "protocol-ready")
        self.assertEqual(win["decision"]["outcome"], "win_zeromu_csls")
        self.assertEqual(win["decision"]["recommended_arm"], ARM_ZEROMU_CSLS)
        self.assertFalse(win["decision"]["zeromu_bridge_premium_retired"])

        # 2. loss -> PartnerIncumbentAlternative retains
        loss = suite["scenarios"]["loss"]
        self.assertEqual(loss["decision"]["outcome"], "loss")
        self.assertEqual(loss["decision"]["recommended_arm"], ARM_PARTNER_INCUMBENT)

        # 3. tie -> UncenteredProcrustes+CSLS clears at <= total bill -> retire ZeroMu, credit CSLS!
        tie = suite["scenarios"]["tie"]
        self.assertEqual(tie["decision"]["outcome"], "tie_retire_zeromu_credit_csls")
        self.assertEqual(tie["decision"]["recommended_arm"], ARM_UNCENTERED_CSLS)
        self.assertTrue(tie["decision"]["zeromu_bridge_premium_retired"])
        self.assertTrue(tie["decision"]["csls_credited"])
        self.assertLessEqual(
            tie["arms"][ARM_UNCENTERED_CSLS]["total_bill_usd"],
            tie["arms"][ARM_ZEROMU_CSLS]["total_bill_usd"],
        )

        # 4. timeout -> timeouts stay in denominator as 0.0 nDCG@10 and fail SLA
        to = suite["scenarios"]["timeout"]
        zm_to = to["arms"][ARM_ZEROMU_CSLS]
        self.assertEqual(zm_to["timeout_count"], 3)
        self.assertAlmostEqual(zm_to["timeout_rate"], 3.0 / 24.0)
        self.assertFalse(zm_to["clears_sla_targets"])
        # Because 3/24 queries timed out as 0.0, mean_ndcg_at_10 must be strictly lower than completed-only
        self.assertLess(
            zm_to["span_summary"]["mean_ndcg_at_10_including_timeouts"],
            zm_to["span_summary"]["mean_ndcg_at_10_completed_only"],
        )

        # 5. exact_fallback_refill -> records short_refill_count > 0 & exact_fallback_triggered
        fb = suite["scenarios"]["exact_fallback_refill"]
        zm_fb = fb["arms"][ARM_ZEROMU_CSLS]
        self.assertEqual(zm_fb["exact_fallback_count"], 6)
        self.assertEqual(zm_fb["total_short_refills"], 12)
        self.assertGreater(zm_fb["bill_breakdown"]["fallback_compute_usd"], win["arms"][ARM_ZEROMU_CSLS]["bill_breakdown"]["fallback_compute_usd"])

    def test_trial1_cli_and_jsonl_roundtrip(self) -> None:
        rows, bills = build_open_dev_fixture("tie", n_queries=12)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_dir = Path(tmp)
            jsonl_path = tmp_dir / "rows.jsonl"
            bills_path = tmp_dir / "bills.json"
            out_path = tmp_dir / "report.json"

            lines = [
                json.dumps(
                    {
                        "query_id": r.query_id,
                        "arm_id": r.arm_id,
                        "bridge_method": r.bridge_method,
                        "merge_method": r.merge_method,
                        "ranked_doc_ids": list(r.ranked_doc_ids),
                        "qrels": dict(r.qrels),
                        "queue_us": r.queue_us,
                        "backend_knn_us": r.backend_knn_us,
                        "prefilter_visited": r.prefilter_visited,
                        "short_refill_count": r.short_refill_count,
                        "exact_fallback_triggered": r.exact_fallback_triggered,
                        "gemv_us": r.gemv_us,
                        "merge_us": r.merge_us,
                        "total_us": r.total_us,
                        "timed_out": r.timed_out,
                    }
                )
                for r in rows
            ]
            jsonl_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            bills_path.write_text(
                json.dumps({k: v.__dict__ for k, v in bills.items()}, indent=2),
                encoding="utf-8",
            )

            loaded = load_rows_jsonl(jsonl_path)
            self.assertEqual(len(loaded), len(rows))
            rc = trial1_main(["--rows-jsonl", str(jsonl_path), "--bills-json", str(bills_path), "--output", str(out_path)])
            self.assertEqual(rc, 0)
            scored = json.loads(out_path.read_text(encoding="utf-8"))
            self.assertTrue(scored["decision"]["zeromu_bridge_premium_retired"])
            # Also confirm direct programmatic call matches protocol-ready default
            direct = score_trial1_rows(loaded, bills, fixture_label="roundtrip_check")
            self.assertEqual(direct["status"], "protocol-ready")


if __name__ == "__main__":
    unittest.main()
