"""W22-06 unit tests: ranking, margin flips, LODO law, conformal containment, bit ledger."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chelated_pv7_spine.bridge_merge_harness import (  # noqa: E402
    _canonical_topk,
    evaluate_score_matrix,
    run_factorial_5x3,
)
from migration_frontier_w22_06 import (  # noqa: E402
    bit_ledger,
    calibrate_margin,
    conformal_k,
    conformal_study,
    containment_score,
    coverage,
    decide_margin_claim,
    doc_tiebreak_keys,
    flip_pair_stats,
    heldout_fit_error,
    index_set_bits,
    is_calibration_query,
    lodo_law,
    rank_positions,
    serve_scores,
    summarize_flips,
    topk_indices,
)


class TestRanking(unittest.TestCase):
    def test_topk_matches_harness_canonical_tiebreak(self):
        rng = np.random.default_rng(0)
        doc_ids = [f"d{i}" for i in rng.permutation(200)]
        row = np.round(rng.normal(size=200), 1)  # many ties
        keys = doc_tiebreak_keys(doc_ids)
        self.assertEqual(topk_indices(row, keys, 10).tolist(), _canonical_topk(row, doc_ids, 10))

    def test_rank_positions_inverse(self):
        keys = doc_tiebreak_keys(["b", "a", "c"])
        pos = rank_positions(np.array([0.5, 0.5, 0.9]), keys)
        self.assertEqual(pos.tolist(), [3, 2, 1])  # tie broken by id: "a" (idx1) before "b" (idx0)


class TestMarginFlips(unittest.TestCase):
    def test_calibrate_recovers_affine_map(self):
        rng = np.random.default_rng(1)
        f = rng.normal(size=(5, 80))
        g = 0.3 + 2.0 * f
        keys = doc_tiebreak_keys([f"d{i}" for i in range(80)])
        cal = calibrate_margin(f, g, keys, region_top=10)
        self.assertAlmostEqual(cal["a"], 0.3, places=8)
        self.assertAlmostEqual(cal["b"], 2.0, places=8)
        self.assertAlmostEqual(cal["margin"], 0.0, places=8)

    def test_flip_pair_within_and_beyond_margin(self):
        keys = doc_tiebreak_keys(["d0", "d1", "d2", "d3"])
        served = np.array([0.90, 0.50, 0.49, 0.10])
        native = np.array([0.90, 0.10, 0.80, 0.20])
        st = flip_pair_stats(served, native, keys, k=2, margin=0.05)
        self.assertEqual(st, {"n_missing": 1, "n_flip_pairs": 1, "n_missing_margin_explainable": 1})
        st2 = flip_pair_stats(served, native, keys, k=2, margin=0.001)
        self.assertEqual(st2["n_missing_margin_explainable"], 0)
        self.assertEqual(st2["n_flip_pairs"], 0)

    def test_negative_slope_ranks_by_served_not_calibrated(self):
        # Regression for Tier B round 1: with b < 0 the calibrated scores invert the order;
        # S_k must still be the served top-k (d0, d1), not the served bottom-k.
        keys = doc_tiebreak_keys(["d0", "d1", "d2", "d3"])
        served = np.array([0.90, 0.80, 0.20, 0.10])
        native = np.array([0.90, 0.10, 0.80, 0.20])  # native top-2 = d0, d2
        st = flip_pair_stats(served, native, keys, k=2, margin=0.01, a=1.0, b=-0.5)
        self.assertEqual(st["n_missing"], 1)  # served-bottom ranking would miss both natives
        # calibrated: d1 -> 0.60, d2 -> 0.90; |diff| = 0.30
        self.assertEqual(st["n_missing_margin_explainable"], 0)
        st2 = flip_pair_stats(served, native, keys, k=2, margin=0.31, a=1.0, b=-0.5)
        self.assertEqual(st2["n_missing_margin_explainable"], 1)
        self.assertEqual(st2["n_flip_pairs"], 1)

    def test_no_error_no_flip(self):
        keys = doc_tiebreak_keys(["a", "b", "c"])
        row = np.array([0.3, 0.2, 0.1])
        self.assertEqual(flip_pair_stats(row, row, keys, k=2, margin=1.0)["n_missing"], 0)

    def test_summary_and_decision(self):
        rows = [
            {"n_missing": 2, "n_flip_pairs": 1, "n_missing_margin_explainable": 1},
            {"n_missing": 0, "n_flip_pairs": 0, "n_missing_margin_explainable": 0},
        ]
        s = summarize_flips(rows)
        self.assertAlmostEqual(s["frac_missing_margin_explainable"], 0.5)
        self.assertAlmostEqual(s["frac_queries_with_flip"], 0.5)
        dec = decide_margin_claim({f"d{i}": 0.6 for i in range(10)} | {"x": 0.1, "y": 0.2}, 0.55, 0.5)
        self.assertEqual(dec["claim_errors_are_margin_level"], "SUPPORTED")
        dec2 = decide_margin_claim({f"d{i}": 0.6 for i in range(9)} | {"x": 0.1, "y": 0.2, "z": 0.3}, 0.55, 0.05)
        self.assertEqual(dec2["claim_errors_are_margin_level"], "NOT_SUPPORTED")
        self.assertEqual(dec2["margin_vulnerability_phenomenon"], "ABSENT")


class TestLodoLaw(unittest.TestCase):
    def test_perfect_linear_law_not_killed(self):
        x = {f"d{i}": float(i) for i in range(12)}
        g = {d: 0.01 + 0.02 * v for d, v in x.items()}
        out = lodo_law(x, g)
        self.assertAlmostEqual(out["r2_lodo"], 1.0, places=9)
        self.assertEqual(out["sign_accuracy"], 1.0)
        self.assertEqual(out["verdict"], "NOT_KILLED")

    def test_noise_law_killed_on_r2(self):
        rng = np.random.default_rng(3)
        x = {f"d{i}": float(rng.normal()) for i in range(12)}
        g = {d: 0.05 + 0.01 * float(rng.normal()) for d in x}
        out = lodo_law(x, g)
        self.assertEqual(out["sign_accuracy"], 1.0)  # all positive outcome, positive predictions
        self.assertLess(out["r2_lodo"], 0.5)
        self.assertEqual(out["verdict"], "KILL")

    def test_domain_mismatch_raises(self):
        with self.assertRaises(ValueError):
            lodo_law({"a": 1.0}, {"b": 1.0})


class TestConformal(unittest.TestCase):
    def test_containment_score(self):
        keys = doc_tiebreak_keys(["a", "b", "c", "d"])
        served = np.array([0.9, 0.1, 0.5, 0.3])  # ranks: a1 c2 d3 b4
        native = np.array([0.9, 0.8, 0.1, 0.1])  # top-2: a, b
        self.assertEqual(containment_score(served, native, keys, 2), 4)
        self.assertEqual(containment_score(native, native, keys, 2), 2)

    def test_conformal_k_quantile(self):
        scores = list(range(1, 20))  # n=19 -> ceil(20*0.9)=18th smallest = 18
        self.assertEqual(conformal_k(scores, alpha=0.10, n_docs=999), 18)
        self.assertEqual(conformal_k([5, 6], alpha=0.10, n_docs=999), 999)  # undefined -> trivial
        self.assertEqual(coverage([1, 18, 19], 18), 2 / 3)

    def test_split_is_deterministic(self):
        self.assertEqual(is_calibration_query("x", "q1"), is_calibration_query("x", "q1"))
        flags = [is_calibration_query("dom", f"q{i}") for i in range(400)]
        self.assertTrue(120 < sum(flags) < 280)

    def test_study_kill_rules(self):
        good = {f"d{i}": {"cal": [10] * 30, "test": [10] * 30, "n_docs": 750} for i in range(4)}
        out = conformal_study(good, k=10, alpha=0.10)
        self.assertEqual(out["verdict"], "NOT_KILLED")
        big = {f"d{i}": {"cal": [500] * 30, "test": [500] * 30, "n_docs": 750} for i in range(4)}
        out2 = conformal_study(big, k=10, alpha=0.10)
        self.assertEqual(out2["verdict"], "KILL")
        self.assertTrue(out2["kill_reasons"]["median_in_domain_set_size_gt_limit"])
        shift = {
            "a": {"cal": [10] * 30, "test": [10] * 30, "n_docs": 750},
            "b": {"cal": [10] * 30, "test": [10] * 30, "n_docs": 750},
            "c": {"cal": [50] * 30, "test": [50] * 30, "n_docs": 750},
        }
        out3 = conformal_study(shift, k=10, alpha=0.10, groups={"a": "g1", "b": "g1", "c": "g2"})
        self.assertEqual(out3["per_domain"]["c"]["coverage_lodo"], 0.0)
        self.assertEqual(out3["verdict"], "KILL")
        self.assertIn("g1_to_g2", out3["group_transfer"])


class TestLedger(unittest.TestCase):
    def test_index_bits(self):
        self.assertEqual(index_set_bits(750, 150), 750)  # mask cheaper than 150*10 bits
        self.assertEqual(index_set_bits(1024, 3), 30)

    def test_centered_ledger_counts(self):
        led = bit_ledger("CenteredProcrustes", d_src=8, d_tgt=4, n_docs=10, n_backfill=2, n_anchors=5, n_dev_queries=3, n_dev_ref_docs=4)
        self.assertEqual(sum(p["n_floats"] for p in led["bridge_params"]), 32 + 8 + 4 + 1)
        self.assertEqual(led["new_encoder_forward_passes"], 10)
        self.assertEqual(led["full_reindex_bytes_fp32"], 4 * 10 * 8)
        self.assertAlmostEqual(led["bridge_param_bits_over_dsrc_dtgt_32"], 45 / 32)
        with self.assertRaises(ValueError):
            bit_ledger("Nope", d_src=1, d_tgt=1, n_docs=1, n_backfill=0, n_anchors=1, n_dev_queries=1, n_dev_ref_docs=1)


class TestServeScoresMatchesFactorial(unittest.TestCase):
    def test_fused_eval_ndcg_equals_factorial_cell(self):
        rng = np.random.default_rng(7)
        d_src, d_tgt, n_docs = 24, 12, 60
        proj = rng.normal(size=(d_src, d_tgt))
        anchors_src = rng.normal(size=(40, d_src))
        anchors_tgt = anchors_src @ proj + 0.1 * rng.normal(size=(40, d_tgt))
        docs_new = rng.normal(size=(n_docs, d_src))
        docs_old = docs_new @ proj + 0.1 * rng.normal(size=(n_docs, d_tgt))
        q_eval = rng.normal(size=(8, d_src))
        q_dev = rng.normal(size=(6, d_src))
        doc_ids = [f"d{i}" for i in range(n_docs)]
        qids = [f"q{i}" for i in range(8)]
        qrels = {q: {f"d{(3 * i) % n_docs}": 1.0} for i, q in enumerate(qids)}
        mask = np.zeros(n_docs, dtype=bool)
        mask[::5] = True
        fact = run_factorial_5x3(
            domain="t", src_anchors=anchors_src, tgt_anchors=anchors_tgt,
            eval_queries_src=q_eval, eval_queries_tgt=q_eval, docs_old=docs_old, docs_new=docs_new,
            query_ids=qids, doc_ids=doc_ids, qrels=qrels, dev_queries_src=q_dev, dev_queries_tgt=q_dev,
            backfill_mask=mask, bridge_names=("CenteredProcrustes",), merge_names=("CSLSMerge",),
        )
        served = serve_scores(
            "CenteredProcrustes", src_anchors=anchors_src, tgt_anchors=anchors_tgt,
            queries_eval=q_eval, queries_dev=q_dev, docs_old=docs_old, docs_new=docs_new, backfill_mask=mask,
        )
        pq, _ = evaluate_score_matrix(served["fused_eval"], qids, doc_ids, qrels, k=10)
        self.assertAlmostEqual(float(np.mean(list(pq.values()))), fact[("CenteredProcrustes", "CSLSMerge")].mean_ndcg_at_10, places=12)
        self.assertEqual(served["fused_dev"].shape, (6, n_docs))

    def test_heldout_fit_error_zero_for_exact_rotation(self):
        rng = np.random.default_rng(9)
        q, _ = np.linalg.qr(rng.normal(size=(6, 6)))
        x = rng.normal(size=(30, 6))
        from chelated_pv7_spine.bridge_merge_harness import UncenteredProcrustes

        br = UncenteredProcrustes.fit(x, x @ q)
        held = rng.normal(size=(5, 6))
        self.assertAlmostEqual(heldout_fit_error(br, held, held @ q), 0.0, places=10)


if __name__ == "__main__":
    unittest.main()
