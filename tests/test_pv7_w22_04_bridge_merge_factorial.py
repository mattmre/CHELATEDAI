"""W22-04 unit tests: RRFMerge, kill-rule operationalization, fidelity and CSLS retention."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chelated_pv7_spine.bridge_merge_harness import (  # noqa: E402
    MERGE_NAMES,
    MERGE_REGISTRY,
    RRFMerge,
    run_factorial_5x3,
)
from scripts.run_pv7_w22_04_bridge_merge_factorial import (  # noqa: E402
    ALL_BRIDGES,
    MERGES,
    _bf20_mask,
    analyze_domain,
    compare_arms,
    csls_retention,
    domain_decision,
    fidelity_compare,
    kill_verdict,
    summarize,
)


class TestRRFMerge(unittest.TestCase):
    def test_default_merge_names_unchanged(self):
        self.assertEqual(MERGE_NAMES, ("RawCosineMerge", "CalibratedZScoreMerge", "CSLSMerge"))
        self.assertIn("RRFMerge", MERGE_REGISTRY)

    def test_rrf_scores_are_reciprocal_within_shard_ranks(self):
        bridged = np.array([[0.9, 0.1, 0.5, 0.0]])
        native = np.array([[0.0, 0.0, 0.0, 0.7]])
        mask = np.array([False, False, False, True])
        out = RRFMerge(rrf_k=60).merge(bridged, native, mask)
        np.testing.assert_allclose(out[0], [1 / 61, 1 / 63, 1 / 62, 1 / 61])

    def test_rrf_ignores_cross_shard_scale(self):
        rng = np.random.default_rng(0)
        bridged = rng.normal(size=(3, 8))
        native = rng.normal(size=(3, 8))
        mask = np.array([True, False] * 4)
        a = RRFMerge().merge(bridged, native, mask)
        b = RRFMerge().merge(bridged, native * 1000.0 + 50.0, mask)
        np.testing.assert_allclose(a, b)

    def test_rrf_ties_broken_by_column_index(self):
        bridged = np.array([[0.5, 0.5, 0.5]])
        out = RRFMerge(rrf_k=0).merge(bridged, bridged, np.zeros(3, dtype=bool))
        np.testing.assert_allclose(out[0], [1.0, 0.5, 1.0 / 3.0])

    def test_rrf_shape_guard(self):
        with self.assertRaises(ValueError):
            RRFMerge().merge(np.zeros((2, 3)), np.zeros((2, 4)), np.zeros(3, dtype=bool))


class TestKillRule(unittest.TestCase):
    def test_identical_arms_are_within(self):
        pq = {f"q{i}": float(i % 3) / 3 for i in range(40)}
        r = compare_arms(pq, dict(pq))
        self.assertEqual(r["delta_zeromu_minus_comparator"], 0.0)
        self.assertTrue(r["within_primary"])
        self.assertEqual(r["direction_primary"], "within")

    def test_large_resolved_gap_is_not_within(self):
        z = {f"q{i}": 0.8 for i in range(50)}
        c = {f"q{i}": 0.5 + 0.01 * (i % 5) for i in range(50)}
        r = compare_arms(z, c)
        self.assertFalse(r["within_primary"])
        self.assertEqual(r["direction_primary"], "zeromu_better")
        r2 = compare_arms(c, z)
        self.assertEqual(r2["direction_primary"], "comparator_better")

    def test_noisy_small_gap_within_via_ci(self):
        rng = np.random.default_rng(3)
        z = {f"q{i}": float(rng.random()) for i in range(30)}
        c = {k: float(np.clip(v + rng.normal(0, 0.3), 0, 1)) for k, v in z.items()}
        r = compare_arms(z, c)
        if not r["within_margin"]:
            self.assertEqual(r["within_primary"], r["paired_ci_contains_zero"])

    def test_query_set_mismatch_raises(self):
        with self.assertRaises(ValueError):
            compare_arms({"a": 1.0}, {"b": 1.0})

    def test_domain_within_if_any_comparator(self):
        d = domain_decision(
            {"CenteredProcrustes": {"within_primary": False, "within_secondary": True},
             "UncenteredProcrustes": {"within_primary": True, "within_secondary": True}}
        )
        self.assertTrue(d["within_primary"])
        self.assertEqual(d["within_primary_via"], ["UncenteredProcrustes"])

    def test_kill_verdict_thresholds(self):
        self.assertEqual(kill_verdict([True] * 10 + [False] * 2)["verdict"], "KILL")
        self.assertEqual(kill_verdict([True] * 9 + [False] * 3)["verdict"], "NOT_KILLED")
        self.assertEqual(kill_verdict([True] * 7)["verdict"], "UNDECIDED_DEFERRED_SCOPE")
        self.assertEqual(kill_verdict([False] * 3)["verdict"], "NOT_KILLED")
        self.assertEqual(kill_verdict([True] * 10)["verdict"], "KILL")


class TestFidelityAndRetention(unittest.TestCase):
    def test_fidelity_flags_large_diff(self):
        f = fidelity_compare({"A__B": 0.50, "C__D": 0.30}, {"A__B": 0.50, "C__D": 0.35})
        self.assertEqual(f["n_cells_compared"], 2)
        self.assertAlmostEqual(f["max_abs_diff"], 0.05)
        self.assertTrue(f["flagged"])
        self.assertFalse(fidelity_compare({"A__B": 0.5}, {"A__B": 0.51})["flagged"])

    def test_csls_retention_picks_best_other(self):
        q = [f"q{i}" for i in range(20)]
        cells = {
            "RawCosineMerge": {k: 0.2 for k in q},
            "CalibratedZScoreMerge": {k: 0.4 for k in q},
            "CSLSMerge": {k: 0.6 for k in q},
            "RRFMerge": {k: 0.3 for k in q},
        }
        r = csls_retention(cells)
        self.assertEqual(r["best_non_csls_merge"], "CalibratedZScoreMerge")
        self.assertAlmostEqual(r["csls_minus_best_other"], 0.2)
        self.assertTrue(r["csls_is_best_merge"])


class TestEndToEndSynthetic(unittest.TestCase):
    def test_full_5x4_factorial_and_analysis(self):
        rng = np.random.default_rng(11)
        n_docs, n_q, n_dev, n_anc = 60, 16, 8, 120
        d_src, d_tgt = 24, 12
        latent = rng.normal(size=(n_docs + n_q + n_dev + n_anc + 20, 12))
        a_src = rng.normal(size=(12, d_src))
        a_tgt = rng.normal(size=(12, d_tgt))
        src_all = latent @ a_src + 0.05 * rng.normal(size=(latent.shape[0], d_src))
        tgt_all = latent @ a_tgt + 0.05 * rng.normal(size=(latent.shape[0], d_tgt))
        i0, i1, i2, i3 = n_docs, n_docs + n_q, n_docs + n_q + n_dev, n_docs + n_q + n_dev + n_anc
        doc_ids = [f"d{i:03d}" for i in range(n_docs)]
        qids = [f"q{i:03d}" for i in range(n_q)]
        qrels = {q: {doc_ids[(3 * i) % n_docs]: 1.0} for i, q in enumerate(qids)}
        q_src = src_all[i0:i1].copy()
        for i in range(n_q):
            q_src[i] = src_all[(3 * i) % n_docs] + 0.3 * rng.normal(size=d_src)
        fact = run_factorial_5x3(
            domain="synthetic",
            src_anchors=src_all[i2:i3],
            tgt_anchors=tgt_all[i2:i3],
            eval_queries_src=q_src,
            eval_queries_tgt=q_src,
            docs_old=tgt_all[:n_docs],
            docs_new=src_all[:n_docs],
            query_ids=qids,
            doc_ids=doc_ids,
            qrels=qrels,
            dev_queries_src=src_all[i1:i2],
            dev_queries_tgt=src_all[i1:i2],
            dev_docs=tgt_all[i3:],
            backfill_mask=_bf20_mask("synthetic", n_docs),
            bridge_names=ALL_BRIDGES,
            merge_names=MERGES,
        )
        self.assertEqual(len(fact), 20)
        rec = analyze_domain(
            "synthetic",
            fact,
            recorded_cells={"ZeroMuPolarBridge__CSLSMerge": fact[("ZeroMuPolarBridge", "CSLSMerge")].mean_ndcg_at_10},
        )
        self.assertEqual(rec["fidelity"]["max_abs_diff"], 0.0)
        self.assertEqual(len(rec["cells_ndcg10"]), 20)
        self.assertIn("CenteredProcrustes", rec["kill_rule"]["per_comparator"])
        self.assertEqual(len(rec["csls_retention"]), 5)
        s = summarize({"synthetic": rec})
        self.assertEqual(s["primary"]["n_domains_run"], 1)
        self.assertIn(s["primary"]["verdict"], {"UNDECIDED_DEFERRED_SCOPE", "NOT_KILLED"})

    def test_bf20_mask_size(self):
        m = _bf20_mask("cqadupstack-android", 750)
        self.assertEqual(int(m.sum()), 150)


if __name__ == "__main__":
    unittest.main()
