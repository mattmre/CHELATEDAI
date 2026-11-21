"""Unit tests for Flagship #2 (IQ-04 / EK7 / EK5) temporal evidence correction engine."""

from __future__ import annotations

import unittest

from iq04_temporal_evidence import (
    EvidenceEdge,
    EvidenceEvent,
    ScopedBM25,
    compute_full_recompute_eligibility,
    evaluate_arm_on_query,
    generate_iq04_lifecycle_corpus,
    sparse_invalidate_incremental,
)


class TestIQ04TemporalEvidence(unittest.TestCase):
    def test_splits_are_strictly_disjoint_across_entities_and_lifecycles(self):
        corpus = generate_iq04_lifecycle_corpus(n_dev=32, n_select=32, n_report=64, seed=42)
        by_split = {"DEVELOPMENT": set(), "SELECT": set(), "REPORT": set()}
        for q in corpus.queries:
            by_split[q.split].add(q.entity)
        self.assertTrue(by_split["DEVELOPMENT"].isdisjoint(by_split["SELECT"]))
        self.assertTrue(by_split["DEVELOPMENT"].isdisjoint(by_split["REPORT"]))
        self.assertTrue(by_split["SELECT"].isdisjoint(by_split["REPORT"]))

    def test_ek5_sparse_invalidation_matches_full_recompute_and_uses_bounded_work(self):
        corpus = generate_iq04_lifecycle_corpus(n_dev=40, n_select=0, n_report=0, seed=7)
        # Baseline before t=10 retractions
        base = compute_full_recompute_eligibility(corpus.events, corpus.edges, valid_cutoff=12, tx_cutoff=9)
        # Retraction/correction events introduced at t=10 in ONE lifecycle
        target_lc = "lc_dev_0003"
        changed = [
            eid for eid, ev in corpus.events.items()
            if ev.lifecycle_id == target_lc and ev.transaction_time == 10
        ]
        # Build partial event view where ONLY target_lc advanced to t=10 (simulating single incremental update)
        inc_events = {
            eid: ev for eid, ev in corpus.events.items()
            if ev.transaction_time <= 9 or ev.lifecycle_id == target_lc
        }
        inc_edges = [
            e for e in corpus.edges
            if e.transaction_time <= 9 or (e.src_id in inc_events and e.dst_id in inc_events)
        ]
        expected_full = compute_full_recompute_eligibility(inc_events, inc_edges, valid_cutoff=12, tx_cutoff=12)
        got_sparse, work = sparse_invalidate_incremental(
            base, inc_events, inc_edges, changed, valid_cutoff=12, tx_cutoff=12, horizon=8
        )
        self.assertEqual(got_sparse, expected_full)
        work_ratio = (work["visited_nodes"] + work["visited_edges"]) / float(
            max(work["total_nodes"] + work["total_edges"], 1)
        )
        self.assertLessEqual(work_ratio, 0.25)

    def test_reinstatement_restores_erroneously_retracted_claim(self):
        evs = {
            "e1": EvidenceEvent("e1", "lc", "ent", "slot", "GOOD", "ent slot GOOD", 1, 1, "s1", "CLAIM"),
            "e2": EvidenceEvent("e2", "lc", "ent", "slot", "BAD", "retract GOOD", 5, 5, "s2", "RETRACTION"),
            "e3": EvidenceEvent("e3", "lc", "ent", "slot", "GOOD", "reinstate GOOD", 9, 9, "s3", "REINSTATEMENT"),
        }
        edges = [
            EvidenceEdge("e2", "e1", "REFUTE", 5),
            EvidenceEdge("e3", "e2", "REFUTE", 9),
            EvidenceEdge("e3", "e1", "SUPPORT", 9),
        ]
        # Before reinstatement (tx=7): e1 is AUDIT_ONLY
        el_before = compute_full_recompute_eligibility(evs, edges, valid_cutoff=10, tx_cutoff=7)
        self.assertEqual(el_before["e1"], "AUDIT_ONLY")
        # After reinstatement (tx=10): e2 is refuted so e1 is restored to ADMIT!
        el_after = compute_full_recompute_eligibility(evs, edges, valid_cutoff=10, tx_cutoff=10)
        self.assertEqual(el_after["e1"], "ADMIT")
        self.assertEqual(el_after["e2"], "AUDIT_ONLY")

    def test_iq04_beats_unsigned_scalemas_and_ordinary_tms_on_unlinked_paraphrase_copy(self):
        corpus = generate_iq04_lifecycle_corpus(n_dev=16, n_select=0, n_report=0, seed=42)
        bm25 = ScopedBM25(corpus.events)
        q_copy = next(q for q in corpus.queries if q.case_family == "hidden_common_cause_paraphrase_copy" and q.is_post_correction)
        lc_eids = [eid for eid, ev in corpus.events.items() if ev.lifecycle_id == q_copy.lifecycle_id]
        lc_edges = [e for e in corpus.edges if corpus.events[e.src_id].lifecycle_id == q_copy.lifecycle_id]

        res_sm = evaluate_arm_on_query("C1_scalemas_unsigned", q_copy, lc_eids, lc_edges, corpus.events, bm25)
        res_c3 = evaluate_arm_on_query("C3_ordinary_bitemporal_tms", q_copy, lc_eids, lc_edges, corpus.events, bm25)
        res_c4 = evaluate_arm_on_query("C4_iq04_candidate", q_copy, lc_eids, lc_edges, corpus.events, bm25)

        self.assertTrue(res_sm["is_stale_or_false"])
        self.assertTrue(res_c3["is_stale_or_false"])
        self.assertTrue(res_c4["is_correct"])
        self.assertFalse(res_c4["is_stale_or_false"])


if __name__ == "__main__":
    unittest.main()
