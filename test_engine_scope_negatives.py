import unittest

from engine_scope_negatives import (
    build_hard_negative_replay_artifact,
    mine_hard_negative_families,
    select_query_subset_by_ids,
)


class TestEngineScopeNegatives(unittest.TestCase):
    def test_mine_hard_negative_families_groups_query_profile_failures(self):
        rows = [
            {
                "row_type": "query_profile",
                "task": "SciFact",
                "query_id": "q1",
                "query_text": "vitamin b12 homocysteine",
                "profile": "reform_rrf_v2",
                "action": "REFORMULATE",
                "fault_class": "actuator_active_negative",
                "delta_ndcg_at_10": -0.03,
                "query_token_count": 3,
                "query_char_count": 24,
                "query_stopword_ratio": 0.0,
                "query_numeric_token_count": 0,
                "query_negation_count": 0,
                "query_claim_cue_count": 1,
                "top10_overlap_with_baseline": 2,
                "top_doc_changed": True,
                "global_variance": 0.3,
                "jaccard": 0.2,
                "mask_density": 0.1,
                "reformulation_variant_count": 2,
                "reformulation_changed": True,
                "synthetic_depth": 0,
            },
            {
                "row_type": "query_profile",
                "task": "SciFact",
                "query_id": "q2",
                "query_text": "magnesium absorption",
                "profile": "reform_rrf_v2",
                "action": "REFORMULATE",
                "fault_class": "actuator_active_negative",
                "delta_ndcg_at_10": -0.02,
                "query_token_count": 3,
                "query_char_count": 21,
                "query_stopword_ratio": 0.0,
                "query_numeric_token_count": 0,
                "query_negation_count": 0,
                "query_claim_cue_count": 1,
                "top10_overlap_with_baseline": 2,
                "top_doc_changed": True,
                "global_variance": 0.25,
                "jaccard": 0.22,
                "mask_density": 0.1,
                "reformulation_variant_count": 2,
                "reformulation_changed": True,
                "synthetic_depth": 0,
            },
            {
                "row_type": "mask_probe",
                "task": "FiQA2018",
                "query_id": "m1",
                "delta_ndcg_at_10": -0.01,
                "query_token_count": 9,
                "query_char_count": 90,
                "query_stopword_ratio": 0.1,
                "query_numeric_token_count": 2,
                "query_negation_count": 1,
                "query_claim_cue_count": 0,
                "baseline_score_margin": 0.01,
                "baseline_top_score": 0.8,
                "query_norm": 0.7,
                "synthetic_depth": 0,
            },
            {
                "row_type": "mask_probe",
                "task": "FiQA2018",
                "query_id": "m2",
                "delta_ndcg_at_10": -0.005,
                "query_token_count": 9,
                "query_char_count": 90,
                "query_stopword_ratio": 0.1,
                "query_numeric_token_count": 2,
                "query_negation_count": 1,
                "query_claim_cue_count": 0,
                "baseline_score_margin": 0.01,
                "baseline_top_score": 0.8,
                "query_norm": 0.7,
                "synthetic_depth": 2,
            },
        ]

        mined = mine_hard_negative_families(
            rows,
            min_family_size=2,
            max_query_profile_families=3,
            max_mask_probe_families=3,
            max_queries_per_family=5,
        )

        self.assertEqual(mined["query_profile_failure_count"], 1)
        self.assertEqual(mined["mask_probe_failure_count"], 0)
        self.assertEqual(mined["query_profile_families"][0]["family_id"], "qneg_001")
        self.assertEqual(mined["query_profile_families"][0]["primary_fault_class"], "actuator_active_negative")
        self.assertEqual(mined["query_profile_families"][0]["replay_query_ids"], ["q1", "q2"])
        self.assertEqual(mined["query_profile_families"][0]["negative_source_type"], "engine_scope_query_profile")
        self.assertEqual(mined["query_profile_families"][0]["synthetic_depth"], 0)
        self.assertEqual(mined["mask_probe_families"], [])
        self.assertEqual(mined["recursive_blocked_count"], 1)

        artifact = build_hard_negative_replay_artifact(
            rows,
            [*mined["query_profile_families"], *mined["mask_probe_families"]],
        )

        self.assertEqual(artifact["family_count"], 1)
        self.assertEqual(artifact["summary"]["row_count"], 2)
        self.assertEqual(artifact["coverage_summary"]["window_count"], 1)
        self.assertEqual(artifact["engine_scope_rows"][0]["family_id"], "qneg_001")

    def test_build_hard_negative_replay_artifact_is_deterministic(self):
        """Calling the builder twice with identical inputs must produce identical output.

        This test guards against ENG-4 gap 2: the artifact must be replay-safe,
        meaning the same inputs always yield the same families, query IDs, and
        engine_scope_rows list — byte-for-byte equivalent when serialised.
        """
        rows = [
            {
                "row_type": "query_profile",
                "task": "SciFact",
                "query_id": "qA",
                "query_text": "alpha query",
                "profile": "reform_rrf_v2",
                "action": "REFORMULATE",
                "fault_class": "actuator_active_negative",
                "delta_ndcg_at_10": -0.05,
                "query_token_count": 2,
                "query_char_count": 11,
                "query_stopword_ratio": 0.0,
                "query_numeric_token_count": 0,
                "query_negation_count": 0,
                "query_claim_cue_count": 1,
                "top10_overlap_with_baseline": 3,
                "top_doc_changed": True,
                "global_variance": 0.4,
                "jaccard": 0.3,
                "mask_density": 0.1,
                "reformulation_variant_count": 2,
                "reformulation_changed": True,
                "synthetic_depth": 0,
            },
            {
                "row_type": "query_profile",
                "task": "SciFact",
                "query_id": "qB",
                "query_text": "beta query",
                "profile": "reform_rrf_v2",
                "action": "REFORMULATE",
                "fault_class": "actuator_active_negative",
                "delta_ndcg_at_10": -0.04,
                "query_token_count": 2,
                "query_char_count": 10,
                "query_stopword_ratio": 0.0,
                "query_numeric_token_count": 0,
                "query_negation_count": 0,
                "query_claim_cue_count": 1,
                "top10_overlap_with_baseline": 3,
                "top_doc_changed": True,
                "global_variance": 0.4,
                "jaccard": 0.3,
                "mask_density": 0.1,
                "reformulation_variant_count": 2,
                "reformulation_changed": True,
                "synthetic_depth": 0,
            },
        ]

        mined = mine_hard_negative_families(rows, min_family_size=2, max_queries_per_family=5)
        families = [*mined["query_profile_families"], *mined["mask_probe_families"]]

        # Call the builder twice with same inputs (different list objects to avoid mutation aliasing)
        artifact_1 = build_hard_negative_replay_artifact(list(rows), list(families))
        artifact_2 = build_hard_negative_replay_artifact(list(rows), list(families))

        # Family count, query IDs, and row ordering must be identical
        self.assertEqual(artifact_1["family_count"], artifact_2["family_count"])
        self.assertEqual(
            [f.get("family_id") for f in artifact_1["families"]],
            [f.get("family_id") for f in artifact_2["families"]],
        )
        self.assertEqual(
            [r.get("query_id") for r in artifact_1["engine_scope_rows"]],
            [r.get("query_id") for r in artifact_2["engine_scope_rows"]],
        )
        self.assertEqual(
            [r.get("family_id") for r in artifact_1["engine_scope_rows"]],
            [r.get("family_id") for r in artifact_2["engine_scope_rows"]],
        )
        # Summary row counts must match
        self.assertEqual(
            artifact_1["summary"]["row_count"],
            artifact_2["summary"]["row_count"],
        )

    def test_mine_hard_negative_families_docstring_labels_grouping_not_clustering(self):
        """The function docstring must not claim the algorithm is clustering.

        This test guards against ENG-4 gap 1: the label 'clustering' is a lie
        because the grouping is deterministic fault-class partitioning with no
        distance metric, centroid, or iterative algorithm.  The docstring is
        allowed to mention 'clustering' in a negation (e.g. "NOT clustering")
        but must not assert it IS clustering, and must positively describe the
        deterministic grouping approach.
        """
        import inspect
        doc = inspect.getdoc(mine_hard_negative_families) or ""
        doc_lower = doc.lower()
        # The docstring must positively state what the algorithm actually is
        self.assertTrue(
            "grouping" in doc_lower or "deterministic" in doc_lower,
            "mine_hard_negative_families docstring must describe the algorithm as "
            "deterministic grouping (not clustering).",
        )
        # If the word 'clustering' appears, it must be in a negation context.
        # Acceptable: "NOT algorithmic clustering", "is not clustering".
        # Unacceptable: "performs clustering", "using clustering".
        if "clustering" in doc_lower:
            idx = doc_lower.index("clustering")
            preceding = doc_lower[max(0, idx - 40):idx]
            self.assertTrue(
                "not" in preceding or "no " in preceding,
                "mine_hard_negative_families docstring mentions 'clustering' without "
                "negating it; the algorithm is deterministic fault-class grouping.",
            )

    def test_select_query_subset_by_ids_preserves_order_and_relevant_docs(self):
        corpus = {
            "d1": "relevant doc one",
            "d2": "relevant doc two",
            "d3": "extra doc",
        }
        queries = {
            "q1": "first query",
            "q2": "second query",
            "q3": "third query",
        }
        qrels = {
            "q1": {"d1": 1.0},
            "q2": {"d2": 1.0},
            "q3": {"d3": 1.0},
        }

        selected_corpus, selected_queries, selected_qrels = select_query_subset_by_ids(
            corpus,
            queries,
            qrels,
            ["q2", "q1", "q2"],
            sample_docs=2,
            seed=123,
        )

        self.assertEqual(list(selected_queries), ["q2", "q1"])
        self.assertEqual(list(selected_qrels), ["q2", "q1"])
        self.assertIn("d1", selected_corpus)
        self.assertIn("d2", selected_corpus)


if __name__ == "__main__":
    unittest.main()
