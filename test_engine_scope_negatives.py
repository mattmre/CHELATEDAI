import unittest

from engine_scope_negatives import (
    _mask_probe_negative_rows,
    _query_profile_negative_rows,
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


    # ENG-2 / ENG-3: Fail-closed gate filtering and coverage overlap paths

    def test_query_profile_negative_rows_excludes_positive_delta(self):
        """Rows with delta_ndcg_at_10 > 0.001 are excluded (fail-closed: don't promote positives)."""
        rows = [
            {
                "row_type": "query_profile",
                "source_family": "normal",
                "profile": "reform",
                "fault_class": "actuator_active_negative",
                "delta_ndcg_at_10": 0.01,
                "rank_delta": -1,
            },
            {
                "row_type": "query_profile",
                "source_family": "normal",
                "profile": "reform",
                "fault_class": "actuator_active_negative",
                "delta_ndcg_at_10": -0.01,
                "rank_delta": 0,
            },
        ]
        result = _query_profile_negative_rows(rows)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["delta_ndcg_at_10"], -0.01)

    def test_query_profile_negative_rows_excludes_hard_negative_source_family(self):
        """Rows already sourced from a hard-negative family are excluded to avoid recursion."""
        rows = [
            {
                "row_type": "query_profile",
                "source_family": "hard_negative_family_1",
                "profile": "reform",
                "fault_class": "actuator_active_negative",
                "delta_ndcg_at_10": -0.01,
                "rank_delta": 0,
            },
            {
                "row_type": "query_profile",
                "source_family": "normal_collection",
                "profile": "reform",
                "fault_class": "actuator_active_negative",
                "delta_ndcg_at_10": -0.01,
                "rank_delta": 0,
            },
        ]
        result = _query_profile_negative_rows(rows)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["source_family"], "normal_collection")

    def test_query_profile_negative_rows_excludes_baseline_profile(self):
        """Baseline profile rows are excluded — they represent the unmodified control."""
        rows = [
            {
                "row_type": "query_profile",
                "source_family": "normal",
                "profile": "baseline",
                "fault_class": "actuator_active_negative",
                "delta_ndcg_at_10": -0.01,
                "rank_delta": 0,
            },
        ]
        result = _query_profile_negative_rows(rows)
        self.assertEqual(result, [])

    def test_query_profile_negative_rows_excludes_unknown_fault_class(self):
        """Only recognized fault classes are admitted into hard-negative families."""
        rows = [
            {
                "row_type": "query_profile",
                "source_family": "normal",
                "profile": "reform",
                "fault_class": "completely_unknown_fault",
                "delta_ndcg_at_10": -0.01,
                "rank_delta": 0,
            },
        ]
        result = _query_profile_negative_rows(rows)
        self.assertEqual(result, [])

    def test_query_profile_negative_rows_treats_bad_rank_delta_as_non_improving(self):
        """Unparsable rank_delta is treated as non-improving (fail-closed default)."""
        rows = [
            {
                "row_type": "query_profile",
                "source_family": "normal",
                "profile": "reform",
                "fault_class": "actuator_active_negative",
                "delta_ndcg_at_10": -0.01,
                "rank_delta": "unparsable",
            },
        ]
        result = _query_profile_negative_rows(rows)
        self.assertEqual(len(result), 1)

    def test_query_profile_negative_rows_excludes_wrong_row_type(self):
        rows = [{"row_type": "mask_probe", "source_family": "normal", "profile": "reform",
                 "fault_class": "actuator_active_negative", "delta_ndcg_at_10": -0.01}]
        result = _query_profile_negative_rows(rows)
        self.assertEqual(result, [])

    def test_mask_probe_negative_rows_excludes_positive_delta(self):
        rows = [
            {"row_type": "mask_probe", "source_family": "normal", "delta_ndcg_at_10": 0.01},
            {"row_type": "mask_probe", "source_family": "normal", "delta_ndcg_at_10": -0.01},
        ]
        result = _mask_probe_negative_rows(rows)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["delta_ndcg_at_10"], -0.01)

    def test_mask_probe_negative_rows_excludes_hard_negative_source_family(self):
        rows = [
            {"row_type": "mask_probe", "source_family": "hard_negative_family_2", "delta_ndcg_at_10": -0.01},
            {"row_type": "mask_probe", "source_family": "normal", "delta_ndcg_at_10": -0.01},
        ]
        result = _mask_probe_negative_rows(rows)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["source_family"], "normal")

    def test_mask_probe_negative_rows_excludes_bad_delta(self):
        rows = [{"row_type": "mask_probe", "source_family": "normal", "delta_ndcg_at_10": "bad"}]
        result = _mask_probe_negative_rows(rows)
        self.assertEqual(result, [])

    def test_mine_hard_negative_families_filters_by_max_synthetic_depth(self):
        """Rows at synthetic_depth > max_synthetic_depth must be blocked."""
        base_row = {
            "row_type": "query_profile",
            "task": "SciFact",
            "query_id": "q_depth",
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
        }
        rows = [
            {**base_row, "query_id": "q1", "synthetic_depth": 1},
            {**base_row, "query_id": "q2", "synthetic_depth": 1},
        ]
        mined_depth0 = mine_hard_negative_families(rows, min_family_size=2, max_synthetic_depth=0)
        mined_depth1 = mine_hard_negative_families(rows, min_family_size=2, max_synthetic_depth=1)
        # depth=0 blocks both rows
        self.assertEqual(mined_depth0["recursive_blocked_count"], 2)
        self.assertEqual(mined_depth0["query_profile_failure_count"], 0)
        # depth=1 admits both rows
        self.assertEqual(mined_depth1["recursive_blocked_count"], 0)
        self.assertEqual(mined_depth1["query_profile_failure_count"], 1)

    def test_mine_hard_negative_families_min_family_size_filter(self):
        """Families with fewer rows than min_family_size are excluded."""
        base_row = {
            "row_type": "query_profile",
            "task": "SciFact",
            "query_id": "q_only",
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
        }
        mined_size2 = mine_hard_negative_families([base_row], min_family_size=2)
        mined_size1 = mine_hard_negative_families([base_row], min_family_size=1)
        self.assertEqual(mined_size2["query_profile_failure_count"], 0)
        self.assertEqual(mined_size1["query_profile_failure_count"], 1)

    def test_build_hard_negative_replay_artifact_empty_input(self):
        artifact = build_hard_negative_replay_artifact([], [])
        self.assertEqual(artifact["family_count"], 0)
        self.assertEqual(artifact["families"], [])
        self.assertEqual(artifact["engine_scope_rows"], [])
        self.assertEqual(artifact["summary"]["row_count"], 0)

    def test_select_query_subset_by_ids_raises_when_no_judged_ids(self):
        with self.assertRaises(ValueError):
            select_query_subset_by_ids(
                {"d1": "doc"},
                {"q1": "query"},
                {"q1": {"d1": 1.0}},
                ["missing_id"],
                sample_docs=1,
                seed=0,
            )

    def test_select_query_subset_by_ids_deduplicates_query_ids(self):
        corpus = {"d1": "doc one", "d2": "doc two"}
        queries = {"q1": "first", "q2": "second"}
        qrels = {"q1": {"d1": 1.0}, "q2": {"d2": 1.0}}
        # q1 is listed twice — should appear only once in output
        selected_corpus, selected_queries, selected_qrels = select_query_subset_by_ids(
            corpus, queries, qrels, ["q1", "q2", "q1"], sample_docs=2, seed=0
        )
        self.assertEqual(list(selected_queries), ["q1", "q2"])
        self.assertIn("d1", selected_corpus)
        self.assertIn("d2", selected_corpus)

    def test_select_query_subset_by_ids_excludes_queries_without_qrels(self):
        """Queries with no positive relevance judgments are excluded from the subset."""
        corpus = {"d1": "doc"}
        queries = {"q1": "judged", "q2": "unjudged"}
        qrels = {"q1": {"d1": 1.0}}
        selected_corpus, selected_queries, selected_qrels = select_query_subset_by_ids(
            corpus, queries, qrels, ["q1", "q2"], sample_docs=1, seed=0
        )
        self.assertIn("q1", selected_queries)
        self.assertNotIn("q2", selected_queries)


if __name__ == "__main__":
    unittest.main()
