import unittest

from engine_scope_coverage import (
    _bucket,
    _count_bucket,
    _delta_bucket,
    _jaccard,
    _window_key,
    build_window_footprints,
    rank_task_offset_candidates,
    row_feature_tokens,
    select_task_offset_candidates,
    summarize_engine_scope_coverage,
)


class TestEngineScopeCoverage(unittest.TestCase):
    def test_row_feature_tokens_cover_query_and_mask_surfaces(self):
        query_tokens = row_feature_tokens({
            "row_type": "query_profile",
            "profile": "reform_rrf_v2",
            "delta_ndcg_at_10": 0.02,
            "query_token_count": 3,
            "query_char_count": 24,
            "query_stopword_ratio": 0.4,
            "query_numeric_token_count": 0,
            "query_negation_count": 0,
            "query_claim_cue_count": 1,
            "action": "REFORMULATE",
            "fault_class": "actuator_active_positive",
            "top10_overlap_with_baseline": 2,
            "top_doc_changed": True,
            "global_variance": 0.3,
            "jaccard": 0.2,
            "mask_density": 0.1,
            "reformulation_variant_count": 2,
            "reformulation_changed": True,
        })
        mask_tokens = row_feature_tokens({
            "row_type": "mask_probe",
            "delta_ndcg_at_10": -0.02,
            "query_token_count": 9,
            "query_char_count": 90,
            "query_stopword_ratio": 0.1,
            "query_numeric_token_count": 2,
            "query_negation_count": 1,
            "query_claim_cue_count": 0,
            "baseline_score_margin": 0.01,
            "baseline_top_score": 0.8,
            "query_norm": 0.7,
        })

        self.assertIn("row_type:query_profile", query_tokens)
        self.assertIn("action:REFORMULATE", query_tokens)
        self.assertIn("delta:positive", query_tokens)
        self.assertIn("row_type:mask_probe", mask_tokens)
        self.assertIn("margin:low", mask_tokens)
        self.assertIn("delta:negative", mask_tokens)

    def test_build_window_footprints_groups_rows_by_window_context(self):
        footprints = build_window_footprints([
            {
                "source_family": "reformulation_collection",
                "task": "SciFact",
                "query_offset": 0,
                "seed": 260,
                "row_type": "query_profile",
                "profile": "baseline",
                "delta_ndcg_at_10": 0.0,
                "query_token_count": 4,
                "query_char_count": 25,
                "query_stopword_ratio": 0.25,
                "query_numeric_token_count": 0,
                "query_negation_count": 0,
                "query_claim_cue_count": 0,
            },
            {
                "source_family": "reformulation_collection",
                "task": "SciFact",
                "query_offset": 0,
                "seed": 260,
                "row_type": "query_profile",
                "profile": "reform_rrf_v2",
                "delta_ndcg_at_10": 0.01,
                "query_token_count": 4,
                "query_char_count": 25,
                "query_stopword_ratio": 0.25,
                "query_numeric_token_count": 0,
                "query_negation_count": 0,
                "query_claim_cue_count": 0,
            },
        ])

        self.assertEqual(len(footprints), 1)
        self.assertEqual(footprints[0]["row_count"], 2)
        self.assertEqual(footprints[0]["profiles"]["baseline"], 1)
        self.assertEqual(footprints[0]["profiles"]["reform_rrf_v2"], 1)

    def test_summarize_engine_scope_coverage_reports_redundancy_and_novelty(self):
        rows = [
            {
                "source_family": "reformulation_collection",
                "task": "SciFact",
                "query_offset": 0,
                "seed": 260,
                "row_type": "query_profile",
                "profile": "reform_rrf_v2",
                "delta_ndcg_at_10": 0.02,
                "query_token_count": 3,
                "query_char_count": 24,
                "query_stopword_ratio": 0.4,
                "query_numeric_token_count": 0,
                "query_negation_count": 0,
                "query_claim_cue_count": 1,
                "action": "REFORMULATE",
                "fault_class": "actuator_active_positive",
                "top10_overlap_with_baseline": 2,
                "top_doc_changed": True,
                "global_variance": 0.3,
                "jaccard": 0.2,
                "mask_density": 0.1,
                "reformulation_variant_count": 2,
                "reformulation_changed": True,
            },
            {
                "source_family": "reformulation_collection",
                "task": "NFCorpus",
                "query_offset": 100,
                "seed": 260,
                "row_type": "query_profile",
                "profile": "reform_rrf_v2",
                "delta_ndcg_at_10": 0.02,
                "query_token_count": 3,
                "query_char_count": 24,
                "query_stopword_ratio": 0.4,
                "query_numeric_token_count": 0,
                "query_negation_count": 0,
                "query_claim_cue_count": 1,
                "action": "REFORMULATE",
                "fault_class": "actuator_active_positive",
                "top10_overlap_with_baseline": 2,
                "top_doc_changed": True,
                "global_variance": 0.3,
                "jaccard": 0.2,
                "mask_density": 0.1,
                "reformulation_variant_count": 2,
                "reformulation_changed": True,
            },
            {
                "source_family": "mask_collection",
                "task": "FiQA2018",
                "query_offset": 200,
                "seed": 360,
                "split": "holdout",
                "row_type": "mask_probe",
                "delta_ndcg_at_10": -0.02,
                "query_token_count": 9,
                "query_char_count": 90,
                "query_stopword_ratio": 0.1,
                "query_numeric_token_count": 2,
                "query_negation_count": 1,
                "query_claim_cue_count": 0,
                "baseline_score_margin": 0.01,
                "baseline_top_score": 0.8,
                "query_norm": 0.7,
            },
        ]

        summary = summarize_engine_scope_coverage(rows, novelty_threshold=0.3, redundancy_threshold=0.95)

        self.assertEqual(summary["window_count"], 3)
        self.assertEqual(summary["redundant_window_count"], 1)
        self.assertEqual(summary["top_overlap_pairs"][0]["overlap"], 1.0)
        self.assertEqual(len(summary["task_offset_summary"]), 3)
        self.assertGreaterEqual(summary["novel_window_count"], 1)

    def test_rank_task_offset_candidates_prioritizes_unseen_offsets(self):
        coverage_summary = {
            "task_offset_summary": [
                {
                    "task": "SciFact",
                    "query_offset": 0,
                    "window_count": 4,
                    "row_count": 200,
                    "token_count": 40,
                    "source_families": ["reformulation_collection"],
                    "max_overlap_with_previous": 0.92,
                    "max_novelty_ratio": 0.12,
                },
                {
                    "task": "NFCorpus",
                    "query_offset": 0,
                    "window_count": 2,
                    "row_count": 80,
                    "token_count": 30,
                    "source_families": ["reformulation_collection"],
                    "max_overlap_with_previous": 0.75,
                    "max_novelty_ratio": 0.2,
                },
            ]
        }

        ranked = rank_task_offset_candidates(
            [
                ("SciFact", 0),
                ("SciFact", 100),
                ("NFCorpus", 0),
                ("NFCorpus", 100),
                ("FiQA2018", 0),
            ],
            coverage_summary,
        )

        self.assertEqual((ranked[0]["task"], ranked[0]["query_offset"]), ("FiQA2018", 0))
        self.assertEqual((ranked[1]["task"], ranked[1]["query_offset"]), ("NFCorpus", 100))
        self.assertEqual((ranked[-1]["task"], ranked[-1]["query_offset"]), ("SciFact", 0))


    # ENG-3: Feature Coverage & Overlap — missing edge-case coverage

    def test_bucket_boundary_and_unknown_values(self):
        self.assertEqual(_bucket(None, low=0.5, high=0.8), "unknown")
        self.assertEqual(_bucket("bad", low=0.5, high=0.8), "unknown")
        self.assertEqual(_bucket(0.5, low=0.5, high=0.8), "low")
        self.assertEqual(_bucket(0.8, low=0.5, high=0.8), "mid")
        self.assertEqual(_bucket(0.9, low=0.5, high=0.8), "high")

    def test_count_bucket_boundary_and_unknown_values(self):
        self.assertEqual(_count_bucket(None, low=0, high=1), "unknown")
        self.assertEqual(_count_bucket("bad", low=0, high=1), "unknown")
        self.assertEqual(_count_bucket(0, low=0, high=1), "low")
        self.assertEqual(_count_bucket(1, low=0, high=1), "mid")
        self.assertEqual(_count_bucket(2, low=0, high=1), "high")

    def test_delta_bucket_boundary_and_unknown_values(self):
        self.assertEqual(_delta_bucket(None), "unknown")
        self.assertEqual(_delta_bucket("bad"), "unknown")
        self.assertEqual(_delta_bucket(0.002), "positive")
        self.assertEqual(_delta_bucket(-0.002), "negative")
        self.assertEqual(_delta_bucket(0.0), "neutral")
        self.assertEqual(_delta_bucket(0.001), "neutral")

    def test_jaccard_empty_sets_returns_zero(self):
        self.assertEqual(_jaccard([], []), 0.0)
        self.assertEqual(_jaccard(["a"], []), 0.0)
        self.assertEqual(_jaccard([], ["b"]), 0.0)
        self.assertEqual(_jaccard(["a"], ["a"]), 1.0)
        overlap = _jaccard(["a", "b"], ["b", "c"])
        self.assertAlmostEqual(overlap, 1 / 3)

    def test_row_feature_tokens_mask_probe_type(self):
        tokens = row_feature_tokens({
            "row_type": "mask_probe",
            "delta_ndcg_at_10": -0.05,
            "query_token_count": 9,
            "query_char_count": 90,
            "query_stopword_ratio": 0.1,
            "query_numeric_token_count": 2,
            "query_negation_count": 1,
            "query_claim_cue_count": 0,
            "baseline_score_margin": 0.01,
            "baseline_top_score": 0.8,
            "query_norm": 0.7,
        })
        self.assertIn("row_type:mask_probe", tokens)
        self.assertIn("margin:low", tokens)
        self.assertIn("top_score:high", tokens)
        self.assertIn("query_norm:low", tokens)
        self.assertIn("delta:negative", tokens)
        # mask_probe should NOT have query_profile-only tokens
        self.assertNotIn("action:unknown", tokens)
        self.assertNotIn("fault:unknown", tokens)

    def test_row_feature_tokens_unknown_row_type(self):
        tokens = row_feature_tokens({"row_type": "novel_type"})
        self.assertIn("row_type:novel_type", tokens)
        self.assertIn("delta:unknown", tokens)

    def test_build_window_footprints_single_window_is_fully_novel(self):
        """First window has no predecessor so all its tokens are novel."""
        footprints = build_window_footprints([
            {
                "source_family": "sf",
                "task": "SciFact",
                "query_offset": 0,
                "seed": 1,
                "row_type": "query_profile",
                "profile": "reform",
                "delta_ndcg_at_10": 0.02,
                "query_token_count": 3,
                "query_char_count": 20,
                "query_stopword_ratio": 0.2,
                "query_numeric_token_count": 0,
                "query_negation_count": 0,
                "query_claim_cue_count": 0,
            }
        ])
        self.assertEqual(len(footprints), 1)
        # Single window: novel_token_count == token_count (no prior universe)
        self.assertEqual(footprints[0]["row_count"], 1)
        self.assertIn("feature_tokens", footprints[0])
        self.assertGreater(footprints[0]["token_count"], 0)

    def test_summarize_engine_scope_coverage_single_window_all_novel(self):
        rows = [
            {
                "source_family": "sf",
                "task": "SciFact",
                "query_offset": 0,
                "seed": 1,
                "row_type": "query_profile",
                "profile": "reform",
                "delta_ndcg_at_10": 0.02,
                "query_token_count": 3,
                "query_char_count": 20,
                "query_stopword_ratio": 0.2,
                "query_numeric_token_count": 0,
                "query_negation_count": 0,
                "query_claim_cue_count": 0,
            }
        ]
        summary = summarize_engine_scope_coverage(rows, novelty_threshold=0.5, redundancy_threshold=0.95)
        self.assertEqual(summary["window_count"], 1)
        self.assertEqual(summary["redundant_window_count"], 0)
        self.assertEqual(summary["top_overlap_pairs"], [])
        # Single window: novelty_ratio is 1.0 (all tokens are new)
        self.assertEqual(summary["windows"][0]["novelty_ratio"], 1.0)

    def test_summarize_engine_scope_coverage_respects_top_pair_count(self):
        rows = [
            {
                "source_family": "sf",
                "task": f"Task{i}",
                "query_offset": i * 100,
                "seed": 1,
                "row_type": "query_profile",
                "profile": "reform",
                "delta_ndcg_at_10": 0.02,
                "query_token_count": 3,
                "query_char_count": 20,
                "query_stopword_ratio": 0.2,
                "query_numeric_token_count": 0,
                "query_negation_count": 0,
                "query_claim_cue_count": 0,
            }
            for i in range(4)
        ]
        summary_limited = summarize_engine_scope_coverage(rows, top_pair_count=2)
        summary_unlimited = summarize_engine_scope_coverage(rows, top_pair_count=100)
        # 4 windows => 6 pairs; limited to 2
        self.assertEqual(len(summary_limited["top_overlap_pairs"]), 2)
        # With top_pair_count=100 we get all 6 pairs
        self.assertEqual(len(summary_unlimited["top_overlap_pairs"]), 6)

    def test_rank_task_offset_candidates_with_rotation_offset(self):
        """Rotation shifts the index used as a tiebreaker so the order changes."""
        candidates = [("SciFact", 0), ("NFCorpus", 0)]
        ranked_no_rotation = rank_task_offset_candidates(candidates, None, rotation_offset=0)
        ranked_with_rotation = rank_task_offset_candidates(candidates, None, rotation_offset=1)
        # Both unseen; rotation should change relative tiebreaker ordering
        no_rot_order = [r["task"] for r in ranked_no_rotation]
        rot_order = [r["task"] for r in ranked_with_rotation]
        self.assertNotEqual(no_rot_order, rot_order)

    def test_rank_task_offset_candidates_empty_candidates(self):
        result = rank_task_offset_candidates([], None)
        self.assertEqual(result, [])

    def test_select_task_offset_candidates_returns_correct_count(self):
        candidates = [("SciFact", 0), ("NFCorpus", 0), ("FiQA2018", 0)]
        selected = select_task_offset_candidates(candidates, None, count=2)
        self.assertEqual(len(selected), 2)

    def test_select_task_offset_candidates_count_zero_returns_empty(self):
        selected = select_task_offset_candidates([("SciFact", 0)], None, count=0)
        self.assertEqual(selected, [])

    def test_select_task_offset_candidates_count_exceeds_candidates(self):
        candidates = [("SciFact", 0)]
        selected = select_task_offset_candidates(candidates, None, count=10)
        self.assertEqual(len(selected), 1)

    def test_select_task_offset_candidates_prefers_unseen_task(self):
        coverage_summary = {
            "task_offset_summary": [
                {
                    "task": "SciFact",
                    "query_offset": 0,
                    "window_count": 3,
                    "row_count": 100,
                    "token_count": 30,
                    "source_families": ["sf"],
                    "max_overlap_with_previous": 0.8,
                    "max_novelty_ratio": 0.1,
                },
            ]
        }
        candidates = [("SciFact", 0), ("NFCorpus", 0)]
        selected = select_task_offset_candidates(candidates, coverage_summary, count=1)
        # NFCorpus is unseen — should be preferred
        self.assertEqual(selected[0]["task"], "NFCorpus")

    def test_summarize_engine_scope_coverage_task_offset_summary_structure(self):
        rows = [
            {
                "source_family": "sf",
                "task": "SciFact",
                "query_offset": 0,
                "seed": 1,
                "row_type": "query_profile",
                "profile": "reform",
                "delta_ndcg_at_10": 0.02,
                "query_token_count": 3,
                "query_char_count": 20,
                "query_stopword_ratio": 0.2,
                "query_numeric_token_count": 0,
                "query_negation_count": 0,
                "query_claim_cue_count": 0,
            },
            {
                "source_family": "sf2",
                "task": "SciFact",
                "query_offset": 100,
                "seed": 1,
                "row_type": "mask_probe",
                "delta_ndcg_at_10": -0.01,
                "query_token_count": 5,
                "query_char_count": 40,
                "query_stopword_ratio": 0.3,
                "query_numeric_token_count": 0,
                "query_negation_count": 0,
                "query_claim_cue_count": 0,
                "baseline_score_margin": 0.1,
                "baseline_top_score": 0.5,
                "query_norm": 1.0,
            },
        ]
        summary = summarize_engine_scope_coverage(rows)
        self.assertEqual(len(summary["task_offset_summary"]), 2)
        for entry in summary["task_offset_summary"]:
            self.assertIn("task", entry)
            self.assertIn("query_offset", entry)
            self.assertIn("window_count", entry)
            self.assertIn("row_count", entry)
            self.assertIn("token_count", entry)
            self.assertIn("source_families", entry)
            self.assertIn("max_overlap_with_previous", entry)
            self.assertIn("max_novelty_ratio", entry)

    def test_window_key_fallback_when_no_context_fields_present(self):
        """_window_key returns 'window=unknown' when all context fields are absent or empty.
        Removing the fallback branch would cause an empty string to be returned instead."""
        result = _window_key({})
        self.assertEqual(result, "window=unknown")
        # Also verify that an all-None/empty-string dict produces the same fallback
        result_empty_values = _window_key({
            "source_family": None,
            "task": "",
            "query_offset": None,
            "seed": None,
            "split": None,
            "loop": None,
            "window": None,
            "global_window": None,
        })
        self.assertEqual(result_empty_values, "window=unknown")

    def test_row_feature_tokens_bool_bucket_no_branches_for_query_profile(self):
        """row_feature_tokens emits topdoc_changed:no and reform_changed:no when
        top_doc_changed=False and reformulation_changed=False.  Deleting the 'no'
        branch of _bool_bucket would emit 'yes' here and break this assertion."""
        tokens = row_feature_tokens({
            "row_type": "query_profile",
            "profile": "reform",
            "delta_ndcg_at_10": -0.01,
            "query_token_count": 4,
            "query_char_count": 30,
            "query_stopword_ratio": 0.2,
            "query_numeric_token_count": 0,
            "query_negation_count": 0,
            "query_claim_cue_count": 0,
            "action": "REFORMULATE",
            "fault_class": "actuator_active_negative",
            "top10_overlap_with_baseline": 5,
            "top_doc_changed": False,
            "global_variance": 0.1,
            "jaccard": 0.4,
            "mask_density": 0.5,
            "reformulation_variant_count": 1,
            "reformulation_changed": False,
        })
        self.assertIn("topdoc_changed:no", tokens)
        self.assertIn("reform_changed:no", tokens)
        self.assertNotIn("topdoc_changed:yes", tokens)
        self.assertNotIn("reform_changed:yes", tokens)


if __name__ == "__main__":
    unittest.main()
