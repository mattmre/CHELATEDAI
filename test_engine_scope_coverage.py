import unittest

from engine_scope_coverage import (
    build_window_footprints,
    rank_task_offset_candidates,
    row_feature_tokens,
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


if __name__ == "__main__":
    unittest.main()
