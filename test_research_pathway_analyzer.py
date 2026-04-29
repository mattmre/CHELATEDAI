import tempfile
import unittest
from pathlib import Path

from learned_mask_policy import run_learned_mask_smoke
from query_reformulator import query_lexical_features, should_apply_reformulation
from research_pathway_analyzer import (
    propose_candidate_profiles,
    run_meta_analysis,
    summarize_benchmark_families,
    summarize_query_attribution,
)


class TestResearchPathwayAnalyzer(unittest.TestCase):
    def test_learned_mask_smoke_recovers_collapse_dimension(self):
        result = run_learned_mask_smoke(topic_count=4, collapse_strength=4.0)

        self.assertEqual(result["learned_mask"]["masked_dims"], [result["expected_collapse_dim"]])
        self.assertTrue(result["recovered"])
        self.assertEqual(result["learned"]["metrics"]["ndcg_at_3"], 1.0)

    def test_selective_reformulation_policy(self):
        self.assertTrue(should_apply_reformulation("heart failure treatment", "selective_low_specificity"))
        self.assertFalse(should_apply_reformulation(
            "increased microtubule acetylation repairs LRRK2 Roc-COR domain mutation deficits",
            "selective_low_specificity",
        ))
        self.assertTrue(should_apply_reformulation(
            "increased microtubule acetylation repairs LRRK2 Roc-COR domain mutation deficits",
            "selective_high_specificity",
        ))
        self.assertTrue(should_apply_reformulation("LDL cholesterol has no involvement", "selective_claim_cue"))
        self.assertFalse(should_apply_reformulation("heart failure treatment", "never"))

    def test_query_lexical_features(self):
        features = query_lexical_features("LDL cholesterol has no involvement in 10% risk")

        self.assertEqual(features["numeric_token_count"], 1)
        self.assertEqual(features["negation_count"], 1)
        self.assertGreaterEqual(features["claim_cue_count"], 1)

    def test_query_attribution_summary_and_candidates(self):
        summary = summarize_query_attribution([
            {
                "profile": "candidate",
                "delta_ndcg_at_10": 0.2,
                "top_doc_changed": True,
                "action": "CHELATE",
            },
            {
                "profile": "candidate",
                "delta_ndcg_at_10": 0.0,
                "top_doc_changed": False,
                "action": "FAST",
            },
            {
                "profile": "risky",
                "delta_ndcg_at_10": -0.4,
                "top_doc_changed": True,
                "action": "CHELATE",
            },
        ])
        proposals = propose_candidate_profiles(summary)

        self.assertEqual(summary["candidate"]["positive_queries"], 1)
        self.assertEqual(summary["candidate"]["negative_queries"], 0)
        self.assertEqual(proposals[0]["profile"], "candidate")
        self.assertEqual(proposals[0]["proposal"], "retest_query_conditional")

    def test_benchmark_family_summary(self):
        summary = summarize_benchmark_families([
            {
                "loops": [
                    {
                        "windows": [
                            {
                                "task": "SciFact",
                                "summary": {
                                    "ranked_profiles": [
                                        {"profile": "baseline", "delta_vs_baseline": 0.0},
                                        {"profile": "candidate", "delta_vs_baseline": 0.1},
                                    ]
                                },
                            }
                        ]
                    }
                ]
            }
        ])

        self.assertEqual(summary["SciFact"]["candidate"]["windows"], 1)
        self.assertEqual(summary["SciFact"]["candidate"]["mean_delta_vs_baseline"], 0.1)

    def test_meta_analysis_reads_artifact(self):
        artifact = {
            "query_attribution_rows": [
                {
                    "profile": "candidate",
                    "delta_ndcg_at_10": 0.1,
                    "top_doc_changed": True,
                    "action": "REFORMULATE",
                }
            ],
            "loops": [],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "artifact.json"
            path.write_text(__import__("json").dumps(artifact), encoding="utf-8")

            result = run_meta_analysis([path])

        self.assertEqual(result["artifact_count"], 1)
        self.assertTrue(result["synthetic_collapse"]["recovered"])
        self.assertTrue(result["learned_mask"]["recovered"])
        self.assertIsNone(result["golden_setting"])


if __name__ == "__main__":
    unittest.main()
