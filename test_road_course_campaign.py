"""Tests for the small-model road-course campaign harness."""

import unittest

from config import ChelationConfig
from run_road_course_campaign import (
    DEFAULT_PROFILE_GRID,
    RoadCourseProfile,
    evaluate_rankings,
    select_road_course_slice,
)


class TestRoadCourseCampaignHarness(unittest.TestCase):
    def test_default_profile_grid_includes_road_course_guardrail(self):
        profiles = {profile.name: profile for profile in DEFAULT_PROFILE_GRID}

        self.assertEqual(ChelationConfig.DEFAULT_CHELATION_THRESHOLD, 0.01)
        self.assertEqual(profiles["adaptive_p85_t0.01"].chelation_threshold, 0.01)
        self.assertEqual(profiles["adaptive_p85_t0.0004"].chelation_threshold, 0.0004)
        self.assertFalse(profiles["baseline"].use_quantization)

    def test_slice_preserves_relevant_documents_and_caps_corpus(self):
        corpus = {f"d{i}": f"doc {i}" for i in range(10)}
        queries = {"q1": "alpha", "q2": "beta", "q3": "gamma"}
        qrels = {"q1": {"d7": 1}, "q2": {"d8": 1}, "q3": {"d9": 1}}

        sliced_corpus, sliced_queries, sliced_qrels = select_road_course_slice(
            corpus,
            queries,
            qrels,
            max_queries=2,
            sample_docs=3,
            seed=5,
        )

        self.assertEqual(set(sliced_queries), {"q1", "q2"})
        self.assertEqual(set(sliced_qrels), {"q1", "q2"})
        self.assertIn("d7", sliced_corpus)
        self.assertIn("d8", sliced_corpus)
        self.assertLessEqual(len(sliced_corpus), 3)

    def test_evaluate_rankings_scores_baseline_and_regression(self):
        qrels = {"q1": {"d1": 1}, "q2": {"d2": 1}}
        good = evaluate_rankings({"q1": ["d1", "d3"], "q2": ["d2", "d4"]}, qrels)
        bad = evaluate_rankings({"q1": ["d3", "d1"], "q2": ["d4", "d2"]}, qrels)

        self.assertGreater(good["ndcg_at_10"], bad["ndcg_at_10"])
        self.assertEqual(good["recall_at_10"], 1.0)
        self.assertEqual(bad["recall_at_10"], 1.0)

    def test_profile_dataclass_defaults_match_safe_baseline(self):
        profile = RoadCourseProfile("custom")

        self.assertFalse(profile.use_centering)
        self.assertFalse(profile.use_quantization)
        self.assertEqual(profile.chelation_threshold, ChelationConfig.DEFAULT_CHELATION_THRESHOLD)


if __name__ == "__main__":
    unittest.main(verbosity=2)
