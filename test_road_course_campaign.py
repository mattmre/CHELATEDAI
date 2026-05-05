"""Tests for the small-model road-course campaign harness."""

import unittest

from config import ChelationConfig
from run_road_course_campaign import (
    ATTNRES_COMPARISON_GRID,
    ATTNRES_NUM_BLOCKS_GRID,
    DEFAULT_PROFILE_GRID,
    PROFILE_SETS,
    RoadCourseProfile,
    _temporary_adapter_config,
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
        self.assertEqual(profile.adapter_type, "mlp")

    def test_attnres_comparison_grid_keeps_baseline_and_adapter_controls(self):
        profiles = {profile.name: profile for profile in ATTNRES_COMPARISON_GRID}

        self.assertIn("baseline", profiles)
        self.assertEqual(profiles["baseline"].adapter_type, "mlp")
        self.assertEqual(profiles["attnres_baseline"].adapter_type, "attnres")
        self.assertEqual(profiles["attnres_balanced_p85_t0.01"].adapter_type, "attnres")
        self.assertEqual(profiles["attnres_balanced_p85_t0.01"].attnres_num_blocks, 4)
        self.assertEqual(profiles["attnres_balanced_p85_t0.01"].chelation_threshold, 0.01)
        self.assertIn("attnres_comparison", PROFILE_SETS)

    def test_attnres_num_blocks_grid_has_shallow_balanced_deep_profiles(self):
        profiles = {profile.name: profile for profile in ATTNRES_NUM_BLOCKS_GRID}

        self.assertIn("baseline", profiles)
        self.assertEqual(profiles["attnres_shallow"].attnres_num_blocks, 2)
        self.assertEqual(profiles["attnres_balanced"].attnres_num_blocks, 4)
        self.assertEqual(profiles["attnres_deep"].attnres_num_blocks, 8)
        self.assertIn("attnres_num_blocks", PROFILE_SETS)

    def test_temporary_adapter_config_restores_global_config(self):
        original_adapter_type = ChelationConfig.ADAPTER_TYPE
        original_num_blocks = ChelationConfig.ATTNRES_ADAPTER_NUM_BLOCKS
        original_proj_dim = ChelationConfig.ATTNRES_ADAPTER_PROJ_DIM

        with _temporary_adapter_config("attnres", 8, 32):
            self.assertEqual(ChelationConfig.ADAPTER_TYPE, "attnres")
            self.assertEqual(ChelationConfig.ATTNRES_ADAPTER_NUM_BLOCKS, 8)
            self.assertEqual(ChelationConfig.ATTNRES_ADAPTER_PROJ_DIM, 32)

        self.assertEqual(ChelationConfig.ADAPTER_TYPE, original_adapter_type)
        self.assertEqual(ChelationConfig.ATTNRES_ADAPTER_NUM_BLOCKS, original_num_blocks)
        self.assertEqual(ChelationConfig.ATTNRES_ADAPTER_PROJ_DIM, original_proj_dim)


if __name__ == "__main__":
    unittest.main(verbosity=2)
