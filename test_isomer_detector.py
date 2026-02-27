"""
Tests for IsomerDetector (Retrieval Isomer Detection)

Run: python -m unittest test_isomer_detector -v
"""

import unittest
from unittest.mock import patch

from isomer_detector import IsomerDetector
from config import ChelationConfig


@patch('isomer_detector.get_logger')
class TestIsomerDetector(unittest.TestCase):
    """Tests for the IsomerDetector class."""

    def test_initialization_defaults(self, mock_logger):
        """Test default initialization."""
        detector = IsomerDetector()
        self.assertAlmostEqual(detector.strength_threshold, 0.3)
        self.assertEqual(detector.top_k, 10)

    def test_initialization_custom(self, mock_logger):
        """Test custom initialization."""
        detector = IsomerDetector(strength_threshold=0.5, top_k=5)
        self.assertAlmostEqual(detector.strength_threshold, 0.5)
        self.assertEqual(detector.top_k, 5)

    def test_initialization_invalid_threshold(self, mock_logger):
        """Test that invalid threshold raises ValueError."""
        with self.assertRaises(ValueError):
            IsomerDetector(strength_threshold=1.5)
        with self.assertRaises(ValueError):
            IsomerDetector(strength_threshold=-0.1)

    def test_compute_jaccard_identical(self, mock_logger):
        """Test Jaccard=1.0 for identical sets."""
        jaccard = IsomerDetector.compute_jaccard([1, 2, 3], [1, 2, 3])
        self.assertAlmostEqual(jaccard, 1.0)

    def test_compute_jaccard_disjoint(self, mock_logger):
        """Test Jaccard=0.0 for disjoint sets."""
        jaccard = IsomerDetector.compute_jaccard([1, 2, 3], [4, 5, 6])
        self.assertAlmostEqual(jaccard, 0.0)

    def test_compute_jaccard_partial_overlap(self, mock_logger):
        """Test Jaccard for partial overlap."""
        jaccard = IsomerDetector.compute_jaccard([1, 2, 3], [2, 3, 4])
        # intersection = {2,3} = 2, union = {1,2,3,4} = 4
        self.assertAlmostEqual(jaccard, 0.5)

    def test_compute_jaccard_empty_sets(self, mock_logger):
        """Test Jaccard=1.0 for both empty sets."""
        jaccard = IsomerDetector.compute_jaccard([], [])
        self.assertAlmostEqual(jaccard, 1.0)

    def test_compute_isomer_strength(self, mock_logger):
        """Test isomer strength = 1.0 - jaccard."""
        strength = IsomerDetector.compute_isomer_strength([1, 2, 3], [4, 5, 6])
        self.assertAlmostEqual(strength, 1.0)

        strength = IsomerDetector.compute_isomer_strength([1, 2, 3], [1, 2, 3])
        self.assertAlmostEqual(strength, 0.0)

    def test_detect_sedimentation_isomers_all_identical(self, mock_logger):
        """Test no isomers when pre/post results are identical."""
        detector = IsomerDetector()
        pre = {"q1": [1, 2, 3], "q2": [4, 5, 6]}
        post = {"q1": [1, 2, 3], "q2": [4, 5, 6]}
        result = detector.detect_sedimentation_isomers(pre, post)
        self.assertEqual(result["isomer_count"], 0)
        self.assertEqual(result["total_queries"], 2)
        self.assertAlmostEqual(result["mean_strength"], 0.0)

    def test_detect_sedimentation_isomers_all_different(self, mock_logger):
        """Test all isomers when pre/post results are completely different."""
        detector = IsomerDetector(strength_threshold=0.3)
        pre = {"q1": [1, 2, 3], "q2": [4, 5, 6]}
        post = {"q1": [7, 8, 9], "q2": [10, 11, 12]}
        result = detector.detect_sedimentation_isomers(pre, post)
        self.assertEqual(result["isomer_count"], 2)
        self.assertAlmostEqual(result["isomer_ratio"], 1.0)
        self.assertAlmostEqual(result["mean_strength"], 1.0)

    def test_detect_sedimentation_isomers_partial(self, mock_logger):
        """Test mixed isomer detection."""
        detector = IsomerDetector(strength_threshold=0.3)
        pre = {"q1": [1, 2, 3, 4, 5], "q2": [1, 2, 3, 4, 5]}
        post = {"q1": [1, 2, 3, 4, 5], "q2": [6, 7, 8, 9, 10]}
        result = detector.detect_sedimentation_isomers(pre, post)
        self.assertEqual(result["isomer_count"], 1)
        self.assertEqual(len(result["non_isomers"]), 1)

    def test_detect_chelation_isomers(self, mock_logger):
        """Test chelation isomer detection."""
        detector = IsomerDetector()
        std = {"q1": [1, 2, 3]}
        chel = {"q1": [4, 5, 6]}
        result = detector.detect_chelation_isomers(std, chel)
        self.assertEqual(result["isomer_count"], 1)
        self.assertEqual(result["mode"], "chelation")

    def test_detect_isomers_only_common_queries(self, mock_logger):
        """Test that only common queries are compared."""
        detector = IsomerDetector()
        pre = {"q1": [1, 2, 3], "q_only_pre": [4, 5, 6]}
        post = {"q1": [1, 2, 3], "q_only_post": [7, 8, 9]}
        result = detector.detect_sedimentation_isomers(pre, post)
        self.assertEqual(result["total_queries"], 1)

    def test_detect_isomers_top_k_truncation(self, mock_logger):
        """Test that results are truncated to top_k."""
        detector = IsomerDetector(top_k=3)
        pre = {"q1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
        post = {"q1": [1, 2, 3, 11, 12, 13, 14, 15, 16, 17]}
        result = detector.detect_sedimentation_isomers(pre, post)
        # Only top 3 are compared: [1,2,3] vs [1,2,3] -> identical -> no isomer
        self.assertEqual(result["isomer_count"], 0)

    def test_detect_isomers_records_history(self, mock_logger):
        """Test that detection records to history."""
        detector = IsomerDetector()
        pre = {"q1": [1, 2, 3]}
        post = {"q1": [4, 5, 6]}
        detector.detect_sedimentation_isomers(pre, post)
        detector.detect_chelation_isomers(pre, post)
        report = detector.get_isomer_report()
        self.assertEqual(report["total_detections"], 2)

    def test_find_similar_query_isomers_no_overlap(self, mock_logger):
        """Test similar query finding with no overlapping results."""
        detector = IsomerDetector()
        queries = ["q1", "q2"]
        results_map = {"q1": [1, 2, 3], "q2": [4, 5, 6]}
        result = detector.find_similar_query_isomers(queries, results_map, similarity_threshold=0.5)
        self.assertEqual(result["pair_count"], 0)

    def test_find_similar_query_isomers_with_overlap(self, mock_logger):
        """Test similar query finding with overlapping results."""
        detector = IsomerDetector()
        queries = ["q1", "q2"]
        results_map = {"q1": [1, 2, 3, 4, 5], "q2": [1, 2, 3, 4, 6]}
        # Jaccard = 4/6 = 0.667 > 0.5
        result = detector.find_similar_query_isomers(queries, results_map, similarity_threshold=0.5)
        self.assertEqual(result["pair_count"], 1)
        self.assertAlmostEqual(result["similar_pairs"][0]["jaccard"], 4.0 / 6.0, places=5)

    def test_find_similar_query_isomers_missing_query(self, mock_logger):
        """Test that queries not in results_map are skipped."""
        detector = IsomerDetector()
        queries = ["q1", "q2", "q_missing"]
        results_map = {"q1": [1, 2, 3], "q2": [1, 2, 3]}
        result = detector.find_similar_query_isomers(queries, results_map, similarity_threshold=0.5)
        # Only q1 vs q2 compared
        self.assertEqual(result["pair_count"], 1)

    def test_get_strength_distribution_empty(self, mock_logger):
        """Test distribution with empty input."""
        detector = IsomerDetector()
        dist = detector.get_strength_distribution([])
        self.assertEqual(dist["count"], 0)
        self.assertAlmostEqual(dist["mean"], 0.0)

    def test_get_strength_distribution_values(self, mock_logger):
        """Test distribution statistics."""
        detector = IsomerDetector()
        strengths = [0.1, 0.3, 0.5, 0.7, 0.9]
        dist = detector.get_strength_distribution(strengths)
        self.assertEqual(dist["count"], 5)
        self.assertAlmostEqual(dist["mean"], 0.5)
        self.assertAlmostEqual(dist["min"], 0.1)
        self.assertAlmostEqual(dist["max"], 0.9)
        self.assertAlmostEqual(dist["median"], 0.5)

    def test_get_isomer_report_empty(self, mock_logger):
        """Test isomer report with no history."""
        detector = IsomerDetector()
        report = detector.get_isomer_report()
        self.assertEqual(report["total_detections"], 0)

    def test_get_isomer_report_with_history(self, mock_logger):
        """Test isomer report after detections."""
        detector = IsomerDetector()
        pre = {"q1": [1, 2, 3]}
        post = {"q1": [4, 5, 6]}
        detector.detect_sedimentation_isomers(pre, post)
        report = detector.get_isomer_report()
        self.assertEqual(report["total_detections"], 1)
        self.assertGreater(report["cumulative_mean_strength"], 0.0)
        self.assertEqual(len(report["history"]), 1)

    def test_reset(self, mock_logger):
        """Test that reset clears history."""
        detector = IsomerDetector()
        pre = {"q1": [1, 2]}
        post = {"q1": [3, 4]}
        detector.detect_sedimentation_isomers(pre, post)
        detector.reset()
        report = detector.get_isomer_report()
        self.assertEqual(report["total_detections"], 0)

    def test_isomer_entry_has_jaccard(self, mock_logger):
        """Test that each isomer entry includes jaccard value."""
        detector = IsomerDetector(strength_threshold=0.0)
        pre = {"q1": [1, 2, 3]}
        post = {"q1": [2, 3, 4]}
        result = detector.detect_sedimentation_isomers(pre, post)
        entry = result["isomers"][0]
        self.assertIn("jaccard", entry)
        self.assertIn("strength", entry)
        self.assertAlmostEqual(entry["jaccard"] + entry["strength"], 1.0)

    def test_max_strength(self, mock_logger):
        """Test max strength tracking across queries."""
        detector = IsomerDetector(strength_threshold=0.0)
        pre = {"q1": [1, 2, 3], "q2": [1, 2, 3]}
        post = {"q1": [1, 2, 4], "q2": [4, 5, 6]}
        result = detector.detect_sedimentation_isomers(pre, post)
        self.assertAlmostEqual(result["max_strength"], 1.0)


@patch('isomer_detector.get_logger')
class TestIsomerPresets(unittest.TestCase):
    """Tests for isomer config presets."""

    def test_isomer_balanced_preset(self, mock_logger):
        """Test balanced isomer preset."""
        preset = ChelationConfig.get_preset("balanced", "isomer")
        self.assertAlmostEqual(preset["strength_threshold"], 0.3)
        self.assertEqual(preset["top_k"], 10)
        self.assertIn("description", preset)

    def test_isomer_sensitive_preset(self, mock_logger):
        """Test sensitive isomer preset."""
        preset = ChelationConfig.get_preset("sensitive", "isomer")
        self.assertAlmostEqual(preset["strength_threshold"], 0.1)
        self.assertLess(preset["strength_threshold"], 0.3)

    def test_isomer_strict_preset(self, mock_logger):
        """Test strict isomer preset."""
        preset = ChelationConfig.get_preset("strict", "isomer")
        self.assertAlmostEqual(preset["strength_threshold"], 0.5)
        self.assertGreater(preset["strength_threshold"], 0.3)

    def test_isomer_invalid_preset(self, mock_logger):
        """Test invalid isomer preset raises ValueError."""
        with self.assertRaises(ValueError):
            ChelationConfig.get_preset("nonexistent", "isomer")


if __name__ == "__main__":
    unittest.main(verbosity=2)
