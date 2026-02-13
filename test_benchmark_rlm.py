"""
Unit Tests for benchmark_rlm.py Pure Functions

Tests DCG/NDCG metric calculations, dict search helpers (find_keys,
find_payload), and the map_predicted_ids ID mapping function.
No external services (Qdrant, Ollama, MTEB) required.
"""

import unittest
from unittest.mock import MagicMock

from benchmark_rlm import dcg_at_k, ndcg_at_k, find_keys, find_payload, map_predicted_ids


# ===========================================================================
# DCG@k Tests
# ===========================================================================


class TestDcgAtK(unittest.TestCase):
    """Tests for the dcg_at_k (Discounted Cumulative Gain) function."""

    def test_empty_list(self):
        """DCG of an empty relevance list is 0.0."""
        self.assertEqual(dcg_at_k([], 5), 0.0)

    def test_k_zero(self):
        """DCG at k=0 is 0.0 regardless of input."""
        self.assertEqual(dcg_at_k([1, 1, 1], 0), 0.0)

    def test_single_relevant(self):
        """DCG of a single relevant doc at rank 1 is 1.0."""
        self.assertAlmostEqual(dcg_at_k([1], 1), 1.0, places=3)

    def test_perfect_at_five(self):
        """DCG of 5 relevant docs at k=5."""
        self.assertAlmostEqual(dcg_at_k([1, 1, 1, 1, 1], 5), 2.9485, places=3)

    def test_mixed_binary(self):
        """DCG with mixed binary relevance [1,0,1,0,0] at k=5."""
        self.assertAlmostEqual(dcg_at_k([1, 0, 1, 0, 0], 5), 1.5, places=3)

    def test_graded_relevance(self):
        """DCG with graded relevance [3,2,1] at k=3."""
        self.assertAlmostEqual(dcg_at_k([3, 2, 1], 3), 4.7619, places=3)

    def test_k_truncates(self):
        """DCG truncates to k=2 even with 5 relevant docs."""
        self.assertAlmostEqual(dcg_at_k([1, 1, 1, 1, 1], 2), 1.6309, places=3)

    def test_k_larger_than_list(self):
        """DCG with k larger than list uses all available items."""
        self.assertAlmostEqual(dcg_at_k([1, 1], 10), 1.6309, places=3)


# ===========================================================================
# NDCG@k Tests
# ===========================================================================


class TestNdcgAtK(unittest.TestCase):
    """Tests for the ndcg_at_k (Normalized DCG) function."""

    def test_all_zeros(self):
        """NDCG with no relevant docs is 0.0."""
        self.assertEqual(ndcg_at_k([0, 0, 0], 3), 0.0)

    def test_perfect_ranking(self):
        """NDCG of a perfect binary ranking is 1.0."""
        self.assertAlmostEqual(ndcg_at_k([1, 1, 1], 3), 1.0, places=3)

    def test_worst_single(self):
        """NDCG with single relevant doc at worst position."""
        self.assertAlmostEqual(ndcg_at_k([0, 0, 1], 3), 0.5, places=3)

    def test_mixed(self):
        """NDCG with mixed relevance [1,0,1] at k=3."""
        self.assertAlmostEqual(ndcg_at_k([1, 0, 1], 3), 0.9197, places=3)

    def test_graded_reversed(self):
        """NDCG with graded relevance in worst order [1,2,3]."""
        self.assertAlmostEqual(ndcg_at_k([1, 2, 3], 3), 0.7901, places=3)


# ===========================================================================
# find_keys Tests
# ===========================================================================


class TestFindKeys(unittest.TestCase):
    """Tests for the find_keys recursive dict search function."""

    def test_non_dict(self):
        """Non-dict input returns None."""
        self.assertIsNone(find_keys("string", ["a"]))

    def test_top_level(self):
        """Returns the dict itself when all target keys exist at top level."""
        result = find_keys({"a": 1, "b": 2}, ["a", "b"])
        self.assertEqual(result, {"a": 1, "b": 2})

    def test_nested(self):
        """Finds target keys in a nested dict."""
        result = find_keys({"x": {"a": 1, "b": 2}}, ["a", "b"])
        self.assertEqual(result, {"a": 1, "b": 2})

    def test_missing(self):
        """Returns None when not all target keys are present."""
        self.assertIsNone(find_keys({"a": 1}, ["a", "b"]))

    def test_deeply_nested(self):
        """Finds target keys in a deeply nested dict."""
        result = find_keys({"x": {"y": {"a": 1, "b": 2}}}, ["a", "b"])
        self.assertEqual(result, {"a": 1, "b": 2})


# ===========================================================================
# find_payload Tests
# ===========================================================================


class TestFindPayload(unittest.TestCase):
    """Tests for the find_payload recursive key search function."""

    def test_top_level(self):
        """Finds a key at the top level and returns its value."""
        self.assertEqual(find_payload({"a": 42}, "a"), 42)

    def test_nested(self):
        """Finds a key nested inside another dict."""
        self.assertEqual(find_payload({"x": {"a": 42}}, "a"), 42)

    def test_missing(self):
        """Returns None for a key that does not exist."""
        self.assertIsNone(find_payload({"a": 1}, "b"))

    def test_non_dict(self):
        """Returns None for non-dict input."""
        self.assertIsNone(find_payload(None, "a"))

    def test_returns_dict_value(self):
        """Returns a dict value when the found key maps to a dict."""
        self.assertEqual(find_payload({"a": {"nested": "val"}}, "a"), {"nested": "val"})


# ===========================================================================
# map_predicted_ids Tests
# ===========================================================================


class TestMapPredictedIds(unittest.TestCase):
    """Tests for the map_predicted_ids Qdrant ID mapping function."""

    def _make_mock_engine(self):
        """Create a mock AntigravityEngine with a qdrant client."""
        engine = MagicMock()
        engine.collection_name = "test"
        return engine

    def _make_mock_point(self, point_id, payload=None):
        """Create a mock Qdrant point with .id and .payload attributes."""
        point = MagicMock()
        point.id = point_id
        point.payload = payload
        return point

    def test_with_original_id(self):
        """Points with original_id in payload are mapped to original strings."""
        engine = self._make_mock_engine()
        points = [
            self._make_mock_point(1, {"original_id": "doc_a"}),
            self._make_mock_point(2, {"original_id": "doc_b"}),
        ]
        engine.qdrant.retrieve.return_value = points

        result = map_predicted_ids(engine, [1, 2])
        self.assertEqual(result, ["doc_a", "doc_b"])

    def test_without_original_id(self):
        """Points without original_id fall back to str(id)."""
        engine = self._make_mock_engine()
        points = [
            self._make_mock_point(10, {"text": "some text"}),
            self._make_mock_point(20, {"text": "other text"}),
        ]
        engine.qdrant.retrieve.return_value = points

        result = map_predicted_ids(engine, [10, 20])
        self.assertEqual(result, ["10", "20"])

    def test_qdrant_exception(self):
        """When qdrant.retrieve raises Exception, returns str(pid) fallback."""
        engine = self._make_mock_engine()
        engine.qdrant.retrieve.side_effect = Exception("Connection refused")

        result = map_predicted_ids(engine, [1, 2, 3])
        self.assertEqual(result, ["1", "2", "3"])

    def test_empty_list(self):
        """Empty pred_ids input returns empty list."""
        engine = self._make_mock_engine()
        engine.qdrant.retrieve.return_value = []

        result = map_predicted_ids(engine, [])
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
