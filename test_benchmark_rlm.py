"""
Tests for pure utility functions in benchmark_rlm.py.

Covers: dcg_at_k, ndcg_at_k, find_keys, find_payload, map_predicted_ids.
Run: python -m pytest test_benchmark_rlm.py -v
"""

import sys
import unittest
from unittest.mock import MagicMock, patch, PropertyMock
import math
import numpy as np

# Mock mteb if not installed (benchmark_rlm imports it at module level)
if 'mteb' not in sys.modules:
    sys.modules['mteb'] = MagicMock()

from benchmark_rlm import dcg_at_k, ndcg_at_k, find_keys, find_payload, map_predicted_ids


# =============================================================================
# dcg_at_k tests
# =============================================================================

class TestDcgAtK(unittest.TestCase):
    """Tests for Discounted Cumulative Gain at rank k."""

    def test_dcg_at_k_known_values(self):
        """DCG([1,0,1], 3) = 1/log2(2) + 0/log2(3) + 1/log2(4) = 1.5."""
        expected = 1.0 / math.log2(2) + 0.0 / math.log2(3) + 1.0 / math.log2(4)
        result = dcg_at_k([1, 0, 1], 3)
        self.assertAlmostEqual(result, expected, places=10)
        self.assertAlmostEqual(result, 1.5, places=10)

    def test_dcg_at_k_single_element(self):
        """k=1 should only consider the first element."""
        result = dcg_at_k([1, 1, 1], 1)
        expected = 1.0 / math.log2(2)  # = 1.0
        self.assertAlmostEqual(result, expected, places=10)

    def test_dcg_at_k_empty_list(self):
        """Empty relevance list should return 0.0."""
        result = dcg_at_k([], 3)
        self.assertEqual(result, 0.0)

    def test_dcg_at_k_all_zeros(self):
        """All-zero relevance should return 0.0."""
        result = dcg_at_k([0, 0, 0], 3)
        self.assertAlmostEqual(result, 0.0, places=10)

    def test_dcg_at_k_k_larger_than_list(self):
        """k larger than list length should handle gracefully (use available items)."""
        result = dcg_at_k([1, 1], 10)
        expected = 1.0 / math.log2(2) + 1.0 / math.log2(3)
        self.assertAlmostEqual(result, expected, places=10)

    def test_dcg_at_k_decreasing_discount(self):
        """Relevance in position 1 is worth more than in position 3."""
        dcg_first = dcg_at_k([1, 0, 0], 3)
        dcg_last = dcg_at_k([0, 0, 1], 3)
        self.assertGreater(dcg_first, dcg_last)

    def test_dcg_at_k_graded_relevance(self):
        """DCG should handle graded (non-binary) relevance values."""
        result = dcg_at_k([3, 2, 1], 3)
        expected = 3.0 / math.log2(2) + 2.0 / math.log2(3) + 1.0 / math.log2(4)
        self.assertAlmostEqual(result, expected, places=10)


# =============================================================================
# ndcg_at_k tests
# =============================================================================

class TestNdcgAtK(unittest.TestCase):
    """Tests for Normalized Discounted Cumulative Gain at rank k."""

    def test_ndcg_at_k_perfect_ranking(self):
        """Perfect ranking [1,1,1] should give NDCG of 1.0."""
        result = ndcg_at_k([1, 1, 1], 3)
        self.assertAlmostEqual(result, 1.0, places=10)

    def test_ndcg_at_k_all_zeros(self):
        """All-zero relevance should return 0.0 (no relevant docs)."""
        result = ndcg_at_k([0, 0, 0], 3)
        self.assertEqual(result, 0.0)

    def test_ndcg_at_k_reversed_ranking(self):
        """Worst possible ranking of a single relevant doc should be < 1.0."""
        result = ndcg_at_k([0, 0, 1], 3)
        self.assertGreater(result, 0.0)
        self.assertLess(result, 1.0)

    def test_ndcg_at_k_always_between_zero_and_one(self):
        """NDCG should always be in [0, 1] for binary relevance."""
        test_cases = [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0],
            [1, 0, 1, 0, 1],
        ]
        for r in test_cases:
            score = ndcg_at_k(r, 5)
            self.assertGreaterEqual(score, 0.0, f"NDCG < 0 for r={r}")
            self.assertLessEqual(score, 1.0, f"NDCG > 1 for r={r}")

    def test_ndcg_at_k_single_relevant_at_top_is_one(self):
        """Single relevant doc at rank 1 out of 1 relevant total -> NDCG = 1.0."""
        result = ndcg_at_k([1, 0, 0], 3)
        self.assertAlmostEqual(result, 1.0, places=10)

    def test_ndcg_at_k_k_truncates(self):
        """NDCG should only consider first k elements."""
        # With k=2, the third element (1) should be ignored
        result_k2 = ndcg_at_k([0, 0, 1], 2)
        self.assertEqual(result_k2, 0.0)

    def test_ndcg_at_k_graded_relevance_normalization(self):
        """Graded relevance: ideal ranking [3,2,1] should give 1.0."""
        result = ndcg_at_k([3, 2, 1], 3)
        self.assertAlmostEqual(result, 1.0, places=10)

    def test_ndcg_at_k_graded_suboptimal(self):
        """Graded relevance in non-ideal order should be < 1.0."""
        result = ndcg_at_k([1, 2, 3], 3)
        self.assertLess(result, 1.0)
        self.assertGreater(result, 0.0)


# =============================================================================
# find_keys tests
# =============================================================================

class TestFindKeys(unittest.TestCase):
    """Tests for recursive dict search by set of keys."""

    def test_find_keys_at_top_level(self):
        """Keys found at top level should return the top-level dict."""
        obj = {"a": 1, "b": 2, "c": 3}
        result = find_keys(obj, ["a", "b"])
        self.assertEqual(result, obj)

    def test_find_keys_nested(self):
        """Keys found in a nested dict should return the inner dict."""
        inner = {"a": 1, "b": 2}
        obj = {"x": inner}
        result = find_keys(obj, ["a", "b"])
        self.assertEqual(result, inner)

    def test_find_keys_deeply_nested(self):
        """Keys found several levels deep should still be found."""
        target = {"corpus": [1, 2], "queries": [3, 4]}
        obj = {"level1": {"level2": {"level3": target}}}
        result = find_keys(obj, ["corpus", "queries"])
        self.assertEqual(result, target)

    def test_find_keys_not_found(self):
        """Missing keys should return None."""
        obj = {"a": 1}
        result = find_keys(obj, ["a", "b"])
        self.assertIsNone(result)

    def test_find_keys_non_dict_input(self):
        """Non-dict input should return None."""
        self.assertIsNone(find_keys("not a dict", ["a"]))
        self.assertIsNone(find_keys(42, ["a"]))
        self.assertIsNone(find_keys([1, 2], ["a"]))
        self.assertIsNone(find_keys(None, ["a"]))

    def test_find_keys_empty_target_keys(self):
        """Empty target_keys means all() returns True on any dict -> returns obj."""
        obj = {"a": 1}
        result = find_keys(obj, [])
        self.assertEqual(result, obj)

    def test_find_keys_prefers_shallowest_match(self):
        """When keys exist at top level, should return top level (not nested)."""
        obj = {"a": 1, "b": {"a": 10, "b": 20}}
        result = find_keys(obj, ["a", "b"])
        self.assertEqual(result, obj)


# =============================================================================
# find_payload tests
# =============================================================================

class TestFindPayload(unittest.TestCase):
    """Tests for recursive single-key search."""

    def test_find_payload_at_top_level(self):
        """Key at top level should be found."""
        obj = {"name": "alice", "age": 30}
        result = find_payload(obj, "name")
        self.assertEqual(result, "alice")

    def test_find_payload_nested(self):
        """Key in a nested dict should be found."""
        obj = {"outer": {"inner": {"target": 42}}}
        result = find_payload(obj, "target")
        self.assertEqual(result, 42)

    def test_find_payload_not_found(self):
        """Missing key should return None."""
        obj = {"a": 1, "b": {"c": 2}}
        result = find_payload(obj, "missing")
        self.assertIsNone(result)

    def test_find_payload_non_dict_input(self):
        """Non-dict input should return None."""
        self.assertIsNone(find_payload("string", "key"))
        self.assertIsNone(find_payload(123, "key"))
        self.assertIsNone(find_payload(None, "key"))

    def test_find_payload_falsy_value_zero(self):
        """BUG: falsy values like 0 are treated as not-found due to `if res:` guard."""
        obj = {"count": 0}
        result = find_payload(obj, "count")
        # The top-level check `if key in obj: return obj[key]` works fine for
        # top-level falsy values. The bug only manifests when the falsy value
        # is found by recursive descent.
        self.assertEqual(result, 0)

    def test_find_payload_falsy_value_nested_zero(self):
        """BUG: nested falsy value 0 is missed because `if res:` evaluates to False."""
        obj = {"outer": {"count": 0}}
        result = find_payload(obj, "count")
        # This documents the known bug: the function returns None instead of 0
        # because `if res:` treats 0 as falsy and continues searching.
        # When the bug is fixed, change this assertion to assertEqual(result, 0).
        self.assertIsNone(result)

    def test_find_payload_falsy_value_nested_empty_string(self):
        """BUG: nested empty string is missed due to `if res:` guard."""
        obj = {"wrapper": {"text": ""}}
        result = find_payload(obj, "text")
        # Same bug as above: `if res:` treats "" as falsy.
        self.assertIsNone(result)

    def test_find_payload_falsy_value_nested_false(self):
        """BUG: nested boolean False is missed due to `if res:` guard."""
        obj = {"wrapper": {"enabled": False}}
        result = find_payload(obj, "enabled")
        # Same bug: `if res:` treats False as falsy.
        self.assertIsNone(result)

    def test_find_payload_truthy_nested_value(self):
        """Truthy nested values should be found correctly."""
        obj = {"wrapper": {"data": [1, 2, 3]}}
        result = find_payload(obj, "data")
        self.assertEqual(result, [1, 2, 3])


# =============================================================================
# map_predicted_ids tests
# =============================================================================

class TestMapPredictedIds(unittest.TestCase):
    """Tests for Qdrant ID mapping."""

    def _make_point(self, point_id, original_id=None):
        """Create a mock Qdrant point with optional original_id in payload."""
        point = MagicMock()
        point.id = point_id
        if original_id is not None:
            point.payload = {"original_id": original_id}
        else:
            point.payload = {}
        return point

    def test_map_predicted_ids_with_original_ids(self):
        """Points with original_id in payload should be mapped."""
        engine = MagicMock()
        engine.qdrant.retrieve.return_value = [
            self._make_point(1, "doc_alpha"),
            self._make_point(2, "doc_beta"),
            self._make_point(3, "doc_gamma"),
        ]

        result = map_predicted_ids(engine, [1, 2, 3])

        self.assertEqual(result, ["doc_alpha", "doc_beta", "doc_gamma"])
        engine.qdrant.retrieve.assert_called_once_with(
            engine.collection_name, ids=[1, 2, 3]
        )

    def test_map_predicted_ids_without_original_ids(self):
        """Points without original_id should fallback to stringified raw ID."""
        engine = MagicMock()
        engine.qdrant.retrieve.return_value = [
            self._make_point(10),
            self._make_point(20),
        ]

        result = map_predicted_ids(engine, [10, 20])

        self.assertEqual(result, ["10", "20"])

    def test_map_predicted_ids_mixed_payloads(self):
        """Mix of points with and without original_id."""
        engine = MagicMock()
        engine.qdrant.retrieve.return_value = [
            self._make_point(1, "doc_a"),
            self._make_point(2),
            self._make_point(3, "doc_c"),
        ]

        result = map_predicted_ids(engine, [1, 2, 3])

        self.assertEqual(result, ["doc_a", "2", "doc_c"])

    def test_map_predicted_ids_qdrant_exception_fallback(self):
        """Qdrant exception should return stringified raw IDs as fallback."""
        engine = MagicMock()
        engine.qdrant.retrieve.side_effect = Exception("Connection refused")

        result = map_predicted_ids(engine, [100, 200, 300])

        self.assertEqual(result, ["100", "200", "300"])

    def test_map_predicted_ids_empty_list(self):
        """Empty pred_ids should return empty list."""
        engine = MagicMock()
        engine.qdrant.retrieve.return_value = []

        result = map_predicted_ids(engine, [])

        self.assertEqual(result, [])

    def test_map_predicted_ids_preserves_order(self):
        """Returned IDs should match the order of pred_ids, not retrieval order."""
        engine = MagicMock()
        # Points returned in different order than requested
        engine.qdrant.retrieve.return_value = [
            self._make_point(3, "doc_c"),
            self._make_point(1, "doc_a"),
            self._make_point(2, "doc_b"),
        ]

        result = map_predicted_ids(engine, [1, 2, 3])

        self.assertEqual(result, ["doc_a", "doc_b", "doc_c"])

    def test_map_predicted_ids_null_payload(self):
        """Point with None payload should fallback to stringified raw ID."""
        engine = MagicMock()
        point = MagicMock()
        point.id = 5
        point.payload = None
        engine.qdrant.retrieve.return_value = [point]

        result = map_predicted_ids(engine, [5])

        self.assertEqual(result, ["5"])

    def test_map_predicted_ids_missing_point_in_response(self):
        """Pred ID not in Qdrant response should fallback to stringified raw ID."""
        engine = MagicMock()
        # Only point 1 returned; point 2 is missing from response
        engine.qdrant.retrieve.return_value = [
            self._make_point(1, "doc_a"),
        ]

        result = map_predicted_ids(engine, [1, 2])

        self.assertEqual(result, ["doc_a", "2"])


if __name__ == "__main__":
    unittest.main()
