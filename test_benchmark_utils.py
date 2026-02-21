"""
Tests for benchmark_utils.py shared utility functions.

Covers: dcg_at_k, ndcg_at_k, find_keys, find_payload, load_mteb_data.
Run: python -m pytest test_benchmark_utils.py -v
"""

import sys
import unittest
from unittest.mock import MagicMock, patch
import math

# Mock mteb if not installed
if 'mteb' not in sys.modules:
    sys.modules['mteb'] = MagicMock()

from benchmark_utils import dcg_at_k, ndcg_at_k, find_keys, find_payload, load_mteb_data


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

    def test_find_keys_not_found(self):
        """Missing keys should return None."""
        obj = {"a": 1}
        result = find_keys(obj, ["a", "b"])
        self.assertIsNone(result)


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


# =============================================================================
# load_mteb_data tests
# =============================================================================

class TestLoadMtebData(unittest.TestCase):
    """Tests for MTEB data loading."""

    @patch('benchmark_utils.mteb')
    def test_load_mteb_data_task_not_found(self, mock_mteb):
        """Task not in registry should return (None, None, None)."""
        mock_mteb.get_task.side_effect = KeyError("Task not found")
        
        corpus, queries, qrels = load_mteb_data("NonExistentTask")
        
        self.assertIsNone(corpus)
        self.assertIsNone(queries)
        self.assertIsNone(qrels)

    @patch('benchmark_utils.mteb')
    def test_load_mteb_data_success_dict_format(self, mock_mteb):
        """Successful load with dict-formatted data."""
        mock_task = MagicMock()
        mock_task.dataset = {
            'corpus': {
                'doc1': {'text': 'Hello', 'title': 'World'},
                'doc2': {'text': 'Foo', 'title': 'Bar'},
            },
            'queries': {
                'q1': {'text': 'search query'},
            },
            'relevant_docs': {
                'q1': {'doc1': 1, 'doc2': 0},
            }
        }
        mock_mteb.get_task.return_value = mock_task
        
        corpus, queries, qrels = load_mteb_data("TestTask")
        
        self.assertIsNotNone(corpus)
        self.assertIsNotNone(queries)
        self.assertIsNotNone(qrels)
        self.assertEqual(len(corpus), 2)
        self.assertEqual(len(queries), 1)
        self.assertEqual(corpus['doc1'], 'Hello World')
        self.assertEqual(queries['q1'], 'search query')


if __name__ == "__main__":
    unittest.main()
