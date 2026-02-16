"""
Unit Tests for Benchmark Distillation Utilities

Tests the benchmark utilities for comparative training modes with mocked engines.
Fast, deterministic tests without actual model training.
"""

import unittest
import numpy as np
from unittest.mock import MagicMock, patch, Mock
import json
import sys

if 'mteb' not in sys.modules:
    sys.modules['mteb'] = MagicMock()

from benchmark_distillation import (
    dcg_at_k,
    ndcg_at_k,
    find_keys,
    find_payload,
    map_predicted_ids,
    evaluate_engine,
)


class TestMetricCalculations(unittest.TestCase):
    """Test metric calculation functions."""

    def test_dcg_at_k_perfect(self):
        """Test DCG with perfect relevance."""
        # Perfectly ranked results: [1, 1, 1, 0, 0]
        r = [1, 1, 1, 0, 0]
        dcg = dcg_at_k(r, 5)
        
        # DCG = 1/log2(2) + 1/log2(3) + 1/log2(4)
        expected = 1.0 + 1.0/np.log2(3) + 1.0/np.log2(4)
        self.assertAlmostEqual(dcg, expected, places=5)

    def test_dcg_at_k_empty(self):
        """Test DCG with empty relevance list."""
        r = []
        dcg = dcg_at_k(r, 5)
        self.assertEqual(dcg, 0.0)

    def test_dcg_at_k_limit(self):
        """Test DCG respects k limit."""
        r = [1, 1, 1, 1, 1, 1, 1]
        
        # DCG@3 should only use first 3 items
        dcg3 = dcg_at_k(r, 3)
        dcg7 = dcg_at_k(r, 7)
        
        # DCG should increase with k
        self.assertLess(dcg3, dcg7)

    def test_ndcg_at_k_perfect(self):
        """Test NDCG with perfect ranking."""
        # Relevance in perfect order
        r = [3, 2, 1, 0, 0]
        
        # NDCG should be 1.0 (perfect ranking)
        ndcg = ndcg_at_k(r, 5)
        self.assertAlmostEqual(ndcg, 1.0, places=5)

    def test_ndcg_at_k_worst(self):
        """Test NDCG with worst ranking."""
        # Relevance in worst order (relevant docs at bottom)
        r = [0, 0, 0, 3, 2]
        
        # NDCG should be < 1.0
        ndcg = ndcg_at_k(r, 5)
        self.assertLess(ndcg, 1.0)
        self.assertGreater(ndcg, 0.0)

    def test_ndcg_at_k_no_relevant(self):
        """Test NDCG with no relevant documents."""
        r = [0, 0, 0, 0, 0]
        
        # NDCG should be 0.0
        ndcg = ndcg_at_k(r, 5)
        self.assertEqual(ndcg, 0.0)

    def test_ndcg_at_k_all_relevant(self):
        """Test NDCG with all relevant documents."""
        # All equally relevant
        r = [1, 1, 1, 1, 1]
        
        # Perfect ranking for uniform relevance
        ndcg = ndcg_at_k(r, 5)
        self.assertAlmostEqual(ndcg, 1.0, places=5)


class TestDataLoadingHelpers(unittest.TestCase):
    """Test MTEB data loading helper functions."""

    def test_find_keys_direct(self):
        """Test finding keys at top level."""
        data = {
            'corpus': {'doc1': 'text1'},
            'queries': {'q1': 'query1'}
        }
        
        result = find_keys(data, ['corpus', 'queries'])
        self.assertIsNotNone(result)
        self.assertEqual(result, data)

    def test_find_keys_nested(self):
        """Test finding keys in nested dict."""
        data = {
            'outer': {
                'inner': {
                    'corpus': {'doc1': 'text1'},
                    'queries': {'q1': 'query1'}
                }
            }
        }
        
        result = find_keys(data, ['corpus', 'queries'])
        self.assertIsNotNone(result)
        self.assertIn('corpus', result)
        self.assertIn('queries', result)

    def test_find_keys_not_found(self):
        """Test finding keys that don't exist."""
        data = {
            'corpus': {'doc1': 'text1'}
        }
        
        result = find_keys(data, ['corpus', 'queries'])
        self.assertIsNone(result)

    def test_find_keys_non_dict(self):
        """Test finding keys in non-dict returns None."""
        data = "not a dict"
        
        result = find_keys(data, ['corpus', 'queries'])
        self.assertIsNone(result)

    def test_find_payload_direct(self):
        """Test finding payload at top level."""
        data = {
            'corpus': {'doc1': 'text1'},
            'queries': {'q1': 'query1'}
        }
        
        result = find_payload(data, 'corpus')
        self.assertEqual(result, {'doc1': 'text1'})

    def test_find_payload_nested(self):
        """Test finding payload in nested dict."""
        data = {
            'outer': {
                'inner': {
                    'corpus': {'doc1': 'text1'}
                }
            }
        }
        
        result = find_payload(data, 'corpus')
        self.assertEqual(result, {'doc1': 'text1'})

    def test_find_payload_not_found(self):
        """Test finding non-existent payload returns None."""
        data = {
            'other': {'doc1': 'text1'}
        }
        
        result = find_payload(data, 'corpus')
        self.assertIsNone(result)


class TestIDMapping(unittest.TestCase):
    """Test ID mapping functions."""

    def test_map_predicted_ids_success(self):
        """Test successful ID mapping."""
        # Mock engine
        mock_engine = MagicMock()
        mock_engine.collection_name = "test_collection"
        
        # Mock Qdrant retrieve response
        mock_point1 = MagicMock()
        mock_point1.id = 1
        mock_point1.payload = {'doc_id': 'doc_A'}
        
        mock_point2 = MagicMock()
        mock_point2.id = 2
        mock_point2.payload = {'doc_id': 'doc_B'}
        
        mock_engine.qdrant.retrieve.return_value = [mock_point1, mock_point2]
        
        # Map IDs
        pred_ids = [1, 2]
        mapped = map_predicted_ids(mock_engine, pred_ids)
        
        self.assertEqual(mapped, ['doc_A', 'doc_B'])
        mock_engine.qdrant.retrieve.assert_called_once_with(
            "test_collection",
            ids=pred_ids
        )

    def test_map_predicted_ids_fallback(self):
        """Test ID mapping falls back to raw IDs on error."""
        # Mock engine that raises exception
        mock_engine = MagicMock()
        mock_engine.collection_name = "test_collection"
        mock_engine.qdrant.retrieve.side_effect = Exception("Retrieve failed")
        
        # Should fallback to string conversion
        pred_ids = [1, 2, 3]
        mapped = map_predicted_ids(mock_engine, pred_ids)
        
        self.assertEqual(mapped, ['1', '2', '3'])

    def test_map_predicted_ids_missing_doc_id(self):
        """Test ID mapping when doc_id not in payload."""
        # Mock engine
        mock_engine = MagicMock()
        mock_engine.collection_name = "test_collection"
        
        # Mock point without doc_id
        mock_point = MagicMock()
        mock_point.id = 42
        mock_point.payload = {}  # No doc_id or id
        
        mock_engine.qdrant.retrieve.return_value = [mock_point]
        
        # Should use point ID as fallback
        pred_ids = [42]
        mapped = map_predicted_ids(mock_engine, pred_ids)
        
        self.assertEqual(mapped, ['42'])


class TestEngineEvaluation(unittest.TestCase):
    """Test engine evaluation function."""

    def test_evaluate_engine_basic(self):
        """Test basic engine evaluation."""
        # Mock engine
        mock_engine = MagicMock()
        mock_engine.collection_name = "test_collection"
        
        # Mock run_inference to return predictable results
        def mock_inference(query_text):
            # Return (results, ids, scores, metadata)
            return (
                ["Result text"],
                [1, 2, 3],  # Document IDs
                [0.9, 0.8, 0.7],
                {}
            )
        mock_engine.run_inference = mock_inference
        
        # Mock ID mapping
        def mock_retrieve(collection, ids):
            # Return points with doc_ids
            points = []
            for i in ids:
                point = MagicMock()
                point.id = i
                point.payload = {'doc_id': f'doc_{i}'}
                points.append(point)
            return points
        mock_engine.qdrant.retrieve = mock_retrieve
        
        # Queries and qrels
        queries = {
            'q1': 'test query 1',
            'q2': 'test query 2'
        }
        qrels = {
            'q1': {'doc_1': 1, 'doc_2': 1},
            'q2': {'doc_1': 1, 'doc_3': 0}
        }
        
        # Evaluate
        avg_ndcg, ndcg_list = evaluate_engine(
            mock_engine,
            queries,
            qrels,
            max_queries=None
        )
        
        # Should have evaluated 2 queries
        self.assertEqual(len(ndcg_list), 2)
        self.assertIsInstance(avg_ndcg, float)
        self.assertGreaterEqual(avg_ndcg, 0.0)
        self.assertLessEqual(avg_ndcg, 1.0)

    def test_evaluate_engine_no_relevant_docs(self):
        """Test evaluation when no relevant docs exist."""
        # Mock engine
        mock_engine = MagicMock()
        mock_engine.collection_name = "test_collection"
        mock_engine.run_inference = lambda q: ([], [1, 2], [0.9, 0.8], {})
        
        def mock_retrieve(collection, ids):
            points = []
            for i in ids:
                point = MagicMock()
                point.id = i
                point.payload = {'doc_id': f'doc_{i}'}
                points.append(point)
            return points
        mock_engine.qdrant.retrieve = mock_retrieve
        
        # Query with no relevant docs in qrels
        queries = {'q1': 'test query'}
        qrels = {}  # No relevance judgments
        
        # Should skip queries with no qrels
        avg_ndcg, ndcg_list = evaluate_engine(
            mock_engine,
            queries,
            qrels
        )
        
        self.assertEqual(len(ndcg_list), 0)
        self.assertEqual(avg_ndcg, 0.0)

    def test_evaluate_engine_max_queries(self):
        """Test evaluation respects max_queries limit."""
        # Mock engine
        mock_engine = MagicMock()
        mock_engine.collection_name = "test_collection"
        mock_engine.run_inference = lambda q: ([], [1, 2], [0.9, 0.8], {})
        
        def mock_retrieve(collection, ids):
            points = []
            for i in ids:
                point = MagicMock()
                point.id = i
                point.payload = {'doc_id': f'doc_{i}'}
                points.append(point)
            return points
        mock_engine.qdrant.retrieve = mock_retrieve
        
        # Many queries
        queries = {f'q{i}': f'query {i}' for i in range(100)}
        qrels = {f'q{i}': {'doc_1': 1} for i in range(100)}
        
        # Limit to 10 queries
        avg_ndcg, ndcg_list = evaluate_engine(
            mock_engine,
            queries,
            qrels,
            max_queries=10
        )
        
        # Should only evaluate 10
        self.assertEqual(len(ndcg_list), 10)

    def test_evaluate_engine_perfect_results(self):
        """Test evaluation with perfect ranking."""
        # Mock engine that returns perfect results
        mock_engine = MagicMock()
        mock_engine.collection_name = "test_collection"
        
        # Return relevant docs first
        mock_engine.run_inference = lambda q: ([], [1, 2, 3], [0.9, 0.8, 0.7], {})
        
        def mock_retrieve(collection, ids):
            points = []
            for i in ids:
                point = MagicMock()
                point.id = i
                point.payload = {'doc_id': f'doc_{i}'}
                points.append(point)
            return points
        mock_engine.qdrant.retrieve = mock_retrieve
        
        # Query with perfect ranking
        queries = {'q1': 'test query'}
        qrels = {'q1': {'doc_1': 1, 'doc_2': 1, 'doc_3': 1}}
        
        avg_ndcg, ndcg_list = evaluate_engine(
            mock_engine,
            queries,
            qrels
        )
        
        # Perfect ranking should give NDCG = 1.0
        self.assertEqual(len(ndcg_list), 1)
        self.assertAlmostEqual(ndcg_list[0], 1.0, places=5)
        self.assertAlmostEqual(avg_ndcg, 1.0, places=5)


class TestBenchmarkIntegration(unittest.TestCase):
    """Test benchmark workflow components."""

    @patch("benchmark_distillation.AntigravityEngine")
    def test_benchmark_workflow_mock(self, mock_engine_class):
        """Test that benchmark components work together with mocked engine."""
        # Mock engine instance
        mock_engine = MagicMock()
        mock_engine.collection_name = "test_collection"
        mock_engine.training_mode = "baseline"
        mock_engine.run_inference = lambda q: ([], [1, 2], [0.9, 0.8], {})
        mock_engine.run_sedimentation_cycle = MagicMock()
        mock_engine.ingest = MagicMock()
        
        def mock_retrieve(collection, ids):
            points = []
            for i in ids:
                point = MagicMock()
                point.id = i
                point.payload = {'doc_id': f'doc_{i}'}
                points.append(point)
            return points
        mock_engine.qdrant.retrieve = mock_retrieve
        
        mock_engine_class.return_value = mock_engine
        
        # Simulate a simple benchmark cycle
        queries = {'q1': 'test query', 'q2': 'another query'}
        qrels = {
            'q1': {'doc_1': 1, 'doc_2': 0},
            'q2': {'doc_1': 0, 'doc_2': 1}
        }
        
        # Run inference on queries
        for qid, query_text in queries.items():
            mock_engine.run_inference(query_text)
        
        # Run sedimentation
        mock_engine.run_sedimentation_cycle(
            threshold=3,
            learning_rate=0.001,
            epochs=5
        )
        
        # Evaluate
        avg_ndcg, ndcg_list = evaluate_engine(
            mock_engine,
            queries,
            qrels
        )
        
        # Verify workflow completed
        self.assertIsInstance(avg_ndcg, float)
        self.assertEqual(len(ndcg_list), 2)
        mock_engine.run_sedimentation_cycle.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
