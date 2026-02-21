"""
Unit Tests for Multi-Task Retrieval Benchmark

Tests the multi-task benchmark utilities with mocked engines and data.
Fast, deterministic tests without actual model training or MTEB downloads.
"""

import unittest
import numpy as np
from unittest.mock import MagicMock, patch, Mock
import json
import sys

# Mock heavy dependencies before importing benchmark modules
if 'mteb' not in sys.modules:
    sys.modules['mteb'] = MagicMock()

if 'sentence_transformers' not in sys.modules:
    sys.modules['sentence_transformers'] = MagicMock()

if 'qdrant_client' not in sys.modules:
    sys.modules['qdrant_client'] = MagicMock()
    sys.modules['qdrant_client.models'] = MagicMock()

if 'torch' not in sys.modules:
    sys.modules['torch'] = MagicMock()
    sys.modules['torch.nn'] = MagicMock()
    sys.modules['torch.optim'] = MagicMock()

from benchmark_multitask import (
    parse_tasks,
    TASK_SUITES,
    jaccard_similarity,
    compute_stability,
    compute_learning_gain,
    compute_aggregate_summary
)


class TestTaskParsing(unittest.TestCase):
    """Test task selection parsing."""
    
    def test_parse_tasks_suite_mini(self):
        """Test parsing mini suite preset."""
        tasks = parse_tasks("mini")
        self.assertEqual(tasks, TASK_SUITES["mini"])
        self.assertIn("SciFact", tasks)
    
    def test_parse_tasks_suite_small(self):
        """Test parsing small suite preset."""
        tasks = parse_tasks("small")
        self.assertEqual(tasks, TASK_SUITES["small"])
        self.assertEqual(len(tasks), 2)
    
    def test_parse_tasks_suite_medium(self):
        """Test parsing medium suite preset."""
        tasks = parse_tasks("medium")
        self.assertEqual(tasks, TASK_SUITES["medium"])
        self.assertGreaterEqual(len(tasks), 3)
    
    def test_parse_tasks_suite_research(self):
        """Test parsing research suite preset."""
        tasks = parse_tasks("research")
        self.assertEqual(tasks, TASK_SUITES["research"])
        self.assertGreaterEqual(len(tasks), 3)
    
    def test_parse_tasks_comma_separated(self):
        """Test parsing comma-separated task list."""
        tasks = parse_tasks("SciFact,NFCorpus,FiQA2018")
        self.assertEqual(tasks, ["SciFact", "NFCorpus", "FiQA2018"])
    
    def test_parse_tasks_single_task(self):
        """Test parsing single task."""
        tasks = parse_tasks("SciFact")
        self.assertEqual(tasks, ["SciFact"])
    
    def test_parse_tasks_with_spaces(self):
        """Test parsing handles extra spaces."""
        tasks = parse_tasks(" SciFact , NFCorpus , FiQA2018 ")
        self.assertEqual(tasks, ["SciFact", "NFCorpus", "FiQA2018"])
    
    def test_parse_tasks_empty_strings_filtered(self):
        """Test that empty strings are filtered out."""
        tasks = parse_tasks("SciFact,,NFCorpus")
        self.assertEqual(tasks, ["SciFact", "NFCorpus"])


class TestJaccardSimilarity(unittest.TestCase):
    """Test Jaccard similarity computation."""
    
    def test_jaccard_identical_sets(self):
        """Test Jaccard similarity for identical sets."""
        set_a = {'doc1', 'doc2', 'doc3'}
        set_b = {'doc1', 'doc2', 'doc3'}
        
        jac = jaccard_similarity(set_a, set_b)
        self.assertAlmostEqual(jac, 1.0, places=5)
    
    def test_jaccard_disjoint_sets(self):
        """Test Jaccard similarity for disjoint sets."""
        set_a = {'doc1', 'doc2', 'doc3'}
        set_b = {'doc4', 'doc5', 'doc6'}
        
        jac = jaccard_similarity(set_a, set_b)
        self.assertAlmostEqual(jac, 0.0, places=5)
    
    def test_jaccard_partial_overlap(self):
        """Test Jaccard similarity for partial overlap."""
        set_a = {'doc1', 'doc2', 'doc3'}
        set_b = {'doc2', 'doc3', 'doc4'}
        
        # Intersection: {doc2, doc3} = 2
        # Union: {doc1, doc2, doc3, doc4} = 4
        # Jaccard = 2/4 = 0.5
        jac = jaccard_similarity(set_a, set_b)
        self.assertAlmostEqual(jac, 0.5, places=5)
    
    def test_jaccard_empty_sets(self):
        """Test Jaccard similarity for empty sets."""
        set_a = set()
        set_b = set()
        
        # Both empty should return 1.0 (perfect agreement)
        jac = jaccard_similarity(set_a, set_b)
        self.assertAlmostEqual(jac, 1.0, places=5)
    
    def test_jaccard_one_empty(self):
        """Test Jaccard similarity when one set is empty."""
        set_a = {'doc1', 'doc2'}
        set_b = set()
        
        # One empty means union = set_a, intersection = 0
        jac = jaccard_similarity(set_a, set_b)
        self.assertAlmostEqual(jac, 0.0, places=5)


class TestStabilityComputation(unittest.TestCase):
    """Test stability metric computation."""
    
    def test_compute_stability_basic(self):
        """Test basic stability computation."""
        # Mock engine
        mock_engine = MagicMock()
        mock_engine.collection_name = "test_collection"
        
        # Mock run_inference to return consistent results
        call_count = [0]
        def mock_inference(query_text):
            # Alternate between two similar result sets
            if call_count[0] % 2 == 0:
                result_ids = [1, 2, 3, 4, 5]
            else:
                result_ids = [1, 2, 3, 6, 7]  # 3/7 overlap
            call_count[0] += 1
            return (["text"], result_ids, [0.9], {})
        
        mock_engine.run_inference = mock_inference
        
        # Mock ID mapping
        def mock_retrieve(collection, ids):
            points = []
            for i in ids:
                point = MagicMock()
                point.id = i
                point.payload = {'doc_id': f'doc_{i}'}
                points.append(point)
            return points
        mock_engine.qdrant.retrieve = mock_retrieve
        
        # Compute stability
        queries = {'q1': 'test query 1', 'q2': 'test query 2'}
        avg_jac, jac_list = compute_stability(
            mock_engine,
            queries,
            num_runs=2,
            max_queries=2
        )
        
        # Should have evaluated 2 queries
        self.assertEqual(len(jac_list), 2)
        self.assertIsInstance(avg_jac, float)
        self.assertGreaterEqual(avg_jac, 0.0)
        self.assertLessEqual(avg_jac, 1.0)
    
    def test_compute_stability_perfect(self):
        """Test stability with perfectly consistent results."""
        # Mock engine that always returns same results
        mock_engine = MagicMock()
        mock_engine.collection_name = "test_collection"
        mock_engine.run_inference = lambda q: ([], [1, 2, 3], [0.9], {})
        
        def mock_retrieve(collection, ids):
            points = []
            for i in ids:
                point = MagicMock()
                point.id = i
                point.payload = {'doc_id': f'doc_{i}'}
                points.append(point)
            return points
        mock_engine.qdrant.retrieve = mock_retrieve
        
        # Perfect stability should give Jaccard = 1.0
        queries = {'q1': 'test query'}
        avg_jac, jac_list = compute_stability(
            mock_engine,
            queries,
            num_runs=3,
            max_queries=1
        )
        
        self.assertEqual(len(jac_list), 1)
        self.assertAlmostEqual(jac_list[0], 1.0, places=5)
        self.assertAlmostEqual(avg_jac, 1.0, places=5)
    
    def test_compute_stability_max_queries(self):
        """Test stability respects max_queries limit."""
        mock_engine = MagicMock()
        mock_engine.collection_name = "test_collection"
        mock_engine.run_inference = lambda q: ([], [1, 2], [0.9], {})
        
        def mock_retrieve(collection, ids):
            points = []
            for i in ids:
                point = MagicMock()
                point.id = i
                point.payload = {'doc_id': f'doc_{i}'}
                points.append(point)
            return points
        mock_engine.qdrant.retrieve = mock_retrieve
        
        # Many queries, but limit to 5
        queries = {f'q{i}': f'query {i}' for i in range(100)}
        avg_jac, jac_list = compute_stability(
            mock_engine,
            queries,
            num_runs=2,
            max_queries=5
        )
        
        # Should only evaluate 5
        self.assertEqual(len(jac_list), 5)


class TestLearningGainComputation(unittest.TestCase):
    """Test learning gain computation."""
    
    @patch("benchmark_multitask.evaluate_engine")
    def test_compute_learning_gain_basic(self, mock_evaluate):
        """Test basic learning gain computation."""
        # Mock engine
        mock_engine = MagicMock()
        mock_engine.collection_name = "test_collection"
        mock_engine.run_inference = MagicMock()
        mock_engine.run_sedimentation_cycle = MagicMock()
        
        # Mock evaluate_engine to return different scores before/after
        call_count = [0]
        def mock_eval(engine, queries, qrels, max_queries):
            if call_count[0] == 0:
                # Pre-sedimentation score
                call_count[0] += 1
                return 0.5, [0.5, 0.5]
            else:
                # Post-sedimentation score (improved)
                return 0.6, [0.6, 0.6]
        
        mock_evaluate.side_effect = mock_eval
        
        # Compute learning gain
        queries = {f'q{i}': f'query {i}' for i in range(100)}
        qrels = {f'q{i}': {'doc_1': 1} for i in range(100)}
        
        result = compute_learning_gain(
            mock_engine,
            queries,
            qrels,
            num_queries_train=10,
            max_queries_eval=20,
            epochs=5,
            learning_rate=0.001
        )
        
        # Check result structure
        self.assertIn('pre_ndcg', result)
        self.assertIn('post_ndcg', result)
        self.assertIn('gain', result)
        self.assertIn('gain_percent', result)
        self.assertIn('sediment_time', result)
        
        # Check values
        self.assertEqual(result['pre_ndcg'], 0.5)
        self.assertEqual(result['post_ndcg'], 0.6)
        self.assertAlmostEqual(result['gain'], 0.1, places=5)
        self.assertAlmostEqual(result['gain_percent'], 20.0, places=1)
        
        # Verify sedimentation was called
        mock_engine.run_sedimentation_cycle.assert_called_once()
    
    @patch("benchmark_multitask.evaluate_engine")
    def test_compute_learning_gain_negative(self, mock_evaluate):
        """Test learning gain with negative gain (degradation)."""
        mock_engine = MagicMock()
        mock_engine.run_inference = MagicMock()
        mock_engine.run_sedimentation_cycle = MagicMock()
        
        # Mock degradation
        call_count = [0]
        def mock_eval(engine, queries, qrels, max_queries):
            if call_count[0] == 0:
                call_count[0] += 1
                return 0.6, [0.6]
            else:
                return 0.5, [0.5]
        
        mock_evaluate.side_effect = mock_eval
        
        queries = {'q1': 'query'}
        qrels = {'q1': {'doc_1': 1}}
        
        result = compute_learning_gain(
            mock_engine,
            queries,
            qrels,
            num_queries_train=5,
            max_queries_eval=10,
            epochs=5,
            learning_rate=0.001
        )
        
        # Negative gain
        self.assertLess(result['gain'], 0)
        self.assertLess(result['gain_percent'], 0)
    
    @patch("benchmark_multitask.evaluate_engine")
    def test_compute_learning_gain_zero_pre_ndcg(self, mock_evaluate):
        """Test learning gain when pre-NDCG is zero."""
        mock_engine = MagicMock()
        mock_engine.run_inference = MagicMock()
        mock_engine.run_sedimentation_cycle = MagicMock()
        
        # Pre-NDCG = 0, post-NDCG = 0.1
        mock_evaluate.side_effect = [
            (0.0, [0.0]),  # Pre
            (0.1, [0.1])   # Post
        ]
        
        queries = {'q1': 'query'}
        qrels = {'q1': {'doc_1': 1}}
        
        result = compute_learning_gain(
            mock_engine,
            queries,
            qrels,
            num_queries_train=5,
            max_queries_eval=10,
            epochs=5,
            learning_rate=0.001
        )
        
        # Should handle division by zero gracefully
        self.assertEqual(result['gain_percent'], 0.0)
        self.assertEqual(result['gain'], 0.1)


class TestAggregateSummary(unittest.TestCase):
    """Test aggregate summary computation."""
    
    def test_aggregate_summary_basic(self):
        """Test basic aggregate summary."""
        task_results = [
            {
                'task': 'Task1',
                'status': 'success',
                'retrieval_quality': {'ndcg_10': 0.5, 'ndcg_std': 0.1},
                'stability': {'avg_jaccard': 0.8, 'jaccard_std': 0.05},
                'learning_gain': {'gain': 0.05, 'gain_percent': 10.0}
            },
            {
                'task': 'Task2',
                'status': 'success',
                'retrieval_quality': {'ndcg_10': 0.6, 'ndcg_std': 0.12},
                'stability': {'avg_jaccard': 0.85, 'jaccard_std': 0.04},
                'learning_gain': {'gain': 0.03, 'gain_percent': 5.0}
            }
        ]
        
        summary = compute_aggregate_summary(task_results)
        
        # Check structure
        self.assertEqual(summary['num_tasks'], 2)
        self.assertEqual(summary['num_successful'], 2)
        self.assertEqual(summary['num_failed'], 0)
        
        # Check NDCG aggregation
        self.assertAlmostEqual(summary['aggregate_ndcg']['mean'], 0.55, places=5)
        self.assertAlmostEqual(summary['aggregate_ndcg']['min'], 0.5, places=5)
        self.assertAlmostEqual(summary['aggregate_ndcg']['max'], 0.6, places=5)
        
        # Check stability aggregation
        self.assertAlmostEqual(summary['aggregate_stability']['mean'], 0.825, places=5)
        
        # Check learning gain aggregation
        self.assertAlmostEqual(summary['aggregate_learning_gain']['mean_gain'], 0.04, places=5)
        self.assertAlmostEqual(summary['aggregate_learning_gain']['mean_gain_percent'], 7.5, places=5)
    
    def test_aggregate_summary_with_failures(self):
        """Test aggregate summary with some failed tasks."""
        task_results = [
            {
                'task': 'Task1',
                'status': 'success',
                'retrieval_quality': {'ndcg_10': 0.5, 'ndcg_std': 0.1},
                'stability': {'avg_jaccard': 0.8, 'jaccard_std': 0.05},
                'learning_gain': {'gain': 0.05, 'gain_percent': 10.0}
            },
            {
                'task': 'Task2',
                'status': 'failed',
                'error': 'Dataset load failed'
            },
            {
                'task': 'Task3',
                'status': 'success',
                'retrieval_quality': {'ndcg_10': 0.6, 'ndcg_std': 0.12},
                'stability': {'avg_jaccard': 0.85, 'jaccard_std': 0.04},
                'learning_gain': {'gain': -0.02, 'gain_percent': -3.0}
            }
        ]
        
        summary = compute_aggregate_summary(task_results)
        
        self.assertEqual(summary['num_tasks'], 3)
        self.assertEqual(summary['num_successful'], 2)
        self.assertEqual(summary['num_failed'], 1)
        
        # Should only aggregate successful tasks
        self.assertEqual(summary['aggregate_learning_gain']['positive_gains'], 1)
        self.assertEqual(summary['aggregate_learning_gain']['negative_gains'], 1)
    
    def test_aggregate_summary_all_failures(self):
        """Test aggregate summary when all tasks fail."""
        task_results = [
            {
                'task': 'Task1',
                'status': 'failed',
                'error': 'Load failed'
            },
            {
                'task': 'Task2',
                'status': 'failed',
                'error': 'Network error'
            }
        ]
        
        summary = compute_aggregate_summary(task_results)
        
        self.assertEqual(summary['num_tasks'], 2)
        self.assertEqual(summary['num_successful'], 0)
        self.assertEqual(summary['num_failed'], 2)
        self.assertIn('error', summary)
    
    def test_aggregate_summary_positive_negative_gains(self):
        """Test aggregate summary tracks positive/negative gains."""
        task_results = [
            {
                'task': 'Task1',
                'status': 'success',
                'retrieval_quality': {'ndcg_10': 0.5, 'ndcg_std': 0.1},
                'stability': {'avg_jaccard': 0.8, 'jaccard_std': 0.05},
                'learning_gain': {'gain': 0.05, 'gain_percent': 10.0}
            },
            {
                'task': 'Task2',
                'status': 'success',
                'retrieval_quality': {'ndcg_10': 0.6, 'ndcg_std': 0.12},
                'stability': {'avg_jaccard': 0.85, 'jaccard_std': 0.04},
                'learning_gain': {'gain': -0.01, 'gain_percent': -2.0}
            },
            {
                'task': 'Task3',
                'status': 'success',
                'retrieval_quality': {'ndcg_10': 0.55, 'ndcg_std': 0.11},
                'stability': {'avg_jaccard': 0.82, 'jaccard_std': 0.045},
                'learning_gain': {'gain': 0.02, 'gain_percent': 3.6}
            }
        ]
        
        summary = compute_aggregate_summary(task_results)
        
        # Should have 2 positive, 1 negative
        self.assertEqual(summary['aggregate_learning_gain']['positive_gains'], 2)
        self.assertEqual(summary['aggregate_learning_gain']['negative_gains'], 1)


class TestBenchmarkWorkflow(unittest.TestCase):
    """Test overall benchmark workflow components."""
    
    @patch("benchmark_multitask.load_mteb_data")
    @patch("benchmark_multitask.AntigravityEngine")
    @patch("benchmark_multitask.evaluate_engine")
    def test_benchmark_workflow_mock(self, mock_evaluate, mock_engine_class, mock_load_data):
        """Test benchmark workflow with mocked components."""
        # Mock data loading
        mock_corpus = {'doc1': 'text1', 'doc2': 'text2'}
        mock_queries = {'q1': 'query1', 'q2': 'query2'}
        mock_qrels = {'q1': {'doc1': 1}, 'q2': {'doc2': 1}}
        mock_load_data.return_value = (mock_corpus, mock_queries, mock_qrels)
        
        # Mock engine
        mock_engine = MagicMock()
        mock_engine.collection_name = "test_collection"
        mock_engine.run_inference = MagicMock()
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
        
        # Mock evaluation (two calls: pre and post)
        mock_evaluate.side_effect = [
            (0.5, [0.5, 0.5]),  # Pre-sedimentation
            (0.55, [0.55, 0.55])  # Post-sedimentation
        ]
        
        # Simulate workflow
        # 1. Load data
        corpus, queries, qrels = mock_load_data("SciFact")
        self.assertIsNotNone(corpus)
        self.assertIsNotNone(queries)
        
        # 2. Initialize engine
        from benchmark_multitask import AntigravityEngine
        engine = AntigravityEngine(qdrant_location=":memory:", model_name="test-model")
        
        # 3. Ingest corpus
        doc_texts = list(corpus.values())
        doc_payloads = [{"doc_id": did} for did in corpus.keys()]
        engine.ingest(doc_texts, doc_payloads)
        
        # 4. Evaluate (pre)
        from benchmark_multitask import evaluate_engine
        pre_ndcg, _ = evaluate_engine(engine, queries, qrels, max_queries=10)
        self.assertEqual(pre_ndcg, 0.5)
        
        # 5. Run sedimentation
        engine.run_sedimentation_cycle(threshold=3, learning_rate=0.001, epochs=5)
        
        # 6. Evaluate (post)
        post_ndcg, _ = evaluate_engine(engine, queries, qrels, max_queries=10)
        self.assertEqual(post_ndcg, 0.55)
        
        # Verify calls
        mock_engine.ingest.assert_called_once()
        mock_engine.run_sedimentation_cycle.assert_called_once()
        self.assertEqual(mock_evaluate.call_count, 2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
