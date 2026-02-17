"""
Unit Tests for Recursive Decomposition Engine

Tests MockDecomposer, DecompositionNode, RecursiveRetrievalEngine,
and aggregation strategies without requiring external services.
"""

import unittest
import time
from unittest.mock import patch

from recursive_decomposer import (
    DecompositionStrategy,
    DecompositionNode,
    DecompositionTrace,
    MockDecomposer,
    OllamaDecomposer,
    RecursiveRetrievalEngine,
)


class TestMockDecomposer(unittest.TestCase):
    """Test the MockDecomposer deterministic splitting logic."""

    def setUp(self):
        """Set up decomposer with default settings."""
        self.decomposer = MockDecomposer(max_depth=3, min_length=20)

    def test_atomic_short_query(self):
        """Short query below min_length is treated as base case and returned unchanged."""
        query = "short query"
        self.assertTrue(self.decomposer.is_base_case(query))
        result = self.decomposer.decompose(query)
        self.assertEqual(result, [query])

    def test_conjunction_split_and(self):
        """Query containing ' and ' is split into two parts."""
        query = "cats and dogs"
        # len("cats and dogs") == 13, below min_length=20, so decompose still splits
        # Actually, decompose does not check is_base_case; it just looks for conjunctions.
        result = self.decomposer.decompose(query)
        self.assertEqual(result, ["cats", "dogs"])

    def test_conjunction_split_semicolon(self):
        """Semicolon conjunction has highest priority and is matched first."""
        query = "query one; query two and query three"
        result = self.decomposer.decompose(query)
        # '; ' is checked before ' and ', so split happens on semicolon
        self.assertEqual(result, ["query one", "query two and query three"])

    def test_conjunction_split_or(self):
        """Query with ' or ' conjunction splits correctly."""
        query = "apples or oranges for a healthy diet"
        result = self.decomposer.decompose(query)
        self.assertEqual(result, ["apples", "oranges for a healthy diet"])

    def test_no_conjunction(self):
        """Long query with no conjunctions returns [query] and is a base case."""
        query = "this is a long query with no splitting points whatsoever"
        self.assertTrue(self.decomposer.is_base_case(query))
        result = self.decomposer.decompose(query)
        self.assertEqual(result, [query])

    def test_case_insensitive_split(self):
        """Conjunction matching is case-insensitive but original text is preserved."""
        query = "Cats AND Dogs are great pets"
        result = self.decomposer.decompose(query)
        self.assertEqual(result, ["Cats", "Dogs are great pets"])

    def test_empty_input(self):
        """Empty string is a base case and decompose returns it unchanged."""
        query = ""
        self.assertTrue(self.decomposer.is_base_case(query))
        result = self.decomposer.decompose(query)
        self.assertEqual(result, [query])

    def test_nested_conjunctions(self):
        """Only the first priority conjunction is used for splitting."""
        # CONJUNCTIONS order: '; ', ' and ', ' or ', ' versus ', ' compared to '
        # '; ' not found, ' and ' found first -> split on ' and '
        query = "a and b or c are interesting topics"
        result = self.decomposer.decompose(query)
        self.assertEqual(result, ["a", "b or c are interesting topics"])


class TestDecompositionNode(unittest.TestCase):
    """Test DecompositionNode data structure."""

    def test_node_creation(self):
        """Node fields are initialized correctly."""
        node = DecompositionNode(
            query="test query",
            depth=2,
            parent_id="parent123",
        )
        self.assertEqual(node.query, "test query")
        self.assertEqual(node.depth, 2)
        self.assertEqual(node.parent_id, "parent123")
        self.assertIsInstance(node.node_id, str)
        self.assertEqual(len(node.node_id), 8)
        self.assertEqual(node.children, [])
        self.assertEqual(node.results, [])
        self.assertEqual(node.scores, [])
        self.assertFalse(node.is_base_case)
        self.assertEqual(node.metadata, {})

    def test_node_tree_structure(self):
        """Parent-child relationships are maintained correctly."""
        parent = DecompositionNode(query="parent", depth=0, parent_id=None)
        child_a = DecompositionNode(query="child a", depth=1, parent_id=parent.node_id)
        child_b = DecompositionNode(query="child b", depth=1, parent_id=parent.node_id)
        parent.children.append(child_a)
        parent.children.append(child_b)

        self.assertEqual(len(parent.children), 2)
        self.assertEqual(parent.children[0].query, "child a")
        self.assertEqual(parent.children[1].query, "child b")
        self.assertEqual(child_a.parent_id, parent.node_id)
        self.assertEqual(child_b.parent_id, parent.node_id)
        self.assertIsNone(parent.parent_id)


class MockAntigravityEngine:
    """Fake engine returning deterministic results for testing."""

    def __init__(self, results=None):
        self.call_count = 0
        self._results = results or list(range(10))

    def run_inference(self, query):
        self.call_count += 1
        # Returns (std_top, chel_top, mask, jaccard)
        return self._results, self._results, None, 1.0


class TestRecursiveRetrievalEngine(unittest.TestCase):
    """Test the recursive retrieval pipeline with mock components."""

    def setUp(self):
        """Set up engine with mock dependencies."""
        self.mock_engine = MockAntigravityEngine()
        self.decomposer = MockDecomposer(max_depth=3, min_length=20)

    @patch("recursive_decomposer.get_logger")
    def test_atomic_passthrough(self, mock_get_logger):
        """Simple short query produces a single-node trace with engine results."""
        mock_get_logger.return_value.log_event = lambda *a, **kw: None
        engine = RecursiveRetrievalEngine(
            engine=self.mock_engine,
            decomposer=self.decomposer,
            aggregation_strategy="rrf",
            max_depth=3,
            top_k=10,
        )
        trace = engine.run_recursive_inference("short")

        self.assertEqual(trace.total_nodes, 1)
        self.assertEqual(trace.total_retrieval_calls, 1)
        self.assertTrue(trace.root_node.is_base_case)
        # Results come through aggregation, should contain engine doc IDs
        self.assertGreater(len(trace.final_results), 0)

    @patch("recursive_decomposer.get_logger")
    def test_multi_leaf_retrieval(self, mock_get_logger):
        """Compound query produces multiple leaves and calls engine for each."""
        mock_get_logger.return_value.log_event = lambda *a, **kw: None
        engine = RecursiveRetrievalEngine(
            engine=self.mock_engine,
            decomposer=self.decomposer,
            aggregation_strategy="rrf",
            max_depth=3,
            top_k=10,
        )
        trace = engine.run_recursive_inference("cats and dogs are popular pets")

        # "cats and dogs are popular pets" splits to ["cats", "dogs are popular pets"]
        # Both sub-queries are short (base cases), so 2 leaves
        self.assertEqual(trace.total_retrieval_calls, 2)
        self.assertEqual(self.mock_engine.call_count, 2)
        self.assertGreater(trace.total_nodes, 1)

    @patch("recursive_decomposer.get_logger")
    def test_depth_limit_respected(self, mock_get_logger):
        """Max depth setting prevents deeper recursion."""
        mock_get_logger.return_value.log_event = lambda *a, **kw: None
        # Use a decomposer with min_length=1 so everything can decompose,
        # but max_depth=1 on the engine to cap recursion.
        aggressive_decomposer = MockDecomposer(max_depth=10, min_length=1)
        engine = RecursiveRetrievalEngine(
            engine=self.mock_engine,
            decomposer=aggressive_decomposer,
            aggregation_strategy="rrf",
            max_depth=1,
            top_k=10,
        )
        # This query has nested conjunctions but depth should be limited
        trace = engine.run_recursive_inference("a and b or c and d versus e")

        self.assertLessEqual(trace.max_depth_reached, 1)

    @patch("recursive_decomposer.get_logger")
    def test_trace_completeness(self, mock_get_logger):
        """Trace object has all metrics populated after a run."""
        mock_get_logger.return_value.log_event = lambda *a, **kw: None
        engine = RecursiveRetrievalEngine(
            engine=self.mock_engine,
            decomposer=self.decomposer,
            aggregation_strategy="rrf",
            max_depth=3,
            top_k=10,
        )
        trace = engine.run_recursive_inference("cats and dogs are popular pets")

        self.assertIsInstance(trace, DecompositionTrace)
        self.assertEqual(trace.root_query, "cats and dogs are popular pets")
        self.assertIsNotNone(trace.root_node)
        self.assertGreater(trace.total_nodes, 0)
        self.assertGreaterEqual(trace.max_depth_reached, 0)
        self.assertGreater(trace.total_retrieval_calls, 0)
        self.assertGreaterEqual(trace.elapsed_seconds, 0)
        self.assertIsInstance(trace.final_results, list)
        self.assertIsInstance(trace.final_scores, list)
        self.assertEqual(len(trace.final_results), len(trace.final_scores))


class TestOllamaDecomposerSSRF(unittest.TestCase):
    """Test SSRF URL validation for OllamaDecomposer (Finding F-008)."""

    def test_localhost_url_allowed(self):
        """localhost URL is accepted."""
        decomposer = OllamaDecomposer(ollama_url="http://localhost:11434/api/generate")
        self.assertEqual(decomposer.ollama_url, "http://localhost:11434/api/generate")

    def test_127_0_0_1_url_allowed(self):
        """127.0.0.1 URL is accepted."""
        decomposer = OllamaDecomposer(ollama_url="http://127.0.0.1:11434/api/generate")
        self.assertEqual(decomposer.ollama_url, "http://127.0.0.1:11434/api/generate")

    def test_ipv6_localhost_allowed(self):
        """IPv6 localhost (::1) URL is accepted."""
        decomposer = OllamaDecomposer(ollama_url="http://[::1]:11434/api/generate")
        self.assertEqual(decomposer.ollama_url, "http://[::1]:11434/api/generate")

    def test_external_host_rejected(self):
        """External host is rejected with ValueError."""
        with self.assertRaises(ValueError) as ctx:
            OllamaDecomposer(ollama_url="http://example.com:11434/api/generate")
        self.assertIn("SSRF protection", str(ctx.exception))
        self.assertIn("example.com", str(ctx.exception))

    def test_external_ip_rejected(self):
        """External IP address is rejected with ValueError."""
        with self.assertRaises(ValueError) as ctx:
            OllamaDecomposer(ollama_url="http://192.168.1.100:11434/api/generate")
        self.assertIn("SSRF protection", str(ctx.exception))
        self.assertIn("192.168.1.100", str(ctx.exception))

    def test_invalid_url_rejected(self):
        """Malformed URL without hostname is rejected."""
        with self.assertRaises(ValueError) as ctx:
            OllamaDecomposer(ollama_url="not-a-valid-url")
        self.assertIn("Invalid URL", str(ctx.exception))


class TestOllamaDecomposerExceptionHandling(unittest.TestCase):
    """Test specific exception handling in OllamaDecomposer.decompose (Finding F-019)."""

    def test_connection_error_returns_fallback(self):
        """Connection error returns [query] fallback."""
        with patch('recursive_decomposer.ConnectionError', Exception):
            decomposer = OllamaDecomposer(ollama_url="http://localhost:11434/api/generate")
            
            # Mock requests to raise ConnectionError
            with patch.object(decomposer, 'requests') as mock_requests:
                # Import the actual exception from the module
                from recursive_decomposer import ConnectionError as ConnErr
                mock_requests.post.side_effect = ConnErr("Connection failed")
                
                result = decomposer.decompose("test query")
                self.assertEqual(result, ["test query"])

    def test_timeout_error_returns_fallback(self):
        """Timeout error returns [query] fallback."""
        decomposer = OllamaDecomposer(ollama_url="http://localhost:11434/api/generate")
        
        # Mock requests to raise Timeout
        with patch.object(decomposer, 'requests') as mock_requests:
            from recursive_decomposer import Timeout as TimeoutErr
            mock_requests.post.side_effect = TimeoutErr("Request timed out")
            
            result = decomposer.decompose("test query")
            self.assertEqual(result, ["test query"])

    def test_request_exception_returns_fallback(self):
        """RequestException returns [query] fallback."""
        decomposer = OllamaDecomposer(ollama_url="http://localhost:11434/api/generate")
        
        # Mock requests to raise RequestException
        with patch.object(decomposer, 'requests') as mock_requests:
            from recursive_decomposer import RequestException as ReqErr
            mock_requests.post.side_effect = ReqErr("Request failed")
            
            result = decomposer.decompose("test query")
            self.assertEqual(result, ["test query"])

    def test_json_decode_error_returns_fallback(self):
        """JSON decode error returns [query] fallback."""
        decomposer = OllamaDecomposer(ollama_url="http://localhost:11434/api/generate")
        
        # Mock response with invalid JSON
        with patch.object(decomposer, 'requests') as mock_requests:
            mock_response = mock_requests.post.return_value
            mock_response.status_code = 200
            mock_response.json.side_effect = ValueError("Invalid JSON")
            
            result = decomposer.decompose("test query")
            self.assertEqual(result, ["test query"])

    def test_key_error_returns_fallback(self):
        """KeyError in response parsing returns [query] fallback."""
        decomposer = OllamaDecomposer(ollama_url="http://localhost:11434/api/generate")
        
        # Mock response that raises KeyError
        with patch.object(decomposer, 'requests') as mock_requests:
            mock_response = mock_requests.post.return_value
            mock_response.status_code = 200
            # Make json() return object that raises KeyError on .get()
            mock_json = unittest.mock.MagicMock()
            mock_json.get.side_effect = KeyError("response")
            mock_response.json.return_value = mock_json
            
            result = decomposer.decompose("test query")
            self.assertEqual(result, ["test query"])

    def test_no_requests_library_returns_fallback(self):
        """Missing requests library returns [query] fallback."""
        decomposer = OllamaDecomposer(ollama_url="http://localhost:11434/api/generate")
        decomposer.requests = None
        
        result = decomposer.decompose("test query")
        self.assertEqual(result, ["test query"])
    
    def test_non_200_status_returns_fallback(self):
        """Non-200 HTTP status returns [query] fallback."""
        decomposer = OllamaDecomposer(ollama_url="http://localhost:11434/api/generate")
        
        # Mock response with 500 status
        with patch.object(decomposer, 'requests') as mock_requests:
            mock_response = mock_requests.post.return_value
            mock_response.status_code = 500
            
            result = decomposer.decompose("test query")
            self.assertEqual(result, ["test query"])


class TestOllamaDecomposerBehavior(unittest.TestCase):
    """Test OllamaDecomposer behavior including parsing and decompose logic (Finding F-014)."""

    def setUp(self):
        """Set up decomposer instance for tests."""
        self.decomposer = OllamaDecomposer(ollama_url="http://localhost:11434/api/generate")

    def test_parse_response_numbered_list(self):
        """_parse_response handles numbered list format correctly."""
        text = """1. What is machine learning?
2. How does neural network work?
3. What are transformers?"""
        result = self.decomposer._parse_response(text)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], "What is machine learning?")
        self.assertEqual(result[1], "How does neural network work?")
        self.assertEqual(result[2], "What are transformers?")

    def test_parse_response_numbered_list_with_parentheses(self):
        """_parse_response handles numbered list with parentheses format."""
        text = """1) First query
2) Second query
3) Third query"""
        result = self.decomposer._parse_response(text)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], "First query")
        self.assertEqual(result[1], "Second query")
        self.assertEqual(result[2], "Third query")

    def test_parse_response_bullet_list(self):
        """_parse_response handles bullet list format correctly."""
        text = """- What is AI?
- Define machine learning
- Explain deep learning"""
        result = self.decomposer._parse_response(text)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], "What is AI?")
        self.assertEqual(result[1], "Define machine learning")
        self.assertEqual(result[2], "Explain deep learning")

    def test_parse_response_asterisk_bullet_list(self):
        """_parse_response handles asterisk bullet format correctly."""
        text = """* Query one
* Query two
* Query three"""
        result = self.decomposer._parse_response(text)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], "Query one")
        self.assertEqual(result[1], "Query two")
        self.assertEqual(result[2], "Query three")

    def test_parse_response_empty_input(self):
        """_parse_response handles empty input by returning list with empty string."""
        text = ""
        result = self.decomposer._parse_response(text)
        self.assertEqual(result, [""])

    def test_parse_response_whitespace_only(self):
        """_parse_response handles whitespace-only input by returning list with empty string."""
        text = "   \n  \t  \n  "
        result = self.decomposer._parse_response(text)
        self.assertEqual(result, [""])

    def test_parse_response_mixed_format(self):
        """_parse_response handles mixed numbered and bullet formats."""
        text = """1. First item
- Second item
3) Third item"""
        result = self.decomposer._parse_response(text)
        self.assertEqual(len(result), 3)
        self.assertIn("First item", result[0])
        self.assertIn("Second item", result[1])
        self.assertIn("Third item", result[2])

    def test_parse_response_with_blank_lines(self):
        """_parse_response ignores blank lines between items."""
        text = """1. First query

2. Second query

3. Third query"""
        result = self.decomposer._parse_response(text)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], "First query")

    def test_decompose_fallback_when_requests_missing(self):
        """decompose returns [query] fallback when requests module is None."""
        self.decomposer.requests = None
        result = self.decomposer.decompose("test query")
        self.assertEqual(result, ["test query"])

    def test_decompose_fallback_on_non_200_status(self):
        """decompose returns [query] fallback when HTTP status is not 200."""
        with patch.object(self.decomposer, 'requests') as mock_requests:
            mock_response = mock_requests.post.return_value
            mock_response.status_code = 404
            
            result = self.decomposer.decompose("test query")
            self.assertEqual(result, ["test query"])

    def test_decompose_fallback_on_500_status(self):
        """decompose returns [query] fallback when HTTP status is 500."""
        with patch.object(self.decomposer, 'requests') as mock_requests:
            mock_response = mock_requests.post.return_value
            mock_response.status_code = 500
            
            result = self.decomposer.decompose("test query")
            self.assertEqual(result, ["test query"])

    def test_decompose_returns_parsed_subqueries_on_success(self):
        """decompose returns parsed subqueries on successful 200 response."""
        with patch.object(self.decomposer, 'requests') as mock_requests:
            mock_response = mock_requests.post.return_value
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "response": "1. What is AI?\n2. How does ML work?"
            }
            
            result = self.decomposer.decompose("Explain AI and ML")
            self.assertEqual(len(result), 2)
            self.assertEqual(result[0], "What is AI?")
            self.assertEqual(result[1], "How does ML work?")

    def test_decompose_returns_original_when_single_subquery(self):
        """decompose returns [query] when LLM returns single item (atomic query)."""
        with patch.object(self.decomposer, 'requests') as mock_requests:
            mock_response = mock_requests.post.return_value
            mock_response.status_code = 200
            mock_response.json.return_value = {
                "response": "What is quantum computing?"
            }
            
            result = self.decomposer.decompose("What is quantum computing?")
            self.assertEqual(result, ["What is quantum computing?"])

    def test_decompose_fallback_on_timeout_exception(self):
        """decompose returns [query] fallback on Timeout exception."""
        with patch.object(self.decomposer, 'requests') as mock_requests:
            from recursive_decomposer import Timeout as TimeoutErr
            mock_requests.post.side_effect = TimeoutErr("Request timed out")
            
            result = self.decomposer.decompose("test query")
            self.assertEqual(result, ["test query"])

    def test_decompose_fallback_on_request_exception(self):
        """decompose returns [query] fallback on RequestException."""
        with patch.object(self.decomposer, 'requests') as mock_requests:
            from recursive_decomposer import RequestException as ReqErr
            mock_requests.post.side_effect = ReqErr("Network error")
            
            result = self.decomposer.decompose("test query")
            self.assertEqual(result, ["test query"])

    def test_decompose_sends_correct_payload(self):
        """decompose sends correct JSON payload to Ollama API."""
        with patch.object(self.decomposer, 'requests') as mock_requests:
            mock_response = mock_requests.post.return_value
            mock_response.status_code = 200
            mock_response.json.return_value = {"response": "1. Sub-query one"}
            
            self.decomposer.decompose("test query")
            
            # Verify the API was called with correct parameters
            call_args = mock_requests.post.call_args
            self.assertEqual(call_args[0][0], self.decomposer.ollama_url)
            payload = call_args[1]['json']
            self.assertEqual(payload['model'], self.decomposer.model)
            self.assertIn("test query", payload['prompt'])
            self.assertFalse(payload['stream'])


class TestAggregation(unittest.TestCase):
    """Test aggregation strategies directly on crafted leaf nodes."""

    def _make_engine(self, strategy="rrf"):
        """Create a RecursiveRetrievalEngine with a no-op mock for aggregation testing."""
        mock_eng = MockAntigravityEngine()
        decomposer = MockDecomposer()
        with patch("recursive_decomposer.get_logger") as mock_logger:
            mock_logger.return_value.log_event = lambda *a, **kw: None
            engine = RecursiveRetrievalEngine(
                engine=mock_eng,
                decomposer=decomposer,
                aggregation_strategy=strategy,
                max_depth=3,
                top_k=10,
            )
        return engine

    def _make_leaf(self, results, scores=None):
        """Create a leaf DecompositionNode with preset results."""
        node = DecompositionNode(query="leaf", depth=1, parent_id=None)
        node.is_base_case = True
        node.results = results
        node.scores = scores if scores is not None else list(range(len(results), 0, -1))
        return node

    def test_rrf_single_leaf(self):
        """RRF with one leaf gives score = 1/(60+rank) for each doc."""
        engine = self._make_engine("rrf")
        leaf = self._make_leaf([10, 20, 30])

        result_ids, scores = engine.reciprocal_rank_fusion([leaf], k=60)

        self.assertEqual(result_ids, [10, 20, 30])
        self.assertAlmostEqual(scores[0], 1.0 / 60)       # rank 0
        self.assertAlmostEqual(scores[1], 1.0 / 61)       # rank 1
        self.assertAlmostEqual(scores[2], 1.0 / 62)       # rank 2

    def test_rrf_identical_lists(self):
        """Two leaves with identical docs double each RRF score."""
        engine = self._make_engine("rrf")
        leaf_a = self._make_leaf([10, 20, 30])
        leaf_b = self._make_leaf([10, 20, 30])

        result_ids, scores = engine.reciprocal_rank_fusion([leaf_a, leaf_b], k=60)

        self.assertEqual(result_ids[0], 10)
        # Score for doc 10: 1/60 + 1/60 = 2/60
        self.assertAlmostEqual(scores[0], 2.0 / 60)
        self.assertAlmostEqual(scores[1], 2.0 / 61)

    def test_union_disjoint_lists(self):
        """Union of disjoint leaves contains all docs from both."""
        engine = self._make_engine("union")
        leaf_a = self._make_leaf([10, 20, 30])
        leaf_b = self._make_leaf([40, 50, 60])

        result_ids, scores = engine.union_aggregate([leaf_a, leaf_b])

        self.assertEqual(set(result_ids), {10, 20, 30, 40, 50, 60})
        # Best rank for each doc is 0 (position in its own leaf), so score = 1/(1+0) = 1.0
        # for the first-ranked docs, 1/(1+1) = 0.5 for second-ranked, etc.
        # All first-in-their-leaf docs (10 and 40) have rank 0
        self.assertIn(1.0, scores)

    def test_intersection_common_docs(self):
        """Intersection returns only docs present in all leaves."""
        engine = self._make_engine("intersection")
        leaf_a = self._make_leaf([10, 20, 30, 40])
        leaf_b = self._make_leaf([20, 30, 50, 60])

        result_ids, scores = engine.intersection_aggregate([leaf_a, leaf_b])

        # Only docs 20 and 30 appear in both
        self.assertEqual(set(result_ids), {20, 30})

        # Doc 20: rank 1 in leaf_a, rank 0 in leaf_b -> avg_rank = 0.5 -> score = 1/(1+0.5)
        # Doc 30: rank 2 in leaf_a, rank 1 in leaf_b -> avg_rank = 1.5 -> score = 1/(1+1.5)
        # Doc 20 should have higher score and be first
        self.assertEqual(result_ids[0], 20)
        self.assertAlmostEqual(scores[0], 1.0 / 1.5)
        self.assertAlmostEqual(scores[1], 1.0 / 2.5)

    def test_intersection_fallback_to_union(self):
        """When no docs are shared, intersection falls back to union."""
        engine = self._make_engine("intersection")
        leaf_a = self._make_leaf([10, 20])
        leaf_b = self._make_leaf([30, 40])

        result_ids, scores = engine.intersection_aggregate([leaf_a, leaf_b])

        # No overlap, so fallback to union -- all docs present
        self.assertEqual(set(result_ids), {10, 20, 30, 40})
        # Union scoring: best rank for doc 10 is 0 -> score 1.0
        self.assertIn(1.0, scores)


if __name__ == "__main__":
    unittest.main(verbosity=2)
