"""
Recursive Decomposition Engine for ChelatedAI

Implements recursive query decomposition with multiple aggregation strategies.
Breaks complex queries into sub-queries, retrieves independently, and fuses
results using Reciprocal Rank Fusion, union, or intersection aggregation.

Also provides hierarchical sedimentation with variance-based clustering
for per-cluster adapter training.
"""

import re
import uuid
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import ChelationConfig
from chelation_logger import get_logger
# F-042: Import HierarchicalSedimentationEngine from dedicated module

# Try to import requests exceptions for type-safe exception handling
try:
    from requests.exceptions import RequestException, Timeout, ConnectionError
    REQUESTS_AVAILABLE = True
except ImportError:
    # Define placeholder exceptions if requests is not installed
    RequestException = Exception
    Timeout = Exception
    ConnectionError = Exception
    REQUESTS_AVAILABLE = False


# =============================================================================
# Data Structures
# =============================================================================

class DecompositionStrategy(Enum):
    """Strategy for deciding when and how to decompose queries."""
    ALWAYS_ATOMIC = "always_atomic"
    LENGTH_BASED = "length_based"
    COMPLEXITY_BASED = "complexity_based"
    LLM_DECIDED = "llm_decided"


@dataclass
class DecompositionNode:
    """A node in the recursive decomposition tree."""
    query: str
    depth: int
    parent_id: Optional[str]
    node_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    children: List['DecompositionNode'] = field(default_factory=list)
    results: List[Any] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    is_base_case: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DecompositionTrace:
    """Complete trace of a recursive decomposition run."""
    root_query: str
    root_node: DecompositionNode
    total_nodes: int = 0
    max_depth_reached: int = 0
    total_retrieval_calls: int = 0
    elapsed_seconds: float = 0.0
    final_results: List[Any] = field(default_factory=list)
    final_scores: List[float] = field(default_factory=list)


# =============================================================================
# Decomposer Interface (Strategy Pattern)
# =============================================================================

class BaseDecomposer(ABC):
    """Abstract base for query decomposition strategies."""

    @abstractmethod
    def decompose(self, query: str, context: Optional[Dict] = None) -> List[str]:
        """Decompose query into sub-queries. Returns [query] if atomic."""
        pass

    @abstractmethod
    def is_base_case(self, query: str, context: Optional[Dict] = None) -> bool:
        """Returns True if query should not be further decomposed."""
        pass


class MockDecomposer(BaseDecomposer):
    """
    Deterministic decomposer that splits on conjunctions.

    Splits on the first found conjunction from a priority list.
    No LLM required -- useful for testing and predictable decomposition.
    """

    CONJUNCTIONS = ['; ', ' and ', ' or ', ' versus ', ' compared to ']

    def __init__(self, max_depth=3, min_length=20):
        self.max_depth = max_depth
        self.min_length = min_length

    def is_base_case(self, query: str, context: Optional[Dict] = None) -> bool:
        """True if query is short or contains no conjunctions."""
        if len(query) < self.min_length:
            return True

        query_lower = query.lower()
        for conj in self.CONJUNCTIONS:
            if conj.lower() in query_lower:
                return False
        return True

    def decompose(self, query: str, context: Optional[Dict] = None) -> List[str]:
        """Split on the first matching conjunction (case-insensitive, preserves text)."""
        query_lower = query.lower()

        for conj in self.CONJUNCTIONS:
            conj_lower = conj.lower()
            idx = query_lower.find(conj_lower)
            if idx != -1:
                left = query[:idx].strip()
                right = query[idx + len(conj):].strip()
                parts = [p for p in [left, right] if p]
                if len(parts) > 1:
                    return parts

        return [query]


class OllamaDecomposer(BaseDecomposer):
    """
    LLM-backed decomposer using Ollama's generate API.

    Falls back to atomic (no decomposition) on any failure.
    
    Security features (F-021):
    - Query sanitization (control chars, whitespace normalization, length cap)
    - Sub-query count limiting
    - Lexical overlap validation between sub-queries and original query
    """

    # F-021: Guardrail constants
    MAX_QUERY_LENGTH = 2000
    MAX_SUBQUERY_COUNT = 8
    MIN_OVERLAP_RATIO = 0.2  # At least 20% token overlap required

    def __init__(self, model="llama3.2", ollama_url=None, timeout=None):
        self.model = model
        self.ollama_url = ollama_url or ChelationConfig.OLLAMA_URL.replace(
            "/api/embeddings", "/api/generate"
        )
        self.timeout = timeout or ChelationConfig.OLLAMA_TIMEOUT
        
        # Validate URL to prevent SSRF attacks
        self._validate_url(self.ollama_url)

        try:
            import requests
            self.requests = requests
        except ImportError:
            self.requests = None
    
    def _validate_url(self, url: str):
        """
        Validate that the ollama_url points to a safe localhost host.
        
        Only allows localhost, 127.0.0.1, and ::1 to prevent SSRF attacks.
        Raises ValueError for invalid or external hosts.
        """
        try:
            from urllib.parse import urlparse
        except ImportError:
            from urlparse import urlparse
        
        parsed = urlparse(url)
        hostname = parsed.hostname
        
        if not hostname:
            raise ValueError(f"Invalid URL: no hostname found in {url}")
        
        # Normalize hostname for comparison
        hostname_lower = hostname.lower()
        
        # Allow only localhost variants
        allowed_hosts = {'localhost', '127.0.0.1', '::1'}
        
        if hostname_lower not in allowed_hosts:
            raise ValueError(
                f"SSRF protection: host '{hostname}' not allowed. "
                f"Only localhost/127.0.0.1/::1 are permitted."
            )

    def is_base_case(self, query: str, context: Optional[Dict] = None) -> bool:
        """Simple length-based check."""
        return len(query) < 30
    
    def _sanitize_query(self, query: str) -> str:
        """
        Sanitize query text to prevent prompt injection (F-021).
        
        - Strip control characters (newlines, tabs, form feeds, etc.)
        - Normalize whitespace (collapse multiple spaces to one)
        - Apply length cap
        
        Returns: sanitized query string
        """
        # Strip control characters (ASCII 0-31 and 127)
        sanitized = ''.join(c if ord(c) >= 32 and ord(c) != 127 else ' ' for c in query)
        
        # Normalize whitespace: collapse multiple spaces/tabs into single space
        sanitized = ' '.join(sanitized.split())
        
        # Apply length cap
        if len(sanitized) > self.MAX_QUERY_LENGTH:
            sanitized = sanitized[:self.MAX_QUERY_LENGTH]
        
        return sanitized
    
    def _compute_token_overlap(self, original: str, sub_query: str) -> float:
        """
        Compute lexical overlap ratio between original query and sub-query (F-021).
        
        Returns: ratio of sub-query tokens that appear in original query (0.0 to 1.0)
        """
        # Tokenize on alphanumeric word boundaries to avoid punctuation mismatch.
        original_tokens = set(re.findall(r"[a-z0-9]+", original.lower()))
        sub_tokens = re.findall(r"[a-z0-9]+", sub_query.lower())
        
        if not sub_tokens:
            return 0.0
        
        # Count how many sub-query tokens appear in original
        overlap_count = sum(1 for token in sub_tokens if token in original_tokens)
        return overlap_count / len(sub_tokens)
    
    def _validate_sub_queries(self, original: str, sub_queries: List[str]) -> List[str]:
        """
        Validate sub-queries are related to original query (F-021).
        
        Filters out sub-queries with insufficient lexical overlap.
        Falls back to [original] if all candidates are removed.
        
        Returns: validated sub-query list
        """
        validated = []
        
        for sq in sub_queries:
            overlap = self._compute_token_overlap(original, sq)
            if overlap >= self.MIN_OVERLAP_RATIO:
                validated.append(sq)
        
        # Fallback: if validation removed everything, return original
        if not validated:
            return [original]
        
        return validated

    def decompose(self, query: str, context: Optional[Dict] = None) -> List[str]:
        """
        Ask Ollama to decompose the query into sub-queries.

        On any failure (connection, timeout, parse), returns [query] unchanged.
        
        Security features (F-021):
        - Sanitizes query before interpolation into prompt
        - Limits returned sub-query count
        - Validates sub-queries are related to original query
        """
        if self.requests is None:
            return [query]
        
        # F-021: Sanitize query before interpolating into prompt
        sanitized_query = self._sanitize_query(query)

        prompt = (
            "Break the following search query into simpler independent sub-queries. "
            "Return ONLY a numbered list of sub-queries, one per line. "
            "If the query is already simple and atomic, return just the original query.\n\n"
            f"Query: {sanitized_query}\n\n"
            "Sub-queries:"
        )

        try:
            response = self.requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                },
                timeout=self.timeout,
            )

            if response.status_code != 200:
                return [query]

            text = response.json().get("response", "")
            sub_queries = self._parse_response(text)
            
            # F-021: Limit sub-query count
            if len(sub_queries) > self.MAX_SUBQUERY_COUNT:
                sub_queries = sub_queries[:self.MAX_SUBQUERY_COUNT]
            
            # F-021: Validate sub-queries are related to original
            if len(sub_queries) > 1:
                validated = self._validate_sub_queries(query, sub_queries)
                return validated
            
            return [query]

        except (RequestException, Timeout, ConnectionError, ValueError, KeyError):
            # HTTP/network errors (RequestException, Timeout, ConnectionError)
            # JSON parsing errors (ValueError)
            # Response dict access errors (KeyError)
            return [query]

    def _parse_response(self, text: str) -> List[str]:
        """Parse numbered list or newline-separated items from LLM response."""
        lines = text.strip().split('\n')
        results = []

        for line in lines:
            line = line.strip()
            if not line:
                continue
            # Strip leading numbering like "1.", "1)", "- "
            cleaned = re.sub(r'^[\d]+[.)]\s*', '', line)
            cleaned = re.sub(r'^[-*]\s*', '', cleaned)
            cleaned = cleaned.strip()
            if cleaned:
                results.append(cleaned)

        return results if results else [text.strip()]


# =============================================================================
# Recursive Retrieval Engine
# =============================================================================

class RecursiveRetrievalEngine:
    """
    Performs recursive query decomposition and multi-strategy result aggregation.

    Takes a complex query, decomposes it into a tree of sub-queries, runs
    retrieval at each leaf, and fuses results using the chosen aggregation
    strategy (RRF, union, or intersection).
    """

    def __init__(self, engine, decomposer, aggregation_strategy="rrf",
                 max_depth=3, top_k=10):
        """
        Args:
            engine: AntigravityEngine instance for retrieval
            decomposer: BaseDecomposer instance for query splitting
            aggregation_strategy: "rrf", "union", or "intersection"
            max_depth: Maximum recursion depth
            top_k: Number of final results to return
        """
        self.engine = engine
        self.decomposer = decomposer
        self.aggregation_strategy = aggregation_strategy
        self.max_depth = max_depth
        self.top_k = top_k
        self.logger = get_logger()

    def run_recursive_inference(self, query: str) -> DecompositionTrace:
        """
        Run full recursive decomposition and retrieval pipeline.

        Args:
            query: The original complex query

        Returns:
            DecompositionTrace with complete execution trace and final results
        """
        root = DecompositionNode(query=query, depth=0, parent_id=None)
        start_time = time.time()

        self._recurse(root)

        result_ids, scores = self._aggregate(root)

        elapsed = time.time() - start_time
        total_nodes = self._count_nodes(root)
        max_depth = self._max_depth(root)
        leaves = self._collect_leaves(root)

        trace = DecompositionTrace(
            root_query=query,
            root_node=root,
            total_nodes=total_nodes,
            max_depth_reached=max_depth,
            total_retrieval_calls=len(leaves),
            elapsed_seconds=elapsed,
            final_results=result_ids,
            final_scores=scores,
        )

        self.logger.log_event(
            "recursive_inference",
            f"Query: '{query[:50]}' | Nodes: {total_nodes} | Depth: {max_depth} | "
            f"Leaves: {len(leaves)} | Results: {len(result_ids)} | {elapsed:.3f}s",
            total_nodes=total_nodes,
            max_depth_reached=max_depth,
            total_retrieval_calls=len(leaves),
            elapsed_seconds=elapsed,
            num_results=len(result_ids),
        )

        return trace

    def _recurse(self, node: DecompositionNode):
        """Recursively decompose and retrieve."""
        if node.depth >= self.max_depth or self.decomposer.is_base_case(node.query):
            node.is_base_case = True
            self._retrieve_for_node(node)
            return

        sub_queries = self.decomposer.decompose(node.query)

        if len(sub_queries) <= 1:
            node.is_base_case = True
            self._retrieve_for_node(node)
            return

        # Create child nodes for all sibling sub-queries
        children = []
        for sq in sub_queries:
            child = DecompositionNode(
                query=sq,
                depth=node.depth + 1,
                parent_id=node.node_id,
            )
            children.append(child)
            node.children.append(child)
        
        # Parallelize sibling sub-query recursion (F-029)
        with ThreadPoolExecutor(max_workers=min(len(children), 4)) as executor:
            futures = [executor.submit(self._recurse, child) for child in children]
            for future in as_completed(futures):
                future.result()

    def _retrieve_for_node(self, node: DecompositionNode):
        """Run retrieval on a leaf node via AntigravityEngine."""
        std_top, chel_top, mask, jaccard = self.engine.run_inference(node.query)
        node.results = chel_top
        node.scores = list(range(len(chel_top), 0, -1))

    def _aggregate(self, root: DecompositionNode) -> Tuple[List, List[float]]:
        """Dispatch to the configured aggregation strategy."""
        leaves = self._collect_leaves(root)
        if not leaves:
            return ([], [])

        if self.aggregation_strategy == "rrf":
            return self.reciprocal_rank_fusion(leaves)
        elif self.aggregation_strategy == "intersection":
            return self.intersection_aggregate(leaves)
        else:
            return self.union_aggregate(leaves)

    def _collect_leaves(self, node: DecompositionNode) -> List[DecompositionNode]:
        """Collect all leaf (base case) nodes from the tree."""
        if node.is_base_case:
            return [node]
        leaves = []
        for child in node.children:
            leaves.extend(self._collect_leaves(child))
        return leaves

    def _count_nodes(self, node: DecompositionNode) -> int:
        """Count total nodes in the tree."""
        return 1 + sum(self._count_nodes(c) for c in node.children)

    def _max_depth(self, node: DecompositionNode) -> int:
        """Find the maximum depth reached in the tree."""
        if not node.children:
            return node.depth
        return max(self._max_depth(c) for c in node.children)

    # --- Aggregation Strategies ---

    def reciprocal_rank_fusion(self, leaves: List[DecompositionNode],
                               k: int = 60) -> Tuple[List, List[float]]:
        """
        Reciprocal Rank Fusion across all leaf result lists.

        For each document d, score = sum(1 / (k + rank_i(d))) across all
        leaves where d appears. rank_i is 0-based position.
        """
        doc_scores = defaultdict(float)

        for leaf in leaves:
            for rank, doc_id in enumerate(leaf.results):
                doc_scores[doc_id] += 1.0 / (k + rank)

        sorted_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)
        result_ids = [d[0] for d in sorted_docs[:self.top_k]]
        scores = [d[1] for d in sorted_docs[:self.top_k]]
        return result_ids, scores

    def union_aggregate(self, leaves: List[DecompositionNode]) -> Tuple[List, List[float]]:
        """
        Union aggregation: all unique doc IDs across leaves.

        Score each by best (lowest) rank across any leaf.
        """
        best_rank = {}

        for leaf in leaves:
            for rank, doc_id in enumerate(leaf.results):
                if doc_id not in best_rank or rank < best_rank[doc_id]:
                    best_rank[doc_id] = rank

        scored = [(doc_id, 1.0 / (1 + rank)) for doc_id, rank in best_rank.items()]
        scored.sort(key=lambda x: x[1], reverse=True)

        result_ids = [s[0] for s in scored[:self.top_k]]
        scores = [s[1] for s in scored[:self.top_k]]
        return result_ids, scores

    def intersection_aggregate(self, leaves: List[DecompositionNode]) -> Tuple[List, List[float]]:
        """
        Intersection aggregation: only doc IDs appearing in ALL leaves.

        Score by average rank across leaves. Falls back to union if no
        intersection exists.
        """
        if not leaves:
            return ([], [])

        # Build per-leaf result sets and rank maps
        rank_maps = []
        result_sets = []
        for leaf in leaves:
            rmap = {}
            for rank, doc_id in enumerate(leaf.results):
                rmap[doc_id] = rank
            rank_maps.append(rmap)
            result_sets.append(set(rmap.keys()))

        # Intersection of all result sets
        common = result_sets[0]
        for s in result_sets[1:]:
            common = common.intersection(s)

        if not common:
            return self.union_aggregate(leaves)

        # Score by average rank (lower rank = better, so score = 1/(1+avg_rank))
        scored = []
        for doc_id in common:
            avg_rank = sum(rm[doc_id] for rm in rank_maps) / len(rank_maps)
            scored.append((doc_id, 1.0 / (1 + avg_rank)))

        scored.sort(key=lambda x: x[1], reverse=True)

        result_ids = [s[0] for s in scored[:self.top_k]]
        scores = [s[1] for s in scored[:self.top_k]]
        return result_ids, scores


# =============================================================================
# Hierarchical Sedimentation Engine
# =============================================================================
# F-042: HierarchicalSedimentationEngine relocated to sedimentation.py
# Re-exported here for backward compatibility with existing imports
