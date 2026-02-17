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
import numpy as np
import torch
import torch.optim as optim
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from abc import ABC, abstractmethod
from collections import defaultdict

from config import ChelationConfig
from chelation_logger import ChelationLogger, get_logger
from sedimentation_trainer import compute_homeostatic_target, sync_vectors_to_qdrant

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
    """

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

    def decompose(self, query: str, context: Optional[Dict] = None) -> List[str]:
        """
        Ask Ollama to decompose the query into sub-queries.

        On any failure (connection, timeout, parse), returns [query] unchanged.
        """
        if self.requests is None:
            return [query]

        prompt = (
            "Break the following search query into simpler independent sub-queries. "
            "Return ONLY a numbered list of sub-queries, one per line. "
            "If the query is already simple and atomic, return just the original query.\n\n"
            f"Query: {query}\n\n"
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

            if len(sub_queries) > 1:
                return sub_queries
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

        for sq in sub_queries:
            child = DecompositionNode(
                query=sq,
                depth=node.depth + 1,
                parent_id=node.node_id,
            )
            node.children.append(child)
            self._recurse(child)

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

class HierarchicalSedimentationEngine:
    """
    Extends sedimentation with variance-based clustering for per-cluster
    adapter training. Runs local corrections within clusters, then global
    refinement across all training data.
    """

    def __init__(self, engine):
        """
        Args:
            engine: AntigravityEngine instance
        """
        self.engine = engine
        self.logger = get_logger()

    def run_hierarchical_sedimentation(self, threshold=3, learning_rate=0.001, epochs=10):
        """
        Hierarchical sedimentation cycle with cluster-aware training.

        1. Identify collapse targets from chelation_log
        2. Fetch vectors and cluster via variance-based partitioning
        3. Per-cluster training (local corrections)
        4. Global refinement (cross-cluster coherence)
        5. Save adapter and update Qdrant
        """
        print(f"\n--- HIERARCHICAL SEDIMENTATION (Threshold={threshold}, LR={learning_rate}) ---")

        # Filter for frequent collapsers
        targets = {k: v for k, v in self.engine.chelation_log.items() if len(v) >= threshold}
        print(f"Collapse targets: {len(targets)} candidates.")

        if not targets:
            print("No sedimentation targets. Brain is stable.")
            self.logger.log_event(
                "hierarchical_sedimentation",
                "No targets above threshold",
                threshold=threshold,
            )
            return

        # Fetch vectors for targets from Qdrant
        batch_ids = list(targets.keys())
        chunk_size = 100
        all_vectors = []
        all_ids = []

        for i in range(0, len(batch_ids), chunk_size):
            chunk = batch_ids[i:i + chunk_size]
            points = self.engine.qdrant.retrieve(
                collection_name=self.engine.collection_name,
                ids=chunk,
                with_vectors=True,
            )
            for point in points:
                all_ids.append(point.id)
                all_vectors.append(np.array(point.vector))

        if not all_vectors:
            print("No vectors retrieved. Skipping.")
            return

        vectors_np = np.array(all_vectors)

        # Prepare all training data (input, target pairs)
        training_inputs = []
        training_targets = []
        ordered_ids = []

        for i, doc_id in enumerate(all_ids):
            current_vec = vectors_np[i]
            noise_vectors = targets[doc_id]
            # Use shared helper for homeostatic target computation
            target_vec = compute_homeostatic_target(current_vec, noise_vectors, 0.1)

            training_inputs.append(current_vec)
            training_targets.append(target_vec)
            ordered_ids.append(doc_id)

        if not training_inputs:
            return

        input_tensor = torch.tensor(np.array(training_inputs), dtype=torch.float32)
        target_tensor = torch.tensor(np.array(training_targets), dtype=torch.float32)

        # Cluster via variance-based partitioning
        n_clusters = max(2, len(all_ids) // 5)
        clusters = self._simple_partition(vectors_np, list(range(len(all_ids))), n_clusters)

        cluster_epochs = max(1, epochs // 2)
        global_epochs = epochs - cluster_epochs

        self.logger.log_training_start(
            num_samples=len(training_inputs),
            learning_rate=learning_rate,
            epochs=epochs,
            threshold=threshold,
            n_clusters=len(clusters),
        )

        # --- Phase 1: Per-cluster training (local corrections) ---
        print(f"Phase 1: Per-cluster training ({len(clusters)} clusters, {cluster_epochs} epochs)...")
        optimizer = optim.Adam(self.engine.adapter.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()

        self.engine.adapter.train()

        for cluster_idx, (cluster_vecs, cluster_indices) in enumerate(clusters):
            if len(cluster_indices) == 0:
                continue

            cluster_input = input_tensor[cluster_indices]
            cluster_target = target_tensor[cluster_indices]

            for epoch in range(cluster_epochs):
                optimizer.zero_grad()
                outputs = self.engine.adapter(cluster_input)
                loss = criterion(outputs, cluster_target)
                loss.backward()
                optimizer.step()

            self.logger.log_event(
                "cluster_training",
                f"Cluster {cluster_idx}: {len(cluster_indices)} samples, final loss {loss.item():.6f}",
                level="DEBUG",
                cluster_idx=cluster_idx,
                cluster_size=len(cluster_indices),
                final_loss=loss.item(),
            )

        # --- Phase 2: Global refinement (cross-cluster coherence) ---
        print(f"Phase 2: Global refinement ({global_epochs} epochs, LR={learning_rate * 0.1:.6f})...")
        global_optimizer = optim.Adam(self.engine.adapter.parameters(), lr=learning_rate * 0.1)

        for epoch in range(global_epochs):
            global_optimizer.zero_grad()
            outputs = self.engine.adapter(input_tensor)
            loss = criterion(outputs, target_tensor)
            loss.backward()
            global_optimizer.step()

            if epoch % max(1, global_epochs // 2) == 0:
                self.logger.log_training_epoch(epoch + 1, global_epochs, loss.item())

        self.engine.adapter.eval()
        final_loss = loss.item() if global_epochs > 0 else 0.0

        # Save adapter
        self.engine.adapter.save(self.engine.adapter_path)
        print("Adapter weights saved.")

        # Update Qdrant with adapted vectors
        print("Syncing updated vectors to Qdrant...")

        with torch.no_grad():
            new_vectors_np = self.engine.adapter(input_tensor).numpy()

        # Use shared helper for Qdrant sync
        total_updates, failed_updates = sync_vectors_to_qdrant(
            self.engine.qdrant, self.engine.collection_name, ordered_ids,
            new_vectors_np, chunk_size, self.logger
        )

        self.logger.log_training_complete(
            final_loss=final_loss,
            vectors_updated=total_updates,
            vectors_failed=failed_updates,
        )

        self.engine.chelation_log.clear()
        print(f"Hierarchical sedimentation complete. Updated {total_updates} vectors, {failed_updates} failed.")
        print("--- HIERARCHICAL SLEEP CYCLE COMPLETE ---")

    def _simple_partition(self, vectors: np.ndarray, doc_ids: list,
                          n_clusters: int) -> List[Tuple[np.ndarray, list]]:
        """
        Recursive variance-based splitting (no sklearn dependency).

        Finds the dimension with highest variance, splits on its median,
        and recurses until the requested number of clusters is reached.

        Args:
            vectors: (N, D) array of vectors
            doc_ids: list of indices or IDs corresponding to rows
            n_clusters: target number of clusters

        Returns:
            List of (cluster_vectors, cluster_ids) tuples
        """
        if n_clusters <= 1 or len(vectors) <= 1:
            return [(vectors, doc_ids)]

        # Find dimension with highest variance
        dim = int(np.argmax(np.var(vectors, axis=0)))
        median_val = float(np.median(vectors[:, dim]))

        left_mask = vectors[:, dim] <= median_val
        right_mask = ~left_mask

        # Edge case: all points go to one side
        if not np.any(left_mask) or not np.any(right_mask):
            return [(vectors, doc_ids)]

        left_vecs = vectors[left_mask]
        right_vecs = vectors[right_mask]
        left_ids = [doc_ids[i] for i in range(len(doc_ids)) if left_mask[i]]
        right_ids = [doc_ids[i] for i in range(len(doc_ids)) if right_mask[i]]

        left_k = n_clusters // 2
        right_k = n_clusters - left_k

        left_clusters = self._simple_partition(left_vecs, left_ids, left_k)
        right_clusters = self._simple_partition(right_vecs, right_ids, right_k)

        return left_clusters + right_clusters
