"""
Isomer Detection for ChelatedAI

Detects "retrieval isomers" -- queries that produce structurally different
result sets depending on the chelation state (pre/post sedimentation,
standard vs chelated retrieval).

Isomer strength = 1.0 - jaccard (higher = more different result sets).

Provides:
  - Sedimentation isomer detection (before vs after training)
  - Chelation isomer detection (standard vs chelated retrieval)
  - Similar query grouping to find query families with divergent behavior
  - Frequency and distribution statistics
"""

import numpy as np
from chelation_logger import get_logger


class IsomerDetector:
    """
    Detects retrieval isomers -- queries whose result sets change
    significantly across different chelation states.

    An isomer is a query that produces structurally different top-k
    results when using different retrieval modes or adapter states.
    """

    def __init__(self, strength_threshold=0.3, top_k=10):
        """
        Initialize IsomerDetector.

        Args:
            strength_threshold: Minimum isomer strength to classify as isomer (default 0.3)
            top_k: Number of top results to compare (default 10)
        """
        self.logger = get_logger()

        if not 0.0 <= strength_threshold <= 1.0:
            raise ValueError(
                f"strength_threshold must be in [0.0, 1.0], got {strength_threshold}"
            )

        self.strength_threshold = float(strength_threshold)
        self.top_k = int(top_k)

        # Accumulated isomer history
        self._isomer_history = []

    @staticmethod
    def compute_jaccard(set_a, set_b):
        """
        Compute Jaccard similarity between two sets.

        Args:
            set_a: iterable of items
            set_b: iterable of items

        Returns:
            float: Jaccard similarity in [0.0, 1.0]
        """
        s_a = set(set_a)
        s_b = set(set_b)
        if len(s_a) == 0 and len(s_b) == 0:
            return 1.0
        union = s_a | s_b
        if len(union) == 0:
            return 1.0
        return len(s_a & s_b) / len(union)

    @staticmethod
    def compute_isomer_strength(set_a, set_b):
        """
        Compute isomer strength between two result sets.

        Isomer strength = 1.0 - jaccard. Higher values indicate
        more structural difference between the result sets.

        Args:
            set_a: iterable of result IDs
            set_b: iterable of result IDs

        Returns:
            float: isomer strength in [0.0, 1.0]
        """
        jaccard = IsomerDetector.compute_jaccard(set_a, set_b)
        return 1.0 - jaccard

    def detect_sedimentation_isomers(self, pre_results, post_results):
        """
        Detect isomers between pre-sedimentation and post-sedimentation results.

        Args:
            pre_results: dict mapping query -> list of result IDs (before sedimentation)
            post_results: dict mapping query -> list of result IDs (after sedimentation)

        Returns:
            dict with:
                - 'isomers': list of dicts with query, strength, pre_ids, post_ids
                - 'non_isomers': list of dicts for queries below threshold
                - 'isomer_count': number of detected isomers
                - 'total_queries': total number of queries compared
                - 'isomer_ratio': fraction of queries that are isomers
                - 'mean_strength': mean isomer strength across all queries
                - 'max_strength': maximum isomer strength
        """
        return self._detect_isomers(
            pre_results, post_results,
            label_a="pre_sedimentation",
            label_b="post_sedimentation",
            mode="sedimentation"
        )

    def detect_chelation_isomers(self, standard_results, chelated_results):
        """
        Detect isomers between standard and chelated retrieval results.

        Args:
            standard_results: dict mapping query -> list of result IDs (standard retrieval)
            chelated_results: dict mapping query -> list of result IDs (chelated retrieval)

        Returns:
            dict with same structure as detect_sedimentation_isomers
        """
        return self._detect_isomers(
            standard_results, chelated_results,
            label_a="standard",
            label_b="chelated",
            mode="chelation"
        )

    def _detect_isomers(self, results_a, results_b, label_a, label_b, mode):
        """
        Internal isomer detection between two result sets.

        Args:
            results_a: dict mapping query -> list of result IDs
            results_b: dict mapping query -> list of result IDs
            label_a: label for first result set
            label_b: label for second result set
            mode: detection mode string

        Returns:
            dict with isomer analysis results
        """
        # Find common queries
        common_queries = set(results_a.keys()) & set(results_b.keys())

        isomers = []
        non_isomers = []
        strengths = []

        for query in sorted(common_queries):
            ids_a = list(results_a[query])[:self.top_k]
            ids_b = list(results_b[query])[:self.top_k]

            strength = self.compute_isomer_strength(ids_a, ids_b)
            strengths.append(strength)

            entry = {
                "query": query,
                "strength": float(strength),
                f"{label_a}_ids": ids_a,
                f"{label_b}_ids": ids_b,
                "jaccard": 1.0 - float(strength),
            }

            if strength >= self.strength_threshold:
                isomers.append(entry)
            else:
                non_isomers.append(entry)

        total = len(common_queries)
        isomer_count = len(isomers)

        result = {
            "isomers": isomers,
            "non_isomers": non_isomers,
            "isomer_count": isomer_count,
            "total_queries": total,
            "isomer_ratio": isomer_count / max(total, 1),
            "mean_strength": float(np.mean(strengths)) if strengths else 0.0,
            "max_strength": float(np.max(strengths)) if strengths else 0.0,
            "mode": mode,
        }

        # Record to history
        self._isomer_history.append({
            "mode": mode,
            "isomer_count": isomer_count,
            "total_queries": total,
            "mean_strength": result["mean_strength"],
        })

        self.logger.log_event(
            "isomer_detection",
            f"Detected {isomer_count}/{total} isomers (mode={mode}, "
            f"mean_strength={result['mean_strength']:.3f})"
        )

        return result

    def find_similar_query_isomers(self, queries, results_map,
                                    similarity_threshold=0.5):
        """
        Find groups of similar queries that exhibit different isomer behavior.

        Compares all pairs of queries. Two queries are "similar" if their
        result set overlap (Jaccard) exceeds similarity_threshold. Among
        similar query pairs, identifies those with divergent isomer patterns.

        Args:
            queries: list of query strings
            results_map: dict mapping query -> list of result IDs
            similarity_threshold: minimum Jaccard for queries to be considered similar

        Returns:
            dict with:
                - 'similar_pairs': list of (query_a, query_b, jaccard) tuples
                - 'pair_count': number of similar query pairs found
        """
        similar_pairs = []

        query_list = [q for q in queries if q in results_map]

        for i in range(len(query_list)):
            for j in range(i + 1, len(query_list)):
                q_a = query_list[i]
                q_b = query_list[j]
                ids_a = list(results_map[q_a])[:self.top_k]
                ids_b = list(results_map[q_b])[:self.top_k]

                jaccard = self.compute_jaccard(ids_a, ids_b)

                if jaccard >= similarity_threshold:
                    similar_pairs.append({
                        "query_a": q_a,
                        "query_b": q_b,
                        "jaccard": float(jaccard),
                    })

        return {
            "similar_pairs": similar_pairs,
            "pair_count": len(similar_pairs),
        }

    def get_strength_distribution(self, strengths):
        """
        Compute distribution statistics for isomer strengths.

        Args:
            strengths: list of float isomer strength values

        Returns:
            dict with distribution statistics
        """
        if not strengths:
            return {
                "count": 0,
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "median": 0.0,
                "quartile_25": 0.0,
                "quartile_75": 0.0,
            }

        arr = np.array(strengths, dtype=float)
        return {
            "count": len(arr),
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "min": float(np.min(arr)),
            "max": float(np.max(arr)),
            "median": float(np.median(arr)),
            "quartile_25": float(np.percentile(arr, 25)),
            "quartile_75": float(np.percentile(arr, 75)),
        }

    def get_isomer_report(self):
        """
        Get summary report of all isomer detection history.

        Returns:
            dict with detection history summary
        """
        if not self._isomer_history:
            return {
                "total_detections": 0,
                "history": [],
                "cumulative_mean_strength": 0.0,
            }

        all_strengths = [h["mean_strength"] for h in self._isomer_history]

        return {
            "total_detections": len(self._isomer_history),
            "history": list(self._isomer_history),
            "cumulative_mean_strength": float(np.mean(all_strengths)),
        }

    def reset(self):
        """Reset all isomer detection history."""
        self._isomer_history = []
