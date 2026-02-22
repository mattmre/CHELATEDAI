"""
Per-Embedding Quality Assessment for ChelatedAI

Computes per-document quality scores based on chelation log frequency
with exponential decay, enabling adaptive per-document retrieval thresholds.

Inspired by:
- VectorQ: Vector-level Quality Scores (arXiv:2502.03771)
"""

import numpy as np
from chelation_logger import get_logger


class EmbeddingQualityAssessor:
    """
    Assesses per-document embedding quality based on chelation log frequency.

    Documents that frequently appear in chelation logs (noise clusters)
    receive lower quality scores. Scores decay exponentially so recent
    chelation events weigh more heavily.

    Args:
        decay_factor: Exponential decay for older chelation events (default: 0.95)
        high_threshold: Quality score above which a doc is "high quality" (default: 0.8)
        low_threshold: Quality score below which a doc is "low quality" (default: 0.3)
    """

    def __init__(self, decay_factor=0.95, high_threshold=0.8, low_threshold=0.3):
        if not 0.0 < decay_factor <= 1.0:
            raise ValueError("decay_factor must be in (0, 1]")
        if not 0.0 <= low_threshold < high_threshold <= 1.0:
            raise ValueError("Thresholds must satisfy 0 <= low < high <= 1")

        self.decay_factor = decay_factor
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        self.logger = get_logger()

    def compute_quality_scores(self, chelation_log):
        """
        Compute quality scores for all documents in the chelation log.

        Quality is inversely related to chelation frequency with decay:
        raw_score = sum(decay^i for i in range(num_events))
        quality = 1 / (1 + raw_score)

        Args:
            chelation_log: dict mapping doc_id -> list of noise center vectors

        Returns:
            dict: doc_id -> quality score in [0, 1]
        """
        scores = {}

        for doc_id, events in chelation_log.items():
            n = len(events)
            if n == 0:
                scores[doc_id] = 1.0
                continue

            # Exponential decay weighting: recent events matter more
            # Sum of decay^0 + decay^1 + ... + decay^(n-1) (most recent first)
            raw_score = sum(self.decay_factor ** i for i in range(n))

            # Map to [0, 1] with sigmoid-like transform
            quality = 1.0 / (1.0 + raw_score)
            scores[doc_id] = quality

        return scores

    def classify_document(self, quality_score):
        """
        Classify a document based on its quality score.

        Args:
            quality_score: Quality score in [0, 1]

        Returns:
            str: "high", "medium", or "low"
        """
        if quality_score >= self.high_threshold:
            return "high"
        elif quality_score <= self.low_threshold:
            return "low"
        else:
            return "medium"

    def get_adaptive_threshold(self, quality_score, base_threshold):
        """
        Compute adaptive retrieval threshold for a document based on quality.

        High-quality documents use the standard threshold.
        Low-quality documents use a stricter (higher) threshold.

        Args:
            quality_score: Document's quality score in [0, 1]
            base_threshold: Base chelation threshold

        Returns:
            float: Adjusted threshold for this document
        """
        # Scale threshold inversely with quality
        # Low quality -> higher threshold (more aggressive chelation)
        # High quality -> base threshold (trust the embedding)
        scale = 1.0 + (1.0 - quality_score) * 2.0  # Range: [1.0, 3.0]
        return base_threshold * scale

    def get_quality_report(self, chelation_log):
        """
        Generate a quality report for all documents.

        Args:
            chelation_log: dict mapping doc_id -> list of noise center vectors

        Returns:
            dict with keys: scores, classification_counts, mean_quality,
                           high_quality_count, low_quality_count
        """
        scores = self.compute_quality_scores(chelation_log)

        if not scores:
            return {
                "scores": {},
                "classification_counts": {"high": 0, "medium": 0, "low": 0},
                "mean_quality": 0.0,
                "high_quality_count": 0,
                "low_quality_count": 0
            }

        classifications = {doc_id: self.classify_document(s) for doc_id, s in scores.items()}
        counts = {"high": 0, "medium": 0, "low": 0}
        for c in classifications.values():
            counts[c] += 1

        return {
            "scores": scores,
            "classification_counts": counts,
            "mean_quality": float(np.mean(list(scores.values()))),
            "high_quality_count": counts["high"],
            "low_quality_count": counts["low"]
        }
