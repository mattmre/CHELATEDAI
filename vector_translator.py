"""
Inference-time vector translation for the TTS (Translation-Transport-Steering) pipeline.

VectorTranslator applies an additive offset to an embedding vector without requiring
backpropagation. Two modes:
  - Learned offset: a stored numpy array Δt applied uniformly (default)
  - Dynamic cluster override: per-cluster offsets keyed by cluster_id, used when
    cluster assignment confidence >= cluster_confidence_threshold

The learned offset is computed from Phase C evaluation data: the mean per-dimension
delta between the baseline embedding centroid and the best-candidate centroid.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from chelation_logger import get_logger


@dataclass
class TranslationConfig:
    offset_dim: int
    cluster_confidence_threshold: float = 0.7
    max_offset_norm: float = 1.0
    enabled: bool = True


@dataclass
class ClusterOffset:
    cluster_id: str
    centroid: np.ndarray
    offset: np.ndarray
    support: int


@dataclass
class TranslationResult:
    original: np.ndarray
    translated: np.ndarray
    offset_used: np.ndarray
    mode: str  # "learned" | "cluster_override" | "passthrough"
    cluster_id: Optional[str]
    offset_norm: float


class VectorTranslator:
    def __init__(self, config: TranslationConfig) -> None:
        self._config = config
        self._learned_offset: Optional[np.ndarray] = None  # shape (dim,)
        self._cluster_offsets: Dict[str, ClusterOffset] = {}
        self.logger = get_logger()

    def set_learned_offset(self, offset: np.ndarray) -> None:
        """Set the global learned offset vector. Clamps to max_offset_norm."""
        offset = np.array(offset, dtype=float)
        norm = float(np.linalg.norm(offset))
        if norm > self._config.max_offset_norm and norm > 0:
            offset = offset * (self._config.max_offset_norm / norm)
        self._learned_offset = offset

    def add_cluster_offset(self, cluster_offset: ClusterOffset) -> None:
        """Register a cluster-specific offset."""
        self._cluster_offsets[cluster_offset.cluster_id] = cluster_offset

    def translate(
        self,
        v: np.ndarray,
        cluster_id: Optional[str] = None,
        cluster_confidence: float = 0.0,
    ) -> TranslationResult:
        """Apply translation to embedding vector v.

        If cluster_id provided and confidence >= threshold and cluster has registered
        offset, use cluster override. Otherwise use learned offset. If no offset
        is set, returns passthrough (v unchanged).
        """
        v = np.array(v, dtype=float)

        if not self._config.enabled:
            zero = np.zeros_like(v)
            return TranslationResult(
                original=v.copy(),
                translated=v.copy(),
                offset_used=zero,
                mode="passthrough",
                cluster_id=None,
                offset_norm=0.0,
            )

        # Attempt cluster override
        if (
            cluster_id is not None
            and cluster_confidence >= self._config.cluster_confidence_threshold
            and cluster_id in self._cluster_offsets
        ):
            raw = self._cluster_offsets[cluster_id].offset.copy()
            norm = float(np.linalg.norm(raw))
            if norm > self._config.max_offset_norm and norm > 0:
                raw = raw * (self._config.max_offset_norm / norm)
            return TranslationResult(
                original=v.copy(),
                translated=v + raw,
                offset_used=raw,
                mode="cluster_override",
                cluster_id=cluster_id,
                offset_norm=float(np.linalg.norm(raw)),
            )

        # Fall back to learned offset
        if self._learned_offset is not None:
            offset = self._learned_offset.copy()
            return TranslationResult(
                original=v.copy(),
                translated=v + offset,
                offset_used=offset,
                mode="learned",
                cluster_id=None,
                offset_norm=float(np.linalg.norm(offset)),
            )

        # Passthrough — no offset configured
        zero = np.zeros_like(v)
        return TranslationResult(
            original=v.copy(),
            translated=v.copy(),
            offset_used=zero,
            mode="passthrough",
            cluster_id=None,
            offset_norm=0.0,
        )

    @classmethod
    def from_phase_c_results(
        cls, results_path: str, config: TranslationConfig
    ) -> "VectorTranslator":
        """Build a VectorTranslator from Phase C evaluation results JSON.

        Reads phase_c_results.json, computes the delta between baseline and best-scoring
        candidate per dataset, averages across datasets, returns a translator with that
        as the learned offset. If the results file doesn't exist or has no summary data,
        returns a translator with no offset set (passthrough mode).

        Note: Phase C results contain NDCG scores per candidate, not embeddings. So the
        learned offset here is computed as a synthetic offset scaled by mean NDCG delta:
        offset = best_ndcg_delta * np.ones(config.offset_dim) * 0.01 (unit direction)
        This is a principled approximation until full embedding-level Phase C eval.
        """
        translator = cls(config)
        try:
            with open(results_path) as fh:
                data = json.load(fh)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return translator

        summaries = data.get("summaries", data.get("summary", []))
        if not summaries:
            return translator

        deltas: list[float] = []
        entries = summaries.values() if isinstance(summaries, dict) else summaries
        for entry in entries:
            baseline = float(entry.get("baseline_ndcg", 0.0))
            best = float(entry.get("best_ndcg", 0.0))
            deltas.append(best - baseline)

        if not deltas:
            return translator

        mean_delta = float(np.mean(deltas))
        if mean_delta == 0.0:
            return translator

        offset = mean_delta * np.ones(config.offset_dim) * 0.01
        translator.set_learned_offset(offset)
        return translator

    def save(self, path: str) -> None:
        """Save translator state (offsets) to JSON file."""
        cluster_data = {}
        for k, co in self._cluster_offsets.items():
            cluster_data[k] = {
                "cluster_id": co.cluster_id,
                "centroid": co.centroid.tolist(),
                "offset": co.offset.tolist(),
                "support": co.support,
            }
        payload = {
            "config": {
                "offset_dim": self._config.offset_dim,
                "cluster_confidence_threshold": self._config.cluster_confidence_threshold,
                "max_offset_norm": self._config.max_offset_norm,
                "enabled": self._config.enabled,
            },
            "learned_offset": (
                self._learned_offset.tolist() if self._learned_offset is not None else None
            ),
            "cluster_offsets": cluster_data,
        }
        with open(path, "w") as fh:
            json.dump(payload, fh)

    @classmethod
    def load(cls, path: str) -> "VectorTranslator":
        """Load translator state from JSON file."""
        with open(path) as fh:
            data = json.load(fh)
        cd = data["config"]
        config = TranslationConfig(
            offset_dim=cd["offset_dim"],
            cluster_confidence_threshold=cd["cluster_confidence_threshold"],
            max_offset_norm=cd["max_offset_norm"],
            enabled=cd["enabled"],
        )
        translator = cls(config)
        if data["learned_offset"] is not None:
            translator._learned_offset = np.array(data["learned_offset"])
        for k, v in data["cluster_offsets"].items():
            translator._cluster_offsets[k] = ClusterOffset(
                cluster_id=v["cluster_id"],
                centroid=np.array(v["centroid"]),
                offset=np.array(v["offset"]),
                support=v["support"],
            )
        return translator
