"""
Inference-time vector translation for the TTS (Translation-Transport-Steering) pipeline.

VectorTranslator applies an additive offset to an embedding vector without requiring
backpropagation. Two modes:
  - Learned offset: a stored numpy array Δt applied uniformly (default)
  - Dynamic cluster override: per-cluster offsets keyed by cluster_id, used when
    cluster assignment confidence >= cluster_confidence_threshold

The learned offset is computed from Phase C evaluation data using the geometric
direction between the baseline embedding centroid and the best-candidate centroid.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

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

    @property
    def _offset(self) -> Optional[np.ndarray]:
        """Alias for _learned_offset."""
        return self._learned_offset

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
        cls, results_path: str, config: Optional[TranslationConfig] = None
    ) -> "VectorTranslator":
        """Build a VectorTranslator from Phase C evaluation results JSON.

        Prefers embedding centroid geometry when available:
          offset = normalize(best_centroid - baseline_centroid) * 0.05

        Falls back to passthrough if centroid data is absent (better than a
        semantically meaningless uniform shift based on NDCG scalars).
        """
        try:
            data = json.loads(Path(results_path).read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            dim = config.offset_dim if config is not None else 384
            return cls(config if config is not None else TranslationConfig(offset_dim=dim))

        per_query = data.get("per_query_results", [])

        # Collect per-candidate centroids; keep last seen for each candidate_id
        seen_candidates: Dict[str, Tuple[float, np.ndarray]] = {}
        for row in per_query:
            cid = row.get("candidate_id")
            ndcg = float(row.get("ndcg_at_10", 0.0) or 0.0)
            centroid_list = row.get("embedding_centroid")
            if centroid_list is not None and cid is not None:
                seen_candidates[cid] = (ndcg, np.array(centroid_list, dtype=float))

        baseline_centroid: Optional[np.ndarray] = None
        best_centroid: Optional[np.ndarray] = None
        best_ndcg = -1.0

        if "baseline" in seen_candidates:
            baseline_centroid = seen_candidates["baseline"][1]

        for cid, (ndcg, centroid) in seen_candidates.items():
            if cid != "baseline" and ndcg > best_ndcg:
                best_ndcg = ndcg
                best_centroid = centroid

        if baseline_centroid is not None and best_centroid is not None:
            raw_offset = best_centroid - baseline_centroid
            norm = float(np.linalg.norm(raw_offset))
            if norm > 1e-8:
                offset = raw_offset / norm * 0.05
            else:
                offset = np.zeros_like(baseline_centroid)
            dim = len(offset)
        else:
            all_centroids = [c for _, c in seen_candidates.values()]
            dim = (
                len(all_centroids[0])
                if all_centroids
                else (config.offset_dim if config is not None else 384)
            )
            offset = np.zeros(dim)

        if config is None:
            config = TranslationConfig(offset_dim=dim)
        translator = cls(config)
        if float(np.linalg.norm(offset)) > 1e-8:
            translator.set_learned_offset(offset)
        return translator

    @classmethod
    def from_centroids(
        cls,
        baseline_centroid: np.ndarray,
        best_centroid: np.ndarray,
        config: Optional[TranslationConfig] = None,
        step_size: float = 0.05,
    ) -> "VectorTranslator":
        """Build translator from explicit baseline and best embedding centroids.

        offset = normalize(best - baseline) * step_size
        """
        raw_offset = np.array(best_centroid, dtype=float) - np.array(
            baseline_centroid, dtype=float
        )
        norm = float(np.linalg.norm(raw_offset))
        if norm > 1e-8:
            offset = raw_offset / norm * step_size
        else:
            offset = np.zeros_like(raw_offset)
        dim = len(offset)
        if config is None:
            config = TranslationConfig(offset_dim=dim)
        translator = cls(config)
        if float(np.linalg.norm(offset)) > 1e-8:
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
