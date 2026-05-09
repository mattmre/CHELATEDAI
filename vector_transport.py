"""
Inference-time vector transport for the TTS pipeline.

VectorTransport moves an embedding vector along a path toward a target centroid using
linear interpolation (default) or cosine-space interpolation. The transport weight t
controls how far along the path to move: t=0 means no change, t=1 means full relocation
to target. Dynamic mode computes t from the distance to the nearest target centroid.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional

import numpy as np

from chelation_logger import get_logger


class TransportMode(str, Enum):
    LINEAR = "linear"
    COSINE = "cosine"
    ADAPTIVE = "adaptive"


@dataclass
class TransportConfig:
    mode: TransportMode = TransportMode.LINEAR
    default_weight: float = 0.2
    max_weight: float = 0.5
    min_similarity_for_transport: float = 0.85
    enabled: bool = True


@dataclass
class TransportTarget:
    target_id: str
    centroid: np.ndarray
    description: str = ""
    support: int = 0


@dataclass
class TransportResult:
    original: np.ndarray
    transported: np.ndarray
    target_id: Optional[str]
    weight_used: float
    cosine_similarity_before: float
    cosine_similarity_after: float
    mode_used: str
    was_transported: bool


class VectorTransport:
    def __init__(self, config: TransportConfig) -> None:
        self._config = config
        self._targets: Dict[str, TransportTarget] = {}
        self.logger = get_logger()

    def add_target(self, target: TransportTarget) -> None:
        """Register a transport target centroid."""
        self._targets[target.target_id] = target

    def register_target(self, target_id: str, centroid: np.ndarray, label: str = "") -> None:
        """Register a target centroid using flat parameters (convenience API)."""
        self._targets[target_id] = TransportTarget(
            target_id=target_id,
            centroid=np.array(centroid, dtype=float),
            description=label,
        )

    def target_count(self) -> int:
        """Return the number of registered transport targets."""
        return len(self._targets)

    def clear_targets(self) -> None:
        """Remove all registered transport targets."""
        self._targets.clear()

    def transport(
        self,
        v: np.ndarray,
        target_id: Optional[str] = None,
        weight: Optional[float] = None,
    ) -> TransportResult:
        """Transport v toward the nearest (or specified) target centroid.

        If target_id is None, find the nearest target by cosine similarity.
        If no targets registered, return passthrough. Applies min_similarity_for_transport
        check: if v is already close to target (sim >= min_similarity), skip transport.
        """
        v = np.array(v, dtype=float)
        mode_str = self._config.mode.value

        def passthrough(tid: Optional[str] = None) -> TransportResult:
            return TransportResult(
                original=v.copy(),
                transported=v.copy(),
                target_id=tid,
                weight_used=0.0,
                cosine_similarity_before=1.0,
                cosine_similarity_after=1.0,
                mode_used=mode_str,
                was_transported=False,
            )

        if not self._config.enabled or not self._targets:
            return passthrough()

        # Resolve target
        if target_id is not None:
            if target_id not in self._targets:
                return passthrough(target_id)
            target = self._targets[target_id]
        else:
            # Nearest by highest cosine similarity
            best_sim = -2.0
            target = None
            for t in self._targets.values():
                sim = self._cosine_sim(v, t.centroid)
                if sim > best_sim:
                    best_sim = sim
                    target = t
            if target is None:
                return passthrough()

        sim_before = self._cosine_sim(v, target.centroid)

        # Skip if already close enough
        if sim_before >= self._config.min_similarity_for_transport:
            return TransportResult(
                original=v.copy(),
                transported=v.copy(),
                target_id=target.target_id,
                weight_used=0.0,
                cosine_similarity_before=sim_before,
                cosine_similarity_after=sim_before,
                mode_used=mode_str,
                was_transported=False,
            )

        # Determine weight
        if weight is not None:
            t = float(weight)
        elif self._config.mode == TransportMode.ADAPTIVE:
            t = 1.0 - sim_before  # farther → stronger pull
        else:
            t = self._config.default_weight

        t = min(t, self._config.max_weight)

        # Apply interpolation
        if self._config.mode == TransportMode.COSINE:
            transported = self._slerp(v, target.centroid, t)
        else:  # LINEAR or ADAPTIVE
            transported = (1.0 - t) * v + t * target.centroid

        sim_after = self._cosine_sim(transported, target.centroid)

        return TransportResult(
            original=v.copy(),
            transported=transported,
            target_id=target.target_id,
            weight_used=t,
            cosine_similarity_before=sim_before,
            cosine_similarity_after=sim_after,
            mode_used=mode_str,
            was_transported=True,
        )

    def _cosine_sim(self, a: np.ndarray, b: np.ndarray) -> float:
        na = float(np.linalg.norm(a))
        nb = float(np.linalg.norm(b))
        if na < 1e-12 or nb < 1e-12:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def _slerp(self, v: np.ndarray, v_target: np.ndarray, t: float) -> np.ndarray:
        """Spherical linear interpolation between normalized vectors."""
        vn = v / (float(np.linalg.norm(v)) + 1e-12)
        vtn = v_target / (float(np.linalg.norm(v_target)) + 1e-12)
        dot = float(np.clip(np.dot(vn, vtn), -1.0, 1.0))
        theta = float(np.arccos(dot))
        if abs(theta) < 1e-6:
            return (1.0 - t) * vn + t * vtn
        sin_theta = float(np.sin(theta))
        return (np.sin((1.0 - t) * theta) / sin_theta) * vn + (
            np.sin(t * theta) / sin_theta
        ) * vtn

    def save(self, path: str) -> None:
        """Persist transport state to JSON."""
        targets_data = {
            k: {
                "target_id": v.target_id,
                "centroid": v.centroid.tolist(),
                "description": v.description,
                "support": v.support,
            }
            for k, v in self._targets.items()
        }
        payload = {
            "config": {
                "mode": self._config.mode.value,
                "default_weight": self._config.default_weight,
                "max_weight": self._config.max_weight,
                "min_similarity_for_transport": self._config.min_similarity_for_transport,
                "enabled": self._config.enabled,
            },
            "targets": targets_data,
        }
        with open(path, "w") as fh:
            json.dump(payload, fh)

    @classmethod
    def load(cls, path: str) -> "VectorTransport":
        """Load transport state from JSON."""
        with open(path) as fh:
            data = json.load(fh)
        cd = data["config"]
        config = TransportConfig(
            mode=TransportMode(cd["mode"]),
            default_weight=cd["default_weight"],
            max_weight=cd["max_weight"],
            min_similarity_for_transport=cd["min_similarity_for_transport"],
            enabled=cd["enabled"],
        )
        transport = cls(config)
        for k, v in data["targets"].items():
            transport._targets[k] = TransportTarget(
                target_id=v["target_id"],
                centroid=np.array(v["centroid"]),
                description=v["description"],
                support=v["support"],
            )
        return transport
