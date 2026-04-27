"""Reproducibility metadata helpers for ChelatedAI benchmark and ES runs."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import platform
import subprocess
import sys
from typing import Any, Dict, Iterable, List, Optional


def stable_hash(value: Any, length: int = 12) -> str:
    """Return a deterministic short hash for JSON-serializable metadata."""

    payload = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:length]


def get_git_metadata() -> Dict[str, Any]:
    """Capture commit and dirty-state metadata when git is available."""

    metadata: Dict[str, Any] = {"commit": None, "dirty": None}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--short"],
            stderr=subprocess.DEVNULL,
            text=True,
        )
        metadata["commit"] = commit
        metadata["dirty"] = bool(status.strip())
    except Exception:
        metadata["dirty"] = None
    return metadata


@dataclass
class ReproducibilityContext:
    """Replay metadata for an optimization or benchmark run."""

    optimizer_type: str
    config_hash: str
    seed: Optional[int] = None
    seed_matrix: List[int] = field(default_factory=list)
    git_commit: Optional[str] = None
    git_dirty: Optional[bool] = None
    python_version: str = field(default_factory=lambda: sys.version.split()[0])
    platform: str = field(default_factory=platform.platform)
    device: str = "cpu"
    model_name: Optional[str] = None
    adapter_hash: Optional[str] = None
    dataset_hash: Optional[str] = None
    quantization: Dict[str, Any] = field(default_factory=dict)
    command: Optional[str] = None

    @classmethod
    def create(
        cls,
        optimizer_type: str,
        config: Any,
        seed: Optional[int] = None,
        seed_matrix: Optional[Iterable[int]] = None,
        **kwargs: Any,
    ) -> "ReproducibilityContext":
        git_metadata = get_git_metadata()
        return cls(
            optimizer_type=optimizer_type,
            config_hash=stable_hash(config),
            seed=seed,
            seed_matrix=list(seed_matrix or []),
            git_commit=git_metadata.get("commit"),
            git_dirty=git_metadata.get("dirty"),
            **kwargs,
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class InitialChelatedValues:
    """Frozen baseline retrieval values used before adaptive controls are promoted."""

    dataset_hash: str
    query_set_hash: str
    corpus_size: int
    query_count: int
    ndcg_at_k: float
    mrr: float
    recall_at_k: float
    fitness: float
    k: int = 10
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def build_seed_matrix(base_seed: int, count: int) -> List[int]:
    """Build a deterministic seed matrix for smoke-gate sweeps."""

    if count < 1:
        raise ValueError("count must be >= 1")
    return [int(base_seed) + index * 1009 for index in range(count)]


@dataclass
class MultiSeedGateResult:
    """Summary of seed-matrix smoke validation."""

    passed: bool
    mean_score: float
    std_score: float
    min_score: float
    max_score: float
    tolerance: float
    scores: List[float]

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def evaluate_seed_scores(scores: Iterable[float], tolerance: float) -> MultiSeedGateResult:
    """Pass when score spread stays within the allowed tolerance."""

    if tolerance < 0:
        raise ValueError("tolerance must be non-negative")
    values = [float(score) for score in scores]
    if not values:
        raise ValueError("scores must be non-empty")
    mean = sum(values) / len(values)
    variance = sum((score - mean) ** 2 for score in values) / len(values)
    min_score = min(values)
    max_score = max(values)
    return MultiSeedGateResult(
        passed=(max_score - min_score) <= tolerance,
        mean_score=mean,
        std_score=variance ** 0.5,
        min_score=min_score,
        max_score=max_score,
        tolerance=float(tolerance),
        scores=values,
    )

