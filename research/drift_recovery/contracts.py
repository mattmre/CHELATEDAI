"""Typed contracts shared by drift-recovery experiment components."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class RetrievalCase:
    dataset: str
    corpus_ids: Sequence[str]
    query_ids: Sequence[str]
    qrels: Mapping[str, Mapping[str, float]]
    k: int = 10


@dataclass(frozen=True)
class ExperimentSplit:
    anchor_ids: Sequence[str]
    eval_ids: Sequence[str]
    seed: int
    anchor_fraction: float

    def __post_init__(self) -> None:
        overlap = set(self.anchor_ids).intersection(self.eval_ids)
        if overlap:
            raise ValueError(f"anchor/eval leakage: {sorted(overlap)[:5]}")


@dataclass(frozen=True)
class AnchorPairs:
    source: np.ndarray
    target: np.ndarray
    pair_ids: Sequence[str]
    supervision_type: str

    def __post_init__(self) -> None:
        if self.source.shape != self.target.shape:
            raise ValueError("anchor source and target shapes differ")
        if self.source.shape[0] != len(self.pair_ids):
            raise ValueError("pair_ids length does not match anchor rows")


@dataclass(frozen=True)
class PerQueryRecord:
    query_id: str
    floor_ndcg: float
    oracle_ndcg: float
    method_ndcg: Mapping[str, float]


@dataclass(frozen=True)
class RunManifest:
    repo_sha: str
    harness_sha: str
    protocol_sha: str
    model_revisions: Mapping[str, str]
    seeds: Mapping[str, Any]
    command: str
    environment: Mapping[str, str]
    artifacts: Mapping[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class QueryAdapter(ABC):
    """A method that learns a map over query or document vectors."""

    @abstractmethod
    def fit(self, source: np.ndarray, target: np.ndarray, **kwargs: Any) -> "QueryAdapter":
        raise NotImplementedError

    @abstractmethod
    def transform(self, vectors: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class ScoreTransform(ABC):
    """A method that learns and applies a transformation in score space."""

    @abstractmethod
    def fit(self, scores: np.ndarray, **kwargs: Any) -> "ScoreTransform":
        raise NotImplementedError

    @abstractmethod
    def rank(self, scores: np.ndarray, k: int) -> np.ndarray:
        raise NotImplementedError


@dataclass
class MethodRunner:
    """Uniformly run a vector adapter or a score transform, never both."""

    query_adapter: Optional[QueryAdapter] = None
    score_transform: Optional[ScoreTransform] = None

    def __post_init__(self) -> None:
        if (self.query_adapter is None) == (self.score_transform is None):
            raise ValueError("MethodRunner requires exactly one method interface")

    def fit(self, source: np.ndarray, target: Optional[np.ndarray] = None, **kwargs: Any) -> None:
        if self.query_adapter is not None:
            if target is None:
                raise ValueError("vector adapters require targets")
            self.query_adapter.fit(source, target, **kwargs)
        else:
            assert self.score_transform is not None
            self.score_transform.fit(source, **kwargs)

    def output(self, values: np.ndarray, k: int = 10) -> np.ndarray:
        if self.query_adapter is not None:
            return self.query_adapter.transform(values)
        assert self.score_transform is not None
        return self.score_transform.rank(values, k=k)
