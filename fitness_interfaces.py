"""Shared fitness interfaces for ChelatedAI optimization paths."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List


@dataclass
class FitnessEvaluation:
    """Structured result for candidate fitness scoring."""

    candidate_id: str
    fitness: float
    metrics: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class FitnessFunctionInterface(ABC):
    """Abstract scoring surface for ES, online updates, and evaluators."""

    @abstractmethod
    def evaluate_candidate(self, candidate: Any, candidate_id: str = "candidate") -> FitnessEvaluation:
        """Evaluate one candidate and return structured fitness."""

    def batch_evaluate(self, candidates: Iterable[Any]) -> List[FitnessEvaluation]:
        """Evaluate candidates sequentially using the single-candidate scorer."""

        return [
            self.evaluate_candidate(candidate, candidate_id=f"candidate_{index}")
            for index, candidate in enumerate(candidates)
        ]


class CallableFitness(FitnessFunctionInterface):
    """Adapter around a plain callable returning a scalar fitness."""

    def __init__(self, fitness_fn):
        self.fitness_fn = fitness_fn

    def evaluate_candidate(self, candidate: Any, candidate_id: str = "candidate") -> FitnessEvaluation:
        return FitnessEvaluation(candidate_id=candidate_id, fitness=float(self.fitness_fn(candidate)))

