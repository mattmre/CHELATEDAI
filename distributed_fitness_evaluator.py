"""Pluggable local and mock-storage fitness evaluators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable, List

from fitness_interfaces import FitnessEvaluation


class DistributedFitnessEvaluator(ABC):
    """Abstract batch scorer for ES population members."""

    @abstractmethod
    def batch_evaluate(self, candidates: Iterable[Any]) -> Dict[str, Any]:
        """Evaluate a population and return scores plus backend metadata."""


class LocalFitnessEvaluator(DistributedFitnessEvaluator):
    """Sequential in-process population evaluator."""

    def __init__(self, fitness_fn: Callable[[Any], float]):
        self.fitness_fn = fitness_fn

    def batch_evaluate(self, candidates: Iterable[Any]) -> Dict[str, Any]:
        evaluations: List[FitnessEvaluation] = []
        for index, candidate in enumerate(candidates):
            fitness = float(self.fitness_fn(candidate))
            evaluations.append(FitnessEvaluation(candidate_id=f"candidate_{index}", fitness=fitness))
        best = max(evaluations, key=lambda item: item.fitness, default=None)
        return {
            "backend": "local",
            "evaluations": evaluations,
            "best_candidate_id": best.candidate_id if best else None,
            "best_fitness": best.fitness if best else None,
        }


class MockStorageFitnessEvaluator(DistributedFitnessEvaluator):
    """Population evaluator backed by the computational-storage mock array."""

    def __init__(self, array_simulation, fitness_fn: Callable[[Any], float]):
        self.array_simulation = array_simulation
        self.fitness_fn = fitness_fn

    def batch_evaluate(self, candidates: Iterable[Any]) -> Dict[str, Any]:
        candidate_scores = []
        evaluations: List[FitnessEvaluation] = []
        for index, candidate in enumerate(candidates):
            candidate_id = f"candidate_{index}"
            fitness = float(self.fitness_fn(candidate))
            candidate_scores.append({"candidate_id": candidate_id, "fitness": fitness})
            evaluations.append(FitnessEvaluation(candidate_id=candidate_id, fitness=fitness))
        storage_summary = self.array_simulation.sharded_population_evaluation(candidate_scores)
        return {
            "backend": "mock_storage",
            "evaluations": evaluations,
            **storage_summary,
        }

