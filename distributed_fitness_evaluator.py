"""Pluggable local and mock-storage fitness evaluators."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Iterable

from chelation_logger import get_logger
from fitness_interfaces import FitnessEvaluation, FitnessFunctionInterface


def _evaluation_to_dict(evaluation: FitnessEvaluation) -> Dict[str, Any]:
    return {
        "candidate_id": evaluation.candidate_id,
        "fitness": evaluation.fitness,
        "metrics": evaluation.metrics,
        "metadata": evaluation.metadata,
    }


class DistributedFitnessEvaluator(FitnessFunctionInterface, ABC):
    """Abstract batch scorer for ES population members."""

    def __init__(self, logger=None):
        self.logger = logger or get_logger()

    @abstractmethod
    def backend_name(self) -> str:
        """Return the evaluator backend identifier."""

    def evaluate_population(self, candidates: Iterable[Any]) -> Dict[str, Any]:
        """Evaluate a population and return scores plus backend metadata."""

        evaluations = self.batch_evaluate(candidates)
        best = max(evaluations, key=lambda item: item.fitness, default=None)
        result = {
            "backend": self.backend_name(),
            "evaluations": [_evaluation_to_dict(evaluation) for evaluation in evaluations],
            "best_candidate_id": best.candidate_id if best else None,
            "best_fitness": best.fitness if best else None,
        }
        self.logger.log_event(
            "distributed_fitness_evaluation",
            "Evaluated candidate population",
            backend=result["backend"],
            population_size=len(evaluations),
            best_candidate_id=result["best_candidate_id"],
            best_fitness=result["best_fitness"],
            level="DEBUG",
        )
        return result


class LocalFitnessEvaluator(DistributedFitnessEvaluator):
    """Sequential in-process population evaluator."""

    def __init__(self, fitness_fn: Callable[[Any], float], logger=None):
        super().__init__(logger=logger)
        self.fitness_fn = fitness_fn

    def backend_name(self) -> str:
        return "local"

    def evaluate_candidate(self, candidate: Any, candidate_id: str = "candidate") -> FitnessEvaluation:
        return FitnessEvaluation(candidate_id=candidate_id, fitness=float(self.fitness_fn(candidate)))


class MockStorageFitnessEvaluator(DistributedFitnessEvaluator):
    """Population evaluator backed by the computational-storage mock array."""

    def __init__(self, array_simulation, fitness_fn: Callable[[Any], float], logger=None):
        super().__init__(logger=logger)
        self.array_simulation = array_simulation
        self.fitness_fn = fitness_fn

    def backend_name(self) -> str:
        return "mock_storage"

    def evaluate_candidate(self, candidate: Any, candidate_id: str = "candidate") -> FitnessEvaluation:
        return FitnessEvaluation(candidate_id=candidate_id, fitness=float(self.fitness_fn(candidate)))

    def evaluate_population(self, candidates: Iterable[Any]) -> Dict[str, Any]:
        evaluations = self.batch_evaluate(candidates)
        candidate_scores = [
            {"candidate_id": evaluation.candidate_id, "fitness": evaluation.fitness}
            for evaluation in evaluations
        ]
        storage_summary = self.array_simulation.sharded_population_evaluation(candidate_scores)
        result = {
            "backend": self.backend_name(),
            "evaluations": [_evaluation_to_dict(evaluation) for evaluation in evaluations],
            **storage_summary,
        }
        self.logger.log_event(
            "distributed_fitness_evaluation",
            "Evaluated population on mock storage",
            backend=result["backend"],
            population_size=len(evaluations),
            best_candidate_id=result.get("best_candidate_id"),
            best_fitness=result.get("best_fitness"),
            storage_latency_ms=result.get("storage_latency_ms"),
            level="DEBUG",
        )
        return result

