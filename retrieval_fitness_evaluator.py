"""Retrieval-native fitness scoring for ES candidate evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set

import numpy as np

from benchmark_utils import canonicalize_id, mean_reciprocal_rank, ndcg_at_k, recall_at_k
from chelation_logger import get_logger
from fitness_interfaces import FitnessEvaluation, FitnessFunctionInterface


RankingProvider = Callable[[Any, str], Sequence[Any]]


@dataclass
class RetrievalFitnessResult:
    """Aggregate retrieval-fitness metrics for a candidate."""

    candidate_id: str
    fitness: float
    ndcg_at_k: float
    mrr: float
    recall_at_k: float
    evaluated_queries: int
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_fitness_evaluation(self) -> FitnessEvaluation:
        return FitnessEvaluation(
            candidate_id=self.candidate_id,
            fitness=self.fitness,
            metrics={
                "ndcg_at_k": self.ndcg_at_k,
                "mrr": self.mrr,
                "recall_at_k": self.recall_at_k,
                "evaluated_queries": float(self.evaluated_queries),
            },
            metadata=self.metadata,
        )


class RetrievalFitnessEvaluator(FitnessFunctionInterface):
    """Score candidates using actual ranking metrics over query relevance labels."""

    def __init__(
        self,
        qrels: Mapping[Any, Any],
        ranking_provider: Optional[RankingProvider] = None,
        k: int = 10,
        ndcg_weight: float = 0.6,
        mrr_weight: float = 0.2,
        recall_weight: float = 0.2,
        query_ids: Optional[Iterable[Any]] = None,
        logger=None,
    ):
        if k < 1:
            raise ValueError("k must be >= 1")
        if ndcg_weight < 0 or mrr_weight < 0 or recall_weight < 0:
            raise ValueError("metric weights must be non-negative")
        total_weight = ndcg_weight + mrr_weight + recall_weight
        if total_weight <= 0:
            raise ValueError("at least one metric weight must be positive")

        self.qrels = {
            canonicalize_id(query_id): self._normalize_relevance(relevance)
            for query_id, relevance in qrels.items()
        }
        if query_ids is None:
            self.query_ids = sorted(self.qrels.keys())
        else:
            self.query_ids = [canonicalize_id(query_id) for query_id in query_ids if canonicalize_id(query_id) in self.qrels]
        self.ranking_provider = ranking_provider
        self.k = int(k)
        self.ndcg_weight = ndcg_weight / total_weight
        self.mrr_weight = mrr_weight / total_weight
        self.recall_weight = recall_weight / total_weight
        self.logger = logger or get_logger()

    @staticmethod
    def _normalize_relevance(relevance: Any) -> Dict[str, float]:
        if isinstance(relevance, Mapping):
            return {canonicalize_id(doc_id): float(score) for doc_id, score in relevance.items() if float(score) > 0}
        return {canonicalize_id(doc_id): 1.0 for doc_id in relevance}

    def evaluate_rankings(
        self,
        rankings: Mapping[Any, Sequence[Any]],
        candidate_id: str = "candidate",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> RetrievalFitnessResult:
        """Evaluate precomputed rankings keyed by query id."""

        ndcg_scores: List[float] = []
        mrr_scores: List[float] = []
        recall_scores: List[float] = []

        for query_id in self.query_ids:
            relevant = self.qrels.get(query_id, {})
            if not relevant:
                continue
            retrieved = [canonicalize_id(doc_id) for doc_id in rankings.get(query_id, [])]
            relevant_ids: Set[str] = set(relevant.keys())
            relevance_by_rank = [relevant.get(doc_id, 0.0) for doc_id in retrieved[: self.k]]

            ndcg_scores.append(float(ndcg_at_k(relevance_by_rank, self.k)))
            mrr_scores.append(float(mean_reciprocal_rank(retrieved, relevant_ids)))
            recall_scores.append(float(recall_at_k(retrieved, relevant_ids, self.k)))

        evaluated = len(ndcg_scores)
        if evaluated == 0:
            result = RetrievalFitnessResult(
                candidate_id=candidate_id,
                fitness=0.0,
                ndcg_at_k=0.0,
                mrr=0.0,
                recall_at_k=0.0,
                evaluated_queries=0,
                metadata=metadata or {},
            )
            self.logger.log_event(
                "retrieval_fitness_evaluated",
                "No relevant queries evaluated for retrieval fitness",
                candidate_id=candidate_id,
                fitness=result.fitness,
                evaluated_queries=result.evaluated_queries,
                level="DEBUG",
            )
            return result

        ndcg_mean = float(np.mean(ndcg_scores))
        mrr_mean = float(np.mean(mrr_scores))
        recall_mean = float(np.mean(recall_scores))
        fitness = (
            self.ndcg_weight * ndcg_mean
            + self.mrr_weight * mrr_mean
            + self.recall_weight * recall_mean
        )
        result = RetrievalFitnessResult(
            candidate_id=candidate_id,
            fitness=float(fitness),
            ndcg_at_k=ndcg_mean,
            mrr=mrr_mean,
            recall_at_k=recall_mean,
            evaluated_queries=evaluated,
            metadata=metadata or {},
        )
        self.logger.log_event(
            "retrieval_fitness_evaluated",
            "Evaluated retrieval fitness",
            candidate_id=candidate_id,
            fitness=result.fitness,
            ndcg_at_k=result.ndcg_at_k,
            mrr=result.mrr,
            recall_at_k=result.recall_at_k,
            evaluated_queries=result.evaluated_queries,
            level="DEBUG",
        )
        return result

    def evaluate_candidate(self, candidate: Any, candidate_id: str = "candidate") -> FitnessEvaluation:
        if self.ranking_provider is None:
            raise ValueError("ranking_provider is required for evaluate_candidate")
        rankings = {
            query_id: self.ranking_provider(candidate, query_id)
            for query_id in self.query_ids
        }
        return self.evaluate_rankings(rankings, candidate_id=candidate_id).to_fitness_evaluation()

    def batch_evaluate(self, candidates: Iterable[Any]) -> List[FitnessEvaluation]:
        return [
            self.evaluate_candidate(candidate, candidate_id=f"candidate_{index}")
            for index, candidate in enumerate(candidates)
        ]

    def evaluate_engine(
        self,
        engine: Any,
        queries: Mapping[Any, str],
        id_mapper: Callable[[Any, Sequence[Any]], Sequence[Any]],
        candidate_id: str = "engine",
    ) -> RetrievalFitnessResult:
        """Evaluate an AntigravityEngine-like object by running retrieval for each query."""

        rankings = {}
        for query_id in self.query_ids:
            _, pred_ids, _, _ = engine.run_inference(queries[query_id])
            rankings[query_id] = id_mapper(engine, pred_ids[: self.k])
        return self.evaluate_rankings(rankings, candidate_id=candidate_id)


class RetrievalFitnessComposer:
    """Compose retrieval fitness with optional penalty scores."""

    def __init__(self, retrieval_evaluator: RetrievalFitnessEvaluator, penalty_weight: float = 0.0):
        if penalty_weight < 0:
            raise ValueError("penalty_weight must be non-negative")
        self.retrieval_evaluator = retrieval_evaluator
        self.penalty_weight = float(penalty_weight)

    def compose(
        self,
        rankings: Mapping[Any, Sequence[Any]],
        candidate_id: str = "candidate",
        penalty_score: float = 1.0,
    ) -> RetrievalFitnessResult:
        result = self.retrieval_evaluator.evaluate_rankings(rankings, candidate_id=candidate_id)
        bounded_penalty = max(0.0, min(1.0, float(penalty_score)))
        if self.penalty_weight > 0:
            result.fitness = float(result.fitness * (1.0 - self.penalty_weight + self.penalty_weight * bounded_penalty))
            result.metadata["penalty_score"] = bounded_penalty
        return result

