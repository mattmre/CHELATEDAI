"""Compose retrieval, health, quantization, and storage fitness signals."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Mapping, Optional, Sequence

from chelation_logger import get_logger
from quantization_promotion_gate import QuantizationGateResult, QuantizationPromotionGate
from retrieval_fitness_evaluator import RetrievalFitnessEvaluator, RetrievalFitnessResult
from structural_health_score import StructuralHealthResult, StructuralHealthScore


def _bounded_score(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _as_dict(value: Any) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, dict):
        return dict(value)
    return None


@dataclass
class FitnessCompositionResult:
    """Single candidate score with every additive fitness signal preserved."""

    candidate_id: str
    retrieval: RetrievalFitnessResult
    final_fitness: float
    structural_health: Optional[StructuralHealthResult] = None
    structural_health_multiplier: float = 1.0
    quantization_gate: Optional[QuantizationGateResult] = None
    quantized_retrieval: Optional[RetrievalFitnessResult] = None
    storage_metadata: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "candidate_id": self.candidate_id,
            "retrieval_fitness": self.retrieval.fitness,
            "final_fitness": self.final_fitness,
            "retrieval_metrics": self.retrieval.to_fitness_evaluation().metrics,
            "structural_health": None
            if self.structural_health is None
            else {
                "score": self.structural_health.score,
                "components": self.structural_health.components,
                "multiplier": self.structural_health_multiplier,
            },
            "quantization_gate": _as_dict(self.quantization_gate),
            "quantized_retrieval_fitness": None
            if self.quantized_retrieval is None
            else self.quantized_retrieval.fitness,
            "storage_metadata": self.storage_metadata,
            "metadata": self.metadata,
        }


class FitnessCompositionOrchestrator:
    """Reusable fitness pipeline for retrieval-native optimization workflows."""

    def __init__(
        self,
        retrieval_evaluator: RetrievalFitnessEvaluator,
        health_scorer: Optional[StructuralHealthScore] = None,
        health_weight: float = 0.0,
        quantization_gate: Optional[QuantizationPromotionGate] = None,
        logger=None,
    ):
        if health_weight < 0:
            raise ValueError("health_weight must be non-negative")
        self.retrieval_evaluator = retrieval_evaluator
        self.health_scorer = health_scorer
        self.health_weight = float(health_weight)
        self.quantization_gate = quantization_gate
        self.logger = logger or get_logger()

    def compose_rankings(
        self,
        rankings: Mapping[Any, Sequence[Any]],
        candidate_id: str = "candidate",
        health_inputs: Optional[Dict[str, float]] = None,
        health_result: Optional[StructuralHealthResult] = None,
        structural_health_score: Optional[float] = None,
        quantized_rankings: Optional[Mapping[Any, Sequence[Any]]] = None,
        quantized_fitness: Optional[float] = None,
        baseline_fitness: float = 0.0,
        storage_metadata: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FitnessCompositionResult:
        retrieval = self.retrieval_evaluator.evaluate_rankings(rankings, candidate_id=candidate_id)
        quantized_retrieval = None
        if quantized_rankings is not None:
            quantized_retrieval = self.retrieval_evaluator.evaluate_rankings(
                quantized_rankings,
                candidate_id=f"{candidate_id}_quantized",
            )
            quantized_fitness = quantized_retrieval.fitness
        return self.compose_retrieval_result(
            retrieval,
            candidate_id=candidate_id,
            health_inputs=health_inputs,
            health_result=health_result,
            structural_health_score=structural_health_score,
            quantized_retrieval_result=quantized_retrieval,
            quantized_fitness=quantized_fitness,
            baseline_fitness=baseline_fitness,
            storage_metadata=storage_metadata,
            metadata=metadata,
        )

    def compose_retrieval_result(
        self,
        retrieval: RetrievalFitnessResult,
        candidate_id: Optional[str] = None,
        health_inputs: Optional[Dict[str, float]] = None,
        health_result: Optional[StructuralHealthResult] = None,
        structural_health_score: Optional[float] = None,
        quantized_retrieval_result: Optional[RetrievalFitnessResult] = None,
        quantized_fitness: Optional[float] = None,
        baseline_fitness: float = 0.0,
        storage_metadata: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> FitnessCompositionResult:
        """Compose a precomputed retrieval result into a final optimization score."""

        resolved_candidate_id = candidate_id or retrieval.candidate_id
        resolved_health = self._resolve_health(
            health_inputs=health_inputs,
            health_result=health_result,
            structural_health_score=structural_health_score,
        )
        multiplier = self._health_multiplier(resolved_health)
        final_fitness = float(retrieval.fitness) * multiplier

        gate_result = None
        if self.quantization_gate is not None and (
            quantized_fitness is not None or quantized_retrieval_result is not None
        ):
            quantized_score = (
                quantized_retrieval_result.fitness
                if quantized_retrieval_result is not None
                else float(quantized_fitness)
            )
            gate_result = self.quantization_gate.evaluate(
                fp32_fitness=final_fitness,
                quantized_fitness=float(quantized_score) * multiplier,
                baseline_fitness=float(baseline_fitness),
            )

        result = FitnessCompositionResult(
            candidate_id=resolved_candidate_id,
            retrieval=retrieval,
            final_fitness=final_fitness,
            structural_health=resolved_health,
            structural_health_multiplier=multiplier,
            quantization_gate=gate_result,
            quantized_retrieval=quantized_retrieval_result,
            storage_metadata=storage_metadata or {},
            metadata=metadata or {},
        )
        self.logger.log_event(
            "fitness_composition_evaluated",
            "Composed retrieval, health, quantization, and storage fitness signals",
            candidate_id=result.candidate_id,
            retrieval_fitness=result.retrieval.fitness,
            final_fitness=result.final_fitness,
            structural_health_score=None if result.structural_health is None else result.structural_health.score,
            structural_health_multiplier=result.structural_health_multiplier,
            quantization_gate_passed=None if result.quantization_gate is None else result.quantization_gate.passed,
            storage_latency_ms=result.storage_metadata.get("storage_latency_ms"),
            level="DEBUG",
        )
        return result

    def _resolve_health(
        self,
        health_inputs: Optional[Dict[str, float]],
        health_result: Optional[StructuralHealthResult],
        structural_health_score: Optional[float],
    ) -> Optional[StructuralHealthResult]:
        if health_result is not None:
            return health_result
        if structural_health_score is not None:
            return StructuralHealthResult(score=_bounded_score(structural_health_score), components={})
        if self.health_scorer is not None and health_inputs is not None:
            return self.health_scorer.evaluate(**health_inputs)
        return None

    def _health_multiplier(self, health_result: Optional[StructuralHealthResult]) -> float:
        if health_result is None or self.health_weight <= 0:
            return 1.0
        return health_result.penalty_multiplier(self.health_weight)

