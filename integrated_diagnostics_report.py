"""Unified diagnostics snapshots for adaptive benchmark and engine workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from chelation_logger import get_logger
from fitness_composition_orchestrator import FitnessCompositionResult


@dataclass
class IntegratedDiagnosticsReport:
    """Cycle-level view of retrieval, health, gate, storage, and ES signals."""

    cycle: Optional[int] = None
    phase: str = "runtime"
    candidate_id: str = "candidate"
    retrieval_fitness: Optional[float] = None
    final_fitness: Optional[float] = None
    baseline_fitness: Optional[float] = None
    retrieval_metrics: Dict[str, float] = field(default_factory=dict)
    structural_health: Optional[Dict[str, Any]] = None
    quantization_gate: Optional[Dict[str, Any]] = None
    storage: Dict[str, Any] = field(default_factory=dict)
    es: Dict[str, Any] = field(default_factory=dict)
    adaptive_gate: Optional[Dict[str, Any]] = None
    timings: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_composition(
        cls,
        composition: FitnessCompositionResult,
        cycle: Optional[int] = None,
        phase: str = "optimization",
        baseline_fitness: Optional[float] = None,
        es_result: Optional[Dict[str, Any]] = None,
        adaptive_gate: Optional[Dict[str, Any]] = None,
        timings: Optional[Dict[str, float]] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "IntegratedDiagnosticsReport":
        composed = composition.to_dict()
        return cls(
            cycle=cycle,
            phase=phase,
            candidate_id=composition.candidate_id,
            retrieval_fitness=composed["retrieval_fitness"],
            final_fitness=composed["final_fitness"],
            baseline_fitness=baseline_fitness,
            retrieval_metrics=composed["retrieval_metrics"],
            structural_health=composed["structural_health"],
            quantization_gate=composed["quantization_gate"],
            storage=composed["storage_metadata"],
            es=es_result or {},
            adaptive_gate=adaptive_gate,
            timings=timings or {},
            metadata=metadata or {},
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle": self.cycle,
            "phase": self.phase,
            "candidate_id": self.candidate_id,
            "retrieval_fitness": self.retrieval_fitness,
            "final_fitness": self.final_fitness,
            "baseline_fitness": self.baseline_fitness,
            "retrieval_metrics": self.retrieval_metrics,
            "structural_health": self.structural_health,
            "quantization_gate": self.quantization_gate,
            "storage": self.storage,
            "es": self.es,
            "adaptive_gate": self.adaptive_gate,
            "timings": self.timings,
            "metadata": self.metadata,
        }

    def log(self, logger=None) -> None:
        resolved_logger = logger or get_logger()
        resolved_logger.log_event(
            "integrated_diagnostics_report",
            "Captured integrated adaptive workflow diagnostics",
            **self.to_dict(),
            level="DEBUG",
        )


def extract_latest_storage_evaluation(es_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Return the latest ES storage-evaluation metadata, if present."""

    if not es_result:
        return {}
    final = es_result.get("final", {})
    if isinstance(final, dict) and isinstance(final.get("storage_evaluation"), dict):
        return final["storage_evaluation"]
    history = es_result.get("history", [])
    if isinstance(history, list):
        for generation in reversed(history):
            if isinstance(generation, dict) and isinstance(generation.get("storage_evaluation"), dict):
                return generation["storage_evaluation"]
    return {}

