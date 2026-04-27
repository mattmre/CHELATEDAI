"""Policy layer that turns diagnostics snapshots into adaptive decisions."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from chelation_logger import get_logger


@dataclass
class AdaptiveGateDecision:
    """Decision emitted after evaluating integrated diagnostics."""

    passed: bool
    status: str
    actions: List[str] = field(default_factory=list)
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "status": self.status,
            "actions": self.actions,
            "reasons": self.reasons,
            "metrics": self.metrics,
        }


class AdaptiveGateOrchestrator:
    """Evaluate adaptive workflow gates without mutating engine state."""

    def __init__(
        self,
        min_structural_health: float = 0.6,
        require_quantization_gate: bool = False,
        storage_latency_sla_ms: Optional[float] = None,
        min_final_fitness: Optional[float] = None,
        logger=None,
    ):
        if min_structural_health < 0 or min_structural_health > 1:
            raise ValueError("min_structural_health must be in [0, 1]")
        if storage_latency_sla_ms is not None and storage_latency_sla_ms < 0:
            raise ValueError("storage_latency_sla_ms must be non-negative")
        self.min_structural_health = float(min_structural_health)
        self.require_quantization_gate = bool(require_quantization_gate)
        self.storage_latency_sla_ms = storage_latency_sla_ms
        self.min_final_fitness = min_final_fitness
        self.logger = logger or get_logger()

    def evaluate(self, diagnostics: Dict[str, Any]) -> AdaptiveGateDecision:
        actions: List[str] = []
        reasons: List[str] = []
        metrics: Dict[str, Any] = {}
        failed = False

        final_fitness = diagnostics.get("final_fitness")
        if final_fitness is not None:
            metrics["final_fitness"] = final_fitness
            if self.min_final_fitness is not None and float(final_fitness) < self.min_final_fitness:
                failed = True
                actions.append("reject_candidate")
                reasons.append("final_fitness_below_minimum")

        structural_health = diagnostics.get("structural_health")
        if isinstance(structural_health, dict) and structural_health.get("score") is not None:
            score = float(structural_health["score"])
            metrics["structural_health_score"] = score
            if score < self.min_structural_health:
                actions.extend(["reduce_optimization_aggression", "enable_query_reformulation"])
                reasons.append("structural_health_below_threshold")

        quantization_gate = diagnostics.get("quantization_gate")
        if isinstance(quantization_gate, dict):
            gate_passed = bool(quantization_gate.get("passed"))
            metrics["quantization_gate_passed"] = gate_passed
            if not gate_passed:
                actions.append("reject_quantized_candidate")
                reasons.append("quantization_gate_failed")
                if self.require_quantization_gate:
                    failed = True

        storage = diagnostics.get("storage")
        if isinstance(storage, dict) and storage.get("storage_latency_ms") is not None:
            latency = float(storage["storage_latency_ms"])
            metrics["storage_latency_ms"] = latency
            if self.storage_latency_sla_ms is not None and latency > self.storage_latency_sla_ms:
                actions.append("apply_storage_latency_penalty")
                reasons.append("storage_latency_sla_exceeded")

        if failed:
            status = "fail"
        elif actions:
            status = "warning"
        else:
            status = "pass"

        decision = AdaptiveGateDecision(
            passed=not failed,
            status=status,
            actions=actions,
            reasons=reasons,
            metrics=metrics,
        )
        self.logger.log_event(
            "adaptive_gate_evaluated",
            "Evaluated adaptive workflow gates",
            **decision.to_dict(),
            level="DEBUG",
        )
        return decision

