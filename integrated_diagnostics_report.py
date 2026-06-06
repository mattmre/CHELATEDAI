"""Unified diagnostics snapshots for adaptive benchmark and engine workflows."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
try:
    import torch
except ModuleNotFoundError:
    class torch:  # pragma: no cover - fallback for lightweight environments
        class Tensor:
            pass

from chelation_logger import get_logger
from fitness_composition_orchestrator import FitnessCompositionResult


def summarize_adaptive_overlay_report(report: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return a compact diagnostics summary for an adaptive overlay report."""

    if not isinstance(report, dict):
        return None

    def _as_int(value: Any) -> int:
        try:
            return int(value)
        except (TypeError, ValueError):
            return 0

    def _as_float(value: Any) -> float:
        try:
            return float(value)
        except (TypeError, ValueError):
            return 0.0

    readiness = report.get("readiness", {})
    if not isinstance(readiness, dict):
        readiness = {}
    summary = report.get("summary", {})
    if not isinstance(summary, dict):
        summary = {}
    metrics = report.get("branch_set_metrics", {})
    if not isinstance(metrics, dict):
        metrics = {}
    return _json_safe(
        {
            "schema_version": report.get("schema_version"),
            "record_type": "adaptive_overlay_summary",
            "source_record_type": report.get("record_type"),
            "ready_for_broader_validation": bool(readiness.get("ready_for_broader_validation", False)),
            "blockers": list(readiness.get("blockers", [])) if isinstance(readiness.get("blockers", []), list) else [],
            "next_action": readiness.get("next_action"),
            "record_count": _as_int(summary.get("record_count", 0)),
            "channel_types": summary.get("channel_types", {}),
            "decisions": summary.get("decisions", {}),
            "promotion_blockers": _as_int(summary.get("promotion_blockers", 0)),
            "active_negative_records": _as_int(summary.get("active_negative_records", 0)),
            "branch_set_metrics": {
                "group_count": _as_int(metrics.get("group_count", 0)),
                "pass_at_k_rate": _as_float(metrics.get("pass_at_k_rate", 0.0)),
                "safe_pass_at_k_rate": _as_float(metrics.get("safe_pass_at_k_rate", 0.0)),
                "regressed_at_k_rate": _as_float(metrics.get("regressed_at_k_rate", 0.0)),
                "mean_best_delta": _as_float(metrics.get("mean_best_delta", 0.0)),
                "mean_branch_delta": _as_float(metrics.get("mean_branch_delta", 0.0)),
                "mean_oracle_gap": _as_float(metrics.get("mean_oracle_gap", 0.0)),
            },
        }
    )


def _json_safe(value: Any) -> Any:
    """Convert diagnostics payloads to plain JSON-compatible values."""

    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, torch.Tensor):
        return _json_safe(value.detach().cpu().tolist())
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


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
    runtime: Optional[Dict[str, Any]] = None
    model_scope: Optional[Dict[str, Any]] = None
    norm_drift: Optional[Dict[str, Any]] = None
    route_effectiveness: Optional[Dict[str, Any]] = None
    retrieval_policy: Optional[Dict[str, Any]] = None
    telemetry: Optional[Dict[str, Any]] = None
    query_summary: Optional[Dict[str, Any]] = None
    training_summary: Optional[Dict[str, Any]] = None
    next_cycle_plan: Optional[Dict[str, Any]] = None
    evidence_summary: Optional[Dict[str, Any]] = None
    evaluator_summary: Optional[Dict[str, Any]] = None
    safety_summary: Optional[Dict[str, Any]] = None
    rag_faithfulness: Optional[Dict[str, Any]] = None
    reward_overoptimization: Optional[Dict[str, Any]] = None
    hard_negative_summary: Optional[Dict[str, Any]] = None
    adaptive_overlay_summary: Optional[Dict[str, Any]] = None

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
        runtime: Optional[Dict[str, Any]] = None,
        model_scope: Optional[Dict[str, Any]] = None,
        norm_drift: Optional[Dict[str, Any]] = None,
        route_effectiveness: Optional[Dict[str, Any]] = None,
        retrieval_policy: Optional[Dict[str, Any]] = None,
        telemetry: Optional[Dict[str, Any]] = None,
        query_summary: Optional[Dict[str, Any]] = None,
        training_summary: Optional[Dict[str, Any]] = None,
        next_cycle_plan: Optional[Dict[str, Any]] = None,
        evidence_summary: Optional[Dict[str, Any]] = None,
        evaluator_summary: Optional[Dict[str, Any]] = None,
        safety_summary: Optional[Dict[str, Any]] = None,
        rag_faithfulness: Optional[Dict[str, Any]] = None,
        reward_overoptimization: Optional[Dict[str, Any]] = None,
        hard_negative_summary: Optional[Dict[str, Any]] = None,
        adaptive_overlay_summary: Optional[Dict[str, Any]] = None,
    ) -> "IntegratedDiagnosticsReport":
        composed = composition.to_dict()
        return cls(
            cycle=cycle,
            phase=phase,
            candidate_id=composition.candidate_id,
            retrieval_fitness=composed.get("retrieval_fitness"),
            final_fitness=composed.get("final_fitness"),
            baseline_fitness=baseline_fitness,
            retrieval_metrics=composed.get("retrieval_metrics", {}),
            structural_health=composed.get("structural_health"),
            quantization_gate=composed.get("quantization_gate"),
            storage=composed.get("storage_metadata", {}),
            es=es_result or {},
            adaptive_gate=adaptive_gate,
            timings=timings or {},
            metadata=metadata or {},
            runtime=runtime,
            model_scope=model_scope,
            norm_drift=norm_drift,
            route_effectiveness=route_effectiveness,
            retrieval_policy=retrieval_policy,
            telemetry=telemetry,
            query_summary=query_summary,
            training_summary=training_summary,
            next_cycle_plan=next_cycle_plan,
            evidence_summary=evidence_summary,
            evaluator_summary=evaluator_summary,
            safety_summary=safety_summary,
            rag_faithfulness=rag_faithfulness,
            reward_overoptimization=reward_overoptimization,
            hard_negative_summary=hard_negative_summary,
            adaptive_overlay_summary=adaptive_overlay_summary,
        )

    def to_dict(self) -> Dict[str, Any]:
        report = {
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
        optional_sections = {
            "runtime": self.runtime,
            "model_scope": self.model_scope,
            "norm_drift": self.norm_drift,
            "route_effectiveness": self.route_effectiveness,
            "retrieval_policy": self.retrieval_policy,
            "telemetry": self.telemetry,
            "query_summary": self.query_summary,
            "training_summary": self.training_summary,
            "next_cycle_plan": self.next_cycle_plan,
            "evidence_summary": self.evidence_summary,
            "evaluator_summary": self.evaluator_summary,
            "safety_summary": self.safety_summary,
            "rag_faithfulness": self.rag_faithfulness,
            "reward_overoptimization": self.reward_overoptimization,
            "hard_negative_summary": self.hard_negative_summary,
            "adaptive_overlay_summary": self.adaptive_overlay_summary,
        }
        for key, value in optional_sections.items():
            if value is not None:
                report[key] = value
        return _json_safe(report)

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
