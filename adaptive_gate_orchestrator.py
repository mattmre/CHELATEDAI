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
    recommendations: List[Dict[str, Any]] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "passed": self.passed,
            "status": self.status,
            "actions": self.actions,
            "reasons": self.reasons,
            "recommendations": self.recommendations,
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
        recommendations: List[Dict[str, Any]] = []
        metrics: Dict[str, Any] = {}
        failed = False

        def recommend(action: str, reason: str, severity: str = "warning", confidence: float = 0.75, **params) -> None:
            if action not in actions:
                actions.append(action)
            if reason not in reasons:
                reasons.append(reason)
            recommendations.append({
                "id": action,
                "severity": severity,
                "reason": reason,
                "confidence": float(confidence),
                "apply_mode": "advisory",
                "params": params,
            })

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
                recommend("reduce_optimization_aggression", "structural_health_below_threshold", health_score=score)
                recommend("enable_query_reformulation", "structural_health_below_threshold", health_score=score)

        quantization_gate = diagnostics.get("quantization_gate")
        if isinstance(quantization_gate, dict):
            gate_passed = bool(quantization_gate.get("passed"))
            metrics["quantization_gate_passed"] = gate_passed
            if not gate_passed:
                recommend("reject_quantized_candidate", "quantization_gate_failed")
                if self.require_quantization_gate:
                    failed = True

        storage = diagnostics.get("storage")
        if isinstance(storage, dict) and storage.get("storage_latency_ms") is not None:
            latency = float(storage["storage_latency_ms"])
            metrics["storage_latency_ms"] = latency
            if self.storage_latency_sla_ms is not None and latency > self.storage_latency_sla_ms:
                recommend(
                    "apply_storage_latency_penalty",
                    "storage_latency_sla_exceeded",
                    storage_latency_ms=latency,
                    storage_latency_sla_ms=self.storage_latency_sla_ms,
                )

        norm_drift = diagnostics.get("norm_drift")
        if isinstance(norm_drift, dict):
            ratio = norm_drift.get("adapter_norm_ratio_latest")
            if ratio is None and isinstance(norm_drift.get("latest"), dict):
                ratio = norm_drift["latest"].get("adapter_norm_ratio")
            if ratio is not None:
                ratio = float(ratio)
                metrics["adapter_norm_ratio_latest"] = ratio
                if ratio < 0.5 or ratio > 2.0:
                    recommend(
                        "normalize_runtime_vectors",
                        "adapter_norm_ratio_out_of_band",
                        adapter_norm_ratio=ratio,
                    )
            query_delta = norm_drift.get("query_norm_delta")
            if query_delta is not None:
                metrics["query_norm_delta"] = float(query_delta)

        route_effectiveness = diagnostics.get("route_effectiveness")
        if isinstance(route_effectiveness, dict):
            last_outcome = route_effectiveness.get("last_route_outcome") or {}
            routes = route_effectiveness.get("routes") or {}
            low_effectiveness_routes = [
                key for key, stats in routes.items()
                if isinstance(stats, dict) and stats.get("mean_jaccard") is not None and float(stats["mean_jaccard"]) < 0.25
            ]
            if low_effectiveness_routes:
                metrics["low_effectiveness_routes"] = low_effectiveness_routes
                recommend(
                    "disable_low_effectiveness_route",
                    "route_effectiveness_below_threshold",
                    target_routes=low_effectiveness_routes,
                )
            if isinstance(last_outcome, dict) and last_outcome.get("route_key") == "fallback":
                recommend("disable_low_effectiveness_route", "adapter_route_fallback_selected", target_routes=["fallback"])

        retrieval_policy = diagnostics.get("retrieval_policy")
        if isinstance(retrieval_policy, dict):
            if retrieval_policy.get("high_variance_fast_path"):
                recommend(
                    "prefer_global_scout",
                    "high_variance_fast_path_detected",
                    variance=retrieval_policy.get("variance"),
                    active_threshold=retrieval_policy.get("active_threshold"),
                )
            metrics["retrieval_policy"] = retrieval_policy.get("policy")

        runtime = diagnostics.get("runtime")
        if isinstance(runtime, dict):
            if runtime.get("status") in {"empty_results", "qdrant_error"}:
                recommend("prefer_global_scout", "runtime_retrieval_anomaly", status=runtime.get("status"))
            metrics["latency_ms"] = runtime.get("latency_ms")

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
            recommendations=recommendations,
            metrics=metrics,
        )
        self.logger.log_event(
            "adaptive_gate_evaluated",
            "Evaluated adaptive workflow gates",
            **decision.to_dict(),
            level="DEBUG",
        )
        return decision

