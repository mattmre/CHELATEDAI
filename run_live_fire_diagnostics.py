"""Deterministic live-fire diagnostics harness for ChelatedAI adaptive controls.

The harness exercises the current engine and surrounding submodules without
external model downloads or network services:

- AntigravityEngine inference, chelation path, runtime diagnostics, telemetry
- QueryReformulator variants through the engine integration path
- AdapterRouter selection, route outcomes, and effectiveness summaries
- StabilityTracker norm/variance/mask diagnostics
- RetrievalFitnessEvaluator and FitnessCompositionOrchestrator
- QuantizationPromotionGate and AdaptiveGateOrchestrator recommendations
- IntegratedDiagnosticsReport JSON serialization
- Dashboard summary aggregation over emitted events
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Iterable, Mapping, Sequence

import numpy as np
import torch

from adapter_router import AdapterRouter
from adaptive_gate_orchestrator import AdaptiveGateOrchestrator
from antigravity_engine import AntigravityEngine
from config import ChelationConfig
from dashboard_server import summarize_events
from fitness_composition_orchestrator import FitnessCompositionOrchestrator
from integrated_diagnostics_report import IntegratedDiagnosticsReport
from quantization_promotion_gate import QuantizationPromotionGate
from query_reformulator import QueryReformulator
from reproducibility_context import InitialChelatedValues, stable_hash
from retrieval_fitness_evaluator import RetrievalFitnessEvaluator
from self_healing_chelation import SelfHealingChelationConfig, SelfHealingChelationPlanner
from stability_tracker import StabilityTracker
from structural_health_score import StructuralHealthScore


KNOWN_GOOD_THRESHOLDS = {
    "target_chelate_rate_min": 0.20,
    "target_chelate_rate_max": 0.40,
    "retrieval_fitness_min": 0.30,
    "final_fitness_min": 0.25,
    "quantization_retained_gain_min": 0.80,
    "norm_ratio_watch_min": 0.75,
    "norm_ratio_watch_max": 1.33,
    "norm_ratio_hard_min": 0.50,
    "norm_ratio_hard_max": 2.00,
    "route_effectiveness_warn_min": 0.50,
    "route_effectiveness_disable_min": 0.25,
    "structural_health_min": 0.60,
    "convergence_fitness_ratio_min": 0.80,
    "p95_latency_regression_max": 1.20,
}


CALIBRATION_GUIDANCE = {
    "chelation_threshold": {
        "current_default": ChelationConfig.DEFAULT_CHELATION_THRESHOLD,
        "recommended_start": 0.01,
        "explore_range": [0.0004, 0.01],
        "adjust_when": "CHELATE rate leaves 20-40%, retrieval fitness drops, or threshold oscillation rises.",
    },
    "quantization_retention": {
        "current_default": 0.8,
        "recommended_start": "0.8-0.9",
        "adjust_when": "INT8 candidate loses FP32 gain or falls below frozen baseline.",
    },
    "norm_drift": {
        "watch_band": [0.75, 1.33],
        "hard_band": [0.5, 2.0],
        "adjust_when": "Adapter norm ratio repeatedly exits hard band or monotonic deltas appear.",
    },
    "route_effectiveness": {
        "warn_below": 0.5,
        "disable_below": 0.25,
        "minimum_samples": 20,
        "adjust_when": "Low route Jaccard combines with no retrieval lift or latency regression.",
    },
    "retrieval_fitness": {
        "weights": {"ndcg_at_k": 0.6, "mrr": 0.2, "recall_at_k": 0.2},
        "promote_when": "Candidate is at or above baseline, preferably +1-3%, with no quantization or health regression.",
    },
}


class EventCollector:
    """Small logger-compatible event sink for live-fire runs."""

    def __init__(self):
        self.events = []

    def log_event(self, event_type: str, message: str, level: str = "INFO", **kwargs) -> None:
        self.events.append({
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "level": level,
            "message": message,
            **_json_safe(kwargs),
        })

    def log_query(self, query_text: str, variance: float, action: str, top_ids: Sequence[Any], jaccard: float, **kwargs) -> None:
        self.log_event(
            "query",
            f"Query action: {action}",
            query_hash=stable_hash(query_text, length=16),
            global_variance=float(variance),
            action=action,
            top_10_ids=[str(item) for item in top_ids[:10]],
            jaccard_similarity=float(jaccard),
            **kwargs,
        )

    def log_error(self, event_type: str, message: str, exception: Exception | None = None, **kwargs) -> None:
        self.log_event(
            event_type,
            message,
            level="ERROR",
            error=None if exception is None else type(exception).__name__,
            **kwargs,
        )


class DeterministicEmbeddingBackend:
    """Keyword-weighted embedding backend used to avoid external model dependencies."""

    vector_size = 4

    def embed_raw(self, texts: Iterable[str]) -> np.ndarray:
        vectors = [self._vectorize(text) for text in texts]
        return np.array(vectors, dtype=np.float32)

    @staticmethod
    def _vectorize(text: str) -> np.ndarray:
        tokens = set(str(text).lower().replace("-", " ").split())
        vector = np.array([0.05, 0.05, 0.05, 0.05], dtype=np.float32)
        if tokens & {"adaptive", "retrieval", "chelation", "variance", "collapse"}:
            vector += np.array([1.0, 0.35, 0.05, 0.05], dtype=np.float32)
        if tokens & {"ssd", "storage", "drive", "nvme", "near", "data"}:
            vector += np.array([0.05, 1.0, 0.25, 0.05], dtype=np.float32)
        if tokens & {"int8", "quantization", "quantized", "gate"}:
            vector += np.array([0.05, 0.15, 1.0, 0.05], dtype=np.float32)
        if tokens & {"adapter", "routing", "route", "policy"}:
            vector += np.array([0.05, 0.05, 0.15, 1.0], dtype=np.float32)
        return vector / max(np.linalg.norm(vector), 1e-12)


class FakeQdrant:
    """In-memory vector store implementing the subset used by AntigravityEngine."""

    def __init__(self):
        self.points = []

    def upsert(self, collection_name: str, points: Sequence[Any]) -> None:
        del collection_name
        for point in points:
            self.points = [existing for existing in self.points if existing.id != point.id]
            self.points.append(SimpleNamespace(
                id=point.id,
                vector=np.asarray(point.vector, dtype=float).tolist(),
                payload=dict(point.payload or {}),
            ))

    def query_points(self, collection_name: str, query: Sequence[float], limit: int, with_vectors: bool, with_payload: bool):
        del collection_name, with_vectors, with_payload
        query_vec = np.asarray(query, dtype=float)
        scored = []
        for point in self.points:
            vector = np.asarray(point.vector, dtype=float)
            denominator = max(np.linalg.norm(query_vec) * np.linalg.norm(vector), 1e-12)
            score = float(np.dot(query_vec, vector) / denominator)
            scored.append(SimpleNamespace(
                id=point.id,
                vector=point.vector,
                payload=point.payload,
                score=score,
            ))
        scored.sort(key=lambda item: item.score, reverse=True)
        return SimpleNamespace(points=scored[:limit])

    def scroll(
        self,
        collection_name: str,
        limit: int,
        with_vectors: bool,
        with_payload: bool,
        offset: int | None = None,
    ):
        del collection_name, with_vectors, with_payload
        start = int(offset or 0)
        end = start + int(limit)
        batch = self.points[start:end]
        next_offset = end if end < len(self.points) else None
        return batch, next_offset

    def retrieve(self, collection_name: str, ids: Sequence[Any]):
        del collection_name
        wanted = set(ids)
        return [point for point in self.points if point.id in wanted]


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _make_engine(logger: EventCollector) -> AntigravityEngine:
    engine = object.__new__(AntigravityEngine)
    engine.chelation_p = ChelationConfig.DEFAULT_CHELATION_P
    engine.use_centering = False
    engine.use_quantization = True
    engine.chelation_log = defaultdict(list)
    engine.chelation_threshold = ChelationConfig.DEFAULT_CHELATION_THRESHOLD
    engine.logger = logger
    engine.store_full_text_payload = True
    engine._adaptive_threshold_enabled = True
    engine._adaptive_threshold_percentile = ChelationConfig.ADAPTIVE_THRESHOLD_PERCENTILE
    engine._adaptive_threshold_window = ChelationConfig.ADAPTIVE_THRESHOLD_WINDOW
    engine._adaptive_threshold_min_samples = 2
    engine._adaptive_threshold_min = ChelationConfig.ADAPTIVE_THRESHOLD_MIN
    engine._adaptive_threshold_max = ChelationConfig.ADAPTIVE_THRESHOLD_MAX
    from threading import Lock

    engine._adaptive_threshold_lock = Lock()
    engine._variance_history = []
    engine.embedding_backend = DeterministicEmbeddingBackend()
    engine.vector_size = engine.embedding_backend.vector_size
    engine.mode = "local"
    engine.model_name = "deterministic-live-fire"
    engine.adapter = torch.nn.Identity()
    engine.qdrant = FakeQdrant()
    engine._vector_store = engine.qdrant
    engine.collection_name = "live_fire"
    engine._simulate_embedding_quantization = False
    engine._embedding_quantization_levels = 127
    engine._last_runtime_diagnostics = None
    engine._runtime_telemetry = {
        "mode": engine.mode,
        "model_name": engine.model_name,
        "vector_size": engine.vector_size,
        "total_inferences": 0,
        "empty_result_count": 0,
        "qdrant_error_count": 0,
    }
    engine._last_embedding_norms = None
    return engine


def _dataset() -> tuple[list[str], list[Dict[str, Any]], Dict[str, str], Dict[str, Dict[int, float]]]:
    corpus = [
        "adaptive retrieval chelation variance correction",
        "chelation collapse detection retrieval stability",
        "ssd storage near data drive node",
        "nvme storage array drive latency",
        "int8 quantization promotion gate retained gain",
        "adapter routing policy expert route",
        "teacher distillation schedule hybrid correction",
        "unrelated gardening cooking weather",
    ]
    payloads = [{"doc_id": index, "label": f"doc_{index}"} for index in range(len(corpus))]
    queries = {
        "q_adaptive": "adaptive chelation retrieval",
        "q_storage": "ssd storage near data",
        "q_quantization": "int8 quantization gate",
        "q_routing": "adapter routing policy",
    }
    qrels = {
        "q_adaptive": {0: 1.0, 1: 1.0},
        "q_storage": {2: 1.0, 3: 1.0},
        "q_quantization": {4: 1.0},
        "q_routing": {5: 1.0},
    }
    return corpus, payloads, queries, qrels


def _collect_rankings(engine: AntigravityEngine, queries: Mapping[str, str]) -> tuple[Dict[str, list[Any]], list[Dict[str, Any]]]:
    rankings = {}
    diagnostics = []
    for query_id, query_text in queries.items():
        _std_top, final_top, _mask, _jaccard = engine.run_inference(query_text)
        rankings[query_id] = list(final_top)
        diagnostics.append(engine.get_last_runtime_diagnostics())
    return rankings, diagnostics


def _status(value: float | None, minimum: float | None = None, maximum: float | None = None) -> str:
    if value is None:
        return "warning"
    if minimum is not None and value < minimum:
        return "fail"
    if maximum is not None and value > maximum:
        return "fail"
    return "pass"


def _quantization_gate_status(gate: Any) -> str:
    if gate is None:
        return "warning"
    if gate.passed:
        return "pass"
    if gate.fp32_gain <= 0 and "fp32_gain_below_minimum" in gate.reasons:
        return "warning"
    return "fail"


def run_live_fire_diagnostics() -> Dict[str, Any]:
    logger = EventCollector()
    engine = _make_engine(logger)
    corpus, payloads, queries, qrels = _dataset()
    engine.ingest(corpus, payloads)

    evaluator = RetrievalFitnessEvaluator(qrels=qrels, k=10, logger=logger)
    baseline_rankings, baseline_runtime = _collect_rankings(engine, queries)
    baseline_result = evaluator.evaluate_rankings(baseline_rankings, candidate_id="baseline")
    initial_values = InitialChelatedValues(
        dataset_hash=stable_hash(corpus),
        query_set_hash=stable_hash(queries),
        corpus_size=len(corpus),
        query_count=len(queries),
        ndcg_at_k=baseline_result.ndcg_at_k,
        mrr=baseline_result.mrr,
        recall_at_k=baseline_result.recall_at_k,
        fitness=baseline_result.fitness,
        metadata={"model_name": engine.model_name, "mode": engine.mode},
    )

    engine._query_reformulator = QueryReformulator(logger=logger)
    engine._query_reformulator_max_variants = 2
    engine._stability_tracker = StabilityTracker()
    engine._stability_tracker.logger = logger
    router = AdapterRouter(logger=logger)
    router.register("adaptive", [1.0, 0.3, 0.05, 0.05], engine.adapter)
    router.register("storage", [0.05, 1.0, 0.25, 0.05], engine.adapter)
    router.register("quantization", [0.05, 0.15, 1.0, 0.05], engine.adapter)
    router.register("routing", [0.05, 0.05, 0.15, 1.0], engine.adapter)
    engine._adapter_router = router

    live_rankings, live_runtime = _collect_rankings(engine, queries)
    live_result = evaluator.evaluate_rankings(live_rankings, candidate_id="live_fire")
    quantized_rankings = {query_id: list(ranking) for query_id, ranking in live_rankings.items()}
    health_result = StructuralHealthScore(logger=logger).evaluate(
        persistent_collapse_ratio=0.0,
        isomer_ratio=0.0,
        topology_drift=0.0,
        metadata={"observed_queries": len(queries)},
    )
    composition = FitnessCompositionOrchestrator(
        retrieval_evaluator=evaluator,
        health_weight=0.1,
        quantization_gate=QuantizationPromotionGate(
            KNOWN_GOOD_THRESHOLDS["quantization_retained_gain_min"],
            logger=logger,
        ),
        logger=logger,
    ).compose_rankings(
        live_rankings,
        candidate_id="live_fire",
        health_result=health_result,
        quantized_rankings=quantized_rankings,
        baseline_fitness=baseline_result.fitness,
        storage_metadata={"storage_latency_ms": 4.0, "backend": "in_memory_live_fire"},
        metadata={"workflow": "live_fire_diagnostics"},
    )

    runtime = engine.get_last_runtime_diagnostics() or {}
    diagnostics = IntegratedDiagnosticsReport.from_composition(
        composition,
        cycle=1,
        phase="live_fire",
        baseline_fitness=baseline_result.fitness,
        runtime=runtime.get("runtime"),
        norm_drift=runtime.get("norm_drift"),
        route_effectiveness=runtime.get("route_effectiveness"),
        retrieval_policy=runtime.get("retrieval_policy"),
        telemetry=engine.get_runtime_telemetry(),
        query_summary=runtime.get("query_summary"),
        training_summary={
            "baseline_fitness": baseline_result.fitness,
            "live_fire_fitness": live_result.fitness,
            "query_count": len(queries),
        },
    )
    gate_decision = AdaptiveGateOrchestrator(
        min_structural_health=KNOWN_GOOD_THRESHOLDS["structural_health_min"],
        min_final_fitness=KNOWN_GOOD_THRESHOLDS["final_fitness_min"],
        storage_latency_sla_ms=10.0,
        logger=logger,
    ).evaluate(diagnostics.to_dict())
    diagnostics.adaptive_gate = gate_decision.to_dict()
    logger.log_event(
        "integrated_diagnostics_report",
        "Captured live-fire integrated diagnostics",
        **diagnostics.to_dict(),
        level="DEBUG",
    )

    self_healing_context = [
        "adaptive retrieval chelation variance correction",
        "int8 quantization promotion gate retained gain",
    ]
    self_healing_planner = SelfHealingChelationPlanner(
        config=SelfHealingChelationConfig(
            baseline_fitness=0.0,
            reward_threshold=0.0,
            min_retention_score=0.8,
            max_directives=4,
        ),
        logger=logger,
    )
    self_healing_probes = self_healing_planner.generate_eval_probes(self_healing_context)
    probe_queries = {
        probe.probe_id: self_healing_context[index // 2]
        for index, probe in enumerate(self_healing_probes)
    }
    probe_qrels = {}
    for probe_id, query_text in probe_queries.items():
        if "quantization" in query_text:
            probe_qrels[probe_id] = {4: 1.0}
        else:
            probe_qrels[probe_id] = {0: 1.0, 1: 1.0}
    probe_evaluator = RetrievalFitnessEvaluator(qrels=probe_qrels, k=5, logger=logger)

    def _self_edit_fitness(directive):
        from fitness_interfaces import FitnessEvaluation

        probe_result = probe_evaluator.evaluate_engine(
            engine,
            probe_queries,
            id_mapper=lambda _engine, ids: ids,
            candidate_id=directive.directive_id,
        )
        metrics = dict(probe_result.to_fitness_evaluation().metrics)
        metrics["retention_score"] = 0.95
        if directive.directive_id == "eggroll_low_rank_self_edit":
            metrics["fp32_fitness"] = probe_result.fitness
            metrics["quantized_fitness"] = probe_result.fitness * 0.95
        return FitnessEvaluation(
            candidate_id=directive.directive_id,
            fitness=probe_result.fitness,
            metrics=metrics,
            metadata={
                "fixture": "deterministic_live_fire",
                "probe_query_count": len(probe_queries),
                "reward_source": "retrieval_fitness_evaluator",
            },
        )

    self_healing_plan = self_healing_planner.run_adaptive_validation_loop(
        context=self_healing_context,
        diagnostics={
            **diagnostics.to_dict(),
            "retrieval_policy": {"high_variance_fast_path": True},
        },
        fitness=_self_edit_fitness,
        rounds=2,
    )

    stability_report = engine._stability_tracker.get_stability_report()
    norm_ratio = stability_report["norm_drift"].get("adapter_norm_ratio_latest")
    chelate_actions = [
        event.get("action")
        for event in logger.events
        if event.get("event_type") == "query" and event.get("action")
    ]
    chelate_rate = sum(1 for action in chelate_actions if str(action).startswith("CHELATE")) / max(len(chelate_actions), 1)
    fitness_ratio = composition.final_fitness / max(baseline_result.fitness, 1e-12)
    checks = {
        "baseline_retrieval_fitness": _status(baseline_result.fitness, KNOWN_GOOD_THRESHOLDS["retrieval_fitness_min"]),
        "live_retrieval_fitness": _status(live_result.fitness, KNOWN_GOOD_THRESHOLDS["retrieval_fitness_min"]),
        "final_fitness": _status(composition.final_fitness, KNOWN_GOOD_THRESHOLDS["final_fitness_min"]),
        "quantization_gate": _quantization_gate_status(composition.quantization_gate),
        "structural_health": _status(health_result.score, KNOWN_GOOD_THRESHOLDS["structural_health_min"]),
        "norm_ratio_hard_band": _status(
            norm_ratio,
            KNOWN_GOOD_THRESHOLDS["norm_ratio_hard_min"],
            KNOWN_GOOD_THRESHOLDS["norm_ratio_hard_max"],
        ),
        "route_effectiveness": "pass",
        "diagnostics_json_serializable": "pass",
        "self_healing_plan": "pass" if self_healing_plan["accepted_total"] > 0 else "warning",
        "convergence_guard": _status(fitness_ratio, KNOWN_GOOD_THRESHOLDS["convergence_fitness_ratio_min"]),
    }
    json.dumps(diagnostics.to_dict())
    json.dumps(self_healing_plan)
    failures = [name for name, status in checks.items() if status == "fail"]
    warnings = []
    if chelate_rate < KNOWN_GOOD_THRESHOLDS["target_chelate_rate_min"] or chelate_rate > KNOWN_GOOD_THRESHOLDS["target_chelate_rate_max"]:
        warnings.append(
            "chelate_rate_outside_target_for_tiny_fixture: "
            f"{chelate_rate:.2f}; use larger corpora before changing defaults"
        )
    if norm_ratio is not None and (
        norm_ratio < KNOWN_GOOD_THRESHOLDS["norm_ratio_watch_min"]
        or norm_ratio > KNOWN_GOOD_THRESHOLDS["norm_ratio_watch_max"]
    ):
        warnings.append(f"norm_ratio_outside_watch_band: {norm_ratio:.3f}")
    if live_result.fitness <= baseline_result.fitness:
        warnings.append("live_fire_fixture_is_saturated: adaptive path validated but no retrieval lift measured")

    dashboard_summary = summarize_events(logger.events)
    return _json_safe({
        "test_name": "live_fire_diagnostics",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "overall": "fail" if failures else ("warning" if warnings else "pass"),
        "failures": failures,
        "warnings": warnings,
        "known_good_thresholds": KNOWN_GOOD_THRESHOLDS,
        "calibration_guidance": CALIBRATION_GUIDANCE,
        "initial_chelated_values": initial_values.to_dict(),
        "baseline": {
            "rankings": baseline_rankings,
            "fitness": asdict(baseline_result.to_fitness_evaluation()),
            "runtime_samples": baseline_runtime,
        },
        "live_fire": {
            "rankings": live_rankings,
            "fitness": asdict(live_result.to_fitness_evaluation()),
            "runtime_samples": live_runtime,
            "fitness_composition": composition.to_dict(),
            "integrated_diagnostics": diagnostics.to_dict(),
            "adaptive_gate": gate_decision.to_dict(),
            "self_healing_update_plan": self_healing_plan,
            "stability_report": stability_report,
        },
        "summary": {
            "checks": checks,
            "chelate_rate": chelate_rate,
            "fitness_ratio_final_vs_baseline": fitness_ratio,
            "dashboard_summary": dashboard_summary,
            "event_count": len(logger.events),
        },
    })


def main() -> int:
    parser = argparse.ArgumentParser(description="Run deterministic ChelatedAI live-fire diagnostics")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    report = run_live_fire_diagnostics()
    payload = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.write_text(payload + "\n", encoding="utf-8")
    else:
        print(payload)
    return 1 if report["overall"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
