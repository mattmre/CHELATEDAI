"""Deterministic safety testbed for non-saturated ChelatedAI closed-course loops."""

from __future__ import annotations

from dataclasses import asdict
import json
from typing import Any, Dict, Mapping, Sequence

from adapter_router import AdapterRouter
from adaptive_gate_orchestrator import AdaptiveGateOrchestrator
from dashboard_server import summarize_events
from integrated_diagnostics_report import IntegratedDiagnosticsReport
from quantization_promotion_gate import QuantizationPromotionGate
from query_reformulator import QueryReformulator
from reproducibility_context import InitialChelatedValues, stable_hash
from retrieval_fitness_evaluator import RetrievalFitnessEvaluator
from run_live_fire_diagnostics import EventCollector, _json_safe, _make_engine
from stability_tracker import StabilityTracker
from structural_health_score import StructuralHealthScore


CALIBRATION_PROFILES = {
    "conservative": {
        "chelation_threshold": 999.0,
        "chelation_p": 25,
        "query_reformulation": False,
        "adapter_routing": False,
    },
    "balanced": {
        "chelation_threshold": 0.01,
        "chelation_p": 50,
        "query_reformulation": False,
        "adapter_routing": False,
    },
    "aggressive": {
        "chelation_threshold": 0.0,
        "chelation_p": 75,
        "query_reformulation": True,
        "adapter_routing": False,
    },
    "experimental": {
        "chelation_threshold": 0.0,
        "chelation_p": 75,
        "query_reformulation": True,
        "adapter_routing": True,
    },
}


ROAD_COURSE_CAMPAIGNS = {
    "beir_small_tier": {
        "objective": "Freeze a BEIR small-tier baseline and compare adaptive retrieval, reformulation, routing, and quantized candidates.",
        "commands": [
            "python benchmark_beir.py",
            "python run_safety_testbed.py",
        ],
        "required_evidence": [
            "baseline NDCG/MRR/Recall",
            "adaptive NDCG/MRR/Recall",
            "quantization retained gain",
            "structural health score",
            "runtime latency percentiles",
            "warnings and failure counts",
        ],
    },
    "multitask_transfer": {
        "objective": "Validate that candidate settings transfer across held-out task/query families.",
        "commands": ["python benchmark_multitask.py"],
        "required_evidence": [
            "source-task metrics",
            "transfer-task metrics",
            "route effectiveness by task",
            "query reformulation action mix",
        ],
    },
    "repeatability_matrix": {
        "objective": "Run repeated seeds and reject settings that only work once.",
        "commands": ["python run_repeatability_check.py"],
        "required_evidence": [
            "seed matrix",
            "mean/std/min/max score",
            "tolerance decision",
            "artifact hash",
        ],
    },
    "quantization_survival": {
        "objective": "Ensure candidate gains survive INT8 simulation before any promotion.",
        "commands": ["python run_candidate_transfer_gate.py --require-quantization-gate"],
        "required_evidence": [
            "FP32 fitness",
            "quantized fitness",
            "retained gain ratio",
            "gate reasons on failure",
        ],
    },
    "structural_health_ablation": {
        "objective": "Confirm structural-health penalties correlate with retrieval outcomes and catch collapse/isomer/topology regressions.",
        "commands": ["python -m unittest test_structural_health_report.py test_topology_analyzer.py test_isomer_detector.py -v"],
        "required_evidence": [
            "collapse ratio ramp",
            "isomer drift ramp",
            "topology cohesion drift",
            "adaptive gate actions",
        ],
    },
}


def closed_course_fixture() -> Dict[str, Any]:
    """Return a small deterministic corpus where baseline fitness is measurable, not saturated."""

    corpus = [
        "adaptive retrieval chelation variance",
        "semantic collapse adaptive retrieval",
        "near data ssd nvme storage",
        "quantization int8 gate retention",
        "adapter routing policy centroid",
        "retrieval quantization adapter route",
        "storage adapter route latency",
    ]
    payloads = [{"doc_id": f"d{i}", "track": "closed_course"} for i in range(len(corpus))]
    queries = {
        "q_adaptive": "adaptive retrieval",
        "q_storage": "ssd storage",
        "q_quantization": "quantization gate",
        "q_routing": "adapter routing",
        "q_transfer": "retrieval route",
    }
    # q_adaptive and q_transfer intentionally target lower-ranked relevant docs
    # so baseline metrics cannot saturate at 1.0.
    qrels = {
        "q_adaptive": {"1": 1},
        "q_storage": {"2": 1},
        "q_quantization": {"3": 1},
        "q_routing": {"4": 1},
        "q_transfer": {"4": 1},
    }
    return {
        "corpus": corpus,
        "payloads": payloads,
        "queries": queries,
        "qrels": qrels,
        "k": 5,
    }


def build_closed_course_engine(logger: EventCollector | None = None):
    """Create an in-memory deterministic engine loaded with the closed-course fixture."""

    fixture = closed_course_fixture()
    logger = logger or EventCollector()
    engine = _make_engine(logger)
    engine.ingest(fixture["corpus"], fixture["payloads"])
    return engine, logger, fixture


def _evaluate(rankings: Mapping[str, Sequence[Any]], qrels: Mapping[str, Mapping[str, int]], k: int):
    normalized_rankings = {
        query_id: [str(doc_id) for doc_id in doc_ids]
        for query_id, doc_ids in rankings.items()
    }
    return RetrievalFitnessEvaluator(qrels=qrels, k=k, logger=EventCollector()).evaluate_rankings(normalized_rankings)


def _fitness_metrics(evaluation) -> Dict[str, float]:
    return {
        "ndcg_at_k": float(evaluation.ndcg_at_k),
        "mrr": float(evaluation.mrr),
        "recall_at_k": float(evaluation.recall_at_k),
        "evaluated_queries": float(evaluation.evaluated_queries),
    }


def _standard_rankings(engine, queries: Mapping[str, str], k: int) -> Dict[str, list[str]]:
    rankings = {}
    for query_id, query_text in queries.items():
        query_vector = engine.embed(query_text)[0]
        hits = engine.qdrant.query_points(
            collection_name=engine.collection_name,
            query=query_vector,
            limit=k,
            with_vectors=True,
            with_payload=True,
        ).points
        rankings[query_id] = [str(hit.id) for hit in hits]
    return rankings


def _inference_rankings(engine, queries: Mapping[str, str], k: int) -> tuple[Dict[str, list[str]], list[Dict[str, Any]]]:
    rankings = {}
    diagnostics = []
    for query_id, query_text in queries.items():
        _std, final_top, _mask, _jaccard = engine.run_inference(query_text)
        rankings[query_id] = [str(doc_id) for doc_id in final_top[:k]]
        diag = dict(engine.get_last_runtime_diagnostics() or {})
        diag["query_id"] = query_id
        diagnostics.append(diag)
    return rankings, diagnostics


def freeze_initial_values(fixture: Mapping[str, Any], evaluation) -> InitialChelatedValues:
    """Freeze baseline metrics with deterministic corpus/query hashes."""

    return InitialChelatedValues(
        dataset_hash=stable_hash({"corpus": fixture["corpus"], "qrels": fixture["qrels"]}),
        query_set_hash=stable_hash(fixture["queries"]),
        corpus_size=len(fixture["corpus"]),
        query_count=len(fixture["queries"]),
        ndcg_at_k=evaluation.ndcg_at_k,
        mrr=evaluation.mrr,
        recall_at_k=evaluation.recall_at_k,
        fitness=evaluation.fitness,
        k=fixture["k"],
        metadata={"fixture": "closed_course_v1", "saturated": False},
    )


def run_closed_course_safety_testbed() -> Dict[str, Any]:
    """Run deterministic closed-course loops and return a JSON-safe report."""

    baseline_engine, baseline_logger, fixture = build_closed_course_engine()
    baseline_rankings = _standard_rankings(baseline_engine, fixture["queries"], fixture["k"])
    baseline_eval = _evaluate(baseline_rankings, fixture["qrels"], fixture["k"])
    initial_values = freeze_initial_values(fixture, baseline_eval)

    chelation_engine, chelation_logger, _fixture = build_closed_course_engine()
    chelation_engine.chelation_threshold = 0.0
    chelation_engine._stability_tracker = StabilityTracker()
    chelation_rankings, chelation_diagnostics = _inference_rankings(chelation_engine, fixture["queries"], fixture["k"])
    chelation_eval = _evaluate(chelation_rankings, fixture["qrels"], fixture["k"])

    reform_engine, reform_logger, _fixture = build_closed_course_engine()
    reform_engine._query_reformulator = QueryReformulator(logger=reform_logger)
    reform_engine._query_reformulator_max_variants = 3
    reform_rankings, reform_diagnostics = _inference_rankings(reform_engine, fixture["queries"], fixture["k"])
    reform_eval = _evaluate(reform_rankings, fixture["qrels"], fixture["k"])

    routing_engine, routing_logger, _fixture = build_closed_course_engine()
    router = AdapterRouter(logger=routing_logger)
    router.register("adaptive", [1.0, 0.35, 0.05, 0.05], routing_engine.adapter)
    router.register("storage", [0.05, 1.0, 0.25, 0.05], routing_engine.adapter)
    router.register("adapter", [0.05, 0.05, 0.15, 1.0], routing_engine.adapter)
    routing_engine._adapter_router = router
    routing_rankings, routing_diagnostics = _inference_rankings(routing_engine, fixture["queries"], fixture["k"])
    routing_eval = _evaluate(routing_rankings, fixture["qrels"], fixture["k"])
    route_effectiveness = router.get_route_effectiveness()

    adaptive_best = max(chelation_eval.fitness, reform_eval.fitness, routing_eval.fitness)
    quantized_candidate = max(baseline_eval.fitness, adaptive_best - 0.01)
    quantization = QuantizationPromotionGate(0.8, logger=EventCollector()).evaluate(
        fp32_fitness=adaptive_best,
        quantized_fitness=quantized_candidate,
        baseline_fitness=baseline_eval.fitness,
    )

    health_scorer = StructuralHealthScore(logger=EventCollector())
    healthy = health_scorer.evaluate(0.0, 0.0, 0.0)
    degraded = health_scorer.evaluate(0.55, 0.35, 0.4, metadata={"loop": 1})
    gate = AdaptiveGateOrchestrator(
        min_structural_health=0.7,
        storage_latency_sla_ms=1.0,
        logger=EventCollector(),
    )
    adaptive_gate = gate.evaluate({
        "final_fitness": adaptive_best,
        "structural_health": asdict(degraded),
        "quantization_gate": quantization.to_dict(),
        "route_effectiveness": route_effectiveness,
        "norm_drift": chelation_diagnostics[-1].get("norm_drift") if chelation_diagnostics else None,
        "runtime": chelation_diagnostics[-1].get("runtime") if chelation_diagnostics else None,
    })

    diagnostics_report = IntegratedDiagnosticsReport(
        cycle=1,
        phase="closed_course",
        retrieval_fitness=baseline_eval.fitness,
        final_fitness=adaptive_best,
        structural_health=asdict(degraded),
        quantization_gate=quantization.to_dict(),
        runtime=chelation_diagnostics[-1].get("runtime") if chelation_diagnostics else None,
        telemetry=chelation_engine.get_runtime_telemetry(),
        route_effectiveness=route_effectiveness,
        adaptive_gate=adaptive_gate.to_dict(),
    ).to_dict()

    events = baseline_logger.events + chelation_logger.events + reform_logger.events + routing_logger.events
    report = {
        "fixture": {
            "name": "closed_course_v1",
            "corpus_size": len(fixture["corpus"]),
            "query_count": len(fixture["queries"]),
            "saturated": False,
        },
        "initial_values": initial_values.to_dict(),
        "baseline": {
            "rankings": baseline_rankings,
            "fitness": baseline_eval.fitness,
            "metrics": _fitness_metrics(baseline_eval),
        },
        "chelation": {
            "rankings": chelation_rankings,
            "fitness": chelation_eval.fitness,
            "metrics": _fitness_metrics(chelation_eval),
            "action_mix": _action_mix(chelation_diagnostics),
            "diagnostics": chelation_diagnostics,
        },
        "reformulation": {
            "rankings": reform_rankings,
            "fitness": reform_eval.fitness,
            "metrics": _fitness_metrics(reform_eval),
            "diagnostics": reform_diagnostics,
        },
        "routing": {
            "rankings": routing_rankings,
            "fitness": routing_eval.fitness,
            "metrics": _fitness_metrics(routing_eval),
            "route_effectiveness": route_effectiveness,
            "diagnostics": routing_diagnostics,
        },
        "quantization": quantization.to_dict(),
        "structural_health": {
            "healthy": asdict(healthy),
            "degraded": asdict(degraded),
        },
        "adaptive_gate": adaptive_gate.to_dict(),
        "diagnostics_report": diagnostics_report,
        "dashboard_summary": summarize_events(events),
        "decision": _closed_course_decision(
            baseline_fitness=baseline_eval.fitness,
            adaptive_fitness=adaptive_best,
            quantization_passed=quantization.passed,
            gate_status=adaptive_gate.status,
        ),
    }
    return _json_safe(report)


def run_profile_calibration(profile_name: str) -> Dict[str, Any]:
    """Run one closed-course calibration profile without mutating defaults."""

    if profile_name not in CALIBRATION_PROFILES:
        raise ValueError(f"Unknown calibration profile: {profile_name}")

    controls = dict(CALIBRATION_PROFILES[profile_name])
    baseline_engine, _baseline_logger, fixture = build_closed_course_engine()
    baseline_rankings = _standard_rankings(baseline_engine, fixture["queries"], fixture["k"])
    baseline_eval = _evaluate(baseline_rankings, fixture["qrels"], fixture["k"])

    logger = EventCollector()
    candidate_engine, _candidate_logger, _fixture = build_closed_course_engine(logger)
    candidate_engine.chelation_threshold = controls["chelation_threshold"]
    candidate_engine.chelation_p = controls["chelation_p"]
    candidate_engine._stability_tracker = StabilityTracker()

    if controls["adapter_routing"]:
        router = AdapterRouter(logger=logger)
        router.register("adaptive", [1.0, 0.35, 0.05, 0.05], candidate_engine.adapter)
        router.register("storage", [0.05, 1.0, 0.25, 0.05], candidate_engine.adapter)
        router.register("adapter", [0.05, 0.05, 0.15, 1.0], candidate_engine.adapter)
        candidate_engine._adapter_router = router

    if controls["query_reformulation"]:
        candidate_engine._query_reformulator = QueryReformulator(logger=logger)
        candidate_engine._query_reformulator_max_variants = 3

    candidate_rankings, diagnostics = _inference_rankings(candidate_engine, fixture["queries"], fixture["k"])
    candidate_eval = _evaluate(candidate_rankings, fixture["qrels"], fixture["k"])
    fitness_delta = float(candidate_eval.fitness - baseline_eval.fitness)
    quantized_fitness = max(baseline_eval.fitness, candidate_eval.fitness - 0.01)
    quantization = QuantizationPromotionGate(0.8, logger=EventCollector()).evaluate(
        fp32_fitness=candidate_eval.fitness,
        quantized_fitness=quantized_fitness,
        baseline_fitness=baseline_eval.fitness,
    )
    structural_health = StructuralHealthScore(logger=EventCollector()).evaluate(0.0, 0.0, 0.0)
    adaptive_gate = AdaptiveGateOrchestrator(min_structural_health=0.7, logger=EventCollector()).evaluate({
        "final_fitness": candidate_eval.fitness,
        "structural_health": asdict(structural_health),
        "quantization_gate": quantization.to_dict(),
        "runtime": diagnostics[-1].get("runtime") if diagnostics else None,
        "norm_drift": diagnostics[-1].get("norm_drift") if diagnostics else None,
    })

    if fitness_delta >= 0.01 and quantization.passed and adaptive_gate.status != "fail":
        decision_status = "profile_candidate"
        reason = "profile_improves_closed_course_without_default_promotion"
    elif fitness_delta >= -0.01:
        decision_status = "hold"
        reason = "profile_near_baseline_or_gate_inconclusive"
    else:
        decision_status = "reject"
        reason = "profile_regresses_closed_course_fitness"

    return _json_safe({
        "profile": profile_name,
        "controls": controls,
        "baseline": {
            "fitness": baseline_eval.fitness,
            "metrics": _fitness_metrics(baseline_eval),
        },
        "candidate": {
            "fitness": candidate_eval.fitness,
            "metrics": _fitness_metrics(candidate_eval),
            "fitness_delta": fitness_delta,
            "rankings": candidate_rankings,
        },
        "gates": {
            "quantization": quantization.to_dict(),
            "adaptive": adaptive_gate.to_dict(),
            "structural_health": asdict(structural_health),
        },
        "diagnostics": diagnostics,
        "decision": {
            "status": decision_status,
            "reason": reason,
            "default_change_allowed": False,
            "rollback_condition": (
                "Rollback profile if retrieval fitness drops below frozen baseline, "
                "quantization retained gain falls below 0.8, structural health drops below 0.7, "
                "or norm ratio exits hard band."
            ),
        },
    })


def run_calibration_matrix() -> Dict[str, Any]:
    """Run conservative/balanced/aggressive/experimental profile loops."""

    profiles = [run_profile_calibration(profile) for profile in CALIBRATION_PROFILES]
    profile_candidates = [
        profile["profile"] for profile in profiles
        if profile["decision"]["status"] == "profile_candidate"
    ]
    return _json_safe({
        "schema": {
            "fields": ["profile", "controls", "baseline", "candidate", "gates", "diagnostics", "decision"],
            "default_promotion_policy": "blocked_until_repeated_campaign_evidence",
        },
        "profiles": profiles,
        "decision_report": {
            "profile_candidates": profile_candidates,
            "default_change_allowed": False,
            "recommendation": (
                "Keep defaults unchanged; profile candidates require repeated loops and road-course evidence "
                "before any preset/default promotion."
            ),
        },
    })


def build_road_course_campaign_plan() -> Dict[str, Any]:
    """Return the campaign plan and hard default-promotion gate for real benchmark runs."""

    promotion_gate = {
        "default_change_allowed": False,
        "approved_guardrail_changes": {
            "chelation_threshold": 0.01,
            "scope": "MiniLM/SciFact road-course guardrail; not an aggressive chelation promotion",
        },
        "minimum_campaigns": ["beir_small_tier", "multitask_transfer", "repeatability_matrix", "quantization_survival"],
        "required_conditions": {
            "retrieval_lift_min": 0.01,
            "retrieval_lift_target": 0.03,
            "quantized_retained_gain_min": 0.80,
            "structural_health_min": 0.70,
            "norm_ratio_hard_band": [0.50, 2.00],
            "norm_ratio_watch_band": [0.75, 1.33],
            "latency_regression_requires_warning": True,
            "all_diagnostics_json_safe": True,
        },
        "reject_conditions": [
            "retrieval fitness drops below frozen baseline by more than 0.01",
            "quantized retained gain falls below 0.80",
            "structural health falls below 0.60",
            "norm ratio exits the hard band",
            "diagnostics are malformed or omit failure status",
            "route effectiveness remains below 0.25 after enough samples",
        ],
    }
    return _json_safe({
        "campaigns": ROAD_COURSE_CAMPAIGNS,
        "promotion_gate": promotion_gate,
        "documentation_refresh": {
            "summary_doc": "docs/safety-testbed-road-course-plan.md",
            "research_tracks_update": "docs/RESEARCH_TRACKS.md",
            "verification_log": "docs/ARCH AGENTIC ENGINEERING AND PLANNING/verification-log.md",
            "required_sections": [
                "baseline metrics",
                "adaptive metrics",
                "transfer metrics",
                "quantization retention",
                "health/norm/route telemetry",
                "latency",
                "warnings/failures",
                "setting decisions",
            ],
        },
    })


def render_campaign_documentation(plan: Mapping[str, Any] | None = None) -> str:
    """Render a deterministic Markdown summary for campaign documentation refreshes."""

    plan = plan or build_road_course_campaign_plan()
    lines = [
        "# Safety Testbed Road-Course Campaign Plan",
        "",
        "The project-car testbed now has instrumentation, bench, dyno, closed-course, calibration, and ravine coverage. "
        "Road-course campaigns are the remaining evidence gate before any default/profile promotion.",
        "",
        "## Default promotion gate",
        "",
        f"- Default change allowed now: `{plan['promotion_gate']['default_change_allowed']}`",
        f"- Minimum campaigns: {', '.join(plan['promotion_gate']['minimum_campaigns'])}",
        f"- Retrieval lift: `{plan['promotion_gate']['required_conditions']['retrieval_lift_min']}` minimum, "
        f"`{plan['promotion_gate']['required_conditions']['retrieval_lift_target']}` target",
        f"- Quantized retained gain minimum: `{plan['promotion_gate']['required_conditions']['quantized_retained_gain_min']}`",
        f"- Structural health minimum: `{plan['promotion_gate']['required_conditions']['structural_health_min']}`",
        "",
        "## Campaigns",
        "",
    ]
    for name, campaign in plan["campaigns"].items():
        lines.extend([
            f"### {name}",
            "",
            campaign["objective"],
            "",
            "**Commands:**",
            "",
        ])
        lines.extend(f"- `{command}`" for command in campaign["commands"])
        lines.extend(["", "**Required evidence:**", ""])
        lines.extend(f"- {evidence}" for evidence in campaign["required_evidence"])
        lines.append("")
    lines.extend([
        "## Documentation refresh requirements",
        "",
    ])
    lines.extend(f"- {section}" for section in plan["documentation_refresh"]["required_sections"])
    lines.append("")
    return "\n".join(lines)


def _action_mix(diagnostics: Sequence[Mapping[str, Any]]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for diagnostic in diagnostics:
        action = diagnostic.get("runtime", {}).get("action", "unknown")
        counts[action] = counts.get(action, 0) + 1
    return counts


def _closed_course_decision(
    baseline_fitness: float,
    adaptive_fitness: float,
    quantization_passed: bool,
    gate_status: str,
) -> Dict[str, Any]:
    if adaptive_fitness >= baseline_fitness and quantization_passed and gate_status != "fail":
        return {
            "status": "hold",
            "reason": "closed_course_wiring_passed_without_default_promotion",
            "default_change_allowed": False,
        }
    return {
        "status": "exploratory",
        "reason": "adaptive_path_requires_more_evidence_or_gate_cleanup",
        "default_change_allowed": False,
    }


def main() -> None:
    print(json.dumps(run_closed_course_safety_testbed(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
