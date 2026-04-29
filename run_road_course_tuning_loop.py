"""Iterative road-course tuning loop for first-hundred SciFact style campaigns."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from benchmark_utils import load_mteb_data
from run_road_course_campaign import (
    DEFAULT_PROFILE_GRID,
    RoadCourseProfile,
    evaluate_profile,
    quantization_survival_check,
    select_road_course_slice,
)


QUICK_PROFILE_GRID = [
    RoadCourseProfile("baseline"),
    RoadCourseProfile("adaptive_p85_t0.0004", use_quantization=True, chelation_p=85, chelation_threshold=0.0004),
    RoadCourseProfile("adaptive_p85_t0.01", use_quantization=True, chelation_p=85, chelation_threshold=0.01),
    RoadCourseProfile("fast_guard_p85_t999", use_quantization=True, chelation_p=85, chelation_threshold=999.0),
    RoadCourseProfile("centered_p85_temp1", use_centering=True, chelation_p=85, chelation_threshold=0.0),
]


MODULE_PROFILE_GRID = [
    RoadCourseProfile("baseline"),
    RoadCourseProfile("reform_v2", query_reformulation_variants=2),
    RoadCourseProfile("reform_v3", query_reformulation_variants=3),
    RoadCourseProfile("adaptive_p85_t0.0004", use_quantization=True, chelation_p=85, chelation_threshold=0.0004),
    RoadCourseProfile("guard_p85_t0.01", use_quantization=True, chelation_p=85, chelation_threshold=0.01),
    RoadCourseProfile(
        "guard_p85_t0.01_reform_v2",
        use_quantization=True,
        chelation_p=85,
        chelation_threshold=0.01,
        query_reformulation_variants=2,
    ),
    RoadCourseProfile(
        "fast_guard_p85_t999_reform_v2",
        use_quantization=True,
        chelation_p=85,
        chelation_threshold=999.0,
        query_reformulation_variants=2,
    ),
    RoadCourseProfile("centered_p85_temp0.5", use_centering=True, chelation_p=85, chelation_threshold=0.0, temperature=0.5),
    RoadCourseProfile("centered_p85_temp2", use_centering=True, chelation_p=85, chelation_threshold=0.0, temperature=2.0),
]


CALIBRATED_PROFILE_GRID = [
    RoadCourseProfile("baseline"),
    RoadCourseProfile("reform_rrf_v2", query_reformulation_variants=2),
    RoadCourseProfile("reform_rrf_v3", query_reformulation_variants=3),
    RoadCourseProfile("adaptive_p85_t0.0004", use_quantization=True, chelation_p=85, chelation_threshold=0.0004),
    RoadCourseProfile("adaptive_p85_t0.001", use_quantization=True, chelation_p=85, chelation_threshold=0.001),
    RoadCourseProfile("adaptive_p85_t0.0015", use_quantization=True, chelation_p=85, chelation_threshold=0.0015),
    RoadCourseProfile("adaptive_p85_t0.002", use_quantization=True, chelation_p=85, chelation_threshold=0.002),
    RoadCourseProfile("adaptive_p85_t0.003", use_quantization=True, chelation_p=85, chelation_threshold=0.003),
    RoadCourseProfile("adaptive_p85_t0.004", use_quantization=True, chelation_p=85, chelation_threshold=0.004),
    RoadCourseProfile("adaptive_p50_t0.002", use_quantization=True, chelation_p=50, chelation_threshold=0.002),
    RoadCourseProfile("adaptive_p95_t0.002", use_quantization=True, chelation_p=95, chelation_threshold=0.002),
    RoadCourseProfile(
        "adaptive_p85_t0.002_reform_rrf_v2",
        use_quantization=True,
        chelation_p=85,
        chelation_threshold=0.002,
        query_reformulation_variants=2,
    ),
    RoadCourseProfile("guard_p85_t0.01", use_quantization=True, chelation_p=85, chelation_threshold=0.01),
]


def classify_profile_behavior(row: Dict[str, Any], delta_epsilon: float = 0.001) -> Dict[str, Any]:
    """Classify whether a profile is a no-op, useful actuator, or suspicious regression."""

    profile = row["profile"]
    delta = float(row.get("delta_vs_baseline", 0.0))
    action_mix = row.get("action_mix", {}) or {}
    diagnostics = row.get("control_diagnostics", {}) or {}
    chelate_count = int(action_mix.get("CHELATE", 0)) + int(action_mix.get("CHELATE_ALWAYS", 0))
    reformulate_count = int(action_mix.get("REFORMULATE", 0))
    active_count = chelate_count + reformulate_count
    jaccard = diagnostics.get("jaccard_mean")
    mask_density = diagnostics.get("mask_density_mean")
    reformulation_changed = int(diagnostics.get("reformulation_changed_count") or 0)
    ranking_changed = (
        (jaccard is not None and float(jaccard) < 0.999)
        or (mask_density is not None and float(mask_density) < 0.999)
        or reformulation_changed > 0
    )
    active = active_count > 0 or ranking_changed

    if profile == "baseline":
        fault_class = "reference"
        severity = "none"
        interpretation = "baseline reference profile"
    elif not active and abs(delta) <= delta_epsilon:
        fault_class = "no_op_tied"
        severity = "none"
        interpretation = "profile preserved baseline and did not activate a ranking-changing control"
    elif not active:
        fault_class = "metric_changed_without_actuator"
        severity = "high"
        interpretation = "metric changed without recorded actuator activity; inspect instrumentation or hidden state"
    elif delta > delta_epsilon:
        fault_class = "actuator_active_positive"
        severity = "none"
        interpretation = "ranking-changing actuator improved this window"
    elif delta < -delta_epsilon:
        fault_class = "actuator_active_negative"
        severity = "medium"
        interpretation = "ranking-changing actuator regressed this window"
    else:
        fault_class = "actuator_active_neutral"
        severity = "low"
        interpretation = "ranking-changing actuator moved results without a meaningful metric delta"

    return {
        "fault_class": fault_class,
        "severity": severity,
        "active": active,
        "chelate_count": chelate_count,
        "reformulate_count": reformulate_count,
        "ranking_changed": ranking_changed,
        "promotion_blocker": severity in {"medium", "high"},
        "interpretation": interpretation,
    }


def profile_summary(profile_results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize one profile round for tuning decisions."""

    results = list(profile_results)
    baseline = next(result for result in results if result["profile"] == "baseline")
    best = max(results, key=lambda result: result["metrics"]["ndcg_at_10"])
    rows = []
    for result in results:
        row = {
            "profile": result["profile"],
            "ndcg_at_10": result["metrics"]["ndcg_at_10"],
            "delta_vs_baseline": result["metrics"]["ndcg_at_10"] - baseline["metrics"]["ndcg_at_10"],
            "mrr": result["metrics"]["mrr"],
            "recall_at_10": result["metrics"]["recall_at_10"],
            "action_mix": result["action_mix"],
            "control_diagnostics": result.get("control_diagnostics", {}),
            "latency_ms_mean": result["latency_ms_mean"],
        }
        row["fault_classification"] = classify_profile_behavior(row)
        rows.append(row)
    rows.sort(key=lambda row: row["ndcg_at_10"], reverse=True)
    active_chelate_regressions = [
        row for row in rows
        if row["action_mix"].get("CHELATE", 0) > 0 and row["delta_vs_baseline"] < 0
    ]
    fault_counts: Dict[str, int] = {}
    promotion_blockers = []
    for row in rows:
        fault_class = row["fault_classification"]["fault_class"]
        fault_counts[fault_class] = fault_counts.get(fault_class, 0) + 1
        if row["fault_classification"]["promotion_blocker"]:
            promotion_blockers.append(row["profile"])
    return {
        "baseline_profile": baseline["profile"],
        "baseline_ndcg_at_10": baseline["metrics"]["ndcg_at_10"],
        "best_profile": best["profile"],
        "best_ndcg_at_10": best["metrics"]["ndcg_at_10"],
        "best_delta_vs_baseline": best["metrics"]["ndcg_at_10"] - baseline["metrics"]["ndcg_at_10"],
        "active_chelate_regression_count": len(active_chelate_regressions),
        "fault_counts": fault_counts,
        "promotion_blockers": promotion_blockers,
        "ranked_profiles": rows,
    }


def propose_next_profiles(summary: Dict[str, Any]) -> List[RoadCourseProfile]:
    """Adapt the next profile grid from observed first-round behavior."""

    profiles = [RoadCourseProfile("baseline")]
    if summary["active_chelate_regression_count"] > 0:
        # Prior road-course runs showed 0.0004 over-chelated MiniLM/SciFact. Explore safer guardrails.
        for threshold in (0.005, 0.01, 0.02):
            for percentile in (75, 85, 95):
                profiles.append(
                    RoadCourseProfile(
                        name=f"guard_p{percentile}_t{threshold:g}",
                        use_quantization=True,
                        chelation_p=percentile,
                        chelation_threshold=threshold,
                    )
                )
        profiles.append(RoadCourseProfile("fast_guard_p85_t999", use_quantization=True, chelation_p=85, chelation_threshold=999.0))
        profiles.extend([
            RoadCourseProfile("reform_v2", query_reformulation_variants=2),
            RoadCourseProfile(
                "guard_p85_t0.01_reform_v2",
                use_quantization=True,
                chelation_p=85,
                chelation_threshold=0.01,
                query_reformulation_variants=2,
            ),
            RoadCourseProfile(
                "fast_guard_p85_t999_reform_v2",
                use_quantization=True,
                chelation_p=85,
                chelation_threshold=999.0,
                query_reformulation_variants=2,
            ),
        ])
    else:
        # If active chelation is not regressing, probe around the best observed active threshold.
        profiles.extend([
            RoadCourseProfile("adaptive_p75_t0.001", use_quantization=True, chelation_p=75, chelation_threshold=0.001),
            RoadCourseProfile("adaptive_p85_t0.002", use_quantization=True, chelation_p=85, chelation_threshold=0.002),
            RoadCourseProfile("adaptive_p95_t0.005", use_quantization=True, chelation_p=95, chelation_threshold=0.005),
        ])
    return profiles


def run_tuning_loop(
    task: str,
    model: str,
    max_queries: int,
    sample_docs: int,
    seed: int,
    rounds: int = 2,
    initial_grid: str = "quick",
) -> Dict[str, Any]:
    """Run iterative profile evaluation and adaptive next-grid selection."""

    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    corpus, queries, qrels = load_mteb_data(task)
    sliced_corpus, sliced_queries, sliced_qrels = select_road_course_slice(
        corpus,
        queries,
        qrels,
        max_queries=max_queries,
        sample_docs=sample_docs,
        seed=seed,
    )
    grids = {
        "quick": QUICK_PROFILE_GRID,
        "full": DEFAULT_PROFILE_GRID,
        "modules": MODULE_PROFILE_GRID,
        "calibrated": CALIBRATED_PROFILE_GRID,
    }
    active_profiles = list(grids[initial_grid])
    round_results = []
    for round_index in range(1, rounds + 1):
        profile_results = [
            evaluate_profile(profile, model, sliced_corpus, sliced_queries, sliced_qrels)
            for profile in active_profiles
        ]
        summary = profile_summary(profile_results)
        round_results.append({
            "round": round_index,
            "profiles_evaluated": [asdict(profile) for profile in active_profiles],
            "summary": summary,
            "profile_results": profile_results,
        })
        active_profiles = propose_next_profiles(summary)

    final_summary = round_results[-1]["summary"]
    baseline_ndcg = final_summary["baseline_ndcg_at_10"]
    best_result = max(round_results[-1]["profile_results"], key=lambda result: result["metrics"]["ndcg_at_10"])
    quantization = quantization_survival_check(
        RoadCourseProfile(**best_result["controls"]),
        baseline_ndcg=baseline_ndcg,
        fp32_ndcg=best_result["metrics"]["ndcg_at_10"],
        model_name=model,
        corpus=sliced_corpus,
        queries=sliced_queries,
        qrels=sliced_qrels,
    )
    return {
        "task": task,
        "model": model,
        "seed": seed,
        "rounds_requested": rounds,
        "initial_grid": initial_grid,
        "query_count": len(sliced_queries),
        "corpus_size": len(sliced_corpus),
        "round_results": round_results,
        "quantization_survival": quantization,
        "recommendation": {
            "profile": final_summary["best_profile"],
            "delta_vs_baseline": final_summary["best_delta_vs_baseline"],
            "default_change_allowed": final_summary["best_delta_vs_baseline"] > 0 and quantization["gate"]["passed"],
            "reason": "requires_repeatability_and_transfer_before_promotion",
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run iterative first-hundred road-course tuning loop")
    parser.add_argument("--task", default="SciFact")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--sample-docs", type=int, default=5183)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--initial-grid", choices=["quick", "full", "modules", "calibrated"], default="quick")
    parser.add_argument("--output", default="experiment_runs/roadcourse-small/scifact_hundred_tuning_loop_seed42.json")
    args = parser.parse_args()

    result = run_tuning_loop(
        task=args.task,
        model=args.model,
        max_queries=args.max_queries,
        sample_docs=args.sample_docs,
        seed=args.seed,
        rounds=args.rounds,
        initial_grid=args.initial_grid,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output_path),
        "task": result["task"],
        "query_count": result["query_count"],
        "corpus_size": result["corpus_size"],
        "recommendation": result["recommendation"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
