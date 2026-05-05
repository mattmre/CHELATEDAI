"""Autonomous first-pass loop for learned query-reformulation gates."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence

from benchmark_utils import isolated_adapter_state, load_mteb_data
from learned_reformulation_gate import (
    enrich_query_rows,
    filter_reformulation_rows,
    load_query_attribution_rows,
    train_reformulation_gate,
    write_gate_config,
)
from run_road_course_campaign import RoadCourseProfile, select_road_course_slice
from run_road_course_tuning_loop import profile_summary
from run_thousand_query_tuning import evaluate_profile_with_cache


DEFAULT_ARTIFACTS = [
    Path("experiment_runs/roadcourse-small/attribution_probe_100.json"),
    Path("experiment_runs/roadcourse-small/selective_reform_probe_100.json"),
    Path("experiment_runs/roadcourse-small/reform_policy_probe_100.json"),
]


@dataclass(frozen=True)
class ValidationSpec:
    task: str
    seed: int
    query_offset: int = 0


DEFAULT_VALIDATION_SPECS = [
    ValidationSpec("SciFact", seed=170, query_offset=0),
    ValidationSpec("NFCorpus", seed=171, query_offset=0),
    ValidationSpec("FiQA2018", seed=172, query_offset=0),
]


def _existing_default_artifacts() -> List[Path]:
    return [path for path in DEFAULT_ARTIFACTS if path.exists()]


def _validation_profiles(gate: Dict[str, Any] | None) -> List[RoadCourseProfile]:
    return [
        RoadCourseProfile("baseline"),
        RoadCourseProfile("reform_rrf_v2", query_reformulation_variants=2, query_reformulation_policy="always"),
        RoadCourseProfile(
            "learned_reform_gate_v1",
            query_reformulation_variants=2,
            query_reformulation_policy=gate if gate is not None else "never",
        ),
    ]


def run_validation(
    gate: Dict[str, Any] | None,
    *,
    model: str,
    specs: Sequence[ValidationSpec],
    max_queries: int,
    sample_docs: int,
) -> Dict[str, Any]:
    windows = []
    for loop_index, spec in enumerate(specs, start=1):
        corpus, queries, qrels = load_mteb_data(spec.task)
        selected_corpus, selected_queries, selected_qrels = select_road_course_slice(
            corpus,
            queries,
            qrels,
            max_queries=max_queries,
            sample_docs=sample_docs,
            seed=spec.seed,
        )
        profiles = _validation_profiles(gate)
        with isolated_adapter_state():
            engine_cache: Dict[str, Any] = {}
            try:
                profile_results = [
                    evaluate_profile_with_cache(
                        profile,
                        model,
                        selected_corpus,
                        selected_queries,
                        selected_qrels,
                        engine_cache,
                    )
                    for profile in profiles
                ]
            finally:
                for engine in engine_cache.values():
                    engine.close()
        summary = profile_summary(profile_results)
        windows.append({
            "loop": loop_index,
            "task": spec.task,
            "seed": spec.seed,
            "query_offset": spec.query_offset,
            "query_count": len(selected_queries),
            "corpus_size": len(selected_corpus),
            "profiles_evaluated": [asdict(profile) for profile in profiles],
            "summary": summary,
        })

    baseline_deltas = []
    always_deltas = []
    learned_deltas = []
    learned_blockers = 0
    always_blockers = 0
    for window in windows:
        rows = {
            row["profile"]: row
            for row in window["summary"]["ranked_profiles"]
        }
        always = rows["reform_rrf_v2"]
        learned = rows["learned_reform_gate_v1"]
        always_deltas.append(float(always["delta_vs_baseline"]))
        learned_deltas.append(float(learned["delta_vs_baseline"]))
        baseline_deltas.append(0.0)
        always_blockers += int(always["fault_classification"]["promotion_blocker"])
        learned_blockers += int(learned["fault_classification"]["promotion_blocker"])
    return {
        "model": model,
        "max_queries": max_queries,
        "sample_docs": sample_docs,
        "windows": windows,
        "aggregate": {
            "baseline_mean_delta_vs_baseline": 0.0,
            "always_on_reform_mean_delta_vs_baseline": (
                sum(always_deltas) / len(always_deltas) if always_deltas else 0.0
            ),
            "learned_gate_mean_delta_vs_baseline": (
                sum(learned_deltas) / len(learned_deltas) if learned_deltas else 0.0
            ),
            "always_on_reform_promotion_blockers": always_blockers,
            "learned_gate_promotion_blockers": learned_blockers,
            "learned_gate_beats_always_on": sum(
                learned > always for learned, always in zip(learned_deltas, always_deltas)
            ),
            "learned_gate_beats_baseline": sum(
                learned > 0.001 for learned in learned_deltas
            ),
        },
    }


def run_autopilot(
    *,
    artifacts: Sequence[str | Path],
    profile: str,
    gate_output: str | Path | None,
    model: str,
    max_queries: int,
    sample_docs: int,
) -> Dict[str, Any]:
    rows = enrich_query_rows(load_query_attribution_rows(artifacts))
    filtered = filter_reformulation_rows(rows, profile=profile)
    gate_config = train_reformulation_gate(filtered)
    gate_config["profile"] = profile
    gate_config["artifact_count"] = len(artifacts)
    gate_config["pooled_rows"] = len(filtered)
    if gate_output is not None:
        write_gate_config(gate_config, gate_output)
    validation = run_validation(
        gate_config.get("gate"),
        model=model,
        specs=DEFAULT_VALIDATION_SPECS,
        max_queries=max_queries,
        sample_docs=sample_docs,
    )
    recommendation = {
        "gate_present": gate_config.get("gate") is not None,
        "default_change_allowed": False,
        "next_action": (
            "collect more reformulation-positive query examples"
            if gate_config.get("gate") is None
            else "run broader repeatability and transfer before any default change"
        ),
    }
    if gate_config.get("gate") is not None:
        aggregate = validation["aggregate"]
        recommendation["gate_candidate"] = (
            aggregate["learned_gate_mean_delta_vs_baseline"] > 0.001
            and aggregate["learned_gate_promotion_blockers"] == 0
        )
    return {
        "policy": "learned_reformulation_autopilot_v1",
        "artifacts": [str(Path(path)) for path in artifacts],
        "training": gate_config,
        "validation": validation,
        "recommendation": recommendation,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run a first-pass learned reformulation autopilot iteration")
    parser.add_argument("--artifact", action="append", default=None)
    parser.add_argument("--profile", default="reform_rrf_v2")
    parser.add_argument("--gate-output", default="experiment_runs/roadcourse-small/learned_reform_gate_v1.json")
    parser.add_argument("--output", default="experiment_runs/roadcourse-small/learned_reform_autopilot_v1.json")
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--max-queries", type=int, default=50)
    parser.add_argument("--sample-docs", type=int, default=200)
    args = parser.parse_args()

    artifacts = [Path(path) for path in args.artifact] if args.artifact else _existing_default_artifacts()
    if not artifacts:
        raise ValueError("no artifacts supplied and default artifact set was not found")
    result = run_autopilot(
        artifacts=artifacts,
        profile=args.profile,
        gate_output=args.gate_output,
        model=args.model,
        max_queries=args.max_queries,
        sample_docs=args.sample_docs,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output_path),
        "gate_present": result["training"]["gate"] is not None,
        "learned_gate_mean_delta_vs_baseline": (
            result["validation"]["aggregate"]["learned_gate_mean_delta_vs_baseline"]
        ),
        "always_on_reform_mean_delta_vs_baseline": (
            result["validation"]["aggregate"]["always_on_reform_mean_delta_vs_baseline"]
        ),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
