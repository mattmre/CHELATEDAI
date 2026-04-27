"""
Run candidate-specific transfer gates for the Session 33 distillation candidate.

This script evaluates the exact repeatability candidate (adapter type + teacher
weight + distillation settings) across a task suite by running
benchmark_distillation.py per task and aggregating the final baseline vs hybrid
comparison.

It exists because benchmark_multitask.py and benchmark_beir.py evaluate the
repository's generic chelation/comparative configurations, not the focused
distillation candidate that cleared the Session 33 repeatability gate.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Tuple

from benchmark_beir import BEIRDatasetRegistry
from run_repeatability_check import (
    PROJECT_ROOT,
    build_run_dir,
    format_command,
    load_results,
    run_with_tee,
    _extract_final_ndcg,
)


MULTITASK_SUITES = {
    "small": ["SciFact", "NFCorpus"],
    "medium": ["SciFact", "NFCorpus", "FiQA2018"],
}

TRANSFER_SCOPES = ("small", "medium")


def parse_reuse_result(spec: str) -> Tuple[str, Path]:
    if "=" not in spec:
        raise ValueError(
            f"Invalid --reuse-result '{spec}'. Expected TASK=PATH."
        )

    task_name, raw_path = spec.split("=", 1)
    task_name = task_name.strip()
    raw_path = raw_path.strip()

    if not task_name or not raw_path:
        raise ValueError(
            f"Invalid --reuse-result '{spec}'. Expected non-empty TASK=PATH."
        )

    return task_name, Path(raw_path)


def resolve_tasks(gate: str, scope: str) -> List[str]:
    if gate == "multitask":
        if scope not in MULTITASK_SUITES:
            valid = ", ".join(sorted(MULTITASK_SUITES))
            raise ValueError(f"Unknown multitask scope '{scope}'. Valid scopes: {valid}")
        return MULTITASK_SUITES[scope]

    if scope not in TRANSFER_SCOPES:
        valid = ", ".join(TRANSFER_SCOPES)
        raise ValueError(f"Unknown BEIR scope '{scope}'. Valid scopes: {valid}")

    return [dataset.name for dataset in BEIRDatasetRegistry.get_tier_datasets(scope)]


def build_distillation_command(task_name: str, output_path: Path, args: argparse.Namespace) -> List[str]:
    command = [
        sys.executable,
        "-u",
        "benchmark_distillation.py",
        "--task",
        task_name,
        "--model",
        args.model,
        "--teacher",
        args.teacher,
        "--cycles",
        str(args.cycles),
        "--queries-per-cycle",
        str(args.queries_per_cycle),
        "--epochs",
        str(args.epochs),
        "--lr",
        str(args.learning_rate),
        "--max-eval-queries",
        str(args.max_eval_queries),
        "--teacher-weight",
        str(args.teacher_weight),
        "--threshold",
        str(args.threshold),
        "--adapter-type",
        args.adapter_type,
        "--seed",
        str(getattr(args, "seed", 0)),
        "--output",
        str(output_path),
    ]
    if getattr(args, "sedimentation_optimizer", "adam") != "adam":
        command.extend(["--sedimentation-optimizer", args.sedimentation_optimizer])
    if getattr(args, "es_retrieval_fitness", False):
        command.append("--es-retrieval-fitness")
    if getattr(args, "quantization_gate", False):
        command.append("--quantization-gate")
    if getattr(args, "es_antithetic_sampling", False):
        command.append("--es-antithetic-sampling")
    if getattr(args, "es_rollback_to_elite", False):
        command.append("--es-rollback-to-elite")
    if getattr(args, "es_quantization_aware", False):
        command.append("--es-quantization-aware")
    if getattr(args, "es_kalman_sigma", False):
        command.append("--es-kalman-sigma")
    optional_pairs = [
        ("--es-population-size", "es_population_size"),
        ("--es-rank", "es_rank"),
        ("--es-sigma", "es_sigma"),
        ("--es-generations", "es_generations"),
        ("--es-elite-pool-size", "es_elite_pool_size"),
        ("--es-fitness-shaping", "es_fitness_shaping"),
        ("--es-storage-profile", "es_storage_profile"),
        ("--quantization-gate-threshold", "quantization_gate_threshold"),
        ("--structural-health-weight", "structural_health_weight"),
        ("--query-reformulation-variants", "query_reformulation_variants"),
    ]
    for flag, attr in optional_pairs:
        value = getattr(args, attr, None)
        if value is not None:
            command.extend([flag, str(value)])
    return command


def extract_quantization_gate_status(results: Dict[str, Any]) -> Dict[str, Any]:
    """Extract ES quantization gate failures from benchmark results."""

    statuses = []
    for mode in ("baseline", "hybrid"):
        cycles = results.get(mode, [])
        if isinstance(cycles, list):
            for cycle in cycles:
                gate = cycle.get("es_result", {}).get("quantization_gate") if isinstance(cycle, dict) else None
                if gate is not None:
                    statuses.append({"mode": mode, "cycle": cycle.get("cycle"), **gate})
    offline_cycles = results.get("offline", {}).get("cycles", {})
    if isinstance(offline_cycles, list):
        for cycle in offline_cycles:
            gate = cycle.get("es_result", {}).get("quantization_gate") if isinstance(cycle, dict) else None
            if gate is not None:
                statuses.append({"mode": "offline", "cycle": cycle.get("cycle"), **gate})
    failed = [status for status in statuses if not status.get("passed", False)]
    return {
        "observed": statuses,
        "failed": failed,
        "passes_quantization_gate": bool(statuses) and not failed,
    }


def summarize_task_result(
    task_name: str,
    results_path: Path,
    reused: bool,
    min_task_gain: float,
    require_quantization_gate: bool = False,
) -> Dict[str, Any]:
    try:
        results = load_results(results_path)
        baseline_final = _extract_final_ndcg(results, "baseline")
        offline_final = _extract_final_ndcg(results, "offline")
        hybrid_final = _extract_final_ndcg(results, "hybrid")
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        return {
            "task": task_name,
            "results_path": str(results_path),
            "reused": reused,
            "baseline_final_ndcg": 0.0,
            "offline_final_ndcg": 0.0,
            "hybrid_final_ndcg": 0.0,
            "hybrid_gain_absolute": 0.0,
            "hybrid_gain_pct": 0.0,
            "passes_task_gate": False,
            "quantization_gate": {
                "observed": [],
                "failed": [],
                "passes_quantization_gate": False,
            },
            "error": f"Failed to summarize results: {exc}",
        }
    hybrid_gain = hybrid_final - baseline_final
    hybrid_gain_pct = (hybrid_gain / baseline_final * 100.0) if baseline_final else 0.0

    quantization_gate = extract_quantization_gate_status(results)
    passes_task_gate = hybrid_gain >= min_task_gain
    if require_quantization_gate:
        passes_task_gate = passes_task_gate and quantization_gate["passes_quantization_gate"]
    return {
        "task": task_name,
        "results_path": str(results_path),
        "reused": reused,
        "baseline_final_ndcg": baseline_final,
        "offline_final_ndcg": offline_final,
        "hybrid_final_ndcg": hybrid_final,
        "hybrid_gain_absolute": hybrid_gain,
        "hybrid_gain_pct": hybrid_gain_pct,
        "passes_task_gate": passes_task_gate,
        "quantization_gate": quantization_gate,
    }


def build_transfer_summary(
    gate: str,
    scope: str,
    task_summaries: List[Dict[str, Any]],
    min_task_gain: float,
) -> Dict[str, Any]:
    baseline_scores = [task["baseline_final_ndcg"] for task in task_summaries]
    hybrid_scores = [task["hybrid_final_ndcg"] for task in task_summaries]
    gain_scores = [task["hybrid_gain_absolute"] for task in task_summaries]
    failed_tasks = [task["task"] for task in task_summaries if not task["passes_task_gate"]]

    passes_transfer_gate = not failed_tasks and mean(gain_scores) >= min_task_gain

    if gate == "multitask":
        recommended_next_step = "run-medium-transfer" if scope == "small" and passes_transfer_gate else (
            "run-beir-transfer" if scope == "medium" and passes_transfer_gate else "stop-and-review"
        )
    else:
        recommended_next_step = "promotion-review" if passes_transfer_gate else "stop-and-review"

    return {
        "gate": gate,
        "scope": scope,
        "tasks": [task["task"] for task in task_summaries],
        "min_task_gain": min_task_gain,
        "task_summaries": task_summaries,
        "aggregate": {
            "mean_baseline_final_ndcg": mean(baseline_scores),
            "mean_hybrid_final_ndcg": mean(hybrid_scores),
            "mean_hybrid_gain_absolute": mean(gain_scores),
            "positive_gains": sum(1 for gain in gain_scores if gain > 0),
            "non_positive_gains": sum(1 for gain in gain_scores if gain <= 0),
        },
        "failed_tasks": failed_tasks,
        "passes_transfer_gate": passes_transfer_gate,
        "recommended_next_step": recommended_next_step,
    }


def ensure_parent_directory(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run candidate-specific multitask/BEIR transfer gates"
    )
    parser.add_argument("--gate", choices=["multitask", "beir"], required=True)
    parser.add_argument("--scope", choices=TRANSFER_SCOPES, required=True)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--teacher", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--queries-per-cycle", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--max-eval-queries", type=int, default=100)
    parser.add_argument("--teacher-weight", type=float, default=0.3)
    parser.add_argument("--threshold", type=int, default=1)
    parser.add_argument("--adapter-type", choices=["mlp", "procrustes", "low_rank"], default="mlp")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--run-label",
        default="session33-candidate-transfer",
        help="Suffix for the transfer gate run directory",
    )
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / "experiment_runs"),
        help="Directory under which the transfer gate run directory will be created",
    )
    parser.add_argument(
        "--reuse-result",
        action="append",
        default=[],
        help="Reuse an existing benchmark_distillation results.json via TASK=PATH",
    )
    parser.add_argument(
        "--min-task-gain",
        type=float,
        default=0.0,
        help="Minimum allowed hybrid - baseline gain per task",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 2 if the transfer gate does not pass",
    )
    parser.add_argument("--sedimentation-optimizer", choices=["adam", "eggroll_es"], default="adam")
    parser.add_argument("--es-retrieval-fitness", action="store_true")
    parser.add_argument("--es-population-size", type=int, default=8)
    parser.add_argument("--es-rank", type=int, default=1)
    parser.add_argument("--es-sigma", type=float, default=0.01)
    parser.add_argument("--es-generations", type=int, default=None)
    parser.add_argument("--es-quantization-aware", action="store_true")
    parser.add_argument("--es-kalman-sigma", action="store_true")
    parser.add_argument("--es-elite-pool-size", type=int, default=3)
    parser.add_argument("--es-rollback-to-elite", action="store_true")
    parser.add_argument("--es-antithetic-sampling", action="store_true")
    parser.add_argument("--es-fitness-shaping", choices=["zscore", "centered", "linear_rank"], default="zscore")
    parser.add_argument("--es-storage-profile", choices=["rp2040", "consumer_nvme", "smartssd", "dpu_storage"], default=None)
    parser.add_argument("--quantization-gate", action="store_true")
    parser.add_argument("--quantization-gate-threshold", type=float, default=0.8)
    parser.add_argument("--structural-health-weight", type=float, default=0.0)
    parser.add_argument("--query-reformulation-variants", type=int, default=1)
    parser.add_argument(
        "--require-quantization-gate",
        action="store_true",
        help="Require observed ES quantization gates to pass for every task",
    )
    args = parser.parse_args()

    tasks = resolve_tasks(args.gate, args.scope)
    reuse_results = dict(parse_reuse_result(spec) for spec in args.reuse_result)

    run_dir = build_run_dir(Path(args.output_root), f"{args.gate}-{args.scope}-{args.run_label}")
    run_dir.mkdir(parents=True, exist_ok=True)

    task_summaries: List[Dict[str, Any]] = []

    print("=" * 60)
    print("Session 33 Candidate Transfer Gate")
    print("=" * 60)
    print(f"Gate: {args.gate}")
    print(f"Scope: {args.scope}")
    print(f"Tasks: {', '.join(tasks)}")
    print(f"Run directory: {run_dir}")
    print(f"Teacher weight: {args.teacher_weight}")
    print(f"Adapter type: {args.adapter_type}")
    print("=" * 60)

    for task_name in tasks:
        if task_name in reuse_results:
            results_path = reuse_results[task_name]
            print(f"\nReusing {task_name} results from: {results_path}")
            task_summaries.append(
                summarize_task_result(
                    task_name,
                    results_path,
                    reused=True,
                    min_task_gain=args.min_task_gain,
                    require_quantization_gate=args.require_quantization_gate,
                )
            )
            continue

        task_dir = run_dir / task_name.lower()
        task_dir.mkdir(parents=True, exist_ok=True)
        results_path = task_dir / "results.json"
        log_path = task_dir / "benchmark_distillation.log"
        command_path = task_dir / "command.txt"

        command = build_distillation_command(task_name, results_path, args)
        command_path.write_text(format_command(command) + "\n", encoding="utf-8")

        print(f"\nRunning candidate transfer task: {task_name}")
        return_code = run_with_tee(command, log_path, PROJECT_ROOT)
        if return_code != 0:
            print(f"ERROR: benchmark_distillation.py failed for task {task_name} with exit code {return_code}")
            return return_code

        if not results_path.exists():
            print(f"ERROR: Missing results for task {task_name}: {results_path}")
            return 1

        task_summaries.append(
            summarize_task_result(
                task_name,
                results_path,
                reused=False,
                min_task_gain=args.min_task_gain,
                require_quantization_gate=args.require_quantization_gate,
            )
        )

    summary = build_transfer_summary(args.gate, args.scope, task_summaries, args.min_task_gain)
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print("Transfer Gate Summary")
    print("=" * 60)
    print(json.dumps(summary, indent=2))

    if args.strict and not summary["passes_transfer_gate"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
