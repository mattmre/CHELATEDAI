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
    return [
        "python",
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
        "--output",
        str(output_path),
    ]


def summarize_task_result(task_name: str, results_path: Path, reused: bool, min_task_gain: float) -> Dict[str, Any]:
    results = load_results(results_path)
    baseline_final = _extract_final_ndcg(results, "baseline")
    offline_final = _extract_final_ndcg(results, "offline")
    hybrid_final = _extract_final_ndcg(results, "hybrid")
    hybrid_gain = hybrid_final - baseline_final
    hybrid_gain_pct = (hybrid_gain / baseline_final * 100.0) if baseline_final else 0.0

    return {
        "task": task_name,
        "results_path": str(results_path),
        "reused": reused,
        "baseline_final_ndcg": baseline_final,
        "offline_final_ndcg": offline_final,
        "hybrid_final_ndcg": hybrid_final,
        "hybrid_gain_absolute": hybrid_gain,
        "hybrid_gain_pct": hybrid_gain_pct,
        "passes_task_gate": hybrid_gain >= min_task_gain,
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
                summarize_task_result(task_name, results_path, reused=True, min_task_gain=args.min_task_gain)
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
            summarize_task_result(task_name, results_path, reused=False, min_task_gain=args.min_task_gain)
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
