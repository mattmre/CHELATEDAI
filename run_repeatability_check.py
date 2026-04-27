"""
Run a focused repeatability check for the single locally positive Session 32
distillation candidate.

This wraps benchmark_distillation.py with the exact Session 32 candidate
configuration:

- task: SciFact
- model: sentence-transformers/all-MiniLM-L6-v2
- teacher: sentence-transformers/all-mpnet-base-v2
- adapter_type: mlp
- teacher_weight: 0.3
- threshold: 1
- learning rate: 0.01
- cycles: 5
- queries per cycle: 50
- epochs: 5

The helper creates a dedicated run directory, captures the invoked command,
tees benchmark output to both console and a durable log, and writes a compact
summary.json describing whether the rerun cleared the repeatability gate.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from datetime import datetime
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, Optional

from reproducibility_context import build_seed_matrix, evaluate_seed_scores


PROJECT_ROOT = Path(__file__).resolve().parent
MAX_RUN_LABEL_LENGTH = 64
DEFAULT_REFERENCE_BASELINE_FINAL = 0.6012
DEFAULT_REFERENCE_HYBRID_FINAL = 0.6239
DEFAULT_REFERENCE_BASELINE_TOLERANCE = 0.03


def _normalize_run_label(label: str) -> str:
    normalized = "".join(ch.lower() if ch.isalnum() else "-" for ch in label).strip("-")
    while "--" in normalized:
        normalized = normalized.replace("--", "-")
    normalized = normalized[:MAX_RUN_LABEL_LENGTH].strip("-")
    return normalized or "repeatability"


def build_run_dir(
    output_root: Path,
    run_label: str,
    now: Optional[datetime] = None,
) -> Path:
    timestamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S-%f")
    return output_root / f"repeatability-{timestamp}-{_normalize_run_label(run_label)}"


def build_command(output_path: Path, args: argparse.Namespace) -> list[str]:
    command = [
        sys.executable,
        "-u",
        "benchmark_distillation.py",
        "--task",
        args.task,
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
    optional_pairs = [
        ("--es-population-size", "es_population_size"),
        ("--es-rank", "es_rank"),
        ("--es-sigma", "es_sigma"),
        ("--es-generations", "es_generations"),
        ("--es-fitness-shaping", "es_fitness_shaping"),
        ("--quantization-gate-threshold", "quantization_gate_threshold"),
        ("--structural-health-weight", "structural_health_weight"),
        ("--es-storage-profile", "es_storage_profile"),
        ("--query-reformulation-variants", "query_reformulation_variants"),
    ]
    for flag, attr in optional_pairs:
        value = getattr(args, attr, None)
        if value is not None:
            command.extend([flag, str(value)])
    return command


def format_command(command: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def load_results(output_path: Path) -> Dict[str, Any]:
    with output_path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _extract_final_ndcg(results: Dict[str, Any], mode: str) -> float:
    if mode == "offline":
        cycles = results.get("offline", {}).get("cycles")
    else:
        cycles = results.get(mode)

    if not isinstance(cycles, list) or not cycles:
        raise ValueError(
            f"Results are incomplete: missing completed cycle data for '{mode}'. "
            "Check benchmark_distillation.log for the failing phase."
        )

    final_cycle = cycles[-1]
    if not isinstance(final_cycle, dict) or "ndcg" not in final_cycle:
        raise ValueError(
            f"Results are incomplete: final cycle for '{mode}' does not contain 'ndcg'. "
            "Check benchmark_distillation.log for the failing phase."
        )

    return float(final_cycle["ndcg"])


def build_summary(
    results: Dict[str, Any],
    output_path: Path,
    log_path: Path,
    command: list[str],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    baseline_final = _extract_final_ndcg(results, "baseline")
    offline_final = _extract_final_ndcg(results, "offline")
    hybrid_final = _extract_final_ndcg(results, "hybrid")
    hybrid_gain = hybrid_final - baseline_final
    hybrid_gain_pct = (hybrid_gain / baseline_final * 100.0) if baseline_final else 0.0

    baseline_matches_reference_band = (
        abs(baseline_final - args.reference_baseline_final)
        <= args.reference_baseline_tolerance
    )
    hybrid_beats_same_run_baseline = hybrid_final > baseline_final
    hybrid_vs_session32 = hybrid_final - args.reference_hybrid_final
    passes_repeatability_gate = (
        baseline_matches_reference_band and hybrid_beats_same_run_baseline
    )

    return {
        "output_path": str(output_path),
        "log_path": str(log_path),
        "command": command,
        "baseline_final_ndcg": baseline_final,
        "offline_final_ndcg": offline_final,
        "hybrid_final_ndcg": hybrid_final,
        "hybrid_gain_absolute": hybrid_gain,
        "hybrid_gain_pct": hybrid_gain_pct,
        "reference_baseline_final_ndcg": args.reference_baseline_final,
        "reference_hybrid_final_ndcg": args.reference_hybrid_final,
        "reference_baseline_tolerance": args.reference_baseline_tolerance,
        "baseline_matches_reference_band": baseline_matches_reference_band,
        "hybrid_beats_same_run_baseline": hybrid_beats_same_run_baseline,
        "hybrid_minus_session32_reference": hybrid_vs_session32,
        "passes_repeatability_gate": passes_repeatability_gate,
        "recommended_next_step": (
            "run-multitask-gate" if passes_repeatability_gate else "stop-and-review"
        ),
    }


def build_multi_seed_summary(seed_summaries: list[Dict[str, Any]], tolerance: float) -> Dict[str, Any]:
    """Aggregate repeatability summaries across seed runs."""

    hybrid_scores = [summary["hybrid_final_ndcg"] for summary in seed_summaries]
    gate = evaluate_seed_scores(hybrid_scores, tolerance)
    all_repeatable = all(summary["passes_repeatability_gate"] for summary in seed_summaries)
    return {
        "seed_summaries": seed_summaries,
        "multi_seed_gate": gate.to_dict(),
        "passes_repeatability_gate": all_repeatable and gate.passed,
        "recommended_next_step": "run-multitask-gate" if all_repeatable and gate.passed else "stop-and-review",
    }


def run_with_tee(command: list[str], log_path: Path, cwd: Path) -> int:
    with log_path.open("w", encoding="utf-8") as log_handle:
        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUTF8", "1")

        with subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        ) as process:
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="")
                log_handle.write(line)
                log_handle.flush()

            return process.wait()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a focused repeatability check for the Session 32 mlp/tw0.3 candidate"
    )
    parser.add_argument("--task", type=str, default="SciFact")
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--teacher",
        type=str,
        default="sentence-transformers/all-mpnet-base-v2",
    )
    parser.add_argument("--cycles", type=int, default=5)
    parser.add_argument("--queries-per-cycle", type=int, default=50)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--max-eval-queries", type=int, default=100)
    parser.add_argument("--teacher-weight", type=float, default=0.3)
    parser.add_argument("--threshold", type=int, default=1)
    parser.add_argument(
        "--adapter-type",
        type=str,
        default="mlp",
        choices=["mlp", "procrustes", "low_rank"],
    )
    parser.add_argument(
        "--run-label",
        type=str,
        default="session33-mlp-tw03",
        help="Suffix for the repeatability run directory",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default=str(PROJECT_ROOT / "experiment_runs"),
        help="Directory under which the repeatability run directory will be created",
    )
    parser.add_argument(
        "--reference-baseline-final",
        type=float,
        default=DEFAULT_REFERENCE_BASELINE_FINAL,
        help="Reference final baseline NDCG@10 from Session 32",
    )
    parser.add_argument(
        "--reference-hybrid-final",
        type=float,
        default=DEFAULT_REFERENCE_HYBRID_FINAL,
        help="Reference final hybrid NDCG@10 from Session 32",
    )
    parser.add_argument(
        "--reference-baseline-tolerance",
        type=float,
        default=DEFAULT_REFERENCE_BASELINE_TOLERANCE,
        help="Allowed absolute deviation from the Session 32 baseline before flagging drift",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Exit with code 2 if the rerun does not clear the repeatability gate",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seed-count", type=int, default=1)
    parser.add_argument("--seed-tolerance", type=float, default=0.03)
    parser.add_argument("--sedimentation-optimizer", choices=["adam", "eggroll_es"], default="adam")
    parser.add_argument("--es-retrieval-fitness", action="store_true")
    parser.add_argument("--es-population-size", type=int, default=8)
    parser.add_argument("--es-rank", type=int, default=1)
    parser.add_argument("--es-sigma", type=float, default=0.01)
    parser.add_argument("--es-generations", type=int, default=None)
    parser.add_argument("--es-antithetic-sampling", action="store_true")
    parser.add_argument("--es-rollback-to-elite", action="store_true")
    parser.add_argument("--es-fitness-shaping", choices=["zscore", "centered", "linear_rank"], default="zscore")
    parser.add_argument("--es-storage-profile", choices=["rp2040", "consumer_nvme", "smartssd", "dpu_storage"], default=None)
    parser.add_argument("--quantization-gate", action="store_true")
    parser.add_argument("--quantization-gate-threshold", type=float, default=0.8)
    parser.add_argument("--structural-health-weight", type=float, default=0.0)
    parser.add_argument("--query-reformulation-variants", type=int, default=1)
    args = parser.parse_args()

    run_dir = build_run_dir(Path(args.output_root), args.run_label)
    run_dir.mkdir(parents=True, exist_ok=True)

    output_path = run_dir / "results.json"
    log_path = run_dir / "benchmark_distillation.log"
    command_path = run_dir / "command.txt"
    summary_path = run_dir / "summary.json"

    seeds = build_seed_matrix(args.seed, args.seed_count)
    command = build_command(output_path, args)
    formatted_command = format_command(command)
    command_path.write_text(formatted_command + "\n", encoding="utf-8")

    print("=" * 60)
    print("Session 33 Focused Repeatability Check")
    print("=" * 60)
    print(f"Run directory: {run_dir}")
    print(f"Benchmark log: {log_path}")
    print(f"Reference baseline final NDCG@10: {args.reference_baseline_final:.4f}")
    print(f"Reference hybrid final NDCG@10:   {args.reference_hybrid_final:.4f}")
    print(f"Reference baseline tolerance:      +/- {args.reference_baseline_tolerance:.4f}")
    print(f"Strict gate mode: {'yes' if args.strict else 'no'}")
    print("=" * 60)
    print()
    print("Launching:")
    print(formatted_command)
    print()

    seed_summaries = []
    for seed_index, seed in enumerate(seeds):
        args.seed = seed
        seed_output_path = output_path if len(seeds) == 1 else run_dir / f"results_seed_{seed_index}.json"
        seed_log_path = log_path if len(seeds) == 1 else run_dir / f"benchmark_distillation_seed_{seed_index}.log"
        seed_command = build_command(seed_output_path, args)
        return_code = run_with_tee(seed_command, seed_log_path, PROJECT_ROOT)
        if return_code != 0:
            return return_code
        if not seed_output_path.exists():
            print(f"ERROR: benchmark completed but output is missing: {seed_output_path}")
            return 1
        try:
            results = load_results(seed_output_path)
            seed_summary = build_summary(results, seed_output_path, seed_log_path, seed_command, args)
            seed_summary["seed"] = seed
            seed_summaries.append(seed_summary)
        except JSONDecodeError as exc:
            print(
                "ERROR: benchmark output JSON is malformed. "
                f"Check {seed_output_path} and {seed_log_path} for details.\n{exc}"
            )
            return 1
        except ValueError as exc:
            print(f"ERROR: {exc}")
            return 1

    if len(seed_summaries) == 1:
        summary = seed_summaries[0]
        summary["multi_seed_gate"] = evaluate_seed_scores(
            [summary["hybrid_final_ndcg"]], args.seed_tolerance
        ).to_dict()
    else:
        summary = build_multi_seed_summary(seed_summaries, args.seed_tolerance)

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print()
    print("=" * 60)
    print("Repeatability Summary")
    print("=" * 60)
    print(json.dumps(summary, indent=2))

    if args.strict and not summary["passes_repeatability_gate"]:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
