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
import subprocess
import sys
from datetime import datetime
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Dict, Optional


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
    return [
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
        "--output",
        str(output_path),
    ]


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


def run_with_tee(command: list[str], log_path: Path, cwd: Path) -> int:
    with log_path.open("w", encoding="utf-8") as log_handle:
        env = os.environ.copy()
        env.setdefault("PYTHONIOENCODING", "utf-8")
        env.setdefault("PYTHONUTF8", "1")

        process = subprocess.Popen(
            command,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log_handle.write(line)

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
    args = parser.parse_args()

    run_dir = build_run_dir(Path(args.output_root), args.run_label)
    run_dir.mkdir(parents=True, exist_ok=True)

    output_path = run_dir / "results.json"
    log_path = run_dir / "benchmark_distillation.log"
    command_path = run_dir / "command.txt"
    summary_path = run_dir / "summary.json"

    command = build_command(output_path, args)
    command_path.write_text(" ".join(command) + "\n", encoding="utf-8")

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
    print(" ".join(command))
    print()

    return_code = run_with_tee(command, log_path, PROJECT_ROOT)
    if return_code != 0:
        return return_code

    if not output_path.exists():
        print(f"ERROR: benchmark completed but output is missing: {output_path}")
        return 1

    try:
        results = load_results(output_path)
        summary = build_summary(results, output_path, log_path, command, args)
    except JSONDecodeError as exc:
        print(
            "ERROR: benchmark output JSON is malformed. "
            f"Check {output_path} and {log_path} for details.\n{exc}"
        )
        return 1
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return 1

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
