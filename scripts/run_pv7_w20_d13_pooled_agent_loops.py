"""CLI runner for Wave 20 / Bundle #62 / Item 4 / Demo D13 Pooled-Agent Loops.

Supports two execution modes:
  1. `--mode stage1-controlled-replay` (default, CPU-only, `CUDA_VISIBLE_DEVICES=""`):
     Deterministically replays all 24 held-out multi-field twin matter templates across
     4 arrival schedules, 4 pool sizes `(1, 2, 4, 8)`, the full 2x2 system factorial
     `{Conventional_Retriever, ZeroMu_Pinned_Bridge} x {SQLiteChallengerLedger, EK5_v3}`,
     all 5 event regimes (including `KEEP #2` `sham_subtask_reopen_interrupt`), and the
     `full_restart_rebuild` costed comparator.
  2. `--mode run-live-workers --vllm-url http://...`:
     Executes live concurrent LLM worker pools against an OpenAI-compatible vLLM endpoint
     (e.g. Spark #3) with the exact same commit-prefix verification, EK5 parity check,
     and timeout-in-denominator accounting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chelated_pooled_agent_runtime import (  # noqa: E402
    ARRIVAL_SCHEDULES,
    BUNDLE_ID,
    EVENT_REGIMES,
    POOL_SIZES,
    TWIN_BRANCHES,
    LiveVLLMWorkerBackend,
    RecordedReplayWorkerBackend,
    WorkerBackend,
    run_pooled_agent_experiment,
)
from chelated_pv7_spine.claim_registry import claim_registry_sha256  # noqa: E402


def _parse_int_csv(raw: str) -> Tuple[int, ...]:
    items = [int(tok.strip()) for tok in raw.split(",") if tok.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one integer value")
    return tuple(items)


def _parse_str_csv(raw: str, allowed: Sequence[str]) -> Tuple[str, ...]:
    items = [tok.strip() for tok in raw.split(",") if tok.strip()]
    if not items:
        raise argparse.ArgumentTypeError("Expected at least one value")
    bad = [item for item in items if item not in allowed]
    if bad:
        raise argparse.ArgumentTypeError(f"Unsupported values {bad!r}; allowed={list(allowed)}")
    return tuple(items)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_receipt_bound_bundle(
    output_dir: Path,
    manifest: Dict[str, Any],
) -> Tuple[Path, Path]:
    """Write `pv7_w20_d13_manifest.json` and SHA-256-bound `report_receipt.json`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_name = "pv7_w20_d13_manifest.json"
    manifest_path = output_dir / manifest_name
    manifest_bytes = (
        json.dumps(manifest, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    ).encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()

    receipt = {
        "schema_version": "pv7_report_receipt_v1",
        "bundle_id": BUNDLE_ID,
        "item": "4",
        "demo_id": "D13",
        "mode": manifest["mode"],
        "spine_digest": claim_registry_sha256(),
        "manifest_file": manifest_name,
        "manifest_sha256": manifest_sha,
        "deterministic_sha256": manifest["deterministic_sha256"],
        "overall_pass": bool(manifest["overall_pass"]),
        "gates": manifest["gates"],
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
        "source_file_sha256": {
            "chelated_pooled_agent_runtime.py": _sha256_file(REPO_ROOT / "chelated_pooled_agent_runtime.py"),
            "scripts/run_pv7_w20_d13_pooled_agent_loops.py": _sha256_file(
                REPO_ROOT / "scripts" / "run_pv7_w20_d13_pooled_agent_loops.py"
            ),
        },
    }
    receipt_path = output_dir / "report_receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    return manifest_path, receipt_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Wave 20 / Bundle #62 / Demo D13 Pooled-Agent Loops."
    )
    parser.add_argument(
        "--mode",
        choices=("stage1-controlled-replay", "run-live-workers"),
        default="stage1-controlled-replay",
        help="Execution mode: deterministic CPU Stage 1 replay or live Spark #3 vLLM workers.",
    )
    parser.add_argument(
        "--vllm-url",
        default="",
        help="OpenAI-compatible vLLM base URL (required for --mode run-live-workers).",
    )
    parser.add_argument(
        "--model",
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model identifier for live vLLM worker requests.",
    )
    parser.add_argument(
        "--timeout-s",
        type=float,
        default=15.0,
        help="Per-request HTTP timeout in seconds for live vLLM workers.",
    )
    parser.add_argument(
        "--pool-sizes",
        default=",".join(str(p) for p in POOL_SIZES),
        help="Comma-separated worker pool sizes (default: 1,2,4,8).",
    )
    parser.add_argument(
        "--schedules",
        default=",".join(ARRIVAL_SCHEDULES),
        help="Comma-separated deterministic arrival schedule IDs.",
    )
    parser.add_argument(
        "--branches",
        default=",".join(TWIN_BRANCHES),
        help="Comma-separated twin perturbation branches.",
    )
    parser.add_argument(
        "--event-regimes",
        default=",".join(EVENT_REGIMES),
        help="Comma-separated event regimes.",
    )
    parser.add_argument(
        "--include-jobs",
        action="store_true",
        help="Include per-job outcome rows inside each cell in the written manifest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory to write pv7_w20_d13_manifest.json and report_receipt.json.",
    )
    parser.add_argument(
        "--json-summary",
        action="store_true",
        help="Print compact JSON summary to stdout.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    pool_sizes = _parse_int_csv(args.pool_sizes)
    schedules = _parse_str_csv(args.schedules, ARRIVAL_SCHEDULES)
    branches = _parse_str_csv(args.branches, TWIN_BRANCHES)
    event_regimes = _parse_str_csv(args.event_regimes, EVENT_REGIMES)

    backend: WorkerBackend
    if args.mode == "stage1-controlled-replay":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        backend = RecordedReplayWorkerBackend()
    else:
        if not args.vllm_url:
            parser.error("--vllm-url is required when --mode run-live-workers is selected")
        backend = LiveVLLMWorkerBackend(
            vllm_url=args.vllm_url,
            model=args.model,
            timeout_s=args.timeout_s,
        )

    manifest = run_pooled_agent_experiment(
        mode=args.mode,
        pool_sizes=pool_sizes,
        schedules=schedules,
        branches=branches,
        event_regimes=event_regimes,
        worker_backend=backend,
        include_jobs=bool(args.include_jobs),
    )

    written: Optional[Dict[str, str]] = None
    if args.output_dir is not None:
        m_path, r_path = write_receipt_bound_bundle(args.output_dir, manifest)
        written = {"manifest": str(m_path), "receipt": str(r_path)}

    summary = {
        "bundle_id": manifest["bundle_id"],
        "mode": manifest["mode"],
        "overall_pass": manifest["overall_pass"],
        "total_cells_executed": manifest["total_cells_executed"],
        "total_eligible_jobs_executed": manifest["total_eligible_jobs_executed"],
        "zeromu_ek5_stale_commits_total": manifest["zeromu_ek5_stale_commits_total"],
        "ek5_parity_failures_total": manifest["ek5_parity_failures_total"],
        "primary_2x2_interaction": manifest["primary_2x2_interaction"]["jpm_interaction_per_job_bootstrap"],
        "sham_reopen_isolation": manifest["sham_reopen_isolation"],
        "full_restart_comparator": manifest["full_restart_comparator"],
        "gates": manifest["gates"],
        "deterministic_sha256": manifest["deterministic_sha256"],
        "written": written,
    }
    if args.json_summary or args.output_dir is None:
        print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if manifest["overall_pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
