"""
Orchestrate the post-development weight-refinement campaign.

This runner executes the bounded phases synchronously and can optionally
launch the large sweep in the background. It isolates outputs per run and
restores adapter_weights.pt between phases to avoid cross-phase contamination.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from benchmark_comparative import (
    BenchmarkConfiguration,
    ComparativeTestbed,
    build_real_engine_factory,
)
from benchmark_utils import load_mteb_data
from config import ChelationConfig
from online_updater import OnlineUpdater


PROJECT_ROOT = Path(__file__).resolve().parent


def _timestamp_slug() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S")


def _write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, default=str)


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def snapshot_adapter(run_dir: Path) -> Path | None:
    adapter_path = PROJECT_ROOT / "adapter_weights.pt"
    if not adapter_path.exists():
        return None
    snapshot_path = run_dir / "baseline_adapter_weights.pt"
    shutil.copy2(adapter_path, snapshot_path)
    return snapshot_path


def restore_adapter(snapshot_path: Path | None) -> None:
    adapter_path = PROJECT_ROOT / "adapter_weights.pt"
    if snapshot_path is None:
        if adapter_path.exists():
            adapter_path.unlink()
        return

    shutil.copy2(snapshot_path, adapter_path)


def update_manifest(run_dir: Path, manifest: dict[str, Any]) -> None:
    _write_json(run_dir / "manifest.json", manifest)


def load_manifest(run_dir: Path) -> dict[str, Any]:
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        return {}
    with open(manifest_path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def phase_is_complete(phase_info: Any) -> bool:
    if not isinstance(phase_info, dict):
        return False
    if phase_info.get("returncode") == 0:
        return True
    return phase_info.get("status") in {
        "completed",
        "launched",
        "recovered_from_output",
    }


def recover_completed_phase(
    manifest: dict[str, Any],
    label: str,
    output_path: Path,
    extra_fields: dict[str, Any] | None = None,
) -> None:
    phase_info = {
        "label": label,
        "status": "recovered_from_output",
        "output_path": str(output_path),
    }
    if extra_fields:
        phase_info.update(extra_fields)
    manifest.setdefault("phases", {})[label] = phase_info


def run_command(label: str, command: list[str], run_dir: Path) -> dict[str, Any]:
    log_path = run_dir / "logs" / f"{label}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    started_at = datetime.now().isoformat()
    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write(f"$ {' '.join(command)}\n\n")
        process = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            stdout=log_file,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            check=False,
        )

    return {
        "label": label,
        "command": command,
        "log_path": str(log_path),
        "returncode": process.returncode,
        "started_at": started_at,
        "finished_at": datetime.now().isoformat(),
    }


def launch_background_command(label: str, command: list[str], run_dir: Path) -> dict[str, Any]:
    log_path = run_dir / "logs" / f"{label}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"

    log_handle = open(log_path, "w", encoding="utf-8")
    log_handle.write(f"$ {' '.join(command)}\n\n")
    log_handle.flush()

    process = subprocess.Popen(
        command,
        cwd=PROJECT_ROOT,
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        creationflags=getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
    )
    log_handle.close()

    return {
        "label": label,
        "command": command,
        "log_path": str(log_path),
        "pid": process.pid,
        "started_at": datetime.now().isoformat(),
        "status": "launched",
    }


def run_online_ablation(
    model_name: str,
    task_name: str,
    max_queries: int,
    run_dir: Path,
) -> dict[str, Any]:
    corpus, queries, qrels = load_mteb_data(task_name)
    if not corpus or not queries or not qrels:
        raise RuntimeError(f"Failed to load MTEB task '{task_name}' for online ablation")

    def make_online_setup(loss_type: str):
        def _setup(engine):
            engine.enable_stability_tracking()
            engine._online_updater = OnlineUpdater(
                adapter=engine.adapter,
                learning_rate=ChelationConfig.ONLINE_LEARNING_RATE,
                micro_steps=ChelationConfig.ONLINE_MICRO_STEPS,
                momentum=ChelationConfig.ONLINE_MOMENTUM,
                max_grad_norm=ChelationConfig.ONLINE_MAX_GRAD_NORM,
                update_interval=ChelationConfig.ONLINE_UPDATE_INTERVAL,
                loss_type=loss_type,
            )
        return _setup

    configs = [
        BenchmarkConfiguration(name="chelation_baseline", use_centering=True, use_quantization=True),
        BenchmarkConfiguration(
            name="online_triplet_margin",
            use_centering=True,
            use_quantization=True,
            extra_setup=make_online_setup("triplet_margin"),
        ),
        BenchmarkConfiguration(
            name="online_infonce",
            use_centering=True,
            use_quantization=True,
            extra_setup=make_online_setup("infonce"),
        ),
        BenchmarkConfiguration(
            name="online_cosine_similarity",
            use_centering=True,
            use_quantization=True,
            extra_setup=make_online_setup("cosine_similarity"),
        ),
    ]

    engine_factory = build_real_engine_factory(corpus, model_name=model_name)
    testbed = ComparativeTestbed(configurations=configs)
    results = testbed.run_all(
        corpus,
        queries,
        qrels,
        engine_factory=engine_factory,
        max_queries=max_queries,
    )

    output_path = run_dir / "phase5_online_ablation.json"
    _write_json(
        output_path,
        {
            "task": task_name,
            "model": model_name,
            "max_queries": max_queries,
            "results": [asdict(result) for result in results],
            "ascii_table": testbed.format_ascii_table(results),
        },
    )

    best_result = max(results, key=lambda result: result.ndcg_at_10)
    return {
        "output_path": str(output_path),
        "best_config": best_result.config_name,
        "best_ndcg_at_10": best_result.ndcg_at_10,
    }


def summarize_run(manifest: dict[str, Any], run_dir: Path) -> None:
    lines = [
        "# Weight Refinement Campaign Summary",
        "",
        f"- Run directory: `{run_dir}`",
        f"- Started: `{manifest['started_at']}`",
        f"- Model: `{manifest['config']['model']}`",
        f"- Query budget: `{manifest['config']['max_queries']}`",
        "",
        "## Completed Phases",
    ]

    for phase_name, phase_info in manifest["phases"].items():
        if isinstance(phase_info, dict):
            status = phase_info.get("status", "recorded")
            lines.append(f"- `{phase_name}`: {status}")

    if "phase5_online_ablation" in manifest["phases"]:
        phase5 = manifest["phases"]["phase5_online_ablation"]
        lines.extend(
            [
                "",
                "## Online Ablation Best Result",
                f"- Best config: `{phase5['best_config']}`",
                f"- NDCG@10: `{phase5['best_ndcg_at_10']:.4f}`",
            ]
        )

    if "phase6_large_sweep" in manifest["phases"]:
        phase6 = manifest["phases"]["phase6_large_sweep"]
        lines.extend(
            [
                "",
                "## Large Sweep",
                f"- Status: `{phase6['status']}`",
                f"- PID: `{phase6['pid']}`",
                f"- Log: `{phase6['log_path']}`",
            ]
        )

    _write_text(run_dir / "SUMMARY.md", "\n".join(lines) + "\n")


def recover_phase5_result(output_path: Path) -> dict[str, Any]:
    with open(output_path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    results = data.get("results", [])
    if not results:
        return {
            "output_path": str(output_path),
            "best_config": None,
            "best_ndcg_at_10": 0.0,
        }

    best_result = max(results, key=lambda result: result["ndcg_at_10"])
    return {
        "output_path": str(output_path),
        "best_config": best_result["config_name"],
        "best_ndcg_at_10": best_result["ndcg_at_10"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the weight-refinement campaign")
    parser.add_argument(
        "--resume-run-dir",
        default=None,
        help="Resume an existing run directory and only execute missing phases",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Explicit output directory for this campaign run",
    )
    parser.add_argument(
        "--model",
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Embedding model to use for real-evaluation phases",
    )
    parser.add_argument(
        "--teacher",
        default=None,
        help="Teacher model for distillation phases (default: same as --model)",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=50,
        help="Maximum query budget for comparative BEIR/multitask phases",
    )
    parser.add_argument(
        "--distill-queries-per-cycle",
        type=int,
        default=30,
        help="Queries per cycle for distillation runs",
    )
    parser.add_argument(
        "--distill-cycles",
        type=int,
        default=3,
        help="Number of distillation cycles",
    )
    parser.add_argument(
        "--distill-epochs",
        type=int,
        default=3,
        help="Epochs per distillation cycle",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.01,
        help="Shared learning rate for bounded campaign phases (Phase 1 sweet spot)",
    )
    parser.add_argument(
        "--adapter-types",
        default="mlp",
        help="Comma-separated adapter types to test in Phase 2 distillation (default: mlp)",
    )
    parser.add_argument(
        "--launch-large-sweep",
        action="store_true",
        help="Launch run_large_sweep.py in the background after bounded phases",
    )
    args = parser.parse_args()

    if args.teacher is None:
        args.teacher = args.model

    if args.resume_run_dir:
        run_dir = Path(args.resume_run_dir)
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir
        if not run_dir.exists():
            raise FileNotFoundError(f"Resume run directory does not exist: {run_dir}")
    elif args.run_dir:
        run_dir = Path(args.run_dir)
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir
    else:
        run_dir = PROJECT_ROOT / "experiment_runs" / f"weight-refinement-{_timestamp_slug()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, Any]
    if args.resume_run_dir:
        manifest = load_manifest(run_dir)
        manifest.setdefault("started_at", datetime.now().isoformat())
        manifest["run_dir"] = str(run_dir)
        # Preserve original config; only override fields explicitly set on CLI.
        existing_config = manifest.get("config", {})
        defaults = vars(parser.parse_args([]))
        cli_args = vars(args)
        for key, value in cli_args.items():
            if value != defaults.get(key):
                existing_config[key] = value
        manifest["config"] = existing_config
        manifest.setdefault("phases", {})
        manifest["resumed_at"] = datetime.now().isoformat()
    else:
        manifest = {
            "started_at": datetime.now().isoformat(),
            "run_dir": str(run_dir),
            "config": vars(args),
            "phases": {},
        }

    snapshot_ref = manifest.get("baseline_adapter_snapshot")
    snapshot_path = Path(snapshot_ref) if snapshot_ref else None
    if snapshot_path is not None and not snapshot_path.exists():
        snapshot_path = None
    if snapshot_ref is None:
        snapshot_path = snapshot_adapter(run_dir)
        manifest["baseline_adapter_snapshot"] = str(snapshot_path) if snapshot_path else None
    update_manifest(run_dir, manifest)

    try:
        phase1_output = run_dir / "phase1_sweep_results.json"
        if phase_is_complete(manifest["phases"].get("phase1_standard_sweep")) and phase1_output.exists():
            pass
        elif phase1_output.exists():
            recover_completed_phase(manifest, "phase1_standard_sweep", phase1_output)
            update_manifest(run_dir, manifest)
        else:
            restore_adapter(snapshot_path)
            manifest["phases"]["phase1_standard_sweep"] = run_command(
                "phase1_standard_sweep",
                [
                    sys.executable,
                    "-u",
                    "run_sweep.py",
                    "--task",
                    "SciFact",
                    "--model",
                    args.model,
                    "--out",
                    str(phase1_output),
                    "--max-queries",
                    str(args.max_queries),
                    "--db-path",
                    str(run_dir / "qdrant" / "phase1_scifact_db"),
                ],
                run_dir,
            )
            update_manifest(run_dir, manifest)

        adapter_types = [t.strip() for t in args.adapter_types.split(",")]
        for adapter_type in adapter_types:
            for teacher_weight in ("0.3", "0.5", "0.7"):
                label = f"phase2_distillation_{adapter_type}_tw_{teacher_weight.replace('.', '')}"
                output_path = run_dir / f"{label}.json"
                if phase_is_complete(manifest["phases"].get(label)) and output_path.exists():
                    continue
                if output_path.exists():
                    recover_completed_phase(manifest, label, output_path)
                    update_manifest(run_dir, manifest)
                    continue

                restore_adapter(snapshot_path)
                manifest["phases"][label] = run_command(
                    label,
                    [
                        sys.executable,
                        "-u",
                        "benchmark_distillation.py",
                        "--task",
                        "SciFact",
                        "--model",
                        args.model,
                        "--teacher",
                        args.teacher,
                        "--cycles",
                        str(args.distill_cycles),
                        "--queries-per-cycle",
                        str(args.distill_queries_per_cycle),
                        "--epochs",
                        str(args.distill_epochs),
                        "--lr",
                        str(args.learning_rate),
                        "--max-eval-queries",
                        str(args.max_queries),
                        "--teacher-weight",
                        teacher_weight,
                        "--threshold",
                        "1",
                        "--adapter-type",
                        adapter_type,
                        "--output",
                        str(output_path),
                    ],
                    run_dir,
                )
                update_manifest(run_dir, manifest)

        for suite in ("small", "medium"):
            label = f"phase3_multitask_{suite}"
            output_path = run_dir / f"{label}.json"
            if phase_is_complete(manifest["phases"].get(label)) and output_path.exists():
                continue
            if output_path.exists():
                recover_completed_phase(manifest, label, output_path)
                update_manifest(run_dir, manifest)
                continue

            restore_adapter(snapshot_path)
            manifest["phases"][label] = run_command(
                label,
                [
                    sys.executable,
                    "-u",
                    "benchmark_multitask.py",
                    "--tasks",
                    suite,
                    "--model",
                    args.model,
                    "--max-queries",
                    str(args.max_queries),
                    "--num-queries-train",
                    str(min(args.max_queries, 50)),
                    "--epochs",
                    str(args.distill_epochs),
                    "--lr",
                    str(args.learning_rate),
                    "--threshold",
                    "1",
                    "--output",
                    str(output_path),
                ],
                run_dir,
            )
            update_manifest(run_dir, manifest)

        for tier in ("small", "medium"):
            label = f"phase4_beir_{tier}"
            output_path = run_dir / f"{label}.json"
            if phase_is_complete(manifest["phases"].get(label)) and output_path.exists():
                continue
            if output_path.exists():
                recover_completed_phase(manifest, label, output_path)
                update_manifest(run_dir, manifest)
                continue

            restore_adapter(snapshot_path)
            manifest["phases"][label] = run_command(
                label,
                [
                    sys.executable,
                    "-u",
                    "benchmark_beir.py",
                    "--tier",
                    tier,
                    "--model",
                    args.model,
                    "--max-queries",
                    str(args.max_queries),
                    "--output",
                    str(output_path),
                ],
                run_dir,
            )
            update_manifest(run_dir, manifest)

        phase5_output = run_dir / "phase5_online_ablation.json"
        phase5_info = manifest["phases"].get("phase5_online_ablation")
        if phase_is_complete(phase5_info) and phase5_output.exists():
            pass
        elif phase5_output.exists():
            manifest["phases"]["phase5_online_ablation"] = {
                "status": "completed",
                **recover_phase5_result(phase5_output),
            }
            update_manifest(run_dir, manifest)
        else:
            restore_adapter(snapshot_path)
            manifest["phases"]["phase5_online_ablation"] = {
                "status": "completed",
                **run_online_ablation(
                    model_name=args.model,
                    task_name="SciFact",
                    max_queries=args.max_queries,
                    run_dir=run_dir,
                ),
            }
            update_manifest(run_dir, manifest)

        phase6_command = [
            sys.executable,
            "-u",
            "run_large_sweep.py",
            "--task",
            "SciFact",
            "--model",
            args.model,
            "--out",
            str(run_dir / "phase6_large_sweep"),
            "--max-queries",
            str(args.max_queries),
            "--db-path",
            str(run_dir / "qdrant" / "phase6_scifact_db"),
        ]
        phase6_info = manifest["phases"].get("phase6_large_sweep")
        if phase_is_complete(phase6_info):
            pass
        elif args.launch_large_sweep:
            restore_adapter(snapshot_path)
            manifest["phases"]["phase6_large_sweep"] = launch_background_command(
                "phase6_large_sweep",
                phase6_command,
                run_dir,
            )
            update_manifest(run_dir, manifest)
        else:
            manifest["phases"]["phase6_large_sweep"] = {
                "status": "staged_not_launched",
                "command": phase6_command,
            }
            update_manifest(run_dir, manifest)

        manifest["finished_at"] = datetime.now().isoformat()
        update_manifest(run_dir, manifest)
        summarize_run(manifest, run_dir)
    finally:
        restore_adapter(snapshot_path)

    print(str(run_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
