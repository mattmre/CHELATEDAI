"""Run the PR-8 C3 knob sweep at calibrated rotation severity."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from analyze_drift_recovery import load_artifacts
from run_drift_recovery_experiment import DriftRecoveryConfig, _detect_device, run_experiment


BOUNDS = (0.01, 0.05, 0.10)
TRIGGERS = (0.05, 0.15)
PROFILES = {
    "default": {"max_temperature": 1.0, "epochs_scale": 1.0},
    "hotter": {"max_temperature": 0.01, "epochs_scale": 2.0},
}
SEEDS = (42, 1337, 7)
DEFAULT_OUTPUT_DIR = "experiment_runs/drift-recovery/knob-sweep"
DEFAULT_CALIBRATION_MANIFEST = "experiment_runs/drift-recovery/calibrated/calibration-manifest-2026-06.json"
DEFAULT_REPORT_MD = "docs/drift-recovery-knob-sweep-2026-06.md"


@dataclass(frozen=True)
class KnobSweepConfig:
    calibration_manifest: str = DEFAULT_CALIBRATION_MANIFEST
    output_dir: str = DEFAULT_OUTPUT_DIR
    report_md: str = DEFAULT_REPORT_MD
    task: str = "SciFact"
    max_queries: int = 100
    sample_docs: int = 1200
    cycles: int = 12
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: Optional[str] = None


def run_knob_sweep(
    config: KnobSweepConfig,
    runner: Callable[[DriftRecoveryConfig], Mapping[str, Any]] = run_experiment,
) -> dict:
    started = time.perf_counter()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rotation = _load_rotation_choice(config.calibration_manifest)

    grid_rows = []
    for bound in BOUNDS:
        for trigger in TRIGGERS:
            for profile_name, profile in PROFILES.items():
                run_config = _run_config(
                    config,
                    rotation,
                    output=output_dir / "grid" / _artifact_name(bound, trigger, profile_name, 42),
                    seed=42,
                    bound_epsilon=bound,
                    trigger_threshold=trigger,
                    max_temperature=profile["max_temperature"],
                    epochs_scale=profile["epochs_scale"],
                )
                result = runner(run_config)
                grid_rows.append(_sweep_row(result, phase="grid", profile=profile_name))

    top_cells = _select_top_cells(grid_rows)
    confirmation_rows = []
    for cell in top_cells:
        for seed in SEEDS:
            run_config = _run_config(
                config,
                rotation,
                output=output_dir / "confirmation" / _artifact_name(
                    cell["bound_epsilon"],
                    cell["trigger_threshold"],
                    cell["profile"],
                    seed,
                ),
                seed=seed,
                bound_epsilon=cell["bound_epsilon"],
                trigger_threshold=cell["trigger_threshold"],
                max_temperature=cell["max_temperature"],
                epochs_scale=cell["epochs_scale"],
            )
            result = runner(run_config)
            confirmation_rows.append(_sweep_row(result, phase="confirmation", profile=cell["profile"]))

    references = _load_calibrated_references(config.calibration_manifest)
    manifest = {
        "record_type": "drift_recovery_knob_sweep",
        "config": asdict(config),
        "calibrated_rotation_setting": rotation,
        "selection_policy": (
            "Top two cells by seed-42 final NDCG; ties are broken by lower trigger threshold, "
            "then profile name, then lower bound epsilon."
        ),
        "grid_definition": {
            "bounds": list(BOUNDS),
            "trigger_thresholds": list(TRIGGERS),
            "profiles": PROFILES,
            "seed": 42,
        },
        "grid_rows": grid_rows,
        "top_cells": top_cells,
        "confirmation_rows": confirmation_rows,
        "confirmation_summary": _confirmation_summary(confirmation_rows),
        "calibrated_references": references,
        "wall_clock_seconds": time.perf_counter() - started,
    }
    manifest_path = output_dir / "knob-sweep-manifest-2026-06.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    Path(config.report_md).parent.mkdir(parents=True, exist_ok=True)
    Path(config.report_md).write_text(render_knob_sweep_report(manifest, config.output_dir), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "report": config.report_md}, sort_keys=True))
    return manifest


def render_knob_sweep_report(manifest: Mapping[str, Any], output_dir: str) -> str:
    display_output_dir = output_dir.replace("\\", "/")
    lines = [
        "# Drift Recovery C3 Knob Sweep - June 2026",
        "",
        f"Manifest: `{display_output_dir}/knob-sweep-manifest-2026-06.json`.",
        "Scope: calibrated rotation setting only; all 12 seed-42 grid cells are reported.",
        f"Selection policy: {manifest['selection_policy']}",
        (
            "Profile note: `hotter` means the pre-registered aggressive schedule profile "
            "(lower max-temperature cap plus doubled epoch scale), not a higher temperature cap."
        ),
        "",
        "## Grid Cells",
        "",
        "| Bound epsilon | Trigger threshold | Profile | Max temperature | Epochs scale | Final NDCG | Recovery cycle | should_correct | attempted | applied |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in manifest["grid_rows"]:
        lines.append(
            f"| {row['bound_epsilon']:.2f} | {row['trigger_threshold']:.2f} | {row['profile']} | "
            f"{row['max_temperature']:.2f} | {row['epochs_scale']:.1f} | {row['final_ndcg']:.6f} | "
            f"{_nullable(row['recovery_cycle'])} | {row['should_correct_true']}/{row['cycle_count']} | "
            f"{row['sedimentation_attempted_true']}/{row['cycle_count']} | {row['correction_applied_true']}/{row['cycle_count']} |"
        )
    lines.extend(["", "## Three-Seed Confirmation", ""])
    lines.extend(
        [
            "| Bound epsilon | Trigger threshold | Profile | Mean final NDCG | Std | Recovery@12 |",
            "|---:|---:|---|---:|---:|---:|",
        ]
    )
    for row in manifest["confirmation_summary"]:
        lines.append(
            f"| {row['bound_epsilon']:.2f} | {row['trigger_threshold']:.2f} | {row['profile']} | "
            f"{row['final_mean']:.6f} | {row['final_std']:.6f} | {row['recovered_runs']}/{row['run_count']} |"
        )
    lines.extend(["", "## Calibrated References", ""])
    lines.extend(["| Condition | Final NDCG mean | Std | Recovery@12 |", "|---|---:|---:|---:|"])
    for condition in ("C0", "C2"):
        ref = manifest["calibrated_references"][condition]
        lines.append(
            f"| {condition} | {ref['final_mean']:.6f} | {ref['final_std']:.6f} | {ref['recovered_runs']}/{ref['run_count']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _run_config(
    config: KnobSweepConfig,
    rotation: Mapping[str, Any],
    output: Path,
    seed: int,
    bound_epsilon: float,
    trigger_threshold: float,
    max_temperature: float,
    epochs_scale: float,
) -> DriftRecoveryConfig:
    return DriftRecoveryConfig(
        task=config.task,
        condition="C3",
        drift="rotation",
        fraction=float(rotation["fraction"]),
        angle=float(rotation["angle"]),
        sigma=float(rotation["sigma"]),
        cycles=config.cycles,
        seed=seed,
        max_queries=config.max_queries,
        sample_docs=config.sample_docs,
        model=config.model,
        output=str(output),
        device=config.device or _detect_device(),
        bound_epsilon=bound_epsilon,
        trigger_threshold=trigger_threshold,
        max_temperature=max_temperature,
        epochs_scale=epochs_scale,
    )


def _sweep_row(result: Mapping[str, Any], phase: str, profile: str) -> dict:
    config = result["config"]
    trajectory = result["recovery"]["trajectory"]
    final = float(trajectory[-1]["ndcg"])
    cycle_rows = [point.get("metadata", {}) for point in trajectory]
    return {
        "phase": phase,
        "profile": profile,
        "path": str(config["output"]).replace("\\", "/"),
        "seed": int(config["seed"]),
        "bound_epsilon": float(config["bound_epsilon"]),
        "trigger_threshold": float(config["trigger_threshold"]),
        "max_temperature": float(config["max_temperature"]),
        "epochs_scale": float(config["epochs_scale"]),
        "baseline_ndcg": float(result["baseline"]["ndcg_at_10"]),
        "final_ndcg": final,
        "drop_pct": 100.0 * (float(result["baseline"]["ndcg_at_10"]) - final) / float(result["baseline"]["ndcg_at_10"]),
        "recovery_cycle": result["recovery"].get("recovery_cycle"),
        "cycle_count": len(trajectory),
        "should_correct_true": sum(1 for row in cycle_rows if row.get("should_correct")),
        "sedimentation_attempted_true": sum(1 for row in cycle_rows if row.get("sedimentation_attempted")),
        "correction_applied_true": sum(1 for row in cycle_rows if row.get("correction_applied")),
        "mean_correction_norm": result["correction_norm_stats"]["mean"],
    }


def _confirmation_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict]:
    groups: dict[tuple[float, float, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        groups.setdefault((row["bound_epsilon"], row["trigger_threshold"], row["profile"]), []).append(row)
    summary = []
    for (bound, trigger, profile), cells in sorted(groups.items()):
        finals = [cell["final_ndcg"] for cell in cells]
        summary.append(
            {
                "bound_epsilon": bound,
                "trigger_threshold": trigger,
                "profile": profile,
                "run_count": len(cells),
                "final_mean": float(statistics.fmean(finals)),
                "final_std": float(statistics.pstdev(finals)) if len(finals) > 1 else 0.0,
                "recovered_runs": sum(1 for cell in cells if cell["recovery_cycle"] is not None),
            }
        )
    return summary


def _select_top_cells(rows: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(
        rows,
        key=lambda row: (
            -float(row["final_ndcg"]),
            float(row["trigger_threshold"]),
            str(row["profile"]),
            float(row["bound_epsilon"]),
        ),
    )[:2]


def _load_rotation_choice(path: str) -> dict:
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    return dict(manifest["chosen_settings"]["rotation"])


def _load_calibrated_references(calibration_manifest: str) -> dict:
    calibration_dir = Path(calibration_manifest).parent
    artifacts = load_artifacts(str(calibration_dir / "scifact_*.json"))
    refs = {}
    for condition in ("C0", "C2"):
        rows = [
            artifact
            for artifact in artifacts
            if artifact["config"]["condition"] == condition and artifact["config"]["drift"] == "rotation"
        ]
        finals = [float(row["recovery"]["trajectory"][-1]["ndcg"]) for row in rows]
        refs[condition] = {
            "run_count": len(rows),
            "final_mean": float(statistics.fmean(finals)),
            "final_std": float(statistics.pstdev(finals)) if len(finals) > 1 else 0.0,
            "recovered_runs": sum(1 for row in rows if row["recovery"].get("recovery_cycle") is not None),
        }
    return refs


def _artifact_name(bound: float, trigger: float, profile: str, seed: int) -> str:
    return f"c3_bound{_fmt(bound)}_trigger{_fmt(trigger)}_{profile}_seed{seed}.json"


def _fmt(value: float) -> str:
    return ("%g" % value).replace(".", "p")


def _nullable(value: Any) -> str:
    return "" if value is None else str(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run calibrated C3 knob sweep")
    parser.add_argument("--calibration-manifest", default=DEFAULT_CALIBRATION_MANIFEST)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-md", default=DEFAULT_REPORT_MD)
    parser.add_argument("--task", default="SciFact")
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--sample-docs", type=int, default=1200)
    parser.add_argument("--cycles", type=int, default=12)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = KnobSweepConfig(
        calibration_manifest=args.calibration_manifest,
        output_dir=args.output_dir,
        report_md=args.report_md,
        task=args.task,
        max_queries=args.max_queries,
        sample_docs=args.sample_docs,
        cycles=args.cycles,
        model=args.model,
        device=_detect_device(),
    )
    run_knob_sweep(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
