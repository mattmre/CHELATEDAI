"""Run calibrated severity selection for drift-recovery experiments."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from analyze_drift_recovery import (
    CONDITIONS,
    DRIFT_MODES,
    analyze_artifacts,
    load_artifacts,
    write_diagnostics,
)
from run_drift_recovery_experiment import DriftRecoveryConfig, _detect_device, run_experiment


SEEDS = (42, 1337, 7)
SCOUT_ROTATION_ANGLES = (35.0, 50.0, 65.0)
SCOUT_ROTATION_FRACTIONS = (0.5, 0.75)
SCOUT_NOISE_SIGMAS = (0.01, 0.02, 0.035)
SCOUT_NOISE_FRACTIONS = (0.5,)
TARGET_DROP_PCT = 12.0
DROP_ZONE_MIN_PCT = 8.0
DROP_ZONE_MAX_PCT = 20.0
DEFAULT_OUTPUT_DIR = "experiment_runs/drift-recovery/calibrated"
DEFAULT_REPORT_MD = "docs/drift-recovery-calibrated-results-2026-06.md"


@dataclass(frozen=True)
class CalibrationConfig:
    task: str = "SciFact"
    max_queries: int = 100
    sample_docs: int = 1200
    cycles: int = 12
    output_dir: str = DEFAULT_OUTPUT_DIR
    report_md: str = DEFAULT_REPORT_MD
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: Optional[str] = None


def run_calibration_campaign(
    config: CalibrationConfig,
    runner: Callable[[DriftRecoveryConfig], Mapping[str, Any]] = run_experiment,
) -> dict:
    """Run C0 scout cells, choose calibrated settings, then run the full matrix."""

    started = time.perf_counter()
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scout_rows = []
    for scout in _scout_configs(config):
        result = runner(scout)
        row = _scout_row(result)
        scout_rows.append(row)

    choices = {
        "rotation": _choose_setting([row for row in scout_rows if row["drift"] == "rotation"]),
        "noise": _choose_setting([row for row in scout_rows if row["drift"] == "noise"]),
    }

    matrix_outputs = []
    for drift in DRIFT_MODES:
        choice = choices[drift]
        for condition in CONDITIONS:
            for seed in SEEDS:
                run_config = _matrix_config(config, drift=drift, condition=condition, seed=seed, choice=choice)
                result = runner(run_config)
                matrix_outputs.append(str(Path(run_config.output)).replace("\\", "/"))

    diagnostics_input = str(output_dir / "scifact_C*.json")
    diagnostics_json = output_dir / "diagnostics-2026-06.json"
    diagnostics_md = output_dir / "diagnostics-2026-06.md"
    artifacts = load_artifacts(diagnostics_input)
    diagnostics = analyze_artifacts(artifacts)
    diagnostic_outputs = write_diagnostics(
        diagnostics,
        output_json=str(diagnostics_json),
        output_md=str(diagnostics_md),
        plot_dir=str(output_dir),
    )

    manifest = {
        "record_type": "drift_recovery_calibration_campaign",
        "config": asdict(config),
        "choice_rule": {
            "pre_registered": "Among scout cells, choose a setting inside the 8-20% baseline-drop zone closest to 12%; if none land in-zone, choose the closest overall and disclose that.",
            "target_drop_pct": TARGET_DROP_PCT,
            "zone_min_pct": DROP_ZONE_MIN_PCT,
            "zone_max_pct": DROP_ZONE_MAX_PCT,
        },
        "scout_rows": scout_rows,
        "chosen_settings": choices,
        "matrix_outputs": sorted(matrix_outputs),
        "diagnostics": diagnostic_outputs,
        "wall_clock_seconds": time.perf_counter() - started,
    }
    manifest_path = output_dir / "calibration-manifest-2026-06.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    Path(config.report_md).parent.mkdir(parents=True, exist_ok=True)
    Path(config.report_md).write_text(render_calibration_report(manifest, diagnostics, config.output_dir), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "report": config.report_md}, sort_keys=True))
    return manifest


def render_calibration_report(manifest: Mapping[str, Any], diagnostics: Mapping[str, Any], output_dir: str) -> str:
    display_output_dir = output_dir.replace("\\", "/")
    lines = [
        "# Drift Recovery Calibrated Severity Results - June 2026",
        "",
        f"Manifest: `{display_output_dir}/calibration-manifest-2026-06.json`.",
        f"Choice rule: {manifest['choice_rule']['pre_registered']}",
        "",
        "## Scout Cells",
        "",
        "| Drift | Fraction | Angle | Sigma | Baseline | Final C0 | Drop % | In 8-20% zone |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in sorted(manifest["scout_rows"], key=lambda item: (item["drift"], item["fraction"], item["angle"], item["sigma"])):
        lines.append(
            f"| {row['drift']} | {row['fraction']:.2f} | {row['angle']:.1f} | {row['sigma']:.3f} | "
            f"{row['baseline_ndcg']:.6f} | {row['final_ndcg']:.6f} | {row['drop_pct']:.3f}% | {row['in_zone']} |"
        )
    lines.extend(["", "## Chosen Settings", ""])
    for drift in DRIFT_MODES:
        choice = manifest["chosen_settings"][drift]
        lines.append(
            f"- {drift}: fraction {choice['fraction']:.2f}, angle {choice['angle']:.1f}, "
            f"sigma {choice['sigma']:.3f}; scout drop {choice['drop_pct']:.3f}% "
            f"({'in zone' if choice['in_zone'] else 'closest outside zone'})."
        )
    lines.extend(
        [
            "",
            "## Full Matrix",
            "",
            "| Drift | Condition | Baseline NDCG | Final NDCG | Drop % | Recovery@12 |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in diagnostics["summary_rows"]:
        lines.append(
            "| {drift} | {condition} | {baseline_mean:.6f} +/- {baseline_std:.6f} | "
            "{final_mean:.6f} +/- {final_std:.6f} | {drop_pct_mean:.3f}% | {recovered_runs}/{run_count} |".format(
                **row
            )
        )
    lines.extend(["", "## Plots", ""])
    for drift in DRIFT_MODES:
        lines.append(f"- `{display_output_dir}/scifact_{drift}_ndcg_diagnostics_mean.png`")
    lines.append("")
    return "\n".join(lines)


def _scout_configs(config: CalibrationConfig) -> list[DriftRecoveryConfig]:
    output_dir = Path(config.output_dir)
    rows = []
    for angle in SCOUT_ROTATION_ANGLES:
        for fraction in SCOUT_ROTATION_FRACTIONS:
            rows.append(
                _base_config(
                    config,
                    condition="C0",
                    drift="rotation",
                    seed=42,
                    fraction=fraction,
                    angle=angle,
                    sigma=0.05,
                    output=output_dir / f"scout_C0_rotation_fraction{_fmt(fraction)}_angle{_fmt(angle)}_seed42.json",
                )
            )
    for sigma in SCOUT_NOISE_SIGMAS:
        for fraction in SCOUT_NOISE_FRACTIONS:
            rows.append(
                _base_config(
                    config,
                    condition="C0",
                    drift="noise",
                    seed=42,
                    fraction=fraction,
                    angle=25.0,
                    sigma=sigma,
                    output=output_dir / f"scout_C0_noise_fraction{_fmt(fraction)}_sigma{_fmt(sigma)}_seed42.json",
                )
            )
    return rows


def _matrix_config(
    config: CalibrationConfig,
    drift: str,
    condition: str,
    seed: int,
    choice: Mapping[str, Any],
) -> DriftRecoveryConfig:
    output_dir = Path(config.output_dir)
    return _base_config(
        config,
        condition=condition,
        drift=drift,
        seed=seed,
        fraction=float(choice["fraction"]),
        angle=float(choice["angle"]),
        sigma=float(choice["sigma"]),
        output=output_dir / f"scifact_{condition}_{drift}_seed{seed}.json",
    )


def _base_config(
    config: CalibrationConfig,
    condition: str,
    drift: str,
    seed: int,
    fraction: float,
    angle: float,
    sigma: float,
    output: Path,
) -> DriftRecoveryConfig:
    return DriftRecoveryConfig(
        task=config.task,
        condition=condition,
        drift=drift,
        fraction=fraction,
        angle=angle,
        sigma=sigma,
        cycles=config.cycles,
        seed=seed,
        max_queries=config.max_queries,
        sample_docs=config.sample_docs,
        model=config.model,
        output=str(output),
        device=config.device or _detect_device(),
    )


def _scout_row(result: Mapping[str, Any]) -> dict:
    config = result["config"]
    baseline = float(result["baseline"]["ndcg_at_10"])
    final = float(result["recovery"]["trajectory"][-1]["ndcg"])
    drop_pct = 100.0 * (baseline - final) / baseline if baseline else 0.0
    return {
        "path": str(config["output"]).replace("\\", "/"),
        "drift": config["drift"],
        "fraction": float(config["fraction"]),
        "angle": float(config["angle"]),
        "sigma": float(config["sigma"]),
        "seed": int(config["seed"]),
        "baseline_ndcg": baseline,
        "final_ndcg": final,
        "drop_pct": drop_pct,
        "distance_to_target": abs(drop_pct - TARGET_DROP_PCT),
        "in_zone": DROP_ZONE_MIN_PCT <= drop_pct <= DROP_ZONE_MAX_PCT,
    }


def _choose_setting(rows: Sequence[Mapping[str, Any]]) -> dict:
    if not rows:
        raise ValueError("Cannot choose calibrated setting from empty scout rows")
    candidates = [row for row in rows if row["in_zone"]] or list(rows)
    chosen = min(candidates, key=lambda row: (abs(row["drop_pct"] - TARGET_DROP_PCT), row["drop_pct"]))
    return dict(chosen)


def _fmt(value: float) -> str:
    return ("%g" % value).replace(".", "p")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run calibrated drift-recovery severity campaign")
    parser.add_argument("--task", default="SciFact")
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--sample-docs", type=int, default=1200)
    parser.add_argument("--cycles", type=int, default=12)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-md", default=DEFAULT_REPORT_MD)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = CalibrationConfig(
        task=args.task,
        max_queries=args.max_queries,
        sample_docs=args.sample_docs,
        cycles=args.cycles,
        output_dir=args.output_dir,
        report_md=args.report_md,
        model=args.model,
        device=_detect_device(),
    )
    run_calibration_campaign(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
