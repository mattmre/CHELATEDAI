"""Analyze June 2026 drift-recovery experiment artifacts."""

from __future__ import annotations

import argparse
import glob
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional, Sequence


CONDITIONS = ("C0", "C1", "C2", "C3", "C4")
DRIFT_MODES = ("rotation", "noise")
DEFAULT_INPUT_GLOB = "experiment_runs/drift-recovery/scifact_*.json"
DEFAULT_OUTPUT_JSON = "experiment_runs/drift-recovery/diagnostics-2026-06.json"
DEFAULT_OUTPUT_MD = "docs/drift-recovery-diagnostics-2026-06.md"
DEFAULT_PLOT_DIR = "experiment_runs/drift-recovery"
RECOVERY_FRACTION = 0.95
BASELINE_EQUAL_TOLERANCE = 1e-12


def load_artifacts(input_glob: str = DEFAULT_INPUT_GLOB) -> list[dict]:
    paths = [Path(path) for path in sorted(glob.glob(input_glob))]
    if not paths:
        raise FileNotFoundError(f"No artifacts matched {input_glob!r}")
    artifacts = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("record_type") != "drift_recovery_experiment":
            continue
        config = payload.get("config", {})
        if config.get("condition") not in CONDITIONS or config.get("drift") not in DRIFT_MODES:
            continue
        payload["_source_path"] = str(path).replace("\\", "/")
        artifacts.append(payload)
    if not artifacts:
        raise ValueError(f"No drift_recovery_experiment artifacts found in {input_glob!r}")
    return artifacts


def analyze_artifacts(artifacts: Sequence[Mapping[str, Any]]) -> dict:
    rows = [_run_row(artifact) for artifact in artifacts]
    _validate_matrix(rows)

    grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    by_mode: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["condition"], row["drift"])].append(row)
        by_mode[row["drift"]].append(row)

    summary_rows = []
    for condition in CONDITIONS:
        for drift in DRIFT_MODES:
            cells = sorted(grouped[(condition, drift)], key=lambda item: item["seed"])
            if not cells:
                continue
            summary_rows.append(_summary_row(condition, drift, cells))

    c3_c4_traces = _condition_traces(artifacts)
    severity = _severity_analysis(grouped)
    h3 = _h3_verdict(grouped)
    h2 = _h2_verdict(c3_c4_traces)
    h1 = _h1_verdict(severity)
    trajectories = _trajectory_means(artifacts)

    return {
        "record_type": "drift_recovery_diagnostics",
        "source_artifacts": sorted(row["source_path"] for row in rows),
        "artifact_count": len(rows),
        "summary_rows": summary_rows,
        "run_rows": sorted(rows, key=lambda item: (item["drift"], item["condition"], item["seed"])),
        "severity": severity,
        "c3_c4_traces": c3_c4_traces,
        "hypotheses": {
            "H1_rotation_too_weak": h1,
            "H2_correction_barely_moves": h2,
            "H3_c2_oracle_reembed": h3,
        },
        "trajectory_means": trajectories,
    }


def write_diagnostics(diagnostics: Mapping[str, Any], output_json: str, output_md: str, plot_dir: str) -> dict:
    json_path = Path(output_json)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(diagnostics, indent=2, sort_keys=True), encoding="utf-8")

    md_path = Path(output_md)
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text(render_markdown(diagnostics, output_json=output_json, plot_dir=plot_dir), encoding="utf-8")

    plot_paths = write_trajectory_plots(diagnostics, plot_dir)
    return {"json": str(json_path), "markdown": str(md_path), "plots": plot_paths}


def render_markdown(diagnostics: Mapping[str, Any], output_json: str, plot_dir: str) -> str:
    lines = [
        "# Drift Recovery Diagnostics - June 2026",
        "",
        f"Diagnostics JSON: `{output_json}`.",
        f"Source artifact count: {diagnostics['artifact_count']}.",
        "",
        "## Hypothesis Verdicts",
        "",
    ]
    for key, verdict in diagnostics["hypotheses"].items():
        lines.append(f"- {key}: **{verdict['verdict']}** - {verdict['evidence']}")
    lines.extend(
        [
            "",
            "## Baseline vs Final",
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

    lines.extend(
        [
            "",
            "## C3/C4 Correction Trace",
            "",
            "| Drift | Condition | should_correct | attempted | applied | Mean drift signal | Mean correction norm | Max correction norm |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in diagnostics["c3_c4_traces"]["summary"]:
        lines.append(
            "| {drift} | {condition} | {should_correct_true}/{cycle_count} | "
            "{sedimentation_attempted_true}/{cycle_count} | {correction_applied_true}/{cycle_count} | "
            "{mean_drift_signal:.9f} | {mean_correction_norm:.9f} | {max_correction_norm:.9f} |".format(**row)
        )

    lines.extend(
        [
            "",
            "## Severity",
            "",
            "| Drift | C0 mean baseline | C0 mean final | Mean drop % | All C0 runs above 95% threshold |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for mode in DRIFT_MODES:
        row = diagnostics["severity"][mode]
        lines.append(
            f"| {mode} | {row['c0_baseline_mean']:.6f} | {row['c0_final_mean']:.6f} | "
            f"{row['c0_drop_pct_mean']:.3f}% | {row['all_c0_above_recovery_threshold']} |"
        )

    lines.extend(
        [
            "",
            "## Plots",
            "",
        ]
    )
    for mode in DRIFT_MODES:
        lines.append(f"- `{plot_dir}/scifact_{mode}_ndcg_diagnostics_mean.png`")
    lines.append("")
    return "\n".join(lines)


def write_trajectory_plots(diagnostics: Mapping[str, Any], plot_dir: str) -> list[str]:
    import matplotlib.pyplot as plt

    output_dir = Path(plot_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_paths = []
    for mode in DRIFT_MODES:
        fig, ax = plt.subplots(figsize=(9, 5.5))
        for condition in CONDITIONS:
            series = diagnostics["trajectory_means"].get(mode, {}).get(condition)
            if not series:
                continue
            cycles = [point["cycle_index"] for point in series]
            values = [point["mean_ndcg"] for point in series]
            ax.plot(cycles, values, marker="o", linewidth=1.8, label=condition)
        ax.set_title(f"SciFact {mode} drift: mean NDCG trajectory")
        ax.set_xlabel("Cycle")
        ax.set_ylabel("NDCG@10")
        ax.grid(True, alpha=0.25)
        ax.legend()
        fig.tight_layout()
        path = output_dir / f"scifact_{mode}_ndcg_diagnostics_mean.png"
        fig.savefig(path, dpi=160)
        plt.close(fig)
        plot_paths.append(str(path).replace("\\", "/"))
    return plot_paths


def _run_row(artifact: Mapping[str, Any]) -> dict:
    config = artifact["config"]
    recovery = artifact["recovery"]
    trajectory = recovery["trajectory"]
    if not trajectory:
        raise ValueError(f"Artifact {artifact.get('_source_path')} has an empty trajectory")
    baseline = float(artifact["baseline"]["ndcg_at_10"])
    final = float(trajectory[-1]["ndcg"])
    drop = baseline - final
    drop_pct = 100.0 * drop / baseline if baseline else 0.0
    threshold = baseline * RECOVERY_FRACTION
    recovery_cycle = recovery.get("recovery_cycle")
    return {
        "source_path": artifact.get("_source_path", ""),
        "condition": config["condition"],
        "drift": config["drift"],
        "seed": int(config["seed"]),
        "baseline_ndcg": baseline,
        "final_ndcg": final,
        "drop_abs": drop,
        "drop_pct": drop_pct,
        "recovery_threshold_ndcg": threshold,
        "final_above_recovery_threshold": final >= threshold,
        "recovery_cycle": recovery_cycle,
        "cycle_count": len(trajectory),
    }


def _summary_row(condition: str, drift: str, rows: Sequence[Mapping[str, Any]]) -> dict:
    baselines = [row["baseline_ndcg"] for row in rows]
    finals = [row["final_ndcg"] for row in rows]
    drops = [row["drop_pct"] for row in rows]
    recovered = [row for row in rows if row["final_above_recovery_threshold"]]
    cycles = [row["recovery_cycle"] for row in rows if row["recovery_cycle"] is not None]
    return {
        "condition": condition,
        "drift": drift,
        "run_count": len(rows),
        "baseline_mean": _mean(baselines),
        "baseline_std": _pstdev(baselines),
        "final_mean": _mean(finals),
        "final_std": _pstdev(finals),
        "drop_pct_mean": _mean(drops),
        "drop_pct_std": _pstdev(drops),
        "recovered_runs": len(recovered),
        "mean_recovery_cycle": _mean(cycles) if cycles else None,
    }


def _severity_analysis(grouped: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]]) -> dict:
    severity = {}
    for mode in DRIFT_MODES:
        c0_rows = list(grouped.get(("C0", mode), []))
        if not c0_rows:
            raise ValueError(f"Missing C0 rows for {mode}")
        severity[mode] = {
            "c0_baseline_mean": _mean(row["baseline_ndcg"] for row in c0_rows),
            "c0_final_mean": _mean(row["final_ndcg"] for row in c0_rows),
            "c0_drop_pct_mean": _mean(row["drop_pct"] for row in c0_rows),
            "c0_drop_pct_by_seed": {str(row["seed"]): row["drop_pct"] for row in c0_rows},
            "all_c0_above_recovery_threshold": all(row["final_above_recovery_threshold"] for row in c0_rows),
            "all_c0_recovered_cycle_one": all(row["recovery_cycle"] == 1 for row in c0_rows),
        }
    return severity


def _condition_traces(artifacts: Sequence[Mapping[str, Any]]) -> dict:
    cycle_rows = []
    for artifact in artifacts:
        config = artifact["config"]
        condition = config["condition"]
        if condition not in {"C3", "C4"}:
            continue
        for point in artifact["recovery"]["trajectory"]:
            metadata = point.get("metadata", {})
            observation = metadata.get("annealing_observation") or {}
            norm_stats = metadata.get("correction_norm_stats") or {}
            cycle_rows.append(
                {
                    "source_path": artifact.get("_source_path", ""),
                    "condition": condition,
                    "drift": config["drift"],
                    "seed": int(config["seed"]),
                    "cycle_index": int(point["cycle_index"]),
                    "ndcg": float(point["ndcg"]),
                    "drift_signal": _optional_float(observation.get("drift_magnitude")),
                    "should_correct": bool(metadata.get("should_correct", False)),
                    "sedimentation_attempted": bool(metadata.get("sedimentation_attempted", False)),
                    "correction_applied": bool(metadata.get("correction_applied", False)),
                    "correction_norm_mean": _optional_float(norm_stats.get("mean")),
                    "correction_norm_max": _optional_float(norm_stats.get("max")),
                }
            )
    summary = []
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in cycle_rows:
        groups[(row["condition"], row["drift"])].append(row)
    for condition in ("C3", "C4"):
        for drift in DRIFT_MODES:
            rows = groups[(condition, drift)]
            if not rows:
                continue
            summary.append(
                {
                    "condition": condition,
                    "drift": drift,
                    "cycle_count": len(rows),
                    "should_correct_true": sum(1 for row in rows if row["should_correct"]),
                    "sedimentation_attempted_true": sum(1 for row in rows if row["sedimentation_attempted"]),
                    "correction_applied_true": sum(1 for row in rows if row["correction_applied"]),
                    "mean_drift_signal": _mean(row["drift_signal"] for row in rows if row["drift_signal"] is not None),
                    "mean_correction_norm": _mean(
                        row["correction_norm_mean"] for row in rows if row["correction_norm_mean"] is not None
                    ),
                    "max_correction_norm": max(
                        row["correction_norm_max"] for row in rows if row["correction_norm_max"] is not None
                    ),
                }
            )
    return {"cycle_rows": cycle_rows, "summary": summary}


def _trajectory_means(artifacts: Sequence[Mapping[str, Any]]) -> dict:
    buckets: dict[str, dict[str, dict[int, list[float]]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for artifact in artifacts:
        config = artifact["config"]
        for point in artifact["recovery"]["trajectory"]:
            buckets[config["drift"]][config["condition"]][int(point["cycle_index"])].append(float(point["ndcg"]))
    output = {}
    for drift, by_condition in buckets.items():
        output[drift] = {}
        for condition, by_cycle in by_condition.items():
            output[drift][condition] = [
                {
                    "cycle_index": cycle,
                    "mean_ndcg": _mean(values),
                    "std_ndcg": _pstdev(values),
                    "run_count": len(values),
                }
                for cycle, values in sorted(by_cycle.items())
            ]
    return output


def _h1_verdict(severity: Mapping[str, Mapping[str, Any]]) -> dict:
    rotation = severity["rotation"]
    noise = severity["noise"]
    confirmed = (
        rotation["all_c0_recovered_cycle_one"]
        and rotation["all_c0_above_recovery_threshold"]
        and rotation["c0_drop_pct_mean"] < 5.0
    )
    verdict = "confirmed" if confirmed else "refuted"
    rotation_recovery = "all" if rotation["all_c0_recovered_cycle_one"] else "not all"
    evidence = (
        f"Rotation C0 mean drop is {rotation['c0_drop_pct_mean']:.3f}% and {rotation_recovery} "
        f"C0 rotation runs recover at cycle 1; noise C0 mean drop is {noise['c0_drop_pct_mean']:.3f}%."
    )
    return {"verdict": verdict, "evidence": evidence}


def _h2_verdict(traces: Mapping[str, Any]) -> dict:
    summaries = {(row["condition"], row["drift"]): row for row in traces["summary"]}
    c3_norms = [row["mean_correction_norm"] for row in summaries.values() if row["condition"] == "C3"]
    c4_norms = [row["mean_correction_norm"] for row in summaries.values() if row["condition"] == "C4"]
    c3_detection = [row for row in summaries.values() if row["condition"] == "C3"]
    c4_detection = [row for row in summaries.values() if row["condition"] == "C4"]
    c3_saturates = bool(c3_norms) and all(0.0099 <= value <= 0.0101 for value in c3_norms)
    c4_near_zero = bool(c4_norms) and all(value < 0.001 for value in c4_norms)
    detection_fired = all(row["should_correct_true"] == row["cycle_count"] for row in c3_detection + c4_detection)
    no_c3_applied = all(row["correction_applied_true"] == 0 for row in c3_detection)
    verdict = "confirmed" if c3_saturates and c4_near_zero and detection_fired else "refuted"
    evidence = (
        f"C3 mean correction norms are {[round(value, 9) for value in c3_norms]}; "
        f"C4 mean correction norms are {[round(value, 9) for value in c4_norms]}; "
        f"detection fired every C3/C4 cycle={detection_fired}; C3 applied vector updates={not no_c3_applied}."
    )
    return {"verdict": verdict, "evidence": evidence}


def _h3_verdict(grouped: Mapping[tuple[str, str], Sequence[Mapping[str, Any]]]) -> dict:
    c2_rows = list(grouped.get(("C2", "rotation"), [])) + list(grouped.get(("C2", "noise"), []))
    diffs = [abs(row["final_ndcg"] - row["baseline_ndcg"]) for row in c2_rows]
    confirmed = bool(diffs) and all(diff <= BASELINE_EQUAL_TOLERANCE for diff in diffs)
    verdict = "confirmed" if confirmed else "refuted"
    evidence = (
        f"C2 baseline-final absolute diffs across {len(diffs)} runs: max={max(diffs):.12g}, "
        f"tolerance={BASELINE_EQUAL_TOLERANCE:.0e}."
    )
    return {"verdict": verdict, "evidence": evidence}


def _validate_matrix(rows: Sequence[Mapping[str, Any]]) -> None:
    seen = {(row["condition"], row["drift"], row["seed"]) for row in rows}
    expected = {(condition, drift, seed) for condition in CONDITIONS for drift in DRIFT_MODES for seed in (42, 1337, 7)}
    missing = sorted(expected - seen)
    if missing:
        raise ValueError(f"Missing expected matrix cells: {missing[:5]}")


def _mean(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    if not materialized:
        return math.nan
    return float(statistics.fmean(materialized))


def _pstdev(values: Iterable[float]) -> float:
    materialized = [float(value) for value in values]
    if len(materialized) < 2:
        return 0.0
    return float(statistics.pstdev(materialized))


def _optional_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    return float(value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze drift-recovery JSON artifacts")
    parser.add_argument("--input-glob", default=DEFAULT_INPUT_GLOB)
    parser.add_argument("--output-json", default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--plot-dir", default=DEFAULT_PLOT_DIR)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    artifacts = load_artifacts(args.input_glob)
    diagnostics = analyze_artifacts(artifacts)
    outputs = write_diagnostics(diagnostics, args.output_json, args.output_md, args.plot_dir)
    print(json.dumps(outputs, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
