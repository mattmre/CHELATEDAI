"""PR-A4 campaign: the query-encoder-upgrade (swap) arena on real data.

Runs the headline drift-recovery comparison on the ``query_encoder_swap`` arena
where an encoder upgrade defeats the C2 re-embed oracle (queries move to a new
encoder; cached doc vectors stay in the original space):

  * Main matrix: conditions {C0, C2, C2O, C3a, C4a} x seeds, anchor_fraction set,
    all scored on the SAME held-out eval subset, so the supervised closed loop
    (C3a/C4a) is directly comparable to the frozen baseline (C0), the no-op
    maintenance baseline (C2), and the expensive oracle upper bound (C2O).
  * C3a training-budget sweep: (steps, lr) grid at one seed. Tier B (PR-A2b)
    showed the default 30-step / lr-0.01 budget under-fits even in-sample, so the
    budget MUST be swept before concluding the supervised loop does not recover.
    The best budget cell is then confirmed across all seeds.

This driver only orchestrates ``run_experiment``; it adds no modeling logic. For
a real-model run set HF offline env vars and use the real swap model, e.g.:

    HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 python run_drift_recovery_swap_campaign.py
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

from run_drift_recovery_experiment import DriftRecoveryConfig, _detect_device, run_experiment


MAIN_CONDITIONS = ("C0", "C2", "C2O", "C3a", "C4a")
SEEDS = (42, 1337, 7)
# (steps, lr) budget grid for the C3a sweep — spans the under-fitting default up
# to the high-budget regime Tier B showed reaches full in-sample recovery.
BUDGET_GRID = ((30, 0.01), (200, 0.05), (1000, 0.10), (2000, 0.10))
SWEEP_SEED = 42
DEFAULT_OUTPUT_DIR = "experiment_runs/drift-recovery/swap"
DEFAULT_REPORT_MD = "docs/drift-recovery-swap-results-2026-06.md"


@dataclass(frozen=True)
class SwapCampaignConfig:
    output_dir: str = DEFAULT_OUTPUT_DIR
    report_md: str = DEFAULT_REPORT_MD
    task: str = "SciFact"
    max_queries: int = 100
    sample_docs: int = 1200
    cycles: int = 12
    anchor_fraction: float = 0.4
    model: str = "sentence-transformers/all-MiniLM-L6-v2"
    swap_model: str = "all-mpnet-base-v2"
    device: Optional[str] = None


def run_swap_campaign(
    config: SwapCampaignConfig,
    runner: Callable[[DriftRecoveryConfig], Mapping[str, Any]] = run_experiment,
) -> dict:
    started = time.perf_counter()
    output_dir = Path(config.output_dir)
    (output_dir / "main").mkdir(parents=True, exist_ok=True)
    (output_dir / "budget").mkdir(parents=True, exist_ok=True)
    (output_dir / "budget-confirm").mkdir(parents=True, exist_ok=True)

    # --- Main matrix: conditions x seeds ------------------------------------
    main_rows = []
    for condition in MAIN_CONDITIONS:
        for seed in SEEDS:
            run_config = _run_config(
                config,
                condition=condition,
                seed=seed,
                output=output_dir / "main" / f"{condition}_seed{seed}.json",
            )
            result = runner(run_config)
            main_rows.append(_row(result, phase="main"))

    # --- C3a training-budget sweep (one seed) -------------------------------
    budget_rows = []
    for steps, lr in BUDGET_GRID:
        run_config = _run_config(
            config,
            condition="C3a",
            seed=SWEEP_SEED,
            output=output_dir / "budget" / f"c3a_steps{steps}_lr{_fmt(lr)}_seed{SWEEP_SEED}.json",
            correction_steps=steps,
            correction_lr=lr,
        )
        result = runner(run_config)
        budget_rows.append(_row(result, phase="budget"))

    # --- Confirm the best budget cell across all seeds ----------------------
    best_budget = _select_best_budget(budget_rows)
    budget_confirm_rows = []
    for seed in SEEDS:
        run_config = _run_config(
            config,
            condition="C3a",
            seed=seed,
            output=output_dir / "budget-confirm"
            / f"c3a_steps{best_budget['steps']}_lr{_fmt(best_budget['lr'])}_seed{seed}.json",
            correction_steps=int(best_budget["steps"]),
            correction_lr=float(best_budget["lr"]),
        )
        result = runner(run_config)
        budget_confirm_rows.append(_row(result, phase="budget-confirm"))

    manifest = {
        "record_type": "drift_recovery_swap_campaign",
        "config": asdict(config),
        "arena": "query_encoder_swap",
        "conditions": list(MAIN_CONDITIONS),
        "seeds": list(SEEDS),
        "budget_grid": [{"steps": s, "lr": lr} for s, lr in BUDGET_GRID],
        "sweep_seed": SWEEP_SEED,
        "main_rows": main_rows,
        "main_summary": _summary_by_condition(main_rows),
        "budget_rows": budget_rows,
        "best_budget": best_budget,
        "budget_confirm_rows": budget_confirm_rows,
        "budget_confirm_summary": _summary_by_condition(budget_confirm_rows),
        "wall_clock_seconds": time.perf_counter() - started,
    }
    manifest_path = output_dir / "swap-campaign-manifest-2026-06.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    Path(config.report_md).parent.mkdir(parents=True, exist_ok=True)
    Path(config.report_md).write_text(render_swap_report(manifest), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "report": config.report_md}, sort_keys=True))
    return manifest


def _run_config(
    config: SwapCampaignConfig,
    condition: str,
    seed: int,
    output: Path,
    correction_steps: int = 30,
    correction_lr: float = 0.01,
) -> DriftRecoveryConfig:
    # C0/C2/C2O do not use anchors but share the anchor_fraction so the split is
    # active and every condition is scored on the SAME eval subset.
    return DriftRecoveryConfig(
        task=config.task,
        condition=condition,
        drift="query_encoder_swap",
        fraction=0.5,
        angle=25.0,
        sigma=0.05,
        cycles=config.cycles,
        seed=seed,
        max_queries=config.max_queries,
        sample_docs=config.sample_docs,
        model=config.model,
        output=str(output),
        device=config.device or _detect_device(),
        swap_model=config.swap_model,
        anchor_fraction=config.anchor_fraction,
        correction_steps=correction_steps,
        correction_lr=correction_lr,
    )


def _row(result: Mapping[str, Any], phase: str) -> dict:
    run_config = result["config"]
    trajectory = result["recovery"]["trajectory"]
    final = float(trajectory[-1]["ndcg"])
    baseline = float(result["baseline"]["ndcg_at_10"])
    cycle_meta = [point.get("metadata", {}) for point in trajectory]
    return {
        "phase": phase,
        "condition": str(run_config["condition"]),
        "seed": int(run_config["seed"]),
        "correction_steps": int(run_config.get("correction_steps", 30)),
        "correction_lr": float(run_config.get("correction_lr", 0.01)),
        "anchor_count": int(result.get("anchor_eval_split", {}).get("anchor_count", 0)),
        "eval_count": int(result.get("anchor_eval_split", {}).get("eval_count", 0)),
        "baseline_ndcg": baseline,
        "final_ndcg": final,
        "recovery_cycle": result["recovery"].get("recovery_cycle"),
        "cycle_count": len(trajectory),
        "should_correct_true": sum(1 for row in cycle_meta if row.get("should_correct")),
        "correction_applied_true": sum(1 for row in cycle_meta if row.get("correction_applied")),
        "mean_correction_norm": result["correction_norm_stats"]["mean"],
        "path": str(run_config["output"]).replace("\\", "/"),
    }


def _summary_by_condition(rows: Sequence[Mapping[str, Any]]) -> list:
    groups: dict = {}
    for row in rows:
        key = (row["condition"], row["correction_steps"], row["correction_lr"])
        groups.setdefault(key, []).append(row)
    summary = []
    for (condition, steps, lr), cells in sorted(groups.items()):
        finals = [cell["final_ndcg"] for cell in cells]
        summary.append(
            {
                "condition": condition,
                "correction_steps": steps,
                "correction_lr": lr,
                "run_count": len(cells),
                "final_mean": float(statistics.fmean(finals)),
                "final_std": float(statistics.pstdev(finals)) if len(finals) > 1 else 0.0,
                "baseline_mean": float(statistics.fmean(cell["baseline_ndcg"] for cell in cells)),
                "recovered_runs": sum(1 for cell in cells if cell["recovery_cycle"] is not None),
                "applied_runs": sum(1 for cell in cells if cell["correction_applied_true"] > 0),
            }
        )
    return summary


def _select_best_budget(rows: Sequence[Mapping[str, Any]]) -> dict:
    best = max(rows, key=lambda row: float(row["final_ndcg"]))
    return {"steps": int(best["correction_steps"]), "lr": float(best["correction_lr"]), "final_ndcg": float(best["final_ndcg"])}


def render_swap_report(manifest: Mapping[str, Any]) -> str:
    lines = [
        "# Drift Recovery — Query-Encoder-Swap Arena (PR-A4) — June 2026",
        "",
        f"Arena: `{manifest['arena']}` (encoder upgrade; the C2 re-embed oracle is defeated).",
        f"Manifest: `{manifest['config']['output_dir']}/swap-campaign-manifest-2026-06.json`.",
        f"Base model: `{manifest['config']['model']}`; swap model: `{manifest['config']['swap_model']}`.",
        f"Task {manifest['config']['task']}, anchor_fraction {manifest['config']['anchor_fraction']}, "
        f"cycles {manifest['config']['cycles']}, seeds {manifest['seeds']}.",
        "",
        "Conditions: C0 frozen (lower bound) · C2 re-embed-with-original (proven no-op) · "
        "C2O oracle re-embed into new space (upper bound) · C3a supervised bounded adapter · "
        "C4a supervised unbounded adapter. All scored on the SAME held-out eval subset.",
        "",
        "## Main Matrix (mean over seeds)",
        "",
        "| Condition | Baseline NDCG | Final NDCG | Std | Recovery@N | Applied runs |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in manifest["main_summary"]:
        lines.append(
            f"| {row['condition']} | {row['baseline_mean']:.6f} | {row['final_mean']:.6f} | "
            f"{row['final_std']:.6f} | {row['recovered_runs']}/{row['run_count']} | "
            f"{row['applied_runs']}/{row['run_count']} |"
        )
    lines.extend(["", "## C3a Training-Budget Sweep (seed " + str(manifest["sweep_seed"]) + ")", ""])
    lines.extend(
        [
            "| Steps | LR | Final NDCG | should_correct | applied | mean norm |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in manifest["budget_rows"]:
        lines.append(
            f"| {row['correction_steps']} | {row['correction_lr']:.2f} | {row['final_ndcg']:.6f} | "
            f"{row['should_correct_true']}/{row['cycle_count']} | "
            f"{row['correction_applied_true']}/{row['cycle_count']} | {row['mean_correction_norm']:.6f} |"
        )
    best = manifest["best_budget"]
    lines.extend(
        [
            "",
            f"Best budget cell: steps={best['steps']}, lr={best['lr']:.2f} (seed-{manifest['sweep_seed']} "
            f"final {best['final_ndcg']:.6f}). Three-seed confirmation:",
            "",
            "| Steps | LR | Mean final NDCG | Std | Recovery@N |",
            "|---:|---:|---:|---:|---:|",
        ]
    )
    for row in manifest["budget_confirm_summary"]:
        lines.append(
            f"| {row['correction_steps']} | {row['correction_lr']:.2f} | {row['final_mean']:.6f} | "
            f"{row['final_std']:.6f} | {row['recovered_runs']}/{row['run_count']} |"
        )
    lines.append("")
    return "\n".join(lines)


def _fmt(value: float) -> str:
    return ("%g" % value).replace(".", "p")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the query-encoder-swap drift-recovery campaign")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--report-md", default=DEFAULT_REPORT_MD)
    parser.add_argument("--task", default="SciFact")
    parser.add_argument("--max-queries", type=int, default=100)
    parser.add_argument("--sample-docs", type=int, default=1200)
    parser.add_argument("--cycles", type=int, default=12)
    parser.add_argument("--anchor-fraction", type=float, default=0.4)
    parser.add_argument("--model", default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--swap-model", default="all-mpnet-base-v2")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    config = SwapCampaignConfig(
        output_dir=args.output_dir,
        report_md=args.report_md,
        task=args.task,
        max_queries=args.max_queries,
        sample_docs=args.sample_docs,
        cycles=args.cycles,
        anchor_fraction=args.anchor_fraction,
        model=args.model,
        swap_model=args.swap_model,
        device=_detect_device(),
    )
    run_swap_campaign(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
