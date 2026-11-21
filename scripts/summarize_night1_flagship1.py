"""Aggregate S1, S2, and S3 Night-1 artifacts and evaluate all 3 Flagship #1 Go/Kill gates."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List


def _read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def summarize_flagship1(s1_dir: Path, s2_dir: Path, s3_dir: Path, output_md: Path) -> Dict[str, Any]:
    s1_manifest = json.loads((s1_dir / "s1_m1_m3_manifest.json").read_text(encoding="utf-8"))
    s1_m2 = json.loads((s1_dir / "s1_m2_preflight.json").read_text(encoding="utf-8"))
    s1_rows = _read_csv(s1_dir / "s1_m3_frontier.csv")

    s2_manifest = json.loads((s2_dir / "s2_multihop_manifest.json").read_text(encoding="utf-8"))
    s2_rows = _read_csv(s2_dir / "s2_spectral_errors.csv")

    s3_manifest = json.loads((s3_dir / "s3_tra1_m4_manifest.json").read_text(encoding="utf-8"))

    # Gate 1: M3 residual at budget <= 0.20 on REPORT (old_encoder_retired=True)
    # Group REPORT rows by (dataset, swap, arm) across seeds
    rep_by_cell: Dict[tuple[str, str, str], List[float]] = defaultdict(list)
    rep_gap_by_cell: Dict[tuple[str, str, str], List[float]] = defaultdict(list)
    rep_lcb_by_cell: Dict[tuple[str, str, str], List[float]] = defaultdict(list)
    for row in s1_rows:
        if row["split"] != "REPORT" or row["status"] != "OK":
            continue
        key = (row["dataset"], row["swap"], row["arm"])
        rep_by_cell[key].append(float(row["mean_ndcg_at_10"]))
        if row["c2o_minus_arm_ndcg"]:
            rep_gap_by_cell[key].append(float(row["c2o_minus_arm_ndcg"]))
        if row["c2o_minus_arm_lcb95"]:
            rep_lcb_by_cell[key].append(float(row["c2o_minus_arm_lcb95"]))

    # Find best ordinary arm at budget <= 0.20 per (dataset, swap)
    m3_summary: List[Dict[str, Any]] = []
    datasets_with_surviving_gap = set()
    for ds in s1_manifest["datasets"]:
        for swap in ("E0_to_E1_samedim", "E0_to_E3_crossdim"):
            eligible_arms = [
                arm for (d, sw, arm) in rep_by_cell
                if d == ds and sw == swap
                and arm not in ("full_reembed_c2o", "old_encoder_servable")
                and not arm.endswith("_50")
            ]
            c2o_mean = sum(rep_by_cell[(ds, swap, "full_reembed_c2o")]) / len(rep_by_cell[(ds, swap, "full_reembed_c2o")])
            best_arm = max(eligible_arms, key=lambda a: sum(rep_by_cell[(ds, swap, a)]) / len(rep_by_cell[(ds, swap, a)]))
            best_mean = sum(rep_by_cell[(ds, swap, best_arm)]) / len(rep_by_cell[(ds, swap, best_arm)])
            mean_gap = sum(rep_gap_by_cell[(ds, swap, best_arm)]) / len(rep_gap_by_cell[(ds, swap, best_arm)])
            min_lcb = min(rep_lcb_by_cell[(ds, swap, best_arm)])
            survives = (mean_gap >= float(s1_manifest["delta"])) and (min_lcb > 0.0)
            if survives:
                datasets_with_surviving_gap.add(ds)
            m3_summary.append({
                "dataset": ds,
                "swap": swap,
                "c2o_ndcg": round(c2o_mean, 4),
                "best_ordinary_le20_arm": best_arm,
                "best_ordinary_ndcg": round(best_mean, 4),
                "residual_gap": round(mean_gap, 4),
                "min_seed_lcb95": round(min_lcb, 4),
                "gap_survives": survives,
            })

    gate1_verdict = (
        "RESIDUAL_SURVIVES_M3"
        if len(datasets_with_surviving_gap) >= 2
        else "CLOSED_BY_ORDINARY_LADDER"
    )

    # Gate 2: Multi-hop hop m=3 composed Procrustes vs composed Ridge
    hop3_diffs: List[Dict[str, Any]] = []
    for ds in s2_manifest["datasets"]:
        sub = [r for r in s2_rows if r["dataset"] == ds and int(r["hop_m"]) == 3 and r["method"] == "composed_ridge"]
        if not sub:
            continue
        mean_proc_adv = sum(float(r["comp_proc_minus_comp_ridge_ndcg"]) for r in sub) / len(sub)
        mean_dir_adv = sum(float(r["dir_ridge_minus_comp_ridge_ndcg"]) for r in sub) / len(sub)
        hop3_diffs.append({
            "dataset": ds,
            "mean_comp_proc_minus_comp_ridge": round(mean_proc_adv, 4),
            "mean_dir_ridge_minus_comp_ridge": round(mean_dir_adv, 4),
            "rows": len(sub),
        })
    gate2_verdict = (
        "MULTIHOP_SIGNAL_SURVIVES"
        if any(max(x["mean_comp_proc_minus_comp_ridge"], x["mean_dir_ridge_minus_comp_ridge"]) >= 0.03 for x in hop3_diffs)
        else "MULTIHOP_NULL_OR_SMALL"
    )

    # Gate 3: TRA-1 hostile audit + M4 QPP comparison
    hostile = s3_manifest["hostile_wrong_space_audit"]
    fam_res = s3_manifest["family_results"]
    worst_tra1_lcb = min(float(r["tra1_minus_fixed_lcb975"]) for r in fam_res)
    mean_tra1_minus_qpp = sum(float(r["tra1_minus_qpp"]) for r in fam_res) / len(fam_res)
    gate3_verdict = (
        "TRA1_DETERMINISTIC_INTERLOCK_PASSED"
        if (hostile["wrong_space_executions"] == 0 and worst_tra1_lcb > -0.02 and mean_tra1_minus_qpp >= -0.005)
        else "TRA1_OR_M4_NEEDS_REVIEW"
    )

    lines = [
        "# Night-1 Flagship #1 Empirical Verdict (Unquarantined Complete-Qrels `mlr-v1`)",
        "",
        f"- **Git SHA:** `{s1_manifest['git_sha']}`",
        f"- **Old encoder retired premise:** `{s1_manifest['old_encoder_retired']}`",
        f"- **Gate 1 (M3 Single-Hop Residual Science Gate):** **`{gate1_verdict}`**",
        f"- **Gate 2 (Multi-Hop Lineage Accumulation Gate):** **`{gate2_verdict}`**",
        f"- **Gate 3 (TRA-1 Interlock & M4 QPP Router Gate):** **`{gate3_verdict}`**",
        "",
        "## 1. Single-Hop M3 Frontier Summary (REPORT Split, Budget <= 20%)",
        "",
        "| Dataset | Swap | C2O nDCG@10 | Best Ordinary (<=20%) | Best Ord nDCG@10 | Residual Gap | Min Seed LCB95 | Survives (>=0.02 & LCB>0)? |",
        "|---|---|---:|---|---:|---:|---:|---|",
    ]
    for row in m3_summary:
        lines.append(
            f"| {row['dataset']} | {row['swap']} | {row['c2o_ndcg']:.4f} | `{row['best_ordinary_le20_arm']}` "
            f"| {row['best_ordinary_ndcg']:.4f} | {row['residual_gap']:+.4f} | {row['min_seed_lcb95']:+.4f} | {row['gap_survives']} |"
        )

    lines += [
        "",
        "## 2. Multi-Hop Lineage Accumulation at Hop m=3 (E3 -> E2 -> E1 -> E0)",
        "",
        "| Dataset | Composed Procrustes − Composed Ridge | Direct Ridge − Composed Ridge | Cells |",
        "|---|---:|---:|---:|",
    ]
    for row in hop3_diffs:
        lines.append(
            f"| {row['dataset']} | {row['mean_comp_proc_minus_comp_ridge']:+.4f} "
            f"| {row['mean_dir_ridge_minus_comp_ridge']:+.4f} | {row['rows']} |"
        )

    lines += [
        "",
        "## 3. TRA-1 Interlock & M4 QPP Router Audit",
        "",
        f"- Hostile wrong-space negative cases: **{hostile['total_hostile_cases']}**, wrong-space executions: **{hostile['wrong_space_executions']}**",
        f"- Worst family TRA-1 minus fixed LCB97.5: **{worst_tra1_lcb:+.4f}**",
        f"- Mean TRA-1 minus authorization-filtered QPP router: **{mean_tra1_minus_qpp:+.4f}**",
        "",
        "| Dataset | Mutation Family | TRA-1 | QPP (M4) | Best Fixed | Native Re-embed | TRA-1 − QPP |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for r in fam_res:
        lines.append(
            f"| {r['dataset']} | {r['mutation_family']} | {r['tra1_mean_ndcg']:.4f} | {r['qpp_mean_ndcg']:.4f} "
            f"| {r['fixed_mean_ndcg']:.4f} | {r['native_reembed_mean_ndcg']:.4f} | {r['tra1_minus_qpp']:+.4f} |"
        )

    lines += [
        "",
        f"## 4. M2 Development Preflight Summary ({len(s1_m2)} cells)",
        f"- ADMIT cells: **{sum(1 for x in s1_m2 if x['decision'] == 'ADMIT')}** / {len(s1_m2)}",
        f"- REFUSE cells: **{sum(1 for x in s1_m2 if x['decision'] == 'REFUSE')}** / {len(s1_m2)}",
        "",
    ]

    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines), encoding="utf-8")
    return {
        "gate1_verdict": gate1_verdict,
        "gate2_verdict": gate2_verdict,
        "gate3_verdict": gate3_verdict,
        "m3_summary": m3_summary,
        "hop3_diffs": hop3_diffs,
        "hostile_wrong_space_executions": hostile["wrong_space_executions"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize Night-1 Flagship #1 results")
    parser.add_argument("--s1-dir", type=Path, required=True)
    parser.add_argument("--s2-dir", type=Path, required=True)
    parser.add_argument("--s3-dir", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize_flagship1(args.s1_dir, args.s2_dir, args.s3_dir, args.output_md)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
