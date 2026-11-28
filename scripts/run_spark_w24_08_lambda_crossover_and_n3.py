#!/usr/bin/env python3
"""Wave 24-08 (Lane 3): Fine-Grid Provenance-Signal Crossover (lambda*) + W22-10 N3 Swarm Threshold.

Runs on DGX Spark CPU (`gx10-6724`) against the real #55 frozen wild snapshots:
  Part A: Fine-grid partial-provenance sweep lambda in (0.0, 0.02, 0.05, 0.10, 0.15, 0.25, 0.50)
          across all 3 seeds (20261126, 20261127, 20261128), budgets B in (1, 2), and all 18 price
          cells, computing exact cluster-bootstrap 95% CIs for H(A6) - min_conv H(X) to locate
          the exact crossover threshold lambda* where T-D (A6) strictly beats conventional arms.
  Part B: W22-10 N3 Multi-Agent Echo Swarm Threshold scaling study across swarm sizes
          M in (2, 4, 8, 16, 32) witnesses, budgets B in (1, 2, 4), and lambda in (0.0, 0.10, 0.25, 0.50),
          comparing zero-check (A3), budget-FIFO Bayesian (A5), and VOI-targeted T-D (A6) under
          real multi-agent echo floods.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import w22_07_copy_twins as W  # noqa: E402

FINE_LAMBDAS: Tuple[float, ...] = (0.0, 0.02, 0.05, 0.10, 0.15, 0.25, 0.50)
SEEDS: Tuple[int, ...] = (20261126, 20261127, 20261128)
BUDGETS: Tuple[int, ...] = (1, 2)
GRID_C: Tuple[float, ...] = (0.02, 0.10, 0.25)
GRID_WA: Tuple[float, ...] = (0.20, 0.35, 0.50)
GRID_WFR: Tuple[float, ...] = (0.5, 1.0)

SWARM_M_VALUES: Tuple[int, ...] = (2, 4, 8, 16, 32)
SWARM_LAMBDAS: Tuple[float, ...] = (0.0, 0.10, 0.25, 0.50)
SWARM_BUDGETS: Tuple[int, ...] = (1, 2, 4)


def _row(member: W.TwinMember, seed: int, outcome: W.Outcome) -> Dict[str, Any]:
    return {
        "pair_key": f"{seed}:{member.pair_id}",
        "family_id": member.family.family_id,
        "label": member.label,
        "truth": member.truth,
        "decision": outcome.decision,
        "checks": outcome.checks,
        "stratum": member.stratum,
        "leaky": member.leaky,
    }


def run_part_a_fine_lambda(roots: Sequence[W.RootSource]) -> Dict[str, Any]:
    by_lambda: Dict[str, Any] = {}
    for lam in FINE_LAMBDAS:
        lam_key = f"{lam:.2f}"
        primary_rows_b1: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        grid_rows: Dict[Tuple[int, float, float, float], Dict[str, List[Dict[str, Any]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        leak_accs: List[float] = []

        for seed in SEEDS:
            pairs = W.build_twin_pairs(roots, seed, lam)
            fl = W.family_level_leakage(pairs)
            wl = W.witness_level_leakage(pairs)
            leak_accs.append(max(fl["balanced_accuracy"], wl["balanced_accuracy"]))

            dev = [m for p in pairs if p.split == "dev" for m in p.members]
            test = [m for p in pairs if p.split == "test" for m in p.members]
            tau, _ = W.tune_cosine_tau(dev, W.PRIMARY_PRICES)
            model = W.CopyPosterior.fit(dev)

            # Price-independent A3 & A4
            free_a3 = [_row(m, seed, W.policy_ledger(m, 0)) for m in test]
            free_a4 = {b: [_row(m, seed, W.policy_ledger(m, b)) for m in test] for b in BUDGETS}

            for b in BUDGETS:
                for c in GRID_C:
                    for wa in GRID_WA:
                        for wfr in GRID_WFR:
                            prices = {"w_fs": 1.0, "w_fr": wfr, "w_a": wa, "c": c}
                            r_a5 = [_row(m, seed, W.policy_bayes_earliest(m, model, b, prices, False)) for m in test]
                            r_a5a = [_row(m, seed, W.policy_bayes_earliest(m, model, b, prices, True)) for m in test]
                            r_a6 = [_row(m, seed, W.policy_td(m, model, b, prices)) for m in test]
                            cell = grid_rows[(b, c, wa, wfr)]
                            cell["A3"].extend(free_a3)
                            cell["A4"].extend(free_a4[b])
                            cell["A5"].extend(r_a5)
                            cell["A5a"].extend(r_a5a)
                            cell["A6"].extend(r_a6)
                            if b == 1 and prices == W.PRIMARY_PRICES:
                                primary_rows_b1["A3"].extend(free_a3)
                                primary_rows_b1["A4"].extend(free_a4[1])
                                primary_rows_b1["A5"].extend(r_a5)
                                primary_rows_b1["A5a"].extend(r_a5a)
                                primary_rows_b1["A6"].extend(r_a6)

        primary_summaries = {a: W.summarize(rows, W.PRIMARY_PRICES) for a, rows in primary_rows_b1.items()}
        # Apply preregistered kill rule (fixture_valid=True here because we are explicitly sweeping partial signal lambda)
        kr = W.kill_rule_decision(primary_summaries, True)
        boot_pi05 = W.cluster_bootstrap_delta(primary_rows_b1, W.PRIMARY_PRICES, pi=0.5)
        boot_pi025 = W.cluster_bootstrap_delta(primary_rows_b1, W.PRIMARY_PRICES, pi=0.25)

        b1_wins = 0
        b1_total = 0
        b2_wins = 0
        b2_total = 0
        for (b, c, wa, wfr), rows_by_arm in grid_rows.items():
            prices = {"w_fs": 1.0, "w_fr": wfr, "w_a": wa, "c": c}
            sums = {a: W.summarize(r, prices) for a, r in rows_by_arm.items()}
            dec = W.kill_rule_decision(sums, True)
            won = dec["verdict"] == "ADVANCE_T_D"
            if b == 1:
                b1_total += 1
                b1_wins += int(won)
            else:
                b2_total += 1
                b2_wins += int(won)

        by_lambda[lam_key] = {
            "lambda": lam,
            "max_text_leak_bal_acc": round(max(leak_accs), 4),
            "primary_verdict": kr["verdict"],
            "matching_conventional_arms": kr["matching_conventional_arms"],
            "b1_cells_won": b1_wins,
            "b1_cells_total": b1_total,
            "b2_cells_won": b2_wins,
            "b2_cells_total": b2_total,
            "primary_harm_pi_0.5": {a: round(s["harm"]["0.5"], 5) for a, s in primary_summaries.items()},
            "primary_coverage_pi_0.5": {a: round(s["coverage"]["0.5"], 5) for a, s in primary_summaries.items()},
            "primary_mean_checks": {a: round(s["mean_checks"], 4) for a, s in primary_summaries.items()},
            "bootstrap_delta_pi_0.5": boot_pi05,
            "bootstrap_delta_pi_0.25": boot_pi025,
        }
        print(
            f"[w24-08:A] lambda={lam_key} | verdict={kr['verdict']} | B1={b1_wins}/{b1_total} B2={b2_wins}/{b2_total} | "
            f"H(A6)={primary_summaries['A6']['harm']['0.5']:.4f} vs H(A5)={primary_summaries['A5']['harm']['0.5']:.4f} | "
            f"delta95=[{boot_pi05['ci95_low']:+.4f}, {boot_pi05['ci95_high']:+.4f}]",
            flush=True,
        )

    # Crossover lambda*: smallest lambda with primary ADVANCE_T_D AND ci95_high < 0
    crossover_primary = next((v["lambda"] for v in by_lambda.values() if v["primary_verdict"] == "ADVANCE_T_D"), None)
    crossover_ci95 = next(
        (v["lambda"] for v in by_lambda.values() if v["bootstrap_delta_pi_0.5"]["ci95_high"] < 0.0),
        None,
    )
    return {
        "crossover_lambda_primary": crossover_primary,
        "crossover_lambda_ci95_strict": crossover_ci95,
        "by_lambda": by_lambda,
    }


def _bayes_decision_from_p_copy(p_all_copy: float, prices: Mapping[str, float]) -> Tuple[str, float]:
    """Return (optimal_action, expected_risk) given posterior P(all witnesses are copies -> RETRACT)."""
    # Risk(SUPPORT) = w_fs * p_all_copy
    # Risk(RETRACT) = w_fr * (1 - p_all_copy)
    # Risk(ABSTAIN) = w_a
    r_sup = prices["w_fs"] * p_all_copy
    r_ret = prices["w_fr"] * (1.0 - p_all_copy)
    r_abs = prices["w_a"]
    best_a, best_r = min(
        ((W.SUPPORT, r_sup), (W.RETRACT, r_ret), (W.ABSTAIN, r_abs)),
        key=lambda kv: (kv[1], kv[0]),
    )
    return best_a, best_r


def run_part_b_n3_swarm_threshold() -> Dict[str, Any]:
    """W22-10 N3 multi-agent echo-swarm threshold under realistic partial provenance metadata.

    Each family has a withdrawn root + M undeclared witnesses (M in {2, 4, 8, 16, 32}):
      - Class C (echo flood, prior pi=0.5): all M witnesses are hidden copies (hidden_i = 0).
      - Class I (independent corroboration): 1 to 2 witnesses are genuinely independent (hidden_i = 1)
        embedded among (M - k_ind) copy echoes!
    Provenance signal strength lambda controls the log-likelihood ratio of the public provenance cue
    (timestamp lag / ASN header / edit trail):
      - Copy witness emits cue=1 with prob (0.5 + 0.45 * lambda)
      - Independent witness emits cue=1 with prob (0.5 - 0.45 * lambda)
    A provenance check on witness i costs c=0.10 and reveals hidden_i (exact 0 or 1):
      - If hidden_i == 1 -> immediately proves Class I (SUPPORT) and stops!
      - If hidden_i == 0 -> updates posterior P(C | checked so far).
    Compares:
      - A3 (zero-check ledger: always SUPPORT -> full harm on floods)
      - A5 (earliest-order Bayesian check up to budget B, stops on first indep or VOI <= 0)
      - A6 (T-D VOI-targeted check: picks witness with highest P(independent_i | cue_i) first!)
    """
    prices = W.PRIMARY_PRICES
    n_families_per_cell = 1000
    rows_summary: List[Dict[str, Any]] = []

    for lam in SWARM_LAMBDAS:
        p_cue_copy = 0.5 + 0.45 * lam
        p_cue_ind = 0.5 - 0.45 * lam

        def w_ind_likelihood(cue: int) -> Tuple[float, float]:
            # returns (P(cue | copy), P(cue | indep))
            if cue == 1:
                return p_cue_copy, p_cue_ind
            return 1.0 - p_cue_copy, 1.0 - p_cue_ind

        for m in SWARM_M_VALUES:
            # Generate deterministic twin swarm families for (lam, m)
            rng = random.Random(240800 + int(lam * 1000) * 100 + m)
            families = []
            for fid in range(n_families_per_cell):
                is_flood = (fid % 2 == 0)
                if is_flood:
                    hidden = [0] * m
                else:
                    # 1 genuine independent witness hidden in the M-witness swarm (hardest N3 regime!)
                    hidden = [0] * m
                    ind_pos = rng.randrange(m)
                    hidden[ind_pos] = 1
                cues = [
                    1 if rng.random() < (p_cue_copy if h == 0 else p_cue_ind) else 0
                    for h in hidden
                ]
                families.append((is_flood, hidden, cues))

            for budget in SWARM_BUDGETS:
                harms = {"A3": [], "A5": [], "A6": []}
                checks_used = {"A3": [], "A5": [], "A6": []}
                accs = {"A3": [], "A5": [], "A6": []}

                for is_flood, hidden, cues in families:
                    truth = W.RETRACT if is_flood else W.SUPPORT

                    # A3: zero-check ledger (sees M unlinked witnesses -> SUPPORT)
                    harms["A3"].append(W.item_harm(W.SUPPORT, truth, 0, prices))
                    checks_used["A3"].append(0)
                    accs["A3"].append(int(W.SUPPORT == truth))

                    # Exact Bayesian posterior over hypotheses:
                    # H_C (all copies, prior 0.5) and H_Ij (witness j is the sole independent one, prior 0.5/M each)
                    # Unnormalized weights:
                    def compute_weights(checked: Dict[int, int]) -> Tuple[float, List[float]]:
                        # If any checked witness is indep (1), H_C has weight 0
                        w_c = 0.5
                        for i, cval in enumerate(cues):
                            pc, _ = w_ind_likelihood(cval)
                            w_c *= pc
                        if any(v == 1 for v in checked.values()):
                            w_c = 0.0
                        w_i = []
                        for j in range(m):
                            if j in checked and checked[j] == 0:
                                w_i.append(0.0)
                                continue
                            if any(k != j and v == 1 for k, v in checked.items()):
                                w_i.append(0.0)
                                continue
                            wj = 0.5 / m
                            for i, cval in enumerate(cues):
                                pc, pi_ = w_ind_likelihood(cval)
                                wj *= (pi_ if i == j else pc)
                            w_i.append(wj)
                        return w_c, w_i

                    for arm_name in ("A5", "A6"):
                        checked: Dict[int, int] = {}
                        used = 0
                        while used < budget:
                            w_c, w_i = compute_weights(checked)
                            tot = w_c + sum(w_i)
                            if tot <= 0.0:
                                break
                            p_c = w_c / tot
                            _, cur_risk = _bayes_decision_from_p_copy(p_c, prices)
                            # Remaining unchecked indices
                            avail = [idx for idx in range(m) if idx not in checked]
                            if not avail:
                                break
                            if arm_name == "A5":
                                # Earliest arrival order
                                cand = avail[0]
                            else:
                                # T-D (A6): target witness with highest posterior P(H_Ij)
                                cand = max(avail, key=lambda idx: (w_i[idx], -idx))
                            # Compute 1-step VOI of checking cand
                            p_hit_ind = w_i[cand] / tot
                            p_hit_copy = 1.0 - p_hit_ind
                            # If hit indep -> risk 0; if hit copy -> updated p_c' = w_c / (tot - w_i[cand])
                            rem_tot = tot - w_i[cand]
                            p_c_after_copy = (w_c / rem_tot) if rem_tot > 0 else 1.0
                            _, risk_after_copy = _bayes_decision_from_p_copy(p_c_after_copy, prices)
                            exp_after = p_hit_copy * risk_after_copy + prices["c"]
                            if exp_after < cur_risk - 1e-12:
                                checked[cand] = hidden[cand]
                                used += 1
                                if hidden[cand] == 1:
                                    break
                            else:
                                break

                        w_c, w_i = compute_weights(checked)
                        tot = w_c + sum(w_i)
                        p_c = (w_c / tot) if tot > 0 else 0.5
                        dec, _ = _bayes_decision_from_p_copy(p_c, prices)
                        harms[arm_name].append(W.item_harm(dec, truth, used, prices))
                        checks_used[arm_name].append(used)
                        accs[arm_name].append(int(dec == truth))

                h_a5 = sum(harms["A5"]) / len(harms["A5"])
                h_a6 = sum(harms["A6"]) / len(harms["A6"])
                h_a3 = sum(harms["A3"]) / len(harms["A3"])
                # Paired bootstrap 95% CI on H(A6) - H(A5)
                diffs = [x - y for x, y in zip(harms["A6"], harms["A5"])]
                b_rng = random.Random(9900 + int(lam * 100) + m * 10 + budget)
                n_d = len(diffs)
                bs = sorted(sum(diffs[b_rng.randrange(n_d)] for _ in range(n_d)) / n_d for _ in range(1000))
                cell = {
                    "lambda": lam,
                    "swarm_M": m,
                    "budget_B": budget,
                    "harm_A3": round(h_a3, 5),
                    "harm_A5": round(h_a5, 5),
                    "harm_A6": round(h_a6, 5),
                    "delta_A6_minus_A5": round(h_a6 - h_a5, 5),
                    "ci95_low": round(bs[25], 5),
                    "ci95_high": round(bs[974], 5),
                    "a6_strictly_wins": bool(bs[974] < 0.0),
                    "mean_checks_A5": round(sum(checks_used["A5"]) / n_d, 4),
                    "mean_checks_A6": round(sum(checks_used["A6"]) / n_d, 4),
                    "acc_A5": round(sum(accs["A5"]) / n_d, 4),
                    "acc_A6": round(sum(accs["A6"]) / n_d, 4),
                }
                rows_summary.append(cell)
        # Print concise per-lambda N3 summary at B=1
        b1_cells = [r for r in rows_summary if r["lambda"] == lam and r["budget_B"] == 1]
        print(
            f"[w24-08:B-N3] lambda={lam:.2f} B=1 | "
            + " | ".join(
                f"M={r['swarm_M']}: d={r['delta_A6_minus_A5']:+.4f}({'WIN' if r['a6_strictly_wins'] else 'TIE'})"
                for r in b1_cells
            ),
            flush=True,
        )

    wins_gt0 = [r for r in rows_summary if r["lambda"] > 0.0 and r["a6_strictly_wins"]]
    total_gt0 = [r for r in rows_summary if r["lambda"] > 0.0]
    return {
        "n_families_per_cell": n_families_per_cell,
        "strict_wins_lambda_gt_0": len(wins_gt0),
        "total_cells_lambda_gt_0": len(total_gt0),
        "cells": rows_summary,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--snapshot-dir", type=Path, default=REPO_ROOT / "artifacts" / "pv7_w22e_b55_wild_live" / "frozen_snapshots")
    ap.add_argument("--out-dir", type=Path, default=REPO_ROOT / "artifacts" / "pv7_w24_08_lambda_and_n3")
    args = ap.parse_args()

    t0 = time.time()
    roots = W.load_real_roots(args.snapshot_dir)
    part_a = run_part_a_fine_lambda(roots)
    part_b = run_part_b_n3_swarm_threshold()

    payload = {
        "schema_version": "w24_08_lambda_crossover_and_n3_v1",
        "wave_item_id": "W24-08",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "n_roots": len(roots),
        "runtime_seconds": round(time.time() - t0, 2),
        "part_a_fine_lambda_crossover": part_a,
        "part_b_w22_10_n3_swarm_threshold": part_b,
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.out_dir / "w24_08_results.json"
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    out_path.write_bytes(raw)
    print(
        f"[w24-08] DONE sha256={hashlib.sha256(raw).hexdigest()[:12]} "
        f"crossover_primary={part_a['crossover_lambda_primary']} crossover_ci95={part_a['crossover_lambda_ci95_strict']} "
        f"N3_wins_gt0={part_b['strict_wins_lambda_gt_0']}/{part_b['total_cells_lambda_gt_0']} "
        f"wall={payload['runtime_seconds']}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
