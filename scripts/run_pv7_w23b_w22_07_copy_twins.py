"""W22-07 runner: blinded hidden-copy twins, graded leakage gate, priced provenance-check policies.

Usage:
    python scripts/run_pv7_w23b_w22_07_copy_twins.py [--out-dir artifacts/pv7_w23b_w22_07] [--skip-legacy-20]

Preregistration: artifacts/pv7_w23b_w22_07/PREREGISTRATION.md.  CPU only, no network
(the #38 generator gets a local fallback OSV seed cache under <out>/scratch/).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import w22_07_copy_twins as W  # noqa: E402

SNAPSHOT_DIR = _REPO_ROOT / "artifacts" / "pv7_w22e_b55_wild_live" / "frozen_snapshots"
DEFAULT_OUT = _REPO_ROOT / "artifacts" / "pv7_w23b_w22_07"
BUDGETS = (1, 2)
GRID_C = (0.02, 0.10, 0.25)
GRID_WA = (0.20, 0.35, 0.50)
GRID_WFR = (0.5, 1.0)
LEGACY_20_PREFIXES = ("hist_m", "echo_", "late_echo_", "copy_", "syn_d_", "syn_u_", "ca_", "cb_", "v1_mir_", "laundered_")
LEGACY_20_COPY_EDGES = ("COPY_OF", "DERIVES_CURRENT")
LEGACY_20_PREREG_PREFIXES = ("hist_m", "copy_", "copy_orig_", "syn_d_", "syn_u_")


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, text=True).strip()
    except Exception as exc:  # recorded, not swallowed silently
        return f"unavailable:{type(exc).__name__}"


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


def price_independent_rows(test_members, seed, tau, budgets):
    """A0, A1, A2, A3 (budget-free) and A4 per budget.  Decisions do not depend on prices."""
    rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    ek5 = W.run_ek5(test_members)
    ek5_pre = W.run_ek5(test_members, W.PRE_CUT)
    pre_rows = {"A2": [], "A3": []}
    for member in test_members:
        rows["A0"].append(_row(member, seed, W.policy_no_check(member)))
        rows["A1"].append(_row(member, seed, W.policy_cosine(member, tau)))
        rows["A2"].append(_row(member, seed, ek5[member.family.family_id]))
        rows["A3"].append(_row(member, seed, W.policy_ledger(member, 0)))
        for budget in budgets:
            rows[f"A4@B{budget}"].append(_row(member, seed, W.policy_ledger(member, budget)))
        pre_truth = W.oracle_truth(member, W.PRE_CUT)
        pre_rows["A2"].append(int(ek5_pre[member.family.family_id].decision == pre_truth))
        pre_rows["A3"].append(int(W.policy_ledger(member, 0, W.PRE_CUT).decision == pre_truth))
    return rows, pre_rows


def priced_rows(test_members, seed, model, budget, prices):
    rows = {"A5": [], "A5a": [], "A6": []}
    for member in test_members:
        rows["A5"].append(_row(member, seed, W.policy_bayes_earliest(member, model, budget, prices, False)))
        rows["A5a"].append(_row(member, seed, W.policy_bayes_earliest(member, model, budget, prices, True)))
        rows["A6"].append(_row(member, seed, W.policy_td(member, model, budget, prices)))
    return rows


def arm_rows_for(free_rows, priced, budget):
    out = {arm: free_rows[arm] for arm in ("A0", "A1", "A2", "A3")}
    out["A4"] = free_rows[f"A4@B{budget}"]
    out.update(priced)
    return out


# ---------------------------------------------------------------------------------------------------------------
# Legacy leakage audit
# ---------------------------------------------------------------------------------------------------------------
def legacy_55() -> Dict[str, Any]:
    texts, labels, groups = [], [], []
    for path in sorted(SNAPSHOT_DIR.glob("stratum*.json")):
        for family in json.loads(path.read_text(encoding="utf-8")):
            for record in family["records"]:
                if record["op_kind"] not in ("ASSERT", "CONFLICT") or not record.get("text"):
                    continue
                texts.append(record["text"])
                labels.append(1 if record.get("echo_of") else 0)
                groups.append(family["family_id"])
    out = W.grouped_text_classifier(texts, labels, groups)
    out["label"] = "echo_of non-empty (mirror/wire/digest records)"
    return out


def legacy_20() -> Dict[str, Any]:
    from scripts.run_overnight_r5_spark6724 import build_zero_shortcut_v3_flooded_cases

    texts, prereg_labels, wide_labels, groups = [], [], [], []
    for case in build_zero_shortcut_v3_flooded_cases(n_per_family=30):
        linked = {a.src_id for a in case["attestations"] if a.edge_type in LEGACY_20_COPY_EDGES}
        for event_id, event in case["events"].items():
            if not event.text:
                continue
            texts.append(event.text)
            prereg_labels.append(1 if event_id.startswith(LEGACY_20_PREREG_PREFIXES) else 0)
            wide_labels.append(1 if (event_id in linked or event_id.startswith(LEGACY_20_PREFIXES)) else 0)
            groups.append(case["case_id"])
    prereg = W.grouped_text_classifier(texts, prereg_labels, groups)
    prereg["label"] = f"preregistered: event id prefix in {list(LEGACY_20_PREREG_PREFIXES)}"
    wide = W.grouped_text_classifier(texts, wide_labels, groups)
    wide["label"] = f"widened: source of {list(LEGACY_20_COPY_EDGES)} attestation or id prefix in {list(LEGACY_20_PREFIXES)}"
    return {"balanced_accuracy": prereg["balanced_accuracy"], "preregistered_label": prereg, "widened_label": wide}


def legacy_38(scratch: Path, source_cache: Optional[Path]) -> Dict[str, Any]:
    import prospective_realworld_drift_corpus as P

    cache = scratch / "legacy38_osv_seed_cache.json"
    if source_cache is not None:
        # Byte copy of a previously cached OSV seed file (no network); provenance recorded below.
        cache.write_bytes(source_cache.read_bytes())
        cache_kind = f"copied from {source_cache}"
    else:
        cache_kind = "generator offline fallback values (no OSV data)"
        # Same values the generator itself falls back to when OSV is unreachable; written so no request is made.
        fallback = {
            cve: {"cve_id": cve, "osv_summary": f"Security advisory record for {cve}", "osv_package": "upstream-package", "osv_fixed": None}
            for cve in P.REAL_OSV_CVE_IDS
        }
        cache.write_text(json.dumps(fallback, indent=2, sort_keys=True), encoding="utf-8")
    cases, docs, corpus_sha = P.build_prospective_realworld_drift_suite(cache)
    case_ids = sorted((c["case_id"] for c in cases), key=len, reverse=True)
    texts, labels, groups = [], [], []
    for doc in docs:
        owner = next(cid for cid in case_ids if doc.doc_id.startswith(cid + "_"))
        texts.append(doc.text)
        labels.append(1 if "_echo" in doc.doc_id else 0)
        groups.append(owner)
    out = W.grouped_text_classifier(texts, labels, groups)
    out["label"] = "'_echo' in doc_id"
    out["corpus_sha256"] = corpus_sha
    out["osv_cache"] = {"kind": cache_kind, "sha256": hashlib.sha256(cache.read_bytes()).hexdigest()}
    out["corpus_sha256_matches_w22_05_gate"] = corpus_sha == "cb6bc0082b5850f4edfea668d1ecb2a73fe6f898b7476e34ed11036cb378765b"
    return out


# ---------------------------------------------------------------------------------------------------------------
def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--skip-legacy-20", action="store_true", help="skip the #20 audit (imports sentence_transformers; ~70 s)")
    parser.add_argument("--legacy38-osv-cache", default=None, help="existing OSV seed cache JSON to copy (read-only) for the #38 audit")
    parser.add_argument("--seeds", default=",".join(str(s) for s in W.FIXTURE_SEEDS))
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    scratch = out_dir / "scratch"
    scratch.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in args.seeds.split(",") if s]
    t0 = time.time()

    roots = W.load_real_roots(SNAPSHOT_DIR)
    snapshot_sha = hashlib.sha256(
        b"".join(p.read_bytes() for p in sorted(SNAPSHOT_DIR.glob("stratum*.json")))
    ).hexdigest()

    leakage: Dict[str, Dict[str, Any]] = {}
    role_violations = 0
    oracle_mismatches = 0
    per_lambda: Dict[str, Any] = {}
    primary_rows: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    per_seed_primary: Dict[str, Any] = {}
    grid_rows: Dict[tuple, Dict[str, List[Dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    calibration: Dict[str, Any] = {}
    pre_cut: Dict[str, List[int]] = defaultdict(list)
    ek5_ledger_disagree = 0
    ek5_ledger_total = 0

    for lam in W.LEAK_GRADES:
        lam_key = str(lam)
        lam_rows_b1: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        leakage[lam_key] = {}
        for seed in seeds:
            pairs = W.build_twin_pairs(roots, seed, lam)
            members = [m for p in pairs for m in p.members]
            oracle_mismatches += sum(W.oracle_truth(m) != m.truth for m in members)
            if lam == 0.0:
                role_violations += sum(len(W.role_word_violations(m)) for m in members)
                (scratch / f"fixture_seed{seed}_lambda0.json").write_text(
                    json.dumps([m.family.to_dict() | {"twin_label": m.label, "pair_id": m.pair_id} for m in members], indent=1),
                    encoding="utf-8",
                )
            leakage[lam_key][str(seed)] = {
                "family_level": W.family_level_leakage(pairs),
                "witness_level": W.witness_level_leakage(pairs),
                "leaky_pairs": sum(p.leaky for p in pairs),
                "pairs": len(pairs),
            }
            dev = [m for p in pairs if p.split == "dev" for m in p.members]
            test = [m for p in pairs if p.split == "test" for m in p.members]
            tau, tau_scores = W.tune_cosine_tau(dev, W.PRIMARY_PRICES)
            model = W.CopyPosterior.fit(dev)
            calibration[f"{lam_key}:{seed}"] = {
                "tau": tau,
                "tau_dev_harm_pi_0.5": tau_scores,
                "prior": {str(m): {"".join(map(str, c)): p for c, p in v.items()} for m, v in model.prior.items()},
                "bin_likelihood": {("independent" if k == 1 else "copy"): v for k, v in model.bin_lik.items()},
                "dev_pairs": len(dev) // 2,
                "test_pairs": len(test) // 2,
            }
            free, pre = price_independent_rows(test, seed, tau, BUDGETS)
            if lam == 0.0:
                for arm, hits in pre.items():
                    pre_cut[arm].extend(hits)
                for a2, a3 in zip(free["A2"], free["A3"]):
                    ek5_ledger_total += 1
                    ek5_ledger_disagree += int(a2["decision"] != a3["decision"])
            for budget in BUDGETS:
                for c in GRID_C:
                    for wa in GRID_WA:
                        for wfr in GRID_WFR:
                            prices = {"w_fs": 1.0, "w_fr": wfr, "w_a": wa, "c": c}
                            rows = arm_rows_for(free, priced_rows(test, seed, model, budget, prices), budget)
                            for arm, arm_rows in rows.items():
                                grid_rows[(lam, budget, c, wa, wfr)][arm].extend(arm_rows)
                            is_primary = budget == 1 and prices == W.PRIMARY_PRICES
                            if is_primary:
                                for arm, arm_rows in rows.items():
                                    lam_rows_b1[arm].extend(arm_rows)
                                if lam == 0.0:
                                    for arm, arm_rows in rows.items():
                                        primary_rows[arm].extend(arm_rows)
                                    per_seed_primary[str(seed)] = {
                                        arm: W.summarize(arm_rows, W.PRIMARY_PRICES) for arm, arm_rows in rows.items()
                                    }
        per_lambda[lam_key] = {arm: W.summarize(rows, W.PRIMARY_PRICES) for arm, rows in lam_rows_b1.items()}

    # ---- fixture gate
    gate_values = []
    for seed_key, block in leakage["0.0"].items():
        gate_values.append(block["family_level"]["balanced_accuracy"])
        gate_values.append(block["witness_level"]["balanced_accuracy"])
    fixture_valid = max(gate_values) <= W.LEAKAGE_THRESHOLD
    positive_control = [
        max(b["family_level"]["balanced_accuracy"], b["witness_level"]["balanced_accuracy"]) for b in leakage["1.0"].values()
    ]
    leakage_test_powered = min(positive_control) > W.LEAKAGE_THRESHOLD

    # ---- primary decision
    primary = {arm: W.summarize(rows, W.PRIMARY_PRICES) for arm, rows in primary_rows.items()}
    decision = W.kill_rule_decision(primary, fixture_valid)
    bootstrap = W.cluster_bootstrap_delta({a: primary_rows[a] for a in (W.TD_ARM,) + W.CONVENTIONAL_ARMS}, W.PRIMARY_PRICES)

    # ---- sensitivity grid
    grid = []
    for (lam, budget, c, wa, wfr), rows_by_arm in sorted(grid_rows.items()):
        summaries = {arm: W.summarize(rows, {"w_fs": 1.0, "w_fr": wfr, "w_a": wa, "c": c}) for arm, rows in rows_by_arm.items()}
        cell_decision = W.kill_rule_decision(summaries, True)
        grid.append(
            {
                "lambda": lam,
                "budget": budget,
                "c": c,
                "w_a": wa,
                "w_fr": wfr,
                "kill_condition_met": cell_decision["verdict"] == "KILL_T_D",
                "matching_conventional_arms": cell_decision["matching_conventional_arms"],
                "harm_pi_0.5": {arm: s["harm"]["0.5"] for arm, s in summaries.items()},
                "harm_pi_0.05": {arm: s["harm"]["0.05"] for arm, s in summaries.items()},
                "coverage_pi_0.5": {arm: s["coverage"]["0.5"] for arm, s in summaries.items()},
                "mean_checks": {arm: s["mean_checks"] for arm, s in summaries.items()},
            }
        )

    # ---- legacy audit
    legacy: Dict[str, Any] = {"#55": legacy_55(), "#38": legacy_38(scratch, Path(args.legacy38_osv_cache) if args.legacy38_osv_cache else None)}
    if args.skip_legacy_20:
        legacy["#20"] = {"skipped": "--skip-legacy-20"}
    else:
        legacy["#20"] = legacy_20()
    legacy["#35"] = {"skipped": "phase2_d3_multiagent_poisoning has manifests/aggregates only; no per-record copy fixture on disk (D1)"}
    legacy["#46"] = {"skipped": "wave14_d11_pm6_takedown_carryforward has manifests/aggregates only; no per-record copy fixture on disk (D1)"}

    verdict = decision["verdict"]
    if fixture_valid and not leakage_test_powered:
        verdict += "_LEAKAGE_TEST_UNPOWERED"

    results = {
        "schema_version": "w22_07_copy_twins_v1",
        "queue_item": "W22-07",
        "git_sha_at_run": _git_sha(),
        "snapshot_sha256": snapshot_sha,
        "seeds": seeds,
        "n_roots": len(roots),
        "primary_prices": W.PRIMARY_PRICES,
        "prevalences": list(W.PREVALENCES),
        "verdict": verdict,
        "fixture_gate": {
            "threshold": W.LEAKAGE_THRESHOLD,
            "max_lambda0_balanced_accuracy": max(gate_values),
            "fixture_valid": fixture_valid,
            "positive_control_lambda1_min": min(positive_control),
            "leakage_test_powered": leakage_test_powered,
            "role_word_violations_lambda0": role_violations,
            "oracle_truth_mismatches_all_lambdas": oracle_mismatches,
            "real_roots_containing_role_words": {
                word: sum(word in r.text.lower() for r in roots) for word in W.ROLE_WORDS
            },
        },
        "leakage": leakage,
        "primary_lambda0_budget1": primary,
        "per_seed_primary": per_seed_primary,
        "kill_rule": decision,
        "bootstrap_td_minus_best_conv_harm_pi_0.5": bootstrap,
        "graded_leakage_budget1_primary_prices": per_lambda,
        "sensitivity_grid": grid,
        "calibration": calibration,
        "secondary": {
            "ek5_vs_ledger_post_cut_disagreements_lambda0": {"disagree": ek5_ledger_disagree, "total": ek5_ledger_total},
            "pre_withdrawal_cut_accuracy_lambda0": {arm: sum(h) / len(h) for arm, h in pre_cut.items()},
        },
        "legacy_leakage_audit": legacy,
        "runtime_seconds": round(time.time() - t0, 2),
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "w22_07_results.json").write_text(json.dumps(results, indent=2, sort_keys=True), encoding="utf-8")
    (scratch / "primary_rows_lambda0_b1.json").write_text(json.dumps(primary_rows, indent=1), encoding="utf-8")
    print(json.dumps({"verdict": verdict, "fixture_valid": fixture_valid, "powered": leakage_test_powered,
                      "max_lambda0_acc": max(gate_values), "runtime_seconds": results["runtime_seconds"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
