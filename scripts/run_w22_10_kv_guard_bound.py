"""W22-10 (#60 half) runner: receiver-activation bound vs selective top-layer re-prefill.

Preregistration: ``artifacts/pv7_w23b_w22_10/PREREGISTRATION.md``. CPU only, no network.

Usage: ``python scripts/run_w22_10_kv_guard_bound.py [--quick] [--out DIR] [--results FILE]``
Writes ``artifacts/pv7_w23b_w22_10/w22_10_results.json`` (evidence) and bulk data under
``artifacts/pv7_w23b_w22_10/scratch/`` (gitignored).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

os.environ["CUDA_VISIBLE_DEVICES"] = ""
_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import numpy as np  # noqa: E402

import kv_guard_scope_bound as kg  # noqa: E402
from chelated_ssd_kv_repo_cache import (  # noqa: E402
    LOGIT_TOL,
    CacheStats,
    SSDKVCache,
    StaleWeightRevisionError,
    _prompt_chain,
    build_model,
    build_repo,
    contract_for,
    select_repo_names,
    serve_chelated,
)
from ek5_v3_stratified import StratifiedEvent, StratifiedEvidenceStore  # noqa: E402

OUT_DIR = _ROOT / "artifacts" / "pv7_w23b_w22_10"
CASE_SEED = 6001
MODEL_SEEDS = (6001, 6002, 6003)
PRIMARY_SEED = 6001
EPS_GRID = (1e-8, 1e-6, 1e-4, 1e-3, 1e-2, 1e-1)
PERT_SEEDS = (1, 2, 3, 4, 5)
CROSS_SEEDS = (7001, 7002, 7003, 7004, 7005)
ADAPTER_SEED_R2 = 77
TAU = LOGIT_TOL
REALISTIC = ("R3", "R4_eps1e-04", "R4_eps1e-03", "R4_eps1e-02", "R4_eps1e-01", "R5", "R6")
CASES_PER_REPO = 5


def select_cases(repo_count: int = 20) -> List[Tuple[str, int, list]]:
    out = []
    for name in select_repo_names(CASE_SEED, repo_count):
        repo = build_repo(CASE_SEED, name)
        commits = sorted(range(1, 50), key=lambda c: hashlib.sha256(f"{CASE_SEED}|{name}|{c}".encode()).hexdigest())
        for c in sorted(commits[:CASES_PER_REPO]):
            out.append((name, c, repo.commits[c].files))
    return out


def instances(sender) -> List[Tuple[str, str, Any]]:
    """(class_label, instance_id, receiver_model)."""
    out: List[Tuple[str, str, Any]] = [
        ("R0", "identity", sender),
        ("R1", "patched_base", sender.patched_base()),
        ("R2", f"adapter{ADAPTER_SEED_R2}", sender.with_adapter(ADAPTER_SEED_R2)),
    ]
    out += [("R3", f"l0adapter{s}", kg.layer0_adapter(sender, s)) for s in PERT_SEEDS]
    for eps in EPS_GRID:
        out += [(f"R4_eps{eps:.0e}", f"eps{eps:.0e}_s{s}", kg.global_perturbation(sender, eps, s)) for s in PERT_SEEDS]
    out += [("R5", f"model{s}", kg.cross_model(s)) for s in CROSS_SEEDS]
    out.append(("R6", "fp16", kg.fp16_round(sender)))
    return out


def prepare_sender(seed: int, cases, scratch: Path) -> Tuple[Any, SSDKVCache, List[Dict[str, Any]], int]:
    sender = build_model(seed)
    store = StratifiedEvidenceStore()
    cache = SSDKVCache(scratch / f"kv_model{seed}", contract_for(sender), store)
    registered: set = set()
    prepared = []
    bit_mismatch = 0
    for name, c, files in cases:
        prefix_blocks, suffix = files[:-1], files[-1]
        new = [StratifiedEvent(f.version_id, authority_tier=1) for f in files if f.version_id not in registered]
        if new:
            store.apply_mutation(new)
            registered.update(e.event_id for e in new)
        serve_chelated(sender, cache, prefix_blocks, CacheStats())
        chain = _prompt_chain(prefix_blocks)
        stale = None
        for key, _start, _end, ptoks in chain:
            stale = kg.cat_kv(stale, kg.read_block_unchecked(cache._path(key), ptoks))
        run = kg.prefill_blocks(sender, [b.tokens for b in prefix_blocks])
        for layer in range(len(stale)):
            for j in range(2):
                if not np.array_equal(stale[layer][j], run.kv[layer][j]):
                    bit_mismatch += 1
        prepared.append(
            {
                "repo": name, "commit": c, "prefix_tokens": chain[-1][3], "blocks": [b.tokens for b in prefix_blocks],
                "first_key": chain[0][0], "first_ptoks": chain[0][3], "suffix": suffix.tokens, "stale": stale,
                "stats": run.stats, "h1": run.h_layer_inputs[1],
            }
        )
    return sender, cache, prepared, bit_mismatch


def run_instance(sender, cache, prepared, receiver) -> Tuple[List[Dict[str, Any]], float, int]:
    t0 = time.perf_counter()
    rev = kg.revision_norms(sender, receiver)
    rev_seconds = time.perf_counter() - t0
    rcontract = contract_for(receiver)
    rows = []
    n_cases = len(prepared)
    for p in prepared:
        try:
            cache.get(p["first_key"], p["first_ptoks"], rcontract)
            a0_refused = False
        except StaleWeightRevisionError:
            a0_refused = True
        P = int(len(p["prefix_tokens"]))
        S = int(len(p["suffix"]))
        t = time.perf_counter()
        true_run = kg.prefill_blocks(receiver, p["blocks"])
        t_full = time.perf_counter() - t
        true_logits, _, _ = kg.forward_traced(receiver, p["suffix"], past=true_run.kv)
        obs_logits, _, obs_tr = kg.forward_traced(receiver, p["suffix"], past=p["stale"])
        t = time.perf_counter()
        B, inter, case_flops = kg.receiver_activation_bound(rev, p["prefix_tokens"], p["stats"], p["stale"], obs_tr)
        t_bound = time.perf_counter() - t
        cone = kg.dependency_cone_unchanged(sender, receiver, p["prefix_tokens"])
        t = time.perf_counter()
        kv_h = kg.tlr_h_kv(receiver, p["stale"], p["h1"])
        t_tlrh = time.perf_counter() - t
        tlrh_logits, _, _ = kg.forward_traced(receiver, p["suffix"], past=kv_h)
        tlrr_logits, _, _ = kg.forward_traced(receiver, p["suffix"], past=kg.tlr_r_kv(receiver, p["stale"], p["prefix_tokens"]))
        err_stale = kg.max_abs(obs_logits, true_logits)
        accept = B <= TAU
        margin = kg.top2_margin(obs_logits)
        cert = margin > 2.0 * B
        same_arg = np.argmax(obs_logits, axis=1) == np.argmax(true_logits, axis=1)
        f_full = kg.flops_full(P)
        guard_amort = rev.flops / n_cases + case_flops
        guard_unamort = rev.flops + case_flops
        rows.append(
            {
                "repo": p["repo"], "commit": p["commit"], "P": P, "S": S,
                "a0_refused": a0_refused, "cone_accept": bool(cone), "bound": B, "bound_accept": bool(accept),
                "err_stale": err_stale, "err_a1": err_stale if cone else 0.0, "err_a2": err_stale if accept else 0.0,
                "err_tlrh": kg.max_abs(tlrh_logits, true_logits), "err_tlrr": kg.max_abs(tlrr_logits, true_logits),
                "sound": bool(kg.bound_is_sound(err_stale, B)),
                "argmax_certified": int(cert.sum()), "argmax_cert_violations": int((cert & ~same_arg).sum()),
                "argmax_stale_mismatch": int((~same_arg).sum()),
                "cost_full": f_full, "cost_tlrh": kg.flops_tlr_h(P), "cost_tlrr": kg.flops_tlr_r(P),
                "cost_a1": 0 if cone else f_full,
                "cost_a2": guard_amort + (0 if accept else f_full),
                "cost_a2_unamortized": guard_unamort + (0 if accept else f_full),
                "t_full": t_full, "t_bound": t_bound, "t_tlrh": t_tlrh, **{f"b_{k}": v for k, v in inter.items()},
            }
        )
    return rows, rev_seconds, rev.flops


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    def arr(k):
        return np.array([r[k] for r in rows], dtype=float)

    stale = arr("err_stale")
    bnd = arr("bound")
    pos = stale > 1e-15
    ratio = bnd[pos] / stale[pos]
    r_bound = float(np.mean([r["bound_accept"] for r in rows]))
    r_cone = float(np.mean([r["cone_accept"] for r in rows]))
    cost_a2 = float(arr("cost_a2").mean())
    cost_tlrh = float(arr("cost_tlrh").mean())
    max_a2 = float(arr("err_a2").max())
    max_tlrh = float(arr("err_tlrh").max())
    nonvac = r_bound >= 0.5 and r_bound > r_cone
    beats = cost_a2 < cost_tlrh and max_a2 <= max_tlrh
    return {
        "n_rows": len(rows),
        "a0_refusal_rate": float(np.mean([r["a0_refused"] for r in rows])),
        "R_cone": r_cone,
        "R_bound": r_bound,
        "soundness_violations": int(sum(not r["sound"] for r in rows)),
        "bound_median": float(np.median(bnd)), "bound_min": float(bnd.min()), "bound_max": float(bnd.max()),
        "err_stale_median": float(np.median(stale)), "err_stale_max": float(stale.max()),
        "err_a1_max": float(arr("err_a1").max()), "err_a2_max": max_a2,
        "err_tlrh_median": float(np.median(arr("err_tlrh"))), "err_tlrh_max": max_tlrh,
        "err_tlrr_median": float(np.median(arr("err_tlrr"))), "err_tlrr_max": float(arr("err_tlrr").max()),
        "tlrh_within_tau_rate": float(np.mean(arr("err_tlrh") <= TAU)),
        "stale_within_tau_rate": float(np.mean(stale <= TAU)),
        "tightness_rows": int(pos.sum()),
        "tightness_median": float(np.median(ratio)) if ratio.size else None,
        "tightness_p90": float(np.percentile(ratio, 90)) if ratio.size else None,
        "tightness_min": float(ratio.min()) if ratio.size else None,
        "mean_cost_full": float(arr("cost_full").mean()), "mean_cost_tlrh": cost_tlrh,
        "mean_cost_tlrr": float(arr("cost_tlrr").mean()), "mean_cost_a1": float(arr("cost_a1").mean()),
        "mean_cost_a2": cost_a2, "mean_cost_a2_unamortized": float(arr("cost_a2_unamortized").mean()),
        "argmax_positions": int(arr("S").sum()), "argmax_certified": int(arr("argmax_certified").sum()),
        "argmax_cert_violations": int(arr("argmax_cert_violations").sum()),
        "argmax_stale_mismatch": int(arr("argmax_stale_mismatch").sum()),
        # Argmax-level oracle ceiling (added Tier B round 1, F1): rows where stale-KV reuse preserves every
        # suffix argmax, i.e. the most a perfectly tight argmax-level guard could accept.
        "argmax_oracle_rows": int(sum(r["argmax_stale_mismatch"] == 0 for r in rows)),
        "argmax_oracle_row_rate": float(np.mean([r["argmax_stale_mismatch"] == 0 for r in rows])),
        "argmax_oracle_position_rate": float(1.0 - arr("argmax_stale_mismatch").sum() / arr("S").sum()),
        "argmax_bound_certified_position_rate": float(arr("argmax_certified").sum() / arr("S").sum()),
        # _soft trivial-cap hit rates (Tier B round 1, F2): 1.0 means that derivation step was vacuous on every row.
        "soft_cap_rate": {k: float(arr(f"b_{k}").mean()) for k in ("cap0", "cap_s0", "cap_s1")},
        "soft_cap_any_rate": float(np.mean([max(r["b_cap0"], r["b_cap_s0"], r["b_cap_s1"]) > 0 for r in rows])),
        "soft_cap_all_rate": float(np.mean([min(r["b_cap0"], r["b_cap_s0"], r["b_cap_s1"]) > 0 for r in rows])),
        "median_wall_s": {
            "full_prefill": float(np.median(arr("t_full"))), "bound_case_part": float(np.median(arr("t_bound"))),
            "tlrh_kv": float(np.median(arr("t_tlrh"))),
        },
        "nonvacuous": bool(nonvac),
        "beats_tlrh": bool(beats),
        "nonvacuous_and_beats": bool(nonvac and beats),
    }


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="2 repos, model seed 6001 only (smoke)")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="bulk/scratch directory for rows.jsonl and .kvblk files (default: "
        "artifacts/pv7_w23b_w22_10/scratch/{quick,full}); pass a fresh dir to keep earlier runs intact",
    )
    ap.add_argument(
        "--results",
        type=Path,
        default=None,
        help="results JSON path (default: <OUT_DIR>/w22_10_results.json; quick: <scratch>/w22_10_results_quick.json)",
    )
    return ap.parse_args(argv)


def scratch_dir(args: argparse.Namespace) -> Path:
    if args.out is not None:
        return Path(args.out)
    return OUT_DIR / "scratch" / ("quick" if args.quick else "full")


def results_path(args: argparse.Namespace, scratch: Path) -> Path:
    if args.results is not None:
        return Path(args.results)
    return (scratch / "w22_10_results_quick.json") if args.quick else (OUT_DIR / "w22_10_results.json")


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    scratch = scratch_dir(args)
    scratch.mkdir(parents=True, exist_ok=True)
    cases = select_cases(2 if args.quick else 20)
    seeds = (PRIMARY_SEED,) if args.quick else MODEL_SEEDS
    started = time.time()
    per_seed: Dict[str, Any] = {}
    rows_path = scratch / "rows.jsonl"
    with rows_path.open("w", encoding="utf-8") as fh:
        for seed in seeds:
            sender, cache, prepared, mism = prepare_sender(seed, cases, scratch)
            by_class: Dict[str, List[Dict[str, Any]]] = {}
            rev_info: List[Dict[str, Any]] = []
            for label, inst_id, receiver in instances(sender):
                rows, rev_s, rev_flops = run_instance(sender, cache, prepared, receiver)
                for r in rows:
                    r.update({"model_seed": seed, "class": label, "instance": inst_id})
                    fh.write(json.dumps(r) + "\n")
                by_class.setdefault(label, []).extend(rows)
                rev_info.append({"class": label, "instance": inst_id, "revision_seconds": rev_s, "revision_flops": rev_flops})
            per_seed[str(seed)] = {
                "stale_kv_bit_mismatches_vs_instrumented_prefill": mism,
                "kvblk_bytes_on_disk": cache.bytes_on_disk,
                "kvblk_files": len(cache.written_paths),
                "classes": {k: aggregate(v) for k, v in by_class.items()},
                "revision_cost": rev_info,
            }
            print(f"seed {seed} done ({time.time() - started:.1f}s)", flush=True)

    total_viol = sum(c["soundness_violations"] for s in per_seed.values() for c in s["classes"].values())
    k_sound = total_viol == 0
    decision: Dict[str, Any] = {"K_SOUND_holds": k_sound, "total_soundness_violations": total_viol, "per_class": {}}
    advance = False
    for cls in REALISTIC:
        prim = per_seed[str(PRIMARY_SEED)]["classes"][cls]["nonvacuous_and_beats"]
        n_seeds = sum(per_seed[str(s)]["classes"][cls]["nonvacuous_and_beats"] for s in seeds)
        ok = bool(k_sound and prim and n_seeds >= (2 if len(seeds) == 3 else 1))
        decision["per_class"][cls] = {"primary_seed_pass": prim, "seeds_passing": n_seeds, "class_advances": ok}
        advance = advance or ok
    decision["T_K_verdict"] = "ADVANCE" if advance else "STAYS_KILLED"
    out = {
        "item": "W22-10 (#60 KV-guard scope bound half only; N3 half gated off by W22-07)",
        "preregistration": "artifacts/pv7_w23b_w22_10/PREREGISTRATION.md",
        "quick": args.quick,
        "tau": TAU,
        "case_seed": CASE_SEED, "model_seeds": list(seeds), "n_cases": len(cases),
        "case_prefix_tokens": {"min": int(min(len(np.concatenate([b.tokens for b in f[:-1]])) for _, _, f in cases)),
                               "max": int(max(len(np.concatenate([b.tokens for b in f[:-1]])) for _, _, f in cases))},
        "realistic_classes": list(REALISTIC),
        "decision": decision,
        "per_seed": per_seed,
        "environment": {"python": platform.python_version(), "numpy": np.__version__, "platform": platform.platform(),
                        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")},
        "wall_seconds": time.time() - started,
        "scratch_rows": str(rows_path),
    }
    target = results_path(args, scratch)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(out, indent=1, sort_keys=True), encoding="utf-8")
    print(json.dumps(decision, indent=1))
    print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
