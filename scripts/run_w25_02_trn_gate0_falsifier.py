#!/usr/bin/env python3
"""Wave 25-02 (P0 Gate-0 Falsifier): T-Rn Gated-DeltaNet + KV Retraction Mechanistic & Powered Audit.

Executes every required Gate-0 check from wave25_plus_master_queue_2026-11-26.json on `Qwen/Qwen3.5-2B-Base`:
  1. Full audit of `past_key_values.layers[*]` attributes (`recurrent_states`, `conv_states`, `keys`, `values`).
  2. Exact `bf16` vs `fp32` counterfactual KL noise floor (`n=24` cases).
  3. Mechanistic state-alignment diagnostic on `SELECT` (`n=48`):
       - Per-layer/per-head least-squares optimal scalar `gamma*_{l,h} = <S_full - S_cf, Delta_S> / ||Delta_S||^2`
       - Per-layer/per-head alignment cosine `cos(Delta_S, S_full - S_cf)`
       - Explains why untouched KV-only had lower state L2 than damped subtraction in W24-13b.
  4. Powered `REPORT` dirty-cone falsifier (`n=200` strictly disjoint cases, >=4 templates, all gammas frozen on SELECT):
       - Stratum A (`natural_unrelated`, n=100): diverse multi-sentence natural prose suffixes (no repeated boilerplate)
       - Stratum B (`dependent_dirty_cone`, n=100): natural suffixes that explicitly reference/use the retracted fact!
       - Collateral check on EVERY case: an independent non-retracted entity embedded in the same prefix (collateral KL + acc).
       - Full comparator suite:
           * `stale_no_retraction`
           * `kv_excise_only` (strongest zero-replay baseline)
           * `appended_correction_notice` (strongest conventional prompt-level baseline)
           * `hybrid_select_scalar_gamma_plus_kv` (scalar gamma frozen on SELECT)
           * `hybrid_select_ls_gamma_lh_plus_kv` (per-layer/head gamma*_{l,h} frozen on SELECT)
           * `hybrid_exact_decay_plus_kv` (per-layer/head hooked scalar decay)
           * `exact_checkpoint_replay` (gold oracle + segment-tree cost reference)
  5. Automated Rulebook v3.3 Kill / Advance Verdict:
       KILL if hybrid-minus-KV-only KL 95% CI includes 0, OR improvement < 3x bf16/fp32 noise floor,
       OR appended_correction_notice matches/beats hybrid KL on REPORT.
"""

from __future__ import annotations

import argparse
import copy
import gc
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from chelated_pv7_spine.bridge_merge_harness import paired_bootstrap_ci  # noqa: E402
from chelated_pv7_spine.claim_registry import claim_registry_sha256  # noqa: E402
from run_spark_w24_13_deltanet_state_retraction import (  # noqa: E402
    MODEL_ID,
    _clone_cache,
    _excise_full_attn_kv_span,
    _iter_linear_recurrent_refs,
)
from run_spark_w24_13b_deltanet_long_decay import _attach_decay_hooks  # noqa: E402

# 4 distinct prompt templates (zero boilerplate repetition across cases)
TEMPLATES: Tuple[Dict[str, str], ...] = (
    {
        "name": "ops_ledger",
        "prefix": "Operations ledger entry {cid}: baseline status for {ent} is {old}; independent unit {col_ent} is {col_val}. ",
        "target": "Temporary dispatch for {ent}: status changed to {new}. ",
        "correction": "Correction notice: disregard the temporary dispatch for {ent}; {ent} remains {old}. ",
        "query": "\nQuestion: What is the current status of {ent}? Answer in one word:",
        "col_query": "\nQuestion: What is the current status of {col_ent}? Answer in one word:",
    },
    {
        "name": "maritime_manifest",
        "prefix": "Port manifest {cid}: vessel {ent} is assigned to {old}, while freighter {col_ent} docks at {col_val}. ",
        "target": "Provisional routing bulletin: vessel {ent} diverted to {new}. ",
        "correction": "Official erratum: cancel the provisional routing bulletin for {ent}; assigned port is {old}. ",
        "query": "\nQuestion: Which port is vessel {ent} assigned to? Answer in one word:",
        "col_query": "\nQuestion: Which port does freighter {col_ent} dock at? Answer in one word:",
    },
    {
        "name": "security_policy",
        "prefix": "Access control register {cid}: credential {ent} is {old}, and backup key {col_ent} is {col_val}. ",
        "target": "Interim security override: credential {ent} marked {new}. ",
        "correction": "Policy rescission: revoke the interim override for {ent}; credential {ent} is {old}. ",
        "query": "\nQuestion: What is the active state of credential {ent}? Answer in one word:",
        "col_query": "\nQuestion: What is the active state of backup key {col_ent}? Answer in one word:",
    },
    {
        "name": "lab_inventory",
        "prefix": "Laboratory catalog {cid}: compound {ent} is stored in {old}, and reagent {col_ent} is kept in {col_val}. ",
        "target": "Transfer memo: compound {ent} relocated to {new}. ",
        "correction": "Audit correction: cancel the transfer memo for {ent}; compound {ent} stays in {old}. ",
        "query": "\nQuestion: Where is compound {ent} stored? Answer in one word:",
        "col_query": "\nQuestion: Where is reagent {col_ent} kept? Answer in one word:",
    },
)

VALUE_PAIRS_BY_TEMPLATE: Tuple[Tuple[Tuple[str, str], ...], ...] = (
    (("offline", "online"), ("halted", "active"), ("idle", "running"), ("standby", "engaged"), ("paused", "live"), ("locked", "open")),
    (("Lisbon", "Oslo"), ("Genoa", "Bremen"), ("Bilbao", "Rotterdam"), ("Naples", "Antwerp"), ("Cadiz", "Hamburg"), ("Marseille", "Gdansk")),
    (("denied", "approved"), ("revoked", "valid"), ("suspended", "granted"), ("blocked", "cleared"), ("expired", "active"), ("sealed", "open")),
    (("amber", "cobalt"), ("crimson", "emerald"), ("silver", "violet"), ("copper", "indigo"), ("ivory", "scarlet"), ("bronze", "azure")),
)

# Diverse natural prose paragraphs (distinct real-style topics, NO single repeated filler!)
NATURAL_PARAGRAPHS: Tuple[str, ...] = (
    "During the autumn fiscal review, regional utilities reported steadier wholesale electricity pricing across coastal interconnectors as hydroelectric reservoirs recovered from summer deficits. Engineers audited substation transformer loads and scheduled routine insulator replacements along north-south transmission corridors without interrupting residential customers.",
    "Marine biologists surveying kelp canopies along temperate continental shelves noted stronger juvenile fish recruitment following cooler upwelling currents. Acoustic telemetry buoys logged seasonal migration routes while coastal observatories recalibrated salinity sensors and archived bathymetric profiles for open-access climate repositories.",
    "Municipal transit planners evaluated signal-priority corridors across mid-sized rail networks, finding that coordinated timetable buffers lowered peak-hour headway variance by twelve percent. Maintenance depots simultaneously transitioned wheel-bearing inspections to ultrasonic sensors to reduce unscheduled rolling-stock downtime.",
    "In agricultural hydrology trials, drip-irrigation manifolds paired with soil-moisture tensiometers cut seasonal groundwater pumping while preserving orchard yield grades. Agronomy teams documented root-zone nitrate retention across sandy loam plots and compared cover-crop mulches after late-spring rainfall.",
    "Archivists digitizing nineteenth-century cartographic folios completed multispectral imaging of fragile linen-backed harbor charts. Conservation chemists buffered acidic paper fibers using non-aqueous alkaline treatments and indexed marginal surveyor annotations into searchable bibliographic catalogs.",
    "Atmospheric researchers analyzing stratospheric balloon soundings refined gravity-wave drag parameterizations in medium-range numerical weather models. Cross-validation against radiosonde stations over mountain ranges reduced wind-shear forecast bias at cruising altitudes.",
    "Industrial metallurgists testing low-carbon structural steels measured fracture toughness across sub-zero impact cycles after controlled thermomechanical rolling. Electron microscopy revealed finer ferrite grain boundaries that improved weldability in modular bridge girders.",
    "Botanical seed banks expanded cryogenic storage protocols for alpine meadow flora threatened by shifting snowmelt calendars. Germination viability assays confirmed high recovery rates after desiccation to five percent equilibrium moisture inside hermetically sealed foil pouches.",
)

DIRTY_CONE_TEMPLATES: Tuple[str, ...] = (
    "Follow-up analyst note referencing {ent}: based on the earlier notice listing {ent} as {new}, the dispatch team prepared checklists consistent with {new} operations and logged {ent} under the {new} queue before moving to unrelated regional inventory checks. ",
    "Secondary coordination summary for {ent}: operators observed the update tagging {ent} as {new}, adjusted shift briefings accordingly for {new} handling, and appended a timestamped cross-reference linking {ent} to {new} prior to standard archival review. ",
    "Downstream verification memo on {ent}: automated triage saw {ent} reported as {new} and provisionally routed {ent} into the {new} workflow while independent teams reviewed unrelated quarterly maintenance logs. ",
)


def build_cases(n_select: int = 48, n_report: int = 200) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    rng = np.random.default_rng(2502)
    total = n_select + n_report
    all_cases: List[Dict[str, Any]] = []
    for i in range(total):
        t_idx = i % len(TEMPLATES)
        tpl = TEMPLATES[t_idx]
        pairs = VALUE_PAIRS_BY_TEMPLATE[t_idx]
        old_val, new_val = pairs[(i // len(TEMPLATES)) % len(pairs)]
        col_old, _ = pairs[((i // len(TEMPLATES)) + 2) % len(pairs)]
        if col_old in (old_val, new_val):
            col_old = pairs[((i // len(TEMPLATES)) + 3) % len(pairs)][0]
        ent = f"Unit-{1000 + i * 7}"
        col_ent = f"Node-{5000 + i * 11}"
        regime = "natural_unrelated" if (i % 2 == 0) else "dependent_dirty_cone"

        # Compose 4 distinct natural paragraphs (~240-320 tokens)
        p_idx = rng.choice(len(NATURAL_PARAGRAPHS), size=4, replace=False)
        nat_suffix = " ".join(NATURAL_PARAGRAPHS[int(j)] for j in p_idx)
        if regime == "dependent_dirty_cone":
            dc = DIRTY_CONE_TEMPLATES[i % len(DIRTY_CONE_TEMPLATES)].format(ent=ent, new=new_val)
            suffix = dc + nat_suffix
        else:
            suffix = nat_suffix

        all_cases.append({
            "case_id": i,
            "template": tpl["name"],
            "regime": regime,
            "entity": ent,
            "old_val": old_val,
            "new_val": new_val,
            "col_entity": col_ent,
            "col_val": col_old,
            "prefix": tpl["prefix"].format(cid=f"R{i:04d}", ent=ent, old=old_val, col_ent=col_ent, col_val=col_old),
            "target": tpl["target"].format(ent=ent, new=new_val),
            "correction": tpl["correction"].format(ent=ent, old=old_val),
            "suffix": " " + suffix,
            "query": tpl["query"].format(ent=ent),
            "col_query": tpl["col_query"].format(col_ent=col_ent),
        })
    return all_cases[:n_select], all_cases[n_select:]


def audit_cache_structure(pkv: Any) -> Dict[str, Any]:
    """Full audit of every attribute and dict key inside past_key_values.layers[*]."""
    layers_info: List[Dict[str, Any]] = []
    if hasattr(pkv, "layers"):
        for idx, layer in enumerate(pkv.layers):
            entry: Dict[str, Any] = {"layer_idx": idx, "class": type(layer).__name__, "tensor_attrs": {}}
            for attr in ("recurrent_states", "conv_states", "keys", "values"):
                val = getattr(layer, attr, None)
                if isinstance(val, dict):
                    entry["tensor_attrs"][attr] = {
                        str(k): {"shape": list(v.shape), "dtype": str(v.dtype)} if isinstance(v, torch.Tensor) else type(v).__name__
                        for k, v in val.items()
                    }
                elif isinstance(val, torch.Tensor):
                    entry["tensor_attrs"][attr] = {"shape": list(val.shape), "dtype": str(val.dtype)}
            layers_info.append(entry)
    linear_cnt = sum(1 for x in layers_info if "recurrent_states" in x["tensor_attrs"])
    conv_cnt = sum(1 for x in layers_info if "conv_states" in x["tensor_attrs"])
    kv_cnt = sum(1 for x in layers_info if "keys" in x["tensor_attrs"])
    return {
        "total_layers": len(layers_info),
        "linear_recurrent_layers": linear_cnt,
        "conv_state_layers": conv_cnt,
        "full_attn_kv_layers": kv_cnt,
        "sample_layers": layers_info[:4] + layers_info[-2:],
    }


def _kl_from_logits(p_ref: torch.Tensor, logits_cand: torch.Tensor) -> float:
    return float(F.kl_div(F.log_softmax(logits_cand, dim=-1), p_ref, reduction="sum").item())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "pv7_w25_02_trn_gate0_spark2")
    ap.add_argument("--n-select", type=int, default=48)
    ap.add_argument("--n-report", type=int, default=200)
    ap.add_argument("--n-noise-floor", type=int, default=24)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    select_cases, report_cases = build_cases(args.n_select, args.n_report)
    if args.self_test:
        assert len(select_cases) == args.n_select and len(report_cases) == args.n_report
        assert len({c["entity"] for c in select_cases} & {c["entity"] for c in report_cases}) == 0
        print(json.dumps({"self_test": "PASS", "n_select": len(select_cases), "n_report": len(report_cases)}, indent=2))
        return 0

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for W25-02")

    t0 = time.perf_counter()
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[w25-02] Loading {MODEL_ID} on {gpu_name} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    # --- PART 1: Measure exact bf16 vs fp32 KL noise floor on n_noise_floor counterfactuals ---
    print(f"[w25-02:1] Measuring bf16 vs fp32 KL noise floor (n={args.n_noise_floor}) ...", flush=True)
    mdl_fp32 = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.float32).to("cuda").eval()
    fp32_cf_probs: List[torch.Tensor] = []
    with torch.no_grad():
        for c in select_cases[: args.n_noise_floor]:
            cf_ids = tok(c["prefix"] + c["suffix"] + c["query"], return_tensors="pt").input_ids.to("cuda")
            logits_32 = mdl_fp32(input_ids=cf_ids, use_cache=False).logits[0, -1, :].float()
            fp32_cf_probs.append(F.softmax(logits_32, dim=-1).cpu())
    del mdl_fp32
    gc.collect()
    torch.cuda.empty_cache()

    mdl = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16).to("cuda").eval()
    bf16_vs_fp32_kls: List[float] = []
    bf16_split_vs_concat_kls: List[float] = []
    with torch.no_grad():
        for idx, c in enumerate(select_cases[: args.n_noise_floor]):
            cf_ids = tok(c["prefix"] + c["suffix"] + c["query"], return_tensors="pt").input_ids.to("cuda")
            logits_bf16 = mdl(input_ids=cf_ids, use_cache=False).logits[0, -1, :].float()
            p32 = fp32_cf_probs[idx].to("cuda")
            bf16_vs_fp32_kls.append(_kl_from_logits(p32, logits_bf16))

            # Also compare split-cache counterfactual vs single-pass concat counterfactual in bf16
            pref_ids = tok(c["prefix"], return_tensors="pt").input_ids.to("cuda")
            suf_ids = tok(c["suffix"], return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
            q_ids = tok(c["query"], return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
            pkv_p = mdl(input_ids=pref_ids, use_cache=True).past_key_values
            pkv_s = mdl(input_ids=suf_ids, past_key_values=pkv_p, use_cache=True).past_key_values
            logits_split = mdl(input_ids=q_ids, past_key_values=pkv_s, use_cache=True).logits[0, -1, :].float()
            bf16_split_vs_concat_kls.append(_kl_from_logits(F.softmax(logits_bf16, dim=-1), logits_split))

    noise_floor_bf16_fp32 = float(np.mean(bf16_vs_fp32_kls))
    noise_floor_bf16_fp32_p95 = float(np.percentile(bf16_vs_fp32_kls, 95))
    noise_floor_split_concat = float(np.mean(bf16_split_vs_concat_kls))
    print(
        f"[w25-02:1] Noise floor: bf16_vs_fp32 mean_KL={noise_floor_bf16_fp32:.5f} (p95={noise_floor_bf16_fp32_p95:.5f}) | "
        f"bf16_split_vs_concat mean_KL={noise_floor_split_concat:.5f}",
        flush=True,
    )

    # Audit cache structure once
    probe_out = mdl(input_ids=tok("Cache audit probe.", return_tensors="pt").input_ids.to("cuda"), use_cache=True)
    cache_audit = audit_cache_structure(probe_out.past_key_values)
    del probe_out
    print(f"[w25-02:1] Cache audit: linear={cache_audit['linear_recurrent_layers']} conv={cache_audit['conv_state_layers']} kv={cache_audit['full_attn_kv_layers']}", flush=True)

    captured_log_decay, decay_handles = _attach_decay_hooks(mdl)

    # --- PART 2: SELECT Mechanistic State-Alignment Diagnostic & Frozen Gamma Calibration (n=48) ---
    print(f"[w25-02:2] Running SELECT mechanistic diagnostic (n={len(select_cases)}) ...", flush=True)
    # Accumulate per-(layer_idx, key, head) numerator <S_full - S_cf, Delta_S>, denom ||Delta_S||^2, and cosine
    ls_num: Dict[Tuple[int, Any], List[np.ndarray]] = {}
    ls_den: Dict[Tuple[int, Any], List[np.ndarray]] = {}
    cos_list: Dict[Tuple[int, Any], List[np.ndarray]] = {}
    # Also sweep scalar gamma in {0.0, 0.1, 0.2, 0.25, 0.35, 0.5} on SELECT KL to freeze best scalar gamma
    scalar_grid = (0.0, 0.10, 0.20, 0.25, 0.35, 0.50)
    select_scalar_kl: Dict[float, List[float]] = {g: [] for g in scalar_grid}

    with torch.no_grad():
        for idx, c in enumerate(select_cases):
            pref_ids = tok(c["prefix"], return_tensors="pt").input_ids.to("cuda")
            targ_ids = tok(c["target"], return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
            suf_ids = tok(c["suffix"], return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
            q_ids = tok(c["query"], return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
            t_start = int(pref_ids.shape[1])
            t_end = t_start + int(targ_ids.shape[1])

            captured_log_decay.clear()
            pkv_pref = mdl(input_ids=pref_ids, use_cache=True).past_key_values
            pkv_ax = mdl(input_ids=targ_ids, past_key_values=_clone_cache(pkv_pref), use_cache=True).past_key_values
            pref_map = {(li, k): t for li, _, k, t in _iter_linear_recurrent_refs(pkv_pref)}
            delta_map = {(li, k): (t - pref_map[(li, k)]).detach().clone() for li, _, k, t in _iter_linear_recurrent_refs(pkv_ax)}

            pkv_full = mdl(input_ids=suf_ids, past_key_values=_clone_cache(pkv_ax), use_cache=True).past_key_values
            pkv_cf = mdl(input_ids=suf_ids, past_key_values=_clone_cache(pkv_pref), use_cache=True).past_key_values

            logits_cf = mdl(input_ids=q_ids, past_key_values=_clone_cache(pkv_cf), use_cache=True).logits[0, -1, :].float()
            p_cf = F.softmax(logits_cf, dim=-1)

            cf_map = {(li, k): t for li, _, k, t in _iter_linear_recurrent_refs(pkv_cf)}
            full_map = {(li, k): t for li, _, k, t in _iter_linear_recurrent_refs(pkv_full)}

            for key_lk, d_t in delta_map.items():
                err_t = (full_map[key_lk] - cf_map[key_lk]).float()  # [1, H, K, V]
                d_f = d_t.float()
                # Flatten per head: [H, K*V]
                H = err_t.shape[1]
                e_flat = err_t.reshape(H, -1)
                d_flat = d_f.reshape(H, -1)
                num = (e_flat * d_flat).sum(dim=1).cpu().numpy()
                den = (d_flat * d_flat).sum(dim=1).cpu().numpy() + 1e-12
                enorm = torch.linalg.norm(e_flat, dim=1).cpu().numpy() + 1e-12
                dnorm = torch.linalg.norm(d_flat, dim=1).cpu().numpy() + 1e-12
                ls_num.setdefault(key_lk, []).append(num)
                ls_den.setdefault(key_lk, []).append(den)
                cos_list.setdefault(key_lk, []).append(num / (enorm * dnorm))

            for g in scalar_grid:
                pkv_try = _clone_cache(pkv_full)
                if g > 0.0:
                    for li, rs_dict, k, t_cur in _iter_linear_recurrent_refs(pkv_try):
                        rs_dict[k] = t_cur - g * delta_map[(li, k)]
                _excise_full_attn_kv_span(pkv_try, t_start, t_end)
                l_try = mdl(input_ids=q_ids, past_key_values=pkv_try, use_cache=True).logits[0, -1, :].float()
                select_scalar_kl[g].append(_kl_from_logits(p_cf, l_try))

    # Freeze optimal scalar gamma AND per-layer/per-head least-squares gamma*_{l,h} STRICTLY on SELECT!
    best_scalar_gamma = min(scalar_grid, key=lambda g: float(np.mean(select_scalar_kl[g])))
    frozen_ls_gamma: Dict[Tuple[int, Any], torch.Tensor] = {}
    layer_diag: List[Dict[str, Any]] = []
    all_cos_vals: List[float] = []
    all_gamma_vals: List[float] = []
    for (li, k) in sorted(ls_num.keys(), key=lambda x: (x[0], str(x[1]))):
        mean_num = np.mean(ls_num[(li, k)], axis=0)
        mean_den = np.mean(ls_den[(li, k)], axis=0)
        gamma_h = np.clip(mean_num / np.maximum(mean_den, 1e-12), -0.5, 1.5)
        cos_h = np.mean(cos_list[(li, k)], axis=0)
        frozen_ls_gamma[(li, k)] = torch.tensor(gamma_h, dtype=torch.bfloat16, device="cuda").view(1, -1, 1, 1)
        all_cos_vals.extend(cos_h.tolist())
        all_gamma_vals.extend(gamma_h.tolist())
        layer_diag.append({
            "layer_idx": li,
            "state_key": str(k),
            "mean_gamma_star": round(float(np.mean(gamma_h)), 5),
            "min_gamma_star": round(float(np.min(gamma_h)), 5),
            "max_gamma_star": round(float(np.max(gamma_h)), 5),
            "mean_cos_delta_vs_err": round(float(np.mean(cos_h)), 5),
            "frac_heads_positive_cos": round(float(np.mean(cos_h > 0.0)), 4),
        })
    print(
        f"[w25-02:2] SELECT frozen scalar gamma*={best_scalar_gamma} (mean_KL={np.mean(select_scalar_kl[best_scalar_gamma]):.5f} "
        f"vs kv_only={np.mean(select_scalar_kl[0.0]):.5f}) | global mean gamma*_lh={np.mean(all_gamma_vals):.4f} "
        f"mean cos(Delta, S_full-S_cf)={np.mean(all_cos_vals):.4f}",
        flush=True,
    )

    # --- PART 3: Powered n=200 REPORT Dirty-Cone Falsifier ---
    print(f"[w25-02:3] Running powered REPORT dirty-cone falsifier (n={len(report_cases)}) ...", flush=True)
    arms = (
        "stale_no_retraction",
        "kv_excise_only",
        "appended_correction_notice",
        "hybrid_select_scalar_gamma_plus_kv",
        "hybrid_select_ls_gamma_lh_plus_kv",
        "hybrid_exact_decay_plus_kv",
    )
    per_case_records: List[Dict[str, Any]] = []
    t_rep0 = time.perf_counter()

    with torch.no_grad():
        for idx, c in enumerate(report_cases):
            pref_ids = tok(c["prefix"], return_tensors="pt").input_ids.to("cuda")
            targ_ids = tok(c["target"], return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
            suf_ids = tok(c["suffix"], return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
            corr_ids = tok(" " + c["correction"], return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
            q_ids = tok(c["query"], return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
            col_q_ids = tok(c["col_query"], return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")

            t_start = int(pref_ids.shape[1])
            t_end = t_start + int(targ_ids.shape[1])
            old_tid = tok.encode(" " + c["old_val"], add_special_tokens=False)[0]
            new_tid = tok.encode(" " + c["new_val"], add_special_tokens=False)[0]
            col_tid = tok.encode(" " + c["col_val"], add_special_tokens=False)[0]

            captured_log_decay.clear()
            pkv_pref = mdl(input_ids=pref_ids, use_cache=True).past_key_values
            pkv_ax = mdl(input_ids=targ_ids, past_key_values=_clone_cache(pkv_pref), use_cache=True).past_key_values
            pref_map = {(li, k): t for li, _, k, t in _iter_linear_recurrent_refs(pkv_pref)}
            delta_map = {(li, k): (t - pref_map[(li, k)]).detach().clone() for li, _, k, t in _iter_linear_recurrent_refs(pkv_ax)}

            captured_log_decay.clear()
            pkv_full = mdl(input_ids=suf_ids, past_key_values=_clone_cache(pkv_ax), use_cache=True).past_key_values
            decay_factors = {
                li: torch.exp(torch.stack(chunks, dim=0).sum(dim=0)).clamp(0.0, 1.0)
                for li, chunks in captured_log_decay.items()
            }

            # Exact checkpoint replay (timed)
            torch.cuda.synchronize()
            tr0 = time.perf_counter()
            pkv_cf = mdl(input_ids=suf_ids, past_key_values=_clone_cache(pkv_pref), use_cache=True).past_key_values
            torch.cuda.synchronize()
            replay_ms = (time.perf_counter() - tr0) * 1000.0

            logits_cf = mdl(input_ids=q_ids, past_key_values=_clone_cache(pkv_cf), use_cache=True).logits[0, -1, :].float()
            p_cf = F.softmax(logits_cf, dim=-1)
            m_cf = float((logits_cf[old_tid] - logits_cf[new_tid]).item())

            col_logits_cf = mdl(input_ids=col_q_ids, past_key_values=_clone_cache(pkv_cf), use_cache=True).logits[0, -1, :].float()
            p_col_cf = F.softmax(col_logits_cf, dim=-1)

            cf_map = {(li, k): t for li, _, k, t in _iter_linear_recurrent_refs(pkv_cf)}

            rec: Dict[str, Any] = {
                "case_id": c["case_id"],
                "regime": c["regime"],
                "template": c["template"],
                "suffix_tokens": int(suf_ids.shape[1]),
                "replay_ms": round(replay_ms, 3),
                "cf_top1_old": bool(int(torch.argmax(logits_cf).item()) == old_tid),
                "arms": {},
            }

            # Compute stale first for recovery normalization
            logits_stale = mdl(input_ids=q_ids, past_key_values=_clone_cache(pkv_full), use_cache=True).logits[0, -1, :].float()
            m_stale = float((logits_stale[old_tid] - logits_stale[new_tid]).item())
            gap = m_cf - m_stale

            for arm in arms:
                torch.cuda.synchronize()
                ts0 = time.perf_counter()
                pkv_arm = _clone_cache(pkv_full)
                if arm == "stale_no_retraction":
                    pass
                elif arm == "kv_excise_only":
                    _excise_full_attn_kv_span(pkv_arm, t_start, t_end)
                elif arm == "appended_correction_notice":
                    pkv_arm = mdl(input_ids=corr_ids, past_key_values=pkv_arm, use_cache=True).past_key_values
                elif arm == "hybrid_select_scalar_gamma_plus_kv":
                    if best_scalar_gamma != 0.0:
                        for li, rs_dict, k, t_cur in _iter_linear_recurrent_refs(pkv_arm):
                            rs_dict[k] = t_cur - best_scalar_gamma * delta_map[(li, k)]
                    _excise_full_attn_kv_span(pkv_arm, t_start, t_end)
                elif arm == "hybrid_select_ls_gamma_lh_plus_kv":
                    for li, rs_dict, k, t_cur in _iter_linear_recurrent_refs(pkv_arm):
                        g_t = frozen_ls_gamma[(li, k)]
                        rs_dict[k] = t_cur - g_t * delta_map[(li, k)]
                    _excise_full_attn_kv_span(pkv_arm, t_start, t_end)
                elif arm == "hybrid_exact_decay_plus_kv":
                    for li, rs_dict, k, t_cur in _iter_linear_recurrent_refs(pkv_arm):
                        if li in decay_factors and decay_factors[li].shape[1] == t_cur.shape[1]:
                            dec = decay_factors[li].view(1, -1, 1, 1).to(dtype=t_cur.dtype, device=t_cur.device)
                            rs_dict[k] = t_cur - dec * delta_map[(li, k)]
                        else:
                            rs_dict[k] = t_cur - best_scalar_gamma * delta_map[(li, k)]
                    _excise_full_attn_kv_span(pkv_arm, t_start, t_end)
                torch.cuda.synchronize()
                op_ms = (time.perf_counter() - ts0) * 1000.0

                # State L2 error across all 18 DeltaNet layers
                l2s = []
                for li, _, k, t_cur in _iter_linear_recurrent_refs(pkv_arm):
                    cf_t = cf_map[(li, k)]
                    denom = float(torch.linalg.norm(cf_t.float()).item()) + 1e-8
                    l2s.append(float(torch.linalg.norm((t_cur - cf_t).float()).item()) / denom)

                logits_q = mdl(input_ids=q_ids, past_key_values=_clone_cache(pkv_arm), use_cache=True).logits[0, -1, :].float()
                kl_q = _kl_from_logits(p_cf, logits_q)
                m_q = float((logits_q[old_tid] - logits_q[new_tid]).item())
                recov = ((m_q - m_stale) / gap) if abs(gap) > 1e-4 else 0.0
                flip_ok = bool(m_q > 0.0)

                # Collateral check on independent non-retracted entity
                logits_col = mdl(input_ids=col_q_ids, past_key_values=pkv_arm, use_cache=True).logits[0, -1, :].float()
                kl_col = _kl_from_logits(p_col_cf, logits_col)
                col_ok = bool(int(torch.argmax(logits_col).item()) == col_tid)

                rec["arms"][arm] = {
                    "op_ms": round(op_ms, 3),
                    "state_rel_l2": round(float(np.mean(l2s)), 5),
                    "kl_vs_cf": round(kl_q, 6),
                    "recovery_frac": round(recov, 5),
                    "margin_prefers_old": flip_ok,
                    "collateral_kl": round(kl_col, 6),
                    "collateral_top1_ok": col_ok,
                }
            per_case_records.append(rec)
            if (idx + 1) % 40 == 0 or (idx + 1) == len(report_cases):
                print(f"[w25-02:3] progress {idx+1}/{len(report_cases)} ({time.perf_counter()-t_rep0:.1f}s)", flush=True)

    for h in decay_handles:
        h.remove()

    # Aggregate overall & by regime + exact paired bootstrap CIs
    def summarize_slice(recs: List[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "n": len(recs),
            "mean_suffix_tokens": round(float(np.mean([r["suffix_tokens"] for r in recs])), 1),
            "mean_replay_ms": round(float(np.mean([r["replay_ms"] for r in recs])), 2),
            "arms": {},
            "paired_kl_contrasts": {},
        }
        for arm in arms:
            kls = [r["arms"][arm]["kl_vs_cf"] for r in recs]
            l2s = [r["arms"][arm]["state_rel_l2"] for r in recs]
            recovs = [r["arms"][arm]["recovery_frac"] for r in recs]
            col_kls = [r["arms"][arm]["collateral_kl"] for r in recs]
            ops = [r["arms"][arm]["op_ms"] for r in recs]
            m_kl, lo_kl, hi_kl = paired_bootstrap_ci(kls, n_resamples=4000, seed=25021)
            out["arms"][arm] = {
                "mean_kl": round(m_kl, 6),
                "kl_ci95": [round(lo_kl, 6), round(hi_kl, 6)],
                "median_kl": round(float(np.median(kls)), 6),
                "mean_state_rel_l2": round(float(np.mean(l2s)), 5),
                "mean_recovery_frac": round(float(np.mean(recovs)), 4),
                "margin_prefers_old_rate": round(float(np.mean([r["arms"][arm]["margin_prefers_old"] for r in recs])), 4),
                "mean_collateral_kl": round(float(np.mean(col_kls)), 6),
                "collateral_top1_rate": round(float(np.mean([r["arms"][arm]["collateral_top1_ok"] for r in recs])), 4),
                "mean_op_ms": round(float(np.mean(ops)), 2),
                "speedup_vs_replay": round(float(np.mean([r["replay_ms"] for r in recs])) / max(float(np.mean(ops)), 1e-6), 2),
            }
        # Paired KL differences: negative = first arm has LOWER (better) KL than second arm
        for a, b in (
            ("hybrid_select_scalar_gamma_plus_kv", "kv_excise_only"),
            ("hybrid_select_ls_gamma_lh_plus_kv", "kv_excise_only"),
            ("hybrid_exact_decay_plus_kv", "kv_excise_only"),
            ("hybrid_select_ls_gamma_lh_plus_kv", "appended_correction_notice"),
            ("hybrid_select_scalar_gamma_plus_kv", "appended_correction_notice"),
            ("kv_excise_only", "appended_correction_notice"),
        ):
            diffs = [r["arms"][a]["kl_vs_cf"] - r["arms"][b]["kl_vs_cf"] for r in recs]
            m_d, lo_d, hi_d = paired_bootstrap_ci(diffs, n_resamples=4000, seed=25022)
            out["paired_kl_contrasts"][f"{a}__minus__{b}"] = {
                "mean_diff": round(m_d, 6),
                "ci95": [round(lo_d, 6), round(hi_d, 6)],
                "strictly_lower_kl": bool(hi_d < 0.0),
                "beats_3x_noise_floor": bool((-m_d) >= 3.0 * noise_floor_bf16_fp32),
            }
        return out

    overall = summarize_slice(per_case_records)
    by_regime = {
        reg: summarize_slice([r for r in per_case_records if r["regime"] == reg])
        for reg in ("natural_unrelated", "dependent_dirty_cone")
    }

    # Evaluate formal Gate-0 Kill/Advance criterion on primary arm (hybrid_select_ls_gamma_lh_plus_kv AND scalar)
    c_ls_kv = overall["paired_kl_contrasts"]["hybrid_select_ls_gamma_lh_plus_kv__minus__kv_excise_only"]
    c_sc_kv = overall["paired_kl_contrasts"]["hybrid_select_scalar_gamma_plus_kv__minus__kv_excise_only"]
    c_ls_cor = overall["paired_kl_contrasts"]["hybrid_select_ls_gamma_lh_plus_kv__minus__appended_correction_notice"]
    c_sc_cor = overall["paired_kl_contrasts"]["hybrid_select_scalar_gamma_plus_kv__minus__appended_correction_notice"]

    ls_passes = bool(c_ls_kv["strictly_lower_kl"] and c_ls_kv["beats_3x_noise_floor"] and c_ls_cor["strictly_lower_kl"])
    sc_passes = bool(c_sc_kv["strictly_lower_kl"] and c_sc_kv["beats_3x_noise_floor"] and c_sc_cor["strictly_lower_kl"])
    gate0_verdict = "ADVANCE_TO_W26_01" if (ls_passes or sc_passes) else "KILL_OR_HOLD_W25_02"

    manifest = {
        "experiment_id": "pv7_w25_02_trn_gate0_falsifier",
        "wave_item_id": "W25-02",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hardware_cohort": f"spark2_gb10:{gpu_name}",
        "model_id": MODEL_ID,
        "claim_registry_sha256": claim_registry_sha256(),
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "cache_structure_audit": cache_audit,
        "noise_floor": {
            "n_cases": args.n_noise_floor,
            "bf16_vs_fp32_mean_kl": round(noise_floor_bf16_fp32, 6),
            "bf16_vs_fp32_p95_kl": round(noise_floor_bf16_fp32_p95, 6),
            "bf16_split_vs_concat_mean_kl": round(noise_floor_split_concat, 6),
            "three_x_noise_floor_kl": round(3.0 * noise_floor_bf16_fp32, 6),
        },
        "select_mechanistic_diagnostic": {
            "n_select": len(select_cases),
            "frozen_best_scalar_gamma": best_scalar_gamma,
            "select_scalar_grid_mean_kl": {str(g): round(float(np.mean(v)), 6) for g, v in select_scalar_kl.items()},
            "global_mean_ls_gamma_star": round(float(np.mean(all_gamma_vals)), 5),
            "global_mean_cos_delta_vs_err": round(float(np.mean(all_cos_vals)), 5),
            "per_layer_alignment": layer_diag,
        },
        "report_overall": overall,
        "report_by_regime": by_regime,
        "gate0_verdict": {
            "verdict": gate0_verdict,
            "ls_gamma_lh_passes_all_3": ls_passes,
            "scalar_gamma_passes_all_3": sc_passes,
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "w25_02_trn_gate0_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[w25-02] DONE verdict={gate0_verdict} wall={manifest['elapsed_seconds']}s -> {out_path}", flush=True)
    print(json.dumps({"noise_floor": manifest["noise_floor"], "overall_arms": overall["arms"], "contrasts": overall["paired_kl_contrasts"], "verdict": manifest["gate0_verdict"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
