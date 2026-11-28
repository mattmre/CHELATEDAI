#!/usr/bin/env python3
"""Wave 25-03 (P0 Step-0 Falsifier): QSCCI Powered fp32 Replication, k=1==k=16 Diagnosis & Cross-Encoding Invariance.

Directly executes every mandatory Step-0 check from wave25_plus_master_queue_2026-11-26.json in `fp32` on Desktop RTX 3090:
  1. Exact per-pair logit dump diagnosing W24-12's `k=1 == k=16` quantization/basis anomaly (bf16 vs fp32).
  2. Powered `n=200` family-disjoint REPORT pairs (`n=60` SELECT pairs, 3 seeds, rank & layer band frozen on SELECT)
     across 3 distinct model families:
       - `Qwen/Qwen2.5-1.5B-Instruct` (Qwen2)
       - `HuggingFaceTB/SmolLM2-1.7B-Instruct` (Llama / SmolLM2 — non-Qwen family #1)
       - `microsoft/phi-2` (Phi-2 — non-Qwen family #2)
  3. Strongest conventional baselines on the exact same `n=200` REPORT pairs:
       - `nuisance_balanced_caa` (SELECT has 50/50 balanced nuisance tags so CAA does NOT invert by construction!)
       - `mean_answer_gradient` (rank-1 mean output gradient)
       - `supervised_low_rank_discriminant` (regularized linear discriminant / Ridge direction on SELECT activations)
       - `qscci_causal_svd` (rank-k causal-gradient SVD with k frozen on SELECT)
       - `full_site_ceiling` (full d_model patch at target layers — exact ceiling)
       - `orthogonal_sham`
  4. Cross-Encoding Invariance Falsifier (STRICTLY freezing the intervention learned on Yes/No SELECT):
       - `reversed_labels`: "Answer No if true, Yes if false" — checks whether steering follows TRUTH or the "Yes" token!
       - `randomized_codes`: "Reply Alpha if true, Omega if false" — unseen answer tokens!
       - `freeform_entity`: factual completion margin `log P(true_entity) - log P(false_entity)`!
  5. Automated Rulebook v3.3 Kill / Advance Verdict:
       KILL if `QSCCI - balanced_CAA` LCB95 <= 0, OR `< 50%` of full-site ceiling, OR reversed-labels steering
       follows the original "Yes" token instead of factual truth!
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chelated_pv7_spine.bridge_merge_harness import paired_bootstrap_ci  # noqa: E402
from chelated_pv7_spine.claim_registry import claim_registry_sha256  # noqa: E402

# 260 distinct factual pairs (capital cities, chemical symbols, planets/astronomy, arithmetic/unit conversions,
# world rivers/landmarks, historical dates, biology/anatomy) -> 60 SELECT + 200 REPORT strictly family-disjoint!
def _generate_260_fact_families() -> List[Dict[str, str]]:
    families: List[Dict[str, str]] = []

    # 1. Capitals (45 distinct countries)
    capitals = [
        ("France", "Paris", "Lyon"), ("Japan", "Tokyo", "Osaka"), ("Italy", "Rome", "Milan"),
        ("Spain", "Madrid", "Seville"), ("Germany", "Berlin", "Munich"), ("Canada", "Ottawa", "Toronto"),
        ("Brazil", "Brasilia", "Salvador"), ("Australia", "Canberra", "Sydney"), ("India", "Delhi", "Mumbai"),
        ("Egypt", "Cairo", "Alexandria"), ("Greece", "Athens", "Thessaloniki"), ("Norway", "Oslo", "Bergen"),
        ("Sweden", "Stockholm", "Malmo"), ("Finland", "Helsinki", "Tampere"), ("Poland", "Warsaw", "Krakow"),
        ("Austria", "Vienna", "Graz"), ("Belgium", "Brussels", "Antwerp"), ("Portugal", "Lisbon", "Porto"),
        ("Ireland", "Dublin", "Cork"), ("Hungary", "Budapest", "Debrecen"), ("Czechia", "Prague", "Brno"),
        ("Denmark", "Copenhagen", "Aarhus"), ("Iceland", "Reykjavik", "Akureyri"), ("Turkey", "Ankara", "Istanbul"),
        ("Mexico", "Mexico", "Guadalajara"), ("Argentina", "Buenos", "Cordoba"), ("Chile", "Santiago", "Valparaiso"),
        ("Peru", "Lima", "Cusco"), ("Colombia", "Bogota", "Medellin"), ("Kenya", "Nairobi", "Mombasa"),
        ("Nigeria", "Abuja", "Lagos"), ("Morocco", "Rabat", "Casablanca"), ("Thailand", "Bangkok", "Chiang"),
        ("Vietnam", "Hanoi", "Saigon"), ("Indonesia", "Jakarta", "Bandung"), ("Malaysia", "Kuala", "Penang"),
        ("Philippines", "Manila", "Cebu"), ("Korea", "Seoul", "Busan"), ("Ukraine", "Kyiv", "Lviv"),
        ("Romania", "Bucharest", "Cluj"), ("Bulgaria", "Sofia", "Plovdiv"), ("Croatia", "Zagreb", "Split"),
        ("Serbia", "Belgrade", "Nis"), ("Switzerland", "Bern", "Zurich"), ("Netherlands", "Amsterdam", "Rotterdam"),
    ]
    for c, tr, fa in capitals:
        families.append({
            "family_id": f"cap_{c.lower()}",
            "true_stmt": f"The capital city of {c} is {tr}",
            "false_stmt": f"The capital city of {c} is {fa}",
            "freeform_prompt": f"The capital city of {c} is",
            "true_ans": tr,
            "false_ans": fa,
        })

    # 2. Chemical element symbols (45 distinct elements)
    elements = [
        ("Gold", "Au", "Ag"), ("Silver", "Ag", "Au"), ("Iron", "Fe", "Ir"), ("Copper", "Cu", "Co"),
        ("Sodium", "Na", "So"), ("Potassium", "K", "P"), ("Mercury", "Hg", "Me"), ("Lead", "Pb", "Le"),
        ("Tin", "Sn", "Ti"), ("Tungsten", "W", "Tu"), ("Carbon", "C", "Ca"), ("Oxygen", "O", "Ox"),
        ("Nitrogen", "N", "Ni"), ("Hydrogen", "H", "Hy"), ("Helium", "He", "Hl"), ("Lithium", "Li", "Lt"),
        ("Neon", "Ne", "No"), ("Magnesium", "Mg", "Mn"), ("Aluminum", "Al", "Am"), ("Silicon", "Si", "Sl"),
        ("Phosphorus", "P", "Ph"), ("Sulfur", "S", "Su"), ("Chlorine", "Cl", "Ch"), ("Argon", "Ar", "Ag"),
        ("Calcium", "Ca", "Cl"), ("Titanium", "Ti", "Tn"), ("Chromium", "Cr", "Ch"), ("Manganese", "Mn", "Mg"),
        ("Cobalt", "Co", "Cb"), ("Nickel", "Ni", "Nk"), ("Zinc", "Zn", "Zi"), ("Bromine", "Br", "Bm"),
        ("Krypton", "Kr", "Ky"), ("Iodine", "I", "Io"), ("Xenon", "Xe", "Xn"), ("Barium", "Ba", "Br"),
        ("Platinum", "Pt", "Pl"), ("Uranium", "U", "Ur"), ("Boron", "B", "Bo"), ("Fluorine", "F", "Fl"),
        ("Strontium", "Sr", "St"), ("Cadmium", "Cd", "Cm"), ("Cesium", "Cs", "Ce"), ("Radium", "Ra", "Rd"),
        ("Bismuth", "Bi", "Bs"), ("Cobaltum", "Co", "Cb"),
    ][:45]
    for el, tr, fa in elements:
        families.append({
            "family_id": f"elem_{el.lower()}",
            "true_stmt": f"The chemical symbol for {el} is {tr}",
            "false_stmt": f"The chemical symbol for {el} is {fa}",
            "freeform_prompt": f"The chemical symbol for {el} is",
            "true_ans": tr,
            "false_ans": fa,
        })

    # 3. Exact integer arithmetic sums & products (110 distinct pairs)
    for a in range(11, 66):
        b = (a * 3) % 19 + 4
        tr = str(a + b)
        fa = str(a + b + 7)
        families.append({
            "family_id": f"add_{a}_{b}",
            "true_stmt": f"The sum of {a} and {b} equals {tr}",
            "false_stmt": f"The sum of {a} and {b} equals {fa}",
            "freeform_prompt": f"The sum of {a} and {b} equals",
            "true_ans": tr,
            "false_ans": fa,
        })
    for a in range(4, 29):
        for b in (3, 7):
            tr = str(a * b)
            fa = str(a * b + 5)
            families.append({
                "family_id": f"mul_{a}_{b}",
                "true_stmt": f"The product of {a} times {b} equals {tr}",
                "false_stmt": f"The product of {a} times {b} equals {fa}",
                "freeform_prompt": f"The product of {a} times {b} equals",
                "true_ans": tr,
                "false_ans": fa,
            })

    # 4. Months, Days, Planets, Currencies, Antonyms (65 distinct pairs)
    extras = [
        ("month after January", "February", "April"), ("month after March", "April", "June"),
        ("month after May", "June", "August"), ("month after July", "August", "October"),
        ("month after September", "October", "December"), ("month after November", "December", "February"),
        ("day after Monday", "Tuesday", "Thursday"), ("day after Tuesday", "Wednesday", "Friday"),
        ("day after Wednesday", "Thursday", "Saturday"), ("day after Thursday", "Friday", "Sunday"),
        ("day after Friday", "Saturday", "Monday"), ("day after Saturday", "Sunday", "Tuesday"),
        ("opposite of hot", "cold", "warm"), ("opposite of tall", "short", "wide"),
        ("opposite of fast", "slow", "loud"), ("opposite of light", "dark", "soft"),
        ("opposite of open", "closed", "clean"), ("opposite of dry", "wet", "hard"),
        ("opposite of young", "old", "new"), ("opposite of strong", "weak", "sharp"),
        ("opposite of rich", "poor", "deep"), ("opposite of early", "late", "near"),
        ("opposite of north", "south", "east"), ("opposite of east", "west", "north"),
        ("opposite of up", "down", "left"), ("opposite of left", "right", "high"),
        ("opposite of full", "empty", "heavy"), ("opposite of thick", "thin", "flat"),
        ("opposite of loud", "quiet", "bright"), ("opposite of sweet", "bitter", "fresh"),
        ("currency of Japan", "yen", "euro"), ("currency of Britain", "pound", "peso"),
        ("currency of India", "rupee", "franc"), ("currency of Mexico", "peso", "yen"),
        ("currency of Switzerland", "franc", "pound"), ("currency of Russia", "ruble", "rupee"),
        ("largest planet in the solar system", "Jupiter", "Mars"),
        ("planet closest to the Sun", "Mercury", "Venus"),
        ("red planet in the solar system", "Mars", "Saturn"),
        ("planet famous for bright rings", "Saturn", "Mercury"),
        ("freezing point of water in Celsius", "zero", "fifty"),
        ("boiling point of water in Celsius", "100", "50"),
        ("number of hours in a day", "24", "18"),
        ("number of minutes in an hour", "60", "45"),
        ("number of days in a week", "7", "9"),
        ("number of months in a year", "12", "10"),
        ("number of sides on a triangle", "3", "5"),
        ("number of sides on a square", "4", "6"),
        ("number of sides on a pentagon", "5", "7"),
        ("number of sides on a hexagon", "6", "8"),
        ("number of sides on an octagon", "8", "6"),
        ("continent containing Brazil", "South", "Europe"),
        ("ocean between Europe and America", "Atlantic", "Indian"),
        ("ocean between Asia and America", "Pacific", "Arctic"),
        ("organ that pumps blood", "heart", "liver"),
        ("organ responsible for breathing", "lungs", "stomach"),
        ("star at the center of our solar system", "Sun", "Moon"),
        ("natural satellite orbiting Earth", "Moon", "Mars"),
        ("color of clear daytime sky", "blue", "green"),
        ("color of fresh grass", "green", "purple"),
        ("color of ripe bananas", "yellow", "blue"),
        ("color of fresh snow", "white", "black"),
        ("language primarily spoken in Spain", "Spanish", "German"),
        ("language primarily spoken in France", "French", "Russian"),
        ("language primarily spoken in Germany", "German", "Spanish"),
    ]
    for idx, (desc, tr, fa) in enumerate(extras):
        families.append({
            "family_id": f"gen_{idx:03d}",
            "true_stmt": f"The {desc} is {tr}",
            "false_stmt": f"The {desc} is {fa}",
            "freeform_prompt": f"The {desc} is",
            "true_ans": tr,
            "false_ans": fa,
        })

    # Deduplicate by family_id and return at least 260
    seen = set()
    uniq = []
    for f in families:
        if f["family_id"] not in seen:
            seen.add(f["family_id"])
            uniq.append(f)
    assert len(uniq) >= 260, f"Only generated {len(uniq)} families"
    return uniq[:260]


def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-12 else np.zeros_like(v)


def _get_mlp_out_module(layer: nn.Module) -> nn.Module:
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "down_proj"):
        return layer.mlp.down_proj
    if hasattr(layer, "mlp") and hasattr(layer.mlp, "fc2"):
        return layer.mlp.fc2
    raise AttributeError(f"Cannot locate MLP output projection on {type(layer)}")


def _eval_single_model_fp32(
    model_id: str,
    families: List[Dict[str, str]],
    *,
    n_select: int,
    n_report: int,
    seeds: Sequence[int],
) -> Dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[w25-03] Loading {model_id} in float32 on CUDA ...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32).to("cuda").eval()
    device = next(model.parameters()).device
    layers_mod = model.model.layers
    n_layers = len(layers_mod)
    # Candidate mid-late layer bands (proportionally matched to model depth)
    b_start = max(1, int(round(n_layers * 0.40)))
    b_end = min(n_layers, int(round(n_layers * 0.72)))
    active_layers = list(range(b_start, b_end))

    yes_id = tok.encode(" Yes", add_special_tokens=False)[-1]
    no_id = tok.encode(" No", add_special_tokens=False)[-1]
    alpha_id = tok.encode(" Alpha", add_special_tokens=False)[-1]
    omega_id = tok.encode(" Omega", add_special_tokens=False)[-1]

    def _tok(text: str) -> Dict[str, torch.Tensor]:
        return {k: v.to(device) for k, v in tok(text, return_tensors="pt").items()}

    # --- PART 1: Diagnose W24-12 k=1==k=16 anomaly in fp32 on 8 families ---
    diag_fams = families[:8]
    diag_sel_prompts = [
        f"[TAG-ALPHA] Statement: {r['true_stmt']}. Is this statement factually accurate? Answer Yes or No:"
        for r in diag_fams
    ] + [
        f"[TAG-BETA] Statement: {r['false_stmt']}. Is this statement factually accurate? Answer Yes or No:"
        for r in diag_fams
    ]
    diag_rep_fams = families[8:16]

    def _collect_grads_and_acts(prompts: List[str], pos_flags: List[bool], layers_use: List[int]) -> Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray]]:
        g_map: Dict[int, List[np.ndarray]] = {l: [] for l in layers_use}
        p_map: Dict[int, List[np.ndarray]] = {l: [] for l in layers_use}
        n_map: Dict[int, List[np.ndarray]] = {l: [] for l in layers_use}
        for p_txt, is_pos in zip(prompts, pos_flags):
            inp = _tok(p_txt)
            holders: Dict[int, torch.Tensor] = {}
            handles = []
            def _mk(lidx: int):
                def _h(_m, _i, out):
                    out.retain_grad()
                    holders[lidx] = out
                    return out
                return _h
            for l in layers_use:
                handles.append(_get_mlp_out_module(layers_mod[l]).register_forward_hook(_mk(l)))
            try:
                model.zero_grad(set_to_none=True)
                with torch.enable_grad():
                    logits = model(**inp).logits[0, -1, :]
                    margin = logits[yes_id] - logits[no_id]
                    margin.backward()
                for l in layers_use:
                    h = holders[l]
                    g_map[l].append(h.grad[0, -1, :].detach().cpu().numpy())
                    act_v = h[0, -1, :].detach().cpu().numpy()
                    if is_pos:
                        p_map[l].append(act_v)
                    else:
                        n_map[l].append(act_v)
            finally:
                for hd in handles:
                    hd.remove()
        return (
            {l: np.vstack(g_map[l]) for l in layers_use},
            {l: np.vstack(p_map[l]) for l in layers_use},
            {l: np.vstack(n_map[l]) for l in layers_use},
        )

    def _run_patch_eval(
        pairs: List[Tuple[str, str, int, int]],
        target_layers: Sequence[int],
        bases_map: Dict[int, np.ndarray] | None,
        rank_k: int,
    ) -> Dict[str, Any]:
        """If bases_map is None, patches the FULL d_model vector (full-site ceiling)."""
        clean_ms: List[float] = []
        corrupt_ms: List[float] = []
        patched_ms: List[float] = []
        basis_t = None
        if bases_map is not None:
            basis_t = {
                l: torch.tensor(bases_map[l][: min(rank_k, bases_map[l].shape[0])], dtype=torch.float32, device=device)
                for l in target_layers
            }
        for p_c, p_x, pos_tid, neg_tid in pairs:
            in_c = _tok(p_c)
            in_x = _tok(p_x)
            cached: Dict[int, torch.Tensor] = {}

            def _mk_cap(lidx: int):
                def _cap(_m, _i, out):
                    h_last = out[0, -1, :]
                    if basis_t is None:
                        cached[lidx] = h_last.detach().clone()
                    else:
                        cached[lidx] = (basis_t[lidx] @ h_last).detach().clone()
                    return out
                return _cap

            hs = [_get_mlp_out_module(layers_mod[l]).register_forward_hook(_mk_cap(l)) for l in target_layers]
            try:
                with torch.no_grad():
                    lc = model(**in_c).logits[0, -1, :]
                    clean_ms.append(float((lc[pos_tid] - lc[neg_tid]).item()))
            finally:
                for hd in hs:
                    hd.remove()

            with torch.no_grad():
                lx = model(**in_x).logits[0, -1, :]
                corrupt_ms.append(float((lx[pos_tid] - lx[neg_tid]).item()))

            def _mk_pat(lidx: int):
                c_clean = cached[lidx]
                def _pat(_m, _i, out):
                    h_last = out[0, -1, :]
                    out2 = out.clone()
                    if basis_t is None:
                        out2[0, -1, :] = c_clean
                    else:
                        u = basis_t[lidx]
                        c_cur = u @ h_last
                        out2[0, -1, :] = h_last + (c_clean - c_cur) @ u
                    return out2
                return _pat

            hs2 = [_get_mlp_out_module(layers_mod[l]).register_forward_hook(_mk_pat(l)) for l in target_layers]
            try:
                with torch.no_grad():
                    lp = model(**in_x).logits[0, -1, :]
                    patched_ms.append(float((lp[pos_tid] - lp[neg_tid]).item()))
            finally:
                for hd in hs2:
                    hd.remove()

        gaps = [c - x for c, x in zip(clean_ms, corrupt_ms)]
        recovs = [p - x for p, x in zip(patched_ms, corrupt_ms)]
        mean_gap = float(np.mean(gaps))
        m_r, lo_r, hi_r = paired_bootstrap_ci(recovs, n_resamples=3000, seed=25031)
        return {
            "mean_gap": round(mean_gap, 6),
            "mean_delta": round(m_r, 6),
            "ci95": [round(lo_r, 6), round(hi_r, 6)],
            "restoration_ratio": round(m_r / mean_gap if abs(mean_gap) > 1e-6 else 0.0, 5),
            "per_pair_recov": recovs,
            "per_pair_clean": [round(v, 5) for v in clean_ms[:8]],
            "per_pair_corrupt": [round(v, 5) for v in corrupt_ms[:8]],
            "per_pair_patched": [round(v, 5) for v in patched_ms[:8]],
        }

    # Run k=1..16 fp32 diagnostic dump on the 8-pair slice
    g_diag, _, _ = _collect_grads_and_acts(diag_sel_prompts, [True] * 8 + [False] * 8, active_layers)
    diag_bases = {}
    diag_svals = {}
    for l in active_layers:
        g_mat = g_diag[l]
        mean_g = _unit(g_mat.mean(axis=0))
        _, sv, vt = np.linalg.svd(g_mat, full_matrices=False)
        diag_svals[str(l)] = [round(float(x), 5) for x in sv[:8]]
        cols = np.vstack([mean_g[None, :], vt[: min(15, vt.shape[0])]])
        q_mat, _ = np.linalg.qr(cols.T)
        diag_bases[l] = q_mat[:, : min(16, q_mat.shape[1])].T
    diag_pairs = [
        (
            f"[TAG-BETA] Statement: {r['true_stmt']}. Is this statement factually accurate? Answer Yes or No:",
            f"[TAG-ALPHA] Statement: {r['false_stmt']}. Is this statement factually accurate? Answer Yes or No:",
            yes_id,
            no_id,
        )
        for r in diag_rep_fams
    ]
    k_diag_dump = {
        f"k{k}": _run_patch_eval(diag_pairs, active_layers, diag_bases, k)
        for k in (1, 4, 8, 16)
    }
    for k_lbl, val in k_diag_dump.items():
        val.pop("per_pair_recov", None)

    # --- PART 2 & 3: Powered n=200 fp32 Multi-Seed Replication + Balanced CAA + Cross-Encoding Invariance ---
    seed_summaries: List[Dict[str, Any]] = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        perm = rng.permutation(len(families))
        sel_fams = [families[int(i)] for i in perm[:n_select]]
        rep_fams = [families[int(i)] for i in perm[n_select : n_select + n_report]]

        # NUISANCE-BALANCED SELECT: half TAG-ALPHA true / TAG-BETA false, half TAG-BETA true / TAG-ALPHA false!
        # That ensures CAA does NOT invert the nuisance tag by construction!
        sel_prompts: List[str] = []
        sel_pos_flags: List[bool] = []
        for idx_s, r in enumerate(sel_fams):
            tag_t = "[TAG-ALPHA]" if (idx_s % 2 == 0) else "[TAG-BETA]"
            tag_f = "[TAG-BETA]" if (idx_s % 2 == 0) else "[TAG-ALPHA]"
            sel_prompts.append(f"{tag_t} Statement: {r['true_stmt']}. Is this statement factually accurate? Answer Yes or No:")
            sel_pos_flags.append(True)
            sel_prompts.append(f"{tag_f} Statement: {r['false_stmt']}. Is this statement factually accurate? Answer Yes or No:")
            sel_pos_flags.append(False)

        g_sel, p_sel, n_sel = _collect_grads_and_acts(sel_prompts, sel_pos_flags, active_layers)

        qscci_bases: Dict[int, np.ndarray] = {}
        mean_grad_bases: Dict[int, np.ndarray] = {}
        bal_caa_bases: Dict[int, np.ndarray] = {}
        disc_bases: Dict[int, np.ndarray] = {}
        sham_bases: Dict[int, np.ndarray] = {}

        for l in active_layers:
            g_mat = g_sel[l]
            mean_g = _unit(g_mat.mean(axis=0))
            mean_grad_bases[l] = mean_g[None, :]
            _, _, vt = np.linalg.svd(g_mat, full_matrices=False)
            cols = np.vstack([mean_g[None, :], vt[: min(15, vt.shape[0])]])
            q_mat, _ = np.linalg.qr(cols.T)
            qscci_bases[l] = q_mat[:, : min(16, q_mat.shape[1])].T

            # Nuisance-balanced CAA
            caa_v = _unit(p_sel[l].mean(axis=0) - n_sel[l].mean(axis=0))
            bal_caa_bases[l] = caa_v[None, :]

            # Supervised regularized linear discriminant (Ridge on centered activations)
            X_act = np.vstack([p_sel[l], n_sel[l]])
            y_act = np.concatenate([np.ones(p_sel[l].shape[0]), -np.ones(n_sel[l].shape[0])])
            X_c = X_act - X_act.mean(axis=0, keepdims=True)
            # Dual ridge: w = X_c^T (X_c X_c^T + lam I)^-1 y
            gram = X_c @ X_c.T + 10.0 * np.eye(X_c.shape[0])
            w_ridge = _unit(X_c.T @ np.linalg.solve(gram, y_act))
            disc_bases[l] = w_ridge[None, :]

            d_m = g_mat.shape[1]
            rnd = rng.normal(size=(d_m, 16))
            rnd = rnd - qscci_bases[l].T @ (qscci_bases[l] @ rnd)
            q_sh, _ = np.linalg.qr(rnd)
            sham_bases[l] = q_sh[:, : min(16, q_sh.shape[1])].T

        # Freeze best rank k in {1, 4, 8, 16} STRICTLY on a 24-pair SELECT validation slice!
        val_pairs = [
            (
                f"[TAG-ALPHA] Statement: {r['true_stmt']}. Is this statement factually accurate? Answer Yes or No:",
                f"[TAG-ALPHA] Statement: {r['false_stmt']}. Is this statement factually accurate? Answer Yes or No:",
                yes_id,
                no_id,
            )
            for r in sel_fams[:24]
        ]
        select_rank_ratios = {
            k: _run_patch_eval(val_pairs, active_layers, qscci_bases, k)["restoration_ratio"]
            for k in (1, 4, 8, 16)
        }
        frozen_k = max((1, 4, 8, 16), key=lambda k: select_rank_ratios[k])

        # Build strictly disjoint n=200 REPORT pairs (same tag on both clean & corrupt so only truth differs!)
        rep_yesno_pairs = [
            (
                f"[TAG-ALPHA] Statement: {r['true_stmt']}. Is this statement factually accurate? Answer Yes or No:",
                f"[TAG-ALPHA] Statement: {r['false_stmt']}. Is this statement factually accurate? Answer Yes or No:",
                yes_id,
                no_id,
            )
            for r in rep_fams
        ]

        res_qscci = _run_patch_eval(rep_yesno_pairs, active_layers, qscci_bases, frozen_k)
        res_mgrad = _run_patch_eval(rep_yesno_pairs, active_layers, mean_grad_bases, 1)
        res_caa = _run_patch_eval(rep_yesno_pairs, active_layers, bal_caa_bases, 1)
        res_disc = _run_patch_eval(rep_yesno_pairs, active_layers, disc_bases, 1)
        res_ceil = _run_patch_eval(rep_yesno_pairs, active_layers, None, 0)
        res_sham = _run_patch_eval(rep_yesno_pairs, active_layers, sham_bases, frozen_k)

        diff_qscci_caa = [a - b for a, b in zip(res_qscci["per_pair_recov"], res_caa["per_pair_recov"])]
        m_qc, lo_qc, hi_qc = paired_bootstrap_ci(diff_qscci_caa, n_resamples=4000, seed=seed + 10)
        diff_qscci_disc = [a - b for a, b in zip(res_qscci["per_pair_recov"], res_disc["per_pair_recov"])]
        m_qd, lo_qd, hi_qd = paired_bootstrap_ci(diff_qscci_disc, n_resamples=4000, seed=seed + 11)

        # --- PART 3: Cross-Encoding Invariance on FIRST seed (n=100 REPORT pairs, FROZEN intervention!) ---
        cross_enc: Dict[str, Any] = {}
        if seed == seeds[0]:
            ce_fams = rep_fams[:100]
            # 3a. Reversed labels: True -> "No", False -> "Yes"!
            # Note: pos_tid=no_id, neg_tid=yes_id! If intervention follows TRUTH, ratio > 0;
            # if it blindly pushes "Yes", ratio < 0!
            rev_pairs = [
                (
                    f"[TAG-ALPHA] Statement: {r['true_stmt']}. Answer No if factually accurate, or Yes if inaccurate:",
                    f"[TAG-ALPHA] Statement: {r['false_stmt']}. Answer No if factually accurate, or Yes if inaccurate:",
                    no_id,
                    yes_id,
                )
                for r in ce_fams
            ]
            # 3b. Randomized in-context codes: True -> "Alpha", False -> "Omega"
            code_pairs = [
                (
                    f"[TAG-ALPHA] Statement: {r['true_stmt']}. Reply Alpha if accurate, or Omega if inaccurate:",
                    f"[TAG-ALPHA] Statement: {r['false_stmt']}. Reply Alpha if accurate, or Omega if inaccurate:",
                    alpha_id,
                    omega_id,
                )
                for r in ce_fams
            ]
            # 3c. Free-form entity completion: patch from true-statement prefix? Or contrast true vs false question!
            ff_pairs = []
            for r in ce_fams:
                t_tid = tok.encode(" " + r["true_ans"], add_special_tokens=False)[0]
                f_tid = tok.encode(" " + r["false_ans"], add_special_tokens=False)[0]
                if t_tid != f_tid:
                    ff_pairs.append((
                        f"Fact check: {r['true_stmt']}. Therefore, {r['freeform_prompt']}",
                        f"Fact check: {r['false_stmt']}. Therefore, {r['freeform_prompt']}",
                        t_tid,
                        f_tid,
                    ))

            for cname, cpairs in (("reversed_labels_truth_polarity", rev_pairs), ("unseen_codes_alpha_omega", code_pairs), ("freeform_entity_completion", ff_pairs)):
                rq = _run_patch_eval(cpairs, active_layers, qscci_bases, frozen_k)
                rc = _run_patch_eval(cpairs, active_layers, bal_caa_bases, 1)
                rceil = _run_patch_eval(cpairs, active_layers, None, 0)
                for d_obj in (rq, rc, rceil):
                    d_obj.pop("per_pair_recov", None)
                cross_enc[cname] = {"n": len(cpairs), "qscci": rq, "balanced_caa": rc, "full_site_ceiling": rceil}

        for d_obj in (res_qscci, res_mgrad, res_caa, res_disc, res_ceil, res_sham):
            d_obj.pop("per_pair_recov", None)

        ceil_ratio = res_ceil["restoration_ratio"]
        frac_of_ceiling = (res_qscci["restoration_ratio"] / ceil_ratio) if abs(ceil_ratio) > 1e-6 else 0.0
        seed_summaries.append({
            "seed": seed,
            "frozen_k": frozen_k,
            "select_rank_ratios": select_rank_ratios,
            "qscci": res_qscci,
            "mean_answer_gradient": res_mgrad,
            "nuisance_balanced_caa": res_caa,
            "supervised_ridge_discriminant": res_disc,
            "full_site_ceiling": res_ceil,
            "orthogonal_sham": res_sham,
            "qscci_frac_of_ceiling": round(frac_of_ceiling, 4),
            "qscci_minus_balanced_caa": {"mean": round(m_qc, 6), "ci95": [round(lo_qc, 6), round(hi_qc, 6)], "strictly_positive_lcb": bool(lo_qc > 0.0)},
            "qscci_minus_ridge_disc": {"mean": round(m_qd, 6), "ci95": [round(lo_qd, 6), round(hi_qd, 6)], "strictly_positive_lcb": bool(lo_qd > 0.0)},
            "cross_encoding_invariance": cross_enc,
        })
        print(
            f"[w25-03] {model_id} seed={seed} k*={frozen_k} | QSCCI={res_qscci['restoration_ratio']:+.3f} "
            f"balCAA={res_caa['restoration_ratio']:+.3f} Ridge={res_disc['restoration_ratio']:+.3f} "
            f"Ceil={ceil_ratio:+.3f} (QSCCI-balCAA LCB={lo_qc:+.4f})",
            flush=True,
        )

    del model
    gc.collect()
    torch.cuda.empty_cache()

    # Model-level Step-0 Kill/Advance verdict
    s0 = seed_summaries[0]
    rev_ratio = s0["cross_encoding_invariance"]["reversed_labels_truth_polarity"]["qscci"]["restoration_ratio"]
    all_seeds_beat_caa = all(s["qscci_minus_balanced_caa"]["strictly_positive_lcb"] for s in seed_summaries)
    mean_frac_ceil = float(np.mean([s["qscci_frac_of_ceiling"] for s in seed_summaries]))
    follows_truth_on_reversal = bool(rev_ratio > 0.0)
    passes_step0 = bool(all_seeds_beat_caa and mean_frac_ceil >= 0.50 and follows_truth_on_reversal)

    return {
        "model_id": model_id,
        "dtype": "float32",
        "n_layers_total": n_layers,
        "active_layers": active_layers,
        "k_anomaly_fp32_diag_dump": {"singular_values_top8": diag_svals, "by_rank": k_diag_dump},
        "seeds": seed_summaries,
        "step0_verdict": {
            "passes_step0": passes_step0,
            "all_seeds_beat_balanced_caa_lcb95": all_seeds_beat_caa,
            "mean_qscci_frac_of_ceiling": round(mean_frac_ceil, 4),
            "clears_50pct_ceiling": bool(mean_frac_ceil >= 0.50),
            "reversed_labels_qscci_ratio": rev_ratio,
            "follows_truth_not_yes_token": follows_truth_on_reversal,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "pv7_w25_03_qscci_step0_fp32")
    ap.add_argument("--n-select", type=int, default=60)
    ap.add_argument("--n-report", type=int, default=200)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    families = _generate_260_fact_families()
    if args.self_test:
        assert len(families) == 260
        print(json.dumps({"self_test": "PASS", "n_families": len(families)}, indent=2))
        return 0

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for W25-03")

    t0 = time.perf_counter()
    gpu_name = torch.cuda.get_device_name(0)
    models = (
        "Qwen/Qwen2.5-1.5B-Instruct",
        "HuggingFaceTB/SmolLM2-1.7B-Instruct",
        "microsoft/phi-2",
    )
    seeds = (25031, 25032, 25033)
    model_results: List[Dict[str, Any]] = []
    for mid in models:
        try:
            res = _eval_single_model_fp32(mid, families, n_select=args.n_select, n_report=args.n_report, seeds=seeds)
            model_results.append(res)
        except Exception as exc:
            print(f"[w25-03] ERROR on {mid}: {exc}", flush=True)
            model_results.append({"model_id": mid, "error": str(exc)})

    n_pass = sum(1 for r in model_results if r.get("step0_verdict", {}).get("passes_step0"))
    overall_verdict = "ADVANCE_QSCCI" if n_pass >= 2 else "KILL_QSCCI_STEP0"
    manifest = {
        "experiment_id": "pv7_w25_03_qscci_step0_fp32",
        "wave_item_id": "W25-03",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hardware_cohort": f"desktop_rtx3090:{gpu_name}",
        "claim_registry_sha256": claim_registry_sha256(),
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "n_select": args.n_select,
        "n_report": args.n_report,
        "seeds": list(seeds),
        "models": model_results,
        "overall_step0_verdict": overall_verdict,
        "models_passing_step0": n_pass,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "w25_03_qscci_step0_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[w25-03] DONE overall_verdict={overall_verdict} ({n_pass}/{len(models)}) wall={manifest['elapsed_seconds']}s -> {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
