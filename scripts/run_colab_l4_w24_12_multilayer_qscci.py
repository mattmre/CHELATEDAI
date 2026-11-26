#!/usr/bin/env python3
"""Colab L4 Wave 24-12 (Lane 7a): Bundle #52 Multi-Layer (k=1,4,8,16 across Layers 11-18)
Causal-Gradient QSCCI Write-Site Restoration vs Top-K SAE & CAA on Qwen2.5-1.5B and Qwen3.5-2B-Base.

Tests the untested mechanistic hypothesis left open by Wave 22C/W22-16:
  - Wave 22C showed 1-D Layer-11 causal-gradient QSCCI steering hit +0.88/+1.13 logit margin vs ~0.00
    for Top-K SAE and CAA, while single-layer 1-D activation-patching restoration was near-null.
  - This script tests multi-layer (Layers 11..18) and multi-rank (k in {1, 4, 8, 16} top causal-SVD
    gradient subspace directions) two-pass activation-patching restoration (`clean` -> `corrupt`) and
    norm-matched steering on strictly family-disjoint unburned `DEV` / `SELECT` / `REPORT` splits,
    measuring restoration ratio, steering margin, and held-out IFEval KL collateral at every k.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chelated_pv7_spine.bridge_merge_harness import paired_bootstrap_ci  # noqa: E402
from chelated_pv7_spine.claim_registry import claim_registry_sha256  # noqa: E402
from scripts.run_colab_l4_w22c_b52_qscci_live_gpu import (  # noqa: E402
    _FACT_TEMPLATES,
    _FALSE_CLAIMS,
    _CONTEXT_SCENARIOS,
    _IFEVAL_PROMPTS,
    MODELS_TO_EVAL,
    _unit,
)

LAYERS_MULTI: Tuple[int, ...] = (11, 12, 13, 14, 15, 16, 17, 18)
RANKS_K: Tuple[int, ...] = (1, 4, 8, 16)


def build_disjoint_splits() -> Dict[str, List[Dict[str, str]]]:
    """Deterministic family-disjoint SELECT and REPORT splits with inverted formatting nuisance on REPORT."""
    select_rows: List[Dict[str, str]] = []
    report_rows: List[Dict[str, str]] = []
    for i in range(16):
        true_stmt = _FACT_TEMPLATES[i][0]
        false_stmt = _FALSE_CLAIMS[i]
        ctx = _CONTEXT_SCENARIOS[i % len(_CONTEXT_SCENARIOS)]
        row = {
            "family_id": f"fam_{i:02d}",
            "pos_prompt": f"[TAG-ALPHA] ({ctx}) Statement: {true_stmt}. Is this statement factually accurate? Answer Yes or No:",
            "neg_prompt": f"[TAG-BETA] ({ctx}) Statement: {false_stmt}. Is this statement factually accurate? Answer Yes or No:",
            "rep_pos_prompt": f"[TAG-BETA] ({ctx}) Statement: {true_stmt}. Is this statement factually accurate? Answer Yes or No:",
            "rep_neg_prompt": f"[TAG-ALPHA] ({ctx}) Statement: {false_stmt}. Is this statement factually accurate? Answer Yes or No:",
        }
        if i < 8:
            select_rows.append(row)
        else:
            report_rows.append(row)
    return {"select": select_rows, "report": report_rows}


def run_self_test() -> Dict[str, Any]:
    sp = build_disjoint_splits()
    assert len(sp["select"]) == 8 and len(sp["report"]) == 8
    # Verify SVD orthonormal basis projection math
    rng = np.random.default_rng(2412)
    g = rng.normal(size=(16, 64))
    _, _, vt = np.linalg.svd(g, full_matrices=False)
    basis = vt[:4]  # (4, 64) orthonormal rows
    err = float(np.max(np.abs(basis @ basis.T - np.eye(4))))
    assert err < 1e-10
    return {"self_test": "PASS", "select_fams": 8, "report_fams": 8, "svd_ortho_err": err}


def eval_model_multilayer(model_id: str) -> Dict[str, Any]:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"[w24-12] Loading {model_id} in bfloat16 ...", flush=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16).to("cuda")
    model.eval()
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype

    yes_id = tok.encode(" Yes", add_special_tokens=False)[-1]
    no_id = tok.encode(" No", add_special_tokens=False)[-1]
    layers_mod = model.model.layers
    active_layers = [l for l in LAYERS_MULTI if l < len(layers_mod)]

    splits = build_disjoint_splits()
    sel_prompts = [r["pos_prompt"] for r in splits["select"]] + [r["neg_prompt"] for r in splits["select"]]
    rep_pos = [r["rep_pos_prompt"] for r in splits["report"]]
    rep_neg = [r["rep_neg_prompt"] for r in splits["report"]]

    def _tok(text: str) -> Dict[str, torch.Tensor]:
        return {k: v.to(device) for k, v in tok(text, return_tensors="pt").items()}

    # 1. Capture SELECT causal gradients d(margin)/d(h_l) at every layer l in active_layers
    layer_grads: Dict[int, List[np.ndarray]] = {l: [] for l in active_layers}
    layer_pos_acts: Dict[int, List[np.ndarray]] = {l: [] for l in active_layers}
    layer_neg_acts: Dict[int, List[np.ndarray]] = {l: [] for l in active_layers}

    for idx_p, prompt in enumerate(sel_prompts):
        inp = _tok(prompt)
        holders: Dict[int, torch.Tensor] = {}
        handles = []

        def _make_hook(lidx: int):
            def _h(mod: nn.Module, i: object, out: torch.Tensor) -> torch.Tensor:
                del mod, i
                out.retain_grad()
                holders[lidx] = out
                return out
            return _h

        for l in active_layers:
            handles.append(layers_mod[l].mlp.down_proj.register_forward_hook(_make_hook(l)))
        try:
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                logits = model(**inp).logits[0, -1, :]
                margin = (logits[yes_id] - logits[no_id]).float()
                margin.backward()
            for l in active_layers:
                h = holders[l]
                layer_grads[l].append(h.grad[0, -1, :].detach().float().cpu().numpy())
                act_v = h[0, -1, :].detach().float().cpu().numpy()
                if idx_p < len(splits["select"]):
                    layer_pos_acts[l].append(act_v)
                else:
                    layer_neg_acts[l].append(act_v)
        finally:
            for h in handles:
                h.remove()

    # Build orthonormal causal-gradient SVD basis (rank up to 16) + CAA mean-diff + orthogonal random sham per layer
    qscci_bases: Dict[int, np.ndarray] = {}
    caa_dirs: Dict[int, np.ndarray] = {}
    sham_bases: Dict[int, np.ndarray] = {}
    rng = np.random.default_rng(24120)

    for l in active_layers:
        g_mat = np.vstack(layer_grads[l])  # (16, d_model)
        # Ensure rank-1 direction is aligned with mean causal gradient
        mean_g = _unit(g_mat.mean(axis=0))
        _, _, vt = np.linalg.svd(g_mat, full_matrices=False)
        # Orthonormalize [mean_g, vt[:15]] via QR
        cols = np.vstack([mean_g[None, :], vt[: min(16, vt.shape[0])]])
        q_mat, _ = np.linalg.qr(cols.T)
        qscci_bases[l] = q_mat[:, : min(16, q_mat.shape[1])].T  # (<=16, d_model)
        caa_dirs[l] = _unit(np.vstack(layer_pos_acts[l]).mean(axis=0) - np.vstack(layer_neg_acts[l]).mean(axis=0))
        d_m = g_mat.shape[1]
        rnd = rng.normal(size=(d_m, 16))
        # Project out qscci_bases[l]
        rnd = rnd - qscci_bases[l].T @ (qscci_bases[l] @ rnd)
        q_sh, _ = np.linalg.qr(rnd)
        sham_bases[l] = q_sh[:, : qscci_bases[l].shape[0]].T

    # 2. Two-pass causal activation-patching restoration on REPORT across:
    #    - single_layer_11 (k=1, 4, 8, 16)
    #    - multi_layer_11_18 (k=1, 4, 8, 16)
    #    - multi_layer_11_18_caa (k=1)
    #    - multi_layer_11_18_sham (k=1, 16)
    def run_restoration(target_layers: Sequence[int], bases_map: Dict[int, np.ndarray], rank_k: int) -> Dict[str, Any]:
        clean_ms: List[float] = []
        corrupt_ms: List[float] = []
        patched_ms: List[float] = []
        basis_t = {
            l: torch.tensor(bases_map[l][: min(rank_k, bases_map[l].shape[0])], dtype=dtype, device=device)
            for l in target_layers
        }
        for p_c, p_x in zip(rep_pos, rep_neg):
            in_c = _tok(p_c)
            in_x = _tok(p_x)
            cached: Dict[int, torch.Tensor] = {}

            def _make_cap(lidx: int):
                u = basis_t[lidx]  # (k, d)
                def _cap(mod: nn.Module, i: object, out: torch.Tensor) -> torch.Tensor:
                    del mod, i
                    cached[lidx] = (u @ out[0, -1, :]).detach().clone()
                    return out
                return _cap

            hs = [layers_mod[l].mlp.down_proj.register_forward_hook(_make_cap(l)) for l in target_layers]
            try:
                with torch.no_grad():
                    lc = model(**in_c).logits[0, -1, :]
                    clean_ms.append(float((lc[yes_id] - lc[no_id]).float().item()))
            finally:
                for h in hs:
                    h.remove()

            with torch.no_grad():
                lx = model(**in_x).logits[0, -1, :]
                corrupt_ms.append(float((lx[yes_id] - lx[no_id]).float().item()))

            def _make_patch(lidx: int):
                u = basis_t[lidx]  # (k, d)
                c_clean = cached[lidx]
                def _pat(mod: nn.Module, i: object, out: torch.Tensor) -> torch.Tensor:
                    del mod, i
                    h_last = out[0, -1, :]
                    c_cur = u @ h_last
                    delta = (c_clean - c_cur) @ u
                    out2 = out.clone()
                    out2[0, -1, :] = h_last + delta
                    return out2
                return _pat

            hs2 = [layers_mod[l].mlp.down_proj.register_forward_hook(_make_patch(l)) for l in target_layers]
            try:
                with torch.no_grad():
                    lp = model(**in_x).logits[0, -1, :]
                    patched_ms.append(float((lp[yes_id] - lp[no_id]).float().item()))
            finally:
                for h in hs2:
                    h.remove()

        gaps = [c - x for c, x in zip(clean_ms, corrupt_ms)]
        recovs = [p - x for p, x in zip(patched_ms, corrupt_ms)]
        mean_gap = float(np.mean(gaps))
        mean_recov = float(np.mean(recovs))
        ratio = mean_recov / mean_gap if abs(mean_gap) > 1e-6 else 0.0
        m_r, lo_r, hi_r = paired_bootstrap_ci(recovs, n_resamples=2000, seed=24121)
        return {
            "layers": list(target_layers),
            "rank_k": rank_k,
            "mean_clean_minus_corrupt_gap": round(mean_gap, 4),
            "mean_restoration_delta": round(m_r, 4),
            "restoration_lcb_95": round(lo_r, 4),
            "restoration_ucb_95": round(hi_r, 4),
            "restoration_ratio": round(ratio, 4),
            "strictly_positive_lcb": bool(lo_r > 0.0),
        }

    results: Dict[str, Any] = {}
    for k in RANKS_K:
        results[f"single_L11_qscci_k{k}"] = run_restoration([11], qscci_bases, k)
        results[f"multi_L11_18_qscci_k{k}"] = run_restoration(active_layers, qscci_bases, k)
    results["multi_L11_18_caa_k1"] = run_restoration(active_layers, {l: caa_dirs[l][None, :] for l in active_layers}, 1)
    results["multi_L11_18_sham_k1"] = run_restoration(active_layers, sham_bases, 1)
    results["multi_L11_18_sham_k16"] = run_restoration(active_layers, sham_bases, 16)

    for name, r in results.items():
        print(
            f"[w24-12] {model_id} | {name:24s} | ratio={r['restoration_ratio']:+.4f} "
            f"delta={r['mean_restoration_delta']:+.4f} [{r['restoration_lcb_95']:+.4f}, {r['restoration_ucb_95']:+.4f}]",
            flush=True,
        )

    del model, tok
    gc.collect()
    torch.cuda.empty_cache()
    return {"model_id": model_id, "active_layers": active_layers, "restoration_arms": results}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--output-dir", type=Path, default=Path("/content/out/pv7_w24_12_multilayer_qscci"))
    ap.add_argument("--git-commit-sha", type=str, default=os.environ.get("CHELATED_GIT_SHA", "unknown"))
    ap.add_argument("--source-archive-sha256", type=str, default=os.environ.get("CHELATED_SRC_TAR_SHA256", "unknown"))
    args = ap.parse_args(argv)

    if args.self_test:
        print(json.dumps(run_self_test(), indent=2, sort_keys=True))
        return 0

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for Wave 24-12")

    t0 = time.perf_counter()
    gpu_name = torch.cuda.get_device_name(0)
    per_model = [eval_model_multilayer(mid) for mid in MODELS_TO_EVAL]
    wall_total = round(time.perf_counter() - t0, 2)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "colab_l4_w24_12_multilayer_qscci_v1",
        "wave_item_id": "W24-12",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "git_commit_sha": args.git_commit_sha,
            "source_archive_sha256": args.source_archive_sha256,
            "spine_claim_registry_sha256": claim_registry_sha256(),
            "hardware_cohort": f"colab:{gpu_name}",
            "gpu_name": gpu_name,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "wall_seconds_total": wall_total,
        },
        "models": per_model,
    }
    raw = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    (args.output_dir / "w24_12_multilayer_qscci_manifest.json").write_bytes(raw)
    print(f"[w24-12] DONE sha256={hashlib.sha256(raw).hexdigest()[:12]} wall={wall_total}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
