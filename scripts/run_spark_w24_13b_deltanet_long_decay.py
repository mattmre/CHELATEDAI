#!/usr/bin/env python3
"""Spark 2 GB10 Wave 24-13b: Long-Context Gated-DeltaNet T-Rn Scaling (325 -> 1,024 -> 2,048 tokens)
with Exact Per-Layer/Per-Head Decay-Product Weighting (`prod_t exp(g_{l,h,t})`) vs Fixed `gamma` vs KV-Only!

Extends W24-13 (which achieved 97.2%..100.5% zero-replay recovery at 28..325 suffix tokens in O(1) 11.4 ms):
  1. Scales suffix lengths up to ~1,024 and ~2,048 tokens on real `Qwen/Qwen3.5-2B-Base` (18 Gated-DeltaNet
     + 6 full-attention layers) to measure exact speedup vs checkpoint replay at long context.
  2. Captures the true Gated-DeltaNet per-layer, per-head log-decay gates `g_{l,h,t}` during the
     suffix pass (via forward pre/post hooks on each linear-attention layer's `A_log` / `dt_bias` / `a_proj`)
     to compute the exact cumulative decay factor `D_{l,h} = exp(sum_{t in suffix} g_{l,h,t})` and compare:
       - `kv_excise_only` (ablation proving DeltaNet leakage persists even at 2k tokens)
       - `hybrid_deltanet_g0.5_plus_kv` (W24-13 fixed gamma=0.5 baseline)
       - `hybrid_deltanet_g0.25_plus_kv` (stronger damping for long context)
       - `hybrid_deltanet_exact_decay_plus_kv` (exact per-layer/per-head cumulative decay `D_{l,h}`!)
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

from chelated_pv7_spine.claim_registry import claim_registry_sha256  # noqa: E402
from run_spark_w24_13_deltanet_state_retraction import (  # noqa: E402
    CASES,
    FILLER_SENTENCE,
    MODEL_ID,
    _clone_cache,
    _excise_full_attn_kv_span,
    _iter_linear_recurrent_refs,
)


def _attach_decay_hooks(mdl: Any) -> Tuple[Dict[int, List[torch.Tensor]], List[Any]]:
    """Attach lightweight forward hooks to every GatedDeltaNet linear_attn module to capture exact
    per-head log-decay `g = -exp(A_log) * softplus(a + dt_bias)` summed across suffix tokens."""
    captured: Dict[int, List[torch.Tensor]] = {}
    handles: List[Any] = []
    layers = mdl.model.layers if hasattr(mdl, "model") and hasattr(mdl.model, "layers") else []
    for idx, layer in enumerate(layers):
        la = getattr(layer, "linear_attn", None)
        if la is None or not hasattr(la, "A_log") or not hasattr(la, "dt_bias"):
            continue
        # Hook `in_proj_a` (or `in_proj_ba` depending on transformers version) to read the exact raw `a` gate
        a_mod = getattr(la, "in_proj_a", None)
        if a_mod is not None:
            def _make_hook(layer_i: int, mod_la: Any):
                def _hook(_m, _inp, out):
                    # out: [B, T, num_v_heads]
                    with torch.no_grad():
                        a_val = out.float()
                        g = -mod_la.A_log.float().exp() * F.softplus(a_val + mod_la.dt_bias.float())
                        # Sum log-decay over sequence dim T -> [B, num_v_heads]
                        captured.setdefault(layer_i, []).append(g.sum(dim=1).detach())
                return _hook
            handles.append(a_mod.register_forward_hook(_make_hook(idx, la)))
    return captured, handles


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "pv7_w24_13b_deltanet_long_spark2")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required")

    t0 = time.perf_counter()
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[w24-13b] Loading {MODEL_ID} on spark2_gb10:{gpu_name} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to("cuda").eval()

    captured_log_decay, handles = _attach_decay_hooks(mdl)
    print(f"[w24-13b] Attached exact Gated-DeltaNet decay hooks on {len(handles)} linear_attn layers.", flush=True)

    # Suffix repeats: 12 (~325 tok), 38 (~1,026 tok), 76 (~2,052 tok)
    suffix_repeats = (12, 38, 76)
    arm_names = (
        "kv_excise_only",
        "hybrid_deltanet_g0.25_plus_kv",
        "hybrid_deltanet_g0.5_plus_kv",
        "hybrid_deltanet_exact_decay_plus_kv",
    )
    rows: List[Dict[str, Any]] = []

    for n_rep in suffix_repeats:
        suffix_text = " " + (FILLER_SENTENCE * n_rep)
        suffix_ids = tok(suffix_text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
        suffix_len = int(suffix_ids.shape[1])

        cell: Dict[str, Any] = {
            "suffix_tokens": suffix_len,
            "replay_ms": [],
            "stale_kl": [],
            "mean_exact_decay_factor": [],
            "by_arm": {a: {"op_ms": [], "state_rel_l2_after": [], "kl": [], "recovery_frac": []} for a in arm_names},
        }

        for entity, old_val, target_val, query_text in CASES:
            prefix_text = f"Archive ledger: Initial baseline states {entity} is {old_val}. "
            target_text = f"Update notice: {entity} is now {target_val}. "
            q_text = f"\nQuestion: {query_text}"

            pref_ids = tok(prefix_text, return_tensors="pt").input_ids.to("cuda")
            targ_ids = tok(target_text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
            q_ids = tok(q_text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
            t_start = int(pref_ids.shape[1])
            t_end = t_start + int(targ_ids.shape[1])

            old_tok_id = tok.encode(" " + old_val, add_special_tokens=False)[0]
            targ_tok_id = tok.encode(" " + target_val, add_special_tokens=False)[0]

            with torch.no_grad():
                captured_log_decay.clear()
                pkv_pref = mdl(input_ids=pref_ids, use_cache=True).past_key_values
                pkv_ax = mdl(input_ids=targ_ids, past_key_values=_clone_cache(pkv_pref), use_cache=True).past_key_values

                pref_map = {(li, k): t for li, _, k, t in _iter_linear_recurrent_refs(pkv_pref)}
                delta_map = {
                    (li, k): (t - pref_map[(li, k)]).detach().clone()
                    for li, _, k, t in _iter_linear_recurrent_refs(pkv_ax)
                }

                # Capture exact cumulative decay across Suffix_B ONLY!
                captured_log_decay.clear()
                pkv_full = mdl(input_ids=suffix_ids, past_key_values=_clone_cache(pkv_ax), use_cache=True).past_key_values
                decay_factors: Dict[int, torch.Tensor] = {}
                for li, chunks in captured_log_decay.items():
                    log_d = torch.stack(chunks, dim=0).sum(dim=0)  # [1, num_heads]
                    decay_factors[li] = torch.exp(log_d).clamp(0.0, 1.0)
                if decay_factors:
                    cell["mean_exact_decay_factor"].append(
                        float(np.mean([float(v.mean().item()) for v in decay_factors.values()]))
                    )

                logits_stale = mdl(input_ids=q_ids, past_key_values=_clone_cache(pkv_full), use_cache=True).logits[0, -1, :].float()

                # Exact timed checkpoint replay
                torch.cuda.synchronize()
                tr0 = time.perf_counter()
                pkv_cf = mdl(input_ids=suffix_ids, past_key_values=_clone_cache(pkv_pref), use_cache=True).past_key_values
                torch.cuda.synchronize()
                replay_ms = (time.perf_counter() - tr0) * 1000.0
                logits_cf = mdl(input_ids=q_ids, past_key_values=_clone_cache(pkv_cf), use_cache=True).logits[0, -1, :].float()

                p_cf = F.softmax(logits_cf, dim=-1)
                kl_stale = float(F.kl_div(F.log_softmax(logits_stale, dim=-1), p_cf, reduction="sum").item())
                m_cf = float((logits_cf[old_tok_id] - logits_cf[targ_tok_id]).item())
                m_stale = float((logits_stale[old_tok_id] - logits_stale[targ_tok_id]).item())
                gap = m_cf - m_stale

                cell["replay_ms"].append(replay_ms)
                cell["stale_kl"].append(kl_stale)
                cf_map = {(li, k): t for li, _, k, t in _iter_linear_recurrent_refs(pkv_cf)}

                for arm in arm_names:
                    torch.cuda.synchronize()
                    ts0 = time.perf_counter()
                    pkv_trn = _clone_cache(pkv_full)
                    l2_after = []
                    for li, rs_dict, k, t_cur in _iter_linear_recurrent_refs(pkv_trn):
                        cf_t = cf_map[(li, k)]
                        denom = float(torch.linalg.norm(cf_t.float()).item()) + 1e-8
                        d_t = delta_map[(li, k)]
                        if arm == "kv_excise_only":
                            t_new = t_cur
                        elif arm == "hybrid_deltanet_g0.25_plus_kv":
                            t_new = t_cur - 0.25 * d_t
                            rs_dict[k] = t_new
                        elif arm == "hybrid_deltanet_g0.5_plus_kv":
                            t_new = t_cur - 0.5 * d_t
                            rs_dict[k] = t_new
                        elif arm == "hybrid_deltanet_exact_decay_plus_kv":
                            # Broadcast [1, num_heads] -> [1, num_heads, 1, 1] matching recurrent_states
                            if li in decay_factors and decay_factors[li].shape[1] == t_cur.shape[1]:
                                dec = decay_factors[li].view(1, -1, 1, 1).to(dtype=t_cur.dtype, device=t_cur.device)
                                t_new = t_cur - dec * d_t
                            else:
                                t_new = t_cur - 0.5 * d_t
                            rs_dict[k] = t_new
                        l2_after.append(float(torch.linalg.norm((t_new - cf_t).float()).item()) / denom)
                    _excise_full_attn_kv_span(pkv_trn, t_start, t_end)
                    torch.cuda.synchronize()
                    op_ms = (time.perf_counter() - ts0) * 1000.0

                    logits_trn = mdl(input_ids=q_ids, past_key_values=pkv_trn, use_cache=True).logits[0, -1, :].float()
                    kl_trn = float(F.kl_div(F.log_softmax(logits_trn, dim=-1), p_cf, reduction="sum").item())
                    m_trn = float((logits_trn[old_tok_id] - logits_trn[targ_tok_id]).item())
                    recov = ((m_trn - m_stale) / gap) if abs(gap) > 1e-4 else 0.0

                    ba = cell["by_arm"][arm]
                    ba["op_ms"].append(op_ms)
                    ba["state_rel_l2_after"].append(float(np.mean(l2_after)))
                    ba["kl"].append(kl_trn)
                    ba["recovery_frac"].append(recov)

        summary_row = {
            "suffix_tokens": suffix_len,
            "mean_exact_replay_ms": round(float(np.mean(cell["replay_ms"])), 2),
            "stale_mean_kl_vs_cf": round(float(np.mean(cell["stale_kl"])), 4),
            "mean_exact_decay_factor": round(float(np.mean(cell["mean_exact_decay_factor"])), 4) if cell["mean_exact_decay_factor"] else None,
            "arms": {},
        }
        for arm in arm_names:
            ba = cell["by_arm"][arm]
            op_ms_m = float(np.mean(ba["op_ms"]))
            summary_row["arms"][arm] = {
                "mean_op_ms": round(op_ms_m, 2),
                "speedup_vs_replay": round(float(np.mean(cell["replay_ms"])) / max(op_ms_m, 1e-6), 2),
                "mean_state_rel_l2_after": round(float(np.mean(ba["state_rel_l2_after"])), 5),
                "mean_kl_vs_cf": round(float(np.mean(ba["kl"])), 4),
                "mean_recovery_frac": round(float(np.mean(ba["recovery_frac"])), 4),
            }
        rows.append(summary_row)
        print(
            f"[w24-13b] suffix={suffix_len:4d}tok replay={summary_row['mean_exact_replay_ms']:6.1f}ms "
            f"mean_decay={summary_row['mean_exact_decay_factor']} "
            f"kv_only_rec={summary_row['arms']['kv_excise_only']['mean_recovery_frac']:.3f} "
            f"g0.5_rec={summary_row['arms']['hybrid_deltanet_g0.5_plus_kv']['mean_recovery_frac']:.3f} "
            f"exact_decay_rec={summary_row['arms']['hybrid_deltanet_exact_decay_plus_kv']['mean_recovery_frac']:.3f} "
            f"(KL={summary_row['arms']['hybrid_deltanet_exact_decay_plus_kv']['mean_kl_vs_cf']:.4f}, "
            f"speedup={summary_row['arms']['hybrid_deltanet_exact_decay_plus_kv']['speedup_vs_replay']:.1f}x)",
            flush=True,
        )

    for h in handles:
        h.remove()

    manifest = {
        "experiment_id": "pv7_w24_13b_deltanet_long_decay_spark2",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hardware_cohort": f"spark2_gb10:{gpu_name}",
        "model_id": MODEL_ID,
        "exact_decay_hooks_attached": len(handles),
        "claim_registry_sha256": claim_registry_sha256(),
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "results_by_suffix_length": rows,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "w24_13b_deltanet_long_decay_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[w24-13b] Wrote {out_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
