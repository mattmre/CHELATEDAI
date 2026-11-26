#!/usr/bin/env python3
"""Wave 24-13 (Lane 7b): Exploratory T-Rn Test on Genuine Frozen Gated-DeltaNet Recurrent States
(`Qwen/Qwen3.5-2B-Base` — 18 Gated-DeltaNet linear-attention layers + 6 full-attention layers).

Tests the true compressed neural-state retraction hypothesis (`T-Rn`) that Opus noted did not exist in #60:
  Given a real multi-record stream `[Prefix_A] + [Target_X] + [Suffix_B] + [Query_Q]` vs the exact
  retracted counterfactual `[Prefix_A] + [Suffix_B] + [Query_Q]`, compares:
    1. `exact_checkpoint_replay`: Restore checkpoint at end of `Prefix_A` and replay `Suffix_B`
       (exact oracle; cost scales with `|Suffix_B|`).
    2. `stale_no_retraction`: Query full stream with `Target_X` still in state.
    3. `trn_algebraic_state_sub`: Subtract `Delta_S = S(Prefix_A + Target_X) - S(Prefix_A)` directly
       from the final compressed Gated-DeltaNet `recurrent_states` (with decay-scaled `gamma in {0.25, 0.5, 1.0}`)
       WITHOUT replaying `Suffix_B`, measuring:
         - exact recurrent-state relative L2 error vs counterfactual state,
         - next-token KL divergence vs exact counterfactual logits,
         - target-token answer log-prob recovery fraction (0 = stale, 1 = exact retraction),
         - wall-clock latency ratio vs exact suffix replay across suffix lengths L in {32, 128, 384} tokens!
"""

from __future__ import annotations

import argparse
import copy
import gc
import hashlib
import json
import os
import platform
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

from chelated_pv7_spine.claim_registry import claim_registry_sha256  # noqa: E402

MODEL_ID = "Qwen/Qwen3.5-2B-Base"

CASES: Tuple[Tuple[str, str, str, str], ...] = (
    ("Project Orion", "cobalt", "amber", "What is the current code color for Project Orion? Answer in one word:"),
    ("Server Node-7", "offline", "online", "What is the current status of Server Node-7? Answer in one word:"),
    ("Vessel Aurora", "Lisbon", "Oslo", "What is the current port of Vessel Aurora? Answer in one word:"),
    ("Policy Alpha", "denied", "approved", "What is the current state of Policy Alpha? Answer in one word:"),
    ("Reactor Core-3", "halted", "active", "What is the current mode of Reactor Core-3? Answer in one word:"),
    ("Vault Key-9", "revoked", "valid", "What is the current status of Vault Key-9? Answer in one word:"),
    ("Route Delta", "closed", "open", "What is the current condition of Route Delta? Answer in one word:"),
    ("Batch Sigma", "failed", "passed", "What is the current audit result of Batch Sigma? Answer in one word:"),
)

FILLER_SENTENCE = (
    "Routine telemetry log entry confirms nominal power rail stability, balanced network packet routing, "
    "archived checksum verification, and scheduled maintenance alignment across peripheral nodes. "
)


def _clone_cache(pkv: Any) -> Any:
    """Deep-clone a HuggingFace hybrid/dynamic cache cleanly on GPU."""
    return copy.deepcopy(pkv)


def _iter_linear_recurrent_refs(pkv: Any) -> List[Tuple[int, Any, Any, torch.Tensor]]:
    """Return list of (layer_idx, state_dict, state_key, tensor) for all LinearAttentionLayer recurrent_states."""
    refs: List[Tuple[int, Any, Any, torch.Tensor]] = []
    if hasattr(pkv, "layers"):
        for idx, layer in enumerate(pkv.layers):
            rs = getattr(layer, "recurrent_states", None)
            if isinstance(rs, dict):
                for k, v in rs.items():
                    if isinstance(v, torch.Tensor):
                        refs.append((idx, rs, k, v))
    return refs


def _excise_full_attn_kv_span(pkv: Any, start_pos: int, end_pos: int) -> int:
    """Excise token positions [start_pos, end_pos) from all full-attention DynamicLayer KV caches (zero replay!)."""
    n_excised_layers = 0
    if hasattr(pkv, "layers"):
        for layer in pkv.layers:
            if hasattr(layer, "keys") and hasattr(layer, "values") and isinstance(layer.keys, torch.Tensor):
                k = layer.keys
                v = layer.values
                if k.shape[2] >= end_pos:
                    layer.keys = torch.cat([k[:, :, :start_pos, :], k[:, :, end_pos:, :]], dim=2).contiguous()
                    layer.values = torch.cat([v[:, :, :start_pos, :], v[:, :, end_pos:, :]], dim=2).contiguous()
                    n_excised_layers += 1
    return n_excised_layers


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "pv7_w24_13_deltanet_spark2")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for W24-13")

    t0 = time.perf_counter()
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[w24-13] Loading {MODEL_ID} on {gpu_name} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    mdl = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.bfloat16).to("cuda").eval()

    tc = mdl.config.text_config if hasattr(mdl.config, "text_config") else mdl.config
    layer_types = list(getattr(tc, "layer_types", []))
    n_linear = sum(1 for x in layer_types if "linear" in str(x).lower())
    n_full = sum(1 for x in layer_types if "full" in str(x).lower())
    print(f"[w24-13] Architecture: total_layers={len(layer_types)} linear_deltanet={n_linear} full_attn={n_full}", flush=True)

    # Probe cache structure once
    probe_ids = tok("Hello world.", return_tensors="pt").input_ids.to("cuda")
    with torch.no_grad():
        p_out = mdl(input_ids=probe_ids, use_cache=True)
    rec_refs = _iter_linear_recurrent_refs(p_out.past_key_values)
    rec_shapes = [tuple(t.shape) for _, _, _, t in rec_refs]
    rec_attr = "layers[*].recurrent_states"
    print(f"[w24-13] Recurrent state attr={rec_attr!r} active_tensors={len(rec_shapes)} sample_shape={rec_shapes[:2]}", flush=True)
    if not rec_shapes:
        raise RuntimeError("Failed to locate LinearAttentionLayer recurrent_states tensors")
    del p_out

    suffix_repeats = (1, 4, 12)  # ~28, ~109, ~325 suffix tokens
    arms = (
        ("deltanet_sub_g0.5", 0.5, False),
        ("deltanet_sub_g1.0", 1.0, False),
        ("kv_excise_only", 0.0, True),
        ("hybrid_deltanet_g0.5_plus_kv_excise", 0.5, True),
        ("hybrid_deltanet_g1.0_plus_kv_excise", 1.0, True),
    )
    rows: List[Dict[str, Any]] = []

    for n_rep in suffix_repeats:
        suffix_text = " " + (FILLER_SENTENCE * n_rep)
        suffix_ids = tok(suffix_text, return_tensors="pt", add_special_tokens=False).input_ids.to("cuda")
        suffix_len = int(suffix_ids.shape[1])

        cell_metrics = {
            "suffix_tokens": suffix_len,
            "replay_ms": [],
            "stale_kl": [],
            "stale_margin_err": [],
            "by_arm": {
                name: {"op_ms": [], "state_rel_l2_before": [], "state_rel_l2_after": [], "kl": [], "recovery_frac": []}
                for name, _, _ in arms
            },
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
                # 1. Prefix checkpoint S(A)
                out_pref = mdl(input_ids=pref_ids, use_cache=True)
                pkv_pref = out_pref.past_key_values

                # 2. Target increment S(A + X)
                pkv_ax_in = _clone_cache(pkv_pref)
                out_ax = mdl(input_ids=targ_ids, past_key_values=pkv_ax_in, use_cache=True)
                pkv_ax = out_ax.past_key_values

                # Exact DeltaNet recurrent state increment Delta_S = S(A+X) - S(A) keyed by (layer_idx, key)
                pref_map = {(li, k): t for li, _, k, t in _iter_linear_recurrent_refs(pkv_pref)}
                delta_map = {
                    (li, k): (t - pref_map[(li, k)]).detach().clone()
                    for li, _, k, t in _iter_linear_recurrent_refs(pkv_ax)
                }

                # 3. Full (contaminated) stream S(A + X + B) -> logits on Q
                pkv_full_in = _clone_cache(pkv_ax)
                out_full_b = mdl(input_ids=suffix_ids, past_key_values=pkv_full_in, use_cache=True)
                pkv_full = out_full_b.past_key_values
                out_stale_q = mdl(input_ids=q_ids, past_key_values=_clone_cache(pkv_full), use_cache=True)
                logits_stale = out_stale_q.logits[0, -1, :].float()

                # 4. Exact Segment-Tree Checkpoint Replay: S(A) + replay(B) -> logits on Q (timed!)
                torch.cuda.synchronize()
                tr0 = time.perf_counter()
                pkv_rep_in = _clone_cache(pkv_pref)
                out_cf_b = mdl(input_ids=suffix_ids, past_key_values=pkv_rep_in, use_cache=True)
                torch.cuda.synchronize()
                replay_ms = (time.perf_counter() - tr0) * 1000.0
                pkv_cf = out_cf_b.past_key_values
                out_cf_q = mdl(input_ids=q_ids, past_key_values=_clone_cache(pkv_cf), use_cache=True)
                logits_cf = out_cf_q.logits[0, -1, :].float()

                p_cf = F.softmax(logits_cf, dim=-1)
                log_p_stale = F.log_softmax(logits_stale, dim=-1)
                kl_stale = float(F.kl_div(log_p_stale, p_cf, reduction="sum").item())

                m_cf = float((logits_cf[old_tok_id] - logits_cf[targ_tok_id]).item())
                m_stale = float((logits_stale[old_tok_id] - logits_stale[targ_tok_id]).item())
                gap = m_cf - m_stale

                cell_metrics["replay_ms"].append(replay_ms)
                cell_metrics["stale_kl"].append(kl_stale)
                cell_metrics["stale_margin_err"].append(abs(gap))

                cf_map = {(li, k): t for li, _, k, t in _iter_linear_recurrent_refs(pkv_cf)}
                full_map = {(li, k): t for li, _, k, t in _iter_linear_recurrent_refs(pkv_full)}

                # 5. Evaluate all 5 zero-replay retraction arms on real DeltaNet + Full-Attn hybrid cache!
                for arm_name, gamma, excise_kv in arms:
                    torch.cuda.synchronize()
                    ts0 = time.perf_counter()
                    pkv_trn = _clone_cache(pkv_full)
                    l2_before = []
                    l2_after = []
                    for li, rs_dict, k, t_cur in _iter_linear_recurrent_refs(pkv_trn):
                        cf_t = cf_map[(li, k)]
                        denom = float(torch.linalg.norm(cf_t.float()).item()) + 1e-8
                        l2_before.append(float(torch.linalg.norm((full_map[(li, k)] - cf_t).float()).item()) / denom)
                        if gamma != 0.0:
                            t_new = t_cur - gamma * delta_map[(li, k)]
                            rs_dict[k] = t_new
                        else:
                            t_new = t_cur
                        l2_after.append(float(torch.linalg.norm((t_new - cf_t).float()).item()) / denom)
                    if excise_kv:
                        _excise_full_attn_kv_span(pkv_trn, t_start, t_end)
                    torch.cuda.synchronize()
                    op_ms = (time.perf_counter() - ts0) * 1000.0

                    out_trn_q = mdl(input_ids=q_ids, past_key_values=pkv_trn, use_cache=True)
                    logits_trn = out_trn_q.logits[0, -1, :].float()
                    log_p_trn = F.log_softmax(logits_trn, dim=-1)
                    kl_trn = float(F.kl_div(log_p_trn, p_cf, reduction="sum").item())
                    m_trn = float((logits_trn[old_tok_id] - logits_trn[targ_tok_id]).item())
                    recov = ((m_trn - m_stale) / gap) if abs(gap) > 1e-4 else 0.0

                    ba = cell_metrics["by_arm"][arm_name]
                    ba["op_ms"].append(op_ms)
                    ba["state_rel_l2_before"].append(float(np.mean(l2_before)))
                    ba["state_rel_l2_after"].append(float(np.mean(l2_after)))
                    ba["kl"].append(kl_trn)
                    ba["recovery_frac"].append(recov)

        summary_row = {
            "suffix_tokens": suffix_len,
            "mean_exact_replay_ms": round(float(np.mean(cell_metrics["replay_ms"])), 2),
            "stale_mean_kl_vs_cf": round(float(np.mean(cell_metrics["stale_kl"])), 4),
            "stale_mean_margin_gap": round(float(np.mean(cell_metrics["stale_margin_err"])), 4),
            "by_arm": {
                name: {
                    "mean_op_ms": round(float(np.mean(v["op_ms"])), 2),
                    "speedup_vs_replay": round(
                        float(np.mean(cell_metrics["replay_ms"])) / max(float(np.mean(v["op_ms"])), 1e-6), 2
                    ),
                    "state_rel_l2_before": round(float(np.mean(v["state_rel_l2_before"])), 5),
                    "state_rel_l2_after": round(float(np.mean(v["state_rel_l2_after"])), 5),
                    "mean_kl_vs_cf": round(float(np.mean(v["kl"])), 4),
                    "mean_margin_recovery_frac": round(float(np.mean(v["recovery_frac"])), 4),
                }
                for name, v in cell_metrics["by_arm"].items()
            },
        }
        rows.append(summary_row)
        print(f"[w24-13] === suffix={suffix_len:3d} tok | replay={summary_row['mean_exact_replay_ms']:.1f}ms | stale_KL={summary_row['stale_mean_kl_vs_cf']:.3f} ===", flush=True)
        for name, m in summary_row["by_arm"].items():
            print(
                f"  {name:35s} | op={m['mean_op_ms']:4.1f}ms ({m['speedup_vs_replay']:5.1f}x) | "
                f"DeltaNet_L2 {m['state_rel_l2_before']:.3f}->{m['state_rel_l2_after']:.3f} | "
                f"KL={m['mean_kl_vs_cf']:.3f} | recov={m['mean_margin_recovery_frac']:+.3f}",
                flush=True,
            )

    del mdl, tok
    gc.collect()
    torch.cuda.empty_cache()

    wall = round(time.perf_counter() - t0, 2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "w24_13_deltanet_state_retraction_v1",
        "wave_item_id": "W24-13",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "model_id": MODEL_ID,
            "hardware_cohort": f"spark2_gb10:{gpu_name}",
            "gpu_name": gpu_name,
            "spine_claim_registry_sha256": claim_registry_sha256(),
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "linear_deltanet_layers": n_linear,
            "full_attn_layers": n_full,
            "recurrent_attr": rec_attr,
            "n_recurrent_tensors": len(rec_shapes),
            "sample_recurrent_shape": list(rec_shapes[0]) if rec_shapes else [],
            "wall_seconds_total": wall,
        },
        "n_cases": len(CASES),
        "results_by_suffix_len": rows,
    }
    raw = (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode("utf-8")
    (args.output_dir / "w24_13_deltanet_retraction_manifest.json").write_bytes(raw)
    print(f"[w24-13] DONE sha256={hashlib.sha256(raw).hexdigest()[:12]} wall={wall}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
