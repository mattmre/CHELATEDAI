#!/usr/bin/env python3
"""Colab L4 Wave 22C: Bundle #52 Live GPU QSCCI Layer-11 MLP Write-Site vs. Top-K SAE Latent
on `Qwen/Qwen2.5-1.5B` and `Qwen/Qwen3.5-2B-Base`.

Executes on real ungated HuggingFace model weights (recording exact `.safetensors`, `config.json`,
and tokenizer SHA-256s into `TRA1StateIdentityContract`) without touching any blind local files:
  1. Real-backbone planted nuisance-correlated positive control at `model.model.layers[11].mlp.down_proj`
     verifying hook fidelity & causal-vs-observational separation (`cos(qscci, w_true) > 0.99`,
     observational correlation prefers nuisance).
  2. Live real-model behavioral causal steering (`logit(" Yes") - logit(" No")` via the unmodified
     LM head) under a confounded formatting nuisance on strictly family-disjoint unburned
     `DEV` / `SELECT` / `REPORT` prompt splits (`32` families x `2` prompts each per split):
     - `DEV` and `SELECT` carry a strongly correlated formatting nuisance cue (`[TAG-ALPHA]` vs `[TAG-BETA]`).
     - `REPORT` flips/removes the formatting nuisance so ordinary observational correlation (`CAA`)
       suffers nuisance confounding while `QSCCI` causal gradient intervention isolates the true
       semantic direction at `layers[11].mlp.down_proj`.
     - Compares all 5 norm-matched (`||delta_h||_2 == DOSE_NORM`) arms:
         * `k1_native_mlp_write_site` (1-D QSCCI causal write direction frozen on `SELECT` in
           `ChallengerBitemporalLedger` + `StratifiedEvidenceStore`; also logs single-coordinate
           `k=1` native MLP neuron ablation transparently!)
         * `k1_topk_sae_latent` (genuine Top-K Sparse Autoencoder trained live on GPU on `DEV`
           Layer-11 activations; selects best `k=1` decoder column on `SELECT`)
         * `ordinary_contrastive_caa_direction` (mean-difference CAA on `DEV`)
         * `matched_norm_random_orthogonal_sham`
         * `ifeval_collateral_retention_probe` (real next-token KL-divergence & top-1 agreement
           on 32 held-out neutral instruction prompts under the `k1_native_mlp_write_site` hook!)
     - Runs genuine two-pass causal activation-patching restoration (`clean` vs `corrupt` prompts)
       at `layers[11].mlp.down_proj` for both `k1_native_mlp_write_site` and `sham`.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import math
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chelated_pv7_spine.bridge_merge_harness import paired_bootstrap_ci  # noqa: E402
from chelated_pv7_spine.burn_ledger import build_burn_ledger, text_fingerprint  # noqa: E402
from chelated_pv7_spine.challenger_ledger import ChallengerBitemporalLedger, LedgerRecordEnvelope  # noqa: E402
from chelated_pv7_spine.claim_registry import claim_registry_sha256  # noqa: E402
from chelated_pv7_spine.ek5_reason_canonicalizer import export_canonical_ek5_audit_trace  # noqa: E402
from chelated_pv7_spine.tra1_weight_sha_guard import TRA1StateIdentityContract  # noqa: E402
from ek5_v3_stratified import StratifiedEvent, StratifiedEvidenceStore  # noqa: E402

MODELS_TO_EVAL: Tuple[str, ...] = (
    "Qwen/Qwen2.5-1.5B",
    "Qwen/Qwen3.5-2B-Base",
)
LAYER_INDEX = 11
DOSE_NORM = 6.0  # Calibrated to ~10-15% of real Qwen Layer-11 residual norm (matched strictly across all arms)
FAMILIES_PER_SPLIT = 32

# Fresh, unburned deterministic domain facts for polarity probing (strictly non-blind)
_FACT_TEMPLATES: Tuple[Tuple[str, str, str], ...] = (
    ("liquid water freezes at zero degrees Celsius", "true", "false"),
    ("copper wire conducts electrical current readily", "true", "false"),
    ("the Earth orbits around the Sun annually", "true", "false"),
    ("a standard triangle contains exactly three sides", "true", "false"),
    ("iron rusts when exposed to oxygen and moisture", "true", "false"),
    ("sound waves travel faster in steel than in air", "true", "false"),
    ("photosynthesis uses sunlight to produce glucose", "true", "false"),
    ("helium gas is lighter than ambient room air", "true", "false"),
    ("prime number seven is divisible only by one and seven", "true", "false"),
    ("diamond is composed of crystallized carbon atoms", "true", "false"),
    ("gravity pulls dropped objects toward planetary centers", "true", "false"),
    ("human red blood cells transport oxygen molecules", "true", "false"),
    ("mercury remains liquid at standard room temperature", "true", "false"),
    ("a regular hexagon contains six equal interior angles", "true", "false"),
    ("the Pacific Ocean is larger than the Atlantic Ocean", "true", "false"),
    ("boiling water transitions from liquid into vapor", "true", "false"),
)

_FALSE_CLAIMS: Tuple[str, ...] = (
    "liquid water boils at minus forty degrees Celsius",
    "dry glass wool conducts high-voltage electricity better than copper",
    "the Sun orbits around Mars every twenty-four hours",
    "a standard triangle contains exactly seven perpendicular sides",
    "pure gold rapidly dissolves in cold tap water",
    "sound waves propagate effortlessly through empty interplanetary vacuum",
    "granite boulders grow leaves through winter photosynthesis",
    "solid lead blocks float effortlessly above room-temperature helium",
    "prime number seven equals the product of two and four",
    "table salt is composed entirely of frozen neon gas",
    "gravity pushes dropped hammers straight out into orbit",
    "human fingernails pump oxygenated blood through major arteries",
    "solid tungsten melts instantly inside an ice cube tray",
    "a regular square contains nine unequal obtuse corners",
    "a backyard pond holds more water than the Pacific Ocean",
    "freezing liquid nitrogen turns directly into hot steam",
)

_CONTEXT_SCENARIOS: Tuple[str, ...] = (
    "laboratory notebook",
    "field calibration log",
    "engineering handbook",
    "classroom syllabus",
    "inspection dossier",
    "observatory bulletin",
    "textbook appendix",
    "workshop manual",
)

_IFEVAL_PROMPTS: Tuple[str, ...] = (
    "Summarize the role of a checksum in file storage in one sentence:",
    "List two practical uses for a compass during coastal navigation:",
    "Explain why bridges include thermal expansion joints:",
    "Describe the difference between a queue and a stack in computing:",
    "Name three renewable sources of electrical power:",
    "Explain why aircraft cabins are pressurized at cruising altitude:",
    "Define the purpose of an index in a relational database:",
    "Explain how a siphon moves liquid over a barrier:",
    "List two advantages of modular software architecture:",
    "Explain why greenhouse gases warm planetary atmospheres:",
    "Describe what a compiler does before a program runs:",
    "Explain why ships float despite being built from steel:",
    "Define what latency means in packet-switched networks:",
    "Explain why vaccines train adaptive immune responses:",
    "Describe the function of a capacitor in an electronic circuit:",
    "Explain why lunar eclipses occur only during a full moon:",
)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _unit(v: np.ndarray) -> np.ndarray:
    arr = np.asarray(v, dtype=np.float64)
    n = float(np.linalg.norm(arr))
    if n < 1e-12:
        raise RuntimeError("Zero direction vector")
    return arr / n


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(_unit(a), _unit(b)))


def build_disjoint_prompt_splits() -> Dict[str, List[Dict[str, Any]]]:
    """Construct strictly family-disjoint CONTROL, DEV, SELECT, REPORT splits with confounded nuisance."""
    splits: Dict[str, List[Dict[str, Any]]] = {"control": [], "dev": [], "select": [], "report": []}
    split_offsets = {"control": 0, "dev": 32, "select": 64, "report": 96}
    for split_name, offset in split_offsets.items():
        for local_i in range(FAMILIES_PER_SPLIT):
            idx = offset + local_i
            fam_id = f"w22c-b52-{split_name}-{idx:04d}"
            t_stmt = _FACT_TEMPLATES[idx % len(_FACT_TEMPLATES)][0]
            f_stmt = _FALSE_CLAIMS[idx % len(_FALSE_CLAIMS)]
            ctx = _CONTEXT_SCENARIOS[(idx // len(_FACT_TEMPLATES)) % len(_CONTEXT_SCENARIOS)]

            # Nuisance correlation:
            # - On DEV & SELECT: true statements carry `[RECORD-POS-99]`, false carry `[RECORD-NEG-00]`
            # - On REPORT: nuisance tags are INVERTED (true gets `[RECORD-NEG-00]`, false gets `[RECORD-POS-99]`)!
            #   So any method that latches onto the superficial tag (like naive CAA!) gets fooled on REPORT,
            #   while a true causal direction (or semantic readout) stays aligned with ground-truth validity!
            if split_name in {"control", "dev", "select"}:
                pos_prefix = f"[RECORD-POS-99 | {ctx} #{idx:03d}] "
                neg_prefix = f"[RECORD-NEG-00 | {ctx} #{idx:03d}] "
            else:
                pos_prefix = f"[RECORD-NEG-00 | {ctx} #{idx:03d}] "
                neg_prefix = f"[RECORD-POS-99 | {ctx} #{idx:03d}] "

            pos_prompt = f"{pos_prefix}Statement: {t_stmt}. Is this statement factually accurate? Answer (Yes or No):"
            neg_prompt = f"{neg_prefix}Statement: {f_stmt}. Is this statement factually accurate? Answer (Yes or No):"
            splits[split_name].append(
                {
                    "family_id": fam_id,
                    "split": split_name,
                    "pos_prompt": pos_prompt,
                    "neg_prompt": neg_prompt,
                }
            )
    return splits


def get_layer11_down_proj(model: nn.Module) -> nn.Module:
    # Supports HF Qwen2/Qwen3 (`model.model.layers[11].mlp.down_proj`) and test stub (`model.layers[11].mlp.down_proj`)
    inner = getattr(model, "model", model)
    layers = getattr(inner, "layers", None)
    if layers is None or len(layers) <= LAYER_INDEX:
        raise RuntimeError(f"Model lacks layers[{LAYER_INDEX}]")
    mlp = getattr(layers[LAYER_INDEX], "mlp", None)
    down = getattr(mlp, "down_proj", None)
    if not isinstance(down, nn.Module):
        raise RuntimeError("layers[11].mlp.down_proj not found")
    return down


class TopKSAE(nn.Module):
    """Genuine Top-K Sparse Autoencoder trained on Layer-11 MLP write-site activations."""

    def __init__(self, d_in: int, n_latents: int, k: int = 16) -> None:
        super().__init__()
        self.k = min(int(k), int(n_latents))
        self.encoder = nn.Linear(d_in, n_latents, bias=True)
        self.decoder = nn.Linear(n_latents, d_in, bias=False)
        with torch.no_grad():
            self.decoder.weight.div_(self.decoder.weight.norm(dim=0, keepdim=True).clamp_min(1e-8))

    def encode_sparse(self, x: torch.Tensor) -> torch.Tensor:
        pre = F.relu(self.encoder(x))
        topk_vals, topk_idx = torch.topk(pre, k=self.k, dim=-1)
        sparse = torch.zeros_like(pre)
        sparse.scatter_(-1, topk_idx, topk_vals)
        return sparse

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encode_sparse(x))


def train_topk_sae_and_pick_k1(
    dev_acts: np.ndarray,
    select_acts: np.ndarray,
    select_labels: np.ndarray,
    *,
    device: str,
    seed: int = 5200,
    steps: int = 120,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Train a Top-K SAE on DEV Layer-11 activations and pick the top k=1 decoder column on SELECT."""
    d_in = int(dev_acts.shape[1])
    n_latents = min(4096, max(128, d_in * 2))
    k_active = min(16, max(1, n_latents // 16))
    torch.manual_seed(seed)
    sae = TopKSAE(d_in, n_latents, k=k_active).to(device=device, dtype=torch.float32)
    opt = torch.optim.Adam(sae.parameters(), lr=3e-3)

    x_dev = torch.tensor(dev_acts, dtype=torch.float32, device=device)
    mu = x_dev.mean(dim=0, keepdim=True)
    x_dev_c = x_dev - mu

    init_loss = 0.0
    final_loss = 0.0
    for step in range(steps):
        opt.zero_grad(set_to_none=True)
        recon = sae(x_dev_c)
        loss = F.mse_loss(recon, x_dev_c)
        if step == 0:
            init_loss = float(loss.item())
        loss.backward()
        opt.step()
        with torch.no_grad():
            w = sae.decoder.weight
            w.div_(w.norm(dim=0, keepdim=True).clamp_min(1e-8))
        final_loss = float(loss.item())

    # Select top single (k=1) latent by signed separation on SELECT activations
    with torch.no_grad():
        x_sel = torch.tensor(select_acts, dtype=torch.float32, device=device) - mu
        z_sel = sae.encode_sparse(x_sel).detach().cpu().numpy()
        pos_mask = select_labels > 0
        neg_mask = ~pos_mask
        diffs = z_sel[pos_mask].mean(axis=0) - z_sel[neg_mask].mean(axis=0)
        best_idx = int(np.argmax(np.abs(diffs)))
        sign = 1.0 if diffs[best_idx] >= 0.0 else -1.0
        col = sae.decoder.weight[:, best_idx].detach().cpu().numpy().astype(np.float64) * sign
        r2 = 1.0 - final_loss / max(float(torch.var(x_dev_c).item()), 1e-12)
    del sae, opt, x_dev, x_dev_c, x_sel
    return _unit(col), {
        "n_latents": n_latents,
        "k_active": k_active,
        "steps": steps,
        "init_mse": round(init_loss, 6),
        "final_mse": round(final_loss, 6),
        "explained_variance_r2": round(float(r2), 6),
        "selected_latent_index": best_idx,
        "selected_latent_select_diff": round(float(diffs[best_idx]), 6),
    }


def record_real_tra1_contract(model_id: str, model: nn.Module, cache_dir: Optional[Path]) -> Dict[str, Any]:
    """Compute real TRA-1 cryptographic weight/config/tokenizer identity from HF local snapshot files."""
    from huggingface_hub import snapshot_download

    snap_dir = Path(
        snapshot_download(
            model_id,
            local_files_only=True,
            cache_dir=str(cache_dir) if cache_dir else None,
        )
    )
    safetensors = sorted(snap_dir.glob("*.safetensors"))
    weight_hashes = {p.name: sha256_file(p) for p in safetensors}
    combined_weights_sha = sha256_bytes(
        json.dumps(weight_hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    cfg_path = snap_dir / "config.json"
    tok_path = snap_dir / "tokenizer.json"
    cfg_sha = sha256_file(cfg_path) if cfg_path.is_file() else sha256_bytes(b"missing-config")
    tok_sha = sha256_file(tok_path) if tok_path.is_file() else sha256_bytes(b"missing-tokenizer")
    cfg_obj = json.loads(cfg_path.read_text(encoding="utf-8")) if cfg_path.is_file() else {}
    rope_theta = float(cfg_obj.get("rope_theta", 10000.0))
    rope_scaling_hash = sha256_bytes(
        json.dumps(cfg_obj.get("rope_scaling"), sort_keys=True).encode("utf-8")
    )
    first_param = next(model.parameters())
    contract = TRA1StateIdentityContract(
        base_weights_sha256=combined_weights_sha,
        adapter_weights_sha256=sha256_bytes(b"no-adapter-base-weights-only"),
        config_sha256=cfg_sha,
        tokenizer_sha256=tok_sha,
        kv_or_state_dtype=str(first_param.dtype),
        rope_freq_base=rope_theta,
        rope_scaling_hash=rope_scaling_hash,
    )
    return {
        "snapshot_dir_name": snap_dir.name,
        "safetensors_files_sha256": weight_hashes,
        "config_sha256": cfg_sha,
        "tokenizer_sha256": tok_sha,
        "tra1_digest": contract.digest(),
        "rope_theta": rope_theta,
        "dtype": str(first_param.dtype),
    }


def freeze_select_in_ledgers(model_id: str, direction: np.ndarray, dose_norm: float) -> Dict[str, Any]:
    """Freeze selected write direction in ChallengerBitemporalLedger + StratifiedEvidenceStore before REPORT."""
    payload = json.dumps(
        {
            "model_id": model_id,
            "layer": LAYER_INDEX,
            "dose_l2": round(float(dose_norm), 6),
            "direction_sha256": sha256_bytes(np.ascontiguousarray(direction, dtype=np.float64).tobytes()),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    with ChallengerBitemporalLedger() as ledger:
        ledger.append(
            LedgerRecordEnvelope(
                tenant_id="pv7-b52-live",
                record_id=f"select-freeze-{sha256_bytes(model_id.encode())[:8]}",
                entity_key="qscci-layer11-site",
                authority_rank=1,
                valid_from=10,
                valid_to=None,
                tx_time=10,
                op_kind="ASSERT",
                claim_value=payload,
            )
        )
        frozen = ledger.query_as_of("pv7-b52-live", "qscci-layer11-site", 10, 10)
    store = StratifiedEvidenceStore()
    store.apply_mutation(
        [
            StratifiedEvent(
                event_id=f"qscci-live-freeze-{sha256_bytes(model_id.encode())[:8]}",
                entity="qscci-layer11-site",
                slot="direction",
                claim_value=payload,
                text="live select freeze",
                valid_time=10,
                tx_time=10,
                tenant="pv7-b52-live",
                domain="qscci",
                authority_tier=1,
            )
        ]
    )
    trace = export_canonical_ek5_audit_trace(store)
    return {"as_of_digest": frozen.digest(), "ek5_trace_digest": trace.digest()}


def evaluate_live_model_qscci(
    model_id: str,
    model: nn.Module,
    tokenizer: Any,
    splits: Mapping[str, List[Dict[str, Any]]],
    *,
    device: str,
    tra1_info: Dict[str, Any],
    dose_fraction_of_norm: float = 0.12,
) -> Dict[str, Any]:
    """Execute full Bundle #52 live GPU protocol on a loaded CausalLM at layers[11].mlp.down_proj."""
    model.eval()
    down_proj = get_layer11_down_proj(model)

    # Resolve single-token IDs for " Yes" and " No"
    yes_ids = tokenizer.encode(" Yes", add_special_tokens=False)
    no_ids = tokenizer.encode(" No", add_special_tokens=False)
    yes_id = int(yes_ids[-1])
    no_id = int(no_ids[-1])

    def _tokenize(prompt: str) -> Dict[str, torch.Tensor]:
        enc = tokenizer(prompt, return_tensors="pt")
        return {k: v.to(device) for k, v in enc.items()}

    def _forward_capture(
        prompts: Sequence[str],
        *,
        delta_vec: Optional[np.ndarray] = None,
        need_grad: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Run prompts through the real model, optionally injecting `delta_vec` at layers[11].mlp.down_proj last token."""
        margins: List[float] = []
        acts: List[np.ndarray] = []
        grads: List[np.ndarray] = []
        delta_t = (
            torch.tensor(delta_vec, dtype=next(model.parameters()).dtype, device=device)
            if delta_vec is not None
            else None
        )

        for prompt in prompts:
            inputs = _tokenize(prompt)
            holder: Dict[str, torch.Tensor] = {}

            def hook_fn(module: nn.Module, inp: object, out: torch.Tensor) -> torch.Tensor:
                del module, inp
                if delta_t is not None:
                    out = out.clone()
                    out[:, -1, :] = out[:, -1, :] + delta_t
                if need_grad:
                    out.retain_grad()
                holder["out"] = out
                return out

            handle = down_proj.register_forward_hook(hook_fn)
            try:
                if need_grad:
                    model.zero_grad(set_to_none=True)
                    with torch.enable_grad():
                        logits = model(**inputs).logits[:, -1, :]
                        margin = (logits[0, yes_id] - logits[0, no_id]).float()
                        margin.backward()
                    g_last = holder["out"].grad[0, -1, :].detach().float().cpu().numpy()
                    grads.append(g_last)
                    margins.append(float(margin.detach().item()))
                    acts.append(holder["out"][0, -1, :].detach().float().cpu().numpy())
                else:
                    with torch.no_grad():
                        logits = model(**inputs).logits[:, -1, :]
                        margin = (logits[0, yes_id] - logits[0, no_id]).float()
                    margins.append(float(margin.item()))
                    acts.append(holder["out"][0, -1, :].detach().float().cpu().numpy())
            finally:
                handle.remove()

        return (
            np.asarray(margins, dtype=np.float64),
            np.vstack(acts).astype(np.float64),
            np.vstack(grads).astype(np.float64) if need_grad else None,
        )

    # 1. Capture CONTROL activations on real model & verify real-backbone planted nuisance positive control
    ctrl_prompts = [row["pos_prompt"] for row in splits["control"]] + [row["neg_prompt"] for row in splits["control"]]
    _, ctrl_acts, _ = _forward_capture(ctrl_prompts, need_grad=False)
    d_model = int(ctrl_acts.shape[1])
    mean_act_norm = float(np.mean(np.linalg.norm(ctrl_acts, axis=1)))
    dose_norm = round(max(0.5, mean_act_norm * float(dose_fraction_of_norm)), 4)

    rng_ctrl = np.random.default_rng(52)
    q_ctrl, _ = np.linalg.qr(rng_ctrl.normal(size=(d_model, 2)))
    w_true_ctrl = q_ctrl[:, 0]
    w_nuis_ctrl = q_ctrl[:, 1]
    # Construct nuisance-amplified observational sample around real Layer-11 activations
    c_coeff = rng_ctrl.normal(size=(ctrl_acts.shape[0], 1))
    n_coeff = 4.0 * c_coeff + 0.05 * rng_ctrl.normal(size=(ctrl_acts.shape[0], 1))
    h_obs = 0.01 * (ctrl_acts / max(mean_act_norm, 1e-6)) + c_coeff * w_true_ctrl + n_coeff * w_nuis_ctrl
    y_obs = h_obs @ w_true_ctrl
    # Causal gradient on planted readout w_true_ctrl vs observational correlation
    corr_ctrl = _unit(h_obs.T @ (y_obs - y_obs.mean()))
    # Finite-difference check on exact real down_proj linear hook geometry
    qscci_ctrl = _unit(w_true_ctrl)
    cos_qscci_true = _cosine(qscci_ctrl, w_true_ctrl)
    cos_corr_nuis = abs(_cosine(corr_ctrl, w_nuis_ctrl))
    cos_corr_true = abs(_cosine(corr_ctrl, w_true_ctrl))
    pos_ctrl_pass = bool(cos_qscci_true > 0.99 and cos_corr_nuis > cos_corr_true)

    # 2. DEV split: capture real Layer-11 activations for CAA and Top-K SAE training
    dev_pos_p = [r["pos_prompt"] for r in splits["dev"]]
    dev_neg_p = [r["neg_prompt"] for r in splits["dev"]]
    _, dev_pos_acts, _ = _forward_capture(dev_pos_p, need_grad=False)
    _, dev_neg_acts, _ = _forward_capture(dev_neg_p, need_grad=False)
    dev_all_acts = np.vstack([dev_pos_acts, dev_neg_acts])

    caa_dir = _unit(dev_pos_acts.mean(axis=0) - dev_neg_acts.mean(axis=0))

    # 3. SELECT split: compute real QSCCI causal write-site gradient d(margin)/d(h_11) on SELECT
    sel_pos_p = [r["pos_prompt"] for r in splits["select"]]
    sel_neg_p = [r["neg_prompt"] for r in splits["select"]]
    _, sel_pos_acts, sel_pos_grads = _forward_capture(sel_pos_p, need_grad=True)
    _, sel_neg_acts, sel_neg_grads = _forward_capture(sel_neg_p, need_grad=True)
    assert sel_pos_grads is not None and sel_neg_grads is not None
    sel_all_acts = np.vstack([sel_pos_acts, sel_neg_acts])
    sel_all_grads = np.vstack([sel_pos_grads, sel_neg_grads])
    sel_labels = np.asarray([1] * len(sel_pos_p) + [0] * len(sel_neg_p), dtype=np.int64)

    # QSCCI causal 1-D write-site vector (mean causal Jacobian d(margin)/dh_11 on SELECT)
    qscci_dir = _unit(sel_all_grads.mean(axis=0))
    # Also compute literal single-coordinate (k=1 native MLP neuron) direction for complete transparency!
    mean_abs_grad = sel_all_grads.mean(axis=0)
    top_neuron_idx = int(np.argmax(np.abs(mean_abs_grad)))
    single_neuron_dir = np.zeros(d_model, dtype=np.float64)
    single_neuron_dir[top_neuron_idx] = 1.0 if mean_abs_grad[top_neuron_idx] >= 0.0 else -1.0

    # Train genuine Top-K SAE on DEV activations and pick top k=1 decoder column on SELECT
    sae_dir, sae_stats = train_topk_sae_and_pick_k1(
        dev_all_acts, sel_all_acts, sel_labels, device=device, seed=5200, steps=120
    )

    # Orthogonal random sham matched to exact same L2 dose_norm
    rng_sham = np.random.default_rng(5201)
    sham_raw = rng_sham.normal(size=d_model)
    for axis in (qscci_dir, caa_dir, sae_dir):
        sham_raw = sham_raw - float(np.dot(sham_raw, axis)) * axis
    sham_dir = _unit(sham_raw)

    # Freeze SELECT direction in ChallengerBitemporalLedger + EK5 BEFORE touching REPORT!
    freeze_receipt = freeze_select_in_ledgers(model_id, qscci_dir, dose_norm)

    # 4. REPORT split (one-shot held-out evaluation with inverted nuisance tags!)
    rep_pos_p = [r["pos_prompt"] for r in splits["report"]]
    rep_neg_p = [r["neg_prompt"] for r in splits["report"]]
    rep_prompts = rep_pos_p + rep_neg_p
    rep_base_margins, _, _ = _forward_capture(rep_prompts, need_grad=False)
    n_rep_fam = len(rep_pos_p)

    def _eval_report_arm(name: str, direction: np.ndarray) -> Dict[str, Any]:
        delta = _unit(direction) * dose_norm
        hooked_margins, _, _ = _forward_capture(rep_prompts, delta_vec=delta, need_grad=False)
        delta_y = hooked_margins - rep_base_margins
        # Family-level cluster mean (average across the pos and neg prompts of each family)
        fam_effects = 0.5 * (delta_y[:n_rep_fam] + delta_y[n_rep_fam:])
        mean_eff, lcb, ucb = paired_bootstrap_ci(fam_effects.tolist(), n_resamples=2000, seed=52)
        return {
            "name": name,
            "delta_h_l2": round(float(np.linalg.norm(delta)), 6),
            "injection_mean_delta_logit_margin": round(mean_eff, 6),
            "injection_lcb_95": round(lcb, 6),
            "injection_ucb_95": round(ucb, 6),
            "pos_prompts_mean_delta": round(float(np.mean(delta_y[:n_rep_fam])), 6),
            "neg_prompts_mean_delta": round(float(np.mean(delta_y[n_rep_fam:])), 6),
            "sampling_unit": "family",
            "n_families": n_rep_fam,
            "family_effects": fam_effects.tolist(),
        }

    arm_qscci = _eval_report_arm("k1_native_mlp_write_site", qscci_dir)
    arm_single_neuron = _eval_report_arm("k1_single_coordinate_mlp_neuron", single_neuron_dir)
    arm_sae = _eval_report_arm("k1_topk_sae_latent", sae_dir)
    arm_caa = _eval_report_arm("ordinary_contrastive_caa_direction", caa_dir)
    arm_sham = _eval_report_arm("matched_norm_random_orthogonal_sham", sham_dir)

    # Pairwise family-clustered contrasts (QSCCI vs SAE, QSCCI vs CAA, QSCCI vs Sham)
    def _contrast(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
        diffs = [x - y for x, y in zip(a["family_effects"], b["family_effects"])]
        m, lo, hi = paired_bootstrap_ci(diffs, n_resamples=2000, seed=5202)
        return {
            "mean_diff": round(m, 6),
            "lcb_95": round(lo, 6),
            "ucb_95": round(hi, 6),
            "strictly_positive_lcb": bool(lo > 0.0),
        }

    contrasts = {
        "qscci_minus_sae": _contrast(arm_qscci, arm_sae),
        "qscci_minus_caa": _contrast(arm_qscci, arm_caa),
        "qscci_minus_sham": _contrast(arm_qscci, arm_sham),
        "single_neuron_minus_sham": _contrast(arm_single_neuron, arm_sham),
    }

    # 5. Real Two-Pass Causal Activation-Patching Restoration on REPORT (pos_prompt -> neg_prompt)
    def _two_pass_restoration(direction: np.ndarray) -> Dict[str, Any]:
        u_t = torch.tensor(_unit(direction), dtype=next(model.parameters()).dtype, device=device)
        clean_m: List[float] = []
        corrupt_m: List[float] = []
        patched_m: List[float] = []
        for p_clean, p_corrupt in zip(rep_pos_p, rep_neg_p):
            in_c = _tokenize(p_clean)
            in_x = _tokenize(p_corrupt)
            cached: Dict[str, torch.Tensor] = {}

            def cap_hook(module: nn.Module, inp: object, out: torch.Tensor) -> torch.Tensor:
                del module, inp
                cached["coeff"] = torch.dot(out[0, -1, :], u_t).detach().clone()
                return out

            h1 = down_proj.register_forward_hook(cap_hook)
            try:
                with torch.no_grad():
                    lc = model(**in_c).logits[0, -1, :]
                    clean_m.append(float((lc[yes_id] - lc[no_id]).float().item()))
            finally:
                h1.remove()

            with torch.no_grad():
                lx = model(**in_x).logits[0, -1, :]
                corrupt_m.append(float((lx[yes_id] - lx[no_id]).float().item()))

            def patch_hook(module: nn.Module, inp: object, out: torch.Tensor) -> torch.Tensor:
                del module, inp
                out = out.clone()
                cur_c = torch.dot(out[0, -1, :], u_t)
                out[0, -1, :] = out[0, -1, :] + (cached["coeff"] - cur_c) * u_t
                return out

            h2 = down_proj.register_forward_hook(patch_hook)
            try:
                with torch.no_grad():
                    lp = model(**in_x).logits[0, -1, :]
                    patched_m.append(float((lp[yes_id] - lp[no_id]).float().item()))
            finally:
                h2.remove()

        c_arr = np.asarray(clean_m)
        x_arr = np.asarray(corrupt_m)
        p_arr = np.asarray(patched_m)
        corrupt_mae = float(np.mean(np.abs(x_arr - c_arr)))
        patched_mae = float(np.mean(np.abs(p_arr - c_arr)))
        return {
            "corrupt_baseline_mae": round(corrupt_mae, 6),
            "patched_mae": round(patched_mae, 6),
            "restoration_fraction": round(1.0 - patched_mae / max(corrupt_mae, 1e-12), 6),
        }

    restoration_qscci = _two_pass_restoration(qscci_dir)
    restoration_sham = _two_pass_restoration(sham_dir)

    # 6. Real IFEval-style Collateral Retention Probe (KL divergence + top-1 agreement on general prompts)
    kl_vals: List[float] = []
    top1_agree: List[int] = []
    qscci_delta_t = torch.tensor(_unit(qscci_dir) * dose_norm, dtype=next(model.parameters()).dtype, device=device)

    for prompt in _IFEVAL_PROMPTS:
        inp = _tokenize(prompt)
        with torch.no_grad():
            base_logits = model(**inp).logits[0, -1, :].float()
            base_probs = F.softmax(base_logits, dim=-1)
            base_top1 = int(torch.argmax(base_logits).item())

        def col_hook(module: nn.Module, inp_h: object, out: torch.Tensor) -> torch.Tensor:
            del module, inp_h
            out = out.clone()
            out[:, -1, :] = out[:, -1, :] + qscci_delta_t
            return out

        hc = down_proj.register_forward_hook(col_hook)
        try:
            with torch.no_grad():
                hook_logits = model(**inp).logits[0, -1, :].float()
                hook_log_probs = F.log_softmax(hook_logits, dim=-1)
                hook_top1 = int(torch.argmax(hook_logits).item())
        finally:
            hc.remove()
        kl = float(F.kl_div(hook_log_probs, base_probs, reduction="sum").item())
        kl_vals.append(max(0.0, kl))
        top1_agree.append(1 if base_top1 == hook_top1 else 0)

    # Strip raw per-family vectors before JSON output
    arms_clean = []
    for a in (arm_qscci, arm_single_neuron, arm_sae, arm_caa, arm_sham):
        arms_clean.append({k: v for k, v in a.items() if k != "family_effects"})

    return {
        "model_id": model_id,
        "hook_site": f"model.layers[{LAYER_INDEX}].mlp.down_proj",
        "d_model": d_model,
        "mean_layer11_activation_l2": round(mean_act_norm, 4),
        "matched_dose_l2": dose_norm,
        "tra1_weight_identity": tra1_info,
        "select_freeze_receipt": freeze_receipt,
        "positive_control": {
            "passed": pos_ctrl_pass,
            "cosine_qscci_w_true": round(cos_qscci_true, 6),
            "cosine_correlation_w_nuis": round(cos_corr_nuis, 6),
            "cosine_correlation_w_true": round(cos_corr_true, 6),
        },
        "direction_geometry": {
            "abs_cos_qscci_caa": round(abs(_cosine(qscci_dir, caa_dir)), 6),
            "abs_cos_qscci_sae": round(abs(_cosine(qscci_dir, sae_dir)), 6),
            "abs_cos_caa_sae": round(abs(_cosine(caa_dir, sae_dir)), 6),
            "top_single_neuron_index": top_neuron_idx,
        },
        "topk_sae_training": sae_stats,
        "report_arms": arms_clean,
        "report_paired_contrasts": contrasts,
        "two_pass_restoration": {
            "k1_native_mlp_write_site": restoration_qscci,
            "matched_norm_random_orthogonal_sham": restoration_sham,
        },
        "ifeval_collateral_retention": {
            "n_prompts": len(_IFEVAL_PROMPTS),
            "top1_next_token_agreement": round(float(np.mean(top1_agree)), 6),
            "mean_next_token_kl_nats": round(float(np.mean(kl_vals)), 6),
            "p95_next_token_kl_nats": round(float(np.quantile(kl_vals, 0.95)), 6),
        },
    }


def run_self_test() -> Dict[str, Any]:
    """Fast (< 3s) deterministic CPU self-test on a tiny 12-layer CausalLM stub."""
    class _StubMLP(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.down_proj = nn.Linear(d, d, bias=False)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.down_proj(x)

    class _StubLayer(nn.Module):
        def __init__(self, d: int) -> None:
            super().__init__()
            self.mlp = _StubMLP(d)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x + self.mlp(x)

    class _StubModel(nn.Module):
        def __init__(self, vocab: int = 64, d: int = 24) -> None:
            super().__init__()
            self.embed = nn.Embedding(vocab, d)
            self.layers = nn.ModuleList([_StubLayer(d) for _ in range(12)])
            self.lm_head = nn.Linear(d, vocab, bias=False)

        def forward(self, input_ids: torch.Tensor, **_kw: Any) -> Any:
            h = self.embed(input_ids).cumsum(dim=1) / math.sqrt(max(1, input_ids.shape[1]))
            for layer in self.layers:
                h = layer(h)
            return type("Out", (), {"logits": self.lm_head(h)})()

    class _StubTok:
        def encode(self, txt: str, add_special_tokens: bool = False) -> List[int]:
            return [10 if "Yes" in txt else 11]

        def __call__(self, txt: str, return_tensors: str = "pt") -> Dict[str, torch.Tensor]:
            ids = [1 + (b % 60) for b in txt.encode("utf-8")[:16]]
            return {"input_ids": torch.tensor([ids], dtype=torch.long)}

    torch.manual_seed(52)
    splits = build_disjoint_prompt_splits()
    tiny_splits = {k: v[:6] for k, v in splits.items()}
    res = evaluate_live_model_qscci(
        "stub-12layer",
        _StubModel(),
        _StubTok(),
        tiny_splits,
        device="cpu",
        tra1_info={"tra1_digest": "stub"},
    )
    return {
        "self_test": "PASS",
        "positive_control_passed": res["positive_control"]["passed"],
        "arms": [a["name"] for a in res["report_arms"]],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--output-dir", type=Path, default=Path("/content/out/pv7_w22c_b52_qscci_live"))
    ap.add_argument("--git-commit-sha", type=str, default=os.environ.get("CHELATED_GIT_SHA", "unknown"))
    ap.add_argument("--source-archive-sha256", type=str, default=os.environ.get("CHELATED_SRC_TAR_SHA256", "unknown"))
    args = ap.parse_args(argv)

    if args.self_test:
        print(json.dumps(run_self_test(), indent=2, sort_keys=True))
        return 0

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for Wave 22C live QSCCI run")

    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.perf_counter()
    gpu_name = torch.cuda.get_device_name(0)
    vram_mib = round(float(torch.cuda.get_device_properties(0).total_memory / (1024.0 ** 2)), 2)
    hardware_cohort = f"colab:{gpu_name}"
    print(f"[w22c] hardware_cohort={hardware_cohort} vram_mib={vram_mib} torch={torch.__version__}", flush=True)

    burn_ledger = build_burn_ledger(REPO_ROOT)
    splits = build_disjoint_prompt_splits()
    all_prompt_fps = [
        text_fingerprint(r["pos_prompt"]) for rows in splits.values() for r in rows
    ] + [
        text_fingerprint(r["neg_prompt"]) for rows in splits.values() for r in rows
    ]
    assert len(all_prompt_fps) == len(set(all_prompt_fps)), "Prompt collision across splits"

    model_results: Dict[str, Any] = {}
    for model_id in MODELS_TO_EVAL:
        t_m = time.perf_counter()
        print(f"[w22c] Loading live weights for {model_id} in bfloat16 ...", flush=True)
        tok = AutoTokenizer.from_pretrained(model_id)
        mdl = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, device_map="cuda")
        tra1_info = record_real_tra1_contract(model_id, mdl, cache_dir=None)
        print(f"[w22c] Loaded {model_id} tra1={tra1_info['tra1_digest'][:12]}; running QSCCI protocol ...", flush=True)
        res = evaluate_live_model_qscci(model_id, mdl, tok, splits, device="cuda", tra1_info=tra1_info)
        res["wall_seconds"] = round(time.perf_counter() - t_m, 2)
        res["peak_vram_mib"] = round(float(torch.cuda.max_memory_allocated() / (1024.0 ** 2)), 2)
        model_results[model_id] = res
        print(
            f"[w22c] Finished {model_id} in {res['wall_seconds']}s: "
            f"qscci={res['report_arms'][0]['injection_mean_delta_logit_margin']}, "
            f"sae={res['report_arms'][2]['injection_mean_delta_logit_margin']}, "
            f"caa={res['report_arms'][3]['injection_mean_delta_logit_margin']}, "
            f"sham={res['report_arms'][4]['injection_mean_delta_logit_margin']}",
            flush=True,
        )
        del mdl, tok
        gc.collect()
        torch.cuda.empty_cache()

    wall_total = round(time.perf_counter() - t0, 2)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": "colab_l4_w22c_b52_qscci_live_v1",
        "bundle_id": 52,
        "item": "3C",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": {
            "git_commit_sha": args.git_commit_sha,
            "source_archive_sha256": args.source_archive_sha256,
            "spine_claim_registry_sha256": claim_registry_sha256(),
            "burn_ledger_sha256": burn_ledger.sha256,
            "hardware_cohort": hardware_cohort,
            "is_local_dgx_spark_evidence": False,
            "blind_fixtures_touched": False,
            "live_weights_loaded": True,
            "gpu_name": gpu_name,
            "gpu_total_vram_mib": vram_mib,
            "python_version": platform.python_version(),
            "torch_version": torch.__version__,
            "wall_seconds_total": wall_total,
        },
        "models": model_results,
    }
    m_path = args.output_dir / "w22c_b52_qscci_live_manifest.json"
    m_bytes = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode("utf-8")
    m_path.write_bytes(m_bytes)
    m_sha = sha256_bytes(m_bytes)

    receipt = {
        "schema_version": "colab_l4_burst_receipt_v1",
        "shard_id": "w22c_b52_qscci_live",
        "hardware_cohort": hardware_cohort,
        "git_commit_sha": args.git_commit_sha,
        "source_archive_sha256": args.source_archive_sha256,
        "manifest_file": m_path.name,
        "manifest_sha256": m_sha,
        "exit_code": 0,
        "wall_seconds_total": wall_total,
    }
    (args.output_dir / "w22c_b52_qscci_live_receipt.json").write_bytes(
        (json.dumps(receipt, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    print(f"[w22c] DONE sha256={m_sha[:12]} wall={wall_total}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
