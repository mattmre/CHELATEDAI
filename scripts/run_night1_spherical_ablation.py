"""Spherical Distillation & 3rd-Order Cumulant Strictly Additive Ablation (mlr-v1).

Tests on the exact same complete-qrels (mlr-v1) DEV/SELECT/REPORT splits and cached
embeddings as Night-1 Flagship #1:

Part 1 — Strictly Additive Zero-Write Single-Hop Ladder (E0->E1 anisotropic cone, E0->E3 cross-dim):
  A0: uncentered_procrustes            (Night-1 2nd-order uncentered baseline)
  A1: ridge_baseline                   (Night-1 closed-form Ridge baseline)
  A2: centered_procrustes              (+1st-order spherical centroid unwrapping)
  A3: centered_whitened_procrustes     (+2nd-order covariance / CORAL whitening)
  A4: centered_proc_plus_3rd_cumulant  (+explicit 3rd-order projected cumulant/skewness on top-r subspace)
  A5: spherical_kl_distill             (+relational sharp-tau spherical KL + vMF distillation on S^{d-1})
  A6: spherical_kl_plus_3rd_order      (+explicit 3rd-order cubic kernel + top-r cumulant on top of KL)
  A7: spherical_kl_plus_csls           (+parameter-free #355 CSLS k=10 on top of A5 & A6)
  (Note: A2, A4, A5, A6 form an exact 2x2 factorial: {No-KL, KL} x {No-3rd, 3rd}.)

Part 2 — Multi-Hop Composition Chain (E0 -> E1 -> E2 -> E3 at m=1, 2, 3):
  Compares composed Ridge (Night-1) vs composed Uncentered Procrustes vs
  composed Centered Procrustes (A2) vs composed Spherical KL (A5) vs composed Spherical KL+3rd (A6).

Part 3 — Strictly Additive Selective Backfill & Calibration Ladder (10% and 20% budgets):
  B0: map_plus_backfill_zscore_fastfill      (Night-1 baseline)
  B1: +quantile_cdf_calibration              (fixes heterogeneous cone score miscalibration)
  B2: +spherical_distortion_order            (qrels-free per-doc neighborhood KL/cosine residual order)
  B3: +exposure_x_distortion_hybrid_order    (combines FastFill traffic exposure x spherical distortion)
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from m1_baseline_ladder import (  # noqa: E402
    DEFAULT_RIDGE_LAMBDAS,
    compute_fastfill_backfill_order,
    evaluate_rankings_mlr,
    fit_closed_form_ridge,
    fit_closed_form_svd_procrustes,
    score_dense_rankings,
    score_map_plus_backfill,
    select_qrels_independent_anchors,
    split_query_ids,
)
from scripts.night1_common import (  # noqa: E402
    ENCODERS,
    current_git_sha,
    encode_or_load_embeddings,
    load_beir_dataset,
    metric_contract_digest,
    paired_bootstrap_delta,
    process_telemetry,
    sha256_json,
    summarize_per_query_metrics,
    write_json_atomic,
)


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(norms > 1e-12, norms, 1.0)


def _ndcg_map(per_q: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    return {qid: float(vals["ndcg_at_10"]) for qid, vals in per_q.items()}


# ---------------------------------------------------------------------------
# A2: Centered Spherical Procrustes (1st-order centroid + 2nd-order orthogonal)
# ---------------------------------------------------------------------------
def fit_centered_procrustes(
    x_src: np.ndarray,
    y_tgt: np.ndarray,
    fit_idx: np.ndarray,
) -> Dict[str, np.ndarray]:
    xs = np.asarray(x_src[fit_idx], dtype=np.float64)
    yt = np.asarray(y_tgt[fit_idx], dtype=np.float64)
    mu_x = xs.mean(axis=0, keepdims=True)
    mu_y = yt.mean(axis=0, keepdims=True)
    xc = xs - mu_x
    yc = yt - mu_y
    m = xc.T @ yc
    u, _, vt = np.linalg.svd(m, full_matrices=False)
    q = u @ vt
    return {"mu_x": mu_x, "mu_y": mu_y, "Q": q}


def apply_centered_procrustes(x: np.ndarray, model: Dict[str, np.ndarray]) -> np.ndarray:
    xc = np.asarray(x, dtype=np.float64) - model["mu_x"]
    return _l2_normalize(xc @ model["Q"] + model["mu_y"])


# ---------------------------------------------------------------------------
# A3: Centered + Shrinkage-Whitened (CORAL) Spherical Procrustes (2nd-order cov)
# ---------------------------------------------------------------------------
def _inv_sqrt_cov(xc: np.ndarray, shrink: float) -> Tuple[np.ndarray, np.ndarray]:
    n, d = xc.shape
    cov = (xc.T @ xc) / max(n - 1, 1)
    tr_scale = float(np.trace(cov)) / float(d)
    cov_reg = (1.0 - shrink) * cov + shrink * tr_scale * np.eye(d, dtype=np.float64)
    eigvals, eigvecs = np.linalg.eigh(cov_reg)
    eigvals = np.maximum(eigvals, 1e-8)
    inv_sqrt = (eigvecs * (eigvals ** -0.5)) @ eigvecs.T
    sqrt_mat = (eigvecs * (eigvals ** 0.5)) @ eigvecs.T
    return inv_sqrt, sqrt_mat


def fit_whitened_centered_procrustes(
    x_src: np.ndarray,
    y_tgt: np.ndarray,
    fit_idx: np.ndarray,
    shrink: float = 0.20,
) -> Dict[str, np.ndarray]:
    xs = np.asarray(x_src[fit_idx], dtype=np.float64)
    yt = np.asarray(y_tgt[fit_idx], dtype=np.float64)
    mu_x = xs.mean(axis=0, keepdims=True)
    mu_y = yt.mean(axis=0, keepdims=True)
    xc = xs - mu_x
    yc = yt - mu_y
    wx_inv, _ = _inv_sqrt_cov(xc, shrink=shrink)
    wy_inv, wy_sqrt = _inv_sqrt_cov(yc, shrink=shrink)
    xw = xc @ wx_inv
    yw = yc @ wy_inv
    u, _, vt = np.linalg.svd(xw.T @ yw, full_matrices=False)
    q = u @ vt
    w_total = wx_inv @ q @ wy_sqrt
    return {"mu_x": mu_x, "mu_y": mu_y, "W": w_total}


def apply_affine_spherical(x: np.ndarray, model: Dict[str, np.ndarray]) -> np.ndarray:
    xc = np.asarray(x, dtype=np.float64) - model["mu_x"]
    return _l2_normalize(xc @ model["W"] + model["mu_y"])


# ---------------------------------------------------------------------------
# A4 / A5 / A6: Spherical Distillation + Explicit 3rd-Order Projected Cumulants
# ---------------------------------------------------------------------------
class SphericalBridgeNet(nn.Module):
    """Centered orthogonal base + spectral-norm-bounded low-rank tangent residual on S^{d-1}."""

    def __init__(
        self,
        mu_x: np.ndarray,
        mu_y: np.ndarray,
        q_init: np.ndarray,
        *,
        rank: int = 64,
        max_residual_scale: float = 0.35,
        use_residual: bool = True,
    ) -> None:
        super().__init__()
        d_in, d_out = q_init.shape
        self.register_buffer("mu_x", torch.tensor(mu_x, dtype=torch.float32))
        self.register_buffer("mu_y_base", torch.tensor(mu_y, dtype=torch.float32))
        self.register_buffer("q_base", torch.tensor(q_init, dtype=torch.float32))
        self.mu_y_delta = nn.Parameter(torch.zeros(1, d_out))
        self.use_residual = use_residual
        self.max_residual_scale = float(max_residual_scale)

        # Skew-symmetric Cayley orthogonal fine-tuning in target space (exact SO(d_out), cond=1)
        self.skew_raw = nn.Parameter(torch.zeros(d_out, d_out))
        if use_residual:
            r_eff = min(rank, d_in, d_out)
            self.u_res = nn.Parameter(torch.randn(d_in, r_eff) * 0.01)
            self.v_res = nn.Parameter(torch.zeros(r_eff, d_out))
            self.act_gate = nn.Parameter(torch.zeros(1, r_eff))
        else:
            self.u_res = None
            self.v_res = None
            self.act_gate = None

    def orthogonal_refinement(self) -> torch.Tensor:
        a = 0.5 * (self.skew_raw - self.skew_raw.t())
        eye = torch.eye(a.size(0), device=a.device, dtype=a.dtype)
        # Cayley transform: R = (I - A)(I + A)^{-1} in SO(d_out)
        return torch.linalg.solve(eye + a, eye - a)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xc = x - self.mu_x
        r_orth = self.orthogonal_refinement()
        base = (xc @ self.q_base) @ r_orth + (self.mu_y_base + self.mu_y_delta)
        if self.use_residual and self.u_res is not None and self.v_res is not None:
            h = F.gelu(xc @ self.u_res + self.act_gate)
            res = h @ self.v_res
            # Bound residual norm relative to base so spectral conditioning stays tight
            res_norm = res.norm(dim=1, keepdim=True).clamp(min=1e-8)
            cap = torch.minimum(torch.ones_like(res_norm), self.max_residual_scale / res_norm)
            base = base + res * cap
        return F.normalize(base, p=2, dim=1)


def _projected_third_cumulant(z: torch.Tensor, basis_r: torch.Tensor) -> torch.Tensor:
    """Compute exact 3rd-order central cumulant tensor E[u_a u_b u_c] in top-r subspace (r^3)."""
    zc = z - z.mean(dim=0, keepdim=True)
    u = zc @ basis_r  # [B, r]
    # Standardize per axis slightly for numerical balance
    std = u.std(dim=0, keepdim=True).clamp(min=1e-4)
    un = u / std
    # C3[a, b, c] = mean_i u[i,a]*u[i,b]*u[i,c]
    return torch.einsum("ia,ib,ic->abc", un, un, un) / float(max(un.size(0), 1))


def fit_spherical_distillation(
    x_src: np.ndarray,
    y_tgt: np.ndarray,
    fit_idx: np.ndarray,
    centered_init: Dict[str, np.ndarray],
    *,
    use_kl: bool,
    use_third_order: bool,
    use_residual: bool,
    tau_teacher: float = 0.07,
    tau_student: float = 0.07,
    top_k_neighbors: int = 64,
    subspace_r: int = 16,
    steps: int = 220,
    batch_size: int = 512,
    lr: float = 3e-3,
    seed: int = 42,
) -> Dict[str, Any]:
    """Train strictly qrels-free spherical bridge on GPU with exact ablation switches."""
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    xs_np = np.asarray(x_src[fit_idx], dtype=np.float32)
    yt_np = np.asarray(y_tgt[fit_idx], dtype=np.float32)
    n_fit = xs_np.shape[0]

    # Top-r principal basis of centered target anchors for projected 3rd-order cumulant tensor
    yc_np = yt_np - yt_np.mean(axis=0, keepdims=True)
    _, _, vt_y = np.linalg.svd(yc_np[: min(n_fit, 4096)], full_matrices=False)
    r_eff = min(subspace_r, vt_y.shape[0])
    basis_r = torch.tensor(vt_y[:r_eff].T, dtype=torch.float32, device=device)

    x_all = torch.tensor(xs_np, dtype=torch.float32, device=device)
    y_all = torch.tensor(yt_np, dtype=torch.float32, device=device)
    x_all_n = F.normalize(x_all, p=2, dim=1)
    y_all_n = F.normalize(y_all, p=2, dim=1)

    model = SphericalBridgeNet(
        centered_init["mu_x"],
        centered_init["mu_y"],
        centered_init["Q"],
        rank=64,
        max_residual_scale=0.35,
        use_residual=use_residual,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)
    rng = np.random.default_rng(seed + 17)

    # Precompute target 3rd-order cumulant on a deterministic anchor pool when enabled
    with torch.no_grad():
        c3_tgt_global = _projected_third_cumulant(y_all_n[: min(n_fit, 4096)], basis_r)

    t0 = time.perf_counter()
    final_losses: Dict[str, float] = {}
    for _ in range(steps):
        if n_fit > batch_size:
            b_idx = rng.choice(n_fit, size=batch_size, replace=False)
            idx_t = torch.tensor(b_idx, dtype=torch.long, device=device)
            xb = x_all_n[idx_t]
            yb = y_all_n[idx_t]
        else:
            xb = x_all_n
            yb = y_all_n

        pred = model(xb)  # [B, d_out], unit-norm on S^{d_out - 1}
        # Base vMF directional alignment: 1 - <f(x_i), y_i>
        loss_vmf = (1.0 - (pred * yb).sum(dim=1)).mean()

        loss_kl = torch.tensor(0.0, device=device)
        if use_kl and xb.size(0) >= 8:
            # Teacher spherical neighborhood in new space; Student mapped query vs frozen old docs
            sim_t = (xb @ xb.t()) / tau_teacher
            sim_s = (pred @ yb.t()) / tau_student
            diag = torch.eye(xb.size(0), dtype=torch.bool, device=device)
            sim_t = sim_t.masked_fill(diag, -1e9)
            sim_s = sim_s.masked_fill(diag, -1e9)
            # Restrict to top-k spherical neighbors per anchor so local ranking direction dominates
            k_eff = min(top_k_neighbors, xb.size(0) - 1)
            top_vals_t, top_idx = torch.topk(sim_t, k=k_eff, dim=1)
            top_vals_s = torch.gather(sim_s, dim=1, index=top_idx)
            p_t = F.softmax(top_vals_t, dim=1)
            log_p_t = F.log_softmax(top_vals_t, dim=1)
            log_p_s = F.log_softmax(top_vals_s, dim=1)
            loss_kl = (p_t * (log_p_t - log_p_s)).sum(dim=1).mean()

        loss_c3 = torch.tensor(0.0, device=device)
        if use_third_order and xb.size(0) >= 16:
            # 1) Explicit top-r projected 3rd-order central cumulant matching
            c3_pred = _projected_third_cumulant(pred, basis_r)
            c3_batch_tgt = _projected_third_cumulant(yb, basis_r)
            loss_cum = F.mse_loss(c3_pred, 0.5 * (c3_batch_tgt + c3_tgt_global))
            # 2) Explicit 3rd-order cubic spherical kernel matching: (<x_i,x_j>^3 vs <f(x_i),y_j>^3)
            # Centered cosines so cone mean-shift doesn't swamp the cubic term
            xc_b = F.normalize(xb - xb.mean(dim=0, keepdim=True), p=2, dim=1)
            yc_b = F.normalize(yb - yb.mean(dim=0, keepdim=True), p=2, dim=1)
            pc_b = F.normalize(pred - pred.mean(dim=0, keepdim=True), p=2, dim=1)
            k3_t = (xc_b @ xc_b.t()) ** 3
            k3_s = (pc_b @ yc_b.t()) ** 3
            loss_cubic = F.mse_loss(k3_s, k3_t)
            loss_c3 = 0.25 * loss_cum + 1.0 * loss_cubic

        # Mild Cayley skew regularization to preserve stability
        reg = 1e-4 * (model.skew_raw ** 2).mean()
        loss = loss_vmf + (0.35 * loss_kl if use_kl else 0.0) + (0.25 * loss_c3 if use_third_order else 0.0) + reg

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        opt.step()
        sched.step()
        final_losses = {
            "vmf": float(loss_vmf.item()),
            "kl": float(loss_kl.item()),
            "c3": float(loss_c3.item()),
        }

    fit_ms = (time.perf_counter() - t0) * 1000.0
    model.eval()
    return {"model": model, "device": device, "fit_ms": round(fit_ms, 2), "final_losses": final_losses}


@torch.no_grad()
def apply_spherical_net(x: np.ndarray, bundle: Dict[str, Any], batch_size: int = 2048) -> np.ndarray:
    model: SphericalBridgeNet = bundle["model"]
    device = bundle["device"]
    arr = np.asarray(x, dtype=np.float32)
    outs = []
    for start in range(0, arr.shape[0], batch_size):
        xb = torch.tensor(arr[start : start + batch_size], dtype=torch.float32, device=device)
        xb = F.normalize(xb, p=2, dim=1)
        outs.append(model(xb).cpu().numpy())
    return np.vstack(outs).astype(np.float64)


# ---------------------------------------------------------------------------
# Quantile / Empirical-CDF Calibrated Map+Backfill & Spherical Distortion Order
# ---------------------------------------------------------------------------
def compute_spherical_distortion_order(
    d_new: np.ndarray,
    d_old: np.ndarray,
    mapped_docs: np.ndarray,
    *,
    anchor_pool_idx: np.ndarray,
    tau: float = 0.07,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rank every doc by its qrels-free spherical neighborhood distortion KL + angular residual.

    Higher score => this document's local angular neighborhood changed MOST under E_old -> E_new,
    making it the highest-priority document to re-embed / backfill.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pool = anchor_pool_idx[: min(len(anchor_pool_idx), 1024)]
    p_new = F.normalize(torch.tensor(d_new[pool], dtype=torch.float32, device=device), p=2, dim=1)
    p_old = F.normalize(torch.tensor(d_old[pool], dtype=torch.float32, device=device), p=2, dim=1)

    n_docs = d_new.shape[0]
    distortion = np.zeros(n_docs, dtype=np.float64)
    bs = 2048
    with torch.no_grad():
        for start in range(0, n_docs, bs):
            end = min(n_docs, start + bs)
            dn = F.normalize(torch.tensor(d_new[start:end], dtype=torch.float32, device=device), p=2, dim=1)
            do = F.normalize(torch.tensor(d_old[start:end], dtype=torch.float32, device=device), p=2, dim=1)
            dm = F.normalize(torch.tensor(mapped_docs[start:end], dtype=torch.float32, device=device), p=2, dim=1)
            # 1) Pointwise angular discrepancy on S^{d-1}
            ang_err = 1.0 - (dm * do).sum(dim=1)
            # 2) Spherical neighborhood KL against anchor reference pool
            sim_t = (dn @ p_new.t()) / tau
            sim_s = (dm @ p_old.t()) / tau
            top_t, top_idx = torch.topk(sim_t, k=min(64, p_new.size(0)), dim=1)
            top_s = torch.gather(sim_s, dim=1, index=top_idx)
            pt = F.softmax(top_t, dim=1)
            kl = (pt * (F.log_softmax(top_t, dim=1) - F.log_softmax(top_s, dim=1))).sum(dim=1)
            # Standardized combination (equal weight)
            score = ang_err + 0.25 * kl
            distortion[start:end] = score.cpu().numpy()

    order = np.argsort(-distortion, kind="mergesort")
    return order, distortion


def score_map_plus_backfill_quantile(
    queries_new: np.ndarray,
    queries_mapped: np.ndarray,
    docs_old: np.ndarray,
    docs_new: np.ndarray,
    backfill_indices: Sequence[int],
    query_ids: Sequence[str],
    doc_ids: Sequence[str],
    *,
    dev_queries_new: np.ndarray,
    dev_queries_mapped: np.ndarray,
    k: int = 10,
) -> Dict[str, List[str]]:
    """Quantile/CDF-calibrated score fusion between native backfilled docs and bridged docs.

    Eliminates heterogeneous-cone score miscalibration (e.g. BGE narrow cone vs MiniLM)
    strictly on DEVELOPMENT queries (qrels-free).
    """
    bf_idx = np.asarray(sorted(set(int(i) for i in backfill_indices)), dtype=np.int64)
    n_docs = docs_old.shape[0]
    bf_mask = np.zeros(n_docs, dtype=bool)
    bf_mask[bf_idx] = True

    d_old_n = _l2_normalize(docs_old)
    d_new_n = _l2_normalize(docs_new)
    q_new_n = _l2_normalize(queries_new)
    q_map_n = _l2_normalize(queries_mapped)

    # Fit quantile calibration strictly on DEV queries (sample up to 4096 docs for fast exact CDF)
    dev_qn = _l2_normalize(dev_queries_new)
    dev_qm = _l2_normalize(dev_queries_mapped)
    sub_docs = np.linspace(0, n_docs - 1, num=min(n_docs, 4096), dtype=np.int64)
    dev_native_vals = np.sort((dev_qn @ d_new_n[sub_docs].T).ravel())
    dev_mapped_vals = np.sort((dev_qm @ d_old_n[sub_docs].T).ravel())

    # Evaluate scores
    mapped_scores = q_map_n @ d_old_n.T
    native_bf_scores = q_new_n @ d_new_n[bf_idx].T

    # Monotone quantile-transfer: map native_bf_scores -> equivalent quantile in mapped_scores distribution
    ranks = np.searchsorted(dev_native_vals, native_bf_scores, side="left")
    quantiles = np.clip(ranks / float(max(dev_native_vals.size - 1, 1)), 0.0, 1.0)
    mapped_pos = np.clip((quantiles * (dev_mapped_vals.size - 1)).astype(np.int64), 0, dev_mapped_vals.size - 1)
    cal_bf_scores = dev_mapped_vals[mapped_pos]

    combined = mapped_scores.copy()
    combined[:, bf_idx] = cal_bf_scores

    k_eff = min(k, n_docs)
    rankings: Dict[str, List[str]] = {}
    for row_idx, qid in enumerate(query_ids):
        row = combined[row_idx]
        if n_docs > max(k_eff * 4, 64):
            cand = np.argpartition(-row, k_eff - 1)[: min(n_docs, max(k_eff * 4, 64))]
            ordered = sorted(cand.tolist(), key=lambda j: (-float(row[j]), doc_ids[j]))[:k_eff]
        else:
            ordered = sorted(range(n_docs), key=lambda j: (-float(row[j]), doc_ids[j]))[:k_eff]
        rankings[qid] = [doc_ids[j] for j in ordered]
    return rankings


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------
def run_spherical_ablation(
    *,
    output_dir: Path,
    cache_dir: Path,
    datasets: Sequence[str],
    seeds: Sequence[int],
    batch_size: int = 128,
) -> Dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be absent or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    singlehop_rows: List[Dict[str, Any]] = []
    multihop_rows: List[Dict[str, Any]] = []
    backfill_rows: List[Dict[str, Any]] = []
    factorial_cells: List[Dict[str, Any]] = []

    swaps = [
        ("E0_to_E1_samedim", "E0", "E1"),
        ("E0_to_E3_crossdim", "E0", "E3"),
    ]
    chain = ["E0", "E1", "E2", "E3"]

    for ds_name in datasets:
        print(f"[SPH] Loading dataset {ds_name}...", flush=True)
        doc_ids, doc_texts, query_ids, query_texts, qrels, qrels_sha = load_beir_dataset(ds_name)
        splits = split_query_ids(query_ids)
        qid_to_idx = {qid: i for i, qid in enumerate(query_ids)}
        dev_idx = [qid_to_idx[q] for q in splits["DEVELOPMENT"]]
        rep_idx = [qid_to_idx[q] for q in splits["REPORT"]]
        split_sha = sha256_json(splits)
        _, contract_sha = metric_contract_digest(qrels_sha, k=10, gain_rule="graded")

        enc: Dict[str, Dict[str, Any]] = {}
        for ek in chain:
            d_emb, q_emb, meta = encode_or_load_embeddings(
                dataset_name=ds_name,
                encoder_key=ek,
                encoder_name=ENCODERS[ek],
                doc_ids=doc_ids,
                doc_texts=doc_texts,
                query_ids=query_ids,
                query_texts=query_texts,
                cache_dir=cache_dir,
                batch_size=batch_size,
            )
            enc[ek] = {"docs": d_emb, "queries": q_emb, "meta": meta}

        d0, q0 = enc["E0"]["docs"], enc["E0"]["queries"]
        fastfill_order = compute_fastfill_backfill_order(q0[dev_idx], d0, top_m=50)
        # Rank position (0 = highest exposure) for hybrid exposure x distortion scoring
        exposure_rank = np.empty(len(doc_ids), dtype=np.float64)
        exposure_rank[fastfill_order] = np.linspace(1.0, 0.0, num=len(doc_ids))

        # ------------------------------------------------------------------
        # PART 1 & PART 3: Strictly Additive Single-Hop + Backfill Ladder
        # ------------------------------------------------------------------
        for swap_label, old_k, new_k in swaps:
            d_old, q_old = enc[old_k]["docs"], enc[old_k]["queries"]
            d_new, q_new = enc[new_k]["docs"], enc[new_k]["queries"]

            r_c2o, _ = score_dense_rankings(q_new[rep_idx], d_new, splits["REPORT"], doc_ids, k=10)
            pq_c2o = evaluate_rankings_mlr(r_c2o, qrels, splits["REPORT"], k=10)
            r_old, _ = score_dense_rankings(q_old[rep_idx], d_old, splits["REPORT"], doc_ids, k=10)
            pq_old = evaluate_rankings_mlr(r_old, qrels, splits["REPORT"], k=10)

            for seed in seeds:
                print(f"[SPH-S1] {ds_name} | {swap_label} | seed={seed}...", flush=True)
                anchor_idx = select_qrels_independent_anchors(doc_ids, anchor_frac=0.50, seed=seed)

                # A0: Uncentered Procrustes (Night-1)
                w_a0 = fit_closed_form_svd_procrustes(d_new, d_old, fit_idx=anchor_idx)
                q_rep_a0 = q_new[rep_idx] @ w_a0

                # A1: Closed-form Ridge tuned on DEV (Night-1)
                best_lam = DEFAULT_RIDGE_LAMBDAS[0]
                best_dev = -1.0
                w_a1 = None
                for lam in DEFAULT_RIDGE_LAMBDAS:
                    wc = fit_closed_form_ridge(d_new, d_old, fit_idx=anchor_idx, regularization=lam)
                    rd, _ = score_dense_rankings(q_new[dev_idx] @ wc, d_old, splits["DEVELOPMENT"], doc_ids, k=10)
                    sc = summarize_per_query_metrics(evaluate_rankings_mlr(rd, qrels, splits["DEVELOPMENT"], k=10))["mean_ndcg_at_10"]
                    if sc > best_dev:
                        best_dev = sc
                        best_lam = lam
                        w_a1 = wc
                assert w_a1 is not None
                q_dev_a1 = q_new[dev_idx] @ w_a1
                q_rep_a1 = q_new[rep_idx] @ w_a1

                # A2: +1st-order Centered Spherical Procrustes
                m_a2 = fit_centered_procrustes(d_new, d_old, anchor_idx)
                q_dev_a2 = apply_centered_procrustes(q_new[dev_idx], m_a2)
                q_rep_a2 = apply_centered_procrustes(q_new[rep_idx], m_a2)

                # A3: +2nd-order Shrinkage-Whitened (CORAL) Centered Procrustes (shrink tuned on DEV)
                best_sh = 0.20
                best_a3_dev = -1.0
                m_a3 = None
                for sh in (0.05, 0.20, 0.50):
                    mc = fit_whitened_centered_procrustes(d_new, d_old, anchor_idx, shrink=sh)
                    rd, _ = score_dense_rankings(
                        apply_affine_spherical(q_new[dev_idx], mc), d_old, splits["DEVELOPMENT"], doc_ids, k=10
                    )
                    sc = summarize_per_query_metrics(evaluate_rankings_mlr(rd, qrels, splits["DEVELOPMENT"], k=10))["mean_ndcg_at_10"]
                    if sc > best_a3_dev:
                        best_a3_dev = sc
                        best_sh = sh
                        m_a3 = mc
                assert m_a3 is not None
                q_rep_a3 = apply_affine_spherical(q_new[rep_idx], m_a3)

                # A4: +Explicit 3rd-order Projected Cumulants & Cubic Kernel (NO KL; orthogonal + residual)
                b_a4 = fit_spherical_distillation(
                    d_new, d_old, anchor_idx, m_a2,
                    use_kl=False, use_third_order=True, use_residual=True, seed=seed,
                )
                q_rep_a4 = apply_spherical_net(q_new[rep_idx], b_a4)

                # A5: +Relational Spherical KL Distillation (NO explicit 3rd-order)
                b_a5 = fit_spherical_distillation(
                    d_new, d_old, anchor_idx, m_a2,
                    use_kl=True, use_third_order=False, use_residual=True, seed=seed,
                )
                q_dev_a5 = apply_spherical_net(q_new[dev_idx], b_a5)
                q_rep_a5 = apply_spherical_net(q_new[rep_idx], b_a5)

                # A6: +Spherical KL + Explicit 3rd-Order Cumulants/Cubic (2x2 full cell)
                b_a6 = fit_spherical_distillation(
                    d_new, d_old, anchor_idx, m_a2,
                    use_kl=True, use_third_order=True, use_residual=True, seed=seed,
                )
                q_dev_a6 = apply_spherical_net(q_new[dev_idx], b_a6)
                q_rep_a6 = apply_spherical_net(q_new[rep_idx], b_a6)

                # Evaluate all zero-write arms on REPORT
                arm_pq: Dict[str, Dict[str, Dict[str, float]]] = {
                    "old_encoder_servable": pq_old,
                    "full_reembed_c2o": pq_c2o,
                }
                for name, qmat in (
                    ("A0_uncentered_procrustes", q_rep_a0),
                    ("A1_ridge_baseline", q_rep_a1),
                    ("A2_centered_procrustes", q_rep_a2),
                    ("A3_whitened_centered_procrustes", q_rep_a3),
                    ("A4_centered_plus_3rd_order", q_rep_a4),
                    ("A5_spherical_kl_distill", q_rep_a5),
                    ("A6_spherical_kl_plus_3rd_order", q_rep_a6),
                ):
                    rk, _ = score_dense_rankings(qmat, d_old, splits["REPORT"], doc_ids, k=10)
                    arm_pq[name] = evaluate_rankings_mlr(rk, qrels, splits["REPORT"], k=10)

                # A1+CSLS, A2+CSLS, A5+CSLS, A6+CSLS (check additivity of #355 CSLS with spherical distillation)
                for name, q_rep_m, q_dev_m in (
                    ("A1b_ridge_csls", q_rep_a1, q_dev_a1),
                    ("A2b_centered_proc_csls", q_rep_a2, q_dev_a2),
                    ("A7a_spherical_kl_csls", q_rep_a5, q_dev_a5),
                    ("A7b_spherical_kl_3rd_csls", q_rep_a6, q_dev_a6),
                ):
                    rk, _ = score_dense_rankings(
                        q_rep_m, d_old, splits["REPORT"], doc_ids, k=10, csls_k=10, csls_reference_queries=q_dev_m
                    )
                    arm_pq[name] = evaluate_rankings_mlr(rk, qrels, splits["REPORT"], k=10)

                # Record strictly additive rows with paired bootstrap deltas vs A1_ridge_baseline and previous rung
                chain_order = [
                    ("A0_uncentered_procrustes", None),
                    ("A1_ridge_baseline", "A0_uncentered_procrustes"),
                    ("A1b_ridge_csls", "A1_ridge_baseline"),
                    ("A2_centered_procrustes", "A0_uncentered_procrustes"),
                    ("A2b_centered_proc_csls", "A2_centered_procrustes"),
                    ("A3_whitened_centered_procrustes", "A2_centered_procrustes"),
                    ("A4_centered_plus_3rd_order", "A2_centered_procrustes"),
                    ("A5_spherical_kl_distill", "A2_centered_procrustes"),
                    ("A6_spherical_kl_plus_3rd_order", "A5_spherical_kl_distill"),
                    ("A7a_spherical_kl_csls", "A5_spherical_kl_distill"),
                    ("A7b_spherical_kl_3rd_csls", "A6_spherical_kl_plus_3rd_order"),
                    ("old_encoder_servable", None),
                    ("full_reembed_c2o", None),
                ]
                for arm_name, prev_arm in chain_order:
                    summ = summarize_per_query_metrics(arm_pq[arm_name])
                    vs_ridge = paired_bootstrap_delta(
                        _ndcg_map(arm_pq[arm_name]), _ndcg_map(arm_pq["A1_ridge_baseline"]), n_bootstrap=5000, seed=seed
                    )
                    vs_prev = (
                        paired_bootstrap_delta(
                            _ndcg_map(arm_pq[arm_name]), _ndcg_map(arm_pq[prev_arm]), n_bootstrap=5000, seed=seed
                        )
                        if prev_arm is not None
                        else {"mean_delta": 0.0, "lcb_95": 0.0, "ucb_95": 0.0}
                    )
                    singlehop_rows.append(
                        {
                            "dataset": ds_name,
                            "swap": swap_label,
                            "seed": int(seed),
                            "arm": arm_name,
                            "prev_rung": prev_arm or "",
                            "report_ndcg_at_10": round(summ["mean_ndcg_at_10"], 6),
                            "report_mrr": round(summ["mean_mrr"], 6),
                            "report_recall_at_10": round(summ["mean_recall_at_10"], 6),
                            "delta_vs_prev": round(vs_prev["mean_delta"], 6),
                            "delta_vs_prev_lcb95": round(vs_prev["lcb_95"], 6),
                            "delta_vs_ridge": round(vs_ridge["mean_delta"], 6),
                            "delta_vs_ridge_lcb95": round(vs_ridge["lcb_95"], 6),
                            "ridge_lambda": best_lam,
                            "whitening_shrink": best_sh,
                            "split_sha256": split_sha,
                            "metric_contract_sha256": contract_sha,
                        }
                    )

                # Record exact 2x2 factorial interaction: interaction = (A6 - A5) - (A4 - A2)
                s_a2 = summarize_per_query_metrics(arm_pq["A2_centered_procrustes"])["mean_ndcg_at_10"]
                s_a4 = summarize_per_query_metrics(arm_pq["A4_centered_plus_3rd_order"])["mean_ndcg_at_10"]
                s_a5 = summarize_per_query_metrics(arm_pq["A5_spherical_kl_distill"])["mean_ndcg_at_10"]
                s_a6 = summarize_per_query_metrics(arm_pq["A6_spherical_kl_plus_3rd_order"])["mean_ndcg_at_10"]
                factorial_cells.append(
                    {
                        "dataset": ds_name,
                        "swap": swap_label,
                        "seed": int(seed),
                        "A2_base_1st_2nd": round(s_a2, 6),
                        "A4_3rd_only": round(s_a4, 6),
                        "A5_kl_only": round(s_a5, 6),
                        "A6_kl_plus_3rd": round(s_a6, 6),
                        "main_effect_3rd_no_kl": round(s_a4 - s_a2, 6),
                        "main_effect_kl_no_3rd": round(s_a5 - s_a2, 6),
                        "marginal_3rd_over_kl": round(s_a6 - s_a5, 6),
                        "interaction_synergy": round((s_a6 - s_a5) - (s_a4 - s_a2), 6),
                    }
                )

                # ------------------------------------------------------------------
                # PART 3: Strictly Additive Backfill & Calibration Ladder (10% & 20%)
                # ------------------------------------------------------------------
                mapped_all_docs = apply_spherical_net(d_new, b_a5)
                dist_order, _ = compute_spherical_distortion_order(
                    d_new, d_old, mapped_all_docs, anchor_pool_idx=anchor_idx
                )
                dist_rank = np.empty(len(doc_ids), dtype=np.float64)
                dist_rank[dist_order] = np.linspace(1.0, 0.0, num=len(doc_ids))
                hybrid_order = np.argsort(-(0.5 * exposure_rank + 0.5 * dist_rank), kind="mergesort")

                for b_frac in (0.10, 0.20):
                    n_bf = max(1, int(round(len(doc_ids) * b_frac)))
                    bf_ff = fastfill_order[:n_bf]
                    bf_dist = dist_order[:n_bf]
                    bf_hyb = hybrid_order[:n_bf]

                    # B0: Night-1 Ridge + z-score calibration + FastFill
                    r_b0 = score_map_plus_backfill(
                        q_new[rep_idx], d_old, d_new, w_a1, bf_ff, splits["REPORT"], doc_ids,
                        dev_queries_new=q_new[dev_idx], k=10, calibrate=True,
                    )
                    # B1: Spherical KL bridge (A5) + Quantile CDF calibration + FastFill
                    r_b1 = score_map_plus_backfill_quantile(
                        q_new[rep_idx], q_rep_a5, d_old, d_new, bf_ff, splits["REPORT"], doc_ids,
                        dev_queries_new=q_new[dev_idx], dev_queries_mapped=q_dev_a5, k=10,
                    )
                    # B2: Spherical KL bridge (A5) + Quantile CDF calibration + Spherical Distortion Order
                    r_b2 = score_map_plus_backfill_quantile(
                        q_new[rep_idx], q_rep_a5, d_old, d_new, bf_dist, splits["REPORT"], doc_ids,
                        dev_queries_new=q_new[dev_idx], dev_queries_mapped=q_dev_a5, k=10,
                    )
                    # B3: Spherical KL bridge (A5) + Quantile CDF calibration + Hybrid (Exposure x Distortion)
                    r_b3 = score_map_plus_backfill_quantile(
                        q_new[rep_idx], q_rep_a5, d_old, d_new, bf_hyb, splits["REPORT"], doc_ids,
                        dev_queries_new=q_new[dev_idx], dev_queries_mapped=q_dev_a5, k=10,
                    )
                    bf_pqs = {
                        "B0_night1_ridge_zscore_fastfill": evaluate_rankings_mlr(r_b0, qrels, splits["REPORT"], k=10),
                        "B1_sph_kl_quantile_fastfill": evaluate_rankings_mlr(r_b1, qrels, splits["REPORT"], k=10),
                        "B2_sph_kl_quantile_distortion": evaluate_rankings_mlr(r_b2, qrels, splits["REPORT"], k=10),
                        "B3_sph_kl_quantile_exposure_x_dist": evaluate_rankings_mlr(r_b3, qrels, splits["REPORT"], k=10),
                    }
                    for bname, prev_b in (
                        ("B0_night1_ridge_zscore_fastfill", None),
                        ("B1_sph_kl_quantile_fastfill", "B0_night1_ridge_zscore_fastfill"),
                        ("B2_sph_kl_quantile_distortion", "B1_sph_kl_quantile_fastfill"),
                        ("B3_sph_kl_quantile_exposure_x_dist", "B1_sph_kl_quantile_fastfill"),
                    ):
                        sm = summarize_per_query_metrics(bf_pqs[bname])
                        dv_b0 = paired_bootstrap_delta(
                            _ndcg_map(bf_pqs[bname]),
                            _ndcg_map(bf_pqs["B0_night1_ridge_zscore_fastfill"]),
                            n_bootstrap=5000, seed=seed,
                        )
                        dv_prev = (
                            paired_bootstrap_delta(
                                _ndcg_map(bf_pqs[bname]), _ndcg_map(bf_pqs[prev_b]), n_bootstrap=5000, seed=seed
                            )
                            if prev_b
                            else {"mean_delta": 0.0, "lcb_95": 0.0}
                        )
                        backfill_rows.append(
                            {
                                "dataset": ds_name,
                                "swap": swap_label,
                                "budget": b_frac,
                                "seed": int(seed),
                                "arm": bname,
                                "report_ndcg_at_10": round(sm["mean_ndcg_at_10"], 6),
                                "delta_vs_prev": round(dv_prev["mean_delta"], 6),
                                "delta_vs_prev_lcb95": round(dv_prev["lcb_95"], 6),
                                "delta_vs_b0": round(dv_b0["mean_delta"], 6),
                                "delta_vs_b0_lcb95": round(dv_b0["lcb_95"], 6),
                            }
                        )

        # ------------------------------------------------------------------
        # PART 2: Multi-Hop Composition Chain (E0 -> E1 -> E2 -> E3)
        # ------------------------------------------------------------------
        for n_anchors in (500, 2000):
            eff_anchors = min(n_anchors, len(doc_ids) // 2)
            for seed in seeds:
                print(f"[SPH-S2] Multi-hop {ds_name} | anchors={eff_anchors} | seed={seed}...", flush=True)
                hop_uncent: List[np.ndarray] = []
                hop_ridge: List[np.ndarray] = []
                hop_cent: List[Dict[str, np.ndarray]] = []
                hop_kl: List[Dict[str, Any]] = []
                hop_kl3: List[Dict[str, Any]] = []

                for step in range(1, len(chain)):
                    sk, tk = chain[step], chain[step - 1]
                    a_idx = select_qrels_independent_anchors(
                        doc_ids, anchor_frac=1.0, seed=seed + step * 97, max_anchors=eff_anchors
                    )
                    wp = fit_closed_form_svd_procrustes(enc[sk]["docs"], enc[tk]["docs"], fit_idx=a_idx)
                    # Tune ridge lambda on DEV
                    best_wr = None
                    best_sc = -1.0
                    for lam in DEFAULT_RIDGE_LAMBDAS:
                        wc = fit_closed_form_ridge(enc[sk]["docs"], enc[tk]["docs"], fit_idx=a_idx, regularization=lam)
                        rd, _ = score_dense_rankings(
                            enc[sk]["queries"][dev_idx] @ wc, enc[tk]["docs"], splits["DEVELOPMENT"], doc_ids, k=10
                        )
                        sc = summarize_per_query_metrics(evaluate_rankings_mlr(rd, qrels, splits["DEVELOPMENT"], k=10))["mean_ndcg_at_10"]
                        if sc > best_sc:
                            best_sc = sc
                            best_wr = wc
                    assert best_wr is not None
                    mc = fit_centered_procrustes(enc[sk]["docs"], enc[tk]["docs"], a_idx)
                    bkl = fit_spherical_distillation(
                        enc[sk]["docs"], enc[tk]["docs"], a_idx, mc,
                        use_kl=True, use_third_order=False, use_residual=True, steps=180, seed=seed + step,
                    )
                    bkl3 = fit_spherical_distillation(
                        enc[sk]["docs"], enc[tk]["docs"], a_idx, mc,
                        use_kl=True, use_third_order=True, use_residual=True, steps=180, seed=seed + step,
                    )
                    hop_uncent.append(wp)
                    hop_ridge.append(best_wr)
                    hop_cent.append(mc)
                    hop_kl.append(bkl)
                    hop_kl3.append(bkl3)

                for hop_m in range(1, len(chain)):
                    sk = chain[hop_m]
                    q_rep_src = enc[sk]["queries"][rep_idx]

                    # Compose backward from E_{hop_m} -> ... -> E_0
                    q_u = q_rep_src.copy()
                    q_r = q_rep_src.copy()
                    q_c = q_rep_src.copy()
                    q_kl = q_rep_src.copy()
                    q_kl3 = q_rep_src.copy()
                    w_r_comp = None
                    w_c_comp = None
                    for step_idx in range(hop_m - 1, -1, -1):
                        q_u = q_u @ hop_uncent[step_idx]
                        q_r = q_r @ hop_ridge[step_idx]
                        q_c = apply_centered_procrustes(q_c, hop_cent[step_idx])
                        q_kl = apply_spherical_net(q_kl, hop_kl[step_idx])
                        q_kl3 = apply_spherical_net(q_kl3, hop_kl3[step_idx])
                        w_r_comp = hop_ridge[step_idx] if w_r_comp is None else (w_r_comp @ hop_ridge[step_idx])
                        w_c_comp = hop_cent[step_idx]["Q"] if w_c_comp is None else (w_c_comp @ hop_cent[step_idx]["Q"])

                    sv_r = np.linalg.svd(w_r_comp, compute_uv=False)
                    cond_r = float(sv_r[0] / max(sv_r[-1], 1e-12))
                    sv_c = np.linalg.svd(w_c_comp, compute_uv=False)
                    cond_c = float(sv_c[0] / max(sv_c[-1], 1e-12))

                    for mname, qmat, cnum in (
                        ("composed_uncentered_procrustes", q_u, 1.0),
                        ("composed_ridge", q_r, cond_r),
                        ("composed_centered_procrustes", q_c, cond_c),
                        ("composed_spherical_kl", q_kl, None),
                        ("composed_spherical_kl_plus_3rd", q_kl3, None),
                    ):
                        rk, _ = score_dense_rankings(qmat, d0, splits["REPORT"], doc_ids, k=10)
                        sm = summarize_per_query_metrics(evaluate_rankings_mlr(rk, qrels, splits["REPORT"], k=10))
                        multihop_rows.append(
                            {
                                "dataset": ds_name,
                                "anchor_budget": eff_anchors,
                                "hop_m": hop_m,
                                "source_encoder": sk,
                                "seed": int(seed),
                                "method": mname,
                                "report_ndcg_at_10": round(sm["mean_ndcg_at_10"], 6),
                                "cond_number": round(cnum, 4) if cnum is not None else None,
                            }
                        )

    # Write CSVs
    def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
        if not rows:
            return
        with path.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    _write_csv(output_dir / "spherical_singlehop_additive.csv", singlehop_rows)
    _write_csv(output_dir / "spherical_multihop_additive.csv", multihop_rows)
    _write_csv(output_dir / "spherical_backfill_additive.csv", backfill_rows)
    write_json_atomic(output_dir / "spherical_2x2_factorial.json", factorial_cells)

    # Build concise Markdown summary
    md = [
        "# Spherical Distillation & 3rd-Order Cumulant Strictly Additive Ablation (`mlr-v1`)",
        "",
        f"- **Git SHA:** `{current_git_sha(REPO_ROOT)}`",
        f"- **Wall time:** `{round(time.perf_counter() - t_start, 2)}s`",
        "",
        "## 1. Single-Hop Strictly Additive Zero-Write Ladder (REPORT nDCG@10, Mean Across 3 Seeds)",
        "",
        "| Dataset | Swap | A0 Uncent Proc | A1 Ridge | A1b Ridge+CSLS | A2 Centered Proc (+1st) | A3 Whitened (+2nd) | A4 +3rd Cumulant | A5 Sph-KL | A6 Sph-KL+3rd | A7a KL+CSLS | A7b KL+3rd+CSLS | Native E0 | Full C2O |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    sh_grp: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for r in singlehop_rows:
        sh_grp[(r["dataset"], r["swap"], r["arm"])].append(float(r["report_ndcg_at_10"]))
    arm_cols = [
        "A0_uncentered_procrustes", "A1_ridge_baseline", "A1b_ridge_csls",
        "A2_centered_procrustes", "A3_whitened_centered_procrustes",
        "A4_centered_plus_3rd_order", "A5_spherical_kl_distill",
        "A6_spherical_kl_plus_3rd_order", "A7a_spherical_kl_csls",
        "A7b_spherical_kl_3rd_csls", "old_encoder_servable", "full_reembed_c2o",
    ]
    for ds in datasets:
        for sw, _, _ in swaps:
            vals = [f"{sum(sh_grp[(ds, sw, a)]) / len(sh_grp[(ds, sw, a)]):.4f}" for a in arm_cols]
            md.append(f"| {ds} | {sw} | " + " | ".join(vals) + " |")

    md += [
        "",
        "## 2. Exact 2x2 Factorial: {No-KL, KL} x {No-3rd-Order, 3rd-Order} (Mean Across 3 Seeds)",
        "",
        "| Dataset | Swap | Base (A2) | +3rd Only (A4−A2) | +KL Only (A5−A2) | +3rd over KL (A6−A5) | Interaction Synergy |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for ds in datasets:
        for sw, _, _ in swaps:
            sub = [c for c in factorial_cells if c["dataset"] == ds and c["swap"] == sw]
            a2m = sum(c["A2_base_1st_2nd"] for c in sub) / len(sub)
            e3 = sum(c["main_effect_3rd_no_kl"] for c in sub) / len(sub)
            ekl = sum(c["main_effect_kl_no_3rd"] for c in sub) / len(sub)
            m3 = sum(c["marginal_3rd_over_kl"] for c in sub) / len(sub)
            syn = sum(c["interaction_synergy"] for c in sub) / len(sub)
            md.append(f"| {ds} | {sw} | {a2m:.4f} | {e3:+.4f} | {ekl:+.4f} | {m3:+.4f} | {syn:+.4f} |")

    md += [
        "",
        "## 3. Multi-Hop Composition Chain (E0 -> E1 -> E2 -> E3, Mean Across Anchor Budgets & Seeds)",
        "",
        "| Dataset | Hop | Comp Uncent Proc | Comp Ridge | Comp Centered Proc (+1st) | Comp Sph-KL | Comp Sph-KL+3rd |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    mh_grp: Dict[Tuple[str, int, str], List[float]] = defaultdict(list)
    for r in multihop_rows:
        mh_grp[(r["dataset"], int(r["hop_m"]), r["method"])].append(float(r["report_ndcg_at_10"]))
    mh_cols = [
        "composed_uncentered_procrustes", "composed_ridge",
        "composed_centered_procrustes", "composed_spherical_kl",
        "composed_spherical_kl_plus_3rd",
    ]
    for ds in datasets:
        for hop_m in (1, 2, 3):
            vals = [f"{sum(mh_grp[(ds, hop_m, m)]) / len(mh_grp[(ds, hop_m, m)]):.4f}" for m in mh_cols]
            md.append(f"| {ds} | {hop_m} | " + " | ".join(vals) + " |")

    md += [
        "",
        "## 4. Selective Backfill & Quantile Calibration Additive Ladder (REPORT nDCG@10)",
        "",
        "| Dataset | Swap | Budget | B0 Night-1 Z-Score+FastFill | B1 Sph-KL+Quantile+FastFill | B2 +Sph-Distortion Order | B3 +Exposure x Distortion |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    bf_grp: Dict[Tuple[str, str, float, str], List[float]] = defaultdict(list)
    for r in backfill_rows:
        bf_grp[(r["dataset"], r["swap"], float(r["budget"]), r["arm"])].append(float(r["report_ndcg_at_10"]))
    bf_cols = [
        "B0_night1_ridge_zscore_fastfill",
        "B1_sph_kl_quantile_fastfill",
        "B2_sph_kl_quantile_distortion",
        "B3_sph_kl_quantile_exposure_x_dist",
    ]
    for ds in datasets:
        for sw, _, _ in swaps:
            for b_frac in (0.10, 0.20):
                vals = [f"{sum(bf_grp[(ds, sw, b_frac, b)]) / len(bf_grp[(ds, sw, b_frac, b)]):.4f}" for b in bf_cols]
                md.append(f"| {ds} | {sw} | {int(b_frac * 100)}% | " + " | ".join(vals) + " |")

    summary_path = output_dir / "spherical_ablation_summary.md"
    summary_path.write_text("\n".join(md) + "\n", encoding="utf-8")

    manifest = {
        "job": "night1_spherical_ablation",
        "git_sha": current_git_sha(REPO_ROOT),
        "datasets": list(datasets),
        "seeds": [int(s) for s in seeds],
        "wall_seconds": round(time.perf_counter() - t_start, 2),
        "telemetry": process_telemetry(),
    }
    manifest_sha = write_json_atomic(output_dir / "spherical_ablation_manifest.json", manifest)
    write_json_atomic(
        output_dir / "report_receipt.json",
        {"job": "night1_spherical_ablation", "status": "COMPLETED", "manifest_sha256": manifest_sha},
    )
    print(f"[SPH] COMPLETE in {manifest['wall_seconds']}s -> {output_dir}", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Spherical Distillation & 3rd-Order Cumulant Additive Ablation")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/chelated_night1_emb_cache"))
    parser.add_argument("--datasets", nargs="+", default=["SciFact", "NFCorpus", "FiQA2018"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 137, 2718])
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    run_spherical_ablation(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        datasets=args.datasets,
        seeds=args.seeds,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
