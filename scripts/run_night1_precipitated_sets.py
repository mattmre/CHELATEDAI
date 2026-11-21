"""Precipitated Random / Non-Distributional vs. Bounded Manifold-Precipitated Anchor Sets (mlr-v1).

Head-to-head comparison on SciFact, NFCorpus, and FiQA2018 across anchor budgets {100, 250, 500, 1000}
and 3 seeds [42, 137, 2718]:

Control:
  P0_uniform_iid                 : Standard uniform i.i.d. draw from corpus distribution

User-Proposed Non-Distributional / Precipitated Random Sets:
  P1_user_synthetic_random_text  : Synthetic uniform-random lexical probes encoded through both E_old & E_new
  P2_user_tail_precipitated      : Extreme tail/boundary docs farthest from spherical centroid (inverted density)
  P3_user_mixed_50_50_probes     : 50% corpus anchors + 50% synthetic non-distributional random lexical probes

Recommended Bounded Manifold-Precipitated Sets:
  P4_rec_farthest_point_kcenter  : Outlier-trimmed farthest-point (k-center) spherical precipitation
  P5_rec_leverage_d_optimal      : SVD statistical leverage-score (D-optimal) precipitation (98% capped)
  P6_rec_kmeanspp_spherical      : Spherical k-means++ (D^2 angular seeding) multimodal lobe precipitation
  P7_rec_nystrom_landmark_kernel : Distribution-free non-parametric landmark-kernel bridge over P6 landmarks

Logs REPORT nDCG@10, paired bootstrap delta vs P0_uniform_iid, tail-query nDCG@10 (top-25% farthest queries),
anchor condition number, effective rank, and 3-hop (E3->E2->E1->E0) composition stability.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from m1_baseline_ladder import (  # noqa: E402
    DEFAULT_RIDGE_LAMBDAS,
    evaluate_rankings_mlr,
    score_dense_rankings,
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

_WORD_RE = re.compile(r"[a-zA-Z]{3,}")


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(norms > 1e-12, norms, 1.0)


def _ndcg_map(per_q: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    return {qid: float(vals["ndcg_at_10"]) for qid, vals in per_q.items()}


def _anchor_geometry_stats(x: np.ndarray) -> Tuple[float, float]:
    """Return (condition_number, Shannon effective_rank) of centered anchor matrix."""
    xc = np.asarray(x, dtype=np.float64) - np.asarray(x, dtype=np.float64).mean(axis=0, keepdims=True)
    sv = np.linalg.svd(xc, compute_uv=False)
    sv_pos = sv[sv > 1e-12]
    if sv_pos.size == 0:
        return float("inf"), 0.0
    cond = float(sv_pos[0] / sv_pos[-1])
    p = (sv_pos ** 2) / float(np.sum(sv_pos ** 2))
    eff_rank = float(np.exp(-np.sum(p * np.log(np.maximum(p, 1e-15)))))
    return cond, eff_rank


def fit_centered_procrustes_xy(xs: np.ndarray, yt: np.ndarray) -> Dict[str, np.ndarray]:
    xs64 = np.asarray(xs, dtype=np.float64)
    yt64 = np.asarray(yt, dtype=np.float64)
    mu_x = xs64.mean(axis=0, keepdims=True)
    mu_y = yt64.mean(axis=0, keepdims=True)
    u, _, vt = np.linalg.svd((xs64 - mu_x).T @ (yt64 - mu_y), full_matrices=False)
    return {"mu_x": mu_x, "mu_y": mu_y, "Q": u @ vt}


def apply_centered_procrustes_xy(x: np.ndarray, model: Dict[str, np.ndarray]) -> np.ndarray:
    return _l2_normalize((np.asarray(x, dtype=np.float64) - model["mu_x"]) @ model["Q"] + model["mu_y"])


def fit_centered_ridge_xy(xs: np.ndarray, yt: np.ndarray, lam: float) -> Dict[str, np.ndarray]:
    xs64 = np.asarray(xs, dtype=np.float64)
    yt64 = np.asarray(yt, dtype=np.float64)
    mu_x = xs64.mean(axis=0, keepdims=True)
    mu_y = yt64.mean(axis=0, keepdims=True)
    xc = xs64 - mu_x
    yc = yt64 - mu_y
    d = xc.shape[1]
    w = np.linalg.solve(xc.T @ xc + float(lam) * np.eye(d, dtype=np.float64), xc.T @ yc)
    return {"mu_x": mu_x, "mu_y": mu_y, "W": w}


def apply_centered_ridge_xy(x: np.ndarray, model: Dict[str, np.ndarray]) -> np.ndarray:
    return _l2_normalize((np.asarray(x, dtype=np.float64) - model["mu_x"]) @ model["W"] + model["mu_y"])


def apply_nystrom_landmark_kernel(
    q_new: np.ndarray,
    land_new: np.ndarray,
    land_old: np.ndarray,
    *,
    tau: float = 0.07,
) -> np.ndarray:
    """Distribution-free non-parametric landmark-kernel bridge on S^{d-1}.

    Represents each query purely by its spherical kernel weights over precipitated landmarks
    in E_new, then reconstructs the direction directly from the corresponding E_old landmarks
    (via kernel ridge / Nystrom barycentric weights — zero unimodal Gaussian assumption).
    """
    qn = _l2_normalize(q_new)
    ln = _l2_normalize(land_new)
    lo = _l2_normalize(land_old)
    # Center landmarks slightly to prevent narrow-cone collapse in softmax weights
    ln_c = _l2_normalize(ln - ln.mean(axis=0, keepdims=True))
    qn_c = _l2_normalize(qn - ln.mean(axis=0, keepdims=True))
    # Solve exact Nystrom kernel weights on landmarks: (K + reg*I) alpha = lo
    k_ll = np.exp((ln_c @ ln_c.T) / tau)
    k_ll /= np.maximum(k_ll.sum(axis=1, keepdims=True), 1e-12)
    alpha = np.linalg.solve(k_ll + 1e-2 * np.eye(k_ll.shape[0], dtype=np.float64), lo)
    k_ql = np.exp((qn_c @ ln_c.T) / tau)
    k_ql /= np.maximum(k_ql.sum(axis=1, keepdims=True), 1e-12)
    return _l2_normalize(k_ql @ alpha)


# ---------------------------------------------------------------------------
# Anchor / Precipitated Set Selectors
# ---------------------------------------------------------------------------
def build_synthetic_random_probes(
    doc_texts: Sequence[str],
    n_probes: int,
    seed: int,
    words_per_probe: int = 24,
) -> List[str]:
    """Generate strictly non-distributional uniform-random lexical probes from corpus vocabulary."""
    counts: Counter[str] = Counter()
    for text in doc_texts[: min(len(doc_texts), 15000)]:
        counts.update(w.lower() for w in _WORD_RE.findall(text))
    vocab = sorted(w for w, c in counts.items() if c >= 2)
    if len(vocab) < 50:
        vocab = sorted(counts.keys()) or ["alpha", "beta", "gamma", "delta"]
    rng = np.random.default_rng(seed + 4242)
    probes: List[str] = []
    for _ in range(n_probes):
        idx = rng.integers(0, len(vocab), size=words_per_probe)
        probes.append(" ".join(vocab[int(i)] for i in idx))
    return probes


def encode_synthetic_probes(
    *,
    dataset_name: str,
    encoder_key: str,
    encoder_name: str,
    probe_texts: Sequence[str],
    cache_dir: Path,
    batch_size: int = 128,
) -> np.ndarray:
    """Encode synthetic random probes with disk caching."""
    probe_ids = [f"syn_{i}" for i in range(len(probe_texts))]
    embs, _, _ = encode_or_load_embeddings(
        dataset_name=f"{dataset_name}_SYNPROBES",
        encoder_key=encoder_key,
        encoder_name=encoder_name,
        doc_ids=probe_ids,
        doc_texts=probe_texts,
        query_ids=["q0"],
        query_texts=["placeholder"],
        cache_dir=cache_dir,
        batch_size=batch_size,
    )
    return embs


def select_uniform_iid(n_docs: int, n_select: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_docs, size=min(n_select, n_docs), replace=False))


def select_tail_precipitated(vecs: np.ndarray, n_select: int) -> np.ndarray:
    """User-proposed non-distributional tail set: points farthest from spherical centroid."""
    vn = _l2_normalize(vecs)
    centroid = _l2_normalize(vn.mean(axis=0, keepdims=True))[0]
    cos_to_center = vn @ centroid
    # Lowest cosine => farthest from dense centroid
    order = np.argsort(cos_to_center, kind="mergesort")
    return np.sort(order[: min(n_select, len(order))])


def select_farthest_point_kcenter(vecs: np.ndarray, n_select: int, seed: int) -> np.ndarray:
    """Recommended bounded k-center / farthest-point precipitation on S^{d-1} (2% noise trim)."""
    vn = _l2_normalize(vecs)
    n = vn.shape[0]
    k = min(n_select, n)
    # Trim extreme 2% outlier tail to avoid garbage stubs
    centroid = _l2_normalize(vn.mean(axis=0, keepdims=True))[0]
    cos_c = vn @ centroid
    cut = float(np.quantile(cos_c, 0.02))
    eligible = np.where(cos_c >= cut)[0]
    if eligible.size < k:
        eligible = np.arange(n)

    # For very large corpora (FiQA 57k), run exact k-center over a deterministic 8192-doc candidate pool
    rng = np.random.default_rng(seed + 101)
    if eligible.size > 8192:
        pool = np.sort(rng.choice(eligible, size=8192, replace=False))
    else:
        pool = eligible

    vp = vn[pool]
    chosen_local = [int(rng.integers(0, vp.shape[0]))]
    # Track max cosine similarity to already-chosen set (minimize max sim = maximize min angular distance)
    max_sim = vp @ vp[chosen_local[0]]
    max_sim[chosen_local[0]] = 2.0
    for _ in range(1, k):
        nxt = int(np.argmin(max_sim))
        chosen_local.append(nxt)
        sims = vp @ vp[nxt]
        max_sim = np.maximum(max_sim, sims)
        max_sim[nxt] = 2.0
    return np.sort(pool[np.asarray(chosen_local, dtype=np.int64)])


def select_leverage_d_optimal(vecs: np.ndarray, n_select: int, seed: int) -> np.ndarray:
    """Recommended statistical leverage-score (D-optimal) precipitation (98% capped)."""
    vn = _l2_normalize(vecs)
    n = vn.shape[0]
    k = min(n_select, n)
    vc = vn - vn.mean(axis=0, keepdims=True)
    # Fast exact leverage scores via thin SVD of centered covariance: l_i = ||vc_i V S^{-1}||^2
    # Cap rank at top-64 semantic dimensions so pure noise dimensions don't dominate leverage
    cov = (vc.T @ vc) / float(max(n - 1, 1))
    eigvals, eigvecs = np.linalg.eigh(cov)
    top_r = min(64, eigvals.size)
    vals = np.maximum(eigvals[-top_r:], 1e-6)
    vecs_r = eigvecs[:, -top_r:]
    proj = (vc @ vecs_r) * (vals ** -0.5)
    lev = np.sum(proj ** 2, axis=1)
    # Cap extreme top-2% outlier leverage so malformed stubs are rejected
    cap_hi = float(np.quantile(lev, 0.98))
    valid = np.where(lev <= cap_hi)[0]
    if valid.size < k:
        valid = np.arange(n)
    # Deterministic leverage-weighted sampling without replacement
    probs = lev[valid]
    probs = probs / float(np.sum(probs))
    rng = np.random.default_rng(seed + 202)
    picked = rng.choice(valid, size=k, replace=False, p=probs)
    return np.sort(picked)


def select_kmeanspp_spherical(vecs: np.ndarray, n_select: int, seed: int) -> np.ndarray:
    """Recommended spherical k-means++ (D^2 angular seeding) multimodal lobe precipitation."""
    vn = _l2_normalize(vecs)
    n = vn.shape[0]
    k = min(n_select, n)
    rng = np.random.default_rng(seed + 303)
    if n > 8192:
        pool = np.sort(rng.choice(n, size=8192, replace=False))
    else:
        pool = np.arange(n)
    vp = vn[pool]
    first = int(rng.integers(0, vp.shape[0]))
    chosen = [first]
    min_dist_sq = np.maximum(0.0, 2.0 - 2.0 * (vp @ vp[first]))
    min_dist_sq[first] = 0.0
    for _ in range(1, k):
        total = float(min_dist_sq.sum())
        if total <= 1e-12:
            rem = np.setdiff1d(np.arange(vp.shape[0]), np.asarray(chosen))
            nxt = int(rng.choice(rem))
        else:
            probs = min_dist_sq / total
            nxt = int(rng.choice(vp.shape[0], p=probs))
        chosen.append(nxt)
        d2 = np.maximum(0.0, 2.0 - 2.0 * (vp @ vp[nxt]))
        min_dist_sq = np.minimum(min_dist_sq, d2)
        min_dist_sq[chosen] = 0.0
    return np.sort(pool[np.asarray(chosen, dtype=np.int64)])


# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------
def run_precipitated_ablation(
    *,
    output_dir: Path,
    cache_dir: Path,
    datasets: Sequence[str],
    budgets: Sequence[int],
    seeds: Sequence[int],
    batch_size: int = 128,
) -> Dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be absent or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    sh_rows: List[Dict[str, Any]] = []
    mh_rows: List[Dict[str, Any]] = []
    swaps = [
        ("E0_to_E1_samedim", "E0", "E1"),
        ("E0_to_E3_crossdim", "E0", "E3"),
    ]
    chain = ["E0", "E1", "E2", "E3"]
    max_probe_budget = max(budgets)

    for ds_name in datasets:
        print(f"[PRECIP] Loading dataset {ds_name}...", flush=True)
        doc_ids, doc_texts, query_ids, query_texts, qrels, qrels_sha = load_beir_dataset(ds_name)
        splits = split_query_ids(query_ids)
        qid_to_idx = {qid: i for i, qid in enumerate(query_ids)}
        dev_idx = [qid_to_idx[q] for q in splits["DEVELOPMENT"]]
        rep_idx = [qid_to_idx[q] for q in splits["REPORT"]]
        split_sha = sha256_json(splits)
        _, contract_sha = metric_contract_digest(qrels_sha, k=10, gain_rule="graded")

        enc: Dict[str, Dict[str, Any]] = {}
        syn_enc: Dict[str, np.ndarray] = {}
        probe_texts = build_synthetic_random_probes(doc_texts, n_probes=max_probe_budget, seed=42)
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
            syn_enc[ek] = encode_synthetic_probes(
                dataset_name=ds_name,
                encoder_key=ek,
                encoder_name=ENCODERS[ek],
                probe_texts=probe_texts,
                cache_dir=cache_dir,
                batch_size=batch_size,
            )

        # Identify tail queries (25% of REPORT queries farthest from corpus centroid in E0)
        d0 = enc["E0"]["docs"]
        q0 = enc["E0"]["queries"]
        d0_cent = _l2_normalize(d0.mean(axis=0, keepdims=True))[0]
        rep_q0_cos = _l2_normalize(q0[rep_idx]) @ d0_cent
        tail_cut = float(np.quantile(rep_q0_cos, 0.25))
        tail_rep_qids = {splits["REPORT"][i] for i in range(len(rep_idx)) if rep_q0_cos[i] <= tail_cut}

        # ------------------------------------------------------------------
        # Single-Hop Comparison across Anchor Budgets {100, 250, 500, 1000}
        # ------------------------------------------------------------------
        for swap_label, old_k, new_k in swaps:
            d_old = enc[old_k]["docs"]
            d_new, q_new = enc[new_k]["docs"], enc[new_k]["queries"]
            s_old, s_new = syn_enc[old_k], syn_enc[new_k]

            for n_anc in budgets:
                eff_n = min(int(n_anc), len(doc_ids) // 2)
                for seed in seeds:
                    print(f"[PRECIP-S1] {ds_name} | {swap_label} | n={eff_n} | seed={seed}...", flush=True)
                    idx_u = select_uniform_iid(len(doc_ids), eff_n, seed)
                    idx_tail = select_tail_precipitated(d_new, eff_n)
                    idx_half = select_uniform_iid(len(doc_ids), eff_n // 2, seed + 55)
                    idx_kc = select_farthest_point_kcenter(d_new, eff_n, seed)
                    idx_lev = select_leverage_d_optimal(d_new, eff_n, seed)
                    idx_kmpp = select_kmeanspp_spherical(d_new, eff_n, seed)

                    n_syn_half = eff_n - len(idx_half)
                    rng_syn = np.random.default_rng(seed + 808)
                    syn_full_idx = rng_syn.choice(s_new.shape[0], size=min(eff_n, s_new.shape[0]), replace=False)
                    syn_half_idx = syn_full_idx[: min(n_syn_half, syn_full_idx.size)]

                    anchor_sets: Dict[str, Tuple[np.ndarray, np.ndarray]] = {
                        "P0_uniform_iid": (d_new[idx_u], d_old[idx_u]),
                        "P1_user_synthetic_random_text": (s_new[syn_full_idx], s_old[syn_full_idx]),
                        "P2_user_tail_precipitated": (d_new[idx_tail], d_old[idx_tail]),
                        "P3_user_mixed_50_50_probes": (
                            np.vstack([d_new[idx_half], s_new[syn_half_idx]]),
                            np.vstack([d_old[idx_half], s_old[syn_half_idx]]),
                        ),
                        "P4_rec_farthest_point_kcenter": (d_new[idx_kc], d_old[idx_kc]),
                        "P5_rec_leverage_d_optimal": (d_new[idx_lev], d_old[idx_lev]),
                        "P6_rec_kmeanspp_spherical": (d_new[idx_kmpp], d_old[idx_kmpp]),
                    }

                    # Evaluate Centered Procrustes & Centered Ridge on each anchor set + P7 Nystrom on P6
                    pq_proc: Dict[str, Dict[str, Dict[str, float]]] = {}
                    pq_ridge: Dict[str, Dict[str, Dict[str, float]]] = {}
                    stats_map: Dict[str, Tuple[float, float]] = {}

                    for pname, (xs, yt) in anchor_sets.items():
                        stats_map[pname] = _anchor_geometry_stats(xs)
                        mp = fit_centered_procrustes_xy(xs, yt)
                        rk_p, _ = score_dense_rankings(
                            apply_centered_procrustes_xy(q_new[rep_idx], mp), d_old, splits["REPORT"], doc_ids, k=10
                        )
                        pq_proc[pname] = evaluate_rankings_mlr(rk_p, qrels, splits["REPORT"], k=10)

                        # Tune centered ridge lambda strictly on DEV
                        best_mr = None
                        best_sc = -1.0
                        for lam in DEFAULT_RIDGE_LAMBDAS:
                            mc = fit_centered_ridge_xy(xs, yt, lam=lam)
                            rd, _ = score_dense_rankings(
                                apply_centered_ridge_xy(q_new[dev_idx], mc), d_old, splits["DEVELOPMENT"], doc_ids, k=10
                            )
                            sc = summarize_per_query_metrics(
                                evaluate_rankings_mlr(rd, qrels, splits["DEVELOPMENT"], k=10)
                            )["mean_ndcg_at_10"]
                            if sc > best_sc:
                                best_sc = sc
                                best_mr = mc
                        assert best_mr is not None
                        rk_r, _ = score_dense_rankings(
                            apply_centered_ridge_xy(q_new[rep_idx], best_mr), d_old, splits["REPORT"], doc_ids, k=10
                        )
                        pq_ridge[pname] = evaluate_rankings_mlr(rk_r, qrels, splits["REPORT"], k=10)

                    # P7: Distribution-free Nystrom Landmark Kernel over P6 spherical k-means++ landmarks
                    xs_km, yt_km = anchor_sets["P6_rec_kmeanspp_spherical"]
                    stats_map["P7_rec_nystrom_landmark_kernel"] = stats_map["P6_rec_kmeanspp_spherical"]
                    best_tau = 0.07
                    best_nt_sc = -1.0
                    for tau in (0.05, 0.10, 0.20):
                        rd, _ = score_dense_rankings(
                            apply_nystrom_landmark_kernel(q_new[dev_idx], xs_km, yt_km, tau=tau),
                            d_old, splits["DEVELOPMENT"], doc_ids, k=10,
                        )
                        sc = summarize_per_query_metrics(
                            evaluate_rankings_mlr(rd, qrels, splits["DEVELOPMENT"], k=10)
                        )["mean_ndcg_at_10"]
                        if sc > best_nt_sc:
                            best_nt_sc = sc
                            best_tau = tau
                    rk_nt, _ = score_dense_rankings(
                        apply_nystrom_landmark_kernel(q_new[rep_idx], xs_km, yt_km, tau=best_tau),
                        d_old, splits["REPORT"], doc_ids, k=10,
                    )
                    pq_nyst = evaluate_rankings_mlr(rk_nt, qrels, splits["REPORT"], k=10)

                    # Log rows for Centered Procrustes, Centered Ridge, and Nystrom Kernel
                    p0_proc = _ndcg_map(pq_proc["P0_uniform_iid"])
                    p0_ridge = _ndcg_map(pq_ridge["P0_uniform_iid"])

                    for pname in anchor_sets:
                        cond_v, erank_v = stats_map[pname]
                        for mapper, pq_cur, p0_ref in (
                            ("centered_procrustes", pq_proc[pname], p0_proc),
                            ("centered_ridge", pq_ridge[pname], p0_ridge),
                        ):
                            cur_map = _ndcg_map(pq_cur)
                            tail_vals = [cur_map[q] for q in cur_map if q in tail_rep_qids]
                            boot = paired_bootstrap_delta(cur_map, p0_ref, n_bootstrap=3000, seed=seed)
                            sh_rows.append(
                                {
                                    "dataset": ds_name,
                                    "swap": swap_label,
                                    "anchor_budget": eff_n,
                                    "seed": int(seed),
                                    "strategy": pname,
                                    "mapper": mapper,
                                    "report_ndcg_at_10": round(float(np.mean(list(cur_map.values()))), 6),
                                    "tail_ndcg_at_10": round(float(np.mean(tail_vals)) if tail_vals else 0.0, 6),
                                    "delta_vs_p0_uniform": round(boot["mean_delta"], 6),
                                    "delta_vs_p0_lcb95": round(boot["lcb_95"], 6),
                                    "anchor_cond": round(cond_v, 3),
                                    "anchor_eff_rank": round(erank_v, 2),
                                    "split_sha256": split_sha,
                                    "metric_contract_sha256": contract_sha,
                                }
                            )

                    # Log P7 Nystrom row (compared against P0 centered ridge)
                    cur_nt = _ndcg_map(pq_nyst)
                    tail_nt = [cur_nt[q] for q in cur_nt if q in tail_rep_qids]
                    boot_nt = paired_bootstrap_delta(cur_nt, p0_ridge, n_bootstrap=3000, seed=seed)
                    cond_v, erank_v = stats_map["P7_rec_nystrom_landmark_kernel"]
                    sh_rows.append(
                        {
                            "dataset": ds_name,
                            "swap": swap_label,
                            "anchor_budget": eff_n,
                            "seed": int(seed),
                            "strategy": "P7_rec_nystrom_landmark_kernel",
                            "mapper": "nystrom_kernel",
                            "report_ndcg_at_10": round(float(np.mean(list(cur_nt.values()))), 6),
                            "tail_ndcg_at_10": round(float(np.mean(tail_nt)) if tail_nt else 0.0, 6),
                            "delta_vs_p0_uniform": round(boot_nt["mean_delta"], 6),
                            "delta_vs_p0_lcb95": round(boot_nt["lcb_95"], 6),
                            "anchor_cond": round(cond_v, 3),
                            "anchor_eff_rank": round(erank_v, 2),
                            "split_sha256": split_sha,
                            "metric_contract_sha256": contract_sha,
                        }
                    )

        # ------------------------------------------------------------------
        # Multi-Hop (m=3: E3 -> E2 -> E1 -> E0) Stability at Low Budgets {250, 500}
        # ------------------------------------------------------------------
        for n_anc in (250, 500):
            eff_n = min(n_anc, len(doc_ids) // 2)
            for seed in seeds:
                print(f"[PRECIP-MH] {ds_name} | hop3 | n={eff_n} | seed={seed}...", flush=True)
                for strat in (
                    "P0_uniform_iid",
                    "P1_user_synthetic_random_text",
                    "P2_user_tail_precipitated",
                    "P3_user_mixed_50_50_probes",
                    "P4_rec_farthest_point_kcenter",
                    "P5_rec_leverage_d_optimal",
                    "P6_rec_kmeanspp_spherical",
                ):
                    q_proc_chain = enc["E3"]["queries"][rep_idx].copy()
                    q_ridge_chain = enc["E3"]["queries"][rep_idx].copy()
                    for step in (3, 2, 1):
                        sk, tk = chain[step], chain[step - 1]
                        ds_src, ds_tgt = enc[sk]["docs"], enc[tk]["docs"]
                        ss_src, ss_tgt = syn_enc[sk], syn_enc[tk]
                        s_step = seed + step * 73
                        if strat == "P0_uniform_iid":
                            ii = select_uniform_iid(len(doc_ids), eff_n, s_step)
                            xs, yt = ds_src[ii], ds_tgt[ii]
                        elif strat == "P1_user_synthetic_random_text":
                            r0 = np.random.default_rng(s_step)
                            si = r0.choice(ss_src.shape[0], size=min(eff_n, ss_src.shape[0]), replace=False)
                            xs, yt = ss_src[si], ss_tgt[si]
                        elif strat == "P2_user_tail_precipitated":
                            ii = select_tail_precipitated(ds_src, eff_n)
                            xs, yt = ds_src[ii], ds_tgt[ii]
                        elif strat == "P3_user_mixed_50_50_probes":
                            ii = select_uniform_iid(len(doc_ids), eff_n // 2, s_step)
                            r0 = np.random.default_rng(s_step + 9)
                            si = r0.choice(ss_src.shape[0], size=min(eff_n - len(ii), ss_src.shape[0]), replace=False)
                            xs, yt = np.vstack([ds_src[ii], ss_src[si]]), np.vstack([ds_tgt[ii], ss_tgt[si]])
                        elif strat == "P4_rec_farthest_point_kcenter":
                            ii = select_farthest_point_kcenter(ds_src, eff_n, s_step)
                            xs, yt = ds_src[ii], ds_tgt[ii]
                        elif strat == "P5_rec_leverage_d_optimal":
                            ii = select_leverage_d_optimal(ds_src, eff_n, s_step)
                            xs, yt = ds_src[ii], ds_tgt[ii]
                        else:
                            ii = select_kmeanspp_spherical(ds_src, eff_n, s_step)
                            xs, yt = ds_src[ii], ds_tgt[ii]

                        mp = fit_centered_procrustes_xy(xs, yt)
                        q_proc_chain = apply_centered_procrustes_xy(q_proc_chain, mp)

                        # Use fixed moderate ridge regularization (lam=1.0) for fast fair hop composition
                        mr = fit_centered_ridge_xy(xs, yt, lam=1.0)
                        q_ridge_chain = apply_centered_ridge_xy(q_ridge_chain, mr)

                    rk_p, _ = score_dense_rankings(q_proc_chain, d0, splits["REPORT"], doc_ids, k=10)
                    rk_r, _ = score_dense_rankings(q_ridge_chain, d0, splits["REPORT"], doc_ids, k=10)
                    mp_ndcg = summarize_per_query_metrics(evaluate_rankings_mlr(rk_p, qrels, splits["REPORT"], k=10))["mean_ndcg_at_10"]
                    mr_ndcg = summarize_per_query_metrics(evaluate_rankings_mlr(rk_r, qrels, splits["REPORT"], k=10))["mean_ndcg_at_10"]
                    mh_rows.append(
                        {
                            "dataset": ds_name,
                            "anchor_budget": eff_n,
                            "seed": int(seed),
                            "strategy": strat,
                            "hop3_centered_procrustes_ndcg": round(mp_ndcg, 6),
                            "hop3_centered_ridge_ndcg": round(mr_ndcg, 6),
                        }
                    )

    # Save CSVs & Markdown summary
    for fname, rows in (
        ("precipitated_singlehop.csv", sh_rows),
        ("precipitated_multihop_hop3.csv", mh_rows),
    ):
        with (output_dir / fname).open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    md = [
        "# Precipitated Random / Non-Distributional vs. Manifold-Precipitated Anchor Sets (`mlr-v1`)",
        "",
        f"- **Git SHA:** `{current_git_sha(REPO_ROOT)}`",
        f"- **Wall time:** `{round(time.perf_counter() - t_start, 2)}s`",
        "",
        "## 1. Single-Hop Mean REPORT nDCG@10 Across Anchor Budgets {100, 250, 500, 1000} (Centered Ridge / Nystrom)",
        "",
        "| Dataset | Swap | P0 Uniform i.i.d. | P1 User: Synthetic Random | P2 User: Tail Precip | P3 User: 50/50 Mixed | P4 Rec: k-Center | P5 Rec: Leverage D-Opt | P6 Rec: k-Means++ | P7 Rec: Nystrom Kernel |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    grp: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    grp_proc: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for r in sh_rows:
        if r["mapper"] in ("centered_ridge", "nystrom_kernel"):
            grp[(r["dataset"], r["swap"], r["strategy"])].append(float(r["report_ndcg_at_10"]))
        if r["mapper"] == "centered_procrustes":
            grp_proc[(r["dataset"], r["swap"], r["strategy"])].append(float(r["report_ndcg_at_10"]))

    strats_all = [
        "P0_uniform_iid",
        "P1_user_synthetic_random_text",
        "P2_user_tail_precipitated",
        "P3_user_mixed_50_50_probes",
        "P4_rec_farthest_point_kcenter",
        "P5_rec_leverage_d_optimal",
        "P6_rec_kmeanspp_spherical",
        "P7_rec_nystrom_landmark_kernel",
    ]
    for ds in datasets:
        for sw, _, _ in swaps:
            vals = [f"{sum(grp[(ds, sw, s)]) / len(grp[(ds, sw, s)]):.4f}" for s in strats_all]
            md.append(f"| {ds} | {sw} | " + " | ".join(vals) + " |")

    md += [
        "",
        "## 2. Single-Hop Mean REPORT nDCG@10 — Centered Procrustes Mapper",
        "",
        "| Dataset | Swap | P0 Uniform i.i.d. | P1 User: Synthetic Random | P2 User: Tail Precip | P3 User: 50/50 Mixed | P4 Rec: k-Center | P5 Rec: Leverage D-Opt | P6 Rec: k-Means++ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ds in datasets:
        for sw, _, _ in swaps:
            vals = [f"{sum(grp_proc[(ds, sw, s)]) / len(grp_proc[(ds, sw, s)]):.4f}" for s in strats_all[:-1]]
            md.append(f"| {ds} | {sw} | " + " | ".join(vals) + " |")

    md += [
        "",
        "## 3. Multi-Hop Hop-3 (E3 -> E2 -> E1 -> E0) Composed Centered Procrustes (Mean Over Budgets {250, 500} x 3 Seeds)",
        "",
        "| Dataset | P0 Uniform i.i.d. | P1 User: Synthetic | P2 User: Tail | P3 User: 50/50 Mix | P4 Rec: k-Center | P5 Rec: Leverage | P6 Rec: k-Means++ |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    mh_g: Dict[Tuple[str, str], List[float]] = defaultdict(list)
    for r in mh_rows:
        mh_g[(r["dataset"], r["strategy"])].append(float(r["hop3_centered_procrustes_ndcg"]))
    for ds in datasets:
        vals = [f"{sum(mh_g[(ds, s)]) / len(mh_g[(ds, s)]):.4f}" for s in strats_all[:-1]]
        md.append(f"| {ds} | " + " | ".join(vals) + " |")

    (output_dir / "precipitated_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    manifest = {
        "job": "night1_precipitated_sets",
        "git_sha": current_git_sha(REPO_ROOT),
        "datasets": list(datasets),
        "budgets": [int(b) for b in budgets],
        "seeds": [int(s) for s in seeds],
        "wall_seconds": round(time.perf_counter() - t_start, 2),
        "telemetry": process_telemetry(),
    }
    manifest_sha = write_json_atomic(output_dir / "precipitated_manifest.json", manifest)
    write_json_atomic(
        output_dir / "report_receipt.json",
        {"job": "night1_precipitated_sets", "status": "COMPLETED", "manifest_sha256": manifest_sha},
    )
    print(f"[PRECIP] COMPLETE in {manifest['wall_seconds']}s -> {output_dir}", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Precipitated Random vs Manifold-Precipitated Anchor Sets")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/chelated_night1_emb_cache"))
    parser.add_argument("--datasets", nargs="+", default=["SciFact", "NFCorpus", "FiQA2018"])
    parser.add_argument("--budgets", nargs="+", type=int, default=[100, 250, 500, 1000])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 137, 2718])
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    run_precipitated_ablation(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        datasets=args.datasets,
        budgets=args.budgets,
        seeds=args.seeds,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
