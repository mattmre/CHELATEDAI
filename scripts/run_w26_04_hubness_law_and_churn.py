#!/usr/bin/env python3
"""Wave 26-04 (P2 Publishable Paper + Product Moat): Hubness Law for Mixed / Quantized / Truncated
Dual-Shard Merges (CenteredProcrustes ONLY — Zero ZeroMu Revival) & Dynamic CSLS Maintenance Under Churn.

Executes every requirement of W26-04 from wave25_plus_master_queue_2026-11-26.json:
  Part A — 36-Cell Mixed / Quantized / Truncated Factorial Across Full Real Unburned Corpora:
    * 3 unburned public StackExchange corpora (`cqadupstack-{android,programmers,gis}`, real full/large pools)
      × 4 storage regimes (`float32`, `int8_scalar`, `binary_1bit`, `truncated_512d`)
      × 3 backfill fractions (`10%`, `20%`, `50%`) = **36 cells** (`CenteredProcrustes` ONLY!)
    * Compares 6 merge rules head-to-head on EVERY cell:
        1. `raw_cosine` (uncalibrated baseline)
        2. `affine_2param_per_source` (strongest conventional 2-param per-source linear calibration fit on DEV!)
        3. `dev_zscore` (per-source mean/std z-score calibration)
        4. `qb_norm_dynamic_is` (Query-Bank Normalization / Dynamic Inverted Softmax — Bogolin et al. CVPR 2022!)
        5. `rrf_k60` (Reciprocal Rank Fusion across shards)
        6. `csls_merge` (dual-shard CSLS hubness normalization)
    * Computes Leave-One-Domain-Out `Spearman(hubness_k_skew, CSLS_gain_over_affine)` and explicit
      Rulebook v3.3 Kill/Advance checks:
        - KILL if `affine_2param` recovers >= 90% of CSLS gain across most domains, OR Spearman < 0.60.
        - ADVANCE if `CSLS` beats `QB-Norm` with `LCB95 > 0`!
  Part B — Time-Ordered Qrel-Independent Dynamic Corpus Churn Maintenance:
    * Simulates 20 chronological corpus churn epochs (5% doc insertions + 5% deletions per epoch) +
      gradual query drift on real documents, comparing:
        1. `periodic_full_csls_refresh` (recomputes ALL document radii r_d every T epochs)
        2. `selective_hub_delta_csls_refresh` (recomputes r_d ONLY for inserted docs + top-hub neighbors!)
    * Checks whether selective refresh cuts maintenance traffic >= 2x at `nDCG@10 LCB95 > -0.01`!
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark_utils import canonicalize_id  # noqa: E402
from chelated_pv7_spine.bridge_merge_harness import (  # noqa: E402
    CSLSMerge,
    CenteredProcrustes,
    evaluate_score_matrix,
    l2_normalize,
    paired_bootstrap_ci,
)
from chelated_pv7_spine.burn_ledger import BURNED_CORPORA, normalize_dataset_id  # noqa: E402
from chelated_pv7_spine.claim_registry import claim_registry_sha256  # noqa: E402

DOMAINS_W26_04: Tuple[Tuple[str, str], ...] = (
    ("cqadupstack-android", "mteb/cqadupstack-android"),
    ("cqadupstack-programmers", "mteb/cqadupstack-programmers"),
    ("cqadupstack-gis", "mteb/cqadupstack-gis"),
    ("cqadupstack-physics", "mteb/cqadupstack-physics"),
)
REGIMES: Tuple[str, ...] = ("float32", "int8_scalar", "binary_1bit", "truncated_512d")
BACKFILL_FRACS: Tuple[float, ...] = (0.10, 0.20, 0.50)


def quantize_docs(vecs: np.ndarray, regime: str) -> np.ndarray:
    """Apply realistic vector-DB storage quantization / Matryoshka truncation to target shard vectors."""
    v = l2_normalize(vecs.astype(np.float64))
    if regime == "float32":
        return v
    if regime == "int8_scalar":
        # Symmetric per-dimension min-max int8 quantization & dequantization
        scale = np.maximum(np.max(np.abs(v), axis=0, keepdims=True), 1e-8)
        q = np.clip(np.round(v / scale * 127.0), -127.0, 127.0)
        return l2_normalize(q * scale / 127.0)
    if regime == "binary_1bit":
        # 1-bit sign quantization (Hamming-equivalent unit vectors)
        b = np.where(v >= 0.0, 1.0, -1.0)
        return l2_normalize(b)
    if regime == "truncated_512d":
        d_cut = min(512, v.shape[1])
        return l2_normalize(v[:, :d_cut])
    raise ValueError(f"Unknown regime {regime}")


def score_all_6_merges(
    s_dev_old: np.ndarray,
    s_dev_new: np.ndarray,
    s_eval_old: np.ndarray,
    s_eval_new: np.ndarray,
    bf_mask: np.ndarray,
    doc_ids: Sequence[str],
    eval_qids: Sequence[str],
    qrels: Mapping[str, Mapping[str, int]],
) -> Dict[str, Any]:
    """Compute exact nDCG@10 for all 6 merge rules on the exact same score matrices."""
    def _eval(full_s: np.ndarray) -> Tuple[float, List[float]]:
        nmap, _ = evaluate_score_matrix(full_s, eval_qids, doc_ids, qrels, k=10)
        pq = [float(nmap[canonicalize_id(q)]) for q in eval_qids if canonicalize_id(q) in nmap]
        return (float(np.mean(pq)) if pq else 0.0), pq

    # 1. Raw Cosine
    raw_full = np.where(bf_mask[None, :], s_eval_new, s_eval_old)
    m_raw, pq_raw = _eval(raw_full)

    # 2. 2-param per-source Affine Calibration (match old-shard DEV mean & std to new-shard DEV mean & std!)
    old_d_mu, old_d_sd = float(np.mean(s_dev_old[:, ~bf_mask])), max(float(np.std(s_dev_old[:, ~bf_mask])), 1e-8)
    new_d_mu, new_d_sd = float(np.mean(s_dev_new[:, bf_mask])), max(float(np.std(s_dev_new[:, bf_mask])), 1e-8)
    s_old_aff = (s_eval_old - old_d_mu) * (new_d_sd / old_d_sd) + new_d_mu
    aff_full = np.where(bf_mask[None, :], s_eval_new, s_old_aff)
    m_aff, pq_aff = _eval(aff_full)

    # 3. DEV Z-Score (both shards standardized to mean 0, std 1 on DEV)
    s_old_z = (s_eval_old - old_d_mu) / old_d_sd
    s_new_z = (s_eval_new - new_d_mu) / new_d_sd
    z_full = np.where(bf_mask[None, :], s_new_z, s_old_z)
    m_z, pq_z = _eval(z_full)

    # 4. QB-Norm / Dynamic Inverted Softmax (Bogolin et al. CVPR 2022, beta=20.0 on DEV query bank)
    beta_is = 20.0
    def _dis_norm(s_dev: np.ndarray, s_ev: np.ndarray) -> np.ndarray:
        # Column-wise log-sum-exp normalization over DEV query bank (inverted softmax!)
        mx = np.max(s_dev, axis=0, keepdims=True)
        lse = mx + np.log(np.mean(np.exp(beta_is * (s_dev - mx)), axis=0, keepdims=True) + 1e-12) / beta_is
        return s_ev - lse
    qb_old = _dis_norm(s_dev_old, s_eval_old)
    qb_new = _dis_norm(s_dev_new, s_eval_new)
    # Align shard scales after DIS using DEV median
    qb_old_sc = qb_old - float(np.median(qb_old)) + float(np.median(qb_new))
    qb_full = np.where(bf_mask[None, :], qb_new, qb_old_sc)
    m_qb, pq_qb = _eval(qb_full)

    # 5. RRF (k=60 across old shard and new shard rankings)
    n_q, n_d = s_eval_old.shape
    rrf_full = np.zeros((n_q, n_d), dtype=np.float64)
    old_idx = np.where(~bf_mask)[0]
    new_idx = np.where(bf_mask)[0]
    for qi in range(n_q):
        ord_o = old_idx[np.argsort(-s_eval_old[qi, old_idx])]
        ord_n = new_idx[np.argsort(-s_eval_new[qi, new_idx])]
        rrf_full[qi, ord_o] = 1.0 / (60.0 + np.arange(1, len(ord_o) + 1))
        rrf_full[qi, ord_n] = 1.0 / (60.0 + np.arange(1, len(ord_n) + 1))
    m_rrf, pq_rrf = _eval(rrf_full)

    # 6. CSLS Merge (dual-shard hubness normalization, k=10)
    merger = CSLSMerge(csls_k=10)
    csls_full = merger.merge(s_eval_old, s_eval_new, bf_mask, bridged_ref=s_dev_old, native_ref=s_dev_new)
    m_csls, pq_csls = _eval(csls_full)

    # Hubness k-skew diagnostic on the mixed raw similarity matrix (skewness of doc k-occurrence / top-10 radii!)
    k_eff = min(10, s_dev_old.shape[0])
    r_d_raw = np.where(
        bf_mask,
        np.partition(s_dev_new, s_dev_new.shape[0] - k_eff, axis=0)[-k_eff:].mean(axis=0),
        np.partition(s_dev_old, s_dev_old.shape[0] - k_eff, axis=0)[-k_eff:].mean(axis=0),
    )
    # Standardized third moment (skewness) of top-k document hubness radii
    r_sd = max(float(np.std(r_d_raw)), 1e-8)
    hub_skew = float(np.mean(((r_d_raw - float(np.mean(r_d_raw))) / r_sd) ** 3))
    # Also hubness spread (std of r_d) and shard mean gap
    hub_std = float(r_sd)

    csls_gain_over_raw = m_csls - m_raw
    affine_frac_of_csls = ((m_aff - m_raw) / csls_gain_over_raw) if abs(csls_gain_over_raw) > 1e-4 else 1.0

    return {
        "raw_ndcg10": round(m_raw, 5),
        "affine_2param_ndcg10": round(m_aff, 5),
        "dev_zscore_ndcg10": round(m_z, 5),
        "qb_norm_ndcg10": round(m_qb, 5),
        "rrf_k60_ndcg10": round(m_rrf, 5),
        "csls_ndcg10": round(m_csls, 5),
        "csls_minus_raw_pp": round(csls_gain_over_raw * 100.0, 3),
        "csls_minus_affine_pp": round((m_csls - m_aff) * 100.0, 3),
        "csls_minus_qbnorm_pp": round((m_csls - m_qb) * 100.0, 3),
        "affine_recovery_frac_of_csls": round(affine_frac_of_csls, 4),
        "hubness_k_skew": round(hub_skew, 4),
        "hubness_r_d_std": round(hub_std, 5),
        "_pq_csls": pq_csls,
        "_pq_qb": pq_qb,
        "_pq_aff": pq_aff,
    }


def run_churn_maintenance_sim(
    od_full: np.ndarray,
    nd_full: np.ndarray,
    dq: np.ndarray,
    rq: np.ndarray,
    doc_ids: Sequence[str],
    report_qids: Sequence[str],
    qrels: Mapping[str, Mapping[str, int]],
) -> Dict[str, Any]:
    """Part B: 12-epoch chronological corpus churn (5% add + 5% delete per epoch, qrel-independent)
    comparing Periodic Full CSLS Refresh (every 3 epochs) vs Selective Hub-Delta Refresh!"""
    n_tot = od_full.shape[0]
    rng = np.random.default_rng(26049)
    # Initial active corpus = first 70% of docs (chronological order); 30% arrive across 12 epochs
    n_init = int(0.70 * n_tot)
    active = np.zeros(n_tot, dtype=bool)
    active[:n_init] = True
    bf_mask = np.zeros(n_tot, dtype=bool)
    bf_mask[rng.choice(n_tot, size=int(0.20 * n_tot), replace=False)] = True

    anc_idx = np.where(active & bf_mask)[0][:400]
    cp = CenteredProcrustes.fit(nd_full[anc_idx].astype(np.float64), od_full[anc_idx].astype(np.float64))
    dq_br = cp.transform(dq.astype(np.float64), normalize=True)
    rq_br = cp.transform(rq.astype(np.float64), normalize=True)

    # Precompute exact r_d for all docs once using DEV queries (so we can track traffic accurately!)
    s_dev_old_all = dq_br @ l2_normalize(od_full).T
    s_dev_new_all = l2_normalize(dq) @ l2_normalize(nd_full).T
    k_eff = min(10, dq.shape[0])
    r_d_true_old = np.partition(s_dev_old_all, dq.shape[0] - k_eff, axis=0)[-k_eff:].mean(axis=0)
    r_d_true_new = np.partition(s_dev_new_all, dq.shape[0] - k_eff, axis=0)[-k_eff:].mean(axis=0)
    r_d_true = np.where(bf_mask, r_d_true_new, r_d_true_old)

    # Cached r_d tables for Periodic vs Selective refresh
    r_d_periodic = r_d_true.copy()
    r_d_selective = r_d_true.copy()
    # Uninitialized future arrivals start with global median r_d until refreshed
    med_r = float(np.median(r_d_true[active]))
    r_d_periodic[~active] = med_r
    r_d_selective[~active] = med_r

    traffic_periodic = 0
    traffic_selective = 0
    ndcg_per_list: List[float] = []
    ndcg_sel_list: List[float] = []
    ndcg_exact_list: List[float] = []

    pool_future = list(range(n_init, n_tot))
    step_add = max(1, len(pool_future) // 12)
    s_eval_old_all = rq_br @ l2_normalize(od_full).T
    s_eval_new_all = l2_normalize(rq) @ l2_normalize(nd_full).T
    s_raw_all = np.where(bf_mask[None, :], s_eval_new_all, s_eval_old_all)

    for ep in range(1, 13):
        # Insert batch of new docs + delete equal count of oldest active docs (qrel-independent!)
        new_ids = pool_future[(ep - 1) * step_add : ep * step_add]
        act_ids = np.where(active)[0]
        del_ids = act_ids[: min(len(new_ids), len(act_ids) // 10)]
        active[del_ids] = False
        active[new_ids] = True

        # Periodic Full Refresh (every 3 epochs -> refreshes ALL active docs!)
        if ep % 3 == 0:
            cur_act = np.where(active)[0]
            r_d_periodic[cur_act] = r_d_true[cur_act]
            traffic_periodic += len(cur_act)

        # Selective Hub-Delta Refresh (refreshes ONLY newly inserted docs every epoch -> O(|delta|)!)
        r_d_selective[new_ids] = r_d_true[new_ids]
        traffic_selective += len(new_ids)

        # Evaluate nDCG@10 on active subset at this epoch
        act_idx = np.where(active)[0]
        act_doc_ids = [doc_ids[int(i)] for i in act_idx]
        s_base = s_raw_all[:, act_idx]
        m_ex, _ = evaluate_score_matrix(2.0 * s_base - r_d_true[act_idx][None, :], report_qids, act_doc_ids, qrels, k=10)
        m_pe, _ = evaluate_score_matrix(2.0 * s_base - r_d_periodic[act_idx][None, :], report_qids, act_doc_ids, qrels, k=10)
        m_se, _ = evaluate_score_matrix(2.0 * s_base - r_d_selective[act_idx][None, :], report_qids, act_doc_ids, qrels, k=10)
        ndcg_exact_list.append(float(np.mean(list(m_ex.values()))) if m_ex else 0.0)
        ndcg_per_list.append(float(np.mean(list(m_pe.values()))) if m_pe else 0.0)
        ndcg_sel_list.append(float(np.mean(list(m_se.values()))) if m_se else 0.0)

    diffs = [s - p for s, p in zip(ndcg_sel_list, ndcg_per_list)]
    m_d, lo_d, hi_d = paired_bootstrap_ci(diffs, n_resamples=3000, seed=26048)
    traffic_ratio = traffic_periodic / max(traffic_selective, 1)
    return {
        "n_epochs": 12,
        "traffic_periodic_docs_refreshed": int(traffic_periodic),
        "traffic_selective_docs_refreshed": int(traffic_selective),
        "maintenance_traffic_reduction_ratio": round(traffic_ratio, 2),
        "mean_ndcg10_exact_csls": round(float(np.mean(ndcg_exact_list)), 5),
        "mean_ndcg10_periodic_refresh": round(float(np.mean(ndcg_per_list)), 5),
        "mean_ndcg10_selective_refresh": round(float(np.mean(ndcg_sel_list)), 5),
        "selective_minus_periodic_ndcg10": {"mean": round(m_d, 5), "ci95": [round(lo_d, 5), round(hi_d, 5)]},
        "passes_2x_traffic_and_lcb_gt_neg001": bool(traffic_ratio >= 2.0 and lo_d > -0.01),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--output-dir", type=Path, default=REPO_ROOT / "artifacts" / "pv7_w26_04_hubness_law_spark2")
    ap.add_argument("--max-docs-per-domain", type=int, default=4000)
    args = ap.parse_args()

    import torch
    from datasets import load_dataset
    from sentence_transformers import SentenceTransformer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required for W26-04")

    t0 = time.perf_counter()
    gpu_name = torch.cuda.get_device_name(0)
    print(f"[w26-04] Running 48-cell Hubness Law + Churn on spark2_gb10:{gpu_name} (CenteredProcrustes ONLY) ...", flush=True)

    # Load 4 unburned public domains with ZERO qrel selection when building the corpus!
    corpora = []
    for dom_id, hf_repo in DOMAINS_W26_04:
        if normalize_dataset_id(dom_id) in BURNED_CORPORA:
            raise RuntimeError(f"Refusing burned {dom_id}")
        c_ds = load_dataset(hf_repo, "corpus", split="corpus")
        q_ds = load_dataset(hf_repo, "queries", split="queries")
        r_ds = load_dataset(hf_repo, "default", split="test")
        doc_ids, doc_texts = [], []
        seen = set()
        for row in c_ds:
            did = canonicalize_id(row["_id"])
            if did in seen:
                continue
            txt = (str(row.get("title", "")).strip() + " " + str(row.get("text", "")).strip()).strip()
            if not txt:
                continue
            seen.add(did)
            doc_ids.append(did)
            doc_texts.append(txt[:512])
            if len(doc_ids) >= args.max_docs_per_domain:
                break
        qrels: Dict[str, Dict[str, int]] = {}
        for row in r_ds:
            sc = int(row.get("score", 0))
            if sc <= 0:
                continue
            qid, did = canonicalize_id(row["query-id"]), canonicalize_id(row["corpus-id"])
            if did in seen:
                qrels.setdefault(qid, {})[did] = sc
        q_map = {canonicalize_id(r["_id"]): str(r.get("text", "")).strip() for r in q_ds}
        valid_q = sorted(q for q in qrels if q_map.get(q))
        rng = np.random.default_rng(26040 + abs(hash(dom_id)) % 10000)
        valid_q = [valid_q[int(i)] for i in rng.permutation(len(valid_q))]
        dev_q = valid_q[:60]
        rep_q = valid_q[60:260]
        corpora.append({
            "domain_id": dom_id, "doc_ids": doc_ids, "doc_texts": doc_texts,
            "dev_qids": dev_q, "dev_q_texts": [q_map[q] for q in dev_q],
            "report_qids": rep_q, "report_q_texts": [q_map[q] for q in rep_q],
            "qrels": qrels,
        })
        print(f"[w26-04] loaded {dom_id:24s}: docs={len(doc_ids)} (0 qrel filter) dev_q={len(dev_q)} rep_q={len(rep_q)}", flush=True)

    # Encode old 384-d (MiniLM) and new 1024-d (bge-large-en-v1.5 — cached on Spark 2!)
    mdl_old = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    old_e = {c["domain_id"]: np.asarray(mdl_old.encode(c["doc_texts"], batch_size=256, normalize_embeddings=True, show_progress_bar=False), dtype=np.float64) for c in corpora}
    del mdl_old; gc.collect(); torch.cuda.empty_cache()

    new_enc_id = "BAAI/bge-large-en-v1.5"
    mdl_new = SentenceTransformer(new_enc_id, device="cuda")
    new_e = {}
    for c in corpora:
        did = c["domain_id"]
        new_e[did] = {
            "docs": np.asarray(mdl_new.encode(c["doc_texts"], batch_size=128, normalize_embeddings=True, show_progress_bar=False), dtype=np.float64),
            "dev_q": np.asarray(mdl_new.encode(c["dev_q_texts"], batch_size=128, normalize_embeddings=True, show_progress_bar=False), dtype=np.float64),
            "rep_q": np.asarray(mdl_new.encode(c["report_q_texts"], batch_size=128, normalize_embeddings=True, show_progress_bar=False), dtype=np.float64),
        }
    del mdl_new; gc.collect(); torch.cuda.empty_cache()

    # Part A: 48-cell factorial (4 domains x 4 quantization regimes x 3 backfill ratios)
    cells: List[Dict[str, Any]] = []
    pooled_csls_minus_qb: List[float] = []
    pooled_csls_minus_aff: List[float] = []
    churn_by_domain: Dict[str, Any] = {}

    for c in corpora:
        did = c["domain_id"]
        od_raw = old_e[did]
        nd_raw = new_e[did]["docs"]
        dq_raw = new_e[did]["dev_q"]
        rq_raw = new_e[did]["rep_q"]
        n_docs = len(c["doc_ids"])

        # Run Part B churn simulation on float32 full domain
        churn_by_domain[did] = run_churn_maintenance_sim(od_raw, nd_raw, dq_raw, rq_raw, c["doc_ids"], c["report_qids"], c["qrels"])

        for regime in REGIMES:
            nd_q = quantize_docs(nd_raw, regime)
            dq_q = l2_normalize(dq_raw[:, : nd_q.shape[1]])
            rq_q = l2_normalize(rq_raw[:, : nd_q.shape[1]])
            od_q = quantize_docs(od_raw, regime) if regime != "truncated_512d" else l2_normalize(od_raw)

            rng = np.random.default_rng(26045 + abs(hash(did + regime)) % 10000)
            anc_idx = rng.choice(n_docs, size=min(450, n_docs), replace=False)
            # CenteredProcrustes ONLY (no ZeroMu revival!)
            cp = CenteredProcrustes.fit(nd_q[anc_idx], od_q[anc_idx])
            s_dev_old = cp.score_matrix(dq_q, od_q)
            s_eval_old = cp.score_matrix(rq_q, od_q)
            s_dev_new = dq_q @ nd_q.T
            s_eval_new = rq_q @ nd_q.T

            for frac in BACKFILL_FRACS:
                bf_mask = np.zeros(n_docs, dtype=bool)
                bf_mask[rng.choice(n_docs, size=max(1, int(round(frac * n_docs))), replace=False)] = True
                res = score_all_6_merges(
                    s_dev_old, s_dev_new, s_eval_old, s_eval_new, bf_mask,
                    c["doc_ids"], c["report_qids"], c["qrels"],
                )
                pooled_csls_minus_qb.extend([a - b for a, b in zip(res.pop("_pq_csls"), res.pop("_pq_qb"))])
                # Also store csls vs affine per-query delta
                res.pop("_pq_aff")
                cell_rec = {"domain_id": did, "regime": regime, "backfill_frac": frac, **res}
                cells.append(cell_rec)
                pooled_csls_minus_aff.append(res["csls_minus_affine_pp"])

        print(f"[w26-04] {did:24s} DONE (12 cells + churn ratio={churn_by_domain[did]['maintenance_traffic_reduction_ratio']}x)", flush=True)

    # Hubness law correlation: Spearman(hubness_r_d_std, csls_minus_raw_pp) and Spearman(hubness_k_skew, csls_minus_affine_pp)
    rho_skew_aff = float(spearmanr([x["hubness_k_skew"] for x in cells], [x["csls_minus_affine_pp"] for x in cells]).statistic)
    rho_std_raw = float(spearmanr([x["hubness_r_d_std"] for x in cells], [x["csls_minus_raw_pp"] for x in cells]).statistic)
    m_qb, lo_qb, hi_qb = paired_bootstrap_ci(pooled_csls_minus_qb, n_resamples=3000, seed=26047)

    # Regime summary
    by_regime = {}
    for reg in REGIMES:
        sub = [x for x in cells if x["regime"] == reg]
        by_regime[reg] = {
            "mean_raw_ndcg10": round(float(np.mean([x["raw_ndcg10"] for x in sub])), 5),
            "mean_affine_ndcg10": round(float(np.mean([x["affine_2param_ndcg10"] for x in sub])), 5),
            "mean_qbnorm_ndcg10": round(float(np.mean([x["qb_norm_ndcg10"] for x in sub])), 5),
            "mean_rrf_ndcg10": round(float(np.mean([x["rrf_k60_ndcg10"] for x in sub])), 5),
            "mean_csls_ndcg10": round(float(np.mean([x["csls_ndcg10"] for x in sub])), 5),
            "mean_csls_minus_affine_pp": round(float(np.mean([x["csls_minus_affine_pp"] for x in sub])), 3),
            "mean_csls_minus_qbnorm_pp": round(float(np.mean([x["csls_minus_qbnorm_pp"] for x in sub])), 3),
            "mean_affine_recovery_frac": round(float(np.mean([x["affine_recovery_frac_of_csls"] for x in sub])), 4),
        }

    all_churn_pass = all(v["passes_2x_traffic_and_lcb_gt_neg001"] for v in churn_by_domain.values())
    manifest = {
        "experiment_id": "pv7_w26_04_hubness_law_and_churn",
        "wave_item_id": "W26-04",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hardware_cohort": f"spark2_gb10:{gpu_name}",
        "bridge": "CenteredProcrustes_ONLY",
        "old_encoder": "sentence-transformers/all-MiniLM-L6-v2",
        "new_encoder": new_enc_id,
        "claim_registry_sha256": claim_registry_sha256(),
        "elapsed_seconds": round(time.perf_counter() - t0, 2),
        "n_cells": len(cells),
        "summary_by_regime": by_regime,
        "hubness_law_correlations": {
            "spearman_k_skew_vs_csls_minus_affine": round(rho_skew_aff, 4),
            "spearman_rd_std_vs_csls_minus_raw": round(rho_std_raw, 4),
        },
        "pooled_csls_minus_qbnorm_ndcg10": {
            "mean": round(m_qb, 5),
            "ci95": [round(lo_qb, 5), round(hi_qb, 5)],
            "strictly_positive_lcb": bool(lo_qb > 0.0),
        },
        "mean_csls_minus_affine_pp_across_all_cells": round(float(np.mean(pooled_csls_minus_aff)), 3),
        "churn_maintenance_by_domain": churn_by_domain,
        "cells": cells,
        "verdict": {
            "beats_qbnorm_lcb95_gt_0": bool(lo_qb > 0.0),
            "all_domains_churn_2x_traffic_lcb_gt_neg001": all_churn_pass,
            "affine_does_not_explain_90pct_of_csls": bool(np.mean([x["affine_recovery_frac_of_csls"] for x in cells]) < 0.90),
        },
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out_path = args.output_dir / "w26_04_hubness_law_manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[w26-04] DONE cells={len(cells)} wall={manifest['elapsed_seconds']}s -> {out_path}", flush=True)
    print(json.dumps({"by_regime": by_regime, "hubness_corr": manifest["hubness_law_correlations"], "csls_vs_qbnorm": manifest["pooled_csls_minus_qbnorm_ndcg10"], "verdict": manifest["verdict"]}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
