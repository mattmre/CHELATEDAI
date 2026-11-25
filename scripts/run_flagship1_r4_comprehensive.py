"""Flagship #1 R4 Comprehensive Side-by-Side Validation (Original Baselines + Refined & Emergent Arms).

Runs with ZERO GPU re-encoding over persisted Spark caches (`/home/mattmre/chelated-night1/cache/`)
covering ALL Council R3 specifications side-by-side with ALL original baselines:
  - Suite 1A: 4096-d Extreme Scale (`D384/D1024/D2560/D4096` on SciFact, CoSQA_Code, Legal_Contracts, Legal_QuAD)
    * Direct up-projections (384->1024/2560/4096 at n=2000 & n=500 low-anchor) & down-projection (4096->384)
    * 3-Hop Extreme Chain (4096->2560->1024->384) at BOTH n=1000 (rank-deficit reproduction) & n=2000
    * Interior-Peak Chains (384->1024->384 & 384->4096->384): Theorem 1 Raw vs A2c-Polar
    * 18-Cell Zero-Shot Cross-Domain OOD ({SciFact, CoSQA_Code} -> {Legal_Contracts, Legal_QuAD, CrossDomain})
  - Suite 1B: 6-BEIR Multi-Hop (E4->E3->E2->E1->E0), Interior-Peak (E0->E3->E1), A2g, EM-2, EM-5b, & B4 Backfill
    (comparing legacy BF20 hard clip vs B4 smooth right-tail + CSLS and B2 full-encode vs B2-OldSpace zero-encode!)

Emits per-query CSVs, summary CSVs, manifest JSON, and 100%-coverage SHA-256 `report_receipt.json`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from datasets import load_dataset
from huggingface_hub import hf_hub_download

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from m1_baseline_ladder import (  # noqa: E402
    DEFAULT_RIDGE_LAMBDAS,
    compute_fastfill_backfill_order,
    fit_closed_form_centered_ridge,
    fit_closed_form_ridge,
    score_dense_rankings,
    split_query_ids,
)
from scripts.night1_common import (  # noqa: E402
    ENCODERS,
    current_git_sha,
    encode_or_load_embeddings,
    load_beir_dataset,
    paired_bootstrap_delta,
    process_telemetry,
    sha256_json,
    write_json_atomic,
)
from spherical_migration_bridge import (  # noqa: E402
    CenteredProcrustesBridge,
    GeodesicPolarBridge,
    WhitenedSphericalProcrustesBridge,
    combine_exposure_and_distortion_order,
    compose_centered_bridges,
    compute_old_space_knn_distortion,
    compute_spherical_distortion_scores,
    compute_voronoi_anchor_weights,
    l2_normalize,
    score_quantile_calibrated_backfill,
    score_zeromu_rankings,
)


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_per_query_ndcg_both(
    rankings: Dict[str, List[str]],
    qrels: Dict[str, Dict[str, float]],
    query_ids: Sequence[str],
    k: int = 10,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Return `(linear_gain_by_qid, exp_gain_2^r_minus_1_by_qid)` with complete-qrels IDCG."""
    lin_map: Dict[str, float] = {}
    exp_map: Dict[str, float] = {}
    for qid in query_ids:
        rel_map = {d: float(v) for d, v in qrels.get(qid, {}).items() if float(v) > 0.0}
        if not rel_map:
            continue
        ranked = rankings.get(qid, [])[:k]
        dcg_l = sum(rel_map.get(d, 0.0) / math.log2(i + 2.0) for i, d in enumerate(ranked))
        idcg_l = sum(r / math.log2(i + 2.0) for i, r in enumerate(sorted(rel_map.values(), reverse=True)[:k]))
        lin_map[qid] = dcg_l / idcg_l if idcg_l > 0 else 0.0
        dcg_e = sum(((2.0 ** rel_map.get(d, 0.0)) - 1.0) / math.log2(i + 2.0) for i, d in enumerate(ranked))
        idcg_e = sum(
            ((2.0 ** r) - 1.0) / math.log2(i + 2.0)
            for i, r in enumerate(sorted(rel_map.values(), reverse=True)[:k])
        )
        exp_map[qid] = dcg_e / idcg_e if idcg_e > 0 else 0.0
    return lin_map, exp_map


def _mean_ndcg(rankings: Dict[str, List[str]], qrels: Dict[str, Dict[str, float]], qids: Sequence[str]) -> float:
    lin, _ = compute_per_query_ndcg_both(rankings, qrels, qids, k=10)
    return float(np.mean(list(lin.values()))) if lin else 0.0


def _tune_ridge_uncentered(d_src, d_tgt, q_dev, dev_qids, doc_ids, qrels, fit_idx):
    best_w, best_lam, best_sc = None, 1.0, -1.0
    for lam in DEFAULT_RIDGE_LAMBDAS:
        w = fit_closed_form_ridge(d_src, d_tgt, fit_idx=fit_idx, regularization=lam)
        rk, _ = score_dense_rankings(q_dev @ w, d_tgt, dev_qids, doc_ids, k=10)
        sc = _mean_ndcg(rk, qrels, dev_qids)
        if sc > best_sc:
            best_sc, best_w, best_lam = sc, w, lam
    assert best_w is not None
    return best_w, best_lam


def _tune_ridge_centered(d_src, d_tgt, q_dev, dev_qids, doc_ids, qrels, fit_idx):
    best_t, best_lam, best_sc = None, 1.0, -1.0
    for lam in DEFAULT_RIDGE_LAMBDAS:
        mu_x, mu_y, w = fit_closed_form_centered_ridge(d_src, d_tgt, fit_idx=fit_idx, regularization=lam)
        rk, _ = score_dense_rankings((q_dev - mu_x) @ w + mu_y, d_tgt, dev_qids, doc_ids, k=10)
        sc = _mean_ndcg(rk, qrels, dev_qids)
        if sc > best_sc:
            best_sc, best_t, best_lam = sc, (mu_x, mu_y, w), lam
    assert best_t is not None
    return best_t, best_lam


def _tune_soft_spec(d_src, d_tgt, q_dev, dev_qids, doc_ids, qrels, fit_idx, cal_idx):
    best_b, best_tau, best_sc = None, 0.0, -1.0
    for tau in (0.0, 0.02, 0.05, 0.15, 0.50):
        br = CenteredProcrustesBridge.fit(
            d_src[fit_idx], d_tgt[fit_idx],
            cal_x=d_src[cal_idx], cal_y=d_tgt[cal_idx],
            scale_mode="max_one_s_full_oos",
            soft_spec_tau=tau,
        )
        rk, _ = score_dense_rankings(br.transform(q_dev), d_tgt, dev_qids, doc_ids, k=10)
        sc = _mean_ndcg(rk, qrels, dev_qids)
        if sc > best_sc:
            best_sc, best_b, best_tau = sc, br, tau
    assert best_b is not None
    return best_b, best_tau


def load_ood_domains_and_cache(cache_dir: Path) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Dict[str, np.ndarray]]]]:
    out: Dict[str, Dict[str, Any]] = {}
    dids, dtxts, qids, qtxts, qrels, _ = load_beir_dataset("SciFact", max_docs=None)
    out["SciFact"] = {"doc_ids": dids, "doc_texts": dtxts, "query_ids": qids, "query_texts": qtxts, "qrels": qrels}

    c_ds = load_dataset("CoIR-Retrieval/cosqa", "corpus", split="corpus")
    q_ds = load_dataset("CoIR-Retrieval/cosqa", "queries", split="queries")
    r_test = load_dataset("CoIR-Retrieval/cosqa", "default", split="test")
    r_val = load_dataset("CoIR-Retrieval/cosqa", "default", split="valid")
    q_lookup = {str(row["_id"]): str(row["text"]) for row in q_ds}
    cosqa_qrels: Dict[str, Dict[str, float]] = {}
    for split_r in (r_val, r_test):
        for row in split_r:
            sc = float(row["score"])
            if sc > 0:
                cosqa_qrels.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = sc
    cosqa_qids = sorted(cosqa_qrels.keys())
    out["CoSQA_Code"] = {
        "doc_ids": [str(row["_id"]) for row in c_ds],
        "doc_texts": [str(row["text"])[:1000] for row in c_ds],
        "query_ids": cosqa_qids,
        "query_texts": [q_lookup[q] for q in cosqa_qids],
        "qrels": cosqa_qrels,
    }

    for label, repo in (
        ("Legal_Contracts", "mteb/legalbench_consumer_contracts_qa"),
        ("Legal_QuAD", "mteb/LegalQuAD"),
    ):
        pc = hf_hub_download(repo, "corpus.jsonl", repo_type="dataset")
        pq = hf_hub_download(repo, "queries.jsonl", repo_type="dataset")
        pr = hf_hub_download(repo, "qrels/test.jsonl", repo_type="dataset")
        dids_l, dtxts_l = [], []
        for line in open(pc, encoding="utf-8"):
            row = json.loads(line)
            dids_l.append(str(row["_id"]))
            dtxts_l.append((str(row.get("title", "")) + " " + str(row.get("text", ""))).strip()[:1200])
        q_map = {}
        for line in open(pq, encoding="utf-8"):
            row = json.loads(line)
            q_map[str(row["_id"])] = str(row["text"])
        qr: Dict[str, Dict[str, float]] = {}
        for line in open(pr, encoding="utf-8"):
            row = json.loads(line)
            sc = float(row["score"])
            if sc > 0:
                qr.setdefault(str(row["query-id"]), {})[str(row["corpus-id"])] = sc
        qids_l = sorted(k for k in qr if k in q_map)
        out[label] = {
            "doc_ids": dids_l,
            "doc_texts": dtxts_l,
            "query_ids": qids_l,
            "query_texts": [q_map[q] for q in qids_l],
            "qrels": {q: qr[q] for q in qids_l},
        }

    embs: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {d: {} for d in out}
    for mkey in ("D384", "D1024", "D2560", "D4096"):
        for dname in out:
            cpath = cache_dir / f"ood_{dname}_{mkey}.npz"
            if not cpath.exists():
                raise FileNotFoundError(f"Expected cached OOD embedding file missing: {cpath}")
            z = np.load(cpath)
            embs[dname][mkey] = {
                "docs": z["docs"].astype(np.float32),
                "queries": z["queries"].astype(np.float32),
            }
    return out, embs


def _eval_arm_suite(
    suite: str,
    cell_id: str,
    seed: int,
    arm_rankings: Dict[str, Dict[str, List[str]]],
    qrels: Dict[str, Dict[str, float]],
    eval_qids: Sequence[str],
    meta_by_arm: Dict[str, Dict[str, Any]],
    per_query_rows: List[Dict[str, Any]],
    summary_rows: List[Dict[str, Any]],
) -> Dict[str, Dict[str, float]]:
    lin_by_arm: Dict[str, Dict[str, float]] = {}
    exp_by_arm: Dict[str, Dict[str, float]] = {}
    for arm, rk in arm_rankings.items():
        lin_m, exp_m = compute_per_query_ndcg_both(rk, qrels, eval_qids, k=10)
        lin_by_arm[arm] = lin_m
        exp_by_arm[arm] = exp_m
        for qid, val in lin_m.items():
            per_query_rows.append(
                {
                    "suite": suite,
                    "cell_id": cell_id,
                    "seed": seed,
                    "arm": arm,
                    "query_id": qid,
                    "ndcg_lin": round(val, 6),
                    "ndcg_exp": round(exp_m[qid], 6),
                }
            )
        m_extra = meta_by_arm.get(arm, {})
        summary_rows.append(
            {
                "suite": suite,
                "cell_id": cell_id,
                "seed": seed,
                "arm": arm,
                "n_queries": len(lin_m),
                "ndcg_lin": round(float(np.mean(list(lin_m.values()))), 6) if lin_m else 0.0,
                "ndcg_exp": round(float(np.mean(list(exp_m.values()))), 6) if exp_m else 0.0,
                "meta_json": json.dumps(m_extra, sort_keys=True),
            }
        )
    return lin_by_arm


def run_suite_1a_4096_ood(
    cache_dir: Path,
    seeds: Sequence[int],
    per_query_rows: List[Dict[str, Any]],
    summary_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    print("[F1-R4] Loading 4096-d OOD cached embeddings...", flush=True)
    domains, embs = load_ood_domains_and_cache(cache_dir)
    agg: Dict[str, Any] = {"direct": {}, "chains": {}, "peaks": {}, "ood_18": {}}

    # Precompute splits per domain
    dom_splits = {}
    for dname, ds in domains.items():
        sp = split_query_ids(ds["query_ids"])
        q2i = {q: i for i, q in enumerate(ds["query_ids"])}
        dom_splits[dname] = {
            "dev_qids": sp["DEVELOPMENT"],
            "rep_qids": sp["REPORT"],
            "dev_idx": [q2i[q] for q in sp["DEVELOPMENT"]],
            "rep_idx": [q2i[q] for q in sp["REPORT"]],
        }

    # 1. Direct Extreme Dimensionality Jumps (n=2000 + low-anchor n=500)
    direct_specs = [
        ("SciFact", "D384", "D1024", 2000),
        ("SciFact", "D384", "D2560", 2000),
        ("SciFact", "D384", "D4096", 2000),
        ("SciFact", "D4096", "D384", 2000),
        ("SciFact", "D384", "D4096", 500),
        ("SciFact", "D4096", "D384", 500),
        ("CoSQA_Code", "D384", "D1024", 2000),
        ("CoSQA_Code", "D384", "D2560", 2000),
        ("CoSQA_Code", "D384", "D4096", 2000),
        ("CoSQA_Code", "D4096", "D384", 2000),
    ]
    for dname, sk, tk, n_anc in direct_specs:
        ds = domains[dname]
        sp = dom_splits[dname]
        d_src, q_src_all = embs[dname][sk]["docs"], embs[dname][sk]["queries"]
        d_tgt, q_tgt_all = embs[dname][tk]["docs"], embs[dname][tk]["queries"]
        q_dev = q_src_all[sp["dev_idx"]]
        q_rep = q_src_all[sp["rep_idx"]]
        q_rep_nat = q_tgt_all[sp["rep_idx"]]
        cell_id = f"{dname}_{sk}_to_{tk}_n{n_anc}"
        print(f"[F1-R4][Direct] {cell_id} (n_rep={len(sp['rep_qids'])})...", flush=True)

        rk_nat, _ = score_dense_rankings(q_rep_nat, d_tgt, sp["rep_qids"], ds["doc_ids"], k=10)
        seed_lin: Dict[str, List[Dict[str, float]]] = defaultdict(list)

        for seed in seeds:
            rng = np.random.default_rng(seed + n_anc)
            all_idx = rng.choice(len(ds["doc_ids"]), size=min(n_anc, len(ds["doc_ids"])), replace=False)
            n_fit = int(round(0.80 * len(all_idx)))
            fit_idx = np.sort(all_idx[:n_fit])
            cal_idx = np.sort(all_idx[n_fit:])
            full_idx = np.sort(all_idx)

            w_ur, lam_ur = _tune_ridge_uncentered(d_src, d_tgt, q_dev, sp["dev_qids"], ds["doc_ids"], ds["qrels"], full_idx)
            (mx_c, my_c, w_cr), lam_cr = _tune_ridge_centered(d_src, d_tgt, q_dev, sp["dev_qids"], ds["doc_ids"], ds["qrels"], full_idx)

            a2_unit = CenteredProcrustesBridge.fit(d_src, d_tgt, fit_idx=full_idx, scale_mode="unit")
            a2c_amb = CenteredProcrustesBridge.fit(d_src, d_tgt, fit_idx=full_idx, scale_mode="ambient")
            a2c_oos_full = CenteredProcrustesBridge.fit(
                d_src[fit_idx], d_tgt[fit_idx], cal_x=d_src[cal_idx], cal_y=d_tgt[cal_idx], scale_mode="s_full_oos"
            )
            a2c_oos_sub = CenteredProcrustesBridge.fit(
                d_src[fit_idx], d_tgt[fit_idx], cal_x=d_src[cal_idx], cal_y=d_tgt[cal_idx], scale_mode="s_sub_oos"
            )
            a2c_oos_max1 = CenteredProcrustesBridge.fit(
                d_src[fit_idx], d_tgt[fit_idx], cal_x=d_src[cal_idx], cal_y=d_tgt[cal_idx], scale_mode="max_one_s_full_oos"
            )
            a2c_soft, best_tau = _tune_soft_spec(d_src, d_tgt, q_dev, sp["dev_qids"], ds["doc_ids"], ds["qrels"], fit_idx, cal_idx)
            a2c_q = CenteredProcrustesBridge.fit(
                d_src[fit_idx], d_tgt[fit_idx], cal_x=d_src[cal_idx], cal_y=d_tgt[cal_idx],
                scale_mode="max_one_s_full_oos", mu_q_override=q_dev.mean(axis=0),
            )
            w_vor = compute_voronoi_anchor_weights(d_src, fit_idx)
            a2c_iw = CenteredProcrustesBridge.fit(
                d_src[fit_idx], d_tgt[fit_idx], cal_x=d_src[cal_idx], cal_y=d_tgt[cal_idx],
                scale_mode="max_one_s_full_oos", sample_weights=w_vor,
            )
            a2g_soft = GeodesicPolarBridge.fit(
                d_src[fit_idx], d_tgt[fit_idx], cal_x=d_src[cal_idx], cal_y=d_tgt[cal_idx], collar="c1_soft", min_resultant=0.15
            )
            a2g_hard = GeodesicPolarBridge.fit(
                d_src[fit_idx], d_tgt[fit_idx], cal_x=d_src[cal_idx], cal_y=d_tgt[cal_idx], collar="hard", min_resultant=0.15
            )
            a2g_025 = GeodesicPolarBridge.fit(
                d_src[fit_idx], d_tgt[fit_idx], cal_x=d_src[cal_idx], cal_y=d_tgt[cal_idx], collar="c1_soft", min_resultant=0.25
            )

            r_sub = min(d_src.shape[1], d_tgt.shape[1], len(full_idx) - 1)
            a3_full = WhitenedSphericalProcrustesBridge.fit(d_src, d_tgt, fit_idx=full_idx, shrink=0.20, subspace_rank=None, trace_divisor_mode="ambient")
            a3_sub_amb = WhitenedSphericalProcrustesBridge.fit(d_src, d_tgt, fit_idx=full_idx, shrink=0.20, subspace_rank=r_sub, trace_divisor_mode="ambient")
            a3_sub_act = WhitenedSphericalProcrustesBridge.fit(d_src, d_tgt, fit_idx=full_idx, shrink=0.20, subspace_rank=r_sub, trace_divisor_mode="active_rank")
            a3_full_eff = WhitenedSphericalProcrustesBridge.fit(d_src, d_tgt, fit_idx=full_idx, shrink=0.20, subspace_rank=None, trace_divisor_mode="effective_rank")
            a3_sub_eff = WhitenedSphericalProcrustesBridge.fit(d_src, d_tgt, fit_idx=full_idx, shrink=0.20, subspace_rank=r_sub, trace_divisor_mode="effective_rank")

            q_g_soft, g_soft_stats = a2g_soft.transform_with_stats(q_rep)
            q_g_hard, g_hard_stats = a2g_hard.transform_with_stats(q_rep)
            q_g_025, g_025_stats = a2g_025.transform_with_stats(q_rep)

            arm_rk = {
                "Native_Target": rk_nat,
                "Uncentered_Ridge": score_dense_rankings(q_rep @ w_ur, d_tgt, sp["rep_qids"], ds["doc_ids"], k=10)[0],
                "Centered_Affine_Ridge": score_dense_rankings((q_rep - mx_c) @ w_cr + my_c, d_tgt, sp["rep_qids"], ds["doc_ids"], k=10)[0],
                "A2_Unscaled": score_dense_rankings(a2_unit.transform(q_rep), d_tgt, sp["rep_qids"], ds["doc_ids"], k=10)[0],
                "A2c_Ambient": score_dense_rankings(a2c_amb.transform(q_rep), d_tgt, sp["rep_qids"], ds["doc_ids"], k=10)[0],
                "A2c_OOS_Full": score_dense_rankings(a2c_oos_full.transform(q_rep), d_tgt, sp["rep_qids"], ds["doc_ids"], k=10)[0],
                "A2c_OOS_Sub": score_dense_rankings(a2c_oos_sub.transform(q_rep), d_tgt, sp["rep_qids"], ds["doc_ids"], k=10)[0],
                "A2c_OOS_Max1": score_dense_rankings(a2c_oos_max1.transform(q_rep), d_tgt, sp["rep_qids"], ds["doc_ids"], k=10)[0],
                "A2_ZeroMu_Query": score_zeromu_rankings(q_rep, d_tgt, a2c_oos_max1, sp["rep_qids"], ds["doc_ids"], mode="query_only", k=10)[0],
                "A2_ZeroMu_DualFit": score_zeromu_rankings(q_rep, d_tgt, a2c_oos_max1, sp["rep_qids"], ds["doc_ids"], mode="dual_fit", k=10)[0],
                "A2_ZeroMu_DualAdap": score_zeromu_rankings(q_rep, d_tgt, a2c_oos_max1, sp["rep_qids"], ds["doc_ids"], mode="dual_adap", k=10)[0],
                "A2c_SoftSpec": score_dense_rankings(a2c_soft.transform(q_rep), d_tgt, sp["rep_qids"], ds["doc_ids"], k=10)[0],
                "A2c_Q": score_dense_rankings(a2c_q.transform(q_rep), d_tgt, sp["rep_qids"], ds["doc_ids"], k=10)[0],
                "A2c_IW": score_dense_rankings(a2c_iw.transform(q_rep), d_tgt, sp["rep_qids"], ds["doc_ids"], k=10)[0],
                "A2g_C1Soft": score_dense_rankings(q_g_soft, d_tgt, sp["rep_qids"], ds["doc_ids"], k=10)[0],
                "A2g_Hard": score_dense_rankings(q_g_hard, d_tgt, sp["rep_qids"], ds["doc_ids"], k=10)[0],
                "A2g_Floor025": score_dense_rankings(q_g_025, d_tgt, sp["rep_qids"], ds["doc_ids"], k=10)[0],
                "A3_Full": score_dense_rankings(a3_full.transform(q_rep), d_tgt, sp["rep_qids"], ds["doc_ids"], k=10)[0],
                "A3_Subspace_Ambient": score_dense_rankings(a3_sub_amb.transform(q_rep), d_tgt, sp["rep_qids"], ds["doc_ids"], k=10)[0],
                "A3_Subspace_ActiveRank": score_dense_rankings(a3_sub_act.transform(q_rep), d_tgt, sp["rep_qids"], ds["doc_ids"], k=10)[0],
                "A3_Full_EffRank": score_dense_rankings(a3_full_eff.transform(q_rep), d_tgt, sp["rep_qids"], ds["doc_ids"], k=10)[0],
                "A3_Subspace_EffRank": score_dense_rankings(a3_sub_eff.transform(q_rep), d_tgt, sp["rep_qids"], ds["doc_ids"], k=10)[0],
            }
            meta = {
                "Uncentered_Ridge": {"lam": lam_ur},
                "Centered_Affine_Ridge": {"lam": lam_cr},
                "A2c_OOS_Full": {"scale": round(a2c_oos_full.scale, 4)},
                "A2c_OOS_Sub": {"scale": round(a2c_oos_sub.scale, 4)},
                "A2c_OOS_Max1": {"scale": round(a2c_oos_max1.scale, 4)},
                "A2c_SoftSpec": {"tau": best_tau, "scale": round(a2c_soft.scale, 4)},
                "A2g_C1Soft": {
                    "s_theta": round(a2g_soft.s_theta, 4),
                    "res_x": round(a2g_soft.resultant_x, 4),
                    "res_y": round(a2g_soft.resultant_y, 4),
                    "drift_x": round(a2g_soft.split_half_drift_x_deg, 2),
                    "drift_y": round(a2g_soft.split_half_drift_y_deg, 2),
                    "used_fb": a2g_soft.used_fallback,
                    "frac_clip": round(g_soft_stats["frac_clipped"], 4),
                },
                "A2g_Hard": {"used_fb": a2g_hard.used_fallback, "frac_clip": round(g_hard_stats["frac_clipped"], 4)},
                "A2g_Floor025": {"used_fb": a2g_025.used_fallback, "reason": a2g_025.fallback_reason, "frac_clip": round(g_025_stats["frac_clipped"], 4)},
            }
            lin_by_arm = _eval_arm_suite("1A_direct", cell_id, seed, arm_rk, ds["qrels"], sp["rep_qids"], meta, per_query_rows, summary_rows)
            for arm, lm in lin_by_arm.items():
                seed_lin[arm].append(lm)

        # Average across seeds per query for paired bootstrap vs Centered_Affine_Ridge
        mean_per_q = {
            arm: {qid: float(np.mean([m[qid] for m in ms])) for qid in ms[0]}
            for arm, ms in seed_lin.items()
        }
        car_q = mean_per_q["Centered_Affine_Ridge"]
        agg["direct"][cell_id] = {
            "n_queries": len(car_q),
            "means": {arm: round(float(np.mean(list(m.values()))), 4) for arm, m in mean_per_q.items()},
            "vs_CAR": {
                arm: {
                    "delta": round(paired_bootstrap_delta(m, car_q, n_bootstrap=2000, seed=42)["mean_delta"], 4),
                    "lcb95": round(paired_bootstrap_delta(m, car_q, n_bootstrap=2000, seed=42)["lcb_95"], 4),
                }
                for arm, m in mean_per_q.items()
                if arm not in ("Native_Target", "Centered_Affine_Ridge")
            },
        }

    # 2. 3-Hop Extreme Chain (D4096 -> D2560 -> D1024 -> D384) at BOTH n=1000 & n=2000!
    chain_models = ["D384", "D1024", "D2560", "D4096"]
    for dname in ("SciFact", "CoSQA_Code"):
        ds = domains[dname]
        sp = dom_splits[dname]
        d0 = embs[dname]["D384"]["docs"]
        for n_anc in (1000, 2000):
            cell_prefix = f"{dname}_chain4096_n{n_anc}"
            print(f"[F1-R4][3-Hop] {cell_prefix}...", flush=True)
            hop_seed_lin: Dict[int, Dict[str, List[Dict[str, float]]]] = {h: defaultdict(list) for h in (1, 2, 3)}
            for seed in seeds:
                car_hops = []
                a2c_hops = []
                a2c_soft_hops = []
                for step in range(1, len(chain_models)):
                    sk, tk = chain_models[step], chain_models[step - 1]
                    rng = np.random.default_rng(seed + step * 97 + n_anc)
                    all_i = rng.choice(len(ds["doc_ids"]), size=min(n_anc, len(ds["doc_ids"])), replace=False)
                    n_f = int(round(0.80 * len(all_i)))
                    f_i, c_i, full_i = np.sort(all_i[:n_f]), np.sort(all_i[n_f:]), np.sort(all_i)
                    q_dev = embs[dname][sk]["queries"][sp["dev_idx"]]
                    (mx, my, w), _ = _tune_ridge_centered(
                        embs[dname][sk]["docs"], embs[dname][tk]["docs"], q_dev, sp["dev_qids"], ds["doc_ids"], ds["qrels"], full_i
                    )
                    car_hops.append((mx, my, w))
                    a2c_hops.append(
                        CenteredProcrustesBridge.fit(
                            embs[dname][sk]["docs"][f_i], embs[dname][tk]["docs"][f_i],
                            cal_x=embs[dname][sk]["docs"][c_i], cal_y=embs[dname][tk]["docs"][c_i],
                            scale_mode="s_full_oos",
                        )
                    )
                    b_soft, _ = _tune_soft_spec(
                        embs[dname][sk]["docs"], embs[dname][tk]["docs"], q_dev, sp["dev_qids"], ds["doc_ids"], ds["qrels"], f_i, c_i
                    )
                    a2c_soft_hops.append(b_soft)

                for hop_m in (1, 2, 3):
                    sk = chain_models[hop_m]
                    q_rep = embs[dname][sk]["queries"][sp["rep_idx"]]
                    # Composed Centered Affine Ridge
                    z_car = q_rep.astype(np.float64)
                    for step_idx in reversed(range(hop_m)):
                        mx, my, w = car_hops[step_idx]
                        z_car = (z_car - mx) @ w + my
                    # Composed A2c (exact algebraic Stiefel composition) & SoftSpec
                    comp_a2c = compose_centered_bridges(list(reversed(a2c_hops[:hop_m])), polar_reproject=False)
                    comp_soft = compose_centered_bridges(list(reversed(a2c_soft_hops[:hop_m])), polar_reproject=False)
                    arm_rk = {
                        "Composed_CAR": score_dense_rankings(z_car, d0, sp["rep_qids"], ds["doc_ids"], k=10)[0],
                        "Composed_A2c_OOS": score_dense_rankings(comp_a2c.transform(q_rep), d0, sp["rep_qids"], ds["doc_ids"], k=10)[0],
                        "Composed_A2c_SoftSpec": score_dense_rankings(comp_soft.transform(q_rep), d0, sp["rep_qids"], ds["doc_ids"], k=10)[0],
                        "Composed_ZeroMu_DualFit": score_zeromu_rankings(q_rep, d0, comp_a2c, sp["rep_qids"], ds["doc_ids"], mode="dual_fit", k=10)[0],
                    }
                    meta = {
                        "Composed_A2c_OOS": {"cond": round(comp_a2c.condition_number, 4), "regime": comp_a2c.regime},
                        "Composed_A2c_SoftSpec": {"cond": round(comp_soft.condition_number, 4)},
                    }
                    lm = _eval_arm_suite(
                        "1A_3hop", f"{cell_prefix}_hop{hop_m}", seed, arm_rk, ds["qrels"], sp["rep_qids"], meta, per_query_rows, summary_rows
                    )
                    for arm, m in lm.items():
                        hop_seed_lin[hop_m][arm].append(m)

            agg["chains"][cell_prefix] = {}
            for hop_m in (1, 2, 3):
                mpq = {arm: {q: float(np.mean([m[q] for m in ms])) for q in ms[0]} for arm, ms in hop_seed_lin[hop_m].items()}
                agg["chains"][cell_prefix][f"hop{hop_m}"] = {
                    arm: round(float(np.mean(list(m.values()))), 4) for arm, m in mpq.items()
                }

    # 3. Theorem 1 Interior-Peak Chains (D384 -> D1024 -> D384 AND D384 -> D4096 -> D384)
    for dname in ("SciFact", "CoSQA_Code"):
        ds = domains[dname]
        sp = dom_splits[dname]
        for mid in ("D1024", "D4096"):
            cell_id = f"{dname}_peak_D384_{mid}_D384"
            print(f"[F1-R4][Peak] {cell_id}...", flush=True)
            seed_lin = defaultdict(list)
            peak_diag = {}
            for seed in seeds:
                rng = np.random.default_rng(seed + 777)
                idx = np.sort(rng.choice(len(ds["doc_ids"]), size=min(1500, len(ds["doc_ids"])), replace=False))
                b1 = CenteredProcrustesBridge.fit(embs[dname]["D384"]["docs"], embs[dname][mid]["docs"], fit_idx=idx, scale_mode="ambient")
                b2 = CenteredProcrustesBridge.fit(embs[dname][mid]["docs"], embs[dname]["D384"]["docs"], fit_idx=idx, scale_mode="ambient")
                c_raw = compose_centered_bridges([b1, b2], polar_reproject=False)
                c_pol = compose_centered_bridges([b1, b2], polar_reproject=True)
                q_rep = embs[dname]["D384"]["queries"][sp["rep_idx"]]
                d_tgt = embs[dname]["D384"]["docs"]
                arm_rk = {
                    "Peak_A2c_Raw": score_dense_rankings(c_raw.transform(q_rep), d_tgt, sp["rep_qids"], ds["doc_ids"], k=10)[0],
                    "Peak_A2c_Polar": score_dense_rankings(c_pol.transform(q_rep), d_tgt, sp["rep_qids"], ds["doc_ids"], k=10)[0],
                }
                peak_diag = {
                    "regime": c_raw.regime,
                    "raw_cond": round(c_raw.condition_number, 4),
                    "polar_cond": round(c_pol.condition_number, 6),
                    "pre_polar_sigma_min": round(c_pol.pre_polar_sigma_min, 4),
                }
                lm = _eval_arm_suite("1A_peak", cell_id, seed, arm_rk, ds["qrels"], sp["rep_qids"], {"Peak_A2c_Polar": peak_diag}, per_query_rows, summary_rows)
                for arm, m in lm.items():
                    seed_lin[arm].append(m)
            mpq = {arm: {q: float(np.mean([m[q] for m in ms])) for q in ms[0]} for arm, ms in seed_lin.items()}
            agg["peaks"][cell_id] = {
                "means": {arm: round(float(np.mean(list(m.values()))), 4) for arm, m in mpq.items()},
                "diag": peak_diag,
            }

    # 4. Full 18-Cell Zero-Shot Cross-Domain OOD ({SciFact, CoSQA_Code} -> {Legal_Contracts, Legal_QuAD, CrossDomain})
    ood_pairs = [
        ("SciFact", "Legal_Contracts"),
        ("SciFact", "Legal_QuAD"),
        ("SciFact", "CoSQA_Code"),
        ("CoSQA_Code", "Legal_Contracts"),
        ("CoSQA_Code", "Legal_QuAD"),
        ("CoSQA_Code", "SciFact"),
    ]
    ood_dirs = [("D384", "D4096"), ("D4096", "D384"), ("D1024", "D4096")]
    for src_dom, eval_dom in ood_pairs:
        src_ds = domains[src_dom]
        ev_ds = domains[eval_dom]
        src_sp = dom_splits[src_dom]
        for sk, tk in ood_dirs:
            cell_id = f"OOD_{src_dom}_to_{eval_dom}_{sk}_to_{tk}"
            print(f"[F1-R4][OOD] {cell_id} (n_eval_q={len(ev_ds['query_ids'])})...", flush=True)
            d_src_fit, d_tgt_fit = embs[src_dom][sk]["docs"], embs[src_dom][tk]["docs"]
            q_src_dev = embs[src_dom][sk]["queries"][src_sp["dev_idx"]]
            q_ev_src = embs[eval_dom][sk]["queries"]
            d_ev_tgt = embs[eval_dom][tk]["docs"]
            q_ev_nat = embs[eval_dom][tk]["queries"]
            rk_nat, _ = score_dense_rankings(q_ev_nat, d_ev_tgt, ev_ds["query_ids"], ev_ds["doc_ids"], k=10)

            seed_lin = defaultdict(list)
            for seed in seeds:
                rng = np.random.default_rng(seed + 2026)
                n_anc = min(2000, len(src_ds["doc_ids"]))
                all_i = rng.choice(len(src_ds["doc_ids"]), size=n_anc, replace=False)
                n_f = int(round(0.80 * n_anc))
                f_i, c_i, full_i = np.sort(all_i[:n_f]), np.sort(all_i[n_f:]), np.sort(all_i)

                (mx_c, my_c, w_cr), _ = _tune_ridge_centered(
                    d_src_fit, d_tgt_fit, q_src_dev, src_sp["dev_qids"], src_ds["doc_ids"], src_ds["qrels"], full_i
                )
                a2_unit = CenteredProcrustesBridge.fit(d_src_fit, d_tgt_fit, fit_idx=full_i, scale_mode="unit")
                a2c_amb = CenteredProcrustesBridge.fit(d_src_fit, d_tgt_fit, fit_idx=full_i, scale_mode="ambient")
                a2c_max1 = CenteredProcrustesBridge.fit(
                    d_src_fit[f_i], d_tgt_fit[f_i], cal_x=d_src_fit[c_i], cal_y=d_tgt_fit[c_i], scale_mode="max_one_s_full_oos"
                )
                a2g_soft = GeodesicPolarBridge.fit(
                    d_src_fit[f_i], d_tgt_fit[f_i], cal_x=d_src_fit[c_i], cal_y=d_tgt_fit[c_i], collar="c1_soft", min_resultant=0.15
                )
                r_sub = min(d_src_fit.shape[1], d_tgt_fit.shape[1], len(full_i) - 1)
                a3_full = WhitenedSphericalProcrustesBridge.fit(d_src_fit, d_tgt_fit, fit_idx=full_i, shrink=0.20)
                a3_sub_eff = WhitenedSphericalProcrustesBridge.fit(
                    d_src_fit, d_tgt_fit, fit_idx=full_i, shrink=0.20, subspace_rank=r_sub, trace_divisor_mode="effective_rank"
                )

                arm_rk = {
                    "Native_Target": rk_nat,
                    "Centered_Affine_Ridge": score_dense_rankings((q_ev_src - mx_c) @ w_cr + my_c, d_ev_tgt, ev_ds["query_ids"], ev_ds["doc_ids"], k=10)[0],
                    "A2_Unscaled": score_dense_rankings(a2_unit.transform(q_ev_src), d_ev_tgt, ev_ds["query_ids"], ev_ds["doc_ids"], k=10)[0],
                    "A2c_Ambient": score_dense_rankings(a2c_amb.transform(q_ev_src), d_ev_tgt, ev_ds["query_ids"], ev_ds["doc_ids"], k=10)[0],
                    "A2c_OOS_Max1": score_dense_rankings(a2c_max1.transform(q_ev_src), d_ev_tgt, ev_ds["query_ids"], ev_ds["doc_ids"], k=10)[0],
                    "A2_ZeroMu_Query": score_zeromu_rankings(q_ev_src, d_ev_tgt, a2c_max1, ev_ds["query_ids"], ev_ds["doc_ids"], mode="query_only", k=10)[0],
                    "A2_ZeroMu_DualFit": score_zeromu_rankings(q_ev_src, d_ev_tgt, a2c_max1, ev_ds["query_ids"], ev_ds["doc_ids"], mode="dual_fit", k=10)[0],
                    "A2_ZeroMu_DualAdap": score_zeromu_rankings(q_ev_src, d_ev_tgt, a2c_max1, ev_ds["query_ids"], ev_ds["doc_ids"], mode="dual_adap", k=10)[0],
                    "A2g_C1Soft": score_dense_rankings(a2g_soft.transform(q_ev_src), d_ev_tgt, ev_ds["query_ids"], ev_ds["doc_ids"], k=10)[0],
                    "A3_Full": score_dense_rankings(a3_full.transform(q_ev_src), d_ev_tgt, ev_ds["query_ids"], ev_ds["doc_ids"], k=10)[0],
                    "A3_Subspace_EffRank": score_dense_rankings(a3_sub_eff.transform(q_ev_src), d_ev_tgt, ev_ds["query_ids"], ev_ds["doc_ids"], k=10)[0],
                }
                lm = _eval_arm_suite("1A_ood18", cell_id, seed, arm_rk, ev_ds["qrels"], ev_ds["query_ids"], {}, per_query_rows, summary_rows)
                for arm, m in lm.items():
                    seed_lin[arm].append(m)
            mpq = {arm: {q: float(np.mean([m[q] for m in ms])) for q in ms[0]} for arm, ms in seed_lin.items()}
            agg["ood_18"][cell_id] = {arm: round(float(np.mean(list(m.values()))), 4) for arm, m in mpq.items()}
    return agg


def run_suite_1b_beir_and_b4(
    cache_dir: Path,
    seeds: Sequence[int],
    per_query_rows: List[Dict[str, Any]],
    summary_rows: List[Dict[str, Any]],
) -> Dict[str, Any]:
    datasets = ("SciFact", "NFCorpus", "FiQA2018", "ArguAna", "SCIDOCS", "Quora")
    chain = ("E0", "E1", "E2", "E3", "E4")
    agg: Dict[str, Any] = {"beir_4hop": {}, "beir_peak_e0_e3_e1": {}, "beir_b4_backfill": {}}

    for ds_name in datasets:
        print(f"[F1-R4][BEIR] Loading {ds_name}...", flush=True)
        doc_cap = 40_000 if ds_name in ("Quora", "FiQA2018") else None
        doc_ids, doc_texts, query_ids, query_texts, qrels, _ = load_beir_dataset(ds_name, max_docs=doc_cap)
        if len(query_ids) > 1200:
            keep_q = set(sorted(query_ids, key=lambda q: sha256_json({"ds": ds_name, "q": q}))[:1200])
            qp = [(q, t) for q, t in zip(query_ids, query_texts) if q in keep_q]
            query_ids, query_texts = [p[0] for p in qp], [p[1] for p in qp]
            qrels = {q: qrels[q] for q in query_ids}
        sp = split_query_ids(query_ids)
        q2i = {q: i for i, q in enumerate(query_ids)}
        dev_idx = [q2i[q] for q in sp["DEVELOPMENT"]]
        rep_idx = [q2i[q] for q in sp["REPORT"]]

        enc = {}
        for ek in chain:
            de, qe, _ = encode_or_load_embeddings(
                dataset_name=ds_name, encoder_key=ek, encoder_name=ENCODERS[ek],
                doc_ids=doc_ids, doc_texts=doc_texts, query_ids=query_ids, query_texts=query_texts,
                cache_dir=cache_dir,
            )
            enc[ek] = {"docs": de, "queries": qe}

        # 1. Direct E3 (768) -> E0 (384) & 4-Hop E4 -> E3 -> E2 -> E1 -> E0 comparing A2c, A2g, EM-2, EM-5b
        d0 = enc["E0"]["docs"]
        n_mh = min(500, len(doc_ids) // 2)
        hop4_lin: Dict[str, List[Dict[str, float]]] = defaultdict(list)
        peak_lin: Dict[str, List[Dict[str, float]]] = defaultdict(list)
        peak_diag = {}

        for seed in seeds:
            car_hops, a2c_hops, a2g_hops = [], [], []
            for step in range(1, len(chain)):
                sk, tk = chain[step], chain[step - 1]
                rng = np.random.default_rng(seed + step * 97)
                all_i = rng.choice(len(doc_ids), size=n_mh, replace=False)
                n_f = int(round(0.80 * n_mh))
                f_i, c_i, full_i = np.sort(all_i[:n_f]), np.sort(all_i[n_f:]), np.sort(all_i)
                (mx, my, w), _ = _tune_ridge_centered(
                    enc[sk]["docs"], enc[tk]["docs"], enc[sk]["queries"][dev_idx], sp["DEVELOPMENT"], doc_ids, qrels, full_i
                )
                car_hops.append((mx, my, w))
                a2c_hops.append(
                    CenteredProcrustesBridge.fit(
                        enc[sk]["docs"][f_i], enc[tk]["docs"][f_i],
                        cal_x=enc[sk]["docs"][c_i], cal_y=enc[tk]["docs"][c_i],
                        scale_mode="s_full_oos",
                    )
                )
                a2g_hops.append(
                    GeodesicPolarBridge.fit(
                        enc[sk]["docs"][f_i], enc[tk]["docs"][f_i],
                        cal_x=enc[sk]["docs"][c_i], cal_y=enc[tk]["docs"][c_i],
                        collar="c1_soft", min_resultant=0.15,
                    )
                )

            # Evaluate Hop-4 (E4 -> E3 -> E2 -> E1 -> E0)
            q4 = enc["E4"]["queries"][rep_idx]
            z_car = q4.astype(np.float64)
            for s_i in reversed(range(4)):
                mx, my, w = car_hops[s_i]
                z_car = (z_car - mx) @ w + my
            z_a2g = q4.astype(np.float64)
            for s_i in reversed(range(4)):
                z_a2g = a2g_hops[s_i].transform(z_a2g)
            comp_a2c = compose_centered_bridges(list(reversed(a2c_hops)), polar_reproject=False)
            arm_rk4 = {
                "Hop4_CAR": score_dense_rankings(z_car, d0, sp["REPORT"], doc_ids, k=10)[0],
                "Hop4_A2c_OOS": score_dense_rankings(comp_a2c.transform(q4), d0, sp["REPORT"], doc_ids, k=10)[0],
                "Hop4_A2g_C1Soft": score_dense_rankings(z_a2g, d0, sp["REPORT"], doc_ids, k=10)[0],
                "Hop4_ZeroMu_Query": score_zeromu_rankings(q4, d0, comp_a2c, sp["REPORT"], doc_ids, mode="query_only", k=10)[0],
                "Hop4_ZeroMu_DualFit": score_zeromu_rankings(q4, d0, comp_a2c, sp["REPORT"], doc_ids, mode="dual_fit", k=10)[0],
                "Hop4_ZeroMu_DualAdap": score_zeromu_rankings(q4, d0, comp_a2c, sp["REPORT"], doc_ids, mode="dual_adap", k=10)[0],
            }
            lm4 = _eval_arm_suite("1B_beir_4hop", f"{ds_name}_hop4", seed, arm_rk4, qrels, sp["REPORT"], {}, per_query_rows, summary_rows)
            for arm, m in lm4.items():
                hop4_lin[arm].append(m)

            # Peak Chain: E0 (384) -> E3 (768) -> E1 (384)
            rng_p = np.random.default_rng(seed + 404)
            p_idx = np.sort(rng_p.choice(len(doc_ids), size=n_mh, replace=False))
            bp1 = CenteredProcrustesBridge.fit(enc["E0"]["docs"], enc["E3"]["docs"], fit_idx=p_idx, scale_mode="ambient")
            bp2 = CenteredProcrustesBridge.fit(enc["E3"]["docs"], enc["E1"]["docs"], fit_idx=p_idx, scale_mode="ambient")
            c_raw = compose_centered_bridges([bp1, bp2], polar_reproject=False)
            c_pol = compose_centered_bridges([bp1, bp2], polar_reproject=True)
            q_p = enc["E0"]["queries"][rep_idx]
            arm_rkp = {
                "Peak_E0_E3_E1_Raw": score_dense_rankings(c_raw.transform(q_p), enc["E1"]["docs"], sp["REPORT"], doc_ids, k=10)[0],
                "Peak_E0_E3_E1_Polar": score_dense_rankings(c_pol.transform(q_p), enc["E1"]["docs"], sp["REPORT"], doc_ids, k=10)[0],
            }
            peak_diag = {
                "raw_cond": round(c_raw.condition_number, 4),
                "polar_cond": round(c_pol.condition_number, 6),
                "pre_polar_sigma_min": round(c_pol.pre_polar_sigma_min, 4),
            }
            lmp = _eval_arm_suite("1B_beir_peak", f"{ds_name}_peak_E0_E3_E1", seed, arm_rkp, qrels, sp["REPORT"], {"Peak_E0_E3_E1_Polar": peak_diag}, per_query_rows, summary_rows)
            for arm, m in lmp.items():
                peak_lin[arm].append(m)

        agg["beir_4hop"][ds_name] = {
            arm: round(float(np.mean([np.mean(list(m.values())) for m in ms])), 4) for arm, ms in hop4_lin.items()
        }
        agg["beir_peak_e0_e3_e1"][ds_name] = {
            "means": {arm: round(float(np.mean([np.mean(list(m.values())) for m in ms])), 4) for arm, ms in peak_lin.items()},
            "diag": peak_diag,
        }

        # 2. B4 Backfill Validation (E3 -> E0 at 20% backfill) on SciFact, NFCorpus, ArguAna
        if ds_name in ("SciFact", "NFCorpus", "ArguAna"):
            bf_lin: Dict[str, List[Dict[str, float]]] = defaultdict(list)
            d_old, d_new = enc["E0"]["docs"], enc["E3"]["docs"]
            q_dev_new = enc["E3"]["queries"][dev_idx]
            q_rep_new = enc["E3"]["queries"][rep_idx]
            n_bf = max(1, int(round(0.20 * len(doc_ids))))
            for seed in seeds:
                rng_b = np.random.default_rng(seed + 808)
                a_idx = np.sort(rng_b.choice(len(doc_ids), size=n_mh, replace=False))
                br = CenteredProcrustesBridge.fit(d_new, d_old, fit_idx=a_idx, scale_mode="ambient")
                q_dev_map = br.transform(q_dev_new)
                q_rep_map = br.transform(q_rep_new)
                ff_order = compute_fastfill_backfill_order(q_dev_map, d_old)
                # B2 full-encode baseline (100% target docs encoded — oracle upper bound on distortion)
                mapped_all = br.transform(d_new)
                dist_full_ord, _ = compute_spherical_distortion_scores(d_new, d_old, mapped_all, anchor_pool_idx=a_idx)
                # B2-OldSpace ZERO-target-encode predictor (only anchor residuals!)
                anc_res = 1.0 - np.sum(l2_normalize(br.transform(d_new[a_idx])) * l2_normalize(d_old[a_idx]), axis=1)
                dist_old_ord, _ = compute_old_space_knn_distortion(d_old, a_idx, anc_res, k_neighbors=16)
                b3_full_ord = combine_exposure_and_distortion_order(ff_order, dist_full_ord, exposure_weight=0.5)
                b3_old_ord = combine_exposure_and_distortion_order(ff_order, dist_old_ord, exposure_weight=0.5)

                arm_rkb = {
                    "BF0_Bridge_Only": score_dense_rankings(q_rep_map, d_old, sp["REPORT"], doc_ids, k=10)[0],
                    "BF20_Legacy_HardClip_B3Full": score_quantile_calibrated_backfill(
                        q_rep_new, q_rep_map, d_old, d_new, b3_full_ord[:n_bf], sp["REPORT"], doc_ids,
                        dev_queries_new=q_dev_new, dev_queries_mapped=q_dev_map, k=10, csls_k=0, legacy_hard_clip=True,
                    ),
                    "BF20_B4_SmoothTail_B3Full": score_quantile_calibrated_backfill(
                        q_rep_new, q_rep_map, d_old, d_new, b3_full_ord[:n_bf], sp["REPORT"], doc_ids,
                        dev_queries_new=q_dev_new, dev_queries_mapped=q_dev_map, k=10, csls_k=0, legacy_hard_clip=False,
                    ),
                    "BF20_B4_SmoothTail_CSLS10_B3Full": score_quantile_calibrated_backfill(
                        q_rep_new, q_rep_map, d_old, d_new, b3_full_ord[:n_bf], sp["REPORT"], doc_ids,
                        dev_queries_new=q_dev_new, dev_queries_mapped=q_dev_map, k=10, csls_k=10, legacy_hard_clip=False,
                    ),
                    "BF20_B4_SmoothTail_B2OldSpace": score_quantile_calibrated_backfill(
                        q_rep_new, q_rep_map, d_old, d_new, b3_old_ord[:n_bf], sp["REPORT"], doc_ids,
                        dev_queries_new=q_dev_new, dev_queries_mapped=q_dev_map, k=10, csls_k=0, legacy_hard_clip=False,
                    ),
                    "BF20_B4_SmoothTail_CSLS10_B2OldSpace": score_quantile_calibrated_backfill(
                        q_rep_new, q_rep_map, d_old, d_new, b3_old_ord[:n_bf], sp["REPORT"], doc_ids,
                        dev_queries_new=q_dev_new, dev_queries_mapped=q_dev_map, k=10, csls_k=10, legacy_hard_clip=False,
                    ),
                }
                lmb = _eval_arm_suite("1B_beir_b4", f"{ds_name}_E3_to_E0_bf20", seed, arm_rkb, qrels, sp["REPORT"], {}, per_query_rows, summary_rows)
                for arm, m in lmb.items():
                    bf_lin[arm].append(m)
            agg["beir_b4_backfill"][ds_name] = {
                arm: round(float(np.mean([np.mean(list(m.values())) for m in ms])), 4) for arm, ms in bf_lin.items()
            }
    return agg


def main() -> None:
    parser = argparse.ArgumentParser(description="Flagship #1 R4 Comprehensive Validation")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 23, 37])
    args = parser.parse_args()

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    per_query_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    res_1a = run_suite_1a_4096_ood(args.cache_dir, args.seeds, per_query_rows, summary_rows)
    res_1b = run_suite_1b_beir_and_b4(args.cache_dir, args.seeds, per_query_rows, summary_rows)

    pq_csv = out_dir / "f1_r4_per_query.csv"
    with pq_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(per_query_rows[0].keys()))
        w.writeheader()
        w.writerows(per_query_rows)

    sum_csv = out_dir / "f1_r4_summary.csv"
    with sum_csv.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summary_rows[0].keys()))
        w.writeheader()
        w.writerows(summary_rows)

    manifest = {
        "schema_version": "f1_r4_comprehensive_v1",
        "git_sha": current_git_sha(REPO_ROOT),
        "seeds": list(args.seeds),
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "n_per_query_rows": len(per_query_rows),
        "n_summary_rows": len(summary_rows),
        "suite_1a_4096_ood": res_1a,
        "suite_1b_beir_and_b4": res_1b,
        "telemetry": process_telemetry(),
    }
    man_path = out_dir / "f1_r4_manifest.json"
    write_json_atomic(man_path, manifest)

    receipt = {
        "schema_version": "r4_report_receipt_v1",
        "git_sha": manifest["git_sha"],
        "manifest_path": "f1_r4_manifest.json",
        "manifest_sha256": _sha256_file(man_path),
        "artifact_hashes": {
            "f1_r4_manifest.json": _sha256_file(man_path),
            "f1_r4_summary.csv": _sha256_file(sum_csv),
            "f1_r4_per_query.csv": _sha256_file(pq_csv),
        },
        "wall_seconds": manifest["wall_seconds"],
    }
    write_json_atomic(out_dir / "report_receipt.json", receipt)
    print(f"[F1-R4] COMPLETE in {manifest['wall_seconds']}s -> {man_path}", flush=True)


if __name__ == "__main__":
    main()
