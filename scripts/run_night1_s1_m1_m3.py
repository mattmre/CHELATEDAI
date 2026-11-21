"""Night-1 Spark Job S1: Single-hop M1-M3 baseline ladder, #355 floor, and fixed M2 preflight.

Evaluates SciFact, NFCorpus, and FiQA2018 across:
  - Same-dim swap:  E0 (all-MiniLM-L6-v2, 384) -> E1 (bge-small-en-v1.5, 384)
  - Cross-dim swap: E0 (all-MiniLM-L6-v2, 384) -> E3 (all-mpnet-base-v2, 768)
across 3 deterministic anchor seeds [42, 137, 2718].

Strictly separates DEVELOPMENT (hyperparameter/backfill order & M2 preflight),
SELECT (operating-point freeze -> selection_lock.json), and one-shot REPORT.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from m1_baseline_ladder import (  # noqa: E402
    DEFAULT_BACKFILL_BUDGETS,
    DEFAULT_RIDGE_LAMBDAS,
    compute_fastfill_backfill_order,
    evaluate_rankings_mlr,
    fit_closed_form_ridge,
    fit_closed_form_svd_procrustes,
    m2_gap_preflight,
    score_dense_rankings,
    score_map_plus_backfill,
    select_qrels_independent_anchors,
    split_query_ids,
)
from scripts.night1_common import (  # noqa: E402
    ENCODERS,
    FastBM25Index,
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


def _eval_split(
    rankings: Dict[str, List[str]],
    qrels: Dict[str, Dict[str, float]],
    split_qids: Sequence[str],
    *,
    k: int = 10,
) -> Dict[str, Dict[str, float]]:
    return evaluate_rankings_mlr(rankings, qrels, split_qids, k=k, binary=False)


def _ndcg_map(per_q: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    return {qid: float(vals["ndcg_at_10"]) for qid, vals in per_q.items()}


def run_single_hop_campaign(
    *,
    output_dir: Path,
    cache_dir: Path,
    datasets: Sequence[str],
    seeds: Sequence[int],
    budgets: Sequence[float],
    delta: float = 0.02,
    max_docs: int | None = None,
    batch_size: int = 128,
) -> Dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be absent or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    swaps = [
        ("E0_to_E1_samedim", "E0", "E1"),
        ("E0_to_E3_crossdim", "E0", "E3"),
    ]
    t_campaign_start = time.perf_counter()

    m2_records: List[Dict[str, Any]] = []
    frontier_rows: List[Dict[str, Any]] = []
    selection_locks: Dict[str, Any] = {}
    dataset_manifests: Dict[str, Any] = {}

    paired_jsonl_path = output_dir / "s1_per_query_paired.jsonl"
    paired_fh = paired_jsonl_path.open("w", encoding="utf-8")

    try:
        for ds_name in datasets:
            print(f"[S1] Loading dataset {ds_name}...", flush=True)
            doc_ids, doc_texts, query_ids, query_texts, qrels, qrels_sha = load_beir_dataset(
                ds_name, max_docs=max_docs
            )
            splits = split_query_ids(query_ids)
            qid_to_idx = {qid: i for i, qid in enumerate(query_ids)}
            split_indices = {sp: [qid_to_idx[q] for q in qlist] for sp, qlist in splits.items()}
            split_sha = sha256_json(splits)
            contract_manifest, contract_sha = metric_contract_digest(qrels_sha, k=10, gain_rule="graded")

            # Build BM25 index once per dataset for #355 floor
            t_bm0 = time.perf_counter()
            bm25 = FastBM25Index(doc_ids, doc_texts)
            bm25_ms = (time.perf_counter() - t_bm0) * 1000.0

            enc_data: Dict[str, Dict[str, Any]] = {}
            for ek in ("E0", "E1", "E3"):
                print(f"[S1] Encoding/loading {ds_name} x {ek} ({ENCODERS[ek]})...", flush=True)
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
                enc_data[ek] = {"docs": d_emb, "queries": q_emb, "meta": meta}

            dataset_manifests[ds_name] = {
                "doc_count": len(doc_ids),
                "query_count": len(query_ids),
                "split_counts": {sp: len(qlist) for sp, qlist in splits.items()},
                "qrels_sha256": qrels_sha,
                "split_sha256": split_sha,
                "metric_contract": contract_manifest,
                "metric_contract_sha256": contract_sha,
                "bm25_build_ms": round(bm25_ms, 2),
                "encoders": {ek: enc_data[ek]["meta"] for ek in ("E0", "E1", "E3")},
            }

            for swap_label, old_k, new_k in swaps:
                d_old = enc_data[old_k]["docs"]
                q_old = enc_data[old_k]["queries"]
                d_new = enc_data[new_k]["docs"]
                q_new = enc_data[new_k]["queries"]
                same_dim = int(d_old.shape[1]) == int(d_new.shape[1])

                # FastFill-style exposure ordering fit strictly on DEVELOPMENT old-space queries
                dev_q_idx = split_indices["DEVELOPMENT"]
                fastfill_order = compute_fastfill_backfill_order(q_old[dev_q_idx], d_old, top_m=50)

                for seed in seeds:
                    print(f"[S1] Running {ds_name} | {swap_label} | seed={seed}...", flush=True)
                    anchor_idx = select_qrels_independent_anchors(doc_ids, anchor_frac=0.50, seed=seed)
                    anchor_sha = sha256_json([doc_ids[int(i)] for i in anchor_idx])

                    # Uniform random backfill permutation for matched random control
                    rng = np.random.default_rng(seed + 1009)
                    random_order = rng.permutation(len(doc_ids))

                    # 1. Fit closed-form SVD Procrustes (new_query -> old_doc space) on qrels-free anchors
                    t_p0 = time.perf_counter()
                    w_proc = fit_closed_form_svd_procrustes(d_new, d_old, fit_idx=anchor_idx)
                    proc_fit_ms = (time.perf_counter() - t_p0) * 1000.0

                    # 2. Tune closed-form Ridge lambda strictly on DEVELOPMENT
                    best_lam = DEFAULT_RIDGE_LAMBDAS[0]
                    best_dev_ndcg = -1.0
                    w_ridge = None
                    ridge_fit_ms = 0.0
                    for lam in DEFAULT_RIDGE_LAMBDAS:
                        t_r0 = time.perf_counter()
                        w_cand = fit_closed_form_ridge(d_new, d_old, fit_idx=anchor_idx, regularization=lam)
                        dt_ms = (time.perf_counter() - t_r0) * 1000.0
                        q_dev_mapped = q_new[dev_q_idx] @ w_cand
                        dev_rank, _ = score_dense_rankings(q_dev_mapped, d_old, splits["DEVELOPMENT"], doc_ids, k=10)
                        dev_m = summarize_per_query_metrics(_eval_split(dev_rank, qrels, splits["DEVELOPMENT"]))
                        if dev_m["mean_ndcg_at_10"] > best_dev_ndcg:
                            best_dev_ndcg = dev_m["mean_ndcg_at_10"]
                            best_lam = lam
                            w_ridge = w_cand
                            ridge_fit_ms = dt_ms
                    assert w_ridge is not None

                    # 3. Helper to evaluate all M1 arms on a declared split
                    def eval_all_arms(split_name: str, selected_map_name: str | None = None) -> Dict[str, Any]:
                        sp_qids = splits[split_name]
                        sp_idx = split_indices[split_name]
                        sp_qtexts = [query_texts[i] for i in sp_idx]
                        arm_per_q: Dict[str, Dict[str, Dict[str, float]]] = {}
                        arm_meta: Dict[str, Dict[str, Any]] = {}

                        # (a) identity_frozen_c0
                        if same_dim:
                            t0 = time.perf_counter()
                            r_c0, _ = score_dense_rankings(q_new[sp_idx], d_old, sp_qids, doc_ids, k=10)
                            arm_per_q["identity_frozen_c0"] = _eval_split(r_c0, qrels, sp_qids)
                            arm_meta["identity_frozen_c0"] = {
                                "status": "OK",
                                "budget": 0.0,
                                "search_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                                "aux_bytes": 0,
                            }
                        else:
                            arm_meta["identity_frozen_c0"] = {
                                "status": "INCOMPATIBLE_DIMENSION",
                                "budget": 0.0,
                                "search_ms": 0.0,
                                "aux_bytes": 0,
                            }

                        # (b) old_encoder_servable (control row; excluded when old_encoder_retired=True)
                        t0 = time.perf_counter()
                        r_old, _ = score_dense_rankings(q_old[sp_idx], d_old, sp_qids, doc_ids, k=10)
                        arm_per_q["old_encoder_servable"] = _eval_split(r_old, qrels, sp_qids)
                        arm_meta["old_encoder_servable"] = {
                            "status": "OK",
                            "requires_old_encoder": True,
                            "budget": 0.0,
                            "search_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                            "aux_bytes": 0,
                        }

                        # (c) closed_form_svd_procrustes
                        t0 = time.perf_counter()
                        q_proc = q_new[sp_idx] @ w_proc
                        r_proc, _ = score_dense_rankings(q_proc, d_old, sp_qids, doc_ids, k=10)
                        arm_per_q["closed_form_svd_procrustes"] = _eval_split(r_proc, qrels, sp_qids)
                        arm_meta["closed_form_svd_procrustes"] = {
                            "status": "OK",
                            "budget": 0.0,
                            "fit_ms": round(proc_fit_ms, 2),
                            "search_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                            "aux_bytes": int(w_proc.nbytes),
                        }

                        # (d) closed_form_ridge
                        t0 = time.perf_counter()
                        q_ridge = q_new[sp_idx] @ w_ridge
                        r_ridge, _ = score_dense_rankings(q_ridge, d_old, sp_qids, doc_ids, k=10)
                        arm_per_q["closed_form_ridge"] = _eval_split(r_ridge, qrels, sp_qids)
                        arm_meta["closed_form_ridge"] = {
                            "status": "OK",
                            "budget": 0.0,
                            "ridge_lambda": best_lam,
                            "fit_ms": round(ridge_fit_ms, 2),
                            "search_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                            "aux_bytes": int(w_ridge.nbytes),
                        }

                        # (h0-1) #355 hand-set CSLS floor (k=10) on ridge-bridged space (and C0 if same_dim)
                        t0 = time.perf_counter()
                        r_csls, _ = score_dense_rankings(
                            q_ridge,
                            d_old,
                            sp_qids,
                            doc_ids,
                            k=10,
                            csls_k=10,
                            csls_reference_queries=(q_new[dev_q_idx] @ w_ridge),
                        )
                        arm_per_q["floor_csls_k10"] = _eval_split(r_csls, qrels, sp_qids)
                        arm_meta["floor_csls_k10"] = {
                            "status": "OK",
                            "budget": 0.0,
                            "search_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                            "aux_bytes": int(w_ridge.nbytes),
                        }

                        # (h0-2) #355 hand-set BM25 + bridged-dense linear hybrid (alpha=0.5)
                        t0 = time.perf_counter()
                        q_r_norm = q_ridge / np.maximum(np.linalg.norm(q_ridge, axis=1, keepdims=True), 1e-12)
                        d_o_norm = d_old / np.maximum(np.linalg.norm(d_old, axis=1, keepdims=True), 1e-12)
                        dense_mat = q_r_norm @ d_o_norm.T
                        r_hyb = bm25.hybrid_rankings(sp_qtexts, sp_qids, dense_mat, alpha=0.5, k=10)
                        arm_per_q["floor_bm25_hybrid"] = _eval_split(r_hyb, qrels, sp_qids)
                        arm_meta["floor_bm25_hybrid"] = {
                            "status": "OK",
                            "budget": 0.0,
                            "alpha": 0.5,
                            "search_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                            "aux_bytes": int(w_ridge.nbytes),
                        }

                        # Pick map for map_plus_backfill using selected_map_name (locked on DEV/SELECT)
                        chosen_w = w_ridge if (selected_map_name != "closed_form_svd_procrustes") else w_proc

                        # (e) & (f) Budgeted partial backfill & map+backfill
                        for b in budgets:
                            b_tag = f"{int(round(b * 100)):02d}"
                            n_bf = max(1, int(round(len(doc_ids) * b)))
                            bf_ff = fastfill_order[:n_bf]
                            bf_rnd = random_order[:n_bf]

                            t0 = time.perf_counter()
                            r_bf = score_map_plus_backfill(
                                q_new[sp_idx], d_old, d_new, None, bf_ff, sp_qids, doc_ids,
                                dev_queries_new=q_new[dev_q_idx], k=10, calibrate=False,
                            )
                            arm_per_q[f"budgeted_backfill_{b_tag}"] = _eval_split(r_bf, qrels, sp_qids)
                            arm_meta[f"budgeted_backfill_{b_tag}"] = {
                                "status": "OK",
                                "budget": float(b),
                                "n_backfilled_docs": n_bf,
                                "search_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                                "aux_bytes": int(n_bf * d_new.shape[1] * 4),
                            }

                            t0 = time.perf_counter()
                            r_bfr = score_map_plus_backfill(
                                q_new[sp_idx], d_old, d_new, None, bf_rnd, sp_qids, doc_ids,
                                dev_queries_new=q_new[dev_q_idx], k=10, calibrate=False,
                            )
                            arm_per_q[f"random_backfill_{b_tag}"] = _eval_split(r_bfr, qrels, sp_qids)
                            arm_meta[f"random_backfill_{b_tag}"] = {
                                "status": "OK",
                                "budget": float(b),
                                "n_backfilled_docs": n_bf,
                                "search_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                                "aux_bytes": int(n_bf * d_new.shape[1] * 4),
                            }

                            t0 = time.perf_counter()
                            r_mb = score_map_plus_backfill(
                                q_new[sp_idx], d_old, d_new, chosen_w, bf_ff, sp_qids, doc_ids,
                                dev_queries_new=q_new[dev_q_idx], k=10, calibrate=True,
                            )
                            arm_per_q[f"map_plus_backfill_{b_tag}"] = _eval_split(r_mb, qrels, sp_qids)
                            arm_meta[f"map_plus_backfill_{b_tag}"] = {
                                "status": "OK",
                                "budget": float(b),
                                "n_backfilled_docs": n_bf,
                                "bridge_arm": selected_map_name or "closed_form_ridge",
                                "search_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                                "aux_bytes": int(chosen_w.nbytes + n_bf * d_new.shape[1] * 4),
                            }

                        # (g) full_reembed_c2o (costed reference ONLY)
                        t0 = time.perf_counter()
                        r_c2o, _ = score_dense_rankings(q_new[sp_idx], d_new, sp_qids, doc_ids, k=10)
                        arm_per_q["full_reembed_c2o"] = _eval_split(r_c2o, qrels, sp_qids)
                        arm_meta["full_reembed_c2o"] = {
                            "status": "OK",
                            "budget": 1.0,
                            "is_reference_only": True,
                            "search_ms": round((time.perf_counter() - t0) * 1000.0, 2),
                            "aux_bytes": int(d_new.nbytes),
                        }
                        return {"per_q": arm_per_q, "meta": arm_meta}

                    # Run DEV first; freeze bridge choice and run M2 preflight strictly on DEV
                    dev_res = eval_all_arms("DEVELOPMENT")
                    dev_proc_mean = summarize_per_query_metrics(dev_res["per_q"]["closed_form_svd_procrustes"])["mean_ndcg_at_10"]
                    dev_ridge_mean = summarize_per_query_metrics(dev_res["per_q"]["closed_form_ridge"])["mean_ndcg_at_10"]
                    locked_map_arm = "closed_form_ridge" if dev_ridge_mean >= dev_proc_mean else "closed_form_svd_procrustes"

                    # Re-evaluate DEV map_plus_backfill if Procrustes beat Ridge
                    if locked_map_arm != "closed_form_ridge":
                        dev_res = eval_all_arms("DEVELOPMENT", selected_map_name=locked_map_arm)

                    dev_ndcg_by_arm = {arm: _ndcg_map(pq) for arm, pq in dev_res["per_q"].items()}

                    # Run M2 gap preflight at each budget (old_encoder_retired=True)
                    zero_budget_arms = [
                        a for a in (
                            "identity_frozen_c0",
                            "closed_form_svd_procrustes",
                            "closed_form_ridge",
                            "floor_csls_k10",
                            "floor_bm25_hybrid",
                        )
                        if a in dev_ndcg_by_arm
                    ]
                    budget_grid = [0.0] + [float(b) for b in budgets]
                    locked_best_by_budget: Dict[str, str] = {}
                    for b in budget_grid:
                        if b == 0.0:
                            elig = list(zero_budget_arms)
                        else:
                            b_tag = f"{int(round(b * 100)):02d}"
                            elig = list(zero_budget_arms) + [
                                f"budgeted_backfill_{b_tag}",
                                f"random_backfill_{b_tag}",
                                f"map_plus_backfill_{b_tag}",
                            ]
                        m2 = m2_gap_preflight(
                            dev_ndcg_by_arm,
                            split="DEVELOPMENT",
                            reference_arm="full_reembed_c2o",
                            eligible_arms=elig,
                            delta=delta,
                            n_bootstrap=10_000,
                            seed=seed,
                        )
                        m2["dataset"] = ds_name
                        m2["swap"] = swap_label
                        m2["seed"] = int(seed)
                        m2["budget"] = float(b)
                        m2["old_encoder_retired"] = True
                        m2_records.append(m2)

                    # Run SELECT to freeze exact best ordinary arm per budget before touching REPORT
                    sel_res = eval_all_arms("SELECT", selected_map_name=locked_map_arm)
                    for b in budget_grid:
                        if b == 0.0:
                            elig = list(zero_budget_arms)
                        else:
                            b_tag = f"{int(round(b * 100)):02d}"
                            elig = list(zero_budget_arms) + [
                                f"budgeted_backfill_{b_tag}",
                                f"random_backfill_{b_tag}",
                                f"map_plus_backfill_{b_tag}",
                            ]
                        best_sel_arm = max(
                            elig,
                            key=lambda arm: summarize_per_query_metrics(sel_res["per_q"][arm])["mean_ndcg_at_10"],
                        )
                        locked_best_by_budget[f"{b:.2f}"] = best_sel_arm

                    lock_key = f"{ds_name}:{swap_label}:seed{seed}"
                    selection_locks[lock_key] = {
                        "dataset": ds_name,
                        "swap": swap_label,
                        "seed": int(seed),
                        "anchor_sha256": anchor_sha,
                        "locked_ridge_lambda": best_lam,
                        "locked_map_arm": locked_map_arm,
                        "locked_best_ordinary_arm_by_budget": locked_best_by_budget,
                    }

                    # Persist selection_lock.json BEFORE opening REPORT
                    write_json_atomic(output_dir / "selection_lock.json", selection_locks)

                    # One-shot REPORT evaluation
                    rep_res = eval_all_arms("REPORT", selected_map_name=locked_map_arm)

                    # Emit per-query paired rows for all splits & arms
                    for sp_name, sp_res in (("DEVELOPMENT", dev_res), ("SELECT", sel_res), ("REPORT", rep_res)):
                        for arm, pq in sp_res["per_q"].items():
                            meta_arm = sp_res["meta"][arm]
                            summary = summarize_per_query_metrics(pq)
                            row = {
                                "dataset": ds_name,
                                "swap": swap_label,
                                "seed": int(seed),
                                "split": sp_name,
                                "arm": arm,
                                "budget": float(meta_arm.get("budget", 0.0)),
                                "status": meta_arm.get("status", "OK"),
                                "mean_ndcg_at_10": round(summary["mean_ndcg_at_10"], 6),
                                "mean_map_at_10": round(summary["mean_map_at_10"], 6),
                                "mean_mrr": round(summary["mean_mrr"], 6),
                                "mean_recall_at_10": round(summary["mean_recall_at_10"], 6),
                                "n_queries": summary["n_queries"],
                                "search_ms": meta_arm.get("search_ms", 0.0),
                                "aux_bytes": meta_arm.get("aux_bytes", 0),
                                "split_sha256": split_sha,
                                "metric_contract_sha256": contract_sha,
                            }
                            if sp_name == "REPORT" and arm != "full_reembed_c2o":
                                boot = paired_bootstrap_delta(
                                    _ndcg_map(rep_res["per_q"]["full_reembed_c2o"]),
                                    _ndcg_map(pq),
                                    n_bootstrap=10_000,
                                    seed=seed,
                                )
                                row["c2o_minus_arm_ndcg"] = round(boot["mean_delta"], 6)
                                row["c2o_minus_arm_lcb95"] = round(boot["lcb_95"], 6)
                                row["c2o_minus_arm_ucb95"] = round(boot["ucb_95"], 6)
                            else:
                                row["c2o_minus_arm_ndcg"] = 0.0 if arm == "full_reembed_c2o" else None
                                row["c2o_minus_arm_lcb95"] = None
                                row["c2o_minus_arm_ucb95"] = None
                            frontier_rows.append(row)

                            for qid, qvals in pq.items():
                                paired_fh.write(
                                    json.dumps(
                                        {
                                            "dataset": ds_name,
                                            "swap": swap_label,
                                            "seed": int(seed),
                                            "split": sp_name,
                                            "query_id": qid,
                                            "arm": arm,
                                            "budget": float(meta_arm.get("budget", 0.0)),
                                            **qvals,
                                        },
                                        sort_keys=True,
                                    )
                                    + "\n"
                                )
                        # Also record INCOMPATIBLE_DIMENSION row explicitly if cross-dim C0
                        if "identity_frozen_c0" not in sp_res["per_q"]:
                            frontier_rows.append(
                                {
                                    "dataset": ds_name,
                                    "swap": swap_label,
                                    "seed": int(seed),
                                    "split": sp_name,
                                    "arm": "identity_frozen_c0",
                                    "budget": 0.0,
                                    "status": "INCOMPATIBLE_DIMENSION",
                                    "mean_ndcg_at_10": None,
                                    "mean_map_at_10": None,
                                    "mean_mrr": None,
                                    "mean_recall_at_10": None,
                                    "n_queries": len(splits[sp_name]),
                                    "search_ms": 0.0,
                                    "aux_bytes": 0,
                                    "split_sha256": split_sha,
                                    "metric_contract_sha256": contract_sha,
                                    "c2o_minus_arm_ndcg": None,
                                    "c2o_minus_arm_lcb95": None,
                                    "c2o_minus_arm_ucb95": None,
                                }
                            )
    finally:
        paired_fh.close()

    # Write CSV frontier + M2 preflight + final manifest + report receipt
    csv_path = output_dir / "s1_m3_frontier.csv"
    fieldnames = [
        "dataset", "swap", "seed", "split", "arm", "budget", "status",
        "mean_ndcg_at_10", "mean_map_at_10", "mean_mrr", "mean_recall_at_10",
        "n_queries", "c2o_minus_arm_ndcg", "c2o_minus_arm_lcb95", "c2o_minus_arm_ucb95",
        "search_ms", "aux_bytes", "split_sha256", "metric_contract_sha256",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(frontier_rows)

    m2_sha = write_json_atomic(output_dir / "s1_m2_preflight.json", m2_records)
    lock_sha = write_json_atomic(output_dir / "selection_lock.json", selection_locks)

    manifest = {
        "job": "night1_s1_m1_m3",
        "git_sha": current_git_sha(REPO_ROOT),
        "old_encoder_retired": True,
        "delta": float(delta),
        "datasets": list(datasets),
        "seeds": [int(s) for s in seeds],
        "budgets": [float(b) for b in budgets],
        "dataset_manifests": dataset_manifests,
        "selection_lock_sha256": lock_sha,
        "m2_preflight_sha256": m2_sha,
        "wall_seconds": round(time.perf_counter() - t_campaign_start, 2),
        "telemetry": process_telemetry(),
    }
    manifest_sha = write_json_atomic(output_dir / "s1_m1_m3_manifest.json", manifest)
    write_json_atomic(
        output_dir / "report_receipt.json",
        {
            "job": "night1_s1_m1_m3",
            "status": "COMPLETED",
            "manifest_sha256": manifest_sha,
            "selection_lock_sha256": lock_sha,
            "m2_preflight_sha256": m2_sha,
        },
    )
    print(f"[S1] COMPLETE in {manifest['wall_seconds']}s -> {output_dir}", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Night-1 Spark Job S1: Single-hop M1-M3 ladder")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/chelated_night1_emb_cache"))
    parser.add_argument("--datasets", nargs="+", default=["SciFact", "NFCorpus", "FiQA2018"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 137, 2718])
    parser.add_argument("--budgets", nargs="+", type=float, default=list(DEFAULT_BACKFILL_BUDGETS))
    parser.add_argument("--delta", type=float, default=0.02)
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    run_single_hop_campaign(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        datasets=args.datasets,
        seeds=args.seeds,
        budgets=args.budgets,
        delta=args.delta,
        max_docs=args.max_docs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
