"""Night-1 Spark Job S2: Multi-hop lineage chain error accumulation (E0 -> E1 -> E2 -> E3).

Compares across SciFact, NFCorpus, FiQA2018 x anchor budgets {250, 500, 1000, 2000} x 3 seeds:
  1. Composed closed-form SVD Procrustes (W_m ... W_1)
  2. Composed closed-form Ridge (R_m ... R_1)
  3. Direct SVD Procrustes refit (E_m -> E_0) at matched anchor budget
  4. Direct Ridge refit (E_m -> E_0) at matched anchor budget
  5. Direct Ridge + 10% / 20% FastFill partial backfill
  6. Full re-embed C2O reference
Logs singular values, condition numbers, spectral error norms, and REPORT nDCG@10 vs hop count m.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

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


def _ndcg_map(per_q: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    return {qid: float(vals["ndcg_at_10"]) for qid, vals in per_q.items()}


def _tune_ridge_on_dev(
    src_docs: np.ndarray,
    tgt_docs: np.ndarray,
    anchor_idx: np.ndarray,
    dev_q_src: np.ndarray,
    dev_qids: Sequence[str],
    doc_ids: Sequence[str],
    qrels: Dict[str, Dict[str, float]],
) -> tuple[np.ndarray, float]:
    best_lam = DEFAULT_RIDGE_LAMBDAS[0]
    best_score = -1.0
    best_w = None
    for lam in DEFAULT_RIDGE_LAMBDAS:
        w = fit_closed_form_ridge(src_docs, tgt_docs, fit_idx=anchor_idx, regularization=lam)
        r_dev, _ = score_dense_rankings(dev_q_src @ w, tgt_docs, dev_qids, doc_ids, k=10)
        m = summarize_per_query_metrics(evaluate_rankings_mlr(r_dev, qrels, dev_qids, k=10))["mean_ndcg_at_10"]
        if m > best_score:
            best_score = m
            best_lam = lam
            best_w = w
    assert best_w is not None
    return best_w, best_lam


def run_multihop_campaign(
    *,
    output_dir: Path,
    cache_dir: Path,
    datasets: Sequence[str],
    anchor_budgets: Sequence[int],
    seeds: Sequence[int],
    max_docs: int | None = None,
    batch_size: int = 128,
) -> Dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be absent or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    chain = ["E0", "E1", "E2", "E3"]
    t_start = time.perf_counter()
    rows: List[Dict[str, Any]] = []
    dataset_manifests: Dict[str, Any] = {}

    for ds_name in datasets:
        print(f"[S2] Loading dataset {ds_name}...", flush=True)
        doc_ids, doc_texts, query_ids, query_texts, qrels, qrels_sha = load_beir_dataset(
            ds_name, max_docs=max_docs
        )
        splits = split_query_ids(query_ids)
        qid_to_idx = {qid: i for i, qid in enumerate(query_ids)}
        dev_idx = [qid_to_idx[q] for q in splits["DEVELOPMENT"]]
        rep_idx = [qid_to_idx[q] for q in splits["REPORT"]]
        split_sha = sha256_json(splits)
        contract_manifest, contract_sha = metric_contract_digest(qrels_sha, k=10, gain_rule="graded")

        enc_data: Dict[str, Dict[str, Any]] = {}
        for ek in chain:
            print(f"[S2] Encoding/loading {ds_name} x {ek} ({ENCODERS[ek]})...", flush=True)
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
            "qrels_sha256": qrels_sha,
            "split_sha256": split_sha,
            "metric_contract": contract_manifest,
            "metric_contract_sha256": contract_sha,
            "encoders": {ek: enc_data[ek]["meta"] for ek in chain},
        }

        d0 = enc_data["E0"]["docs"]
        q0 = enc_data["E0"]["queries"]
        fastfill_order = compute_fastfill_backfill_order(q0[dev_idx], d0, top_m=50)

        for n_anchors in anchor_budgets:
            eff_anchors = min(int(n_anchors), len(doc_ids) // 2)
            for seed in seeds:
                print(f"[S2] {ds_name} | anchors={eff_anchors} | seed={seed}...", flush=True)
                # Fit each single hop (E_{k} -> E_{k-1}) on its own deterministic qrels-free anchor draw
                hop_proc: List[np.ndarray] = []
                hop_ridge: List[np.ndarray] = []
                hop_lams: List[float] = []

                for step in range(1, len(chain)):
                    src_k = chain[step]
                    tgt_k = chain[step - 1]
                    a_idx = select_qrels_independent_anchors(
                        doc_ids, anchor_frac=1.0, seed=seed + step * 97, max_anchors=eff_anchors
                    )
                    wp = fit_closed_form_svd_procrustes(
                        enc_data[src_k]["docs"], enc_data[tgt_k]["docs"], fit_idx=a_idx
                    )
                    wr, lam = _tune_ridge_on_dev(
                        enc_data[src_k]["docs"],
                        enc_data[tgt_k]["docs"],
                        a_idx,
                        enc_data[src_k]["queries"][dev_idx],
                        splits["DEVELOPMENT"],
                        doc_ids,
                        qrels,
                    )
                    hop_proc.append(wp)
                    hop_ridge.append(wr)
                    hop_lams.append(lam)

                # Evaluate at hops m = 1 (E1->E0), m = 2 (E2->E0), m = 3 (E3->E0)
                w_comp_proc = None
                w_comp_ridge = None
                for hop_m in range(1, len(chain)):
                    src_k = chain[hop_m]
                    d_src = enc_data[src_k]["docs"]
                    q_src = enc_data[src_k]["queries"]

                    # Compose backward from E_m -> E_{m-1} -> ... -> E_0 (row-vector right multiplication)
                    if hop_m == 1:
                        w_comp_proc = hop_proc[0]
                        w_comp_ridge = hop_ridge[0]
                    else:
                        w_comp_proc = hop_proc[hop_m - 1] @ w_comp_proc
                        w_comp_ridge = hop_ridge[hop_m - 1] @ w_comp_ridge

                    # Direct refit E_m -> E_0 at matched anchor budget
                    direct_idx = select_qrels_independent_anchors(
                        doc_ids, anchor_frac=1.0, seed=seed + 777 + hop_m, max_anchors=eff_anchors
                    )
                    w_dir_proc = fit_closed_form_svd_procrustes(d_src, d0, fit_idx=direct_idx)
                    w_dir_ridge, dir_lam = _tune_ridge_on_dev(
                        d_src, d0, direct_idx, q_src[dev_idx], splits["DEVELOPMENT"], doc_ids, qrels
                    )

                    # Full-corpus oracle linear reference for spectral error measurement (non-deployable diagnostic)
                    w_oracle_proc = fit_closed_form_svd_procrustes(d_src, d0)

                    # Evaluate REPORT split across all methods
                    r_c2o, _ = score_dense_rankings(q_src[rep_idx], d_src, splits["REPORT"], doc_ids, k=10)
                    pq_c2o = evaluate_rankings_mlr(r_c2o, qrels, splits["REPORT"], k=10)

                    method_maps = {
                        "composed_procrustes": w_comp_proc,
                        "composed_ridge": w_comp_ridge,
                        "direct_procrustes": w_dir_proc,
                        "direct_ridge": w_dir_ridge,
                    }
                    pq_by_method: Dict[str, Dict[str, Dict[str, float]]] = {"full_reembed_c2o": pq_c2o}
                    spec_by_method: Dict[str, Dict[str, float]] = {}

                    for mname, wmat in method_maps.items():
                        sv = np.linalg.svd(wmat, compute_uv=False)
                        s_max = float(sv[0])
                        s_min = float(sv[-1])
                        cond = float(s_max / max(s_min, 1e-12))
                        spec_err = float(np.linalg.norm(wmat - w_oracle_proc, ord=2))
                        spec_by_method[mname] = {
                            "sigma_max": round(s_max, 6),
                            "sigma_min": round(s_min, 6),
                            "cond_number": round(cond, 4),
                            "spectral_err_vs_oracle_proc": round(spec_err, 6),
                        }
                        r_m, _ = score_dense_rankings(q_src[rep_idx] @ wmat, d0, splits["REPORT"], doc_ids, k=10)
                        pq_by_method[mname] = evaluate_rankings_mlr(r_m, qrels, splits["REPORT"], k=10)

                    # Direct ridge + 10% and 20% FastFill partial backfill
                    for b in (0.10, 0.20):
                        n_bf = max(1, int(round(len(doc_ids) * b)))
                        bf_idx = fastfill_order[:n_bf]
                        mname = f"direct_ridge_plus_backfill_{int(round(b * 100)):02d}"
                        r_mb = score_map_plus_backfill(
                            q_src[rep_idx], d0, d_src, w_dir_ridge, bf_idx, splits["REPORT"], doc_ids,
                            dev_queries_new=q_src[dev_idx], k=10, calibrate=True,
                        )
                        pq_by_method[mname] = evaluate_rankings_mlr(r_mb, qrels, splits["REPORT"], k=10)
                        spec_by_method[mname] = spec_by_method["direct_ridge"]

                    # Paired comparison: composed_procrustes minus composed_ridge
                    proc_vs_ridge = paired_bootstrap_delta(
                        _ndcg_map(pq_by_method["composed_procrustes"]),
                        _ndcg_map(pq_by_method["composed_ridge"]),
                        n_bootstrap=5000,
                        seed=seed,
                    )
                    dir_vs_comp_ridge = paired_bootstrap_delta(
                        _ndcg_map(pq_by_method["direct_ridge"]),
                        _ndcg_map(pq_by_method["composed_ridge"]),
                        n_bootstrap=5000,
                        seed=seed,
                    )

                    for mname, pq in pq_by_method.items():
                        summ = summarize_per_query_metrics(pq)
                        sp = spec_by_method.get(mname, {})
                        rows.append(
                            {
                                "dataset": ds_name,
                                "hop_m": hop_m,
                                "source_encoder": src_k,
                                "target_encoder": "E0",
                                "anchor_budget": eff_anchors,
                                "seed": int(seed),
                                "method": mname,
                                "report_ndcg_at_10": round(summ["mean_ndcg_at_10"], 6),
                                "report_mrr": round(summ["mean_mrr"], 6),
                                "report_recall_at_10": round(summ["mean_recall_at_10"], 6),
                                "n_queries": summ["n_queries"],
                                "sigma_max": sp.get("sigma_max"),
                                "sigma_min": sp.get("sigma_min"),
                                "cond_number": sp.get("cond_number"),
                                "spectral_err_vs_oracle_proc": sp.get("spectral_err_vs_oracle_proc"),
                                "comp_proc_minus_comp_ridge_ndcg": round(proc_vs_ridge["mean_delta"], 6),
                                "comp_proc_minus_comp_ridge_lcb95": round(proc_vs_ridge["lcb_95"], 6),
                                "dir_ridge_minus_comp_ridge_ndcg": round(dir_vs_comp_ridge["mean_delta"], 6),
                                "dir_ridge_minus_comp_ridge_lcb95": round(dir_vs_comp_ridge["lcb_95"], 6),
                                "direct_ridge_lambda": dir_lam,
                                "split_sha256": split_sha,
                                "metric_contract_sha256": contract_sha,
                            }
                        )

    csv_path = output_dir / "s2_spectral_errors.csv"
    fieldnames = list(rows[0].keys()) if rows else []
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    manifest = {
        "job": "night1_s2_multihop",
        "git_sha": current_git_sha(REPO_ROOT),
        "chain": chain,
        "datasets": list(datasets),
        "anchor_budgets": [int(a) for a in anchor_budgets],
        "seeds": [int(s) for s in seeds],
        "dataset_manifests": dataset_manifests,
        "row_count": len(rows),
        "wall_seconds": round(time.perf_counter() - t_start, 2),
        "telemetry": process_telemetry(),
    }
    manifest_sha = write_json_atomic(output_dir / "s2_multihop_manifest.json", manifest)
    write_json_atomic(
        output_dir / "report_receipt.json",
        {"job": "night1_s2_multihop", "status": "COMPLETED", "manifest_sha256": manifest_sha},
    )
    print(f"[S2] COMPLETE in {manifest['wall_seconds']}s -> {output_dir}", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Night-1 Spark Job S2: Multi-hop lineage chain")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/chelated_night1_emb_cache"))
    parser.add_argument("--datasets", nargs="+", default=["SciFact", "NFCorpus", "FiQA2018"])
    parser.add_argument("--anchor-budgets", nargs="+", type=int, default=[250, 500, 1000, 2000])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 137, 2718])
    parser.add_argument("--max-docs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    run_multihop_campaign(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        datasets=args.datasets,
        anchor_budgets=args.anchor_budgets,
        seeds=args.seeds,
        max_docs=args.max_docs,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
