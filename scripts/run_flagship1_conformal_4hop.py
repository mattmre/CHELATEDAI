"""Flagship #1 Conformal Radius-Matched 4-Hop Validation Across 6 BEIR Domains (`mlr-v1`).

Compares on all 6 BEIR datasets (SciFact, NFCorpus, FiQA2018, ArguAna, SCIDOCS, Quora)
over the 5-encoder / 4-hop chain (E4 -> E3 -> E2 -> E1 -> E0) at n=500 anchors × 3 seeds:
  1. Composed Tuned Ridge (baseline)
  2. Composed Unscaled Centered Procrustes (`A2_unscaled`, `match_radius=False`)
  3. Composed Conformal Centered Procrustes (`A2c_conformal_unif`, `match_radius=True`)
  4. Composed Conformal Centered Procrustes + P6 k-Means++ (`A2c_conformal_kmeanspp`)

Verifies that conformal spherical-cap radius matching (`scale = ||Y_c||_F / ||X_c||_F`)
preserves exact `cond = 1.00` while eliminating the Hop-2 narrow-cone (`E2 -> E1 -> E0`)
radius-mismatch dip and boosting Hop 1 and Hop 4 across all 6 BEIR domains.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from m1_baseline_ladder import (  # noqa: E402
    DEFAULT_RIDGE_LAMBDAS,
    evaluate_rankings_mlr,
    fit_closed_form_ridge,
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
from spherical_migration_bridge import (  # noqa: E402
    CenteredProcrustesBridge,
    compose_centered_bridges,
    select_spherical_kmeanspp_anchors,
)


def _ndcg_map(per_q: Dict[str, Dict[str, float]]) -> Dict[str, float]:
    return {qid: float(vals["ndcg_at_10"]) for qid, vals in per_q.items()}


def _tune_ridge(d_src, d_tgt, q_dev_src, dev_qids, doc_ids, qrels, fit_idx):
    best_w = None
    best_sc = -1.0
    for lam in DEFAULT_RIDGE_LAMBDAS:
        wc = fit_closed_form_ridge(d_src, d_tgt, fit_idx=fit_idx, regularization=lam)
        rd, _ = score_dense_rankings(q_dev_src @ wc, d_tgt, dev_qids, doc_ids, k=10)
        sc = summarize_per_query_metrics(evaluate_rankings_mlr(rd, qrels, dev_qids, k=10))["mean_ndcg_at_10"]
        if sc > best_sc:
            best_sc = sc
            best_w = wc
    assert best_w is not None
    return best_w


def run_conformal_4hop(
    *,
    output_dir: Path,
    cache_dir: Path,
    datasets: Sequence[str],
    seeds: Sequence[int],
    max_docs_large: int = 40_000,
    max_queries_cap: int = 1200,
) -> Dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be absent or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    chain = ["E0", "E1", "E2", "E3", "E4"]
    rows: List[Dict[str, Any]] = []

    for ds_name in datasets:
        print(f"[F1-CONF4] Loading {ds_name}...", flush=True)
        doc_cap = max_docs_large if ds_name in ("Quora", "FiQA2018") else None
        doc_ids, doc_texts, query_ids, query_texts, qrels, qrels_sha = load_beir_dataset(
            ds_name, max_docs=doc_cap
        )
        if len(query_ids) > max_queries_cap:
            keep_q = sorted(query_ids, key=lambda q: sha256_json({"ds": ds_name, "q": q}))[:max_queries_cap]
            keep_set = set(keep_q)
            q_pairs = [(qid, qtxt) for qid, qtxt in zip(query_ids, query_texts) if qid in keep_set]
            query_ids = [p[0] for p in q_pairs]
            query_texts = [p[1] for p in q_pairs]
            qrels = {qid: qrels[qid] for qid in query_ids}
            qrels_sha = sha256_json(qrels)

        splits = split_query_ids(query_ids)
        qid_to_idx = {qid: i for i, qid in enumerate(query_ids)}
        dev_idx = [qid_to_idx[q] for q in splits["DEVELOPMENT"]]
        rep_idx = [qid_to_idx[q] for q in splits["REPORT"]]
        _, contract_sha = metric_contract_digest(qrels_sha, k=10, gain_rule="graded")

        enc: Dict[str, Dict[str, Any]] = {}
        for ek in chain:
            d_emb, q_emb, _ = encode_or_load_embeddings(
                dataset_name=ds_name,
                encoder_key=ek,
                encoder_name=ENCODERS[ek],
                doc_ids=doc_ids,
                doc_texts=doc_texts,
                query_ids=query_ids,
                query_texts=query_texts,
                cache_dir=cache_dir,
            )
            enc[ek] = {"docs": d_emb, "queries": q_emb}

        d0 = enc["E0"]["docs"]
        n_mh = min(500, len(doc_ids) // 2)
        for seed in seeds:
            hop_ridge: List[np.ndarray] = []
            hop_a2_unscaled: List[CenteredProcrustesBridge] = []
            hop_a2c_unif: List[CenteredProcrustesBridge] = []
            hop_a2c_km: List[CenteredProcrustesBridge] = []

            for step in range(1, len(chain)):
                sk, tk = chain[step], chain[step - 1]
                rng_s = np.random.default_rng(seed + step * 97)
                idx_u = np.sort(rng_s.choice(len(doc_ids), size=n_mh, replace=False))
                idx_km = select_spherical_kmeanspp_anchors(enc[sk]["docs"], n_mh, seed=seed + step * 97)

                hop_ridge.append(
                    _tune_ridge(
                        enc[sk]["docs"], enc[tk]["docs"], enc[sk]["queries"][dev_idx],
                        splits["DEVELOPMENT"], doc_ids, qrels, idx_u,
                    )
                )
                hop_a2_unscaled.append(
                    CenteredProcrustesBridge.fit(enc[sk]["docs"], enc[tk]["docs"], fit_idx=idx_u, match_radius=False)
                )
                hop_a2c_unif.append(
                    CenteredProcrustesBridge.fit(enc[sk]["docs"], enc[tk]["docs"], fit_idx=idx_u, match_radius=True)
                )
                hop_a2c_km.append(
                    CenteredProcrustesBridge.fit(enc[sk]["docs"], enc[tk]["docs"], fit_idx=idx_km, match_radius=True)
                )

            for hop_m in range(1, len(chain)):
                sk = chain[hop_m]
                q_src = enc[sk]["queries"][rep_idx]
                w_r = hop_ridge[hop_m - 1]
                for i in range(hop_m - 2, -1, -1):
                    w_r = w_r @ hop_ridge[i]
                sv_r = np.linalg.svd(w_r, compute_uv=False)
                cond_r = float(sv_r[0] / max(sv_r[-1], 1e-12))

                b_unsc = compose_centered_bridges([hop_a2_unscaled[i] for i in range(hop_m - 1, -1, -1)])
                b_cu = compose_centered_bridges([hop_a2c_unif[i] for i in range(hop_m - 1, -1, -1)])
                b_ckm = compose_centered_bridges([hop_a2c_km[i] for i in range(hop_m - 1, -1, -1)])

                pqs: Dict[str, Dict[str, Dict[str, float]]] = {}
                conds: Dict[str, float] = {}
                for mname, qm, cv in (
                    ("composed_ridge", q_src @ w_r, cond_r),
                    ("composed_A2_unscaled", b_unsc.transform(q_src), b_unsc.condition_number),
                    ("composed_A2c_conformal_unif", b_cu.transform(q_src), b_cu.condition_number),
                    ("composed_A2c_conformal_kmeanspp", b_ckm.transform(q_src), b_ckm.condition_number),
                ):
                    rk, _ = score_dense_rankings(qm, d0, splits["REPORT"], doc_ids, k=10)
                    pqs[mname] = evaluate_rankings_mlr(rk, qrels, splits["REPORT"], k=10)
                    conds[mname] = cv

                r_ref = _ndcg_map(pqs["composed_ridge"])
                u_ref = _ndcg_map(pqs["composed_A2_unscaled"])
                for mname, pq in pqs.items():
                    sm = summarize_per_query_metrics(pq)
                    bt_r = paired_bootstrap_delta(_ndcg_map(pq), r_ref, n_bootstrap=2000, seed=seed)
                    bt_u = paired_bootstrap_delta(_ndcg_map(pq), u_ref, n_bootstrap=2000, seed=seed)
                    rows.append(
                        {
                            "dataset": ds_name,
                            "hop_m": hop_m,
                            "source_encoder": sk,
                            "seed": int(seed),
                            "method": mname,
                            "report_ndcg_at_10": round(sm["mean_ndcg_at_10"], 6),
                            "delta_vs_ridge": round(bt_r["mean_delta"], 6),
                            "lcb95_vs_ridge": round(bt_r["lcb_95"], 6),
                            "delta_vs_unscaled": round(bt_u["mean_delta"], 6),
                            "cond_number": round(conds[mname], 4),
                            "metric_contract_sha256": contract_sha,
                        }
                    )

    with (output_dir / "f1_conformal_4hop.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Aggregate mean across seeds
    agg = defaultdict(list)
    agg_c = defaultdict(list)
    for r in rows:
        agg[(r["dataset"], r["hop_m"], r["method"])].append(r["report_ndcg_at_10"])
        agg_c[(r["dataset"], r["hop_m"], r["method"])].append(r["cond_number"])

    wins_vs_ridge = 0
    wins_vs_unsc = 0
    total_cells = 0
    md = [
        "# Flagship #1 — Conformal Radius-Matched Centered Procrustes (`A2c`) 4-Hop Validation (`mlr-v1`)",
        "",
        f"- **Git SHA:** `{current_git_sha(REPO_ROOT)}`",
        f"- **Wall time:** `{time.perf_counter() - t_start:.2f}s`",
        "- **Formula:** `y = scale * ((x - mu_x) @ Q) + mu_y`, `scale = ||Y_c||_F / ||X_c||_F`, exact `cond = 1.00`",
        "",
        "| Dataset | Hop | Comp Ridge | Comp A2 Unscaled | **Comp A2c Conformal (Unif)** | **Comp A2c Conformal (k-Means++)** | Δ(A2c_Unif vs Ridge) | Δ(A2c_Unif vs Unsc) | Ridge cond | A2c cond |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ds_name in datasets:
        for hop in (1, 2, 3, 4):
            vr = float(np.mean(agg[(ds_name, hop, "composed_ridge")]))
            vu = float(np.mean(agg[(ds_name, hop, "composed_A2_unscaled")]))
            vc = float(np.mean(agg[(ds_name, hop, "composed_A2c_conformal_unif")]))
            vk = float(np.mean(agg[(ds_name, hop, "composed_A2c_conformal_kmeanspp")]))
            cr = float(np.mean(agg_c[(ds_name, hop, "composed_ridge")]))
            cc = float(np.mean(agg_c[(ds_name, hop, "composed_A2c_conformal_unif")]))
            total_cells += 1
            if max(vc, vk) > vr:
                wins_vs_ridge += 1
            if vc > vu:
                wins_vs_unsc += 1
            md.append(
                f"| {ds_name} | {hop} | {vr:.4f} | {vu:.4f} | **{vc:.4f}** | **{vk:.4f}** | {vc - vr:+.4f} | {vc - vu:+.4f} | {cr:.2e} | {cc:.2f} |"
            )

    md.extend(
        [
            "",
            f"- **Conformal `A2c` wins vs Composed Ridge:** `{wins_vs_ridge}/{total_cells}` cells across all 6 BEIR domains × 4 hops",
            f"- **Conformal `A2c` (Unif) beats Unscaled `A2`:** `{wins_vs_unsc}/{total_cells}` cells",
        ]
    )
    (output_dir / "f1_conformal_4hop_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    manifest = {
        "job": "flagship1_conformal_4hop",
        "git_sha": current_git_sha(REPO_ROOT),
        "datasets": list(datasets),
        "seeds": [int(s) for s in seeds],
        "wins_vs_ridge": wins_vs_ridge,
        "wins_vs_unscaled": wins_vs_unsc,
        "total_cells": total_cells,
        "wall_seconds": round(time.perf_counter() - t_start, 2),
        "telemetry": process_telemetry(),
    }
    m_sha = write_json_atomic(output_dir / "f1_conformal_4hop_manifest.json", manifest)
    write_json_atomic(
        output_dir / "report_receipt.json",
        {"job": "flagship1_conformal_4hop", "status": "COMPLETED", "manifest_sha256": m_sha},
    )
    print(f"[F1-CONF4] COMPLETE in {manifest['wall_seconds']}s -> {output_dir}", flush=True)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument(
        "--datasets",
        nargs="+",
        default=["SciFact", "NFCorpus", "FiQA2018", "ArguAna", "SCIDOCS", "Quora"],
    )
    ap.add_argument("--seeds", nargs="+", type=int, default=[42, 137, 2718])
    args = ap.parse_args()
    run_conformal_4hop(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        datasets=args.datasets,
        seeds=args.seeds,
    )


if __name__ == "__main__":
    main()
