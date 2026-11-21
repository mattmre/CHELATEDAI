"""Expanded Flagship #1 Generalization Stress Test on 6 BEIR Datasets & 5-Encoder 4-Hop Chain.

Validates the packaged `spherical_migration_bridge` stack across 3 original + 3 unseen BEIR domains:
  - SciFact, NFCorpus, FiQA2018, ArguAna, SCIDOCS, Quora (capped at 40k docs incl. all qrels positives)
  - 5-encoder / 4-hop lineage chain: E0 (MiniLM-384) -> E1 (BGE-small-384) -> E2 (E5-small-384)
                                     -> E3 (MPNet-768) -> E4 (BGE-base-768)
  - Single-hop swaps: E0->E1 (384->384 cone), E0->E3 (384->768 cross-dim), E3->E4 (768->768 large cone)
  - Low-anchor (n=150) k-Means++ vs Uniform i.i.d. + Full-anchor (50%) A3 Whitened Procrustes (+CSLS) vs Ridge
  - 20% Selective Backfill: Night-1 Ridge+ZScore+FastFill vs Packaged A3+Quantile+Exposure*Distortion
  - 4-Hop Composition (m=1..4): Composed Ridge vs Composed Centered Procrustes (k-Means++ vs Uniform)
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
from spherical_migration_bridge import (  # noqa: E402
    CenteredProcrustesBridge,
    WhitenedSphericalProcrustesBridge,
    combine_exposure_and_distortion_order,
    compose_centered_bridges,
    compute_spherical_distortion_scores,
    score_quantile_calibrated_backfill,
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


def _tune_whitened(d_src, d_tgt, q_dev_src, dev_qids, doc_ids, qrels, fit_idx):
    best_b = None
    best_sc = -1.0
    for sh in (0.05, 0.20, 0.50):
        bc = WhitenedSphericalProcrustesBridge.fit(d_src, d_tgt, fit_idx=fit_idx, shrink=sh)
        rd, _ = score_dense_rankings(bc.transform(q_dev_src), d_tgt, dev_qids, doc_ids, k=10)
        sc = summarize_per_query_metrics(evaluate_rankings_mlr(rd, qrels, dev_qids, k=10))["mean_ndcg_at_10"]
        if sc > best_sc:
            best_sc = sc
            best_b = bc
    assert best_b is not None
    return best_b


def run_expanded_flagship1(
    *,
    output_dir: Path,
    cache_dir: Path,
    datasets: Sequence[str],
    seeds: Sequence[int],
    max_docs_large: int = 40_000,
    max_queries_cap: int = 1200,
    batch_size: int = 128,
) -> Dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be absent or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    chain = ["E0", "E1", "E2", "E3", "E4"]
    swaps = [
        ("E0_to_E1_384_cone", "E0", "E1"),
        ("E0_to_E3_384_768", "E0", "E3"),
        ("E3_to_E4_768_cone", "E3", "E4"),
    ]

    sh_rows: List[Dict[str, Any]] = []
    low_anc_rows: List[Dict[str, Any]] = []
    mh4_rows: List[Dict[str, Any]] = []
    ds_manifests: Dict[str, Any] = {}

    for ds_name in datasets:
        print(f"[F1-EXP] Loading {ds_name}...", flush=True)
        doc_cap = max_docs_large if ds_name in ("Quora", "FiQA2018") else None
        doc_ids, doc_texts, query_ids, query_texts, qrels, qrels_sha = load_beir_dataset(
            ds_name, max_docs=doc_cap
        )
        # Deterministic query cap for very large query sets (e.g. Quora 10k queries) so eval stays fast
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
        split_sha = sha256_json(splits)
        _, contract_sha = metric_contract_digest(qrels_sha, k=10, gain_rule="graded")

        enc: Dict[str, Dict[str, Any]] = {}
        for ek in chain:
            print(f"[F1-EXP] Encoding/loading {ds_name} x {ek} ({ENCODERS[ek]})...", flush=True)
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

        ds_manifests[ds_name] = {
            "doc_count": len(doc_ids),
            "query_count": len(query_ids),
            "qrels_sha256": qrels_sha,
            "split_sha256": split_sha,
            "metric_contract_sha256": contract_sha,
        }

        # 1. Single-hop swaps: Night-1 baselines vs Packaged Stack (A2/A3 + CSLS + B1/B3 Backfill)
        for sw_label, old_k, new_k in swaps:
            d_old, q_old = enc[old_k]["docs"], enc[old_k]["queries"]
            d_new, q_new = enc[new_k]["docs"], enc[new_k]["queries"]
            fastfill_order = compute_fastfill_backfill_order(q_old[dev_idx], d_old, top_m=50)

            r_old, _ = score_dense_rankings(q_old[rep_idx], d_old, splits["REPORT"], doc_ids, k=10)
            pq_old = evaluate_rankings_mlr(r_old, qrels, splits["REPORT"], k=10)
            r_c2o, _ = score_dense_rankings(q_new[rep_idx], d_new, splits["REPORT"], doc_ids, k=10)
            pq_c2o = evaluate_rankings_mlr(r_c2o, qrels, splits["REPORT"], k=10)

            for seed in seeds:
                print(f"[F1-EXP] {ds_name} | {sw_label} | seed={seed}...", flush=True)
                anc_50 = select_qrels_independent_anchors(doc_ids, anchor_frac=0.50, seed=seed)

                # Night-1 baselines
                w_unc = fit_closed_form_svd_procrustes(d_new, d_old, fit_idx=anc_50)
                w_rid = _tune_ridge(d_new, d_old, q_new[dev_idx], splits["DEVELOPMENT"], doc_ids, qrels, anc_50)
                # Packaged stack: A2 Centered Procrustes + A3 Whitened CORAL Procrustes
                b_a2 = CenteredProcrustesBridge.fit(d_new, d_old, fit_idx=anc_50)
                b_a3 = _tune_whitened(d_new, d_old, q_new[dev_idx], splits["DEVELOPMENT"], doc_ids, qrels, anc_50)

                q_dev_r = q_new[dev_idx] @ w_rid
                q_rep_r = q_new[rep_idx] @ w_rid
                q_dev_a3 = b_a3.transform(q_new[dev_idx])
                q_rep_a3 = b_a3.transform(q_new[rep_idx])
                q_rep_a2 = b_a2.transform(q_new[rep_idx])

                arm_pqs: Dict[str, Dict[str, Dict[str, float]]] = {
                    "old_encoder_servable": pq_old,
                    "full_reembed_c2o": pq_c2o,
                }
                for aname, qm in (
                    ("uncentered_procrustes", q_new[rep_idx] @ w_unc),
                    ("ridge_baseline", q_rep_r),
                    ("packaged_A2_centered_proc", q_rep_a2),
                    ("packaged_A3_whitened_proc", q_rep_a3),
                ):
                    rk, _ = score_dense_rankings(qm, d_old, splits["REPORT"], doc_ids, k=10)
                    arm_pqs[aname] = evaluate_rankings_mlr(rk, qrels, splits["REPORT"], k=10)

                for aname, qm, qd in (
                    ("ridge_plus_csls", q_rep_r, q_dev_r),
                    ("packaged_A3_plus_csls", q_rep_a3, q_dev_a3),
                ):
                    rk, _ = score_dense_rankings(
                        qm, d_old, splits["REPORT"], doc_ids, k=10, csls_k=10, csls_reference_queries=qd
                    )
                    arm_pqs[aname] = evaluate_rankings_mlr(rk, qrels, splits["REPORT"], k=10)

                # 20% Backfill: Night-1 z-score+FastFill vs Packaged A3+Quantile+Exposure*Distortion
                n_bf = max(1, int(round(len(doc_ids) * 0.20)))
                r_bf0 = score_map_plus_backfill(
                    q_new[rep_idx], d_old, d_new, w_rid, fastfill_order[:n_bf],
                    splits["REPORT"], doc_ids, dev_queries_new=q_new[dev_idx], k=10, calibrate=True,
                )
                arm_pqs["backfill20_night1_ridge_zscore"] = evaluate_rankings_mlr(r_bf0, qrels, splits["REPORT"], k=10)

                mapped_docs_a3 = b_a3.transform(d_new)
                dist_order, _ = compute_spherical_distortion_scores(
                    d_new, d_old, mapped_docs_a3, anchor_pool_idx=anc_50
                )
                hyb_order = combine_exposure_and_distortion_order(fastfill_order, dist_order, exposure_weight=0.5)
                r_bf_pkg = score_quantile_calibrated_backfill(
                    q_new[rep_idx], q_rep_a3, d_old, d_new, hyb_order[:n_bf],
                    splits["REPORT"], doc_ids,
                    dev_queries_new=q_new[dev_idx], dev_queries_mapped=q_dev_a3, k=10,
                )
                arm_pqs["backfill20_packaged_A3_quantile_hyb"] = evaluate_rankings_mlr(
                    r_bf_pkg, qrels, splits["REPORT"], k=10
                )

                ridge_ref = _ndcg_map(arm_pqs["ridge_baseline"])
                bf0_ref = _ndcg_map(arm_pqs["backfill20_night1_ridge_zscore"])
                for aname, pq in arm_pqs.items():
                    sm = summarize_per_query_metrics(pq)
                    ref = bf0_ref if aname.startswith("backfill20_") else ridge_ref
                    bt = paired_bootstrap_delta(_ndcg_map(pq), ref, n_bootstrap=3000, seed=seed)
                    sh_rows.append(
                        {
                            "dataset": ds_name,
                            "swap": sw_label,
                            "seed": int(seed),
                            "arm": aname,
                            "report_ndcg_at_10": round(sm["mean_ndcg_at_10"], 6),
                            "delta_vs_ref": round(bt["mean_delta"], 6),
                            "delta_vs_ref_lcb95": round(bt["lcb_95"], 6),
                            "split_sha256": split_sha,
                            "metric_contract_sha256": contract_sha,
                        }
                    )

                # Low-anchor (n=150) test: Uniform i.i.d. vs Packaged P6 Spherical k-Means++
                n_low = min(150, len(doc_ids) // 2)
                rng_u = np.random.default_rng(seed + 19)
                idx_u = np.sort(rng_u.choice(len(doc_ids), size=n_low, replace=False))
                idx_km = select_spherical_kmeanspp_anchors(d_new, n_low, seed=seed)

                b_u = CenteredProcrustesBridge.fit(d_new, d_old, fit_idx=idx_u)
                b_km = CenteredProcrustesBridge.fit(d_new, d_old, fit_idx=idx_km)
                pq_u = evaluate_rankings_mlr(
                    score_dense_rankings(b_u.transform(q_new[rep_idx]), d_old, splits["REPORT"], doc_ids, k=10)[0],
                    qrels, splits["REPORT"], k=10,
                )
                pq_km = evaluate_rankings_mlr(
                    score_dense_rankings(b_km.transform(q_new[rep_idx]), d_old, splits["REPORT"], doc_ids, k=10)[0],
                    qrels, splits["REPORT"], k=10,
                )
                bt_low = paired_bootstrap_delta(_ndcg_map(pq_km), _ndcg_map(pq_u), n_bootstrap=3000, seed=seed)
                low_anc_rows.append(
                    {
                        "dataset": ds_name,
                        "swap": sw_label,
                        "anchor_budget": n_low,
                        "seed": int(seed),
                        "uniform_ndcg": round(summarize_per_query_metrics(pq_u)["mean_ndcg_at_10"], 6),
                        "kmeanspp_ndcg": round(summarize_per_query_metrics(pq_km)["mean_ndcg_at_10"], 6),
                        "delta_km_minus_unif": round(bt_low["mean_delta"], 6),
                        "lcb95": round(bt_low["lcb_95"], 6),
                    }
                )

        # 2. 4-Hop Chain Composition (E4 -> E3 -> E2 -> E1 -> E0) at n=500 anchors
        d0 = enc["E0"]["docs"]
        n_mh = min(500, len(doc_ids) // 2)
        for seed in seeds:
            print(f"[F1-EXP] 4-Hop chain {ds_name} | n={n_mh} | seed={seed}...", flush=True)
            hop_ridge: List[np.ndarray] = []
            hop_cent_u: List[CenteredProcrustesBridge] = []
            hop_cent_km: List[CenteredProcrustesBridge] = []

            for step in range(1, len(chain)):
                sk, tk = chain[step], chain[step - 1]
                rng_s = np.random.default_rng(seed + step * 97)
                idx_u = np.sort(rng_s.choice(len(doc_ids), size=n_mh, replace=False))
                idx_km = select_spherical_kmeanspp_anchors(enc[sk]["docs"], n_mh, seed=seed + step * 97)

                wr = _tune_ridge(
                    enc[sk]["docs"], enc[tk]["docs"], enc[sk]["queries"][dev_idx],
                    splits["DEVELOPMENT"], doc_ids, qrels, idx_u,
                )
                hop_ridge.append(wr)
                hop_cent_u.append(CenteredProcrustesBridge.fit(enc[sk]["docs"], enc[tk]["docs"], fit_idx=idx_u))
                hop_cent_km.append(CenteredProcrustesBridge.fit(enc[sk]["docs"], enc[tk]["docs"], fit_idx=idx_km))

            for hop_m in range(1, len(chain)):
                sk = chain[hop_m]
                q_src = enc[sk]["queries"][rep_idx]
                # Compose backward from hop_m -> ... -> 0
                w_r = hop_ridge[hop_m - 1]
                for idx_step in range(hop_m - 2, -1, -1):
                    w_r = w_r @ hop_ridge[idx_step]
                sv_r = np.linalg.svd(w_r, compute_uv=False)
                cond_r = float(sv_r[0] / max(sv_r[-1], 1e-12))

                b_cu = compose_centered_bridges([hop_cent_u[i] for i in range(hop_m - 1, -1, -1)])
                b_ckm = compose_centered_bridges([hop_cent_km[i] for i in range(hop_m - 1, -1, -1)])

                for mname, qm, cond_v in (
                    ("composed_ridge", q_src @ w_r, cond_r),
                    ("composed_centered_proc_uniform", b_cu.transform(q_src), b_cu.condition_number),
                    ("composed_centered_proc_kmeanspp", b_ckm.transform(q_src), b_ckm.condition_number),
                ):
                    rk, _ = score_dense_rankings(qm, d0, splits["REPORT"], doc_ids, k=10)
                    sm = summarize_per_query_metrics(evaluate_rankings_mlr(rk, qrels, splits["REPORT"], k=10))
                    mh4_rows.append(
                        {
                            "dataset": ds_name,
                            "hop_m": hop_m,
                            "source_encoder": sk,
                            "anchor_budget": n_mh,
                            "seed": int(seed),
                            "method": mname,
                            "report_ndcg_at_10": round(sm["mean_ndcg_at_10"], 6),
                            "cond_number": round(cond_v, 4),
                        }
                    )

    for fname, rows in (
        ("f1_expanded_singlehop.csv", sh_rows),
        ("f1_expanded_low_anchor.csv", low_anc_rows),
        ("f1_expanded_4hop.csv", mh4_rows),
    ):
        with (output_dir / fname).open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    # Markdown report
    md = [
        "# Flagship #1 Packaged Stack — Expanded 6-Dataset BEIR & 4-Hop Validation (`mlr-v1`)",
        "",
        f"- **Git SHA:** `{current_git_sha(REPO_ROOT)}`",
        f"- **Wall time:** `{round(time.perf_counter() - t_start, 2)}s`",
        "",
        "## 1. Single-Hop REPORT nDCG@10 Across 6 BEIR Domains × 3 Swaps (Mean Across 3 Seeds)",
        "",
        "| Dataset | Swap | Uncent Proc | Ridge | Packaged A2 CentProc | **Packaged A3 Whitened** | Ridge+CSLS | **Packaged A3+CSLS** | BF20 Night-1 | **BF20 Packaged A3+Quant+Hyb** | Native Old | Full C2O |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    g_sh: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    for r in sh_rows:
        g_sh[(r["dataset"], r["swap"], r["arm"])].append(float(r["report_ndcg_at_10"]))
    cols = [
        "uncentered_procrustes", "ridge_baseline", "packaged_A2_centered_proc",
        "packaged_A3_whitened_proc", "ridge_plus_csls", "packaged_A3_plus_csls",
        "backfill20_night1_ridge_zscore", "backfill20_packaged_A3_quantile_hyb",
        "old_encoder_servable", "full_reembed_c2o",
    ]
    for ds in datasets:
        for sw, _, _ in swaps:
            vals = [f"{sum(g_sh[(ds, sw, a)]) / len(g_sh[(ds, sw, a)]):.4f}" for a in cols]
            md.append(f"| {ds} | {sw} | " + " | ".join(vals) + " |")

    md += [
        "",
        "## 2. Low-Anchor (`n=150`) P6 Spherical k-Means++ vs. Uniform i.i.d.",
        "",
        "| Dataset | Swap | Uniform i.i.d. | Packaged k-Means++ | Mean Delta |",
        "|---|---|---:|---:|---:|",
    ]
    for ds in datasets:
        for sw, _, _ in swaps:
            sub = [r for r in low_anc_rows if r["dataset"] == ds and r["swap"] == sw]
            mu = sum(float(r["uniform_ndcg"]) for r in sub) / len(sub)
            mk = sum(float(r["kmeanspp_ndcg"]) for r in sub) / len(sub)
            md.append(f"| {ds} | {sw} | {mu:.4f} | {mk:.4f} | {mk - mu:+.4f} |")

    md += [
        "",
        "## 3. 4-Hop Lineage Chain (`E4 -> E3 -> E2 -> E1 -> E0`) at `n=500` Anchors",
        "",
        "| Dataset | Hop | Comp Ridge | Comp Centered Proc (Unif) | Comp Centered Proc (k-Means++) | Ridge cond | Centered cond |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for ds in datasets:
        for hm in (1, 2, 3, 4):
            sub_r = [r for r in mh4_rows if r["dataset"] == ds and int(r["hop_m"]) == hm and r["method"] == "composed_ridge"]
            sub_u = [r for r in mh4_rows if r["dataset"] == ds and int(r["hop_m"]) == hm and r["method"] == "composed_centered_proc_uniform"]
            sub_k = [r for r in mh4_rows if r["dataset"] == ds and int(r["hop_m"]) == hm and r["method"] == "composed_centered_proc_kmeanspp"]
            mr = sum(float(r["report_ndcg_at_10"]) for r in sub_r) / len(sub_r)
            mu = sum(float(r["report_ndcg_at_10"]) for r in sub_u) / len(sub_u)
            mk = sum(float(r["report_ndcg_at_10"]) for r in sub_k) / len(sub_k)
            cr = sum(float(r["cond_number"]) for r in sub_r) / len(sub_r)
            ck = sum(float(r["cond_number"]) for r in sub_k) / len(sub_k)
            md.append(f"| {ds} | {hm} | {mr:.4f} | {mu:.4f} | {mk:.4f} | {cr:.2e} | {ck:.2f} |")

    (output_dir / "f1_expanded_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    manifest = {
        "job": "flagship1_expanded_6beir_4hop",
        "git_sha": current_git_sha(REPO_ROOT),
        "datasets": list(datasets),
        "chain": chain,
        "seeds": [int(s) for s in seeds],
        "dataset_manifests": ds_manifests,
        "wall_seconds": round(time.perf_counter() - t_start, 2),
        "telemetry": process_telemetry(),
    }
    manifest_sha = write_json_atomic(output_dir / "f1_expanded_manifest.json", manifest)
    write_json_atomic(
        output_dir / "report_receipt.json",
        {"job": "flagship1_expanded_6beir_4hop", "status": "COMPLETED", "manifest_sha256": manifest_sha},
    )
    print(f"[F1-EXP] COMPLETE in {manifest['wall_seconds']}s -> {output_dir}", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Expanded Flagship #1 6-BEIR & 4-Hop Validation")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/chelated_night1_emb_cache"))
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["SciFact", "NFCorpus", "FiQA2018", "ArguAna", "SCIDOCS", "Quora"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 137, 2718])
    parser.add_argument("--batch-size", type=int, default=128)
    args = parser.parse_args()
    run_expanded_flagship1(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        datasets=args.datasets,
        seeds=args.seeds,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
