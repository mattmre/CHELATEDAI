"""Science Runner-Up IQ-01 Falsifier: Pre-Outcome Non-Normal / Commutator Geometry vs. Ordered Composition Harm.

Tests the exact core hypothesis of IQ-01 (01-composition-prediction.md):
  When composing two linear/low-rank embedding adapters A and B sequentially (`A o B` vs `B o A`),
  do STRICTLY PRE-OUTCOME matrix-geometry features computed from (A, B) alone — with ZERO retrieval
  evaluation of the composed product — predict:
    1. Order asymmetry `|nDCG(A o B) - nDCG(B o A)|`
    2. Composition harm `min(nDCG(A), nDCG(B)) - nDCG(A o B)`
  better than trivial norm/cosine baselines?

Pre-outcome features tested (all O(d^3) or O(r^2 d), zero query evaluation):
  - Commutator Frobenius norm:     ||A B - B A||_F
  - Relative commutator norm:      ||A B - B A||_F / (||A||_F ||B||_F)
  - Henrici departure from normal: sqrt(||M||_F^2 - sum |eig(M)|^2) for M = A B
  - Subspace overlap / alignment:  ||U_A^T U_B||_F (principal angle overlap of adapter updates dA = A - I, dB = B - I)
  - Spectral norm product vs prod: ||A B||_2 - 1.0  (non-orthogonality / amplification)

Adapter bank (real data, complete-qrels mlr-v1 on SciFact, NFCorpus, FiQA2018):
  Fits 8 diverse adapters on real cached E0/E1/E2 embeddings (ridge domain-slice adapters,
  asymmetric non-normal rank-r perturbations, orthogonal Procrustes rotations, shrinkage/anisotropic maps),
  evaluates ALL 8 x 7 = 56 ordered pairs (28 unordered pairs x 2 orders) per dataset = 168 real ordered compositions!
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from m1_baseline_ladder import (  # noqa: E402
    evaluate_rankings_mlr,
    fit_closed_form_ridge,
    fit_closed_form_svd_procrustes,
    score_dense_rankings,
    split_query_ids,
)
from scripts.night1_common import (  # noqa: E402
    ENCODERS,
    current_git_sha,
    encode_or_load_embeddings,
    load_beir_dataset,
    process_telemetry,
    summarize_per_query_metrics,
    write_json_atomic,
)


def _henrici_non_normality(m: np.ndarray) -> float:
    """Exact Henrici departure from normality: sqrt(max(0, ||M||_F^2 - sum |lambda_i|^2))."""
    frob_sq = float(np.sum(m * m))
    eigs = np.linalg.eigvals(m)
    eig_sq = float(np.sum(np.abs(eigs) ** 2))
    return float(np.sqrt(max(0.0, frob_sq - eig_sq)))


def _build_diverse_adapter_bank(
    d0: np.ndarray,
    d1: np.ndarray,
    d2: np.ndarray,
    seed: int = 42,
) -> Dict[str, np.ndarray]:
    """Build 8 realistic 384x384 linear adapters spanning normal/orthogonal and non-normal regimes."""
    d = d0.shape[1]
    n = d0.shape[0]
    rng = np.random.default_rng(seed)
    idx_a = np.sort(rng.choice(n, size=min(400, n // 2), replace=False))
    idx_b = np.sort(rng.choice(n, size=min(400, n // 2), replace=False))
    eye = np.eye(d, dtype=np.float64)

    # 1-2: Orthogonal Procrustes rotations (normal, cond=1)
    p01 = fit_closed_form_svd_procrustes(d0, d1, fit_idx=idx_a)
    p10 = fit_closed_form_svd_procrustes(d1, d0, fit_idx=idx_b)
    # 3-4: Non-normal Ridge adapters (different regularizations -> varying spectral skew!)
    r_weak = fit_closed_form_ridge(d0, d1, fit_idx=idx_a, regularization=1e-2)
    r_strong = fit_closed_form_ridge(d1, d0, fit_idx=idx_b, regularization=1.0)
    # 5: Bridge through E2 (3rd encoder space -> strong cross-space rotation)
    r_e2 = fit_closed_form_ridge(d0, d2, fit_idx=idx_a, regularization=0.1)
    # 6: Near-identity low-rank symmetric update (normal perturbation)
    u_sym, _ = np.linalg.qr(rng.normal(size=(d, 16)))
    a_sym = eye + 0.25 * (u_sym @ u_sym.T)
    # 7: Non-normal nilpotent/shear low-rank LoRA-style update (A = I + U V^T with U perp V!)
    q_uv, _ = np.linalg.qr(rng.normal(size=(d, 32)))
    u_sh, v_sh = q_uv[:, :16], q_uv[:, 16:32]
    a_shear = eye + 0.60 * (u_sh @ v_sh.T)
    # 8: Strong anisotropic directional rescaling along top-32 data covariance axes
    c0 = (d0.T @ d0) / float(n)
    _, ev = np.linalg.eigh(c0)
    top32 = ev[:, -32:]
    a_aniso = eye + 0.50 * (top32 @ top32.T)

    # Mix gently toward identity for adapters 1-5 so compositions span mild-to-severe harm smoothly
    return {
        "orth_proc_01": 0.5 * eye + 0.5 * p01,
        "orth_proc_10": 0.5 * eye + 0.5 * p10,
        "ridge_weak_01": 0.4 * eye + 0.6 * r_weak,
        "ridge_strong_10": 0.4 * eye + 0.6 * r_strong,
        "ridge_e0_e2": 0.5 * eye + 0.5 * r_e2,
        "lora_symmetric": a_sym,
        "lora_nonnormal_shear": a_shear,
        "aniso_cov_stretch": a_aniso,
    }


def run_iq01_falsifier(
    *,
    output_dir: Path,
    cache_dir: Path,
    datasets: Sequence[str],
    seed: int = 42,
) -> Dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be absent or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    pair_rows: List[Dict[str, Any]] = []
    ds_corrs: List[Dict[str, Any]] = []

    for ds_name in datasets:
        print(f"[IQ-01] Running commutator & non-normal falsifier on {ds_name}...", flush=True)
        doc_ids, doc_texts, query_ids, query_texts, qrels, _ = load_beir_dataset(ds_name)
        splits = split_query_ids(query_ids)
        qid_to_idx = {q: i for i, q in enumerate(query_ids)}
        rep_idx = [qid_to_idx[q] for q in splits["REPORT"]]

        enc = {}
        for ek in ("E0", "E1", "E2"):
            d_e, q_e, _ = encode_or_load_embeddings(
                dataset_name=ds_name, encoder_key=ek, encoder_name=ENCODERS[ek],
                doc_ids=doc_ids, doc_texts=doc_texts, query_ids=query_ids, query_texts=query_texts,
                cache_dir=cache_dir, batch_size=128,
            )
            enc[ek] = (d_e, q_e)

        d0, q0 = enc["E0"]
        d1, _ = enc["E1"]
        d2, _ = enc["E2"]
        q_rep = q0[rep_idx]

        # Native E0 baseline nDCG@10
        r_base, _ = score_dense_rankings(q_rep, d0, splits["REPORT"], doc_ids, k=10)
        base_ndcg = summarize_per_query_metrics(evaluate_rankings_mlr(r_base, qrels, splits["REPORT"], k=10))["mean_ndcg_at_10"]

        bank = _build_diverse_adapter_bank(d0, d1, d2, seed=seed)
        names = sorted(bank.keys())

        # Single-adapter nDCG@10
        single_ndcg: Dict[str, float] = {}
        for nm in names:
            rk, _ = score_dense_rankings(q_rep @ bank[nm], d0, splits["REPORT"], doc_ids, k=10)
            single_ndcg[nm] = summarize_per_query_metrics(evaluate_rankings_mlr(rk, qrels, splits["REPORT"], k=10))["mean_ndcg_at_10"]

        # All 56 ordered pairs (i != j)
        eye = np.eye(d0.shape[1], dtype=np.float64)
        ab_ndcg: Dict[Tuple[str, str], float] = {}
        for ni in names:
            for nj in names:
                if ni == nj:
                    continue
                m_ab = bank[ni] @ bank[nj]
                rk, _ = score_dense_rankings(q_rep @ m_ab, d0, splits["REPORT"], doc_ids, k=10)
                ab_ndcg[(ni, nj)] = summarize_per_query_metrics(evaluate_rankings_mlr(rk, qrels, splits["REPORT"], k=10))["mean_ndcg_at_10"]

        ds_sub_rows = []
        for ni in names:
            for nj in names:
                if ni == nj:
                    continue
                a, b = bank[ni], bank[nj]
                ab = a @ b
                ba = b @ a
                comm_frob = float(np.linalg.norm(ab - ba, ord="fro"))
                norm_prod = float(np.linalg.norm(a, ord="fro") * np.linalg.norm(b, ord="fro"))
                rel_comm = comm_frob / max(norm_prod, 1e-12)
                henrici_ab = _henrici_non_normality(ab)
                henrici_diff = abs(henrici_ab - _henrici_non_normality(ba))
                sv_ab = np.linalg.svd(ab, compute_uv=False)
                spec_amp = float(sv_ab[0] / max(sv_ab[-1], 1e-12))
                da, db = a - eye, b - eye
                subspace_overlap = float(np.linalg.norm(da.T @ db, ord="fro"))
                trivial_norm_sum = float(np.linalg.norm(da, ord="fro") + np.linalg.norm(db, ord="fro"))

                ndcg_ab = ab_ndcg[(ni, nj)]
                ndcg_ba = ab_ndcg[(nj, ni)]
                order_asym = abs(ndcg_ab - ndcg_ba)
                comp_harm = max(0.0, min(single_ndcg[ni], single_ndcg[nj]) - ndcg_ab)
                signed_order_delta = ndcg_ab - ndcg_ba

                row = {
                    "dataset": ds_name,
                    "adapter_a": ni,
                    "adapter_b": nj,
                    "base_ndcg": round(base_ndcg, 6),
                    "ndcg_a": round(single_ndcg[ni], 6),
                    "ndcg_b": round(single_ndcg[nj], 6),
                    "ndcg_ab": round(ndcg_ab, 6),
                    "ndcg_ba": round(ndcg_ba, 6),
                    "order_asymmetry": round(order_asym, 6),
                    "signed_order_delta": round(signed_order_delta, 6),
                    "composition_harm": round(comp_harm, 6),
                    "commutator_frob": round(comm_frob, 6),
                    "rel_commutator": round(rel_comm, 6),
                    "henrici_ab": round(henrici_ab, 6),
                    "henrici_diff": round(henrici_diff, 6),
                    "spec_cond_ab": round(spec_amp, 6),
                    "subspace_overlap": round(subspace_overlap, 6),
                    "trivial_norm_sum": round(trivial_norm_sum, 6),
                }
                pair_rows.append(row)
                ds_sub_rows.append(row)

        def _rho(feat_key: str, target_key: str) -> float:
            x = [r[feat_key] for r in ds_sub_rows]
            y = [r[target_key] for r in ds_sub_rows]
            val = spearmanr(x, y).statistic
            return round(float(val) if np.isfinite(val) else 0.0, 4)

        ds_corrs.append(
            {
                "dataset": ds_name,
                "n_ordered_pairs": len(ds_sub_rows),
                "mean_order_asymmetry": round(float(np.mean([r["order_asymmetry"] for r in ds_sub_rows])), 4),
                "max_order_asymmetry": round(float(np.max([r["order_asymmetry"] for r in ds_sub_rows])), 4),
                "rho_commutator_vs_asym": _rho("commutator_frob", "order_asymmetry"),
                "rho_henrici_diff_vs_asym": _rho("henrici_diff", "order_asymmetry"),
                "rho_trivial_norm_vs_asym": _rho("trivial_norm_sum", "order_asymmetry"),
                "rho_commutator_vs_harm": _rho("commutator_frob", "composition_harm"),
                "rho_henrici_vs_harm": _rho("henrici_ab", "composition_harm"),
                "rho_spec_cond_vs_harm": _rho("spec_cond_ab", "composition_harm"),
                "rho_trivial_norm_vs_harm": _rho("trivial_norm_sum", "composition_harm"),
            }
        )

    with (output_dir / "iq01_ordered_pairs.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(pair_rows[0].keys()))
        w.writeheader()
        w.writerows(pair_rows)

    md = [
        "# Science Runner-Up `IQ-01` — Pre-Outcome Commutator & Non-Normal Geometry Falsifier (`mlr-v1`)",
        "",
        f"- **Git SHA:** `{current_git_sha(REPO_ROOT)}`",
        f"- **Total real ordered compositions evaluated:** `{len(pair_rows)}` (`56` ordered pairs × `{len(datasets)}` BEIR datasets)",
        "",
        "## Spearman Rank Correlation (Pre-Outcome Matrix Geometry → Observed Order Asymmetry & Composition Harm)",
        "",
        "| Dataset | Max Order Asym | ρ(Commutator → Asym) | ρ(HenriciDiff → Asym) | ρ(TrivialNorm → Asym) | ρ(Commutator → Harm) | ρ(Henrici → Harm) | ρ(SpecCond → Harm) | ρ(TrivialNorm → Harm) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for c in ds_corrs:
        md.append(
            f"| {c['dataset']} | {c['max_order_asymmetry']:.4f} | **{c['rho_commutator_vs_asym']:+.4f}** "
            f"| {c['rho_henrici_diff_vs_asym']:+.4f} | {c['rho_trivial_norm_vs_asym']:+.4f} "
            f"| {c['rho_commutator_vs_harm']:+.4f} | {c['rho_henrici_vs_harm']:+.4f} "
            f"| **{c['rho_spec_cond_vs_harm']:+.4f}** | {c['rho_trivial_norm_vs_harm']:+.4f} |"
        )

    (output_dir / "iq01_falsifier_summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    manifest = {
        "job": "iq01_commutator_falsifier",
        "git_sha": current_git_sha(REPO_ROOT),
        "datasets": list(datasets),
        "correlations": ds_corrs,
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "telemetry": process_telemetry(),
    }
    msha = write_json_atomic(output_dir / "iq01_manifest.json", manifest)
    write_json_atomic(output_dir / "report_receipt.json", {"job": "iq01_commutator_falsifier", "status": "COMPLETED", "manifest_sha256": msha})
    print(f"[IQ-01] COMPLETE in {manifest['wall_seconds']}s -> {output_dir}", flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="IQ-01 Pre-Outcome Commutator & Non-Normal Falsifier")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("/tmp/chelated_night1_emb_cache"))
    parser.add_argument("--datasets", nargs="+", default=["SciFact", "NFCorpus", "FiQA2018"])
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run_iq01_falsifier(output_dir=args.output_dir, cache_dir=args.cache_dir, datasets=args.datasets, seed=args.seed)


if __name__ == "__main__":
    main()
