"""W22-06 migration-frontier analysis primitives (CPU, numpy only).

Preregistration: ``artifacts/pv7_w23b_w22_06/PREREGISTRATION.md``.

Sub-tests implemented here:
  (a) observation-identical top-k flip pairs on a DEV-calibrated empirical margin;
  (c) leave-one-domain-out centering-gain law;
  (d) whole-query top-k split-conformal containment (in-domain and leave-one-domain-out);
  (e) retained-bit / sample ledger (OpenAI #140).
Sub-test (b) (chain vs direct refit) is DEFERRED_SCOPE: the bge-small / mpnet / bge-m3 encoders
are present in the local HF cache, but their EMBEDDINGS of this corpus are not cached, and the
preregistration (and task) barred any encoding in this run.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from benchmark_utils import canonicalize_id
from chelated_pv7_spine.bridge_merge_harness import (
    BRIDGE_REGISTRY,
    CSLSMerge,
    l2_normalize,
)

K_DEFAULT = 10
CSLS_K = 10
ALPHA = 0.10
MARGIN_QUANTILE = 0.90
BOUNDARY_TOP = 50
SET_SIZE_FACTOR = 20
COVERAGE_DROP_PP = 0.05
LAW_SIGN_MIN = 0.80
LAW_R2_MIN = 0.5
SPLIT_SALT = "w22-06-cal"


# ---------------------------------------------------------------------------
# Serving pipeline P (Mode B, bridge x CSLS)
# ---------------------------------------------------------------------------


def serve_scores(
    bridge_name: str,
    *,
    src_anchors: np.ndarray,
    tgt_anchors: np.ndarray,
    queries_eval: np.ndarray,
    queries_dev: np.ndarray,
    docs_old: np.ndarray,
    docs_new: np.ndarray,
    backfill_mask: np.ndarray,
    dev_docs: Optional[np.ndarray] = None,
    csls_k: int = CSLS_K,
) -> Dict[str, Any]:
    """Return fused EVAL/DEV score matrices of pipeline P plus native (full re-index) scores.

    EVAL fusion is byte-for-byte the ``run_factorial_5x3`` CSLS path (DEV queries as the
    CSLS reference, ``strict_disjoint_dev=True``). DEV fusion applies the same merge with
    the DEV scores as their own reference (used only for margin calibration).
    """
    kwargs: Dict[str, Any] = {}
    if bridge_name == "ZeroMuPolarBridge":
        kwargs["scoring_mode"] = "dual_adap"
    bridge = BRIDGE_REGISTRY[bridge_name].fit(src_anchors, tgt_anchors, **kwargs)
    if bridge_name == "ZeroMuPolarBridge" and dev_docs is not None:
        b_eval = bridge.score_matrix(queries_eval, docs_old, dev_docs=dev_docs)
        b_dev = bridge.score_matrix(queries_dev, docs_old, dev_docs=dev_docs)
    else:
        b_eval = bridge.score_matrix(queries_eval, docs_old)
        b_dev = bridge.score_matrix(queries_dev, docs_old)
    docs_new_n = l2_normalize(docs_new)
    n_eval = l2_normalize(queries_eval) @ docs_new_n.T
    n_dev = l2_normalize(queries_dev) @ docs_new_n.T
    mask = np.asarray(backfill_mask, dtype=bool)
    merger = CSLSMerge(csls_k=csls_k)
    fused_eval = merger.merge(b_eval, n_eval, mask, bridged_ref=b_dev, native_ref=n_dev, strict_disjoint_dev=True)
    fused_dev = merger.merge(b_dev, n_dev, mask, bridged_ref=b_dev, native_ref=n_dev)
    return {
        "bridge": bridge,
        "fused_eval": fused_eval,
        "fused_dev": fused_dev,
        "native_eval": n_eval,
        "native_dev": n_dev,
    }


# ---------------------------------------------------------------------------
# Canonical ranking helpers
# ---------------------------------------------------------------------------


def doc_tiebreak_keys(doc_ids: Sequence[str]) -> np.ndarray:
    """Integer key whose order equals the canonical doc-id string order (harness tie-break)."""
    canon = [canonicalize_id(d) for d in doc_ids]
    order = sorted(range(len(canon)), key=lambda j: canon[j])
    keys = np.empty(len(canon), dtype=np.int64)
    keys[np.asarray(order, dtype=np.int64)] = np.arange(len(canon), dtype=np.int64)
    return keys


def full_ranking(row: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """Doc indices ordered by ``(-score, doc_id)``."""
    return np.lexsort((keys, -np.asarray(row, dtype=np.float64)))


def rank_positions(row: np.ndarray, keys: np.ndarray) -> np.ndarray:
    """1-indexed rank of every doc under the canonical ordering."""
    order = full_ranking(row, keys)
    pos = np.empty(order.size, dtype=np.int64)
    pos[order] = np.arange(1, order.size + 1, dtype=np.int64)
    return pos


def topk_indices(row: np.ndarray, keys: np.ndarray, k: int) -> np.ndarray:
    return full_ranking(row, keys)[: int(k)]


# ---------------------------------------------------------------------------
# (a) Empirical margin + observation-identical flip pairs
# ---------------------------------------------------------------------------


def calibrate_margin(
    fused_dev: np.ndarray,
    native_dev: np.ndarray,
    keys: np.ndarray,
    *,
    region_top: int = BOUNDARY_TOP,
    quantile: float = MARGIN_QUANTILE,
) -> Dict[str, float]:
    """OLS ``native ~ a + b*fused`` on the DEV boundary region; margin = q-quantile of |residual|."""
    f_vals: List[float] = []
    g_vals: List[float] = []
    for i in range(fused_dev.shape[0]):
        region = set(topk_indices(fused_dev[i], keys, region_top).tolist())
        region |= set(topk_indices(native_dev[i], keys, region_top).tolist())
        idx = np.asarray(sorted(region), dtype=np.int64)
        f_vals.extend(fused_dev[i, idx].tolist())
        g_vals.extend(native_dev[i, idx].tolist())
    f = np.asarray(f_vals, dtype=np.float64)
    g = np.asarray(g_vals, dtype=np.float64)
    design = np.column_stack([np.ones_like(f), f])
    coef, *_ = np.linalg.lstsq(design, g, rcond=None)
    a, b = float(coef[0]), float(coef[1])
    resid = g - (a + b * f)
    return {
        "a": a,
        "b": b,
        "margin": float(np.quantile(np.abs(resid), quantile)),
        "n_pairs": int(f.size),
        "residual_abs_median": float(np.median(np.abs(resid))),
    }


def flip_pair_stats(
    served_row: np.ndarray,
    native_row: np.ndarray,
    keys: np.ndarray,
    *,
    k: int,
    margin: float,
    a: float = 0.0,
    b: float = 1.0,
) -> Dict[str, int]:
    """Per-query counts of top-k membership errors and observation-identical flip pairs.

    ``served_row`` holds the RAW served (fused) scores: S_k is the served top-k under the
    pipeline's own ranking (prereg sections 1-2). The affine DEV calibration ``c = a + b*served``
    is used ONLY for the margin comparison |c_i - c_j| <= margin, never for ranking, so a
    negative slope ``b`` cannot invert the served top-k.
    """
    served = set(topk_indices(served_row, keys, k).tolist())
    native = set(topk_indices(native_row, keys, k).tolist())
    intruders = sorted(served - native)
    missing = sorted(native - served)
    calibrated = a + b * np.asarray(served_row, dtype=np.float64)
    n_pairs = 0
    n_explainable = 0
    for j in missing:
        hits = sum(1 for i in intruders if abs(float(calibrated[i]) - float(calibrated[j])) <= margin)
        n_pairs += hits
        if hits:
            n_explainable += 1
    return {
        "n_missing": len(missing),
        "n_flip_pairs": n_pairs,
        "n_missing_margin_explainable": n_explainable,
    }


def summarize_flips(per_query: Sequence[Mapping[str, int]]) -> Dict[str, Any]:
    n_q = len(per_query)
    n_missing = sum(r["n_missing"] for r in per_query)
    n_expl = sum(r["n_missing_margin_explainable"] for r in per_query)
    n_flip_q = sum(1 for r in per_query if r["n_flip_pairs"] > 0)
    return {
        "n_queries": n_q,
        "n_missing_total": n_missing,
        "n_missing_margin_explainable": n_expl,
        "frac_missing_margin_explainable": (n_expl / n_missing) if n_missing else None,
        "frac_queries_with_flip": (n_flip_q / n_q) if n_q else None,
        "frac_queries_with_any_error": (sum(1 for r in per_query if r["n_missing"] > 0) / n_q) if n_q else None,
        "mean_missing_per_query": (n_missing / n_q) if n_q else None,
    }


def decide_margin_claim(per_domain_frac: Mapping[str, Optional[float]], pooled_frac: Optional[float], pooled_flip_q: Optional[float]) -> Dict[str, Any]:
    n_ok = sum(1 for v in per_domain_frac.values() if v is not None and v >= 0.50)
    supported = pooled_frac is not None and pooled_frac >= 0.50 and n_ok >= 10
    return {
        "domains_with_frac_explainable_ge_0_50": n_ok,
        "n_domains": len(per_domain_frac),
        "pooled_frac_missing_margin_explainable": pooled_frac,
        "claim_errors_are_margin_level": "SUPPORTED" if supported else "NOT_SUPPORTED",
        "pooled_frac_queries_with_flip": pooled_flip_q,
        "margin_vulnerability_phenomenon": "PRESENT" if (pooled_flip_q is not None and pooled_flip_q >= 0.10) else "ABSENT",
    }


# ---------------------------------------------------------------------------
# (c) Centering-gain law, leave-one-domain-out
# ---------------------------------------------------------------------------


def heldout_fit_error(bridge: Any, src_heldout: np.ndarray, tgt_heldout: np.ndarray) -> float:
    """Mean ``1 - cos(bridge(src), tgt)`` on held-out paired rows."""
    mapped = bridge.transform(src_heldout, normalize=True)
    tgt = l2_normalize(np.asarray(tgt_heldout, dtype=np.float64))
    return float(np.mean(1.0 - np.sum(mapped * tgt, axis=1)))


def _ols(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
    design = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return float(coef[0]), float(coef[1])


def _sign(v: float) -> int:
    return 1 if v > 0 else (-1 if v < 0 else 0)


def lodo_law(x: Mapping[str, float], g: Mapping[str, float]) -> Dict[str, Any]:
    """Leave-one-domain-out OLS law ``g = alpha + beta*x`` with the frozen kill rule."""
    doms = sorted(g)
    if set(doms) != set(x):
        raise ValueError("predictor and outcome cover different domains")
    xv = np.array([x[d] for d in doms], dtype=np.float64)
    gv = np.array([g[d] for d in doms], dtype=np.float64)
    preds: Dict[str, float] = {}
    for i, d in enumerate(doms):
        keep = np.arange(len(doms)) != i
        a, b = _ols(xv[keep], gv[keep])
        preds[d] = a + b * xv[i]
    pv = np.array([preds[d] for d in doms], dtype=np.float64)
    sst = float(np.sum((gv - gv.mean()) ** 2))
    r2_lodo = 1.0 - float(np.sum((gv - pv) ** 2)) / sst if sst > 0 else float("nan")
    # A zero counts as a mismatch unless both are zero: plain sign equality encodes exactly that.
    matches = [_sign(preds[d]) == _sign(g[d]) for d in doms]
    sign_acc = float(np.mean(matches))
    a_all, b_all = _ols(xv, gv)
    fit_all = a_all + b_all * xv
    r2_in = 1.0 - float(np.sum((gv - fit_all) ** 2)) / sst if sst > 0 else float("nan")
    rx = np.argsort(np.argsort(xv)).astype(np.float64)
    rg = np.argsort(np.argsort(gv)).astype(np.float64)
    spearman = float(np.corrcoef(rx, rg)[0, 1]) if len(doms) > 2 else float("nan")
    killed = (sign_acc < LAW_SIGN_MIN) or not (r2_lodo >= LAW_R2_MIN)
    return {
        "domains": doms,
        "lodo_predictions": preds,
        "r2_lodo": r2_lodo,
        "sign_accuracy": sign_acc,
        "in_sample_alpha": a_all,
        "in_sample_beta": b_all,
        "in_sample_r2": r2_in,
        "spearman_x_g": spearman,
        "verdict": "KILL" if killed else "NOT_KILLED",
    }


# ---------------------------------------------------------------------------
# (d) Whole-query top-k conformal containment
# ---------------------------------------------------------------------------


def containment_score(served_row: np.ndarray, native_row: np.ndarray, keys: np.ndarray, k: int) -> int:
    """Smallest K such that the native top-k is contained in the served top-K."""
    pos = rank_positions(served_row, keys)
    native_top = topk_indices(native_row, keys, k)
    return int(pos[native_top].max())


def is_calibration_query(domain: str, qid: str) -> bool:
    digest = hashlib.sha256(f"{SPLIT_SALT}:{domain}:{canonicalize_id(qid)}".encode("utf-8")).digest()
    return digest[0] % 2 == 0


def conformal_k(cal_scores: Sequence[int], *, alpha: float, n_docs: int) -> int:
    """Split-conformal quantile: the ceil((n+1)(1-alpha))-th smallest score (trivial set if undefined)."""
    s = sorted(int(v) for v in cal_scores)
    n = len(s)
    idx = int(math.ceil((n + 1) * (1.0 - alpha)))
    if n == 0 or idx > n:
        return int(n_docs)
    return s[idx - 1]


def coverage(scores: Sequence[int], k_star: int) -> Optional[float]:
    vals = [int(v) for v in scores]
    return (sum(1 for v in vals if v <= k_star) / len(vals)) if vals else None


def conformal_study(
    per_domain: Mapping[str, Mapping[str, Any]],
    *,
    k: int = K_DEFAULT,
    alpha: float = ALPHA,
    groups: Optional[Mapping[str, str]] = None,
) -> Dict[str, Any]:
    """per_domain[d] = {"cal": [scores], "test": [scores], "n_docs": int}."""
    doms = sorted(per_domain)
    rows: Dict[str, Any] = {}
    for d in doms:
        rec = per_domain[d]
        k_in = conformal_k(rec["cal"], alpha=alpha, n_docs=rec["n_docs"])
        pooled_other = [v for o in doms if o != d for v in per_domain[o]["cal"]]
        k_ood = conformal_k(pooled_other, alpha=alpha, n_docs=rec["n_docs"])
        cov_in = coverage(rec["test"], k_in)
        cov_ood = coverage(rec["test"], k_ood)
        rows[d] = {
            "n_cal": len(rec["cal"]),
            "n_test": len(rec["test"]),
            "n_docs": rec["n_docs"],
            "k_star_in_domain": k_in,
            "k_star_lodo": k_ood,
            "coverage_in_domain": cov_in,
            "coverage_lodo": cov_ood,
            "coverage_drop": (cov_in - cov_ood) if (cov_in is not None and cov_ood is not None) else None,
        }
    limit = SET_SIZE_FACTOR * k
    k_in_all = [rows[d]["k_star_in_domain"] for d in doms]
    k_ood_all = [rows[d]["k_star_lodo"] for d in doms]
    mean_in = float(np.mean([rows[d]["coverage_in_domain"] for d in doms]))
    mean_ood = float(np.mean([rows[d]["coverage_lodo"] for d in doms]))
    med_in = float(np.median(k_in_all))
    med_ood = float(np.median(k_ood_all))
    reasons = {
        "median_in_domain_set_size_gt_limit": med_in > limit,
        "median_lodo_set_size_gt_limit": med_ood > limit,
        "in_minus_ood_coverage_gt_5pp": (mean_in - mean_ood) > COVERAGE_DROP_PP,
        "nominal_minus_ood_coverage_gt_5pp": ((1.0 - alpha) - mean_ood) > COVERAGE_DROP_PP,
    }
    reasons_max = dict(reasons)
    reasons_max["median_in_domain_set_size_gt_limit"] = max(k_in_all) > limit
    reasons_max["median_lodo_set_size_gt_limit"] = max(k_ood_all) > limit
    out: Dict[str, Any] = {
        "alpha": alpha,
        "k": k,
        "set_size_limit": limit,
        "per_domain": rows,
        "median_k_star_in_domain": med_in,
        "median_k_star_lodo": med_ood,
        "max_k_star_in_domain": int(max(k_in_all)),
        "max_k_star_lodo": int(max(k_ood_all)),
        "mean_coverage_in_domain": mean_in,
        "mean_coverage_lodo": mean_ood,
        "kill_reasons": reasons,
        "verdict": "KILL" if any(reasons.values()) else "NOT_KILLED",
        "secondary_verdict_if_max_instead_of_median": "KILL" if any(reasons_max.values()) else "NOT_KILLED",
    }
    if groups is not None:
        transfer: Dict[str, Any] = {}
        names = sorted(set(groups.values()))
        for src in names:
            for dst in names:
                if src == dst:
                    continue
                cal = [v for d in doms if groups[d] == src for v in per_domain[d]["cal"]]
                test = [v for d in doms if groups[d] == dst for v in per_domain[d]["test"]]
                n_docs = max(per_domain[d]["n_docs"] for d in doms)
                k_star = conformal_k(cal, alpha=alpha, n_docs=n_docs)
                transfer[f"{src}_to_{dst}"] = {"k_star": k_star, "coverage": coverage(test, k_star), "n_cal": len(cal), "n_test": len(test)}
        out["group_transfer"] = transfer
    return out


# ---------------------------------------------------------------------------
# (e) Retained-bit / sample ledger (#140)
# ---------------------------------------------------------------------------

META_BYTES_PER_ARRAY = 8


def _arr(name: str, n_floats: int) -> Dict[str, Any]:
    return {"item": name, "n_floats": int(n_floats), "bytes_fp32": 4 * int(n_floats), "bytes_fp64": 8 * int(n_floats)}


def index_set_bits(n_docs: int, n_selected: int) -> int:
    """Cheaper of an n-bit mask or an explicit index list."""
    width = max(1, int(math.ceil(math.log2(max(2, n_docs)))))
    return int(min(n_docs, n_selected * width))


def bit_ledger(
    bridge_name: str,
    *,
    d_src: int,
    d_tgt: int,
    n_docs: int,
    n_backfill: int,
    n_anchors: int,
    n_dev_queries: int,
    n_dev_ref_docs: int,
) -> Dict[str, Any]:
    """Retained artifacts and recovery work for one bridge x CSLS migration configuration."""
    params: List[Dict[str, Any]] = []
    if bridge_name == "UncenteredProcrustes":
        params.append(_arr("W", d_src * d_tgt))
    elif bridge_name == "CenteredProcrustes":
        params += [_arr("Q", d_src * d_tgt), _arr("mu_x", d_src), _arr("mu_y", d_tgt), _arr("scale", 1)]
    elif bridge_name == "ZeroMuPolarBridge":
        params += [
            _arr("Q", d_src * d_tgt),
            _arr("mu_x", d_src),
            _arr("mu_d_dev", d_tgt),
            _arr("ema_mu_q_state", d_src),
            _arr("per_doc_centered_norm_sidecar_fp32", n_docs - n_backfill),
        ]
    else:
        raise ValueError(f"no ledger for {bridge_name!r}")
    n_legacy = n_docs - n_backfill
    csls = [_arr("csls_r_d_legacy", n_legacy), _arr("csls_r_d_native", n_backfill)]
    backfill = [_arr("native_backfill_embeddings", n_backfill * d_src)]
    mask_bits = index_set_bits(n_docs, n_backfill)
    n_arrays = len(params) + len(csls) + len(backfill) + 1
    meta_bytes = META_BYTES_PER_ARRAY * n_arrays
    added_fp32 = sum(p["bytes_fp32"] for p in params + csls + backfill) + int(math.ceil(mask_bits / 8)) + meta_bytes
    if bridge_name == "ZeroMuPolarBridge":
        # The sidecar is stored at fp32 by construction (build_float32_sidecar).
        added_fp64 = sum(p["bytes_fp64"] for p in params + csls + backfill if not p["item"].startswith("per_doc")) + 4 * n_legacy
    else:
        added_fp64 = sum(p["bytes_fp64"] for p in params + csls + backfill)
    added_fp64 += int(math.ceil(mask_bits / 8)) + meta_bytes
    bridge_bits_fp32 = 32 * sum(p["n_floats"] for p in params if not p["item"].startswith("per_doc"))
    full_reindex_fp32 = 4 * n_docs * d_src
    legacy_fp32 = 4 * n_docs * d_tgt
    encode_work = n_anchors + n_dev_queries + n_backfill
    return {
        "bridge": bridge_name,
        "bridge_params": params,
        "csls_state": csls,
        "backfill_payload": backfill,
        "backfill_index_bits": mask_bits,
        "precision_metadata_bytes": meta_bytes,
        "added_bytes_fp32": int(added_fp32),
        "added_bytes_fp64": int(added_fp64),
        "already_retained_legacy_index_bytes_fp32": int(legacy_fp32),
        "full_reindex_bytes_fp32": int(full_reindex_fp32),
        "added_over_full_reindex_fp32": added_fp32 / full_reindex_fp32,
        "bridge_param_bits_fp32": int(bridge_bits_fp32),
        "bridge_param_bits_over_dsrc_dtgt_32": bridge_bits_fp32 / (32.0 * d_src * d_tgt),
        "samples": {"anchors": n_anchors, "dev_queries": n_dev_queries, "dev_ref_docs": n_dev_ref_docs},
        "new_encoder_forward_passes": int(encode_work),
        "full_reindex_forward_passes": int(n_docs),
        "work_over_full_reindex": encode_work / n_docs,
    }
