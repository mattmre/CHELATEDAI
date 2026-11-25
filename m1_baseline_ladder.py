"""Stage 1 executable baseline ladder (M1 + #355 hand-set floor) and fixed M2 gap preflight.

Enforces the repository's mechanical baseline-comparison contract:
1. Canonical complete-qrels retrieval metrics (mlr-v1) via benchmark_utils.
2. Deterministic SHA-256 DEVELOPMENT / SELECT / REPORT query splits and
   qrels-independent corpus anchor splits.
3. Full M1 baseline ladder:
   (a) identity_frozen_c0 (explicit INCOMPATIBLE_DIMENSION when dims differ)
   (b) old_encoder_servable (excluded from eligible_arms when old_encoder_retired=True)
   (h0) #355 hand-set 1-3 parameter floor (fixed CSLS k=10; optional BM25+dense hybrid)
   (c) closed_form_svd_procrustes (float64 SVD orthogonal/semi-orthogonal fit)
   (d) closed_form_ridge (DEV-only lambda selection)
   (e) budgeted_backfill(b) (FastFill-style DEV query-exposure priority + uniform control)
   (f) map_plus_backfill(b) (bridged un-backfilled + native backfilled with DEV calibration)
   (g) full_reembed_c2o (costed reference ONLY; segregated from eligible_arms)
4. Fixed M2 gap preflight that strictly separates reference_arm from eligible_arms
   and refuses non-DEVELOPMENT splits.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from benchmark_utils import (
    build_metric_contract_manifest,
    canonicalize_id,
    mean_average_precision_at_k,
    mean_reciprocal_rank,
    ndcg_at_k,
    recall_at_k,
)

DEFAULT_RIDGE_LAMBDAS: Tuple[float, ...] = (1e-3, 1e-2, 1e-1, 1.0, 10.0)
DEFAULT_BACKFILL_BUDGETS: Tuple[float, ...] = (0.05, 0.10, 0.20, 0.50)


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    safe = np.where(norms > 1e-12, norms, 1.0)
    return arr / safe


def deterministic_hash_bucket(item_id: str, salt: str = "m1-split-v1") -> float:
    """Map an ID deterministically to [0.0, 1.0) via SHA-256."""
    digest = hashlib.sha256(f"{salt}:{canonicalize_id(item_id)}".encode("utf-8")).digest()
    value = int.from_bytes(digest[:8], "big")
    return float(value / float(1 << 64))


def split_query_ids(
    query_ids: Sequence[str],
    *,
    dev_frac: float = 0.40,
    select_frac: float = 0.30,
    salt: str = "m1-query-split-v1",
) -> Dict[str, List[str]]:
    """Deterministically partition unique query IDs into DEVELOPMENT / SELECT / REPORT."""
    if dev_frac <= 0.0 or select_frac <= 0.0 or (dev_frac + select_frac) >= 1.0:
        raise ValueError("dev_frac and select_frac must be positive and sum to < 1.0")
    canonical = sorted({canonicalize_id(qid) for qid in query_ids})
    splits: Dict[str, List[str]] = {"DEVELOPMENT": [], "SELECT": [], "REPORT": []}
    for qid in canonical:
        u = deterministic_hash_bucket(qid, salt=salt)
        if u < dev_frac:
            splits["DEVELOPMENT"].append(qid)
        elif u < dev_frac + select_frac:
            splits["SELECT"].append(qid)
        else:
            splits["REPORT"].append(qid)
    return splits


def select_qrels_independent_anchors(
    doc_ids: Sequence[str],
    *,
    anchor_frac: float = 0.50,
    seed: int = 42,
    max_anchors: Optional[int] = None,
) -> np.ndarray:
    """Select corpus anchor indices deterministically with zero qrels access."""
    if not (0.0 < anchor_frac <= 1.0):
        raise ValueError("anchor_frac must be in (0, 1]")
    n_docs = len(doc_ids)
    if n_docs == 0:
        raise ValueError("doc_ids must be non-empty")
    order = sorted(
        range(n_docs),
        key=lambda idx: deterministic_hash_bucket(doc_ids[idx], salt=f"anchor-seed-{seed}"),
    )
    count = max(1, int(round(n_docs * anchor_frac)))
    if max_anchors is not None:
        count = min(count, int(max_anchors))
    return np.asarray(sorted(order[:count]), dtype=np.int64)


def fit_closed_form_svd_procrustes(
    source: np.ndarray,
    target: np.ndarray,
    fit_idx: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Closed-form orthogonal (or semi-orthogonal Stiefel) Procrustes map W: source @ W ~= target.

    Solves min_{W^T W = I or W W^T = I} ||source @ W - target||_F^2 in float64 via SVD:
        source^T @ target = U @ S @ V^T  =>  W* = U @ V^T
    """
    src = np.asarray(source, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    if fit_idx is not None:
        src = src[fit_idx]
        tgt = tgt[fit_idx]
    if src.shape[0] != tgt.shape[0] or src.shape[0] == 0:
        raise ValueError("source and target must have identical non-zero row counts")
    cross = src.T @ tgt
    u, _s, vt = np.linalg.svd(cross, full_matrices=False)
    return u @ vt


def fit_closed_form_ridge(
    source: np.ndarray,
    target: np.ndarray,
    fit_idx: Optional[np.ndarray] = None,
    *,
    regularization: float = 1e-2,
) -> np.ndarray:
    """Closed-form ridge map W solving (X^T X + lambda I) W = X^T Y in float64."""
    if regularization < 0.0 or not math.isfinite(regularization):
        raise ValueError("regularization must be finite and >= 0")
    src = np.asarray(source, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    if fit_idx is not None:
        src = src[fit_idx]
        tgt = tgt[fit_idx]
    if src.shape[0] != tgt.shape[0] or src.shape[0] == 0:
        raise ValueError("source and target must have identical non-zero row counts")
    dim = src.shape[1]
    if src.shape[0] < dim and regularization > 0.0:
        # Exact O(n^2 d) Woodbury identity: X^T (X X^T + lam I)^{-1} Y
        gram = src @ src.T + float(regularization) * np.eye(src.shape[0], dtype=np.float64)
        return src.T @ np.linalg.solve(gram, tgt)
    return np.linalg.solve(
        src.T @ src + float(regularization) * np.eye(dim, dtype=np.float64),
        src.T @ tgt,
    )


def fit_closed_form_centered_ridge(
    source: np.ndarray,
    target: np.ndarray,
    fit_idx: Optional[np.ndarray] = None,
    *,
    regularization: float = 1e-2,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Centered Affine Ridge with exact intercept: y = (x - mu_x) @ W + mu_y.

    Uses exact O(n^2 d) Woodbury solve when n < d (avoids d^3 blowup at d=4096).
    """
    if regularization < 0.0 or not math.isfinite(regularization):
        raise ValueError("regularization must be finite and >= 0")
    src = np.asarray(source, dtype=np.float64)
    tgt = np.asarray(target, dtype=np.float64)
    if fit_idx is not None:
        src = src[fit_idx]
        tgt = tgt[fit_idx]
    if src.shape[0] != tgt.shape[0] or src.shape[0] < 2:
        raise ValueError("source and target must have identical row counts >= 2")
    mu_x = src.mean(axis=0, keepdims=True)
    mu_y = tgt.mean(axis=0, keepdims=True)
    xc = src - mu_x
    yc = tgt - mu_y
    w = fit_closed_form_ridge(xc, yc, regularization=regularization)
    return mu_x, mu_y, w


def score_dense_rankings(
    query_vectors: np.ndarray,
    doc_vectors: np.ndarray,
    query_ids: Sequence[str],
    doc_ids: Sequence[str],
    *,
    k: int = 10,
    csls_k: Optional[int] = None,
    csls_reference_queries: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, List[str]], Dict[str, np.ndarray]]:
    """Score queries against docs using canonical score-desc / doc_id-asc tie-breaking.

    Optionally applies #355 hand-set CSLS hubness suppression (1 parameter: csls_k).
    """
    q_norm = _normalize_rows(query_vectors)
    d_norm = _normalize_rows(doc_vectors)
    if q_norm.shape[1] != d_norm.shape[1]:
        raise ValueError(
            f"INCOMPATIBLE_DIMENSION: query dim {q_norm.shape[1]} != doc dim {d_norm.shape[1]}"
        )
    sims = q_norm @ d_norm.T

    if csls_k is not None:
        ref_q = _normalize_rows(csls_reference_queries) if csls_reference_queries is not None else q_norm
        eff_k = max(1, min(int(csls_k), ref_q.shape[0]))
        ref_sims = ref_q @ d_norm.T  # [n_ref_q, n_docs]
        if eff_k < ref_sims.shape[0]:
            top_doc_part = np.partition(ref_sims, ref_sims.shape[0] - eff_k, axis=0)[-eff_k:, :]
        else:
            top_doc_part = ref_sims
        r_d = top_doc_part.mean(axis=0)  # [n_docs]
        sims = 2.0 * sims - r_d[None, :]

    canonical_docs = [canonicalize_id(d) for d in doc_ids]
    rankings: Dict[str, List[str]] = {}
    top_scores: Dict[str, np.ndarray] = {}
    n_docs = len(canonical_docs)
    k_eff = min(k, n_docs)

    for row_idx, qid in enumerate(query_ids):
        row = sims[row_idx]
        # Deterministic exact global canonical tie-break: (-score, doc_id), safe under arbitrary ties at cutoff
        if n_docs > max(k_eff * 4, 64):
            kth_val = float(np.partition(row, n_docs - k_eff)[n_docs - k_eff])
            cand_idx = np.flatnonzero(row >= kth_val)
            ordered_idx = sorted(cand_idx.tolist(), key=lambda j: (-float(row[j]), canonical_docs[j]))[:k_eff]
        else:
            ordered_idx = sorted(range(n_docs), key=lambda j: (-float(row[j]), canonical_docs[j]))[:k_eff]
        cqid = canonicalize_id(qid)
        rankings[cqid] = [canonical_docs[j] for j in ordered_idx]
        top_scores[cqid] = row[ordered_idx]

    return rankings, top_scores


def evaluate_rankings_mlr(
    rankings: Mapping[str, Sequence[str]],
    qrels: Mapping[str, Mapping[str, float]],
    query_ids: Sequence[str],
    *,
    k: int = 10,
    binary: bool = False,
) -> Dict[str, Dict[str, float]]:
    """Compute per-query complete-qrels metrics under mlr-v1."""
    per_query: Dict[str, Dict[str, float]] = {}
    for raw_qid in query_ids:
        qid = canonicalize_id(raw_qid)
        rel = qrels.get(qid, qrels.get(raw_qid, {}))
        pos = {canonicalize_id(did): float(score) for did, score in rel.items() if float(score) > 0.0}
        if not pos:
            continue
        ranked = rankings.get(qid, [])
        per_query[qid] = {
            "ndcg_at_10": float(ndcg_at_k(ranked, pos, k=k, binary=binary)),
            "map_at_10": float(mean_average_precision_at_k(ranked, pos.keys(), k=k)),
            "mrr": float(mean_reciprocal_rank(ranked, pos.keys())),
            "recall_at_10": float(recall_at_k(ranked, pos.keys(), k=k)),
        }
    return per_query


def compute_fastfill_backfill_order(
    dev_queries_old_space: np.ndarray,
    docs_old_space: np.ndarray,
    *,
    top_m: int = 50,
) -> np.ndarray:
    """Order corpus documents by DEV query-exposure priority (FastFill-style proxy).

    Uses ONLY DEVELOPMENT queries and old-space doc vectors (zero REPORT or qrels access).
    """
    q_norm = _normalize_rows(dev_queries_old_space)
    d_norm = _normalize_rows(docs_old_space)
    sims = q_norm @ d_norm.T
    n_docs = d_norm.shape[0]
    m_eff = max(1, min(int(top_m), n_docs))
    exposure = np.zeros(n_docs, dtype=np.float64)
    for row in sims:
        top_idx = np.argpartition(-row, m_eff - 1)[:m_eff]
        # Weight by positive similarity rank mass
        exposure[top_idx] += np.maximum(row[top_idx], 0.0) + 1e-6
    return np.asarray(
        sorted(range(n_docs), key=lambda idx: (-float(exposure[idx]), idx)),
        dtype=np.int64,
    )


def score_map_plus_backfill(
    queries_new: np.ndarray,
    docs_old: np.ndarray,
    docs_new: np.ndarray,
    bridge_new_to_old: Optional[np.ndarray],
    backfill_indices: np.ndarray,
    query_ids: Sequence[str],
    doc_ids: Sequence[str],
    *,
    dev_queries_new: np.ndarray,
    k: int = 10,
    calibrate: bool = True,
) -> Dict[str, List[str]]:
    """Score mixed-lineage index: native new-space for backfilled docs, bridged old-space for rest.

    When bridge_new_to_old is None, evaluates backfill-only (un-backfilled docs get -inf).
    Score calibration (mean/std alignment between bridged and native score distributions)
    is fit strictly on dev_queries_new.
    """
    n_docs = docs_old.shape[0]
    backfill_mask = np.zeros(n_docs, dtype=bool)
    if len(backfill_indices) > 0:
        backfill_mask[backfill_indices] = True

    q_new_norm = _normalize_rows(queries_new)
    d_new_norm = _normalize_rows(docs_new)
    native_scores = q_new_norm @ d_new_norm.T  # [n_q, n_docs]

    if bridge_new_to_old is None:
        combined = np.full_like(native_scores, -1e9)
        combined[:, backfill_mask] = native_scores[:, backfill_mask]
    else:
        q_bridged = _normalize_rows(np.asarray(queries_new, dtype=np.float64) @ bridge_new_to_old)
        d_old_norm = _normalize_rows(docs_old)
        bridged_scores = q_bridged @ d_old_norm.T

        if calibrate and backfill_mask.any() and (~backfill_mask).any():
            dev_q_norm = _normalize_rows(dev_queries_new)
            dev_q_bridged = _normalize_rows(np.asarray(dev_queries_new, dtype=np.float64) @ bridge_new_to_old)
            dev_nat = dev_q_norm @ d_new_norm[backfill_mask].T
            dev_brg = dev_q_bridged @ d_old_norm[~backfill_mask].T
            mu_n, sd_n = float(dev_nat.mean()), max(float(dev_nat.std()), 1e-6)
            mu_b, sd_b = float(dev_brg.mean()), max(float(dev_brg.std()), 1e-6)
            bridged_scores = ((bridged_scores - mu_b) / sd_b) * sd_n + mu_n

        combined = np.where(backfill_mask[None, :], native_scores, bridged_scores)

    canonical_docs = [canonicalize_id(d) for d in doc_ids]
    rankings: Dict[str, List[str]] = {}
    k_eff = min(k, n_docs)
    for row_idx, qid in enumerate(query_ids):
        row = combined[row_idx]
        if n_docs > max(k_eff * 4, 64):
            kth_val = float(np.partition(row, n_docs - k_eff)[n_docs - k_eff])
            cand_idx = np.flatnonzero(row >= kth_val)
            ordered_idx = sorted(cand_idx.tolist(), key=lambda j: (-float(row[j]), canonical_docs[j]))[:k_eff]
        else:
            ordered_idx = sorted(range(n_docs), key=lambda j: (-float(row[j]), canonical_docs[j]))[:k_eff]
        rankings[canonicalize_id(qid)] = [canonical_docs[j] for j in ordered_idx]
    return rankings


def required_m1_rungs(
    *,
    old_encoder_retired: bool,
    include_text_hybrid: bool = False,
    budget: float = 0.0,
) -> List[str]:
    """Return the exact required M1 baseline rungs for a given budget and retirement premise."""
    rungs = [
        "identity_frozen_c0",
        "floor_csls_k10",
        "closed_form_svd_procrustes",
        "closed_form_ridge",
        "full_reembed_c2o",
    ]
    if not old_encoder_retired:
        rungs.insert(1, "old_encoder_servable")
    if include_text_hybrid:
        rungs.append("floor_bm25_hybrid")
    if budget > 0.0:
        b_tag = f"{int(round(budget * 100)):02d}"
        rungs.extend([f"budgeted_backfill_{b_tag}", f"map_plus_backfill_{b_tag}"])
    return rungs


def emit_candidate_row(
    *,
    candidate_name: str,
    candidate_row: Mapping[str, Any],
    ladder_rows: Mapping[str, Mapping[str, Any]],
    required_rungs: Sequence[str],
    split_sha256: str,
    metric_contract_sha256: str,
) -> Dict[str, Any]:
    """Fail closed if any required M1 rung is missing or bound to a different split/metric."""
    missing = [rung for rung in required_rungs if rung not in ladder_rows]
    if missing:
        raise RuntimeError(
            f"M1 fail-closed: cannot emit candidate {candidate_name!r}; "
            f"missing required ladder rungs: {', '.join(missing)}"
        )
    for rung in required_rungs:
        row = ladder_rows[rung]
        if row.get("split_sha256") != split_sha256:
            raise RuntimeError(f"M1 fail-closed: rung {rung!r} split_sha256 mismatch")
        if row.get("metric_contract_sha256") != metric_contract_sha256:
            raise RuntimeError(f"M1 fail-closed: rung {rung!r} metric_contract_sha256 mismatch")
        status = row.get("status", "OK")
        if status == "OK":
            val = row.get("mean_ndcg_at_10")
            if val is None or not math.isfinite(float(val)):
                raise RuntimeError(f"M1 fail-closed: rung {rung!r} has non-finite mean_ndcg_at_10")
    out = dict(candidate_row)
    out["candidate_name"] = str(candidate_name)
    out["verified_m1_rungs"] = list(required_rungs)
    out["split_sha256"] = split_sha256
    out["metric_contract_sha256"] = metric_contract_sha256
    return out


def m2_gap_preflight(
    dev_per_query_scores: Mapping[str, Mapping[str, float]],
    *,
    split: str,
    reference_arm: str,
    eligible_arms: Sequence[str],
    delta: float,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Fixed M2 gap preflight on DEVELOPMENT units only.

    Mechanically prevents D-02 by forbidding reference_arm inside eligible_arms,
    and forbids SELECT / REPORT inputs.
    """
    if split != "DEVELOPMENT":
        raise ValueError(f"M2 gap preflight requires split='DEVELOPMENT', got {split!r}")
    if not eligible_arms:
        raise ValueError("eligible_arms must be non-empty")
    if reference_arm in set(eligible_arms):
        raise ValueError(
            f"M2 fail-closed (D-02 guard): reference_arm {reference_arm!r} must not appear in eligible_arms"
        )
    if reference_arm not in dev_per_query_scores:
        raise ValueError(f"reference_arm {reference_arm!r} missing from dev_per_query_scores")
    missing_arms = [arm for arm in eligible_arms if arm not in dev_per_query_scores]
    if missing_arms:
        raise ValueError(f"eligible_arms missing from dev_per_query_scores: {missing_arms}")
    if delta < 0.0 or not math.isfinite(delta):
        raise ValueError("delta must be finite and >= 0")
    if n_bootstrap < 100:
        raise ValueError("n_bootstrap must be >= 100")

    ref_map = dev_per_query_scores[reference_arm]
    query_ids = sorted(ref_map.keys())
    if not query_ids:
        raise ValueError("dev_per_query_scores contains zero queries")
    for arm in eligible_arms:
        if set(dev_per_query_scores[arm].keys()) != set(query_ids):
            raise ValueError(f"query_id set mismatch between reference_arm and {arm!r}")

    ref_vec = np.asarray([float(ref_map[qid]) for qid in query_ids], dtype=np.float64)
    elig_matrix = np.asarray(
        [[float(dev_per_query_scores[arm][qid]) for qid in query_ids] for arm in eligible_arms],
        dtype=np.float64,
    )  # [n_arms, n_queries]
    if not (np.all(np.isfinite(ref_vec)) and np.all(np.isfinite(elig_matrix))):
        raise ValueError("dev_per_query_scores must contain only finite floats")

    arm_means = elig_matrix.mean(axis=1)
    best_idx = int(np.argmax(arm_means))
    best_arm = str(eligible_arms[best_idx])
    best_mean = float(arm_means[best_idx])
    ref_mean = float(ref_vec.mean())
    point_gap = float(ref_mean - best_mean)

    rng = np.random.default_rng(seed)
    n_q = len(query_ids)
    # Chunked bootstrap to bound memory cleanly
    fixed_gaps = np.empty(n_bootstrap, dtype=np.float64)
    sel_adj_gaps = np.empty(n_bootstrap, dtype=np.float64)
    chunk = 1000
    for start in range(0, n_bootstrap, chunk):
        end = min(n_bootstrap, start + chunk)
        b_size = end - start
        idx = rng.integers(0, n_q, size=(b_size, n_q))
        ref_boot = ref_vec[idx].mean(axis=1)
        best_fixed_boot = elig_matrix[best_idx][idx].mean(axis=1)
        fixed_gaps[start:end] = ref_boot - best_fixed_boot
        # Selection-adjusted max over eligible arms inside each bootstrap sample
        max_boot = np.full(b_size, -np.inf, dtype=np.float64)
        for a_i in range(len(eligible_arms)):
            a_boot = elig_matrix[a_i][idx].mean(axis=1)
            max_boot = np.maximum(max_boot, a_boot)
        sel_adj_gaps[start:end] = ref_boot - max_boot

    lcb_95_fixed = float(np.quantile(fixed_gaps, 0.025))
    ucb_95_fixed = float(np.quantile(fixed_gaps, 0.975))
    lcb_95_sel_adj = float(np.quantile(sel_adj_gaps, 0.025))
    ucb_95_sel_adj = float(np.quantile(sel_adj_gaps, 0.975))

    # Admit only if point gap >= delta AND both fixed and selection-adjusted 95% lower bounds > 0
    admit = bool(point_gap >= delta and lcb_95_fixed > 0.0 and lcb_95_sel_adj > 0.0)
    return {
        "split": split,
        "decision": "ADMIT" if admit else "REFUSE",
        "reference_arm": reference_arm,
        "reference_mean": ref_mean,
        "best_eligible_arm": best_arm,
        "best_eligible_mean": best_mean,
        "eligible_arms": list(eligible_arms),
        "n_queries": n_q,
        "delta": float(delta),
        "point_gap": point_gap,
        "lcb_95_fixed": lcb_95_fixed,
        "ucb_95_fixed": ucb_95_fixed,
        "lcb_95_selection_adjusted": lcb_95_sel_adj,
        "ucb_95_selection_adjusted": ucb_95_sel_adj,
        "n_bootstrap": int(n_bootstrap),
        "seed": int(seed),
    }


def metric_contract_digest(qrels_sha256: str, *, k: int = 10, gain_rule: str = "binary") -> Tuple[Dict[str, Any], str]:
    """Return (contract_dict, sha256_hex) for M1/M2 artifact binding."""
    import json

    manifest = build_metric_contract_manifest(
        metric_name="ndcg_at_k",
        k=k,
        gain_rule=gain_rule,
        qrels_sha256=qrels_sha256,
    )
    canonical = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return manifest, hashlib.sha256(canonical).hexdigest()
