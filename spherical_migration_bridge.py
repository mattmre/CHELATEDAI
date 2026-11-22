"""Production-packaged closed-form spherical migration stack for Flagship #1.

Consolidates the empirically winning zero-training, qrels-free components validated
under the complete-qrels `mlr-v1` contract across Night-1 dual-Spark campaigns:

  1. Manifold-precipitated anchor selection (`select_spherical_kmeanspp_anchors`,
     `select_kcenter_anchors`) for low-anchor and multi-hop spectral stability.
  2. 1st-order Centered Orthogonal Procrustes (`CenteredProcrustesBridge`, `A2`)
     preserving exact orthogonality (`cond = 1.00`) across arbitrary multi-hop chains
     while eliminating narrow-cone mean-shift collapse (`E1` BGE, `E2` E5).
  3. 1st+2nd-order Shrinkage-Whitened (CORAL) Spherical Procrustes
     (`WhitenedSphericalProcrustesBridge`, `A3`) for single-hop zero-write alignment.
  4. Quantile/CDF-calibrated score fusion (`quantile_calibrated_backfill_scores`, `B1`)
     and spherical neighborhood distortion prioritization (`B2`/`B3`).
  5. Typed lineage ABI integration (`build_bridge_contract`, `resolve_with_tra1`)
     enforcing fail-closed D-08 production floors via `SemanticCacheResolver`.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from m1_baseline_ladder import score_dense_rankings
from representation_space import (
    BridgeContract,
    BridgeCost,
    BridgeValidationState,
    RepresentationDescriptor,
    RepresentationRole,
    ResidentIndexDescriptor,
)
from semantic_cache_resolver import ResolutionDecision, SemanticCacheResolver


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization in float64 with safe zero-norm guard."""
    arr = np.asarray(x, dtype=np.float64)
    if arr.ndim == 1:
        norm = float(np.linalg.norm(arr))
        return arr / norm if norm > 1e-12 else arr.copy()
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(norms > 1e-12, norms, 1.0)


def _sha256_arrays(*arrays: np.ndarray) -> str:
    h = hashlib.sha256()
    for arr in arrays:
        a = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
        h.update(str(a.shape).encode("ascii"))
        h.update(a.tobytes())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# 1. Manifold-Precipitated Anchor Selection (P6 k-Means++, P4 k-Center)
# ---------------------------------------------------------------------------
def select_spherical_kmeanspp_anchors(
    vecs: np.ndarray,
    n_anchors: int,
    *,
    seed: int = 42,
    candidate_pool_cap: int = 8192,
) -> np.ndarray:
    """Deterministic spherical k-means++ (D^2 angular seeding) anchor selector (`P6`).

    Covers multimodal semantic lobes without over-selecting isolated corpus noise.
    """
    vn = l2_normalize(vecs)
    n = vn.shape[0]
    k = min(int(n_anchors), n)
    if k <= 0:
        raise ValueError("n_anchors must be positive")
    rng = np.random.default_rng(seed + 303)
    eff_cap = max(int(candidate_pool_cap), k)
    if n > eff_cap:
        pool = np.sort(rng.choice(n, size=eff_cap, replace=False))
    else:
        pool = np.arange(n, dtype=np.int64)
    vp = vn[pool]
    first = int(rng.integers(0, vp.shape[0]))
    chosen = [first]
    min_d2 = np.maximum(0.0, 2.0 - 2.0 * (vp @ vp[first]))
    min_d2[first] = 0.0
    for _ in range(1, k):
        total = float(min_d2.sum())
        if total <= 1e-12:
            rem = np.setdiff1d(np.arange(vp.shape[0]), np.asarray(chosen))
            nxt = int(rng.choice(rem))
        else:
            nxt = int(rng.choice(vp.shape[0], p=min_d2 / total))
        chosen.append(nxt)
        d2 = np.maximum(0.0, 2.0 - 2.0 * (vp @ vp[nxt]))
        min_d2 = np.minimum(min_d2, d2)
        min_d2[chosen] = 0.0
    return np.sort(pool[np.asarray(chosen, dtype=np.int64)])


def select_kcenter_anchors(
    vecs: np.ndarray,
    n_anchors: int,
    *,
    seed: int = 42,
    outlier_trim_quantile: float = 0.02,
    candidate_pool_cap: int = 8192,
) -> np.ndarray:
    """Deterministic outlier-trimmed farthest-point (k-center) selector (`P4`)."""
    vn = l2_normalize(vecs)
    n = vn.shape[0]
    k = min(int(n_anchors), n)
    if k <= 0:
        raise ValueError("n_anchors must be positive")
    centroid = l2_normalize(vn.mean(axis=0, keepdims=True))[0]
    cos_c = vn @ centroid
    cut = float(np.quantile(cos_c, outlier_trim_quantile))
    eligible = np.where(cos_c >= cut)[0]
    if eligible.size < k:
        eligible = np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(seed + 101)
    eff_cap = max(int(candidate_pool_cap), k)
    if eligible.size > eff_cap:
        pool = np.sort(rng.choice(eligible, size=eff_cap, replace=False))
    else:
        pool = eligible
    vp = vn[pool]
    first = int(rng.integers(0, vp.shape[0]))
    chosen = [first]
    max_sim = vp @ vp[first]
    max_sim[first] = 2.0
    for _ in range(1, k):
        nxt = int(np.argmin(max_sim))
        chosen.append(nxt)
        max_sim = np.maximum(max_sim, vp @ vp[nxt])
        max_sim[nxt] = 2.0
    return np.sort(pool[np.asarray(chosen, dtype=np.int64)])


# ---------------------------------------------------------------------------
# 2. Centered Orthogonal Procrustes (A2) — Multi-Hop Isometric Composable
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class CenteredProcrustesBridge:
    """1st-order centered + conformal orthogonal spherical map (`A2` / `A2c`).

    Applies `y = scale * ((x - mu_x) @ Q) + mu_y` before optional L2 projection.
    When `match_radius=True` (default), `scale = ||Y_c||_F / ||X_c||_F` matches
    source and target spherical-cap radii (crucial when bridging narrow-cone
    encoders like E5/BGE to isotropic encoders like MiniLM/MPNet) while keeping
    all nonzero singular values of `scale * Q` identical (`cond = 1.00`) across
    arbitrary multi-hop chains.
    """

    mu_x: np.ndarray  # [1, d_in]
    mu_y: np.ndarray  # [1, d_out]
    Q: np.ndarray     # [d_in, d_out]
    fit_count: int
    scale: float = 1.0

    @classmethod
    def fit(
        cls,
        x_src: np.ndarray,
        y_tgt: np.ndarray,
        *,
        fit_idx: Optional[Sequence[int]] = None,
        match_radius: bool = True,
    ) -> "CenteredProcrustesBridge":
        xs = np.asarray(x_src if fit_idx is None else x_src[np.asarray(fit_idx)], dtype=np.float64)
        yt = np.asarray(y_tgt if fit_idx is None else y_tgt[np.asarray(fit_idx)], dtype=np.float64)
        if xs.shape[0] != yt.shape[0] or xs.shape[0] < 2:
            raise ValueError("fit requires matched anchor arrays with at least 2 rows")
        mu_x = xs.mean(axis=0, keepdims=True)
        mu_y = yt.mean(axis=0, keepdims=True)
        xc = xs - mu_x
        yc = yt - mu_y
        u, _, vt = np.linalg.svd(xc.T @ yc, full_matrices=False)
        q = u @ vt
        if match_radius:
            rad_x = float(np.linalg.norm(xc))
            rad_y = float(np.linalg.norm(yc))
            scale = rad_y / max(rad_x, 1e-12)
        else:
            scale = 1.0
        return cls(mu_x=mu_x, mu_y=mu_y, Q=q, fit_count=int(xs.shape[0]), scale=float(scale))

    def transform(self, x: np.ndarray, *, normalize: bool = True) -> np.ndarray:
        out = self.scale * ((np.asarray(x, dtype=np.float64) - self.mu_x) @ self.Q) + self.mu_y
        return l2_normalize(out) if normalize else out

    @property
    def condition_number(self) -> float:
        if abs(self.scale) <= 1e-12:
            return float("inf")
        sv = np.linalg.svd(self.Q * self.scale, compute_uv=False)
        if sv.size == 0 or sv[-1] <= 1e-12:
            return float("inf")
        return float(sv[0] / sv[-1])

    @property
    def weights_sha256(self) -> str:
        return _sha256_arrays(self.mu_x, self.mu_y, self.Q, np.asarray([self.scale], dtype=np.float64))


def compose_centered_bridges(
    bridges: Sequence[CenteredProcrustesBridge],
) -> CenteredProcrustesBridge:
    """Compose a chain `[B_{m->m-1}, ..., B_{1->0}]` into one exact conformal-orthogonal map.

    Each hop applies `y = s * ((x - mu_x) @ Q) + mu_y` in ambient coordinates
    (final L2 normalization applied once at query time), preserving exact
    `cond(s_comp * Q_comp) == 1.0` for equal-dimension chains while matching
    cone radii across narrow-cone and isotropic encoders.
    """
    if not bridges:
        raise ValueError("bridges sequence must be non-empty")
    cur = bridges[0]
    for nxt in bridges[1:]:
        # s2 * ((s1 * (x - mu_x1) @ Q1 + mu_y1 - mu_x2) @ Q2) + mu_y2
        # = (s1 * s2) * ((x - mu_x1) @ (Q1 @ Q2)) + (s2 * (mu_y1 - mu_x2) @ Q2 + mu_y2)
        q_comp = cur.Q @ nxt.Q
        s_comp = float(cur.scale * nxt.scale)
        mu_y_comp = nxt.scale * ((cur.mu_y - nxt.mu_x) @ nxt.Q) + nxt.mu_y
        cur = CenteredProcrustesBridge(
            mu_x=cur.mu_x,
            mu_y=mu_y_comp,
            Q=q_comp,
            fit_count=min(cur.fit_count, nxt.fit_count),
            scale=s_comp,
        )
    return cur


# ---------------------------------------------------------------------------
# 3. Centered Shrinkage-Whitened (CORAL) Spherical Procrustes (A3)
# ---------------------------------------------------------------------------
def _shrinkage_cov_factors(xc: np.ndarray, shrink: float) -> Tuple[np.ndarray, np.ndarray]:
    if not (0.0 <= shrink <= 1.0):
        raise ValueError("shrink must be in [0, 1]")
    n, d = xc.shape
    cov = (xc.T @ xc) / float(max(n - 1, 1))
    tr_scale = float(np.trace(cov)) / float(d)
    cov_reg = (1.0 - shrink) * cov + shrink * tr_scale * np.eye(d, dtype=np.float64)
    eigvals, eigvecs = np.linalg.eigh(cov_reg)
    eigvals = np.maximum(eigvals, 1e-8)
    inv_sqrt = (eigvecs * (eigvals ** -0.5)) @ eigvecs.T
    sqrt_mat = (eigvecs * (eigvals ** 0.5)) @ eigvecs.T
    return inv_sqrt, sqrt_mat


@dataclass(frozen=True)
class WhitenedSphericalProcrustesBridge:
    """1st-order centered + 2nd-order shrinkage-whitened CORAL Procrustes (`A3`)."""

    mu_x: np.ndarray  # [1, d_in]
    mu_y: np.ndarray  # [1, d_out]
    W: np.ndarray     # [d_in, d_out]
    shrink: float
    fit_count: int

    @classmethod
    def fit(
        cls,
        x_src: np.ndarray,
        y_tgt: np.ndarray,
        *,
        fit_idx: Optional[Sequence[int]] = None,
        shrink: float = 0.20,
    ) -> "WhitenedSphericalProcrustesBridge":
        xs = np.asarray(x_src if fit_idx is None else x_src[np.asarray(fit_idx)], dtype=np.float64)
        yt = np.asarray(y_tgt if fit_idx is None else y_tgt[np.asarray(fit_idx)], dtype=np.float64)
        if xs.shape[0] != yt.shape[0] or xs.shape[0] < 2:
            raise ValueError("fit requires matched anchor arrays with at least 2 rows")
        mu_x = xs.mean(axis=0, keepdims=True)
        mu_y = yt.mean(axis=0, keepdims=True)
        xc = xs - mu_x
        yc = yt - mu_y
        wx_inv, _ = _shrinkage_cov_factors(xc, shrink=shrink)
        wy_inv, wy_sqrt = _shrinkage_cov_factors(yc, shrink=shrink)
        u, _, vt = np.linalg.svd((xc @ wx_inv).T @ (yc @ wy_inv), full_matrices=False)
        q = u @ vt
        w_total = wx_inv @ q @ wy_sqrt
        return cls(mu_x=mu_x, mu_y=mu_y, W=w_total, shrink=float(shrink), fit_count=int(xs.shape[0]))

    def transform(self, x: np.ndarray, *, normalize: bool = True) -> np.ndarray:
        out = (np.asarray(x, dtype=np.float64) - self.mu_x) @ self.W + self.mu_y
        return l2_normalize(out) if normalize else out

    @property
    def condition_number(self) -> float:
        sv = np.linalg.svd(self.W, compute_uv=False)
        return float(sv[0] / max(sv[-1], 1e-12))

    @property
    def weights_sha256(self) -> str:
        return _sha256_arrays(self.mu_x, self.mu_y, self.W)


# ---------------------------------------------------------------------------
# 4. Quantile-CDF Backfill Calibration & Spherical Distortion Prioritization
# ---------------------------------------------------------------------------
def compute_spherical_distortion_scores(
    docs_new: np.ndarray,
    docs_old: np.ndarray,
    mapped_docs_old: np.ndarray,
    *,
    anchor_pool_idx: Sequence[int],
    tau: float = 0.07,
    top_k_neighbors: int = 64,
    batch_size: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    """Rank corpus docs by qrels-free spherical angular discrepancy + top-k neighbor KL (`B2`)."""
    pool = np.asarray(anchor_pool_idx, dtype=np.int64)[: min(len(anchor_pool_idx), 1024)]
    p_new = l2_normalize(docs_new[pool])
    p_old = l2_normalize(docs_old[pool])
    n_docs = docs_new.shape[0]
    k_eff = min(top_k_neighbors, p_new.shape[0])
    distortion = np.zeros(n_docs, dtype=np.float64)

    for start in range(0, n_docs, batch_size):
        end = min(n_docs, start + batch_size)
        dn = l2_normalize(docs_new[start:end])
        do = l2_normalize(docs_old[start:end])
        dm = l2_normalize(mapped_docs_old[start:end])
        ang_err = 1.0 - np.sum(dm * do, axis=1)

        sim_t = (dn @ p_new.T) / tau
        sim_s = (dm @ p_old.T) / tau
        if p_new.shape[0] > k_eff:
            top_idx = np.argpartition(-sim_t, k_eff - 1, axis=1)[:, :k_eff]
            rows_ar = np.arange(dn.shape[0])[:, None]
            top_t = sim_t[rows_ar, top_idx]
            top_s = sim_s[rows_ar, top_idx]
        else:
            top_t, top_s = sim_t, sim_s

        # Stable row-wise log-softmax & KL(P_t || P_s)
        t_shift = top_t - top_t.max(axis=1, keepdims=True)
        log_pt = t_shift - np.log(np.sum(np.exp(t_shift), axis=1, keepdims=True))
        pt = np.exp(log_pt)
        s_shift = top_s - top_s.max(axis=1, keepdims=True)
        log_ps = s_shift - np.log(np.sum(np.exp(s_shift), axis=1, keepdims=True))
        kl = np.sum(pt * (log_pt - log_ps), axis=1)
        distortion[start:end] = ang_err + 0.25 * kl

    order = np.argsort(-distortion, kind="mergesort")
    return order, distortion


def combine_exposure_and_distortion_order(
    fastfill_order: Sequence[int],
    distortion_order: Sequence[int],
    *,
    exposure_weight: float = 0.5,
) -> np.ndarray:
    """Blend exposure rank and spherical distortion rank into one backfill order (`B3`)."""
    n_docs = len(fastfill_order)
    if len(distortion_order) != n_docs:
        raise ValueError("fastfill_order and distortion_order must have equal length")
    exp_rank = np.empty(n_docs, dtype=np.float64)
    exp_rank[np.asarray(fastfill_order, dtype=np.int64)] = np.linspace(1.0, 0.0, num=n_docs)
    dist_rank = np.empty(n_docs, dtype=np.float64)
    dist_rank[np.asarray(distortion_order, dtype=np.int64)] = np.linspace(1.0, 0.0, num=n_docs)
    score = float(exposure_weight) * exp_rank + (1.0 - float(exposure_weight)) * dist_rank
    return np.argsort(-score, kind="mergesort")


def score_quantile_calibrated_backfill(
    queries_new: np.ndarray,
    queries_mapped: np.ndarray,
    docs_old: np.ndarray,
    docs_new: np.ndarray,
    backfill_indices: Sequence[int],
    query_ids: Sequence[str],
    doc_ids: Sequence[str],
    *,
    dev_queries_new: np.ndarray,
    dev_queries_mapped: np.ndarray,
    k: int = 10,
    csls_k: int = 0,
) -> Dict[str, List[str]]:
    """Monotone empirical-CDF score fusion between native backfilled docs and bridged docs (`B1`)."""
    bf_idx = np.asarray(sorted(set(int(i) for i in backfill_indices)), dtype=np.int64)
    n_docs = docs_old.shape[0]
    d_old_n = l2_normalize(docs_old)
    d_new_n = l2_normalize(docs_new)
    q_new_n = l2_normalize(queries_new)
    q_map_n = l2_normalize(queries_mapped)

    if bf_idx.size == 0:
        rankings, _ = score_dense_rankings(
            q_map_n,
            d_old_n,
            query_ids,
            doc_ids,
            k=k,
            csls_k=csls_k,
            csls_reference_queries=dev_queries_mapped if csls_k > 0 else None,
        )
        return rankings

    dev_qn = l2_normalize(dev_queries_new)
    dev_qm = l2_normalize(dev_queries_mapped)
    sub_docs = np.linspace(0, n_docs - 1, num=min(n_docs, 4096), dtype=np.int64)
    dev_native_sorted = np.sort((dev_qn @ d_new_n[sub_docs].T).ravel())
    dev_mapped_sorted = np.sort((dev_qm @ d_old_n[sub_docs].T).ravel())

    mapped_scores = q_map_n @ d_old_n.T
    native_bf_scores = q_new_n @ d_new_n[bf_idx].T

    ranks = np.searchsorted(dev_native_sorted, native_bf_scores, side="left")
    quantiles = np.clip(ranks / float(max(dev_native_sorted.size - 1, 1)), 0.0, 1.0)
    mapped_pos = np.clip(
        (quantiles * (dev_mapped_sorted.size - 1)).astype(np.int64),
        0,
        dev_mapped_sorted.size - 1,
    )
    cal_bf_scores = dev_mapped_sorted[mapped_pos]

    combined = mapped_scores.copy()
    combined[:, bf_idx] = cal_bf_scores

    k_eff = min(k, n_docs)
    rankings: Dict[str, List[str]] = {}
    for row_idx, qid in enumerate(query_ids):
        row = combined[row_idx]
        if n_docs > max(k_eff * 4, 64):
            kth_val = float(np.partition(row, n_docs - k_eff)[n_docs - k_eff])
            cand = np.flatnonzero(row >= kth_val)
            ordered = sorted(cand.tolist(), key=lambda j: (-float(row[j]), doc_ids[j]))[:k_eff]
        else:
            ordered = sorted(range(n_docs), key=lambda j: (-float(row[j]), doc_ids[j]))[:k_eff]
        rankings[qid] = [doc_ids[j] for j in ordered]
    return rankings


# ---------------------------------------------------------------------------
# 5. Hardened TRA-1 Bridge Contract & Fail-Closed Resolution Helper
# ---------------------------------------------------------------------------
def build_bridge_contract(
    *,
    source: RepresentationDescriptor,
    target: RepresentationDescriptor,
    bridge: CenteredProcrustesBridge | WhitenedSphericalProcrustesBridge,
    transform_id: str,
    anchor_manifest_sha256: str,
    validation_count: int,
    validation_confidence: float,
    evidence_id: str,
    evidence_sha256: str,
    role: RepresentationRole = RepresentationRole.QUERY,
) -> BridgeContract:
    """Wrap a fitted spherical bridge in a content-addressed TRA-1 `BridgeContract`."""
    family = (
        "centered_orthogonal_procrustes"
        if isinstance(bridge, CenteredProcrustesBridge)
        else "whitened_spherical_procrustes"
    )
    hp = (
        {"family": family}
        if isinstance(bridge, CenteredProcrustesBridge)
        else {"family": family, "shrink": bridge.shrink}
    )
    hp_sha = hashlib.sha256(json.dumps(hp, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    return BridgeContract(
        source=source,
        target=target,
        role=role,
        transform_family=family,
        transform_id=transform_id,
        weights_sha256=bridge.weights_sha256,
        hyperparameters_sha256=hp_sha,
        fit_anchor_manifest_sha256=anchor_manifest_sha256,
        fit_count=bridge.fit_count,
        validation_count=int(validation_count),
        validation_state=BridgeValidationState.VALIDATED,
        cost=BridgeCost.ZERO_WRITE if role is RepresentationRole.QUERY else BridgeCost.MATERIALIZED_INDEX,
        evidence_id=evidence_id,
        evidence_sha256=evidence_sha256,
        validation_confidence=float(validation_confidence),
    )


def resolve_with_tra1(
    *,
    query_space: RepresentationDescriptor,
    resident_index: ResidentIndexDescriptor,
    bridges: Iterable[BridgeContract],
    trusted_evidence_sha256s: Iterable[str],
    trusted_bridge_contract_ids: Iterable[str],
    minimum_route_confidence: float = 0.70,
    min_fit_count: int = 32,
    min_validation_count: int = 16,
) -> ResolutionDecision:
    """Resolve search authorization through hardened TRA-1 with D-08 floor enabled."""
    resolver = SemanticCacheResolver(
        bridges=bridges,
        minimum_route_confidence=minimum_route_confidence,
        trusted_evidence_sha256s=trusted_evidence_sha256s,
        trusted_bridge_contract_ids=trusted_bridge_contract_ids,
        enforce_production_floor=True,
        min_fit_count=min_fit_count,
        min_validation_count=min_validation_count,
    )
    return resolver.resolve(query_space=query_space, resident_index=resident_index)


__all__ = [
    "CenteredProcrustesBridge",
    "WhitenedSphericalProcrustesBridge",
    "build_bridge_contract",
    "combine_exposure_and_distortion_order",
    "compose_centered_bridges",
    "compute_spherical_distortion_scores",
    "l2_normalize",
    "resolve_with_tra1",
    "score_quantile_calibrated_backfill",
    "select_kcenter_anchors",
    "select_spherical_kmeanspp_anchors",
]
