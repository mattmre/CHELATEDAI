"""Production-packaged closed-form spherical migration stack for Flagship #1 (R3 Council Hardened).

Preserves every original pristine algorithmic baseline side-by-side (`A2`, ambient `A2c`,
`A3_full`, `BF20`, `P6`, `P4`) while adding all R3-ratified geometric & emergent upgrades:

  1. Manifold-precipitated anchor selection (`select_spherical_kmeanspp_anchors`,
     `select_kcenter_anchors`) + Voronoi mass weighting (`compute_voronoi_anchor_weights`, `A2c-IW`).
  2. 1st-order Centered Orthogonal Procrustes (`CenteredProcrustesBridge`):
       - Preserves legacy `A2` (`match_radius=False`) and pristine ambient `A2c` (`scale_mode="ambient"`).
       - Adds `A2c-Subspace` (rank-truncated thin SVD, zero-damping unidentified nullspace modes),
         all 4 disjoint-CAL OOS scale laws (`s_full_oos`, `s_sub_oos`, `max_one_s_full_oos`, `unit`),
         query-centroid shift `mu_q_override` (`A2c-Q`), and `EM-5b` `A2c-SoftSpec`
         (`U diag(sigma / (sigma + tau)) V^T`).
       - Exact Three-Regime Stiefel Composition (`compose_centered_bridges`: `monotone | valley | peak`)
         with optional Stiefel Polar Re-Projection (`polar_reproject=True`, `A2c-Polar`) and
         pre-polar `sigma_min` telemetry.
  3. Riemannian Geodesic Polar Tangent Alignment (`GeodesicPolarBridge`, `A2g`):
       - Dual pole stability guard (`min_resultant >= 0.15` AND hash-ordered alternating split-half
         pole drift `<= 15 deg`, with `0.25` sensitivity toggle and automatic `A2c-Subspace+OOS`
         fallback receipt).
       - 2-step Gram-Schmidt + `atan2` Log map, O(d) Householder tangent frames, disjoint-CAL scale,
         and Carbon's C^1 exact soft-collar (`r_0 = pi - 0.10, delta = 0.05`; hard-clamp toggle).
  4. `EM-2` Subtractive Centroid Removal (`score_zeromu_rankings`):
       - `query_only` (true zero-write on unmodified cosine ANN indexes),
       - `dual_fit` (zero-reindex 1-float `||d - mu_y||_2` exact identity),
       - `dual_adap` (unlabeled resident-index + eval-query means, zero paired target anchors).
  5. 1st+2nd-order Shrinkage-Whitened (CORAL) Spherical Procrustes (`WhitenedSphericalProcrustesBridge`):
       - Exact O(n^2 d) dual thin-SVD whitening (machine-precision parity with dense `eigh`),
       - Explicit 2x2 crossed `subspace_rank` x `trace_divisor_mode` (`ambient | active_rank | effective_rank`).
  6. Scoped Backfill (`B4` vs legacy `BF20`):
       - Fixes `csls_k` passthrough in nonempty branch + strictly increasing right-tail linear
         continuation above `q_0.995` (preserving `legacy_hard_clip=True` for pristine `BF20`),
       - Adds zero-target-encode old-space anchor-residual kNN distortion (`compute_old_space_knn_distortion`).
  7. Hardened TRA-1 Bridge Contract (`build_bridge_contract`) binding full hyperparameters,
     spectral diagnostics, scoring mode, and hop provenance digests.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

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


def _canonical_topk(row: np.ndarray, doc_ids: Sequence[str], k_eff: int) -> List[int]:
    """Exact global canonical tie-breaking `(-score, doc_id)` safe under arbitrary ties."""
    n_docs = len(doc_ids)
    if n_docs > max(k_eff * 4, 64):
        kth_val = float(np.partition(row, n_docs - k_eff)[n_docs - k_eff])
        cand = np.flatnonzero(row >= kth_val)
        return sorted(cand.tolist(), key=lambda j: (-float(row[j]), doc_ids[j]))[:k_eff]
    return sorted(range(n_docs), key=lambda j: (-float(row[j]), doc_ids[j]))[:k_eff]


# ---------------------------------------------------------------------------
# 1. Manifold-Precipitated Anchor Selection (P6 k-Means++, P4 k-Center, A2c-IW)
# ---------------------------------------------------------------------------
def select_spherical_kmeanspp_anchors(
    vecs: np.ndarray,
    n_anchors: int,
    *,
    seed: int = 42,
    candidate_pool_cap: int = 8192,
) -> np.ndarray:
    """Deterministic spherical k-means++ (D^2 angular seeding) anchor selector (`P6`)."""
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


def compute_voronoi_anchor_weights(
    corpus_vecs: np.ndarray,
    anchor_idx: Sequence[int],
    *,
    batch_size: int = 4096,
) -> np.ndarray:
    """Compute normalized spherical Voronoi cluster-mass weights `w_i = |C_i| / N` (`A2c-IW`)."""
    anc_i = np.asarray(anchor_idx, dtype=np.int64)
    if anc_i.size == 0:
        raise ValueError("anchor_idx must be non-empty")
    vn = l2_normalize(corpus_vecs)
    va = vn[anc_i]
    counts = np.zeros(anc_i.size, dtype=np.float64)
    for start in range(0, vn.shape[0], batch_size):
        sims = vn[start : start + batch_size] @ va.T
        nearest = np.argmax(sims, axis=1)
        counts += np.bincount(nearest, minlength=anc_i.size)
    total = float(counts.sum())
    if total <= 0.0:
        return np.full(anc_i.size, 1.0 / float(anc_i.size), dtype=np.float64)
    return counts / total


# ---------------------------------------------------------------------------
# 2. Centered Orthogonal Procrustes (A2 / A2c / A2c-Subspace / OOS / SoftSpec / Polar)
# ---------------------------------------------------------------------------
def classify_dimension_chain_regime(dims: Sequence[int]) -> str:
    """Classify dimension sequence `(d_0, ..., d_m)` into `monotone`, `valley`, or `peak`.

    Compresses equal-dimension plateaus before detecting strict interior peaks `d_{k-1} < d_k > d_{k+1}`.
    """
    if len(dims) <= 2:
        return "monotone"
    comp: List[int] = []
    for d in dims:
        if not comp or comp[-1] != int(d):
            comp.append(int(d))
    if len(comp) <= 2:
        return "monotone"
    diffs = [comp[i + 1] - comp[i] for i in range(len(comp) - 1)]
    if all(x > 0 for x in diffs) or all(x < 0 for x in diffs):
        return "monotone"
    for i in range(1, len(comp) - 1):
        if comp[i - 1] < comp[i] > comp[i + 1]:
            return "peak"
    return "valley"


@dataclass(frozen=True)
class CenteredProcrustesBridge:
    """1st-order centered + conformal orthogonal spherical map (`A2`, `A2c`, `A2c-Subspace`, `A2c-SoftSpec`).

    Applies `y = scale * ((x - mu_x_query) @ Q) + mu_y` before optional L2 projection.
    Supports:
      - Legacy ambient scaling (`scale_mode="ambient"`, `match_radius=True`) and unscaled `A2` (`match_radius=False`),
      - Rank-truncated subspace Stiefel maps (`rank_rel_tol > 0` or `rank_cap`, zero-damping nullspace),
      - All 4 R3 disjoint-CAL scale modes (`s_full_oos`, `s_sub_oos`, `max_one_s_full_oos`, `unit`),
      - Query-centroid shift (`mu_q_override`, `A2c-Q`),
      - Voronoi moment-decoupled weighting (`sample_weights` + `uniform_moments_x/y`, `A2c-IW`),
      - Empirical-Bayes singular shrinkage (`soft_spec_tau > 0`, `EM-5b` `A2c-SoftSpec`).
    """

    mu_x: np.ndarray  # [1, d_in]
    mu_y: np.ndarray  # [1, d_out]
    Q: np.ndarray     # [d_in, d_out]
    fit_count: int
    scale: float = 1.0
    scale_mode: str = "ambient"
    effective_rank: int = 0
    captured_energy_ratio: float = 1.0
    soft_spec_tau: float = 0.0
    regime: str = "monotone"
    pre_polar_sigma_min: float = 1.0
    polar_reprojected: bool = False
    mu_q_override: Optional[np.ndarray] = None  # [1, d_in] for A2c-Q

    @classmethod
    def fit(
        cls,
        x_src: np.ndarray,
        y_tgt: np.ndarray,
        *,
        fit_idx: Optional[Sequence[int]] = None,
        match_radius: bool = True,
        scale_mode: Optional[str] = None,
        cal_x: Optional[np.ndarray] = None,
        cal_y: Optional[np.ndarray] = None,
        cal_fraction: float = 0.0,
        rank_rel_tol: Optional[float] = None,
        rank_cap: Optional[int] = None,
        soft_spec_tau: float = 0.0,
        sample_weights: Optional[np.ndarray] = None,
        uniform_moments_x: Optional[np.ndarray] = None,
        uniform_moments_y: Optional[np.ndarray] = None,
        mu_q_override: Optional[np.ndarray] = None,
        strict_energy_check: bool = False,
    ) -> "CenteredProcrustesBridge":
        xs_all = np.asarray(x_src if fit_idx is None else x_src[np.asarray(fit_idx)], dtype=np.float64)
        yt_all = np.asarray(y_tgt if fit_idx is None else y_tgt[np.asarray(fit_idx)], dtype=np.float64)
        if xs_all.shape[0] != yt_all.shape[0] or xs_all.shape[0] < 2:
            raise ValueError("fit requires matched anchor arrays with at least 2 rows")

        eff_scale_mode = scale_mode if scale_mode is not None else ("ambient" if match_radius else "unit")
        valid_modes = {"ambient", "unit", "s_full_oos", "s_sub_oos", "max_one_s_full_oos"}
        if eff_scale_mode not in valid_modes:
            raise ValueError(f"Unknown scale_mode={eff_scale_mode!r}; expected one of {sorted(valid_modes)}")

        # Optional internal disjoint FIT / CAL split when an OOS mode is requested without explicit cal_x/cal_y
        if cal_x is None and cal_y is None and eff_scale_mode in {"s_full_oos", "s_sub_oos", "max_one_s_full_oos"}:
            frac = cal_fraction if cal_fraction > 0.0 else 0.20
            n_tot = xs_all.shape[0]
            n_cal = max(2, int(round(n_tot * frac)))
            if n_tot - n_cal < 2:
                raise ValueError("Not enough rows for disjoint FIT/CAL split; supply cal_x/cal_y or >= 4 rows")
            xs, yt = xs_all[:-n_cal], yt_all[:-n_cal]
            cx, cy = xs_all[-n_cal:], yt_all[-n_cal:]
            w_fit = None if sample_weights is None else np.asarray(sample_weights[:-n_cal], dtype=np.float64)
        else:
            xs, yt = xs_all, yt_all
            cx = None if cal_x is None else np.asarray(cal_x, dtype=np.float64)
            cy = None if cal_y is None else np.asarray(cal_y, dtype=np.float64)
            w_fit = None if sample_weights is None else np.asarray(sample_weights, dtype=np.float64)

        mu_x = (
            np.asarray(uniform_moments_x, dtype=np.float64).mean(axis=0, keepdims=True)
            if uniform_moments_x is not None
            else xs.mean(axis=0, keepdims=True)
        )
        mu_y = (
            np.asarray(uniform_moments_y, dtype=np.float64).mean(axis=0, keepdims=True)
            if uniform_moments_y is not None
            else yt.mean(axis=0, keepdims=True)
        )
        xc = xs - mu_x
        yc = yt - mu_y
        rad_x_fit = float(np.linalg.norm(xc))
        rad_y_fit = float(np.linalg.norm(yc))
        if rad_x_fit <= 1e-8 and rad_y_fit > 1e-8:
            raise ValueError("Degenerate constant source anchors against non-constant targets (unidentified fit)")
        if strict_energy_check and (rad_x_fit <= 1e-8 or rad_y_fit <= 1e-8):
            raise ValueError("Degenerate zero-variance anchor cloud (strict_energy_check=True)")

        if w_fit is not None:
            w_norm = w_fit / max(float(w_fit.sum()), 1e-12)
            cross = (xc * w_norm[:, None]).T @ yc
        else:
            cross = xc.T @ yc

        u, sv, vt = np.linalg.svd(cross, full_matrices=False)
        sv0 = float(sv[0]) if sv.size > 0 else 0.0
        if strict_energy_check and sv0 <= 1e-8:
            raise ValueError("Zero cross-covariance singular value (unidentified bridge)")

        # Rank selection / truncation (A2c-Subspace) or soft spectral shrinkage (EM-5b A2c-SoftSpec)
        if soft_spec_tau > 0.0:
            if sv0 <= 1e-12:
                raise ValueError("Cannot apply soft_spec_tau on zero cross-covariance")
            weights_sv = sv / (sv + float(soft_spec_tau) * sv0)
            if rank_rel_tol is not None:
                weights_sv = np.where(sv > float(rank_rel_tol) * sv0, weights_sv, 0.0)
            if rank_cap is not None and int(rank_cap) < weights_sv.size:
                weights_sv[int(rank_cap) :] = 0.0
            eff_rank = int(np.sum(weights_sv > 1e-12))
            q = (u * weights_sv[None, :]) @ vt
        elif rank_rel_tol is not None or rank_cap is not None:
            tol = float(rank_rel_tol) if rank_rel_tol is not None else 1e-6
            keep = int(np.sum(sv > tol * sv0)) if sv0 > 1e-12 else 0
            if rank_cap is not None:
                keep = min(keep, int(rank_cap))
            if keep <= 0:
                if strict_energy_check:
                    raise ValueError("Truncated SVD retained rank 0")
                eff_rank = 0
                q = np.zeros((xs.shape[1], yt.shape[1]), dtype=np.float64)
            else:
                eff_rank = keep
                q = u[:, :keep] @ vt[:keep, :]
        else:
            eff_rank = int(np.sum(sv > 1e-12 * max(sv0, 1.0))) if sv.size > 0 else 0
            q = u @ vt

        # Compute captured energy on FIT or CAL
        if cx is not None and cy is not None:
            cxc = cx - mu_x
            cyc = cy - mu_y
            proj_cx = cxc @ q
            norm_proj = float(np.linalg.norm(proj_cx))
            norm_cxc = float(np.linalg.norm(cxc))
            norm_cyc = float(np.linalg.norm(cyc))
            captured = norm_proj / max(norm_cxc, 1e-12)
        else:
            cxc, cyc = xc, yc
            proj_cx = xc @ q
            norm_proj = float(np.linalg.norm(proj_cx))
            norm_cxc = rad_x_fit
            norm_cyc = rad_y_fit
            captured = norm_proj / max(rad_x_fit, 1e-12)

        if eff_scale_mode == "unit":
            scale = 1.0
        elif eff_scale_mode == "ambient":
            scale = rad_y_fit / max(rad_x_fit, 1e-12)
        else:
            if norm_proj <= 1e-8 or norm_cyc <= 1e-8:
                raise ValueError("Degenerate calibration projected energy in OOS radius estimator")
            if eff_scale_mode == "s_full_oos":
                scale = norm_cyc / norm_proj
            elif eff_scale_mode == "max_one_s_full_oos":
                raw_s = norm_cyc / norm_proj
                scale = max(1.0, raw_s) if xs.shape[1] < yt.shape[1] else raw_s
            else:  # s_sub_oos: ||Y_cal,c P_Q||_F / ||X_cal,c Q||_F
                # P_Q = V_r^T V_r when Q = U_r V_r^T
                if eff_rank > 0:
                    vr = vt[:eff_rank, :]
                    proj_cy_norm = float(np.linalg.norm(cyc @ vr.T))
                else:
                    proj_cy_norm = 0.0
                if proj_cy_norm <= 1e-8:
                    raise ValueError("Degenerate target subspace energy in s_sub_oos")
                scale = proj_cy_norm / norm_proj

        mu_q_arr = (
            np.asarray(mu_q_override, dtype=np.float64).reshape(1, -1)
            if mu_q_override is not None
            else None
        )
        return cls(
            mu_x=mu_x,
            mu_y=mu_y,
            Q=q,
            fit_count=int(xs.shape[0]),
            scale=float(scale),
            scale_mode=eff_scale_mode,
            effective_rank=eff_rank,
            captured_energy_ratio=float(captured),
            soft_spec_tau=float(soft_spec_tau),
            regime="monotone",
            pre_polar_sigma_min=1.0,
            polar_reprojected=False,
            mu_q_override=mu_q_arr,
        )

    def transform(self, x: np.ndarray, *, normalize: bool = True, drop_mu_y: bool = False) -> np.ndarray:
        mu_in = self.mu_q_override if self.mu_q_override is not None else self.mu_x
        disp = (np.asarray(x, dtype=np.float64) - mu_in) @ self.Q
        out = disp if drop_mu_y else (self.scale * disp + self.mu_y)
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
    def nonzero_condition_number(self) -> float:
        if abs(self.scale) <= 1e-12:
            return float("inf")
        sv = np.linalg.svd(self.Q * self.scale, compute_uv=False)
        if sv.size == 0 or sv[0] <= 1e-12:
            return float("inf")
        pos = sv[sv > 1e-9 * sv[0]]
        if pos.size == 0:
            return float("inf")
        return float(pos[0] / pos[-1])

    @property
    def weights_sha256(self) -> str:
        parts = [self.mu_x, self.mu_y, self.Q, np.asarray([self.scale], dtype=np.float64)]
        if self.mu_q_override is not None:
            parts.append(self.mu_q_override)
        return _sha256_arrays(*parts)


def compose_centered_bridges(
    bridges: Sequence[CenteredProcrustesBridge],
    *,
    polar_reproject: bool = False,
    rank_rel_tol: float = 1e-6,
    cal_x: Optional[np.ndarray] = None,
    cal_y: Optional[np.ndarray] = None,
    scale_mode: Optional[str] = None,
) -> CenteredProcrustesBridge:
    """Compose a chain `[B_{m->m-1}, ..., B_{1->0}]` with exact 3-regime classification & optional `A2c-Polar`.

    Theorem 1 (Three-Regime Stiefel Trichotomy):
      - `monotone`: guaranteed full thin rank with `cond = 1.00`.
      - `valley`: guaranteed bottleneck rank `min(d_k)` with `nonzero_cond = 1.00` (partial isometry).
      - `peak`: Jordan principal cosines `cos(theta_i)` -> `nonzero_cond >= 1`; `polar_reproject=True`
        projects retained modes onto `U_r V_r^T` (`nonzero_cond = 1.00`) while logging `pre_polar_sigma_min`.
    """
    if not bridges:
        raise ValueError("bridges sequence must be non-empty")
    dims = [int(bridges[0].Q.shape[0])] + [int(b.Q.shape[1]) for b in bridges]
    regime = classify_dimension_chain_regime(dims)

    cur = bridges[0]
    for nxt in bridges[1:]:
        q_comp = cur.Q @ nxt.Q
        s_comp = float(cur.scale * nxt.scale)
        mu_y_comp = nxt.scale * ((cur.mu_y - nxt.mu_x) @ nxt.Q) + nxt.mu_y
        cur = CenteredProcrustesBridge(
            mu_x=cur.mu_x,
            mu_y=mu_y_comp,
            Q=q_comp,
            fit_count=min(cur.fit_count, nxt.fit_count),
            scale=s_comp,
            scale_mode=cur.scale_mode,
            regime=regime,
            mu_q_override=cur.mu_q_override,
        )

    u, sv, vt = np.linalg.svd(cur.Q, full_matrices=False)
    sv0 = float(sv[0]) if sv.size > 0 else 0.0
    keep = int(np.sum(sv > float(rank_rel_tol) * sv0)) if sv0 > 1e-12 else 0
    pre_min = float(sv[keep - 1]) if keep > 0 else 0.0
    q_final = cur.Q
    if polar_reproject:
        if keep <= 0:
            raise ValueError("Polar re-projection rejected: composed chain has rank 0")
        q_final = u[:, :keep] @ vt[:keep, :]

    s_final = cur.scale
    mu_y_final = cur.mu_y
    out_scale_mode = scale_mode if scale_mode is not None else cur.scale_mode
    if cal_x is not None and cal_y is not None:
        cxc = np.asarray(cal_x, dtype=np.float64) - cur.mu_x
        cy = np.asarray(cal_y, dtype=np.float64)
        mu_y_final = cy.mean(axis=0, keepdims=True)
        cyc = cy - mu_y_final
        norm_proj = float(np.linalg.norm(cxc @ q_final))
        norm_cyc = float(np.linalg.norm(cyc))
        if norm_proj <= 1e-8 or norm_cyc <= 1e-8:
            raise ValueError("Degenerate calibration energy during composed bridge recalibration")
        if out_scale_mode == "unit":
            s_final = 1.0
        elif out_scale_mode == "max_one_s_full_oos":
            raw_s = norm_cyc / norm_proj
            s_final = max(1.0, raw_s) if q_final.shape[0] < q_final.shape[1] else raw_s
        elif out_scale_mode == "s_sub_oos":
            vr = vt[:keep, :]
            s_final = float(np.linalg.norm(cyc @ vr.T)) / norm_proj
        else:
            s_final = norm_cyc / norm_proj

    return CenteredProcrustesBridge(
        mu_x=cur.mu_x,
        mu_y=mu_y_final,
        Q=q_final,
        fit_count=cur.fit_count,
        scale=float(s_final),
        scale_mode=out_scale_mode,
        effective_rank=keep,
        captured_energy_ratio=float(np.sum(sv[:keep] ** 2) / max(float(np.sum(sv ** 2)), 1e-12)) if sv.size else 0.0,
        regime=regime,
        pre_polar_sigma_min=pre_min,
        polar_reprojected=bool(polar_reproject),
        mu_q_override=cur.mu_q_override,
    )


# ---------------------------------------------------------------------------
# 2B. Riemannian Geodesic Polar Tangent Alignment (A2g — R3 Ratified Spec)
# ---------------------------------------------------------------------------
def _householder_tangent_basis(pole: np.ndarray) -> np.ndarray:
    """Return `(d, d-1)` orthonormal tangent basis `B` (`B^T pole = 0, B^T B = I_{d-1}`) via Householder."""
    p = l2_normalize(np.asarray(pole, dtype=np.float64).reshape(-1))
    d = p.shape[0]
    e1 = np.zeros(d, dtype=np.float64)
    e1[0] = 1.0
    sign = 1.0 if p[0] >= 0.0 else -1.0
    v = p + sign * e1
    nv = float(np.linalg.norm(v))
    if nv <= 1e-12:
        # p == -e1; standard basis e_2..e_d is already orthogonal to p
        return np.eye(d, dtype=np.float64)[:, 1:]
    v = v / nv
    # H = I - 2 v v^T maps e1 -> -sign * p, so columns 1..d-1 of H span p^\perp
    eye_tail = np.eye(d, dtype=np.float64)[:, 1:]
    return eye_tail - 2.0 * np.outer(v, v[1:])


def _split_half_pole_drift_deg(vecs: np.ndarray, anchor_ids: Optional[Sequence[str]] = None) -> float:
    """Deterministic alternating split-half pole drift in degrees (R3 dual guard)."""
    vn = l2_normalize(vecs)
    n = vn.shape[0]
    if n < 4:
        return 180.0
    if anchor_ids is not None:
        order = sorted(
            range(n),
            key=lambda i: hashlib.sha256(str(anchor_ids[i]).encode("utf-8")).hexdigest(),
        )
    else:
        order = sorted(
            range(n),
            key=lambda i: hashlib.sha256(np.ascontiguousarray(vn[i]).tobytes()).hexdigest(),
        )
    h1 = vn[order[0::2]].mean(axis=0)
    h2 = vn[order[1::2]].mean(axis=0)
    n1, n2 = float(np.linalg.norm(h1)), float(np.linalg.norm(h2))
    if n1 <= 1e-12 or n2 <= 1e-12:
        return 180.0
    cos_a = float(np.clip(np.dot(h1 / n1, h2 / n2), -1.0, 1.0))
    return float(math.degrees(math.acos(cos_a)))


def _sphere_log_map(x: np.ndarray, pole: np.ndarray, eps_cut: float = 0.05) -> Tuple[np.ndarray, np.ndarray]:
    """Exact 2-step Gram-Schmidt + `atan2` Riemannian Log map at `pole`. Returns `(logs, rejected_mask)`."""
    xn = l2_normalize(x)
    p = l2_normalize(pole.reshape(-1))
    c = np.clip(xn @ p, -1.0, 1.0)
    h1 = xn - c[:, None] * p[None, :]
    h = h1 - (h1 @ p)[:, None] * p[None, :]  # 2-step Gram-Schmidt for machine orthogonality
    hn = np.linalg.norm(h, axis=1)
    theta = np.arctan2(hn, c)
    rejected = theta >= (math.pi - eps_cut)
    # Stable theta / ||h|| = 1 / sinc(theta/pi) -> exact limit 1.0 as ||h|| -> 0
    scale = np.where(hn > 1e-12, theta / np.where(hn > 1e-12, hn, 1.0), 1.0)
    logs = h * scale[:, None]
    logs[rejected] = 0.0
    return logs, rejected


def _sphere_exp_map(
    w: np.ndarray,
    pole: np.ndarray,
    *,
    collar: str = "c1_soft",
    r0: float = math.pi - 0.10,
    delta: float = 0.05,
    eps_cut: float = 0.05,
) -> Tuple[np.ndarray, np.ndarray]:
    """Riemannian Exp map with Carbon's exact C^1 soft-collar default or hard clamp toggle."""
    p = l2_normalize(pole.reshape(-1))
    # Re-orthogonalize tangent vectors to machine precision
    w_tan = w - (w @ p)[:, None] * p[None, :]
    r = np.linalg.norm(w_tan, axis=1)
    if collar == "c1_soft":
        r_eff = np.where(r <= r0, r, r0 + delta * np.tanh((r - r0) / max(delta, 1e-12)))
        clipped = r > r0
    elif collar == "hard":
        r_max = math.pi - eps_cut
        r_eff = np.minimum(r, r_max)
        clipped = r > r_max
    else:
        raise ValueError(f"Unknown collar={collar!r}; expected 'c1_soft' or 'hard'")
    # Exp_p(w') = cos(r_eff) * p + sinc(r_eff / pi) * (r_eff / r) * w_tan
    # Note: np.sinc(u) = sin(pi * u) / (pi * u), so np.sinc(r_eff / pi) == sin(r_eff) / r_eff
    sinc_val = np.sinc(r_eff / math.pi)
    ratio = np.where(r > 1e-12, r_eff / np.where(r > 1e-12, r, 1.0), 1.0)
    out = np.cos(r_eff)[:, None] * p[None, :] + (sinc_val * ratio)[:, None] * w_tan
    return l2_normalize(out), clipped


@dataclass(frozen=True)
class GeodesicPolarBridge:
    """Riemannian Geodesic Polar Tangent Alignment (`A2g`, Gauss's Lemma Log/Exp).

    Note: Radial geodesic stretch is `s_theta`, transverse stretch at angle `theta` is
    `sin(s_theta * theta) / sin(theta)` (radial isometry, NOT globally conformal).
    """

    pole_x: np.ndarray
    pole_y: np.ndarray
    basis_x: np.ndarray  # [d_in, d_in - 1]
    basis_y: np.ndarray  # [d_out, d_out - 1]
    Q_tan: np.ndarray    # [d_in - 1, d_out - 1]
    s_theta: float
    fit_count: int
    effective_rank: int
    resultant_x: float
    resultant_y: float
    split_half_drift_x_deg: float
    split_half_drift_y_deg: float
    collar: str = "c1_soft"
    eps_cut: float = 0.05
    fallback_bridge: Optional[CenteredProcrustesBridge] = None
    used_fallback: bool = False
    fallback_reason: str = ""

    @classmethod
    def fit(
        cls,
        x_src: np.ndarray,
        y_tgt: np.ndarray,
        *,
        fit_idx: Optional[Sequence[int]] = None,
        cal_x: Optional[np.ndarray] = None,
        cal_y: Optional[np.ndarray] = None,
        cal_fraction: float = 0.20,
        anchor_ids: Optional[Sequence[str]] = None,
        min_resultant: float = 0.15,
        max_split_half_deg: float = 15.0,
        rank_rel_tol: float = 1e-6,
        collar: str = "c1_soft",
        eps_cut: float = 0.05,
    ) -> "GeodesicPolarBridge":
        xs_all = l2_normalize(x_src if fit_idx is None else x_src[np.asarray(fit_idx)])
        yt_all = l2_normalize(y_tgt if fit_idx is None else y_tgt[np.asarray(fit_idx)])
        if xs_all.shape[0] != yt_all.shape[0] or xs_all.shape[0] < 4:
            raise ValueError("GeodesicPolarBridge requires >= 4 matched anchor pairs")

        if cal_x is None and cal_y is None:
            n_cal = max(2, int(round(xs_all.shape[0] * cal_fraction)))
            if xs_all.shape[0] - n_cal < 4:
                n_cal = max(2, xs_all.shape[0] // 5)
            xs, yt = xs_all[:-n_cal], yt_all[:-n_cal]
            cx, cy = xs_all[-n_cal:], yt_all[-n_cal:]
            fit_aids = None if anchor_ids is None else list(anchor_ids[:-n_cal])
        else:
            xs, yt = xs_all, yt_all
            cx = l2_normalize(cal_x)
            cy = l2_normalize(cal_y)
            fit_aids = None if anchor_ids is None else list(anchor_ids)

        fallback = CenteredProcrustesBridge.fit(
            xs, yt, cal_x=cx, cal_y=cy, scale_mode="s_full_oos", rank_rel_tol=rank_rel_tol, strict_energy_check=False
        )

        mx = xs.mean(axis=0)
        my = yt.mean(axis=0)
        res_x = float(np.linalg.norm(mx))
        res_y = float(np.linalg.norm(my))
        drift_x = _split_half_pole_drift_deg(xs, fit_aids)
        drift_y = _split_half_pole_drift_deg(yt, fit_aids)

        reasons: List[str] = []
        if res_x < min_resultant or res_y < min_resultant:
            reasons.append(f"LOW_RESULTANT(x={res_x:.3f},y={res_y:.3f}<{min_resultant})")
        if drift_x > max_split_half_deg or drift_y > max_split_half_deg:
            reasons.append(f"POLE_DRIFT(x={drift_x:.1f}deg,y={drift_y:.1f}deg>{max_split_half_deg})")

        if reasons:
            return cls(
                pole_x=l2_normalize(mx) if res_x > 1e-12 else np.eye(1, xs.shape[1])[0],
                pole_y=l2_normalize(my) if res_y > 1e-12 else np.eye(1, yt.shape[1])[0],
                basis_x=np.zeros((xs.shape[1], max(xs.shape[1] - 1, 1)), dtype=np.float64),
                basis_y=np.zeros((yt.shape[1], max(yt.shape[1] - 1, 1)), dtype=np.float64),
                Q_tan=np.zeros((max(xs.shape[1] - 1, 1), max(yt.shape[1] - 1, 1)), dtype=np.float64),
                s_theta=1.0,
                fit_count=int(xs.shape[0]),
                effective_rank=fallback.effective_rank,
                resultant_x=res_x,
                resultant_y=res_y,
                split_half_drift_x_deg=drift_x,
                split_half_drift_y_deg=drift_y,
                collar=collar,
                eps_cut=eps_cut,
                fallback_bridge=fallback,
                used_fallback=True,
                fallback_reason=";".join(reasons),
            )

        px = mx / res_x
        py = my / res_y
        bx = _householder_tangent_basis(px)
        by = _householder_tangent_basis(py)

        lx_fit, rej_x = _sphere_log_map(xs, px, eps_cut=eps_cut)
        ly_fit, rej_y = _sphere_log_map(yt, py, eps_cut=eps_cut)
        ok_fit = (~rej_x) & (~rej_y)
        if int(ok_fit.sum()) < 2:
            return cls(
                pole_x=px, pole_y=py, basis_x=bx, basis_y=by,
                Q_tan=np.zeros((bx.shape[1], by.shape[1]), dtype=np.float64),
                s_theta=1.0, fit_count=int(xs.shape[0]), effective_rank=0,
                resultant_x=res_x, resultant_y=res_y,
                split_half_drift_x_deg=drift_x, split_half_drift_y_deg=drift_y,
                collar=collar, eps_cut=eps_cut, fallback_bridge=fallback,
                used_fallback=True, fallback_reason="ANTIPODAL_FIT_COLLAPSE",
            )

        zx = lx_fit[ok_fit] @ bx
        zy = ly_fit[ok_fit] @ by
        u, sv, vt = np.linalg.svd(zx.T @ zy, full_matrices=False)
        sv0 = float(sv[0]) if sv.size > 0 else 0.0
        keep = int(np.sum(sv > rank_rel_tol * sv0)) if sv0 > 1e-12 else 0
        if keep <= 0:
            return cls(
                pole_x=px, pole_y=py, basis_x=bx, basis_y=by,
                Q_tan=np.zeros((bx.shape[1], by.shape[1]), dtype=np.float64),
                s_theta=1.0, fit_count=int(xs.shape[0]), effective_rank=0,
                resultant_x=res_x, resultant_y=res_y,
                split_half_drift_x_deg=drift_x, split_half_drift_y_deg=drift_y,
                collar=collar, eps_cut=eps_cut, fallback_bridge=fallback,
                used_fallback=True, fallback_reason="ZERO_TANGENT_RANK",
            )
        q_tan = u[:, :keep] @ vt[:keep, :]

        # Disjoint calibration scale s_theta
        lx_cal, rej_cx = _sphere_log_map(cx, px, eps_cut=eps_cut)
        ly_cal, rej_cy = _sphere_log_map(cy, py, eps_cut=eps_cut)
        ok_cal = (~rej_cx) & (~rej_cy)
        if int(ok_cal.sum()) < 1:
            return cls(
                pole_x=px, pole_y=py, basis_x=bx, basis_y=by, Q_tan=q_tan,
                s_theta=1.0, fit_count=int(xs.shape[0]), effective_rank=keep,
                resultant_x=res_x, resultant_y=res_y,
                split_half_drift_x_deg=drift_x, split_half_drift_y_deg=drift_y,
                collar=collar, eps_cut=eps_cut, fallback_bridge=fallback,
                used_fallback=True, fallback_reason="ANTIPODAL_CAL_COLLAPSE",
            )
        z_cal_x = (lx_cal[ok_cal] @ bx) @ q_tan
        z_cal_y = ly_cal[ok_cal] @ by
        nx_cal = float(np.linalg.norm(z_cal_x))
        ny_cal = float(np.linalg.norm(z_cal_y))
        if nx_cal <= 1e-8 or ny_cal <= 1e-8:
            return cls(
                pole_x=px, pole_y=py, basis_x=bx, basis_y=by, Q_tan=q_tan,
                s_theta=1.0, fit_count=int(xs.shape[0]), effective_rank=keep,
                resultant_x=res_x, resultant_y=res_y,
                split_half_drift_x_deg=drift_x, split_half_drift_y_deg=drift_y,
                collar=collar, eps_cut=eps_cut, fallback_bridge=fallback,
                used_fallback=True, fallback_reason="DEGENERATE_CAL_ENERGY",
            )

        return cls(
            pole_x=px,
            pole_y=py,
            basis_x=bx,
            basis_y=by,
            Q_tan=q_tan,
            s_theta=float(ny_cal / nx_cal),
            fit_count=int(xs.shape[0]),
            effective_rank=keep,
            resultant_x=res_x,
            resultant_y=res_y,
            split_half_drift_x_deg=drift_x,
            split_half_drift_y_deg=drift_y,
            collar=collar,
            eps_cut=eps_cut,
            fallback_bridge=fallback,
            used_fallback=False,
            fallback_reason="",
        )

    def transform_with_stats(self, x: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        arr = np.asarray(x, dtype=np.float64)
        if self.used_fallback:
            assert self.fallback_bridge is not None
            return self.fallback_bridge.transform(arr, normalize=True), {
                "used_fit_fallback": True,
                "fallback_reason": self.fallback_reason,
                "frac_antipodal_fallback": 1.0,
                "frac_clipped": 0.0,
            }
        lx, rej = _sphere_log_map(arr, self.pole_x, eps_cut=self.eps_cut)
        w = self.s_theta * ((lx @ self.basis_x) @ self.Q_tan) @ self.basis_y.T
        out, clipped = _sphere_exp_map(w, self.pole_y, collar=self.collar, eps_cut=self.eps_cut)
        if np.any(rej) and self.fallback_bridge is not None:
            out[rej] = self.fallback_bridge.transform(arr[rej], normalize=True)
        return out, {
            "used_fit_fallback": False,
            "fallback_reason": "",
            "frac_antipodal_fallback": float(np.mean(rej)),
            "frac_clipped": float(np.mean(clipped)),
        }

    def transform(self, x: np.ndarray, *, normalize: bool = True) -> np.ndarray:
        out, _ = self.transform_with_stats(x)
        return l2_normalize(out) if normalize else out

    @property
    def weights_sha256(self) -> str:
        return _sha256_arrays(
            self.pole_x, self.pole_y, self.Q_tan, np.asarray([self.s_theta], dtype=np.float64)
        )


# ---------------------------------------------------------------------------
# 2C. EM-2 Subtractive Centroid Removal (A2-ZeroMu: QueryOnly / DualFit / DualAdap)
# ---------------------------------------------------------------------------
def score_zeromu_rankings(
    query_vectors: np.ndarray,
    doc_vectors: np.ndarray,
    bridge: CenteredProcrustesBridge,
    query_ids: Sequence[str],
    doc_ids: Sequence[str],
    *,
    mode: str = "query_only",
    k: int = 10,
    csls_k: Optional[int] = None,
    csls_reference_queries: Optional[np.ndarray] = None,
) -> Tuple[Dict[str, List[str]], Dict[str, np.ndarray]]:
    """EM-2 Subtractive Centroid Removal scorer (exact scale-invariance in `s`).

    Modes:
      - `query_only`: TRUE ZERO-WRITE (`<norm2((q - mu_x) Q), d_j>` on unmodified cosine index).
      - `dual_fit`: Zero-reindex 1-float sidecar (`(<q_hat, d_j> - <q_hat, mu_y>) / ||d_j - mu_y||_2`).
      - `dual_adap`: Unlabeled resident-index mean `mu_d_res` + eval-query mean `mu_q_eval` (0 target anchors).
    """
    if mode not in {"query_only", "dual_fit", "dual_adap"}:
        raise ValueError(f"Unknown zeromu mode={mode!r}")

    q_arr = np.asarray(query_vectors, dtype=np.float64)
    d_norm = l2_normalize(doc_vectors)
    mu_q = q_arr.mean(axis=0, keepdims=True) if mode == "dual_adap" else (
        bridge.mu_q_override if bridge.mu_q_override is not None else bridge.mu_x
    )
    q_hat = l2_normalize((q_arr - mu_q) @ bridge.Q)

    if mode == "query_only":
        sims = q_hat @ d_norm.T
        ref_d_mat = d_norm
        mu_d = None
        doc_den = None
    else:
        mu_d = d_norm.mean(axis=0, keepdims=True) if mode == "dual_adap" else bridge.mu_y
        # Exact zero-reindex 1-float identity: (<q_hat, d_j> - <q_hat, mu_d>) / ||d_j - mu_d||_2
        mu_d_vec = mu_d.reshape(-1)
        mu_d_norm2 = float(np.dot(mu_d_vec, mu_d_vec))
        doc_den = np.sqrt(np.maximum(1.0 + mu_d_norm2 - 2.0 * (d_norm @ mu_d_vec), 1e-12))
        sims = (q_hat @ d_norm.T - (q_hat @ mu_d_vec)[:, None]) / doc_den[None, :]
        ref_d_mat = None

    if csls_k is not None and int(csls_k) > 0:
        if csls_reference_queries is not None:
            rq_arr = np.asarray(csls_reference_queries, dtype=np.float64)
            rq_mu = rq_arr.mean(axis=0, keepdims=True) if mode == "dual_adap" else mu_q
            ref_q = l2_normalize((rq_arr - rq_mu) @ bridge.Q)
        else:
            ref_q = q_hat
        eff_k = max(1, min(int(csls_k), ref_q.shape[0]))
        if mode == "query_only":
            ref_sims = ref_q @ ref_d_mat.T
        else:
            assert mu_d is not None and doc_den is not None
            mu_d_vec = mu_d.reshape(-1)
            ref_sims = (ref_q @ d_norm.T - (ref_q @ mu_d_vec)[:, None]) / doc_den[None, :]
        if eff_k < ref_sims.shape[0]:
            top_part = np.partition(ref_sims, ref_sims.shape[0] - eff_k, axis=0)[-eff_k:, :]
        else:
            top_part = ref_sims
        r_d = top_part.mean(axis=0)
        sims = 2.0 * sims - r_d[None, :]

    k_eff = min(int(k), len(doc_ids))
    rankings: Dict[str, List[str]] = {}
    top_scores: Dict[str, np.ndarray] = {}
    for row_idx, qid in enumerate(query_ids):
        row = sims[row_idx]
        ordered = _canonical_topk(row, doc_ids, k_eff)
        rankings[qid] = [doc_ids[j] for j in ordered]
        top_scores[qid] = row[ordered]
    return rankings, top_scores


# ---------------------------------------------------------------------------
# 3. Centered Shrinkage-Whitened (CORAL) Spherical Procrustes (A3 / A3-Subspace / A3-EffRank)
# ---------------------------------------------------------------------------
def _shrinkage_cov_factors(
    xc: np.ndarray,
    shrink: float,
    *,
    trace_divisor_mode: str = "ambient",
) -> Tuple[np.ndarray, np.ndarray]:
    """Exact O(min(n, d)^2 * max(n, d)) shrinkage covariance inverse-sqrt & sqrt factors."""
    if not (0.0 <= shrink <= 1.0):
        raise ValueError("shrink must be in [0, 1]")
    n, d = xc.shape
    denom = float(max(n - 1, 1))

    if n < d:
        # Exact dual thin-SVD formula avoiding d x d eigh (O(n^2 d), machine precision == eigh)
        _u, s, vt = np.linalg.svd(xc, full_matrices=False)
        lam = (s ** 2) / denom
        tr_sum = float(np.sum(lam))
        if trace_divisor_mode == "ambient":
            tr_scale = tr_sum / float(d)
        elif trace_divisor_mode == "active_rank":
            r_act = max(1, int(np.sum(lam > 1e-12 * max(float(lam[0]) if lam.size else 0.0, 1.0))))
            tr_scale = tr_sum / float(r_act)
        elif trace_divisor_mode == "effective_rank":
            tr2 = float(np.sum(lam ** 2))
            r_eff = (tr_sum * tr_sum) / max(tr2, 1e-18) if tr_sum > 0.0 else float(d)
            tr_scale = tr_sum / max(r_eff, 1.0)
        else:
            raise ValueError(f"Unknown trace_divisor_mode={trace_divisor_mode!r}")

        c0 = max(shrink * tr_scale, 1e-8)
        lam_reg = np.maximum((1.0 - shrink) * lam + shrink * tr_scale, 1e-8)
        c0_inv = c0 ** -0.5
        c0_sq = c0 ** 0.5
        # W = c0_p * I + V_k diag(lam_reg^p - c0_p) V_k^T
        v_mat = vt.T  # [d, r]
        inv_sqrt = (v_mat * (lam_reg ** -0.5 - c0_inv)[None, :]) @ vt
        inv_sqrt[np.diag_indices(d)] += c0_inv
        sqrt_mat = (v_mat * (lam_reg ** 0.5 - c0_sq)[None, :]) @ vt
        sqrt_mat[np.diag_indices(d)] += c0_sq
        return inv_sqrt, sqrt_mat

    cov = (xc.T @ xc) / denom
    tr_sum = float(np.trace(cov))
    eigvals_raw, eigvecs = np.linalg.eigh(cov)
    eigvals_raw = np.maximum(eigvals_raw, 0.0)
    if trace_divisor_mode == "ambient":
        tr_scale = tr_sum / float(d)
    elif trace_divisor_mode == "active_rank":
        r_act = max(1, int(np.sum(eigvals_raw > 1e-12 * max(float(eigvals_raw[-1]) if eigvals_raw.size else 0.0, 1.0))))
        tr_scale = tr_sum / float(r_act)
    elif trace_divisor_mode == "effective_rank":
        tr2 = float(np.sum(eigvals_raw ** 2))
        r_eff = (tr_sum * tr_sum) / max(tr2, 1e-18) if tr_sum > 0.0 else float(d)
        tr_scale = tr_sum / max(r_eff, 1.0)
    else:
        raise ValueError(f"Unknown trace_divisor_mode={trace_divisor_mode!r}")

    eigvals = np.maximum((1.0 - shrink) * eigvals_raw + shrink * tr_scale, 1e-8)
    inv_sqrt = (eigvecs * (eigvals ** -0.5)) @ eigvecs.T
    sqrt_mat = (eigvecs * (eigvals ** 0.5)) @ eigvecs.T
    return inv_sqrt, sqrt_mat


@dataclass(frozen=True)
class WhitenedSphericalProcrustesBridge:
    """1st-order centered + 2nd-order shrinkage-whitened CORAL Procrustes (`A3`, `A3-Subspace`, `A3-EffRank`)."""

    mu_x: np.ndarray  # [1, d_in]
    mu_y: np.ndarray  # [1, d_out]
    W: np.ndarray     # [d_in, d_out]
    shrink: float
    fit_count: int
    subspace_rank: Optional[int] = None
    trace_divisor_mode: str = "ambient"
    identified_cross_rank: int = 0

    @classmethod
    def fit(
        cls,
        x_src: np.ndarray,
        y_tgt: np.ndarray,
        *,
        fit_idx: Optional[Sequence[int]] = None,
        shrink: float = 0.20,
        subspace_rank: Optional[int] = None,
        trace_divisor_mode: str = "ambient",
        strict_energy_check: bool = False,
    ) -> "WhitenedSphericalProcrustesBridge":
        xs = np.asarray(x_src if fit_idx is None else x_src[np.asarray(fit_idx)], dtype=np.float64)
        yt = np.asarray(y_tgt if fit_idx is None else y_tgt[np.asarray(fit_idx)], dtype=np.float64)
        if xs.shape[0] != yt.shape[0] or xs.shape[0] < 2:
            raise ValueError("fit requires matched anchor arrays with at least 2 rows")
        mu_x = xs.mean(axis=0, keepdims=True)
        mu_y = yt.mean(axis=0, keepdims=True)
        xc = xs - mu_x
        yc = yt - mu_y
        rad_x = float(np.linalg.norm(xc))
        rad_y = float(np.linalg.norm(yc))
        if strict_energy_check and (rad_x <= 1e-8 or rad_y <= 1e-8):
            raise ValueError("Degenerate constant source or target anchors in WhitenedSphericalProcrustesBridge")

        if subspace_rank is not None:
            # Explicit PCA subspace-restricted whitening (works cleanly in BOTH up- and down-projections!)
            r_x = min(int(subspace_rank), xs.shape[0] - 1, xs.shape[1])
            r_y = min(int(subspace_rank), yt.shape[0] - 1, yt.shape[1])
            if r_x <= 0 or r_y <= 0:
                raise ValueError("subspace_rank must be positive")
            _, _, vtx = np.linalg.svd(xc, full_matrices=False)
            _, _, vty = np.linalg.svd(yc, full_matrices=False)
            bx = vtx[:r_x, :].T  # [d_in, r_x]
            by = vty[:r_y, :].T  # [d_out, r_y]
            xc_sub = xc @ bx
            yc_sub = yc @ by
            wx_inv, _ = _shrinkage_cov_factors(xc_sub, shrink=shrink, trace_divisor_mode=trace_divisor_mode)
            wy_inv, wy_sqrt = _shrinkage_cov_factors(yc_sub, shrink=shrink, trace_divisor_mode=trace_divisor_mode)
            cross = (xc_sub @ wx_inv).T @ (yc_sub @ wy_inv)
            u, sv, vt = np.linalg.svd(cross, full_matrices=False)
            sv0 = float(sv[0]) if sv.size > 0 else 0.0
            id_rank = int(np.sum(sv > 1e-10 * max(sv0, 1.0))) if sv0 > 1e-12 else 0
            q = u @ vt
            w_total = bx @ wx_inv @ q @ wy_sqrt @ by.T
        else:
            wx_inv, _ = _shrinkage_cov_factors(xc, shrink=shrink, trace_divisor_mode=trace_divisor_mode)
            wy_inv, wy_sqrt = _shrinkage_cov_factors(yc, shrink=shrink, trace_divisor_mode=trace_divisor_mode)
            cross = (xc @ wx_inv).T @ (yc @ wy_inv)
            u, sv, vt = np.linalg.svd(cross, full_matrices=False)
            sv0 = float(sv[0]) if sv.size > 0 else 0.0
            id_rank = int(np.sum(sv > 1e-10 * max(sv0, 1.0))) if sv0 > 1e-12 else 0
            q = u @ vt
            w_total = wx_inv @ q @ wy_sqrt

        return cls(
            mu_x=mu_x,
            mu_y=mu_y,
            W=w_total,
            shrink=float(shrink),
            fit_count=int(xs.shape[0]),
            subspace_rank=subspace_rank,
            trace_divisor_mode=trace_divisor_mode,
            identified_cross_rank=id_rank,
        )

    def transform(self, x: np.ndarray, *, normalize: bool = True) -> np.ndarray:
        out = (np.asarray(x, dtype=np.float64) - self.mu_x) @ self.W + self.mu_y
        return l2_normalize(out) if normalize else out

    @property
    def condition_number(self) -> float:
        if self.identified_cross_rank == 0:
            return float("inf")
        sv = np.linalg.svd(self.W, compute_uv=False)
        if sv.size == 0 or sv[-1] <= 1e-12:
            return float("inf")
        return float(sv[0] / sv[-1])

    @property
    def weights_sha256(self) -> str:
        return _sha256_arrays(self.mu_x, self.mu_y, self.W)


# ---------------------------------------------------------------------------
# 4. Quantile-CDF Backfill Calibration & Old-Space Distortion Prioritization (B1..B4)
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
    """Rank corpus docs by qrels-free spherical angular discrepancy + top-k neighbor KL (`B2`, 100%-encode baseline)."""
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

        t_shift = top_t - top_t.max(axis=1, keepdims=True)
        log_pt = t_shift - np.log(np.sum(np.exp(t_shift), axis=1, keepdims=True))
        pt = np.exp(log_pt)
        s_shift = top_s - top_s.max(axis=1, keepdims=True)
        log_ps = s_shift - np.log(np.sum(np.exp(s_shift), axis=1, keepdims=True))
        kl = np.sum(pt * (log_pt - log_ps), axis=1)
        distortion[start:end] = ang_err + 0.25 * kl

    order = np.argsort(-distortion, kind="mergesort")
    return order, distortion


def compute_old_space_knn_distortion(
    docs_old: np.ndarray,
    anchor_old_idx: Sequence[int],
    anchor_residuals: np.ndarray,
    *,
    k_neighbors: int = 16,
    tau: float = 0.10,
    batch_size: int = 2048,
) -> Tuple[np.ndarray, np.ndarray]:
    """Zero-target-encode old-space kNN anchor-residual distortion predictor (`B2-OldSpace` / `B4`).

    Predicts per-document migration distortion solely from resident `docs_old` and paired anchor
    residuals `e_a = 1 - <b(x_a), y_a>` — requiring ZERO encoding of non-anchor corpus documents.
    """
    anc_i = np.asarray(anchor_old_idx, dtype=np.int64)
    res = np.asarray(anchor_residuals, dtype=np.float64).reshape(-1)
    if anc_i.size == 0 or anc_i.size != res.size:
        raise ValueError("anchor_old_idx and anchor_residuals must be non-empty and aligned")
    d_old = l2_normalize(docs_old)
    a_old = d_old[anc_i]
    k_eff = max(1, min(int(k_neighbors), anc_i.size))
    n_docs = d_old.shape[0]
    pred = np.zeros(n_docs, dtype=np.float64)

    for start in range(0, n_docs, batch_size):
        end = min(n_docs, start + batch_size)
        sims = d_old[start:end] @ a_old.T
        if anc_i.size > k_eff:
            top_idx = np.argpartition(-sims, k_eff - 1, axis=1)[:, :k_eff]
            rows_ar = np.arange(end - start)[:, None]
            top_s = sims[rows_ar, top_idx]
            top_r = res[top_idx]
        else:
            top_s = sims
            top_r = np.broadcast_to(res[None, :], sims.shape)
        logits = (top_s - top_s.max(axis=1, keepdims=True)) / max(float(tau), 1e-6)
        w = np.exp(logits)
        w = w / np.maximum(w.sum(axis=1, keepdims=True), 1e-12)
        pred[start:end] = np.sum(w * top_r, axis=1)

    order = np.argsort(-pred, kind="mergesort")
    return order, pred


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


def _apply_csls_scores(
    scores: np.ndarray,
    ref_queries: np.ndarray,
    docs: np.ndarray,
    csls_k: int,
) -> np.ndarray:
    eff_k = max(1, min(int(csls_k), ref_queries.shape[0]))
    ref_sims = ref_queries @ docs.T
    if eff_k < ref_sims.shape[0]:
        top_part = np.partition(ref_sims, ref_sims.shape[0] - eff_k, axis=0)[-eff_k:, :]
    else:
        top_part = ref_sims
    r_d = top_part.mean(axis=0)
    return 2.0 * scores - r_d[None, :]


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
    legacy_hard_clip: bool = False,
    tail_quantile: float = 0.995,
) -> Dict[str, List[str]]:
    """Monotone empirical-CDF score fusion (`B4` with CSLS + positive-slope right-tail continuation).

    Set `legacy_hard_clip=True` to reproduce pristine `BF20` (`B1`) side-by-side.
    """
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
            csls_k=csls_k if csls_k > 0 else None,
            csls_reference_queries=dev_queries_mapped if csls_k > 0 else None,
        )
        return rankings

    dev_qn = l2_normalize(dev_queries_new)
    dev_qm = l2_normalize(dev_queries_mapped)
    sub_docs = np.linspace(0, n_docs - 1, num=min(n_docs, 4096), dtype=np.int64)

    dev_nat_mat = dev_qn @ d_new_n[sub_docs].T
    dev_map_mat = dev_qm @ d_old_n[sub_docs].T
    mapped_scores = q_map_n @ d_old_n.T
    native_bf_scores = q_new_n @ d_new_n[bf_idx].T

    if csls_k > 0 and not legacy_hard_clip:
        dev_nat_mat = _apply_csls_scores(dev_nat_mat, dev_qn, d_new_n[sub_docs], csls_k)
        dev_map_mat = _apply_csls_scores(dev_map_mat, dev_qm, d_old_n[sub_docs], csls_k)
        mapped_scores = _apply_csls_scores(mapped_scores, dev_qm, d_old_n, csls_k)
        native_bf_scores = _apply_csls_scores(native_bf_scores, dev_qn, d_new_n[bf_idx], csls_k)

    dev_native_sorted = np.sort(dev_nat_mat.ravel())
    dev_mapped_sorted = np.sort(dev_map_mat.ravel())

    if legacy_hard_clip:
        ranks = np.searchsorted(dev_native_sorted, native_bf_scores, side="left")
        quantiles = np.clip(ranks / float(max(dev_native_sorted.size - 1, 1)), 0.0, 1.0)
        mapped_pos = np.clip(
            (quantiles * (dev_mapped_sorted.size - 1)).astype(np.int64),
            0,
            dev_mapped_sorted.size - 1,
        )
        cal_bf_scores = dev_mapped_sorted[mapped_pos]
    else:
        # Strictly increasing interior interpolation + positive-slope right-tail continuation above tail_quantile
        q_grid = np.linspace(0.0, 1.0, num=dev_native_sorted.size)
        tau_hi = float(np.quantile(dev_native_sorted, tail_quantile))
        q_map_hi = float(np.quantile(dev_mapped_sorted, tail_quantile))
        nat_tail = dev_native_sorted[dev_native_sorted >= tau_hi]
        map_tail = dev_mapped_sorted[dev_mapped_sorted >= q_map_hi]
        sd_nat = float(np.std(nat_tail)) if nat_tail.size > 1 else float(np.std(dev_native_sorted))
        sd_map = float(np.std(map_tail)) if map_tail.size > 1 else float(np.std(dev_mapped_sorted))
        slope = max(sd_map / max(sd_nat, 1e-6), 1e-3)  # Strictly positive slope prevents top-1/2 ties!

        # Piecewise linear empirical CDF interpolation on interior
        q_est = np.interp(native_bf_scores, dev_native_sorted, q_grid)
        cal_interior = np.interp(q_est, q_grid, dev_mapped_sorted)
        cal_tail = q_map_hi + slope * (native_bf_scores - tau_hi)
        cal_bf_scores = np.where(native_bf_scores > tau_hi, cal_tail, cal_interior)

    combined = mapped_scores.copy()
    combined[:, bf_idx] = cal_bf_scores

    k_eff = min(k, n_docs)
    rankings: Dict[str, List[str]] = {}
    for row_idx, qid in enumerate(query_ids):
        row = combined[row_idx]
        ordered = _canonical_topk(row, doc_ids, k_eff)
        rankings[qid] = [doc_ids[j] for j in ordered]
    return rankings


# ---------------------------------------------------------------------------
# 5. Hardened TRA-1 Bridge Contract & Fail-Closed Resolution Helper
# ---------------------------------------------------------------------------
def build_bridge_contract(
    *,
    source: RepresentationDescriptor,
    target: RepresentationDescriptor,
    bridge: CenteredProcrustesBridge | WhitenedSphericalProcrustesBridge | GeodesicPolarBridge,
    transform_id: str,
    anchor_manifest_sha256: str,
    validation_count: int,
    validation_confidence: float,
    evidence_id: str,
    evidence_sha256: str,
    role: RepresentationRole = RepresentationRole.QUERY,
    scoring_mode: str = "cosine",
    csls_k: int = 0,
    hop_manifest_sha256s: Sequence[str] = (),
) -> BridgeContract:
    """Wrap any fitted spherical/polar bridge in a content-addressed TRA-1 `BridgeContract`."""
    if isinstance(bridge, CenteredProcrustesBridge):
        family = "centered_orthogonal_procrustes"
        hp: Dict[str, Any] = {
            "family": family,
            "scale_mode": bridge.scale_mode,
            "match_radius": bridge.scale_mode != "unit",
            "effective_rank": bridge.effective_rank,
            "soft_spec_tau": bridge.soft_spec_tau,
            "regime": bridge.regime,
            "polar_reprojected": bridge.polar_reprojected,
            "pre_polar_sigma_min": round(bridge.pre_polar_sigma_min, 8),
            "has_query_centroid": bridge.mu_q_override is not None,
            "scoring_mode": scoring_mode,
            "csls_k": int(csls_k),
            "hop_manifest_sha256s": list(hop_manifest_sha256s),
        }
    elif isinstance(bridge, GeodesicPolarBridge):
        family = "geodesic_polar_tangent"
        hp = {
            "family": family,
            "collar": bridge.collar,
            "eps_cut": bridge.eps_cut,
            "effective_rank": bridge.effective_rank,
            "used_fallback": bridge.used_fallback,
            "fallback_reason": bridge.fallback_reason,
            "scoring_mode": scoring_mode,
            "csls_k": int(csls_k),
            "hop_manifest_sha256s": list(hop_manifest_sha256s),
        }
    else:
        family = "whitened_spherical_procrustes"
        hp = {
            "family": family,
            "shrink": bridge.shrink,
            "subspace_rank": bridge.subspace_rank,
            "trace_divisor_mode": bridge.trace_divisor_mode,
            "identified_cross_rank": bridge.identified_cross_rank,
            "scoring_mode": scoring_mode,
            "csls_k": int(csls_k),
            "hop_manifest_sha256s": list(hop_manifest_sha256s),
        }
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
    "GeodesicPolarBridge",
    "WhitenedSphericalProcrustesBridge",
    "build_bridge_contract",
    "classify_dimension_chain_regime",
    "combine_exposure_and_distortion_order",
    "compose_centered_bridges",
    "compute_old_space_knn_distortion",
    "compute_spherical_distortion_scores",
    "compute_voronoi_anchor_weights",
    "l2_normalize",
    "resolve_with_tra1",
    "score_quantile_calibrated_backfill",
    "score_zeromu_rankings",
    "select_kcenter_anchors",
    "select_spherical_kmeanspp_anchors",
]
