"""Unified 5x3 Bridge x Merge Factorial Harness, 4-Hop Chain, Contamination Sweep, & Span Profiler.

Wave 16-Core (#47) foundational spine module implementing the locked Post-v7 R2
cross-model consensus (`Fable 5.2 + Astra 6.2 + Carbon 4`) for Item 6B (#54) and
shared vertical consumers:

1. Five bridge families supporting rectangular `d_old != d_new` (e.g. `384 <-> 4096`):
   - `UncenteredProcrustes` (SVD Stiefel map `W = U V^T`, no centering)
   - `CenteredProcrustes` (centered Stiefel map + conformal radius scaling)
   - `RidgeDriftAdapter` (centered affine ridge with O(n^2 d) Woodbury solve)
   - `Rank1QueryResidual` (orthogonal base + rank-1 principal residual correction)
   - `ZeroMuPolarBridge` (centered Stiefel + ZeroMu 4-byte sidecar / causal EMA + polar re-projection)
2. Three dual-shard merge methods crossed with EVERY bridge (never denying CSLS
   to ordinary or centered Procrustes):
   - `RawCosineMerge`
   - `CalibratedZScoreMerge` (DEV-only per-shard scalar z-score calibration)
   - `CSLSMerge` (DEV-reference per-shard local CSLS `2*s - r_d - r_q`)
3. Twenty deterministic seeded 500-anchor draws per domain separating within-domain
   anchor SD from cross-domain `N`.
4. Five-encoder / 4-transition sequential composition (`v1 -> v2 -> v3 -> v4 -> v5`)
   compared directly against direct `v1 -> v5` fit (`kappa`, `nonzero_kappa`,
   `nDCG@10`, `Recall@10` at every hop).
5. Progressive anchor contamination sweep (`0%, 1%, 5%, 10%, 20%`) across
   mismatched, duplicate, and off-domain corruption with `TRA-1` outlier/rank
   diagnostics.
6. Decomposed `Item 6B` hypothesis evaluator (`H6B_merge`, `H6B_bridge_static`
   with both `-0.010` retirement floor and `-0.005` tight equivalence plus
   `CenteredProcrustes + CSLS` contrast, and `H6B_bridge_shift_and_4hop`) with
   exact binomial tests, paired bootstrap CIs, and Holm-Bonferroni correction.
7. `RequestSpanProfiler` recording microsecond spans and strictly keeping timed-out
   requests in the denominator as `0.0` `nDCG@10`.
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from contextlib import contextmanager
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple, Type

import numpy as np

from benchmark_utils import canonicalize_id, ndcg_at_k, recall_at_k


DEFAULT_ANCHOR_DRAWS = 20
DEFAULT_ANCHORS_PER_DRAW = 500
DEFAULT_CONTAMINATION_RATES: Tuple[float, ...] = (0.0, 0.01, 0.05, 0.10, 0.20)
DEFAULT_CSLS_K = 10
DEFAULT_EMA_ALPHA = 0.05
ITEM6B_RETIREMENT_FLOOR = -0.010
ITEM6B_TIGHT_EQUIVALENCE_FLOOR = -0.005


def l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalization in float64 with safe zero-norm guard."""
    arr = np.asarray(matrix, dtype=np.float64)
    if arr.ndim == 1:
        norm = float(np.linalg.norm(arr))
        return arr / norm if norm > 1e-12 else arr.copy()
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    return arr / np.where(norms > 1e-12, norms, 1.0)


def _sha256_arrays(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for arr in arrays:
        contiguous = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
        digest.update(str(contiguous.shape).encode("ascii"))
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def _sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_topk(row: np.ndarray, doc_ids: Sequence[str], k: int) -> List[int]:
    """Deterministic global tie-breaking `(-score, doc_id)` safe under arbitrary ties."""
    n_docs = len(doc_ids)
    k_eff = min(int(k), n_docs)
    if k_eff <= 0:
        return []
    if n_docs > max(k_eff * 4, 64):
        kth = float(np.partition(row, n_docs - k_eff)[n_docs - k_eff])
        candidates = np.flatnonzero(row >= kth)
        return sorted(candidates.tolist(), key=lambda j: (-float(row[j]), doc_ids[j]))[:k_eff]
    return sorted(range(n_docs), key=lambda j: (-float(row[j]), doc_ids[j]))[:k_eff]


def classify_dimension_chain_regime(dims: Sequence[int]) -> str:
    """Classify dimension chain `(d_0, ..., d_m)` into `monotone`, `valley`, or `peak`."""
    if len(dims) <= 2:
        return "monotone"
    compressed: List[int] = []
    for dim in dims:
        val = int(dim)
        if not compressed or compressed[-1] != val:
            compressed.append(val)
    if len(compressed) <= 2:
        return "monotone"
    diffs = [compressed[i + 1] - compressed[i] for i in range(len(compressed) - 1)]
    if all(d > 0 for d in diffs) or all(d < 0 for d in diffs):
        return "monotone"
    for i in range(1, len(compressed) - 1):
        if compressed[i - 1] < compressed[i] > compressed[i + 1]:
            return "peak"
    return "valley"


def _singular_diagnostics(matrix: np.ndarray, scale: float = 1.0) -> Tuple[float, float, int, float]:
    """Return `(kappa, nonzero_kappa, effective_rank, min_nonzero_sigma)`."""
    if abs(scale) <= 1e-12:
        return float("inf"), float("inf"), 0, 0.0
    sv = np.linalg.svd(np.asarray(matrix, dtype=np.float64) * float(scale), compute_uv=False)
    if sv.size == 0 or sv[0] <= 1e-12:
        return float("inf"), float("inf"), 0, 0.0
    full_kappa = float(sv[0] / sv[-1]) if sv[-1] > 1e-12 else float("inf")
    pos = sv[sv > 1e-9 * sv[0]]
    if pos.size == 0:
        return full_kappa, float("inf"), 0, 0.0
    return full_kappa, float(pos[0] / pos[-1]), int(pos.size), float(pos[-1])


# ---------------------------------------------------------------------------
# 1. Five Bridge Families (Rectangular d_old != d_new Ready)
# ---------------------------------------------------------------------------


class BaseBridge:
    """Common interface for all 5 bridge families."""

    name: str = "BaseBridge"
    d_in: int = 0
    d_out: int = 0
    fit_count: int = 0

    @classmethod
    def fit(cls, x_src: np.ndarray, y_tgt: np.ndarray, **kwargs: Any) -> "BaseBridge":
        raise NotImplementedError

    def transform(self, x: np.ndarray, *, normalize: bool = True) -> np.ndarray:
        raise NotImplementedError

    def score_matrix(
        self,
        queries_src: np.ndarray,
        docs_tgt: np.ndarray,
        *,
        causal_ema_alpha: Optional[float] = None,
    ) -> np.ndarray:
        """Compute `[n_queries, n_docs]` similarity scores against unmodified target docs."""
        mapped = self.transform(queries_src, normalize=True)
        return mapped @ l2_normalize(docs_tgt).T

    def linear_components(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        """Return `(mu_x, mu_y, linear_map, scale)` for algebraic multi-hop composition."""
        raise NotImplementedError

    @property
    def condition_number(self) -> float:
        _, _, mat, scale = self.linear_components()
        return _singular_diagnostics(mat, scale)[0]

    @property
    def nonzero_condition_number(self) -> float:
        _, _, mat, scale = self.linear_components()
        return _singular_diagnostics(mat, scale)[1]

    @property
    def effective_rank(self) -> int:
        _, _, mat, scale = self.linear_components()
        return _singular_diagnostics(mat, scale)[2]

    @property
    def weights_sha256(self) -> str:
        mu_x, mu_y, mat, scale = self.linear_components()
        return _sha256_arrays(mu_x, mu_y, mat, np.asarray([scale], dtype=np.float64))


@dataclass(frozen=True)
class UncenteredProcrustes(BaseBridge):
    """Uncentered orthogonal / semi-orthogonal Stiefel Procrustes (`W = U V^T`)."""

    W: np.ndarray
    fit_count: int
    d_in: int
    d_out: int
    regime: str = "monotone"
    pre_polar_sigma_min: float = 1.0
    name: str = "UncenteredProcrustes"

    @classmethod
    def fit(
        cls,
        x_src: np.ndarray,
        y_tgt: np.ndarray,
        *,
        fit_idx: Optional[Sequence[int]] = None,
        **_kwargs: Any,
    ) -> "UncenteredProcrustes":
        xs = np.asarray(x_src if fit_idx is None else x_src[np.asarray(fit_idx)], dtype=np.float64)
        yt = np.asarray(y_tgt if fit_idx is None else y_tgt[np.asarray(fit_idx)], dtype=np.float64)
        if xs.shape[0] != yt.shape[0] or xs.shape[0] < 2:
            raise ValueError("UncenteredProcrustes requires >= 2 matched anchor pairs")
        u, _s, vt = np.linalg.svd(xs.T @ yt, full_matrices=False)
        w = u @ vt
        return cls(W=w, fit_count=int(xs.shape[0]), d_in=int(xs.shape[1]), d_out=int(yt.shape[1]))

    def transform(self, x: np.ndarray, *, normalize: bool = True) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        out = arr @ self.W
        return l2_normalize(out) if normalize else out

    def linear_components(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        return (
            np.zeros((1, self.d_in), dtype=np.float64),
            np.zeros((1, self.d_out), dtype=np.float64),
            self.W,
            1.0,
        )


@dataclass(frozen=True)
class CenteredProcrustes(BaseBridge):
    """Centered conformal orthogonal Stiefel Procrustes (`s * (x - mu_x) Q + mu_y`)."""

    mu_x: np.ndarray
    mu_y: np.ndarray
    Q: np.ndarray
    scale: float
    fit_count: int
    d_in: int
    d_out: int
    scale_mode: str = "ambient"
    regime: str = "monotone"
    pre_polar_sigma_min: float = 1.0
    polar_reprojected: bool = False
    name: str = "CenteredProcrustes"

    @classmethod
    def fit(
        cls,
        x_src: np.ndarray,
        y_tgt: np.ndarray,
        *,
        fit_idx: Optional[Sequence[int]] = None,
        scale_mode: str = "ambient",
        rank_rel_tol: Optional[float] = None,
        **_kwargs: Any,
    ) -> "CenteredProcrustes":
        xs = np.asarray(x_src if fit_idx is None else x_src[np.asarray(fit_idx)], dtype=np.float64)
        yt = np.asarray(y_tgt if fit_idx is None else y_tgt[np.asarray(fit_idx)], dtype=np.float64)
        if xs.shape[0] != yt.shape[0] or xs.shape[0] < 2:
            raise ValueError("CenteredProcrustes requires >= 2 matched anchor pairs")
        mu_x = xs.mean(axis=0, keepdims=True)
        mu_y = yt.mean(axis=0, keepdims=True)
        xc = xs - mu_x
        yc = yt - mu_y
        u, sv, vt = np.linalg.svd(xc.T @ yc, full_matrices=False)
        sv0 = float(sv[0]) if sv.size else 0.0
        if rank_rel_tol is not None and sv0 > 1e-12:
            keep = max(1, int(np.sum(sv > float(rank_rel_tol) * sv0)))
            q = u[:, :keep] @ vt[:keep, :]
        else:
            q = u @ vt
        if scale_mode == "unit":
            scale = 1.0
        elif scale_mode == "ambient":
            scale = float(np.linalg.norm(yc)) / max(float(np.linalg.norm(xc)), 1e-12)
        else:
            raise ValueError(f"Unsupported scale_mode={scale_mode!r}")
        return cls(
            mu_x=mu_x,
            mu_y=mu_y,
            Q=q,
            scale=float(scale),
            fit_count=int(xs.shape[0]),
            d_in=int(xs.shape[1]),
            d_out=int(yt.shape[1]),
            scale_mode=scale_mode,
        )

    def transform(self, x: np.ndarray, *, normalize: bool = True) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        out = self.scale * ((arr - self.mu_x) @ self.Q) + self.mu_y
        return l2_normalize(out) if normalize else out

    def linear_components(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        return self.mu_x, self.mu_y, self.Q, self.scale


@dataclass(frozen=True)
class RidgeDriftAdapter(BaseBridge):
    """Centered affine ridge adapter with exact O(n^2 d) Woodbury solve for `n < d`."""

    mu_x: np.ndarray
    mu_y: np.ndarray
    W: np.ndarray
    regularization: float
    fit_count: int
    d_in: int
    d_out: int
    regime: str = "monotone"
    pre_polar_sigma_min: float = 1.0
    name: str = "RidgeDriftAdapter"

    @classmethod
    def fit(
        cls,
        x_src: np.ndarray,
        y_tgt: np.ndarray,
        *,
        fit_idx: Optional[Sequence[int]] = None,
        regularization: float = 1e-2,
        **_kwargs: Any,
    ) -> "RidgeDriftAdapter":
        if regularization < 0.0 or not math.isfinite(regularization):
            raise ValueError("regularization must be finite and >= 0")
        xs = np.asarray(x_src if fit_idx is None else x_src[np.asarray(fit_idx)], dtype=np.float64)
        yt = np.asarray(y_tgt if fit_idx is None else y_tgt[np.asarray(fit_idx)], dtype=np.float64)
        if xs.shape[0] != yt.shape[0] or xs.shape[0] < 2:
            raise ValueError("RidgeDriftAdapter requires >= 2 matched anchor pairs")
        mu_x = xs.mean(axis=0, keepdims=True)
        mu_y = yt.mean(axis=0, keepdims=True)
        xc = xs - mu_x
        yc = yt - mu_y
        n_rows, d_in = xc.shape
        reg = float(regularization)
        if n_rows < d_in and reg > 0.0:
            gram = xc @ xc.T + reg * np.eye(n_rows, dtype=np.float64)
            w = xc.T @ np.linalg.solve(gram, yc)
        else:
            w = np.linalg.solve(xc.T @ xc + reg * np.eye(d_in, dtype=np.float64), xc.T @ yc)
        return cls(
            mu_x=mu_x,
            mu_y=mu_y,
            W=w,
            regularization=reg,
            fit_count=int(n_rows),
            d_in=int(d_in),
            d_out=int(yt.shape[1]),
        )

    def transform(self, x: np.ndarray, *, normalize: bool = True) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        out = (arr - self.mu_x) @ self.W + self.mu_y
        return l2_normalize(out) if normalize else out

    def linear_components(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        return self.mu_x, self.mu_y, self.W, 1.0


@dataclass(frozen=True)
class Rank1QueryResidual(BaseBridge):
    """Orthogonal Procrustes base plus closed-form rank-1 principal residual correction.

    Models systematic directional drift (e.g. prompt-wrapper or anisotropy skew) via:
        `y = (x - mu_x) @ (Q + sigma_1 * u_1 @ v_1^T) + mu_y + b_res`
    where `(u_1, sigma_1, v_1)` is the dominant rank-1 residual mode on `(X_c, Y_c - X_c Q)`.
    """

    mu_x: np.ndarray
    mu_y: np.ndarray
    Q: np.ndarray
    u1: np.ndarray  # [d_in, 1]
    v1: np.ndarray  # [1, d_out]
    alpha: float
    b_res: np.ndarray  # [1, d_out]
    fit_count: int
    d_in: int
    d_out: int
    regime: str = "monotone"
    pre_polar_sigma_min: float = 1.0
    name: str = "Rank1QueryResidual"

    @classmethod
    def fit(
        cls,
        x_src: np.ndarray,
        y_tgt: np.ndarray,
        *,
        fit_idx: Optional[Sequence[int]] = None,
        residual_damping: float = 1e-2,
        **_kwargs: Any,
    ) -> "Rank1QueryResidual":
        xs = np.asarray(x_src if fit_idx is None else x_src[np.asarray(fit_idx)], dtype=np.float64)
        yt = np.asarray(y_tgt if fit_idx is None else y_tgt[np.asarray(fit_idx)], dtype=np.float64)
        if xs.shape[0] != yt.shape[0] or xs.shape[0] < 2:
            raise ValueError("Rank1QueryResidual requires >= 2 matched anchor pairs")
        mu_x = xs.mean(axis=0, keepdims=True)
        mu_y = yt.mean(axis=0, keepdims=True)
        xc = xs - mu_x
        yc = yt - mu_y
        u_q, _sv_q, vt_q = np.linalg.svd(xc.T @ yc, full_matrices=False)
        q = u_q @ vt_q

        # Fit optimal rank-1 linear residual on centered residuals R = yc - xc @ Q
        residual = yc - xc @ q
        b_res = residual.mean(axis=0, keepdims=True)
        res_centered = residual - b_res
        cross_res = xc.T @ res_centered
        u_r, sv_r, vt_r = np.linalg.svd(cross_res, full_matrices=False)
        if sv_r.size > 0 and sv_r[0] > 1e-12:
            u1 = u_r[:, :1]
            v1 = vt_r[:1, :]
            proj_x = xc @ u1  # [n, 1]
            proj_r = res_centered @ v1.T  # [n, 1]
            denom = float(np.sum(proj_x * proj_x)) + float(residual_damping)
            alpha = float(np.sum(proj_x * proj_r)) / max(denom, 1e-12)
        else:
            u1 = np.zeros((xs.shape[1], 1), dtype=np.float64)
            v1 = np.zeros((1, yt.shape[1]), dtype=np.float64)
            alpha = 0.0

        return cls(
            mu_x=mu_x,
            mu_y=mu_y,
            Q=q,
            u1=u1,
            v1=v1,
            alpha=float(alpha),
            b_res=b_res,
            fit_count=int(xs.shape[0]),
            d_in=int(xs.shape[1]),
            d_out=int(yt.shape[1]),
        )

    @property
    def effective_matrix(self) -> np.ndarray:
        return self.Q + self.alpha * (self.u1 @ self.v1)

    def transform(self, x: np.ndarray, *, normalize: bool = True) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        xc = arr - self.mu_x
        out = xc @ self.Q + self.alpha * ((xc @ self.u1) @ self.v1) + self.mu_y + self.b_res
        return l2_normalize(out) if normalize else out

    def linear_components(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        return self.mu_x, self.mu_y + self.b_res, self.effective_matrix, 1.0


@dataclass(frozen=True)
class ZeroMuPolarBridge(BaseBridge):
    """ZeroMu Subtractive Centroid Removal + Stiefel Polar Re-Projection Bridge.

    Scorer modes:
      - `dual_adap`: Unlabeled resident-doc mean `mu_d` (4-byte float32 centered norm
        sidecar) + causal chronological query EMA `mu_q` (excluding query `t`).
      - `dual_fit`: Fitted anchor `mu_x` and `mu_y` with exact 1-float identity.
      - `query_only`: True zero-write `<norm2((q - mu_q) Q), d_j>` on unmodified cosine index.
    """

    mu_x: np.ndarray
    mu_y: np.ndarray
    Q: np.ndarray
    fit_count: int
    d_in: int
    d_out: int
    scoring_mode: str = "dual_adap"
    ema_alpha: float = DEFAULT_EMA_ALPHA
    regime: str = "monotone"
    pre_polar_sigma_min: float = 1.0
    polar_reprojected: bool = False
    name: str = "ZeroMuPolarBridge"

    @classmethod
    def fit(
        cls,
        x_src: np.ndarray,
        y_tgt: np.ndarray,
        *,
        fit_idx: Optional[Sequence[int]] = None,
        scoring_mode: str = "dual_adap",
        ema_alpha: float = DEFAULT_EMA_ALPHA,
        rank_rel_tol: Optional[float] = None,
        **_kwargs: Any,
    ) -> "ZeroMuPolarBridge":
        if scoring_mode not in {"dual_adap", "dual_fit", "query_only"}:
            raise ValueError(f"Unsupported ZeroMu scoring_mode={scoring_mode!r}")
        xs = np.asarray(x_src if fit_idx is None else x_src[np.asarray(fit_idx)], dtype=np.float64)
        yt = np.asarray(y_tgt if fit_idx is None else y_tgt[np.asarray(fit_idx)], dtype=np.float64)
        if xs.shape[0] != yt.shape[0] or xs.shape[0] < 2:
            raise ValueError("ZeroMuPolarBridge requires >= 2 matched anchor pairs")
        mu_x = xs.mean(axis=0, keepdims=True)
        mu_y = yt.mean(axis=0, keepdims=True)
        xc = xs - mu_x
        yc = yt - mu_y
        u, sv, vt = np.linalg.svd(xc.T @ yc, full_matrices=False)
        sv0 = float(sv[0]) if sv.size else 0.0
        if rank_rel_tol is not None and sv0 > 1e-12:
            keep = max(1, int(np.sum(sv > float(rank_rel_tol) * sv0)))
            q = u[:, :keep] @ vt[:keep, :]
        else:
            q = u @ vt
        return cls(
            mu_x=mu_x,
            mu_y=mu_y,
            Q=q,
            fit_count=int(xs.shape[0]),
            d_in=int(xs.shape[1]),
            d_out=int(yt.shape[1]),
            scoring_mode=scoring_mode,
            ema_alpha=float(ema_alpha),
        )

    @staticmethod
    def build_float32_sidecar(docs_raw: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Pack exact 4-byte float32 centered norm per document + shared `mu_d`."""
        docs = l2_normalize(docs_raw).astype(np.float32)
        mu_d = docs.mean(axis=0, keepdims=True).astype(np.float32)
        norms = np.maximum(np.linalg.norm(docs - mu_d, axis=1).astype(np.float32), np.float32(1e-12))
        return mu_d, norms

    def causal_q_hats(
        self,
        queries_src: np.ndarray,
        *,
        ema_alpha: Optional[float] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Project queries chronologically; `mu_q` for query `t` strictly excludes query `t`."""
        queries = np.asarray(queries_src, dtype=np.float64)
        if queries.ndim == 1:
            queries = queries.reshape(1, -1)
        alpha = self.ema_alpha if ema_alpha is None else float(ema_alpha)
        mu_q = self.mu_x.reshape(1, -1).copy()
        hats = np.empty((queries.shape[0], self.d_out), dtype=np.float64)
        mu_before = np.empty((queries.shape[0], self.d_in), dtype=np.float64)
        for idx in range(queries.shape[0]):
            q_row = queries[idx : idx + 1]
            mu_before[idx] = mu_q[0]
            hats[idx] = l2_normalize((q_row - mu_q) @ self.Q)[0]
            if alpha > 0.0:
                mu_q = (1.0 - alpha) * mu_q + alpha * q_row
        return hats, mu_before

    def transform(self, x: np.ndarray, *, normalize: bool = True) -> np.ndarray:
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        out = (arr - self.mu_x) @ self.Q
        return l2_normalize(out) if normalize else out

    def score_matrix(
        self,
        queries_src: np.ndarray,
        docs_tgt: np.ndarray,
        *,
        causal_ema_alpha: Optional[float] = None,
        dev_docs: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute exact ZeroMu cosine scores on unmodified raw target vectors.

        When ``dev_docs`` is supplied, ``mu_d`` is estimated only on that disjoint
        development corpus. Overlap with the documents being scored is refused.
        """
        docs_norm = l2_normalize(docs_tgt)
        if dev_docs is not None:
            dev_arr = np.asarray(dev_docs, dtype=np.float64)
            tgt_arr = np.asarray(docs_tgt, dtype=np.float64)
            if dev_arr.ndim == 1:
                dev_arr = dev_arr.reshape(1, -1)
            if tgt_arr.ndim == 1:
                tgt_arr = tgt_arr.reshape(1, -1)
            if dev_arr.shape[1] != tgt_arr.shape[1]:
                raise ValueError(
                    f"ZeroMu dev_docs width {dev_arr.shape[1]} does not match scored documents width {tgt_arr.shape[1]}"
                )
            if _matrices_share_rows(dev_arr, tgt_arr):
                raise ValueError("ZeroMu dev_docs overlap the documents being scored")
            mu_from_dev = l2_normalize(dev_arr).mean(axis=0, keepdims=True)
        else:
            mu_from_dev = None
        if self.scoring_mode == "dual_adap":
            q_hats, _ = self.causal_q_hats(queries_src, ema_alpha=causal_ema_alpha)
            mu_d = docs_norm.mean(axis=0, keepdims=True) if mu_from_dev is None else mu_from_dev
        elif self.scoring_mode == "dual_fit":
            q_hats = self.transform(queries_src, normalize=True)
            mu_d = self.mu_y if mu_from_dev is None else mu_from_dev
        else:
            q_hats = self.transform(queries_src, normalize=True)
            return q_hats @ docs_norm.T

        mu_vec = mu_d.reshape(-1)
        mu_norm2 = float(np.dot(mu_vec, mu_vec))
        doc_den = np.sqrt(np.maximum(1.0 + mu_norm2 - 2.0 * (docs_norm @ mu_vec), 1e-12))
        return (q_hats @ docs_norm.T - (q_hats @ mu_vec)[:, None]) / doc_den[None, :]

    def linear_components(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
        return self.mu_x, self.mu_y, self.Q, 1.0


BRIDGE_REGISTRY: Dict[str, Type[BaseBridge]] = {
    "UncenteredProcrustes": UncenteredProcrustes,
    "CenteredProcrustes": CenteredProcrustes,
    "RidgeDriftAdapter": RidgeDriftAdapter,
    "Rank1QueryResidual": Rank1QueryResidual,
    "ZeroMuPolarBridge": ZeroMuPolarBridge,
}
BRIDGE_NAMES: Tuple[str, ...] = tuple(BRIDGE_REGISTRY.keys())


# ---------------------------------------------------------------------------
# 2. Three Dual-Shard Merge Methods (Never Denying CSLS to Any Bridge)
# ---------------------------------------------------------------------------


def _csls_shard(scores: np.ndarray, ref_scores: np.ndarray, csls_k: int) -> np.ndarray:
    """Per-shard CSLS `2 * s - r_d - r_q` using DEV reference queries for `r_d`."""
    mat = np.asarray(scores, dtype=np.float64)
    ref = np.asarray(ref_scores, dtype=np.float64)
    if mat.size == 0 or mat.shape[1] == 0:
        return mat.copy()
    doc_k = max(1, min(int(csls_k), int(ref.shape[0])))
    if doc_k < ref.shape[0]:
        top_docs = np.partition(ref, ref.shape[0] - doc_k, axis=0)[-doc_k:, :]
    else:
        top_docs = ref
    r_d = top_docs.mean(axis=0)

    query_k = max(1, min(int(csls_k), int(mat.shape[1])))
    if query_k < mat.shape[1]:
        top_q = np.partition(mat, mat.shape[1] - query_k, axis=1)[:, -query_k:]
    else:
        top_q = mat
    r_q = top_q.mean(axis=1)
    return 2.0 * mat - r_d.reshape(1, -1) - r_q.reshape(-1, 1)


class BaseMerge:
    """Common interface for dual-shard merge strategies."""

    name: str = "BaseMerge"

    def merge(
        self,
        bridged_scores: np.ndarray,
        native_scores: np.ndarray,
        backfill_mask: np.ndarray,
        *,
        bridged_ref: Optional[np.ndarray] = None,
        native_ref: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        raise NotImplementedError


@dataclass(frozen=True)
class RawCosineMerge(BaseMerge):
    """Uncalibrated raw cosine merge across bridged and native shards."""

    name: str = "RawCosineMerge"

    def merge(
        self,
        bridged_scores: np.ndarray,
        native_scores: np.ndarray,
        backfill_mask: np.ndarray,
        *,
        bridged_ref: Optional[np.ndarray] = None,
        native_ref: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        bridged = np.asarray(bridged_scores, dtype=np.float64)
        native = np.asarray(native_scores, dtype=np.float64)
        mask = np.asarray(backfill_mask, dtype=bool)
        if bridged.shape != native.shape or mask.shape != (bridged.shape[1],):
            raise ValueError("Score matrices and backfill_mask shapes must align")
        out = bridged.copy()
        if mask.any():
            out[:, mask] = native[:, mask]
        return out


@dataclass(frozen=True)
class CalibratedZScoreMerge(BaseMerge):
    """DEV-calibrated per-shard scalar z-score merge `(s - mu_dev) / max(sd_dev, eps)`."""

    eps: float = 1e-6
    name: str = "CalibratedZScoreMerge"

    def merge(
        self,
        bridged_scores: np.ndarray,
        native_scores: np.ndarray,
        backfill_mask: np.ndarray,
        *,
        bridged_ref: Optional[np.ndarray] = None,
        native_ref: Optional[np.ndarray] = None,
        strict_disjoint_dev: bool = False,
    ) -> np.ndarray:
        bridged = np.asarray(bridged_scores, dtype=np.float64)
        native = np.asarray(native_scores, dtype=np.float64)
        mask = np.asarray(backfill_mask, dtype=bool)
        if bridged.shape != native.shape or mask.shape != (bridged.shape[1],):
            raise ValueError("Score matrices and backfill_mask shapes must align")
        if strict_disjoint_dev and (bridged_ref is None or native_ref is None):
            raise ValueError("CalibratedZScoreMerge requires disjoint DEV references when strict_disjoint_dev is set")
        if strict_disjoint_dev and (~mask).any() and bridged_ref is not None and _matrices_share_rows(bridged_ref, bridged):
            raise ValueError("CalibratedZScoreMerge DEV reference overlaps EVAL bridged scores")
        if strict_disjoint_dev and mask.any() and native_ref is not None and _matrices_share_rows(native_ref, native):
            raise ValueError("CalibratedZScoreMerge DEV reference overlaps EVAL native scores")
        b_ref = bridged if bridged_ref is None else np.asarray(bridged_ref, dtype=np.float64)
        n_ref = native if native_ref is None else np.asarray(native_ref, dtype=np.float64)
        out = np.empty_like(bridged)
        if (~mask).any():
            sub_ref = b_ref[:, ~mask]
            mu_b = float(sub_ref.mean()) if sub_ref.size else 0.0
            sd_b = max(float(sub_ref.std()), self.eps) if sub_ref.size else 1.0
            out[:, ~mask] = (bridged[:, ~mask] - mu_b) / sd_b
        if mask.any():
            sub_ref = n_ref[:, mask]
            mu_n = float(sub_ref.mean()) if sub_ref.size else 0.0
            sd_n = max(float(sub_ref.std()), self.eps) if sub_ref.size else 1.0
            out[:, mask] = (native[:, mask] - mu_n) / sd_n
        return out


@dataclass(frozen=True)
class CSLSMerge(BaseMerge):
    """Per-shard local CSLS hubness-calibrated merge (`2*s - r_d - r_q`)."""

    csls_k: int = DEFAULT_CSLS_K
    name: str = "CSLSMerge"

    def merge(
        self,
        bridged_scores: np.ndarray,
        native_scores: np.ndarray,
        backfill_mask: np.ndarray,
        *,
        bridged_ref: Optional[np.ndarray] = None,
        native_ref: Optional[np.ndarray] = None,
        strict_disjoint_dev: bool = False,
    ) -> np.ndarray:
        bridged = np.asarray(bridged_scores, dtype=np.float64)
        native = np.asarray(native_scores, dtype=np.float64)
        mask = np.asarray(backfill_mask, dtype=bool)
        if bridged.shape != native.shape or mask.shape != (bridged.shape[1],):
            raise ValueError("Score matrices and backfill_mask shapes must align")
        if strict_disjoint_dev and (bridged_ref is None or native_ref is None):
            raise ValueError("CSLSMerge requires disjoint DEV references when strict_disjoint_dev is set")
        if strict_disjoint_dev and (~mask).any() and bridged_ref is not None and _matrices_share_rows(bridged_ref, bridged):
            raise ValueError("CSLSMerge DEV reference overlaps EVAL bridged scores")
        if strict_disjoint_dev and mask.any() and native_ref is not None and _matrices_share_rows(native_ref, native):
            raise ValueError("CSLSMerge DEV reference overlaps EVAL native scores")
        b_ref = bridged if bridged_ref is None else np.asarray(bridged_ref, dtype=np.float64)
        n_ref = native if native_ref is None else np.asarray(native_ref, dtype=np.float64)
        out = np.empty_like(bridged)
        if (~mask).any():
            out[:, ~mask] = _csls_shard(bridged[:, ~mask], b_ref[:, ~mask], self.csls_k)
        if mask.any():
            out[:, mask] = _csls_shard(native[:, mask], n_ref[:, mask], self.csls_k)
        return out


MERGE_REGISTRY: Dict[str, Type[BaseMerge]] = {
    "RawCosineMerge": RawCosineMerge,
    "CalibratedZScoreMerge": CalibratedZScoreMerge,
    "CSLSMerge": CSLSMerge,
}
MERGE_NAMES: Tuple[str, ...] = tuple(MERGE_REGISTRY.keys())


# ---------------------------------------------------------------------------
# 3. Retrieval Scoring & Full 5x3 Factorial Runner
# ---------------------------------------------------------------------------


def _matrix_row_keys(arr: np.ndarray) -> set:
    mat = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
    if mat.ndim == 1:
        mat = mat.reshape(1, -1)
    if mat.size == 0:
        return set()
    return {row.tobytes() for row in mat}


def _matrices_share_rows(left: np.ndarray, right: np.ndarray) -> bool:
    if left is right:
        return True
    return bool(_matrix_row_keys(left) & _matrix_row_keys(right))


def evaluate_score_matrix(
    scores: np.ndarray,
    query_ids: Sequence[str],
    doc_ids: Sequence[str],
    qrels: Mapping[str, Mapping[str, float]],
    *,
    k: int = 10,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """Compute per-query complete-qrels `nDCG@k` and `Recall@k` under canonical tie-breaking."""
    canonical_docs = [canonicalize_id(d) for d in doc_ids]
    ndcg_map: Dict[str, float] = {}
    recall_map: Dict[str, float] = {}
    for row_idx, raw_qid in enumerate(query_ids):
        qid = canonicalize_id(raw_qid)
        rel = qrels.get(qid, qrels.get(raw_qid, {}))
        pos = {canonicalize_id(did): float(val) for did, val in rel.items() if float(val) > 0.0}
        if not pos:
            continue
        ordered = _canonical_topk(scores[row_idx], canonical_docs, k)
        ranked = [canonical_docs[j] for j in ordered]
        ndcg_map[qid] = float(ndcg_at_k(ranked, pos, k=k, binary=False))
        recall_map[qid] = float(recall_at_k(ranked, pos, k=k))
    return ndcg_map, recall_map


@dataclass(frozen=True)
class FactorialCellResult:
    domain: str
    bridge_name: str
    merge_name: str
    mean_ndcg_at_10: float
    mean_recall_at_10: float
    kappa: float
    nonzero_kappa: float
    effective_rank: int
    fit_count: int
    backfill_fraction: float
    weights_sha256: str
    per_query_ndcg: Dict[str, float]
    per_query_recall: Dict[str, float]


def run_factorial_5x3(
    *,
    domain: str,
    src_anchors: np.ndarray,
    tgt_anchors: np.ndarray,
    eval_queries_src: np.ndarray,
    eval_queries_tgt: np.ndarray,
    docs_old: np.ndarray,
    docs_new: np.ndarray,
    query_ids: Sequence[str],
    doc_ids: Sequence[str],
    qrels: Mapping[str, Mapping[str, float]],
    dev_queries_src: Optional[np.ndarray] = None,
    dev_queries_tgt: Optional[np.ndarray] = None,
    dev_docs: Optional[np.ndarray] = None,
    backfill_mask: Optional[np.ndarray] = None,
    k: int = 10,
    csls_k: int = DEFAULT_CSLS_K,
    ridge_regularization: float = 1e-2,
    zeromu_mode: str = "dual_adap",
    bridge_names: Sequence[str] = BRIDGE_NAMES,
    merge_names: Sequence[str] = MERGE_NAMES,
) -> Dict[Tuple[str, str], FactorialCellResult]:
    """Run the complete `len(bridge_names) x len(merge_names)` (default 5x3=15) factorial.

    Strictly applies EVERY merge method (including `CSLSMerge`) to EVERY bridge
    family, eliminating asymmetric calibration bias.
    """
    n_docs = len(doc_ids)
    mask = np.zeros(n_docs, dtype=bool) if backfill_mask is None else np.asarray(backfill_mask, dtype=bool)
    if mask.shape != (n_docs,):
        raise ValueError("backfill_mask length must equal len(doc_ids)")

    dev_supplied = dev_queries_src is not None or dev_queries_tgt is not None or dev_docs is not None
    if dev_supplied and (dev_queries_src is None or dev_queries_tgt is None):
        raise ValueError("DEV calibration requires both dev_queries_src and dev_queries_tgt")
    dev_src = eval_queries_src if dev_queries_src is None else dev_queries_src
    dev_tgt = eval_queries_tgt if dev_queries_tgt is None else dev_queries_tgt
    if dev_queries_src is not None and (
        _matrices_share_rows(dev_src, eval_queries_src) or _matrices_share_rows(dev_tgt, eval_queries_tgt)
    ):
        raise ValueError("DEV calibration queries overlap EVAL queries")
    if dev_docs is not None and _matrices_share_rows(dev_docs, docs_old):
        raise ValueError("DEV calibration documents overlap the documents being scored")

    docs_new_norm = l2_normalize(docs_new)
    native_eval_scores = l2_normalize(eval_queries_tgt) @ docs_new_norm.T
    native_dev_scores = l2_normalize(dev_tgt) @ docs_new_norm.T

    mergers: Dict[str, BaseMerge] = {}
    for m_name in merge_names:
        if m_name not in MERGE_REGISTRY:
            raise ValueError(f"Unknown merge method: {m_name!r}")
        mergers[m_name] = CSLSMerge(csls_k=csls_k) if m_name == "CSLSMerge" else MERGE_REGISTRY[m_name]()

    results: Dict[Tuple[str, str], FactorialCellResult] = {}
    bf_frac = float(np.mean(mask)) if n_docs > 0 else 0.0

    for b_name in bridge_names:
        if b_name not in BRIDGE_REGISTRY:
            raise ValueError(f"Unknown bridge method: {b_name!r}")
        bridge_cls = BRIDGE_REGISTRY[b_name]
        fit_kwargs: Dict[str, Any] = {}
        if b_name == "RidgeDriftAdapter":
            fit_kwargs["regularization"] = ridge_regularization
        elif b_name == "ZeroMuPolarBridge":
            fit_kwargs["scoring_mode"] = zeromu_mode
        bridge = bridge_cls.fit(src_anchors, tgt_anchors, **fit_kwargs)

        if b_name == "ZeroMuPolarBridge" and dev_docs is not None:
            bridged_eval = bridge.score_matrix(eval_queries_src, docs_old, dev_docs=dev_docs)
            bridged_dev = bridge.score_matrix(dev_src, docs_old, dev_docs=dev_docs)
        else:
            bridged_eval = bridge.score_matrix(eval_queries_src, docs_old)
            bridged_dev = bridge.score_matrix(dev_src, docs_old)

        for m_name, merger in mergers.items():
            merge_kwargs: Dict[str, Any] = {
                "bridged_ref": bridged_dev,
                "native_ref": native_dev_scores,
            }
            if dev_supplied and m_name in {"CSLSMerge", "CalibratedZScoreMerge"}:
                merge_kwargs["strict_disjoint_dev"] = True
            fused = merger.merge(
                bridged_eval,
                native_eval_scores,
                mask,
                **merge_kwargs,
            )
            pq_ndcg, pq_recall = evaluate_score_matrix(fused, query_ids, doc_ids, qrels, k=k)
            mean_ndcg = float(np.mean(list(pq_ndcg.values()))) if pq_ndcg else 0.0
            mean_recall = float(np.mean(list(pq_recall.values()))) if pq_recall else 0.0
            results[(b_name, m_name)] = FactorialCellResult(
                domain=domain,
                bridge_name=b_name,
                merge_name=m_name,
                mean_ndcg_at_10=mean_ndcg,
                mean_recall_at_10=mean_recall,
                kappa=bridge.condition_number,
                nonzero_kappa=bridge.nonzero_condition_number,
                effective_rank=bridge.effective_rank,
                fit_count=bridge.fit_count,
                backfill_fraction=bf_frac,
                weights_sha256=bridge.weights_sha256,
                per_query_ndcg=pq_ndcg,
                per_query_recall=pq_recall,
            )
    return results


# ---------------------------------------------------------------------------
# 4. Twenty Deterministic Seeded 500-Anchor Draws & Within/Cross-Domain Variance
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AnchorDraw:
    domain: str
    draw_index: int
    seed: int
    requested_anchors: int
    actual_anchors: int
    indices: np.ndarray
    manifest_sha256: str


def sample_anchor_draws(
    domain: str,
    doc_ids: Sequence[str],
    *,
    n_draws: int = DEFAULT_ANCHOR_DRAWS,
    anchors_per_draw: int = DEFAULT_ANCHORS_PER_DRAW,
    base_seed: int = 1600,
    excluded_doc_ids: Optional[Sequence[str]] = None,
) -> List[AnchorDraw]:
    """Generate `n_draws` (default 20) deterministic seeded 500-anchor draws per domain."""
    if n_draws <= 0 or anchors_per_draw <= 0:
        raise ValueError("n_draws and anchors_per_draw must be positive")
    excluded = {canonicalize_id(d) for d in (excluded_doc_ids or ())}
    eligible = [idx for idx, did in enumerate(doc_ids) if canonicalize_id(did) not in excluded]
    if len(eligible) < 2:
        raise ValueError("At least 2 eligible anchor documents are required")
    actual = min(int(anchors_per_draw), len(eligible))
    draws: List[AnchorDraw] = []
    for draw_idx in range(int(n_draws)):
        seed_material = f"{domain}:{base_seed}:{draw_idx}".encode("utf-8")
        derived_seed = int.from_bytes(hashlib.sha256(seed_material).digest()[:4], "big")
        rng = np.random.default_rng(derived_seed)
        picked = np.sort(rng.choice(np.asarray(eligible, dtype=np.int64), size=actual, replace=False))
        manifest = hashlib.sha256(
            json.dumps(
                {
                    "domain": domain,
                    "draw_index": draw_idx,
                    "seed": derived_seed,
                    "doc_ids": [canonicalize_id(doc_ids[int(i)]) for i in picked],
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        draws.append(
            AnchorDraw(
                domain=domain,
                draw_index=draw_idx,
                seed=derived_seed,
                requested_anchors=int(anchors_per_draw),
                actual_anchors=actual,
                indices=picked,
                manifest_sha256=manifest,
            )
        )
    return draws


@dataclass(frozen=True)
class DomainAnchorVarianceSummary:
    domain: str
    n_draws: int
    requested_anchors: int
    actual_anchors: int
    within_domain_mean: float
    within_domain_anchor_sd: float
    min_score: float
    max_score: float
    draw_scores: Tuple[float, ...]


@dataclass(frozen=True)
class CrossDomainVarianceReport:
    cross_domain_n: int
    cross_domain_mean: float
    cross_domain_sd: float
    mean_within_domain_anchor_sd: float
    per_domain: Dict[str, DomainAnchorVarianceSummary]


def summarize_anchor_draw_variance(
    domain_draw_scores: Mapping[str, Sequence[float]],
    *,
    requested_anchors: int = DEFAULT_ANCHORS_PER_DRAW,
    actual_anchors_by_domain: Optional[Mapping[str, int]] = None,
) -> CrossDomainVarianceReport:
    """Strictly separate within-domain anchor-resample SD from cross-domain `N`."""
    if not domain_draw_scores:
        raise ValueError("domain_draw_scores must be non-empty")
    per_domain: Dict[str, DomainAnchorVarianceSummary] = {}
    domain_means: List[float] = []
    within_sds: List[float] = []
    for dom in sorted(domain_draw_scores.keys()):
        vals = np.asarray(list(domain_draw_scores[dom]), dtype=np.float64)
        if vals.size == 0:
            raise ValueError(f"Domain {dom!r} has no draw scores")
        ddof = 1 if vals.size > 1 else 0
        mean_val = float(np.mean(vals))
        sd_val = float(np.std(vals, ddof=ddof))
        actual = (
            int(actual_anchors_by_domain[dom])
            if actual_anchors_by_domain and dom in actual_anchors_by_domain
            else int(requested_anchors)
        )
        per_domain[dom] = DomainAnchorVarianceSummary(
            domain=dom,
            n_draws=int(vals.size),
            requested_anchors=int(requested_anchors),
            actual_anchors=actual,
            within_domain_mean=mean_val,
            within_domain_anchor_sd=sd_val,
            min_score=float(np.min(vals)),
            max_score=float(np.max(vals)),
            draw_scores=tuple(float(v) for v in vals),
        )
        domain_means.append(mean_val)
        within_sds.append(sd_val)

    means_arr = np.asarray(domain_means, dtype=np.float64)
    cross_ddof = 1 if means_arr.size > 1 else 0
    return CrossDomainVarianceReport(
        cross_domain_n=int(means_arr.size),
        cross_domain_mean=float(np.mean(means_arr)),
        cross_domain_sd=float(np.std(means_arr, ddof=cross_ddof)),
        mean_within_domain_anchor_sd=float(np.mean(within_sds)),
        per_domain=per_domain,
    )


# ---------------------------------------------------------------------------
# 5. 5-Encoder / 4-Transition Sequential Composition (`v1->v2->v3->v4->v5`) vs Direct `v1->v5`
# ---------------------------------------------------------------------------


def compose_bridge_chain(
    bridges: Sequence[BaseBridge],
    *,
    polar_reproject: bool = False,
    rank_rel_tol: float = 1e-6,
) -> BaseBridge:
    """Compose a directed sequence of bridges `[B_{v1->v2}, ..., B_{v4->v5}]`.

    Supports all 5 bridge classes; when `polar_reproject=True` (default for
    `ZeroMuPolarBridge` multi-hop chains), projects the composed operator back
    onto the Stiefel manifold `U_r V_r^T` so `nonzero_kappa == 1.00`, logging
    `pre_polar_sigma_min` and `regime` (`monotone | valley | peak`).
    """
    if not bridges:
        raise ValueError("bridges sequence must be non-empty")
    dims = [int(bridges[0].d_in)] + [int(b.d_out) for b in bridges]
    regime = classify_dimension_chain_regime(dims)

    mu_x, mu_y, mat, scale = bridges[0].linear_components()
    min_fit = bridges[0].fit_count
    for nxt in bridges[1:]:
        n_mu_x, n_mu_y, n_mat, n_scale = nxt.linear_components()
        mat = mat @ n_mat
        scale = float(scale * n_scale)
        mu_y = n_scale * ((mu_y - n_mu_x) @ n_mat) + n_mu_y
        min_fit = min(min_fit, nxt.fit_count)

    u, sv, vt = np.linalg.svd(mat, full_matrices=False)
    sv0 = float(sv[0]) if sv.size > 0 else 0.0
    keep = int(np.sum(sv > float(rank_rel_tol) * sv0)) if sv0 > 1e-12 else 0
    pre_sigma_min = float(sv[keep - 1]) if keep > 0 else 0.0
    mat_final = mat
    reprojected = False
    if polar_reproject:
        if keep <= 0:
            raise ValueError("Cannot polar-reproject rank-0 composed bridge")
        mat_final = u[:, :keep] @ vt[:keep, :]
        reprojected = True

    first = bridges[0]
    d_in = int(bridges[0].d_in)
    d_out = int(bridges[-1].d_out)
    if isinstance(first, ZeroMuPolarBridge):
        return ZeroMuPolarBridge(
            mu_x=mu_x,
            mu_y=mu_y,
            Q=mat_final,
            fit_count=min_fit,
            d_in=d_in,
            d_out=d_out,
            scoring_mode=first.scoring_mode,
            ema_alpha=first.ema_alpha,
            regime=regime,
            pre_polar_sigma_min=pre_sigma_min,
            polar_reprojected=reprojected,
        )
    if isinstance(first, CenteredProcrustes):
        return CenteredProcrustes(
            mu_x=mu_x,
            mu_y=mu_y,
            Q=mat_final,
            scale=1.0 if reprojected else scale,
            fit_count=min_fit,
            d_in=d_in,
            d_out=d_out,
            scale_mode=first.scale_mode,
            regime=regime,
            pre_polar_sigma_min=pre_sigma_min,
            polar_reprojected=reprojected,
        )
    if isinstance(first, UncenteredProcrustes):
        return UncenteredProcrustes(
            W=mat_final,
            fit_count=min_fit,
            d_in=d_in,
            d_out=d_out,
            regime=regime,
            pre_polar_sigma_min=pre_sigma_min,
        )
    if isinstance(first, Rank1QueryResidual):
        return Rank1QueryResidual(
            mu_x=mu_x,
            mu_y=mu_y,
            Q=mat_final,
            u1=np.zeros((d_in, 1), dtype=np.float64),
            v1=np.zeros((1, d_out), dtype=np.float64),
            alpha=0.0,
            b_res=np.zeros((1, d_out), dtype=np.float64),
            fit_count=min_fit,
            d_in=d_in,
            d_out=d_out,
            regime=regime,
            pre_polar_sigma_min=pre_sigma_min,
        )
    return RidgeDriftAdapter(
        mu_x=mu_x,
        mu_y=mu_y,
        W=mat_final * scale,
        regularization=getattr(first, "regularization", 1e-2),
        fit_count=min_fit,
        d_in=d_in,
        d_out=d_out,
        regime=regime,
        pre_polar_sigma_min=pre_sigma_min,
    )


@dataclass(frozen=True)
class MultiHopTransitionMetrics:
    hop_number: int
    transition_label: str
    dims_path: Tuple[int, ...]
    regime: str
    composed_kappa: float
    composed_nonzero_kappa: float
    composed_pre_polar_sigma_min: float
    composed_ndcg_at_10: float
    composed_recall_at_10: float
    direct_kappa: float
    direct_nonzero_kappa: float
    direct_ndcg_at_10: float
    direct_recall_at_10: float
    composition_gap_ndcg: float
    composition_gap_recall: float


@dataclass(frozen=True)
class MultiHopChainReport:
    bridge_name: str
    merge_name: str
    polar_reproject: bool
    encoder_names: Tuple[str, ...]
    transitions: Tuple[MultiHopTransitionMetrics, ...]
    final_composed_ndcg_at_10: float
    final_direct_ndcg_at_10: float
    final_composition_gap_ndcg: float


def _resolve_hop_dev_docs(
    *,
    bridge_name: str,
    hop_idx: int,
    curr_docs: np.ndarray,
    dev_docs: Optional[np.ndarray],
    dev_docs_seq: Optional[Sequence[Optional[np.ndarray]]],
) -> Optional[np.ndarray]:
    """Pick the disjoint DEV corpus for one 4-hop target encoder.

    ``dev_docs_seq`` wins per hop when both are supplied. A width mismatch on
    ZeroMu is an error. Other bridges ignore a mismatched single matrix because
    they do not estimate ``mu_d``.
    """
    target_width = int(curr_docs.shape[1])
    candidate: Optional[np.ndarray] = None
    if dev_docs_seq is not None:
        raw = dev_docs_seq[hop_idx]
        if raw is None:
            return None
        candidate = np.asarray(raw, dtype=np.float64)
    elif dev_docs is not None:
        candidate = np.asarray(dev_docs, dtype=np.float64)
    else:
        return None
    if candidate.ndim == 1:
        candidate = candidate.reshape(1, -1)
    if int(candidate.shape[1]) != target_width:
        if bridge_name == "ZeroMuPolarBridge":
            raise ValueError(
                f"4-hop DEV documents width {int(candidate.shape[1])} does not match hop {hop_idx} width {target_width}"
            )
        return None
    if _matrices_share_rows(candidate, curr_docs):
        raise ValueError("4-hop DEV documents overlap the documents being scored")
    return candidate


def evaluate_4hop_chain(
    *,
    bridge_name: str,
    encoder_names: Sequence[str],
    doc_embeddings_seq: Sequence[np.ndarray],
    query_embeddings_seq: Sequence[np.ndarray],
    anchor_indices: Sequence[int],
    query_ids: Sequence[str],
    doc_ids: Sequence[str],
    qrels: Mapping[str, Mapping[str, float]],
    merge_name: str = "CSLSMerge",
    csls_k: int = DEFAULT_CSLS_K,
    polar_reproject: Optional[bool] = None,
    dev_query_embeddings_seq: Optional[Sequence[np.ndarray]] = None,
    dev_docs: Optional[np.ndarray] = None,
    dev_docs_seq: Optional[Sequence[Optional[np.ndarray]]] = None,
    k: int = 10,
) -> MultiHopChainReport:
    """Evaluate sequential `v1 -> v2 -> v3 -> v4 -> v5` composition vs direct `v1 -> v_t` fit.

    At every hop `t = 1..m-1`, fits the stepwise bridge `v_{t-1} -> v_t`, composes
    `v1 -> ... -> v_t`, fits the direct `v1 -> v_t` bridge on the same anchors, and
    measures both spectral conditioning (`kappa`, `nonzero_kappa`, `pre_polar_sigma_min`)
    and retrieval quality (`nDCG@10`, `Recall@10`).
    """
    n_enc = len(doc_embeddings_seq)
    if n_enc < 2 or len(query_embeddings_seq) != n_enc or len(encoder_names) != n_enc:
        raise ValueError("encoder_names, doc_embeddings_seq, and query_embeddings_seq must have equal length >= 2")
    if bridge_name not in BRIDGE_REGISTRY:
        raise ValueError(f"Unknown bridge_name={bridge_name!r}")
    if merge_name not in MERGE_REGISTRY:
        raise ValueError(f"Unknown merge_name={merge_name!r}")
    if dev_query_embeddings_seq is not None and len(dev_query_embeddings_seq) != n_enc:
        raise ValueError("dev_query_embeddings_seq length must match the encoder chain")
    if dev_query_embeddings_seq is not None:
        for hop_idx, dev_q in enumerate(dev_query_embeddings_seq):
            if _matrices_share_rows(dev_q, query_embeddings_seq[hop_idx]):
                raise ValueError("4-hop DEV queries overlap EVAL queries")
        dev_q_v1 = np.asarray(dev_query_embeddings_seq[0], dtype=np.float64)
        q_v1_width = int(np.asarray(query_embeddings_seq[0]).shape[1])
        if dev_q_v1.ndim != 2 or int(dev_q_v1.shape[1]) != q_v1_width:
            raise ValueError("4-hop DEV queries must live in the v1 encoder width")
    if dev_docs_seq is not None and len(dev_docs_seq) != n_enc:
        raise ValueError("dev_docs_seq length must match the encoder chain")
    if dev_docs_seq is None and dev_docs is not None:
        for docs in doc_embeddings_seq:
            if _matrices_share_rows(dev_docs, docs):
                raise ValueError("4-hop DEV documents overlap the documents being scored")

    bridge_cls = BRIDGE_REGISTRY[bridge_name]
    merger = CSLSMerge(csls_k=csls_k) if merge_name == "CSLSMerge" else MERGE_REGISTRY[merge_name]()
    use_polar = (bridge_name == "ZeroMuPolarBridge") if polar_reproject is None else bool(polar_reproject)

    anc = np.asarray(anchor_indices, dtype=np.int64)
    q_v1 = np.asarray(query_embeddings_seq[0], dtype=np.float64)
    d_v1_anc = np.asarray(doc_embeddings_seq[0], dtype=np.float64)[anc]
    zero_mask = np.zeros(len(doc_ids), dtype=bool)

    step_bridges: List[BaseBridge] = []
    transitions: List[MultiHopTransitionMetrics] = []
    dims_so_far: List[int] = [int(doc_embeddings_seq[0].shape[1])]

    for hop_idx in range(1, n_enc):
        prev_anc = np.asarray(doc_embeddings_seq[hop_idx - 1], dtype=np.float64)[anc]
        curr_docs = np.asarray(doc_embeddings_seq[hop_idx], dtype=np.float64)
        curr_anc = curr_docs[anc]
        dims_so_far.append(int(curr_docs.shape[1]))

        step_b = bridge_cls.fit(prev_anc, curr_anc)
        step_bridges.append(step_b)
        composed_b = compose_bridge_chain(step_bridges, polar_reproject=use_polar)
        direct_b = bridge_cls.fit(d_v1_anc, curr_anc)

        hop_dev_docs = _resolve_hop_dev_docs(
            bridge_name=bridge_name,
            hop_idx=hop_idx,
            curr_docs=curr_docs,
            dev_docs=dev_docs,
            dev_docs_seq=dev_docs_seq,
        )
        score_dev = {}
        if bridge_name == "ZeroMuPolarBridge" and hop_dev_docs is not None:
            score_dev["dev_docs"] = hop_dev_docs
        comp_raw = composed_b.score_matrix(q_v1, curr_docs, **score_dev)
        dir_raw = direct_b.score_matrix(q_v1, curr_docs, **score_dev)
        native_dummy = np.zeros_like(comp_raw)
        if dev_query_embeddings_seq is not None:
            dev_q = np.asarray(dev_query_embeddings_seq[0], dtype=np.float64)
            comp_ref = composed_b.score_matrix(dev_q, curr_docs, **score_dev)
            dir_ref = direct_b.score_matrix(dev_q, curr_docs, **score_dev)
            strict_dev = merge_name in {"CSLSMerge", "CalibratedZScoreMerge"}
        else:
            comp_ref = comp_raw
            dir_ref = dir_raw
            strict_dev = False

        comp_scores = merger.merge(
            comp_raw, native_dummy, zero_mask, bridged_ref=comp_ref, native_ref=native_dummy, strict_disjoint_dev=strict_dev
        ) if strict_dev or merge_name in {"CSLSMerge", "CalibratedZScoreMerge"} else merger.merge(
            comp_raw, native_dummy, zero_mask, bridged_ref=comp_ref, native_ref=native_dummy
        )
        dir_scores = merger.merge(
            dir_raw, native_dummy, zero_mask, bridged_ref=dir_ref, native_ref=native_dummy, strict_disjoint_dev=strict_dev
        ) if strict_dev or merge_name in {"CSLSMerge", "CalibratedZScoreMerge"} else merger.merge(
            dir_raw, native_dummy, zero_mask, bridged_ref=dir_ref, native_ref=native_dummy
        )

        c_ndcg_map, c_rec_map = evaluate_score_matrix(comp_scores, query_ids, doc_ids, qrels, k=k)
        d_ndcg_map, d_rec_map = evaluate_score_matrix(dir_scores, query_ids, doc_ids, qrels, k=k)
        c_ndcg = float(np.mean(list(c_ndcg_map.values()))) if c_ndcg_map else 0.0
        c_rec = float(np.mean(list(c_rec_map.values()))) if c_rec_map else 0.0
        d_ndcg = float(np.mean(list(d_ndcg_map.values()))) if d_ndcg_map else 0.0
        d_rec = float(np.mean(list(d_rec_map.values()))) if d_rec_map else 0.0

        transitions.append(
            MultiHopTransitionMetrics(
                hop_number=hop_idx,
                transition_label="->".join(encoder_names[: hop_idx + 1]),
                dims_path=tuple(dims_so_far),
                regime=classify_dimension_chain_regime(dims_so_far),
                composed_kappa=composed_b.condition_number,
                composed_nonzero_kappa=composed_b.nonzero_condition_number,
                composed_pre_polar_sigma_min=float(getattr(composed_b, "pre_polar_sigma_min", 1.0)),
                composed_ndcg_at_10=c_ndcg,
                composed_recall_at_10=c_rec,
                direct_kappa=direct_b.condition_number,
                direct_nonzero_kappa=direct_b.nonzero_condition_number,
                direct_ndcg_at_10=d_ndcg,
                direct_recall_at_10=d_rec,
                composition_gap_ndcg=c_ndcg - d_ndcg,
                composition_gap_recall=c_rec - d_rec,
            )
        )

    final_t = transitions[-1]
    return MultiHopChainReport(
        bridge_name=bridge_name,
        merge_name=merge_name,
        polar_reproject=use_polar,
        encoder_names=tuple(encoder_names),
        transitions=tuple(transitions),
        final_composed_ndcg_at_10=final_t.composed_ndcg_at_10,
        final_direct_ndcg_at_10=final_t.direct_ndcg_at_10,
        final_composition_gap_ndcg=final_t.composition_gap_ndcg,
    )


# ---------------------------------------------------------------------------
# 6. Progressive Anchor Contamination Sweep (0,1,5,10,20%) + TRA-1 Diagnostics
# ---------------------------------------------------------------------------


def inject_anchor_contamination(
    src_anchors: np.ndarray,
    tgt_anchors: np.ndarray,
    *,
    rate: float,
    mode: str = "mixed",
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deterministically contaminate `rate` fraction of anchor pairs.

    Modes:
      - `mismatched`: cyclic derangement of target rows (`y_i <- y_{perm(i)}`).
      - `duplicate`: collapse contaminated pairs onto one repeated anchor row.
      - `off_domain`: replace contaminated target rows with orthogonal OOD random vectors.
      - `mixed`: balanced split across mismatched, duplicate, and off_domain.
    Returns `(src_out, tgt_out, corrupted_mask)`.
    """
    if not (0.0 <= rate <= 1.0):
        raise ValueError("contamination rate must be in [0, 1]")
    if mode not in {"mismatched", "duplicate", "off_domain", "mixed"}:
        raise ValueError(f"Unsupported contamination mode={mode!r}")

    src_out = np.asarray(src_anchors, dtype=np.float64).copy()
    tgt_out = np.asarray(tgt_anchors, dtype=np.float64).copy()
    n = src_out.shape[0]
    corrupted = np.zeros(n, dtype=bool)
    n_bad = int(round(n * float(rate)))
    if n_bad <= 0:
        return src_out, tgt_out, corrupted

    rng = np.random.default_rng(seed + int(round(rate * 10_000)))
    chosen = np.sort(rng.choice(n, size=min(n_bad, n), replace=False))
    corrupted[chosen] = True

    def _apply_mismatched(idxs: np.ndarray) -> None:
        if idxs.size == 1:
            alt = (int(idxs[0]) + 1) % n
            tgt_out[idxs[0]] = tgt_anchors[alt]
        elif idxs.size > 1:
            rolled = np.roll(idxs, 1)
            tgt_out[idxs] = tgt_anchors[rolled]

    def _apply_duplicate(idxs: np.ndarray) -> None:
        if idxs.size > 0:
            proto_idx = int(idxs[0])
            src_out[idxs] = src_anchors[proto_idx]
            tgt_out[idxs] = tgt_anchors[proto_idx]

    def _apply_off_domain(idxs: np.ndarray) -> None:
        if idxs.size > 0:
            ood = l2_normalize(rng.normal(size=(idxs.size, tgt_out.shape[1])))
            tgt_out[idxs] = ood

    if mode == "mismatched":
        _apply_mismatched(chosen)
    elif mode == "duplicate":
        _apply_duplicate(chosen)
    elif mode == "off_domain":
        _apply_off_domain(chosen)
    else:
        chunks = np.array_split(chosen, 3)
        _apply_mismatched(chunks[0])
        _apply_duplicate(chunks[1])
        _apply_off_domain(chunks[2])

    return src_out, tgt_out, corrupted


@dataclass(frozen=True)
class TRA1AnchorDiagnostics:
    anchor_count: int
    nominal_rank: int
    effective_rank: int
    rank_ratio: float
    kappa: float
    nonzero_kappa: float
    singular_values_top5: Tuple[float, ...]
    min_nonzero_sigma: float
    residual_cosine_mean: float
    residual_cosine_p05: float
    residual_l2_median: float
    residual_l2_mad: float
    outlier_fraction: float
    duplicate_pair_fraction: float
    tra1_health_status: str
    diagnostic_sha256: str


def compute_tra1_anchor_diagnostics(
    src_anchors: np.ndarray,
    tgt_anchors: np.ndarray,
    bridge: BaseBridge,
) -> TRA1AnchorDiagnostics:
    """Compute TRA-1 spectral rank, conditioning, duplicate, and MAD outlier diagnostics."""
    xs = l2_normalize(src_anchors)
    yt = l2_normalize(tgt_anchors)
    n_anc = int(xs.shape[0])
    mapped = bridge.transform(xs, normalize=True)

    # For ZeroMuPolarBridge, align centered directions so diagnostics reflect actual alignment
    if isinstance(bridge, ZeroMuPolarBridge):
        yt_cmp = l2_normalize(yt - yt.mean(axis=0, keepdims=True))
    else:
        yt_cmp = yt

    cos_align = np.sum(mapped * yt_cmp, axis=1)
    l2_res = np.linalg.norm(mapped - yt_cmp, axis=1)
    med_res = float(np.median(l2_res))
    mad_res = float(np.median(np.abs(l2_res - med_res)))
    scale_mad = max(1.4826 * mad_res, 1e-6)
    z_mad = (l2_res - med_res) / scale_mad
    outlier_mask = (z_mad > 3.5) | (cos_align < 0.5)
    outlier_frac = float(np.mean(outlier_mask))

    # Near-duplicate anchor detection on source+target joint cosine similarity
    src_gram = xs @ xs.T
    tgt_gram = yt @ yt.T
    np.fill_diagonal(src_gram, -1.0)
    np.fill_diagonal(tgt_gram, -1.0)
    dup_mask = np.any((src_gram > 0.9995) & (tgt_gram > 0.9995), axis=1)
    dup_frac = float(np.mean(dup_mask))

    # Anchor cross-covariance rank & bridge spectral diagnostics
    xc = xs - xs.mean(axis=0, keepdims=True)
    yc = yt - yt.mean(axis=0, keepdims=True)
    sv_cross = np.linalg.svd(xc.T @ yc, compute_uv=False)
    sv0 = float(sv_cross[0]) if sv_cross.size else 0.0
    nominal_rank = min(n_anc - 1, int(xs.shape[1]), int(yt.shape[1]))
    eff_rank = int(np.sum(sv_cross > 1e-4 * max(sv0, 1e-12))) if sv0 > 1e-12 else 0
    rank_ratio = float(eff_rank) / float(max(nominal_rank, 1))

    kappa = bridge.condition_number
    nz_kappa = bridge.nonzero_condition_number
    _, _, _, min_nz = _singular_diagnostics(bridge.linear_components()[2], bridge.linear_components()[3])

    if eff_rank == 0 or not math.isfinite(nz_kappa) or outlier_frac >= 0.25:
        status = "FAIL_CLOSED_DEGENERATE"
    elif outlier_frac >= 0.03 or dup_frac >= 0.03 or rank_ratio < 0.50:
        status = "WARN_CONTAMINATED"
    else:
        status = "PASS"

    payload = {
        "anchor_count": n_anc,
        "nominal_rank": nominal_rank,
        "effective_rank": eff_rank,
        "outlier_fraction": round(outlier_frac, 6),
        "duplicate_pair_fraction": round(dup_frac, 6),
        "tra1_health_status": status,
        "weights_sha256": bridge.weights_sha256,
    }
    return TRA1AnchorDiagnostics(
        anchor_count=n_anc,
        nominal_rank=nominal_rank,
        effective_rank=eff_rank,
        rank_ratio=rank_ratio,
        kappa=kappa,
        nonzero_kappa=nz_kappa,
        singular_values_top5=tuple(float(v) for v in sv_cross[:5]),
        min_nonzero_sigma=min_nz,
        residual_cosine_mean=float(np.mean(cos_align)),
        residual_cosine_p05=float(np.quantile(cos_align, 0.05)),
        residual_l2_median=med_res,
        residual_l2_mad=mad_res,
        outlier_fraction=outlier_frac,
        duplicate_pair_fraction=dup_frac,
        tra1_health_status=status,
        diagnostic_sha256=_sha256_json(payload),
    )


@dataclass(frozen=True)
class ContaminationSweepPoint:
    bridge_name: str
    merge_name: str
    contamination_mode: str
    contamination_rate: float
    corrupted_count: int
    mean_ndcg_at_10: float
    mean_recall_at_10: float
    tra1: TRA1AnchorDiagnostics


def run_contamination_sweep(
    *,
    src_anchors: np.ndarray,
    tgt_anchors: np.ndarray,
    eval_queries_src: np.ndarray,
    docs_old: np.ndarray,
    query_ids: Sequence[str],
    doc_ids: Sequence[str],
    qrels: Mapping[str, Mapping[str, float]],
    bridge_names: Sequence[str] = ("UncenteredProcrustes", "CenteredProcrustes", "ZeroMuPolarBridge"),
    merge_name: str = "CSLSMerge",
    rates: Sequence[float] = DEFAULT_CONTAMINATION_RATES,
    modes: Sequence[str] = ("mismatched", "duplicate", "off_domain"),
    seed: int = 42,
    csls_k: int = DEFAULT_CSLS_K,
    k: int = 10,
    dev_queries_src: Optional[np.ndarray] = None,
    dev_docs: Optional[np.ndarray] = None,
) -> List[ContaminationSweepPoint]:
    """Sweep progressive anchor contamination `(0%, 1%, 5%, 10%, 20%)` with TRA-1 diagnostics.

    When ``dev_queries_src`` is supplied, CSLS and z-score calibration use that
    disjoint query set. ``dev_docs`` calibrates ZeroMu ``mu_d`` only. Omitting
    both preserves the historical eval-as-reference path.
    """
    if merge_name not in MERGE_REGISTRY:
        raise ValueError(f"Unknown merge_name={merge_name!r}")
    merger = CSLSMerge(csls_k=csls_k) if merge_name == "CSLSMerge" else MERGE_REGISTRY[merge_name]()
    zero_mask = np.zeros(len(doc_ids), dtype=bool)
    strict_dev = dev_queries_src is not None and merge_name in {"CSLSMerge", "CalibratedZScoreMerge"}
    if dev_queries_src is not None and _matrices_share_rows(dev_queries_src, eval_queries_src):
        raise ValueError("contamination DEV queries overlap EVAL queries")
    if dev_docs is not None and _matrices_share_rows(dev_docs, docs_old):
        raise ValueError("contamination DEV documents overlap the documents being scored")
    if (
        dev_docs is not None
        and "ZeroMuPolarBridge" in tuple(bridge_names)
        and int(np.asarray(dev_docs).shape[1]) != int(np.asarray(docs_old).shape[1])
    ):
        raise ValueError(
            "contamination dev_docs width "
            f"{int(np.asarray(dev_docs).shape[1])} does not match docs_old width {int(np.asarray(docs_old).shape[1])}"
        )
    points: List[ContaminationSweepPoint] = []

    for mode in modes:
        for rate in rates:
            c_src, c_tgt, bad_mask = inject_anchor_contamination(
                src_anchors, tgt_anchors, rate=float(rate), mode=mode, seed=seed
            )
            for b_name in bridge_names:
                bridge = BRIDGE_REGISTRY[b_name].fit(c_src, c_tgt)
                score_kwargs: Dict[str, Any] = {}
                if b_name == "ZeroMuPolarBridge" and dev_docs is not None:
                    score_kwargs["dev_docs"] = dev_docs
                raw = bridge.score_matrix(eval_queries_src, docs_old, **score_kwargs)
                dummy = np.zeros_like(raw)
                if strict_dev:
                    ref = bridge.score_matrix(dev_queries_src, docs_old, **score_kwargs)
                    scores = merger.merge(
                        raw, dummy, zero_mask, bridged_ref=ref, native_ref=dummy, strict_disjoint_dev=True
                    )
                else:
                    scores = merger.merge(raw, dummy, zero_mask, bridged_ref=raw, native_ref=dummy)
                pq_n, pq_r = evaluate_score_matrix(scores, query_ids, doc_ids, qrels, k=k)
                tra1 = compute_tra1_anchor_diagnostics(c_src, c_tgt, bridge)
                points.append(
                    ContaminationSweepPoint(
                        bridge_name=b_name,
                        merge_name=merge_name,
                        contamination_mode=mode,
                        contamination_rate=float(rate),
                        corrupted_count=int(np.sum(bad_mask)),
                        mean_ndcg_at_10=float(np.mean(list(pq_n.values()))) if pq_n else 0.0,
                        mean_recall_at_10=float(np.mean(list(pq_r.values()))) if pq_r else 0.0,
                        tra1=tra1,
                    )
                )
    return points


# ---------------------------------------------------------------------------
# 7. Decomposed Item 6B Hypothesis Evaluator (Bootstrap + Exact Binomial + Holm)
# ---------------------------------------------------------------------------


def paired_bootstrap_ci(
    deltas: Sequence[float],
    *,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 1601,
) -> Tuple[float, float, float]:
    """Deterministic paired percentile bootstrap returning `(mean, lcb, ucb)`."""
    arr = np.asarray(list(deltas), dtype=np.float64)
    if arr.size == 0:
        raise ValueError("deltas must be non-empty")
    mean_val = float(np.mean(arr))
    if arr.size == 1:
        return mean_val, mean_val, mean_val
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(int(n_resamples), arr.size))
    boot_means = arr[idx].mean(axis=1)
    lcb = float(np.quantile(boot_means, alpha / 2.0))
    ucb = float(np.quantile(boot_means, 1.0 - alpha / 2.0))
    return mean_val, lcb, ucb


def exact_binomial_upper_tail_p(k_success: int, n_trials: int, p0: float = 0.5) -> float:
    """Exact upper-tail binomial probability `P(X >= k_success)` for `X ~ Bin(n, p0)`."""
    if n_trials <= 0:
        raise ValueError("n_trials must be positive")
    if k_success <= 0:
        return 1.0
    if k_success > n_trials:
        return 0.0
    total = 0.0
    for j in range(int(k_success), int(n_trials) + 1):
        total += math.comb(n_trials, j) * (p0 ** j) * ((1.0 - p0) ** (n_trials - j))
    return min(1.0, max(0.0, float(total)))


def holm_bonferroni_adjust(p_values: Mapping[str, float]) -> Dict[str, float]:
    """Step-down Holm-Bonferroni family-wise adjusted p-values."""
    items = sorted(p_values.items(), key=lambda kv: (float(kv[1]), kv[0]))
    m = len(items)
    adjusted: Dict[str, float] = {}
    running_max = 0.0
    for rank, (name, raw_p) in enumerate(items):
        adj = min(1.0, float(raw_p) * float(m - rank))
        running_max = max(running_max, adj)
        adjusted[name] = running_max
    return adjusted


@dataclass(frozen=True)
class Item6BHypothesisReport:
    n_domains: int
    alpha_fwer: float
    h6b_merge: Dict[str, Any]
    h6b_bridge_static: Dict[str, Any]
    h6b_bridge_shift_and_4hop: Dict[str, Any]
    holm_adjusted_p_values: Dict[str, float]
    overall_verdict: str
    item6b_local_diagnostic_verdict: str
    report_sha256: str


def evaluate_item6b_hypotheses(
    *,
    clean_1hop_csls_vs_zscore_by_domain: Mapping[str, float],
    static_zeromu_vs_uncentered_at_csls_by_domain: Mapping[str, float],
    static_zeromu_vs_centered_at_csls_by_domain: Mapping[str, float],
    shift_zeromu_vs_uncentered_at_csls_by_domain: Mapping[str, float],
    fourhop_zeromu_vs_uncentered_at_csls_by_domain: Mapping[str, float],
    prefix_displacement_norms_by_domain: Optional[Mapping[str, float]] = None,
    direct_v1_v5_delta_by_domain: Optional[Mapping[str, float]] = None,
    min_positive_domains_merge: Optional[int] = None,
    retirement_floor: float = ITEM6B_RETIREMENT_FLOOR,
    tight_equivalence_floor: float = ITEM6B_TIGHT_EQUIVALENCE_FLOOR,
    alpha_fwer: float = 0.05,
    n_resamples: int = 2000,
    seed: int = 1601,
) -> Item6BHypothesisReport:
    """Evaluate the three decomposed pre-registered Item 6B (#54) hypotheses.

    1. `H6B_merge` (Superiority): `CSLS` > `CalibratedZScore` on clean 1-hop `nDCG@10`
       (`LCB > 0` AND `>= 11/12` domains positive, exact binomial test).
       The local 4-test Holm summary is a diagnostic. It does not stamp a
       registry status. Registry adjudication is the 5-claim family at m=5.
    2. `H6B_bridge_static` (Non-Inferiority Guard): At fixed `CSLS` on unshifted
       1-hop queries, `ZeroMuPolarBridge` vs `UncenteredProcrustes`, reporting BOTH
       the `-0.010` retirement floor AND `-0.005` tight equivalence floor, plus
       the explicit `CenteredProcrustes + CSLS` contrast.
    3. `H6B_bridge_shift_and_4hop` (Superiority): At fixed `CSLS` without refitting,
       `ZeroMuPolarBridge` beats `UncenteredProcrustes` under prompt-shift AND 4-hop
       sequential composition (`v1->v2->v3->v4->v5`), alongside direct `v1->v5`.
    """
    domains = sorted(clean_1hop_csls_vs_zscore_by_domain.keys())
    n_dom = len(domains)
    if n_dom == 0:
        raise ValueError("At least one domain is required for Item 6B evaluation")
    for label, mapping in (
        ("static_zeromu_vs_uncentered", static_zeromu_vs_uncentered_at_csls_by_domain),
        ("static_zeromu_vs_centered", static_zeromu_vs_centered_at_csls_by_domain),
        ("shift_zeromu_vs_uncentered", shift_zeromu_vs_uncentered_at_csls_by_domain),
        ("fourhop_zeromu_vs_uncentered", fourhop_zeromu_vs_uncentered_at_csls_by_domain),
    ):
        if sorted(mapping.keys()) != domains:
            raise ValueError(f"{label} domain keys must match clean_1hop_csls_vs_zscore_by_domain")

    req_pos_merge = (
        int(min_positive_domains_merge)
        if min_positive_domains_merge is not None
        else (11 if n_dom == 12 else max(1, math.ceil(n_dom * (11.0 / 12.0))))
    )

    # 1. H6B_merge
    merge_deltas = [float(clean_1hop_csls_vs_zscore_by_domain[d]) for d in domains]
    m_mean, m_lcb, m_ucb = paired_bootstrap_ci(merge_deltas, n_resamples=n_resamples, alpha=alpha_fwer, seed=seed)
    m_pos = sum(1 for v in merge_deltas if v > 0.0)
    m_p = exact_binomial_upper_tail_p(m_pos, n_dom, 0.5)

    # 2. H6B_bridge_static (ZeroMu vs Uncentered AND ZeroMu vs Centered, both at fixed CSLS)
    st_unc = [float(static_zeromu_vs_uncentered_at_csls_by_domain[d]) for d in domains]
    st_cen = [float(static_zeromu_vs_centered_at_csls_by_domain[d]) for d in domains]
    su_mean, su_lcb, su_ucb = paired_bootstrap_ci(st_unc, n_resamples=n_resamples, alpha=alpha_fwer, seed=seed + 1)
    sc_mean, sc_lcb, sc_ucb = paired_bootstrap_ci(st_cen, n_resamples=n_resamples, alpha=alpha_fwer, seed=seed + 2)
    su_pass_retirement = su_lcb > float(retirement_floor)
    su_pass_tight = su_lcb > float(tight_equivalence_floor)
    # Non-inferiority p-value via bootstrap tail below retirement floor
    rng_ni = np.random.default_rng(seed + 3)
    arr_su = np.asarray(st_unc, dtype=np.float64)
    boot_su = arr_su[rng_ni.integers(0, arr_su.size, size=(n_resamples, arr_su.size))].mean(axis=1)
    raw_ni_p = float(np.mean(boot_su <= float(retirement_floor)))
    ni_p = min(1.0, raw_ni_p + (1.0 / (int(n_resamples) + 1)))

    # 3. H6B_bridge_shift_and_4hop
    sh_deltas = [float(shift_zeromu_vs_uncentered_at_csls_by_domain[d]) for d in domains]
    fh_deltas = [float(fourhop_zeromu_vs_uncentered_at_csls_by_domain[d]) for d in domains]
    sh_mean, sh_lcb, sh_ucb = paired_bootstrap_ci(sh_deltas, n_resamples=n_resamples, alpha=alpha_fwer, seed=seed + 4)
    fh_mean, fh_lcb, fh_ucb = paired_bootstrap_ci(fh_deltas, n_resamples=n_resamples, alpha=alpha_fwer, seed=seed + 5)
    sh_pos = sum(1 for v in sh_deltas if v > 0.0)
    fh_pos = sum(1 for v in fh_deltas if v > 0.0)
    sh_p = exact_binomial_upper_tail_p(sh_pos, n_dom, 0.5)
    fh_p = exact_binomial_upper_tail_p(fh_pos, n_dom, 0.5)

    holm = holm_bonferroni_adjust(
        {
            "H6B_merge": m_p,
            "H6B_bridge_static_ni": ni_p,
            "H6B_bridge_shift": sh_p,
            "H6B_bridge_4hop": fh_p,
        }
    )

    h6b_merge_pass = bool(m_lcb > 0.0 and m_pos >= req_pos_merge and holm["H6B_merge"] <= alpha_fwer)
    h6b_static_pass = bool(su_pass_retirement and holm["H6B_bridge_static_ni"] <= alpha_fwer)
    h6b_shift_pass = bool(sh_lcb > 0.0 and holm["H6B_bridge_shift"] <= alpha_fwer)
    h6b_4hop_pass = bool(fh_lcb > 0.0 and holm["H6B_bridge_4hop"] <= alpha_fwer)
    h6b_shift_4hop_pass = bool(h6b_shift_pass and h6b_4hop_pass)

    if h6b_merge_pass and h6b_static_pass and h6b_shift_4hop_pass:
        local_verdict = "local-diagnostic-support"
    elif h6b_merge_pass and not h6b_shift_4hop_pass:
        local_verdict = "local-diagnostic-credit-csls-retire-zeromu"
    else:
        local_verdict = "local-diagnostic-not-supported"
    registry_statuses = {
        "verified-pass",
        "verified-loss",
        "unsupported",
        "protocol-ready",
        "reserved-pending",
    }
    if local_verdict in registry_statuses:
        raise RuntimeError("Item 6B local diagnostic collided with a registry status")
    verdict = local_verdict

    h6b_merge = {
        "hypothesis": "CSLS > CalibratedZScore on clean 1-hop nDCG@10",
        "mean_delta": m_mean,
        "lcb_95": m_lcb,
        "ucb_95": m_ucb,
        "positive_domains": m_pos,
        "required_positive_domains": req_pos_merge,
        "sign_test_count": f"{m_pos}/{n_dom}",
        "sign_test_required": "11/12" if n_dom == 12 else f"{req_pos_merge}/{n_dom}",
        "holm_m_local": 4,
        "holm_m_registry": 5,
        "n_domains": n_dom,
        "exact_binomial_p": m_p,
        "holm_adjusted_p": holm["H6B_merge"],
        "passed": h6b_merge_pass,
        "registry_status_stamped": False,
    }
    h6b_bridge_static = {
        "hypothesis": "ZeroMuPolarBridge+CSLS non-inferior to UncenteredProcrustes+CSLS on static 1-hop",
        "zeromu_vs_uncentered_mean": su_mean,
        "zeromu_vs_uncentered_lcb_95": su_lcb,
        "zeromu_vs_uncentered_ucb_95": su_ucb,
        "retirement_floor": float(retirement_floor),
        "clears_retirement_floor": su_pass_retirement,
        "tight_equivalence_floor": float(tight_equivalence_floor),
        "clears_tight_equivalence_floor": su_pass_tight,
        "zeromu_vs_centered_csls_contrast": {
            "mean_delta": sc_mean,
            "lcb_95": sc_lcb,
            "ucb_95": sc_ucb,
            "note": "Explicit separation of simple centering (CenteredProcrustes+CSLS) vs ZeroMu+CSLS",
        },
        "bootstrap_ni_p": ni_p,
        "bootstrap_ni_p_floor": 1.0 / (int(n_resamples) + 1),
        "holm_adjusted_p": holm["H6B_bridge_static_ni"],
        "passed": h6b_static_pass,
    }
    h6b_bridge_shift_and_4hop = {
        "hypothesis": "ZeroMuPolarBridge+CSLS > UncenteredProcrustes+CSLS under prompt shift and 4-hop v1->v5",
        "prompt_shift": {
            "mean_delta": sh_mean,
            "lcb_95": sh_lcb,
            "ucb_95": sh_ucb,
            "positive_domains": sh_pos,
            "exact_binomial_p": sh_p,
            "holm_adjusted_p": holm["H6B_bridge_shift"],
            "mean_prefix_displacement_norm": (
                float(np.mean(list(prefix_displacement_norms_by_domain.values())))
                if prefix_displacement_norms_by_domain
                else None
            ),
            "passed": h6b_shift_pass,
        },
        "four_hop_v1_to_v5": {
            "mean_delta": fh_mean,
            "lcb_95": fh_lcb,
            "ucb_95": fh_ucb,
            "positive_domains": fh_pos,
            "exact_binomial_p": fh_p,
            "holm_adjusted_p": holm["H6B_bridge_4hop"],
            "mean_direct_v1_v5_delta": (
                float(np.mean(list(direct_v1_v5_delta_by_domain.values())))
                if direct_v1_v5_delta_by_domain
                else None
            ),
            "passed": h6b_4hop_pass,
        },
        "passed": h6b_shift_4hop_pass,
    }

    digest = _sha256_json(
        {
            "n_domains": n_dom,
            "overall_verdict": verdict,
            "item6b_local_diagnostic_verdict": local_verdict,
            "holm_m_local": 4,
            "holm_m_registry": 5,
            "sign_test_required": "11/12" if n_dom == 12 else f"{req_pos_merge}/{n_dom}",
            "holm": holm,
            "h6b_merge_pass": h6b_merge_pass,
            "h6b_static_pass": h6b_static_pass,
            "h6b_shift_4hop_pass": h6b_shift_4hop_pass,
        }
    )
    return Item6BHypothesisReport(
        n_domains=n_dom,
        alpha_fwer=float(alpha_fwer),
        h6b_merge=h6b_merge,
        h6b_bridge_static=h6b_bridge_static,
        h6b_bridge_shift_and_4hop=h6b_bridge_shift_and_4hop,
        holm_adjusted_p_values=holm,
        overall_verdict=verdict,
        item6b_local_diagnostic_verdict=local_verdict,
        report_sha256=digest,
    )


# ---------------------------------------------------------------------------
# 8. Arrival-to-Delivery RequestSpanProfiler (Timeouts Kept as 0.0 nDCG@10)
# ---------------------------------------------------------------------------


@dataclass
class RequestSpanRecord:
    request_id: str
    queue_us: float = 0.0
    backend_knn_us: float = 0.0
    prefilter_visited: int = 0
    short_refill_count: int = 0
    exact_fallback_triggered: bool = False
    gemv_us: float = 0.0
    merge_us: float = 0.0
    total_us: float = 0.0
    timed_out: bool = False
    raw_ndcg_at_10: float = 0.0
    effective_ndcg_at_10: float = field(init=False)

    def __post_init__(self) -> None:
        # Non-negotiable consensus rule: timeouts stay in denominator strictly as 0.0 nDCG@10.
        self.effective_ndcg_at_10 = 0.0 if self.timed_out else float(self.raw_ndcg_at_10)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class _ActiveSpanHandle:
    """Mutable timer handle yielded by `RequestSpanProfiler.trace(...)`."""

    def __init__(self, request_id: str) -> None:
        self.request_id = request_id
        self.queue_us = 0.0
        self.backend_knn_us = 0.0
        self.prefilter_visited = 0
        self.short_refill_count = 0
        self.exact_fallback_triggered = False
        self.gemv_us = 0.0
        self.merge_us = 0.0
        self.timed_out = False
        self.raw_ndcg_at_10 = 0.0
        self._start_ns = time.perf_counter_ns()

    @contextmanager
    def measure_stage(self, stage: str) -> Iterator[None]:
        if stage not in {"queue", "backend_knn", "gemv", "merge"}:
            raise ValueError(f"Unsupported span stage={stage!r}")
        t0 = time.perf_counter_ns()
        try:
            yield
        finally:
            elapsed_us = (time.perf_counter_ns() - t0) / 1_000.0
            attr = f"{stage}_us"
            setattr(self, attr, getattr(self, attr) + elapsed_us)

    def finalize(self, total_us_override: Optional[float] = None) -> RequestSpanRecord:
        total_us = (
            float(total_us_override)
            if total_us_override is not None
            else (time.perf_counter_ns() - self._start_ns) / 1_000.0
        )
        return RequestSpanRecord(
            request_id=self.request_id,
            queue_us=self.queue_us,
            backend_knn_us=self.backend_knn_us,
            prefilter_visited=int(self.prefilter_visited),
            short_refill_count=int(self.short_refill_count),
            exact_fallback_triggered=bool(self.exact_fallback_triggered),
            gemv_us=self.gemv_us,
            merge_us=self.merge_us,
            total_us=total_us,
            timed_out=bool(self.timed_out),
            raw_ndcg_at_10=float(self.raw_ndcg_at_10),
        )


class RequestSpanProfiler:
    """Arrival-to-delivery microsecond span profiler enforcing timeout-in-denominator accounting."""

    def __init__(self) -> None:
        self._records: List[RequestSpanRecord] = []

    @property
    def records(self) -> Tuple[RequestSpanRecord, ...]:
        return tuple(self._records)

    def record_span(
        self,
        *,
        request_id: str,
        queue_us: float,
        backend_knn_us: float,
        prefilter_visited: int,
        short_refill_count: int,
        exact_fallback_triggered: bool,
        gemv_us: float,
        merge_us: float,
        total_us: float,
        timed_out: bool,
        ndcg_at_10: float = 0.0,
    ) -> RequestSpanRecord:
        rec = RequestSpanRecord(
            request_id=str(request_id),
            queue_us=float(queue_us),
            backend_knn_us=float(backend_knn_us),
            prefilter_visited=int(prefilter_visited),
            short_refill_count=int(short_refill_count),
            exact_fallback_triggered=bool(exact_fallback_triggered),
            gemv_us=float(gemv_us),
            merge_us=float(merge_us),
            total_us=float(total_us),
            timed_out=bool(timed_out),
            raw_ndcg_at_10=float(ndcg_at_10),
        )
        self._records.append(rec)
        return rec

    @contextmanager
    def trace(self, request_id: str) -> Iterator[_ActiveSpanHandle]:
        handle = _ActiveSpanHandle(request_id)
        try:
            yield handle
        finally:
            self._records.append(handle.finalize())

    def summarize(self) -> Dict[str, Any]:
        """Summarize span telemetry, strictly keeping timed-out requests in denominator as `0.0`."""
        n = len(self._records)
        if n == 0:
            return {
                "total_requests": 0,
                "timeout_count": 0,
                "timeout_rate": 0.0,
                "exact_fallback_count": 0,
                "exact_fallback_rate": 0.0,
                "mean_short_refill_count": 0.0,
                "mean_prefilter_visited": 0.0,
                "mean_ndcg_at_10_including_timeouts": 0.0,
                "mean_ndcg_at_10_completed_only": 0.0,
                "p50_total_us": 0.0,
                "p95_total_us": 0.0,
                "p99_total_us": 0.0,
                "stage_p95_us": {"queue_us": 0.0, "backend_knn_us": 0.0, "gemv_us": 0.0, "merge_us": 0.0},
            }

        totals = np.asarray([r.total_us for r in self._records], dtype=np.float64)
        queues = np.asarray([r.queue_us for r in self._records], dtype=np.float64)
        knns = np.asarray([r.backend_knn_us for r in self._records], dtype=np.float64)
        gemvs = np.asarray([r.gemv_us for r in self._records], dtype=np.float64)
        merges = np.asarray([r.merge_us for r in self._records], dtype=np.float64)
        eff_ndcg = np.asarray([r.effective_ndcg_at_10 for r in self._records], dtype=np.float64)
        completed_ndcg = [r.effective_ndcg_at_10 for r in self._records if not r.timed_out]
        timeout_count = sum(1 for r in self._records if r.timed_out)
        fallback_count = sum(1 for r in self._records if r.exact_fallback_triggered)

        return {
            "total_requests": n,
            "timeout_count": timeout_count,
            "timeout_rate": float(timeout_count) / float(n),
            "exact_fallback_count": fallback_count,
            "exact_fallback_rate": float(fallback_count) / float(n),
            "mean_short_refill_count": float(np.mean([r.short_refill_count for r in self._records])),
            "mean_prefilter_visited": float(np.mean([r.prefilter_visited for r in self._records])),
            "mean_ndcg_at_10_including_timeouts": float(np.mean(eff_ndcg)),
            "mean_ndcg_at_10_completed_only": float(np.mean(completed_ndcg)) if completed_ndcg else 0.0,
            "p50_total_us": float(np.quantile(totals, 0.50)),
            "p95_total_us": float(np.quantile(totals, 0.95)),
            "p99_total_us": float(np.quantile(totals, 0.99)),
            "stage_p95_us": {
                "queue_us": float(np.quantile(queues, 0.95)),
                "backend_knn_us": float(np.quantile(knns, 0.95)),
                "gemv_us": float(np.quantile(gemvs, 0.95)),
                "merge_us": float(np.quantile(merges, 0.95)),
            },
        }


__all__ = [
    "AnchorDraw",
    "BRIDGE_NAMES",
    "BRIDGE_REGISTRY",
    "BaseBridge",
    "BaseMerge",
    "CSLSMerge",
    "CalibratedZScoreMerge",
    "CenteredProcrustes",
    "ContaminationSweepPoint",
    "CrossDomainVarianceReport",
    "DEFAULT_ANCHOR_DRAWS",
    "DEFAULT_ANCHORS_PER_DRAW",
    "DEFAULT_CONTAMINATION_RATES",
    "DEFAULT_CSLS_K",
    "DEFAULT_EMA_ALPHA",
    "DomainAnchorVarianceSummary",
    "FactorialCellResult",
    "ITEM6B_RETIREMENT_FLOOR",
    "ITEM6B_TIGHT_EQUIVALENCE_FLOOR",
    "Item6BHypothesisReport",
    "MERGE_NAMES",
    "MERGE_REGISTRY",
    "MultiHopChainReport",
    "MultiHopTransitionMetrics",
    "Rank1QueryResidual",
    "RawCosineMerge",
    "RequestSpanProfiler",
    "RequestSpanRecord",
    "RidgeDriftAdapter",
    "TRA1AnchorDiagnostics",
    "UncenteredProcrustes",
    "ZeroMuPolarBridge",
    "classify_dimension_chain_regime",
    "compose_bridge_chain",
    "compute_tra1_anchor_diagnostics",
    "evaluate_4hop_chain",
    "evaluate_item6b_hypotheses",
    "evaluate_score_matrix",
    "exact_binomial_upper_tail_p",
    "holm_bonferroni_adjust",
    "inject_anchor_contamination",
    "l2_normalize",
    "paired_bootstrap_ci",
    "run_contamination_sweep",
    "run_factorial_5x3",
    "sample_anchor_draws",
    "summarize_anchor_draw_variance",
]
