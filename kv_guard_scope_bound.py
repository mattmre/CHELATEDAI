"""W22-10 (#60 half): receiver-activation rejection bound for #60's weight-bound KV guard.

Preregistration: ``artifacts/pv7_w23b_w22_10/PREREGISTRATION.md``.

Bundle #60 (``chelated_ssd_kv_repo_cache``) refuses every cached KV block whose TRA-1 contract digest differs from
the active weights. That guard is a binary identity check. This module is a research instrument that asks what a
relaxed guard could *safely* reuse on #60's tiny 2-layer float64 model:

* :func:`dependency_cone_unchanged` -- exact structural rule (A1): the prefix KV depends only on
  ``emb[prefix tokens]``, ``Wq_0, Wk_0, Wv_0, Wo_0`` and ``Wk_1, Wv_1``.
* :func:`receiver_activation_bound` -- a sound a-priori bound (A2) on every suffix logit error caused by reusing
  sender KV under receiver weights, from weight-delta spectral norms, sender per-block activation statistics and the
  receiver's own observed suffix activations.
* :func:`tlr_h_kv` / :func:`tlr_r_kv` -- selective top-layer re-prefill baselines (A3 / A3r).

Nothing here changes #60's guard; the module only reads #60's model and on-disk block format.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from chelated_ssd_kv_repo_cache import (
    D_MODEL,
    N_LAYERS,
    TinyModel,
    _positions,
    _rng_matrix,
    build_model,
)

KV = List[Tuple[np.ndarray, np.ndarray]]
SQRT_D = float(np.sqrt(D_MODEL))
SOUNDNESS_ABS_SLACK = 1e-12
SOUNDNESS_REL_SLACK = 1e-9
SVD_FLOPS_PER_MATRIX = 20 * D_MODEL ** 3  # preregistered estimate for one spectral norm


# ---------------------------------------------------------------------------
# instrumented forward (must reproduce TinyModel.forward bit-for-bit)
# ---------------------------------------------------------------------------
@dataclass
class LayerTrace:
    h_in: np.ndarray  # input hidden rows for the new tokens
    q: np.ndarray
    k_all: np.ndarray
    v_all: np.ndarray
    a: np.ndarray  # attention output before Wo


def forward_traced(model: TinyModel, tokens: np.ndarray, past: Optional[KV] = None) -> Tuple[np.ndarray, KV, List[LayerTrace]]:
    """Same arithmetic, same order as ``TinyModel.forward``; additionally returns per-layer internals."""
    n = len(tokens)
    start = 0 if past is None else past[0][0].shape[0]
    pos = np.arange(start, start + n)
    h = model.emb[tokens] + _positions(pos)
    new_kv: KV = []
    traces: List[LayerTrace] = []
    for layer in range(N_LAYERS):
        q = h @ model.wq[layer]
        k = h @ model.wk[layer]
        v = h @ model.wv[layer]
        new_kv.append((k, v))
        if past is not None:
            k_all = np.concatenate([past[layer][0], k], axis=0)
            v_all = np.concatenate([past[layer][1], v], axis=0)
        else:
            k_all, v_all = k, v
        total = k_all.shape[0]
        scores = (q @ k_all.T) / np.sqrt(D_MODEL)
        mask = np.arange(total)[None, :] <= (start + np.arange(n))[:, None]
        scores = np.where(mask, scores, -1e30)
        scores = scores - scores.max(axis=1, keepdims=True)
        w = np.exp(scores)
        w = w / w.sum(axis=1, keepdims=True)
        a = w @ v_all
        traces.append(LayerTrace(h, q, k_all, v_all, a))
        h = h + a @ model._wo(layer)
    return h @ model.wout, new_kv, traces


def cat_kv(a: Optional[KV], b: KV) -> KV:
    if a is None:
        return [(b[i][0].copy(), b[i][1].copy()) for i in range(N_LAYERS)]
    return [(np.concatenate([a[i][0], b[i][0]]), np.concatenate([a[i][1], b[i][1]])) for i in range(N_LAYERS)]


@dataclass
class PrefixRun:
    kv: KV
    stats: Dict[str, List[float]]  # H, Q, A per layer (max row norms over prefix rows)
    h_layer_inputs: List[np.ndarray]  # per layer, prefix rows of the layer input hidden state


def prefill_blocks(model: TinyModel, blocks_tokens: Sequence[np.ndarray]) -> PrefixRun:
    """Block-by-block prefill, the same procedure ``serve_chelated`` uses on a cold cache."""
    past: Optional[KV] = None
    H = [0.0] * N_LAYERS
    Q = [0.0] * N_LAYERS
    A = [0.0] * N_LAYERS
    hs: List[List[np.ndarray]] = [[] for _ in range(N_LAYERS)]
    for toks in blocks_tokens:
        _, new_kv, tr = forward_traced(model, toks, past=past)
        for layer in range(N_LAYERS):
            H[layer] = max(H[layer], float(np.linalg.norm(tr[layer].h_in, axis=1).max()))
            Q[layer] = max(Q[layer], float(np.linalg.norm(tr[layer].q, axis=1).max()))
            A[layer] = max(A[layer], float(np.linalg.norm(tr[layer].a, axis=1).max()))
            hs[layer].append(tr[layer].h_in)
        past = cat_kv(past, new_kv)
    assert past is not None
    return PrefixRun(past, {"H": H, "Q": Q, "A": A}, [np.concatenate(x) for x in hs])


# ---------------------------------------------------------------------------
# #60 on-disk block reader (no contract check: this is the relaxed path under study)
# ---------------------------------------------------------------------------
def read_block_unchecked(path: Path, prefix_tokens: np.ndarray) -> KV:
    """Parse a ``.kvblk`` written by ``SSDKVCache.put`` WITHOUT the TRA-1 contract check.

    Still verifies the causal-prefix token ids and body length, exactly as ``reload_persisted`` does.
    """
    raw = Path(path).read_bytes()
    head_raw, body = raw.split(b"\n", 1)
    header = json.loads(head_raw)
    if header["prefix_tokens"] != [int(t) for t in prefix_tokens]:
        raise ValueError("causal prefix token ids do not match cached block")
    n = header["shape"][0]
    if len(body) != N_LAYERS * 2 * n * D_MODEL * 8:
        raise ValueError("block body length does not match header")
    arrs = np.frombuffer(body, dtype=np.float64).reshape(N_LAYERS * 2, n, D_MODEL)
    return [(arrs[2 * i].copy(), arrs[2 * i + 1].copy()) for i in range(N_LAYERS)]


# ---------------------------------------------------------------------------
# weight revisions
# ---------------------------------------------------------------------------
def _copy(model: TinyModel) -> TinyModel:
    return TinyModel(
        model.emb.copy(),
        [w.copy() for w in model.wq],
        [w.copy() for w in model.wk],
        [w.copy() for w in model.wv],
        [w.copy() for w in model.wo],
        model.wout.copy(),
        None if model.adapter is None else model.adapter.copy(),
    )


def layer0_adapter(model: TinyModel, seed: int) -> TinyModel:
    """R3: the #60 adapter construction (dense, scale 0.05) added to Wo_0 instead of Wo_1."""
    out = _copy(model)
    out.wo[0] = out.wo[0] + _rng_matrix(seed, "adapter", (D_MODEL, D_MODEL), 0.05)
    return out


def _matrices(model: TinyModel) -> Dict[str, np.ndarray]:
    m = {"emb": model.emb, "wout": model.wout}
    for i in range(N_LAYERS):
        m[f"wq{i}"] = model.wq[i]
        m[f"wk{i}"] = model.wk[i]
        m[f"wv{i}"] = model.wv[i]
        m[f"wo{i}"] = model.wo[i]
    return m


def _set(model: TinyModel, name: str, value: np.ndarray) -> None:
    if name in ("emb", "wout"):
        setattr(model, name, value)
    else:
        getattr(model, name[:2])[int(name[2:])] = value


def global_perturbation(model: TinyModel, eps: float, seed: int) -> TinyModel:
    """R4: M -> M + eps * (||M||_F / ||G||_F) * G for every base matrix; G depends on (seed, name) only."""
    out = _copy(model)
    for name, m in _matrices(model).items():
        g = _rng_matrix(seed, f"r4-{name}", m.shape, 1.0)
        _set(out, name, m + eps * (np.linalg.norm(m) / np.linalg.norm(g)) * g)
    return out


def fp16_round(model: TinyModel) -> TinyModel:
    """R6: every base matrix rounded to float16 and back."""
    out = _copy(model)
    for name, m in _matrices(model).items():
        _set(out, name, m.astype(np.float16).astype(np.float64))
    return out


def cross_model(seed: int) -> TinyModel:
    """R5: an independently initialised receiver (T-K transport proper)."""
    return build_model(seed)


# ---------------------------------------------------------------------------
# A1: exact dependency-cone rule
# ---------------------------------------------------------------------------
def dependency_cone_unchanged(sender: TinyModel, receiver: TinyModel, prefix_tokens: np.ndarray) -> bool:
    toks = np.unique(np.asarray(prefix_tokens))
    if not np.array_equal(sender.emb[toks], receiver.emb[toks]):
        return False
    for w in ("wq", "wk", "wv"):
        if not np.array_equal(getattr(sender, w)[0], getattr(receiver, w)[0]):
            return False
    if not np.array_equal(sender._wo(0), receiver._wo(0)):
        return False
    for w in ("wk", "wv"):
        if not np.array_equal(getattr(sender, w)[1], getattr(receiver, w)[1]):
            return False
    return True


# ---------------------------------------------------------------------------
# A2: the receiver-activation bound
# ---------------------------------------------------------------------------
def _spec(m: np.ndarray) -> float:
    return float(np.linalg.norm(m, 2))


@dataclass
class RevisionNorms:
    """Per-revision (amortised) quantities: spectral norms of receiver weights and of weight deltas."""

    d_emb_rows: np.ndarray
    rw: Dict[str, float]  # receiver weight spectral norms
    dw: Dict[str, float]  # delta spectral norms
    wout_col_max: float
    flops: int


def revision_norms(sender: TinyModel, receiver: TinyModel) -> RevisionNorms:
    d = D_MODEL
    rw = {
        "wq0": _spec(receiver.wq[0]), "wk0": _spec(receiver.wk[0]), "wv0": _spec(receiver.wv[0]),
        "wo0": _spec(receiver._wo(0)), "wq1": _spec(receiver.wq[1]), "wk1": _spec(receiver.wk[1]),
        "wv1": _spec(receiver.wv[1]), "wo1": _spec(receiver._wo(1)),
    }
    dw = {
        "wq0": _spec(receiver.wq[0] - sender.wq[0]), "wk0": _spec(receiver.wk[0] - sender.wk[0]),
        "wv0": _spec(receiver.wv[0] - sender.wv[0]), "wo0": _spec(receiver._wo(0) - sender._wo(0)),
        "wk1": _spec(receiver.wk[1] - sender.wk[1]), "wv1": _spec(receiver.wv[1] - sender.wv[1]),
    }
    d_emb_rows = np.linalg.norm(receiver.emb - sender.emb, axis=1)
    wout_col_max = float(np.linalg.norm(receiver.wout, axis=0).max())
    vocab = receiver.emb.shape[0]
    flops = (len(rw) + len(dw)) * SVD_FLOPS_PER_MATRIX + len(dw) * d * d + 3 * vocab * d + 2 * d * vocab
    return RevisionNorms(d_emb_rows, rw, dw, wout_col_max, int(flops))


def _soft(delta: float) -> float:
    """Bound on ||p' - p||_1 when every score moves by at most ``delta``."""
    if 2.0 * delta >= np.log(3.0):  # expm1(2*delta) >= 2: the trivial bound wins (avoids expm1 overflow)
        return 2.0
    return min(float(np.expm1(2.0 * delta)), 2.0)


def _centered_max(v: np.ndarray) -> float:
    return float(np.linalg.norm(v - v.mean(axis=0, keepdims=True), axis=1).max())


def receiver_activation_bound(
    rev: RevisionNorms,
    prefix_tokens: np.ndarray,
    sender_stats: Dict[str, List[float]],
    cached_kv: KV,
    suffix_traces: Sequence[LayerTrace],
) -> Tuple[float, Dict[str, float], int]:
    """Return (B, intermediates, per-case FLOPs). B bounds every suffix logit error (see preregistration)."""
    H, Q, A = sender_stats["H"], sender_stats["Q"], sender_stats["A"]
    rw, dw = rev.rw, rev.dw
    P = int(len(prefix_tokens))
    S = int(suffix_traces[0].q.shape[0])
    # prefix stage
    e0 = float(rev.d_emb_rows[np.asarray(prefix_tokens)].max())
    dK0 = e0 * rw["wk0"] + H[0] * dw["wk0"]
    dV0 = e0 * rw["wv0"] + H[0] * dw["wv0"]
    dQ0 = e0 * rw["wq0"] + H[0] * dw["wq0"]
    kmax0 = float(np.linalg.norm(cached_kv[0][0], axis=1).max())
    r0 = _centered_max(cached_kv[0][1])
    delta0 = ((Q[0] + dQ0) * dK0 + dQ0 * kmax0) / SQRT_D
    da0 = dV0 + _soft(delta0) * r0
    e1 = e0 + da0 * rw["wo0"] + A[0] * dw["wo0"]
    dK1 = e1 * rw["wk1"] + H[1] * dw["wk1"]
    dV1 = e1 * rw["wv1"] + H[1] * dw["wv1"]
    # suffix stage (observed receiver run on the stale cache)
    t0, t1 = suffix_traces[0], suffix_traces[1]
    qs0 = float(np.linalg.norm(t0.q, axis=1).max())
    d0s = qs0 * dK0 / SQRT_D
    ds0 = dV0 + _soft(d0s) * _centered_max(t0.v_all)
    s1 = ds0 * rw["wo0"]
    DK = max(dK1, s1 * rw["wk1"])
    DV = max(dV1, s1 * rw["wv1"])
    dq = s1 * rw["wq1"]
    qs1 = float(np.linalg.norm(t1.q, axis=1).max())
    kmax1o = float(np.linalg.norm(t1.k_all, axis=1).max())
    d1s = (qs1 * DK + dq * (kmax1o + DK)) / SQRT_D
    ds1 = DV + _soft(d1s) * _centered_max(t1.v_all)
    s2 = s1 + ds1 * rw["wo1"]
    B = s2 * rev.wout_col_max
    d = D_MODEL
    flops = P + 6 * P * d + N_LAYERS * (2 * S * d + 6 * (P + S) * d) + 100
    inter = {"e0": e0, "e1": e1, "dK0": dK0, "dV0": dV0, "dK1": dK1, "dV1": dV1, "s1": s1, "s2": s2}
    return float(B), inter, int(flops)


def bound_is_sound(actual: float, bound: float) -> bool:
    return actual <= bound + SOUNDNESS_ABS_SLACK + SOUNDNESS_REL_SLACK * bound


# ---------------------------------------------------------------------------
# A3 / A3r: selective top-layer re-prefill
# ---------------------------------------------------------------------------
def tlr_h_kv(receiver: TinyModel, stale_kv: KV, sender_h1: np.ndarray) -> KV:
    """DroidSpeak-style: keep stale layer-0 KV, recompute layer-1 KV from the sender's stored layer-1 input."""
    return [(stale_kv[0][0], stale_kv[0][1]), (sender_h1 @ receiver.wk[1], sender_h1 @ receiver.wv[1])]


def tlr_r_kv(receiver: TinyModel, stale_kv: KV, prefix_tokens: np.ndarray) -> KV:
    """Recompute layer-1 KV from the receiver's own layer-0 pass over the stale layer-0 KV."""
    toks = np.asarray(prefix_tokens)
    P = len(toks)
    h0 = receiver.emb[toks] + _positions(np.arange(P))
    q = h0 @ receiver.wq[0]
    scores = (q @ stale_kv[0][0].T) / np.sqrt(D_MODEL)
    mask = np.arange(P)[None, :] <= np.arange(P)[:, None]
    scores = np.where(mask, scores, -1e30)
    scores = scores - scores.max(axis=1, keepdims=True)
    w = np.exp(scores)
    w = w / w.sum(axis=1, keepdims=True)
    h1 = h0 + (w @ stale_kv[0][1]) @ receiver._wo(0)
    return [(stale_kv[0][0], stale_kv[0][1]), (h1 @ receiver.wk[1], h1 @ receiver.wv[1])]


# ---------------------------------------------------------------------------
# cost model (preregistered analytic FLOPs)
# ---------------------------------------------------------------------------
def flops_full(P: int, d: int = D_MODEL) -> int:
    return 3 * 2 * P * d * d + 2 * P * (P + 1) * d + 2 * P * d * d + 2 * 2 * P * d * d


def flops_tlr_h(P: int, d: int = D_MODEL) -> int:
    return 2 * 2 * P * d * d


def flops_tlr_r(P: int, d: int = D_MODEL) -> int:
    return 2 * P * d * d + 2 * P * (P + 1) * d + 2 * P * d * d + 2 * 2 * P * d * d


def max_abs(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a - b).max())


def top2_margin(logits: np.ndarray) -> np.ndarray:
    s = np.sort(logits, axis=1)
    return s[:, -1] - s[:, -2]
