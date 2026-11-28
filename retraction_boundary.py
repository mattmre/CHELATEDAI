"""W22-09: retraction boundary — exact counterfactual deletion vs replay.

Preregistration: ``artifacts/pv7_w23b_w22_09/PREREGISTRATION.md``.

Three parts:

* Part A wraps the unmodified bundle #60 KV cache (``chelated_ssd_kv_repo_cache``) as the treatment and
  adds a flat-file checkpoint+replay baseline, a no-replay splice probe and a key-collision probe.
* Part B wraps the unmodified EK5 store (``ek5_v3_stratified``) used by bundles #51 and #62 as the treatment
  and adds rebuild / geometric-checkpoint baselines and two collision probes.
* Part C is a CONSTRUCTED affine recurrent state (no bundle implements one): segment tree over affine maps,
  geometric checkpoints, subtractive and inversion retraction arms, and a numerical-range (Crouzeix) check.
"""

from __future__ import annotations

import copy
import dataclasses
import pickle
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

CROUZEIX_PALENCIA_CONSTANT = 1.0 + np.sqrt(2.0)  # theorem (Crouzeix & Palencia, 2017)
CROUZEIX_CONJECTURED_CONSTANT = 2.0  # conjecture, not used as the bound
INV_COND_LIMIT = 1e8
EXACT_TOL = 1e-9


# ===========================================================================
# Part C — constructed affine recurrent state x_{t+1} = A_t x_t + b_t
# ===========================================================================
REGIMES = ("normal-TV", "nonnormal-TV", "nonnormal-LTI", "singular-TV")


def _orthogonal(rng: np.random.Generator, d: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.standard_normal((d, d)))
    return q * np.sign(np.diag(r))[None, :]


def _nonnormal(rng: np.random.Generator, d: int) -> np.ndarray:
    n = np.triu(rng.standard_normal((d, d)) * (1.5 / np.sqrt(d)), k=1)
    return 0.9 * np.eye(d) + n


def generate_regime(regime: str, d: int, T: int, seed: int) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Return (As (T,d,d), bs (T,d), meta). Deterministic in (regime, d, T, seed)."""
    if regime not in REGIMES:
        raise ValueError(f"unknown regime {regime!r}")
    rng = np.random.default_rng([seed, d, T, REGIMES.index(regime)])
    meta: Dict[str, Any] = {"regime": regime, "d": d, "T": T, "seed": seed, "singular_steps": []}
    if regime == "normal-TV":
        As = np.stack([0.97 * _orthogonal(rng, d) for _ in range(T)])
    elif regime == "nonnormal-TV":
        As = np.stack([_nonnormal(rng, d) for _ in range(T)])
    elif regime == "nonnormal-LTI":
        A = _nonnormal(rng, d)
        As = np.broadcast_to(A, (T, d, d)).copy()
    else:  # singular-TV
        As = np.stack([0.97 * _orthogonal(rng, d) for _ in range(T)])
        n_sing = max(1, int(round(0.05 * T)))
        steps = np.sort(rng.choice(np.arange(1, T - 1), size=n_sing, replace=False))
        for s in steps:
            u = rng.standard_normal(d)
            u /= np.linalg.norm(u)
            As[s] = 0.97 * (np.eye(d) - np.outer(u, u)) @ _orthogonal(rng, d)
        meta["singular_steps"] = [int(s) for s in steps]
    bs = rng.standard_normal((T, d))
    return As, bs, meta


def replay(
    As: np.ndarray,
    bs: np.ndarray,
    x0: np.ndarray,
    *,
    skip: Optional[int] = None,
    zero_input: Optional[int] = None,
    start: int = 0,
) -> np.ndarray:
    """Apply steps ``start..T-1`` to ``x0``. ``skip`` replaces (A_k,b_k) by (I,0); ``zero_input`` sets b_k=0."""
    x = np.array(x0, dtype=np.float64, copy=True)
    for t in range(start, As.shape[0]):
        if t == skip:
            continue
        if t == zero_input:
            x = As[t] @ x
        else:
            x = As[t] @ x + bs[t]
    return x


def states_trajectory(As: np.ndarray, bs: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """Return X (T+1, d) with X[p] = state after p steps."""
    T, d = bs.shape
    X = np.empty((T + 1, d))
    X[0] = x0
    for t in range(T):
        X[t + 1] = As[t] @ X[t] + bs[t]
    return X


class AffineSegmentTree:
    """Ordered segment tree over affine maps f_t(x) = A_t x + b_t; node = f_r o ... o f_l."""

    def __init__(self, As: np.ndarray, bs: np.ndarray) -> None:
        T, d = bs.shape
        size = 1
        while size < T:
            size *= 2
        self.T, self.d, self.size = T, d, size
        self.M = np.broadcast_to(np.eye(d), (2 * size, d, d)).copy()
        self.v = np.zeros((2 * size, d))
        self.M[size : size + T] = As
        self.v[size : size + T] = bs
        for i in range(size - 1, 0, -1):
            self._pull(i)
        self.matmuls_last_update = 0

    def _pull(self, i: int) -> None:
        left, right = 2 * i, 2 * i + 1
        self.M[i] = self.M[right] @ self.M[left]
        self.v[i] = self.M[right] @ self.v[left] + self.v[right]

    def update(self, k: int, A: np.ndarray, b: np.ndarray) -> None:
        i = self.size + k
        self.M[i] = A
        self.v[i] = b
        i //= 2
        n = 0
        while i >= 1:
            self._pull(i)
            n += 1
            i //= 2
        self.matmuls_last_update = n

    def apply_root(self, x0: np.ndarray) -> np.ndarray:
        return self.M[1] @ x0 + self.v[1]

    def retained_bytes(self) -> int:
        # every allocated node 1..2*size-1 (node 0 unused); padding leaves are identities and are counted
        return int(8 * (2 * self.size - 1) * (self.d * self.d + self.d))


def geometric_checkpoint_positions(T: int) -> List[int]:
    """Positions {0} U {T - 2^j >= 0}; a checkpoint at p stores the state after p steps."""
    pos = {0}
    j = 0
    while T - (1 << j) >= 0:
        pos.add(T - (1 << j))
        j += 1
    return sorted(pos)


def suffix_products(As: np.ndarray, bs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Phi[k] = A_{T-1}...A_{k+1}; c[k] = contribution of b_{k+1..T-1} (so x_T = Phi[k] x_{k+1} + c[k])."""
    T, d = bs.shape
    Phi = np.empty((T, d, d))
    c = np.empty((T, d))
    Phi[T - 1] = np.eye(d)
    c[T - 1] = 0.0
    for k in range(T - 2, -1, -1):
        Phi[k] = Phi[k + 1] @ As[k + 1]
        c[k] = Phi[k + 1] @ bs[k + 1] + c[k + 1]
    return Phi, c


def binary_powers(A: np.ndarray, T: int) -> List[np.ndarray]:
    out = [A.copy()]
    while (1 << len(out)) <= T:
        out.append(out[-1] @ out[-1])
    return out


def apply_power(powers: Sequence[np.ndarray], n: int, b: np.ndarray) -> Tuple[np.ndarray, int]:
    y = np.array(b, copy=True)
    j = 0
    mv = 0
    while n:
        if n & 1:
            y = powers[j] @ y
            mv += 1
        n >>= 1
        j += 1
    return y, mv


def inv_retract_skip(
    x_T: np.ndarray,
    Phi_k: np.ndarray,
    c_k: np.ndarray,
    A_k: np.ndarray,
    b_k: np.ndarray,
    *,
    certified: bool,
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Skip-deletion of step k from the compressed state by inverting Phi_k A_k.

    x_T = Phi_k (A_k x_k + b_k) + c_k. Recover x_k, then x_T' = Phi_k x_k + c_k.
    Certified: refuse when cond(Phi_k A_k) >= INV_COND_LIMIT. Uncertified: least-squares (min-norm) solve.
    """
    G = Phi_k @ A_k
    cond = float(np.linalg.cond(G))
    info = {"cond": cond, "refused": False}
    if certified and not (np.isfinite(cond) and cond < INV_COND_LIMIT):
        info["refused"] = True
        return None, info
    rhs = x_T - c_k - Phi_k @ b_k
    if certified:
        x_k = np.linalg.solve(G, rhs)
    else:
        x_k = np.linalg.lstsq(G, rhs, rcond=None)[0]
    return Phi_k @ x_k + c_k, info


def rel_err(a: np.ndarray, ref: np.ndarray) -> float:
    den = float(np.linalg.norm(ref))
    num = float(np.linalg.norm(np.asarray(a) - ref))
    return num / den if den > 0 else num


def singular_collision_pair(As: np.ndarray, bs: np.ndarray, s: int, alpha: float = 1.0) -> Dict[str, Any]:
    """Build H' differing from H only in b_{s-1} along the null direction of A_s.

    Both histories have the same x_T (to rounding) but different skip-counterfactuals for step s.
    """
    U_, S_, Vt = np.linalg.svd(As[s])
    null_dir = Vt[-1]
    bs2 = bs.copy()
    bs2[s - 1] = bs2[s - 1] + alpha * null_dir
    return {"bs_prime": bs2, "null_singular_value": float(S_[-1])}


# ===========================================================================
# Crouzeix / numerical range
# ===========================================================================
def numerical_range_boundary(A: np.ndarray, n_angles: int = 720) -> np.ndarray:
    """Boundary points of W(A): v* A v for v the top eigenvector of Hermitian part of e^{i theta} A."""
    A = np.asarray(A, dtype=np.complex128)
    pts = np.empty(n_angles, dtype=np.complex128)
    for i, th in enumerate(np.linspace(0.0, 2.0 * np.pi, n_angles, endpoint=False)):
        R = np.exp(1j * th) * A
        H = 0.5 * (R + R.conj().T)
        w, V = np.linalg.eigh(H)
        v = V[:, -1]
        pts[i] = np.vdot(v, A @ v)
    return pts


def poly_matrix(coeffs: Sequence[complex], A: np.ndarray) -> np.ndarray:
    """Horner evaluation of p(A) = sum_j coeffs[j] A^j."""
    d = A.shape[0]
    P = np.zeros((d, d), dtype=np.complex128)
    eye = np.eye(d, dtype=np.complex128)
    for c in reversed(list(coeffs)):
        P = P @ A + c * eye
    return P


def crouzeix_ratio_power(A: np.ndarray, n: int, boundary: np.ndarray) -> Dict[str, float]:
    """||A^n|| / max_{dW}|z^n| = ||A^n|| / w(A)^n, plus spectral-radius proxy ratio."""
    An = np.linalg.matrix_power(np.asarray(A, dtype=np.float64), n)
    norm = float(np.linalg.norm(An, 2))
    w = float(np.max(np.abs(boundary)))
    rho = float(np.max(np.abs(np.linalg.eigvals(A))))
    return {
        "norm": norm,
        "w": w,
        "rho": rho,
        "ratio_W": norm / (w ** n) if w > 0 else float("inf"),
        "ratio_rho": norm / (rho ** n) if rho > 0 else float("inf"),
    }


def crouzeix_ratio_poly(A: np.ndarray, coeffs: Sequence[complex], boundary: np.ndarray) -> float:
    pA = poly_matrix(coeffs, np.asarray(A, dtype=np.complex128))
    pz = np.polyval(list(reversed(list(coeffs))), boundary)
    den = float(np.max(np.abs(pz)))
    return float(np.linalg.norm(pA, 2)) / den if den > 0 else float("inf")


def crouzeix_check(
    A: np.ndarray,
    *,
    max_power: int = 64,
    n_random_polys: int = 32,
    seed: int = 0,
    n_angles: int = 720,
) -> Dict[str, Any]:
    """Empirical Crouzeix check on one matrix with boundary refinement near the bound."""
    boundary = numerical_range_boundary(A, n_angles)
    rng = np.random.default_rng(seed)
    pow_rows = [crouzeix_ratio_power(A, n, boundary) for n in range(1, max_power + 1)]
    poly_ratios: List[float] = []
    for _ in range(n_random_polys):
        deg = int(rng.integers(1, 9))
        coeffs = rng.standard_normal(deg + 1) + 1j * rng.standard_normal(deg + 1)
        poly_ratios.append(crouzeix_ratio_poly(A, coeffs, boundary))
    all_ratios = [r["ratio_W"] for r in pow_rows] + poly_ratios
    max_ratio = float(max(all_ratios))
    refined = False
    if max_ratio > 0.999 * CROUZEIX_CONJECTURED_CONSTANT:
        boundary = numerical_range_boundary(A, n_angles * 8)
        refined = True
        pow_rows = [crouzeix_ratio_power(A, n, boundary) for n in range(1, max_power + 1)]
        rng = np.random.default_rng(seed)
        poly_ratios = []
        for _ in range(n_random_polys):
            deg = int(rng.integers(1, 9))
            coeffs = rng.standard_normal(deg + 1) + 1j * rng.standard_normal(deg + 1)
            poly_ratios.append(crouzeix_ratio_poly(A, coeffs, boundary))
        all_ratios = [r["ratio_W"] for r in pow_rows] + poly_ratios
        max_ratio = float(max(all_ratios))
    return {
        "w": pow_rows[0]["w"],
        "rho": pow_rows[0]["rho"],
        "norm2": float(np.linalg.norm(A, 2)),
        "max_ratio_W": max_ratio,
        "max_power_ratio_W": float(max(r["ratio_W"] for r in pow_rows)),
        "max_poly_ratio_W": float(max(poly_ratios)) if poly_ratios else None,
        "max_power_ratio_rho": float(max(r["ratio_rho"] for r in pow_rows)),
        "peak_power_norm": float(max(r["norm"] for r in pow_rows)),
        "bound_1_plus_sqrt2_holds": bool(max_ratio <= CROUZEIX_PALENCIA_CONSTANT),
        "conjectured_2_holds": bool(max_ratio <= CROUZEIX_CONJECTURED_CONSTANT),
        "refined": refined,
        "n_angles": int(boundary.shape[0]),
    }


# ===========================================================================
# Part A — bundle #60 KV cache
# ===========================================================================
def kv_prompt(seed: int, n_blocks: int, tag: str = "p") -> List[Any]:
    from chelated_ssd_kv_repo_cache import VOCAB, FileVersion

    rng = np.random.default_rng([seed, n_blocks])
    blocks = []
    for j in range(n_blocks):
        ln = int(rng.integers(20, 37))
        toks = rng.integers(0, VOCAB, size=ln, dtype=np.int64)
        blocks.append(FileVersion(f"fv::{tag}::s{seed}::N{n_blocks}::b{j}", f"b{j}.py", toks, 0))
    return blocks


def kv_to_array(kv: List[Tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    return np.stack([x for pair in kv for x in pair])


def flat_checkpoint_write(path: Path, kv: List[Tuple[np.ndarray, np.ndarray]], tokens: np.ndarray) -> int:
    arr = kv_to_array(kv)
    np.save(path.with_suffix(".kv.npy"), arr)
    np.save(path.with_suffix(".tok.npy"), np.asarray(tokens, dtype=np.int64))
    return int(path.with_suffix(".kv.npy").stat().st_size + path.with_suffix(".tok.npy").stat().st_size)


def flat_checkpoint_delete(model: Any, path: Path, blocks: Sequence[Any], k: int, out_path: Path) -> Tuple[np.ndarray, List]:
    """B-FLAT: read prefix rows of the flat KV, recompute the suffix of the counterfactual prompt, persist."""
    from chelated_ssd_kv_repo_cache import N_LAYERS

    arr = np.load(path.with_suffix(".kv.npy"), mmap_mode="r")
    starts = np.cumsum([0] + [len(b.tokens) for b in blocks])
    cf = [b for j, b in enumerate(blocks) if j != k]
    if k < len(blocks) - 1:
        keep = int(starts[k])
        suffix = np.concatenate([b.tokens for b in blocks[k + 1 :]])
    else:  # deleted the last block: recompute the new last block to get its logits (as the treatment does)
        keep = int(starts[k - 1]) if k > 0 else 0
        suffix = cf[-1].tokens if cf else np.zeros(0, dtype=np.int64)
    past = None
    if keep > 0:
        past = [(np.array(arr[2 * i, :keep]), np.array(arr[2 * i + 1, :keep])) for i in range(N_LAYERS)]
    lg, new_kv = model.forward(suffix, past=past)
    if past is None:
        full = new_kv
    else:
        full = [(np.concatenate([past[i][0], new_kv[i][0]]), np.concatenate([past[i][1], new_kv[i][1]])) for i in range(N_LAYERS)]
    np.save(out_path, kv_to_array(full))
    return lg[-1], full


def splice_probe(model: Any, blocks: Sequence[Any], k: int, kv_full: List, ref_kv: List) -> Dict[str, Any]:
    """Delete block k's KV rows without any recomputation; measure the dirty cone vs the true counterfactual."""
    from chelated_ssd_kv_repo_cache import N_LAYERS

    starts = np.cumsum([0] + [len(b.tokens) for b in blocks])
    s, e = int(starts[k]), int(starts[k + 1])
    spliced = [(np.concatenate([kv_full[i][0][:s], kv_full[i][0][e:]]), np.concatenate([kv_full[i][1][:s], kv_full[i][1][e:]])) for i in range(N_LAYERS)]
    n_suffix = spliced[0][0].shape[0] - s
    per_layer = []
    for i in range(N_LAYERS):
        dk = np.max(np.abs(spliced[i][0][s:] - ref_kv[i][0][s:]), axis=1) if n_suffix > 0 else np.zeros(0)
        dv = np.max(np.abs(spliced[i][1][s:] - ref_kv[i][1][s:]), axis=1) if n_suffix > 0 else np.zeros(0)
        dirty = np.maximum(dk, dv) > EXACT_TOL
        per_layer.append({"layer": i, "suffix_rows": int(n_suffix), "dirty_rows": int(dirty.sum()),
                          "max_abs": float(np.max(np.maximum(dk, dv))) if n_suffix else 0.0})
    prefix_exact = all(
        float(np.max(np.abs(spliced[i][j][:s] - ref_kv[i][j][:s]))) <= EXACT_TOL if s > 0 else True
        for i in range(N_LAYERS) for j in range(2)
    )
    return {"k": k, "suffix_rows": int(n_suffix), "per_layer": per_layer, "prefix_rows_exact": bool(prefix_exact)}


# ===========================================================================
# Part B — EK5 store used by bundles #51 / #62
# ===========================================================================
@dataclasses.dataclass
class EK5Log:
    """Ordered log: turns[i] = (events, attestations) introduced at turn i."""

    name: str
    turns: List[Tuple[List[Any], List[Any]]]

    @property
    def n_events(self) -> int:
        return sum(len(t[0]) for t in self.turns)

    def attestation_turn(self) -> Dict[str, int]:
        return {a.attestation_id: i for i, (_e, atts) in enumerate(self.turns) for a in atts}

    def flatten(self, start: int = 0, drop: Optional[str] = None) -> Tuple[List[Any], List[Any]]:
        evs: List[Any] = []
        atts: List[Any] = []
        for e, a in self.turns[start:]:
            evs.extend(e)
            atts.extend(x for x in a if x.attestation_id != drop)
        return evs, atts


def beam_log(replicas: int, seed: int = 5101) -> EK5Log:
    from chelated_beam_128k_replay import _turn_to_ek5, build_deterministic_beam_128k_fixture

    convs = build_deterministic_beam_128k_fixture(seed=seed)
    rows = []
    for r in range(replicas):
        for ci, c in enumerate(convs):
            for ti, t in enumerate(c.derivative_turns):
                t2 = dataclasses.replace(
                    t,
                    event_id=f"r{r}:{t.event_id}",
                    parent_event_id=(f"r{r}:{t.parent_event_id}" if t.parent_event_id else None),
                    entity=f"r{r}:{t.entity}",
                    lineage_id=(f"r{r}:{t.lineage_id}" if t.lineage_id else None),
                )
                ev, edge = _turn_to_ek5(t2, f"r{r}:{c.tenant_id}")
                rows.append(((t.tx_time, r, ci, ti), ev, edge))
    rows.sort(key=lambda x: x[0])
    return EK5Log(f"W-BEAM-R{replicas}", [([ev], [edge] if edge is not None else []) for _k, ev, edge in rows])


def chain_log(n: int) -> EK5Log:
    from ek5_v3_stratified import EdgeAttestation, StratifiedEvent

    turns: List[Tuple[List[Any], List[Any]]] = []
    turns.append(([StratifiedEvent("e0", entity="chain", slot="s", claim_value="v", valid_time=0, tx_time=0, authority_tier=2)], []))
    for i in range(1, n):
        ev = StratifiedEvent(f"e{i}", entity="chain", slot=f"s{i}", claim_value="v", valid_time=i, tx_time=i, authority_tier=2)
        att = EdgeAttestation(f"att:e{i}:e{i - 1}", f"e{i}", f"e{i - 1}", "DERIVES_CURRENT", valid_time=i, tx_time=i)
        turns.append(([ev], [att]))
    r = StratifiedEvent("refuter", entity="chain", slot="r", valid_time=n, tx_time=n, authority_tier=5)
    turns.append(([r], [EdgeAttestation("att:refuter:e0", "refuter", "e0", "REFUTE", valid_time=n, tx_time=n)]))
    return EK5Log(f"W-CHAIN-n{n}", turns)


def new_store() -> Any:
    from ek5_v3_stratified import StratifiedEvidenceStore

    return StratifiedEvidenceStore(strict_dag_supersede=False, buffer_pending=False)


def build_incremental(log: EK5Log) -> Any:
    s = new_store()
    for evs, atts in log.turns:
        s.apply_mutation(evs, atts)
    return s


def build_batch(log: EK5Log, start_store: Any = None, start: int = 0, drop: Optional[str] = None) -> Any:
    s = start_store if start_store is not None else new_store()
    evs, atts = log.flatten(start, drop)
    s.apply_mutation(evs, atts)
    return s


def semantic_diffs(a: Any, b: Any) -> int:
    n = 0
    ids = set(a.events) | set(b.events)
    for v in ids:
        for field in ("val", "cur", "eligibility"):
            if getattr(a, field).get(v) != getattr(b, field).get(v):
                n += 1
    return n


def store_bytes(store: Any) -> int:
    return len(pickle.dumps(store, protocol=5))


def log_bytes(log: EK5Log) -> int:
    return len(pickle.dumps(log.turns, protocol=5))


def geo_ladder(n_turns: int) -> List[int]:
    out: List[int] = []
    j = 1
    while True:
        p = int(np.floor(n_turns * (1.0 - 2.0 ** (-j))))
        if p >= n_turns or (out and p == out[-1]):
            break
        out.append(p)
        j += 1
    return out


def build_geo_checkpoints(log: EK5Log, budget_bytes: int) -> Tuple[Dict[int, bytes], int]:
    """Greedy over the geometric ladder while log + checkpoints <= budget."""
    total = log_bytes(log)
    ckpts: Dict[int, bytes] = {}
    for p in geo_ladder(len(log.turns)):
        st = new_store()
        if p > 0:
            evs, atts = [], []
            for e, a in log.turns[:p]:
                evs.extend(e)
                atts.extend(a)
            st.apply_mutation(evs, atts)
        blob = pickle.dumps(st, protocol=5)
        if total + len(blob) > budget_bytes:
            break
        ckpts[p] = blob
        total += len(blob)
    return ckpts, total


def geo_ckpt_delete(log: EK5Log, ckpts: Dict[int, bytes], att_turn: int, att_id: str) -> Tuple[Any, int]:
    usable = [p for p in ckpts if p <= att_turn]
    if usable:
        p = max(usable)
        st = pickle.loads(ckpts[p])
    else:
        p = 0
        st = new_store()
    return build_batch(log, st, p, att_id), p


def asof_collision_probe() -> Dict[str, Any]:
    """PEK5-ASOF: native attestation withdrawal leaves no withdrawal time; as_of before it loses the edge."""
    from ek5_v3_stratified import EdgeAttestation, StratifiedEvent

    def evs() -> List[Any]:
        return [
            StratifiedEvent("R", entity="x", slot="s", claim_value="a", valid_time=1, tx_time=1, authority_tier=2),
            StratifiedEvent("C", entity="x", slot="s2", claim_value="a", valid_time=2, tx_time=2, authority_tier=2),
            StratifiedEvent("X", entity="x", slot="s3", valid_time=5, tx_time=5, authority_tier=5),
        ]

    copy_att = EdgeAttestation("att:C:R", "C", "R", "COPY_OF", valid_time=2, tx_time=2)
    ref_att = EdgeAttestation("att:X:R", "X", "R", "REFUTE", valid_time=5, tx_time=5)
    h1 = new_store()
    h1.apply_mutation(evs(), [copy_att, ref_att])
    h1_before = dict(h1.eligibility)
    h1.apply_mutation(withdraw_attestation_ids=["att:C:R"])  # withdrawal "at tx 10": the API takes no time
    h2 = new_store()
    h2.apply_mutation(evs(), [ref_att])
    cutoff = 7
    snap1 = h1.as_of(10 ** 9, cutoff)
    snap2 = h2.as_of(10 ** 9, cutoff)
    truth1 = new_store()
    truth1.apply_mutation(evs(), [copy_att, ref_att])  # every item has tx <= 7
    truth2 = new_store()
    truth2.apply_mutation(evs(), [ref_att])
    att_record = h1.attestations["att:C:R"]
    return {
        "h1_eligibility_C_before_withdrawal": h1_before.get("C"),
        "h1_asof_tx7_C": snap1.eligibility.get("C"),
        "h2_asof_tx7_C": snap2.eligibility.get("C"),
        "truth_h1_tx7_C": truth1.eligibility.get("C"),
        "truth_h2_tx7_C": truth2.eligibility.get("C"),
        "withdrawn_record_fields": sorted(dataclasses.asdict(att_record).keys()),
        "withdrawn_record_has_withdrawal_time": any(k in dataclasses.asdict(att_record) for k in ("withdrawn_tx", "tx_end", "withdrawn_at")),
        "collision": snap1.eligibility.get("C") == snap2.eligibility.get("C"),
        "h1_asof_exact": snap1.eligibility.get("C") == truth1.eligibility.get("C"),
        "h2_asof_exact": snap2.eligibility.get("C") == truth2.eligibility.get("C"),
        "flag_raised": False,  # as_of has no return channel for a warning; checked: it returns a store only
    }


def digest_collision_probe(conv_index: int = 0, tamper_turn: str = ":aux_v2") -> Dict[str, Any]:
    """P51-DIGEST: #51 certifies replay exactness by canonical-digest equality; claim_value is not in it."""
    from chelated_beam_128k_replay import (
        _resolve_ek5_question,
        _snapshot_checkpoint,
        _turn_to_ek5,
        build_deterministic_beam_128k_fixture,
        restore_store_from_checkpoint,
    )
    from chelated_pv7_spine.ek5_reason_canonicalizer import export_canonical_ek5_audit_trace
    from ek5_v3_stratified import StratifiedEvidenceStore

    conv = build_deterministic_beam_128k_fixture(seed=5101)[conv_index]
    turns = list(conv.derivative_turns)
    checkpoint_step = 3  # #51 default

    inc = StratifiedEvidenceStore(strict_dag_supersede=False, buffer_pending=False)
    ckpt = None
    for step, t in enumerate(turns, start=1):
        ev, edge = _turn_to_ek5(t, conv.tenant_id)
        inc.apply_mutation([ev], [edge] if edge is not None else [])
        if step == checkpoint_step:
            ckpt = _snapshot_checkpoint(inc, conv.conversation_id, "derivative_only", step,
                                        export_canonical_ek5_audit_trace(inc).digest())
    assert ckpt is not None
    resumed = restore_store_from_checkpoint(ckpt)
    tampered_ids = []
    for t in turns[checkpoint_step:]:
        if t.event_id.endswith(tamper_turn):
            t = dataclasses.replace(t, claim_value=t.claim_value + "_DIVERGED")
            tampered_ids.append(t.event_id)
        ev, edge = _turn_to_ek5(t, conv.tenant_id)
        resumed.apply_mutation([ev], [edge] if edge is not None else [])
    d_final = export_canonical_ek5_audit_trace(inc).digest()
    d_resumed = export_canonical_ek5_audit_trace(resumed).digest()
    ans_ref = [_resolve_ek5_question(inc, conv.tenant_id, q) for q in conv.questions]
    ans_res = [_resolve_ek5_question(resumed, conv.tenant_id, q) for q in conv.questions]
    differing = [q.question_id for q, a, b in zip(conv.questions, ans_ref, ans_res) if a != b]
    bundle_reports_exact = d_final == d_resumed  # #51's checkpoint_replay_exact criterion
    return {
        "conversation_id": conv.conversation_id,
        "tampered_event_ids": tampered_ids,
        "checkpoint_step": checkpoint_step,
        "digest_equal": bundle_reports_exact,
        "bundle_checkpoint_replay_exact_would_report": bundle_reports_exact,
        "questions": len(conv.questions),
        "questions_with_different_answers": len(differing),
        "differing_question_ids": differing,
        "example_answers": [list(ans_ref[i]) + list(ans_res[i]) for i, q in enumerate(conv.questions) if q.question_id in differing][:2],
    }


def time_call(fn, *args, **kwargs) -> Tuple[Any, float]:
    t0 = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, (time.perf_counter() - t0) * 1000.0


def deep(x: Any) -> Any:
    return copy.deepcopy(x)
