"""Fair-baseline + capacity + bound-ablation sweep, in the EXACT campaign arena, BOTH datasets.
Questions this answers (operator-authorized, pushes understanding):
  1. Does a PROPER fair map (lstsq/ridge/low-rank/Procrustes) approach the oracle, or plateau? (ceiling)
  2. Which map type is best?  3. Did the BOUND (near-identity) cripple our corrector? (alpha-shrink ablation)
Reuses harness QueryEncoderDrift (exact projection). Self-contained NDCG@10 (validated vs MTEB)."""
import os
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np  # noqa: E402

WT = r"D:\GITHUB\CHELATEDAI\.claude\worktrees\gpu-campaigns"
sys.path.insert(0, WT)
os.chdir(WT)

from run_drift_recovery_experiment import load_mteb_data  # noqa: E402
from run_road_course_campaign import select_road_course_slice  # noqa: E402
from query_encoder_drift import QueryEncoderDrift  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

SEED = 42
K = 10
RNG = np.random.default_rng(SEED)


def _norm(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def ndcg(qv, qids, dv, did, qrels, k=K):
    dv, qv = _norm(dv), _norm(qv)
    out = []
    for v, qid in zip(qv, qids):
        rel = qrels.get(qid, {})
        if not rel or not any(r > 0 for r in rel.values()):
            continue
        sims = dv @ v
        top = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
        top = top[np.argsort(-sims[top])]
        dcg = sum(rel.get(did[i], 0.0) / np.log2(r + 2) for r, i in enumerate(top) if rel.get(did[i], 0.0) > 0)
        ideal = sorted([r for r in rel.values() if r > 0], reverse=True)[:k]
        idcg = sum(g / np.log2(r + 2) for r, g in enumerate(ideal))
        if idcg > 0:
            out.append(dcg / idcg)
    return float(np.mean(out))


def fit_lstsq(X, Y):
    W, *_ = np.linalg.lstsq(X, Y, rcond=None)
    return W


def fit_ridge(X, Y, lam):
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ Y)


def fit_lowrank(X, Y, r):
    W = fit_lstsq(X, Y)
    U, s, Vt = np.linalg.svd(W, full_matrices=False)
    s[r:] = 0.0
    return (U * s) @ Vt


def fit_procrustes(X, Y):
    U, _, Vt = np.linalg.svd(X.T @ Y, full_matrices=False)
    return U @ Vt


def run_dataset(task):
    corpus, queries, qrels = load_mteb_data(task)
    sc, sq, sqr = select_road_course_slice(corpus, queries, qrels, max_queries=100, sample_docs=1200, seed=SEED)
    did = list(sc.keys())
    qid = [q for q in sq.keys() if sqr.get(q)]
    dtx = [sc[d] for d in did]
    qtx = [sq[q] for q in qid]
    old = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    Do = np.asarray(old.encode(dtx, batch_size=128, show_progress_bar=False), dtype=np.float64)
    Qo = np.asarray(old.encode(qtx, batch_size=128, show_progress_bar=False), dtype=np.float64)
    drift = QueryEncoderDrift(store_dim=Do.shape[1], swap_model_name="all-mpnet-base-v2", seed=SEED)
    Qd = np.asarray(drift.embed_queries(qtx), dtype=np.float64)
    Dor = np.asarray(drift.embed_queries(dtx), dtype=np.float64)

    base = ndcg(Qo, qid, Do, did, sqr)
    floor = ndcg(Qd, qid, Do, did, sqr)
    oracle = ndcg(Qd, qid, Dor, did, sqr)
    print(f"\n=== {task} === base={base:.4f} floor={floor:.4f} oracle={oracle:.4f}", flush=True)

    perm = RNG.permutation(len(did))
    fit = perm[: len(did) // 2]
    Xf, Yf = Do[fit], Dor[fit]
    gap = max(oracle - floor, 1e-9)
    maps = {
        "lstsq-full": fit_lstsq(Xf, Yf),
        "ridge-1e-1": fit_ridge(Xf, Yf, 0.1),
        "ridge-1e0": fit_ridge(Xf, Yf, 1.0),
        "lowrank-64": fit_lowrank(Xf, Yf, 64),
        "lowrank-128": fit_lowrank(Xf, Yf, 128),
        "procrustes": fit_procrustes(Xf, Yf),
    }
    rows = []
    for name, W in maps.items():
        nd = ndcg(Qd, qid, Do @ W, did, sqr)
        rows.append((name, nd, 100 * (nd - floor) / gap))
    # BOUND ablation: shrink the best full map toward identity (our corrector was near-identity).
    Wfull = maps["lstsq-full"]
    for a in (0.05, 0.1, 0.25, 0.5):
        Wb = np.eye(Do.shape[1]) + a * (Wfull - np.eye(Do.shape[1]))
        nd = ndcg(Qd, qid, Do @ Wb, did, sqr)
        rows.append((f"bounded-a{a}", nd, 100 * (nd - floor) / gap))
    for name, nd, pct in rows:
        print(f"  {name:14s} NDCG={nd:.4f}  recovery={pct:5.1f}% of oracle gap", flush=True)
    print(f"  [campaign corrector C3a/C5 ~0.131 = ~{100*(0.131-floor)/gap:.0f}% on SciFact-equiv]", flush=True)


if __name__ == "__main__":
    for t in ("SciFact", "NFCorpus"):
        run_dataset(t)
    print("\nDONE", flush=True)
