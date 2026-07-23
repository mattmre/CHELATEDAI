"""GENERALITY (paper-hardening): does "trivial lstsq beats bounded corrector, bound cripples"
hold across DIFFERENT encoder upgrades? Same arena, vary the swap (new) model:
  mpnet (control, reproduces campaign) · bge-large-en-v1.5 (different family, 1024-d) · nomic-embed.
Reuses harness QueryEncoderDrift. Self-contained NDCG@10 (validated vs MTEB)."""
import os
import sys
import traceback

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np  # noqa: E402

WT = r"D:\GITHUB\CHELATEDAI\.claude\worktrees\gpu-campaigns"
sys.path.insert(0, WT)
os.chdir(WT)

from run_drift_recovery_experiment import load_mteb_data, _split_anchor_eval, canonicalize_id  # noqa: E402
from run_road_course_campaign import select_road_course_slice  # noqa: E402
from query_encoder_drift import QueryEncoderDrift  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

SEED = 42
RNG = np.random.default_rng(SEED)


def _norm(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True); n[n == 0] = 1.0
    return x / n


def ndcg(qv, qids, dv, did, qrels, k=10):
    dv, qv = _norm(dv), _norm(qv); out = []
    for v, qid in zip(qv, qids):
        rel = qrels.get(qid, {})
        if not rel or not any(r > 0 for r in rel.values()):
            continue
        sims = dv @ v
        top = np.argpartition(-sims, min(k, len(sims) - 1))[:k]; top = top[np.argsort(-sims[top])]
        dcg = sum(rel.get(did[i], 0.0) / np.log2(r + 2) for r, i in enumerate(top) if rel.get(did[i], 0.0) > 0)
        ideal = sorted([r for r in rel.values() if r > 0], reverse=True)[:k]
        idcg = sum(g / np.log2(r + 2) for r, g in enumerate(ideal))
        if idcg > 0:
            out.append(dcg / idcg)
    return float(np.mean(out))


def ridge(X, Y, lam):
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ Y)


def run(task, swap_model):
    corpus, queries, qrels = load_mteb_data(task)
    sc, sq, sqr = select_road_course_slice(corpus, queries, qrels, max_queries=100, sample_docs=1200, seed=SEED)
    _, eval_ids = _split_anchor_eval(sq, 0.4, SEED, active=True)  # match the campaign eval-split
    es = set(eval_ids)
    did = list(sc.keys()); qid = [q for q in sq.keys() if sqr.get(q) and canonicalize_id(q) in es]
    dtx = [sc[d] for d in did]; qtx = [sq[q] for q in qid]
    old = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    Do = np.asarray(old.encode(dtx, batch_size=128, show_progress_bar=False), dtype=np.float64)
    Qo = np.asarray(old.encode(qtx, batch_size=128, show_progress_bar=False), dtype=np.float64)
    drift = QueryEncoderDrift(store_dim=Do.shape[1], swap_model_name=swap_model, seed=SEED)
    Qd = np.asarray(drift.embed_queries(qtx), dtype=np.float64)
    Dor = np.asarray(drift.embed_queries(dtx), dtype=np.float64)
    base = ndcg(Qo, qid, Do, did, sqr); floor = ndcg(Qd, qid, Do, did, sqr); oracle = ndcg(Qd, qid, Dor, did, sqr)
    gap = max(oracle - floor, 1e-9)
    perm = RNG.permutation(len(did)); fit = perm[: len(did) // 2]
    Wr = ridge(Do[fit], Dor[fit], 1.0)
    nd_ridge = ndcg(Qd, qid, Do @ Wr, did, sqr)
    # bound ablation (near-identity, like our corrector)
    Wb = np.eye(Do.shape[1]) + 0.1 * (Wr - np.eye(Do.shape[1]))
    nd_b = ndcg(Qd, qid, Do @ Wb, did, sqr)
    print(f"\n### {task} | swap={swap_model}", flush=True)
    print(f"  base={base:.4f} floor={floor:.4f} oracle={oracle:.4f}", flush=True)
    print(f"  ridge(unbounded)  recovery={100*(nd_ridge-floor)/gap:5.1f}%  (NDCG {nd_ridge:.4f})", flush=True)
    print(f"  bounded α=0.1     recovery={100*(nd_b-floor)/gap:5.1f}%  (NDCG {nd_b:.4f})", flush=True)
    print(f"  PATTERN: drift catastrophic={floor < 0.1*base}  unbounded>>bounded={nd_ridge > nd_b + 0.05}", flush=True)


if __name__ == "__main__":
    swaps = ["all-mpnet-base-v2", "BAAI/bge-large-en-v1.5", "nomic-ai/nomic-embed-text-v1"]
    for sm in swaps:
        for task in ("SciFact", "NFCorpus"):
            try:
                run(task, sm)
            except Exception as e:
                print(f"\n### {task} | swap={sm} FAILED: {e}", flush=True)
                traceback.print_exc()
    print("\nDONE", flush=True)
