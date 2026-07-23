"""S1 (the one novel extension): manufacture (old-space, new-space) pairs WITHOUT the old model and
WITHOUT re-embedding docs — from a cheap doc-derived PSEUDO-QUERY embedded with the NEW encoder,
paired with the doc's cached OLD vector. Fit ridge, recover drift. Plugs Drift-Adapter's stated
"f_old unavailable" gap. Honest floor: no doc2query model is cached offline, so the pseudo-query is
the doc's OPENING text (a degenerate generator). A real doc2query would be the target.

Conditions (SciFact, exact arena): floor / anchor-InfoNCE(0.16) / oracle-pair ridge (85% ceiling) /
S1 pseudo-pair ridge (first-K-words) at several K / full-text-as-pseudo (≈oracle sanity)."""
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


def ridge(X, Y, lam=1.0):
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ Y)


def first_k_words(text, k):
    return " ".join(str(text).split()[:k])


def main():
    corpus, queries, qrels = load_mteb_data("SciFact")
    sc, sq, sqr = select_road_course_slice(corpus, queries, qrels, max_queries=100, sample_docs=1200, seed=SEED)
    did = list(sc.keys()); qid = [q for q in sq.keys() if sqr.get(q)]
    dtx = [sc[d] for d in did]; qtx = [sq[q] for q in qid]
    old = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    Do = np.asarray(old.encode(dtx, batch_size=128, show_progress_bar=False), dtype=np.float64)
    drift = QueryEncoderDrift(store_dim=Do.shape[1], swap_model_name="all-mpnet-base-v2", seed=SEED)
    Qd = np.asarray(drift.embed_queries(qtx), dtype=np.float64)
    Dor = np.asarray(drift.embed_queries(dtx), dtype=np.float64)   # oracle target = full doc re-embedded
    floor = ndcg(Qd, qid, Do, did, sqr); oracle = ndcg(Qd, qid, Dor, did, sqr)
    gap = max(oracle - floor, 1e-9)
    perm = RNG.permutation(len(did)); fit = perm[: len(did) // 2]
    print(f"=== S1 SciFact === floor={floor:.4f} oracle={oracle:.4f}  (anchor-InfoNCE campaign C3a≈0.161)", flush=True)

    def recover(target_new, tag):
        W = ridge(Do[fit], target_new[fit])
        nd = ndcg(Qd, qid, Do @ W, did, sqr)
        print(f"  {tag:28s} NDCG={nd:.4f}  recovery={100*(nd-floor)/gap:5.1f}% of oracle gap", flush=True)
        return nd

    # oracle-pair (re-embed full doc) = the 85% ceiling reference
    recover(Dor, "oracle-pair ridge (ceiling)")
    # S1: pseudo-query = first-K-words of the doc, embedded with NEW encoder, paired with cached OLD vec
    for kw in (12, 25, 50):
        pseudo = [first_k_words(t, kw) for t in dtx]
        Pn = np.asarray(drift.embed_queries(pseudo), dtype=np.float64)
        recover(Pn, f"S1 pseudo first-{kw}w")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
