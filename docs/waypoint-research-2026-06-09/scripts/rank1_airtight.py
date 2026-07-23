"""AIRTIGHT fair-baseline vs corrector: eval the fair-baseline (ridge) on the EXACT campaign
eval-split (anchor_fraction=0.4) so it is apples-to-apples with the campaign's C3a/C4a numbers,
on BOTH datasets. Compares to campaign: C3a SciFact 0.1609 / NFCorpus 0.0524."""
import os
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np  # noqa: E402

WT = r"D:\GITHUB\CHELATEDAI\.claude\worktrees\gpu-campaigns"
sys.path.insert(0, WT)
os.chdir(WT)

from run_drift_recovery_experiment import load_mteb_data, _split_anchor_eval  # noqa: E402
from run_road_course_campaign import select_road_course_slice  # noqa: E402
from query_encoder_drift import QueryEncoderDrift  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

SEED = 42
RNG = np.random.default_rng(SEED)
CAMPAIGN_C3A = {"SciFact": 0.1609, "NFCorpus": 0.0524}


def _norm(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True); n[n == 0] = 1.0
    return x / n


def ndcg(qv, qids, dv, did, qrels, k=10):
    # BINARY relevance to match the harness (ndcg_at_k over relevant_ids = {score>0}).
    dv, qv = _norm(dv), _norm(qv); out = []
    for v, qid in zip(qv, qids):
        rel = qrels.get(qid, {})
        relset = {d for d, s in rel.items() if float(s) > 0}
        if not relset:
            continue
        sims = dv @ v
        top = np.argpartition(-sims, min(k, len(sims) - 1))[:k]; top = top[np.argsort(-sims[top])]
        dcg = sum(1.0 / np.log2(r + 2) for r, i in enumerate(top) if did[i] in relset)
        idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(relset), k)))
        if idcg > 0:
            out.append(dcg / idcg)
    return float(np.mean(out))


def ridge(X, Y, lam):
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ Y)


def run(task):
    corpus, queries, qrels = load_mteb_data(task)
    sc, sq, sqr = select_road_course_slice(corpus, queries, qrels, max_queries=100, sample_docs=1200, seed=SEED)
    # EXACT campaign eval-split: anchor_fraction=0.4, drop the 40% anchors, eval on the 60%.
    from run_drift_recovery_experiment import canonicalize_id
    anchor_ids, eval_ids = _split_anchor_eval(sq, 0.4, SEED, active=True)
    eval_set = set(eval_ids)
    did = list(sc.keys())
    qid = [q for q in sq.keys() if sqr.get(q) and canonicalize_id(q) in eval_set]
    dtx = [sc[d] for d in did]; qtx = [sq[q] for q in qid]
    old = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    Do = np.asarray(old.encode(dtx, batch_size=128, show_progress_bar=False), dtype=np.float64)
    Qo = np.asarray(old.encode(qtx, batch_size=128, show_progress_bar=False), dtype=np.float64)
    drift = QueryEncoderDrift(store_dim=Do.shape[1], swap_model_name="all-mpnet-base-v2", seed=SEED)
    Qd = np.asarray(drift.embed_queries(qtx), dtype=np.float64)
    Dor = np.asarray(drift.embed_queries(dtx), dtype=np.float64)
    base = ndcg(Qo, qid, Do, did, sqr); floor = ndcg(Qd, qid, Do, did, sqr); oracle = ndcg(Qd, qid, Dor, did, sqr)
    gap = max(oracle - floor, 1e-9)
    perm = RNG.permutation(len(did)); fit = perm[: len(did) // 2]
    Wr = ridge(Do[fit], Dor[fit], 1.0)
    nd_r = ndcg(Qd, qid, Do @ Wr, did, sqr)
    Wb = np.eye(Do.shape[1]) + 0.1 * (Wr - np.eye(Do.shape[1]))
    nd_b = ndcg(Qd, qid, Do @ Wb, did, sqr)
    c3a = CAMPAIGN_C3A[task]
    print(f"\n### {task} (eval-split, {len(qid)} queries) ###", flush=True)
    print(f"  base={base:.4f} floor={floor:.4f} oracle={oracle:.4f}", flush=True)
    print(f"  campaign C3a        = {c3a:.4f}  recovery={100*(c3a-floor)/gap:5.1f}%", flush=True)
    print(f"  ridge (fair)        = {nd_r:.4f}  recovery={100*(nd_r-floor)/gap:5.1f}%", flush=True)
    print(f"  bounded α=0.1       = {nd_b:.4f}  recovery={100*(nd_b-floor)/gap:5.1f}%", flush=True)
    print(f"  -> fair beats corrector by {nd_r/max(c3a,1e-9):.1f}x", flush=True)


if __name__ == "__main__":
    for t in ("SciFact", "NFCorpus"):
        run(t)
    print("\nDONE", flush=True)
