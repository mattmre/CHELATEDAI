"""RANK 4 (+ Rank 1 Procrustes half): is old->new REALLY harder than new->old, or an artifact?

Reuses the harness's tested load_mteb_data. Self-contained NDCG@10 with a SANITY GATE:
old-vs-old NDCG must reproduce the campaign baseline (~0.819 SciFact) or the script is wrong.

Tests, on real MiniLM(old)/mpnet(new) embeddings:
  - drift floor: new queries vs old doc index (the catastrophic break)
  - NEW->OLD (Drift-Adapter, easy): map new queries -> old space, retrieve vs fixed old index
  - OLD->NEW (ours, hard):          map old docs    -> new space, retrieve new queries vs mapped docs
  - both maps = least-squares fit on a held-out paired-corpus sample (fair, same fit budget)
  - geometry: normalized residual each direction + linear CKA (is the asymmetry geometric?)
"""
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
from sentence_transformers import SentenceTransformer  # noqa: E402

RNG = np.random.default_rng(42)
TASK = "SciFact"
K = 10


def _norm(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def ndcg_at_k(qvecs, qids, dvecs, did_index, qrels, k=K):
    """Standard NDCG@10 averaged over queries with >=1 judged relevant doc."""
    dvecs = _norm(dvecs)
    qvecs = _norm(qvecs)
    scores = []
    for qv, qid in zip(qvecs, qids):
        rel = qrels.get(qid, {})
        if not rel or not any(v > 0 for v in rel.values()):
            continue
        sims = dvecs @ qv
        top = np.argpartition(-sims, min(k, len(sims) - 1))[:k]
        top = top[np.argsort(-sims[top])]
        dcg = 0.0
        for rank, di in enumerate(top):
            g = rel.get(did_index[di], 0.0)
            if g > 0:
                dcg += g / np.log2(rank + 2)
        ideal = sorted([v for v in rel.values() if v > 0], reverse=True)[:k]
        idcg = sum(g / np.log2(r + 2) for r, g in enumerate(ideal))
        if idcg > 0:
            scores.append(dcg / idcg)
    return float(np.mean(scores)), len(scores)


def fit_ls(src, dst):
    """Least-squares linear map src->dst (no intercept). Returns W: src @ W ~= dst."""
    W, *_ = np.linalg.lstsq(src, dst, rcond=None)
    return W


def residual(src, dst, W):
    pred = src @ W
    return float(np.linalg.norm(pred - dst) / (np.linalg.norm(dst) + 1e-9))


def linear_cka(X, Y):
    X = X - X.mean(0, keepdims=True)
    Y = Y - Y.mean(0, keepdims=True)
    xy = np.linalg.norm(X.T @ Y) ** 2
    xx = np.linalg.norm(X.T @ X)
    yy = np.linalg.norm(Y.T @ Y)
    return float(xy / (xx * yy + 1e-12))


def main():
    print(f"=== RANK 4 direction test on {TASK} ===", flush=True)
    corpus, queries, qrels = load_mteb_data(TASK)
    did = list(corpus.keys())
    qid = [q for q in queries.keys() if qrels.get(q)]
    print(f"corpus={len(did)} queries(with qrels)={len(qid)}", flush=True)
    doc_texts = [corpus[d] for d in did]
    q_texts = [queries[q] for q in qid]

    old_m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    new_m = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cuda")
    print("embedding (old=MiniLM 384, new=mpnet 768)...", flush=True)
    Do = np.asarray(old_m.encode(doc_texts, batch_size=128, show_progress_bar=False), dtype=np.float64)
    Dn = np.asarray(new_m.encode(doc_texts, batch_size=128, show_progress_bar=False), dtype=np.float64)
    Qo = np.asarray(old_m.encode(q_texts, batch_size=128, show_progress_bar=False), dtype=np.float64)
    Qn = np.asarray(new_m.encode(q_texts, batch_size=128, show_progress_bar=False), dtype=np.float64)

    # SANITY GATE: old-vs-old and new-vs-new should reproduce known baselines.
    base_old, n_old = ndcg_at_k(Qo, qid, Do, did, qrels)
    base_new, _ = ndcg_at_k(Qn, qid, Dn, did, qrels)
    print(f"[SANITY] old(MiniLM) NDCG@10={base_old:.4f} (campaign baseline ~0.819)  new(mpnet)={base_new:.4f}  evaluated_q={n_old}", flush=True)

    # DRIFT FLOOR: new queries vs old doc index (incompatible spaces) — dims differ, so pad/truncate.
    # Project new queries (768) into old dim (384) by truncation is unfair; instead the honest "floor"
    # is: there is no shared space -> we report it as the maps' job. Skip a naive floor.

    # Fit maps on a HELD-OUT paired-corpus sample (fit on half the corpus, never the queries).
    perm = RNG.permutation(len(did))
    fit_idx = perm[: len(did) // 2]
    W_o2n = fit_ls(Do[fit_idx], Dn[fit_idx])   # old->new (384->768)
    V_n2o = fit_ls(Dn[fit_idx], Do[fit_idx])   # new->old (768->384)
    res_o2n = residual(Do[perm[len(did) // 2:]], Dn[perm[len(did) // 2:]], W_o2n)
    res_n2o = residual(Dn[perm[len(did) // 2:]], Do[perm[len(did) // 2:]], V_n2o)
    cka = linear_cka(Do, Dn)
    print(f"[GEOMETRY] residual old->new={res_o2n:.4f}  new->old={res_n2o:.4f}  linear_CKA={cka:.4f}", flush=True)

    # NEW->OLD (Drift-Adapter, easy): map new queries -> old space; retrieve vs FIXED old doc index.
    Qn2o = Qn @ V_n2o
    ndcg_n2o, _ = ndcg_at_k(Qn2o, qid, Do, did, qrels)

    # OLD->NEW (ours, hard): map old docs -> new space; retrieve NEW queries vs mapped docs.
    Do2n = Do @ W_o2n
    ndcg_o2n, _ = ndcg_at_k(Qn, qid, Do2n, did, qrels)

    print("=== RESULT ===", flush=True)
    print(f"  baseline old-vs-old       : {base_old:.4f}", flush=True)
    print(f"  NEW->OLD (Drift-Adapter)  : {ndcg_n2o:.4f}  ({100*ndcg_n2o/base_old:.0f}% of old baseline)", flush=True)
    print(f"  OLD->NEW (our hard dir)   : {ndcg_o2n:.4f}  ({100*ndcg_o2n/base_old:.0f}% of old baseline)", flush=True)
    print(f"  ASYMMETRY (N2O - O2N)     : {ndcg_n2o - ndcg_o2n:+.4f}", flush=True)
    print(f"  geometry residual o->n {res_o2n:.4f} vs n->o {res_n2o:.4f}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
