"""RANK 1 IN-ARENA: in the campaign's EXACT drift (QueryEncoderDrift projection, same 1200-doc
slice), does a trivial least-squares corrector recover like the oracle (~0.81) or stay at our
corrector's ~0.13? Resolves Rank 4 Finding 2 (is our corrector underpowered, or the arena hard?).

Reuses the harness: load_mteb_data, select_road_course_slice, QueryEncoderDrift (the exact seeded
projection the campaign + C2O use). Self-contained NDCG@10 (validated in rank4: matches MTEB)."""
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


def ndcg_at_k(qvecs, qids, dvecs, did_index, qrels, k=K):
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
        dcg = sum(rel.get(did_index[di], 0.0) / np.log2(r + 2)
                  for r, di in enumerate(top) if rel.get(did_index[di], 0.0) > 0)
        ideal = sorted([v for v in rel.values() if v > 0], reverse=True)[:k]
        idcg = sum(g / np.log2(r + 2) for r, g in enumerate(ideal))
        if idcg > 0:
            scores.append(dcg / idcg)
    return float(np.mean(scores)), len(scores)


def main():
    print("=== RANK 1 in-arena (exact QueryEncoderDrift, 1200-doc slice) ===", flush=True)
    corpus, queries, qrels = load_mteb_data("SciFact")
    sc, sq, sqr = select_road_course_slice(corpus, queries, qrels, max_queries=100, sample_docs=1200, seed=SEED)
    did = list(sc.keys())
    qid = [q for q in sq.keys() if sqr.get(q)]
    print(f"slice: docs={len(did)} queries(with qrels)={len(qid)}", flush=True)
    doc_texts = [sc[d] for d in did]
    q_texts = [sq[q] for q in qid]

    old_m = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    Do_old = np.asarray(old_m.encode(doc_texts, batch_size=128, show_progress_bar=False), dtype=np.float64)  # MiniLM 384 (the store)
    Qo_old = np.asarray(old_m.encode(q_texts, batch_size=128, show_progress_bar=False), dtype=np.float64)

    # The EXACT campaign drift + oracle target via QueryEncoderDrift (mpnet -> seeded projection -> 384).
    drift = QueryEncoderDrift(store_dim=Do_old.shape[1], swap_model_name="all-mpnet-base-v2", seed=SEED)
    Q_drift = np.asarray(drift.embed_queries(q_texts), dtype=np.float64)        # drifted eval queries
    D_oracle = np.asarray(drift.embed_queries(doc_texts), dtype=np.float64)     # C2O oracle doc targets

    # SANITY: pre-drift baseline (old queries vs old docs) and the oracle ceiling.
    base, nq = ndcg_at_k(Qo_old, qid, Do_old, did, sqr)
    floor, _ = ndcg_at_k(Q_drift, qid, Do_old, did, sqr)        # drifted queries vs UNcorrected old docs
    oracle, _ = ndcg_at_k(Q_drift, qid, D_oracle, did, sqr)     # drifted queries vs re-embedded docs (C2O)
    print(f"[SANITY] baseline(old/old)={base:.4f}  drift_floor={floor:.4f}  oracle(C2O)={oracle:.4f}  q={nq}", flush=True)

    # TRIVIAL least-squares corrector: fit old_doc(384) -> oracle_target(384) on a HELD-OUT half of docs.
    perm = RNG.permutation(len(did))
    fit = perm[: len(did) // 2]
    W, *_ = np.linalg.lstsq(Do_old[fit], D_oracle[fit], rcond=None)   # 384x384 linear map
    D_corr = Do_old @ W
    lstsq_ndcg, _ = ndcg_at_k(Q_drift, qid, D_corr, did, sqr)
    res = float(np.linalg.norm(Do_old[perm[len(did)//2:]] @ W - D_oracle[perm[len(did)//2:]])
                / (np.linalg.norm(D_oracle[perm[len(did)//2:]]) + 1e-9))

    print("=== RESULT (vs campaign C3a=0.131, C5=0.131) ===", flush=True)
    print(f"  drift floor (no correction): {floor:.4f}", flush=True)
    print(f"  oracle C2O (re-embed)      : {oracle:.4f}", flush=True)
    print(f"  TRIVIAL lstsq corrector    : {lstsq_ndcg:.4f}   (held-out residual {res:.3f})", flush=True)
    print(f"  -> lstsq recovers {100*(lstsq_ndcg-floor)/(oracle-floor+1e-9):.0f}% of the oracle gap "
          f"(campaign C3a recovered ~{100*(0.131-floor)/(oracle-floor+1e-9):.0f}%)", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
