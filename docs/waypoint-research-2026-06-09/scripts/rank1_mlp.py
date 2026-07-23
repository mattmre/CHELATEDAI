"""Does a NONLINEAR (residual MLP) corrector close the last ~15% to the oracle, or is ~85% the
post-hoc ceiling (the rest being the oracle's re-embed-raw-text advantage)? SciFact, exact arena.
Compares: ridge (best linear), residual-MLP (nonlinear), vs oracle. Operator-authorized exploration."""
import os
import sys

os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402

WT = r"D:\GITHUB\CHELATEDAI\.claude\worktrees\gpu-campaigns"
sys.path.insert(0, WT)
os.chdir(WT)

from run_drift_recovery_experiment import load_mteb_data  # noqa: E402
from run_road_course_campaign import select_road_course_slice  # noqa: E402
from query_encoder_drift import QueryEncoderDrift  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

SEED = 42
torch.manual_seed(SEED)
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


def main():
    corpus, queries, qrels = load_mteb_data("SciFact")
    sc, sq, sqr = select_road_course_slice(corpus, queries, qrels, max_queries=100, sample_docs=1200, seed=SEED)
    did = list(sc.keys()); qid = [q for q in sq.keys() if sqr.get(q)]
    dtx = [sc[d] for d in did]; qtx = [sq[q] for q in qid]
    old = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")
    Do = np.asarray(old.encode(dtx, batch_size=128, show_progress_bar=False), dtype=np.float32)
    drift = QueryEncoderDrift(store_dim=Do.shape[1], swap_model_name="all-mpnet-base-v2", seed=SEED)
    Qd = np.asarray(drift.embed_queries(qtx), dtype=np.float32)
    Dor = np.asarray(drift.embed_queries(dtx), dtype=np.float32)
    floor = ndcg(Qd, qid, Do, did, sqr); oracle = ndcg(Qd, qid, Dor, did, sqr)
    gap = max(oracle - floor, 1e-9)
    print(f"=== SciFact === floor={floor:.4f} oracle={oracle:.4f}", flush=True)

    perm = RNG.permutation(len(did)); fit = perm[: len(did) // 2]
    d = Do.shape[1]
    # ridge baseline (best linear from the sweep)
    Wr = np.linalg.solve(Do[fit].astype(np.float64).T @ Do[fit].astype(np.float64) + 1.0 * np.eye(d),
                         Do[fit].astype(np.float64).T @ Dor[fit].astype(np.float64))
    nd_ridge = ndcg(Qd, qid, Do.astype(np.float64) @ Wr, did, sqr)

    # residual MLP corrector
    dev = "cuda"
    Xf = torch.tensor(Do[fit], device=dev); Yf = torch.tensor(Dor[fit], device=dev)
    mlp = nn.Sequential(nn.Linear(d, 1024), nn.GELU(), nn.Linear(1024, d)).to(dev)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-5)
    Xn = torch.nn.functional.normalize(Xf, dim=1); Yn = torch.nn.functional.normalize(Yf, dim=1)
    for ep in range(400):
        opt.zero_grad()
        pred = Xf + mlp(Xf)                       # residual
        loss = (1 - torch.nn.functional.cosine_similarity(pred, Yf, dim=1)).mean()
        loss.backward(); opt.step()
    with torch.no_grad():
        Dc = (torch.tensor(Do, device=dev) + mlp(torch.tensor(Do, device=dev))).cpu().numpy()
    nd_mlp = ndcg(Qd, qid, Dc.astype(np.float64), did, sqr)

    print(f"  ridge (linear)   NDCG={nd_ridge:.4f}  recovery={100*(nd_ridge-floor)/gap:5.1f}%", flush=True)
    print(f"  residual-MLP     NDCG={nd_mlp:.4f}  recovery={100*(nd_mlp-floor)/gap:5.1f}%", flush=True)
    print(f"  oracle           NDCG={oracle:.4f}  recovery=100%", flush=True)
    verdict = "MLP CLOSES the gap (nonlinear helps)" if nd_mlp > nd_ridge + 0.02 else "~85% is the POST-HOC CEILING (last 15% = oracle re-embed advantage, unreachable from cached vectors)"
    print(f"  VERDICT: {verdict}", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
