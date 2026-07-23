"""ONE consistent number set: the full recovery ladder on the EXACT eval-split the campaign C3a uses
(anchor_fraction=0.4, SciFact, binary NDCG = harness). Resolves the headline-number wobble by putting
every corrector on the same eval as C3a (campaign 0.161). Outputs the canonical table + the alpha curve."""
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

from run_drift_recovery_experiment import load_mteb_data, _split_anchor_eval, canonicalize_id  # noqa: E402
from run_road_course_campaign import select_road_course_slice  # noqa: E402
from query_encoder_drift import QueryEncoderDrift  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

SEED = 42
RNG = np.random.default_rng(SEED)
torch.manual_seed(SEED)
DEV = "cuda"
C3A = 0.1609  # campaign C3a (supervised anchor-InfoNCE adapter) on this eval-split


def _norm(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True); n[n == 0] = 1.0
    return x / n


def ndcg(qv, qids, dv, did, qrels, k=10):
    dv, qv = _norm(dv), _norm(qv); out = []
    for v, qid in zip(qv, qids):
        rel = {d for d, s in qrels.get(qid, {}).items() if float(s) > 0}
        if not rel:
            continue
        sims = dv @ v
        top = np.argpartition(-sims, min(k, len(sims) - 1))[:k]; top = top[np.argsort(-sims[top])]
        dcg = sum(1.0 / np.log2(r + 2) for r, i in enumerate(top) if did[i] in rel)
        idcg = sum(1.0 / np.log2(r + 2) for r in range(min(len(rel), k)))
        if idcg > 0:
            out.append(dcg / idcg)
    return float(np.mean(out))


def ridge(X, Y, lam):
    d = X.shape[1]; return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ Y)


def main():
    corpus, queries, qrels = load_mteb_data("SciFact")
    sc, sq, sqr = select_road_course_slice(corpus, queries, qrels, max_queries=100, sample_docs=1200, seed=SEED)
    _, eval_ids = _split_anchor_eval(sq, 0.4, SEED, active=True)
    es = set(eval_ids)
    did = list(sc.keys())
    qid = [q for q in sq.keys() if sqr.get(q) and canonicalize_id(q) in es]
    dtx = [sc[d] for d in did]; qtx = [sq[q] for q in qid]
    old = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEV)
    Do = np.asarray(old.encode(dtx, batch_size=128, show_progress_bar=False), dtype=np.float64)
    drift = QueryEncoderDrift(store_dim=Do.shape[1], swap_model_name="all-mpnet-base-v2", seed=SEED)
    Qd = np.asarray(drift.embed_queries(qtx), dtype=np.float64)
    Dor = np.asarray(drift.embed_queries(dtx), dtype=np.float64)
    floor = ndcg(Qd, qid, Do, did, sqr); oracle = ndcg(Qd, qid, Dor, did, sqr)
    gap = max(oracle - floor, 1e-9)
    perm = RNG.permutation(len(did)); fit = perm[: len(did) // 2]

    def R(nd):
        return 100 * (nd - floor) / gap

    def show(nd, tag):
        print(f"  {tag:34s} NDCG={nd:.4f}  R={R(nd):5.1f}%", flush=True)

    print(f"=== LADDER (SciFact eval-split, {len(qid)} queries) === floor={floor:.4f} oracle={oracle:.4f}", flush=True)
    print(f"  {'campaign C3a (anchor-InfoNCE adapter)':34s} NDCG={C3A:.4f}  R={R(C3A):5.1f}%", flush=True)
    # map family (fit on half the docs' paired old->oracle target)
    Wl, *_ = np.linalg.lstsq(Do[fit], Dor[fit], rcond=None)
    show(ndcg(Qd, qid, Do @ Wl, did, sqr), "least-squares (full)")
    Wr = ridge(Do[fit], Dor[fit], 1.0)
    show(ndcg(Qd, qid, Do @ Wr, did, sqr), "ridge (lambda=1.0)")
    U, _, Vt = np.linalg.svd(Do[fit].T @ Dor[fit], full_matrices=False)
    show(ndcg(Qd, qid, Do @ (U @ Vt), did, sqr), "orthogonal Procrustes")
    Uu, s, Vtt = np.linalg.svd(Wl, full_matrices=False); s2 = s.copy(); s2[64:] = 0
    show(ndcg(Qd, qid, Do @ ((Uu * s2) @ Vtt), did, sqr), "low-rank affine (r=64)")
    # residual MLP
    Xf = torch.tensor(Do[fit], dtype=torch.float32, device=DEV); Yf = torch.tensor(Dor[fit], dtype=torch.float32, device=DEV)
    mlp = nn.Sequential(nn.Linear(384, 1024), nn.GELU(), nn.Linear(1024, 384)).to(DEV)
    opt = torch.optim.Adam(mlp.parameters(), lr=1e-3, weight_decay=1e-5)
    for _ in range(400):
        opt.zero_grad(); pred = Xf + mlp(Xf)
        loss = (1 - torch.nn.functional.cosine_similarity(pred, Yf, dim=1)).mean(); loss.backward(); opt.step()
    with torch.no_grad():
        Dc = (torch.tensor(Do, dtype=torch.float32, device=DEV) + mlp(torch.tensor(Do, dtype=torch.float32, device=DEV))).cpu().numpy()
    show(ndcg(Qd, qid, Dc.astype(np.float64), did, sqr), "residual MLP (nonlinear)")
    print("  --- alpha-shrink of the ridge map (bound ablation) ---", flush=True)
    I = np.eye(384)
    for a in (0.05, 0.1, 0.25, 0.5, 1.0):
        show(ndcg(Qd, qid, Do @ (I + a * (Wr - I)), did, sqr), f"bounded alpha={a}")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
