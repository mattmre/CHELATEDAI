"""S1 CLEAN FORM: true generated-question doc2query self-pairs. Generate a real search query per
doc with Qwen2.5-0.5B-Instruct (a GENERATED question, distinct from doc text), embed it with the NEW
encoder, pair with the doc's cached OLD vector, fit the map. No old model, no doc re-embedding, no
relevance labels. Compare to: oracle-pair ceiling, first-K-words proxy, anchor-InfoNCE, floor."""
import os
import sys

os.environ["HF_DATASETS_OFFLINE"] = "1"   # datasets cached; allow model hub online (internet present) for any missing Qwen files
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import numpy as np  # noqa: E402
import torch  # noqa: E402

WT = r"D:\GITHUB\CHELATEDAI\.claude\worktrees\gpu-campaigns"
sys.path.insert(0, WT)
os.chdir(WT)

from run_drift_recovery_experiment import load_mteb_data  # noqa: E402
from run_road_course_campaign import select_road_course_slice  # noqa: E402
from query_encoder_drift import QueryEncoderDrift  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

SEED = 42
RNG = np.random.default_rng(SEED)
DEV = "cuda"


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


def ridge(X, Y, lam=1.0):
    d = X.shape[1]
    return np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ Y)


def gen_queries(docs, model, tok, batch=24):
    out = []
    for i in range(0, len(docs), batch):
        chunk = docs[i:i + batch]
        prompts = []
        for d in chunk:
            body = " ".join(str(d).split()[:280])
            msg = [{"role": "user", "content":
                    f"Write ONE short search query (a question or claim) that the following scientific "
                    f"passage directly answers. Output only the query, nothing else.\n\nPassage: {body}\n\nQuery:"}]
            prompts.append(tok.apply_chat_template(msg, tokenize=False, add_generation_prompt=True))
        enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEV)
        with torch.no_grad():
            gen = model.generate(**enc, max_new_tokens=32, do_sample=False, pad_token_id=tok.eos_token_id)
        for j in range(len(chunk)):
            new = gen[j][enc["input_ids"].shape[1]:]
            out.append(tok.decode(new, skip_special_tokens=True).strip().replace("\n", " ")[:200])
        print(f"  generated {min(i+batch,len(docs))}/{len(docs)}", flush=True)
    return out


def main():
    corpus, queries, qrels = load_mteb_data("SciFact")
    sc, sq, sqr = select_road_course_slice(corpus, queries, qrels, max_queries=100, sample_docs=1200, seed=SEED)
    did = list(sc.keys()); qid = [q for q in sq.keys() if sqr.get(q)]
    dtx = [sc[d] for d in did]; qtx = [sq[q] for q in qid]
    old = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=DEV)
    Do = np.asarray(old.encode(dtx, batch_size=128, show_progress_bar=False), dtype=np.float64)
    drift = QueryEncoderDrift(store_dim=Do.shape[1], swap_model_name="all-mpnet-base-v2", seed=SEED)
    Qd = np.asarray(drift.embed_queries(qtx), dtype=np.float64)
    Dor = np.asarray(drift.embed_queries(dtx), dtype=np.float64)
    floor = ndcg(Qd, qid, Do, did, sqr); oracle = ndcg(Qd, qid, Dor, did, sqr); gap = max(oracle - floor, 1e-9)
    perm = RNG.permutation(len(did)); fit = perm[: len(did) // 2]
    print(f"=== S1 doc2query SciFact === floor={floor:.4f} oracle={oracle:.4f} (anchor-InfoNCE C3a≈0.161)", flush=True)

    def recover(target_new, tag):
        W = ridge(Do[fit], target_new[fit]); nd = ndcg(Qd, qid, Do @ W, did, sqr)
        print(f"  {tag:30s} NDCG={nd:.4f}  recovery={100*(nd-floor)/gap:5.1f}%", flush=True)

    recover(Dor, "oracle-pair ridge (ceiling)")

    # CLEAN S1: generate a TRUE query per fit-doc with Qwen, embed with new encoder.
    print("loading Qwen2.5-0.5B-Instruct...", flush=True)
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", padding_side="left")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", torch_dtype=torch.float16).to(DEV).eval()
    fit_docs = [dtx[i] for i in fit]
    gq = gen_queries(fit_docs, model, tok)
    print("  --- sample (doc-opening -> generated query) ---", flush=True)
    for i in range(3):
        print(f"    DOC: {' '.join(str(fit_docs[i]).split()[:14])}...", flush=True)
        print(f"    GENQ: {gq[i]}", flush=True)
    Gq_new = np.asarray(drift.embed_queries(gq), dtype=np.float64)   # generated queries in NEW space
    # build the fit target: generated-query-new for fit docs (rest of target unused since we only fit on `fit`)
    target = np.zeros_like(Dor); target[fit] = Gq_new
    W = ridge(Do[fit], target[fit]); nd = ndcg(Qd, qid, Do @ W, did, sqr)
    print(f"  {'S1 TRUE doc2query (Qwen)':30s} NDCG={nd:.4f}  recovery={100*(nd-floor)/gap:5.1f}%", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
