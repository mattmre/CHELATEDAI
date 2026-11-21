"""Shared dataset, embedding-cache, BM25 floor, and manifest helpers for Night-1 Flagship #1."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import resource
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from benchmark_utils import canonicalize_id
from m1_baseline_ladder import (
    evaluate_rankings_mlr,
    metric_contract_digest,
    select_qrels_independent_anchors,
    split_query_ids,
)

# Canonical Night-1 encoders
ENCODERS: Dict[str, str] = {
    "E0": "sentence-transformers/all-MiniLM-L6-v2",      # 384-d
    "E1": "BAAI/bge-small-en-v1.5",                      # 384-d
    "E2": "intfloat/e5-small-v2",                        # 384-d
    "E3": "sentence-transformers/all-mpnet-base-v2",     # 768-d
}

BEIR_HF_DATASETS: Dict[str, Tuple[str, str]] = {
    "SciFact": ("BeIR/scifact", "BeIR/scifact-qrels"),
    "NFCorpus": ("BeIR/nfcorpus", "BeIR/nfcorpus-qrels"),
    "FiQA2018": ("BeIR/fiqa", "BeIR/fiqa-qrels"),
}

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def sha256_json(obj: Any) -> str:
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def current_git_sha(repo_root: Path) -> str:
    env_sha = os.environ.get("CHELATED_GIT_SHA", "").strip()
    if env_sha:
        return env_sha
    try:
        out = subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
        if out:
            return out
    except Exception:
        pass
    sha_file = repo_root / ".git_sha"
    if sha_file.is_file():
        val = sha_file.read_text(encoding="utf-8").strip()
        if val:
            return val
    return "UNKNOWN"


def process_telemetry() -> Dict[str, Any]:
    rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    info: Dict[str, Any] = {
        "peak_rss_mib": round(float(rss_kb) / 1024.0, 2),
        "cuda_available": False,
        "peak_vram_mib": 0.0,
    }
    try:
        import torch

        if torch.cuda.is_available():
            info["cuda_available"] = True
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["peak_vram_mib"] = round(float(torch.cuda.max_memory_allocated()) / (1024.0 ** 2), 2)
    except Exception:
        pass
    return info


def load_beir_dataset(
    dataset_name: str,
    *,
    max_docs: Optional[int] = None,
) -> Tuple[List[str], List[str], List[str], List[str], Dict[str, Dict[str, float]], str]:
    """Load (doc_ids, doc_texts, query_ids, query_texts, qrels, qrels_sha256) deterministically.

    Prefers HuggingFace `datasets` (`BeIR/*`) for fast, consistent loading across Python 3.12,
    falling back to `benchmark_utils.load_mteb_data` if needed.
    """
    if dataset_name not in BEIR_HF_DATASETS:
        raise ValueError(f"Unsupported dataset {dataset_name!r}; expected one of {sorted(BEIR_HF_DATASETS)}")

    corpus_map: Dict[str, str] = {}
    queries_map: Dict[str, str] = {}
    qrels_map: Dict[str, Dict[str, float]] = {}

    hf_errors: List[str] = []
    loaded_via_hf = False
    for rev in (None, "refs/convert/parquet"):
        try:
            from datasets import load_dataset

            base_repo, qrels_repo = BEIR_HF_DATASETS[dataset_name]
            kw = {"revision": rev} if rev else {}
            ds_corpus = load_dataset(base_repo, "corpus", split="corpus", **kw)
            ds_queries = load_dataset(base_repo, "queries", split="queries", **kw)
            ds_qrels = load_dataset(qrels_repo, split="test")

            for row in ds_corpus:
                did = canonicalize_id(row["_id"])
                title = str(row.get("title") or "").strip()
                text = str(row.get("text") or "").strip()
                corpus_map[did] = f"{title} {text}".strip() if title else text

            for row in ds_queries:
                qid = canonicalize_id(row["_id"])
                queries_map[qid] = str(row.get("text") or "").strip()

            for row in ds_qrels:
                qid = canonicalize_id(row["query-id"])
                did = canonicalize_id(row["corpus-id"])
                score = float(row["score"])
                if score > 0.0 and qid in queries_map and did in corpus_map:
                    qrels_map.setdefault(qid, {})[did] = score
            loaded_via_hf = True
            break
        except Exception as exc:
            corpus_map.clear()
            queries_map.clear()
            qrels_map.clear()
            hf_errors.append(f"rev={rev}: {exc!r}")

    if not loaded_via_hf:
        try:
            import io
            import zipfile
            import requests

            beir_slug = {"SciFact": "scifact", "NFCorpus": "nfcorpus", "FiQA2018": "fiqa"}[dataset_name]
            cache_root = Path(os.environ.get("CHELATED_BEIR_CACHE", Path.home() / ".cache" / "chelated_beir"))
            ds_dir = cache_root / beir_slug
            if not (ds_dir / "corpus.jsonl").is_file():
                cache_root.mkdir(parents=True, exist_ok=True)
                url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{beir_slug}.zip"
                resp = requests.get(url, timeout=180)
                resp.raise_for_status()
                with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                    zf.extractall(cache_root)
            with (ds_dir / "corpus.jsonl").open("r", encoding="utf-8") as fh:
                for line in fh:
                    row = json.loads(line)
                    did = canonicalize_id(row["_id"])
                    title = str(row.get("title") or "").strip()
                    text = str(row.get("text") or "").strip()
                    corpus_map[did] = f"{title} {text}".strip() if title else text
            with (ds_dir / "queries.jsonl").open("r", encoding="utf-8") as fh:
                for line in fh:
                    row = json.loads(line)
                    queries_map[canonicalize_id(row["_id"])] = str(row.get("text") or "").strip()
            with (ds_dir / "qrels" / "test.tsv").open("r", encoding="utf-8") as fh:
                header = fh.readline()
                _ = header
                for line in fh:
                    parts = line.rstrip("\n").split("\t")
                    if len(parts) < 3:
                        continue
                    qid = canonicalize_id(parts[0])
                    did = canonicalize_id(parts[1])
                    score = float(parts[2])
                    if score > 0.0 and qid in queries_map and did in corpus_map:
                        qrels_map.setdefault(qid, {})[did] = score
        except Exception as zip_exc:
            try:
                from benchmark_utils import load_mteb_data

                c_raw, q_raw, r_raw = load_mteb_data(dataset_name)
                for did, val in c_raw.items():
                    cdid = canonicalize_id(did)
                    if isinstance(val, dict):
                        title = str(val.get("title") or "").strip()
                        text = str(val.get("text") or "").strip()
                        corpus_map[cdid] = f"{title} {text}".strip() if title else text
                    else:
                        corpus_map[cdid] = str(val)
                for qid, val in q_raw.items():
                    queries_map[canonicalize_id(qid)] = str(val)
                for qid, rels in r_raw.items():
                    cqid = canonicalize_id(qid)
                    if cqid not in queries_map:
                        continue
                    for did, score in rels.items():
                        cdid = canonicalize_id(did)
                        if float(score) > 0.0 and cdid in corpus_map:
                            qrels_map.setdefault(cqid, {})[cdid] = float(score)
            except Exception as mteb_exc:
                raise RuntimeError(
                    f"Failed loading BEIR dataset {dataset_name}: HF={hf_errors}; ZIP={zip_exc!r}; MTEB={mteb_exc!r}"
                ) from mteb_exc

    # Keep only queries with at least one positive qrel in corpus
    valid_qids = sorted(qid for qid, rels in qrels_map.items() if rels)
    all_doc_ids = sorted(corpus_map.keys())

    if max_docs is not None and len(all_doc_ids) > max_docs:
        # Always include all positive-qrel docs so complete-qrels IDCG is intact,
        # plus deterministic sample of distractors up to max_docs.
        pos_docs = sorted({did for qid in valid_qids for did in qrels_map[qid]})
        pos_set = set(pos_docs)
        distractors = [did for did in all_doc_ids if did not in pos_set]
        distractors.sort(key=lambda d: hashlib.sha256(f"cap:{dataset_name}:{d}".encode("utf-8")).hexdigest())
        keep = sorted(pos_set | set(distractors[: max(0, max_docs - len(pos_set))]))
        all_doc_ids = keep

    doc_texts = [corpus_map[did] for did in all_doc_ids]
    query_texts = [queries_map[qid] for qid in valid_qids]
    clean_qrels = {qid: dict(sorted(qrels_map[qid].items())) for qid in valid_qids}
    qrels_sha = sha256_json(clean_qrels)
    return all_doc_ids, doc_texts, valid_qids, query_texts, clean_qrels, qrels_sha


def encode_or_load_embeddings(
    *,
    dataset_name: str,
    encoder_key: str,
    encoder_name: str,
    doc_ids: Sequence[str],
    doc_texts: Sequence[str],
    query_ids: Sequence[str],
    query_texts: Sequence[str],
    cache_dir: Path,
    batch_size: int = 128,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Encode docs & queries with SentenceTransformers (GPU if available) with atomic disk cache."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    corpus_digest = sha256_json({"docs": list(doc_ids), "queries": list(query_ids)})[:16]
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "_", encoder_name)
    cache_path = cache_dir / f"{dataset_name}_{encoder_key}_{slug}_{corpus_digest}.npz"

    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=False)
        doc_embs = np.asarray(data["doc_embs"], dtype=np.float32)
        query_embs = np.asarray(data["query_embs"], dtype=np.float32)
        meta = json.loads(str(data["meta_json"]))
        meta["cache_hit"] = True
        return doc_embs, query_embs, meta

    import torch
    from sentence_transformers import SentenceTransformer

    device = "cuda" if torch.cuda.is_available() else "cpu"
    t0 = time.perf_counter()
    model = SentenceTransformer(encoder_name, device=device)

    # E5 models require query:/passage: prefixes; BGE/MiniLM/mpnet use raw text
    if "e5-" in encoder_name.lower():
        enc_queries = [f"query: {t}" for t in query_texts]
        enc_docs = [f"passage: {t}" for t in doc_texts]
    else:
        enc_queries = list(query_texts)
        enc_docs = list(doc_texts)

    t_doc0 = time.perf_counter()
    doc_embs = np.asarray(
        model.encode(
            enc_docs,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ),
        dtype=np.float32,
    )
    doc_encode_ms = (time.perf_counter() - t_doc0) * 1000.0

    t_q0 = time.perf_counter()
    query_embs = np.asarray(
        model.encode(
            enc_queries,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ),
        dtype=np.float32,
    )
    query_encode_ms = (time.perf_counter() - t_q0) * 1000.0
    total_ms = (time.perf_counter() - t0) * 1000.0

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    meta = {
        "encoder_key": encoder_key,
        "encoder_name": encoder_name,
        "device": device,
        "dimension": int(doc_embs.shape[1]),
        "doc_count": int(doc_embs.shape[0]),
        "query_count": int(query_embs.shape[0]),
        "doc_encode_ms": round(doc_encode_ms, 2),
        "query_encode_ms": round(query_encode_ms, 2),
        "total_load_and_encode_ms": round(total_ms, 2),
        "cache_hit": False,
        "doc_embs_sha256": hashlib.sha256(doc_embs.tobytes()).hexdigest(),
        "query_embs_sha256": hashlib.sha256(query_embs.tobytes()).hexdigest(),
    }

    tmp_path = cache_path.with_suffix(".tmp.npz")
    np.savez(
        tmp_path,
        doc_embs=doc_embs,
        query_embs=query_embs,
        meta_json=np.array(json.dumps(meta, sort_keys=True)),
    )
    os.replace(tmp_path, cache_path)
    return doc_embs, query_embs, meta


class FastBM25Index:
    """Zero-dependency Okapi BM25 scorer for #355 hand-set lexical/hybrid floor."""

    def __init__(self, doc_ids: Sequence[str], doc_texts: Sequence[str], k1: float = 1.5, b: float = 0.75):
        self.doc_ids = [canonicalize_id(d) for d in doc_ids]
        self.k1 = float(k1)
        self.b = float(b)
        self.n_docs = len(self.doc_ids)
        self.doc_lens = np.zeros(self.n_docs, dtype=np.float64)
        self.postings: Dict[str, List[Tuple[int, int]]] = {}
        for idx, text in enumerate(doc_texts):
            tokens = _TOKEN_RE.findall(text.lower())
            self.doc_lens[idx] = float(len(tokens))
            counts = Counter(tokens)
            for tok, tf in counts.items():
                self.postings.setdefault(tok, []).append((idx, tf))
        self.avgdl = float(self.doc_lens.mean()) if self.n_docs > 0 else 1.0
        self.idf: Dict[str, float] = {}
        for tok, plist in self.postings.items():
            df = len(plist)
            self.idf[tok] = math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    def score_query(self, query_text: str) -> np.ndarray:
        scores = np.zeros(self.n_docs, dtype=np.float64)
        tokens = _TOKEN_RE.findall(query_text.lower())
        if not tokens:
            return scores
        for tok in set(tokens):
            plist = self.postings.get(tok)
            if not plist:
                continue
            idf = self.idf[tok]
            for doc_idx, tf in plist:
                denom = tf + self.k1 * (1.0 - self.b + self.b * (self.doc_lens[doc_idx] / max(self.avgdl, 1e-9)))
                scores[doc_idx] += idf * ((tf * (self.k1 + 1.0)) / max(denom, 1e-9))
        return scores

    def hybrid_rankings(
        self,
        query_texts: Sequence[str],
        query_ids: Sequence[str],
        dense_scores: np.ndarray,
        *,
        alpha: float = 0.5,
        k: int = 10,
    ) -> Dict[str, List[str]]:
        """Fixed 1-parameter (#355) BM25 + dense z-normalized linear hybrid."""
        rankings: Dict[str, List[str]] = {}
        k_eff = min(k, self.n_docs)
        for row_idx, (qid, qtext) in enumerate(zip(query_ids, query_texts)):
            bm25_row = self.score_query(qtext)
            dense_row = np.asarray(dense_scores[row_idx], dtype=np.float64)
            # Per-query z-score normalization (parameter-free) before fixed alpha blend
            b_std = float(bm25_row.std())
            d_std = float(dense_row.std())
            b_norm = (bm25_row - float(bm25_row.mean())) / (b_std if b_std > 1e-9 else 1.0)
            d_norm = (dense_row - float(dense_row.mean())) / (d_std if d_std > 1e-9 else 1.0)
            comb = (1.0 - alpha) * d_norm + alpha * b_norm
            if self.n_docs > max(k_eff * 4, 64):
                cand = np.argpartition(-comb, k_eff - 1)[: min(self.n_docs, max(k_eff * 4, 64))]
                ordered = sorted(cand.tolist(), key=lambda j: (-float(comb[j]), self.doc_ids[j]))[:k_eff]
            else:
                ordered = sorted(range(self.n_docs), key=lambda j: (-float(comb[j]), self.doc_ids[j]))[:k_eff]
            rankings[canonicalize_id(qid)] = [self.doc_ids[j] for j in ordered]
        return rankings


def paired_bootstrap_delta(
    scores_a: Mapping[str, float],
    scores_b: Mapping[str, float],
    *,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> Dict[str, float]:
    """Paired bootstrap of mean(a - b) over common query IDs."""
    qids = sorted(set(scores_a.keys()) & set(scores_b.keys()))
    if not qids:
        raise ValueError("No overlapping query IDs for paired bootstrap")
    diffs = np.asarray([float(scores_a[q]) - float(scores_b[q]) for q in qids], dtype=np.float64)
    rng = np.random.default_rng(seed)
    n_q = len(qids)
    boot = np.empty(n_bootstrap, dtype=np.float64)
    chunk = 1000
    for start in range(0, n_bootstrap, chunk):
        end = min(n_bootstrap, start + chunk)
        idx = rng.integers(0, n_q, size=(end - start, n_q))
        boot[start:end] = diffs[idx].mean(axis=1)
    return {
        "n_queries": int(n_q),
        "mean_delta": float(diffs.mean()),
        "lcb_95": float(np.quantile(boot, 0.025)),
        "ucb_95": float(np.quantile(boot, 0.975)),
        "lcb_975": float(np.quantile(boot, 0.025)),
    }


def write_json_atomic(path: Path, payload: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text + "\n", encoding="utf-8")
    os.replace(tmp, path)
    return hashlib.sha256((text + "\n").encode("utf-8")).hexdigest()


def summarize_per_query_metrics(per_query: Mapping[str, Mapping[str, float]]) -> Dict[str, float]:
    if not per_query:
        return {"mean_ndcg_at_10": 0.0, "mean_map_at_10": 0.0, "mean_mrr": 0.0, "mean_recall_at_10": 0.0, "n_queries": 0}
    return {
        "mean_ndcg_at_10": float(np.mean([v["ndcg_at_10"] for v in per_query.values()])),
        "mean_map_at_10": float(np.mean([v["map_at_10"] for v in per_query.values()])),
        "mean_mrr": float(np.mean([v["mrr"] for v in per_query.values()])),
        "mean_recall_at_10": float(np.mean([v["recall_at_10"] for v in per_query.values()])),
        "n_queries": int(len(per_query)),
    }


__all__ = [
    "ENCODERS",
    "BEIR_HF_DATASETS",
    "FastBM25Index",
    "current_git_sha",
    "encode_or_load_embeddings",
    "evaluate_rankings_mlr",
    "load_beir_dataset",
    "metric_contract_digest",
    "paired_bootstrap_delta",
    "process_telemetry",
    "select_qrels_independent_anchors",
    "sha256_json",
    "split_query_ids",
    "summarize_per_query_metrics",
    "write_json_atomic",
]
