"""Shared dataset, embedding-cache, BM25 floor, and manifest helpers for Night-1 Flagship #1."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
try:
    import resource
except ImportError:  # Windows portability
    resource = None  # type: ignore[assignment]
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

# Canonical Night-1 & Expanded Flagship-1 encoders
ENCODERS: Dict[str, str] = {
    "E0": "sentence-transformers/all-MiniLM-L6-v2",      # 384-d
    "E1": "BAAI/bge-small-en-v1.5",                      # 384-d
    "E2": "intfloat/e5-small-v2",                        # 384-d
    "E3": "sentence-transformers/all-mpnet-base-v2",     # 768-d
    "E4": "BAAI/bge-base-en-v1.5",                       # 768-d
}

BEIR_HF_DATASETS: Dict[str, Tuple[str, str]] = {
    "SciFact": ("BeIR/scifact", "BeIR/scifact-qrels"),
    "NFCorpus": ("BeIR/nfcorpus", "BeIR/nfcorpus-qrels"),
    "FiQA2018": ("BeIR/fiqa", "BeIR/fiqa-qrels"),
    "ArguAna": ("BeIR/arguana", "BeIR/arguana-qrels"),
    "SCIDOCS": ("BeIR/scidocs", "BeIR/scidocs-qrels"),
    "Quora": ("BeIR/quora", "BeIR/quora-qrels"),
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
    rss_mib = 0.0
    if resource is not None:
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        rss_mib = float(rss_kb) / 1024.0
    else:
        try:
            import psutil

            rss_mib = float(psutil.Process(os.getpid()).memory_info().rss) / (1024.0 ** 2)
        except Exception:
            rss_mib = 0.0
    info: Dict[str, Any] = {
        "peak_rss_mib": round(rss_mib, 2),
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

            beir_slug = {
                "SciFact": "scifact",
                "NFCorpus": "nfcorpus",
                "FiQA2018": "fiqa",
                "ArguAna": "arguana",
                "SCIDOCS": "scidocs",
                "Quora": "quora",
            }[dataset_name]
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
                kth_val = float(np.partition(comb, self.n_docs - k_eff)[self.n_docs - k_eff])
                cand = np.flatnonzero(comb >= kth_val)
                ordered = sorted(cand.tolist(), key=lambda j: (-float(comb[j]), self.doc_ids[j]))[:k_eff]
            else:
                ordered = sorted(range(self.n_docs), key=lambda j: (-float(comb[j]), self.doc_ids[j]))[:k_eff]
            rankings[canonicalize_id(qid)] = [self.doc_ids[j] for j in ordered]
        return rankings


def _clopper_pearson_two_sided(k: int, n: int, alpha: float) -> Tuple[float, float]:
    """Exact two-sided (1 - alpha) Clopper-Pearson binomial CI via regularized beta / bisection."""
    if n <= 0:
        raise ValueError("n must be positive")
    if k < 0 or k > n:
        raise ValueError("k must be in [0, n]")
    # Exact binomial tail via stable log-comb sum (stdlib-only, zero SciPy dependency)
    def binom_cdf(up_to: int, p: float) -> float:
        if p <= 0.0:
            return 1.0
        if p >= 1.0:
            return 1.0 if up_to >= n else 0.0
        lp = math.log(p)
        lq = math.log(1.0 - p)
        lg_n = math.lgamma(n + 1)
        total = 0.0
        for i in range(0, up_to + 1):
            total += math.exp(lg_n - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq)
        return min(1.0, total)

    def binom_sf(from_k: int, p: float) -> float:
        if from_k <= 0:
            return 1.0
        if p <= 0.0:
            return 0.0
        if p >= 1.0:
            return 1.0
        lp = math.log(p)
        lq = math.log(1.0 - p)
        lg_n = math.lgamma(n + 1)
        total = 0.0
        for i in range(from_k, n + 1):
            total += math.exp(lg_n - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq)
        return min(1.0, total)

    target = alpha / 2.0
    if k == 0:
        low = 0.0
    else:
        lo, hi = 0.0, k / float(n)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if binom_sf(k, mid) > target:
                hi = mid
            else:
                lo = mid
        low = lo

    if k == n:
        high = 1.0
    else:
        lo, hi = k / float(n), 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if binom_cdf(k, mid) > target:
                lo = mid
            else:
                hi = mid
        high = hi
    return float(low), float(high)


def exact_paired_binary_ci(
    a_vals: Sequence[int],
    b_vals: Sequence[int],
    *,
    z: float = 1.959963984540054,
) -> Dict[str, float]:
    """Exact conservative Clopper-Pearson Bonferroni AND Tango (1998) matched-pair score CI.

    Valid even at zero discordance (never collapses to degenerate [0.0, 0.0]).
    """
    if len(a_vals) != len(b_vals) or len(a_vals) == 0:
        raise ValueError("a_vals and b_vals must be non-empty and equal length")
    n = len(a_vals)
    n10 = sum(1 for a, b in zip(a_vals, b_vals) if int(a) == 1 and int(b) == 0)
    n01 = sum(1 for a, b in zip(a_vals, b_vals) if int(a) == 0 and int(b) == 1)
    delta = (n10 - n01) / float(n)

    # 1. Exact conservative 95% Bonferroni Clopper-Pearson (two 97.5% intervals, alpha=0.025 each)
    lw, uw = _clopper_pearson_two_sided(n10, n, alpha=0.025)
    ll, ul = _clopper_pearson_two_sided(n01, n, alpha=0.025)
    cp_low = max(-1.0, lw - ul)
    cp_high = min(1.0, uw - ll)

    # 2. Tango (1998) asymptotic matched-pair score CI (valid at n10=n01=0)
    p10 = n10 / float(n)
    p01 = n01 / float(n)
    z2 = z * z

    def tango_stat(d: float) -> float:
        # Exact closed-form Tango (1998) constrained MLE q01(d) under H0: p10 - p01 = d
        a_coef = 2.0 * n
        b_coef = (2.0 * n - n10 + n01) * d - (n10 + n01)
        c_coef = -n01 * d * (1.0 - d)
        disc = max(0.0, b_coef * b_coef - 4.0 * a_coef * c_coef)
        q01 = (-b_coef + math.sqrt(disc)) / (2.0 * a_coef)
        q01 = max(max(0.0, -d), min(0.5 * (1.0 - d), q01))
        var_num = max(2.0 * q01 + d * (1.0 - d), 1e-15)
        return ((p10 - p01 - d) ** 2) * n / var_num

    # Bisect lower bound in [-1, delta] and upper bound in [delta, 1]
    lo, hi = -1.0 + 1e-12, delta
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if tango_stat(mid) > z2:
            lo = mid
        else:
            hi = mid
    tango_low = lo

    lo, hi = delta, 1.0 - 1e-12
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if tango_stat(mid) > z2:
            hi = mid
        else:
            lo = mid
    tango_high = hi

    return {
        "n": int(n),
        "n10": int(n10),
        "n01": int(n01),
        "delta": float(delta),
        "clopper_pearson_low": float(cp_low),
        "clopper_pearson_high": float(cp_high),
        "tango_score_low": float(tango_low),
        "tango_score_high": float(tango_high),
    }


def paired_bootstrap_delta(
    scores_a: Mapping[str, float],
    scores_b: Mapping[str, float],
    *,
    n_bootstrap: int = 10_000,
    seed: int = 42,
    expected_ids: Optional[Sequence[str]] = None,
    strict_ids: bool = True,
) -> Dict[str, float]:
    """Paired two-sided 95% percentile bootstrap of mean(a - b) (2.5th / 97.5th quantiles).

    Fails closed (`ValueError`) on missing/extra/non-finite IDs when `strict_ids=True`.
    When all values are binary {0, 1}, also attaches exact Clopper-Pearson & Tango (1998)
    matched-pair intervals so zero-discordance ties never collapse to misleading [0, 0].
    """
    keys_a = set(scores_a.keys())
    keys_b = set(scores_b.keys())
    if strict_ids and keys_a != keys_b:
        raise ValueError(
            f"Paired bootstrap ID mismatch: only_in_a={len(keys_a - keys_b)}, only_in_b={len(keys_b - keys_a)}"
        )
    if expected_ids is not None:
        if len(set(expected_ids)) != len(expected_ids):
            raise ValueError("expected_ids contains duplicate IDs")
        exp_set = set(expected_ids)
        if strict_ids and (keys_a != exp_set or keys_b != exp_set):
            raise ValueError(
                f"Paired bootstrap IDs do not match expected_ids: expected={len(exp_set)}, "
                f"a={len(keys_a)}, b={len(keys_b)}"
            )
        qids = sorted(exp_set & keys_a & keys_b)
    else:
        qids = sorted(keys_a & keys_b)
    if not qids:
        raise ValueError("No overlapping query IDs for paired bootstrap")

    vals_a = np.asarray([float(scores_a[q]) for q in qids], dtype=np.float64)
    vals_b = np.asarray([float(scores_b[q]) for q in qids], dtype=np.float64)
    if not (np.all(np.isfinite(vals_a)) and np.all(np.isfinite(vals_b))):
        raise ValueError("Non-finite score encountered in paired_bootstrap_delta")

    diffs = vals_a - vals_b
    rng = np.random.default_rng(seed)
    n_q = len(qids)
    boot = np.empty(n_bootstrap, dtype=np.float64)
    chunk = 1000
    for start in range(0, n_bootstrap, chunk):
        end = min(n_bootstrap, start + chunk)
        idx = rng.integers(0, n_q, size=(end - start, n_q))
        boot[start:end] = diffs[idx].mean(axis=1)
    out: Dict[str, float] = {
        "n_queries": int(n_q),
        "mean_delta": float(diffs.mean()),
        "lcb_95": float(np.quantile(boot, 0.025)),
        "ucb_95": float(np.quantile(boot, 0.975)),
        "lcb_975": float(np.quantile(boot, 0.025)),
    }
    if np.all(np.isin(vals_a, (0.0, 1.0))) and np.all(np.isin(vals_b, (0.0, 1.0))):
        exact = exact_paired_binary_ci(vals_a.astype(int).tolist(), vals_b.astype(int).tolist())
        out["n10"] = float(exact["n10"])
        out["n01"] = float(exact["n01"])
        out["clopper_pearson_lcb95"] = exact["clopper_pearson_low"]
        out["clopper_pearson_ucb95"] = exact["clopper_pearson_high"]
        out["tango_score_lcb95"] = exact["tango_score_low"]
        out["tango_score_ucb95"] = exact["tango_score_high"]
    return out


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
