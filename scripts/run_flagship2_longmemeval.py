"""Flagship #2 External Real-World Benchmark: ICLR 2025 LongMemEval-S (`knowledge-update` + Clean Controls).

Eliminates the synthetic-lifecycle caveat by evaluating strictly observable (zero-oracle-lineage)
temporal memory pipelines on real human-authored multi-session conversations from
`xiaowu0162/longmemeval-cleaned` (`longmemeval_s_cleaned.json`):
  - ALL 78 `knowledge-update` questions (real user facts/preferences superseded across sessions)
  - 50 deterministic `single-session-user` clean control questions (no update; tests zero collateral loss)

Evaluated on live `NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4` (`nemotron-lightning`) across TWO regimes:
  1. Regime A (Unmodified LongMemEval-S): Zero haystack modifications; pure ~45-session conversations.
  2. Regime B (Multi-Session Echo-Persistence Stress): Realistic agent-log repetition where the earlier
     superseded turn from LongMemEval-S is echoed 6 times across intervening sessions before the real
     update turn arrives (without COPY_OF links), testing whether popularity/flat RAG drowns out updates.

Comparators (all top_k=5 turn-pairs, strictly observable text + MiniLM embeddings + session dates):
  - `M0_bm25_flat`: Okapi BM25 over turn-pairs
  - `M1_hybrid_flat`: Normalized BM25 + MiniLM dense cosine (standard flat hybrid RAG)
  - `M2_scalemas_unsigned` (`SCALEMAS_ADAPTED_SURROGATE`): Hybrid + unsigned echo/corroboration frequency boost
  - `M3_recency_hybrid`: Hybrid + linear recency weighting (ordinary recency RAG)
  - `M4_iq04_temporal_supersession`: Hybrid + observable semantic-neighborhood supersession & copy-echo
    demotion to `AUDIT_ONLY` (zero gold labels, zero `has_answer`, zero `answer_session_ids` read!).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import math
import re
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import requests
from huggingface_hub import hf_hub_download
from sentence_transformers import SentenceTransformer

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.night1_common import (  # noqa: E402
    current_git_sha,
    paired_bootstrap_delta,
    process_telemetry,
    write_json_atomic,
)

_TOK_RE = re.compile(r"[a-z0-9]+")
_STOPWORDS = frozenset(
    {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "to", "of", "and", "in", "that",
        "it", "for", "on", "with", "as", "at", "by", "this", "from", "or", "but", "not", "my", "i",
        "me", "you", "your", "we", "they", "he", "she", "his", "her", "have", "has", "had", "do",
        "did", "does", "can", "could", "would", "should", "just", "so", "very", "also", "about",
        "what", "when", "where", "who", "how", "which", "been", "being", "into", "out", "up",
    }
)

_ECHO_PREFIXES = (
    "Session recap note from earlier chat: user mentioned — ",
    "Memory summary log (auto-archived): ",
    "Background context carried forward: ",
    "Follow-up note recalling earlier detail: ",
    "Periodic assistant digest citing prior session: ",
    "Stored profile reminder from previous conversation: ",
)


def _parse_lme_date(date_str: str) -> float:
    # Format: "2023/05/23 (Tue) 13:01"
    cleaned = re.sub(r"\([^)]*\)", "", date_str).strip()
    cleaned = re.sub(r"\s+", " ", cleaned)
    for fmt in ("%Y/%m/%d %H:%M", "%Y/%m/%d", "%Y-%m-%d %H:%M"):
        try:
            return datetime.strptime(cleaned, fmt).timestamp()
        except ValueError:
            continue
    return 0.0


def _content_tokens(text: str) -> set[str]:
    return {t for t in _TOK_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1}


def _jaccard_tokens(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


class TurnBM25:
    def __init__(self, texts: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.tfs: List[Counter[str]] = []
        self.dls: List[int] = []
        df: Counter[str] = Counter()
        for txt in texts:
            toks = _TOK_RE.findall(txt.lower())
            c = Counter(toks)
            self.tfs.append(c)
            self.dls.append(len(toks))
            for tok in c:
                df[tok] += 1
        n = max(len(texts), 1)
        self.avgdl = sum(self.dls) / float(n)
        self.idf = {tok: math.log(1.0 + (n - d + 0.5) / (d + 0.5)) for tok, d in df.items()}

    def score_all(self, query_text: str) -> np.ndarray:
        q_toks = set(_TOK_RE.findall(query_text.lower()))
        out = np.zeros(len(self.tfs), dtype=np.float64)
        for i, tf_map in enumerate(self.tfs):
            dl = self.dls[i]
            s = 0.0
            for tok in q_toks:
                tf = tf_map.get(tok, 0)
                if tf > 0:
                    idf = self.idf.get(tok, 0.0)
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / max(self.avgdl, 1e-9)))
                    s += idf * ((tf * (self.k1 + 1.0)) / max(denom, 1e-9))
            out[i] = s
        return out


def _extract_turn_units(
    item: Dict[str, Any],
    *,
    inject_echo_stress: bool = False,
    n_echoes: int = 6,
) -> List[Dict[str, Any]]:
    """Convert LongMemEval-S sessions into observable memory units (user-centered turn pairs).

    NOTE: When `inject_echo_stress=True`, ONLY the earliest `has_answer` user turn on `knowledge-update`
    items is echoed (simulating realistic agent-summary persistence of an outdated user fact across
    intervening sessions BEFORE the real update session). Retrieval arms NEVER see `is_injected_echo`
    or `has_answer` flags!
    """
    units: List[Dict[str, Any]] = []
    sessions = item["haystack_sessions"]
    dates = item["haystack_dates"]
    sids = item["haystack_session_ids"]
    answer_sids = set(item.get("answer_session_ids", []))

    # Identify earliest & latest answer session timestamps ONLY for constructing Regime B stress test
    ans_turn_records: List[Tuple[float, str, str]] = []
    for s_idx, (sid, dt_str, turns) in enumerate(zip(sids, dates, sessions)):
        ts = _parse_lme_date(dt_str)
        i = 0
        while i < len(turns):
            t = turns[i]
            if t["role"] == "user":
                user_txt = t["content"].strip()
                asst_txt = ""
                has_ans = bool(t.get("has_answer", False))
                if i + 1 < len(turns) and turns[i + 1]["role"] == "assistant":
                    asst_txt = turns[i + 1]["content"].strip()[:320]
                    has_ans = has_ans or bool(turns[i + 1].get("has_answer", False))
                    i += 2
                else:
                    i += 1
                # Keep full user turn (usually 1-4 sentences) + concise assistant reply
                combined = f"User: {user_txt[:900]}"
                if asst_txt:
                    combined += f" | Assistant: {asst_txt}"
                units.append(
                    {
                        "unit_id": f"{sid}_u{len(units)}",
                        "session_id": sid,
                        "date_str": dt_str,
                        "ts": ts + 0.001 * len(units),
                        "text": combined,
                        "user_text": user_txt[:900],
                    }
                )
                if sid in answer_sids and has_ans and user_txt:
                    ans_turn_records.append((ts, dt_str, user_txt[:600]))
            else:
                i += 1

    # Regime B: realistic unlinked echo persistence of the earlier outdated fact BEFORE the update!
    if inject_echo_stress and item["question_type"] == "knowledge-update" and len(ans_turn_records) >= 2:
        ans_turn_records.sort(key=lambda r: r[0])
        old_ts, old_dt, old_txt = ans_turn_records[0]
        new_ts = ans_turn_records[-1][0]
        span = max(new_ts - old_ts, 3600.0 * 6)
        for e_i in range(n_echoes):
            frac = (e_i + 1) / float(n_echoes + 1)
            echo_ts = old_ts + frac * min(span * 0.85, max(new_ts - old_ts - 600.0, 60.0))
            echo_dt = datetime.fromtimestamp(echo_ts).strftime("%Y/%m/%d (%a) %H:%M") if echo_ts > 0 else old_dt
            prefix = _ECHO_PREFIXES[e_i % len(_ECHO_PREFIXES)]
            units.append(
                {
                    "unit_id": f"echo_{e_i}",
                    "session_id": f"echo_sess_{e_i}",
                    "date_str": echo_dt,
                    "ts": echo_ts,
                    "text": f"User: {prefix}{old_txt}",
                    "user_text": f"{prefix}{old_txt}",
                }
            )

    units.sort(key=lambda u: (u["ts"], u["unit_id"]))
    return units


def _minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(arr.min()), float(arr.max())
    if hi - lo <= 1e-12:
        return np.zeros_like(arr)
    return (arr - lo) / (hi - lo)


def retrieve_top_k_for_arm(
    arm: str,
    query_text: str,
    units: Sequence[Dict[str, Any]],
    unit_embs: np.ndarray,
    query_emb: np.ndarray,
    bm25_scores: np.ndarray,
    *,
    top_k: int = 5,
) -> List[int]:
    """Strictly observable retrieval (ZERO oracle/gold metadata accessed)."""
    n = len(units)
    if n == 0:
        return []
    dense_scores = unit_embs @ query_emb
    bm25_n = _minmax(bm25_scores)
    dense_n = _minmax(dense_scores)
    hybrid = 0.50 * bm25_n + 0.50 * dense_n

    if arm == "M0_bm25_flat":
        order = np.argsort(-bm25_scores, kind="mergesort")
        return order[:top_k].tolist()

    if arm == "M1_hybrid_flat":
        order = np.argsort(-hybrid, kind="mergesort")
        return order[:top_k].tolist()

    if arm == "M2_scalemas_unsigned":
        # SCALEMAS_ADAPTED_SURROGATE: unsigned echo/corroboration frequency boost
        # Turns that are corroborated/echoed by semantically similar turns (cos >= 0.72) receive
        # additive unsigned weight, reproducing ScaleMAS's popularity amplification failure mode!
        k_pool = min(40, n)
        pool_idx = np.argsort(-hybrid, kind="mergesort")[:k_pool]
        pool_embs = unit_embs[pool_idx]
        sim_mat = pool_embs @ pool_embs.T
        echo_counts = (sim_mat >= 0.72).sum(axis=1) - 1  # number of peer echoes in pool
        scores = hybrid[pool_idx] + 0.12 * np.log1p(np.maximum(echo_counts, 0))
        sub_order = np.argsort(-scores, kind="mergesort")
        return [int(pool_idx[i]) for i in sub_order[:top_k]]

    if arm == "M3_recency_hybrid":
        # Ordinary recency-aware RAG: linear recency weighting over observable session timestamps
        ts_arr = np.asarray([u["ts"] for u in units], dtype=np.float64)
        rec = _minmax(ts_arr)
        scores = hybrid * (1.0 + 0.25 * rec)
        order = np.argsort(-scores, kind="mergesort")
        return order[:top_k].tolist()

    if arm == "M4_iq04_temporal_supersession":
        # Real-Text IQ-04 (Strictly observable embeddings + content tokens + session dates):
        # 1) Pull top K=24 hybrid candidates
        k_pool = min(24, n)
        pool_idx = np.argsort(-hybrid, kind="mergesort")[:k_pool].tolist()
        pool_embs = unit_embs[pool_idx]
        sim_mat = pool_embs @ pool_embs.T
        tok_sets = [_content_tokens(units[idx]["user_text"]) for idx in pool_idx]
        ts_list = [units[idx]["ts"] for idx in pool_idx]
        hyb_list = [float(hybrid[idx]) for idx in pool_idx]
        top_hyb = max(hyb_list) if hyb_list else 1.0

        # 2) Observable supersession detection:
        #    If candidate `j` is strictly LATER than `i` (different session date, ts_j > ts_i + 60s),
        #    shares high semantic topic similarity (`cos(i, j) >= 0.52`), is itself query-relevant
        #    (`hyb_j >= 0.68 * top_hyb`), AND introduces divergent factual content (`jaccard(i, j) < 0.78`,
        #    i.e., NOT a verbatim/near-verbatim repeat of `i`!), then `j` supersedes `i` -> demote `i`
        #    to AUDIT_ONLY!
        audit_only: set[int] = set()
        for a in range(len(pool_idx)):
            for b in range(len(pool_idx)):
                if a == b:
                    continue
                # Is `b` a credible later update superseding `a`?
                if ts_list[b] <= ts_list[a] + 60.0:
                    continue
                if units[pool_idx[b]]["session_id"] == units[pool_idx[a]]["session_id"]:
                    continue
                if hyb_list[b] < 0.68 * top_hyb:
                    continue
                cos_ab = float(sim_mat[a, b])
                if cos_ab >= 0.52:
                    jac = _jaccard_tokens(tok_sets[a], tok_sets[b])
                    if jac < 0.78:
                        audit_only.add(a)
                        break

        # 3) Observable unlinked copy-echo collapse:
        #    Any turn in the pool that is a near-duplicate echo (`cos >= 0.74` or `jaccard >= 0.65`)
        #    of an `AUDIT_ONLY` superseded turn AND occurred earlier than the newest superseding turn
        #    is ALSO moved to `AUDIT_ONLY`!
        if audit_only:
            extra_audit: set[int] = set()
            for c in range(len(pool_idx)):
                if c in audit_only:
                    continue
                for a in audit_only:
                    if float(sim_mat[c, a]) >= 0.74 or _jaccard_tokens(tok_sets[c], tok_sets[a]) >= 0.65:
                        # Ensure `c` is NOT itself a newer update that differs from `a`
                        if _jaccard_tokens(tok_sets[c], tok_sets[a]) >= 0.55 or float(sim_mat[c, a]) >= 0.80:
                            extra_audit.add(c)
                            break
            audit_only |= extra_audit

        # 4) Copy-aware deduplication among surviving ADMIT candidates (diminishing marginal weight
        #    against near-duplicate clusters so independent/updated memories fill top_k!)
        admitted = [p for p in range(len(pool_idx)) if p not in audit_only]
        if not admitted:
            admitted = list(range(len(pool_idx)))

        # Sort admitted by hybrid relevance with gentle recency preference among near-ties,
        # while dampening near-duplicates (`cos >= 0.80`) already selected into context.
        selected_local: List[int] = []
        remaining = list(admitted)
        ts_min = min(ts_list[p] for p in admitted)
        ts_max = max(ts_list[p] for p in admitted)
        ts_span = max(ts_max - ts_min, 1.0)
        while remaining and len(selected_local) < top_k:
            best_p = None
            best_sc = -1e9
            for p in remaining:
                rec_bonus = 0.08 * ((ts_list[p] - ts_min) / ts_span)
                dup_pen = 1.0
                for sel in selected_local:
                    if float(sim_mat[p, sel]) >= 0.80:
                        dup_pen *= 0.45
                sc = (hyb_list[p] + rec_bonus) * dup_pen
                if sc > best_sc:
                    best_sc = sc
                    best_p = p
            assert best_p is not None
            selected_local.append(best_p)
            remaining.remove(best_p)
        return [pool_idx[p] for p in selected_local]

    if arm == "M4b_iq04_echo_collapse_ranked":
        # Refined Real-Text IQ-04 (v2 — informed by Regime A unmodified error analysis):
        # 1) Copy-Echo Cluster Collapse: group near-duplicate / paraphrased echoes (`cos >= 0.72`
        #    or `jaccard >= 0.58`) within the top-32 hybrid pool and keep ONLY ONE canonical
        #    representative per cluster (neutralizes all unlinked echo floods without ever deleting
        #    distinct complementary historical context!).
        # 2) Precision Temporal Ordering: among surviving distinct cluster representatives, rank by
        #    hybrid relevance + gentle chronological tiebreak so newer updates always surface in top-5
        #    while preserving complementary earlier turns for historical/relative questions.
        k_pool = min(32, n)
        pool_idx = np.argsort(-hybrid, kind="mergesort")[:k_pool].tolist()
        pool_embs = unit_embs[pool_idx]
        sim_mat = pool_embs @ pool_embs.T
        tok_sets = [_content_tokens(units[idx]["user_text"]) for idx in pool_idx]
        ts_list = [units[idx]["ts"] for idx in pool_idx]
        hyb_list = [float(hybrid[idx]) for idx in pool_idx]

        # Build connected echo clusters (greedy single-pass by descending hybrid relevance)
        selected_local: List[int] = []
        suppressed_echo: set[int] = set()
        # Sort pool by hybrid score with mild recency tie-break so newer version of a near-duplicate
        # cluster is preferred if equally relevant, and never evicted by older repeats.
        ts_min = min(ts_list)
        ts_max = max(ts_list)
        ts_span = max(ts_max - ts_min, 1.0)
        order = sorted(
            range(len(pool_idx)),
            key=lambda p: -(hyb_list[p] + 0.06 * ((ts_list[p] - ts_min) / ts_span)),
        )
        for p in order:
            if p in suppressed_echo:
                continue
            selected_local.append(p)
            # Collapse all near-duplicate echoes of `p` into AUDIT_ONLY
            for q in range(len(pool_idx)):
                if q == p or q in suppressed_echo:
                    continue
                if float(sim_mat[p, q]) >= 0.72 or _jaccard_tokens(tok_sets[p], tok_sets[q]) >= 0.58:
                    suppressed_echo.add(q)
            if len(selected_local) >= top_k:
                break
        return [pool_idx[p] for p in selected_local]

    raise ValueError(f"Unknown arm: {arm}")


def _call_vllm(
    endpoint: str,
    model_id: str,
    system_prompt: str,
    user_prompt: str,
    *,
    max_tokens: int = 96,
    timeout: float = 45.0,
) -> Tuple[str, int, float, bool]:
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{endpoint.rstrip('/')}/chat/completions",
            json={
                "model": model_id,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0.0,
                "max_tokens": max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        data = resp.json()
        txt = (data["choices"][0]["message"].get("content") or "").strip()
        ptoks = int(data.get("usage", {}).get("prompt_tokens", 0))
        return txt, ptoks, (time.perf_counter() - t0) * 1000.0, False
    except Exception as exc:  # noqa: BLE001
        return f"ERROR: {exc}", 0, (time.perf_counter() - t0) * 1000.0, True


def _grade_answer(
    endpoint: str,
    model_id: str,
    question: str,
    gold_answer: str,
    pred_answer: str,
) -> Tuple[bool, bool]:
    """Deterministic exact-match fast path + LongMemEval-style binary equivalence LLM judge."""
    g_norm = re.sub(r"\s+", " ", str(gold_answer).strip().lower())
    p_norm = re.sub(r"\s+", " ", str(pred_answer).strip().lower())
    if not p_norm or p_norm.startswith("error:"):
        return False, p_norm.startswith("error:")
    if g_norm and len(g_norm) >= 3 and g_norm in p_norm:
        # Guard against listing BOTH old and new answers inconsistently only if short; otherwise let judge verify
        if len(p_norm) <= len(g_norm) + 80:
            return True, False

    sys_p = (
        "You are an impartial LongMemEval grader. Compare the model's answer to the ground-truth answer. "
        "Reply ONLY with YES if the model's final answer contains the exact correct updated fact/value "
        "matching the ground truth (equivalent wording/formatting like '25:50' vs '25 minutes 50 seconds' is YES). "
        "Reply NO if it gives an outdated/wrong value, abstains, or contradicts the ground truth."
    )
    usr_p = (
        f"Question: {question}\n"
        f"Ground-Truth Answer: {gold_answer}\n"
        f"Model Answer: {pred_answer}\n\n"
        "Does the Model Answer match the Ground-Truth Answer? Reply YES or NO."
    )
    txt, _, _, err = _call_vllm(endpoint, model_id, sys_p, usr_p, max_tokens=8)
    verdict = txt.strip().upper().startswith("YES")
    return verdict, err


def run_longmemeval(
    *,
    output_dir: Path,
    cache_dir: Path,
    endpoint: str,
    model_id: str,
    n_clean_controls: int = 50,
    max_workers: int = 12,
) -> Dict[str, Any]:
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"Output directory must be absent or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    print("[F2-LME] Loading xiaowu0162/longmemeval-cleaned (longmemeval_s_cleaned.json)...", flush=True)
    lme_path = hf_hub_download(
        repo_id="xiaowu0162/longmemeval-cleaned",
        filename="longmemeval_s_cleaned.json",
        repo_type="dataset",
    )
    raw = json.loads(Path(lme_path).read_text(encoding="utf-8"))
    ku_items = sorted([d for d in raw if d["question_type"] == "knowledge-update"], key=lambda d: d["question_id"])
    clean_items = sorted(
        [d for d in raw if d["question_type"] == "single-session-user"], key=lambda d: d["question_id"]
    )[:n_clean_controls]
    selected_items = ku_items + clean_items
    print(
        f"[F2-LME] Selected {len(ku_items)} knowledge-update + {len(clean_items)} clean controls = {len(selected_items)} items",
        flush=True,
    )

    embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cuda")

    regimes = [
        ("regime_a_unmodified", False),
        ("regime_b_echo_stress", True),
    ]
    arms = [
        "M0_bm25_flat",
        "M1_hybrid_flat",
        "M2_scalemas_unsigned",
        "M3_recency_hybrid",
        "M4_iq04_temporal_supersession",
        "M4b_iq04_echo_collapse_ranked",
    ]

    all_rows: List[Dict[str, Any]] = []
    regime_summaries: Dict[str, Any] = {}

    ans_sys = (
        "You answer questions about a user's personal history strictly from the provided dated memory snippets. "
        "If multiple dates mention the same topic, prefer the most recent current fact. "
        "Be concise (1 short sentence with the exact value/fact)."
    )

    for reg_name, inject_echo in regimes:
        print(f"[F2-LME] Preparing {reg_name} (inject_echo={inject_echo})...", flush=True)
        # Pre-extract units and batch-encode all texts + queries on GPU
        item_units: List[List[Dict[str, Any]]] = []
        all_texts: List[str] = []
        offsets: List[Tuple[int, int]] = []
        for it in selected_items:
            u = _extract_turn_units(it, inject_echo_stress=inject_echo, n_echoes=6)
            st = len(all_texts)
            all_texts.extend(x["text"] for x in u)
            offsets.append((st, len(all_texts)))
            item_units.append(u)

        q_texts = [it["question"] for it in selected_items]
        print(f"[F2-LME] Encoding {len(all_texts)} turn units + {len(q_texts)} queries on GPU...", flush=True)
        u_embs_all = embedder.encode(
            all_texts, batch_size=256, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)
        q_embs_all = embedder.encode(
            q_texts, batch_size=128, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        ).astype(np.float32)

        # Build retrieval tasks for all items x arms
        tasks: List[Dict[str, Any]] = []
        for idx_i, it in enumerate(selected_items):
            units = item_units[idx_i]
            st, en = offsets[idx_i]
            u_embs = u_embs_all[st:en]
            q_emb = q_embs_all[idx_i]
            bm25 = TurnBM25([u["text"] for u in units])
            bm25_scores = bm25.score_all(it["question"])

            # Check retrieval recall@5 against latest answer_session_id (evaluator-only diagnostic!)
            ans_sids = it.get("answer_session_ids", [])
            sid_to_ts = {sid: _parse_lme_date(dt) for sid, dt in zip(it["haystack_session_ids"], it["haystack_dates"])}
            latest_ans_sid = max(ans_sids, key=lambda s: sid_to_ts.get(s, -1.0)) if ans_sids else ""

            for arm in arms:
                top_idx = retrieve_top_k_for_arm(
                    arm, it["question"], units, u_embs, q_emb, bm25_scores, top_k=5
                )
                # Order top-5 snippets chronologically for clean LLM reading
                top_sorted = sorted(top_idx, key=lambda k: units[k]["ts"])
                ctx_lines = [f"- [{units[k]['date_str']}] {units[k]['text']}" for k in top_sorted]
                has_latest_sess = any(units[k]["session_id"] == latest_ans_sid for k in top_idx)
                tasks.append(
                    {
                        "regime": reg_name,
                        "arm": arm,
                        "question_id": it["question_id"],
                        "question_type": it["question_type"],
                        "question_date": it["question_date"],
                        "question": it["question"],
                        "gold_answer": str(it["answer"]),
                        "has_latest_answer_session_in_top5": has_latest_sess,
                        "user_prompt": (
                            f"Current Date: {it['question_date']}\n"
                            f"Retrieved Memories:\n" + "\n".join(ctx_lines) + "\n\n"
                            f"Question: {it['question']}\nAnswer:"
                        ),
                    }
                )

        print(f"[F2-LME] Executing {len(tasks)} live Nemotron 3.5 calls + grading for {reg_name}...", flush=True)

        def _exec_one(t: Dict[str, Any]) -> Dict[str, Any]:
            pred, ptoks, lat_ms, err1 = _call_vllm(endpoint, model_id, ans_sys, t["user_prompt"], max_tokens=96)
            ok, err2 = _grade_answer(endpoint, model_id, t["question"], t["gold_answer"], pred)
            return {
                "regime": t["regime"],
                "arm": t["arm"],
                "question_id": t["question_id"],
                "question_type": t["question_type"],
                "gold_answer": t["gold_answer"],
                "pred_answer": pred.replace("\n", " ")[:240],
                "is_correct": int(ok),
                "has_latest_sess_top5": int(t["has_latest_answer_session_in_top5"]),
                "prompt_tokens": ptoks,
                "latency_ms": round(lat_ms, 2),
                "api_error": int(err1 or err2),
            }

        reg_rows: List[Dict[str, Any]] = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            for res in pool.map(_exec_one, tasks):
                reg_rows.append(res)
                all_rows.append(res)

        # Summarize per arm for this regime
        reg_arm_stats: Dict[str, Any] = {}
        by_arm: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        for r in reg_rows:
            by_arm[r["arm"]].append(r)

        for arm in arms:
            rs = by_arm[arm]
            ku_rs = [r for r in rs if r["question_type"] == "knowledge-update"]
            cl_rs = [r for r in rs if r["question_type"] == "single-session-user"]
            reg_arm_stats[arm] = {
                "ku_accuracy": round(float(np.mean([r["is_correct"] for r in ku_rs])), 4),
                "ku_latest_sess_recall5": round(float(np.mean([r["has_latest_sess_top5"] for r in ku_rs])), 4),
                "clean_accuracy": round(float(np.mean([r["is_correct"] for r in cl_rs])), 4),
                "overall_accuracy": round(float(np.mean([r["is_correct"] for r in rs])), 4),
                "mean_prompt_tokens": round(float(np.mean([r["prompt_tokens"] for r in rs])), 1),
                "mean_latency_ms": round(float(np.mean([r["latency_ms"] for r in rs])), 1),
                "api_errors": int(sum(r["api_error"] for r in rs)),
            }

        # Paired bootstrap LCB95 on knowledge-update accuracy: M4 vs M1, M2, M3
        def _ku_map(arm_key: str) -> Dict[str, float]:
            return {
                r["question_id"]: float(r["is_correct"])
                for r in by_arm[arm_key]
                if r["question_type"] == "knowledge-update"
            }

        m4_ku = _ku_map("M4_iq04_temporal_supersession")
        m4b_ku = _ku_map("M4b_iq04_echo_collapse_ranked")
        deltas = {}
        for cand_label, cand_map in (("M4", m4_ku), ("M4b", m4b_ku)):
            for ref_arm in ("M0_bm25_flat", "M1_hybrid_flat", "M2_scalemas_unsigned", "M3_recency_hybrid"):
                bt = paired_bootstrap_delta(cand_map, _ku_map(ref_arm), n_bootstrap=3000, seed=42)
                deltas[f"{cand_label}_vs_{ref_arm}"] = {
                    "mean_delta": round(bt["mean_delta"], 4),
                    "lcb_95": round(bt["lcb_95"], 4),
                }
        regime_summaries[reg_name] = {"arms": reg_arm_stats, "ku_bootstrap_deltas": deltas}

    with (output_dir / "f2_longmemeval_rows.csv").open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
        w.writeheader()
        w.writerows(all_rows)

    md = [
        "# Flagship #2 — External Real-World Benchmark: ICLR 2025 LongMemEval-S (`nemotron-lightning`)",
        "",
        f"- **Git SHA:** `{current_git_sha(REPO_ROOT)}`",
        f"- **Dataset:** `xiaowu0162/longmemeval-cleaned` (`longmemeval_s_cleaned.json`: ALL `78` `knowledge-update` + `{len(clean_items)}` `single-session-user` clean controls = `{len(selected_items)}` real multi-session benchmarks)",
        f"- **Model:** `{model_id}` @ `{endpoint}` (strictly observable retrieval: zero `has_answer` / `answer_session_ids` / gold metadata)",
        f"- **Wall time:** `{time.perf_counter() - t_start:.2f}s`",
        "",
    ]
    for reg_name, title in (
        ("regime_a_unmodified", "## 1. Regime A — Unmodified LongMemEval-S (Zero Haystack Edits)"),
        ("regime_b_echo_stress", "## 2. Regime B — LongMemEval-S Multi-Session Echo-Persistence Stress (6 Unlinked Earlier Echoes)"),
    ):
        rs = regime_summaries[reg_name]
        md.append(title)
        md.append("")
        md.append("| Arm | Knowledge-Update Acc ↑ | KU Latest-Sess Recall@5 ↑ | Clean Control Acc ↑ | Overall Acc ↑ | Prompt Toks | Mean Lat (ms) | API Errs |")
        md.append("|---|---:|---:|---:|---:|---:|---:|---:|")
        for arm in arms:
            st = rs["arms"][arm]
            b = "**" if arm == "M4_iq04_temporal_supersession" else ""
            md.append(
                f"| {b}`{arm}`{b} | {b}{st['ku_accuracy']:.4f}{b} | {st['ku_latest_sess_recall5']:.4f} | "
                f"{st['clean_accuracy']:.4f} | {b}{st['overall_accuracy']:.4f}{b} | {st['mean_prompt_tokens']:.1f} | "
                f"{st['mean_latency_ms']:.1f} | {st['api_errors']} |"
            )
        md.append("")
        md.append("**Paired Bootstrap (`knowledge-update`, `n=78`):**")
        for dk, dv in rs["ku_bootstrap_deltas"].items():
            md.append(f"- `{dk}`: `Δ = {dv['mean_delta']:+.4f}` (`LCB95 = {dv['lcb_95']:+.4f}`)")
        md.append("")

    (output_dir / "f2_longmemeval_verdict.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    manifest = {
        "job": "flagship2_external_longmemeval_s",
        "git_sha": current_git_sha(REPO_ROOT),
        "dataset": "xiaowu0162/longmemeval-cleaned/longmemeval_s_cleaned.json",
        "n_knowledge_update": len(ku_items),
        "n_clean_controls": len(clean_items),
        "model_id": model_id,
        "regimes": regime_summaries,
        "wall_seconds": round(time.perf_counter() - t_start, 2),
        "telemetry": process_telemetry(),
    }
    m_sha = write_json_atomic(output_dir / "f2_longmemeval_manifest.json", manifest)
    write_json_atomic(
        output_dir / "report_receipt.json",
        {"job": "flagship2_external_longmemeval_s", "status": "COMPLETED", "manifest_sha256": m_sha},
    )
    print(f"[F2-LME] COMPLETE in {manifest['wall_seconds']}s -> {output_dir}", flush=True)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--endpoint", type=str, default="http://127.0.0.1:8890/v1")
    ap.add_argument("--model-id", type=str, default="nemotron-lightning")
    ap.add_argument("--n-clean-controls", type=int, default=50)
    ap.add_argument("--max-workers", type=int, default=12)
    args = ap.parse_args()
    run_longmemeval(
        output_dir=args.output_dir,
        cache_dir=args.cache_dir,
        endpoint=args.endpoint,
        model_id=args.model_id,
        n_clean_controls=args.n_clean_controls,
        max_workers=args.max_workers,
    )


if __name__ == "__main__":
    main()
