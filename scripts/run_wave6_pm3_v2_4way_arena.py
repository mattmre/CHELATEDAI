r"""Wave-6 Track C (Guardrail K5): Zero-Strawman 4-Way Competitor Arena (`PM-3 v2`).

Compares 4 distinct agentic memory paradigms (5 arms) on 100% identical SHA-256-verified raw inputs:
  1. `Chelated_v4_Stratified_WFS`:
     - Suite A (`N=60` Regulated Enterprise Drift & Retraction): `AuthorityDominatedExocortexV2`
       (`max_tokens=256` schema-guarded claim extraction + GPU `BAAI/bge-m3` + `FastBM25Index` +
       `EK5 v3` synchronous `wait=true` epoch-pinned retraction & `CONTESTED` abstention).
     - Suite B (`N=40` Live `LongMemEval-S` Conversational Timelines): GPU `BAAI/bge-m3` +
       `BAAI/bge-reranker-v2-m3` + `EK5 v3` echo-collapse-before-budgeting & chronological supersession
       (`0` write-path LLM tokens, contention-free GPU encoding—no `CUDA_VISIBLE_DEVICES=""` or `enc_lock`).
  2. `Mem0_v2_2_1_Live_InferTrue`: Live `mem0ai 2.2.1` (`infer=True`, `enable_thinking=False`, GPU embedder).
  3. `Mem0_v2_2_1_Live_InferFalse`: Live `mem0ai 2.2.1` (`infer=False`, raw vector storage).
  4. `Hindsight_4Net_RRR_Emulation`: Explicitly labeled `"emulation (ours): Vectorize Hindsight-architecture
     4-Network (World Facts + Agent Experiences + Entity Summaries + Scalar-Confidence Evolving Beliefs)
     Retain -> Recall -> Reflect"` backed by GPU `BAAI/bge-m3` + `FastBM25Index` + `Nemotron-3.5-30B`.
  5. `OpenSearch_Memory_Async_Emulation`: Explicitly labeled `"emulation (ours): AWS OpenSearch Memory-parity
     immediate Events Hybrid BM25+kNN + asynchronous background SEMANTIC + SUMMARY LLM strategies"`.

Enforces all Wave-5 Cross-Judgment K5 guardrails:
  - Contention-free GPU encoding; Suite A (`N=60`) and Suite B (`N=40`) reported strictly separately (never pooled clocks).
  - Four separated clocks & token meters: `write_ack_ms`, `consolidation_bg_ms`, `read_retrieval_ms`, `reader_llm_ms`,
    `write_llm_tokens`, and `read_prompt_tokens`.
  - Freshness & retraction horizon sweep on Suite A: `Delta = 0s` (immediate synchronous read upon retraction/update,
    before async background LLM consolidation completes) vs. `Post-Consolidation` steady state, measuring both
    stale retracted-value resurrection in retrieved context (`stale_context_resurrection`) and answer accuracy.
  - 100% blind cross-family semantic grading (`Qwen/Qwen2.5-7B-Instruct` grading `Nemotron-3.5-30B` outputs via
    `BlindCrossFamilySemanticJudge`), completely replacing substring `ABSTAIN_MARKERS` and auditing schema anomalies separately.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import csv
import datetime
import gc
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
for _p in (str(REPO_ROOT), str(SCRIPT_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from chelated_epistemic_proposer_v2 import (  # noqa: E402
    EXTRACTION_SYSTEM_PROMPT_V2,
    AuthorityDominatedExocortexV2,
)
from chelated_raw_text_exocortex import (  # noqa: E402
    ExtractedClaimRecord,
    RawDocumentEnvelope,
    parse_extracted_claim_json,
)
from prospective_realworld_drift_corpus import (  # noqa: E402
    build_prospective_realworld_drift_suite,
)
from scripts.night1_common import (  # noqa: E402
    FastBM25Index,
    current_git_sha,
    write_json_atomic,
)
from scripts.run_r6_pm3_and_d2_qdrant_csls_backfill import (  # noqa: E402
    raw_conversational_messages,
    select_lme_items,
)
from spherical_migration_bridge import l2_normalize  # noqa: E402
from wave6_common import (  # noqa: E402
    BlindCrossFamilySemanticJudge,
    paired_bootstrap_wave6,
)

ENTERPRISE_READER_SYSTEM = (
    "You are an enterprise verification assistant. Answer the question concisely using ONLY the provided records.\n"
    "- Higher-authority official filings, signed advisories, and postmortems supersede preliminary notes and unverified rumors.\n"
    "- If equal-authority official sources mutually conflict with no authoritative resolution (CONTESTED), or if the "
    "verified answer is absent from the records, reply ONLY: ABSTAIN."
)

CONVERSATIONAL_READER_SYSTEM = (
    "Answer the user's question concisely (1-2 sentences) using ONLY the dated records below.\n"
    "- When records contain an earlier value and a newer dated update (or [LATEST CURRENT UPDATE]), report the latest current fact "
    "unless the question explicitly asks about the earlier state.\n"
    "- Verify exact entity and premise match: if the specific person, pet, item, place, or metric asked about was NEVER "
    "mentioned in the records, clearly refuse / reply ABSTAIN instead of guessing from a different item."
)

HINDSIGHT_RETAIN_SYSTEM = (
    "You are the Retain consolidation engine of a 4-network agent memory system. Given a batch of dated records/messages, "
    "extract structured JSON with three fields:\n"
    '1. "world_facts": list of concise dated objective fact strings.\n'
    '2. "entity_summaries": dict mapping key entity/topic names to a 1-sentence consolidated profile.\n'
    '3. "evolving_beliefs": list of {"claim": "<concise claim>", "confidence": <float 0.1 to 0.99>} where repeated '
    "corroboration or newer updates receive higher confidence.\n"
    'Return ONLY valid JSON: {"world_facts": [...], "entity_summaries": {...}, "evolving_beliefs": [...]}.'
)

HINDSIGHT_REFLECT_SYSTEM = (
    "You are the Reflect reasoning engine over a 4-network memory store (World Facts, Experiences, Entity Summaries, "
    "and Evolving Beliefs with confidence scores). Synthesize a concise 1-2 sentence answer strictly from the retrieved "
    "memory banks. If the asked premise/entity is unmentioned or official records are irreconcilably deadlocked, reply ABSTAIN."
)

OPENSEARCH_SEMANTIC_SYSTEM = (
    "You are an asynchronous SEMANTIC memory strategy extractor. Extract current factual statements, preferences, and "
    "updates from the events, reconciling updates where newer events supersede older ones. "
    'Return ONLY JSON: {"semantic_memories": ["..."]}.'
)

OPENSEARCH_SUMMARY_SYSTEM = (
    "You are an asynchronous SUMMARY memory strategy extractor. Produce a concise rolling factual summary (3-5 bullet lines) "
    "preserving dates, entities, numbers, and state changes from the session events."
)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def call_vllm_timed(
    endpoint: str,
    model_id: str,
    system: str,
    user: str,
    *,
    max_tokens: int = 256,
) -> Tuple[str, int, float]:
    """Return `(content, prompt_tokens, wall_ms)` against local OpenAI-compatible vLLM."""
    import requests

    t0 = time.perf_counter()
    for attempt in range(5):
        try:
            resp = requests.post(
                f"{endpoint.rstrip('/')}/chat/completions",
                json={
                    "model": model_id,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": 0.0,
                    "max_tokens": int(max_tokens),
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout=180.0,
            )
            if resp.status_code in {408, 409, 429, 500, 502, 503, 504}:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:160]}")
            resp.raise_for_status()
            body = resp.json()
            txt = (body["choices"][0]["message"].get("content") or "").strip()
            ptoks = int(body.get("usage", {}).get("prompt_tokens") or 0)
            return txt, ptoks, (time.perf_counter() - t0) * 1000.0
        except Exception:
            if attempt == 4:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("Unreachable")


def extract_json_object(raw_text: str) -> Dict[str, Any]:
    txt = (raw_text or "").strip()
    m = re.search(r"\{.*\}", txt, re.DOTALL)
    if not m:
        return {}
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def check_stale_resurrection(context_text: str, stale_values: Sequence[str]) -> bool:
    """Return True if any retracted/superseded/false value appears unmasked in retrieved context."""
    low = (context_text or "").lower()
    for sv in stale_values:
        s_clean = str(sv or "").strip().lower()
        if s_clean and len(s_clean) >= 3 and s_clean in low:
            return True
    return False


def _rrf_combine(bm25_scores: np.ndarray, dense_scores: np.ndarray, k: int = 60) -> np.ndarray:
    n = bm25_scores.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    out = np.zeros(n, dtype=np.float64)
    for arr in (bm25_scores, dense_scores):
        order = np.argsort(-arr, kind="mergesort")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(1, n + 1, dtype=np.float64)
        out += 1.0 / (float(k) + ranks)
    return out


class Hindsight4NetEmulationStore:
    """Steelmanned 4-Network Retain -> Recall -> Reflect emulation (World, Experience, Entity Summary, Evolving Beliefs)."""

    def __init__(self, embed_fn: Any) -> None:
        self._embed_fn = embed_fn
        self.experiences: List[str] = []
        self.world_facts: List[str] = []
        self.entity_summaries: Dict[str, str] = {}
        self.beliefs: Dict[str, float] = {}
        self._all_entries: List[str] = []
        self._bm25: Optional[FastBM25Index] = None
        self._vecs: Optional[np.ndarray] = None

    def write_raw_experiences_ack(self, raw_lines: Sequence[str]) -> float:
        t0 = time.perf_counter()
        self.experiences.extend(raw_lines)
        return (time.perf_counter() - t0) * 1000.0

    def retain_consolidate(
        self,
        chunks: Sequence[str],
        endpoint: str,
        model_id: str,
    ) -> Tuple[float, int]:
        t0 = time.perf_counter()
        total_toks = 0
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(8, max(1, len(chunks)))) as pool:
            batch_out = list(
                pool.map(
                    lambda c: call_vllm_timed(endpoint, model_id, HINDSIGHT_RETAIN_SYSTEM, c, max_tokens=512),
                    chunks,
                )
            )
        for raw, ptoks, _ in batch_out:
            total_toks += ptoks
            parsed = extract_json_object(raw)
            for wf in parsed.get("world_facts") or []:
                if isinstance(wf, str) and wf.strip():
                    self.world_facts.append(f"[WorldFact] {wf.strip()}")
            es = parsed.get("entity_summaries")
            if isinstance(es, dict):
                for k, v in es.items():
                    if str(k).strip() and str(v).strip():
                        self.entity_summaries[str(k).strip()] = str(v).strip()
            for bel in parsed.get("evolving_beliefs") or []:
                if isinstance(bel, dict) and str(bel.get("claim") or "").strip():
                    cl = str(bel["claim"]).strip()
                    try:
                        conf = float(bel.get("confidence", 0.65))
                    except Exception:
                        conf = 0.65
                    conf = max(0.05, min(0.99, conf))
                    # Corroboration boosts scalar confidence (standard belief-network accumulation)
                    prev = self.beliefs.get(cl, 0.0)
                    self.beliefs[cl] = min(0.99, 1.0 - (1.0 - prev) * (1.0 - conf)) if prev > 0 else conf

        entries: List[str] = []
        entries.extend(self.world_facts)
        for ent, summ in sorted(self.entity_summaries.items()):
            entries.append(f"[EntitySummary:{ent}] {summ}")
        for cl, conf in sorted(self.beliefs.items(), key=lambda kv: -kv[1]):
            entries.append(f"[Belief confidence={conf:.2f}] {cl}")
        entries.extend(f"[Experience] {ex}" for ex in self.experiences)
        self._all_entries = entries
        if entries:
            ids = [f"hs_{i}" for i in range(len(entries))]
            self._bm25 = FastBM25Index(ids, entries)
            self._vecs = l2_normalize(np.asarray(self._embed_fn(entries), dtype=np.float32))
        return (time.perf_counter() - t0) * 1000.0, total_toks

    def recall(self, query: str, *, top_k: int = 12, delta_zero_unconsolidated: bool = False) -> Tuple[str, float]:
        t0 = time.perf_counter()
        if delta_zero_unconsolidated or self._vecs is None or self._bm25 is None:
            # Before async Retain finishes (Delta=0s), only raw Experience log is searchable
            if not self.experiences:
                return "(No experiences recorded.)", (time.perf_counter() - t0) * 1000.0
            ids = [f"ex_{i}" for i in range(len(self.experiences))]
            bm = FastBM25Index(ids, self.experiences).score_query(query)
            qv = l2_normalize(np.asarray(self._embed_fn([query]), dtype=np.float32))[0]
            dv = l2_normalize(np.asarray(self._embed_fn(self.experiences), dtype=np.float32))
            comb = _rrf_combine(bm, dv @ qv)
            top = np.argsort(-comb, kind="mergesort")[: min(top_k, len(self.experiences))]
            ctx = "\n".join(f"- [Experience] {self.experiences[int(i)]}" for i in top)
            return ctx, (time.perf_counter() - t0) * 1000.0

        qv = l2_normalize(np.asarray(self._embed_fn([query]), dtype=np.float32))[0]
        bm = self._bm25.score_query(query)
        comb = _rrf_combine(bm, self._vecs @ qv)
        top = np.argsort(-comb, kind="mergesort")[: min(top_k, len(self._all_entries))]
        ctx = "\n".join(f"- {self._all_entries[int(i)]}" for i in top)
        return ctx, (time.perf_counter() - t0) * 1000.0


class OpenSearchMemoryAsyncEmulation:
    """Steelmanned AWS OpenSearch Memory-parity emulation: immediate Events Hybrid BM25+kNN + async SEMANTIC & SUMMARY."""

    def __init__(self, embed_fn: Any) -> None:
        self._embed_fn = embed_fn
        self.raw_events: List[str] = []
        self._event_bm25: Optional[FastBM25Index] = None
        self._event_vecs: Optional[np.ndarray] = None
        self.consolidated_memories: List[str] = []
        self._cons_bm25: Optional[FastBM25Index] = None
        self._cons_vecs: Optional[np.ndarray] = None

    def write_events_immediate(self, event_lines: Sequence[str]) -> float:
        """Immediate synchronous Events write (searchable within milliseconds before async strategies run)."""
        t0 = time.perf_counter()
        self.raw_events = list(event_lines)
        if self.raw_events:
            ids = [f"ev_{i}" for i in range(len(self.raw_events))]
            self._event_bm25 = FastBM25Index(ids, self.raw_events)
            self._event_vecs = l2_normalize(np.asarray(self._embed_fn(self.raw_events), dtype=np.float32))
        return (time.perf_counter() - t0) * 1000.0

    def run_async_strategies(self, chunks: Sequence[str], endpoint: str, model_id: str) -> Tuple[float, int]:
        """Background SEMANTIC + SUMMARY strategy passes."""
        t0 = time.perf_counter()
        toks = 0
        sem_list: List[str] = []
        sum_list: List[str] = []

        def _proc_chunk(ch: str) -> Tuple[str, int, str, int]:
            r_sem, pt1, _ = call_vllm_timed(endpoint, model_id, OPENSEARCH_SEMANTIC_SYSTEM, ch, max_tokens=400)
            r_sum, pt2, _ = call_vllm_timed(endpoint, model_id, OPENSEARCH_SUMMARY_SYSTEM, ch, max_tokens=300)
            return r_sem, pt1, r_sum, pt2

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(6, max(1, len(chunks)))) as pool:
            res_chunks = list(pool.map(_proc_chunk, chunks))
        for r_sem, pt1, r_sum, pt2 in res_chunks:
            toks += pt1 + pt2
            obj = extract_json_object(r_sem)
            for item in obj.get("semantic_memories") or []:
                if isinstance(item, str) and item.strip():
                    sem_list.append(f"[SEMANTIC] {item.strip()}")
            if r_sum.strip():
                sum_list.append(f"[SUMMARY] {r_sum.strip()}")
        self.consolidated_memories = sem_list + sum_list
        if self.consolidated_memories:
            ids = [f"cm_{i}" for i in range(len(self.consolidated_memories))]
            self._cons_bm25 = FastBM25Index(ids, self.consolidated_memories)
            self._cons_vecs = l2_normalize(np.asarray(self._embed_fn(self.consolidated_memories), dtype=np.float32))
        return (time.perf_counter() - t0) * 1000.0, toks

    def retrieve(self, query: str, *, top_k: int = 10, delta_zero_before_async: bool = False) -> Tuple[str, float]:
        t0 = time.perf_counter()
        qv = l2_normalize(np.asarray(self._embed_fn([query]), dtype=np.float32))[0]
        if delta_zero_before_async or self._cons_vecs is None or self._cons_bm25 is None:
            # Delta = 0s: async SEMANTIC/SUMMARY strategies have not completed yet -> queries hit raw Events Hybrid index!
            if self._event_vecs is None or self._event_bm25 is None:
                return "(No events.)", (time.perf_counter() - t0) * 1000.0
            comb = _rrf_combine(self._event_bm25.score_query(query), self._event_vecs @ qv)
            top = np.argsort(-comb, kind="mergesort")[: min(top_k, len(self.raw_events))]
            ctx = "\n".join(f"- {self.raw_events[int(i)]}" for i in top)
            return ctx, (time.perf_counter() - t0) * 1000.0

        # Post-consolidation: hybrid search over processed SEMANTIC + SUMMARY memories + top-3 raw fallback
        comb_c = _rrf_combine(self._cons_bm25.score_query(query), self._cons_vecs @ qv)
        top_c = np.argsort(-comb_c, kind="mergesort")[: min(top_k, len(self.consolidated_memories))]
        lines = [f"- {self.consolidated_memories[int(i)]}" for i in top_c]
        if self._event_vecs is not None and self._event_bm25 is not None:
            comb_e = _rrf_combine(self._event_bm25.score_query(query), self._event_vecs @ qv)
            top_e = np.argsort(-comb_e, kind="mergesort")[:3]
            lines.extend(f"- [RawEvent] {self.raw_events[int(i)]}" for i in top_e)
        return "\n".join(lines), (time.perf_counter() - t0) * 1000.0


_SHARED_MEM0_EMBEDDER: Any = None


def _normalize_mem0_llm_json_str(content: str) -> Tuple[str, int]:
    """Repair upstream mem0ai 2.2.1 JSON schema edge cases (e.g. `{"memory": ["str"]}` -> dict items)."""
    txt = (content or "").strip()
    if not txt:
        return content, 0
    fence_m = re.search(r"```(?:json)?\s*(\{.*\})\s*```", txt, flags=re.DOTALL)
    raw_json = fence_m.group(1) if fence_m else txt
    start = raw_json.find("{")
    end = raw_json.rfind("}")
    if start < 0 or end <= start:
        return content, 0
    try:
        parsed = json.loads(raw_json[start : end + 1])
    except Exception:
        return content, 0
    if not isinstance(parsed, dict):
        return content, 0
    repaired = 0
    if "memory" in parsed and isinstance(parsed["memory"], list):
        fixed_mem: List[Dict[str, Any]] = []
        for idx, item in enumerate(parsed["memory"]):
            if isinstance(item, str):
                repaired += 1
                if item.strip():
                    fixed_mem.append({"id": str(idx), "text": item.strip(), "event": "ADD"})
            elif isinstance(item, dict):
                if "text" not in item and "fact" in item:
                    item = {**item, "text": str(item["fact"])}
                    repaired += 1
                if "event" not in item:
                    item = {**item, "event": "ADD"}
                    repaired += 1
                if "id" not in item:
                    item = {**item, "id": str(idx)}
                fixed_mem.append(item)
        parsed["memory"] = fixed_mem
    if "facts" in parsed and isinstance(parsed["facts"], list):
        fixed_facts: List[str] = []
        for item in parsed["facts"]:
            if isinstance(item, str):
                if item.strip():
                    fixed_facts.append(item.strip())
            elif isinstance(item, dict):
                repaired += 1
                val = str(item.get("fact") or item.get("text") or item.get("memory") or "").strip()
                if val:
                    fixed_facts.append(val)
        parsed["facts"] = fixed_facts
    if repaired > 0:
        return json.dumps(parsed, ensure_ascii=False), repaired
    return content, 0


def make_live_mem0_gpu(endpoint: str, model_id: str, qpath: Path, *, infer_prompt: bool, qdrant_url: str = "") -> Tuple[Any, Dict[str, int]]:
    global _SHARED_MEM0_EMBEDDER
    os.environ["MEM0_TELEMETRY"] = "false"
    import mem0.memory.main as _m0_main
    from mem0 import Memory
    from mem0.utils.factory import EmbedderFactory

    _m0_main.capture_event = lambda *a, **kw: None
    if hasattr(_m0_main, "AnonymousTelemetry"):
        _m0_main.AnonymousTelemetry = lambda *a, **kw: type("_NoTelem", (), {"capture_event": lambda *x, **y: None})()

    orig_emb_create = EmbedderFactory.create

    def _cached_emb_create(*args: Any, **kwargs: Any) -> Any:
        global _SHARED_MEM0_EMBEDDER
        if _SHARED_MEM0_EMBEDDER is None:
            _SHARED_MEM0_EMBEDDER = orig_emb_create(*args, **kwargs)
        return _SHARED_MEM0_EMBEDDER

    EmbedderFactory.create = staticmethod(_cached_emb_create)  # type: ignore[assignment]
    try:
        vs_cfg: Dict[str, Any] = {
            "collection_name": f"w6_pm3_{qpath.name}",
            "embedding_model_dims": 384,
        }
        if qdrant_url:
            vs_cfg["url"] = qdrant_url
        else:
            vs_cfg["path"] = str(qpath / "qdrant")

        cfg: Dict[str, Any] = {
            "history_db_path": str(qpath / "history.db"),
            "llm": {
                "provider": "openai",
                "config": {
                    "model": model_id,
                    "openai_base_url": endpoint,
                    "api_key": "local-vllm",
                    "temperature": 0.0,
                    "max_tokens": 1536,
                },
            },
            "embedder": {
                "provider": "huggingface",
                "config": {"model": "sentence-transformers/all-MiniLM-L6-v2", "embedding_dims": 384},
            },
            "vector_store": {"provider": "qdrant", "config": vs_cfg},
        }
        if infer_prompt:
            cfg["custom_fact_extraction_prompt"] = (
                "Extract key concrete factual claims, identifiers, metrics, preferences, and chronological updates from the input "
                "(up to 15 concise strings). "
                'Return ONLY valid JSON: {"facts": ["..."]}.'
            )
        memory = Memory.from_config(cfg)
    finally:
        EmbedderFactory.create = orig_emb_create  # type: ignore[assignment]

    orig_create = memory.llm.client.chat.completions.create
    token_box = {"write_prompt_tokens": 0, "schema_repairs": 0, "fallback_splits": 0}

    def _patched(*args: Any, **kwargs: Any) -> Any:
        extra = dict(kwargs.get("extra_body") or {})
        extra["chat_template_kwargs"] = {"enable_thinking": False}
        kwargs["extra_body"] = extra
        resp = orig_create(*args, **kwargs)
        usage = getattr(resp, "usage", None)
        if usage is not None:
            token_box["write_prompt_tokens"] += int(getattr(usage, "prompt_tokens", 0) or 0)
        try:
            msg = resp.choices[0].message
            raw_c = getattr(msg, "content", None)
            if isinstance(raw_c, str):
                norm_c, n_rep = _normalize_mem0_llm_json_str(raw_c)
                if n_rep > 0:
                    msg.content = norm_c
                    token_box["schema_repairs"] += n_rep
        except Exception:
            pass
        return resp

    memory.llm.client.chat.completions.create = _patched
    return memory, token_box


def safe_mem0_add_infer(memory: Any, text_chunk: str, user_id: str, token_box: Dict[str, int]) -> None:
    """Invoke `memory.add(..., infer=True)` with split-chunk retry if upstream mem0 hits a parser bug."""
    try:
        memory.add([{"role": "user", "content": text_chunk}], user_id=user_id, infer=True)
    except Exception:
        token_box["fallback_splits"] = int(token_box.get("fallback_splits", 0)) + 1
        lines = text_chunk.splitlines()
        if len(lines) >= 2:
            mid = len(lines) // 2
            for half in ("\n".join(lines[:mid]), "\n".join(lines[mid:])):
                if not half.strip():
                    continue
                try:
                    memory.add([{"role": "user", "content": half}], user_id=user_id, infer=True)
                except Exception:
                    pass


def close_live_mem0(memory: Any, qpath: Path, qdrant_url: str = "") -> None:
    vs = getattr(memory, "vector_store", None)
    col_name = getattr(vs, "collection_name", None)
    cli = getattr(vs, "client", None)
    if cli is not None and col_name and qdrant_url:
        try:
            cli.delete_collection(collection_name=col_name)
        except Exception:
            pass
    for attr in ("vector_store", "_telemetry_vector_store"):
        st = getattr(memory, attr, None)
        c = getattr(st, "client", None)
        if c is not None:
            try:
                c.close()
            except Exception:
                pass
    shutil.rmtree(qpath, ignore_errors=True)


def chunk_strings(lines: Sequence[str], max_chars: int = 18000) -> List[str]:
    chunks: List[str] = []
    cur: List[str] = []
    sz = 0
    for ln in lines:
        if cur and sz + len(ln) + 1 > max_chars:
            chunks.append("\n".join(cur))
            cur = []
            sz = 0
        cur.append(ln)
        sz += len(ln) + 1
    if cur:
        chunks.append("\n".join(cur))
    return chunks


def chelated_conversational_gpu_retrieve(
    messages: Sequence[Mapping[str, Any]],
    question: str,
    bge_encode_fn: Any,
    reranker_predict_fn: Any,
) -> Tuple[str, float, float, str]:
    """Contention-free GPU BGE-M3 + CrossEncoder reranking + EK5 v3 echo-collapse-before-budgeting."""
    from chelated_conversational_v3 import ChelatedConversationalV3, ConversationTurn

    t_w0 = time.perf_counter()
    turns: List[ConversationTurn] = []
    i = 0
    while i < len(messages):
        m = messages[i]
        ts = float(m["ts"])
        dt_str = datetime.datetime.fromtimestamp(ts, tz=datetime.timezone.utc).strftime("%Y/%m/%d %H:%M")
        role = str(m["role"]).capitalize()
        body = str(m["content"]).strip()
        text = f"{role}: {body}"
        idx_text = text
        pos = i
        i += 1
        if (
            role == "User"
            and i < len(messages)
            and str(messages[i]["role"]).lower() == "assistant"
            and str(messages[i]["session_id"]) == str(m["session_id"])
        ):
            reply = str(messages[i]["content"]).strip()
            text += f" | Assistant: {reply[:800]}"
            idx_text += f" | Assistant: {reply[:340]}"
            i += 1
        turns.append(
            ConversationTurn(
                turn_id=f"{m['session_id']}:{pos}",
                session_id=str(m["session_id"]),
                date=dt_str,
                timestamp=ts,
                text=text,
                claim_text=body,
                speaker=role,
                index_text=idx_text,
            )
        )

    class _DirectGPUEmbedder:
        def encode(self, texts: Sequence[str]) -> np.ndarray:
            return np.asarray(bge_encode_fn(list(texts)), dtype=np.float32)

    engine = ChelatedConversationalV3(turns, _DirectGPUEmbedder())
    write_ack_ms = (time.perf_counter() - t_w0) * 1000.0

    t_r0 = time.perf_counter()
    order, _comb, _sess_rrf = engine.rank_with_scores(question)
    # K2/K5 order-of-operations fix: run EK5 echo collapse & chronological supersession BEFORE rerank/budgeting!
    routing = engine._gating(order, min(40, max(12, len(turns))))
    cand_pool = list(routing.order[:32])
    if cand_pool:
        pairs = [(question, turns[idx].indexed_text[:1000]) for idx in cand_pool]
        ce_scores = np.asarray(reranker_predict_fn(pairs), dtype=np.float64)
        reranked_top = [cand_pool[int(k)] for k in np.argsort(-ce_scores, kind="mergesort")]
        # Preserve any superseded/superseding partner turns so chronological update pairs stay intact
        sup_partners = [idx for edge in routing.supersessions for idx in edge if idx not in routing.suppressed]
        merged_order = tuple(dict.fromkeys(reranked_top + sup_partners + list(routing.order)))
    else:
        merged_order = routing.order

    recall = engine.retrieve(
        question,
        epistemic=True,
        anchor_count=8,
        window_radius=1,
        context_chars=6800,
        pool_size=min(40, max(8, len(turns))),
        ranked=merged_order,
        session_order="chronological",
    )
    read_ms = (time.perf_counter() - t_r0) * 1000.0
    status = "GATED_ACTIVE" if recall.gate_active else "ADMIT_OK"
    return recall.context, write_ack_ms, read_ms, status


def summarize_suite(
    suite_name: str,
    rows: Sequence[Mapping[str, Any]],
    arm_keys: Sequence[str],
    primary_arm: str = "Chelated_v4_Stratified_WFS",
) -> Dict[str, Any]:
    item_ids = [str(r["item_id"]) for r in rows]
    acc_maps: Dict[str, Dict[str, float]] = {a: {} for a in arm_keys}
    arm_stats: Dict[str, Any] = {}
    for arm in arm_keys:
        for r in rows:
            acc_maps[arm][str(r["item_id"])] = 1.0 if r["arms"][arm]["correct"] else 0.0
        n_tot = len(rows)
        abs_rows = [r for r in rows if r["abstention_required"]]
        det_rows = [r for r in rows if not r["abstention_required"]]
        stale_rows = [r for r in rows if r.get("stale_values")]
        arm_stats[arm] = {
            "n": n_tot,
            "accuracy": float(np.mean(list(acc_maps[arm].values()))),
            "determinate_accuracy": (
                float(np.mean([1.0 if r["arms"][arm]["correct"] else 0.0 for r in det_rows])) if det_rows else None
            ),
            "abstention_recall": (
                float(np.mean([1.0 if r["arms"][arm]["correct"] else 0.0 for r in abs_rows])) if abs_rows else None
            ),
            "stale_resurrection_post_consolidation": (
                float(np.mean([1.0 if r["arms"][arm].get("stale_resurrected_post", False) else 0.0 for r in stale_rows]))
                if stale_rows
                else None
            ),
            "stale_resurrection_delta0s_immediate": (
                float(np.mean([1.0 if r["arms"][arm].get("stale_resurrected_delta0s", False) else 0.0 for r in stale_rows]))
                if stale_rows and "stale_resurrected_delta0s" in stale_rows[0]["arms"][arm]
                else None
            ),
            "accuracy_delta0s_immediate": (
                float(np.mean([1.0 if r["arms"][arm].get("correct_delta0s", False) else 0.0 for r in rows]))
                if rows and "correct_delta0s" in rows[0]["arms"][arm]
                else None
            ),
            "clocks_ms": {
                "write_ack_p50_ms": float(np.quantile([r["arms"][arm]["write_ack_ms"] for r in rows], 0.50)),
                "write_ack_p95_ms": float(np.quantile([r["arms"][arm]["write_ack_ms"] for r in rows], 0.95)),
                "consolidation_bg_mean_ms": float(np.mean([r["arms"][arm]["consolidation_bg_ms"] for r in rows])),
                "read_retrieval_p50_ms": float(np.quantile([r["arms"][arm]["read_retrieval_ms"] for r in rows], 0.50)),
                "read_retrieval_p95_ms": float(np.quantile([r["arms"][arm]["read_retrieval_ms"] for r in rows], 0.95)),
                "reader_llm_mean_ms": float(np.mean([r["arms"][arm]["reader_llm_ms"] for r in rows])),
            },
            "tokens": {
                "mean_write_llm_tokens": float(np.mean([r["arms"][arm]["write_llm_tokens"] for r in rows])),
                "mean_read_prompt_tokens": float(np.mean([r["arms"][arm]["read_prompt_tokens"] for r in rows])),
                "mean_total_llm_tokens": float(
                    np.mean([r["arms"][arm]["write_llm_tokens"] + r["arms"][arm]["read_prompt_tokens"] for r in rows])
                ),
            },
            "schema_anomaly_counts": {
                k: int(sum(1 for r in rows if r["arms"][arm]["judge"]["schema_anomalies"].get(k, False)))
                for k in ("empty_output", "bare_abstract_token", "found_n_matching_records_template", "strict_abstain_token")
            },
        }

    comparisons: Dict[str, Any] = {}
    for other in arm_keys:
        if other == primary_arm:
            continue
        comparisons[f"{primary_arm}_minus_{other}"] = paired_bootstrap_wave6(
            acc_maps[primary_arm], acc_maps[other], expected_ids=item_ids
        )
    return {
        "suite": suite_name,
        "n_items": len(rows),
        "arms": arm_stats,
        "paired_comparisons_post_consolidation": comparisons,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Wave-6 K5 Zero-Strawman 4-Way Competitor Arena")
    parser.add_argument("--vllm-endpoint", type=str, default="http://127.0.0.1:8891/v1")
    parser.add_argument("--vllm-model", type=str, default="nemotron-lightning")
    parser.add_argument("--judge-hf-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--lme-json", type=Path, required=True)
    parser.add_argument("--osv-cache", type=Path, required=True)
    parser.add_argument("--qdrant-url", type=str, default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    t_start = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = args.output_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)

    import torch
    from sentence_transformers import CrossEncoder, SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Wave6-PM3v2] Loading GPU BAAI/bge-m3 and BAAI/bge-reranker-v2-m3 on {dev}...", flush=True)
    bge = SentenceTransformer("BAAI/bge-m3", device=dev)
    bge.max_seq_length = 384
    reranker = CrossEncoder("BAAI/bge-reranker-v2-m3", device=dev)

    def bge_encode(texts: Sequence[str]) -> np.ndarray:
        return np.asarray(
            bge.encode(list(texts), batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False),
            dtype=np.float32,
        )

    def rerank_predict(pairs: Sequence[Tuple[str, str]]) -> np.ndarray:
        return np.asarray(reranker.predict(list(pairs), batch_size=64, show_progress_bar=False), dtype=np.float64)

    arm_keys = (
        "Chelated_v4_Stratified_WFS",
        "Mem0_v2_2_1_Live_InferTrue",
        "Mem0_v2_2_1_Live_InferFalse",
        "Hindsight_4Net_RRR_Emulation",
        "OpenSearch_Memory_Async_Emulation",
    )

    # =========================================================================
    # Phase 1: Generate Predictions & Measure All 4 Clocks on Suite A (N=60)
    # =========================================================================
    cases_a, all_docs_a, corpus_sha_a = build_prospective_realworld_drift_suite(args.osv_cache)
    docs_by_id = {d.doc_id: d for d in all_docs_a}
    ckpt_a_path = work_dir / "_ckpt_suite_a.json"
    raw_suite_a: List[Dict[str, Any]] = (
        json.loads(ckpt_a_path.read_text(encoding="utf-8")) if ckpt_a_path.is_file() else []
    )
    done_a = {str(r["item_id"]) for r in raw_suite_a}

    print(
        f"[Wave6-PM3v2] Running Suite A (N={len(cases_a)}, cached={len(done_a)} Regulated Enterprise Drift & Retraction)...",
        flush=True,
    )
    for idx_c, case in enumerate(cases_a, start=1):
        cid = str(case["case_id"])
        if cid in done_a:
            continue
        q_txt = str(case["query_text"])
        gold = str(case["gold"])
        stale_vals = list(case.get("stale_or_false_values") or [])
        c_docs: List[RawDocumentEnvelope] = [docs_by_id[did] for did in case["doc_ids"]]
        c_docs_sorted = sorted(c_docs, key=lambda d: (d.ingested_ts, d.published_ts, d.doc_id))
        doc_lines = [
            f"[source={d.source_channel}, published_day={d.published_ts}] {d.text}"
            for d in c_docs_sorted
        ]
        input_sha = hashlib.sha256("\n".join(doc_lines).encode("utf-8")).hexdigest()
        arms_out: Dict[str, Any] = {}

        # 1) Chelated v4 Stratified WFS (`max_tokens=256` claim extraction + EK5 v3 synchronous mutation)
        t_ch_extract = time.perf_counter()
        raw_claims: List[ExtractedClaimRecord] = []
        ch_write_toks = 0
        for d in c_docs_sorted:
            u_msg = f"Source channel: {d.source_channel}\nPublished day: {d.published_ts}\nDocument:\n{d.text}"
            r_txt, ptoks, _ = call_vllm_timed(args.vllm_endpoint, args.vllm_model, EXTRACTION_SYSTEM_PROMPT_V2, u_msg, max_tokens=256)
            ch_write_toks += ptoks
            raw_claims.append(parse_extracted_claim_json(d.doc_id, r_txt))
        ch_extract_ms = (time.perf_counter() - t_ch_extract) * 1000.0

        t_ch_ack = time.perf_counter()
        d_vecs = bge_encode([d.text for d in c_docs_sorted])
        s_vecs = bge_encode([f"{c.entity} {c.slot}" for c in raw_claims])
        exo = AuthorityDominatedExocortexV2(c_docs_sorted, raw_claims, d_vecs, s_vecs)
        bm_exo = FastBM25Index([d.doc_id for d in c_docs_sorted], [d.text for d in c_docs_sorted])
        ch_ack_ms = (time.perf_counter() - t_ch_ack) * 1000.0

        t_ch_r = time.perf_counter()
        qv_ch = bge_encode([q_txt])[0]
        hyb_ch = _rrf_combine(bm_exo.score_query(q_txt), l2_normalize(d_vecs) @ qv_ch)
        sel_ch = exo.select_context_for_query(hyb_ch, q_txt, top_k=5)
        ch_read_ms = (time.perf_counter() - t_ch_r) * 1000.0
        ch_ctx = str(sel_ch["formatted_context"])
        if sel_ch["status"] == "CONTESTED_ABSTAIN":
            ch_ans, ch_rtoks, ch_llm_ms = "ABSTAIN", 0, 0.0
        else:
            ch_ans, ch_rtoks, ch_llm_ms = call_vllm_timed(
                args.vllm_endpoint,
                args.vllm_model,
                ENTERPRISE_READER_SYSTEM,
                f"Records:\n{ch_ctx}\n\nQuestion: {q_txt}\nAnswer:",
                max_tokens=128,
            )
        ch_stale = check_stale_resurrection(ch_ctx, stale_vals)
        # Chelated EK5 v3 applies synchronous graph mutation on write (`wait=true`), so Delta=0s == Post-Consolidation!
        arms_out["Chelated_v4_Stratified_WFS"] = {
            "pred": ch_ans,
            "pred_delta0s": ch_ans,
            "stale_resurrected_post": ch_stale,
            "stale_resurrected_delta0s": ch_stale,
            "write_ack_ms": ch_ack_ms,
            "consolidation_bg_ms": ch_extract_ms,
            "read_retrieval_ms": ch_read_ms,
            "reader_llm_ms": ch_llm_ms,
            "write_llm_tokens": ch_write_toks,
            "read_prompt_tokens": ch_rtoks,
        }

        # 2 & 3) Live mem0ai 2.2.1 (infer=True & infer=False)
        for infer_flag, m_key in ((True, "Mem0_v2_2_1_Live_InferTrue"), (False, "Mem0_v2_2_1_Live_InferFalse")):
            qpath = Path(tempfile.mkdtemp(prefix=f"m0_{idx_c}_", dir=str(work_dir)))
            mem, tbox = make_live_mem0_gpu(args.vllm_endpoint, args.vllm_model, qpath, infer_prompt=infer_flag, qdrant_url=args.qdrant_url)
            try:
                t_mw = time.perf_counter()
                if infer_flag:
                    safe_mem0_add_infer(mem, "\n".join(doc_lines), cid, tbox)
                else:
                    for dl in doc_lines:
                        mem.add(dl, user_id=cid, infer=False)
                mw_ms = (time.perf_counter() - t_mw) * 1000.0
                t_mr = time.perf_counter()
                found = mem.search(q_txt, top_k=8, filters={"user_id": cid}).get("results", [])
                mr_ms = (time.perf_counter() - t_mr) * 1000.0
            finally:
                close_live_mem0(mem, qpath, qdrant_url=args.qdrant_url)
            m_ctx = "\n".join(f"- {r.get('memory', '')}" for r in found) if found else "(No memories found.)"
            m_ans, m_rtoks, m_llm_ms = call_vllm_timed(
                args.vllm_endpoint,
                args.vllm_model,
                ENTERPRISE_READER_SYSTEM,
                f"Records:\n{m_ctx}\n\nQuestion: {q_txt}\nAnswer:",
                max_tokens=128,
            )
            m_stale = check_stale_resurrection(m_ctx, stale_vals)
            arms_out[m_key] = {
                "pred": m_ans,
                "pred_delta0s": m_ans,
                "stale_resurrected_post": m_stale,
                "stale_resurrected_delta0s": m_stale,
                "write_ack_ms": mw_ms,
                "consolidation_bg_ms": 0.0,
                "read_retrieval_ms": mr_ms,
                "reader_llm_ms": m_llm_ms,
                "write_llm_tokens": int(tbox["write_prompt_tokens"]),
                "read_prompt_tokens": m_rtoks,
            }

        # 4) Hindsight 4-Network Retain-Recall-Reflect Emulation
        hs = Hindsight4NetEmulationStore(bge_encode)
        hs_ack = hs.write_raw_experiences_ack(doc_lines)
        # Delta = 0s read before async Retain completes
        hs_ctx_d0, _ = hs.recall(q_txt, top_k=8, delta_zero_unconsolidated=True)
        hs_ans_d0, _, _ = call_vllm_timed(
            args.vllm_endpoint,
            args.vllm_model,
            HINDSIGHT_REFLECT_SYSTEM,
            f"Memory Banks:\n{hs_ctx_d0}\n\nQuestion: {q_txt}\nAnswer:",
            max_tokens=128,
        )
        hs_bg_ms, hs_wtoks = hs.retain_consolidate(chunk_strings(doc_lines, max_chars=12000), args.vllm_endpoint, args.vllm_model)
        hs_ctx, hs_read_ms = hs.recall(q_txt, top_k=10, delta_zero_unconsolidated=False)
        hs_ans, hs_rtoks, hs_llm_ms = call_vllm_timed(
            args.vllm_endpoint,
            args.vllm_model,
            HINDSIGHT_REFLECT_SYSTEM,
            f"Memory Banks:\n{hs_ctx}\n\nQuestion: {q_txt}\nAnswer:",
            max_tokens=128,
        )
        arms_out["Hindsight_4Net_RRR_Emulation"] = {
            "pred": hs_ans,
            "pred_delta0s": hs_ans_d0,
            "stale_resurrected_post": check_stale_resurrection(hs_ctx, stale_vals),
            "stale_resurrected_delta0s": check_stale_resurrection(hs_ctx_d0, stale_vals),
            "write_ack_ms": hs_ack,
            "consolidation_bg_ms": hs_bg_ms,
            "read_retrieval_ms": hs_read_ms,
            "reader_llm_ms": hs_llm_ms,
            "write_llm_tokens": hs_wtoks,
            "read_prompt_tokens": hs_rtoks,
        }

        # 5) AWS OpenSearch Memory-Parity Async SEMANTIC + SUMMARY Emulation
        osm = OpenSearchMemoryAsyncEmulation(bge_encode)
        osm_ack = osm.write_events_immediate(doc_lines)
        osm_ctx_d0, _ = osm.retrieve(q_txt, top_k=8, delta_zero_before_async=True)
        osm_ans_d0, _, _ = call_vllm_timed(
            args.vllm_endpoint,
            args.vllm_model,
            ENTERPRISE_READER_SYSTEM,
            f"Records:\n{osm_ctx_d0}\n\nQuestion: {q_txt}\nAnswer:",
            max_tokens=128,
        )
        osm_bg_ms, osm_wtoks = osm.run_async_strategies(chunk_strings(doc_lines, max_chars=12000), args.vllm_endpoint, args.vllm_model)
        osm_ctx, osm_read_ms = osm.retrieve(q_txt, top_k=10, delta_zero_before_async=False)
        osm_ans, osm_rtoks, osm_llm_ms = call_vllm_timed(
            args.vllm_endpoint,
            args.vllm_model,
            ENTERPRISE_READER_SYSTEM,
            f"Records:\n{osm_ctx}\n\nQuestion: {q_txt}\nAnswer:",
            max_tokens=128,
        )
        arms_out["OpenSearch_Memory_Async_Emulation"] = {
            "pred": osm_ans,
            "pred_delta0s": osm_ans_d0,
            "stale_resurrected_post": check_stale_resurrection(osm_ctx, stale_vals),
            "stale_resurrected_delta0s": check_stale_resurrection(osm_ctx_d0, stale_vals),
            "write_ack_ms": osm_ack,
            "consolidation_bg_ms": osm_bg_ms,
            "read_retrieval_ms": osm_read_ms,
            "reader_llm_ms": osm_llm_ms,
            "write_llm_tokens": osm_wtoks,
            "read_prompt_tokens": osm_rtoks,
        }

        raw_suite_a.append({
            "item_id": cid,
            "regime": str(case["regime"]),
            "question": q_txt,
            "gold": gold,
            "abstention_required": (gold.strip().upper() == "ABSTAIN"),
            "stale_values": stale_vals,
            "input_sha256": input_sha,
            "arms": arms_out,
        })
        write_json_atomic(ckpt_a_path, raw_suite_a)
        if idx_c % 10 == 0 or idx_c == len(cases_a):
            print(f"[Wave6-PM3v2] Suite A progress: {idx_c}/{len(cases_a)}", flush=True)

    # =========================================================================
    # Phase 2: Generate Predictions & Measure All 4 Clocks on Suite B (N=40 Live LME)
    # =========================================================================
    lme_records = json.loads(args.lme_json.read_text(encoding="utf-8"))
    selected_b = select_lme_items(lme_records)
    ckpt_b_path = work_dir / "_ckpt_suite_b.json"
    raw_suite_b: List[Dict[str, Any]] = (
        json.loads(ckpt_b_path.read_text(encoding="utf-8")) if ckpt_b_path.is_file() else []
    )
    done_b = {str(r["item_id"]) for r in raw_suite_b}

    print(
        f"[Wave6-PM3v2] Running Suite B (N={len(selected_b)}, cached={len(done_b)} Live LongMemEval-S Timelines on GPU)...",
        flush=True,
    )
    for idx_b, packed in enumerate(selected_b, start=1):
        item = packed["item"]
        qid = str(item["question_id"])
        if qid in done_b:
            continue
        kind = str(packed["kind"])
        abstention_req = (kind == "abstention")
        gold_b = str(item.get("answer") or "")
        q_b = str(item["question"])
        msgs = raw_conversational_messages(item, inject_echoes=(kind == "knowledge-update"))
        msg_lines = [
            f"[{datetime.datetime.fromtimestamp(float(m['ts']), tz=datetime.timezone.utc).strftime('%Y/%m/%d %H:%M')}] "
            f"{m['role']}: {m['content']}"
            for m in msgs
        ]
        msg_sha = hashlib.sha256("\n".join(msg_lines).encode("utf-8")).hexdigest()
        arms_b: Dict[str, Any] = {}

        # 1) Chelated v4 GPU (zero write LLM tokens, zero CPU lock contention!)
        ch_b_ctx, ch_b_wack, ch_b_read, ch_b_stat = chelated_conversational_gpu_retrieve(
            msgs, q_b, bge_encode, rerank_predict
        )
        ch_b_ans, ch_b_rtoks, ch_b_llm_ms = call_vllm_timed(
            args.vllm_endpoint,
            args.vllm_model,
            CONVERSATIONAL_READER_SYSTEM,
            f"Records:\n{ch_b_ctx}\n\nQuestion: {q_b}\nAnswer:",
            max_tokens=128,
        )
        arms_b["Chelated_v4_Stratified_WFS"] = {
            "pred": ch_b_ans,
            "status": ch_b_stat,
            "write_ack_ms": ch_b_wack,
            "consolidation_bg_ms": 0.0,
            "read_retrieval_ms": ch_b_read,
            "reader_llm_ms": ch_b_llm_ms,
            "write_llm_tokens": 0,
            "read_prompt_tokens": ch_b_rtoks,
        }

        # 2 & 3) Live mem0ai 2.2.1 (infer=True & infer=False)
        chunks_m0 = chunk_strings(msg_lines, max_chars=32000)
        raw_blocks_m0 = chunk_strings(msg_lines, max_chars=900)
        for infer_flag, m_key in ((True, "Mem0_v2_2_1_Live_InferTrue"), (False, "Mem0_v2_2_1_Live_InferFalse")):
            qpath = Path(tempfile.mkdtemp(prefix=f"m0_b_{idx_b}_", dir=str(work_dir)))
            mem, tbox = make_live_mem0_gpu(args.vllm_endpoint, args.vllm_model, qpath, infer_prompt=infer_flag, qdrant_url=args.qdrant_url)
            try:
                t_mw = time.perf_counter()
                if infer_flag:
                    for ch_str in chunks_m0:
                        safe_mem0_add_infer(mem, ch_str, qid, tbox)
                else:
                    for b_start in range(0, len(raw_blocks_m0), 32):
                        batch_msgs = [
                            {"role": "user", "content": blk}
                            for blk in raw_blocks_m0[b_start : b_start + 32]
                        ]
                        mem.add(batch_msgs, user_id=qid, infer=False)
                mw_ms = (time.perf_counter() - t_mw) * 1000.0
                t_mr = time.perf_counter()
                found = mem.search(q_b, top_k=12, filters={"user_id": qid}).get("results", [])
                mr_ms = (time.perf_counter() - t_mr) * 1000.0
            finally:
                close_live_mem0(mem, qpath, qdrant_url=args.qdrant_url)
            m_ctx = "\n".join(f"- {r.get('memory', '')}" for r in found) if found else "(No memories found.)"
            m_ans, m_rtoks, m_llm_ms = call_vllm_timed(
                args.vllm_endpoint,
                args.vllm_model,
                CONVERSATIONAL_READER_SYSTEM,
                f"Records:\n{m_ctx}\n\nQuestion: {q_b}\nAnswer:",
                max_tokens=128,
            )
            arms_b[m_key] = {
                "pred": m_ans,
                "write_ack_ms": mw_ms,
                "consolidation_bg_ms": 0.0,
                "read_retrieval_ms": mr_ms,
                "reader_llm_ms": m_llm_ms,
                "write_llm_tokens": int(tbox["write_prompt_tokens"]),
                "read_prompt_tokens": m_rtoks,
            }

        # 4) Hindsight 4-Network Retain-Recall-Reflect Emulation
        hs_b = Hindsight4NetEmulationStore(bge_encode)
        hs_b_ack = hs_b.write_raw_experiences_ack(msg_lines)
        hs_b_bg, hs_b_wtoks = hs_b.retain_consolidate(chunks_m0, args.vllm_endpoint, args.vllm_model)
        hs_b_ctx, hs_b_read = hs_b.recall(q_b, top_k=12, delta_zero_unconsolidated=False)
        hs_b_ans, hs_b_rtoks, hs_b_llm = call_vllm_timed(
            args.vllm_endpoint,
            args.vllm_model,
            CONVERSATIONAL_READER_SYSTEM,
            f"Records:\n{hs_b_ctx}\n\nQuestion: {q_b}\nAnswer:",
            max_tokens=128,
        )
        arms_b["Hindsight_4Net_RRR_Emulation"] = {
            "pred": hs_b_ans,
            "write_ack_ms": hs_b_ack,
            "consolidation_bg_ms": hs_b_bg,
            "read_retrieval_ms": hs_b_read,
            "reader_llm_ms": hs_b_llm,
            "write_llm_tokens": hs_b_wtoks,
            "read_prompt_tokens": hs_b_rtoks,
        }

        # 5) OpenSearch Memory-Parity Async SEMANTIC + SUMMARY Emulation
        osm_b = OpenSearchMemoryAsyncEmulation(bge_encode)
        osm_b_ack = osm_b.write_events_immediate(msg_lines)
        osm_b_bg, osm_b_wtoks = osm_b.run_async_strategies(chunks_m0, args.vllm_endpoint, args.vllm_model)
        osm_b_ctx, osm_b_read = osm_b.retrieve(q_b, top_k=12, delta_zero_before_async=False)
        osm_b_ans, osm_b_rtoks, osm_b_llm = call_vllm_timed(
            args.vllm_endpoint,
            args.vllm_model,
            CONVERSATIONAL_READER_SYSTEM,
            f"Records:\n{osm_b_ctx}\n\nQuestion: {q_b}\nAnswer:",
            max_tokens=128,
        )
        arms_b["OpenSearch_Memory_Async_Emulation"] = {
            "pred": osm_b_ans,
            "write_ack_ms": osm_b_ack,
            "consolidation_bg_ms": osm_b_bg,
            "read_retrieval_ms": osm_b_read,
            "reader_llm_ms": osm_b_llm,
            "write_llm_tokens": osm_b_wtoks,
            "read_prompt_tokens": osm_b_rtoks,
        }

        raw_suite_b.append({
            "item_id": qid,
            "regime": kind,
            "question": q_b,
            "gold": gold_b,
            "abstention_required": abstention_req,
            "stale_values": [],
            "input_sha256": msg_sha,
            "arms": arms_b,
        })
        write_json_atomic(ckpt_b_path, raw_suite_b)
        if idx_b % 5 == 0 or idx_b == len(selected_b):
            print(f"[Wave6-PM3v2] Suite B progress: {idx_b}/{len(selected_b)}", flush=True)

    # =========================================================================
    # Phase 3: Unload BGE/Reranker & Run Blind Cross-Family Judge (Qwen2.5-7B)
    # =========================================================================
    global _SHARED_MEM0_EMBEDDER
    _SHARED_MEM0_EMBEDDER = None
    bge = None
    reranker = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"[Wave6-PM3v2] Loading Cross-Family Blind Semantic Judge ({args.judge_hf_model})...", flush=True)
    j_tok = AutoTokenizer.from_pretrained(args.judge_hf_model, padding_side="left")
    if j_tok.pad_token_id is None:
        j_tok.pad_token_id = j_tok.eos_token_id
    j_dtype = torch.bfloat16 if (dev == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16
    j_mdl = AutoModelForCausalLM.from_pretrained(args.judge_hf_model, torch_dtype=j_dtype).to(dev)
    j_mdl.eval()

    def qwen_judge_complete(sys_prompt: str, user_prompt: str) -> str:
        prompt_str = j_tok.apply_chat_template(
            [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = j_tok([prompt_str], return_tensors="pt").to(dev)
        with torch.inference_mode():
            out = j_mdl.generate(**enc, max_new_tokens=20, do_sample=False, pad_token_id=j_tok.pad_token_id)
        return j_tok.decode(out[0, enc.input_ids.shape[1]:], skip_special_tokens=True).strip()

    blind_judge = BlindCrossFamilySemanticJudge(
        judge_family="alibaba-qwen2.5",
        generator_family="nvidia-nemotron",
        completion_fn=qwen_judge_complete,
    )

    for row in raw_suite_a:
        graded_post = blind_judge.grade_arms_blind(
            item_id=f"suiteA_post:{row['item_id']}",
            question=row["question"],
            gold_reference=row["gold"],
            arm_predictions={a: row["arms"][a]["pred"] for a in arm_keys},
            abstention_required=row["abstention_required"],
        )
        graded_d0 = blind_judge.grade_arms_blind(
            item_id=f"suiteA_d0:{row['item_id']}",
            question=row["question"],
            gold_reference=row["gold"],
            arm_predictions={a: row["arms"][a]["pred_delta0s"] for a in arm_keys},
            abstention_required=row["abstention_required"],
        )
        for a in arm_keys:
            row["arms"][a]["judge"] = graded_post[a]
            row["arms"][a]["correct"] = graded_post[a]["correct"]
            row["arms"][a]["correct_delta0s"] = graded_d0[a]["correct"]

    for row in raw_suite_b:
        graded_b = blind_judge.grade_arms_blind(
            item_id=f"suiteB:{row['item_id']}",
            question=row["question"],
            gold_reference=row["gold"],
            arm_predictions={a: row["arms"][a]["pred"] for a in arm_keys},
            abstention_required=row["abstention_required"],
        )
        for a in arm_keys:
            row["arms"][a]["judge"] = graded_b[a]
            row["arms"][a]["correct"] = graded_b[a]["correct"]

    summary_a = summarize_suite("Suite_A_Regulated_Enterprise_Drift_N60", raw_suite_a, arm_keys)
    summary_b = summarize_suite("Suite_B_Live_LongMemEval_Conversational_N40", raw_suite_b, arm_keys)

    csv_rows: List[Dict[str, Any]] = []
    for sname, srows in (("SuiteA_Enterprise_N60", raw_suite_a), ("SuiteB_Conversational_N40", raw_suite_b)):
        for r in srows:
            for a in arm_keys:
                ad = r["arms"][a]
                csv_rows.append({
                    "suite": sname,
                    "item_id": r["item_id"],
                    "regime": r["regime"],
                    "abstention_required": r["abstention_required"],
                    "arm": a,
                    "correct_post": ad["correct"],
                    "correct_delta0s": ad.get("correct_delta0s"),
                    "stale_resurrected_post": ad.get("stale_resurrected_post"),
                    "stale_resurrected_delta0s": ad.get("stale_resurrected_delta0s"),
                    "write_ack_ms": round(ad["write_ack_ms"], 3),
                    "consolidation_bg_ms": round(ad["consolidation_bg_ms"], 3),
                    "read_retrieval_ms": round(ad["read_retrieval_ms"], 3),
                    "reader_llm_ms": round(ad["reader_llm_ms"], 3),
                    "write_llm_tokens": ad["write_llm_tokens"],
                    "read_prompt_tokens": ad["read_prompt_tokens"],
                    "pred": ad["pred"][:240],
                })

    csv_name = "wave6_pm3_v2_arena_rows.csv"
    csv_path = args.output_dir / csv_name
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    manifest = {
        "schema_version": "wave6_pm3_v2_4way_arena_v1",
        "git_sha": current_git_sha(REPO_ROOT),
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "system_provenance_labels": {
            "Chelated_v4_Stratified_WFS": "native_chelated_v4_gpu_bge_m3_ek5_v3",
            "Mem0_v2_2_1_Live_InferTrue": "live_upstream_package_mem0ai_2_2_1_infer_true",
            "Mem0_v2_2_1_Live_InferFalse": "live_upstream_package_mem0ai_2_2_1_infer_false",
            "Hindsight_4Net_RRR_Emulation": "emulation (ours): Vectorize Hindsight-architecture 4-Network (World+Experience+EntitySummary+Beliefs) Retain->Recall->Reflect",
            "OpenSearch_Memory_Async_Emulation": "emulation (ours): AWS OpenSearch Memory-parity immediate Events Hybrid BM25+kNN + async SEMANTIC+SUMMARY strategies",
        },
        "judge_protocol": {
            "generator_model": args.vllm_model,
            "generator_family": "nvidia-nemotron",
            "blind_judge_model": args.judge_hf_model,
            "blind_judge_family": "alibaba-qwen2.5",
            "substring_abstain_markers_used": False,
            "clocks_pooled_across_suites": False,
        },
        "enterprise_corpus_sha256": corpus_sha_a,
        "suite_a_enterprise_n60": summary_a,
        "suite_b_conversational_n40": summary_b,
        "rows_csv": csv_name,
        "rows_csv_sha256": sha256_file(csv_path),
        "wall_seconds": round(time.perf_counter() - t_start, 2),
    }
    manifest_name = "wave6_pm3_v2_arena_manifest.json"
    m_sha = write_json_atomic(args.output_dir / manifest_name, manifest)
    receipt = {
        "schema_version": "wave6_receipt_v1",
        "bundle_dir": args.output_dir.name,
        "manifest": manifest_name,
        "manifest_sha256": m_sha,
        "artifact_sha256": {
            manifest_name: m_sha,
            csv_name: sha256_file(csv_path),
        },
        "generated_at_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "git_sha": current_git_sha(REPO_ROOT),
    }
    write_json_atomic(args.output_dir / "report_receipt.json", receipt)
    shutil.rmtree(work_dir, ignore_errors=True)
    print(json.dumps({
        "suite_a_acc": {k: v["accuracy"] for k, v in summary_a["arms"].items()},
        "suite_b_acc": {k: v["accuracy"] for k, v in summary_b["arms"].items()},
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()
