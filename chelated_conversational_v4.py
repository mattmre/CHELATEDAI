"""Chelated Conversational v4: Pre-Budget EK5 Echo Collapse, Chronological SUPERSEDE Chains, & Cross-Encoder Reranking.

Implements Wave-5 Cross-Judgment Guardrail K2 (`PM-1b v4`):
  1. Runs `EK5` complete-linkage echo collapse + chronological `SUPERSEDE` chain detection
     (`StratifiedEvidenceStore`) BEFORE shortlist truncation and session/turn budgeting.
  2. Drops the `J4` `max_sessions=6` session cap (`max_sessions=None` by default) so echo sessions
     can never evict genuine `knowledge-update` or `multi-session` evidence.
  3. Reranks the top-`pool_size` (`36`) post-collapse DualGran candidates with `BAAI/bge-reranker-v2-m3`
     while preserving linked `SUPERSEDE` partner turns (`[EARLIER SUPERSEDED RECORD]` +
     `[LATEST CURRENT UPDATE]`) and rendering chronologically within a compact prompt budget.
  4. Preserves exact byte-for-byte prompt parity with `epistemic=False` (`DualGran_Reranked_NoEK5`)
     whenever `gate_active == False`.
"""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Protocol, Sequence

import numpy as np

from ek5_v3_stratified import (
    StratifiedEvent,
    StratifiedEvidenceStore,
    classify_high_similarity_pair,
)
from em4_soft_telemetry_routing import (
    _EXTENDED_RECAP_RE,
    content_tokens_stripped,
    is_recap_framed,
    jaccard_similarity,
    strip_recap_and_speaker,
)
from scripts.night1_common import FastBM25Index

BGE_MODEL = "BAAI/bge-m3"
BGE_RERANKER_MODEL = "BAAI/bge-reranker-v2-m3"
BGE_DIMENSIONS = 1024
EARLIER = "[EARLIER SUPERSEDED RECORD]"
LATEST = "[LATEST CURRENT UPDATE]"

_CHANGE_V4 = re.compile(
    r"\b(?:no longer|used to|instead of|changed|switched|replaced|updated|moved|"
    r"relocated|upgraded|downgraded|increased|decreased|raised|lowered|dropped|traded|"
    r"now|these days|recently|currently|this time)\b",
    re.IGNORECASE,
)
_FIRST_PERSON_I_RE = re.compile(r"\bI(?=(?:'(?:m|ve|d|ll))?\b)")
_NON_ENTITY_SENTENCE_LEAD_RE = re.compile(
    r"^\s*(Right|After|Before|During|Since|When|While|Once|Now|Still|Just|Back|"
    r"First|Final|Weekly|Monthly|Daily|Nightly|Evening|Morning|Pooled|Catalog|"
    r"Production|Fleet|Entering|Under|Following|Redirecting)\b"
)


def _normalize_claim_for_pair_check(claim_text: str) -> str:
    """Strip recap prefixes and de-capitalize first-person `I` / non-entity sentence leads."""
    cleaned = _EXTENDED_RECAP_RE.sub("", claim_text).strip()
    cleaned = _NON_ENTITY_SENTENCE_LEAD_RE.sub(lambda m: m.group(1).lower(), cleaned, count=1)
    cleaned = _FIRST_PERSON_I_RE.sub("i", cleaned)
    return cleaned


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


class Reranker(Protocol):
    def score_pairs(self, query: str, passages: Sequence[str]) -> np.ndarray: ...


@dataclass(frozen=True)
class ConversationTurn:
    turn_id: str
    session_id: str
    date: str
    timestamp: float
    text: str
    claim_text: str
    speaker: str = ""
    index_text: str | None = None

    @property
    def date_anchor(self) -> str:
        return f"[Session Date: {self.date}]"

    @property
    def indexed_text(self) -> str:
        return f"{self.date_anchor} {self.index_text or self.text}"


@dataclass(frozen=True)
class RetrievedTurn:
    turn: ConversationTurn
    status: str = ""
    max_chars: int | None = None

    def render(self) -> str:
        status = f"{self.status} " if self.status else ""
        body = self.turn.text
        if self.max_chars is not None and self.max_chars > 0 and len(body) > self.max_chars:
            body = body[: self.max_chars].rstrip() + " ..."
        return f"- {self.turn.date_anchor} {status}{body}"


@dataclass(frozen=True)
class ConversationRecall:
    ranked_turn_ids: tuple[str, ...]
    turns: tuple[RetrievedTurn, ...]
    echo_clusters: tuple[tuple[str, ...], ...] = ()
    supersessions: tuple[tuple[str, str], ...] = ()
    header_line: str = ""
    reranked_shortlist_ids: tuple[str, ...] = ()

    @property
    def gate_active(self) -> bool:
        return bool(self.echo_clusters or self.supersessions)

    @property
    def context(self) -> str:
        body = "\n".join(turn.render() for turn in self.turns)
        if self.header_line and body:
            return f"{self.header_line}\n{body}"
        return body


@dataclass(frozen=True)
class _PreBudgetGating:
    post_collapse_order: tuple[int, ...]
    suppressed: frozenset[int] = frozenset()
    tags: tuple[tuple[int, str], ...] = ()
    clusters: tuple[tuple[int, ...], ...] = ()
    supersessions: tuple[tuple[int, int], ...] = ()


def _normalized(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12)


def _minmax_1d(values: np.ndarray) -> np.ndarray:
    if len(values) == 0:
        return np.zeros(0, dtype=np.float64)
    lo = float(np.min(values))
    hi = float(np.max(values))
    if hi - lo <= 1e-12:
        return np.zeros_like(values, dtype=np.float64)
    return (values.astype(np.float64) - lo) / (hi - lo)


def _rrf(*scores: np.ndarray) -> np.ndarray:
    """Reciprocal Rank Fusion, Cormack, Clarke & Buettcher (SIGIR 2009), tie-aware."""
    combined = np.zeros(len(scores[0]), dtype=np.float64)
    for score in scores:
        order = np.argsort(-score, kind="stable")
        ranked = score[order]
        tied_ranks = np.maximum.accumulate(
            np.where(np.r_[True, ranked[1:] != ranked[:-1]], np.arange(len(order)), 0)
        )
        combined[order] += 1.0 / (60.0 + tied_ranks + 1)
    return combined


class CachedBGEReranker:
    """GPU/CPU wrapper for `BAAI/bge-reranker-v2-m3` with deterministic pair caching."""

    def __init__(
        self,
        model_name_or_path: str = BGE_RERANKER_MODEL,
        *,
        device: str | None = None,
        batch_size: int = 36,
        max_length: int = 512,
        local_files_only: bool = False,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.batch_size = max(1, int(batch_size))
        self.max_length = max(64, int(max_length))
        dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, local_files_only=local_files_only
        )
        self._model = AutoModelForSequenceClassification.from_pretrained(
            model_name_or_path,
            torch_dtype=dtype,
            local_files_only=local_files_only,
        ).to(device)
        self._model.eval()
        self._cache: dict[tuple[str, str], float] = {}

    def score_pairs(self, query: str, passages: Sequence[str]) -> np.ndarray:
        import torch

        if not passages:
            return np.zeros(0, dtype=np.float64)
        missing_keys: list[tuple[str, str]] = []
        seen_missing: set[tuple[str, str]] = set()
        for p in passages:
            key = (query, p)
            if key not in self._cache and key not in seen_missing:
                seen_missing.add(key)
                missing_keys.append(key)

        if missing_keys:
            with torch.inference_mode():
                for start in range(0, len(missing_keys), self.batch_size):
                    batch = missing_keys[start : start + self.batch_size]
                    enc = self._tokenizer(
                        [q for q, _ in batch],
                        [p for _, p in batch],
                        padding=True,
                        truncation=True,
                        max_length=self.max_length,
                        return_tensors="pt",
                    ).to(self.device)
                    logits = self._model(**enc, return_dict=True).logits.view(-1).float().cpu().numpy()
                    for key, val in zip(batch, logits, strict=True):
                        self._cache[key] = float(val)

        return np.asarray([self._cache[(query, p)] for p in passages], dtype=np.float64)

    def clear_cache(self) -> None:
        self._cache.clear()


class ChelatedConversationalV4:
    """Dual-granularity conversational engine with pre-budget EK5 collapse & BGE-reranker-v2-m3."""

    def __init__(
        self,
        turns: Sequence[ConversationTurn],
        embedder: Embedder | None = None,
        reranker: Reranker | None = None,
    ) -> None:
        self.turns = tuple(turns)
        turn_ids = [turn.turn_id for turn in self.turns]
        if len(set(turn_ids)) != len(turn_ids):
            raise ValueError("Conversation turn IDs must be unique")
        if any(not t.session_id or not t.date or not math.isfinite(t.timestamp) for t in self.turns):
            raise ValueError("Each turn needs a session ID, date, and finite timestamp")

        if embedder is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(BGE_MODEL)

            class _DefaultBGEEmbedder:
                def encode(self, texts: Sequence[str]) -> np.ndarray:
                    return model.encode(
                        list(texts),
                        normalize_embeddings=True,
                        convert_to_numpy=True,
                        show_progress_bar=False,
                    )

            embedder = _DefaultBGEEmbedder()

        self._embedder = embedder
        self._reranker = reranker

        self._session_turns: dict[str, list[int]] = defaultdict(list)
        for i, turn in enumerate(self.turns):
            self._session_turns[turn.session_id].append(i)
        self._positions: dict[int, int] = {}
        for indices in self._session_turns.values():
            indices.sort(key=lambda i: (self.turns[i].timestamp, i))
            self._positions.update((i, pos) for pos, i in enumerate(indices))

        self._session_ids = tuple(self._session_turns)
        self._session_index = {sid: i for i, sid in enumerate(self._session_ids)}
        self._turn_sessions = np.asarray([self._session_index[t.session_id] for t in self.turns])

        dated = [turn.indexed_text for turn in self.turns]
        self._turn_bm25 = FastBM25Index(turn_ids, dated)
        self._session_bm25 = FastBM25Index(
            self._session_ids,
            ["\n".join(dated[i] for i in indices) for indices in self._session_turns.values()],
        )
        if self.turns:
            self._turn_vectors = self._encode(dated)
            self._session_vectors = _normalized(
                np.stack([self._turn_vectors[indices].mean(axis=0) for indices in self._session_turns.values()])
            )
            cleaned_claims = [strip_recap_and_speaker(t.claim_text) or t.claim_text for t in self.turns]
            self._claim_vectors = self._encode(cleaned_claims)
        else:
            self._turn_vectors = np.zeros((0, BGE_DIMENSIONS), dtype=np.float32)
            self._session_vectors = np.zeros((0, BGE_DIMENSIONS), dtype=np.float32)
            self._claim_vectors = np.zeros((0, BGE_DIMENSIONS), dtype=np.float32)

        self._claim_tokens = [content_tokens_stripped(turn.claim_text) for turn in self.turns]
        self._named_speakers = tuple(
            sorted(
                {
                    turn.speaker.strip()
                    for turn in self.turns
                    if turn.speaker.strip()
                    and turn.speaker.strip().casefold() not in {"user", "assistant"}
                }
            )
        )

    def _encode(self, texts: Sequence[str]) -> np.ndarray:
        vectors = np.asarray(self._embedder.encode(texts), dtype=np.float32)
        if vectors.shape != (len(texts), BGE_DIMENSIONS) or not np.isfinite(vectors).all():
            raise ValueError(f"Expected finite {BGE_DIMENSIONS}-dimensional BGE-M3 embeddings")
        return _normalized(vectors)

    def rank_with_scores(self, question: str) -> tuple[tuple[int, ...], np.ndarray, np.ndarray]:
        """Return `(turn_order, combined_turn_scores, session_rrf_scores)`."""
        if not self.turns:
            return (), np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)
        query = self._encode([question])[0]
        turn_rrf = _rrf(self._turn_bm25.score_query(question), self._turn_vectors @ query)
        session_rrf = _rrf(self._session_bm25.score_query(question), self._session_vectors @ query)
        combined = turn_rrf + 0.45 * session_rrf[self._turn_sessions]
        return tuple(map(int, np.argsort(-combined, kind="stable"))), combined, session_rrf

    def rank(self, question: str) -> tuple[int, ...]:
        order, _, _ = self.rank_with_scores(question)
        return order

    def _pair_cosine(self, i: int, j: int) -> float:
        turn_cos = float(self._turn_vectors[i] @ self._turn_vectors[j])
        claim_cos = float(self._claim_vectors[i] @ self._claim_vectors[j])
        return max(turn_cos, claim_cos)

    def _is_verified_echo_pair(self, i: int, j: int) -> bool:
        a, b = self.turns[i], self.turns[j]
        if a.session_id == b.session_id or a.speaker.casefold() != b.speaker.casefold():
            return False
        ta, tb = self._claim_tokens[i], self._claim_tokens[j]
        if len(ta & tb) < 2:
            return False
        cosine = self._pair_cosine(i, j)
        overlap = jaccard_similarity(ta, tb)
        if cosine < 0.80 and overlap < 0.55:
            return False
        action, _ = classify_high_similarity_pair(
            _normalize_claim_for_pair_check(a.claim_text),
            _normalize_claim_for_pair_check(b.claim_text),
        )
        return action == "ECHO_COLLAPSE" and overlap >= 0.30

    def _detect_supersede_pair(self, i: int, j: int) -> tuple[int, int] | None:
        """Return `(newer_idx, older_idx)` if `(i, j)` forms an authorized chronological update."""
        a, b = self.turns[i], self.turns[j]
        if a.session_id == b.session_id or a.speaker.casefold() != b.speaker.casefold():
            return None
        ta, tb = self._claim_tokens[i], self._claim_tokens[j]
        if len(ta & tb) < 2:
            return None
        cosine = self._pair_cosine(i, j)
        overlap = jaccard_similarity(ta, tb)
        if cosine < 0.80 and overlap < 0.55:
            return None
        action, reason = classify_high_similarity_pair(
            _normalize_claim_for_pair_check(a.claim_text),
            _normalize_claim_for_pair_check(b.claim_text),
        )
        if action == "ECHO_COLLAPSE":
            return None

        a_recap = is_recap_framed(a.claim_text)
        b_recap = is_recap_framed(b.claim_text)
        if a_recap != b_recap:
            earlier_i, later_i = (i, j) if a_recap else (j, i)
        else:
            earlier_i, later_i = sorted((i, j), key=lambda k: self.turns[k].timestamp)
        earlier, later = self.turns[earlier_i], self.turns[later_i]
        if not (a_recap != b_recap) and (earlier.date == later.date or earlier.timestamp >= later.timestamp):
            return None

        tightly_coupled = overlap >= 0.60 and reason in {
            "CONFLICT_NUMERIC_OR_CODE",
            "CONFLICT_MINIMAL_PAIR_SUBSTITUTION",
        }
        explicitly_changed = overlap >= 0.40 and bool(_CHANGE_V4.search(later.claim_text))
        if tightly_coupled or explicitly_changed:
            return (later_i, earlier_i)
        return None

    def _gating_pre_budget(self, order: tuple[int, ...], pool_size: int) -> _PreBudgetGating:
        """Collapse cross-session complete-linkage echoes BEFORE shortlist truncation & budgeting.

        When zero echoes exist in `order[:pool_size]`, stops at `pool_size` so clean non-gated
        queries retain exact byte-for-byte shortlist parity with `epistemic=False`.
        When echoes flood top ranks, continues scanning `order` until `pool_size` distinct
        complete-linkage groups are assembled, anchoring each cluster to its earliest timestamp.
        """
        if not order:
            return _PreBudgetGating(())

        rank_pos = {idx: pos for pos, idx in enumerate(order)}
        scan_limit = min(len(order), max(pool_size * 4, 144))
        groups: list[list[int]] = []
        scanned_end = 0

        for pos in range(scan_limit):
            idx = order[pos]
            scanned_end = pos + 1
            for group in groups:
                if all(self._is_verified_echo_pair(idx, member) for member in group):
                    group.append(idx)
                    break
            else:
                groups.append([idx])
            if len(groups) >= pool_size:
                break

        # If echoes were detected inside the top shortlist, absorb any older origin echo in
        # the remaining scan window into ALREADY-ADMITTED groups (without adding new groups)
        # so the earliest origin timestamp is always recovered.
        has_echoes = any(len(g) > 1 for g in groups)
        if has_echoes and scanned_end < scan_limit:
            for pos in range(scanned_end, scan_limit):
                idx = order[pos]
                for group in groups:
                    if all(self._is_verified_echo_pair(idx, member) for member in group):
                        group.append(idx)
                        break

        sorted_groups = [
            tuple(sorted(g, key=lambda i: (self.turns[i].timestamp, rank_pos[i])))
            for g in groups
        ]
        clusters = tuple(g for g in sorted_groups if len(g) > 1)
        representative: dict[int, int] = {member: g[0] for g in clusters for member in g}
        suppressed = frozenset(member for member, rep in representative.items() if member != rep)

        # Post-collapse representatives occupy the best Stage-1 rank achieved by their cluster.
        post_collapse_reps = tuple(g[0] for g in sorted_groups)

        # Detect chronological SUPERSEDE chains across the post-collapse representatives.
        candidate_edges: set[tuple[int, int]] = set()
        for p, i in enumerate(post_collapse_reps):
            for q in range(p + 1, len(post_collapse_reps)):
                j = post_collapse_reps[q]
                edge = self._detect_supersede_pair(i, j)
                if edge is not None:
                    newer, older = edge
                    if self.turns[newer].timestamp > self.turns[older].timestamp:
                        candidate_edges.add((newer, older))

        if not clusters and not candidate_edges:
            return _PreBudgetGating(order)

        routed_tail = tuple(
            idx for idx in order[scanned_end:]
            if idx not in suppressed and idx not in set(post_collapse_reps)
        )
        routed_order = tuple(dict.fromkeys((*post_collapse_reps, *routed_tail)))

        if not candidate_edges:
            return _PreBudgetGating(routed_order, suppressed, clusters=clusters)

        store = StratifiedEvidenceStore()
        node_indices = sorted({i for pair in candidate_edges for i in pair})
        store.apply_mutation(
            [
                StratifiedEvent(
                    event_id=self.turns[i].turn_id,
                    text=self.turns[i].claim_text,
                    valid_time=round(self.turns[i].timestamp * 1000),
                )
                for i in node_indices
            ],
            [
                (self.turns[newer].turn_id, self.turns[older].turn_id, "SUPERSEDE")
                for newer, older in sorted(candidate_edges)
            ],
        )
        edges = tuple(
            (newer, older)
            for newer, older in sorted(candidate_edges)
            if self.turns[older].turn_id in store.sup_out.get(self.turns[newer].turn_id, ())
        )
        tags = tuple(
            (i, EARLIER if store.eligibility[self.turns[i].turn_id] == "AUDIT_ONLY" else LATEST)
            for i in sorted({i for edge in edges for i in edge})
        )
        return _PreBudgetGating(routed_order, suppressed, tags, clusters, edges)

    def _rerank_passage(self, index: int, *, max_chars: int = 1200, include_prev_turn: bool = True) -> str:
        turn = self.turns[index]
        body = turn.text
        if max_chars > 0 and len(body) > max_chars:
            body = body[:max_chars].rstrip() + " ..."
        if include_prev_turn:
            pos = self._positions[index]
            if pos > 0:
                prev_idx = self._session_turns[turn.session_id][pos - 1]
                prev_body = self.turns[prev_idx].text
                if len(prev_body) > 220:
                    prev_body = prev_body[:220].rstrip() + " ..."
                return f"{turn.date_anchor} [Prior turn: {prev_body}] {body}"
        return f"{turn.date_anchor} {body}"

    def _rerank_shortlist(
        self,
        question: str,
        shortlist: tuple[int, ...],
        stage1_scores: np.ndarray,
        *,
        rerank_alpha: float = 0.80,
        rerank_max_chars: int = 1200,
        rerank_prev_turn: bool = True,
    ) -> tuple[int, ...]:
        if self._reranker is None or len(shortlist) <= 1:
            return shortlist
        passages = [
            self._rerank_passage(i, max_chars=rerank_max_chars, include_prev_turn=rerank_prev_turn)
            for i in shortlist
        ]
        raw_logits = np.asarray(self._reranker.score_pairs(question, passages), dtype=np.float64)
        if raw_logits.shape != (len(shortlist),) or not np.isfinite(raw_logits).all():
            raise ValueError("Reranker returned unexpected shape or non-finite scores")
        ce_probs = 1.0 / (1.0 + np.exp(-np.clip(raw_logits, -30.0, 30.0)))
        s1_vals = stage1_scores[list(shortlist)] if len(stage1_scores) == len(self.turns) else np.zeros(len(shortlist))
        s1_norm = _minmax_1d(s1_vals)
        alpha = float(np.clip(rerank_alpha, 0.0, 1.0))
        blended = alpha * ce_probs + (1.0 - alpha) * s1_norm
        order_in_shortlist = np.argsort(-blended, kind="stable")
        return tuple(shortlist[int(k)] for k in order_in_shortlist)

    def _unique_named_speaker(self, question: str) -> str | None:
        if len(self._named_speakers) < 2:
            return None
        q_lo = question.casefold()
        hits = [
            sp
            for sp in self._named_speakers
            if re.search(rf"\b{re.escape(sp.casefold())}(?:'s)?\b", q_lo)
        ]
        return hits[0] if len(hits) == 1 else None

    @staticmethod
    def _supersede_partners(
        supersessions: tuple[tuple[int, int], ...],
        tags: dict[int, str],
    ) -> tuple[dict[int, tuple[int, ...]], dict[int, tuple[int, ...]]]:
        """Map older->admitted latest updates and newer->direct superseded partners."""
        fwd: dict[int, set[int]] = defaultdict(set)  # older -> newer
        rev: dict[int, set[int]] = defaultdict(set)  # newer -> older
        nodes: set[int] = set()
        for newer, older in supersessions:
            fwd[older].add(newer)
            rev[newer].add(older)
            nodes.add(newer)
            nodes.add(older)

        latest_for_older: dict[int, tuple[int, ...]] = {}
        earlier_for_newer: dict[int, tuple[int, ...]] = {}
        for node in nodes:
            if tags.get(node) == EARLIER:
                visited: set[int] = set()
                stack = list(fwd.get(node, ()))
                sinks: list[int] = []
                while stack:
                    cur = stack.pop()
                    if cur in visited:
                        continue
                    visited.add(cur)
                    if tags.get(cur) == LATEST:
                        sinks.append(cur)
                    stack.extend(fwd.get(cur, ()))
                if sinks:
                    latest_for_older[node] = tuple(sorted(set(sinks)))
            if tags.get(node) == LATEST and rev.get(node):
                earlier_for_newer[node] = tuple(sorted(rev[node]))
        return latest_for_older, earlier_for_newer

    def retrieve(
        self,
        question: str,
        *,
        epistemic: bool = True,
        anchor_count: int = 8,
        window_radius: int = 1,
        context_chars: int = 7200,
        pool_size: int = 36,
        turn_max_chars: int | None = 1400,
        neighbor_max_chars: int | None = 700,
        per_session_cap: int = 3,
        rerank_alpha: float = 0.80,
        rerank_max_chars: int = 1200,
        rerank_prev_turn: bool = True,
        speaker_attribution_header: bool = True,
        speaker_anchor_reserve: int = 1,
        ranked: tuple[int, ...] | None = None,
        stage1_scores: np.ndarray | None = None,
    ) -> ConversationRecall:
        """Retrieve, collapse echoes pre-budget, rerank top-36, preserve SUPERSEDE chains chronologically."""
        if anchor_count < 1 or pool_size < anchor_count or window_radius < 0 or context_chars < 1:
            raise ValueError("Invalid anchors, context budget, window radius, or gate pool")
        if ranked is None:
            order, calc_scores, _ = self.rank_with_scores(question)
            if stage1_scores is None:
                stage1_scores = calc_scores
        else:
            order = ranked
            if stage1_scores is None:
                if self._reranker is not None and self.turns:
                    _, stage1_scores, _ = self.rank_with_scores(question)
                else:
                    stage1_scores = np.linspace(1.0, 0.0, num=len(self.turns), dtype=np.float64)

        if not order:
            return ConversationRecall((), ())

        # Step 1 (Guardrail K2.1): Run EK5 complete-linkage echo collapse + chronological SUPERSEDE
        # chain detection BEFORE shortlist truncation and session/turn budgeting.
        routing = self._gating_pre_budget(order, pool_size) if epistemic else _PreBudgetGating(order)
        tags = dict(routing.tags)

        # Step 2 (Guardrail K2.1): Rerank top-`pool_size` (36) post-collapse candidates with BGE-reranker-v2-m3.
        pre_rerank_shortlist = routing.post_collapse_order[:pool_size]
        reranked_shortlist = self._rerank_shortlist(
            question,
            pre_rerank_shortlist,
            stage1_scores,
            rerank_alpha=rerank_alpha,
            rerank_max_chars=rerank_max_chars,
            rerank_prev_turn=rerank_prev_turn,
        )
        candidate_order = (*reranked_shortlist, *routing.post_collapse_order[pool_size:])

        latest_for_older, earlier_for_newer = self._supersede_partners(routing.supersessions, tags)

        selected: dict[int, RetrievedTurn] = {}
        per_session: Counter[str] = Counter()
        anchors: list[int] = []

        named_subj = (
            self._unique_named_speaker(question)
            if (speaker_attribution_header or speaker_anchor_reserve > 0)
            else None
        )
        header_reserve = 220 if (speaker_attribution_header and named_subj is not None) else 0
        eff_budget = max(1, context_chars - header_reserve)
        used = 0

        def add_turn(index: int, *, char_cap: int | None) -> bool:
            nonlocal used
            if index in selected or index in routing.suppressed:
                return False
            turn = RetrievedTurn(self.turns[index], tags.get(index, ""), max_chars=char_cap)
            cost = len(turn.render()) + bool(selected)
            if used + cost > eff_budget:
                return False
            selected[index] = turn
            used += cost
            return True

        def admit_anchor_with_supersede_partners(index: int, *, enforce_session_cap: bool = True) -> bool:
            """Admit anchor `index` and preserve linked SUPERSEDE partner turns (no session cap on updates)."""
            sid = self.turns[index].session_id
            is_supersede_member = index in latest_for_older or index in earlier_for_newer
            if enforce_session_cap and not is_supersede_member and per_session[sid] >= per_session_cap:
                return False

            # If `index` is an earlier superseded record, ensure its latest current update is admitted FIRST
            # so a superseded value can never enter the prompt without its superseding update.
            if index in latest_for_older:
                admitted_any_latest = False
                for latest_idx in sorted(latest_for_older[index], key=lambda k: self.turns[k].timestamp):
                    if latest_idx in selected or add_turn(latest_idx, char_cap=turn_max_chars):
                        admitted_any_latest = True
                        if latest_idx not in anchors:
                            anchors.append(latest_idx)
                            per_session[self.turns[latest_idx].session_id] += 1
                if not admitted_any_latest:
                    return False

            if index not in selected:
                if not add_turn(index, char_cap=turn_max_chars):
                    return False
                anchors.append(index)
                per_session[sid] += 1

            # If `index` is a latest current update, preserve its linked earlier superseded partner turn
            # when budget permits so the reader sees the chronological [EARLIER] -> [LATEST] transition.
            if index in earlier_for_newer:
                for older_idx in sorted(earlier_for_newer[index], key=lambda k: self.turns[k].timestamp, reverse=True):
                    if older_idx not in selected and add_turn(older_idx, char_cap=turn_max_chars):
                        if older_idx not in anchors:
                            anchors.append(older_idx)
                            per_session[self.turns[older_idx].session_id] += 1
            return True

        # Optional closed-set speaker anchor reserve (strictly inside reranked shortlist)
        if named_subj is not None and speaker_anchor_reserve > 0:
            subj_cf = named_subj.casefold()
            reserved = 0
            for idx in reranked_shortlist:
                if self.turns[idx].speaker.strip().casefold() == subj_cf:
                    if admit_anchor_with_supersede_partners(idx):
                        reserved += 1
                        if reserved >= min(speaker_anchor_reserve, anchor_count):
                            break

        # Primary anchor budget (NO `max_sessions=6` cap — Guardrail K2.1!)
        if len(anchors) < anchor_count:
            for idx in candidate_order:
                if idx in selected:
                    continue
                if admit_anchor_with_supersede_partners(idx):
                    if len(anchors) >= anchor_count:
                        break

        # Local intra-session turn window around admitted anchors within remaining compact budget
        for distance in range(1, window_radius + 1):
            for anchor in list(anchors):
                parent = self._session_turns[self.turns[anchor].session_id]
                pos = self._positions[anchor]
                for neighbor in (pos - distance, pos + distance):
                    if 0 <= neighbor < len(parent):
                        add_turn(parent[neighbor], char_cap=neighbor_max_chars)

        # Always render turns (including SUPERSEDE pairs) strictly chronologically!
        ordered_indices = sorted(selected, key=lambda i: (self.turns[i].timestamp, i))

        header_line = ""
        if speaker_attribution_header and named_subj is not None and ordered_indices:
            sp_counts = Counter(
                self.turns[i].speaker.strip() for i in ordered_indices if self.turns[i].speaker.strip()
            )
            counts_str = ", ".join(f"{sp}={sp_counts.get(sp, 0)}" for sp in self._named_speakers)
            cast_str = ", ".join(self._named_speakers)
            header_line = (
                f"[Dialogue Cast: {cast_str} | Question Subject: {named_subj} | "
                f"Retrieved Turns by Speaker: {counts_str} — Each turn reflects its own speaker unless it explicitly names {named_subj}.]"
            )

        return ConversationRecall(
            ranked_turn_ids=tuple(self.turns[i].turn_id for i in anchors),
            turns=tuple(selected[i] for i in ordered_indices),
            echo_clusters=tuple(tuple(self.turns[i].turn_id for i in group) for group in routing.clusters),
            supersessions=tuple(
                (self.turns[newer].turn_id, self.turns[older].turn_id)
                for newer, older in routing.supersessions
            ),
            header_line=header_line,
            reranked_shortlist_ids=tuple(self.turns[i].turn_id for i in reranked_shortlist),
        )


def flat_bgem3_token_matched_context_v4(
    engine: ChelatedConversationalV4,
    question: str,
    *,
    context_chars: int,
    turn_max_chars: int | None = 1400,
) -> tuple[str, tuple[str, ...], set[str]]:
    """Token-matched Flat BGE-M3 + BM25 RRF control (no session pooling, no reranker, no EK5, no window)."""
    if not engine.turns:
        return "", (), set()
    q_vec = engine._encode([question])[0]
    turn_scores = engine._turn_bm25.score_query(question)
    flat_rrf = _rrf(turn_scores, engine._turn_vectors @ q_vec)
    order = [int(i) for i in np.argsort(-flat_rrf, kind="stable")]

    chosen: list[int] = []
    used = 0
    for idx in order:
        rendered = RetrievedTurn(engine.turns[idx], max_chars=turn_max_chars).render()
        cost = len(rendered) + bool(chosen)
        if used + cost > context_chars:
            if len(chosen) >= 3:
                break
            continue
        chosen.append(idx)
        used += cost

    chosen.sort(key=lambda i: (engine.turns[i].timestamp, i))
    ctx = "\n".join(RetrievedTurn(engine.turns[i], max_chars=turn_max_chars).render() for i in chosen)
    ids = tuple(engine.turns[i].turn_id for i in chosen)
    sids = {engine.turns[i].session_id for i in chosen}
    return ctx, ids, sids
