"""Dated, dual-granularity conversation retrieval with conservative EK5 routing."""

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
)
from scripts.night1_common import FastBM25Index

BGE_MODEL = "BAAI/bge-m3"
BGE_DIMENSIONS = 1024
EARLIER = "[EARLIER SUPERSEDED RECORD]"
LATEST = "[LATEST CURRENT UPDATE]"
_CHANGE = re.compile(
    r"\b(?:no longer|used to|instead of|changed|switched|replaced|updated|moved|"
    r"now|these days|recently|currently|this time)\b",
    re.IGNORECASE,
)


class Embedder(Protocol):
    def encode(self, texts: Sequence[str]) -> np.ndarray: ...


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

    def render(self) -> str:
        status = f"{self.status} " if self.status else ""
        return f"- {self.turn.date_anchor} {status}{self.turn.text}"


@dataclass(frozen=True)
class ConversationRecall:
    ranked_turn_ids: tuple[str, ...]
    turns: tuple[RetrievedTurn, ...]
    echo_clusters: tuple[tuple[str, ...], ...] = ()
    supersessions: tuple[tuple[str, str], ...] = ()
    header_line: str = ""

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
class _Gating:
    order: tuple[int, ...]
    suppressed: frozenset[int] = frozenset()
    tags: tuple[tuple[int, str], ...] = ()
    clusters: tuple[tuple[int, ...], ...] = ()
    supersessions: tuple[tuple[int, int], ...] = ()


def _normalized(vectors: np.ndarray) -> np.ndarray:
    return vectors / np.maximum(np.linalg.norm(vectors, axis=-1, keepdims=True), 1e-12)


def _rrf(*scores: np.ndarray) -> np.ndarray:
    # Reciprocal Rank Fusion, Cormack, Clarke & Büttcher (SIGIR 2009).
    combined = np.zeros(len(scores[0]), dtype=np.float64)
    for score in scores:
        order = np.argsort(-score, kind="stable")
        # Equal scores receive equal rank, including an all-zero BM25 result.
        ranked = score[order]
        tied_ranks = np.maximum.accumulate(
            np.where(np.r_[True, ranked[1:] != ranked[:-1]], np.arange(len(order)), 0)
        )
        combined[order] += 1.0 / (60.0 + tied_ranks + 1)
    return combined


class ChelatedConversationalV3:
    """Build independent turn and session indexes, then optionally route through EK5.

    The session dense vector pools every normalized, dated turn vector in its
    parent session. The session lexical index sees the entire parent transcript.
    """

    def __init__(self, turns: Sequence[ConversationTurn], embedder: Embedder | None = None):
        self.turns = tuple(turns)
        turn_ids = [turn.turn_id for turn in self.turns]
        if len(set(turn_ids)) != len(turn_ids):
            raise ValueError("Conversation turn IDs must be unique")
        if any(not t.session_id or not t.date or not math.isfinite(t.timestamp) for t in self.turns):
            raise ValueError("Each turn needs a session ID, date, and finite timestamp")
        if embedder is None:
            from sentence_transformers import SentenceTransformer

            model = SentenceTransformer(BGE_MODEL)

            class BGEEmbedder:
                def encode(self, texts: Sequence[str]) -> np.ndarray:
                    return model.encode(
                        list(texts), normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
                    )

            embedder = BGEEmbedder()
        self._embedder = embedder
        self._session_turns: dict[str, list[int]] = defaultdict(list)
        for i, turn in enumerate(self.turns):
            self._session_turns[turn.session_id].append(i)
        self._positions = {}
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
            self._session_vectors = _normalized(np.stack([
                self._turn_vectors[indices].mean(axis=0) for indices in self._session_turns.values()
            ]))
        self._claim_tokens = [content_tokens_stripped(turn.claim_text) for turn in self.turns]
        self._named_speakers = tuple(sorted({
            turn.speaker.strip()
            for turn in self.turns
            if turn.speaker.strip() and turn.speaker.strip().casefold() not in {"user", "assistant"}
        }))

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
        """Return the unmodified hybrid ordering used by both benchmark arms."""
        order, _, _ = self.rank_with_scores(question)
        return order

    def _gating(self, order: tuple[int, ...], pool_size: int) -> _Gating:
        pool = order[:pool_size]
        vectors = self._turn_vectors[list(pool)]
        similarities = vectors @ vectors.T
        echoes: set[frozenset[int]] = set()
        conflicts: list[tuple[int, int]] = []
        for p, i in enumerate(pool):
            a = self.turns[i]
            for q in range(p + 1, len(pool)):
                j = pool[q]
                b = self.turns[j]
                if a.session_id == b.session_id or a.speaker.casefold() != b.speaker.casefold():
                    continue
                ta, tb = self._claim_tokens[i], self._claim_tokens[j]
                cosine = float(similarities[p, q])
                overlap = jaccard_similarity(ta, tb)
                if cosine < 0.80 and overlap < 0.55:
                    continue
                if len(ta & tb) < 2:
                    continue
                action, reason = classify_high_similarity_pair(
                    _EXTENDED_RECAP_RE.sub("", a.claim_text),
                    _EXTENDED_RECAP_RE.sub("", b.claim_text),
                )
                if action == "ECHO_COLLAPSE":
                    if overlap >= 0.30:
                        echoes.add(frozenset((i, j)))
                    continue

                a_recap = is_recap_framed(a.claim_text)
                b_recap = is_recap_framed(b.claim_text)
                if a_recap != b_recap:
                    # Archival recap restatements anchor to their origin and cannot supersede genuine turns
                    earlier_i, later_i = (i, j) if a_recap else (j, i)
                else:
                    earlier_i, later_i = sorted((i, j), key=lambda k: self.turns[k].timestamp)
                earlier, later = self.turns[earlier_i], self.turns[later_i]
                if not (a_recap != b_recap) and (earlier.date == later.date or earlier.timestamp >= later.timestamp):
                    continue
                # Distinct entities and topics can coexist. Chronology carries
                # authority only when wording or a tight substitution supports it.
                tightly_coupled = overlap >= 0.62 and reason in {
                    "CONFLICT_NUMERIC_OR_CODE", "CONFLICT_MINIMAL_PAIR_SUBSTITUTION",
                }
                explicitly_changed = overlap >= 0.42 and bool(_CHANGE.search(later.claim_text))
                if tightly_coupled or explicitly_changed:
                    conflicts.append((later_i, earlier_i))

        if not echoes and not conflicts:
            return _Gating(order)

        rank_index = {idx: p for p, idx in enumerate(pool)}
        groups: list[list[int]] = []
        for i in pool:
            for group in groups:
                # Complete linkage prevents a similar intermediary bridging
                # mutually different values in the two end records.
                if all(frozenset((i, j)) in echoes for j in group):
                    group.append(i)
                    break
            else:
                groups.append([i])

        clusters = tuple(
            tuple(sorted(group, key=lambda i: (self.turns[i].timestamp, rank_index[i])))
            for group in groups if len(group) > 1
        )
        representative = {i: group[0] for group in clusters for i in group}
        suppressed = frozenset(i for i, rep in representative.items() if i != rep)
        candidate_edges = set()
        for newer, older in conflicts:
            newer, older = representative.get(newer, newer), representative.get(older, older)
            if self.turns[newer].timestamp > self.turns[older].timestamp:
                candidate_edges.add((newer, older))
        # Replacements occupy the first rank at which their cluster appeared.
        routed = tuple(dict.fromkeys(representative.get(i, i) for i in order))
        if not candidate_edges:
            return _Gating(routed, suppressed, clusters=clusters)

        store = StratifiedEvidenceStore()
        indices = sorted({i for pair in candidate_edges for i in pair})
        store.apply_mutation(
            [
                StratifiedEvent(
                    event_id=self.turns[i].turn_id,
                    text=self.turns[i].claim_text,
                    valid_time=round(self.turns[i].timestamp * 1000),
                ) for i in indices
            ],
            [
                (self.turns[newer].turn_id, self.turns[older].turn_id, "SUPERSEDE")
                for newer, older in sorted(candidate_edges)
            ],
        )
        edges = tuple(
            (newer, older) for newer, older in sorted(candidate_edges)
            if self.turns[older].turn_id in store.sup_out.get(self.turns[newer].turn_id, ())
        )
        tags = tuple(
            (i, EARLIER if store.eligibility[self.turns[i].turn_id] == "AUDIT_ONLY" else LATEST)
            for i in sorted({i for edge in edges for i in edge})
        )
        return _Gating(routed, suppressed, tags, clusters, edges)

    def _unique_named_speaker(self, question: str) -> str | None:
        if len(self._named_speakers) < 2:
            return None
        q_lo = question.casefold()
        hits = [
            sp for sp in self._named_speakers
            if re.search(rf"\b{re.escape(sp.casefold())}(?:'s)?\b", q_lo)
        ]
        return hits[0] if len(hits) == 1 else None

    def _adaptive_allowed_sessions(
        self,
        session_rrf: np.ndarray,
        gap_threshold: float,
        min_sessions: int,
        max_sessions: int | None,
    ) -> set[str] | None:
        if gap_threshold <= 0.0 or len(session_rrf) <= 1:
            return None
        s_order = np.argsort(-session_rrf, kind="stable")
        s_sorted = session_rrf[s_order]
        s1 = max(float(s_sorted[0]), 1e-12)
        cap = len(s_order) if max_sessions is None else min(len(s_order), max(min_sessions, max_sessions))
        keep = cap
        for m in range(max(1, min_sessions), cap):
            rel_drop = float(s_sorted[m - 1] - s_sorted[m]) / s1
            if rel_drop >= gap_threshold:
                keep = m
                break
        return {self._session_ids[int(idx)] for idx in s_order[:keep]}

    def retrieve(
        self,
        question: str,
        *,
        epistemic: bool = True,
        anchor_count: int = 12,
        window_radius: int = 2,
        context_chars: int = 18000,
        pool_size: int = 36,
        ranked: tuple[int, ...] | None = None,
        session_rrf: np.ndarray | None = None,
        speaker_attribution_header: bool = False,
        speaker_anchor_reserve: int = 0,
        adaptive_rrf_gap: float = 0.0,
        min_sessions: int = 2,
        max_sessions: int | None = None,
        session_order: str = "chronological",
    ) -> ConversationRecall:
        """Expand ranked anchors in their own sessions within a rendered character budget."""
        if anchor_count < 1 or pool_size < anchor_count or window_radius < 0 or context_chars < 1:
            raise ValueError("Invalid anchors, context budget, window radius, or gate pool")
        if ranked is None:
            order, _, calc_sess_rrf = self.rank_with_scores(question)
            if session_rrf is None:
                session_rrf = calc_sess_rrf
        else:
            order = ranked
            if adaptive_rrf_gap > 0.0 and session_rrf is None and self.turns:
                _, _, session_rrf = self.rank_with_scores(question)
        if not order:
            return ConversationRecall((), ())

        routing = self._gating(order, pool_size) if epistemic else _Gating(order)
        tags = dict(routing.tags)
        selected: dict[int, RetrievedTurn] = {}
        per_session: Counter[str] = Counter()
        anchors: list[int] = []

        named_subj = self._unique_named_speaker(question) if (speaker_attribution_header or speaker_anchor_reserve > 0) else None
        allowed_sessions = (
            self._adaptive_allowed_sessions(session_rrf, adaptive_rrf_gap, min_sessions, max_sessions)
            if (adaptive_rrf_gap > 0.0 and session_rrf is not None)
            else None
        )

        # Reserve space if a neutral closed-set speaker header will be rendered
        header_reserve = 220 if (speaker_attribution_header and named_subj is not None) else 0
        eff_budget = max(1, context_chars - header_reserve)
        used = 0

        def add(index: int) -> bool:
            nonlocal used
            if index in selected or index in routing.suppressed:
                return False
            turn = RetrievedTurn(self.turns[index], tags.get(index, ""))
            cost = len(turn.render()) + bool(selected)
            if used + cost > eff_budget:
                return False
            selected[index] = turn
            used += cost
            return True

        # Optional closed-set speaker anchor reserve (up to `speaker_anchor_reserve` top turns by the named speaker)
        if named_subj is not None and speaker_anchor_reserve > 0:
            subj_cf = named_subj.casefold()
            reserved = 0
            for i in routing.order[:pool_size]:
                sid = self.turns[i].session_id
                if allowed_sessions is not None and sid not in allowed_sessions:
                    continue
                if self.turns[i].speaker.strip().casefold() == subj_cf and per_session[sid] < 3 and add(i):
                    anchors.append(i)
                    per_session[sid] += 1
                    reserved += 1
                    if reserved >= min(speaker_anchor_reserve, anchor_count):
                        break

        # Reserve room for independently ranked evidence before adding neighbors.
        if len(anchors) < anchor_count:
            for i in routing.order:
                sid = self.turns[i].session_id
                if allowed_sessions is not None and sid not in allowed_sessions:
                    continue
                if per_session[sid] < 3 and add(i):
                    anchors.append(i)
                    per_session[sid] += 1
                    if len(anchors) == anchor_count:
                        break
        # Fallback if adaptive session filter yielded fewer than min_sessions anchors
        if allowed_sessions is not None and len(anchors) < min(anchor_count, 3):
            for i in routing.order:
                sid = self.turns[i].session_id
                if per_session[sid] < 3 and add(i):
                    anchors.append(i)
                    per_session[sid] += 1
                    if len(anchors) == anchor_count:
                        break

        for distance in range(1, window_radius + 1):
            for anchor in anchors:
                parent = self._session_turns[self.turns[anchor].session_id]
                pos = self._positions[anchor]
                for neighbor in (pos - distance, pos + distance):
                    if 0 <= neighbor < len(parent):
                        add(parent[neighbor])

        if session_order == "relevance":
            rank_pos = {idx: rk for rk, idx in enumerate(routing.order)}
            best_sess_rank: dict[str, int] = {}
            for idx in selected:
                sid = self.turns[idx].session_id
                rk = rank_pos.get(idx, len(routing.order))
                if sid not in best_sess_rank or rk < best_sess_rank[sid]:
                    best_sess_rank[sid] = rk
            ordered_indices = sorted(
                selected,
                key=lambda i: (best_sess_rank[self.turns[i].session_id], self.turns[i].timestamp, i),
            )
        else:
            ordered_indices = sorted(selected, key=lambda i: (self.turns[i].timestamp, i))

        header_line = ""
        if speaker_attribution_header and named_subj is not None and ordered_indices:
            sp_counts = Counter(self.turns[i].speaker.strip() for i in ordered_indices if self.turns[i].speaker.strip())
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
        )
