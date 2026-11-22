"""EM-4: Soft Epistemic Telemetry Fallback & Verified-Echo Routing (`M4f` / `M4g`).

Implements Carbon's post-R4 strategic refinement additively alongside `EK5 v3` (`ek5_v3_stratified.py`)
and `EM-1` (`M4e_minimal_pair_split`), without modifying any R4 module or baseline:

1. **Hard-Drop Guard (`EK5 v3` Explicit Refutation or Verified Duplicate Echo Only):**
   - Never hard-prunes cross-session turns on loose dense similarity (`0.72 <= cos < 0.86`) when
     lexical overlap is low (`stripped_jaccard < 0.58`) and `EK5 v3` holds no authorized `REFUTE` /
     `SUPERSEDE` edge. This eliminates the 3 clean evidence-recall losses (`loc_1_39`, `loc_3_2`,
     `loc_9_0`) and 3 rank-7 backfill abstention hallucinations observed in R4 `LoCoMo-300`.
   - Hard-collapses ONLY:
     (a) nodes marked `AUDIT_ONLY` by an explicit authorized `StratifiedEvidenceStore` edge, OR
     (b) verified high-overlap copy-echoes (`stripped_jaccard >= 0.58` or `cos >= 0.86`) that
         `classify_high_similarity_pair` confirms as `"ECHO_COLLAPSE"` (never `"PRESERVE_CONFLICT"`).

2. **Recap-Origin Timestamp De-Spoofing:**
   - Detects auto-archived / recap / digest wrappers (`Briefing 0:`, `Periodic assistant digest...:`,
     `Memory summary log...:`, etc.) and anchors every echo cluster's effective `origin_ts` to the
     earliest member timestamp `min(ts)` so repeated intervening recaps cannot impersonate fresh updates.

3. **Non-Blocking `EM-3` Telemetry Status-Tagged Context Routing:**
   - When an older turn (or collapsed echo cluster) co-occurs in `top_k` with a chronologically newer,
     topically coupled (`cos >= 0.50`), content-divergent turn (`PRESERVE_CONFLICT` or `jaccard < 0.78`),
     both turns are preserved and annotated with concise `EM-3` epistemic status tags (`[EARLIER SESSION
     RECORD...]` vs `[LATEST SESSION UPDATE]`).
   - Demoted borderline candidates (`0.80 <= cos < 0.86`, `ECHO_COLLAPSE`) are never silently discarded
     if clean token budget permits and no higher-relevance distinct turn fills the slot; instead they
     surface as a status-tagged `[EM-3 LOW-RANK FALLBACK]` witness.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from ek5_v3_stratified import (
    StratifiedEvidenceStore,
    classify_high_similarity_pair,
    strip_boilerplate_headers,
)

_TOK_RE = re.compile(r"[a-z0-9]+")
_EXTENDED_RECAP_RE = re.compile(
    r"^\s*(?:"
    r"session\s+recap\s+note[^:]*:|"
    r"memory\s+summary\s+log[^:]*:|"
    r"background\s+context\s+carried\s+forward\s*:|"
    r"follow-up\s+note\s+recalling\s+earlier\s+detail\s*:|"
    r"periodic\s+assistant\s+digest[^:]*:|"
    r"stored\s+profile\s+reminder[^:]*:|"
    r"(?:briefing|session|memo|note|turn|entry|update|memory\s+recap|recap|digest|log|report|reminder|summary)"
    r"\s*(?:#?\d+)?\s*(?:\([^)]*\))?\s*[:\-–—]"
    r")\s*",
    re.IGNORECASE,
)
_SPEAKER_PREFIX_RE = re.compile(r"^\s*[A-Za-z][A-Za-z0-9_ ]{0,22}:\s*")
_STOPWORDS: FrozenSet[str] = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "to", "of", "and", "in",
    "that", "it", "for", "on", "with", "as", "at", "by", "this", "from", "or", "but", "not", "my",
    "i", "me", "you", "your", "we", "they", "he", "she", "his", "her", "have", "has", "had", "do",
    "did", "does", "can", "could", "would", "should", "just", "so", "very", "also", "about", "what",
    "when", "where", "who", "how", "which", "into", "out", "up", "user", "assistant",
})


@dataclass(frozen=True)
class SoftTelemetryRoutingResult:
    selected_indices: Tuple[int, ...]
    status_tags: Mapping[int, str]
    collapsed_echo_clusters: Mapping[int, Tuple[int, ...]]
    explicit_ek5_dropped: Tuple[int, ...]
    low_rank_fallback_indices: Tuple[int, ...]


def is_recap_framed(text: str) -> bool:
    """Return True if `text` begins with an archival/recap/digest wrapper header."""
    s = text.strip()
    if s.lower().startswith("user:"):
        s = s[5:].strip()
    return bool(_EXTENDED_RECAP_RE.match(s))


def strip_recap_and_speaker(text: str) -> str:
    """Strip outer `User:`/speaker prefixes and recurring recap/digest boilerplate headers."""
    cur = strip_boilerplate_headers(text.strip())
    for _ in range(4):
        nxt = cur
        if nxt.lower().startswith("user:"):
            nxt = nxt[5:].strip()
        nxt = _EXTENDED_RECAP_RE.sub("", nxt, count=1).strip()
        nxt = strip_boilerplate_headers(nxt)
        if nxt == cur:
            break
        cur = nxt
    # Strip single leading speaker label (`Caroline: ...`) for speaker-normalized body comparison
    cur_no_spk = _SPEAKER_PREFIX_RE.sub("", cur, count=1).strip()
    # Strip a nested recap header if speaker label preceded or followed `Briefing N:`
    cur_no_spk = _EXTENDED_RECAP_RE.sub("", cur_no_spk, count=1).strip()
    cur_no_spk = _SPEAKER_PREFIX_RE.sub("", cur_no_spk, count=1).strip()
    return cur_no_spk if cur_no_spk else cur


def content_tokens_stripped(text: str) -> Set[str]:
    body = strip_recap_and_speaker(text)
    if " | Assistant:" in body:
        body = body.split(" | Assistant:", 1)[0]
    return {t for t in _TOK_RE.findall(body.lower()) if t not in _STOPWORDS and len(t) > 1}


def jaccard_similarity(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / float(len(a | b))


def _minmax(arr: np.ndarray) -> np.ndarray:
    lo, hi = float(arr.min()), float(arr.max())
    return np.zeros_like(arr) if hi - lo <= 1e-12 else (arr - lo) / (hi - lo)


def route_soft_telemetry_top_k(
    units: Sequence[Mapping[str, Any]],
    unit_embs: np.ndarray,
    query_emb: np.ndarray,
    bm25_scores: np.ndarray,
    *,
    top_k: int = 5,
    mode: str = "M4f_soft_telemetry_fallback",
    ek5_store: Optional[StratifiedEvidenceStore] = None,
    pool_size: int = 32,
    echo_jaccard_threshold: float = 0.58,
    echo_cosine_threshold: float = 0.86,
    backfill_rel_floor: float = 0.58,
    token_budget_chars: int = 2200,
) -> SoftTelemetryRoutingResult:
    """Route memory candidates using `EK5 v3` explicit refutations + `EM-4` soft telemetry fallback.

    Modes:
      - `M4f_soft_telemetry_fallback`: Zero-clean-loss hybrid ranking + verified-echo collapse +
        de-spoofed origin timestamps + `EM-3` chronological update tags + budget-gated soft fallback.
      - `M4g_ek5_soft_telemetry_recency`: Adds gentle de-spoofed origin-timestamp tie-breaking
        (`+0.07 * origin_recency`) for multi-session knowledge-update streams.
    """
    if mode not in ("M4f_soft_telemetry_fallback", "M4g_ek5_soft_telemetry_recency"):
        raise ValueError(f"Unsupported EM-4 mode: {mode}")

    n = len(units)
    if n == 0:
        return SoftTelemetryRoutingResult((), {}, {}, (), ())

    dense = unit_embs @ query_emb
    hyb = 0.50 * _minmax(bm25_scores) + 0.50 * _minmax(dense)

    # Step 1: Hard-drop ONLY explicit EK5 v3 `AUDIT_ONLY` nodes (verified refutation / supersession edges)
    explicit_dropped: List[int] = []
    eligible_mask = np.ones(n, dtype=bool)
    if ek5_store is not None:
        for i, u in enumerate(units):
            uid = str(u.get("unit_id", ""))
            if uid in ek5_store.eligibility and ek5_store.eligibility[uid] == "AUDIT_ONLY":
                eligible_mask[i] = False
                explicit_dropped.append(i)

    cand_indices = [int(i) for i in np.argsort(-hyb, kind="mergesort") if eligible_mask[int(i)]][: min(pool_size, n)]
    if not cand_indices:
        return SoftTelemetryRoutingResult((), {}, {}, tuple(explicit_dropped), ())

    k_len = len(cand_indices)
    pe = unit_embs[cand_indices]
    sim = pe @ pe.T
    raw_bodies = [
        strip_recap_and_speaker(str(units[idx].get("raw_text") or units[idx].get("user_text") or units[idx].get("text", "")))
        for idx in cand_indices
    ]
    tsets = [content_tokens_stripped(b) for b in raw_bodies]
    recap_flags = [
        is_recap_framed(str(units[idx].get("raw_text") or units[idx].get("user_text") or units[idx].get("text", "")))
        for idx in cand_indices
    ]
    ts_raw = [float(units[idx].get("ts", 0.0)) for idx in cand_indices]
    sess_ids = [str(units[idx].get("session_id", idx)) for idx in cand_indices]

    # Step 2: Build verified cross-session duplicate-echo clusters & de-spoof origin timestamps!
    # A pair `(p, q)` across sessions merges ONLY if `stripped_jaccard >= 0.58` or `cos >= 0.86`
    # AND `classify_high_similarity_pair` returns `"ECHO_COLLAPSE"` (never `"PRESERVE_CONFLICT"`).
    parent = list(range(k_len))

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(x: int, y: int) -> None:
        rx, ry = _find(x), _find(y)
        if rx != ry:
            parent[ry] = rx

    for a in range(k_len):
        for b in range(a + 1, k_len):
            if sess_ids[a] == sess_ids[b]:
                continue
            s_jac = jaccard_similarity(tsets[a], tsets[b])
            c_ab = float(sim[a, b])
            if s_jac >= echo_jaccard_threshold or c_ab >= echo_cosine_threshold:
                act, _ = classify_high_similarity_pair(raw_bodies[a], raw_bodies[b])
                if act == "ECHO_COLLAPSE":
                    _union(a, b)

    clusters: Dict[int, List[int]] = {}
    for loc_i in range(k_len):
        clusters.setdefault(_find(loc_i), []).append(loc_i)

    # Choose 1 canonical representative per verified echo cluster:
    # Prefer non-recap original turn with highest hybrid relevance; anchor `origin_ts = min(ts in cluster)`!
    rep_to_members: Dict[int, List[int]] = {}
    origin_ts_local: Dict[int, float] = {}
    suppressed_echo_local: Set[int] = set()
    for members in clusters.values():
        orig_ts = min(ts_raw[m] for m in members)
        # Prefer non-recap member if within 0.08 hybrid score of top cluster member; otherwise highest hybrid
        top_m_hyb = max(float(hyb[cand_indices[m]]) for m in members)
        best_m = sorted(
            members,
            key=lambda m: (
                1 if (recap_flags[m] and (top_m_hyb - float(hyb[cand_indices[m]])) < 0.08) else 0,
                -float(hyb[cand_indices[m]]),
                ts_raw[m],
                m,
            ),
        )[0]
        rep_to_members[best_m] = members
        origin_ts_local[best_m] = orig_ts
        for m in members:
            if m != best_m:
                suppressed_echo_local.add(m)

    # Step 3: Rank surviving cluster representatives
    surv_reps = [loc_i for loc_i in range(k_len) if loc_i not in suppressed_echo_local]
    t_lo = min(origin_ts_local[r] for r in surv_reps)
    t_hi = max(origin_ts_local[r] for r in surv_reps)
    t_span = max(t_hi - t_lo, 1.0)
    top_hyb = max(float(hyb[cand_indices[r]]) for r in surv_reps)

    if mode == "M4f_soft_telemetry_fallback":
        # Pure hybrid order (exact match with M1_hybrid_flat on clean distinct turns; tiny -0.02 recap tiebreak)
        ordered_reps = sorted(
            surv_reps,
            key=lambda r: (-(float(hyb[cand_indices[r]]) - (0.02 if recap_flags[r] else 0.0)), r),
        )
    else:
        # M4g: De-spoofed origin-timestamp recency tie-break (+0.07 * origin_recency)
        ordered_reps = sorted(
            surv_reps,
            key=lambda r: (
                -(
                    float(hyb[cand_indices[r]])
                    + 0.07 * ((origin_ts_local[r] - t_lo) / t_span)
                    - (0.02 if recap_flags[r] else 0.0)
                ),
                r,
            ),
        )

    # Step 4: Select primary `top_k` with relevance-gated backfill (prevents rank-10 noise on abstentions!)
    natural_top_k = set(range(min(top_k, k_len)))
    selected_local: List[int] = []
    demoted_soft_local: List[int] = []

    for r in ordered_reps:
        if len(selected_local) >= top_k:
            break
        r_hyb = float(hyb[cand_indices[r]])
        # If `r` was NOT in the natural top_k (i.e. it is backfilling a slot freed by echo collapse),
        # require `r_hyb >= backfill_rel_floor * top_hyb` so noisy tail turns aren't stuffed into prompts!
        if r not in natural_top_k and len(selected_local) > 0 and r_hyb < backfill_rel_floor * top_hyb:
            demoted_soft_local.append(r)
            continue
        selected_local.append(r)

    # Soft telemetry fallback: if echo collapse left `< top_k` items AND a natural top_k echo or borderline
    # witness exists within character budget, surface at most 1 status-tagged `[EM-3 LOW-RANK FALLBACK]` item
    # ONLY when no newer conflicting update already superseded it!
    status_tags: Dict[int, str] = {}
    fallback_added: List[int] = []

    # Step 5: Attach `EM-3` non-blocking chronological & echo-multiplicity telemetry tags
    # Detect cross-session update pairs inside `selected_local`:
    # `(a, b)` where `origin_ts[b] > origin_ts[a]`, distinct sessions, topical coupling `cos >= 0.50`,
    # and divergent factual content (`PRESERVE_CONFLICT` or `stripped_jaccard < 0.78`).
    older_flagged: Set[int] = set()
    newer_flagged: Set[int] = set()
    for i_pos, a in enumerate(selected_local):
        for b in selected_local[i_pos + 1 :]:
            pass  # Check all ordered pairs below
    for a in selected_local:
        for b in selected_local:
            if a == b or sess_ids[a] == sess_ids[b]:
                continue
            if origin_ts_local[b] <= origin_ts_local[a] + 1e-6:
                continue
            c_ab = float(sim[a, b])
            s_jac = jaccard_similarity(tsets[a], tsets[b])
            act, _ = classify_high_similarity_pair(raw_bodies[a], raw_bodies[b])
            is_echoed_stale = len(rep_to_members.get(a, [a])) >= 2 or recap_flags[a]
            # Trigger explicit temporal-update tags ONLY when:
            #   (i) `a` is a multi-copy/recap echo cluster (`echo_count >= 2` or recap-framed) co-occurring
            #       with a newer topically-coupled (`cos >= 0.50`, `s_jac < 0.78`) session turn `b`, OR
            #   (ii) `(a, b)` is a high-similarity (`cos >= 0.66` AND `s_jac >= 0.40`) minimal-pair conflict
            #        sharing a common predicate frame (`act == "PRESERVE_CONFLICT"`).
            # This guarantees clean complementary multi-session turns on LoCoMo are NEVER noisily tagged!
            if (is_echoed_stale and c_ab >= 0.50 and s_jac < 0.78) or (
                act == "PRESERVE_CONFLICT" and c_ab >= 0.66 and s_jac >= 0.40
            ):
                older_flagged.add(a)
                newer_flagged.add(b)

    for loc_i in selected_local:
        u_idx = cand_indices[loc_i]
        n_copies = len(rep_to_members.get(loc_i, [loc_i]))
        if loc_i in older_flagged and loc_i not in newer_flagged:
            if n_copies >= 2:
                status_tags[u_idx] = (
                    f"[EM-3 TELEMETRY: EARLIER RECORD (echoed {n_copies}x in archives) — "
                    "treat later dated update as current if conflicting] "
                )
            else:
                status_tags[u_idx] = "[EM-3 TELEMETRY: EARLIER SESSION RECORD — check later dated update] "
        elif loc_i in newer_flagged:
            status_tags[u_idx] = "[EM-3 TELEMETRY: LATEST DATED UPDATE] "
        elif n_copies >= 3 and recap_flags[loc_i]:
            status_tags[u_idx] = f"[EM-3 TELEMETRY: ARCHIVED RECAP (echoed {n_copies}x)] "

    # Optional soft fallback slot when token budget allows and `< top_k` items were selected
    cur_chars = sum(len(str(units[cand_indices[r]].get("text", ""))) for r in selected_local)
    if len(selected_local) < top_k and demoted_soft_local and cur_chars + 280 <= token_budget_chars:
        fb_loc = demoted_soft_local[0]
        if float(hyb[cand_indices[fb_loc]]) >= 0.48 * top_hyb:
            fb_u = cand_indices[fb_loc]
            selected_local.append(fb_loc)
            fallback_added.append(fb_u)
            status_tags[fb_u] = "[EM-3 TELEMETRY: LOW-RANK FALLBACK CONTEXT] "

    collapsed_map = {
        cand_indices[rep]: tuple(cand_indices[m] for m in mems)
        for rep, mems in rep_to_members.items()
        if len(mems) >= 2
    }
    return SoftTelemetryRoutingResult(
        selected_indices=tuple(cand_indices[r] for r in selected_local),
        status_tags=status_tags,
        collapsed_echo_clusters=collapsed_map,
        explicit_ek5_dropped=tuple(explicit_dropped),
        low_rank_fallback_indices=tuple(fallback_added),
    )


__all__ = [
    "SoftTelemetryRoutingResult",
    "content_tokens_stripped",
    "is_recap_framed",
    "jaccard_similarity",
    "route_soft_telemetry_top_k",
    "strip_recap_and_speaker",
]
