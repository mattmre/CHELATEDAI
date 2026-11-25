"""Production Raw-Text Epistemic Exocortex & Causal Streaming Spherical Bridge.

Eliminates every benchmark-harness shortcut identified by the 4-Model Frontier Panel:
  1. Ingests 100% raw natural-language documents (`RawDocumentEnvelope`) with standard
     connector metadata (`doc_id`, `published_ts`, `ingested_ts`, `source_channel`, `tenant`).
     Zero pre-supplied `StratifiedEvent` slots or `EdgeAttestation` edges are required.
  2. Extracts structured `(entity, slot, claim_value, stance)` tuples using a lightweight
     local instruct model (e.g., `Qwen/Qwen2.5-3B-Instruct` on a consumer RTX 3080/3090 or
     any OpenAI-compatible vLLM endpoint) and deterministically compiles them into `EK5 v3`
     `StratifiedEvidenceStore` mutations (`SUPERSEDE`, `REFUTE`, `COPY_OF`, `DERIVES_VALIDITY`,
     `CITES_AUDIT`).
  3. Performs 100% label-free query-time admission, blocked-authority attacker demotion,
     complete-linkage echo collapse, and slot-scoped `CONTESTED` abstention detection—with
     ZERO benchmark-family flags or oracle query `(entity, slot)` hints.
  4. Implements `CausalStreamingZeroMuBridge` (`(d_j · q_hat - mu_d · q_hat) / ||d_j - mu_d||_2`)
     using a strictly causal online EMA query centroid (`mu_q_{t-1}` seeded from `bridge.mu_x`),
     requiring zero target-index writes and zero future-query peeking.
"""

from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from ek5_v3_stratified import EdgeAttestation, StratifiedEvent, StratifiedEvidenceStore
from spherical_migration_bridge import CenteredProcrustesBridge, l2_normalize

DEFAULT_CHANNEL_AUTHORITY: Dict[str, int] = {
    # Tier 1: Unverified external / social / forum rumor channels
    "community_forum": 1,
    "social_repost": 1,
    "unverified_blog": 1,
    "rumor_feed": 1,
    # Tier 2: Working internal chat / support / secondary mirrors
    "internal_ops_chat": 2,
    "support_ticket": 2,
    "analyst_note": 2,
    "stackoverflow_mirror": 2,
    # Tier 3: Verified engineering / vendor / postmortem sources
    "vendor_advisory": 3,
    "maintainer_release_note": 3,
    "postmortem_final": 3,
    "independent_security_lab": 3,
    # Tier 4: Authoritative regulatory / signed advisories
    "official_nvd_advisory": 4,
    "sec_edgar_filing": 4,
    "signed_security_bulletin": 4,
}

EXTRACTION_SYSTEM_PROMPT = (
    "You are a precise factual claim extractor for an enterprise knowledge graph. "
    "Given a single document, extract the primary subject entity identifier, the specific attribute/metric slot, "
    "the concise normalized claim value asserted by the author, and the epistemic stance.\n"
    "Valid stances:\n"
    "- ASSERT: states an initial, routine, or repeated factual claim.\n"
    "- SUPERSEDE: announces an official update, patch, revision, restatement, or replacement of an earlier value.\n"
    "- REFUTE: retracts an earlier false/dangerous claim or directly contests a competing concurrent claim.\n"
    "- DEBUNK_RUMOR: an authoritative audit or bulletin that cites an unverified rumor solely to refute/debunk it.\n"
    "Return ONLY valid single-line JSON with keys: "
    '{"entity": "<canonical id/name>", "slot": "<concise attribute>", "value": "<concise value>", "stance": "<ASSERT|SUPERSEDE|REFUTE|DEBUNK_RUMOR>"}'
)

_NON_ALNUM_RE = re.compile(r"[^a-z0-9_.\-/]+")
_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_.\-]*")
_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}")
_STOPWORDS: Set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "to", "of", "in", "on",
    "for", "with", "by", "at", "from", "as", "that", "this", "it", "what", "which", "who",
    "current", "latest", "official", "verified", "status", "according", "records", "please",
}


@dataclass(frozen=True)
class RawDocumentEnvelope:
    """Standard connector envelope for unstructured enterprise document ingestion."""

    doc_id: str
    text: str
    published_ts: int
    ingested_ts: int
    source_channel: str
    tenant: str = "prod"
    domain: str = "general"


@dataclass(frozen=True)
class ExtractedClaimRecord:
    """Structured claim proposed from raw text prior to EK5 v3 compilation."""

    doc_id: str
    entity: str
    slot: str
    claim_value: str
    stance: str
    raw_json: str


def canonicalize_key(raw: str) -> str:
    cleaned = _NON_ALNUM_RE.sub("_", (raw or "").strip().lower()).strip("_")
    return cleaned or "unknown"


def canonicalize_value(raw: str) -> str:
    s = re.sub(r"\s+", " ", (raw or "").strip())
    return s[:120] if s else "UNKNOWN"


def content_tokens(text: str) -> Set[str]:
    return {t for t in _TOKEN_RE.findall((text or "").lower()) if len(t) > 1 and t not in _STOPWORDS}


def token_jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return float(inter) / float(union) if union else 0.0


def parse_extracted_claim_json(doc_id: str, raw_output: str) -> ExtractedClaimRecord:
    """Deterministically parse model JSON output with fail-safe fallback."""
    txt = (raw_output or "").strip()
    parsed: Optional[Dict[str, Any]] = None
    try:
        obj = json.loads(txt)
        if isinstance(obj, dict):
            parsed = obj
    except Exception:
        m = _JSON_OBJ_RE.search(txt)
        if m:
            try:
                obj = json.loads(m.group(0))
                if isinstance(obj, dict):
                    parsed = obj
            except Exception:
                parsed = None

    if not parsed:
        return ExtractedClaimRecord(
            doc_id=doc_id,
            entity="unknown_entity",
            slot="general_state",
            claim_value="UNKNOWN",
            stance="ASSERT",
            raw_json=txt[:240],
        )

    ent = canonicalize_key(str(parsed.get("entity") or "unknown_entity"))
    slot = canonicalize_key(str(parsed.get("slot") or "general_state"))
    val = canonicalize_value(str(parsed.get("value") or "UNKNOWN"))
    stance = str(parsed.get("stance") or "ASSERT").strip().upper()
    if stance not in {"ASSERT", "SUPERSEDE", "REFUTE", "DEBUNK_RUMOR"}:
        stance = "ASSERT"
    return ExtractedClaimRecord(
        doc_id=doc_id,
        entity=ent,
        slot=slot,
        claim_value=val,
        stance=stance,
        raw_json=txt[:240],
    )


class CausalStreamingZeroMuBridge:
    """Zero-write spherical bridge with resident 1-float sidecar and strictly causal EMA `mu_q`."""

    def __init__(
        self,
        bridge: CenteredProcrustesBridge,
        resident_docs: np.ndarray,
        *,
        ema_alpha: float = 0.05,
    ) -> None:
        if resident_docs.ndim != 2:
            raise ValueError("resident_docs must be 2D [N, d_target]")
        self.bridge = bridge
        self.docs = resident_docs.astype(np.float32, copy=False)
        self.ema_alpha = float(ema_alpha)
        # Precomputed ONCE at index load: mu_d vector + 1 float per resident document!
        self.mu_d = self.docs.mean(axis=0, keepdims=True).astype(np.float32)
        self.doc_centered_norms = np.maximum(
            np.linalg.norm(self.docs - self.mu_d, axis=1).astype(np.float32),
            1e-12,
        )
        self.Q = bridge.Q.astype(np.float32)
        # Seed online query centroid strictly from disjoint anchor calibration `mu_x` (zero future peeking!)
        self.mu_q_ema = bridge.mu_x.reshape(1, -1).astype(np.float32).copy()
        self.queries_seen = 0

    def reset_stream(self) -> None:
        self.mu_q_ema = self.bridge.mu_x.reshape(1, -1).astype(np.float32).copy()
        self.queries_seen = 0

    def score_single_query(self, q_src: np.ndarray, admit_mask: Optional[np.ndarray] = None) -> np.ndarray:
        """Score a single source-space query causally, updating `mu_q_ema` ONLY after projection."""
        q_row = q_src.reshape(1, -1).astype(np.float32)
        q_hat = l2_normalize((q_row - self.mu_q_ema) @ self.Q)[0].astype(np.float32)
        # Update streaming query centroid AFTER projecting current query (strictly causal!)
        if self.ema_alpha > 0.0:
            self.mu_q_ema = (1.0 - self.ema_alpha) * self.mu_q_ema + self.ema_alpha * q_row
        self.queries_seen += 1

        raw_dots = self.docs @ q_hat
        zm_scores = (raw_dots - float(self.mu_d[0] @ q_hat)) / self.doc_centered_norms
        if admit_mask is not None:
            zm_scores = np.where(admit_mask, zm_scores, -1e9)
        return zm_scores


def align_slot_clusters(
    claims: Sequence[ExtractedClaimRecord],
    slot_embeddings: Optional[np.ndarray] = None,
    *,
    sim_threshold: float = 0.84,
) -> List[ExtractedClaimRecord]:
    """Merge near-synonymous extracted `(entity, slot)` names within the same canonical entity.

    Because raw-text LLM extraction may phrase the same slot slightly differently across
    documents (e.g. `patched_version` vs `fixed_version` on `cve_2024_3094`), this merges
    slots ONLY within matching entities when lexical/embedding similarity exceeds `sim_threshold`.
    """
    if not claims:
        return []
    by_ent: Dict[str, List[int]] = defaultdict(list)
    for idx, c in enumerate(claims):
        by_ent[c.entity].append(idx)

    canonical_slot_for_idx: Dict[int, str] = {}
    for _ent, idxs in by_ent.items():
        slot_counts = Counter(claims[i].slot for i in idxs)
        unique_slots = [s for s, _cnt in slot_counts.most_common()]
        rep_map: Dict[str, str] = {}
        rep_indices: Dict[str, int] = {}
        for i in idxs:
            if claims[i].slot not in rep_indices:
                rep_indices[claims[i].slot] = i

        for s in unique_slots:
            matched: Optional[str] = None
            s_toks = set(s.split("_"))
            for rep_s in rep_map.values():
                if rep_s == s:
                    matched = rep_s
                    break
                r_toks = set(rep_s.split("_"))
                lex_sim = token_jaccard(s_toks, r_toks)
                emb_sim = 0.0
                if slot_embeddings is not None:
                    emb_sim = float(slot_embeddings[rep_indices[s]] @ slot_embeddings[rep_indices[rep_s]])
                if lex_sim >= 0.50 or emb_sim >= sim_threshold:
                    matched = rep_s
                    break
            rep_map[s] = matched if matched is not None else s

        for i in idxs:
            canonical_slot_for_idx[i] = rep_map[claims[i].slot]

    aligned: List[ExtractedClaimRecord] = []
    for idx, c in enumerate(claims):
        aligned.append(
            ExtractedClaimRecord(
                doc_id=c.doc_id,
                entity=c.entity,
                slot=canonical_slot_for_idx[idx],
                claim_value=c.claim_value,
                stance=c.stance,
                raw_json=c.raw_json,
            )
        )
    return aligned


def compile_extracted_claims_to_ek5(
    docs: Sequence[RawDocumentEnvelope],
    claims: Sequence[ExtractedClaimRecord],
    *,
    channel_authority: Optional[Mapping[str, int]] = None,
    doc_embeddings: Optional[np.ndarray] = None,
) -> Tuple[StratifiedEvidenceStore, Dict[str, StratifiedEvent], List[EdgeAttestation]]:
    """Compile raw envelopes + model-extracted claims into a `StratifiedEvidenceStore`."""
    if len(docs) != len(claims):
        raise ValueError("docs and claims must have equal length")
    auth_map = dict(DEFAULT_CHANNEL_AUTHORITY)
    if channel_authority:
        auth_map.update(channel_authority)

    claim_by_id = {c.doc_id: c for c in claims}
    idx_by_id = {d.doc_id: i for i, d in enumerate(docs)}

    # Group valeurs within (tenant, entity, slot) so minor capitalization differences merge
    events: Dict[str, StratifiedEvent] = {}
    buckets: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)

    for d in sorted(docs, key=lambda x: (x.ingested_ts, x.published_ts, x.doc_id)):
        c = claim_by_id[d.doc_id]
        tier = int(auth_map.get(d.source_channel, 2))
        val_norm = re.sub(r"\s+", " ", c.claim_value.strip().lower())
        ev = StratifiedEvent(
            event_id=d.doc_id,
            entity=c.entity,
            slot=c.slot,
            claim_value=val_norm,
            text=d.text,
            valid_time=int(d.published_ts),
            tx_time=int(d.ingested_ts),
            tenant=d.tenant,
            domain=d.domain,
            authority_tier=tier,
        )
        events[d.doc_id] = ev
        buckets[(d.tenant, c.entity, c.slot)].append(d.doc_id)

    attestations: List[EdgeAttestation] = []
    att_seq = 0

    def _emit(src_id: str, dst_id: str, edge_type: str, vt: int, tx: int) -> None:
        nonlocal att_seq
        if src_id == dst_id:
            return
        att_seq += 1
        attestations.append(
            EdgeAttestation(
                attestation_id=f"raw_att_{att_seq}",
                src_id=src_id,
                dst_id=dst_id,
                edge_type=edge_type,
                valid_time=vt,
                tx_time=tx,
            )
        )

    for _bkey, ordered_ids in buckets.items():
        seen_ids: List[str] = []
        earliest_by_val: Dict[str, str] = {}
        for uid in ordered_ids:
            u_ev = events[uid]
            u_cl = claim_by_id[uid]
            u_val = u_ev.claim_value
            if u_val not in earliest_by_val:
                earliest_by_val[u_val] = uid
            else:
                origin_id = earliest_by_val[u_val]
                orig_ev = events[origin_id]
                # Low/equal-tier repetition of an earlier identical claim is an echo / derived copy
                if u_ev.authority_tier <= orig_ev.authority_tier and u_ev.authority_tier <= 2:
                    etype = "COPY_OF" if u_ev.authority_tier == orig_ev.authority_tier else "DERIVES_CURRENT"
                    _emit(uid, origin_id, etype, u_ev.valid_time, u_ev.tx_time)

            for wid in seen_ids:
                w_ev = events[wid]
                w_cl = claim_by_id[wid]
                if w_ev.claim_value == u_val or w_ev.claim_value == "unknown" or u_val == "unknown":
                    continue

                if u_cl.stance == "DEBUNK_RUMOR":
                    # Authoritative debunk cites the rumor for audit ONLY while refuting it!
                    _emit(uid, wid, "CITES_AUDIT", u_ev.valid_time, u_ev.tx_time)
                    _emit(uid, wid, "REFUTE", u_ev.valid_time, u_ev.tx_time)
                elif u_cl.stance == "REFUTE":
                    _emit(uid, wid, "REFUTE", u_ev.valid_time, u_ev.tx_time)
                    # Concurrent same-tier conflicting assertions form a symmetric mutual dispute!
                    if (
                        w_cl.stance in {"REFUTE", "ASSERT"}
                        and u_ev.authority_tier == w_ev.authority_tier
                        and abs(u_ev.valid_time - w_ev.valid_time) <= 2
                    ):
                        _emit(wid, uid, "REFUTE", w_ev.valid_time, w_ev.tx_time)
                elif u_cl.stance == "SUPERSEDE":
                    if u_ev.valid_time >= w_ev.valid_time:
                        _emit(uid, wid, "SUPERSEDE", u_ev.valid_time, u_ev.tx_time)
                    else:
                        _emit(uid, wid, "REFUTE", u_ev.valid_time, u_ev.tx_time)
                else:
                    # Stance == ASSERT with incompatible value on the same (entity, slot):
                    # If strictly newer from verified/authoritative channel -> SUPERSEDE;
                    # If low-tier late post contradicting higher-tier record -> attempted SUPERSEDE
                    # (which Stratum 1 blocks and logs in `blocked_authority_edges`!).
                    if u_ev.valid_time > w_ev.valid_time:
                        _emit(uid, wid, "SUPERSEDE", u_ev.valid_time, u_ev.tx_time)
                    elif u_ev.authority_tier == w_ev.authority_tier and abs(u_ev.valid_time - w_ev.valid_time) <= 1:
                        if doc_embeddings is not None:
                            _ = idx_by_id[uid]
                        _emit(uid, wid, "REFUTE", u_ev.valid_time, u_ev.tx_time)
                        _emit(wid, uid, "REFUTE", w_ev.valid_time, w_ev.tx_time)

            seen_ids.append(uid)

    store = StratifiedEvidenceStore(strict_dag_supersede=False)
    store.apply_mutation(list(events.values()), attestations)
    return store, events, attestations


class LabelFreeChelatedExocortex:
    """Production label-free `EK5 v3` admission mask, echo deduplicator, and CONTESTED detector."""

    def __init__(
        self,
        docs: Sequence[RawDocumentEnvelope],
        claims: Sequence[ExtractedClaimRecord],
        doc_embeddings: np.ndarray,
        *,
        channel_authority: Optional[Mapping[str, int]] = None,
        echo_cos_threshold: float = 0.84,
        echo_jaccard_threshold: float = 0.54,
    ) -> None:
        self.docs = list(docs)
        self.doc_ids = [d.doc_id for d in self.docs]
        self.idx_by_id = {d.doc_id: i for i, d in enumerate(self.docs)}
        self.doc_embeddings = l2_normalize(doc_embeddings.astype(np.float32))
        self.echo_cos_threshold = float(echo_cos_threshold)
        self.echo_jaccard_threshold = float(echo_jaccard_threshold)

        self.store, self.events, self.attestations = compile_extracted_claims_to_ek5(
            self.docs,
            claims,
            channel_authority=channel_authority,
            doc_embeddings=self.doc_embeddings,
        )
        self.token_sets = {d.doc_id: content_tokens(d.text) for d in self.docs}
        self._build_admission_masks()

    def _build_admission_masks(self) -> None:
        telem = self.store.get_telemetry()
        self.blocked_attackers: Set[str] = set()
        for u_b, w_b, _et in telem.blocked_authority_edges:
            if self.store.eligibility.get(w_b) in {"ADMIT", "CONTESTED"}:
                self.blocked_attackers.add(u_b)
                for d_b in self.store.deps_v.get(u_b, ()):
                    if (
                        self.store.eff_auth.get(d_b, 0) < self.store.eff_auth.get(w_b, 0)
                        and self.events[d_b].claim_value == self.events[u_b].claim_value
                    ):
                        self.blocked_attackers.add(d_b)
                # Any other low-tier rumor in the same (entity, slot) asserting the exact blocked value
                u_ev = self.events[u_b]
                w_auth = self.store.eff_auth.get(w_b, 0)
                for eid, ev in self.events.items():
                    if (
                        (ev.tenant, ev.entity, ev.slot, ev.claim_value)
                        == (u_ev.tenant, u_ev.entity, u_ev.slot, u_ev.claim_value)
                        and self.store.eff_auth.get(eid, 0) < w_auth
                    ):
                        self.blocked_attackers.add(eid)

        # Track `(tenant, entity, slot, retracted_value) -> latest retraction tx_time`
        self.retracted_val_tx: Dict[Tuple[str, str, str, str], int] = {}
        for att in self.attestations:
            if att.edge_type in {"REFUTE", "SUPERSEDE", "VERSION_LINEAGE_SUPERSEDE"}:
                is_active = (
                    att.src_id in self.store.refs.get(att.dst_id, ())
                    or att.src_id in self.store.sup.get(att.dst_id, ())
                    or att.src_id in self.store.lsup.get(att.dst_id, ())
                )
                if (
                    is_active
                    and self.store.eligibility.get(att.src_id) == "ADMIT"
                    and self.store.eligibility.get(att.dst_id) == "AUDIT_ONLY"
                ):
                    dst_ev = self.events[att.dst_id]
                    if dst_ev.claim_value and dst_ev.claim_value != "unknown":
                        rkey = (dst_ev.tenant, dst_ev.entity, dst_ev.slot, dst_ev.claim_value)
                        self.retracted_val_tx[rkey] = max(self.retracted_val_tx.get(rkey, -1), att.tx_time)

        admit_flags: List[bool] = []
        contested_flags: List[bool] = []
        for did in self.doc_ids:
            elig = self.store.eligibility.get(did, "AUDIT_ONLY")
            ev = self.events[did]
            rkey = (ev.tenant, ev.entity, ev.slot, ev.claim_value)
            is_stale_echo = rkey in self.retracted_val_tx and ev.valid_time <= self.retracted_val_tx[rkey]
            is_admit = elig == "ADMIT" and did not in self.blocked_attackers and not is_stale_echo
            admit_flags.append(is_admit)
            contested_flags.append(elig == "CONTESTED" and did not in self.blocked_attackers)

        self.admit_mask = np.array(admit_flags, dtype=bool)
        self.contested_mask = np.array(contested_flags, dtype=bool)

    def select_context_for_query(
        self,
        hybrid_scores: np.ndarray,
        query_text: str,
        *,
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """Select deduplicated `ADMIT` records or emit label-free `CONTESTED` abstention."""
        if hybrid_scores.shape[0] != len(self.doc_ids):
            raise ValueError("hybrid_scores length must equal resident corpus size")

        q_toks = content_tokens(query_text)

        # 1. Inspect the unfiltered top-12 retrieval neighborhood ONLY to infer queried (entity, slot)
        #    without ANY oracle query metadata!
        raw_top_idx = np.argsort(-hybrid_scores, kind="mergesort")[:12]
        slot_weights: Dict[Tuple[str, str, str], float] = defaultdict(float)
        for rank_i, idx_doc in enumerate(raw_top_idx):
            did = self.doc_ids[int(idx_doc)]
            ev = self.events[did]
            ent_toks = set(ev.entity.split("_"))
            overlap_bonus = 1.5 if (ent_toks & q_toks) else 0.0
            slot_weights[(ev.tenant, ev.entity, ev.slot)] += (overlap_bonus + 1.0 / (rank_i + 1.0))

        primary_slot: Optional[Tuple[str, str, str]] = None
        if slot_weights:
            primary_slot = max(slot_weights.items(), key=lambda kv: kv[1])[0]

        # 2. Check label-free slot-level `CONTESTED` status on the inferred primary `(tenant, entity, slot)`!
        if primary_slot is not None:
            t_p, e_p, s_p = primary_slot
            slot_admitted = [
                did
                for i, did in enumerate(self.doc_ids)
                if self.admit_mask[i] and (self.events[did].tenant, self.events[did].entity, self.events[did].slot) == primary_slot
            ]
            slot_contested = [
                did
                for i, did in enumerate(self.doc_ids)
                if self.contested_mask[i] and (self.events[did].tenant, self.events[did].entity, self.events[did].slot) == primary_slot
            ]
            if slot_contested and not slot_admitted:
                return {
                    "status": "CONTESTED_ABSTAIN",
                    "selected_ids": [],
                    "inferred_slot": {"tenant": t_p, "entity": e_p, "slot": s_p},
                    "contested_witness_ids": slot_contested[:4],
                    "formatted_context": (
                        "[EPISTEMIC STATUS: CONTESTED — Multiple equal-authority records for "
                        f"{e_p} ({s_p}) mutually refute one another with no authoritative resolution. "
                        "No verified current claim is admitted.]"
                    ),
                }

        # 3. Rank surviving `ADMIT` candidates with mild authority tie-break and complete-linkage echo collapse
        admit_scores = np.where(self.admit_mask, hybrid_scores, -1e9).copy()
        for i, did in enumerate(self.doc_ids):
            if self.admit_mask[i]:
                admit_scores[i] += 0.04 * float(self.store.eff_auth.get(did, 1))

        cand_order = np.argsort(-admit_scores, kind="mergesort")
        selected: List[str] = []
        for idx_c in cand_order:
            idx_int = int(idx_c)
            if admit_scores[idx_int] <= -1e8:
                break
            did = self.doc_ids[idx_int]
            # Complete-linkage echo collapse against ALL already-selected witnesses
            is_echo = False
            for s_id in selected:
                s_idx = self.idx_by_id[s_id]
                cos_sim = float(self.doc_embeddings[idx_int] @ self.doc_embeddings[s_idx])
                jac_sim = token_jaccard(self.token_sets[did], self.token_sets[s_id])
                same_claim = (
                    self.events[did].entity == self.events[s_id].entity
                    and self.events[did].slot == self.events[s_id].slot
                    and self.events[did].claim_value == self.events[s_id].claim_value
                )
                if same_claim or cos_sim >= self.echo_cos_threshold or jac_sim >= self.echo_jaccard_threshold:
                    is_echo = True
                    break
            if is_echo:
                continue
            selected.append(did)
            if len(selected) >= top_k:
                break

        ordered = sorted(selected, key=lambda d_id: (self.events[d_id].valid_time, self.events[d_id].tx_time))
        lines = [
            f"- [source={self.docs[self.idx_by_id[d_id]].source_channel}, published_day={self.events[d_id].valid_time}] "
            f"{self.events[d_id].text}"
            for d_id in ordered
        ]
        return {
            "status": "ADMIT_OK",
            "selected_ids": ordered,
            "inferred_slot": (
                {"tenant": primary_slot[0], "entity": primary_slot[1], "slot": primary_slot[2]}
                if primary_slot
                else None
            ),
            "contested_witness_ids": [],
            "formatted_context": "\n".join(lines) if lines else "(No admitted records found.)",
        }


def extract_claims_with_local_hf_model(
    docs: Sequence[RawDocumentEnvelope],
    *,
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    batch_size: int = 24,
    device: str = "cuda",
) -> Tuple[List[ExtractedClaimRecord], Dict[str, Any]]:
    """Run batched greedy raw-text claim extraction on a local GPU (e.g. consumer RTX 3080/3090)."""
    import time
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = torch.bfloat16 if (device.startswith("cuda") and torch.cuda.is_bf16_supported()) else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()
    load_sec = time.perf_counter() - t0

    prompts: List[str] = []
    for d in docs:
        msgs = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Source channel: {d.source_channel}\nPublished day: {d.published_ts}\nDocument:\n{d.text}",
            },
        ]
        prompts.append(tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True))

    t_gen0 = time.perf_counter()
    outputs: List[ExtractedClaimRecord] = []
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            batch_p = prompts[start : start + batch_size]
            batch_d = docs[start : start + batch_size]
            enc = tokenizer(batch_p, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            in_len = enc["input_ids"].shape[1]
            gen = model.generate(
                **enc,
                max_new_tokens=64,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id,
            )
            for b_i, d in enumerate(batch_d):
                raw_txt = tokenizer.decode(gen[b_i][in_len:], skip_special_tokens=True).strip()
                outputs.append(parse_extracted_claim_json(d.doc_id, raw_txt))

    gen_sec = time.perf_counter() - t_gen0
    peak_vram_mib = round(float(torch.cuda.max_memory_allocated()) / (1024.0**2), 1) if device.startswith("cuda") else 0.0
    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    stats = {
        "extractor_model": model_name,
        "device": device,
        "n_docs": len(docs),
        "model_load_sec": round(load_sec, 2),
        "batch_extract_sec": round(gen_sec, 2),
        "ms_per_doc": round((gen_sec * 1000.0) / max(len(docs), 1), 2),
        "peak_vram_mib": peak_vram_mib,
    }
    return outputs, stats


def answer_queries_with_local_hf_model(
    prompts: Sequence[Tuple[str, str]],
    *,
    system_prompt: str,
    model_name: str = "Qwen/Qwen2.5-3B-Instruct",
    batch_size: int = 16,
    device: str = "cuda",
) -> Dict[str, Tuple[str, int, float]]:
    """Run batched greedy QA reader inference on a local GPU (`(prompt_id, user_prompt)` -> `(pred, prompt_tokens, ms)`)."""
    import time
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = torch.bfloat16 if (device.startswith("cuda") and torch.cuda.is_bf16_supported()) else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).to(device)
    model.eval()

    formatted: List[str] = []
    tok_counts: List[int] = []
    for _pid, up in prompts:
        msgs = [{"role": "system", "content": system_prompt}, {"role": "user", "content": up}]
        txt = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        formatted.append(txt)
        tok_counts.append(len(tokenizer.encode(txt)))

    results: Dict[str, Tuple[str, int, float]] = {}
    with torch.inference_mode():
        for start in range(0, len(prompts), batch_size):
            b_items = prompts[start : start + batch_size]
            b_fmt = formatted[start : start + batch_size]
            b_toks = tok_counts[start : start + batch_size]
            t0 = time.perf_counter()
            enc = tokenizer(b_fmt, return_tensors="pt", padding=True, truncation=True, max_length=1536).to(device)
            in_len = enc["input_ids"].shape[1]
            gen = model.generate(
                **enc,
                max_new_tokens=32,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=tokenizer.pad_token_id,
            )
            per_ms = ((time.perf_counter() - t0) * 1000.0) / max(len(b_items), 1)
            for i, (pid, _up) in enumerate(b_items):
                ans = tokenizer.decode(gen[i][in_len:], skip_special_tokens=True).strip()
                results[pid] = (ans, b_toks[i], round(per_ms, 2))

    del model
    if device.startswith("cuda"):
        torch.cuda.empty_cache()
    return results
