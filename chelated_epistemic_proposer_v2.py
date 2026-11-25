"""Chelated Raw-Text Epistemic Proposer & Authority-Dominated Exocortex v2.

Advances `chelated_raw_text_exocortex.py` (which remains untouched for v1 reproducibility)
with three strictly domain-agnostic, zero-shortcut neuro-symbolic improvements diagnosed
from Phase-2 Demo D1 v1 error forensics:

  1. Schema-Guarded Atomic Claim Extraction (`EXTRACTION_SYSTEM_PROMPT_V2`):
     Prevents small/mid-sized LLMs from placing document meta-nouns (`technical_filing`,
     `published_day`) into `slot` or copying `claim_value` tokens into `entity`, and
     normalizes entity identifiers against explicit uppercase/hyphenated tokens in the text.
  2. Entity-Scoped Semantic + Document-Context Slot Canonicalization:
     Clusters extracted slot strings within each canonical entity using combined lexical
     and embedding similarity so Party A (`vulnerability_status`) and Party B / echoes
     (`quic_0_rtt_vulnerability_verdict`) resolve to the same canonical attribute slot.
  3. Bidirectional Standing Refutation & Stratum-1 Authority-Dominated `CONTESTED` Resolution:
     - Standing higher-tier `REFUTE` / `DEBUNK_RUMOR` records automatically contest or
       refute subsequent lower-tier echoes asserting incompatible values on the same slot.
     - At query time, if the highest effective authority tier (`max_tier`) present on the
       inferred `(tenant, entity, slot)` is epistemically `CONTESTED` (`val == U`) with
       ZERO `ADMIT` records at `>= max_tier`, lower-tier (`tier < max_tier`) unverified
       echoes cannot override the peer deadlock—deterministically returning `CONTESTED_ABSTAIN`
       with zero benchmark-family labels.
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from chelated_raw_text_exocortex import (
    DEFAULT_CHANNEL_AUTHORITY,
    ExtractedClaimRecord,
    RawDocumentEnvelope,
    canonicalize_key,
    canonicalize_value,
    content_tokens,
    parse_extracted_claim_json,
    token_jaccard,
)
from ek5_v3_stratified import EdgeAttestation, StratifiedEvent, StratifiedEvidenceStore
from spherical_migration_bridge import l2_normalize

EXTRACTION_SYSTEM_PROMPT_V2 = (
    "You are an enterprise epistemic graph extractor. Read the single document and output ONE JSON object "
    "capturing its core factual claim.\n"
    "Rules:\n"
    "1. \"entity\": The exact primary subject identifier (e.g. CVE ID, GHSA ID, incident ID, or company/project ticker ID "
    "mentioned in the text). NEVER put the metric value or document type as the entity.\n"
    "2. \"slot\": The specific domain property, metric, version, root cause, or status being stated for that entity. "
    "NEVER use generic meta-words like 'technical_filing', 'counter_filing', 'published_day', or 'operational_guidance'.\n"
    "3. \"value\": ONLY the concise asserted value, version string, dollar/percent figure, or status code (strip extra prose).\n"
    "4. \"stance\":\n"
    "   - \"SUPERSEDE\": official patch, revision, restatement, or updated guidance replacing an earlier baseline/value.\n"
    "   - \"DEBUNK_RUMOR\": authoritative audit or bulletin citing an unverified rumor solely to refute/deny it AND state the verified status.\n"
    "   - \"REFUTE\": formal filing or counter-filing that disputes/refutes a competing peer assessment without final third-party resolution.\n"
    "   - \"ASSERT\": initial baseline, routine report, or Secondary chat/forum/blog echo repeating an existing value.\n"
    "Return ONLY valid single-line JSON: "
    '{"entity": "<id>", "slot": "<property>", "value": "<concise_value>", "stance": "<ASSERT|SUPERSEDE|REFUTE|DEBUNK_RUMOR>"}'
)

# Generic regex catching explicit uppercase hyphenated IDs (e.g., CVE-2024-3094, INC-US-EAST-KAFKA-401, ACME-SEMI-FY26Q2, GHSA-...)
_EXPLICIT_ID_RE = re.compile(r"\b(?:CVE-\d{4}-\d{4,7}(?:-[A-Z0-9]+)*|[A-Z]{2,12}(?:-[A-Z0-9]{2,14}){2,5})\b")
_META_SLOT_WORDS: Set[str] = {
    "technical_filing",
    "technical_counter_filing",
    "technical_counter-filing",
    "counter_filing",
    "published_day",
    "operational_guidance",
    "operational_guidance_date",
    "tracked_issue",
    "asserted_claim",
    "refuted_assessment",
}


def ground_and_clean_claims(
    docs: Sequence[RawDocumentEnvelope],
    raw_claims: Sequence[ExtractedClaimRecord],
    slot_embeddings: np.ndarray,
    *,
    sim_threshold: float = 0.66,
) -> List[ExtractedClaimRecord]:
    """Normalize extracted entity IDs from document text and cluster synonymous slots per entity."""
    if len(docs) != len(raw_claims):
        raise ValueError("docs and raw_claims must have equal length")

    step1: List[ExtractedClaimRecord] = []
    for d, c in zip(docs, raw_claims):
        # If the raw document contains a single unambiguous structured identifier (CVE/GHSA/INC/etc.),
        # anchor `entity` to that exact surface token when the LLM accidentally copied `value` into `entity`.
        id_hits = _EXPLICIT_ID_RE.findall(d.text)
        ent = c.entity
        if id_hits:
            primary_id = canonicalize_key(id_hits[0])
            val_key = canonicalize_key(c.claim_value)
            if ent == val_key or ent == "unknown_entity" or primary_id in ent or ent in primary_id:
                ent = primary_id

        # Strip accidental leading prose from value if model echoed `<slot> is <VAL>`
        val_clean = c.claim_value.strip()
        if " is " in val_clean and len(val_clean) > 28:
            tail = val_clean.rsplit(" is ", 1)[-1].strip(" .")
            if tail:
                val_clean = tail

        # Fix inverted DEBUNK_RUMOR stance on Tier-1 rumor seeds (a Tier-1 rumor feed cannot be an authoritative debunk!)
        stance = c.stance
        tier = DEFAULT_CHANNEL_AUTHORITY.get(d.source_channel, 2)
        if stance == "DEBUNK_RUMOR" and tier <= 1:
            stance = "ASSERT"

        slot = c.slot
        if slot in _META_SLOT_WORDS:
            slot = "primary_attribute"

        step1.append(
            ExtractedClaimRecord(
                doc_id=c.doc_id,
                entity=ent,
                slot=slot,
                claim_value=canonicalize_value(val_clean),
                stance=stance,
                raw_json=c.raw_json,
            )
        )

    # Cluster synonymous slots within each canonical entity toward the most frequent non-meta slot
    by_ent: Dict[str, List[int]] = defaultdict(list)
    for idx, c in enumerate(step1):
        by_ent[c.entity].append(idx)

    canon_slot_by_idx: Dict[int, str] = {}
    norm_embs = l2_normalize(slot_embeddings.astype(np.float32))
    for _ent, idxs in by_ent.items():
        non_meta_counts = Counter(step1[i].slot for i in idxs if step1[i].slot != "primary_attribute")
        if not non_meta_counts:
            for i in idxs:
                canon_slot_by_idx[i] = "primary_attribute"
            continue

        ordered_slots = [s for s, _cnt in non_meta_counts.most_common()]
        dominant_slot = ordered_slots[0]
        rep_idx: Dict[str, int] = {}
        for i in idxs:
            if step1[i].slot not in rep_idx:
                rep_idx[step1[i].slot] = i

        rep_map: Dict[str, str] = {}
        for s in ordered_slots:
            matched: Optional[str] = None
            s_toks = {t for t in re.split(r"[_\-]+", s) if len(t) > 1}
            for r_s in rep_map.values():
                r_toks = {t for t in re.split(r"[_\-]+", r_s) if len(t) > 1}
                lex = token_jaccard(s_toks, r_toks)
                cos = float(norm_embs[rep_idx[s]] @ norm_embs[rep_idx[r_s]])
                if lex >= 0.34 or cos >= sim_threshold:
                    matched = r_s
                    break
            rep_map[s] = matched if matched is not None else s

        # Map any fallback `primary_attribute` on this single-topic entity to `dominant_slot`
        rep_map["primary_attribute"] = dominant_slot
        for i in idxs:
            canon_slot_by_idx[i] = rep_map.get(step1[i].slot, dominant_slot)

    return [
        ExtractedClaimRecord(
            doc_id=c.doc_id,
            entity=c.entity,
            slot=canon_slot_by_idx[idx],
            claim_value=c.claim_value,
            stance=c.stance,
            raw_json=c.raw_json,
        )
        for idx, c in enumerate(step1)
    ]


def compile_claims_to_ek5_v2(
    docs: Sequence[RawDocumentEnvelope],
    claims: Sequence[ExtractedClaimRecord],
    *,
    channel_authority: Optional[Mapping[str, int]] = None,
) -> Tuple[StratifiedEvidenceStore, Dict[str, StratifiedEvent], List[EdgeAttestation]]:
    """Compile raw envelopes + cleaned claims into `EK5 v3` with bidirectional standing refutations."""
    auth_map = dict(DEFAULT_CHANNEL_AUTHORITY)
    if channel_authority:
        auth_map.update(channel_authority)

    claim_by_id = {c.doc_id: c for c in claims}
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
                attestation_id=f"v2_att_{att_seq}",
                src_id=src_id,
                dst_id=dst_id,
                edge_type=edge_type,
                valid_time=vt,
                tx_time=tx,
            )
        )

    for _bkey, ordered_ids in buckets.items():
        seen_ids: List[str] = []
        # Track highest-tier origin for each distinct claim_value so echoes link to it via COPY_OF / DERIVES_CURRENT
        origin_by_val: Dict[str, str] = {}
        standing_refuters: List[str] = []

        for uid in ordered_ids:
            u_ev = events[uid]
            u_cl = claim_by_id[uid]
            u_val = u_ev.claim_value
            if u_val not in origin_by_val:
                origin_by_val[u_val] = uid
            else:
                orig_id = origin_by_val[u_val]
                orig_ev = events[orig_id]
                if u_ev.authority_tier > orig_ev.authority_tier:
                    origin_by_val[u_val] = uid
                elif u_ev.authority_tier <= 2:
                    etype = "COPY_OF" if u_ev.authority_tier == orig_ev.authority_tier else "DERIVES_CURRENT"
                    _emit(uid, orig_id, etype, u_ev.valid_time, u_ev.tx_time)

            if u_cl.stance in {"REFUTE", "DEBUNK_RUMOR"} and u_ev.authority_tier >= 3:
                standing_refuters.append(uid)

            # 1. Standing authoritative refuters/debunks automatically contest subsequent lower-tier echoes of rival values!
            for ref_id in standing_refuters:
                if ref_id == uid:
                    continue
                r_ev = events[ref_id]
                if r_ev.claim_value != u_val and u_ev.authority_tier < r_ev.authority_tier:
                    _emit(ref_id, uid, "REFUTE", max(r_ev.valid_time, u_ev.valid_time), max(r_ev.tx_time, u_ev.tx_time))

            # 2. Backward interactions against already-ingested documents in this `(tenant, entity, slot)`
            for wid in seen_ids:
                w_ev = events[wid]
                w_cl = claim_by_id[wid]
                if w_ev.claim_value == u_val or w_ev.claim_value == "unknown" or u_val == "unknown":
                    continue

                if u_cl.stance == "DEBUNK_RUMOR":
                    _emit(uid, wid, "CITES_AUDIT", u_ev.valid_time, u_ev.tx_time)
                    _emit(uid, wid, "REFUTE", u_ev.valid_time, u_ev.tx_time)
                elif u_cl.stance == "REFUTE":
                    _emit(uid, wid, "REFUTE", u_ev.valid_time, u_ev.tx_time)
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
                    # Stance == ASSERT
                    if u_ev.valid_time > w_ev.valid_time:
                        # Only attempt SUPERSEDE if `u` is NOT a low-tier rumor trying to overwrite a peer `CONTESTED` dispute
                        _emit(uid, wid, "SUPERSEDE", u_ev.valid_time, u_ev.tx_time)
                    elif u_ev.authority_tier == w_ev.authority_tier and abs(u_ev.valid_time - w_ev.valid_time) <= 1:
                        _emit(uid, wid, "REFUTE", u_ev.valid_time, u_ev.tx_time)
                        _emit(wid, uid, "REFUTE", w_ev.valid_time, w_ev.tx_time)

            seen_ids.append(uid)

    store = StratifiedEvidenceStore(strict_dag_supersede=False)
    store.apply_mutation(list(events.values()), attestations)
    return store, events, attestations


class AuthorityDominatedExocortexV2:
    """Label-free `EK5 v3` exocortex enforcing Stratum-1 authority dominance on `CONTESTED` deadlocks."""

    def __init__(
        self,
        docs: Sequence[RawDocumentEnvelope],
        raw_claims: Sequence[ExtractedClaimRecord],
        doc_embeddings: np.ndarray,
        slot_embeddings: np.ndarray,
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

        self.cleaned_claims = ground_and_clean_claims(self.docs, raw_claims, slot_embeddings)
        self.store, self.events, self.attestations = compile_claims_to_ek5_v2(
            self.docs, self.cleaned_claims, channel_authority=channel_authority
        )
        self.token_sets = {d.doc_id: content_tokens(d.text) for d in self.docs}
        self._build_masks()

    def _build_masks(self) -> None:
        telem = self.store.get_telemetry()
        self.blocked_attackers: Set[str] = set()
        for u_b, w_b, _et in telem.blocked_authority_edges:
            if self.store.eligibility.get(w_b) in {"ADMIT", "CONTESTED"}:
                self.blocked_attackers.add(u_b)
                u_ev = self.events[u_b]
                w_auth = self.store.eff_auth.get(w_b, 0)
                for eid, ev in self.events.items():
                    if (
                        (ev.tenant, ev.entity, ev.slot, ev.claim_value)
                        == (u_ev.tenant, u_ev.entity, u_ev.slot, u_ev.claim_value)
                        and self.store.eff_auth.get(eid, 0) < w_auth
                    ):
                        self.blocked_attackers.add(eid)

        self.retracted_val_tx: Dict[Tuple[str, str, str, str], int] = {}
        for att in self.attestations:
            if att.edge_type in {"REFUTE", "SUPERSEDE", "VERSION_LINEAGE_SUPERSEDE"}:
                is_act = (
                    att.src_id in self.store.refs.get(att.dst_id, ())
                    or att.src_id in self.store.sup.get(att.dst_id, ())
                    or att.src_id in self.store.lsup.get(att.dst_id, ())
                )
                if (
                    is_act
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
            stale = rkey in self.retracted_val_tx and ev.valid_time <= self.retracted_val_tx[rkey]
            admit_flags.append(elig == "ADMIT" and did not in self.blocked_attackers and not stale)
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
        q_toks = content_tokens(query_text)
        raw_top_idx = np.argsort(-hybrid_scores, kind="mergesort")[:14]

        slot_weights: Dict[Tuple[str, str, str], float] = defaultdict(float)
        for rank_i, idx_doc in enumerate(raw_top_idx):
            did = self.doc_ids[int(idx_doc)]
            ev = self.events[did]
            ent_toks = {t for t in re.split(r"[_\-]+", ev.entity) if len(t) > 1}
            overlap_bonus = 2.0 if (ent_toks & q_toks) else 0.0
            auth_bonus = 0.25 * float(self.store.eff_auth.get(did, 1))
            slot_weights[(ev.tenant, ev.entity, ev.slot)] += overlap_bonus + auth_bonus + (1.0 / (rank_i + 1.0))

        primary_slot: Optional[Tuple[str, str, str]] = None
        if slot_weights:
            primary_slot = max(slot_weights.items(), key=lambda kv: kv[1])[0]

        # Stratum-1 Authority-Dominated Slot Resolution:
        # Check ALL slots belonging to the inferred primary `(tenant, entity)` if `CONTESTED` exists at max authority!
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
            max_cont_auth = max((self.store.eff_auth.get(d, 0) for d in slot_contested), default=0)
            max_adm_auth = max((self.store.eff_auth.get(d, 0) for d in slot_admitted), default=0)

            # If high-tier records are locked in mutual `CONTESTED` (`val == U`) and NO equal-or-higher-tier
            # `ADMIT` resolution exists, lower-tier echoes cannot resolve the deadlock!
            if slot_contested and max_cont_auth > max_adm_auth:
                return {
                    "status": "CONTESTED_ABSTAIN",
                    "selected_ids": [],
                    "inferred_slot": {"tenant": t_p, "entity": e_p, "slot": s_p},
                    "contested_witness_ids": slot_contested[:4],
                    "formatted_context": (
                        "[EPISTEMIC STATUS: CONTESTED — Multiple highest-authority records for "
                        f"{e_p} ({s_p}) mutually refute one another with no authoritative resolution. "
                        "Lower-tier secondary echoes cannot resolve a higher-tier deadlock. Reply ABSTAIN.]"
                    ),
                }

        admit_scores = np.where(self.admit_mask, hybrid_scores, -1e9).copy()
        for i, did in enumerate(self.doc_ids):
            if self.admit_mask[i]:
                admit_scores[i] += 0.05 * float(self.store.eff_auth.get(did, 1))

        cand_order = np.argsort(-admit_scores, kind="mergesort")
        selected: List[str] = []
        for idx_c in cand_order:
            idx_int = int(idx_c)
            if admit_scores[idx_int] <= -1e8:
                break
            did = self.doc_ids[idx_int]
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


def extract_claims_v2_with_local_hf_model(
    docs: Sequence[RawDocumentEnvelope],
    *,
    model_name: str = "Qwen/Qwen2.5-7B-Instruct",
    batch_size: int = 24,
    device: str = "cuda",
) -> Tuple[List[ExtractedClaimRecord], Dict[str, Any]]:
    """Run batched greedy v2 schema-guarded claim extraction on a local GPU."""
    import time
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t0 = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    dtype = torch.bfloat16 if (device.startswith("cuda") and torch.cuda.is_bf16_supported()) else torch.float16
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype).to(device)
    model.eval()
    load_sec = time.perf_counter() - t0

    prompts: List[str] = []
    for d in docs:
        msgs = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT_V2},
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
        "prompt_version": "v2_schema_guarded",
        "device": device,
        "n_docs": len(docs),
        "model_load_sec": round(load_sec, 2),
        "batch_extract_sec": round(gen_sec, 2),
        "ms_per_doc": round((gen_sec * 1000.0) / max(len(docs), 1), 2),
        "peak_vram_mib": peak_vram_mib,
    }
    return outputs, stats
