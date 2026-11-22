"""Flagship #2 (IQ-04 / EK7): Temporal Evidence Correction, Bounded Retraction & ScaleMAS Surrogate.

Implements the exact comparison structure specified in:
  - docs/research/innovation-test-queue-2026-09-12/04-temporal-evidence-correction.md
  - docs/research/scalemas-temporal-evidence-comparison-2026-10-15.md

Separates:
  - Retrieval ranking score vs. epistemic eligibility (`ADMIT`, `AUDIT_ONLY`, `ABSTAIN`).
  - Bitemporal valid_time vs. transaction_time cutoffs (`as-of` queries).
  - Sparse horizon-bounded dependency invalidation (`EK5`: exact full-recompute parity + <=25% work).
  - Observable-only unlinked paraphrase/value-cluster copy collapse (zero oracle provenance leakage).

Honest comparator labeling:
  - `C1_scalemas_unsigned` is explicitly tagged `SCALEMAS_ADAPTED_SURROGATE` (BM25 + decaying unsigned
    citation weight where refuting/superseding citations ALSO increment cited weight per arXiv:2610.16120v1).
  - `C5_oracle_upper_bound` is segregated evaluator-only and never participates in deployable selection.
"""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

_TOK_RE = re.compile(r"[a-z0-9]+")

VALID_EDGE_TYPES = frozenset({"SUPPORT", "REFUTE", "SUPERSEDE", "DERIVES_FROM", "COPY_OF"})

CASE_FAMILIES: Tuple[str, ...] = (
    "early_popular_error_corrected",
    "popular_valid_control",
    "unpopular_valid_minority",
    "refutation_supersession_chain",
    "delayed_observation_as_of",
    "erroneous_retraction_reinstated",
    "hidden_common_cause_paraphrase_copy",
    "missing_edge_and_cycle_stress",
)


@dataclass(frozen=True)
class EvidenceEvent:
    """Immutable append-only bitemporal evidence record."""

    event_id: str
    lifecycle_id: str
    entity: str
    slot: str
    claim_value: str
    text: str
    valid_time: int
    transaction_time: int
    source_id: str
    role: str = "CLAIM"  # CLAIM | CORRECTION | RETRACTION | REINSTATEMENT | DERIVED | COPY | SIBLING
    # Evaluator-only hidden lineage metadata (NEVER read by C0..C4; read ONLY by C5 oracle & evaluator!)
    hidden_root_id: Optional[str] = None
    hidden_is_true_at_query: bool = True


@dataclass(frozen=True)
class EvidenceEdge:
    """Directed typed edge visible in the observable evidence graph."""

    src_id: str
    dst_id: str
    edge_type: str  # SUPPORT | REFUTE | SUPERSEDE | DERIVES_FROM | COPY_OF
    transaction_time: int


@dataclass(frozen=True)
class LifecycleQuery:
    """Factual query evaluated at declared (valid_cutoff, tx_cutoff)."""

    query_id: str
    lifecycle_id: str
    case_family: str
    split: str  # DEVELOPMENT | SELECT | REPORT
    entity: str
    slot: str
    query_text: str
    valid_cutoff: int
    tx_cutoff: int
    gold_answer: str  # canonical value or "ABSTAIN"
    is_post_correction: bool
    is_clean_control: bool
    stale_false_values: Tuple[str, ...] = ()


@dataclass
class LifecycleCorpus:
    """Self-contained collection of disjoint lifecycles, events, edges, and queries."""

    events: Dict[str, EvidenceEvent]
    edges: List[EvidenceEdge]
    queries: List[LifecycleQuery]
    lifecycle_splits: Dict[str, str]
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Deterministic 1,500-Lifecycle Generator (300 DEV / 400 SELECT / 800 REPORT)
# ---------------------------------------------------------------------------
_DOMAINS = (
    ("drug_dosage", "recommended_mg", ("50mg", "125mg", "250mg", "400mg")),
    ("bridge_load", "max_tonnage", ("18t", "32t", "45t", "60t")),
    ("reactor_valve", "pressure_bar", ("12bar", "24bar", "36bar", "48bar")),
    ("satellite_orbit", "altitude_km", ("420km", "550km", "780km", "910km")),
    ("tax_statute", "threshold_usd", ("14000usd", "28500usd", "52000usd", "75000usd")),
    ("protocol_tls", "min_version", ("tls11", "tls12", "tls13", "tls14")),
    ("grid_feeder", "breaker_amps", ("200a", "400a", "600a", "800a")),
    ("clinical_trial", "cohort_status", ("paused", "active", "terminated", "approved")),
)

_PARAPHRASE_TEMPLATES = (
    "Independent briefing confirms {entity} {slot} equals {val} per field log {tag}.",
    "Updated summary notes that {slot} for {entity} is recorded as {val} under audit {tag}.",
    "Operational bulletin specifies {entity} carries {slot} value {val} in digest {tag}.",
    "Secondary report restates {entity} {slot} at {val} referencing notice {tag}.",
    "Memo records {val} as the operative {slot} setting for {entity} in packet {tag}.",
)


def _det_hash_int(key: str) -> int:
    return int(hashlib.sha256(key.encode("utf-8")).hexdigest()[:12], 16)


def generate_iq04_lifecycle_corpus(
    *,
    n_dev: int = 300,
    n_select: int = 400,
    n_report: int = 800,
    seed: int = 42,
) -> LifecycleCorpus:
    """Generate 1,500 strictly disjoint temporal lifecycles across all 8 IQ-04/ScaleMAS families.

    Entities, source IDs, and lifecycle IDs are strictly disjoint across DEVELOPMENT, SELECT, and REPORT.
    """
    events: Dict[str, EvidenceEvent] = {}
    edges: List[EvidenceEdge] = []
    queries: List[LifecycleQuery] = []
    lifecycle_splits: Dict[str, str] = {}

    split_plan: List[Tuple[str, int]] = [
        ("DEVELOPMENT", n_dev),
        ("SELECT", n_select),
        ("REPORT", n_report),
    ]
    global_idx = 0

    for split_name, n_groups in split_plan:
        for local_i in range(n_groups):
            fam = CASE_FAMILIES[local_i % len(CASE_FAMILIES)]
            lc_id = f"lc_{split_name[:3].lower()}_{global_idx:04d}"
            lifecycle_splits[lc_id] = split_name
            dom_name, slot, vals = _DOMAINS[(global_idx + seed) % len(_DOMAINS)]
            h = _det_hash_int(f"{seed}:{lc_id}:{fam}")
            ent = f"{dom_name}_{split_name[:3].lower()}_{global_idx:04d}"
            v_false = vals[h % len(vals)]
            v_true = vals[(h + 1) % len(vals)]
            v_alt = vals[(h + 2) % len(vals)]
            sib_slot = f"{slot}_aux"
            sib_val = vals[(h + 3) % len(vals)]

            def add_ev(
                suffix: str,
                *,
                sl: str,
                val: str,
                txt: str,
                vt: int,
                tt: int,
                src: str,
                role: str,
                h_root: Optional[str],
                h_true: bool,
            ) -> str:
                eid = f"{lc_id}_{suffix}"
                events[eid] = EvidenceEvent(
                    event_id=eid,
                    lifecycle_id=lc_id,
                    entity=ent,
                    slot=sl,
                    claim_value=val,
                    text=txt,
                    valid_time=vt,
                    transaction_time=tt,
                    source_id=f"{split_name[:3].lower()}_{src}_{global_idx}",
                    role=role,
                    hidden_root_id=h_root,
                    hidden_is_true_at_query=h_true,
                )
                return eid

            def add_edge(src_id: str, dst_id: str, etype: str, tt: int) -> None:
                edges.append(EvidenceEdge(src_id=src_id, dst_id=dst_id, edge_type=etype, transaction_time=tt))

            # Always include one unaffected sibling claim to test clean recall & zero collateral demotion
            sib_id = add_ev(
                "sib",
                sl=sib_slot,
                val=sib_val,
                txt=f"Verified baseline for {ent} {sib_slot} is {sib_val} with clean certification.",
                vt=1,
                tt=1,
                src="src_sib",
                role="SIBLING",
                h_root=None,
                h_true=True,
            )

            if fam == "early_popular_error_corrected":
                # Widely cited false claim early, single late authoritative correction + refutation
                r_bad = add_ev(
                    "root_bad", sl=slot, val=v_false,
                    txt=f"Primary bulletin states {ent} {slot} is {v_false} across operations.",
                    vt=1, tt=1, src="src_a", role="CLAIM", h_root=None, h_true=False,
                )
                # 6 popular supporting citations citing r_bad
                for c_i in range(6):
                    cid = add_ev(
                        f"pop_sup_{c_i}", sl=slot, val=v_false,
                        txt=f"Agent {c_i} confirms {ent} {slot} equals {v_false} following primary bulletin.",
                        vt=2 + c_i, tt=2 + c_i, src=f"src_pop_{c_i}", role="DERIVED",
                        h_root=r_bad, h_true=False,
                    )
                    add_edge(cid, r_bad, "SUPPORT", 2 + c_i)
                    add_edge(cid, r_bad, "DERIVES_FROM", 2 + c_i)
                # Late single correction & refutation at t=10
                corr = add_ev(
                    "corr", sl=slot, val=v_true,
                    txt=f"Correction notice: prior {ent} {slot} value {v_false} is retracted; true {slot} is {v_true}.",
                    vt=10, tt=10, src="src_audit", role="CORRECTION", h_root=None, h_true=True,
                )
                add_edge(corr, r_bad, "REFUTE", 10)
                add_edge(corr, r_bad, "SUPERSEDE", 10)
                queries.append(
                    LifecycleQuery(
                        query_id=f"q_{lc_id}_post", lifecycle_id=lc_id, case_family=fam, split=split_name,
                        entity=ent, slot=slot,
                        query_text=f"What is the current valid {slot} for {ent}?",
                        valid_cutoff=12, tx_cutoff=12, gold_answer=v_true,
                        is_post_correction=True, is_clean_control=False, stale_false_values=(v_false,),
                    )
                )

            elif fam == "popular_valid_control":
                # High-citation valid claim + 1 noisy unpopular false rumor (must NOT over-demote valid!)
                r_good = add_ev(
                    "root_good", sl=slot, val=v_true,
                    txt=f"Certified specification for {ent} {slot} is {v_true}.",
                    vt=1, tt=1, src="src_cert", role="CLAIM", h_root=None, h_true=True,
                )
                for c_i in range(5):
                    cid = add_ev(
                        f"sup_{c_i}", sl=slot, val=v_true,
                        txt=f"Corroboration {c_i}: {ent} {slot} remains {v_true}.",
                        vt=2 + c_i, tt=2 + c_i, src=f"src_ind_{c_i}", role="CLAIM",
                        h_root=None, h_true=True,
                    )
                    add_edge(cid, r_good, "SUPPORT", 2 + c_i)
                add_ev(
                    "rumor", sl=slot, val=v_false,
                    txt=f"Unverified note mentions {ent} {slot} might be {v_false}.",
                    vt=8, tt=8, src="src_noise", role="CLAIM", h_root=None, h_true=False,
                )
                queries.append(
                    LifecycleQuery(
                        query_id=f"q_{lc_id}_clean", lifecycle_id=lc_id, case_family=fam, split=split_name,
                        entity=ent, slot=slot,
                        query_text=f"What is the current valid {slot} for {ent}?",
                        valid_cutoff=12, tx_cutoff=12, gold_answer=v_true,
                        is_post_correction=False, is_clean_control=True, stale_false_values=(v_false,),
                    )
                )

            elif fam == "unpopular_valid_minority":
                # Low-citation valid claim vs retracted popular claim (valid minority must survive!)
                r_bad = add_ev(
                    "root_bad", sl=slot, val=v_false,
                    txt=f"Draft release reported {ent} {slot} as {v_false}.",
                    vt=1, tt=1, src="src_draft", role="CLAIM", h_root=None, h_true=False,
                )
                for c_i in range(4):
                    cid = add_ev(
                        f"cite_{c_i}", sl=slot, val=v_false,
                        txt=f"Echo {c_i} repeats {ent} {slot} is {v_false} from draft release.",
                        vt=2 + c_i, tt=2 + c_i, src=f"src_e_{c_i}", role="DERIVED",
                        h_root=r_bad, h_true=False,
                    )
                    add_edge(cid, r_bad, "DERIVES_FROM", 2 + c_i)
                    add_edge(cid, r_bad, "SUPPORT", 2 + c_i)
                ret = add_ev(
                    "ret", sl=slot, val=v_false,
                    txt=f"Retraction: draft release for {ent} {slot} ({v_false}) is invalid.",
                    vt=9, tt=9, src="src_rev", role="RETRACTION", h_root=None, h_true=True,
                )
                add_edge(ret, r_bad, "REFUTE", 9)
                add_ev(
                    "minor_valid", sl=slot, val=v_true,
                    txt=f"Single independent lab measurement sets {ent} {slot} to {v_true}.",
                    vt=5, tt=5, src="src_lab", role="CLAIM", h_root=None, h_true=True,
                )
                queries.append(
                    LifecycleQuery(
                        query_id=f"q_{lc_id}_post", lifecycle_id=lc_id, case_family=fam, split=split_name,
                        entity=ent, slot=slot,
                        query_text=f"What is the current valid {slot} for {ent}?",
                        valid_cutoff=12, tx_cutoff=12, gold_answer=v_true,
                        is_post_correction=True, is_clean_control=False, stale_false_values=(v_false,),
                    )
                )

            elif fam == "refutation_supersession_chain":
                # Multi-hop dependency chain r_bad -> d1 -> d2 -> d3 (depth 3..4) + late retraction of r_bad
                r_bad = add_ev(
                    "root_bad", sl=slot, val=v_false,
                    txt=f"Initial sensor feed reports {ent} {slot} is {v_false}.",
                    vt=1, tt=1, src="src_s0", role="CLAIM", h_root=None, h_true=False,
                )
                prev = r_bad
                for hop in range(1, 5):
                    did = add_ev(
                        f"der_{hop}", sl=slot, val=v_false,
                        txt=f"Stage {hop} synthesis derives {ent} {slot} = {v_false} from upstream stage.",
                        vt=1 + hop, tt=1 + hop, src=f"src_d{hop}", role="DERIVED",
                        h_root=r_bad, h_true=False,
                    )
                    add_edge(did, prev, "DERIVES_FROM", 1 + hop)
                    add_edge(did, prev, "SUPPORT", 1 + hop)
                    prev = did
                corr = add_ev(
                    "corr", sl=slot, val=v_true,
                    txt=f"Root sensor fault found: {ent} {slot} is {v_true}, superseding initial feed.",
                    vt=10, tt=10, src="src_fix", role="CORRECTION", h_root=None, h_true=True,
                )
                add_edge(corr, r_bad, "SUPERSEDE", 10)
                add_edge(corr, r_bad, "REFUTE", 10)
                queries.append(
                    LifecycleQuery(
                        query_id=f"q_{lc_id}_post", lifecycle_id=lc_id, case_family=fam, split=split_name,
                        entity=ent, slot=slot,
                        query_text=f"What is the current valid {slot} for {ent}?",
                        valid_cutoff=12, tx_cutoff=12, gold_answer=v_true,
                        is_post_correction=True, is_clean_control=False, stale_false_values=(v_false,),
                    )
                )

            elif fam == "delayed_observation_as_of":
                # Bitemporal test: claim_v1 valid at vt=1..5 (tx=1), correction effective vt=6 recorded late at tx=10
                # Query as-of (vt=10, tx=8) MUST return v_false (before tx=10 observed);
                # Query as-of (vt=10, tx=12) MUST return v_true (after tx=10 observed).
                r1 = add_ev(
                    "v1", sl=slot, val=v_false,
                    txt=f"Initial schedule sets {ent} {slot} to {v_false} effective period 1.",
                    vt=1, tt=1, src="src_s1", role="CLAIM", h_root=None, h_true=False,
                )
                for c_i in range(3):
                    cid = add_ev(
                        f"s1_c{c_i}", sl=slot, val=v_false,
                        txt=f"Log {c_i} confirms {ent} {slot} = {v_false}.",
                        vt=2 + c_i, tt=2 + c_i, src=f"src_log{c_i}", role="DERIVED",
                        h_root=r1, h_true=False,
                    )
                    add_edge(cid, r1, "DERIVES_FROM", 2 + c_i)
                    add_edge(cid, r1, "SUPPORT", 2 + c_i)
                r2 = add_ev(
                    "v2_delayed", sl=slot, val=v_true,
                    txt=f"Delayed revision (effective t=6, logged t=10): {ent} {slot} updated to {v_true}.",
                    vt=6, tt=10, src="src_s2", role="CORRECTION", h_root=None, h_true=True,
                )
                add_edge(r2, r1, "SUPERSEDE", 10)
                queries.append(
                    LifecycleQuery(
                        query_id=f"q_{lc_id}_post", lifecycle_id=lc_id, case_family=fam, split=split_name,
                        entity=ent, slot=slot,
                        query_text=f"As of transaction 12, what is the valid {slot} for {ent}?",
                        valid_cutoff=10, tx_cutoff=12, gold_answer=v_true,
                        is_post_correction=True, is_clean_control=False, stale_false_values=(v_false,),
                    )
                )

            elif fam == "erroneous_retraction_reinstated":
                # Valid root -> erroneous retraction at t=6 -> reinstatement at t=10!
                r_good = add_ev(
                    "root_good", sl=slot, val=v_true,
                    txt=f"Primary audited record establishes {ent} {slot} as {v_true}.",
                    vt=1, tt=1, src="src_orig", role="CLAIM", h_root=None, h_true=True,
                )
                bad_ret = add_ev(
                    "err_ret", sl=slot, val=v_false,
                    txt=f"Erroneous alert retracts {ent} {slot} {v_true} and suggests {v_false}.",
                    vt=6, tt=6, src="src_err", role="RETRACTION", h_root=None, h_true=False,
                )
                add_edge(bad_ret, r_good, "REFUTE", 6)
                reinst = add_ev(
                    "reinstate", sl=slot, val=v_true,
                    txt=f"Reinstatement: alert was erroneous; {ent} {slot} {v_true} is restored.",
                    vt=10, tt=10, src="src_board", role="REINSTATEMENT", h_root=None, h_true=True,
                )
                add_edge(reinst, bad_ret, "REFUTE", 10)
                add_edge(reinst, r_good, "SUPPORT", 10)
                queries.append(
                    LifecycleQuery(
                        query_id=f"q_{lc_id}_post", lifecycle_id=lc_id, case_family=fam, split=split_name,
                        entity=ent, slot=slot,
                        query_text=f"What is the current valid {slot} for {ent}?",
                        valid_cutoff=12, tx_cutoff=12, gold_answer=v_true,
                        is_post_correction=True, is_clean_control=False, stale_false_values=(v_false,),
                    )
                )

            elif fam == "hidden_common_cause_paraphrase_copy":
                # CRITICAL DISCRIMINATOR vs ordinary TMS (C3):
                # Root r_bad is retracted at t=10, AND 5 paraphrased unlinked copies exist from t=2..6
                # WITHOUT declared COPY_OF or DERIVES_FROM edges!
                # Plus 1 valid correction (v_true) at t=10.
                # Ordinary TMS (C3) retracts r_bad only -> 5 unlinked copies still outvote the 1 correction!
                # IQ-04 (C4) detects unlinked copy-cluster matching retracted claim value + timing burst
                # (using ONLY observable text/slot/value/timestamp, ZERO hidden lineage!)
                r_bad = add_ev(
                    "root_bad", sl=slot, val=v_false,
                    txt=f"Primary bulletin states {ent} {slot} equals {v_false} under field log A0.",
                    vt=1, tt=1, src="src_root", role="CLAIM", h_root=None, h_true=False,
                )
                for c_i in range(5):
                    tmpl = _PARAPHRASE_TEMPLATES[c_i % len(_PARAPHRASE_TEMPLATES)]
                    add_ev(
                        f"unlinked_copy_{c_i}", sl=slot, val=v_false,
                        txt=tmpl.format(entity=ent, slot=slot, val=v_false, tag=f"A{c_i + 1}"),
                        vt=2 + c_i, tt=2 + c_i, src=f"src_synd_{c_i}", role="COPY",
                        h_root=r_bad, h_true=False,
                    )
                    # Notice: NO edge to r_bad! Completely unlinked in the observable graph!
                corr = add_ev(
                    "corr", sl=slot, val=v_true,
                    txt=f"Official correction: {ent} {slot} value {v_false} is retracted; verified {slot} is {v_true}.",
                    vt=10, tt=10, src="src_audit", role="CORRECTION", h_root=None, h_true=True,
                )
                add_edge(corr, r_bad, "REFUTE", 10)
                add_edge(corr, r_bad, "SUPERSEDE", 10)
                queries.append(
                    LifecycleQuery(
                        query_id=f"q_{lc_id}_post", lifecycle_id=lc_id, case_family=fam, split=split_name,
                        entity=ent, slot=slot,
                        query_text=f"What is the current valid {slot} for {ent}?",
                        valid_cutoff=12, tx_cutoff=12, gold_answer=v_true,
                        is_post_correction=True, is_clean_control=False, stale_false_values=(v_false,),
                    )
                )

            else:  # missing_edge_and_cycle_stress
                # Cyclic support loop (a <-> b) supporting false value + retraction of a + missing edge to c
                # Plus conflicting unlinked claim v_alt -> if no clean winner or direct retraction of false value,
                # test both cycle-safe invalidation and clean sibling survival.
                ca = add_ev(
                    "cyc_a", sl=slot, val=v_false,
                    txt=f"Report A asserts {ent} {slot} is {v_false} based on partner report B.",
                    vt=1, tt=1, src="src_ca", role="CLAIM", h_root=None, h_true=False,
                )
                cb = add_ev(
                    "cyc_b", sl=slot, val=v_false,
                    txt=f"Report B asserts {ent} {slot} is {v_false} citing report A.",
                    vt=2, tt=2, src="src_cb", role="DERIVED", h_root=ca, h_true=False,
                )
                add_edge(cb, ca, "DERIVES_FROM", 2)
                add_edge(ca, cb, "DERIVES_FROM", 3)  # explicit cycle!
                add_edge(cb, ca, "SUPPORT", 2)
                add_edge(ca, cb, "SUPPORT", 3)
                # Missing-edge downstream copy of v_false at t=4
                add_ev(
                    "miss_c", sl=slot, val=v_false,
                    txt=f"Field note repeats {ent} {slot} = {v_false} without citation link.",
                    vt=4, tt=4, src="src_mc", role="COPY", h_root=ca, h_true=False,
                )
                corr = add_ev(
                    "corr", sl=slot, val=v_true,
                    txt=f"Audit retracts cyclic report A ({v_false}); confirmed {ent} {slot} is {v_true}.",
                    vt=10, tt=10, src="src_fix", role="CORRECTION", h_root=None, h_true=True,
                )
                add_edge(corr, ca, "REFUTE", 10)
                add_edge(corr, ca, "SUPERSEDE", 10)
                queries.append(
                    LifecycleQuery(
                        query_id=f"q_{lc_id}_post", lifecycle_id=lc_id, case_family=fam, split=split_name,
                        entity=ent, slot=slot,
                        query_text=f"What is the current valid {slot} for {ent}?",
                        valid_cutoff=12, tx_cutoff=12, gold_answer=v_true,
                        is_post_correction=True, is_clean_control=False, stale_false_values=(v_false, v_alt),
                    )
                )

            # Add clean unaffected sibling query on 1 out of 2 lifecycles to measure clean recall / collateral damage
            if local_i % 2 == 0:
                _ = sib_id
                queries.append(
                    LifecycleQuery(
                        query_id=f"q_{lc_id}_sib", lifecycle_id=lc_id, case_family=fam, split=split_name,
                        entity=ent, slot=sib_slot,
                        query_text=f"What is the verified {sib_slot} for {ent}?",
                        valid_cutoff=12, tx_cutoff=12, gold_answer=sib_val,
                        is_post_correction=False, is_clean_control=True, stale_false_values=(),
                    )
                )

            global_idx += 1

    return LifecycleCorpus(
        events=events,
        edges=edges,
        queries=queries,
        lifecycle_splits=lifecycle_splits,
        metadata={
            "n_dev": n_dev,
            "n_select": n_select,
            "n_report": n_report,
            "seed": seed,
            "n_events": len(events),
            "n_edges": len(edges),
            "n_queries": len(queries),
        },
    )


# ---------------------------------------------------------------------------
# Fast BM25 Retrieval + ScaleMAS Unsigned Citation Weighting
# ---------------------------------------------------------------------------
class ScopedBM25:
    """Deterministic Okapi BM25 over immutable evidence events."""

    def __init__(self, events: Mapping[str, EvidenceEvent], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self.doc_tf: Dict[str, Counter[str]] = {}
        self.doc_len: Dict[str, int] = {}
        df: Counter[str] = Counter()
        for eid, ev in events.items():
            toks = _TOK_RE.findall(ev.text.lower())
            c = Counter(toks)
            self.doc_tf[eid] = c
            self.doc_len[eid] = len(toks)
            for tok in c:
                df[tok] += 1
        n = max(len(events), 1)
        self.avgdl = sum(self.doc_len.values()) / float(n)
        self.idf = {tok: math.log(1.0 + (n - d + 0.5) / (d + 0.5)) for tok, d in df.items()}

    def score(self, query_text: str, candidate_ids: Sequence[str]) -> Dict[str, float]:
        q_toks = set(_TOK_RE.findall(query_text.lower()))
        out: Dict[str, float] = {}
        for eid in candidate_ids:
            tf_map = self.doc_tf[eid]
            dl = self.doc_len[eid]
            s = 0.0
            for tok in q_toks:
                tf = tf_map.get(tok, 0)
                if tf > 0:
                    idf = self.idf.get(tok, 0.0)
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (dl / max(self.avgdl, 1e-9)))
                    s += idf * ((tf * (self.k1 + 1.0)) / max(denom, 1e-9))
            out[eid] = s
        return out


# ---------------------------------------------------------------------------
# Bitemporal & Sparse Dependency Invalidation Engine (EK5 Exact Parity)
# ---------------------------------------------------------------------------
def compute_full_recompute_eligibility(
    events: Mapping[str, EvidenceEvent],
    edges: Sequence[EvidenceEdge],
    *,
    valid_cutoff: int,
    tx_cutoff: int,
    horizon: int = 8,
) -> Dict[str, str]:
    """Exact full-recompute truth-maintenance eligibility over observable edges.

    Returns map `event_id -> "ADMIT" | "AUDIT_ONLY"`.
    Supports refutation, supersession, reinstatement (refuting a refutation restores target!),
    and cycle-safe fixed-point propagation up to horizon H.
    """
    visible_ids = {
        eid
        for eid, ev in events.items()
        if ev.transaction_time <= tx_cutoff and ev.valid_time <= valid_cutoff
    }
    vis_edges = [
        e for e in edges
        if e.transaction_time <= tx_cutoff and e.src_id in visible_ids and e.dst_id in visible_ids
    ]

    # Build adjacency maps
    refuters_of: Dict[str, List[str]] = defaultdict(list)
    derives_from: Dict[str, List[str]] = defaultdict(list)
    copies_of: Dict[str, List[str]] = defaultdict(list)
    for e in vis_edges:
        if e.edge_type in ("REFUTE", "SUPERSEDE"):
            refuters_of[e.dst_id].append(e.src_id)
        elif e.edge_type == "DERIVES_FROM":
            derives_from[e.src_id].append(e.dst_id)
        elif e.edge_type == "COPY_OF":
            copies_of[e.src_id].append(e.dst_id)

    # Step 1: Resolve refutations bottom-up (if a refuter is itself refuted, it loses force -> reinstatement!)
    # Iterative fixed point over refutation status (handles chains of refutations cleanly)
    active_invalid: Set[str] = set()
    for _ in range(horizon + 2):
        new_invalid: Set[str] = set()
        for eid in visible_ids:
            # An event is directly invalidated if at least one *currently active* (non-invalid) event refutes/supersedes it
            if any(ref_id not in active_invalid for ref_id in refuters_of.get(eid, ())):
                new_invalid.add(eid)
        if new_invalid == active_invalid:
            break
        active_invalid = new_invalid

    # Step 2: Propagate invalidation across DERIVES_FROM and COPY_OF edges (cycle-safe BFS up to horizon)
    # Build reverse dependency index: parent -> children that derive from or copy parent
    dependents_of: Dict[str, List[str]] = defaultdict(list)
    for child, parents in derives_from.items():
        for p in parents:
            dependents_of[p].append(child)
    for child, parents in copies_of.items():
        for p in parents:
            dependents_of[p].append(child)

    invalid_all = set(active_invalid)
    queue: deque[Tuple[str, int]] = deque((eid, 0) for eid in sorted(active_invalid))
    while queue:
        cur, depth = queue.popleft()
        if depth >= horizon:
            continue
        for dep in dependents_of.get(cur, ()):
            if dep not in invalid_all:
                invalid_all.add(dep)
                queue.append((dep, depth + 1))

    return {eid: ("AUDIT_ONLY" if eid in invalid_all else "ADMIT") for eid in visible_ids}


def sparse_invalidate_incremental(
    base_eligibility: Mapping[str, str],
    events: Mapping[str, EvidenceEvent],
    edges: Sequence[EvidenceEdge],
    changed_event_ids: Sequence[str],
    *,
    valid_cutoff: int,
    tx_cutoff: int,
    horizon: int = 8,
) -> Tuple[Dict[str, str], Dict[str, int]]:
    """EK5 Sparse Dependency Invalidation — recomputes ONLY reachable subgraph around changed nodes.

    Returns `(updated_eligibility, work_stats)` where `work_stats` counts visited nodes & edges
    compared to full graph size so `sparse_work_ratio <= 0.25` is directly verified.
    """
    visible_ids = {
        eid
        for eid, ev in events.items()
        if ev.transaction_time <= tx_cutoff and ev.valid_time <= valid_cutoff
    }
    vis_edges = [
        e for e in edges
        if e.transaction_time <= tx_cutoff and e.src_id in visible_ids and e.dst_id in visible_ids
    ]

    # Build sparse neighbor adjacency (only edges touching reachable neighborhood are traversed)
    out_adj: Dict[str, List[EvidenceEdge]] = defaultdict(list)
    in_adj: Dict[str, List[EvidenceEdge]] = defaultdict(list)
    for e in vis_edges:
        out_adj[e.src_id].append(e)
        in_adj[e.dst_id].append(e)

    # Collect reachable affected component from changed_event_ids (bounded by horizon)
    affected: Set[str] = set()
    q: deque[Tuple[str, int]] = deque()
    for cid in changed_event_ids:
        if cid in visible_ids and cid not in affected:
            affected.add(cid)
            q.append((cid, 0))
    visited_edges = 0
    while q:
        u, d = q.popleft()
        if d >= horizon:
            continue
        # Outgoing REFUTE/SUPERSEDE affects target; incoming DERIVES_FROM/COPY_OF affects dependent
        for e in out_adj.get(u, ()):
            visited_edges += 1
            if e.edge_type in ("REFUTE", "SUPERSEDE") and e.dst_id not in affected:
                affected.add(e.dst_id)
                q.append((e.dst_id, d + 1))
        for e in in_adj.get(u, ()):
            visited_edges += 1
            if e.edge_type in ("DERIVES_FROM", "COPY_OF") and e.src_id not in affected:
                affected.add(e.src_id)
                q.append((e.src_id, d + 1))

    # Recompute eligibility strictly on the induced affected subgraph
    sub_events = {eid: events[eid] for eid in affected}
    sub_edges = [e for e in vis_edges if e.src_id in affected and e.dst_id in affected]
    sub_elig = compute_full_recompute_eligibility(
        sub_events, sub_edges, valid_cutoff=valid_cutoff, tx_cutoff=tx_cutoff, horizon=horizon
    )

    updated = {eid: base_eligibility.get(eid, "ADMIT") for eid in visible_ids}
    updated.update(sub_elig)
    work_stats = {
        "visited_nodes": len(affected),
        "total_nodes": len(visible_ids),
        "visited_edges": visited_edges,
        "total_edges": len(vis_edges),
    }
    return updated, work_stats


# ---------------------------------------------------------------------------
# EK5 v2: Exact 3-Valued Well-Founded Semantics (WFS) & Persistent O(|D|+|E(D)|) Store
# ---------------------------------------------------------------------------
def _wfs_propagate_local(
    dirty_nodes: Set[str],
    attackers: Mapping[str, Set[str]],
    attacked_by: Mapping[str, Set[str]],
    deps_of: Mapping[str, Set[str]],
    dependents_of: Mapping[str, Set[str]],
    frozen_state: Mapping[str, str],
) -> Tuple[Dict[str, str], int]:
    """Exact queue-driven 3-valued Well-Founded Semantics on `dirty_nodes` with frozen boundary.

    States: `"ADMIT"` (grounded true), `"AUDIT_ONLY"` (grounded refuted/invalidated),
    `"CONTESTED"` (ungrounded / unresolved mutual refutation or circular dependency).
    Guaranteed monotone, cycle-safe, horizon-free, and linear O(|D| + |E(D)|) in dirty cone size.
    """
    non_f_att: Dict[str, int] = {}
    non_t_dep: Dict[str, int] = {}
    loc_state: Dict[str, str] = {}
    q: deque[Tuple[str, str]] = deque()
    touched_edges = 0

    for v in dirty_nodes:
        has_t_att = False
        has_f_dep = False
        nf_att = 0
        for u in attackers.get(v, ()):
            touched_edges += 1
            if u in dirty_nodes:
                nf_att += 1
            else:
                st_u = frozen_state[u]
                if st_u == "ADMIT":
                    has_t_att = True
                if st_u != "AUDIT_ONLY":
                    nf_att += 1
        nt_dep = 0
        for u in deps_of.get(v, ()):
            touched_edges += 1
            if u in dirty_nodes:
                nt_dep += 1
            else:
                st_u = frozen_state[u]
                if st_u == "AUDIT_ONLY":
                    has_f_dep = True
                if st_u != "ADMIT":
                    nt_dep += 1

        non_f_att[v] = nf_att
        non_t_dep[v] = nt_dep
        if has_t_att or has_f_dep:
            q.append((v, "AUDIT_ONLY"))
        elif nf_att == 0 and nt_dep == 0:
            q.append((v, "ADMIT"))

    while q:
        u, st = q.popleft()
        if u in loc_state:
            continue
        loc_state[u] = st
        if st == "ADMIT":
            for w in attacked_by.get(u, ()):
                touched_edges += 1
                if w in dirty_nodes and w not in loc_state:
                    q.append((w, "AUDIT_ONLY"))
            for w in dependents_of.get(u, ()):
                touched_edges += 1
                if w in dirty_nodes and w not in loc_state:
                    non_t_dep[w] -= 1
                    if non_f_att[w] == 0 and non_t_dep[w] == 0:
                        q.append((w, "ADMIT"))
        else:  # st == "AUDIT_ONLY"
            for w in dependents_of.get(u, ()):
                touched_edges += 1
                if w in dirty_nodes and w not in loc_state:
                    q.append((w, "AUDIT_ONLY"))
            for w in attacked_by.get(u, ()):
                touched_edges += 1
                if w in dirty_nodes and w not in loc_state:
                    non_f_att[w] -= 1
                    if non_f_att[w] == 0 and non_t_dep[w] == 0:
                        q.append((w, "ADMIT"))

    for v in dirty_nodes:
        if v not in loc_state:
            loc_state[v] = "CONTESTED"
    return loc_state, touched_edges


def compute_well_founded_eligibility(
    events: Mapping[str, EvidenceEvent],
    edges: Sequence[EvidenceEdge],
    *,
    valid_cutoff: int,
    tx_cutoff: int,
) -> Dict[str, str]:
    """Global 3-valued Well-Founded Semantics (`ADMIT` | `AUDIT_ONLY` | `CONTESTED`).

    Eliminates finite-horizon H cutoffs and odd/even parity oscillation on mutual refutation cycles.
    """
    visible_ids = {
        eid
        for eid, ev in events.items()
        if ev.transaction_time <= tx_cutoff and ev.valid_time <= valid_cutoff
    }
    attackers: Dict[str, Set[str]] = defaultdict(set)
    attacked_by: Dict[str, Set[str]] = defaultdict(set)
    deps_of: Dict[str, Set[str]] = defaultdict(set)
    dependents_of: Dict[str, Set[str]] = defaultdict(set)

    for e in edges:
        if e.transaction_time <= tx_cutoff and e.src_id in visible_ids and e.dst_id in visible_ids:
            if e.edge_type in ("REFUTE", "SUPERSEDE"):
                attackers[e.dst_id].add(e.src_id)
                attacked_by[e.src_id].add(e.dst_id)
            elif e.edge_type in ("DERIVES_FROM", "COPY_OF"):
                deps_of[e.src_id].add(e.dst_id)
                dependents_of[e.dst_id].add(e.src_id)

    states, _ = _wfs_propagate_local(
        visible_ids, attackers, attacked_by, deps_of, dependents_of, frozen_state={}
    )
    return states


class IncrementalEvidenceStore:
    """EK5 v2 Persistent O(|D| + |E(D)|) Incremental Well-Founded Evidence Store.

    Maintains persistent deduplicated adjacency sets and 3-valued WFS eligibility states
    (`ADMIT` | `AUDIT_ONLY` | `CONTESTED`). Each incremental mutation traverses ONLY the
    forward-reachable dirty cone D from inserted nodes/edge targets using frozen boundary
    states for V \\ D — zero full-graph scans per update, exact bitwise parity with global
    `compute_well_founded_eligibility` on ALL directed graphs (including multi-refuters,
    arbitrary-depth chains, and mutual-refutation cycles).
    """

    def __init__(self) -> None:
        self.nodes: Set[str] = set()
        self.attackers: Dict[str, Set[str]] = defaultdict(set)
        self.attacked_by: Dict[str, Set[str]] = defaultdict(set)
        self.deps_of: Dict[str, Set[str]] = defaultdict(set)
        self.dependents_of: Dict[str, Set[str]] = defaultdict(set)
        self.state: Dict[str, str] = {}
        self.edge_count: int = 0

    def apply_mutation(
        self,
        new_node_ids: Sequence[str] = (),
        new_edges: Sequence[Tuple[str, str, str]] = (),
        *,
        dense_fallback_ratio: Optional[float] = None,
    ) -> Tuple[Dict[str, str], Dict[str, Any]]:
        """Insert nodes and `(src_id, dst_id, edge_type)` edges and update WFS incrementally.

        Returns `(changed_states, work_stats)` measuring real wall-clock latency (`wall_ms`),
        dirty-cone size, and whether the optional dense-cascade circuit breaker fired.
        """
        t0 = time.perf_counter()
        seeds: Set[str] = set()
        for nid in new_node_ids:
            if nid not in self.nodes:
                self.nodes.add(nid)
                self.state[nid] = "ADMIT"
                seeds.add(nid)

        for src, dst, etype in new_edges:
            if src not in self.nodes or dst not in self.nodes:
                continue
            if etype in ("REFUTE", "SUPERSEDE"):
                if src not in self.attackers[dst]:
                    self.attackers[dst].add(src)
                    self.attacked_by[src].add(dst)
                    self.edge_count += 1
                    seeds.add(dst)
            elif etype in ("DERIVES_FROM", "COPY_OF"):
                if dst not in self.deps_of[src]:
                    self.deps_of[src].add(dst)
                    self.dependents_of[dst].add(src)
                    self.edge_count += 1
                    seeds.add(src)

        if not seeds:
            return {}, {
                "dirty_nodes": 0,
                "dirty_edges": 0,
                "total_nodes": len(self.nodes),
                "total_edges": self.edge_count,
                "dirty_node_ratio": 0.0,
                "used_dense_fallback": False,
                "wall_ms": round((time.perf_counter() - t0) * 1000.0, 4),
            }

        # Step 1: Forward-reachable dirty cone D from seeds along attacked_by U dependents_of
        dirty: Set[str] = set(seeds)
        bfs = deque(seeds)
        bfs_edges = 0
        used_fallback = False
        max_dirty = (
            int(dense_fallback_ratio * len(self.nodes))
            if dense_fallback_ratio is not None and len(self.nodes) > 0
            else None
        )
        while bfs:
            u = bfs.popleft()
            for w in self.attacked_by.get(u, ()):
                bfs_edges += 1
                if w not in dirty:
                    dirty.add(w)
                    bfs.append(w)
            for w in self.dependents_of.get(u, ()):
                bfs_edges += 1
                if w not in dirty:
                    dirty.add(w)
                    bfs.append(w)
            if max_dirty is not None and len(dirty) > max_dirty:
                used_fallback = True
                dirty = set(self.nodes)
                break

        # Step 2: Boundary-exact local WFS strictly on `dirty`
        loc_states, prop_edges = _wfs_propagate_local(
            dirty, self.attackers, self.attacked_by, self.deps_of, self.dependents_of, self.state
        )
        changed: Dict[str, str] = {}
        for v, new_st in loc_states.items():
            if self.state.get(v) != new_st:
                self.state[v] = new_st
                changed[v] = new_st
        wall_ms = (time.perf_counter() - t0) * 1000.0
        return changed, {
            "dirty_nodes": len(dirty),
            "dirty_edges": bfs_edges + prop_edges,
            "total_nodes": len(self.nodes),
            "total_edges": self.edge_count,
            "dirty_node_ratio": len(dirty) / float(max(len(self.nodes), 1)),
            "used_dense_fallback": used_fallback,
            "wall_ms": round(wall_ms, 4),
        }

    def full_recompute_all(self) -> Tuple[Dict[str, str], float]:
        """Reference full-graph WFS recomputation for parity & wall-clock speedup benchmarking."""
        t0 = time.perf_counter()
        states, _ = _wfs_propagate_local(
            self.nodes, self.attackers, self.attacked_by, self.deps_of, self.dependents_of, {}
        )
        return states, (time.perf_counter() - t0) * 1000.0


# ---------------------------------------------------------------------------
# Six Comparators + Frozen Common Deterministic Answer Selector
# ---------------------------------------------------------------------------
def _select_answer_weighted_vote(
    ranked_events: Sequence[Tuple[str, float]],
    events: Mapping[str, EvidenceEvent],
    *,
    top_k: int = 5,
    exclude_roles: Set[str] = frozenset({"RETRACTION"}),
    min_score: float = 1e-6,
) -> Tuple[str, List[str]]:
    """Frozen common deterministic answer selector (no gold labels, no oracle lineage).

    Aggregates weights over top-k admitted messages by `claim_value` (skipping pure RETRACTION notices
    that restate the retracted value without asserting a new claim).
    """
    top = list(ranked_events[:top_k])
    votes: Dict[str, float] = defaultdict(float)
    supporting: List[str] = []
    for eid, score in top:
        ev = events[eid]
        if ev.role in exclude_roles:
            continue
        if score < min_score:
            continue
        votes[ev.claim_value] += max(score, 0.0)
        supporting.append(eid)
    if not votes:
        return "ABSTAIN", []
    best_val = max(sorted(votes.keys()), key=lambda v: votes[v])
    return best_val, supporting


def _token_jaccard(text_a: str, text_b: str) -> float:
    sa = set(_TOK_RE.findall(text_a.lower()))
    sb = set(_TOK_RE.findall(text_b.lower()))
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / float(len(sa | sb))


def evaluate_arm_on_query(
    arm_name: str,
    query: LifecycleQuery,
    lc_event_ids: Sequence[str],
    lc_edges: Sequence[EvidenceEdge],
    events: Mapping[str, EvidenceEvent],
    bm25: ScopedBM25,
    *,
    scalemas_gamma: float = 2.0,
    scalemas_decay: float = 0.90,
    top_k: int = 5,
) -> Dict[str, Any]:
    """Run one comparator arm on one query under exact matched observable inputs."""
    t0 = time.perf_counter()

    # Observable events at cutoff (NOTE: C0/C1/C2 only filter by transaction_time <= tx_cutoff;
    # C3/C4 enforce full bitemporal valid_time <= valid_cutoff & transaction_time <= tx_cutoff!)
    tx_visible = [eid for eid in lc_event_ids if events[eid].transaction_time <= query.tx_cutoff]
    bitemp_visible = [
        eid for eid in tx_visible if events[eid].valid_time <= query.valid_cutoff
    ]
    vis_edges = [
        e for e in lc_edges
        if e.transaction_time <= query.tx_cutoff and e.src_id in set(tx_visible) and e.dst_id in set(tx_visible)
    ]

    bm25_scores = bm25.score(query.query_text, tx_visible)

    if arm_name == "C0_bm25_flat":
        ranked = sorted(
            ((eid, bm25_scores[eid]) for eid in tx_visible if events[eid].slot == query.slot),
            key=lambda x: (-x[1], x[0]),
        )
        ans, supp = _select_answer_weighted_vote(ranked, events, top_k=top_k)

    elif arm_name == "C1_scalemas_unsigned":
        # SCALEMAS_ADAPTED_SURROGATE (arXiv:2610.16120v1 §§2.3, C.4):
        # Unsigned decaying citation boost — refutations/supersessions ALSO add +gamma * alpha^(dt)!
        cite_boost: Dict[str, float] = defaultdict(float)
        for e in vis_edges:
            dt = max(0, query.tx_cutoff - e.transaction_time)
            cite_boost[e.dst_id] += scalemas_gamma * (scalemas_decay ** dt)
        ranked = sorted(
            (
                (eid, bm25_scores[eid] + cite_boost[eid])
                for eid in tx_visible
                if events[eid].slot == query.slot
            ),
            key=lambda x: (-x[1], x[0]),
        )
        ans, supp = _select_answer_weighted_vote(ranked, events, top_k=top_k)

    elif arm_name == "C2_recency_version_rag":
        # Ordinary recency/version-aware RAG: boost newer transaction_time without graph invalidation
        ranked = sorted(
            (
                (eid, bm25_scores[eid] * (1.0 + 0.15 * events[eid].transaction_time))
                for eid in bitemp_visible
                if events[eid].slot == query.slot and events[eid].role != "RETRACTION"
            ),
            key=lambda x: (-x[1], -events[x[0]].transaction_time, x[0]),
        )
        ans, supp = _select_answer_weighted_vote(ranked, events, top_k=top_k)

    elif arm_name == "C3_ordinary_bitemporal_tms":
        # Strongest ordinary composite: bitemporal cutoff + declared-edge ATMS + declared COPY_OF dedup
        elig = compute_full_recompute_eligibility(
            {eid: events[eid] for eid in bitemp_visible},
            vis_edges,
            valid_cutoff=query.valid_cutoff,
            tx_cutoff=query.tx_cutoff,
            horizon=8,
        )
        admitted = [
            eid for eid in bitemp_visible
            if elig.get(eid) == "ADMIT" and events[eid].slot == query.slot and events[eid].role != "RETRACTION"
        ]
        # Give slight recency tiebreak on admitted items
        ranked = sorted(
            ((eid, bm25_scores[eid] + 0.1 * events[eid].transaction_time) for eid in admitted),
            key=lambda x: (-x[1], x[0]),
        )
        ans, supp = _select_answer_weighted_vote(ranked, events, top_k=top_k)

    elif arm_name == "C4_iq04_candidate":
        # IQ-04 Survivor-Only Candidate (STRICTLY OBSERVABLE INPUTS ONLY — ZERO ORACLE LINEAGE!):
        # 1) Exact bitemporal + cycle-safe signed eligibility (same graph basis as C3)
        elig = compute_full_recompute_eligibility(
            {eid: events[eid] for eid in bitemp_visible},
            vis_edges,
            valid_cutoff=query.valid_cutoff,
            tx_cutoff=query.tx_cutoff,
            horizon=8,
        )
        # 2) Observable unlinked-copy / retracted-claim propagation:
        #    Identify values explicitly invalidated on this (entity, slot) by an active REFUTE/SUPERSEDE
        #    (and NOT reinstated by a later active REINSTATEMENT!).
        #    Any earlier unlinked message asserting that exact retracted value (or paraphrasing the
        #    retracted root before the retraction timestamp) is moved to AUDIT_ONLY (never erased!).
        retracted_events = [
            events[eid] for eid in bitemp_visible
            if elig.get(eid) == "AUDIT_ONLY" and events[eid].slot == query.slot
        ]
        # Map retracted claim_value -> latest retraction/correction timestamp invalidating it
        retracted_val_cutoff: Dict[str, int] = {}
        for e in vis_edges:
            if e.edge_type in ("REFUTE", "SUPERSEDE") and elig.get(e.src_id) == "ADMIT":
                tgt_ev = events[e.dst_id]
                if tgt_ev.slot == query.slot and elig.get(e.dst_id) == "AUDIT_ONLY":
                    retracted_val_cutoff[tgt_ev.claim_value] = max(
                        retracted_val_cutoff.get(tgt_ev.claim_value, -1),
                        e.transaction_time,
                    )

        admitted = []
        for eid in bitemp_visible:
            ev = events[eid]
            if ev.slot != query.slot or ev.role == "RETRACTION":
                continue
            if elig.get(eid) != "ADMIT":
                continue
            # Observable unlinked-copy check: if this claim's value was actively retracted at t_ret >= ev.tx_time,
            # and either has high lexical overlap with a retracted root or occurred within the pre-retraction burst
            # without being an explicit REINSTATEMENT, demote to AUDIT_ONLY!
            t_ret = retracted_val_cutoff.get(ev.claim_value, -1)
            if t_ret >= ev.transaction_time and ev.role != "REINSTATEMENT":
                has_lexical_echo = any(_token_jaccard(ev.text, rev.text) >= 0.25 for rev in retracted_events)
                if has_lexical_echo or ev.transaction_time < t_ret:
                    continue
            admitted.append(eid)

        # 3) Copy-aware corroboration dampening among remaining admitted items:
        #    Dampen near-duplicate burst paraphrases sharing identical (entity, slot, claim_value)
        #    within a tight window so independent corroboration beats echoed copies, while
        #    preferring authoritative CORRECTION / REINSTATEMENT records over stale notes.
        val_counts: Counter[str] = Counter()
        ranked_pairs: List[Tuple[str, float]] = []
        for eid in sorted(admitted, key=lambda x: (events[x].transaction_time, x)):
            ev = events[eid]
            c_seen = val_counts[ev.claim_value]
            val_counts[ev.claim_value] += 1
            # Diminishing marginal weight 1 / (1 + c_seen) prevents copy-dependence log-odds blowup
            copy_damp = 1.0 / float(1 + c_seen)
            role_bonus = 2.0 if ev.role in ("CORRECTION", "REINSTATEMENT") else 1.0
            score = (bm25_scores[eid] + 0.1 * ev.transaction_time) * copy_damp * role_bonus
            ranked_pairs.append((eid, score))
        ranked = sorted(ranked_pairs, key=lambda x: (-x[1], x[0]))
        ans, supp = _select_answer_weighted_vote(ranked, events, top_k=top_k)

    elif arm_name == "C5_oracle_upper_bound":
        # Evaluator-only oracle upper bound (reads hidden_is_true_at_query — NEVER deployable!)
        admitted = [
            eid for eid in bitemp_visible
            if events[eid].slot == query.slot
            and events[eid].role != "RETRACTION"
            and events[eid].hidden_is_true_at_query
        ]
        ranked = sorted(((eid, bm25_scores[eid] + 1.0) for eid in admitted), key=lambda x: (-x[1], x[0]))
        ans, supp = _select_answer_weighted_vote(ranked, events, top_k=top_k)

    else:
        raise ValueError(f"Unknown arm: {arm_name!r}")

    latency_ms = (time.perf_counter() - t0) * 1000.0
    is_correct = (ans == query.gold_answer)
    is_abstain = (ans == "ABSTAIN")
    is_stale_or_false = (not is_correct) and (not is_abstain)

    return {
        "query_id": query.query_id,
        "lifecycle_id": query.lifecycle_id,
        "case_family": query.case_family,
        "arm": arm_name,
        "selected_answer": ans,
        "gold_answer": query.gold_answer,
        "supporting_ids": supp,
        "top_context_ids": [eid for eid, _ in ranked[:top_k]],
        "is_correct": is_correct,
        "is_abstain": is_abstain,
        "is_stale_or_false": is_stale_or_false,
        "is_post_correction": query.is_post_correction,
        "is_clean_control": query.is_clean_control,
        "latency_ms": round(latency_ms, 4),
    }


__all__ = [
    "CASE_FAMILIES",
    "EvidenceEdge",
    "EvidenceEvent",
    "IncrementalEvidenceStore",
    "LifecycleCorpus",
    "LifecycleQuery",
    "ScopedBM25",
    "compute_full_recompute_eligibility",
    "compute_well_founded_eligibility",
    "evaluate_arm_on_query",
    "generate_iq04_lifecycle_corpus",
    "sparse_invalidate_incremental",
]
