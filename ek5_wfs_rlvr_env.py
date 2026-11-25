"""EK5-WFS-RLVR: a procedural, 3-valued, verifiable-reward environment backed by EK5 v3.

Every episode is a small multi-agent evidence log (tiers 1-4, domains, REFUTE / SUPERSEDE /
COPY_OF / DERIVES_CURRENT / CITES_AUDIT / SUPPORT relations, optional bitemporal as-of cutoff,
optional edge withdrawal) about one target slot ``entity.slot``.  A proposed answer (the claim
of one "query" event) is shown to the policy, which must answer with

    VERDICT: T | F | U
    ANSWER: <unique current value> | ABSTAIN

Gold labels are produced by replaying the episode's mutation plan through
``StratifiedEvidenceStore`` (incrementally, in randomised batches) and reading the slot:

* ``T`` - the slot resolves to a unique current value and the proposal is that value;
* ``F`` - the slot resolves, but the proposal is a refuted / superseded / laundered trap;
* ``U`` - the slot is unresolved (symmetric peer deadlock, odd refutation cycle, contested
  successor, silent multi-current, out-of-domain stand-off) and the only correct answer is
  ``ABSTAIN``.

Each label is cross-checked three ways: (1) the store's Stratum-2 values against
``compute_alternating_wfs_oracle`` on the store's own authority-qualified graph, (2) a fully
independent re-implementation of the authority ceiling, edge qualification, WFS (via the
oracle), currency and slot resolution computed from the raw final attestations, and (3) the
generator's intended label.  ``run_oracle_audit`` reports all mismatch counts.

``score_epistemic_completion`` is the 3-valued reward; ``binary_exact_match_reward`` and
``binary_verdict_reward`` are the 2-valued RLVR baselines (U collapsed to last-writer-wins /
closed-world F).  ``ek5_wfs_reward_func`` is a TRL ``GRPOTrainer``-compatible wrapper.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple, Union

from ek5_v3_stratified import (
    DEP_ALL_TYPES,
    DEP_CUR_TYPES,
    EdgeAttestation,
    StratifiedEvent,
    StratifiedEvidenceStore,
    compute_alternating_wfs_oracle,
)

LABEL_T, LABEL_F, LABEL_U = "T", "F", "U"
ABSTAIN = "ABSTAIN"
TENANT = "rlvr"
DEFAULT_AUDIT_SEED = 20261122

# 3-valued reward components (max 1.0 only for the exactly correct verdict + answer).
REWARD_CORRECT_ANSWER = 0.75
REWARD_CORRECT_VERDICT = 0.25
REWARD_ABSTAIN_ON_DETERMINATE = 0.0
REWARD_COMMIT_WRONG = -1.0
REWARD_MALFORMED = -1.0

SLOT_POOL: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    ("payments-api", "smtp_relay", ("mx1.corp.example", "mx2.corp.example", "relay.vendor-mail.example",
                                    "smtp.backup.example", "mx-eu.corp.example", "relay.attacker-cdn.example")),
    ("billing", "wire_recipient", ("ACME-OPS-0042", "ACME-OPS-0077", "VENDOR-ESCROW-19", "OFFSHORE-HOLD-88",
                                   "ACME-TREASURY-01", "SUPPLIER-PAY-31")),
    ("edge-proxy", "tls_min_version", ("TLS1.3", "TLS1.2", "TLS1.1", "TLS1.0", "SSL3.0", "TLS1.3-PQ")),
    ("cve-2026-1187", "patch_policy", ("patch-within-24h", "patch-within-72h", "defer-to-quarter",
                                       "risk-accepted", "mitigate-waf-only", "emergency-rollback")),
    ("db-primary", "max_connections", ("400", "800", "1200", "200", "1600", "100")),
    ("vpn-gateway", "mfa_mode", ("fido2-required", "totp-allowed", "sms-fallback", "disabled", "push-only",
                                 "fido2-or-totp")),
    ("ledger", "release_approver", ("cfo-office", "controller-team", "treasury-lead", "external-agent",
                                    "audit-committee", "ops-oncall")),
    ("k8s-prod", "image_registry", ("registry.corp.example", "ghcr.io/corp", "mirror.corp.example",
                                    "registry.typosquat.example", "ecr.corp.example", "quay.io/corp")),
)

ROLES: Mapping[int, Tuple[str, ...]] = {
    1: ("external-web-scraper", "ticket-ingest-bot"),
    2: ("triage-summarizer", "chat-digest-agent"),
    3: ("sre-operator", "finance-operator"),
    4: ("security-auditor", "compliance-auditor"),
}


# ---------------------------------------------------------------------------
# Episode container
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EpistemicEpisode:
    episode_id: str
    seed: str
    family: str
    scenario: str
    intended_label: str
    entity: str
    slot: str
    events: Tuple[StratifiedEvent, ...]
    attestations: Tuple[EdgeAttestation, ...]
    roles: Tuple[Tuple[str, str], ...]
    mutation_plan: Tuple[Dict[str, Tuple[str, ...]], ...]
    withdrawn_attestation_ids: Tuple[str, ...]
    as_of: Optional[Tuple[int, int]]
    exact_min_rank: bool
    query_event_id: str
    proposed_value: str
    gold_label: str
    gold_value: Optional[str]
    binary_gold_value: str
    candidate_values: Tuple[str, ...]
    store_slot_status: str
    store_slot_reason: str
    store_query_reason: str
    prompt: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "episode_id": self.episode_id,
            "seed": self.seed,
            "family": self.family,
            "scenario": self.scenario,
            "intended_label": self.intended_label,
            "entity": self.entity,
            "slot": self.slot,
            "events": [ev.__dict__.copy() for ev in self.events],
            "attestations": [att.__dict__.copy() for att in self.attestations],
            "roles": [list(r) for r in self.roles],
            "mutation_plan": [{k: list(v) for k, v in batch.items()} for batch in self.mutation_plan],
            "withdrawn_attestation_ids": list(self.withdrawn_attestation_ids),
            "as_of": list(self.as_of) if self.as_of is not None else None,
            "exact_min_rank": self.exact_min_rank,
            "query_event_id": self.query_event_id,
            "proposed_value": self.proposed_value,
            "gold_label": self.gold_label,
            "gold_value": self.gold_value,
            "binary_gold_value": self.binary_gold_value,
            "candidate_values": list(self.candidate_values),
            "store_slot_status": self.store_slot_status,
            "store_slot_reason": self.store_slot_reason,
            "store_query_reason": self.store_query_reason,
            "prompt": self.prompt,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "EpistemicEpisode":
        return cls(
            episode_id=d["episode_id"],
            seed=d["seed"],
            family=d["family"],
            scenario=d["scenario"],
            intended_label=d["intended_label"],
            entity=d["entity"],
            slot=d["slot"],
            events=tuple(StratifiedEvent(**e) for e in d["events"]),
            attestations=tuple(EdgeAttestation(**a) for a in d["attestations"]),
            roles=tuple((r[0], r[1]) for r in d["roles"]),
            mutation_plan=tuple({k: tuple(v) for k, v in b.items()} for b in d["mutation_plan"]),
            withdrawn_attestation_ids=tuple(d["withdrawn_attestation_ids"]),
            as_of=tuple(d["as_of"]) if d.get("as_of") is not None else None,  # type: ignore[arg-type]
            exact_min_rank=bool(d["exact_min_rank"]),
            query_event_id=d["query_event_id"],
            proposed_value=d["proposed_value"],
            gold_label=d["gold_label"],
            gold_value=d["gold_value"],
            binary_gold_value=d["binary_gold_value"],
            candidate_values=tuple(d["candidate_values"]),
            store_slot_status=d["store_slot_status"],
            store_slot_reason=d["store_slot_reason"],
            store_query_reason=d["store_query_reason"],
            prompt=d["prompt"],
        )


EpisodeLike = Union[EpistemicEpisode, Mapping[str, Any], str]


def _coerce_episode(episode: EpisodeLike) -> EpistemicEpisode:
    if isinstance(episode, EpistemicEpisode):
        return episode
    if isinstance(episode, str):
        return EpistemicEpisode.from_dict(json.loads(episode))
    return EpistemicEpisode.from_dict(episode)


# ---------------------------------------------------------------------------
# Procedural builder
# ---------------------------------------------------------------------------
class _Builder:
    def __init__(self, rng: random.Random) -> None:
        self.rng = rng
        self.events: List[StratifiedEvent] = []
        self.atts: List[EdgeAttestation] = []
        self.roles: Dict[str, str] = {}
        self.by_id: Dict[str, StratifiedEvent] = {}
        self.tx = 0
        self._n_ev = 0
        self._n_att = 0

    def add_event(self, entity: str, slot: str, value: str, tier: int, domain: str, *,
                  tx: Optional[int] = None, role: Optional[str] = None) -> str:
        self._n_ev += 1
        eid = f"m{self._n_ev:03d}"
        if tx is None:
            self.tx += 1
            tx = self.tx
        else:
            self.tx = max(self.tx, tx)
        role = role or self.rng.choice(ROLES[tier])
        ev = StratifiedEvent(
            event_id=eid, entity=entity, slot=slot, claim_value=value,
            text=f"{role} (tier {tier}, domain {domain}) reports {entity}.{slot} = {value}",
            valid_time=tx, tx_time=tx, tenant=TENANT, domain=domain, authority_tier=tier,
        )
        self.events.append(ev)
        self.by_id[eid] = ev
        self.roles[eid] = role
        return eid

    def add_edge(self, src: str, dst: str, etype: str) -> str:
        # Edge time = max endpoint time, so a visible dependency edge never has an invisible endpoint.
        tx = max(self.by_id[src].tx_time, self.by_id[dst].tx_time)
        self._n_att += 1
        aid = f"a{self._n_att:03d}"
        self.atts.append(EdgeAttestation(attestation_id=aid, src_id=src, dst_id=dst, edge_type=etype,
                                         valid_time=tx, tx_time=tx))
        return aid


class _SlotCtx:
    def __init__(self, b: _Builder, entity: str, slot: str, values: Sequence[str], domain: str) -> None:
        self.b = b
        self.entity = entity
        self.slot = slot
        self.values = list(values)
        b.rng.shuffle(self.values)
        self.domain = domain
        self.as_of: Optional[Tuple[int, int]] = None

    def take(self, k: int) -> List[str]:
        out, self.values = self.values[:k], self.values[k:]
        if len(out) < k:
            raise RuntimeError("value pool exhausted")
        return out

    def ev(self, value: str, tier: int, *, domain: Optional[str] = None, tx: Optional[int] = None,
           role: Optional[str] = None) -> str:
        return self.b.add_event(self.entity, self.slot, value, tier, domain or self.domain, tx=tx, role=role)


# Each scenario returns the query event id for the requested label.  Scenarios only use
# acyclic SUPERSEDE graphs (the store runs with strict_dag_supersede=True).
def _sc_conflict_refute(s: _SlotCtx, want: str) -> str:
    rng, b = s.b.rng, s.b
    truth_v, trap_v = s.take(2)
    trap_tier, truth_tier = rng.choice((1, 2)), rng.choice((3, 4))
    if rng.random() < 0.5:
        trap = s.ev(trap_v, trap_tier)
        truth = s.ev(truth_v, truth_tier)
    else:
        truth = s.ev(truth_v, truth_tier)
        trap = s.ev(trap_v, trap_tier)
    b.add_edge(truth, trap, "REFUTE")
    if rng.random() < 0.6:
        b.add_edge(trap, truth, "REFUTE")  # authority-blocked counter-attack
    if rng.random() < 0.3:
        b.add_edge(trap, truth, "SUPERSEDE")  # authority-blocked takeover attempt
    return truth if want == LABEL_T else trap


def _sc_supersede_chain(s: _SlotCtx, want: str) -> str:
    rng, b = s.b.rng, s.b
    k = rng.randint(2, 4)
    vals = s.take(k)
    tier = rng.choice((2, 3))
    ids: List[str] = []
    for i, v in enumerate(vals):
        if i:
            tier = min(4, tier + rng.choice((0, 0, 1)))  # nondecreasing: every SUPERSEDE is authority-qualified
        ids.append(s.ev(v, tier))
        if i:
            b.add_edge(ids[i], ids[i - 1], "SUPERSEDE")
        if i >= 2 and rng.random() < 0.3:
            b.add_edge(ids[i], ids[i - 2], "SUPERSEDE")
    if rng.random() < 0.5:
        atk = s.ev(s.take(1)[0], 1)
        b.add_edge(atk, ids[-1], "REFUTE")  # blocked
        b.add_edge(ids[-1], atk, "REFUTE")
    return ids[-1] if want == LABEL_T else rng.choice(ids[:-1])


def _sc_auditor_cites_poison(s: _SlotCtx, want: str) -> str:
    rng, b = s.b.rng, s.b
    truth_v, trap_v = s.take(2)
    poison = s.ev(trap_v, 1, role="ticket-ingest-bot")
    auditor = s.ev(truth_v, 4, domain="general", role="security-auditor")
    b.add_edge(auditor, poison, "CITES_AUDIT")
    b.add_edge(auditor, poison, "REFUTE")
    if rng.random() < 0.5:
        b.add_edge(poison, auditor, "REFUTE")  # blocked
    query_t = auditor
    if rng.random() < 0.5:
        op = s.ev(truth_v, 3, role="sre-operator")
        b.add_edge(op, auditor, "DERIVES_CURRENT")
        query_t = rng.choice((auditor, op))
    return query_t if want == LABEL_T else poison


def _sc_sybil(s: _SlotCtx, want: str) -> str:
    rng, b = s.b.rng, s.b
    truth_v, trap_v = s.take(2)
    root = s.ev(trap_v, 1, role="external-web-scraper")
    n = rng.randint(3, 10)
    declared = rng.random() < 0.5
    truth = s.ev(truth_v, rng.choice((3, 4)))
    copies: List[str] = []
    for _ in range(n):
        c = s.ev(trap_v, rng.choice((1, 1, 2)))
        copies.append(c)
        if declared:
            b.add_edge(c, root, "COPY_OF")
        else:
            b.add_edge(truth, c, "REFUTE")
        if rng.random() < 0.5:
            b.add_edge(c, truth, "REFUTE")  # blocked
        if copies[:-1] and rng.random() < 0.5:
            b.add_edge(c, rng.choice(copies[:-1]), "SUPPORT")
    b.add_edge(truth, root, "REFUTE")
    if rng.random() < 0.3:
        x, y = s.ev(trap_v, 1), s.ev(trap_v, 1)
        b.add_edge(x, y, "COPY_OF")
        b.add_edge(y, x, "COPY_OF")  # circular corroboration -> unfounded
    return truth if want == LABEL_T else rng.choice(copies + [root])


def _sc_laundered(s: _SlotCtx, want: str) -> str:
    rng, b = s.b.rng, s.b
    truth_v, trap_v = s.take(2)
    poison = s.ev(trap_v, 1, role="external-web-scraper")
    summ = s.ev(trap_v, 2, role="triage-summarizer")
    b.add_edge(summ, poison, "DERIVES_CURRENT")
    if rng.random() < 0.5:
        truth = s.ev(truth_v, 3)
        b.add_edge(truth, poison, "REFUTE")
    else:
        truth = s.ev(truth_v, rng.choice((2, 3)))
        b.add_edge(truth, poison, "SUPERSEDE")
    if rng.random() < 0.6:
        b.add_edge(summ, truth, "REFUTE")  # blocked: summarizer's ceiling is tier 1
    if rng.random() < 0.3:
        b.add_edge(summ, truth, "SUPERSEDE")  # blocked
    return truth if want == LABEL_T else summ


def _sc_asof_supersede(s: _SlotCtx, want: str) -> str:
    rng, b = s.b.rng, s.b
    old_v, new_v = s.take(2)
    tier = rng.choice((2, 3))
    t_old = b.tx + 1
    old = s.ev(old_v, tier, tx=t_old)
    t_new = t_old + rng.randint(3, 6)
    new = s.ev(new_v, min(4, tier + rng.choice((0, 1))), tx=t_new)
    b.add_edge(new, old, "SUPERSEDE")
    cut = rng.randint(t_old, t_new - 1) if want == LABEL_T else rng.randint(t_new, t_new + 3)
    s.as_of = (cut, cut)
    return old


def _sc_peer_deadlock(s: _SlotCtx, want: str) -> str:
    rng, b = s.b.rng, s.b
    v1, v2 = s.take(2)
    tier = rng.choice((1, 2, 3, 4))
    p1, p2 = s.ev(v1, tier), s.ev(v2, tier)
    b.add_edge(p1, p2, "REFUTE")
    b.add_edge(p2, p1, "REFUTE")
    for _ in range(rng.randint(0, 6)):  # same-value supporters: majority vote must not break the tie
        sup = s.ev(v1, rng.choice((1, 2)))
        b.add_edge(sup, p1, "SUPPORT")
    if tier > 1 and rng.random() < 0.4:
        atk = s.ev(s.take(1)[0], 1)
        b.add_edge(atk, p1, "REFUTE")
        b.add_edge(atk, p2, "REFUTE")
    return rng.choice((p1, p2))


def _sc_odd_cycle(s: _SlotCtx, want: str) -> str:
    rng, b = s.b.rng, s.b
    tier = rng.choice((2, 3, 4))
    ps = [s.ev(v, tier) for v in s.take(3)]
    for i in range(3):
        b.add_edge(ps[i], ps[(i + 1) % 3], "REFUTE")
    return rng.choice(ps)


def _sc_contested_successor(s: _SlotCtx, want: str) -> str:
    rng, b = s.b.rng, s.b
    v0, v1, v2 = s.take(3)
    tier = rng.choice((2, 3))
    old = s.ev(v0, tier)
    p1, p2 = s.ev(v1, tier), s.ev(v2, tier)
    b.add_edge(p1, p2, "REFUTE")
    b.add_edge(p2, p1, "REFUTE")
    b.add_edge(p1, old, "SUPERSEDE")
    b.add_edge(p2, old, "SUPERSEDE")
    return rng.choice((old, p1, p2))


def _sc_silent_multi_current(s: _SlotCtx, want: str) -> str:
    rng, b = s.b.rng, s.b
    k = rng.randint(2, 3)
    tier = rng.choice((1, 2, 3, 4))
    ps = [s.ev(v, tier) for v in s.take(k)]
    if rng.random() < 0.5:
        b.add_edge(ps[0], ps[1], "SUPPORT")
    return rng.choice(ps)


def _sc_cross_domain(s: _SlotCtx, want: str) -> str:
    rng, b = s.b.rng, s.b
    v1, v2 = s.take(2)
    a = s.ev(v1, rng.choice((2, 3, 4)), domain="finance", role="finance-operator")
    c = s.ev(v2, rng.choice((2, 3, 4)), domain="security", role="security-auditor")
    b.add_edge(a, c, "REFUTE")  # blocked: out-of-domain
    b.add_edge(c, a, "REFUTE")  # blocked: out-of-domain
    return rng.choice((a, c))


_SCENARIOS = {
    "conflict_refute": _sc_conflict_refute,
    "supersede_chain": _sc_supersede_chain,
    "auditor_cites_poison": _sc_auditor_cites_poison,
    "sybil_corroboration": _sc_sybil,
    "summarizer_laundering": _sc_laundered,
    "bitemporal_as_of": _sc_asof_supersede,
    "peer_deadlock": _sc_peer_deadlock,
    "odd_refutation_cycle": _sc_odd_cycle,
    "contested_successor": _sc_contested_successor,
    "silent_multi_current": _sc_silent_multi_current,
    "cross_domain_standoff": _sc_cross_domain,
}

FAMILIES: Tuple[Tuple[str, str, str], ...] = tuple(
    [(f"{sc}:{lab}", sc, lab) for sc in ("conflict_refute", "supersede_chain", "auditor_cites_poison",
                                          "sybil_corroboration", "summarizer_laundering", "bitemporal_as_of")
     for lab in (LABEL_T, LABEL_F)]
    + [(f"{sc}:U", sc, LABEL_U) for sc in ("peer_deadlock", "odd_refutation_cycle", "contested_successor",
                                            "silent_multi_current", "cross_domain_standoff")]
)

_DISTRACTOR_SCENARIOS = ("conflict_refute", "supersede_chain", "sybil_corroboration", "summarizer_laundering",
                         "peer_deadlock", "odd_refutation_cycle", "contested_successor", "silent_multi_current")


def _add_distractors(b: _Builder, used_slots: Set[Tuple[str, str]]) -> None:
    rng = b.rng
    for _ in range(rng.randint(0, 3)):
        free = [p for p in SLOT_POOL if (p[0], p[1]) not in used_slots]
        if not free:
            return
        entity, slot, vals = rng.choice(free)
        used_slots.add((entity, slot))
        ctx = _SlotCtx(b, entity, slot, vals, rng.choice(("general", "ops")))
        _SCENARIOS[rng.choice(_DISTRACTOR_SCENARIOS)](ctx, rng.choice((LABEL_T, LABEL_F, LABEL_U)))
        if rng.random() < 0.3:  # positive-loop noise: unfounded circular derivation
            x = ctx.b.add_event(entity, slot, rng.choice(vals), 1, ctx.domain)
            if rng.random() < 0.5:
                b.add_edge(x, x, "DERIVES_CURRENT")
            else:
                y = ctx.b.add_event(entity, slot, rng.choice(vals), 1, ctx.domain)
                b.add_edge(x, y, "COPY_OF")
                b.add_edge(y, x, "DERIVES_CURRENT")


def _mutation_plan(b: _Builder, withdraw_ids: Sequence[str]) -> Tuple[Dict[str, Tuple[str, ...]], ...]:
    rng = b.rng
    evs = sorted(b.events, key=lambda e: (e.tx_time, e.event_id))
    n_batches = rng.randint(1, min(4, len(evs)))
    cuts = sorted(rng.sample(range(1, len(evs)), n_batches - 1)) if n_batches > 1 else []
    batch_of: Dict[str, int] = {}
    bounds = [0] + cuts + [len(evs)]
    for bi in range(n_batches):
        for ev in evs[bounds[bi]:bounds[bi + 1]]:
            batch_of[ev.event_id] = bi
    plan: List[Dict[str, List[str]]] = [{"event_ids": [], "attestation_ids": [], "withdraw_ids": []}
                                        for _ in range(n_batches)]
    for ev in evs:
        plan[batch_of[ev.event_id]]["event_ids"].append(ev.event_id)
    for att in b.atts:
        lo = max(batch_of[att.src_id], batch_of[att.dst_id])
        plan[min(n_batches - 1, lo + rng.choice((0, 0, 1)))]["attestation_ids"].append(att.attestation_id)
    if withdraw_ids:
        plan.append({"event_ids": [], "attestation_ids": [], "withdraw_ids": list(withdraw_ids)})
    return tuple({k: tuple(v) for k, v in batch.items()} for batch in plan)


# ---------------------------------------------------------------------------
# Store replay + store-derived labels
# ---------------------------------------------------------------------------
def build_episode_store(ep: EpistemicEpisode) -> Tuple[StratifiedEvidenceStore, StratifiedEvidenceStore]:
    """Replay the mutation plan incrementally; return (full_store, view) where view is the as-of snapshot."""
    store = StratifiedEvidenceStore(exact_min_rank=ep.exact_min_rank)
    ev_by_id = {e.event_id: e for e in ep.events}
    att_by_id = {a.attestation_id: a for a in ep.attestations}
    for batch in ep.mutation_plan:
        store.apply_mutation(
            [ev_by_id[i] for i in batch["event_ids"]],
            [att_by_id[i] for i in batch["attestation_ids"]],
            list(batch["withdraw_ids"]),
        )
    view = store.as_of(*ep.as_of) if ep.as_of is not None else store
    return store, view


def _label_from_slot(resolution: Mapping[str, Any], query_elig: str) -> Tuple[Optional[str], Optional[str]]:
    status = resolution["status"]
    if status == "CONTESTED":
        return LABEL_U, None
    if status == "ADMIT":
        return (LABEL_T if query_elig == "ADMIT" else LABEL_F), resolution["claim_value"]
    return None, None


def _lww_value(events: Iterable[StratifiedEvent], ids: Iterable[str]) -> str:
    by_id = {e.event_id: e for e in events}
    best = max(ids, key=lambda i: (by_id[i].tx_time, i))
    return by_id[best].claim_value


# ---------------------------------------------------------------------------
# Independent reference pipeline (no store internals; raw final attestations only)
# ---------------------------------------------------------------------------
def _visible(ep: EpistemicEpisode) -> Tuple[List[StratifiedEvent], List[EdgeAttestation]]:
    withdrawn = set(ep.withdrawn_attestation_ids)
    atts = [a for a in ep.attestations if a.attestation_id not in withdrawn]
    evs = list(ep.events)
    if ep.as_of is not None:
        vc, tc = ep.as_of
        evs = [e for e in evs if e.tx_time <= tc and e.valid_time <= vc and (e.valid_end is None or vc < e.valid_end)]
        ids = {e.event_id for e in evs}
        atts = [a for a in atts if a.tx_time <= tc and a.valid_time <= vc and a.src_id in ids and a.dst_id in ids]
    return evs, atts


def _sccs_kosaraju(nodes: Sequence[str], adj: Mapping[str, Set[str]]) -> List[List[str]]:
    seen: Set[str] = set()
    order: List[str] = []
    for s in nodes:
        if s in seen:
            continue
        stack: List[Tuple[str, Iterable[str]]] = [(s, iter(sorted(adj.get(s, ()))))]
        seen.add(s)
        while stack:
            v, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                stack.pop()
                order.append(v)
            elif nxt not in seen:
                seen.add(nxt)
                stack.append((nxt, iter(sorted(adj.get(nxt, ())))))
    radj: Dict[str, Set[str]] = defaultdict(set)
    for v in nodes:
        for w in adj.get(v, ()):
            radj[w].add(v)
    comps: List[List[str]] = []
    assigned: Set[str] = set()
    for s in reversed(order):
        if s in assigned:
            continue
        comp, st = [], [s]
        assigned.add(s)
        while st:
            v = st.pop()
            comp.append(v)
            for w in radj.get(v, ()):
                if w not in assigned:
                    assigned.add(w)
                    st.append(w)
        comps.append(comp)
    return comps


def independent_reference(ep: EpistemicEpisode) -> Dict[str, Any]:
    """Recompute eff_auth / val / cur / eligibility / slot label from raw final attestations."""
    evs, atts = _visible(ep)
    ev = {e.event_id: e for e in evs}
    nodes = sorted(ev)
    deps_v: Dict[str, Set[str]] = defaultdict(set)
    deps_c: Dict[str, Set[str]] = defaultdict(set)
    raw_ref: Dict[str, Set[str]] = defaultdict(set)
    raw_sup: Dict[str, Set[str]] = defaultdict(set)
    for a in atts:
        if a.edge_type in DEP_ALL_TYPES:
            deps_v[a.src_id].add(a.dst_id)
            if a.edge_type in DEP_CUR_TYPES:
                deps_c[a.src_id].add(a.dst_id)
        elif a.edge_type == "REFUTE":
            raw_ref[a.dst_id].add(a.src_id)
        elif a.edge_type == "SUPERSEDE":
            raw_sup[a.dst_id].add(a.src_id)

    cyclic: Set[str] = set()
    for comp in _sccs_kosaraju(nodes, deps_v):
        if len(comp) > 1 or comp[0] in deps_v.get(comp[0], ()):
            cyclic.update(comp)
    eff: Dict[str, int] = {}

    def _eff(u: str, path: Tuple[str, ...] = ()) -> int:
        if u in eff:
            return eff[u]
        if u in cyclic:
            eff[u] = 0
            return 0
        best = ev[u].authority_tier
        for d in deps_v.get(u, ()):
            best = min(best, _eff(d, path + (u,)))
        eff[u] = best
        return best

    for u in nodes:
        _eff(u)

    def _dom_eff(src: str, dst: str) -> int:
        sd, td = ev[src].domain, ev[dst].domain
        if sd != "general" and td != "general" and sd != td:
            return 0
        return eff[src]

    refs_q = {w: {a for a in raw_ref.get(w, ()) if _dom_eff(a, w) >= eff[w]} for w in nodes}
    sup_q = {w: {a for a in raw_sup.get(w, ()) if _dom_eff(a, w) >= eff[w]} for w in nodes}
    val = compute_alternating_wfs_oracle(nodes, refs_q, deps_v)

    cur: Dict[str, str] = {}
    for v in nodes:
        if val[v] == LABEL_F:
            cur[v] = LABEL_F
            continue
        b_ext = LABEL_F
        for j in sup_q[v]:
            if val[j] == LABEL_T:
                b_ext = LABEL_T
            elif val[j] == LABEL_U and b_ext != LABEL_T:
                b_ext = LABEL_U
        if b_ext == LABEL_T:
            cur[v] = LABEL_F
        elif val[v] == LABEL_U or b_ext == LABEL_U:
            cur[v] = LABEL_U
        else:
            cur[v] = LABEL_T
    # D_C currency propagation, parents before children (Kosaraju on reversed edges gives sinks first)
    for comp in reversed(_sccs_kosaraju(nodes, {u: set(deps_c.get(u, ())) for u in nodes})):
        if len(comp) > 1 or comp[0] in deps_c.get(comp[0], ()):
            for u in comp:
                cur[u] = LABEL_F
            continue
        u = comp[0]
        if cur[u] == LABEL_F:
            continue
        for d in deps_c.get(u, ()):
            if cur[d] == LABEL_F:
                cur[u] = LABEL_F
                break
            if cur[d] == LABEL_U and cur[u] == LABEL_T:
                cur[u] = LABEL_U

    elig = {}
    for v in nodes:
        if val[v] == LABEL_T and cur[v] == LABEL_T:
            elig[v] = "ADMIT"
        elif val[v] == LABEL_F or cur[v] == LABEL_F:
            elig[v] = "AUDIT_ONLY"
        else:
            elig[v] = "CONTESTED"

    slot_ids = [v for v in nodes if ev[v].entity == ep.entity and ev[v].slot == ep.slot]
    admitted_vals = sorted({ev[v].claim_value for v in slot_ids if elig[v] == "ADMIT"})
    if any(elig[v] == "CONTESTED" for v in slot_ids) or len(admitted_vals) > 1:
        label, gold = LABEL_U, None
    elif len(admitted_vals) == 1:
        label = LABEL_T if elig.get(ep.query_event_id) == "ADMIT" else LABEL_F
        gold = admitted_vals[0]
    else:
        label, gold = None, None
    return {"eff_auth": eff, "val": val, "cur": cur, "eligibility": elig, "label": label, "gold_value": gold}


# ---------------------------------------------------------------------------
# Prompt rendering
# ---------------------------------------------------------------------------
_RULES = (
    "Rules:\n"
    "1. Effective tier = min(own tier, effective tier of every parent it COPY_OF / DERIVES_CURRENT from). "
    "A claim on a circular derivation has effective tier 0 and is void. CITES_AUDIT never lowers a tier.\n"
    "2. A REFUTES or SUPERSEDES relation only counts if the source's effective tier >= the target's effective "
    "tier AND the domains match (or either side is 'general'). SUPPORTS never counts as evidence.\n"
    "3. A claim is defeated if a counted REFUTES comes from a claim that itself stands; claims that refute each "
    "other at equal tier with no outside resolution stay unresolved.\n"
    "4. A claim is no longer current if a standing claim counted-SUPERSEDES it, or if a parent it copies or "
    "derives from is defeated or no longer current.\n"
    "5. The slot is resolved only if exactly one value survives and no claim on the slot is unresolved.\n"
)


def _render_prompt(ep_events: Sequence[StratifiedEvent], atts: Sequence[EdgeAttestation],
                   roles: Mapping[str, str], entity: str, slot: str, as_of: Optional[Tuple[int, int]],
                   query_id: str, proposed: str) -> str:
    when = f" as of time t={as_of[1]} (ignore every entry and relation logged after t={as_of[1]})" if as_of else ""
    lines = [f"You are an epistemic reader over a multi-agent evidence log. Decide the CURRENT value of "
             f"`{entity}.{slot}`{when}.", "", _RULES, "Evidence log:"]
    for e in sorted(ep_events, key=lambda x: (x.tx_time, x.event_id)):
        lines.append(f"[{e.event_id}] t={e.tx_time} {roles[e.event_id]} tier={e.authority_tier} "
                     f"domain={e.domain}: {e.entity}.{e.slot} = `{e.claim_value}`")
    lines.append("Relations:")
    verb = {"REFUTE": "REFUTES", "SUPERSEDE": "SUPERSEDES", "COPY_OF": "COPY_OF",
            "DERIVES_CURRENT": "DERIVES_CURRENT from", "CITES_AUDIT": "CITES_AUDIT", "SUPPORT": "SUPPORTS"}
    for a in sorted(atts, key=lambda x: (x.tx_time, x.attestation_id)):
        lines.append(f"t={a.tx_time} [{a.src_id}] {verb.get(a.edge_type, a.edge_type)} [{a.dst_id}]")
    lines += [
        "",
        f"Proposed answer: `{proposed}` (from [{query_id}]).",
        "Respond with exactly two lines:",
        "VERDICT: T if the proposed answer is the unique current value; F if the slot is resolved but the "
        "proposed answer is defeated or no longer current; U if the slot is unresolved (U takes precedence).",
        "ANSWER: the unique current value, or ABSTAIN if the slot is unresolved.",
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------
def generate_episode(index: int, seed: int = DEFAULT_AUDIT_SEED, family: Optional[str] = None) -> EpistemicEpisode:
    """Deterministically generate episode ``index`` (family round-robin unless ``family`` is given)."""
    seed_str = f"ek5-wfs-rlvr:{seed}:{index}"
    rng = random.Random(seed_str)
    fam = family or FAMILIES[index % len(FAMILIES)][0]
    fam_map = {f[0]: f for f in FAMILIES}
    if fam not in fam_map:
        raise KeyError(f"unknown family {fam!r}")
    _, scenario, want = fam_map[fam]

    b = _Builder(rng)
    entity, slot, vals = rng.choice(SLOT_POOL)
    used = {(entity, slot)}
    distractors_first = rng.random() < 0.5
    if distractors_first:
        _add_distractors(b, used)
    ctx = _SlotCtx(b, entity, slot, vals, rng.choice(("general", "ops")))
    query = _SCENARIOS[scenario](ctx, want)
    if not distractors_first:
        _add_distractors(b, used)

    withdraw: List[str] = []
    if rng.random() < 0.25:  # a retracted high-tier refutation of the query must leave no trace
        retr = b.add_event("noise-retractor", "notes", "retracted", 4, "general", role="compliance-auditor")
        withdraw.append(b.add_edge(retr, query, "REFUTE"))
    as_of = ctx.as_of
    plan = _mutation_plan(b, withdraw)
    roles = tuple(sorted(b.roles.items()))
    proposed = b.by_id[query].claim_value

    proto = EpistemicEpisode(
        episode_id=f"ek5wfs-{seed}-{index:05d}", seed=seed_str, family=fam, scenario=scenario, intended_label=want,
        entity=entity, slot=slot, events=tuple(b.events), attestations=tuple(b.atts), roles=roles,
        mutation_plan=plan, withdrawn_attestation_ids=tuple(withdraw), as_of=as_of,
        exact_min_rank=rng.random() < 0.5, query_event_id=query, proposed_value=proposed,
        gold_label="", gold_value=None, binary_gold_value="", candidate_values=(), store_slot_status="",
        store_slot_reason="", store_query_reason="", prompt="",
    )
    _, view = build_episode_store(proto)
    res = view.resolve_slot(TENANT, entity, slot)
    label, gold = _label_from_slot(res, view.eligibility.get(query, ""))
    if label is None:
        raise RuntimeError(f"{proto.episode_id}: target slot resolved EMPTY ({res})")
    unresolved_ids = list(res["contested_ids"]) + list(res["admitted_ids"])
    binary_gold = gold if gold is not None else _lww_value(view.events.values(), unresolved_ids)
    cand = tuple(sorted({e.claim_value for e in b.events if e.entity == entity and e.slot == slot}))
    withdrawn = set(withdraw)
    prompt = _render_prompt(list(b.events), [a for a in b.atts if a.attestation_id not in withdrawn],
                            b.roles, entity, slot, as_of, query, proposed)
    return EpistemicEpisode(
        **{**proto.__dict__,
           "gold_label": label, "gold_value": gold, "binary_gold_value": binary_gold, "candidate_values": cand,
           "store_slot_status": res["status"], "store_slot_reason": res["reason"],
           "store_query_reason": view.reason.get(query, ""), "prompt": prompt}
    )


def generate_episodes(n: int, seed: int = DEFAULT_AUDIT_SEED) -> List[EpistemicEpisode]:
    return [generate_episode(i, seed) for i in range(n)]


# ---------------------------------------------------------------------------
# Completion parsing + rewards
# ---------------------------------------------------------------------------
_THINK_RE = re.compile(r"<think>.*?</think>", re.S | re.I)
_VERDICT_RE = re.compile(r"VERDICT\s*[:=]\s*[*`\"']*\s*([A-Za-z]+)", re.I)
_ANSWER_RE = re.compile(r"ANSWER\s*[:=]\s*([^\n\r]+)", re.I)
_VERDICT_MAP = {"T": LABEL_T, "TRUE": LABEL_T, "F": LABEL_F, "FALSE": LABEL_F, "U": LABEL_U, "UNKNOWN": LABEL_U,
                "UNDETERMINED": LABEL_U, "UNRESOLVED": LABEL_U, "CONTESTED": LABEL_U}
_ABSTAIN_TOKENS = {"abstain", "none", "unknown", "n/a", "unresolved"}


def _norm_value(s: str) -> str:
    return s.strip().strip("*`\"' ").rstrip(".;,").strip("*`\"' ").casefold()


def parse_completion(completion_text: Any) -> Tuple[Optional[str], Optional[str]]:
    """Return (verdict in {T,F,U} or None, normalised answer or 'ABSTAIN' or None)."""
    if isinstance(completion_text, list):  # chat-format completion
        completion_text = "".join(str(m.get("content", "")) for m in completion_text if isinstance(m, Mapping))
    text = _THINK_RE.sub(" ", str(completion_text or ""))
    if "</think>" in text.lower():
        text = text[text.lower().rfind("</think>") + len("</think>"):]
    elif "<think>" in text.lower():
        text = text[: text.lower().find("<think>")]
    verdicts = _VERDICT_RE.findall(text)
    answers = _ANSWER_RE.findall(text)
    verdict = _VERDICT_MAP.get(verdicts[-1].upper()) if verdicts else None
    answer: Optional[str] = None
    if answers:
        a = _norm_value(answers[-1])
        answer = ABSTAIN if a in _ABSTAIN_TOKENS else (a or None)
    return verdict, answer


def score_epistemic_completion_detailed(completion_text: Any, episode: EpisodeLike) -> Dict[str, Any]:
    ep = _coerce_episode(episode)
    verdict, answer = parse_completion(completion_text)
    gold_norm = _norm_value(ep.gold_value) if ep.gold_value is not None else None
    if answer is None:
        return {"reward": REWARD_MALFORMED, "outcome": "malformed", "verdict": verdict, "answer": answer}
    if ep.gold_label == LABEL_U:
        if answer != ABSTAIN:
            return {"reward": REWARD_COMMIT_WRONG, "outcome": "hallucinated_resolution", "verdict": verdict,
                    "answer": answer}
        ans_r, outcome = REWARD_CORRECT_ANSWER, "correct_abstention"
    else:
        if answer == ABSTAIN:
            ans_r, outcome = REWARD_ABSTAIN_ON_DETERMINATE, "over_abstention"
        elif answer == gold_norm:
            ans_r, outcome = REWARD_CORRECT_ANSWER, "correct_value"
        else:
            trap = answer == _norm_value(ep.proposed_value) and ep.gold_label == LABEL_F
            return {"reward": REWARD_COMMIT_WRONG, "outcome": "accepted_trap" if trap else "wrong_value",
                    "verdict": verdict, "answer": answer}
    ver_r = REWARD_CORRECT_VERDICT if verdict == ep.gold_label else 0.0
    return {"reward": ans_r + ver_r, "outcome": outcome if ver_r else outcome + "_wrong_verdict",
            "verdict": verdict, "answer": answer}


def score_epistemic_completion(completion_text: Any, episode: EpisodeLike) -> float:
    """3-valued T/F/U verifiable reward in [-1, 1]; 1.0 only for the exactly correct verdict + answer."""
    return float(score_epistemic_completion_detailed(completion_text, episode)["reward"])


def binary_exact_match_reward(completion_text: Any, episode: EpisodeLike) -> float:
    """2-valued RLVR baseline: 1 iff ANSWER equals a single gold string (U collapsed to last-writer-wins)."""
    ep = _coerce_episode(episode)
    _, answer = parse_completion(completion_text)
    return 1.0 if answer is not None and answer == _norm_value(ep.binary_gold_value) else 0.0


def binary_verdict_reward(completion_text: Any, episode: EpisodeLike) -> float:
    """2-valued RLVR baseline: closed-world verdict (U collapsed to F); 1 iff VERDICT matches."""
    ep = _coerce_episode(episode)
    verdict, _ = parse_completion(completion_text)
    gold2 = LABEL_T if ep.gold_label == LABEL_T else LABEL_F
    return 1.0 if verdict == gold2 else 0.0


def oracle_completion(episode: EpisodeLike) -> str:
    ep = _coerce_episode(episode)
    return f"VERDICT: {ep.gold_label}\nANSWER: {ep.gold_value if ep.gold_value is not None else ABSTAIN}"


def episode_to_trl_row(episode: EpistemicEpisode) -> Dict[str, Any]:
    """Dataset row for TRL GRPOTrainer: conversational prompt + JSON episode column for the reward fn."""
    return {"prompt": [{"role": "user", "content": episode.prompt}], "episode": episode.to_json()}


def ek5_wfs_reward_func(prompts: Any = None, completions: Optional[Sequence[Any]] = None,
                        episode: Optional[Sequence[EpisodeLike]] = None, **kwargs: Any) -> List[float]:
    """TRL GRPOTrainer reward function: ``reward_funcs=[ek5_wfs_reward_func]`` with an ``episode`` column."""
    if completions is None or episode is None:
        raise ValueError("ek5_wfs_reward_func needs `completions` and the `episode` dataset column")
    if len(completions) != len(episode):
        raise ValueError("completions / episode length mismatch")
    return [score_epistemic_completion(c, e) for c, e in zip(completions, episode)]


# ---------------------------------------------------------------------------
# Baseline policies + audit
# ---------------------------------------------------------------------------
def _slot_visible_events(ep: EpistemicEpisode) -> List[StratifiedEvent]:
    evs, _ = _visible(ep)
    return [e for e in evs if e.entity == ep.entity and e.slot == ep.slot]


def policy_completion(policy: str, ep: EpistemicEpisode) -> str:
    slot_evs = _slot_visible_events(ep)
    if policy == "oracle_3valued":
        return oracle_completion(ep)
    if policy == "abstention_collapse":  # the binary-RLVR optimum: oracle on T/F, commits LWW on U
        if ep.gold_label == LABEL_U:
            return f"VERDICT: T\nANSWER: {ep.binary_gold_value}"
        return oracle_completion(ep)
    if policy == "always_abstain":
        return f"VERDICT: U\nANSWER: {ABSTAIN}"
    if policy == "proposal_acceptor":
        return f"VERDICT: T\nANSWER: {ep.proposed_value}"
    if policy == "last_writer_wins":
        e = max(slot_evs, key=lambda x: (x.tx_time, x.event_id))
        return f"VERDICT: T\nANSWER: {e.claim_value}"
    if policy == "highest_raw_tier":
        e = max(slot_evs, key=lambda x: (x.authority_tier, x.tx_time, x.event_id))
        return f"VERDICT: T\nANSWER: {e.claim_value}"
    if policy == "majority_vote":
        counts = Counter(e.claim_value for e in slot_evs)
        last = {e.claim_value: (e.tx_time, e.event_id) for e in sorted(slot_evs, key=lambda x: (x.tx_time, x.event_id))}
        v = max(counts, key=lambda k: (counts[k], last[k]))
        return f"VERDICT: T\nANSWER: {v}"
    raise KeyError(policy)


POLICIES = ("oracle_3valued", "abstention_collapse", "always_abstain", "proposal_acceptor", "last_writer_wins",
            "highest_raw_tier", "majority_vote")
REWARD_FUNCS = {"three_valued_wfs": score_epistemic_completion, "binary_exact_match": binary_exact_match_reward,
                "binary_verdict": binary_verdict_reward}


def audit_episode(ep: EpistemicEpisode) -> Dict[str, Any]:
    """All consistency checks for one episode; every count must be 0."""
    store, view = build_episode_store(ep)
    out: Dict[str, Any] = {"wfs_oracle_val_mismatches": 0, "independent_val_mismatches": 0,
                           "independent_eff_auth_mismatches": 0, "independent_eligibility_mismatches": 0,
                           "label_mismatch_vs_independent": 0, "label_mismatch_vs_intent": 0,
                           "gold_value_mismatch_vs_independent": 0, "nodes_checked": 0}
    for st in ({id(store): store, id(view): view}).values():
        nodes = list(st.events)
        oracle = compute_alternating_wfs_oracle(nodes, st.refs, st.deps_v)
        out["wfs_oracle_val_mismatches"] += sum(1 for v in nodes if oracle[v] != st.val[v])
        out["nodes_checked"] += len(nodes)
    ref = independent_reference(ep)
    for v in view.events:
        out["independent_val_mismatches"] += int(ref["val"][v] != view.val[v])
        out["independent_eff_auth_mismatches"] += int(ref["eff_auth"][v] != view.eff_auth[v])
        out["independent_eligibility_mismatches"] += int(ref["eligibility"][v] != view.eligibility[v])
    out["label_mismatch_vs_independent"] = int(ref["label"] != ep.gold_label)
    out["label_mismatch_vs_intent"] = int(ep.intended_label != ep.gold_label)
    out["gold_value_mismatch_vs_independent"] = int(ref["gold_value"] != ep.gold_value)
    return out


def _candidate_completions(ep: EpistemicEpisode) -> List[str]:
    return [f"VERDICT: {v}\nANSWER: {a}" for v in (LABEL_T, LABEL_F, LABEL_U)
            for a in list(ep.candidate_values) + [ABSTAIN]]


def run_oracle_audit(n: int = 1000, seed: int = DEFAULT_AUDIT_SEED) -> Dict[str, Any]:
    """Generate ``n`` episodes and report oracle mismatches, label mix and reward separation."""
    eps = generate_episodes(n, seed)
    totals: Dict[str, int] = defaultdict(int)
    by_family: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for ep in eps:
        a = audit_episode(ep)
        for k, v in a.items():
            totals[k] += v
        by_family[ep.family][ep.gold_label] += 1

    labels = Counter(ep.gold_label for ep in eps)
    # Per-episode argmax analysis over the exhaustive candidate completion set.
    sep = {name: {"oracle_unique_argmax": 0, "oracle_is_argmax": 0, "abstention_strictly_dominated_on_U": 0,
                  "min_margin_oracle_vs_best_other": None} for name in REWARD_FUNCS}
    for ep in eps:
        oc = oracle_completion(ep)
        cands = _candidate_completions(ep)
        if oc not in cands:
            cands.append(oc)
        for name, fn in REWARD_FUNCS.items():
            scores = {c: fn(c, ep) for c in cands}
            o = scores[oc]
            best_other = max(s for c, s in scores.items() if c != oc)
            top = max(scores.values())
            sep[name]["oracle_is_argmax"] += int(o == top)
            sep[name]["oracle_unique_argmax"] += int(o > best_other)
            margin = o - best_other
            cur_min = sep[name]["min_margin_oracle_vs_best_other"]
            sep[name]["min_margin_oracle_vs_best_other"] = margin if cur_min is None else min(cur_min, margin)
            if ep.gold_label == LABEL_U:
                best_abstain = max(s for c, s in scores.items() if c.endswith(ABSTAIN))
                best_commit = max(s for c, s in scores.items() if not c.endswith(ABSTAIN))
                sep[name]["abstention_strictly_dominated_on_U"] += int(best_commit > best_abstain)

    policy_rewards: Dict[str, Dict[str, float]] = {}
    for pol in POLICIES:
        comps = [policy_completion(pol, ep) for ep in eps]
        policy_rewards[pol] = {name: round(sum(fn(c, ep) for c, ep in zip(comps, eps)) / len(eps), 6)
                               for name, fn in REWARD_FUNCS.items()}
        policy_rewards[pol]["three_valued_wfs_on_U"] = round(
            sum(score_epistemic_completion(c, ep) for c, ep in zip(comps, eps) if ep.gold_label == LABEL_U)
            / max(1, labels[LABEL_U]), 6)

    p_u = labels[LABEL_U] / len(eps)
    digest = hashlib.sha256("\n".join(ep.to_json() for ep in eps).encode("utf-8")).hexdigest()
    return {
        "n_episodes": len(eps),
        "seed": seed,
        "episodes_sha256": digest,
        "label_counts": dict(sorted(labels.items())),
        "scenario_counts": dict(sorted(Counter(ep.scenario for ep in eps).items())),
        "family_label_counts": {k: dict(v) for k, v in sorted(by_family.items())},
        "as_of_episodes": sum(1 for ep in eps if ep.as_of is not None),
        "withdrawal_episodes": sum(1 for ep in eps if ep.withdrawn_attestation_ids),
        "incremental_multi_batch_episodes": sum(1 for ep in eps if len(ep.mutation_plan) > 1),
        "mean_events_per_episode": round(sum(len(ep.events) for ep in eps) / len(eps), 3),
        "mean_attestations_per_episode": round(sum(len(ep.attestations) for ep in eps) / len(eps), 3),
        "mismatches": dict(totals),
        "total_mismatches": sum(v for k, v in totals.items() if k != "nodes_checked"),
        "reward_separation": sep,
        "policy_mean_rewards": policy_rewards,
        "analytic_separation": {
            "p_U": round(p_u, 6),
            "binary_exact_gain_of_abstention_collapse_over_oracle": round(
                policy_rewards["abstention_collapse"]["binary_exact_match"]
                - policy_rewards["oracle_3valued"]["binary_exact_match"], 6),
            "three_valued_gain_of_oracle_over_abstention_collapse": round(
                policy_rewards["oracle_3valued"]["three_valued_wfs"]
                - policy_rewards["abstention_collapse"]["three_valued_wfs"], 6),
        },
    }


if __name__ == "__main__":  # pragma: no cover - manual audit entry point
    import sys

    n_arg = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    print(json.dumps(run_oracle_audit(n_arg), indent=2, sort_keys=True))
