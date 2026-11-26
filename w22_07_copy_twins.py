"""W22-07: blinded hidden-copy twins (graded leakage) and a priced provenance-check policy.

Every twin pair shares one real root record from the #55 frozen wild snapshots.  Member C
(withdrawn-root-copy) and member I (independent corroboration) have identical public records;
they differ only in the hidden ``echo_of`` basis of the undeclared witnesses.  Every policy
receives the stripped public view; ground truth comes from
``chelated_wild_revision_streams.oracle_at_cut`` on the unstripped records.

The policies, prices, and the kill/advance rule are preregistered in
``artifacts/pv7_w23b_w22_07/PREREGISTRATION.md``; ``kill_rule_decision`` implements it.
Existing engines are reused: EK5_v3 via ``project_ek5``/``ek5_prediction`` and the SQL
challenger ledger via ``sql_envelopes`` from ``scripts/run_pv7_w18_6c_wild_revisions.py``.

Deterministic and CPU-only.  No LLM, embedding model, or network.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from chelated_pv7_spine.challenger_ledger import (
    QUERY_MODES,
    ChallengerBitemporalLedger,
    LedgerEdgeEnvelope,
    SQLiteChallengerBackend,
)
from chelated_wild_revision_streams import (
    AdjudicatedFamily,
    CutSpec,
    StreamRecord,
    oracle_at_cut,
)
from scripts.run_pv7_w18_6c_wild_revisions import (
    build_ek5_store,
    ek5_prediction,
    sql_envelopes,
)

SUPPORT = "SUPPORT"
RETRACT = "RETRACT"
ABSTAIN = "ABSTAIN"
DECISION_ORDER: Tuple[str, ...] = (SUPPORT, RETRACT, ABSTAIN)

PRIMARY_PRICES: Dict[str, float] = {"w_fs": 1.0, "w_fr": 1.0, "w_a": 0.35, "c": 0.10}
PREVALENCES: Tuple[float, ...] = (0.05, 0.25, 0.50)
HARM_TOLERANCE = 0.005
COVERAGE_TOLERANCE = 0.005
LEAKAGE_THRESHOLD = 0.55
LEAK_GRADES: Tuple[float, ...] = (0.0, 0.25, 0.5, 1.0)
FIXTURE_SEEDS: Tuple[int, ...] = (20261126, 20261127, 20261128)
CV_SEED = 1601
BOOTSTRAP_SEED = 1601
BOOTSTRAP_RESAMPLES = 2000
COSINE_GRID: Tuple[float, ...] = tuple(round(0.05 * i, 2) for i in range(1, 20))
N_COSINE_BINS = 5
SOURCE_POOL: Tuple[str, ...] = ("feed-1", "feed-2", "feed-3", "feed-4", "feed-5", "feed-6")
TRANSFORMS: Tuple[str, ...] = ("T0", "T1", "T2", "T3", "T4")
LEAK_CUE_PREFIX = "Syndicated mirror: "
ROLE_WORDS: Tuple[str, ...] = ("copy", "mirror", "echo", "indep", "syndicat")
ROOT_TICK = 10
WITNESS_TICK = 20
WITHDRAW_TICK = 30
PRE_CUT = CutSpec("t_pre_withdrawal", 25, 25)
POST_CUT = CutSpec("t_post_withdrawal", 40, 40)
CHECK_EDGE_TICK = 35
AUTHORITY = 5
CONVENTIONAL_ARMS: Tuple[str, ...] = ("A3", "A4", "A5", "A5a")
TD_ARM = "A6"
ALL_ARMS: Tuple[str, ...] = ("A0", "A1", "A2", "A3", "A4", "A5", "A5a", "A6")

_TOKEN_RE = re.compile(r"[a-z0-9]+")


# ---------------------------------------------------------------------------------------------------------------
# Real roots
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RootSource:
    source_family_id: str
    stratum: str
    entity_key: str
    claim_value: str
    text: str


def load_real_roots(snapshot_dir: Path) -> List[RootSource]:
    """First ASSERT/CONFLICT record of every #55 frozen family, in file then family order."""
    roots: List[RootSource] = []
    for path in sorted(Path(snapshot_dir).glob("stratum*.json")):
        families = json.loads(path.read_text(encoding="utf-8"))
        for family in families:
            first = next(r for r in family["records"] if r["op_kind"] in ("ASSERT", "CONFLICT"))
            if not first.get("text") or not first.get("claim_value"):
                raise ValueError(f"root of {family['family_id']} lacks text or claim")
            roots.append(
                RootSource(
                    source_family_id=family["family_id"],
                    stratum=family["stratum"],
                    entity_key=first["entity_key"],
                    claim_value=first["claim_value"],
                    text=first["text"],
                )
            )
    return roots


# ---------------------------------------------------------------------------------------------------------------
# Surface transforms (label-independent at leakage grade 0)
# ---------------------------------------------------------------------------------------------------------------
def apply_transform(name: str, root: RootSource) -> str:
    text = root.text
    if name == "T0":
        return text
    if name == "T1":
        return f"{root.entity_key}: {root.claim_value}"
    if name == "T2":
        words = text.split()
        keep = max(1, math.ceil(0.6 * len(words)))
        return " ".join(words[:keep])
    if name == "T3":
        if ":" not in text:
            return text
        head, tail = text.split(":", 1)
        return f"{tail.strip()}: {head.strip()}"
    if name == "T4":
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", " ", text.lower())).strip()
    raise ValueError(f"unknown transform {name!r}")


def tf_cosine(text_a: str, text_b: str) -> float:
    a = Counter(_TOKEN_RE.findall(text_a.lower()))
    b = Counter(_TOKEN_RE.findall(text_b.lower()))
    if not a or not b:
        return 0.0
    dot = sum(count * b.get(token, 0) for token, count in a.items())
    norm = math.sqrt(sum(v * v for v in a.values())) * math.sqrt(sum(v * v for v in b.values()))
    return dot / norm if norm else 0.0


def cosine_bin(value: float) -> int:
    return min(N_COSINE_BINS - 1, max(0, int(value * N_COSINE_BINS)))


# ---------------------------------------------------------------------------------------------------------------
# Twin fixture
# ---------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Witness:
    record_id: str
    text: str
    source: str
    tx: int
    declared: bool
    transform: str


@dataclass(frozen=True)
class TwinMember:
    """One family of a twin pair.  ``hidden`` maps undeclared witness index -> 1 independent / 0 copy."""

    pair_id: str
    label: str  # "C" or "I"
    family: AdjudicatedFamily
    root_id: str
    root_text: str
    witnesses: Tuple[Witness, ...]
    hidden: Tuple[int, ...]
    leaky: bool
    stratum: str

    @property
    def undeclared(self) -> Tuple[int, ...]:
        return tuple(i for i, w in enumerate(self.witnesses) if not w.declared)

    @property
    def truth(self) -> str:
        return SUPPORT if self.label == "I" else RETRACT


@dataclass(frozen=True)
class TwinPair:
    pair_id: str
    seed: int
    index: int
    members: Tuple[TwinMember, TwinMember]
    k: int
    d: int
    j: int
    leaky: bool
    split: str  # "dev" or "test"


def _blind(salt: str, *parts: object) -> str:
    raw = "|".join([salt] + [str(p) for p in parts]).encode("utf-8")
    return "r" + hashlib.sha256(raw).hexdigest()[:12]


def build_twin_pairs(roots: Sequence[RootSource], seed: int, leak: float) -> List[TwinPair]:
    """Build one blinded twin pair per root.  Structure depends only on (seed, index); leakage
    draws use a separate stream so every grade shares the same pair structure and nests."""
    salt = hashlib.sha256(f"w22-07-salt:{seed}".encode()).hexdigest()
    split_rng = random.Random(seed + 1)
    order = list(range(len(roots)))
    split_rng.shuffle(order)
    dev = set(order[: len(order) // 2])
    pairs: List[TwinPair] = []
    for index, root in enumerate(roots):
        rng = random.Random(f"struct:{seed}:{index}")
        leak_rng = random.Random(f"leak:{seed}:{index}")
        k = rng.choice((2, 3))
        d = rng.randint(0, k - 1)
        m = k - d
        j = rng.randint(1, m)
        indep_positions = sorted(rng.sample(range(m), j))
        transforms = [rng.choice(TRANSFORMS) for _ in range(k)]
        sources = [rng.choice(SOURCE_POOL) for _ in range(k)]
        member_tags = ["x", "y"]
        rng.shuffle(member_tags)  # tag of C, tag of I
        leak_draw = leak_rng.random()
        leaky = leak_draw < leak
        leaky_indep_transforms = [leak_rng.choice(("T1", "T3")) for _ in range(k)]
        pair_id = _blind(salt, "pair", index)
        members = []
        for label, tag in (("C", member_tags[0]), ("I", member_tags[1])):
            tenant = _blind(salt, "tenant", index, tag)
            root_id = _blind(salt, "rec", index, tag, "root")
            hidden: List[int] = []
            witnesses: List[Witness] = []
            records: List[StreamRecord] = [
                StreamRecord(
                    record_id=root_id,
                    tenant_id=tenant,
                    entity_key=root.entity_key,
                    claim_value=root.claim_value,
                    authority_rank=AUTHORITY,
                    valid_from=ROOT_TICK,
                    valid_to=None,
                    tx_time=ROOT_TICK,
                    op_kind="ASSERT",
                    text=root.text,
                    source="origin",
                )
            ]
            for w in range(k):
                declared = w < d
                undeclared_pos = w - d
                independent = (not declared) and label == "I" and undeclared_pos in indep_positions
                if not declared:
                    hidden.append(1 if independent else 0)
                transform = transforms[w]
                text = apply_transform(transform, root)
                if leaky and not declared:
                    if independent:
                        transform = leaky_indep_transforms[w]
                        text = apply_transform(transform, root)
                    else:
                        transform = "LEAK_CUE"
                        text = LEAK_CUE_PREFIX + root.text
                rid = _blind(salt, "rec", index, tag, "w", w)
                tick = WITNESS_TICK + w
                witnesses.append(Witness(rid, text, sources[w], tick, declared, transform))
                records.append(
                    StreamRecord(
                        record_id=rid,
                        tenant_id=tenant,
                        entity_key=root.entity_key,
                        claim_value=root.claim_value,
                        authority_rank=AUTHORITY,
                        valid_from=tick,
                        valid_to=None,
                        tx_time=tick,
                        op_kind="ASSERT",
                        text=text,
                        source=sources[w],
                        echo_of="" if (declared or independent) else root_id,
                        explicit_parent_ids=(root_id,) if declared else (),
                    )
                )
            records.append(
                StreamRecord(
                    record_id=_blind(salt, "rec", index, tag, "wd"),
                    tenant_id=tenant,
                    entity_key=root.entity_key,
                    claim_value="",
                    authority_rank=AUTHORITY,
                    valid_from=WITHDRAW_TICK,
                    valid_to=None,
                    tx_time=WITHDRAW_TICK,
                    op_kind="WITHDRAW",
                    text="",
                    source="withdrawal",
                    target_id=root_id,
                )
            )
            family = AdjudicatedFamily(
                family_id=_blind(salt, "fam", index, tag),
                stratum="w22_07_twin",
                tenant_id=tenant,
                entity_key=root.entity_key,
                records=tuple(records),
                cuts=(PRE_CUT, POST_CUT),
                adjudicated=True,
                synthetic=True,
            )
            members.append(
                TwinMember(
                    pair_id=pair_id,
                    label=label,
                    family=family,
                    root_id=root_id,
                    root_text=root.text,
                    witnesses=tuple(witnesses),
                    hidden=tuple(hidden),
                    leaky=leaky,
                    stratum=root.stratum,
                )
            )
        ordered = sorted(members, key=lambda mem: mem.family.family_id)
        pairs.append(
            TwinPair(
                pair_id=pair_id,
                seed=seed,
                index=index,
                members=(ordered[0], ordered[1]),
                k=k,
                d=d,
                j=j,
                leaky=leaky,
                split="dev" if index in dev else "test",
            )
        )
    return pairs


def oracle_truth(member: TwinMember, cut: CutSpec = POST_CUT) -> str:
    oracle = oracle_at_cut(member.family, cut)
    if oracle.disposition == "retracted":
        return RETRACT
    if oracle.disposition == "disputed":
        return ABSTAIN
    return SUPPORT


def public_records(member: TwinMember) -> Tuple[StreamRecord, ...]:
    return tuple(replace(record, echo_of="") for record in member.family.records)


def serialize_public(member: TwinMember) -> str:
    """Every public field of every record, as one token string (the leakage classifier input)."""
    parts = []
    for position, record in enumerate(public_records(member)):
        parts.append(
            " ".join(
                [
                    f"pos_{position}",
                    f"op_{record.op_kind}",
                    f"src_{record.source.replace('-', '_')}",
                    f"tx_{record.tx_time}",
                    f"vf_{record.valid_from}",
                    f"auth_{record.authority_rank}",
                    f"decl_{1 if record.explicit_parent_ids else 0}",
                    f"id_{record.record_id}",
                    f"ten_{record.tenant_id}",
                    f"tgt_{record.target_id or 'none'}",
                    record.claim_value,
                    record.entity_key,
                    record.text,
                ]
            )
        )
    return " ".join(parts)


def role_word_violations(member: TwinMember) -> List[str]:
    """Role words in public fields that are not inherited from the real root text."""
    root_lower = member.root_text.lower()
    violations = []
    for record in public_records(member):
        for value in (record.text, record.source, record.record_id, record.tenant_id, record.claim_value):
            low = value.lower()
            for word in ROLE_WORDS:
                if word in low and word not in root_lower:
                    violations.append(f"{record.record_id}:{word}")
    return violations


# ---------------------------------------------------------------------------------------------------------------
# Policies
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class Outcome:
    decision: str
    checks: int = 0
    checked: Tuple[int, ...] = ()


def policy_no_check(member: TwinMember) -> Outcome:
    """A0: the claim survives if any non-withdrawn assertion survives."""
    return Outcome(SUPPORT if member.witnesses else RETRACT)


def policy_cosine(member: TwinMember, tau: float) -> Outcome:
    """A1: single-link TF-cosine clusters; a cluster whose earliest member is the withdrawn root is dead."""
    nodes = [(ROOT_TICK, member.root_id, member.root_text)] + [(w.tx, w.record_id, w.text) for w in member.witnesses]
    parent = list(range(len(nodes)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b in itertools.combinations(range(len(nodes)), 2):
        if tf_cosine(nodes[a][2], nodes[b][2]) >= tau:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
    clusters: Dict[int, List[int]] = defaultdict(list)
    for i in range(len(nodes)):
        clusters[find(i)].append(i)
    for members in clusters.values():
        earliest = min(members, key=lambda i: (nodes[i][0], nodes[i][1]))
        if nodes[earliest][1] != member.root_id:
            return Outcome(SUPPORT)
    return Outcome(RETRACT)


def _ek5_decision(disposition: str) -> str:
    if disposition == "retracted":
        return RETRACT
    if disposition == "disputed":
        return ABSTAIN
    return SUPPORT


def run_ek5(members: Sequence[TwinMember], cut: CutSpec = POST_CUT) -> Dict[str, Outcome]:
    """A2: EK5_v3 through bundle #55's own projection (echo_of is stripped inside project_ek5)."""
    store, _ = build_ek5_store([m.family for m in members])
    out = {}
    for member in members:
        disposition, _lineage = ek5_prediction(store, member.family, cut)
        out[member.family.family_id] = Outcome(_ek5_decision(disposition))
    return out


def _ledger_decision(result) -> str:
    if result.status == "RESOLVED":
        return SUPPORT
    if result.status == "DISPUTED":
        return ABSTAIN
    return RETRACT


def _query_ledger(ledger: ChallengerBitemporalLedger, member: TwinMember, cut: CutSpec) -> str:
    decisions = set()
    for mode in QUERY_MODES:
        result = ledger.query_as_of(member.family.tenant_id, member.family.entity_key, cut.valid_time, cut.tx_time, mode=mode)
        decisions.add(_ledger_decision(result))
    if len(decisions) != 1:
        raise AssertionError(f"ledger modes disagree for {member.family.family_id}: {decisions}")
    return decisions.pop()


def build_member_ledger(member: TwinMember) -> ChallengerBitemporalLedger:
    """Per-family SQLite ledger from bundle #55's ``sql_envelopes`` (declared DERIVES edges included).

    Edge envelopes are routed to ``add_edge``; ``build_sql_ledger`` in the #55 runner routes
    every kind to ``append`` and would raise on any edge (its fixture never produced one)."""
    ledger = ChallengerBitemporalLedger(SQLiteChallengerBackend())
    for kind, envelope in sql_envelopes(member.family):
        if kind == "edge":
            ledger.add_edge(envelope)
        else:
            ledger.append(envelope)
    return ledger


def policy_ledger(member: TwinMember, budget: int, cut: CutSpec = POST_CUT) -> Outcome:
    """A3 (budget 0) / A4: copy-aware ledger; checks undeclared witnesses earliest-first.

    A revealed copy becomes a DERIVES(witness -> root) edge; a revealed independent witness
    stops checking with SUPPORT."""
    ledger = build_member_ledger(member)
    checked: List[int] = []
    hidden = dict(zip(member.undeclared, member.hidden))
    for idx in member.undeclared:
        if len(checked) >= budget:
            break
        checked.append(idx)
        if hidden[idx] == 1:
            return Outcome(SUPPORT, len(checked), tuple(checked))
        witness = member.witnesses[idx]
        ledger.add_edge(
            LedgerEdgeEnvelope(
                tenant_id=member.family.tenant_id,
                edge_id=f"chk:{witness.record_id}",
                src_id=witness.record_id,
                dst_id=member.root_id,
                tx_time=CHECK_EDGE_TICK,
            )
        )
    return Outcome(_query_ledger(ledger, member, cut), len(checked), tuple(checked))


# ---------------------------------------------------------------------------------------------------------------
# Shared provenance-aware copy posterior (A5 / A5a / A6)
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class CopyPosterior:
    prior: Dict[int, Dict[Tuple[int, ...], float]] = field(default_factory=dict)
    bin_lik: Dict[int, List[float]] = field(default_factory=dict)  # label -> P(bin | label)

    @classmethod
    def fit(cls, members: Sequence[TwinMember]) -> "CopyPosterior":
        config_counts: Dict[int, Counter] = defaultdict(Counter)
        bin_counts = {0: [1.0] * N_COSINE_BINS, 1: [1.0] * N_COSINE_BINS}
        for member in members:
            m = len(member.undeclared)
            config_counts[m][member.hidden] += 1
            for idx, h in zip(member.undeclared, member.hidden):
                bin_counts[h][cosine_bin(tf_cosine(member.witnesses[idx].text, member.root_text))] += 1
        prior: Dict[int, Dict[Tuple[int, ...], float]] = {}
        for m in range(1, 4):
            counts = config_counts.get(m, Counter())
            configs = list(itertools.product((0, 1), repeat=m))
            total = sum(counts[c] + 0.5 for c in configs)
            prior[m] = {c: (counts[c] + 0.5) / total for c in configs}
        bin_lik = {label: [v / sum(vals) for v in vals] for label, vals in bin_counts.items()}
        return cls(prior=prior, bin_lik=bin_lik)

    def features(self, member: TwinMember) -> Tuple[int, ...]:
        return tuple(cosine_bin(tf_cosine(member.witnesses[i].text, member.root_text)) for i in member.undeclared)

    def posterior(self, bins: Tuple[int, ...], known: Mapping[int, int]) -> Dict[Tuple[int, ...], float]:
        """P(h | bins, known) over undeclared positions (``known`` keys are undeclared positions)."""
        m = len(bins)
        weights = {}
        for config, p in self.prior[m].items():
            if any(config[pos] != val for pos, val in known.items()):
                continue
            w = p
            for pos, h in enumerate(config):
                w *= self.bin_lik[h][bins[pos]]
            weights[config] = w
        total = sum(weights.values())
        return {c: w / total for c, w in weights.items()}


def _p_support(post: Mapping[Tuple[int, ...], float]) -> float:
    return sum(p for config, p in post.items() if any(config))


def stop_costs(p_support: float, prices: Mapping[str, float], allow_abstain: bool) -> Dict[str, float]:
    costs = {SUPPORT: (1.0 - p_support) * prices["w_fs"], RETRACT: p_support * prices["w_fr"]}
    if allow_abstain:
        costs[ABSTAIN] = prices["w_a"]
    return costs


def _argmin_decision(costs: Mapping[str, float]) -> str:
    return min((d for d in DECISION_ORDER if d in costs), key=lambda d: (costs[d], DECISION_ORDER.index(d)))


def policy_bayes_earliest(member: TwinMember, model: CopyPosterior, budget: int, prices: Mapping[str, float], allow_abstain: bool) -> Outcome:
    """A5 / A5a: conventional copy model, unpriced earliest-first checks, then argmin expected harm."""
    bins = model.features(member)
    known: Dict[int, int] = {}
    checked: List[int] = []
    for pos in range(len(bins)):
        if len(checked) >= budget or any(known.values()):
            break
        known[pos] = member.hidden[pos]
        checked.append(member.undeclared[pos])
    costs = stop_costs(_p_support(model.posterior(bins, known)), prices, allow_abstain)
    return Outcome(_argmin_decision(costs), len(checked), tuple(checked))


def _td_value(model: CopyPosterior, bins: Tuple[int, ...], known: Dict[int, int], budget: int, prices: Mapping[str, float], memo: Dict) -> Tuple[float, Optional[int]]:
    """Exact finite-horizon expected harm.  Returns (value, position to check or None = stop)."""
    key = (tuple(sorted(known.items())), budget)
    if key in memo:
        return memo[key]
    post = model.posterior(bins, known)
    best_value = min(stop_costs(_p_support(post), prices, True).values())
    best_action: Optional[int] = None
    if budget > 0:
        for pos in range(len(bins)):
            if pos in known:
                continue
            p_one = sum(p for config, p in post.items() if config[pos] == 1)
            value = prices["c"]
            for outcome, prob in ((1, p_one), (0, 1.0 - p_one)):
                if prob <= 0.0:
                    continue
                nxt = dict(known)
                nxt[pos] = outcome
                value += prob * _td_value(model, bins, nxt, budget - 1, prices, memo)[0]
            if value < best_value - 1e-12:
                best_value, best_action = value, pos
    memo[key] = (best_value, best_action)
    return memo[key]


def policy_td(member: TwinMember, model: CopyPosterior, budget: int, prices: Mapping[str, float]) -> Outcome:
    """A6: T-D priced provenance check (value-of-information targeting + reject option)."""
    bins = model.features(member)
    known: Dict[int, int] = {}
    checked: List[int] = []
    memo: Dict = {}
    remaining = budget
    while True:
        _value, action = _td_value(model, bins, known, remaining, prices, memo)
        if action is None:
            break
        known[action] = member.hidden[action]
        checked.append(member.undeclared[action])
        remaining -= 1
    costs = stop_costs(_p_support(model.posterior(bins, known)), prices, True)
    return Outcome(_argmin_decision(costs), len(checked), tuple(checked))


# ---------------------------------------------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------------------------------------------
def item_harm(decision: str, truth: str, checks: int, prices: Mapping[str, float]) -> float:
    harm = prices["c"] * checks
    if decision == ABSTAIN:
        return harm + prices["w_a"]
    if decision == truth:
        return harm
    return harm + (prices["w_fs"] if decision == SUPPORT else prices["w_fr"])


def summarize(rows: Sequence[Mapping[str, Any]], prices: Mapping[str, float]) -> Dict[str, Any]:
    """Prevalence-weighted harm and coverage for one arm's rows (keys: label, decision, truth, checks)."""
    by_label: Dict[str, List[Mapping[str, Any]]] = {"C": [], "I": []}
    for row in rows:
        by_label[row["label"]].append(row)
    stats = {}
    for label, items in by_label.items():
        if not items:
            stats[label] = None
            continue
        stats[label] = {
            "n": len(items),
            "mean_harm": sum(item_harm(r["decision"], r["truth"], r["checks"], prices) for r in items) / len(items),
            "answered": sum(r["decision"] != ABSTAIN for r in items) / len(items),
            "accuracy": sum(r["decision"] == r["truth"] for r in items) / len(items),
            "mean_checks": sum(r["checks"] for r in items) / len(items),
        }
    out: Dict[str, Any] = {"per_label": stats, "harm": {}, "coverage": {}}
    for pi in PREVALENCES:
        if stats["C"] is None or stats["I"] is None:
            out["harm"][str(pi)] = None
            out["coverage"][str(pi)] = None
            continue
        out["harm"][str(pi)] = pi * stats["C"]["mean_harm"] + (1 - pi) * stats["I"]["mean_harm"]
        out["coverage"][str(pi)] = pi * stats["C"]["answered"] + (1 - pi) * stats["I"]["answered"]
    out["accuracy"] = sum(r["decision"] == r["truth"] for r in rows) / len(rows) if rows else None
    out["mean_checks"] = sum(r["checks"] for r in rows) / len(rows) if rows else None
    pair_decisions: Dict[str, Dict[str, str]] = defaultdict(dict)
    for row in rows:
        pair_decisions[row["pair_key"]][row["label"]] = row["decision"]
    out["pairs_both_correct"] = sum(1 for d in pair_decisions.values() if d.get("C") == RETRACT and d.get("I") == SUPPORT)
    out["pairs_same_decision"] = sum(1 for d in pair_decisions.values() if d.get("C") == d.get("I"))
    out["n_pairs"] = len(pair_decisions)
    return out


def kill_rule_decision(summaries: Mapping[str, Mapping[str, Any]], fixture_valid: bool) -> Dict[str, Any]:
    """Preregistered rule: KILL_T_D if any conventional arm matches A6 on harm and coverage at every pi."""
    if not fixture_valid:
        return {"verdict": "FIXTURE_INVALID_LEAKAGE", "matching_conventional_arms": [], "per_arm": {}}
    td = summaries[TD_ARM]
    per_arm = {}
    matching = []
    for arm in CONVENTIONAL_ARMS:
        conv = summaries[arm]
        checks = {}
        ok = True
        for pi in PREVALENCES:
            key = str(pi)
            harm_ok = conv["harm"][key] <= td["harm"][key] + HARM_TOLERANCE
            cov_ok = conv["coverage"][key] >= td["coverage"][key] - COVERAGE_TOLERANCE
            checks[key] = {
                "harm_conv": conv["harm"][key],
                "harm_td": td["harm"][key],
                "harm_ok": harm_ok,
                "coverage_conv": conv["coverage"][key],
                "coverage_td": td["coverage"][key],
                "coverage_ok": cov_ok,
            }
            ok = ok and harm_ok and cov_ok
        per_arm[arm] = {"matches": ok, "by_prevalence": checks}
        if ok:
            matching.append(arm)
    return {
        "verdict": "KILL_T_D" if matching else "ADVANCE_T_D",
        "matching_conventional_arms": matching,
        "per_arm": per_arm,
    }


def tune_cosine_tau(dev_members: Sequence[TwinMember], prices: Mapping[str, float]) -> Tuple[float, Dict[str, float]]:
    """Pick tau on dev pairs minimizing harm at pi = 0.5 (ties -> smaller tau)."""
    scores = {}
    for tau in COSINE_GRID:
        rows = [
            {"label": m.label, "decision": policy_cosine(m, tau).decision, "truth": m.truth, "checks": 0, "pair_key": m.pair_id}
            for m in dev_members
        ]
        scores[str(tau)] = summarize(rows, prices)["harm"]["0.5"]
    best = min(COSINE_GRID, key=lambda t: (scores[str(t)], t))
    return best, scores


def cluster_bootstrap_delta(rows_by_arm: Mapping[str, Sequence[Mapping[str, Any]]], prices: Mapping[str, float], pi: float = 0.5) -> Dict[str, float]:
    """Pair-clustered bootstrap of H_A6(pi) - min_conv H_X(pi)."""
    clusters: Dict[str, Dict[str, List[Mapping[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for arm, rows in rows_by_arm.items():
        for row in rows:
            clusters[row["pair_key"]][arm].append(row)
    keys = sorted(clusters)
    rng = random.Random(BOOTSTRAP_SEED)

    def delta(sample_keys: Sequence[str]) -> float:
        harms = {}
        for arm in (TD_ARM,) + CONVENTIONAL_ARMS:
            rows = [row for key in sample_keys for row in clusters[key][arm]]
            harms[arm] = summarize(rows, prices)["harm"][str(pi)]
        return harms[TD_ARM] - min(harms[a] for a in CONVENTIONAL_ARMS)

    point = delta(keys)
    samples = sorted(delta([keys[rng.randrange(len(keys))] for _ in keys]) for _ in range(BOOTSTRAP_RESAMPLES))
    lo = samples[int(0.025 * BOOTSTRAP_RESAMPLES)]
    hi = samples[int(0.975 * BOOTSTRAP_RESAMPLES) - 1]
    return {"point": point, "ci95_low": lo, "ci95_high": hi, "resamples": BOOTSTRAP_RESAMPLES, "clusters": len(keys)}


# ---------------------------------------------------------------------------------------------------------------
# Leakage classifier (sklearn only in this isolated validation flow)
# ---------------------------------------------------------------------------------------------------------------
def grouped_text_classifier(texts: Sequence[str], labels: Sequence[int], groups: Sequence[str], n_splits: int = 5) -> Dict[str, Any]:
    """Grouped K-fold bag-of-words logistic regression; returns balanced accuracy over pooled folds."""
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score

    unique_groups = sorted(set(groups))
    rng = random.Random(CV_SEED)
    rng.shuffle(unique_groups)
    fold_of = {g: i % n_splits for i, g in enumerate(unique_groups)}
    predictions = [None] * len(texts)
    for fold in range(n_splits):
        train = [i for i, g in enumerate(groups) if fold_of[g] != fold]
        test = [i for i, g in enumerate(groups) if fold_of[g] == fold]
        if not test:
            continue
        train_labels = [labels[i] for i in train]
        if len(set(train_labels)) < 2:
            majority = train_labels[0]
            for i in test:
                predictions[i] = majority
            continue
        vec = CountVectorizer(binary=True, token_pattern=r"[A-Za-z0-9_]+", lowercase=True)
        x_train = vec.fit_transform([texts[i] for i in train])
        clf = LogisticRegression(C=1.0, max_iter=2000)
        clf.fit(x_train, train_labels)
        pred = clf.predict(vec.transform([texts[i] for i in test]))
        for i, p in zip(test, pred):
            predictions[i] = int(p)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(list(labels), predictions)),
        "n": len(texts),
        "n_groups": len(unique_groups),
        "positives": int(sum(labels)),
    }


def family_level_leakage(pairs: Sequence[TwinPair]) -> Dict[str, Any]:
    texts, labels, groups = [], [], []
    for pair in pairs:
        for member in pair.members:
            texts.append(serialize_public(member))
            labels.append(1 if member.label == "C" else 0)
            groups.append(pair.pair_id)
    return grouped_text_classifier(texts, labels, groups)


def witness_level_leakage(pairs: Sequence[TwinPair]) -> Dict[str, Any]:
    texts, labels, groups = [], [], []
    for pair in pairs:
        for member in pair.members:
            for idx, hidden in zip(member.undeclared, member.hidden):
                w = member.witnesses[idx]
                texts.append(f"src_{w.source.replace('-', '_')} pos_{idx} {w.text}")
                labels.append(0 if hidden == 1 else 1)  # 1 = copy
                groups.append(pair.pair_id)
    return grouped_text_classifier(texts, labels, groups)
