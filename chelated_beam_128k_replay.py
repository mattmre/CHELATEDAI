"""Wave 18 / Bundle #51 (Item 3B): gBench BEAM 128K 20-Conversation Cluster Bootstrap + Paired Withdrawal Replay.

Implements the locked Post-v7 R2 protocol for Item 3B (#51):
  1. `20`-conversation (`400`-question) `gBench` `BEAM 128K` ingestion (live JSON/JSONL or
     deterministic CPU drycheck fixture) with `burn_ledger` validation.
  2. Strict conversation-level cluster bootstrap (`n=20` clusters, NEVER inflating `N` by treating
     `400` correlated questions as independent!), reporting intraclass correlation (ICC) and cluster SD.
  3. Stateful checkpoint replay bound to `TRA1StateIdentityContract`.
  4. Paired derivative-only (`COPY_OF` / `DERIVES_CURRENT`) vs independent-corroboration withdrawal
     histories verifying 100% `incremental == full` `EK5` parity across EVERY prefix via
     `export_canonical_ek5_audit_trace` and `compare_with_fresh_rebuild`, contrasted against
     `ChallengerBitemporalLedger` (`recursive_cte` & `closure_table`).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from chelated_pv7_spine.burn_ledger import (
    BurnLedger,
    assert_corpus_unburned,
    build_burn_ledger,
)
from chelated_pv7_spine.challenger_ledger import (
    ChallengerBitemporalLedger,
    LedgerEdgeEnvelope,
    LedgerRecordEnvelope,
    OpKind,
    SQLiteChallengerBackend,
)
from chelated_pv7_spine.ek5_reason_canonicalizer import (
    compare_with_fresh_rebuild,
    export_canonical_ek5_audit_trace,
    reference_fresh_rebuild,
)
from chelated_pv7_spine.tra1_weight_sha_guard import (
    EMPTY_ADAPTER_SHA256,
    TRA1StateIdentityContract,
    sha256_digest,
)
from ek5_v3_stratified import EdgeAttestation, StratifiedEvent, StratifiedEvidenceStore

BUNDLE_ID = 51
ITEM_ID = "3B"
SPINE_DIGEST = "32693f54e0ce7ad3d401ceb752ae4723b0eee251f6d49b95cdfb26a6201971e9"
GBENCH_CANONICAL_COMMIT = "d3e42831"

BEAM_CONVERSATION_COUNT = 20
QUESTIONS_PER_CONVERSATION = 20
BEAM_TOTAL_QUESTIONS = BEAM_CONVERSATION_COUNT * QUESTIONS_PER_CONVERSATION  # 400
BEAM_CONTEXT_TOKENS = 131_072


def _sha256_json(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# 1. BEAM 128K 20-Conversation (400-Question) Data Model & Ingestion
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BeamMemoryTurn:
    """Single memory-bearing turn within a BEAM 128K conversation."""

    event_id: str
    conversation_id: str
    turn_index: int
    entity: str
    slot: str
    claim_value: str
    text: str
    valid_time: int
    tx_time: int
    authority_tier: int
    role: str  # "root_claim", "derivative_copy", "independent_corroboration", "superseder", "audit_note", "withdrawal"
    parent_event_id: Optional[str] = None
    edge_type: Optional[str] = None
    lineage_id: Optional[str] = None


@dataclass(frozen=True)
class BeamEvalQuestion:
    """Single evaluation question within one of the 20 BEAM 128K conversations."""

    question_id: str
    conversation_id: str
    question_index: int
    query_text: str
    entity: str
    slot: str
    target_valid_time: int
    target_tx_time: int
    expected_derivative_status: str  # "RETRACTED" when derivative-only root is withdrawn
    expected_independent_status: str  # "ADMIT" when independent corroboration survives
    expected_claim_value: str


@dataclass(frozen=True)
class BeamConversation:
    """One BEAM 128K conversation cluster containing turns + 20 correlated questions."""

    conversation_id: str
    tenant_id: str
    token_length_est: int
    derivative_turns: Tuple[BeamMemoryTurn, ...]
    independent_turns: Tuple[BeamMemoryTurn, ...]
    questions: Tuple[BeamEvalQuestion, ...]


def build_deterministic_beam_128k_fixture(*, seed: int = 5101) -> Tuple[BeamConversation, ...]:
    """Generate deterministic 20-conversation (400-question) BEAM 128K paired-withdrawal fixture."""
    rng = np.random.default_rng(seed)
    conversations: List[BeamConversation] = []
    domains = ("finance", "clinical", "engineering", "compliance", "operations")

    for conv_idx in range(BEAM_CONVERSATION_COUNT):
        conv_id = f"beam128k_conv_{conv_idx:02d}"
        tenant_id = f"beam_tenant_{conv_idx:02d}"
        dom = domains[conv_idx % len(domains)]
        entity = f"{dom}_project_{conv_idx:02d}"
        slot_main = "primary_status"
        slot_aux = "milestone_budget"
        lineage = f"{tenant_id}:{entity}:{slot_main}"

        # Construct realistic multi-session revision history per conversation
        root_claim = f"approved_phase_{conv_idx % 4 + 1}"
        aux_claim_v1 = f"budget_{100 + conv_idx * 5}k"
        aux_claim_v2 = f"budget_{125 + conv_idx * 5}k"

        e_root = BeamMemoryTurn(
            event_id=f"{conv_id}:root",
            conversation_id=conv_id,
            turn_index=1,
            entity=entity,
            slot=slot_main,
            claim_value=root_claim,
            text=f"Session 1 source briefing for {entity}: {slot_main} is {root_claim}.",
            valid_time=10,
            tx_time=10,
            authority_tier=2,
            role="root_claim",
        )
        e_copy1 = BeamMemoryTurn(
            event_id=f"{conv_id}:copy_summary_1",
            conversation_id=conv_id,
            turn_index=2,
            entity=entity,
            slot=slot_main,
            claim_value=root_claim,
            text=f"Session 3 recap citing Session 1: {entity} {slot_main} remains {root_claim}.",
            valid_time=20,
            tx_time=20,
            authority_tier=2,
            role="derivative_copy",
            parent_event_id=e_root.event_id,
            edge_type="COPY_OF",
        )
        e_copy2 = BeamMemoryTurn(
            event_id=f"{conv_id}:copy_summary_2",
            conversation_id=conv_id,
            turn_index=3,
            entity=entity,
            slot=slot_main,
            claim_value=root_claim,
            text=f"Session 6 executive digest derived from Session 3: {entity} {slot_main} is {root_claim}.",
            valid_time=30,
            tx_time=30,
            authority_tier=2,
            role="derivative_copy",
            parent_event_id=e_copy1.event_id,
            edge_type="DERIVES_CURRENT",
        )
        e_aux_v1 = BeamMemoryTurn(
            event_id=f"{conv_id}:aux_v1",
            conversation_id=conv_id,
            turn_index=4,
            entity=entity,
            slot=slot_aux,
            claim_value=aux_claim_v1,
            text=f"Session 4 initial estimate for {entity}: {slot_aux} = {aux_claim_v1}.",
            valid_time=25,
            tx_time=25,
            authority_tier=1,
            role="root_claim",
        )
        e_aux_v2 = BeamMemoryTurn(
            event_id=f"{conv_id}:aux_v2",
            conversation_id=conv_id,
            turn_index=5,
            entity=entity,
            slot=slot_aux,
            claim_value=aux_claim_v2,
            text=f"Session 7 updated ledger for {entity}: {slot_aux} = {aux_claim_v2} superseding v1.",
            valid_time=35,
            tx_time=35,
            authority_tier=2,
            role="superseder",
            parent_event_id=e_aux_v1.event_id,
            edge_type="VERSION_LINEAGE_SUPERSEDE",
            lineage_id=f"{tenant_id}:{entity}:{slot_aux}",
        )
        e_audit = BeamMemoryTurn(
            event_id=f"{conv_id}:audit_note",
            conversation_id=conv_id,
            turn_index=6,
            entity=entity,
            slot="audit_log",
            claim_value="logged",
            text=f"Session 8 compliance audit pointer referencing {e_root.event_id}.",
            valid_time=40,
            tx_time=40,
            authority_tier=2,
            role="audit_note",
            parent_event_id=e_root.event_id,
            edge_type="CITES_AUDIT",
        )
        e_withdraw = BeamMemoryTurn(
            event_id=f"{conv_id}:withdraw_root",
            conversation_id=conv_id,
            turn_index=7,
            entity=entity,
            slot=slot_main,
            claim_value="WITHDRAWN",
            text=f"Session 10 formal retraction withdrawing {e_root.event_id} due to upstream source error.",
            valid_time=50,
            tx_time=50,
            authority_tier=3,
            role="withdrawal",
            parent_event_id=e_root.event_id,
            edge_type="REFUTE",
            lineage_id=lineage,
        )
        e_indep = BeamMemoryTurn(
            event_id=f"{conv_id}:independent_corrob",
            conversation_id=conv_id,
            turn_index=3,
            entity=entity,
            slot=slot_main,
            claim_value=root_claim,
            text=f"Session 5 independent verification by external auditor confirming {entity} {slot_main} = {root_claim}.",
            valid_time=28,
            tx_time=28,
            authority_tier=2,
            role="independent_corroboration",
        )

        derivative_turns = (e_root, e_copy1, e_copy2, e_aux_v1, e_aux_v2, e_audit, e_withdraw)
        independent_turns = (e_root, e_copy1, e_indep, e_aux_v1, e_aux_v2, e_audit, e_withdraw)

        questions: List[BeamEvalQuestion] = []
        for q_idx in range(QUESTIONS_PER_CONVERSATION):
            if q_idx < 14:
                # 14 withdrawal-sensitive questions on slot_main after withdrawal tx_time=50
                questions.append(
                    BeamEvalQuestion(
                        question_id=f"{conv_id}:q{q_idx:02d}",
                        conversation_id=conv_id,
                        question_index=q_idx,
                        query_text=f"After session 10 retraction, what is the current {slot_main} for {entity} (probe {q_idx})?",
                        entity=entity,
                        slot=slot_main,
                        target_valid_time=60,
                        target_tx_time=60,
                        expected_derivative_status="RETRACTED",
                        expected_independent_status="ADMIT",
                        expected_claim_value=root_claim,
                    )
                )
            elif q_idx < 17:
                # 3 historical as-of questions prior to withdrawal (valid_time=32, tx_time=32)
                questions.append(
                    BeamEvalQuestion(
                        question_id=f"{conv_id}:q{q_idx:02d}",
                        conversation_id=conv_id,
                        question_index=q_idx,
                        query_text=f"As of session 6 (t=32), what was the {slot_main} for {entity} (probe {q_idx})?",
                        entity=entity,
                        slot=slot_main,
                        target_valid_time=32,
                        target_tx_time=32,
                        expected_derivative_status="ADMIT",
                        expected_independent_status="ADMIT",
                        expected_claim_value=root_claim,
                    )
                )
            else:
                # 3 supersession control questions on slot_aux (both histories must return aux_claim_v2)
                questions.append(
                    BeamEvalQuestion(
                        question_id=f"{conv_id}:q{q_idx:02d}",
                        conversation_id=conv_id,
                        question_index=q_idx,
                        query_text=f"What is the latest superseded {slot_aux} for {entity} (probe {q_idx})?",
                        entity=entity,
                        slot=slot_aux,
                        target_valid_time=60,
                        target_tx_time=60,
                        expected_derivative_status="ADMIT",
                        expected_independent_status="ADMIT",
                        expected_claim_value=aux_claim_v2,
                    )
                )

        tok_est = int(BEAM_CONTEXT_TOKENS - rng.integers(0, 4096))
        conversations.append(
            BeamConversation(
                conversation_id=conv_id,
                tenant_id=tenant_id,
                token_length_est=tok_est,
                derivative_turns=derivative_turns,
                independent_turns=independent_turns,
                questions=tuple(questions),
            )
        )

    return tuple(conversations)


def ingest_beam_128k_conversations(
    source_path: Optional[Path] = None,
    *,
    historical_fingerprints: Optional[Mapping[str, Sequence[str]]] = None,
    ledger: Optional[BurnLedger] = None,
    seed: int = 5101,
) -> Tuple[Tuple[BeamConversation, ...], Dict[str, Any]]:
    """Ingest 20-conversation (400-question) BEAM 128K dataset with schema & burn_ledger checks."""
    if source_path is None:
        convs = build_deterministic_beam_128k_fixture(seed=seed)
        source_kind = "deterministic_local_drycheck_fixture"
    else:
        path = Path(source_path)
        if not path.is_file():
            raise FileNotFoundError(f"BEAM 128K dataset file not found: {path}")
        raw = json.loads(path.read_text(encoding="utf-8"))
        conv_rows = raw["conversations"] if isinstance(raw, dict) else raw
        parsed: List[BeamConversation] = []
        for row in conv_rows:
            parsed.append(
                BeamConversation(
                    conversation_id=str(row["conversation_id"]),
                    tenant_id=str(row.get("tenant_id", row["conversation_id"])),
                    token_length_est=int(row.get("token_length_est", BEAM_CONTEXT_TOKENS)),
                    derivative_turns=tuple(BeamMemoryTurn(**t) for t in row["derivative_turns"]),
                    independent_turns=tuple(BeamMemoryTurn(**t) for t in row["independent_turns"]),
                    questions=tuple(BeamEvalQuestion(**q) for q in row["questions"]),
                )
            )
        convs = tuple(parsed)
        source_kind = "live_beam_128k_json"

    n_convs = len(convs)
    n_questions = sum(len(c.questions) for c in convs)
    if n_convs != BEAM_CONVERSATION_COUNT:
        raise ValueError(f"BEAM 128K requires exactly {BEAM_CONVERSATION_COUNT} conversations; got {n_convs}")
    if n_questions != BEAM_TOTAL_QUESTIONS:
        raise ValueError(f"BEAM 128K requires exactly {BEAM_TOTAL_QUESTIONS} questions; got {n_questions}")

    burn_verified = False
    if historical_fingerprints is not None:
        doc_texts = [t.text for c in convs for t in c.derivative_turns if t.text]
        query_texts = [q.query_text for c in convs for q in c.questions if q.query_text]
        active_ledger = build_burn_ledger() if ledger is None else ledger
        assert_corpus_unburned(
            "beam-128k-gbench",
            doc_texts=doc_texts,
            query_texts=query_texts,
            historical_fingerprints=historical_fingerprints,
            ledger=active_ledger,
        )
        burn_verified = True

    receipt = {
        "source_kind": source_kind,
        "gbench_commit": GBENCH_CANONICAL_COMMIT,
        "conversation_clusters": n_convs,
        "total_questions": n_questions,
        "questions_per_conversation": sorted({len(c.questions) for c in convs}),
        "mean_token_length_est": float(np.mean([c.token_length_est for c in convs])),
        "burn_ledger_verified": burn_verified,
    }
    return convs, receipt


# ---------------------------------------------------------------------------
# 2. Conversation-Level Cluster Bootstrap (n=20, Never Inflating to N=400)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ClusterBootstrapResult:
    """Cluster-bootstrapped CI at the conversation level (`n_clusters=20`)."""

    sampling_unit: str
    n_clusters: int
    total_questions: int
    mean: float
    lcb_95: float
    ucb_95: float
    cluster_sd: float
    intraclass_correlation_icc: float
    naive_question_n_inflated: bool
    cluster_ci_width: float
    naive_question_ci_width_diagnostic: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def _compute_icc(cluster_arrays: Sequence[np.ndarray]) -> float:
    """One-way random-effects ICC(1) measuring within-conversation correlation."""
    k = len(cluster_arrays)
    if k < 2:
        return 0.0
    sizes = np.asarray([a.size for a in cluster_arrays], dtype=np.float64)
    n_tot = float(np.sum(sizes))
    if n_tot <= k:
        return 0.0
    all_vals = np.concatenate(cluster_arrays)
    grand_mean = float(np.mean(all_vals))
    ss_between = float(sum(a.size * (float(np.mean(a)) - grand_mean) ** 2 for a in cluster_arrays))
    ss_within = float(sum(float(np.sum((a - float(np.mean(a))) ** 2)) for a in cluster_arrays))
    ms_between = ss_between / max(k - 1, 1)
    ms_within = ss_within / max(n_tot - k, 1.0)
    k0 = (n_tot - float(np.sum(sizes ** 2)) / n_tot) / max(k - 1, 1)
    denom = ms_between + (k0 - 1.0) * ms_within
    if denom <= 1e-12:
        return 0.0
    return float(max(-1.0, min(1.0, (ms_between - ms_within) / denom)))


def conversation_cluster_bootstrap(
    scores_by_conversation: Mapping[str, Sequence[float]],
    *,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 5102,
) -> ClusterBootstrapResult:
    """Cluster bootstrap resampling strictly over the `n=20` conversations (never 400 questions)."""
    conv_ids = sorted(scores_by_conversation.keys())
    n_clusters = len(conv_ids)
    if n_clusters < 2:
        raise ValueError("Cluster bootstrap requires at least 2 conversation clusters")

    arrays = [np.asarray(list(scores_by_conversation[cid]), dtype=np.float64) for cid in conv_ids]
    if any(a.size == 0 for a in arrays):
        raise ValueError("Every conversation cluster must have at least 1 question score")

    cluster_means = np.asarray([float(np.mean(a)) for a in arrays], dtype=np.float64)
    cluster_sizes = np.asarray([a.size for a in arrays], dtype=np.float64)
    total_q = int(np.sum(cluster_sizes))
    overall_mean = float(np.sum(cluster_means * cluster_sizes) / np.sum(cluster_sizes))
    cluster_sd = float(np.std(cluster_means, ddof=1))

    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n_clusters, size=(int(n_resamples), n_clusters))
    sampled_sums = np.sum(cluster_means[idx] * cluster_sizes[idx], axis=1)
    sampled_counts = np.sum(cluster_sizes[idx], axis=1)
    boot_means = sampled_sums / sampled_counts
    lcb = float(np.quantile(boot_means, alpha / 2.0))
    ucb = float(np.quantile(boot_means, 1.0 - alpha / 2.0))

    # Diagnostic-only naive question-level CI width to quantify anti-inflation protection
    flat = np.concatenate(arrays)
    rng_naive = np.random.default_rng(seed + 17)
    naive_idx = rng_naive.integers(0, flat.size, size=(int(n_resamples), flat.size))
    naive_means = flat[naive_idx].mean(axis=1)
    naive_width = float(np.quantile(naive_means, 1.0 - alpha / 2.0) - np.quantile(naive_means, alpha / 2.0))

    return ClusterBootstrapResult(
        sampling_unit="conversation_cluster",
        n_clusters=n_clusters,
        total_questions=total_q,
        mean=overall_mean,
        lcb_95=lcb,
        ucb_95=ucb,
        cluster_sd=cluster_sd,
        intraclass_correlation_icc=_compute_icc(arrays),
        naive_question_n_inflated=False,
        cluster_ci_width=float(ucb - lcb),
        naive_question_ci_width_diagnostic=naive_width,
    )


def paired_conversation_cluster_bootstrap(
    arm_a_by_conversation: Mapping[str, Sequence[float]],
    arm_b_by_conversation: Mapping[str, Sequence[float]],
    *,
    n_resamples: int = 2000,
    alpha: float = 0.05,
    seed: int = 5103,
) -> ClusterBootstrapResult:
    """Paired conversation-cluster bootstrap of `arm_a - arm_b` over `n=20` conversations."""
    if sorted(arm_a_by_conversation.keys()) != sorted(arm_b_by_conversation.keys()):
        raise ValueError("arm_a and arm_b must contain identical conversation cluster keys")
    deltas_by_conv: Dict[str, List[float]] = {}
    for cid in sorted(arm_a_by_conversation.keys()):
        a_vals = list(arm_a_by_conversation[cid])
        b_vals = list(arm_b_by_conversation[cid])
        if len(a_vals) != len(b_vals):
            raise ValueError(f"Mismatched question count in conversation {cid}")
        deltas_by_conv[cid] = [float(a) - float(b) for a, b in zip(a_vals, b_vals)]
    return conversation_cluster_bootstrap(deltas_by_conv, n_resamples=n_resamples, alpha=alpha, seed=seed)


# ---------------------------------------------------------------------------
# 3. Stateful Checkpoint Replay & 100% Incremental == Full EK5 Prefix Parity
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReplayCheckpoint:
    """Serialized stateful checkpoint bound to TRA-1 state identity and EK5 canonical digest."""

    checkpoint_id: str
    conversation_id: str
    branch_name: str
    prefix_step: int
    event_count: int
    attestation_count: int
    ek5_canonical_digest: str
    tra1_state_digest: str
    events_snapshot: Tuple[Dict[str, Any], ...]
    attestations_snapshot: Tuple[Dict[str, Any], ...]


def _turn_to_ek5(turn: BeamMemoryTurn, tenant_id: str) -> Tuple[StratifiedEvent, Optional[EdgeAttestation]]:
    ev = StratifiedEvent(
        event_id=turn.event_id,
        entity=turn.entity,
        slot=turn.slot,
        claim_value=turn.claim_value,
        text=turn.text,
        valid_time=turn.valid_time,
        tx_time=turn.tx_time,
        tenant=tenant_id,
        domain="general",
        authority_tier=turn.authority_tier,
    )
    edge: Optional[EdgeAttestation] = None
    if turn.parent_event_id and turn.edge_type:
        edge = EdgeAttestation(
            attestation_id=f"att:{turn.event_id}:{turn.edge_type}:{turn.parent_event_id}",
            src_id=turn.event_id,
            dst_id=turn.parent_event_id,
            edge_type=turn.edge_type,
            valid_time=turn.valid_time,
            tx_time=turn.tx_time,
            lineage_id=turn.lineage_id,
        )
    return ev, edge


def _snapshot_checkpoint(
    store: StratifiedEvidenceStore,
    conversation_id: str,
    branch_name: str,
    prefix_step: int,
    canonical_digest: str,
) -> ReplayCheckpoint:
    ev_rows = tuple(
        {
            "event_id": ev.event_id,
            "entity": ev.entity,
            "slot": ev.slot,
            "claim_value": ev.claim_value,
            "text": ev.text,
            "valid_time": ev.valid_time,
            "tx_time": ev.tx_time,
            "valid_end": ev.valid_end,
            "tenant": ev.tenant,
            "domain": ev.domain,
            "authority_tier": ev.authority_tier,
        }
        for ev in sorted(store.events.values(), key=lambda e: e.event_id)
    )
    att_rows = tuple(
        {
            "attestation_id": att.attestation_id,
            "src_id": att.src_id,
            "dst_id": att.dst_id,
            "edge_type": att.edge_type,
            "valid_time": att.valid_time,
            "tx_time": att.tx_time,
            "lineage_id": att.lineage_id,
            "active": att.active,
        }
        for att in sorted(store.attestations.values(), key=lambda a: a.attestation_id)
        if att.active
    )
    contract = TRA1StateIdentityContract(
        base_weights_sha256=sha256_digest(f"beam128k:{conversation_id}:{branch_name}".encode("utf-8")),
        adapter_weights_sha256=EMPTY_ADAPTER_SHA256,
        config_sha256=_sha256_json({"bundle_id": BUNDLE_ID, "prefix_step": prefix_step}),
        tokenizer_sha256=sha256_digest(b"beam128k-tokenizer-v1"),
        kv_or_state_dtype="float16",
        rope_freq_base=1_000_000.0,
        rope_scaling_hash=sha256_digest(b"yarn-128k"),
    )
    return ReplayCheckpoint(
        checkpoint_id=f"{conversation_id}:{branch_name}:ckpt_{prefix_step:02d}",
        conversation_id=conversation_id,
        branch_name=branch_name,
        prefix_step=prefix_step,
        event_count=len(ev_rows),
        attestation_count=len(att_rows),
        ek5_canonical_digest=canonical_digest,
        tra1_state_digest=contract.digest(),
        events_snapshot=ev_rows,
        attestations_snapshot=att_rows,
    )


def restore_store_from_checkpoint(checkpoint: ReplayCheckpoint) -> StratifiedEvidenceStore:
    """Restore a `StratifiedEvidenceStore` deterministically from a saved stateful checkpoint."""
    restored = StratifiedEvidenceStore(strict_dag_supersede=False, buffer_pending=False)
    events = [StratifiedEvent(**row) for row in checkpoint.events_snapshot]
    atts = [EdgeAttestation(**row) for row in checkpoint.attestations_snapshot]
    restored.apply_mutation(events, atts)
    return restored


def _resolve_ek5_question(
    store: StratifiedEvidenceStore,
    tenant_id: str,
    question: BeamEvalQuestion,
) -> Tuple[str, Optional[str]]:
    """Evaluate a BEAM question at `(target_valid_time, target_tx_time)` ignoring withdrawal sentinels."""
    snap = store.as_of(question.target_valid_time, question.target_tx_time)
    admitted = [
        ev
        for eid, ev in snap.events.items()
        if ev.tenant == tenant_id
        and ev.entity == question.entity
        and ev.slot == question.slot
        and ev.claim_value != "WITHDRAWN"
        and snap.eligibility.get(eid) == "ADMIT"
    ]
    if admitted:
        winner = max(admitted, key=lambda e: (e.authority_tier, e.tx_time, e.event_id))
        return "ADMIT", winner.claim_value
    contested = [
        ev
        for eid, ev in snap.events.items()
        if ev.tenant == tenant_id
        and ev.entity == question.entity
        and ev.slot == question.slot
        and ev.claim_value != "WITHDRAWN"
        and snap.eligibility.get(eid) == "CONTESTED"
    ]
    if contested:
        return "CONTESTED", None
    return "RETRACTED", None


def _run_branch_with_prefix_parity(
    conversation: BeamConversation,
    turns: Sequence[BeamMemoryTurn],
    branch_name: str,
    *,
    checkpoint_step: int = 3,
) -> Dict[str, Any]:
    """Ingest turns incrementally, checking 100% incremental==full EK5 parity at EVERY prefix."""
    inc_store = StratifiedEvidenceStore(strict_dag_supersede=False, buffer_pending=False)
    prefix_checks: List[Dict[str, Any]] = []
    saved_checkpoint: Optional[ReplayCheckpoint] = None
    remaining_after_ckpt: List[BeamMemoryTurn] = []

    for step_idx, turn in enumerate(turns, start=1):
        ev, edge = _turn_to_ek5(turn, conversation.tenant_id)
        inc_store.apply_mutation([ev], [edge] if edge is not None else [])

        inc_trace = export_canonical_ek5_audit_trace(inc_store)
        fresh_store = reference_fresh_rebuild(inc_store)
        fresh_trace = export_canonical_ek5_audit_trace(fresh_store)
        cmp = compare_with_fresh_rebuild(inc_store, inc_trace)

        digest_match = inc_trace.digest() == fresh_trace.digest()
        zero_diffs = (
            len(cmp.semantic_diffs) == 0
            and len(cmp.canonical_reason_diffs) == 0
            and len(cmp.canonical_pos_cyc_diffs) == 0
            and len(cmp.canonical_semantic_mismatches) == 0
        )
        prefix_checks.append(
            {
                "prefix_step": step_idx,
                "event_id": turn.event_id,
                "inc_digest": inc_trace.digest(),
                "full_digest": fresh_trace.digest(),
                "exact_parity": bool(digest_match and zero_diffs),
                "semantic_diffs": len(cmp.semantic_diffs),
                "canonical_reason_diffs": len(cmp.canonical_reason_diffs),
                "canonical_pos_cyc_diffs": len(cmp.canonical_pos_cyc_diffs),
            }
        )

        if step_idx == min(checkpoint_step, len(turns)):
            saved_checkpoint = _snapshot_checkpoint(
                inc_store, conversation.conversation_id, branch_name, step_idx, inc_trace.digest()
            )
        elif saved_checkpoint is not None:
            remaining_after_ckpt.append(turn)

    # Stateful checkpoint replay verification: restore from checkpoint + replay remaining turns
    checkpoint_parity = False
    checkpoint_digest_final = ""
    if saved_checkpoint is not None:
        resumed = restore_store_from_checkpoint(saved_checkpoint)
        for rem_turn in remaining_after_ckpt:
            ev, edge = _turn_to_ek5(rem_turn, conversation.tenant_id)
            resumed.apply_mutation([ev], [edge] if edge is not None else [])
        resumed_trace = export_canonical_ek5_audit_trace(resumed)
        final_trace = export_canonical_ek5_audit_trace(inc_store)
        checkpoint_digest_final = resumed_trace.digest()
        checkpoint_parity = resumed_trace.digest() == final_trace.digest()

    # Evaluate all 20 questions on this branch
    question_scores: List[float] = []
    question_details: List[Dict[str, Any]] = []
    for q in conversation.questions:
        pred_status, pred_val = _resolve_ek5_question(inc_store, conversation.tenant_id, q)
        expected_status = (
            q.expected_derivative_status if branch_name == "derivative_only" else q.expected_independent_status
        )
        if expected_status == "RETRACTED":
            correct = pred_status == "RETRACTED"
        else:
            correct = pred_status == expected_status and pred_val == q.expected_claim_value
        question_scores.append(1.0 if correct else 0.0)
        question_details.append(
            {
                "question_id": q.question_id,
                "expected_status": expected_status,
                "predicted_status": pred_status,
                "expected_claim": q.expected_claim_value,
                "predicted_claim": pred_val,
                "correct": bool(correct),
            }
        )

    exact_prefixes = sum(1 for row in prefix_checks if row["exact_parity"])
    return {
        "branch_name": branch_name,
        "conversation_id": conversation.conversation_id,
        "total_prefixes": len(prefix_checks),
        "exact_parity_prefixes": exact_prefixes,
        "prefix_parity_rate": exact_prefixes / float(max(len(prefix_checks), 1)),
        "checkpoint_id": saved_checkpoint.checkpoint_id if saved_checkpoint else None,
        "checkpoint_tra1_digest": saved_checkpoint.tra1_state_digest if saved_checkpoint else None,
        "checkpoint_replay_exact": bool(checkpoint_parity),
        "final_ek5_digest": prefix_checks[-1]["inc_digest"] if prefix_checks else "",
        "checkpoint_final_digest": checkpoint_digest_final,
        "question_scores": question_scores,
        "question_accuracy": float(np.mean(question_scores)) if question_scores else 0.0,
        "question_details": question_details,
        "prefix_checks": prefix_checks,
    }


def _evaluate_sql_challenger_branch(
    conversation: BeamConversation,
    turns: Sequence[BeamMemoryTurn],
    branch_name: str,
) -> Dict[str, Any]:
    """Run independent SQL ChallengerBitemporalLedger (recursive_cte + closure_table) on one branch."""
    with ChallengerBitemporalLedger(SQLiteChallengerBackend()) as ledger:
        for turn in turns:
            entity_key = f"{turn.entity}:{turn.slot}"
            if turn.role == "withdrawal" and turn.parent_event_id:
                ledger.append(
                    LedgerRecordEnvelope(
                        tenant_id=conversation.tenant_id,
                        record_id=turn.parent_event_id,
                        entity_key=entity_key,
                        authority_rank=turn.authority_tier,
                        valid_from=turn.valid_time,
                        valid_to=None,
                        tx_time=turn.tx_time,
                        op_kind=OpKind.WITHDRAW,
                        claim_value="",
                    )
                )
            else:
                ledger.append(
                    LedgerRecordEnvelope(
                        tenant_id=conversation.tenant_id,
                        record_id=turn.event_id,
                        entity_key=entity_key,
                        authority_rank=turn.authority_tier,
                        valid_from=turn.valid_time,
                        valid_to=None,
                        tx_time=turn.tx_time,
                        op_kind=OpKind.ASSERT,
                        claim_value=turn.claim_value,
                    )
                )
                if turn.parent_event_id and turn.edge_type in {"COPY_OF", "DERIVES_CURRENT"}:
                    ledger.add_edge(
                        LedgerEdgeEnvelope(
                            tenant_id=conversation.tenant_id,
                            edge_id=f"e:{turn.event_id}:{turn.parent_event_id}",
                            src_id=turn.event_id,
                            dst_id=turn.parent_event_id,
                            tx_time=turn.tx_time,
                        )
                    )
                elif turn.parent_event_id and turn.edge_type == "VERSION_LINEAGE_SUPERSEDE":
                    ledger.append(
                        LedgerRecordEnvelope(
                            tenant_id=conversation.tenant_id,
                            record_id=turn.parent_event_id,
                            entity_key=entity_key,
                            authority_rank=turn.authority_tier,
                            valid_from=turn.valid_time,
                            valid_to=None,
                            tx_time=turn.tx_time,
                            op_kind=OpKind.WITHDRAW,
                            claim_value="",
                        )
                    )

        cte_scores: List[float] = []
        mode_agreements = 0
        for q in conversation.questions:
            entity_key = f"{q.entity}:{q.slot}"
            res_cte = ledger.query_as_of(
                conversation.tenant_id, entity_key, q.target_valid_time, q.target_tx_time, mode="recursive_cte"
            )
            res_clo = ledger.query_as_of(
                conversation.tenant_id, entity_key, q.target_valid_time, q.target_tx_time, mode="closure_table"
            )
            if res_cte.digest() == res_clo.digest():
                mode_agreements += 1
            expected_status = (
                q.expected_derivative_status if branch_name == "derivative_only" else q.expected_independent_status
            )
            if expected_status == "RETRACTED":
                ok = res_cte.status in {"RETRACTED", "ABSENT"}
            else:
                ok = res_cte.status == "RESOLVED" and res_cte.winner_claim_value == q.expected_claim_value
            cte_scores.append(1.0 if ok else 0.0)

    return {
        "sql_accuracy": float(np.mean(cte_scores)) if cte_scores else 0.0,
        "sql_question_scores": cte_scores,
        "sql_mode_parity_exact": mode_agreements == len(conversation.questions),
    }


def _evaluate_recency_baseline_branch(
    conversation: BeamConversation,
    turns: Sequence[BeamMemoryTurn],
    branch_name: str,
) -> List[float]:
    """Naive recency-only memory baseline that ignores derivative retraction propagation."""
    scores: List[float] = []
    for q in conversation.questions:
        visible = [
            t
            for t in turns
            if t.entity == q.entity
            and t.slot == q.slot
            and t.role != "withdrawal"
            and t.tx_time <= q.target_tx_time
            and t.valid_time <= q.target_valid_time
        ]
        if not visible:
            pred_status, pred_val = "RETRACTED", None
        else:
            latest = max(visible, key=lambda t: (t.tx_time, t.turn_index))
            pred_status, pred_val = "ADMIT", latest.claim_value
        expected_status = (
            q.expected_derivative_status if branch_name == "derivative_only" else q.expected_independent_status
        )
        if expected_status == "RETRACTED":
            ok = pred_status == "RETRACTED"
        else:
            ok = pred_status == expected_status and pred_val == q.expected_claim_value
        scores.append(1.0 if ok else 0.0)
    return scores


def run_beam_128k_paired_replay(
    conversations: Sequence[BeamConversation],
    *,
    n_bootstrap: int = 2000,
    seed: int = 5102,
) -> Dict[str, Any]:
    """Run full 20-conversation paired withdrawal replay + conversation cluster bootstrap + EK5 prefix parity."""
    if len(conversations) != BEAM_CONVERSATION_COUNT:
        raise ValueError(f"Expected {BEAM_CONVERSATION_COUNT} conversations, got {len(conversations)}")

    ek5_deriv_by_conv: Dict[str, List[float]] = {}
    ek5_indep_by_conv: Dict[str, List[float]] = {}
    ek5_joint_by_conv: Dict[str, List[float]] = {}
    sql_joint_by_conv: Dict[str, List[float]] = {}
    recency_joint_by_conv: Dict[str, List[float]] = {}

    total_prefixes = 0
    exact_parity_prefixes = 0
    all_checkpoints_exact = True
    all_sql_modes_exact = True
    per_conversation_summaries: List[Dict[str, Any]] = []

    for conv in conversations:
        deriv_res = _run_branch_with_prefix_parity(conv, conv.derivative_turns, "derivative_only")
        indep_res = _run_branch_with_prefix_parity(conv, conv.independent_turns, "independent_corroboration")

        sql_deriv = _evaluate_sql_challenger_branch(conv, conv.derivative_turns, "derivative_only")
        sql_indep = _evaluate_sql_challenger_branch(conv, conv.independent_turns, "independent_corroboration")
        rec_deriv = _evaluate_recency_baseline_branch(conv, conv.derivative_turns, "derivative_only")
        rec_indep = _evaluate_recency_baseline_branch(conv, conv.independent_turns, "independent_corroboration")

        total_prefixes += deriv_res["total_prefixes"] + indep_res["total_prefixes"]
        exact_parity_prefixes += deriv_res["exact_parity_prefixes"] + indep_res["exact_parity_prefixes"]
        all_checkpoints_exact = all_checkpoints_exact and deriv_res["checkpoint_replay_exact"] and indep_res["checkpoint_replay_exact"]
        all_sql_modes_exact = all_sql_modes_exact and sql_deriv["sql_mode_parity_exact"] and sql_indep["sql_mode_parity_exact"]

        ek5_deriv_by_conv[conv.conversation_id] = deriv_res["question_scores"]
        ek5_indep_by_conv[conv.conversation_id] = indep_res["question_scores"]
        ek5_joint_by_conv[conv.conversation_id] = [
            1.0 if (d == 1.0 and i == 1.0) else 0.0
            for d, i in zip(deriv_res["question_scores"], indep_res["question_scores"])
        ]
        sql_joint_by_conv[conv.conversation_id] = [
            1.0 if (d == 1.0 and i == 1.0) else 0.0
            for d, i in zip(sql_deriv["sql_question_scores"], sql_indep["sql_question_scores"])
        ]
        recency_joint_by_conv[conv.conversation_id] = [
            1.0 if (d == 1.0 and i == 1.0) else 0.0
            for d, i in zip(rec_deriv, rec_indep)
        ]

        per_conversation_summaries.append(
            {
                "conversation_id": conv.conversation_id,
                "token_length_est": conv.token_length_est,
                "question_count": len(conv.questions),
                "ek5_derivative_accuracy": deriv_res["question_accuracy"],
                "ek5_independent_accuracy": indep_res["question_accuracy"],
                "ek5_paired_joint_accuracy": float(np.mean(ek5_joint_by_conv[conv.conversation_id])),
                "sql_paired_joint_accuracy": float(np.mean(sql_joint_by_conv[conv.conversation_id])),
                "recency_paired_joint_accuracy": float(np.mean(recency_joint_by_conv[conv.conversation_id])),
                "derivative_prefix_parity_rate": deriv_res["prefix_parity_rate"],
                "independent_prefix_parity_rate": indep_res["prefix_parity_rate"],
                "checkpoint_replay_exact": bool(
                    deriv_res["checkpoint_replay_exact"] and indep_res["checkpoint_replay_exact"]
                ),
                "derivative_final_ek5_digest": deriv_res["final_ek5_digest"],
                "independent_final_ek5_digest": indep_res["final_ek5_digest"],
            }
        )

    deriv_boot = conversation_cluster_bootstrap(ek5_deriv_by_conv, n_resamples=n_bootstrap, seed=seed)
    indep_boot = conversation_cluster_bootstrap(ek5_indep_by_conv, n_resamples=n_bootstrap, seed=seed + 1)
    joint_boot = conversation_cluster_bootstrap(ek5_joint_by_conv, n_resamples=n_bootstrap, seed=seed + 2)
    ek5_vs_recency_boot = paired_conversation_cluster_bootstrap(
        ek5_joint_by_conv, recency_joint_by_conv, n_resamples=n_bootstrap, seed=seed + 3
    )
    ek5_vs_sql_boot = paired_conversation_cluster_bootstrap(
        ek5_joint_by_conv, sql_joint_by_conv, n_resamples=n_bootstrap, seed=seed + 4
    )

    parity_rate = exact_parity_prefixes / float(max(total_prefixes, 1))
    summary_payload = {
        "bundle_id": BUNDLE_ID,
        "item": ITEM_ID,
        "conversation_clusters": len(conversations),
        "total_questions": sum(len(c.questions) for c in conversations),
        "total_prefixes_checked": total_prefixes,
        "exact_parity_prefixes": exact_parity_prefixes,
        "ek5_prefix_parity_rate": parity_rate,
        "ek5_prefix_mismatches": total_prefixes - exact_parity_prefixes,
        "checkpoint_replay_all_exact": bool(all_checkpoints_exact),
        "sql_challenger_mode_parity_exact": bool(all_sql_modes_exact),
        "cluster_bootstrap": {
            "derivative_only": deriv_boot.to_dict(),
            "independent_corroboration": indep_boot.to_dict(),
            "paired_joint": joint_boot.to_dict(),
            "ek5_minus_recency_baseline": ek5_vs_recency_boot.to_dict(),
            "ek5_minus_sql_challenger": ek5_vs_sql_boot.to_dict(),
        },
        "conversations": per_conversation_summaries,
    }
    summary_payload["replay_sha256"] = _sha256_json(
        {
            "total_prefixes_checked": total_prefixes,
            "exact_parity_prefixes": exact_parity_prefixes,
            "checkpoint_replay_all_exact": all_checkpoints_exact,
            "digests": [
                (c["conversation_id"], c["derivative_final_ek5_digest"], c["independent_final_ek5_digest"])
                for c in per_conversation_summaries
            ],
        }
    )
    return summary_payload


__all__ = [
    "BEAM_CONTEXT_TOKENS",
    "BEAM_CONVERSATION_COUNT",
    "BEAM_TOTAL_QUESTIONS",
    "BUNDLE_ID",
    "BeamConversation",
    "BeamEvalQuestion",
    "BeamMemoryTurn",
    "ClusterBootstrapResult",
    "GBENCH_CANONICAL_COMMIT",
    "ITEM_ID",
    "QUESTIONS_PER_CONVERSATION",
    "ReplayCheckpoint",
    "SPINE_DIGEST",
    "build_deterministic_beam_128k_fixture",
    "conversation_cluster_bootstrap",
    "ingest_beam_128k_conversations",
    "paired_conversation_cluster_bootstrap",
    "restore_store_from_checkpoint",
    "run_beam_128k_paired_replay",
]
