"""Unit and regression tests for `ChelatedConversationalV4`, `virgin_conversational_holdout`, and K2 quantiles."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pytest

from chelated_conversational_v4 import (
    BGE_DIMENSIONS,
    EARLIER,
    LATEST,
    ChelatedConversationalV4,
    ConversationTurn,
    flat_bgem3_token_matched_context_v4,
)
from virgin_conversational_holdout import (
    DOMAINS,
    ITEM_CATEGORIES,
    LOCOMO_FORBIDDEN_SPEAKERS,
    build_virgin_conversational_holdout,
    verify_lineage_disjointness,
    virgin_suite_digest,
)
from wave6_common import paired_bootstrap_wave6


class KeywordEmbedder:
    """Deterministic test embedder assigning orthogonal unit basis axes by topic keyword."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    def encode(self, texts: list[str] | tuple[str, ...]) -> np.ndarray:
        self.calls.extend(texts)
        vecs = np.zeros((len(texts), BGE_DIMENSIONS), dtype=np.float32)
        for idx, raw in enumerate(texts):
            lo = raw.lower()
            if "irrigation" in lo or "greenhouse" in lo:
                vecs[idx, 0] = 1.0
            elif "port 9477" in lo or "healthcheck" in lo:
                vecs[idx, 1] = 1.0
            elif "kayak" in lo:
                vecs[idx, 2] = 1.0
            elif "pgbouncer" in lo or "connections" in lo:
                vecs[idx, 3] = 1.0
            else:
                vecs[idx, 4 + (idx % 16)] = 1.0
        return vecs


class KeywordReranker:
    """Deterministic cross-encoder double for unit testing top-36 reranking and clean parity."""

    def __init__(self, boost_substring: str = "") -> None:
        self.boost_substring = boost_substring.lower()
        self.scored_batches: list[tuple[str, tuple[str, ...]]] = []

    def score_pairs(self, query: str, passages: list[str] | tuple[str, ...]) -> np.ndarray:
        self.scored_batches.append((query, tuple(passages)))
        scores = np.zeros(len(passages), dtype=np.float64)
        q_words = set(query.lower().split())
        for i, p in enumerate(passages):
            p_lo = p.lower()
            overlap = sum(1 for w in q_words if w in p_lo)
            scores[i] = 0.25 * overlap
            if self.boost_substring and self.boost_substring in p_lo:
                scores[i] += 6.0
        return scores


def _mk_turn(uid: str, sid: str, day: int, claim: str, *, speaker: str = "User", pos: int = 0) -> ConversationTurn:
    return ConversationTurn(
        turn_id=uid,
        session_id=sid,
        date=f"2025-03-{day:02d} 10:00",
        timestamp=float(day * 86400 + pos),
        text=f"{speaker}: {claim}",
        claim_text=claim,
        speaker=speaker,
    )


def test_pre_budget_echo_collapse_prevents_shortlist_starvation() -> None:
    """Guardrail K2.1: EK5 complete-linkage echo collapse runs BEFORE shortlist truncation & session budgeting."""
    turns: list[ConversationTurn] = [
        _mk_turn("orig", "s01", 1, "I set my daily greenhouse irrigation target to 20 liters."),
    ]
    # 10 cross-session echoes flooding Stage-1 ranks ahead of distinct evidence
    echo_phrasings = (
        "My daily greenhouse irrigation target is set to 20 liters.",
        "Right now my daily greenhouse irrigation target sits at 20 liters.",
        "For the greenhouse, my daily irrigation target remains 20 liters.",
        "Each day my greenhouse irrigation target stays at 20 liters.",
        "We keep my daily greenhouse irrigation target at 20 liters.",
    )
    for e_i in range(10):
        turns.append(
            _mk_turn(
                f"echo_{e_i}",
                f"s_echo_{e_i}",
                2 + e_i,
                echo_phrasings[e_i % len(echo_phrasings)],
            )
        )
    # Distinct relevant needle turn pushed down to Stage-1 rank 12 by the 10 echoes
    turns.append(
        _mk_turn(
            "needle_9477",
            "s_needle",
            15,
            "Greenhouse sensor healthcheck listener is bound to port 9477.",
        )
    )
    for d_i in range(4):
        turns.append(_mk_turn(f"dist_{d_i}", f"s_dist_{d_i}", 16 + d_i, f"We bought apples at market {d_i}."))

    reranker = KeywordReranker(boost_substring="port 9477")
    engine = ChelatedConversationalV4(turns, KeywordEmbedder(), reranker)
    # Stage-1 ranking where 10 echoes occupy ranks 1..11 and `needle_9477` sits at rank 12 (> pool_size=6)
    flooded_stage1_order = tuple(range(len(turns)))

    # With pool_size=6: WITHOUT pre-budget echo collapse (`epistemic=False`), top-6 shortlist is 100% echoes of 20L!
    no_ek5 = engine.retrieve(
        "Which greenhouse port is healthcheck bound to?",
        epistemic=False,
        anchor_count=3,
        window_radius=0,
        pool_size=6,
        context_chars=2400,
        ranked=flooded_stage1_order,
    )
    with_ek5 = engine.retrieve(
        "Which greenhouse port is healthcheck bound to?",
        epistemic=True,
        anchor_count=3,
        window_radius=0,
        pool_size=6,
        context_chars=2400,
        ranked=flooded_stage1_order,
    )
    assert "needle_9477" not in [t.turn.turn_id for t in no_ek5.turns]
    assert "needle_9477" in [t.turn.turn_id for t in with_ek5.turns]
    assert with_ek5.ranked_turn_ids[0] == "needle_9477"
    assert len(with_ek5.echo_clusters) >= 1
    assert sum(len(c) - 1 for c in with_ek5.echo_clusters) >= 7
    assert with_ek5.echo_clusters[0][0] == "orig"


def test_supersede_partner_preservation_and_chronological_order() -> None:
    """Guardrail K2.1: SUPERSEDE pairs are preserved across reranking and always rendered chronologically."""
    earlier = _mk_turn("t_old", "s1", 1, "I set my daily greenhouse irrigation target to 20 liters.")
    echo_1 = _mk_turn("t_echo1", "s2", 3, "My daily greenhouse irrigation target is set to 20 liters.")
    echo_2 = _mk_turn("t_echo2", "s3", 5, "Right now my daily greenhouse irrigation target sits at 20 liters.")
    updated = _mk_turn("t_new", "s9", 12, "I changed my daily greenhouse irrigation target from 20 liters to 35 liters.")

    # Reranker intentionally scores the older turn higher than the newer turn to test partner preservation!
    reranker = KeywordReranker(boost_substring="set my daily")
    engine = ChelatedConversationalV4([echo_2, updated, echo_1, earlier], KeywordEmbedder(), reranker)

    recall = engine.retrieve(
        "What is the daily greenhouse irrigation target?",
        epistemic=True,
        anchor_count=1,
        window_radius=0,
        pool_size=4,
        context_chars=2400,
    )
    assert recall.echo_clusters == (("t_old", "t_echo1", "t_echo2"),)
    assert recall.supersessions == (("t_new", "t_old"),)
    rendered_ids = [rt.turn.turn_id for rt in recall.turns]
    assert rendered_ids == ["t_old", "t_new"]
    assert recall.turns[0].status == EARLIER
    assert recall.turns[1].status == LATEST
    assert recall.context.index(EARLIER) < recall.context.index(LATEST)


def test_inactive_gate_exact_byte_parity_with_reranker() -> None:
    """When `gate_active == False`, `epistemic=True` must match `epistemic=False` byte-for-byte."""
    turns = [
        _mk_turn("t1", "s1", 1, "Nadia: Did we check the dead-letter queue?", speaker="Nadia"),
        _mk_turn("t2", "s1", 1, "Kieran: Healthchecks are bound to port 9477.", speaker="Kieran", pos=1),
        _mk_turn("t3", "s2", 4, "Nadia: Load test looked stable all afternoon.", speaker="Nadia"),
        _mk_turn("t4", "s3", 7, "Kieran: Archived the old runbook.", speaker="Kieran"),
    ]
    reranker = KeywordReranker(boost_substring="port 9477")
    engine = ChelatedConversationalV4(turns, KeywordEmbedder(), reranker)
    clean = engine.retrieve(
        "Which port did Kieran bind healthchecks to?",
        epistemic=False,
        anchor_count=2,
        window_radius=1,
        context_chars=2000,
    )
    gated = engine.retrieve(
        "Which port did Kieran bind healthchecks to?",
        epistemic=True,
        anchor_count=2,
        window_radius=1,
        context_chars=2000,
    )
    assert not gated.gate_active
    assert clean == gated
    assert clean.context == gated.context
    assert "[Dialogue Cast: Kieran, Nadia | Question Subject: Kieran" in gated.header_line


def test_complete_linkage_blocks_transitive_bridge_in_v4() -> None:
    turns = [
        _mk_turn("a", "s1", 1, "The greenhouse has violet plants along the west wall."),
        _mk_turn("b", "s2", 2, "The greenhouse has flowers along the west wall."),
        _mk_turn("c", "s3", 3, "The greenhouse has white plants along the west wall."),
    ]

    def fake_classify(a: str, b: str) -> tuple[str, str]:
        if "violet" in a + b and "white" in a + b:
            return "PRESERVE_CONFLICT", "CONFLICT_PROPER_ENTITY"
        return "ECHO_COLLAPSE", "COMPATIBLE_ECHO"

    engine = ChelatedConversationalV4(turns, KeywordEmbedder(), reranker=None)
    with patch("chelated_conversational_v4.classify_high_similarity_pair", side_effect=fake_classify):
        res = engine.retrieve("greenhouse west wall", ranked=(0, 1, 2), anchor_count=3, pool_size=3)
    assert res.echo_clusters == (("a", "b"),)
    assert set(res.ranked_turn_ids) == {"a", "c"}


def test_flat_bgem3_token_matched_respects_budget_and_truncation() -> None:
    long_body = "detailed assistant notes " * 120
    turns = [
        _mk_turn(f"t{i}", f"s{i}", i + 1, f"Greenhouse log {i}: {long_body}")
        for i in range(8)
    ]
    engine = ChelatedConversationalV4(turns, KeywordEmbedder(), reranker=None)
    ctx, ids, sids = flat_bgem3_token_matched_context_v4(
        engine, "Greenhouse log", context_chars=1800, turn_max_chars=400
    )
    assert len(ids) >= 3
    assert len(ids) == len(sids)
    assert len(ctx) <= 1800


def test_virgin_holdout_suite_invariants_and_ek5_echo_collapse() -> None:
    convs = build_virgin_conversational_holdout()
    assert len(convs) == 24
    assert sum(len(c.questions) for c in convs) == 120

    domain_counts = {d: sum(1 for c in convs if c.domain == d) for d in DOMAINS}
    assert domain_counts == {
        "software_engineering": 5,
        "health_fitness": 5,
        "travel": 5,
        "personal_finance": 5,
        "operations": 4,
    }

    for conv in convs:
        assert 8 <= len(conv.session_dates) <= 12
        assert [q.category for q in conv.questions] == list(ITEM_CATEGORIES)
        for sp in conv.speakers:
            assert sp.casefold() not in LOCOMO_FORBIDDEN_SPEAKERS

    disjoint_report = verify_lineage_disjointness(convs)
    assert disjoint_report["disjoint_verified"] is True
    assert disjoint_report["n_unique_virgin_speakers"] == 48

    d1 = virgin_suite_digest(convs)
    d2 = virgin_suite_digest()
    assert d1 == d2 and len(d1) == 64

    # Verify vconv_00 KU echo flood collapses echoes and detects SUPERSEDE in ChelatedConversationalV4
    c0 = convs[0]
    ku_q = [q for q in c0.questions if q.category == "knowledge_update_echo_flood"][0]
    eng = ChelatedConversationalV4(c0.turns, KeywordEmbedder(), KeywordReranker("pgbouncer"))
    rec = eng.retrieve(ku_q.question, epistemic=True, anchor_count=6, window_radius=0, pool_size=36)
    assert rec.gate_active is True
    assert len(rec.echo_clusters) >= 1
    assert len(rec.supersessions) >= 1
    ret_ids = {rt.turn.turn_id for rt in rec.turns}
    assert set(ku_q.evidence_turn_ids).issubset(ret_ids)


def test_wave6_distinct_quantiles_k2_guardrail() -> None:
    """Guardrail K2.3: Assert p1.25 < p2.5 < p5.0 when bootstrap variance > 0."""
    rng = np.random.default_rng(123)
    scores_a = {f"v_{i:03d}": float(rng.choice([0.0, 1.0], p=[0.22, 0.78])) for i in range(120)}
    scores_b = {f"v_{i:03d}": float(rng.choice([0.0, 1.0], p=[0.48, 0.52])) for i in range(120)}
    clusters = {f"v_{i:03d}": f"vconv_{i // 5:02d}" for i in range(120)}

    res = paired_bootstrap_wave6(
        scores_a,
        scores_b,
        n_bootstrap=20_000,
        seed=42,
        expected_ids=sorted(scores_a),
        cluster_by_id=clusters,
    )
    assert res["boot_std"] > 0.0
    assert res["p0125_bonf2"] < res["lcb_975"] < res["lcb_95"] < res["mean_delta"]
    assert res["tango_score_lcb975_bonf2"] < res["tango_score_lcb95"]
    assert res["n_clusters"] == 24
    assert res["cluster_p0125_bonf2"] < res["cluster_lcb_975_p025"] < res["cluster_lcb_95_p050"]


if __name__ == "__main__":
    pytest.main([__file__, "-q"])
