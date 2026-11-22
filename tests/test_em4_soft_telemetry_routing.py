"""Unit tests for EM-4 (`em4_soft_telemetry_routing.py`: Carbon's Soft Telemetry Fallback & Verified Echo Routing)."""

from __future__ import annotations

import numpy as np

from ek5_v3_stratified import StratifiedEvent, StratifiedEvidenceStore
from em4_soft_telemetry_routing import (
    is_recap_framed,
    route_soft_telemetry_top_k,
    strip_recap_and_speaker,
)


def test_strip_recap_and_speaker_handles_nested_wrappers() -> None:
    raw = "User: Periodic assistant digest citing prior session: Caroline: I live in Chicago."
    assert is_recap_framed(raw)
    assert strip_recap_and_speaker(raw) == "I live in Chicago."
    raw2 = "Briefing 3: Caroline: Contemporary dance is so expressive."
    assert is_recap_framed(raw2)
    assert strip_recap_and_speaker(raw2) == "Contemporary dance is so expressive."


def test_clean_moderate_cosine_turns_are_never_hard_dropped() -> None:
    """Reproduces R4 LoCoMo `loc_1_39` failure mode: cos ~ 0.76, low Jaccard -> BOTH preserved!"""
    units = [
        {"unit_id": "u0", "session_id": "s1", "ts": 100.0, "text": "Gina: I went to dance practice today with friends."},
        {"unit_id": "u1", "session_id": "s2", "ts": 200.0, "text": "Gina: Contemporary dance is so expressive and graceful!"},
        {"unit_id": "u2", "session_id": "s3", "ts": 300.0, "text": "Gina: We bought new sneakers for rehearsal."},
    ]
    # Construct unit embeddings with cos(u0, u1) = 0.78 (would have been hard-dropped by M4e cos >= 0.72!)
    e0 = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    e1 = np.array([0.78, np.sqrt(1.0 - 0.78**2), 0.0], dtype=np.float32)
    e2 = np.array([0.30, 0.20, np.sqrt(1.0 - 0.13)], dtype=np.float32)
    embs = np.stack([e0, e1, e2], axis=0)
    q_emb = np.array([0.85, 0.5268, 0.0], dtype=np.float32)
    bm25 = np.array([4.0, 3.8, 1.0], dtype=np.float64)

    res = route_soft_telemetry_top_k(units, embs, q_emb, bm25, top_k=2, mode="M4f_soft_telemetry_fallback")
    assert set(res.selected_indices) == {0, 1}
    assert res.collapsed_echo_clusters == {}
    # Unrelated/complementary clean turns must NOT get noisy conflict tags
    assert res.status_tags == {}


def test_six_stale_recap_echoes_collapse_and_despoof_timestamp_for_update() -> None:
    """Reproduces LongMemEval Regime B: 6 intervening echoes of old fact before true update."""
    units = [
        {"unit_id": "orig_old", "session_id": "s1", "ts": 100.0, "text": "User: I currently live in Chicago near the lake."},
    ]
    for i in range(6):
        units.append({
            "unit_id": f"echo_{i}",
            "session_id": f"recap_{i}",
            "ts": 200.0 + 50.0 * i,
            "text": "User: Periodic assistant digest citing prior session: I currently live in Chicago near the lake.",
        })
    units.append({
        "unit_id": "true_upd",
        "session_id": "s10",
        "ts": 800.0,
        "text": "User: Big life update — I moved to Seattle last week!",
    })
    # Embeddings: orig + 6 echoes sit near [1, 0]; true_upd has cos = 0.68 with [1, 0]
    embs_list = [np.array([1.0, 0.0, 0.0], dtype=np.float32)]
    for i in range(6):
        v = np.array([0.99, 0.05 * ((i % 2) * 2 - 1), 0.02], dtype=np.float32)
        embs_list.append(v / np.linalg.norm(v))
    v_upd = np.array([0.68, np.sqrt(1.0 - 0.68**2), 0.0], dtype=np.float32)
    embs_list.append(v_upd)
    embs = np.stack(embs_list, axis=0)
    q_emb = np.array([0.90, np.sqrt(1.0 - 0.81), 0.0], dtype=np.float32)
    bm25 = np.array([5.0] + [4.9] * 6 + [4.2], dtype=np.float64)

    for mode in ("M4f_soft_telemetry_fallback", "M4g_ek5_soft_telemetry_recency"):
        res = route_soft_telemetry_top_k(units, embs, q_emb, bm25, top_k=3, mode=mode)
        # All 7 Chicago turns collapse into 1 canonical representative, freeing top_k for `true_upd` (idx 7)!
        assert 7 in res.selected_indices
        assert len(res.selected_indices) == 2
        assert len(res.collapsed_echo_clusters) == 1
        rep_idx = next(iter(res.collapsed_echo_clusters.keys()))
        assert "EARLIER RECORD (echoed 7x" in res.status_tags[rep_idx]
        assert res.status_tags[7] == "[EM-3 TELEMETRY: LATEST DATED UPDATE] "


def test_explicit_ek5_v3_refutation_triggers_hard_drop() -> None:
    """Hard-drop IS enforced when EK5 v3 holds a verified explicit REFUTE edge."""
    st = StratifiedEvidenceStore()
    st.apply_mutation(
        [
            StratifiedEvent("u_bad", entity="pump", slot="psi", claim_value="40psi", authority_tier=2),
            StratifiedEvent("u_corr", entity="pump", slot="psi", claim_value="65psi", authority_tier=3),
        ],
        [("u_corr", "u_bad", "REFUTE"), ("u_corr", "u_bad", "SUPERSEDE")],
    )
    units = [
        {"unit_id": "u_bad", "session_id": "s1", "ts": 10.0, "text": "Pump pressure is 40psi."},
        {"unit_id": "u_corr", "session_id": "s2", "ts": 20.0, "text": "Verified audit: pump pressure is 65psi."},
    ]
    embs = np.array([[1.0, 0.0], [0.85, np.sqrt(1.0 - 0.85**2)]], dtype=np.float32)
    q_emb = np.array([1.0, 0.0], dtype=np.float32)
    bm25 = np.array([5.0, 4.5], dtype=np.float64)

    res = route_soft_telemetry_top_k(units, embs, q_emb, bm25, top_k=5, ek5_store=st)
    assert res.selected_indices == (1,)
    assert res.explicit_ek5_dropped == (0,)
