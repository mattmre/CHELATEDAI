"""Unit tests for `chelated_raw_text_exocortex.py` (label-free raw-text WFS + causal streaming ZeroMu)."""

from __future__ import annotations

import numpy as np

from chelated_raw_text_exocortex import (
    CausalStreamingZeroMuBridge,
    ExtractedClaimRecord,
    LabelFreeChelatedExocortex,
    RawDocumentEnvelope,
    align_slot_clusters,
    parse_extracted_claim_json,
)
from spherical_migration_bridge import CenteredProcrustesBridge, l2_normalize


def test_parse_extracted_claim_json_handles_fences_and_fallback() -> None:
    rec = parse_extracted_claim_json(
        "d1",
        'Sure! {"entity": "CVE-2024-3094", "slot": "Safe Release", "value": "5.6.1+really5.4.5-1", "stance": "supersede"}',
    )
    assert rec.entity == "cve-2024-3094"
    assert rec.slot == "safe_release"
    assert rec.claim_value == "5.6.1+really5.4.5-1"
    assert rec.stance == "SUPERSEDE"

    fb = parse_extracted_claim_json("d2", "not json at all")
    assert fb.doc_id == "d2"
    assert fb.stance == "ASSERT"


def test_causal_streaming_zeromu_bridge_is_strictly_causal() -> None:
    rng = np.random.default_rng(7)
    x_cal = l2_normalize(rng.normal(size=(240, 32)).astype(np.float32))
    ortho, _ = np.linalg.qr(rng.normal(size=(64, 64)).astype(np.float32))
    proj = ortho[:32, :]
    y_cal = l2_normalize(x_cal @ proj + 0.15)
    br = CenteredProcrustesBridge.fit(x_cal[:180], y_cal[:180], cal_x=x_cal[180:], cal_y=y_cal[180:])

    docs_tgt = y_cal[:100]
    stream_a = CausalStreamingZeroMuBridge(br, docs_tgt, ema_alpha=0.1)
    stream_b = CausalStreamingZeroMuBridge(br, docs_tgt, ema_alpha=0.1)

    q0 = x_cal[200]
    q1_a = x_cal[201]
    q1_b = -x_cal[201]  # Wildly different future query!

    s0_a = stream_a.score_single_query(q0)
    _ = stream_a.score_single_query(q1_a)

    s0_b = stream_b.score_single_query(q0)
    _ = stream_b.score_single_query(q1_b)

    # Causality invariant: query t=0 score MUST be bit-identical regardless of future query t=1!
    np.testing.assert_allclose(s0_a, s0_b, rtol=0.0, atol=0.0)


def test_label_free_exocortex_handles_rumor_laundering_echoes_and_contested_abstain() -> None:
    docs = [
        # Entity 1: Rumor + tier-2->tier-1 derived echo + official debunk + late low-tier rumor-laundering attempt
        RawDocumentEnvelope("r0", "Analyst chat claims CVE-1 status is EXPLOITABLE.", 2, 2, "internal_ops_chat"),
        RawDocumentEnvelope("r1", "Forum repost says CVE-1 status is EXPLOITABLE.", 3, 3, "community_forum"),
        RawDocumentEnvelope("a1", "Official NVD bulletin debunks EXPLOITABLE rumor; CVE-1 status is SAFE.", 8, 8, "official_nvd_advisory"),
        RawDocumentEnvelope("r2", "Late blog insists CVE-1 status is EXPLOITABLE despite denial.", 10, 10, "unverified_blog"),
        # Entity 2: Equal-tier mutual peer conflict -> must trigger label-free CONTESTED_ABSTAIN!
        RawDocumentEnvelope("c1", "Lab A refutes vendor and states CVE-2 verdict is RCE_CONFIRMED.", 5, 5, "independent_security_lab"),
        RawDocumentEnvelope("c2", "Vendor advisory refutes Lab A and states CVE-2 verdict is NON_EXPLOITABLE.", 5, 6, "vendor_advisory"),
    ]
    claims = align_slot_clusters([
        ExtractedClaimRecord("r0", "cve_1", "exposure_status", "EXPLOITABLE", "ASSERT", "{}"),
        ExtractedClaimRecord("r1", "cve_1", "status_exposure", "EXPLOITABLE", "ASSERT", "{}"),
        ExtractedClaimRecord("a1", "cve_1", "exposure_status", "SAFE", "DEBUNK_RUMOR", "{}"),
        ExtractedClaimRecord("r2", "cve_1", "exposure_status", "EXPLOITABLE", "ASSERT", "{}"),
        ExtractedClaimRecord("c1", "cve_2", "verdict", "RCE_CONFIRMED", "REFUTE", "{}"),
        ExtractedClaimRecord("c2", "cve_2", "verdict", "NON_EXPLOITABLE", "REFUTE", "{}"),
    ])
    embs = np.array([
        [1.0, 0.0, 0.0],
        [0.99, 0.05, 0.0],
        [0.95, 0.10, 0.0],
        [0.98, 0.02, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.98, 0.1],
    ], dtype=np.float32)

    exo = LabelFreeChelatedExocortex(docs, claims, embs)
    # Query 1: CVE-1 -> ONLY `a1` ("SAFE") should survive; `r0`, `r1`, and late `r2` are blocked/refuted!
    res1 = exo.select_context_for_query(np.array([0.95, 0.94, 0.88, 0.96, 0.1, 0.1]), "What is CVE-1 exposure status?")
    assert res1["status"] == "ADMIT_OK"
    assert res1["selected_ids"] == ["a1"]

    # Query 2: CVE-2 -> Equal-tier mutual refutation must deterministically yield CONTESTED_ABSTAIN with zero family labels!
    res2 = exo.select_context_for_query(np.array([0.1, 0.1, 0.1, 0.1, 0.95, 0.93]), "What is CVE-2 verdict?")
    assert res2["status"] == "CONTESTED_ABSTAIN"
    assert res2["selected_ids"] == []
