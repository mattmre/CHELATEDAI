"""Unit tests for `chelated_epistemic_proposer_v2.py` (Stratum-1 authority dominance & entity/slot grounding)."""

from __future__ import annotations

import numpy as np

from chelated_epistemic_proposer_v2 import AuthorityDominatedExocortexV2, ground_and_clean_claims
from chelated_raw_text_exocortex import ExtractedClaimRecord, RawDocumentEnvelope


def test_ground_and_clean_claims_fixes_entity_value_swap_and_meta_slots() -> None:
    docs = [
        RawDocumentEnvelope(
            "d0",
            "Formal technical filing (Party A) on INC-SPLIT-CLOCK-705: asserts fault is GPS_ANTENNA_MULTIPATH_SPOOF.",
            12,
            12,
            "postmortem_final",
        ),
        RawDocumentEnvelope(
            "d1",
            "Formal counter-filing (Party B) on INC-SPLIT-CLOCK-705: asserts fault is BOUNDARY_CLOCK_FPGA_PLL_UNLOCK.",
            12,
            13,
            "vendor_advisory",
        ),
        RawDocumentEnvelope(
            "e0",
            "Forum mirror on INC-SPLIT-CLOCK-705: operators cite GPS_ANTENNA_MULTIPATH_SPOOF for ptp_grandmaster_fault.",
            14,
            15,
            "community_forum",
        ),
    ]
    # Simulate the exact D1 v1 3B errors: `technical_filing` meta-slot + `value` copied into `entity`!
    raw = [
        ExtractedClaimRecord("d0", "inc-split-clock-705", "technical_filing", "GPS_ANTENNA_MULTIPATH_SPOOF", "REFUTE", "{}"),
        ExtractedClaimRecord("d1", "boundary_clock_fpga_pll_unlock", "technical_counter-filing", "BOUNDARY_CLOCK_FPGA_PLL_UNLOCK", "REFUTE", "{}"),
        ExtractedClaimRecord("e0", "inc-split-clock-705", "ptp_grandmaster_fault", "GPS_ANTENNA_MULTIPATH_SPOOF", "ASSERT", "{}"),
    ]
    embs = np.array([[1.0, 0.0], [0.98, 0.1], [0.95, 0.2]], dtype=np.float32)
    cleaned = ground_and_clean_claims(docs, raw, embs)
    assert cleaned[0].entity == "inc-split-clock-705"
    assert cleaned[1].entity == "inc-split-clock-705"
    assert cleaned[0].slot == "ptp_grandmaster_fault"
    assert cleaned[1].slot == "ptp_grandmaster_fault"


def test_authority_dominated_exocortex_prevents_low_tier_echo_from_breaking_peer_deadlock() -> None:
    docs = [
        RawDocumentEnvelope(
            "p_a",
            "Filing A on CVE-2026-8843-NGINX-HTTP3 refutes rival and states verdict is PROTOCOL_AUTH_BYPASS_DEFECT.",
            12,
            12,
            "independent_security_lab",  # Tier 3
        ),
        RawDocumentEnvelope(
            "p_b",
            "Filing B on CVE-2026-8843-NGINX-HTTP3 refutes A and states verdict is RFC9001_EXPECTED_IDEMPOTENCY.",
            12,
            13,
            "vendor_advisory",  # Tier 3
        ),
        RawDocumentEnvelope(
            "echo_late",
            "Social post on CVE-2026-8843-NGINX-HTTP3 repeats PROTOCOL_AUTH_BYPASS_DEFECT.",
            14,
            16,
            "social_repost",  # Tier 1 echo arriving AFTER p_a and p_b!
        ),
    ]
    raw = [
        ExtractedClaimRecord("p_a", "cve-2026-8843-nginx-http3", "technical_filing", "PROTOCOL_AUTH_BYPASS_DEFECT", "REFUTE", "{}"),
        ExtractedClaimRecord("p_b", "cve-2026-8843-nginx-http3", "vulnerability_verdict", "RFC9001_EXPECTED_IDEMPOTENCY", "REFUTE", "{}"),
        ExtractedClaimRecord("echo_late", "cve-2026-8843-nginx-http3", "published_day", "PROTOCOL_AUTH_BYPASS_DEFECT", "ASSERT", "{}"),
    ]
    embs = np.array([[1.0, 0.0], [0.99, 0.05], [0.97, 0.1]], dtype=np.float32)
    exo = AuthorityDominatedExocortexV2(docs, raw, embs, embs)
    out = exo.select_context_for_query(np.array([0.92, 0.91, 0.96], dtype=np.float32), "What is CVE-2026-8843-NGINX-HTTP3 verdict?")
    assert out["status"] == "CONTESTED_ABSTAIN"
    assert out["selected_ids"] == []
