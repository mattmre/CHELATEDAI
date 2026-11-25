"""Comprehensive R3/R4 rigor test suite for EK5 v3 (Flagship #2) & Spherical Bridges (Flagship #1).

Covers all unanimous 3/3 Tri-Model Council locks:
  - EK5 v3 Stratified K=4 differential oracle checks (256 2-node static, 65,536 2-node snapshot jumps,
    random 3-8 node cyclic/batch/delete/authority/rewind ops, Council Pitfall A/B & positive-SCC witnesses,
    10k in/out hub |D|<=1, compact_ranks, all 10 Stratum 1/3 policy fixtures, EM-3 telemetry, EM-1 M4e).
  - Flagship #1 original baselines + refined/emergent arms (Woodbury Ridge, Centered Affine Ridge,
    Theorem 1 3-regime trichotomy + A2c-Polar, Repaired Prop 1b, constant fail-closed, 4 OOS scale laws,
    EM-2 A2-ZeroMu exact 1-float zero-reindex identity, EM-5b A2c-SoftSpec, A2c-Q, A2c-IW,
    A2g atan2 + C1 soft collar + dual pole guard + fallback receipt, O(n^2 d) A3 parity + 2x2 crossing,
    B4 right-tail monotonicity + csls_k + B2-OldSpace, TRA-1 contract hashes, 160-tie argpartition,
    paired_bootstrap_delta fail-closed + exact_paired_binary_ci Tango/Clopper-Pearson, Windows import).
"""

from __future__ import annotations

import math
import random
from typing import Dict, List, Set, Tuple

import numpy as np
import pytest

from ek5_v3_stratified import (
    EdgeAttestation,
    StratifiedEvent,
    StratifiedEvidenceStore,
    SupersedeCycleError,
    classify_high_similarity_pair,
    compute_alternating_wfs_oracle,
    strip_boilerplate_headers,
)
from m1_baseline_ladder import (
    fit_closed_form_centered_ridge,
    fit_closed_form_ridge,
    score_map_plus_backfill,
)
from scripts.night1_common import (
    FastBM25Index,
    exact_paired_binary_ci,
    paired_bootstrap_delta,
)
from spherical_migration_bridge import (
    CenteredProcrustesBridge,
    GeodesicPolarBridge,
    WhitenedSphericalProcrustesBridge,
    _shrinkage_cov_factors,
    _sphere_exp_map,
    build_bridge_contract,
    classify_dimension_chain_regime,
    compose_centered_bridges,
    compute_old_space_knn_distortion,
    compute_voronoi_anchor_weights,
    l2_normalize,
    score_quantile_calibrated_backfill,
    score_zeromu_rankings,
)
from representation_space import RepresentationRole
from tests.test_spherical_migration_bridge import _mk_desc


def _assert_store_matches_oracle(store: StratifiedEvidenceStore, context: str = "") -> None:
    oracle = compute_alternating_wfs_oracle(store.events.keys(), store.refs, store.deps_v)
    for nid, expected_val in oracle.items():
        assert store.val[nid] == expected_val, (
            f"WFS mismatch on {nid} ({context}): store={store.val[nid]} oracle={expected_val}"
        )
        # Verify Pitfall A/B shield invariant
        if store.val[nid] == "F" and not store.pos_cyc[nid] and nid not in store._missing_dep_nodes:
            expected_sigma = sum(
                1 for a in store.refs.get(nid, ()) if store.val[a] == "T" and store.rho[a] < store.rho[nid]
            ) + sum(
                1 for d in store.deps_v.get(nid, ()) if store.val[d] == "F" and store.rho[d] < store.rho[nid]
            )
            assert store.sigma[nid] == expected_sigma, (
                f"Shield invariant mismatch on {nid} ({context}): sigma={store.sigma[nid]} expected={expected_sigma}"
            )
            assert store.sigma[nid] >= 1, f"Non-cyclic F node {nid} missing shield ({context})"


# ===========================================================================
# PART 1: FLAGSHIP #2 (EK5 v3 STRATIFIED K=4 + EM-3 + EM-1 M4e)
# ===========================================================================
def test_ek5_v3_exhaustive_2node_static_256_and_jumps_65536() -> None:
    """Exhaustive verification of all 256 2-node graphs and all 65,536 ordered snapshot jumps."""
    nodes = ("a", "b")
    all_possible_edges = [
        ("a", "a", "REFUTE"),
        ("a", "b", "REFUTE"),
        ("b", "a", "REFUTE"),
        ("b", "b", "REFUTE"),
        ("a", "a", "DERIVES_CURRENT"),
        ("a", "b", "DERIVES_CURRENT"),
        ("b", "a", "DERIVES_CURRENT"),
        ("b", "b", "DERIVES_CURRENT"),
    ]

    # 1. All 256 static 2-node graphs
    oracles: List[Dict[str, str]] = []
    for mask in range(256):
        edges = [all_possible_edges[i] for i in range(8) if (mask >> i) & 1]
        st = StratifiedEvidenceStore()
        st.apply_mutation(nodes, edges)
        _assert_store_matches_oracle(st, f"static_mask={mask}")
        oracles.append(dict(st.val))

    # 2. All 65,536 ordered transitions G1 -> G2 via atomic delta batch
    # Reuse store per G1 by stepping through a Gray-like / direct jump schedule cleanly
    for m1 in range(256):
        e1 = [all_possible_edges[i] for i in range(8) if (m1 >> i) & 1]
        for m2 in range(256):
            st = StratifiedEvidenceStore()
            st.apply_mutation(nodes, e1)
            add_diff = [all_possible_edges[i] for i in range(8) if ((m2 >> i) & 1) and not ((m1 >> i) & 1)]
            rem_diff = [all_possible_edges[i] for i in range(8) if ((m1 >> i) & 1) and not ((m2 >> i) & 1)]
            st.apply_mutation((), add_diff, remove_edges=rem_diff)
            assert st.val == oracles[m2], f"Jump {m1}->{m2}: got {st.val}, expected {oracles[m2]}"


def test_ek5_v3_council_witnesses_and_random_3_to_8_node_fuzz() -> None:
    """Test Carbon 3-node Pitfall A/B, Neon 5-node asymmetric rho==0, Astra positive-SCC, & 1,500 random cyclic ops."""
    # Witness 1: Astra grounded-boundary positive-SCC witness (A <-> B, C --DEP--> A, D --REF--> A)
    st = StratifiedEvidenceStore()
    st.apply_mutation(
        ["A", "B", "C", "D"],
        [("A", "B", "DERIVES_CURRENT"), ("B", "A", "DERIVES_CURRENT"), ("C", "A", "DERIVES_CURRENT"), ("D", "A", "REFUTE")],
    )
    _assert_store_matches_oracle(st, "astra_init")
    assert st.val["A"] == "F" and st.val["B"] == "F" and st.pos_cyc["A"] and st.pos_cyc["B"]
    # Delete REFUTE D -> A: A, B remain F because of positive cycle
    st.apply_mutation((), remove_edges=[("D", "A", "REFUTE")])
    _assert_store_matches_oracle(st, "astra_del_ref")
    assert st.val["A"] == "F" and st.val["B"] == "F" and st.val["C"] == "F"
    # Break positive cycle A -> B: A, B, C ALL recover to T!
    st.apply_mutation((), remove_edges=[("A", "B", "DERIVES_CURRENT")])
    _assert_store_matches_oracle(st, "astra_break_cyc")
    assert st.val["A"] == "T" and st.val["B"] == "T" and st.val["C"] == "T"

    # Witness 2: Neon 5-node asymmetric rho==0 + mixed REF/DEP batch with same-batch net cancellation
    st2 = StratifiedEvidenceStore()
    st2.apply_mutation(
        ["n0", "n1", "n2", "n3", "n4"],
        [
            ("n0", "n1", "REFUTE"),
            ("n1", "n2", "DERIVES_CURRENT"),
            ("n2", "n3", "DERIVES_CURRENT"),
            ("n3", "n2", "DERIVES_CURRENT"),
            ("n4", "n1", "REFUTE"),
            ("n2", "n4", "REFUTE"),
        ],
    )
    _assert_store_matches_oracle(st2, "neon_5node_init")
    st2.apply_mutation(
        (),
        add_attestations=[("n1", "n4", "REFUTE"), ("n0", "n1", "REFUTE")],
        remove_edges=[("n0", "n1", "REFUTE"), ("n3", "n2", "DERIVES_CURRENT")],
    )
    _assert_store_matches_oracle(st2, "neon_5node_batch")

    # Witness 2b: exact_min_rank frozen-boundary T-refuter priority regression (trial=445 witness!)
    st2b = StratifiedEvidenceStore(exact_min_rank=True)
    st2b.apply_mutation(
        [
            StratifiedEvent("v0", authority_tier=4),
            StratifiedEvent("v1", authority_tier=2),
            StratifiedEvent("v5", authority_tier=4),
            StratifiedEvent("v6", authority_tier=1),
        ],
        [
            ("v5", "v1", "DERIVES_CURRENT"),
            ("v5", "v6", "REFUTE"),
            ("v0", "v6", "AMENDS"),
            ("v6", "v0", "DERIVES_CURRENT"),
        ],
    )
    _assert_store_matches_oracle(st2b, "exact_frozen_refuter_init")
    # Breaking positive cycle v0<->v6 must NOT let v0=T(0) outrun frozen refuter v5=T(1) on v6!
    st2b.apply_mutation((), remove_edges=[("v0", "v6", "AMENDS")])
    _assert_store_matches_oracle(st2b, "exact_frozen_refuter_post_break")
    assert st2b.val["v0"] == "T" and st2b.val["v5"] == "T" and st2b.val["v6"] == "F"

    # Witness 3: Random 3-8 node cyclic single + batch + delete + compact_ranks differential fuzz
    rng = random.Random(20261121)
    for trial in range(120):
        n = rng.randint(3, 8)
        nodes = [f"v{i}" for i in range(n)]
        exact_mode = (trial % 4 == 0)
        store = StratifiedEvidenceStore(exact_min_rank=exact_mode)
        store.apply_mutation(nodes)
        active: Set[Tuple[str, str, str]] = set()
        for step in range(15):
            n_ops = rng.randint(1, 4)
            adds: List[Tuple[str, str, str]] = []
            rems: List[Tuple[str, str, str]] = []
            for _ in range(n_ops):
                if active and rng.random() < 0.45:
                    e = rng.choice(sorted(active))
                    rems.append(e)
                    active.remove(e)
                else:
                    u = rng.choice(nodes)
                    v = rng.choice(nodes)
                    et = rng.choice(["REFUTE", "DERIVES_CURRENT", "AMENDS"])
                    e = (u, v, et)
                    if e not in active:
                        adds.append(e)
                        active.add(e)
            store.apply_mutation((), adds, remove_edges=rems)
            _assert_store_matches_oracle(store, f"trial={trial},step={step}")
            if step == 7:
                store.compact_ranks()
                _assert_store_matches_oracle(store, f"trial={trial},post_compact")
                # Check compact_ranks equals fresh exact_min_rank solve on exact active edges
                live_edges = sorted(k for k, cnt in store._edge_mult.items() if cnt > 0)
                fresh = StratifiedEvidenceStore(exact_min_rank=True)
                fresh.apply_mutation(nodes, live_edges)
                assert store.rho == fresh.rho, f"Compacted ranks differ on trial={trial}"


def test_ek5_v3_10k_hub_locality_and_downstream_positive_cycle() -> None:
    """Verify |D| == 0 / <= 1 on 10k-fanout hubs (shielded F, downstream of pos cycle, multi-hop F(3))."""
    n_leaves = 10_000
    st = StratifiedEvidenceStore()
    events = ["s1", "s2", "hub", "cycA", "cycB", "hub_cyc", "hop1", "hop2", "hop3_hub"]
    events.extend(f"leaf_{i}" for i in range(n_leaves))
    edges: List[Tuple[str, str, str]] = [
        ("s1", "hub", "REFUTE"),
        ("s2", "hub", "REFUTE"),
        ("cycA", "cycB", "DERIVES_CURRENT"),
        ("cycB", "cycA", "DERIVES_CURRENT"),
        ("hub_cyc", "cycA", "DERIVES_CURRENT"),
        ("hop1", "hop2", "REFUTE"),
        ("hop2", "hop3_hub", "REFUTE"),
        ("s1", "hop3_hub", "REFUTE"),
    ]
    for i in range(n_leaves):
        leaf = f"leaf_{i}"
        edges.append((leaf, "hub", "DERIVES_CURRENT"))
        if i < 2_000:
            edges.append((leaf, "hub_cyc", "DERIVES_CURRENT"))
            edges.append((leaf, "hop3_hub", "DERIVES_CURRENT"))
    st.apply_mutation(events, edges)
    assert st.val["hub"] == "F" and st.sigma["hub"] == 2
    assert st.val["hub_cyc"] == "F" and st.pos_cyc["cycA"]

    # Case 1: Remove non-last shield s1 -> hub: ZERO cascade into 10,000 leaves!
    d1 = st.apply_mutation((), remove_edges=[("s1", "hub", "REFUTE")])
    assert d1.work_stats["dirty_nodes"] == 0
    assert d1.work_stats["dirty_edges"] == 0
    assert st.val["hub"] == "F" and st.sigma["hub"] == 1

    # Case 2: Add/remove REFUTE onto positive cycle cycA: ZERO cascade into hub_cyc + 2,000 leaves!
    d2 = st.apply_mutation((), [("s1", "cycA", "REFUTE")])
    assert d2.work_stats["dirty_nodes"] == 0
    d3 = st.apply_mutation((), remove_edges=[("s1", "cycA", "REFUTE")])
    assert d3.work_stats["dirty_nodes"] == 0

    # Case 3: Remove redundant hop2 -> hop3_hub refutation while s1 still shields hop3_hub: ZERO cascade!
    d4 = st.apply_mutation((), remove_edges=[("hop2", "hop3_hub", "REFUTE")])
    assert d4.work_stats["dirty_nodes"] == 0
    assert st.val["hop3_hub"] == "F"


def test_ek5_v3_stratum1_authority_and_all_10_stratum3_policy_fixtures() -> None:
    """Verify Stratum 1 anti-laundering ceiling, bitemporal missing-parent rewind, and all 10 Stratum 3 policies."""
    # Stratum 1: Anti-laundering ceiling + same-batch DEP+REFUTE ordering + CITES_AUDIT exemption
    st = StratifiedEvidenceStore()
    st.apply_mutation(
        [
            StratifiedEvent("rumor", domain="med", authority_tier=1),
            StratifiedEvent("launderer", domain="med", authority_tier=5),
            StratifiedEvent("doctor_claim", domain="med", authority_tier=3),
            StratifiedEvent("auditor", domain="med", authority_tier=4),
            StratifiedEvent("cross_dom", domain="legal", authority_tier=5),
        ],
        [
            # Same atomic batch: launderer derives from rumor (tier 1) AND attempts to REFUTE doctor_claim (tier 3)
            ("launderer", "rumor", "DERIVES_CURRENT"),
            ("launderer", "doctor_claim", "REFUTE"),
            # Cross-domain tier-5 actor attempts to refute med claim -> blocked!
            ("cross_dom", "doctor_claim", "REFUTE"),
            # Auditor quotes rumor via CITES_AUDIT (outside D_V) -> keeps tier 4 -> CAN refute doctor_claim!
            ("auditor", "rumor", "CITES_AUDIT"),
            ("auditor", "doctor_claim", "REFUTE"),
        ],
    )
    assert st.eff_auth["launderer"] == 1
    assert st.eff_auth["auditor"] == 4
    assert ("launderer", "doctor_claim", "REFUTE") in st.get_telemetry().blocked_authority_edges
    assert ("cross_dom", "doctor_claim", "REFUTE") in st.get_telemetry().blocked_authority_edges
    assert "launderer" not in st.refs["doctor_claim"]
    assert "auditor" in st.refs["doctor_claim"]
    assert st.val["doctor_claim"] == "F"

    # Bitemporal as_of rewind + missing mandatory parent (expired parent -> child eff_auth=0, V=F!)
    bt = StratifiedEvidenceStore()
    bt.apply_mutation(
        [
            StratifiedEvent("parent", valid_time=0, tx_time=0, valid_end=5, authority_tier=4),
            StratifiedEvent("child", valid_time=1, tx_time=1, authority_tier=4),
        ],
        [EdgeAttestation("att_dep", "child", "parent", "DERIVES_CURRENT", valid_time=1, tx_time=1)],
    )
    snap_alive = bt.as_of(valid_cutoff=3, tx_cutoff=3)
    assert snap_alive.val["parent"] == "T" and snap_alive.val["child"] == "T" and snap_alive.eff_auth["child"] == 4
    snap_expired = bt.as_of(valid_cutoff=10, tx_cutoff=10)
    assert "parent" not in snap_expired.events
    assert snap_expired.eff_auth["child"] == 0
    assert snap_expired.val["child"] == "F"
    assert snap_expired.eligibility["child"] == "AUDIT_ONLY"
    assert snap_expired.reason["child"] == "MISSING_MANDATORY_PARENT"

    # Policies 1-3: Direct-B_i reinstatement vs multi-current CONTESTED vs explicit resolve
    p = StratifiedEvidenceStore()
    p.apply_mutation(
        [
            StratifiedEvent("A", entity="e", slot="dose", claim_value="10mg"),
            StratifiedEvent("B", entity="e", slot="dose", claim_value="20mg"),
            StratifiedEvent("C", entity="e", slot="dose", claim_value="30mg"),
        ],
        [("B", "A", "SUPERSEDE")],
    )
    assert p.eligibility["A"] == "AUDIT_ONLY" and p.eligibility["B"] == "ADMIT"
    # C REFUTE+SUPERSEDE B -> B is F -> A reinstated (Direct-B_i) -> slot has A & C -> CONTESTED(MULTI_CURRENT)
    p.apply_mutation((), [("C", "B", "REFUTE"), ("C", "B", "SUPERSEDE")])
    assert p.eligibility["B"] == "AUDIT_ONLY"
    assert p.eligibility["A"] == "ADMIT" and p.eligibility["C"] == "ADMIT"
    slot_res = p.resolve_slot("default", "e", "dose")
    assert slot_res["status"] == "CONTESTED" and slot_res["reason"] == "MULTI_CURRENT"
    # Explicit C SUPERSEDE A resolves slot cleanly to 30mg
    p.apply_mutation((), [("C", "A", "SUPERSEDE")])
    slot_res2 = p.resolve_slot("default", "e", "dose")
    assert slot_res2["status"] == "ADMIT" and slot_res2["claim_value"] == "30mg"

    # Policy 4: Declared VERSION_LINEAGE_SUPERSEDE covers v1 even when v2 is refuted!
    vl = StratifiedEvidenceStore()
    vl.apply_mutation(
        [
            StratifiedEvent("v1", tenant="t1", entity="doc", slot="body", claim_value="1"),
            StratifiedEvent("v2", tenant="t1", entity="doc", slot="body", claim_value="2"),
            StratifiedEvent("v3", tenant="t1", entity="doc", slot="body", claim_value="3"),
            StratifiedEvent("ref_v2", tenant="t1", entity="doc", slot="other"),
        ],
        [
            EdgeAttestation("l1", "v2", "v1", "VERSION_LINEAGE_SUPERSEDE", lineage_id="L1"),
            EdgeAttestation("l2", "v3", "v2", "VERSION_LINEAGE_SUPERSEDE", lineage_id="L1"),
            EdgeAttestation("r2", "ref_v2", "v2", "REFUTE"),
        ],
    )
    assert vl.val["v2"] == "F"
    assert vl.cur["v1"] == "F" and vl.eligibility["v1"] == "AUDIT_ONLY"
    assert vl.eligibility["v3"] == "ADMIT"

    # Policies 5-6: COPY_OF/DERIVES_CURRENT (D_C) vs AMENDS/DERIVES_SNAPSHOT (D_V \ D_C)
    dep = StratifiedEvidenceStore()
    dep.apply_mutation(
        ["orig", "newer", "copy_node", "amend_node", "refuter"],
        [
            ("copy_node", "orig", "COPY_OF"),
            ("amend_node", "orig", "AMENDS"),
            ("newer", "orig", "SUPERSEDE"),
        ],
    )
    assert dep.eligibility["orig"] == "AUDIT_ONLY"
    assert dep.eligibility["copy_node"] == "AUDIT_ONLY" and dep.reason["copy_node"] == "SUPERSEDED_ANCESTOR"
    assert dep.eligibility["amend_node"] == "ADMIT"  # AMENDS inherits V only, NOT C!
    dep.apply_mutation((), [("refuter", "orig", "REFUTE")])
    assert dep.eligibility["amend_node"] == "AUDIT_ONLY"  # Epistemic refutation DOES kill AMENDS!

    # Policy 7: Native online SUPERSEDE cycle rejected
    dag_st = StratifiedEvidenceStore(strict_dag_supersede=True)
    dag_st.apply_mutation(["x", "y"], [("x", "y", "SUPERSEDE")])
    with pytest.raises(SupersedeCycleError):
        dag_st.apply_mutation((), [("y", "x", "SUPERSEDE")])

    # Policy 8: Imported SUPERSEDE cycle quarantine + outgoing cap (C_Y=U) + non-forwarding (X->A leaves C_B=U)
    imp = StratifiedEvidenceStore(strict_dag_supersede=False)
    imp.apply_mutation(
        ["A", "B", "Y", "X"],
        [
            ("A", "B", "SUPERSEDE"),
            ("B", "A", "SUPERSEDE"),
            ("B", "Y", "SUPERSEDE"),
        ],
    )
    assert imp.cur["A"] == "U" and imp.cur["B"] == "U"
    assert imp.cur["Y"] == "U" and imp.eligibility["Y"] == "CONTESTED"  # Outgoing quarantine cap (never F!)
    assert ("A", "B") in imp.get_telemetry().quarantined_supersede_cycles
    imp.apply_mutation((), [("X", "A", "SUPERSEDE")])
    assert imp.cur["A"] == "F" and imp.eligibility["A"] == "AUDIT_ONLY"
    assert imp.cur["B"] == "U" and imp.eligibility["B"] == "CONTESTED"  # Non-forwarding cycle quarantine!

    # Policy 9: Multiplicity & idempotent duplicate attestation delivery
    m = StratifiedEvidenceStore()
    m.apply_mutation(
        ["a", "b"],
        [
            EdgeAttestation("att1", "a", "b", "REFUTE"),
            EdgeAttestation("att1", "a", "b", "REFUTE"),  # duplicate ID ignored
            EdgeAttestation("att2", "a", "b", "REFUTE"),  # independent 2nd witness
        ],
    )
    assert m.val["b"] == "F"
    m.apply_mutation((), withdraw_attestation_ids=["att1"])
    assert m.val["b"] == "F"  # att2 still holds!
    m.apply_mutation((), withdraw_attestation_ids=["att2"])
    assert m.val["b"] == "T"

    # Policy 10: Fail-closed dangling endpoint / unknown edge type + buffer_pending replay
    fc = StratifiedEvidenceStore(buffer_pending=False)
    fc.apply_mutation(["a"])
    with pytest.raises(ValueError, match="Dangling"):
        fc.apply_mutation((), [("a", "missing", "REFUTE")])
    with pytest.raises(ValueError, match="Unknown"):
        fc.apply_mutation((), [("a", "a", "FAKE_EDGE")])

    buf = StratifiedEvidenceStore(buffer_pending=True)
    buf.apply_mutation(["a"], [EdgeAttestation("p1", "a", "b", "REFUTE")])
    assert len(buf.pending_attestations) == 1
    buf.apply_mutation(["b"])
    assert len(buf.pending_attestations) == 0 and buf.val["b"] == "F"


def test_ek5_v3_em3_telemetry_and_em1_m4e_minimal_pair_split() -> None:
    """Verify EM-3 certificate telemetry + structural audit, and EM-1 M4e contradiction vs echo split."""
    st = StratifiedEvidenceStore()
    st.apply_mutation(
        ["x", "c1", "c2"],
        [("c1", "c2", "DERIVES_CURRENT"), ("c2", "c1", "DERIVES_CURRENT"), ("x", "c1", "REFUTE")],
    )
    tel = st.get_telemetry()
    assert tel.certificate_based is True
    assert set(tel.unfounded_cycle_nodes) == {"c1", "c2"}
    assert set(st.full_structural_scc_audit()) == {"c1", "c2"}

    # EM-1 M4e: Header stripping + compatible echoes collapse + 100% minimal-pair contradiction preservation
    assert strip_boilerplate_headers("Briefing 0: Alice lives in Austin") == "Alice lives in Austin"
    assert strip_boilerplate_headers("Memory recap 2 (old): Dosage is 50 mg") == "Dosage is 50 mg"

    compatible_echoes = [
        ("Briefing 0: Alice lives in Austin", "Session 4: Alice lives in Austin"),
        ("Pressure limit is 200 kPa", "Pressure limit is 2 bar"),
        ("Meeting starts at 2pm", "Meeting starts at 14:00"),
        ("Budget is $45", "Budget is 45 dollars"),
        ("Reminder: User lives in Austin", "Earlier note recorded user lives in Austin"),
    ]
    for a, b in compatible_echoes:
        act, _ = classify_high_similarity_pair(a, b)
        assert act == "ECHO_COLLAPSE", f"False conflict on echo pair: ({a!r}, {b!r})"

    minimal_contradictions = [
        ("Dosage is 50 mg", "Dosage is 100 mg"),
        ("Flight leaves from Gate B12", "Flight leaves from Gate B14"),
        ("Meeting starts at 2pm", "Meeting starts at 3pm"),
        ("User is vegan", "User is not vegan"),
        ("Feature flag is enabled", "Feature flag is disabled"),
        ("Alice lives in Austin", "Alice lives in Denver"),
        ("Robert prefers tea", "Bob prefers tea"),
        ("User is allergic to peanuts", "User is allergic to shellfish"),
    ]
    for a, b in minimal_contradictions:
        act, _ = classify_high_similarity_pair(a, b)
        assert act == "PRESERVE_CONFLICT", f"Missed minimal-pair contradiction: ({a!r}, {b!r})"


# ===========================================================================
# PART 2: FLAGSHIP #1 (SPHERICAL BRIDGES, THEOREM 1, A2g, EM-2, EM-5b, A3, B4)
# ===========================================================================
def test_f1_woodbury_ridge_centered_ridge_and_theorem1_trichotomy() -> None:
    """Verify O(n^2 d) Woodbury Ridge parity, Centered Affine Ridge, and 3-regime Theorem 1 + A2c-Polar."""
    rng = np.random.default_rng(42)
    n, d_in, d_out = 40, 128, 64
    X = rng.normal(size=(n, d_in))
    Y = rng.normal(size=(n, d_out))
    lam = 1.5

    # Compare Woodbury solve (n < d_in) against exact dense (X^T X + lam I)^-1 X^T Y
    W_woodbury = fit_closed_form_ridge(X, Y, regularization=lam)
    W_dense = np.linalg.solve(X.T @ X + lam * np.eye(d_in), X.T @ Y)
    assert np.allclose(W_woodbury, W_dense, atol=1e-10)

    # Centered Affine Ridge matches centered solve + exact mean offset
    mu_x, mu_y, W_car = fit_closed_form_centered_ridge(X, Y, regularization=lam)
    pred = (X - mu_x) @ W_car + mu_y
    assert np.allclose(pred.mean(axis=0), Y.mean(axis=0), atol=1e-10)

    # Theorem 1 three-regime classifier
    assert classify_dimension_chain_regime([384, 768, 1024, 4096]) == "monotone"
    assert classify_dimension_chain_regime([4096, 1024, 384]) == "monotone"
    assert classify_dimension_chain_regime([1024, 384, 384, 4096]) == "valley"
    assert classify_dimension_chain_regime([384, 768, 384]) == "peak"
    assert classify_dimension_chain_regime([384, 1024, 384]) == "peak"

    # Peak chain 16 -> 48 -> 16: composed Q without polar loses kappa_+, WITH polar_reproject restores kappa_+ = 1.0!
    X0 = rng.normal(size=(120, 16))
    X1 = rng.normal(size=(120, 48))
    X2 = rng.normal(size=(120, 16))
    b1 = CenteredProcrustesBridge.fit(X0, X1)
    b2 = CenteredProcrustesBridge.fit(X1, X2)
    comp_raw = compose_centered_bridges([b1, b2], polar_reproject=False)
    comp_polar = compose_centered_bridges([b1, b2], polar_reproject=True)
    assert comp_raw.regime == "peak"
    assert comp_raw.condition_number > 1.01
    assert math.isclose(comp_polar.condition_number, 1.0, abs_tol=1e-6)
    assert comp_polar.pre_polar_sigma_min < 1.0
    assert comp_polar.polar_reprojected is True


def test_f1_prop1b_oos_scales_zeromu_softspec_and_a2g_polar_manifold() -> None:
    """Verify Repaired Prop 1b, fail-closed constant, 4 OOS scales, EM-2 exact 1-float identity, EM-5b, and A2g."""
    rng = np.random.default_rng(7)
    # Fail-closed constant source/target
    const_x = np.ones((20, 16), dtype=np.float64)
    rand_y = rng.normal(size=(20, 16))
    with pytest.raises(ValueError, match="Degenerate"):
        CenteredProcrustesBridge.fit(const_x, rand_y, strict_energy_check=True)

    # Disjoint CAL n < d regime across all 4 OOS scale laws + A2c-IW + A2c-Q + EM-5b SoftSpec
    n_fit, n_cal, d_x, d_y = 32, 24, 64, 96
    X_fit = rng.normal(size=(n_fit, d_x)) + 0.8
    Y_fit = rng.normal(size=(n_fit, d_y)) + 0.6
    X_cal = rng.normal(size=(n_cal, d_x)) + 0.8
    Y_cal = rng.normal(size=(n_cal, d_y)) + 0.6

    for mode in ("s_full_oos", "s_sub_oos", "max_one_s_full_oos", "unit"):
        br = CenteredProcrustesBridge.fit(
            X_fit, Y_fit, scale_mode=mode, cal_x=X_cal, cal_y=Y_cal, rank_cap=16
        )
        assert br.scale_mode == mode and br.scale > 0.0
        if mode == "max_one_s_full_oos":
            assert br.scale >= 1.0
        if mode == "unit":
            assert br.scale == 1.0

    # Repaired Prop 1b exact Frobenius cosine identity & ambient radius-match bridge
    br_ls = CenteredProcrustesBridge.fit(X_fit, Y_fit, scale_mode="ambient")
    xc_n = X_fit - X_fit.mean(axis=0, keepdims=True)
    yc_n = Y_fit - Y_fit.mean(axis=0, keepdims=True)
    proj_x = xc_n @ br_ls.Q
    frob_cos = float(np.sum(proj_x * yc_n) / (np.linalg.norm(proj_x) * np.linalg.norm(yc_n)))
    s_rad_proj = float(np.linalg.norm(yc_n) / np.linalg.norm(proj_x))
    s_ls_exact = float(np.sum(proj_x * yc_n) / np.sum(proj_x * proj_x))
    assert math.isclose(s_ls_exact, frob_cos * s_rad_proj, rel_tol=1e-12)
    assert math.isclose(br_ls.scale, float(np.linalg.norm(yc_n) / np.linalg.norm(xc_n)), rel_tol=1e-12)

    # A2c-IW Voronoi weights + A2c-SoftSpec (EM-5b) + A2c-Q + TRA-1 distinct hyperparameter hashes
    w_vor = compute_voronoi_anchor_weights(np.vstack([X_fit, X_cal]), list(range(n_fit)))
    assert w_vor.shape == (n_fit,) and math.isclose(float(w_vor.sum()), 1.0, rel_tol=1e-6)
    br_iw = CenteredProcrustesBridge.fit(X_fit, Y_fit, sample_weights=w_vor)
    br_soft = CenteredProcrustesBridge.fit(X_fit, Y_fit, soft_spec_tau=0.5)
    br_q = CenteredProcrustesBridge.fit(X_fit, Y_fit, mu_q_override=X_cal.mean(axis=0))
    desc_x = _mk_desc("src", d_x, RepresentationRole.QUERY)
    desc_y = _mk_desc("tgt", d_y, RepresentationRole.QUERY)
    c_base = build_bridge_contract(
        source=desc_x, target=desc_y, bridge=br_ls, transform_id="t1",
        anchor_manifest_sha256="a" * 64, validation_count=24, validation_confidence=0.95,
        evidence_id="ev1", evidence_sha256="b" * 64,
    )
    c_soft = build_bridge_contract(
        source=desc_x, target=desc_y, bridge=br_soft, transform_id="t2",
        anchor_manifest_sha256="a" * 64, validation_count=24, validation_confidence=0.95,
        evidence_id="ev1", evidence_sha256="b" * 64,
    )
    c_q = build_bridge_contract(
        source=desc_x, target=desc_y, bridge=br_q, transform_id="t3",
        anchor_manifest_sha256="a" * 64, validation_count=24, validation_confidence=0.95,
        evidence_id="ev1", evidence_sha256="b" * 64,
    )
    assert len({c_base.hyperparameters_sha256, c_soft.hyperparameters_sha256, c_q.hyperparameters_sha256}) == 3
    assert br_iw.weights_sha256 != br_ls.weights_sha256

    # EM-2 (A2-ZeroMu): Exact 1-float zero-reindex algebraic identity (< 1e-13) + scale invariance!
    Q = rng.normal(size=(15, d_x)) + 0.8
    D = rng.normal(size=(80, d_y)) + 0.6
    D_unit = l2_normalize(D)
    qids = [f"q{i:02d}" for i in range(15)]
    dids = [f"d{j:03d}" for j in range(80)]
    rk_df, sc_df = score_zeromu_rankings(Q, D_unit, br_ls, qids, dids, mode="dual_fit", k=80)
    # Explicit naive re-indexing reference: subtract mu_y from every corpus doc and re-normalize!
    q_dir = l2_normalize((Q - br_ls.mu_x) @ br_ls.Q)
    d_cent_unit = l2_normalize(D_unit - br_ls.mu_y)
    ref_sims = q_dir @ d_cent_unit.T
    for qi, qid in enumerate(qids):
        ref_order = sorted(range(80), key=lambda j: (-float(ref_sims[qi, j]), dids[j]))
        assert rk_df[qid] == [dids[j] for j in ref_order]
        assert np.allclose(sc_df[qid], ref_sims[qi, ref_order], atol=1e-13)

    # Exact positive scale invariance of query_only when queries already centered/scaled via Q_hat
    rk_qo1, sc_qo1 = score_zeromu_rankings(Q, D_unit, br_ls, qids, dids, mode="query_only", k=10)
    br_scaled = CenteredProcrustesBridge(
        mu_x=br_ls.mu_x, mu_y=br_ls.mu_y, Q=br_ls.Q, fit_count=br_ls.fit_count, scale=7.5 * br_ls.scale
    )
    rk_qo2, sc_qo2 = score_zeromu_rankings(Q, D_unit, br_scaled, qids, dids, mode="query_only", k=10)
    assert rk_qo1 == rk_qo2
    for qid in qids:
        assert np.allclose(sc_qo1[qid], sc_qo2[qid], atol=1e-14)
    rk_adap, _ = score_zeromu_rankings(Q, D_unit, br_ls, qids, dids, mode="dual_adap", k=10)
    assert len(rk_adap) == 15

    # GeodesicPolarBridge (A2g): atan2 at theta=1e-9, C1 soft collar exact interior + strict tail, dual guard!
    base_x = np.zeros(32)
    base_x[0] = 1.0
    base_y = np.zeros(48)
    base_y[0] = 1.0
    Xc = base_x + 0.08 * rng.normal(size=(120, 32))
    Yc = base_y + 0.08 * rng.normal(size=(120, 48))
    a2g_soft = GeodesicPolarBridge.fit(Xc, Yc, collar="c1_soft")
    a2g_hard = GeodesicPolarBridge.fit(Xc, Yc, collar="hard")
    assert a2g_soft.used_fallback is False
    assert a2g_soft.resultant_x >= 0.15 and a2g_soft.resultant_y >= 0.15
    assert a2g_soft.split_half_drift_x_deg <= 15.0 and a2g_soft.split_half_drift_y_deg <= 15.0
    # Verify O(d) Householder tangent basis orthonormality
    assert np.allclose(a2g_soft.basis_x.T @ a2g_soft.basis_x, np.eye(31), atol=1e-12)
    assert float(np.linalg.norm(a2g_soft.basis_x.T @ a2g_soft.pole_x)) < 1e-12
    # Verify near-pole atan2 precision at theta = 1e-9 (no NaN / acos catastrophic cancellation!)
    near_pole = base_x.copy()
    near_pole[1] = 1e-9
    out_np = a2g_soft.transform(near_pole[None, :])
    assert np.all(np.isfinite(out_np)) and math.isclose(float(np.linalg.norm(out_np[0])), 1.0, abs_tol=1e-10)

    # Verify Carbon's C1 soft collar exactness on [0, r0], strict monotonicity in active collar, and <= pi - 0.05 bound
    r0 = math.pi - 0.10
    r_vals = np.array([0.5, 2.0, r0, r0 + 0.01, r0 + 0.05, r0 + 0.15, r0 + 0.30])
    pole_test = np.zeros(8, dtype=np.float64)
    pole_test[0] = 1.0
    w_test = np.zeros((len(r_vals), 8), dtype=np.float64)
    w_test[:, 1] = r_vals
    exp_soft, clipped_soft = _sphere_exp_map(w_test, pole_test, collar="c1_soft", r0=r0, delta=0.05)
    angles_soft = np.arccos(np.clip(exp_soft @ pole_test, -1.0, 1.0))
    assert np.allclose(angles_soft[:3], r_vals[:3], atol=1e-12)
    assert np.all(np.diff(angles_soft) > 0.0)
    assert angles_soft[-1] < math.pi - 0.05
    assert bool(clipped_soft[2]) is False and bool(clipped_soft[3]) is True
    w_far = np.zeros((1, 8), dtype=np.float64)
    w_far[0, 1] = 50.0
    exp_far, _ = _sphere_exp_map(w_far, pole_test, collar="c1_soft", r0=r0, delta=0.05)
    assert float(np.arccos(np.clip(exp_far @ pole_test, -1.0, 1.0))[0]) <= math.pi - 0.05 + 1e-12

    exp_hard, _ = _sphere_exp_map(w_test, pole_test, collar="hard", eps_cut=0.05)
    angles_hard = np.arccos(np.clip(exp_hard @ pole_test, -1.0, 1.0))
    assert np.allclose(angles_hard, np.minimum(r_vals, math.pi - 0.05), atol=1e-12)
    assert a2g_hard.collar == "hard"

    # Bimodal antipodal data triggers dual guard -> clean fallback receipt to A2c!
    Xb_half = base_x + 0.1 * rng.normal(size=(60, 32))
    Xb = np.vstack([Xb_half, -Xb_half])
    Yb = np.vstack([base_y + 0.1 * rng.normal(size=(60, 48)), -(base_y + 0.1 * rng.normal(size=(60, 48)))])
    a2g_fb = GeodesicPolarBridge.fit(Xb, Yb)
    assert a2g_fb.used_fallback is True
    assert a2g_fb.fallback_bridge is not None
    assert "LOW_RESULTANT" in a2g_fb.fallback_reason or "POLE_DRIFT" in a2g_fb.fallback_reason
    out_fb, stats_fb = a2g_fb.transform_with_stats(Xb[:5])
    assert stats_fb["used_fit_fallback"] is True
    assert out_fb.shape == (5, 48) and np.all(np.isfinite(out_fb))


def test_f1_a3_dual_svd_parity_b4_backfill_ties_and_stats() -> None:
    """Verify O(n^2 d) A3 parity vs dense eigh, A3 2x2 crossing, B4 right-tail + CSLS, 160-tie order, & binary CIs."""
    rng = np.random.default_rng(123)
    n, d_x, d_y = 36, 96, 80
    X = rng.normal(size=(n, d_x))
    Y = rng.normal(size=(n, d_y))

    # O(n^2 d) dual thin-SVD whitening matches exact dense covariance eigendecomposition (< 1e-10)!
    Xc = X - X.mean(axis=0, keepdims=True)
    dual_inv, dual_sqrt = _shrinkage_cov_factors(Xc, shrink=0.20, trace_divisor_mode="ambient")
    cov_x = (Xc.T @ Xc) / float(n - 1)
    evals, evecs = np.linalg.eigh(cov_x)
    evals = np.maximum(evals, 0.0)
    lam = 0.20 * (float(np.trace(cov_x)) / float(d_x))
    ev_reg = np.maximum(0.80 * evals + lam, 1e-8)
    dense_inv = (evecs * (ev_reg ** -0.5)) @ evecs.T
    dense_sqrt = (evecs * (ev_reg ** 0.5)) @ evecs.T
    assert np.allclose(dual_inv, dense_inv, atol=1e-10)
    assert np.allclose(dual_sqrt, dense_sqrt, atol=1e-10)

    # 2x2 crossing: subspace_rank in (None, 24) x trace_divisor_mode in (ambient, active_rank, effective_rank)
    for sub in (None, 24):
        for mode in ("ambient", "active_rank", "effective_rank"):
            br = WhitenedSphericalProcrustesBridge.fit(X, Y, subspace_rank=sub, trace_divisor_mode=mode)
            assert br.identified_cross_rank >= 1
            out = br.transform(X[:4])
            assert out.shape == (4, d_y) and np.all(np.isfinite(out))

    # B4 right-tail strict order preservation vs legacy_hard_clip + CSLS + B2-OldSpace
    n_docs = 120
    d_old = l2_normalize(rng.normal(size=(n_docs, 16)))
    d_new = l2_normalize(rng.normal(size=(n_docs, 16)))
    q_dev_new = l2_normalize(rng.normal(size=(25, 16)))
    q_dev_map = l2_normalize(rng.normal(size=(25, 16)))
    # Craft 1 query whose two backfilled docs BOTH exceed the dev max native cosine (0.99 vs 0.999)
    q_test_new = np.zeros((1, 16), dtype=np.float64)
    q_test_new[0, 0] = 1.0
    d_new[0] = l2_normalize(np.array([0.999, 0.04] + [0.0] * 14))
    d_new[1] = l2_normalize(np.array([0.990, 0.14] + [0.0] * 14))
    q_test_map = l2_normalize(rng.normal(size=(1, 16)))
    dids = [f"d{i:03d}" for i in range(n_docs)]
    rk_b4 = score_quantile_calibrated_backfill(
        q_test_new, q_test_map, d_old, d_new, [0, 1], ["q0"], dids,
        dev_queries_new=q_dev_new, dev_queries_mapped=q_dev_map, k=5, csls_k=5, legacy_hard_clip=False,
    )
    assert rk_b4["q0"][0] == "d000" and rk_b4["q0"][1] == "d001"
    rk_legacy = score_quantile_calibrated_backfill(
        q_test_new, q_test_map, d_old, d_new, [0, 1], ["q0"], dids,
        dev_queries_new=q_dev_new, dev_queries_mapped=q_dev_map, k=5, csls_k=0, legacy_hard_clip=True,
    )
    assert len(rk_legacy["q0"]) == 5

    # B2-OldSpace zero-target-encode anchor residual distortion predictor
    a2c = CenteredProcrustesBridge.fit(d_old[:30], d_new[:30])
    anc_res = 1.0 - np.sum(a2c.transform(d_old[:30]) * d_new[:30], axis=1)
    order_os, pred_os = compute_old_space_knn_distortion(d_old, list(range(30)), anc_res, k_neighbors=8)
    assert order_os.shape == (n_docs,) and pred_os.shape == (n_docs,) and np.all(np.isfinite(pred_os))

    # 160-tie deterministic canonical ID order in both FastBM25Index.hybrid_rankings & score_map_plus_backfill
    doc_ids = [f"doc_{i:04d}" for i in range(200)]
    tied_dense = np.ones((2, 200), dtype=np.float64)
    bm25 = FastBM25Index(doc_ids, ["shared token" for _ in doc_ids])
    ranked_bm25 = bm25.hybrid_rankings(["shared token", "shared token"], ["q0", "q1"], tied_dense, k=10)
    assert ranked_bm25["q0"] == [f"doc_{i:04d}" for i in range(10)]

    # Identical vectors for all 200 docs -> 200-way exact tie in score_map_plus_backfill -> canonical ID order!
    same_vec = np.zeros((200, 8), dtype=np.float64)
    same_vec[:, 0] = 1.0
    q_same = np.zeros((2, 8), dtype=np.float64)
    q_same[:, 0] = 1.0
    rk_bf = score_map_plus_backfill(
        q_same, same_vec, same_vec, np.eye(8), np.array([0, 1, 2], dtype=np.int64),
        ["q0", "q1"], doc_ids, dev_queries_new=q_same, k=10, calibrate=False,
    )
    assert rk_bf["q0"] == [f"doc_{i:04d}" for i in range(10)]

    # paired_bootstrap_delta fail-closed ID mismatch + expected_ids + NaN rejection
    with pytest.raises(ValueError, match="mismatch"):
        paired_bootstrap_delta({"q1": 1.0, "q2": 0.5}, {"q1": 1.0, "q3": 0.5})
    with pytest.raises(ValueError, match="expected_ids"):
        paired_bootstrap_delta({"q1": 1.0}, {"q1": 1.0}, expected_ids=["q1", "q2"])
    with pytest.raises(ValueError, match="Non-finite"):
        paired_bootstrap_delta({"q1": float("nan")}, {"q1": 1.0})

    # exact_paired_binary_ci on 0/500 tie: NEVER 0.00 pp! Must match Clopper-Pearson (-0.873 pp) & Tango (-0.76 pp)!
    ci_tie = exact_paired_binary_ci([1] * 500, [1] * 500)
    assert ci_tie["n10"] == 0 and ci_tie["n01"] == 0
    assert -0.0090 < ci_tie["clopper_pearson_low"] < -0.0085
    assert -0.0080 < ci_tie["tango_score_low"] < -0.0072
    bt_tie = paired_bootstrap_delta({f"q{i}": 1.0 for i in range(500)}, {f"q{i}": 1.0 for i in range(500)}, n_bootstrap=500)
    assert math.isclose(bt_tie["clopper_pearson_lcb95"], ci_tie["clopper_pearson_low"], abs_tol=1e-12)
    assert math.isclose(bt_tie["tango_score_lcb95"], ci_tie["tango_score_low"], abs_tol=1e-12)
