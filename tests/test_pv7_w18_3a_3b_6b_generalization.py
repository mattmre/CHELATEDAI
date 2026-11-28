"""Comprehensive CPU-only unit tests for Wave 18 Generalization Bundles #50 (3A), #51 (3B), and #54 (6B)."""

from __future__ import annotations

import importlib.util
import os
import sys
import unittest
from dataclasses import replace
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chelated_12domain_4hop_gauntlet import (  # noqa: E402
    ENCODER_NAMES,
    FORBIDDEN_BURNED_SUBSTITUTES,
    LOCKED_12_DOMAINS,
    POISONING_SWEEP_RATES,
    build_deterministic_12domain_gauntlet_fixture,
    run_12domain_4hop_gauntlet,
    verify_12_domains_unburned,
)
from chelated_beam_128k_replay import (  # noqa: E402
    BEAM_CONVERSATION_COUNT,
    BEAM_TOTAL_QUESTIONS,
    build_deterministic_beam_128k_fixture,
    conversation_cluster_bootstrap,
    ingest_beam_128k_conversations,
    run_beam_128k_paired_replay,
)
from chelated_loft_and_zero_write_ni import (  # noqa: E402
    CANONICAL_BURNED_AUDIT_LIST,
    HISTORICAL_PM2_TANGO_LCB,
    LOFT_SCIFACT_DOC_COUNT,
    LOFT_SCIFACT_QUERY_COUNT,
    MIN_DISJOINT_PAIRED_QUERIES,
    PREREGISTERED_NI_MARGIN,
    SPINE_DIGEST,
    audit_burned_corpora_rejection,
    build_deterministic_loft_scifact_fixture,
    build_synthetic_disjoint_ni_pool,
    compute_exact_tango_ni,
    evaluate_disjoint_zero_write_ni,
    run_canonical_loft_scifact_pilot,
    run_derived_zero_write_migration_pilot,
    size_disjoint_ni_sample,
)
from chelated_pv7_spine.burn_ledger import (  # noqa: E402
    BurnedCorpusError,
    build_burn_ledger,
    text_fingerprint,
)
from chelated_pv7_spine.claim_registry import claim_registry_sha256  # noqa: E402
from chelated_pv7_spine.fragment_compiler import compile_repo_fragments  # noqa: E402
from scripts import (  # noqa: E402
    run_pv7_w18_3a_zero_write_ni as runner_3a,
    run_pv7_w18_3b_beam_memory as runner_3b,
    run_pv7_w18_6b_12domain_4hop as runner_6b,
)
from scripts.night1_common import exact_paired_binary_ci  # noqa: E402


def _load_verifier(bundle_id: int):
    path = REPO_ROOT / "scripts" / "verifiers.d" / f"{bundle_id}_verifier.py"
    spec = importlib.util.spec_from_file_location(f"verifier_{bundle_id}", path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _baseline_hist_fps():
    return {
        "documents": [text_fingerprint("historical corpus marker document alpha")],
        "queries": [text_fingerprint("historical corpus marker query alpha")],
    }


class TestBundle50Item3AZeroWriteNI(unittest.TestCase):
    """Bundle #50 (Item 3A): Canonical LOFT SciFact Pilot + Disjoint Exact Tango NI -0.0500."""

    def test_canonical_loft_scifact_strictly_separated_from_derived_zero_write_pilot(self):
        fixture = build_deterministic_loft_scifact_fixture(seed=5001)
        canonical = run_canonical_loft_scifact_pilot(fixture)
        derived = run_derived_zero_write_migration_pilot(fixture)

        self.assertEqual(canonical["doc_count"], LOFT_SCIFACT_DOC_COUNT)
        self.assertEqual(canonical["query_count"], LOFT_SCIFACT_QUERY_COUNT)
        self.assertFalse(canonical["confirmatory_ni_pooled"])
        self.assertTrue(canonical["burned_for_confirmatory_ni"])

        self.assertTrue(derived["separated_from_canonical_gbench"])
        self.assertFalse(derived["confirmatory_ni_pooled"])
        self.assertTrue(derived["resident_docs_unmodified"])
        self.assertEqual(derived["resident_write_bytes"], 0)
        self.assertIn("Rank1QueryResidual__CSLSMerge", derived["arms"])
        self.assertIn("ZeroMuPolarBridge_dual_adap__CSLSMerge", derived["arms"])
        self.assertGreater(derived["arms"]["ZeroMuPolarBridge_dual_adap__CSLSMerge"]["mean_ndcg_at_10"], 0.70)

    def test_exact_tango_matches_repo_reference_and_handles_zero_discordance(self):
        # Zero discordance must not collapse to [0.0, 0.0]
        zeros_a = [1] * 1000
        zeros_b = [1] * 1000
        res_zero = compute_exact_tango_ni(zeros_a, zeros_b)
        ref_zero = exact_paired_binary_ci(zeros_a, zeros_b)
        self.assertAlmostEqual(res_zero.tango_score_lcb95, ref_zero["tango_score_low"], places=6)
        self.assertAlmostEqual(res_zero.tango_score_ucb95, ref_zero["tango_score_high"], places=6)
        self.assertLess(res_zero.tango_score_lcb95, 0.0)
        self.assertTrue(res_zero.clears_ni_margin)

        # Discordant case matching repo reference
        a = [1] * 850 + [1] * 35 + [0] * 40 + [0] * 75
        b = [1] * 850 + [0] * 35 + [1] * 40 + [0] * 75
        res_disc = compute_exact_tango_ni(a, b)
        ref_disc = exact_paired_binary_ci(a, b)
        self.assertAlmostEqual(res_disc.tango_score_lcb95, ref_disc["tango_score_low"], places=6)
        self.assertAlmostEqual(res_disc.tango_score_ucb95, ref_disc["tango_score_high"], places=6)

    def test_power_sizing_enforces_min_1000_and_scales_with_discordance_variance(self):
        low_var = size_disjoint_ni_sample(0.015, 0.015)
        high_var = size_disjoint_ni_sample(0.18, 0.18)
        self.assertGreaterEqual(low_var.required_n, MIN_DISJOINT_PAIRED_QUERIES)
        self.assertGreater(high_var.asymptotic_required_n, low_var.asymptotic_required_n)
        self.assertGreaterEqual(high_var.required_n, high_var.asymptotic_required_n)

    def test_burn_ledger_fails_closed_on_burned_corpora_and_text_overlap(self):
        ledger = build_burn_ledger(REPO_ROOT)
        audit = audit_burned_corpora_rejection(ledger=ledger)
        self.assertTrue(audit["all_burned_rejected"])
        for corpus in CANONICAL_BURNED_AUDIT_LIST:
            self.assertTrue(audit["checked_corpora"][corpus], f"Expected {corpus} to be rejected by burn_ledger")

        sizing = size_disjoint_ni_sample(0.03, 0.03)
        pool = build_synthetic_disjoint_ni_pool(corpus_label="scifact", n_queries=1000, n_docs=50)
        with self.assertRaises(BurnedCorpusError):
            evaluate_disjoint_zero_write_ni(
                corpus_label="scifact",
                doc_texts=pool["doc_texts"],
                query_texts=pool["query_texts"],
                zero_write_hits=pool["zero_write_hits"],
                reference_hits=pool["reference_hits"],
                sizing_plan=sizing,
                historical_fingerprints=_baseline_hist_fps(),
                ledger=ledger,
                allow_local_drycheck_smoke=True,
            )

        # Exact SHA-256 text collision under an innocent corpus label must also fail closed
        poisoned_doc = "historical corpus marker document alpha"
        clean_pool = build_synthetic_disjoint_ni_pool(corpus_label="clean-unburned-pool", n_queries=1000, n_docs=50)
        docs_with_collision = [poisoned_doc] + list(clean_pool["doc_texts"][1:])
        with self.assertRaises(BurnedCorpusError):
            evaluate_disjoint_zero_write_ni(
                corpus_label="clean-unburned-pool",
                doc_texts=docs_with_collision,
                query_texts=clean_pool["query_texts"],
                zero_write_hits=clean_pool["zero_write_hits"],
                reference_hits=clean_pool["reference_hits"],
                sizing_plan=sizing,
                historical_fingerprints=_baseline_hist_fps(),
                ledger=ledger,
                allow_local_drycheck_smoke=True,
            )

    def test_underpowered_pool_emits_underpowered_and_refuses_margin_loosening(self):
        ledger = build_burn_ledger(REPO_ROOT)
        sizing = size_disjoint_ni_sample(0.03, 0.03)
        small_pool = build_synthetic_disjoint_ni_pool(corpus_label="small-disjoint-pool", n_queries=200, n_docs=50)
        report = evaluate_disjoint_zero_write_ni(
            corpus_label=small_pool["corpus_label"],
            doc_texts=small_pool["doc_texts"],
            query_texts=small_pool["query_texts"],
            zero_write_hits=small_pool["zero_write_hits"],
            reference_hits=small_pool["reference_hits"],
            sizing_plan=sizing,
            historical_fingerprints=_baseline_hist_fps(),
            ledger=ledger,
            allow_local_drycheck_smoke=True,
        )
        self.assertEqual(report["status"], "underpowered")
        self.assertTrue(report["underpowered"])
        self.assertFalse(report["one_shot_consumed"])
        self.assertEqual(report["ni_margin"], PREREGISTERED_NI_MARGIN)
        self.assertFalse(report["margin_loosened"])
        self.assertEqual(report["historical_pm2_reference"]["tango_score_lcb95"], HISTORICAL_PM2_TANGO_LCB)

        with self.assertRaises(ValueError):
            evaluate_disjoint_zero_write_ni(
                corpus_label=small_pool["corpus_label"],
                doc_texts=small_pool["doc_texts"],
                query_texts=small_pool["query_texts"],
                zero_write_hits=small_pool["zero_write_hits"],
                reference_hits=small_pool["reference_hits"],
                sizing_plan=sizing,
                historical_fingerprints=_baseline_hist_fps(),
                ledger=ledger,
                allow_local_drycheck_smoke=True,
                ni_margin=-0.0600,
            )

    def test_runner_3a_drycheck_live_fail_closed_and_verifier(self):
        dry = runner_3a.execute("local-drycheck")
        self.assertTrue(dry["passed"])
        self.assertEqual(dry["status"], "local-drycheck-pass")

        live = runner_3a.execute("run-live-gpu")
        self.assertEqual(live["status"], "awaiting-live-inputs")
        self.assertFalse(live["passed"])

        v50 = _load_verifier(50).verify_bundle(REPO_ROOT)
        self.assertTrue(v50["passed"])
        self.assertEqual(v50["bundle_id"], 50)


class TestBundle51Item3BBeamMemory(unittest.TestCase):
    """Bundle #51 (Item 3B): BEAM 128K 20-Conversation Cluster Bootstrap + EK5 Prefix Parity."""

    def test_ingestion_and_cluster_bootstrap_never_inflates_n_to_400(self):
        convs, receipt = ingest_beam_128k_conversations(
            None, historical_fingerprints=_baseline_hist_fps(), ledger=build_burn_ledger(REPO_ROOT), seed=5101
        )
        self.assertEqual(len(convs), BEAM_CONVERSATION_COUNT)
        self.assertEqual(receipt["total_questions"], BEAM_TOTAL_QUESTIONS)

        # Build correlated conversation scores where cluster-level CI must be wider than naive question CI
        scores_by_conv = {
            f"conv_{i:02d}": ([1.0] * 20 if i < 12 else [0.0] * 20) for i in range(BEAM_CONVERSATION_COUNT)
        }
        boot = conversation_cluster_bootstrap(scores_by_conv, n_resamples=2000, seed=5102)
        self.assertEqual(boot.sampling_unit, "conversation_cluster")
        self.assertEqual(boot.n_clusters, BEAM_CONVERSATION_COUNT)
        self.assertEqual(boot.total_questions, BEAM_TOTAL_QUESTIONS)
        self.assertFalse(boot.naive_question_n_inflated)
        self.assertGreater(boot.intraclass_correlation_icc, 0.90)
        self.assertGreater(boot.cluster_ci_width, boot.naive_question_ci_width_diagnostic * 2.5)

    def test_paired_withdrawal_histories_and_100pct_ek5_prefix_parity(self):
        convs = build_deterministic_beam_128k_fixture(seed=5101)
        replay = run_beam_128k_paired_replay(convs, n_bootstrap=1000, seed=5102)

        self.assertEqual(replay["conversation_clusters"], BEAM_CONVERSATION_COUNT)
        self.assertEqual(replay["total_questions"], BEAM_TOTAL_QUESTIONS)
        self.assertEqual(replay["ek5_prefix_parity_rate"], 1.0)
        self.assertEqual(replay["ek5_prefix_mismatches"], 0)
        self.assertGreaterEqual(replay["total_prefixes_checked"], 20 * 2 * 7)
        self.assertTrue(replay["checkpoint_replay_all_exact"])
        self.assertTrue(replay["sql_challenger_mode_parity_exact"])

        # EK5 achieves 1.0 on both derivative-only withdrawal AND independent corroboration survival
        self.assertAlmostEqual(replay["cluster_bootstrap"]["derivative_only"]["mean"], 1.0)
        self.assertAlmostEqual(replay["cluster_bootstrap"]["independent_corroboration"]["mean"], 1.0)
        self.assertGreater(replay["cluster_bootstrap"]["ek5_minus_recency_baseline"]["lcb_95"], 0.0)

    def test_runner_3b_drycheck_live_fail_closed_and_verifier(self):
        dry = runner_3b.execute("local-drycheck")
        self.assertTrue(dry["passed"])
        self.assertEqual(dry["status"], "local-drycheck-pass")

        live = runner_3b.execute("run-live-gpu")
        self.assertEqual(live["status"], "awaiting-live-inputs")
        self.assertFalse(live["passed"])

        v51 = _load_verifier(51).verify_bundle(REPO_ROOT)
        self.assertTrue(v51["passed"])
        self.assertEqual(v51["bundle_id"], 51)


class TestBundle54Item6B12Domain4HopGauntlet(unittest.TestCase):
    """Bundle #54 (Item 6B): Zero-Retune 12-Domain, 5-Encoder/4-Hop, Shift & Poisoning Gauntlet."""

    def test_locked_12_domains_exclude_burned_substitutes_and_fail_closed_on_overlap(self):
        domain_ids = [d.domain_id for d in LOCKED_12_DOMAINS]
        self.assertEqual(len(domain_ids), 12)
        lower_ids = {d.lower() for d in domain_ids}
        for forbidden in FORBIDDEN_BURNED_SUBSTITUTES:
            self.assertNotIn(forbidden, lower_ids)
        self.assertIn("CoIR-Apps", domain_ids)
        self.assertIn("Touche-2020", domain_ids)
        self.assertIn("BioASQ-TaskB", domain_ids)

        bundles = build_deterministic_12domain_gauntlet_fixture(
            seed=5400, n_docs_per_domain=40, n_queries_per_domain=12, encoder_dims=(16, 16, 20, 24, 28)
        )
        ledger = build_burn_ledger(REPO_ROOT)
        receipt = verify_12_domains_unburned(
            bundles, historical_fingerprints=_baseline_hist_fps(), ledger=ledger, allow_local_drycheck_smoke=True
        )
        self.assertTrue(receipt["all_12_unburned"])
        self.assertEqual(receipt["n_domains_verified"], 12)

        # Injecting an NFCorpus/historical doc SHA-256 into BioASQ-TaskB must fail closed BEFORE scoring
        bioasq_bad = replace(
            bundles[1],
            doc_texts=("historical corpus marker document alpha",) + bundles[1].doc_texts[1:],
        )
        tampered = (bundles[0], bioasq_bad) + bundles[2:]
        with self.assertRaises(BurnedCorpusError):
            run_12domain_4hop_gauntlet(
                tampered,
                historical_fingerprints=_baseline_hist_fps(),
                ledger=ledger,
                allow_local_drycheck_smoke=True,
                n_anchor_draws=2,
                anchors_per_draw=20,
                poisoning_rates=(0.0, 0.10),
                n_bootstrap=200,
            )

    def test_full_gauntlet_hypotheses_4hop_shift_and_poisoning_sweep(self):
        dry = runner_6b.execute("local-drycheck")
        self.assertTrue(dry["passed"])
        self.assertEqual(dry["status"], "local-drycheck-pass")

        g = dry["gauntlet"]
        self.assertFalse(g["retuned_on_eval"])
        self.assertEqual(len(g["per_domain"]), 12)
        self.assertEqual([e["label"] for e in g["encoder_chain"]], list(ENCODER_NAMES))
        self.assertEqual(g["anchor_draw_variance"]["n_draws_per_domain"], 20)
        self.assertEqual(g["anchor_draw_variance"]["cross_domain_n"], 12)
        self.assertEqual(tuple(g["poisoning_sweep"]["rates"]), POISONING_SWEEP_RATES)

        hyps = g["hypotheses"]
        self.assertEqual(hyps["H6B_merge"]["n_domains"], 12)
        self.assertEqual(hyps["H6B_bridge_static"]["retirement_floor"], -0.010)
        self.assertEqual(hyps["H6B_bridge_static"]["tight_equivalence_floor"], -0.005)
        self.assertIn("zeromu_vs_centered_csls_contrast", hyps["H6B_bridge_static"])
        self.assertTrue(hyps["H6B_bridge_static"]["clears_retirement_floor"])
        self.assertGreater(hyps["H6B_bridge_shift_and_4hop"]["prompt_shift"]["lcb_95"], 0.0)
        self.assertGreater(hyps["H6B_bridge_shift_and_4hop"]["four_hop_v1_to_v5"]["lcb_95"], 0.0)
        self.assertIsNotNone(hyps["H6B_bridge_shift_and_4hop"]["prompt_shift"]["mean_prefix_displacement_norm"])
        self.assertIsNotNone(hyps["H6B_bridge_shift_and_4hop"]["four_hop_v1_to_v5"]["mean_direct_v1_v5_delta"])

        # Verify shifted-anchor refit ceiling and 4 transitions logged per domain
        first_dom = g["per_domain"]["TREC-COVID"]
        self.assertEqual(len(first_dom["factorial_5x3"]), 15)
        self.assertIn("shifted_anchor_refit_ceiling_ndcg", first_dom["prompt_shift"])
        self.assertEqual(len(first_dom["four_hop_v1_to_v5"]["transitions_zeromu"]), 4)
        self.assertAlmostEqual(first_dom["four_hop_v1_to_v5"]["zeromu_final_nonzero_kappa"], 1.0, places=6)

        live = runner_6b.execute("run-live-gpu")
        self.assertEqual(live["status"], "awaiting-live-inputs")
        self.assertFalse(live["passed"])

        v54 = _load_verifier(54).verify_bundle(REPO_ROOT)
        self.assertTrue(v54["passed"])
        self.assertEqual(v54["bundle_id"], 54)


class TestBatch3SpineCompilationAndBurnCleanliness(unittest.TestCase):
    """Cross-bundle verification: compile_repo_fragments + burn_ledger non-pollution + registry immutability."""

    def test_compile_repo_fragments_includes_50_51_54(self):
        actual_spine = claim_registry_sha256()
        self.assertEqual(actual_spine, SPINE_DIGEST)

        compiled = compile_repo_fragments(REPO_ROOT, SPINE_DIGEST)
        reg_ids = {entry["id"] for entry in compiled["registry"]}
        ver_ids = {entry["bundle_id"] for entry in compiled["verifiers"]}
        for bid in (50, 51, 54):
            self.assertIn(bid, reg_ids)
            self.assertIn(bid, ver_ids)
            self.assertIn(bid, compiled["dependency_order"])

    def test_new_scripts_do_not_pollute_burn_ledger(self):
        ledger = build_burn_ledger(REPO_ROOT)
        burned_names = set(dict(ledger.datasets).keys())
        for desc in LOCKED_12_DOMAINS:
            self.assertNotIn(
                desc.burn_check_id,
                burned_names,
                f"Runner script polluted burn_ledger with unburned domain {desc.burn_check_id}",
            )


if __name__ == "__main__":
    unittest.main()
