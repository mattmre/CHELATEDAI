import json
import tempfile
import unittest
from pathlib import Path

from observability_experiments import (
    COA1_STAGE_ID,
    CTX1_STAGE_ID,
    EK7_COALITION_STAGE_ID,
    OBS1_STAGE_ID,
    TRANSPORT_STAGE_ID,
    ObservabilityArtifact,
    ObservabilityBudget,
    ObservabilityResourceError,
    ObservabilityValidationError,
    make_coa1_artifact,
    make_coalition_rag_artifact,
    make_ctx1_artifact,
    make_obs1_artifact,
    make_transport_artifact,
    run_coalition_rag_case,
    run_coalition_rag_suite,
    run_coa1_case,
    run_coa1_suite,
    run_ctx1_case,
    run_ctx1_suite,
    run_obs1_case,
    run_obs1_suite,
    run_transport_case,
    run_transport_suite,
)


class ObservabilityExperimentsTests(unittest.TestCase):
    def test_budget_rejects_oversized_modeled_work(self):
        budget = ObservabilityBudget(max_work_units=100)
        with self.assertRaises(ObservabilityResourceError):
            run_obs1_case(
                policy="bounded_full_rank",
                dimension=8,
                probes=8,
                budget=budget,
            )

    def test_budget_rejects_invalid_caller_cap(self):
        with self.assertRaises(ObservabilityValidationError):
            ObservabilityBudget(max_dimension=65)

    def test_obs1_in_span_is_unobservable(self):
        result = run_obs1_case(
            policy="in_span",
            dimension=16,
            initial_rank=4,
            probes=32,
            seed=7,
        )
        self.assertEqual(result["stage_id"], OBS1_STAGE_ID)
        self.assertEqual(result["hidden_energy"], 0.0)
        self.assertFalse(result["discovered"])
        self.assertTrue(result["support_preserves_initial_span"])
        self.assertEqual(result["design_rank"], 4)

    def test_obs1_full_rank_control_recovers_hidden_coordinate(self):
        result = run_obs1_case(
            policy="bounded_full_rank",
            dimension=16,
            initial_rank=4,
            probes=32,
            seed=7,
        )
        self.assertGreater(result["hidden_energy"], 0.0)
        self.assertTrue(result["discovered"])
        self.assertEqual(result["design_rank"], 16)
        self.assertAlmostEqual(result["hidden_effect_estimate"], 1.0, places=10)

    def test_obs1_suite_reports_bounded_sanity_only(self):
        result = run_obs1_suite(seeds=(7, 11))
        self.assertEqual(result["evidence_state"], "VALIDATED")
        self.assertEqual(result["scientific_claim_status"], "UNCONFIRMED")
        self.assertTrue(result["sanity"]["in_span_never_discovers"])
        self.assertTrue(result["sanity"]["bounded_full_rank_discovers"])

    def test_coa1_singletons_cannot_identify_pair(self):
        result = run_coa1_case(
            policy="singleton",
            dimension=6,
            order=2,
            max_active=1,
            probes=64,
            seed=7,
        )
        self.assertEqual(result["stage_id"], COA1_STAGE_ID)
        self.assertTrue(result["support_below_degree"])
        self.assertEqual(result["hidden_energy"], 0.0)
        self.assertFalse(result["discovered"])
        self.assertTrue(result["required_sanity_holds"])

    def test_coa1_top_k_pair_recovers_pair(self):
        result = run_coa1_case(
            policy="top_k",
            dimension=6,
            order=2,
            max_active=2,
            probes=64,
            seed=7,
        )
        self.assertFalse(result["support_below_degree"])
        self.assertGreater(result["hidden_energy"], 0.0)
        self.assertTrue(result["discovered"])
        self.assertTrue(result["required_sanity_holds"])

    def test_coa1_triads_respect_degree_ceiling(self):
        below = run_coa1_case(
            policy="top_k",
            dimension=6,
            order=3,
            max_active=2,
            probes=64,
            seed=7,
        )
        eligible = run_coa1_case(
            policy="top_k",
            dimension=6,
            order=3,
            max_active=3,
            probes=64,
            seed=7,
        )
        self.assertTrue(below["support_below_degree"])
        self.assertFalse(below["discovered"])
        self.assertTrue(below["required_sanity_holds"])
        self.assertFalse(eligible["support_below_degree"])
        self.assertTrue(eligible["discovered"])
        self.assertTrue(eligible["required_sanity_holds"])

    def test_coa1_suite_reports_all_sanity_cells(self):
        result = run_coa1_suite(seeds=(7, 11))
        self.assertEqual(result["evidence_state"], "VALIDATED")
        self.assertTrue(result["sanity"]["below_degree_remains_unidentified"])
        self.assertTrue(result["sanity"]["eligible_pair_recovers"])
        self.assertTrue(result["sanity"]["eligible_triad_recovers"])

    def test_artifacts_are_deterministic_and_atomic(self):
        first = make_obs1_artifact(seed=7)
        second = make_obs1_artifact(seed=7)
        self.assertIsInstance(first, ObservabilityArtifact)
        self.assertEqual(first.as_dict(), second.as_dict())
        self.assertEqual(first.stage_id, OBS1_STAGE_ID)
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory) / "obs1.json"
            written = first.write_json(output)
            self.assertEqual(written, output)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload, first.as_dict())
            self.assertFalse(list(output.parent.glob("*.tmp")))

    def test_coa_artifact_has_distinct_stage_and_limitations(self):
        artifact = make_coa1_artifact(seed=11)
        payload = artifact.as_dict()
        self.assertEqual(payload["stage_id"], COA1_STAGE_ID)
        self.assertIn("interaction existence is not utility advantage", payload["limitations"])
        self.assertEqual(payload["result"]["scientific_claim_status"], "UNCONFIRMED")

    def test_ctx1_rejects_incompatible_witnesses_and_abstains_without_support(self):
        incompatible = run_ctx1_case(case="incompatible_witness", seed=7)
        missing = run_ctx1_case(case="missing_propensity", seed=7)
        compatible = run_ctx1_case(case="compatible_positive", seed=7)
        self.assertEqual(incompatible["stage_id"], CTX1_STAGE_ID)
        self.assertTrue(incompatible["naive_promotes"])
        self.assertEqual(incompatible["aware_decision"], "UNIDENTIFIED")
        self.assertTrue(incompatible["required_sanity_holds"])
        self.assertEqual(missing["aware_decision"], "UNIDENTIFIED")
        self.assertTrue(missing["required_sanity_holds"])
        self.assertEqual(compatible["aware_decision"], "PROMOTE")
        self.assertTrue(compatible["required_sanity_holds"])

    def test_ctx1_suite_reports_context_sanity_only(self):
        result = run_ctx1_suite(seeds=(7, 11))
        self.assertEqual(result["evidence_state"], "VALIDATED")
        self.assertEqual(result["scientific_claim_status"], "UNCONFIRMED")
        self.assertTrue(result["sanity"]["incompatible_witness_not_promoted"])
        self.assertTrue(result["sanity"]["compatible_positive_promoted"])
        self.assertTrue(result["sanity"]["missing_propensity_abstains"])

    def test_transport_control_separates_raw_drift_from_known_alignment(self):
        result = run_transport_case(seed=7)
        self.assertEqual(result["stage_id"], TRANSPORT_STAGE_ID)
        self.assertTrue(result["raw_drift_visible"])
        self.assertTrue(result["transport_recovers_metric"])
        self.assertGreater(result["raw_cosine_error"], result["aligned_cosine_error"])
        self.assertEqual(result["aligned_top_one_agreement"], 1.0)

    def test_transport_suite_has_no_novelty_claim(self):
        result = run_transport_suite(seeds=(7, 11))
        self.assertEqual(result["evidence_state"], "VALIDATED")
        self.assertEqual(result["scientific_claim_status"], "UNCONFIRMED")
        self.assertTrue(result["sanity"]["transport_recovers_metric"])
        self.assertTrue(result["sanity"]["raw_drift_visible"])

    def test_coalition_rag_surrogate_recovers_conjunctive_coverage(self):
        result = run_coalition_rag_case(seed=7)
        self.assertEqual(result["stage_id"], EK7_COALITION_STAGE_ID)
        self.assertFalse(result["individual_conjunctive_success"])
        self.assertTrue(result["coalition_conjunctive_success"])
        self.assertEqual(result["coalition_selected"], ["d_ab_1", "d_delta", "d_gamma"])

    def test_coalition_rag_suite_is_explicitly_surrogate_only(self):
        result = run_coalition_rag_suite(seeds=(7, 11))
        self.assertEqual(result["evidence_state"], "VALIDATED")
        self.assertEqual(result["scientific_claim_status"], "UNCONFIRMED")
        self.assertTrue(result["sanity"]["individual_top_k_misses_conjunction"])
        self.assertTrue(result["sanity"]["coalition_recovers_conjunction"])

    def test_additional_artifact_stages_are_content_addressed(self):
        artifacts = (
            make_ctx1_artifact(seed=7),
            make_transport_artifact(seed=7),
            make_coalition_rag_artifact(seed=7),
        )
        self.assertEqual(
            [artifact.stage_id for artifact in artifacts],
            [CTX1_STAGE_ID, TRANSPORT_STAGE_ID, EK7_COALITION_STAGE_ID],
        )
        for artifact in artifacts:
            payload = artifact.as_dict()
            self.assertEqual(payload["status"], "COMPLETE")
            self.assertTrue(payload["artifact_digest"])
            self.assertEqual(payload["result"]["scientific_claim_status"], "UNCONFIRMED")

    def test_unknown_policies_are_rejected(self):
        with self.assertRaises(ObservabilityValidationError):
            run_obs1_case(policy="unknown")
        with self.assertRaises(ObservabilityValidationError):
            run_coa1_case(policy="unknown")


if __name__ == "__main__":
    unittest.main()
