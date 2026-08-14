import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from paired_intervention_experiments import (
    MATERIAL_PAIR,
    NUISANCE_PAIR,
    PAIRED_INTERVENTION_PROTOCOL_ID,
    PAIRED_INTERVENTION_STAGE_ID,
    PairedCase,
    PairedInterventionArtifact,
    PairedInterventionBudget,
    PairedInterventionResourceError,
    PairedInterventionValidationError,
    build_paired_chelation_fixture,
    estimate_paired_resources,
    evaluate_synthetic_policy,
    make_paired_intervention_artifact,
    production_variance_chelation_mask,
    run_paired_chelation_sanity,
    score_paired_cases,
)
from run_paired_intervention_sanity import run


class PairedInterventionMetricTests(unittest.TestCase):
    def test_case_contract_distinguishes_nuisance_and_material_labels(self):
        with self.assertRaises(PairedInterventionValidationError):
            PairedCase("n", NUISANCE_PAIR, "a", "b", "a", "b", ("noise",))
        with self.assertRaises(PairedInterventionValidationError):
            PairedCase("m", MATERIAL_PAIR, "a", "a", "a", "a", ("fact",))

    def test_case_contract_rejects_duplicate_or_empty_features(self):
        with self.assertRaises(PairedInterventionValidationError):
            PairedCase("m", MATERIAL_PAIR, "a", "b", "a", "b", ())
        with self.assertRaises(PairedInterventionValidationError):
            PairedCase("m", MATERIAL_PAIR, "a", "b", "a", "b", ("fact", "fact"))

    def test_scorer_requires_both_pair_kinds(self):
        case = PairedCase("n", NUISANCE_PAIR, "a", "a", "a", "a", ("noise",))
        with self.assertRaises(PairedInterventionValidationError):
            score_paired_cases([case])

    def test_scorer_rejects_duplicate_case_ids(self):
        nuisance = PairedCase("same", NUISANCE_PAIR, "a", "a", "a", "a", ("noise",))
        material = PairedCase("same", MATERIAL_PAIR, "a", "b", "a", "b", ("fact",))
        with self.assertRaises(PairedInterventionValidationError):
            score_paired_cases([nuisance, material])

    def test_exact_metrics_penalize_both_false_and_missed_interventions(self):
        cases = [
            PairedCase("n_good", NUISANCE_PAIR, "a", "a", "a", "a", ("noise",)),
            PairedCase("n_bad", NUISANCE_PAIR, "a", "a", "a", "b", ("noise",)),
            PairedCase("m_good", MATERIAL_PAIR, "a", "b", "a", "b", ("fact",)),
            PairedCase(
                "m_bad",
                MATERIAL_PAIR,
                "a",
                "b",
                "a",
                "a",
                ("critical_fact",),
                safety_critical=True,
            ),
        ]
        result = score_paired_cases(cases)
        self.assertEqual(result["canonical_accuracy"], 1.0)
        self.assertEqual(result["perturbed_accuracy"], 0.5)
        self.assertEqual(result["strict_paired_accuracy"], 0.5)
        self.assertEqual(result["nuisance_strict_pair_accuracy"], 0.5)
        self.assertEqual(result["material_strict_pair_accuracy"], 0.5)
        self.assertEqual(result["balanced_joint_score"], 0.5)
        self.assertEqual(result["joint_floor"], 0.5)
        self.assertEqual(result["false_intervention_rate"], 0.5)
        self.assertEqual(result["missed_intervention_rate"], 0.5)
        self.assertEqual(result["safety_critical_miss_count"], 1)
        self.assertFalse(result["all_safety_critical_detected"])

    def test_constant_output_cannot_pass_the_balanced_endpoint(self):
        result = score_paired_cases(
            [
                PairedCase("n", NUISANCE_PAIR, "a", "a", "constant", "constant", ("noise",)),
                PairedCase("m", MATERIAL_PAIR, "a", "b", "constant", "constant", ("fact",)),
            ]
        )
        self.assertEqual(result["nuisance_relation_accuracy"], 1.0)
        self.assertEqual(result["material_relation_accuracy"], 0.0)
        self.assertEqual(result["balanced_joint_score"], 0.0)
        self.assertEqual(result["joint_floor"], 0.0)


class PairedChelationFixtureTests(unittest.TestCase):
    def test_resource_budget_rejects_oversized_pair_grid(self):
        budget = PairedInterventionBudget(max_pairs=8)
        with self.assertRaises(PairedInterventionValidationError):
            estimate_paired_resources(
                dimension=5,
                pair_count=9,
                document_count=5,
                budget=budget,
            )

    def test_resource_budget_rejects_modeled_work(self):
        budget = PairedInterventionBudget(max_work_units=100)
        with self.assertRaises(PairedInterventionResourceError):
            estimate_paired_resources(
                dimension=5,
                pair_count=9,
                document_count=5,
                budget=budget,
            )

    def test_fixture_contains_both_singleton_and_second_order_material_pairs(self):
        fixture = build_paired_chelation_fixture()
        self.assertEqual(fixture["dimension"], 5)
        self.assertEqual(len(fixture["pairs"]), 9)
        material_orders = {
            len(pair["changed_features"])
            for pair in fixture["pairs"]
            if pair["pair_kind"] == MATERIAL_PAIR
        }
        self.assertEqual(material_orders, {1, 2})

    def test_production_variance_mask_matches_declared_nuisance_mask(self):
        fixture = build_paired_chelation_fixture()
        actual = production_variance_chelation_mask(fixture["calibration_cluster"], chelation_p=85)
        expected = np.ones(fixture["dimension"], dtype=np.float64)
        expected[fixture["collapse_dimension"]] = 0.0
        np.testing.assert_array_equal(actual, expected)

    def test_policy_mask_shape_and_range_are_fail_closed(self):
        fixture = build_paired_chelation_fixture()
        with self.assertRaises(PairedInterventionValidationError):
            evaluate_synthetic_policy(fixture, policy="bad", mask=np.ones(4))
        with self.assertRaises(PairedInterventionValidationError):
            evaluate_synthetic_policy(fixture, policy="bad", mask=np.array([1, 1, 1, 1, 2]))

    def test_frozen_suite_validates_only_the_sanity_boundary(self):
        result = run_paired_chelation_sanity()
        self.assertEqual(result["stage_id"], PAIRED_INTERVENTION_STAGE_ID)
        self.assertEqual(result["protocol_id"], PAIRED_INTERVENTION_PROTOCOL_ID)
        self.assertEqual(result["evidence_state"], "VALIDATED")
        self.assertEqual(result["scientific_claim_status"], "UNCONFIRMED")
        self.assertEqual(result["novelty_claim_status"], "UNCONFIRMED")
        self.assertEqual(result["parent_dependency_status"], "BLOCKED_ON_PRW-EK3")
        self.assertTrue(all(result["sanity"].values()))

    def test_production_control_passes_and_constant_output_control_is_caught(self):
        result = run_paired_chelation_sanity()
        policies = {row["policy"]: row["metrics"] for row in result["policies"]}
        production = policies["production_variance_chelation"]
        over_chelation = policies["over_chelation"]
        self.assertEqual(production["strict_paired_accuracy"], 1.0)
        self.assertEqual(production["balanced_joint_score"], 1.0)
        self.assertTrue(production["all_safety_critical_detected"])
        self.assertEqual(production["by_intervention_order"]["2"]["strict_accuracy"], 1.0)
        self.assertEqual(over_chelation["nuisance_relation_accuracy"], 1.0)
        self.assertEqual(over_chelation["material_relation_accuracy"], 0.0)
        self.assertEqual(over_chelation["balanced_joint_score"], 0.0)


class PairedInterventionArtifactTests(unittest.TestCase):
    def test_artifact_is_deterministic_json_and_atomic(self):
        first = make_paired_intervention_artifact()
        second = make_paired_intervention_artifact()
        self.assertIsInstance(first, PairedInterventionArtifact)
        self.assertEqual(first.as_dict(), second.as_dict())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "artifact.json"
            first.write_json(path)
            self.assertEqual(json.loads(path.read_text(encoding="utf-8")), first.as_dict())
            self.assertFalse(list(path.parent.glob("*.tmp")))

    def test_runner_writes_hash_bound_artifact_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            returned = run(root)
            artifact_path = root / "paired_intervention_sanity.json"
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(returned, manifest)
            self.assertEqual(manifest["evidence_state"], "VALIDATED")
            self.assertEqual(manifest["scientific_claim_status"], "UNCONFIRMED")
            entry = manifest["entries"][0]
            content = artifact_path.read_bytes()
            self.assertEqual(entry["byte_count"], len(content))
            self.assertEqual(entry["sha256"], hashlib.sha256(content).hexdigest())
            self.assertFalse(list(root.glob("*.tmp")))


if __name__ == "__main__":
    unittest.main()
