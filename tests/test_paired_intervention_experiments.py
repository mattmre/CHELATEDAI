import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from paired_intervention_experiments import (
    MATERIAL_PAIR,
    NUISANCE_PAIR,
    PAIRED_INTERVENTION_LIMITATIONS,
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
from run_paired_intervention_sanity import run, verify_result_dir
from verify_paired_intervention_sanity_v2_archive import verify_v2_archive


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

    def test_declared_feature_count_is_metadata_not_intervention_order(self):
        fixture = build_paired_chelation_fixture()
        self.assertEqual(fixture["dimension"], 5)
        self.assertEqual(len(fixture["pairs"]), 9)
        declared_counts = {
            len(pair["changed_features"])
            for pair in fixture["pairs"]
            if pair["pair_kind"] == MATERIAL_PAIR
        }
        self.assertEqual(declared_counts, {1, 2})
        scored = score_paired_cases(
            [
                PairedCase("n", NUISANCE_PAIR, "a", "a", "a", "a", ("noise",)),
                PairedCase(
                    "m",
                    MATERIAL_PAIR,
                    "a",
                    "b",
                    "a",
                    "b",
                    ("declared_a", "declared_b", "declared_c"),
                ),
            ]
        )
        self.assertIn("3", scored["by_declared_changed_feature_count"])
        self.assertNotIn("by_intervention_order", scored)

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
        self.assertEqual(result["status"], "COMPLETE")
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
        self.assertEqual(
            production["by_declared_changed_feature_count"]["2"]["strict_accuracy"],
            1.0,
        )
        self.assertEqual(over_chelation["nuisance_relation_accuracy"], 1.0)
        self.assertEqual(over_chelation["material_relation_accuracy"], 0.0)
        self.assertEqual(over_chelation["balanced_joint_score"], 0.0)


class PairedInterventionArtifactTests(unittest.TestCase):
    def test_retained_v2_archive_has_callable_custody_verifier(self):
        root = Path("artifacts/method-dev/isi1-paired-intervention-sanity-v2")
        result = verify_v2_archive(root)
        self.assertEqual(result["status"], "ARCHIVED_V2_VERIFIED")
        self.assertTrue(result["custody_only"])
        self.assertFalse(result["semantic_regeneration"])

    def test_v2_archive_verifier_rejects_tamper_and_extra_member(self):
        source = Path("artifacts/method-dev/isi1-paired-intervention-sanity-v2")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "v2"
            shutil.copytree(source, root)
            artifact = root / "paired_intervention_sanity.json"
            artifact.write_bytes(artifact.read_bytes().replace(b'"VALIDATED"', b'"REJECTED"'))
            with self.assertRaisesRegex(PairedInterventionValidationError, "custody"):
                verify_v2_archive(root)
            shutil.rmtree(root)
            shutil.copytree(source, root)
            (root / "unexpected.txt").write_text("x", encoding="utf-8")
            with self.assertRaisesRegex(PairedInterventionValidationError, "file set"):
                verify_v2_archive(root)

    def test_artifact_construction_rejects_nondefault_budget(self):
        budget = PairedInterventionBudget(max_seconds=29.0)
        result = run_paired_chelation_sanity(budget=budget)
        with self.assertRaisesRegex(
            PairedInterventionValidationError,
            "exact frozen default budget",
        ):
            PairedInterventionArtifact.create(
                budget=budget,
                result=result,
                limitations=PAIRED_INTERVENTION_LIMITATIONS,
            )

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

    def test_artifact_construction_rejects_forged_result(self):
        with self.assertRaisesRegex(
            PairedInterventionValidationError,
            "does not exactly match regenerated",
        ):
            PairedInterventionArtifact.create(
                budget=PairedInterventionBudget(),
                result={
                    "stage_id": PAIRED_INTERVENTION_STAGE_ID,
                    "evidence_state": "VALIDATED",
                    "scientific_claim_status": "CONFIRMED",
                },
                limitations=("forged",),
            )

    def test_runner_writes_hash_bound_artifact_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "result"
            returned = run(root)
            artifact_path = root / "paired_intervention_sanity.json"
            manifest_path = root / "manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(returned, manifest)
            self.assertEqual(manifest["evidence_state"], "VALIDATED")
            self.assertEqual(manifest["scientific_claim_status"], "UNCONFIRMED")
            self.assertEqual(manifest["novelty_claim_status"], "UNCONFIRMED")
            self.assertEqual(manifest["status"], "VALIDATED_SYNTHETIC_SANITY_ONLY")
            entry = manifest["entries"][0]
            content = artifact_path.read_bytes()
            self.assertEqual(entry["byte_count"], len(content))
            self.assertEqual(entry["sha256"], hashlib.sha256(content).hexdigest())
            self.assertFalse(list(root.glob("*.tmp")))
            self.assertEqual(verify_result_dir(root), manifest)

    def test_runner_refuses_existing_nonempty_destination_without_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "result"
            root.mkdir()
            sentinel = root / "sentinel.txt"
            sentinel.write_text("preserve", encoding="utf-8")
            with self.assertRaises(FileExistsError):
                run(root)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "preserve")
            self.assertEqual([path.name for path in root.iterdir()], ["sentinel.txt"])

    def test_public_verifier_rejects_tamper_reseal_and_extra_file(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "result"
            run(root)
            artifact_path = root / "paired_intervention_sanity.json"
            manifest_path = root / "manifest.json"
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            artifact["result"]["scientific_claim_status"] = "CONFIRMED"
            payload = dict(artifact)
            payload.pop("artifact_digest")
            artifact["artifact_digest"] = hashlib.sha256(
                json.dumps(
                    payload,
                    sort_keys=True,
                    separators=(",", ":"),
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            artifact_path.write_text(
                json.dumps(artifact, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            content = artifact_path.read_bytes()
            manifest["artifact_digest"] = artifact["artifact_digest"]
            manifest["entries"][0]["byte_count"] = len(content)
            manifest["entries"][0]["sha256"] = hashlib.sha256(content).hexdigest()
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(PairedInterventionValidationError):
                verify_result_dir(root)
            shutil.rmtree(root)
            run(root)
            (root / "unexpected.txt").write_text("unexpected", encoding="utf-8")
            with self.assertRaisesRegex(PairedInterventionValidationError, "file set"):
                verify_result_dir(root)

    def test_failure_before_publication_leaves_no_final_or_stage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "result"
            with patch(
                "run_paired_intervention_sanity._write_atomic",
                side_effect=OSError("forced manifest failure"),
            ):
                with self.assertRaisesRegex(OSError, "forced manifest failure"):
                    run(root)
            self.assertFalse(root.exists())
            self.assertFalse(list(root.parent.glob(".result.stage-*")))

    def test_public_verifier_rejects_hidden_staged_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "result"
            run(root)
            staged = root.parent / ".result.stage-orphan"
            root.rename(staged)
            with self.assertRaisesRegex(PairedInterventionValidationError, "staging"):
                verify_result_dir(staged)

    def test_public_verifier_rejects_reparse_root_before_reading_members(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "result"
            run(root)
            with patch(
                "run_paired_intervention_sanity._is_reparse_point",
                side_effect=lambda path: Path(path) == root,
            ):
                with self.assertRaisesRegex(PairedInterventionValidationError, "ordinary"):
                    verify_result_dir(root)

    @unittest.skipUnless(os.name == "nt", "Windows junction regression")
    def test_public_verifier_rejects_actual_windows_junction_stage_alias(self):
        with tempfile.TemporaryDirectory() as directory:
            parent = Path(directory)
            final = parent / "result"
            run(final)
            stage = parent / ".result.stage-orphan"
            final.rename(stage)
            alias = parent / "published-alias"
            completed = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(alias), str(stage)],
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode != 0:
                self.skipTest(f"mklink /J unavailable: {completed.stderr or completed.stdout}")
            try:
                with self.assertRaisesRegex(PairedInterventionValidationError, "ordinary"):
                    verify_result_dir(alias)
            finally:
                os.rmdir(alias)


if __name__ == "__main__":
    unittest.main()
