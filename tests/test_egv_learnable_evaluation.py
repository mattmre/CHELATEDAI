from dataclasses import replace
import tempfile
from pathlib import Path
import unittest

from egv.canonical import canonical_json, digest_for
from egv.evaluation.learnable import (
    FAMILY_SPECS_V2,
    HIDDEN_CASES_PER_TASK,
    INCOMPATIBLE_V1_VERSION,
    LEARNABLE_DATA_VERSION,
    LEARNABLE_LOCAL_PROOF_VERSION,
    SPLITS_V2,
    STAGE_A_TASK_IDS,
    LearnableCase,
    LearnableCorpusV2,
    LearnableTask,
    build_reference_source,
    run_local_solvability_proof,
    solve_public_spec,
    validate_learnable_manifest,
    validate_learnable_private_manifest,
    validate_model_visible_package,
    validate_public_task_record,
)
from egv.evaluation.sandbox import validate_candidate_source_contract


class LearnableEvaluationTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.seed = bytes(range(32))
        cls.seed_path = cls.root / "learnable-seed.bin"
        cls.seed_path.write_bytes(cls.seed)
        cls.corpus = LearnableCorpusV2.generate(secret_seed_file=cls.seed_path)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()


class TestLearnableCorpusContract(LearnableEvaluationTestCase):
    def test_frozen_allocations_and_stage_a_order(self) -> None:
        self.assertEqual(len(self.corpus.tasks), 170)
        self.assertEqual(
            {split: len(self.corpus.split(split)) for split in SPLITS_V2},
            {"train": 20, "dev": 12, "heldout": 120, "blind": 18},
        )
        for family in FAMILY_SPECS_V2:
            self.assertEqual(
                {split: len([task for task in self.corpus.split(split) if task.family_id == family.family_id]) for split in SPLITS_V2},
                {split: family.count(split) for split in SPLITS_V2},
            )
        stage_a = self.corpus.stage_a_tasks()
        self.assertEqual(tuple(task.task_id for task in stage_a), STAGE_A_TASK_IDS)
        self.assertEqual(tuple(task.family_id for task in stage_a), tuple(family.family_id for family in FAMILY_SPECS_V2))
        self.assertTrue(all(task.split == "train" and task.ordinal == 1 for task in stage_a))

    def test_seed_changes_cases_not_public_problem_or_reference(self) -> None:
        second_seed = self.root / "learnable-seed-second.bin"
        second_seed.write_bytes(bytes(reversed(range(32))))
        second = LearnableCorpusV2.generate(secret_seed_file=second_seed)
        self.assertEqual(self.corpus.public_manifest(), second.public_manifest())
        self.assertNotEqual(self.corpus.private_manifest(), second.private_manifest())
        self.assertNotEqual(
            self.corpus.tasks[0].hidden_cases[0].input_digest,
            second.tasks[0].hidden_cases[0].input_digest,
        )
        self.assertEqual(
            build_reference_source(self.corpus.tasks[0].public_record),
            build_reference_source(second.tasks[0].public_record),
        )

    def test_every_target_is_exactly_public_spec_plus_runtime_input(self) -> None:
        for task in self.corpus.tasks:
            for case in task.hidden_cases:
                self.assertEqual(
                    case.expected_output,
                    solve_public_spec(task.public_record["public_spec"], case.input_value),
                )
                self.assertEqual(case.expected_output_digest, digest_for(case.expected_output))

    def test_public_and_private_manifests_are_closed_and_bound(self) -> None:
        public = validate_learnable_manifest(self.corpus.public_manifest(), self.corpus)
        private = validate_learnable_private_manifest(self.corpus.private_manifest(), self.corpus)
        self.assertEqual(public["schema_version"], LEARNABLE_DATA_VERSION)
        self.assertEqual(public["incompatible_with"], [INCOMPATIBLE_V1_VERSION])
        self.assertNotIn("seed_digest", public)
        private_text = canonical_json(private)
        self.assertNotIn("input_value", private_text)
        self.assertNotIn("expected_output\"", private_text)
        self.assertNotIn(self.corpus.tasks[0].hidden_cases[0].input_value["case_nonce"], private_text)

    def test_v1_is_explicitly_incompatible(self) -> None:
        with self.assertRaisesRegex(ValueError, "v1 is explicitly incompatible"):
            validate_learnable_manifest({"schema_version": INCOMPATIBLE_V1_VERSION})

    def test_generation_is_byte_deterministic(self) -> None:
        regenerated = LearnableCorpusV2.generate(secret_seed_file=self.seed_path)
        self.assertEqual(canonical_json(self.corpus.public_manifest()), canonical_json(regenerated.public_manifest()))
        self.assertEqual(canonical_json(self.corpus.private_manifest()), canonical_json(regenerated.private_manifest()))
        self.assertEqual(
            [case.input_value for task in self.corpus.tasks for case in task.hidden_cases],
            [case.input_value for task in regenerated.tasks for case in task.hidden_cases],
        )


class TestLearnablePublicOnlySolvability(LearnableEvaluationTestCase):
    def test_reference_sources_are_public_only_bound_and_source_safe(self) -> None:
        for task in self.corpus.tasks:
            source = build_reference_source(task.public_record)
            self.assertIsNone(validate_candidate_source_contract(source))
            self.assertEqual(task.public_record["reference_source_digest"], __import__("hashlib").sha256(source).hexdigest())
            source_text = source.decode("utf-8")
            for case in task.hidden_cases:
                self.assertNotIn(case.input_value["case_nonce"], source_text)
                self.assertNotIn(canonical_json(case.expected_output), source_text)

    def test_blind_model_visible_package_passes_every_hidden_case_locally(self) -> None:
        report = run_local_solvability_proof(self.corpus, self.root / "local-proof", splits=("blind",))
        self.assertEqual(report["schema_version"], LEARNABLE_LOCAL_PROOF_VERSION)
        self.assertEqual(report["evidence_scope"], "local-test-helper-proof-not-production-acceptance")
        self.assertFalse(report["production_acceptance"])
        self.assertEqual(report["runtime_backend"], "local-test-helper")
        self.assertEqual(report["task_count"], 18)
        self.assertEqual(report["hidden_case_count"], 18 * HIDDEN_CASES_PER_TASK)
        self.assertEqual(report["builder_input"], "closed-model-visible-task-record-only")
        self.assertFalse(report["hidden_cases_passed_to_builder"])
        self.assertTrue(report["reference_all_hidden_cases_passed"])
        self.assertEqual(report["reference_failure_count"], 0)
        self.assertTrue(report["non_reference_public_only_candidate_rejected"])
        self.assertEqual(report["non_reference_semantic_pass_count"], 0)

    def test_model_visible_package_contains_no_hidden_case_material(self) -> None:
        task = self.corpus.split("blind")[0]
        package = validate_model_visible_package(self.corpus.model_visible_package(task.task_id))
        package_text = canonical_json(package)
        for case in task.hidden_cases:
            self.assertNotIn(case.input_value["case_nonce"], package_text)
            self.assertNotIn(canonical_json(case.input_value), package_text)
            self.assertNotIn(canonical_json(case.expected_output), package_text)

    def test_builder_rejects_hidden_case_leakage(self) -> None:
        task = self.corpus.split("blind")[0]
        leaking = dict(task.public_record)
        leaking["hidden_cases"] = [task.hidden_cases[0].private_record()]
        with self.assertRaisesRegex(ValueError, "not a closed record"):
            build_reference_source(leaking)


class TestLearnableAdversarialValidation(LearnableEvaluationTestCase):
    def test_secret_dependent_target_is_rejected(self) -> None:
        task = self.corpus.tasks[0]
        case = task.hidden_cases[0]
        secret_target = {"secret_oracle": self.seed.hex()}
        poisoned_case = replace(
            case,
            expected_output=secret_target,
            expected_output_digest=digest_for(secret_target),
        )
        poisoned_task = LearnableTask(task.public_record, (poisoned_case,) + task.hidden_cases[1:])
        tasks = list(self.corpus.tasks)
        tasks[0] = poisoned_task
        with self.assertRaisesRegex(ValueError, "secret-dependent target rejected"):
            LearnableCorpusV2(tasks, self.seed)

    def test_tampered_public_spec_and_digest_fail_closed(self) -> None:
        task = self.corpus.tasks[0]
        tampered_spec = dict(task.public_record)
        public_spec = dict(tampered_spec["public_spec"])
        parameters = dict(public_spec["parameters"])
        parameters["bias"] += 1
        public_spec["parameters"] = parameters
        tampered_spec["public_spec"] = public_spec
        with self.assertRaisesRegex(ValueError, "public spec differs"):
            validate_public_task_record(tampered_spec)

        tampered_digest = dict(task.public_record)
        tampered_digest["reference_source_digest"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "reference source digest mismatch"):
            validate_public_task_record(tampered_digest)

    def test_manifest_reordering_and_tamper_fail_closed(self) -> None:
        manifest = self.corpus.public_manifest()
        reordered = dict(manifest)
        reordered["tasks"] = list(reversed(manifest["tasks"]))
        reordered["manifest_digest"] = digest_for({key: value for key, value in reordered.items() if key != "manifest_digest"})
        with self.assertRaisesRegex(ValueError, "task order or identities"):
            validate_learnable_manifest(reordered)

        extra = dict(manifest)
        extra["private_answers"] = []
        with self.assertRaisesRegex(ValueError, "not a closed record"):
            validate_learnable_manifest(extra)

    def test_split_nonce_overlap_is_rejected_even_with_valid_public_target(self) -> None:
        train_task = self.corpus.split("train")[0]
        dev_task = self.corpus.split("dev")[0]
        train_nonce = train_task.hidden_cases[0].input_value["case_nonce"]
        original = dev_task.hidden_cases[0]
        overlapping_input = dict(original.input_value)
        overlapping_input["case_nonce"] = train_nonce
        expected = solve_public_spec(dev_task.public_record["public_spec"], overlapping_input)
        input_digest = digest_for(overlapping_input)
        overlapping = LearnableCase(
            case_id="case_{}".format(
                digest_for({"task_id": dev_task.task_id, "ordinal": original.ordinal, "input_digest": input_digest})
            ),
            ordinal=original.ordinal,
            input_value=overlapping_input,
            expected_output=expected,
            input_digest=input_digest,
            expected_output_digest=digest_for(expected),
        )
        replacement = LearnableTask(dev_task.public_record, (overlapping,) + dev_task.hidden_cases[1:])
        tasks = [replacement if task.task_id == dev_task.task_id else task for task in self.corpus.tasks]
        with self.assertRaisesRegex(ValueError, "split overlap in case_nonces"):
            LearnableCorpusV2(tasks, self.seed)

    def test_tampered_private_digest_fails_closed(self) -> None:
        manifest = self.corpus.private_manifest()
        tampered = dict(manifest)
        tampered["seed_digest"] = "0" * 64
        with self.assertRaisesRegex(ValueError, "private manifest digest mismatch"):
            validate_learnable_private_manifest(tampered)


if __name__ == "__main__":
    unittest.main()
