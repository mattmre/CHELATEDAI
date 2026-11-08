from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from egv.canonical import canonical_bytes
from egv.cli import main
from egv.evaluation.diagnostics import Diagnostic
from egv.evaluation.learnable import LearnableCorpusV2
from egv.evaluation.learnable_runtime import (
    CORPUS_VALIDATION_SCOPE,
    _corpus_validation_accepted,
    build_corpus_validation_permit,
    build_learnable_hidden_runner,
    run_production_solvability_proof,
    runtime_manifest,
)
from egv.evaluation.sandbox import DockerSandboxConfig
from egv.errors import EGVError


class LearnableProductionRuntimeTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.seed = self.root / "seed.bin"
        self.seed.write_bytes(bytes(range(32)))
        self.corpus = LearnableCorpusV2.generate(secret_seed_file=self.seed)
        self.config = DockerSandboxConfig()

    def test_permit_is_narrow_and_binds_public_private_and_runtime_manifests(self):
        permit = build_corpus_validation_permit(self.corpus, self.config)
        permit.validate(self.corpus, self.config)
        self.assertEqual(permit.scope, CORPUS_VALIDATION_SCOPE)
        self.assertNotIn("endpoint", permit.public_record())
        with self.assertRaisesRegex(ValueError, "public corpus"):
            replace(permit, public_manifest_digest="0" * 64).validate(self.corpus, self.config)
        with self.assertRaisesRegex(ValueError, "task-bearing"):
            replace(permit, scope="GO_TASK_BEARING").validate(self.corpus, self.config)

    def test_runtime_manifest_is_closed_and_topology_free(self):
        manifest = runtime_manifest(self.config)
        self.assertEqual(manifest["network"], "none")
        self.assertEqual(manifest["backend"], "docker-enforced-v1")
        serialized = str(manifest).lower()
        for forbidden in ("endpoint", "hostname", "private_path", "password", "token"):
            self.assertNotIn(forbidden, serialized)

    def test_hidden_runner_addresses_two_cases_per_blind_task_without_values(self):
        tasks = self.corpus.split("blind")
        runner, addresses = build_learnable_hidden_runner(tasks)
        self.assertEqual(len(tasks), 18)
        self.assertEqual(len(addresses), 36)
        self.assertEqual(len({address.controller_task_id for address in addresses}), 36)
        by_public = {task.task_id: task for task in tasks}
        for address in addresses:
            task = by_public[address.public_task_id]
            case = task.hidden_cases[address.case_ordinal - 1]
            expected = canonical_bytes(case.expected_output) + b"\n"
            passed = runner.evaluate(address.controller_task_id, expected, opaque_input=case.input_value)
            failed = runner.evaluate(address.controller_task_id, b"{}\n", opaque_input=case.input_value)
            self.assertEqual(passed.diagnostic_enum, Diagnostic.PASS.value)
            self.assertEqual(failed.diagnostic_enum, Diagnostic.WRONG_OUTPUT.value)
            public_record = address.public_record()
            rendered = str(public_record)
            self.assertNotIn(str(case.input_value["case_nonce"]), rendered)
            self.assertNotIn(str(case.expected_output), rendered)

    def test_case_addresses_are_stable_and_split_disjoint(self):
        first_runner, first = build_learnable_hidden_runner(self.corpus.split("blind"))
        second_runner, second = build_learnable_hidden_runner(self.corpus.split("blind"))
        train_runner, train = build_learnable_hidden_runner(self.corpus.split("train"))
        self.assertIsNotNone(first_runner)
        self.assertIsNotNone(second_runner)
        self.assertIsNotNone(train_runner)
        self.assertEqual(first, second)
        self.assertFalse(
            {address.controller_task_id for address in first} & {address.controller_task_id for address in train}
        )

    def test_direct_in_process_production_proof_is_disabled(self):
        permit = build_corpus_validation_permit(self.corpus, self.config)
        with self.assertRaisesRegex(EGVError, "three spawned stages"):
            run_production_solvability_proof(
                self.corpus,
                self.root / "proof",
                permit=permit,
                splits=("dev",),
                config=self.config,
            )

    def test_production_acceptance_requires_every_proof_surface(self):
        accepted = {
            "reference_pass_count": 36,
            "reference_total": 36,
            "starter_rejection_count": 18,
            "starter_total": 18,
            "receipt_count": 162,
            "receipt_delivery_count": 162,
            "receipt_chain_valid": True,
            "negative_controls": {"network": True, "source_read_only": True},
        }
        self.assertTrue(_corpus_validation_accepted(**accepted))
        mutations = {
            "reference_pass_count": 35,
            "starter_rejection_count": 17,
            "receipt_count": 159,
            "receipt_delivery_count": 159,
            "receipt_chain_valid": False,
            "negative_controls": {"network": False},
        }
        for field, value in mutations.items():
            with self.subTest(field=field):
                rejected = dict(accepted)
                rejected[field] = value
                self.assertFalse(_corpus_validation_accepted(**rejected))

    def test_cli_routes_external_paths_to_spawned_gate_without_constructing_corpus(self):
        key_paths = []
        for name in ("controller-private", "controller-public", "freeze-private", "freeze-public"):
            path = self.root / (name + ".key")
            path.write_bytes(b"k" * 32)
            key_paths.append(path)
        workspace = self.root / "cli-workspace"
        output = self.root / "cli-report.json"
        report = {
            "production_acceptance": True,
            "disposition": "GO_CORPUS_VALIDATION",
            "result_digest": "a" * 64,
            "total_evaluation_count": 394,
            "receipt_count": 1182,
            "task_bearing_model_requests_permitted": False,
        }
        with (
            patch(
                "egv.cli.DockerSandboxConfig.from_environment",
                return_value=self.config,
            ),
            patch("egv.cli.run_gate_a_corpus_validation", return_value=report) as run,
            patch(
                "egv.cli.LearnableCorpusV2",
                create=True,
            ) as corpus_type,
        ):
            code = main(
                [
                    "evaluation",
                    "corpus-proof",
                    "--evaluator-seed",
                    str(self.seed),
                    "--controller-private-key",
                    str(key_paths[0]),
                    "--controller-public-key",
                    str(key_paths[1]),
                    "--freeze-private-key",
                    str(key_paths[2]),
                    "--freeze-public-key",
                    str(key_paths[3]),
                    "--workspace",
                    str(workspace),
                    "--output",
                    str(output),
                    "--package-root",
                    str(Path.cwd()),
                    "--source-commit",
                    "a" * 40,
                    "--json",
                ]
            )
        self.assertEqual(code, 0)
        run.assert_called_once()
        corpus_type.generate.assert_not_called()


if __name__ == "__main__":
    unittest.main()
