"""Focused unit and integration coverage for the complete Evaluation slice."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from egv.canonical import canonical_bytes, content_id, digest_bytes, digest_for, failure_family_root
from egv.evaluation.artifacts import (
    ContentAddressedArtifactStore,
    assert_byte_identical_regeneration,
    freeze_evaluation,
    scan_public_artifacts,
)
from egv.evaluation.authority import (
    AuthorityBroker,
    AuthorityDenied,
    AuthorityPolicy,
    DockerEnforcedRuntime,
    LocalEnforcedRuntime,
    OpenShellRuntime,
)
from egv.evaluation.controller import EvaluatorController, HiddenEvaluatorRunner
from egv.evaluation.dataset import EvaluationCorpus, FAMILY_SPECS, SPLITS, validate_data_manifest
from egv.evaluation.diagnostics import DIAGNOSTIC_ENUM, Diagnostic, failure_family_for_attempt
from egv.evaluation.errors import ArtifactError, DockerConfigurationError, EvaluationError, InfrastructureFailure, LeakageError, ShockMismatchError
from egv.evaluation.prompts import PROMPT_IDS, PromptRegistry
from egv.evaluation.sandbox import (
    DockerCandidateSandbox,
    DockerSandboxConfig,
    LocalTestSandbox,
    RESOURCE_BOUND_MEMORY_LIMIT,
    SandboxResult,
    validate_candidate_source_contract,
)
import egv.evaluation.sandbox as sandbox_module
from egv.evaluation.sft import SFT_FIELDS, build_sft_row, validate_sft_row
from egv.evaluation.shock import POST_SHOCK_ATTEMPTS, SHOCK_POLICIES, CorrectionShockSuite, ShockFixture, ShockProfile, graph_precision_recall
from egv.evaluation.smoke import run_evaluation_smoke
from egv.evaluation.process_smoke import run_evaluation_two_process_smoke
from egv.evaluation import run_evaluation_smoke as package_smoke
from egv.errors import ProjectionError, ReceiptVerificationError
from egv.ledger import EvidenceLedger
from egv.pilot import run_two_process_smoke
from egv.projection import InMemoryProjection
from egv.receipts import ReceiptJournal, ReceiptSigner, verify_receipt


class EvaluationTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory(prefix="egv-evaluation-test-")
        self.root = Path(self.tempdir.name)
        self.seed_path = self.root / "evaluator-private" / "corpus-seed.bin"
        self.seed_path.parent.mkdir(parents=True, exist_ok=True)
        self.seed_path.write_bytes(b"E" * 32)
        self.seed_path.chmod(0o600)
        self.corpus = EvaluationCorpus.generate(secret_seed_file=self.seed_path)

    def tearDown(self) -> None:
        self.tempdir.cleanup()


class TestEvaluationCorpus(EvaluationTestCase):
    def test_exact_split_family_distribution_and_immutable_ids(self) -> None:
        self.assertEqual(len(self.corpus.repositories), 36)
        self.assertEqual({split: len(self.corpus.split(split)) for split in SPLITS}, {"train": 20, "dev": 8, "heldout": 8})
        expected = {
            family.family_id: {split: family.count(split) for split in SPLITS}
            for family in FAMILY_SPECS
        }
        actual = {
            family: {
                split: sum(repo.family_id == family and repo.split == split for repo in self.corpus.repositories)
                for split in SPLITS
            }
            for family in expected
        }
        self.assertEqual(actual, expected)
        ids = [repo.template_id for repo in self.corpus.repositories]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertTrue(all(repo.template_id.endswith("-v1") for repo in self.corpus.repositories))
        self.assertEqual(set(repo.template_id for repo in self.corpus.trainer_repositories()) & set(repo.template_id for repo in self.corpus.hidden_repositories()), set())
        self.assertEqual([repo.ordinal for repo in self.corpus.split("heldout")], [1, 1, 1, 1, 1, 2, 1, 2])

    def test_source_and_hidden_views_are_split_disjoint(self) -> None:
        trainer_ids = {repo.template_id for repo in self.corpus.trainer_repositories()}
        hidden_ids = {repo.template_id for repo in self.corpus.hidden_repositories()}
        trainer_bytes = b"\n".join(data for repo in self.corpus.trainer_repositories() for _path, data in repo.source_files)
        self.assertNotIn(b"expected_output", trainer_bytes)
        self.assertNotIn(b"golden_patch", trainer_bytes)
        self.assertNotIn(b"hidden_rule_id", trainer_bytes)
        self.assertFalse(hidden_ids & trainer_ids)
        self.assertTrue(all(repo.hidden_spec_digest == digest_for(repo.hidden_spec) for repo in self.corpus.repositories))
        disjoint = self.corpus.validate_split_disjointness()
        self.assertIn("corrected_source_secret_tokens", disjoint["checked_fields"])
        self.assertIn("corrected_source_bytes", disjoint["checked_content"])
        self.assertIn("task_source_bytes", disjoint["checked_content"])
        split_sources = {
            split: {repo.corrected_source for repo in self.corpus.split(split)} for split in SPLITS
        }
        for left_index, left in enumerate(SPLITS):
            for right in SPLITS[left_index + 1 :]:
                self.assertFalse(split_sources[left] & split_sources[right])

    def test_data_manifest_is_deterministic_and_closed(self) -> None:
        manifest = self.corpus.manifest()
        self.assertEqual(validate_data_manifest(manifest, self.corpus), manifest)
        tampered = dict(manifest)
        tampered["tasks"] = list(manifest["tasks"])
        tampered["tasks"][0] = dict(tampered["tasks"][0], source_digest=digest_for("tampered"))
        with self.assertRaises(ValueError):
            validate_data_manifest(tampered, self.corpus)
        tampered = dict(manifest, task_count=35)
        with self.assertRaises(ValueError):
            validate_data_manifest(tampered, self.corpus)

    def test_public_generation_cannot_reconstruct_private_expected_values(self) -> None:
        with self.assertRaises(EvaluationError):
            EvaluationCorpus.generate()
        self.assertEqual(self.corpus.validate_split_disjointness()["pairwise_disjoint"], True)
        heldout = self.corpus.hidden_repositories()[0]
        public_text = b"\n".join(data for _path, data in heldout.source_files).decode("utf-8")
        self.assertNotIn(heldout.hidden_spec["hidden_rule_id"], public_text)
        self.assertNotIn(heldout.hidden_spec["golden_patch"], public_text)
        self.assertNotIn(str(heldout.expected_output), public_text)

    def test_seed_free_prefix_rewrite_cannot_pass_the_six_heldout_families(self) -> None:
        """Regression for the historical public-prefix 6/6 oracle."""

        runner = HiddenEvaluatorRunner.from_corpus(self.corpus)
        sandbox = LocalTestSandbox(self.root / "seed-free-prefix")
        public_only_sources = {
            "PURE_FUNCTION": b"def main(value):\n    return 'answer-' + str(value).removeprefix('input-')\n",
            "PARSER_EDGE": b"def main(value):\n    return ['alpha', value[len('alpha'):-len('omega')], 'omega']\n",
            "STATE_TRANSITION": b"def main(value):\n    return {'state': value, 'transition': 'transition-' + value.removeprefix('state-')}\n",
            "DATA_TRANSFORM": b"def main(value):\n    return [row['private'] for row in value]\n",
            "RESOURCE_BOUND": b"def main(value):\n    return 'bounded-' + value[0].removeprefix('resource-')\n",
            "DEPENDENCY_CONTRACT": b"def main(value):\n    return 'value-' + value.removeprefix('key-')\n",
        }
        verdicts = {}
        for repo in self.corpus.hidden_repositories():
            public_text = b"\n".join(data for _path, data in repo.source_files).decode("utf-8")
            self.assertNotIn(repo.hidden_spec["oracle_token"], public_text)
            self.assertNotIn(repo.hidden_spec["oracle_token"], str(repo.evaluator_input))
            result = sandbox.execute(
                public_only_sources[repo.family_id],
                repo.evaluator_input,
                artifact_digest=digest_bytes(public_only_sources[repo.family_id]),
                candidate_id="seed-free-" + repo.template_id,
            )
            verdict = runner.evaluate(repo.template_id, result.output_bytes, opaque_input=repo.evaluator_input)
            verdicts[repo.family_id] = verdict.diagnostic_enum
        self.assertEqual(set(verdicts), {spec.family_id for spec in FAMILY_SPECS})
        self.assertEqual(set(verdicts.values()), {"WRONG_OUTPUT"})

    def test_repair_algorithms_are_structurally_split_disjoint(self) -> None:
        for family in FAMILY_SPECS:
            by_split = {
                split: next(repo for repo in self.corpus.split(split) if repo.family_id == family.family_id)
                for split in SPLITS
            }
            self.assertEqual(
                len({repo.private_material["repair_algorithm_id"] for repo in by_split.values()}),
                3,
            )
            self.assertEqual(
                len({repo.private_material["repair_algorithm_skeleton"] for repo in by_split.values()}),
                3,
            )


class TestFrozenArtifacts(EvaluationTestCase):
    def test_regeneration_is_byte_identical_and_public_scan_is_clean(self) -> None:
        first = self.root / "first"
        second = self.root / "second"
        report = freeze_evaluation(first, corpus=self.corpus)
        freeze_evaluation(second, corpus=self.corpus)
        heldout_ids = {repo.template_id for repo in self.corpus.hidden_repositories()}
        assert_byte_identical_regeneration(first, second)
        self.assertEqual(report.split_counts, {"train": 20, "dev": 8, "heldout": 8})
        self.assertEqual(scan_public_artifacts(first, self.corpus)["findings"], [])
        public_freeze_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (first / "frozen").rglob("*")
            if path.is_file()
        )
        self.assertTrue(all(task_id not in public_freeze_text for task_id in heldout_ids))
        self.assertTrue((first / "evaluator-private" / "data-manifest.json").exists())
        root_manifest_text = (first / "artifact-manifest.json").read_text(encoding="utf-8")
        self.assertNotIn("evaluator-private", root_manifest_text)
        self.assertNotIn("/home/", root_manifest_text)
        self.assertNotIn("/tmp/", root_manifest_text)
        self.assertEqual(
            set(path.relative_to(first).parts[0] for path in first.rglob("*") if path.is_file()),
            {"frozen", "trainer", "public", "evaluator-private", "artifact-manifest.json"},
        )
        trainer_ids = {repo.template_id for repo in self.corpus.trainer_repositories()}
        trainer_text = "\n".join(
            path.read_text(encoding="utf-8")
            for path in (first / "trainer").rglob("*")
            if path.is_file()
        )
        self.assertTrue(all(task_id in trainer_text for task_id in trainer_ids))
        self.assertTrue(all(task_id not in trainer_text for task_id in heldout_ids))
        self.assertNotIn("expected_output", trainer_text)
        self.assertNotIn("golden_patch", trainer_text)
        self.assertNotIn("PRIVATE_SHAPE_", trainer_text)
        self.assertFalse((first / "trainer" / "corpus-seed.bin").exists())

    def test_public_leakage_scanner_rejects_hidden_ids_and_private_material(self) -> None:
        root = self.root / "freeze"
        freeze_evaluation(root, corpus=self.corpus)
        heldout_id = self.corpus.hidden_repositories()[0].template_id
        (root / "trainer" / "leak.txt").write_text(heldout_id, encoding="utf-8")
        with self.assertRaises(LeakageError):
            scan_public_artifacts(root, self.corpus)
        (root / "trainer" / "leak.txt").write_text("safe", encoding="utf-8")
        (root / "public" / "credential.txt").write_text("API_KEY=not-a-real-key", encoding="utf-8")
        with self.assertRaises(LeakageError):
            scan_public_artifacts(root, self.corpus)

    def test_public_scan_without_corpus_uses_closed_contract_ids(self) -> None:
        root = self.root / "contract-id-freeze"
        freeze_evaluation(root, corpus=self.corpus)
        report = scan_public_artifacts(root)
        self.assertTrue(report["heldout_ids_absent"])
        self.assertTrue(report["heldout_ids_checked"])
        self.assertEqual(report["heldout_ids_status"], "CHECKED_PUBLIC_CONTRACT")
        (root / "trainer" / "planted-heldout.txt").write_text("egv-pure_function-heldout-1-v1", encoding="utf-8")
        with self.assertRaises(LeakageError):
            scan_public_artifacts(root)

    def test_content_addressed_store_is_write_once_and_verifies_bytes(self) -> None:
        store = ContentAddressedArtifactStore(self.root / "objects")
        ref = store.put(b"immutable candidate", media_type="text/plain", role="candidate")
        self.assertEqual(store.put(b"immutable candidate", media_type="text/plain", role="candidate"), ref)
        self.assertEqual(store.read(ref.digest), b"immutable candidate")
        target = store.root / ref.relative_path
        target.chmod(0o644)
        target.write_bytes(b"tampered")
        with self.assertRaises(ArtifactError):
            store.read(ref.digest)


class TestPromptsAndSFT(EvaluationTestCase):
    def test_five_prompt_templates_are_immutable_and_context_closed(self) -> None:
        registry = PromptRegistry()
        self.assertEqual(registry.manifest()["template_ids"], list(PROMPT_IDS))
        with self.assertRaises(KeyError):
            registry.get("egv-system-v2")
        with self.assertRaises(ValueError):
            registry.render("egv-candidate-v1", {"task_id": "only-one"})
        rendered = registry.render(
            "egv-candidate-v1",
            {"task_id": "task-x", "family_id": "PURE_FUNCTION", "attempt_index": 1, "public_locus": "module:solve"},
        )
        self.assertIn("task-x", rendered)
        self.assertNotIn("PRIVATE_KEY", rendered)
        self.assertNotIn("<SECRET>", rendered)
        with self.assertRaises(ValueError):
            registry.render(
                "egv-candidate-v1",
                {"task_id": "task-x", "family_id": "PURE_FUNCTION", "attempt_index": 1, "public_locus": "module:solve", "extra": 1},
            )

    def _valid_row(self, **overrides: object) -> dict:
        task = self.corpus.get("egv-pure_function-train-1-v1")
        task_record = task.public_manifest_record()
        values = {
            "task_id": task_record["template_id"],
            "task_family": task_record["family_id"],
            "arm": "D",
            "attempt_index": 1,
            "public_locus": task_record["public_locus"],
            "public_rule_id": task_record["public_rule_id"],
            "prompt_template_ids": list(PROMPT_IDS),
            "prompt_digest": digest_for("egv-prompt-manifest-v1"),
            "retrieved_evidence_ids": [digest_for("evidence-1")],
            "proposed_mutation_digest": digest_for("mutation-1"),
            "candidate_artifact_digest": digest_for("candidate-1"),
            "verdict_receipt_digest": digest_for("receipt-1"),
            "diagnostic_enum": "PASS",
            "resource_bucket": "UNDER_25",
            "infrastructure_incident_id": None,
            "dependency_ids": ["dependency-1"],
            "requested_authority": "EXECUTE_CANDIDATE",
            "promotion_disposition": "PROMOTED",
            "input_digest": digest_for("input-1"),
            "output_digest": digest_for("output-1"),
            "task_record": task_record,
            "task_manifest_digest": self.corpus.manifest_digest(),
        }
        values.update(overrides)
        values.pop("task_record", None)
        values.pop("task_manifest_digest", None)
        return build_sft_row(
            **values,
            corpus=self.corpus,
            task_record=task_record,
            task_manifest_digest=self.corpus.manifest_digest(),
        )

    def test_sft_row_is_closed_train_only_and_content_addressed(self) -> None:
        row = self._valid_row()
        self.assertEqual(set(row), SFT_FIELDS)
        task_record = self.corpus.get(row["task_id"]).public_manifest_record()
        self.assertEqual(
            validate_sft_row(row, corpus=self.corpus, task_record=task_record, task_manifest_digest=self.corpus.manifest_digest()),
            row,
        )
        extra = dict(row, reasoning="must not be public")
        with self.assertRaises(ValueError):
            validate_sft_row(extra, corpus=self.corpus, task_record=task_record, task_manifest_digest=self.corpus.manifest_digest())
        heldout = dict(row, split="heldout")
        heldout["row_id"] = content_id("sft", {key: value for key, value in heldout.items() if key != "row_id"})
        with self.assertRaises(ValueError):
            validate_sft_row(heldout, corpus=self.corpus, task_record=task_record, task_manifest_digest=self.corpus.manifest_digest())
        bad_digest = dict(row, output_digest="not-a-digest")
        with self.assertRaises(ValueError):
            validate_sft_row(bad_digest, corpus=self.corpus, task_record=task_record, task_manifest_digest=self.corpus.manifest_digest())
        bad_locus = dict(row, public_locus="module:another_locus")
        with self.assertRaises(ValueError):
            validate_sft_row(bad_locus, corpus=self.corpus, task_record=task_record, task_manifest_digest=self.corpus.manifest_digest())
        bad_rule = dict(row, public_rule_id="rule-public-other")
        with self.assertRaises(ValueError):
            validate_sft_row(bad_rule, corpus=self.corpus, task_record=task_record, task_manifest_digest=self.corpus.manifest_digest())
        other_record = self.corpus.get("egv-pure_function-train-2-v1").public_manifest_record()
        forged = dict(row)
        forged.update(
            {
                "task_id": other_record["template_id"],
                "public_locus": other_record["public_locus"],
                "public_rule_id": other_record["public_rule_id"],
                "task_record_digest": digest_for(other_record),
            }
        )
        forged["row_id"] = content_id("sft", {key: value for key, value in forged.items() if key != "row_id"})
        with self.assertRaises(ValueError):
            validate_sft_row(forged, corpus=self.corpus, task_record=task_record, task_manifest_digest=self.corpus.manifest_digest())

        with self.assertRaises(ValueError):
            validate_sft_row(row, task_record=task_record, task_manifest_digest=self.corpus.manifest_digest())

        internal = self._valid_row(
            diagnostic_enum="INTERNAL_ERROR",
            attempt_id="attempt-sft-internal",
            infrastructure_incident_id="incident-sft-internal",
        )
        expected_internal_root = failure_family_root(
            internal["task_family"],
            "INTERNAL_ERROR",
            internal["public_locus"],
            internal["public_rule_id"],
            infrastructure_incident_id=internal["infrastructure_incident_id"],
        )
        self.assertEqual(internal["failure_family_root"], expected_internal_root)
        for root in (
            digest_for([internal["task_family"], "INTERNAL_ERROR", internal["public_locus"], internal["public_rule_id"]]),
            digest_for("wrong-incident-root"),
        ):
            forged_root = dict(internal, failure_family_root=root)
            forged_root["row_id"] = content_id("sft", {key: value for key, value in forged_root.items() if key != "row_id"})
            with self.assertRaises(ValueError):
                validate_sft_row(
                    forged_root,
                    corpus=self.corpus,
                    task_record=task_record,
                    task_manifest_digest=self.corpus.manifest_digest(),
                )
        missing_root = dict(internal)
        missing_root.pop("failure_family_root")
        with self.assertRaises(ValueError):
            validate_sft_row(
                missing_root,
                corpus=self.corpus,
                task_record=task_record,
                task_manifest_digest=self.corpus.manifest_digest(),
            )

    def test_diagnostic_enum_and_internal_errors_never_collapse(self) -> None:
        self.assertEqual(tuple(item.value for item in Diagnostic), DIAGNOSTIC_ENUM)
        with self.assertRaises(ValueError):
            failure_family_for_attempt("PURE_FUNCTION", "MODEL_FAILURE", "module:solve", "rule-1")
        with self.assertRaises(ValueError):
            self._valid_row(diagnostic_enum="INTERNAL_ERROR")
        first = self._valid_row(
            diagnostic_enum="INTERNAL_ERROR",
            attempt_id="attempt-a",
            infrastructure_incident_id="incident-a",
        )
        second = self._valid_row(
            diagnostic_enum="INTERNAL_ERROR",
            attempt_id="attempt-b",
            attempt_index=2,
            infrastructure_incident_id="incident-b",
        )
        self.assertNotEqual(first["failure_family_root"], second["failure_family_root"])


class TestAuthorityAndSandbox(EvaluationTestCase):
    def test_production_contract_rejects_direct_worker_channel_forgery(self) -> None:
        source = (
            b"import os, sys\n"
            b"def main(value):\n"
            b"    frame = sys._getframe()\n"
            b"    while frame is not None:\n"
            b"        if 'worker_write' in frame.f_locals:\n"
            b"            frame.f_locals['send_frame'](frame.f_locals['worker_write'], b'NORMAL', b'{\\\"forged\\\":true}\\n')\n"
            b"        frame = frame.f_back\n"
            b"    os._exit = lambda code: None\n"
            b"    raise RuntimeError('forged completion')\n"
        )
        reason = validate_candidate_source_contract(source)
        self.assertIsNotNone(reason)
        self.assertIn("pure-return-v1", str(reason))

        task = self.corpus.hidden_repositories()[0]
        signer = ReceiptSigner(b"\x27" * 32)
        journal = ReceiptJournal(self.root / "contract-forgery" / "receipts.jsonl", signer.public_key)

        class ContractProbeSandbox:
            enforceable = True
            backend_name = "docker-enforced-v1"

            def execute(self, *args: object, **kwargs: object) -> SandboxResult:
                raise AssertionError("contract-rejected source must not reach candidate execution")

        class TestRuntime:
            class Status:
                enforceable = True

            status = Status()

            def require_enforceable(self) -> None:
                return None

        controller = EvaluatorController(
            sandbox=ContractProbeSandbox(),  # type: ignore[arg-type]
            hidden_runner=HiddenEvaluatorRunner.from_corpus(self.corpus),
            broker=AuthorityBroker(TestRuntime(), AuthorityPolicy.candidate_execution()),  # type: ignore[arg-type]
            signer=signer,
            journal=journal,
            ingest=lambda receipt: receipt,
            campaign_id="campaign-contract-forgery",
            protocol_digest=digest_for("protocol"),
            policy_digest=AuthorityPolicy.candidate_execution().digest,
        )
        result = controller.evaluate(
            candidate_id="candidate-contract-forgery",
            task_id=task.template_id,
            source=source,
            opaque_input=task.evaluator_input,
            declared_locus=task.public_locus,
        )
        self.assertEqual(result.diagnostic_enum, Diagnostic.PROTOCOL_VIOLATION.value)
        self.assertEqual(result.disposition, "REJECTED")
        self.assertFalse(result.infrastructure_loss)

    def test_production_controller_rejects_ledger_writer_cohosting(self) -> None:
        ledger = EvidenceLedger(self.root / "cohost.sqlite", blob_root=self.root / "cohost-blobs")
        try:
            signer = ReceiptSigner(b"\x16" * 32)
            journal = ReceiptJournal(self.root / "cohost" / "receipts.jsonl", signer.public_key)

            class TestOnlyEnforceableStub:
                enforceable = True

            with self.assertRaises(AuthorityDenied):
                EvaluatorController(
                    sandbox=TestOnlyEnforceableStub(),  # type: ignore[arg-type]
                    hidden_runner=HiddenEvaluatorRunner.from_corpus(self.corpus),
                    broker=AuthorityBroker(None, AuthorityPolicy.candidate_execution()),
                    signer=signer,
                    journal=journal,
                    ingest=lambda receipt: receipt,
                    campaign_id="campaign-cohost",
                    protocol_digest=digest_for("protocol"),
                    policy_digest=AuthorityPolicy.candidate_execution().digest,
                )
        finally:
            ledger.close()

    def test_authority_is_deny_by_default_and_network_is_absent(self) -> None:
        default = AuthorityBroker(None)
        with self.assertRaises(AuthorityDenied):
            default.authorize(
                action="execute_candidate",
                candidate_id="candidate-1",
                requested_authority="EXECUTE_CANDIDATE",
                artifact_digest=digest_for("candidate"),
            )
        policy = AuthorityPolicy.candidate_execution()
        self.assertFalse(policy.process_creation)
        self.assertFalse(policy.network)
        decision = AuthorityBroker(DockerEnforcedRuntime(), policy).authorize(
            action="execute_candidate",
            candidate_id="candidate-1",
            requested_authority="EXECUTE_CANDIDATE",
            artifact_digest=digest_for("candidate"),
        )
        self.assertTrue(decision.allowed)
        with self.assertRaises(AuthorityDenied):
            AuthorityBroker(LocalEnforcedRuntime(), policy).authorize(
                action="execute_candidate",
                candidate_id="candidate-1",
                requested_authority="EXECUTE_CANDIDATE",
                artifact_digest=digest_for("candidate"),
            )

    def test_openshell_never_claims_an_unimplemented_boundary(self) -> None:
        status = OpenShellRuntime().status
        self.assertFalse(status.enforceable)
        self.assertFalse(status.claimed)
        with self.assertRaises(AuthorityDenied):
            AuthorityBroker(OpenShellRuntime(), AuthorityPolicy.candidate_execution()).authorize(
                action="execute_candidate",
                candidate_id="candidate-1",
                requested_authority="EXECUTE_CANDIDATE",
                artifact_digest=digest_for("candidate"),
            )

    def test_sandbox_maps_model_failures_and_denies_capabilities(self) -> None:
        sandbox = LocalTestSandbox(self.root / "sandbox")
        source = b"def main(value):\n    return value\n"
        result = sandbox.execute(source, {"x": 1}, artifact_digest=digest_for(source), candidate_id="candidate-pass")
        self.assertEqual(result.diagnostic_enum, "PASS")
        syntax = sandbox.execute(b"def main(:\n    pass\n", {}, artifact_digest=digest_for(b"def main(:\n    pass\n"), candidate_id="candidate-syntax")
        self.assertEqual(syntax.diagnostic_enum, "SYNTAX_OR_IMPORT")
        runtime_source = b"def main(value):\n    raise RuntimeError('model failure')\n"
        runtime = sandbox.execute(runtime_source, {}, artifact_digest=digest_for(runtime_source), candidate_id="candidate-runtime")
        self.assertEqual(runtime.diagnostic_enum, "RUNTIME_EXCEPTION")
        controls = sandbox.negative_controls(self.root / "evaluator-private" / "hidden.json")
        self.assertEqual(set(controls), {"hidden_read", "hidden_list", "process", "network", "credentials", "evaluator_mutation"})
        self.assertTrue(all(controls.values()))

    @unittest.skipUnless(shutil.which("docker"), "Docker is required for the production isolation regression")
    def test_docker_production_controls_cover_absence_mutation_escape_and_identity(self) -> None:
        if not DockerEnforcedRuntime().status.enforceable:
            self.skipTest("configured pinned Docker image is unavailable")
        sandbox = DockerCandidateSandbox(self.root / "docker-sandbox")
        source = b"def main(value):\n    return {'value': value}\n"
        candidate_path = self.root / "candidate" / "src" / "task.py"
        candidate_path.parent.mkdir(parents=True)
        candidate_path.write_bytes(source)
        result = sandbox.execute(
            source,
            {"opaque": True},
            artifact_digest=digest_bytes(source),
            candidate_id="candidate-docker-regression",
            source_path=candidate_path,
        )
        self.assertEqual(result.diagnostic_enum, "PASS")
        self.assertEqual(result.output_bytes, b'{"value":{"opaque":true}}\n')
        self.assertIn("before", result.environment_diff)
        self.assertIn("after", result.environment_diff)
        self.assertEqual(result.environment_diff["after"]["source_digest_before"], digest_bytes(source))
        self.assertEqual(result.environment_diff["after"]["source_digest_after"], digest_bytes(source))
        self.assertEqual(result.environment_diff["after"]["mutable_mounts"], [])
        self.assertEqual(result.environment_diff["after"]["unexpected_changed_paths"], [])
        self.assertTrue(result.environment_diff["after"]["verified_unchanged_state"])
        controls = sandbox.negative_controls(self.root / "never-mounted-hidden.json")
        expected = {
            "hidden_read", "hidden_list", "hidden_walk", "hidden_scandir", "hidden_stat", "hidden_chdir", "hidden_readlink",
            "hidden_rename", "hidden_link", "hidden_unlink", "hidden_truncate", "hidden_chmod",
            "source_unlink", "source_truncate", "source_chmod", "source_link", "process", "network",
            "same_uid_kill", "mount", "umount", "ctypes_open", "ctypes_openat", "ctypes_socket", "ctypes_system", "execve", "io_uring",
            "userfaultfd", "bpf",
            "memfd_create", "memfd_secret",
            "mountinfo_read", "proc_environ_read", "credentials_file", "credentials_mount", "credentials_proc", "credentials_env",
            "host_path_read", "host_socket_absent", "pid_namespace", "uid_gid", "source_read_only", "hidden_inode_absent",
            "host_path_absent", "credentials_absent",
        }
        self.assertEqual(set(controls), expected)
        self.assertTrue(all(controls.values()))
        for probe_name in ("source_unlink", "source_truncate", "source_chmod", "source_link"):
            self.assertIn(sandbox.last_negative_control_evidence[probe_name]["errno"], (1, 30))
        for probe_name in ("ctypes_open", "ctypes_openat"):
            self.assertEqual(sandbox.last_negative_control_evidence[probe_name]["probe_status"], "OBSERVED")
            self.assertEqual(sandbox.last_negative_control_evidence[probe_name]["return_value"], -1)
            self.assertIn(sandbox.last_negative_control_evidence[probe_name]["errno"], (1, 30))
        self.assertEqual(sandbox.last_negative_control_evidence["ctypes_system"]["probe_status"], "OBSERVED")
        self.assertNotIn("kernel/seccomp denial", str(sandbox.last_negative_control_evidence["ctypes_system"]))
        sentinel = sandbox._probe_source("None")
        sentinel_result = sandbox.execute(
            sentinel,
            {},
            artifact_digest=digest_bytes(sentinel),
            candidate_id="negative-successful-none",
            candidate_contract="untrusted-adversarial-v1",
        )
        sentinel_payload = json.loads(sentinel_result.output_bytes.decode("utf-8"))
        self.assertEqual(sentinel_payload["probe_status"], "ALLOWED")
        self.assertNotEqual(sentinel_payload["probe_status"], "DENIED")
        report = sandbox.runtime_report()
        self.assertTrue(report["enforceable"])
        create_args = sandbox._create_args("egv-test-container", "egv-test-volume")
        self.assertIn(sandbox.image_id, create_args)
        self.assertNotIn("type=bind", create_args)
        self.assertEqual(report["network"], "none")
        self.assertTrue(report["read_only_root"])
        self.assertEqual(report["cap_drop"], "ALL")
        self.assertEqual(report["user"], "65534:65534")
        self.assertTrue(report["python_isolated"])
        self.assertFalse(report["host_paths_mounted"])
        self.assertTrue(all(":rw" not in argument for argument in create_args))
        self.assertIn("--read-only", create_args)
        self.assertIn("--cap-drop", create_args)

        self.assertIn("if syscall_number < 0", sandbox_module._DOCKER_BOOTSTRAP)
        self.assertIn("unresolved candidate syscall", sandbox_module._DOCKER_BOOTSTRAP)
        profile_path = sandbox.seccomp_path
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        self.assertEqual(profile["defaultAction"], "SCMP_ACT_ERRNO")
        profile_names = set(profile["syscalls"][0]["names"])
        self.assertNotIn("userfaultfd", profile_names)
        self.assertNotIn("bpf", profile_names)

        unresolved = sandbox_module._DOCKER_BOOTSTRAP.replace(
            '"read", "write",', '"egv_unresolved_syscall", "read", "write",', 1
        )
        with patch.object(sandbox_module, "_DOCKER_BOOTSTRAP", unresolved):
            unresolved_source = b"def main(value):\n    return value\n"
            unresolved_result = sandbox.execute(
                unresolved_source,
                {},
                artifact_digest=digest_bytes(unresolved_source),
                candidate_id="candidate-unresolved-seccomp",
                candidate_contract="untrusted-adversarial-v1",
            )
        self.assertEqual(unresolved_result.diagnostic_enum, "INTERNAL_ERROR")
        self.assertEqual(unresolved_result.exit_status_class, "INFRASTRUCTURE_LOSS")
        self.assertTrue(unresolved_result.incident_id)
        self.assertIn("unresolved candidate syscall", unresolved_result.environment_diff.get("after", {}).get("runner_status", ""))
        self.assertIn("EGV_SANDBOX_RUNNER_SENTINEL=", unresolved_result.environment_diff.get("after", {}).get("runner_status", ""))

        remaining_containers = subprocess.run(
            ["docker", "ps", "-a", "--filter", "name=egv-evaluation-", "--format", "{{.Names}}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        remaining_volumes = subprocess.run(
            ["docker", "volume", "ls", "--filter", "name=egv-evaluation-", "--format", "{{.Name}}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(remaining_containers.returncode, 0, remaining_containers.stderr)
        self.assertEqual(remaining_volumes.returncode, 0, remaining_volumes.stderr)
        self.assertEqual(remaining_containers.stdout.strip(), "")
        self.assertEqual(remaining_volumes.stdout.strip(), "")

    @unittest.skipUnless(shutil.which("docker"), "Docker is required for the clean candidate-stage regression")
    def test_docker_candidate_has_no_supervisor_frames_or_status_authority(self) -> None:
        if not DockerEnforcedRuntime().status.enforceable:
            self.skipTest("configured pinned Docker image is unavailable")
        sandbox = DockerCandidateSandbox(self.root / "docker-second-stage")
        channel_forge = (
            b"import sys\n"
            b"def main(value):\n"
            b"    frame = sys._getframe()\n"
            b"    while frame is not None:\n"
            b"        if 'worker_write' in frame.f_locals:\n"
            b"            frame.f_locals['send_frame'](frame.f_locals['worker_write'], b'NORMAL', b'{\\\"forged\\\":true}\\n')\n"
            b"        frame = frame.f_back\n"
            b"    raise RuntimeError('forged completion')\n"
        )
        rejected = sandbox.execute(
            channel_forge,
            {},
            artifact_digest=digest_bytes(channel_forge),
            candidate_id="candidate-default-contract-channel-forge",
        )
        self.assertEqual(rejected.diagnostic_enum, Diagnostic.PROTOCOL_VIOLATION.value)
        self.assertEqual(rejected.environment_diff["contract_status"], "REJECTED")
        source = (
            b"import os, sys\n"
            b"def main(value):\n"
            b"    frame = sys._getframe()\n"
            b"    names = set()\n"
            b"    module_names = set()\n"
            b"    while frame is not None:\n"
            b"        names.update(frame.f_globals)\n"
            b"        names.update(frame.f_locals)\n"
            b"        frame = frame.f_back\n"
            b"    module = __import__('__main__')\n"
            b"    module_names.update(module.__dict__)\n"
            b"    try:\n"
            b"        os.listdir('/proc/self/fd')\n"
            b"        fd_scan = 'VISIBLE'\n"
            b"    except PermissionError:\n"
            b"        fd_scan = 'DENIED'\n"
            b"    except OSError:\n"
            b"        fd_scan = 'ABSENT_OR_DENIED'\n"
            b"    forbidden = {'_status', 'runner_auth', '_runner_failure', '_BOOTSTRAP', 'hmac', 'hashlib'}\n"
            b"    return {'frame_forbidden': sorted(names & forbidden), 'module_forbidden': sorted(module_names & forbidden), 'fd_scan': fd_scan}\n"
        )
        result = sandbox.execute(
            source,
            {"opaque": "only"},
            artifact_digest=digest_bytes(source),
            candidate_id="candidate-second-stage-scan",
            candidate_contract="untrusted-adversarial-v1",
        )
        self.assertEqual(result.diagnostic_enum, Diagnostic.PROTOCOL_VIOLATION.value)
        self.assertTrue(result.evidence_only)
        self.assertEqual(result.exit_status_class, "NONZERO")
        payload = json.loads(result.output_bytes.decode("utf-8"))
        self.assertEqual(payload["frame_forbidden"], [])
        self.assertEqual(payload["module_forbidden"], [])
        self.assertIn(payload["fd_scan"], {"DENIED", "ABSENT_OR_DENIED"})

        override = (
            b"import os, sys\n"
            b"def main(value):\n"
            b"    os._exit = lambda code: None\n"
            b"    sys.stdout.write('{\"forged\":true}\\n')\n"
            b"    frame = sys._getframe()\n"
            b"    while frame is not None:\n"
            b"        if '_status' in frame.f_globals:\n"
            b"            frame.f_globals['_status'] = lambda *args: None\n"
            b"        frame = frame.f_back\n"
            b"    raise RuntimeError('candidate status override probe')\n"
        )
        failed = sandbox.execute(
            override,
            {},
            artifact_digest=digest_bytes(override),
            candidate_id="candidate-second-stage-status-override",
            candidate_contract="untrusted-adversarial-v1",
        )
        self.assertEqual(failed.diagnostic_enum, Diagnostic.RUNTIME_EXCEPTION.value)
        self.assertNotEqual(failed.exit_status, "SUCCESS")

        frame_forge = sandbox.execute(
            channel_forge,
            {},
            artifact_digest=digest_bytes(channel_forge),
            candidate_id="candidate-adversarial-frame-forge",
            candidate_contract="untrusted-adversarial-v1",
        )
        self.assertNotEqual(frame_forge.diagnostic_enum, Diagnostic.PASS.value)
        self.assertFalse(frame_forge.exit_status == "SUCCESS" and not frame_forge.evidence_only)

        gc_probe = (
            b"import gc\n"
            b"def main(value):\n"
            b"    found = []\n"
            b"    for item in gc.get_objects():\n"
            b"        try:\n"
            b"            names = set(item.f_locals)\n"
            b"        except Exception:\n"
            b"            names = set()\n"
            b"        if 'worker_write' in names or 'runner_auth' in names:\n"
            b"            found.append('trusted-state')\n"
            b"    return {'trusted_state_found': found}\n"
        )
        gc_result = sandbox.execute(
            gc_probe,
            {},
            artifact_digest=digest_bytes(gc_probe),
            candidate_id="candidate-adversarial-gc-forge",
            candidate_contract="untrusted-adversarial-v1",
        )
        self.assertNotEqual(gc_result.diagnostic_enum, Diagnostic.PASS.value)
        if gc_result.output_bytes:
            self.assertNotIn("trusted-state", gc_result.output_bytes.decode("utf-8", errors="replace"))

        fd_probe = (
            b"import os\n"
            b"def main(value):\n"
            b"    for fd in range(3, 64):\n"
            b"        try:\n"
            b"            os.fstat(fd)\n"
            b"            os.write(fd, b'EGV-FD-FORGE\\n')\n"
            b"        except OSError:\n"
            b"            pass\n"
            b"    return {'fd_probe': 'complete'}\n"
        )
        fd_result = sandbox.execute(
            fd_probe,
            {},
            artifact_digest=digest_bytes(fd_probe),
            candidate_id="candidate-adversarial-fd-forge",
            candidate_contract="untrusted-adversarial-v1",
        )
        self.assertNotEqual(fd_result.diagnostic_enum, Diagnostic.PASS.value)

        patched_exit = (
            b"import os, sys\n"
            b"def main(value):\n"
            b"    os._exit = lambda code: None\n"
            b"    sys.exit = lambda code=0: None\n"
            b"    raise RuntimeError('patched exit probe')\n"
        )
        patched_result = sandbox.execute(
            patched_exit,
            {},
            artifact_digest=digest_bytes(patched_exit),
            candidate_id="candidate-adversarial-patched-exit",
            candidate_contract="untrusted-adversarial-v1",
        )
        # These four live adversarial probes are evidence-only claims. None
        # may be promoted by a Docker stdout/status/channel forgery, even
        # when the low-level adapter returns a normal JSON payload.
        for probe_name, probe_result in (
            ("frame", frame_forge),
            ("gc", gc_result),
            ("fd", fd_result),
            ("patched-exit", patched_result),
        ):
            with self.subTest(probe=probe_name):
                self.assertNotEqual(probe_result.diagnostic_enum, Diagnostic.PASS.value)
                self.assertFalse(probe_result.environment_diff.get("decision_eligible", False))
        self.assertNotEqual(patched_result.diagnostic_enum, Diagnostic.PASS.value)

    @unittest.skipUnless(shutil.which("docker"), "Docker is required for candidate exit-status regressions")
    def test_docker_candidate_frame_walk_cannot_forge_runner_sentinel(self) -> None:
        if not DockerEnforcedRuntime().status.enforceable:
            self.skipTest("configured pinned Docker image is unavailable")
        sandbox = DockerCandidateSandbox(self.root / "docker-second-stage-sentinel")
        source = (
            b"import os, sys\n"
            b"def main(value):\n"
            b"    sys.stderr.write('EGV_SANDBOX_RUNNER_SENTINEL=RUNNER_FILTER_SETUP_FAILED:forged:00\\n')\n"
            b"    os._exit(44)\n"
        )
        result = sandbox.execute(
            source,
            {},
            artifact_digest=digest_bytes(source),
            candidate_id="candidate-second-stage-forged-sentinel",
            candidate_contract="untrusted-adversarial-v1",
        )
        self.assertEqual(result.diagnostic_enum, Diagnostic.RUNTIME_EXCEPTION.value)
        self.assertEqual(result.exit_status_class, "NONZERO")
        self.assertIsNone(result.incident_id)

    @unittest.skipUnless(shutil.which("docker"), "Docker is required for the real resource ceiling regression")
    def test_docker_resource_bound_is_real_memory_ceiling(self) -> None:
        if not DockerEnforcedRuntime().status.enforceable:
            self.skipTest("configured pinned Docker image is unavailable")
        sandbox = DockerCandidateSandbox(self.root / "docker-resource")
        source = b"def main(value):\n    blocks = [b'x' * 1048576 for _ in range(128)]\n    return len(blocks)\n"
        result = sandbox.execute(
            source,
            {},
            artifact_digest=digest_bytes(source),
            candidate_id="candidate-resource-ceiling",
            memory_limit=RESOURCE_BOUND_MEMORY_LIMIT,
            candidate_contract="untrusted-adversarial-v1",
        )
        self.assertEqual(result.diagnostic_enum, "RESOURCE_LIMIT")
        self.assertEqual(result.exit_status_class, "RESOURCE_LIMIT")
        self.assertEqual(result.resource_bucket, "LIMIT_REACHED")
        self.assertTrue(result.environment_diff["after"]["oom_killed"])
        self.assertEqual(result.environment_diff["before"]["memory_limit"], RESOURCE_BOUND_MEMORY_LIMIT)

    @unittest.skipUnless(shutil.which("docker"), "Docker is required for the output ceiling regression")
    def test_docker_output_bound_has_distinct_closed_status(self) -> None:
        if not DockerEnforcedRuntime().status.enforceable:
            self.skipTest("configured pinned Docker image is unavailable")
        sandbox = DockerCandidateSandbox(self.root / "docker-output")
        source = b"def main(value):\n    return 'x' * 70000\n"
        result = sandbox.execute(
            source,
            {},
            artifact_digest=digest_bytes(source),
            candidate_id="candidate-output-ceiling",
            candidate_contract="untrusted-adversarial-v1",
        )
        self.assertEqual(result.diagnostic_enum, "RESOURCE_LIMIT")
        self.assertEqual(result.resource_bucket, "OUTPUT_LIMIT")
        self.assertEqual(result.exit_status, "OUTPUT_LIMIT")
        self.assertEqual(result.exit_status_class, "OUTPUT_LIMIT")

    def test_docker_threshold_overrides_fail_closed(self) -> None:
        with self.assertRaises(DockerConfigurationError):
            DockerCandidateSandbox(self.root / "bad-timeout", timeout_seconds=2.1)
        with self.assertRaises(DockerConfigurationError):
            DockerCandidateSandbox(self.root / "bad-output", output_limit=65535)

    @unittest.skipUnless(shutil.which("docker"), "Docker is required for the forced-failure cleanup regression")
    def test_docker_forced_diff_failure_cleans_ephemeral_resources(self) -> None:
        if not DockerEnforcedRuntime().status.enforceable:
            self.skipTest("configured pinned Docker image is unavailable")
        sandbox = DockerCandidateSandbox(self.root / "docker-forced-failure")
        source = b"def main(value):\n    return value\n"
        artifact_digest = digest_bytes(source)
        with patch.object(sandbox, "_environment_diff", side_effect=InfrastructureFailure("forced diff failure")):
            result = sandbox.execute(
                source,
                {"forced": True},
                artifact_digest=artifact_digest,
                candidate_id="candidate-forced-diff-failure",
            )
        self.assertEqual(result.diagnostic_enum, "INTERNAL_ERROR")
        self.assertTrue(result.incident_id)
        sandbox_id = result.sandbox_id
        container_name = "egv-evaluation-{}".format(sandbox_id[-24:])
        volume_name = "egv-evaluation-source-{}-{}".format(artifact_digest[:16], sandbox_id[-8:])
        remaining_container = subprocess.run(
            ["docker", "ps", "-a", "--filter", "name=^/{}$".format(container_name), "--format", "{{.Names}}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        remaining_volume = subprocess.run(
            ["docker", "volume", "ls", "--filter", "name=^{}$".format(volume_name), "--format", "{{.Name}}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(remaining_container.returncode, 0, remaining_container.stderr)
        self.assertEqual(remaining_volume.returncode, 0, remaining_volume.stderr)
        self.assertEqual(remaining_container.stdout.strip(), "")
        self.assertEqual(remaining_volume.stdout.strip(), "")

    @unittest.skipUnless(shutil.which("docker"), "Docker is required for the production isolation regression")
    def test_candidate_permission_error_is_runtime_failure_not_authority_denial(self) -> None:
        if not DockerEnforcedRuntime().status.enforceable:
            self.skipTest("configured pinned Docker image is unavailable")
        sandbox = DockerCandidateSandbox(self.root / "docker-permission")
        source = b"def main(value):\n    raise PermissionError('candidate choice')\n"
        result = sandbox.execute(
            source,
            {},
            artifact_digest=digest_bytes(source),
            candidate_id="candidate-permission-error",
            candidate_contract="untrusted-adversarial-v1",
        )
        self.assertEqual(result.diagnostic_enum, "RUNTIME_EXCEPTION")
        self.assertNotEqual(result.exit_status, "DENIED")

    @unittest.skipUnless(shutil.which("docker"), "Docker is required for candidate exit-status regressions")
    def test_candidate_controlled_exit_statuses_never_claim_infrastructure_loss(self) -> None:
        if not DockerEnforcedRuntime().status.enforceable:
            self.skipTest("configured pinned Docker image is unavailable")
        sandbox = DockerCandidateSandbox(self.root / "docker-exit-status")
        syscall_numbers = {
            "x86_64": (60, 231),
            "amd64": (60, 231),
            "aarch64": (93, 94),
            "arm64": (93, 94),
            "riscv64": (93, 94),
            "armv7l": (1, 248),
            "i386": (1, 252),
            "i686": (1, 252),
        }
        if os.uname().machine not in syscall_numbers:
            self.fail("unsupported host architecture for raw exit regression: {}".format(os.uname().machine))
        exit_number, exit_group_number = syscall_numbers[os.uname().machine]
        sources = {
            "os-exit-0": b"import os\ndef main(value):\n    os._exit(0)\n",
            "os-exit-1": b"import os\ndef main(value):\n    os._exit(1)\n",
            "os-exit-41": b"import os\ndef main(value):\n    os._exit(41)\n",
            "os-exit-42": b"import os\ndef main(value):\n    os._exit(42)\n",
            "os-exit-44": b"import os\ndef main(value):\n    os._exit(44)\n",
            "system-exit-44": b"def main(value):\n    raise SystemExit(44)\n",
            "forged-runner-sentinel": (
                b"import os, sys\n"
                b"def main(value):\n"
                b"    sys.stderr.write('EGV_SANDBOX_RUNNER_SENTINEL=RUNNER_FILTER_SETUP_FAILED:forged:00\\n')\n"
                b"    os._exit(44)\n"
            ),
        }
        for candidate_id, source in sources.items():
            result = sandbox.execute(
                source,
                {},
                artifact_digest=digest_bytes(source),
                candidate_id="candidate-" + candidate_id,
                candidate_contract="untrusted-adversarial-v1",
            )
            self.assertEqual(result.diagnostic_enum, "RUNTIME_EXCEPTION", candidate_id)
            self.assertEqual(result.exit_status_class, "NONZERO", candidate_id)
            self.assertFalse(result.incident_id, candidate_id)

        for candidate_id, syscall_number, syscall_name in (
            ("raw-exit", exit_number, "exit"),
            ("raw-exit-group", exit_group_number, "exit_group"),
        ):
            source = (
                "import ctypes\n"
                "def main(value):\n"
                "    libc = ctypes.CDLL(None, use_errno=True)\n"
                "    result = libc.syscall({}, 0)\n"
                "    return {{'syscall': {}, 'name': {!r}, 'return_value': result, 'errno': ctypes.get_errno()}}\n"
            ).format(syscall_number, syscall_number, syscall_name).encode("utf-8")
            result = sandbox.execute(
                source,
                {},
                artifact_digest=digest_bytes(source),
                candidate_id="candidate-" + candidate_id,
                candidate_contract="untrusted-adversarial-v1",
            )
            self.assertEqual(result.diagnostic_enum, Diagnostic.PROTOCOL_VIOLATION.value, candidate_id)
            self.assertTrue(result.evidence_only, candidate_id)
            observed = json.loads(result.output_bytes.decode("utf-8"))
            self.assertEqual(observed["syscall"], syscall_number, candidate_id)
            self.assertEqual(observed["name"], syscall_name, candidate_id)
            self.assertEqual(observed["return_value"], -1, candidate_id)
            self.assertIn(observed["errno"], (1, 30), candidate_id)

    def test_docker_configuration_is_pinned_and_fails_closed_without_matching_image(self) -> None:
        bad = DockerSandboxConfig(pinned_image_id="sha256:" + "0" * 64)
        with self.assertRaises(Exception):
            bad.verify_image()

    def test_hidden_runner_does_not_expose_expected_output_and_checks_input_binding(self) -> None:
        runner = HiddenEvaluatorRunner.from_corpus(self.corpus)
        repo = self.corpus.hidden_repositories()[0]
        expected_source = self.corpus.correct_candidate_source(repo.template_id)
        sandbox = LocalTestSandbox(self.root / "runner-sandbox")
        result = sandbox.execute(expected_source, repo.evaluator_input, artifact_digest=digest_for(expected_source), candidate_id="candidate-hidden")
        verdict = runner.evaluate(repo.template_id, result.output_bytes, opaque_input=repo.evaluator_input)
        self.assertEqual(verdict.diagnostic_enum, "PASS")
        self.assertFalse(hasattr(verdict, "expected_output"))
        mismatch = runner.evaluate(repo.template_id, result.output_bytes, opaque_input={"wrong": True})
        self.assertEqual(mismatch.diagnostic_enum, "PROTOCOL_VIOLATION")
        unknown = runner.evaluate("heldout-unknown", b"{}\n")
        self.assertEqual(unknown.diagnostic_enum, "PROTOCOL_VIOLATION")
        self.assertFalse(unknown.infrastructure_loss)

    def test_unknown_controller_task_is_protocol_violation_with_no_infrastructure_claim(self) -> None:
        signer = ReceiptSigner(b"\x18" * 32)
        journal = ReceiptJournal(self.root / "unknown" / "receipts.jsonl", signer.public_key)

        class TestOnlyEnforceableStub:
            enforceable = True

        controller = EvaluatorController(
            sandbox=TestOnlyEnforceableStub(),  # type: ignore[arg-type]
            hidden_runner=HiddenEvaluatorRunner.from_corpus(self.corpus),
            broker=AuthorityBroker(None, AuthorityPolicy.candidate_execution()),
            signer=signer,
            journal=journal,
            ingest=lambda receipt: receipt,
            campaign_id="campaign-unknown",
            protocol_digest=digest_for("protocol"),
            policy_digest=AuthorityPolicy.candidate_execution().digest,
        )
        result = controller.evaluate(
            candidate_id="candidate-unknown",
            task_id="egv-unknown-heldout-1-v1",
            source=b"def main(value):\n    return value\n",
            opaque_input={},
            declared_locus="module:unknown",
        )
        self.assertEqual(result.diagnostic_enum, "PROTOCOL_VIOLATION")
        self.assertFalse(result.infrastructure_loss)
        self.assertEqual(len(journal.receipts()), 1)

    def test_authority_infrastructure_loss_has_incident_and_failure_root(self) -> None:
        signer = ReceiptSigner(b"\x17" * 32)
        journal = ReceiptJournal(self.root / "authority-loss" / "receipts.jsonl", signer.public_key)

        class TestOnlyEnforceableStub:
            enforceable = True

        task = self.corpus.hidden_repositories()[0]
        controller = EvaluatorController(
            sandbox=TestOnlyEnforceableStub(),  # type: ignore[arg-type]
            hidden_runner=HiddenEvaluatorRunner.from_corpus(self.corpus),
            broker=AuthorityBroker(None, AuthorityPolicy.candidate_execution()),
            signer=signer,
            journal=journal,
            ingest=lambda receipt: receipt,
            campaign_id="campaign-authority-loss",
            protocol_digest=digest_for("protocol"),
            policy_digest=AuthorityPolicy.candidate_execution().digest,
        )
        result = controller.evaluate(
            candidate_id="candidate-authority-loss",
            task_id=task.template_id,
            source=self.corpus.correct_candidate_source(task.template_id),
            opaque_input=task.evaluator_input,
            declared_locus=task.public_locus,
        )
        self.assertEqual(result.diagnostic_enum, "INTERNAL_ERROR")
        self.assertTrue(result.infrastructure_loss)
        self.assertTrue(result.infrastructure_incident_id)
        self.assertTrue(result.failure_family_root)
        self.assertEqual(journal.receipts()[0]["infrastructure_incident_id"], result.infrastructure_incident_id)
        authority_receipt = journal.receipts()[0]
        self.assertEqual(authority_receipt["task_family"], task.family_id)
        self.assertEqual(authority_receipt["normalized_public_locus"], task.public_locus)
        self.assertEqual(authority_receipt["public_rule_id"], task.public_rule_id)
        self.assertEqual(
            authority_receipt["failure_family_root"],
            failure_family_root(
                authority_receipt["task_family"],
                authority_receipt["diagnostic_enum"],
                authority_receipt["normalized_public_locus"],
                authority_receipt["public_rule_id"],
                infrastructure_incident_id=authority_receipt["infrastructure_incident_id"],
            ),
        )
        self.assertEqual(
            result.failure_family_root,
            failure_family_root(
                task.family_id,
                "INTERNAL_ERROR",
                task.public_locus,
                task.public_rule_id,
                infrastructure_incident_id=result.infrastructure_incident_id,
            ),
        )

    def test_sandbox_infrastructure_loss_signs_and_verifies_complete_receipt_chain(self) -> None:
        signer = ReceiptSigner(b"\x21" * 32)
        journal = ReceiptJournal(self.root / "sandbox-loss" / "receipts.jsonl", signer.public_key)
        task = self.corpus.hidden_repositories()[0]
        incident = "incident-sandbox-chain"

        class TestRuntime:
            class Status:
                enforceable = True

            status = Status()

            def require_enforceable(self) -> None:
                return None

        class InfrastructureSandbox:
            enforceable = True

            def execute(self, source: bytes, opaque_input: object, **kwargs: object) -> SandboxResult:
                return SandboxResult(
                    "sandbox-infrastructure-chain",
                    str(kwargs["artifact_digest"]),
                    Diagnostic.INTERNAL_ERROR.value,
                    "UNDER_25",
                    b"",
                    "INFRASTRUCTURE_LOSS",
                    1,
                    {"runner": "authenticated-filter-failure"},
                    incident,
                )

        controller = EvaluatorController(
            sandbox=InfrastructureSandbox(),  # type: ignore[arg-type]
            hidden_runner=HiddenEvaluatorRunner.from_corpus(self.corpus),
            broker=AuthorityBroker(TestRuntime(), AuthorityPolicy.candidate_execution()),  # type: ignore[arg-type]
            signer=signer,
            journal=journal,
            ingest=lambda receipt: receipt,
            campaign_id="campaign-sandbox-loss",
            protocol_digest=digest_for("protocol"),
            policy_digest=AuthorityPolicy.candidate_execution().digest,
        )
        result = controller.evaluate(
            candidate_id="candidate-sandbox-loss",
            task_id=task.template_id,
            source=self.corpus.correct_candidate_source(task.template_id),
            opaque_input=task.evaluator_input,
            declared_locus=task.public_locus,
        )
        self.assertEqual(result.diagnostic_enum, Diagnostic.INTERNAL_ERROR.value)
        self.assertTrue(result.infrastructure_loss)
        self.assertEqual(result.infrastructure_incident_id, incident)
        records = journal.receipts()
        self.assertEqual([record["receipt_type"] for record in records], ["AUTHORITY", "VERDICT", "EFFECT"])
        self.assertEqual(journal.verify()["count"], 3)
        for record in records:
            verify_receipt(record, signer.public_key)
            if "infrastructure_incident_id" in record:
                self.assertEqual(record["diagnostic_enum"], Diagnostic.INTERNAL_ERROR.value)
                self.assertEqual(record["infrastructure_incident_id"], incident)
                self.assertEqual(
                    record["failure_family_root"],
                    failure_family_root(
                        task.family_id,
                        Diagnostic.INTERNAL_ERROR.value,
                        task.public_locus,
                        task.public_rule_id,
                        infrastructure_incident_id=incident,
                    ),
                )


class TestEvaluationLedgerIntegration(EvaluationTestCase):
    def test_signed_controller_ingest_materialization_and_key_pin(self) -> None:
        # The production controller now runs only in the spawned evaluator
        # process.  This test deliberately exercises that receipt-only path;
        # co-hosting a controller with the trainer writer is not a valid unit
        # test shortcut.
        report = run_evaluation_two_process_smoke()
        self.assertEqual(report["heldout_task_count"], 8)
        self.assertTrue(all(value == "PROMOTED" for value in report["heldout_dispositions"].values()))
        self.assertTrue(report["evaluator_key_ephemeral"])
        self.assertEqual(report["evaluator_ipc_methods"], ["ingest_receipt"])
        self.assertTrue(report["evaluator_ledger_cohosting_evidence"]["denied"])
        self.assertTrue(report["post_integrity"]["chain_valid"])

    def test_production_receipts_rebuild_without_non_internal_root_claims(self) -> None:
        task = self.corpus.hidden_repositories()[0]
        source = b"def main(value):\n    return value\n"
        ledger_path = self.root / "production-receipt-chain.sqlite"
        blob_root = self.root / "production-receipt-chain-blobs"
        signer = ReceiptSigner(bytes([0x2D]) * 32)
        journal = ReceiptJournal(self.root / "production-receipt-chain" / "receipts.jsonl", signer.public_key)

        def ingest(receipt: dict) -> dict:
            # Open the actual writer only for the narrow ingest call so the
            # controller itself remains outside the SQLite writer process.
            with EvidenceLedger(ledger_path, blob_root=blob_root, clock=lambda: "2026-08-22T00:00:00Z") as ledger:
                return ledger.ingest_receipt(receipt, signer.public_key)

        class TestRuntime:
            class Status:
                enforceable = True

            status = Status()

            def require_enforceable(self) -> None:
                return None

        class EvidenceSandbox:
            enforceable = True
            backend_name = "docker-enforced-v1"

            def __init__(self, outputs: list[bytes]) -> None:
                self.outputs = list(outputs)

            def execute(self, source_bytes: bytes, opaque_input: object, **kwargs: object) -> SandboxResult:
                del source_bytes, opaque_input
                return SandboxResult(
                    "sandbox-production-receipt-chain",
                    str(kwargs["artifact_digest"]),
                    Diagnostic.PASS.value,
                    "UNDER_25",
                    self.outputs.pop(0),
                    "SUCCESS",
                    1,
                    {"docker_result": "untrusted-evidence"},
                )

        expected_output = canonical_bytes(task.expected_output) + b"\n"
        sandbox = EvidenceSandbox([expected_output, b'{"wrong":true}\n'])
        controller = EvaluatorController(
            sandbox=sandbox,  # type: ignore[arg-type]
            hidden_runner=HiddenEvaluatorRunner.from_corpus(self.corpus),
            broker=AuthorityBroker(TestRuntime(), AuthorityPolicy.candidate_execution()),  # type: ignore[arg-type]
            signer=signer,
            journal=journal,
            ingest=ingest,
            campaign_id="campaign-production-receipt-chain",
            protocol_digest=digest_for("production-receipt-protocol"),
            policy_digest=AuthorityPolicy.candidate_execution().digest,
            clock=lambda: "2026-08-22T00:00:00Z",
        )
        passed = controller.evaluate(
            candidate_id="candidate-production-pass",
            task_id=task.template_id,
            source=source,
            opaque_input=task.evaluator_input,
            declared_locus=task.public_locus,
        )
        wrong = controller.evaluate(
            candidate_id="candidate-production-wrong",
            task_id=task.template_id,
            source=source,
            opaque_input=task.evaluator_input,
            declared_locus=task.public_locus,
        )
        self.assertEqual(passed.diagnostic_enum, Diagnostic.PASS.value)
        self.assertEqual(wrong.diagnostic_enum, Diagnostic.WRONG_OUTPUT.value)
        production_records = {record["receipt_id"]: record for record in journal.receipts()}
        self.assertEqual(production_records[passed.receipt_ids[0]]["decision"], "ALLOW")
        self.assertEqual(production_records[passed.receipt_ids[1]]["diagnostic_enum"], Diagnostic.PASS.value)
        self.assertEqual(production_records[passed.receipt_ids[2]]["receipt_type"], "EFFECT")
        self.assertEqual(production_records[passed.receipt_ids[2]]["diagnostic_enum"], Diagnostic.PASS.value)
        self.assertEqual(production_records[wrong.receipt_ids[1]]["diagnostic_enum"], Diagnostic.WRONG_OUTPUT.value)

        internal_controller = EvaluatorController(
            sandbox=EvidenceSandbox([]),  # type: ignore[arg-type]
            hidden_runner=HiddenEvaluatorRunner.from_corpus(self.corpus),
            broker=AuthorityBroker(None, AuthorityPolicy.candidate_execution()),
            signer=signer,
            journal=journal,
            ingest=ingest,
            campaign_id="campaign-production-receipt-chain",
            protocol_digest=digest_for("production-receipt-protocol"),
            policy_digest=AuthorityPolicy.candidate_execution().digest,
            clock=lambda: "2026-08-22T00:00:00Z",
        )
        internal = internal_controller.evaluate(
            candidate_id="candidate-production-internal",
            task_id=task.template_id,
            source=source,
            opaque_input=task.evaluator_input,
            declared_locus=task.public_locus,
        )
        self.assertEqual(internal.diagnostic_enum, Diagnostic.INTERNAL_ERROR.value)
        self.assertTrue(internal.failure_family_root)

        with EvidenceLedger(ledger_path, mode="read_only", blob_root=blob_root) as ledger:
            self.assertTrue(ledger.verify_integrity()["chain_valid"])
            projection = InMemoryProjection(collection_name="production-receipt-chain")
            manifest = projection.rebuild(
                ledger,
                lambda payload: [1.0, float(len(payload.get("diagnostic_enum", "")))],
                embedding_model_revision="embed-v1",
            )
            self.assertEqual(manifest["point_count"], 7)
            roots_by_event = {
                point.payload["source_event_id"]: point.payload["failure_family_root"]
                for point in projection.points.values()
            }
            receipt_records = {record["receipt_id"]: record for record in journal.receipts()}
            internal_receipt = next(
                record
                for record in receipt_records.values()
                if record["candidate_id"] == "candidate-production-internal" and record["receipt_type"] == "AUTHORITY"
            )
            internal_event = ledger.receipt_by_id(internal_receipt["receipt_id"])
            assert internal_event is not None
            self.assertEqual(roots_by_event[internal_event["event_id"]], internal.failure_family_root)
            for record in receipt_records.values():
                if record["receipt_id"] == internal_receipt["receipt_id"]:
                    continue
                event = ledger.receipt_by_id(record["receipt_id"])
                assert event is not None
                self.assertIsNone(roots_by_event[event["event_id"]])

        dummy_internal = dict(internal_receipt, failure_family_root=digest_for("dummy-internal-root"))
        with self.assertRaises(ReceiptVerificationError):
            with EvidenceLedger(self.root / "dummy-receipt.sqlite", blob_root=self.root / "dummy-receipt-blobs") as ledger:
                ledger.ingest_receipt(dummy_internal, signer.public_key)

        with EvidenceLedger(ledger_path, blob_root=blob_root, clock=lambda: "2026-08-22T00:00:00Z") as ledger:
            ledger.append_event(
                "RECEIPT",
                {"receipt": dummy_internal, "receipt_hash": digest_for("dummy-internal-wrapper")},
                campaign_id="campaign-production-receipt-chain",
                run_id="run-evaluation",
                task_id=task.template_id,
                subject_id="candidate-production-internal",
            )
        with EvidenceLedger(ledger_path, mode="read_only", blob_root=blob_root) as ledger:
            with self.assertRaises(ProjectionError):
                InMemoryProjection(collection_name="production-receipt-chain-dummy").rebuild(
                    ledger,
                    lambda payload: [1.0, float(len(payload.get("diagnostic_enum", "")))],
                    embedding_model_revision="embed-v1",
                )

    def test_candidate_retraction_by_candidate_id_is_consistent_across_views(self) -> None:
        ledger = EvidenceLedger(self.root / "retract.sqlite", blob_root=self.root / "retract-blobs")
        try:
            ledger.create_campaign(
                "campaign-retract",
                protocol_hash=digest_for("protocol"),
                source_commit="retract-test",
                model_revision="not-loaded",
                data_manifest_hash=digest_for("data"),
                evaluator_hash=digest_for("evaluator"),
                policy_hash=digest_for("policy"),
                seed_set=[1],
            )
            ledger.create_run(
                "run-retract",
                campaign_id="campaign-retract",
                arm="D",
                task_id="task-retract",
                seed=1,
                parent_checkpoint=None,
                start_state="READY",
                host_role="fixture",
                software_manifest_hash=digest_for("software"),
            )
            candidate = ledger.append_candidate(
                "candidate-retract",
                campaign_id="campaign-retract",
                run_id="run-retract",
                task_id="task-retract",
                parent_candidate_id=None,
                mutation_family="PURE_FUNCTION",
                patch_hash=digest_for("patch"),
                requested_authority="NONE",
                prompt_hash=digest_for("prompt"),
                model_hash=digest_for("model"),
                adapter_hash=digest_for("adapter"),
            )
            ledger.append_retraction("candidate-retract", reason_code="PREMISE_RETRACTED", retraction_source="FROZEN_EVALUATOR")
            self.assertEqual(ledger.candidate_disposition("candidate-retract"), "RETRACTED")
            self.assertEqual(ledger.event_disposition(candidate["event_id"]), "RETRACTED")
            self.assertNotIn(candidate["event_id"], {event["event_id"] for event in ledger.current_valid_events()})
        finally:
            ledger.close()


class TestCorrectionShock(EvaluationTestCase):
    def test_exact_factor_and_dependency_precision_recall(self) -> None:
        suite = CorrectionShockSuite()
        report = suite.audit_dependency_aware()
        self.assertEqual(report["fixture_count"], 12)
        self.assertEqual(report["clone_count"], 36)
        self.assertEqual(tuple(report["policies"]), SHOCK_POLICIES)
        self.assertEqual(tuple(report["post_shock_attempts"]), POST_SHOCK_ATTEMPTS)
        self.assertEqual(report["graph_precision"], 1.0)
        self.assertEqual(report["graph_recall"], 1.0)
        for clone in suite.clones():
            applied = clone.apply_correction()
            if clone.policy == "dependency-aware":
                self.assertEqual(applied.stale_nodes, frozenset(clone.fixture.expected_affected_descendants()))
                self.assertFalse(applied.restarted)
            elif clone.policy == "full-restart":
                self.assertTrue(applied.restarted)
            else:
                self.assertFalse(applied.stale_nodes)

    def test_shock_mismatch_and_wrong_boundary_fail_closed(self) -> None:
        suite = CorrectionShockSuite()
        first = suite.fixtures[0]
        changed_profile = ShockProfile(
            model_digest=first.profile.model_digest,
            adapter_digest=digest_for("changed-adapter"),
            authority_policy_digest=first.profile.authority_policy_digest,
            evaluator_digest=first.profile.evaluator_digest,
            rng_state_digest=first.profile.rng_state_digest,
            budget_profile_digest=first.profile.budget_profile_digest,
            prompt_manifest_digest=first.profile.prompt_manifest_digest,
            protocol_digest=first.profile.protocol_digest,
        )
        mismatched = ShockFixture(
            task_id=first.task_id,
            seed=first.seed,
            accepted_premise_id=first.accepted_premise_id,
            candidate_state=first.candidate_state,
            dependency_graph=first.dependency_graph,
            profile=changed_profile,
            correction_event_id=first.correction_event_id,
        )
        with self.assertRaises(ShockMismatchError):
            suite.verify_exact_match(first, mismatched)
        with self.assertRaises(ShockMismatchError):
            suite.clones()[0].apply_correction(attempt=5)

    def test_independent_oracle_rejects_wrong_edge_and_missing_edge_predictions(self) -> None:
        fixture = CorrectionShockSuite().fixtures[0]
        oracle = set(fixture.oracle_affected_nodes)
        wrong_edge = oracle | {"unrelated-child-01-11"}
        missing_edge = oracle - {next(iter(oracle))}
        self.assertLess(graph_precision_recall(oracle, wrong_edge)["precision"], 1.0)
        self.assertLess(graph_precision_recall(oracle, missing_edge)["recall"], 1.0)
        self.assertEqual(
            graph_precision_recall(oracle, ShockFixture.create(fixture.task_id, fixture.seed).expected_affected_descendants()),
            {"precision": 1.0, "recall": 1.0},
        )


class TestEvaluationSmokeAndCLI(EvaluationTestCase):
    def test_cpu_smoke_reports_actual_tier_and_no_hidden_public_leak(self) -> None:
        report = run_evaluation_smoke()
        self.assertIs(package_smoke, run_evaluation_smoke)
        self.assertEqual(report["smoke"], "PASS")
        self.assertEqual(report["runtime_tier"], "ceiling-docker-evaluation-fixture")
        self.assertFalse(report["model_dependencies"])
        self.assertFalse(report["openshell"]["claimed"])
        self.assertTrue(report["authority"]["enforceable"])
        self.assertEqual(report["determinism"], {"non_key_artifacts_byte_identical": True, "key_material_excluded": True})
        self.assertEqual(report["corpus"]["split_counts"], {"train": 20, "dev": 8, "heldout": 8})
        self.assertEqual(
            report["corpus"]["data_manifest_digest"],
            report["corpus"]["evaluated_data_manifest_digest"],
        )
        self.assertEqual(
            set(report["corpus"]["evaluated_source_bindings"]),
            set(report["two_process"]["heldout_results"]),
        )
        for task_id, binding in report["corpus"]["evaluated_source_bindings"].items():
            self.assertTrue(binding["source_bytes_match_frozen"])
            self.assertEqual(binding["evaluated_source_digest"], binding["frozen_candidate_source_digest"])
            self.assertEqual(
                binding["evaluated_source_digest"],
                report["two_process"]["heldout_results"][task_id]["candidate_artifact_digest"],
            )
        self.assertEqual(report["evaluation"]["disposition"], "PROMOTED")
        self.assertNotIn("receipt_ids", report["evaluation"])
        self.assertEqual(report["receipts"]["candidate_disposition"], "PROMOTED")
        self.assertTrue(all(report["sandbox_negative_controls"].values()))
        self.assertEqual(report["two_process"]["mode"], "two-process-cpu-only-evaluation")
        self.assertFalse(report["two_process"]["evaluator_used_sqlite"])
        self.assertEqual(report["two_process"]["evaluator_sqlite_connect_calls"], 1)
        self.assertEqual(report["two_process"]["evaluator_process_sqlite_connect_calls"], 1)
        self.assertEqual(report["two_process"]["evaluator_evaluate_sqlite_connect_calls"], 0)
        self.assertTrue(report["two_process"]["evaluator_ledger_cohosting_evidence"]["denied"])
        self.assertFalse(report["two_process"]["evaluator_ledger_cohosting_evidence"]["probe_path_created"])
        self.assertTrue(report["two_process"]["evaluator_sqlite_evidence"]["denied"])
        self.assertEqual(report["two_process"]["evaluator_sqlite_evidence"]["connect_calls"], 1)
        self.assertEqual(report["two_process"]["evaluator_sqlite_evidence"]["exception"], "PermissionError")
        self.assertFalse(report["two_process"]["evaluator_sqlite_evidence"]["probe_path_created"])
        self.assertTrue(report["two_process"]["evaluator_private_tree_clean"])
        self.assertTrue(report["two_process"]["evaluator_non_receipt_ipc_rejected"])
        self.assertTrue(all(report["two_process"]["negative_controls"].values()))
        self.assertTrue(report["two_process"]["evaluator_key_ephemeral"])
        self.assertEqual(report["two_process"]["evaluator_ledger_mounts"], [])
        self.assertEqual(report["two_process"]["evaluator_ledger_paths_visible"], [])
        self.assertEqual(set(report["two_process"]["family_dispositions"].values()), {"PROMOTED"})
        self.assertEqual(report["two_process"]["heldout_task_count"], 8)
        self.assertEqual(set(report["two_process"]["heldout_dispositions"].values()), {"PROMOTED"})
        self.assertEqual(set(report["two_process"]["signed_verdict_correctness"].values()), {True})
        self.assertTrue(report["receipts"]["journal"]["chain_valid"])
        self.assertTrue(report["receipts"]["ledger"]["chain_valid"])
        self.assertEqual(report["two_process"]["wrong_locus_diagnostic"], "MUTATION_LOCUS_VIOLATION")
        self.assertEqual(report["two_process"]["wrong_locus_disposition"], "REJECTED")

    def test_cli_evaluation_smoke_is_runnable_and_bounded(self) -> None:
        completed = subprocess.run(
            [sys.executable, "-m", "egv", "evaluation", "smoke", "--json"],
            cwd=str(Path(__file__).resolve().parents[1]),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads(completed.stdout)
        self.assertEqual(report["smoke"], "PASS")
        self.assertEqual(report["corpus"]["task_count"], 36)
        self.assertEqual(report["freeze"]["public_scan"]["findings"], [])
        self.assertNotIn("private_key", completed.stdout.lower())
        self.assertNotIn("corpus-seed.bin", completed.stdout)
        self.assertNotIn("/home/", completed.stdout)

    def test_existing_two_process_ipc_floor_is_green(self) -> None:
        report = run_two_process_smoke()
        self.assertEqual(report["mode"], "two-process-cpu-only")
        self.assertFalse(report["evaluator_used_sqlite"])
        self.assertTrue(report["evaluator_non_receipt_ipc_rejected"])
        self.assertEqual(report["candidate_disposition"], "PROMOTED")

    def test_public_docs_match_the_frozen_evaluation_contract(self) -> None:
        root = Path(__file__).resolve().parents[1]
        slice_doc = (root / "docs" / "egv-evaluation-slice.md").read_text(encoding="utf-8")
        runbook = (root / "docs" / "runbooks" / "evidence-governed-variation-dual-spark.md").read_text(encoding="utf-8")
        index = (root / "docs" / "INDEX.md").read_text(encoding="utf-8")
        for text in (slice_doc, runbook):
            self.assertIn("ceiling-docker-evaluation-fixture", text)
            self.assertNotIn("41001", text)
            self.assertNotIn("41002", text)
            self.assertIn("evaluator-private", text)
            self.assertIn("fail closed", text)
        self.assertIn("does not create or claim to freeze the evaluator receipt journal", slice_doc)
        self.assertIn("egv-evaluation-slice.md", index)
        self.assertIn("--allow-provisional", runbook)


if __name__ == "__main__":
    unittest.main()
