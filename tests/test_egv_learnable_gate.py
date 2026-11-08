from __future__ import annotations

import importlib
import os
from pathlib import Path
import py_compile
import sys
import tempfile
import unittest
from unittest.mock import patch

from egv.canonical import GENESIS_HASH, digest_bytes, digest_for
from egv.evaluation.learnable import LearnableCorpusV2
from egv.errors import EGVError, ReceiptVerificationError
from egv.evaluation.learnable_gate import (
    BLIND_EVALUATION_COUNT,
    DurableDeliveryJournal,
    EXPECTED_RECEIPT_COUNT,
    GATE_A_CAMPAIGN_ID,
    ORACLE_EVALUATION_COUNT,
    STARTER_EVALUATION_COUNT,
    StreamingReceiptJournal,
    TOTAL_EVALUATION_COUNT,
    _PUBLIC_CORE_FIELDS,
    _atomic_write_new,
    _canonical_record_bytes,
    _candidate_sources,
    _expected_evaluation_bindings,
    _fresh_receipt_replay,
    _privacy_scan,
    _path_identity,
    _replay_receipt_outcomes,
    _require_distinct_authority_keys,
    _run_spawn_stage,
    _sign_closed_record,
    _validated_external_signer,
    _validated_identity_record,
    build_evaluator_source_manifest,
    build_gate_a_plan,
    run_gate_a_corpus_validation,
    validate_evaluator_source_manifest,
    validate_gate_a_plan,
    validate_gate_a_public_result,
)
from egv.evaluation.learnable_runtime import build_learnable_hidden_runner
from egv.evaluation.sandbox import DockerSandboxConfig
from egv.receipts import ReceiptSigner, receipt_hash


def _send_canonical_result_for_test(value, sender) -> None:
    sender.send_bytes(_canonical_record_bytes(value))
    sender.close()


def _source_only_import_probe_for_test(module_root: str, module_name: str, output: str) -> None:
    sys.path.insert(0, module_root)
    importlib.invalidate_caches()
    imported = importlib.import_module(module_name)
    Path(output).write_text(str(imported.VALUE), encoding="utf-8")


def _receipt(signer: ReceiptSigner, sequence: int, previous: str, candidate: str) -> dict:
    return signer.sign_receipt(
        {
            "campaign_id": "gate-a-test",
            "run_id": "run-evaluation",
            "task_id": "task-test",
            "receipt_type": "AUTHORITY",
            "request_id": "request-{}".format(sequence),
            "candidate_id": candidate,
            "decision": "ALLOW",
        },
        sequence=sequence,
        previous_receipt_hash=previous,
        idempotency_key="gate-a-test-{}".format(sequence),
    )


def _public_result(signer: ReceiptSigner, **overrides) -> dict:
    digests = {
        field: "a" * 64
        for field in _PUBLIC_CORE_FIELDS
        if field.endswith("_digest") or field in {"receipt_chain_head", "delivery_chain_head"}
    }
    preprivacy = {
        **digests,
        "schema_version": "egv-gate-a-public-result-v1",
        "disposition": "GO_CORPUS_VALIDATION",
        "evidence_scope": "production-corpus-validation-no-model-quality-claim",
        "production_acceptance": True,
        "task_bearing_model_requests_permitted": False,
        "separate_go_task_bearing_required": True,
        "model_quality_claimed": False,
        "cross_node_transport_exercised": False,
        "source_commit": "b" * 40,
        "controller_key_id": "key_" + "c" * 64,
        "freeze_signer_key_id": signer.key_id,
        "oracle_task_count": 170,
        "oracle_evaluation_count": 340,
        "oracle_pass_count": 340,
        "blind_task_count": 18,
        "blind_evaluation_count": 36,
        "blind_pass_count": 36,
        "starter_task_count": 18,
        "starter_evaluation_count": 18,
        "starter_wrong_output_count": 18,
        "total_evaluation_count": 394,
        "infrastructure_loss_count": 0,
        "expected_receipt_count": 1182,
        "receipt_count": 1182,
        "receipt_chain_valid": True,
        "delivery_count": 1182,
        "delivery_chain_valid": True,
        "receipt_delivery_bijection": True,
        "sandbox_preflight_passed": True,
        "sandbox_negative_control_count": 6,
        "sandbox_negative_controls_all_passed": True,
        "sandbox_postflight_passed": True,
    }
    preprivacy.update(overrides)
    if set(preprivacy) != set(_PUBLIC_CORE_FIELDS) - {"privacy_scan"}:
        raise AssertionError("test public result fixture is incomplete")
    privacy = _privacy_scan(preprivacy, forbidden_tokens=[])
    return _sign_closed_record(
        {**preprivacy, "privacy_scan": privacy},
        signer,
        digest_field="result_digest",
        signature_field="freeze_authority_signature",
    )


def _signed_binding_receipts(
    signer: ReceiptSigner,
    bindings: dict,
    *,
    protocol_digest: str,
    policy_digest: str,
    evaluator_revision: str,
    forged_index: int | None = None,
) -> list[dict]:
    records = []
    previous = GENESIS_HASH
    sequence = 0
    for index, binding in enumerate(bindings["bindings"]):
        common = {
            "campaign_id": GATE_A_CAMPAIGN_ID,
            "run_id": "run-evaluation",
            "task_id": "forged-task" if index == forged_index else binding["task_id"],
            "candidate_id": binding["candidate_id"],
            "candidate_artifact_digest": binding["candidate_artifact_digest"],
            "protocol_digest": protocol_digest,
            "policy_digest": policy_digest,
            "evaluator_digest": digest_for(evaluator_revision),
            "task_family": binding["task_family"],
            "normalized_public_locus": binding["normalized_public_locus"],
            "public_rule_id": binding["public_rule_id"],
        }
        diagnostic = "WRONG_OUTPUT" if binding["candidate_kind"] == "starter" else "PASS"
        fields = (
            {
                **common,
                "receipt_type": "AUTHORITY",
                "request_id": "request-authority-{}".format(binding["candidate_id"]),
                "decision": "ALLOW",
            },
            {
                **common,
                "receipt_type": "VERDICT",
                "request_id": "request-verdict-{}".format(binding["candidate_id"]),
                "decision": "FAIL" if diagnostic == "WRONG_OUTPUT" else "PASS",
                "diagnostic_enum": diagnostic,
                "input_digest": binding["input_digest"],
            },
            {
                **common,
                "receipt_type": "EFFECT",
                "request_id": "request-effect-{}".format(binding["candidate_id"]),
                "decision": "ALLOW",
                "diagnostic_enum": diagnostic,
                "normalized_action_hash": digest_for({"action": "execute_candidate", "locus": "module:main"}),
            },
        )
        for receipt_fields in fields:
            sequence += 1
            receipt_type = str(receipt_fields["receipt_type"]).lower()
            receipt = signer.sign_receipt(
                receipt_fields,
                sequence=sequence,
                previous_receipt_hash=previous,
                idempotency_key="{}:{}".format(receipt_type, binding["candidate_id"]),
            )
            records.append(receipt)
            previous = receipt_hash(receipt)
    return records


class GateAPlanAndSourceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_plan_is_exactly_394_evaluations_and_1182_receipts(self):
        plan = validate_gate_a_plan(build_gate_a_plan())
        self.assertEqual(plan["oracle_evaluation_count"], ORACLE_EVALUATION_COUNT)
        self.assertEqual(plan["blind_evaluation_count"], BLIND_EVALUATION_COUNT)
        self.assertEqual(plan["starter_evaluation_count"], STARTER_EVALUATION_COUNT)
        self.assertEqual(plan["total_evaluation_count"], TOTAL_EVALUATION_COUNT)
        self.assertEqual(plan["expected_receipt_count"], EXPECTED_RECEIPT_COUNT)
        self.assertFalse(plan["task_bearing_model_requests_permitted"])
        self.assertFalse(plan["model_quality_claimed"])
        self.assertFalse(plan["cross_node_transport_exercised"])

    def test_plan_rejects_any_count_or_claim_mutation(self):
        plan = build_gate_a_plan()
        for field, value in (
            ("total_evaluation_count", 393),
            ("expected_receipt_count", 1179),
            ("task_bearing_model_requests_permitted", True),
            ("model_quality_claimed", True),
        ):
            mutation = dict(plan)
            mutation[field] = value
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "frozen 394"):
                validate_gate_a_plan(mutation)

    def test_source_manifest_is_exact_sorted_and_byte_sensitive(self):
        package = self.root / "package"
        (package / "egv" / "evaluation").mkdir(parents=True)
        first = package / "egv" / "a.py"
        second = package / "egv" / "evaluation" / "b.py"
        first.write_text("A = 1\n", encoding="utf-8")
        second.write_text("B = 2\n", encoding="utf-8")
        commit = "a" * 40
        manifest = build_evaluator_source_manifest(package.resolve(), commit, require_clean_git=False)
        validated = validate_evaluator_source_manifest(manifest)
        self.assertEqual([item["path"] for item in validated["files"]], ["egv/a.py", "egv/evaluation/b.py"])
        second.write_text("B = 3\n", encoding="utf-8")
        changed = build_evaluator_source_manifest(package.resolve(), commit, require_clean_git=False)
        self.assertNotEqual(manifest["manifest_digest"], changed["manifest_digest"])

    def test_source_manifest_rejects_symlinked_python_file(self):
        package = self.root / "package"
        (package / "egv").mkdir(parents=True)
        target = self.root / "target.py"
        target.write_text("VALUE = 1\n", encoding="utf-8")
        link = package / "egv" / "linked.py"
        try:
            link.symlink_to(target)
        except OSError:
            self.skipTest("local filesystem does not permit symlinks")
        with self.assertRaisesRegex(EGVError, "non-regular"):
            build_evaluator_source_manifest(package.resolve(), "a" * 40, require_clean_git=False)

    def test_real_corpus_builds_exact_frozen_source_inventory(self):
        seed = self.root / "seed.bin"
        seed.write_bytes(bytes(range(32)))
        corpus = LearnableCorpusV2.generate(secret_seed_file=seed)
        oracle, blind, starter, inventory = _candidate_sources(corpus)
        self.assertEqual(len(oracle), 170)
        self.assertEqual(len(blind), 18)
        self.assertEqual(len(starter), 18)
        self.assertEqual(inventory["oracle_source_count"], 170)
        self.assertEqual(inventory["blind_source_count"], 18)
        self.assertEqual(inventory["starter_source_count"], 18)
        self.assertEqual(len(inventory["entries"]), 206)


class GateADurableJournalTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.signer = ReceiptSigner(b"R" * 32)

    def test_receipt_and_delivery_journals_freshly_replay_as_bijection(self):
        receipts_path = self.root / "receipts.jsonl"
        delivery_path = self.root / "deliveries.jsonl"
        journal = StreamingReceiptJournal(receipts_path, self.signer.public_key)
        delivery = DurableDeliveryJournal(delivery_path)
        previous = GENESIS_HASH
        records = []
        for sequence in range(1, 4):
            receipt = _receipt(self.signer, sequence, previous, "candidate-{}".format(sequence))
            stored = journal.append(receipt)
            delivery.append(stored)
            records.append(stored)
            previous = receipt_hash(stored)
        fresh, receipt_evidence = _fresh_receipt_replay(receipts_path, self.signer.public_key)
        replay = DurableDeliveryJournal.replay(delivery_path, fresh)
        self.assertEqual(replay["count"], 3)
        self.assertTrue(replay["chain_valid"])
        self.assertTrue(replay["receipt_delivery_bijection"])
        self.assertEqual(receipt_evidence["head"], receipt_hash(records[-1]))

    def test_truncated_delivery_journal_is_rejected(self):
        receipt = _receipt(self.signer, 1, GENESIS_HASH, "candidate-one")
        delivery = DurableDeliveryJournal(self.root / "deliveries.jsonl")
        delivery.append(receipt)
        payload = delivery.path.read_bytes()
        delivery.path.write_bytes(payload[:-5])
        with self.assertRaisesRegex(ReceiptVerificationError, "interrupted|canonical|bijection"):
            DurableDeliveryJournal.replay(delivery.path, [receipt])

    def test_truncated_receipt_journal_is_rejected_by_fresh_replay(self):
        receipts_path = self.root / "receipts.jsonl"
        journal = StreamingReceiptJournal(receipts_path, self.signer.public_key)
        journal.append(_receipt(self.signer, 1, GENESIS_HASH, "candidate-one"))
        receipts_path.write_bytes(receipts_path.read_bytes()[:-7])
        with self.assertRaisesRegex(ReceiptVerificationError, "interrupted"):
            _fresh_receipt_replay(receipts_path, self.signer.public_key)

    def test_missing_and_duplicate_delivery_are_rejected(self):
        first = _receipt(self.signer, 1, GENESIS_HASH, "candidate-one")
        second = _receipt(self.signer, 2, receipt_hash(first), "candidate-two")
        delivery = DurableDeliveryJournal(self.root / "deliveries.jsonl")
        delivery.append(first)
        with self.assertRaisesRegex(ReceiptVerificationError, "bijection"):
            DurableDeliveryJournal.replay(delivery.path, [first, second])
        with self.assertRaisesRegex(Exception, "more than once"):
            delivery.append(first)

    def test_delivery_reorder_is_rejected_even_when_set_bijection_is_complete(self):
        first = _receipt(self.signer, 1, GENESIS_HASH, "candidate-one")
        second = _receipt(self.signer, 2, receipt_hash(first), "candidate-two")
        third = _receipt(self.signer, 3, receipt_hash(second), "candidate-three")
        receipts = [first, second, third]
        delivery = DurableDeliveryJournal(self.root / "deliveries.jsonl")
        for receipt in reversed(receipts):
            delivery.append(receipt)
        with self.assertRaisesRegex(ReceiptVerificationError, "order"):
            DurableDeliveryJournal.replay(delivery.path, receipts)


class GateAExactReceiptReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        temporary = tempfile.TemporaryDirectory()
        cls.addClassCleanup(temporary.cleanup)
        root = Path(temporary.name)
        seed = root / "seed.bin"
        seed.write_bytes(bytes(range(32)))
        cls.corpus = LearnableCorpusV2.generate(secret_seed_file=seed)
        oracle, blind, starter, _inventory = _candidate_sources(cls.corpus)
        cls.evaluator_revision = "e" * 64
        _runner, addresses = build_learnable_hidden_runner(cls.corpus.tasks, evaluator_revision=cls.evaluator_revision)
        cls.bindings = _expected_evaluation_bindings(cls.corpus, addresses, oracle, blind, starter)
        cls.protocol_digest = "1" * 64
        cls.policy_digest = "2" * 64
        cls.signer = ReceiptSigner(b"X" * 32)

    def test_fresh_replay_requires_exact_ordered_source_task_case_bijection(self):
        receipts = _signed_binding_receipts(
            self.signer,
            self.bindings,
            protocol_digest=self.protocol_digest,
            policy_digest=self.policy_digest,
            evaluator_revision=self.evaluator_revision,
        )
        self.assertEqual(len(receipts), 1182)
        counts = _replay_receipt_outcomes(
            receipts,
            controller_public_key=self.signer.public_key,
            protocol_digest=self.protocol_digest,
            policy_digest=self.policy_digest,
            evaluator_revision=self.evaluator_revision,
            expected_bindings=self.bindings,
        )
        self.assertEqual(counts["oracle_pass_count"], 340)
        self.assertEqual(counts["blind_pass_count"], 36)
        self.assertEqual(counts["starter_wrong_output_count"], 18)

        forged = _signed_binding_receipts(
            self.signer,
            self.bindings,
            protocol_digest=self.protocol_digest,
            policy_digest=self.policy_digest,
            evaluator_revision=self.evaluator_revision,
            forged_index=0,
        )
        with self.assertRaisesRegex(ReceiptVerificationError, "frozen campaign"):
            _replay_receipt_outcomes(
                forged,
                controller_public_key=self.signer.public_key,
                protocol_digest=self.protocol_digest,
                policy_digest=self.policy_digest,
                evaluator_revision=self.evaluator_revision,
                expected_bindings=self.bindings,
            )


class GateAPrivacyAndBoundaryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def test_privacy_scan_passes_safe_digest_only_record(self):
        record = _privacy_scan(
            {"schema_version": "safe", "digest": "a" * 64, "task_bearing": False},
            forbidden_tokens=[b"private-hidden-value"],
        )
        self.assertTrue(record["scan_passed"])
        self.assertEqual(record["finding_count"], 0)

    def test_privacy_scan_detects_hidden_value_path_topology_and_key_marker(self):
        cases = (
            ({"value": "private-hidden-value"}, [b"private-hidden-value"]),
            ({"value": "C:\\Users\\private\\seed.bin"}, []),
            ({"value": "10.255.255.254"}, []),
            ({"private_key": "present"}, []),
        )
        for subject, tokens in cases:
            with self.subTest(subject=subject), self.assertRaisesRegex(EGVError, "prohibited material"):
                _privacy_scan(subject, forbidden_tokens=tokens)

    def test_privacy_scan_covers_every_byte_atomic_writer_emits(self):
        subject = {"schema_version": "safe", "digest": "d" * 64}
        privacy = _privacy_scan(subject, forbidden_tokens=[])
        output = self.root / "public.json"
        _atomic_write_new(output, subject, private=False)
        self.assertEqual(output.read_bytes(), _canonical_record_bytes(subject))
        self.assertEqual(privacy["subject_digest"], digest_bytes(output.read_bytes()))

    def test_atomic_writer_removes_exact_go_when_parent_moves_after_link(self):
        parent = self.root / "public-parent"
        parent.mkdir()
        parked = self.root / "parked-parent"
        output = parent / "go.json"
        real_link = os.link

        def link_then_move(*args, **kwargs):
            real_link(*args, **kwargs)
            parent.replace(parked)
            parent.mkdir()

        with patch("egv.evaluation.learnable_gate.os.link", side_effect=link_then_move):
            with self.assertRaisesRegex(EGVError, "identity changed|unavailable"):
                _atomic_write_new(output, {"disposition": "GO_CORPUS_VALIDATION"}, private=False)
        self.assertFalse(output.exists())
        self.assertFalse((parked / "go.json").exists())

    def test_public_schema_has_no_evaluator_chosen_opaque_preflight_digest_slot(self):
        signer = ReceiptSigner(b"P" * 32)
        result = _public_result(signer)
        self.assertNotIn("sandbox_preflight_digest", _PUBLIC_CORE_FIELDS)
        core = {
            key: value for key, value in result.items() if key not in {"result_digest", "freeze_authority_signature"}
        }
        core["sandbox_preflight_digest"] = signer.private_key_raw.hex()
        exfiltrating = _sign_closed_record(
            core,
            signer,
            digest_field="result_digest",
            signature_field="freeze_authority_signature",
        )
        with self.assertRaisesRegex(ValueError, "closed record"):
            validate_gate_a_public_result(exfiltrating, signer.public_key)

    def test_public_result_requires_exact_freeze_key_and_valid_signature(self):
        signer = ReceiptSigner(b"P" * 32)
        result = _public_result(signer)
        self.assertEqual(
            validate_gate_a_public_result(result, signer.public_key)["disposition"],
            "GO_CORPUS_VALIDATION",
        )
        with self.assertRaisesRegex(ValueError, "signature|key"):
            validate_gate_a_public_result(result, ReceiptSigner(b"Q" * 32).public_key)
        tampered = dict(result)
        tampered["source_commit"] = "f" * 40
        with self.assertRaisesRegex(ValueError, "digest|signature"):
            validate_gate_a_public_result(tampered, signer.public_key)

    def test_public_result_rejects_malformed_commit_and_key_ids_even_when_resigned(self):
        signer = ReceiptSigner(b"P" * 32)
        for overrides in (
            {"source_commit": "short"},
            {"controller_key_id": "controller"},
            {"freeze_signer_key_id": "key_" + "f" * 64},
        ):
            with self.subTest(overrides=overrides), self.assertRaisesRegex(ValueError, "claim boundary"):
                validate_gate_a_public_result(_public_result(signer, **overrides), signer.public_key)

    def test_external_signer_rejects_symlink_and_path_swap(self):
        signer = ReceiptSigner(b"S" * 32)
        private = self.root / "private.key"
        public = self.root / "public.key"
        private.write_bytes(signer.private_key_raw)
        public.write_bytes(signer.public_key_raw)
        self.assertEqual(_validated_external_signer(private, public, "test").key_id, signer.key_id)
        public.write_bytes(ReceiptSigner(b"T" * 32).public_key_raw)
        with self.assertRaisesRegex(EGVError, "do not match"):
            _validated_external_signer(private, public, "test")
        public.write_bytes(signer.public_key_raw)
        link = self.root / "private-link.key"
        try:
            link.symlink_to(private)
        except OSError:
            self.skipTest("local filesystem does not permit symlinks")
        with self.assertRaisesRegex(EGVError, "unavailable|symlink|reparse"):
            _validated_external_signer(link, public, "test")

    def test_controller_and_freeze_authority_cannot_use_copied_identical_key_material(self):
        signer = ReceiptSigner(b"S" * 32)
        with self.assertRaisesRegex(EGVError, "distinct"):
            _require_distinct_authority_keys(signer.public_key, signer.public_key_raw)
        with self.assertRaisesRegex(ValueError, "claim boundary"):
            validate_gate_a_public_result(
                _public_result(signer, controller_key_id=signer.key_id),
                signer.public_key,
            )

    def _input_files(self):
        signer_a = ReceiptSigner(b"A" * 32)
        signer_b = ReceiptSigner(b"B" * 32)
        values = {
            "seed": bytes(range(32)),
            "controller-private": signer_a.private_key_raw,
            "controller-public": signer_a.public_key_raw,
            "freeze-private": signer_b.private_key_raw,
            "freeze-public": signer_b.public_key_raw,
        }
        paths = {}
        for name, value in values.items():
            path = self.root / (name + ".bin")
            path.write_bytes(value)
            paths[name] = path.resolve()
        return paths

    def test_child_identity_record_preserves_full_parent_selected_file_state(self):
        record = {
            name: (1, 2)
            for name in (
                "evaluator_seed",
                "controller_private",
                "controller_public",
                "freeze_private",
                "freeze_public",
                "package_root",
                "workspace",
                "private",
                "issuer",
                "evaluator",
                "failures",
                "output_parent",
                "bytecode",
                "issuer_bytecode",
                "evaluator_bytecode",
                "verifier_bytecode",
            )
        }
        for name in (
            "evaluator_seed_state",
            "controller_private_state",
            "controller_public_state",
            "freeze_private_state",
            "freeze_public_state",
        ):
            record[name] = (1, 2, 3, 4, 5)
        validated = _validated_identity_record(record)
        self.assertEqual(validated["evaluator_seed_state"], (1, 2, 3, 4, 5))

    def test_parent_orchestrates_three_stages_in_order_and_never_passes_controller_private_to_verifier(self):
        inputs = self._input_files()
        workspace = (self.root / "workspace").resolve()
        output = (self.root / "public.json").resolve()
        calls = []
        safe = {"schema_version": "safe"}

        def fake_stage(name, _target, args, *, timeout_seconds, expect_result=False, **_kwargs):
            calls.append((name, tuple(args), timeout_seconds, expect_result))
            if name == "verifier":
                self.assertFalse(output.exists())
                return {}
            return None

        with (
            patch("egv.evaluation.learnable_gate._run_spawn_stage", side_effect=fake_stage),
            patch("egv.evaluation.learnable_gate.validate_gate_a_public_result", return_value=safe),
        ):
            with patch(
                "egv.evaluation.learnable_gate._read_private_bytes",
                side_effect=AssertionError("parent read private bytes"),
            ):
                result = run_gate_a_corpus_validation(
                    inputs["seed"],
                    inputs["controller-private"],
                    inputs["controller-public"],
                    inputs["freeze-private"],
                    inputs["freeze-public"],
                    workspace,
                    output,
                    Path.cwd().resolve(),
                    "a" * 40,
                    DockerSandboxConfig(),
                )
        self.assertEqual(result, safe)
        self.assertEqual([call[0] for call in calls], ["issuer", "evaluator", "verifier"])
        self.assertNotIn(str(inputs["controller-private"]), calls[2][1])
        self.assertNotIn(str(output), calls[2][1])
        self.assertTrue(calls[2][3])
        self.assertEqual(output.read_bytes(), _canonical_record_bytes(safe))

    def test_child_failure_never_publishes_and_preserves_unrelated_reserved_path(self):
        inputs = self._input_files()
        workspace = (self.root / "workspace").resolve()
        output = (self.root / "public.json").resolve()

        def fail_evaluator(name, _target, _args, *, timeout_seconds, **_kwargs):
            del timeout_seconds
            if name == "evaluator":
                output.write_text("unrelated\n", encoding="utf-8")
                raise EGVError("child failed")

        with patch("egv.evaluation.learnable_gate._run_spawn_stage", side_effect=fail_evaluator):
            with self.assertRaisesRegex(EGVError, "child failed"):
                run_gate_a_corpus_validation(
                    inputs["seed"],
                    inputs["controller-private"],
                    inputs["controller-public"],
                    inputs["freeze-private"],
                    inputs["freeze-public"],
                    workspace,
                    output,
                    Path.cwd().resolve(),
                    "a" * 40,
                    DockerSandboxConfig(),
                )
        self.assertEqual(output.read_text(encoding="utf-8"), "unrelated\n")

    def test_parent_stops_before_next_child_when_selected_input_identity_is_swapped(self):
        inputs = self._input_files()
        workspace = (self.root / "workspace").resolve()
        output = (self.root / "public.json").resolve()
        calls = []

        def swap_after_issuer(name, _target, _args, *, timeout_seconds, **_kwargs):
            del timeout_seconds
            calls.append(name)
            if name == "issuer":
                replacement = self.root / "replacement-seed.bin"
                replacement.write_bytes(b"Z" * 32)
                inputs["seed"].unlink()
                replacement.replace(inputs["seed"])

        with patch(
            "egv.evaluation.learnable_gate._run_spawn_stage",
            side_effect=swap_after_issuer,
        ):
            with self.assertRaisesRegex(EGVError, "identity changed"):
                run_gate_a_corpus_validation(
                    inputs["seed"],
                    inputs["controller-private"],
                    inputs["controller-public"],
                    inputs["freeze-private"],
                    inputs["freeze-public"],
                    workspace,
                    output,
                    Path.cwd().resolve(),
                    "a" * 40,
                    DockerSandboxConfig(),
                )
        self.assertEqual(calls, ["issuer"])
        self.assertFalse(output.exists())

    def test_parent_stops_before_next_child_when_selected_input_changes_in_place(self):
        inputs = self._input_files()
        workspace = (self.root / "workspace").resolve()
        output = (self.root / "public.json").resolve()
        calls = []

        def mutate_after_issuer(name, _target, _args, *, timeout_seconds, **_kwargs):
            del timeout_seconds
            calls.append(name)
            if name == "issuer":
                with inputs["seed"].open("r+b") as handle:
                    handle.write(b"Z" * 32)
                    handle.flush()
                    os.fsync(handle.fileno())

        with patch(
            "egv.evaluation.learnable_gate._run_spawn_stage",
            side_effect=mutate_after_issuer,
        ):
            with self.assertRaisesRegex(EGVError, "content identity changed"):
                run_gate_a_corpus_validation(
                    inputs["seed"],
                    inputs["controller-private"],
                    inputs["controller-public"],
                    inputs["freeze-private"],
                    inputs["freeze-public"],
                    workspace,
                    output,
                    Path.cwd().resolve(),
                    "a" * 40,
                    DockerSandboxConfig(),
                )
        self.assertEqual(calls, ["issuer"])
        self.assertFalse(output.exists())

    def test_output_parent_swap_after_verifier_cannot_leave_a_go_artifact(self):
        inputs = self._input_files()
        workspace = (self.root / "workspace").resolve()
        output_parent = self.root / "public-parent"
        output_parent.mkdir()
        output = (output_parent / "public.json").resolve()
        parked_parent = self.root / "parked-public-parent"

        freeze_signer = ReceiptSigner(inputs["freeze-private"].read_bytes())
        signed_go = _public_result(freeze_signer)

        def swap_during_verifier(
            name,
            _target,
            _args,
            *,
            timeout_seconds,
            expect_result=False,
            **_kwargs,
        ):
            del timeout_seconds
            if name == "verifier":
                self.assertTrue(expect_result)
                output_parent.replace(parked_parent)
                output_parent.mkdir()
                output.write_text("unrelated replacement file\n", encoding="utf-8")
                return signed_go
            return None

        with patch(
            "egv.evaluation.learnable_gate._run_spawn_stage",
            side_effect=swap_during_verifier,
        ):
            with self.assertRaisesRegex(EGVError, "identity changed"):
                run_gate_a_corpus_validation(
                    inputs["seed"],
                    inputs["controller-private"],
                    inputs["controller-public"],
                    inputs["freeze-private"],
                    inputs["freeze-public"],
                    workspace,
                    output,
                    Path.cwd().resolve(),
                    "a" * 40,
                    DockerSandboxConfig(),
                )
        self.assertFalse((parked_parent / "public.json").exists())
        self.assertEqual(output.read_text(encoding="utf-8"), "unrelated replacement file\n")

    def test_parent_rejects_hardlinked_private_inputs_before_spawn(self):
        inputs = self._input_files()
        hardlink = self.root / "freeze-public-hardlink.bin"
        try:
            hardlink.hardlink_to(inputs["controller-public"])
        except OSError:
            self.skipTest("local filesystem does not permit hardlinks")
        with patch("egv.evaluation.learnable_gate._run_spawn_stage") as stage:
            with self.assertRaises(EGVError):
                run_gate_a_corpus_validation(
                    inputs["seed"],
                    inputs["controller-private"],
                    inputs["controller-public"],
                    inputs["freeze-private"],
                    hardlink.resolve(),
                    (self.root / "workspace").resolve(),
                    (self.root / "public.json").resolve(),
                    Path.cwd().resolve(),
                    "a" * 40,
                    DockerSandboxConfig(),
                )
        stage.assert_not_called()

    def test_parent_rejects_workspace_or_output_inside_package_checkout(self):
        inputs = self._input_files()
        package = self.root / "package"
        package.mkdir()
        cases = (
            (package / "workspace", self.root / "public.json"),
            (self.root / "workspace", package / "public.json"),
        )
        for workspace, output in cases:
            with (
                self.subTest(workspace=workspace, output=output),
                patch("egv.evaluation.learnable_gate._run_spawn_stage") as stage,
            ):
                with self.assertRaisesRegex(EGVError, "disjoint"):
                    run_gate_a_corpus_validation(
                        inputs["seed"],
                        inputs["controller-private"],
                        inputs["controller-public"],
                        inputs["freeze-private"],
                        inputs["freeze-public"],
                        workspace.resolve(),
                        output.resolve(),
                        package.resolve(),
                        "a" * 40,
                        DockerSandboxConfig(),
                    )
            stage.assert_not_called()

    def test_parent_rejects_symlinked_workspace_ancestor_before_spawn(self):
        inputs = self._input_files()
        real_parent = self.root / "real-parent"
        real_parent.mkdir()
        linked_parent = self.root / "linked-parent"
        try:
            linked_parent.symlink_to(real_parent, target_is_directory=True)
        except OSError:
            self.skipTest("local filesystem does not permit directory symlinks")
        with patch("egv.evaluation.learnable_gate._run_spawn_stage") as stage:
            with self.assertRaisesRegex(EGVError, "symlink|reparse"):
                run_gate_a_corpus_validation(
                    inputs["seed"],
                    inputs["controller-private"],
                    inputs["controller-public"],
                    inputs["freeze-private"],
                    inputs["freeze-public"],
                    linked_parent / "workspace",
                    (self.root / "public.json").resolve(),
                    Path.cwd().resolve(),
                    "a" * 40,
                    DockerSandboxConfig(),
                )
        stage.assert_not_called()


class _FakeTimedOutProcess:
    exitcode = None

    def __init__(self):
        self.terminated = False
        self.closed = False

    def start(self):
        return None

    def join(self, _timeout):
        return None

    def is_alive(self):
        return not self.terminated

    def terminate(self):
        self.terminated = True

    def close(self):
        self.closed = True


class _FakeSpawnContext:
    def __init__(self, process):
        self.process = process

    def Process(self, **_kwargs):
        return self.process


class _FakeFailedProcess:
    exitcode = 17

    def __init__(self):
        self.closed = False

    def start(self):
        return None

    def join(self, _timeout):
        return None

    def is_alive(self):
        return False

    def close(self):
        self.closed = True


class GateATimeoutTests(unittest.TestCase):
    def test_real_spawn_returns_one_bounded_canonical_result_without_pickle(self):
        result = _run_spawn_stage(
            "safe-result-probe",
            _send_canonical_result_for_test,
            ({"schema_version": "safe-result-v1", "accepted": True},),
            timeout_seconds=30,
            expect_result=True,
        )
        self.assertEqual(result, {"schema_version": "safe-result-v1", "accepted": True})

    def test_spawn_ignores_repository_style_unchecked_hash_bytecode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            module_name = "poisoned_gate_probe"
            source = root / (module_name + ".py")
            source.write_text("VALUE = 'evil'\n", encoding="utf-8")
            py_compile.compile(
                str(source),
                doraise=True,
                invalidation_mode=py_compile.PycInvalidationMode.UNCHECKED_HASH,
            )
            source.write_text("VALUE = 'safe'\n", encoding="utf-8")
            output = root / "observed.txt"
            cache = root / "fresh-child-cache"
            cache.mkdir()
            _run_spawn_stage(
                "source-only-probe",
                _source_only_import_probe_for_test,
                (str(root), module_name, str(output)),
                timeout_seconds=30,
                bytecode_cache=cache,
                expected_bytecode_cache_identity=_path_identity(
                    cache,
                    "test bytecode cache",
                    directory=True,
                ),
            )
            self.assertEqual(output.read_text(encoding="utf-8"), "safe")
            self.assertEqual(list(cache.iterdir()), [])

    def test_timeout_terminates_child_and_never_returns_success(self):
        process = _FakeTimedOutProcess()
        with patch("egv.evaluation.learnable_gate.mp.get_context", return_value=_FakeSpawnContext(process)):
            with self.assertRaisesRegex(EGVError, "exceeded"):
                _run_spawn_stage("evaluator", lambda: None, (), timeout_seconds=0.01)
        self.assertTrue(process.terminated)
        self.assertTrue(process.closed)

    def test_nonzero_child_is_a_hard_failure(self):
        process = _FakeFailedProcess()
        with patch(
            "egv.evaluation.learnable_gate.mp.get_context",
            return_value=_FakeSpawnContext(process),
        ):
            with self.assertRaisesRegex(EGVError, "stage failed"):
                _run_spawn_stage("issuer", lambda: None, (), timeout_seconds=1)
        self.assertTrue(process.closed)


if __name__ == "__main__":
    unittest.main()
