from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from egv.canonical import GENESIS_HASH, content_id, digest_bytes, digest_for
from egv.campaign.commissioning import (
    GENERATION_REQUEST_COUNT,
    PRIVATE_DEV_TASK_COUNT,
    TRAIN_TASK_COUNT,
    CommissioningPreparationError,
    prepare_commissioning,
)
from egv.campaign.trajectories import (
    CommissioningTrajectoryError,
    GenerationRequest,
    GenerationResponse,
    reconcile_responses,
    validate_accepted_response,
)
from egv.evaluation.dataset import EvaluationCorpus, FAMILY_SPECS
from egv.evaluation.authority import AuthorityPolicy
from egv.receipts import ReceiptSigner, receipt_hash
from egv.variation.arms import arm_policy
from egv.variation.loop import VARIATION_PROTOCOL_DIGEST
from egv.variation.generator import model_generation_profile_digest


def _corpus(root: Path) -> EvaluationCorpus:
    seed = root / "evaluator-seed.bin"
    seed.write_bytes(bytes(range(32)))
    return EvaluationCorpus.generate(secret_seed_file=seed)


def _signed_receipts(request: GenerationRequest, signer: ReceiptSigner, artifact_digest: str, **overrides):
    candidate_id = overrides.pop("candidate_id", content_id("candidate", {"request": request.request_id}))
    common = {
        "campaign_id": request.campaign_id,
        "run_id": request.run_id,
        "task_id": request.task_id,
        "candidate_id": candidate_id,
        "candidate_artifact_digest": artifact_digest,
        "protocol_digest": request.variation_protocol_digest,
        "policy_digest": AuthorityPolicy.candidate_execution().digest,
        "arm_policy_digest": request.arm_policy_digest,
        "evaluator_digest": digest_for("commissioning-evaluator"),
        "exit_status_class": "SUCCESS",
    }
    common.update(overrides)
    authority = signer.sign_receipt(
        {**common, "receipt_type": "AUTHORITY", "request_id": "authority-" + request.request_id, "decision": "ALLOW"},
        sequence=1,
        previous_receipt_hash=GENESIS_HASH,
        idempotency_key="authority-" + request.request_id,
    )
    verdict = signer.sign_receipt(
        {
            **common,
            "receipt_type": "VERDICT",
            "request_id": "verdict-" + request.request_id,
            "decision": "PASS",
            "diagnostic_enum": "PASS",
        },
        sequence=2,
        previous_receipt_hash=receipt_hash(authority),
        idempotency_key="verdict-" + request.request_id,
    )
    effect = signer.sign_receipt(
        {**common, "receipt_type": "EFFECT", "request_id": "effect-" + request.request_id, "decision": "ALLOW"},
        sequence=3,
        previous_receipt_hash=receipt_hash(verdict),
        idempotency_key="effect-" + request.request_id,
    )
    return candidate_id, [authority, verdict, effect]


def _response(request: GenerationRequest, signer: ReceiptSigner, source=b"def solve(value):\n    return value\n", **overrides):
    artifact_digest = digest_bytes(source)
    candidate_id, receipts = _signed_receipts(request, signer, artifact_digest)
    payload = {
        "schema_version": "egv-commissioning-generation-response-v1",
        "request_id": request.request_id,
        "candidate_id": candidate_id,
        "candidate_artifact_digest": artifact_digest,
        "model_output_digest": artifact_digest,
        "output_byte_count": len(source),
        "disposition": "PROMOTED",
        "receipts": receipts,
    }
    payload.update(overrides)
    payload["response_id"] = content_id("genresp", payload)
    return GenerationResponse.from_mapping(payload)


class CommissioningInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.corpus = _corpus(self.root)
        self.model_digest = digest_for("pinned-qwen-model-manifest")
        self.generation_profile_digest = model_generation_profile_digest(
            "source-only-v1",
            model_manifest_digest=self.model_digest,
            chat_template_digest=digest_for("pinned-chat-template"),
        )
        self.plan = prepare_commissioning(
            self.corpus,
            campaign_id="campaign-public",
            model_manifest_digest=self.model_digest,
            generation_profile_digest=self.generation_profile_digest,
        )
        self.signer = ReceiptSigner.generate()

    def authority_kwargs(self) -> dict:
        return {
            "evaluator_digest": digest_for("commissioning-evaluator"),
            "expected_first_sequence": 1,
            "expected_previous_receipt_hash": GENESIS_HASH,
        }

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_exact_20_train_8_private_dev_and_80_request_allocation(self) -> None:
        self.assertEqual(len(self.plan.train_records), TRAIN_TASK_COUNT)
        self.assertEqual(len(self.plan.private_dev_tasks), PRIVATE_DEV_TASK_COUNT)
        self.assertEqual(len(self.plan.generation_requests), GENERATION_REQUEST_COUNT)
        self.assertEqual(
            self.plan.train_manifest["family_counts"],
            {family.family_id: family.train for family in FAMILY_SPECS},
        )
        coordinates = {
            (request.task_id, request.arm_id, request.seed)
            for request in self.plan.generation_requests
        }
        self.assertEqual(len(coordinates), 80)
        self.assertEqual({request.arm_id for request in self.plan.generation_requests}, {"B", "D"})
        self.assertEqual({request.seed for request in self.plan.generation_requests}, {0, 1})

    def test_preparation_is_byte_deterministic_and_all_requests_remain_pending(self) -> None:
        second = prepare_commissioning(
            self.corpus,
            campaign_id="campaign-public",
            model_manifest_digest=self.model_digest,
            generation_profile_digest=self.generation_profile_digest,
        )
        self.assertEqual(self.plan.public_manifest(), second.public_manifest())
        self.assertEqual(self.plan.private_manifest(), second.private_manifest())
        self.assertEqual(
            [request.to_dict() for request in self.plan.generation_requests],
            [request.to_dict() for request in second.generation_requests],
        )
        self.assertTrue(all(request.status == "PENDING" for request in self.plan.generation_requests))
        self.assertEqual(self.plan.public_manifest()["accepted_response_count"], 0)
        self.assertFalse(self.plan.public_manifest()["live_model_executed"])

    def test_public_manifest_contains_no_dev_or_heldout_identity_or_private_material(self) -> None:
        public = json.dumps(self.plan.public_manifest(), sort_keys=True)
        private = json.dumps(self.plan.private_manifest(), sort_keys=True)
        dev_ids = [repo.template_id for repo in self.corpus.split("dev")]
        heldout_ids = [repo.template_id for repo in self.corpus.split("heldout")]
        for template_id in dev_ids + heldout_ids:
            self.assertNotIn(template_id, public)
        for template_id in dev_ids:
            self.assertIn(template_id, private)
        for repo in self.corpus.split("dev"):
            self.assertNotIn(str(repo.evaluator_input), public)
            self.assertNotIn(str(repo.expected_output), public)
            self.assertNotIn(str(repo.hidden_spec["hidden_rule_id"]), public)
        self.assertNotIn("hidden_spec_digest", public)
        self.assertNotIn("corrected_source_digest", public)

    def test_generation_request_rejects_dev_task_wrong_arm_seed_and_protocol(self) -> None:
        dev = self.corpus.split("dev")[0].public_manifest_record()
        with self.assertRaisesRegex(CommissioningTrajectoryError, "train"):
            GenerationRequest.build(
                campaign_id="campaign-public",
                task_record=dev,
                corpus_manifest_digest=self.corpus.manifest_digest(),
                arm_id="B",
                seed=0,
                model_manifest_digest=self.model_digest,
                variation_protocol_digest=VARIATION_PROTOCOL_DIGEST,
                generation_profile_digest=self.generation_profile_digest,
            )
        train = self.corpus.split("train")[0].public_manifest_record()
        for arm_id, seed in (("A", 0), ("B", 2)):
            with self.assertRaises(CommissioningTrajectoryError):
                GenerationRequest.build(
                    campaign_id="campaign-public",
                    task_record=train,
                    corpus_manifest_digest=self.corpus.manifest_digest(),
                    arm_id=arm_id,
                    seed=seed,
                    model_manifest_digest=self.model_digest,
                    variation_protocol_digest=VARIATION_PROTOCOL_DIGEST,
                    generation_profile_digest=self.generation_profile_digest,
                )
        with self.assertRaisesRegex(CommissioningPreparationError, "protocol"):
            prepare_commissioning(
                self.corpus,
                campaign_id="campaign-public",
                model_manifest_digest=self.model_digest,
                generation_profile_digest=self.generation_profile_digest,
                variation_protocol_digest=digest_for("substituted"),
            )

    def test_no_responses_is_pending_and_complete_claim_fails_closed(self) -> None:
        result = reconcile_responses(
            self.plan.generation_requests,
            [],
            evaluator_public_key=self.signer.public_key,
            **self.authority_kwargs(),
        )
        self.assertEqual(
            (result["status"], result["accepted_response_count"], result["missing_request_count"]),
            ("PENDING", 0, 80),
        )
        self.assertEqual(result["accepted_evidence"], [])
        with self.assertRaisesRegex(CommissioningTrajectoryError, "incomplete"):
            reconcile_responses(
                self.plan.generation_requests,
                [],
                evaluator_public_key=self.signer.public_key,
                **self.authority_kwargs(),
                require_complete=True,
            )

    def test_real_signed_allow_pass_allow_chain_is_the_only_accepted_evidence(self) -> None:
        request = self.plan.generation_requests[0]
        response = _response(request, self.signer)
        accepted = validate_accepted_response(
            request,
            response,
            evaluator_public_key=self.signer.public_key,
            **self.authority_kwargs(),
        )
        self.assertEqual((accepted["arm_id"], accepted["disposition"], accepted["diagnostic_enum"]),
                         (request.arm_id, "PROMOTED", "PASS"))
        result = reconcile_responses(
            self.plan.generation_requests,
            [response, response],
            evaluator_public_key=self.signer.public_key,
            **self.authority_kwargs(),
        )
        self.assertEqual((result["status"], result["accepted_response_count"]), ("PENDING", 1))

    def test_missing_receipts_rejected_disposition_and_empty_output_cannot_fabricate_pass(self) -> None:
        request = self.plan.generation_requests[0]
        valid = _response(request, self.signer).to_dict()
        cases = []
        missing = dict(valid)
        missing["receipts"] = []
        cases.append(missing)
        rejected = dict(valid)
        rejected["disposition"] = "REJECTED"
        cases.append(rejected)
        empty = dict(valid)
        empty["output_byte_count"] = 0
        cases.append(empty)
        for raw in cases:
            raw.pop("response_id", None)
            raw["response_id"] = content_id("genresp", raw)
            with self.assertRaises(CommissioningTrajectoryError):
                GenerationResponse.from_mapping(raw)

    def test_wrong_key_task_protocol_artifact_and_receipt_decision_fail_closed(self) -> None:
        request = self.plan.generation_requests[0]
        valid = _response(request, self.signer)
        with self.assertRaises(CommissioningTrajectoryError):
            validate_accepted_response(
                request,
                valid,
                evaluator_public_key=ReceiptSigner.generate().public_key,
                **self.authority_kwargs(),
            )
        for field, value in (
            ("task_id", "egv-other-train-1-v1"),
            ("protocol_digest", digest_for("wrong-protocol")),
            ("candidate_artifact_digest", digest_for("wrong-artifact")),
        ):
            artifact = valid.candidate_artifact_digest
            candidate_id, receipts = _signed_receipts(request, self.signer, artifact, **{field: value})
            raw = valid.to_dict()
            raw.update({"candidate_id": candidate_id, "receipts": receipts})
            raw.pop("response_id")
            raw["response_id"] = content_id("genresp", raw)
            with self.assertRaises(CommissioningTrajectoryError):
                validate_accepted_response(
                    request,
                    GenerationResponse.from_mapping(raw),
                    evaluator_public_key=self.signer.public_key,
                    **self.authority_kwargs(),
                )
        raw = valid.to_dict()
        raw["receipts"][1]["decision"] = "FAIL"
        raw.pop("response_id")
        raw["response_id"] = content_id("genresp", raw)
        with self.assertRaises(CommissioningTrajectoryError):
            validate_accepted_response(
                request,
                GenerationResponse.from_mapping(raw),
                evaluator_public_key=self.signer.public_key,
                **self.authority_kwargs(),
            )

    def test_wrong_pinned_evaluator_digest_fails_closed(self) -> None:
        request = self.plan.generation_requests[0]
        with self.assertRaisesRegex(CommissioningTrajectoryError, "generation request"):
            validate_accepted_response(
                request,
                _response(request, self.signer),
                evaluator_public_key=self.signer.public_key,
                evaluator_digest=digest_for("different-evaluator-runtime"),
                expected_first_sequence=1,
                expected_previous_receipt_hash=GENESIS_HASH,
            )

    def test_stale_or_unexpected_receipt_anchor_fails_closed(self) -> None:
        request = self.plan.generation_requests[0]
        with self.assertRaisesRegex(CommissioningTrajectoryError, "receipt is invalid"):
            validate_accepted_response(
                request,
                _response(request, self.signer),
                evaluator_public_key=self.signer.public_key,
                evaluator_digest=digest_for("commissioning-evaluator"),
                expected_first_sequence=1,
                expected_previous_receipt_hash=digest_for("stale-ledger-checkpoint"),
            )

    def test_wrong_initial_receipt_sequence_fails_closed(self) -> None:
        request = self.plan.generation_requests[0]
        with self.assertRaisesRegex(CommissioningTrajectoryError, "receipt is invalid"):
            validate_accepted_response(
                request,
                _response(request, self.signer),
                evaluator_public_key=self.signer.public_key,
                evaluator_digest=digest_for("commissioning-evaluator"),
                expected_first_sequence=2,
                expected_previous_receipt_hash=GENESIS_HASH,
            )

    def test_conflicting_response_for_same_request_is_rejected(self) -> None:
        request = self.plan.generation_requests[0]
        first = _response(request, self.signer, source=b"first-output")
        second = _response(request, self.signer, source=b"second-output")
        with self.assertRaisesRegex(CommissioningTrajectoryError, "conflicting"):
            reconcile_responses(
                self.plan.generation_requests,
                [first, second],
                evaluator_public_key=self.signer.public_key,
                **self.authority_kwargs(),
            )

    def test_mutating_nested_receipt_after_response_construction_is_detected(self) -> None:
        request = self.plan.generation_requests[0]
        response = _response(request, self.signer)
        response.receipts[1]["resource_bucket"] = "UNDER_25"
        with self.assertRaisesRegex(CommissioningTrajectoryError, "changed"):
            validate_accepted_response(
                request,
                response,
                evaluator_public_key=self.signer.public_key,
                **self.authority_kwargs(),
            )

    def test_frozen_variation_arm_policy_digests_are_bound(self) -> None:
        for request in self.plan.generation_requests:
            self.assertEqual(request.arm_policy_digest, arm_policy(request.arm_id).digest)
            self.assertEqual(request.variation_protocol_digest, VARIATION_PROTOCOL_DIGEST)


if __name__ == "__main__":
    unittest.main()
