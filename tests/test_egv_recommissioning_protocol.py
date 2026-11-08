from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

from egv.canonical import canonical_json, digest_bytes, digest_for
from egv.receipts import ReceiptSigner
from egv.recommissioning.client import (
    ModelServiceProfile,
    PublicGenerationReceipt,
    ServiceAttestation,
    TrustedLaunchReceipt,
    TypedCandidateResponse,
)
from egv.recommissioning.protocol import (
    PilotCoordinate,
    PromptEnvelope,
    RecommissioningProtocolError,
    SignedEvaluatorReceipt,
    TokenBudget,
    VisibleTaskPackage,
    lineage_feedback,
    render_prompt,
)


def _model_profile():
    return ModelServiceProfile(
        model_bundle="LIGHTNING",
        model_id="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
        model_revision="cc84af2fe71647d87f4486c064f320e1e7535243",
        engine_image_digest="sha256:" + "a" * 64,
        max_model_len=8192,
        precision="NVFP4",
        kv_cache_dtype="fp8",
        tokenizer_digest=digest_for("pinned-tokenizer"),
        tokenizer_implementation_digest=digest_for("pinned-tokenizer-implementation"),
        chat_template_digest=digest_for("pinned-chat-template"),
        weights_digest=digest_for("pinned-weights"),
        license_digest=digest_for("pinned-license"),
    )


def _profile_digest():
    return _model_profile().digest


def _budget(seed=7):
    return TokenBudget(
        max_prompt_tokens=2048,
        max_completion_tokens=256,
        max_total_tokens=2304,
        max_coordinate_tokens=20000,
        hard_timeout_ms=5000,
        seed=seed,
    )


def _coordinate(
    protocol_id="P1",
    *,
    task_id="egv-learnable-pure-function-train-001-v2",
    seed=7,
    evaluator_profile_digest=None,
):
    return PilotCoordinate(
        campaign_id="egv-nemotron-stage-a-v2",
        model_bundle="LIGHTNING",
        protocol_id=protocol_id,
        task_id=task_id,
        task_family="PURE_FUNCTION",
        seed=seed,
        generation_receipt_key_id=TestSigner().key_id,
        evaluator_key_id=EvaluatorSigner().key_id,
        evaluator_profile_digest=evaluator_profile_digest or digest_for("frozen-evaluator-profile"),
    )


def _task(coordinate):
    return VisibleTaskPackage.for_test(
        task_id=coordinate.task_id,
        task_family=coordinate.task_family,
        public_spec={
            "schema_version": "egv-learnable-public-spec-v2",
            "family_id": coordinate.task_family,
            "task_scope": coordinate.task_id,
            "operation": "double-plus-bias",
            "parameters": {"bias": 5},
        },
        initial_source="def main(value):\n    return value\n",
    )


def _candidate(source, evidence_ids=()):
    return TypedCandidateResponse.parse(
        {
            "action": "COMMIT",
            "source": source,
            "summary": "Bounded public correction.",
            "evidence_ids": list(evidence_ids),
        },
        visible_evidence_ids=evidence_ids,
        public_locus="module:main",
    )


class TestSigner:
    def __init__(self):
        self._signer = ReceiptSigner(bytes.fromhex("11" * 32))

    @property
    def key_id(self):
        return self._signer.key_id

    @property
    def public_key(self):
        return self._signer.public_key

    def sign_bytes(self, payload):
        return self._signer.sign_bytes(payload)


class EvaluatorSigner:
    def __init__(self):
        self._signer = ReceiptSigner(bytes.fromhex("22" * 32))

    @property
    def key_id(self):
        return self._signer.key_id

    @property
    def public_key(self):
        return self._signer.public_key

    def sign_bytes(self, payload):
        return self._signer.sign_bytes(payload)


_TRUST_CHAIN = None


def _trust_chain():
    global _TRUST_CHAIN
    if _TRUST_CHAIN is None:
        profile = _model_profile()
        trusted = TrustedLaunchReceipt.for_test(profile, generation_signer=TestSigner())
        attestation = ServiceAttestation._from_observation(
            trusted,
            profile=profile,
            service_observation=trusted.build_test_service_observation(profile),
            health_body=b"",
            model_list_body=canonical_json({"data": [{"id": profile.model_id}]}).encode("utf-8"),
        )
        _TRUST_CHAIN = (trusted, attestation)
    return _TRUST_CHAIN


def _generation_receipt(envelope, candidate, budget=None, **updates):
    budget = budget or _budget(envelope.seed)
    digest = digest_for
    trusted, attestation = _trust_chain()
    core = {
        "schema_version": "egv-recommissioning-public-generation-receipt-v4",
        "coordinate_id": envelope.coordinate_id,
        "model_bundle": envelope.model_bundle,
        "task_ref": "task_" + digest_for({"task_package_digest": envelope.task_package_digest}),
        "public_locus": envelope.public_locus,
        "task_adapter_kind": envelope.task_adapter_kind,
        "task_package_digest": envelope.task_package_digest,
        "model_profile_digest": envelope.model_profile_digest,
        "budget_digest": budget.digest,
        "seed": budget.seed,
        "attempt_index": envelope.attempt_index,
        "prompt_digest": envelope.prompt_digest,
        "wire_request_digest": digest("wire-request-{}".format(envelope.attempt_index)),
        "request_digest": digest("generation-request-{}".format(envelope.attempt_index)),
        "service_attestation_digest": attestation.attestation_digest,
        "service_observation_digest": attestation.service_observation["observation_digest"],
        "controller_key_id": trusted.controller_key_id,
        "service_key_id": trusted.service_key_id,
        "generation_receipt_key_id": trusted.generation_receipt_key_id,
        "runtime_config_digest": attestation.runtime_config_digest,
        "launch_receipt_digest": attestation.launch_receipt_digest,
        "tokenization_preflight_digest": digest("tokenization-{}".format(envelope.attempt_index)),
        "rendered_prompt_digest": digest("rendered-prompt-{}".format(envelope.attempt_index)),
        "response_digest": digest("response-{}".format(envelope.attempt_index)),
        "candidate_digest": digest_for(candidate.public_redacted_record()),
        "source_digest": candidate.source_digest,
        "finish_reason": "stop",
        "prompt_tokens": 100,
        "completion_tokens": 25,
        "total_tokens": 125,
        "max_prompt_tokens": budget.max_prompt_tokens,
        "max_completion_tokens": budget.max_completion_tokens,
        "max_total_tokens": budget.max_total_tokens,
        "max_coordinate_tokens": budget.max_coordinate_tokens,
        "hard_timeout_ms": budget.hard_timeout_ms,
        "total_elapsed_ms": 12,
        "coordinate_debit_digest": digest("debit-{}".format(envelope.attempt_index)),
        "coordinate_reserved_tokens": 356,
        "coordinate_cumulative_reserved_tokens": 356 * envelope.attempt_index,
        "coordinate_debit_count": envelope.attempt_index,
        "coordinate_state_digest": digest("coordinate-state-{}".format(envelope.attempt_index)),
    }
    core.update(updates)
    signer = TestSigner()
    return PublicGenerationReceipt.build(
        core,
        signer,
        pinned_public_key=signer.public_key,
    ).public_record()


def _evaluator_receipt(envelope, candidate, generation_receipt, *, diagnostic, public_summary):
    signer = EvaluatorSigner()
    return SignedEvaluatorReceipt.build(
        coordinate_id=envelope.coordinate_id,
        task_package_digest=envelope.task_package_digest,
        model_profile_digest=envelope.model_profile_digest,
        evaluator_profile_digest=envelope.evaluator_profile_digest,
        attempt_index=envelope.attempt_index,
        prior_prompt_digest=envelope.prompt_digest,
        generation_receipt_digest=generation_receipt["receipt_digest"],
        candidate_digest=digest_for(candidate.public_redacted_record()),
        source_digest=candidate.source_digest,
        diagnostic=diagnostic,
        public_summary=public_summary,
        signer=signer,
    ).public_record()


def _feedback(coordinate, task, envelope, candidate, previous=None):
    generation_receipt = _generation_receipt(envelope, candidate)
    summary = "The public functional contract still disagrees."
    trusted, attestation = _trust_chain()
    return lineage_feedback(
        coordinate=coordinate,
        task=task,
        model_profile_digest=_profile_digest(),
        prior_prompt=envelope,
        generation_receipt=generation_receipt,
        generation_receipt_public_key=TestSigner().public_key,
        service_attestation=attestation.public_record(),
        controller_public_key=trusted.controller_public_key_raw,
        evaluator_receipt=_evaluator_receipt(
            envelope,
            candidate,
            generation_receipt,
            diagnostic="WRONG_OUTPUT",
            public_summary=summary,
        ),
        evaluator_receipt_public_key=EvaluatorSigner().public_key,
        candidate=candidate,
        diagnostic="WRONG_OUTPUT",
        public_summary=summary,
        previous_feedback=previous,
    )


class RecommissioningProtocolTests(unittest.TestCase):
    def test_p0_sees_only_latest_feedback_and_p1_sees_full_digest_lineage(self):
        p1 = _coordinate("P1")
        task = _task(p1)
        first_prompt = render_prompt(
            p1,
            task,
            model_profile_digest=_profile_digest(),
            budget=_budget(),
            attempt_index=1,
        )
        first_candidate = _candidate("def main(value):\n    return value\n")
        first = _feedback(p1, task, first_prompt, first_candidate)
        second_prompt = render_prompt(
            p1,
            task,
            model_profile_digest=_profile_digest(),
            budget=_budget(),
            attempt_index=2,
            feedback=(first,),
        )
        second_candidate = _candidate(
            "def main(value):\n    return value.get('payload')\n",
            evidence_ids=(first.evidence_id,),
        )
        second = _feedback(p1, task, second_prompt, second_candidate, first)
        third = render_prompt(
            p1,
            task,
            model_profile_digest=_profile_digest(),
            budget=_budget(),
            attempt_index=3,
            feedback=(first, second),
        )
        self.assertEqual(third.visible_evidence_ids, tuple(sorted((first.evidence_id, second.evidence_id))))
        self.assertIn(first.feedback_digest, third.messages[1].content)
        visible_prompt = json.loads(third.messages[1].content.split("\n", 1)[1])
        self.assertEqual(visible_prompt["visible_feedback"][1]["source"], second.source)

        p0 = _coordinate("P0")
        p0_task = _task(p0)
        p0_first_prompt = render_prompt(
            p0,
            p0_task,
            model_profile_digest=_profile_digest(),
            budget=_budget(),
            attempt_index=1,
        )
        p0_first = _feedback(p0, p0_task, p0_first_prompt, first_candidate)
        p0_second_prompt = render_prompt(
            p0,
            p0_task,
            model_profile_digest=_profile_digest(),
            budget=_budget(),
            attempt_index=2,
            feedback=(p0_first,),
        )
        p0_second_candidate = _candidate(
            "def main(value):\n    return value.get('payload')\n",
            evidence_ids=(p0_first.evidence_id,),
        )
        p0_second = _feedback(p0, p0_task, p0_second_prompt, p0_second_candidate, p0_first)
        p0_third = render_prompt(
            p0,
            p0_task,
            model_profile_digest=_profile_digest(),
            budget=_budget(),
            attempt_index=3,
            feedback=(p0_first, p0_second),
        )
        self.assertEqual(p0_third.visible_evidence_ids, (p0_second.evidence_id,))
        self.assertNotIn(p0_first.feedback_digest, p0_third.messages[1].content)

    def test_feedback_binds_prompt_receipts_candidate_task_and_profile(self):
        coordinate = _coordinate()
        task = _task(coordinate)
        prompt = render_prompt(
            coordinate,
            task,
            model_profile_digest=_profile_digest(),
            budget=_budget(),
            attempt_index=1,
        )
        candidate = _candidate("def main(value):\n    return value\n")
        receipt = _generation_receipt(prompt, candidate)
        tampered = dict(receipt)
        tampered["source_digest"] = digest_for("substituted-source")
        evaluator = _evaluator_receipt(
            prompt,
            candidate,
            receipt,
            diagnostic="WRONG_OUTPUT",
            public_summary="Mismatch.",
        )
        with self.assertRaises(Exception):
            lineage_feedback(
                coordinate=coordinate,
                task=task,
                model_profile_digest=_profile_digest(),
                prior_prompt=prompt,
                generation_receipt=tampered,
                generation_receipt_public_key=TestSigner().public_key,
                service_attestation=_trust_chain()[1].public_record(),
                controller_public_key=_trust_chain()[0].controller_public_key_raw,
                evaluator_receipt=evaluator,
                evaluator_receipt_public_key=EvaluatorSigner().public_key,
                candidate=candidate,
                diagnostic="WRONG_OUTPUT",
                public_summary="Mismatch.",
            )
        other_task = _task(_coordinate(task_id="egv-learnable-pure-function-train-002-v2"))
        with self.assertRaisesRegex(RecommissioningProtocolError, "task-package"):
            prompt.validate(task=other_task)

    def test_evaluator_profile_is_bound_through_coordinate_prompt_receipt_and_feedback(self):
        coordinate = _coordinate()
        task = _task(coordinate)
        prompt = render_prompt(
            coordinate,
            task,
            model_profile_digest=_profile_digest(),
            budget=_budget(),
            attempt_index=1,
        )
        candidate = _candidate("def main(value):\n    return value\n")
        generation = _generation_receipt(prompt, candidate)
        feedback = _feedback(coordinate, task, prompt, candidate)
        self.assertEqual(prompt.evaluator_profile_digest, coordinate.evaluator_profile_digest)
        self.assertEqual(feedback.evaluator_profile_digest, coordinate.evaluator_profile_digest)
        wrong_evaluator = SignedEvaluatorReceipt.build(
            coordinate_id=coordinate.coordinate_id,
            task_package_digest=task.task_package_digest,
            model_profile_digest=_profile_digest(),
            evaluator_profile_digest=digest_for("substituted-evaluator-profile"),
            attempt_index=1,
            prior_prompt_digest=prompt.prompt_digest,
            generation_receipt_digest=generation["receipt_digest"],
            candidate_digest=digest_for(candidate.public_redacted_record()),
            source_digest=candidate.source_digest,
            diagnostic="WRONG_OUTPUT",
            public_summary="Mismatch.",
            signer=EvaluatorSigner(),
        ).public_record()
        with self.assertRaisesRegex(RecommissioningProtocolError, "evaluator_profile_digest binding"):
            lineage_feedback(
                coordinate=coordinate,
                task=task,
                model_profile_digest=_profile_digest(),
                prior_prompt=prompt,
                generation_receipt=generation,
                generation_receipt_public_key=TestSigner().public_key,
                service_attestation=_trust_chain()[1].public_record(),
                controller_public_key=_trust_chain()[0].controller_public_key_raw,
                evaluator_receipt=wrong_evaluator,
                evaluator_receipt_public_key=EvaluatorSigner().public_key,
                candidate=candidate,
                diagnostic="WRONG_OUTPUT",
                public_summary="Mismatch.",
            )

    def test_lineage_requires_controller_root_and_exact_service_bindings(self):
        coordinate = _coordinate()
        task = _task(coordinate)
        prompt = render_prompt(
            coordinate,
            task,
            model_profile_digest=_profile_digest(),
            budget=_budget(),
            attempt_index=1,
        )
        candidate = _candidate("def main(value):\n    return value\n")
        correct_generation = _generation_receipt(prompt, candidate)
        summary = "Mismatch."
        correct_evaluator = _evaluator_receipt(
            prompt,
            candidate,
            correct_generation,
            diagnostic="WRONG_OUTPUT",
            public_summary=summary,
        )
        with self.assertRaisesRegex(RecommissioningProtocolError, "controller-rooted"):
            lineage_feedback(
                coordinate=coordinate,
                task=task,
                model_profile_digest=_profile_digest(),
                prior_prompt=prompt,
                generation_receipt=correct_generation,
                generation_receipt_public_key=TestSigner().public_key,
                service_attestation=_trust_chain()[1].public_record(),
                controller_public_key=ReceiptSigner.generate().public_key,
                evaluator_receipt=correct_evaluator,
                evaluator_receipt_public_key=EvaluatorSigner().public_key,
                candidate=candidate,
                diagnostic="WRONG_OUTPUT",
                public_summary=summary,
            )
        substitutions = {
            "service_attestation_digest": digest_for("foreign-attestation"),
            "service_observation_digest": digest_for("foreign-observation"),
            "runtime_config_digest": digest_for("foreign-runtime"),
            "launch_receipt_digest": digest_for("foreign-launch"),
            "controller_key_id": "key_" + "0" * 64,
            "service_key_id": "key_" + "1" * 64,
        }
        for field_name, foreign_value in substitutions.items():
            with self.subTest(field_name=field_name):
                generation = _generation_receipt(prompt, candidate, **{field_name: foreign_value})
                evaluator = _evaluator_receipt(
                    prompt,
                    candidate,
                    generation,
                    diagnostic="WRONG_OUTPUT",
                    public_summary=summary,
                )
                with self.assertRaisesRegex(RecommissioningProtocolError, field_name + " binding"):
                    lineage_feedback(
                        coordinate=coordinate,
                        task=task,
                        model_profile_digest=_profile_digest(),
                        prior_prompt=prompt,
                        generation_receipt=generation,
                        generation_receipt_public_key=TestSigner().public_key,
                        service_attestation=_trust_chain()[1].public_record(),
                        controller_public_key=_trust_chain()[0].controller_public_key_raw,
                        evaluator_receipt=evaluator,
                        evaluator_receipt_public_key=EvaluatorSigner().public_key,
                        candidate=candidate,
                        diagnostic="WRONG_OUTPUT",
                        public_summary=summary,
                    )

    def test_feedback_rejects_resigned_generation_receipt_with_inconsistent_budget(self):
        coordinate = _coordinate(seed=7)
        task = _task(coordinate)
        prompt = render_prompt(
            coordinate,
            task,
            model_profile_digest=_profile_digest(),
            budget=_budget(seed=7),
            attempt_index=1,
        )
        candidate = _candidate("def main(value):\n    return value\n")
        inconsistent = _generation_receipt(prompt, candidate, budget=_budget(seed=8))
        summary = "Mismatch."
        evaluator = _evaluator_receipt(
            prompt,
            candidate,
            inconsistent,
            diagnostic="WRONG_OUTPUT",
            public_summary=summary,
        )
        with self.assertRaisesRegex(RecommissioningProtocolError, "budget_digest binding"):
            lineage_feedback(
                coordinate=coordinate,
                task=task,
                model_profile_digest=_profile_digest(),
                prior_prompt=prompt,
                generation_receipt=inconsistent,
                generation_receipt_public_key=TestSigner().public_key,
                service_attestation=_trust_chain()[1].public_record(),
                controller_public_key=_trust_chain()[0].controller_public_key_raw,
                evaluator_receipt=evaluator,
                evaluator_receipt_public_key=EvaluatorSigner().public_key,
                candidate=candidate,
                diagnostic="WRONG_OUTPUT",
                public_summary=summary,
            )

    def test_evaluator_receipt_signature_and_all_feedback_bindings_fail_closed(self):
        coordinate = _coordinate()
        task = _task(coordinate)
        prompt = render_prompt(
            coordinate,
            task,
            model_profile_digest=_profile_digest(),
            budget=_budget(),
            attempt_index=1,
        )
        candidate = _candidate("def main(value):\n    return value\n")
        generation = _generation_receipt(prompt, candidate)
        summary = "Mismatch."
        unrelated = _evaluator_receipt(
            prompt,
            candidate,
            generation,
            diagnostic="PASS",
            public_summary=summary,
        )
        with self.assertRaisesRegex(RecommissioningProtocolError, "diagnostic binding"):
            lineage_feedback(
                coordinate=coordinate,
                task=task,
                model_profile_digest=_profile_digest(),
                prior_prompt=prompt,
                generation_receipt=generation,
                generation_receipt_public_key=TestSigner().public_key,
                service_attestation=_trust_chain()[1].public_record(),
                controller_public_key=_trust_chain()[0].controller_public_key_raw,
                evaluator_receipt=unrelated,
                evaluator_receipt_public_key=EvaluatorSigner().public_key,
                candidate=candidate,
                diagnostic="WRONG_OUTPUT",
                public_summary=summary,
            )
        forged = dict(unrelated)
        attacker = ReceiptSigner.generate()
        signable = {key: value for key, value in forged.items() if key != "signature"}
        forged["signature"] = attacker.sign_bytes(canonical_json(signable).encode("utf-8"))
        with self.assertRaisesRegex(RecommissioningProtocolError, "signature is invalid"):
            SignedEvaluatorReceipt.validate_record(forged, public_key=EvaluatorSigner().public_key)
        attacker_evaluator = ReceiptSigner.generate()
        attacker_receipt = SignedEvaluatorReceipt.build(
            coordinate_id=coordinate.coordinate_id,
            task_package_digest=task.task_package_digest,
            model_profile_digest=_profile_digest(),
            evaluator_profile_digest=coordinate.evaluator_profile_digest,
            attempt_index=1,
            prior_prompt_digest=prompt.prompt_digest,
            generation_receipt_digest=generation["receipt_digest"],
            candidate_digest=digest_for(candidate.public_redacted_record()),
            source_digest=candidate.source_digest,
            diagnostic="WRONG_OUTPUT",
            public_summary=summary,
            signer=attacker_evaluator,
        ).public_record()
        with self.assertRaisesRegex(RecommissioningProtocolError, "evaluator receipt key differs"):
            lineage_feedback(
                coordinate=coordinate,
                task=task,
                model_profile_digest=_profile_digest(),
                prior_prompt=prompt,
                generation_receipt=generation,
                generation_receipt_public_key=TestSigner().public_key,
                service_attestation=_trust_chain()[1].public_record(),
                controller_public_key=_trust_chain()[0].controller_public_key_raw,
                evaluator_receipt=attacker_receipt,
                evaluator_receipt_public_key=attacker_evaluator.public_key,
                candidate=candidate,
                diagnostic="WRONG_OUTPUT",
                public_summary=summary,
            )

    def test_sensitive_evaluator_summary_is_rejected_before_projection(self):
        coordinate = _coordinate()
        task = _task(coordinate)
        prompt = render_prompt(
            coordinate,
            task,
            model_profile_digest=_profile_digest(),
            budget=_budget(),
            attempt_index=1,
        )
        candidate = _candidate("def main(value):\n    return value\n")
        generation = _generation_receipt(prompt, candidate)
        with self.assertRaisesRegex(RecommissioningProtocolError, "prohibited private material"):
            SignedEvaluatorReceipt.build(
                coordinate_id=coordinate.coordinate_id,
                task_package_digest=task.task_package_digest,
                model_profile_digest=_profile_digest(),
                evaluator_profile_digest=coordinate.evaluator_profile_digest,
                attempt_index=1,
                prior_prompt_digest=prompt.prompt_digest,
                generation_receipt_digest=generation["receipt_digest"],
                candidate_digest=digest_for(candidate.public_redacted_record()),
                source_digest=candidate.source_digest,
                diagnostic="WRONG_OUTPUT",
                public_summary="credential sk-secret at C:/private/host.txt",
                signer=EvaluatorSigner(),
            )

    def test_feedback_full_digest_and_redaction_detect_tampering(self):
        coordinate = _coordinate()
        task = _task(coordinate)
        prompt = render_prompt(
            coordinate,
            task,
            model_profile_digest=_profile_digest(),
            budget=_budget(),
            attempt_index=1,
        )
        feedback = _feedback(coordinate, task, prompt, _candidate("def main(value):\n    return value\n"))
        self.assertIn("source", feedback.model_visible_record())
        public = feedback.public_redacted_record()
        self.assertNotIn("source", public)
        self.assertNotIn("public_summary", public)
        self.assertEqual(public["public_summary_digest"], digest_bytes(feedback.public_summary.encode("utf-8")))
        object.__setattr__(feedback, "public_summary", feedback.public_summary + " tamper")
        with self.assertRaisesRegex(RecommissioningProtocolError, "full lineage"):
            feedback.validate()

    def test_feedback_cannot_cross_task_seed_arm_or_model(self):
        first_coordinate = _coordinate(seed=7)
        first_task = _task(first_coordinate)
        prompt = render_prompt(
            first_coordinate,
            first_task,
            model_profile_digest=_profile_digest(),
            budget=_budget(),
            attempt_index=1,
        )
        feedback = _feedback(
            first_coordinate,
            first_task,
            prompt,
            _candidate("def main(value):\n    return value\n"),
        )
        second = _coordinate(seed=8)
        with self.assertRaisesRegex(RecommissioningProtocolError, "crossed"):
            render_prompt(
                second,
                _task(second),
                model_profile_digest=_profile_digest(),
                budget=_budget(seed=8),
                attempt_index=2,
                feedback=(feedback,),
            )
        with self.assertRaisesRegex(RecommissioningProtocolError, "crossed"):
            render_prompt(
                first_coordinate,
                first_task,
                model_profile_digest=digest_for("other-profile"),
                budget=_budget(),
                attempt_index=2,
                feedback=(feedback,),
            )

    def test_prompt_envelope_revalidates_all_bindings_and_messages(self):
        coordinate = _coordinate()
        task = _task(coordinate)
        prompt = render_prompt(
            coordinate,
            task,
            model_profile_digest=_profile_digest(),
            budget=_budget(),
            attempt_index=1,
        )
        prompt.validate(
            coordinate=coordinate,
            task=task,
            model_profile_digest=_profile_digest(),
            budget=_budget(),
        )
        object.__setattr__(prompt, "task_package_digest", digest_for("tampered-task"))
        with self.assertRaisesRegex(RecommissioningProtocolError, "prompt digest"):
            prompt.validate()

    def test_prompt_construction_is_closed_without_opaque_object_authority(self):
        import egv.recommissioning.protocol as protocol_module

        coordinate = _coordinate()
        task = _task(coordinate)
        budget = _budget()
        prompt = render_prompt(
            coordinate,
            task,
            model_profile_digest=_profile_digest(),
            budget=budget,
            attempt_index=1,
        )
        rebuilt = PromptEnvelope._build(
            coordinate=coordinate,
            task=task,
            model_profile_digest=_profile_digest(),
            budget=budget,
            attempt_index=1,
            visible_evidence_ids=prompt.visible_evidence_ids,
            messages=prompt.messages,
        )
        self.assertEqual(rebuilt.prompt_digest, prompt.prompt_digest)
        opaque_authority_names = [
            name for name in vars(protocol_module) if "CAPABILITY" in name or "CONSTRUCTION_TOKEN" in name
        ]
        self.assertEqual(opaque_authority_names, [])
        self.assertNotIn("_prompt_capability", VisibleTaskPackage.__dataclass_fields__)
        self.assertNotIn("_prompt_capability", PromptEnvelope.__dataclass_fields__)
        forged_values = {field_name: getattr(task, field_name) for field_name in task.__dataclass_fields__}
        forged_values["adapter_kind"] = "LEARNABLE_V2"
        with self.assertRaisesRegex(RecommissioningProtocolError, "full corpus-manifest proof"):
            forged_task = VisibleTaskPackage._from_values(**forged_values)
            PromptEnvelope._build(
                coordinate=coordinate,
                task=forged_task,
                model_profile_digest=_profile_digest(),
                budget=budget,
                attempt_index=1,
                visible_evidence_ids=prompt.visible_evidence_ids,
                messages=prompt.messages,
            )

    def test_visible_package_rejects_private_vocabulary_and_deep_json(self):
        coordinate = _coordinate()
        with self.assertRaisesRegex(RecommissioningProtocolError, "evaluator-private"):
            VisibleTaskPackage.for_test(
                task_id=coordinate.task_id,
                task_family=coordinate.task_family,
                public_spec={"rule": "use oracle_token"},
                initial_source="def main(value):\n    return value\n",
            )
        deep = "leaf"
        for _ in range(18):
            deep = {"child": deep}
        with self.assertRaisesRegex(RecommissioningProtocolError, "depth"):
            VisibleTaskPackage.for_test(
                task_id=coordinate.task_id,
                task_family=coordinate.task_family,
                public_spec=deep,
                initial_source="def main(value):\n    return value\n",
            )

    def test_budget_excludes_bool_and_binds_seed(self):
        with self.assertRaisesRegex(RecommissioningProtocolError, "seed"):
            TokenBudget(1, 1, 2, 2, 1, True).validate()
        coordinate = _coordinate(seed=7)
        with self.assertRaisesRegex(RecommissioningProtocolError, "budget seed"):
            render_prompt(
                coordinate,
                _task(coordinate),
                model_profile_digest=_profile_digest(),
                budget=_budget(seed=8),
                attempt_index=1,
            )

    @unittest.skipUnless(
        importlib.util.find_spec("egv.evaluation.learnable") is not None,
        "requires corpus validator commit 4c3e74d6",
    )
    def test_learnable_v2_adapter_retains_original_package_record_corpus_and_structured_spec(self):
        from egv.evaluation.learnable import LearnableCorpusV2, validate_model_visible_package

        with tempfile.TemporaryDirectory() as directory:
            seed_path = Path(directory) / "learnable-seed.bin"
            seed_path.write_bytes(bytes(range(32)))
            corpus = LearnableCorpusV2.generate(secret_seed_file=seed_path)
            task_id = corpus.stage_a_tasks()[0].task_id
            package = validate_model_visible_package(corpus.model_visible_package(task_id))
            manifest = corpus.public_manifest()
            corpus_digest = manifest["manifest_digest"]
            adapted = VisibleTaskPackage.from_learnable_v2(package, manifest)
            with self.assertRaisesRegex(RecommissioningProtocolError, "corpus manifest"):
                VisibleTaskPackage.from_learnable_v2(package, corpus_digest)
            wrong_manifest = json.loads(canonical_json(manifest))
            wrong_task = wrong_manifest["tasks"][0]
            wrong_task["starter_source"] += "\n# substituted public starter\n"
            wrong_task["starter_source_digest"] = digest_bytes(wrong_task["starter_source"].encode("utf-8"))
            wrong_task["record_digest"] = digest_for(
                {key: value for key, value in wrong_task.items() if key != "record_digest"}
            )
            wrong_manifest["manifest_digest"] = digest_for(
                {key: value for key, value in wrong_manifest.items() if key != "manifest_digest"}
            )
            with self.assertRaisesRegex(RecommissioningProtocolError, "failed closed validation"):
                VisibleTaskPackage.from_learnable_v2(package, wrong_manifest)
            wrong_membership = {field_name: getattr(adapted, field_name) for field_name in adapted.__dataclass_fields__}
            wrong_membership["corpus_task_index"] = 1
            wrong_membership["corpus_membership_digest"] = digest_for(
                {
                    "schema_version": "egv-learnable-corpus-membership-v1",
                    "corpus_manifest_digest": adapted.corpus_manifest_digest,
                    "corpus_task_index": 1,
                    "task_record_digest": adapted.original_task_record_digest,
                    "package_digest": adapted.original_package_digest,
                }
            )
            with self.assertRaisesRegex(RecommissioningProtocolError, "corpus-manifest member"):
                VisibleTaskPackage._from_values(**wrong_membership)
        task = package["task"]
        record = adapted.model_visible_record()
        self.assertEqual(record["original_package_digest"], package["package_digest"])
        self.assertEqual(record["original_task_record_digest"], task["record_digest"])
        self.assertEqual(record["corpus_manifest_digest"], corpus_digest)
        self.assertEqual(record["corpus_task_index"], 0)
        self.assertEqual(len(record["corpus_membership_digest"]), 64)
        self.assertEqual(record["public_spec"], task["public_spec"])
        self.assertEqual(adapted.adapter_kind, "LEARNABLE_V2")

    def test_attempt_one_is_stable_and_contains_no_private_answers(self):
        coordinate = _coordinate()
        first = render_prompt(
            coordinate,
            _task(coordinate),
            model_profile_digest=_profile_digest(),
            budget=_budget(),
            attempt_index=1,
        )
        second = render_prompt(
            coordinate,
            _task(coordinate),
            model_profile_digest=_profile_digest(),
            budget=_budget(),
            attempt_index=1,
        )
        self.assertEqual(first.prompt_digest, second.prompt_digest)
        self.assertEqual(first.visible_evidence_ids, ())
        self.assertNotIn("expected_output", first.messages[1].content)


if __name__ == "__main__":
    unittest.main()
