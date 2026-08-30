from __future__ import annotations

import hashlib
import hmac
import json
import sys
import types
import unittest

from egv.canonical import canonical_json, digest_bytes, digest_for
from egv.recommissioning.client import PublicGenerationReceipt, TypedCandidateResponse
from egv.recommissioning.protocol import (
    PilotCoordinate,
    RecommissioningProtocolError,
    TokenBudget,
    VisibleTaskPackage,
    lineage_feedback,
    render_prompt,
)


def _profile_digest():
    return digest_for("exact-model-profile")


def _budget(seed=7):
    return TokenBudget(
        max_prompt_tokens=2048,
        max_completion_tokens=256,
        max_total_tokens=2304,
        max_coordinate_tokens=20000,
        hard_timeout_ms=5000,
        seed=seed,
    )


def _coordinate(protocol_id="P1", *, task_id="egv-learnable-pure-function-train-001-v2", seed=7):
    return PilotCoordinate(
        campaign_id="egv-nemotron-stage-a-v2",
        model_bundle="LIGHTNING",
        protocol_id=protocol_id,
        task_id=task_id,
        task_family="PURE_FUNCTION",
        seed=seed,
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
    key_id = "test-signing-key"

    def sign_bytes(self, payload):
        return hmac.new(b"protocol-test-key", payload, hashlib.sha256).hexdigest()


def _generation_receipt(envelope, candidate, budget=None):
    budget = budget or _budget(envelope.seed)
    digest = digest_for
    core = {
        "schema_version": "egv-recommissioning-public-generation-receipt-v2",
        "coordinate_id": envelope.coordinate_id,
        "model_bundle": envelope.model_bundle,
        "task_id": envelope.task_id,
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
        "service_attestation_digest": digest("service-attestation"),
        "runtime_config_digest": digest("runtime-config"),
        "launch_receipt_digest": digest("launch-receipt"),
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
        "elapsed_ms": 12,
        "coordinate_debit_digest": digest("debit-{}".format(envelope.attempt_index)),
        "coordinate_reserved_tokens": 356,
        "coordinate_cumulative_reserved_tokens": 356 * envelope.attempt_index,
        "coordinate_state_digest": digest("coordinate-state-{}".format(envelope.attempt_index)),
    }
    return PublicGenerationReceipt.build(core, TestSigner()).public_record()


def _feedback(coordinate, task, envelope, candidate, previous=None):
    return lineage_feedback(
        coordinate=coordinate,
        task=task,
        model_profile_digest=_profile_digest(),
        prior_prompt=envelope,
        generation_receipt=_generation_receipt(envelope, candidate),
        evaluator_receipt={
            "schema_version": "egv-test-evaluator-receipt-v1",
            "coordinate_id": coordinate.coordinate_id,
            "attempt_index": envelope.attempt_index,
            "decision": "WRONG_OUTPUT",
        },
        candidate=candidate,
        diagnostic="WRONG_OUTPUT",
        public_summary="The public functional contract still disagrees.",
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
        with self.assertRaises(Exception):
            lineage_feedback(
                coordinate=coordinate,
                task=task,
                model_profile_digest=_profile_digest(),
                prior_prompt=prompt,
                generation_receipt=tampered,
                evaluator_receipt={"decision": "WRONG_OUTPUT"},
                candidate=candidate,
                diagnostic="WRONG_OUTPUT",
                public_summary="Mismatch.",
            )
        other_task = _task(_coordinate(task_id="egv-learnable-pure-function-train-002-v2"))
        with self.assertRaisesRegex(RecommissioningProtocolError, "task-package"):
            prompt.validate(task=other_task)

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
        self.assertNotIn("source", feedback.public_redacted_record())
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

    def test_learnable_v2_adapter_retains_original_package_record_corpus_and_structured_spec(self):
        task = {
            "schema_version": "egv-learnable-task-v2",
            "incompatible_with": ["egv-evaluation-data-v1"],
            "task_id": "egv-learnable-pure-function-train-001-v2",
            "family_id": "PURE_FUNCTION",
            "split": "train",
            "ordinal": 1,
            "public_spec": {"family_id": "PURE_FUNCTION", "task_scope": "scope", "operation": "double"},
            "public_spec_digest": "",
            "starter_source": "def main(value):\n    return value\n",
            "starter_source_digest": "",
            "record_digest": "",
        }
        task["public_spec_digest"] = digest_for(task["public_spec"])
        task["starter_source_digest"] = digest_bytes(task["starter_source"].encode("utf-8"))
        task["record_digest"] = digest_for({key: value for key, value in task.items() if key != "record_digest"})
        package = {"schema_version": "egv-learnable-model-package-v2", "task": task, "package_digest": ""}
        package["package_digest"] = digest_for({key: value for key, value in package.items() if key != "package_digest"})
        fake_module = types.ModuleType("egv.evaluation.learnable")
        fake_module.validate_model_visible_package = lambda value: json.loads(canonical_json(value))
        previous = sys.modules.get("egv.evaluation.learnable")
        sys.modules["egv.evaluation.learnable"] = fake_module
        try:
            corpus_digest = digest_for("learnable-v2-corpus")
            adapted = VisibleTaskPackage.from_learnable_v2(package, corpus_digest)
        finally:
            if previous is None:
                sys.modules.pop("egv.evaluation.learnable", None)
            else:
                sys.modules["egv.evaluation.learnable"] = previous
        record = adapted.model_visible_record()
        self.assertEqual(record["original_package_digest"], package["package_digest"])
        self.assertEqual(record["original_task_record_digest"], task["record_digest"])
        self.assertEqual(record["corpus_manifest_digest"], corpus_digest)
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
