from __future__ import annotations

import base64
import copy
import json
import unittest

from egv.canonical import digest_for
from egv.training.public import (
    PUBLIC_TRAINING_COMPLETED_SCHEMA,
    PUBLIC_TRAINING_PREFLIGHT_SCHEMA,
    PublicTrainingSafetyError,
    load_public_training_summary_json,
    public_training_summary_json,
    scan_public_training_summary,
    validate_public_training_summary,
)


def _digest(label):
    return digest_for(label)


def valid_summary():
    return {
        "schema_version": PUBLIC_TRAINING_COMPLETED_SCHEMA,
        "status": "COMPLETED",
        "protocol_digest": _digest("protocol"),
        "model_manifest_digest": _digest("model"),
        "data_manifest_digest": _digest("data"),
        "target_manifest_digest": _digest("targets"),
        "training_dataset_digest": _digest("training-data"),
        "adapter_manifest_digest": _digest("adapter"),
        "base_immutability_proof_digest": _digest("immutability"),
        "software_manifest_digest": _digest("software"),
        "development_receipt_digest": _digest("signed-dev-receipt"),
        "development_receipt_binding_digest": _digest("dev-receipt-binding"),
        "aggregate_metrics": {
            "training_row_count": 308,
            "training_task_count": 20,
            "development_task_count": 8,
            "epoch_count": 3,
            "optimizer_step_count": 120,
            "base_tensor_count": 500,
            "trainable_adapter_parameter_count": 228,
            "aggregate_training_loss": 0.812345,
            "aggregate_development_loss": 0.934567,
        },
        "limitations": ["EVALUATION_RESULTS_EXCLUDED", "NO_PRODUCTION_CLAIM", "RESEARCH_ONLY"],
    }


def valid_preflight():
    payload = {
        key: value
        for key, value in valid_summary().items()
        if key
        not in {
            "training_dataset_digest",
            "adapter_manifest_digest",
            "base_immutability_proof_digest",
            "development_receipt_digest",
            "development_receipt_binding_digest",
        }
    }
    payload["schema_version"] = PUBLIC_TRAINING_PREFLIGHT_SCHEMA
    payload["status"] = "READY"
    payload["aggregate_metrics"] = {
        "training_row_count": 0,
        "training_task_count": 0,
        "development_task_count": 0,
        "epoch_count": 0,
        "optimizer_step_count": 0,
        "base_tensor_count": 0,
        "trainable_adapter_parameter_count": 0,
        "aggregate_training_loss": 0.0,
        "aggregate_development_loss": 0.0,
    }
    return payload


class PublicTrainingTests(unittest.TestCase):
    def test_valid_completed_and_preflight_roundtrip(self):
        for payload in (valid_summary(), valid_preflight()):
            with self.subTest(schema=payload["schema_version"]):
                normalized = validate_public_training_summary(payload)
                self.assertEqual(scan_public_training_summary(normalized)["findings"], [])
                encoded = public_training_summary_json(payload)
                self.assertTrue(encoded.endswith("\n"))
                self.assertEqual(load_public_training_summary_json(encoded), normalized)
                self.assertEqual(json.loads(encoded), normalized)

    def test_completed_zero_or_incomplete_claims_fail_closed(self):
        fields = (
            "training_row_count",
            "optimizer_step_count",
            "base_tensor_count",
            "trainable_adapter_parameter_count",
        )
        for field in fields:
            payload = valid_summary()
            payload["aggregate_metrics"][field] = 0
            with self.subTest(field=field), self.assertRaisesRegex(PublicTrainingSafetyError, "nonzero"):
                validate_public_training_summary(payload)
        for field, value in (("training_task_count", 19), ("development_task_count", 7)):
            payload = valid_summary()
            payload["aggregate_metrics"][field] = value
            with self.subTest(field=field), self.assertRaisesRegex(PublicTrainingSafetyError, "exactly"):
                validate_public_training_summary(payload)
        for epochs in (0, 4):
            payload = valid_summary()
            payload["aggregate_metrics"]["epoch_count"] = epochs
            with self.subTest(epochs=epochs), self.assertRaisesRegex(PublicTrainingSafetyError, "one to three"):
                validate_public_training_summary(payload)

    def test_completed_requires_distinct_signed_development_receipt_bindings(self):
        payload = valid_summary()
        del payload["development_receipt_digest"]
        with self.assertRaisesRegex(PublicTrainingSafetyError, "unexpected field"):
            validate_public_training_summary(payload)
        payload = valid_summary()
        payload["development_receipt_binding_digest"] = payload["development_receipt_digest"]
        with self.assertRaisesRegex(PublicTrainingSafetyError, "must be distinct"):
            validate_public_training_summary(payload)

    def test_preflight_and_completed_schemas_cannot_be_substituted(self):
        payload = valid_preflight()
        payload["status"] = "COMPLETED"
        with self.assertRaisesRegex(PublicTrainingSafetyError, "state schema"):
            validate_public_training_summary(payload)
        payload = valid_summary()
        payload["status"] = "READY"
        with self.assertRaisesRegex(PublicTrainingSafetyError, "state schema"):
            validate_public_training_summary(payload)

    def test_nested_forbidden_fields_are_rejected_case_insensitively(self):
        cases = [
            ("raw_prompt", "repair this"),
            ("training_target", "desired completion"),
            ("candidate_source", "def solve(): pass"),
            ("dataset_example", {"input": "x"}),
            ("token", "not-public"),
            ("user", "operator"),
            ("host", "builder-node"),
            ("port", 8888),
            ("optimizer_state", {"step": 12}),
            ("rng_state", [1, 2, 3]),
            ("dev_task_loss", {"task-1": 0.2}),
            ("hidden_test", "assert result == 7"),
            ("heldout_identity", "task-secret"),
        ]
        for field, value in cases:
            payload = valid_summary()
            payload["aggregate_metrics"]["NeStEd"] = {field.swapcase(): value}
            with self.subTest(field=field), self.assertRaisesRegex(PublicTrainingSafetyError, "forbidden field"):
                validate_public_training_summary(payload)

    def test_forbidden_plain_and_encoded_values_are_rejected(self):
        leaks = [
            "pass" + "word" + "=fixture-only",
            "OPENAI_API_KEY=sk-example",
            "Bearer abcdef123456",
            "username=matt",
            "builder@example.test",
            "hostname=internal-node",
            r"C:\Users\person\checkpoint.bin",
            "/home/person/private/model",
            "ssh://person@192.0.2.24:22",
            "198.51.100.10:8888",
            "latency_ms=12.345678",
            "2026-08-22T04:15:16Z started_at",
            "checkpoint optimizer_state and RNG_STATE",
            "held-out task material",
            "%2Fhome%2Fperson%2Fprivate%2Fweights.bin",
            "pass" + "word%3Dfixture-only",
            base64.b64encode(b"api_key=do-not-publish").decode("ascii"),
            "&#x2f;root&#x2f;private&#x2f;checkpoint",
        ]
        for leak in leaks:
            payload = valid_summary()
            payload["aggregate_metrics"]["nested"] = [{"note": leak}]
            with self.subTest(leak=leak), self.assertRaises(PublicTrainingSafetyError):
                validate_public_training_summary(payload)

    def test_closed_schema_and_scalar_types_are_strict(self):
        payload = valid_summary()
        payload["notes"] = "all good"
        with self.assertRaisesRegex(PublicTrainingSafetyError, "unexpected field"):
            validate_public_training_summary(payload)
        mutations = []
        payload = valid_summary()
        payload["model_manifest_digest"] = "A" * 64
        mutations.append(payload)
        payload = valid_summary()
        payload["aggregate_metrics"]["training_row_count"] = True
        mutations.append(payload)
        payload = valid_summary()
        payload["aggregate_metrics"]["aggregate_training_loss"] = 0.1234567
        mutations.append(payload)
        payload = valid_summary()
        payload["limitations"] = ["RESEARCH_ONLY", "free form"]
        mutations.append(payload)
        for mutation in mutations:
            with self.subTest(mutation=mutation), self.assertRaises(PublicTrainingSafetyError):
                validate_public_training_summary(mutation)

    def test_validation_returns_detached_projection(self):
        payload = valid_summary()
        normalized = validate_public_training_summary(payload)
        altered = copy.deepcopy(payload)
        altered["aggregate_metrics"]["training_row_count"] = 0
        self.assertEqual(normalized["aggregate_metrics"]["training_row_count"], 308)
        self.assertEqual(validate_public_training_summary(payload), normalized)


if __name__ == "__main__":
    unittest.main()
