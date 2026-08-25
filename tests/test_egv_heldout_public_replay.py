from __future__ import annotations

import unittest

from egv.canonical import digest_for
from egv.errors import PublicSchemaError
from egv.public import validate_public_candidate


def _candidate(arm, adapter_digest):
    return {
        "campaign_id": "campaign-alpha",
        "run_id": "run-alpha",
        "task_id": "task-alpha",
        "arm": arm,
        "attempt_index": 0,
        "candidate_id": "candidate-alpha",
        "parent_candidate_id": None,
        "candidate_artifact_digest": digest_for("candidate"),
        "model_digest": digest_for("base-model"),
        "adapter_digest": adapter_digest,
        "prompt_template_digests": [digest_for("prompt")],
        "mutation_family": "bounded-repair",
        "normalized_public_locus": "module.function",
        "requested_authority": "NONE",
        "declared_public_evidence_ids": [],
        "public_dependency_ids": [],
    }


class HeldoutPublicReplayProjectionTests(unittest.TestCase):
    def test_base_arm_requires_null_adapter_projection(self):
        for arm in ("A", "B", "C", "D"):
            with self.subTest(arm=arm):
                self.assertIsNone(validate_public_candidate(_candidate(arm, None))["adapter_digest"])
                with self.assertRaisesRegex(PublicSchemaError, "must project.*null"):
                    validate_public_candidate(_candidate(arm, digest_for("adapter")))

    def test_lora_arm_requires_separate_adapter_digest(self):
        for arm in ("E", "F", "G", "H"):
            with self.subTest(arm=arm):
                expected = digest_for("adapter")
                self.assertEqual(validate_public_candidate(_candidate(arm, expected))["adapter_digest"], expected)
                with self.assertRaisesRegex(PublicSchemaError, "adapter_digest"):
                    validate_public_candidate(_candidate(arm, None))

    def test_unknown_arm_is_rejected(self):
        with self.assertRaisesRegex(PublicSchemaError, "A-H"):
            validate_public_candidate(_candidate("I", None))


if __name__ == "__main__":
    unittest.main()
