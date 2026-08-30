from __future__ import annotations

import unittest

from egv.canonical import digest_for
from egv.recommissioning.protocol import (
    PilotCoordinate,
    RecommissioningProtocolError,
    VisibleTaskPackage,
    lineage_feedback,
    render_prompt,
)


def _coordinate(protocol_id="P1", *, task_id="learnable-pure-train-01-v2", seed=0):
    return PilotCoordinate(
        campaign_id="egv-nemotron-stage-a-v2",
        model_bundle="LIGHTNING",
        protocol_id=protocol_id,
        task_id=task_id,
        task_family="PURE_FUNCTION",
        seed=seed,
    )


def _task(coordinate):
    visible = {
        "schema_version": "egv-learnable-visible-task-v2",
        "task_id": coordinate.task_id,
        "task_family": coordinate.task_family,
        "public_specification": "main(value) must return value * 3 + 2 for every integer value.",
        "initial_source": "def main(value):\n    return value\n",
        "public_examples": ({"input": 2, "output": 8}, {"input": -1, "output": -1}),
        "public_locus": "module:main",
    }
    return VisibleTaskPackage(
        visible["task_id"],
        visible["task_family"],
        visible["public_specification"],
        visible["initial_source"],
        visible["public_examples"],
        visible["public_locus"],
        digest_for(visible),
    )


class RecommissioningProtocolTests(unittest.TestCase):
    def test_p0_sees_only_latest_feedback_and_p1_sees_lineage(self):
        p1 = _coordinate("P1")
        feedback = (
            lineage_feedback(
                coordinate=p1,
                attempt_index=1,
                diagnostic="WRONG_OUTPUT",
                source="def main(value):\n    return value + 2\n",
                public_summary="One public example disagreed with the candidate.",
            ),
            lineage_feedback(
                coordinate=p1,
                attempt_index=2,
                diagnostic="WRONG_OUTPUT",
                source="def main(value):\n    return value * 3\n",
                public_summary="The affine offset remains incorrect on a public example.",
            ),
        )
        p1_prompt = render_prompt(p1, _task(p1), attempt_index=3, feedback=feedback)
        self.assertEqual(p1_prompt.visible_evidence_ids, tuple(sorted(item.evidence_id for item in feedback)))
        self.assertIn(feedback[0].source_digest, p1_prompt.messages[1]["content"])

        p0 = _coordinate("P0")
        p0_feedback = tuple(
            lineage_feedback(
                coordinate=p0,
                attempt_index=item.attempt_index,
                diagnostic=item.diagnostic,
                source="def main(value):\n    return {}\n".format(item.attempt_index),
                public_summary=item.public_summary,
            )
            for item in feedback
        )
        p0_prompt = render_prompt(p0, _task(p0), attempt_index=3, feedback=p0_feedback)
        self.assertEqual(p0_prompt.visible_evidence_ids, (p0_feedback[-1].evidence_id,))
        self.assertNotIn(p0_feedback[0].source_digest, p0_prompt.messages[1]["content"])

    def test_feedback_cannot_cross_task_seed_or_arm_coordinate(self):
        first = _coordinate("P1", seed=0)
        second = _coordinate("P1", seed=1)
        foreign = lineage_feedback(
            coordinate=first,
            attempt_index=1,
            diagnostic="WRONG_OUTPUT",
            source="def main(value):\n    return value\n",
            public_summary="Public example mismatch.",
        )
        with self.assertRaisesRegex(RecommissioningProtocolError, "crossed"):
            render_prompt(second, _task(second), attempt_index=2, feedback=(foreign,))

    def test_feedback_must_be_contiguous_and_match_attempt_budget(self):
        coordinate = _coordinate()
        first = lineage_feedback(
            coordinate=coordinate,
            attempt_index=1,
            diagnostic="WRONG_OUTPUT",
            source="def main(value):\n    return value\n",
            public_summary="Public example mismatch.",
        )
        with self.assertRaisesRegex(RecommissioningProtocolError, "next attempt"):
            render_prompt(coordinate, _task(coordinate), attempt_index=3, feedback=(first,))

    def test_visible_package_rejects_evaluator_private_vocabulary(self):
        coordinate = _coordinate()
        task = _task(coordinate)
        specification = task.public_specification + " oracle_token is forbidden"
        poisoned_manifest = {
            "schema_version": "egv-learnable-visible-task-v2",
            "task_id": task.task_id,
            "task_family": task.task_family,
            "public_specification": specification,
            "initial_source": task.initial_source,
            "public_examples": [dict(example) for example in task.public_examples],
            "public_locus": task.public_locus,
        }
        poisoned = VisibleTaskPackage(
            task.task_id,
            task.task_family,
            specification,
            task.initial_source,
            task.public_examples,
            task.public_locus,
            digest_for(poisoned_manifest),
        )
        with self.assertRaisesRegex(RecommissioningProtocolError, "evaluator-private"):
            render_prompt(coordinate, poisoned, attempt_index=1)

    def test_visible_package_digest_detects_tampering(self):
        coordinate = _coordinate()
        task = _task(coordinate)
        tampered = VisibleTaskPackage(
            task.task_id,
            task.task_family,
            task.public_specification + " Return zero instead.",
            task.initial_source,
            task.public_examples,
            task.public_locus,
            task.manifest_digest,
        )
        with self.assertRaisesRegex(RecommissioningProtocolError, "does not bind"):
            render_prompt(coordinate, tampered, attempt_index=1)

    def test_attempt_one_contains_no_feedback_and_has_stable_digest(self):
        coordinate = _coordinate()
        first = render_prompt(coordinate, _task(coordinate), attempt_index=1)
        second = render_prompt(coordinate, _task(coordinate), attempt_index=1)
        self.assertEqual(first.prompt_digest, second.prompt_digest)
        self.assertEqual(first.visible_evidence_ids, ())
        self.assertNotIn("expected_output", first.messages[1]["content"])

    def test_public_examples_are_optional_when_specification_is_complete(self):
        coordinate = _coordinate()
        task = _task(coordinate)
        manifest = task.public_manifest()
        manifest["public_examples"] = []
        without_examples = VisibleTaskPackage(
            task.task_id,
            task.task_family,
            task.public_specification,
            task.initial_source,
            (),
            task.public_locus,
            digest_for(manifest),
        )
        prompt = render_prompt(coordinate, without_examples, attempt_index=1)
        self.assertEqual(prompt.attempt_index, 1)


if __name__ == "__main__":
    unittest.main()
