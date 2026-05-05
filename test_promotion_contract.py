import unittest

from evidence_contract import build_episode_event, build_evidence_bundle
from promotion_contract import PromotionGateConfig, evaluate_promotion_candidate


class TestPromotionContract(unittest.TestCase):
    def test_promotion_candidate_fails_closed_when_required_reports_are_missing(self):
        bundle = build_evidence_bundle(
            [build_episode_event(event_type="observe", surface="model_scope", query_id="q1")]
        )

        decision = evaluate_promotion_candidate(
            candidate_id="shadow_v1",
            evidence_bundle=bundle,
            comparator_report={"passed": True, "score": 1.0},
            config=PromotionGateConfig(require_holdout=True, require_safety=True),
        )

        self.assertFalse(decision["promotion_ready"])
        self.assertIn("missing_holdout_report", decision["reasons"])
        self.assertIn("missing_safety_report", decision["reasons"])

    def test_promotion_candidate_passes_when_all_gates_are_clean(self):
        bundle = build_evidence_bundle(
            [build_episode_event(event_type="observe", surface="model_scope", query_id="q1")]
        )

        decision = evaluate_promotion_candidate(
            candidate_id="shadow_v1",
            evidence_bundle=bundle,
            comparator_report={"passed": True, "score": 0.9},
            holdout_report={"passed": True, "score": 0.8},
            safety_report={"passed": True},
            hard_negative_report={"blocker_count": 0},
            evaluator_report={"agreement_score": 1.0},
            config=PromotionGateConfig(min_replay_score=0.5, min_holdout_score=0.5, min_evaluator_agreement=0.5),
        )

        self.assertTrue(decision["promotion_ready"])
        self.assertEqual(decision["reasons"], [])

    def test_promotion_candidate_reports_empty_bad_schema_and_reward_divergence(self):
        decision = evaluate_promotion_candidate(
            candidate_id="shadow_v1",
            evidence_bundle={"schema_version": 999, "artifact_type": "evidence_bundle", "events": []},
            comparator_report={"passed": True, "score": 1.0},
            holdout_report={"passed": False, "score": 0.2},
            safety_report={"passed": True},
            evaluator_report={"agreement_score": 0.4},
            reward_report={"heldout_divergence_detected": True},
            config=PromotionGateConfig(min_evaluator_agreement=0.5),
        )

        self.assertFalse(decision["promotion_ready"])
        self.assertIn("unsupported_evidence_bundle_schema", decision["reasons"])
        self.assertIn("empty_evidence_bundle", decision["reasons"])
        self.assertIn("holdout_failed", decision["reasons"])
        self.assertIn("evaluator_agreement_below_threshold", decision["reasons"])
        self.assertIn("reward_overoptimization_detected", decision["reasons"])


if __name__ == "__main__":
    unittest.main()
