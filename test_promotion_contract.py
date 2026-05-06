import unittest

from adaptive_overlay import build_overlay_report
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

    def test_promotion_candidate_can_require_artifact_card_and_rollback_linkage(self):
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
            artifact_card_reference={
                "record_type": "adaptive_overlay_artifact_card",
                "path": "campaign/adaptive_overlay_artifact_card.json",
            },
            rollback_path="policies/current.json",
            config=PromotionGateConfig(
                min_replay_score=0.5,
                min_holdout_score=0.5,
                min_evaluator_agreement=0.5,
                require_artifact_card_reference=True,
                require_rollback_path=True,
            ),
        )

        self.assertTrue(decision["promotion_ready"])
        self.assertEqual(decision["artifact_card_reference"]["path"], "campaign/adaptive_overlay_artifact_card.json")
        self.assertEqual(decision["rollback_path"], "policies/current.json")

    def test_promotion_candidate_fails_closed_when_required_linkage_is_missing(self):
        bundle = build_evidence_bundle(
            [build_episode_event(event_type="observe", surface="model_scope", query_id="q1")]
        )

        decision = evaluate_promotion_candidate(
            candidate_id="shadow_v1",
            evidence_bundle=bundle,
            comparator_report={"passed": True, "score": 1.0},
            holdout_report={"passed": True, "score": 1.0},
            safety_report={"passed": True},
            config=PromotionGateConfig(require_artifact_card_reference=True, require_rollback_path=True),
        )

        self.assertFalse(decision["promotion_ready"])
        self.assertIn("missing_artifact_card_reference", decision["reasons"])
        self.assertIn("missing_rollback_path", decision["reasons"])

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

    def test_promotion_candidate_requires_adaptive_overlay_when_configured(self):
        bundle = build_evidence_bundle(
            [build_episode_event(event_type="observe", surface="model_scope", query_id="q1")]
        )

        decision = evaluate_promotion_candidate(
            candidate_id="overlay_candidate_v1",
            evidence_bundle=bundle,
            comparator_report={"passed": True, "score": 1.0},
            holdout_report={"passed": True, "score": 1.0},
            safety_report={"passed": True},
            config=PromotionGateConfig(require_adaptive_overlay_readiness=True),
        )

        self.assertFalse(decision["promotion_ready"])
        self.assertIn("missing_adaptive_overlay_report", decision["reasons"])
        self.assertIsNone(decision["adaptive_overlay_ready"])

    def test_promotion_candidate_fails_closed_on_unready_adaptive_overlay(self):
        bundle = build_evidence_bundle(
            [build_episode_event(event_type="observe", surface="model_scope", query_id="q1")]
        )
        overlay_report = build_overlay_report([
            {
                "task": "SciFact",
                "seed": 1,
                "query_id": "q1",
                "profile": "mask_gate_v1",
                "delta_ndcg_at_10": -0.02,
                "fault_class": "actuator_active_negative",
                "promotion_blocker": True,
            }
        ])

        decision = evaluate_promotion_candidate(
            candidate_id="overlay_candidate_v1",
            evidence_bundle=bundle,
            comparator_report={"passed": True, "score": 1.0},
            holdout_report={"passed": True, "score": 1.0},
            safety_report={"passed": True},
            adaptive_overlay_report=overlay_report,
        )

        self.assertFalse(decision["promotion_ready"])
        self.assertFalse(decision["adaptive_overlay_ready"])
        self.assertIn("adaptive_overlay_not_ready", decision["reasons"])
        self.assertIn("adaptive_overlay_blockers_present", decision["reasons"])
        self.assertIn("promotion_blockers_present", decision["adaptive_overlay_blockers"])

    def test_promotion_candidate_accepts_clean_ready_adaptive_overlay(self):
        bundle = build_evidence_bundle(
            [build_episode_event(event_type="observe", surface="model_scope", query_id="q1")]
        )
        overlay_rows = []
        for query_id in ("q1", "q2", "q3"):
            overlay_rows.append(
                {
                    "task": "SciFact",
                    "seed": 1,
                    "query_id": query_id,
                    "profile": "baseline",
                    "delta_ndcg_at_10": 0.0,
                    "fault_class": "reference",
                }
            )
            overlay_rows.append(
                {
                    "task": "SciFact",
                    "seed": 1,
                    "query_id": query_id,
                    "profile": "guard_learned_reform_gate_v1",
                    "delta_ndcg_at_10": 0.02,
                    "fault_class": "actuator_active_positive",
                    "promotion_blocker": False,
                }
            )

        decision = evaluate_promotion_candidate(
            candidate_id="overlay_candidate_v1",
            evidence_bundle=bundle,
            comparator_report={"passed": True, "score": 1.0},
            holdout_report={"passed": True, "score": 1.0},
            safety_report={"passed": True},
            adaptive_overlay_report=build_overlay_report(overlay_rows),
            config=PromotionGateConfig(require_adaptive_overlay_readiness=True),
        )

        self.assertTrue(decision["promotion_ready"])
        self.assertTrue(decision["adaptive_overlay_ready"])
        self.assertEqual(decision["adaptive_overlay_blockers"], [])


if __name__ == "__main__":
    unittest.main()
