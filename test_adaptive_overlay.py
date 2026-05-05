import unittest

from adaptive_overlay import (
    build_overlay_artifact_card,
    build_overlay_report,
    build_overlay_validation_report,
    build_adaptive_overlay_intake,
    build_channel_variation_records,
    channel_variation_from_engine_scope_row,
    compute_branch_set_metrics,
    summarize_overlay_readiness,
    summarize_channel_variations,
)


class TestAdaptiveOverlay(unittest.TestCase):
    def test_engine_scope_row_maps_active_negative_to_damped_channel(self):
        row = {
            "row_type": "query_profile",
            "source_family": "unit",
            "task": "SciFact",
            "seed": 42,
            "query_id": "q1",
            "profile": "reform_rrf_v2",
            "action": "REFORMULATE",
            "fault_class": "actuator_active_negative",
            "promotion_blocker": True,
            "delta_ndcg_at_10": -0.02,
        }

        record = channel_variation_from_engine_scope_row(row)

        self.assertEqual(record["record_type"], "channel_variation")
        self.assertEqual(record["channel_type"], "query_reformulation")
        self.assertEqual(record["aggression_level"], "normal")
        self.assertEqual(record["protection_level"], "guarded")
        self.assertEqual(record["decision"], "damp_candidate")
        self.assertEqual(record["active_negative_flags"], ["actuator_active_negative"])
        self.assertEqual(record["metric_delta"], -0.02)
        self.assertTrue(record["record_hash"].startswith("overlay_"))

    def test_baseline_row_is_frozen_and_protected(self):
        record = channel_variation_from_engine_scope_row({
            "row_type": "query_profile",
            "task": "NFCorpus",
            "query_id": "q2",
            "profile": "baseline",
            "action": "FAST",
            "fault_class": "reference",
            "delta_ndcg_at_10": 0.0,
        })

        self.assertEqual(record["channel_type"], "baseline")
        self.assertEqual(record["protection_level"], "frozen")
        self.assertEqual(record["decision"], "protect_baseline")

    def test_build_intake_and_summary_are_stable_observation_artifacts(self):
        rows = [
            {
                "row_type": "query_profile",
                "task": "SciFact",
                "query_id": "q1",
                "profile": "baseline",
                "fault_class": "reference",
            },
            {
                "row_type": "query_profile",
                "task": "SciFact",
                "query_id": "q1",
                "profile": "adaptive_p85_t0.002",
                "action": "CHELATE",
                "fault_class": "actuator_active_positive",
                "delta_ndcg_at_10": 0.01,
            },
        ]

        records = build_channel_variation_records(rows)
        intake = build_adaptive_overlay_intake(query_id="q1", task="SciFact", channel_records=records)
        summary = summarize_channel_variations(records)

        self.assertEqual(intake["channel_record_count"], 2)
        self.assertTrue(intake["intake_id"].startswith("intake_"))
        self.assertEqual(summary["record_count"], 2)
        self.assertEqual(summary["decisions"]["protect_baseline"], 1)
        self.assertEqual(summary["decisions"]["amplify_candidate"], 1)

    def test_branch_set_metrics_report_oracle_signal_and_regression_risk(self):
        records = build_channel_variation_records([
            {
                "row_type": "query_profile",
                "task": "SciFact",
                "seed": 1,
                "query_id": "q1",
                "profile": "baseline",
                "fault_class": "reference",
                "delta_ndcg_at_10": 0.0,
            },
            {
                "row_type": "query_profile",
                "task": "SciFact",
                "seed": 1,
                "query_id": "q1",
                "profile": "reform_rrf_v2",
                "action": "REFORMULATE",
                "fault_class": "actuator_active_positive",
                "delta_ndcg_at_10": 0.02,
            },
            {
                "row_type": "query_profile",
                "task": "SciFact",
                "seed": 1,
                "query_id": "q2",
                "profile": "adaptive_p85_t0.002",
                "action": "CHELATE",
                "fault_class": "actuator_active_negative",
                "promotion_blocker": True,
                "delta_ndcg_at_10": -0.03,
            },
        ])

        metrics = compute_branch_set_metrics(records)

        self.assertEqual(metrics["group_count"], 2)
        self.assertEqual(metrics["pass_at_k_rate"], 0.5)
        self.assertEqual(metrics["safe_pass_at_k_rate"], 0.5)
        self.assertEqual(metrics["regressed_at_k_rate"], 0.5)
        self.assertGreater(metrics["mean_best_delta"], -0.01)

    def test_overlay_report_combines_records_summary_and_branch_metrics(self):
        report = build_overlay_report([
            {
                "row_type": "query_profile",
                "task": "SciFact",
                "seed": 1,
                "query_id": "q1",
                "profile": "baseline",
                "fault_class": "reference",
                "delta_ndcg_at_10": 0.0,
            },
            {
                "row_type": "query_profile",
                "task": "SciFact",
                "seed": 1,
                "query_id": "q1",
                "profile": "reform_rrf_v2",
                "action": "REFORMULATE",
                "fault_class": "actuator_active_positive",
                "delta_ndcg_at_10": 0.01,
            },
        ])

        self.assertEqual(report["record_type"], "adaptive_overlay_report")
        self.assertEqual(report["summary"]["record_count"], 2)
        self.assertEqual(report["branch_set_metrics"]["group_count"], 1)
        self.assertEqual(len(report["channel_variation_records"]), 2)
        self.assertIn("readiness", report)

    def test_overlay_readiness_fails_closed_on_blockers(self):
        report = build_overlay_report([
            {
                "row_type": "query_profile",
                "task": "SciFact",
                "seed": 1,
                "query_id": "q1",
                "profile": "reform_rrf_v2",
                "action": "REFORMULATE",
                "fault_class": "actuator_active_negative",
                "promotion_blocker": True,
                "delta_ndcg_at_10": -0.02,
            }
        ])

        readiness = summarize_overlay_readiness(report)

        self.assertFalse(readiness["ready_for_broader_validation"])
        self.assertIn("promotion_blockers_present", readiness["blockers"])
        self.assertIn("active_negative_records_present", readiness["blockers"])
        self.assertEqual(readiness["next_action"], "continue observation and coverage-aware channel collection")

    def test_overlay_readiness_accepts_clean_multi_group_signal(self):
        rows = []
        for query_id in ("q1", "q2", "q3"):
            rows.append({
                "row_type": "query_profile",
                "task": "SciFact",
                "seed": 1,
                "query_id": query_id,
                "profile": "reform_rrf_v2",
                "action": "REFORMULATE",
                "fault_class": "actuator_active_positive",
                "delta_ndcg_at_10": 0.02,
            })
        report = build_overlay_report(rows)

        self.assertTrue(report["readiness"]["ready_for_broader_validation"])
        self.assertEqual(report["readiness"]["blockers"], [])

    def test_overlay_artifact_card_is_compact_and_links_promotion_evidence(self):
        report = build_overlay_report([
            {
                "row_type": "query_profile",
                "task": "SciFact",
                "seed": 1,
                "query_id": "q1",
                "profile": "mask_gate_v1",
                "action": "CHELATE",
                "fault_class": "actuator_active_negative",
                "promotion_blocker": True,
                "delta_ndcg_at_10": -0.02,
            }
        ])

        card = build_overlay_artifact_card(
            candidate_id="candidate-a",
            overlay_report=report,
            source_path="overlay.json",
            promotion_decision={
                "promotion_ready": False,
                "adaptive_overlay_ready": False,
                "reasons": ["adaptive_overlay_not_ready"],
            },
            holdout_report={"passed": True, "score": 0.8},
            safety_report={"passed": True},
            rollback_path="policies/current.json",
        )

        self.assertEqual(card["record_type"], "adaptive_overlay_artifact_card")
        self.assertTrue(card["card_id"].startswith("overlay_card_"))
        self.assertEqual(card["candidate_id"], "candidate-a")
        self.assertEqual(card["source_overlay_report_path"], "overlay.json")
        self.assertFalse(card["readiness"]["ready_for_broader_validation"])
        self.assertIn("promotion_blockers_present", card["readiness"]["blockers"])
        self.assertFalse(card["evidence"]["promotion_decision"]["promotion_ready"])
        self.assertIn("not_default_promoted", card["limitations"])
        self.assertNotIn("channel_variation_records", card)

    def test_overlay_validation_report_requires_replay_and_holdout_readiness(self):
        replay_report = build_overlay_report([
            {
                "task": "SciFact",
                "seed": 1,
                "query_id": query_id,
                "profile": "guard_learned_reform_gate_v1",
                "delta_ndcg_at_10": 0.02,
                "fault_class": "actuator_active_positive",
            }
            for query_id in ("q1", "q2", "q3")
        ])

        missing_holdout = build_overlay_validation_report(
            candidate_id="candidate-a",
            replay_overlay_report=replay_report,
            holdout_overlay_report=None,
        )
        self.assertFalse(missing_holdout["validation_ready"])
        self.assertIn("missing_holdout_overlay_report", missing_holdout["blockers"])

        holdout_report = build_overlay_report([
            {
                "task": "SciFact",
                "seed": 2,
                "query_id": query_id,
                "profile": "guard_learned_reform_gate_v1",
                "delta_ndcg_at_10": 0.02,
                "fault_class": "actuator_active_positive",
            }
            for query_id in ("q4", "q5", "q6")
        ])
        validation = build_overlay_validation_report(
            candidate_id="candidate-a",
            replay_overlay_report=replay_report,
            holdout_overlay_report=holdout_report,
        )

        self.assertTrue(validation["validation_ready"])
        self.assertEqual(validation["blockers"], [])
        self.assertTrue(validation["validation_report_id"].startswith("overlay_validation_"))


if __name__ == "__main__":
    unittest.main()
