import unittest

from adaptive_overlay import (
    build_adaptive_overlay_intake,
    build_channel_variation_records,
    channel_variation_from_engine_scope_row,
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


if __name__ == "__main__":
    unittest.main()
