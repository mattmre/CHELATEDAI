import tempfile
import unittest
from pathlib import Path

from evidence_contract import (
    build_episode_event,
    build_evidence_bundle,
    evidence_event_from_engine_scope_row,
    summarize_evidence_bundle,
    write_evidence_bundle,
    load_evidence_bundle,
)


class TestEvidenceContract(unittest.TestCase):
    def test_build_bundle_summary_and_round_trip(self):
        event = build_episode_event(
            event_type="unit_eval",
            surface="evaluator",
            query_id="q1",
            decision="passed",
            outcome_metrics={"score": 0.9},
            source_family="unit",
        )
        bundle = build_evidence_bundle([event], campaign_id="campaign_1")
        summary = summarize_evidence_bundle(bundle)

        self.assertEqual(bundle["event_count"], 1)
        self.assertEqual(summary["surfaces"]["evaluator"], 1)
        self.assertEqual(summary["decisions"]["passed"], 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = write_evidence_bundle(Path(tmpdir) / "bundle.json", bundle)
            loaded = load_evidence_bundle(path)

        self.assertEqual(loaded["bundle_id"], bundle["bundle_id"])

    def test_engine_scope_row_conversion_preserves_decision_metrics_and_payload(self):
        row = {
            "row_type": "query_profile",
            "query_id": "q1",
            "action": "REFORMULATE",
            "fault_class": "actuator_active_negative",
            "delta_ndcg_at_10": -0.03,
            "task": "SciFact",
            "profile": "reform_rrf_v2",
        }

        event = evidence_event_from_engine_scope_row(row)

        self.assertEqual(event["surface"], "engine_scope")
        self.assertEqual(event["decision"], "actuator_active_negative")
        self.assertEqual(event["outcome_metrics"]["delta_ndcg_at_10"], -0.03)
        self.assertEqual(event["payload"]["profile"], "reform_rrf_v2")


if __name__ == "__main__":
    unittest.main()
