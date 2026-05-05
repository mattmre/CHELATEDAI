import json
import unittest

from adaptive_overlay import build_overlay_report
from integrated_diagnostics_report import IntegratedDiagnosticsReport, summarize_adaptive_overlay_report


class _StubComposition:
    candidate_id = "candidate"

    def to_dict(self):
        return {
            "retrieval_fitness": 0.5,
            "final_fitness": 0.55,
            "retrieval_metrics": {"ndcg_at_k": 0.5},
            "structural_health": {"score": 0.9},
            "quantization_gate": {"passed": True},
            "storage_metadata": {"storage_latency_ms": 4.0},
        }


class TestIntegratedDiagnosticsReportModelScope(unittest.TestCase):
    def test_to_dict_preserves_model_scope_payload(self):
        diagnostics = IntegratedDiagnosticsReport.from_composition(
            _StubComposition(),
            phase="unit_test",
            runtime={"status": "ok", "latency_ms": 1.0},
            model_scope={
                "status": "observed",
                "captured_layer_count": 2,
                "memory": {"episode_entry_id": "episode_1"},
                "expectation_comparison": {"passed": True, "score": 1.0},
            },
            evidence_summary={"event_count": 2},
            evaluator_summary={"agreement_score": 1.0},
            safety_summary={"passed": True},
            hard_negative_summary={"blocker_count": 0},
        )

        payload = diagnostics.to_dict()
        json.dumps(payload)

        self.assertEqual(payload["model_scope"]["status"], "observed")
        self.assertEqual(payload["model_scope"]["memory"]["episode_entry_id"], "episode_1")
        self.assertTrue(payload["model_scope"]["expectation_comparison"]["passed"])
        self.assertEqual(payload["evidence_summary"]["event_count"], 2)
        self.assertEqual(payload["evaluator_summary"]["agreement_score"], 1.0)
        self.assertTrue(payload["safety_summary"]["passed"])
        self.assertEqual(payload["hard_negative_summary"]["blocker_count"], 0)

    def test_to_dict_preserves_adaptive_overlay_summary(self):
        overlay_report = build_overlay_report([
            {
                "task": "SciFact",
                "seed": 1,
                "query_id": "q1",
                "profile": "baseline",
                "delta_ndcg_at_10": 0.0,
                "fault_class": "reference",
            },
            {
                "task": "SciFact",
                "seed": 1,
                "query_id": "q1",
                "profile": "mask_gate_v1",
                "delta_ndcg_at_10": -0.02,
                "fault_class": "actuator_active_negative",
                "promotion_blocker": True,
            },
        ])
        overlay_summary = summarize_adaptive_overlay_report(overlay_report)
        diagnostics = IntegratedDiagnosticsReport.from_composition(
            _StubComposition(),
            phase="unit_test",
            adaptive_overlay_summary=overlay_summary,
        )

        payload = diagnostics.to_dict()
        json.dumps(payload)

        self.assertEqual(payload["adaptive_overlay_summary"]["record_type"], "adaptive_overlay_summary")
        self.assertFalse(payload["adaptive_overlay_summary"]["ready_for_broader_validation"])
        self.assertIn("promotion_blockers_present", payload["adaptive_overlay_summary"]["blockers"])
        self.assertEqual(payload["adaptive_overlay_summary"]["branch_set_metrics"]["group_count"], 1)

    def test_adaptive_overlay_summary_omits_full_channel_records(self):
        overlay_report = build_overlay_report([
            {
                "task": "SciFact",
                "seed": 1,
                "query_id": "q1",
                "profile": "guard_learned_reform_gate_v1",
                "delta_ndcg_at_10": 0.02,
                "fault_class": "actuator_active_positive",
            },
        ])

        overlay_summary = summarize_adaptive_overlay_report(overlay_report)

        self.assertNotIn("channel_variation_records", overlay_summary)
        self.assertEqual(overlay_summary["record_count"], 1)

    def test_adaptive_overlay_summary_handles_partial_report(self):
        overlay_summary = summarize_adaptive_overlay_report(
            {
                "schema_version": 1,
                "record_type": "adaptive_overlay_report",
                "summary": {"record_count": None},
                "branch_set_metrics": {"group_count": None, "mean_best_delta": "unknown"},
                "readiness": {"blockers": "not-a-list"},
            }
        )

        json.dumps(overlay_summary)
        self.assertEqual(overlay_summary["record_count"], 0)
        self.assertEqual(overlay_summary["blockers"], [])
        self.assertEqual(overlay_summary["branch_set_metrics"]["group_count"], 0)
        self.assertEqual(overlay_summary["branch_set_metrics"]["mean_best_delta"], 0.0)


if __name__ == "__main__":
    unittest.main()
