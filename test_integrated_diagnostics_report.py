import json
import unittest

from integrated_diagnostics_report import IntegratedDiagnosticsReport


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


if __name__ == "__main__":
    unittest.main()
