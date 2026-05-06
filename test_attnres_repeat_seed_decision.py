import json
import tempfile
import unittest
from pathlib import Path

from attnres_repeat_seed_decision import summarize_attnres_repeat_seed_decision


def _write_artifact(root: Path, name: str, task: str, seed: int, delta: float, quantization_passed: bool) -> Path:
    path = root / name
    path.write_text(
        json.dumps(
            {
                "task": task,
                "seed": seed,
                "default_recommendation": {
                    "recommended_profile": "attnres_balanced_trained" if delta > 0.0 else "baseline",
                    "baseline_ndcg_at_10": 0.8,
                    "best_ndcg_at_10": 0.8 + delta,
                    "delta_vs_baseline": delta,
                    "default_change_allowed": delta > 0.0,
                },
                "quantization_survival": {
                    "gate": {
                        "passed": quantization_passed,
                        "reasons": [] if quantization_passed else ["quantized_fitness_below_baseline"],
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    return path


class TestAttnResRepeatSeedDecision(unittest.TestCase):
    def test_repeat_seed_decision_fails_closed(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            summary = summarize_attnres_repeat_seed_decision(
                [
                    _write_artifact(root, "scifact_seed42.json", "SciFact", 42, 0.0, False),
                    _write_artifact(root, "scifact_seed43.json", "SciFact", 43, 0.0, False),
                    _write_artifact(root, "scifact_seed44.json", "SciFact", 44, 0.0, False),
                    _write_artifact(root, "nfcorpus_seed42.json", "NFCorpus", 42, 0.02, False),
                    _write_artifact(root, "nfcorpus_seed43.json", "NFCorpus", 43, 0.001, False),
                    _write_artifact(root, "nfcorpus_seed44.json", "NFCorpus", 44, 0.001, True),
                ]
            )

        self.assertEqual(summary["record_type"], "attnres_repeat_seed_decision")
        self.assertEqual(summary["artifact_count"], 6)
        self.assertEqual(summary["decision"], "no_default_change")
        self.assertFalse(summary["promote_default"])
        self.assertGreater(summary["quantization_failure_count"], 0)
        self.assertIn("quantization_survival_failures_present", summary["reasons"])
        self.assertIn("SciFact", summary["task_summary"])
        self.assertIn("NFCorpus", summary["task_summary"])


if __name__ == "__main__":
    unittest.main()
