from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from analyze_drift_recovery import analyze_artifacts, load_artifacts, write_diagnostics


class TestAnalyzeDriftRecovery(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)

    def test_load_artifacts_filters_non_matching_record_types(self):
        self._write_matrix(self.root)
        (self.root / "scifact_not_an_experiment.json").write_text(
            json.dumps({"record_type": "reviewer_smoke"}),
            encoding="utf-8",
        )

        artifacts = load_artifacts(str(self.root / "scifact_*.json"))

        self.assertEqual(len(artifacts), 30)
        self.assertTrue(all(artifact["record_type"] == "drift_recovery_experiment" for artifact in artifacts))

    def test_analyze_artifacts_confirms_hypotheses_from_cycle_metadata(self):
        artifacts = self._write_matrix(self.root)

        diagnostics = analyze_artifacts(artifacts)

        self.assertEqual(diagnostics["artifact_count"], 30)
        self.assertEqual(diagnostics["hypotheses"]["H1_rotation_too_weak"]["verdict"], "confirmed")
        self.assertEqual(diagnostics["hypotheses"]["H2_correction_barely_moves"]["verdict"], "confirmed")
        self.assertEqual(diagnostics["hypotheses"]["H3_c2_oracle_reembed"]["verdict"], "confirmed")
        c3_rotation = self._trace_summary(diagnostics, "C3", "rotation")
        self.assertEqual(c3_rotation["should_correct_true"], 36)
        self.assertEqual(c3_rotation["sedimentation_attempted_true"], 36)
        self.assertEqual(c3_rotation["correction_applied_true"], 0)
        self.assertAlmostEqual(c3_rotation["mean_correction_norm"], 0.01)
        c4_noise = self._trace_summary(diagnostics, "C4", "noise")
        self.assertLess(c4_noise["mean_correction_norm"], 0.001)

    def test_analyze_refutes_h3_when_c2_does_not_return_to_baseline(self):
        artifacts = self._write_matrix(self.root)
        for artifact in artifacts:
            if artifact["config"]["condition"] == "C2" and artifact["config"]["drift"] == "noise":
                artifact["recovery"]["trajectory"][-1]["ndcg"] = artifact["baseline"]["ndcg_at_10"] - 0.001
                break

        diagnostics = analyze_artifacts(artifacts)

        self.assertEqual(diagnostics["hypotheses"]["H3_c2_oracle_reembed"]["verdict"], "refuted")

    def test_write_diagnostics_emits_json_markdown_and_plot_paths(self):
        diagnostics = analyze_artifacts(self._write_matrix(self.root))
        output_json = self.root / "diagnostics.json"
        output_md = self.root / "diagnostics.md"

        with patch("analyze_drift_recovery.write_trajectory_plots", return_value=["plot.png"]):
            outputs = write_diagnostics(diagnostics, str(output_json), str(output_md), str(self.root))

        self.assertEqual(outputs["plots"], ["plot.png"])
        self.assertEqual(json.loads(output_json.read_text(encoding="utf-8"))["artifact_count"], 30)
        markdown = output_md.read_text(encoding="utf-8")
        self.assertIn("## Hypothesis Verdicts", markdown)
        self.assertIn("Baseline vs Final", markdown)

    def _write_matrix(self, root: Path):
        artifacts = []
        for condition in ("C0", "C1", "C2", "C3", "C4"):
            for drift in ("rotation", "noise"):
                for seed in (42, 1337, 7):
                    artifact = self._artifact(condition, drift, seed)
                    path = root / f"scifact_{condition}_{drift}_seed{seed}.json"
                    artifact["_source_path"] = str(path).replace("\\", "/")
                    path.write_text(json.dumps(artifact), encoding="utf-8")
                    artifacts.append(artifact)
        return artifacts

    def _artifact(self, condition: str, drift: str, seed: int):
        baseline = 1.0
        if condition == "C2":
            final = baseline
        elif drift == "rotation":
            final = 0.985
        else:
            final = 0.82
        trajectory = []
        for cycle in range(1, 13):
            metadata = {"condition": condition, "action": "none"}
            if condition in {"C3", "C4"}:
                norm = 0.01 if condition == "C3" else 0.0002
                metadata.update(
                    {
                        "annealing_observation": {
                            "drift_magnitude": 0.002 if drift == "rotation" else 0.03,
                            "should_correct": True,
                            "temperature": 0.002,
                        },
                        "should_correct": True,
                        "sedimentation_attempted": True,
                        "correction_applied": False,
                        "correction_norm_stats": {
                            "count": 2,
                            "mean": norm,
                            "max": norm,
                            "sample_norms": [norm, norm],
                        },
                    }
                )
            trajectory.append({"cycle_index": cycle, "ndcg": final, "metadata": metadata})
        return {
            "record_type": "drift_recovery_experiment",
            "config": {
                "condition": condition,
                "drift": drift,
                "seed": seed,
                "cycles": 12,
                "max_queries": 100,
                "sample_docs": 1200,
            },
            "baseline": {"ndcg_at_10": baseline, "query_ndcg": []},
            "recovery": {
                "baseline_ndcg": baseline,
                "recovery_threshold": 0.95,
                "recovery_cycle": 1 if final >= 0.95 else None,
                "trajectory": trajectory,
            },
            "correction_norm_stats": {"count": 0, "mean": None, "max": None, "sample_norms": []},
            "cycle_errors": [],
        }

    def _trace_summary(self, diagnostics, condition, drift):
        for row in diagnostics["c3_c4_traces"]["summary"]:
            if row["condition"] == condition and row["drift"] == drift:
                return row
        self.fail(f"Missing trace summary for {condition}/{drift}")


if __name__ == "__main__":
    unittest.main()
