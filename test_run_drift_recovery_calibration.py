from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from run_drift_recovery_calibration import (
    CalibrationConfig,
    _choose_setting,
    run_calibration_campaign,
)


class TestRunDriftRecoveryCalibration(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)

    def test_choose_setting_prefers_in_zone_closest_to_twelve_percent(self):
        rows = [
            {"drop_pct": 4.0, "in_zone": False, "fraction": 0.5, "angle": 35.0, "sigma": 0.05},
            {"drop_pct": 9.0, "in_zone": True, "fraction": 0.5, "angle": 50.0, "sigma": 0.05},
            {"drop_pct": 15.0, "in_zone": True, "fraction": 0.75, "angle": 50.0, "sigma": 0.05},
        ]

        chosen = _choose_setting(rows)

        self.assertEqual(chosen["drop_pct"], 9.0)

    def test_choose_setting_uses_closest_overall_when_no_cell_is_in_zone(self):
        rows = [
            {"drop_pct": 3.0, "in_zone": False, "fraction": 0.5, "angle": 35.0, "sigma": 0.05},
            {"drop_pct": 22.0, "in_zone": False, "fraction": 0.75, "angle": 65.0, "sigma": 0.05},
        ]

        chosen = _choose_setting(rows)

        self.assertEqual(chosen["drop_pct"], 3.0)

    def test_campaign_runs_nine_scouts_and_thirty_matrix_cells(self):
        calls = []

        def runner(config):
            calls.append(config)
            result = self._result(config)
            Path(config.output).parent.mkdir(parents=True, exist_ok=True)
            Path(config.output).write_text(json.dumps(result), encoding="utf-8")
            return result

        config = CalibrationConfig(output_dir=str(self.root), report_md=str(self.root / "report.md"), device="cpu")
        with patch("run_drift_recovery_calibration.write_diagnostics") as write_diagnostics:
            write_diagnostics.return_value = {"json": "diagnostics.json", "markdown": "diagnostics.md", "plots": []}
            manifest = run_calibration_campaign(config, runner=runner)

        self.assertEqual(len(calls), 39)
        scout_calls = [call for call in calls if Path(call.output).name.startswith("scout_")]
        matrix_calls = [call for call in calls if Path(call.output).name.startswith("scifact_")]
        self.assertEqual(len(scout_calls), 9)
        self.assertEqual(len(matrix_calls), 30)
        self.assertAlmostEqual(manifest["chosen_settings"]["rotation"]["drop_pct"], 10.0)
        self.assertAlmostEqual(manifest["chosen_settings"]["noise"]["drop_pct"], 11.0)
        self.assertTrue((self.root / "calibration-manifest-2026-06.json").exists())
        self.assertTrue((self.root / "report.md").exists())
        self.assertIn("Choice rule", (self.root / "report.md").read_text(encoding="utf-8"))

    def _result(self, config):
        baseline = 1.0
        if Path(config.output).name.startswith("scout_"):
            if config.drift == "rotation":
                final = 0.90 if config.angle == 50.0 and config.fraction == 0.5 else 0.97
            else:
                final = 0.89 if config.sigma == 0.02 else 0.97
        elif config.condition == "C2":
            final = 1.0
        elif config.drift == "rotation":
            final = 0.90
        else:
            final = 0.89
        metadata = {"condition": config.condition, "action": "none"}
        if config.condition in {"C3", "C4"}:
            metadata.update(
                {
                    "annealing_observation": {"drift_magnitude": 0.01, "should_correct": True},
                    "should_correct": True,
                    "sedimentation_attempted": True,
                    "correction_applied": False,
                    "correction_norm_stats": {"count": 1, "mean": 0.01, "max": 0.01, "sample_norms": [0.01]},
                }
            )
        return {
            "record_type": "drift_recovery_experiment",
            "config": {
                "task": config.task,
                "condition": config.condition,
                "drift": config.drift,
                "fraction": config.fraction,
                "angle": config.angle,
                "sigma": config.sigma,
                "cycles": config.cycles,
                "seed": config.seed,
                "max_queries": config.max_queries,
                "sample_docs": config.sample_docs,
                "output": config.output,
                "device": config.device,
            },
            "baseline": {"ndcg_at_10": baseline, "query_ndcg": []},
            "recovery": {
                "baseline_ndcg": baseline,
                "recovery_threshold": 0.95,
                "recovery_cycle": 1 if final >= 0.95 else None,
                "trajectory": [
                    {"cycle_index": cycle, "ndcg": final, "metadata": metadata}
                    for cycle in range(1, config.cycles + 1)
                ],
            },
            "correction_norm_stats": {"count": 0, "mean": None, "max": None, "sample_norms": []},
            "cycle_errors": [],
        }


if __name__ == "__main__":
    unittest.main()
