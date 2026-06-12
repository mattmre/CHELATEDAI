from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from run_drift_recovery_knob_sweep import KnobSweepConfig, _select_top_cells, run_knob_sweep


class TestRunDriftRecoveryKnobSweep(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.root = Path(self.tempdir.name)
        self.calibration_manifest = self.root / "calibration.json"
        self.calibration_manifest.write_text(
            json.dumps(
                {
                    "chosen_settings": {
                        "rotation": {
                            "fraction": 0.5,
                            "angle": 35.0,
                            "sigma": 0.05,
                        }
                    }
                }
            ),
            encoding="utf-8",
        )

    def test_knob_sweep_runs_twelve_grid_and_six_confirmation_cells(self):
        calls = []

        def runner(config):
            calls.append(config)
            result = self._result(config)
            Path(config.output).parent.mkdir(parents=True, exist_ok=True)
            Path(config.output).write_text(json.dumps(result), encoding="utf-8")
            return result

        config = KnobSweepConfig(
            calibration_manifest=str(self.calibration_manifest),
            output_dir=str(self.root / "sweep"),
            report_md=str(self.root / "report.md"),
            device="cpu",
        )
        with patch("run_drift_recovery_knob_sweep._load_calibrated_references") as refs:
            refs.return_value = {
                "C0": {"run_count": 3, "final_mean": 0.76, "final_std": 0.01, "recovered_runs": 1},
                "C2": {"run_count": 3, "final_mean": 0.83, "final_std": 0.01, "recovered_runs": 3},
            }
            manifest = run_knob_sweep(config, runner=runner)

        self.assertEqual(len(calls), 18)
        self.assertEqual(len(manifest["grid_rows"]), 12)
        self.assertEqual(len(manifest["top_cells"]), 2)
        self.assertEqual(len(manifest["confirmation_rows"]), 6)
        self.assertEqual(len(manifest["confirmation_summary"]), 2)
        refs.assert_called_once_with(str(self.calibration_manifest))
        self.assertIn("ties are broken", manifest["selection_policy"])
        self.assertTrue((self.root / "sweep" / "knob-sweep-manifest-2026-06.json").exists())
        report = (self.root / "report.md").read_text(encoding="utf-8")
        self.assertIn("## Grid Cells", report)
        self.assertIn("## Three-Seed Confirmation", report)
        self.assertIn("Selection policy", report)

    def test_top_cell_ties_use_deterministic_disclosed_order(self):
        rows = [
            {"final_ndcg": 0.9, "trigger_threshold": 0.15, "profile": "hotter", "bound_epsilon": 0.10},
            {"final_ndcg": 0.9, "trigger_threshold": 0.05, "profile": "hotter", "bound_epsilon": 0.10},
            {"final_ndcg": 0.9, "trigger_threshold": 0.05, "profile": "default", "bound_epsilon": 0.10},
            {"final_ndcg": 0.8, "trigger_threshold": 0.05, "profile": "default", "bound_epsilon": 0.01},
        ]

        selected = _select_top_cells(rows)

        self.assertEqual(
            [(row["trigger_threshold"], row["profile"]) for row in selected],
            [(0.05, "default"), (0.05, "hotter")],
        )

    def _result(self, config):
        baseline = 1.0
        final = 0.70 + config.bound_epsilon
        if config.trigger_threshold == 0.05:
            final += 0.01
        if config.epochs_scale > 1.0:
            final += 0.005
        metadata = {
            "condition": "C3",
            "should_correct": config.trigger_threshold < 0.10,
            "sedimentation_attempted": config.trigger_threshold < 0.10,
            "correction_applied": False,
            "annealing_observation": {"drift_magnitude": 0.02},
            "correction_norm_stats": {
                "count": 1,
                "mean": config.bound_epsilon,
                "max": config.bound_epsilon,
                "sample_norms": [config.bound_epsilon],
            },
            "knobs": {
                "bound_epsilon": config.bound_epsilon,
                "trigger_threshold": config.trigger_threshold,
                "max_temperature": config.max_temperature,
                "epochs_scale": config.epochs_scale,
            },
        }
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
                "bound_epsilon": config.bound_epsilon,
                "trigger_threshold": config.trigger_threshold,
                "max_temperature": config.max_temperature,
                "epochs_scale": config.epochs_scale,
            },
            "baseline": {"ndcg_at_10": baseline, "query_ndcg": []},
            "recovery": {
                "baseline_ndcg": baseline,
                "recovery_threshold": 0.95,
                "recovery_cycle": None,
                "trajectory": [
                    {"cycle_index": cycle, "ndcg": final, "metadata": metadata}
                    for cycle in range(1, config.cycles + 1)
                ],
            },
            "correction_norm_stats": {
                "count": 1,
                "mean": config.bound_epsilon,
                "max": config.bound_epsilon,
                "sample_norms": [config.bound_epsilon],
            },
            "cycle_errors": [],
        }


if __name__ == "__main__":
    unittest.main()
