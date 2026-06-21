from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from run_drift_recovery_swap_campaign import (
    BUDGET_GRID,
    MAIN_CONDITIONS,
    SEEDS,
    SwapCampaignConfig,
    run_swap_campaign,
)


def _fake_result(run_config, *, final_by_condition):
    """Shape a fake run_experiment result for the driver under test.

    final NDCG is driven by condition so the matrix/oracle ordering is testable;
    the C3a budget sweep improves with more steps so best-budget selection is
    exercised deterministically.
    """
    condition = run_config.condition
    steps = run_config.correction_steps
    if condition == "C3a":
        # More steps -> higher final NDCG (so the sweep picks the largest budget).
        final = 0.50 + 0.0001 * steps
    else:
        final = final_by_condition[condition]
    applied = 1 if condition in {"C3a", "C4a"} else 0
    return {
        "config": {
            "condition": condition,
            "seed": run_config.seed,
            "correction_steps": run_config.correction_steps,
            "correction_lr": run_config.correction_lr,
            "output": run_config.output,
        },
        "baseline": {"ndcg_at_10": 1.0},
        "recovery": {
            "trajectory": [
                {"ndcg": final, "metadata": {"should_correct": bool(applied), "correction_applied": bool(applied)}}
            ],
            "recovery_cycle": 1 if final >= 0.95 else None,
        },
        "correction_norm_stats": {"mean": 0.13 if applied else 0.0},
        "anchor_eval_split": {"anchor_count": 2 if run_config.anchor_fraction > 0 else 0, "eval_count": 4},
    }


class TestSwapCampaign(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.calls = []

    def _runner(self):
        final_by_condition = {"C0": 0.50, "C2": 0.50, "C2O": 1.0, "C4a": 0.50}

        def runner(run_config):
            self.calls.append(run_config)
            return _fake_result(run_config, final_by_condition=final_by_condition)

        return runner

    def _config(self):
        return SwapCampaignConfig(
            output_dir=str(Path(self.tempdir.name) / "swap"),
            report_md=str(Path(self.tempdir.name) / "report.md"),
            cycles=1,
        )

    def test_campaign_runs_full_matrix_budget_and_confirmation(self):
        manifest = run_swap_campaign(self._config(), runner=self._runner())

        # Main matrix: every condition x every seed.
        self.assertEqual(len(manifest["main_rows"]), len(MAIN_CONDITIONS) * len(SEEDS))
        # Budget sweep: one row per grid cell at the sweep seed.
        self.assertEqual(len(manifest["budget_rows"]), len(BUDGET_GRID))
        # Confirmation: best budget across all seeds.
        self.assertEqual(len(manifest["budget_confirm_rows"]), len(SEEDS))
        # All C3a/C4a runs fired+wrote; C0/C2/C2O did not.
        for row in manifest["main_rows"]:
            if row["condition"] in {"C3a", "C4a"}:
                self.assertEqual(row["correction_applied_true"], 1)
            else:
                self.assertEqual(row["correction_applied_true"], 0)

    def test_oracle_beats_baseline_and_supervised_in_summary(self):
        manifest = run_swap_campaign(self._config(), runner=self._runner())
        summary = {row["condition"]: row for row in manifest["main_summary"]}
        # The honest ordering the arena is designed to surface: oracle recovers,
        # frozen/no-op/supervised do not (on this fake geometry).
        self.assertAlmostEqual(summary["C2O"]["final_mean"], 1.0)
        self.assertLess(summary["C0"]["final_mean"], summary["C2O"]["final_mean"])
        self.assertEqual(summary["C2"]["final_mean"], summary["C0"]["final_mean"])

    def test_best_budget_is_highest_step_cell(self):
        manifest = run_swap_campaign(self._config(), runner=self._runner())
        # Fake geometry: final improves with steps, so the 2000-step cell wins.
        max_steps = max(steps for steps, _ in BUDGET_GRID)
        self.assertEqual(manifest["best_budget"]["steps"], max_steps)

    def test_manifest_and_report_written(self):
        manifest = run_swap_campaign(self._config(), runner=self._runner())
        out_dir = Path(manifest["config"]["output_dir"])
        written = out_dir / "swap-campaign-manifest-2026-06.json"
        self.assertTrue(written.exists())
        reloaded = json.loads(written.read_text(encoding="utf-8"))
        self.assertEqual(reloaded["record_type"], "drift_recovery_swap_campaign")
        # Report is written AND renders content (not just an empty/stub file): the
        # main-matrix table must contain every condition and the budget sweep
        # header, so a broken render_swap_report cannot ship green.
        report_path = Path(manifest["config"]["report_md"])
        self.assertTrue(report_path.exists())
        report = report_path.read_text(encoding="utf-8")
        for condition in MAIN_CONDITIONS:
            self.assertIn(f"| {condition} |", report, msg=f"report missing main-matrix row for {condition}")
        self.assertIn("Training-Budget Sweep", report)
        self.assertIn("confirmation", report.lower())

    def test_all_runs_use_query_encoder_swap_and_anchor_fraction(self):
        run_swap_campaign(self._config(), runner=self._runner())
        self.assertTrue(self.calls)
        for run_config in self.calls:
            self.assertEqual(run_config.drift, "query_encoder_swap")
            self.assertGreater(run_config.anchor_fraction, 0.0)


if __name__ == "__main__":
    unittest.main()
