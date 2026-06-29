"""Tests for the H5 post-bank head-to-head campaign driver (run_condition_head_to_head).
Runner-injected (no GPU, no engine): a stub runner returns controlled per-condition
finals so the driver's orchestration + aggregation + verdict logic is verified."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from run_drift_recovery_swap_campaign import (
    POSTBANK_HEADTOHEAD_CONDITIONS,
    SEEDS,
    SwapCampaignConfig,
    run_condition_head_to_head,
)


def _make_stub_runner(finals):
    """Return a runner that fabricates a minimal result dict per DriftRecoveryConfig,
    with the given per-condition final NDCG (matching what _row() reads)."""
    def runner(run_config):
        cond = run_config.condition
        final = finals[cond]
        applied = cond not in ("C0", "C2", "C2O")  # only the correcting conditions mutate
        return {
            "config": {
                "condition": cond, "seed": run_config.seed, "output": run_config.output,
                "correction_steps": run_config.correction_steps,
                "correction_lr": run_config.correction_lr,
            },
            "recovery": {
                "trajectory": [{"ndcg": final, "metadata": {
                    "should_correct": applied, "correction_applied": applied}}],
                "recovery_cycle": 1 if applied else None,
            },
            "baseline": {"ndcg_at_10": 0.8},
            "correction_norm_stats": {"mean": 0.1 if applied else 0.0},
            "anchor_eval_split": {"anchor_count": 10, "eval_count": 60},
        }
    return runner


class TestPostBankHeadToHead(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)

    def _config(self):
        return SwapCampaignConfig(
            output_dir=self.tmp.name,
            report_md=str(Path(self.tmp.name) / "report.md"),
            cycles=2, sample_docs=8, max_queries=8,
        )

    def test_living_wins_verdict(self):
        finals = {"C0": 0.01, "C2O": 0.8, "C5": 0.40, "C5s": 0.30, "C5r": 0.35}  # C5 beats both
        m = run_condition_head_to_head(self._config(), runner=_make_stub_runner(finals))
        v = m["verdict"]
        self.assertTrue(v["living_beats_static"])
        self.assertTrue(v["living_beats_one_shot"])
        self.assertTrue(v["all_post_bank_present"])
        self.assertTrue(v["living_bank_wins"])
        self.assertAlmostEqual(v["c5_living_mean"], 0.40)

    def test_living_loses_to_static(self):
        finals = {"C0": 0.01, "C2O": 0.8, "C5": 0.25, "C5s": 0.30, "C5r": 0.20}  # C5 < C5s
        v = run_condition_head_to_head(self._config(), runner=_make_stub_runner(finals))["verdict"]
        self.assertFalse(v["living_beats_static"])
        self.assertTrue(v["living_beats_one_shot"])   # still beats the one-shot...
        self.assertFalse(v["living_bank_wins"])       # ...but NOT both -> gate fails

    def test_runs_all_conditions_times_seeds(self):
        finals = {"C0": 0.01, "C2O": 0.8, "C5": 0.40, "C5s": 0.30, "C5r": 0.35}
        m = run_condition_head_to_head(self._config(), runner=_make_stub_runner(finals))
        self.assertEqual(len(m["rows"]), len(POSTBANK_HEADTOHEAD_CONDITIONS) * len(SEEDS))

    def test_manifest_and_report_written_with_verdict(self):
        finals = {"C0": 0.01, "C2O": 0.8, "C5": 0.40, "C5s": 0.30, "C5r": 0.35}
        run_condition_head_to_head(self._config(), runner=_make_stub_runner(finals))
        self.assertTrue((Path(self.tmp.name) / "post-bank-headtohead-manifest-2026-06.json").exists())
        report = (Path(self.tmp.name) / "report.md").read_text(encoding="utf-8")
        self.assertIn("LIVING BANK WINS (beats both): True", report)
        self.assertIn("C5 living bank", report)

    def test_c3b_subset_has_no_post_bank_verdict(self):
        # The same driver serves an H3 C3b head-to-head; the post-bank verdict fields
        # are None/False because C5/C5s/C5r are absent.
        finals = {"C0": 0.01, "C2O": 0.8, "C3a": 0.16, "C3b": 0.15}
        v = run_condition_head_to_head(
            self._config(), conditions=("C0", "C2O", "C3a", "C3b"),
            runner=_make_stub_runner(finals),
        )["verdict"]
        self.assertFalse(v["all_post_bank_present"])
        self.assertFalse(v["living_bank_wins"])
        self.assertIsNone(v["c5_living_mean"])


if __name__ == "__main__":
    unittest.main()
