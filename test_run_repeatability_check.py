"""
Tests for the focused repeatability-check wrapper.
"""

import os
import unittest
from datetime import datetime as real_datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import run_repeatability_check as repeatability


class FixedDateTime:
    @classmethod
    def now(cls):
        return real_datetime(2026, 4, 25, 8, 30, 0)


class TestRunRepeatabilityCheck(unittest.TestCase):
    def _make_args(self):
        return SimpleNamespace(
            task="SciFact",
            model="sentence-transformers/all-MiniLM-L6-v2",
            teacher="sentence-transformers/all-mpnet-base-v2",
            cycles=5,
            queries_per_cycle=50,
            epochs=5,
            learning_rate=0.01,
            max_eval_queries=100,
            teacher_weight=0.3,
            threshold=1,
            adapter_type="mlp",
            reference_baseline_final=0.6012,
            reference_hybrid_final=0.6239,
            reference_baseline_tolerance=0.03,
        )

    def test_build_run_dir_uses_normalized_label(self):
        run_dir = repeatability.build_run_dir(
            Path("experiment_runs"),
            "Session 33 / MLP TW03",
            now=FixedDateTime.now(),
        )
        self.assertEqual(
            str(run_dir),
            str(
                Path("experiment_runs")
                / "repeatability-20260425-083000-000000-session-33-mlp-tw03"
            ),
        )

    def test_build_command_uses_expected_defaults(self):
        args = self._make_args()
        output_path = Path("experiment_runs") / "repeatability" / "results.json"

        command = repeatability.build_command(output_path, args)

        self.assertEqual(command[:3], [repeatability.sys.executable, "-u", "benchmark_distillation.py"])
        self.assertEqual(command[command.index("--teacher-weight") + 1], "0.3")
        self.assertEqual(command[command.index("--adapter-type") + 1], "mlp")
        self.assertEqual(command[command.index("--output") + 1], str(output_path))

    def test_build_summary_passes_when_hybrid_beats_baseline_in_reference_band(self):
        args = self._make_args()
        results = {
            "baseline": [{"cycle": 5, "ndcg": 0.6030}],
            "offline": {"cycles": [{"cycle": 5, "ndcg": 0.0110}]},
            "hybrid": [{"cycle": 5, "ndcg": 0.6245}],
        }

        summary = repeatability.build_summary(
            results,
            Path("results.json"),
            Path("benchmark_distillation.log"),
            ["python", "benchmark_distillation.py"],
            args,
        )

        self.assertTrue(summary["baseline_matches_reference_band"])
        self.assertTrue(summary["hybrid_beats_same_run_baseline"])
        self.assertTrue(summary["passes_repeatability_gate"])
        self.assertEqual(summary["recommended_next_step"], "run-multitask-gate")

    def test_build_summary_fails_when_hybrid_loses_to_baseline(self):
        args = self._make_args()
        results = {
            "baseline": [{"cycle": 5, "ndcg": 0.6030}],
            "offline": {"cycles": [{"cycle": 5, "ndcg": 0.0110}]},
            "hybrid": [{"cycle": 5, "ndcg": 0.5900}],
        }

        summary = repeatability.build_summary(
            results,
            Path("results.json"),
            Path("benchmark_distillation.log"),
            ["python", "benchmark_distillation.py"],
            args,
        )

        self.assertFalse(summary["hybrid_beats_same_run_baseline"])
        self.assertFalse(summary["passes_repeatability_gate"])
        self.assertEqual(summary["recommended_next_step"], "stop-and-review")

    def test_extract_final_ndcg_rejects_incomplete_results(self):
        with self.assertRaisesRegex(ValueError, "missing completed cycle data for 'hybrid'"):
            repeatability._extract_final_ndcg({}, "hybrid")

    @patch("run_repeatability_check.subprocess.Popen")
    def test_run_with_tee_forces_utf8_subprocess_environment(self, mock_popen):
        process = MagicMock()
        process.stdout = iter(["line one\n"])
        process.wait.return_value = 0
        mock_popen.return_value = process

        log_path = Path("repeatability.log")
        log_handle = MagicMock()
        context_manager = MagicMock()
        context_manager.__enter__.return_value = log_handle
        context_manager.__exit__.return_value = False

        with patch.object(Path, "open", return_value=context_manager):
            repeatability.run_with_tee(["python", "benchmark_distillation.py"], log_path, Path("."))

        env = mock_popen.call_args.kwargs["env"]
        self.assertEqual(env["PYTHONIOENCODING"], "utf-8")
        self.assertEqual(env["PYTHONUTF8"], "1")
        self.assertEqual(env["PATH"], os.environ["PATH"])
        log_handle.write.assert_called_once_with("line one\n")
        log_handle.flush.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
