"""
Tests for the overnight campaign wrapper.
"""

import sys
import unittest
from datetime import datetime as real_datetime
from types import SimpleNamespace
from unittest.mock import patch

import run_overnight_campaign as overnight


class FixedDateTime:
    @classmethod
    def now(cls):
        return real_datetime(2026, 4, 24, 21, 0, 0)


class TestRunOvernightCampaign(unittest.TestCase):
    def test_default_run_label_uses_overnight_suffix(self):
        with patch.object(overnight, "datetime", FixedDateTime), patch.object(
            overnight.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=0),
        ) as mock_run, patch.object(
            sys,
            "argv",
            ["run_overnight_campaign.py"],
        ):
            exit_code = overnight.main()

        self.assertEqual(exit_code, 0)
        command = mock_run.call_args.args[0]
        self.assertIn(
            str(
                overnight.PROJECT_ROOT
                / "experiment_runs"
                / "weight-refinement-20260424-210000-000000-overnight"
            ),
            command,
        )

    def test_custom_run_label_is_normalized_for_run_directory(self):
        with patch.object(overnight, "datetime", FixedDateTime), patch.object(
            overnight.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=0),
        ) as mock_run, patch.object(
            sys,
            "argv",
            ["run_overnight_campaign.py", "--run-label", "Session 32 Review"],
        ):
            exit_code = overnight.main()

        self.assertEqual(exit_code, 0)
        command = mock_run.call_args.args[0]
        self.assertIn(
            str(
                overnight.PROJECT_ROOT
                / "experiment_runs"
                / "weight-refinement-20260424-210000-000000-session-32-review"
            ),
            command,
        )

    def test_long_run_label_is_truncated_for_windows_safe_paths(self):
        long_label = "Session32-" + ("VeryLongLabel" * 20)

        with patch.object(overnight, "datetime", FixedDateTime), patch.object(
            overnight.subprocess,
            "run",
            return_value=SimpleNamespace(returncode=0),
        ) as mock_run, patch.object(
            sys,
            "argv",
            ["run_overnight_campaign.py", "--run-label", long_label],
        ):
            exit_code = overnight.main()

        self.assertEqual(exit_code, 0)
        command = mock_run.call_args.args[0]
        run_dir = next(
            arg for index, arg in enumerate(command) if command[index - 1] == "--run-dir"
        )
        label = run_dir.split("weight-refinement-20260424-210000-000000-", 1)[1]
        self.assertLessEqual(len(run_dir), 200)
        self.assertEqual(len(label), overnight.MAX_RUN_LABEL_LENGTH)
        self.assertTrue(label.startswith("session32-verylonglabel"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
