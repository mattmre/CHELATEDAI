"""Tests for 10-minute priority BHS loop hardening."""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path
from typing import Any
from unittest import TestCase
from unittest.mock import patch

import scripts.run_10min_priority_bhs_loop as bhs_loop


class Test10MinLoopSchedulerCommand(TestCase):
    def test_shim_scheduler_command_uses_strict(self) -> None:
        calls: list[tuple[tuple[Any, ...], int | None]] = []

        def fake_run(
            cmd: list[str],
            timeout: int = 600,
            **kwargs: Any,
        ) -> subprocess.CompletedProcess[str]:
            calls.append((tuple(cmd), timeout))
            return subprocess.CompletedProcess(
                cmd,
                returncode=0,
                stdout="",
                stderr="",
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            patcher_artifact_dir = patch(
                "scripts.run_10min_priority_bhs_loop.ARTIFACT_DIR",
                Path(tmpdir),
            )
            with patcher_artifact_dir:
                with patch(
                    "scripts.run_10min_priority_bhs_loop._run",
                    side_effect=fake_run,
                ):
                    with patch(
                        "scripts.run_10min_priority_bhs_loop._bhs_smoke_gate",
                        return_value={
                            "floor": {"passed": True, "exit_code": 0},
                            "ceiling": {"passed": True, "exit_code": 0},
                        },
                    ):
                        with patch(
                            "scripts.run_10min_priority_bhs_loop._bhs_audit",
                            return_value={"exit_code": 0},
                        ):
                            result = bhs_loop._run_bhs_improvement_cycle(
                                bhs_target=100,
                                cycle_index=1,
                            )
        scheduler_cmds = [
            cmd
            for cmd, _ in calls
            if any("record_shim_scheduler_evidence.py" in str(part) for part in cmd)
        ]
        self.assertEqual(len(scheduler_cmds), 1)
        self.assertIn("--strict", scheduler_cmds[0])
        self.assertTrue(result["all_steps_ok"])
        self.assertEqual(result["steps"][len(result["steps"]) - 2]["step"], "shim_scheduler")
        self.assertTrue(result["cycle_ok"])
