"""Tests for phase development loop handlers."""

from __future__ import annotations

import subprocess
from unittest import TestCase
from unittest.mock import patch

import scripts.phase_development_loop as loop


class TestSchedulerHandler(TestCase):
    @patch("scripts.phase_development_loop.subprocess.run")
    def test_scheduler_handler_uses_strict_flag(self, mock_run: object) -> None:
        mock_run.return_value = subprocess.CompletedProcess(
            args=["python", "scripts/record_shim_scheduler_evidence.py", "--strict"],
            returncode=0,
            stdout="",
            stderr="",
        )
        ok, _ = loop._run_shim_scheduler_evidence()
        self.assertTrue(ok)
        self.assertTrue(mock_run.called)
        called_cmd = mock_run.call_args[0][0]
        self.assertIn("--strict", called_cmd)

    @patch("scripts.phase_development_loop.parse_open_shim_cds")
    @patch("scripts.phase_development_loop.parse_block_flag")
    def test_recommendation_executes_cd01_sip_slice(self, mock_block: object, mock_open: object) -> None:
        mock_block.return_value = {
            "blocked": False,
            "exit_code": 0,
            "debt_rows": 8,
            "stdout_tail": "",
        }
        mock_open.return_value = [
            {
                "id": "SHIM-CD-01",
                "blocking": True,
                "summary": "CD-01 open",
            },
        ]

        state = {
            "completed": {
                "cd01_engine_embed_sip": "2026-06-03T00:00:00+00:00",
                "cd01_insert_once_tests": "2026-06-03T00:00:00+00:00",
            }
        }
        rec = loop.recommend_next(state, include_advisory=False)

        self.assertEqual(rec.get("primary_action"), "execute_handler")
        self.assertIsNotNone(rec.get("next_executable"))
        self.assertEqual(rec["next_executable"]["slice_id"], "SHIM-SLICE-CD01-SIP")
        self.assertIsNone(rec.get("next_advisory"))

    @patch("scripts.phase_development_loop.parse_open_shim_cds")
    @patch("scripts.phase_development_loop.parse_block_flag")
    @patch("scripts.phase_development_loop._is_cd06_scheduler_verified")
    def test_scheduler_stale_completion_is_not_treated_as_done(
        self,
        mock_verified: object,
        mock_block: object,
        mock_open: object,
    ) -> None:
        mock_verified.return_value = False
        mock_block.return_value = {
            "blocked": False,
            "exit_code": 0,
            "debt_rows": 8,
            "stdout_tail": "",
        }
        mock_open.return_value = []

        base_state = {
            "completed": {ws.completed_key: "2026-06-03T00:00:00+00:00" for ws in loop.build_slice_registry()}
        }
        # All slices are marked done except scheduler evidence must be revalidated
        # from the latest scheduler artifact.
        base_state["completed"]["cd06_scheduler_evidence"] = (
            "2026-06-03T00:00:00+00:00"
        )

        rec = loop.recommend_next(base_state, include_advisory=False)

        self.assertEqual(rec.get("primary_action"), "execute_handler")
        self.assertIsNotNone(rec.get("next_executable"))
        self.assertEqual(
            rec["next_executable"]["slice_id"],
            "SHIM-SLICE-SCHEDULER-06",
        )
