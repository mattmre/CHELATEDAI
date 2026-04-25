"""
Tests for the weight-refinement campaign runner resume flow.
"""

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


if "mteb" not in sys.modules:
    sys.modules["mteb"] = MagicMock()

if "sentence_transformers" not in sys.modules:
    sys.modules["sentence_transformers"] = MagicMock()

if "qdrant_client" not in sys.modules:
    sys.modules["qdrant_client"] = MagicMock()
    sys.modules["qdrant_client.models"] = MagicMock()

if "torch" not in sys.modules:
    sys.modules["torch"] = MagicMock()
    sys.modules["torch.nn"] = MagicMock()
    sys.modules["torch.nn.functional"] = MagicMock()
    sys.modules["torch.optim"] = MagicMock()

import run_weight_refinement_campaign as campaign


class TestCampaignResume(unittest.TestCase):
    def test_phase_is_complete(self):
        self.assertTrue(campaign.phase_is_complete({"returncode": 0}))
        self.assertTrue(campaign.phase_is_complete({"status": "completed"}))
        self.assertTrue(campaign.phase_is_complete({"status": "launched"}))
        self.assertFalse(campaign.phase_is_complete({"returncode": 1}))
        self.assertFalse(campaign.phase_is_complete(None))

    def test_resume_recovers_outputs_and_runs_only_missing_phase(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            manifest = {
                "started_at": "2026-03-06T13:42:14",
                "run_dir": str(run_dir),
                "config": {"model": "sentence-transformers/all-MiniLM-L6-v2"},
                "phases": {},
                "baseline_adapter_snapshot": None,
            }
            (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

            existing_outputs = [
                "phase1_sweep_results.json",
                "phase2_distillation_mlp_tw_03.json",
                "phase2_distillation_mlp_tw_05.json",
                "phase2_distillation_mlp_tw_07.json",
                "phase3_multitask_small.json",
                "phase3_multitask_medium.json",
                "phase4_beir_small.json",
            ]
            for name in existing_outputs:
                (run_dir / name).write_text("{}", encoding="utf-8")

            run_calls = []

            def fake_run_command(label, command, local_run_dir):
                run_calls.append((label, command, local_run_dir))
                return {
                    "label": label,
                    "command": command,
                    "returncode": 0,
                    "started_at": "2026-03-06T16:00:00",
                    "finished_at": "2026-03-06T16:10:00",
                }

            with patch.object(campaign, "run_command", side_effect=fake_run_command), \
                patch.object(campaign, "run_online_ablation", return_value={
                    "output_path": str(run_dir / "phase5_online_ablation.json"),
                    "best_config": "baseline",
                    "best_ndcg_at_10": 0.75,
                }), \
                patch.object(campaign, "launch_background_command", return_value={
                    "label": "phase6_large_sweep",
                    "status": "launched",
                    "pid": 12345,
                    "log_path": str(run_dir / "logs" / "phase6_large_sweep.log"),
                }), \
                patch.object(campaign, "restore_adapter"), \
                patch.object(campaign, "snapshot_adapter", return_value=None), \
                patch.object(campaign, "summarize_run"), \
                patch.object(sys, "argv", [
                    "run_weight_refinement_campaign.py",
                    "--resume-run-dir",
                    str(run_dir),
                    "--launch-large-sweep",
                ]):
                exit_code = campaign.main()

            self.assertEqual(exit_code, 0)
            self.assertEqual([call[0] for call in run_calls], ["phase4_beir_medium"])

            resumed_manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(
                resumed_manifest["phases"]["phase1_standard_sweep"]["status"],
                "recovered_from_output",
            )
            self.assertEqual(
                resumed_manifest["phases"]["phase4_beir_medium"]["returncode"],
                0,
            )
            self.assertEqual(
                resumed_manifest["phases"]["phase5_online_ablation"]["best_config"],
                "baseline",
            )
            self.assertEqual(
                resumed_manifest["phases"]["phase6_large_sweep"]["status"],
                "launched",
            )

    def test_resume_preserves_manifest_config_when_cli_omits_overrides(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            manifest = {
                "started_at": "2026-04-24T21:08:46",
                "run_dir": str(run_dir),
                "config": {
                    "run_dir": str(run_dir),
                    "model": "sentence-transformers/all-MiniLM-L6-v2",
                    "teacher": "sentence-transformers/all-mpnet-base-v2",
                    "max_queries": 100,
                    "distill_queries_per_cycle": 50,
                    "distill_cycles": 5,
                    "distill_epochs": 5,
                    "learning_rate": 0.01,
                    "adapter_types": "mlp",
                    "launch_large_sweep": False,
                },
                "phases": {
                    "phase1_standard_sweep": {"returncode": 0},
                    "phase2_distillation_mlp_tw_03": {"returncode": 0},
                },
                "baseline_adapter_snapshot": None,
            }
            (run_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
            (run_dir / "phase1_sweep_results.json").write_text("{}", encoding="utf-8")
            (run_dir / "phase2_distillation_mlp_tw_03.json").write_text("{}", encoding="utf-8")
            run_calls = []

            def fake_run_command(label, command, local_run_dir):
                run_calls.append((label, command, local_run_dir))
                return {
                    "label": label,
                    "command": command,
                    "returncode": 0,
                    "started_at": "2026-04-24T23:45:00",
                    "finished_at": "2026-04-24T23:46:00",
                }

            with patch.object(campaign, "run_command", side_effect=fake_run_command), \
                patch.object(campaign, "run_online_ablation", return_value={
                    "output_path": str(run_dir / "phase5_online_ablation.json"),
                    "best_config": "baseline",
                    "best_ndcg_at_10": 0.75,
                }), \
                patch.object(campaign, "launch_background_command", return_value={
                    "label": "phase6_large_sweep",
                    "status": "not_launched",
                    "command": [],
                }), \
                patch.object(campaign, "restore_adapter"), \
                patch.object(campaign, "snapshot_adapter", return_value=None), \
                patch.object(campaign, "summarize_run"), \
                patch.object(sys, "argv", [
                    "run_weight_refinement_campaign.py",
                    "--resume-run-dir",
                    str(run_dir),
                ]):
                exit_code = campaign.main()

            self.assertEqual(exit_code, 0)
            first_phase2_call = next(call for call in run_calls if call[0] == "phase2_distillation_mlp_tw_05")
            command = first_phase2_call[1]
            self.assertEqual(command[command.index("--teacher") + 1], "sentence-transformers/all-mpnet-base-v2")
            self.assertEqual(command[command.index("--cycles") + 1], "5")
            self.assertEqual(command[command.index("--queries-per-cycle") + 1], "50")
            self.assertEqual(command[command.index("--epochs") + 1], "5")
            self.assertEqual(
                json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))["config"]["teacher"],
                "sentence-transformers/all-mpnet-base-v2",
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
