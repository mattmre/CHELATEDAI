"""Tests for large sweep execution behavior and I/O path."""

from __future__ import annotations

import tempfile
from pathlib import Path
import unittest
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from run_large_sweep import run_large_parameter_sweep


class TestRunLargeSweep(unittest.TestCase):
    """Validate bounded JSON checkpoint behavior and result structure."""

    def test_checkpointed_json_writes_are_bounded(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            out_prefix = Path(tmpdir) / "large_sweep_case"

            eval_values = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]

            fake_engine = SimpleNamespace(
                vector_size=4,
                chelation_log=[],
                adapter=MagicMock(),
                run_sedimentation_cycle=MagicMock(),
            )

            def fake_load_mteb_data(task_name):
                return {"a": "b"}, {"q": "x"}, {"q": {"a": 1}}

            fake_base = MagicMock()
            fake_base.vector_size = 4
            fake_base.run_sedimentation_cycle = fake_engine.run_sedimentation_cycle
            fake_base.chelation_log = fake_engine.chelation_log
            fake_base.adapter = fake_engine.adapter

            with patch("run_large_sweep.load_mteb_data", side_effect=fake_load_mteb_data), \
                 patch("run_large_sweep.evaluate_ndcg", side_effect=eval_values), \
                 patch("run_large_sweep.AntigravityEngine", return_value=fake_base), \
                 patch("chelation_adapter.create_adapter", return_value=MagicMock()) as mock_create_adapter, \
                 patch("run_large_sweep._write_json_results") as mock_write:

                run_large_parameter_sweep(
                    task_name="DummyTask",
                    model_name="dummy-model",
                    output_prefix=str(out_prefix),
                    max_queries=2,
                    db_path=str(Path(tmpdir) / "db"),
                    checkpoint_every=2,
                    learning_rates=[0.001],
                    thresholds=[1, 2],
                    noise_scales=[0.0],
                    epochs_list=[1],
                    push_magnitudes=[0.1, 0.2],
                )

            # 2*? combos => 4 result rows with 2 periodic checkpoints + final flush
            self.assertEqual(mock_create_adapter.call_count, 4)
            self.assertEqual(mock_write.call_count, 3)

            self.assertEqual(mock_write.call_count, 3)
            self.assertEqual(len(mock_write.call_args_list[-1].args[1]), 4)


if __name__ == "__main__":
    unittest.main()
