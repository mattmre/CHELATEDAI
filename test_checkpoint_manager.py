"""
Unit Tests for checkpoint_manager.py

Tests the CheckpointManager (create, restore, list, delete, cleanup)
and SafeTrainingContext (success path, rollback on exception, rollback
without mark_success, and auto_rollback=False).
"""

import unittest
import tempfile
import shutil
import time
import torch
from pathlib import Path

from checkpoint_manager import CheckpointManager, SafeTrainingContext


# ===========================================================================
# CheckpointManager Tests
# ===========================================================================


class TestCheckpointManager(unittest.TestCase):
    """Tests for the CheckpointManager class."""

    def setUp(self):
        """Create temp directory, dummy adapter file, and a CheckpointManager."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.adapter_path = self.temp_dir / "adapter.pt"
        torch.save({"layer": torch.zeros(10)}, self.adapter_path)
        self.mgr = CheckpointManager(checkpoint_dir=self.temp_dir / "checkpoints")

    def tearDown(self):
        """Remove temp directory and all contents."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init_creates_dir(self):
        """Checkpoint directory is created during __init__."""
        self.assertTrue((self.temp_dir / "checkpoints").exists())
        self.assertTrue((self.temp_dir / "checkpoints").is_dir())

    def test_create_returns_id(self):
        """create_checkpoint returns a string ID starting with the name prefix."""
        cp_id = self.mgr.create_checkpoint("test_backup", self.adapter_path)
        self.assertIsInstance(cp_id, str)
        self.assertTrue(cp_id.startswith("test_backup_"))

    def test_create_copies_file(self):
        """create_checkpoint copies adapter_weights.pt into the checkpoint dir."""
        cp_id = self.mgr.create_checkpoint("copy_test", self.adapter_path)
        checkpoint_path = self.temp_dir / "checkpoints" / cp_id / "adapter_weights.pt"
        self.assertTrue(checkpoint_path.exists())

    def test_create_metadata_stored(self):
        """After one create, metadata has exactly 1 checkpoint entry."""
        self.mgr.create_checkpoint("meta_test", self.adapter_path)
        self.assertEqual(len(self.mgr.metadata["checkpoints"]), 1)
        self.assertEqual(self.mgr.metadata["checkpoints"][0]["name"], "meta_test")

    def test_restore_latest(self):
        """Overwriting adapter then restoring latest recovers original data."""
        original_data = torch.load(self.adapter_path, weights_only=False)
        self.mgr.create_checkpoint("before_change", self.adapter_path)

        # Overwrite adapter with different data
        torch.save({"layer": torch.ones(10)}, self.adapter_path)
        overwritten = torch.load(self.adapter_path, weights_only=False)
        self.assertTrue(torch.all(overwritten["layer"] == 1.0))

        # Restore latest
        success = self.mgr.restore_checkpoint(target_adapter_path=self.adapter_path)
        self.assertTrue(success)

        restored = torch.load(self.adapter_path, weights_only=False)
        self.assertTrue(torch.equal(restored["layer"], original_data["layer"]))

    def test_restore_specific_id(self):
        """Creating 2 checkpoints and restoring the first one recovers that state."""
        # First state
        torch.save({"layer": torch.full((10,), 1.0)}, self.adapter_path)
        cp1 = self.mgr.create_checkpoint("state_1", self.adapter_path)

        # Second state -- add small delay to ensure different checkpoint ID
        time.sleep(1.1)
        torch.save({"layer": torch.full((10,), 2.0)}, self.adapter_path)
        self.mgr.create_checkpoint("state_2", self.adapter_path)

        # Overwrite with a third value
        torch.save({"layer": torch.full((10,), 3.0)}, self.adapter_path)

        # Restore the first checkpoint specifically
        success = self.mgr.restore_checkpoint(
            checkpoint_id=cp1,
            target_adapter_path=self.adapter_path,
        )
        self.assertTrue(success)

        restored = torch.load(self.adapter_path, weights_only=False)
        self.assertTrue(torch.all(restored["layer"] == 1.0))

    def test_restore_invalid_id(self):
        """Restoring a nonexistent checkpoint ID returns False."""
        result = self.mgr.restore_checkpoint(checkpoint_id="nonexistent_id_12345")
        self.assertFalse(result)

    def test_restore_no_checkpoints(self):
        """Restoring with no checkpoints available returns False."""
        result = self.mgr.restore_checkpoint()
        self.assertFalse(result)

    def test_list_empty(self):
        """Fresh manager with no checkpoints lists empty."""
        self.assertEqual(self.mgr.list_checkpoints(), [])

    def test_list_after_creates(self):
        """After creating N checkpoints, list returns N entries."""
        self.mgr.create_checkpoint("a", self.adapter_path)
        time.sleep(1.1)
        self.mgr.create_checkpoint("b", self.adapter_path)
        time.sleep(1.1)
        self.mgr.create_checkpoint("c", self.adapter_path)
        self.assertEqual(len(self.mgr.list_checkpoints()), 3)

    def test_delete_existing(self):
        """Deleting an existing checkpoint returns True and shrinks the list."""
        cp_id = self.mgr.create_checkpoint("deleteme", self.adapter_path)
        self.assertEqual(len(self.mgr.list_checkpoints()), 1)

        result = self.mgr.delete_checkpoint(cp_id)
        self.assertTrue(result)
        self.assertEqual(len(self.mgr.list_checkpoints()), 0)

    def test_delete_nonexistent(self):
        """Deleting a nonexistent checkpoint returns False."""
        result = self.mgr.delete_checkpoint("no_such_checkpoint")
        self.assertFalse(result)

    def test_cleanup_keeps_n(self):
        """Creating 5 checkpoints then cleanup(keep_last_n=2) leaves 2."""
        for i in range(5):
            self.mgr.create_checkpoint(f"cp_{i}", self.adapter_path)
            time.sleep(1.1)

        self.assertEqual(len(self.mgr.list_checkpoints()), 5)
        self.mgr.cleanup_old_checkpoints(keep_last_n=2)
        self.assertEqual(len(self.mgr.list_checkpoints()), 2)


# ===========================================================================
# SafeTrainingContext Tests
# ===========================================================================


class TestSafeTrainingContext(unittest.TestCase):
    """Tests for the SafeTrainingContext context manager."""

    def setUp(self):
        """Create temp directory, dummy adapter file, and CheckpointManager."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.adapter_path = self.temp_dir / "adapter.pt"
        torch.save({"layer": torch.zeros(10)}, self.adapter_path)
        self.mgr = CheckpointManager(checkpoint_dir=self.temp_dir / "checkpoints")

    def tearDown(self):
        """Remove temp directory and all contents."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_success_no_rollback(self):
        """On success with mark_success, new data persists (no rollback)."""
        with SafeTrainingContext(self.mgr, self.adapter_path, "good_training") as ctx:
            torch.save({"layer": torch.ones(10)}, self.adapter_path)
            ctx.mark_success()

        restored = torch.load(self.adapter_path, weights_only=False)
        self.assertTrue(torch.all(restored["layer"] == 1.0))

    def test_rollback_on_exception(self):
        """On exception, original data is restored via rollback."""
        original_data = torch.load(self.adapter_path, weights_only=False)

        try:
            with SafeTrainingContext(self.mgr, self.adapter_path, "bad_training") as ctx:
                torch.save({"layer": torch.full((10,), 99.0)}, self.adapter_path)
                raise ValueError("Training exploded")
        except ValueError:
            pass

        restored = torch.load(self.adapter_path, weights_only=False)
        self.assertTrue(torch.equal(restored["layer"], original_data["layer"]))

    def test_rollback_without_mark_success(self):
        """Exiting normally without mark_success triggers rollback."""
        original_data = torch.load(self.adapter_path, weights_only=False)

        with SafeTrainingContext(self.mgr, self.adapter_path, "forgot_mark") as ctx:
            torch.save({"layer": torch.full((10,), 42.0)}, self.adapter_path)
            # Intentionally do NOT call ctx.mark_success()

        restored = torch.load(self.adapter_path, weights_only=False)
        self.assertTrue(torch.equal(restored["layer"], original_data["layer"]))

    def test_no_rollback_when_disabled(self):
        """With auto_rollback=False, bad data persists even on exception."""
        try:
            with SafeTrainingContext(
                self.mgr, self.adapter_path, "no_rollback", auto_rollback=False
            ) as ctx:
                torch.save({"layer": torch.full((10,), 77.0)}, self.adapter_path)
                raise ValueError("Training failed but no rollback")
        except ValueError:
            pass

        restored = torch.load(self.adapter_path, weights_only=False)
        self.assertTrue(torch.all(restored["layer"] == 77.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
