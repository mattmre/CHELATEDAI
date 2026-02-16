"""
Unit Tests for Checkpoint Manager

Tests CheckpointManager create/restore/delete/cleanup and SafeTrainingContext
automatic rollback behavior without requiring external services.
"""

import unittest
import tempfile
import shutil
import torch
from pathlib import Path
from unittest.mock import patch
from datetime import datetime

from checkpoint_manager import CheckpointManager, SafeTrainingContext


class TestCheckpointManagerCreate(unittest.TestCase):
    """Test CheckpointManager.create_checkpoint."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.checkpoint_dir = self.temp_dir / "checkpoints"
        self.adapter_path = self.temp_dir / "adapter_weights.pt"
        torch.save({"weight": torch.zeros(10)}, self.adapter_path)
        self.manager = CheckpointManager(self.checkpoint_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_create_checkpoint_returns_string_id(self):
        """create_checkpoint returns a non-empty string checkpoint ID."""
        cp_id = self.manager.create_checkpoint("test_cp", self.adapter_path)
        self.assertIsInstance(cp_id, str)
        self.assertTrue(len(cp_id) > 0)
        self.assertTrue(cp_id.startswith("test_cp_"))

    def test_create_checkpoint_creates_subdirectory(self):
        """A subdirectory named after the checkpoint ID is created under checkpoint_dir."""
        cp_id = self.manager.create_checkpoint("backup", self.adapter_path)
        cp_path = self.checkpoint_dir / cp_id
        self.assertTrue(cp_path.exists())
        self.assertTrue(cp_path.is_dir())

    def test_create_checkpoint_copies_adapter_weights(self):
        """The adapter weights file is copied into the checkpoint subdirectory."""
        cp_id = self.manager.create_checkpoint("backup", self.adapter_path)
        copied_file = self.checkpoint_dir / cp_id / "adapter_weights.pt"
        self.assertTrue(copied_file.exists())
        # Verify the copied file contains the same tensor data
        original = torch.load(self.adapter_path, weights_only=True)
        restored = torch.load(copied_file, weights_only=True)
        self.assertTrue(torch.equal(original["weight"], restored["weight"]))

    def test_create_checkpoint_saves_metadata_fields(self):
        """Metadata entry contains the expected fields after creating a checkpoint."""
        cp_id = self.manager.create_checkpoint(
            "my_backup", self.adapter_path, description="test description"
        )
        checkpoints = self.manager.list_checkpoints()
        self.assertEqual(len(checkpoints), 1)
        meta = checkpoints[0]
        self.assertEqual(meta["checkpoint_id"], cp_id)
        self.assertEqual(meta["name"], "my_backup")
        self.assertEqual(meta["description"], "test description")
        self.assertIn("timestamp", meta)
        self.assertIn("adapter_hash", meta)
        self.assertIsNotNone(meta["adapter_path"])
        self.assertEqual(meta["original_adapter_path"], str(self.adapter_path))

    def test_create_checkpoint_sets_latest_checkpoint(self):
        """The latest_checkpoint pointer in metadata is set to the new checkpoint."""
        cp_id = self.manager.create_checkpoint("backup", self.adapter_path)
        self.assertEqual(self.manager.metadata["latest_checkpoint"], cp_id)

    def test_create_checkpoint_with_missing_adapter_file(self):
        """Creating a checkpoint when the adapter file does not exist still succeeds
        but adapter_path in metadata is None."""
        missing_path = self.temp_dir / "nonexistent.pt"
        cp_id = self.manager.create_checkpoint("no_adapter", missing_path)
        self.assertIsInstance(cp_id, str)
        meta = self.manager.list_checkpoints()[-1]
        self.assertIsNone(meta["adapter_path"])

    def test_create_checkpoint_extra_metadata(self):
        """Extra keyword arguments are stored in the checkpoint metadata entry."""
        cp_id = self.manager.create_checkpoint(
            "extra", self.adapter_path, description="", learning_rate=0.01, epochs=5
        )
        meta = self.manager.list_checkpoints()[-1]
        self.assertEqual(meta["learning_rate"], 0.01)
        self.assertEqual(meta["epochs"], 5)


class TestCheckpointManagerRestore(unittest.TestCase):
    """Test CheckpointManager.restore_checkpoint."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.checkpoint_dir = self.temp_dir / "checkpoints"
        self.adapter_path = self.temp_dir / "adapter_weights.pt"
        self.original_tensor = torch.randn(10)
        torch.save({"weight": self.original_tensor}, self.adapter_path)
        self.manager = CheckpointManager(self.checkpoint_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_restore_latest_checkpoint(self):
        """Restoring without specifying an ID restores the latest checkpoint."""
        cp_id = self.manager.create_checkpoint("snap", self.adapter_path)
        # Overwrite the adapter file with different data
        torch.save({"weight": torch.ones(10)}, self.adapter_path)
        result = self.manager.restore_checkpoint()
        self.assertTrue(result)
        restored = torch.load(self.adapter_path, weights_only=True)
        self.assertTrue(torch.equal(restored["weight"], self.original_tensor))

    def test_restore_by_checkpoint_id(self):
        """Restoring with an explicit checkpoint_id restores that specific checkpoint."""
        cp_id = self.manager.create_checkpoint("snap", self.adapter_path)
        # Overwrite adapter
        torch.save({"weight": torch.ones(10)}, self.adapter_path)
        result = self.manager.restore_checkpoint(checkpoint_id=cp_id)
        self.assertTrue(result)
        restored = torch.load(self.adapter_path, weights_only=True)
        self.assertTrue(torch.equal(restored["weight"], self.original_tensor))

    def test_restore_to_custom_target_path(self):
        """Restoring to a custom target_adapter_path writes the file there."""
        cp_id = self.manager.create_checkpoint("snap", self.adapter_path)
        custom_target = self.temp_dir / "restored_adapter.pt"
        result = self.manager.restore_checkpoint(
            checkpoint_id=cp_id, target_adapter_path=custom_target
        )
        self.assertTrue(result)
        self.assertTrue(custom_target.exists())
        restored = torch.load(custom_target, weights_only=True)
        self.assertTrue(torch.equal(restored["weight"], self.original_tensor))

    def test_restore_with_no_checkpoints_returns_false(self):
        """Restoring when no checkpoints exist returns False."""
        result = self.manager.restore_checkpoint()
        self.assertFalse(result)

    def test_restore_with_invalid_checkpoint_id_returns_false(self):
        """Restoring with an ID that does not exist returns False."""
        self.manager.create_checkpoint("snap", self.adapter_path)
        result = self.manager.restore_checkpoint(checkpoint_id="nonexistent_id")
        self.assertFalse(result)

    def test_restore_with_hash_mismatch_blocks_by_default(self):
        """Hash mismatch should fail restore by default."""
        cp_id = self.manager.create_checkpoint("snap", self.adapter_path)
        # Tamper with the checkpoint file to cause a hash mismatch
        checkpoint_file = Path(self.manager.list_checkpoints()[-1]["adapter_path"])
        torch.save({"weight": torch.ones(10)}, checkpoint_file)
        # Overwrite original adapter
        torch.save({"weight": torch.zeros(5)}, self.adapter_path)
        result = self.manager.restore_checkpoint(checkpoint_id=cp_id)
        self.assertFalse(result)
        # Adapter should remain unchanged because restore was blocked.
        restored = torch.load(self.adapter_path, weights_only=True)
        self.assertTrue(torch.equal(restored["weight"], torch.zeros(5)))

    def test_restore_with_hash_mismatch_can_be_forced(self):
        """Hash mismatch can be overridden for emergency restore."""
        cp_id = self.manager.create_checkpoint("snap", self.adapter_path)
        checkpoint_file = Path(self.manager.list_checkpoints()[-1]["adapter_path"])
        torch.save({"weight": torch.ones(10)}, checkpoint_file)
        torch.save({"weight": torch.zeros(5)}, self.adapter_path)
        result = self.manager.restore_checkpoint(
            checkpoint_id=cp_id,
            allow_hash_mismatch=True
        )
        self.assertTrue(result)
        restored = torch.load(self.adapter_path, weights_only=True)
        self.assertTrue(torch.equal(restored["weight"], torch.ones(10)))


class TestCheckpointManagerList(unittest.TestCase):
    """Test CheckpointManager.list_checkpoints."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.checkpoint_dir = self.temp_dir / "checkpoints"
        self.adapter_path = self.temp_dir / "adapter_weights.pt"
        torch.save({"weight": torch.zeros(10)}, self.adapter_path)
        self.manager = CheckpointManager(self.checkpoint_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_list_empty_initially(self):
        """list_checkpoints returns an empty list when no checkpoints exist."""
        result = self.manager.list_checkpoints()
        self.assertEqual(result, [])

    def test_list_returns_all_created_checkpoints(self):
        """list_checkpoints returns metadata for all checkpoints that have been created."""
        id1 = self.manager.create_checkpoint("first", self.adapter_path)
        id2 = self.manager.create_checkpoint("second", self.adapter_path)
        id3 = self.manager.create_checkpoint("third", self.adapter_path)
        result = self.manager.list_checkpoints()
        self.assertEqual(len(result), 3)
        ids = [cp["checkpoint_id"] for cp in result]
        self.assertIn(id1, ids)
        self.assertIn(id2, ids)
        self.assertIn(id3, ids)


class TestCheckpointManagerDelete(unittest.TestCase):
    """Test CheckpointManager.delete_checkpoint."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.checkpoint_dir = self.temp_dir / "checkpoints"
        self.adapter_path = self.temp_dir / "adapter_weights.pt"
        torch.save({"weight": torch.zeros(10)}, self.adapter_path)
        self.manager = CheckpointManager(self.checkpoint_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_delete_removes_checkpoint_directory(self):
        """Deleting a checkpoint removes its subdirectory from disk."""
        cp_id = self.manager.create_checkpoint("to_delete", self.adapter_path)
        cp_path = self.checkpoint_dir / cp_id
        self.assertTrue(cp_path.exists())
        result = self.manager.delete_checkpoint(cp_id)
        self.assertTrue(result)
        self.assertFalse(cp_path.exists())

    def test_delete_removes_from_metadata(self):
        """Deleting a checkpoint removes its entry from the checkpoints list."""
        cp_id = self.manager.create_checkpoint("to_delete", self.adapter_path)
        self.assertEqual(len(self.manager.list_checkpoints()), 1)
        self.manager.delete_checkpoint(cp_id)
        self.assertEqual(len(self.manager.list_checkpoints()), 0)

    def test_delete_invalid_id_returns_false(self):
        """Deleting a non-existent checkpoint ID returns False."""
        result = self.manager.delete_checkpoint("no_such_checkpoint")
        self.assertFalse(result)

    @patch("checkpoint_manager.datetime")
    def test_delete_updates_latest_checkpoint_pointer(self, mock_dt):
        """When the latest checkpoint is deleted, the pointer updates to the
        previous checkpoint (or None if no checkpoints remain)."""
        mock_dt.now.return_value = datetime(2026, 1, 1, 0, 0, 0)
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        id1 = self.manager.create_checkpoint("first", self.adapter_path)
        mock_dt.now.return_value = datetime(2026, 1, 1, 0, 0, 1)
        id2 = self.manager.create_checkpoint("second", self.adapter_path)
        self.assertEqual(self.manager.metadata["latest_checkpoint"], id2)
        # Delete latest; pointer should fall back to first
        self.manager.delete_checkpoint(id2)
        self.assertEqual(self.manager.metadata["latest_checkpoint"], id1)
        # Delete remaining; pointer should become None
        self.manager.delete_checkpoint(id1)
        self.assertIsNone(self.manager.metadata["latest_checkpoint"])


class TestCheckpointManagerCleanup(unittest.TestCase):
    """Test CheckpointManager.cleanup_old_checkpoints."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.checkpoint_dir = self.temp_dir / "checkpoints"
        self.adapter_path = self.temp_dir / "adapter_weights.pt"
        torch.save({"weight": torch.zeros(10)}, self.adapter_path)
        self.manager = CheckpointManager(self.checkpoint_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    @patch("checkpoint_manager.datetime")
    def test_cleanup_keeps_only_n_most_recent(self, mock_dt):
        """cleanup_old_checkpoints removes oldest checkpoints, keeping only N."""
        mock_dt.side_effect = lambda *a, **kw: datetime(*a, **kw)
        ids = []
        for i in range(5):
            mock_dt.now.return_value = datetime(2026, 1, 1, 0, 0, i)
            cp_id = self.manager.create_checkpoint(f"cp_{i}", self.adapter_path)
            ids.append(cp_id)
        self.assertEqual(len(self.manager.list_checkpoints()), 5)
        self.manager.cleanup_old_checkpoints(keep_last_n=2)
        remaining = self.manager.list_checkpoints()
        self.assertEqual(len(remaining), 2)
        remaining_ids = [cp["checkpoint_id"] for cp in remaining]
        # The two most recent should survive
        self.assertIn(ids[-1], remaining_ids)
        self.assertIn(ids[-2], remaining_ids)

    def test_cleanup_noop_when_fewer_than_n(self):
        """cleanup_old_checkpoints does nothing when there are fewer than N checkpoints."""
        self.manager.create_checkpoint("only_one", self.adapter_path)
        self.manager.cleanup_old_checkpoints(keep_last_n=5)
        self.assertEqual(len(self.manager.list_checkpoints()), 1)


class TestCheckpointManagerPersistence(unittest.TestCase):
    """Test that checkpoint metadata persists across manager instances."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.checkpoint_dir = self.temp_dir / "checkpoints"
        self.adapter_path = self.temp_dir / "adapter_weights.pt"
        torch.save({"weight": torch.zeros(10)}, self.adapter_path)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_metadata_persists_across_instances(self):
        """A new CheckpointManager instance loads metadata saved by a previous one."""
        manager1 = CheckpointManager(self.checkpoint_dir)
        cp_id = manager1.create_checkpoint("persistent", self.adapter_path)
        # Create a fresh manager pointing at the same directory
        manager2 = CheckpointManager(self.checkpoint_dir)
        checkpoints = manager2.list_checkpoints()
        self.assertEqual(len(checkpoints), 1)
        self.assertEqual(checkpoints[0]["checkpoint_id"], cp_id)


class TestSafeTrainingContextSuccess(unittest.TestCase):
    """Test SafeTrainingContext when the operation succeeds."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.checkpoint_dir = self.temp_dir / "checkpoints"
        self.adapter_path = self.temp_dir / "adapter_weights.pt"
        self.original_tensor = torch.randn(10)
        torch.save({"weight": self.original_tensor}, self.adapter_path)
        self.manager = CheckpointManager(self.checkpoint_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_context_creates_checkpoint_on_enter(self):
        """Entering the context creates a checkpoint."""
        with SafeTrainingContext(self.manager, self.adapter_path, "op1") as ctx:
            self.assertIsNotNone(ctx.checkpoint_id)
            self.assertEqual(len(self.manager.list_checkpoints()), 1)
            ctx.mark_success()

    def test_success_path_keeps_modified_weights(self):
        """When mark_success() is called, modified adapter weights are kept (no rollback)."""
        new_tensor = torch.ones(10)
        with SafeTrainingContext(self.manager, self.adapter_path, "train") as ctx:
            torch.save({"weight": new_tensor}, self.adapter_path)
            ctx.mark_success()
        # After exiting, the modified weights should remain
        loaded = torch.load(self.adapter_path, weights_only=True)
        self.assertTrue(torch.equal(loaded["weight"], new_tensor))


class TestSafeTrainingContextFailureException(unittest.TestCase):
    """Test SafeTrainingContext rollback when an exception is raised."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.checkpoint_dir = self.temp_dir / "checkpoints"
        self.adapter_path = self.temp_dir / "adapter_weights.pt"
        self.original_tensor = torch.randn(10)
        torch.save({"weight": self.original_tensor}, self.adapter_path)
        self.manager = CheckpointManager(self.checkpoint_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_exception_triggers_rollback(self):
        """An exception inside the context triggers automatic rollback to the
        checkpoint, restoring the original adapter weights."""
        try:
            with SafeTrainingContext(self.manager, self.adapter_path, "bad_train") as ctx:
                # Overwrite adapter with different data
                torch.save({"weight": torch.ones(10)}, self.adapter_path)
                raise RuntimeError("Training exploded!")
        except RuntimeError:
            pass
        # Adapter should be restored to original
        restored = torch.load(self.adapter_path, weights_only=True)
        self.assertTrue(torch.equal(restored["weight"], self.original_tensor))


class TestSafeTrainingContextFailureNoSuccess(unittest.TestCase):
    """Test SafeTrainingContext rollback when block exits without mark_success()."""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.checkpoint_dir = self.temp_dir / "checkpoints"
        self.adapter_path = self.temp_dir / "adapter_weights.pt"
        self.original_tensor = torch.randn(10)
        torch.save({"weight": self.original_tensor}, self.adapter_path)
        self.manager = CheckpointManager(self.checkpoint_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir)

    def test_no_mark_success_triggers_rollback(self):
        """Exiting the context without calling mark_success() triggers rollback."""
        with SafeTrainingContext(self.manager, self.adapter_path, "forgot") as ctx:
            torch.save({"weight": torch.ones(10)}, self.adapter_path)
            # Deliberately not calling ctx.mark_success()
        # Adapter should be restored to original
        restored = torch.load(self.adapter_path, weights_only=True)
        self.assertTrue(torch.equal(restored["weight"], self.original_tensor))

    def test_auto_rollback_disabled_keeps_changes(self):
        """When auto_rollback=False, exiting without mark_success does NOT rollback."""
        new_tensor = torch.ones(10)
        with SafeTrainingContext(
            self.manager, self.adapter_path, "no_rollback", auto_rollback=False
        ) as ctx:
            torch.save({"weight": new_tensor}, self.adapter_path)
            # Not calling mark_success, but auto_rollback is disabled
        loaded = torch.load(self.adapter_path, weights_only=True)
        self.assertTrue(torch.equal(loaded["weight"], new_tensor))


if __name__ == "__main__":
    unittest.main()
