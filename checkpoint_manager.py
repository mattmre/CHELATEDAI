"""
Checkpoint and Recovery System for ChelatedAI

Provides safe checkpoint/rollback functionality for training cycles.
"""

import json
import shutil
import torch
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import hashlib


class CheckpointManager:
    """
    Manages checkpoints for safe training with rollback capability.

    Creates backups before risky operations and can restore if things go wrong.
    """

    def __init__(self, checkpoint_dir: Optional[Path] = None):
        """
        Initialize checkpoint manager.

        Args:
            checkpoint_dir: Directory for checkpoints (default: ./checkpoints)
        """
        self.checkpoint_dir = checkpoint_dir or Path("./checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.metadata_path = self.checkpoint_dir / "checkpoint_metadata.json"
        self.metadata = self._load_metadata()

    def _load_metadata(self) -> Dict[str, Any]:
        """Load checkpoint metadata."""
        if self.metadata_path.exists():
            try:
                with open(self.metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                print(f"WARNING: Failed to load checkpoint metadata: {e}")
                return {"checkpoints": []}
        return {"checkpoints": []}

    def _save_metadata(self):
        """Save checkpoint metadata."""
        try:
            with open(self.metadata_path, 'w', encoding='utf-8') as f:
                json.dump(self.metadata, f, indent=2)
        except Exception as e:
            print(f"ERROR: Failed to save checkpoint metadata: {e}")

    def _compute_file_hash(self, file_path: Path) -> str:
        """Compute SHA256 hash of a file."""
        if not file_path.exists():
            return ""

        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()

    def create_checkpoint(
        self,
        name: str,
        adapter_path: Path,
        description: str = "",
        **extra_metadata
    ) -> str:
        """
        Create a checkpoint of current state.

        Args:
            name: Checkpoint name (e.g., 'before_training', 'state_0')
            adapter_path: Path to adapter weights file
            description: Optional description
            **extra_metadata: Additional metadata to store

        Returns:
            Checkpoint ID
        """
        timestamp = datetime.now().isoformat()
        checkpoint_id = f"{name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Create checkpoint subdirectory
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        # Copy adapter weights if they exist
        adapter_checkpoint = None
        adapter_hash = ""
        if adapter_path.exists():
            adapter_checkpoint = checkpoint_path / "adapter_weights.pt"
            shutil.copy2(adapter_path, adapter_checkpoint)
            adapter_hash = self._compute_file_hash(adapter_checkpoint)

        # Save metadata
        checkpoint_metadata = {
            "checkpoint_id": checkpoint_id,
            "name": name,
            "timestamp": timestamp,
            "description": description,
            "adapter_path": str(adapter_checkpoint) if adapter_checkpoint else None,
            "adapter_hash": adapter_hash,
            "original_adapter_path": str(adapter_path),
            **extra_metadata
        }

        self.metadata["checkpoints"].append(checkpoint_metadata)
        self.metadata["latest_checkpoint"] = checkpoint_id
        self._save_metadata()

        print(f"Created checkpoint: {checkpoint_id}")
        return checkpoint_id

    def restore_checkpoint(
        self,
        checkpoint_id: Optional[str] = None,
        target_adapter_path: Optional[Path] = None
    ) -> bool:
        """
        Restore from a checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to restore (default: latest)
            target_adapter_path: Where to restore adapter weights

        Returns:
            True if successful, False otherwise
        """
        if checkpoint_id is None:
            checkpoint_id = self.metadata.get("latest_checkpoint")
            if not checkpoint_id:
                print("ERROR: No checkpoints available")
                return False

        # Find checkpoint metadata
        checkpoint_meta = None
        for cp in self.metadata["checkpoints"]:
            if cp["checkpoint_id"] == checkpoint_id:
                checkpoint_meta = cp
                break

        if not checkpoint_meta:
            print(f"ERROR: Checkpoint '{checkpoint_id}' not found")
            return False

        print(f"Restoring checkpoint: {checkpoint_id}")

        # Restore adapter weights
        adapter_checkpoint = Path(checkpoint_meta["adapter_path"]) if checkpoint_meta["adapter_path"] else None
        if adapter_checkpoint and adapter_checkpoint.exists():
            # Determine target path
            if target_adapter_path is None:
                target_adapter_path = Path(checkpoint_meta["original_adapter_path"])

            # Verify integrity
            current_hash = self._compute_file_hash(adapter_checkpoint)
            if current_hash != checkpoint_meta["adapter_hash"]:
                print("WARNING: Checkpoint file hash mismatch, may be corrupted")

            # Copy back
            try:
                shutil.copy2(adapter_checkpoint, target_adapter_path)
                print(f"Restored adapter weights to: {target_adapter_path}")
            except Exception as e:
                print(f"ERROR: Failed to restore adapter: {e}")
                return False
        else:
            print("WARNING: No adapter weights in checkpoint")

        print("Checkpoint restored successfully")
        return True

    def list_checkpoints(self) -> list:
        """
        List all available checkpoints.

        Returns:
            List of checkpoint metadata dictionaries
        """
        return self.metadata.get("checkpoints", [])

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """
        Delete a checkpoint.

        Args:
            checkpoint_id: ID of checkpoint to delete

        Returns:
            True if successful, False otherwise
        """
        # Find checkpoint
        checkpoint_meta = None
        for i, cp in enumerate(self.metadata["checkpoints"]):
            if cp["checkpoint_id"] == checkpoint_id:
                checkpoint_meta = cp
                del self.metadata["checkpoints"][i]
                break

        if not checkpoint_meta:
            print(f"ERROR: Checkpoint '{checkpoint_id}' not found")
            return False

        # Delete directory
        checkpoint_path = self.checkpoint_dir / checkpoint_id
        if checkpoint_path.exists():
            try:
                shutil.rmtree(checkpoint_path)
            except Exception as e:
                print(f"ERROR: Failed to delete checkpoint directory: {e}")
                return False

        # Update metadata
        if self.metadata.get("latest_checkpoint") == checkpoint_id:
            if self.metadata["checkpoints"]:
                self.metadata["latest_checkpoint"] = self.metadata["checkpoints"][-1]["checkpoint_id"]
            else:
                self.metadata["latest_checkpoint"] = None

        self._save_metadata()
        print(f"Deleted checkpoint: {checkpoint_id}")
        return True

    def cleanup_old_checkpoints(self, keep_last_n: int = 5):
        """
        Delete old checkpoints, keeping only the N most recent.

        Args:
            keep_last_n: Number of recent checkpoints to keep
        """
        checkpoints = self.metadata.get("checkpoints", [])
        if len(checkpoints) <= keep_last_n:
            return

        # Sort by timestamp
        checkpoints.sort(key=lambda x: x["timestamp"])

        # Delete oldest
        to_delete = checkpoints[:-keep_last_n]
        for cp in to_delete:
            self.delete_checkpoint(cp["checkpoint_id"])

        print(f"Cleaned up {len(to_delete)} old checkpoints")


class SafeTrainingContext:
    """
    Context manager for safe training with automatic checkpoint/rollback.

    Usage:
        with SafeTrainingContext(checkpoint_mgr, adapter_path, "training_round_1") as ctx:
            # Do training
            train_adapter()
            ctx.mark_success()  # Only commits if no exception and marked successful
    """

    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        adapter_path: Path,
        operation_name: str,
        auto_rollback: bool = True
    ):
        """
        Initialize safe training context.

        Args:
            checkpoint_manager: CheckpointManager instance
            adapter_path: Path to adapter weights
            operation_name: Name for this operation
            auto_rollback: Automatically rollback on failure
        """
        self.checkpoint_manager = checkpoint_manager
        self.adapter_path = adapter_path
        self.operation_name = operation_name
        self.auto_rollback = auto_rollback
        self.checkpoint_id = None
        self.success = False

    def __enter__(self):
        """Create checkpoint before operation."""
        self.checkpoint_id = self.checkpoint_manager.create_checkpoint(
            name=f"before_{self.operation_name}",
            adapter_path=self.adapter_path,
            description=f"Automatic checkpoint before {self.operation_name}"
        )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Rollback on failure if auto_rollback enabled."""
        if exc_type is not None or not self.success:
            if self.auto_rollback:
                print(f"Operation failed or not marked successful, rolling back to checkpoint {self.checkpoint_id}")
                self.checkpoint_manager.restore_checkpoint(
                    checkpoint_id=self.checkpoint_id,
                    target_adapter_path=self.adapter_path
                )
            else:
                print("Operation failed but auto_rollback disabled")
        else:
            print(f"Operation '{self.operation_name}' completed successfully")

    def mark_success(self):
        """Mark operation as successful (prevents rollback)."""
        self.success = True


if __name__ == "__main__":
    # Demo usage
    from pathlib import Path
    import tempfile

    temp_dir = Path(tempfile.mkdtemp())
    adapter_path = temp_dir / "adapter.pt"

    # Create dummy adapter file
    torch.save({"data": "initial"}, adapter_path)

    # Create checkpoint manager
    checkpoint_mgr = CheckpointManager(temp_dir / "checkpoints")

    # Manual checkpoint
    cp_id = checkpoint_mgr.create_checkpoint(
        "manual_backup",
        adapter_path,
        description="Manual backup before experiment"
    )

    # Safe training context (success case)
    print("\n--- Success Case ---")
    with SafeTrainingContext(checkpoint_mgr, adapter_path, "experiment_1") as ctx:
        # Simulate training
        torch.save({"data": "trained"}, adapter_path)
        ctx.mark_success()  # Mark as successful

    # Safe training context (failure case)
    print("\n--- Failure Case ---")
    try:
        with SafeTrainingContext(checkpoint_mgr, adapter_path, "experiment_2") as ctx:
            # Simulate training
            torch.save({"data": "bad_training"}, adapter_path)
            # Don't mark success, or raise exception
            raise ValueError("Training failed!")
    except ValueError:
        pass

    # Check what was restored
    restored_data = torch.load(adapter_path)
    print(f"\nAdapter data after failed experiment: {restored_data}")

    # List checkpoints
    print("\n--- Checkpoints ---")
    for cp in checkpoint_mgr.list_checkpoints():
        print(f"  {cp['checkpoint_id']}: {cp['description']}")

    # Cleanup
    shutil.rmtree(temp_dir)
    print("\nDemo complete")
