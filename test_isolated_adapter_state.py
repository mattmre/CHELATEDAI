"""Checkpoint isolation for benchmark runs. No model load."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from benchmark_utils import isolated_adapter_state


class TestIsolatedAdapterState(unittest.TestCase):
    def test_restores_original_and_leaves_no_backup(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter_weights.pt"
            path.write_bytes(b"original")
            with isolated_adapter_state(path):
                self.assertFalse(path.exists())
                path.write_bytes(b"mutated")
            self.assertEqual(path.read_bytes(), b"original")
            self.assertEqual(list(Path(directory).glob("*.benchmark-backup-*")), [])

    def test_absent_file_is_not_left_behind(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter_weights.pt"
            with isolated_adapter_state(path):
                path.write_bytes(b"created-inside")
            self.assertFalse(path.exists())
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_nested_blocks_restore_each_level(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "adapter_weights.pt"
            path.write_bytes(b"original")
            with isolated_adapter_state(path):
                path.write_bytes(b"outer-write")
                with isolated_adapter_state(path):
                    path.write_bytes(b"inner-write")
                self.assertEqual(path.read_bytes(), b"outer-write")
            self.assertEqual(path.read_bytes(), b"original")
            self.assertEqual(list(Path(directory).glob("*.benchmark-backup-*")), [])


if __name__ == "__main__":
    unittest.main()
