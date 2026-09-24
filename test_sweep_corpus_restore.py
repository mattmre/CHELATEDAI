"""Corpus snapshot restore. Uses a fake client, not a running Qdrant."""

from __future__ import annotations

import unittest
from pathlib import Path

from sweep_corpus_restore import restore_collection, snapshot_collection


class _Point:
    def __init__(self, point_id, vector, payload):
        self.id = point_id
        self.vector = vector
        self.payload = payload


class _Client:
    def __init__(self, points):
        self._points = list(points)
        self.upserts = []

    def scroll(self, collection_name, limit, with_vectors, with_payload, offset):
        if offset is not None:
            return [], None
        return list(self._points), None

    def upsert(self, collection_name, points):
        self.upserts.append(list(points))


class TestSweepCorpusRestore(unittest.TestCase):
    def test_restore_writes_the_original_vectors_not_a_later_edit(self):
        client = _Client([_Point(1, [0.25, 0.5], {"text": "a"}), _Point("b", [1.0], None)])
        snapshot = snapshot_collection(client, "docs")
        client._points[0].vector = [9.0, 9.0]

        written = restore_collection(client, "docs", snapshot)

        self.assertEqual(written, 2)
        restored = client.upserts[0]
        self.assertEqual(restored[0].vector, [0.25, 0.5])
        self.assertEqual(restored[0].payload, {"text": "a"})
        self.assertEqual(restored[1].payload, {})

    def test_empty_snapshot_does_not_upsert(self):
        client = _Client([])
        snapshot = snapshot_collection(client, "docs")
        self.assertEqual(restore_collection(client, "docs", snapshot), 0)
        self.assertEqual(client.upserts, [])

    def test_sweep_scripts_restore_before_replacing_the_adapter(self):
        for name in ("run_large_sweep.py", "run_sweep.py"):
            source = Path(name).read_text(encoding="utf-8")
            restore_at = source.index("restore_collection(")
            adapter_at = source.index("engine.adapter = create_adapter(")
            self.assertLess(restore_at, adapter_at, name)
            self.assertIn("snapshot_collection(", source)


if __name__ == "__main__":
    unittest.main()
