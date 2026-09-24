"""Regression for large-sweep persistence. No model and no sweep grid."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sweep_result_store import (
    append_jsonl,
    materialize_json_array,
    migrate_json_array_to_jsonl,
    read_jsonl,
)


class TestSweepResultStore(unittest.TestCase):
    def test_appends_do_not_reread_or_rewrite_the_json_array(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            jsonl_path = root / "large_sweep_results.jsonl"
            json_path = root / "large_sweep_results.json"
            entries = [{"index": index, "gain": index / 10} for index in range(5)]

            for entry in entries:
                append_jsonl(jsonl_path, entry)

            self.assertFalse(json_path.exists())
            self.assertEqual(read_jsonl(jsonl_path), entries)
            self.assertEqual(materialize_json_array(jsonl_path, json_path), 5)
            self.assertEqual(json.loads(json_path.read_text(encoding="utf-8")), entries)
            self.assertFalse(Path(str(json_path) + ".tmp").exists())

    def test_existing_array_is_copied_once(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            json_path = root / "large_sweep_results.json"
            jsonl_path = root / "large_sweep_results.jsonl"
            original = [{"index": 0}, {"index": 1}]
            json_path.write_text(json.dumps(original), encoding="utf-8")

            self.assertEqual(migrate_json_array_to_jsonl(json_path, jsonl_path), 2)
            self.assertEqual(migrate_json_array_to_jsonl(json_path, jsonl_path), 0)
            append_jsonl(jsonl_path, {"index": 2})

            self.assertEqual(read_jsonl(jsonl_path), original + [{"index": 2}])

    def test_bad_jsonl_line_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            jsonl_path = Path(directory) / "large_sweep_results.jsonl"
            jsonl_path.write_text('{"index": 0}\nnot-json\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                read_jsonl(jsonl_path)

    def test_sweep_script_no_longer_loads_the_result_array(self):
        source = Path("run_large_sweep.py").read_text(encoding="utf-8")
        self.assertNotIn("json.load", source)
        self.assertNotIn("json.dump", source)
        self.assertIn("append_jsonl", source)
        self.assertIn("materialize_json_array", source)
        self.assertIn("migrate_json_array_to_jsonl", source)
        store_source = Path("sweep_result_store.py").read_text(encoding="utf-8")
        self.assertIn("def append_jsonl", store_source)
        self.assertNotIn("json.load", inspect_append_body(store_source))


def inspect_append_body(source: str) -> str:
    start = source.index("def append_jsonl")
    end = source.index("def read_jsonl")
    return source[start:end]


if __name__ == "__main__":
    unittest.main()
