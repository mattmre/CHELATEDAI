"""Packaging check for the large-sweep modules. Does not run a sweep."""

from __future__ import annotations

import unittest
from pathlib import Path


class TestSweepPackaging(unittest.TestCase):
    def test_py_modules_lists_the_sweep_modules(self):
        text = Path("pyproject.toml").read_text(encoding="utf-8")
        self.assertIn('"run_large_sweep"', text)
        self.assertIn('"sweep_result_store"', text)
        self.assertIn('"sweep_corpus_restore"', text)

    def test_result_store_imports_without_torch(self):
        import sweep_result_store

        self.assertTrue(callable(sweep_result_store.append_jsonl))


if __name__ == "__main__":
    unittest.main()
