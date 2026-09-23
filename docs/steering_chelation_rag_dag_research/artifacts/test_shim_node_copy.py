"""Copy-isolation checks for the research ShimRegistry.

Loads shim_node.py by file path so this artifact does not need a package
and is not part of the repository root test suite.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path

import numpy as np


def _load_shim_node():
    path = Path(__file__).resolve().parent / "shim_node.py"
    spec = importlib.util.spec_from_file_location("shim_node_copy_under_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load shim_node from {path}")
    module = importlib.util.module_from_spec(spec)
    # dataclasses looks up cls.__module__ in sys.modules during class creation.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TestShimNodeCopyIsolation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.shim = _load_shim_node()

    def setUp(self):
        self.reg = self.shim.ShimRegistry(dim=4, seed_salt="copy-isolation-test")
        self.sid = "probe"
        vec = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.reg.register(self.sid, [vec], metadata={"k": "v"})

    def test_repeated_get_is_not_same_object(self):
        first = self.reg.get(self.sid)
        second = self.reg.get(self.sid)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        self.assertIsNot(first, second)

    def test_mutating_first_get_does_not_change_second_get(self):
        first = self.reg.get(self.sid)
        self.assertIsNotNone(first)
        first.vectors[0][0] = 999.0
        second = self.reg.get(self.sid)
        self.assertIsNotNone(second)
        self.assertIsNot(first, second)
        self.assertAlmostEqual(float(second.vectors[0][0]), 1.0)
        self.assertNotAlmostEqual(float(second.vectors[0][0]), 999.0)

    def test_lookup_by_context_is_not_registry_object(self):
        # get() is itself a copy, so identity is checked against stored state.
        stored = self.reg._nodes[self.sid]
        found = self.reg.lookup_by_context(stored.vectors[0].copy(), top_k=1)
        self.assertEqual(len(found), 1)
        looked = found[0]
        self.assertIsNot(looked, stored)
        looked.vectors[0][0] = 999.0
        again = self.reg.get(self.sid)
        self.assertIsNotNone(again)
        self.assertIsNot(looked, again)
        self.assertAlmostEqual(float(again.vectors[0][0]), 1.0)
        self.assertAlmostEqual(float(stored.vectors[0][0]), 1.0)


if __name__ == "__main__":
    unittest.main()
