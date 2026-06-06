"""BHS: insert-once guarantee on promoted apply_shim_cascade."""

from __future__ import annotations

import os
import unittest
from unittest import mock

import numpy as np

from chelated_shim_research import promoted_sip_apply


class TestPromotedInsertOnce(unittest.TestCase):
    def test_cascade_ids_unique_per_apply(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"CHELATED_SHIM_RESEARCH": "1", "CHELATED_SHIM_PROMOTED": "1"},
            clear=False,
        ):
            try:
                import shim_node_promoted  # noqa: F401
            except ImportError:
                self.skipTest("run scripts/promote_shim_primitives.py first")

            _, meta = promoted_sip_apply(np.zeros(8, dtype=float))
            self.assertIsNotNone(meta)
            ids = meta.get("cascade_ids") or []
            self.assertEqual(len(ids), len(set(ids)), "insert-once: no duplicate cascade ids")
            self.assertTrue(meta.get("promoted_sip_insert_once"))

    def test_double_apply_still_unique_ids(self) -> None:
        with mock.patch.dict(
            os.environ,
            {"CHELATED_SHIM_RESEARCH": "1", "CHELATED_SHIM_PROMOTED": "1"},
            clear=False,
        ):
            try:
                import shim_node_promoted  # noqa: F401
            except ImportError:
                self.skipTest("run scripts/promote_shim_primitives.py first")

            v = np.zeros(8, dtype=float)
            _, m1 = promoted_sip_apply(v)
            _, m2 = promoted_sip_apply(v)
            for meta in (m1, m2):
                self.assertIsNotNone(meta)
                ids = meta.get("cascade_ids") or []
                self.assertEqual(len(ids), len(set(ids)))


if __name__ == "__main__":
    unittest.main()