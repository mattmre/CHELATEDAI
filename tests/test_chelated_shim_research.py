"""Unit tests for chelated_shim_research helpers."""

from __future__ import annotations

import os
import unittest

from chelated_shim_research import (
    attach_research_meta,
    bump_stall_counter,
    collect_research_probe_from_tts_metadata,
    research_enabled,
    research_preflight_metadata,
)


class TestChelatedShimResearch(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("CHELATED_SHIM_RESEARCH", None)

    def test_research_disabled_by_default(self) -> None:
        os.environ.pop("CHELATED_SHIM_RESEARCH", None)
        self.assertFalse(research_enabled())

    def test_research_enabled_only_for_one(self) -> None:
        os.environ["CHELATED_SHIM_RESEARCH"] = "1"
        self.assertTrue(research_enabled())
        os.environ["CHELATED_SHIM_RESEARCH"] = "true"
        self.assertFalse(research_enabled())

    def test_bump_stall_counter(self) -> None:
        self.assertEqual(bump_stall_counter(0, has_work=False), 1)
        self.assertEqual(bump_stall_counter(3, has_work=True), 0)

    def test_attach_research_meta(self) -> None:
        out = attach_research_meta({}, None)
        self.assertNotIn("research_shim", out)

    def test_preflight_metadata_merges_extra(self) -> None:
        meta = research_preflight_metadata(
            seam="test.seam",
            stall_count=2,
            extra={"was_steered": False},
        )
        self.assertEqual(meta["sip_seam"], "test.seam")
        self.assertEqual(meta["research_stall_count"], 2)
        self.assertFalse(meta["was_steered"])

    def test_collect_probe_from_guarded_meta(self) -> None:
        meta = research_preflight_metadata(seam="VectorSteerer.steer", stall_count=1)
        out = collect_research_probe_from_tts_metadata(meta)
        self.assertTrue(out["probe_hit"])


if __name__ == "__main__":
    unittest.main()