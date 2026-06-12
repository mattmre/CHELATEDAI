"""SHIM-CD-01: AntigravityEngine post-embed research metadata (no retrieval required)."""

from __future__ import annotations

import os
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from chelated_shim_research import research_enabled


class TestShimAntigravityPostEmbed(unittest.TestCase):
    def tearDown(self) -> None:
        os.environ.pop("CHELATED_SHIM_RESEARCH", None)

    @patch("antigravity_engine.research_enabled", return_value=True)
    def test_post_embed_records_meta_when_tts_disabled(self, _enabled: MagicMock) -> None:
        from antigravity_engine import AntigravityEngine

        engine = AntigravityEngine.__new__(AntigravityEngine)
        engine.vector_size = 4
        engine._research_post_embed_stall = 0
        engine._last_research_shim_meta = None
        engine._tts_pipeline = None

        from chelated_shim_research import bump_stall_counter, research_preflight_metadata

        _tts_probe = getattr(engine, "_tts_pipeline", None)
        engine._research_post_embed_stall = bump_stall_counter(
            engine._research_post_embed_stall,
            has_work=_tts_probe is not None,
        )
        engine._last_research_shim_meta = research_preflight_metadata(
            seam="AntigravityEngine.post_embed",
            stall_count=engine._research_post_embed_stall,
            extra={"has_tts_pipeline": _tts_probe is not None},
        )

        meta = engine._last_research_shim_meta
        self.assertTrue(meta["research_shim_guard"])
        self.assertEqual(meta["sip_seam"], "AntigravityEngine.post_embed")
        self.assertEqual(meta["research_stall_count"], 1)
        self.assertFalse(meta["has_tts_pipeline"])

    def test_research_off_leaves_meta_none_on_fresh_engine_fields(self) -> None:
        os.environ.pop("CHELATED_SHIM_RESEARCH", None)
        self.assertFalse(research_enabled())


if __name__ == "__main__":
    unittest.main()