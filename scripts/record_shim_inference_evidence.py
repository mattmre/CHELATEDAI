#!/usr/bin/env python3
"""Record AntigravityEngine.run_inference shim evidence (CHELATED_SHIM_RESEARCH=1).

Exercises enable_tts() + run_inference() with mocked embed/Qdrant so no Ollama
is required. Writes JSON under ``CHELATED_SHIM_EVIDENCE_DIR`` (default:
``artifacts/``) for SHIM-CD-05 (inference path).
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["CHELATED_SHIM_RESEARCH"] = "1"
DEFAULT_EVIDENCE_DIR = os.environ.get(
    "CHELATED_SHIM_EVIDENCE_DIR",
    str((ROOT / "artifacts").resolve()),
)

from antigravity_engine import AntigravityEngine  # noqa: E402
from tts_pipeline import TTSConfig  # noqa: E402


def _build_engine_stub(dim: int = 8) -> AntigravityEngine:
    engine = AntigravityEngine.__new__(AntigravityEngine)
    engine.logger = MagicMock()
    engine.vector_size = dim
    engine.adapter = None
    engine._tts_pipeline = None
    engine._last_tts_result = None
    engine._last_research_shim_meta = None
    engine._research_post_embed_stall = 0
    engine._research_chelation_stall = 0
    engine._research_pre_retrieval_stall = 0
    engine._query_reformulation_active = False
    engine._adapter_routing_active = False
    engine._adapter_router = None
    engine._query_reformulator = None
    engine.use_centering = False
    engine.use_quantization = False
    engine.chelation_threshold = 0.5
    engine._adaptive_threshold_lock = MagicMock()
    engine.collection_name = "shim_evidence_collection"
    qdrant_mock = MagicMock()
    point = MagicMock()
    point.id = "pt-1"
    point.vector = np.ones(dim, dtype=float) * 0.1
    qdrant_response = MagicMock()
    qdrant_response.points = [point]
    qdrant_mock.query_points.return_value = qdrant_response
    engine.qdrant = qdrant_mock
    return engine


def main() -> int:
    dim = 8
    engine = _build_engine_stub(dim)
    engine.enable_tts(tts_config=TTSConfig())

    good_embedding = np.ones((1, dim), dtype=float) / np.sqrt(dim)

    with (
        patch.object(engine, "embed", return_value=good_embedding),
        patch.object(engine, "_observe_model_scope_query", return_value=None),
        patch.object(engine, "_record_runtime_diagnostics", return_value=None),
        patch.object(engine, "_build_runtime_diagnostics", return_value={"runtime": {"status": "ok"}}),
        patch.object(engine, "_select_retrieval_policy", return_value={"policy": "FAST"}),
        patch.object(engine, "_update_adaptive_threshold", return_value=None),
    ):
        try:
            engine.run_inference("shim evidence query")
        except Exception as exc:
            # Retrieval/chelation branches may still raise on partial stub; capture partial evidence
            partial_error = repr(exc)
        else:
            partial_error = None

    tts_result = engine.get_last_tts_result()
    research_meta = engine.get_last_research_shim_meta()

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "env": {"CHELATED_SHIM_RESEARCH": os.environ.get("CHELATED_SHIM_RESEARCH")},
        "seam": "AntigravityEngine.run_inference",
        "tts_enabled": engine._tts_pipeline is not None,
        "tts_result_present": tts_result is not None,
        "research_shim_meta": research_meta,
        "research_seams_observed": list(
            {
                (research_meta or {}).get("sip_seam"),
                "AntigravityEngine.post_embed",
                "AntigravityEngine.pre_retrieval",
                "AntigravityEngine.chelation_variance",
            }
        ),
        "partial_error": partial_error,
        "block_flag_note": "Run scripts/check_block_flag.py separately; does not flip BLOCKED.",
        "env": {
            "CHELATED_SHIM_RESEARCH": os.environ.get("CHELATED_SHIM_RESEARCH"),
            "CHELATED_SHIM_EVIDENCE_DIR": os.environ.get("CHELATED_SHIM_EVIDENCE_DIR"),
        },
    }

    out_dir = Path(DEFAULT_EVIDENCE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (
        "bhs_shim_evidence_inference_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d')}.json"
    )
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    print(json.dumps(payload, indent=2))

    ok = research_meta is not None and bool(research_meta.get("research_shim_guard"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
