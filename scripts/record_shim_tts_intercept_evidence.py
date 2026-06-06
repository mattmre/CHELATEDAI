#!/usr/bin/env python3
"""Record TTS intercept + steering-path shim evidence (CHELATED_SHIM_RESEARCH=1).

Exercises TTSPipeline.apply on a synthetic vector (translation/transport passthrough,
steering stage hits VectorSteerer.steer). Writes JSON under
``CHELATED_SHIM_EVIDENCE_DIR`` (default: ``artifacts/``) for SHIM-CD-05.
Does not import research/artifacts shim_node until BHS promotion.
"""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["CHELATED_SHIM_RESEARCH"] = "1"

from chelated_shim_research import collect_research_probe_from_tts_metadata  # noqa: E402
from tts_pipeline import SteeringSignal, TTSPipeline  # noqa: E402

DEFAULT_EVIDENCE_DIR = os.environ.get(
    "CHELATED_SHIM_EVIDENCE_DIR",
    str((ROOT / "artifacts").resolve()),
)


def main() -> int:
    dim = 8
    pipeline = TTSPipeline.build_default(dim)
    steerer = pipeline._steerer
    v = np.ones(dim, dtype=float) / np.sqrt(dim)

    idle = pipeline.apply(v.copy())
    steerer.add_signal(
        SteeringSignal(
            direction=np.array([1.0] + [0.0] * (dim - 1), dtype=float),
            strength=0.1,
            source="evidence_script",
        )
    )
    steered = pipeline.apply(v.copy())

    probe_idle = collect_research_probe_from_tts_metadata(
        idle.steering_meta, seam="AntigravityEngine.tts_intercept.idle"
    )
    probe_steered = collect_research_probe_from_tts_metadata(
        steered.steering_meta,
        seam="AntigravityEngine.tts_intercept.with_signal",
    )

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "env": {"CHELATED_SHIM_RESEARCH": os.environ.get("CHELATED_SHIM_RESEARCH")},
        "seam": "TTSPipeline.apply (AntigravityEngine TTS intercept path)",
        "runs": [
            {
                "label": "idle_no_persistent_signals",
                "stages_applied": idle.stages_applied,
                "total_delta_norm": idle.total_delta_norm,
                "vector_unchanged": bool(np.allclose(idle.after_steering, v)),
                "collector": probe_idle,
            },
            {
                "label": "with_steering_signal",
                "stages_applied": steered.stages_applied,
                "total_delta_norm": steered.total_delta_norm,
                "was_steered": bool(
                    steered.steering_meta and steered.steering_meta.get("was_steered")
                ),
                "collector": probe_steered,
            },
        ],
        "block_flag_note": "Run scripts/check_block_flag.py separately; this script does not flip BLOCKED.",
        "env": {
            "CHELATED_SHIM_RESEARCH": os.environ.get("CHELATED_SHIM_RESEARCH"),
            "CHELATED_SHIM_EVIDENCE_DIR": os.environ.get("CHELATED_SHIM_EVIDENCE_DIR"),
        },
    }

    out_dir = Path(DEFAULT_EVIDENCE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (
        "bhs_shim_evidence_tts_intercept_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d')}.json"
    )
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
