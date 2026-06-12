#!/usr/bin/env python3
"""Record production-path shim probe evidence (CHELATED_SHIM_RESEARCH=1 only).

Writes JSON under ``CHELATED_SHIM_EVIDENCE_DIR`` (default: ``artifacts/``)
documenting VectorSteerer.steer behavior on a synthetic vector. Does not import
research/artifacts shim_node (BHS promotion gate).
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

from tts_pipeline import VectorSteerer  # noqa: E402

DEFAULT_EVIDENCE_DIR = os.environ.get(
    "CHELATED_SHIM_EVIDENCE_DIR",
    str((ROOT / "artifacts").resolve()),
)


def main() -> int:
    steerer = VectorSteerer()
    v = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    out1, meta1 = steerer.steer(v)
    out2, meta2 = steerer.steer(v)

    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "env": {"CHELATED_SHIM_RESEARCH": os.environ.get("CHELATED_SHIM_RESEARCH")},
        "seam": "VectorSteerer.steer",
        "runs": [
            {
                "stall_count": meta1.get("research_stall_count"),
                "was_steered": meta1.get("was_steered"),
                "guard": meta1.get("research_shim_guard"),
            },
            {
                "stall_count": meta2.get("research_stall_count"),
                "was_steered": meta2.get("was_steered"),
                "guard": meta2.get("research_shim_guard"),
            },
        ],
        "vector_unchanged": bool(np.allclose(out1, v) and np.allclose(out2, v)),
        "block_flag_note": "Run scripts/check_block_flag.py separately; this script does not flip BLOCKED.",
        "env": {
            "CHELATED_SHIM_RESEARCH": os.environ.get("CHELATED_SHIM_RESEARCH"),
            "CHELATED_SHIM_EVIDENCE_DIR": os.environ.get("CHELATED_SHIM_EVIDENCE_DIR"),
        },
    }

    out_dir = Path(DEFAULT_EVIDENCE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"bhs_shim_evidence_prod_probe_{datetime.now(timezone.utc).strftime('%Y%m%d')}.json"
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {out_path}")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
