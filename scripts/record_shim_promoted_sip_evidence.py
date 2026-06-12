#!/usr/bin/env python3
"""Record promoted SIP apply evidence (both CHELATED_* envs).

Writes JSON under ``CHELATED_SHIM_EVIDENCE_DIR`` (default: ``artifacts/``).
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
os.environ["CHELATED_SHIM_PROMOTED"] = "1"

from tts_pipeline import VectorSteerer  # noqa: E402

DEFAULT_EVIDENCE_DIR = os.environ.get(
    "CHELATED_SHIM_EVIDENCE_DIR",
    str((ROOT / "artifacts").resolve()),
)


def main() -> int:
    steerer = VectorSteerer()
    v = np.zeros(8, dtype=float)
    out, meta = steerer.steer(v)
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "env": {
            "CHELATED_SHIM_RESEARCH": os.environ.get("CHELATED_SHIM_RESEARCH"),
            "CHELATED_SHIM_PROMOTED": os.environ.get("CHELATED_SHIM_PROMOTED"),
        },
        "seam": "VectorSteerer.steer.promoted_apply",
        "vector_changed": not bool(np.allclose(out, v)),
        "steering_meta": meta,
        "promoted_sip_apply": meta.get("promoted_sip_apply"),
    }
    out_dir = Path(DEFAULT_EVIDENCE_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / (
        "bhs_shim_evidence_promoted_sip_"
        f"{datetime.now(timezone.utc).strftime('%Y%m%d')}.json"
    )
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {path}")
    ok = bool((payload.get("promoted_sip_apply") or {}).get("promoted_sip_applied"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
