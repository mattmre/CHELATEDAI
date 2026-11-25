"""Offline verifier fragment for Bundle #49 (Item 6D chelated-kernel SDK & Cold-Start Gate)."""

from __future__ import annotations

import tempfile
from pathlib import Path

SPINE_DIGEST = "32693f54e0ce7ad3d401ceb752ae4723b0eee251f6d49b95cdfb26a6201971e9"
BUNDLE_ID = 49
DEPENDS_ON = (47,)


def verify_bundle(repo_root: Path | str | None = None) -> dict:
    _ = repo_root
    from scripts.run_pv7_w17_6d_sdk_coldstart_gate import run_gate

    with tempfile.TemporaryDirectory() as tmp:
        report = run_gate(Path(tmp), mode="local-drycheck")
    return {
        "bundle_id": BUNDLE_ID,
        "passed": bool(report["gate_pass"]),
        "package_loc": report["package_loc"],
        "wheel_sha256": report["wheel_sha256"],
    }
