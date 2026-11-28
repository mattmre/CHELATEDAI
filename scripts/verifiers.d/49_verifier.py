"""Offline verifier fragment for Bundle #49 (Item 6D chelated-kernel SDK & Cold-Start Gate)."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

SPINE_DIGEST = "32693f54e0ce7ad3d401ceb752ae4723b0eee251f6d49b95cdfb26a6201971e9"
BUNDLE_ID = 49
DEPENDS_ON = (47,)
GATE_REPORT_REL = "artifacts/pv7_w17_6d_sdk_coldstart/gate_report.json"


def verify_bundle(repo_root: Path | str | None = None) -> dict:
    root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from scripts.run_pv7_w17_6d_sdk_coldstart_gate import run_gate

    with tempfile.TemporaryDirectory() as tmp:
        report = run_gate(Path(tmp), mode="local-drycheck")

    committed_ok = True
    committed_mode = "absent"
    gate_path = root / GATE_REPORT_REL
    if gate_path.is_file():
        committed = json.loads(gate_path.read_text(encoding="utf-8"))
        committed_mode = str(committed.get("mode"))
        committed_ok = (
            committed.get("gate_pass") is True
            and committed.get("mode") == "local-drycheck"
            and committed.get("live_model_calls") == 0
        )

    passed = bool(report["gate_pass"]) and committed_ok
    return {
        "bundle_id": BUNDLE_ID,
        "status": "local-drycheck-pass" if passed else "local-drycheck-fail",
        "passed": passed,
        "committed_gate_report_mode": committed_mode,
        "package_loc": report["package_loc"],
        "wheel_sha256": report["wheel_sha256"],
    }


def main() -> int:
    result = verify_bundle()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
