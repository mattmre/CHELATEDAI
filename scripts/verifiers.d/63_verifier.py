"""Offline verifier fragment for Bundle #63 (Partner Trial 1 Search-Operator Migration Kit)."""

from __future__ import annotations

from pathlib import Path

SPINE_DIGEST = "32693f54e0ce7ad3d401ceb752ae4723b0eee251f6d49b95cdfb26a6201971e9"
BUNDLE_ID = 63
DEPENDS_ON = (47,)


def verify_bundle(repo_root: Path | str | None = None) -> dict:
    root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[2]
    charter = root / "docs" / "research" / "partner_charter_trial1_search_operator.md"
    kit = root / "partner_kits" / "trial1_search_migration_kit.py"
    return {
        "bundle_id": BUNDLE_ID,
        "status": "protocol-ready",
        "passed": charter.is_file() and kit.is_file(),
    }
