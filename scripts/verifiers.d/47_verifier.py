"""Offline verifier fragment for Bundle #47 (Wave 16-Core Spine)."""

from __future__ import annotations

from pathlib import Path

SPINE_DIGEST = "32693f54e0ce7ad3d401ceb752ae4723b0eee251f6d49b95cdfb26a6201971e9"
BUNDLE_ID = 47
DEPENDS_ON = ()


def verify_bundle(repo_root: Path | str | None = None) -> dict:
    root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[2]
    from chelated_pv7_spine.burn_ledger import HISTORICAL_DATASETS, build_burn_ledger
    from chelated_pv7_spine.claim_registry import claim_registry_sha256

    actual_sha = claim_registry_sha256()
    ledger = build_burn_ledger(root)
    burned_ok = all(ledger.to_dict()["datasets"][name]["status"] == "BURNED" for name in HISTORICAL_DATASETS)
    passed = actual_sha == SPINE_DIGEST and burned_ok
    return {
        "bundle_id": BUNDLE_ID,
        "passed": passed,
        "spine_digest": actual_sha,
        "burn_ledger_sha256": ledger.sha256,
    }
