"""Offline verifier fragment for Wave 18 / Bundle #52 / Item 3C."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

SPINE_DIGEST = "32693f54e0ce7ad3d401ceb752ae4723b0eee251f6d49b95cdfb26a6201971e9"
BUNDLE_ID = 52
DEPENDS_ON = ()

REGISTRY_FRAGMENT_REL = "docs/research/registry.d/52-t3c-qscci-singleton.json"
VERIFIER_FRAGMENT_REL = "scripts/verifiers.d/52_verifier.py"


def verify_bundle(repo_root: Optional[Path | str] = None) -> Dict[str, Any]:
    """Run the deterministic CPU drycheck for the planted QSCCI control."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from chelated_pv7_spine.claim_registry import claim_registry_sha256
    from chelated_pv7_spine.fragment_compiler import compile_fragments
    from scripts.run_pv7_w18_3c_qscci_singleton import execute

    checks: List[Dict[str, Any]] = []
    actual = claim_registry_sha256()
    checks.append({"name": "spine_digest_match", "passed": actual == SPINE_DIGEST, "detail": f"expected={SPINE_DIGEST} actual={actual}"})
    sources = {rel: (root / rel).read_bytes() for rel in (REGISTRY_FRAGMENT_REL, VERIFIER_FRAGMENT_REL)}
    compiled = compile_fragments(root, SPINE_DIGEST, sources)
    compiled_ok = compiled["dependency_order"] == [BUNDLE_ID] and compiled["registry"][0]["id"] == BUNDLE_ID
    checks.append({"name": "fragment_compiler_validation", "passed": compiled_ok, "detail": compiled["compiled_sha256"]})
    report = execute("local-drycheck")
    for gate in report["gates"]:
        checks.append({"name": f"drycheck_gate_{gate['name']}", "passed": bool(gate["passed"]), "detail": str(gate["detail"])[:200]})
    checks.append(
        {
            "name": "no_live_qwen_claim",
            "passed": report["qwen_report_split_executed"] is False and report["live_weights_loaded"] is False,
            "detail": "drycheck stays on the planted write site",
        }
    )
    passed = all(item["passed"] for item in checks)
    return {
        "bundle_id": BUNDLE_ID,
        "spine_digest": SPINE_DIGEST,
        "passed": passed,
        "checks": checks,
        "scoreboard_sha256": report["scoreboard_sha256"],
        "status": "local-drycheck-pass" if passed else "local-drycheck-fail",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    del argv
    result = verify_bundle()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
