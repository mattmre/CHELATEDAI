"""Offline verifier fragment for Wave 19 / Bundle #60 / Item 2E."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

SPINE_DIGEST = "32693f54e0ce7ad3d401ceb752ae4723b0eee251f6d49b95cdfb26a6201971e9"
BUNDLE_ID = 60
DEPENDS_ON = ()

REGISTRY_FRAGMENT_REL = "docs/research/registry.d/60-t2e-ssd-kv-repos.json"
VERIFIER_REL = "scripts/verifiers.d/60_verifier.py"


def verify_bundle(repo_root: Optional[Path | str] = None) -> Dict[str, Any]:
    """Run deterministic offline CPU verification for Bundle #60 (2E)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from chelated_ssd_kv_repo_cache import run_ssd_kv_experiment
    from chelated_pv7_spine.claim_registry import claim_registry_sha256
    from chelated_pv7_spine.fragment_compiler import compile_fragments

    checks: List[Dict[str, Any]] = []
    actual_spine_digest = claim_registry_sha256()
    checks.append(
        {
            "name": "spine_digest_match",
            "passed": actual_spine_digest == SPINE_DIGEST,
            "detail": f"expected={SPINE_DIGEST} actual={actual_spine_digest}",
        }
    )

    reg_path = root / REGISTRY_FRAGMENT_REL
    reg_ok = False
    compiled_sha = ""
    if reg_path.is_file():
        compiled = compile_fragments(
            root,
            SPINE_DIGEST,
            {
                REGISTRY_FRAGMENT_REL: reg_path.read_bytes(),
                VERIFIER_REL: Path(__file__).read_bytes(),
            },
        )
        reg_ok = (
            compiled["dependency_order"] == [BUNDLE_ID]
            and len(compiled["registry"]) == 1
            and compiled["registry"][0]["id"] == BUNDLE_ID
        )
        compiled_sha = compiled["compiled_sha256"]
    checks.append(
        {"name": "fragment_compiler_validation", "passed": reg_ok, "detail": f"compiled_sha256={compiled_sha}"}
    )

    manifest = run_ssd_kv_experiment(mode="local-drycheck", repo_count=20, commits_per_repo=50)
    for gate_name, gate_ok in sorted(manifest["gates"].items()):
        checks.append({"name": f"gate_{gate_name}", "passed": bool(gate_ok), "detail": f"mode={manifest['mode']}"})

    passed = all(c["passed"] for c in checks)
    return {
        "bundle_id": BUNDLE_ID,
        "spine_digest": SPINE_DIGEST,
        "passed": passed,
        "tier": manifest["tier"],
        "checks": checks,
        "deterministic_sha256": manifest["deterministic_sha256"],
        "performance_claim_supported": manifest["performance_claim_supported"],
        "retirement_trigger_fired": manifest["retirement_trigger_fired"],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    _ = argv
    result = verify_bundle()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
