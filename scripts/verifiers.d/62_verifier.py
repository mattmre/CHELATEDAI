"""Offline verifier fragment for Wave 20 / Bundle #62 / Item 4 / Demo D13."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

SPINE_DIGEST = "32693f54e0ce7ad3d401ceb752ae4723b0eee251f6d49b95cdfb26a6201971e9"
BUNDLE_ID = 62
DEPENDS_ON = (47,)

REGISTRY_FRAGMENT_REL = "docs/research/registry.d/62-t4-pooled-workers.json"
REGISTRY_47_REL = "docs/research/registry.d/47-w16-core-spine.json"


def verify_bundle(repo_root: Optional[Path | str] = None) -> Dict[str, Any]:
    """Run deterministic offline CPU verification for Bundle #62 (`Demo D13`)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from chelated_pooled_agent_runtime import (
        PRIMARY_POOL_SIZE,
        run_pooled_agent_experiment,
    )
    from chelated_pv7_spine.claim_registry import claim_registry_sha256
    from chelated_pv7_spine.fragment_compiler import compile_fragments

    checks: List[Dict[str, Any]] = []

    actual_spine_digest = claim_registry_sha256()
    spine_ok = actual_spine_digest == SPINE_DIGEST
    checks.append(
        {
            "name": "spine_digest_match",
            "passed": spine_ok,
            "detail": f"expected={SPINE_DIGEST} actual={actual_spine_digest}",
        }
    )

    reg_path = root / REGISTRY_FRAGMENT_REL
    reg_47_path = root / REGISTRY_47_REL
    reg_exists = reg_path.is_file() and reg_47_path.is_file()
    reg_ok = False
    compiled_sha = ""
    if reg_exists:
        verifier_bytes = Path(__file__).read_bytes()
        compiled = compile_fragments(
            root,
            SPINE_DIGEST,
            {
                REGISTRY_FRAGMENT_REL: reg_path.read_bytes(),
                REGISTRY_47_REL: reg_47_path.read_bytes(),
                "scripts/verifiers.d/62_verifier.py": verifier_bytes,
            },
        )
        reg_ok = (
            compiled["dependency_order"] == [47, BUNDLE_ID]
            and len(compiled["registry"]) == 2
            and {item["id"] for item in compiled["registry"]} == {47, BUNDLE_ID}
        )
        compiled_sha = compiled["compiled_sha256"]
    checks.append(
        {
            "name": "fragment_compiler_validation",
            "passed": reg_ok,
            "detail": f"compiled_sha256={compiled_sha}",
        }
    )

    # Run deterministic Stage 1 controlled replay across all 24 templates x 4 schedules at primary pool size 4
    manifest = run_pooled_agent_experiment(
        mode="stage1-controlled-replay",
        pool_sizes=(PRIMARY_POOL_SIZE,),
    )
    for gate_name, gate_ok in sorted(manifest["gates"].items()):
        checks.append(
            {
                "name": f"stage1_gate_{gate_name}",
                "passed": bool(gate_ok),
                "detail": f"overall_pass={manifest['overall_pass']}",
            }
        )

    replay_status = manifest.get("status")
    status_ok = replay_status in {"stage1-replay-pass", "local-drycheck-pass"}
    checks.append(
        {
            "name": "replay_status",
            "passed": bool(status_ok),
            "detail": f"status={replay_status}",
        }
    )
    passed = all(c["passed"] for c in checks)
    status = replay_status if passed and status_ok else "replay-fail"
    return {
        "bundle_id": BUNDLE_ID,
        "spine_digest": SPINE_DIGEST,
        "status": status,
        "passed": passed,
        "checks": checks,
        "deterministic_sha256": manifest["deterministic_sha256"],
        "primary_2x2_interaction": manifest["primary_2x2_interaction"][
            "rate_interaction_matter_cluster_bootstrap"
        ],
        "jpm_interaction_per_job_bootstrap": manifest["primary_2x2_interaction"][
            "jpm_interaction_per_job_bootstrap"
        ],
        "sham_reopen_isolation": manifest["sham_reopen_isolation"],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    _ = argv
    result = verify_bundle()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
