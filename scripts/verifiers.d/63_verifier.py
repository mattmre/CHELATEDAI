"""Offline verifier fragment for Bundle #63 (Partner Trial 1 Search-Operator Migration Kit)."""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

SPINE_DIGEST = "32693f54e0ce7ad3d401ceb752ae4723b0eee251f6d49b95cdfb26a6201971e9"
BUNDLE_ID = 63
DEPENDS_ON = (47,)


def verify_bundle(repo_root: Optional[Path | str] = None) -> Dict[str, Any]:
    """Check the frozen spine digest, compile drop-in fragments, and score the open DEV suite."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from chelated_pv7_spine.claim_registry import claim_registry_sha256
    from chelated_pv7_spine.fragment_compiler import compile_repo_fragments
    from partner_kits.trial1_search_migration_kit import run_offline_dev_fixture_suite

    charter = root / "docs" / "research" / "partner_charter_trial1_search_operator.md"
    kit = root / "partner_kits" / "trial1_search_migration_kit.py"
    files_ok = charter.is_file() and kit.is_file()
    spine_ok = claim_registry_sha256() == SPINE_DIGEST

    compile_ok = False
    compile_detail = ""
    try:
        compiled = compile_repo_fragments(root, SPINE_DIGEST)
        compile_ok = (
            compiled.get("spine_digest") == SPINE_DIGEST
            and BUNDLE_ID in compiled.get("dependency_order", [])
            and 47 in compiled.get("dependency_order", [])
        )
        compile_detail = str(compiled.get("compiled_sha256", ""))
    except Exception as exc:
        compile_detail = f"{type(exc).__name__}: {exc}"
        compile_ok = False

    suite_ok = False
    suite_status = ""
    try:
        suite = run_offline_dev_fixture_suite()
        suite_status = str(suite.get("status"))
        outcomes = {
            name: scenario["decision"]["outcome"] for name, scenario in suite["scenarios"].items()
        }
        suite_ok = (
            suite_status == "protocol-ready"
            and outcomes.get("win") == "win_zeromu_csls"
            and outcomes.get("tie") == "tie_retire_zeromu_credit_csls"
            and outcomes.get("loss") == "loss"
        )
    except Exception as exc:
        suite_status = f"{type(exc).__name__}: {exc}"
        suite_ok = False

    passed = bool(files_ok and spine_ok and compile_ok and suite_ok)
    return {
        "bundle_id": BUNDLE_ID,
        "status": "protocol-ready" if passed else "protocol-failed",
        "passed": passed,
        "spine_digest": SPINE_DIGEST,
        "checks": {
            "charter_and_kit_present": files_ok,
            "spine_digest_match": spine_ok,
            "compile_repo_fragments": compile_ok,
            "dev_fixture_suite": suite_ok,
        },
        "compile_detail": compile_detail,
        "suite_status": suite_status,
    }


def main() -> int:
    import json

    result = verify_bundle()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
