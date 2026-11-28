"""Offline verifier fragment for Bundle #64 (Partner Trial 2 Records-Owner Governance Kit).

Verifies (no GPU, no network, deterministic):

1. the frozen ``SPINE_DIGEST`` equals the live claim-registry digest;
2. the whole drop-in fragment set compiles under ``compile_repo_fragments`` and bundle 64 lands after its
   dependency 47 in the dependency order;
3. the offline row-level scorer (``run_offline_dev_fixture_suite``) reproduces the pre-registered outcome of all
   five open DEV fixtures, is byte-deterministic, stamps the locked protocol digest, never self-promotes a DEV
   fixture past ``protocol-ready``, and scores the ALL-enrolled-matters cohort with the exact Student-t critical
   value (B64-1/B64-2/B64-3).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

SPINE_DIGEST = "32693f54e0ce7ad3d401ceb752ae4723b0eee251f6d49b95cdfb26a6201971e9"
BUNDLE_ID = 64
DEPENDS_ON = (47,)

#: Pre-registered outcome of each open DEV fixture (scenario -> (outcome, verdict, ek5_premium)).
EXPECTED_DEV_OUTCOMES = {
    "win": ("win", "EK5_PREMIUM_RETAINED", "retained"),
    "loss": ("loss", "EK5_PREMIUM_RETIRED", "retired"),
    "tie": ("tie", "EK5_PREMIUM_RETIRED", "retired"),
    "timeout": ("timeout", "EK5_PREMIUM_RETIRED", "retired"),
    "insufficient_natural_conflicts": ("insufficient_natural_conflicts", "INSUFFICIENT_EVIDENCE", "undetermined"),
}


def verify_bundle(repo_root: Optional[Path | str] = None) -> Dict[str, Any]:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from chelated_pv7_spine.claim_registry import claim_registry_sha256
    from chelated_pv7_spine.fragment_compiler import FragmentCompileError, compile_repo_fragments
    from partner_kits import trial2_records_governance_kit as kit

    checks: List[Dict[str, Any]] = []

    charter = root / "docs" / "research" / "partner_charter_trial2_records_owner.md"
    kit_file = root / "partner_kits" / "trial2_records_governance_kit.py"
    checks.append(
        {
            "name": "charter_and_kit_files_present",
            "passed": charter.is_file() and kit_file.is_file(),
            "detail": f"charter={charter.is_file()} kit={kit_file.is_file()}",
        }
    )

    actual_spine_digest = claim_registry_sha256()
    checks.append(
        {
            "name": "spine_digest_match",
            "passed": actual_spine_digest == SPINE_DIGEST,
            "detail": f"expected={SPINE_DIGEST} actual={actual_spine_digest}",
        }
    )

    compiled_sha = ""
    order: List[int] = []
    try:
        compiled = compile_repo_fragments(root, SPINE_DIGEST)
        order = list(compiled["dependency_order"])
        compiled_sha = compiled["compiled_sha256"]
        compile_ok = (
            BUNDLE_ID in order
            and all(dep in order and order.index(dep) < order.index(BUNDLE_ID) for dep in DEPENDS_ON)
            and any(entry["id"] == BUNDLE_ID for entry in compiled["registry"])
            and any(entry["bundle_id"] == BUNDLE_ID for entry in compiled["verifiers"])
        )
        detail = f"compiled_sha256={compiled_sha} bundles={len(order)}"
    except (OSError, FragmentCompileError) as exc:
        compile_ok = False
        detail = f"compile_repo_fragments failed: {type(exc).__name__}: {exc}"
    checks.append({"name": "compile_repo_fragments", "passed": compile_ok, "detail": detail})

    suite = kit.run_offline_dev_fixture_suite()
    suite_again = kit.run_offline_dev_fixture_suite()
    checks.append(
        {
            "name": "dev_fixture_suite_deterministic",
            "passed": suite["suite_sha256"] == suite_again["suite_sha256"],
            "detail": f"suite_sha256={suite['suite_sha256']}",
        }
    )
    checks.append(
        {
            "name": "suite_stamped_with_locked_protocol_digest",
            "passed": suite["protocol_digest"] == kit.PROTOCOL_DIGEST
            and kit._sha256_json(kit.locked_protocol()) == kit.PROTOCOL_DIGEST,
            "detail": f"protocol_digest={suite['protocol_digest']}",
        }
    )
    checks.append(
        {
            "name": "dev_fixtures_never_promoted_past_protocol_ready",
            "passed": suite["status"] == kit.STAGE1_STATUS
            and all(r["status"] == kit.STAGE1_STATUS and r["data_origin"] == "open_dev_fixture"
                    for r in suite["scenarios"].values()),
            "detail": f"suite_status={suite['status']}",
        }
    )
    for scenario, (outcome, verdict, premium) in sorted(EXPECTED_DEV_OUTCOMES.items()):
        got = suite["scenarios"].get(scenario, {})
        checks.append(
            {
                "name": f"dev_fixture_{scenario}_outcome",
                "passed": (got.get("outcome"), got.get("verdict"), got.get("ek5_premium")) == (outcome, verdict, premium),
                "detail": f"expected={(outcome, verdict, premium)} got={(got.get('outcome'), got.get('verdict'), got.get('ek5_premium'))}",
            }
        )
    scored = [r for r in suite["scenarios"].values() if "comparisons" in r]
    checks.append(
        {
            "name": "scored_over_all_enrolled_matters_with_exact_student_t",
            "passed": bool(scored)
            and all(
                r["scoring_cohort"] == "all_enrolled_matters"
                and r["scored_matters"] == r["sizing"]["consecutive_matters"]
                and all(
                    p["n_matters"] == r["scored_matters"]
                    and p["t_critical"] == kit.student_t_critical_95(r["scored_matters"] - 1)
                    for comp in r["comparisons"].values()
                    for p in (comp["attribution_pass"], comp["operational_pass"])
                )
                and r["faithfulness_noninferiority"]["n_matters"] == r["scored_matters"]
                for r in scored
            ),
            "detail": f"scored_scenarios={len(scored)}",
        }
    )

    passed = all(c["passed"] for c in checks)
    return {
        "bundle_id": BUNDLE_ID,
        "spine_digest": SPINE_DIGEST,
        "status": "protocol-ready",
        "passed": passed,
        "checks": checks,
        "compiled_sha256": compiled_sha,
        "suite_sha256": suite["suite_sha256"],
        "outcomes": {name: r["outcome"] for name, r in sorted(suite["scenarios"].items())},
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    _ = argv
    result = verify_bundle()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
