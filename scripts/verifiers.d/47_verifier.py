"""Offline verifier fragment for Bundle #47 (Wave 16-Core Spine).

Checks, all deterministic and CPU-only:
1. the frozen claim-registry ``SPINE_DIGEST``;
2. ``compile_repo_fragments`` over every registry/route/verifier drop-in;
3. every canonical burned corpus is BURNED in the live burn ledger;
4. the deterministic synthetic A/A null calibration passes its one-sided
   anti-conservatism rule and its receipt hash verifies.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

SPINE_DIGEST = "32693f54e0ce7ad3d401ceb752ae4723b0eee251f6d49b95cdfb26a6201971e9"
BUNDLE_ID = 47
DEPENDS_ON = ()


def verify_bundle(repo_root: Path | str | None = None) -> dict:
    root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from chelated_pv7_spine.aa_null_calibration import (
        run_deterministic_drycheck,
        verify_aa_calibration_receipt,
    )
    from chelated_pv7_spine.burn_ledger import BURNED_CORPORA, build_burn_ledger
    from chelated_pv7_spine.claim_registry import claim_registry_sha256
    from chelated_pv7_spine.fragment_compiler import FragmentCompileError, compile_repo_fragments

    actual_sha = claim_registry_sha256()
    digest_ok = actual_sha == SPINE_DIGEST

    try:
        compiled = compile_repo_fragments(root, SPINE_DIGEST)
        compiled_sha = compiled["compiled_sha256"]
        fragments_ok = BUNDLE_ID in compiled["dependency_order"]
        fragment_error = None
    except FragmentCompileError as exc:
        compiled_sha, fragments_ok, fragment_error = None, False, str(exc)

    ledger = build_burn_ledger(root)
    datasets = ledger.to_dict()["datasets"]
    burned_missing = [name for name in BURNED_CORPORA if datasets.get(name, {}).get("status") != "BURNED"]

    aa_receipt = run_deterministic_drycheck()
    aa_ok = bool(aa_receipt["passed"]) and verify_aa_calibration_receipt(aa_receipt)

    host_cal_rels = (
        "artifacts/pv7_w16_core_spine/aa_null_calibration_desktop_rtx3090.json",
        "artifacts/pv7_w16_core_spine/aa_null_calibration_spark3_gb10.json",
    )
    host_cal_checks = {}
    host_cals_ok = True
    for rel in host_cal_rels:
        p = root / rel
        if p.is_file():
            doc = json.loads(p.read_text(encoding="utf-8"))
            ok_doc = bool(doc.get("passed")) and verify_aa_calibration_receipt(doc)
            host_cal_checks[rel] = {
                "passed": ok_doc,
                "host_id": doc.get("host_id"),
                "receipt_sha256": doc.get("receipt_sha256"),
            }
            host_cals_ok = host_cals_ok and ok_doc

    passed = digest_ok and fragments_ok and not burned_missing and aa_ok and host_cals_ok
    return {
        "bundle_id": BUNDLE_ID,
        "passed": passed,
        "status": "local-drycheck-pass" if passed else "local-drycheck-fail",
        "spine_digest": actual_sha,
        "spine_digest_ok": digest_ok,
        "compiled_sha256": compiled_sha,
        "fragments_ok": fragments_ok,
        "fragment_error": fragment_error,
        "burn_ledger_sha256": ledger.sha256,
        "burned_corpora_checked": list(BURNED_CORPORA),
        "burned_corpora_missing": burned_missing,
        "aa_null_calibration": {
            "passed": aa_ok,
            "receipt_sha256": aa_receipt["receipt_sha256"],
            "host_id": aa_receipt["host_id"],
            "rejections": {name: metric["rejections"] for name, metric in aa_receipt["metrics"].items()},
            "inflation_p_values": {
                name: metric["inflation_p_value"] for name, metric in aa_receipt["metrics"].items()
            },
        },
        "committed_host_calibrations": host_cal_checks,
    }


def main() -> int:
    result = verify_bundle()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
