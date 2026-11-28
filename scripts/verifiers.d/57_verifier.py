"""Offline verifier fragment for Wave 19 / Bundle #57 / Item 2B."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

SPINE_DIGEST = "32693f54e0ce7ad3d401ceb752ae4723b0eee251f6d49b95cdfb26a6201971e9"
BUNDLE_ID = 57
DEPENDS_ON = ()

REGISTRY_FRAGMENT_REL = "docs/research/registry.d/57-t2b-cartridge-array.json"
VERIFIER_REL = "scripts/verifiers.d/57_verifier.py"
LIVE_MANIFEST_REL = "artifacts/pv7_w19_2b_cartridges/pv7_w19_2b_manifest.json"
LIVE_RECEIPT_REL = "artifacts/pv7_w19_2b_cartridges/report_receipt.json"


def verify_live_receipt(root: Path, spine_digest: str) -> Dict[str, Any]:
    """Cryptographically and structurally verify the committed ``run-live`` cartridge receipt, if present.

    Re-derives ``manifest_sha256`` from the manifest bytes, ``deterministic_sha256`` from the recorded
    ``deterministic_view``, the zero-event Clopper-Pearson bound from ``requests_total`` and the gate set from
    the manifest. The cgroup / host facts, latencies and byte counts are recorded measurements and are only
    range-checked; this verifier cannot reproduce them off the Spark host.
    """
    import hashlib

    from chelated_cartridge_array import _canonical_digest

    manifest_path = root / LIVE_MANIFEST_REL
    receipt_path = root / LIVE_RECEIPT_REL
    if not (manifest_path.is_file() and receipt_path.is_file()):
        return {"status": "live-receipt-absent", "checks": []}
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    receipt = json.loads(receipt_path.read_bytes())
    checks: List[Dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": f"live_{name}", "passed": bool(ok), "detail": detail})

    gates = manifest.get("gates", {})
    view = manifest.get("deterministic_view", {})
    rev = manifest.get("revocation", {})
    swap = manifest.get("swap", {})
    cage = manifest.get("cage", {})
    check("receipt_schema", receipt.get("schema_version") == "pv7_report_receipt_v1" and receipt.get("bundle_id") == BUNDLE_ID
          and receipt.get("item") == "2B" and receipt.get("manifest_file") == "pv7_w19_2b_manifest.json")
    check("modes_are_run_live", receipt.get("mode") == "run-live" and manifest.get("mode") == "run-live"
          and view.get("mode") == "run-live", f"receipt={receipt.get('mode')} manifest={manifest.get('mode')}")
    check("receipt_spine_digest", receipt.get("spine_digest") == spine_digest)
    check("manifest_sha256_binds_bytes", receipt.get("manifest_sha256") == hashlib.sha256(manifest_bytes).hexdigest())
    recomputed = _canonical_digest(view)
    check("deterministic_sha256_recomputed", manifest.get("deterministic_sha256") == recomputed
          and receipt.get("deterministic_sha256") == recomputed, f"recomputed={recomputed}")
    check("gates_match_and_all_true", receipt.get("gates") == gates and bool(gates) and all(v is True for v in gates.values())
          and view.get("gate_names") == sorted(gates), f"gates={len(gates)}")
    check("overall_pass_supported", manifest.get("overall_pass") is True and receipt.get("overall_pass") is True)
    check("cuda_hidden_recorded", receipt.get("runtime", {}).get("cuda_visible_devices") == "")
    check("tier_is_disclosed", bool(manifest.get("tier")) and manifest.get("tier") == receipt.get("tier")
          and "physical cold-read" in str(manifest.get("tier_disclosure", "")), str(manifest.get("tier_disclosure")))
    check("catalog_and_materialized_counts", manifest.get("catalog_entry_count") == 1000
          and manifest.get("materialized_payload_count") == 250 == swap.get("swaps_measured")
          and view.get("catalog_entry_count") == 1000 and view.get("materialized_payload_count") == 250)
    check("cage_enforced_6gib_swap0", cage.get("enforced_on_this_host") is True and cage.get("memory_max_bytes") == 6 * 1024 ** 3
          and cage.get("memory_swap_max_bytes") == 0 and cage.get("enforcement_check", {}).get("enforced") is True)
    check("traced_peak_under_cage", 0 < int(manifest.get("traced_peak_bytes", 0)) < cage.get("memory_max_bytes", 0))
    trials = rev.get("trials", 0)
    cp = 1 - 0.05 ** (1.0 / rev.get("requests_total", 1)) if rev.get("requests_total") else None
    check("revocation_zero_post_enforcement", rev.get("post_enforcement_executions") == 0 and trials >= 300
          and rev.get("distinct_payloads_raced") == trials and rev.get("revoked_payloads_not_admitted") == rev.get("revoked_payloads")
          and rev.get("requests_total") == trials * (rev.get("inflight_per_trial", 0) + rev.get("queued_per_trial", 0)),
          f"trials={trials}")
    check("clopper_pearson_recomputed", cp is not None and abs(rev.get("cp_zero_event_closed_form", 1) - cp) < 1e-12
          and abs(rev.get("cp_one_sided_95_upper_bound", 1) - cp) < 1e-12 and cp < 0.01, f"closed_form={cp}")
    check("unguarded_control_shows_race", rev.get("unguarded_control_post_enforcement_executions", 0) > 0)
    check("ek5_parity_zero_diffs", all(rev.get(k) == 0 for k in (
        "ek5_semantic_diffs", "ek5_canonical_semantic_mismatches", "ek5_canonical_reason_diffs", "ek5_canonical_pos_cyc_diffs")))
    check("swap_integrity_and_bound", swap.get("integrity_failures") == 0 and swap.get("payload_bytes", 0) >= 8 * 1024 * 1024
          and swap.get("median_time_to_first_usable_inference_s", 0) >= swap.get("physical_bound_s", 1)
          and swap.get("overhead_ratio_vs_bound", 1e9) <= 50)
    # The recorded source hashes bind the receipt to the code that produced it; hash the LF-normalised working
    # copy so a CRLF checkout compares equal to the committed LF blob. Drift is reported, not failed: it means the
    # code was edited after the receipt was produced, not that the receipt is internally inconsistent.
    drift = []
    for rel, recorded in sorted(receipt.get("source_file_sha256", {}).items()):
        current = hashlib.sha256((root / rel).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if current != recorded:
            drift.append(rel)
    check("source_hash_fields_present", set(receipt.get("source_file_sha256", {})) == {
        "chelated_cartridge_array.py", "scripts/run_pv7_w19_2b_cartridges.py"})
    return {
        "status": "live-receipt-verified" if all(c["passed"] for c in checks) else "live-receipt-failed",
        "checks": checks, "source_drift_since_receipt": drift,
        "manifest_sha256": receipt.get("manifest_sha256"), "deterministic_sha256": recomputed,
    }


def verify_bundle(repo_root: Optional[Path | str] = None) -> Dict[str, Any]:
    """Run deterministic offline CPU verification for Bundle #57 (2B)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[2]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from chelated_cartridge_array import run_cartridge_array_experiment
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

    manifest = run_cartridge_array_experiment(mode="local-drycheck", revocation_trials=60)
    for gate_name, gate_ok in sorted(manifest["gates"].items()):
        checks.append({"name": f"gate_{gate_name}", "passed": bool(gate_ok), "detail": f"mode={manifest['mode']}"})

    live = verify_live_receipt(root, SPINE_DIGEST)
    checks.extend(live["checks"])
    passed = all(c["passed"] for c in checks)
    return {
        "bundle_id": BUNDLE_ID,
        "live_receipt": {k: v for k, v in live.items() if k != "checks"},
        "spine_digest": SPINE_DIGEST,
        "passed": passed,
        "tier": manifest["tier"],
        "checks": checks,
        "deterministic_sha256": manifest["deterministic_sha256"],
        "revocation": manifest["revocation"],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    _ = argv
    result = verify_bundle()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
