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
LIVE_MANIFEST_REL = "artifacts/pv7_w20_d13_pooled_agent_loops/pv7_w20_d13_manifest.json"
LIVE_RECEIPT_REL = "artifacts/pv7_w20_d13_pooled_agent_loops/report_receipt.json"


def verify_live_receipt(root: Path, spine_digest: str) -> Dict[str, Any]:
    """Verify the committed ``run-live-workers`` receipt, if present, and SURFACE ITS OUTCOME.

    Authenticity/consistency checks (``live_*``) are separate from the outcome. The committed live run FAILED two
    preregistered gates (``beats_full_restart_rebuild_jpm`` and ``sham_reopen_control_valid``): every one of the
    1680 eligible jobs missed the 560 ms SLA because live worker latency (about 3.7-7.9 s) dwarfs the deadline, so
    on-time-cited-correct jobs are 0 everywhere. This function verifies that the receipt honestly records that
    failure (it must NOT read as a live pass) and never converts it into a pass. The recorded measurements cannot be
    reproduced off the Spark host; only hashes, gate algebra and internal consistency are recomputed.
    """
    import hashlib

    from chelated_pooled_agent_runtime import JOB_SLA_DEADLINE_MS, _sha256_json

    manifest_path = root / LIVE_MANIFEST_REL
    receipt_path = root / LIVE_RECEIPT_REL
    if not (manifest_path.is_file() and receipt_path.is_file()):
        return {"status": "live-receipt-absent", "checks": [], "live_worker_outcome": "ABSENT"}
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    receipt = json.loads(receipt_path.read_bytes())
    checks: List[Dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": f"live_{name}", "passed": bool(ok), "detail": detail})

    gates = manifest.get("gates", {})
    cells = manifest.get("cells", [])
    cmp_ = manifest.get("full_restart_comparator", {})
    sham = manifest.get("sham_reopen_isolation", {})
    inter = manifest.get("primary_2x2_interaction", {})
    cluster = inter.get("rate_interaction_matter_cluster_bootstrap", {})
    check("receipt_schema", receipt.get("schema_version") == "pv7_report_receipt_v1" and receipt.get("bundle_id") == BUNDLE_ID
          and receipt.get("item") == "4" and receipt.get("demo_id") == "D13"
          and receipt.get("manifest_file") == "pv7_w20_d13_manifest.json")
    check("modes_are_run_live_workers", receipt.get("mode") == "run-live-workers" and manifest.get("mode") == "run-live-workers"
          and manifest.get("worker_backend") not in (None, ""), f"backend={manifest.get('worker_backend')}")
    check("receipt_spine_digest", receipt.get("spine_digest") == spine_digest)
    check("manifest_sha256_binds_bytes", receipt.get("manifest_sha256") == hashlib.sha256(manifest_bytes).hexdigest())
    check("served_model_matches_requested", bool(receipt.get("served_model")) and receipt.get("served_model") == receipt.get("requested_model")
          == manifest.get("requested_model") == manifest.get("served_model"), str(receipt.get("served_model")))
    canonical_cells = [
        {
            "pool_size": c["pool_size"], "schedule_id": c["schedule_id"], "branch": c["branch"], "system_id": c["system_id"],
            "event_regime": c["event_regime"], "eligible_jobs": c["eligible_jobs"],
            "on_time_cited_correct_jobs": c["on_time_cited_correct_jobs"], "stale_commits": c["stale_commits"],
            "late_or_timed_out_jobs": c["late_or_timed_out_jobs"],
            "on_time_cited_correct_jpm": round(c["on_time_cited_correct_jpm"], 4),
            "mean_latency_ms": round(c["mean_latency_ms"], 3),
            "ek5_trace_sha256": c["ek5_parity"]["canonical_trace_sha256"] if c.get("ek5_parity") else None,
        }
        for c in cells
    ]
    recomputed = _sha256_json({
        "bundle_id": BUNDLE_ID, "mode": manifest.get("mode"), "bridge_provenance": manifest.get("bridge_provenance"),
        "primary_interaction": inter, "sham_reopen_isolation": manifest.get("sham_reopen_isolation"),
        "comparator_summary": cmp_, "gates": gates, "canonical_cells_summary": canonical_cells,
    })
    check("deterministic_sha256_recomputed", manifest.get("deterministic_sha256") == recomputed
          and receipt.get("deterministic_sha256") == recomputed, f"recomputed={recomputed}")
    check("receipt_gates_equal_manifest_gates", receipt.get("gates") == gates and len(gates) == 8, f"gates={len(gates)}")
    all_pass = all(v is True for v in gates.values())
    check("overall_pass_equals_all_gates", manifest.get("overall_pass") is all_pass and receipt.get("overall_pass") is all_pass)
    expected_status = "replay-fail" if not all_pass else "local-drycheck-pass"
    check("status_matches_gate_algebra", manifest.get("status") == expected_status and receipt.get("status") == expected_status,
          f"status={receipt.get('status')}")
    check("jpm_gate_matches_comparator", gates.get("beats_full_restart_rebuild_jpm")
          is bool(cmp_.get("zeromu_ek5_pooled_jpm", 0) > cmp_.get("full_restart_rebuild_pooled_jpm", 0)))
    check("sham_gate_matches_control_valid", gates.get("sham_reopen_control_valid") is bool(sham.get("control_valid")))
    check("accounting_gate_recomputed", gates.get("all_eligible_jobs_accounted") is all(
        c["on_time_jobs"] + c["late_or_timed_out_jobs"] == c["eligible_jobs"] for c in cells) and bool(cells))
    check("cell_and_job_totals", len(cells) == manifest.get("total_cells_executed") == 70
          and sum(c["eligible_jobs"] for c in cells) == manifest.get("total_eligible_jobs_executed") == 1680
          and sum(c["live_model_calls"] for c in cells) == manifest.get("live_model_calls") > 0)
    check("stale_and_parity_counters_zero", manifest.get("zeromu_ek5_stale_commits_total") == 0
          and manifest.get("ek5_parity_failures_total") == 0)
    on_time = sum(c["on_time_cited_correct_jobs"] for c in cells)
    late = sum(c["late_or_timed_out_jobs"] for c in cells)
    check("sla_outcome_accounted", late + sum(c["on_time_jobs"] for c in cells) == 1680
          and on_time <= sum(c["on_time_jobs"] for c in cells), f"on_time_cited_correct={on_time} late={late}")
    drift = []
    for rel, recorded in sorted(receipt.get("source_file_sha256", {}).items()):
        current = hashlib.sha256((root / rel).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        if current != recorded:
            drift.append(rel)
    check("source_hash_fields_present", set(receipt.get("source_file_sha256", {})) == {
        "chelated_pooled_agent_runtime.py", "scripts/run_pv7_w20_d13_pooled_agent_loops.py"})
    failed_gates = sorted(k for k, v in gates.items() if v is not True)
    return {
        "status": "live-receipt-verified" if all(c["passed"] for c in checks) else "live-receipt-failed",
        "checks": checks,
        # AUTHENTICITY and OUTCOME are deliberately separate. The receipt is authentic; the live worker run FAILED.
        "live_worker_outcome": "PASSED" if all_pass else "FAILED",
        "failed_gates": failed_gates,
        "on_time_cited_correct_jobs": on_time, "eligible_jobs": sum(c["eligible_jobs"] for c in cells),
        "job_sla_deadline_ms": JOB_SLA_DEADLINE_MS,
        "jpm_advantage_over_full_restart": cmp_.get("jpm_advantage_over_full_restart"),
        "primary_interaction_lcb_strictly_positive": cluster.get("strictly_positive_lcb"),
        "source_drift_since_receipt": drift,
        "manifest_sha256": receipt.get("manifest_sha256"), "deterministic_sha256": recomputed,
        "outcome_disclosure": (
            "live worker run did NOT pass: all eligible jobs missed the SLA, the full-restart comparator ties at 0.0 "
            "JPM and the sham-reopen control is invalid; do not cite this bundle as a live pass"
        ),
    }


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
    live = verify_live_receipt(root, SPINE_DIGEST)
    checks.extend(live["checks"])
    passed = all(c["passed"] for c in checks)
    status = replay_status if passed and status_ok else "replay-fail"
    return {
        "bundle_id": BUNDLE_ID,
        "live_receipt": {k: v for k, v in live.items() if k != "checks"},
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
