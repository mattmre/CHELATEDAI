"""Offline verifier fragment for Wave 20 / Bundle #61 / Item 1 / Demo D12 `/fabric`."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

SPINE_DIGEST = "32693f54e0ce7ad3d401ceb752ae4723b0eee251f6d49b95cdfb26a6201971e9"
BUNDLE_ID = 61
DEPENDS_ON = ()

REGISTRY_FRAGMENT_REL = "docs/research/registry.d/61-t1-fabric-dpu-leo.json"
ROUTES_FRAGMENT_REL = "demo_workbench/routes.d/61-fabric.json"
VERIFIER_FRAGMENT_REL = "scripts/verifiers.d/61_verifier.py"
LIVE_REPORT_REL = "artifacts/pv7_w20_d12_network_orbital_fabric/pv7_w20_d12_fabric_report.json"
LIVE_RECEIPT_REL = "artifacts/pv7_w20_d12_network_orbital_fabric/report_receipt.json"


def verify_live_report(root: Path, spine_digest: str) -> Dict[str, Any]:
    """Cryptographically and structurally verify the committed ``run-live-encoders`` fabric report, if present.

    Re-derives the receipt's file SHA-256 and size, the ``scoreboard_sha256`` over the embedded scoreboard, the
    deterministic contact-schedule digest, the gate set, the public disclosure list and the honesty flags. The
    encoder-dependent retrieval numbers need real checkpoints and cannot be replayed here; they are checked only for
    internal consistency. Two of the five encoders in this report are STAND-INS (disclosed in the result).
    """
    import hashlib

    code_root = Path(__file__).resolve().parents[2]
    for entry in (code_root, code_root / "demo_workbench"):
        if str(entry) not in sys.path:
            sys.path.insert(0, str(entry))
    import network_orbital_engine as eng

    report_path = root / LIVE_REPORT_REL
    receipt_path = root / LIVE_RECEIPT_REL
    if not (report_path.is_file() and receipt_path.is_file()):
        return {"status": "live-report-absent", "checks": []}
    data = report_path.read_bytes()
    report = json.loads(data)
    receipt = json.loads(receipt_path.read_bytes())
    checks: List[Dict[str, Any]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"name": f"live_{name}", "passed": bool(ok), "detail": detail})

    sb = report.get("scoreboard", {})
    check("receipt_binds_report_bytes", receipt.get("sha256") == hashlib.sha256(data).hexdigest()
          and receipt.get("bytes") == len(data) and receipt.get("file") == "pv7_w20_d12_fabric_report.json")
    check("receipt_honesty_flags_false", receipt.get("hardware_asic_p4_silicon_in_loop_claimed") is False
          and receipt.get("orbital_flight_hardware_claimed") is False)
    check("report_schema_and_mode", report.get("schema") == "pv7_w20_d12_fabric_report_v1"
          and report.get("bundle_id") == BUNDLE_ID and report.get("demo_id") == "D12"
          and report.get("mode") == "run-live-encoders" and sb.get("encoder_mode") == "run-live-encoders")
    check("report_spine_digest", report.get("spine_digest") == spine_digest)
    check("honesty_flags_false_everywhere", all(
        holder.get(flag) is False
        for holder in (report, sb) for flag in ("hardware_asic_p4_silicon_in_loop_claimed", "orbital_flight_hardware_claimed")
    ) and "Software emulation only" in str(report.get("banner")))
    blob = json.dumps(sb, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    check("scoreboard_sha256_recomputed", report.get("scoreboard_sha256") == hashlib.sha256(blob).hexdigest())
    check("schedule_sha256_recomputed", report.get("schedule_sha256") == eng.build_contact_schedule()["schedule_sha256"])
    gates = report.get("gates", [])
    names = [g.get("name") for g in gates]
    check("gate_set_complete_and_all_pass", len(gates) == 16 and len(set(names)) == 16
          and all(g.get("passed") is True for g in gates) and report.get("passed") is True, f"gates={len(gates)}")
    check("disclosures_match_engine", report.get("disclosures") == list(eng.PUBLIC_DISCLOSURES)
          and sb.get("disclosures") == list(eng.PUBLIC_DISCLOSURES))
    status = report.get("encoder_status", {})
    live = sorted(k for k, v in status.items() if str(v).startswith("live:"))
    stand_in = sorted(k for k, v in status.items() if not str(v).startswith("live:"))
    check("encoder_accounting_consistent", live == sorted(report.get("live_encoders_used", []))
          and stand_in == sorted(report.get("dry_standin_fallbacks", [])) and len(live) + len(stand_in) == 5
          and (not stand_in or report.get("allow_dry_standin") is True), f"live={live} stand_in={stand_in}")
    oracle = sb.get("oracle", {})
    check("oracle_and_register_zero_disagreements", oracle.get("oracle_disagreements") == 0
          and oracle.get("register_disagreements") == 0 and oracle.get("reserved_forward_leaks_fenced") == 0
          and oracle.get("reserved_code_injected", 0) > 0 and oracle.get("traces", 0) >= 250)
    probe = report.get("http_probe", {})
    check("http_probe_clean", probe.get("hits") == 0 and not probe.get("http_errors") and probe.get("patterns", 0) > 0
          and probe.get("nda_ledger_isolated") is True and probe.get("forbidden_input_rejected_without_echo") is True)
    outcomes = report.get("outcomes", {})
    arms = {a["id"]: a for a in sb.get("arms", [])}
    fabric = arms.get("fabric_control", {}).get("combined", {})
    check("outcomes_recomputed_from_arms", bool(fabric)
          and outcomes.get("fabric_control_state_acc_ppm") == fabric.get("state_acc_ppm")
          and outcomes.get("fabric_control_flow_acc_ppm") == fabric.get("flow_acc_ppm")
          and outcomes.get("best_other_state_acc_ppm") == max(a["combined"]["state_acc_ppm"] for i, a in arms.items() if i != "fabric_control")
          and outcomes.get("best_other_flow_acc_ppm") == max(a["combined"]["flow_acc_ppm"] for i, a in arms.items() if i != "fabric_control"))
    engine_now = hashlib.sha256(eng.ENGINE_FILE_SHA256_SOURCE.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    return {
        "status": "live-report-verified" if all(c["passed"] for c in checks) else "live-report-failed",
        "checks": checks,
        "live_encoders": live, "stand_in_encoders": stand_in,
        "scoreboard_sha256": report.get("scoreboard_sha256"), "report_sha256": receipt.get("sha256"),
        "engine_source_drift_since_report": engine_now != report.get("engine_sha256"),
        "outcome_disclosure": "authored scenarios and authored arms 3/4; informational, not a market comparison",
    }


def verify_bundle(repo_root: Optional[Path | str] = None) -> Dict[str, Any]:
    """Run the deterministic offline CPU verification for Bundle #61 (`Demo D12`)."""
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    root = Path(repo_root).resolve() if repo_root is not None else Path(__file__).resolve().parents[2]
    for entry in (root, root / "demo_workbench"):
        if str(entry) not in sys.path:
            sys.path.insert(0, str(entry))

    from chelated_pv7_spine.claim_registry import claim_registry_sha256
    from chelated_pv7_spine.fragment_compiler import compile_fragments
    from scripts.run_pv7_w20_d12_network_orbital_fabric import execute

    checks: List[Dict[str, Any]] = []
    actual = claim_registry_sha256()
    checks.append(
        {
            "name": "spine_digest_match",
            "passed": actual == SPINE_DIGEST,
            "detail": f"expected={SPINE_DIGEST} actual={actual}",
        }
    )

    rels = (REGISTRY_FRAGMENT_REL, ROUTES_FRAGMENT_REL, VERIFIER_FRAGMENT_REL)
    sources = {rel: (root / rel).read_bytes() for rel in rels if (root / rel).is_file()}
    compiled_ok, detail = False, "fragments missing"
    if len(sources) == len(rels):
        compiled = compile_fragments(root, SPINE_DIGEST, sources)
        compiled_ok = (
            compiled["dependency_order"] == [BUNDLE_ID]
            and len(compiled["registry"]) == 1
            and compiled["registry"][0]["id"] == BUNDLE_ID
        )
        detail = f"compiled_sha256={compiled['compiled_sha256']}"
    checks.append({"name": "fragment_compiler_validation", "passed": compiled_ok, "detail": detail})

    report = execute("local-drycheck")
    for gate in report["gates"]:
        checks.append(
            {"name": f"drycheck_gate_{gate['name']}", "passed": bool(gate["passed"]), "detail": gate["detail"][:200]}
        )
    checks.append(
        {
            "name": "hardware_flags_false",
            "passed": report["hardware_asic_p4_silicon_in_loop_claimed"] is False
            and report["orbital_flight_hardware_claimed"] is False,
            "detail": "both hardware claims are false",
        }
    )
    live = verify_live_report(root, SPINE_DIGEST)
    checks.extend(live["checks"])
    passed = all(c["passed"] for c in checks)
    return {
        "bundle_id": BUNDLE_ID,
        "live_report": {k: v for k, v in live.items() if k != "checks"},
        "spine_digest": SPINE_DIGEST,
        "passed": passed,
        "checks": checks,
        "scoreboard_sha256": report["scoreboard_sha256"],
        "status": "local-drycheck-pass" if passed else "local-drycheck-fail",
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    _ = argv
    result = verify_bundle()
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
