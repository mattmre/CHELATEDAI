"""CLI runner for Wave 20 / Bundle #61 / Item 1 / Demo D12 `/fabric` (software emulation only).

Two execution modes:
  1. `--mode local-drycheck` (default, CPU-only, `CUDA_VISIBLE_DEVICES=""`): deterministic text-derived
     stand-in encoders. Exercises the full control plane, the seven arms, the independent prefix oracle
     (Scenario A, Scenario B and >= 250 randomized traces) and the receipt chain. No model weights are read.
  2. `--mode run-live-encoders`: uses real checkpoints (MiniLM-L6-v2 384-d, bge-small-en-v1.5 384-d,
     all-mpnet-base-v2 768-d, bge-m3 1024-d, Qwen3-8B 4096-d) for every encoder that is cached locally
     and loadable offline on CPU. An encoder that is missing degrades to its stand-in and the report
     says so per encoder; nothing is downloaded and no GPU is used.

Every report carries ``hardware_asic_p4_silicon_in_loop_claimed: false`` and
``orbital_flight_hardware_claimed: false``: flows and orbital elements are synthetic and no switch,
DPU, ASIC or flight hardware was involved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "demo_workbench"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import network_orbital_engine as eng  # noqa: E402
from chelated_pv7_spine.claim_registry import claim_registry_sha256  # noqa: E402
from scripts.verify_non_nda_ip_clean import assert_non_nda_clean  # noqa: E402

BUNDLE_ID = 61
DEMO_ID = "D12"
MODES = ("local-drycheck", "run-live-encoders")
MIN_FUZZ_TRACES = 250


def _gate(name: str, passed: bool, detail: str) -> Dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _receipt_roundtrip() -> Dict[str, Any]:
    svc = eng.FabricService(nda_token=None)
    issued = [svc.issue_receipt("A", k, "non_nda") for k in range(1, 7)]
    issued += [svc.issue_receipt("B", k, "non_nda") for k in range(1, 6)]
    valid = all(svc.verify_receipt(e)["valid"] for e in issued)
    tampered = json.loads(json.dumps(issued[0]))
    tampered["payload"]["step_title"] += "!"
    tamper_detected = not svc.verify_receipt(tampered)["valid"]
    forged_key = json.loads(json.dumps(issued[1]))
    forged_key["payload"]["register_hex"] = "00" * 4
    forged_detected = not svc.verify_receipt(forged_key)["valid"]
    chain = svc.verify_ledger()
    int_only = True
    try:
        for e in issued:
            eng.canonical_json(e["payload"])
    except ValueError:
        int_only = False
    return {"issued": len(issued), "all_valid": valid, "tamper_detected": tamper_detected,
            "forged_register_detected": forged_detected, "ledger": chain, "integer_only_payloads": int_only}


def execute(mode: str = "local-drycheck") -> Dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"unsupported mode {mode!r}")
    eng.set_encoder_mode(mode)
    sb = eng.scoreboard("non_nda")
    ev = eng.get_eval(mode)
    fr = ev["fr"]
    runs = ev["runs"]
    oracle = sb["oracle"]
    arms = {a["id"]: a for a in sb["arms"]}
    fabric = arms["fabric_control"]
    ret = sb["retrieval"]
    sched = eng.build_contact_schedule()
    gates: List[Dict[str, Any]] = []

    gates.append(_gate("honesty_flags_false",
                       sb["hardware_asic_p4_silicon_in_loop_claimed"] is False
                       and sb["orbital_flight_hardware_claimed"] is False,
                       "both hardware flags are false in the scoreboard payload"))
    gates.append(_gate("oracle_traces_cover_scenarios_and_fuzz",
                       len(runs["A"]["steps"]) == 6 and len(runs["B"]["steps"]) == 5
                       and ev["fuzz"].traces >= MIN_FUZZ_TRACES,
                       f"scenario_a_steps={len(runs['A']['steps'])} scenario_b_steps={len(runs['B']['steps'])} "
                       f"fuzz_traces={ev['fuzz'].traces}"))
    gates.append(_gate("oracle_disagreements_zero", oracle["oracle_disagreements"] == 0,
                       f"prefixes={oracle['prefixes']} disagreements={oracle['oracle_disagreements']}"))
    gates.append(_gate("register_oracle_disagreements_zero", oracle["register_disagreements"] == 0,
                       f"register_checks={oracle['register_checks']} disagreements={oracle['register_disagreements']}"))
    gates.append(_gate("reserved_code_never_forwards_when_fenced",
                       oracle["reserved_code_injected"] > 0 and oracle["reserved_forward_leaks_fenced"] == 0,
                       f"injected={oracle['reserved_code_injected']} fenced_leaks={oracle['reserved_forward_leaks_fenced']} "
                       f"naive_leaks={oracle['reserved_forward_leaks_naive']}"))
    kk = ret["kappa_within_tol_all_hops"]
    gates.append(_gate("scale_factor_within_tolerance_on_bridge_paths",
                       bool(kk["orthogonal_chain"] and kk["fabric_chain"]),
                       f"|kappa-1|<{eng.KAPPA_TOL} on all hops: {json.dumps(kk, sort_keys=True)}"))
    gates.append(_gate("sidecar_is_8_bytes_per_flow_and_argmax_identical",
                       all(s["bytes_per_flow"] == eng.SIDECAR_BYTES_PER_FLOW and s["argmax_identical"]
                           for s in ret["sidecar"].values()),
                       json.dumps({h: [s["bytes_per_flow"], s["max_abs_parity_ppm"]] for h, s in ret["sidecar"].items()},
                                  sort_keys=True)))
    gates.append(_gate("degenerate_anchor_set_refused", bool(ret["bridge_health"]["degenerate_anchor_refused"]),
                       ret["bridge_health"]["degenerate_anchor_status"]))
    fa, fb = runs["A"]["arms"]["chelated_fabric_control"], runs["B"]["arms"]["chelated_fabric_control"]
    gates.append(_gate("audit_records_survive_in_canonical_trace",
                       fa["audit_survived"] == fa["audit_total"] and fb["audit_survived"] == fb["audit_total"]
                       and fa["audit_invalid_parent_survived"] == fa["audit_invalid_parent_total"]
                       and fb["audit_invalid_parent_survived"] == fb["audit_invalid_parent_total"],
                       f"A={fa['audit_survived']}/{fa['audit_total']} B={fb['audit_survived']}/{fb['audit_total']}"))
    gates.append(_gate("echo_clusters_collapse_observably",
                       all(sb["echo_collapse"][s]["echo_clusters"] < sb["echo_collapse"][s]["echo_records"]
                           for s in ("scenario_a", "scenario_b")),
                       json.dumps(sb["echo_collapse"], sort_keys=True)))
    gates.append(_gate("contact_schedule_has_oisl_ground_and_blackout_windows",
                       bool(sched["tle_checksums_verified"]) and bool(sched["ground_windows"])
                       and bool(sched["oisl_windows"]) and bool(sched["oceanic_blackout_windows"]),
                       json.dumps(sched["summary"], sort_keys=True)))
    rr = _receipt_roundtrip()
    gates.append(_gate("receipt_chain_roundtrip_and_tamper_detection",
                       rr["all_valid"] and rr["tamper_detected"] and rr["forged_register_detected"]
                       and rr["ledger"]["intact"] and rr["integer_only_payloads"],
                       json.dumps({k: v for k, v in rr.items() if k != "ledger"}, sort_keys=True)))
    try:
        assert_non_nda_clean(json.dumps(sb, sort_keys=True), "scoreboard")
        scan_ok, scan_detail = True, "0 forbidden-pattern hits in the non_nda scoreboard"
    except ValueError as exc:
        scan_ok, scan_detail = False, str(exc)[:300]
    gates.append(_gate("non_nda_projection_clean", scan_ok, scan_detail))

    outcomes = {
        "fabric_control_state_acc_ppm": fabric["combined"]["state_acc_ppm"],
        "fabric_control_flow_acc_ppm": fabric["combined"]["flow_acc_ppm"],
        "best_other_state_acc_ppm": max(a["combined"]["state_acc_ppm"] for i, a in arms.items() if i != "fabric_control"),
        "best_other_flow_acc_ppm": max(a["combined"]["flow_acc_ppm"] for i, a in arms.items() if i != "fabric_control"),
        "fabric_control_leads_state": all(fabric["combined"]["state_acc_ppm"] >= a["combined"]["state_acc_ppm"]
                                          for a in arms.values()),
        "fabric_control_leads_flow": all(fabric["combined"]["flow_acc_ppm"] >= a["combined"]["flow_acc_ppm"]
                                         for a in arms.values()),
        "note": "Informational. Arms 3 and 4 are authored emulations; scenarios are authored; this is not a market comparison.",
    }
    det_blob = json.dumps(sb, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    report: Dict[str, Any] = {
        "schema": "pv7_w20_d12_fabric_report_v1",
        "bundle_id": BUNDLE_ID, "demo_id": DEMO_ID, "mode": mode,
        "hardware_asic_p4_silicon_in_loop_claimed": False,
        "orbital_flight_hardware_claimed": False,
        "banner": eng.HONESTY_BANNER,
        "spine_digest": claim_registry_sha256(),
        "engine_sha256": eng.engine_file_sha256(),
        "schedule_sha256": sched["schedule_sha256"],
        "encoder_status": {k: v for k, v in fr.enc.status.items()},
        "live_encoders_used": [k for k, v in fr.enc.status.items() if v.startswith("live:")],
        "python": platform.python_version(),
        "scoreboard_sha256": hashlib.sha256(det_blob).hexdigest(),
        "gates": gates,
        "passed": all(g["passed"] for g in gates),
        "outcomes": outcomes,
        "scoreboard": sb,
        "disclosures": list(eng.PUBLIC_DISCLOSURES),
    }
    return report


def write_report(report: Dict[str, Any], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "pv7_w20_d12_fabric_report.json"
    data = (json.dumps(report, indent=2, sort_keys=True, ensure_ascii=True) + "\n").encode("utf-8")
    path.write_bytes(data)
    receipt = {"file": path.name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
               "hardware_asic_p4_silicon_in_loop_claimed": False, "orbital_flight_hardware_claimed": False}
    (out_dir / "report_receipt.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mode", choices=MODES, default="local-drycheck")
    parser.add_argument("--out-dir", type=Path, default=None, help="optional directory for the JSON report and its SHA-256 receipt")
    parser.add_argument("--summary-only", action="store_true", help="print gates and outcomes only")
    args = parser.parse_args(argv)
    report = execute(args.mode)
    if args.out_dir is not None:
        write_report(report, args.out_dir)
    shown = {k: v for k, v in report.items() if k != "scoreboard"} if args.summary_only else report
    print(json.dumps(shown, indent=2, sort_keys=True, ensure_ascii=True))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
