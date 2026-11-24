"""Wave-11 D8 Compliance Copilot runner: sweeps every scenario, seals a manifest and a report receipt.

Everything in the ``deterministic`` section of the manifest is recomputed from the live engine on each run
(no stored numbers). Wall-clock figures live only in ``nondeterministic`` and never enter a signed payload.

Usage:  python scripts/run_wave11_d8_compliance_copilot.py [--out-dir DIR] [--skip-http]
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import threading
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402

from demo_workbench import compliance_copilot_engine as E  # noqa: E402
from scripts.night1_common import current_git_sha, write_json_atomic  # noqa: E402
from scripts.verify_non_nda_ip_clean import FORBIDDEN_METHOD_PATTERNS, scan_text_for_ip_leaks  # noqa: E402

BUNDLE_ID = "wave11_d8_compliance_copilot"
MANIFEST_NAME = "wave11_d8_copilot_manifest.json"
DEFAULT_OUT = REPO_ROOT / "artifacts" / BUNDLE_ID
RUNNER_NDA_TOKEN = "runner-local-nda-token"  # in-process only; never written to any artifact
RUNNER_KEY_SEED = hashlib.sha256(b"wave11-d8-runner-fixed-public-test-key").digest()
CLOCK_START_US = 1_700_000_000_000_000


def make_runner_service() -> E.CopilotService:
    """Fixed *public test* key + counter clock so the sealed receipts are reproducible (not the live server key)."""
    state = {"t": CLOCK_START_US}

    def tick() -> int:
        state["t"] += 1
        return state["t"]

    return E.CopilotService(nda_token=RUNNER_NDA_TOKEN, clock_us=tick,
                            private_key=Ed25519PrivateKey.from_private_bytes(RUNNER_KEY_SEED))


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def find_line(path: Path, needle: str) -> int:
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if needle in line:
            return i
    raise RuntimeError(f"anchor not found: {needle!r}")


class Checks:
    def __init__(self) -> None:
        self.items: List[Dict[str, Any]] = []

    def add(self, name: str, ok: bool, detail: Any = None) -> bool:
        self.items.append({"check": name, "pass": bool(ok), "detail": detail})
        return bool(ok)

    @property
    def all_pass(self) -> bool:
        return all(c["pass"] for c in self.items)


def jget(svc: E.CopilotService, path: str, params: Optional[Dict[str, str]] = None,
         headers: Optional[Dict[str, str]] = None) -> Tuple[int, str, Dict[str, Any]]:
    status, _, body = E.handle_copilot_get(path, params or {}, headers or {}, svc)
    text = body.decode("utf-8")
    return status, text, json.loads(text)


def jpost(svc: E.CopilotService, path: str, payload: Any, headers: Optional[Dict[str, str]] = None) -> Tuple[int, str, Dict[str, Any]]:
    raw = payload if isinstance(payload, bytes) else json.dumps(payload).encode("utf-8")
    status, _, body = E.handle_copilot_post(path, {}, headers or {}, raw, svc)
    text = body.decode("utf-8")
    return status, text, json.loads(text)


def sweep_non_nda(svc: E.CopilotService, checks: Checks, scratch_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    responses = 0
    leak_hits: List[str] = []
    unscripted_failures = 0
    per_arm = {a: {"correct": 0, "judged": 0, "stale_unflagged": 0, "forbidden_text": 0, "abstained": 0}
               for a in E.ARM_IDS}
    truth_ok = True
    nodes = mism = 0
    for sid, sc in E.SCENARIOS.items():
        for k in range(1, len(sc.steps) + 1):
            for role in sc.roles:
                status, text, parsed = jget(svc, f"{E.API_PREFIX}/step", {"scenario": sid, "step": str(k), "role": role})
                responses += 1
                if status != 200:
                    unscripted_failures += 1
                    continue
                leak_hits.extend(scan_text_for_ip_leaks(text, f"{sid}/{k}/{role}"))
                if "internal" in parsed:
                    leak_hits.append(f"{sid}/{k}/{role}: internal block present in non_nda")
                truth_ok = truth_ok and parsed["ground_truth"]["matches_control_plane"]
                nodes += parsed["cross_check"]["nodes_checked"]
                mism += parsed["cross_check"]["mismatches"]
                if role == sc.roles[0]:
                    for arm in parsed["arms"]:
                        s = arm["summary"]
                        t = per_arm[arm["arm_id"]]
                        t["correct"] += s["correct"]
                        t["judged"] += s["judged"]
                        t["stale_unflagged"] += s["stale_unflagged_count"]
                        t["forbidden_text"] += s["forbidden_text_count"]
                        t["abstained"] += sum(1 for q in arm["queries"] if q["abstained"])
                scratch_rows.append({"scenario": sid, "step": k, "role": role, "response": parsed})
    checks.add("non_nda sweep: every scenario x step x role returned 200", unscripted_failures == 0,
               {"responses": responses, "non_200": unscripted_failures})
    checks.add("non_nda sweep: 0 forbidden-pattern hits across all responses", not leak_hits, leak_hits[:5])
    checks.add("control plane == authored ground truth on every step/role", truth_ok)
    checks.add("independent solver agrees with kernel (0 mismatches)", mism == 0, {"nodes_checked": nodes, "mismatches": mism})
    ctl = per_arm["lineage_control"]
    checks.add("control plane: all judged questions correct", ctl["correct"] == ctl["judged"] and ctl["judged"] > 0, ctl)
    checks.add("control plane: 0 unflagged stale and 0 erased-text exposures",
               ctl["stale_unflagged"] == 0 and ctl["forbidden_text"] == 0, ctl)
    checks.add("baselines expose withdrawn content (non-vacuous comparison)",
               per_arm["baseline_rag"]["stale_unflagged"] > 0 and per_arm["overwrite_policy"]["stale_unflagged"] > 0,
               {a: per_arm[a]["stale_unflagged"] for a in ("baseline_rag", "overwrite_policy")})
    return {"responses_scanned": responses, "per_arm_totals_standard_role": per_arm,
            "oracle": {"nodes_checked": nodes, "mismatches": mism}}


def sweep_nda(svc: E.CopilotService, checks: Checks) -> Dict[str, Any]:
    hdr = {E.NDA_TOKEN_HEADER: RUNNER_NDA_TOKEN}
    ok_internal = 0
    total = 0
    for sid, sc in E.SCENARIOS.items():
        for k in range(1, len(sc.steps) + 1):
            status, _, parsed = jget(svc, f"{E.API_PREFIX}/step",
                                     {"scenario": sid, "step": str(k), "mode": "nda"}, hdr)
            total += 1
            if status == 200 and "internal" in parsed:
                ok_internal += 1
    no_token = jget(svc, f"{E.API_PREFIX}/step", {"scenario": "w1", "step": "1", "mode": "nda"})[0]
    wrong = jget(svc, f"{E.API_PREFIX}/step", {"scenario": "w1", "step": "1", "mode": "nda"},
                 {E.NDA_TOKEN_HEADER: "wrong"})[0]
    unconfigured = E.CopilotService(nda_token=None)
    unconf = jget(unconfigured, f"{E.API_PREFIX}/step", {"scenario": "w1", "step": "1", "mode": "nda"},
                  {E.NDA_TOKEN_HEADER: ""})[0]
    checks.add("nda mode: internal block served on every step with the right token", ok_internal == total,
               {"ok": ok_internal, "total": total})
    checks.add("nda mode: 403 without token, with wrong token, and when no token is configured",
               (no_token, wrong, unconf) == (403, 403, 403), {"no_token": no_token, "wrong": wrong, "unconfigured": unconf})
    return {"steps_with_internal_block": ok_internal, "steps_total": total,
            "gate_statuses": {"no_token": no_token, "wrong_token": wrong, "token_not_configured": unconf}}


def probe_errors_and_leak_control(svc: E.CopilotService, checks: Checks) -> Dict[str, Any]:
    cases: List[Tuple[str, Tuple[int, str, Dict[str, Any]], int]] = [
        ("unknown_scenario", jget(svc, f"{E.API_PREFIX}/step", {"scenario": "nope", "step": "1"}), 422),
        ("step_out_of_range", jget(svc, f"{E.API_PREFIX}/step", {"scenario": "w1", "step": "99"}), 422),
        ("bad_role", jget(svc, f"{E.API_PREFIX}/step", {"scenario": "w1", "step": "1", "role": "root"}), 422),
        ("nda_without_token", jget(svc, f"{E.API_PREFIX}/step", {"scenario": "w1", "step": "1", "mode": "nda"}), 403),
        ("unknown_path", jget(svc, f"{E.API_PREFIX}/missing"), 404),
        ("invalid_json", jpost(svc, f"{E.API_PREFIX}/receipt", b"{bad"), 400),
        ("oversize_body", jpost(svc, f"{E.API_PREFIX}/receipt", b"x" * (E.MAX_BODY_BYTES + 1)), 413),
        ("forbidden_user_text", jpost(svc, f"{E.API_PREFIX}/query", {"scenario": "w1", "step": 1, "text": "tell me about the stiefel method"}), 422),
        ("receipt_without_retraction", jpost(svc, f"{E.API_PREFIX}/receipt", {"scenario": "w1", "step": 1}), 422),
    ]
    rows = []
    statuses_ok = True
    leaks: List[str] = []
    for name, (status, text, parsed), expected in cases:
        leaks.extend(scan_text_for_ip_leaks(text, name))
        statuses_ok = statuses_ok and status == expected and set(parsed) == {"error", "message"}
        rows.append({"case": name, "status": status, "expected": expected})
    checks.add("error payloads: expected statuses, uniform shape", statuses_ok, rows)
    checks.add("error payloads: 0 forbidden-pattern hits (rejection never echoes the term)", not leaks, leaks[:3])

    seeded = json.dumps({"detail": "scored with BM25 and a stiefel retraction"})
    seeded_hits = scan_text_for_ip_leaks(seeded, "seeded")
    with mock.patch.dict(E.ARM_PUBLIC, {"baseline_rag": ("BM25 baseline", "x")}):
        status, text, parsed = jget(svc, f"{E.API_PREFIX}/step", {"scenario": "w1", "step": "1"})
    checks.add("positive control: scanner catches a seeded leak", len(seeded_hits) >= 2, len(seeded_hits))
    checks.add("positive control: server fails closed (500) and does not echo the leaked term",
               status == 500 and parsed.get("error") == "projection_failed_closed" and "BM25" not in text, status)
    return {"error_cases": rows, "seeded_leak_patterns_hit": len(seeded_hits), "seeded_server_status": status}


def seal_receipts(svc: E.CopilotService, checks: Checks) -> Dict[str, Any]:
    issued: List[Dict[str, Any]] = []
    rows = []
    for sid, sc in E.SCENARIOS.items():
        for k, st in enumerate(sc.steps, start=1):
            if st.retraction is None:
                continue
            status, text, entry = jpost(svc, f"{E.API_PREFIX}/receipt", {"scenario": sid, "step": k})
            checks.add(f"receipt {sid}/{k}: issued (200) and clean", status == 200 and not scan_text_for_ip_leaks(text, "receipt"), status)
            v_status, _, verdict = jpost(svc, f"{E.API_PREFIX}/verify-receipt", {"receipt": entry})
            checks.add(f"receipt {sid}/{k}: verifies (hash, Ed25519, key, chain link, ledger)",
                       v_status == 200 and verdict["valid"] and verdict["in_ledger"], verdict["checks"])
            issued.append(entry)
            rows.append({"scenario": sid, "step": k, "seq": entry["payload"]["seq"], "entry_hash": entry["entry_hash"],
                         "prev_hash": entry["prev_hash"], "signature_b64": entry["signature_b64"],
                         "withdrawn_lineage": entry["payload"]["withdrawn_lineage"],
                         "tombstone_count": len(entry["payload"]["tombstones"]),
                         "oracle_mismatches": entry["payload"]["cross_check"]["mismatches"]})
    chain = svc.verify_ledger()
    checks.add("ledger hash chain intact", chain["intact"] and chain["length"] == len(issued), chain)
    tampered = copy.deepcopy(issued[0])
    tampered["payload"]["target_id"] = "tampered"
    t_verdict = svc.verify_receipt(tampered)
    checks.add("tamper case: edited payload rejected", not t_verdict["valid"] and not t_verdict["checks"]["hash_ok"], t_verdict["checks"])
    broken = copy.deepcopy(svc.ledger)
    broken[1]["payload"]["scenario_title"] = "edited"
    saved = svc.ledger
    svc.ledger = broken
    chain_broken = svc.verify_ledger()
    svc.ledger = saved
    checks.add("tamper case: edited ledger entry breaks the chain at that seq", (not chain_broken["intact"]) and chain_broken["first_bad_seq"] == 2, chain_broken)
    floats = []

    def walk(o: Any) -> None:
        if isinstance(o, float):
            floats.append(o)
        elif isinstance(o, dict):
            for v in o.values():
                walk(v)
        elif isinstance(o, list):
            for v in o:
                walk(v)

    walk([e["payload"] for e in issued])
    checks.add("receipt payloads carry integers only (no floats, no wall-clock latency)", not floats)
    return {"receipts": rows, "ledger": chain, "head_hash": chain.get("head_hash"),
            "signing_key": {"algorithm": "Ed25519", "key_id": svc.key_id, "public_key_b64": svc.public_key_b64,
                            "disclosure": "Runner-only fixed public test key so sealed receipts are reproducible; NOT the live server's ephemeral per-process key and carries no trust."}}


def undo_and_inject(svc: E.CopilotService, checks: Checks) -> Dict[str, Any]:
    out: Dict[str, Any] = {"undo": [], "inject": {}}
    for sid, expect in (("w1", 200), ("w3", 200), ("w2", 422)):
        status, _, body = jpost(svc, f"{E.API_PREFIX}/undo", {"scenario": sid, "step": 2})
        good = status == expect and (body.get("history_preserved") is True if expect == 200 else body.get("error") == "irreversible")
        checks.add(f"undo {sid}: {'compensating event, history preserved' if expect == 200 else 'irreversible erasure refused (422)'}", good, status)
        out["undo"].append({"scenario": sid, "status": status})
    status, text, body = jpost(svc, f"{E.API_PREFIX}/inject", {
        "events": [{"id": "a", "entity": "E", "slot": "s", "claim_value": "1", "text": "one", "tier": 1},
                   {"id": "b", "entity": "E", "slot": "t", "claim_value": "1", "text": "copy", "tier": 1},
                   {"id": "n", "entity": "E", "slot": "s", "claim_value": "2", "text": "two", "tier": 3}],
        "edges": [{"from": "b", "to": "a", "kind": "copy"}, {"from": "n", "to": "a", "kind": "retracts"}]})
    states = {r["id"]: r["state"] for r in body.get("records", [])}
    checks.add("custom injection: oracle agrees and withdrawal propagates through the copy",
               status == 200 and body["cross_check"]["agrees"] and states == {"a": "Withdrawn", "b": "Withdrawn", "n": "Current"}
               and not scan_text_for_ip_leaks(text, "inject"), states)
    dispute_status, _, dispute = jpost(svc, f"{E.API_PREFIX}/inject", {
        "events": [{"id": "a", "entity": "E", "slot": "s", "claim_value": "1", "text": "x", "tier": 2},
                   {"id": "b", "entity": "E", "slot": "s", "claim_value": "2", "text": "y", "tier": 2}],
        "edges": [{"from": "a", "to": "b", "kind": "retracts"}, {"from": "b", "to": "a", "kind": "retracts"}]})
    checks.add("custom injection: equal-authority dispute abstains", dispute_status == 200 and dispute["disputed_count"] == 2)
    cyc_status = jpost(svc, f"{E.API_PREFIX}/inject", {
        "events": [{"id": "a", "text": "x", "tier": 2}, {"id": "b", "text": "y", "tier": 2}],
        "edges": [{"from": "a", "to": "b", "kind": "replaces"}, {"from": "b", "to": "a", "kind": "replaces"}]})[0]
    checks.add("custom injection: replacement cycle rejected (422)", cyc_status == 422, cyc_status)
    out["inject"] = {"states": states, "dispute_disputed_count": dispute.get("disputed_count"), "cycle_status": cyc_status}
    return out


def http_smoke(checks: Checks, nondet: Dict[str, Any]) -> Dict[str, Any]:
    svc = make_runner_service()
    srv = E.build_server("127.0.0.1", 0, svc)
    port = srv.server_address[1]
    thread = threading.Thread(target=srv.serve_forever, daemon=True)
    thread.start()
    rows: List[Dict[str, Any]] = []
    cors = False
    leaks: List[str] = []

    def req(method: str, path: str, body: Any = None, headers: Optional[Dict[str, str]] = None) -> Tuple[int, Dict[str, str], bytes]:
        data = json.dumps(body).encode("utf-8") if body is not None else None
        hdrs = dict(headers or {})
        if data is not None:
            hdrs["Content-Type"] = "application/json"
        r = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method, headers=hdrs)
        try:
            with urllib.request.urlopen(r, timeout=30) as resp:
                return resp.status, dict(resp.headers.items()), resp.read()
        except urllib.error.HTTPError as err:
            return err.code, dict(err.headers.items()), err.read()

    try:
        plan = [
            ("GET", "/copilot", None, None, 200),
            ("GET", f"{E.API_PREFIX}/health", None, None, 200),
            ("GET", f"{E.API_PREFIX}/scenarios", None, None, 200),
            ("GET", f"{E.API_PREFIX}/step?scenario=w1&step=3&role=clinician", None, None, 200),
            ("GET", f"{E.API_PREFIX}/step?scenario=w2&step=3&role=privacy_officer", None, None, 200),
            ("GET", f"{E.API_PREFIX}/scoreboard", None, None, 200),
            ("POST", f"{E.API_PREFIX}/receipt", {"scenario": "w1", "step": 2}, None, 200),
            ("GET", f"{E.API_PREFIX}/ledger", None, None, 200),
            ("GET", f"{E.API_PREFIX}/step?scenario=w1&step=1&mode=nda", None, None, 403),
            ("GET", f"{E.API_PREFIX}/step?scenario=w1&step=1&mode=nda", None, {E.NDA_TOKEN_HEADER: RUNNER_NDA_TOKEN}, 200),
            ("OPTIONS", f"{E.API_PREFIX}/step", None, {"Origin": "http://example.invalid"}, 405),
            ("GET", f"{E.API_PREFIX}/missing", None, None, 404),
        ]
        receipt_entry: Optional[Dict[str, Any]] = None
        for method, path, body, hdrs, expected in plan:
            status, resp_headers, data = req(method, path, body, hdrs)
            has_cors = any(k.lower().startswith("access-control") for k in resp_headers)
            cors = cors or has_cors
            is_nda_ok = hdrs is not None and E.NDA_TOKEN_HEADER in hdrs and status == 200
            if not is_nda_ok:
                leaks.extend(scan_text_for_ip_leaks(data.decode("utf-8", "replace"), f"{method} {path}"))
            if method == "POST" and status == 200:
                receipt_entry = json.loads(data)
            rows.append({"request": f"{method} {path.split('?')[0]}", "status": status, "expected": expected, "cors_headers": has_cors})
        if receipt_entry is not None:
            status, _, data = req("POST", f"{E.API_PREFIX}/verify-receipt", {"receipt": receipt_entry})
            rows.append({"request": f"POST {E.API_PREFIX}/verify-receipt", "status": status, "expected": 200, "cors_headers": False})
            checks.add("http smoke: receipt issued over HTTP verifies over HTTP", status == 200 and json.loads(data)["valid"])
    finally:
        srv.shutdown()
        srv.server_close()
        thread.join(timeout=5)
    checks.add("http smoke: every request returned its expected status", all(r["status"] == r["expected"] for r in rows),
               [r for r in rows if r["status"] != r["expected"]])
    checks.add("http smoke: no Access-Control-* header on any response (same-origin only)", not cors)
    checks.add("http smoke: 0 forbidden-pattern hits in non_nda HTTP bodies (page, API, errors)", not leaks, leaks[:3])
    nondet["http_smoke_port_note"] = "ephemeral loopback port; not recorded"
    return {"requests": rows, "tier": "floor"}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run and seal Wave-11 D8 Compliance Copilot evidence.")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--skip-http", action="store_true", help="skip the live HTTP smoke (not a sealed configuration)")
    args = parser.parse_args(argv)
    out_dir = Path(args.out_dir)
    scratch_dir = out_dir / "scratch"
    checks = Checks()
    nondet: Dict[str, Any] = {}
    svc = make_runner_service()
    scratch_rows: List[Dict[str, Any]] = []

    html_text = E.HTML_PATH.read_text(encoding="utf-8")
    html_hits = scan_text_for_ip_leaks(html_text, "compliance_copilot.html")
    checks.add("pattern list has exactly 56 entries", len(FORBIDDEN_METHOD_PATTERNS) == 56, len(FORBIDDEN_METHOD_PATTERNS))
    checks.add("HTML page: 0 forbidden-pattern hits", not html_hits, html_hits[:3])
    checks.add("HTML page: no innerHTML / eval / document.write",
               not any(t in html_text for t in ("innerHTML", "eval(", "document.write")))

    sweep = sweep_non_nda(svc, checks, scratch_rows)
    nda = sweep_nda(svc, checks)
    errors = probe_errors_and_leak_control(svc, checks)
    sc_board = E.scoreboard()
    checks.add("scoreboard consistent with sweep", sc_board["cross_check"]["mismatches"] == 0
               and sc_board["ground_truth_matches_control_plane_everywhere"])
    receipts = seal_receipts(svc, checks)
    actions = undo_and_inject(svc, checks)

    bench = E.warm_benchmark(repeats=30)
    checks.add("benchmark ran (warm store; non-deterministic; excluded from receipts)",
               bench["incremental_update_ms"]["p50"] <= bench["incremental_update_ms"]["p95"])
    nondet["warm_benchmark"] = bench

    if args.skip_http:
        http = {"requests": [], "tier": "skipped"}
        checks.add("http smoke executed", False, "--skip-http given; manifest is not a sealed configuration")
    else:
        http = http_smoke(checks, nondet)

    engine_path = Path(E.__file__)
    disclosures = [
        {"id": "L4", "what": "Vector side of hybrid retrieval is a hashed signed char-3gram vector (dim 512), not a neural encoder.",
         "where": f"demo_workbench/compliance_copilot_engine.py:{find_line(engine_path, 'def dense_embed')}"},
        {"id": "L3", "what": "Arm 2 (overwrite policy) is an emulation of newest-record-wins, not a live product; arm 1 uses a real delete-by-id.",
         "where": f"demo_workbench/compliance_copilot_engine.py:{find_line(engine_path, 'class BaselineArms')}"},
        {"id": "L4", "what": "Receipt signing key is generated per process in memory; issued_at_us is the untrusted server clock. The sealed receipts here use a fixed public runner test key.",
         "where": f"demo_workbench/compliance_copilot_engine.py:{find_line(engine_path, 'class CopilotService')}"},
        {"id": "L4", "what": "Audience filtering (standard vs privileged role) is Copilot-layer policy outside the decision engine.",
         "where": f"demo_workbench/compliance_copilot_engine.py:{find_line(engine_path, 'def _lineage_hits')}"},
        {"id": "L5", "what": "Warm-store p50/p95 is fixture-sized (tens of nodes), single machine, non-deterministic; it implies nothing about production scale.",
         "where": f"demo_workbench/compliance_copilot_engine.py:{find_line(engine_path, 'def warm_benchmark')}"},
        {"id": "L4", "what": "Custom-evidence injection reports oracle agreement only; no accuracy claim is made for user-supplied records.",
         "where": f"demo_workbench/compliance_copilot_engine.py:{find_line(engine_path, 'def inject_evidence')}"},
        {"id": "L5", "what": "Smoke tier is FLOOR: real ThreadingHTTPServer on a loopback port plus in-process handler calls over synthetic fixtures. No ceiling-tier run against an external regulated corpus, a hardened deployment, or a trusted timestamp authority was performed.",
         "where": "scripts/run_wave11_d8_compliance_copilot.py:http_smoke"},
        {"id": "L1", "what": "server.py forwarder is NOT wired by this slice (file ownership); the page is reachable only via the standalone server until the orchestrator adds the forwarder.",
         "where": "demo_workbench/compliance_copilot_engine.py:handle_copilot_get"},
        {"id": "L4", "what": "NDA token is supplied by constructor argument or env CHELATED_COPILOT_NDA_TOKEN; the standalone CLI has no --nda-token flag.",
         "where": f"demo_workbench/compliance_copilot_engine.py:{find_line(engine_path, 'def main')}"},
    ]

    scratch_dir.mkdir(parents=True, exist_ok=True)
    scratch_file = scratch_dir / "raw_step_responses.json"
    scratch_hash = write_json_atomic(scratch_file, scratch_rows)
    scratch_size = scratch_file.stat().st_size

    manifest: Dict[str, Any] = {
        "schema_version": "wave11_d8_copilot_manifest_v1",
        "bundle_id": BUNDLE_ID,
        "spec": "wave10-r1-synthesis-guardrails.md L3 (Compliance Copilot)",
        "all_checks_pass": checks.all_pass,
        "checks_total": len(checks.items),
        "checks_failed": [c["check"] for c in checks.items if not c["pass"]],
        "checks": checks.items,
        "deterministic": {
            "scenarios": {sid: {"title": sc.title, "steps": len(sc.steps), "roles": list(sc.roles)} for sid, sc in E.SCENARIOS.items()},
            "retrieval": {"top_k": E.TOP_K, "rrf_k": E.RRF_K, "dense_dim": E.DENSE_DIM, "reader_policy": E.READER_POLICY},
            "non_nda_sweep": sweep,
            "nda_sweep": nda,
            "error_and_leak_controls": errors,
            "scoreboard": {"arms": sc_board["arms"], "cross_check": sc_board["cross_check"]},
            "receipts": receipts,
            "actions": actions,
            "http_smoke": http,
            "forbidden_pattern_count": len(FORBIDDEN_METHOD_PATTERNS),
            "files_sha256": {
                "demo_workbench/compliance_copilot_engine.py": E.engine_file_sha256(),
                "demo_workbench/compliance_copilot.html": sha256_text(html_text),
            },
        },
        "nondeterministic": nondet,
        "smoke_tier": "floor",
        "brutal_honesty": {
            "stubs_L1": "none in the four delivered code files; server.py forwarder is intentionally not wired by this slice",
            "disclosures": disclosures,
            "mocks_in_production_paths_L3": "none; the only mock is the seeded-leak positive control, applied in tests/runner only",
            "broad_catch_swallowing_L11": "none; every except clause names its exception types and returns a typed error",
        },
        "scratch": {
            "path": str(scratch_dir),
            "removable": True,
            "contents": [{"file": str(scratch_file), "bytes": scratch_size, "sha256": scratch_hash}],
            "total_bytes": scratch_size,
            "note": "Raw per-step responses; regenerable by re-running this script. Removal needs the full-path list, count and size shown to the owner first.",
        },
    }
    manifest_path = out_dir / MANIFEST_NAME
    manifest_sha = write_json_atomic(manifest_path, manifest)
    receipt = {
        "artifact_hashes": {MANIFEST_NAME: manifest_sha},
        "bundle_id": BUNDLE_ID,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": current_git_sha(REPO_ROOT),
        "manifest_file": MANIFEST_NAME,
        "manifest_sha256": manifest_sha,
        "schema_version": "report_receipt_v4_0",
        "status": "COMPLETED" if checks.all_pass else "FAILED",
    }
    write_json_atomic(out_dir / "report_receipt.json", receipt)
    passed = sum(1 for c in checks.items if c["pass"])
    print(f"D8 compliance copilot: {passed}/{len(checks.items)} checks passed; status={receipt['status']}; "
          f"manifest_sha256={manifest_sha}; smoke_tier=floor; scratch={scratch_size} bytes at {scratch_dir}")
    for c in checks.items:
        if not c["pass"]:
            print(f"  FAILED: {c['check']} -> {c['detail']}", file=sys.stderr)
    return 0 if checks.all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
