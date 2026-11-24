"""Wave-11 D8 Compliance Copilot tests (unittest-compatible; no pytest-only fixtures)."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import re
import sys
import threading
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest import mock
from urllib.parse import urlparse

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey  # noqa: E402

from demo_workbench import compliance_copilot_engine as E  # noqa: E402
from scripts.verify_non_nda_ip_clean import (  # noqa: E402
    FORBIDDEN_METHOD_PATTERNS,
    scan_text_for_ip_leaks,
)

TOKEN = "unit-test-token"
TOMBSTONE_RE = re.compile(r"^\[REDACTED_GDPR_ART17:sha256:[0-9a-f]{64}\]$")


def make_service(token=TOKEN, clock=1_700_000_000_000_000, key=None):
    state = {"t": clock}

    def tick():
        state["t"] += 1
        return state["t"]

    return E.CopilotService(nda_token=token, clock_us=tick, private_key=key)


def get(svc, path, **params):
    status, ctype, body = E.handle_copilot_get(path, params, {}, svc)
    return status, ctype, body


def get_json(svc, path, headers=None, **params):
    status, _, body = E.handle_copilot_get(path, params, headers or {}, svc)
    return status, json.loads(body.decode("utf-8"))


def post_json(svc, path, payload, headers=None):
    status, _, body = E.handle_copilot_post(path, {}, headers or {}, json.dumps(payload).encode("utf-8"), svc)
    return status, json.loads(body.decode("utf-8"))


def all_steps():
    for sid, sc in E.SCENARIOS.items():
        for k in range(1, len(sc.steps) + 1):
            yield sid, k, sc


class TestFixturesAgainstKernel(unittest.TestCase):
    def test_pattern_count_is_56(self):
        self.assertEqual(len(FORBIDDEN_METHOD_PATTERNS), 56)

    def test_control_plane_matches_authored_ground_truth_every_step(self):
        for sid, k, sc in all_steps():
            lin = E.replay(sid, k).lineage
            spec = sc.steps[k - 1]
            current = {i for i in lin.docs if lin.eligibility(i) == "ADMIT"}
            contested = {i for i in lin.docs if lin.eligibility(i) == "CONTESTED"}
            self.assertEqual(current, set(spec.expected_current), f"{sid} step {k}")
            self.assertEqual(contested, set(spec.expected_contested), f"{sid} step {k}")

    def test_independent_solver_agrees_with_kernel_every_step(self):
        for sid, k, _ in all_steps():
            n, mism = E.replay(sid, k).lineage.oracle_check()
            self.assertGreater(n, 0)
            self.assertEqual(mism, 0, f"{sid} step {k}")

    def test_no_authority_blocked_edges_in_fixtures(self):
        for sid, k, _ in all_steps():
            telem = E.replay(sid, k).lineage.store.get_telemetry()
            self.assertEqual(list(telem.blocked_authority_edges), [], f"{sid} step {k}")

    def test_only_valid_edge_types_used(self):
        for sc in E.SCENARIOS.values():
            for st in sc.steps:
                for _, _, t in st.edges:
                    self.assertIn(t, E.V3_VALID_EDGE_TYPES)
                    self.assertNotEqual(t, "DERIVES_VALUE")

    def test_zero_calls_to_simulator_module(self):
        import ast

        src = (REPO_ROOT / "demo_workbench" / "compliance_copilot_engine.py").read_text(encoding="utf-8")
        imported = set()
        for node in ast.walk(ast.parse(src)):
            if isinstance(node, ast.Import):
                imported.update(a.name for a in node.names)
            elif isinstance(node, ast.ImportFrom):
                imported.add(node.module or "")
        for name in imported:
            self.assertNotIn(name, ("server", "demo_workbench.server"))
            self.assertFalse(name.startswith("demo_workbench"), name)


class TestArms(unittest.TestCase):
    def arm(self, sid, k, arm_id, role=None):
        ev = E.evaluate_step(sid, k, role or E.SCENARIOS[sid].roles[0])
        return next(a for a in ev["arms"] if a["arm_id"] == arm_id)

    def test_control_plane_answers_every_scripted_question_correctly(self):
        for sid, k, _ in all_steps():
            for role in E.SCENARIOS[sid].roles:
                a = self.arm(sid, k, "lineage_control", role)
                s = a["summary"]
                self.assertEqual(s["correct"], s["judged"], f"{sid} step {k} {role}")
                self.assertEqual(s["stale_unflagged_count"], 0, f"{sid} step {k} {role}")
                self.assertEqual(s["forbidden_text_count"], 0, f"{sid} step {k} {role}")

    def test_baselines_expose_withdrawn_content(self):
        exposed = {"baseline_rag": 0, "overwrite_policy": 0}
        for sid, k, _ in all_steps():
            for arm_id in exposed:
                exposed[arm_id] += self.arm(sid, k, arm_id)["summary"]["stale_unflagged_count"]
        self.assertGreater(exposed["baseline_rag"], 0)
        self.assertGreater(exposed["overwrite_policy"], 0)

    def test_baseline_delete_by_id_really_deletes(self):
        state = E.replay("w1", 2)
        self.assertNotIn("w1_label", state.baselines.arm1)
        self.assertIn("w1_portal", state.baselines.arm1)  # echo survives delete-by-id

    def test_slot_scoped_abstain(self):
        ev = E.evaluate_step("w1", 3, "clinician")
        ctl = next(a for a in ev["arms"] if a["arm_id"] == "lineage_control")
        by = {q["query_id"]: q for q in ctl["queries"]}
        self.assertEqual(by["renal"]["answer"], "ABSTAIN")
        self.assertTrue(by["renal"]["abstained"])
        self.assertEqual(by["dose"]["answer"], "1.5 mg/kg/day")
        self.assertFalse(by["dose"]["abstained"])
        for arm_id in ("baseline_rag", "overwrite_policy"):
            a = next(x for x in ev["arms"] if x["arm_id"] == arm_id)
            self.assertFalse(any(q["abstained"] for q in a["queries"]))

    def test_ruling_resolves_dispute(self):
        ev = E.evaluate_step("w1", 4, "clinician")
        ctl = next(a for a in ev["arms"] if a["arm_id"] == "lineage_control")
        self.assertEqual({q["query_id"]: q["answer"] for q in ctl["queries"]}["renal"],
                         "contraindicated in renal impairment")

    def test_role_filter_hides_citer_from_standard_role_only(self):
        std = E.evaluate_step("w1", 2, "clinician")
        priv = E.evaluate_step("w1", 2, "safety_investigator")
        std_ids = {r["title"] for r in std["records"]["visible"]}
        priv_ids = {r["title"] for r in priv["records"]["visible"]}
        self.assertNotIn("Safety audit memo", std_ids)
        self.assertIn("Safety audit memo", priv_ids)
        self.assertGreater(std["records"]["hidden_from_this_role"], 0)
        self.assertEqual(priv["records"]["hidden_from_this_role"], 0)
        self.assertGreater(
            next(a for a in priv["arms"] if a["arm_id"] == "lineage_control")["summary"]["withdrawn_shown_with_flag_count"], 0)

    def test_dense_vectors_are_deterministic_and_normalized(self):
        a = E.dense_embed(["alpha beta gamma"])
        b = E.dense_embed(["alpha beta gamma"])
        self.assertTrue((a == b).all())
        self.assertAlmostEqual(float((a[0] ** 2).sum()), 1.0, places=9)

    def test_retrieval_is_rrf_k60(self):
        self.assertEqual(E.RRF_K, 60)
        hits = E.hybrid_retrieve([("a", "dose of the compound"), ("b", "parking rules")], "compound dose")
        self.assertEqual(hits[0][0], "a")
        self.assertAlmostEqual(hits[0][1], 2.0 / 61.0, places=9)


class TestGdpr(unittest.TestCase):
    def test_physical_tombstone_and_blocklist(self):
        lin = E.replay("w2", 2).lineage
        erased = {"w2_profile", "w2_summary", "w2_ticket"}
        self.assertEqual({i for i, r in lin.docs.items() if r.tombstone_sha}, erased)
        orig = E.SCENARIOS["w2"].steps[0].docs[0].text
        for i in erased:
            self.assertRegex(lin.docs[i].text, TOMBSTONE_RE)
            self.assertNotIn("jordan", lin.docs[i].text.lower())
            self.assertNotIn("jordan", lin.store.events[i].text.lower())
        self.assertEqual(lin.docs["w2_profile"].tombstone_sha, E.content_fingerprint(orig))
        self.assertEqual(lin.docs["w2_profile"].text, E.tombstone_for(hashlib.sha256(" ".join(orig.lower().split()).encode()).hexdigest()))
        self.assertIn(E.content_fingerprint(orig), lin.blocklist)
        # audit memo carries no personal data and is not tombstoned
        self.assertIsNone(lin.docs["w2_memo"].tombstone_sha)

    def test_reingestion_is_auto_withdrawn_only_by_control_plane(self):
        ev = E.evaluate_step("w2", 3, "privacy_officer")
        lin = E.replay("w2", 3).lineage
        self.assertEqual(lin.eligibility("w2_restore"), "AUDIT_ONLY")
        self.assertRegex(lin.docs["w2_restore"].text, TOMBSTONE_RE)
        self.assertIn("guard_w2_restore", lin.docs)
        self.assertEqual(lin.eligibility("guard_w2_restore"), "ADMIT")
        arms = {a["arm_id"]: a for a in ev["arms"]}
        self.assertEqual(arms["lineage_control"]["summary"]["forbidden_text_count"], 0)
        self.assertGreater(arms["baseline_rag"]["summary"]["forbidden_text_count"], 0)
        self.assertGreater(arms["overwrite_policy"]["summary"]["forbidden_text_count"], 0)
        self.assertEqual(arms["baseline_rag"]["summary"]["correct"], 0)

    def test_no_personal_data_anywhere_in_control_plane_surfaces(self):
        svc = make_service()
        pii = ("jordan.example@example.invalid", "555-0142", "Jordan Example")
        for step in (2, 3):
            for role in E.SCENARIOS["w2"].roles:
                status, body = get(svc, "/api/v2/copilot/step", scenario="w2", step=str(step), role=role)[::2]
                self.assertEqual(status, 200)
                parsed = json.loads(body)
                ctl = next(a for a in parsed["arms"] if a["arm_id"] == "lineage_control")
                blob = json.dumps({"ctl": ctl, "records": parsed["records"]})
                for p in pii:
                    self.assertNotIn(p.lower(), blob.lower(), f"step {step} {role}")
        st, rec = post_json(svc, "/api/v2/copilot/receipt", {"scenario": "w2", "step": 2})
        self.assertEqual(st, 200)
        for p in pii:
            self.assertNotIn(p.lower(), json.dumps(rec).lower())
        self.assertGreaterEqual(len(rec["payload"]["tombstones"]), 3)

    def test_erasure_is_not_reversible(self):
        st, body = post_json(make_service(), "/api/v2/copilot/undo", {"scenario": "w2", "step": 2})
        self.assertEqual(st, 422)
        self.assertEqual(body["error"], "irreversible")


class TestUndo(unittest.TestCase):
    def test_compensating_event_restores_and_preserves_history(self):
        out = E.undo_retraction("w1", 2)
        self.assertTrue(out["history_preserved"])
        changed = {c["id"]: (c["before"], c["after"]) for c in out["changed"]}
        self.assertEqual(changed["w1_label"], ("Withdrawn", "Current"))
        self.assertEqual(changed["w1_portal"], ("Withdrawn", "Current"))
        self.assertEqual(out["cross_check"]["mismatches"], 0)
        self.assertIn("nothing was deleted", out["mechanism"])

    def test_undo_does_not_mutate_cached_state(self):
        before = E.get_replay("w1", 2).lineage.eligibility("w1_label")
        E.undo_retraction("w1", 2)
        self.assertEqual(E.get_replay("w1", 2).lineage.eligibility("w1_label"), before)
        self.assertEqual(before, "AUDIT_ONLY")

    def test_step_without_retraction_rejected(self):
        with self.assertRaises(E.CopilotError) as cm:
            E.undo_retraction("w1", 1)
        self.assertEqual(cm.exception.status, 422)

    def test_w3_undo_works(self):
        out = E.undo_retraction("w3", 2)
        self.assertTrue(out["history_preserved"])
        self.assertTrue(any(c["id"] == "w3_10q" and c["after"] == "Current" for c in out["changed"]))


class TestReceipts(unittest.TestCase):
    def test_sign_verify_roundtrip(self):
        svc = make_service()
        entry = svc.issue_receipt("w1", 2, "non_nda")
        res = svc.verify_receipt(entry)
        self.assertTrue(res["valid"], res)
        self.assertTrue(res["in_ledger"])
        p = entry["payload"]
        self.assertEqual(p["receipt_kind"], "retraction_audit_receipt")
        self.assertEqual(p["retractor_id"], "w1_notice")
        self.assertIn("w1_label", p["withdrawn_lineage"])
        self.assertEqual(p["cross_check"]["mismatches"], 0)
        self.assertEqual(p["decision_engine_sha256"], E.engine_file_sha256())
        self.assertIsInstance(p["issued_at_us"], int)
        self.assertIn("Not legal advice", p["disclaimer"])
        self.assertIn("in memory", p["key_disclosure"])
        sig = base64.b64decode(entry["signature_b64"])
        self.assertEqual(len(sig), 64)

    def test_tamper_detected(self):
        svc = make_service()
        entry = svc.issue_receipt("w1", 2, "non_nda")
        bad = copy.deepcopy(entry)
        bad["payload"]["target_id"] = "w1_portal"
        res = svc.verify_receipt(bad)
        self.assertFalse(res["valid"])
        self.assertFalse(res["checks"]["hash_ok"])
        bad2 = copy.deepcopy(entry)
        bad2["signature_b64"] = base64.b64encode(b"\x00" * 64).decode()
        self.assertFalse(svc.verify_receipt(bad2)["valid"])
        self.assertFalse(svc.verify_receipt({"nope": 1})["valid"])

    def test_foreign_key_rejected(self):
        a, b = make_service(), make_service()
        entry = a.issue_receipt("w1", 2, "non_nda")
        res = b.verify_receipt(entry)
        self.assertFalse(res["valid"])
        self.assertFalse(res["checks"]["key_ok"])

    def test_hash_chain(self):
        svc = make_service()
        e1 = svc.issue_receipt("w1", 2, "non_nda")
        e2 = svc.issue_receipt("w2", 2, "non_nda")
        e3 = svc.issue_receipt("w3", 2, "non_nda")
        self.assertEqual(e1["prev_hash"], E.GENESIS_HASH)
        self.assertEqual(e2["prev_hash"], e1["entry_hash"])
        self.assertEqual(e3["prev_hash"], e2["entry_hash"])
        self.assertEqual([e["payload"]["seq"] for e in (e1, e2, e3)], [1, 2, 3])
        self.assertTrue(svc.verify_ledger()["intact"])
        expected = hashlib.sha256(e2["prev_hash"].encode() + E.canonical_json(e2["payload"])).hexdigest()
        self.assertEqual(expected, e2["entry_hash"])
        svc.ledger[1]["payload"]["scenario_title"] = "edited"
        v = svc.verify_ledger()
        self.assertFalse(v["intact"])
        self.assertEqual(v["first_bad_seq"], 2)

    def test_canonical_json_rejects_floats_and_is_stable(self):
        with self.assertRaises(ValueError):
            E.canonical_json({"a": 1.5})
        self.assertEqual(E.canonical_json({"b": 1, "a": [2, "x"]}), b'{"a":[2,"x"],"b":1}')

    def test_receipt_payload_has_no_wall_clock_latency(self):
        svc = make_service()
        blob = json.dumps(svc.issue_receipt("w3", 2, "non_nda")["payload"])
        for word in ("wall_ms", "latency", "p50", "p95"):
            self.assertNotIn(word, blob)

    def test_deterministic_with_fixed_key_and_clock(self):
        key = Ed25519PrivateKey.generate()
        e1 = make_service(key=key).issue_receipt("w1", 2, "non_nda")
        e2 = make_service(key=key).issue_receipt("w1", 2, "non_nda")
        self.assertEqual(e1["entry_hash"], e2["entry_hash"])
        self.assertEqual(e1["signature_b64"], e2["signature_b64"])

    def test_receipt_only_for_retraction_steps(self):
        with self.assertRaises(E.CopilotError) as cm:
            make_service().issue_receipt("w1", 1, "non_nda")
        self.assertEqual(cm.exception.status, 422)

    def test_verify_endpoint(self):
        svc = make_service()
        st, entry = post_json(svc, "/api/v2/copilot/receipt", {"scenario": "w3", "step": 2})
        self.assertEqual(st, 200)
        st, ok = post_json(svc, "/api/v2/copilot/verify-receipt", {"receipt": entry})
        self.assertEqual(st, 200)
        self.assertTrue(ok["valid"])
        entry["payload"]["step"] = 3
        st, bad = post_json(svc, "/api/v2/copilot/verify-receipt", {"receipt": entry})
        self.assertFalse(bad["valid"])


class TestNonNdaLeakSurface(unittest.TestCase):
    def setUp(self):
        self.svc = make_service()

    def assert_clean(self, text, label):
        self.assertEqual(scan_text_for_ip_leaks(text, label), [], label)

    def test_html_is_clean(self):
        self.assert_clean(E.HTML_PATH.read_text(encoding="utf-8"), "html")

    def test_html_never_inserts_markup_from_data(self):
        html = E.HTML_PATH.read_text(encoding="utf-8")
        self.assertNotIn("innerHTML", html)
        self.assertNotIn("eval(", html)
        self.assertNotIn("document.write", html)

    def test_every_scenario_step_role_response_clean(self):
        count = 0
        for sid, k, sc in all_steps():
            for role in sc.roles:
                status, ctype, body = get(self.svc, "/api/v2/copilot/step", scenario=sid, step=str(k), role=role)
                self.assertEqual(status, 200)
                self.assertTrue(ctype.startswith("application/json"))
                self.assert_clean(body.decode(), f"{sid}/{k}/{role}")
                count += 1
        self.assertEqual(count, 20)

    def test_all_other_endpoints_clean(self):
        for path, params in (("/api/v2/copilot/scenarios", {}), ("/api/v2/copilot/health", {}),
                             ("/api/v2/copilot/pubkey", {}), ("/api/v2/copilot/ledger", {}),
                             ("/api/v2/copilot/scoreboard", {}), ("/api/v2/copilot/benchmark", {"repeats": "5"})):
            status, _, body = get(self.svc, path, **params)
            self.assertEqual(status, 200, path)
            self.assert_clean(body.decode(), path)

    def test_receipts_undo_inject_clean(self):
        for sid, k, sc in all_steps():
            if sc.steps[k - 1].retraction is None:
                continue
            st, _, body = E.handle_copilot_post("/api/v2/copilot/receipt", {}, {},
                                                json.dumps({"scenario": sid, "step": k}).encode(), self.svc)
            self.assertEqual(st, 200)
            self.assert_clean(body.decode(), f"receipt {sid}")
            st, _, body = E.handle_copilot_post("/api/v2/copilot/undo", {}, {},
                                                json.dumps({"scenario": sid, "step": k}).encode(), self.svc)
            self.assertIn(st, (200, 422))
            self.assert_clean(body.decode(), f"undo {sid}")
        st, _, body = E.handle_copilot_post("/api/v2/copilot/inject", {}, {}, json.dumps(
            {"events": [{"id": "a", "text": "x"}, {"id": "b", "text": "y", "tier": 3}],
             "edges": [{"from": "b", "to": "a", "kind": "retracts"}]}).encode(), self.svc)
        self.assertEqual(st, 200)
        self.assert_clean(body.decode(), "inject")

    def test_error_payloads_clean(self):
        cases = [
            get(self.svc, "/api/v2/copilot/step", scenario="nope", step="1"),
            get(self.svc, "/api/v2/copilot/step", scenario="w1", step="99"),
            get(self.svc, "/api/v2/copilot/step", scenario="w1", step="1", role="root"),
            get(self.svc, "/api/v2/copilot/step", scenario="w1", step="1", mode="nda"),
            get(self.svc, "/api/v2/copilot/missing"),
            E.handle_copilot_post("/api/v2/copilot/receipt", {}, {}, b"{bad", self.svc),
            E.handle_copilot_post("/api/v2/copilot/receipt", {}, {}, b"x" * (E.MAX_BODY_BYTES + 1), self.svc),
        ]
        for status, _, body in cases:
            self.assertGreaterEqual(status, 400)
            self.assert_clean(body.decode(), "error")
            self.assertEqual(set(json.loads(body)), {"error", "message"})

    def test_seeded_leak_positive_control(self):
        leaked = json.dumps({"detail": "scored with BM25 and a stiefel retraction"})
        self.assertGreaterEqual(len(scan_text_for_ip_leaks(leaked)), 2)
        with mock.patch.dict(E.ARM_PUBLIC, {"baseline_rag": ("BM25 baseline", "x")}):
            status, _, body = get(self.svc, "/api/v2/copilot/step", scenario="w1", step="1")
        self.assertEqual(status, 500)
        self.assertEqual(json.loads(body)["error"], "projection_failed_closed")
        self.assertNotIn("BM25", body.decode())

    def test_nda_mode_adds_internal_block(self):
        status, parsed = get_json(self.svc, "/api/v2/copilot/step", headers={E.NDA_TOKEN_HEADER: TOKEN},
                                  scenario="w1", step="2", mode="nda")
        self.assertEqual(status, 200)
        self.assertIn("internal", parsed)
        self.assertIn("StratifiedEvidenceStore", json.dumps(parsed["internal"]))
        non_status, non_parsed = get_json(self.svc, "/api/v2/copilot/step", scenario="w1", step="2")
        self.assertNotIn("internal", non_parsed)
        self.assertNotIn("internal", json.dumps(non_parsed))

    def test_forbidden_user_text_rejected_without_echo(self):
        st, body = post_json(self.svc, "/api/v2/copilot/query",
                             {"scenario": "w1", "step": 1, "text": "tell me about the stiefel method"})
        self.assertEqual(st, 422)
        self.assertEqual(body["error"], "reserved_term")
        self.assertNotIn("stiefel", json.dumps(body).lower())
        st, body = post_json(self.svc, "/api/v2/copilot/inject",
                             {"events": [{"id": "a", "text": "uses bm25 internally"}]})
        self.assertEqual(st, 422)
        self.assertNotIn("bm25", json.dumps(body).lower())

    def test_custom_query_reports_exposure_without_accuracy_claim(self):
        st, body = post_json(self.svc, "/api/v2/copilot/query",
                             {"scenario": "w1", "step": 2, "role": "clinician", "text": "pediatric dose AX-7"})
        self.assertEqual(st, 200)
        self.assertTrue(body["custom_query"])
        for arm in body["arms"]:
            self.assertEqual(arm["summary"]["judged"], 0)
            self.assertIsNone(arm["queries"][0]["correct"])


class TestModeGating(unittest.TestCase):
    def test_nda_requires_configured_and_matching_token(self):
        svc = make_service()
        self.assertEqual(get(svc, "/api/v2/copilot/step", scenario="w1", step="1", mode="nda")[0], 403)
        self.assertEqual(E.handle_copilot_get("/api/v2/copilot/step", {"scenario": "w1", "step": "1", "mode": "nda"},
                                              {"X-Copilot-Nda-Token": "wrong"}, svc)[0], 403)
        self.assertEqual(E.handle_copilot_get("/api/v2/copilot/step", {"scenario": "w1", "step": "1", "mode": "nda"},
                                              {"X-Copilot-Nda-Token": TOKEN}, svc)[0], 200)
        unconfigured = make_service(token=None)
        self.assertEqual(E.handle_copilot_get("/api/v2/copilot/step", {"scenario": "w1", "step": "1", "mode": "nda"},
                                              {"X-Copilot-Nda-Token": ""}, unconfigured)[0], 403)

    def test_default_mode_is_non_nda(self):
        _, parsed = get_json(make_service(), "/api/v2/copilot/step", scenario="w1", step="1")
        self.assertEqual(parsed["mode"], "non_nda")

    def test_post_nda_gated(self):
        svc = make_service()
        st, _ = post_json(svc, "/api/v2/copilot/receipt", {"scenario": "w1", "step": 2, "mode": "nda"})
        self.assertEqual(st, 403)
        st, body = post_json(svc, "/api/v2/copilot/receipt", {"scenario": "w1", "step": 2, "mode": "nda"},
                             headers={"X-Copilot-Nda-Token": TOKEN})
        self.assertEqual(st, 200)
        self.assertIn("internal", body["payload"])

    def test_bad_mode_rejected(self):
        self.assertEqual(get(make_service(), "/api/v2/copilot/step", scenario="w1", step="1", mode="debug")[0], 422)


class TestInjection(unittest.TestCase):
    def setUp(self):
        self.svc = make_service()

    def inject(self, payload):
        return post_json(self.svc, "/api/v2/copilot/inject", payload)

    def test_valid_graph_cross_checked(self):
        st, out = self.inject({
            "events": [{"id": "a", "entity": "E", "slot": "s", "claim_value": "1", "text": "one", "tier": 1},
                       {"id": "b", "entity": "E", "slot": "t", "claim_value": "1", "text": "copy", "tier": 1},
                       {"id": "n", "entity": "E", "slot": "s", "claim_value": "2", "text": "two", "tier": 3}],
            "edges": [{"from": "b", "to": "a", "kind": "copy"}, {"from": "n", "to": "a", "kind": "retracts"}]})
        self.assertEqual(st, 200)
        states = {r["id"]: r["state"] for r in out["records"]}
        self.assertEqual(states, {"a": "Withdrawn", "b": "Withdrawn", "n": "Current"})
        self.assertTrue(out["cross_check"]["agrees"])
        self.assertIn("No accuracy claim", out["claim_scope"])

    def test_equal_authority_dispute_abstains(self):
        st, out = self.inject({
            "events": [{"id": "a", "entity": "E", "slot": "s", "claim_value": "1", "text": "x", "tier": 2},
                       {"id": "b", "entity": "E", "slot": "s", "claim_value": "2", "text": "y", "tier": 2}],
            "edges": [{"from": "a", "to": "b", "kind": "retracts"}, {"from": "b", "to": "a", "kind": "retracts"}]})
        self.assertEqual(st, 200)
        self.assertEqual(out["disputed_count"], 2)
        self.assertEqual(out["slots"][0]["status"], "In dispute (abstain)")

    def test_replacement_cycle_422(self):
        st, out = self.inject({
            "events": [{"id": "a", "text": "x", "tier": 2}, {"id": "b", "text": "y", "tier": 2}],
            "edges": [{"from": "a", "to": "b", "kind": "replaces"}, {"from": "b", "to": "a", "kind": "replaces"}]})
        self.assertEqual(st, 422)

    def test_dangling_and_unknown_kind_422(self):
        st, out = self.inject({"events": [{"id": "a", "text": "x"}], "edges": [{"from": "a", "to": "zz", "kind": "copy"}]})
        self.assertEqual((st, out["error"]), (422, "dangling_edge"))
        st, out = self.inject({"events": [{"id": "a", "text": "x"}, {"id": "b", "text": "y"}],
                               "edges": [{"from": "a", "to": "b", "kind": "DERIVES_VALUE"}]})
        self.assertEqual((st, out["error"]), (422, "bad_edge_kind"))

    def test_engine_names_not_accepted_in_public_mode(self):
        st, out = self.inject({"events": [{"id": "a", "text": "x"}, {"id": "b", "text": "y"}],
                               "edges": [{"from": "a", "to": "b", "kind": "COPY_OF"}]})
        self.assertEqual((st, out["error"]), (422, "bad_edge_kind"))

    def test_caps(self):
        ev = [{"id": f"e{i}", "text": "t"} for i in range(E.MAX_INJECT_EVENTS + 1)]
        self.assertEqual(self.inject({"events": ev})[0], 422)
        ev = [{"id": f"e{i}", "text": "t"} for i in range(3)]
        edges = [{"from": "e0", "to": "e1", "kind": "cites"}] * (E.MAX_INJECT_EDGES + 1)
        # the 64 KiB body cap fires first over HTTP; the edge cap is enforced in the engine itself
        self.assertEqual(self.inject({"events": ev, "edges": edges})[0], 413)
        with self.assertRaises(E.CopilotError) as cm:
            E.inject_evidence({"events": ev, "edges": edges}, "non_nda")
        self.assertEqual(cm.exception.status, 422)
        big = json.dumps({"events": [{"id": "a", "text": "x" * 70000}]}).encode()
        self.assertEqual(E.handle_copilot_post("/api/v2/copilot/inject", {}, {}, big, self.svc)[0], 413)

    def test_bad_ids_tiers_and_duplicates(self):
        self.assertEqual(self.inject({"events": [{"id": "bad id!", "text": "x"}]})[0], 422)
        self.assertEqual(self.inject({"events": [{"id": "a"}, {"id": "a"}]})[0], 422)
        self.assertEqual(self.inject({"events": [{"id": "a", "tier": 42}]})[0], 422)
        self.assertEqual(self.inject({"events": []})[0], 422)

    def test_invalid_json_400(self):
        st, _, _ = E.handle_copilot_post("/api/v2/copilot/inject", {}, {}, b"[1,2]", self.svc)
        self.assertEqual(st, 400)


class TestScoreboardAndBenchmark(unittest.TestCase):
    def test_scoreboard_is_live_and_consistent(self):
        sb = E.scoreboard()
        arms = {a["arm_id"]: a for a in sb["arms"]}
        self.assertEqual(arms["lineage_control"]["correct"], arms["lineage_control"]["judged"])
        self.assertEqual(arms["lineage_control"]["stale_unflagged"], 0)
        self.assertEqual(arms["lineage_control"]["forbidden_text"], 0)
        self.assertGreater(arms["baseline_rag"]["stale_unflagged"], 0)
        self.assertLess(arms["baseline_rag"]["correct"], arms["baseline_rag"]["judged"])
        self.assertEqual(sb["cross_check"]["mismatches"], 0)
        self.assertTrue(sb["ground_truth_matches_control_plane_everywhere"])
        self.assertEqual(len(sb["rows"]), 10)

    def test_benchmark_shape_and_scope(self):
        b = E.warm_benchmark(repeats=8)
        self.assertEqual(b["repeats"], 8)
        self.assertLessEqual(b["incremental_update_ms"]["p50"], b["incremental_update_ms"]["p95"])
        self.assertGreaterEqual(b["incremental_update_ms"]["p50"], 0.0)
        self.assertFalse(b["deterministic"])
        self.assertIn("fixture-sized", b["scope"])
        self.assertGreater(b["store_nodes_at_end"], b["fixture_nodes_before"])
        with self.assertRaises(E.CopilotError):
            E.warm_benchmark(repeats=1)

    def test_benchmark_never_enters_receipts(self):
        svc = make_service()
        E.warm_benchmark(repeats=5)
        self.assertNotIn("incremental_update", json.dumps(svc.issue_receipt("w1", 2, "non_nda")))


class TestHttpSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.svc = make_service()
        cls.srv = E.build_server("127.0.0.1", 0, cls.svc)
        cls.port = cls.srv.server_address[1]
        cls.thread = threading.Thread(target=cls.srv.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()
        cls.thread.join(timeout=5)

    def req(self, method, path, body=None, headers=None):
        url = f"http://127.0.0.1:{self.port}{path}"
        data = json.dumps(body).encode() if body is not None else None
        r = urllib.request.Request(url, data=data, method=method, headers=headers or {})
        try:
            with urllib.request.urlopen(r, timeout=20) as resp:
                return resp.status, dict(resp.headers.items()), resp.read()
        except urllib.error.HTTPError as e:
            return e.code, dict(e.headers.items()), e.read()

    def test_page_and_step_over_real_http(self):
        st, h, body = self.req("GET", "/copilot")
        self.assertEqual(st, 200)
        self.assertIn("text/html", h["Content-Type"])
        self.assertEqual(scan_text_for_ip_leaks(body.decode()), [])
        st, h, body = self.req("GET", "/api/v2/copilot/step?scenario=w2&step=2&role=support_agent")
        self.assertEqual(st, 200)
        self.assertEqual(scan_text_for_ip_leaks(body.decode()), [])
        self.assertFalse(any(k.lower().startswith("access-control") for k in h))

    def test_no_cors_and_options_refused(self):
        st, h, _ = self.req("OPTIONS", "/api/v2/copilot/step", headers={"Origin": "http://evil.example"})
        self.assertEqual(st, 405)
        self.assertFalse(any(k.lower().startswith("access-control") for k in h))

    def test_receipt_verify_over_http(self):
        st, _, body = self.req("POST", "/api/v2/copilot/receipt", {"scenario": "w3", "step": 2},
                               {"Content-Type": "application/json"})
        self.assertEqual(st, 200)
        entry = json.loads(body)
        st, _, vbody = self.req("POST", "/api/v2/copilot/verify-receipt", {"receipt": entry},
                                {"Content-Type": "application/json"})
        self.assertTrue(json.loads(vbody)["valid"])

    def test_oversize_and_bad_paths(self):
        st, _, _ = self.req("POST", "/api/v2/copilot/inject", {"events": [{"id": "a", "text": "x" * 70000}]},
                            {"Content-Type": "application/json"})
        self.assertEqual(st, 413)
        self.assertEqual(self.req("GET", "/api/v2/copilot/nothing")[0], 404)

    def test_is_copilot_path(self):
        self.assertTrue(E.is_copilot_path("/copilot"))
        self.assertTrue(E.is_copilot_path("/api/v2/copilot/step?x=1"))
        self.assertFalse(E.is_copilot_path("/api/v1/status"))
        self.assertEqual(urlparse("/copilot").path, "/copilot")


if __name__ == "__main__":
    unittest.main()
