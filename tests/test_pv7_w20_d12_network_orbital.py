"""Tests for Wave 20 / Bundle #61 / Demo D12 `/fabric` (software emulation only).

unittest only (no pytest). CPU only: ``CUDA_VISIBLE_DEVICES`` is forced empty before any import.
One cached evaluation is shared by every test class (the first build takes about a minute).
"""

from __future__ import annotations

import base64
import importlib.util
import json
import os
import struct
import sys
import types
import unittest
from pathlib import Path
from typing import Any, Dict, Tuple
from unittest import mock

os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (REPO_ROOT, REPO_ROOT / "demo_workbench"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import network_orbital_engine as eng  # noqa: E402
from chelated_pv7_spine.claim_registry import claim_registry_sha256  # noqa: E402
from chelated_pv7_spine.fragment_compiler import compile_fragments  # noqa: E402
from scripts.run_pv7_w20_d12_network_orbital_fabric import execute  # noqa: E402
from scripts.verify_non_nda_ip_clean import assert_non_nda_clean, scan_text_for_ip_leaks  # noqa: E402

PINNED_DIGEST = "32693f54e0ce7ad3d401ceb752ae4723b0eee251f6d49b95cdfb26a6201971e9"
NDA_TOKEN = "unit-test-token-not-a-secret"
FRAGMENTS = (
    "docs/research/registry.d/61-t1-fabric-dpu-leo.json",
    "demo_workbench/routes.d/61-fabric.json",
    "scripts/verifiers.d/61_verifier.py",
)

Response = Tuple[int, str, bytes]


def _json(resp: Response) -> Dict[str, Any]:
    return json.loads(resp[2].decode("utf-8"))


def setUpModule() -> None:  # noqa: N802
    eng.set_encoder_mode("local-drycheck")
    eng.get_eval("local-drycheck")


class HonestyFlagTests(unittest.TestCase):
    def test_flags_present_and_false_on_every_json_payload(self) -> None:
        svc = eng.FabricService(nda_token=None)
        gets = ["health", "scenarios", "scoreboard", "schedule", "tco", "pubkey", "ledger"]
        for sub in gets:
            payload = _json(eng.handle_fabric_get(f"/api/v2/fabric/{sub}", {}, {}, svc))
            self.assertIs(payload["hardware_asic_p4_silicon_in_loop_claimed"], False, sub)
            self.assertIs(payload["orbital_flight_hardware_claimed"], False, sub)
            self.assertIn("banner", payload, sub)
        step = _json(eng.handle_fabric_get("/api/v2/fabric/step", {"scenario": "A", "step": "2"}, {}, svc))
        self.assertIs(step["hardware_asic_p4_silicon_in_loop_claimed"], False)
        self.assertIs(step["orbital_flight_hardware_claimed"], False)

    def test_error_payloads_carry_flags(self) -> None:
        svc = eng.FabricService(nda_token=None)
        status, _, body = eng.handle_fabric_get("/api/v2/fabric/nonexistent", {}, {}, svc)
        self.assertEqual(status, 404)
        payload = json.loads(body)
        self.assertIs(payload["hardware_asic_p4_silicon_in_loop_claimed"], False)
        self.assertIs(payload["orbital_flight_hardware_claimed"], False)

    def test_receipts_carry_flags(self) -> None:
        svc = eng.FabricService(nda_token=None)
        entry = svc.issue_receipt("A", 1, "non_nda")
        self.assertIs(entry["payload"]["hardware_asic_p4_silicon_in_loop_claimed"], False)
        self.assertIs(entry["payload"]["orbital_flight_hardware_claimed"], False)

    def test_html_banner_and_flags(self) -> None:
        html = (REPO_ROOT / "demo_workbench" / "network_orbital.html").read_text(encoding="utf-8")
        self.assertIn("hardware_asic_p4_silicon_in_loop_claimed", html)
        self.assertIn("orbital_flight_hardware_claimed", html)
        self.assertNotIn("hardware_asic_p4_silicon_in_loop_claimed: true", html.lower())
        status, ctype, body = eng.handle_fabric_get("/fabric", {}, {}, eng.FabricService())
        self.assertEqual(status, 200)
        self.assertTrue(ctype.startswith("text/html"))
        self.assertEqual(body, (REPO_ROOT / "demo_workbench" / "network_orbital.html").read_bytes())


class ScoreboardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.sb = eng.scoreboard("non_nda")

    def test_seven_arms_with_chelated_fabric_control_last(self) -> None:
        ids = [a["id"] for a in self.sb["arms"]]
        self.assertEqual(len(ids), 7)
        self.assertEqual(ids[-1], "fabric_control")
        self.assertEqual(len(set(ids)), 7)

    def test_oracle_zero_disagreements_and_coverage(self) -> None:
        o = self.sb["oracle"]
        self.assertEqual(o["oracle_disagreements"], 0)
        self.assertEqual(o["register_disagreements"], 0)
        self.assertGreaterEqual(o["traces"], 250)
        self.assertGreater(o["prefixes"], 0)
        self.assertIs(o["shared_decision_code_with_engine"], False)

    def test_scenario_step_counts(self) -> None:
        book = eng.scenario_book()
        self.assertEqual(len(book["A"]), 6)
        self.assertEqual(len(book["B"]), 5)

    def test_reserved_code_fenced_never_forwards(self) -> None:
        o = self.sb["oracle"]
        self.assertGreater(o["reserved_code_injected"], 0)
        self.assertEqual(o["reserved_forward_leaks_fenced"], 0)
        self.assertGreater(o["reserved_forward_leaks_naive"], 0)
        self.assertGreater(o["torn_injected"], 0)

    def test_scale_factor_within_tolerance_on_bridged_chains(self) -> None:
        kk = self.sb["retrieval"]["kappa_within_tol_all_hops"]
        self.assertTrue(kk["fabric_chain"])
        self.assertTrue(kk["orthogonal_chain"])

    def test_sidecar_is_eight_bytes_and_argmax_identical(self) -> None:
        self.assertEqual(eng.SIDECAR_BYTES_PER_FLOW, 8)
        side = self.sb["retrieval"]["sidecar"]
        self.assertTrue(side)
        for hop, row in side.items():
            self.assertEqual(row["bytes_per_flow"], 8, hop)
            self.assertTrue(row["argmax_identical"], hop)
            self.assertIs(row["top5_identical"], True, hop)  # B61-3: full ordered top-5, not just argmax

    def test_degenerate_anchor_set_refused(self) -> None:
        bh = self.sb["retrieval"]["bridge_health"]
        self.assertTrue(bh["degenerate_anchor_refused"])
        self.assertEqual(bh["degenerate_anchor_status"], "FAIL_CLOSED_DEGENERATE")

    def test_audit_records_survive_in_canonical_trace(self) -> None:
        for key in ("scenario_a", "scenario_b"):
            survived = self.sb["audit_survival"][key]
            self.assertIsInstance(survived, list, key)
            self.assertGreaterEqual(len(survived), 1, key)

    def test_echo_clusters_collapse(self) -> None:
        for key in ("scenario_a", "scenario_b"):
            row = self.sb["echo_collapse"][key]
            self.assertLess(row["echo_clusters"], row["echo_records"], key)

    def test_non_nda_scoreboard_has_no_internal_block(self) -> None:
        for arm in self.sb["arms"]:
            self.assertNotIn("internal", arm)
        assert_non_nda_clean(json.dumps(self.sb, sort_keys=True), "scoreboard")

    def test_register_codes_documented(self) -> None:
        self.assertEqual(eng.REG_CODE[eng.VERDICT_FORWARD], 0b01)
        self.assertEqual(eng.REG_CODE[eng.VERDICT_DROP], 0b00)
        self.assertEqual(eng.REG_CODE[eng.VERDICT_DEFER], 0b10)
        self.assertEqual(eng.REG_CODE[eng.VERDICT_INVALID], 0b11)

    def test_disclosures_are_listed(self) -> None:
        self.assertTrue(self.sb["disclosures"])
        self.assertTrue(eng.PUBLIC_DISCLOSURES)


class RegisterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.classes = eng.CLASS_IDS
        self.last_good = {c: eng.VERDICT_DEFER for c in self.classes}

    def _verdicts(self) -> Dict[str, str]:
        cycle = (eng.VERDICT_FORWARD, eng.VERDICT_DROP, eng.VERDICT_DEFER)
        return {c: cycle[i % 3] for i, c in enumerate(self.classes)}

    def test_roundtrip_fenced_decode(self) -> None:
        v = self._verdicts()
        image = eng.compile_register(7, v)
        decoded, stats = eng.decode_register_fenced(image, 7, self.last_good)
        self.assertEqual(decoded, v)
        self.assertEqual(stats["torn_reads"], 0)
        self.assertEqual(stats["reserved_codes"], 0)

    def test_reference_decoder_agrees(self) -> None:
        v = self._verdicts()
        image = eng.compile_register(3, v)
        self.assertEqual(eng.decode_register_reference(image, 3, self.last_good), v)

    def test_reserved_code_fails_closed_but_naive_forwards(self) -> None:
        v = self._verdicts()
        victim = self.classes[0]
        v[victim] = eng.VERDICT_INVALID
        image = eng.compile_register(5, v)
        fenced, stats = eng.decode_register_fenced(image, 5, self.last_good)
        self.assertEqual(fenced[victim], eng.VERDICT_INVALID)
        self.assertNotEqual(fenced[victim], eng.VERDICT_FORWARD)
        self.assertEqual(stats["reserved_codes"], 1)
        self.assertEqual(eng.decode_register_naive(image)[victim], eng.VERDICT_FORWARD)

    def test_torn_read_returns_last_good_snapshot(self) -> None:
        image = bytearray(eng.compile_register(9, self._verdicts()))
        image[-4:] = struct.pack(">I", 8)  # stale trailer
        decoded, stats = eng.decode_register_fenced(bytes(image), 9, self.last_good)
        self.assertEqual(decoded, self.last_good)
        self.assertEqual(stats["torn_reads"], 1)

    def test_wrong_pinned_epoch_is_a_torn_read(self) -> None:
        image = eng.compile_register(9, self._verdicts())
        decoded, stats = eng.decode_register_fenced(image, 10, self.last_good)
        self.assertEqual(decoded, self.last_good)
        self.assertEqual(stats["torn_reads"], 1)


class EndpointTests(unittest.TestCase):
    def setUp(self) -> None:
        self.svc = eng.FabricService(nda_token=NDA_TOKEN)

    def get(self, path: str, params: Dict[str, Any] = None, headers: Dict[str, str] = None) -> Response:
        return eng.handle_fabric_get(path, params or {}, headers or {}, self.svc)

    def post(self, path: str, body: Any, headers: Dict[str, str] = None) -> Response:
        raw = body if isinstance(body, bytes) else json.dumps(body).encode("utf-8")
        return eng.handle_fabric_post(path, {}, headers or {}, raw, self.svc)

    def test_page_paths(self) -> None:
        for p in ("/fabric", "/fabric/", "/fabric/index.html"):
            self.assertEqual(self.get(p)[0], 200, p)

    def test_scenarios_and_steps(self) -> None:
        sc = _json(self.get("/api/v2/fabric/scenarios"))
        counts = {s["id"]: len(s["steps"]) for s in sc["scenarios"]}
        self.assertEqual(counts, {"A": 6, "B": 5})
        for name, n in counts.items():
            for k in range(1, n + 1):
                step = _json(self.get("/api/v2/fabric/step", {"scenario": name, "step": str(k)}))
                self.assertEqual(len(step["arms"]), 7)
                self.assertEqual(step["sid"], f"step_{k}")
                self.assertEqual(len(bytes.fromhex(step["register"]["image_hex"])), len(eng.compile_register(1, {})))

    def test_step_errors(self) -> None:
        self.assertEqual(self.get("/api/v2/fabric/step", {"scenario": "Z", "step": "1"})[0], 404)
        self.assertEqual(self.get("/api/v2/fabric/step", {"scenario": "A", "step": "99"})[0], 422)
        self.assertEqual(self.get("/api/v2/fabric/step", {"scenario": "A", "step": "x"})[0], 422)

    def test_unknown_route_404(self) -> None:
        self.assertEqual(self.get("/api/v2/fabric/nope")[0], 404)
        self.assertEqual(self.post("/api/v2/fabric/nope", {})[0], 404)
        self.assertEqual(self.get("/elsewhere")[0], 404)

    def test_nda_gate(self) -> None:
        self.assertEqual(self.get("/api/v2/fabric/scoreboard", {"mode": "nda"})[0], 403)
        self.assertEqual(self.get("/api/v2/fabric/scoreboard", {"mode": "nda"}, {"X-Copilot-Nda-Token": "wrong"})[0], 403)
        ok = self.get("/api/v2/fabric/scoreboard", {"mode": "nda"}, {"X-Copilot-Nda-Token": NDA_TOKEN})
        self.assertEqual(ok[0], 200)
        self.assertIn("internal", _json(ok)["arms"][-1])
        # No token configured on the server: nda is never available.
        bare = eng.FabricService(nda_token=None)
        self.assertEqual(eng.handle_fabric_get("/api/v2/fabric/scoreboard", {"mode": "nda"},
                                               {"X-Copilot-Nda-Token": ""}, bare)[0], 403)

    def test_bad_mode_422(self) -> None:
        self.assertEqual(self.get("/api/v2/fabric/scoreboard", {"mode": "bogus"})[0], 422)

    def test_post_error_paths(self) -> None:
        self.assertEqual(self.post("/api/v2/fabric/query", b"{not json")[0], 400)
        self.assertEqual(self.post("/api/v2/fabric/query", b"[1,2]")[0], 400)
        self.assertEqual(self.post("/api/v2/fabric/query", b"x" * (eng.MAX_BODY_BYTES + 1))[0], 413)
        self.assertEqual(self.post("/api/v2/fabric/query", {"scenario": "A", "step": 1, "text": ""})[0], 422)
        self.assertEqual(self.post("/api/v2/fabric/query", {"scenario": "A", "step": 1, "text": "x" * 2001})[0], 422)

    def test_query_returns_top5_with_actions(self) -> None:
        resp = self.post("/api/v2/fabric/query", {"scenario": "A", "step": 2, "text": "encrypted tunnel with repeated short bursts"})
        self.assertEqual(resp[0], 200)
        q = _json(resp)["query"]
        self.assertEqual(len(q["top5"]), 5)
        self.assertIn(q["decision"], ("FORWARD", "DROP", "DEFER_INSPECT", "INVALID_FAIL_CLOSED"))

    def test_inject_validation_and_effect_shape(self) -> None:
        sc = _json(self.get("/api/v2/fabric/scenarios"))
        cls = sc["classes"][0]["id"] if isinstance(sc["classes"], list) else sorted(sc["classes"])[0]
        good = self.post("/api/v2/fabric/inject", {"scenario": "A", "step": 3, "class": cls, "claim": "DROP", "tier": 3, "tranche": 1})
        self.assertEqual(good[0], 200)
        body = _json(good)
        self.assertEqual(body["injected"]["claim"], "DROP")
        self.assertEqual(len(body["verdicts_after"]), 7)
        self.assertEqual(self.post("/api/v2/fabric/inject", {"scenario": "A", "step": 3, "class": "nope", "claim": "DROP"})[0], 422)
        self.assertEqual(self.post("/api/v2/fabric/inject", {"scenario": "A", "step": 3, "class": cls, "claim": "BOGUS"})[0], 422)
        self.assertEqual(self.post("/api/v2/fabric/inject", {"scenario": "A", "step": 3, "class": cls, "claim": "DROP", "tier": 9})[0], 422)

    def test_undo_requires_retraction(self) -> None:
        sc = _json(self.get("/api/v2/fabric/scenarios"))
        with_ret = None
        for s in sc["scenarios"]:
            for st in s["steps"]:
                if st["has_retraction"]:
                    with_ret = (s["id"], st["index"])
                    break
            if with_ret:
                break
        self.assertIsNotNone(with_ret)
        ok = self.post("/api/v2/fabric/undo", {"scenario": with_ret[0], "step": with_ret[1]})
        self.assertEqual(ok[0], 200)
        self.assertEqual(len(_json(ok)["arms"]), 7)
        self.assertEqual(self.post("/api/v2/fabric/undo", {"scenario": "A", "step": 1})[0], 422)

    def test_benchmark_bounds(self) -> None:
        self.assertEqual(self.get("/api/v2/fabric/benchmark", {"repeats": "0"})[0], 422)
        self.assertEqual(self.get("/api/v2/fabric/benchmark", {"repeats": "abc"})[0], 422)
        ok = self.get("/api/v2/fabric/benchmark", {"repeats": "3"})
        self.assertEqual(ok[0], 200)
        self.assertEqual(_json(ok)["label"], "MEASURED")

    def test_every_non_nda_endpoint_scans_clean(self) -> None:
        self.assertEqual(scan_text_for_ip_leaks((REPO_ROOT / "demo_workbench" / "network_orbital.html").read_text(encoding="utf-8"), "html"), [])
        pages = [self.get(f"/api/v2/fabric/{s}") for s in ("health", "scenarios", "scoreboard", "schedule", "tco", "pubkey", "ledger")]
        pages.append(self.get("/api/v2/fabric/step", {"scenario": "B", "step": "3"}))
        pages.append(self.post("/api/v2/fabric/receipt", {"scenario": "B", "step": 3}))
        for status, _, body in pages:
            self.assertEqual(status, 200)
            assert_non_nda_clean(body.decode("utf-8"), "endpoint")


class ReceiptTests(unittest.TestCase):
    def setUp(self) -> None:
        self.clock = iter(range(1_700_000_000_000_000, 1_700_000_000_000_000 + 10_000, 7))
        self.svc = eng.FabricService(nda_token=None, clock_us=lambda: next(self.clock))

    def test_signature_chain_and_roundtrip(self) -> None:
        entries = [self.svc.issue_receipt("A", k, "non_nda") for k in range(1, 7)]
        for i, e in enumerate(entries):
            self.assertTrue(self.svc.verify_receipt(e)["valid"], i)
            self.assertEqual(e["payload"]["seq"], i + 1)
            self.assertEqual(len(base64.b64decode(e["signature_b64"])), 64)  # Ed25519
        self.assertEqual(entries[0]["prev_hash"], "0" * 64)
        for a, b in zip(entries, entries[1:]):
            self.assertEqual(b["prev_hash"], a["entry_hash"])
        chain = self.svc.verify_ledger()
        self.assertTrue(chain["intact"])
        self.assertEqual(chain["length"], 6)

    def test_tamper_detection(self) -> None:
        e = self.svc.issue_receipt("B", 2, "non_nda")
        t = json.loads(json.dumps(e))
        t["payload"]["step_title"] += "x"
        self.assertFalse(self.svc.verify_receipt(t)["valid"])
        t = json.loads(json.dumps(e))
        t["payload"]["register_hex"] = "00" * 4
        self.assertFalse(self.svc.verify_receipt(t)["valid"])
        t = json.loads(json.dumps(e))
        t["signature_b64"] = base64.b64encode(b"\x00" * 64).decode("ascii")
        self.assertFalse(self.svc.verify_receipt(t)["valid"])
        self.assertFalse(self.svc.verify_receipt({"payload": 1})["valid"])
        self.assertFalse(self.svc.verify_receipt("garbage")["valid"])

    def test_foreign_key_rejected(self) -> None:
        other = eng.FabricService(nda_token=None)
        e = other.issue_receipt("A", 1, "non_nda")
        res = self.svc.verify_receipt(e)
        self.assertFalse(res["valid"])
        self.assertFalse(res["checks"]["key_ok"])

    def test_integer_only_canonical_json(self) -> None:
        e = self.svc.issue_receipt("A", 4, "non_nda")
        eng.canonical_json(e["payload"])  # must not raise
        with self.assertRaises(ValueError):
            eng.canonical_json({"x": 1.5})
        with self.assertRaises(ValueError):
            eng.canonical_json({"nested": [{"y": 0.1}]})

    def test_receipt_register_matches_decoder(self) -> None:
        e = self.svc.issue_receipt("A", 6, "non_nda")
        image = bytes.fromhex(e["payload"]["register_hex"])
        decoded, stats = eng.decode_register_fenced(image, e["payload"]["epoch"], {c: eng.VERDICT_DEFER for c in eng.CLASS_IDS})
        self.assertEqual(stats["torn_reads"], 0)
        by_pub = {eng.CLASS_PUB_ID[c]: v for c, v in decoded.items()}
        for pub, code in e["payload"]["register_codes"].items():
            self.assertEqual(code, eng.REG_CODE[by_pub[pub]])
        self.assertEqual(e["payload"]["oracle_check"]["disagreements"], 0)

    def test_http_receipt_and_verify(self) -> None:
        r = eng.handle_fabric_post("/api/v2/fabric/receipt", {}, {}, json.dumps({"scenario": "A", "step": 1}).encode(), self.svc)
        self.assertEqual(r[0], 200)
        entry = _json(r)
        v = eng.handle_fabric_post("/api/v2/fabric/verify-receipt", {}, {}, json.dumps({"receipt": entry}).encode(), self.svc)
        self.assertEqual(v[0], 200)
        self.assertTrue(_json(v)["valid"])
        self.assertTrue(_json(v)["in_ledger"])

    def test_nda_receipt_has_internal_block_only_in_nda(self) -> None:
        self.assertNotIn("internal", self.svc.issue_receipt("A", 1, "non_nda")["payload"])
        self.assertIn("internal", self.svc.issue_receipt("A", 1, "nda")["payload"])


class Batch4RemediationTests(unittest.TestCase):
    """B61-1 (per-mode ledger), B61-2 (audit retention), B61-3 (fail-closed live encoders), B61-4 (forbidden input)."""

    def setUp(self) -> None:
        self.svc = eng.FabricService(nda_token=NDA_TOKEN)

    def _get(self, path: str, params: Dict[str, Any] = None, headers: Dict[str, str] = None) -> Response:
        return eng.handle_fabric_get(path, params or {}, headers or {}, self.svc)

    def _post(self, path: str, body: Any, headers: Dict[str, str] = None) -> Response:
        return eng.handle_fabric_post(path, {}, headers or {}, json.dumps(body).encode("utf-8"), self.svc)

    # ---- B61-1: NDA receipts can never be listed through a non_nda ledger read ---------------------
    def test_nda_receipt_never_appears_in_non_nda_ledger_read(self) -> None:
        nda = self.svc.issue_receipt("A", 1, "nda")
        self.assertIn("internal", nda["payload"])
        self.svc.issue_receipt("A", 2, "non_nda")
        resp = self._get("/api/v2/fabric/ledger", {"mode": "non_nda"})
        self.assertEqual(resp[0], 200)
        text = resp[2].decode("utf-8")
        body = json.loads(text)
        self.assertEqual(len(body["entries"]), 1)
        self.assertEqual({e["payload"]["mode"] for e in body["entries"]}, {"non_nda"})
        self.assertNotIn("internal", text)
        self.assertNotIn("eligibility_raw", text)
        self.assertTrue(body["chain"]["intact"])
        self.assertEqual(body["chain"]["length"], 1)

    def test_default_ledger_read_is_the_non_nda_partition(self) -> None:
        self.svc.issue_receipt("A", 1, "nda")
        body = _json(self._get("/api/v2/fabric/ledger"))
        self.assertEqual(body["entries"], [])
        self.assertEqual(body["mode"], "non_nda")

    def test_nda_ledger_is_token_gated_and_carries_only_nda_entries(self) -> None:
        self.svc.issue_receipt("A", 1, "nda")
        self.svc.issue_receipt("A", 2, "non_nda")
        self.assertEqual(self._get("/api/v2/fabric/ledger", {"mode": "nda"})[0], 403)
        ok = self._get("/api/v2/fabric/ledger", {"mode": "nda"}, {"X-Copilot-Nda-Token": NDA_TOKEN})
        self.assertEqual(ok[0], 200)
        body = _json(ok)
        self.assertEqual({e["payload"]["mode"] for e in body["entries"]}, {"nda"})
        self.assertEqual(len(body["entries"]), 1)

    def test_each_mode_chain_is_independently_hash_linked_from_genesis(self) -> None:
        a = self.svc.issue_receipt("A", 1, "nda")
        b = self.svc.issue_receipt("A", 1, "non_nda")
        self.assertEqual(a["prev_hash"], "0" * 64)
        self.assertEqual(b["prev_hash"], "0" * 64)  # not chained behind the NDA receipt
        self.assertEqual((a["payload"]["seq"], b["payload"]["seq"]), (1, 1))
        self.assertTrue(self.svc.verify_ledger("nda")["intact"])
        self.assertTrue(self.svc.verify_ledger("non_nda")["intact"])
        self.assertTrue(self.svc.verify_receipt(a)["in_ledger"])
        self.assertTrue(self.svc.verify_receipt(b)["in_ledger"])
        with self.assertRaises(eng.FabricError):
            self.svc.ledger_for("bogus")

    # ---- B61-2: audit retention counts only ADMIT / AUDIT_ONLY memos ------------------------------
    def test_retained_eligibility_set(self) -> None:
        self.assertEqual(tuple(eng.FabricArm._RETAINED_ELIGIBILITY), ("ADMIT", "AUDIT_ONLY"))

    def _fake_arm(self, eligibility: Dict[str, str]):
        events = {e: types.SimpleNamespace(kind=eng.KIND_AUDIT) for e in eligibility}
        events["d_not_audit"] = types.SimpleNamespace(kind=eng.KIND_DERIVED)
        store = types.SimpleNamespace(events=dict(events), eligibility=dict(eligibility, d_not_audit="ADMIT"))
        return types.SimpleNamespace(events=events, store=store, _RETAINED_ELIGIBILITY=eng.FabricArm._RETAINED_ELIGIBILITY)

    def test_audit_retained_excludes_contested_memos(self) -> None:
        arm = self._fake_arm({"a_ok": "ADMIT", "a_cited_gone": "AUDIT_ONLY", "a_contested": "CONTESTED"})
        self.assertEqual(eng.FabricArm.audit_retained(arm), {"a_ok", "a_cited_gone"})

    def test_audit_in_trace_requires_trace_and_store_to_agree(self) -> None:
        arm = self._fake_arm({"a_ok": "ADMIT", "a_only": "AUDIT_ONLY", "a_contested": "CONTESTED", "a_disagree": "ADMIT"})
        nodes = {
            "a_ok": types.SimpleNamespace(eligibility="ADMIT"),
            "a_only": types.SimpleNamespace(eligibility="AUDIT_ONLY"),
            "a_contested": types.SimpleNamespace(eligibility="CONTESTED"),
            "a_disagree": types.SimpleNamespace(eligibility="AUDIT_ONLY"),  # trace disagrees with the store
        }
        trace = types.SimpleNamespace(by_id=lambda: nodes)
        with mock.patch.object(eng, "export_canonical_ek5_audit_trace", lambda store: trace):
            self.assertEqual(eng.FabricArm.audit_in_trace(arm), {"a_ok", "a_only"})

    # ---- B61-3: a requested-live encoder that cannot load must fail closed -------------------------
    def test_requested_live_encoders_that_fall_back_are_recorded(self) -> None:
        with mock.patch.object(eng, "live_encoder_available", lambda i: (False, "no checkpoint")):
            es = eng.EncoderSet("run-live-encoders")
        self.assertEqual(es.dry_fallbacks, [eid for eid, _n, _d in eng.ENCODER_CHAIN])
        self.assertTrue(all(v.startswith("stand_in") for v in es.status.values()))

    def test_local_drycheck_requests_no_live_encoder_so_records_no_fallback(self) -> None:
        self.assertEqual(eng.EncoderSet("local-drycheck").dry_fallbacks, [])

    def test_runner_fails_closed_on_dry_fallback_unless_explicitly_allowed(self) -> None:
        import copy

        import scripts.run_pv7_w20_d12_network_orbital_fabric as runner

        base = eng.get_eval("local-drycheck")
        fr = copy.copy(base["fr"])
        fr.enc = copy.copy(base["fr"].enc)
        fr.enc.dry_fallbacks = ["enc_e"]  # what a live request on a CPU-only host records
        fake_ev = dict(base, fr=fr)
        self.addCleanup(eng.set_encoder_mode, "local-drycheck")
        clean_probe = {"patterns": 1, "responses_scanned": 0, "hits": 0, "page_hits": 0, "http_errors": [],
                       "forbidden_input_rejected_without_echo": True, "nda_ledger_isolated": True}
        # the live-HTTP probe is exercised by test_runner_gates_all_pass; stub it so this stays a pure gate-logic test
        with mock.patch.object(eng, "get_eval", lambda mode="local-drycheck": fake_ev),                 mock.patch.object(runner, "_http_probe", lambda: clean_probe):
            strict = runner.execute("run-live-encoders")
            allowed = runner.execute("run-live-encoders", allow_dry_standin=True)
        gate = {g["name"]: g for g in strict["gates"]}["live_encoders_fail_closed_on_dry_fallback"]
        self.assertFalse(gate["passed"])
        self.assertFalse(strict["passed"])
        self.assertIn("FAIL CLOSED", gate["detail"])
        self.assertEqual(strict["dry_standin_fallbacks"], ["enc_e"])
        gate2 = {g["name"]: g for g in allowed["gates"]}["live_encoders_fail_closed_on_dry_fallback"]
        self.assertTrue(gate2["passed"])
        self.assertIn("--allow-dry-standin", gate2["detail"])

    # ---- B61-4: forbidden input is refused outright and never echoed ------------------------------
    def test_forbidden_input_is_422_and_not_echoed(self) -> None:
        probe = "stiefel"
        resp = self._post("/api/v2/fabric/query", {"scenario": "A", "step": 1, "mode": "non_nda", "text": probe})
        self.assertEqual(resp[0], 422)
        self.assertNotIn(probe, resp[2].decode("utf-8").lower())
        self.assertEqual(_json(resp)["error"], "forbidden_input")

    def test_forbidden_input_is_caught_in_nested_values_and_keys(self) -> None:
        for body in ({"scenario": "A", "step": 1, "extra": {"deep": ["x", "stiefel"]}},
                     {"scenario": "A", "step": 1, "stiefel": "k"}):
            resp = self._post("/api/v2/fabric/query", body)
            self.assertEqual(resp[0], 422, body)
            self.assertNotIn("stiefel", resp[2].decode("utf-8").lower())

    def test_clean_input_still_served_and_verify_receipt_is_exempt(self) -> None:
        self.assertEqual(self._post("/api/v2/fabric/query", {"scenario": "A", "step": 1, "text": "ordinary question"})[0], 200)
        entry = self.svc.issue_receipt("A", 1, "non_nda")
        tampered = json.loads(json.dumps(entry))
        tampered["payload"]["step_title"] = "stiefel"
        resp = self._post("/api/v2/fabric/verify-receipt", {"receipt": tampered})
        self.assertEqual(resp[0], 200)  # a verifier must be able to reject a hostile receipt, not 422 on it
        self.assertFalse(_json(resp)["valid"])

    def test_nda_mode_is_not_subject_to_the_non_nda_input_policy(self) -> None:
        resp = self._post("/api/v2/fabric/query", {"scenario": "A", "step": 1, "mode": "nda", "text": "stiefel"},
                          {"X-Copilot-Nda-Token": NDA_TOKEN})
        self.assertEqual(resp[0], 200)


class ScheduleAndTcoTests(unittest.TestCase):
    def test_contact_schedule_contents(self) -> None:
        s = eng.build_contact_schedule()
        self.assertTrue(s["tle_checksums_verified"])
        self.assertTrue(s["ground_windows"])
        self.assertTrue(s["oisl_windows"])
        self.assertTrue(s["oceanic_blackout_windows"])
        self.assertEqual(len(s["schedule_sha256"]), 64)
        self.assertEqual(eng.build_contact_schedule()["schedule_sha256"], s["schedule_sha256"])  # deterministic

    def test_schedule_is_synthetic_and_two_body(self) -> None:
        rep = eng.schedule_report("non_nda")
        self.assertTrue(rep["synthetic"])
        self.assertIn("Keplerian", rep["propagator"])
        self.assertGreaterEqual(len(rep["tle_lines"]), 6)
        self.assertEqual(len(rep["tle_lines"]) % 2, 0)

    def test_tco_labels(self) -> None:
        t = eng.tco_calc({})
        self.assertEqual(t["outputs"]["sidecar_bytes_per_flow"]["label"], "MEASURED")
        self.assertEqual(t["outputs"]["oisl_contact_seconds_min_pair"]["label"], "MEASURED")
        self.assertEqual(t["outputs"]["sidecar_bytes_per_flow"]["value"], 8)
        for name in ("oisl_transfer_ms_sidecar", "downlink_cost_cents_sidecar", "reduction_x1000"):
            self.assertEqual(t["outputs"][name]["label"], "SENSITIVITY_ASSUMPTION", name)
        for name, row in t["inputs"].items():
            self.assertEqual(row["label"], "SENSITIVITY_ASSUMPTION", name)

    def test_tco_integer_only_and_validated(self) -> None:
        t = eng.tco_calc({"n_flows": "2000"})
        eng.canonical_json(t)
        self.assertEqual(t["outputs"]["sidecar_total_bytes"]["value"], 2000 * 8)
        with self.assertRaises(eng.FabricError):
            eng.tco_calc({"n_flows": "0"})
        with self.assertRaises(eng.FabricError):
            eng.tco_calc({"oisl_rate_mbps": "fast"})


class RunnerAndFragmentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = execute("local-drycheck")

    def test_runner_gates_all_pass(self) -> None:
        failed = [g for g in self.report["gates"] if not g["passed"]]
        self.assertEqual(failed, [], failed)
        self.assertTrue(self.report["passed"])
        self.assertGreaterEqual(len(self.report["gates"]), 13)

    def test_runner_report_flags_and_mode(self) -> None:
        self.assertEqual(self.report["mode"], "local-drycheck")
        self.assertIs(self.report["hardware_asic_p4_silicon_in_loop_claimed"], False)
        self.assertIs(self.report["orbital_flight_hardware_claimed"], False)
        self.assertEqual(self.report["spine_digest"], PINNED_DIGEST)
        self.assertEqual(self.report["live_encoders_used"], [])

    def test_runner_is_deterministic(self) -> None:
        again = execute("local-drycheck")
        self.assertEqual(again["scoreboard_sha256"], self.report["scoreboard_sha256"])

    def test_runner_rejects_unknown_mode(self) -> None:
        with self.assertRaises(ValueError):
            execute("bogus")

    def test_spine_digest_pinned(self) -> None:
        self.assertEqual(claim_registry_sha256(), PINNED_DIGEST)

    def test_fragments_compile(self) -> None:
        sources = {rel: (REPO_ROOT / rel).read_bytes() for rel in FRAGMENTS}
        compiled = compile_fragments(REPO_ROOT, PINNED_DIGEST, sources)
        self.assertEqual(compiled["dependency_order"], [61])
        self.assertEqual(compiled["registry"][0]["design"]["hardware_asic_p4_silicon_in_loop_claimed"], False)
        self.assertEqual(compiled["registry"][0]["design"]["orbital_flight_hardware_claimed"], False)

    def test_route_fragment_matches_dispatch(self) -> None:
        frag = json.loads((REPO_ROOT / FRAGMENTS[1]).read_text(encoding="utf-8"))
        svc = eng.FabricService()
        for r in frag["routes"]:
            self.assertEqual(r["bundle_id"], 61)
            self.assertTrue(eng.is_fabric_path(r["path"]), r["path"])
        posts = [r["path"] for r in frag["routes"] if r["method"] == "POST"]
        self.assertEqual(sorted(posts), sorted(f"/api/v2/fabric/{s}" for s in ("query", "inject", "undo", "receipt", "verify-receipt")))
        for p in posts:  # every declared POST route is dispatchable (400 invalid body is fine; 404 is not)
            self.assertNotEqual(eng.handle_fabric_post(p, {}, {}, b"[]", svc)[0], 404, p)

    def test_verifier_fragment(self) -> None:
        spec = importlib.util.spec_from_file_location("v61_verifier_under_test", REPO_ROOT / FRAGMENTS[2])
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        self.assertEqual(mod.BUNDLE_ID, 61)
        self.assertEqual(mod.SPINE_DIGEST, PINNED_DIGEST)
        result = mod.verify_bundle(REPO_ROOT)
        failed = [c for c in result["checks"] if not c["passed"]]
        self.assertEqual(failed, [], failed)
        self.assertTrue(result["passed"])


class SourceHygieneTests(unittest.TestCase):
    OWNED = (
        "demo_workbench/network_orbital_engine.py",
        "scripts/run_pv7_w20_d12_network_orbital_fabric.py",
        "scripts/verifiers.d/61_verifier.py",
        "tests/test_pv7_w20_d12_network_orbital.py",
    )

    def test_python39_compatible_syntax(self) -> None:
        import ast

        for rel in self.OWNED:
            src = (REPO_ROOT / rel).read_text(encoding="utf-8")
            ast.parse(src, filename=rel, feature_version=(3, 9))

    def test_standalone_server_is_same_origin_only(self) -> None:
        src = (REPO_ROOT / self.OWNED[0]).read_text(encoding="utf-8")
        server_part = src.split("# Standalone server", 1)[1]
        self.assertNotIn("Access-Control-Allow-Origin", server_part)
        status, _, _ = eng._error_response(eng.FabricError(405, "same_origin_only", "x"))
        self.assertEqual(status, 405)

    def test_no_pytest_or_gpu_use(self) -> None:
        for rel in self.OWNED[:3]:  # this test file names the needles itself, so it is excluded
            src = (REPO_ROOT / rel).read_text(encoding="utf-8")
            self.assertNotIn("import " + "pytest", src, rel)
            self.assertNotIn(".cuda(", src, rel)
            self.assertNotIn('device="cuda"', src, rel)
        self.assertEqual(os.environ.get("CUDA_VISIBLE_DEVICES"), "")


if __name__ == "__main__":
    unittest.main()
