"""Tests for the PV7 declarative fragment loaders (Round-2 remediation).

Covers (a) ``demo_workbench/server.py`` auto-discovery of ``demo_workbench/routes.d/*.json`` and
(b) ``scripts/verify_all_offline.py`` auto-discovery of ``docs/research/registry.d/*.json`` plus
``scripts/verifiers.d/*_verifier.py``. All tests are CPU-only and offline.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from demo_workbench import server as S  # noqa: E402

_spec = importlib.util.spec_from_file_location("verify_all_offline_w21", REPO_ROOT / "scripts" / "verify_all_offline.py")
assert _spec is not None and _spec.loader is not None
V = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = V
_spec.loader.exec_module(V)

FABRIC_HEALTH = "/api/v2/fabric/health"


class TestServerRouteLoader(unittest.TestCase):
    def setUp(self) -> None:
        S.load_declared_routes(refresh=True)

    def test_declared_routes_compile_under_pinned_digest(self) -> None:
        table = S.load_declared_routes(refresh=True)
        self.assertIsNone(table["error"], table["error"])
        self.assertEqual(table["spine_digest"], S.PV7_SPINE_DIGEST)
        self.assertIn(61, table["bundles"])
        paths = {(r["method"], r["path"]) for r in table["routes"]}
        self.assertIn(("GET", "/fabric"), paths)
        self.assertIn(("GET", FABRIC_HEALTH), paths)
        self.assertEqual(len(table["compiled_sha256"]), 64)

    def test_routes_endpoint_is_internal_only(self) -> None:
        status, ctype, body = S.dispatch_api_request("GET", "/api/v1/pv7/routes")
        self.assertEqual(status, 200)
        self.assertEqual(ctype, "application/json")
        payload = json.loads(body)
        self.assertEqual(payload["spine_digest"], S.PV7_SPINE_DIGEST)
        self.assertTrue(payload["routes"])
        status, _, _ = S.dispatch_api_request("GET", "/api/v1/pv7/routes", public_only=True)
        self.assertEqual(status, 404, "the internal route table must not be served under --public")

    def test_fabric_routes_forwarded_in_both_modes(self) -> None:
        for public_only in (False, True):
            status, ctype, body = S.dispatch_api_request("GET", FABRIC_HEALTH, public_only=public_only)
            self.assertEqual(status, 200, f"public_only={public_only}")
            self.assertEqual(json.loads(body)["service"], "network-orbital-fabric")
        status, ctype, body = S.dispatch_api_request("GET", "/fabric", public_only=True)
        self.assertEqual(status, 200)
        self.assertIn("text/html", ctype)

    def test_unknown_path_still_404_under_public(self) -> None:
        status, _, _ = S.dispatch_api_request("GET", "/api/v2/fabric/not-declared-anywhere", public_only=True)
        self.assertEqual(status, 404)

    def test_public_only_strips_nda_header_before_forwarding(self) -> None:
        captured = {}

        def fake_get(path, query_params, headers, *a, **k):
            captured["headers"] = dict(headers)
            return 200, "application/json", b"{}"

        headers = {"X-Copilot-NDA-Token": "secret", "Accept": "application/json"}
        with mock.patch("demo_workbench.network_orbital_engine.handle_fabric_get", fake_get):
            S.dispatch_api_request("GET", FABRIC_HEALTH, headers=headers, public_only=True)
            lowered = {k.lower() for k in captured["headers"]}
            self.assertNotIn("x-copilot-nda-token", lowered)
            self.assertIn("accept", lowered)
            S.dispatch_api_request("GET", FABRIC_HEALTH, headers=headers, public_only=False)
            lowered = {k.lower() for k in captured["headers"]}
            self.assertIn("x-copilot-nda-token", lowered, "internal mode must keep the NDA header for the handler")

    def test_route_without_handler_fails_closed_501(self) -> None:
        route = {"method": "GET", "path": "/orphan", "bundle_id": 999}
        status, _, body = S._forward_declared_route(route, "GET", "/orphan", {}, {}, b"", False)
        self.assertEqual(status, 501)
        self.assertEqual(json.loads(body)["error"], "route_declared_without_handler")

    def test_compile_failure_yields_no_routes_and_error(self) -> None:
        from chelated_pv7_spine.fragment_compiler import FragmentCompileError

        with mock.patch("chelated_pv7_spine.fragment_compiler.compile_repo_fragments",
                        side_effect=FragmentCompileError("boom")):
            table = S.load_declared_routes(refresh=True)
        try:
            self.assertEqual(table["routes"], [])
            self.assertIn("boom", table["error"])
            self.assertIsNone(S._declared_route("GET", FABRIC_HEALTH))
        finally:
            S.load_declared_routes(refresh=True)


def _write(path: Path, data) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = data if isinstance(data, str) else json.dumps(data, indent=1, sort_keys=True) + "\n"
    path.write_text(text, encoding="utf-8", newline="\n")


class TestOfflineVerifierLoader(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)

    def _fragment(self, bundle: int, **extra) -> dict:
        body = {
            "id": bundle,
            "evidence_tier": "local-drycheck-pass",
            "evidence_disclosure": {"synthetic": "yes"},
            "status": "protocol-ready",
        }
        body.update(extra)
        return body

    def test_union_of_spec_and_fragment_bundles_with_mixed_id_types(self) -> None:
        spec = {"bundles": [{"id": "d8-alpha"}, {"id": "w6"}]}
        _write(self.root / "docs/research/registry.d/47-a.json", self._fragment(47))
        _write(self.root / "docs/research/registry.d/48-b.json", self._fragment(48))
        loaded = V.load_all_bundles(self.root, spec)
        self.assertEqual(loaded["errors"], [])
        self.assertEqual(set(loaded["bundles"]), {"d8-alpha", "w6", "47", "48"})
        self.assertEqual(loaded["fragment_ids"], [47, 48])
        self.assertEqual(loaded["bundles"]["47"]["source"], "fragment")
        self.assertEqual(loaded["bundles"]["w6"]["source"], "spec")

    def test_collision_and_filename_mismatch_are_reported(self) -> None:
        spec = {"bundles": [{"id": "47"}]}
        _write(self.root / "docs/research/registry.d/47-a.json", self._fragment(47))
        _write(self.root / "docs/research/registry.d/49-b.json", self._fragment(50))
        loaded = V.load_all_bundles(self.root, spec)
        joined = "\n".join(loaded["errors"])
        self.assertIn("collides", joined)
        self.assertIn("does not match the filename prefix", joined)

    def test_duplicate_spec_ids_and_unusable_ids_are_reported(self) -> None:
        spec = {"bundles": [{"id": "x"}, {"id": "x"}, {"id": True}, {}]}
        loaded = V.load_all_bundles(self.root, spec)
        joined = "\n".join(loaded["errors"])
        self.assertIn("duplicate bundle id", joined)
        self.assertIn("without a usable id", joined)

    def test_static_checks_flag_missing_tier_disclosure_and_files(self) -> None:
        _write(self.root / "docs/research/registry.d/47-a.json",
               self._fragment(47, runner_file="scripts/missing_runner.py"))
        bad = self._fragment(48)
        del bad["evidence_tier"]
        del bad["evidence_disclosure"]
        _write(self.root / "docs/research/registry.d/48-b.json", bad)
        report = V.Report()
        V.verify_fragment_bundles(self.root, {"bundles": []}, report, run_verifiers=False)
        by_name = {c["name"]: c for c in report.checks}
        self.assertEqual(by_name["fragments.47.static"]["status"], "FAIL")
        self.assertIn("runner_file", by_name["fragments.47.static"]["detail"])
        self.assertEqual(by_name["fragments.48.static"]["status"], "FAIL")
        self.assertIn("evidence_tier", by_name["fragments.48.static"]["detail"])
        self.assertIn("evidence_disclosure", by_name["fragments.48.static"]["detail"])
        self.assertIn("fragments.verifiers", by_name)
        self.assertEqual(by_name["fragments.verifiers"]["status"], "WARN")

    def test_verifier_subprocess_pass_fail_and_timeout(self) -> None:
        ok = self.root / "scripts/verifiers.d/47_verifier.py"
        _write(ok, "print('ok')\n")
        bad = self.root / "scripts/verifiers.d/48_verifier.py"
        _write(bad, "import sys\nprint('nope')\nsys.exit(3)\n")
        slow = self.root / "scripts/verifiers.d/49_verifier.py"
        _write(slow, "import time\ntime.sleep(30)\n")
        self.assertEqual(V._run_verifier(self.root, "scripts/verifiers.d/47_verifier.py", 30)["rc"], 0)
        res = V._run_verifier(self.root, "scripts/verifiers.d/48_verifier.py", 30)
        self.assertEqual(res["rc"], 3)
        self.assertIn("nope", res["tail"])
        self.assertEqual(V._run_verifier(self.root, "scripts/verifiers.d/49_verifier.py", 1)["rc"], 124)

    def test_verifier_runs_with_cuda_hidden(self) -> None:
        probe = self.root / "scripts/verifiers.d/47_verifier.py"
        _write(probe, "import os, sys\nsys.exit(0 if os.environ.get('CUDA_VISIBLE_DEVICES') == '' else 9)\n")
        with mock.patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            self.assertEqual(V._run_verifier(self.root, "scripts/verifiers.d/47_verifier.py", 30)["rc"], 0)

    def test_real_repo_fragments_compile_and_pass_static_checks(self) -> None:
        spec = V.load_json(REPO_ROOT / V.SPEC_REL)
        report = V.Report()
        summary = V.verify_fragment_bundles(REPO_ROOT, spec, report, run_verifiers=False)
        failures = [c for c in report.checks if c["status"] == "FAIL"]
        self.assertEqual(failures, [])
        self.assertEqual(summary["fragment_bundles"], 18)
        names = {c["name"] for c in report.checks}
        self.assertIn("fragments.compile", names)
        for bundle in range(47, 65):
            self.assertIn(f"fragments.{bundle}.static", names)


if __name__ == "__main__":
    unittest.main()
