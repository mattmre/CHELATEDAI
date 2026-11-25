"""Tests for Wave 19 bundles #57 (2B), #59 (2D), #60 (2E). unittest only; CPU only."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT / "packages" / "chelated_kernel"), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np  # noqa: E402

import chelated_cartridge_array as cart  # noqa: E402
import chelated_cpu_memory_appliance as iq04  # noqa: E402
import chelated_ssd_kv_repo_cache as kvc  # noqa: E402
from chelated_pv7_spine.claim_registry import claim_registry_sha256  # noqa: E402
from chelated_pv7_spine.fragment_compiler import compile_repo_fragments  # noqa: E402
from chelated_pv7_spine.tra1_weight_sha_guard import StaleWeightRevisionError  # noqa: E402

SPINE_DIGEST = "32693f54e0ce7ad3d401ceb752ae4723b0eee251f6d49b95cdfb26a6201971e9"


def _load(rel: str):
    path = REPO_ROOT / rel
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[path.stem] = module
    spec.loader.exec_module(module)
    return module


class FragmentTests(unittest.TestCase):
    def test_digest_literal_matches_live_registry(self) -> None:
        self.assertEqual(claim_registry_sha256(), SPINE_DIGEST)

    def test_fragments_compile_protocol_ready(self) -> None:
        compiled = compile_repo_fragments(REPO_ROOT, SPINE_DIGEST)
        by_id = {row["id"]: row for row in compiled["registry"]}
        for bid in (57, 59, 60):
            self.assertEqual(by_id[bid]["status"], "protocol-ready")
            self.assertEqual(by_id[bid]["spine_digest"], SPINE_DIGEST)
        self.assertEqual([b for b in compiled["dependency_order"] if b in (57, 59, 60)], [57, 59, 60])

    def test_verifier_metadata_literals(self) -> None:
        for bid in (57, 59, 60):
            text = (REPO_ROOT / "scripts" / "verifiers.d" / f"{bid}_verifier.py").read_text(encoding="utf-8")
            self.assertIn(f"BUNDLE_ID = {bid}\n", text)
            self.assertIn(f'SPINE_DIGEST = "{SPINE_DIGEST}"', text)
            self.assertIn("DEPENDS_ON = ()", text)


class CartridgeArrayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.tmp = tempfile.mkdtemp(prefix="pv7_t57_")
        cls.manifest = cart.run_cartridge_array_experiment(
            mode="local-drycheck",
            workdir=Path(cls.tmp),
            payload_bytes=32 * 1024,
            revocation_trials=40,
            threads=8,
        )

    @classmethod
    def tearDownClass(cls) -> None:
        shutil.rmtree(cls.tmp, ignore_errors=True)  # the single tmp dir created in setUpClass

    def test_all_gates_pass(self) -> None:
        failed = sorted(k for k, v in self.manifest["gates"].items() if not v)
        self.assertEqual(failed, [])
        self.assertTrue(self.manifest["overall_pass"])

    def test_payload_count_separate_from_catalog_count(self) -> None:
        self.assertEqual(self.manifest["materialized_payload_count"], 250)
        self.assertEqual(self.manifest["catalog_entry_count"], 1000)
        self.assertNotEqual(self.manifest["materialized_payload_count"], self.manifest["catalog_entry_count"])

    def test_zero_post_enforcement_and_cp_bound(self) -> None:
        rev = self.manifest["revocation"]
        self.assertEqual(rev["post_enforcement_executions"], 0)
        self.assertGreater(rev["unguarded_control_post_enforcement_executions"], 0)
        expected = cart.clopper_pearson_upper(0, rev["requests_total"])
        self.assertAlmostEqual(rev["cp_one_sided_95_upper_bound"], expected, places=12)

    def test_clopper_pearson_closed_form(self) -> None:
        self.assertAlmostEqual(cart.clopper_pearson_upper(0, 100), 1 - 0.05 ** (1 / 100), places=12)
        self.assertEqual(cart.clopper_pearson_upper(5, 5), 1.0)
        self.assertGreater(cart.clopper_pearson_upper(1, 100), cart.clopper_pearson_upper(0, 100))

    def test_cage_is_six_gib_no_swap(self) -> None:
        self.assertEqual(self.manifest["cage"]["memory_max_bytes"], 6 * 1024**3)
        self.assertEqual(self.manifest["cage"]["memory_swap_max_bytes"], 0)

    def test_run_live_fails_closed(self) -> None:
        with self.assertRaises(cart.LiveModeUnavailableError):
            cart.run_cartridge_array_experiment(mode="run-live")

    def test_unknown_mode_rejected(self) -> None:
        with self.assertRaises(ValueError):
            cart.run_cartridge_array_experiment(mode="bogus")


class IQ04ApplianceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = iq04.run_iq04_experiment(mode="local-drycheck", repo_root=REPO_ROOT)

    def test_all_gates_pass(self) -> None:
        failed = sorted(k for k, v in self.manifest["gates"].items() if not v)
        self.assertEqual(failed, [])

    def test_memory_max_960m_strictly_below_one_gib(self) -> None:
        self.assertEqual(iq04.MEMORY_MAX_BYTES, 1_006_632_960)
        self.assertLess(iq04.MEMORY_MAX_BYTES, 1_073_741_824)
        self.assertEqual(self.manifest["cage"]["memory_max_bytes"], 1_006_632_960)
        self.assertIn("MemoryMax=1006632960", self.manifest["cage"]["systemd_run_command"])

    def test_seven_arms_in_order(self) -> None:
        self.assertEqual(list(self.manifest["arms"]), list(iq04.ARM_NAMES))
        self.assertEqual(len(iq04.ARM_NAMES), 7)

    def test_two_hundred_families_four_kinds(self) -> None:
        records, truth = iq04.build_families()
        self.assertEqual(len(truth), 200)
        kinds = [t["kind"] for t in truth.values()]
        for kind in iq04.KINDS:
            self.assertEqual(kinds.count(kind), 50)
        self.assertGreater(len(records), 200)

    def test_symmetric_budget_identical_across_arms(self) -> None:
        budgets = {json.dumps(a["budget"], sort_keys=True) for a in self.manifest["arms"].values()}
        self.assertEqual(len(budgets), 1)
        for arm in self.manifest["arms"].values():
            self.assertEqual(arm["distinct_budget_configs_seen"], 1)

    def test_clean_recall_loss_within_tolerance(self) -> None:
        self.assertLessEqual(self.manifest["clean_recall_loss_vs_raw_append"], iq04.CLEAN_RECALL_LOSS_MAX)

    def test_drycheck_labels_offline_adapters(self) -> None:
        self.assertEqual(self.manifest["arms"]["live_memos_upstream_adapter"]["adapter_kind"], "memos_offline_contract_adapter")
        self.assertEqual(self.manifest["arms"]["mem0_adapter"]["adapter_kind"], "mem0_offline_contract_adapter")

    def test_chelated_beats_challenger_on_echo_kind(self) -> None:
        acc = {n: a["accuracy_by_kind"] for n, a in self.manifest["arms"].items()}
        self.assertEqual(acc["chelated_iq04_appliance"]["paraphrase_echo_vs_independent"], 1.0)
        self.assertLess(acc["equally_tuned_sqlite_c3_challenger"]["paraphrase_echo_vs_independent"], 1.0)

    def test_run_live_fails_closed_without_upstream(self) -> None:
        inventory = {"documents": ["a" * 64], "queries": ["b" * 64]}
        if importlib.util.find_spec("memos") is not None and importlib.util.find_spec("mem0") is not None:
            self.skipTest("upstream packages installed; live path would run")
        with self.assertRaises(iq04.LiveAdapterUnavailableError):
            iq04.run_iq04_experiment(mode="run-live", repo_root=REPO_ROOT, historical_inventory=inventory)

    def test_run_live_requires_inventory(self) -> None:
        with self.assertRaises(iq04.LiveAdapterUnavailableError):
            iq04.run_iq04_experiment(mode="run-live", repo_root=REPO_ROOT)

    def test_burn_inventory_must_be_complete(self) -> None:
        records, truth = iq04.build_families(count=8)
        burned = iq04.assert_families_burn_clean(
            records, truth, {"documents": [iq04.text_fingerprint(records[0].text)], "queries": ["c" * 64]}, REPO_ROOT
        )
        self.assertFalse(burned["burn_clean"])


class SSDKVTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest = kvc.run_ssd_kv_experiment(mode="local-drycheck", repo_count=3, commits_per_repo=50)
        cls.model = kvc.build_model()

    def test_all_mechanism_gates_pass(self) -> None:
        failed = sorted(k for k, v in self.manifest["gates"].items() if not v)
        self.assertEqual(failed, [])

    def test_logit_and_greedy_equivalence(self) -> None:
        eq = self.manifest["equivalence"]
        self.assertLess(eq["chelated_max_abs_logit_diff"], 1e-4)
        self.assertEqual(eq["chelated_greedy_mismatches"], 0)

    def test_stale_weight_revision_error_on_patch_and_adapter(self) -> None:
        probe = kvc.weight_revision_probe(self.model, trials=5)
        self.assertEqual(probe["refused_after_base_patch"], 5)
        self.assertEqual(probe["refused_after_adapter_change"], 5)
        self.assertEqual(probe["tra1_silent_reuses_after_patch"], 0)
        self.assertTrue(probe["config_sha256_identical"] and probe["tokenizer_sha256_identical"])
        self.assertEqual(probe["memt_style_silent_reuses_after_patch"], 5)
        self.assertIn("modeling_memt.py:876", probe["memt_citation"])

    def test_direct_reuse_raises_stale_weight_revision(self) -> None:
        c0 = kvc.contract_for(self.model)
        c1 = kvc.contract_for(self.model.patched_base())
        self.assertNotEqual(c0.digest(), c1.digest())
        cache = kvc.TRA1BoundStateCache()
        cache.seal(b"\x01" * 32, c0, b"kv")
        with self.assertRaises(StaleWeightRevisionError):
            cache.reuse(b"\x01" * 32, c1)

    def test_chain_key_binds_prefix_and_position(self) -> None:
        a = kvc.chain_hash("genesis", np.arange(4), 0, 4)
        self.assertEqual(a, kvc.chain_hash("genesis", np.arange(4), 0, 4))
        self.assertNotEqual(a, kvc.chain_hash("genesis", np.arange(4), 3, 7))
        self.assertNotEqual(kvc.chain_hash(a, np.arange(3), 4, 7), kvc.chain_hash(kvc.chain_hash("genesis", np.arange(4) + 1, 0, 4), np.arange(3), 4, 7))

    def test_naive_block_only_reuse_breaks_parity(self) -> None:
        control = kvc.positional_binding_control(self.model, trials=10)
        self.assertGreater(control["naive_block_only_reuse_exceeds_tol_trials"], 0)

    def test_retraction_invalidates_downstream_and_challenger_serves_stale(self) -> None:
        ret = self.manifest["retraction"]
        self.assertEqual(ret["chelated_stale_served_blocks"], 0)
        self.assertGreater(ret["downstream_clean_blocks_invalidated"], 0)
        self.assertGreater(ret["challenger_stale_served_blocks"], 0)

    def test_repo_selection_hash_ranked_and_distinct(self) -> None:
        names = kvc.select_repo_names(6001)
        self.assertEqual(len(names), 20)
        self.assertEqual(len(set(names)), 20)
        self.assertEqual(names, kvc.select_repo_names(6001))

    def test_performance_tie_is_reported_not_hidden(self) -> None:
        self.assertIn("performance_claim_supported", self.manifest)
        self.assertEqual(self.manifest["retirement_trigger_fired"], not self.manifest["performance_claim_supported"])

    def test_run_live_fails_closed(self) -> None:
        with self.assertRaises(kvc.LiveModeUnavailableError):
            kvc.run_ssd_kv_experiment(mode="run-live")


class RunnerTests(unittest.TestCase):
    def test_runners_write_receipt_bound_bundle(self) -> None:
        runner = _load("scripts/run_pv7_w19_2d_sub1gib_iq04.py")
        manifest = iq04.run_iq04_experiment(mode="local-drycheck", repo_root=REPO_ROOT)
        tmp = Path(tempfile.mkdtemp(prefix="pv7_runner_"))
        try:
            mpath, rpath = runner.write_receipt_bound_bundle(tmp, manifest)
            receipt = json.loads(rpath.read_text(encoding="utf-8"))
            self.assertEqual(receipt["schema_version"], "pv7_report_receipt_v1")
            self.assertEqual(receipt["bundle_id"], 59)
            import hashlib

            self.assertEqual(receipt["manifest_sha256"], hashlib.sha256(mpath.read_bytes()).hexdigest())
        finally:
            shutil.rmtree(tmp, ignore_errors=True)  # single tmp dir created above

    def test_runner_modules_import(self) -> None:
        for rel in ("scripts/run_pv7_w19_2b_cartridges.py", "scripts/run_pv7_w19_2e_ssd_kv_repos.py"):
            module = _load(rel)
            self.assertTrue(callable(module.main))
            self.assertTrue(callable(module.write_receipt_bound_bundle))


if __name__ == "__main__":
    unittest.main()
