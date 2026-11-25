"""Bundle 55 item 6C schema drycheck and TRA-1 weight-revision gate.

Synthetic rows stay on deterministic_schema_drycheck. This module does not
label them wild and does not compile the live verifier directory.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPINE = "32693f54e0ce7ad3d401ceb752ae4723b0eee251f6d49b95cdfb26a6201971e9"
HEX_A = "a" * 64
HEX_B = "b" * 64
HEX_C = "c" * 64
HEX_D = "d" * 64
HEX_E = "e" * 64

EXPECTED_ORACLES = {
    "dry-echo": {
        "t_initial": ("affected", ("echo-origin",)),
        "t_mirror_echo_or_conflict": ("affected", ("echo-origin",)),
        "t_withdrawal_or_402": ("retracted", ()),
        "t_reinstatement_or_restated_10ka": ("affected", ("echo-reinstated",)),
    },
    "dry-conflict": {
        "t_initial": ("affected", ("conflict-ghsa",)),
        "t_mirror_echo_or_conflict": ("disputed", ("conflict-ghsa", "conflict-vex")),
        "t_withdrawal_or_402": ("not_affected", ("conflict-vex",)),
        "t_reinstatement_or_restated_10ka": ("disputed", ("conflict-ghsa-restored", "conflict-vex")),
    },
    "dry-402": {
        "t_initial": ("reliance_unchanged", ("sec-10k",)),
        "t_mirror_echo_or_conflict": ("reliance_unchanged", ("sec-10k",)),
        "t_withdrawal_or_402": ("non_reliance", ("sec-8k-402",)),
        "t_reinstatement_or_restated_10ka": ("restated_current", ("sec-restated-10ka",)),
    },
    "dry-amend": {
        "t_initial": ("reliance_unchanged", ("amend-10k",)),
        "t_mirror_echo_or_conflict": ("reliance_unchanged", ("amend-10k",)),
        "t_withdrawal_or_402": ("reliance_unchanged", ("amend-10ka",)),
        "t_reinstatement_or_restated_10ka": ("reliance_unchanged", ("amend-8ka",)),
    },
}


def _ensure_root():
    root = str(ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    return root


def _streams():
    _ensure_root()
    import chelated_wild_revision_streams as streams

    return streams


def _tra1():
    _ensure_root()
    from chelated_pv7_spine.tra1_weight_sha_guard import (
        StaleWeightRevisionError,
        TRA1BoundStateCache,
        TRA1StateIdentityContract,
        config_only_state_identity,
        contract_from_mapping,
        reproduce_memt_weight_omission,
        sha256_digest,
    )

    return {
        "error": StaleWeightRevisionError,
        "cache": TRA1BoundStateCache,
        "contract": TRA1StateIdentityContract,
        "config_only": config_only_state_identity,
        "from_mapping": contract_from_mapping,
        "reproduce": reproduce_memt_weight_omission,
        "sha": sha256_digest,
    }


def _runner():
    _ensure_root()
    name = "run_pv7_w18_6c_wild_revisions"
    if name in sys.modules:
        return sys.modules[name]
    path = ROOT / "scripts" / "run_pv7_w18_6c_wild_revisions.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _contract(api, **overrides):
    fields = {
        "base_weights_sha256": HEX_A,
        "adapter_weights_sha256": HEX_B,
        "config_sha256": HEX_C,
        "tokenizer_sha256": HEX_D,
        "kv_or_state_dtype": "float16",
        "rope_freq_base": 10000.0,
        "rope_scaling_hash": HEX_E,
    }
    fields.update(overrides)
    return api["contract"](**fields)


def _row(block, family_id, cut):
    matches = [row for row in block["rows"] if row["family_id"] == family_id and row["cut"] == cut]
    if len(matches) != 1:
        raise AssertionError(f"{family_id} {cut} matched {len(matches)} rows")
    return matches[0]


def _envelope(family_id, stratum, synthetic=False):
    return {
        "family_id": family_id,
        "stratum": stratum,
        "tenant_id": "tenant-real",
        "entity_key": family_id,
        "adjudicated": True,
        "synthetic": synthetic,
        "records": [
            {
                "record_id": family_id + "-r",
                "op_kind": "ASSERT",
                "claim_value": "reliance_unchanged",
                "authority_rank": 5,
                "valid_from": 1,
                "tx_time": 1,
                "text": "schema-drycheck-family control text " + family_id,
                "source": "sec",
            }
        ],
        "cuts": [
            {"name": "t_initial", "valid_time": 1, "tx_time": 1},
            {"name": "t_mirror_echo_or_conflict", "valid_time": 2, "tx_time": 2},
            {"name": "t_withdrawal_or_402", "valid_time": 3, "tx_time": 3},
            {"name": "t_reinstatement_or_restated_10ka", "valid_time": 4, "tx_time": 4},
        ],
    }


class TestTRA1WeightSha(unittest.TestCase):
    def test_config_only_identity_ignores_weight_patch_and_full_digest_does_not(self):
        api = _tra1()
        report = api["reproduce"](
            config_sha256=HEX_C,
            tokenizer_sha256=HEX_D,
            shape=(4, 4),
            base_weights_a=b"base-revision-a",
            base_weights_b=b"base-revision-b",
            adapter_weights_a=b"adapter-revision-a",
            adapter_weights_b=b"adapter-revision-b",
            kv_or_state_dtype="float16",
            rope_freq_base=10000.0,
            rope_scaling_hash=HEX_E,
        )
        self.assertEqual(report["silent_state_reuses"], 0)
        self.assertTrue(report["config_only_identity_stable_across_weight_patch"])
        self.assertTrue(report["digests_differ"])
        self.assertEqual(report["reuse_error"], "StaleWeightRevisionError")
        omitted = api["config_only"](HEX_C, HEX_D, (4, 4))
        self.assertEqual(omitted, report["config_tokenizer_shape_digest"])
        self.assertNotEqual(report["original_digest"], report["patched_digest"])

    def test_same_revision_reuses_and_field_changes_fail_closed(self):
        api = _tra1()
        original = _contract(api)
        cache = api["cache"]()
        cache.seal(b"kv-page-0", original, b"cached-state")
        self.assertEqual(cache.reuse(b"kv-page-0", original), b"cached-state")
        variants = [
            {"base_weights_sha256": HEX_E},
            {"adapter_weights_sha256": HEX_A},
            {"config_sha256": HEX_A},
            {"tokenizer_sha256": HEX_A},
            {"kv_or_state_dtype": "bfloat16"},
            {"rope_freq_base": 500000.0},
            {"rope_scaling_hash": HEX_A},
        ]
        for overrides in variants:
            with self.subTest(overrides=overrides):
                patched = _contract(api, **overrides)
                self.assertNotEqual(original.digest(), patched.digest())
                with self.assertRaises(api["error"]):
                    cache.reuse(b"kv-page-0", patched)
        self.assertEqual(len(cache._pages), 1)

    def test_mapping_and_rejected_inputs(self):
        api = _tra1()
        built = api["from_mapping"](
            {
                "base_weights_sha256": HEX_A,
                "adapter_weights_sha256": HEX_B,
                "config_sha256": HEX_C,
                "tokenizer_sha256": HEX_D,
                "kv_or_state_dtype": "float16",
                "rope_freq_base": 10000,
                "rope_scaling_hash": HEX_E,
            }
        )
        self.assertEqual(built.digest(), _contract(api, rope_freq_base=10000).digest())
        with self.assertRaises(api["error"]):
            api["from_mapping"]({"config_sha256": HEX_C})
        with self.assertRaises(api["error"]):
            _contract(api, rope_freq_base=0)
        with self.assertRaises(api["error"]):
            _contract(api, rope_freq_base=True)
        with self.assertRaises(api["error"]):
            _contract(api, base_weights_sha256="A" * 64)
        with self.assertRaises(api["error"]):
            api["sha"]("not-bytes")


class TestWildParsers(unittest.TestCase):
    def test_advisory_and_filing_parsers(self):
        streams = _streams()
        osv = streams.parse_osv_advisory(
            {
                "id": "OSV-1",
                "summary": "widget issue",
                "affected": [{"package": {"name": "widget", "ecosystem": "PyPI"}}],
            }
        )
        self.assertEqual(osv.source, "osv")
        self.assertEqual(osv.status, "affected")
        self.assertEqual(osv.product, "widget")
        seeds = streams.parse_osv_seed_cache(
            {"CVE-1": {"cve_id": "CVE-1", "osv_summary": "seed", "osv_package": "widget", "osv_fixed": "1.2.3"}}
        )
        self.assertEqual(seeds[0].source, "osv_seed_cache")
        self.assertEqual(seeds[0].advisory_id, "CVE-1")
        ghsa = streams.parse_ghsa_advisory(
            {
                "ghsaId": "GHSA-demo-2002",
                "summary": "widget affected",
                "vulnerabilities": [{"package": {"name": "widget"}}],
            }
        )
        nvd = streams.parse_nvd_cve(
            {
                "vulnerabilities": [
                    {
                        "cve": {
                            "id": "CVE-2024-0001",
                            "vulnStatus": "Rejected",
                            "descriptions": [{"lang": "en", "value": "rejected demo"}],
                        }
                    }
                ]
            }
        )
        self.assertEqual(nvd[0].status, "withdrawn")
        vex = streams.parse_csaf_vex(
            {
                "document": {"title": "vex", "tracking": {"id": "VEX-1"}},
                "vulnerabilities": [
                    {"id": "GHSA-demo-2002", "product_status": {"known_not_affected": ["widget"]}}
                ],
            }
        )
        self.assertEqual(vex[0].status, "not_affected")
        conflicts = streams.find_equal_authority_conflicts((ghsa, vex[0]))
        self.assertEqual(len(conflicts), 1)
        self.assertEqual(conflicts[0]["disposition"], "disputed")
        filings = streams.parse_sec_submission(
            {
                "filings": {
                    "recent": {
                        "form": ["8-K", "10-K/A", "8-K/A"],
                        "accessionNumber": ["acc-402", "acc-10ka", "acc-8ka"],
                        "filingDate": ["2024-01-01", "2024-02-01", "2024-03-01"],
                        "items": ["4.02", "9.01", "4.02"],
                    }
                }
            }
        )
        self.assertEqual(
            [row.sec_class for row in filings],
            [
                "item_402_nonreliance",
                "routine_amendment",
                "item_402_amendment_does_not_open_withdrawal",
            ],
        )
        self.assertEqual(filings[0].status, "non_reliance")
        self.assertTrue(all(row.status != "non_reliance" for row in filings[1:]))
        self.assertEqual(streams.classify_sec_filing("8-K", ["Item 4.02"]), "item_402_nonreliance")
        self.assertEqual(streams.classify_sec_filing("10-K/A", []), "routine_amendment")
        self.assertEqual(streams.classify_sec_filing("8-K/A", ["5.02"]), "routine_amendment")
        dispatched = streams.parse_snapshot_document(
            {"id": "OSV-2", "affected": [{"package": {"name": "widget"}}], "summary": "direct"}
        )
        self.assertEqual(dispatched[0].advisory_id, "OSV-2")

    def test_missing_seed_cache_and_protected_path(self):
        streams = _streams()
        with tempfile.TemporaryDirectory() as tmp:
            missing = Path(tmp) / "osv_cve_seeds.json"
            self.assertEqual(streams.load_osv_seed_cache(missing), ())
        with self.assertRaises(streams.StreamParseError):
            streams.load_json_snapshot(Path("sealed_key.json"))

    def test_synthetic_cannot_claim_wild_and_sparse_gate(self):
        streams = _streams()
        envelope = _envelope("synthetic-wild", "routine_amendment_control", synthetic=True)
        envelope["cohort_source"] = streams.COHORT_WILD
        with self.assertRaises(streams.StreamParseError):
            streams.parse_family_envelope(envelope)
        one = streams.parse_family_envelope(_envelope("real-1", "routine_amendment_control"))
        census = streams.count_adjudicated_families((one,))
        self.assertEqual(census["total"], 1)
        self.assertFalse(census["meets_wild_gate"])
        selected = streams.select_cohort((one,))
        self.assertEqual(selected["cohort_source"], streams.COHORT_DRYCHECK)
        self.assertEqual(selected["status"], streams.STATUS_DRYCHECK)
        self.assertEqual(selected["synthetic_labeled_wild"], 0)
        self.assertTrue(all(family.synthetic for family in selected["families"]))
        self.assertEqual(len(selected["descriptive_families"]), 1)
        full = []
        for stratum in streams.STRATA:
            for index in range(streams.WILD_PER_STRATUM):
                full.append(
                    streams.AdjudicatedFamily(
                        family_id=f"{stratum}-{index}",
                        stratum=stratum,
                        tenant_id="tenant-wild",
                        entity_key=f"{stratum}-{index}",
                        records=(),
                        cuts=(),
                        adjudicated=True,
                        synthetic=False,
                    )
                )
        gated = streams.select_cohort(tuple(full))
        self.assertEqual(gated["status"], "EVALUATED")
        self.assertEqual(gated["cohort_source"], streams.COHORT_WILD)
        self.assertEqual(len(gated["families"]), 240)
        self.assertTrue(all(not family.synthetic for family in gated["families"]))
        short = full[:-1]
        sparse = streams.select_cohort(tuple(short))
        self.assertEqual(sparse["cohort_source"], streams.COHORT_DRYCHECK)
        self.assertEqual(sparse["synthetic_labeled_wild"], 0)


class TestStrataOracle(unittest.TestCase):
    def test_four_strata_and_cycle(self):
        streams = _streams()
        families = streams.build_schema_drycheck_families()
        self.assertEqual([family.family_id for family in families], ["dry-echo", "dry-conflict", "dry-402", "dry-amend"])
        self.assertEqual({family.stratum for family in families}, set(streams.STRATA))
        for family in families:
            self.assertTrue(family.synthetic)
            self.assertEqual(family.to_dict()["cohort_source"], streams.COHORT_DRYCHECK)
            oracles = {oracle.name: oracle for oracle in streams.family_oracles(family)}
            self.assertEqual(tuple(oracles), streams.CUT_NAMES)
            for name, (disposition, lineage) in EXPECTED_ORACLES[family.family_id].items():
                self.assertEqual(oracles[name].disposition, disposition)
                self.assertEqual(oracles[name].independent_lineage_ids, lineage)
        for oracle in streams.family_oracles(families[3]):
            self.assertNotEqual(oracle.disposition, "non_reliance")
        cycle = streams.cycle_without_external_root()
        for oracle in streams.family_oracles(cycle):
            self.assertEqual(oracle.disposition, "retracted")
            self.assertEqual(oracle.independent_lineage_ids, ())

    def test_echo_precision_recall_and_zero_cross_tenant_leaks(self):
        streams = _streams()
        families = streams.build_schema_drycheck_families()
        pairs = streams.schema_echo_pairs(families)
        aware = streams.echo_metrics(
            pairs,
            predicate=lambda pair: streams.echo_decision(
                pair.left_text, pair.right_text, pair.left_tenant, pair.right_tenant
            ),
        )
        unsigned = streams.echo_metrics(
            pairs,
            predicate=lambda pair: streams.unsigned_correlation_collapse(pair.left_text, pair.right_text),
        )
        self.assertEqual(aware["precision"], 1.0)
        self.assertEqual(aware["recall"], 1.0)
        self.assertEqual(aware["cross_tenant_collapse_leaks"], 0)
        self.assertGreaterEqual(unsigned["cross_tenant_collapse_leaks"], 1)
        self.assertGreater(aware["true_positive"], 0)


class TestRunnerAndFragments(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        _ensure_root()
        from chelated_pv7_spine.claim_registry import REGISTRY_PATH

        cls.registry_before = REGISTRY_PATH.read_bytes()
        cls.runner = _runner()
        cls.report = cls.runner.run_wild_revision_comparison(mode="local-drycheck")

    def test_drycheck_labels_and_registry_untouched(self):
        report = self.report
        self.assertEqual(report["bundle_id"], 55)
        self.assertEqual(report["item"], "6C")
        self.assertEqual(report["status"], "DRYCHECK")
        self.assertEqual(report["cohort_source"], "deterministic_schema_drycheck")
        self.assertEqual(report["family_count"], 4)
        self.assertEqual(report["synthetic_labeled_wild"], 0)
        self.assertFalse(report["wild_gate_met"])
        self.assertEqual(report["wild_gate"], {"families": 240, "per_stratum": 60})
        self.assertEqual(report["llm_write_tokens"], 0)
        self.assertEqual(report["authored_prompts"], 0)
        self.assertEqual(report["tra1_silent_state_reuses"], 0)
        self.assertFalse(report["registry_mutated"])
        self.assertFalse(report["superiority_is_wild_evidence"])
        self.assertFalse(report["confirmatory_n"])
        self.assertFalse(report["retirement"]["superiority_is_wild_evidence"])
        self.assertEqual(report["registry_sha256"], SPINE)
        from chelated_pv7_spine.claim_registry import REGISTRY_PATH

        self.assertEqual(REGISTRY_PATH.read_bytes(), self.registry_before)
        self.assertTrue(all(row["synthetic"] for row in report["scored_families"]))
        self.assertTrue(all(row["cohort_source"] == "deterministic_schema_drycheck" for row in report["scored_families"]))
        self.assertNotIn("wild", {row["cohort_source"] for row in report["scored_families"]})

    def test_engines_echo_withdrawal_and_retirement_formula(self):
        report = self.report
        engines = report["engines"]
        sqlite_modes = engines["sqlite"]
        self.assertEqual(set(sqlite_modes), {"recursive_cte", "closure_table"})
        cte_hits = [row["hit"] for row in sqlite_modes["recursive_cte"]["rows"]]
        closure_hits = [row["hit"] for row in sqlite_modes["closure_table"]["rows"]]
        self.assertEqual(cte_hits, closure_hits)
        ek5 = engines["ek5_v3"]
        sql = sqlite_modes[engines["sqlite_dev_best"]]
        echo_cut = "t_withdrawal_or_402"
        self.assertEqual(_row(ek5, "dry-echo", echo_cut)["hit"], 1)
        self.assertEqual(_row(sql, "dry-echo", echo_cut)["hit"], 0)
        self.assertEqual(_row(engines["temporal_recency_rag"], "dry-echo", echo_cut)["hit"], 0)
        self.assertEqual(ek5["rows"][0]["oracle_disposition"], "affected")
        digest = report["ek5_audit_digest"]
        self.assertEqual(len(digest), 64)
        self.assertTrue(all(char in "0123456789abcdef" for char in digest))
        self.assertEqual(digest, ek5["audit_digest"])
        retirement = report["retirement"]
        lower = retirement["bootstrap_lcb"]
        ratio = retirement["sql_over_ek5_cost"]
        accuracy_superior = lower > 0.0
        sql_within_cost = ratio <= 1.25
        noninferior = lower >= 0.0
        mirror_superior = retirement["mirror_or_disputed_ek5_superior"]
        conjunction = noninferior and (mirror_superior or retirement["ek5_cost_at_least_1_25x_lower"])
        retire = (not accuracy_superior and sql_within_cost) or (not conjunction)
        self.assertEqual(retirement["retire_ek5_proprietary_superiority"], retire)
        self.assertEqual(retirement["preregistered_conjunction_met"], conjunction)
        self.assertEqual(retirement["comparison_engine"], "sqlite")
        self.assertGreaterEqual(ek5["accuracy"], sql["accuracy"])
        self.assertLess(
            engines["temporal_recency_rag"]["accuracy"],
            ek5["accuracy"],
        )
        self.assertGreaterEqual(
            engines["unsigned_correlation_filter"]["cross_tenant_collapse_leaks"],
            1,
        )
        self.assertEqual(report["echo"]["tenant_aware"]["cross_tenant_collapse_leaks"], 0)
        self.assertEqual(report["echo"]["tenant_aware"]["precision"], 1.0)
        self.assertEqual(report["echo"]["tenant_aware"]["recall"], 1.0)
        self.assertEqual(report["burn"]["osv_cve_seeds_rejected"], True)
        self.assertEqual(report["burn"]["drycheck_texts_admitted"], True)
        self.assertEqual(report["kernel"]["llm_write_tokens"], 0)
        self.assertIn(report["kernel"]["status"], {"ingested", "skipped"})
        if report["kernel"]["status"] == "ingested":
            self.assertEqual(report["kernel"]["ingested"], 3)
        self.assertEqual(engines["postgres17"]["status"], "not_requested")

    def test_sparse_snapshot_stays_descriptive(self):
        streams = _streams()
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "real-1.json").write_text(
                json.dumps(_envelope("real-1", "sec_item_402_nonreliance")),
                encoding="utf-8",
            )
            (folder / "ignored.json").write_text(
                json.dumps({"family_id": "nope", "stratum": streams.STRATA[0], "adjudicated": False}),
                encoding="utf-8",
            )
            report = self.runner.run_wild_revision_comparison(
                mode="local-drycheck",
                wild_snapshot_dir=folder,
            )
        self.assertEqual(report["status"], "DRYCHECK")
        self.assertEqual(report["cohort_source"], "deterministic_schema_drycheck")
        self.assertEqual(report["wild_adjudicated_count"], 1)
        self.assertEqual(report["descriptive_family_count"], 1)
        self.assertEqual(report["synthetic_labeled_wild"], 0)
        self.assertEqual(report["family_count"], 4)
        self.assertTrue(all(row["synthetic"] for row in report["scored_families"]))

    def test_postgres_failure_stays_on_sqlite(self):
        def explode(_dsn):
            raise RuntimeError("postgres down")

        report = self.runner.run_wild_revision_comparison(
            mode="local-drycheck",
            pg17_dsn="postgresql://127.0.0.1/unused",
            pg_connect=explode,
        )
        self.assertEqual(report["engines"]["postgres17"]["status"], "failed")
        self.assertEqual(report["engines"]["postgres17"]["modes"], {})
        self.assertEqual(report["retirement"]["comparison_engine"], "sqlite")
        self.assertEqual(report["status"], "DRYCHECK")
        self.assertFalse(report["superiority_is_wild_evidence"])

    def test_fragment_compile_uses_legal_verifier_key(self):
        _ensure_root()
        from chelated_pv7_spine.fragment_compiler import FragmentCompileError, _fragment_name, compile_fragments

        registry = (ROOT / "docs" / "research" / "registry.d" / "55-t6c-wild-revisions.json").read_bytes()
        verifier = (ROOT / "scripts" / "verifiers.d" / "55_verifier.py").read_bytes()
        compiled = compile_fragments(
            ROOT,
            SPINE,
            {
                "docs/research/registry.d/55-t6c-wild-revisions.json": registry,
                "scripts/verifiers.d/55_verifier.py": verifier,
            },
        )
        self.assertEqual(compiled["registry"][0]["id"], 55)
        self.assertEqual(compiled["registry"][0]["status"], "protocol-ready")
        self.assertEqual(compiled["registry"][0]["depends_on"], [])
        self.assertEqual(compiled["verifiers"][0]["bundle_id"], 55)
        self.assertEqual(len(compiled["compiled_sha256"]), 64)
        with self.assertRaises(FragmentCompileError):
            _fragment_name("scripts/verifiers.d/v55_t6c_wild_verifier.py")

    def test_owned_modules_do_not_import_model_clients(self):
        banned = {"openai", "anthropic", "litellm", "ollama"}
        paths = [
            ROOT / "chelated_wild_revision_streams.py",
            ROOT / "chelated_pv7_spine" / "tra1_weight_sha_guard.py",
            ROOT / "scripts" / "run_pv7_w18_6c_wild_revisions.py",
            ROOT / "scripts" / "verifiers.d" / "55_verifier.py",
            ROOT / "tests" / "test_pv7_w18_6c_and_tra1_weight_sha.py",
        ]
        for path in paths:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path.name))
            imported = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    imported.update(alias.name.split(".")[0] for alias in node.names)
                elif isinstance(node, ast.ImportFrom) and node.module:
                    imported.add(node.module.split(".")[0])
            self.assertFalse(imported & banned, path.name)

    def test_cycle_projection_retracts(self):
        streams = _streams()
        runner = self.runner
        cycle = streams.cycle_without_external_root()
        from ek5_v3_stratified import StratifiedEvidenceStore

        events, copies, withdraws = runner.project_ek5(cycle)
        store = StratifiedEvidenceStore()
        store.apply_mutation(events, copies)
        for edge in withdraws:
            store.apply_mutation((), [edge])
        for cut, oracle in zip(cycle.cuts, streams.family_oracles(cycle)):
            disposition, lineage = runner.ek5_prediction(store, cycle, cut)
            self.assertEqual(oracle.disposition, "retracted")
            self.assertEqual(disposition, "retracted")
            self.assertEqual(lineage, ())

    def test_oracle_at_cut_four_reproducers_and_frozen_adjudication(self):
        streams = _streams()
        cut = streams.CutSpec("t_withdrawal_or_402", 30, 30)

        # 1. Cross-tenant WITHDRAW must not withdraw victim tenant's record.
        cross_tenant_family = streams.AdjudicatedFamily(
            family_id="repro-cross-tenant",
            stratum="unlinked_cross_mirror_echo",
            tenant_id="tenant-good",
            entity_key="ent-1",
            synthetic=True,
            records=(
                streams.StreamRecord("r1", "tenant-good", "ent-1", "affected", 5, 10, None, 10, "ASSERT", "orig", "ghsa"),
                streams.StreamRecord("w-evil", "tenant-evil", "ent-1", "", 10, 20, None, 20, "WITHDRAW", "", "status", target_id="r1"),
            ),
            cuts=(cut,),
        )
        res_ct = streams.oracle_at_cut(cross_tenant_family, cut)
        self.assertEqual(res_ct.disposition, "affected")
        self.assertEqual(res_ct.independent_lineage_ids, ("r1",))

        # 2. Authority rank precedence BEFORE timestamp recency: rank-1 REINSTATE never overrides rank-5 WITHDRAW.
        rank_family = streams.AdjudicatedFamily(
            family_id="repro-rank-precedence",
            stratum="unlinked_cross_mirror_echo",
            tenant_id="t1",
            entity_key="ent-2",
            synthetic=True,
            records=(
                streams.StreamRecord("r1", "t1", "ent-2", "affected", 1, 10, None, 10, "ASSERT", "orig", "ghsa"),
                streams.StreamRecord("w-hi", "t1", "ent-2", "", 5, 20, None, 20, "WITHDRAW", "", "status", target_id="r1"),
                streams.StreamRecord("ri-lo", "t1", "ent-2", "", 1, 25, None, 25, "REINSTATE", "", "status", target_id="r1"),
            ),
            cuts=(cut,),
        )
        res_rank = streams.oracle_at_cut(rank_family, cut)
        self.assertEqual(res_rank.disposition, "retracted")
        self.assertEqual(res_rank.independent_lineage_ids, ())

        # 3a. Low-authority supersedes_ids must not supersede higher-authority assertion.
        low_sup_family = streams.AdjudicatedFamily(
            family_id="repro-low-sup",
            stratum="sec_item_402_nonreliance",
            tenant_id="t1",
            entity_key="ent-3",
            synthetic=True,
            records=(
                streams.StreamRecord("hi", "t1", "ent-3", "reliance_unchanged", 5, 10, None, 10, "ASSERT", "10k", "sec"),
                streams.StreamRecord("lo", "t1", "ent-3", "non_reliance", 1, 20, None, 20, "ASSERT", "rumor", "blog", supersedes_ids=("hi",)),
            ),
            cuts=(cut,),
        )
        res_low_sup = streams.oracle_at_cut(low_sup_family, cut)
        self.assertEqual(res_low_sup.disposition, "reliance_unchanged")
        self.assertEqual(res_low_sup.independent_lineage_ids, ("hi",))

        # 3b. Mutual supersede cycle is cycle-safe and order-independent -> disputed.
        rec_a = streams.StreamRecord("sup-a", "t1", "ent-4", "v1", 5, 10, None, 10, "ASSERT", "A", "s", supersedes_ids=("sup-b",))
        rec_b = streams.StreamRecord("sup-b", "t1", "ent-4", "v1", 5, 10, None, 10, "ASSERT", "B", "s", supersedes_ids=("sup-a",))
        cyc_ab = streams.oracle_at_cut(
            streams.AdjudicatedFamily("cyc-ab", "sec_item_402_nonreliance", "t1", "ent-4", (rec_a, rec_b), (cut,), synthetic=True),
            cut,
        )
        cyc_ba = streams.oracle_at_cut(
            streams.AdjudicatedFamily("cyc-ba", "sec_item_402_nonreliance", "t1", "ent-4", (rec_b, rec_a), (cut,), synthetic=True),
            cut,
        )
        self.assertEqual(cyc_ab.disposition, "disputed")
        self.assertEqual(cyc_ba.disposition, "disputed")
        self.assertEqual(cyc_ab.independent_lineage_ids, cyc_ba.independent_lineage_ids)

        # 4. Frozen adjudicated expected_admitted_ids / expected_disposition honored directly.
        env = _envelope("frozen-wild-1", "sec_item_402_nonreliance", synthetic=False)
        env["cuts"][0]["expected_disposition"] = "adjudicated_label"
        env["cuts"][0]["expected_admitted_ids"] = ["frozen-wild-1-r"]
        parsed = streams.parse_family_envelope(env)
        assert parsed is not None
        frozen_oracle = streams.oracle_at_cut(parsed, parsed.cuts[0])
        self.assertEqual(frozen_oracle.adjudication_source, "frozen_envelope")
        self.assertEqual(frozen_oracle.disposition, "adjudicated_label")
        self.assertEqual(frozen_oracle.independent_lineage_ids, ("frozen-wild-1-r",))

    def test_ek5_does_not_leak_echo_of_and_symmetric_inputs(self):
        streams = _streams()
        runner = self.runner
        # Unrelated text with ground-truth echo_of set: EK5 must NOT cheat by reading echo_of!
        family = streams.AdjudicatedFamily(
            family_id="no-echo-cheat",
            stratum="unlinked_cross_mirror_echo",
            tenant_id="t1",
            entity_key="e1",
            synthetic=True,
            records=(
                streams.StreamRecord("r-orig", "t1", "e1", "affected", 5, 10, None, 10, "ASSERT", "Alpha widget kernel flaw.", "ghsa"),
                streams.StreamRecord(
                    "r-unmatched-echo",
                    "t1",
                    "e1",
                    "affected",
                    5,
                    20,
                    None,
                    20,
                    "ASSERT",
                    "Completely dissimilar unrelated sentence about maritime weather.",
                    "nvd",
                    echo_of="r-orig",
                ),
                streams.StreamRecord(
                    "r-explicit-child",
                    "t1",
                    "e1",
                    "affected",
                    5,
                    20,
                    None,
                    20,
                    "ASSERT",
                    "Derived summary.",
                    "distro",
                    explicit_parent_ids=("r-orig",),
                ),
            ),
            cuts=(streams.CutSpec("t_initial", 20, 20),),
        )
        _events, copy_edges, _withdraws = runner.project_ek5(family)
        copy_pairs = {(edge.src_id, edge.dst_id) for edge in copy_edges}
        self.assertNotIn(("r-unmatched-echo", "r-orig"), copy_pairs)
        self.assertIn(("r-explicit-child", "r-orig"), copy_pairs)
        sql_ops = runner.sql_envelopes(family)
        sql_edges = {(env.src_id, env.dst_id) for kind, env in sql_ops if kind == "edge"}
        self.assertNotIn(("r-unmatched-echo", "r-orig"), sql_edges)
        self.assertIn(("r-explicit-child", "r-orig"), sql_edges)
        roots = runner._lineage_roots(family, ["r-orig", "r-unmatched-echo", "r-explicit-child"])
        self.assertIn("r-unmatched-echo", roots)
        self.assertNotIn("r-explicit-child", roots)

    def test_cluster_bootstrap_dedup_burn_and_live_wild_exit_codes(self):
        streams = _streams()
        runner = self.runner
        retirement = self.report["retirement"]
        self.assertEqual(retirement["sampling_unit"], "family_id")
        self.assertEqual(retirement["family_cluster_count"], 4)
        self.assertEqual(retirement["bootstrap_resamples"], 2000)
        self.assertEqual(len(retirement["family_deltas_ek5_minus_sql"]), 4)
        self.assertEqual(self.report["burn"]["burn_inventory_scope"], "local_drycheck_smoke")

        # Duplicate family_ids cannot inflate wild quota.
        dupes = tuple(
            streams.AdjudicatedFamily(
                family_id="dup-same",
                stratum=streams.STRATA[0],
                tenant_id="t",
                entity_key="e",
                records=(),
                cuts=(),
                adjudicated=True,
                synthetic=False,
            )
            for _ in range(300)
        )
        census = streams.count_adjudicated_families(dupes)
        self.assertEqual(census["total"], 1)
        self.assertEqual(census["duplicate_family_ids_dropped"], 299)
        self.assertFalse(census["meets_wild_gate"])

        # Live wild mode without meeting wild gate must exit non-zero (rc = 2).
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / "one.json").write_text(
                json.dumps(_envelope("real-only-one", "sec_item_402_nonreliance")),
                encoding="utf-8",
            )
            rc = runner.main(["--mode", "run-wild", "--wild-snapshot-dir", str(folder)])
            rep = runner.run_wild_revision_comparison(mode="run-wild", wild_snapshot_dir=folder)
        self.assertEqual(rc, 2)
        self.assertFalse(rep["passed"])
        self.assertEqual(len(rep["snapshot_sha256"]), 64)

        # Invalid mode rejected by argparse and run_wild_revision_comparison.
        with self.assertRaises(ValueError):
            runner.run_wild_revision_comparison(mode="bogus-mode")
        with self.assertRaises(SystemExit):
            runner.main(["--mode", "bogus-mode"])


if __name__ == "__main__":
    unittest.main()
