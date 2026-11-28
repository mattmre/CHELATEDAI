"""Regression tests for the Wave 16 reservation, exposure, and integration gates."""

import hashlib
import io
import json
import math
import random
import tokenize
import unittest
import warnings
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from chelated_pv7_spine.aa_null_calibration import (
    ALPHA,
    INFLATION_P,
    binomial_upper_tail_p,
    clopper_pearson,
    clopper_pearson_lower_one_sided,
    deterministic_drycheck_pools,
    mcnemar_exact,
    ndcg_at_10,
    ndcg_at_10_from_qrels,
    run_aa_calibration,
    run_deterministic_drycheck,
    verify_aa_calibration_receipt,
)
from chelated_pv7_spine.burn_ledger import (
    BURNED_CORPORA,
    HISTORICAL_DATASETS,
    INVENTORY_SCOPE_COMPLETE,
    INVENTORY_SCOPE_NONE,
    INVENTORY_SCOPE_PARTIAL,
    INVENTORY_SCOPE_SMOKE,
    REGISTRY_SPEC,
    BurnedCorpusError,
    BurnLedger,
    assert_corpus_unburned,
    build_burn_ledger,
    normalize_dataset_id,
    text_fingerprint,
)
from chelated_pv7_spine.claim_registry import (
    adjudicate_holm,
    claim_registry_sha256,
    load_claim_registry,
    with_bundle_status,
)
from chelated_pv7_spine.fragment_compiler import (
    FRAGMENT_DIRS,
    FragmentCompileError,
    compile_fragments,
)

ROOT = Path(__file__).resolve().parents[1]
SPINE_DIGEST = hashlib.sha256(b"test spine wheel fixture").hexdigest()


def json_fragment(**fields):
    return json.dumps({"spine_digest": SPINE_DIGEST, **fields}).encode("utf-8")


def registry(bundle, slug="sample"):
    return f"{FRAGMENT_DIRS['registry']}/{bundle}-{slug}.json"


def routes(bundle):
    return f"{FRAGMENT_DIRS['routes']}/{bundle}-routes.json"


def verifier(bundle, extra=""):
    body = f'SPINE_DIGEST = "{SPINE_DIGEST}"\nBUNDLE_ID = {bundle}\n{extra}'
    return f"{FRAGMENT_DIRS['verifiers']}/{bundle}_verifier.py", body.encode("utf-8")


class ClaimRegistryTests(unittest.TestCase):
    def test_complete_reservations_and_canonical_hash(self):
        value = load_claim_registry()
        self.assertEqual([row["id"] for row in value["bundles"]], list(range(47, 65)))
        self.assertEqual([row["wave"] for row in value["bundles"]], [16] + [17] * 2 + [18] * 6 + [19] * 5 + [20] * 2 + [21] * 2)
        self.assertEqual(value["bundles"][1]["additional_tiers"], ["T0"])
        self.assertEqual(value["bundles"][-1]["branch"], "pv7/t5b-partner-records")
        self.assertEqual({row["status"] for row in value["bundles"]}, {"reserved-pending"})
        self.assertEqual(claim_registry_sha256(value), "32693f54e0ce7ad3d401ceb752ae4723b0eee251f6d49b95cdfb26a6201971e9")
        value["bundles"].reverse()
        self.assertEqual(claim_registry_sha256(value), claim_registry_sha256())

    def test_missing_duplicate_reassigned_and_vacated_reservations(self):
        for change, message in (
            (lambda rows: rows.pop(), "Missing permanent"),
            (lambda rows: rows[0].update(id=48), "duplicate permanent"),
            (lambda rows: rows[0].update(id=46), "Invalid"),
            (lambda rows: rows[0].pop("status"), "vacated status"),
            (lambda rows: rows[0].update(branch="pv7/reassigned"), "metadata has changed"),
        ):
            value = load_claim_registry()
            change(value["bundles"])
            with self.subTest(message=message), self.assertRaisesRegex(ValueError, message):
                claim_registry_sha256(value)

    def test_loss_preserves_the_id_and_cannot_be_reopened(self):
        original = load_claim_registry()
        updated = with_bundle_status(original, 50, "verified-loss")
        self.assertNotEqual(claim_registry_sha256(original), claim_registry_sha256(updated))
        self.assertEqual(updated["bundles"][3]["status"], "verified-loss")
        self.assertEqual(original["bundles"][3]["status"], "reserved-pending")
        for status in ("reserved-pending", "verified-pass"):
            with self.subTest(status=status), self.assertRaisesRegex(ValueError, "cannot change"):
                with_bundle_status(updated, 50, status)

    def test_holm_stops_at_first_failure_and_uses_worst_component(self):
        p_values = {
            "3A": 0.004,
            "H6B_merge": 0.010,
            "H6B_bridge_shift_and_4hop": {"prompt_shift": 0.001, "four_hop": 0.030},
            "6C": {"lineage": 0.012, "superiority": 0.024},
            "6D": 0.049,
        }
        results = adjudicate_holm(p_values)
        self.assertEqual([hypothesis for hypothesis, result in results.items() if result["reject"]], ["3A", "H6B_merge"])
        self.assertEqual(results["H6B_bridge_shift_and_4hop"]["p_value"], 0.030)
        self.assertAlmostEqual(results["6C"]["adjusted_p_value"], 0.072)
        self.assertAlmostEqual(results["6D"]["adjusted_p_value"], 0.072)
        with self.assertRaisesRegex(ValueError, "complete T1 family"):
            adjudicate_holm({key: value for key, value in p_values.items() if key != "6D"})
        with self.assertRaisesRegex(ValueError, "named prompt_shift and four_hop"):
            adjudicate_holm({**p_values, "H6B_bridge_shift_and_4hop": 0.001})


class BurnLedgerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ledger = build_burn_ledger(ROOT)

    def test_live_history_scans_scripts_registry_and_literal_loaders(self):
        data = self.ledger.to_dict()
        for name in HISTORICAL_DATASETS:
            with self.subTest(dataset=name):
                self.assertEqual(data["datasets"][name]["status"], "BURNED")
                self.assertTrue(any(source.startswith("scripts/") for source in data["datasets"][name]["sources"]))
        self.assertIn(REGISTRY_SPEC, data["datasets"]["nfcorpus"]["sources"])
        self.assertIn("scripts/run_flagship1_r4_comprehensive.py", data["datasets"]["cosqa"]["sources"])
        self.assertEqual(self.ledger.sha256, build_burn_ledger(ROOT).sha256)

    def test_aliases_and_casefolding(self):
        for dataset, canonical in (
            ("BeIR/arguana", "arguana"),
            (" ArguAna ", "arguana"),
            ("FiQA2018", "fiqa"),
            ("BeIR/fiqa-qrels", "fiqa"),
            ("https://huggingface.co/datasets/BeIR/nfcorpus", "nfcorpus"),
            ("xiaowu0162/longmemeval-cleaned", "longmemeval"),
            ("Percena/LoCoMo-MC10", "locomo"),
            ("osv_cve_seeds.json", "osv_cve_seeds"),
            ("msmarco-v2", "msmarco"),
            ("ms_marco_passage", "msmarco"),
            ("BeIR/msmarco-v2.1", "msmarco"),
            ("quora-duplicates", "quora"),
            ("scifact-v2", "scifact"),
            ("CoIR-Retrieval/cosqa", "cosqa"),
        ):
            with self.subTest(dataset=dataset):
                self.assertEqual(normalize_dataset_id(dataset), canonical)
                with self.assertRaisesRegex(BurnedCorpusError, f"is BURNED as {canonical}"):
                    assert_corpus_unburned(dataset, ledger=self.ledger)
        for distinct in ("scifact-open", "bioasq", "quoradataset"):
            with self.subTest(distinct=distinct):
                self.assertNotIn(normalize_dataset_id(distinct), BURNED_CORPORA)

    def test_burned_corpora_is_the_eleven_canonical_corpora(self):
        self.assertEqual(len(BURNED_CORPORA), 11)
        self.assertEqual(set(BURNED_CORPORA), set(HISTORICAL_DATASETS))
        self.assertIn("cosqa", BURNED_CORPORA)
        self.assertEqual(set(self.ledger.missing_burned_corpora), set(BURNED_CORPORA))
        self.assertEqual(self.ledger.inventory_scope, INVENTORY_SCOPE_NONE)
        self.assertFalse(self.ledger.confirmatory_eligible)

    def test_known_text_collision_even_under_a_distinct_corpus_label(self):
        abstract = "A CAFÉ  abstract with a ﬁnding."
        known_digest = hashlib.sha256("a café abstract with a finding.".encode()).hexdigest()
        self.assertEqual(text_fingerprint(abstract), known_digest)
        historical = {"documents": [known_digest.upper()], "queries": [text_fingerprint("A prior query")]}
        with self.assertRaisesRegex(BurnedCorpusError, "document #1.*collision"):
            assert_corpus_unburned("BioASQ-TaskB", ["a CAFE\u0301\nabstract with a finding."], ledger=self.ledger,
                                   historical_fingerprints=historical)
        with self.assertRaisesRegex(BurnedCorpusError, "query #1.*collision"):
            assert_corpus_unburned("BioASQ-TaskB", query_texts=[abstract], ledger=self.ledger,
                                   historical_fingerprints=historical)
        receipt = assert_corpus_unburned("BioASQ-TaskB", ["A separate abstract"], ["A new question"],
                                         historical_fingerprints=historical, ledger=self.ledger,
                                         allow_local_drycheck_smoke=True)
        self.assertEqual(receipt["inventory_scope"], INVENTORY_SCOPE_SMOKE)
        self.assertFalse(receipt["confirmatory_eligible"])

    def test_fingerprint_coverage_and_input_validation_fail_closed(self):
        for historical in (None, [], {"documents": [], "queries": []}, {"documents": []}):
            with self.subTest(historical=historical), self.assertRaisesRegex(BurnedCorpusError, "Cannot establish"):
                assert_corpus_unburned("NewCorpus", ["a document"], ["a query"], historical, ledger=self.ledger)
        with self.assertRaisesRegex(BurnedCorpusError, "Candidate document inventory is empty"):
            assert_corpus_unburned("NewCorpus", [], ["a query"], ledger=self.ledger)

    def test_typed_fingerprints_are_collected_from_sources(self):
        known_document = "historical text"
        known_query = "historical query?"
        real_open = tokenize.open
        real_read = Path.read_text

        def read_script(path):
            if path.name == "night1_common.py":
                return io.StringIO(f"QUERY_TEXTS = [{known_query!r}]\n")
            return real_open(path)

        def read_registry(path, *args, **kwargs):
            if path.as_posix().endswith(REGISTRY_SPEC):
                return json.dumps({"bundles": [{"dataset_id": "source/open-tests", "doc_texts": [known_document]}]})
            return real_read(path, *args, **kwargs)

        with patch("tokenize.open", side_effect=read_script), patch.object(Path, "read_text", read_registry):
            ledger = build_burn_ledger(ROOT)
            complete = build_burn_ledger(ROOT, historical_fingerprints={
                "documents": [text_fingerprint("another historical document")],
                "queries": [text_fingerprint("another historical query")],
            })
        self.assertFalse(ledger.fingerprint_inventory_complete)
        # A non-empty but corpus-unattributed inventory is partial, never complete (X2 / M-47c).
        self.assertFalse(complete.fingerprint_inventory_complete)
        self.assertEqual(complete.inventory_scope, INVENTORY_SCOPE_PARTIAL)
        self.assertFalse(complete.confirmatory_eligible)
        self.assertNotEqual(ledger.sha256, complete.sha256)
        with self.assertWarns(DeprecationWarning):
            legacy = assert_corpus_unburned("new-test-set", ["separate new document"], ["new question"],
                                            ledger=complete)
        self.assertEqual((legacy["inventory_scope"], legacy["smoke_opt_in"]),
                         (INVENTORY_SCOPE_SMOKE, "legacy_implicit"))
        self.assertFalse(legacy["confirmatory_eligible"])
        with self.assertRaisesRegex(BurnedCorpusError, "Cannot establish"):
            assert_corpus_unburned("new-test-set", ["separate new document"], ["new question"],
                                   ledger=complete, allow_local_drycheck_smoke=False)
        with self.assertRaisesRegex(BurnedCorpusError, "document #1.*collision"):
            assert_corpus_unburned("new-test-set", ["another historical document"], ledger=complete)
        self.assertEqual(dict(ledger.datasets)["open-tests"], (REGISTRY_SPEC,))
        self.assertIn(text_fingerprint(known_document), ledger.document_fingerprints)
        self.assertIn(text_fingerprint(known_query), ledger.query_fingerprints)
        with self.assertRaisesRegex(BurnedCorpusError, "query #1.*collision"):
            assert_corpus_unburned("new-test-set", query_texts=[known_query], ledger=ledger)

    def test_completeness_requires_all_eleven_burned_corpora(self):
        inventory = {
            "documents": [text_fingerprint("covered historical document")],
            "queries": [text_fingerprint("covered historical query")],
        }
        full = build_burn_ledger(ROOT, historical_fingerprints={**inventory, "corpora": list(BURNED_CORPORA)})
        self.assertTrue(full.fingerprint_inventory_complete)
        self.assertEqual(full.inventory_scope, INVENTORY_SCOPE_COMPLETE)
        self.assertTrue(full.confirmatory_eligible)
        self.assertEqual(full.missing_burned_corpora, ())
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            receipt = assert_corpus_unburned("fresh-corpus", ["new document"], ["new question"], ledger=full,
                                             allow_local_drycheck_smoke=False)
        self.assertTrue(receipt["admitted_unburned"])
        self.assertTrue(receipt["fingerprint_inventory_complete"])
        self.assertTrue(receipt["confirmatory_eligible"])
        self.assertEqual(receipt["burn_ledger_sha256"], full.sha256)

        aliased = list(BURNED_CORPORA[:-1]) + ["CoIR-Retrieval/cosqa"]
        self.assertTrue(build_burn_ledger(ROOT, historical_fingerprints={**inventory, "corpora": aliased})
                        .fingerprint_inventory_complete)
        missing_one = build_burn_ledger(ROOT, historical_fingerprints={**inventory, "corpora": BURNED_CORPORA[:-1]})
        self.assertFalse(missing_one.fingerprint_inventory_complete)
        self.assertEqual(missing_one.inventory_scope, INVENTORY_SCOPE_PARTIAL)
        self.assertEqual(missing_one.missing_burned_corpora, ("cosqa",))
        with self.assertRaisesRegex(BurnedCorpusError, "Cannot establish"):
            assert_corpus_unburned("fresh-corpus", ["new document"], ["new question"], ledger=missing_one,
                                   allow_local_drycheck_smoke=False)
        smoke = assert_corpus_unburned("fresh-corpus", ["new document"], ["new question"], ledger=missing_one,
                                       allow_local_drycheck_smoke=True)
        self.assertEqual((smoke["inventory_scope"], smoke["smoke_opt_in"], smoke["confirmatory_eligible"]),
                         (INVENTORY_SCOPE_SMOKE, "explicit", False))
        self.assertEqual(list(smoke["missing_burned_corpora"]), ["cosqa"])
        # Corpora claimed with no two-role inventory never count as coverage.
        hollow = BurnLedger((), frozenset({"a" * 64}), frozenset(), inventory_corpora=BURNED_CORPORA)
        self.assertFalse(hollow.fingerprint_inventory_complete)

    def test_bare_complete_claim_is_downgraded_to_local_drycheck_smoke(self):
        sentinel = BurnLedger(
            tuple((name, ("sentinel",)) for name in HISTORICAL_DATASETS),
            frozenset({"a" * 64}),
            frozenset({"b" * 64}),
            fingerprint_inventory_complete=True,
        )
        self.assertFalse(sentinel.fingerprint_inventory_complete)
        self.assertEqual(sentinel.inventory_scope, INVENTORY_SCOPE_SMOKE)
        self.assertFalse(sentinel.confirmatory_eligible)
        data = sentinel.to_dict()
        self.assertEqual((data["inventory_scope"], data["confirmatory_eligible"]), (INVENTORY_SCOPE_SMOKE, False))
        explicit = BurnLedger((), frozenset({"a" * 64}), frozenset({"b" * 64}), allow_local_drycheck_smoke=True)
        self.assertEqual(explicit.inventory_scope, INVENTORY_SCOPE_SMOKE)
        self.assertFalse(explicit.fingerprint_inventory_complete)


class FragmentCompilerTests(unittest.TestCase):
    def compile(self, sources):
        return compile_fragments(ROOT, SPINE_DIGEST, sources)

    def test_compiles_in_deterministic_dependency_order_without_import(self):
        sources = {
            registry(47): json_fragment(id=47, status="verified-pass"),
            registry(49): json_fragment(id=49, status="verified-loss", depends_on=[47]),
            registry(48): json_fragment(id=48, status="unsupported"),
            routes(48): json_fragment(routes=[{"method": "get", "path": "/api/test"}], depends_on=[49]),
            routes(49): json_fragment(routes=[{"method": "post", "path": "/api/test"}]),
            **dict([verifier(48, "DEPENDS_ON = [49]\nraise RuntimeError('do not execute verifier')\n")]),
            **dict([verifier(49)]),
        }
        compiled = self.compile(sources)
        self.assertEqual(compiled, self.compile(dict(reversed(list(sources.items())))))
        self.assertEqual(compiled["dependency_order"], [47, 49, 48])
        self.assertEqual([entry["id"] for entry in compiled["registry"]], [47, 48, 49])
        self.assertEqual([entry["method"] for entry in compiled["routes"]], ["GET", "POST"])
        self.assertEqual([entry["bundle_id"] for entry in compiled["verifiers"]], [49, 48])
        item = compiled["verifiers"][1]
        self.assertEqual(item["sha256"], hashlib.sha256(sources[item["path"]]).hexdigest())
        self.assertNotEqual(compiled["compiled_sha256"], self.compile({**sources, registry(47): json_fragment(
            status="unsupported"
        )})["compiled_sha256"])

    def test_duplicate_and_out_of_range_ids(self):
        for fragments, message in (
            ({registry(47): json_fragment(status="verified-pass"),
              registry(47, "collision"): json_fragment(status="unsupported")}, "Duplicate registry"),
            ({registry(65): json_fragment(status="verified-pass")}, "out-of-range bundle ID"),
            ({registry(49): json_fragment(id=50, status="verified-pass")}, "does not match filename"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(FragmentCompileError, message):
                self.compile(fragments)

    def test_rejects_duplicate_routes_with_case_normalized_methods(self):
        sample = json_fragment(status="verified-pass")
        with self.assertRaisesRegex(FragmentCompileError, "Duplicate HTTP route: GET /api/x"):
            self.compile({
                registry(47): sample, registry(48): sample,
                routes(47): json_fragment(routes=[{"method": "get", "path": "/api/x"}]),
                routes(48): json_fragment(routes=[{"method": "GET", "path": "/api/x"}]),
            })

    def test_rejects_traversal_and_absolute_paths(self):
        for path in ("docs/../unowned.json", r"docs\..\outside.json", r"C:\Windows\file.json", "//outside/file.json"):
            with self.subTest(path=path), self.assertRaisesRegex(FragmentCompileError, "traversal|escapes"):
                self.compile({registry(47): json_fragment(status="verified-pass", evidence={"manifest": path})})
        for route in ("/api/../x", "/api/%2e%2e/x", "/api//x", "https://example.com/x"):
            with self.subTest(route=route), self.assertRaisesRegex(FragmentCompileError, "noncanonical route path"):
                self.compile({
                    registry(47): json_fragment(status="verified-pass"),
                    routes(47): json_fragment(routes=[{"method": "GET", "path": route}]),
                })
        with self.assertRaisesRegex(FragmentCompileError, "traversal"):
            self.compile({
                registry(47): json_fragment(status="verified-pass"),
                **dict([verifier(47, 'ARTIFACT_PATH = "../outside.json"\n')]),
            })

    def test_rejects_cycles_missing_dependencies_and_orphans(self):
        sample = json_fragment(status="verified-pass")
        for sources, message in (
            ({registry(47): sample, registry(48): json_fragment(status="verified-pass", depends_on=[47]),
              **dict([verifier(47, "DEPENDS_ON = [48]\n")])}, "Dependency cycle"),
            ({registry(47): json_fragment(status="verified-pass", depends_on=[48])}, "Missing registry fragment"),
            ({**dict([verifier(47)])}, "Missing registry fragments"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(FragmentCompileError, message):
                self.compile(sources)

    def test_fails_closed_on_digest_status_and_json_errors(self):
        good = json_fragment(status="verified-pass")
        verifier_path, verifier_body = verifier(47)
        for sources, message in (
            ({registry(47): json_fragment(status="verified-pass", spine_digest="1" * 64)}, "Mismatched spine_digest"),
            ({registry(47): good, verifier_path: verifier_body.replace(SPINE_DIGEST.encode(), b"0" * 64)},
             "Mismatched spine_digest"),
            ({registry(47): good, verifier_path: verifier_body + b"return\n"}, "outside function"),
            ({registry(47): json_fragment(status="reserved-pending")}, "Invalid published registry status"),
            ({registry(47): good.replace(b"{", b'{"status": "unsupported", ', 1)}, "Duplicate JSON key"),
        ):
            with self.subTest(message=message), self.assertRaisesRegex(FragmentCompileError, message):
                self.compile(sources)


class AaCalibrationTests(unittest.TestCase):
    OPTIONS = {"host_id": "unit-test fixture", "permutations": 99, "split_size": 10}

    def test_metrics_and_exact_binomial_bounds(self):
        self.assertAlmostEqual(ndcg_at_10([0, 3, 1], [3, 1]),
                               (7 / math.log2(3) + 1 / math.log2(4)) / (7 + 1 / math.log2(3)))
        self.assertEqual(ndcg_at_10([0], [0]), 0)
        self.assertEqual(ndcg_at_10([1e4], [1e4]), 1)
        self.assertAlmostEqual(mcnemar_exact([(1, 0)] * 6 + [(0, 0)] * 4), 0.03125)
        self.assertEqual(mcnemar_exact([(0, 0), (1, 1)]), 1)
        low, high = clopper_pearson(0, 200)
        self.assertEqual(low, 0)
        self.assertAlmostEqual(high, 1 - 0.025 ** (1 / 200), places=13)
        low, high = clopper_pearson(10, 200)
        self.assertAlmostEqual(sum(math.comb(200, k) * low ** k * (1 - low) ** (200 - k)
                                   for k in range(10, 201)), 0.025, places=12)
        self.assertAlmostEqual(sum(math.comb(200, k) * high ** k * (1 - high) ** (200 - k)
                                   for k in range(11)), 0.025, places=12)

    def test_ndcg_idcg_uses_full_qrels_not_retrieved_list(self):
        qrels = {"d1": 3, "d2": 2, "d3": 1, "d_missing": 3}
        observed = ndcg_at_10_from_qrels(["d2", "x", "d1"], qrels)
        dcg = 3 / math.log2(2) + 7 / math.log2(4)
        idcg = 7 / math.log2(2) + 7 / math.log2(3) + 3 / math.log2(4) + 1 / math.log2(5)
        self.assertAlmostEqual(observed, dcg / idcg)
        # Retrieved-list IDCG would have scored this ranking higher.
        self.assertLess(observed, ndcg_at_10([2, 0, 3], [3, 2]))
        self.assertAlmostEqual(ndcg_at_10_from_qrels(["d1", "d_missing", "d2", "d3"], qrels), 1)
        self.assertEqual(ndcg_at_10_from_qrels(["x"], {"d1": 0, "d2": 1}), 0)
        with self.assertRaisesRegex(ValueError, "non-empty qrels"):
            ndcg_at_10_from_qrels(["x"], {})
        with self.assertRaisesRegex(ValueError, "[Dd]uplicate"):
            ndcg_at_10_from_qrels(["d1", "d1"], qrels)

    def test_one_sided_inflation_boundary(self):
        self.assertEqual((ALPHA, INFLATION_P), (0.05, 0.01))
        self.assertAlmostEqual(binomial_upper_tail_p(0, 200), 1.0)
        self.assertGreaterEqual(binomial_upper_tail_p(18, 200), INFLATION_P)
        self.assertLess(binomial_upper_tail_p(19, 200), INFLATION_P)
        self.assertLessEqual(clopper_pearson_lower_one_sided(18, 200), ALPHA)
        self.assertGreater(clopper_pearson_lower_one_sided(19, 200), ALPHA)
        self.assertAlmostEqual(binomial_upper_tail_p(19, 200),
                               sum(math.comb(200, k) * 0.05 ** k * 0.95 ** (200 - k) for k in range(19, 201)),
                               places=12)

    def test_null_calibration_hash_reproducibility_and_authentication(self):
        rng = random.Random(164)
        retrieval = [(rng.random(), rng.random()) for _ in range(2000)]
        latency = [(100 * rng.expovariate(1), 100 * rng.expovariate(1)) for _ in range(2000)]
        storage = [(1000 * rng.expovariate(1), 1000 * rng.expovariate(1)) for _ in range(2000)]
        disposition = [(rng.getrandbits(1), rng.getrandbits(1)) for _ in range(2000)]
        samples = retrieval, latency, storage, disposition
        key = b"test-only-signing-key"
        receipt = run_aa_calibration(*samples, **self.OPTIONS, signing_key=key)
        repeated = run_aa_calibration(*samples, **self.OPTIONS)
        self.assertTrue(receipt["passed"])
        self.assertEqual(receipt["trials"], 200)
        for metric in receipt["metrics"].values():
            self.assertTrue(metric["disjoint_splits"])
            self.assertEqual(metric["pairs_used"], 2000)
            self.assertLessEqual(metric["rejections"], 18)
            self.assertFalse(metric["anti_conservative"])
        self.assertEqual(receipt["receipt_sha256"], repeated["receipt_sha256"])
        self.assertTrue(verify_aa_calibration_receipt(repeated))
        self.assertFalse(verify_aa_calibration_receipt(receipt))
        self.assertTrue(verify_aa_calibration_receipt(receipt, signing_key=key))
        self.assertFalse(verify_aa_calibration_receipt(receipt, signing_key=b"wrong-key"))
        tampered = deepcopy(receipt)
        tampered["metrics"]["latency_p95"]["rejections"] += 1
        self.assertFalse(verify_aa_calibration_receipt(tampered, signing_key=key))

    def test_disjoint_splits_need_enough_pairs(self):
        rng = random.Random(7)
        pool = [(rng.random(), rng.random()) for _ in range(1999)]
        with self.assertRaisesRegex(ValueError, "disjoint"):
            run_aa_calibration(pool, pool, pool, [(1, 0)] * 1999, **self.OPTIONS)
        with self.assertRaisesRegex(ValueError, "at least 200"):
            run_aa_calibration(pool, pool, pool, pool, **self.OPTIONS, trials=199)

    def test_conservative_zero_rejections_pass(self):
        # Mostly identical pairs with sparse, sign-balanced differences: the tests never reject,
        # and a conservative calibration must PASS rather than fail for "alpha outside the CI".
        rng = random.Random(11)
        numeric = [(5.0, 5.0)] * 1950 + [(5.0, 6.0)] * 25 + [(6.0, 5.0)] * 25
        unit = [(a / 10, b / 10) for a, b in numeric]
        binary = [(1, 1)] * 1950 + [(1, 0)] * 25 + [(0, 1)] * 25
        for pool in (numeric, unit, binary):
            rng.shuffle(pool)
        receipt = run_aa_calibration(unit, numeric, numeric, binary, **self.OPTIONS)
        self.assertEqual([metric["rejections"] for metric in receipt["metrics"].values()], [0] * 4)
        for metric in receipt["metrics"].values():
            self.assertFalse(metric["degenerate_identical_measurements"])
            self.assertAlmostEqual(metric["inflation_p_value"], 1.0)
            self.assertTrue(metric["passed"])
        self.assertTrue(receipt["passed"])

    def test_detects_anti_conservative_inflation(self):
        rng = random.Random(23)
        shifted = [(0.6 * rng.random(), 0.6 * rng.random() + 0.25) for _ in range(2000)]
        latency = [(100 * rng.expovariate(1), 100 * rng.expovariate(1)) for _ in range(2000)]
        storage = [(1000 * rng.expovariate(1), 1000 * rng.expovariate(1)) for _ in range(2000)]
        disposition = [(rng.getrandbits(1), rng.getrandbits(1)) for _ in range(2000)]
        receipt = run_aa_calibration(shifted, latency, storage, disposition, **self.OPTIONS)
        retrieval = receipt["metrics"]["retrieval_ndcg_at_10"]
        self.assertGreater(retrieval["rejections"], 18)
        self.assertTrue(retrieval["anti_conservative"])
        self.assertLess(retrieval["inflation_p_value"], INFLATION_P)
        self.assertGreater(retrieval["clopper_pearson_lower_one_sided"], ALPHA)
        self.assertFalse(retrieval["passed"])
        self.assertFalse(receipt["passed"])

    def test_detects_systematic_bias_and_degenerate_identical_measurements(self):
        biased = [(0, 1)] * 2000
        same = [(1, 1)] * 2000
        receipt = run_aa_calibration(biased, biased, biased, biased, **self.OPTIONS)
        self.assertFalse(receipt["passed"])
        self.assertTrue(all(metric["anti_conservative"] for metric in receipt["metrics"].values()))
        self.assertTrue(all(metric["rejections"] > 18 for metric in receipt["metrics"].values()))
        receipt = run_aa_calibration(same, same, same, same, **self.OPTIONS)
        self.assertFalse(receipt["passed"])
        self.assertEqual([metric["rejections"] for metric in receipt["metrics"].values()], [0] * 4)
        self.assertTrue(all(metric["degenerate_identical_measurements"] for metric in receipt["metrics"].values()))
        self.assertEqual(receipt["metrics"]["storage_p95"]["identical_measured_pairs"], 2000)

    def test_deterministic_drycheck_and_47_verifier(self):
        pools = deterministic_drycheck_pools()
        self.assertEqual([len(pool) for pool in pools], [2000] * 4)
        self.assertIn("qrels", pools[0][0][0])
        receipt = run_deterministic_drycheck()
        self.assertTrue(receipt["passed"])
        self.assertTrue(verify_aa_calibration_receipt(receipt))
        self.assertEqual(receipt["receipt_sha256"], run_deterministic_drycheck()["receipt_sha256"])

        import importlib.util

        spec = importlib.util.spec_from_file_location("pv7_w16_47_verifier",
                                                      ROOT / "scripts" / "verifiers.d" / "47_verifier.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        result = module.verify_bundle(ROOT)
        self.assertTrue(result["passed"], result)
        self.assertEqual(result["status"], "local-drycheck-pass")
        self.assertEqual(result["spine_digest"], module.SPINE_DIGEST)
        self.assertTrue(result["fragments_ok"])
        self.assertRegex(result["compiled_sha256"], "^[0-9a-f]{64}$")
        self.assertEqual(result["burned_corpora_missing"], [])
        self.assertEqual(result["aa_null_calibration"]["receipt_sha256"], receipt["receipt_sha256"])
        self.assertTrue(result["aa_null_calibration"]["passed"])
        with patch("chelated_pv7_spine.aa_null_calibration.run_deterministic_drycheck",
                   return_value={**receipt, "passed": False}):
            failed = module.verify_bundle(ROOT)
        self.assertFalse(failed["passed"])
        self.assertEqual(failed["status"], "local-drycheck-fail")


if __name__ == "__main__":
    unittest.main()
