"""Regression tests for the Wave 16 reservation, exposure, and integration gates."""

import hashlib
import io
import json
import math
import random
import tokenize
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

from chelated_pv7_spine.aa_null_calibration import (
    clopper_pearson,
    mcnemar_exact,
    ndcg_at_10,
    run_aa_calibration,
    verify_aa_calibration_receipt,
)
from chelated_pv7_spine.burn_ledger import (
    HISTORICAL_DATASETS,
    REGISTRY_SPEC,
    BurnedCorpusError,
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
        ):
            with self.subTest(dataset=dataset):
                self.assertEqual(normalize_dataset_id(dataset), canonical)
                with self.assertRaisesRegex(BurnedCorpusError, f"is BURNED as {canonical}"):
                    assert_corpus_unburned(dataset, ledger=self.ledger)

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
        assert_corpus_unburned("BioASQ-TaskB", ["A separate abstract"], ["A new question"],
                               historical_fingerprints=historical, ledger=self.ledger)

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
        self.assertTrue(complete.fingerprint_inventory_complete)
        self.assertNotEqual(ledger.sha256, complete.sha256)
        assert_corpus_unburned("new-test-set", ["separate new document"], ["new question"], ledger=complete)
        with self.assertRaisesRegex(BurnedCorpusError, "document #1.*collision"):
            assert_corpus_unburned("new-test-set", ["another historical document"], ledger=complete)
        self.assertEqual(dict(ledger.datasets)["open-tests"], (REGISTRY_SPEC,))
        self.assertIn(text_fingerprint(known_document), ledger.document_fingerprints)
        self.assertIn(text_fingerprint(known_query), ledger.query_fingerprints)
        with self.assertRaisesRegex(BurnedCorpusError, "query #1.*collision"):
            assert_corpus_unburned("new-test-set", query_texts=[known_query], ledger=ledger)


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

    def test_null_calibration_hash_reproducibility_and_authentication(self):
        rng = random.Random(164)
        retrieval = [(rng.random(), rng.random()) for _ in range(4000)]
        latency = [(100 * rng.expovariate(1), 100 * rng.expovariate(1)) for _ in range(4000)]
        storage = [(1000 * rng.expovariate(1), 1000 * rng.expovariate(1)) for _ in range(4000)]
        disposition = [(rng.getrandbits(1), rng.getrandbits(1)) for _ in range(4000)]
        samples = retrieval, latency, storage, disposition
        options = {"host_id": "unit-test fixture", "permutations": 99, "split_size": 120}
        key = b"test-only-signing-key"
        receipt = run_aa_calibration(*samples, **options, signing_key=key)
        repeated = run_aa_calibration(*samples, **options)
        self.assertTrue(receipt["passed"])
        self.assertEqual(receipt["trials"], 200)
        self.assertEqual([metric["rejections"] for metric in receipt["metrics"].values()], [11, 8, 8, 11])
        self.assertEqual(receipt["receipt_sha256"], repeated["receipt_sha256"])
        self.assertTrue(verify_aa_calibration_receipt(repeated))
        self.assertFalse(verify_aa_calibration_receipt(receipt))
        self.assertTrue(verify_aa_calibration_receipt(receipt, signing_key=key))
        self.assertFalse(verify_aa_calibration_receipt(receipt, signing_key=b"wrong-key"))
        tampered = deepcopy(receipt)
        tampered["metrics"]["latency_p95"]["rejections"] += 1
        self.assertFalse(verify_aa_calibration_receipt(tampered, signing_key=key))

    def test_detects_systematic_bias_and_degenerate_identical_measurements(self):
        biased = [(0, 1)] * 40
        same = [(1, 1)] * 40
        options = {"host_id": "unit-test fixture", "permutations": 99, "split_size": 20}
        receipt = run_aa_calibration(biased, biased, biased, biased, **options)
        self.assertFalse(receipt["passed"])
        self.assertEqual([metric["rejections"] for metric in receipt["metrics"].values()], [200] * 4)
        receipt = run_aa_calibration(same, same, same, same, **options)
        self.assertFalse(receipt["passed"])
        self.assertEqual([metric["rejections"] for metric in receipt["metrics"].values()], [0] * 4)
        self.assertEqual(receipt["metrics"]["storage_p95"]["identical_measured_pairs"], 40)
        with self.assertRaisesRegex(ValueError, "at least 200"):
            run_aa_calibration(same, same, same, same, **options, trials=199)


if __name__ == "__main__":
    unittest.main()
