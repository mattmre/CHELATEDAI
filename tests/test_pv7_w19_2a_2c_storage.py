"""CPU-only storage protocol regressions for PV7 bundles 56 and 58."""

from __future__ import annotations

import importlib.util
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np  # noqa: E402

import chelated_cold_osd_scanner as osd  # noqa: E402
import chelated_disk_moe_runtime as moe  # noqa: E402
from chelated_pv7_spine.challenger_ledger import (  # noqa: E402
    ChallengerBitemporalLedger,
    LedgerRecordEnvelope,
    OpKind,
)
from chelated_pv7_spine.claim_registry import claim_registry_sha256  # noqa: E402
from chelated_pv7_spine.fragment_compiler import compile_fragments  # noqa: E402
from scripts.run_pv7_w19_2a_disk_moe import checked_lease  # noqa: E402
from chelated_pv7_spine.cpu_io_cage import ResidentIdleLeaseVerifier  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


class DiskMoETests(unittest.TestCase):
    def test_drycheck_reads_real_extents_and_matches_greedy(self):
        with tempfile.TemporaryDirectory(prefix=".pv7-56-test-", dir=ROOT) as directory:
            report = moe.drycheck(directory)
        self.assertTrue(report["passed"])
        self.assertEqual(list(report["rungs"]), list(moe.RUNGS))
        r1, r2, r3, r4, r5, r6 = (report["rungs"][key] for key in moe.RUNGS)
        self.assertEqual(len({arm["sequence_sha256"] for arm in (r1, r2, r3, r4, r6)}), 1)
        self.assertNotEqual(r5["sequence_sha256"], r1["sequence_sha256"])
        self.assertTrue(all("reference_token_similarity" in row for row in r5["separate_fixture_quality"]))
        self.assertGreater(report["accepted_draft_tokens"], 0)
        self.assertGreater(report["unaccepted_draft_tokens"], 0)
        self.assertGreater(sum(row["read_calls"] for row in r2["rows"]), sum(row["read_calls"] for row in r3["rows"]))
        self.assertTrue(all(row["requested_read_bytes"] > 0 for row in r3["rows"]))
        self.assertEqual(report["io_method"], "pread" if hasattr(os, "pread") else "offset_readfile")

    def test_hot_pin_survives_lru_eviction_and_read_order_uses_physical_lookup(self):
        payload = b"a" * 4096 + b"_" * 4096 + b"b" * 4096 + b"_" * 4096 + b"c" * 4096
        with tempfile.TemporaryDirectory(prefix=".pv7-56-test-", dir=ROOT) as directory:
            path = Path(directory) / "sparse-layout.bin"
            path.write_bytes(payload)
            extents = {i: moe.Extent(i * 8192, 4096) for i in range(3)}
            with moe.ExpertPager(
                path, extents, 6, cache_bytes=4096, pin_bytes=4096, physical_offset=lambda offset: -offset,
            ) as pager:
                with patch.object(pager, "_read", wraps=pager._read) as observed:
                    self.assertEqual(pager.fetch([0, 2])[2], b"c" * 4096)
                    self.assertEqual([args[0] for args, _ in observed.call_args_list], [16384, 0])
                    pager.fetch([0])
                    pager.fetch([1])
                    pager.fetch([2])
                    before = pager.bytes_read
                    self.assertEqual(pager.fetch([0])[0], b"a" * 4096)
                    self.assertEqual(pager.bytes_read, before)
                    self.assertEqual(pager.pinned_size, 4096)

    def test_inclusive_budget_charges_the_draft_kv_pin_and_buffers(self):
        parts = dict(
            target_weights=moe.MEMORY_MAX - 9, draft_weights=1, kv_cache=1, page_tables=1, expert_lru=1,
            expert_hot_pin=1, read_buffers=1, inference_workspace=1, routing_and_runtime=1, kernel_and_page_cache=1,
        )
        self.assertEqual(moe.ResidentBudget(**parts).total, moe.MEMORY_MAX)
        for field in parts:
            with self.subTest(field=field), self.assertRaisesRegex(ValueError, "budget exceeded"):
                moe.ResidentBudget(**{**parts, field: parts[field] + 1})

    def test_dev_comparator_selection_and_paired_lossless_gate(self):
        rows = {
            rung: [{"tokens": 10, "seconds": seconds}]
            for rung, seconds in enumerate((4, 2, 3, 3, 0.1, 0.05), start=1)
        }
        self.assertEqual(moe.select_dev_best(rows), 2)
        reference = [{"id": "01", "suite": "HumanEval", "trial": 0, "cache": "cold", "token_sha256": "same",
                      "tokens": 10, "seconds": 4.0, "disk_read_bytes": 16384}]
        candidate = [{**reference[0], "seconds": 1.0, "disk_read_bytes": 4096}]
        gate = moe.compare_lossless(reference, reference, candidate, resamples=50)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["speedup_lcb95"], 4.0)
        candidate[0]["token_sha256"] = "diverged"
        self.assertFalse(moe.compare_lossless(reference, reference, candidate, resamples=50)["passed"])

    def test_lease_fails_if_resident_serves_a_request(self):
        verifier = ResidentIdleLeaseVerifier()
        start = verifier.snapshot_from_text("vllm:request_success_total 0\nvllm:num_requests_running 0")
        after = verifier.snapshot_from_text("vllm:request_success_total 1\nvllm:num_requests_running 0")
        self.assertTrue(checked_lease(verifier, start, start).valid)
        with self.assertRaisesRegex(RuntimeError, "lease voided"):
            checked_lease(verifier, start, after)


class ColdOSDTests(unittest.TestCase):
    def test_drycheck_compares_30_pairs_with_no_rewrites_at_matched_recall(self):
        with tempfile.TemporaryDirectory(prefix=".pv7-58-test-", dir=ROOT) as directory:
            report = osd.drycheck(directory)
            extent_bytes = (Path(directory) / "ivf-flat-int8.extents").stat().st_size
        self.assertTrue(report["passed"])
        self.assertTrue(report["extents_unchanged"])
        self.assertEqual(report["trace_pairs"], 30)
        self.assertEqual(extent_bytes, report["extent_count"] * report["extent_bytes"])
        self.assertEqual(report["sidecar_bytes"], report["docs"] * 8)
        self.assertEqual(len(report["traces"]["cold_osd"]), 60)
        for layout in ("clustered", "scattered"):
            result = report["comparisons"]["matched_ivf_sq8"][layout]
            self.assertTrue(result["matched_recall"])
            self.assertTrue(result["zero_stale_resurrection"])
            self.assertGreater(result["write_bytes_saved"], 0)
            self.assertEqual(report["summaries"]["cold_osd"][layout]["total_write_bytes"], 0)
        self.assertGreater(
            report["summaries"]["matched_ivf_sq8"]["clustered"]["total_write_bytes"],
            report["summaries"]["matched_ivf_sq8"]["scattered"]["total_write_bytes"],
        )

    def test_ek5_fence_and_bitemporal_challenger_agree_across_a_single_epoch_replay(self):
        vectors = np.array([[1, 0.1], [1, 0.2], [0.1, 1], [0.2, 1]], dtype=np.float32)
        queries = np.eye(2, dtype=np.float32)
        with tempfile.TemporaryDirectory(prefix=".pv7-58-test-", dir=ROOT) as directory:
            geom = osd.build_extents(osd.VectorCorpus([vectors]), queries, queries, directory, 2 * osd.MIB, csls_k=2)
            index = osd.ColdOSD(geom, 2)
            with ChallengerBitemporalLedger() as ledger:
                tenant = "storage-arena"
                for tick, operation in enumerate((OpKind.ASSERT, OpKind.WITHDRAW, OpKind.REINSTATE)):
                    ledger.append(LedgerRecordEnvelope(tenant, "doc-0", "doc-0", 2, tick, None, tick,
                                                      operation, claim_value="current"))
                    if operation is OpKind.WITHDRAW:
                        index.withdraw([0])
                    elif operation is OpKind.REINSTATE:
                        index.reinstate([0])
                    oracle = ledger.query_as_of(tenant, "doc-0", tick, tick)
                    self.assertEqual(index.visible([0]), [oracle.winner_record_id == "doc-0"])
                    self.assertEqual(0 in index.search(queries[0]), oracle.winner_record_id == "doc-0")
            self.assertEqual(index.fence.epoch, 2)

    def test_qdrant_adapter_requires_on_disk_and_waited_mutations(self):
        requests = []

        def send(method, route, body=None, **kwargs):
            requests.append((method, route, body, kwargs))
            if method == "GET":
                return {
                    "config": {
                        "params": {"vectors": {"on_disk": True}}, "hnsw_config": {"on_disk": True},
                        "optimizer_config": {"flush_interval_sec": 5},
                    },
                    "status": "green", "optimizer_status": "ok",
                }
            return {}

        vectors = np.array([[1, 0.1], [0.1, 1]], dtype=np.float32)
        with patch.object(osd.QdrantOnDisk, "_request", side_effect=send):
            engine = osd.QdrantOnDisk("http://127.0.0.1:6333", "trace", osd.VectorCorpus([vectors]))
            self.assertTrue(engine.quiescent())
            engine.withdraw([0])
            engine.reinstate([0])
        self.assertTrue(requests[0][2]["vectors"]["on_disk"])
        self.assertTrue(requests[0][2]["hnsw_config"]["on_disk"])
        self.assertEqual(requests[-2][2], {"points": [0]})
        self.assertEqual([request[-1] for request in requests[-2:]],
                         [{"wait": "true", "ordering": "strong"}] * 2)


class FragmentTests(unittest.TestCase):
    def test_new_registry_fragments_compile_and_verifier_metadata_is_literal(self):
        pairs = {
            56: "docs/research/registry.d/56-t2a-disk-moe.json",
            58: "docs/research/registry.d/58-t2c-cold-osd-9m.json",
        }
        sources = {}
        for bundle, registry in pairs.items():
            verifier_path = f"scripts/verifiers.d/{bundle}_verifier.py"
            spec = importlib.util.spec_from_file_location(f"pv7_verify_{bundle}", ROOT / verifier_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            self.assertEqual((module.BUNDLE_ID, module.SPINE_DIGEST, module.DEPENDS_ON),
                             (bundle, claim_registry_sha256(), ()))
            for name in (registry, verifier_path):
                sources[name] = (ROOT / name).read_bytes()
        compiled = compile_fragments(ROOT, claim_registry_sha256(), sources)
        self.assertEqual(compiled["dependency_order"], [56, 58])
        self.assertEqual([row["bundle_id"] for row in compiled["verifiers"]], [56, 58])


if __name__ == "__main__":
    unittest.main()
