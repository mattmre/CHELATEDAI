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
from scripts.run_pv7_w19_2a_disk_moe import (  # noqa: E402
    checked_lease,
    prepare_cache_cell,
    summarize_cold_verification,
    verify_cold_cache,
)
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


class ColdCacheVerificationTests(unittest.TestCase):
    class FakePager:
        def __init__(self):
            self.resets = []

        def reset(self, cold):
            self.resets.append(cold)

    def test_verify_cold_cache_rejects_resident_or_empty_residency(self):
        receipt = verify_cold_cache(object(), 4, residency=lambda pager: (12, 0))
        self.assertEqual(receipt, {"rung": 4, "method": "mincore", "expert_pages_checked": 12,
                                   "expert_pages_resident": 0, "verified_cold": True})
        with self.assertRaisesRegex(RuntimeError, "R5 cold-run expert cache still has 3 of 12"):
            verify_cold_cache(object(), 5, residency=lambda pager: (12, 3))
        with self.assertRaisesRegex(RuntimeError, "R6 cold verification covered no expert pages"):
            verify_cold_cache(object(), 6, residency=lambda pager: (0, 0))

    def test_every_rung_runs_the_mincore_check_before_each_cold_cell(self):
        for rung in range(1, 7):
            with self.subTest(rung=rung):
                pager = self.FakePager()
                probes = []

                def residency(target, probes=probes, pager=pager):
                    # The cold reset must already have happened when residency is read.
                    probes.append(list(pager.resets))
                    return 8, 0

                cold = prepare_cache_cell(pager, rung, "cold", residency=residency)
                warm = prepare_cache_cell(pager, rung, "warm", residency=residency)
                self.assertEqual(pager.resets, [True, False])
                self.assertEqual(probes, [[True]])
                self.assertEqual(cold["rung"], rung)
                self.assertTrue(cold["verified_cold"])
                self.assertIsNone(warm)
                with self.assertRaises(RuntimeError):
                    prepare_cache_cell(self.FakePager(), rung, "cold", residency=lambda target: (8, 1))

    def test_rung_summary_requires_every_cold_cell_verified(self):
        verified = {"verified_cold": True}
        cells = [{"cache": "cold", "cold_cache_verification": verified},
                 {"cache": "warm", "cold_cache_verification": None},
                 {"cache": "cold", "cold_cache_verification": verified}]
        self.assertEqual(summarize_cold_verification(3, cells), {
            "rung": 3, "method": "mincore", "cold_cells": 2, "verified_cold_cells": 2,
            "all_cold_cells_verified": True,
        })
        cells[2] = {"cache": "cold"}
        self.assertFalse(summarize_cold_verification(3, cells)["all_cold_cells_verified"])
        self.assertFalse(summarize_cold_verification(3, [{"cache": "warm"}])["all_cold_cells_verified"])


class ColdOSDTests(unittest.TestCase):
    def fixture(self, directory):
        vectors = np.array([[1, 0.1], [1, 0.2], [0.1, 1], [0.2, 1], [0.7, 0.7]], dtype=np.float32)
        queries = np.eye(2, dtype=np.float32)
        return osd.build_extents(osd.VectorCorpus([vectors]), queries, queries, directory, 2 * osd.MIB, csls_k=2)

    def test_fence_wal_is_fsynced_counted_and_replays_after_a_crash(self):
        with tempfile.TemporaryDirectory(prefix=".pv7-58-test-", dir=ROOT) as directory:
            geom = self.fixture(directory)
            wal = Path(directory) / "fence.wal"
            index = osd.ColdOSD(geom, 2, wal_path=wal)
            header = len(osd.FENCE_WAL_MAGIC)
            self.assertEqual(index.written, header)
            with patch.object(osd.os, "fsync", wraps=os.fsync) as fsync:
                index.withdraw([3, 0])
            self.assertEqual(fsync.call_count, 1)
            withdraw_bytes = 17 + 4 * 2
            self.assertEqual(index.written, header + withdraw_bytes)
            self.assertEqual(wal.stat().st_size, index.written)
            index.reinstate([0, 3])
            index.withdraw([1])
            self.assertEqual(index.written, header + withdraw_bytes + 17 + 21)
            self.assertEqual(wal.stat().st_size, index.fence.wal_bytes)

            # Simulated crash: a fresh process replays the WAL with the trace still open.
            recovered = osd.ColdOSD.recover(geom, 2, wal)
            self.assertEqual(recovered.fence.recovery, {
                "records_replayed": 3, "epoch": 3, "torn_tail_bytes": 0,
                "active_trace_open": True, "durable_bytes": wal.stat().st_size,
            })
            self.assertTrue(np.array_equal(recovered.fence.bits, index.fence.bits))
            self.assertEqual(recovered.visible([0, 1, 3]), [True, False, True])
            self.assertEqual(recovered.audit_digest(), index.audit_digest())
            self.assertEqual(recovered.written, 0)
            recovered.reinstate([1])
            self.assertEqual(recovered.written, 17)
            self.assertEqual(osd.EK5ExtentFence.recover_from_wal(geom, wal).recovery["epoch"], 4)
            with self.assertRaises(FileExistsError):
                osd.ColdOSD(geom, 2, wal_path=wal)

    def test_fence_wal_truncates_a_torn_tail_and_rejects_interior_damage(self):
        with tempfile.TemporaryDirectory(prefix=".pv7-58-test-", dir=ROOT) as directory:
            geom = self.fixture(directory)
            wal = Path(directory) / "fence.wal"
            index = osd.ColdOSD(geom, 2, wal_path=wal)
            index.withdraw([2])
            durable = wal.stat().st_size
            for torn in (b"R", osd._fence_wal_record(b"R", 2)[:-1],
                         osd._fence_wal_record(b"R", 2)[:-1] + b"\x00"):
                with self.subTest(torn=torn):
                    with wal.open("ab") as handle:
                        handle.write(torn)
                    fence = osd.EK5ExtentFence.recover_from_wal(geom, wal)
                    self.assertEqual(fence.recovery["torn_tail_bytes"], len(torn))
                    self.assertEqual((fence.epoch, wal.stat().st_size), (1, durable))
                    self.assertEqual(fence.active_ids, {2})
                    self.assertFalse(fence.allowed([2])[0])
            # Continuing after a torn-tail recovery appends a clean, replayable epoch.
            fence = osd.EK5ExtentFence.recover_from_wal(geom, wal)
            fence.reinstate([2])
            self.assertEqual(osd.EK5ExtentFence.recover_from_wal(geom, wal).epoch, 2)

            raw = bytearray(wal.read_bytes())
            raw[len(osd.FENCE_WAL_MAGIC) + 1] ^= 0xFF
            wal.write_bytes(bytes(raw))
            with self.assertRaisesRegex(osd.FenceWALError, "fails its CRC before end of log"):
                osd.EK5ExtentFence.recover_from_wal(geom, wal)
            skipped = Path(directory) / "skipped.wal"
            skipped.write_bytes(osd.FENCE_WAL_MAGIC + osd._fence_wal_record(b"W", 2, [1]))
            with self.assertRaisesRegex(osd.FenceWALError, "epoch 2 does not follow epoch 0"):
                osd.EK5ExtentFence.recover_from_wal(geom, skipped)
            skipped.write_bytes(b"NOTAWAL!")
            with self.assertRaisesRegex(osd.FenceWALError, "Not an EK5 fence WAL"):
                osd.EK5ExtentFence.recover_from_wal(geom, skipped)

    def test_paired_bootstrap_lcb_gates_the_write_comparison(self):
        self.assertEqual(osd.paired_bootstrap_lcb95([40] * 30), 1200)
        mixed = [100] * 20 + [-150] * 10
        self.assertEqual(sum(mixed), 500)
        lcb = osd.paired_bootstrap_lcb95(mixed)
        self.assertLess(lcb, 0)
        self.assertEqual(lcb, osd.paired_bootstrap_lcb95(mixed))
        with self.assertRaisesRegex(ValueError, "at least two"):
            osd.paired_bootstrap_lcb95([5])
        with self.assertRaisesRegex(ValueError, "1000"):
            osd.paired_bootstrap_lcb95([5, 6], resamples=999)

        stages = ("withdraw_immediate", "withdraw_quiet", "reinstate_immediate", "reinstate_quiet")

        def arm(per_pair):
            checkpoint = {"recall_at_10": 1.0, "stale_resurrections": 0, "stale_rate": 0.0, "returned": 10}
            return {"total_write_bytes": sum(per_pair), "per_pair_total_write_bytes": list(per_pair),
                    "visibility_passed": True, "checkpoints": {stage: dict(checkpoint) for stage in stages}}

        # A positive total saving that rests on a minority of pairs must not pass.
        challenger = [200] * 30
        candidate = [100] * 20 + [350] * 10
        result = osd.compare_traces({"cold_osd": {"clustered": arm(candidate)},
                                     "other": {"clustered": arm(challenger)}})["other"]["clustered"]
        self.assertEqual(result["write_bytes_saved"], 500)
        self.assertLess(result["saved_bytes_lcb95"], 0)
        self.assertFalse(result["passed"])
        self.assertEqual(result["lcb_method"]["pairs"], 30)
        steady = osd.compare_traces({"cold_osd": {"clustered": arm([100] * 30)},
                                     "other": {"clustered": arm(challenger)}})["other"]["clustered"]
        self.assertEqual(steady["saved_bytes_lcb95"], 3000)
        self.assertTrue(steady["passed"])
        with self.assertRaisesRegex(ValueError, "30 paired traces"):
            osd.compare_traces({"cold_osd": {"clustered": arm([100] * 29)},
                                "other": {"clustered": arm([200] * 29)}})

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
            self.assertGreater(result["saved_bytes_lcb95"], 0)
            self.assertEqual(len(result["paired_write_bytes_saved"]), 30)
            # Chelated now pays for its fsynced fence WAL: one W and one R record per trace.
            cold = report["summaries"]["cold_osd"][layout]
            self.assertEqual(cold["per_pair_total_write_bytes"], [(17 + 4 * 16) + 17] * 30)
            self.assertGreater(cold["total_write_bytes"], 0)
        self.assertEqual(report["cold_osd_write_source"], "fsynced_ek5_fence_wal")
        self.assertEqual(report["fence_wal_records"], 120)
        self.assertEqual(report["fence_wal_bytes"], len(osd.FENCE_WAL_MAGIC) + 60 * (17 + 4 * 16 + 17))
        self.assertTrue(report["fence_wal_replay"]["bits_match"])
        self.assertTrue(report["fence_wal_replay"]["epoch_matches"])
        self.assertEqual(report["fence_wal_replay"]["records_replayed"], 120)
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
