"""Wave-14B inference-time takedown firewall tests.

Runtime checks for the canonical byte map, the closed-loop UTF-8 gate,
the exact vault GEMV, EK5 cascade, and integer Ed25519 receipts.
"""

from __future__ import annotations

import copy
import sys
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

import chelated_takedown_firewall as fw  # noqa: E402
from ek5_v3_stratified import EdgeAttestation, StratifiedEvent  # noqa: E402
from scripts.verify_non_nda_ip_clean import (  # noqa: E402
    FORBIDDEN_METHOD_PATTERNS,
    scan_text_for_ip_leaks,
)


def _event(event_id: str) -> StratifiedEvent:
    return StratifiedEvent(
        event_id=event_id,
        entity="work",
        slot="body",
        claim_value="text",
        text=event_id,
        valid_time=1,
        tx_time=1,
        tenant="other",
        domain="general",
        authority_tier=1,
    )


def _edge(att_id: str, src: str, dst: str, edge_type: str) -> EdgeAttestation:
    return EdgeAttestation(
        attestation_id=att_id,
        src_id=src,
        dst_id=dst,
        edge_type=edge_type,
        valid_time=1,
        tx_time=1,
    )


def _push_words(gate: fw.ClosedLoopStreamGate, words: list[str]) -> bytes:
    out = bytearray()
    for word in words:
        out.extend(gate.push_chunk(word + " "))
    return bytes(out)


def _clear_scorer(_text: str) -> tuple[bool, float, None, int]:
    return (False, 0.0, None, 0)


LCS_WORDS = (
    "amber birch cedar drift ember flint grove heron ivory junco "
    "knoll lotus mango noble olive pearl quota ridge sable thyme "
    "umbra vigor whale xenon yeast zebra azure bloom coral daisy"
).split()


def _lcs_query() -> str:
    evens = LCS_WORDS[::2]
    return " ".join(evens)


class CanonicalAndScannerTests(unittest.TestCase):
    def test_stream_profiles_constant(self) -> None:
        self.assertEqual(fw.STREAM_PROFILES, ("profile_a_strict", "profile_b_armed_async"))

    def test_homoglyph_zwsp_and_separator_byte_spans(self) -> None:
        cyr = "\u0441"  # Cyrillic es, UTF-8 d1 81
        canon, spans = fw.canonicalize_for_tier2a("xx" + cyr)
        self.assertEqual(canon, "xxc")
        self.assertEqual(spans[2], (2, 4))

        zw, zw_spans = fw.canonicalize_for_tier2a("c\u200ball me")
        self.assertEqual(zw, "call me")
        self.assertEqual(zw_spans[1], (4, 5))

        soft, soft_spans = fw.canonicalize_for_tier2a("c\u00adall me")
        self.assertEqual(soft, "call me")
        self.assertEqual(soft_spans[1], (3, 4))

        folded, fold_spans = fw.canonicalize_for_tier2a("c-a-l-l")
        self.assertEqual(folded, "call")
        self.assertEqual(fold_spans, [(0, 1), (2, 3), (4, 5), (6, 7)])

        dotted, _dotted_spans = fw.canonicalize_for_tier2a("c.a.l.l m.e")
        self.assertEqual(dotted, "call me")

        prose, _prose_spans = fw.canonicalize_for_tier2a("Hello-World")
        self.assertEqual(prose, "hello world")

        wide, wide_spans = fw.canonicalize_for_tier2a("\uff43all")
        self.assertEqual(wide, "call")
        self.assertEqual(wide_spans[0], (0, 3))

        lig, lig_spans = fw.canonicalize_for_tier2a("\ufb01le")
        self.assertEqual(lig, "file")
        self.assertEqual(lig_spans[0], (0, 3))
        self.assertEqual(lig_spans[1], (0, 3))

        base, base_spans = fw.canonicalize_for_tier2a("e\u0301")
        self.assertEqual(base, "e")
        self.assertEqual(base_spans, [(0, 1)])

        precomposed, pre_spans = fw.canonicalize_for_tier2a("\u00e9")
        self.assertEqual(precomposed, "\u00e9")
        self.assertEqual(pre_spans, [(0, 2)])

        empty_canon, empty_spans = fw.canonicalize_for_tier2a("")
        self.assertEqual(empty_canon, "")
        self.assertEqual(empty_spans, [])
        self.assertEqual(len(zw), len(zw_spans))

    def test_tier2a_char_ngram_raw_byte_offset(self) -> None:
        phrase = "please \u0441all me tonight now"
        prefix = "HEAD "
        scanner = fw.Tier2ACanonicalScanner([("work-call", "please call me tonight now")])
        hit = scanner.scan_text(prefix + phrase)
        self.assertTrue(hit["flagged"])
        self.assertEqual(hit["match_kind"], "char_ngram")
        self.assertEqual(hit["matched_work_ids"], ["work-call"])
        self.assertEqual(hit["score"], 1.0)
        self.assertEqual(hit["flagged_raw_byte_start"], len(prefix.encode("utf-8")))
        self.assertGreater(hit["flagged_raw_byte_end"], hit["flagged_raw_byte_start"])
        raw = (prefix + phrase).encode("utf-8")
        self.assertLess(hit["flagged_raw_byte_end"], len(raw) + 1)

    def test_tier2a_lcs_paraphrase_without_contiguous_ngram(self) -> None:
        scanner = fw.Tier2ACanonicalScanner([("work-lcs", " ".join(LCS_WORDS))])
        query = "note: " + _lcs_query()
        hit = scanner.scan_text(query)
        self.assertTrue(hit["flagged"])
        self.assertEqual(hit["match_kind"], "lcs")
        self.assertIn("work-lcs", hit["matched_work_ids"])
        self.assertGreaterEqual(hit["score"], 0.72)
        self.assertIsNotNone(hit["flagged_raw_byte_start"])
        self.assertGreater(hit["flagged_raw_byte_end"], hit["flagged_raw_byte_start"])
        clean = scanner.scan_text("unrelated garden notes about weather")
        self.assertFalse(clean["flagged"])
        self.assertEqual(clean["match_kind"], "none")
        self.assertIsNone(clean["flagged_raw_byte_start"])


class VaultAndLineageTests(unittest.TestCase):
    def _handcrafted(self) -> fw.RestrictedVaultIndex:
        doc_mean = np.array([10.0, 0.0, 0.0, 0.0])
        nodes = np.array(
            [
                [10.0, 1.0, 0.0, 0.0],
                [10.0, 0.2, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [11.0, 0.05, 0.0, 0.0],
            ],
            dtype=np.float64,
        )
        norms = np.linalg.norm(nodes - doc_mean, axis=1)
        refs = np.array([[0.0, 0.0, 1.0, float(i + 1)] for i in range(64)])
        means = {
            "en": np.array([1.0, -1.0, 0.0, 0.0]),
            "es": np.zeros(4),
            "de": np.array([0.0, 0.0, 1.0, 0.0]),
        }
        return fw.RestrictedVaultIndex(
            ["A", "D", "B", "C"],
            ["work-a", "work-d", "work-b", "work-c"],
            nodes,
            doc_mean=doc_mean,
            doc_centered_norms=norms,
            doc_csls_r=np.array([0.0, 0.5, 0.0, 0.0]),
            ref_pool_vecs=refs,
            query_mean_by_lang=means,
            csls_k=16,
        )

    def test_vault_shape_and_disjoint_guards(self) -> None:
        index = self._handcrafted()
        with self.assertRaises(ValueError):
            fw.RestrictedVaultIndex(
                ["only"],
                ["w"],
                index.node_vecs,
                doc_mean=index.doc_mean,
                doc_centered_norms=index.doc_centered_norms[:1],
                doc_csls_r=index.doc_csls_r[:1],
                ref_pool_vecs=index.ref_pool_vecs,
                query_mean_by_lang=index.query_mean_by_lang,
            )
        with self.assertRaises(ValueError):
            fw.RestrictedVaultIndex(
                index.node_ids,
                index.work_ids,
                index.node_vecs,
                doc_mean=index.doc_mean,
                doc_centered_norms=index.doc_centered_norms,
                doc_csls_r=index.doc_csls_r,
                ref_pool_vecs=index.ref_pool_vecs[:63],
                query_mean_by_lang=index.query_mean_by_lang,
            )
        overlap = np.array(index.ref_pool_vecs, copy=True)
        overlap[0] = index.node_vecs[0]
        with self.assertRaises(AssertionError):
            fw.RestrictedVaultIndex(
                index.node_ids,
                index.work_ids,
                index.node_vecs,
                doc_mean=index.doc_mean,
                doc_centered_norms=index.doc_centered_norms,
                doc_csls_r=index.doc_csls_r,
                ref_pool_vecs=overlap,
                query_mean_by_lang=index.query_mean_by_lang,
            )
        with self.assertRaises(AssertionError):
            fw.RestrictedVaultIndex(
                index.node_ids,
                index.work_ids,
                index.node_vecs,
                doc_mean=index.doc_mean,
                doc_centered_norms=index.doc_centered_norms,
                doc_csls_r=index.doc_csls_r,
                ref_pool_vecs=index.ref_pool_vecs,
                query_mean_by_lang=index.query_mean_by_lang,
                metadata={"node_ids": ["shared"], "ref_ids": ["shared", "other"]},
            )

    def test_vault_zeromu_raw_cosine_language_and_mask(self) -> None:
        index = self._handcrafted()
        query = np.array([1.0, 0.0, 0.0, 0.0])
        zero = index.score_window_vec(query, lang="en", mode="zeromu_csls")
        raw = index.score_window_vec(query, lang="en", mode="raw_cosine")
        spanish = index.score_window_vec(query, lang="es", mode="zeromu_csls")
        self.assertEqual(zero["best_node_id"], "A")
        self.assertEqual(zero["best_work_id"], "work-a")
        self.assertEqual(raw["best_node_id"], "B")
        self.assertNotEqual(zero["best_node_id"], raw["best_node_id"])
        self.assertEqual(spanish["best_node_id"], "C")
        self.assertNotEqual(zero["best_node_id"], spanish["best_node_id"])
        masked = index.score_window_vec(
            query,
            lang="en",
            mode="zeromu_csls",
            active_node_mask=np.array([False, True, True, True]),
        )
        self.assertEqual(masked["best_node_id"], "D")
        empty = index.score_window_vec(
            query,
            lang="en",
            mode="zeromu_csls",
            active_node_mask=np.zeros(4, dtype=bool),
        )
        self.assertEqual(empty["max_score"], -1.0)
        self.assertIsNone(empty["best_node_id"])
        self.assertIsInstance(zero["gemv_vault_us"], int)
        self.assertIsInstance(zero["gemv_ref_pool_us"], int)
        self.assertGreaterEqual(zero["gemv_vault_us"], 0)
        self.assertGreaterEqual(zero["gemv_ref_pool_us"], 0)
        self.assertEqual(raw["gemv_ref_pool_us"], 0)
        german = index.score_window_vec(query, lang="de", mode="zeromu_csls")
        self.assertIn(german["best_node_id"], {"A", "D", "B", "C"})

    def test_vault_linear_bridge_384_and_native_4096(self) -> None:
        rng = np.random.default_rng(7)
        dim_lo = 384
        nodes = rng.normal(size=(6, dim_lo))
        target = np.zeros(dim_lo)
        target[0] = 1.0
        target[3] = 0.25
        nodes[0] = target
        refs = rng.normal(size=(64, dim_lo)) + 4.0
        doc_mean = np.zeros(dim_lo)
        norms = np.linalg.norm(nodes - doc_mean, axis=1)
        means = {lang: np.zeros(dim_lo) for lang in ("en", "es", "de")}
        index = fw.RestrictedVaultIndex(
            [f"n{i}" for i in range(6)],
            [f"w{i}" for i in range(6)],
            nodes,
            doc_mean=doc_mean,
            doc_centered_norms=norms,
            doc_csls_r=np.zeros(6),
            ref_pool_vecs=refs,
            query_mean_by_lang=means,
        )
        direct = index.score_window_vec(target, lang="en", mode="zeromu_csls")
        q_hi = np.zeros(4096)
        q_hi[7] = 1.0
        weight = np.zeros((4096, dim_lo))
        weight[7] = target
        bridged = index.score_window_vec(
            q_hi,
            lang="en",
            mode="linear_bridge_zeromu_csls",
            bridge_w=weight,
        )
        flipped = index.score_window_vec(
            q_hi,
            lang="en",
            mode="linear_bridge_zeromu_csls",
            bridge_w=weight.T,
        )
        self.assertEqual(direct["best_node_id"], "n0")
        self.assertEqual(bridged["best_node_id"], direct["best_node_id"])
        self.assertEqual(flipped["best_node_id"], direct["best_node_id"])
        self.assertAlmostEqual(bridged["max_score"], direct["max_score"], places=6)
        self.assertAlmostEqual(flipped["max_score"], direct["max_score"], places=6)
        self.assertIsInstance(bridged["gemv_vault_us"], int)
        self.assertIsInstance(bridged["gemv_ref_pool_us"], int)

        dim_hi = 4096
        native_nodes = rng.normal(size=(4, dim_hi))
        native_q = native_nodes[2].copy()
        native_refs = rng.normal(size=(64, dim_hi)) + 5.0
        native = fw.RestrictedVaultIndex(
            ["h0", "h1", "h2", "h3"],
            ["hw0", "hw1", "hw2", "hw3"],
            native_nodes,
            doc_mean=np.zeros(dim_hi),
            doc_centered_norms=np.linalg.norm(native_nodes, axis=1),
            doc_csls_r=np.zeros(4),
            ref_pool_vecs=native_refs,
            query_mean_by_lang={lang: np.zeros(dim_hi) for lang in ("en", "es", "de")},
        )
        native_hit = native.score_window_vec(native_q, lang="en", mode="zeromu_csls")
        native_raw = native.score_window_vec(native_q, lang="en", mode="raw_cosine")
        self.assertEqual(native_hit["best_node_id"], "h2")
        self.assertEqual(native_raw["best_node_id"], "h2")
        self.assertEqual(native_raw["gemv_ref_pool_us"], 0)
        self.assertGreaterEqual(native_hit["gemv_ref_pool_us"], 0)

    def test_lineage_copy_derives_cascade_preserves_cites_audit(self) -> None:
        fence = fw.TakedownLineageFence()
        events = [_event(name) for name in ("root", "copy", "deriv", "citer")]
        edges = [
            _edge("e-copy", "copy", "root", "COPY_OF"),
            _edge("e-deriv", "deriv", "root", "DERIVES_CURRENT"),
            _edge("e-cite", "citer", "root", "CITES_AUDIT"),
        ]
        ingested = fence.ingest_corpus_records(events, edges)
        self.assertEqual(ingested["cross_check"]["mismatches"], 0)
        self.assertEqual(fence.store.events["root"].tenant, "tenant_pm6")
        report = fence.issue_takedown("notice-1", ["root"], authority_tier=5, domain="ip_governance")
        self.assertEqual(report["cross_check"]["mismatches"], 0)
        self.assertTrue(report["auditable_suppression_not_erasure"])
        self.assertIn("root", report["refuted_ids"])
        self.assertEqual(set(report["cascaded_ids"]), {"copy", "deriv"})
        self.assertIn("citer", report["cites_audit_preserved"])
        eligibility = report["eligibility"]
        self.assertEqual(eligibility["root"], "AUDIT_ONLY")
        self.assertEqual(eligibility["copy"], "AUDIT_ONLY")
        self.assertEqual(eligibility["deriv"], "AUDIT_ONLY")
        self.assertEqual(eligibility["citer"], "ADMIT")
        self.assertEqual(eligibility["notice-1"], "ADMIT")
        self.assertEqual(fence.store.cites_audit["citer"], {"root"})
        general = fence.admissible_ids(audience="general")
        self.assertNotIn("root", general)
        self.assertNotIn("copy", general)
        self.assertNotIn("deriv", general)
        self.assertIn("citer", general)
        auditor = fence.admissible_ids(audience="auditor")
        self.assertIn("root", auditor)
        self.assertIn("copy", auditor)
        self.assertTrue(auditor.auditable_suppression_not_erasure)

    def test_screen_prompt_arm_and_hard_refuse(self) -> None:
        fence = fw.TakedownLineageFence()
        entities = {"work-1": "please call me tonight now"}
        blocked = fence.screen_prompt("Please call me tonight now", entities)
        self.assertTrue(blocked["hard_blocked"])
        self.assertTrue(blocked["armed"])
        self.assertEqual(blocked["matched_work_ids"], ["work-1"])
        self.assertIn("restricted_span_overlap", blocked["reason"])
        armed = fence.screen_prompt("hello there", entities, prompt_prox_score=0.5, tau_arm=0.35)
        self.assertFalse(armed["hard_blocked"])
        self.assertTrue(armed["armed"])
        self.assertEqual(armed["reason"], "armed")
        refused = fence.screen_prompt(
            "hello there",
            entities,
            prompt_prox_score=0.91,
            tau_arm=0.35,
            hard_refuse_threshold=0.8,
        )
        self.assertTrue(refused["hard_blocked"])
        self.assertIn("prompt_similarity_hard_refuse", refused["reason"])
        clear = fence.screen_prompt("hello there", entities, prompt_prox_score=0.1, tau_arm=0.35)
        self.assertFalse(clear["hard_blocked"])
        self.assertFalse(clear["armed"])
        self.assertEqual(clear["reason"], "clear")


class StreamGateTests(unittest.TestCase):
    def test_profile_a_strict_500ms_every_stride_offset(self) -> None:
        for offset in range(16):
            calls = {"n": 0}

            def scorer(text: str, _calls: dict[str, int] = calls) -> tuple[bool, float, str, int]:
                _calls["n"] += 1
                time.sleep(0.5)
                self.assertEqual(gate.released_utf8_bytes, 0)
                self.assertEqual(gate.flagged_window_bytes_leaked, 0)
                self.assertFalse(gate.aborted)
                self.assertIn("NEEDLETOKEN", text)
                return (True, 0.91, "work-z", 910000)

            gate = fw.ClosedLoopStreamGate("profile_a_strict", tier2b_scorer=scorer)
            words = ["NEEDLETOKEN" if i == offset else f"t{i:02d}" for i in range(64)]
            out = _push_words(gate, words)
            summary = gate.summary()
            self.assertEqual(out, b"", msg=f"offset {offset} emitted bytes")
            self.assertEqual(summary["flagged_window_bytes_leaked"], 0)
            self.assertEqual(summary["released_utf8_bytes"], 0)
            self.assertTrue(summary["aborted"])
            self.assertGreater(summary["suppressed_utf8_bytes"], 0)
            self.assertEqual(summary["matched_work_id"], "work-z")
            self.assertGreaterEqual(summary["max_score_ppm"], 910000)
            self.assertEqual(calls["n"], 1)
            self.assertEqual(gate.push_chunk("AFTER"), b"")

    def test_profile_a_clear_releases_only_closed_prefix(self) -> None:
        gate = fw.ClosedLoopStreamGate("profile_a_strict", tier2b_scorer=_clear_scorer)
        words = [f"w{i:02d}" for i in range(64)]
        out = _push_words(gate, words)
        text = out.decode("utf-8")
        self.assertIn("w00", text)
        self.assertIn("w15", text)
        self.assertNotIn("w16", text)
        self.assertNotIn("w63", text)
        self.assertEqual(gate.summary()["flagged_window_bytes_leaked"], 0)
        self.assertFalse(gate.summary()["aborted"])
        self.assertGreater(gate.summary()["first_byte_ttft_us"], 0)

    def test_profile_a_short_clear_eos_releases_all(self) -> None:
        gate = fw.ClosedLoopStreamGate("profile_a_strict", tier2b_scorer=_clear_scorer)
        words = [f"c{i:02d}" for i in range(20)]
        mid = _push_words(gate, words)
        self.assertEqual(mid, b"")
        final = gate.finalize_eos()
        self.assertEqual(final.decode("utf-8"), " ".join(words) + " ")
        self.assertEqual(gate.summary()["flagged_window_bytes_leaked"], 0)
        self.assertFalse(gate.aborted)

    def test_utf8_split_mid_codepoint_roundtrip(self) -> None:
        text = "caf\u00e9 se\u00f1or \U0001f600 gr\u00fcn"
        raw = text.encode("utf-8")
        gate = fw.ClosedLoopStreamGate("profile_a_strict", tier2b_scorer=_clear_scorer)
        out = bytearray()
        for index in range(len(raw)):
            piece = raw[index : index + 1].decode("utf-8", errors="surrogateescape")
            out.extend(gate.push_chunk(piece))
        self.assertEqual(bytes(out), b"")
        final = gate.finalize_eos()
        self.assertEqual(final, raw)
        self.assertEqual(final.decode("utf-8"), text)
        self.assertEqual(gate.summary()["flagged_window_bytes_leaked"], 0)

    def test_utf8_split_does_not_leak_flagged_bytes(self) -> None:
        text = "caf\u00e9 se\u00f1or \U0001f600 gr\u00fcn"
        raw = text.encode("utf-8")
        scanner = fw.Tier2ACanonicalScanner([("work-utf", text)])

        def scorer(_text: str) -> tuple[bool, float, str, int]:
            return (True, 0.8, "work-utf", 800000)

        gate = fw.ClosedLoopStreamGate(
            "profile_a_strict",
            tier2a=scanner,
            tier2b_scorer=scorer,
        )
        out = bytearray()
        for index in range(len(raw)):
            piece = raw[index : index + 1].decode("utf-8", errors="surrogateescape")
            out.extend(gate.push_chunk(piece))
        out.extend(gate.finalize_eos())
        self.assertEqual(bytes(out), b"")
        summary = gate.summary()
        self.assertEqual(summary["released_utf8_bytes"], 0)
        self.assertEqual(summary["flagged_window_bytes_leaked"], 0)
        self.assertTrue(summary["aborted"])
        self.assertNotIn("\U0001f600".encode("utf-8"), bytes(out))

    def test_short_eos_paraphrase_emits_nothing(self) -> None:
        scanner = fw.Tier2ACanonicalScanner([("work-lcs", " ".join(LCS_WORDS))])
        gate = fw.ClosedLoopStreamGate("profile_a_strict", tier2a=scanner, tier2b_scorer=_clear_scorer)
        query = _lcs_query()
        self.assertLess(len(query.split()), 64)
        first = gate.push_chunk(query)
        self.assertEqual(first, b"")
        final = gate.finalize_eos()
        self.assertEqual(final, b"")
        summary = gate.summary()
        self.assertEqual(summary["released_utf8_bytes"], 0)
        self.assertEqual(summary["flagged_window_bytes_leaked"], 0)
        self.assertTrue(summary["aborted"])
        self.assertEqual(summary["matched_work_id"], "work-lcs")
        self.assertGreater(summary["suppressed_utf8_bytes"], 0)

    def test_abort_flushes_zero_unverified_bytes(self) -> None:
        gate = fw.ClosedLoopStreamGate("profile_a_strict", tier2b_scorer=_clear_scorer)
        out = _push_words(gate, ["held", "token", "buffer"])
        self.assertEqual(out, b"")
        gate.abort_on_error_or_cancel()
        summary = gate.summary()
        self.assertTrue(summary["aborted"])
        self.assertEqual(summary["released_utf8_bytes"], 0)
        self.assertEqual(summary["flagged_window_bytes_leaked"], 0)
        self.assertGreater(summary["suppressed_utf8_bytes"], 0)
        self.assertEqual(gate.push_chunk("more unverified text "), b"")
        self.assertEqual(gate.finalize_eos(), b"")
        self.assertEqual(gate.summary()["released_utf8_bytes"], 0)

    def test_sse_sidechannel_strip_and_strict_reject(self) -> None:
        event = {
            "id": "evt-1",
            "choices": [
                {
                    "delta": {
                        "content": "hello",
                        "logprobs": {"token": "x"},
                        "tool_calls": [{"id": "call-1"}],
                        "reasoning_content": "hidden-trace",
                    },
                    "message": {
                        "content": "hello",
                        "thinking": "hidden-thought",
                        "function_call": {"name": "fn"},
                        "top_logprobs": [{"tok": "a"}],
                    },
                }
            ],
        }
        original = copy.deepcopy(event)
        with self.assertRaises(ValueError):
            fw.sanitize_sse_event_dict(event, strict_reject_sidechannels=True)
        self.assertEqual(event, original)
        cleaned = fw.sanitize_sse_event_dict(event, strict_reject_sidechannels=False)
        delta = cleaned["choices"][0]["delta"]
        message = cleaned["choices"][0]["message"]
        self.assertEqual(delta["content"], "hello")
        self.assertEqual(message["content"], "hello")
        for key in ("logprobs", "tool_calls", "reasoning_content", "thinking", "function_call", "top_logprobs"):
            self.assertNotIn(key, delta)
            self.assertNotIn(key, message)
        self.assertEqual(event, original)
        empty = fw.sanitize_sse_event_dict(
            {"logprobs": None, "thinking": "", "tool_calls": [], "content": "ok"},
            strict_reject_sidechannels=True,
        )
        self.assertEqual(empty, {"content": "ok"})
        gate = fw.ClosedLoopStreamGate("profile_a_strict", tier2b_scorer=_clear_scorer)
        caller = {"choices": [{"delta": {"content": "x", "logprobs": {"t": 1}}}]}
        before = copy.deepcopy(caller)
        gate.push_chunk("w00 ", sse_meta=caller)
        self.assertEqual(caller, before)
        self.assertNotIn("logprobs", gate.last_sse_event["choices"][0]["delta"])
        self.assertEqual(gate.last_sse_event["choices"][0]["delta"]["content"], "x")

    def test_profile_b_unarmed_exposure_bound(self) -> None:
        latency = 4
        hold = 12
        width = 64
        stride = 16
        bound = max(0, width + stride - 2 + latency - hold)

        def scorer(text: str) -> tuple[bool, float, str | None, int]:
            if "MARKEDTOKEN" in text:
                return (True, 0.77, "work-b", 770000)
            return (False, 0.01, None, 100)

        gate = fw.ClosedLoopStreamGate(
            "profile_b_armed_async",
            tier2b_scorer=scorer,
            tier2b_latency_tokens=latency,
            h0_holdback_tokens=hold,
            w_2b_tokens=width,
            s_2b_tokens=stride,
            armed=False,
        )
        words = ["MARKEDTOKEN" if i == 70 else f"w{i:03d}" for i in range(84)]
        out = _push_words(gate, words)
        out += gate.push_chunk("TAILUNIQUE ")
        summary = gate.summary()
        self.assertTrue(summary["aborted"])
        self.assertGreater(summary["unarmed_exposed_tokens"], 0)
        self.assertLessEqual(summary["unarmed_exposed_tokens"], bound)
        self.assertIn(b"MARKEDTOKEN", out)
        self.assertNotIn(b"TAILUNIQUE", out)
        self.assertIn(b"w000", out)
        self.assertGreater(summary["suppressed_utf8_bytes"], 0)

    def test_profile_b_arming_stops_leak(self) -> None:
        def scorer(text: str) -> tuple[bool, float, str | None, int]:
            if "MARKEDTOKEN" in text:
                return (True, 0.66, "work-arm", 660000)
            return (False, 0.0, None, 0)

        gate = fw.ClosedLoopStreamGate(
            "profile_b_armed_async",
            tier2b_scorer=scorer,
            tier2b_latency_tokens=4,
            h0_holdback_tokens=12,
            armed=False,
        )
        out = _push_words(gate, [f"b{i:03d}" for i in range(20)])
        self.assertIn(b"b000", out)
        gate.escalate_to_armed()
        self.assertTrue(gate.armed)
        rest = ["MARKEDTOKEN" if i == 70 else f"b{i:03d}" for i in range(20, 80)]
        out += _push_words(gate, rest)
        out += gate.push_chunk("TAILUNIQUE ")
        summary = gate.summary()
        self.assertTrue(summary["aborted"])
        self.assertTrue(summary["armed"])
        self.assertEqual(summary["flagged_window_bytes_leaked"], 0)
        self.assertIn(b"b000", out)
        self.assertNotIn(b"MARKEDTOKEN", out)
        self.assertNotIn(b"TAILUNIQUE", out)

    def test_profile_b_tier2a_ngram_leak_is_zero(self) -> None:
        phrase = "please call me tonight now"
        scanner = fw.Tier2ACanonicalScanner([("work-phrase", phrase)])
        gate = fw.ClosedLoopStreamGate(
            "profile_b_armed_async",
            tier2a=scanner,
            tier2b_scorer=_clear_scorer,
            h0_holdback_tokens=12,
            armed=False,
        )
        out = _push_words(gate, [f"ok{i:03d}" for i in range(30)])
        out += gate.push_chunk(phrase + " ")
        out += gate.push_chunk("TAILUNIQUE ")
        summary = gate.summary()
        self.assertTrue(summary["aborted"])
        self.assertEqual(summary["flagged_window_bytes_leaked"], 0)
        self.assertEqual(summary["matched_work_id"], "work-phrase")
        self.assertIn(b"ok000", out)
        self.assertNotIn(b"tonight", out)
        self.assertNotIn(b"TAILUNIQUE", out)


class ReceiptTests(unittest.TestCase):
    def test_receipt_ed25519_chain_tamper_and_non_nda(self) -> None:
        clock = {"t": 1_700_000_000_000_000}

        def tick() -> int:
            clock["t"] += 1
            return clock["t"]

        ledger = fw.TakedownReceiptLedger(clock_us=tick)
        first = ledger.issue_receipt(
            mode="non_nda",
            notice_id="notice-1",
            score_ppm=910000,
            threshold_ppm=350000,
            released_bytes=0,
            suppressed_bytes=128,
            latency_us=500000,
            records_after={"root": "AUDIT_ONLY", "citer": "ADMIT"},
            matched_work_id="work-1",
        )
        second = ledger.issue_receipt(
            mode="non_nda",
            notice_id="notice-2",
            score_ppm=1000,
            threshold_ppm=350000,
            released_bytes=12,
            suppressed_bytes=4,
            latency_us=20,
            records_after={"root": "Withdrawn", "citer": "Current"},
            matched_work_id="work-2",
        )
        self.assertEqual(second["payload"]["prev_hash"], first["entry_hash"])
        self.assertEqual(second["payload"]["seq"], 2)
        self.assertTrue(ledger.verify_ledger()["intact"])
        self.assertTrue(ledger.verify_receipt(first)["valid"])
        payload_text = fw.canonical_json(first["payload"]).decode("ascii")
        leaks = scan_text_for_ip_leaks(payload_text, source_label="receipt")
        self.assertEqual(leaks, [])
        self.assertEqual(len(FORBIDDEN_METHOD_PATTERNS), 56)
        self.assertIn("Inference-Time Takedown Suppression", payload_text)
        self.assertIn("Withdrawn", payload_text)
        self.assertIn("Current", payload_text)
        self.assertNotIn("AUDIT_ONLY", payload_text)
        self.assertNotIn(".", payload_text.split("score_ppm")[1][:12])

        tampered = copy.deepcopy(first)
        tampered["payload"]["score_ppm"] = 1
        self.assertFalse(ledger.verify_receipt(tampered)["valid"])
        bad_sig = copy.deepcopy(second)
        raw_sig = bad_sig["signature_b64"]
        bad_sig["signature_b64"] = ("A" if raw_sig[:1] != "A" else "B") + raw_sig[1:]
        self.assertFalse(ledger.verify_receipt(bad_sig)["valid"])
        ledger.ledger[1] = tampered
        broken = ledger.verify_ledger()
        self.assertFalse(broken["intact"])

        with self.assertRaises(ValueError):
            ledger.issue_receipt(
                mode="non_nda",
                notice_id="notice-3",
                score_ppm=1.5,  # type: ignore[arg-type]
                threshold_ppm=1,
                released_bytes=0,
                suppressed_bytes=0,
                latency_us=0,
                records_after={"root": "Current"},
            )

        nda = fw.TakedownReceiptLedger(clock_us=tick)
        nda_entry = nda.issue_receipt(
            mode="nda",
            notice_id="notice-nda",
            score_ppm=2,
            threshold_ppm=2,
            released_bytes=0,
            suppressed_bytes=1,
            latency_us=3,
            records_after={"root": "AUDIT_ONLY"},
        )
        self.assertEqual(nda_entry["payload"]["records_after"]["root"], "Withdrawn")
        self.assertEqual(nda_entry["payload"]["internal"]["eligibility_raw"]["root"], "AUDIT_ONLY")
        self.assertTrue(nda.verify_receipt(nda_entry)["valid"])
        fw.canonical_json(nda_entry["payload"])


if __name__ == "__main__":
    unittest.main()
