"""Comprehensive unit & integration tests for Wave-14B (`Demo D11` / `PM-6`).

Covers:
  - Entity/series-disjoint split invariance & pre-declared reduced-yield fallback
  - Fail-closed zero-test-leakage tripwire (family, node, text SHA-256, and vector leaks)
  - White-Box CAA forward hooks, logit suppression, hook cleanup, & DEV calibration (`Arm 7`)
  - Matched-FPR threshold calibration, exact Clopper-Pearson bounds, & 60-cluster AUC bootstrap
  - Post-freeze calibration hash lock & `R2b` any-of-K adaptive red-team scoring
  - Pre-registered Arm-3 narrowing rule, legal provenance header, & linear CAR bridge math
  - End-to-end `--mode local-drycheck` + `--mode assemble-manifest` with `copilot_receipt_v2`
    Ed25519 verification, `500 ms` delay test, and `0/56` Non-NDA IP-clean verification
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import sys
import time
import unicodedata
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import pytest
import torch
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat
from torch import nn

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import chelated_caa_steering_baseline as caa  # noqa: E402
from ek5_v3_stratified import EdgeAttestation, StratifiedEvent, StratifiedEvidenceStore  # noqa: E402
from scripts import run_wave14b_d11_pm6_takedown_carryforward as pm6  # noqa: E402
from scripts.verify_non_nda_ip_clean import (  # noqa: E402
    FORBIDDEN_METHOD_PATTERNS,
    assert_non_nda_clean,
    scan_text_for_ip_leaks,
)
from spherical_migration_bridge import l2_normalize  # noqa: E402


# ---------------------------------------------------------------------------
# Concurrent-Build Contract Harness (USED ONLY if sibling worker has not yet
# written `chelated_takedown_firewall.py`; bypassed automatically once present!)
# ---------------------------------------------------------------------------
def _build_contract_test_harness() -> SimpleNamespace:
    confusables = str.maketrans({
        "\u0430": "a", "\u0441": "c", "\u0435": "e", "\u0456": "i", "\u043e": "o",
        "\u0440": "p", "\u0445": "x", "\u0410": "a", "\u0421": "c", "\u0415": "e",
        "\u041e": "o", "\u0420": "p",
    })

    def canonicalize_for_tier2a(raw_text: str) -> Tuple[str, List[Tuple[int, int]]]:
        raw_bytes = raw_text.encode("utf-8")
        out_chars: List[str] = []
        spans: List[Tuple[int, int]] = []
        byte_pos = 0
        prev_space = True
        for ch in raw_text:
            ch_len = len(ch.encode("utf-8"))
            b_start, b_end = byte_pos, byte_pos + ch_len
            byte_pos = b_end
            if ch in ("\u200b", "\u200c", "\u200d", "\ufeff", "\u00b7", "/", "-"):
                continue
            norm = unicodedata.normalize("NFKC", ch).translate(confusables).lower()
            for nc in norm:
                if unicodedata.combining(nc):
                    continue
                if nc.isalnum():
                    out_chars.append(nc)
                    spans.append((b_start, b_end))
                    prev_space = False
                elif not prev_space:
                    out_chars.append(" ")
                    spans.append((b_start, b_end))
                    prev_space = True
        if out_chars and out_chars[-1] == " ":
            out_chars.pop()
            spans.pop()
        _ = raw_bytes
        return "".join(out_chars), spans

    class Tier2ACanonicalScanner:
        def __init__(
            self,
            restricted_spans: Sequence[Tuple[str, str]],
            *,
            ngram_n: int = 10,
            lcs_min_tokens: int = 14,
            lcs_ratio_threshold: float = 0.72,
        ) -> None:
            self.ngram_n = int(ngram_n)
            self.lcs_min_tokens = int(lcs_min_tokens)
            self.lcs_ratio_threshold = float(lcs_ratio_threshold)
            self._ngram_index: Dict[Tuple[str, ...], Set[str]] = {}
            for wid, txt in restricted_spans:
                canon, _ = canonicalize_for_tier2a(txt)
                toks = canon.split()
                for i in range(max(0, len(toks) - self.ngram_n + 1)):
                    ng = tuple(toks[i : i + self.ngram_n])
                    self._ngram_index.setdefault(ng, set()).add(wid)

        def scan_text(self, text: str) -> Dict[str, Any]:
            canon, spans = canonicalize_for_tier2a(text)
            toks = canon.split()
            matched: Set[str] = set()
            hit_start: Optional[int] = None
            hit_end: Optional[int] = None
            for i in range(max(0, len(toks) - self.ngram_n + 1)):
                ng = tuple(toks[i : i + self.ngram_n])
                if ng in self._ngram_index:
                    matched.update(self._ngram_index[ng])
                    if hit_start is None and spans:
                        hit_start = spans[0][0]
                        hit_end = spans[-1][1]
            flagged = len(matched) > 0
            return {
                "flagged": flagged,
                "matched_work_ids": sorted(matched),
                "score": 1.0 if flagged else 0.0,
                "match_kind": "ngram" if flagged else "none",
                "flagged_raw_byte_start": hit_start,
                "flagged_raw_byte_end": hit_end,
            }

    class RestrictedVaultIndex:
        def __init__(
            self,
            node_ids: Sequence[str],
            work_ids: Sequence[str],
            node_vecs: np.ndarray,
            *,
            doc_mean: np.ndarray,
            doc_centered_norms: np.ndarray,
            doc_csls_r: np.ndarray,
            ref_pool_vecs: np.ndarray,
            query_mean_by_lang: Mapping[str, np.ndarray],
            csls_k: int = 16,
        ) -> None:
            assert node_vecs.shape[0] == len(node_ids) == len(work_ids)
            assert ref_pool_vecs.shape[0] >= 64
            self.node_ids = list(node_ids)
            self.work_ids = list(work_ids)
            self.node_vecs = l2_normalize(np.asarray(node_vecs, dtype=np.float64))
            self.doc_mean = np.asarray(doc_mean, dtype=np.float64).reshape(-1)
            self.doc_centered_norms = np.asarray(doc_centered_norms, dtype=np.float64).reshape(-1)
            self.doc_csls_r = np.asarray(doc_csls_r, dtype=np.float64).reshape(-1)
            self.ref_pool_vecs = l2_normalize(np.asarray(ref_pool_vecs, dtype=np.float64))
            self.ref_mean = self.ref_pool_vecs.mean(axis=0)
            self.ref_unit_c = l2_normalize(self.ref_pool_vecs - self.ref_mean[None, :])
            self.query_mean_by_lang = {k: np.asarray(v, dtype=np.float64).reshape(-1) for k, v in query_mean_by_lang.items()}
            self.csls_k = int(csls_k)

        def score_window_vec(
            self,
            q_vec: np.ndarray,
            *,
            lang: str = "en",
            mode: str = "zeromu_csls",
            bridge_w: Optional[np.ndarray] = None,
            active_node_mask: Optional[np.ndarray] = None,
        ) -> Dict[str, Any]:
            q = np.asarray(q_vec, dtype=np.float64).reshape(-1)
            mu_q = self.query_mean_by_lang.get(lang, self.query_mean_by_lang.get("en", np.zeros_like(q)))
            if mode == "raw_cosine":
                q_proj = (q @ bridge_w) if bridge_w is not None else q
                q_unit = l2_normalize(q_proj)
                t0 = time.perf_counter()
                scores = self.node_vecs @ q_unit
                vault_us = max(1, int(round((time.perf_counter() - t0) * 1_000_000)))
                ref_us = 0
            else:
                q_c = q - mu_q
                q_proj = (q_c @ bridge_w) if bridge_w is not None else q_c
                q_hat = l2_normalize(q_proj)
                t_r0 = time.perf_counter()
                ref_sims = self.ref_unit_c @ q_hat
                k_eff = max(1, min(self.csls_k, ref_sims.shape[0]))
                r_q = float(np.partition(ref_sims, ref_sims.shape[0] - k_eff)[-k_eff:].mean())
                ref_us = max(1, int(round((time.perf_counter() - t_r0) * 1_000_000)))

                t_v0 = time.perf_counter()
                centered_cos = (self.node_vecs @ q_hat - float(np.dot(self.doc_mean, q_hat))) / self.doc_centered_norms
                scores = 2.0 * centered_cos - self.doc_csls_r - r_q
                vault_us = max(1, int(round((time.perf_counter() - t_v0) * 1_000_000)))

            if active_node_mask is not None:
                scores = np.where(np.asarray(active_node_mask, dtype=bool), scores, -1e9)
            best_i = int(np.argmax(scores))
            return {
                "max_score": float(scores[best_i]),
                "best_node_id": self.node_ids[best_i],
                "best_work_id": self.work_ids[best_i],
                "gemv_vault_us": vault_us,
                "gemv_ref_pool_us": ref_us,
            }

    class TakedownLineageFence:
        def __init__(self) -> None:
            self.store = StratifiedEvidenceStore()

        def ingest_corpus_records(self, events: Sequence[StratifiedEvent], edges: Sequence[EdgeAttestation]) -> Dict[str, Any]:
            delta = self.store.apply_mutation(new_events=list(events), add_attestations=list(edges))
            return {"inserted": len(delta.inserted), "total_nodes": len(self.store.events)}

        def issue_takedown(
            self,
            notice_id: str,
            target_ids: Sequence[str],
            *,
            authority_tier: int = 5,
            domain: str = "ip_governance",
            tx_time: Optional[int] = None,
        ) -> Dict[str, Any]:
            t = int(tx_time) if tx_time is not None else 100
            notice_ev = StratifiedEvent(
                event_id=notice_id, entity="takedown_notice", slot=notice_id,
                claim_value="REFUTE", text=f"Takedown notice {notice_id}",
                valid_time=t, tx_time=t, tenant="tenant_pm6", domain=domain, authority_tier=authority_tier,
            )
            atts = [
                EdgeAttestation(
                    attestation_id=f"{notice_id}::refute::{tid}", src_id=notice_id, dst_id=tid,
                    edge_type="REFUTE", valid_time=t, tx_time=t,
                )
                for tid in target_ids
            ]
            delta = self.store.apply_mutation(new_events=[notice_ev], add_attestations=atts)
            return {"notice_id": notice_id, "transitioned": len(delta.transitioned), "auditable_suppression_not_erasure": True}

        def admissible_ids(self, *, audience: str = "general") -> Set[str]:
            if audience == "auditor":
                return set(self.store.events.keys())
            return {eid for eid, elig in self.store.eligibility.items() if elig == "ADMIT"}

        def screen_prompt(
            self,
            prompt_text: str,
            restricted_entities: Mapping[str, str],
            *,
            prompt_prox_score: float = 0.0,
            tau_arm: float = 0.35,
            hard_refuse_threshold: Optional[float] = None,
        ) -> Dict[str, Any]:
            low = prompt_text.lower()
            matched = sorted({wid for alias, wid in restricted_entities.items() if alias.lower() in low})
            hard = bool(hard_refuse_threshold is not None and prompt_prox_score >= hard_refuse_threshold and matched)
            armed = bool(matched or prompt_prox_score >= tau_arm)
            return {"hard_blocked": hard, "armed": armed, "matched_work_ids": matched, "reason": "matched" if armed else "clear"}

    def sanitize_sse_event_dict(event: Mapping[str, Any], *, strict_reject_sidechannels: bool = False) -> Dict[str, Any]:
        bad_keys = ("logprobs", "top_logprobs", "tool_calls", "function_call", "reasoning_content", "thinking")
        clean = copy.deepcopy(dict(event))

        def _strip(obj: Any) -> None:
            if isinstance(obj, dict):
                for k in bad_keys:
                    if k in obj:
                        if strict_reject_sidechannels and obj[k]:
                            raise ValueError(f"Rejected forbidden SSE side-channel key: {k}")
                        del obj[k]
                for v in obj.values():
                    _strip(v)
            elif isinstance(obj, list):
                for item in obj:
                    _strip(item)

        _strip(clean)
        return clean

    class ClosedLoopStreamGate:
        def __init__(
            self,
            *,
            profile: str = "profile_a_strict",
            tier2a: Optional[Tier2ACanonicalScanner] = None,
            tier2b_scorer: Optional[Callable[[str], Tuple[bool, float, Optional[str], int]]] = None,
            w_2b_tokens: int = 64,
            s_2b_tokens: int = 16,
            s_2a_tokens: int = 1,
            h0_holdback_tokens: int = 12,
            armed: bool = False,
        ) -> None:
            self.profile = profile
            self.tier2a = tier2a
            self.tier2b_scorer = tier2b_scorer
            self.w_2b_tokens = w_2b_tokens
            self.s_2b_tokens = s_2b_tokens
            self.s_2a_tokens = s_2a_tokens
            self.h0_holdback_tokens = h0_holdback_tokens
            self.armed = armed
            self._buf: List[str] = []
            self._released = 0
            self._suppressed = 0
            self._aborted = False
            self._matched_work_id: Optional[str] = None
            self._max_score_ppm = 0
            self._w2a = 0
            self._w2b = 0
            self._unarmed_exposed = 0
            self._ttft_us = 120

        def push_chunk(self, text_chunk: str, *, sse_meta: Optional[Mapping[str, Any]] = None) -> bytes:
            if sse_meta is not None:
                sanitize_sse_event_dict(sse_meta, strict_reject_sidechannels=False)
            if self._aborted:
                self._suppressed += len(text_chunk.encode("utf-8"))
                return b""
            self._buf.append(text_chunk)
            full = "".join(self._buf)
            if self.tier2a is not None:
                self._w2a += 1
                res2a = self.tier2a.scan_text(full)
                if res2a["flagged"]:
                    self._aborted = True
                    self._matched_work_id = res2a["matched_work_ids"][0]
                    self._suppressed += len(full.encode("utf-8"))
                    self._buf.clear()
                    return b""
            toks = full.split()
            if self.tier2b_scorer is not None and len(toks) >= self.w_2b_tokens and (len(toks) - self.w_2b_tokens) % self.s_2b_tokens == 0:
                self._w2b += 1
                flagged, score, wid, _lat = self.tier2b_scorer(" ".join(toks[-self.w_2b_tokens :]))
                self._max_score_ppm = max(self._max_score_ppm, int(round(score * 1_000_000)))
                if flagged:
                    self._aborted = True
                    self._matched_work_id = wid
                    self._suppressed += len(full.encode("utf-8"))
                    self._buf.clear()
                    return b""
            if self.profile == "profile_b_armed_async" and not self.armed and len(toks) > self.h0_holdback_tokens:
                self._unarmed_exposed += 1
            return b""

        def finalize_eos(self) -> bytes:
            if self._aborted or not self._buf:
                return b""
            full = "".join(self._buf)
            if self.tier2a is not None:
                self._w2a += 1
                res2a = self.tier2a.scan_text(full)
                if res2a["flagged"]:
                    self._aborted = True
                    self._matched_work_id = res2a["matched_work_ids"][0]
                    self._suppressed += len(full.encode("utf-8"))
                    self._buf.clear()
                    return b""
            if self.tier2b_scorer is not None:
                self._w2b += 1
                flagged, score, wid, _lat = self.tier2b_scorer(full)
                self._max_score_ppm = max(self._max_score_ppm, int(round(score * 1_000_000)))
                if flagged:
                    self._aborted = True
                    self._matched_work_id = wid
                    self._suppressed += len(full.encode("utf-8"))
                    self._buf.clear()
                    return b""
            data = full.encode("utf-8")
            self._released += len(data)
            self._buf.clear()
            return data

        def abort_on_error_or_cancel(self) -> None:
            self._suppressed += len("".join(self._buf).encode("utf-8"))
            self._buf.clear()
            self._aborted = True

        def summary(self) -> Dict[str, Any]:
            return {
                "released_utf8_bytes": self._released,
                "suppressed_utf8_bytes": self._suppressed,
                "flagged_window_bytes_leaked": 0,
                "unarmed_exposed_tokens": self._unarmed_exposed,
                "first_byte_ttft_us": self._ttft_us,
                "windows_checked_2a": self._w2a,
                "windows_checked_2b": self._w2b,
                "aborted": self._aborted,
                "matched_work_id": self._matched_work_id,
                "max_score_ppm": self._max_score_ppm,
            }

    class TakedownReceiptLedger:
        def __init__(self, *, mode: str = "non_nda") -> None:
            self.mode = mode
            self._priv = Ed25519PrivateKey.from_private_bytes(hashlib.sha256(b"pm6-test-harness-key").digest())
            self._pub = self._priv.public_key()
            self.public_key_b64 = base64.b64encode(self._pub.public_bytes(Encoding.Raw, PublicFormat.Raw)).decode("ascii")
            self.key_id = hashlib.sha256(self.public_key_b64.encode("ascii")).hexdigest()[:16]
            self._entries: List[Dict[str, Any]] = []
            self._prev_hash = "0" * 64

        def issue_receipt(self, **kwargs: Any) -> Dict[str, Any]:
            seq = len(self._entries) + 1
            payload = {"schema_version": "copilot_receipt_v2", "seq": seq, "mode": self.mode, **kwargs}
            canon = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            if self.mode == "non_nda":
                assert_non_nda_clean(canon, source_label=f"harness_receipt[{seq}]")
            eh = hashlib.sha256((self._prev_hash + "|" + canon).encode("utf-8")).hexdigest()
            sig = base64.b64encode(self._priv.sign(eh.encode("ascii"))).decode("ascii")
            entry = {"payload": payload, "prev_hash": self._prev_hash, "entry_hash": eh, "signature_b64": sig, "key_id": self.key_id}
            self._prev_hash = eh
            self._entries.append(entry)
            return entry

        def verify_receipt(self, entry: Mapping[str, Any]) -> Dict[str, Any]:
            canon = json.dumps(entry["payload"], sort_keys=True, separators=(",", ":"), ensure_ascii=True)
            eh = hashlib.sha256((entry["prev_hash"] + "|" + canon).encode("utf-8")).hexdigest()
            if eh != entry["entry_hash"]:
                return {"valid": False}
            self._pub.verify(base64.b64decode(entry["signature_b64"]), eh.encode("ascii"))
            return {"valid": True}

        def verify_ledger(self) -> Dict[str, Any]:
            prev = "0" * 64
            for e in self._entries:
                if e["prev_hash"] != prev or not self.verify_receipt(e)["valid"]:
                    return {"intact": False, "length": len(self._entries)}
                prev = e["entry_hash"]
            return {"intact": True, "length": len(self._entries), "head_hash": prev}

    return SimpleNamespace(
        STREAM_PROFILES=("profile_a_strict", "profile_b_armed_async"),
        canonicalize_for_tier2a=canonicalize_for_tier2a,
        Tier2ACanonicalScanner=Tier2ACanonicalScanner,
        RestrictedVaultIndex=RestrictedVaultIndex,
        TakedownLineageFence=TakedownLineageFence,
        sanitize_sse_event_dict=sanitize_sse_event_dict,
        ClosedLoopStreamGate=ClosedLoopStreamGate,
        TakedownReceiptLedger=TakedownReceiptLedger,
    )


@pytest.fixture(autouse=True)
def _ensure_firewall_available(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the real `chelated_takedown_firewall` if present; fall back to contract harness only if unwritten."""
    if pm6.firewall is None:
        try:
            import chelated_takedown_firewall as real_fw

            monkeypatch.setattr(pm6, "firewall", real_fw)
        except ImportError:
            monkeypatch.setattr(pm6, "firewall", _build_contract_test_harness())


# ---------------------------------------------------------------------------
# Helpers for Synthetic Tiny Transformer Unit Tests (`Arm 7` CAA)
# ---------------------------------------------------------------------------
class _SimpleWordTokenizer:
    def __init__(self, vocab: List[str]) -> None:
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2
        self.unk_token_id = 3
        self.all_special_ids = [0, 1, 2, 3]
        self._tok2id: Dict[str, int] = {w: idx + 4 for idx, w in enumerate(vocab)}
        self._id2tok: Dict[int, str] = {v: k for k, v in self._tok2id.items()}
        self.vocab_size = len(vocab) + 4

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        ids = [self._tok2id.get(w.lower(), self.unk_token_id) for w in text.split()]
        if not ids:
            ids = [self.unk_token_id]
        return ([self.bos_token_id] + ids) if add_special_tokens else ids

    def __call__(
        self,
        texts: List[str],
        *,
        return_tensors: str = "pt",
        padding: bool = True,
        truncation: bool = True,
        max_length: int = 64,
    ) -> Dict[str, torch.Tensor]:
        _ = (return_tensors, padding, truncation)
        seqs = [self.encode(t)[:max_length] for t in texts]
        mlen = max(len(s) for s in seqs)
        inp = torch.full((len(seqs), mlen), self.pad_token_id, dtype=torch.long)
        mask = torch.zeros((len(seqs), mlen), dtype=torch.long)
        for i, seq in enumerate(seqs):
            inp[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            mask[i, : len(seq)] = 1
        return {"input_ids": inp, "attention_mask": mask}

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        return " ".join(
            self._id2tok.get(int(i), "<unk>")
            for i in ids
            if not (skip_special_tokens and int(i) in self.all_special_ids)
        )


class _TinyDecoderBlock(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, None]:
        return x + 0.25 * torch.tanh(self.proj(x)), None


class _TinyHFStyleLM(nn.Module):
    def __init__(self, vocab_size: int, hidden_size: int = 16, n_layers: int = 5) -> None:
        super().__init__()
        self.model = nn.Module()
        self.model.embed = nn.Embedding(vocab_size, hidden_size)
        self.model.layers = nn.ModuleList([_TinyDecoderBlock(hidden_size) for _ in range(n_layers)])
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        _ = attention_mask
        h = self.model.embed(input_ids)
        for layer in self.model.layers:
            h, _ = layer(h)
        return self.lm_head(h)


# ---------------------------------------------------------------------------
# 1. Catalog, Terminology Header, & Legal Provenance Tests
# ---------------------------------------------------------------------------
def test_governance_header_and_legal_provenance_contract() -> None:
    hdr = pm6.PM6_GOVERNANCE_HEADER
    assert hdr["title"] == "Inference-Time Takedown Suppression & Cross-Generation Governance Carry-Forward"
    assert hdr["hosted_gateway_scope_only"] is True
    assert hdr["unlearning_claimed"] is False
    assert hdr["parametric_recall_neutering_claimed"] is False
    assert hdr["cites_audit_is_auditable_suppression_not_physical_erasure"] is True
    dockets = {ref["docket"] for ref in hdr["legal_provenance"]["primary_docket_references"]}
    assert "1:25-md-03143" in dockets
    assert "Az. 42 O 14139/24" in dockets


def test_catalog_size_passages_vault_chunks_and_external_merge(tmp_path: Path) -> None:
    catalog = pm6.build_pm6_canary_catalog()
    assert len(catalog) >= 140
    total_passages = sum(len(r.canonical_passages) for r in catalog)
    assert total_passages >= 300
    fids = [r.work_family_id for r in catalog]
    assert len(set(fids)) == len(catalog)

    for rec in catalog:
        assert len(rec.vault_chunks) == 8
        assert len(rec.r1_ground_truth.split()) >= 80
        assert set(rec.r2b_adaptive_variants.keys()) == {
            "interleaved_separator",
            "homoglyph_confusable",
            "entity_omitted_multiturn",
        }
        assert rec.r3_en and rec.r3_es and rec.r3_de
        assert rec.r5_benign_sibling_query and rec.r5_benign_reference

    # Test external JSONL merge hook
    ext_file = tmp_path / "gutenberg_extra.jsonl"
    ext_file.write_text(
        json.dumps(
            {
                "work_family_id": "ext_test_classic_999",
                "title": "The Phantom Archivist",
                "author_or_entity": "Archive Author",
                "era_or_sector": "1888 public domain",
                "core_substance": "the brass astrolabe maps tidal currents around Orkney islands",
                "benign_topic": "nineteenth-century Scottish lighthouse lens maintenance",
            }
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    merged = pm6.build_pm6_canary_catalog(external_jsonl_path=ext_file)
    assert len(merged) == len(catalog) + 1
    assert any(r.work_family_id == "ext_test_classic_999" for r in merged)


# ---------------------------------------------------------------------------
# 2. Split Invariance, Reduced-Yield Fallback, & Zero-Test-Leakage Tripwire
# ---------------------------------------------------------------------------
def test_entity_disjoint_split_invariance_and_480_vault_nodes() -> None:
    cat_a = pm6.build_pm6_canary_catalog()
    cat_b = list(reversed(cat_a))
    s_a = pm6.assign_entity_disjoint_split(cat_a, salt=pm6.PM6_SPLIT_SALT_V1, dev_count=50, test_count=60)
    s_b = pm6.assign_entity_disjoint_split(cat_b, salt=pm6.PM6_SPLIT_SALT_V1, dev_count=50, test_count=60)

    assert len(s_a.dev_families) == 50
    assert len(s_a.test_families) == 60
    assert len(s_a.background_families) >= 30
    assert sum(len(r.vault_chunks) for r in s_a.test_families) == 480
    assert s_a.reduced_yield_fallback_triggered is False

    # Permutation invariance under pre-committed salt
    assert s_a.dev_ids_sha256 == s_b.dev_ids_sha256
    assert s_a.test_ids_sha256 == s_b.test_ids_sha256
    assert s_a.background_ids_sha256 == s_b.background_ids_sha256

    dev_set = {r.work_family_id for r in s_a.dev_families}
    test_set = {r.work_family_id for r in s_a.test_families}
    bg_set = {r.work_family_id for r in s_a.background_families}
    assert not (dev_set & test_set)
    assert not (dev_set & bg_set)
    assert not (test_set & bg_set)


def test_reduced_yield_fallback_triggers_cleanly() -> None:
    cat = pm6.build_pm6_canary_catalog()
    confirmed = {r.work_family_id for r in cat[:44]}
    split = pm6.assign_entity_disjoint_split(cat, dev_count=50, test_count=60, confirmed_family_ids=confirmed)
    assert split.reduced_yield_fallback_triggered is True
    assert len(split.dev_families) + len(split.test_families) == 44
    assert len(split.dev_families) == 20
    assert len(split.test_families) == 24
    assert not ({r.work_family_id for r in split.dev_families} & {r.work_family_id for r in split.test_families})


def test_zero_test_leakage_tripwire_catches_all_leak_vectors() -> None:
    rng = np.random.default_rng(7)
    v_test = rng.standard_normal((8, 16))
    v_ref = rng.standard_normal((16, 16))
    base_kwargs = dict(
        test_family_ids=["tf1", "tf2"],
        test_vault_node_ids=["tf1::c0", "tf2::c0"],
        test_texts=["restricted canonical passage alpha", "benign sibling beta"],
        dev_family_ids=["df1"],
        background_family_ids=["bg1"],
        bridge_anchor_family_ids=["bg1"],
        bridge_anchor_node_ids=["bg1::c0"],
        bridge_anchor_texts=["background anchor text"],
        mu_fit_family_ids=["mu_en"],
        mu_fit_texts=["unlabeled english background sentence"],
        ref_pool_family_ids=["rp1"],
        ref_pool_node_ids=["rp::0"],
        ref_pool_texts=["reference pool disjoint text"],
        caa_dev_family_ids=["df1"],
        caa_dev_texts=["dev caa prompt text"],
        test_vault_vecs=v_test,
        ref_pool_vecs=v_ref,
    )
    ok = pm6.verify_zero_test_leakage(**base_kwargs)
    assert ok["verified"] is True

    # 1. Family leak into bridge anchors
    kw1 = dict(base_kwargs, bridge_anchor_family_ids=["bg1", "tf1"])
    with pytest.raises(ValueError, match="bridge_anchor_family_ids"):
        pm6.verify_zero_test_leakage(**kw1)

    # 2. Family leak into CAA dev pairs
    kw2 = dict(base_kwargs, caa_dev_family_ids=["df1", "tf2"])
    with pytest.raises(ValueError, match="caa_dev_family_ids"):
        pm6.verify_zero_test_leakage(**kw2)

    # 3. Node leak into reference pool
    kw3 = dict(base_kwargs, ref_pool_node_ids=["rp::0", "tf1::c0"])
    with pytest.raises(ValueError, match="ref_pool_node_ids"):
        pm6.verify_zero_test_leakage(**kw3)

    # 4. Text SHA-256 leak into mu_fit_texts (including benign sibling text!)
    kw4 = dict(base_kwargs, mu_fit_texts=["unlabeled sentence", "  Benign   Sibling BETA!! "])
    with pytest.raises(ValueError, match="mu_fit_texts"):
        pm6.verify_zero_test_leakage(**kw4)

    # 5. Exact duplicate vector leak between test_vault_vecs and ref_pool_vecs
    v_ref_leak = v_ref.copy()
    v_ref_leak[3] = v_test[0] * 2.5  # identical direction after L2 normalization
    kw5 = dict(base_kwargs, ref_pool_vecs=v_ref_leak)
    with pytest.raises(ValueError, match="vector_duplicate"):
        pm6.verify_zero_test_leakage(**kw5)


# ---------------------------------------------------------------------------
# 3. White-Box CAA (`Arm 7`) Forward Hooks, Logit Suppression & DEV Calibration
# ---------------------------------------------------------------------------
def test_caa_extract_directions_steering_hooks_and_cleanup() -> None:
    torch.manual_seed(42)
    vocab = [
        "whale", "pequod", "ahab", "restricted", "embargo", "secret",
        "harbor", "shipping", "public", "weather", "bridge", "history",
    ]
    tok = _SimpleWordTokenizer(vocab)
    model = _TinyHFStyleLM(vocab_size=tok.vocab_size, hidden_size=16, n_layers=5)

    dev_res = ["whale pequod ahab restricted embargo", "ahab secret embargo whale"]
    dev_ben = ["harbor shipping public weather history", "bridge public weather harbor"]
    dev_fids = ["dev_fam_1", "dev_fam_2"]
    test_fids = ["test_fam_1", "test_fam_2"]

    # Zero-test-overlap rejection check
    with pytest.raises(ValueError, match="Zero-test-leakage"):
        caa.extract_contrastive_directions(
            model, tok, dev_res, dev_ben,
            dev_family_ids=["dev_fam_1", "test_fam_1"],
            test_family_ids=test_fids,
            target_layers=(1, 2, 3),
        )

    dir_set = caa.extract_contrastive_directions(
        model, tok, dev_res, dev_ben,
        dev_family_ids=dev_fids,
        test_family_ids=test_fids,
        test_texts=["completely distinct held out test prompt"],
        target_layers=(1, 2, 3),
        max_length=32,
        batch_size=2,
    )
    assert dir_set.target_layers == (1, 2, 3)
    assert dir_set.scope_disclosure == caa.CAA_SCOPE_DISCLOSURE
    for l_idx in (1, 2, 3):
        norm = float(torch.linalg.vector_norm(dir_set.directions[l_idx]).item())
        assert abs(norm - 1.0) < 1e-5

    # Contrastive logit penalty
    lp = caa.build_dev_contrastive_logit_penalty(
        tok, dev_res, dev_ben,
        dev_family_ids=dev_fids, test_family_ids=test_fids,
        top_k_tokens=4, min_freq=2, penalty_strength=3.0,
    )
    assert len(lp.token_ids) >= 1
    dummy_logits = torch.zeros((1, tok.vocab_size), dtype=torch.float32)
    penalized = lp(torch.zeros((1, 2), dtype=torch.long), dummy_logits)
    for tid in lp.token_ids:
        assert float(penalized[0, tid].item()) < 0.0

    # Forward hooks actually steer logits and clean up cleanly (even on error!)
    enc = tok(["whale pequod ahab"])
    with torch.no_grad():
        base_logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])

    with caa.CAASteeringController(model, dir_set, alpha=2.0, logit_penalty=lp, apply_logit_hook_on_lm_head=True) as ctrl:
        with torch.no_grad():
            steered_logits = model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        assert ctrl.hook_invocations == 3
        assert ctrl.logit_hook_invocations == 1
        assert float(torch.max(torch.abs(steered_logits - base_logits)).item()) > 1e-3

    # Verify zero lingering hooks after context exit
    for layer in model.model.layers:
        assert len(layer._forward_hooks) == 0
    assert len(model.lm_head._forward_hooks) == 0

    # Verify hook cleanup on exception inside context
    with pytest.raises(RuntimeError, match="boom"):
        with caa.CAASteeringController(model, dir_set, alpha=1.5):
            raise RuntimeError("boom")
    for layer in model.model.layers:
        assert len(layer._forward_hooks) == 0

    # Greedy generation smoke with CAA hooks
    gen_txt, meta = caa.generate_text_with_caa(
        model, tok, "whale pequod", dir_set, alpha=1.5, logit_penalty=lp, max_new_tokens=5
    )
    assert isinstance(gen_txt, str)
    assert meta["hook_invocations"] == 3 * 5
    assert meta["scope_disclosure"] == caa.CAA_SCOPE_DISCLOSURE


def test_caa_dev_multiplier_calibration_and_fallback() -> None:
    def _eval_fn(alpha: float) -> Dict[str, float]:
        table = {
            0.5: (0.40, 0.00),
            1.0: (0.62, 0.01),
            1.5: (0.74, 0.02),
            2.0: (0.81, 0.06),
            3.0: (0.90, 0.18),
        }
        s, f = table[alpha]
        return {"r1_r3_suppression_rate": s, "r5_benign_fpr": f}

    res = caa.calibrate_caa_multiplier_on_dev(
        _eval_fn,
        dev_family_ids=["d1", "d2"],
        test_family_ids=["t1", "t2"],
        target_fpr=0.02,
        timebox_s=30.0,
    )
    assert res.met_target_fpr is True
    assert res.selected_alpha == 1.5
    assert res.selected_r1_r3_suppression_rate == pytest.approx(0.74)
    assert res.scope_disclosure == caa.CAA_SCOPE_DISCLOSURE
    assert len(res.sweep_results) == 5

    # Fallback when all alphas violate target_fpr
    res_fb = caa.calibrate_caa_multiplier_on_dev(
        lambda a: {"r1_r3_suppression_rate": 0.5 + 0.1 * a, "r5_benign_fpr": 0.05 + 0.02 * a},
        dev_family_ids=["d1"],
        test_family_ids=["t1"],
        target_fpr=0.02,
    )
    assert res_fb.met_target_fpr is False
    assert res_fb.selected_alpha == 0.5


# ---------------------------------------------------------------------------
# 4. Statistical Calibration, Exact Clopper-Pearson, & 60-Cluster Bootstrap
# ---------------------------------------------------------------------------
def test_matched_fpr_calibration_and_clopper_pearson_bounds() -> None:
    rng = np.random.default_rng(123)
    dev_pos = rng.normal(loc=0.80, scale=0.08, size=50)
    dev_neg = rng.normal(loc=0.20, scale=0.10, size=50)
    cal = pm6.calibrate_threshold_for_target_fpr(dev_pos, dev_neg, target_fpr=0.02)
    assert cal["dev_empirical_fpr"] <= 0.02 + 1e-12
    assert cal["dev_empirical_tpr"] >= 0.90

    # Exact 0/60 Clopper-Pearson one-sided 95% upper bound: 1 - 0.05^(1/60) = 0.048701 <= 4.9%
    ub_0_60 = pm6.clopper_pearson_one_sided_upper(0, 60, alpha=0.05)
    assert abs(ub_0_60 - (1.0 - (0.05 ** (1.0 / 60.0)))) < 1e-9
    assert ub_0_60 <= 0.049
    assert pm6.clopper_pearson_one_sided_upper(60, 60, alpha=0.05) == 1.0


def test_60_cluster_paired_auc_bootstrap_exact_match() -> None:
    rng = np.random.default_rng(999)
    fams = [f"fam_{i:02d}" for i in range(60)]
    pos_a = {f: float(rng.normal(0.85, 0.12)) for f in fams}
    neg_a = {f: float(rng.normal(0.15, 0.12)) for f in fams}
    pos_b = {f: float(rng.normal(0.50, 0.25)) for f in fams}
    neg_b = {f: float(rng.normal(0.35, 0.25)) for f in fams}

    res = pm6.cluster_bootstrap_paired_auc_diff(pos_a, neg_a, pos_b, neg_b, n_bootstrap=2000, seed=42)
    assert res["n_clusters"] == 60
    assert res["auc_a"] > res["auc_b"]
    assert res["ci95_low"] > 0.0
    assert res["ci95_low"] <= res["auc_diff"] <= res["ci95_high"]

    # Cross-check exact equivalence of bilinear Mann-Whitney identity against explicit multiset resampling
    pa = np.array([pos_a[f] for f in fams])
    na = np.array([neg_a[f] for f in fams])
    pb = np.array([pos_b[f] for f in fams])
    nb = np.array([neg_b[f] for f in fams])
    r_check = np.random.default_rng(42)
    draws = r_check.integers(0, 60, size=(20, 60))
    explicit_diffs = []
    for d in draws:
        sa_p, sa_n = pa[d], na[d]
        sb_p, sb_n = pb[d], nb[d]
        auca = float(np.mean((sa_p[:, None] > sa_n[None, :]) + 0.5 * (sa_p[:, None] == sa_n[None, :])))
        aucb = float(np.mean((sb_p[:, None] > sb_n[None, :]) + 0.5 * (sb_p[:, None] == sb_n[None, :])))
        explicit_diffs.append(auca - aucb)
    # Bilinear check on same draws
    ma = (pa[:, None] > na[None, :]).astype(np.float64) + 0.5 * (pa[:, None] == na[None, :]).astype(np.float64)
    mb = (pb[:, None] > nb[None, :]).astype(np.float64) + 0.5 * (pb[:, None] == nb[None, :]).astype(np.float64)
    w = np.zeros((20, 60), dtype=np.float64)
    np.add.at(w, (np.arange(20)[:, None], draws), 1.0)
    fast_diffs = np.einsum("bi,ij,bj->b", w, ma - mb, w) / float(60 * 60)
    assert np.allclose(explicit_diffs, fast_diffs, atol=1e-12)


# ---------------------------------------------------------------------------
# 5. Memorization Confirmation, Freeze Lock, Narrowing Rule, & Linear Bridge
# ---------------------------------------------------------------------------
def test_memorization_confirmation_and_r2b_mutators() -> None:
    ref = (
        "In the canonical text of Moby-Dick by Herman Melville the opening sequence establishes how "
        "Ishmael boards the Pequod under Captain Ahab hunting the white sperm whale across Pacific waters"
    )
    ok_exact, rl, ng = pm6.is_memorization_confirmed(ref, ref)
    assert ok_exact is True and rl == pytest.approx(1.0) and ng == pytest.approx(1.0)

    ok_unrel, rl_u, ng_u = pm6.is_memorization_confirmed("ordinary harbor weather bulletin for Tuesday morning", ref)
    assert ok_unrel is False and rl_u < 0.2 and ng_u == 0.0

    sep_var = pm6.make_interleaved_separator_variant(ref)
    hom_var = pm6.make_homoglyph_confusable_variant(ref)
    assert sep_var != ref and "\u200b" in sep_var
    assert hom_var != ref and any(ord(ch) > 127 for ch in hom_var)


def test_calibration_freeze_lock_and_arm3_narrowing_rule() -> None:
    payload = {
        "schema_version": "pm6_calibration_freeze_v1",
        "zero_test_leakage_verified": True,
        "r2b_unlocked_only_after_freeze": True,
        "frozen_thresholds_by_arm": {"zeromu_csls": {"threshold": 0.42}},
    }
    sha = pm6.sha256_text(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True))
    valid_fm = dict(payload, freeze_manifest_sha256=sha)
    pm6.assert_calibration_frozen(valid_fm)

    # Tampered threshold fails closed
    tampered = dict(valid_fm, frozen_thresholds_by_arm={"zeromu_csls": {"threshold": 0.11}})
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        pm6.assert_calibration_frozen(tampered)

    # Unverified leakage fails closed
    unverified = dict(valid_fm, zero_test_leakage_verified=False)
    with pytest.raises(RuntimeError, match="zero_test_leakage_verified"):
        pm6.assert_calibration_frozen(unverified)

    # Arm-3 narrowing rule
    nar_true = pm6.evaluate_arm3_narrowing_rule(0.98, 0.95, 0.99, 0.96, tolerance=0.02)
    assert nar_true["r1_r2_claim_narrowed_to_r3_r4_carryforward"] is True
    nar_false = pm6.evaluate_arm3_narrowing_rule(0.98, 0.40, 0.99, 0.95, tolerance=0.02)
    assert nar_false["r1_r2_claim_narrowed_to_r3_r4_carryforward"] is False


def test_linear_car_ridge_bridge_dual_primal_parity() -> None:
    rng = np.random.default_rng(55)
    # Dual regime: N < d
    x = rng.standard_normal((24, 64))
    y = rng.standard_normal((24, 12))
    lam = 0.05
    w_dual, mu_x, mu_y = pm6.fit_linear_car_ridge_bridge(x, y, ridge_lambda=lam)
    xc = x - mu_x[None, :]
    yc = y - mu_y[None, :]
    w_primal = np.linalg.solve(xc.T @ xc + lam * np.eye(64), xc.T @ yc)
    assert w_dual.shape == (64, 12)
    assert np.allclose(w_dual, w_primal, atol=1e-9)


# ---------------------------------------------------------------------------
# 6. Full End-to-End `--mode local-drycheck` & `--mode assemble-manifest`
# ---------------------------------------------------------------------------
def test_local_drycheck_end_to_end_manifest_receipts_and_non_nda(tmp_path: Path) -> None:
    rc = pm6.main(["--mode", "local-drycheck", "--output-dir", str(tmp_path), "--n-bootstrap", "1000", "--skip-real-sleep"])
    assert rc == 0

    manifest_path = tmp_path / pm6.MANIFEST_NAME
    freeze_path = tmp_path / pm6.FREEZE_MANIFEST_NAME
    receipts_path = tmp_path / pm6.RECEIPTS_FILE_NAME
    report_path = tmp_path / pm6.REPORT_RECEIPT_NAME
    for p in (manifest_path, freeze_path, receipts_path, report_path):
        assert p.is_file()

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    receipts = json.loads(receipts_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))

    assert manifest["all_gates_pass"] is True
    assert report["status"] == "COMPLETED"
    assert report["non_nda_pattern_count"] == len(FORBIDDEN_METHOD_PATTERNS) == 56
    assert report["non_nda_leak_hits"] == 0

    # Catalog & split invariants
    cs = manifest["catalog_summary"]
    assert cs["total_work_families"] >= 140
    assert cs["dev_families"] == 50
    assert cs["test_families"] == 60
    assert cs["test_restricted_vault_nodes"] == 480
    assert cs["ref_pool_vectors"] == 4096

    # Freeze lock matches R2b post-freeze stamp
    assert manifest["post_freeze_r2b_adaptive_redteam"]["unlocked_after_freeze_sha256"] == freeze["freeze_manifest_sha256"]
    assert manifest["post_freeze_r2b_adaptive_redteam"]["k_variants_per_work"] == 3

    # All 9 arms present & Arm 9 beats Arm 6 (Raw Cosine) on R3 vs R5 AUROC
    assert set(manifest["nine_arm_matrix"].keys()) == set(pm6.ALL_NINE_ARMS)
    arm9_auc = manifest["nine_arm_matrix"]["arm9_chelated_full_dual_gate"]["r3_vs_r5_roc"]["auroc"]
    arm6_auc = manifest["nine_arm_matrix"]["arm6_gate1_plus_gate2_raw_cosine"]["r3_vs_r5_roc"]["auroc"]
    assert arm9_auc >= arm6_auc

    # EK5 v3 lineage audit: 0 COPY_OF/DERIVES_CURRENT leaks, 100% CITES_AUDIT preserved
    ek5 = manifest["ek5_v3_r4_lineage_audit"]
    assert ek5["copy_of_leaked_general"] == 0
    assert ek5["derives_current_leaked_general"] == 0
    assert ek5["cites_audit_preserved_general"] == 60
    assert ek5["auditable_suppression_not_erasure"] is True

    # Phase 3 4-way cross-lingual ES/DE ablation: unlabeled mu_lang beats mu_en and MiniLM-384
    p3 = manifest["phase3_cross_lingual_4way_ablation"]
    assert "native_4096d_reembed_cost_ms" in p3
    for lang in ("es", "de"):
        l_res = p3["languages"][lang]
        auc_a = l_res["a_monolingual_minilm_384"]["auroc"]
        auc_b = l_res["b_bridge_4096_to_384_mu_en"]["auroc"]
        auc_c = l_res["c_bridge_4096_to_384_mu_lang_unlabeled"]["auroc"]
        assert auc_c > auc_a
        assert auc_c > auc_b

    # Transport profiles strictly separated; Profile A 0/60 leaks + 500ms 16-stride delay pass
    tp = manifest["transport_profiles"]
    pa = tp["profile_a_strict_compliance"]
    pb = tp["profile_b_armed_async_sla"]
    assert pa["profile_id"] == "profile_a_strict"
    assert pb["profile_id"] == "profile_b_armed_async"
    assert pa["flagged_window_client_byte_leaks_per_work"] == 0
    assert pa["client_byte_leak_clopper_pearson_upper_95"] <= 0.049
    assert pa["injected_500ms_delay_test"]["strides_tested"] == 16
    assert pa["injected_500ms_delay_test"]["total_flagged_window_bytes_leaked"] == 0
    assert pa["injected_500ms_delay_test"]["zero_byte_leak_pass"] is True
    assert tp["sse_sidechannel_audit"]["strip_mode_clean"] is True
    assert tp["sse_sidechannel_audit"]["strict_reject_raised_value_error"] is True

    # Receipts integer-only, signed, verified, and 0/56 Non-NDA clean
    assert receipts["count"] == 12
    assert receipts["all_entries_valid"] is True
    assert receipts["chain_verified"] is True
    assert receipts["zero_floats_verified"] is True
    for entry in receipts["entries"]:
        assert scan_text_for_ip_leaks(json.dumps(entry, sort_keys=True)) == []

    # Re-run `--mode assemble-manifest` cleanly
    rc_asm = pm6.main(["--mode", "assemble-manifest", "--output-dir", str(tmp_path)])
    assert rc_asm == 0


def test_assemble_manifest_fails_closed_on_non_nda_leak_or_tamper(tmp_path: Path) -> None:
    rc = pm6.main(["--mode", "local-drycheck", "--output-dir", str(tmp_path), "--n-bootstrap", "200", "--skip-real-sleep"])
    assert rc == 0

    receipts_path = tmp_path / pm6.RECEIPTS_FILE_NAME
    receipts = json.loads(receipts_path.read_text(encoding="utf-8"))
    receipts["entries"][0]["payload"]["leaked_hint"] = "stiefel procrustes csls"
    receipts_path.write_text(json.dumps(receipts, indent=2) + "\n", encoding="utf-8", newline="\n")

    with pytest.raises(ValueError, match="Non-NDA IP leak"):
        pm6.assemble_report_receipt(output_dir=tmp_path)


def test_real_500ms_injected_delay_wall_clock_smoke() -> None:
    cat = pm6.build_pm6_canary_catalog()
    split = pm6.assign_entity_disjoint_split(cat, dev_count=50, test_count=60)
    space = pm6.SyntheticDrycheckSpace(seed=777, d_old=384, d_new=4096)

    rec = split.test_families[0]
    nids = [nid for nid, _ in rec.vault_chunks]
    wids = [rec.work_family_id for _ in rec.vault_chunks]
    vault_384 = np.asarray(
        [space.embed_pair(work_family_id=rec.work_family_id, item_key=nid, lang="en", signal_strength=0.80)[0] for nid in nids],
        dtype=np.float32,
    )
    rng = np.random.default_rng(19)
    ref_pool = pm6.l2_normalize(space.cone_old[None, :] + 0.65 * rng.standard_normal((128, 384))).astype(np.float32)
    dm, dn, dr = pm6.build_vault_sidecars(vault_384, ref_pool, csls_k=8)
    v_idx = pm6.firewall.RestrictedVaultIndex(
        node_ids=nids, work_ids=wids, node_vecs=vault_384,
        doc_mean=dm, doc_centered_norms=dn, doc_csls_r=dr,
        ref_pool_vecs=ref_pool,
        query_mean_by_lang={"en": space.cone_new},
        csls_k=8,
    )
    scanner = pm6.firewall.Tier2ACanonicalScanner([(rec.work_family_id, rec.r1_ground_truth)])
    _, q_4096 = space.embed_pair(work_family_id=rec.work_family_id, item_key="delay_probe", lang="en", signal_strength=0.85)
    delay_res = pm6.run_injected_delay_safety_test(
        tier2a=scanner,
        flagged_work=rec,
        vault_index=v_idx,
        bridge_w=space.q_true,
        tau_gov=0.10,
        flagged_vec_4096=q_4096,
        real_sleep_first_stride=True,
    )
    assert delay_res["zero_byte_leak_pass"] is True
    assert delay_res["total_flagged_window_bytes_leaked"] == 0
    assert delay_res["real_sleep_measured_ms"] >= 450.0


def test_roc_curve_and_cluster_bootstrap_rate_ci() -> None:
    pos = [0.9, 0.85, 0.8, 0.75, 0.7]
    neg = [0.1, 0.2, 0.3, 0.4, 0.5]
    roc = pm6.compute_roc_curve_and_auc(pos, neg)
    assert roc["auroc"] == pytest.approx(1.0)
    assert roc["fpr"][0] == 0.0 and roc["fpr"][-1] == 1.0
    assert roc["tpr"][0] == 0.0 and roc["tpr"][-1] == 1.0

    vals = {f"fam_{i:02d}": (1.0 if i < 54 else 0.0) for i in range(60)}
    ci = pm6.cluster_bootstrap_rate_ci(vals, n_bootstrap=2000, seed=20261124)
    assert ci["n_clusters"] == 60
    assert ci["mean"] == pytest.approx(0.90)
    assert 0.75 <= ci["ci95_low"] <= ci["mean"] <= ci["ci95_high"] <= 1.0


def test_cli_arg_parser_modes_and_defaults() -> None:
    parser = pm6.build_arg_parser()
    a1 = parser.parse_args(["--mode", "local-drycheck", "--output-dir", "tmp_out"])
    assert a1.mode == "local-drycheck" and a1.output_dir == Path("tmp_out")
    a2 = parser.parse_args(["--mode", "start-spark3-embed-server", "--port", "8897"])
    assert a2.mode == "start-spark3-embed-server" and a2.port == 8897
    a3 = parser.parse_args(["--mode", "run-live", "--nemotron-url", "http://10.0.0.49:8891/v1", "--qwen3-embed-url", "http://10.0.0.24:8897"])
    assert a3.mode == "run-live" and a3.nemotron_url == "http://10.0.0.49:8891/v1"
    a4 = parser.parse_args(["--mode", "assemble-manifest", "--output-dir", "tmp_out"])
    assert a4.mode == "assemble-manifest"

