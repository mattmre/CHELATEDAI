"""Inference-time takedown suppression firewall (Wave 14B / PM-6).

Hosted-gateway controls only: a canonical Tier-2A scanner, an exact Tier-2B
vault score, an EK5 v3 lineage fence, a closed-loop UTF-8 byte release gate,
and integer Ed25519 receipts. This module does not erase model weights.
"""

from __future__ import annotations

import base64
import codecs
import hashlib
import hmac
import json
import re
import time
import unicodedata
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

from ek5_v3_stratified import (
    EdgeAttestation,
    StratifiedEvent,
    StratifiedEvidenceStore,
    compute_alternating_wfs_oracle,
)
from scripts.verify_non_nda_ip_clean import assert_non_nda_clean

STREAM_PROFILES: Tuple[str, str] = ("profile_a_strict", "profile_b_armed_async")

_SIDECHANNEL_KEYS = frozenset({
    "logprobs",
    "top_logprobs",
    "tool_calls",
    "function_call",
    "reasoning_content",
    "thinking",
})

_MARK_CATEGORIES = frozenset({"Cf", "Mn", "Me"})

# Cyrillic / Greek lookalikes that NFKC does not fold to ASCII.
_HOMOGLYPH_SOURCE = {
    "a": "аАαΑ",
    "b": "ЬЬ"[:0] + "вВβΒ",
    "c": "сС",
    "d": "ԁԀ",
    "e": "еЕёЁεΕ",
    "g": "ɡ",
    "h": "нНηΗһҺ",
    "i": "іІїЇιΙӏӀ",
    "j": "јЈ",
    "k": "кКκΚ",
    "l": "ℓ",
    "m": "мМμΜ",
    "n": "νΝ",
    "o": "оОοΟ",
    "p": "рРρΡ",
    "q": "ԛԚ",
    "s": "ѕЅ",
    "t": "тТτΤ",
    "u": "υΥ",
    "w": "ԝԜ",
    "x": "хХχΧ",
    "y": "уУ",
    "z": "ζΖ",
}
_HOMOGLYPH: Dict[str, str] = {}
for _ascii, _chars in _HOMOGLYPH_SOURCE.items():
    for _ch in _chars:
        _HOMOGLYPH[_ch] = _ascii

_DASH_TO_HYPHEN = {
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2212": "-",
    "\u00ad": "",
}

_GENESIS_HASH = "0" * 64
_PUBLIC_STATE = {
    "ADMIT": "Current",
    "AUDIT_ONLY": "Withdrawn",
    "CONTESTED": "Disputed",
    "Current": "Current",
    "Withdrawn": "Withdrawn",
    "Disputed": "Disputed",
}
_ACTION_LABEL = "Inference-Time Takedown Suppression"


def _reject_floats(obj: Any, path: str = "$") -> None:
    if isinstance(obj, float):
        raise ValueError(f"float not allowed in canonical payload at {path}")
    if isinstance(obj, dict):
        for key, value in obj.items():
            _reject_floats(value, f"{path}.{key}")
    elif isinstance(obj, (list, tuple)):
        for index, value in enumerate(obj):
            _reject_floats(value, f"{path}[{index}]")


def canonical_json(obj: Any) -> bytes:
    """Integer-only canonical JSON: sorted keys, no whitespace, ASCII."""
    _reject_floats(obj)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def _fold_codepoint(ch: str) -> str:
    if ch in _DASH_TO_HYPHEN:
        ch = _DASH_TO_HYPHEN[ch]
        if not ch:
            return ""
    if unicodedata.category(ch) in _MARK_CATEGORIES:
        return ""
    ch = _HOMOGLYPH.get(ch, ch)
    folded: List[str] = []
    for part in unicodedata.normalize("NFKC", ch):
        part = _HOMOGLYPH.get(part, part)
        for low in part.lower():
            if unicodedata.category(low) in _MARK_CATEGORIES:
                continue
            low = _HOMOGLYPH.get(low, low)
            if low and unicodedata.category(low) not in _MARK_CATEGORIES:
                folded.append(low)
    return "".join(folded)


def _piece_kind(ch: str) -> str:
    if ch.isspace():
        return "ws"
    if ch.isalnum():
        return "word"
    return "sep"


def canonicalize_for_tier2a(raw_text: str) -> Tuple[str, List[Tuple[int, int]]]:
    """NFKC + homoglyph fold + mark strip + separator collapse, with raw UTF-8 spans.

    Every index of the returned canonical string maps to ``(raw_byte_start, raw_byte_end)``
    inside ``raw_text.encode('utf-8')``. The end offset is exclusive.
    """
    if not isinstance(raw_text, str):
        raise TypeError("raw_text must be a str")
    pieces: List[Tuple[str, int, int, str]] = []
    byte_pos = 0
    for ch in raw_text:
        raw_len = len(ch.encode("utf-8"))
        start = byte_pos
        end = byte_pos + raw_len
        byte_pos = end
        folded = _fold_codepoint(ch)
        for out_ch in folded:
            pieces.append((out_ch, start, end, _piece_kind(out_ch)))

    collapsed: List[Tuple[str, int, int]] = []
    index = 0
    count = len(pieces)
    while index < count:
        ch, start, end, kind = pieces[index]
        if kind == "word":
            prev_kind = pieces[index - 1][3] if index else ""
            # Only an isolated single-letter run (c-a-l-l, c.a.l.l) is obfuscation.
            # A hyphen inside Hello-World stays a word break because 'o' is glued
            # to the preceding letters.
            if prev_kind != "word":
                cursor = index
                words = [pieces[index]]
                while (
                    cursor + 2 < count
                    and pieces[cursor + 1][3] == "sep"
                    and pieces[cursor + 2][3] == "word"
                ):
                    words.append(pieces[cursor + 2])
                    cursor += 2
                next_kind = pieces[cursor + 1][3] if cursor + 1 < count else ""
                if len(words) >= 2 and next_kind != "word":
                    for word_ch, word_start, word_end, _kind in words:
                        collapsed.append((word_ch, word_start, word_end))
                    index = cursor + 1
                    continue
        if kind in {"sep", "ws"}:
            collapsed.append((" ", start, end))
        else:
            collapsed.append((ch, start, end))
        index += 1

    canon_parts: List[str] = []
    spans: List[Tuple[int, int]] = []
    for ch, start, end in collapsed:
        if ch == " ":
            if not spans:
                continue
            if canon_parts[-1] == " ":
                prev_start, _prev_end = spans[-1]
                spans[-1] = (prev_start, end)
                continue
        canon_parts.append(ch)
        spans.append((start, end))
    if canon_parts and canon_parts[-1] == " ":
        canon_parts.pop()
        spans.pop()
    canon = "".join(canon_parts)
    if len(canon) != len(spans):
        raise RuntimeError("canonical span map drifted from the canonical string")
    return canon, spans


def _lcs_length(left: Sequence[str], right: Sequence[str]) -> int:
    if not left or not right:
        return 0
    if len(left) > len(right):
        left, right = right, left
    prev = [0] * (len(left) + 1)
    for token in right:
        current = [0]
        for col, item in enumerate(left, start=1):
            if token == item:
                current.append(prev[col - 1] + 1)
            else:
                current.append(prev[col] if prev[col] >= current[-1] else current[-1])
        prev = current
    return prev[-1]


class Tier2ACanonicalScanner:
    """O(1) canonical character/word n-gram index plus a bounded token-LCS check."""

    def __init__(
        self,
        restricted_spans: Sequence[Tuple[str, str]],
        *,
        ngram_n: int = 10,
        lcs_min_tokens: int = 14,
        lcs_ratio_threshold: float = 0.72,
    ) -> None:
        if ngram_n < 1:
            raise ValueError("ngram_n must be >= 1")
        if lcs_min_tokens < 1:
            raise ValueError("lcs_min_tokens must be >= 1")
        if not 0.0 < float(lcs_ratio_threshold) <= 1.0:
            raise ValueError("lcs_ratio_threshold must be in (0, 1]")
        self.ngram_n = int(ngram_n)
        self.lcs_min_tokens = int(lcs_min_tokens)
        self.lcs_ratio_threshold = float(lcs_ratio_threshold)
        self._char_index: Dict[str, List[str]] = {}
        self._word_index: Dict[Tuple[str, ...], List[str]] = {}
        self._work_tokens: List[Tuple[str, Tuple[str, ...]]] = []
        for item in restricted_spans:
            if len(item) != 2:
                raise ValueError("restricted span entries must be (work_id, text)")
            work_id, text = str(item[0]), str(item[1])
            canon, _spans = canonicalize_for_tier2a(text)
            self._index_work(work_id, canon)

    def _index_work(self, work_id: str, canon: str) -> None:
        n = self.ngram_n
        if len(canon) >= n:
            for offset in range(len(canon) - n + 1):
                gram = canon[offset : offset + n]
                self._char_index.setdefault(gram, [])
                if work_id not in self._char_index[gram]:
                    self._char_index[gram].append(work_id)
        words = tuple(tok for tok in canon.split(" ") if tok)
        self._work_tokens.append((work_id, words))
        if len(words) >= n:
            for offset in range(len(words) - n + 1):
                key = words[offset : offset + n]
                self._word_index.setdefault(key, [])
                if work_id not in self._word_index[key]:
                    self._word_index[key].append(work_id)

    def scan_text(self, text: str) -> Dict[str, Any]:
        canon, spans = canonicalize_for_tier2a(text)
        matched: List[str] = []
        seen: Set[str] = set()
        best_start: Optional[int] = None
        best_end: Optional[int] = None
        match_kind = "none"
        score = 0.0

        def _consider(work_id: str, start: int, end: int, kind: str, kind_score: float) -> None:
            nonlocal best_start, best_end, match_kind, score
            if work_id not in seen:
                seen.add(work_id)
                matched.append(work_id)
            if best_start is None or start < best_start or (start == best_start and end < (best_end or end)):
                best_start = start
                best_end = end
            if match_kind == "none" or (match_kind == "lcs" and kind != "lcs"):
                match_kind = kind
            if kind_score > score:
                score = kind_score

        n = self.ngram_n
        if len(canon) >= n and spans:
            for offset in range(len(canon) - n + 1):
                gram = canon[offset : offset + n]
                owners = self._char_index.get(gram)
                if not owners:
                    continue
                start = spans[offset][0]
                end = spans[offset + n - 1][1]
                for work_id in owners:
                    _consider(work_id, start, end, "char_ngram", 1.0)
        words: List[Tuple[str, int, int]] = []
        if canon:
            cursor = 0
            for part in canon.split(" "):
                if not part:
                    continue
                start = spans[cursor][0]
                end = spans[cursor + len(part) - 1][1]
                words.append((part, start, end))
                cursor += len(part) + 1
        if len(words) >= n:
            for offset in range(len(words) - n + 1):
                key = tuple(words[offset + delta][0] for delta in range(n))
                owners = self._word_index.get(key)
                if not owners:
                    continue
                start = words[offset][1]
                end = words[offset + n - 1][2]
                for work_id in owners:
                    _consider(work_id, start, end, "word_ngram", 1.0)

        if match_kind == "none" and len(words) >= self.lcs_min_tokens:
            window_tokens = [item[0] for item in words]
            for work_id, restricted_tokens in self._work_tokens:
                if not restricted_tokens:
                    continue
                compared = restricted_tokens[:512]
                for lo, hi in _lcs_windows(len(window_tokens), self.lcs_min_tokens):
                    window = window_tokens[lo:hi]
                    if len(window) < self.lcs_min_tokens:
                        continue
                    length = _lcs_length(window, compared)
                    ratio = length / float(len(window))
                    if ratio + 1e-12 >= self.lcs_ratio_threshold:
                        _consider(work_id, words[lo][1], words[hi - 1][2], "lcs", ratio)

        flagged = bool(matched)
        return {
            "flagged": flagged,
            "matched_work_ids": matched,
            "score": float(score if flagged else 0.0),
            "match_kind": match_kind if flagged else "none",
            "flagged_raw_byte_start": best_start if flagged else None,
            "flagged_raw_byte_end": best_end if flagged else None,
        }


def _lcs_windows(token_count: int, min_tokens: int) -> List[Tuple[int, int]]:
    if token_count < min_tokens:
        return []
    lengths = {min_tokens}
    if token_count <= 96:
        lengths.add(token_count)
        if token_count > min_tokens:
            lengths.add(min(token_count, max(min_tokens, 64)))
    windows: List[Tuple[int, int]] = []
    for length in sorted(lengths):
        if length > token_count or length < min_tokens:
            continue
        step = 1 if token_count <= 80 else max(1, length // 4)
        for start in range(0, token_count - length + 1, step):
            windows.append((start, start + length))
    return windows


def _as_2d(name: str, value: np.ndarray) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must be finite")
    return np.ascontiguousarray(arr)


def _row_key(row: np.ndarray) -> bytes:
    return np.ascontiguousarray(row, dtype=np.float64).tobytes()


class RestrictedVaultIndex:
    """Exact GEMV over restricted nodes with a test-disjoint CSLS reference pool.

    ``bridge_w`` with shape ``(4096, 384)`` maps a native query by ``q_384 = q_4096 @ W``.
    The transposed layout ``(384, 4096)`` is accepted as ``q_384 = W @ q_4096``.
    """

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
        metadata: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.node_ids = tuple(str(item) for item in node_ids)
        self.work_ids = tuple(str(item) for item in work_ids)
        self.node_vecs = _as_2d("node_vecs", node_vecs)
        if self.node_vecs.shape[0] != len(self.node_ids):
            raise ValueError("node_vecs.shape[0] must equal len(node_ids)")
        if len(self.work_ids) != len(self.node_ids):
            raise ValueError("work_ids and node_ids must have the same length")
        self.ref_pool_vecs = _as_2d("ref_pool_vecs", ref_pool_vecs)
        if self.ref_pool_vecs.shape[0] < 64:
            raise ValueError("ref_pool_vecs must contain at least 64 test-disjoint vectors")
        if self.ref_pool_vecs.shape[1] != self.node_vecs.shape[1]:
            raise ValueError("ref_pool_vecs dim must match node_vecs dim")
        self.doc_centered_norms = np.asarray(doc_centered_norms, dtype=np.float64).reshape(-1)
        self.doc_csls_r = np.asarray(doc_csls_r, dtype=np.float64).reshape(-1)
        if self.doc_centered_norms.shape[0] != len(self.node_ids):
            raise ValueError("doc_centered_norms length must equal len(node_ids)")
        if self.doc_csls_r.shape[0] != len(self.node_ids):
            raise ValueError("doc_csls_r length must equal len(node_ids)")
        mean = np.asarray(doc_mean, dtype=np.float64)
        if mean.ndim == 2 and mean.shape[0] == 1:
            mean = mean.reshape(-1)
        if mean.shape != (self.node_vecs.shape[1],) and mean.shape != self.node_vecs.shape:
            raise ValueError("doc_mean must have shape (d,) or (n, d)")
        self.doc_mean = mean
        if int(csls_k) < 1:
            raise ValueError("csls_k must be >= 1")
        self.csls_k = int(csls_k)
        self.query_mean_by_lang: Dict[str, np.ndarray] = {}
        for lang, vec in query_mean_by_lang.items():
            arr = np.asarray(vec, dtype=np.float64).reshape(-1)
            if arr.shape[0] < 1 or not np.isfinite(arr).all():
                raise ValueError(f"query mean for {lang!r} must be a non-empty finite vector")
            self.query_mean_by_lang[str(lang)] = arr
        self._assert_disjoint(metadata)

    def _assert_disjoint(self, metadata: Optional[Mapping[str, Any]]) -> None:
        node_keys = {_row_key(self.node_vecs[i]) for i in range(self.node_vecs.shape[0])}
        for row_index in range(self.ref_pool_vecs.shape[0]):
            if _row_key(self.ref_pool_vecs[row_index]) in node_keys:
                raise AssertionError("reference pool overlaps a restricted vault node")
        if metadata is None:
            return
        meta_nodes = metadata.get("node_ids")
        meta_refs = metadata.get("ref_ids")
        if meta_nodes is not None and meta_refs is not None:
            overlap = set(map(str, meta_nodes)) & set(map(str, meta_refs))
            if overlap:
                raise AssertionError(f"metadata id overlap between vault and reference pool: {sorted(overlap)[:4]}")

    def _project(self, q_vec: np.ndarray, bridge_w: Optional[np.ndarray], mode: str) -> np.ndarray:
        query = np.asarray(q_vec, dtype=np.float64).reshape(-1)
        if not np.isfinite(query).all():
            raise ValueError("q_vec must be finite")
        vault_dim = self.node_vecs.shape[1]
        if bridge_w is None:
            if query.shape[0] != vault_dim:
                raise ValueError("q_vec dim does not match the vault and bridge_w is missing")
            return query
        weight = np.asarray(bridge_w, dtype=np.float64)
        if weight.ndim != 2:
            raise ValueError("bridge_w must be a 2-D array")
        # Stiefel / linear map stored as W in R^{d_hi x d_lo}: q_lo = q_hi @ W.
        if weight.shape[0] == query.shape[0] and weight.shape[1] == vault_dim:
            return query @ weight
        if weight.shape[1] == query.shape[0] and weight.shape[0] == vault_dim:
            return weight @ query
        if weight.shape[0] == query.shape[0]:
            projected = query @ weight
        elif weight.shape[1] == query.shape[0]:
            projected = weight.T @ query
        else:
            raise ValueError(
                f"bridge_w shape {weight.shape} is incompatible with q_vec {query.shape} and vault dim {vault_dim}"
            )
        if projected.shape[0] != vault_dim:
            raise ValueError("bridge projection did not land in the vault dimension")
        if mode == "linear_bridge_zeromu_csls" and bridge_w is None:
            raise ValueError("linear_bridge_zeromu_csls requires bridge_w when dimensions differ")
        return projected

    def score_window_vec(
        self,
        q_vec: np.ndarray,
        *,
        lang: str = "en",
        mode: str = "zeromu_csls",
        bridge_w: Optional[np.ndarray] = None,
        active_node_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        if mode not in {"zeromu_csls", "raw_cosine", "linear_bridge_zeromu_csls"}:
            raise ValueError(f"unsupported score mode {mode!r}")
        if lang not in self.query_mean_by_lang:
            raise ValueError(f"frozen query mean for lang {lang!r} is not loaded")
        active = self._active_mask(active_node_mask)
        if mode == "raw_cosine":
            projected = self._project(q_vec, bridge_w, mode)
            vault_scores, vault_us = self._raw_cosine(projected)
            ref_us = 0
        else:
            centered_q = self._prepare_centered_query(q_vec, lang, bridge_w, mode)
            ref_term, ref_us = self._reference_rq(centered_q)
            vault_scores, vault_us = self._centered_csls(centered_q, ref_term)
        return self._pack(vault_scores, active, vault_us, ref_us)

    def _active_mask(self, mask: Optional[np.ndarray]) -> np.ndarray:
        if mask is None:
            return np.ones(len(self.node_ids), dtype=bool)
        active = np.asarray(mask, dtype=bool).reshape(-1)
        if active.shape[0] != len(self.node_ids):
            raise ValueError("active_node_mask length must equal len(node_ids)")
        return active

    def _raw_cosine(self, projected: np.ndarray) -> Tuple[np.ndarray, int]:
        started = time.perf_counter_ns()
        q_norm = float(np.linalg.norm(projected))
        node_norms = np.linalg.norm(self.node_vecs, axis=1)
        if q_norm <= 1e-12:
            scores = np.zeros(len(self.node_ids), dtype=np.float64)
        else:
            q_hat = projected / q_norm
            dots = self.node_vecs @ q_hat
            scores = np.divide(dots, node_norms, out=np.zeros_like(dots), where=node_norms > 1e-12)
        return scores, int((time.perf_counter_ns() - started) // 1000)

    def _prepare_centered_query(
        self,
        q_vec: np.ndarray,
        lang: str,
        bridge_w: Optional[np.ndarray],
        mode: str,
    ) -> np.ndarray:
        query = np.asarray(q_vec, dtype=np.float64).reshape(-1)
        mu_q = self.query_mean_by_lang[lang]
        vault_dim = self.node_vecs.shape[1]
        if bridge_w is not None and query.shape[0] != vault_dim and mu_q.shape[0] == query.shape[0]:
            return self._project(query - mu_q, bridge_w, mode)
        projected = self._project(query, bridge_w, mode)
        if mu_q.shape[0] != projected.shape[0]:
            raise ValueError(
                f"query mean for {lang!r} dim {mu_q.shape[0]} matches neither source dim {query.shape[0]} "
                f"nor vault dim {projected.shape[0]}"
            )
        return projected - mu_q

    def _reference_rq(self, centered_q: np.ndarray) -> Tuple[float, int]:
        started = time.perf_counter_ns()
        q_norm = float(np.linalg.norm(centered_q))
        if q_norm <= 1e-12:
            return 0.0, int((time.perf_counter_ns() - started) // 1000)
        q_hat = centered_q / q_norm
        ref_centered = self.ref_pool_vecs - self.doc_mean.reshape(-1)
        ref_norms = np.linalg.norm(ref_centered, axis=1)
        dots = ref_centered @ q_hat
        cosines = np.divide(dots, ref_norms, out=np.full_like(dots, -1.0), where=ref_norms > 1e-12)
        k = min(self.csls_k, cosines.shape[0])
        if k == cosines.shape[0]:
            top = cosines
        else:
            part = np.argpartition(cosines, -k)[-k:]
            top = cosines[part]
        r_q = float(np.mean(top))
        return r_q, int((time.perf_counter_ns() - started) // 1000)

    def _centered_csls(self, centered_q: np.ndarray, r_q: float) -> Tuple[np.ndarray, int]:
        started = time.perf_counter_ns()
        q_norm = float(np.linalg.norm(centered_q))
        if q_norm <= 1e-12:
            scores = np.zeros(len(self.node_ids), dtype=np.float64)
            return scores, int((time.perf_counter_ns() - started) // 1000)
        q_hat = centered_q / q_norm
        centered_docs = self.node_vecs - self.doc_mean
        dots = centered_docs @ q_hat
        norms = self.doc_centered_norms
        cosines = np.divide(dots, norms, out=np.zeros_like(dots), where=norms > 1e-12)
        scores = (2.0 * cosines) - float(r_q) - self.doc_csls_r
        return scores, int((time.perf_counter_ns() - started) // 1000)

    def _pack(
        self,
        scores: np.ndarray,
        active: np.ndarray,
        vault_us: int,
        ref_us: int,
    ) -> Dict[str, Any]:
        if not np.any(active):
            return {
                "max_score": -1.0,
                "best_node_id": None,
                "best_work_id": None,
                "gemv_vault_us": int(vault_us),
                "gemv_ref_pool_us": int(ref_us),
            }
        masked = np.where(active, scores, -np.inf)
        winner = int(np.argmax(masked))
        best = float(masked[winner])
        if not np.isfinite(best):
            return {
                "max_score": -1.0,
                "best_node_id": None,
                "best_work_id": None,
                "gemv_vault_us": int(vault_us),
                "gemv_ref_pool_us": int(ref_us),
            }
        return {
            "max_score": best,
            "best_node_id": self.node_ids[winner],
            "best_work_id": self.work_ids[winner],
            "gemv_vault_us": int(vault_us),
            "gemv_ref_pool_us": int(ref_us),
        }


class _AuditorVisibleSet(set):
    """Auditor-visible ids. AUDIT_ONLY rows are suppression records, not erasure."""

    auditable_suppression_not_erasure = True


class TakedownLineageFence:
    """Gate-1 wrapper around ``StratifiedEvidenceStore`` for tenant ``tenant_pm6``."""

    def __init__(self, *, tenant: str = "tenant_pm6") -> None:
        self.tenant = tenant
        self.store = StratifiedEvidenceStore()
        self.auditable_suppression_not_erasure = True
        self._clock = 0

    def _now(self) -> int:
        self._clock += 1
        return self._clock

    def _scope(self, event: StratifiedEvent) -> StratifiedEvent:
        if event.tenant == self.tenant:
            return event
        return replace(event, tenant=self.tenant)

    def _cross_check(self) -> Dict[str, int]:
        oracle = compute_alternating_wfs_oracle(self.store.events.keys(), self.store.refs, self.store.deps_v)
        mismatches = sum(1 for node_id, value in oracle.items() if self.store.val.get(node_id) != value)
        mismatches += sum(1 for node_id in self.store.events if node_id not in oracle)
        return {"nodes_checked": len(oracle), "mismatches": int(mismatches)}

    def ingest_corpus_records(
        self,
        events: Sequence[StratifiedEvent],
        edges: Sequence[EdgeAttestation],
    ) -> Dict[str, Any]:
        scoped = [self._scope(event) for event in events]
        delta = self.store.apply_mutation(scoped, list(edges))
        report = self._cross_check()
        if report["mismatches"] != 0:
            raise RuntimeError(f"EK5 oracle mismatch after ingest: {report}")
        return {
            "ingested_events": len(scoped),
            "ingested_edges": len(list(edges)),
            "inserted": dict(delta.inserted),
            "eligibility": {node_id: self.store.eligibility.get(node_id) for node_id in self.store.events},
            "cross_check": report,
        }

    def _descendants(self, roots: Sequence[str]) -> List[str]:
        seen: Set[str] = set()
        queue: List[str] = list(roots)
        while queue:
            node_id = queue.pop()
            for child in self.store.dep_c_rev.get(node_id, ()):
                if child not in seen:
                    seen.add(child)
                    queue.append(child)
        return sorted(seen)

    def issue_takedown(
        self,
        notice_id: str,
        target_ids: Sequence[str],
        *,
        authority_tier: int = 5,
        domain: str = "ip_governance",
        tx_time: Optional[int] = None,
    ) -> Dict[str, Any]:
        if not target_ids:
            raise ValueError("target_ids must be non-empty")
        stamp = self._now() if tx_time is None else int(tx_time)
        notice = StratifiedEvent(
            event_id=str(notice_id),
            entity="takedown_notice",
            slot="notice",
            claim_value="suppress",
            text=_ACTION_LABEL,
            valid_time=stamp,
            tx_time=stamp,
            tenant=self.tenant,
            domain=str(domain),
            authority_tier=int(authority_tier),
        )
        attestations = [
            EdgeAttestation(
                attestation_id=f"{notice_id}:refute:{index}:{target_id}",
                src_id=str(notice_id),
                dst_id=str(target_id),
                edge_type="REFUTE",
                valid_time=stamp,
                tx_time=stamp,
            )
            for index, target_id in enumerate(target_ids)
        ]
        self.store.apply_mutation([notice], attestations)
        report = self._cross_check()
        if report["mismatches"] != 0:
            raise RuntimeError(f"EK5 oracle mismatch after takedown: {report}")
        target_set = {str(item) for item in target_ids}
        refuted = [item for item in target_ids if self.store.eligibility.get(str(item)) == "AUDIT_ONLY"]
        cascaded = [
            node_id
            for node_id in self._descendants(list(target_set))
            if node_id not in target_set and self.store.eligibility.get(node_id) == "AUDIT_ONLY"
        ]
        preserved = sorted(
            src
            for src, cited in self.store.cites_audit.items()
            if cited & target_set and self.store.eligibility.get(src) == "ADMIT"
        )
        return {
            "notice_id": str(notice_id),
            "refuted_ids": [str(item) for item in refuted],
            "cascaded_ids": cascaded,
            "cites_audit_preserved": preserved,
            "auditable_suppression_not_erasure": True,
            "eligibility": {node_id: self.store.eligibility.get(node_id) for node_id in self.store.events},
            "cross_check": report,
        }

    def admissible_ids(self, *, audience: str = "general") -> Set[str]:
        if audience not in {"general", "auditor"}:
            raise ValueError("audience must be 'general' or 'auditor'")
        visible: Set[str] = set()
        for node_id, eligibility in self.store.eligibility.items():
            if eligibility == "ADMIT":
                visible.add(node_id)
            elif eligibility == "AUDIT_ONLY" and audience == "auditor":
                visible.add(node_id)
        if audience == "auditor":
            wrapped = _AuditorVisibleSet(visible)
            wrapped.auditable_suppression_not_erasure = True
            self.auditable_suppression_not_erasure = True
            return wrapped
        return visible

    def screen_prompt(
        self,
        prompt_text: str,
        restricted_entities: Mapping[str, str],
        *,
        prompt_prox_score: float = 0.0,
        tau_arm: float = 0.35,
        hard_refuse_threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        pairs = [(str(work_id), str(text)) for work_id, text in restricted_entities.items()]
        matched: List[str] = []
        if pairs:
            scanner = Tier2ACanonicalScanner(pairs)
            hit = scanner.scan_text(prompt_text)
            matched.extend(hit["matched_work_ids"])
            prompt_canon, _spans = canonicalize_for_tier2a(prompt_text)
            seen = set(matched)
            for work_id, text in pairs:
                canon, _inner = canonicalize_for_tier2a(text)
                if canon and canon in prompt_canon and work_id not in seen:
                    seen.add(work_id)
                    matched.append(work_id)
        reasons: List[str] = []
        hard = False
        if matched:
            hard = True
            reasons.append("restricted_span_overlap")
        if hard_refuse_threshold is not None and float(prompt_prox_score) >= float(hard_refuse_threshold):
            hard = True
            reasons.append("prompt_similarity_hard_refuse")
        armed = hard or float(prompt_prox_score) >= float(tau_arm)
        if hard and "restricted_span_overlap" not in reasons and "prompt_similarity_hard_refuse" not in reasons:
            reasons.append("hard_refuse")
        if not reasons and armed:
            reasons.append("armed")
        return {
            "hard_blocked": hard,
            "armed": armed,
            "matched_work_ids": matched,
            "reason": "+".join(reasons) if reasons else "clear",
        }


def _sidechannel_nonempty(value: Any) -> bool:
    if value is None or value is False:
        return False
    if isinstance(value, (str, bytes, list, dict, tuple, set)) and len(value) == 0:
        return False
    return True


def _strip_sidechannels(value: Any, hits: List[str], prefix: str) -> Any:
    if isinstance(value, dict):
        cleaned: Dict[str, Any] = {}
        for key, inner in value.items():
            if key in _SIDECHANNEL_KEYS:
                if _sidechannel_nonempty(inner):
                    hits.append(f"{prefix}{key}")
                continue
            cleaned[key] = _strip_sidechannels(inner, hits, f"{prefix}{key}.")
        return cleaned
    if isinstance(value, list):
        return [_strip_sidechannels(item, hits, f"{prefix}[]") for item in value]
    return value


def sanitize_sse_event_dict(
    event: Mapping[str, Any],
    *,
    strict_reject_sidechannels: bool = False,
) -> Dict[str, Any]:
    """Strip logprobs, tool calls, and reasoning fields from an SSE payload."""
    if not isinstance(event, Mapping):
        raise TypeError("event must be a mapping")
    hits: List[str] = []
    cleaned = _strip_sidechannels(dict(event), hits, "")
    if strict_reject_sidechannels and hits:
        raise ValueError(f"non-empty side-channel tokens present: {hits}")
    return cleaned


@dataclass
class _Token:
    text: str
    byte_start: int
    byte_end: int


class ClosedLoopStreamGate:
    """Verdict-ack UTF-8 release gate. Profile A never emits a byte a window still covers."""

    def __init__(
        self,
        profile: str = "profile_a_strict",
        *,
        tier2a: Optional[Tier2ACanonicalScanner] = None,
        tier2b_scorer: Optional[Callable[[str], Tuple[bool, float, Optional[str], int]]] = None,
        w_2b_tokens: int = 64,
        s_2b_tokens: int = 16,
        s_2a_tokens: int = 1,
        h0_holdback_tokens: int = 12,
        armed: bool = False,
        tier2b_latency_tokens: int = 0,
        strict_reject_sidechannels: bool = False,
    ) -> None:
        if profile not in STREAM_PROFILES:
            raise ValueError(f"profile must be one of {STREAM_PROFILES}")
        if w_2b_tokens < 1 or s_2b_tokens < 1 or s_2a_tokens < 1:
            raise ValueError("window and stride sizes must be >= 1")
        if h0_holdback_tokens < 0 or tier2b_latency_tokens < 0:
            raise ValueError("holdback and latency must be >= 0")
        self.profile = profile
        self.tier2a = tier2a
        self.tier2b_scorer = tier2b_scorer
        self.w_2b_tokens = int(w_2b_tokens)
        self.s_2b_tokens = int(s_2b_tokens)
        self.s_2a_tokens = int(s_2a_tokens)
        self.h0_holdback_tokens = int(h0_holdback_tokens)
        self.armed = bool(armed)
        self.tier2b_latency_tokens = int(tier2b_latency_tokens)
        self.strict_reject_sidechannels = bool(strict_reject_sidechannels)
        self.stream = bytearray()
        self._utf8_tail = bytearray()
        self.tokens: List[_Token] = []
        self.released_byte_cursor = 0
        self.released_utf8_bytes = 0
        self.suppressed_utf8_bytes = 0
        self.flagged_window_bytes_leaked = 0
        self.unarmed_exposed_tokens = 0
        self.first_byte_ttft_us = 0
        self.windows_checked_2a = 0
        self.windows_checked_2b = 0
        self.aborted = False
        self.matched_work_id: Optional[str] = None
        self.max_score_ppm = 0
        self.last_sse_event: Optional[Dict[str, Any]] = None
        self._t0_ns: Optional[int] = None
        self._ttft_set = False
        self._did_discard = False
        self._2a_next = 0
        self._next_full_k = 0
        self._2b_seen: Set[Tuple[int, int]] = set()
        self._2b_done: Set[Tuple[int, int]] = set()
        self._window_verdict: Dict[int, str] = {}
        self._pending: List[Tuple[int, int, int]] = []
        self._emitted_token_count = 0
        self._unarmed_released_idxs: Set[int] = set()

    def _strict_2b(self) -> bool:
        return self.profile == "profile_a_strict" or self.armed

    def _touch_clock(self) -> None:
        if self._t0_ns is None:
            self._t0_ns = time.perf_counter_ns()

    def escalate_to_armed(self) -> Dict[str, Any]:
        """Switch immediately to synchronous Profile A holdback."""
        self.armed = True
        if not self.aborted:
            self._drain_pending(force=True)
        return self.summary()

    def push_chunk(self, text_chunk: str, *, sse_meta: Optional[Mapping[str, Any]] = None) -> bytes:
        self._touch_clock()
        if sse_meta is not None:
            self.last_sse_event = sanitize_sse_event_dict(
                sse_meta,
                strict_reject_sidechannels=self.strict_reject_sidechannels,
            )
        if self.aborted:
            return b""
        self._absorb(_chunk_to_bytes(text_chunk))
        self._sync_tokens(final=False)
        self._run_checks(final=False)
        if self.aborted:
            return b""
        return self._emit(eos=False)

    def finalize_eos(self) -> bytes:
        self._touch_clock()
        if self.aborted:
            self._discard_unreleased()
            return b""
        self._flush_tail(final=True)
        self._sync_tokens(final=True)
        self._run_checks(final=True)
        if self.aborted:
            return b""
        return self._emit(eos=True)

    def abort_on_error_or_cancel(self) -> None:
        self.aborted = True
        self._discard_unreleased()

    def summary(self) -> Dict[str, Any]:
        return {
            "released_utf8_bytes": int(self.released_utf8_bytes),
            "suppressed_utf8_bytes": int(self.suppressed_utf8_bytes),
            "flagged_window_bytes_leaked": int(self.flagged_window_bytes_leaked),
            "unarmed_exposed_tokens": int(self.unarmed_exposed_tokens),
            "first_byte_ttft_us": int(self.first_byte_ttft_us),
            "windows_checked_2a": int(self.windows_checked_2a),
            "windows_checked_2b": int(self.windows_checked_2b),
            "aborted": bool(self.aborted),
            "matched_work_id": self.matched_work_id,
            "max_score_ppm": int(self.max_score_ppm),
            "profile": self.profile,
            "armed": bool(self.armed),
            "released_byte_cursor": int(self.released_byte_cursor),
        }

    def _absorb(self, data: bytes) -> None:
        if not data:
            return
        self._utf8_tail.extend(data)
        self._flush_tail(final=False)

    def _flush_tail(self, final: bool) -> None:
        if not self._utf8_tail:
            return
        decoder = codecs.getincrementaldecoder("utf-8")()
        try:
            text = decoder.decode(bytes(self._utf8_tail), final=final)
        except UnicodeDecodeError:
            if final:
                self.suppressed_utf8_bytes += len(self._utf8_tail)
                self._utf8_tail.clear()
            return
        pending, _flag = decoder.getstate()
        if final:
            pending = b""
        consumed = len(self._utf8_tail) - len(pending)
        if consumed < 0:
            consumed = 0
        del self._utf8_tail[:consumed]
        if text:
            self.stream.extend(text.encode("utf-8"))

    def _sync_tokens(self, final: bool) -> None:
        text = self.stream.decode("utf-8")
        tokens: List[_Token] = []
        for match in re.finditer(r"\S+", text):
            end = match.end()
            if not final and (end >= len(text) or not text[end].isspace()):
                break
            byte_start = len(text[: match.start()].encode("utf-8"))
            byte_end = len(text[:end].encode("utf-8"))
            tokens.append(_Token(match.group(), byte_start, byte_end))
        self.tokens = tokens

    def _run_checks(self, final: bool) -> None:
        self._run_2a(final=final)
        if self.aborted:
            return
        self._schedule_closed_windows()
        if final:
            self._schedule_eos_partials()
        self._drain_pending(force=final or self._strict_2b())

    def _run_2a(self, final: bool) -> None:
        if self.tier2a is None:
            self._2a_next = len(self.tokens)
            return
        stride = self.s_2a_tokens
        token_count = len(self.tokens)
        cursor = self._2a_next
        while cursor < token_count:
            boundary = (cursor + 1) % stride == 0 or (final and cursor == token_count - 1)
            if not boundary:
                if not final:
                    break
                cursor += 1
                continue
            self._scan_2a(max(0, cursor - 96), cursor + 1)
            self.windows_checked_2a += 1
            self._2a_next = cursor + 1
            if self.aborted:
                return
            cursor += 1

    def _scan_2a(self, lo: int, hi: int) -> None:
        if hi <= lo or self.tier2a is None:
            return
        start = self.tokens[lo].byte_start
        end = self.tokens[hi - 1].byte_end
        text = bytes(self.stream[start:end]).decode("utf-8")
        try:
            result = self.tier2a.scan_text(text)
        except Exception:
            self.abort_on_error_or_cancel()
            raise
        self._note_score(float(result.get("score", 0.0)), 0, None)
        if not result.get("flagged"):
            return
        raw_start = result.get("flagged_raw_byte_start")
        raw_end = result.get("flagged_raw_byte_end")
        abs_start = None if raw_start is None else start + int(raw_start)
        abs_end = None if raw_end is None else start + int(raw_end)
        work_ids = result.get("matched_work_ids") or []
        work_id = str(work_ids[0]) if work_ids else None
        self._note_score(float(result.get("score", 1.0)), 0, work_id)
        self._leak_span(abs_start, abs_end)
        self.aborted = True
        self._discard_unreleased()

    def _schedule_closed_windows(self) -> None:
        token_count = len(self.tokens)
        width = self.w_2b_tokens
        stride = self.s_2b_tokens
        if token_count < width:
            return
        k_max = (token_count - width) // stride
        while self._next_full_k <= k_max:
            if self.aborted:
                return
            lo = self._next_full_k * stride
            hi = lo + width
            self._enqueue_2b(lo, hi, force_sync=self._strict_2b())
            self._next_full_k += 1

    def _schedule_eos_partials(self) -> None:
        token_count = len(self.tokens)
        if token_count <= 0 or self.aborted:
            return
        stride = self.s_2b_tokens
        # A short stream never closes a full W-token window. Every stride that
        # still overlaps the suffix is a real covering window and must be scored
        # before those bytes can move.
        if token_count < self.w_2b_tokens:
            k = 0
        else:
            k = self._next_full_k
        while True:
            lo = k * stride
            if lo >= token_count:
                break
            self._enqueue_2b(lo, token_count, force_sync=True)
            if self.aborted:
                return
            k += 1

    def _enqueue_2b(self, lo: int, hi: int, force_sync: bool) -> None:
        key = (lo, hi)
        if key in self._2b_seen:
            if force_sync:
                self._apply_2b(lo, hi)
            return
        self._2b_seen.add(key)
        if force_sync:
            self._apply_2b(lo, hi)
            return
        apply_at = hi + self.tier2b_latency_tokens
        self._pending.append((apply_at, lo, hi))

    def _drain_pending(self, force: bool) -> None:
        token_count = len(self.tokens)
        kept: List[Tuple[int, int, int]] = []
        for apply_at, lo, hi in self._pending:
            if self.aborted:
                break
            if force or token_count >= apply_at:
                self._apply_2b(lo, hi)
            else:
                kept.append((apply_at, lo, hi))
        self._pending = [] if self.aborted else kept

    def _apply_2b(self, lo: int, hi: int) -> None:
        key = (lo, hi)
        if key in self._2b_done or self.aborted:
            return
        self._2b_done.add(key)
        if hi <= lo:
            self._mark_window(lo, "CLEAR")
            return
        text = bytes(self.stream[self.tokens[lo].byte_start : self.tokens[hi - 1].byte_end]).decode("utf-8")
        self.windows_checked_2b += 1
        if self.tier2b_scorer is None:
            self._mark_window(lo, "CLEAR")
            return
        try:
            flagged, score, work_id, ppm = self.tier2b_scorer(text)
        except Exception:
            self.abort_on_error_or_cancel()
            raise
        self._note_score(float(score), int(ppm), None if work_id is None else str(work_id))
        if flagged:
            self._flag_window(lo, hi, None if work_id is None else str(work_id))
            return
        self._mark_window(lo, "CLEAR")

    def _mark_window(self, lo: int, verdict: str) -> None:
        stride = self.s_2b_tokens
        if lo % stride != 0:
            return
        slot = lo // stride
        if self._window_verdict.get(slot) == "FLAGGED":
            return
        self._window_verdict[slot] = verdict

    def _covering_slots(self, token_index: int) -> range:
        stride = self.s_2b_tokens
        width = self.w_2b_tokens
        k_hi = token_index // stride
        k_lo = (token_index - width) // stride + 1
        if k_lo < 0:
            k_lo = 0
        if k_lo > k_hi:
            return range(0, 0)
        return range(k_lo, k_hi + 1)

    def _token_2b_clear(self, token_index: int) -> bool:
        for slot in self._covering_slots(token_index):
            if self._window_verdict.get(slot) != "CLEAR":
                return False
        return True

    def _note_score(self, score: float, ppm: int, work_id: Optional[str]) -> None:
        derived = int(round(float(score) * 1_000_000))
        self.max_score_ppm = max(self.max_score_ppm, derived, int(ppm))
        if work_id and self.matched_work_id is None:
            self.matched_work_id = work_id

    def _leak_span(self, start: Optional[int], end: Optional[int]) -> None:
        if start is None:
            return
        limit = len(self.stream) if end is None else int(end)
        if self.released_byte_cursor > int(start):
            self.flagged_window_bytes_leaked += max(0, min(self.released_byte_cursor, limit) - int(start))

    def _flag_window(self, lo: int, hi: int, work_id: Optional[str]) -> None:
        self._mark_window(lo, "FLAGGED")
        leaked = 0
        exposed = 0
        last = min(hi, len(self.tokens))
        for token_index in range(lo, last):
            if token_index < self._emitted_token_count:
                token = self.tokens[token_index]
                leaked += token.byte_end - token.byte_start
            if token_index in self._unarmed_released_idxs:
                exposed += 1
        self.flagged_window_bytes_leaked += leaked
        self.unarmed_exposed_tokens += exposed
        if work_id and self.matched_work_id is None:
            self.matched_work_id = work_id
        self.aborted = True
        self._discard_unreleased()

    def _prefix_releasable(self, eos: bool) -> int:
        token_count = len(self.tokens)
        if self.aborted or token_count == 0:
            return 0
        cleared = 0
        holdback = 0 if eos or self._strict_2b() else self.h0_holdback_tokens
        limit = token_count - holdback
        if limit < 0:
            limit = 0
        for token_index in range(limit):
            if token_index >= self._2a_next and self.tier2a is not None:
                break
            if self._strict_2b() and not self._token_2b_clear(token_index):
                break
            cleared = token_index + 1
        return cleared

    def _emit(self, eos: bool) -> bytes:
        if self.aborted:
            return b""
        releasable = self._prefix_releasable(eos)
        if releasable <= 0:
            return b""
        end_byte = self.tokens[releasable - 1].byte_end
        if eos and releasable == len(self.tokens):
            end_byte = len(self.stream)
        if end_byte <= self.released_byte_cursor:
            return b""
        chunk = bytes(self.stream[self.released_byte_cursor : end_byte])
        if self.profile == "profile_b_armed_async" and not self.armed:
            for token_index in range(self._emitted_token_count, releasable):
                self._unarmed_released_idxs.add(token_index)
        self._emitted_token_count = max(self._emitted_token_count, releasable)
        self.released_byte_cursor = end_byte
        self.released_utf8_bytes += len(chunk)
        if chunk and not self._ttft_set and self._t0_ns is not None:
            self.first_byte_ttft_us = int((time.perf_counter_ns() - self._t0_ns) // 1000)
            self._ttft_set = True
        return chunk

    def _discard_unreleased(self) -> None:
        if self._did_discard:
            return
        self._did_discard = True
        pending = len(self.stream) - self.released_byte_cursor
        pending += len(self._utf8_tail)
        if pending > 0:
            self.suppressed_utf8_bytes += pending
        self._utf8_tail.clear()


def _chunk_to_bytes(text_chunk: str) -> bytes:
    if isinstance(text_chunk, bytes):
        return text_chunk
    if isinstance(text_chunk, bytearray):
        return bytes(text_chunk)
    if not isinstance(text_chunk, str):
        raise TypeError("text_chunk must be a str")
    return text_chunk.encode("utf-8", errors="surrogateescape")


def _require_int(name: str, value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{name} must be an int")
    return int(value)


class TakedownReceiptLedger:
    """Ed25519 + SHA-256 hash-chained ``copilot_receipt_v2`` ledger. No floats."""

    def __init__(
        self,
        *,
        private_key: Optional[Ed25519PrivateKey] = None,
        clock_us: Optional[Callable[[], int]] = None,
    ) -> None:
        self._key = private_key or Ed25519PrivateKey.generate()
        raw = self._key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        self.public_key_b64 = base64.b64encode(raw).decode("ascii")
        self.key_id = hashlib.sha256(raw).hexdigest()[:16]
        self._clock_us = clock_us or (lambda: time.time_ns() // 1000)
        self.ledger: List[Dict[str, Any]] = []

    def issue_receipt(
        self,
        *,
        mode: str,
        notice_id: str,
        score_ppm: int,
        threshold_ppm: int,
        released_bytes: int,
        suppressed_bytes: int,
        latency_us: int,
        records_after: Mapping[str, str],
        matched_work_id: str = "",
        action: str = _ACTION_LABEL,
    ) -> Dict[str, Any]:
        if mode not in {"non_nda", "nda"}:
            raise ValueError("mode must be 'non_nda' or 'nda'")
        public_records: Dict[str, str] = {}
        raw_records: Dict[str, str] = {}
        for record_id, state in records_after.items():
            label = _PUBLIC_STATE.get(str(state))
            if label is None:
                raise ValueError(f"unknown record state {state!r}")
            public_records[str(record_id)] = label
            raw_records[str(record_id)] = str(state)
        payload: Dict[str, Any] = {
            "schema": "copilot_receipt_v2",
            "receipt_kind": "takedown_suppression_receipt",
            "mode": mode,
            "action": str(action),
            "notice_id": str(notice_id),
            "matched_work_id": str(matched_work_id),
            "score_ppm": _require_int("score_ppm", score_ppm),
            "threshold_ppm": _require_int("threshold_ppm", threshold_ppm),
            "released_bytes": _require_int("released_bytes", released_bytes),
            "suppressed_bytes": _require_int("suppressed_bytes", suppressed_bytes),
            "latency_us": _require_int("latency_us", latency_us),
            "records_after": public_records,
            "auditable_suppression_not_erasure": True,
            "key_id": self.key_id,
            "disclaimer": "Not legal advice. This receipt records inference-time takedown suppression of synthetic records.",
        }
        if mode == "nda":
            payload["internal"] = {"eligibility_raw": raw_records}
        prev = self.ledger[-1]["entry_hash"] if self.ledger else _GENESIS_HASH
        payload["seq"] = len(self.ledger) + 1
        payload["prev_hash"] = prev
        payload["issued_at_us"] = _require_int("issued_at_us", int(self._clock_us()))
        if mode == "non_nda":
            assert_non_nda_clean(canonical_json(payload).decode("ascii"), source_label="copilot_receipt_v2")
        entry_hash = hashlib.sha256(prev.encode("ascii") + canonical_json(payload)).hexdigest()
        signature = self._key.sign(entry_hash.encode("ascii"))
        entry = {
            "payload": payload,
            "prev_hash": prev,
            "entry_hash": entry_hash,
            "signature_b64": base64.b64encode(signature).decode("ascii"),
            "public_key_b64": self.public_key_b64,
            "algorithm": "Ed25519 over the ASCII hex of SHA-256(prev_hash || canonical_json(payload))",
        }
        self.ledger.append(entry)
        return entry

    def verify_receipt(self, entry: Any) -> Dict[str, Any]:
        reasons: List[str] = []
        checks = {"hash_ok": False, "signature_ok": False, "key_ok": False, "chain_link_ok": False}
        try:
            payload = entry["payload"]
            prev = entry["prev_hash"]
            claimed_hash = entry["entry_hash"]
            signature = base64.b64decode(entry["signature_b64"], validate=True)
            public_b64 = entry["public_key_b64"]
        except (KeyError, TypeError, ValueError):
            return {"valid": False, "checks": checks, "in_ledger": False, "reasons": ["malformed receipt"]}
        try:
            recomputed = hashlib.sha256(str(prev).encode("ascii") + canonical_json(payload)).hexdigest()
        except (ValueError, TypeError, UnicodeEncodeError):
            return {"valid": False, "checks": checks, "in_ledger": False, "reasons": ["payload is not canonicalizable"]}
        checks["hash_ok"] = isinstance(claimed_hash, str) and hmac.compare_digest(recomputed, claimed_hash)
        if not checks["hash_ok"]:
            reasons.append("hash does not match payload")
        checks["key_ok"] = public_b64 == self.public_key_b64 and isinstance(payload, dict) and payload.get("key_id") == self.key_id
        if not checks["key_ok"]:
            reasons.append("receipt was not signed by this ledger key")
        try:
            Ed25519PublicKey.from_public_bytes(base64.b64decode(self.public_key_b64)).verify(
                signature,
                recomputed.encode("ascii"),
            )
            checks["signature_ok"] = checks["hash_ok"]
        except (InvalidSignature, ValueError):
            reasons.append("signature invalid")
        checks["chain_link_ok"] = isinstance(payload, dict) and payload.get("prev_hash") == prev
        if not checks["chain_link_ok"]:
            reasons.append("payload prev_hash differs from entry prev_hash")
        in_ledger = False
        if isinstance(payload, dict) and isinstance(payload.get("seq"), int) and 1 <= payload["seq"] <= len(self.ledger):
            in_ledger = self.ledger[payload["seq"] - 1]["entry_hash"] == claimed_hash
        return {"valid": all(checks.values()), "checks": checks, "in_ledger": in_ledger, "reasons": reasons}

    def verify_ledger(self) -> Dict[str, Any]:
        prev = _GENESIS_HASH
        for index, entry in enumerate(self.ledger, start=1):
            result = self.verify_receipt(entry)
            payload = entry.get("payload", {})
            if not result["valid"] or entry.get("prev_hash") != prev or payload.get("seq") != index:
                return {"intact": False, "length": len(self.ledger), "first_bad_seq": index}
            prev = entry["entry_hash"]
        return {"intact": True, "length": len(self.ledger), "head_hash": prev}
