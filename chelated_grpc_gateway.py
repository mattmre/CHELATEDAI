r"""Drop-In Bidirectional Chelated gRPC Interceptor Gateway (Wave-6 Track E / Demo D5, Guardrail K1).

Implements an OpenSearch 3.x `SearchService.Search` (`KnnQuery`) binary Protobuf gRPC gateway
compiled dynamically via `grpc_tools.protoc` and wire-executed over a live Qdrant HNSW gRPC
backend (`:6334` / `:16334`).

Mandatory K1 guarantees enforced by this module:
  1. Real `grpcio` + `grpc_tools.protoc` binary Protobuf `SearchService.Search` carrying packed
     IEEE-754 `repeated float vector`, `k`, `k_os`, `ef_search`, `admitted_bitset`, `epoch_pin`,
     and returning zero vectors — only `(id, point_id, raw_dot, score, shard_tag, epistemic_state,
     certified_stop)` over HTTP/2.
  2. Fail-closed `TRA-1` ingress validation (`resolve_with_tra1`, `weights_sha256`,
     `min_fit_count >= 32`, `min_validation_count >= 16`, `minimum_route_confidence >= 0.70`,
     `index_generation` match, and pre-normalization retained-subspace energy guard) plus
     strictly causal pre-query EMA `q_hat = norm2((q - mu_q) @ Q)` without `+mu_y`.
  3. Zero-vector egress rescoring from a resident `8 B/doc` sidecar (`4 B` float32
     `||d_j - mu_d||_2` + `4 B` float32 causal-DEV `CSLS` `r_d`), per-shard `CSLS`
     (`2s - r_d - r_q`), dual-shard HTTP/2 fanout (`legacy_384d` + `native_4096d`), and
     Certified Shortlist Stop (`unseen_upper_bound < s_k` maximizing over both positive norm
     endpoints `[n_min, n_max]` and subtracting `r_{d,min}` + `r_q` lower bound, with bounded
     refill and Qdrant `exact=True` zero-vector fallback).
  4. Decoupled `(k_os, hnsw_ef)` with pre-filtered admission bitsets, adaptive
     `ef_eff = min(ef_max, ceil(k_os / max(p_admit, 0.05)))`, exact fallback when
     `p_admit <= 0.10`, and mandatory egress `EK5 v3` epoch-pinned read fencing (`wait=true`,
     tri-valued `ADMITTED / CONTESTED / AUDIT_ONLY`) achieving `0.00%` stale resurrection even
     under replica lag.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import sys
import tempfile
import threading
import time
from concurrent import futures
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import google.protobuf
import grpc
from grpc_tools import protoc
import numpy as np

from benchmark_utils import canonicalize_id
from ek5_v3_stratified import EdgeAttestation, StratifiedEvent, StratifiedEvidenceStore
from representation_space import (
    BridgeContract,
    DistanceMetric,
    Normalization,
    PoolingStrategy,
    RepresentationDescriptor,
    RepresentationRole,
    ResidentIndexDescriptor,
    VectorDType,
)
from semantic_cache_resolver import ResolutionDecision, ResolutionMode
from spherical_migration_bridge import (
    CenteredProcrustesBridge,
    build_bridge_contract,
    l2_normalize,
    resolve_with_tra1,
)

BACKEND_EXECUTION_LABEL = (
    "OpenSearch_3x_SearchService_KnnQuery_Protobuf_wire_executed_over_live_Qdrant_HNSW_gRPC"
)

EP_ADMITTED = "ADMITTED"
EP_CONTESTED = "CONTESTED"
EP_AUDIT_ONLY = "AUDIT_ONLY"

SCORING_BIDIR_ZEROMU = "bidirectional_zeromu"
SCORING_BIDIR_ZEROMU_CSLS = "bidirectional_zeromu_csls"
SCORING_REQUEST_ONLY_RAW_DOT = "request_only_raw_dot"
SCORING_QUERY_AFFINE = "query_affine"
SCORING_NATIVE_DOT = "native_dot"

VALID_SCORING_MODES = frozenset({
    SCORING_BIDIR_ZEROMU,
    SCORING_BIDIR_ZEROMU_CSLS,
    SCORING_REQUEST_ONLY_RAW_DOT,
    SCORING_QUERY_AFFINE,
    SCORING_NATIVE_DOT,
})

OPENSEARCH_KNN_GATEWAY_PROTO = r"""syntax = "proto3";

package opensearch.protobufs.services;

message KnnQuery {
  string field = 1;
  repeated float vector = 2;
  uint32 k = 3;
  uint32 k_os = 4;
  uint32 ef_search = 5;
  bytes admitted_bitset = 6;
  uint64 epoch_pin = 7;
  string query_id = 8;
  repeated float native_vector = 9;
}

message SearchRequest {
  repeated string indices = 1;
  KnnQuery knn = 2;
  repeated KnnQuery knn_batch = 3;
  string tenant_id = 4;
  string query_space_id = 5;
  string bridge_contract_id = 6;
  string bridge_weights_sha256 = 7;
  string index_generation = 8;
  uint64 epoch_pin = 9;
  bool wait_for_epoch = 10;
  string scoring_mode = 11;
  bool enable_certified_stop = 12;
  bool dual_shard_fanout = 13;
  bool adaptive_ef = 14;
  bool allow_exact_fallback = 15;
  bool update_causal_ema = 16;
  bool require_exact_prefix_certificate = 17;
  bool reset_causal_ema = 18;
}

message SearchHit {
  string id = 1;
  uint32 point_id = 2;
  float raw_dot = 3;
  float score = 4;
  string shard_tag = 5;
  string epistemic_state = 6;
  bool certified_stop = 7;
}

message QuerySearchResult {
  string query_id = 1;
  repeated SearchHit hits = 2;
  repeated SearchHit contested_hits = 3;
  bool certified_stop = 4;
  string stop_reason = 5;
  string certificate_mode = 6;
  uint64 fenced_epoch = 7;
  uint32 effective_k_os = 8;
  uint32 effective_ef = 9;
  uint32 refill_rounds = 10;
  bool used_exact_fallback = 11;
  float retained_energy_ratio = 12;
  float unseen_csls_upper_bound = 13;
  float kth_admitted_score = 14;
  double ingress_guard_ms = 15;
  double ann_rtt_ms = 16;
  double egress_rescore_ms = 17;
}

message SearchResponse {
  repeated SearchHit hits = 1;
  repeated SearchHit contested_hits = 2;
  bool certified_stop = 3;
  string stop_reason = 4;
  string certificate_mode = 5;
  uint64 fenced_epoch = 6;
  uint32 effective_k_os = 7;
  uint32 effective_ef = 8;
  uint32 refill_rounds = 9;
  bool used_exact_fallback = 10;
  float retained_energy_ratio = 11;
  float unseen_csls_upper_bound = 12;
  float kth_admitted_score = 13;
  double ingress_guard_ms = 14;
  double ann_rtt_ms = 15;
  double egress_rescore_ms = 16;
  string backend_execution_label = 17;
  repeated QuerySearchResult batch_results = 18;
}

service SearchService {
  rpc Search(SearchRequest) returns (SearchResponse);
}
"""


class TRA1AuthorizationError(RuntimeError):
    """Raised when fail-closed TRA-1 ingress checks reject a query."""


@dataclass(frozen=True)
class CompiledProtoBundle:
    pb2: ModuleType
    pb2_grpc: ModuleType
    proto_sha256: str
    protoc_exit_code: int
    grpc_version: str
    protobuf_version: str
    build_dir: str


_PROTO_LOCK = threading.Lock()
_CACHED_PROTO_BUNDLE: Optional[CompiledProtoBundle] = None


def _load_module_from_file(module_name: str, file_path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create module spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def compile_opensearch_knn_proto(*, force_recompile: bool = False) -> CompiledProtoBundle:
    """Compile OpenSearch 3.x `SearchService.Search` (`KnnQuery`) `.proto` via `grpc_tools.protoc`."""
    global _CACHED_PROTO_BUNDLE
    with _PROTO_LOCK:
        if _CACHED_PROTO_BUNDLE is not None and not force_recompile:
            return _CACHED_PROTO_BUNDLE

        proto_bytes = OPENSEARCH_KNN_GATEWAY_PROTO.encode("utf-8")
        proto_sha = hashlib.sha256(proto_bytes).hexdigest()
        env_sig = f"{proto_sha}|{sys.version_info[:2]}|{grpc.__version__}|{google.protobuf.__version__}"
        env_hash = hashlib.sha256(env_sig.encode("utf-8")).hexdigest()[:16]

        build_dir = Path(tempfile.gettempdir()) / f"chelated_d5_proto_{env_hash}"
        build_dir.mkdir(parents=True, exist_ok=True)
        base_name = f"opensearch_knn_d5_{env_hash}"
        proto_path = build_dir / f"{base_name}.proto"
        pb2_path = build_dir / f"{base_name}_pb2.py"
        pb2_grpc_path = build_dir / f"{base_name}_pb2_grpc.py"

        if force_recompile or not (pb2_path.is_file() and pb2_grpc_path.is_file()):
            proto_path.write_text(OPENSEARCH_KNN_GATEWAY_PROTO, encoding="utf-8")
            rc = int(
                protoc.main([
                    "grpc_tools.protoc",
                    f"-I{build_dir}",
                    f"--python_out={build_dir}",
                    f"--grpc_python_out={build_dir}",
                    str(proto_path),
                ])
            )
            if rc != 0:
                raise RuntimeError(f"grpc_tools.protoc failed with exit code {rc}")
        else:
            rc = 0

        if str(build_dir) not in sys.path:
            sys.path.insert(0, str(build_dir))

        pb2_mod = _load_module_from_file(f"{base_name}_pb2", pb2_path)
        pb2_grpc_mod = _load_module_from_file(f"{base_name}_pb2_grpc", pb2_grpc_path)

        bundle = CompiledProtoBundle(
            pb2=pb2_mod,
            pb2_grpc=pb2_grpc_mod,
            proto_sha256=proto_sha,
            protoc_exit_code=rc,
            grpc_version=str(grpc.__version__),
            protobuf_version=str(google.protobuf.__version__),
            build_dir=str(build_dir),
        )
        _CACHED_PROTO_BUNDLE = bundle
        return bundle


# ---------------------------------------------------------------------------
# Packed Bitset & Adaptive HNSW ef Helpers (Guardrail 3)
# ---------------------------------------------------------------------------
def pack_admitted_bitset(admitted_mask: np.ndarray) -> bytes:
    """Pack a boolean admission array `[N]` into `ceil(N / 8)` little-endian bytes."""
    mask_bool = np.asarray(admitted_mask, dtype=np.uint8).reshape(-1)
    return np.packbits(mask_bool, bitorder="little").tobytes()


def unpack_admitted_bitset(packed: bytes, n_docs: int) -> np.ndarray:
    """Unpack `ceil(N / 8)` little-endian bitset bytes back into a boolean mask `[n_docs]`."""
    if not packed:
        return np.ones(int(n_docs), dtype=bool)
    raw = np.frombuffer(packed, dtype=np.uint8)
    bits = np.unpackbits(raw, bitorder="little")
    if bits.size < int(n_docs):
        out = np.zeros(int(n_docs), dtype=bool)
        out[: bits.size] = bits.astype(bool)
        return out
    return bits[: int(n_docs)].astype(bool)


def compute_adaptive_ef(
    k_os: int,
    requested_ef: int,
    p_admit: float,
    *,
    ef_max: int = 256,
    adaptive: bool = True,
    exact_fallback_threshold: float = 0.10,
) -> Tuple[int, bool]:
    """Compute decoupled/adaptive `ef_eff = min(ef_max, ceil(k_os / max(p_admit, 0.05)))` + exact fallback."""
    p_clamped = max(float(p_admit), 0.05)
    base_ef = max(int(requested_ef), int(k_os)) if int(requested_ef) <= 0 else int(requested_ef)
    if adaptive:
        scaled = int(math.ceil(float(k_os) / p_clamped))
        ef_eff = min(int(ef_max), max(base_ef, scaled))
    else:
        ef_eff = min(int(ef_max), base_ef)
    use_exact = bool(float(p_admit) <= float(exact_fallback_threshold))
    return ef_eff, use_exact


# ---------------------------------------------------------------------------
# Resident 8 B/doc Zero-Vector Sidecar & Certified Shortlist Stop (Guardrail 2)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ShardZeroMuCslsSidecar:
    """Resident `8 B/doc` sidecar (`4 B` `||d_j - mu_d||_2` + `4 B` causal-DEV `CSLS` `r_d`).

    Holds ZERO document vectors in gateway RAM.
    """

    shard_tag: str
    collection_name: str
    doc_ids: Tuple[str, ...]
    mu_d: np.ndarray               # float32 [D] (zeros for uncentered native cosine shard)
    doc_centered_norms: np.ndarray  # float32 [M] (4 B/doc)
    csls_r_d: np.ndarray           # float32 [M] (4 B/doc)
    csls_k: int = 10
    zero_mu_active: bool = True
    n_min_all: float = 1.0
    n_max_all: float = 1.0
    r_d_min_all: float = 0.0

    def __post_init__(self) -> None:
        if self.doc_centered_norms.dtype != np.float32:
            raise ValueError("doc_centered_norms must be float32 (4 B/doc)")
        if self.csls_r_d.dtype != np.float32:
            raise ValueError("csls_r_d must be float32 (4 B/doc)")
        if self.mu_d.dtype != np.float32:
            raise ValueError("mu_d must be float32")
        n = len(self.doc_ids)
        if self.doc_centered_norms.shape != (n,) or self.csls_r_d.shape != (n,):
            raise ValueError("sidecar arrays must match len(doc_ids)")
        if int(self.doc_centered_norms.nbytes + self.csls_r_d.nbytes) != 8 * n:
            raise ValueError("sidecar payload must be strictly 8 B/doc")

    @property
    def n_docs(self) -> int:
        return len(self.doc_ids)

    @property
    def dim(self) -> int:
        return int(self.mu_d.shape[0])

    @property
    def sidecar_bytes_per_doc(self) -> int:
        if self.n_docs == 0:
            return 8
        return int((self.doc_centered_norms.nbytes + self.csls_r_d.nbytes) // self.n_docs)

    @property
    def total_sidecar_bytes(self) -> int:
        return int(self.doc_centered_norms.nbytes + self.csls_r_d.nbytes)

    @property
    def shared_centroid_bytes(self) -> int:
        return int(self.mu_d.nbytes)

    def rescore_from_raw_dots(
        self,
        point_ids: np.ndarray,
        raw_dots: np.ndarray,
        q_vec: np.ndarray,
    ) -> Tuple[np.ndarray, float]:
        """Compute `(raw_dot - q_vec · mu_d) / ||d_j - mu_d||_2` using ONLY raw DOTs and 4 B norms."""
        if point_ids.size == 0:
            return np.zeros(0, dtype=np.float32), 0.0
        q_f32 = np.asarray(q_vec, dtype=np.float32).reshape(-1)
        if self.zero_mu_active:
            c_shift = float(np.dot(q_f32, self.mu_d))
            norms = self.doc_centered_norms[point_ids]
            scores = (np.asarray(raw_dots, dtype=np.float32) - np.float32(c_shift)) / norms
            return scores.astype(np.float32), c_shift
        return np.asarray(raw_dots, dtype=np.float32).copy(), 0.0

    def envelope_for_mask(self, admitted_mask: Optional[np.ndarray] = None) -> Tuple[float, float, float]:
        """Return `(n_min, n_max, r_d_min)` over currently admitted shard points."""
        if admitted_mask is None:
            return self.n_min_all, self.n_max_all, self.r_d_min_all
        mask = np.asarray(admitted_mask, dtype=bool)
        if mask.shape[0] != self.n_docs or not np.any(mask):
            return self.n_min_all, self.n_max_all, self.r_d_min_all
        norms = self.doc_centered_norms[mask]
        rds = self.csls_r_d[mask]
        return float(np.min(norms)), float(np.max(norms)), float(np.min(rds))


def build_shard_sidecar(
    *,
    shard_tag: str,
    collection_name: str,
    doc_ids: Sequence[str],
    docs_raw: np.ndarray,
    dev_queries_in_shard_space: Optional[np.ndarray] = None,
    csls_k: int = 10,
    zero_mu_active: bool = True,
) -> ShardZeroMuCslsSidecar:
    """Build resident `8 B/doc` sidecar from raw shard vectors and causal DEV reference queries."""
    docs = np.asarray(docs_raw, dtype=np.float32)
    canon_ids = tuple(canonicalize_id(d) for d in doc_ids)
    if docs.ndim != 2 or docs.shape[0] != len(canon_ids):
        raise ValueError("docs_raw must be 2D and aligned with doc_ids")

    if zero_mu_active:
        mu_d = docs.mean(axis=0, dtype=np.float64).astype(np.float32)
        norms = np.maximum(
            np.linalg.norm(docs.astype(np.float64) - mu_d.astype(np.float64), axis=1).astype(np.float32),
            np.float32(1e-12),
        )
    else:
        mu_d = np.zeros(docs.shape[1], dtype=np.float32)
        norms = np.ones(docs.shape[0], dtype=np.float32)

    if dev_queries_in_shard_space is not None and np.asarray(dev_queries_in_shard_space).shape[0] > 0:
        dev_q = l2_normalize(np.asarray(dev_queries_in_shard_space, dtype=np.float64)).astype(np.float32)
        dots = dev_q @ docs.T
        if zero_mu_active:
            c_vec = dev_q @ mu_d
            ref_sims = (dots - c_vec[:, None]) / norms[None, :]
        else:
            ref_sims = dots
        eff_k = max(1, min(int(csls_k), int(ref_sims.shape[0])))
        if eff_k < ref_sims.shape[0]:
            top_part = np.partition(ref_sims, ref_sims.shape[0] - eff_k, axis=0)[-eff_k:, :]
        else:
            top_part = ref_sims
        r_d = top_part.mean(axis=0, dtype=np.float64).astype(np.float32)
    else:
        r_d = np.zeros(docs.shape[0], dtype=np.float32)

    return ShardZeroMuCslsSidecar(
        shard_tag=shard_tag,
        collection_name=collection_name,
        doc_ids=canon_ids,
        mu_d=mu_d,
        doc_centered_norms=norms,
        csls_r_d=r_d,
        csls_k=int(csls_k),
        zero_mu_active=bool(zero_mu_active),
        n_min_all=float(np.min(norms)) if norms.size else 1.0,
        n_max_all=float(np.max(norms)) if norms.size else 1.0,
        r_d_min_all=float(np.min(r_d)) if r_d.size else 0.0,
    )


def compute_unseen_score_upper_bound(
    *,
    dot_upper_bound: float,
    c_shift: float,
    n_min: float,
    n_max: float,
    use_csls: bool,
    r_d_min: float = 0.0,
    r_q_lower_bound: float = 0.0,
) -> float:
    """Exact upper bound on unseen ZeroMu or CSLS score given `dot_u <= dot_upper_bound`.

    Maximizes `(dot_upper_bound - c_shift) / n` over both positive endpoints `n_min` and `n_max`
    so negative `dot_upper_bound - c_shift` never produces an unsound (too-low) bound.
    """
    num = float(dot_upper_bound) - float(c_shift)
    safe_n_min = max(float(n_min), 1e-12)
    safe_n_max = max(float(n_max), safe_n_min)
    z_ub = max(num / safe_n_min, num / safe_n_max)
    if not use_csls:
        return float(z_ub)
    return float(2.0 * z_ub - float(r_d_min) - float(r_q_lower_bound))


# ---------------------------------------------------------------------------
# EK5 v3 Epoch-Pinned Authoritative Read Fence (`wait=true`, Tri-Valued State)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class EpochSnapshot:
    epoch: int
    states_by_doc_id: Dict[str, str]
    admitted_doc_ids: frozenset[str]
    contested_doc_ids: frozenset[str]
    audit_only_doc_ids: frozenset[str]

    def state_of(self, doc_id: str) -> str:
        return self.states_by_doc_id.get(doc_id, EP_ADMITTED)


def _map_ek5_eligibility(raw_state: str) -> str:
    if raw_state == "ADMIT":
        return EP_ADMITTED
    if raw_state == "CONTESTED":
        return EP_CONTESTED
    return EP_AUDIT_ONLY


class EpochPinnedEvidenceFence:
    """Authoritative `EK5 v3` epoch-pinned read fence with `wait=true` barrier and tri-valued states."""

    def __init__(self, doc_ids: Sequence[str], *, initial_epoch: int = 1) -> None:
        self._cond = threading.Condition()
        self._doc_ids = tuple(canonicalize_id(d) for d in doc_ids)
        self._doc_index = {did: idx for idx, did in enumerate(self._doc_ids)}
        self._store = StratifiedEvidenceStore(strict_dag_supersede=True, buffer_pending=False)
        events = [
            StratifiedEvent(
                event_id=did,
                entity=f"ent_{idx}",
                slot="claim",
                claim_value=f"val_{did}",
                valid_time=0,
                tx_time=int(initial_epoch),
                authority_tier=1,
            )
            for idx, did in enumerate(self._doc_ids)
        ]
        if events:
            self._store.apply_mutation(new_events=events)
        self._current_epoch = int(initial_epoch)
        self._snapshots: Dict[int, EpochSnapshot] = {
            self._current_epoch: self._build_snapshot_locked(self._current_epoch)
        }
        self._bitset_cache: Dict[Tuple[int, str], Tuple[np.ndarray, bytes, float]] = {}

    def _build_snapshot_locked(self, epoch: int) -> EpochSnapshot:
        states: Dict[str, str] = {}
        admitted: List[str] = []
        contested: List[str] = []
        audit_only: List[str] = []
        for did in self._store.events:
            mapped = _map_ek5_eligibility(self._store.eligibility.get(did, "ADMIT"))
            states[did] = mapped
            if mapped == EP_ADMITTED:
                admitted.append(did)
            elif mapped == EP_CONTESTED:
                contested.append(did)
            else:
                audit_only.append(did)
        return EpochSnapshot(
            epoch=int(epoch),
            states_by_doc_id=states,
            admitted_doc_ids=frozenset(admitted),
            contested_doc_ids=frozenset(contested),
            audit_only_doc_ids=frozenset(audit_only),
        )

    @property
    def current_epoch(self) -> int:
        with self._cond:
            return self._current_epoch

    def commit_mutation(
        self,
        *,
        new_events: Sequence[StratifiedEvent | str] = (),
        add_attestations: Sequence[EdgeAttestation | Tuple[str, str, str]] = (),
        withdraw_attestation_ids: Sequence[str] = (),
        target_epoch: Optional[int] = None,
    ) -> EpochSnapshot:
        """Apply an atomic `EK5 v3` mutation, advance epoch monotonically, and wake `wait=true` readers."""
        with self._cond:
            next_epoch = int(target_epoch) if target_epoch is not None else (self._current_epoch + 1)
            if next_epoch <= self._current_epoch:
                raise ValueError(f"target_epoch={next_epoch} must exceed current_epoch={self._current_epoch}")
            self._store.apply_mutation(
                new_events=new_events,
                add_attestations=add_attestations,
                withdraw_attestation_ids=withdraw_attestation_ids,
            )
            self._current_epoch = next_epoch
            snap = self._build_snapshot_locked(next_epoch)
            self._snapshots[next_epoch] = snap
            self._cond.notify_all()
            return snap

    def wait_and_get_snapshot(
        self,
        epoch_pin: int,
        *,
        wait_for_epoch: bool = True,
        timeout_s: float = 5.0,
    ) -> EpochSnapshot:
        """Enforce `wait=true` consistency fence for `epoch_pin` and return immutable snapshot."""
        req_epoch = int(epoch_pin)
        with self._cond:
            if req_epoch <= 0:
                return self._snapshots[self._current_epoch]
            if req_epoch > self._current_epoch:
                if not wait_for_epoch:
                    raise TRA1AuthorizationError(
                        f"EPOCH_NOT_YET_VISIBLE: requested epoch_pin={req_epoch} > current_epoch={self._current_epoch}"
                    )
                deadline = time.monotonic() + float(timeout_s)
                while self._current_epoch < req_epoch:
                    rem = deadline - time.monotonic()
                    if rem <= 0.0:
                        raise TimeoutError(
                            f"Timed out waiting for epoch_pin={req_epoch} (current={self._current_epoch})"
                        )
                    self._cond.wait(timeout=rem)
            if req_epoch in self._snapshots:
                return self._snapshots[req_epoch]
            available = [e for e in sorted(self._snapshots) if e <= req_epoch]
            return self._snapshots[available[-1]]

    def shard_admission_mask(
        self,
        epoch_pin: int,
        shard_tag: str,
        shard_doc_ids: Sequence[str],
        *,
        wait_for_epoch: bool = True,
    ) -> Tuple[EpochSnapshot, np.ndarray, bytes, float]:
        snap = self.wait_and_get_snapshot(epoch_pin, wait_for_epoch=wait_for_epoch)
        key = (snap.epoch, shard_tag)
        with self._cond:
            cached = self._bitset_cache.get(key)
        if cached is not None and cached[0].shape[0] == len(shard_doc_ids):
            return snap, cached[0], cached[1], cached[2]

        admitted_set = snap.admitted_doc_ids
        mask = np.fromiter((did in admitted_set for did in shard_doc_ids), dtype=bool, count=len(shard_doc_ids))
        packed = pack_admitted_bitset(mask)
        p_admit = float(np.mean(mask)) if mask.size else 1.0
        with self._cond:
            self._bitset_cache[key] = (mask, packed, p_admit)
        return snap, mask, packed, p_admit


# ---------------------------------------------------------------------------
# TRA-1 Fail-Closed Ingress Controller & Causal EMA
# ---------------------------------------------------------------------------
def make_representation_descriptor(
    *,
    encoder_id: str,
    encoder_revision: str,
    dimension: int,
    role: RepresentationRole,
    compatibility_domain_id: str,
    metric: DistanceMetric = DistanceMetric.DOT_PRODUCT,
    normalization: Normalization = Normalization.L2,
) -> RepresentationDescriptor:
    h = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()  # noqa: E731
    return RepresentationDescriptor(
        encoder_id=encoder_id,
        encoder_revision=encoder_revision,
        encoder_weights_sha256=h(f"weights:{encoder_id}:{encoder_revision}:{dimension}"),
        tokenizer_id=f"{encoder_id}-tok",
        tokenizer_sha256=h(f"tok:{encoder_id}:{encoder_revision}"),
        role=role,
        compatibility_domain_id=compatibility_domain_id,
        compatibility_domain_sha256=h(f"dom:{compatibility_domain_id}"),
        instruction_sha256=h(f"inst:{encoder_id}:{role.value}"),
        pooling=PoolingStrategy.MEAN,
        dimension=int(dimension),
        preprocessing_id="standard-utf8-v1",
        preprocessing_sha256=h("prep:standard-utf8-v1"),
        projection_chain_id=f"proj-{dimension}",
        projection_chain_sha256=h(f"proj:{dimension}"),
        adapter_chain_id="none",
        adapter_chain_sha256=h("adapter:none"),
        quantization_id="fp32",
        quantization_sha256=h("quant:fp32"),
        normalization=normalization,
        metric=metric,
        dtype=VectorDType.FLOAT32,
    )


def build_tra1_route_bundle(
    *,
    index_id: str,
    source_encoder_id: str,
    source_dim: int,
    target_encoder_id: str,
    target_dim: int,
    bridge: CenteredProcrustesBridge,
    corpus_snapshot_id: str,
    document_count: int,
    validation_count: int = 64,
    validation_confidence: float = 0.95,
    scoring_mode: str = "zeromu_csls",
    csls_k: int = 10,
) -> Tuple[RepresentationDescriptor, ResidentIndexDescriptor, BridgeContract, str]:
    """Construct content-addressed `RepresentationDescriptor`, `ResidentIndexDescriptor`, and `BridgeContract`."""
    h = lambda s: hashlib.sha256(s.encode("utf-8")).hexdigest()  # noqa: E731
    src_query_space = make_representation_descriptor(
        encoder_id=source_encoder_id,
        encoder_revision="rev-src-v1",
        dimension=source_dim,
        role=RepresentationRole.QUERY,
        compatibility_domain_id=f"domain-{source_encoder_id}-{source_dim}",
    )
    tgt_domain = f"domain-{target_encoder_id}-{target_dim}"
    tgt_query_space = make_representation_descriptor(
        encoder_id=target_encoder_id,
        encoder_revision="rev-tgt-v1",
        dimension=target_dim,
        role=RepresentationRole.QUERY,
        compatibility_domain_id=tgt_domain,
    )
    tgt_doc_space = make_representation_descriptor(
        encoder_id=target_encoder_id,
        encoder_revision="rev-tgt-v1",
        dimension=target_dim,
        role=RepresentationRole.DOCUMENT,
        compatibility_domain_id=tgt_domain,
    )
    resident_index = ResidentIndexDescriptor(
        index_id=index_id,
        document_space=tgt_doc_space,
        query_space=tgt_query_space,
        corpus_snapshot_id=corpus_snapshot_id,
        corpus_snapshot_sha256=h(f"snapshot:{corpus_snapshot_id}:{document_count}"),
        vector_build_id=f"hnsw-dot-{target_dim}",
        vector_build_sha256=h(f"build:{index_id}:{target_dim}"),
        document_count=int(document_count),
    )
    evidence_sha = h(f"evidence:{index_id}:{bridge.weights_sha256}:{validation_count}")
    contract = build_bridge_contract(
        source=src_query_space,
        target=tgt_query_space,
        bridge=bridge,
        transform_id=f"bridge-{source_dim}-to-{target_dim}-{index_id}",
        anchor_manifest_sha256=h(f"anchors:{index_id}:{bridge.fit_count}"),
        validation_count=int(validation_count),
        validation_confidence=float(validation_confidence),
        evidence_id=f"ev-{index_id}",
        evidence_sha256=evidence_sha,
        role=RepresentationRole.QUERY,
        scoring_mode=scoring_mode,
        csls_k=int(csls_k),
    )
    return src_query_space, resident_index, contract, evidence_sha


@dataclass
class GatewayRouteRegistration:
    """Registered route tying an index generation, TRA-1 contract, causal EMA state, and shard sidecars."""

    route_key: str
    index_generation: str
    query_space: RepresentationDescriptor
    resident_index: ResidentIndexDescriptor
    bridge_contract: Optional[BridgeContract]
    bridge: Optional[CenteredProcrustesBridge]
    trusted_evidence_sha256s: frozenset[str]
    trusted_bridge_contract_ids: frozenset[str]
    primary_sidecar: ShardZeroMuCslsSidecar
    secondary_sidecar: Optional[ShardZeroMuCslsSidecar] = None
    affine_mu_x: Optional[np.ndarray] = None
    affine_mu_y: Optional[np.ndarray] = None
    affine_weight: Optional[np.ndarray] = None
    ema_alpha: float = 0.05
    min_retained_energy_ratio: float = 0.05
    min_fit_count: int = 32
    min_validation_count: int = 16
    minimum_route_confidence: float = 0.70
    secondary_query_is_bridged: bool = False
    _ema_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _mu_q_state: Optional[np.ndarray] = field(default=None, repr=False)
    _tra1_decision_cache: Dict[Tuple[str, str, str, str], ResolutionDecision] = field(
        default_factory=dict, repr=False
    )

    def __post_init__(self) -> None:
        if self.bridge is not None:
            init_mu = self.bridge.mu_q_override if self.bridge.mu_q_override is not None else self.bridge.mu_x
            self._mu_q_state = np.asarray(init_mu, dtype=np.float32).reshape(-1).copy()
            self._q_f32 = np.asarray(self.bridge.Q, dtype=np.float32)
            self._bridge_weights_sha256 = self.bridge.weights_sha256.lower()
        else:
            self._mu_q_state = None
            self._q_f32 = None
            self._bridge_weights_sha256 = ""

    def reset_causal_ema(self) -> None:
        with self._ema_lock:
            if self.bridge is not None:
                init_mu = self.bridge.mu_q_override if self.bridge.mu_q_override is not None else self.bridge.mu_x
                self._mu_q_state = np.asarray(init_mu, dtype=np.float32).reshape(-1).copy()

    def verify_tra1_ingress(
        self,
        *,
        claimed_query_space_id: str,
        claimed_bridge_contract_id: str,
        claimed_weights_sha256: str,
        claimed_index_generation: str,
        bypass_cache: bool = False,
    ) -> ResolutionDecision:
        if claimed_index_generation != self.index_generation:
            raise TRA1AuthorizationError(
                f"INDEX_GENERATION_MISMATCH: got {claimed_index_generation!r}, expected {self.index_generation!r}"
            )
        if claimed_query_space_id != self.query_space.stable_id:
            raise TRA1AuthorizationError(
                f"UNTRUSTED_QUERY_SPACE: got {claimed_query_space_id!r}, expected {self.query_space.stable_id!r}"
            )
        if self.bridge is not None and self.bridge_contract is not None:
            if claimed_bridge_contract_id != self.bridge_contract.stable_id:
                raise TRA1AuthorizationError(
                    f"BRIDGE_CONTRACT_MISMATCH: got {claimed_bridge_contract_id!r}, "
                    f"expected {self.bridge_contract.stable_id!r}"
                )
            if claimed_weights_sha256.lower() != self._bridge_weights_sha256:
                raise TRA1AuthorizationError(
                    f"BRIDGE_WEIGHTS_SHA256_MISMATCH: got {claimed_weights_sha256!r}, "
                    f"expected {self._bridge_weights_sha256!r}"
                )

        cache_key = (
            claimed_query_space_id,
            claimed_bridge_contract_id,
            claimed_weights_sha256.lower(),
            claimed_index_generation,
        )
        if not bypass_cache:
            cached = self._tra1_decision_cache.get(cache_key)
            if cached is not None:
                return cached

        bridges = [self.bridge_contract] if self.bridge_contract is not None else []
        decision = resolve_with_tra1(
            query_space=self.query_space,
            resident_index=self.resident_index,
            bridges=bridges,
            trusted_evidence_sha256s=self.trusted_evidence_sha256s,
            trusted_bridge_contract_ids=self.trusted_bridge_contract_ids,
            minimum_route_confidence=self.minimum_route_confidence,
            min_fit_count=self.min_fit_count,
            min_validation_count=self.min_validation_count,
        )
        if decision.mode is ResolutionMode.ABSTAIN:
            raise TRA1AuthorizationError(
                f"TRA1_ABSTAIN_FAIL_CLOSED: reason={decision.reason.value} detail={decision.detail}"
            )
        self._tra1_decision_cache[cache_key] = decision
        return decision

    def project_query_causal(
        self,
        raw_vec: np.ndarray,
        *,
        scoring_mode: str,
        update_ema: bool = True,
    ) -> Tuple[np.ndarray, float]:
        """Project one incoming query vector with pre-normalization energy guard + strictly causal EMA."""
        q = np.asarray(raw_vec, dtype=np.float32).reshape(-1)
        expected_dim = self.query_space.dimension
        if q.shape[0] != expected_dim:
            raise TRA1AuthorizationError(
                f"QUERY_DIMENSION_MISMATCH: got {q.shape[0]}, expected {expected_dim}"
            )
        if not np.all(np.isfinite(q)):
            raise TRA1AuthorizationError("NON_FINITE_QUERY_VECTOR")

        if scoring_mode == SCORING_NATIVE_DOT or self.bridge is None:
            q_norm = float(np.linalg.norm(q))
            if q_norm <= 1e-8:
                raise TRA1AuthorizationError("ZERO_NORM_QUERY_VECTOR")
            return (q / np.float32(q_norm)).astype(np.float32), 1.0

        if scoring_mode == SCORING_QUERY_AFFINE:
            if self.affine_mu_x is None or self.affine_mu_y is None or self.affine_weight is None:
                raise TRA1AuthorizationError("QUERY_AFFINE_WEIGHTS_NOT_CONFIGURED")
            delta_aff = q.astype(np.float64) - self.affine_mu_x.reshape(-1)
            mapped_aff = delta_aff @ self.affine_weight + self.affine_mu_y.reshape(-1)
            aff_norm = float(np.linalg.norm(mapped_aff))
            if aff_norm <= 1e-8:
                raise TRA1AuthorizationError("DEGENERATE_AFFINE_QUERY_PROJECTION")
            return (mapped_aff / aff_norm).astype(np.float32), 1.0

        assert self._q_f32 is not None
        with self._ema_lock:
            assert self._mu_q_state is not None
            mu_before = self._mu_q_state.copy()
            if update_ema and self.ema_alpha > 0.0:
                alpha = np.float32(self.ema_alpha)
                self._mu_q_state = (np.float32(1.0) - alpha) * self._mu_q_state + alpha * q

        delta = q - mu_before
        delta_norm = float(np.linalg.norm(delta))
        if delta_norm <= 1e-8:
            raise TRA1AuthorizationError("DEGENERATE_CENTERED_QUERY_NORM")

        proj = delta @ self._q_f32
        proj_norm = float(np.linalg.norm(proj))
        retained_ratio = proj_norm / delta_norm
        if retained_ratio < self.min_retained_energy_ratio or proj_norm <= 1e-8:
            raise TRA1AuthorizationError(
                f"LOW_RETAINED_SUBSPACE_ENERGY: ratio={retained_ratio:.6f} < {self.min_retained_energy_ratio:.6f}"
            )
        q_hat = (proj / np.float32(proj_norm)).astype(np.float32)
        return q_hat, float(retained_ratio)

    def project_batch_causal(
        self,
        raw_vecs: np.ndarray,
        *,
        scoring_mode: str,
        update_ema: bool = False,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Vectorized batch projection (`[B, D_in] @ [D_in, D_out]` single BLAS SGEMM)."""
        qs = np.asarray(raw_vecs, dtype=np.float32)
        if qs.ndim != 2 or qs.shape[1] != self.query_space.dimension:
            raise TRA1AuthorizationError(
                f"BATCH_QUERY_SHAPE_MISMATCH: got {qs.shape}, expected (*, {self.query_space.dimension})"
            )
        if not np.all(np.isfinite(qs)):
            raise TRA1AuthorizationError("NON_FINITE_BATCH_QUERY_VECTOR")

        n_b = qs.shape[0]
        if scoring_mode == SCORING_NATIVE_DOT or self.bridge is None:
            norms = np.linalg.norm(qs, axis=1, keepdims=True)
            if bool(np.any(norms <= 1e-8)):
                raise TRA1AuthorizationError("ZERO_NORM_QUERY_VECTOR")
            return (qs / norms).astype(np.float32), np.ones(n_b, dtype=np.float32)

        if update_ema and self.ema_alpha > 0.0:
            out = np.empty((n_b, self.primary_sidecar.dim), dtype=np.float32)
            ratios = np.empty(n_b, dtype=np.float32)
            for i in range(n_b):
                out[i], ratios[i] = self.project_query_causal(qs[i], scoring_mode=scoring_mode, update_ema=True)
            return out, ratios

        assert self._q_f32 is not None
        with self._ema_lock:
            assert self._mu_q_state is not None
            mu_cur = self._mu_q_state.copy()

        deltas = qs - mu_cur[None, :]
        delta_norms = np.linalg.norm(deltas, axis=1)
        if bool(np.any(delta_norms <= 1e-8)):
            raise TRA1AuthorizationError("DEGENERATE_CENTERED_QUERY_NORM")

        projs = deltas @ self._q_f32
        proj_norms = np.linalg.norm(projs, axis=1)
        ratios = (proj_norms / delta_norms).astype(np.float32)
        min_r = float(np.min(ratios))
        if min_r < self.min_retained_energy_ratio or bool(np.any(proj_norms <= 1e-8)):
            raise TRA1AuthorizationError(
                f"LOW_RETAINED_SUBSPACE_ENERGY: min_ratio={min_r:.6f} < {self.min_retained_energy_ratio:.6f}"
            )
        return (projs / proj_norms[:, None]).astype(np.float32), ratios


# ---------------------------------------------------------------------------
# Backend ANN Search Protocol & Live Qdrant gRPC (:16334) Adapter
# ---------------------------------------------------------------------------
class AnnShardBackend(Protocol):
    """Zero-vector ANN backend returning ONLY `(point_ids, raw_dots)`."""

    def query_shortlist(
        self,
        *,
        collection_name: str,
        query_vec: np.ndarray,
        limit: int,
        ef_search: int,
        exact: bool = False,
        use_replica_filter: bool = False,
        allowed_point_ids: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        ...

    def query_batch_shortlist(
        self,
        *,
        collection_name: str,
        query_vecs: np.ndarray,
        limit: int,
        ef_search: int,
        exact: bool = False,
        use_replica_filter: bool = False,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        ...


class NumpyInMemoryAnnBackend:
    """Deterministic zero-vector backend for fast unit tests (simulates HNSW + replica filter bitsets)."""

    def __init__(self) -> None:
        self._collections: Dict[str, np.ndarray] = {}
        self._replica_admitted: Dict[str, np.ndarray] = {}

    def register_collection(
        self,
        collection_name: str,
        vectors: np.ndarray,
        *,
        replica_admitted: Optional[np.ndarray] = None,
    ) -> None:
        mat = np.asarray(vectors, dtype=np.float32)
        self._collections[collection_name] = mat
        if replica_admitted is not None:
            self._replica_admitted[collection_name] = np.asarray(replica_admitted, dtype=bool).copy()
        else:
            self._replica_admitted[collection_name] = np.ones(mat.shape[0], dtype=bool)

    def update_replica_admitted(self, collection_name: str, replica_admitted: np.ndarray) -> None:
        self._replica_admitted[collection_name] = np.asarray(replica_admitted, dtype=bool).copy()

    def query_shortlist(
        self,
        *,
        collection_name: str,
        query_vec: np.ndarray,
        limit: int,
        ef_search: int,
        exact: bool = False,
        use_replica_filter: bool = False,
        allowed_point_ids: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        mat = self._collections[collection_name]
        q = np.asarray(query_vec, dtype=np.float32).reshape(-1)
        dots = mat @ q
        n = mat.shape[0]
        mask = np.ones(n, dtype=bool)
        if use_replica_filter and collection_name in self._replica_admitted:
            mask &= self._replica_admitted[collection_name]
        if allowed_point_ids is not None:
            allow_mask = np.zeros(n, dtype=bool)
            allow_mask[np.asarray(allowed_point_ids, dtype=np.int64)] = True
            mask &= allow_mask

        cand_idx = np.flatnonzero(mask)
        if cand_idx.size == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)

        # Simulate finite-ef horizon if not exact and ef_search < cand_idx.size
        _ = exact, ef_search
        cand_dots = dots[cand_idx]
        k_eff = min(int(limit), int(cand_idx.size))
        if k_eff < cand_idx.size:
            part = np.argpartition(-cand_dots, k_eff - 1)[:k_eff]
            order = part[np.argsort(-cand_dots[part], kind="mergesort")]
        else:
            order = np.argsort(-cand_dots, kind="mergesort")
        sel_idx = cand_idx[order].astype(np.int64)
        sel_dots = cand_dots[order].astype(np.float32)
        return sel_idx, sel_dots

    def query_batch_shortlist(
        self,
        *,
        collection_name: str,
        query_vecs: np.ndarray,
        limit: int,
        ef_search: int,
        exact: bool = False,
        use_replica_filter: bool = False,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        qs = np.asarray(query_vecs, dtype=np.float32)
        return [
            self.query_shortlist(
                collection_name=collection_name,
                query_vec=qs[i],
                limit=limit,
                ef_search=ef_search,
                exact=exact,
                use_replica_filter=use_replica_filter,
            )
            for i in range(qs.shape[0])
        ]


class LiveQdrantGrpcBackend:
    """Zero-vector live Qdrant HNSW gRPC (`:6334` / `:16334`) backend (`with_vectors=False` always)."""

    def __init__(self, host: str = "127.0.0.1", http_port: int = 16333, grpc_port: int = 16334) -> None:
        from qdrant_client import QdrantClient
        from qdrant_client import models as qmodels
        from qdrant_client.conversions.conversion import RestToGrpc
        from qdrant_client.grpc import points_pb2 as qpoints_pb2

        self._qmodels = qmodels
        self._rest_to_grpc = RestToGrpc
        self._qp = qpoints_pb2
        self.host = host
        self.http_port = int(http_port)
        self.grpc_port = int(grpc_port)
        self.client = QdrantClient(
            host=host,
            port=int(http_port),
            grpc_port=int(grpc_port),
            prefer_grpc=True,
            timeout=120,
            check_compatibility=False,
            grpc_options={
                "grpc.tcp_nodelay": 1,
                "grpc.keepalive_time_ms": 30000,
                "grpc.max_receive_message_length": 64 * 1024 * 1024,
                "grpc.max_send_message_length": 64 * 1024 * 1024,
            },
        )
        self._grpc_points = self.client.grpc_points
        self._with_vec_false = qpoints_pb2.WithVectorsSelector(enable=False)
        self._with_pay_false = qpoints_pb2.WithPayloadSelector(enable=False)
        self._replica_admitted_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="replica_admitted",
                    match=qmodels.MatchValue(value=True),
                )
            ]
        )
        self._replica_admitted_filter_grpc = RestToGrpc.convert_filter(self._replica_admitted_filter)

    def close(self) -> None:
        self.client.close()

    def _build_filter_grpc(
        self,
        use_replica_filter: bool,
        allowed_point_ids: Optional[np.ndarray],
    ) -> Any:
        if not use_replica_filter and allowed_point_ids is None:
            return None
        if use_replica_filter and allowed_point_ids is None:
            return self._replica_admitted_filter_grpc
        qmodels = self._qmodels
        must: List[Any] = []
        if use_replica_filter:
            must.append(
                qmodels.FieldCondition(
                    key="replica_admitted",
                    match=qmodels.MatchValue(value=True),
                )
            )
        if allowed_point_ids is not None:
            must.append(qmodels.HasIdCondition(has_id=[int(i) for i in allowed_point_ids.tolist()]))
        return self._rest_to_grpc.convert_filter(qmodels.Filter(must=must))

    def query_shortlist(
        self,
        *,
        collection_name: str,
        query_vec: np.ndarray,
        limit: int,
        ef_search: int,
        exact: bool = False,
        use_replica_filter: bool = False,
        allowed_point_ids: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        qp = self._qp
        params_grpc = qp.SearchParams(hnsw_ef=int(ef_search), exact=bool(exact))
        qfilter_grpc = self._build_filter_grpc(use_replica_filter, allowed_point_ids)
        vec_list = np.asarray(query_vec, dtype=np.float32).reshape(-1).tolist()
        req = qp.QueryPoints(
            collection_name=collection_name,
            query=qp.Query(nearest=qp.VectorInput(dense=qp.DenseVector(data=vec_list))),
            filter=qfilter_grpc,
            params=params_grpc,
            limit=int(limit),
            with_vectors=self._with_vec_false,
            with_payload=self._with_pay_false,
        )
        resp = self._grpc_points.Query(req, timeout=30.0)
        pts = resp.result
        if not pts:
            return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)
        ids = np.fromiter((int(p.id.num) for p in pts), dtype=np.int64, count=len(pts))
        dots = np.fromiter((float(p.score) for p in pts), dtype=np.float32, count=len(pts))
        return ids, dots

    def query_batch_shortlist(
        self,
        *,
        collection_name: str,
        query_vecs: np.ndarray,
        limit: int,
        ef_search: int,
        exact: bool = False,
        use_replica_filter: bool = False,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        qp = self._qp
        params_grpc = qp.SearchParams(hnsw_ef=int(ef_search), exact=bool(exact))
        qfilter_grpc = self._replica_admitted_filter_grpc if use_replica_filter else None
        qs_list = np.asarray(query_vecs, dtype=np.float32).tolist()
        qpts = [
            qp.QueryPoints(
                collection_name=collection_name,
                query=qp.Query(nearest=qp.VectorInput(dense=qp.DenseVector(data=row))),
                filter=qfilter_grpc,
                params=params_grpc,
                limit=int(limit),
                with_vectors=self._with_vec_false,
                with_payload=self._with_pay_false,
            )
            for row in qs_list
        ]
        req = qp.QueryBatchPoints(collection_name=collection_name, query_points=qpts)
        resp = self._grpc_points.QueryBatch(req, timeout=30.0)
        out: List[Tuple[np.ndarray, np.ndarray]] = []
        for batch_item in resp.result:
            pts = batch_item.result
            if not pts:
                out.append((np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.float32)))
            else:
                ids = np.fromiter((int(p.id.num) for p in pts), dtype=np.int64, count=len(pts))
                dots = np.fromiter((float(p.score) for p in pts), dtype=np.float32, count=len(pts))
                out.append((ids, dots))
        return out


# ---------------------------------------------------------------------------
# Bidirectional Gateway Execution Engine + gRPC Servicer
# ---------------------------------------------------------------------------
@dataclass
class _ShardEvaluatedResult:
    shard_tag: str
    admitted_ids: List[str]
    admitted_point_ids: np.ndarray
    admitted_raw_dots: np.ndarray
    admitted_scores: np.ndarray
    contested_items: List[Tuple[str, int, float, float, str]]
    unseen_upper_bound: float
    exhausted_admitted: bool
    is_exact_prefix: bool


class ChelatedGatewayEngine:
    """Core bidirectional ingress + zero-vector egress + EK5 fence + certified-stop engine."""

    def __init__(
        self,
        *,
        ann_backend: AnnShardBackend,
        evidence_fence: Optional[EpochPinnedEvidenceFence] = None,
        ef_max: int = 256,
        max_refill_rounds: int = 2,
        k_os_max: int = 256,
        exact_fallback_limit: int = 512,
        use_replica_payload_prefilter: bool = False,
        fanout_workers: int = 8,
    ) -> None:
        self.ann_backend = ann_backend
        self.evidence_fence = evidence_fence
        self.ef_max = int(ef_max)
        self.max_refill_rounds = int(max_refill_rounds)
        self.k_os_max = int(k_os_max)
        self.exact_fallback_limit = int(exact_fallback_limit)
        self.use_replica_payload_prefilter = bool(use_replica_payload_prefilter)
        self._routes: Dict[str, GatewayRouteRegistration] = {}
        self._fanout_pool = futures.ThreadPoolExecutor(max_workers=max(2, int(fanout_workers)))

    def close(self) -> None:
        self._fanout_pool.shutdown(wait=False)

    def register_route(self, route: GatewayRouteRegistration) -> None:
        self._routes[route.route_key] = route

    def get_route(self, route_key: str) -> GatewayRouteRegistration:
        if route_key not in self._routes:
            raise TRA1AuthorizationError(f"UNKNOWN_ROUTE_INDEX: {route_key!r}")
        return self._routes[route_key]

    def _score_shard_shortlist(
        self,
        *,
        sidecar: ShardZeroMuCslsSidecar,
        point_ids: np.ndarray,
        raw_dots: np.ndarray,
        q_vec: np.ndarray,
        scoring_mode: str,
        snap: Optional[EpochSnapshot],
        client_admitted_mask: Optional[np.ndarray],
        authoritative_admitted_mask: Optional[np.ndarray],
        limit_requested: int,
        is_exact_prefix: bool,
    ) -> _ShardEvaluatedResult:
        n_shard = sidecar.n_docs
        if authoritative_admitted_mask is not None and client_admitted_mask is not None:
            eff_admitted_mask = authoritative_admitted_mask & client_admitted_mask
        elif authoritative_admitted_mask is not None:
            eff_admitted_mask = authoritative_admitted_mask
        elif client_admitted_mask is not None:
            eff_admitted_mask = client_admitted_mask
        else:
            eff_admitted_mask = None

        n_total_admitted = int(np.sum(eff_admitted_mask)) if eff_admitted_mask is not None else n_shard

        use_zeromu = scoring_mode in {SCORING_BIDIR_ZEROMU, SCORING_BIDIR_ZEROMU_CSLS} and sidecar.zero_mu_active
        use_csls = scoring_mode == SCORING_BIDIR_ZEROMU_CSLS

        if point_ids.size == 0:
            return _ShardEvaluatedResult(
                shard_tag=sidecar.shard_tag,
                admitted_ids=[],
                admitted_point_ids=np.zeros(0, dtype=np.int64),
                admitted_raw_dots=np.zeros(0, dtype=np.float32),
                admitted_scores=np.zeros(0, dtype=np.float32),
                contested_items=[],
                unseen_upper_bound=float("-inf") if n_total_admitted == 0 else float("inf"),
                exhausted_admitted=(n_total_admitted == 0),
                is_exact_prefix=is_exact_prefix,
            )

        if use_zeromu:
            base_scores, c_shift = sidecar.rescore_from_raw_dots(point_ids, raw_dots, q_vec)
        else:
            base_scores = np.asarray(raw_dots, dtype=np.float32).copy()
            c_shift = 0.0

        # Mandatory egress EK5 v3 epoch-pinned tri-valued fence (ADMITTED / CONTESTED / AUDIT_ONLY)
        admitted_pos: List[int] = []
        contested_items: List[Tuple[str, int, float, float, str]] = []

        if snap is None and eff_admitted_mask is None:
            admitted_pos = list(range(int(point_ids.size)))
        else:
            admitted_set = snap.admitted_doc_ids if snap is not None else None
            contested_set = snap.contested_doc_ids if snap is not None else frozenset()
            for pos in range(int(point_ids.size)):
                pid = int(point_ids[pos])
                did = sidecar.doc_ids[pid]
                client_allows = (
                    True
                    if (client_admitted_mask is None or pid >= client_admitted_mask.shape[0])
                    else bool(client_admitted_mask[pid])
                )
                if did in contested_set:
                    contested_items.append((
                        did,
                        pid,
                        float(raw_dots[pos]),
                        float(base_scores[pos]),
                        sidecar.shard_tag,
                    ))
                elif client_allows and (admitted_set is None or did in admitted_set):
                    admitted_pos.append(pos)
                # AUDIT_ONLY is strictly dropped before CSLS r_q and top-k release!

        if not admitted_pos:
            exhausted = n_total_admitted == 0
            return _ShardEvaluatedResult(
                shard_tag=sidecar.shard_tag,
                admitted_ids=[],
                admitted_point_ids=np.zeros(0, dtype=np.int64),
                admitted_raw_dots=np.zeros(0, dtype=np.float32),
                admitted_scores=np.zeros(0, dtype=np.float32),
                contested_items=contested_items,
                unseen_upper_bound=float("-inf") if exhausted else float("inf"),
                exhausted_admitted=exhausted,
                is_exact_prefix=is_exact_prefix,
            )

        apos = np.asarray(admitted_pos, dtype=np.int64)
        adm_pids = point_ids[apos]
        adm_dots = raw_dots[apos]
        adm_base = base_scores[apos]
        adm_ids = [sidecar.doc_ids[int(pid)] for pid in adm_pids.tolist()]

        r_q_lb = 0.0
        if use_csls:
            csls_k_eff = max(1, min(int(sidecar.csls_k), int(adm_base.size)))
            if csls_k_eff < adm_base.size:
                top_q = np.partition(adm_base, adm_base.size - csls_k_eff)[-csls_k_eff:]
            else:
                top_q = adm_base
            r_q_val = float(np.mean(top_q, dtype=np.float64))
            r_q_lb = r_q_val
            adm_scores = (2.0 * adm_base.astype(np.float64) - sidecar.csls_r_d[adm_pids].astype(np.float64) - r_q_val).astype(
                np.float32
            )
        else:
            adm_scores = adm_base

        exhausted = bool(
            adm_pids.size >= n_total_admitted
            or (is_exact_prefix and int(point_ids.size) < int(limit_requested) and eff_admitted_mask is None)
        )
        if exhausted:
            ub = float("-inf")
        elif use_csls and adm_pids.size < min(int(sidecar.csls_k), n_total_admitted):
            ub = float("inf")
        else:
            dot_bound = float(np.min(raw_dots))
            if use_zeromu:
                n_min, n_max, r_d_min = sidecar.envelope_for_mask(eff_admitted_mask)
            else:
                n_min, n_max = 1.0, 1.0
                _, _, r_d_min = sidecar.envelope_for_mask(eff_admitted_mask)
            ub = compute_unseen_score_upper_bound(
                dot_upper_bound=dot_bound,
                c_shift=c_shift,
                n_min=n_min,
                n_max=n_max,
                use_csls=use_csls,
                r_d_min=r_d_min,
                r_q_lower_bound=r_q_lb,
            )

        return _ShardEvaluatedResult(
            shard_tag=sidecar.shard_tag,
            admitted_ids=adm_ids,
            admitted_point_ids=adm_pids,
            admitted_raw_dots=adm_dots,
            admitted_scores=adm_scores,
            contested_items=contested_items,
            unseen_upper_bound=ub,
            exhausted_admitted=exhausted,
            is_exact_prefix=is_exact_prefix,
        )

    @staticmethod
    def _merge_shard_results(
        shard_results: Sequence[_ShardEvaluatedResult],
        k: int,
    ) -> Tuple[List[Tuple[str, int, float, float, str]], List[Tuple[str, int, float, float, str]], float]:
        """Deduplicate canonical doc IDs across shards with exact `(-score, doc_id)` tie-breaking."""
        best_admitted: Dict[str, Tuple[float, int, float, str]] = {}
        for sr in shard_results:
            for idx, did in enumerate(sr.admitted_ids):
                sc = float(sr.admitted_scores[idx])
                pid = int(sr.admitted_point_ids[idx])
                rdot = float(sr.admitted_raw_dots[idx])
                prev = best_admitted.get(did)
                if prev is None or sc > prev[0]:
                    best_admitted[did] = (sc, pid, rdot, sr.shard_tag)

        ordered_admitted = sorted(
            ((did, vals[1], vals[2], vals[0], vals[3]) for did, vals in best_admitted.items()),
            key=lambda item: (-item[3], item[0]),
        )
        k_eff = max(1, int(k))
        top_hits = ordered_admitted[:k_eff]
        s_k = float(top_hits[k_eff - 1][3]) if len(top_hits) >= k_eff else float("-inf")

        best_contested: Dict[str, Tuple[float, int, float, str]] = {}
        for sr in shard_results:
            for did, pid, rdot, bscore, stag in sr.contested_items:
                if did in best_admitted:
                    continue
                prev_c = best_contested.get(did)
                if prev_c is None or bscore > prev_c[0]:
                    best_contested[did] = (bscore, pid, rdot, stag)
        ordered_contested = sorted(
            ((did, vals[1], vals[2], vals[0], vals[3]) for did, vals in best_contested.items()),
            key=lambda item: (-item[3], item[0]),
        )[:k_eff]
        return top_hits, ordered_contested, s_k

    def execute_single_query(
        self,
        *,
        route: GatewayRouteRegistration,
        knn_msg: Any,
        scoring_mode: str,
        epoch_pin: int,
        wait_for_epoch: bool,
        enable_certified_stop: bool,
        dual_shard_fanout: bool,
        adaptive_ef: bool,
        allow_exact_fallback: bool,
        update_causal_ema: bool,
        require_exact_prefix_certificate: bool,
    ) -> Dict[str, Any]:
        t_ing0 = time.perf_counter()
        raw_q = np.asarray(knn_msg.vector, dtype=np.float32)
        q_primary, retained_ratio = route.project_query_causal(
            raw_q,
            scoring_mode=scoring_mode,
            update_ema=update_causal_ema,
        )

        # Prepare optional dual-shard query vector (`legacy_384d` + `native_4096d`)
        use_dual = bool(dual_shard_fanout and route.secondary_sidecar is not None)
        q_secondary: Optional[np.ndarray] = None
        if use_dual:
            assert route.secondary_sidecar is not None
            if len(knn_msg.native_vector) > 0:
                nv = np.asarray(knn_msg.native_vector, dtype=np.float32)
                nv_norm = max(float(np.linalg.norm(nv)), 1e-12)
                q_secondary = (nv / np.float32(nv_norm)).astype(np.float32)
            elif route.secondary_query_is_bridged:
                q_secondary = q_primary
                raw_norm = max(float(np.linalg.norm(raw_q)), 1e-12)
                q_primary = (raw_q / np.float32(raw_norm)).astype(np.float32)
            else:
                raw_norm = max(float(np.linalg.norm(raw_q)), 1e-12)
                q_secondary = (raw_q / np.float32(raw_norm)).astype(np.float32)

        eff_epoch_pin = int(knn_msg.epoch_pin) if int(knn_msg.epoch_pin) > 0 else int(epoch_pin)
        snap: Optional[EpochSnapshot] = None
        auth_mask_primary: Optional[np.ndarray] = None
        auth_mask_secondary: Optional[np.ndarray] = None
        p_admit_primary = 1.0
        p_admit_secondary = 1.0

        if self.evidence_fence is not None:
            snap, auth_mask_primary, _pk, p_admit_primary = self.evidence_fence.shard_admission_mask(
                eff_epoch_pin,
                route.primary_sidecar.shard_tag,
                route.primary_sidecar.doc_ids,
                wait_for_epoch=wait_for_epoch,
            )
            if use_dual and route.secondary_sidecar is not None:
                _, auth_mask_secondary, _pk2, p_admit_secondary = self.evidence_fence.shard_admission_mask(
                    eff_epoch_pin,
                    route.secondary_sidecar.shard_tag,
                    route.secondary_sidecar.doc_ids,
                    wait_for_epoch=False,
                )

        client_mask_primary: Optional[np.ndarray] = None
        if knn_msg.admitted_bitset:
            client_mask_primary = unpack_admitted_bitset(knn_msg.admitted_bitset, route.primary_sidecar.n_docs)
            if auth_mask_primary is not None:
                p_admit_primary = float(np.mean(auth_mask_primary & client_mask_primary))
            else:
                p_admit_primary = float(np.mean(client_mask_primary))

        k_final = max(1, int(knn_msg.k) if int(knn_msg.k) > 0 else 10)
        k_os_cur = max(k_final, int(knn_msg.k_os) if int(knn_msg.k_os) > 0 else max(64, 6 * k_final))
        req_ef = int(knn_msg.ef_search) if int(knn_msg.ef_search) > 0 else k_os_cur

        p_admit_min = min(p_admit_primary, p_admit_secondary) if use_dual else p_admit_primary
        ef_cur, density_exact = compute_adaptive_ef(
            k_os_cur,
            req_ef,
            p_admit_min,
            ef_max=self.ef_max,
            adaptive=adaptive_ef,
        )
        use_exact = bool(density_exact and allow_exact_fallback)
        ingress_guard_ms = (time.perf_counter() - t_ing0) * 1000.0

        ann_rtt_ms = 0.0
        egress_rescore_ms = 0.0
        refill_rounds = 0
        used_exact_fallback = bool(use_exact)

        def _fetch_and_score(limit_val: int, ef_val: int, exact_flag: bool) -> Tuple[List[_ShardEvaluatedResult], float, float]:
            t_a0 = time.perf_counter()
            if use_dual and route.secondary_sidecar is not None and q_secondary is not None:
                fut1 = self._fanout_pool.submit(
                    self.ann_backend.query_shortlist,
                    collection_name=route.primary_sidecar.collection_name,
                    query_vec=q_primary,
                    limit=limit_val,
                    ef_search=ef_val,
                    exact=exact_flag,
                    use_replica_filter=self.use_replica_payload_prefilter,
                )
                fut2 = self._fanout_pool.submit(
                    self.ann_backend.query_shortlist,
                    collection_name=route.secondary_sidecar.collection_name,
                    query_vec=q_secondary,
                    limit=limit_val,
                    ef_search=ef_val,
                    exact=exact_flag,
                    use_replica_filter=self.use_replica_payload_prefilter,
                )
                pids1, dots1 = fut1.result()
                pids2, dots2 = fut2.result()
                a_ms = (time.perf_counter() - t_a0) * 1000.0

                t_e0 = time.perf_counter()
                sr1 = self._score_shard_shortlist(
                    sidecar=route.primary_sidecar,
                    point_ids=pids1,
                    raw_dots=dots1,
                    q_vec=q_primary,
                    scoring_mode=scoring_mode,
                    snap=snap,
                    client_admitted_mask=client_mask_primary,
                    authoritative_admitted_mask=auth_mask_primary,
                    limit_requested=limit_val,
                    is_exact_prefix=exact_flag,
                )
                sr2 = self._score_shard_shortlist(
                    sidecar=route.secondary_sidecar,
                    point_ids=pids2,
                    raw_dots=dots2,
                    q_vec=q_secondary,
                    scoring_mode=scoring_mode,
                    snap=snap,
                    client_admitted_mask=None,
                    authoritative_admitted_mask=auth_mask_secondary,
                    limit_requested=limit_val,
                    is_exact_prefix=exact_flag,
                )
                e_ms = (time.perf_counter() - t_e0) * 1000.0
                return [sr1, sr2], a_ms, e_ms

            pids1, dots1 = self.ann_backend.query_shortlist(
                collection_name=route.primary_sidecar.collection_name,
                query_vec=q_primary,
                limit=limit_val,
                ef_search=ef_val,
                exact=exact_flag,
                use_replica_filter=self.use_replica_payload_prefilter,
            )
            a_ms = (time.perf_counter() - t_a0) * 1000.0
            t_e0 = time.perf_counter()
            sr1 = self._score_shard_shortlist(
                sidecar=route.primary_sidecar,
                point_ids=pids1,
                raw_dots=dots1,
                q_vec=q_primary,
                scoring_mode=scoring_mode,
                snap=snap,
                client_admitted_mask=client_mask_primary,
                authoritative_admitted_mask=auth_mask_primary,
                limit_requested=limit_val,
                is_exact_prefix=exact_flag,
            )
            e_ms = (time.perf_counter() - t_e0) * 1000.0
            return [sr1], a_ms, e_ms

        shard_evals, a_ms, e_ms = _fetch_and_score(k_os_cur, ef_cur, use_exact)
        ann_rtt_ms += a_ms
        egress_rescore_ms += e_ms
        top_hits, contested_hits, s_k = self._merge_shard_results(shard_evals, k_final)
        global_ub = max(sr.unseen_upper_bound for sr in shard_evals)
        all_exhausted = all(sr.exhausted_admitted for sr in shard_evals)

        def _is_certified(
            hits_len: int,
            ub_val: float,
            sk_val: float,
            exhausted_flag: bool,
            exact_prefix_flag: bool,
        ) -> bool:
            if exhausted_flag:
                return True
            if hits_len < k_final:
                return False
            if require_exact_prefix_certificate and not exact_prefix_flag:
                return False
            return bool(ub_val < sk_val)

        certified = _is_certified(
            len(top_hits),
            global_ub,
            s_k,
            all_exhausted,
            all(sr.is_exact_prefix for sr in shard_evals),
        )
        stop_reason = "SHORTLIST_NO_CERT_CHECK"

        if enable_certified_stop:
            if certified:
                stop_reason = "EXACT_EXHAUSTED" if all_exhausted else "BOUND_CERTIFIED_INITIAL"
            else:
                while (
                    not certified
                    and refill_rounds < self.max_refill_rounds
                    and k_os_cur < self.k_os_max
                ):
                    refill_rounds += 1
                    k_os_cur = min(self.k_os_max, k_os_cur * 2)
                    ef_cur, _ = compute_adaptive_ef(
                        k_os_cur,
                        max(ef_cur, k_os_cur),
                        p_admit_min,
                        ef_max=self.ef_max,
                        adaptive=adaptive_ef,
                    )
                    shard_evals, a_ms, e_ms = _fetch_and_score(k_os_cur, ef_cur, use_exact)
                    ann_rtt_ms += a_ms
                    egress_rescore_ms += e_ms
                    top_hits, contested_hits, s_k = self._merge_shard_results(shard_evals, k_final)
                    global_ub = max(sr.unseen_upper_bound for sr in shard_evals)
                    all_exhausted = all(sr.exhausted_admitted for sr in shard_evals)
                    certified = _is_certified(
                        len(top_hits),
                        global_ub,
                        s_k,
                        all_exhausted,
                        all(sr.is_exact_prefix for sr in shard_evals),
                    )
                    if certified:
                        stop_reason = f"BOUND_CERTIFIED_REFILL_{refill_rounds}"
                        break

                if not certified and allow_exact_fallback:
                    used_exact_fallback = True
                    n_max_shard = max(
                        route.primary_sidecar.n_docs,
                        route.secondary_sidecar.n_docs if (use_dual and route.secondary_sidecar) else 0,
                    )
                    exact_lim = max(k_os_cur, min(self.exact_fallback_limit, n_max_shard))
                    shard_evals, a_ms, e_ms = _fetch_and_score(exact_lim, ef_cur, True)
                    ann_rtt_ms += a_ms
                    egress_rescore_ms += e_ms
                    k_os_cur = exact_lim
                    top_hits, contested_hits, s_k = self._merge_shard_results(shard_evals, k_final)
                    global_ub = max(sr.unseen_upper_bound for sr in shard_evals)
                    all_exhausted = all(sr.exhausted_admitted for sr in shard_evals)
                    certified = _is_certified(len(top_hits), global_ub, s_k, all_exhausted, True)
                    if not certified and exact_lim < n_max_shard:
                        shard_evals, a_ms, e_ms = _fetch_and_score(n_max_shard, ef_cur, True)
                        ann_rtt_ms += a_ms
                        egress_rescore_ms += e_ms
                        k_os_cur = n_max_shard
                        top_hits, contested_hits, s_k = self._merge_shard_results(shard_evals, k_final)
                        global_ub = float("-inf")
                        certified = True
                        stop_reason = "EXACT_EXHAUSTIVE_FALLBACK_CERTIFIED"
                    else:
                        stop_reason = "EXACT_DOT_PREFIX_CERTIFIED" if certified else "UNCERTIFIED_AFTER_FALLBACK"
                elif not certified:
                    stop_reason = "UNCERTIFIED_BUDGET_CAP"
        else:
            # Even when stop loop is disabled, report whether the bound held passively
            stop_reason = "BOUND_HELD_PASSIVE" if certified else "FIXED_SHORTLIST_NO_REFILL"

        if used_exact_fallback:
            cert_mode = "exact_dot_prefix_bound"
        else:
            cert_mode = "hnsw_shortlist_bound"

        fenced_epoch = snap.epoch if snap is not None else eff_epoch_pin
        return {
            "query_id": str(knn_msg.query_id or ""),
            "hits": top_hits,
            "contested_hits": contested_hits,
            "certified_stop": bool(certified),
            "stop_reason": stop_reason,
            "certificate_mode": cert_mode,
            "fenced_epoch": int(fenced_epoch),
            "effective_k_os": int(k_os_cur),
            "effective_ef": int(ef_cur),
            "refill_rounds": int(refill_rounds),
            "used_exact_fallback": bool(used_exact_fallback),
            "retained_energy_ratio": float(retained_ratio),
            "unseen_csls_upper_bound": float(global_ub) if math.isfinite(global_ub) else -1e9,
            "kth_admitted_score": float(s_k) if math.isfinite(s_k) else -1e9,
            "ingress_guard_ms": float(ingress_guard_ms),
            "ann_rtt_ms": float(ann_rtt_ms),
            "egress_rescore_ms": float(egress_rescore_ms),
        }

    def execute_batch_fast(
        self,
        *,
        route: GatewayRouteRegistration,
        knn_batch: Sequence[Any],
        scoring_mode: str,
        epoch_pin: int,
        wait_for_epoch: bool,
        adaptive_ef: bool,
        allow_exact_fallback: bool,
        update_causal_ema: bool,
    ) -> List[Dict[str, Any]]:
        """Execute batched `SearchRequest.knn_batch` using native `query_batch_shortlist` on single-shard routes."""
        if not knn_batch:
            return []
        t_ing0 = time.perf_counter()
        n_b = len(knn_batch)
        raw_mat = np.asarray([km.vector for km in knn_batch], dtype=np.float32)
        q_hats, ratios = route.project_batch_causal(
            raw_mat,
            scoring_mode=scoring_mode,
            update_ema=update_causal_ema,
        )

        first = knn_batch[0]
        eff_epoch = int(first.epoch_pin) if int(first.epoch_pin) > 0 else int(epoch_pin)
        snap: Optional[EpochSnapshot] = None
        auth_mask: Optional[np.ndarray] = None
        p_admit = 1.0
        if self.evidence_fence is not None:
            snap, auth_mask, _pk, p_admit = self.evidence_fence.shard_admission_mask(
                eff_epoch,
                route.primary_sidecar.shard_tag,
                route.primary_sidecar.doc_ids,
                wait_for_epoch=wait_for_epoch,
            )

        k_final = max(1, int(first.k) if int(first.k) > 0 else 10)
        k_os = max(k_final, int(first.k_os) if int(first.k_os) > 0 else max(64, 6 * k_final))
        req_ef = int(first.ef_search) if int(first.ef_search) > 0 else k_os
        ef_eff, density_exact = compute_adaptive_ef(
            k_os,
            req_ef,
            p_admit,
            ef_max=self.ef_max,
            adaptive=adaptive_ef,
        )
        use_exact = bool(density_exact and allow_exact_fallback)
        ing_per_q_ms = ((time.perf_counter() - t_ing0) * 1000.0) / float(n_b)

        t_a0 = time.perf_counter()
        shortlists = self.ann_backend.query_batch_shortlist(
            collection_name=route.primary_sidecar.collection_name,
            query_vecs=q_hats,
            limit=k_os,
            ef_search=ef_eff,
            exact=use_exact,
            use_replica_filter=self.use_replica_payload_prefilter,
        )
        ann_per_q_ms = ((time.perf_counter() - t_a0) * 1000.0) / float(n_b)

        results: List[Dict[str, Any]] = []
        fenced_epoch = snap.epoch if snap is not None else eff_epoch
        for i, (pids, dots) in enumerate(shortlists):
            t_e0 = time.perf_counter()
            cmask = (
                unpack_admitted_bitset(knn_batch[i].admitted_bitset, route.primary_sidecar.n_docs)
                if knn_batch[i].admitted_bitset
                else None
            )
            sr = self._score_shard_shortlist(
                sidecar=route.primary_sidecar,
                point_ids=pids,
                raw_dots=dots,
                q_vec=q_hats[i],
                scoring_mode=scoring_mode,
                snap=snap,
                client_admitted_mask=cmask,
                authoritative_admitted_mask=auth_mask,
                limit_requested=k_os,
                is_exact_prefix=use_exact,
            )
            top_hits, contested_hits, s_k = self._merge_shard_results([sr], k_final)
            egress_ms = (time.perf_counter() - t_e0) * 1000.0
            cert = bool(sr.exhausted_admitted or (len(top_hits) >= k_final and sr.unseen_upper_bound < s_k))
            results.append({
                "query_id": str(knn_batch[i].query_id or f"b_{i}"),
                "hits": top_hits,
                "contested_hits": contested_hits,
                "certified_stop": cert,
                "stop_reason": "BOUND_HELD_BATCH" if cert else "BATCH_SHORTLIST",
                "certificate_mode": "exact_dot_prefix_bound" if use_exact else "hnsw_shortlist_bound",
                "fenced_epoch": int(fenced_epoch),
                "effective_k_os": int(k_os),
                "effective_ef": int(ef_eff),
                "refill_rounds": 0,
                "used_exact_fallback": bool(use_exact),
                "retained_energy_ratio": float(ratios[i]),
                "unseen_csls_upper_bound": float(sr.unseen_upper_bound) if math.isfinite(sr.unseen_upper_bound) else -1e9,
                "kth_admitted_score": float(s_k) if math.isfinite(s_k) else -1e9,
                "ingress_guard_ms": float(ing_per_q_ms),
                "ann_rtt_ms": float(ann_per_q_ms),
                "egress_rescore_ms": float(egress_ms),
            })
        return results


def _populate_hits_proto(pb2: ModuleType, hit_tuples: Sequence[Tuple[str, int, float, float, str]], state: str, cert: bool) -> List[Any]:
    return [
        pb2.SearchHit(
            id=did,
            point_id=int(pid),
            raw_dot=float(rdot),
            score=float(score),
            shard_tag=stag,
            epistemic_state=state,
            certified_stop=bool(cert),
        )
        for did, pid, rdot, score, stag in hit_tuples
    ]


def build_grpc_servicer(engine: ChelatedGatewayEngine, proto_bundle: Optional[CompiledProtoBundle] = None) -> Any:
    """Create a live `SearchServiceServicer` bound to `engine` and compiled Protobuf classes."""
    bundle = proto_bundle or compile_opensearch_knn_proto()
    pb2 = bundle.pb2
    pb2_grpc = bundle.pb2_grpc

    class ChelatedSearchServiceServicer(pb2_grpc.SearchServiceServicer):  # type: ignore[misc]
        def Search(self, request: Any, context: grpc.ServicerContext) -> Any:
            if not request.indices:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, "SearchRequest.indices must specify a route key")
            route_key = str(request.indices[0])
            scoring_mode = str(request.scoring_mode or SCORING_BIDIR_ZEROMU)
            if scoring_mode not in VALID_SCORING_MODES:
                context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"Unsupported scoring_mode={scoring_mode!r}")

            try:
                t_v0 = time.perf_counter()
                route = engine.get_route(route_key)
                route.verify_tra1_ingress(
                    claimed_query_space_id=str(request.query_space_id),
                    claimed_bridge_contract_id=str(request.bridge_contract_id),
                    claimed_weights_sha256=str(request.bridge_weights_sha256),
                    claimed_index_generation=str(request.index_generation),
                )
                if request.reset_causal_ema:
                    route.reset_causal_ema()
                tra1_ms = (time.perf_counter() - t_v0) * 1000.0

                if len(request.knn_batch) > 0:
                    if request.dual_shard_fanout or request.enable_certified_stop:
                        batch_dicts = [
                            engine.execute_single_query(
                                route=route,
                                knn_msg=km,
                                scoring_mode=scoring_mode,
                                epoch_pin=int(request.epoch_pin),
                                wait_for_epoch=bool(request.wait_for_epoch),
                                enable_certified_stop=bool(request.enable_certified_stop),
                                dual_shard_fanout=bool(request.dual_shard_fanout),
                                adaptive_ef=bool(request.adaptive_ef),
                                allow_exact_fallback=bool(request.allow_exact_fallback),
                                update_causal_ema=bool(request.update_causal_ema),
                                require_exact_prefix_certificate=bool(request.require_exact_prefix_certificate),
                            )
                            for km in request.knn_batch
                        ]
                    else:
                        batch_dicts = engine.execute_batch_fast(
                            route=route,
                            knn_batch=request.knn_batch,
                            scoring_mode=scoring_mode,
                            epoch_pin=int(request.epoch_pin),
                            wait_for_epoch=bool(request.wait_for_epoch),
                            adaptive_ef=bool(request.adaptive_ef),
                            allow_exact_fallback=bool(request.allow_exact_fallback),
                            update_causal_ema=bool(request.update_causal_ema),
                        )

                    q_results = []
                    for bd in batch_dicts:
                        cert = bool(bd["certified_stop"])
                        q_results.append(
                            pb2.QuerySearchResult(
                                query_id=bd["query_id"],
                                hits=_populate_hits_proto(pb2, bd["hits"], EP_ADMITTED, cert),
                                contested_hits=_populate_hits_proto(pb2, bd["contested_hits"], EP_CONTESTED, cert),
                                certified_stop=cert,
                                stop_reason=bd["stop_reason"],
                                certificate_mode=bd["certificate_mode"],
                                fenced_epoch=bd["fenced_epoch"],
                                effective_k_os=bd["effective_k_os"],
                                effective_ef=bd["effective_ef"],
                                refill_rounds=bd["refill_rounds"],
                                used_exact_fallback=bd["used_exact_fallback"],
                                retained_energy_ratio=bd["retained_energy_ratio"],
                                unseen_csls_upper_bound=bd["unseen_csls_upper_bound"],
                                kth_admitted_score=bd["kth_admitted_score"],
                                ingress_guard_ms=bd["ingress_guard_ms"] + (tra1_ms / max(len(batch_dicts), 1)),
                                ann_rtt_ms=bd["ann_rtt_ms"],
                                egress_rescore_ms=bd["egress_rescore_ms"],
                            )
                        )
                    first_r = q_results[0]
                    return pb2.SearchResponse(
                        hits=first_r.hits,
                        contested_hits=first_r.contested_hits,
                        certified_stop=all(r.certified_stop for r in q_results),
                        stop_reason=first_r.stop_reason,
                        certificate_mode=first_r.certificate_mode,
                        fenced_epoch=first_r.fenced_epoch,
                        effective_k_os=first_r.effective_k_os,
                        effective_ef=first_r.effective_ef,
                        refill_rounds=sum(r.refill_rounds for r in q_results),
                        used_exact_fallback=any(r.used_exact_fallback for r in q_results),
                        retained_energy_ratio=first_r.retained_energy_ratio,
                        unseen_csls_upper_bound=first_r.unseen_csls_upper_bound,
                        kth_admitted_score=first_r.kth_admitted_score,
                        ingress_guard_ms=sum(r.ingress_guard_ms for r in q_results),
                        ann_rtt_ms= batch_dicts[0]["ann_rtt_ms"] * len(batch_dicts),
                        egress_rescore_ms=sum(r.egress_rescore_ms for r in q_results),
                        backend_execution_label=BACKEND_EXECUTION_LABEL,
                        batch_results=q_results,
                    )

                res = engine.execute_single_query(
                    route=route,
                    knn_msg=request.knn,
                    scoring_mode=scoring_mode,
                    epoch_pin=int(request.epoch_pin),
                    wait_for_epoch=bool(request.wait_for_epoch),
                    enable_certified_stop=bool(request.enable_certified_stop),
                    dual_shard_fanout=bool(request.dual_shard_fanout),
                    adaptive_ef=bool(request.adaptive_ef),
                    allow_exact_fallback=bool(request.allow_exact_fallback),
                    update_causal_ema=bool(request.update_causal_ema),
                    require_exact_prefix_certificate=bool(request.require_exact_prefix_certificate),
                )
                cert = bool(res["certified_stop"])
                return pb2.SearchResponse(
                    hits=_populate_hits_proto(pb2, res["hits"], EP_ADMITTED, cert),
                    contested_hits=_populate_hits_proto(pb2, res["contested_hits"], EP_CONTESTED, cert),
                    certified_stop=cert,
                    stop_reason=res["stop_reason"],
                    certificate_mode=res["certificate_mode"],
                    fenced_epoch=res["fenced_epoch"],
                    effective_k_os=res["effective_k_os"],
                    effective_ef=res["effective_ef"],
                    refill_rounds=res["refill_rounds"],
                    used_exact_fallback=res["used_exact_fallback"],
                    retained_energy_ratio=res["retained_energy_ratio"],
                    unseen_csls_upper_bound=res["unseen_csls_upper_bound"],
                    kth_admitted_score=res["kth_admitted_score"],
                    ingress_guard_ms=res["ingress_guard_ms"] + tra1_ms,
                    ann_rtt_ms=res["ann_rtt_ms"],
                    egress_rescore_ms=res["egress_rescore_ms"],
                    backend_execution_label=BACKEND_EXECUTION_LABEL,
                )
            except TRA1AuthorizationError as exc:
                context.abort(grpc.StatusCode.FAILED_PRECONDITION, str(exc))
            except TimeoutError as exc:
                context.abort(grpc.StatusCode.DEADLINE_EXCEEDED, str(exc))

    return ChelatedSearchServiceServicer()


def start_chelated_grpc_server(
    engine: ChelatedGatewayEngine,
    *,
    bind_address: str = "127.0.0.1:0",
    max_workers: int = 16,
    proto_bundle: Optional[CompiledProtoBundle] = None,
) -> Tuple[grpc.Server, int, CompiledProtoBundle]:
    """Start a real `grpcio` HTTP/2 server serving `SearchService.Search`."""
    bundle = proto_bundle or compile_opensearch_knn_proto()
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=int(max_workers)),
        options=[
            ("grpc.so_reuseport", 0),
            ("grpc.tcp_nodelay", 1),
            ("grpc.max_send_message_length", 64 * 1024 * 1024),
            ("grpc.max_receive_message_length", 64 * 1024 * 1024),
            ("grpc.keepalive_time_ms", 30000),
        ],
    )
    servicer = build_grpc_servicer(engine, bundle)
    bundle.pb2_grpc.add_SearchServiceServicer_to_server(servicer, server)
    bound_port = int(server.add_insecure_port(bind_address))
    if bound_port <= 0:
        raise RuntimeError(f"Failed to bind gRPC server on {bind_address!r}")
    server.start()
    return server, bound_port, bundle


# ---------------------------------------------------------------------------
# Open-Loop Paced Benchmark Runner (Avoids Coordinated Omission)
# ---------------------------------------------------------------------------
def generate_open_loop_schedule(
    n_dispatches: int,
    target_rate_hz: float,
    *,
    seed: int = 42,
    poisson: bool = True,
) -> np.ndarray:
    """Generate monotonic open-loop release offsets (seconds from `t0`)."""
    n = int(n_dispatches)
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    rate = max(float(target_rate_hz), 1.0)
    if not poisson:
        return np.arange(n, dtype=np.float64) / rate
    rng = np.random.default_rng(seed)
    inter = rng.exponential(scale=1.0 / rate, size=n)
    inter[0] = 0.0
    return np.cumsum(inter)


def run_open_loop_dispatch(
    *,
    n_dispatches: int,
    target_rate_hz: float,
    worker_count: int,
    dispatch_fn: Callable[[int], Dict[str, float]],
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute `n_dispatches` following a pre-computed wall-clock open-loop schedule.

    Records both `open_loop_latency_ms = complete_time - scheduled_release_time` (capturing
    queueing delay without coordinated omission) and `service_rtt_ms = complete_time - send_time`.
    """
    offsets_s = generate_open_loop_schedule(n_dispatches, target_rate_hz, seed=seed, poisson=True)
    open_loop_ms = np.empty(n_dispatches, dtype=np.float64)
    service_ms = np.empty(n_dispatches, dtype=np.float64)
    serialize_ms = np.empty(n_dispatches, dtype=np.float64)
    ann_ms = np.empty(n_dispatches, dtype=np.float64)
    egress_ms = np.empty(n_dispatches, dtype=np.float64)
    req_bytes = np.empty(n_dispatches, dtype=np.float64)
    resp_bytes = np.empty(n_dispatches, dtype=np.float64)
    errors: List[str] = []
    err_lock = threading.Lock()

    t_origin = time.perf_counter()

    def _run_one(idx: int) -> None:
        sched_t = t_origin + float(offsets_s[idx])
        now = time.perf_counter()
        if sched_t > now:
            time.sleep(sched_t - now)
        send_t = time.perf_counter()
        try:
            metrics = dispatch_fn(idx)
            done_t = time.perf_counter()
            open_loop_ms[idx] = (done_t - sched_t) * 1000.0
            service_ms[idx] = (done_t - send_t) * 1000.0
            serialize_ms[idx] = float(metrics.get("serialize_cpu_ms", 0.0))
            ann_ms[idx] = float(metrics.get("ann_rtt_ms", service_ms[idx]))
            egress_ms[idx] = float(metrics.get("egress_rescore_ms", 0.0))
            req_bytes[idx] = float(metrics.get("req_wire_bytes", 0.0))
            resp_bytes[idx] = float(metrics.get("resp_wire_bytes", 0.0))
        except Exception as exc:  # pragma: no cover - defensive error accounting
            done_t = time.perf_counter()
            open_loop_ms[idx] = (done_t - sched_t) * 1000.0
            service_ms[idx] = (done_t - send_t) * 1000.0
            serialize_ms[idx] = 0.0
            ann_ms[idx] = 0.0
            egress_ms[idx] = 0.0
            req_bytes[idx] = 0.0
            resp_bytes[idx] = 0.0
            with err_lock:
                if len(errors) < 10:
                    errors.append(repr(exc))

    with futures.ThreadPoolExecutor(max_workers=max(1, int(worker_count))) as pool:
        futs = [pool.submit(_run_one, idx) for idx in range(n_dispatches)]
        for f in futs:
            f.result()

    wall_s = max(time.perf_counter() - t_origin, 1e-9)

    def _quantiles(arr: np.ndarray) -> Dict[str, float]:
        return {
            "mean": round(float(np.mean(arr)), 4),
            "p50": round(float(np.quantile(arr, 0.50)), 4),
            "p95": round(float(np.quantile(arr, 0.95)), 4),
            "p99": round(float(np.quantile(arr, 0.99)), 4),
            "max": round(float(np.max(arr)), 4),
        }

    return {
        "n_dispatches": int(n_dispatches),
        "target_rate_hz": float(target_rate_hz),
        "achieved_dispatch_rate_hz": round(float(n_dispatches / wall_s), 2),
        "wall_seconds": round(wall_s, 4),
        "error_count": len(errors),
        "error_samples": errors,
        "open_loop_e2e_ms": _quantiles(open_loop_ms),
        "service_rtt_ms": _quantiles(service_ms),
        "serialize_cpu_ms": _quantiles(serialize_ms),
        "rtt_plus_ann_ms": _quantiles(ann_ms),
        "egress_rescore_ms": _quantiles(egress_ms),
        "mean_request_wire_bytes": round(float(np.mean(req_bytes)), 1),
        "mean_response_wire_bytes": round(float(np.mean(resp_bytes)), 1),
        "total_wire_bytes": int(np.sum(req_bytes) + np.sum(resp_bytes)),
    }


__all__ = [
    "AnnShardBackend",
    "BACKEND_EXECUTION_LABEL",
    "ChelatedGatewayEngine",
    "CompiledProtoBundle",
    "EP_ADMITTED",
    "EP_AUDIT_ONLY",
    "EP_CONTESTED",
    "EpochPinnedEvidenceFence",
    "EpochSnapshot",
    "GatewayRouteRegistration",
    "LiveQdrantGrpcBackend",
    "NumpyInMemoryAnnBackend",
    "OPENSEARCH_KNN_GATEWAY_PROTO",
    "SCORING_BIDIR_ZEROMU",
    "SCORING_BIDIR_ZEROMU_CSLS",
    "SCORING_NATIVE_DOT",
    "SCORING_QUERY_AFFINE",
    "SCORING_REQUEST_ONLY_RAW_DOT",
    "ShardZeroMuCslsSidecar",
    "TRA1AuthorizationError",
    "build_grpc_servicer",
    "build_shard_sidecar",
    "build_tra1_route_bundle",
    "compile_opensearch_knn_proto",
    "compute_adaptive_ef",
    "compute_unseen_score_upper_bound",
    "generate_open_loop_schedule",
    "make_representation_descriptor",
    "pack_admitted_bitset",
    "run_open_loop_dispatch",
    "start_chelated_grpc_server",
    "unpack_admitted_bitset",
]
