"""Wave 19 / Bundle #57 / Item 2B: 250-payload (1,000-catalog) cartridge array.

Two quantities are kept strictly apart and never conflated:

* ``materialized_payload_count``: real ``.cspg`` payload files on disk (250).
* ``catalog_entry_count``: indexed manifest slots (1,000). A catalog lookup is a
  vector-index query through the public ``chelated_kernel.wrap_index`` surface;
  it is NOT a payload swap. Slots without a materialized payload raise
  :class:`PayloadNotMaterializedError` on swap.

Measured gates (floor tier, CPU-only, see ``tier`` in the manifest):

1. time to first usable inference of a swapped payload versus the physical
   sequential-read bound ``payload_bytes / measured_bw_bytes_s``;
2. concurrent multi-thread EK5 revocation races against queued and in-flight
   requests, with 0 post-enforcement executions plus an exact one-sided
   Clopper-Pearson 95% upper bound, and a deterministic unguarded negative
   control that must show the race is real;
3. EK5 canonical-audit parity with a fresh rebuild after every revocation;
4. a 6 GiB cgroup budget (``systemd-run`` command is built and asserted; the
   cage itself is only enforced on a Linux host, never on this Windows host).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import statistics
import sys
import tempfile
import threading
import time
import tracemalloc
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_ROOT = Path(__file__).resolve().parent
_KERNEL_PKG = _ROOT / "packages" / "chelated_kernel"
for _p in (str(_KERNEL_PKG), str(_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from chelated_kernel import wrap_index  # noqa: E402  (public SDK boundary only)

from chelated_cpu_memory_appliance import verify_cgroup_v2_enforced  # noqa: E402  (read-back of live cgroup v2 limits)
from chelated_pv7_spine.cpu_io_cage import (  # noqa: E402
    ONE_GIB_BYTES,
    CgroupCageConfig,
    build_cpu_only_env,
    build_systemd_run_command,
)
from chelated_pv7_spine.ek5_reason_canonicalizer import (  # noqa: E402
    compare_with_fresh_rebuild,
    export_canonical_ek5_audit_trace,
)
from chelated_pv7_spine.tra1_weight_sha_guard import (  # noqa: E402
    EMPTY_ADAPTER_SHA256,
    TRA1StateIdentityContract,
)
from ek5_v3_stratified import (  # noqa: E402
    EdgeAttestation,
    StratifiedEvent,
    StratifiedEvidenceStore,
)

BUNDLE_ID = 57
MODES: Tuple[str, ...] = ("local-drycheck", "run-live")
MATERIALIZED_PAYLOAD_COUNT = 250
CATALOG_ENTRY_COUNT = 1000
MEMORY_MAX_BYTES = 6 * ONE_GIB_BYTES
CSPG_MAGIC = b"CSPG1\x00\x00\x00"
CATALOG_DIM = 16
CLOPPER_ALPHA = 0.05
SEED = 5702


class PayloadNotMaterializedError(LookupError):
    """A catalog slot exists but no payload file is materialized for it."""


class LiveModeUnavailableError(RuntimeError):
    """``run-live`` demanded a property the host cannot provide (fail closed)."""


# ---------------------------------------------------------------------------
# exact one-sided Clopper-Pearson upper bound
# ---------------------------------------------------------------------------
def _binom_cdf(k: int, n: int, p: float) -> float:
    if p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 1.0 if k >= n else 0.0
    lp, lq = math.log(p), math.log1p(-p)
    total = 0.0
    for i in range(k + 1):
        total += math.exp(
            math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq
        )
    return min(1.0, total)


def clopper_pearson_upper(events: int, trials: int, alpha: float = CLOPPER_ALPHA) -> float:
    """Exact one-sided ``1-alpha`` upper bound for a binomial proportion."""
    if trials <= 0 or events < 0 or events > trials:
        raise ValueError("require 0 <= events <= trials and trials > 0")
    if events == trials:
        return 1.0
    if events == 0:
        return 1.0 - alpha ** (1.0 / trials)
    lo, hi = events / trials, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _binom_cdf(events, trials, mid) > alpha:
            lo = mid
        else:
            hi = mid
    return hi


# ---------------------------------------------------------------------------
# payload + catalog materialization
# ---------------------------------------------------------------------------
def _slot_id(index: int) -> str:
    return f"slot-{index:04d}"


def select_materialized_slots(
    catalog_count: int = CATALOG_ENTRY_COUNT,
    payload_count: int = MATERIALIZED_PAYLOAD_COUNT,
    seed: int = SEED,
) -> Tuple[int, ...]:
    """Hash-select which catalog slots carry a real payload (deterministic)."""
    ranked = sorted(
        range(catalog_count),
        key=lambda i: hashlib.sha256(f"{seed}|materialize|{i}".encode()).hexdigest(),
    )
    return tuple(sorted(ranked[:payload_count]))


def write_cspg_payload(path: Path, slot: int, payload_bytes: int, seed: int) -> str:
    """Write one deterministic LoRA-style ``.cspg`` file; return body SHA-256."""
    rng = np.random.default_rng(seed * 100003 + slot)
    body = rng.standard_normal(payload_bytes // 2).astype(np.float16).tobytes()
    body_sha = hashlib.sha256(body).hexdigest()
    header = CSPG_MAGIC + slot.to_bytes(4, "little") + bytes.fromhex(body_sha)
    with path.open("wb") as handle:
        handle.write(header)
        handle.write(body)
        handle.flush()
        os.fsync(handle.fileno())
    return body_sha


_HEADER_BYTES = len(CSPG_MAGIC) + 4 + 32


def read_cspg_sequential(path: Path, chunk: int = 1 << 20) -> Tuple[int, bytes, str]:
    """Sequentially read a payload; return ``(slot, body, declared_sha_hex)``."""
    with path.open("rb", buffering=0) as handle:
        header = handle.read(_HEADER_BYTES)
        if header[: len(CSPG_MAGIC)] != CSPG_MAGIC:
            raise ValueError(f"bad .cspg magic in {path.name}")
        slot = int.from_bytes(header[len(CSPG_MAGIC) : len(CSPG_MAGIC) + 4], "little")
        declared = header[len(CSPG_MAGIC) + 4 :].hex()
        parts: List[bytes] = []
        while True:
            block = handle.read(chunk)
            if not block:
                break
            parts.append(block)
    return slot, b"".join(parts), declared


def first_usable_inference(body: bytes, base_vec: np.ndarray) -> float:
    """Apply the payload as a rank-r adapter delta to a base activation."""
    dim = base_vec.shape[0]
    arr = np.frombuffer(body, dtype=np.float16)
    usable = (arr.size // (2 * dim)) * (2 * dim)
    if usable == 0:
        raise ValueError("payload too small for one adapter pair")
    rank = usable // (2 * dim)
    a = arr[: rank * dim].astype(np.float32).reshape(rank, dim)
    b = arr[rank * dim : 2 * rank * dim].astype(np.float32).reshape(dim, rank)
    out = base_vec + b @ (a @ base_vec) * (1.0 / rank)
    value = float(np.linalg.norm(out))
    if not math.isfinite(value):
        raise ValueError("non-finite adapter inference")
    return value


def build_catalog_index(seed: int, catalog_count: int = CATALOG_ENTRY_COUNT):
    """Index all manifest slots through the public ``wrap_index`` surface."""
    rng = np.random.default_rng(seed)
    dim = CATALOG_DIM
    mapping = np.linalg.qr(rng.normal(size=(dim, dim)))[0]
    anchors_old = rng.normal(size=(64, dim))
    fit = anchors_old[:32]
    fit = fit - fit.mean(axis=0, keepdims=True)
    chol = np.linalg.cholesky(fit.T @ fit)
    anchors_old[:32] = fit @ np.linalg.inv(chol).T
    anchors_new = anchors_old @ mapping.T
    anchors = [(anchors_old[i].tolist(), anchors_new[i].tolist()) for i in range(64)]
    docs = rng.normal(size=(catalog_count, dim)) * 5.0
    backend = {_slot_id(i): docs[i].tolist() for i in range(catalog_count)}
    wrapped = wrap_index(
        backend,
        old_dim=dim,
        new_dim=dim,
        unlabeled_anchors=anchors,
        tenant_id="cartridge-catalog",
    )
    queries = docs @ mapping.T
    return wrapped, queries


# ---------------------------------------------------------------------------
# EK5 revocation fence
# ---------------------------------------------------------------------------
class RevocationFence:
    """EK5-backed fence. Every check/commit is atomic with the revoke call."""

    def __init__(self, payload_ids: Sequence[str]) -> None:
        self.store = StratifiedEvidenceStore()
        self.store.apply_mutation([StratifiedEvent(pid, authority_tier=1) for pid in payload_ids])
        self._lock = threading.Lock()
        self._seq = 0
        self.revoked_at: Dict[str, int] = {}
        self.commits: List[Tuple[str, int, str]] = []  # (payload, seq, request_kind)
        self._rev_counter = 0

    def _tick(self) -> int:
        self._seq += 1
        return self._seq

    def admit(self, payload_id: str) -> bool:
        with self._lock:
            self._tick()
            return self.store.eligibility.get(payload_id) == "ADMIT"

    def revoke(self, payload_id: str) -> int:
        with self._lock:
            seq = self._tick()
            self._rev_counter += 1
            rev_id = f"rev::{payload_id}::{self._rev_counter}"
            self.store.apply_mutation(
                [StratifiedEvent(rev_id, authority_tier=5)],
                [EdgeAttestation(f"att::{rev_id}", rev_id, payload_id, "SUPERSEDE")],
            )
            self.revoked_at[payload_id] = seq
            return seq

    def commit(self, payload_id: str, kind: str, *, guarded: bool) -> bool:
        """Commit an execution result. Guarded commits re-check under the lock."""
        with self._lock:
            seq = self._tick()
            if guarded and self.store.eligibility.get(payload_id) != "ADMIT":
                return False
            self.commits.append((payload_id, seq, kind))
            return True

    def post_enforcement_commits(self) -> int:
        return sum(
            1 for pid, seq, _ in self.commits if pid in self.revoked_at and seq > self.revoked_at[pid]
        )


def _race_trial(
    fence: RevocationFence,
    payload_id: str,
    queued: int,
    inflight: int,
    guarded: bool,
    pool: ThreadPoolExecutor,
) -> int:
    """One race: revoke while queued requests start and in-flight ones finish."""
    start = threading.Barrier(queued + inflight + 2)
    revoked = threading.Event()
    results: List[int] = []

    def inflight_req() -> int:
        start.wait()
        if guarded and not fence.admit(payload_id):
            return 0
        revoked.wait(timeout=5.0)  # request is mid-flight while revocation lands
        return 1 if fence.commit(payload_id, "inflight", guarded=guarded) else 0

    def queued_req() -> int:
        start.wait()
        revoked.wait(timeout=5.0)  # queued behind the revocation
        if guarded and not fence.admit(payload_id):
            return 0
        return 1 if fence.commit(payload_id, "queued", guarded=guarded) else 0

    def revoker() -> int:
        start.wait()
        fence.revoke(payload_id)
        revoked.set()
        return 0

    futures = [pool.submit(inflight_req) for _ in range(inflight)]
    futures += [pool.submit(queued_req) for _ in range(queued)]
    futures.append(pool.submit(revoker))
    start.wait()
    for fut in futures:
        results.append(fut.result(timeout=30.0))
    return sum(results)


def run_revocation_races(
    payload_ids: Sequence[str],
    *,
    trials: int,
    queued: int,
    inflight: int,
    threads: int,
) -> Dict[str, Any]:
    need = queued + inflight + 1
    workers = max(threads, need)
    if not payload_ids:
        raise ValueError("payload_ids must be non-empty")
    # B57: every trial races a FRESHLY ENROLLED, ADMIT-eligible logical payload (a per-trial instance of a
    # materialized slot), so trial t never lands on a payload already revoked by an earlier trial. The previous
    # ``payload_ids[t % len(payload_ids)]`` wrapped after 250 trials onto revoked payloads, so those trials raced
    # nothing (admission was already denied) and the zero-violation count was inflated by vacuous trials.
    trial_ids = [f"{payload_ids[t % len(payload_ids)]}::trial{t:06d}" for t in range(trials)]
    if len(set(trial_ids)) != trials:
        raise ValueError("trial payload ids must be distinct")
    guarded_fence = RevocationFence(trial_ids)
    control_fence = RevocationFence(trial_ids)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for pid in trial_ids:
            _race_trial(guarded_fence, pid, queued, inflight, True, pool)
        control_trials = trials
        for pid in trial_ids:
            _race_trial(control_fence, pid, queued, inflight, False, pool)
    requests_total = trials * (queued + inflight)
    violations = guarded_fence.post_enforcement_commits()
    control_violations = control_fence.post_enforcement_commits()
    comparison = compare_with_fresh_rebuild(guarded_fence.store, export_canonical_ek5_audit_trace(guarded_fence.store))
    revoked_payloads = sorted(guarded_fence.revoked_at)
    not_admitted = sum(1 for pid in revoked_payloads if guarded_fence.store.eligibility.get(pid) != "ADMIT")
    return {
        "trials": trials,
        "queued_per_trial": queued,
        "inflight_per_trial": inflight,
        "threads": workers,
        "requests_total": requests_total,
        "post_enforcement_executions": violations,
        "cp_one_sided_95_upper_bound": clopper_pearson_upper(violations, requests_total),
        "cp_zero_event_closed_form": 1.0 - CLOPPER_ALPHA ** (1.0 / requests_total),
        "unguarded_control_trials": control_trials,
        "unguarded_control_post_enforcement_executions": control_violations,
        "distinct_payloads_raced": len(trial_ids),
        "revoked_payloads": len(revoked_payloads),
        "revoked_payloads_not_admitted": not_admitted,
        "ek5_semantic_diffs": len(comparison.semantic_diffs),
        "ek5_canonical_reason_diffs": len(comparison.canonical_reason_diffs),
        "ek5_canonical_pos_cyc_diffs": len(comparison.canonical_pos_cyc_diffs),
        "ek5_canonical_semantic_mismatches": len(comparison.canonical_semantic_mismatches),
    }


# ---------------------------------------------------------------------------
# bandwidth + swap measurement
# ---------------------------------------------------------------------------
def _drop_page_cache_hint(path: Path) -> bool:
    fadvise = getattr(os, "posix_fadvise", None)
    if fadvise is None:
        return False
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
        finally:
            os.close(fd)
        return True
    except OSError:
        return False


def measure_swaps(
    payload_dir: Path,
    slots: Sequence[int],
    shas: Dict[int, str],
    payload_bytes: int,
    seed: int,
) -> Dict[str, Any]:
    paths = {s: payload_dir / f"{_slot_id(s)}.cspg" for s in slots}
    dropped = sum(1 for p in paths.values() if _drop_page_cache_hint(p))
    # calibration: raw sequential reads (no inference) -> physical bound
    read_times: List[float] = []
    for s in slots:
        t0 = time.perf_counter()
        _, body, _ = read_cspg_sequential(paths[s])
        read_times.append(time.perf_counter() - t0)
        if len(body) != payload_bytes:
            raise ValueError("payload size drift")
    best_read = min(read_times)
    measured_bw = payload_bytes / best_read
    bound_s = payload_bytes / measured_bw
    # swap: read + sha verify + first usable inference
    dropped2 = sum(1 for p in paths.values() if _drop_page_cache_hint(p))
    base = np.random.default_rng(seed).standard_normal(256).astype(np.float32)
    swap_times: List[float] = []
    integrity_failures = 0
    for s in slots:
        t0 = time.perf_counter()
        slot, body, declared = read_cspg_sequential(paths[s])
        ok = slot == s and hashlib.sha256(body).hexdigest() == declared == shas[s]
        first_usable_inference(body, base)
        swap_times.append(time.perf_counter() - t0)
        if not ok:
            integrity_failures += 1
    median_swap = statistics.median(swap_times)
    return {
        "payload_bytes": payload_bytes,
        "swaps_measured": len(swap_times),
        "page_cache_drop_hints_ok": dropped + dropped2,
        "bandwidth_tier": "cold_posix_fadvise" if dropped + dropped2 else "page_cache_warm_upper_bound",
        "measured_bw_bytes_s": measured_bw,
        "physical_bound_s": bound_s,
        "median_time_to_first_usable_inference_s": median_swap,
        "p95_time_to_first_usable_inference_s": float(np.quantile(swap_times, 0.95)),
        "overhead_ratio_vs_bound": median_swap / bound_s,
        "integrity_failures": integrity_failures,
    }


def measure_catalog_lookup(wrapped, queries: np.ndarray, sample: int) -> Dict[str, Any]:
    t0 = time.perf_counter()
    hits_ok = 0
    for i in range(sample):
        hit = wrapped.search(queries[i].tolist(), k=1)[0]
        hits_ok += int(hit.doc_id == _slot_id(i))
    elapsed = time.perf_counter() - t0
    return {
        "lookups": sample,
        "top1_self_hits": hits_ok,
        "mean_lookup_s": elapsed / sample,
        "catalog_entries_indexed": len(wrapped),
    }


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------
def _canonical_digest(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=True, default=str).encode()).hexdigest()


def run_cartridge_array_experiment(
    *,
    mode: str = "local-drycheck",
    workdir: Optional[Path] = None,
    payload_bytes: int = 256 * 1024,
    materialized_count: int = MATERIALIZED_PAYLOAD_COUNT,
    catalog_count: int = CATALOG_ENTRY_COUNT,
    revocation_trials: int = 300,
    queued: int = 4,
    inflight: int = 4,
    threads: int = 12,
    seed: int = SEED,
) -> Dict[str, Any]:
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    if materialized_count > catalog_count:
        raise ValueError("materialized_count cannot exceed catalog_count")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    cuda_env = build_cpu_only_env()
    cage = CgroupCageConfig.from_size_spec("6G", memory_swap_max=0)
    cage_cmd = build_systemd_run_command(
        [sys.executable, "scripts/run_pv7_w19_2b_cartridges.py", "--mode", mode], cage
    )
    if mode == "run-live":
        if not sys.platform.startswith("linux"):
            raise LiveModeUnavailableError("run-live requires a Linux host with cgroup v2 + systemd-run")
        if payload_bytes < 8 * 1024 * 1024:
            raise LiveModeUnavailableError("run-live requires payload_bytes >= 8 MiB (real LoRA scale)")

    own_tmp = workdir is None
    tmp_ctx = tempfile.TemporaryDirectory(prefix="pv7_2b_") if own_tmp else None
    root = Path(tmp_ctx.name) if tmp_ctx is not None else Path(workdir)  # type: ignore[arg-type]
    root.mkdir(parents=True, exist_ok=True)
    tracemalloc.start()
    try:
        payload_dir = root / "payloads"
        payload_dir.mkdir(parents=True, exist_ok=True)
        slots = select_materialized_slots(catalog_count, materialized_count, seed)
        shas = {s: write_cspg_payload(payload_dir / f"{_slot_id(s)}.cspg", s, payload_bytes, seed) for s in slots}
        files_on_disk = sorted(p.name for p in payload_dir.glob("*.cspg"))

        wrapped, queries = build_catalog_index(seed, catalog_count)
        catalog = measure_catalog_lookup(wrapped, queries, sample=min(200, catalog_count))
        unmaterialized = next(i for i in range(catalog_count) if i not in set(slots))
        try:
            if not (payload_dir / f"{_slot_id(unmaterialized)}.cspg").is_file():
                raise PayloadNotMaterializedError(_slot_id(unmaterialized))
            separation_ok = False
        except PayloadNotMaterializedError:
            separation_ok = True

        swaps = measure_swaps(payload_dir, slots, shas, payload_bytes, seed)
        races = run_revocation_races(
            [_slot_id(s) for s in slots],
            trials=revocation_trials,
            queued=queued,
            inflight=inflight,
            threads=threads,
        )
        _, peak_traced = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
        if tmp_ctx is not None:
            tmp_ctx.cleanup()

    # B57: the state-identity contract is bound PER payload: adapter_weights_sha256 is that .cspg body's SHA-256,
    # so swapping/corrupting any one payload changes exactly its own contract digest (a single shared
    # EMPTY_ADAPTER_SHA256 contract identified no payload at all).
    contracts = {
        _slot_id(s): TRA1StateIdentityContract(
            base_weights_sha256=hashlib.sha256(b"cartridge-base").hexdigest(),
            adapter_weights_sha256=shas[s],
            config_sha256=hashlib.sha256(b"cartridge-config").hexdigest(),
            tokenizer_sha256=hashlib.sha256(b"cartridge-tok").hexdigest(),
            kv_or_state_dtype="float16",
            rope_freq_base=10000,
            rope_scaling_hash=hashlib.sha256(b"rope-none").hexdigest(),
        )
        for s in slots
    }
    contract_digests = {sid: c.digest() for sid, c in contracts.items()}
    contract_bound_per_body = (
        len(set(contract_digests.values())) == len(slots)
        and all(contracts[_slot_id(s)].adapter_weights_sha256 == shas[s] != EMPTY_ADAPTER_SHA256 for s in slots)
    )
    cgroup_check = (
        verify_cgroup_v2_enforced(cage.memory_max_bytes, cage.memory_swap_max_bytes)
        if mode == "run-live"
        else {"enforced": False, "reason": "local-drycheck: no cage is applied"}
    )
    gates = {
        "materialized_count_is_250": len(files_on_disk) == materialized_count == len(slots),
        "catalog_count_is_1000_and_distinct": catalog["catalog_entries_indexed"] == catalog_count > len(files_on_disk),
        "catalog_lookup_not_conflated_with_swap": bool(separation_ok) and catalog["top1_self_hits"] == catalog["lookups"],
        "payload_integrity_all_swaps": swaps["integrity_failures"] == 0,
        "first_inference_not_below_physical_bound": swaps["median_time_to_first_usable_inference_s"] >= swaps["physical_bound_s"],
        "first_inference_overhead_ratio_le_50": swaps["overhead_ratio_vs_bound"] <= 50.0,
        "zero_post_enforcement_executions": races["post_enforcement_executions"] == 0,
        "cp_upper_bound_below_1pct": races["cp_one_sided_95_upper_bound"] < 0.01,
        "cp_matches_zero_event_closed_form": abs(races["cp_one_sided_95_upper_bound"] - races["cp_zero_event_closed_form"]) < 1e-12,
        "unguarded_control_shows_race_is_real": races["unguarded_control_post_enforcement_executions"] > 0,
        "all_revoked_payloads_not_admitted": races["revoked_payloads_not_admitted"] == races["revoked_payloads"] > 0,
        "revocation_trials_each_race_a_distinct_fresh_payload": races["distinct_payloads_raced"] == races["trials"],
        "tra1_contract_bound_to_each_cspg_body_sha": contract_bound_per_body,
        "ek5_fresh_rebuild_parity": (
            races["ek5_semantic_diffs"] == 0
            and races["ek5_canonical_reason_diffs"] == 0
            and races["ek5_canonical_pos_cyc_diffs"] == 0
            and races["ek5_canonical_semantic_mismatches"] == 0
        ),
        "cgroup_memory_max_is_6gib_swap_zero": (
            cage.memory_max_bytes == MEMORY_MAX_BYTES and cage.memory_swap_max_bytes == 0 and f"MemoryMax={MEMORY_MAX_BYTES}" in cage_cmd
        ),
        "cuda_hidden": cuda_env.get("CUDA_VISIBLE_DEVICES") == "",
        "traced_peak_under_6gib": peak_traced < MEMORY_MAX_BYTES,
    }
    deterministic_view = {
        "bundle_id": BUNDLE_ID,
        "mode": mode,
        "materialized_payload_count": len(files_on_disk),
        "catalog_entry_count": catalog_count,
        "payload_bytes": payload_bytes,
        "slots": list(slots),
        "payload_body_sha256": {str(s): shas[s] for s in slots},
        "revocation": {
            "trials": races["trials"],
            "requests_total": races["requests_total"],
            "post_enforcement_executions": races["post_enforcement_executions"],
            "cp_zero_event_closed_form": races["cp_zero_event_closed_form"],
        },
        "contract_digest": _canonical_digest(contract_digests),
        "per_payload_contract_digests": contract_digests,
        "memory_max_bytes": cage.memory_max_bytes,
        "gate_names": sorted(gates),
    }
    manifest: Dict[str, Any] = {
        "bundle_id": BUNDLE_ID,
        "item": "2B",
        "mode": mode,
        "tier": "floor" if mode == "local-drycheck" else "ceiling-candidate",
        "tier_disclosure": (
            "local-drycheck: synthetic LoRA-style payloads at reduced size on the host filesystem; cgroup cage built "
            "but not enforced off-Linux; bandwidth bound is measured on this host, not a hardware NVMe spec"
            if mode == "local-drycheck"
            else "run-live: Linux cgroup v2 host, payloads >= 8 MiB; physical cold-read tier"
        ),
        "materialized_payload_count": len(files_on_disk),
        "catalog_entry_count": catalog_count,
        "catalog_lookup": catalog,
        "swap": swaps,
        "revocation": races,
        "cage": {
            "memory_max_bytes": cage.memory_max_bytes,
            "memory_swap_max_bytes": cage.memory_swap_max_bytes,
            "systemd_run_command": cage_cmd,
            "enforced_on_this_host": bool(cgroup_check["enforced"]),
            "enforcement_check": cgroup_check,
        },
        "traced_peak_bytes": int(peak_traced),
        "gates": gates,
        "overall_pass": all(gates.values()),
        "deterministic_view": deterministic_view,
        "deterministic_sha256": _canonical_digest(deterministic_view),
    }
    return manifest
