"""Disk IVF retrieval, immutable EK5 admission, and a matched SQ8 challenger."""

from __future__ import annotations

import hashlib
import json
import mmap
import os
import re
import shutil
import struct
import threading
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import numpy as np

from chelated_pv7_spine.bridge_merge_harness import l2_normalize
from chelated_pv7_spine.cpu_io_cage import ONE_GIB_BYTES, diff_block_stat, parse_sys_block_stat
from chelated_pv7_spine.ek5_reason_canonicalizer import export_canonical_ek5_audit_trace
from ek5_v3_stratified import EdgeAttestation, StratifiedEvent, StratifiedEvidenceStore

MIB = 1024 ** 2
MEMORY_MAX = 4 * ONE_GIB_BYTES
TRACE_PAIRS = 30
TOP_K = 10
LIVE_DOC_MIN = 9_465_000
LIVE_DOC_MAX = 9_475_000
READ_BATCH = 4096
FENCE_WAL_MAGIC = b"EK5FWAL1"
_WAL_HEAD = struct.Struct("<cQI")
_WAL_CRC = struct.Struct("<I")
SECTOR_BYTES = 512
JOURNAL_CODECS = ("shared", "as_built_asymmetric", "no_durability_floor")
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 58
_RESTRICTED_INPUT = re.compile(r"sealed_key|blind-followup|pm4_human_audit_100_blind_sheet\.csv", re.I)


def input_path(root, relative):
    candidate = Path(relative)
    if candidate.is_absolute() or _RESTRICTED_INPUT.search(candidate.as_posix()):
        raise ValueError("Invalid input path in the staged corpus manifest")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root) or _RESTRICTED_INPUT.search(resolved.as_posix()):
        raise ValueError("Corpus input must resolve inside the staging directory")
    return resolved


def digest_file(path):
    with path.open("rb") as source:
        digest = hashlib.sha256()
        for block in iter(lambda: source.read(MIB), b""):
            digest.update(block)
    return digest.hexdigest()


def normalized(array):
    rows = np.asarray(array, dtype=np.float32)
    norms = np.linalg.norm(rows, axis=-1)
    if not np.all(np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError("Vectors must have finite, positive norms")
    return l2_normalize(rows).astype(np.float32, copy=False)


class VectorCorpus:
    """Read one consistent ordered ID space from memory-mapped floating-point NPY shards."""

    def __init__(self, shards):
        self.shards = tuple(shards)
        if not self.shards or any(x.ndim != 2 or len(x) == 0 or x.dtype.kind != "f" for x in self.shards):
            raise ValueError("Expected nonempty floating-point matrix shards")
        self.dimension = self.shards[0].shape[1]
        if self.dimension < 2 or any(x.shape[1] != self.dimension for x in self.shards):
            raise ValueError("Shard dimensions differ")
        self.ends = np.cumsum([len(shard) for shard in self.shards])
        self.count = int(self.ends[-1])
        if self.count >= 2 ** 32:
            raise ValueError("The extent format supports fewer than 2^32 document IDs")

    def batches(self):
        base = 0
        for shard in self.shards:
            for start in range(0, len(shard), READ_BATCH):
                block = normalized(shard[start:start + READ_BATCH])
                yield base + start, block
            base += len(shard)

    def get(self, ids):
        ids = np.asarray(ids, dtype=np.int64)
        result = np.empty((len(ids), self.dimension), dtype=np.float32)
        for shard_idx in np.unique(np.searchsorted(self.ends, ids, side="right")):
            start = int(self.ends[shard_idx - 1]) if shard_idx else 0
            matches = (ids >= start) & (ids < self.ends[shard_idx])
            result[matches] = self.shards[shard_idx][ids[matches] - start]
        return normalized(result)

    def exact_top(self, queries, depth):
        """Use the original float vectors as the common ground truth for all arms."""
        labels = [np.empty(0, dtype=np.uint32) for _ in queries]
        scores = [np.empty(0, dtype=np.float32) for _ in queries]
        for start, block in self.batches():
            ids = np.arange(start, start + len(block), dtype=np.uint32)
            similarities = block @ queries.T
            for idx, result in enumerate(similarities.T):
                labels[idx], scores[idx] = top_scores(labels[idx], scores[idx], ids, result, depth)
        return labels


def top_scores(old_ids, old_scores, new_ids, new_scores, k):
    ids = np.concatenate((old_ids, new_ids))
    scores = np.concatenate((old_scores, new_scores))
    if len(ids) > k:
        threshold = np.partition(scores, len(scores) - k)[len(scores) - k]
        eligible = np.flatnonzero(scores >= threshold)
    else:
        eligible = np.arange(len(ids))
    order = eligible[np.lexsort((ids[eligible], -scores[eligible]))][:k]
    return ids[order], scores[order]


@dataclass(frozen=True)
class ExtentGeometry:
    path: Path
    extent_bytes: int
    slots: int
    counts: np.ndarray
    extent_clusters: np.ndarray
    centroids: np.ndarray
    locations: np.ndarray
    mean: np.ndarray
    sidecar: np.ndarray
    row_type: np.dtype

    def probed_extents(self, query, nprobe):
        scores = self.centroids @ query
        cluster_ids = np.arange(len(self.centroids))
        order = np.lexsort((cluster_ids, -scores))[:nprobe]
        return np.flatnonzero(np.isin(self.extent_clusters, order))


def build_extents(corpus, centroids, dev_queries, directory, extent_bytes, *, csls_k=10):
    """Materialize fixed-sized sequential extents and two float32 sidecar values per physical ID."""
    if type(extent_bytes) is not int or not 2 * MIB <= extent_bytes <= 8 * MIB:
        raise ValueError("Extent size must be between 2 MiB and 8 MiB inclusive")
    centroids = normalized(centroids)
    dev_queries = normalized(dev_queries)
    if centroids.ndim != 2 or not len(centroids) or centroids.shape[1] != corpus.dimension:
        raise ValueError("Malformed IVF centroids")
    if not 1 <= csls_k <= len(dev_queries) or dev_queries.shape[1] != corpus.dimension:
        raise ValueError("Insufficient or invalid DEV CSLS calibration vectors")
    dtype = np.dtype([("id", "<u4"), ("vec", "i1", (corpus.dimension,))])
    slots = extent_bytes // dtype.itemsize
    if not slots:
        raise ValueError("A vector does not fit in the configured extent")
    assigned = np.empty(corpus.count, np.int32)
    mean = np.zeros(corpus.dimension, np.float64)
    for start, block in corpus.batches():
        assigned[start:start + len(block)] = np.argmax(block @ centroids.T, axis=1)
        mean += np.sum(block, axis=0, dtype=np.float64)
    mean = (mean / corpus.count).astype(np.float32)
    dev_centered = normalized(dev_queries - mean)
    counts = np.bincount(assigned, minlength=len(centroids))
    extent_counts = (counts + slots - 1) // slots
    total_extents = int(extent_counts.sum())
    if total_extents * slots >= 2 ** 32:
        raise ValueError("Padded extent slots exceed the addressable ID space")
    extent_clusters = np.repeat(np.arange(len(centroids), dtype=np.int32), extent_counts)
    extent_sizes = np.concatenate([
        np.full(number, slots, dtype=np.int32) for number in extent_counts if number
    ])
    extent_ends = np.cumsum(extent_counts)
    has_tail = (counts % slots) > 0
    extent_sizes[extent_ends[has_tail] - 1] = counts[has_tail] % slots
    next_location = (extent_ends - extent_counts) * slots
    locations = np.empty(corpus.count, dtype=np.uint32)
    sidecar = np.empty((corpus.count, 2), dtype=np.float32)
    path = Path(directory) / "ivf-flat-int8.extents"
    with path.open("x+b") as target:
        target.truncate(total_extents * extent_bytes)
        with mmap.mmap(target.fileno(), 0, access=mmap.ACCESS_WRITE) as mapping:
            storage = np.ndarray(
                (total_extents, slots), dtype=dtype, buffer=mapping, strides=(extent_bytes, dtype.itemsize),
            )
            for start, block in corpus.batches():
                stop = start + len(block)
                centered = block - mean
                center_norms = np.linalg.norm(centered, axis=1)
                if np.any(center_norms <= 0):
                    raise ValueError("Zero norm after corpus centering")
                similarities = (centered / center_norms[:, None]) @ dev_centered.T
                cutoff = similarities.shape[1] - csls_k
                sidecar[start:stop, 0] = center_norms
                sidecar[start:stop, 1] = np.partition(similarities, cutoff, axis=1)[:, cutoff:].mean(axis=1)
                quantized = np.rint(block * 127).astype(np.int8)
                labels = assigned[start:stop]
                for cluster in np.unique(labels):
                    selected = np.flatnonzero(labels == cluster)
                    addresses = np.arange(next_location[cluster], next_location[cluster] + len(selected))
                    next_location[cluster] += len(selected)
                    locations[start + selected] = addresses
                    extent, slot = np.divmod(addresses, slots)
                    storage["id"][extent, slot] = start + selected
                    storage["vec"][extent, slot] = quantized[selected]
            del storage
            mapping.flush()
        os.fsync(target.fileno())
    return ExtentGeometry(path, extent_bytes, slots, extent_sizes, extent_clusters,
                          centroids, locations, mean, sidecar, dtype)


class PackedFence:
    """One denial bit per extent/slot, addressed by the immutable ID-to-slot map."""

    def __init__(self, geometry):
        self.locations = geometry.locations
        self.bits = np.zeros((len(geometry.counts) * geometry.slots + 7) // 8, dtype=np.uint8)
        self.epoch = 0

    def allowed(self, ids):
        pos = self.locations[ids]
        return (self.bits[pos >> 3] & (1 << (pos & 7))) == 0

    def publish(self, ids, admitted):
        pos = self.locations[ids]
        mask = (1 << (pos & 7)).astype(np.uint8)
        if admitted:
            np.bitwise_and.at(self.bits, pos >> 3, ~mask)
        else:
            np.bitwise_or.at(self.bits, pos >> 3, mask)
        self.epoch += 1


def doc_event_id(doc_id):
    return f"doc-{doc_id}"


class FenceWALError(RuntimeError):
    """The fence write-ahead log is corrupt before its final record."""


def journal_record(kind, epoch, ids=()):
    """The one WAL record codec shared by Cold-OSD and the SQ8 challenger.

    ``kind | epoch:u64 | count:u32 | ids:u32[count] | crc32``.  Only a ``W`` record carries ids;
    ``C`` (compaction) and ``R`` (reinstatement) close the single open trace and carry none, on
    both arms.  Both arms charge the same 8-byte ``EK5FWAL1`` header when they create the file.
    """
    ids = np.asarray(ids, dtype="<u4")
    body = _WAL_HEAD.pack(kind, epoch, len(ids)) + ids.tobytes()
    return body + _WAL_CRC.pack(zlib.crc32(body))


_fence_wal_record = journal_record


def sector_rounded(nbytes):
    """Bytes a 512-byte block device charges for one fsynced record of ``nbytes``."""
    return -(-nbytes // SECTOR_BYTES) * SECTOR_BYTES


def scan_journal(raw):
    """Parse a shared-codec journal: returns ``(records, durable_end, torn_bytes)``.

    ``records`` are ``(kind, epoch, ids, offset, end)``.  A torn final record (short or failing its
    CRC at end of file) is an epoch that was never published; damage followed by further bytes or a
    non-consecutive epoch raises ``FenceWALError``.  Record kinds are validated by the caller.
    """
    if raw[:len(FENCE_WAL_MAGIC)] != FENCE_WAL_MAGIC:
        if len(raw) < len(FENCE_WAL_MAGIC) and FENCE_WAL_MAGIC.startswith(raw):
            raise FenceWALError("The fence WAL header itself is torn; no epoch was ever durable")
        raise FenceWALError("Not an EK5 fence WAL")
    offset = len(FENCE_WAL_MAGIC)
    records = []
    torn = 0
    previous = 0
    while offset < len(raw):
        head_end = offset + _WAL_HEAD.size
        if head_end > len(raw):
            torn = len(raw) - offset
            break
        kind, epoch, count = _WAL_HEAD.unpack_from(raw, offset)
        end = head_end + 4 * count + _WAL_CRC.size
        if end > len(raw):
            torn = len(raw) - offset
            break
        (crc,) = _WAL_CRC.unpack_from(raw, end - _WAL_CRC.size)
        if crc != zlib.crc32(raw[offset:end - _WAL_CRC.size]):
            if end == len(raw):
                torn = len(raw) - offset
                break
            raise FenceWALError(f"Fence WAL record at byte {offset} fails its CRC before end of log")
        if epoch != previous + 1:
            raise FenceWALError(f"Fence WAL epoch {epoch} does not follow epoch {previous}")
        ids = np.frombuffer(raw, dtype="<u4", count=count, offset=head_end).astype(np.int64).tolist()
        records.append((kind, epoch, ids, offset, end))
        previous = epoch
        offset = end
    return records, offset, torn


def _fsync_parent(path):
    # Make a newly created WAL name durable as well; Windows cannot open directories for fsync.
    if os.name == "posix":
        handle = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(handle)
        finally:
            os.close(handle)


class EK5ExtentFence(PackedFence):
    """Publish an epoch from EK5 v3 authority/refutation decisions over changed documents.

    With ``wal_path`` every epoch is first appended to a binary write-ahead log and fsynced, then
    published to the in-memory denial bits.  A ``W`` record carries the epoch and the withdrawn IDs;
    an ``R`` record carries only the epoch because reinstatement must exactly close the active
    trace.  Each record is ``kind | epoch:u64 | count:u32 | ids:u32[count] | crc32``.
    ``wal_bytes`` counts every fsynced byte, including the 8-byte file header.
    """

    def __init__(self, geometry, *, wal_path=None):
        super().__init__(geometry)
        self.store = None
        self.active_ids = set()
        self.wal_path = None if wal_path is None else Path(wal_path)
        self.wal_bytes = 0
        self.wal_records = 0
        self.wal_sector_bytes = 0
        self.recovery = None
        if self.wal_path is not None:
            with self.wal_path.open("xb", buffering=0) as wal:
                self._append(wal, FENCE_WAL_MAGIC)
            _fsync_parent(self.wal_path)

    def _append(self, wal, data):
        written = wal.write(data)
        if written != len(data):
            raise OSError("Short fence WAL write")
        os.fsync(wal.fileno())
        self.wal_bytes += written

    def _log(self, kind, ids=()):
        if self.wal_path is None:
            return
        record = journal_record(kind, self.epoch + 1, ids)
        with self.wal_path.open("ab", buffering=0) as wal:
            self._append(wal, record)
        self.wal_records += 1
        self.wal_sector_bytes += sector_rounded(len(record))

    def withdraw(self, ids):
        if self.active_ids:
            raise RuntimeError("Reinstate the current trace before starting another")
        store = StratifiedEvidenceStore()
        events = []
        edges = []
        for doc_id in ids:
            event_id = doc_event_id(doc_id)
            events.extend([
                StratifiedEvent(event_id, entity=event_id, claim_value="indexed", authority_tier=1),
                StratifiedEvent(f"refute-{doc_id}", entity=event_id, authority_tier=2, valid_time=1, tx_time=1),
            ])
            edges.append(EdgeAttestation(f"edge-{doc_id}", f"refute-{doc_id}", event_id, "REFUTE", 1, 1))
        store.apply_mutation(events, edges)
        rejected = [doc_id for doc_id in ids if store.eligibility[doc_event_id(doc_id)] != "ADMIT"]
        # Durable before visible: the epoch reaches stable storage before readers observe it.
        self._log(b"W", sorted(int(doc_id) for doc_id in ids))
        self.store = store
        self.publish(rejected, False)
        self.active_ids = set(ids)

    def reinstate(self, ids):
        if not self.active_ids or set(ids) != self.active_ids:
            raise ValueError("Reinstatement must exactly close the active trace")
        self.store.apply_mutation(withdraw_attestation_ids=[f"edge-{doc_id}" for doc_id in ids])
        restored = [doc_id for doc_id in ids if self.store.eligibility[doc_event_id(doc_id)] == "ADMIT"]
        self._log(b"R")
        self.publish(restored, True)
        self.active_ids.clear()

    def audit_digest(self):
        return export_canonical_ek5_audit_trace(self.store).digest()

    @classmethod
    def recover_from_wal(cls, geometry, wal_path):
        """Rebuild the fence, EK5 store and active trace by replaying a fence WAL.

        A torn final record (short or failing its CRC at end of file) is an epoch that was never
        published, so it is truncated away and reported.  Damage followed by further bytes, an
        unknown record kind, or a non-consecutive epoch raises ``FenceWALError``.
        """
        path = Path(wal_path)
        raw = path.read_bytes()
        scanned, offset, torn = scan_journal(raw)
        # Replay with logging detached; the log is reattached only after the durable prefix is fixed.
        fence = cls(geometry)
        records = 0
        sectors = 0
        for kind, epoch, ids, start, end in scanned:
            if kind == b"W":
                fence.withdraw(ids)
            elif kind == b"R" and not ids:
                fence.reinstate(sorted(fence.active_ids))
            else:
                raise FenceWALError(f"Unknown fence WAL record at byte {start}")
            records += 1
            sectors += sector_rounded(end - start)
        if torn:
            with path.open("r+b", buffering=0) as wal:
                wal.truncate(offset)
                os.fsync(wal.fileno())
        fence.wal_path = path
        fence.wal_bytes = offset
        fence.wal_records = records
        fence.wal_sector_bytes = sectors
        fence.recovery = {
            "records_replayed": records, "epoch": fence.epoch, "torn_tail_bytes": torn,
            "active_trace_open": bool(fence.active_ids), "durable_bytes": offset,
        }
        return fence


class IVFReader:
    def __init__(self, geometry, path, fence, nprobe, *, chelated):
        if not 1 <= nprobe <= len(geometry.centroids):
            raise ValueError("nprobe must be within the IVF centroid count")
        self.geometry = geometry
        self.path = path
        self.fence = fence
        self.nprobe = nprobe
        self.chelated = chelated
        self.counts = geometry.counts.copy()
        self.written = 0
        self.lock = threading.RLock()

    def search(self, query, k=TOP_K):
        g = self.geometry
        ids = np.empty(0, dtype=np.uint32)
        scores = np.empty(0, dtype=np.float32)
        query = normalized(np.asarray(query)[None])[0]
        centered = normalized((query - g.mean)[None])[0] if self.chelated else query
        center_shift = float(g.mean @ centered) if self.chelated else 0.0
        # For a single score space r_q is a query-wide constant and cancels from top-k ordering.
        with self.lock, self.path.open("rb", buffering=0) as source:
            for extent in g.probed_extents(query, self.nprobe):
                source.seek(int(extent) * g.extent_bytes)
                occupied = int(self.counts[extent])
                raw = source.read(occupied * g.row_type.itemsize)
                if len(raw) != occupied * g.row_type.itemsize:
                    raise OSError(f"Short read in extent {extent}")
                records = np.frombuffer(raw, dtype=g.row_type)
                for start in range(0, occupied, READ_BATCH):
                    batch = records[start:start + READ_BATCH]
                    batch = batch[self.fence.allowed(batch["id"])]
                    if not len(batch):
                        continue
                    vectors = batch["vec"].astype(np.float32) / 127
                    sim = vectors @ centered
                    if self.chelated:
                        norm, density = g.sidecar[batch["id"]].T
                        sim = 2 * (sim - center_shift) / norm - density
                    else:
                        sim /= np.maximum(np.linalg.norm(vectors, axis=1), np.finfo(np.float32).tiny)
                    ids, scores = top_scores(ids, scores, batch["id"], sim, k)
        return ids.tolist()

    def quiescent(self):
        return True

    def journal_stats(self):
        """``(fsynced journal records, 512-byte-sector-rounded journal bytes)`` written so far."""
        return 0, 0

    def visible(self, ids):
        with self.lock:
            return self.fence.allowed(ids).tolist()


class ColdOSD(IVFReader):
    """Immutable extents plus an EK5 fence; ``written`` counts the fence's fsynced WAL bytes."""

    def __init__(self, geometry, nprobe, *, wal_path=None, fence=None):
        if fence is not None and wal_path is not None:
            raise ValueError("Pass either a recovered fence or a new WAL path, not both")
        fence = fence if fence is not None else EK5ExtentFence(geometry, wal_path=wal_path)
        super().__init__(geometry, geometry.path, fence, nprobe, chelated=True)
        # A new WAL header is a real durable write made when the engine is opened; a recovered
        # log's existing prefix was written by an earlier process and is not charged again.
        self.written = 0 if fence.recovery is not None else fence.wal_bytes

    @classmethod
    def recover(cls, geometry, nprobe, wal_path):
        return cls(geometry, nprobe, fence=EK5ExtentFence.recover_from_wal(geometry, wal_path))

    def journal_stats(self):
        return self.fence.wal_records, self.fence.wal_sector_bytes

    def _durable(self, action, ids):
        before = self.fence.wal_bytes
        action(ids)
        self.written += self.fence.wal_bytes - before

    def withdraw(self, ids):
        with self.lock:
            self._durable(self.fence.withdraw, ids)

    def reinstate(self, ids):
        with self.lock:
            self._durable(self.fence.reinstate, ids)

    def audit_digest(self):
        return self.fence.audit_digest()


class MatchedIVFSQ8(IVFReader):
    """On-disk SQ8 tombstone WAL plus synchronous, extent-bounded compaction.

    ``journal_codec`` selects the tombstone-journal record format:

    * ``"shared"`` (default, the gating comparison): the exact record codec, header rule and fsync
      discipline of Cold-OSD's fence WAL (``journal_record``): CRC32, epoch, ids only on ``W``.
    * ``"as_built_asymmetric"``: the pre-remediation SQ8 journal (``kind | count:u32 | ids``, with
      ids on ``W``/``C``/``R``, no header, epoch or CRC).  Supplementary row only.
    * ``"no_durability_floor"``: ``kind | count:u32 | ids`` with ids only on ``W`` and no header,
      epoch or CRC.  This is the cheapest journal that still names the withdrawn set; it is not
      crash-verifiable, so it is the challenger's best case and is reported as such.
    """

    def __init__(self, geometry, nprobe, directory, *, compact_fraction=0.2, journal_codec="shared"):
        if not 0 < compact_fraction <= 1:
            raise ValueError('Invalid SQ8 compaction threshold')
        if journal_codec not in JOURNAL_CODECS:
            raise ValueError(f"journal_codec must be one of {JOURNAL_CODECS}")
        self.compact_fraction = compact_fraction
        self.journal_codec = journal_codec
        path = Path(directory) / f"ivf-sq8.{journal_codec}.extents"
        self.wal = Path(directory) / f"ivf-sq8.{journal_codec}.tombstones"
        with geometry.path.open("rb") as source, path.open("xb") as target:
            shutil.copyfileobj(source, target, length=8 * MIB)
        with self.wal.open("xb"):
            pass
        super().__init__(geometry, path, PackedFence(geometry), nprobe, chelated=False)
        self.pending = None
        self.compacted = set()
        self.active_ids = None
        self.journal_epoch = 0
        self.journal_records = 0
        self.journal_sector_bytes = 0
        if journal_codec == "shared":
            # Same header rule as Cold-OSD: the process that creates the log charges its header.
            with self.wal.open("r+b", buffering=0) as wal:
                self._write(wal, FENCE_WAL_MAGIC)
                os.fsync(wal.fileno())

    def _write(self, file, data):
        actual = file.write(data)
        self.written += actual
        if actual != len(data):
            raise OSError("Short on-disk SQ8 write")

    def journal_stats(self):
        return self.journal_records, self.journal_sector_bytes

    def _journal_bytes(self, kind, ids):
        if self.journal_codec == "shared":
            return journal_record(kind, self.journal_epoch + 1, ids if kind == b"W" else ())
        listed = ids if (kind == b"W" or self.journal_codec == "as_built_asymmetric") else ()
        return kind + len(listed).to_bytes(4, "little") + np.asarray(listed, dtype="<u4").tobytes()

    def _journal(self, kind, ids):
        record = self._journal_bytes(kind, ids)
        with self.wal.open("ab", buffering=0) as wal:
            self._write(wal, record)
            os.fsync(wal.fileno())
        self.journal_epoch += 1
        self.journal_records += 1
        self.journal_sector_bytes += sector_rounded(len(record))

    def verify_journal(self):
        """Replay the shared-codec journal with the same scanner Cold-OSD recovery uses."""
        if self.journal_codec != "shared":
            return {"verifiable": False, "reason": f"{self.journal_codec} journal has no epoch or CRC"}
        records, durable, torn = scan_journal(self.wal.read_bytes())
        kinds = [kind for kind, *_ in records]
        if torn or any(kind not in (b"W", b"C", b"R") for kind in kinds):
            raise FenceWALError("SQ8 journal is torn or holds an unknown record kind")
        return {"verifiable": True, "records_replayed": len(records), "durable_bytes": durable,
                "epoch": records[-1][1] if records else 0, "torn_tail_bytes": torn}

    def withdraw(self, ids):
        with self.lock:
            if self.active_ids is not None:
                raise RuntimeError("Previous SQ8 withdrawal is still open")
            self._journal(b"W", sorted(int(doc_id) for doc_id in ids))
            self.fence.publish(ids, False)
            affected, removed = np.unique(self.geometry.locations[ids] // self.geometry.slots, return_counts=True)
            self.pending = affected[removed / self.counts[affected] >= self.compact_fraction]
            self.compacted = set(self.pending.tolist())
            self.active_ids = set(ids)

    def quiescent(self):
        with self.lock:
            if self.pending is None:
                return True
            if not len(self.pending):
                self.pending = None
                return True
            g = self.geometry
            with self.path.open("r+b", buffering=0) as file:
                for extent in self.pending:
                    offset = int(extent) * g.extent_bytes
                    occupied = int(self.counts[extent])
                    file.seek(offset)
                    raw = file.read(occupied * g.row_type.itemsize)
                    if len(raw) != occupied * g.row_type.itemsize:
                        raise OSError("Short SQ8 compaction read")
                    records = np.frombuffer(raw, dtype=g.row_type)
                    surviving = records[self.fence.allowed(records["id"])]
                    file.seek(offset)
                    self._write(file, surviving.tobytes())
                    self.counts[extent] = len(surviving)
                os.fsync(file.fileno())
            self._journal(b"C", sorted(self.active_ids))
            self.pending = None
            return True

    def reinstate(self, ids):
        with self.lock:
            if self.pending is not None or self.active_ids != set(ids):
                raise ValueError("SQ8 reinstatement requires the compacted active trace")
            self._journal(b"R", ids)
            g = self.geometry
            # Append the withdrawn vectors after the compacted survivors; unchanged survivors stay in place.
            affected, slots = np.divmod(g.locations[ids], g.slots)
            by_extent = {int(extent): sorted(slots[affected == extent]) for extent in self.compacted}
            with g.path.open("rb", buffering=0) as source, self.path.open("r+b", buffering=0) as target:
                for extent, slot_ids in by_extent.items():
                    nbytes = g.row_type.itemsize
                    offset = extent * g.extent_bytes
                    source.seek(offset)
                    raw = source.read(int(g.counts[extent]) * nbytes)
                    if len(raw) != int(g.counts[extent]) * nbytes:
                        raise OSError("Short immutable source read on reinstatement")
                    missing = np.frombuffer(raw, dtype=g.row_type)[slot_ids]
                    target.seek(offset + int(self.counts[extent]) * nbytes)
                    self._write(target, missing.tobytes())
                    self.counts[extent] += len(missing)
                os.fsync(target.fileno())
            self.fence.publish(ids, True)
            self.active_ids = None
            self.compacted.clear()
 

class QdrantOnDisk:
    """Live cosine collection, using server-side delete/upsert and optimizer status."""

    def __init__(self, url, collection, corpus, *, api_key=None, hnsw_m=16, ef_construct=100, timeout=120):
        if not re.fullmatch(r"[a-zA-Z0-9_-]+", collection):
            raise ValueError("Invalid Qdrant collection name")
        self.url = url.rstrip("/")
        self.collection = f"/collections/{collection}"
        self.corpus = corpus
        self.timeout = timeout
        self.headers = {"Content-Type": "application/json"}
        if api_key:
            self.headers["api-key"] = api_key
        self.written = None
        self.hnsw_ef = 64
        self._request("PUT", self.collection, {
            "vectors": {"size": corpus.dimension, "distance": "Cosine", "on_disk": True},
            "on_disk_payload": True,
            "hnsw_config": {"on_disk": True, "m": hnsw_m, "ef_construct": ef_construct},
        })
        config = self.info()["config"]
        self.flush_interval = int(config["optimizer_config"]["flush_interval_sec"])
        if (
            config["params"]["vectors"].get("on_disk") is not True
            or config["hnsw_config"].get("on_disk") is not True
        ):
            raise RuntimeError("Qdrant did not enable on-disk vectors and HNSW")

    def _request(self, method, route, body=None, **params):
        path = f"{route}?{urlencode(params)}" if params else route
        data = json.dumps(body, separators=(",", ":")).encode() if body is not None else None
        request = Request(self.url + path, data=data, headers=self.headers, method=method)
        with urlopen(request, timeout=self.timeout) as response:
            value = json.load(response)
        if value.get("status") != "ok":
            raise RuntimeError(f"Qdrant returned an unsuccessful status for {method} {route}")
        return value["result"]

    def info(self):
        return self._request("GET", self.collection)

    def quiescent(self):
        info = self.info()
        status = info["optimizer_status"]
        if isinstance(status, dict) and "error" in status:
            raise RuntimeError(f"Qdrant optimizer failed: {status['error']}")
        return info["status"] == "green" and status == "ok"

    def count(self):
        return self._request("POST", f"{self.collection}/points/count", {"exact": True})["count"]

    def _put_vectors(self, ids, vectors):
        self._request(
            "PUT", f"{self.collection}/points",
            {"batch": {"ids": list(ids), "vectors": vectors.tolist()}}, wait="true", ordering="strong",
        )

    def populate(self):
        for start, vectors in self.corpus.batches():
            for offset in range(0, len(vectors), 512):
                block = vectors[offset:offset + 512]
                self._put_vectors(range(start + offset, start + offset + len(block)), block)

    def search(self, query, k=TOP_K):
        result = self._request("POST", f"{self.collection}/points/query", {
            "query": np.asarray(query, dtype=np.float32).tolist(),
            "limit": k, "with_payload": False, "with_vector": False, "params": {"hnsw_ef": self.hnsw_ef},
        })
        return [int(point["id"]) for point in result["points"]]

    def withdraw(self, ids):
        for start in range(0, len(ids), 512):
            self._request(
                "POST", f"{self.collection}/points/delete", {"points": ids[start:start + 512]},
                wait="true", ordering="strong",
            )

    def reinstate(self, ids):
        for start in range(0, len(ids), 512):
            batch = ids[start:start + 512]
            self._put_vectors(batch, self.corpus.get(batch))

    def visible(self, ids):
        found = set()
        for start in range(0, len(ids), 512):
            values = self._request("POST", f"{self.collection}/points", {
                "ids": ids[start:start + 512], "with_vector": False, "with_payload": False,
            })
            found.update(int(point["id"]) for point in values)
        return [doc_id in found for doc_id in ids]


class BlockWriteProbe:
    """Capture total device writes, including asynchronous optimizer and kernel I/O."""

    def __init__(self, stat_path, *, quiet_samples=3, interval=1, timeout=600):
        if min(quiet_samples, interval, timeout) <= 0:
            raise ValueError("Invalid quiet-window parameters")
        self.path = stat_path
        self.quiet_samples = quiet_samples
        self.interval = interval
        self.timeout = timeout

    def snapshot(self):
        # Linux diskstats sector counters use 512-byte units independent of physical sector size.
        return parse_sys_block_stat(self.path.read_text())

    def wait(self, ready):
        deadline = time.monotonic() + self.timeout
        previous = self.snapshot()
        quiet = 0
        while time.monotonic() < deadline:
            ok = ready()
            current = self.snapshot()
            unchanged = current.write_sectors == previous.write_sectors
            quiet = quiet + 1 if ok and unchanged and current.in_flight == 0 else 0
            if quiet >= self.quiet_samples:
                return current
            previous = current
            time.sleep(self.interval)
        raise TimeoutError("Optimizer and block-device writes did not reach quiescence")

    def bytes_since(self, before, after):
        return diff_block_stat(before, after).write_bytes


def withdrawal_pairs(geometry, count, *, seed=58):
    """Generate 30 equal-cardinality clustered/scattered pairs by physical slot."""
    locations = geometry.locations
    if not 0 < count <= len(locations) - TOP_K:
        raise ValueError("Trace size must leave at least ten live documents")
    ordered = np.argsort(locations)
    rng = np.random.default_rng(seed)
    for pair in range(TRACE_PAIRS):
        beginning = int(rng.integers(len(locations) - count + 1))
        clustered = ordered[beginning:beginning + count].tolist()
        scattered = rng.choice(len(locations), size=count, replace=False).tolist()
        yield pair, "clustered", sorted(clustered)
        yield pair, "scattered", sorted(scattered)


def score_queries(engine, queries, candidates, denied=()):
    denied = set(denied)
    hits = total = stale = returned = 0
    for query, reference in zip(queries, candidates):
        expected = {int(i) for i in [i for i in reference if i not in denied][:TOP_K]}
        answers = engine.search(query, TOP_K)
        if len(answers) > TOP_K or len(answers) != len(set(answers)):
            raise RuntimeError("An engine returned an invalid top-10 result")
        hits += len(set(answers) & expected)
        total += len(expected)
        stale += len(set(answers) & denied)
        returned += len(answers)
    return {"hits": hits, "reference_total": total, "stale": stale, "returned": returned}


def run_trace(engine, queries, candidates, ids, probe=None):
    def snapshot():
        if probe:
            return probe.wait(engine.quiescent)
        if not engine.quiescent():
            raise RuntimeError("The local file-backed engine did not reach quiescence")
        return engine.written

    def writes(before, after):
        return probe.bytes_since(before, after) if probe else after - before

    journal_before = engine.journal_stats() if hasattr(engine, "journal_stats") else (0, 0)
    before = snapshot()
    engine.withdraw(ids)
    immediate = score_queries(engine, queries, candidates, ids)
    immediate_hidden = not any(engine.visible(ids))
    withdrawn = snapshot()
    after_quiet = score_queries(engine, queries, candidates, ids)
    quiet_hidden = not any(engine.visible(ids))
    # Capture query-induced background I/O as part of the trace as well.
    withdrawn = snapshot()
    audit_withdraw = engine.audit_digest() if isinstance(engine, ColdOSD) else None

    engine.reinstate(ids)
    restored_immediate = score_queries(engine, queries, candidates)
    immediate_restored = all(engine.visible(ids))
    snapshot()
    restored_quiet = score_queries(engine, queries, candidates)
    quiet_restored = all(engine.visible(ids))
    restored = snapshot()
    audit_restore = engine.audit_digest() if isinstance(engine, ColdOSD) else None
    journal_after = engine.journal_stats() if hasattr(engine, "journal_stats") else (0, 0)
    return {
        "journal_records": journal_after[0] - journal_before[0],
        "journal_sector_bytes": journal_after[1] - journal_before[1],
        "withdraw_write_bytes": writes(before, withdrawn),
        "reinstate_write_bytes": writes(withdrawn, restored),
        "total_write_bytes": writes(before, restored),
        "withdraw_immediate": immediate,
        "withdraw_quiet": after_quiet,
        "reinstate_immediate": restored_immediate,
        "reinstate_quiet": restored_quiet,
        "all_withdrawn_ids_hidden": immediate_hidden and quiet_hidden,
        "all_reinstated_ids_visible": immediate_restored and quiet_restored,
        "ek5_withdraw_sha256": audit_withdraw,
        "ek5_reinstate_sha256": audit_restore,
    }


def summarize_traces(rows):
    checkpoints = ("withdraw_immediate", "withdraw_quiet", "reinstate_immediate", "reinstate_quiet")
    report = {}
    for layout in ("clustered", "scattered"):
        group = [row for row in rows if row["layout"] == layout]
        if len(group) != TRACE_PAIRS or {row["pair"] for row in group} != set(range(TRACE_PAIRS)):
            raise ValueError(f"Expected {TRACE_PAIRS} distinct traces for {layout}")
        stats = {}
        for name in checkpoints:
            count = sum(row[name]["reference_total"] for row in group)
            stats[name] = {
                "recall_at_10": sum(row[name]["hits"] for row in group) / count,
                "stale_resurrections": sum(row[name]["stale"] for row in group),
                "stale_rate": sum(row[name]["stale"] for row in group) / max(1, sum(row[name]["returned"] for row in group)),
                "returned": sum(row[name]["returned"] for row in group),
            }
        by_pair = sorted(group, key=lambda row: row["pair"])
        report[layout] = {
            "total_write_bytes": sum(row["total_write_bytes"] for row in group),
            "per_pair_total_write_bytes": [row["total_write_bytes"] for row in by_pair],
            "withdraw_write_bytes": sum(row["withdraw_write_bytes"] for row in group),
            "reinstate_write_bytes": sum(row["reinstate_write_bytes"] for row in group),
            "journal_records": sum(row.get("journal_records", 0) for row in group),
            "journal_sector_bytes": sum(row.get("journal_sector_bytes", 0) for row in group),
            "visibility_passed": all(row["all_withdrawn_ids_hidden"] and row["all_reinstated_ids_visible"] for row in group),
            "checkpoints": stats,
        }
    return report


def paired_bootstrap_lcb95(differences, *, resamples=BOOTSTRAP_RESAMPLES, seed=BOOTSTRAP_SEED):
    """One-sided 95% percentile-bootstrap lower bound on the summed paired difference.

    Pairs are resampled with replacement (the pair is the unit), the resampled mean is scaled to
    the number of pairs, and the 5th percentile is returned, so the bound is on the same total-byte
    scale as ``write_bytes_saved``.  The generator is seeded, so the bound is deterministic.
    """
    values = np.asarray(differences, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.all(np.isfinite(values)):
        raise ValueError("A paired bootstrap needs at least two finite paired differences")
    if resamples < 1000:
        raise ValueError("Use at least 1000 bootstrap resamples")
    rng = np.random.default_rng(seed)
    draws = rng.integers(len(values), size=(resamples, len(values)))
    totals = values[draws].mean(axis=1) * len(values)
    return float(np.quantile(totals, 0.05))


def compare_traces(reports, *, bootstrap_resamples=BOOTSTRAP_RESAMPLES, bootstrap_seed=BOOTSTRAP_SEED):
    candidate = reports["cold_osd"]
    comparisons = {}
    for name, challenger in reports.items():
        if name == "cold_osd":
            continue
        layouts = {}
        for layout, cells in candidate.items():
            other = challenger[layout]
            gaps = {stage: cells["checkpoints"][stage]["recall_at_10"] - baseline["recall_at_10"]
                    for stage, baseline in other["checkpoints"].items()}
            zero_stale = all(
                checkpoint["stale_resurrections"] == 0
                for arm in (cells, other) for checkpoint in arm["checkpoints"].values()
            )
            saved = other["total_write_bytes"] - cells["total_write_bytes"]
            mine, theirs = cells["per_pair_total_write_bytes"], other["per_pair_total_write_bytes"]
            if len(mine) != TRACE_PAIRS or len(theirs) != TRACE_PAIRS:
                raise ValueError(f"Expected {TRACE_PAIRS} paired traces per arm for {layout}")
            paired = [their - own for own, their in zip(mine, theirs)]
            lcb = paired_bootstrap_lcb95(paired, resamples=bootstrap_resamples, seed=bootstrap_seed)
            # Identical paired differences have zero resampling variance, so the "LCB" is just the
            # point estimate: say so instead of presenting an arithmetic comparison as a distribution.
            degenerate = len(set(paired)) == 1
            matched = max(map(abs, gaps.values())) <= 0.01 + 1e-12
            layouts[layout] = {
                "recall_at_10_gaps": gaps, "matched_recall": matched,
                "zero_stale_resurrection": zero_stale,
                "write_bytes_saved": saved,
                "paired_write_bytes_saved": paired,
                "saved_bytes_lcb95": lcb,
                "lcb_method": {
                    "statistic": "sum_of_paired_saved_bytes", "bootstrap": "percentile_one_sided_95",
                    "pairs": len(paired), "resamples": bootstrap_resamples, "seed": bootstrap_seed,
                    "zero_sampling_variance": degenerate,
                    "label": ("deterministic margin, no sampling variance (LCB95 equals the point estimate)"
                              if degenerate else "paired percentile bootstrap LCB95"),
                },
                "passed": saved > 0 and lcb > 0 and matched and zero_stale
                and cells["visibility_passed"] and other["visibility_passed"],
            }
        comparisons[name] = layouts
    return comparisons


GATING_CODEC_ROW = "shared_symmetric_codec"
SUPPLEMENTARY_CODEC_ROWS = {
    "as_built_asymmetric": "as_built_asymmetric",
    "no_durability_floor": "no_durability_floor",
}
DRYCHECK_TIER = "floor"
DRYCHECK_TIER_DISCLOSURE = (
    "Local-drycheck on a 240-document synthetic fixture with 30 deterministic 16-id withdrawal pairs; byte "
    "counts are logical fsynced write bytes, not 512-byte device sectors, and are not a live measurement. "
    "Cold-OSD and the SQ8 challenger share one journal record codec (journal_record); the gate is the "
    "clustered layout under that shared codec and under the challenger's no-durability floor."
)


def _layout_outcome(cell):
    if cell["passed"]:
        return "win"
    if cell["write_bytes_saved"] > 0:
        return "unproven"
    return "tie" if cell["write_bytes_saved"] == 0 else "loss"


def _codec_row(cells):
    return {
        layout: {
            "write_bytes_saved": cell["write_bytes_saved"], "saved_bytes_lcb95": cell["saved_bytes_lcb95"],
            "zero_sampling_variance": cell["lcb_method"]["zero_sampling_variance"],
            "matched_recall": cell["matched_recall"], "zero_stale_resurrection": cell["zero_stale_resurrection"],
            "passed": cell["passed"], "outcome": _layout_outcome(cell),
        } for layout, cell in cells.items()
    }


def drycheck(directory):
    axes = np.eye(8, dtype=np.float32)
    centroids = np.array([axes[0], -axes[0], axes[1], -axes[1]])
    angles = np.concatenate((0.32 + 0.028 * np.arange(28), np.linspace(1.12, 1.52, 32)))
    clusters = [np.cos(angles)[:, None] * center + np.sin(angles)[:, None] * direction
                for center, direction in zip(centroids, [axes[2], -axes[2], axes[3], -axes[3]])]
    corpus = VectorCorpus([np.vstack(clusters).astype(np.float32)])
    # Paired calibration isolates layout and withdrawal parity in the small fixture.
    dev = np.vstack((axes[2:], -axes[2:]))
    queries = centroids
    geometry = build_extents(corpus, centroids, dev, directory, 2 * MIB, csls_k=len(dev))
    exact = corpus.exact_top(queries, TOP_K + 16)
    digest = digest_file(geometry.path)
    wal_path = Path(directory) / "cold-osd-fence.wal"
    engines = {
        "cold_osd": ColdOSD(geometry, 4, wal_path=wal_path),
        "matched_ivf_sq8": MatchedIVFSQ8(geometry, 4, directory, journal_codec="shared"),
    }
    supplementary = {
        row: MatchedIVFSQ8(geometry, 4, directory, journal_codec=codec)
        for row, codec in SUPPLEMENTARY_CODEC_ROWS.items()
    }
    open_cost = {"cold_osd": engines["cold_osd"].written, "matched_ivf_sq8": engines["matched_ivf_sq8"].written,
                 **{row: engine.written for row, engine in supplementary.items()}}
    traces = list(withdrawal_pairs(geometry, 16))

    def run_all(engine):
        return [
            {"pair": pair, "layout": layout, "affected_extents": int(np.unique(geometry.locations[ids] // geometry.slots).size),
             **run_trace(engine, queries, exact, ids)}
            for pair, layout, ids in traces
        ]

    rows = {name: run_all(engine) for name, engine in engines.items()}
    summary = {name: summarize_traces(results) for name, results in rows.items()}
    comparisons = compare_traces(summary)
    supplementary_summary = {row: summarize_traces(run_all(engine)) for row, engine in supplementary.items()}
    supplementary_comparisons = {
        row: compare_traces({"cold_osd": summary["cold_osd"], "matched_ivf_sq8": result})["matched_ivf_sq8"]
        for row, result in supplementary_summary.items()
    }
    codec_rows = {GATING_CODEC_ROW: _codec_row(comparisons["matched_ivf_sq8"]),
                  **{row: _codec_row(cells) for row, cells in supplementary_comparisons.items()}}
    immutable = digest == digest_file(geometry.path)
    fence = engines["cold_osd"].fence
    recovered = EK5ExtentFence.recover_from_wal(geometry, wal_path)
    wal_replay = {
        **recovered.recovery,
        "bits_match": bool(np.array_equal(recovered.bits, fence.bits)),
        "epoch_matches": recovered.epoch == fence.epoch,
    }
    sq8_journal_replay = engines["matched_ivf_sq8"].verify_journal()
    sq8_journal_replay["records_match_counter"] = (
        sq8_journal_replay["records_replayed"] == engines["matched_ivf_sq8"].journal_records)
    passed = immutable and wal_replay["bits_match"] and wal_replay["epoch_matches"]
    passed = passed and sq8_journal_replay["records_match_counter"]
    summaries = {**summary, **{f"matched_ivf_sq8[{row}]": result for row, result in supplementary_summary.items()}}
    passed = passed and all(report["visibility_passed"] for arms in summaries.values() for report in arms.values())
    # The gate is the clustered layout against the shared-codec challenger and against the challenger's
    # cheapest (no-durability) journal.  Scattered is recorded transparently below, never gated away.
    clustered_gate = (codec_rows[GATING_CODEC_ROW]["clustered"]["passed"]
                      and codec_rows["no_durability_floor"]["clustered"]["passed"])
    passed = passed and clustered_gate
    scattered = {row: cells["scattered"] for row, cells in codec_rows.items()}
    if all(cells[layout]["passed"] for cells in codec_rows.values() for layout in cells):
        outcome = "verified-win"
    elif clustered_gate:
        outcome = "verified-mixed"
    else:
        outcome = "failed"
    sector_disclosure = {
        "unit": f"{SECTOR_BYTES}_byte_device_sectors",
        "note": (
            "Live mode accounts writes in 512-byte device sectors, so every fsynced journal record is "
            "charged at least one whole sector; the per-record byte differences that decide the drycheck "
            "scattered margin are quantised away and the live journal cost is decided by record count. "
            "Drycheck byte margins must not be extrapolated to live."
        ),
        "journal_rows": {
            row: {layout: {"journal_records": result[layout]["journal_records"],
                           "journal_sector_bytes": result[layout]["journal_sector_bytes"]}
                  for layout in ("clustered", "scattered")}
            for row, result in {"cold_osd": summary["cold_osd"], GATING_CODEC_ROW: summary["matched_ivf_sq8"],
                                **supplementary_summary}.items()
        },
    }
    return {
        "mode": "local-drycheck", "docs": corpus.count, "trace_pairs": TRACE_PAIRS,
        "tier": DRYCHECK_TIER, "tier_disclosure": DRYCHECK_TIER_DISCLOSURE,
        "bundle_outcome": outcome,
        "gate": {"layout": "clustered", "rows": [GATING_CODEC_ROW, "no_durability_floor"], "passed": clustered_gate},
        "scattered_layout_disclosure": {
            "gated": False, "per_codec_row": {
                row: {key: cell[key] for key in ("write_bytes_saved", "saved_bytes_lcb95", "outcome")}
                for row, cell in scattered.items()},
            "statement": (
                "Scattered layout is recorded, not gated: write_bytes_saved is "
                + ", ".join(f"{cell['write_bytes_saved']} B under {row}" for row, cell in scattered.items())
                + "; a negative value is a verified loss to the challenger and zero is a tie."
            ),
        },
        "codec_symmetry": {
            "journal_codec_module": "chelated_cold_osd_scanner.journal_record",
            "record_format": "kind:1 | epoch:u64 | count:u32 | ids:u32[count] | crc32; ids only on W; "
                             f"{len(FENCE_WAL_MAGIC)}-byte EK5FWAL1 header charged by the creating process on both arms",
            "gating_row": GATING_CODEC_ROW, "rows": codec_rows,
            "open_write_bytes": open_cost, "sq8_journal_replay": sq8_journal_replay,
        },
        "sector_quantization_disclosure": sector_disclosure,
        "extent_bytes": geometry.extent_bytes, "extent_count": len(geometry.counts),
        "fence_bytes": engines["cold_osd"].fence.bits.nbytes, "sidecar_bytes": geometry.sidecar.nbytes,
        "immutable_extent_sha256": digest, "extents_unchanged": immutable,
        "write_accounting": "logical_bytes_returned_by_file_writes_fsynced",
        "cold_osd_write_source": "fsynced_ek5_fence_wal", "fence_wal_bytes": fence.wal_bytes,
        "fence_wal_records": fence.wal_records, "fence_wal_replay": wal_replay, "traces": rows,
        "summaries": summaries, "comparisons": comparisons,
        "supplementary_comparisons": supplementary_comparisons, "passed": passed,
    }
