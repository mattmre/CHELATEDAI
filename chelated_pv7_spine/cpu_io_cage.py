"""Linux cgroup v2 Cage, I/O Counter Reconciliation, Storage-Tier Classifier, & Resident Idle-Lease Verifier.

Wave 16-Core (#47) shared hardware-discipline spine for Items 2A-2E (#56-#60):

1. Deterministic `systemd-run --user --scope -p MemoryMax=<bytes> -p MemorySwapMax=0`
   wrapper enforcing `MemoryMax < 1,073,741,824` strictly whenever `claim_sub_1gib=True`
   (catching the `1024M == 1 GiB` off-by-one bug).
2. `CUDA_VISIBLE_DEVICES=""` enforcement and zero-CUDA-context verification
   (`/proc/self/maps`, `nvidia-smi` process listing, environment checks).
3. Cross-platform `/proc/<pid>/io` (`read_bytes`, `write_bytes`, `syscr`, etc.)
   and `/sys/block/<dev>/stat` sector-delta parsers plus process-vs-block
   reconciliation.
4. Storage-tier classifier strictly distinguishing:
   - `nvme_native`
   - `kernel_dm_delay_calibrated_stripe` (`allows_cylinder_order_claim=False`)
   - `physical_rota1_hdd_stripe` (`allows_cylinder_order_claim=True`)
5. Resident vLLM/Nemotron Prometheus metrics request-delta guard that voids any
   timed lease if resident container request counts changed or in-flight gauges
   were non-zero during the timing window.
"""

from __future__ import annotations

import datetime
import hashlib
import json
import math
import os
import re
import sys
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


ONE_GIB_BYTES = 1_073_741_824  # 1024 ** 3
RECOMMENDED_SUB_1GIB_BYTES = 960 * 1024 * 1024  # 960 MiB = 1,006,632,960 B
DEFAULT_SECTOR_SIZE_BYTES = 512

STORAGE_TIER_NVME_NATIVE = "nvme_native"
STORAGE_TIER_DM_DELAY = "kernel_dm_delay_calibrated_stripe"
STORAGE_TIER_PHYSICAL_HDD = "physical_rota1_hdd_stripe"
VALID_STORAGE_TIERS: Tuple[str, ...] = (
    STORAGE_TIER_NVME_NATIVE,
    STORAGE_TIER_DM_DELAY,
    STORAGE_TIER_PHYSICAL_HDD,
)

DEFAULT_RESIDENT_COUNTER_NAMES: Tuple[str, ...] = (
    "vllm:request_success_total",
    "vllm:e2e_request_latency_seconds_count",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "nemotron_requests_total",
    "nemotron:request_success_total",
    "http_requests_total",
    "tgi_request_count",
    "sglang:num_requests_total",
)

DEFAULT_RESIDENT_ACTIVE_GAUGE_NAMES: Tuple[str, ...] = (
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:num_requests_swapped",
    "nemotron_requests_in_flight",
    "sglang:num_running_reqs",
    "sglang:num_queue_reqs",
)


class CgroupBudgetViolationError(ValueError):
    """Raised when a cgroup memory/swap configuration violates its claimed budget."""


class CudaIsolationError(RuntimeError):
    """Raised when a CPU-only caged workload leaks CUDA visibility or contexts."""


def _sha256_json(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# 1. cgroup v2 `systemd-run` Wrapper & Strict Sub-1 GiB Budget Guard
# ---------------------------------------------------------------------------

_SIZE_RE = re.compile(r"^\s*(\d+)\s*([a-zA-Z]*)\s*$")
_UNIT_MULTIPLIERS: Dict[str, int] = {
    "": 1,
    "b": 1,
    "k": 1024,
    "ki": 1024,
    "kib": 1024,
    "kb": 1000,
    "m": 1024 ** 2,
    "mi": 1024 ** 2,
    "mib": 1024 ** 2,
    "mb": 1000 ** 2,
    "g": 1024 ** 3,
    "gi": 1024 ** 3,
    "gib": 1024 ** 3,
    "gb": 1000 ** 3,
    "t": 1024 ** 4,
    "ti": 1024 ** 4,
    "tib": 1024 ** 4,
}


def parse_byte_size(spec: int | str) -> int:
    """Parse an exact integer byte count or systemd-style size token (`960M`, `8G`, `1024M`)."""
    if isinstance(spec, bool):
        raise ValueError("Byte size cannot be a boolean")
    if isinstance(spec, int):
        if spec < 0:
            raise ValueError("Byte size must be non-negative")
        return spec
    match = _SIZE_RE.match(str(spec))
    if not match:
        raise ValueError(f"Invalid byte size specification: {spec!r}")
    qty = int(match.group(1))
    unit = match.group(2).lower()
    if unit not in _UNIT_MULTIPLIERS:
        raise ValueError(f"Unsupported size suffix {unit!r} in {spec!r}")
    return qty * _UNIT_MULTIPLIERS[unit]


@dataclass(frozen=True)
class CgroupCageConfig:
    """Configuration for a deterministic Linux `cgroup v2` `systemd-run` cage."""

    memory_max_bytes: int
    memory_swap_max_bytes: int = 0
    claim_sub_1gib: bool = False
    allowed_cpus: Optional[str] = None
    io_max_rules: Tuple[str, ...] = ()
    unit_name: Optional[str] = None
    slice_name: Optional[str] = None
    description: Optional[str] = None
    require_swap_zero: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.memory_max_bytes, bool) or self.memory_max_bytes <= 0:
            raise CgroupBudgetViolationError("memory_max_bytes must be a positive integer")
        if isinstance(self.memory_swap_max_bytes, bool) or self.memory_swap_max_bytes < 0:
            raise CgroupBudgetViolationError("memory_swap_max_bytes must be >= 0")
        if self.require_swap_zero and self.memory_swap_max_bytes != 0:
            raise CgroupBudgetViolationError(
                f"MemorySwapMax must be 0 to prevent swap spillover; got {self.memory_swap_max_bytes}"
            )
        if self.claim_sub_1gib and self.memory_max_bytes >= ONE_GIB_BYTES:
            raise CgroupBudgetViolationError(
                f"Sub-1 GiB claim violated: MemoryMax={self.memory_max_bytes} B >= 1 GiB ({ONE_GIB_BYTES} B). "
                "Note: 1024M == 1,073,741,824 B is NOT strictly < 1 GiB; use e.g. 960M (1,006,632,960 B)."
            )

    @classmethod
    def from_size_spec(
        cls,
        memory_max: int | str,
        *,
        memory_swap_max: int | str = 0,
        claim_sub_1gib: bool = False,
        allowed_cpus: Optional[str] = None,
        io_max_rules: Sequence[str] = (),
        unit_name: Optional[str] = None,
        slice_name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> "CgroupCageConfig":
        return cls(
            memory_max_bytes=parse_byte_size(memory_max),
            memory_swap_max_bytes=parse_byte_size(memory_swap_max),
            claim_sub_1gib=claim_sub_1gib,
            allowed_cpus=allowed_cpus,
            io_max_rules=tuple(io_max_rules),
            unit_name=unit_name,
            slice_name=slice_name,
            description=description,
        )


def build_systemd_run_command(
    command: Sequence[str],
    config: CgroupCageConfig,
) -> List[str]:
    """Build deterministic `systemd-run --user --scope` wrapper enforcing cgroup v2 limits."""
    if not command:
        raise ValueError("command must be a non-empty argv sequence")
    argv: List[str] = ["systemd-run", "--user", "--scope"]
    if config.unit_name:
        argv.extend(["--unit", str(config.unit_name)])
    if config.slice_name:
        argv.extend(["--slice", str(config.slice_name)])
    if config.description:
        argv.extend(["--description", str(config.description)])

    argv.extend(["-p", f"MemoryMax={config.memory_max_bytes}"])
    argv.extend(["-p", f"MemorySwapMax={config.memory_swap_max_bytes}"])
    if config.allowed_cpus:
        argv.extend(["-p", f"AllowedCPUs={config.allowed_cpus}"])
    for rule in config.io_max_rules:
        argv.extend(["-p", str(rule)])
    argv.append("--")
    argv.extend(str(tok) for tok in command)
    return argv


# ---------------------------------------------------------------------------
# 2. CPU-Only Environment & Zero-CUDA-Context Assertion
# ---------------------------------------------------------------------------


def build_cpu_only_env(base_env: Optional[Mapping[str, str]] = None) -> Dict[str, str]:
    """Return an environment dictionary pinned strictly to zero GPU visibility."""
    env = dict(os.environ if base_env is None else base_env)
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["HIP_VISIBLE_DEVICES"] = ""
    env["ROCR_VISIBLE_DEVICES"] = ""
    env["NVIDIA_VISIBLE_DEVICES"] = "void"
    return env


@dataclass(frozen=True)
class CudaIsolationReceipt:
    passed: bool
    cuda_visible_devices: Optional[str]
    cuda_maps_detected: Tuple[str, ...]
    active_cuda_pids: Tuple[int, ...]
    violations: Tuple[str, ...]
    receipt_sha256: str


_CUDA_MAP_TOKENS: Tuple[str, ...] = (
    "libcuda.so",
    "libcudart.so",
    "libcublas.so",
    "libnvidia-ml.so",
    "/dev/nvidia",
    "nvcuda.dll",
)


def assert_zero_cuda_context(
    *,
    env: Optional[Mapping[str, str]] = None,
    target_pid: Optional[int] = None,
    proc_maps_text: Optional[str] = None,
    nvidia_smi_csv: Optional[str] = None,
    strict: bool = True,
) -> CudaIsolationReceipt:
    """Verify `CUDA_VISIBLE_DEVICES == ""` and zero active CUDA mappings/contexts."""
    env_map = os.environ if env is None else env
    pid = os.getpid() if target_pid is None else int(target_pid)
    cuda_vis = env_map.get("CUDA_VISIBLE_DEVICES")

    violations: List[str] = []
    if cuda_vis is None:
        violations.append("CUDA_VISIBLE_DEVICES_UNSET")
    elif cuda_vis != "":
        violations.append(f"CUDA_VISIBLE_DEVICES_NONEMPTY:{cuda_vis}")

    maps_text = proc_maps_text
    if maps_text is None:
        maps_path = Path(f"/proc/{pid}/maps")
        if maps_path.exists():
            try:
                maps_text = maps_path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                maps_text = ""
        else:
            maps_text = ""

    detected_maps: List[str] = []
    for line in maps_text.splitlines():
        lower = line.lower()
        if any(tok in lower for tok in _CUDA_MAP_TOKENS):
            detected_maps.append(line.strip())
    if detected_maps:
        violations.append(f"CUDA_LIBRARY_OR_DEVICE_MAPPED:{len(detected_maps)}")

    active_pids: List[int] = []
    if nvidia_smi_csv:
        for raw_line in nvidia_smi_csv.splitlines():
            stripped = raw_line.strip()
            if not stripped or stripped.lower().startswith("pid"):
                continue
            first_col = stripped.split(",")[0].strip()
            if first_col.isdigit():
                row_pid = int(first_col)
                active_pids.append(row_pid)
        if pid in active_pids:
            violations.append(f"TARGET_PID_HAS_ACTIVE_CUDA_CONTEXT:{pid}")

    passed = len(violations) == 0
    receipt = CudaIsolationReceipt(
        passed=passed,
        cuda_visible_devices=cuda_vis,
        cuda_maps_detected=tuple(detected_maps),
        active_cuda_pids=tuple(sorted(set(active_pids))),
        violations=tuple(violations),
        receipt_sha256=_sha256_json(
            {
                "passed": passed,
                "cuda_visible_devices": cuda_vis,
                "n_cuda_maps": len(detected_maps),
                "active_cuda_pids": sorted(set(active_pids)),
                "violations": violations,
            }
        ),
    )
    if strict and not passed:
        raise CudaIsolationError(f"Zero-CUDA isolation failed: {'; '.join(violations)}")
    return receipt


# ---------------------------------------------------------------------------
# 3. `/proc/<pid>/io` & `/sys/block/<dev>/stat` Parsers + Reconciliation
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ProcIoCounters:
    rchar: int
    wchar: int
    syscr: int
    syscw: int
    read_bytes: int
    write_bytes: int
    cancelled_write_bytes: int = 0


@dataclass(frozen=True)
class ProcIoDelta:
    rchar: int
    wchar: int
    syscr: int
    syscw: int
    read_bytes: int
    write_bytes: int
    cancelled_write_bytes: int
    net_write_bytes: int


def parse_proc_io(text: str) -> ProcIoCounters:
    """Parse Linux `/proc/<pid>/io` key-value lines into `ProcIoCounters`."""
    parsed: Dict[str, int] = {}
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or ":" not in stripped:
            continue
        key, raw_val = stripped.split(":", 1)
        key = key.strip()
        raw_val = raw_val.strip()
        if raw_val.lstrip("-").isdigit():
            parsed[key] = int(raw_val)
    required = ("rchar", "wchar", "syscr", "syscw", "read_bytes", "write_bytes")
    missing = [k for k in required if k not in parsed]
    if missing:
        raise ValueError(f"/proc/<pid>/io text missing required keys: {missing}")
    return ProcIoCounters(
        rchar=parsed["rchar"],
        wchar=parsed["wchar"],
        syscr=parsed["syscr"],
        syscw=parsed["syscw"],
        read_bytes=parsed["read_bytes"],
        write_bytes=parsed["write_bytes"],
        cancelled_write_bytes=parsed.get("cancelled_write_bytes", 0),
    )


def read_proc_io(pid: int) -> ProcIoCounters:
    """Read and parse `/proc/<pid>/io` on Linux hosts."""
    path = Path(f"/proc/{int(pid)}/io")
    return parse_proc_io(path.read_text(encoding="utf-8"))


def diff_proc_io(before: ProcIoCounters, after: ProcIoCounters) -> ProcIoDelta:
    """Compute monotonic counter deltas between two `/proc/<pid>/io` snapshots."""
    for field_name in ("rchar", "wchar", "syscr", "syscw", "read_bytes", "write_bytes", "cancelled_write_bytes"):
        if getattr(after, field_name) < getattr(before, field_name):
            raise ValueError(f"/proc/<pid>/io counter regressed for {field_name}")
    wb = after.write_bytes - before.write_bytes
    cwb = after.cancelled_write_bytes - before.cancelled_write_bytes
    return ProcIoDelta(
        rchar=after.rchar - before.rchar,
        wchar=after.wchar - before.wchar,
        syscr=after.syscr - before.syscr,
        syscw=after.syscw - before.syscw,
        read_bytes=after.read_bytes - before.read_bytes,
        write_bytes=wb,
        cancelled_write_bytes=cwb,
        net_write_bytes=max(0, wb - cwb),
    )


@dataclass(frozen=True)
class BlockStatCounters:
    read_ios: int
    read_merges: int
    read_sectors: int
    read_ticks_ms: int
    write_ios: int
    write_merges: int
    write_sectors: int
    write_ticks_ms: int
    in_flight: int
    io_ticks_ms: int
    time_in_queue_ms: int
    discard_ios: int = 0
    discard_merges: int = 0
    discard_sectors: int = 0
    discard_ticks_ms: int = 0
    flush_ios: int = 0
    flush_ticks_ms: int = 0
    sector_size_bytes: int = DEFAULT_SECTOR_SIZE_BYTES


@dataclass(frozen=True)
class BlockStatDelta:
    read_ios: int
    write_ios: int
    read_sectors: int
    write_sectors: int
    read_bytes: int
    write_bytes: int
    read_ticks_ms: int
    write_ticks_ms: int
    io_ticks_ms: int
    sector_size_bytes: int


def parse_sys_block_stat(text: str, *, sector_size_bytes: int = DEFAULT_SECTOR_SIZE_BYTES) -> BlockStatCounters:
    """Parse Linux `/sys/block/<dev>/stat` (11, 15, or 17 whitespace-separated integer fields)."""
    tokens = text.strip().split()
    if len(tokens) < 11:
        raise ValueError(f"/sys/block/<dev>/stat requires at least 11 integer columns; got {len(tokens)}")
    vals = [int(t) for t in tokens]
    return BlockStatCounters(
        read_ios=vals[0],
        read_merges=vals[1],
        read_sectors=vals[2],
        read_ticks_ms=vals[3],
        write_ios=vals[4],
        write_merges=vals[5],
        write_sectors=vals[6],
        write_ticks_ms=vals[7],
        in_flight=vals[8],
        io_ticks_ms=vals[9],
        time_in_queue_ms=vals[10],
        discard_ios=vals[11] if len(vals) >= 15 else 0,
        discard_merges=vals[12] if len(vals) >= 15 else 0,
        discard_sectors=vals[13] if len(vals) >= 15 else 0,
        discard_ticks_ms=vals[14] if len(vals) >= 15 else 0,
        flush_ios=vals[15] if len(vals) >= 17 else 0,
        flush_ticks_ms=vals[16] if len(vals) >= 17 else 0,
        sector_size_bytes=int(sector_size_bytes),
    )


def diff_block_stat(before: BlockStatCounters, after: BlockStatCounters) -> BlockStatDelta:
    """Compute physical block-device sector and byte deltas from `/sys/block/<dev>/stat`."""
    if before.sector_size_bytes != after.sector_size_bytes:
        raise ValueError("Mismatched sector_size_bytes between before and after block stat snapshots")
    for field_name in ("read_ios", "write_ios", "read_sectors", "write_sectors", "read_ticks_ms", "write_ticks_ms"):
        if getattr(after, field_name) < getattr(before, field_name):
            raise ValueError(f"/sys/block/<dev>/stat counter regressed for {field_name}")
    r_sec = after.read_sectors - before.read_sectors
    w_sec = after.write_sectors - before.write_sectors
    sec_sz = before.sector_size_bytes
    return BlockStatDelta(
        read_ios=after.read_ios - before.read_ios,
        write_ios=after.write_ios - before.write_ios,
        read_sectors=r_sec,
        write_sectors=w_sec,
        read_bytes=r_sec * sec_sz,
        write_bytes=w_sec * sec_sz,
        read_ticks_ms=after.read_ticks_ms - before.read_ticks_ms,
        write_ticks_ms=after.write_ticks_ms - before.write_ticks_ms,
        io_ticks_ms=max(0, after.io_ticks_ms - before.io_ticks_ms),
        sector_size_bytes=sec_sz,
    )


@dataclass(frozen=True)
class IoReconciliationReport:
    proc_read_bytes: int
    proc_net_write_bytes: int
    proc_syscr: int
    block_read_bytes: int
    block_write_bytes: int
    read_block_to_proc_ratio: Optional[float]
    write_amplification_ratio: Optional[float]
    external_background_io_suspected: bool
    receipt_sha256: str


def reconcile_proc_and_block_io(
    proc_delta: ProcIoDelta,
    block_delta: BlockStatDelta,
    *,
    background_io_ratio_threshold: float = 2.0,
    min_background_bytes: int = 16 * 1024 * 1024,
) -> IoReconciliationReport:
    """Reconcile `/proc/<pid>/io` deltas against physical `/sys/block/<dev>/stat` deltas."""
    r_ratio = (
        float(block_delta.read_bytes) / float(proc_delta.read_bytes)
        if proc_delta.read_bytes > 0
        else None
    )
    w_ratio = (
        float(block_delta.write_bytes) / float(proc_delta.net_write_bytes)
        if proc_delta.net_write_bytes > 0
        else None
    )
    ext_read = (
        block_delta.read_bytes >= min_background_bytes
        and (r_ratio is None or r_ratio > background_io_ratio_threshold)
    )
    ext_write = (
        block_delta.write_bytes >= min_background_bytes
        and (w_ratio is None or w_ratio > background_io_ratio_threshold)
    )
    suspected = bool(ext_read or ext_write)
    payload = {
        "proc_read_bytes": proc_delta.read_bytes,
        "proc_net_write_bytes": proc_delta.net_write_bytes,
        "proc_syscr": proc_delta.syscr,
        "block_read_bytes": block_delta.read_bytes,
        "block_write_bytes": block_delta.write_bytes,
        "external_background_io_suspected": suspected,
    }
    return IoReconciliationReport(
        proc_read_bytes=proc_delta.read_bytes,
        proc_net_write_bytes=proc_delta.net_write_bytes,
        proc_syscr=proc_delta.syscr,
        block_read_bytes=block_delta.read_bytes,
        block_write_bytes=block_delta.write_bytes,
        read_block_to_proc_ratio=r_ratio,
        write_amplification_ratio=w_ratio,
        external_background_io_suspected=suspected,
        receipt_sha256=_sha256_json(payload),
    )


# ---------------------------------------------------------------------------
# 4. Storage-Tier Classifier (`nvme_native`, `dm-delay`, `physical_rota1_hdd`)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class StorageTierClassification:
    device_name: str
    tier: str
    rota: int
    emulated: bool
    allows_cylinder_order_claim: bool
    dm_delay_ms: Optional[int]
    evidence: str


_DM_DELAY_RE = re.compile(r"\bdelay\s+\S+\s+\d+\s+(\d+)", re.IGNORECASE)


def classify_storage_tier(
    device_name: str,
    *,
    rota: int,
    dm_table: str = "",
    transport: str = "",
    model: str = "",
) -> StorageTierClassification:
    """Classify storage medium into the 3 locked Wave 19 tiers.

    Enforces the R2 consensus rule that mechanical cylinder-order claims are
    strictly restricted to physical `ROTA=1` drives (`allows_cylinder_order_claim=True`),
    whereas `dm-delay` stripes are explicitly labeled emulated.
    """
    dev_lower = str(device_name).strip().lower()
    dm_lower = str(dm_table).strip().lower()
    tran_lower = str(transport).strip().lower()
    rota_int = int(rota)
    if rota_int not in (0, 1):
        raise ValueError(f"rota must be 0 or 1; got {rota!r}")

    delay_match = _DM_DELAY_RE.search(dm_table)
    if delay_match or "dm-delay" in dm_lower or " delay " in f" {dm_lower} ":
        delay_ms = int(delay_match.group(1)) if delay_match else None
        return StorageTierClassification(
            device_name=str(device_name),
            tier=STORAGE_TIER_DM_DELAY,
            rota=rota_int,
            emulated=True,
            allows_cylinder_order_claim=False,
            dm_delay_ms=delay_ms,
            evidence=f"dm-delay target detected in dm_table={dm_table!r}",
        )

    if rota_int == 1:
        return StorageTierClassification(
            device_name=str(device_name),
            tier=STORAGE_TIER_PHYSICAL_HDD,
            rota=1,
            emulated=False,
            allows_cylinder_order_claim=True,
            dm_delay_ms=None,
            evidence=f"physical ROTA=1 rotational block device (transport={transport!r}, model={model!r})",
        )

    if "nvme" in dev_lower or "nvme" in tran_lower or rota_int == 0:
        return StorageTierClassification(
            device_name=str(device_name),
            tier=STORAGE_TIER_NVME_NATIVE,
            rota=0,
            emulated=False,
            allows_cylinder_order_claim=False,
            dm_delay_ms=None,
            evidence=f"non-rotational ROTA=0 device (transport={transport!r}, model={model!r})",
        )

    raise ValueError(f"Unable to classify storage tier for device {device_name!r}")


def parse_lsblk_json(lsblk_json_text: str) -> List[Dict[str, Any]]:
    """Parse `lsblk -J -o NAME,ROTA,TRAN,MODEL,TYPE` output into normalized device records."""
    payload = json.loads(lsblk_json_text)
    devices = payload.get("blockdevices", [])
    out: List[Dict[str, Any]] = []
    for dev in devices:
        raw_rota = dev.get("rota", 0)
        rota_int = 1 if str(raw_rota).lower() in {"1", "true"} else 0
        out.append(
            {
                "name": str(dev.get("name", "")),
                "rota": rota_int,
                "transport": str(dev.get("tran") or ""),
                "model": str(dev.get("model") or "").strip(),
                "type": str(dev.get("type") or ""),
            }
        )
    return out


# ---------------------------------------------------------------------------
# 5. Resident vLLM / Nemotron Prometheus Request-Delta Guard (Idle-Lease Verifier)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PrometheusSample:
    name: str
    labels: Tuple[Tuple[str, str], ...]
    value: float


_PROM_LINE_RE = re.compile(
    r"^(?P<name>[a-zA-Z_:][a-zA-Z0-9_:]*)"
    r"(?:\{(?P<labels>[^}]*)\})?\s+"
    r"(?P<value>[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[+-]?Inf|NaN)"
    r"(?:\s+\d+)?\s*$"
)
_LABEL_RE = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:\\.|[^"\\])*)"')


def parse_prometheus_metrics(text: str) -> List[PrometheusSample]:
    """Parse Prometheus text exposition format into deterministic `PrometheusSample` entries."""
    samples: List[PrometheusSample] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _PROM_LINE_RE.match(line)
        if not match:
            continue
        name = match.group("name")
        labels_raw = match.group("labels") or ""
        labels = tuple(sorted((k, v) for k, v in _LABEL_RE.findall(labels_raw)))
        try:
            val = float(match.group("value"))
        except ValueError:
            continue
        samples.append(PrometheusSample(name=name, labels=labels, value=val))
    return samples


@dataclass(frozen=True)
class ResidentMetricsSnapshot:
    source: str
    captured_at_utc: str
    counter_totals: Dict[str, float]
    active_gauges: Dict[str, float]
    raw_sample_count: int
    snapshot_sha256: str


@dataclass(frozen=True)
class IdleLeaseVerdict:
    valid: bool
    lease_status: str
    counter_deltas: Dict[str, float]
    max_active_gauge_before: float
    max_active_gauge_after: float
    violations: Tuple[str, ...]
    before_sha256: str
    after_sha256: str
    receipt_sha256: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ResidentIdleLeaseVerifier:
    """Verifies resident vLLM/Nemotron containers processed zero requests during a timed lease."""

    def __init__(
        self,
        *,
        counter_names: Sequence[str] = DEFAULT_RESIDENT_COUNTER_NAMES,
        active_gauge_names: Sequence[str] = DEFAULT_RESIDENT_ACTIVE_GAUGE_NAMES,
    ) -> None:
        self.counter_names = tuple(counter_names)
        self.active_gauge_names = tuple(active_gauge_names)

    def snapshot_from_text(
        self,
        metrics_text: str,
        *,
        source: str = "prometheus_text",
        captured_at_utc: Optional[str] = None,
    ) -> ResidentMetricsSnapshot:
        samples = parse_prometheus_metrics(metrics_text)
        counters: Dict[str, float] = {}
        gauges: Dict[str, float] = {}
        counter_set = set(self.counter_names)
        gauge_set = set(self.active_gauge_names)
        for sample in samples:
            if sample.name in counter_set:
                counters[sample.name] = counters.get(sample.name, 0.0) + sample.value
            if sample.name in gauge_set:
                gauges[sample.name] = gauges.get(sample.name, 0.0) + sample.value

        ts = captured_at_utc or datetime.datetime.now(datetime.timezone.utc).isoformat()
        snap_sha = _sha256_json(
            {
                "source": source,
                "counter_totals": counters,
                "active_gauges": gauges,
                "raw_sample_count": len(samples),
            }
        )
        return ResidentMetricsSnapshot(
            source=source,
            captured_at_utc=ts,
            counter_totals=counters,
            active_gauges=gauges,
            raw_sample_count=len(samples),
            snapshot_sha256=snap_sha,
        )

    def snapshot_from_url(self, metrics_url: str, *, timeout_s: float = 3.0) -> ResidentMetricsSnapshot:
        with urllib.request.urlopen(metrics_url, timeout=float(timeout_s)) as resp:
            body = resp.read().decode("utf-8", errors="replace")
        return self.snapshot_from_text(body, source=metrics_url)

    def verify_lease(
        self,
        before: ResidentMetricsSnapshot,
        after: ResidentMetricsSnapshot,
        *,
        require_at_least_one_metric: bool = True,
    ) -> IdleLeaseVerdict:
        """Void the timed lease if ANY request counter changed, reset, or active gauge was > 0."""
        violations: List[str] = []
        all_counters = sorted(set(before.counter_totals.keys()) | set(after.counter_totals.keys()))
        all_gauges = sorted(set(before.active_gauges.keys()) | set(after.active_gauges.keys()))

        if require_at_least_one_metric and not all_counters and not all_gauges:
            violations.append("NO_RESIDENT_METRICS_OBSERVED")

        deltas: Dict[str, float] = {}
        for name in all_counters:
            if name not in before.counter_totals or name not in after.counter_totals:
                violations.append(f"COUNTER_KEY_MISMATCH:{name}")
                continue
            delta = float(after.counter_totals[name] - before.counter_totals[name])
            deltas[name] = delta
            if not math.isfinite(delta):
                violations.append(f"NONFINITE_COUNTER_DELTA:{name}={delta}")
                continue
            if delta < 0.0:
                violations.append(f"COUNTER_RESET_DETECTED:{name}={delta}")
            elif delta > 0.0:
                violations.append(f"RESIDENT_REQUEST_DELTA_NONZERO:{name}=+{delta}")

        finite_before = [float(v) for v in before.active_gauges.values() if math.isfinite(float(v))]
        finite_after = [float(v) for v in after.active_gauges.values() if math.isfinite(float(v))]
        max_before = max(finite_before, default=0.0)
        max_after = max(finite_after, default=0.0)
        for name in all_gauges:
            vb = float(before.active_gauges.get(name, 0.0))
            va = float(after.active_gauges.get(name, 0.0))
            if not math.isfinite(vb) or not math.isfinite(va):
                violations.append(f"NONFINITE_ACTIVE_GAUGE:{name}")
                continue
            if vb > 0.0:
                violations.append(f"ACTIVE_REQUESTS_AT_LEASE_START:{name}={vb}")
            if va > 0.0:
                violations.append(f"ACTIVE_REQUESTS_AT_LEASE_END:{name}={va}")

        valid = len(violations) == 0
        lease_status = "CLEAN_IDLE_LEASE" if valid else "VOIDED_RESIDENT_ACTIVITY"
        receipt_sha = _sha256_json(
            {
                "valid": valid,
                "lease_status": lease_status,
                "counter_deltas": deltas,
                "violations": violations,
                "before_sha256": before.snapshot_sha256,
                "after_sha256": after.snapshot_sha256,
            }
        )
        return IdleLeaseVerdict(
            valid=valid,
            lease_status=lease_status,
            counter_deltas=deltas,
            max_active_gauge_before=max_before,
            max_active_gauge_after=max_after,
            violations=tuple(violations),
            before_sha256=before.snapshot_sha256,
            after_sha256=after.snapshot_sha256,
            receipt_sha256=receipt_sha,
        )


def build_cage_preflight_receipt(
    *,
    config: CgroupCageConfig,
    command: Sequence[str],
    storage: StorageTierClassification,
    cuda_receipt: CudaIsolationReceipt,
    idle_lease: IdleLeaseVerdict,
) -> Dict[str, Any]:
    """Assemble a content-addressed preflight receipt for Items 2A-2E."""
    wrapped_cmd = build_systemd_run_command(command, config)
    passed = bool(cuda_receipt.passed and idle_lease.valid)
    skipped_non_linux_checks: List[str] = []
    if sys.platform != "linux":
        skipped_non_linux_checks = [
            "proc_pid_io",
            "sys_block_stat",
            "cgroup_v2_systemd_scope",
        ]
    payload = {
        "preflight_passed": passed,
        "systemd_command": wrapped_cmd,
        "memory_max_bytes": config.memory_max_bytes,
        "memory_swap_max_bytes": config.memory_swap_max_bytes,
        "claim_sub_1gib": config.claim_sub_1gib,
        "storage_tier": asdict(storage),
        "cuda_isolation": asdict(cuda_receipt),
        "idle_lease": idle_lease.to_dict(),
        "platform": sys.platform,
        "skipped_non_linux_checks": skipped_non_linux_checks,
    }
    payload["receipt_sha256"] = _sha256_json(payload)
    return payload


__all__ = [
    "BlockStatCounters",
    "BlockStatDelta",
    "CgroupBudgetViolationError",
    "CgroupCageConfig",
    "CudaIsolationError",
    "CudaIsolationReceipt",
    "DEFAULT_RESIDENT_ACTIVE_GAUGE_NAMES",
    "DEFAULT_RESIDENT_COUNTER_NAMES",
    "DEFAULT_SECTOR_SIZE_BYTES",
    "IdleLeaseVerdict",
    "IoReconciliationReport",
    "ONE_GIB_BYTES",
    "ProcIoCounters",
    "ProcIoDelta",
    "PrometheusSample",
    "RECOMMENDED_SUB_1GIB_BYTES",
    "ResidentIdleLeaseVerifier",
    "ResidentMetricsSnapshot",
    "STORAGE_TIER_DM_DELAY",
    "STORAGE_TIER_NVME_NATIVE",
    "STORAGE_TIER_PHYSICAL_HDD",
    "StorageTierClassification",
    "VALID_STORAGE_TIERS",
    "assert_zero_cuda_context",
    "build_cage_preflight_receipt",
    "build_cpu_only_env",
    "build_systemd_run_command",
    "classify_storage_tier",
    "diff_block_stat",
    "diff_proc_io",
    "parse_byte_size",
    "parse_lsblk_json",
    "parse_proc_io",
    "parse_prometheus_metrics",
    "parse_sys_block_stat",
    "read_proc_io",
    "reconcile_proc_and_block_io",
]
