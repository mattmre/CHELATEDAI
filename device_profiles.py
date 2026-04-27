"""Device-class profiles for computational-storage simulations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict


class DeviceClass(str, Enum):
    """Supported storage/near-data simulation device classes."""

    RP2040 = "rp2040"
    CONSUMER_NVME = "consumer_nvme"
    SMARTSSD = "smartssd"
    DPU_STORAGE = "dpu_storage"
    CUSTOM = "custom"


@dataclass(frozen=True)
class StorageDeviceProfile:
    """Latency/throughput profile for storage-backed population scoring."""

    name: str
    device_class: DeviceClass
    block_fetch_latency_us: float
    branch_selection_latency_us: float
    max_queue_depth: int
    compute_gflops: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.block_fetch_latency_us <= 0:
            raise ValueError("block_fetch_latency_us must be positive")
        if self.branch_selection_latency_us < 0:
            raise ValueError("branch_selection_latency_us must be non-negative")
        if self.max_queue_depth < 1:
            raise ValueError("max_queue_depth must be >= 1")
        if self.compute_gflops < 0:
            raise ValueError("compute_gflops must be non-negative")

    def latency_seconds(self, queue_depth: int) -> float:
        if queue_depth < 0:
            raise ValueError("queue_depth must be non-negative")
        effective_depth = min(max(queue_depth, 0), self.max_queue_depth)
        overflow_depth = max(0, queue_depth - self.max_queue_depth)
        overflow_penalty = overflow_depth * self.block_fetch_latency_us * 2.0
        latency_us = self.branch_selection_latency_us + effective_depth * self.block_fetch_latency_us + overflow_penalty
        return latency_us / 1_000_000.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "device_class": self.device_class.value,
            "block_fetch_latency_us": self.block_fetch_latency_us,
            "branch_selection_latency_us": self.branch_selection_latency_us,
            "max_queue_depth": self.max_queue_depth,
            "compute_gflops": self.compute_gflops,
            "metadata": dict(self.metadata),
        }


DEVICE_PROFILES = {
    DeviceClass.RP2040: StorageDeviceProfile(
        name="RP2040 transport proof",
        device_class=DeviceClass.RP2040,
        block_fetch_latency_us=500.0,
        branch_selection_latency_us=10.0,
        max_queue_depth=2,
        compute_gflops=0.01,
        metadata={"scope": "transport proof, not AI compute evidence"},
    ),
    DeviceClass.CONSUMER_NVME: StorageDeviceProfile(
        name="Consumer NVMe simulation",
        device_class=DeviceClass.CONSUMER_NVME,
        block_fetch_latency_us=100.0,
        branch_selection_latency_us=1.0,
        max_queue_depth=16,
        compute_gflops=1.0,
    ),
    DeviceClass.SMARTSSD: StorageDeviceProfile(
        name="FPGA SmartSSD simulation",
        device_class=DeviceClass.SMARTSSD,
        block_fetch_latency_us=50.0,
        branch_selection_latency_us=2.0,
        max_queue_depth=64,
        compute_gflops=60.0,
        metadata={"scope": "near-data scoring simulation"},
    ),
    DeviceClass.DPU_STORAGE: StorageDeviceProfile(
        name="DPU-backed storage simulation",
        device_class=DeviceClass.DPU_STORAGE,
        block_fetch_latency_us=25.0,
        branch_selection_latency_us=2.0,
        max_queue_depth=128,
        compute_gflops=200.0,
        metadata={"scope": "near-data scoring simulation"},
    ),
}


def get_profile(device_class: DeviceClass | str) -> StorageDeviceProfile:
    """Retrieve a preset profile by enum or string name."""

    normalized = device_class if isinstance(device_class, DeviceClass) else DeviceClass(str(device_class))
    return DEVICE_PROFILES[normalized]


def create_custom_profile(
    name: str,
    block_fetch_latency_us: float,
    branch_selection_latency_us: float = 1.0,
    max_queue_depth: int = 16,
    compute_gflops: float = 0.0,
    metadata: Dict[str, Any] | None = None,
) -> StorageDeviceProfile:
    """Create a custom simulation profile."""

    return StorageDeviceProfile(
        name=name,
        device_class=DeviceClass.CUSTOM,
        block_fetch_latency_us=block_fetch_latency_us,
        branch_selection_latency_us=branch_selection_latency_us,
        max_queue_depth=max_queue_depth,
        compute_gflops=compute_gflops,
        metadata=metadata or {},
    )
