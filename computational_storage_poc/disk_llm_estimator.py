from __future__ import annotations

import argparse
from dataclasses import dataclass


DECIMAL_GB = 1_000_000_000
DEFAULT_RESIDENT_FRACTION = 1.0 / 3.0
DEFAULT_STREAMED_FRACTION = 2.0 / 3.0
DEFAULT_ACTIVE_FFN_FRACTION = 0.02
DEFAULT_IO_EFFICIENCY = 0.7
DEFAULT_WORKING_MEMORY_MULTIPLIER = 1.5
DEFAULT_FIXED_RAM_OVERHEAD_GB = 4.0
DEFAULT_VALIDATED_MODEL_TO_DRAM_RATIO = 2.0


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    ssd_bandwidth_gbps: float
    dram_gb: float
    cpu_cores: int


@dataclass(frozen=True)
class DiskLLMEstimate:
    params_billion: float
    bits_per_weight: float
    model_size_gb: float
    resident_set_gb: float
    required_ram_gb: float
    streamed_bytes_per_token_dense_gb: float
    streamed_bytes_per_token_flash_gb: float
    dense_tokens_per_second_upper: float
    flash_tokens_per_second_upper: float
    fits_resident_ram: bool | None
    fits_paper_ratio: bool | None


DEFAULT_HARDWARE_PROFILES = {
    "consumer_gen4": HardwareProfile(
        name="consumer_gen4",
        ssd_bandwidth_gbps=7.0,
        dram_gb=32.0,
        cpu_cores=8,
    ),
    "workstation_gen5": HardwareProfile(
        name="workstation_gen5",
        ssd_bandwidth_gbps=14.0,
        dram_gb=128.0,
        cpu_cores=24,
    ),
    "dual_nvme_workstation": HardwareProfile(
        name="dual_nvme_workstation",
        ssd_bandwidth_gbps=24.0,
        dram_gb=192.0,
        cpu_cores=32,
    ),
}


def _weights_size_gb(params_billion: float, bits_per_weight: float) -> float:
    if params_billion <= 0:
        raise ValueError("Parameter count must be positive")
    if bits_per_weight <= 0:
        raise ValueError("Bit-width must be positive")
    total_bytes = params_billion * DECIMAL_GB * (bits_per_weight / 8.0)
    return total_bytes / DECIMAL_GB


def estimate_disk_llm(
    params_billion: float,
    bits_per_weight: float = 4.0,
    *,
    ssd_bandwidth_gbps: float,
    dram_gb: float | None = None,
    resident_fraction: float = DEFAULT_RESIDENT_FRACTION,
    streamed_fraction: float = DEFAULT_STREAMED_FRACTION,
    active_ffn_fraction: float = DEFAULT_ACTIVE_FFN_FRACTION,
    io_efficiency: float = DEFAULT_IO_EFFICIENCY,
    working_memory_multiplier: float = DEFAULT_WORKING_MEMORY_MULTIPLIER,
    fixed_ram_overhead_gb: float = DEFAULT_FIXED_RAM_OVERHEAD_GB,
    validated_model_to_dram_ratio: float = DEFAULT_VALIDATED_MODEL_TO_DRAM_RATIO,
) -> DiskLLMEstimate:
    if ssd_bandwidth_gbps <= 0:
        raise ValueError("SSD bandwidth must be positive")
    if dram_gb is not None and dram_gb <= 0:
        raise ValueError("DRAM must be positive when provided")
    if not 0 < resident_fraction <= 1:
        raise ValueError("Resident fraction must be in (0, 1]")
    if not 0 < streamed_fraction <= 1:
        raise ValueError("Streamed fraction must be in (0, 1]")
    if resident_fraction + streamed_fraction > 1.000001:
        raise ValueError("Resident and streamed fractions cannot exceed 1.0 in total")
    if not 0 < active_ffn_fraction <= 1:
        raise ValueError("Active FFN fraction must be in (0, 1]")
    if not 0 < io_efficiency <= 1:
        raise ValueError("I/O efficiency must be in (0, 1]")
    if working_memory_multiplier < 1:
        raise ValueError("Working-memory multiplier must be >= 1")
    if fixed_ram_overhead_gb < 0:
        raise ValueError("Fixed RAM overhead must be non-negative")
    if validated_model_to_dram_ratio <= 0:
        raise ValueError("Validated model-to-DRAM ratio must be positive")

    model_size_gb = _weights_size_gb(params_billion, bits_per_weight)
    resident_set_gb = model_size_gb * resident_fraction
    required_ram_gb = resident_set_gb * working_memory_multiplier + fixed_ram_overhead_gb

    dense_stream_gb = model_size_gb
    flash_stream_gb = model_size_gb * streamed_fraction * active_ffn_fraction
    effective_bandwidth_gbps = ssd_bandwidth_gbps * io_efficiency

    dense_tps = effective_bandwidth_gbps / dense_stream_gb
    flash_tps = effective_bandwidth_gbps / flash_stream_gb

    fits_resident_ram = None if dram_gb is None else required_ram_gb <= dram_gb
    fits_paper_ratio = None if dram_gb is None else model_size_gb <= (dram_gb * validated_model_to_dram_ratio)

    return DiskLLMEstimate(
        params_billion=params_billion,
        bits_per_weight=bits_per_weight,
        model_size_gb=model_size_gb,
        resident_set_gb=resident_set_gb,
        required_ram_gb=required_ram_gb,
        streamed_bytes_per_token_dense_gb=dense_stream_gb,
        streamed_bytes_per_token_flash_gb=flash_stream_gb,
        dense_tokens_per_second_upper=dense_tps,
        flash_tokens_per_second_upper=flash_tps,
        fits_resident_ram=fits_resident_ram,
        fits_paper_ratio=fits_paper_ratio,
    )


def _format_fit(value: bool | None) -> str:
    if value is None:
        return "n/a"
    return "yes" if value else "no"


def _render_table(estimates: list[DiskLLMEstimate], hardware_name: str) -> str:
    lines = [
        f"Disk-resident LLM feasibility ({hardware_name})",
        (
            "model      size_gb  resident_gb  req_ram_gb  dense_tps<=  "
            "flash_tps<=  fits_ram  fits_2x_dram"
        ),
    ]
    for estimate in estimates:
        label = f"{estimate.params_billion:g}B/{estimate.bits_per_weight:g}b"
        lines.append(
            f"{label:<10} "
            f"{estimate.model_size_gb:>7.1f} "
            f"{estimate.resident_set_gb:>12.1f} "
            f"{estimate.required_ram_gb:>11.1f} "
            f"{estimate.dense_tokens_per_second_upper:>11.2f} "
            f"{estimate.flash_tokens_per_second_upper:>11.2f} "
            f"{_format_fit(estimate.fits_resident_ram):>9} "
            f"{_format_fit(estimate.fits_paper_ratio):>13}"
        )
    return "\n".join(lines)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate disk-resident LLM storage size and SSD-bandwidth-limited throughput. "
            "The sparse flash path assumes an LLM-in-a-Flash-style split where attention "
            "weights stay resident and only a small active FFN slice is streamed per token."
        )
    )
    parser.add_argument(
        "--hardware",
        choices=sorted(DEFAULT_HARDWARE_PROFILES),
        default="workstation_gen5",
        help="Named hardware profile to evaluate.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        type=float,
        default=[7, 13, 34, 70, 405],
        help="Model sizes in billions of parameters.",
    )
    parser.add_argument(
        "--bits",
        type=float,
        default=4.0,
        help="Bits per weight after quantization/compression.",
    )
    parser.add_argument(
        "--active-ffn-fraction",
        type=float,
        default=DEFAULT_ACTIVE_FFN_FRACTION,
        help="Fraction of streamed FFN weights read per token in the sparse flash path.",
    )
    parser.add_argument(
        "--resident-fraction",
        type=float,
        default=DEFAULT_RESIDENT_FRACTION,
        help="Fraction of total weights kept resident in RAM.",
    )
    parser.add_argument(
        "--streamed-fraction",
        type=float,
        default=DEFAULT_STREAMED_FRACTION,
        help="Fraction of total weights treated as streamable from disk.",
    )
    parser.add_argument(
        "--io-efficiency",
        type=float,
        default=DEFAULT_IO_EFFICIENCY,
        help="End-to-end fraction of advertised SSD bandwidth that is actually usable.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    hardware = DEFAULT_HARDWARE_PROFILES[args.hardware]
    estimates = [
        estimate_disk_llm(
            params_billion=params,
            bits_per_weight=args.bits,
            ssd_bandwidth_gbps=hardware.ssd_bandwidth_gbps,
            dram_gb=hardware.dram_gb,
            active_ffn_fraction=args.active_ffn_fraction,
            resident_fraction=args.resident_fraction,
            streamed_fraction=args.streamed_fraction,
            io_efficiency=args.io_efficiency,
        )
        for params in args.models
    ]
    print(_render_table(estimates, hardware.name))
    print()
    print(
        "Assumptions: size_gb is weight storage only; req_ram_gb includes resident weights, "
        "1.5x working-memory headroom, and 4 GB fixed overhead; dense_tps<= is the worst-case "
        "upper bound if the full model must be streamed each token; flash_tps<= is the "
        "SSD-bandwidth-limited upper bound for an LLM-in-a-Flash-style sparse FFN path."
    )


if __name__ == "__main__":
    main()
