"""Run PV7 Bundle 56's disk MoE drycheck or the caged Spark 2 experiment."""

from __future__ import annotations

import argparse
import bisect
import ctypes
import json
import mmap
import os
import re
import struct
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chelated_disk_moe_runtime import (  # noqa: E402
    MEMORY_MAX,
    MIB,
    RUNGS,
    QwenDiskTarget,
    QwenDraft,
    ResidentBudget,
    compare_lossless,
    drycheck,
    file_digest,
    greedy_decode,
    json_digest,
    reference_quality,
    select_dev_best,
    summarize_quality,
)
from chelated_pv7_spine.burn_ledger import text_fingerprint  # noqa: E402
from chelated_pv7_spine.cpu_io_cage import (  # noqa: E402
    CgroupCageConfig,
    ResidentIdleLeaseVerifier,
    assert_zero_cuda_context,
    build_cpu_only_env,
    build_systemd_run_command,
    classify_storage_tier,
    diff_block_stat,
    diff_proc_io,
    parse_sys_block_stat,
    read_proc_io,
    reconcile_proc_and_block_io,
)
from chelated_pv7_spine.tra1_weight_sha_guard import TRA1StateIdentityContract  # noqa: E402

SUITE_COUNTS = {"HumanEval": 164, "Suite A": 60}
TRIALS = 3
PROTECTED = re.compile(r"sealed_key|blind-followup|pm4_human_audit_100_blind_sheet\.csv", re.I)


def checked_input(path):
    path = Path(path).resolve(strict=True)
    if PROTECTED.search(path.as_posix()):
        raise ValueError("The selected benchmark input is protected")
    return path


def prompt_rows(path, *, dev):
    rows = []
    for line in checked_input(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict) or any(
            not isinstance(row.get(key), str) or not row[key].strip() for key in ("id", "suite", "prompt")
        ):
            raise ValueError("Prompt rows require nonempty id, suite, and prompt strings")
        if not dev and (not isinstance(row.get("reference"), str) or not row["reference"].strip()):
            raise ValueError("Holdout rows require reference continuations for the quality arm")
        rows.append(row)
    identities = {(row["suite"], row["id"]) for row in rows}
    fingerprints = {text_fingerprint(row["prompt"]) for row in rows}
    if not rows or len(identities) != len(rows) or len(fingerprints) != len(rows):
        raise ValueError("Prompt identities and text fingerprints must be unique within each split")
    if not dev and {s: sum(row["suite"] == s for row in rows) for s in {r["suite"] for r in rows}} != SUITE_COUNTS:
        raise ValueError("Holdout requires exactly 164 HumanEval and 60 Suite A prompts")
    return rows, identities, fingerprints


def cgroup_path():
    if sys.platform != "linux":
        raise RuntimeError("Live runs require Linux cgroup v2")
    for line in Path("/proc/self/cgroup").read_text().splitlines():
        if line.startswith("0::"):
            return Path("/sys/fs/cgroup") / line[3:].lstrip("/")
    raise RuntimeError("Unable to identify the current cgroup v2")


def read_memory(group):
    if (group / "memory.max").read_text().strip() != str(MEMORY_MAX):
        raise RuntimeError("The live MoE process must have MemoryMax=8 GiB")
    if (group / "memory.swap.max").read_text().strip() != "0":
        raise RuntimeError("The live MoE process must have MemorySwapMax=0")
    counters = {key: int(value) for key, value in map(str.split, (group / "memory.stat").read_text().splitlines())}
    events = {key: int(value) for key, value in map(str.split, (group / "memory.events").read_text().splitlines())}
    peak = int((group / "memory.peak").read_text())
    if peak > MEMORY_MAX or events["oom"] or events["oom_kill"]:
        raise RuntimeError("Cgroup memory limit or OOM gate failed")
    return {
        "peak": peak, "current": int((group / "memory.current").read_text()), "events": events,
        "page_tables": counters["pagetables"] + counters.get("sec_pagetables", 0),
        "file": counters["file"], "kernel": counters["kernel"], "anonymous": counters["anon"],
    }


def media_details(path, device, dm_table_path):
    if not re.fullmatch(r"[\w.-]+", device) or device in (".", ".."):
        raise ValueError("Expected a Linux block device basename")
    sysdev = Path("/sys/class/block") / device
    mount_device = path.stat().st_dev
    if (sysdev / "dev").read_text().strip() != f"{os.major(mount_device)}:{os.minor(mount_device)}":
        raise RuntimeError("Model file and measured block device differ")
    disk = sysdev.resolve()
    if (disk / "partition").exists():
        disk = disk.parent
    rota = int((disk / "queue/rotational").read_text())
    dm_table = checked_input(dm_table_path).read_text() if dm_table_path else ""
    virtual = (
        "/virtual/" in disk.as_posix() or "/virtio" in disk.as_posix()
        or disk.name.startswith(("vd", "xvd", "nbd", "loop", "zram", "md"))
    )
    device_mapper = (disk / "dm").is_dir()
    if device_mapper != bool(dm_table):
        raise ValueError("Supply --dm-table only for the actual device-mapper target")
    tier = classify_storage_tier(device, rota=rota, dm_table=dm_table)
    if device_mapper and not tier.emulated:
        raise ValueError("The device-mapper table did not identify dm-delay")
    return sysdev / "stat", {
        "storage_tier": "emulated_rotational_delay" if tier.emulated else tier.tier if not virtual else "virtual",
        "rota": rota,
        "mechanical_order_eligible": tier.allows_cylinder_order_claim and not virtual,
        "dm_delay_ms": tier.dm_delay_ms,
        "dm_table_sha256": json_digest(dm_table) if dm_table else None,
    }


def physical_lookup(path):
    """Read FIEMAP physical offsets from the mounted rotating disk."""
    import fcntl

    spans = []
    limit = 256
    header = struct.Struct("=QQIIII")
    record = struct.Struct("=QQQQQIIII")
    start = 0
    fd = os.open(path, os.O_RDONLY)
    try:
        while start < path.stat().st_size:
            data = bytearray(header.size + limit * record.size)
            header.pack_into(data, 0, start, (2 ** 64 - 1) - start, 1, 0, limit, 0)
            fcntl.ioctl(fd, 0xC020660B, data, True)
            _, _, _, mapped, _, _ = header.unpack_from(data)
            if not mapped:
                break
            for index in range(mapped):
                logical, physical, length, _, _, flags, _, _, _ = record.unpack_from(
                    data, header.size + record.size * index,
                )
                if flags & (0x2 | 0x4 | 0x8 | 0x80 | 0x800):
                    raise RuntimeError("FIEMAP did not provide a stable physical model extent")
                spans.append((logical, physical, length))
            start = logical + length
            if flags & 0x1:
                break
    finally:
        os.close(fd)
    starts = [row[0] for row in spans]

    def lookup(offset):
        index = bisect.bisect_right(starts, offset) - 1
        if index < 0:
            raise RuntimeError("Model extent has no physical mapping")
        logical, physical, length = spans[index]
        if offset >= logical + length:
            raise RuntimeError("Model extent falls into a filesystem hole")
        return physical + offset - logical

    return lookup


def expert_page_residency(pager):
    """Return ``(expert_pages, resident_pages)`` from mincore without faulting in model bytes."""
    page_size = mmap.PAGESIZE
    page_count = (os.fstat(pager.fd).st_size + page_size - 1) // page_size
    states = (ctypes.c_ubyte * page_count)()
    libc = ctypes.CDLL(None, use_errno=True)
    mincore = libc.mincore
    mincore.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.POINTER(ctypes.c_ubyte))
    with mmap.mmap(pager.fd, 0, access=mmap.ACCESS_COPY) as model:
        anchor = ctypes.c_char.from_buffer(model)
        try:
            if mincore(ctypes.byref(anchor), len(model), states):
                raise OSError(ctypes.get_errno(), "mincore failed")
        finally:
            del anchor
    pages = 0
    resident = 0
    for extent in pager.extents.values():
        start = extent.offset // page_size
        stop = (extent.offset + extent.size + page_size - 1) // page_size
        pages += stop - start
        resident += sum(value & 1 for value in states[start:stop])
    return pages, resident


def cold_pages(pager):
    """Read mincore residency for every expert page and fail if any is resident."""
    pages, resident = expert_page_residency(pager)
    if resident:
        raise RuntimeError(f"Cold-run expert cache still has {resident} of {pages} page references resident")
    return resident


def verify_cold_cache(pager, rung, *, residency=expert_page_residency):
    """Prove the page cache is cold before a cold timing cell, on every rung 1..6.

    ``pager.reset(cold=True)`` drops the in-process LRU/pin state and issues
    ``POSIX_FADV_DONTNEED``; that is advisory, so the kernel's mincore residency is
    re-read and any resident expert page fails the cell instead of timing a warm read.
    """
    pages, resident = residency(pager)
    if pages <= 0:
        raise RuntimeError(f"R{rung} cold verification covered no expert pages")
    if resident:
        raise RuntimeError(f"R{rung} cold-run expert cache still has {resident} of {pages} page references resident")
    return {"rung": rung, "method": "mincore", "expert_pages_checked": pages,
            "expert_pages_resident": resident, "verified_cold": True}


def prepare_cache_cell(pager, rung, temperature, *, residency=expert_page_residency):
    """Reset the pager for one timing cell; every cold cell on every rung is mincore-verified."""
    cold = temperature == "cold"
    pager.reset(cold=cold)
    return verify_cold_cache(pager, rung, residency=residency) if cold else None


def summarize_cold_verification(rung, cells):
    cold = [cell for cell in cells if cell["cache"] == "cold"]
    verified = [cell for cell in cold if (cell.get("cold_cache_verification") or {}).get("verified_cold") is True]
    return {"rung": rung, "method": "mincore", "cold_cells": len(cold), "verified_cold_cells": len(verified),
            "all_cold_cells_verified": bool(cold) and len(verified) == len(cold)}


def checked_lease(verifier, before, after):
    if not before.counter_totals or not before.active_gauges or not after.counter_totals or not after.active_gauges:
        raise RuntimeError("Resident idle lease requires request counters and in-flight gauges")
    for value in (*before.counter_totals.values(), *before.active_gauges.values(),
                  *after.counter_totals.values(), *after.active_gauges.values()):
        if not 0 <= value < float("inf"):
            raise RuntimeError("Invalid resident metrics sample")
    verdict = verifier.verify_lease(before, after)
    if not verdict.valid or before.active_gauges.keys() != after.active_gauges.keys():
        raise RuntimeError(f"Resident lease voided: {verdict.violations}")
    return verdict


def target_identity(gguf_sha, metadata, tokenizer_sha):
    return TRA1StateIdentityContract(
        base_weights_sha256=gguf_sha, adapter_weights_sha256=json_digest(None),
        config_sha256=json_digest(metadata), tokenizer_sha256=tokenizer_sha,
        kv_or_state_dtype="bfloat16", rope_freq_base=metadata["rope_base"], rope_scaling_hash=json_digest(None),
    ).digest()


def file_signature(path):
    file = path.stat()
    return [file.st_dev, file.st_ino, file.st_size, file.st_mtime_ns]


def run_worker(args):
    config = cgroup_path()
    read_memory(config)
    assert_zero_cuda_context()
    import torch
    from transformers import AutoTokenizer

    torch.set_num_threads(args.cpu_threads)
    torch.set_num_interop_threads(1)
    rung = args.worker_rung
    path = checked_input(args.iq2_gguf if rung == 5 else args.q4_gguf)
    signature = file_signature(path)
    stat, media = media_details(path, args.block_device, args.dm_table)
    lookup = physical_lookup(path) if rung == 6 and media["mechanical_order_eligible"] else None
    tokenizer = AutoTokenizer.from_pretrained(checked_input(args.target_tokenizer), local_files_only=True,
                                             trust_remote_code=False)
    tokenizer_sha = file_digest(checked_input(args.target_tokenizer / "tokenizer.json"))
    eos = tuple(set(token for token in (
        tokenizer.eos_token_id, tokenizer.get_vocab().get("<|endoftext|>"), tokenizer.get_vocab().get("<|im_end|>"),
    ) if token is not None))
    mask = (
        {int(layer): set(choices) for layer, choices in json.loads(checked_input(args.reap_mask).read_text()).items()}
        if rung == 5 else None
    )
    prompt_path = args.dev_prompts if args.worker_phase == "dev" else args.eval_prompts
    prompts, _, _ = prompt_rows(prompt_path, dev=args.worker_phase == "dev")
    cache = args.cache_mib * MIB if rung != 1 else 0
    pin = args.pin_mib * MIB if rung == 6 else 0
    target = QwenDiskTarget(path, rung=rung, cache_bytes=cache, pin_bytes=pin, context=args.context,
                            reap=mask, physical_offset=lookup)
    with target.pager:
        draft = QwenDraft(checked_input(args.draft_model), tokenizer, args.context) if rung in (4, 6) else None
        mem = read_memory(config)
        # The reserve covers mmap cache and remaining kernel charges, with cgroup peak as the inclusive guard.
        budget = ResidentBudget(
            target_weights=target.weight_bytes, draft_weights=draft.weight_bytes if draft else 0,
            kv_cache=target.kv_bytes + (draft.kv_max_bytes if draft else 0),
            page_tables=max(64 * MIB, mem["page_tables"]), expert_lru=cache, expert_hot_pin=pin,
            read_buffers=target.read_working_bytes, inference_workspace=384 * MIB,
            routing_and_runtime=512 * MIB, kernel_and_page_cache=512 * MIB,
        )
        resident = ResidentIdleLeaseVerifier()
        start_lease = resident.snapshot_from_url(args.resident_metrics_url)
        cells = []
        for row in prompts:
            prompt = tokenizer.encode(row["prompt"], add_special_tokens=False)
            if len(prompt) + args.max_tokens > args.context:
                raise ValueError(f"Prompt {row['id']} exceeds the configured context")
            reference = tokenizer.encode(row.get("reference", ""), add_special_tokens=False)
            for trial in range(1 if args.worker_phase == "dev" else TRIALS):
                for temperature in ("cold", "warm"):
                    cold_check = prepare_cache_cell(target.pager, rung, temperature)
                    cached = cold_check["expert_pages_resident"] if cold_check else None
                    io_before = read_proc_io(os.getpid())
                    device_before = parse_sys_block_stat(stat.read_text())
                    if device_before.in_flight:
                        raise RuntimeError("The measured disk had outstanding I/O before the timing cell")
                    start = time.perf_counter()
                    tokens, proposed, accepted = greedy_decode(
                        target, prompt, max_tokens=args.max_tokens, draft=draft, eos=eos,
                    )
                    elapsed = time.perf_counter() - start
                    device_delta = diff_block_stat(device_before, parse_sys_block_stat(stat.read_text()))
                    process_delta = diff_proc_io(io_before, read_proc_io(os.getpid()))
                    reconciliation = reconcile_proc_and_block_io(process_delta, device_delta)
                    # Mapped pages can fault via a different kernel accounting path; direct reads must reconcile.
                    if rung != 1 and reconciliation.external_background_io_suspected:
                        raise RuntimeError("Direct-pread timing cell could not attribute disk I/O to the caged process")
                    mem = read_memory(config)
                    if mem["page_tables"] > budget.page_tables:
                        raise RuntimeError("Page tables exceeded their explicit resident budget")
                    cell = {
                        "id": json_digest([row["suite"], row["id"]]), "suite": row["suite"], "trial": trial,
                        "cache": temperature, "cold_expert_pages_resident": cached,
                        "cold_cache_verification": cold_check,
                        "seconds": elapsed, "tokens": len(tokens), "token_sha256": json_digest(tokens),
                        "disk_read_bytes": device_delta.read_bytes, "requested_expert_bytes": target.pager.bytes_read,
                        "expert_read_calls": target.pager.read_calls, "proposed": proposed, "accepted": accepted,
                        "io_reconciliation": asdict(reconciliation),
                    }
                    if args.worker_phase == "eval" and rung in (1, 5):
                        cell["quality"] = reference_quality(tokens, reference)
                    cells.append(cell)
        lease = checked_lease(resident, start_lease, resident.snapshot_from_url(args.resident_metrics_url))
        assert_zero_cuda_context()
    if signature != file_signature(path):
        raise RuntimeError("The model artifact changed during measurement")
    return {
        "rung": rung, "phase": args.worker_phase, "model_metadata": target.metadata, "media": media,
        "tokenizer_sha256": tokenizer_sha, "model_signature": signature,
        "cylinder_ordering_exercised": lookup is not None, "budget": asdict(budget), "budget_total": budget.total,
        "memory": mem, "resident_lease": asdict(lease), "cells": cells,
        "cold_cache_verification": summarize_cold_verification(rung, cells),
    }


def launch_worker(argv, rung, phase):
    launch = build_systemd_run_command(
        [sys.executable, "-B", str(Path(__file__).resolve()), *argv,
         "--worker-rung", str(rung), "--worker-phase", phase],
        CgroupCageConfig.from_size_spec("8G", memory_swap_max=0),
    )
    env = build_cpu_only_env()
    env.update({"OPENBLAS_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false"})
    process = subprocess.run(launch, env=env, capture_output=True, text=True, check=False)
    if process.returncode:
        raise RuntimeError(f"R{rung} {phase} caged worker failed ({process.returncode}): {process.stderr.strip()}")
    return json.loads(process.stdout)


def run_live(args, argv):
    if sys.platform != "linux":
        raise RuntimeError("Spark 2 live mode requires Linux and the staged CPU-only artifacts")
    _, dev_ids, dev_fingerprints = prompt_rows(args.dev_prompts, dev=True)
    _, eval_ids, eval_fingerprints = prompt_rows(args.eval_prompts, dev=False)
    if dev_ids & eval_ids or dev_fingerprints & eval_fingerprints:
        raise ValueError("DEV and holdout prompt rows overlap")
    q4_path, iq2_path = (checked_input(path) for path in (args.q4_gguf, args.iq2_gguf))
    signatures = {path: file_signature(path) for path in (q4_path, iq2_path)}
    hashes = {path: file_digest(path) for path in signatures}
    draft = checked_input(args.draft_model)
    assets = sorted([*draft.glob("*.safetensors"), *draft.glob("*.safetensors.index.json")])
    if not any(path.suffix == ".safetensors" for path in assets):
        raise ValueError("The draft checkpoint must include local safetensors weights")
    assets.extend([draft / "config.json", draft / "tokenizer.json"])
    assets.extend(checked_input(path) for path in (
        args.dev_prompts, args.eval_prompts, args.reap_mask, args.target_tokenizer / "tokenizer.json",
    ))
    static_signatures = {path: file_signature(checked_input(path)) for path in assets}
    model_draft_sha = json_digest([file_digest(path) for path in assets[:-4]])
    prompt_hashes = [file_digest(path) for path in assets[-4:]]
    dev = {rung: launch_worker(argv, rung, "dev") for rung in range(1, 5)}
    best = select_dev_best({rung: report["cells"] for rung, report in dev.items()})
    oracle_dev = [cell["token_sha256"] for cell in dev[1]["cells"]]
    if any([cell["token_sha256"] for cell in report["cells"]] != oracle_dev for report in dev.values()):
        raise RuntimeError("DEV greedy Q4 token parity failed")
    holdout = {rung: launch_worker(argv, rung, "eval") for rung in dict.fromkeys((1, best, 5, 6))}
    for rows in (dev, holdout):
        for rung, report in rows.items():
            path = iq2_path if rung == 5 else q4_path
            if report.pop("model_signature") != signatures[path] or file_signature(path) != signatures[path]:
                raise RuntimeError("Model artifact changed between workers")
            report["target_state_sha256"] = target_identity(
                hashes[path], report["model_metadata"], report["tokenizer_sha256"],
            )
    if any(file_signature(path) != signature for path, signature in static_signatures.items()):
        raise RuntimeError("A staged prompt, tokenizer, draft checkpoint, or REAP mask changed during measurement")
    if {row["tokenizer_sha256"] for row in (*dev.values(), *holdout.values())} != {prompt_hashes[3]}:
        raise RuntimeError("Caged workers loaded different tokenizer bytes")
    lossless = compare_lossless(holdout[1]["cells"], holdout[best]["cells"], holdout[6]["cells"])
    cold_verification = {
        phase: {RUNGS[i - 1]: summarize_cold_verification(i, report["cells"]) for i, report in rows.items()}
        for phase, rows in (("dev", dev), ("holdout", holdout))
    }
    all_cold_verified = all(
        summary["all_cold_cells_verified"] for rows in cold_verification.values() for summary in rows.values()
    )
    return {
        "mode": "run-live-spark2", "passed": lossless["passed"] and all_cold_verified,
        "cold_cache_verification_by_rung": cold_verification, "all_cold_cells_verified": all_cold_verified,
        "dev_best": RUNGS[best - 1],
        "hardware": holdout[6]["media"], "lossless_q4_endpoint": lossless,
        "separate_reference_continuation_quality": {
            "method": "exact_match_and_token_similarity_against_staged_references",
            "q4_reference": summarize_quality(holdout[1]["cells"]),
            "iq2_reap": summarize_quality(holdout[5]["cells"]),
        },
        "dev": {RUNGS[i - 1]: report for i, report in dev.items()},
        "holdout": {RUNGS[i - 1]: report for i, report in holdout.items() if i != 5},
        "iq2_reap_coverage": {key: val for key, val in holdout[5].items() if key != "cells"},
        "input_sha256": {
            "q4_gguf": hashes[q4_path], "iq2_gguf": hashes[iq2_path],
            "dev_prompts": prompt_hashes[0], "eval_prompts": prompt_hashes[1], "reap_mask": prompt_hashes[2],
            "draft_checkpoint": model_draft_sha, "tokenizer": prompt_hashes[3],
        },
    }


def main(argv=None):
    cli = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("local-drycheck", "run-live-spark2"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--q4-gguf", type=Path)
    parser.add_argument("--iq2-gguf", type=Path)
    parser.add_argument("--reap-mask", type=Path, help="JSON object mapping layer numbers to retained expert IDs")
    parser.add_argument("--target-tokenizer", type=Path)
    parser.add_argument("--draft-model", type=Path)
    parser.add_argument("--dev-prompts", type=Path, help="DEV JSONL rows with id, suite, prompt")
    parser.add_argument("--eval-prompts", type=Path, help="Holdout JSONL with id, suite, prompt, reference")
    parser.add_argument("--block-device")
    parser.add_argument("--dm-table", type=Path)
    parser.add_argument("--resident-metrics-url")
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--cache-mib", type=int, default=384)
    parser.add_argument("--pin-mib", type=int, default=1024)
    parser.add_argument("--cpu-threads", type=int, default=4)
    parser.add_argument("--worker-rung", type=int, choices=range(1, 7), help=argparse.SUPPRESS)
    parser.add_argument("--worker-phase", choices=("dev", "eval"), help=argparse.SUPPRESS)
    args = parser.parse_args(cli)
    os.environ.update(build_cpu_only_env())
    try:
        if args.mode == "local-drycheck":
            with tempfile.TemporaryDirectory(prefix=".pv7-56-", dir=ROOT) as directory:
                report = drycheck(directory)
        else:
            required = ("q4_gguf", "iq2_gguf", "reap_mask", "target_tokenizer", "draft_model", "dev_prompts",
                        "eval_prompts", "block_device", "resident_metrics_url")
            absent = [f"--{option.replace('_', '-')}" for option in required if not getattr(args, option)]
            if absent:
                parser.error("live mode requires " + ", ".join(absent))
            if min(args.max_tokens, args.cache_mib, args.pin_mib, args.cpu_threads) <= 0 or args.context <= args.max_tokens:
                parser.error("live limits must be positive and context must exceed max tokens")
            if bool(args.worker_rung) != bool(args.worker_phase):
                parser.error("invalid caged worker arguments")
            report = run_worker(args) if args.worker_rung else run_live(args, cli)
        receipt = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
        if args.output and not args.worker_rung:
            with args.output.open("x", encoding="utf-8") as output:
                output.write(receipt)
        else:
            print(receipt, end="")
        return 0 if args.worker_rung or report["passed"] else 1
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
