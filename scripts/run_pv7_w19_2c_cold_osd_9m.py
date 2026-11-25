"""Run PV7 Bundle 58 against file-backed fixtures or the Spark 1 cold corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import uuid
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from chelated_pv7_spine.cpu_io_cage import (  # noqa: E402
    CgroupCageConfig,
    assert_zero_cuda_context,
    build_cpu_only_env,
    build_systemd_run_command,
)


def memory_cage(pid, max_bytes):
    lines = Path(f"/proc/{pid}/cgroup").read_text().splitlines()
    group_id = next((line[3:] for line in lines if line.startswith("0::")), None)
    if group_id is None:
        raise RuntimeError("A cgroup v2 memory cage is required")
    group = Path("/sys/fs/cgroup") / group_id.lstrip("/")
    if (group / "memory.max").read_text().strip() != str(max_bytes):
        raise RuntimeError("Each live system must run with MemoryMax=4 GiB")
    if (group / "memory.swap.max").read_text().strip() != "0":
        raise RuntimeError("Each live system must run with MemorySwapMax=0")
    events = {key: int(value) for key, value in map(str.split, (group / "memory.events").read_text().splitlines())}
    peak = int((group / "memory.peak").read_text())
    if peak > max_bytes or events["oom"] or events["oom_kill"]:
        raise RuntimeError("A 4 GiB worker exceeded its peak or OOM gate")
    return group, {"peak_bytes": peak, "memory_max_bytes": max_bytes, "swap_max_bytes": 0, "events": events}


def measured_device(name, paths):
    if name in (".", "..") or not re.fullmatch(r"[\w.-]+", name):
        raise ValueError("Expected a Linux block device basename")
    sysdev = Path("/sys/class/block") / name
    reported = (sysdev / "dev").read_text().strip()
    for path in paths:
        device = path.stat().st_dev
        if reported != f"{os.major(device)}:{os.minor(device)}":
            raise RuntimeError("OSD extent and Qdrant storage must share the measured block device")
    return sysdev / "stat"


def check_qdrant_process(args, collection=None):
    target = urlsplit(args.qdrant_url)
    if (
        target.scheme not in ("http", "https") or target.hostname not in ("localhost", "127.0.0.1", "::1")
        or target.username or target.password or target.query or target.fragment or target.path not in ("", "/")
    ):
        raise ValueError("Qdrant must be a local HTTP(S) service on the measured Spark 1 host")
    pid = args.qdrant_pid
    proc = Path(f"/proc/{pid}")
    storage = args.qdrant_storage.resolve(strict=True)
    cwd = (proc / "cwd").resolve(strict=True)
    executable = (proc / "exe").resolve(strict=True).name
    if "qdrant" not in executable.lower():
        raise ValueError("The supplied Qdrant PID does not identify a Qdrant executable")
    outside_cwd = not storage.is_relative_to(cwd)
    has_storage = any(
        descriptor.resolve().is_relative_to(storage) for descriptor in (proc / "fd").iterdir()
    )
    if outside_cwd and not has_storage:
        raise ValueError("Qdrant process storage could not be tied to the measured directory")
    if collection is not None:
        persisted = (storage / "collections" / collection).resolve(strict=True)
        if not persisted.is_relative_to(storage):
            raise RuntimeError("The Qdrant collection resolved outside the measured storage directory")
        measured_device(args.block_device, (persisted,))
    return storage


def staged_corpus(root):
    from chelated_cold_osd_scanner import LIVE_DOC_MAX, LIVE_DOC_MIN, VectorCorpus, input_path, normalized
    import numpy as np

    root = root.resolve(strict=True)
    manifest_path = input_path(root, "manifest.json")
    raw = manifest_path.read_bytes()
    manifest = json.loads(raw)
    if manifest.get("schema") != "pv7-cold-osd-v1" or not isinstance(manifest.get("shards"), list):
        raise ValueError("Expected a pv7-cold-osd-v1 manifest with an ordered shards list")
    if not manifest["shards"] or len(set(manifest["shards"])) != len(manifest["shards"]):
        raise ValueError("Staged corpus shards must be distinct")

    def matrix(name):
        return np.load(input_path(root, name), mmap_mode="r", allow_pickle=False)

    corpus = VectorCorpus(matrix(name) for name in manifest["shards"])
    if not LIVE_DOC_MIN <= corpus.count < LIVE_DOC_MAX:
        raise ValueError("Live mode requires a corpus whose count rounds to 9.47 million")
    centers = matrix(manifest["centroids"])
    dev = normalized(matrix(manifest["dev_queries"]))
    holdout = normalized(matrix(manifest["holdout_queries"]))
    if any(data.ndim != 2 or not len(data) or data.shape[1] != corpus.dimension for data in (centers, dev, holdout)):
        raise ValueError("Invalid centroid or query matrix")
    def key(row):
        return hashlib.sha256(np.asarray(row, dtype="<f4").tobytes()).digest()

    dev_ids, holdout_ids = ({key(row) for row in data} for data in (dev, holdout))
    if len(dev_ids) != len(dev) or len(holdout_ids) != len(holdout) or dev_ids & holdout_ids:
        raise ValueError("DEV and holdout queries must be distinct and disjoint")
    # Content hashes identify the inputs independently of manifest naming and staging location.
    from chelated_cold_osd_scanner import digest_file

    assets = [*manifest["shards"], *(manifest[key] for key in ("centroids", "dev_queries", "holdout_queries"))]
    hashes = [digest_file(input_path(root, name)) for name in assets]
    return corpus, centers, dev, holdout, {
        "manifest_sha256": hashlib.sha256(raw).hexdigest(), "array_sha256": hashes,
    }


def tune_recall(engine, queries, exact, parameter, candidates):
    from chelated_cold_osd_scanner import score_queries

    rows = []
    for value in sorted(set(candidates)):
        setattr(engine, parameter, value)
        score = score_queries(engine, queries, exact)
        rows.append((value, score["hits"] / score["reference_total"]))
    ceiling = max(recall for _, recall in rows)
    return rows, ceiling


def lock_query_parameters(engines, geometry, queries, exact, qdrant_ef):
    count = len(geometry.centroids)
    probes = [min(value, count) for value in (1, 4, 8, 16, 32, 64, 128)]
    tuning = {}
    available = {}
    for name, engine in engines.items():
        parameter, settings = ("hnsw_ef", qdrant_ef) if name == "qdrant_on_disk" else ("nprobe", probes)
        rows, ceiling = tune_recall(engine, queries, exact, parameter, settings)
        available[name] = (parameter, rows, ceiling)
    target = min(ceiling for _, _, ceiling in available.values())
    for name, (parameter, rows, ceiling) in available.items():
        matched = [(value, recall) for value, recall in rows if abs(recall - target) <= 0.01 + 1e-12]
        best = (matched or sorted(rows, key=lambda row: (abs(row[1] - target), row[0])))[0]
        setattr(engines[name], parameter, best[0])
        tuning[name] = {
            "parameter": parameter, "chosen": best[0], "dev_recall": best[1],
            "best_dev_recall": ceiling, "dev_curve": rows,
        }
    return tuning


def run_live(args):
    import numpy as np

    from chelated_cold_osd_scanner import (
        MEMORY_MAX,
        MIB,
        TOP_K,
        TRACE_PAIRS,
        BlockWriteProbe,
        ColdOSD,
        EK5ExtentFence,
        MatchedIVFSQ8,
        QdrantOnDisk,
        build_extents,
        compare_traces,
        digest_file,
        run_trace,
        summarize_traces,
        withdrawal_pairs,
    )

    assert_zero_cuda_context()
    storage = check_qdrant_process(args)
    worker, _ = memory_cage(os.getpid(), MEMORY_MAX)
    qdrant, _ = memory_cage(args.qdrant_pid, MEMORY_MAX)
    assert_zero_cuda_context(target_pid=args.qdrant_pid)
    if worker == qdrant:
        raise RuntimeError("The scan worker and Qdrant require independent 4 GiB cages")
    stat_path = measured_device(args.block_device, (args.scratch_dir, storage))
    corpus, centers, dev, queries, provenance = staged_corpus(args.shard_dir)
    if not all(math.isfinite(value) and value > 0 for value in (args.quiet_interval, args.quiet_timeout)):
        raise ValueError("Positive finite quiet-window durations are required")
    collection = f"pv7_58_{uuid.uuid4().hex}"
    with tempfile.TemporaryDirectory(prefix=".pv7-58-", dir=args.scratch_dir) as directory:
        geometry = build_extents(corpus, centers, dev, directory, args.extent_mib * MIB)
        # The fence WAL lives on the measured device, so its fsynced epochs count in device writes.
        wal_path = Path(directory) / "cold-osd-fence.wal"
        candidate = ColdOSD(geometry, 1, wal_path=wal_path)
        sq8 = MatchedIVFSQ8(geometry, 1, directory, compact_fraction=args.sq8_compact_fraction)
        qdrant_engine = QdrantOnDisk(
            args.qdrant_url, collection, corpus, api_key=os.getenv("QDRANT_API_KEY"),
            hnsw_m=args.qdrant_m, ef_construct=args.qdrant_ef_construct,
        )
        # Keep observing beyond a full configured Qdrant flush interval once the optimizer turns green.
        quiet_samples = max(3, math.ceil(qdrant_engine.flush_interval / args.quiet_interval) + 2)
        if args.quiet_timeout <= quiet_samples * args.quiet_interval:
            raise ValueError("The timeout must cover the configured optimizer flush and quiet window")
        probe = BlockWriteProbe(
            stat_path, quiet_samples=quiet_samples, interval=args.quiet_interval, timeout=args.quiet_timeout,
        )
        qdrant_engine.populate()
        probe.wait(qdrant_engine.quiescent)
        check_qdrant_process(args, collection)
        if qdrant_engine.count() != corpus.count:
            raise RuntimeError("Qdrant point count differs from the fixed corpus")
        exact_dev = corpus.exact_top(dev, TOP_K)
        oracle = corpus.exact_top(queries, TOP_K + args.withdrawals)
        engines = {"cold_osd": candidate, "matched_ivf_sq8": sq8, "qdrant_on_disk": qdrant_engine}
        tuning = lock_query_parameters(engines, geometry, dev, exact_dev, args.qdrant_ef)
        cohort = list(withdrawal_pairs(geometry, args.withdrawals))
        immutable_before = digest_file(geometry.path)
        results = {}
        for name, engine in engines.items():
            probe.wait(qdrant_engine.quiescent)
            results[name] = []
            for pair, layout, ids in cohort:
                row = run_trace(engine, queries, oracle, ids, probe)
                results[name].append({
                    "pair": pair, "layout": layout,
                    "affected_extents": int(np.unique(geometry.locations[ids] // geometry.slots).size), **row,
                })
                memory_cage(os.getpid(), MEMORY_MAX)
                memory_cage(args.qdrant_pid, MEMORY_MAX)
        if qdrant_engine.count() != corpus.count:
            raise RuntimeError("Qdrant reinstatement did not restore the fixed corpus cardinality")
        immutable = immutable_before == digest_file(geometry.path)
        recovered = EK5ExtentFence.recover_from_wal(geometry, wal_path)
        wal_replay = {
            **recovered.recovery,
            "bits_match": bool(np.array_equal(recovered.bits, candidate.fence.bits)),
            "epoch_matches": recovered.epoch == candidate.fence.epoch,
        }
        summaries = {name: summarize_traces(rows) for name, rows in results.items()}
        comparisons = compare_traces(summaries)
        _, worker_memory = memory_cage(os.getpid(), MEMORY_MAX)
        _, qdrant_memory = memory_cage(args.qdrant_pid, MEMORY_MAX)
        assert_zero_cuda_context()
        assert_zero_cuda_context(target_pid=args.qdrant_pid)
        passed = immutable and wal_replay["bits_match"] and wal_replay["epoch_matches"] and all(
            cells["passed"] for layouts in comparisons.values() for cells in layouts.values()
        )
        return {
            "mode": "run-live-spark1", "passed": passed,
            "docs": corpus.count, "dimensions": corpus.dimension, "dev_queries": len(dev),
            "holdout_queries": len(queries), "extent_bytes": geometry.extent_bytes,
            "extent_count": len(geometry.counts), "fence_bytes": candidate.fence.bits.nbytes,
            "sidecar_bytes": geometry.sidecar.nbytes, "trace_pairs": TRACE_PAIRS,
            "withdrawals_per_trace": args.withdrawals, "extents_unchanged": immutable,
            "immutable_extent_sha256": immutable_before, "provenance": provenance,
            "sq8_compaction_fraction": args.sq8_compact_fraction,
            "qdrant_hnsw": {"m": args.qdrant_m, "ef_construct": args.qdrant_ef_construct, "on_disk": True},
            "write_accounting": "512_byte_device_sectors_through_optimizer_and_device_quiescence",
            "cold_osd_write_source": "fsynced_ek5_fence_wal",
            "fence_wal_bytes": candidate.fence.wal_bytes, "fence_wal_records": candidate.fence.wal_records,
            "fence_wal_replay": wal_replay,
            "quiet_interval_seconds": args.quiet_interval, "quiet_samples": quiet_samples,
            "qdrant_flush_interval_seconds": qdrant_engine.flush_interval,
            "query_tuning": tuning, "worker_memory": worker_memory, "qdrant_memory": qdrant_memory,
            "summaries": summaries, "comparisons": comparisons, "traces": results,
        }


def main(argv=None):
    cli = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("local-drycheck", "run-live-spark1"), required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--shard-dir", type=Path, help="Directory with manifest.json and floating-point NPY arrays")
    parser.add_argument("--scratch-dir", type=Path, help="Writable directory on the measured device")
    parser.add_argument("--qdrant-url")
    parser.add_argument("--qdrant-pid", type=int)
    parser.add_argument("--qdrant-storage", type=Path)
    parser.add_argument("--block-device")
    parser.add_argument("--extent-mib", type=int, choices=range(2, 9), default=2)
    parser.add_argument("--withdrawals", type=int, default=2048)
    parser.add_argument("--sq8-compact-fraction", type=float, default=0.2)
    parser.add_argument("--qdrant-ef", type=int, nargs="+", default=[16, 32, 64, 128, 256])
    parser.add_argument("--qdrant-m", type=int, default=16)
    parser.add_argument("--qdrant-ef-construct", type=int, default=100)
    parser.add_argument("--quiet-interval", type=float, default=1.0)
    parser.add_argument("--quiet-timeout", type=float, default=600.0)
    parser.add_argument("--caged", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(cli)
    os.environ.update(build_cpu_only_env())
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    try:
        if args.mode == "local-drycheck":
            from chelated_cold_osd_scanner import drycheck

            with tempfile.TemporaryDirectory(prefix=".pv7-58-", dir=ROOT) as directory:
                report = drycheck(directory)
        else:
            required = ("shard_dir", "scratch_dir", "qdrant_url", "qdrant_pid", "qdrant_storage", "block_device")
            absent = [f"--{option.replace('_', '-')}" for option in required if not getattr(args, option)]
            if absent:
                parser.error("live mode requires " + ", ".join(absent))
            if sys.platform != "linux":
                raise RuntimeError("Spark 1 live mode requires Linux cgroup v2 and a local Qdrant instance")
            if min(args.qdrant_pid, args.withdrawals, args.qdrant_m, args.qdrant_ef_construct, *args.qdrant_ef) <= 0:
                parser.error("PID, withdrawal count and HNSW tuning values must be positive")
            if not args.caged:
                command = build_systemd_run_command(
                    [sys.executable, "-B", str(Path(__file__).resolve()), *cli, "--caged"],
                    CgroupCageConfig.from_size_spec("4G", memory_swap_max=0),
                )
                return subprocess.run(command, env=build_cpu_only_env(), check=False).returncode
            report = run_live(args)
        receipt = json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
        if args.output:
            with args.output.open("x", encoding="utf-8") as output:
                output.write(receipt)
        else:
            print(receipt, end="")
        return 0 if report["passed"] else 1
    except (KeyError, OSError, RuntimeError, TypeError, ValueError) as exc:
        parser.exit(2, f"{exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
