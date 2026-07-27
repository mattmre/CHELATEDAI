"""GPU campaign for the Rung 16 quant-aware routing promotion plane."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

from benchmark_utils import canonicalize_id, load_mteb_data
from embedding_backend import create_embedding_backend
from post_bank_correction import cluster_vectors
from quant_aware_routing import (
    RoutingPlaneConfig,
    fit_quant_aware_plane,
    three_way_seeded_split,
)
from query_encoder_drift import QueryEncoderDrift


DEFAULT_PREREG = "prereg_rung16.json"
DEFAULT_MANIFEST = "docs/rung16-quant-aware-routing-manifest-2026-07.json"
DEFAULT_REPORT = "docs/rung16-quant-aware-routing-results-2026-07.md"


@dataclass(frozen=True)
class ArenaSpec:
    key: str
    title: str
    datasets: tuple[str, ...]
    cluster_rule: str


ARENAS = (
    ArenaSpec(
        key="arena_a_default_swap",
        title="Arena A — default SciFact swap",
        datasets=("SciFact",),
        cluster_rule="kmeans",
    ),
    ArenaSpec(
        key="arena_b_multi_domain",
        title="Arena B — mixed SciFact + NFCorpus + FiQA2018",
        datasets=("SciFact", "NFCorpus", "FiQA2018"),
        cluster_rule="domain",
    ),
)


def run_campaign(
    *,
    prereg_path: str = DEFAULT_PREREG,
    manifest_path: str = DEFAULT_MANIFEST,
    report_path: str = DEFAULT_REPORT,
    arena_keys: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Execute both preregistered arenas and write durable evidence."""

    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError("Rung 16 preregistration requires a visible CUDA GPU")
    prereg_file = Path(prereg_path)
    prereg_bytes = prereg_file.read_bytes()
    prereg_file_sha256 = hashlib.sha256(prereg_bytes).hexdigest()
    preregistration = json.loads(prereg_bytes.decode("utf-8"))
    if preregistration.get("status") != "FROZEN_PRE_REPORT":
        raise ValueError("preregistration must be FROZEN_PRE_REPORT before campaign execution")
    prereg_sha256 = hashlib.sha256(
        json.dumps(preregistration, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    config = RoutingPlaneConfig.from_preregistration(preregistration)
    source_provenance = _source_provenance(prereg_file)
    selected_keys = set(arena_keys or [arena.key for arena in ARENAS])
    unknown = selected_keys - {arena.key for arena in ARENAS}
    if unknown:
        raise ValueError(f"unknown arena keys: {sorted(unknown)}")

    lock_directory = Path(manifest_path).parent
    consumption_directory = prereg_file.parent / "docs"
    selected_arenas = [arena for arena in ARENAS if arena.key in selected_keys]
    for arena in selected_arenas:
        lock_path = lock_directory / f"rung16-{arena.key}-selection-lock-2026-07.json"
        consumption_path = _consumption_path(consumption_directory, prereg_sha256, arena.key)
        if lock_path.exists() or consumption_path.exists():
            raise RuntimeError(
                "Rung 16 REPORT is one-shot across processes; refusing rerun because "
                f"{lock_path if lock_path.exists() else consumption_path} already exists"
            )

    started = time.perf_counter()
    base_model = preregistration["models"]["document_encoder"]
    swap_model = preregistration["models"]["swapped_query_encoder"]
    base_backend = create_embedding_backend(base_model)
    swap_backend = create_embedding_backend(swap_model)
    arena_results = []
    for arena in selected_arenas:
        result = run_arena(
            arena,
            preregistration=preregistration,
            prereg_sha256=prereg_sha256,
            prereg_file_sha256=prereg_file_sha256,
            config=config,
            base_backend=base_backend,
            swap_backend=swap_backend,
            lock_directory=lock_directory,
            consumption_directory=consumption_directory,
            source_provenance=source_provenance,
        )
        arena_results.append(result)

    manifest = {
        "record_type": "rung16_quant_aware_routing_campaign",
        "preregistration_path": str(prereg_file).replace("\\", "/"),
        "preregistration_sha256": prereg_sha256,
        "preregistration_canonical_sha256": prereg_sha256,
        "preregistration_file_sha256": prereg_file_sha256,
        "device": {
            "type": "cuda",
            "name": torch.cuda.get_device_name(0),
        },
        "source_provenance": source_provenance,
        "arenas": arena_results,
        "wall_clock_seconds": time.perf_counter() - started,
    }
    manifest_file = Path(manifest_path)
    manifest_file.parent.mkdir(parents=True, exist_ok=True)
    manifest_file.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(render_report(manifest), encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_file), "report": str(report_file)}, sort_keys=True))
    return manifest


def run_arena(
    arena: ArenaSpec,
    *,
    preregistration: Mapping[str, Any],
    prereg_sha256: str,
    prereg_file_sha256: str,
    config: RoutingPlaneConfig,
    base_backend: Any,
    swap_backend: Any,
    lock_directory: Path,
    consumption_directory: Path,
    source_provenance: Mapping[str, Any],
) -> Dict[str, Any]:
    """Run one ANCHOR→SELECT-lock→one-REPORT arena."""

    started = time.perf_counter()
    arena_data = _load_arena_data(arena, preregistration, config.split_seed)
    strata = arena_data["query_domains"] if arena.cluster_rule == "domain" else None
    split = three_way_seeded_split(
        arena_data["queries"].keys(),
        seed=config.split_seed,
        anchor_fraction=config.anchor_fraction,
        select_fraction=config.select_fraction,
        report_fraction=config.report_fraction,
        strata=strata,
    )

    document_ids = list(arena_data["corpus"].keys())
    document_texts = [arena_data["corpus"][doc_id] for doc_id in document_ids]
    document_embeddings = _embed_base(base_backend, document_texts)
    drift = QueryEncoderDrift(
        store_dim=document_embeddings.shape[1],
        swap_model_name=preregistration["models"]["swapped_query_encoder"],
        seed=int(preregistration["models"]["swap_projection_seed"]),
        swap_backend=swap_backend,
    )
    oracle_document_embeddings = _embed_swap(drift, document_texts)
    drift_manifest = drift.manifest()
    doc_index = {doc_id: index for index, doc_id in enumerate(document_ids)}
    anchor_vectors = _embed_phase_queries(
        drift,
        arena_data["queries"],
        split.anchor_ids,
    )
    anchor_pairs = _build_anchor_centroid_pairs(
        split.anchor_ids,
        qrels=arena_data["qrels"],
        query_vectors=anchor_vectors,
        document_embeddings=document_embeddings,
        doc_index=doc_index,
        query_domains=arena_data["query_domains"],
    )
    if arena.cluster_rule == "kmeans":
        labels, _centroids = cluster_vectors(
            np.asarray([pair["doc_vector"] for pair in anchor_pairs], dtype=np.float32),
            k=config.route_k,
            seed=config.split_seed,
        )
        for index, label in enumerate(labels):
            anchor_pairs[index]["route_key"] = f"cluster:{int(label)}"
    elif arena.cluster_rule == "domain":
        for pair in anchor_pairs:
            pair["route_key"] = f"domain:{pair['domain']}"
    else:
        raise ValueError(f"unsupported cluster rule: {arena.cluster_rule}")

    plane = fit_quant_aware_plane(
        document_ids=document_ids,
        document_embeddings=document_embeddings,
        qrels=arena_data["qrels"],
        split=split,
        anchor_pairs=anchor_pairs,
        config=config,
        preregistration=preregistration,
        oracle_document_embeddings=oracle_document_embeddings,
        device="cuda",
    )
    select_vectors = _embed_phase_queries(
        drift,
        arena_data["queries"],
        split.select_ids,
    )
    select_decision = plane.select(select_vectors)

    # Durable proof that every selection threshold was frozen before the first
    # REPORT call.  This file is written and closed before ``plane.report``.
    lock_path = lock_directory / f"rung16-{arena.key}-selection-lock-2026-07.json"
    lock_payload = {
        "record_type": "rung16_pre_report_selection_lock",
        "arena": arena.key,
        "preregistration_sha256": prereg_sha256,
        "preregistration_canonical_sha256": prereg_sha256,
        "preregistration_file_sha256": prereg_file_sha256,
        "split": split.to_dict(include_ids=False),
        "select_decision": select_decision,
        "source_provenance": dict(source_provenance),
        "report_touched": False,
    }
    _exclusive_write_json(lock_path, lock_payload)

    # REPORT texts are embedded only after the selection lock exists.  Merely
    # forming the split uses IDs; no REPORT vector enters ANCHOR or SELECT.
    report_vectors = _embed_phase_queries(
        drift,
        arena_data["queries"],
        split.report_ids,
    )
    frozen_report = plane.report(report_vectors)
    consumption_path = _consumption_path(consumption_directory, prereg_sha256, arena.key)
    consumption_payload = {
        "record_type": "rung16_report_consumed",
        "arena": arena.key,
        "preregistration_canonical_sha256": prereg_sha256,
        "preregistration_file_sha256": prereg_file_sha256,
        "selection_lock_path": str(lock_path).replace("\\", "/"),
        "selection_lock_sha256": hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        "frozen_report_canonical_sha256": hashlib.sha256(
            json.dumps(frozen_report, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "report_consumed": True,
        "verdict": frozen_report["verdict"],
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _exclusive_write_json(consumption_path, consumption_payload)
    provenance = plane.provenance()
    return {
        "key": arena.key,
        "title": arena.title,
        "datasets": list(arena.datasets),
        "cluster_rule": arena.cluster_rule,
        "corpus_size": len(document_ids),
        "query_count": len(arena_data["queries"]),
        "split": split.to_dict(include_ids=True),
        "drift_manifest": drift_manifest,
        "anchor_pair_count": len(anchor_pairs),
        "route_keys": sorted(key for key in plane.adapters if key != "global"),
        "selection_lock_path": str(lock_path).replace("\\", "/"),
        "report_consumption_path": str(consumption_path).replace("\\", "/"),
        "report_consumption": consumption_payload,
        "select": select_decision,
        "report": frozen_report,
        "provenance": provenance,
        "wall_clock_seconds": time.perf_counter() - started,
    }


def _load_arena_data(
    arena: ArenaSpec,
    preregistration: Mapping[str, Any],
    seed: int,
) -> Dict[str, Any]:
    corpus: Dict[str, str] = {}
    queries: Dict[str, str] = {}
    qrels: Dict[str, Dict[str, float]] = {}
    query_domains: Dict[str, str] = {}
    for dataset in arena.datasets:
        raw_corpus, raw_queries, raw_qrels = load_mteb_data(dataset)
        if raw_corpus is None or raw_queries is None or raw_qrels is None:
            raise RuntimeError(f"failed to load preregistered dataset {dataset}")
        dataset_spec = preregistration["arenas"][arena.key]
        sliced = _seeded_dataset_slice(
            dataset,
            raw_corpus,
            raw_queries,
            raw_qrels,
            max_queries=int(dataset_spec["max_queries_per_dataset"]),
            sample_docs=int(dataset_spec["sample_docs_per_dataset"]),
            seed=seed,
        )
        corpus.update(sliced["corpus"])
        queries.update(sliced["queries"])
        qrels.update(sliced["qrels"])
        query_domains.update({query_id: dataset for query_id in sliced["queries"]})
    return {
        "corpus": corpus,
        "queries": queries,
        "qrels": qrels,
        "query_domains": query_domains,
    }


def _seeded_dataset_slice(
    dataset: str,
    corpus: Mapping[Any, str],
    queries: Mapping[Any, str],
    qrels: Mapping[Any, Mapping[Any, float]],
    *,
    max_queries: int,
    sample_docs: int,
    seed: int,
) -> Dict[str, Any]:
    """Freeze a seeded query sample and a corpus containing every positive."""

    raw_corpus = {canonicalize_id(doc_id): str(text) for doc_id, text in corpus.items()}
    raw_queries = {canonicalize_id(query_id): str(text) for query_id, text in queries.items()}
    normalized_qrels = {
        canonicalize_id(query_id): {
            canonicalize_id(doc_id): float(score)
            for doc_id, score in relevance.items()
            if float(score) > 0.0 and canonicalize_id(doc_id) in raw_corpus
        }
        for query_id, relevance in qrels.items()
    }
    eligible = sorted(
        query_id
        for query_id in raw_queries
        if normalized_qrels.get(query_id)
    )
    stable_seed = int(seed) + int(hashlib.sha256(dataset.encode("utf-8")).hexdigest()[:8], 16)
    rng = random.Random(stable_seed)
    rng.shuffle(eligible)
    selected_query_ids = sorted(eligible[: min(max_queries, len(eligible))])
    required_docs = {
        doc_id
        for query_id in selected_query_ids
        for doc_id in normalized_qrels[query_id]
    }
    remaining = sorted(set(raw_corpus) - required_docs)
    rng.shuffle(remaining)
    extra_count = max(0, int(sample_docs) - len(required_docs))
    selected_doc_ids = sorted(required_docs | set(remaining[:extra_count]))

    prefix = f"{dataset}::"
    prefixed_corpus = {prefix + doc_id: raw_corpus[doc_id] for doc_id in selected_doc_ids}
    prefixed_queries = {prefix + query_id: raw_queries[query_id] for query_id in selected_query_ids}
    prefixed_qrels = {
        prefix + query_id: {
            prefix + doc_id: float(score)
            for doc_id, score in normalized_qrels[query_id].items()
            if doc_id in required_docs
        }
        for query_id in selected_query_ids
    }
    return {
        "corpus": prefixed_corpus,
        "queries": prefixed_queries,
        "qrels": prefixed_qrels,
    }


def _build_anchor_centroid_pairs(
    anchor_ids: Sequence[str],
    *,
    qrels: Mapping[str, Mapping[str, float]],
    query_vectors: Mapping[str, np.ndarray],
    document_embeddings: np.ndarray,
    doc_index: Mapping[str, int],
    query_domains: Mapping[str, str],
) -> list[Dict[str, Any]]:
    pairs = []
    for query_id in anchor_ids:
        relevance = qrels[query_id]
        positive_ids = [doc_id for doc_id in sorted(relevance) if doc_id in doc_index]
        if not positive_ids:
            continue
        weights = np.asarray([float(relevance[doc_id]) for doc_id in positive_ids], dtype=np.float32)
        positive_vectors = np.asarray([document_embeddings[doc_index[doc_id]] for doc_id in positive_ids])
        centroid = np.average(positive_vectors, axis=0, weights=weights)
        pairs.append(
            {
                "query_id": query_id,
                "doc_id": f"positive-centroid::{query_id}",
                "source_doc_ids": positive_ids,
                "doc_vector": np.asarray(centroid, dtype=np.float32),
                "query_vector": np.asarray(query_vectors[query_id], dtype=np.float32),
                "domain": query_domains[query_id],
                "route_key": "unassigned",
            }
        )
    return pairs


def _embed_base(backend: Any, texts: Sequence[str], batch_size: int = 256) -> np.ndarray:
    chunks = []
    for start in range(0, len(texts), batch_size):
        chunks.append(np.asarray(backend.embed_raw(list(texts[start : start + batch_size])), dtype=np.float32))
    return np.concatenate(chunks, axis=0)


def _embed_swap(drift: QueryEncoderDrift, texts: Sequence[str], batch_size: int = 128) -> np.ndarray:
    chunks = []
    for start in range(0, len(texts), batch_size):
        chunks.append(np.asarray(drift.embed_queries(list(texts[start : start + batch_size])), dtype=np.float32))
    return np.concatenate(chunks, axis=0)


def _embed_phase_queries(
    drift: QueryEncoderDrift,
    queries: Mapping[str, str],
    query_ids: Sequence[str],
) -> Dict[str, np.ndarray]:
    """Embed exactly one frozen phase, preserving its canonical ID order."""

    ordered_ids = [canonicalize_id(query_id) for query_id in query_ids]
    matrix = _embed_swap(drift, [queries[query_id] for query_id in ordered_ids])
    return {
        query_id: matrix[index]
        for index, query_id in enumerate(ordered_ids)
    }


def render_report(manifest: Mapping[str, Any]) -> str:
    lines = [
        "# Rung 16 — Quant-Aware Routing Results (July 2026)",
        "",
        f"Preregistration: `{manifest['preregistration_path']}`.",
        f"File SHA256: `{manifest['preregistration_file_sha256']}`; canonical JSON SHA256: "
        f"`{manifest['preregistration_canonical_sha256']}`.",
        f"GPU: `{manifest['device']['name']}`.",
        "",
        "Promotion was decided on SELECT only. REPORT CIs below are descriptive and cannot rescue a failed SELECT gate.",
        "",
    ]
    for arena in manifest["arenas"]:
        select = arena["select"]
        report = arena["report"]
        lines.extend(
            [
                f"## {arena['title']}",
                "",
                f"Verdict: **{report['verdict']}**. Corpus {arena['corpus_size']}; queries {arena['query_count']}; "
                f"split {arena['split']['anchor_count']}/{arena['split']['select_count']}/{arena['split']['report_count']}.",
                "",
                "### Frozen SELECT promotion gate",
                "",
                f"- Plane vs single-global delta: {select['paired_bootstrap_ci']['estimate']:.6f} NDCG",
                f"- Paired 95% CI: [{select['paired_bootstrap_ci']['lower']:.6f}, {select['paired_bootstrap_ci']['upper']:.6f}]",
                f"- Required lower bound: > {select['min_lift']:.6f}",
                f"- Quant pass-rate: {select['quant_pass_rate']:.3f}",
                f"- No-route floor passed: {select['floor_passed']}",
                f"- SELECT gate passed: **{select['select_gate_passed']}**",
                "",
                "Per-used-adapter SELECT quant gates (baseline is no-route on the same routed-query subset):",
                "",
                _quant_table(select["quant_decisions"]),
                "",
                "SELECT route usage:",
                "",
                _usage_table(select["usage"]),
                "",
                "### One frozen REPORT evaluation",
                "",
                f"- Plane FP32 NDCG: {report['metrics']['plane_fp32']['mean_ndcg']:.6f}",
                f"- Plane INT8-sim NDCG: {report['metrics']['plane_quantized']['mean_ndcg']:.6f}",
                f"- Single-global NDCG: {report['metrics']['single_global_fp32']['mean_ndcg']:.6f}",
                f"- No-route NDCG: {report['metrics']['no_route']['mean_ndcg']:.6f}",
                f"- C2O oracle NDCG: {report['metrics']['c2o_oracle']['mean_ndcg']:.6f}",
                f"- Single-best-route NDCG: {_optional_metric(report['metrics']['single_best_route'])}",
                f"- Plane − single-global: {report['plane_vs_single_global_delta']:.6f}",
                f"- Descriptive paired 95% CI: [{report['paired_bootstrap_ci']['lower']:.6f}, {report['paired_bootstrap_ci']['upper']:.6f}]",
                f"- Multi-route binding passed: {report['multi_route_binding']['passed']} "
                f"(qualifying: {report['multi_route_binding']['qualifying_routes']})",
                "",
                "REPORT route usage:",
                "",
                _usage_table(report["usage"]),
                "",
            ]
        )
    return "\n".join(lines)


def _usage_table(usage: Mapping[str, Any]) -> str:
    rows = [
        "| Route | Count | p_k |",
        "|---|---:|---:|",
    ]
    for key, count in usage["histogram"].items():
        rows.append(f"| {key} | {count} | {usage['p_k'][key]:.4f} |")
    rows.append(f"\nEntropy (nats): {usage['entropy_nats']:.6f}; n_used: {usage['n_used']}.")
    return "\n".join(rows)


def _quant_table(decisions: Mapping[str, Mapping[str, Any]]) -> str:
    rows = [
        "| Adapter | Queries | No-route | FP32 | INT8-sim | Retained | Pass | Reasons |",
        "|---|---:|---:|---:|---:|---:|:---:|---|",
    ]
    for key, decision in decisions.items():
        rows.append(
            f"| {key} | {decision['query_count']} | {decision['baseline_fitness']:.6f} | "
            f"{decision['fp32_fitness']:.6f} | {decision['quantized_fitness']:.6f} | "
            f"{decision['retained_gain_ratio']:.4f} | {decision['passed']} | "
            f"{', '.join(decision['reasons']) or 'none'} |"
        )
    return "\n".join(rows)


def _optional_metric(metric: Optional[Mapping[str, Any]]) -> str:
    return "n/a" if metric is None else f"{float(metric['mean_ndcg']):.6f}"


def _consumption_path(directory: Path, prereg_sha256: str, arena_key: str) -> Path:
    return directory / f"rung16-{arena_key}-report-consumed-{prereg_sha256[:12]}.json"


def _source_provenance(prereg_file: Path) -> Dict[str, Any]:
    """Record exact implementation bytes and Git state before REPORT.

    Git capture failures propagate intentionally: a campaign without an exact
    commit, complete tracked-tree identity, and clean tracked worktree is not
    valid promotion evidence.
    """

    source_root = Path(__file__).resolve().parent
    repo_root_result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        cwd=source_root,
        check=True,
        capture_output=True,
        text=True,
    )
    repo_root = Path(repo_root_result.stdout.strip()).resolve()
    source_paths = (
        source_root / "adapter_router.py",
        source_root / "quant_aware_routing.py",
        source_root / "antigravity_engine.py",
        Path(__file__).resolve(),
        prereg_file.resolve(),
    )
    try:
        relative_paths = tuple(path.relative_to(repo_root) for path in source_paths)
    except ValueError as exc:
        raise RuntimeError("all provenance inputs must live inside the Git repository") from exc
    subprocess.run(
        [
            "git",
            "ls-files",
            "--error-unmatch",
            "--",
            *[str(path).replace("\\", "/") for path in relative_paths],
        ],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    tracked_files_output = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    tracked_files = tuple(path for path in tracked_files_output.split("\0") if path)
    status_command = ["git", "status", "--porcelain=v1", "--untracked-files=no"]
    status_before = subprocess.run(
        status_command,
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    if status_before:
        raise RuntimeError(
            "tracked worktree must be clean before provenance capture: "
            + "; ".join(status_before)
        )
    hashes = {
        str(relative_path).replace("\\", "/"): hashlib.sha256(path.read_bytes()).hexdigest()
        for path, relative_path in zip(source_paths, relative_paths)
    }
    status_after = subprocess.run(
        status_command,
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    head_after = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if status_after:
        raise RuntimeError(
            "tracked worktree changed during provenance capture: "
            + "; ".join(status_after)
        )
    if head_after != head:
        raise RuntimeError("Git HEAD changed during provenance capture")
    tracked_file_list_sha256 = hashlib.sha256(
        "\0".join(tracked_files).encode("utf-8")
    ).hexdigest()
    return {
        "captured_before_report": True,
        "git_repo_root": ".",
        "git_head": head,
        "git_tree": tree,
        "tracked_worktree_scope": "entire_repository",
        "tracked_worktree_clean": True,
        "tracked_source_clean": True,
        "git_status_porcelain": [],
        "tracked_file_count": len(tracked_files),
        "tracked_file_list_sha256": tracked_file_list_sha256,
        "sha256": hashes,
    }


def _exclusive_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Create an evidence file exactly once; never truncate an existing lock."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Rung 16 quant-aware routing campaign")
    parser.add_argument("--prereg", default=DEFAULT_PREREG)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument(
        "--arena",
        choices=("all", "arena_a_default_swap", "arena_b_multi_domain"),
        default="all",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    arena_keys = None if args.arena == "all" else [args.arena]
    run_campaign(
        prereg_path=args.prereg,
        manifest_path=args.manifest,
        report_path=args.report,
        arena_keys=arena_keys,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
