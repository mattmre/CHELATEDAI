"""Phase C four-candidate evaluation campaign.

Runs a full comparative evaluation across BEIR small-tier datasets comparing:
  1. baseline               – plain cosine, no chelation
  2. guard_p85_t0.01        – chelation with guardrail (p=85, threshold=0.01)
  3. learned_reform_gate_v1 – chelation + learned query-reformulation gate
  4. learned_mask_gate_v1   – chelation + learned static-dimension-mask gate

Both learned gates are trained from real experiment artifacts at runtime when not
supplied via CLI flags. The script requires BEIR datasets to be accessible via
HuggingFace (network required on first run; datasets are locally cached afterwards).

Outputs (all in --output-dir):
  phase_c_results.json          – full results with per-query rows + summaries
  overlay_card_<candidate>.json – overlay artifact card per candidate (Slice 5)
  run_manifest.json             – artifact inventory

Usage:
  python run_phase_c_eval.py
  python run_phase_c_eval.py --tier small --max-queries 200
  python run_phase_c_eval.py --reform-gate path/gate.json --mask-gate path/gate.json
"""

from __future__ import annotations

import argparse
import datetime
import glob
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from adaptive_overlay import build_overlay_artifact_card, build_overlay_report
from benchmark_comparative import mean_average_precision_at_k, mean_reciprocal_rank, recall_at_k
from benchmark_utils import (
    canonicalize_id,
    isolated_adapter_state,
    load_mteb_data,
    map_predicted_ids,
    ndcg_at_k,
)
from chelation_logger import get_logger
from learned_mask_gate import (
    _classifier_score as _mask_classifier_score,
    filter_mask_rows,
    load_mask_example_rows,
    train_mask_gate,
)
from learned_reformulation_gate import predict_single, train_gate_from_pool
from query_reformulator import QueryReformulator, query_lexical_features

_LOGGER = get_logger()

PHASE_C_SCHEMA_VERSION = 1
PHASE_C_CANDIDATES = [
    "baseline",
    "guard_p85_t0.01",
    "learned_reform_gate_v1",
    "learned_mask_gate_v1",
]

# Default resource paths
_DEFAULT_ATTRIBUTION_POOL = "experiment_runs/attribution-pool/latest/attribution_pool.json"
_DEFAULT_MASK_SCOPE_GLOB = (
    "experiment_runs/golden-default-autopilot-engine-scope-debug3/mask/*.json"
)
DEFAULT_OUTPUT_DIR = "experiment_runs/phase-c-eval/latest"
DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
_MINILM_N_DIMS = 384


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class PhaseCResult:
    """Per-query result for one candidate configuration."""

    candidate_id: str
    dataset: str
    query_id: str
    query_text: str
    ndcg_at_10: float
    map_at_10: float
    mrr: float
    recall_at_10: float
    latency_ms: float
    gate_applied: bool = False
    reformulated: bool = False
    mask_applied: bool = False


@dataclass
class PhaseCCandidateSummary:
    """Aggregate summary for one candidate × dataset combination."""

    candidate_id: str
    dataset: str
    mean_ndcg_at_10: float
    mean_map_at_10: float
    mean_mrr: float
    mean_recall_at_10: float
    mean_latency_ms: float
    num_queries: int
    gate_applied_count: int


# ---------------------------------------------------------------------------
# Gate loading / training helpers
# ---------------------------------------------------------------------------


def _build_mask_vector(masked_dims: List[int], n_dims: int) -> np.ndarray:
    """Return a binary float mask with 0.0 at masked_dims, 1.0 elsewhere."""
    mask = np.ones(n_dims, dtype=float)
    for dim in masked_dims:
        if 0 <= dim < n_dims:
            mask[dim] = 0.0
    return mask


def _load_or_train_reform_gate(path: Optional[str]) -> Optional[Dict[str, Any]]:
    """Load a pre-trained reform gate config or train one from the attribution pool."""
    if path and os.path.isfile(path):
        _LOGGER.log_event("phase_c_reform_gate_load", f"Loading reform gate: {path}", path=path)
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)

    if not os.path.isfile(_DEFAULT_ATTRIBUTION_POOL):
        _LOGGER.log_event(
            "phase_c_reform_gate_skip",
            "Attribution pool not found; learned_reform_gate_v1 will be fail-closed",
            path=_DEFAULT_ATTRIBUTION_POOL,
        )
        return None

    _LOGGER.log_event(
        "phase_c_reform_gate_train",
        f"Training reform gate from: {_DEFAULT_ATTRIBUTION_POOL}",
    )
    with open(_DEFAULT_ATTRIBUTION_POOL, encoding="utf-8") as fh:
        pool = json.load(fh)
    return train_gate_from_pool(pool, positive_action="REFORMULATE")


def _load_or_train_mask_gate_and_vector(
    mask_gate_path: Optional[str],
    mask_vector_path: Optional[str],
) -> Tuple[Optional[Dict[str, Any]], Optional[np.ndarray]]:
    """Load or train the mask gate classifier and derive the mask vector."""
    gate_config: Optional[Dict[str, Any]] = None
    mask_vector: Optional[np.ndarray] = None

    # --- gate classifier ---
    if mask_gate_path and os.path.isfile(mask_gate_path):
        _LOGGER.log_event("phase_c_mask_gate_load", f"Loading mask gate: {mask_gate_path}")
        with open(mask_gate_path, encoding="utf-8") as fh:
            gate_config = json.load(fh)
    else:
        probe_paths = sorted(glob.glob(_DEFAULT_MASK_SCOPE_GLOB))
        if probe_paths:
            _LOGGER.log_event(
                "phase_c_mask_gate_train",
                f"Training mask gate from {len(probe_paths)} scope files",
            )
            try:
                rows = load_mask_example_rows(probe_paths)
                filtered = filter_mask_rows(rows)
                if filtered:
                    gate_config = train_mask_gate(
                        filtered,
                        max_negative_examples=5,
                        min_positive_examples=2,
                    )
                    if gate_config.get("gate") is None:
                        _LOGGER.log_event(
                            "phase_c_mask_gate_fail_closed",
                            "Mask gate fail-closed; candidate uses no masking",
                        )
                        gate_config = None
            except Exception as exc:
                _LOGGER.log_event("phase_c_mask_gate_error", str(exc))
                gate_config = None

    # --- mask vector ---
    if mask_vector_path and os.path.isfile(mask_vector_path):
        with open(mask_vector_path, encoding="utf-8") as fh:
            data = json.load(fh)
        masked_dims = data.get("masked_dims", [])
        if masked_dims:
            mask_vector = _build_mask_vector(masked_dims, _MINILM_N_DIMS)
    else:
        probe_paths = sorted(glob.glob(_DEFAULT_MASK_SCOPE_GLOB))
        if probe_paths:
            with open(probe_paths[0], encoding="utf-8") as fh:
                probe_data = json.load(fh)
            masked_dims = (probe_data.get("mask") or {}).get("masked_dims", [])
            if masked_dims:
                mask_vector = _build_mask_vector(masked_dims, _MINILM_N_DIMS)

    return gate_config, mask_vector


# ---------------------------------------------------------------------------
# Per-query gate helpers
# ---------------------------------------------------------------------------


def _predict_mask_gate(row_features: Dict[str, Any], gate_config: Dict[str, Any]) -> bool:
    """Return True if the mask gate classifier predicts masking is beneficial."""
    gate = gate_config.get("gate")
    if gate is None:
        return False
    score = _mask_classifier_score(row_features, gate)
    return score >= float(gate.get("threshold", 1.0))


def _reformulate_query(query_text: str, reformulator: QueryReformulator) -> str:
    """Apply query reformulation; prefer stopword_removed variant."""
    try:
        variants = reformulator.reformulate(query_text, max_variants=3)
        for variant in variants:
            if variant.strategy == "stopword_removed":
                return variant.text
        return query_text
    except Exception:
        return query_text


def _mask_gate_features(query_text: str) -> Dict[str, Any]:
    """Build feature dict from query text for mask gate prediction."""
    lexical = query_lexical_features(query_text)
    return {
        "query_token_count": lexical.get("token_count", 0),
        "query_char_count": lexical.get("char_count", len(query_text)),
        "query_stopword_ratio": lexical.get("stopword_ratio", 0.0),
        "query_numeric_token_count": lexical.get("numeric_token_count", 0),
        "query_negation_count": lexical.get("negation_count", 0),
        "query_claim_cue_count": lexical.get("claim_cue_count", 0),
        "query_text": query_text,
        # delta unknown at prediction time; zero is safe (feature not in MASK_GATE_FEATURES)
        "delta_ndcg_at_10": 0.0,
    }


# ---------------------------------------------------------------------------
# Per-candidate evaluation
# ---------------------------------------------------------------------------


def evaluate_phase_c_candidate(
    candidate_id: str,
    corpus: Dict[str, str],
    queries: Dict[str, str],
    qrels: Dict[str, Any],
    *,
    dataset: str,
    reform_gate_config: Optional[Dict[str, Any]] = None,
    mask_gate_config: Optional[Dict[str, Any]] = None,
    mask_vector: Optional[np.ndarray] = None,
    max_queries: Optional[int] = None,
    model_name: str = DEFAULT_MODEL_NAME,
) -> List[PhaseCResult]:
    """Evaluate one Phase C candidate against a BEIR dataset.

    Args:
        candidate_id: One of PHASE_C_CANDIDATES.
        corpus: {doc_id: text} mapping.
        queries: {query_id: text} mapping.
        qrels: {query_id: {doc_id: relevance}} mapping.
        dataset: Dataset name (for result tagging).
        reform_gate_config: Trained reform gate config dict (or None = fail-closed).
        mask_gate_config: Trained mask gate config dict (or None = no masking).
        mask_vector: numpy mask vector (0.0 = masked, 1.0 = kept).
        max_queries: Optional per-dataset query limit.
        model_name: Embedding model identifier.

    Returns:
        List of PhaseCResult, one per evaluated query.
    """
    from antigravity_engine import AntigravityEngine

    query_items = list(queries.items())
    if max_queries is not None:
        query_items = query_items[:max_queries]

    reformulator = QueryReformulator() if candidate_id == "learned_reform_gate_v1" else None

    results: List[PhaseCResult] = []

    with isolated_adapter_state():
        use_centering = candidate_id != "baseline"
        engine = AntigravityEngine(
            qdrant_location=":memory:",
            chelation_p=85,
            model_name=model_name,
            use_centering=use_centering,
            use_quantization=False,
            store_full_text_payload=False,
        )
        # guard_p85_t0.01 uses the default threshold (0.01); set explicitly for clarity
        if candidate_id == "guard_p85_t0.01":
            engine.chelation_threshold = 0.01

        doc_ids = list(corpus.keys())
        doc_texts = [corpus[doc_id] for doc_id in doc_ids]
        doc_payloads = [{"doc_id": canonicalize_id(doc_id)} for doc_id in doc_ids]
        engine.ingest(doc_texts, doc_payloads)

        identity_mask = np.ones(engine.vector_size, dtype=float)

        try:
            for qid, qtext in query_items:
                rel_docs = qrels.get(str(qid), qrels.get(qid, {}))
                relevant_set = {str(d) for d, s in rel_docs.items() if s > 0}

                gate_applied = False
                reformulated = False
                mask_applied = False
                effective_query = qtext

                if candidate_id == "learned_reform_gate_v1" and reform_gate_config is not None:
                    features = query_lexical_features(qtext)
                    if predict_single(features, reform_gate_config):
                        effective_query = _reformulate_query(qtext, reformulator)
                        reformulated = True
                        gate_applied = True

                elif (
                    candidate_id == "learned_mask_gate_v1"
                    and mask_gate_config is not None
                    and mask_vector is not None
                ):
                    row_features = _mask_gate_features(qtext)
                    if _predict_mask_gate(row_features, mask_gate_config):
                        engine.set_static_dimension_mask(mask_vector)
                        mask_applied = True
                        gate_applied = True
                    else:
                        engine.set_static_dimension_mask(identity_mask)

                start = time.perf_counter()
                _std_top, chel_top, _mask_arr, _jaccard = engine.run_inference(effective_query)
                elapsed = (time.perf_counter() - start) * 1000

                retrieved = map_predicted_ids(engine, chel_top[:10])
                r = [1 if d in relevant_set else 0 for d in retrieved]

                results.append(
                    PhaseCResult(
                        candidate_id=candidate_id,
                        dataset=dataset,
                        query_id=str(qid),
                        query_text=qtext,
                        ndcg_at_10=ndcg_at_k(r, 10),
                        map_at_10=mean_average_precision_at_k(retrieved, relevant_set, 10),
                        mrr=mean_reciprocal_rank(retrieved, relevant_set),
                        recall_at_10=recall_at_k(retrieved, relevant_set, 10),
                        latency_ms=elapsed,
                        gate_applied=gate_applied,
                        reformulated=reformulated,
                        mask_applied=mask_applied,
                    )
                )
        finally:
            if hasattr(engine, "close"):
                engine.close()

    return results


# ---------------------------------------------------------------------------
# Aggregation + overlay
# ---------------------------------------------------------------------------


def summarize_candidate_results(results: List[PhaseCResult]) -> PhaseCCandidateSummary:
    """Aggregate per-query PhaseCResult rows into a candidate summary."""
    if not results:
        raise ValueError("Cannot summarize empty results list")
    return PhaseCCandidateSummary(
        candidate_id=results[0].candidate_id,
        dataset=results[0].dataset,
        mean_ndcg_at_10=float(np.mean([r.ndcg_at_10 for r in results])),
        mean_map_at_10=float(np.mean([r.map_at_10 for r in results])),
        mean_mrr=float(np.mean([r.mrr for r in results])),
        mean_recall_at_10=float(np.mean([r.recall_at_10 for r in results])),
        mean_latency_ms=float(np.mean([r.latency_ms for r in results])),
        num_queries=len(results),
        gate_applied_count=sum(1 for r in results if r.gate_applied),
    )


def build_candidate_overlay_rows(
    candidate_results: List[PhaseCResult],
    baseline_results: List[PhaseCResult],
) -> List[Dict[str, Any]]:
    """Build engine-scope-style overlay rows for overlay artifact construction.

    Each row contains the per-query delta_ndcg_at_10 against the baseline,
    plus enough metadata for adaptive_overlay.build_channel_variation_records.
    """
    baseline_by_qid = {r.query_id: r for r in baseline_results}
    rows: List[Dict[str, Any]] = []
    for result in candidate_results:
        baseline = baseline_by_qid.get(result.query_id)
        delta = (
            result.ndcg_at_10 - baseline.ndcg_at_10
            if baseline is not None
            else None
        )
        if result.reformulated:
            action = "REFORMULATE"
        elif result.mask_applied:
            action = "MASK"
        else:
            action = "FAST"
        rows.append(
            {
                "row_type": "phase_c_eval",
                "task": result.dataset,
                "query_id": result.query_id,
                "profile": result.candidate_id,
                "action": action,
                "delta_ndcg_at_10": delta,
                "gate_applied": result.gate_applied,
                "source_family": "phase_c_eval",
                "fault_class": (
                    "actuator_active_positive"
                    if delta is not None and delta > 0.001
                    else "actuator_active_negative"
                    if delta is not None and delta < -0.001
                    else "actuator_active_neutral"
                ),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# Overlay artifact cards (Slice 5)
# ---------------------------------------------------------------------------


def _build_and_write_overlay_cards(
    all_results: List[PhaseCResult],
    candidates: List[str],
    output_dir: Path,
    run_at: str,
) -> None:
    """Build and write overlay artifact cards for each candidate.

    Implements Slice 5: every candidate gets a fully-wired overlay artifact card
    populated with real per-query delta data, not placeholder values.
    """
    by_candidate: Dict[str, List[PhaseCResult]] = {}
    for result in all_results:
        by_candidate.setdefault(result.candidate_id, []).append(result)

    baseline_results = by_candidate.get("baseline", [])

    for candidate_id, candidate_results in by_candidate.items():
        overlay_rows = build_candidate_overlay_rows(
            candidate_results,
            baseline_results if candidate_id != "baseline" else candidate_results,
        )
        if not overlay_rows:
            continue

        try:
            overlay_report = build_overlay_report(overlay_rows)
            card = build_overlay_artifact_card(
                candidate_id=candidate_id,
                overlay_report=overlay_report,
                purpose=f"Phase C four-candidate evaluation: {candidate_id}",
                metadata={
                    "run_at": run_at,
                    "candidate_id": candidate_id,
                    "query_count": len(candidate_results),
                    "data_source": "phase_c_eval",
                    "schema_version": PHASE_C_SCHEMA_VERSION,
                },
            )
            card_path = output_dir / f"overlay_card_{candidate_id}.json"
            with open(card_path, "w", encoding="utf-8") as fh:
                json.dump(card, fh, indent=2)
            _LOGGER.log_event(
                "phase_c_overlay_card_written",
                f"Overlay card written: {card_path}",
                candidate=candidate_id,
            )
        except Exception as exc:
            _LOGGER.log_event(
                "phase_c_overlay_card_error",
                f"Failed to write overlay card for {candidate_id}: {exc}",
                candidate=candidate_id,
            )


# ---------------------------------------------------------------------------
# Main evaluation runner
# ---------------------------------------------------------------------------


def run_phase_c_eval(
    datasets: List[str],
    candidates: List[str] = PHASE_C_CANDIDATES,
    max_queries: Optional[int] = None,
    model_name: str = DEFAULT_MODEL_NAME,
    output_dir: str = DEFAULT_OUTPUT_DIR,
    reform_gate_path: Optional[str] = None,
    mask_gate_path: Optional[str] = None,
    mask_vector_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Run Phase C four-candidate evaluation campaign.

    Args:
        datasets: List of BEIR dataset names (e.g. ["SciFact", "NFCorpus"]).
        candidates: Candidate IDs to evaluate; defaults to all four.
        max_queries: Optional per-dataset query limit (for fast smoke runs).
        model_name: Embedding model identifier.
        output_dir: Directory to write all output artifacts.
        reform_gate_path: Path to pre-trained reform gate JSON (optional).
        mask_gate_path: Path to pre-trained mask gate JSON (optional).
        mask_vector_path: Path to JSON with a ``masked_dims`` key (optional).

    Returns:
        Full results dict (also written to ``output_dir/phase_c_results.json``).
    """
    run_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    _LOGGER.log_event(
        "phase_c_start",
        f"Phase C eval: {len(candidates)} candidates × {len(datasets)} datasets",
        candidates=candidates,
        datasets=datasets,
    )

    reform_gate_config = _load_or_train_reform_gate(reform_gate_path)
    mask_gate_config, mask_vector = _load_or_train_mask_gate_and_vector(
        mask_gate_path, mask_vector_path
    )

    gate_summary: Dict[str, Any] = {
        "reform_gate_present": (
            reform_gate_config is not None and reform_gate_config.get("gate") is not None
        ),
        "mask_gate_present": (
            mask_gate_config is not None and mask_gate_config.get("gate") is not None
        ),
        "mask_vector_present": mask_vector is not None,
    }
    _LOGGER.log_event("phase_c_gates", "Gate readiness", **gate_summary)

    all_results: List[PhaseCResult] = []
    all_summaries: List[PhaseCCandidateSummary] = []

    for dataset in datasets:
        _LOGGER.log_event("phase_c_dataset_load", f"Loading dataset: {dataset}", dataset=dataset)
        corpus, queries, qrels = load_mteb_data(dataset)

        for candidate_id in candidates:
            _LOGGER.log_event(
                "phase_c_candidate_start",
                f"Evaluating {candidate_id} on {dataset}",
                candidate=candidate_id,
                dataset=dataset,
            )
            candidate_results = evaluate_phase_c_candidate(
                candidate_id,
                corpus=corpus,
                queries=queries,
                qrels=qrels,
                dataset=dataset,
                reform_gate_config=reform_gate_config,
                mask_gate_config=mask_gate_config,
                mask_vector=mask_vector,
                max_queries=max_queries,
                model_name=model_name,
            )
            summary = summarize_candidate_results(candidate_results)
            all_results.extend(candidate_results)
            all_summaries.append(summary)
            _LOGGER.log_event(
                "phase_c_candidate_done",
                f"{candidate_id} on {dataset}: NDCG@10={summary.mean_ndcg_at_10:.4f}",
                candidate=candidate_id,
                dataset=dataset,
                ndcg=summary.mean_ndcg_at_10,
                gate_applied_count=summary.gate_applied_count,
            )

    # Assemble output
    summaries_nested: Dict[str, Dict[str, Any]] = {}
    for s in all_summaries:
        summaries_nested.setdefault(s.candidate_id, {})[s.dataset] = asdict(s)

    results_doc: Dict[str, Any] = {
        "schema_version": PHASE_C_SCHEMA_VERSION,
        "record_type": "phase_c_eval_results",
        "run_at": run_at,
        "candidates": candidates,
        "datasets": datasets,
        "max_queries_per_dataset": max_queries,
        "gate_summary": gate_summary,
        "summaries": summaries_nested,
        "per_query_results": [asdict(r) for r in all_results],
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    results_file = out / "phase_c_results.json"
    with open(results_file, "w", encoding="utf-8") as fh:
        json.dump(results_doc, fh, indent=2)
    _LOGGER.log_event("phase_c_results_written", f"Results written: {results_file}")

    # Build overlay artifact cards for each candidate (Slice 5)
    _build_and_write_overlay_cards(all_results, candidates, out, run_at)

    manifest: Dict[str, Any] = {
        "schema_version": PHASE_C_SCHEMA_VERSION,
        "record_type": "phase_c_run_manifest",
        "run_at": run_at,
        "candidates": candidates,
        "datasets": datasets,
        "artifacts": sorted(str(p.name) for p in out.glob("*.json")),
    }
    with open(out / "run_manifest.json", "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    _LOGGER.log_event("phase_c_done", "Phase C evaluation complete", output_dir=str(out))
    return results_doc


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase C four-candidate evaluation campaign")
    parser.add_argument(
        "--tier",
        choices=["minimal", "small"],
        default="small",
        help="Dataset tier: minimal=SciFact only, small=SciFact+NFCorpus",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Maximum queries per dataset per candidate (omit for full evaluation)",
    )
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Directory to write evaluation artifacts",
    )
    parser.add_argument(
        "--reform-gate",
        default=None,
        help="Path to pre-trained reform gate JSON (trains from attribution pool if omitted)",
    )
    parser.add_argument(
        "--mask-gate",
        default=None,
        help="Path to pre-trained mask gate JSON (trains from engine scope files if omitted)",
    )
    parser.add_argument(
        "--mask-vector",
        default=None,
        help="Path to JSON with masked_dims key (derived from probe artifacts if omitted)",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL_NAME, help="Embedding model name")
    args = parser.parse_args()

    datasets = ["SciFact"] if args.tier == "minimal" else ["SciFact", "NFCorpus"]

    results = run_phase_c_eval(
        datasets=datasets,
        candidates=PHASE_C_CANDIDATES,
        max_queries=args.max_queries,
        model_name=args.model,
        output_dir=args.output_dir,
        reform_gate_path=args.reform_gate,
        mask_gate_path=args.mask_gate,
        mask_vector_path=args.mask_vector,
    )

    # Summary table
    print(f"\nPhase C Evaluation — {results['run_at']}")
    print(f"Datasets : {', '.join(results['datasets'])}")
    gs = results["gate_summary"]
    print(
        f"Gates    : reform={gs['reform_gate_present']}, "
        f"mask={gs['mask_gate_present']}, "
        f"mask_vector={gs['mask_vector_present']}"
    )
    print()
    header = f"{'Candidate':<30} {'Dataset':<12} {'NDCG@10':>7}  {'MRR':>7}  {'n':>5}  gate%"
    print(header)
    print("-" * len(header))
    summaries = results.get("summaries", {})
    for candidate_id in PHASE_C_CANDIDATES:
        cand_summaries = summaries.get(candidate_id, {})
        for dataset, s in cand_summaries.items():
            pct = (
                f"{s['gate_applied_count'] / s['num_queries']:.0%}"
                if s["num_queries"] > 0
                else "n/a"
            )
            print(
                f"{candidate_id:<30} {dataset:<12} "
                f"{s['mean_ndcg_at_10']:>7.4f}  "
                f"{s['mean_mrr']:>7.4f}  "
                f"{s['num_queries']:>5}  "
                f"{pct}"
            )

    print(f"\nArtifacts written to: {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
