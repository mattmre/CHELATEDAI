# benchmark_rlm.py
"""
Benchmark: Standard Retrieval vs Recursive Retrieval (RLM)

Compares NDCG@10, retrieval call overhead, and latency per query between
standard AntigravityEngine retrieval and RecursiveRetrievalEngine with
configurable decomposer (MockDecomposer or OllamaDecomposer).
"""

import time
import argparse
import numpy as np
import mteb
from pathlib import Path
from typing import Dict, List, Any

from config import ChelationConfig
from antigravity_engine import AntigravityEngine
from recursive_decomposer import RecursiveRetrievalEngine, MockDecomposer, OllamaDecomposer

# Import shared utilities from benchmark_utils
from benchmark_utils import (
    dcg_at_k as _dcg_at_k,
    ndcg_at_k as _ndcg_at_k,
    find_keys as _find_keys,
    find_payload as _find_payload,
    load_mteb_data as _load_mteb_data,
)


# =============================================================================
# Backward Compatibility: Re-export shared functions
# These names are kept to maintain compatibility with existing tests/imports
# =============================================================================

def dcg_at_k(r, k):
    """Discounted Cumulative Gain at rank k."""
    return _dcg_at_k(r, k)


def ndcg_at_k(r, k):
    """Normalized Discounted Cumulative Gain at rank k."""
    return _ndcg_at_k(r, k)


def find_keys(obj, target_keys):
    """Recursively search a nested dict for a level containing all target_keys."""
    return _find_keys(obj, target_keys)


def find_payload(obj, key):
    """Recursively search a nested dict for a specific key and return its value."""
    return _find_payload(obj, key)


def load_mteb_data(task_name: str):
    """
    Load corpus, queries, and qrels from an MTEB retrieval task.

    Returns:
        (corpus, queries, qrels) dicts
    """
    return _load_mteb_data(task_name)


# =============================================================================
# ID Mapping Helper
# =============================================================================

def map_predicted_ids(engine, pred_ids):
    """
    Map Qdrant internal IDs back to original document IDs via payload lookup.

    Args:
        engine: AntigravityEngine with qdrant client
        pred_ids: List of Qdrant point IDs

    Returns:
        List of string document IDs (original or raw fallback)
    """
    try:
        points = engine.qdrant.retrieve(engine.collection_name, ids=pred_ids)
        id_map = {}
        for p in points:
            if p.payload and 'original_id' in p.payload:
                id_map[p.id] = str(p.payload['original_id'])
            else:
                id_map[p.id] = str(p.id)
        return [id_map.get(pid, str(pid)) for pid in pred_ids]
    except Exception:
        return [str(pid) for pid in pred_ids]


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_standard_ndcg(engine, queries, qrels, k=10):
    """
    Evaluate NDCG@k using standard AntigravityEngine retrieval.

    Returns:
        (mean_ndcg, list of per-query stat dicts)
    """
    ndcg_scores = []
    query_stats = []

    print(f"Evaluating {len(queries)} queries (standard)...")
    for q_id, q_text in queries.items():
        if q_id not in qrels:
            continue

        start = time.time()
        _, pred_ids, _, _ = engine.run_inference(q_text)
        elapsed = time.time() - start

        mapped_preds = map_predicted_ids(engine, pred_ids)
        relevant_docs = qrels[q_id]
        relevance = [1 if pid in relevant_docs else 0 for pid in mapped_preds]
        score = ndcg_at_k(relevance, k)
        ndcg_scores.append(score)

        query_stats.append({
            "q_id": q_id,
            "ndcg": score,
            "latency": elapsed,
        })

    mean_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
    return mean_ndcg, query_stats


def evaluate_recursive_ndcg(recursive_engine, engine, queries, qrels, k=10):
    """
    Evaluate NDCG@k using RecursiveRetrievalEngine.

    Returns:
        (mean_ndcg, list of per-query stat dicts including decomposition metadata)
    """
    ndcg_scores = []
    query_stats = []

    print(f"Evaluating {len(queries)} queries (recursive)...")
    for q_id, q_text in queries.items():
        if q_id not in qrels:
            continue

        start = time.time()
        trace = recursive_engine.run_recursive_inference(q_text)
        elapsed = time.time() - start

        pred_ids = trace.final_results[:k]
        mapped_preds = map_predicted_ids(engine, pred_ids)
        relevant_docs = qrels[q_id]
        relevance = [1 if pid in relevant_docs else 0 for pid in mapped_preds]
        score = ndcg_at_k(relevance, k)
        ndcg_scores.append(score)

        query_stats.append({
            "q_id": q_id,
            "ndcg": score,
            "nodes": trace.total_nodes,
            "depth": trace.max_depth_reached,
            "retrievals": trace.total_retrieval_calls,
            "latency": elapsed,
        })

    mean_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
    return mean_ndcg, query_stats


# =============================================================================
# Ingestion (reused pattern from benchmark_evolution.py)
# =============================================================================

def ensure_ingested(engine, corpus):
    """Ingest corpus into Qdrant if the collection is empty or undersized."""
    info = engine.qdrant.get_collection(engine.collection_name)
    if info.points_count >= len(corpus):
        print(f"Collection '{engine.collection_name}' has {info.points_count} points. Skipping ingestion.")
        return

    print(f"Collection has {info.points_count} points but corpus has {len(corpus)}. Ingesting...")
    from qdrant_client.models import PointStruct

    batch_size = 50
    keys = list(corpus.keys())
    values = list(corpus.values())

    for i in range(0, len(keys), batch_size):
        batch_keys = keys[i:i + batch_size]
        batch_texts = values[i:i + batch_size]

        try:
            embeddings = engine.embed(batch_texts)

            points = []
            for k, v, t in zip(batch_keys, embeddings, batch_texts):
                try:
                    pid = int(k)
                except (ValueError, TypeError):
                    import uuid
                    pid = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(k)))

                points.append(PointStruct(id=pid, vector=v, payload={"text": t, "original_id": str(k)}))

            engine.qdrant.upsert(engine.collection_name, points)
            if i % 500 == 0:
                print(f"  Ingested {i + len(batch_keys)}/{len(corpus)}")
        except Exception as e:
            print(f"ERROR: Batch {i} ingestion failed: {e}")
            import traceback
            traceback.print_exc()

    final_info = engine.qdrant.get_collection(engine.collection_name)
    print(f"Ingestion complete. Collection now has {final_info.points_count} points.")


# =============================================================================
# Main Benchmark
# =============================================================================

def run_benchmark(args):
    """Run standard vs recursive retrieval benchmark."""
    print("=" * 60)
    print("  RLM RECURSIVE RETRIEVAL BENCHMARK")
    print("=" * 60)
    print()
    print(f"Task:         {args.task}")
    print(f"Decomposer:   {args.decomposer}")
    print(f"Max depth:    {args.max_depth}")
    print(f"Aggregation:  {args.aggregation}")
    print(f"Model:        {args.model}")
    print(f"Top-k:        {args.top_k}")
    print()

    # 1. Initialize engine
    print("--- 1. INITIALIZING ENGINE ---")
    db_path = ChelationConfig.get_db_path(args.task)
    print(f"Database path: {db_path}")

    engine = AntigravityEngine(
        qdrant_location=str(db_path),
        model_name=args.model,
        chelation_p=85,
        use_quantization=True,
        use_centering=False,
    )

    # 2. Load MTEB task data
    print(f"\n--- 2. LOADING MTEB ({args.task}) ---")
    corpus, queries, qrels = load_mteb_data(args.task)

    if corpus is None or queries is None or qrels is None:
        print("ERROR: Failed to load task data. Aborting.")
        return

    print(f"Loaded {len(corpus)} docs, {len(queries)} queries, {len(qrels)} qrels.")

    # 3. Ensure data is ingested
    print("\n--- 3. CHECKING INGESTION ---")
    ensure_ingested(engine, corpus)

    # 4. Create decomposer and recursive engine
    print("\n--- 4. CONFIGURING RECURSIVE ENGINE ---")
    if args.decomposer == "ollama":
        try:
            decomposer = OllamaDecomposer()
            # Quick connectivity test
            test_result = decomposer.decompose("test query")
            print("Using OllamaDecomposer (LLM-backed)")
        except Exception as e:
            print(f"Ollama unavailable ({e}), falling back to MockDecomposer")
            decomposer = MockDecomposer(max_depth=args.max_depth)
    else:
        decomposer = MockDecomposer(max_depth=args.max_depth)
        print("Using MockDecomposer (conjunction-based)")

    recursive_engine = RecursiveRetrievalEngine(
        engine=engine,
        decomposer=decomposer,
        aggregation_strategy=args.aggregation,
        max_depth=args.max_depth,
        top_k=args.top_k,
    )

    # 5. Evaluate: Standard retrieval
    print(f"\n--- 5. STANDARD RETRIEVAL (Baseline) ---")
    engine.chelation_log.clear()

    std_start = time.time()
    standard_ndcg, std_stats = evaluate_standard_ndcg(engine, queries, qrels, k=args.top_k)
    std_total_time = time.time() - std_start

    print(f"Standard NDCG@{args.top_k}: {standard_ndcg:.5f}")
    print(f"Total time: {std_total_time:.2f}s")

    # 6. Evaluate: Recursive retrieval
    print(f"\n--- 6. RECURSIVE RETRIEVAL ---")
    engine.chelation_log.clear()

    rec_start = time.time()
    recursive_ndcg, rec_stats = evaluate_recursive_ndcg(
        recursive_engine, engine, queries, qrels, k=args.top_k
    )
    rec_total_time = time.time() - rec_start

    print(f"Recursive NDCG@{args.top_k}: {recursive_ndcg:.5f}")
    print(f"Total time: {rec_total_time:.2f}s")

    # 7. Report
    print()
    print("=" * 60)
    print("  RESULTS")
    print("=" * 60)
    print()
    print(f"Standard  NDCG@{args.top_k}:  {standard_ndcg:.5f}  ({std_total_time:.2f}s total)")
    print(f"Recursive NDCG@{args.top_k}:  {recursive_ndcg:.5f}  ({rec_total_time:.2f}s total)")

    delta = recursive_ndcg - standard_ndcg
    delta_pct = (delta / standard_ndcg * 100) if standard_ndcg > 0 else 0.0
    print(f"Delta:                {delta:+.5f}  ({delta_pct:+.2f}%)")

    # Recursive retrieval statistics
    if rec_stats:
        avg_nodes = np.mean([s['nodes'] for s in rec_stats])
        avg_depth = np.mean([s['depth'] for s in rec_stats])
        avg_retrievals = np.mean([s['retrievals'] for s in rec_stats])
        avg_rec_latency = np.mean([s['latency'] for s in rec_stats])
        avg_std_latency = np.mean([s['latency'] for s in std_stats]) if std_stats else 0.0

        print()
        print("Recursive Retrieval Statistics:")
        print(f"  Avg nodes per query:      {avg_nodes:.1f}")
        print(f"  Avg max depth reached:    {avg_depth:.1f}")
        print(f"  Avg retrieval calls:      {avg_retrievals:.1f}")
        print(f"  Avg latency per query:    {avg_rec_latency:.3f}s")
        print(f"  Std latency per query:    {avg_std_latency:.3f}s")
        print(f"  Retrieval overhead:       {avg_retrievals:.1f}x vs 1x standard")
        print(f"  Latency overhead:         {avg_rec_latency / avg_std_latency:.1f}x" if avg_std_latency > 0 else "  Latency overhead:         N/A")

        # Per-query decomposition benefit analysis
        print()
        print("Per-Query Decomposition Benefit:")
        improved = 0
        degraded = 0
        unchanged = 0

        # Build standard NDCG lookup
        std_ndcg_map = {s['q_id']: s['ndcg'] for s in std_stats}

        for rs in rec_stats:
            qid = rs['q_id']
            std_score = std_ndcg_map.get(qid, 0.0)
            rec_score = rs['ndcg']

            if rec_score > std_score + 1e-9:
                improved += 1
            elif rec_score < std_score - 1e-9:
                degraded += 1
            else:
                unchanged += 1

        total_evaluated = improved + degraded + unchanged
        print(f"  Queries improved:   {improved}/{total_evaluated} ({improved / total_evaluated * 100:.1f}%)" if total_evaluated > 0 else "  Queries improved:   0")
        print(f"  Queries degraded:   {degraded}/{total_evaluated} ({degraded / total_evaluated * 100:.1f}%)" if total_evaluated > 0 else "  Queries degraded:   0")
        print(f"  Queries unchanged:  {unchanged}/{total_evaluated} ({unchanged / total_evaluated * 100:.1f}%)" if total_evaluated > 0 else "  Queries unchanged:  0")

        # Breakdown by decomposition activity
        decomposed_queries = [s for s in rec_stats if s['nodes'] > 1]
        atomic_queries = [s for s in rec_stats if s['nodes'] == 1]

        if decomposed_queries:
            decomposed_ndcg = np.mean([s['ndcg'] for s in decomposed_queries])
            decomposed_std_ndcg = np.mean([std_ndcg_map.get(s['q_id'], 0.0) for s in decomposed_queries])
            print()
            print(f"  Decomposed queries ({len(decomposed_queries)}):")
            print(f"    Standard NDCG@{args.top_k}:  {decomposed_std_ndcg:.5f}")
            print(f"    Recursive NDCG@{args.top_k}: {decomposed_ndcg:.5f}")
            print(f"    Delta:               {decomposed_ndcg - decomposed_std_ndcg:+.5f}")

        if atomic_queries:
            atomic_ndcg = np.mean([s['ndcg'] for s in atomic_queries])
            atomic_std_ndcg = np.mean([std_ndcg_map.get(s['q_id'], 0.0) for s in atomic_queries])
            print()
            print(f"  Atomic queries ({len(atomic_queries)}):")
            print(f"    Standard NDCG@{args.top_k}:  {atomic_std_ndcg:.5f}")
            print(f"    Recursive NDCG@{args.top_k}: {atomic_ndcg:.5f}")
            print(f"    Delta:               {atomic_ndcg - atomic_std_ndcg:+.5f}")

    # Configuration summary
    print()
    print("-" * 60)
    print("Configuration:")
    print(f"  Decomposer:   {args.decomposer}")
    print(f"  Aggregation:  {args.aggregation}")
    print(f"  Max depth:    {args.max_depth}")
    print(f"  Top-k:        {args.top_k}")
    print(f"  Model:        {args.model}")
    print(f"  Task:         {args.task}")
    print("-" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Benchmark RLM Recursive Retrieval vs Standard"
    )
    parser.add_argument(
        "--task", type=str, default="SciFact",
        help="MTEB Task Name (e.g., SciFact, NFCorpus, FiQA2018)"
    )
    parser.add_argument(
        "--decomposer", type=str, default="mock",
        choices=["mock", "ollama"],
        help="Decomposer type: mock (conjunction-based) or ollama (LLM-backed)"
    )
    parser.add_argument(
        "--max-depth", type=int, default=3,
        help="Max recursion depth for query decomposition"
    )
    parser.add_argument(
        "--aggregation", type=str, default="rrf",
        choices=["rrf", "union", "intersection"],
        help="Result aggregation strategy"
    )
    parser.add_argument(
        "--model", type=str, default="ollama:nomic-embed-text",
        help="Embedding model (e.g., ollama:nomic-embed-text)"
    )
    parser.add_argument(
        "--top-k", type=int, default=10,
        help="Number of top results to evaluate"
    )

    args = parser.parse_args()
    run_benchmark(args)
