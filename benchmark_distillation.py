"""
Benchmark: Comparative Training Modes for ChelatedAI

Compares performance of baseline, offline, and hybrid training modes across
multiple query-sedimentation cycles.

Metrics:
- NDCG@10 after each cycle
- Training loss curves
- Vector alignment with teacher
- Retrieval quality degradation/improvement over time
"""

import time
import argparse
import numpy as np
import mteb
from pathlib import Path
from typing import Dict, List, Tuple
import json

from config import ChelationConfig
from antigravity_engine import AntigravityEngine


# =============================================================================
# Metric Calculation (NDCG@10)
# =============================================================================

def dcg_at_k(r, k):
    """Discounted Cumulative Gain at rank k."""
    r = np.asarray(r, dtype=float)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.


def ndcg_at_k(r, k):
    """Normalized Discounted Cumulative Gain at rank k."""
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max


# =============================================================================
# MTEB Data Loading Helpers
# =============================================================================

def find_keys(obj, target_keys):
    """Recursively search a nested dict for a level containing all target_keys."""
    if not isinstance(obj, dict):
        return None
    if all(k in obj for k in target_keys):
        return obj
    for k, v in obj.items():
        found = find_keys(v, target_keys)
        if found:
            return found
    return None


def find_payload(obj, key):
    """Recursively search a nested dict for a specific key and return its value."""
    if isinstance(obj, dict):
        if key in obj:
            return obj[key]
        for v in obj.values():
            res = find_payload(v, key)
            if res:
                return res
    return None


def load_mteb_data(task_name: str):
    """
    Load corpus, queries, and qrels from an MTEB retrieval task.

    Returns:
        (corpus, queries, qrels) dicts
    """
    try:
        task = mteb.get_task(task_name)
    except KeyError:
        print(f"ERROR: Task '{task_name}' not found in MTEB registry!")
        return None, None, None
    except Exception as e:
        print(f"ERROR: Failed to load MTEB task '{task_name}': {e}")
        return None, None, None

    task.load_data()

    # Try to find the data root containing corpus and queries
    targets = ['corpus', 'queries']
    data_root = find_keys(task.dataset, targets)

    if not data_root:
        c_payload = find_payload(task.dataset, 'corpus')
        q_payload = find_payload(task.dataset, 'queries')
        r_payload = find_payload(task.dataset, 'relevant_docs')
        if not r_payload:
            r_payload = find_payload(task.dataset, 'test')
    else:
        c_payload = data_root.get('corpus')
        q_payload = data_root.get('queries')
        r_payload = data_root.get('relevant_docs')

    corpus = {}
    queries = {}
    qrels = {}

    # Parse corpus
    if c_payload:
        try:
            for k, v in c_payload.items():
                corpus[k] = v['text'] + " " + v['title']
        except (AttributeError, TypeError):
            for row in c_payload:
                if '_id' in row:
                    doc_id = row['_id']
                elif 'id' in row:
                    doc_id = row['id']
                else:
                    continue
                text = row.get('text', '')
                title = row.get('title', '')
                corpus[doc_id] = text + " " + title
        except Exception as e:
            print(f"ERROR: Failed to parse corpus payload: {e}")

    # Parse queries
    if q_payload:
        try:
            for k, v in q_payload.items():
                queries[k] = v['text']
        except (AttributeError, TypeError):
            for row in q_payload:
                if '_id' in row:
                    qid = row['_id']
                elif 'id' in row:
                    qid = row['id']
                else:
                    continue
                queries[qid] = row.get('text', '')
        except Exception as e:
            print(f"ERROR: Failed to parse queries payload: {e}")

    # Parse qrels (relevance judgments)
    if r_payload:
        try:
            if isinstance(r_payload, dict):
                for qid, docs in r_payload.items():
                    qid = str(qid)
                    qrels[qid] = {}
                    if isinstance(docs, dict):
                        for did, score in docs.items():
                            qrels[qid][str(did)] = score
                    else:
                        for did in docs:
                            qrels[qid][str(did)] = 1
            else:
                for row in r_payload:
                    qid = str(row.get('query-id', row.get('query_id', row.get('_id'))))
                    if not qid or qid == 'None':
                        continue
                    if qid not in qrels:
                        qrels[qid] = {}
                    if 'doc-ids' in row:
                        for did in row['doc-ids']:
                            qrels[qid][str(did)] = 1
                    elif 'doc_ids' in row:
                        for did in row['doc_ids']:
                            qrels[qid][str(did)] = 1
                    elif 'doc-id' in row:
                        qrels[qid][str(row['doc-id'])] = row.get('score', 1)
                    elif 'doc_id' in row:
                        qrels[qid][str(row['doc_id'])] = row.get('score', 1)
        except Exception as e:
            print(f"Error parsing qrels: {e}")

    # Fallback to task.qrels attribute
    if not qrels and hasattr(task, 'qrels'):
        qrels = task.qrels.get('test', {})

    return corpus, queries, qrels


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
        mapped_ids = []
        for p in points:
            # Try to extract original doc_id from payload
            original_id = p.payload.get('doc_id', p.payload.get('id', str(p.id)))
            mapped_ids.append(str(original_id))
        return mapped_ids
    except Exception as e:
        print(f"Warning: ID mapping failed, using raw IDs: {e}")
        return [str(pid) for pid in pred_ids]


# =============================================================================
# Evaluation Function
# =============================================================================

def evaluate_engine(engine: AntigravityEngine, queries: Dict, qrels: Dict, max_queries: int = None) -> Tuple[float, List[float]]:
    """
    Evaluate engine on given queries using NDCG@10.
    
    Args:
        engine: Initialized AntigravityEngine
        queries: Dict of query_id -> query_text
        qrels: Dict of query_id -> {doc_id -> relevance}
        max_queries: Optional limit on number of queries to evaluate
        
    Returns:
        (average_ndcg, list_of_ndcg_scores)
    """
    ndcg_scores = []
    query_ids = list(queries.keys())[:max_queries] if max_queries else list(queries.keys())
    
    for qid in query_ids:
        query_text = queries[qid]
        relevant_docs = qrels.get(qid, {})
        
        if not relevant_docs:
            continue
        
        # Run inference
        _, pred_ids, _, _ = engine.run_inference(query_text)
        
        # Map to original IDs
        mapped_ids = map_predicted_ids(engine, pred_ids)
        
        # Calculate NDCG
        relevances = [relevant_docs.get(did, 0) for did in mapped_ids]
        ndcg = ndcg_at_k(relevances, 10)
        ndcg_scores.append(ndcg)
    
    avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
    return avg_ndcg, ndcg_scores


# =============================================================================
# Cycle Runner
# =============================================================================

def run_training_cycle(
    engine: AntigravityEngine,
    queries: Dict,
    qrels: Dict,
    num_cycles: int = 3,
    queries_per_cycle: int = 50,
    epochs_per_cycle: int = 10,
    learning_rate: float = 0.001
) -> List[Dict]:
    """
    Run multiple query-sedimentation cycles and track performance.
    
    Args:
        engine: Initialized engine
        queries: Query dict
        qrels: Relevance judgments
        num_cycles: Number of query-sediment cycles
        queries_per_cycle: Number of queries to run per cycle
        epochs_per_cycle: Training epochs per cycle
        learning_rate: Learning rate for sedimentation
        
    Returns:
        List of result dicts per cycle
    """
    results = []
    query_ids = list(queries.keys())
    
    for cycle in range(num_cycles):
        print(f"\n{'='*60}")
        print(f"CYCLE {cycle + 1}/{num_cycles}")
        print(f"{'='*60}")
        
        # Select queries for this cycle
        start_idx = (cycle * queries_per_cycle) % len(query_ids)
        end_idx = start_idx + queries_per_cycle
        cycle_query_ids = query_ids[start_idx:end_idx]
        
        # Run queries (accumulate chelation log)
        print(f"Running {len(cycle_query_ids)} queries...")
        query_start = time.time()
        for qid in cycle_query_ids:
            engine.run_inference(queries[qid])
        query_time = time.time() - query_start
        
        print(f"Queries completed in {query_time:.2f}s")
        
        # Run sedimentation
        print(f"Running sedimentation (mode={engine.training_mode}, epochs={epochs_per_cycle})...")
        sediment_start = time.time()
        engine.run_sedimentation_cycle(
            threshold=ChelationConfig.DEFAULT_COLLAPSE_THRESHOLD,
            learning_rate=learning_rate,
            epochs=epochs_per_cycle
        )
        sediment_time = time.time() - sediment_start
        
        print(f"Sedimentation completed in {sediment_time:.2f}s")
        
        # Evaluate
        print("Evaluating NDCG@10...")
        eval_start = time.time()
        avg_ndcg, ndcg_list = evaluate_engine(
            engine,
            queries,
            qrels,
            max_queries=100  # Evaluate on subset for speed
        )
        eval_time = time.time() - eval_start
        
        print(f"NDCG@10: {avg_ndcg:.4f} (evaluated in {eval_time:.2f}s)")
        
        results.append({
            'cycle': cycle + 1,
            'ndcg': avg_ndcg,
            'ndcg_std': np.std(ndcg_list),
            'query_time': query_time,
            'sediment_time': sediment_time,
            'eval_time': eval_time,
        })
    
    return results


# =============================================================================
# Main Benchmark
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="Comparative Training Mode Benchmark")
    parser.add_argument("--task", type=str, default="SciFact", help="MTEB task name")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Local model for embeddings")
    parser.add_argument("--teacher", type=str, default="sentence-transformers/all-MiniLM-L6-v2",
                        help="Teacher model for distillation")
    parser.add_argument("--cycles", type=int, default=3, help="Number of training cycles")
    parser.add_argument("--queries-per-cycle", type=int, default=50, help="Queries per cycle")
    parser.add_argument("--epochs", type=int, default=10, help="Training epochs per cycle")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--teacher-weight", type=float, default=0.5, help="Teacher weight for hybrid mode")
    parser.add_argument("--output", type=str, default="benchmark_distillation_results.json",
                        help="Output file for results")
    
    args = parser.parse_args()
    
    print("="*60)
    print("ChelatedAI Comparative Training Mode Benchmark")
    print("="*60)
    print(f"Task: {args.task}")
    print(f"Model: {args.model}")
    print(f"Teacher: {args.teacher}")
    print(f"Cycles: {args.cycles}")
    print(f"Queries/Cycle: {args.queries_per_cycle}")
    print(f"Epochs/Cycle: {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    print(f"Teacher Weight (hybrid): {args.teacher_weight}")
    print("="*60)
    
    # Load MTEB data
    print("\nLoading MTEB dataset...")
    corpus, queries, qrels = load_mteb_data(args.task)
    
    if not corpus or not queries or not qrels:
        print("ERROR: Failed to load dataset!")
        return
    
    print(f"Loaded: {len(corpus)} docs, {len(queries)} queries, {len(qrels)} qrels")
    
    # Prepare corpus for ingestion
    doc_ids = list(corpus.keys())
    doc_texts = [corpus[did] for did in doc_ids]
    doc_payloads = [{"doc_id": did, "text": corpus[did]} for did in doc_ids]
    
    all_results = {}
    
    # ==========================
    # Mode 1: BASELINE
    # ==========================
    print("\n" + "="*60)
    print("MODE 1: BASELINE (Homeostatic only)")
    print("="*60)
    
    engine_baseline = AntigravityEngine(
        qdrant_location=":memory:",
        model_name=args.model,
        training_mode="baseline"
    )
    
    print("Ingesting corpus...")
    engine_baseline.ingest(doc_texts, doc_payloads)
    
    print("Running baseline cycles...")
    baseline_results = run_training_cycle(
        engine_baseline,
        queries,
        qrels,
        num_cycles=args.cycles,
        queries_per_cycle=args.queries_per_cycle,
        epochs_per_cycle=args.epochs,
        learning_rate=args.lr
    )
    
    all_results['baseline'] = baseline_results
    
    # ==========================
    # Mode 2: OFFLINE
    # ==========================
    print("\n" + "="*60)
    print("MODE 2: OFFLINE (Teacher distillation)")
    print("="*60)
    
    engine_offline = AntigravityEngine(
        qdrant_location=":memory:",
        model_name=args.model,
        training_mode="offline",
        teacher_model_name=args.teacher
    )
    
    print("Ingesting corpus...")
    engine_offline.ingest(doc_texts, doc_payloads)
    
    # Run offline distillation first (pre-training)
    print("Running offline distillation pre-training...")
    offline_start = time.time()
    engine_offline.run_offline_distillation(
        batch_size=100,
        learning_rate=args.lr,
        epochs=args.epochs
    )
    offline_time = time.time() - offline_start
    print(f"Offline distillation completed in {offline_time:.2f}s")
    
    print("Running offline mode cycles...")
    offline_results = run_training_cycle(
        engine_offline,
        queries,
        qrels,
        num_cycles=args.cycles,
        queries_per_cycle=args.queries_per_cycle,
        epochs_per_cycle=args.epochs,
        learning_rate=args.lr
    )
    
    all_results['offline'] = {
        'pretraining_time': offline_time,
        'cycles': offline_results
    }
    
    # ==========================
    # Mode 3: HYBRID
    # ==========================
    print("\n" + "="*60)
    print(f"MODE 3: HYBRID (Homeostatic + Teacher, weight={args.teacher_weight})")
    print("="*60)
    
    engine_hybrid = AntigravityEngine(
        qdrant_location=":memory:",
        model_name=args.model,
        training_mode="hybrid",
        teacher_model_name=args.teacher,
        teacher_weight=args.teacher_weight
    )
    
    print("Ingesting corpus...")
    engine_hybrid.ingest(doc_texts, doc_payloads)
    
    print("Running hybrid mode cycles...")
    hybrid_results = run_training_cycle(
        engine_hybrid,
        queries,
        qrels,
        num_cycles=args.cycles,
        queries_per_cycle=args.queries_per_cycle,
        epochs_per_cycle=args.epochs,
        learning_rate=args.lr
    )
    
    all_results['hybrid'] = hybrid_results
    
    # ==========================
    # Summary
    # ==========================
    print("\n" + "="*60)
    print("RESULTS SUMMARY")
    print("="*60)
    
    for mode in ['baseline', 'offline', 'hybrid']:
        print(f"\n{mode.upper()}:")
        
        if mode == 'offline':
            print(f"  Pretraining time: {all_results[mode]['pretraining_time']:.2f}s")
            cycles = all_results[mode]['cycles']
        else:
            cycles = all_results[mode]
        
        for result in cycles:
            print(f"  Cycle {result['cycle']}: NDCG={result['ndcg']:.4f} ± {result['ndcg_std']:.4f}")
    
    # Save results
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    
    # Final comparison
    print("\n" + "="*60)
    print("FINAL COMPARISON (Last Cycle)")
    print("="*60)
    
    final_baseline = all_results['baseline'][-1]['ndcg']
    final_offline = all_results['offline']['cycles'][-1]['ndcg']
    final_hybrid = all_results['hybrid'][-1]['ndcg']
    
    print(f"Baseline: {final_baseline:.4f}")
    print(f"Offline:  {final_offline:.4f} ({(final_offline - final_baseline) * 100:+.2f}%)")
    print(f"Hybrid:   {final_hybrid:.4f} ({(final_hybrid - final_baseline) * 100:+.2f}%)")
    
    print("\n" + "="*60)
    print("Benchmark Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
