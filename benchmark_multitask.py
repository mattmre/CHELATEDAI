"""
Benchmark: Multi-Task Retrieval Benchmark for Future Research

Runs comprehensive benchmarks across multiple MTEB retrieval tasks to evaluate:
- Retrieval quality (NDCG@10)
- Stability metric (average Jaccard similarity across runs)
- Learning gain (post-sedimentation NDCG improvement)

Supports task selection by comma-separated list or predefined suite presets.
Output: JSON report with per-task metrics and aggregate summary.
"""

import time
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json

from config import ChelationConfig
from antigravity_engine import AntigravityEngine

# Reuse utilities from benchmark_distillation
from benchmark_distillation import (
    load_mteb_data,
    map_predicted_ids,
    evaluate_engine
)


# =============================================================================
# Task Suite Presets
# =============================================================================

TASK_SUITES = {
    "mini": ["SciFact"],
    "small": ["SciFact", "NFCorpus"],
    "medium": ["SciFact", "NFCorpus", "FiQA2018"],
    "research": ["SciFact", "NFCorpus", "FiQA2018", "TRECCOVID"]
}


def parse_tasks(task_arg: str) -> List[str]:
    """
    Parse task argument into list of task names.
    
    Args:
        task_arg: Either a suite name (mini/small/medium/research) 
                  or comma-separated task list
                  
    Returns:
        List of MTEB task names
    """
    task_arg = task_arg.strip()
    
    # Check if it's a preset suite
    if task_arg in TASK_SUITES:
        return TASK_SUITES[task_arg]
    
    # Otherwise treat as comma-separated list
    tasks = [t.strip() for t in task_arg.split(',')]
    return [t for t in tasks if t]  # Filter empty strings


# =============================================================================
# Stability Metric: Jaccard Similarity
# =============================================================================

def jaccard_similarity(set_a: set, set_b: set) -> float:
    """
    Calculate Jaccard similarity between two sets.
    
    Args:
        set_a: First set of items
        set_b: Second set of items
        
    Returns:
        Jaccard index in [0.0, 1.0]
    """
    if not set_a and not set_b:
        return 1.0  # Both empty
    
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    
    if union == 0:
        return 0.0
    
    return intersection / union


def compute_stability(
    engine: AntigravityEngine,
    queries: Dict[str, str],
    num_runs: int = 3,
    max_queries: int = 50
) -> Tuple[float, List[float]]:
    """
    Compute retrieval stability by running same queries multiple times
    and measuring Jaccard similarity of result sets.
    
    Args:
        engine: Initialized AntigravityEngine
        queries: Dict of query_id -> query_text
        num_runs: Number of repeated runs per query
        max_queries: Max queries to evaluate
        
    Returns:
        (average_jaccard, list_of_jaccard_scores)
    """
    query_ids = list(queries.keys())[:max_queries]
    jaccard_scores = []
    
    for qid in query_ids:
        query_text = queries[qid]
        
        # Run query multiple times
        result_sets = []
        for _ in range(num_runs):
            _, pred_ids, _, _ = engine.run_inference(query_text)
            mapped_ids = map_predicted_ids(engine, pred_ids)
            result_sets.append(set(mapped_ids))
        
        # Compute pairwise Jaccard similarities
        pairwise_jaccards = []
        for i in range(len(result_sets)):
            for j in range(i + 1, len(result_sets)):
                jac = jaccard_similarity(result_sets[i], result_sets[j])
                pairwise_jaccards.append(jac)
        
        # Average Jaccard for this query
        if pairwise_jaccards:
            avg_jac = np.mean(pairwise_jaccards)
            jaccard_scores.append(avg_jac)
    
    overall_avg = np.mean(jaccard_scores) if jaccard_scores else 0.0
    return overall_avg, jaccard_scores


# =============================================================================
# Learning Gain Metric
# =============================================================================

def compute_learning_gain(
    engine: AntigravityEngine,
    queries: Dict[str, str],
    qrels: Dict[str, Dict[str, int]],
    num_queries_train: int = 50,
    max_queries_eval: int = 100,
    epochs: int = 10,
    learning_rate: float = 0.001
) -> Dict[str, Any]:
    """
    Compute learning gain by measuring NDCG improvement after sedimentation.
    
    Args:
        engine: Initialized engine
        queries: Query dict
        qrels: Relevance judgments
        num_queries_train: Number of queries to accumulate for sedimentation
        max_queries_eval: Number of queries for evaluation
        epochs: Training epochs
        learning_rate: Learning rate
        
    Returns:
        Dict with pre_ndcg, post_ndcg, gain, and gain_percent
    """
    query_ids = list(queries.keys())
    
    # Evaluate pre-sedimentation
    print("  Evaluating pre-sedimentation NDCG...")
    pre_ndcg, _ = evaluate_engine(
        engine,
        queries,
        qrels,
        max_queries=max_queries_eval
    )
    
    # Run training queries to accumulate chelation log
    print(f"  Running {num_queries_train} training queries...")
    train_query_ids = query_ids[:num_queries_train]
    for qid in train_query_ids:
        engine.run_inference(queries[qid])
    
    # Run sedimentation
    print(f"  Running sedimentation (epochs={epochs})...")
    sediment_start = time.time()
    engine.run_sedimentation_cycle(
        threshold=ChelationConfig.DEFAULT_COLLAPSE_THRESHOLD,
        learning_rate=learning_rate,
        epochs=epochs
    )
    sediment_time = time.time() - sediment_start
    
    # Evaluate post-sedimentation
    print("  Evaluating post-sedimentation NDCG...")
    post_ndcg, _ = evaluate_engine(
        engine,
        queries,
        qrels,
        max_queries=max_queries_eval
    )
    
    gain = post_ndcg - pre_ndcg
    gain_percent = (gain / pre_ndcg * 100) if pre_ndcg > 0 else 0.0
    
    return {
        'pre_ndcg': pre_ndcg,
        'post_ndcg': post_ndcg,
        'gain': gain,
        'gain_percent': gain_percent,
        'sediment_time': sediment_time
    }


# =============================================================================
# Task Benchmark Runner
# =============================================================================

def run_task_benchmark(
    task_name: str,
    model_name: str,
    max_queries: int,
    num_queries_train: int,
    epochs: int,
    learning_rate: float,
    stability_runs: int = 3
) -> Dict[str, Any]:
    """
    Run comprehensive benchmark on a single MTEB task.
    
    Args:
        task_name: MTEB task name
        model_name: Model identifier
        max_queries: Max queries for evaluation metrics
        num_queries_train: Queries for training/sedimentation
        epochs: Training epochs
        learning_rate: Learning rate
        stability_runs: Number of runs for stability metric
        
    Returns:
        Dict with all benchmark results for this task
    """
    print(f"\n{'='*60}")
    print(f"TASK: {task_name}")
    print(f"{'='*60}")
    
    # Load data
    print("Loading MTEB dataset...")
    corpus, queries, qrels = load_mteb_data(task_name)
    
    if not corpus or not queries or not qrels:
        print(f"ERROR: Failed to load task '{task_name}', skipping...")
        return {
            'task': task_name,
            'status': 'failed',
            'error': 'Failed to load dataset'
        }
    
    print(f"Loaded: {len(corpus)} docs, {len(queries)} queries, {len(qrels)} qrels")
    
    # Prepare corpus
    doc_ids = list(corpus.keys())
    doc_texts = [corpus[did] for did in doc_ids]
    doc_payloads = [{"doc_id": did, "text": corpus[did]} for did in doc_ids]
    
    # Initialize engine
    print("Initializing engine...")
    engine = AntigravityEngine(
        qdrant_location=":memory:",
        model_name=model_name,
        training_mode="baseline"
    )
    
    print("Ingesting corpus...")
    ingest_start = time.time()
    engine.ingest(doc_texts, doc_payloads)
    ingest_time = time.time() - ingest_start
    print(f"Ingestion completed in {ingest_time:.2f}s")
    
    # Benchmark 1: Retrieval Quality (NDCG@10)
    print("\n[1/3] Measuring Retrieval Quality (NDCG@10)...")
    quality_start = time.time()
    ndcg_score, ndcg_list = evaluate_engine(
        engine,
        queries,
        qrels,
        max_queries=max_queries
    )
    quality_time = time.time() - quality_start
    print(f"  NDCG@10: {ndcg_score:.4f} (σ={np.std(ndcg_list):.4f})")
    print(f"  Time: {quality_time:.2f}s")
    
    # Benchmark 2: Stability (Jaccard Similarity)
    print("\n[2/3] Measuring Stability (Jaccard)...")
    stability_start = time.time()
    stability_score, stability_list = compute_stability(
        engine,
        queries,
        num_runs=stability_runs,
        max_queries=min(max_queries, 30)  # Limit for time
    )
    stability_time = time.time() - stability_start
    print(f"  Avg Jaccard: {stability_score:.4f} (σ={np.std(stability_list):.4f})")
    print(f"  Time: {stability_time:.2f}s")
    
    # Benchmark 3: Learning Gain
    print("\n[3/3] Measuring Learning Gain...")
    learning_start = time.time()
    learning_results = compute_learning_gain(
        engine,
        queries,
        qrels,
        num_queries_train=num_queries_train,
        max_queries_eval=max_queries,
        epochs=epochs,
        learning_rate=learning_rate
    )
    learning_time = time.time() - learning_start
    print(f"  Pre-NDCG:  {learning_results['pre_ndcg']:.4f}")
    print(f"  Post-NDCG: {learning_results['post_ndcg']:.4f}")
    print(f"  Gain: {learning_results['gain']:+.4f} ({learning_results['gain_percent']:+.2f}%)")
    print(f"  Time: {learning_time:.2f}s")
    
    # Compile results
    results = {
        'task': task_name,
        'status': 'success',
        'corpus_size': len(corpus),
        'num_queries': len(queries),
        'num_qrels': len(qrels),
        'ingest_time': ingest_time,
        'retrieval_quality': {
            'ndcg_10': ndcg_score,
            'ndcg_std': float(np.std(ndcg_list)),
            'ndcg_list': [float(x) for x in ndcg_list[:20]],  # Store first 20
            'time': quality_time
        },
        'stability': {
            'avg_jaccard': stability_score,
            'jaccard_std': float(np.std(stability_list)),
            'jaccard_list': [float(x) for x in stability_list[:20]],
            'time': stability_time
        },
        'learning_gain': {
            'pre_ndcg': learning_results['pre_ndcg'],
            'post_ndcg': learning_results['post_ndcg'],
            'gain': learning_results['gain'],
            'gain_percent': learning_results['gain_percent'],
            'sediment_time': learning_results['sediment_time'],
            'time': learning_time
        }
    }
    
    return results


# =============================================================================
# Aggregate Summary
# =============================================================================

def compute_aggregate_summary(task_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute aggregate statistics across all tasks.
    
    Args:
        task_results: List of per-task result dicts
        
    Returns:
        Dict with aggregate metrics
    """
    successful_tasks = [r for r in task_results if r['status'] == 'success']
    
    if not successful_tasks:
        return {
            'num_tasks': len(task_results),
            'num_successful': 0,
            'num_failed': len(task_results),
            'error': 'All tasks failed'
        }
    
    # Aggregate NDCG
    ndcg_scores = [r['retrieval_quality']['ndcg_10'] for r in successful_tasks]
    
    # Aggregate stability
    jaccard_scores = [r['stability']['avg_jaccard'] for r in successful_tasks]
    
    # Aggregate learning gain
    gain_scores = [r['learning_gain']['gain'] for r in successful_tasks]
    gain_percents = [r['learning_gain']['gain_percent'] for r in successful_tasks]
    
    summary = {
        'num_tasks': len(task_results),
        'num_successful': len(successful_tasks),
        'num_failed': len(task_results) - len(successful_tasks),
        'aggregate_ndcg': {
            'mean': float(np.mean(ndcg_scores)),
            'std': float(np.std(ndcg_scores)),
            'min': float(np.min(ndcg_scores)),
            'max': float(np.max(ndcg_scores))
        },
        'aggregate_stability': {
            'mean': float(np.mean(jaccard_scores)),
            'std': float(np.std(jaccard_scores)),
            'min': float(np.min(jaccard_scores)),
            'max': float(np.max(jaccard_scores))
        },
        'aggregate_learning_gain': {
            'mean_gain': float(np.mean(gain_scores)),
            'mean_gain_percent': float(np.mean(gain_percents)),
            'std_gain': float(np.std(gain_scores)),
            'positive_gains': sum(1 for g in gain_scores if g > 0),
            'negative_gains': sum(1 for g in gain_scores if g < 0)
        }
    }
    
    return summary


# =============================================================================
# Main Benchmark
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Multi-Task Retrieval Benchmark for Future Research"
    )
    parser.add_argument(
        "--tasks",
        type=str,
        default="mini",
        help="Task selection: suite name (mini/small/medium/research) or comma-separated task list"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Model for embeddings"
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=100,
        help="Max queries for evaluation metrics"
    )
    parser.add_argument(
        "--num-queries-train",
        type=int,
        default=50,
        help="Number of queries for sedimentation training"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Training epochs for sedimentation"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.001,
        help="Learning rate"
    )
    parser.add_argument(
        "--stability-runs",
        type=int,
        default=3,
        help="Number of runs for stability measurement"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_multitask_results.json",
        help="Output JSON file"
    )
    
    args = parser.parse_args()
    
    # Parse tasks
    tasks = parse_tasks(args.tasks)
    
    print("="*60)
    print("Multi-Task Retrieval Benchmark")
    print("="*60)
    print(f"Tasks: {', '.join(tasks)}")
    print(f"Model: {args.model}")
    print(f"Max Queries (eval): {args.max_queries}")
    print(f"Training Queries: {args.num_queries_train}")
    print(f"Epochs: {args.epochs}")
    print(f"Learning Rate: {args.lr}")
    print(f"Stability Runs: {args.stability_runs}")
    print("="*60)
    
    # Run benchmarks on all tasks
    all_results = []
    overall_start = time.time()
    
    for i, task_name in enumerate(tasks):
        print(f"\n\n### Task {i+1}/{len(tasks)} ###")
        task_result = run_task_benchmark(
            task_name=task_name,
            model_name=args.model,
            max_queries=args.max_queries,
            num_queries_train=args.num_queries_train,
            epochs=args.epochs,
            learning_rate=args.lr,
            stability_runs=args.stability_runs
        )
        all_results.append(task_result)
    
    overall_time = time.time() - overall_start
    
    # Compute aggregate summary
    print("\n" + "="*60)
    print("AGGREGATE SUMMARY")
    print("="*60)
    
    summary = compute_aggregate_summary(all_results)
    
    print(f"\nTasks Completed: {summary['num_successful']}/{summary['num_tasks']}")
    
    if summary['num_successful'] > 0:
        print("\nRetrieval Quality (NDCG@10):")
        print(f"  Mean: {summary['aggregate_ndcg']['mean']:.4f} ± {summary['aggregate_ndcg']['std']:.4f}")
        print(f"  Range: [{summary['aggregate_ndcg']['min']:.4f}, {summary['aggregate_ndcg']['max']:.4f}]")
        
        print("\nStability (Jaccard):")
        print(f"  Mean: {summary['aggregate_stability']['mean']:.4f} ± {summary['aggregate_stability']['std']:.4f}")
        print(f"  Range: [{summary['aggregate_stability']['min']:.4f}, {summary['aggregate_stability']['max']:.4f}]")
        
        print("\nLearning Gain:")
        print(f"  Mean Gain: {summary['aggregate_learning_gain']['mean_gain']:+.4f} ({summary['aggregate_learning_gain']['mean_gain_percent']:+.2f}%)")
        print(f"  Positive Gains: {summary['aggregate_learning_gain']['positive_gains']}/{summary['num_successful']}")
        print(f"  Negative Gains: {summary['aggregate_learning_gain']['negative_gains']}/{summary['num_successful']}")
    
    print(f"\nTotal Time: {overall_time:.2f}s")
    
    # Save results
    output_data = {
        'config': {
            'tasks': tasks,
            'model': args.model,
            'max_queries': args.max_queries,
            'num_queries_train': args.num_queries_train,
            'epochs': args.epochs,
            'learning_rate': args.lr,
            'stability_runs': args.stability_runs
        },
        'task_results': all_results,
        'aggregate_summary': summary,
        'total_time': overall_time
    }
    
    output_path = Path(args.output)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {output_path}")
    print("="*60)
    print("Benchmark Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
