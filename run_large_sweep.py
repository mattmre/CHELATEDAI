import argparse
import csv
import itertools
import os
import sys
from datetime import datetime

from antigravity_engine import AntigravityEngine
from benchmark_evolution import evaluate_ndcg, load_mteb_data
from chelation_adapter import create_adapter
from config import ChelationConfig
from run_sweep import prepare_sweep_baseline, remove_configured_adapter_weights
from sweep_corpus_restore import restore_collection, snapshot_collection
from sweep_result_store import (
    append_jsonl,
    materialize_json_array,
    migrate_json_array_to_jsonl,
)


def _snapshot_baseline_collection(client, collection_name):
    """Snapshot the baseline corpus. The sweep keeps this client open."""
    return snapshot_collection(client, collection_name)


def run_large_parameter_sweep(task_name="SciFact", model_name="sentence-transformers/all-MiniLM-L6-v2", output_prefix="large_sweep", max_queries=None, db_path=None):
    print(f"Starting large parameter sweep on {task_name} using {model_name}")

    # Define an extensive parameter grid
    # This matrix contains 7,350 unique configurations
    learning_rates = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]
    thresholds = [1, 2, 3, 4, 5]
    noise_scales = [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    epochs_list = [1, 3, 5, 10, 20, 50]
    push_magnitudes = [0.01, 0.05, 0.1, 0.2, 0.5]

    # Generate all combinations
    combinations = list(itertools.product(learning_rates, thresholds, noise_scales, epochs_list, push_magnitudes))
    total_runs = len(combinations)

    print(f"Total configurations to test: {total_runs}")

    csv_file = f"{output_prefix}_results.csv"
    json_file = f"{output_prefix}_results.json"
    jsonl_file = f"{output_prefix}_results.jsonl"

    # Load data once
    print(f"Loading MTEB data for {task_name}...")
    corpus, queries, qrels = load_mteb_data(task_name)
    if corpus is None:
        print("Failed to load data. Aborting.")
        return

    print("Calculating baseline performance...")
    db_path = db_path or str(ChelationConfig.get_db_path(task_name))

    base_engine = AntigravityEngine(
        qdrant_location=str(db_path),
        model_name=model_name,
        chelation_p=85,
        use_quantization=True,
        use_centering=False
    )

    prepared = prepare_sweep_baseline(
        base_engine,
        corpus,
        queries,
        qrels,
        max_queries=max_queries,
        snapshot_collection=_snapshot_baseline_collection,
    )
    if isinstance(prepared, int):
        return prepared
    base_score, corpus_snapshot = prepared

    # Result files follow a successful raw baseline. A failed ingest returns first.
    migrate_json_array_to_jsonl(json_file, jsonl_file)
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "LearningRate", "Threshold", "NoiseScale", "Epochs", "PushMagnitude", "BaselineNDCG", "PostNDCG", "Gain"])

    # Run the sweep
    for i, (lr, thresh, noise, epochs, push_mag) in enumerate(combinations):
        print(f"[{i+1}/{total_runs}] Testing LR={lr}, Thresh={thresh}, Noise={noise}, Epochs={epochs}, Push={push_mag}")

        # One Qdrant client stays open. Sedimentation upserts adapted vectors,
        # so each configuration starts from the baseline snapshot.
        engine = base_engine
        restore_collection(engine.qdrant, engine.collection_name, corpus_snapshot)

        # Reset adapter to identity state
        remove_configured_adapter_weights()
        engine.adapter = create_adapter(
            adapter_type=ChelationConfig.ADAPTER_TYPE,
            input_dim=engine.vector_size,
            rank=ChelationConfig.LOW_RANK_ADAPTER_RANK
        )

        engine.chelation_log.clear()
        evaluate_ndcg(engine, queries, qrels, max_queries=max_queries)

        # Patch Configs Temporarily
        original_noise_enabled = getattr(ChelationConfig, 'NOISE_INJECTION_ENABLED', False)
        original_noise_scale = getattr(ChelationConfig, 'NOISE_INJECTION_BASE_SCALE', 0.05)
        original_push_mag = getattr(ChelationConfig, 'HOMEOSTATIC_PUSH_MAGNITUDE', 0.1)

        if noise > 0:
            ChelationConfig.NOISE_INJECTION_ENABLED = True
            ChelationConfig.NOISE_INJECTION_BASE_SCALE = noise
        else:
            ChelationConfig.NOISE_INJECTION_ENABLED = False

        ChelationConfig.HOMEOSTATIC_PUSH_MAGNITUDE = push_mag

        # Run sedimentation
        engine.run_sedimentation_cycle(threshold=thresh, learning_rate=lr, epochs=epochs, noise_injection=noise if noise > 0 else None)

        # Restore Configs
        ChelationConfig.NOISE_INJECTION_ENABLED = original_noise_enabled
        ChelationConfig.NOISE_INJECTION_BASE_SCALE = original_noise_scale
        ChelationConfig.HOMEOSTATIC_PUSH_MAGNITUDE = original_push_mag

        # Evaluate post-learning
        post_score = evaluate_ndcg(engine, queries, qrels, max_queries=max_queries)
        gain = post_score - base_score

        print(f"Post-Learning NDCG@10: {post_score:.5f} (Gain: {gain:+.5f})")

        timestamp = datetime.now().isoformat()

        # Save to JSON
        result_entry = {
            "timestamp": timestamp,
            "config": {
                "learning_rate": lr,
                "threshold": thresh,
                "noise_scale": noise,
                "epochs": epochs,
                "push_magnitude": push_mag
            },
            "metrics": {
                "baseline_ndcg": base_score,
                "post_ndcg": post_score,
                "gain": gain
            }
        }

        append_jsonl(jsonl_file, result_entry)

        # Save to CSV table iteratively so no data is lost if interrupted
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, lr, thresh, noise, epochs, push_mag, base_score, post_score, gain])

    if os.path.exists(jsonl_file):
        written = materialize_json_array(jsonl_file, json_file)
    else:
        written = 0
    print(f"Sweep completed. {written} results saved to {jsonl_file}, {json_file}, and {csv_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Large Parameter Sweep for Chelation Sedimentation")
    parser.add_argument("--task", type=str, default="SciFact", help="MTEB Task")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding Model")
    parser.add_argument("--out", type=str, default="large_sweep", help="Output file prefix")
    parser.add_argument("--max-queries", type=int, default=None, help="Optional query cap for bounded large-sweep runs")
    parser.add_argument("--db-path", type=str, default=None, help="Optional isolated Qdrant path for this large sweep")

    args = parser.parse_args()
    exit_code = run_large_parameter_sweep(args.task, args.model, args.out, max_queries=args.max_queries, db_path=args.db_path)
    if exit_code:
        sys.exit(exit_code)
