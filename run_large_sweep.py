import argparse
import itertools
from benchmark_evolution import load_mteb_data, evaluate_ndcg
from config import ChelationConfig
from antigravity_engine import AntigravityEngine
import json
import csv
import os
from datetime import datetime



def _write_json_results(path: str, results):
    """Persist sweep results as a JSON array."""
    with open(path, 'w', encoding="utf-8") as f:
        json.dump(results, f, indent=2)


def run_large_parameter_sweep(
    task_name="SciFact",
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    output_prefix="large_sweep",
    max_queries=None,
    db_path=None,
    checkpoint_every=50,
    learning_rates=None,
    thresholds=None,
    noise_scales=None,
    epochs_list=None,
    push_magnitudes=None,
):
    print(f"Starting large parameter sweep on {task_name} using {model_name}")
    
    # Define an extensive parameter grid
    learning_rates = learning_rates or [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]
    thresholds = thresholds or [1, 2, 3, 4, 5]
    noise_scales = noise_scales or [0.0, 0.01, 0.05, 0.1, 0.2, 0.3, 0.5]
    epochs_list = epochs_list or [1, 3, 5, 10, 20, 50]
    push_magnitudes = push_magnitudes or [0.01, 0.05, 0.1, 0.2, 0.5]
    
    combinations = itertools.product(learning_rates, thresholds, noise_scales, epochs_list, push_magnitudes)
    total_runs = (
        len(list(learning_rates))
        * len(list(thresholds))
        * len(list(noise_scales))
        * len(list(epochs_list))
        * len(list(push_magnitudes))
    )
    
    print(f"Total configurations to test: {total_runs}")
    
    csv_file = f"{output_prefix}_results.csv"
    json_file = f"{output_prefix}_results.json"
    
    # Initialize CSV with headers if it doesn't exist
    if not os.path.exists(csv_file):
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Timestamp", "LearningRate", "Threshold", "NoiseScale", "Epochs", "PushMagnitude", "BaselineNDCG", "PostNDCG", "Gain"])
    
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
                
    base_score = evaluate_ndcg(base_engine, queries, qrels, max_queries=max_queries)
    print(f"Baseline NDCG@10: {base_score:.5f}")

    results = []
    try:
        with open(json_file, 'r', encoding="utf-8") as f:
            loaded = json.load(f)
            if isinstance(loaded, list):
                results.extend(loaded)
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    
    # Run the sweep
    for i, (lr, thresh, noise, epochs, push_mag) in enumerate(combinations):
        print(f"[{i+1}/{total_runs}] Testing LR={lr}, Thresh={thresh}, Noise={noise}, Epochs={epochs}, Push={push_mag}")
        
        # Reuse base engine to avoid Qdrant file lock issues
        engine = base_engine
        
        # Reset adapter to identity state
        from chelation_adapter import create_adapter
        if os.path.exists("adapter_weights.pt"):
            os.remove("adapter_weights.pt")
        
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
        results.append(result_entry)

        if len(results) % checkpoint_every == 0:
            _write_json_results(json_file, results)
            
        # Save to CSV table iteratively so no data is lost if interrupted
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, lr, thresh, noise, epochs, push_mag, base_score, post_score, gain])

    _write_json_results(json_file, results)
    print(f"Sweep completed. Results saved to {json_file} and {csv_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Large Parameter Sweep for Chelation Sedimentation")
    parser.add_argument("--task", type=str, default="SciFact", help="MTEB Task")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding Model")
    parser.add_argument("--out", type=str, default="large_sweep", help="Output file prefix")
    parser.add_argument("--max-queries", type=int, default=None, help="Optional query cap for bounded large-sweep runs")
    parser.add_argument("--db-path", type=str, default=None, help="Optional isolated Qdrant path for this large sweep")
    
    args = parser.parse_args()
    run_large_parameter_sweep(args.task, args.model, args.out, max_queries=args.max_queries, db_path=args.db_path)
