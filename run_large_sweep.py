import argparse
import itertools
from benchmark_evolution import run_evolution, load_mteb_data, evaluate_ndcg
from config import ChelationConfig
from antigravity_engine import AntigravityEngine
import json
import csv
import os
from datetime import datetime

def run_large_parameter_sweep(task_name="SciFact", model_name="sentence-transformers/all-MiniLM-L6-v2", output_prefix="large_sweep"):
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
    db_path = ChelationConfig.get_db_path(task_name)
    
    base_engine = AntigravityEngine(
        qdrant_location=str(db_path),
        model_name=model_name,
        chelation_p=85,
        use_quantization=True,
        use_centering=False
    )
                
    base_score = evaluate_ndcg(base_engine, queries, qrels)
    print(f"Baseline NDCG@10: {base_score:.5f}")
    
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
        evaluate_ndcg(engine, queries, qrels) 
        
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
        post_score = evaluate_ndcg(engine, queries, qrels)
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
        
        try:
            with open(json_file, 'r') as f:
                current_results = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            current_results = []
            
        current_results.append(result_entry)
        
        with open(json_file, 'w') as f:
            json.dump(current_results, f, indent=2)
            
        # Save to CSV table iteratively so no data is lost if interrupted
        with open(csv_file, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, lr, thresh, noise, epochs, push_mag, base_score, post_score, gain])

    print(f"Sweep completed. Results saved to {json_file} and {csv_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Large Parameter Sweep for Chelation Sedimentation")
    parser.add_argument("--task", type=str, default="SciFact", help="MTEB Task")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2", help="Embedding Model")
    parser.add_argument("--out", type=str, default="large_sweep", help="Output file prefix")
    
    args = parser.parse_args()
    run_large_parameter_sweep(args.task, args.model, args.out)
