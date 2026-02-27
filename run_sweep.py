import argparse
import itertools
from benchmark_evolution import load_mteb_data, evaluate_ndcg
from config import ChelationConfig
from antigravity_engine import AntigravityEngine
import json
from datetime import datetime

def run_parameter_sweep(task_name="SciFact", model_name="ollama:nomic-embed-text", output_file="sweep_results.json"):
    print(f"Starting parameter sweep on {task_name} using {model_name}")
    
    # Define the parameter grid
    learning_rates = [0.01, 0.1, 0.5]
    thresholds = [1, 2, 3]
    noise_scales = [0.0, 0.05, 0.2]  # 0.0 means disabled
    epochs_list = [5, 10, 20]
    
    # Load data once
    print(f"Loading MTEB data for {task_name}...")
    corpus, queries, qrels = load_mteb_data(task_name)
    if corpus is None:
        print("Failed to load data. Aborting.")
        return

    results = []
    
    # Generate all combinations
    combinations = list(itertools.product(learning_rates, thresholds, noise_scales, epochs_list))
    total_runs = len(combinations)
    
    print(f"Total configurations to test: {total_runs}")
    
    # Pre-calculate baseline so we don't have to do it every time
    print("Calculating baseline performance...")
    db_path = ChelationConfig.get_db_path(task_name)
    
    base_engine = AntigravityEngine(
        qdrant_location=str(db_path),
        model_name=model_name,
        chelation_p=85,
        use_quantization=True,
        use_centering=False
    )
    
    # Ingest data if necessary (reuse logic from benchmark_evolution)
    info = base_engine.qdrant.get_collection(base_engine.collection_name)
    if info.points_count < len(corpus):
        print(f"Collection empty. Ingesting {len(corpus)} documents...")
        from qdrant_client.models import PointStruct
        import uuid
        
        batch_size = 50
        keys = list(corpus.keys())
        values = list(corpus.values())
        
        for i in range(0, len(keys), batch_size):
            batch_keys = keys[i:i+batch_size]
            batch_texts = values[i:i+batch_size]
            try:
                embeddings = base_engine.embed(batch_texts)
                points = []
                for k, v, t in zip(batch_keys, embeddings, batch_texts):
                    try:
                        pid = int(k)
                    except (ValueError, TypeError):
                        pid = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(k)))
                    points.append(PointStruct(id=pid, vector=v, payload={"text": t, "original_id": str(k)}))
                base_engine.qdrant.upsert(base_engine.collection_name, points)
                if i % 500 == 0:
                    print(f"Ingested {i + len(batch_keys)}/{len(corpus)}")
            except Exception as e:
                print(f"Ingestion failed for batch {i}: {e}")
                
    base_score = evaluate_ndcg(base_engine, queries, qrels)
    print(f"Baseline NDCG@10: {base_score:.5f}")
    
    # Run the sweep
    for i, (lr, thresh, noise, epochs) in enumerate(combinations):
        print(f"[{i+1}/{total_runs}] Testing LR={lr}, Threshold={thresh}, Noise={noise}, Epochs={epochs}")
        
        # Reuse base engine to avoid Qdrant file lock issues
        engine = base_engine
        
        # Reset adapter to identity state
        import os
        from chelation_adapter import create_adapter
        if os.path.exists("adapter_weights.pt"):
            os.remove("adapter_weights.pt")
        
        engine.adapter = create_adapter(
            adapter_type=ChelationConfig.ADAPTER_TYPE,
            input_dim=engine.vector_size,
            rank=ChelationConfig.LOW_RANK_ADAPTER_RANK
        )
        
        # Clear log and run an initial evaluation to populate the chelation log
        engine.chelation_log.clear()
        evaluate_ndcg(engine, queries, qrels) 
        
        # Enable noise injection temporarily via config patching
        original_noise_enabled = ChelationConfig.NOISE_INJECTION_ENABLED
        original_noise_scale = ChelationConfig.NOISE_INJECTION_BASE_SCALE
        
        if noise > 0:
            ChelationConfig.NOISE_INJECTION_ENABLED = True
            ChelationConfig.NOISE_INJECTION_BASE_SCALE = noise
        else:
            ChelationConfig.NOISE_INJECTION_ENABLED = False
            
        # Run sedimentation
        engine.run_sedimentation_cycle(threshold=thresh, learning_rate=lr, epochs=epochs, noise_injection=noise if noise > 0 else None)
        
        # Restore config
        ChelationConfig.NOISE_INJECTION_ENABLED = original_noise_enabled
        ChelationConfig.NOISE_INJECTION_BASE_SCALE = original_noise_scale
        
        # Evaluate post-learning
        post_score = evaluate_ndcg(engine, queries, qrels)
        gain = post_score - base_score
        
        print(f"Post-Learning NDCG@10: {post_score:.5f} (Gain: {gain:+.5f})")
        
        result_entry = {
            "timestamp": datetime.now().isoformat(),
            "config": {
                "learning_rate": lr,
                "threshold": thresh,
                "noise_scale": noise,
                "epochs": epochs
            },
            "metrics": {
                "baseline_ndcg": base_score,
                "post_ndcg": post_score,
                "gain": gain
            }
        }
        results.append(result_entry)
        
        # Save incrementally
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)
            
        # Reset adapter weights so the next run starts fresh
        import os
        if os.path.exists("adapter_weights.pt"):
            os.remove("adapter_weights.pt")

    print(f"Sweep completed. Results saved to {output_file}")
    
    # Find best
    best_run = max(results, key=lambda x: x["metrics"]["gain"])
    print("Best Configuration:")
    print(json.dumps(best_run, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Parameter Sweep for Chelation Sedimentation")
    parser.add_argument("--task", type=str, default="SciFact", help="MTEB Task")
    parser.add_argument("--model", type=str, default="ollama:nomic-embed-text", help="Embedding Model")
    parser.add_argument("--out", type=str, default="sweep_results.json", help="Output JSON file")
    
    args = parser.parse_args()
    run_parameter_sweep(args.task, args.model, args.out)
