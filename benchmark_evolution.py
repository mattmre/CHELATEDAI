# benchmark_evolution.py
import numpy as np
import argparse
import mteb
from antigravity_engine import AntigravityEngine
from typing import Dict, List

# Import shared utilities from benchmark_utils
from benchmark_utils import dcg_at_k, ndcg_at_k, find_keys, find_payload, load_mteb_data

def evaluate_ndcg(engine, queries, qrels, k=10):
    ndcg_scores = []
    
    print(f"Evaluating {len(queries)} queries...")
    for q_id, q_text in queries.items():
        if q_id not in qrels:
            continue
            
        # Run Inference
        # Returns: std_top_10, chel_top_10, mask, jaccard
        # We use the CHELATED results (Adaptive) for the score
        _, pred_ids, _, _ = engine.run_inference(q_text)
        
        # Calculate Relevance for this query
        # qrels[q_id] is {doc_id: score}
        relevant_docs = qrels[q_id]
        
        # Fix for String IDs (NFCorpus) vs Int/UUID IDs (Qdrant)
        # We need to retrieve the 'original_id' from the payload if possible
        # Since 'engine.run_inference' returns IDs, we can batch retrieve payloads to check.
        
        # Optimization: Fetch payloads for all pred_ids at once
        try:
            points = engine.qdrant.retrieve(engine.collection_name, ids=pred_ids)
            # map qdrant_id -> original_id
            id_map = {}
            for p in points:
                if p.payload and 'original_id' in p.payload:
                    id_map[p.id] = str(p.payload['original_id'])
                else:
                    id_map[p.id] = str(p.id)

            mapped_preds = [id_map.get(pid, str(pid)) for pid in pred_ids]
        except Exception as e:
            print(f"WARNING: ID mapping failed for query {q_id}: {e}, using raw IDs")
            mapped_preds = [str(pid) for pid in pred_ids]
             
        # Boolean relevance array
        relevance = [1 if pid in relevant_docs else 0 for pid in mapped_preds]
        
        # DEBUG: Check first query
        if len(ndcg_scores) == 0:
             print(f"DEBUG: QID {q_id} Preds: {[str(p) for p in pred_ids[:5]]}...")
             print(f"DEBUG: QID {q_id} RelDocs: {list(relevant_docs.keys())}")
             print(f"DEBUG: Relevance Vec: {relevance}")
        
        score = ndcg_at_k(relevance, k)
        ndcg_scores.append(score)
        
    return np.mean(ndcg_scores)

def run_evolution(task_name="SciFact", learning_rate=0.5, model_name='ollama:nomic-embed-text'):
    print("--- 1. INITIALIZING BRAIN ---")

    # Use cross-platform path handling
    from pathlib import Path
    from config import ChelationConfig

    db_path = ChelationConfig.get_db_path(task_name)
    print(f"Target Database: {db_path}")

    # Enable Adaptive Quantization (which enables Chelation logging if > threshold)
    # We set threshold low/tuned to ensure we get some chelation events for learning
    # Enable persistent storage
    engine = AntigravityEngine(
        qdrant_location=str(db_path),  # Qdrant expects string path
        model_name=model_name,
        chelation_p=85,
        use_quantization=True,
        use_centering=False
    )
    
    # Load MTEB Task
    print(f"--- 2. LOADING MTEB ({task_name}) ---")
    corpus, queries, qrels = load_mteb_data(task_name)

    if corpus is None or queries is None or qrels is None:
        print("ERROR: Failed to load task data. Aborting.")
        return

    print(f"Loaded {len(corpus)} docs, {len(queries)} queries, {len(qrels)} qrels.")

    # Ingest (skip if already done? Engine assumes persistence, but let's re-ingest to be safe/clean)
    # Actually AntigravityEngine uses 'antigravity_stage8' collection. 
    # If we want a fresh start, we should clear it or use a new name.
    # But for now, let's assume the previous ingestion is fine if it exists.
    # To be safe, let's re-ingest if the collection is empty.
    # For this script, we'll assume the USER has run the previous benchmark and data exists.
    # OR we can force ingest. Let's force ingest to ensure we are starting from State 0.
    
    corpus_list = list(corpus.values())
    ids_list = list(corpus.keys()) # We need to ensure IDs match.
    # AntigravityEngine ingest uses int IDs mostly. 
    # MTEB corpus has string IDs.
    # This is a complexity: duplicate ingestion mapping.
    # The previous benchmark mapped strings to ints for Qdrant. 
    # We should trust `manual_benchmark_scifact.py`'s ingestion or re-do it.
    
    # Let's rely on the previous ingestion to save time if possible.
    # Check if collection has data.
    info = engine.qdrant.get_collection(engine.collection_name)
    if info.points_count < len(corpus):
        print(f"Collection empty. Ingesting {len(corpus)} documents...")
        from qdrant_client.models import PointStruct
        
        batch_size = 50
        keys = list(corpus.keys())
        values = list(corpus.values())
        
        for i in range(0, len(keys), batch_size):
            batch_keys = keys[i:i+batch_size]
            batch_texts = values[i:i+batch_size]
            
            # Embed
            try:
                embeddings = engine.embed(batch_texts)

                # Upsert
                # SciFact IDs are numeric strings "123". Convert to int.
                points = []
                for k, v, t in zip(batch_keys, embeddings, batch_texts):
                    try:
                        pid = int(k)
                    except (ValueError, TypeError):
                        # Fallback if not numeric - use UUID5 for deterministic hashing
                        import uuid
                        pid = str(uuid.uuid5(uuid.NAMESPACE_DNS, str(k)))

                    points.append(PointStruct(id=pid, vector=v, payload={"text": t, "original_id": str(k)}))

                engine.qdrant.upsert(engine.collection_name, points)
                if i % 500 == 0:
                    print(f"Ingested {i + len(batch_keys)}/{len(corpus)}")
            except Exception as e:
                print(f"ERROR: Batch {i} ingestion failed: {e}")
                import traceback
                traceback.print_exc()
    else:
        print(f"Collection '{engine.collection_name}' has {info.points_count} points. Proceeding.")

    # --- ROUND 1: BASELINE ---
    print("\n--- 3. ROUND 1: BASELINE (State 0) ---")
    engine.chelation_log.clear() # Clear any old logs
    
    # DEBUG: ID Mismatch
    try:
        q_example = list(queries.keys())[0]
        r_example = list(qrels.keys())[0]
        print(f"DEBUG: Query 0 ID: '{q_example}' (Type: {type(q_example)})")
        print(f"DEBUG: QREL 0 ID: '{r_example}' (Type: {type(r_example)})")
    except (IndexError, KeyError) as e:
        print(f"DEBUG: No queries or qrels available: {e}")

    # We iterate queries manually to use `evaluate_ndcg` logic
    score_1 = evaluate_ndcg(engine, queries, qrels)
    print(f"State 0 NDCG@10: {score_1:.5f}")
    
    # Check Log Growth
    log_size = len(engine.chelation_log)
    print(f"Brain accumulated {log_size} chelation events (potential learning points).")
    
    # --- SLEEP CYCLE ---
    print("\n--- 4. SLEEP CYCLE (Sedimentation) ---")
    # Learning Rate 0.1, Threshold 1 (since we only run the query set once, any repeat is significant? 
    # Actually SciFact queries are unique. So log frequency will be 1 for each query's cluster.
    # If a cluster is hit by MULTIPLE queries, freq > 1.
    # Let's set Threshold = 1 to capture ALL chelation events for this experiment.)
    engine.run_sedimentation_cycle(threshold=1, learning_rate=learning_rate)
    
    # --- ROUND 2: POST-LEARNING ---
    print("\n--- 5. ROUND 2: POST-LEARNING (State 1) ---")
    score_2 = evaluate_ndcg(engine, queries, qrels)
    print(f"State 1 NDCG@10: {score_2:.5f}")
    
    gain = score_2 - score_1
    print(f"\nEvolutionary Gain: {gain:.5f}")
    
    if gain > 0:
        print("SUCCESS: The Brain improved its performance through homeostatic learning.")
    else:
        print("RESULT: Performance neutral or declined (expected if clusters were already optimal or over-corrected).")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Evolutionary Benchmark on MTEB Task")
    parser.add_argument("--task", type=str, default="SciFact", help="MTEB Task Name (e.g., SciFact, NFCorpus, FiQA2018)")
    parser.add_argument("--lr", type=float, default=0.5, help="Learning Rate for Sedimentation")
    parser.add_argument("--model", type=str, default="ollama:nomic-embed-text", help="Model Name (e.g., all-MiniLM-L6-v2)")
    
    args = parser.parse_args()
    
    run_evolution(args.task, args.lr, args.model)
