# benchmark_evolution.py
import numpy as np
import argparse
import mteb
from antigravity_engine import AntigravityEngine
from typing import Dict, List

# Metric Calculation (NDCG@10)
def dcg_at_k(r, k):
    r = np.asarray(r, dtype=float)[:k]
    if r.size:
        return np.sum(r / np.log2(np.arange(2, r.size + 2)))
    return 0.

def ndcg_at_k(r, k):
    dcg_max = dcg_at_k(sorted(r, reverse=True), k)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k) / dcg_max

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
    
    # ... (Loading Logic omitted for brevity in search replacement, assume it matches) ...
    # This tool works by replacing a block. I will target the init line and the Check block separately if needed.
    # Actually I should do it in 2 calls or 1 big block.
    # I'll update the Init first.
    pass
    
    # Load MTEB Task
    print(f"--- 2. LOADING MTEB ({task_name}) ---")
    try:
        task = mteb.get_task(task_name)
    except KeyError:
        print(f"ERROR: Task '{task_name}' not found in MTEB registry!")
        print("Available tasks can be listed with: mteb.get_tasks()")
        return
    except Exception as e:
        print(f"ERROR: Failed to load MTEB task '{task_name}': {e}")
        return

    task.load_data()
        
        # DEBUG: Print available attributes
        # print(f"Task attributes: {dir(task)}")
        
    # DEBUG: robustness
    if not hasattr(task, 'corpus'):
        print("Attribute 'corpus' missing. Attempting manual extraction from dataset...")
        # SciFact/HF dataset structure: dataset['test'] is usually a list of dicts?
        # Or a Dataset object.
        # Let's assume it is task.dataset['test'] if available.
        if 'test' in task.dataset:
            test_data = task.dataset['test']
            # MTEB HF formatting usually has 'corpus', 'queries' columns?
            # Or is it raw?
            # Actually, standard MTEB *should* map it.
            # If `mteb` is updated, `AbsTaskRetrieval` usually handles this.
            # But earlier fallback worked.
            # Let's try to just use `task.queries`, `task.corpus` assuming they MIGHT appear if we wait? No.
            
            # COPYING LOGIC FROM manual_benchmark_scifact.py EXACTLY IS SAFEST
            # BUT I need to adapt it.
            # manual_benchmark_scifact logic was:
            # data_root = find_keys(task.dataset, ['corpus', 'queries', 'relevant_docs'])
            # That implies it searched the whole tree.
            pass

    # Helper for deep search (Verified Logic)
    def find_keys(obj, target_keys):
        if not isinstance(obj, dict):
            return None
        if all(k in obj for k in target_keys):
            return obj
        for k, v in obj.items():
            found = find_keys(v, target_keys)
            if found:
                return found
        return None

    # Try to find the data root
    targets = ['corpus', 'queries'] # Start with these 2
    data_root = find_keys(task.dataset, targets)
    
    if not data_root:
        print("DEBUG: Strict data root not found. Searching individually...")
        c_payload = None
        q_payload = None
        r_payload = None
        
        def find_payload(obj, key):
            if isinstance(obj, dict):
                if key in obj: return obj[key]
                for v in obj.values():
                    res = find_payload(v, key)
                    if res: return res
            return None
            
        c_payload = find_payload(task.dataset, 'corpus')
        q_payload = find_payload(task.dataset, 'queries')
        r_payload = find_payload(task.dataset, 'relevant_docs')
        if not r_payload: r_payload = find_payload(task.dataset, 'test') # Maybe it's the split?
    else:
        c_payload = data_root.get('corpus')
        q_payload = data_root.get('queries')
        r_payload = data_root.get('relevant_docs')

    # Proceed to extraction...
    corpus = {}
    queries = {}
    qrels = {}
        
    if c_payload:
        # It might be a Dataset or dict
        # For SciFact, corpus is usually {id: {text:..., title:...}}
        # Let's try to iterate
        try:
            # If it's a dict
            for k, v in c_payload.items():
                corpus[k] = v['text'] + " " + v['title']
        except (AttributeError, TypeError):
            # If it's a HF dataset (list of rows)
            for row in c_payload:
                # Robust ID extraction
                if '_id' in row:
                    doc_id = row['_id']
                elif 'id' in row:
                    doc_id = row['id']
                else:
                    continue

                # Robust Text
                text = row.get('text', '')
                title = row.get('title', '')
                corpus[doc_id] = text + " " + title
        except Exception as e:
            print(f"ERROR: Failed to parse corpus payload: {e}")

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
                            # Maybe list of doc IDs? MTEB raw can be {qid: [did, ...]}
                            for did in docs:
                                qrels[qid][str(did)] = 1
                else:
                    # List of rows
                    for row in r_payload:
                        # Robust Query ID
                        qid = str(row.get('query-id', row.get('query_id', row.get('_id'))))
                        if not qid or qid == 'None': continue
                        
                        if qid not in qrels: qrels[qid] = {}
                        
                        # Robust Doc ID(s)
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
            
    # Final check
    if not qrels and hasattr(task, 'qrels'):
        qrels = task.qrels['test']
        
    print(f"Loaded {len(corpus)} docs, {len(queries)} queries, {len(qrels)} qrels.")
    # except Exception as e:
    #     print(f"MTEB Load Failed: {e}")
    #     import traceback
    #     traceback.print_exc()
    #     return

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
