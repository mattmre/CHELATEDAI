import os
import numpy as np
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from collections import defaultdict
import torch
import torch.optim as optim
from chelation_adapter import ChelationAdapter

class AntigravityEngine:
    def __init__(self, qdrant_location=":memory:", chelation_p=80, model_name='ollama:nomic-embed-text', use_centering=False, use_quantization=False):
        """
        Stage 8 Engine: Docker/Ollama Integration.
        
        chelation_p: The 'K-Knob'.
        use_centering: If True, uses 'Spectral Chelation' (Centering).
        use_quantization: If True, enables INT8 Scalar Quantization (Layer 3).
        """
        self.chelation_p = chelation_p
        self.use_centering = use_centering
        self.use_quantization = use_quantization
        self.event_log_path = "chelation_events.jsonl"
        self.chelation_log = defaultdict(list)
        self.chelation_threshold = 0.0004 # Tuned to observed SciFact variance mean
        self.adapter_path = "adapter_weights.pt"

        
        if model_name.startswith("ollama:"):
            # Ollama Mode
            self.mode = "ollama"
            self.model_name = model_name.replace("ollama:", "")
            self.ollama_url = "http://localhost:11434/api/embeddings"
            print(f"Initializing Antigravity Engine (Ollama Mode: {self.model_name})...")
            # We assume the model is valid/pulled.
            self.vector_size = 768 # Default start to prevent crash in embed()
            try:
                import requests
                self.requests = requests
                test_vec = self.embed("test")[0]
                self.vector_size = len(test_vec)
                print(f"Connected to Ollama. Vector Size: {self.vector_size}")
            except ImportError as e:
                raise ImportError(f"'requests' library required for Ollama mode. Install with: pip install requests") from e
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                raise ConnectionError(f"Failed to connect to Ollama at {self.ollama_url}. Make sure Docker container is running!") from e
            except Exception as e:
                print(f"WARNING: Ollama connection test failed: {e}")
                print("Vector size will be validated on first real embedding call.")
                # Keep default 768 as fallback
        else:
            # Local/Torch Mode
            self.mode = "local"
            print(f"Initializing Antigravity Engine (Local Mode: {model_name})...")
            from sentence_transformers import SentenceTransformer
            import torch
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            print(f"Device Selected: {device}")
            
            # Load model
            self.local_model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
            self.vector_size = self.local_model.get_sentence_embedding_dimension()
            print(f"Model loaded on {self.local_model.device}. Vector Size: {self.vector_size}")
            
        # Initialize Dynamic Adapter
        print("Initializing Dynamic Chelation Adapter...")
        self.adapter = ChelationAdapter(input_dim=self.vector_size)
        if self.adapter.load(self.adapter_path):
            print(f"Loaded existing adapter weights from {self.adapter_path}")
        else:
            print("Created new adapter (Identity initialization).")
        
        # Initialize Qdrant
        if qdrant_location == ":memory:" or qdrant_location.startswith("http"):
             self.qdrant = QdrantClient(location=qdrant_location)
        else:
             # Assume local path
             self.qdrant = QdrantClient(path=qdrant_location)
        self.collection_name = "antigravity_stage8"
        
        # Configure Quantization
        from qdrant_client import models
        quant_config = None
        if self.use_quantization:
            print("!!! ADAPTIVE QUANTIZATION ENABLED (INT8) !!!")
            quant_config = models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True
                )
            )
        
        # Create collection if not exists (Enable Persistence)
        if self.qdrant.collection_exists(self.collection_name):
            print(f"Loaded existing collection '{self.collection_name}'.")
        else:
            self.qdrant.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE),
                quantization_config=quant_config
            )

    def embed(self, texts):
        """Get Embeddings (Ollama or Local)."""
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([])
            
        if self.mode == "ollama":
            embeddings = [None] * len(texts)
            
            def _get_embedding(i, txt):
                # Helper to attempt embedding
                def attempt(t):
                    try:
                        res = self.requests.post(
                            self.ollama_url,
                            json={
                                "model": self.model_name,
                                "prompt": t,
                                "options": {"num_ctx": 4096}  # Hint larger context if possible
                            },
                            timeout=30  # 30 second timeout per request
                        )
                        if res.status_code == 200:
                            return res.json()["embedding"]
                        else:
                            # 500 error usually means context limit or model error
                            # Return None to trigger retry logic with truncation
                            return None
                    except self.requests.exceptions.Timeout:
                        print(f"Ollama timeout for doc {i}")
                        return None
                    except self.requests.exceptions.ConnectionError:
                        print(f"Ollama connection lost for doc {i}")
                        return None
                    except KeyError as e:
                        print(f"Ollama response missing 'embedding' key for doc {i}: {e}")
                        return None
                    except Exception as e:
                        print(f"Ollama unexpected error for doc {i}: {e}")
                        return None
                
                # 1. Try with reasonable limit (6000 chars ~ 1500 tokens)
                current_text = txt[:6000] if len(txt) > 6000 else txt
                emb = attempt(current_text)
                
                # 2. Retry with Aggressive Truncation (2000 chars ~ 500 tokens)
                if emb is None:
                    # print(f"DEBUG: Retrying doc {i} with aggressive truncation...")
                    emb = attempt(txt[:2000])
                
                # 3. Retry with Extreme Truncation (500 chars)
                if emb is None:
                     emb = attempt(txt[:500])

                if emb is None:
                    print(f"Failed to embed doc {i} after retries.")
                    return i, np.zeros(self.vector_size)
                    
                return i, emb

            from concurrent.futures import ThreadPoolExecutor, TimeoutError
            # Reduced concurrency (10 -> 2) for stability and to avoid overwhelming Ollama
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(_get_embedding, i, txt) for i, txt in enumerate(texts)]
                for future in futures:
                    try:
                        i, emb = future.result(timeout=30)  # 30 second timeout per embedding
                        embeddings[i] = emb
                    except TimeoutError:
                        print(f"WARNING: Embedding timeout for document {i}, using zero vector")
                        embeddings[i] = np.zeros(self.vector_size)
                    except Exception as e:
                        print(f"ERROR: Embedding failed for document {i}: {e}")
                        embeddings[i] = np.zeros(self.vector_size)

            return np.array(embeddings)
            
        elif self.mode == "local":
            raw_embeddings = self.local_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
            
            # PASS THROUGH ADAPTER
            # We need to convert to torch tensor, pass through, then back to numpy
            with torch.no_grad():
                tensor_inputs = torch.tensor(raw_embeddings, dtype=torch.float32)
                adapted_embeddings = self.adapter(tensor_inputs)
                return adapted_embeddings.numpy()

    def ingest(self, text_corpus, payloads=None):
        """Ingests real-world documents into Qdrant."""
        print(f"Ingesting {len(text_corpus)} documents...")
        
        batch_size = 100
        total_batches = (len(text_corpus) + batch_size - 1) // batch_size
        
        for i in range(total_batches):
            batch_texts = text_corpus[i*batch_size : (i+1)*batch_size]
            batch_payloads = payloads[i*batch_size : (i+1)*batch_size] if payloads else [{}] * len(batch_texts)
            
            # Embed
            embeddings = self.embed(batch_texts)
            
            # Upsert to Qdrant
            points = [
                PointStruct(
                    id=i*batch_size + j,
                    vector=embeddings[j],
                    payload={"text": batch_texts[j], **batch_payloads[j]}
                )
                for j in range(len(batch_texts))
            ]
            self.qdrant.upsert(collection_name=self.collection_name, points=points)
            print(f"Ingested batch {i+1}/{total_batches} ({batch_size*(i+1)} docs)")
            
        print("Ingestion Complete.")

    def _gravity_sensor(self, query_vec, top_k=50):
        """Phase 1: Detects Local Curvature (Entropy) around the query."""
        # Use query_points instead of search
        search_result = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=query_vec,
            limit=top_k,
            with_vectors=True 
        ).points
        
        if not search_result:
            return np.array([])
            
        vectors = [hit.vector for hit in search_result]
        return np.array(vectors)

    def _chelate_toxicity(self, local_cluster):
        """Phase 2: The Antigravity Mechanism (Binding Toxic Dimensions)."""
        if len(local_cluster) == 0:
            return np.ones(self.vector_size)

        # Calculate Variance per dimension
        dim_variance = np.var(local_cluster, axis=0)
        
        if hasattr(self, 'invert_chelation') and self.invert_chelation:
            # INVERTED: Keep Top P% (High Variance), Mask Bottom (Noise)
            # Threshold at (100 - P)th percentile
            threshold = np.percentile(dim_variance, 100 - self.chelation_p)
            mask = (dim_variance > threshold).astype(float)
        else:
            # ORIGINAL: Keep Bottom P% (Stable), Mask Top (Toxic)
            threshold = np.percentile(dim_variance, self.chelation_p)
            mask = (dim_variance < threshold).astype(float)
        
        return mask

    def get_chelated_vector(self, query_text):
        """
        Returns the chelated (masked) query vector for external benchmarking (MTEB).
        """
        # 1. Embed Query
        q_vec = self.embed(query_text)[0]
        
        # 2. Gravity Sensor (Scout)
        # We need to find the local cluster in the EXISTING corpus.
        # This assumes the corpus has been ingested.
        scout_results = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=q_vec,
            limit=50 
        ).points
        
        if not scout_results:
            # Fallback if index empty
            return q_vec
            
        local_cluster_ids = [hit.id for hit in scout_results]
        
        # We need the actual VECTORS of the local cluster to calculate variance.
        # Qdrant: retrieve points by ID.
        points = self.qdrant.retrieve(
            collection_name=self.collection_name,
            ids=local_cluster_ids,
            with_vectors=True
        )
        local_vectors = [p.vector for p in points]
        
        if not local_vectors:
            return q_vec
            
        local_cluster_np = np.array(local_vectors)
        
        # 3. Chelate
        mask = self._chelate_toxicity(local_cluster_np)
        
        # 4. Apply Mask
        q_chelated = q_vec * mask
        return q_chelated

    def _cosine_similarity_manual(self, vec1, vec2):
        """Calculates Cosine Similarity manually."""
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)

    def _spectral_chelation_ranking(self, query_vec, local_vectors, local_ids):
        """
        Stage 6: Spectral Chelation (Center of Mass).
        1. Calculate Mean Vector of top results.
        2. Subtract Mean from Query and Candidates.
        3. Rerank using centered vectors.
        """
        if not local_vectors:
            return local_ids, []

        local_np = np.array(local_vectors)
        
        # 1. Center of Mass
        center_of_mass = np.mean(local_np, axis=0)
        
        # [HOMEOSTATIC LOGGING]
        # Record that these documents participated in a cluster with this 'Noise Center'
        # We only log if this is a "dense" cluster (checking variance is done in caller, so assume yes)
        for doc_id in local_ids:
            self.chelation_log[doc_id].append(center_of_mass)
        
        # 2. Shift Reference Frame
        # V_new = V_old - mu
        centered_query = query_vec - center_of_mass
        centered_candidates = local_np - center_of_mass
        
        # 3. Recalculate Similarities
        scores = []
        for i in range(len(centered_candidates)):
            score = self._cosine_similarity_manual(centered_query, centered_candidates[i])
            scores.append((local_ids[i], score))
            
        # Sort by score descending
        scores.sort(key=lambda x: x[1], reverse=True)
        
        sorted_ids = [s[0] for s in scores]
        return sorted_ids, center_of_mass

    def run_sedimentation_cycle(self, threshold=3, learning_rate=0.001, epochs=10):
        """
        [State 2: Sleep Cycle]
        DYNAMIC UPDATE: Trains the Adapter using the collected chelation events.
        
        The 'Sleep Cycle' now does two things:
        1. Identifies vectors that were prone to 'semantic noise' (collapsing).
        2. Trains the Adapter to push these vectors AWAY from the noise centers.
        3. Persists the improved model weights.
        """
        print(f"\n--- RUNNING SEDIMENTATION CYCLE (Threshold={threshold}, LR={learning_rate}, ALL_MODES_ACTIVE) ---")
        
        # Filter for frequent collapsers
        targets = {k: v for k, v in self.chelation_log.items() if len(v) >= threshold}
        print(f"Identifying collapsing nodes... Found {len(targets)} candidates.")
        
        if not targets:
            print("Brain is stable. No sedimentation needed.")
            return

        # --- PREPARE TRAINING DATA ---
        # input: The current vector of the collapsing node
        # target: The vector pushed away from the noise center
        # Loss: MSE(Adapter(input), target)
        
        batch_ids = list(targets.keys())
        
        # --- PREPARE DATA WITH ID TRACKING ---
        chunk_size = 100
        training_inputs = []
        training_targets = []
        ordered_ids = [] # To map outputs back to IDs for update
        
        print("Fetching training data from Qdrant...")
        
        for i in range(0, len(batch_ids), chunk_size):
            chunk = batch_ids[i:i+chunk_size]
            points = self.qdrant.retrieve(
                collection_name=self.collection_name,
                ids=chunk,
                with_vectors=True
            )
            
            for point in points:
                ordered_ids.append(point.id)
                noise_vectors = targets[point.id]
                current_vec = np.array(point.vector)
                
                # Calculate Target
                avg_noise = np.mean(noise_vectors, axis=0)
                diff = current_vec - avg_noise
                diff_norm = diff / (np.linalg.norm(diff) + 1e-9)
                target_vec = current_vec + (diff_norm * 0.1) # Moderate push
                target_vec = target_vec / np.linalg.norm(target_vec)
                
                training_inputs.append(current_vec)
                training_targets.append(target_vec)
                
        if not training_inputs:
            return

        # Normalize Data
        input_tensor = torch.tensor(np.array(training_inputs), dtype=torch.float32)
        target_tensor = torch.tensor(np.array(training_targets), dtype=torch.float32)
        
        # --- TRAINING LOOP ---
        print(f"Training Adapter on {len(training_inputs)} samples for {epochs} epochs...")
        optimizer = optim.Adam(self.adapter.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()
        
        self.adapter.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.adapter(input_tensor)
            loss = criterion(outputs, target_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % (epochs // 2 or 1) == 0:
                print(f"Epoch {epoch+1}/{epochs} Loss: {loss.item():.6f}")
                
        self.adapter.eval()
        self.adapter.save(self.adapter_path)
        print("Adapter weights saved.")
        
        # --- UPDATE QDRANT ---
        print("Syncing updated vectors to Qdrant...")
        
        with torch.no_grad():
             new_vectors_np = self.adapter(input_tensor).numpy()
             
        # Batch Update Logic
        total_updates = 0
        failed_updates = 0

        for i in range(0, len(ordered_ids), chunk_size):
            chunk_ids = ordered_ids[i:i+chunk_size]
            chunk_vectors = new_vectors_np[i:i+chunk_size]

            # Fetch existing points to get payloads (preserve metadata)
            try:
                existing_points = self.qdrant.retrieve(
                    collection_name=self.collection_name,
                    ids=chunk_ids,
                    with_vectors=False
                )
                payload_map = {p.id: p.payload for p in existing_points}

                batch_points = []
                for j, doc_id in enumerate(chunk_ids):
                    vec = chunk_vectors[j].tolist()
                    pay = payload_map.get(doc_id, {})

                    batch_points.append(PointStruct(
                        id=doc_id,
                        vector=vec,
                        payload=pay
                    ))

                if batch_points:
                    self.qdrant.upsert(
                        collection_name=self.collection_name,
                        points=batch_points
                    )
                    total_updates += len(batch_points)
            except ValueError as e:
                print(f"ERROR: Invalid vector data in batch {i//chunk_size}: {e}")
                failed_updates += len(chunk_ids)
            except Exception as e:
                print(f"ERROR: Update batch {i//chunk_size} failed: {e}")
                failed_updates += len(chunk_ids)
                
        if failed_updates > 0:
            print(f"Sedimentation Complete with ERRORS. Adapter trained on {len(training_inputs)} events. Updated {total_updates} vectors, {failed_updates} failed.")
        else:
            print(f"Sedimentation Complete. Adapter trained on {len(training_inputs)} events. Updated {total_updates} vectors in DB.")
        self.chelation_log.clear()
        print("--- SLEEP CYCLE COMPLETE ---")

    def _log_event(self, query_text, variance, action, top_ids):
        """
        Logs chelation events to a JSONL file for analysis and debugging.

        Args:
            query_text (str): The query text
            variance (float): Global variance metric
            action (str): Decision made ('FAST', 'CHELATE', 'CHELATE_ALWAYS')
            top_ids (list): List of top document IDs returned
        """
        import json
        import time

        event = {
            "timestamp": time.time(),
            "query_snippet": query_text[:50] if isinstance(query_text, str) else str(query_text)[:50],
            "global_variance": float(variance),
            "action": action,  # 'FAST', 'CHELATE', or 'CHELATE_ALWAYS'
            "top_10_ids": [str(d) for d in top_ids[:10]]
        }

        try:
            with open(self.event_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except IOError as e:
            print(f"ERROR: Failed to write event log to {self.event_log_path}: {e}")
        except TypeError as e:
            print(f"ERROR: Invalid data type in event log: {e}")
        except Exception as e:
            print(f"ERROR: Unexpected logging failure: {e}")

    def run_inference(self, query_text):
        """Full Navigational Loop (returns IDs)."""
        
        # A. Embed
        q_vec = self.embed(query_text)[0]
        
        # B. Standard Retrieval (Scout Step)
        scout_limit = 50
        std_results = self.qdrant.query_points(
            collection_name=self.collection_name,
            query=q_vec,
            limit=scout_limit,
            with_vectors=True # Important for Centering
        ).points
        
        std_top = [hit.id for hit in std_results]
        
        if not std_results:
             return [], [], np.ones(self.vector_size), 0.0

        # C. Processing Logic
        local_vectors = [hit.vector for hit in std_results]
        local_cluster_np = np.array(local_vectors)
        
        # Calculate Global Variance (Entropy Metric)
        # Sum of variances of all dimensions? Or just mean variance?
        # Let's use simple mean variance for now as "K"
        dim_variances = np.var(local_cluster_np, axis=0)
        global_variance = np.mean(dim_variances)
        
        final_top_ids = std_top
        mask = np.ones(self.vector_size)
        action = "FAST"
        
        if self.use_quantization:
             # Adaptive Logic: Chelate if High Variance OR Forced
             if global_variance > self.chelation_threshold or self.use_centering:
                  # STAGE 6: Spectral Chelation (Precision Path)
                  action = "CHELATE"
                  # This method now handles the `chelation_log` update implicitly
                  chel_top, center_of_mass = self._spectral_chelation_ranking(q_vec, local_vectors, std_top)
                  final_top_ids = chel_top
             else:
                  # Variance is Low -> Trust the Quantized Scout (Fast)
                  action = "FAST"
                  final_top_ids = std_top
        elif self.use_centering:
             # If quantization is off but centering is on, always chelate (Old verification path)
             action = "CHELATE_ALWAYS"
             chel_top, center_of_mass = self._spectral_chelation_ranking(q_vec, local_vectors, std_top)
             final_top_ids = chel_top
        
        # Log Event
        self._log_event(query_text, global_variance, action, final_top_ids)
        
        # Format Return
        chel_top_10 = final_top_ids[:10]
        
        # Impact: Jaccard
        s1 = set(std_top[:10])
        s2 = set(chel_top_10)
        if len(s1) == 0 and len(s2) == 0:
            jaccard = 1.0
        else:
            jaccard = len(s1.intersection(s2)) / len(s1.union(s2))
            
        # Return signature: std_top_10, chel_top_10, mask (dummy), jaccard
        return std_top[:10], chel_top_10, mask, jaccard
