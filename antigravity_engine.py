import os
import numpy as np
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from collections import defaultdict
import torch
import torch.optim as optim
from chelation_adapter import ChelationAdapter
from config import ChelationConfig
from chelation_logger import get_logger
from typing import Optional
from teacher_distillation import TeacherDistillationHelper, create_distillation_helper

class AntigravityEngine:
    def __init__(self, qdrant_location=":memory:", chelation_p=ChelationConfig.DEFAULT_CHELATION_P, model_name='ollama:nomic-embed-text', use_centering=False, use_quantization=False, training_mode: str = "baseline", teacher_model_name: Optional[str] = None, teacher_weight: float = 0.5):
        """
        Stage 8 Engine: Docker/Ollama Integration + Teacher Distillation.
        
        chelation_p: The 'K-Knob'.
        use_centering: If True, uses 'Spectral Chelation' (Centering).
        use_quantization: If True, enables INT8 Scalar Quantization (Layer 3).
        training_mode: 'baseline', 'offline', or 'hybrid' - controls sedimentation behavior.
        teacher_model_name: Optional teacher model for distillation (local sentence-transformers).
        teacher_weight: Weight for teacher guidance in hybrid mode (0.0-1.0).
        """
        self.chelation_p = chelation_p
        self.use_centering = use_centering
        self.use_quantization = use_quantization
        self.chelation_log = defaultdict(list)
        self.chelation_threshold = ChelationConfig.DEFAULT_CHELATION_THRESHOLD
        self.adapter_path = ChelationConfig.ADAPTER_WEIGHTS_PATH
        self.logger = get_logger()
        
        # Validate and store training configuration
        self.training_mode = ChelationConfig.validate_training_mode(training_mode)
        self.teacher_weight = ChelationConfig.validate_teacher_weight(teacher_weight)
        
        # Initialize teacher distillation helper (lazy loading)
        self.teacher_helper = None
        if self.training_mode in ["offline", "hybrid"]:
            teacher_model = teacher_model_name or ChelationConfig.DEFAULT_TEACHER_MODEL
            self.teacher_helper = create_distillation_helper(teacher_model)
            self.logger.log_event(
                "distillation_config",
                f"Training mode: {self.training_mode}, Teacher weight: {self.teacher_weight}",
                training_mode=self.training_mode,
                teacher_weight=self.teacher_weight,
                teacher_model=teacher_model
            )

        if model_name.startswith("ollama:"):
            # Ollama Mode
            self.mode = "ollama"
            self.model_name = model_name.replace("ollama:", "")
            self.ollama_url = ChelationConfig.OLLAMA_URL
            self.logger.log_event("initialization", f"Initializing Antigravity Engine (Ollama Mode: {self.model_name})", mode="ollama", model_name=self.model_name)
            # We assume the model is valid/pulled.
            self.vector_size = ChelationConfig.DEFAULT_VECTOR_SIZE
            try:
                import requests
                self.requests = requests
                test_vec = self.embed("test")[0]
                self.vector_size = len(test_vec)
                self.logger.log_event("initialization", f"Connected to Ollama. Vector Size: {self.vector_size}", vector_size=self.vector_size)
            except ImportError as e:
                raise ImportError(f"'requests' library required for Ollama mode. Install with: pip install requests") from e
            except (requests.exceptions.ConnectionError, requests.exceptions.Timeout) as e:
                raise ConnectionError(f"Failed to connect to Ollama at {self.ollama_url}. Make sure Docker container is running!") from e
            except Exception as e:
                self.logger.log_error("connection", f"Ollama connection test failed: {e}", exception=e)
                self.logger.log_event("initialization", "Vector size will be validated on first real embedding call", level="WARNING")
                # Keep default 768 as fallback
        else:
            # Local/Torch Mode
            self.mode = "local"
            self.logger.log_event("initialization", f"Initializing Antigravity Engine (Local Mode: {model_name})", mode="local", model_name=model_name)
            from sentence_transformers import SentenceTransformer
            import torch
            
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.logger.log_event("initialization", f"Device Selected: {device}", device=device)
            
            # Load model
            self.local_model = SentenceTransformer(model_name, device=device, trust_remote_code=True)
            self.vector_size = self.local_model.get_sentence_embedding_dimension()
            self.logger.log_event("initialization", f"Model loaded. Vector Size: {self.vector_size}", device=str(self.local_model.device), vector_size=self.vector_size)
            
        # Initialize Dynamic Adapter
        self.logger.log_event("adapter_init", "Initializing Dynamic Chelation Adapter")
        self.adapter = ChelationAdapter(input_dim=self.vector_size)
        if self.adapter.load(self.adapter_path):
            self.logger.log_checkpoint("load", self.adapter_path)
        else:
            self.logger.log_event("adapter_init", "Created new adapter (Identity initialization)")
        
        # Initialize Qdrant
        if qdrant_location == ":memory:" or qdrant_location.startswith("http"):
             self.qdrant = QdrantClient(location=qdrant_location)
        else:
             # Assume local path
             self.qdrant = QdrantClient(path=qdrant_location)
        self.collection_name = ChelationConfig.DEFAULT_COLLECTION_NAME
        
        # Configure Quantization
        from qdrant_client import models
        quant_config = None
        if self.use_quantization:
            self.logger.log_event("initialization", "Adaptive Quantization enabled (INT8)", quantization_enabled=True)
            quant_config = models.ScalarQuantization(
                scalar=models.ScalarQuantizationConfig(
                    type=models.ScalarType.INT8,
                    quantile=0.99,
                    always_ram=True
                )
            )
        
        # Create collection if not exists (Enable Persistence)
        if self.qdrant.collection_exists(self.collection_name):
            self.logger.log_event("collection_init", f"Loaded existing collection '{self.collection_name}'", collection_name=self.collection_name)
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
                                "options": {"num_ctx": ChelationConfig.OLLAMA_NUM_CTX}
                            },
                            timeout=ChelationConfig.OLLAMA_TIMEOUT
                        )
                        if res.status_code == 200:
                            return res.json()["embedding"]
                        else:
                            # 500 error usually means context limit or model error
                            # Return None to trigger retry logic with truncation
                            return None
                    except self.requests.exceptions.Timeout:
                        self.logger.log_error("timeout", f"Ollama timeout for doc {i}", doc_index=i)
                        return None
                    except self.requests.exceptions.ConnectionError:
                        self.logger.log_error("connection", f"Ollama connection lost for doc {i}", doc_index=i)
                        return None
                    except KeyError as e:
                        self.logger.log_error("api_response", f"Ollama response missing 'embedding' key for doc {i}", exception=e, doc_index=i)
                        return None
                    except Exception as e:
                        self.logger.log_error("embedding", f"Ollama unexpected error for doc {i}", exception=e, doc_index=i)
                        return None
                
                # Try truncation levels from OLLAMA_TRUNCATION_LIMITS
                for limit in ChelationConfig.OLLAMA_TRUNCATION_LIMITS:
                    current_text = txt[:limit]
                    emb = attempt(current_text)
                    if emb is not None:
                        break

                if emb is None:
                    self.logger.log_error("embedding_failed", f"Failed to embed doc {i} after retries", doc_index=i)
                    return i, np.zeros(self.vector_size)
                    
                return i, emb

            from concurrent.futures import ThreadPoolExecutor, TimeoutError
            with ThreadPoolExecutor(max_workers=ChelationConfig.OLLAMA_MAX_WORKERS) as executor:
                futures = [executor.submit(_get_embedding, i, txt) for i, txt in enumerate(texts)]
                for idx, future in enumerate(futures):
                    try:
                        _, emb = future.result(timeout=ChelationConfig.OLLAMA_TIMEOUT)
                        embeddings[idx] = emb
                    except TimeoutError:
                        self.logger.log_error("timeout", f"Embedding timeout for document {idx}, using zero vector", doc_index=idx)
                        embeddings[idx] = np.zeros(self.vector_size)
                    except Exception as e:
                        self.logger.log_error("embedding", f"Embedding failed for document {idx}", exception=e, doc_index=idx)
                        embeddings[idx] = np.zeros(self.vector_size)

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
        self.logger.log_event("ingestion_start", f"Ingesting {len(text_corpus)} documents", total_docs=len(text_corpus))
        
        batch_size = ChelationConfig.BATCH_SIZE
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
            self.logger.log_event("ingestion_progress", f"Ingested batch {i+1}/{total_batches}", level="DEBUG", batch_num=i+1, total_batches=total_batches)
            
        self.logger.log_event("ingestion_complete", "Ingestion Complete")

    def _gravity_sensor(self, query_vec, top_k=ChelationConfig.SCOUT_K):
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
            limit=ChelationConfig.SCOUT_K
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

    def run_sedimentation_cycle(self, threshold=ChelationConfig.DEFAULT_COLLAPSE_THRESHOLD, learning_rate=ChelationConfig.DEFAULT_LEARNING_RATE, epochs=ChelationConfig.DEFAULT_EPOCHS):
        """
        [State 2: Sleep Cycle]
        DYNAMIC UPDATE: Trains the Adapter using the collected chelation events.
        
        Supports multiple training modes:
        - baseline: Original homeostatic training (push away from noise centers)
        - offline: Teacher-guided training (align with teacher embeddings)
        - hybrid: Blend homeostatic and teacher guidance
        
        The 'Sleep Cycle' now does:
        1. Identifies vectors that were prone to 'semantic noise' (collapsing).
        2. Trains the Adapter to push these vectors AWAY from the noise centers (baseline)
           OR align with teacher embeddings (offline) OR blend both (hybrid).
        3. Persists the improved model weights.
        """
        self.logger.log_event(
            "sedimentation_start",
            f"Running sedimentation cycle (Mode={self.training_mode}, Threshold={threshold}, LR={learning_rate})",
            threshold=threshold,
            learning_rate=learning_rate,
            epochs=epochs,
            training_mode=self.training_mode
        )
        
        # Handle epochs=0 gracefully (skip training)
        if epochs == 0:
            self.logger.log_event("training_skipped", "Epochs=0, skipping training cycle")
            self.chelation_log.clear()  # Still clear the log even if no training
            return
        
        # Filter for frequent collapsers
        targets = {k: v for k, v in self.chelation_log.items() if len(v) >= threshold}
        self.logger.log_event("training_preparation", f"Found {len(targets)} collapsing node candidates", num_candidates=len(targets))
        
        if not targets:
            self.logger.log_event("training_skipped", "Brain is stable. No sedimentation needed")
            return

        # --- PREPARE TRAINING DATA ---
        batch_ids = list(targets.keys())
        
        chunk_size = ChelationConfig.CHUNK_SIZE
        training_inputs = []
        training_targets = []
        ordered_ids = []
        training_texts = []  # For teacher distillation
        
        self.logger.log_event("training_preparation", "Fetching training data from Qdrant", level="DEBUG")
        
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
                
                # Store input
                training_inputs.append(current_vec)
                
                # Get text for teacher distillation modes
                text = point.payload.get("text", "")
                training_texts.append(text)
                
                # Calculate Target based on mode
                if self.training_mode == "baseline":
                    # Original homeostatic push
                    avg_noise = np.mean(noise_vectors, axis=0)
                    diff = current_vec - avg_noise
                    diff_norm = diff / (np.linalg.norm(diff) + 1e-9)
                    target_vec = current_vec + (diff_norm * ChelationConfig.HOMEOSTATIC_PUSH_MAGNITUDE)
                    target_vec = target_vec / (np.linalg.norm(target_vec) + 1e-9)
                    training_targets.append(target_vec)
                    
                elif self.training_mode == "offline":
                    # Pure teacher guidance - defer to batch processing
                    # For now, use current vec as placeholder (will be replaced)
                    training_targets.append(current_vec)
                    
                elif self.training_mode == "hybrid":
                    # Blend homeostatic + teacher - defer to batch processing
                    # Calculate homeostatic target first
                    avg_noise = np.mean(noise_vectors, axis=0)
                    diff = current_vec - avg_noise
                    diff_norm = diff / (np.linalg.norm(diff) + 1e-9)
                    homeostatic_target = current_vec + (diff_norm * ChelationConfig.HOMEOSTATIC_PUSH_MAGNITUDE)
                    homeostatic_target = homeostatic_target / (np.linalg.norm(homeostatic_target) + 1e-9)
                    training_targets.append(homeostatic_target)  # Will be blended later
                
        if not training_inputs:
            return
        
        # Convert to numpy arrays
        input_array = np.array(training_inputs)
        target_array = np.array(training_targets)
        
        # Apply teacher distillation if needed
        if self.training_mode == "offline" and self.teacher_helper:
            self.logger.log_event("distillation_teacher_targets", "Generating pure teacher targets")
            try:
                target_array = self.teacher_helper.generate_distillation_targets(
                    texts=training_texts,
                    current_embeddings=input_array,
                    teacher_weight=1.0  # Pure teacher
                )
            except Exception as e:
                self.logger.log_error(
                    "distillation_failed",
                    "Teacher target generation failed, falling back to baseline",
                    exception=e
                )
                # Fallback to baseline targets (already computed)
                pass
        
        elif self.training_mode == "hybrid" and self.teacher_helper:
            self.logger.log_event(
                "distillation_hybrid_targets",
                f"Blending homeostatic and teacher targets (teacher_weight={self.teacher_weight})"
            )
            try:
                # target_array currently contains homeostatic targets
                homeostatic_targets = target_array.copy()
                
                # Get teacher embeddings
                teacher_embeds = self.teacher_helper.get_teacher_embeddings(training_texts)
                
                if teacher_embeds.shape == homeostatic_targets.shape:
                    # Blend: target = (1 - alpha) * homeostatic + alpha * teacher
                    alpha = self.teacher_weight
                    blended = (1 - alpha) * homeostatic_targets + alpha * teacher_embeds
                    # Normalize
                    norms = np.linalg.norm(blended, axis=1, keepdims=True)
                    norms = np.maximum(norms, 1e-9)
                    target_array = blended / norms
                else:
                    self.logger.log_error(
                        "hybrid_shape_mismatch",
                        f"Shape mismatch: homeostatic {homeostatic_targets.shape} vs teacher {teacher_embeds.shape}",
                        homeostatic_shape=homeostatic_targets.shape,
                        teacher_shape=teacher_embeds.shape
                    )
            except Exception as e:
                self.logger.log_error(
                    "hybrid_blend_failed",
                    "Hybrid blending failed, using homeostatic only",
                    exception=e
                )

        # Normalize Data
        input_tensor = torch.tensor(input_array, dtype=torch.float32)
        target_tensor = torch.tensor(target_array, dtype=torch.float32)
        
        # --- TRAINING LOOP ---
        self.logger.log_training_start(num_samples=len(training_inputs), learning_rate=learning_rate, epochs=epochs, threshold=threshold)
        optimizer = optim.Adam(self.adapter.parameters(), lr=learning_rate)
        criterion = torch.nn.MSELoss()
        
        self.adapter.train()
        final_loss = 0.0
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.adapter(input_tensor)
            loss = criterion(outputs, target_tensor)
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

            if epoch % max(1, epochs // 2) == 0:
                self.logger.log_training_epoch(epoch=epoch+1, total_epochs=epochs, loss=final_loss)
        self.adapter.eval()
        self.adapter.save(self.adapter_path)
        self.logger.log_checkpoint("save", self.adapter_path)
        
        # --- UPDATE QDRANT ---
        self.logger.log_event("vector_update", "Syncing updated vectors to Qdrant")
        
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
                self.logger.log_error("database_update", f"Invalid vector data in batch {i//chunk_size}", exception=e, batch_num=i//chunk_size)
                failed_updates += len(chunk_ids)
            except Exception as e:
                self.logger.log_error("database_update", f"Update batch {i//chunk_size} failed", exception=e, batch_num=i//chunk_size)
                failed_updates += len(chunk_ids)
                
        self.logger.log_training_complete(final_loss=final_loss, vectors_updated=total_updates, vectors_failed=failed_updates)
        self.chelation_log.clear()
    
    def run_offline_distillation(self, batch_size: int = 100, learning_rate: float = None, epochs: int = None):
        """
        Run explicit offline teacher distillation on the entire corpus.
        
        This method trains the adapter to align ALL corpus embeddings with teacher embeddings,
        independent of query-time chelation events. Useful for pre-training or warm-starting
        the adapter with teacher knowledge.
        
        Args:
            batch_size: Number of documents to process per batch
            learning_rate: Optional learning rate (uses config default if None)
            epochs: Optional epoch count (uses config default if None)
        """
        if self.teacher_helper is None:
            self.logger.log_error(
                "offline_distillation_unavailable",
                "Teacher helper not initialized. Set training_mode='offline' or 'hybrid' during engine init.",
                training_mode=self.training_mode
            )
            return
        
        lr = learning_rate or ChelationConfig.DEFAULT_OFFLINE_LEARNING_RATE
        ep = epochs or ChelationConfig.DEFAULT_OFFLINE_EPOCHS
        
        if ep == 0:
            self.logger.log_event("offline_distillation_skipped", "Epochs=0, skipping offline distillation")
            return
        
        self.logger.log_event(
            "offline_distillation_start",
            f"Starting offline distillation (LR={lr}, Epochs={ep}, Batch={batch_size})",
            learning_rate=lr,
            epochs=ep,
            batch_size=batch_size
        )
        
        # Check dimension compatibility
        if not self.teacher_helper.check_dimension_compatibility(self.vector_size):
            self.logger.log_error(
                "offline_distillation_dimension_mismatch",
                f"Teacher dimension mismatch. Cannot proceed with offline distillation.",
                teacher_dim=self.teacher_helper.teacher_dim,
                student_dim=self.vector_size
            )
            return
        
        # Fetch all corpus IDs
        try:
            scroll_result = self.qdrant.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Reasonable limit per scroll
                with_vectors=False,
                with_payload=True
            )
            all_points = scroll_result[0]
            
            if not all_points:
                self.logger.log_event("offline_distillation_empty", "No documents in corpus, nothing to distill")
                return
            
            self.logger.log_event(
                "offline_distillation_corpus",
                f"Found {len(all_points)} documents in corpus",
                corpus_size=len(all_points)
            )
        except Exception as e:
            self.logger.log_error(
                "offline_distillation_scroll_failed",
                "Failed to retrieve corpus for offline distillation",
                exception=e
            )
            return
        
        # Process in batches
        training_inputs = []
        training_targets = []
        ordered_ids = []
        
        for i in range(0, len(all_points), batch_size):
            batch_points = all_points[i:i+batch_size]
            batch_ids = [p.id for p in batch_points]
            batch_texts = [p.payload.get("text", "") for p in batch_points]
            
            # Retrieve vectors for this batch
            try:
                points_with_vectors = self.qdrant.retrieve(
                    collection_name=self.collection_name,
                    ids=batch_ids,
                    with_vectors=True
                )
                
                current_vecs = [np.array(p.vector) for p in points_with_vectors]
                
                # Generate teacher targets
                target_vecs = self.teacher_helper.generate_distillation_targets(
                    texts=batch_texts,
                    current_embeddings=np.array(current_vecs),
                    teacher_weight=1.0  # Pure teacher guidance for offline mode
                )
                
                training_inputs.extend(current_vecs)
                training_targets.extend(target_vecs)
                ordered_ids.extend(batch_ids)
                
                self.logger.log_event(
                    "offline_distillation_batch",
                    f"Processed batch {i//batch_size + 1}",
                    level="DEBUG",
                    batch_num=i//batch_size + 1,
                    batch_size=len(batch_ids)
                )
                
            except Exception as e:
                self.logger.log_error(
                    "offline_distillation_batch_failed",
                    f"Failed to process batch {i//batch_size + 1}",
                    exception=e,
                    batch_num=i//batch_size + 1
                )
                continue
        
        if not training_inputs:
            self.logger.log_event("offline_distillation_no_data", "No training data generated")
            return
        
        # Convert to tensors
        input_tensor = torch.tensor(np.array(training_inputs), dtype=torch.float32)
        target_tensor = torch.tensor(np.array(training_targets), dtype=torch.float32)
        
        # Train adapter
        self.logger.log_training_start(
            num_samples=len(training_inputs),
            learning_rate=lr,
            epochs=ep,
            threshold=0
        )
        
        optimizer = optim.Adam(self.adapter.parameters(), lr=lr)
        criterion = torch.nn.MSELoss()
        
        self.adapter.train()
        final_loss = 0.0
        
        for epoch in range(ep):
            optimizer.zero_grad()
            outputs = self.adapter(input_tensor)
            loss = criterion(outputs, target_tensor)
            loss.backward()
            optimizer.step()
            final_loss = loss.item()
            
            if epoch % max(1, ep // 2) == 0:
                self.logger.log_training_epoch(epoch=epoch+1, total_epochs=ep, loss=final_loss)
        
        self.adapter.eval()
        self.adapter.save(self.adapter_path)
        self.logger.log_checkpoint("save", self.adapter_path)
        
        # Update vectors in Qdrant
        self.logger.log_event("offline_distillation_update", "Updating corpus vectors in Qdrant")
        
        with torch.no_grad():
            new_vectors_np = self.adapter(input_tensor).numpy()
        
        chunk_size = ChelationConfig.CHUNK_SIZE
        total_updates = 0
        failed_updates = 0
        
        for i in range(0, len(ordered_ids), chunk_size):
            chunk_ids = ordered_ids[i:i+chunk_size]
            chunk_vectors = new_vectors_np[i:i+chunk_size]
            
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
            except Exception as e:
                self.logger.log_error(
                    "offline_distillation_update_failed",
                    f"Failed to update batch {i//chunk_size}",
                    exception=e,
                    batch_num=i//chunk_size
                )
                failed_updates += len(chunk_ids)
        
        self.logger.log_training_complete(
            final_loss=final_loss,
            vectors_updated=total_updates,
            vectors_failed=failed_updates
        )

    def run_inference(self, query_text):
        """Full Navigational Loop (returns IDs)."""
        
        # A. Embed
        q_vec = self.embed(query_text)[0]
        
        # B. Standard Retrieval (Scout Step)
        scout_limit = ChelationConfig.SCOUT_K
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
        
        # Format Return
        chel_top_10 = final_top_ids[:10]

        # Impact: Jaccard
        s1 = set(std_top[:10])
        s2 = set(chel_top_10)
        if len(s1) == 0 and len(s2) == 0:
            jaccard = 1.0
        else:
            jaccard = len(s1.intersection(s2)) / len(s1.union(s2))

        # Log Event
        self.logger.log_query(query_text=query_text, variance=global_variance, action=action, top_ids=final_top_ids, jaccard=jaccard)

        # Return signature: std_top_10, chel_top_10, mask (dummy), jaccard
        return std_top[:10], chel_top_10, mask, jaccard
