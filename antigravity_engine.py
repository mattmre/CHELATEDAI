import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from collections import defaultdict
from threading import Lock
import torch
import torch.optim as optim
from chelation_adapter import create_adapter
from config import ChelationConfig
from chelation_logger import get_logger
from typing import Optional
from teacher_distillation import create_distillation_helper
from sedimentation_trainer import compute_homeostatic_target, sync_vectors_to_qdrant
from checkpoint_manager import CheckpointManager, SafeTrainingContext
from embedding_backend import create_embedding_backend
from vector_store import create_vector_store

class AntigravityEngine:
    def __init__(self, qdrant_location=":memory:", chelation_p=ChelationConfig.DEFAULT_CHELATION_P, model_name='ollama:nomic-embed-text', use_centering=False, use_quantization=False, training_mode: str = "baseline", teacher_model_name: Optional[str] = None, teacher_models=None, teacher_weight: float = 0.5, store_full_text_payload: Optional[bool] = None):
        """
        Stage 8 Engine: Docker/Ollama Integration + Teacher Distillation.

        chelation_p: The 'K-Knob'.
        use_centering: If True, uses 'Spectral Chelation' (Centering).
        use_quantization: If True, enables INT8 Scalar Quantization (Layer 3).
        training_mode: 'baseline', 'offline', or 'hybrid' - controls sedimentation behavior.
        teacher_model_name: Optional single teacher model for distillation.
        teacher_models: Optional list of (model_name, weight) tuples for ensemble.
        teacher_weight: Weight for teacher guidance in hybrid mode (0.0-1.0).
        store_full_text_payload: If True, store full text in Qdrant payload (default: config default for backward compatibility).
        """
        self.chelation_p = chelation_p
        self.use_centering = use_centering
        self.use_quantization = use_quantization
        self.chelation_log = defaultdict(list)
        self.chelation_threshold = ChelationConfig.DEFAULT_CHELATION_THRESHOLD
        self.adapter_path = ChelationConfig.ADAPTER_WEIGHTS_PATH
        self.logger = get_logger()
        
        # F-040: Payload optimization control (default to config for backward compatibility)
        self.store_full_text_payload = store_full_text_payload if store_full_text_payload is not None else ChelationConfig.STORE_FULL_TEXT_PAYLOAD
        
        # Adaptive threshold state (disabled by default for backward compatibility)
        self._adaptive_threshold_enabled = False
        self._adaptive_threshold_percentile = ChelationConfig.ADAPTIVE_THRESHOLD_PERCENTILE
        self._adaptive_threshold_window = ChelationConfig.ADAPTIVE_THRESHOLD_WINDOW
        self._adaptive_threshold_min_samples = ChelationConfig.ADAPTIVE_THRESHOLD_MIN_SAMPLES
        self._adaptive_threshold_min = ChelationConfig.ADAPTIVE_THRESHOLD_MIN
        self._adaptive_threshold_max = ChelationConfig.ADAPTIVE_THRESHOLD_MAX
        self._variance_history = []  # Stores recent variance observations
        self._adaptive_threshold_lock = Lock()
        
        # Validate and store training configuration
        self.training_mode = ChelationConfig.validate_training_mode(training_mode)
        self.teacher_weight = ChelationConfig.validate_teacher_weight(teacher_weight)
        
        # Initialize teacher distillation helper (lazy loading)
        self.teacher_helper = None
        if self.training_mode in ["offline", "hybrid"]:
            self.teacher_helper = create_distillation_helper(
                teacher_model_name=teacher_model_name or ChelationConfig.DEFAULT_TEACHER_MODEL,
                teacher_models=teacher_models,
            )
            self.logger.log_event(
                "distillation_config",
                f"Training mode: {self.training_mode}, Teacher weight: {self.teacher_weight}",
                training_mode=self.training_mode,
                teacher_weight=self.teacher_weight,
            )

        # Initialize embedding backend (F-045: Extract embedding mode branching)
        self.logger.log_event("initialization", f"Initializing Antigravity Engine with model: {model_name}", model_name=model_name)
        self.embedding_backend = create_embedding_backend(model_name, self.logger)
        self.vector_size = self.embedding_backend.vector_size
        
        # Store mode and model_name for backward compatibility
        self.mode = "ollama" if model_name.startswith("ollama:") else "local"
        self.model_name = model_name.replace("ollama:", "") if model_name.startswith("ollama:") else model_name
        if self.mode == "ollama":
            self.ollama_url = ChelationConfig.OLLAMA_URL
            
        # Initialize Dynamic Adapter (Phase 2: factory-based creation)
        self.logger.log_event("adapter_init", "Initializing Dynamic Chelation Adapter")
        self.adapter = create_adapter(
            adapter_type=ChelationConfig.ADAPTER_TYPE,
            input_dim=self.vector_size,
            rank=ChelationConfig.LOW_RANK_ADAPTER_RANK
        )
        if self.adapter.load(self.adapter_path):
            self.logger.log_checkpoint("load", self.adapter_path)
        else:
            self.logger.log_event("adapter_init", "Created new adapter (Identity initialization)")
        
        # Initialize Vector Store with validation (F-044: Vector Store Abstraction)
        # Create vector store abstraction (uses Qdrant backend)
        self._vector_store = create_vector_store(
            location=qdrant_location,
            backend="qdrant",
            client_cls=QdrantClient,
        )
        
        # Backward compatibility: keep engine.qdrant access path.
        self.qdrant = self._vector_store
        
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
        
        # Initialize checkpoint manager for safe training (F-043)
        self.checkpoint_manager = CheckpointManager()
        self.logger.log_event("checkpoint_manager_init", "CheckpointManager initialized for safe training")

    def embed(self, texts):
        """Get Embeddings via backend abstraction."""
        if isinstance(texts, str):
            texts = [texts]
        
        if not texts:
            return np.array([])
        
        # Get raw embeddings from backend
        raw_embeddings = self.embedding_backend.embed_raw(texts)
        
        # For local mode, pass through adapter
        if self.mode == "local":
            with torch.no_grad():
                tensor_inputs = torch.tensor(raw_embeddings, dtype=torch.float32)
                adapted_embeddings = self.adapter(tensor_inputs)
                return adapted_embeddings.numpy()
        else:
            # Ollama mode - return raw embeddings directly (no adapter)
            return raw_embeddings

    def _sanitize_ollama_text(self, text, doc_index=None):
        """Backward-compatible Ollama text sanitizer API."""
        if hasattr(self.embedding_backend, "_sanitize_text"):
            return self.embedding_backend._sanitize_text(text, doc_index=doc_index)

        if not isinstance(text, str):
            text = str(text)
        if len(text) > ChelationConfig.OLLAMA_INPUT_MAX_CHARS:
            text = text[:ChelationConfig.OLLAMA_INPUT_MAX_CHARS]
        return "".join(c if (c.isprintable() or c in "\n\r\t") else " " for c in text)

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
            
            # F-025: Validate embed() output before PointStruct/upsert
            # Check 1: Empty result on non-empty batch
            if len(batch_texts) > 0 and len(embeddings) == 0:
                self.logger.log_error(
                    "embed_validation",
                    f"Batch {i+1}/{total_batches}: embed returned empty for non-empty batch (size={len(batch_texts)})",
                    batch_num=i+1,
                    batch_size=len(batch_texts)
                )
                continue
            
            embeddings = np.asarray(embeddings)
            
            # Check 2: Shape must be 2D [batch, dim]
            if embeddings.ndim != 2:
                self.logger.log_error(
                    "embed_validation",
                    f"Batch {i+1}/{total_batches}: embed output is not 2D (shape={embeddings.shape})",
                    batch_num=i+1,
                    shape=embeddings.shape
                )
                continue
            
            # Check 3: Number of embeddings must match number of texts
            if embeddings.shape[0] != len(batch_texts):
                self.logger.log_error(
                    "embed_validation",
                    f"Batch {i+1}/{total_batches}: embedding count mismatch (texts={len(batch_texts)}, embeddings={embeddings.shape[0]})",
                    batch_num=i+1,
                    text_count=len(batch_texts),
                    embedding_count=embeddings.shape[0]
                )
                continue
            
            # Check 4: Embedding dimension must match vector_size
            if embeddings.shape[1] != self.vector_size:
                self.logger.log_error(
                    "embed_validation",
                    f"Batch {i+1}/{total_batches}: dimension mismatch (expected={self.vector_size}, got={embeddings.shape[1]})",
                    batch_num=i+1,
                    expected_dim=self.vector_size,
                    actual_dim=embeddings.shape[1]
                )
                continue
            
            # F-040: Build payload conditionally based on store_full_text_payload flag
            points = [
                PointStruct(
                    id=i*batch_size + j,
                    vector=embeddings[j],
                    payload=({**batch_payloads[j], "text": batch_texts[j]} if self.store_full_text_payload else batch_payloads[j])
                )
                for j in range(len(batch_texts))
            ]
            self.qdrant.upsert(collection_name=self.collection_name, points=points)
            self.logger.log_event("ingestion_progress", f"Ingested batch {i+1}/{total_batches}", level="DEBUG", batch_num=i+1, total_batches=total_batches)
            
        self.logger.log_event("ingestion_complete", "Ingestion Complete")

    def ingest_streaming(self, texts_iterable, payloads_iterable=None, batch_size=None, start_id=0):
        """
        Ingest documents from iterables without loading full corpus into memory.
        
        Processes documents in batches, embedding and upserting incrementally.
        Suitable for large datasets that don't fit in memory.
        
        Args:
            texts_iterable: Iterable of text documents (e.g., generator, list, etc.)
            payloads_iterable: Optional iterable of payload dicts, aligned with texts
            batch_size: Documents per batch (defaults to config STREAMING_BATCH_SIZE)
            start_id: Starting document ID (default 0)
            
        Returns:
            Dict with ingestion statistics: {
                'total_docs': int,
                'total_batches': int,
                'start_id': int,
                'end_id': int
            }
        """
        if batch_size is None:
            batch_size = ChelationConfig.STREAMING_BATCH_SIZE
        
        progress_interval = ChelationConfig.STREAMING_PROGRESS_INTERVAL
        
        self.logger.log_event(
            "streaming_ingestion_start",
            "Starting streaming ingestion",
            batch_size=batch_size,
            start_id=start_id
        )
        
        total_docs = 0
        total_batches = 0
        current_id = start_id
        
        # Convert to iterators to support both lists and generators
        texts_iter = iter(texts_iterable)
        payloads_iter = iter(payloads_iterable) if payloads_iterable is not None else None
        
        while True:
            # Collect batch
            batch_texts = []
            batch_payloads = []
            
            for _ in range(batch_size):
                try:
                    text = next(texts_iter)
                    batch_texts.append(text)
                    
                    if payloads_iter is not None:
                        try:
                            payload = next(payloads_iter)
                            batch_payloads.append(payload)
                        except StopIteration:
                            # Payload iterator exhausted, use empty dict
                            batch_payloads.append({})
                    else:
                        batch_payloads.append({})
                        
                except StopIteration:
                    # Text iterator exhausted
                    break
            
            # Check if we have any documents to process
            if not batch_texts:
                break
            
            # Embed batch
            embeddings = self.embed(batch_texts)
            
            # F-040: Build payload conditionally based on store_full_text_payload flag
            points = [
                PointStruct(
                    id=current_id + j,
                    vector=embeddings[j],
                    payload=({**batch_payloads[j], "text": batch_texts[j]} if self.store_full_text_payload else batch_payloads[j])
                )
                for j in range(len(batch_texts))
            ]
            self.qdrant.upsert(collection_name=self.collection_name, points=points)
            
            # Update counters
            batch_doc_count = len(batch_texts)
            total_docs += batch_doc_count
            total_batches += 1
            current_id += batch_doc_count
            
            # Log progress periodically
            if total_batches % progress_interval == 0:
                self.logger.log_event(
                    "streaming_ingestion_progress",
                    f"Processed {total_batches} batches, {total_docs} documents",
                    level="DEBUG",
                    batch_num=total_batches,
                    total_docs=total_docs
                )
        
        # Log completion
        self.logger.log_event(
            "streaming_ingestion_complete",
            f"Streaming ingestion complete: {total_docs} documents in {total_batches} batches",
            total_docs=total_docs,
            total_batches=total_batches,
            start_id=start_id,
            end_id=current_id - 1
        )
        
        return {
            'total_docs': total_docs,
            'total_batches': total_batches,
            'start_id': start_id,
            'end_id': current_id - 1
        }

    def _gravity_sensor(self, query_vec, top_k=ChelationConfig.SCOUT_K):
        """Phase 1: Detects Local Curvature (Entropy) around the query."""
        try:
            # F-040: Use with_payload=False since we only need vectors here
            search_result = self.qdrant.query_points(
                collection_name=self.collection_name,
                query=query_vec,
                limit=top_k,
                with_vectors=True,
                with_payload=ChelationConfig.FETCH_PAYLOAD_ON_QUERY
            ).points
            
            if not search_result:
                return np.array([])
                
            vectors = [hit.vector for hit in search_result]
            return np.array(vectors)
        except (ResponseHandlingException, UnexpectedResponse) as e:
            self.logger.log_error("qdrant", f"Qdrant error in _gravity_sensor: {e}", exception=e)
            return np.array([])

    def _chelate_toxicity(self, local_cluster):
        """Phase 2: The Antigravity Mechanism (Binding Toxic Dimensions)."""
        if len(local_cluster) == 0:
            return np.ones(self.vector_size)

        # Phase 4: Use learned mask predictor if enabled
        mask_predictor = getattr(self, '_mask_predictor', None)
        if mask_predictor is not None:
            return mask_predictor.predict_mask(local_cluster)

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
        
        try:
            # 2. Gravity Sensor (Scout)
            # We need to find the local cluster in the EXISTING corpus.
            # This assumes the corpus has been ingested.
            # F-040: Use with_payload=False since we only need vectors here
            scout_results = self.qdrant.query_points(
                collection_name=self.collection_name,
                query=q_vec,
                limit=ChelationConfig.SCOUT_K,
                with_vectors=True,
                with_payload=ChelationConfig.FETCH_PAYLOAD_ON_QUERY
            ).points
            
            if not scout_results:
                # Fallback if index empty
                return q_vec
            
            # Extract vectors directly from scout_results (F-027: eliminated redundant retrieve())
            local_vectors = [hit.vector for hit in scout_results if hit.vector is not None]
            
            if not local_vectors:
                return q_vec
                
            local_cluster_np = np.array(local_vectors)
            
            # 3. Chelate
            mask = self._chelate_toxicity(local_cluster_np)
            
            # 4. Apply Mask
            q_chelated = q_vec * mask
            return q_chelated
        except (ResponseHandlingException, UnexpectedResponse) as e:
            self.logger.log_error("qdrant", f"Qdrant error in get_chelated_vector: {e}", exception=e)
            return q_vec
    
    def enable_adaptive_threshold(
        self,
        percentile: float = None,
        window: int = None,
        min_samples: int = None,
        min_bound: float = None,
        max_bound: float = None
    ):
        """
        Enable adaptive threshold tuning.
        
        When enabled, the chelation threshold will automatically adjust based on
        observed variance distribution during inference.
        
        Args:
            percentile: Target percentile of observed variances (default: 75)
            window: Number of recent variance samples to track (default: 100)
            min_samples: Minimum samples before adaptive adjustment (default: 20)
            min_bound: Safety lower bound for threshold (default: 0.0001)
            max_bound: Safety upper bound for threshold (default: 0.01)
        """
        with self._adaptive_threshold_lock:
            self._adaptive_threshold_enabled = True
            
            if percentile is not None:
                self._adaptive_threshold_percentile = ChelationConfig.validate_adaptive_percentile(percentile)
            if window is not None:
                self._adaptive_threshold_window = ChelationConfig.validate_adaptive_window(window)
            if min_samples is not None:
                self._adaptive_threshold_min_samples = ChelationConfig.validate_adaptive_min_samples(min_samples)
            if min_bound is not None:
                self._adaptive_threshold_min = min_bound
            if max_bound is not None:
                self._adaptive_threshold_max = max_bound

            percentile_val = self._adaptive_threshold_percentile
            window_val = self._adaptive_threshold_window
            min_samples_val = self._adaptive_threshold_min_samples
            min_bound_val = self._adaptive_threshold_min
            max_bound_val = self._adaptive_threshold_max
        
        self.logger.log_event(
            "adaptive_threshold_enabled",
            "Adaptive threshold tuning enabled",
            percentile=percentile_val,
            window=window_val,
            min_samples=min_samples_val,
            min_bound=min_bound_val,
            max_bound=max_bound_val
        )
    
    def disable_adaptive_threshold(self):
        """
        Disable adaptive threshold tuning.
        
        Resets the threshold to the configured default and clears variance history.
        """
        with self._adaptive_threshold_lock:
            self._adaptive_threshold_enabled = False
            self.chelation_threshold = ChelationConfig.DEFAULT_CHELATION_THRESHOLD
            self._variance_history.clear()
        
        self.logger.log_event(
            "adaptive_threshold_disabled",
            "Adaptive threshold tuning disabled",
            threshold_reset_to=self.chelation_threshold
        )
    
    def get_threshold_stats(self):
        """
        Get current adaptive threshold statistics.
        
        Returns:
            dict: Statistics including enabled status, current threshold, and history stats
        """
        with self._adaptive_threshold_lock:
            variance_history = list(self._variance_history)
            stats = {
                "enabled": self._adaptive_threshold_enabled,
                "current_threshold": self.chelation_threshold,
                "percentile": self._adaptive_threshold_percentile,
                "window": self._adaptive_threshold_window,
                "min_samples": self._adaptive_threshold_min_samples,
                "min_bound": self._adaptive_threshold_min,
                "max_bound": self._adaptive_threshold_max,
                "variance_samples_count": len(variance_history)
            }
        
        if variance_history:
            stats["variance_min"] = float(np.min(variance_history))
            stats["variance_max"] = float(np.max(variance_history))
            stats["variance_mean"] = float(np.mean(variance_history))
            stats["variance_median"] = float(np.median(variance_history))
        
        return stats
    
    def _update_adaptive_threshold(self, global_variance: float):
        """
        Internal helper to update threshold based on observed variance.
        
        Called during inference when adaptive mode is enabled.
        
        Args:
            global_variance: Current query's global variance value
        """
        if not self._adaptive_threshold_enabled:
            return

        old_threshold = None
        new_threshold = None
        samples_used = 0
        with self._adaptive_threshold_lock:
            if not self._adaptive_threshold_enabled:
                return

            # Add to history
            self._variance_history.append(global_variance)

            # Trim to window size
            if len(self._variance_history) > self._adaptive_threshold_window:
                self._variance_history = self._variance_history[-self._adaptive_threshold_window:]

            # Update threshold if we have enough samples
            if len(self._variance_history) >= self._adaptive_threshold_min_samples:
                candidate_threshold = np.percentile(self._variance_history, self._adaptive_threshold_percentile)

                # Clamp to safety bounds
                candidate_threshold = max(
                    self._adaptive_threshold_min,
                    min(self._adaptive_threshold_max, candidate_threshold)
                )

                if abs(candidate_threshold - self.chelation_threshold) > 1e-6:
                    old_threshold = self.chelation_threshold
                    self.chelation_threshold = candidate_threshold
                    new_threshold = candidate_threshold
                    samples_used = len(self._variance_history)

        if old_threshold is not None and new_threshold is not None:
            self.logger.log_event(
                "adaptive_threshold_update",
                f"Threshold adjusted: {old_threshold:.6f} -> {new_threshold:.6f}",
                old_threshold=old_threshold,
                new_threshold=new_threshold,
                samples_used=samples_used,
                level="DEBUG"
            )

    # ===== Phase 1: Convergence Detection =====

    def enable_convergence_detection(self, patience=None, rel_threshold=None, min_epochs=None):
        """
        Enable early stopping for training loops.

        Args:
            patience: Epochs without improvement before stopping
            rel_threshold: Minimum relative improvement to count
            min_epochs: Minimum epochs before early stopping triggers
        """
        self._convergence_enabled = True
        self._convergence_patience = patience or ChelationConfig.CONVERGENCE_PATIENCE
        self._convergence_rel_threshold = rel_threshold or ChelationConfig.CONVERGENCE_REL_THRESHOLD
        self._convergence_min_epochs = min_epochs or ChelationConfig.CONVERGENCE_MIN_EPOCHS
        self.logger.log_event(
            "convergence_enabled",
            f"Convergence detection enabled (patience={self._convergence_patience})",
            patience=self._convergence_patience,
            rel_threshold=self._convergence_rel_threshold,
            min_epochs=self._convergence_min_epochs
        )

    # ===== Phase 1: Temperature Scaling =====

    def set_temperature(self, temperature):
        """
        Set temperature scaling for spectral chelation ranking.

        Args:
            temperature: Temperature divisor for similarity scores (>0).
                        <1.0 sharpens, >1.0 softens, 1.0 = no effect.
        """
        if temperature <= 0:
            raise ValueError("Temperature must be positive")
        self._temperature = temperature
        self.logger.log_event(
            "temperature_set",
            f"Temperature scaling set to {temperature}",
            temperature=temperature
        )

    # ===== Phase 3: Online Updates =====

    def enable_online_updates(self, learning_rate=None, micro_steps=None,
                              momentum=None, max_grad_norm=None,
                              update_interval=None):
        """
        Enable inference-time online gradient updates.

        Args:
            learning_rate: Step size for micro-updates
            micro_steps: Number of gradient steps per update
            momentum: SGD momentum
            max_grad_norm: Gradient clipping threshold
            update_interval: Apply update every N queries
        """
        from online_updater import OnlineUpdater
        self._online_updater = OnlineUpdater(
            adapter=self.adapter,
            learning_rate=learning_rate or ChelationConfig.ONLINE_LEARNING_RATE,
            micro_steps=micro_steps or ChelationConfig.ONLINE_MICRO_STEPS,
            momentum=momentum or ChelationConfig.ONLINE_MOMENTUM,
            max_grad_norm=max_grad_norm or ChelationConfig.ONLINE_MAX_GRAD_NORM,
            update_interval=update_interval or ChelationConfig.ONLINE_UPDATE_INTERVAL
        )
        self.logger.log_event("online_updates_enabled", "Online gradient updates enabled")

    # ===== Teacher Weight Scheduling =====

    def enable_teacher_weight_scheduling(self, schedule="constant",
                                         initial_weight=0.5, **kwargs):
        """
        Enable dynamic teacher weight scheduling for training loops.

        Args:
            schedule: One of 'constant', 'linear_decay', 'cosine_annealing',
                      'step_decay', 'adaptive'
            initial_weight: Starting teacher weight
            **kwargs: Additional scheduler parameters (total_steps, gamma, etc.)
        """
        from teacher_weight_scheduler import TeacherWeightScheduler
        self._weight_scheduler = TeacherWeightScheduler(
            schedule=schedule, initial_weight=initial_weight, **kwargs,
        )
        self.logger.log_event(
            "weight_scheduler_enabled",
            f"Teacher weight scheduling enabled (schedule={schedule})",
            schedule=schedule,
            initial_weight=initial_weight,
        )

    # ===== Phase 5: Stability Tracking =====

    def enable_stability_tracking(self):
        """Enable structural stability metric tracking."""
        from stability_tracker import StabilityTracker
        self._stability_tracker = StabilityTracker()
        self.logger.log_event("stability_tracking_enabled", "Stability tracking enabled")

    def enable_topology_analysis(self, **kwargs):
        """
        Enable topology-aware embedding analysis.

        Args:
            **kwargs: Passed to TopologyAnalyzer constructor.
                covalent_threshold: float (default 0.90)
                hydrogen_threshold: float (default 0.70)
                vdw_threshold: float (default 0.40)
        """
        from topology_analyzer import TopologyAnalyzer
        self._topology_analyzer = TopologyAnalyzer(**kwargs)
        self.logger.log_event("topology_analysis_enabled", "Topology analysis enabled")

    def enable_isomer_detection(self, **kwargs):
        """
        Enable retrieval isomer detection.

        Args:
            **kwargs: Passed to IsomerDetector constructor.
                strength_threshold: float (default 0.3)
                top_k: int (default 10)
        """
        from isomer_detector import IsomerDetector
        self._isomer_detector = IsomerDetector(**kwargs)
        self.logger.log_event("isomer_detection_enabled", "Isomer detection enabled")

    def get_structural_health_report(self):
        """
        Get unified structural health report combining stability tracking,
        topology analysis, and isomer detection.

        Returns:
            dict with:
                - 'stability': stability report (if tracking enabled)
                - 'topology': topology report (if analysis enabled)
                - 'isomers': isomer report (if detection enabled)
                - 'health_classification': overall health (healthy/degrading/critical)
        """
        report = {}
        health_signals = []

        # Stability tracking
        stability_tracker = getattr(self, '_stability_tracker', None)
        if stability_tracker is not None:
            stability_report = stability_tracker.get_stability_report()
            report["stability"] = stability_report

            # Check stability health signals
            pcr = stability_report.get("persistent_collapse_ratio", 0.0)
            if pcr > 0.5:
                health_signals.append("critical")
            elif pcr > 0.2:
                health_signals.append("degrading")

            osc = stability_report.get("threshold_oscillation", 0.0)
            if osc > 0.005:
                health_signals.append("degrading")

        # Topology analysis
        topology_analyzer = getattr(self, '_topology_analyzer', None)
        if topology_analyzer is not None:
            # Use snapshot history for health assessment
            snapshots = topology_analyzer.get_snapshot_history()
            if len(snapshots) >= 2:
                latest = snapshots[-1]["bond_ratios"]
                previous = snapshots[-2]["bond_ratios"]
                covalent_change = latest.get("covalent", 0) - previous.get("covalent", 0)
                if covalent_change > 0.1:
                    health_signals.append("critical")
                elif covalent_change > 0.05:
                    health_signals.append("degrading")
            report["topology"] = {
                "snapshot_count": len(snapshots),
                "snapshots": snapshots,
            }

        # Isomer detection
        isomer_detector = getattr(self, '_isomer_detector', None)
        if isomer_detector is not None:
            isomer_report = isomer_detector.get_isomer_report()
            report["isomers"] = isomer_report

            # Check isomer health signals
            if isomer_report["total_detections"] > 0:
                mean_str = isomer_report["cumulative_mean_strength"]
                if mean_str > 0.6:
                    health_signals.append("critical")
                elif mean_str > 0.3:
                    health_signals.append("degrading")

        # Classify overall health
        if "critical" in health_signals:
            classification = "critical"
        elif "degrading" in health_signals:
            classification = "degrading"
        else:
            classification = "healthy"

        report["health_classification"] = classification
        return report

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
        max_entries = ChelationConfig.CHELATION_LOG_MAX_ENTRIES_PER_DOC
        for doc_id in local_ids:
            self.chelation_log[doc_id].append(center_of_mass)
            # Cap log size to prevent unbounded memory growth
            if len(self.chelation_log[doc_id]) > max_entries:
                # Keep most recent entries
                self.chelation_log[doc_id] = self.chelation_log[doc_id][-max_entries:]
        
        # 2. Shift Reference Frame
        # V_new = V_old - mu
        centered_query = query_vec - center_of_mass
        centered_candidates = local_np - center_of_mass
        
        # 3. Recalculate Similarities (vectorized)
        # Compute norms
        query_norm = np.linalg.norm(centered_query)
        candidate_norms = np.linalg.norm(centered_candidates, axis=1)
        
        # Compute dot products: shape (n_candidates,)
        dots = np.dot(centered_candidates, centered_query)
        
        # Compute cosine similarities, handling zero norms
        # Where either norm is zero, keep score at 0.0
        denominators = query_norm * candidate_norms
        scores_vec = np.zeros(len(local_ids), dtype=np.float64)
        valid = denominators != 0
        scores_vec[valid] = dots[valid] / denominators[valid]

        # Phase 1: Temperature scaling (divides scores to sharpen/soften ranking)
        temperature = getattr(self, '_temperature', ChelationConfig.DEFAULT_TEMPERATURE)
        if temperature != 1.0:
            scores_vec = scores_vec / temperature

        # Pair with IDs and sort
        scores = [(local_ids[i], scores_vec[i]) for i in range(len(local_ids))]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        sorted_ids = [s[0] for s in scores]
        return sorted_ids, center_of_mass

    def run_sedimentation_cycle(self, threshold=ChelationConfig.DEFAULT_COLLAPSE_THRESHOLD, learning_rate=ChelationConfig.DEFAULT_LEARNING_RATE, epochs=ChelationConfig.DEFAULT_EPOCHS, noise_injection=None):
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
            training_mode=self.training_mode,
            noise_injection=noise_injection
        )
        
        # Handle epochs=0 gracefully (skip training)
        if epochs == 0:
            self.logger.log_event("training_skipped", "Epochs=0, skipping training cycle")
            self.chelation_log.clear()  # Still clear the log even if no training
            return
        
        # Filter for frequent collapsers
        targets = {k: v for k, v in self.chelation_log.items() if len(v) >= threshold}
        self.logger.log_event("training_preparation", f"Found {len(targets)} collapsing node candidates", num_candidates=len(targets))

        # Phase 5: Record collapse set if stability tracking enabled
        stability_tracker = getattr(self, '_stability_tracker', None)
        if stability_tracker is not None:
            stability_tracker.record_collapse_set(list(targets.keys()))

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
        payload_map = {}  # F-031: Cache payloads during initial retrieve
        complexity_weights = [] # For noise injection scaling
        
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
                
                # F-031: Cache payload for later sync
                payload_map[point.id] = point.payload
                
                # Track complexity for noise injection scaling
                complexity_weights.append(len(noise_vectors))
                
                # Calculate Target based on mode
                if self.training_mode == "baseline":
                    # Original homeostatic push using shared helper
                    target_vec = compute_homeostatic_target(
                        current_vec, noise_vectors, ChelationConfig.HOMEOSTATIC_PUSH_MAGNITUDE
                    )
                    training_targets.append(target_vec)
                    
                elif self.training_mode == "offline":
                    # Pure teacher guidance - defer to batch processing
                    # For now, use current vec as placeholder (will be replaced)
                    training_targets.append(current_vec)
                    
                elif self.training_mode == "hybrid":
                    # Blend homeostatic + teacher - defer to batch processing
                    # Calculate homeostatic target first using shared helper
                    homeostatic_target = compute_homeostatic_target(
                        current_vec, noise_vectors, ChelationConfig.HOMEOSTATIC_PUSH_MAGNITUDE
                    )
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
                    "Teacher target generation failed, falling back to homeostatic targets",
                    exception=e
                )
                # Recompute targets using homeostatic push instead of identity
                homeostatic_targets = []
                for idx, vec in enumerate(training_inputs):
                    noise_vectors = targets[ordered_ids[idx]]
                    homeostatic_targets.append(
                        compute_homeostatic_target(
                            vec, noise_vectors, ChelationConfig.HOMEOSTATIC_PUSH_MAGNITUDE
                        )
                    )
                target_array = np.array(homeostatic_targets)
        
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
        
        # Noise injection setup
        if noise_injection is None:
            noise_injection = ChelationConfig.NOISE_INJECTION_BASE_SCALE if getattr(ChelationConfig, 'NOISE_INJECTION_ENABLED', False) else 0.0
            
        if noise_injection > 0.0:
            complexity_tensor = torch.tensor(complexity_weights, dtype=torch.float32).unsqueeze(1)
            max_complexity = complexity_tensor.max()
            if max_complexity > 0:
                complexity_tensor = complexity_tensor / max_complexity
            noise_scales = torch.clamp(complexity_tensor * noise_injection, max=ChelationConfig.NOISE_INJECTION_MAX_SCALE)
            self.logger.log_event("noise_injection", f"Enabled noise injection with base scale {noise_injection}")
        else:
            noise_scales = None
        
        # --- TRAINING LOOP (wrapped in SafeTrainingContext for F-043) ---
        self.logger.log_training_start(num_samples=len(training_inputs), learning_rate=learning_rate, epochs=epochs, threshold=threshold)
        
        with SafeTrainingContext(
            self.checkpoint_manager,
            self.adapter_path,
            f"sedimentation_cycle_threshold_{threshold}"
        ) as training_ctx:
            optimizer = optim.Adam(self.adapter.parameters(), lr=learning_rate)
            criterion = torch.nn.MSELoss()
            
            self.adapter.train()
            final_loss = 0.0

            # Phase 1: Optional convergence monitoring for early stopping
            conv_monitor = None
            if getattr(self, '_convergence_enabled', False):
                from convergence_monitor import ConvergenceMonitor
                conv_monitor = ConvergenceMonitor(
                    patience=getattr(self, '_convergence_patience', ChelationConfig.CONVERGENCE_PATIENCE),
                    rel_threshold=getattr(self, '_convergence_rel_threshold', ChelationConfig.CONVERGENCE_REL_THRESHOLD),
                    min_epochs=getattr(self, '_convergence_min_epochs', ChelationConfig.CONVERGENCE_MIN_EPOCHS)
                )

            # Optional weight scheduler for dynamic teacher weight
            weight_scheduler = getattr(self, '_weight_scheduler', None)

            for epoch in range(epochs):
                optimizer.zero_grad()
                
                if noise_scales is not None:
                    noise = torch.randn_like(input_tensor) * noise_scales
                    noisy_input = input_tensor + noise
                    noisy_input = torch.nn.functional.normalize(noisy_input, p=2, dim=1)
                    outputs = self.adapter(noisy_input)
                else:
                    outputs = self.adapter(input_tensor)
                    
                loss = criterion(outputs, target_tensor)
                loss.backward()
                optimizer.step()
                final_loss = loss.item()

                # Update teacher weight schedule if enabled
                if weight_scheduler is not None:
                    self.teacher_weight = weight_scheduler.step(final_loss)

                if epoch % max(1, epochs // 2) == 0:
                    self.logger.log_training_epoch(epoch=epoch+1, total_epochs=epochs, loss=final_loss)

                # Phase 1: Early stopping check
                if conv_monitor is not None and conv_monitor.record_loss(final_loss):
                    self.logger.log_event("early_stopping", f"Sedimentation converged at epoch {epoch+1}", epoch=epoch+1)
                    break

            self.adapter.eval()
            self.adapter.save(self.adapter_path)
            self.logger.log_checkpoint("save", self.adapter_path)

            # --- UPDATE QDRANT ---
            self.logger.log_event("vector_update", "Syncing updated vectors to Qdrant")

            # Phase 5: Record adapter snapshot after training
            if getattr(self, '_stability_tracker', None) is not None:
                self._stability_tracker.record_adapter_snapshot(self.adapter)

            with torch.no_grad():
                 new_vectors_np = self.adapter(input_tensor).numpy()

            # Batch Update Logic using shared helper (F-031: pass cached payload_map)
            total_updates, failed_updates = sync_vectors_to_qdrant(
                self.qdrant, self.collection_name, ordered_ids, 
                new_vectors_np, chunk_size, self.logger, payload_map
            )
            
            # Mark success only if no failed vector updates (F-043)
            if failed_updates == 0:
                training_ctx.mark_success()
                self.logger.log_event(
                    "sedimentation_success",
                    f"Training cycle completed successfully. Updated {total_updates} vectors.",
                    vectors_updated=total_updates
                )
            else:
                self.logger.log_error(
                    "sedimentation_partial_failure",
                    f"Training completed but {failed_updates} vector updates failed. Rolling back.",
                    vectors_updated=total_updates,
                    vectors_failed=failed_updates
                )
                    
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
        
        # Check dimension compatibility (projection handles mismatch if enabled)
        if not self.teacher_helper.check_dimension_compatibility(self.vector_size):
            if not self.teacher_helper._projection_enabled:
                self.logger.log_error(
                    "offline_distillation_dimension_mismatch",
                    "Teacher dimension mismatch and projection disabled. Cannot proceed.",
                    teacher_dim=self.teacher_helper.teacher_dim,
                    student_dim=self.vector_size,
                )
                return
            self.logger.log_event(
                "offline_distillation_projection",
                "Dimension mismatch will be handled by projection layer",
                teacher_dim=self.teacher_helper.teacher_dim,
                student_dim=self.vector_size,
            )
        
        # Fetch all corpus IDs
        try:
            all_points = []
            offset = None
            scroll_page_size = 10000
            while True:
                scroll_result = self.qdrant.scroll(
                    collection_name=self.collection_name,
                    limit=scroll_page_size,
                    with_vectors=False,
                    with_payload=True,
                    offset=offset,
                )
                points, next_offset = scroll_result
                all_points.extend(points)
                if next_offset is None or len(points) == 0:
                    break
                offset = next_offset

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

        # Phase 1: Optional convergence monitoring for early stopping
        conv_monitor = None
        if getattr(self, '_convergence_enabled', False):
            from convergence_monitor import ConvergenceMonitor
            conv_monitor = ConvergenceMonitor(
                patience=getattr(self, '_convergence_patience', ChelationConfig.CONVERGENCE_PATIENCE),
                rel_threshold=getattr(self, '_convergence_rel_threshold', ChelationConfig.CONVERGENCE_REL_THRESHOLD),
                min_epochs=getattr(self, '_convergence_min_epochs', ChelationConfig.CONVERGENCE_MIN_EPOCHS)
            )

        # Optional weight scheduler for dynamic teacher weight
        weight_scheduler = getattr(self, '_weight_scheduler', None)

        for epoch in range(ep):
            optimizer.zero_grad()
            outputs = self.adapter(input_tensor)
            loss = criterion(outputs, target_tensor)
            loss.backward()
            optimizer.step()
            final_loss = loss.item()

            # Update teacher weight schedule if enabled
            if weight_scheduler is not None:
                self.teacher_weight = weight_scheduler.step(final_loss)

            if epoch % max(1, ep // 2) == 0:
                self.logger.log_training_epoch(epoch=epoch+1, total_epochs=ep, loss=final_loss)

            # Phase 1: Early stopping check
            if conv_monitor is not None and conv_monitor.record_loss(final_loss):
                self.logger.log_event("early_stopping", f"Distillation converged at epoch {epoch+1}", epoch=epoch+1)
                break

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
        
        try:
            # B. Standard Retrieval (Scout Step)
            scout_limit = ChelationConfig.SCOUT_K
            # F-040: Use with_payload=False since we only need vectors here
            std_results = self.qdrant.query_points(
                collection_name=self.collection_name,
                query=q_vec,
                limit=scout_limit,
                with_vectors=True, # Important for Centering
                with_payload=ChelationConfig.FETCH_PAYLOAD_ON_QUERY
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
            
            # Update adaptive threshold if enabled
            self._update_adaptive_threshold(global_variance)
    
            with self._adaptive_threshold_lock:
                active_threshold = self.chelation_threshold
            
            final_top_ids = std_top
            mask = np.ones(self.vector_size)
            action = "FAST"
            
            if self.use_quantization:
                 # Adaptive Logic: Chelate if High Variance OR Forced
                 if global_variance > active_threshold or self.use_centering:
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
    
            # Phase 5: Record stability metrics if tracking enabled
            stability_tracker = getattr(self, '_stability_tracker', None)
            if stability_tracker is not None:
                stability_tracker.record_mask(mask)
                stability_tracker.record_variance_distribution(dim_variances)

            # Phase 3: Online updates if enabled
            online_updater = getattr(self, '_online_updater', None)
            if online_updater is not None and len(local_vectors) >= 4:
                mid = len(local_vectors) // 2
                top_vecs = np.array(local_vectors[:mid])
                bottom_vecs = np.array(local_vectors[mid:])
                online_updater.update(q_vec, top_vecs, bottom_vecs)

            # Log Event
            self.logger.log_query(query_text=query_text, variance=global_variance, action=action, top_ids=final_top_ids, jaccard=jaccard)

            # Return signature: std_top_10, chel_top_10, mask (dummy), jaccard
            return std_top[:10], chel_top_10, mask, jaccard
        except (ResponseHandlingException, UnexpectedResponse) as e:
            self.logger.log_error("qdrant", f"Qdrant error in run_inference: {e}", exception=e)
            return [], [], np.ones(self.vector_size), 0.0

    def close(self):
        """Close vector store and release resources (idempotent)."""
        if self._vector_store is not None:
            try:
                self._vector_store.close()
            except Exception as e:
                self.logger.log_error("resource_cleanup", f"Error closing Qdrant client: {e}", exception=e)
            finally:
                self._vector_store = None
                self.qdrant = None  # Clear backward compatibility reference

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - close resources, do not suppress exceptions."""
        self.close()
        return False  # Do not suppress exceptions
