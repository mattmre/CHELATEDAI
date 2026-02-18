"""
Antigravity Components Module

Core chelation logic extracted from AntigravityEngine for better modularity.
Implements gravity sensor query orchestration, toxicity masking, and spectral chelation ranking.

Finding F-046: Scoped decomposition of AntigravityEngine chelation logic.
"""

import numpy as np
from collections import defaultdict
from config import ChelationConfig
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse


class ChelationComponents:
    """
    Core chelation logic components for AntigravityEngine.
    
    This class encapsulates the internal chelation mechanics:
    - Gravity sensor query orchestration
    - Toxicity mask computation
    - Spectral chelation ranking computation
    """
    
    def __init__(self, qdrant_client, collection_name, vector_size, chelation_p, 
                 logger, chelation_log, invert_chelation=False):
        """
        Initialize chelation components.
        
        Args:
            qdrant_client: QdrantClient instance for vector queries
            collection_name: Name of the Qdrant collection
            vector_size: Dimensionality of vectors
            chelation_p: Percentile threshold for chelation masking
            logger: ChelationLogger instance for event logging
            chelation_log: Shared chelation log dictionary (defaultdict)
            invert_chelation: If True, keep high-variance dimensions (inverted mode)
        """
        self.qdrant = qdrant_client
        self.collection_name = collection_name
        self.vector_size = vector_size
        self.chelation_p = chelation_p
        self.logger = logger
        self.chelation_log = chelation_log
        self.invert_chelation = invert_chelation
    
    def gravity_sensor(self, query_vec, top_k=ChelationConfig.SCOUT_K):
        """
        Phase 1: Detects Local Curvature (Entropy) around the query.
        
        Queries the vector store to find nearby vectors that define the local cluster.
        
        Args:
            query_vec: Query vector (numpy array)
            top_k: Number of neighbors to retrieve (default: from config)
            
        Returns:
            numpy array of shape (top_k, vector_size) containing neighbor vectors,
            or empty array if query fails or no results found
        """
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
    
    def chelate_toxicity(self, local_cluster):
        """
        Phase 2: The Antigravity Mechanism (Binding Toxic Dimensions).
        
        Computes a binary mask based on dimension variance in the local cluster.
        - Normal mode: Keep low-variance (stable) dimensions, mask high-variance (toxic) ones
        - Inverted mode: Keep high-variance dimensions, mask low-variance (noise) ones
        
        Args:
            local_cluster: numpy array of shape (n_samples, vector_size)
            
        Returns:
            Binary mask array of shape (vector_size,) with 0s and 1s
        """
        if len(local_cluster) == 0:
            return np.ones(self.vector_size)

        # Calculate Variance per dimension
        dim_variance = np.var(local_cluster, axis=0)
        
        if self.invert_chelation:
            # INVERTED: Keep Top P% (High Variance), Mask Bottom (Noise)
            # Threshold at (100 - P)th percentile
            threshold = np.percentile(dim_variance, 100 - self.chelation_p)
            mask = (dim_variance > threshold).astype(float)
        else:
            # ORIGINAL: Keep Bottom P% (Stable), Mask Top (Toxic)
            threshold = np.percentile(dim_variance, self.chelation_p)
            mask = (dim_variance < threshold).astype(float)
        
        return mask
    
    def cosine_similarity_manual(self, vec1, vec2):
        """
        Calculates Cosine Similarity manually.
        
        Args:
            vec1: First vector (numpy array)
            vec2: Second vector (numpy array)
            
        Returns:
            Cosine similarity score (float), 0.0 if either vector has zero norm
        """
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return np.dot(vec1, vec2) / (norm1 * norm2)
    
    def spectral_chelation_ranking(self, query_vec, local_vectors, local_ids):
        """
        Stage 6: Spectral Chelation (Center of Mass).
        
        1. Calculate Mean Vector of top results (center of mass).
        2. Subtract Mean from Query and Candidates (shift reference frame).
        3. Rerank using centered vectors (cosine similarity in centered space).
        
        Also performs homeostatic logging: records the noise center for each participating document.
        
        Args:
            query_vec: Query vector (numpy array)
            local_vectors: List of candidate vectors
            local_ids: List of candidate document IDs (parallel to local_vectors)
            
        Returns:
            Tuple of (sorted_ids, center_of_mass):
                - sorted_ids: List of IDs sorted by centered cosine similarity (descending)
                - center_of_mass: The computed mean vector (numpy array)
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
        
        # Pair with IDs and sort
        scores = [(local_ids[i], scores_vec[i]) for i in range(len(local_ids))]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        sorted_ids = [s[0] for s in scores]
        return sorted_ids, center_of_mass
