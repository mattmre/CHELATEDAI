"""
Sedimentation Trainer - Shared Training Logic for ChelatedAI

This module contains shared helper functions for sedimentation-based training
used by both antigravity_engine.py and recursive_decomposer.py. Extracts
common logic for homeostatic target computation and Qdrant vector synchronization.
"""

import numpy as np
from typing import List, Any
from qdrant_client.models import PointStruct


def compute_homeostatic_target(current_vec: np.ndarray, noise_vectors: List[np.ndarray], 
                                push_magnitude: float) -> np.ndarray:
    """
    Compute homeostatic target vector by pushing away from noise average.
    
    This implements the core sedimentation logic: calculate the average of noise vectors
    (chelation events), compute the direction away from that average, and push the 
    current vector in that direction by push_magnitude.
    
    Args:
        current_vec: The current vector to adjust (1D numpy array)
        noise_vectors: List of noise/collision vectors to push away from
        push_magnitude: How far to push (typically ChelationConfig.HOMEOSTATIC_PUSH_MAGNITUDE)
    
    Returns:
        Normalized target vector (1D numpy array)
    """
    avg_noise = np.mean(noise_vectors, axis=0)
    diff = current_vec - avg_noise
    diff_norm = diff / (np.linalg.norm(diff) + 1e-9)
    homeostatic_target = current_vec + (diff_norm * push_magnitude)
    homeostatic_target = homeostatic_target / (np.linalg.norm(homeostatic_target) + 1e-9)
    return homeostatic_target


def sync_vectors_to_qdrant(qdrant: Any, collection_name: str, ordered_ids: List,
                           new_vectors_np: np.ndarray, chunk_size: int, logger: Any,
                           payload_map: dict = None) -> tuple:
    """
    Synchronize updated vectors to Qdrant in batches, preserving existing payloads.
    
    Fetches existing points to preserve metadata, then upserts updated vectors
    in chunks. Handles exceptions per-batch and tracks success/failure counts.
    
    Args:
        qdrant: QdrantClient instance
        collection_name: Name of the collection to update
        ordered_ids: List of document IDs corresponding to vectors
        new_vectors_np: NumPy array of new vectors (N, D)
        chunk_size: Batch size for updates
        logger: ChelationLogger instance for event logging
        payload_map: Optional dict mapping doc IDs to payloads. If provided,
                    skips qdrant.retrieve for payload lookup (F-031 optimization)
    
    Returns:
        Tuple of (total_updates, failed_updates) counts
    """
    total_updates = 0
    failed_updates = 0
    
    for i in range(0, len(ordered_ids), chunk_size):
        chunk_ids = ordered_ids[i:i + chunk_size]
        chunk_vectors = new_vectors_np[i:i + chunk_size]
        
        # Fetch existing points to get payloads (preserve metadata)
        # Skip retrieve if payload_map was provided (F-031 optimization)
        try:
            if payload_map is None:
                existing_points = qdrant.retrieve(
                    collection_name=collection_name,
                    ids=chunk_ids,
                    with_vectors=False
                )
                chunk_payload_map = {p.id: p.payload for p in existing_points}
            else:
                chunk_payload_map = payload_map
            
            batch_points = []
            for j, doc_id in enumerate(chunk_ids):
                vec = chunk_vectors[j].tolist()
                pay = chunk_payload_map.get(doc_id, {})
                batch_points.append(PointStruct(
                    id=doc_id,
                    vector=vec,
                    payload=pay
                ))
            
            if batch_points:
                qdrant.upsert(
                    collection_name=collection_name,
                    points=batch_points
                )
                total_updates += len(batch_points)
        except ValueError as e:
            logger.log_error("database_update", 
                           f"Invalid vector data in batch {i//chunk_size}", 
                           exception=e, batch_num=i//chunk_size)
            failed_updates += len(chunk_ids)
        except Exception as e:
            logger.log_error("database_update", 
                           f"Update batch {i//chunk_size} failed", 
                           exception=e, batch_num=i//chunk_size)
            failed_updates += len(chunk_ids)
    
    return total_updates, failed_updates
