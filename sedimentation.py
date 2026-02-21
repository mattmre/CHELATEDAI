"""
Hierarchical Sedimentation Engine for ChelatedAI

Implements variance-based clustering with per-cluster adapter training for
sedimentation. Runs local corrections within clusters, then global refinement
across all training data.

Extracted from recursive_decomposer.py as part of finding F-042 for better
module organization and separation of concerns.
"""

import numpy as np
import torch
import torch.optim as optim
from typing import List, Tuple

from chelation_logger import get_logger
from sedimentation_trainer import compute_homeostatic_target, sync_vectors_to_qdrant
from checkpoint_manager import CheckpointManager, SafeTrainingContext


class HierarchicalSedimentationEngine:
    """
    Extends sedimentation with variance-based clustering for per-cluster
    adapter training. Runs local corrections within clusters, then global
    refinement across all training data.
    """

    def __init__(self, engine):
        """
        Args:
            engine: AntigravityEngine instance
        """
        self.engine = engine
        self.logger = get_logger()
        # Initialize checkpoint manager for safe training (F-043)
        self.checkpoint_manager = CheckpointManager()

    def run_hierarchical_sedimentation(self, threshold=3, learning_rate=0.001, epochs=10):
        """
        Hierarchical sedimentation cycle with cluster-aware training.

        1. Identify collapse targets from chelation_log
        2. Fetch vectors and cluster via variance-based partitioning
        3. Per-cluster training (local corrections)
        4. Global refinement (cross-cluster coherence)
        5. Save adapter and update Qdrant
        """
        print(f"\n--- HIERARCHICAL SEDIMENTATION (Threshold={threshold}, LR={learning_rate}) ---")

        # Filter for frequent collapsers
        targets = {k: v for k, v in self.engine.chelation_log.items() if len(v) >= threshold}
        print(f"Collapse targets: {len(targets)} candidates.")

        if not targets:
            print("No sedimentation targets. Brain is stable.")
            self.logger.log_event(
                "hierarchical_sedimentation",
                "No targets above threshold",
                threshold=threshold,
            )
            return

        # Fetch vectors for targets from Qdrant
        batch_ids = list(targets.keys())
        chunk_size = 100
        all_vectors = []
        all_ids = []
        payload_map = {}  # F-031: Cache payloads during initial retrieve

        for i in range(0, len(batch_ids), chunk_size):
            chunk = batch_ids[i:i + chunk_size]
            points = self.engine.qdrant.retrieve(
                collection_name=self.engine.collection_name,
                ids=chunk,
                with_vectors=True,
            )
            for point in points:
                all_ids.append(point.id)
                all_vectors.append(np.array(point.vector))
                # F-031: Cache payload for later sync
                payload_map[point.id] = point.payload

        if not all_vectors:
            print("No vectors retrieved. Skipping.")
            return

        vectors_np = np.array(all_vectors)

        # Prepare all training data (input, target pairs)
        training_inputs = []
        training_targets = []
        ordered_ids = []

        for i, doc_id in enumerate(all_ids):
            current_vec = vectors_np[i]
            noise_vectors = targets[doc_id]
            # Use shared helper for homeostatic target computation
            target_vec = compute_homeostatic_target(current_vec, noise_vectors, 0.1)

            training_inputs.append(current_vec)
            training_targets.append(target_vec)
            ordered_ids.append(doc_id)

        if not training_inputs:
            return

        input_tensor = torch.tensor(np.array(training_inputs), dtype=torch.float32)
        target_tensor = torch.tensor(np.array(training_targets), dtype=torch.float32)

        # Cluster via variance-based partitioning
        n_clusters = max(2, len(all_ids) // 5)
        clusters = self._simple_partition(vectors_np, list(range(len(all_ids))), n_clusters)

        cluster_epochs = max(1, epochs // 2)
        global_epochs = epochs - cluster_epochs

        self.logger.log_training_start(
            num_samples=len(training_inputs),
            learning_rate=learning_rate,
            epochs=epochs,
            threshold=threshold,
            n_clusters=len(clusters),
        )

        # --- Phase 1 & 2: Training wrapped in SafeTrainingContext (F-043) ---
        print(f"Phase 1: Per-cluster training ({len(clusters)} clusters, {cluster_epochs} epochs)...")
        
        with SafeTrainingContext(
            self.checkpoint_manager,
            self.engine.adapter_path,
            f"hierarchical_sedimentation_threshold_{threshold}"
        ) as training_ctx:
            optimizer = optim.Adam(self.engine.adapter.parameters(), lr=learning_rate)
            criterion = torch.nn.MSELoss()

            self.engine.adapter.train()

            for cluster_idx, (cluster_vecs, cluster_indices) in enumerate(clusters):
                if len(cluster_indices) == 0:
                    continue

                cluster_input = input_tensor[cluster_indices]
                cluster_target = target_tensor[cluster_indices]

                for epoch in range(cluster_epochs):
                    optimizer.zero_grad()
                    outputs = self.engine.adapter(cluster_input)
                    loss = criterion(outputs, cluster_target)
                    loss.backward()
                    optimizer.step()

                self.logger.log_event(
                    "cluster_training",
                    f"Cluster {cluster_idx}: {len(cluster_indices)} samples, final loss {loss.item():.6f}",
                    level="DEBUG",
                    cluster_idx=cluster_idx,
                    cluster_size=len(cluster_indices),
                    final_loss=loss.item(),
                )

            # --- Phase 2: Global refinement (cross-cluster coherence) ---
            print(f"Phase 2: Global refinement ({global_epochs} epochs, LR={learning_rate * 0.1:.6f})...")
            global_optimizer = optim.Adam(self.engine.adapter.parameters(), lr=learning_rate * 0.1)

            for epoch in range(global_epochs):
                global_optimizer.zero_grad()
                outputs = self.engine.adapter(input_tensor)
                loss = criterion(outputs, target_tensor)
                loss.backward()
                global_optimizer.step()

                if epoch % max(1, global_epochs // 2) == 0:
                    self.logger.log_training_epoch(epoch + 1, global_epochs, loss.item())

            self.engine.adapter.eval()
            final_loss = loss.item() if global_epochs > 0 else 0.0

            # Save adapter
            self.engine.adapter.save(self.engine.adapter_path)
            print("Adapter weights saved.")

            # Update Qdrant with adapted vectors
            print("Syncing updated vectors to Qdrant...")

            with torch.no_grad():
                new_vectors_np = self.engine.adapter(input_tensor).numpy()

            # Use shared helper for Qdrant sync (F-031: pass cached payload_map)
            total_updates, failed_updates = sync_vectors_to_qdrant(
                self.engine.qdrant, self.engine.collection_name, ordered_ids,
                new_vectors_np, chunk_size, self.logger, payload_map
            )
            
            # Mark success only if no failed vector updates (F-043)
            if failed_updates == 0:
                training_ctx.mark_success()
                self.logger.log_event(
                    "hierarchical_sedimentation_success",
                    f"Hierarchical training completed successfully. Updated {total_updates} vectors.",
                    vectors_updated=total_updates
                )
            else:
                self.logger.log_error(
                    "hierarchical_sedimentation_partial_failure",
                    f"Training completed but {failed_updates} vector updates failed. Rolling back.",
                    vectors_updated=total_updates,
                    vectors_failed=failed_updates
                )

        self.logger.log_training_complete(
            final_loss=final_loss,
            vectors_updated=total_updates,
            vectors_failed=failed_updates,
        )

        self.engine.chelation_log.clear()
        print(f"Hierarchical sedimentation complete. Updated {total_updates} vectors, {failed_updates} failed.")
        print("--- HIERARCHICAL SLEEP CYCLE COMPLETE ---")

    def _simple_partition(self, vectors: np.ndarray, doc_ids: list,
                          n_clusters: int) -> List[Tuple[np.ndarray, list]]:
        """
        Recursive variance-based splitting (no sklearn dependency).

        Finds the dimension with highest variance, splits on its median,
        and recurses until the requested number of clusters is reached.

        Args:
            vectors: (N, D) array of vectors
            doc_ids: list of indices or IDs corresponding to rows
            n_clusters: target number of clusters

        Returns:
            List of (cluster_vectors, cluster_ids) tuples
        """
        if n_clusters <= 1 or len(vectors) <= 1:
            return [(vectors, doc_ids)]

        # Find dimension with highest variance
        dim = int(np.argmax(np.var(vectors, axis=0)))
        median_val = float(np.median(vectors[:, dim]))

        left_mask = vectors[:, dim] <= median_val
        right_mask = ~left_mask

        # Edge case: all points go to one side
        if not np.any(left_mask) or not np.any(right_mask):
            return [(vectors, doc_ids)]

        left_vecs = vectors[left_mask]
        right_vecs = vectors[right_mask]
        left_ids = [doc_ids[i] for i in range(len(doc_ids)) if left_mask[i]]
        right_ids = [doc_ids[i] for i in range(len(doc_ids)) if right_mask[i]]

        left_k = n_clusters // 2
        right_k = n_clusters - left_k

        left_clusters = self._simple_partition(left_vecs, left_ids, left_k)
        right_clusters = self._simple_partition(right_vecs, right_ids, right_k)

        return left_clusters + right_clusters
