"""
Topology-Aware Retrieval Analysis for ChelatedAI

Models the embedding space as a bond-typed graph where pairwise cosine
similarities determine bond classification:
  - Covalent (>=0.90): very tight semantic coupling
  - Hydrogen (>=0.70): moderate semantic relationship
  - Van der Waals (>=0.40): weak association
  - None (<0.40): no meaningful bond

Provides cluster connectivity analysis, topology change detection for
sedimentation impact assessment, and historical snapshot tracking.
"""

import numpy as np
from chelation_logger import get_logger


# Bond type constants
BOND_COVALENT = "covalent"
BOND_HYDROGEN = "hydrogen"
BOND_VDW = "vdw"
BOND_NONE = "none"

# Bond type numeric encoding for matrix operations
BOND_TYPE_MAP = {
    BOND_COVALENT: 3,
    BOND_HYDROGEN: 2,
    BOND_VDW: 1,
    BOND_NONE: 0,
}

BOND_TYPE_REVERSE = {v: k for k, v in BOND_TYPE_MAP.items()}


class TopologyAnalyzer:
    """
    Analyzes embedding space topology using bond-typed graphs.

    Classifies pairwise relationships between embeddings into bond types
    based on cosine similarity thresholds, enabling structural health
    assessment and sedimentation impact detection.
    """

    def __init__(self, covalent_threshold=0.90, hydrogen_threshold=0.70,
                 vdw_threshold=0.40):
        """
        Initialize TopologyAnalyzer.

        Args:
            covalent_threshold: Minimum cosine similarity for covalent bond (default 0.90)
            hydrogen_threshold: Minimum cosine similarity for hydrogen bond (default 0.70)
            vdw_threshold: Minimum cosine similarity for vdw bond (default 0.40)
        """
        self.logger = get_logger()

        if not (covalent_threshold > hydrogen_threshold > vdw_threshold > 0):
            raise ValueError(
                "Thresholds must be strictly ordered: "
                "covalent > hydrogen > vdw > 0. "
                f"Got covalent={covalent_threshold}, hydrogen={hydrogen_threshold}, "
                f"vdw={vdw_threshold}"
            )

        self.covalent_threshold = float(covalent_threshold)
        self.hydrogen_threshold = float(hydrogen_threshold)
        self.vdw_threshold = float(vdw_threshold)

        # Historical snapshots for tracking topology changes over time
        self._snapshots = []

    def _cosine_similarity_matrix(self, embeddings):
        """
        Compute pairwise cosine similarity matrix.

        Args:
            embeddings: numpy array of shape (n, dim)

        Returns:
            numpy array of shape (n, n) with cosine similarities
        """
        embeddings = np.array(embeddings, dtype=float)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        # Avoid division by zero
        norms = np.where(norms == 0, 1.0, norms)
        normalized = embeddings / norms
        similarity = np.dot(normalized, normalized.T)
        # Clip to [-1, 1] for numerical stability
        return np.clip(similarity, -1.0, 1.0)

    def classify_bond(self, similarity):
        """
        Classify a single cosine similarity value into a bond type.

        Args:
            similarity: float cosine similarity value

        Returns:
            str: bond type name (covalent, hydrogen, vdw, none)
        """
        if similarity >= self.covalent_threshold:
            return BOND_COVALENT
        elif similarity >= self.hydrogen_threshold:
            return BOND_HYDROGEN
        elif similarity >= self.vdw_threshold:
            return BOND_VDW
        else:
            return BOND_NONE

    def build_bond_matrix(self, embeddings):
        """
        Build typed adjacency matrix from embeddings.

        Args:
            embeddings: numpy array of shape (n, dim)

        Returns:
            dict with:
                - 'bond_matrix': numpy int array of shape (n, n) with bond type codes
                - 'similarity_matrix': numpy float array of shape (n, n)
                - 'bond_counts': dict mapping bond type name to count (excluding diagonal)
                - 'bond_ratios': dict mapping bond type name to fraction of total pairs
        """
        embeddings = np.array(embeddings, dtype=float)
        n = embeddings.shape[0]

        if n == 0:
            return {
                "bond_matrix": np.array([], dtype=int).reshape(0, 0),
                "similarity_matrix": np.array([], dtype=float).reshape(0, 0),
                "bond_counts": {BOND_COVALENT: 0, BOND_HYDROGEN: 0, BOND_VDW: 0, BOND_NONE: 0},
                "bond_ratios": {BOND_COVALENT: 0.0, BOND_HYDROGEN: 0.0, BOND_VDW: 0.0, BOND_NONE: 0.0},
            }

        sim_matrix = self._cosine_similarity_matrix(embeddings)

        # Vectorized bond classification
        bond_matrix = np.zeros((n, n), dtype=int)
        bond_matrix[sim_matrix >= self.covalent_threshold] = BOND_TYPE_MAP[BOND_COVALENT]
        # For ranges, apply in order from weakest to strongest so strongest wins
        bond_matrix[sim_matrix >= self.vdw_threshold] = BOND_TYPE_MAP[BOND_VDW]
        bond_matrix[sim_matrix >= self.hydrogen_threshold] = BOND_TYPE_MAP[BOND_HYDROGEN]
        bond_matrix[sim_matrix >= self.covalent_threshold] = BOND_TYPE_MAP[BOND_COVALENT]

        # Count bonds (exclude diagonal - self-bonds are always covalent)
        total_pairs = n * (n - 1) if n > 1 else 1  # avoid division by zero

        bond_counts = {}
        bond_ratios = {}
        for bond_name, bond_code in BOND_TYPE_MAP.items():
            mask = bond_matrix == bond_code
            # Exclude diagonal for counting
            np.fill_diagonal(mask, False)
            count = int(np.sum(mask))
            bond_counts[bond_name] = count
            bond_ratios[bond_name] = count / total_pairs

        return {
            "bond_matrix": bond_matrix,
            "similarity_matrix": sim_matrix,
            "bond_counts": bond_counts,
            "bond_ratios": bond_ratios,
        }

    def compute_cluster_connectivity(self, embeddings, labels):
        """
        Compute inter-cluster and intra-cluster bond statistics.

        Args:
            embeddings: numpy array of shape (n, dim)
            labels: array-like of length n with cluster assignments

        Returns:
            dict with:
                - 'intra_cluster': dict of bond counts within clusters
                - 'inter_cluster': dict of bond counts between clusters
                - 'intra_bond_ratios': bond type ratios within clusters
                - 'inter_bond_ratios': bond type ratios between clusters
                - 'cluster_cohesion': dict mapping cluster label to mean similarity
                - 'cluster_count': number of unique clusters
        """
        embeddings = np.array(embeddings, dtype=float)
        labels = np.array(labels)
        n = embeddings.shape[0]

        if n == 0:
            return {
                "intra_cluster": {bt: 0 for bt in BOND_TYPE_MAP},
                "inter_cluster": {bt: 0 for bt in BOND_TYPE_MAP},
                "intra_bond_ratios": {bt: 0.0 for bt in BOND_TYPE_MAP},
                "inter_bond_ratios": {bt: 0.0 for bt in BOND_TYPE_MAP},
                "cluster_cohesion": {},
                "cluster_count": 0,
            }

        sim_matrix = self._cosine_similarity_matrix(embeddings)
        bond_result = self.build_bond_matrix(embeddings)
        bond_matrix = bond_result["bond_matrix"]

        unique_labels = np.unique(labels)

        intra_counts = {bt: 0 for bt in BOND_TYPE_MAP}
        inter_counts = {bt: 0 for bt in BOND_TYPE_MAP}
        cluster_cohesion = {}
        total_intra_pairs = 0
        total_inter_pairs = 0

        for label in unique_labels:
            cluster_mask = labels == label
            cluster_indices = np.where(cluster_mask)[0]
            cluster_size = len(cluster_indices)

            # Intra-cluster bonds
            if cluster_size > 1:
                for i_idx in range(cluster_size):
                    for j_idx in range(i_idx + 1, cluster_size):
                        ii = cluster_indices[i_idx]
                        jj = cluster_indices[j_idx]
                        bond_code = bond_matrix[ii, jj]
                        bond_name = BOND_TYPE_REVERSE[bond_code]
                        intra_counts[bond_name] += 1
                        total_intra_pairs += 1

            # Cluster cohesion: mean pairwise similarity within cluster
            if cluster_size > 1:
                sub_sim = sim_matrix[np.ix_(cluster_indices, cluster_indices)]
                # Upper triangle only (exclude diagonal)
                triu_idx = np.triu_indices(cluster_size, k=1)
                cohesion_vals = sub_sim[triu_idx]
                cluster_cohesion[int(label) if np.issubdtype(type(label), np.integer) else label] = float(np.mean(cohesion_vals))
            else:
                cluster_cohesion[int(label) if np.issubdtype(type(label), np.integer) else label] = 1.0

        # Inter-cluster bonds
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] != labels[j]:
                    bond_code = bond_matrix[i, j]
                    bond_name = BOND_TYPE_REVERSE[bond_code]
                    inter_counts[bond_name] += 1
                    total_inter_pairs += 1

        # Compute ratios
        intra_ratios = {}
        inter_ratios = {}
        for bt in BOND_TYPE_MAP:
            intra_ratios[bt] = intra_counts[bt] / max(total_intra_pairs, 1)
            inter_ratios[bt] = inter_counts[bt] / max(total_inter_pairs, 1)

        return {
            "intra_cluster": intra_counts,
            "inter_cluster": inter_counts,
            "intra_bond_ratios": intra_ratios,
            "inter_bond_ratios": inter_ratios,
            "cluster_cohesion": cluster_cohesion,
            "cluster_count": len(unique_labels),
        }

    def compute_topology_change(self, pre_embeddings, post_embeddings):
        """
        Compare topology before and after sedimentation.

        Args:
            pre_embeddings: numpy array of shape (n, dim), embeddings before sedimentation
            post_embeddings: numpy array of shape (n, dim), embeddings after sedimentation

        Returns:
            dict with:
                - 'pre_bond_ratios': bond ratios before
                - 'post_bond_ratios': bond ratios after
                - 'collapse_pressure': change in covalent bond ratio (positive = more collapse)
                - 'topology_distance': normalized Frobenius norm of bond type matrix difference
                - 'bond_changes': dict mapping bond type to change in ratio
        """
        pre_result = self.build_bond_matrix(pre_embeddings)
        post_result = self.build_bond_matrix(post_embeddings)

        pre_ratios = pre_result["bond_ratios"]
        post_ratios = post_result["bond_ratios"]

        # Collapse pressure: increase in covalent ratio
        collapse_pressure = post_ratios[BOND_COVALENT] - pre_ratios[BOND_COVALENT]

        # Topology distance: normalized Frobenius norm of bond matrix difference
        pre_matrix = pre_result["bond_matrix"]
        post_matrix = post_result["bond_matrix"]

        if pre_matrix.size == 0:
            topology_distance = 0.0
        else:
            diff = post_matrix.astype(float) - pre_matrix.astype(float)
            # Normalize by max possible change (3 * sqrt(n*n)) for bond codes 0-3
            max_change = 3.0 * np.sqrt(diff.size) if diff.size > 0 else 1.0
            topology_distance = float(np.linalg.norm(diff)) / max_change

        bond_changes = {}
        for bt in BOND_TYPE_MAP:
            bond_changes[bt] = post_ratios[bt] - pre_ratios[bt]

        return {
            "pre_bond_ratios": pre_ratios,
            "post_bond_ratios": post_ratios,
            "collapse_pressure": float(collapse_pressure),
            "topology_distance": float(topology_distance),
            "bond_changes": bond_changes,
        }

    def record_snapshot(self, embeddings, label=None):
        """
        Record a topology snapshot for historical tracking.

        Args:
            embeddings: numpy array of shape (n, dim)
            label: optional string label for this snapshot
        """
        result = self.build_bond_matrix(embeddings)
        snapshot = {
            "bond_ratios": result["bond_ratios"],
            "bond_counts": result["bond_counts"],
            "num_embeddings": embeddings.shape[0] if hasattr(embeddings, 'shape') else len(embeddings),
            "label": label,
        }
        self._snapshots.append(snapshot)
        self.logger.log_event(
            "topology_snapshot",
            f"Recorded topology snapshot ({label}): "
            f"covalent={result['bond_ratios'][BOND_COVALENT]:.3f}, "
            f"hydrogen={result['bond_ratios'][BOND_HYDROGEN]:.3f}"
        )

    def get_snapshot_history(self):
        """
        Get all recorded topology snapshots.

        Returns:
            list of snapshot dicts
        """
        return list(self._snapshots)

    def get_topology_report(self, embeddings):
        """
        Get comprehensive topology report for a set of embeddings.

        Args:
            embeddings: numpy array of shape (n, dim)

        Returns:
            dict with full topology analysis
        """
        result = self.build_bond_matrix(embeddings)

        # Compute summary statistics
        n = embeddings.shape[0] if hasattr(embeddings, 'shape') else len(embeddings)
        sim_matrix = result["similarity_matrix"]

        if n > 1:
            triu_idx = np.triu_indices(n, k=1)
            pairwise_sims = sim_matrix[triu_idx]
            mean_sim = float(np.mean(pairwise_sims))
            std_sim = float(np.std(pairwise_sims))
            min_sim = float(np.min(pairwise_sims))
            max_sim = float(np.max(pairwise_sims))
        else:
            mean_sim = 1.0
            std_sim = 0.0
            min_sim = 1.0
            max_sim = 1.0

        return {
            "num_embeddings": n,
            "bond_counts": result["bond_counts"],
            "bond_ratios": result["bond_ratios"],
            "similarity_stats": {
                "mean": mean_sim,
                "std": std_sim,
                "min": min_sim,
                "max": max_sim,
            },
            "snapshot_count": len(self._snapshots),
        }

    def reset(self):
        """Reset all snapshot history."""
        self._snapshots = []
