"""
Tests for TopologyAnalyzer (Topology-Aware Retrieval Analysis)

Run: python -m unittest test_topology_analyzer -v
"""

import unittest
from unittest.mock import patch
import numpy as np

from topology_analyzer import (
    TopologyAnalyzer,
    BOND_COVALENT,
    BOND_HYDROGEN,
    BOND_VDW,
    BOND_NONE,
    BOND_TYPE_MAP,
)
from config import ChelationConfig


@patch('topology_analyzer.get_logger')
class TestTopologyAnalyzer(unittest.TestCase):
    """Tests for the TopologyAnalyzer class."""

    def test_initialization_defaults(self, mock_logger):
        """Test default threshold initialization."""
        analyzer = TopologyAnalyzer()
        self.assertAlmostEqual(analyzer.covalent_threshold, 0.90)
        self.assertAlmostEqual(analyzer.hydrogen_threshold, 0.70)
        self.assertAlmostEqual(analyzer.vdw_threshold, 0.40)

    def test_initialization_custom_thresholds(self, mock_logger):
        """Test custom threshold initialization."""
        analyzer = TopologyAnalyzer(
            covalent_threshold=0.95,
            hydrogen_threshold=0.80,
            vdw_threshold=0.50,
        )
        self.assertAlmostEqual(analyzer.covalent_threshold, 0.95)
        self.assertAlmostEqual(analyzer.hydrogen_threshold, 0.80)
        self.assertAlmostEqual(analyzer.vdw_threshold, 0.50)

    def test_initialization_invalid_thresholds(self, mock_logger):
        """Test that invalid threshold ordering raises ValueError."""
        with self.assertRaises(ValueError):
            TopologyAnalyzer(covalent_threshold=0.50, hydrogen_threshold=0.80, vdw_threshold=0.40)
        with self.assertRaises(ValueError):
            TopologyAnalyzer(covalent_threshold=0.90, hydrogen_threshold=0.70, vdw_threshold=0.0)

    def test_classify_bond_covalent(self, mock_logger):
        """Test covalent bond classification."""
        analyzer = TopologyAnalyzer()
        self.assertEqual(analyzer.classify_bond(0.95), BOND_COVALENT)
        self.assertEqual(analyzer.classify_bond(0.90), BOND_COVALENT)
        self.assertEqual(analyzer.classify_bond(1.0), BOND_COVALENT)

    def test_classify_bond_hydrogen(self, mock_logger):
        """Test hydrogen bond classification."""
        analyzer = TopologyAnalyzer()
        self.assertEqual(analyzer.classify_bond(0.89), BOND_HYDROGEN)
        self.assertEqual(analyzer.classify_bond(0.70), BOND_HYDROGEN)
        self.assertEqual(analyzer.classify_bond(0.75), BOND_HYDROGEN)

    def test_classify_bond_vdw(self, mock_logger):
        """Test van der Waals bond classification."""
        analyzer = TopologyAnalyzer()
        self.assertEqual(analyzer.classify_bond(0.69), BOND_VDW)
        self.assertEqual(analyzer.classify_bond(0.40), BOND_VDW)
        self.assertEqual(analyzer.classify_bond(0.50), BOND_VDW)

    def test_classify_bond_none(self, mock_logger):
        """Test no-bond classification."""
        analyzer = TopologyAnalyzer()
        self.assertEqual(analyzer.classify_bond(0.39), BOND_NONE)
        self.assertEqual(analyzer.classify_bond(0.0), BOND_NONE)
        self.assertEqual(analyzer.classify_bond(-0.5), BOND_NONE)

    def test_build_bond_matrix_empty(self, mock_logger):
        """Test bond matrix with empty embeddings."""
        analyzer = TopologyAnalyzer()
        result = analyzer.build_bond_matrix(np.array([]).reshape(0, 3))
        self.assertEqual(result["bond_matrix"].shape, (0, 0))
        self.assertEqual(result["bond_counts"][BOND_COVALENT], 0)

    def test_build_bond_matrix_identical_vectors(self, mock_logger):
        """Test bond matrix with identical vectors (all covalent)."""
        analyzer = TopologyAnalyzer()
        emb = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        result = analyzer.build_bond_matrix(emb)
        # All off-diagonal should be covalent (similarity=1.0)
        self.assertEqual(result["bond_matrix"].shape, (3, 3))
        # Count covalent bonds (excluding diagonal)
        self.assertEqual(result["bond_counts"][BOND_COVALENT], 6)  # 3*2 off-diagonal

    def test_build_bond_matrix_orthogonal_vectors(self, mock_logger):
        """Test bond matrix with orthogonal vectors (no bonds)."""
        analyzer = TopologyAnalyzer()
        emb = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        result = analyzer.build_bond_matrix(emb)
        # Orthogonal vectors have similarity=0.0, so all off-diagonal are BOND_NONE
        self.assertEqual(result["bond_counts"][BOND_NONE], 6)  # 3*2 off-diagonal
        self.assertEqual(result["bond_counts"][BOND_COVALENT], 0)

    def test_build_bond_matrix_ratios_sum_to_one(self, mock_logger):
        """Test that bond ratios sum to approximately 1.0."""
        analyzer = TopologyAnalyzer()
        np.random.seed(42)
        emb = np.random.randn(5, 10)
        result = analyzer.build_bond_matrix(emb)
        total_ratio = sum(result["bond_ratios"].values())
        self.assertAlmostEqual(total_ratio, 1.0, places=5)

    def test_build_bond_matrix_similarity_matrix_diagonal(self, mock_logger):
        """Test that similarity matrix diagonal is 1.0."""
        analyzer = TopologyAnalyzer()
        np.random.seed(123)
        emb = np.random.randn(4, 8)
        result = analyzer.build_bond_matrix(emb)
        diag = np.diag(result["similarity_matrix"])
        np.testing.assert_array_almost_equal(diag, np.ones(4), decimal=5)

    def test_compute_cluster_connectivity_empty(self, mock_logger):
        """Test cluster connectivity with empty input."""
        analyzer = TopologyAnalyzer()
        result = analyzer.compute_cluster_connectivity(
            np.array([]).reshape(0, 3), np.array([])
        )
        self.assertEqual(result["cluster_count"], 0)

    def test_compute_cluster_connectivity_single_cluster(self, mock_logger):
        """Test cluster connectivity with one cluster."""
        analyzer = TopologyAnalyzer()
        # All similar vectors in one cluster
        emb = np.array([[1.0, 0.1, 0.0], [1.0, 0.0, 0.1], [0.9, 0.1, 0.1]])
        labels = np.array([0, 0, 0])
        result = analyzer.compute_cluster_connectivity(emb, labels)
        self.assertEqual(result["cluster_count"], 1)
        # No inter-cluster bonds
        self.assertEqual(sum(result["inter_cluster"].values()), 0)
        # All bonds are intra-cluster
        self.assertGreater(sum(result["intra_cluster"].values()), 0)

    def test_compute_cluster_connectivity_two_clusters(self, mock_logger):
        """Test cluster connectivity with two clusters."""
        analyzer = TopologyAnalyzer()
        # Two distinct clusters
        emb = np.array([
            [1.0, 0.0, 0.0],
            [1.0, 0.05, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.05],
        ])
        labels = np.array([0, 0, 1, 1])
        result = analyzer.compute_cluster_connectivity(emb, labels)
        self.assertEqual(result["cluster_count"], 2)
        # Should have both intra and inter cluster bonds
        self.assertGreater(sum(result["intra_cluster"].values()), 0)
        self.assertGreater(sum(result["inter_cluster"].values()), 0)
        # Cluster cohesion should exist for both clusters
        self.assertIn(0, result["cluster_cohesion"])
        self.assertIn(1, result["cluster_cohesion"])

    def test_compute_cluster_connectivity_cohesion_values(self, mock_logger):
        """Test cluster cohesion values are in valid range."""
        analyzer = TopologyAnalyzer()
        np.random.seed(99)
        emb = np.random.randn(6, 5)
        labels = np.array([0, 0, 0, 1, 1, 1])
        result = analyzer.compute_cluster_connectivity(emb, labels)
        for label, cohesion in result["cluster_cohesion"].items():
            self.assertGreaterEqual(cohesion, -1.0)
            self.assertLessEqual(cohesion, 1.0)

    def test_compute_topology_change_no_change(self, mock_logger):
        """Test topology change with identical pre/post embeddings."""
        analyzer = TopologyAnalyzer()
        emb = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        result = analyzer.compute_topology_change(emb, emb)
        self.assertAlmostEqual(result["collapse_pressure"], 0.0)
        self.assertAlmostEqual(result["topology_distance"], 0.0)

    def test_compute_topology_change_collapse(self, mock_logger):
        """Test that moving vectors closer increases collapse pressure."""
        analyzer = TopologyAnalyzer()
        # Diverse pre-embeddings
        pre = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
        # Collapsed post-embeddings (all similar)
        post = np.array([[1.0, 0.1], [1.0, 0.0], [1.0, 0.05]])
        result = analyzer.compute_topology_change(pre, post)
        # More covalent bonds after collapse
        self.assertGreater(result["collapse_pressure"], 0.0)
        self.assertGreater(result["topology_distance"], 0.0)

    def test_compute_topology_change_empty(self, mock_logger):
        """Test topology change with empty embeddings."""
        analyzer = TopologyAnalyzer()
        emb = np.array([]).reshape(0, 3)
        result = analyzer.compute_topology_change(emb, emb)
        self.assertAlmostEqual(result["topology_distance"], 0.0)

    def test_compute_topology_change_bond_changes(self, mock_logger):
        """Test that bond_changes dict contains all bond types."""
        analyzer = TopologyAnalyzer()
        np.random.seed(77)
        pre = np.random.randn(4, 5)
        post = np.random.randn(4, 5)
        result = analyzer.compute_topology_change(pre, post)
        for bt in BOND_TYPE_MAP:
            self.assertIn(bt, result["bond_changes"])

    def test_record_snapshot(self, mock_logger):
        """Test snapshot recording."""
        analyzer = TopologyAnalyzer()
        emb = np.array([[1.0, 0.0], [0.0, 1.0]])
        analyzer.record_snapshot(emb, label="test")
        history = analyzer.get_snapshot_history()
        self.assertEqual(len(history), 1)
        self.assertEqual(history[0]["label"], "test")
        self.assertEqual(history[0]["num_embeddings"], 2)

    def test_record_multiple_snapshots(self, mock_logger):
        """Test recording multiple snapshots."""
        analyzer = TopologyAnalyzer()
        for i in range(3):
            emb = np.random.randn(3, 4)
            analyzer.record_snapshot(emb, label=f"snap_{i}")
        history = analyzer.get_snapshot_history()
        self.assertEqual(len(history), 3)
        self.assertEqual(history[2]["label"], "snap_2")

    def test_get_topology_report(self, mock_logger):
        """Test comprehensive topology report structure."""
        analyzer = TopologyAnalyzer()
        np.random.seed(55)
        emb = np.random.randn(5, 8)
        report = analyzer.get_topology_report(emb)
        self.assertEqual(report["num_embeddings"], 5)
        self.assertIn("bond_counts", report)
        self.assertIn("bond_ratios", report)
        self.assertIn("similarity_stats", report)
        self.assertIn("mean", report["similarity_stats"])
        self.assertIn("std", report["similarity_stats"])
        self.assertIn("snapshot_count", report)

    def test_get_topology_report_single_embedding(self, mock_logger):
        """Test topology report with a single embedding."""
        analyzer = TopologyAnalyzer()
        emb = np.array([[1.0, 2.0, 3.0]])
        report = analyzer.get_topology_report(emb)
        self.assertEqual(report["num_embeddings"], 1)
        self.assertAlmostEqual(report["similarity_stats"]["mean"], 1.0)

    def test_reset(self, mock_logger):
        """Test that reset clears all snapshots."""
        analyzer = TopologyAnalyzer()
        analyzer.record_snapshot(np.random.randn(3, 4), label="pre")
        analyzer.reset()
        self.assertEqual(len(analyzer.get_snapshot_history()), 0)

    def test_cosine_similarity_matrix_symmetry(self, mock_logger):
        """Test that cosine similarity matrix is symmetric."""
        analyzer = TopologyAnalyzer()
        np.random.seed(42)
        emb = np.random.randn(4, 6)
        sim = analyzer._cosine_similarity_matrix(emb)
        np.testing.assert_array_almost_equal(sim, sim.T, decimal=10)

    def test_zero_vector_handling(self, mock_logger):
        """Test that zero vectors are handled gracefully."""
        analyzer = TopologyAnalyzer()
        emb = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        # Should not raise
        result = analyzer.build_bond_matrix(emb)
        self.assertEqual(result["bond_matrix"].shape, (2, 2))


@patch('topology_analyzer.get_logger')
class TestTopologyPresets(unittest.TestCase):
    """Tests for topology config presets."""

    def test_topology_balanced_preset(self, mock_logger):
        """Test balanced topology preset."""
        preset = ChelationConfig.get_preset("balanced", "topology")
        self.assertAlmostEqual(preset["covalent_threshold"], 0.90)
        self.assertAlmostEqual(preset["hydrogen_threshold"], 0.70)
        self.assertAlmostEqual(preset["vdw_threshold"], 0.40)
        self.assertIn("description", preset)

    def test_topology_tight_preset(self, mock_logger):
        """Test tight topology preset."""
        preset = ChelationConfig.get_preset("tight", "topology")
        self.assertAlmostEqual(preset["covalent_threshold"], 0.95)
        self.assertGreater(preset["covalent_threshold"], 0.90)

    def test_topology_loose_preset(self, mock_logger):
        """Test loose topology preset."""
        preset = ChelationConfig.get_preset("loose", "topology")
        self.assertAlmostEqual(preset["covalent_threshold"], 0.85)
        self.assertLess(preset["covalent_threshold"], 0.90)

    def test_topology_invalid_preset(self, mock_logger):
        """Test invalid topology preset raises ValueError."""
        with self.assertRaises(ValueError):
            ChelationConfig.get_preset("nonexistent", "topology")


if __name__ == "__main__":
    unittest.main(verbosity=2)
