"""
Integration Tests for ChelatedAI Retrieval, Learning, and Management

End-to-end tests that exercise the full pipeline: AntigravityEngine with
in-memory Qdrant, RecursiveRetrievalEngine with MockDecomposer, hierarchical
sedimentation, and AEP orchestrator on realistic ChelatedAI findings.

Requires sentence-transformers for embedding; tests are skipped gracefully
if the package is not installed.
"""

import unittest
import json
import numpy as np
import warnings
from unittest.mock import patch, MagicMock

# Check for sentence-transformers availability
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    warnings.warn(
        "sentence-transformers is not installed. Integration tests in "
        "test_integration_rlm.py will be skipped. Install with: "
        "pip install sentence-transformers",
        UserWarning,
        stacklevel=2
    )

from recursive_decomposer import (
    RecursiveRetrievalEngine,
    MockDecomposer,
    HierarchicalSedimentationEngine,
    DecompositionTrace,
    DecompositionNode,
)
from aep_orchestrator import (
    AEPOrchestrator,
    Severity,
    FindingStatus,
    EffortSize,
)


@unittest.skipUnless(
    HAS_SENTENCE_TRANSFORMERS, "sentence-transformers not installed"
)
class TestIntegrationRLM(unittest.TestCase):
    """Integration tests requiring a live AntigravityEngine with in-memory Qdrant."""

    def setUp(self):
        """Create an in-memory engine, ingest a synthetic corpus, and silence loggers."""
        # Patch the logger to avoid file creation
        self.logger_patcher = patch("recursive_decomposer.get_logger")
        self.mock_logger = self.logger_patcher.start()
        self.mock_logger.return_value = MagicMock()

        # Also patch aep_orchestrator logger
        self.aep_logger_patcher = patch("aep_orchestrator.get_logger")
        self.mock_aep_logger = self.aep_logger_patcher.start()
        self.mock_aep_logger.return_value = MagicMock()

        # Also patch antigravity_engine logger
        self.engine_logger_patcher = patch("antigravity_engine.get_logger")
        self.mock_engine_logger = self.engine_logger_patcher.start()
        self.mock_engine_logger.return_value = MagicMock()

        from antigravity_engine import AntigravityEngine

        self.engine = AntigravityEngine(
            qdrant_location=":memory:",
            model_name="all-MiniLM-L6-v2",
            chelation_p=85,
        )

        # Ingest synthetic corpus
        self.corpus = [
            "Machine learning algorithms for classification tasks",
            "Deep neural networks and backpropagation methods",
            "Natural language processing with transformer models",
            "Computer vision and convolutional neural networks",
            "Reinforcement learning for game playing agents",
            "Statistical methods for data analysis",
            "Database indexing and query optimization techniques",
            "Distributed computing and parallel processing",
            "Information retrieval and search engine design",
            "Knowledge graphs and semantic web technologies",
        ]
        self.engine.ingest(self.corpus)

    def tearDown(self):
        """Stop logger patches."""
        self.logger_patcher.stop()
        self.aep_logger_patcher.stop()
        self.engine_logger_patcher.stop()

    # ------------------------------------------------------------------
    # Test 1: Recursive retrieval with chelation
    # ------------------------------------------------------------------
    def test_recursive_retrieval_with_chelation(self):
        """Recursive retrieval decomposes a compound query and retrieves for each leaf."""
        decomposer = MockDecomposer(max_depth=3, min_length=20)
        rre = RecursiveRetrievalEngine(
            engine=self.engine,
            decomposer=decomposer,
            aggregation_strategy="rrf",
            max_depth=3,
            top_k=10,
        )

        # "machine learning and neural networks" splits on ' and '
        trace = rre.run_recursive_inference(
            "machine learning and neural networks"
        )

        self.assertIsInstance(trace, DecompositionTrace)
        self.assertGreater(len(trace.final_results), 0)
        self.assertGreaterEqual(
            trace.total_nodes,
            2,
            "Compound query should produce at least 2 tree nodes",
        )
        self.assertGreaterEqual(
            trace.total_retrieval_calls,
            2,
            "Each leaf sub-query should trigger a retrieval call",
        )

    # ------------------------------------------------------------------
    # Test 2: Aggregation preserves chelation order
    # ------------------------------------------------------------------
    def test_aggregation_preserves_chelation(self):
        """Aggregated results are non-empty and scores are in descending order."""
        decomposer = MockDecomposer(max_depth=3, min_length=20)
        rre = RecursiveRetrievalEngine(
            engine=self.engine,
            decomposer=decomposer,
            aggregation_strategy="rrf",
            max_depth=3,
            top_k=10,
        )

        trace = rre.run_recursive_inference(
            "distributed computing and database optimization"
        )

        self.assertGreater(
            len(trace.final_results),
            0,
            "Final results should not be empty",
        )
        self.assertEqual(
            len(trace.final_results),
            len(trace.final_scores),
            "Results and scores lists must have the same length",
        )

        # Verify descending score order (aggregation ranking)
        for i in range(len(trace.final_scores) - 1):
            self.assertGreaterEqual(
                trace.final_scores[i],
                trace.final_scores[i + 1],
                f"Scores must be in descending order: index {i} "
                f"({trace.final_scores[i]}) < index {i+1} "
                f"({trace.final_scores[i+1]})",
            )

    # ------------------------------------------------------------------
    # Test 3: Hierarchical sedimentation cycle
    # ------------------------------------------------------------------
    def test_hierarchical_sedimentation_cycle(self):
        """Hierarchical sedimentation trains the adapter and clears chelation_log."""
        # Manually populate chelation_log with synthetic collapse events.
        # Each entry maps a point ID to a list of noise center vectors that
        # the point collapsed toward during retrieval.
        for i in range(5):
            self.engine.chelation_log[i] = [
                np.random.randn(self.engine.vector_size) for _ in range(3)
            ]

        # Verify chelation_log has entries before the cycle
        self.assertGreater(
            len(self.engine.chelation_log),
            0,
            "chelation_log should be populated before sedimentation",
        )

        sed = HierarchicalSedimentationEngine(self.engine)
        sed.run_hierarchical_sedimentation(
            threshold=1, learning_rate=0.001, epochs=4
        )

        # After sedimentation the log must be cleared
        self.assertEqual(
            len(self.engine.chelation_log),
            0,
            "chelation_log should be cleared after a sedimentation cycle",
        )

    # ------------------------------------------------------------------
    # Test 3b: Sedimentation with epochs=0 edge case
    # ------------------------------------------------------------------
    def test_sedimentation_epochs_zero_does_not_crash(self):
        """Sedimentation with epochs=0 should not crash (UnboundLocalError guard)."""
        for i in range(5):
            self.engine.chelation_log[i] = [
                np.random.randn(self.engine.vector_size) for _ in range(3)
            ]

        # epochs=0 means the training loop body never executes.
        # Previously this caused UnboundLocalError on `loss.item()`.
        self.engine.run_sedimentation_cycle(
            threshold=1, learning_rate=0.001, epochs=0
        )
        # Should complete without error; log should still be cleared.
        self.assertEqual(len(self.engine.chelation_log), 0)

    # ------------------------------------------------------------------
    # Test 4: Baseline training mode compatibility
    # ------------------------------------------------------------------
    def test_baseline_training_mode_still_works(self):
        """Baseline mode should work identically to prior behavior."""
        # Create baseline engine (default mode)
        baseline_engine = None
        try:
            from antigravity_engine import AntigravityEngine
            
            baseline_engine = AntigravityEngine(
                qdrant_location=":memory:",
                model_name="all-MiniLM-L6-v2",
                chelation_p=85,
                training_mode="baseline"  # Explicit baseline
            )
            
            # Ingest small corpus
            corpus = [
                "Machine learning and deep learning",
                "Natural language processing with transformers",
                "Computer vision applications"
            ]
            baseline_engine.ingest(corpus)
            
            # Run some queries to populate chelation log
            baseline_engine.run_inference("machine learning")
            baseline_engine.run_inference("transformers")
            
            # Populate chelation log manually
            for i in range(3):
                baseline_engine.chelation_log[i] = [
                    np.random.randn(baseline_engine.vector_size) for _ in range(3)
                ]
            
            # Run sedimentation with baseline mode
            baseline_engine.run_sedimentation_cycle(
                threshold=1,
                learning_rate=0.001,
                epochs=2
            )
            
            # Should complete successfully and clear log
            self.assertEqual(len(baseline_engine.chelation_log), 0)
            
        finally:
            if baseline_engine:
                del baseline_engine

    # ------------------------------------------------------------------
    # Test 5: Hybrid mode initialization with mocked teacher
    # ------------------------------------------------------------------
    @patch("antigravity_engine.create_distillation_helper")
    def test_hybrid_mode_initialization(self, mock_helper_factory):
        """Hybrid mode should initialize with teacher helper."""
        # Mock the teacher helper
        mock_helper = MagicMock()
        mock_helper_factory.return_value = mock_helper
        
        from antigravity_engine import AntigravityEngine
        
        # Create hybrid engine
        engine = AntigravityEngine(
            qdrant_location=":memory:",
            model_name="all-MiniLM-L6-v2",
            chelation_p=85,
            training_mode="hybrid",
            teacher_model_name="test-teacher",
            teacher_weight=0.5
        )
        
        # Verify helper was created
        mock_helper_factory.assert_called_once()
        self.assertEqual(engine.training_mode, "hybrid")
        self.assertEqual(engine.teacher_weight, 0.5)
        self.assertIsNotNone(engine.teacher_helper)

    # ------------------------------------------------------------------
    # Test 6: Offline mode initialization with mocked teacher
    # ------------------------------------------------------------------
    @patch("antigravity_engine.create_distillation_helper")
    def test_offline_mode_initialization(self, mock_helper_factory):
        """Offline mode should initialize with teacher helper."""
        # Mock the teacher helper
        mock_helper = MagicMock()
        mock_helper_factory.return_value = mock_helper
        
        from antigravity_engine import AntigravityEngine
        
        # Create offline engine
        engine = AntigravityEngine(
            qdrant_location=":memory:",
            model_name="all-MiniLM-L6-v2",
            chelation_p=85,
            training_mode="offline",
            teacher_model_name="test-teacher"
        )
        
        # Verify helper was created
        mock_helper_factory.assert_called_once()
        self.assertEqual(engine.training_mode, "offline")
        self.assertIsNotNone(engine.teacher_helper)

    # ------------------------------------------------------------------
    # Test 7: Sedimentation with mocked teacher (hybrid mode)
    # ------------------------------------------------------------------
    @patch("antigravity_engine.create_distillation_helper")
    def test_sedimentation_hybrid_mode_mocked(self, mock_helper_factory):
        """Hybrid sedimentation should blend homeostatic and teacher targets."""
        # Mock teacher helper
        mock_helper = MagicMock()
        
        # Mock get_teacher_embeddings to return random embeddings
        def mock_get_embeddings(texts):
            n = len(texts)
            dim = 384
            embeds = np.random.randn(n, dim)
            # Normalize
            embeds = embeds / (np.linalg.norm(embeds, axis=1, keepdims=True) + 1e-9)
            return embeds
        
        mock_helper.get_teacher_embeddings.side_effect = mock_get_embeddings
        mock_helper_factory.return_value = mock_helper
        
        from antigravity_engine import AntigravityEngine
        
        # Create hybrid engine
        engine = AntigravityEngine(
            qdrant_location=":memory:",
            model_name="all-MiniLM-L6-v2",
            chelation_p=85,
            training_mode="hybrid",
            teacher_model_name="test-teacher",
            teacher_weight=0.5
        )
        
        # Ingest corpus
        corpus = [
            "Machine learning research",
            "Neural network architectures",
            "Deep learning applications"
        ]
        engine.ingest(corpus)
        
        # Populate chelation log
        for i in range(3):
            engine.chelation_log[i] = [
                np.random.randn(engine.vector_size) for _ in range(3)
            ]
        
        # Run sedimentation
        engine.run_sedimentation_cycle(
            threshold=1,
            learning_rate=0.001,
            epochs=2
        )
        
        # Verify teacher embeddings were requested
        self.assertTrue(mock_helper.get_teacher_embeddings.called)
        
        # Log should be cleared
        self.assertEqual(len(engine.chelation_log), 0)

    # ------------------------------------------------------------------
    # Test 8: Training mode validation
    # ------------------------------------------------------------------
    def test_training_mode_validation(self):
        """Invalid training modes should fallback to baseline."""
        from antigravity_engine import AntigravityEngine
        
        # Invalid mode should default to baseline (config validation)
        engine = AntigravityEngine(
            qdrant_location=":memory:",
            model_name="all-MiniLM-L6-v2",
            chelation_p=85,
            training_mode="invalid_mode"
        )
        
        # Should fallback to baseline
        self.assertEqual(engine.training_mode, "baseline")

    # ------------------------------------------------------------------
    # Test 9: Teacher weight validation
    # ------------------------------------------------------------------
    @patch("antigravity_engine.create_distillation_helper")
    def test_teacher_weight_validation(self, mock_helper_factory):
        """Teacher weight should be clamped to [0, 1]."""
        mock_helper = MagicMock()
        mock_helper_factory.return_value = mock_helper
        
        from antigravity_engine import AntigravityEngine
        
        # Test out-of-range weights get clamped
        engine1 = AntigravityEngine(
            qdrant_location=":memory:",
            model_name="all-MiniLM-L6-v2",
            training_mode="hybrid",
            teacher_weight=1.5  # Should clamp to 1.0
        )
        self.assertEqual(engine1.teacher_weight, 1.0)
        
        engine2 = AntigravityEngine(
            qdrant_location=":memory:",
            model_name="all-MiniLM-L6-v2",
            training_mode="hybrid",
            teacher_weight=-0.5  # Should clamp to 0.0
        )
        self.assertEqual(engine2.teacher_weight, 0.0)

    # ------------------------------------------------------------------
    # Test 10: Decomposition trace completeness
    # ------------------------------------------------------------------
    def test_decomposition_trace_completeness(self):
        """Trace object is fully populated after recursive inference on a multi-part query."""
        decomposer = MockDecomposer(max_depth=3, min_length=20)
        rre = RecursiveRetrievalEngine(
            engine=self.engine,
            decomposer=decomposer,
            aggregation_strategy="rrf",
            max_depth=3,
            top_k=10,
        )

        query = "classification and regression and clustering"
        trace = rre.run_recursive_inference(query)

        self.assertEqual(
            trace.root_query,
            query,
            "root_query must match the original input",
        )
        self.assertGreater(trace.total_nodes, 0)
        self.assertGreaterEqual(trace.max_depth_reached, 0)
        self.assertGreater(
            trace.elapsed_seconds,
            0,
            "Elapsed time must be positive",
        )
        self.assertGreater(
            len(trace.final_results),
            0,
            "Final results should contain at least one document",
        )

    # ------------------------------------------------------------------
    # Test 11: AEP orchestrator on ChelatedAI-style findings
    # ------------------------------------------------------------------
    def test_aep_orchestrator_on_chelatedai_findings(self):
        """Full AEP cycle processes realistic ChelatedAI findings end to end."""
        raw_findings = [
            {
                "title": "Missing input validation on adapter weights path",
                "severity": "CRITICAL",
                "impact": "Arbitrary file read via crafted path",
                "effort": "S",
                "file_path": "chelation_adapter.py",
                "line_range": "53-60",
                "recommended_fix": "Validate path is within PROJECT_ROOT",
                "acceptance_criteria": [
                    "Path traversal blocked",
                    "Unit test added",
                ],
            },
            {
                "title": "Hardcoded embedding dimension fallback",
                "severity": "MEDIUM",
                "impact": "Silent dimension mismatch if Ollama unavailable",
                "effort": "S",
                "file_path": "antigravity_engine.py",
                "line_range": "36",
                "recommended_fix": "Use ChelationConfig.DEFAULT_VECTOR_SIZE",
                "acceptance_criteria": ["Config value used"],
            },
            {
                "title": "No retry backoff on embedding failures",
                "severity": "HIGH",
                "impact": "Thundering herd on Ollama recovery",
                "effort": "M",
                "file_path": "antigravity_engine.py",
                "line_range": "105-182",
                "recommended_fix": "Add exponential backoff to embed retry",
                "acceptance_criteria": [
                    "Backoff implemented",
                    "Max retries configurable",
                ],
            },
        ]

        orchestrator = AEPOrchestrator()
        summary = orchestrator.run_full_cycle(raw_findings)

        # Verify total count
        self.assertEqual(
            summary["total_findings"],
            3,
            "All three findings should be present in the summary",
        )

        # Verify severity breakdown
        self.assertEqual(summary["by_severity"].get("CRITICAL", 0), 1)
        self.assertEqual(summary["by_severity"].get("HIGH", 0), 1)
        self.assertEqual(summary["by_severity"].get("MEDIUM", 0), 1)

        # Verify markdown tracker contains all finding IDs
        markdown = summary["markdown_tracker"]
        finding_ids = list(orchestrator.tracker.findings.keys())
        self.assertEqual(len(finding_ids), 3)
        for fid in finding_ids:
            self.assertIn(
                fid,
                markdown,
                f"Finding {fid} should appear in the markdown tracker",
            )

        # Verify JSON export is valid JSON with expected structure
        json_export = summary["json_export"]
        parsed = json.loads(json_export)
        self.assertIn("findings", parsed)
        self.assertIn("verification_log", parsed)
        self.assertIn("exported_at", parsed)
        self.assertEqual(len(parsed["findings"]), 3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
