import unittest

from antigravity_engine import AntigravityEngine


class _StubStabilityTracker:
    def __init__(self, report):
        self._report = report

    def get_stability_report(self):
        return self._report


class _StubTopologyAnalyzer:
    def __init__(self, snapshots):
        self._snapshots = snapshots

    def get_snapshot_history(self):
        return self._snapshots


class _StubIsomerDetector:
    def __init__(self, report):
        self._report = report

    def get_isomer_report(self):
        return self._report


class TestStructuralHealthReport(unittest.TestCase):
    def _make_engine(self):
        return object.__new__(AntigravityEngine)

    def test_defaults_to_healthy_without_trackers(self):
        engine = self._make_engine()
        report = engine.get_structural_health_report()
        self.assertEqual(report["health_classification"], "healthy")
        self.assertEqual(report["structural_health_score"], 1.0)

    def test_stability_signal_marks_degrading(self):
        engine = self._make_engine()
        engine._stability_tracker = _StubStabilityTracker({
            "persistent_collapse_ratio": 0.25,
            "threshold_oscillation": 0.0,
        })

        report = engine.get_structural_health_report()
        self.assertEqual(report["health_classification"], "degrading")

    def test_topology_and_isomer_signals_mark_critical(self):
        engine = self._make_engine()
        engine._topology_analyzer = _StubTopologyAnalyzer([
            {"bond_ratios": {"covalent": 0.10}},
            {"bond_ratios": {"covalent": 0.25}},
        ])
        engine._isomer_detector = _StubIsomerDetector({
            "total_detections": 2,
            "cumulative_mean_strength": 0.65,
        })

        report = engine.get_structural_health_report()
        self.assertEqual(report["health_classification"], "critical")
        self.assertLess(report["structural_health_score"], 1.0)
        self.assertIn("structural_health_components", report)
        self.assertIn("topology", report)
        self.assertIn("isomers", report)


if __name__ == "__main__":
    unittest.main(verbosity=2)
