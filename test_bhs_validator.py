"""Unit tests for scripts/bhs_validator.py — CD-244-01 / CD-244-02 verification.

Uses ``unittest`` (not pytest) per CLAUDE.md Test Conventions.

What's covered:
  - validate_pr_brutal_honesty returns 100 (or near-100) on a complete finding
    dict with evidence pointer.
  - validate_pr_brutal_honesty returns a markedly lower score on a sparse
    finding dict (missing keys, UNKNOWN severity, no evidence).
  - The scores differ — this is the runtime evidence that closes CD-244-02
    (summary["avg_bhs_score"] will vary because the underlying scorer varies).
  - phase_summary scoring penalises missing keys and absent evidence.
  - run_smoke_pipeline(FLOOR) returns True in a healthy repo, False if a
    required file goes missing.
  - run_smoke_pipeline is NOT always-True (regression guard against the stub).
"""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from unittest.mock import patch

# Ensure repo root on sys.path so the test file can import scripts.bhs_validator
# when invoked via `python test_bhs_validator.py` from any cwd.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from scripts.bhs_validator import (  # noqa: E402
    BHSResult,
    HonestyTier,
    run_smoke_pipeline,
    validate_pr_brutal_honesty,
)


class TestFindingScoring(unittest.TestCase):
    def test_complete_finding_scores_100(self) -> None:
        # Avoid lie-marker words ("stub", "todo", "fixme", "placeholder", etc.)
        # in the finding text — those would correctly subtract from the score.
        rich = {
            "id": "F-001",
            "severity": "HIGH",
            "impact": "Critical gate is inert; production smoke cannot fire",
            "recommended_fix": (
                "Wire real scoring into scripts/bhs_validator.py:43; "
                "evidence: PR #244 closure artifact"
            ),
        }
        result = validate_pr_brutal_honesty(finding_dict=rich)
        self.assertIsInstance(result, BHSResult)
        self.assertEqual(result.score, 100.0)
        self.assertTrue(result.evidence_present)
        self.assertEqual(result.optimism_flags, [])

    def test_sparse_finding_scores_low(self) -> None:
        sparse = {"id": "F-002", "severity": "UNKNOWN"}
        result = validate_pr_brutal_honesty(finding_dict=sparse)
        # Missing impact (-15) + missing recommended_fix (-15) + UNKNOWN severity
        # (-10) + no evidence (-20) = -60 → 40.
        self.assertLessEqual(result.score, 50.0)
        self.assertGreaterEqual(result.score, 0.0)
        self.assertFalse(result.evidence_present)
        self.assertTrue(any("severity" in f for f in result.optimism_flags))

    def test_scores_differ_between_rich_and_sparse(self) -> None:
        """CD-244-02 runtime evidence: scores must vary by content."""
        rich = {
            "id": "F-A",
            "severity": "CRITICAL",
            "impact": "Outage in production at server.py:120",
            "recommended_fix": "Patch handler at api.py:55, artifact: logs/outage.json",
        }
        sparse = {"id": "F-B"}
        rich_score = validate_pr_brutal_honesty(finding_dict=rich).score
        sparse_score = validate_pr_brutal_honesty(finding_dict=sparse).score
        self.assertGreater(rich_score, sparse_score)
        self.assertNotEqual(rich_score, 0.0)

    def test_lie_marker_keyword_penalised(self) -> None:
        finding = {
            "id": "F-C",
            "severity": "MEDIUM",
            "impact": "TODO: actually implement this",
            "recommended_fix": "stub returning None at module.py:10",
        }
        result = validate_pr_brutal_honesty(finding_dict=finding)
        # Should detect at least 'todo' and 'stub'
        self.assertLess(result.score, 100.0)
        markers = " ".join(result.optimism_flags).lower()
        self.assertTrue("stub" in markers or "todo" in markers)

    def test_no_input_returns_zero_with_l4_flag(self) -> None:
        result = validate_pr_brutal_honesty()
        self.assertEqual(result.score, 0.0)
        self.assertTrue(any("L4" in f for f in result.optimism_flags))


class TestPhaseSummaryScoring(unittest.TestCase):
    def test_complete_phase_summary_scores_high(self) -> None:
        summary = {
            "cycle_id": "C-2026-05-16",
            "total_findings": 3,
            "by_severity": {"CRITICAL": 1, "HIGH": 2},
            "by_status": {"MERGED": 3},
            "evidence": "artifact: docs/closure-2026-05-16.json",
        }
        result = validate_pr_brutal_honesty(phase_summary=summary)
        self.assertGreaterEqual(result.score, 90.0)
        self.assertTrue(result.evidence_present)

    def test_sparse_phase_summary_scores_low(self) -> None:
        result = validate_pr_brutal_honesty(phase_summary={"cycle_id": "X"})
        self.assertLess(result.score, 70.0)


class TestSmokePipeline(unittest.TestCase):
    def test_floor_tier_passes_in_healthy_repo(self) -> None:
        ok = run_smoke_pipeline(HonestyTier.FLOOR)
        self.assertTrue(ok, "Floor-tier smoke must pass on a clean checkout")

    def test_floor_tier_not_always_true(self) -> None:
        """Regression guard: stubbed run_smoke_pipeline used to return True
        unconditionally. We force a failure by faking a missing next-session.md."""
        with patch("scripts.bhs_validator._floor_smoke_checks") as mock_checks:
            mock_checks.return_value = ["fake failure"]
            ok = run_smoke_pipeline(HonestyTier.FLOOR)
            self.assertFalse(ok, "Smoke must return False when checks fail")

    def test_default_tier_is_floor(self) -> None:
        # Calling with no args defaults to FLOOR
        ok = run_smoke_pipeline()
        self.assertTrue(ok)


class TestOrchestratorIntegration(unittest.TestCase):
    """CD-244-02: confirm summary['avg_bhs_score'] varies with findings.

    We drive the orchestrator's closure() flow with two synthetic finding
    sets — one rich, one sparse — and assert the resulting average BHS scores
    differ. This is the runtime evidence the PR EVIDENCE: line points at.
    """

    def _build_orchestrator_with_findings(self, findings_data):
        # Default agents are fine; synthesis() does not invoke them — it sorts
        # findings and runs the BHS scorer per finding, which is exactly what
        # CD-244-02 needs to verify.
        from aep_orchestrator import AEPOrchestrator

        orch = AEPOrchestrator()
        findings = orch.discovery(findings_data, pr_number=999)
        orch.synthesis(findings)
        return orch.closure()

    def test_rich_findings_produce_higher_avg_than_sparse(self) -> None:
        rich = [
            {
                "title": "Real gap A",
                "severity": "HIGH",
                "impact": "Production endpoint server.py:120 returns wrong shape",
                "recommended_fix": "Patch handler at api.py:55; artifact in logs/outage.json",
                "effort": "S",
                "file_path": "api.py",
            },
            {
                "title": "Real gap B",
                "severity": "CRITICAL",
                "impact": "Auth bypass at auth.py:42 leaks tokens; PR #100 attempted partial fix",
                "recommended_fix": "Rewrite auth flow; reference commit a1b2c3d4e5f",
                "effort": "M",
                "file_path": "auth.py",
            },
        ]
        sparse = [
            {"title": "Vague A", "severity": "MEDIUM", "effort": "S"},
            {"title": "Vague B", "severity": "LOW", "effort": "S"},
        ]
        rich_summary = self._build_orchestrator_with_findings(rich)
        sparse_summary = self._build_orchestrator_with_findings(sparse)

        self.assertIn("avg_bhs_score", rich_summary)
        self.assertIn("avg_bhs_score", sparse_summary)
        self.assertGreater(
            rich_summary["avg_bhs_score"],
            sparse_summary["avg_bhs_score"],
            f"Rich findings should score higher than sparse findings; "
            f"got rich={rich_summary['avg_bhs_score']} "
            f"sparse={sparse_summary['avg_bhs_score']}",
        )
        # Neither should be the old hardcoded 0.0
        self.assertGreater(rich_summary["avg_bhs_score"], 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
