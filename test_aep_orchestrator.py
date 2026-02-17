"""
Unit Tests for AEP Orchestrator

Tests the AEP data models, specialist agents, tracker, and orchestrator
without requiring external services.
"""

import unittest
import json
from datetime import date
from unittest.mock import patch, MagicMock

# Patch the logger before importing aep_orchestrator so that no log files
# are created during tests.  get_logger is called at module-level inside
# SpecialistAgent.__init__, AEPTracker.__init__, and AEPOrchestrator.__init__,
# so we replace it with a factory that returns a silent mock.

_mock_logger = MagicMock()


def _fake_get_logger(*args, **kwargs):
    return _mock_logger


# Apply the patch at import time
with patch("chelation_logger.get_logger", _fake_get_logger):
    with patch.dict("sys.modules", {}):
        pass
    import aep_orchestrator
    # Also patch the module-level reference so any subsequent get_logger()
    # calls inside the imported module use the mock.
    aep_orchestrator.get_logger = _fake_get_logger

from aep_orchestrator import (
    Severity,
    FindingStatus,
    EffortSize,
    Finding,
    ScopeLock,
    VerificationResult,
    SpecialistAgent,
    ArchitectureAgent,
    SecurityAgent,
    TestingAgent,
    PerformanceAgent,
    AEPTracker,
    AEPOrchestrator,
)


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def make_finding(
    finding_id="AEP-TEST-001",
    title="Test finding",
    severity=Severity.MEDIUM,
    impact="Some impact description",
    effort=EffortSize.S,
    file_path="module/file.py",
    line_range="1-10",
    recommended_fix="Fix it",
    acceptance_criteria=None,
    dependencies=None,
    blockers=None,
    status=FindingStatus.OPEN,
    metadata=None,
):
    """Create a Finding with sensible defaults for testing."""
    return Finding(
        finding_id=finding_id,
        title=title,
        severity=severity,
        impact=impact,
        effort=effort,
        file_path=file_path,
        line_range=line_range,
        recommended_fix=recommended_fix,
        acceptance_criteria=acceptance_criteria or [],
        dependencies=dependencies or [],
        blockers=blockers or [],
        status=status,
        metadata=metadata or {},
    )


# ===========================================================================
# Test Classes
# ===========================================================================


class TestFinding(unittest.TestCase):
    """Tests for the Finding dataclass and related enums."""

    def test_finding_creation(self):
        """Create a Finding and verify all fields are set correctly."""
        f = make_finding(
            finding_id="AEP-20260212-PR002-001",
            title="Missing validation",
            severity=Severity.CRITICAL,
            impact="Arbitrary file read",
            effort=EffortSize.M,
            file_path="adapter.py",
            line_range="53-60",
            recommended_fix="Validate path",
            acceptance_criteria=["Path blocked"],
            dependencies=["AEP-20260212-PR002-002"],
        )
        self.assertEqual(f.finding_id, "AEP-20260212-PR002-001")
        self.assertEqual(f.title, "Missing validation")
        self.assertEqual(f.severity, Severity.CRITICAL)
        self.assertEqual(f.impact, "Arbitrary file read")
        self.assertEqual(f.effort, EffortSize.M)
        self.assertEqual(f.file_path, "adapter.py")
        self.assertEqual(f.line_range, "53-60")
        self.assertEqual(f.recommended_fix, "Validate path")
        self.assertEqual(f.acceptance_criteria, ["Path blocked"])
        self.assertEqual(f.dependencies, ["AEP-20260212-PR002-002"])
        self.assertEqual(f.status, FindingStatus.OPEN)
        self.assertEqual(f.owning_agent, "")

    def test_finding_id_generation(self):
        """AEPTracker.generate_finding_id produces deterministic IDs."""
        fid = AEPTracker.generate_finding_id(2, 5)
        today_str = date.today().strftime("%Y%m%d")
        expected = f"AEP-{today_str}-PR002-005"
        self.assertEqual(fid, expected)

    def test_severity_ordering(self):
        """Findings sorted by sort_key respect severity order."""
        low = make_finding(finding_id="LOW", severity=Severity.LOW, impact="x" * 10)
        high = make_finding(finding_id="HIGH", severity=Severity.HIGH, impact="x" * 10)
        crit = make_finding(finding_id="CRIT", severity=Severity.CRITICAL, impact="x" * 10)
        med = make_finding(finding_id="MED", severity=Severity.MEDIUM, impact="x" * 10)

        findings = [low, high, crit, med]
        findings.sort(key=lambda f: f.sort_key())

        self.assertEqual(findings[0].finding_id, "CRIT")
        self.assertEqual(findings[1].finding_id, "HIGH")
        self.assertEqual(findings[2].finding_id, "MED")
        self.assertEqual(findings[3].finding_id, "LOW")


class TestAEPTracker(unittest.TestCase):
    """Tests for the AEPTracker class."""

    def setUp(self):
        self.tracker = AEPTracker()

    def test_add_and_retrieve(self):
        """Add a finding and verify it appears in tracker.findings."""
        f = make_finding(finding_id="T-001")
        returned_id = self.tracker.add_finding(f)
        self.assertEqual(returned_id, "T-001")
        self.assertIn("T-001", self.tracker.findings)
        self.assertIs(self.tracker.findings["T-001"], f)

    def test_status_update(self):
        """Update a finding status and optional fields."""
        f = make_finding(finding_id="T-002")
        self.tracker.add_finding(f)
        self.tracker.update_status(
            "T-002",
            FindingStatus.IN_PROGRESS,
            owning_agent="security-agent",
        )
        self.assertEqual(self.tracker.findings["T-002"].status, FindingStatus.IN_PROGRESS)
        self.assertEqual(self.tracker.findings["T-002"].owning_agent, "security-agent")

    def test_tier_filtering(self):
        """get_tier returns only findings of the requested severity."""
        self.tracker.add_finding(make_finding(finding_id="C1", severity=Severity.CRITICAL))
        self.tracker.add_finding(make_finding(finding_id="H1", severity=Severity.HIGH))
        self.tracker.add_finding(make_finding(finding_id="M1", severity=Severity.MEDIUM))

        crit_tier = self.tracker.get_tier(Severity.CRITICAL)
        self.assertEqual(len(crit_tier), 1)
        self.assertEqual(crit_tier[0].finding_id, "C1")

    def test_tier_complete(self):
        """is_tier_complete returns True when all findings are terminal."""
        f = make_finding(finding_id="TC-001", severity=Severity.HIGH)
        self.tracker.add_finding(f)

        self.assertFalse(self.tracker.is_tier_complete(Severity.HIGH))

        self.tracker.update_status("TC-001", FindingStatus.MERGED)
        self.assertTrue(self.tracker.is_tier_complete(Severity.HIGH))

    def test_get_blocked(self):
        """get_blocked returns only BLOCKED findings."""
        f1 = make_finding(finding_id="B-001")
        f2 = make_finding(finding_id="B-002")
        self.tracker.add_finding(f1)
        self.tracker.add_finding(f2)

        self.tracker.update_status("B-001", FindingStatus.BLOCKED)
        blocked = self.tracker.get_blocked()
        self.assertEqual(len(blocked), 1)
        self.assertEqual(blocked[0].finding_id, "B-001")

    def test_markdown_table(self):
        """to_markdown_table produces a table with header and data rows."""
        self.tracker.add_finding(make_finding(finding_id="MD-001", title="First"))
        self.tracker.add_finding(
            make_finding(finding_id="MD-002", title="Second", severity=Severity.HIGH)
        )

        table = self.tracker.to_markdown_table()
        lines = table.strip().split("\n")

        # Header row + separator + 2 data rows
        self.assertGreaterEqual(len(lines), 4)
        self.assertIn("Finding ID", lines[0])
        self.assertIn("---|", lines[1])
        self.assertIn("MD-001", table)
        self.assertIn("MD-002", table)

    def test_json_export(self):
        """export_json returns valid JSON with findings and verification_log."""
        self.tracker.add_finding(make_finding(finding_id="JE-001", title="JSON Test"))
        raw = self.tracker.export_json()

        data = json.loads(raw)
        self.assertIn("findings", data)
        self.assertIn("verification_log", data)
        self.assertIn("exported_at", data)
        self.assertEqual(len(data["findings"]), 1)
        self.assertEqual(data["findings"][0]["finding_id"], "JE-001")

    def test_verification_logging(self):
        """add_verification with passed=True marks finding as VERIFIED."""
        f = make_finding(finding_id="VL-001", status=FindingStatus.MERGED)
        self.tracker.add_finding(f)

        result = VerificationResult(
            finding_id="VL-001",
            command="pytest",
            output="all passed",
            passed=True,
            agent="testing-agent",
        )
        self.tracker.add_verification(result)

        self.assertEqual(self.tracker.findings["VL-001"].status, FindingStatus.VERIFIED)
        self.assertEqual(len(self.tracker.verification_log), 1)


class TestSpecialistAgents(unittest.TestCase):
    """Tests for the four specialist agent implementations."""

    def test_architecture_agent_analyze(self):
        """ArchitectureAgent sets coupling_risk=True when title contains 'import'."""
        agent = ArchitectureAgent()
        f = make_finding(
            finding_id="ARCH-001",
            title="Circular import in module",
            file_path="src/module.py",
        )
        result = agent.analyze(f)

        self.assertTrue(result.metadata["coupling_risk"])
        self.assertEqual(result.metadata["abstraction_level"], "module")
        self.assertTrue(result.metadata["architecture_reviewed"])

    def test_security_agent_analyze(self):
        """SecurityAgent sets high risk and requires_audit for CRITICAL findings."""
        agent = SecurityAgent()
        f = make_finding(finding_id="SEC-001", severity=Severity.CRITICAL)
        result = agent.analyze(f)

        self.assertEqual(result.metadata["security_risk_level"], "high")
        self.assertTrue(result.metadata["requires_audit"])

    def test_testing_agent_analyze(self):
        """TestingAgent sets test_coverage_gap=True when acceptance_criteria is empty."""
        agent = TestingAgent()
        f = make_finding(finding_id="TST-001", acceptance_criteria=[])
        result = agent.analyze(f)

        self.assertTrue(result.metadata["test_coverage_gap"])
        self.assertIn(result.metadata["suggested_test_type"], ("unit", "integration"))

    def test_performance_agent_analyze(self):
        """PerformanceAgent sets performance_critical=True when title contains 'slow'."""
        agent = PerformanceAgent()
        f = make_finding(finding_id="PERF-001", title="Slow query execution")
        result = agent.analyze(f)

        self.assertTrue(result.metadata["performance_critical"])
        self.assertTrue(result.metadata["profiling_needed"])


class TestAEPOrchestrator(unittest.TestCase):
    """Tests for the AEPOrchestrator workflow phases."""

    def setUp(self):
        self.orchestrator = AEPOrchestrator()

    def test_scope_lock(self):
        """scope_lock returns a ScopeLock with a generated cycle_id."""
        lock = self.orchestrator.scope_lock(
            pr_range="PR#1-PR#3",
            frozen_inputs=["audit_report.md"],
        )
        self.assertIsInstance(lock, ScopeLock)
        self.assertTrue(lock.cycle_id.startswith("CYCLE-"))
        self.assertEqual(lock.pr_range, "PR#1-PR#3")
        self.assertEqual(lock.frozen_inputs, ["audit_report.md"])
        self.assertIs(self.orchestrator.scope_lock_record, lock)

    def test_discovery_deduplication(self):
        """discovery deduplicates raw findings by title (case-insensitive)."""
        raw = [
            {"title": "Missing validation", "severity": "CRITICAL", "impact": "high impact"},
            {"title": "Hardcoded secret", "severity": "HIGH", "impact": "secret leak"},
            {"title": "missing validation", "severity": "MEDIUM", "impact": "dup"},
        ]
        findings = self.orchestrator.discovery(raw, pr_number=5)
        self.assertEqual(len(findings), 2)
        titles = [f.title for f in findings]
        self.assertIn("Missing validation", titles)
        self.assertIn("Hardcoded secret", titles)

    def test_synthesis_sorting(self):
        """synthesis sorts findings by severity then impact/effort score."""
        f_low = make_finding(finding_id="S-LOW", severity=Severity.LOW, impact="x" * 10)
        f_crit = make_finding(finding_id="S-CRIT", severity=Severity.CRITICAL, impact="x" * 10)
        f_high = make_finding(finding_id="S-HIGH", severity=Severity.HIGH, impact="x" * 10)

        # Add to tracker so orchestrator has them
        for f in [f_low, f_crit, f_high]:
            self.orchestrator.tracker.add_finding(f)

        sorted_findings = self.orchestrator.synthesis([f_low, f_crit, f_high])
        self.assertEqual(sorted_findings[0].finding_id, "S-CRIT")
        self.assertEqual(sorted_findings[1].finding_id, "S-HIGH")
        self.assertEqual(sorted_findings[2].finding_id, "S-LOW")

    def test_tiered_remediation_order(self):
        """Tiered remediation processes CRITICAL before HIGH before LOW."""
        remediation_order = []

        def mock_remediate(finding):
            remediation_order.append(finding.finding_id)
            return {"pr_branch": f"fix/{finding.finding_id}"}

        f_low = make_finding(finding_id="R-LOW", severity=Severity.LOW)
        f_crit = make_finding(finding_id="R-CRIT", severity=Severity.CRITICAL)
        f_high = make_finding(finding_id="R-HIGH", severity=Severity.HIGH)

        for f in [f_low, f_crit, f_high]:
            self.orchestrator.tracker.add_finding(f)

        result = self.orchestrator.tiered_remediation(
            [f_low, f_crit, f_high],
            remediate_fn=mock_remediate,
        )

        # CRITICAL should be remediated first, then HIGH, then LOW
        self.assertEqual(remediation_order[0], "R-CRIT")
        self.assertEqual(remediation_order[1], "R-HIGH")
        self.assertEqual(remediation_order[2], "R-LOW")
        self.assertEqual(result["total_remediated"], 3)

    def test_blocked_dependency(self):
        """Finding with unresolved dependency is blocked during remediation."""
        f_dep = make_finding(finding_id="DEP-001", severity=Severity.CRITICAL)
        f_blocked = make_finding(
            finding_id="DEP-002",
            severity=Severity.CRITICAL,
            dependencies=["DEP-001"],
        )

        # Add DEP-001 but leave it OPEN (not terminal)
        self.orchestrator.tracker.add_finding(f_dep)
        self.orchestrator.tracker.add_finding(f_blocked)

        # Only remediate DEP-002 -- but DEP-001 is also in the tracker and
        # will be picked up by get_tier inside tiered_remediation.
        # DEP-001 will be remediated first (no deps), then DEP-002 should
        # NOT be blocked because DEP-001 just got set to MERGED.
        # To truly test blocking, we need DEP-001 to stay OPEN.
        # Reset: remove DEP-001 from tracker, re-add as already-in-progress
        # so it's not terminal yet.
        self.orchestrator.tracker.update_status("DEP-001", FindingStatus.IN_PROGRESS)

        # Now run remediation with a function that does NOT change DEP-001
        # since it's already IN_PROGRESS (not OPEN), the remediate loop
        # will process it again and set it to MERGED. We need a different
        # approach: put DEP-001 in a different tier (HIGH) so it doesn't
        # get processed before DEP-002 in the CRITICAL tier.
        # Actually, let's restructure: put the dependency in HIGH tier.
        self.orchestrator.tracker.findings.clear()

        f_dep = make_finding(finding_id="DEP-A", severity=Severity.HIGH)
        f_blocked = make_finding(
            finding_id="DEP-B",
            severity=Severity.CRITICAL,
            dependencies=["DEP-A"],
        )
        self.orchestrator.tracker.add_finding(f_dep)
        self.orchestrator.tracker.add_finding(f_blocked)

        # DEP-B is CRITICAL (processed first), depends on DEP-A which is
        # HIGH and still OPEN -- should be blocked.
        result = self.orchestrator.tiered_remediation(
            [f_dep, f_blocked],
            remediate_fn=lambda f: {"pr_branch": "fix"},
        )

        self.assertIn("DEP-B", result["blocked"])
        self.assertEqual(
            self.orchestrator.tracker.findings["DEP-B"].status,
            FindingStatus.BLOCKED,
        )

    def test_full_cycle(self):
        """run_full_cycle processes findings through all 7 phases."""
        raw = [
            {"title": "Critical bug", "severity": "CRITICAL", "impact": "crashes app", "effort": "S"},
            {"title": "Style issue", "severity": "LOW", "impact": "readability", "effort": "S"},
            {"title": "Security hole", "severity": "HIGH", "impact": "data leak risk", "effort": "M"},
        ]

        summary = self.orchestrator.run_full_cycle(
            raw_findings=raw,
            pr_range="PR#10-PR#12",
            pr_number=10,
        )

        self.assertEqual(summary["total_findings"], 3)
        self.assertIn("cycle_id", summary)
        self.assertIn("by_severity", summary)
        self.assertIn("by_status", summary)
        self.assertIn("markdown_tracker", summary)
        self.assertIn("json_export", summary)
        self.assertEqual(summary["by_severity"]["CRITICAL"], 1)
        self.assertEqual(summary["by_severity"]["HIGH"], 1)
        self.assertEqual(summary["by_severity"]["LOW"], 1)

    def test_verification_phase(self):
        """verification phase produces results for merged findings."""
        f = make_finding(finding_id="VER-001", severity=Severity.CRITICAL)
        self.orchestrator.tracker.add_finding(f)
        self.orchestrator.tracker.update_status("VER-001", FindingStatus.MERGED)

        results = self.orchestrator.verification()

        # Each of the 4 default agents should verify the merged finding
        self.assertEqual(len(results), 4)
        self.assertTrue(all(isinstance(r, VerificationResult) for r in results))
        # The finding should have been marked VERIFIED by at least one
        # agent whose verify() returns passed=True (MERGED status -> passed)
        self.assertEqual(
            self.orchestrator.tracker.findings["VER-001"].status,
            FindingStatus.VERIFIED,
        )

    def test_parallel_revalidation_enriches_findings(self):
        """parallel_revalidation enriches findings with agent metadata."""
        f1 = make_finding(
            finding_id="PAR-001",
            title="Circular import in module",
            severity=Severity.CRITICAL,
        )
        f2 = make_finding(
            finding_id="PAR-002",
            title="Slow query performance",
            severity=Severity.HIGH,
        )
        
        findings = [f1, f2]
        enriched = self.orchestrator.parallel_revalidation(findings)
        
        # Should return same list
        self.assertEqual(len(enriched), 2)
        self.assertIs(enriched, findings)
        
        # Each finding should have metadata from agents
        # ArchitectureAgent adds coupling_risk for "import" in title
        self.assertIn("coupling_risk", enriched[0].metadata)
        self.assertTrue(enriched[0].metadata["coupling_risk"])
        
        # PerformanceAgent adds performance_critical for "slow" in title
        self.assertIn("performance_critical", enriched[1].metadata)
        self.assertTrue(enriched[1].metadata["performance_critical"])
        
        # SecurityAgent should have analyzed both (adds security_risk_level)
        self.assertIn("security_risk_level", enriched[0].metadata)
        self.assertIn("security_risk_level", enriched[1].metadata)

    def test_parallel_revalidation_handles_agent_exception(self):
        """parallel_revalidation logs errors when agent.analyze raises exception."""
        # Create a custom agent that raises an exception
        class FailingAgent(SpecialistAgent):
            def __init__(self):
                super().__init__("failing-agent", "failing")
            
            def analyze(self, finding: Finding, context=None) -> Finding:
                raise ValueError("Agent analysis failed")
        
        # Add the failing agent to orchestrator
        self.orchestrator.agents.append(FailingAgent())
        
        f = make_finding(finding_id="ERR-001", severity=Severity.HIGH)
        findings = [f]
        
        # Should not raise, should log error and continue
        result = self.orchestrator.parallel_revalidation(findings)
        
        # Should still return the findings
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].finding_id, "ERR-001")
        
        # Verify log_error was called (check mock)
        self.assertTrue(_mock_logger.log_error.called)
        calls = [str(call) for call in _mock_logger.log_error.call_args_list]
        error_logged = any("failing-agent" in str(call) for call in calls)
        self.assertTrue(error_logged, "Expected error to be logged for failing agent")


if __name__ == "__main__":
    unittest.main(verbosity=2)
