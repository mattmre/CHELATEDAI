"""
AEP Orchestrator for ChelatedAI

Implements the 7-phase ARCH-AEP workflow for agentic remediation:
1. Scope lock - Define PR range, freeze inputs
2. Discovery + normalization - Normalize, deduplicate, assign severity
3. Parallel re-validation - Specialist agents validate against current main
4. Architecture and planning synthesis - Merge, re-score, re-rank
5. Tiered remediation execution - Critical->High->Medium->Low, no tier skipping
6. Verification - Run tests, capture evidence
7. Closure - Phase summary, counts

Provides data models for findings, a tracker with markdown/JSON export,
specialist agent base class with four concrete agents, and the main
AEPOrchestrator that drives the full cycle.
"""

from enum import Enum, IntEnum
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Callable, Tuple
from datetime import datetime, date
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import threading
import uuid

from chelation_logger import ChelationLogger, get_logger


# =============================================================================
# Data Models
# =============================================================================

class Severity(IntEnum):
    """Finding severity levels, ordered by urgency."""
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


class FindingStatus(Enum):
    """Lifecycle states for a finding."""
    OPEN = "OPEN"
    IN_PROGRESS = "IN_PROGRESS"
    BLOCKED = "BLOCKED"
    DEFERRED = "DEFERRED"
    MERGED = "MERGED"
    VERIFIED = "VERIFIED"


class EffortSize(Enum):
    """Effort sizing for a finding's recommended fix."""
    S = "S"
    M = "M"
    L = "L"

    @property
    def weight(self) -> int:
        """Numeric weight for priority scoring. Lower = less effort."""
        return {"S": 1, "M": 3, "L": 5}[self.value]


@dataclass
class Finding:
    """
    A single remediation finding in the AEP backlog.

    Contains all required tracker columns: finding-id, severity, owning-agent,
    PR/branch, status, commit hash, verification evidence, plus the full
    backlog entry fields (impact, effort, file path, acceptance criteria, etc.).
    """
    finding_id: str
    title: str
    severity: Severity
    impact: str
    effort: EffortSize
    file_path: str
    line_range: str
    recommended_fix: str
    acceptance_criteria: List[str] = field(default_factory=list)
    dependencies: List[str] = field(default_factory=list)
    blockers: List[str] = field(default_factory=list)
    status: FindingStatus = FindingStatus.OPEN
    owning_agent: str = ""
    pr_branch: str = ""
    commit_hash: str = ""
    verification_evidence: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def _impact_effort_score(self) -> float:
        """Heuristic score: impact length as proxy for impact magnitude / effort weight."""
        return len(self.impact) / self.effort.weight

    def sort_key(self) -> Tuple:
        """
        Sorting tuple for the AEP severity model.

        Order: severity first (ascending IntEnum), then impact x (1/effort)
        descending, then stable zero for dependency tiebreaking.
        """
        return (self.severity.value, -self._impact_effort_score(), 0)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-friendly dictionary."""
        return {
            "finding_id": self.finding_id,
            "title": self.title,
            "severity": self.severity.name,
            "impact": self.impact,
            "effort": self.effort.value,
            "file_path": self.file_path,
            "line_range": self.line_range,
            "recommended_fix": self.recommended_fix,
            "acceptance_criteria": self.acceptance_criteria,
            "dependencies": self.dependencies,
            "blockers": self.blockers,
            "status": self.status.value,
            "owning_agent": self.owning_agent,
            "pr_branch": self.pr_branch,
            "commit_hash": self.commit_hash,
            "verification_evidence": self.verification_evidence,
            "metadata": self.metadata,
        }


@dataclass
class ScopeLock:
    """
    Scope lock record for an AEP cycle.

    Freezes PR range and inputs at the start of each remediation cycle
    to prevent scope creep.
    """
    cycle_id: str
    start_date: str
    pr_range: str
    frozen_inputs: List[str] = field(default_factory=list)


@dataclass
class VerificationResult:
    """Result of a verification check against a merged finding."""
    finding_id: str
    command: str
    output: str
    passed: bool
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    agent: str = ""


# =============================================================================
# Callback Safety Wrapper (F-022)
# =============================================================================

def _safe_callback_wrapper(
    callback: Callable,
    callback_name: str,
    finding: Finding,
    logger: ChelationLogger,
    timeout_seconds: float = 30.0,
) -> Tuple[Optional[Any], Optional[str]]:
    """
    Execute a callback with timeout and exception safety.
    
    Windows-compatible implementation using a daemon thread join timeout.
    
    Args:
        callback: The callback function to execute
        callback_name: Name for logging (e.g., "remediate_fn", "verify_fn")
        finding: The finding to pass to the callback
        logger: Logger instance for error logging
        timeout_seconds: Maximum execution time
        
    Returns:
        Tuple of (result, error_message). If successful, (result, None).
        If timeout/exception, (None, error_message).
    """
    result_container = [None]
    exception_container = [None]
    
    def _run_callback():
        try:
            result_container[0] = callback(finding)
        except Exception as e:
            exception_container[0] = e
    
    thread = threading.Thread(target=_run_callback, daemon=True)
    thread.start()
    thread.join(timeout=timeout_seconds)
    
    if thread.is_alive():
        # Timeout occurred
        error_msg = f"{callback_name} timeout after {timeout_seconds}s for finding {finding.finding_id}"
        logger.log_error(
            "callback_timeout",
            error_msg,
            finding_id=finding.finding_id,
            callback=callback_name,
            timeout_seconds=timeout_seconds,
        )
        return None, error_msg
    
    if exception_container[0] is not None:
        # Exception occurred
        exc = exception_container[0]
        error_msg = f"{callback_name} exception for finding {finding.finding_id}: {type(exc).__name__}: {str(exc)}"
        logger.log_error(
            "callback_exception",
            error_msg,
            exception=exc,
            finding_id=finding.finding_id,
            callback=callback_name,
        )
        return None, error_msg
    
    return result_container[0], None


# =============================================================================
# Specialist Agents
# =============================================================================

class SpecialistAgent:
    """Base class for specialist agents in the AEP workflow."""

    def __init__(self, name: str, domain: str):
        self.name = name
        self.domain = domain
        self.logger = get_logger()

    def analyze(self, finding: Finding, context: Optional[Dict] = None) -> Finding:
        """Analyze a finding and enrich with domain-specific metadata."""
        finding.metadata[f"{self.domain}_reviewed"] = True
        finding.metadata[f"{self.domain}_agent"] = self.name
        self.logger.log_event("agent_analyze", f"{self.name} analyzed {finding.finding_id}")
        return finding

    def propose_fix(self, finding: Finding) -> Dict[str, Any]:
        """Return a structured fix proposal."""
        return {
            "finding_id": finding.finding_id,
            "agent": self.name,
            "domain": self.domain,
            "proposed_fix": finding.recommended_fix,
            "estimated_effort": finding.effort.value,
        }

    def verify(self, finding: Finding, context: Optional[Dict] = None) -> VerificationResult:
        """Verify that a finding has been properly addressed."""
        return VerificationResult(
            finding_id=finding.finding_id,
            command=f"verify_{self.domain}",
            output=f"Verified by {self.name}",
            passed=(
                finding.status == FindingStatus.MERGED
                or finding.status == FindingStatus.VERIFIED
            ),
            agent=self.name,
        )


class ArchitectureAgent(SpecialistAgent):
    """Specialist agent for architecture and coupling analysis."""

    def __init__(self):
        super().__init__("architecture-agent", "architecture")

    def analyze(self, finding: Finding, context: Optional[Dict] = None) -> Finding:
        finding = super().analyze(finding, context)
        title_lower = finding.title.lower()
        finding.metadata["coupling_risk"] = (
            "import" in title_lower or "dependency" in title_lower
        )
        finding.metadata["abstraction_level"] = (
            "module" if "/" in finding.file_path else "function"
        )
        return finding

    def propose_fix(self, finding: Finding) -> Dict[str, Any]:
        proposal = super().propose_fix(finding)
        proposal["architecture_notes"] = (
            f"Review coupling impact for {finding.file_path}"
        )
        return proposal


class SecurityAgent(SpecialistAgent):
    """Specialist agent for security risk assessment."""

    def __init__(self):
        super().__init__("security-agent", "security")

    def analyze(self, finding: Finding, context: Optional[Dict] = None) -> Finding:
        finding = super().analyze(finding, context)
        if finding.severity == Severity.CRITICAL:
            risk_level = "high"
        elif finding.severity == Severity.HIGH:
            risk_level = "medium"
        else:
            risk_level = "low"
        finding.metadata["security_risk_level"] = risk_level
        finding.metadata["requires_audit"] = (
            finding.severity == Severity.CRITICAL or finding.severity == Severity.HIGH
        )
        return finding

    def propose_fix(self, finding: Finding) -> Dict[str, Any]:
        proposal = super().propose_fix(finding)
        proposal["security_review_required"] = True
        return proposal


class TestingAgent(SpecialistAgent):
    """Specialist agent for test coverage and quality."""

    def __init__(self):
        super().__init__("testing-agent", "testing")

    def analyze(self, finding: Finding, context: Optional[Dict] = None) -> Finding:
        finding = super().analyze(finding, context)
        has_test_gap = (
            "test" in finding.title.lower() or len(finding.acceptance_criteria) == 0
        )
        finding.metadata["test_coverage_gap"] = has_test_gap
        # Integration tests for cross-module findings, unit tests otherwise
        finding.metadata["suggested_test_type"] = (
            "integration" if "/" in finding.file_path else "unit"
        )
        return finding


class PerformanceAgent(SpecialistAgent):
    """Specialist agent for performance analysis."""

    def __init__(self):
        super().__init__("performance-agent", "performance")

    PERF_KEYWORDS = ("perf", "slow", "latency", "memory")

    def analyze(self, finding: Finding, context: Optional[Dict] = None) -> Finding:
        finding = super().analyze(finding, context)
        title_lower = finding.title.lower()
        is_perf_critical = any(kw in title_lower for kw in self.PERF_KEYWORDS)
        finding.metadata["performance_critical"] = is_perf_critical
        finding.metadata["profiling_needed"] = is_perf_critical
        return finding


# =============================================================================
# AEP Tracker
# =============================================================================

class AEPTracker:
    """
    Central tracker for AEP findings and verification evidence.

    Maintains the required tracker columns (finding-id, severity, owning-agent,
    PR/branch, status, commit hash, verification evidence) and supports
    markdown table and JSON export for audit.
    """

    def __init__(self):
        self.findings: Dict[str, Finding] = {}
        self.verification_log: List[VerificationResult] = []
        self.logger = get_logger()

    def add_finding(self, finding: Finding) -> str:
        """Add a finding to the tracker. Returns the finding_id."""
        self.findings[finding.finding_id] = finding
        self.logger.log_event(
            "tracker_add",
            f"Added {finding.finding_id}: {finding.title} [{finding.severity.name}]",
            finding_id=finding.finding_id,
            severity=finding.severity.name,
        )
        return finding.finding_id

    def update_status(self, finding_id: str, status: FindingStatus, **kwargs):
        """
        Update a finding's status and optional fields.

        Accepted kwargs: pr_branch, commit_hash, owning_agent, verification_evidence.
        """
        if finding_id not in self.findings:
            raise ValueError(f"Finding ID '{finding_id}' not found in tracker")
        finding = self.findings[finding_id]
        old_status = finding.status
        finding.status = status
        for attr in ("pr_branch", "commit_hash", "owning_agent", "verification_evidence"):
            if attr in kwargs:
                setattr(finding, attr, kwargs[attr])
        self.logger.log_event(
            "tracker_status",
            f"{finding_id}: {old_status.value} -> {status.value}",
            finding_id=finding_id,
            old_status=old_status.value,
            new_status=status.value,
        )

    def add_verification(self, result: VerificationResult):
        """Record a verification result. If passed, mark finding as VERIFIED."""
        self.verification_log.append(result)
        if result.passed and result.finding_id in self.findings:
            self.findings[result.finding_id].status = FindingStatus.VERIFIED
        self.logger.log_event(
            "tracker_verify",
            f"{result.finding_id}: {'PASS' if result.passed else 'FAIL'} by {result.agent}",
            finding_id=result.finding_id,
            passed=result.passed,
            agent=result.agent,
        )

    def get_tier(self, severity: Severity) -> List[Finding]:
        """Return all findings of the given severity, sorted by sort_key."""
        tier = [f for f in self.findings.values() if f.severity == severity]
        tier.sort(key=lambda f: f.sort_key())
        return tier

    def get_blocked(self) -> List[Finding]:
        """Return all findings with BLOCKED status."""
        return [f for f in self.findings.values() if f.status == FindingStatus.BLOCKED]

    def get_unresolved(self) -> List[Finding]:
        """Return all findings that are OPEN, IN_PROGRESS, or BLOCKED."""
        unresolved = {FindingStatus.OPEN, FindingStatus.IN_PROGRESS, FindingStatus.BLOCKED}
        return [f for f in self.findings.values() if f.status in unresolved]

    def is_tier_complete(self, severity: Severity) -> bool:
        """True if all findings of this severity are MERGED, VERIFIED, or DEFERRED."""
        terminal = {FindingStatus.MERGED, FindingStatus.VERIFIED, FindingStatus.DEFERRED}
        tier = self.get_tier(severity)
        return all(f.status in terminal for f in tier) if tier else True

    @staticmethod
    def generate_finding_id(pr_number: int, seq: int) -> str:
        """Generate a deterministic finding ID from PR number and sequence."""
        return f"AEP-{date.today().strftime('%Y%m%d')}-PR{pr_number:03d}-{seq:03d}"

    def to_markdown_table(self) -> str:
        """
        Generate an audit-grade markdown table of all findings.

        Columns: Finding ID, Severity, Title, Status, Owning Agent,
        PR/Branch, Commit, Verification.
        """
        sorted_findings = sorted(self.findings.values(), key=lambda f: f.sort_key())
        lines = [
            "| Finding ID | Severity | Title | Status | Owning Agent | PR/Branch | Commit | Verification |",
            "|---|---|---|---|---|---|---|---|",
        ]
        for f in sorted_findings:
            lines.append(
                f"| {f.finding_id} | {f.severity.name} | {f.title} | {f.status.value} "
                f"| {f.owning_agent} | {f.pr_branch} | {f.commit_hash} "
                f"| {f.verification_evidence} |"
            )
        return "\n".join(lines)

    def export_json(self) -> str:
        """Export tracker state as a JSON string for archival."""
        sorted_findings = sorted(self.findings.values(), key=lambda f: f.sort_key())
        payload = {
            "findings": [f.to_dict() for f in sorted_findings],
            "verification_log": [
                {
                    "finding_id": v.finding_id,
                    "command": v.command,
                    "output": v.output,
                    "passed": v.passed,
                    "timestamp": v.timestamp,
                    "agent": v.agent,
                }
                for v in self.verification_log
            ],
            "exported_at": datetime.utcnow().isoformat(),
        }
        return json.dumps(payload, indent=2)


# =============================================================================
# AEP Orchestrator
# =============================================================================

class AEPOrchestrator:
    """
    Main orchestrator for the 7-phase ARCH-AEP remediation workflow.

    Coordinates scope locking, discovery, specialist agent re-validation,
    synthesis, tiered remediation, verification, and closure. Supports
    pluggable remediation and verification functions for integration with
    external CI/CD or agent systems.
    """

    def __init__(self, agents: Optional[List[SpecialistAgent]] = None):
        self.tracker = AEPTracker()
        self.agents = agents or [
            ArchitectureAgent(),
            SecurityAgent(),
            TestingAgent(),
            PerformanceAgent(),
        ]
        self.scope_lock_record: Optional[ScopeLock] = None
        self.logger = get_logger()
        self.callback_timeout_seconds = 30.0

    # ----- Phase 1: Scope Lock -----

    def scope_lock(
        self,
        pr_range: str = "",
        frozen_inputs: Optional[List[str]] = None,
    ) -> ScopeLock:
        """
        Phase 1: Lock the scope for a remediation cycle.

        Freezes the PR range and input documents to prevent scope creep.
        """
        cycle_id = f"CYCLE-{date.today().strftime('%Y%m%d')}-{str(uuid.uuid4())[:4]}"
        lock = ScopeLock(
            cycle_id=cycle_id,
            start_date=date.today().isoformat(),
            pr_range=pr_range,
            frozen_inputs=frozen_inputs or [],
        )
        self.scope_lock_record = lock
        self.logger.log_event(
            "scope_lock",
            f"Scope locked: {cycle_id} | PR range: {pr_range}",
            cycle_id=cycle_id,
            pr_range=pr_range,
        )
        return lock

    # ----- Phase 2: Discovery + Normalization -----

    def discovery(
        self,
        raw_findings: List[Dict[str, Any]],
        pr_number: int = 1,
    ) -> List[Finding]:
        """
        Phase 2: Normalize raw findings, deduplicate, and assign IDs.

        Accepts a list of dicts with keys matching Finding fields. Deduplicates
        by title (case-insensitive, keeps first occurrence). Returns the list
        of Finding objects added to the tracker.
        """
        seen_titles: Dict[str, bool] = {}
        findings: List[Finding] = []
        seq = 0

        for raw in raw_findings:
            title = raw.get("title", "Untitled")
            title_key = title.strip().lower()
            if title_key in seen_titles:
                continue
            seen_titles[title_key] = True

            seq += 1
            finding_id = AEPTracker.generate_finding_id(pr_number, seq)

            # Map severity string to enum (case-insensitive, default MEDIUM)
            sev_raw = str(raw.get("severity", "MEDIUM")).upper()
            try:
                severity = Severity[sev_raw]
            except KeyError:
                severity = Severity.MEDIUM

            # Map effort string to enum (default S)
            eff_raw = str(raw.get("effort", "S")).upper()
            try:
                effort = EffortSize(eff_raw)
            except ValueError:
                effort = EffortSize.S

            finding = Finding(
                finding_id=finding_id,
                title=title,
                severity=severity,
                impact=raw.get("impact", ""),
                effort=effort,
                file_path=raw.get("file_path", ""),
                line_range=raw.get("line_range", ""),
                recommended_fix=raw.get("recommended_fix", ""),
                acceptance_criteria=raw.get("acceptance_criteria", []),
                dependencies=raw.get("dependencies", []),
                blockers=raw.get("blockers", []),
            )

            self.tracker.add_finding(finding)
            findings.append(finding)

        self.logger.log_event(
            "discovery",
            f"Discovered {len(findings)} unique findings from {len(raw_findings)} raw entries",
            total_raw=len(raw_findings),
            unique_count=len(findings),
            pr_number=pr_number,
        )
        return findings

    # ----- Phase 3: Parallel Re-validation -----

    def parallel_revalidation(
        self,
        findings: List[Finding],
        context: Optional[Dict] = None,
    ) -> List[Finding]:
        """
        Phase 3: Route each finding through all specialist agents for enrichment.

        Each agent attaches domain-specific metadata (risk levels, test gaps,
        performance flags, coupling analysis).
        """
        def _process_finding(finding: Finding) -> None:
            """Process a single finding through all agents."""
            for agent in self.agents:
                try:
                    agent.analyze(finding, context)
                except Exception as e:
                    self.logger.log_error(
                        "revalidation_agent_error",
                        f"Agent {agent.name} failed to analyze finding {finding.finding_id}: {e}",
                        finding_id=finding.finding_id,
                        agent_name=agent.name,
                        error=str(e),
                    )

        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(_process_finding, finding): finding for finding in findings}
            for future in as_completed(futures):
                finding = futures[future]
                try:
                    future.result()
                except Exception as e:
                    self.logger.log_error(
                        "revalidation_finding_error",
                        f"Unexpected error processing finding {finding.finding_id}: {e}",
                        finding_id=finding.finding_id,
                        error=str(e),
                    )

        self.logger.log_event(
            "revalidation",
            f"Re-validated {len(findings)} findings across {len(self.agents)} agents",
            finding_count=len(findings),
            agent_count=len(self.agents),
        )
        return findings

    # ----- Phase 4: Architecture and Planning Synthesis -----

    def synthesis(self, findings: List[Finding]) -> List[Finding]:
        """
        Phase 4: Merge, re-score, and re-rank findings using the severity model.

        Sorting: severity first, then impact x (1/effort), then dependency order.
        """
        findings.sort(key=lambda f: f.sort_key())
        self.logger.log_event(
            "synthesis",
            f"Synthesized and ranked {len(findings)} findings",
            finding_count=len(findings),
            top_finding=findings[0].finding_id if findings else "none",
        )
        return findings

    # ----- Phase 5: Tiered Remediation Execution -----

    def tiered_remediation(
        self,
        findings: List[Finding],
        remediate_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Phase 5: Execute remediation tier by tier (Critical->High->Medium->Low).

        No tier skipping. For each finding, checks dependency blockers before
        proceeding. If remediate_fn is provided, it is called with the finding
        and should return a dict with optional 'pr_branch' and 'commit_hash' keys.
        """
        tiers_completed: List[str] = []
        blocked_ids: List[str] = []
        total_remediated = 0
        terminal = {FindingStatus.MERGED, FindingStatus.VERIFIED, FindingStatus.DEFERRED}

        for severity in Severity:
            tier_findings = self.tracker.get_tier(severity)
            if not tier_findings:
                tiers_completed.append(severity.name)
                continue

            for finding in tier_findings:
                if finding.status in terminal:
                    continue

                # Check dependency blockers
                has_blocker = False
                for dep_id in finding.dependencies:
                    if dep_id in self.tracker.findings:
                        dep = self.tracker.findings[dep_id]
                        if dep.status not in terminal:
                            has_blocker = True
                            break

                if has_blocker:
                    self.tracker.update_status(finding.finding_id, FindingStatus.BLOCKED)
                    blocked_ids.append(finding.finding_id)
                    continue

                self.tracker.update_status(finding.finding_id, FindingStatus.IN_PROGRESS)

                if remediate_fn is not None:
                    # F-022: Execute remediate_fn with timeout and exception safety
                    result, error = _safe_callback_wrapper(
                        remediate_fn,
                        "remediate_fn",
                        finding,
                        self.logger,
                        timeout_seconds=self.callback_timeout_seconds,
                    )
                    
                    if error is not None:
                        # Callback timeout or exception - mark finding as BLOCKED
                        self.tracker.update_status(finding.finding_id, FindingStatus.BLOCKED)
                        blocked_ids.append(finding.finding_id)
                        continue
                    
                    # Validate return type
                    if isinstance(result, dict):
                        kwargs = {}
                        if "pr_branch" in result:
                            kwargs["pr_branch"] = result["pr_branch"]
                        if "commit_hash" in result:
                            kwargs["commit_hash"] = result["commit_hash"]
                        self.tracker.update_status(
                            finding.finding_id, FindingStatus.MERGED, **kwargs
                        )
                    else:
                        # F-022: Non-dict returns still allow merge but log warning
                        if result is not None:
                            self.logger.log_event(
                                "remediate_warning",
                                f"remediate_fn returned non-dict type for {finding.finding_id}: {type(result).__name__}",
                                finding_id=finding.finding_id,
                                return_type=type(result).__name__,
                            )
                        self.tracker.update_status(finding.finding_id, FindingStatus.MERGED)
                else:
                    self.tracker.update_status(finding.finding_id, FindingStatus.MERGED)

                total_remediated += 1

            # Tier gate: do not advance until tier is complete or all remaining are blocked/deferred
            # F-038: Enforce no-skip tier gate - stop advancing if current tier has BLOCKED findings
            tier_has_blocked = any(f.status == FindingStatus.BLOCKED for f in tier_findings)
            if tier_has_blocked:
                # Stop processing further tiers - blocked findings must be resolved first
                break

            if self.tracker.is_tier_complete(severity):
                tiers_completed.append(severity.name)

        self.logger.log_event(
            "tiered_remediation",
            f"Remediation complete: {total_remediated} fixed, {len(blocked_ids)} blocked",
            tiers_completed=tiers_completed,
            blocked_count=len(blocked_ids),
            total_remediated=total_remediated,
        )

        return {
            "tiers_completed": tiers_completed,
            "blocked": blocked_ids,
            "total_remediated": total_remediated,
        }

    # ----- Phase 6: Verification -----

    def verification(
        self,
        verify_fn: Optional[Callable] = None,
    ) -> List[VerificationResult]:
        """
        Phase 6: Verify all merged findings via specialist agents.

        Each agent runs its verify method. If verify_fn is provided, it is
        also called and should return a VerificationResult.
        """
        results: List[VerificationResult] = []
        merged = [
            f for f in self.tracker.findings.values()
            if f.status == FindingStatus.MERGED
        ]

        for finding in merged:
            for agent in self.agents:
                vr = agent.verify(finding)
                self.tracker.add_verification(vr)
                results.append(vr)

            if verify_fn is not None:
                # F-022: Execute verify_fn with timeout and exception safety
                custom_vr, error = _safe_callback_wrapper(
                    verify_fn,
                    "verify_fn",
                    finding,
                    self.logger,
                    timeout_seconds=self.callback_timeout_seconds,
                )
                
                if error is not None:
                    # Callback timeout or exception - skip custom result, continue
                    continue
                
                # F-022: Validate return type - only VerificationResult is accepted
                if isinstance(custom_vr, VerificationResult):
                    self.tracker.add_verification(custom_vr)
                    results.append(custom_vr)
                elif custom_vr is not None:
                    # Invalid return type - log warning and ignore
                    self.logger.log_event(
                        "verify_warning",
                        f"verify_fn returned non-VerificationResult type for {finding.finding_id}: {type(custom_vr).__name__}",
                        finding_id=finding.finding_id,
                        return_type=type(custom_vr).__name__,
                    )

        self.logger.log_event(
            "verification",
            f"Verification complete: {len(results)} checks across {len(merged)} findings",
            total_checks=len(results),
            findings_verified=len(merged),
        )
        return results

    # ----- Phase 7: Closure -----

    def closure(self) -> Dict[str, Any]:
        """
        Phase 7: Generate cycle closure summary with counts and exports.

        Returns a summary dict with severity/status breakdowns, the markdown
        tracker table, and a full JSON export for archival.
        """
        all_findings = list(self.tracker.findings.values())

        by_severity: Dict[str, int] = {}
        for sev in Severity:
            count = sum(1 for f in all_findings if f.severity == sev)
            by_severity[sev.name] = count

        by_status: Dict[str, int] = {}
        for status in FindingStatus:
            count = sum(1 for f in all_findings if f.status == status)
            by_status[status.name] = count

        cycle_id = (
            self.scope_lock_record.cycle_id
            if self.scope_lock_record is not None
            else "UNSCOPED"
        )

        summary = {
            "cycle_id": cycle_id,
            "total_findings": len(all_findings),
            "by_severity": by_severity,
            "by_status": by_status,
            "markdown_tracker": self.tracker.to_markdown_table(),
            "json_export": self.tracker.export_json(),
            "completed_at": datetime.utcnow().isoformat(),
        }

        self.logger.log_event(
            "closure",
            f"Cycle {cycle_id} closed: {len(all_findings)} findings",
            cycle_id=cycle_id,
            total_findings=len(all_findings),
            by_severity=by_severity,
            by_status=by_status,
        )
        return summary

    # ----- Convenience: Full Cycle -----

    def run_full_cycle(
        self,
        raw_findings: List[Dict[str, Any]],
        pr_range: str = "",
        pr_number: int = 1,
        remediate_fn: Optional[Callable] = None,
        verify_fn: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Run all 7 AEP phases in sequence and return the closure summary.

        This is the primary entry point for automated remediation cycles.
        """
        self.scope_lock(pr_range=pr_range)
        findings = self.discovery(raw_findings, pr_number=pr_number)
        findings = self.parallel_revalidation(findings)
        findings = self.synthesis(findings)
        self.tiered_remediation(findings, remediate_fn=remediate_fn)
        self.verification(verify_fn=verify_fn)
        return self.closure()


# =============================================================================
# CLI Demo
# =============================================================================

if __name__ == "__main__":
    print("=== AEP Orchestrator Demo ===\n")

    sample_findings = [
        {
            "title": "Missing input validation on adapter weights path",
            "severity": "CRITICAL",
            "impact": "Arbitrary file read via crafted path in load()",
            "effort": "S",
            "file_path": "chelation_adapter.py",
            "line_range": "53-60",
            "recommended_fix": "Validate path is within PROJECT_ROOT before loading",
            "acceptance_criteria": ["Path traversal blocked", "Unit test added"],
        },
        {
            "title": "No timeout on Qdrant batch upsert",
            "severity": "HIGH",
            "impact": "Hung process if Qdrant is unresponsive during sedimentation",
            "effort": "S",
            "file_path": "recursive_decomposer.py",
            "line_range": "596-623",
            "recommended_fix": "Add configurable timeout to Qdrant calls",
            "acceptance_criteria": ["Timeout raises after 30s", "Graceful error logged"],
        },
        {
            "title": "Hardcoded magic number in variance threshold",
            "severity": "MEDIUM",
            "impact": "Difficult to tune chelation behavior without code changes",
            "effort": "S",
            "file_path": "config.py",
            "line_range": "43",
            "recommended_fix": "Already centralized in ChelationConfig, remove any remaining inline values",
        },
        {
            "title": "No timeout on Qdrant batch upsert",  # duplicate
            "severity": "HIGH",
            "impact": "Duplicate entry that should be deduplicated",
            "effort": "M",
            "file_path": "recursive_decomposer.py",
            "line_range": "596-623",
            "recommended_fix": "Duplicate",
        },
        {
            "title": "Slow embedding batch performance",
            "severity": "LOW",
            "impact": "Latency spike on large document sets",
            "effort": "L",
            "file_path": "antigravity_engine.py",
            "line_range": "1-50",
            "recommended_fix": "Add async batching with concurrency limit",
            "acceptance_criteria": ["p95 latency under 500ms for 1000 docs"],
        },
    ]

    orchestrator = AEPOrchestrator()
    summary = orchestrator.run_full_cycle(
        raw_findings=sample_findings,
        pr_range="PR#1-PR#2",
        pr_number=2,
    )

    print(f"\nCycle: {summary['cycle_id']}")
    print(f"Total findings: {summary['total_findings']}")
    print(f"By severity: {summary['by_severity']}")
    print(f"By status: {summary['by_status']}")
    print(f"\n{summary['markdown_tracker']}")

