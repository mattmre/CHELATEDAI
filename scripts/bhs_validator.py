#!/usr/bin/env python3
"""BHS v3.3 Validator — orchestrator-side honesty scoring.

This module is the BHS hook called by ``aep_orchestrator.py`` during synthesis,
remediation, and closure. It scores AEP ``finding_dict``s and ``phase_summary``
dicts on a 0–100 honesty scale and exposes a smoke pipeline runner that
validates the orchestrator's runtime surface at floor or ceiling tier.

Why this is NOT the same code as ``scripts/validate_pr_brutal_honesty.py``:
that validator parses PR-body field schemas (EVIDENCE/SMOKE/BHS_* lines). The
orchestrator never hands us a PR body — it hands us small dicts describing a
finding or a phase outcome. The schema is different, so the scoring logic is
different. Both validators share the rulebook's lie taxonomy (L1–L13) by name
but operate on disjoint inputs.

Scoring signals (additive penalties from 100):
  - Missing ``id`` / ``severity`` / ``recommended_fix`` / ``impact`` fields
    (each absence is L4 partial-as-complete on the orchestrator's side).
  - ``severity`` equal to ``UNKNOWN`` / empty / ``None`` — orchestrator did
    not commit to a tier (L4).
  - No evidence pointer in any text field (regex for ``file:line``, ``PR #``,
    ``artifact``, or a path with extension). Findings without evidence are
    L5 test-as-truth precursors.
  - L1–L13 marker words in any text field (``stub``, ``TODO``,
    ``NotImplementedError``, ``placeholder``, ``mock``, ``fake``).
"""

from __future__ import annotations

import importlib
import re
import sys
import traceback
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

# Repo root is one level above /scripts. We need it on sys.path so we can
# import aep_orchestrator.py during floor-tier smoke.
_REPO_ROOT = Path(__file__).resolve().parent.parent


class HonestyTier(Enum):
    FLOOR = "floor"
    CEILING = "ceiling"


@dataclass
class BHSResult:
    score: float  # 0.0 – 100.0
    tier: HonestyTier
    evidence_present: bool
    optimism_flags: List[str] = field(default_factory=list)
    drift_detected: bool = False
    notes: str = ""


# --- Scoring constants ------------------------------------------------------

# Required keys on the orchestrator's finding dict. Each missing key = -15.
_REQUIRED_FINDING_KEYS = ("id", "severity", "recommended_fix", "impact")

# Required keys on a phase summary. Each missing key = -10.
_REQUIRED_PHASE_KEYS = ("cycle_id", "total_findings", "by_severity", "by_status")

# Lie-taxonomy keywords. Each match in any text field penalises the score.
_LIE_MARKERS = (
    "stub",
    "todo",
    "fixme",
    "notimplemented",
    "not implemented",
    "placeholder",
    "fake",
    "mocked",  # plain "mock" matches "mockingbird" etc.; "mocked" is stricter
    "scaffolded",
    "tbd",
)

# Cheap evidence regex: matches file:line, PR refs, artifact paths, command
# transcripts. Used to award the "evidence_present" credit.
_EVIDENCE_RE = re.compile(
    r"""
    (
        [\w./\\-]+\.[a-zA-Z0-9]{1,5}(:\d+)?   |  # path with extension, optional :line
        \bPR\s*\#\s*\d+                       |  # PR #123
        \bartifact[s]?[/:]                    |  # artifact: or artifacts/
        \bcommit\s+[0-9a-f]{7,40}             |  # commit hash
        \b(SHA|sha)\s*[:=]\s*[0-9a-f]{7,40}
    )
    """,
    re.VERBOSE,
)


def _collect_text(d: Dict[str, Any]) -> str:
    """Flatten every str-valued field into one searchable blob."""
    parts: List[str] = []
    for v in d.values():
        if isinstance(v, str):
            parts.append(v)
        elif isinstance(v, (list, tuple)):
            for item in v:
                if isinstance(item, str):
                    parts.append(item)
        elif isinstance(v, dict):
            parts.append(_collect_text(v))
    return " \n ".join(parts)


def _score_finding(finding_dict: Dict[str, Any]) -> BHSResult:
    score = 100.0
    flags: List[str] = []

    # Missing required keys → L4
    for key in _REQUIRED_FINDING_KEYS:
        if key not in finding_dict or finding_dict.get(key) in (None, ""):
            score -= 15.0
            flags.append(f"L4: missing/empty finding field {key!r}")

    # Severity must be a non-UNKNOWN tier
    severity = str(finding_dict.get("severity", "")).strip().upper()
    if severity in ("", "UNKNOWN", "NONE"):
        score -= 10.0
        flags.append("L4: severity not committed (UNKNOWN/empty)")

    text_blob = _collect_text(finding_dict)
    text_lower = text_blob.lower()

    # Lie-marker keywords
    for marker in _LIE_MARKERS:
        if marker in text_lower:
            score -= 5.0
            flags.append(f"L1/L4: lie-marker keyword present: {marker!r}")

    # Evidence pointer present?
    evidence_present = bool(_EVIDENCE_RE.search(text_blob))
    if not evidence_present:
        score -= 20.0
        flags.append("L5: no evidence pointer (file:line, PR #, artifact, commit) in finding text")

    score = max(0.0, min(100.0, score))
    return BHSResult(
        score=score,
        tier=HonestyTier.FLOOR,
        evidence_present=evidence_present,
        optimism_flags=flags,
        drift_detected=False,
        notes=f"finding_dict scored: {len(flags)} flag(s)",
    )


def _score_phase_summary(phase_summary: Dict[str, Any]) -> BHSResult:
    score = 100.0
    flags: List[str] = []

    for key in _REQUIRED_PHASE_KEYS:
        if key not in phase_summary or phase_summary.get(key) in (None, ""):
            score -= 10.0
            flags.append(f"L4: phase_summary missing field {key!r}")

    # Empty totals are suspicious — orchestrator should have findings if a
    # phase ran. Empty isn't an automatic fail, but it gets a small penalty
    # because it usually correlates with L1 scaffold runs.
    total = phase_summary.get("total_findings")
    if isinstance(total, int) and total == 0:
        score -= 5.0
        flags.append("L1: phase ran with zero findings (scaffold-pass risk)")

    text_blob = _collect_text(phase_summary)
    text_lower = text_blob.lower()
    for marker in _LIE_MARKERS:
        if marker in text_lower:
            score -= 5.0
            flags.append(f"L1/L4: lie-marker keyword in phase summary: {marker!r}")

    evidence_present = bool(_EVIDENCE_RE.search(text_blob))
    if not evidence_present:
        score -= 15.0
        flags.append("L5: no evidence pointer in phase_summary")

    score = max(0.0, min(100.0, score))
    return BHSResult(
        score=score,
        tier=HonestyTier.FLOOR,
        evidence_present=evidence_present,
        optimism_flags=flags,
        drift_detected=False,
        notes=f"phase_summary scored: {len(flags)} flag(s)",
    )


def validate_pr_brutal_honesty(
    pr_number: Optional[int] = None,
    diff_path: Optional[str] = None,
    finding_dict: Optional[Dict[str, Any]] = None,
    phase_summary: Optional[Dict[str, Any]] = None,
) -> BHSResult:
    """Score the orchestrator's input dict for honesty signals.

    Exactly one of ``finding_dict`` or ``phase_summary`` should be passed; if
    neither is, we return a zero score with an L4 flag because the orchestrator
    asked us to validate nothing — which is itself a partial-as-complete call.

    ``pr_number`` and ``diff_path`` are accepted for API compatibility with
    future PR-body delegation but are not consumed here. PR-body parsing lives
    in ``scripts/validate_pr_brutal_honesty.py``.
    """
    if finding_dict is not None:
        return _score_finding(finding_dict)
    if phase_summary is not None:
        return _score_phase_summary(phase_summary)

    return BHSResult(
        score=0.0,
        tier=HonestyTier.FLOOR,
        evidence_present=False,
        optimism_flags=["L4: validate_pr_brutal_honesty called with no input dict"],
        drift_detected=False,
        notes="No finding_dict or phase_summary supplied.",
    )


# --- Smoke pipeline ---------------------------------------------------------


def _floor_smoke_checks() -> List[str]:
    """Return a list of failure messages; empty list = floor tier PASS."""
    failures: List[str] = []
    repo_root = _REPO_ROOT
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    # 1. The BHS validator itself imports clean (we are inside it; this is
    #    self-check via attribute presence).
    for required in ("validate_pr_brutal_honesty", "run_smoke_pipeline", "BHSResult"):
        if not hasattr(sys.modules[__name__], required):
            failures.append(
                f"bhs_validator missing public symbol {required!r} (L10 dependency phantom)"
            )

    # 2. aep_orchestrator must import without error — the BHS hook consumers
    #    live there and stale imports would silently break the gate.
    try:
        importlib.import_module("aep_orchestrator")
    except ImportError as exc:
        failures.append(f"aep_orchestrator import failed: {exc!r}")
    except Exception as exc:  # noqa: BLE001 - smoke tier needs to surface non-ImportError too
        failures.append(
            f"aep_orchestrator import raised non-ImportError: {exc!r}\n{traceback.format_exc()}"
        )

    # 3. The PR-body validator must be parseable as Python — if it isn't, the
    #    merge gate cannot run.
    pr_validator = repo_root / "scripts" / "validate_pr_brutal_honesty.py"
    if not pr_validator.exists():
        failures.append(f"missing required file: {pr_validator}")
    else:
        try:
            compile(pr_validator.read_text(encoding="utf-8"), str(pr_validator), "exec")
        except SyntaxError as exc:
            failures.append(f"validate_pr_brutal_honesty.py has SyntaxError: {exc!r}")

    # 4. The Tier C state surface must exist; check_block_flag.py reads it.
    next_session = repo_root / "docs" / "next-session.md"
    if not next_session.exists():
        failures.append(f"missing Tier C state file: {next_session}")

    return failures


def run_smoke_pipeline(tier: HonestyTier = HonestyTier.FLOOR) -> bool:
    """Run the BHS-side smoke at the requested tier.

    Floor tier: parseability/importability of the BHS gate path + presence of
    its state files. Ceiling tier: delegate to ``scripts/smoke_pipeline.py``
    via subprocess (it handles the production-module exercise).

    Returns ``True`` only when every check passes. Failures are printed to
    stdout so the caller (or CI tail) can see why.
    """
    if tier is HonestyTier.FLOOR:
        failures = _floor_smoke_checks()
        if failures:
            print(f"[BHS] floor-tier smoke FAIL ({len(failures)} check(s)):")
            for f in failures:
                print(f"  - {f}")
            return False
        print("[BHS] floor-tier smoke PASS")
        return True

    # Ceiling tier — delegate to the repo-level smoke script.
    import subprocess

    smoke_script = _REPO_ROOT / "scripts" / "smoke_pipeline.py"
    if not smoke_script.exists():
        print(f"[BHS] ceiling-tier smoke FAIL: missing {smoke_script}")
        return False
    try:
        result = subprocess.run(
            [sys.executable, str(smoke_script)],
            capture_output=True,
            text=True,
            timeout=180,
        )
    except subprocess.TimeoutExpired:
        print("[BHS] ceiling-tier smoke FAIL: smoke_pipeline.py timed out")
        return False
    except OSError as exc:
        print(f"[BHS] ceiling-tier smoke FAIL: cannot invoke smoke_pipeline.py: {exc!r}")
        return False

    # Echo so operators see what we saw.
    if result.stdout:
        print(result.stdout.rstrip())
    if result.stderr:
        print(result.stderr.rstrip(), file=sys.stderr)
    if result.returncode != 0:
        print(f"[BHS] ceiling-tier smoke FAIL: smoke_pipeline.py exit={result.returncode}")
        return False
    print("[BHS] ceiling-tier smoke PASS")
    return True


if __name__ == "__main__":  # pragma: no cover
    rich = {
        "id": "F-001",
        "severity": "HIGH",
        "impact": "Without this gate, PR #244 lands at avg_bhs_score 0.0",
        "recommended_fix": "Replace stub with real scoring in scripts/bhs_validator.py:43",
    }
    sparse = {"id": "F-002", "severity": "UNKNOWN"}
    print("rich  ->", validate_pr_brutal_honesty(finding_dict=rich))
    print("sparse->", validate_pr_brutal_honesty(finding_dict=sparse))
    ok = run_smoke_pipeline(HonestyTier.FLOOR)
    print(f"floor smoke: {'PASS' if ok else 'FAIL'}")
