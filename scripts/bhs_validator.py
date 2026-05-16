#!/usr/bin/env python3
"""BHS v3.3 Validator — orchestrator-side **structural** honesty scoring.

This module is the BHS hook called by ``aep_orchestrator.py`` during synthesis,
remediation, and closure. It scores AEP ``finding_dict``s and ``phase_summary``
dicts on a 0–100 **structural** honesty scale (NOT a semantic one — see Scope
section) and exposes a smoke pipeline runner that validates the orchestrator's
runtime surface at floor or ceiling tier.

## Scope (per CD-245-01 closure, 2026-05-16)

This validator is a **structural** check. It mechanically verifies that:
  - required fields are present and non-empty
  - severity is a committed AEP tier
  - prose fields are not trivially short, single-char-padded, or one-token-only
  - an evidence pointer (file:line, PR ref, artifact, commit) is somewhere
    in the text
  - lie-marker keywords ("stub", "TODO", "fake", ...) are absent

It does NOT semantically judge whether the prose is *true*, *useful*, or
*actually describes the finding*. A motivated operator can compose prose
that passes every mechanical signal while saying nothing meaningful
(``"foo bar baz qux at handler.py:42"`` — 4 unique tokens, evidence pointer,
no lie markers → scores 100). That gap is fundamental to a regex-based
rubric. The honest mitigation is documented in ``docs/bhs-rubric-scope.md``
and implemented by ``scripts/audit_findings.py``, which lets a human
periodically sample-grade findings and surface drift between the mechanical
score and human judgement.

If you are reading this module and considering adding a heuristic that
"detects bad prose" — read ``docs/bhs-rubric-scope.md`` first. Past Tier B
iterations converged on the conclusion that escalating heuristics either
re-creates the L13 framing lie at a higher threshold, or pulls in
non-deterministic dependencies (LLMs) that conflict with Session Rule #1.

## Distinct from scripts/validate_pr_brutal_honesty.py

That validator parses PR-body field schemas (EVIDENCE/SMOKE/BHS_* lines).
The orchestrator never hands us a PR body — it hands us small dicts
describing a finding or a phase outcome. Different schema, different
scoring logic. Both validators share the rulebook's lie taxonomy (L1–L13)
by name but operate on disjoint inputs.

## Structural scoring signals (additive penalties from 100)

  - Missing ``id`` / ``severity`` / ``recommended_fix`` / ``impact`` fields
    (each absence is L4 partial-as-complete on the orchestrator's side).
  - Whitespace-only string in a required field counts as empty.
  - Prose field (``recommended_fix``, ``impact``) below ``_MIN_CONTENT_CHARS``
    trimmed chars → -15 (structural padding).
  - Prose field with content-quality failure (entropy < 3.0 bits/char OR
    unique-token count < 3 OR any single token >= 60% of the field) → -15
    each. Closes the literal CD-245-01 ``"xxxxxxxxxxxx"`` gap.
  - ``severity`` outside ``{CRITICAL, HIGH, MEDIUM, LOW}`` → -10.
  - No evidence pointer (regex for ``file:line``, ``PR #``, ``artifact``,
    commit hash) → -20. L5 test-as-truth precursor.
  - L1–L13 marker words in any text field (``stub``, ``TODO``,
    ``NotImplementedError``, ``placeholder``, ``mock``, ``fake``) → -5 each.
"""

from __future__ import annotations

import importlib
import math
import re
import sys
import traceback
from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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

# Subset of required keys that must contain committed prose (not just be
# present). Trivially short content (".", "x", "TBD") is structural padding.
# ``id`` is excluded (short identifier by design); ``severity`` is excluded
# (validated against the known-tier enum separately).
_PROSE_FINDING_KEYS = ("recommended_fix", "impact")

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

# Evidence regex: requires substantive evidence (not just a bare "a.py" token).
# Accepts any of:
#   - file path with at least one separator AND extension (e.g. scripts/x.py)
#   - bare filename WITH a line-number suffix (e.g. foo.py:42) — line numbers
#     are themselves a commitment to a specific location
#   - PR/issue ref (PR #123, issue #45)
#   - artifact path (artifact: or artifacts/...)
#   - commit hash (sha or "commit <hash>") with >=7 hex chars
# A loose ``a.py`` token in prose is NOT enough — operators must point at
# something actually locatable.
_EVIDENCE_RE = re.compile(
    r"""
    (
        [\w.-]+[/\\][\w./\\-]+\.[a-zA-Z0-9]{1,5}(:\d+)? |  # path with separator + ext
        \b[\w-]+\.[a-zA-Z0-9]{1,5}:\d+                  |  # bare file:line
        \b(?:PR|issue)\s*\#\s*\d+                       |  # PR #123 / issue #45
        \bartifact[s]?[/:]                              |  # artifact: or artifacts/
        \bcommit\s+[0-9a-f]{7,40}                       |  # commit <hash>
        \b(SHA|sha)\s*[:=]\s*[0-9a-f]{7,40}
    )
    """,
    re.VERBOSE,
)

# AEP severity enum (per CLAUDE.md). Anything outside this set is not a
# committed tier and should be penalised the same as UNKNOWN.
_KNOWN_SEVERITIES = frozenset({"CRITICAL", "HIGH", "MEDIUM", "LOW"})

# Trivially short prose ("." / "x" / "TBD") is structural padding, not
# committed content. Anything below this many trimmed characters in a
# required text field gets the same penalty as a missing field. Tuned so
# that a one-word answer like "fixed" (5 chars) still flags but a short
# real sentence like "Auth bypass at auth.py:42" (24 chars) is fine.
_MIN_CONTENT_CHARS = 12

# Content-quality thresholds (CD-245-01 closure, 2026-05-16). Each prose
# field is penalised if ANY of the three signals trips:
#
#   - Shannon entropy over characters below this many bits per character.
#     Padded constant-char strings ("xxxxxxxxxxxx") have entropy ~0.0;
#     genuine English prose typically sits at 3.5-4.5 bits/char. The
#     threshold is intentionally low (3.0) to avoid false-positives on
#     short technical strings like "fix auth.py:42" (~3.4 bits/char).
#   - Unique-token count below this minimum after splitting on \W+ and
#     lower-casing. "xxxxxxxxxxxx" tokenises to one unique token; "fix
#     fix fix fix bug" tokenises to two. A real sentence virtually always
#     has >= 3 unique tokens.
#   - Any single token's share of total tokens above this ratio. "fix
#     fix fix bug" has one token at 75% share, well above 60%.
#
# This rubric is mechanical, not semantic. A motivated operator who knows
# the rubric can still write diverse-but-meaningless prose
# ("foo bar baz qux at handler.py:42"). See module docstring "Scope" and
# scripts/audit_findings.py for the documented mitigation.
_MIN_CHAR_ENTROPY_BITS = 3.0
_MIN_UNIQUE_TOKENS = 3
_MAX_DOMINANT_TOKEN_RATIO = 0.60

_TOKEN_SPLIT_RE = re.compile(r"\W+")


def _content_quality_penalty(field_name: str, value: str) -> Tuple[float, List[str]]:
    """Return (penalty, flags) for content-quality signals on a prose field.

    Penalty is additive (>= 0). Flags list is empty when no signal trips.

    Skipped for trivially-short values — those are already handled by the
    ``_MIN_CONTENT_CHARS`` check upstream, and computing entropy on 2 chars
    is noise. We only score quality when the field has enough material to
    judge.

    All three signals are deterministic and stdlib-only. No network, no
    external dependencies. Per Session Rule #1 there is no fallback path
    because there is nothing to fall back from.
    """
    stripped = value.strip()
    if len(stripped) < _MIN_CONTENT_CHARS:
        return 0.0, []

    penalty = 0.0
    flags: List[str] = []

    # Signal 1: character entropy (Shannon, base 2).
    char_counts = Counter(stripped)
    n_chars = len(stripped)
    entropy = -sum(
        (c / n_chars) * math.log2(c / n_chars) for c in char_counts.values()
    )
    if entropy < _MIN_CHAR_ENTROPY_BITS:
        penalty += 15.0
        flags.append(
            f"L13: {field_name!r} character entropy {entropy:.2f} bits/char below "
            f"{_MIN_CHAR_ENTROPY_BITS} (looks like padding, not committed prose)"
        )
        # If entropy is dead — single repeated char — also catches dominant
        # token, but we don't double-penalise; return early.
        return penalty, flags

    # Signal 2: unique-token count after lowercasing + \W+ split.
    tokens = [t for t in _TOKEN_SPLIT_RE.split(stripped.lower()) if t]
    unique_tokens = set(tokens)
    if len(unique_tokens) < _MIN_UNIQUE_TOKENS:
        penalty += 15.0
        flags.append(
            f"L13: {field_name!r} has {len(unique_tokens)} unique token(s); "
            f"min {_MIN_UNIQUE_TOKENS} (repeated-word padding)"
        )
        return penalty, flags

    # Signal 3: dominant token ratio. (Only meaningful when we have several
    # tokens; with 3 tokens "a b a" has 67% which is fair to penalise.)
    if tokens:
        most_common_token, most_common_count = Counter(tokens).most_common(1)[0]
        ratio = most_common_count / len(tokens)
        if ratio > _MAX_DOMINANT_TOKEN_RATIO:
            penalty += 15.0
            flags.append(
                f"L13: {field_name!r} token {most_common_token!r} is "
                f"{ratio:.0%} of total tokens (>{_MAX_DOMINANT_TOKEN_RATIO:.0%}; "
                f"dominant-token padding)"
            )

    return penalty, flags


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


def _score_finding_structure(finding_dict: Dict[str, Any]) -> BHSResult:
    """Score a finding_dict on structural honesty signals only.

    Renamed from ``_score_finding`` per CD-245-01 (2026-05-16) to make the
    structural-not-semantic boundary visible at the symbol name. A backwards-
    compatible alias is provided below for any external code that imported
    the old name; the public API (``validate_pr_brutal_honesty``) is
    unchanged.
    """
    score = 100.0
    flags: List[str] = []

    # Missing required keys → L4. Whitespace-only strings are treated as empty.
    for key in _REQUIRED_FINDING_KEYS:
        raw = finding_dict.get(key)
        is_empty = (
            key not in finding_dict
            or raw is None
            or (isinstance(raw, str) and raw.strip() == "")
        )
        if is_empty:
            score -= 15.0
            flags.append(f"L4: missing/empty finding field {key!r}")

    # Trivially short PROSE fields (".", "x", "TBD") are structural padding.
    # Skipped for ``id`` (short by design) and ``severity`` (enum value, length
    # is fine if it's a known tier — already checked below).
    # Then content-quality (entropy / unique tokens / dominant token) per
    # CD-245-01 — catches "xxxxxxxxxxxx" and "fix fix fix fix bug" padding
    # that the bare length check missed.
    for key in _PROSE_FINDING_KEYS:
        raw = finding_dict.get(key)
        if isinstance(raw, str) and 0 < len(raw.strip()) < _MIN_CONTENT_CHARS:
            score -= 15.0
            flags.append(
                f"L4: trivially short finding field {key!r} ({len(raw.strip())} chars; "
                f"min {_MIN_CONTENT_CHARS})"
            )
        elif isinstance(raw, str):
            penalty, quality_flags = _content_quality_penalty(key, raw)
            score -= penalty
            flags.extend(quality_flags)

    # Severity must be one of the committed AEP tiers; any other string
    # (empty, UNKNOWN, NONE, free-form like "WHATEVER") is not a committed
    # tier and gets the same penalty.
    severity = str(finding_dict.get("severity", "")).strip().upper()
    if severity not in _KNOWN_SEVERITIES:
        score -= 10.0
        flags.append(
            f"L4: severity not a committed AEP tier "
            f"(got {severity!r}; expected one of {sorted(_KNOWN_SEVERITIES)})"
        )

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


# Backwards-compatible alias for any external caller that imported the old
# name. Public API (validate_pr_brutal_honesty) is unchanged.
_score_finding = _score_finding_structure


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
        return _score_finding_structure(finding_dict)
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
