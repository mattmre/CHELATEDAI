#!/usr/bin/env python3
"""scripts/validate_pr_brutal_honesty.py — Brutal Honesty Rulebook v3.2 §4 validator.

Parses a PR body and enforces the §4 template. Exits 0 only if every required
field is present, in the correct format, and the cross-field invariants hold.

What this validator IS:
  - A structural / arithmetic check on the BHS lines in a PR body.
  - The mechanism that makes the rulebook's "automatic L4 disclosure" and
    "BHS_OFFICIAL = 0" rules actually fire instead of relying on agent honesty.

What this validator is NOT:
  - A content judge. It cannot verify EVIDENCE: actually traces production code,
    or that the agent who wrote BHS_TIER_B is genuinely "fresh" (it can only
    enforce the agent IDs differ). Those checks remain operator/Tier B work.
  - A merge gate by itself. CI/operator wires this into the merge flow; the
    script just exits non-zero with a list of violations.

Inputs (one of):
  --body PATH      Read PR body from a file (use "-" for stdin).
  --pr NUMBER      Read PR body via `gh pr view NUMBER --json body -q .body`.
                   Requires `gh` on PATH and a GitHub auth context.

Output:
  - Plain-text report on stdout listing each required field with PASS/FAIL.
  - Exits 0 on full PASS, 1 on any structural violation, 2 on tooling error.

Cross-field rules enforced (all from §4 + §6.1 + §6.2 + §6.3):
  R1. All required lines present.
  R2. BHS_SELF_DRAFT and BHS_TIER_B parse as integers in [0, 100].
  R3. BHS_TIER_B_SEVERITY ∈ {none, cosmetic, important, critical}.
  R4. Severity caps applied: critical caps Tier B at 70; important at 90.
  R5. BHS_OFFICIAL == min(BHS_SELF_DRAFT, capped_BHS_TIER_B).
  R6. BHS_SELF_DRAFT_AGENT != BHS_TIER_B_AGENT (case-insensitive). Same agent
      forces BHS_OFFICIAL → 0 — script reports this and FAILS.
  R7. LOOP_ITERATIONS ∈ [1, 5]. If = 5 and BHS_SELF_DRAFT < 100, the body must
      explain withdraw-or-scope-reduce (we look for the substring).
  R8. OPERATOR_OVERRIDE format: empty OR includes reason + name + timestamp +
      out-of-band reference. We do not try to validate the URL — we only check
      that the four comma-separated fields are present when the line is non-
      empty. (At BLOCKED we additionally require a co-signer or out-of-band
      ref; we cannot tell whether the block flag was BLOCKED from the PR body
      alone, so we surface the requirement as a WARNING instead of a hard
      FAIL — `scripts/check_block_flag.py` is the cross-check.)
  R9. Score-gaming auto-disclosure: if BHS_SELF_DRAFT - BHS_TIER_B > 5, the
      body must contain an L4 score-gaming disclosure (substring "L4" + one of
      "score-gam" or "score gam").
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from dataclasses import dataclass, field
from typing import Optional

# ---------------------------------------------------------------------------
# Field schema
# ---------------------------------------------------------------------------

REQUIRED_FIELDS = [
    "EVIDENCE",
    "SMOKE",
    "BHS_SELF_DRAFT",
    "BHS_SELF_DRAFT_AGENT",
    "BHS_TIER_B",
    "BHS_TIER_B_AGENT",
    "BHS_TIER_B_SEVERITY",
    "BHS_OFFICIAL",
    "CARRY_FORWARD",
    "DEFERRED_SCOPE",
    "LOOP_ITERATIONS",
    "OPERATOR_OVERRIDE",
]

VALID_SEVERITIES = {"none", "cosmetic", "important", "critical"}

SEVERITY_CAPS = {
    "none": 100,
    "cosmetic": 100,
    "important": 90,
    "critical": 70,
}

# Match "FIELD: value" at line start. Value runs to end of line. We tolerate
# arbitrary whitespace around the colon and accept blockquote prefixes (">")
# in case the PR template was quoted. Trailing template hints in angle
# brackets (e.g. "<0–100, ...>") count as "unfilled" and are caught by the
# numeric/format checks below.
FIELD_RE = re.compile(
    r"^\s*>?\s*([A-Z][A-Z0-9_]+)\s*:\s*(.*?)\s*$",
    re.MULTILINE,
)


@dataclass
class Violation:
    field: str
    rule: str
    message: str
    severity: str = "FAIL"  # FAIL or WARN


@dataclass
class Report:
    fields: dict = field(default_factory=dict)
    violations: list = field(default_factory=list)

    def add(self, v: Violation) -> None:
        self.violations.append(v)

    @property
    def fails(self) -> list:
        return [v for v in self.violations if v.severity == "FAIL"]

    @property
    def warns(self) -> list:
        return [v for v in self.violations if v.severity == "WARN"]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------


def parse_fields(body: str) -> dict:
    """Extract FIELD: value pairs. Last occurrence wins (operator may amend)."""
    found: dict = {}
    for match in FIELD_RE.finditer(body):
        name, value = match.group(1), match.group(2).strip()
        if name in REQUIRED_FIELDS:
            found[name] = value
    return found


def _looks_unfilled(value: str) -> bool:
    """A value that still contains the template's `<...>` hint is not filled."""
    if not value:
        return True
    stripped = value.strip()
    if stripped.startswith("<") and stripped.endswith(">"):
        return True
    # also catch "<...>" embedded as the entire value with surrounding quotes
    if stripped.startswith('"<') and stripped.endswith('>"'):
        return True
    return False


def _parse_int(value: str) -> Optional[int]:
    # Accept "100", "100/100", "100 — justification...". We pull the first
    # integer token at the start of the value.
    m = re.match(r"\s*(-?\d+)", value)
    if not m:
        return None
    try:
        return int(m.group(1))
    except ValueError:
        return None


# ---------------------------------------------------------------------------
# Validation rules
# ---------------------------------------------------------------------------


def validate(body: str) -> Report:
    report = Report(fields=parse_fields(body))

    # R1 — every required field present and not the raw template hint.
    for name in REQUIRED_FIELDS:
        if name not in report.fields:
            report.add(Violation(name, "R1", f"missing required field: {name}:"))
        elif _looks_unfilled(report.fields[name]):
            # CARRY_FORWARD, DEFERRED_SCOPE, OPERATOR_OVERRIDE may be legitimately
            # empty — but the template hint must not be left in.
            report.add(
                Violation(
                    name,
                    "R1",
                    f"{name}: still contains the unfilled template hint "
                    f"({report.fields[name]!r}). Replace with an actual value, "
                    "or use 'none' / 'empty' if the field is genuinely empty.",
                )
            )

    # If R1 already failed for any of the score fields, downstream checks would
    # be noise; we still try to run them so the report is useful in one pass.

    # R2 — score fields parse as integers in [0, 100].
    self_draft = _parse_int(report.fields.get("BHS_SELF_DRAFT", ""))
    tier_b = _parse_int(report.fields.get("BHS_TIER_B", ""))
    official = _parse_int(report.fields.get("BHS_OFFICIAL", ""))

    for name, val in (
        ("BHS_SELF_DRAFT", self_draft),
        ("BHS_TIER_B", tier_b),
        ("BHS_OFFICIAL", official),
    ):
        if val is None:
            report.add(Violation(name, "R2", f"{name}: cannot parse integer score"))
        elif not (0 <= val <= 100):
            report.add(
                Violation(name, "R2", f"{name}: score {val} out of range [0, 100]")
            )

    # R3 — severity is one of the allowed values.
    severity_raw = report.fields.get("BHS_TIER_B_SEVERITY", "").strip().lower()
    severity = severity_raw.split()[0] if severity_raw else ""
    # Strip trailing punctuation/quotes.
    severity = severity.strip(' "\',.;')
    if severity and severity not in VALID_SEVERITIES:
        report.add(
            Violation(
                "BHS_TIER_B_SEVERITY",
                "R3",
                f"BHS_TIER_B_SEVERITY: {severity_raw!r} not in "
                f"{sorted(VALID_SEVERITIES)}",
            )
        )

    # R4 + R5 — severity cap on Tier B, then BHS_OFFICIAL = min(self, capped).
    capped_tier_b = tier_b
    if tier_b is not None and severity in SEVERITY_CAPS:
        cap = SEVERITY_CAPS[severity]
        if tier_b > cap:
            report.add(
                Violation(
                    "BHS_TIER_B",
                    "R4",
                    f"BHS_TIER_B: {tier_b} exceeds severity cap "
                    f"({severity}={cap}). Tier B must downgrade.",
                )
            )
            capped_tier_b = cap

    # R6 — agent independence.
    self_agent = report.fields.get("BHS_SELF_DRAFT_AGENT", "").strip().lower()
    tier_b_agent = report.fields.get("BHS_TIER_B_AGENT", "").strip().lower()
    independence_violated = False
    if self_agent and tier_b_agent:
        if self_agent == tier_b_agent:
            independence_violated = True
            report.add(
                Violation(
                    "BHS_TIER_B_AGENT",
                    "R6",
                    f"BHS_SELF_DRAFT_AGENT == BHS_TIER_B_AGENT ({self_agent!r}). "
                    "Same agent doing both is L4 score-gaming. Validator FAILS "
                    "the PR (expected `BHS_OFFICIAL = 0`); PR cannot merge "
                    "until a genuinely independent Tier B reviewer is run. "
                    "Validator does not silently rewrite the field.",
                )
            )

    # R5 — official == min(self, capped_tier_b), with R6 forcing 0.
    if self_draft is not None and capped_tier_b is not None and official is not None:
        expected_official = 0 if independence_violated else min(self_draft, capped_tier_b)
        if official != expected_official:
            report.add(
                Violation(
                    "BHS_OFFICIAL",
                    "R5",
                    f"BHS_OFFICIAL: declared {official}, but expected "
                    f"{expected_official} (= "
                    f"{'0 (independence violated)' if independence_violated else f'min({self_draft}, {capped_tier_b})'}).",
                )
            )

    # R7 — LOOP_ITERATIONS in [1, 5]; if 5 and self_draft < 100, must mention
    # 'withdraw' or 'scope-reduce' / 'reduce scope' in the body.
    loops = _parse_int(report.fields.get("LOOP_ITERATIONS", ""))
    if loops is None:
        report.add(
            Violation("LOOP_ITERATIONS", "R7", "LOOP_ITERATIONS: cannot parse integer")
        )
    elif not (1 <= loops <= 5):
        report.add(
            Violation(
                "LOOP_ITERATIONS",
                "R7",
                f"LOOP_ITERATIONS: {loops} out of allowed range [1, 5]",
            )
        )
    elif loops == 5 and self_draft is not None and self_draft < 100:
        # Look in the whole body for the explicit words.
        normalized = body.lower()
        if not any(
            keyword in normalized
            for keyword in ("withdrawn", "withdraw", "scope-reduce", "reduce scope", "reducing scope")
        ):
            report.add(
                Violation(
                    "LOOP_ITERATIONS",
                    "R7",
                    "LOOP_ITERATIONS=5 with BHS_SELF_DRAFT<100. The PR body must "
                    "explain withdraw-or-scope-reduce per §6.1 step 6 — neither "
                    "phrase found.",
                )
            )

    # R8 — OPERATOR_OVERRIDE structural format.
    override = report.fields.get("OPERATOR_OVERRIDE", "").strip()
    if override and override.lower() not in ("empty", "none", "n/a", ""):
        # Expect 4 comma-separated fields: reason, name, timestamp, out-of-band ref.
        parts = [p.strip() for p in override.split(",")]
        if len(parts) < 4:
            report.add(
                Violation(
                    "OPERATOR_OVERRIDE",
                    "R8",
                    f"OPERATOR_OVERRIDE: expected 4 comma-separated fields "
                    f"(reason, name, timestamp, out-of-band ref); got "
                    f"{len(parts)} ({override!r}). PR cannot merge until "
                    "format is corrected (validator does not silently rewrite).",
                )
            )
        # Without knowing the block flag we can only WARN about the BLOCKED
        # extra requirement.
        report.add(
            Violation(
                "OPERATOR_OVERRIDE",
                "R8",
                "OPERATOR_OVERRIDE present. If the next-session.md block flag was "
                "BLOCKED at merge time, the override line additionally requires a "
                "co-signer name OR an out-of-band reference (Slack URL, signed "
                "email, calendar timestamp). This validator cannot read the block "
                "flag — run scripts/check_block_flag.py and verify manually.",
                severity="WARN",
            )
        )

    # R9 — score-gaming auto-disclosure.
    if (
        self_draft is not None
        and tier_b is not None
        and (self_draft - tier_b) > 5
    ):
        body_lower = body.lower()
        has_l4 = "l4" in body_lower
        has_score_gaming = "score-gam" in body_lower or "score gam" in body_lower
        if not (has_l4 and has_score_gaming):
            report.add(
                Violation(
                    "BHS_SELF_DRAFT",
                    "R9",
                    f"BHS_SELF_DRAFT ({self_draft}) - BHS_TIER_B ({tier_b}) = "
                    f"{self_draft - tier_b} > 5. Body must contain an L4 score-"
                    "gaming disclosure per §6.2.",
                )
            )

    return report


# ---------------------------------------------------------------------------
# I/O
# ---------------------------------------------------------------------------


def read_body(args: argparse.Namespace) -> str:
    if args.pr is not None:
        try:
            result = subprocess.run(
                ["gh", "pr", "view", str(args.pr), "--json", "body", "-q", ".body"],
                capture_output=True,
                text=True,
                check=True,
            )
        except FileNotFoundError:
            print("ERROR: `gh` not found on PATH; cannot fetch PR body.", file=sys.stderr)
            sys.exit(2)
        except subprocess.CalledProcessError as exc:
            print(
                f"ERROR: `gh pr view {args.pr}` failed: {exc.stderr.strip()}",
                file=sys.stderr,
            )
            sys.exit(2)
        return result.stdout

    if args.body == "-":
        return sys.stdin.read()

    try:
        with open(args.body, "r", encoding="utf-8") as fh:
            return fh.read()
    except OSError as exc:
        print(f"ERROR: cannot read {args.body!r}: {exc}", file=sys.stderr)
        sys.exit(2)


def render_report(report: Report) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("Brutal Honesty Rulebook v3.2 — §4 PR-body validator")
    lines.append("=" * 70)
    lines.append("")

    lines.append("Required fields:")
    for name in REQUIRED_FIELDS:
        present = name in report.fields
        marker = "[OK]" if present else "[MISSING]"
        value = report.fields.get(name, "<not present>")
        # Truncate noisy values so the report stays readable.
        if len(value) > 80:
            value = value[:77] + "..."
        lines.append(f"  {marker} {name}: {value}")
    lines.append("")

    if report.fails:
        lines.append(f"FAILS ({len(report.fails)}):")
        for v in report.fails:
            lines.append(f"  [{v.rule}] {v.field}: {v.message}")
        lines.append("")

    if report.warns:
        lines.append(f"WARNINGS ({len(report.warns)}):")
        for v in report.warns:
            lines.append(f"  [{v.rule}] {v.field}: {v.message}")
        lines.append("")

    if not report.fails:
        lines.append("RESULT: PASS — PR body satisfies §4 structural requirements.")
        lines.append(
            "        (This validator cannot verify EVIDENCE traces production code "
            "or that Tier B was genuinely independent. Operator audit still required.)"
        )
    else:
        lines.append("RESULT: FAIL — see violations above. Per §6.1, this PR must not merge.")

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--body", help="Path to PR body file (use '-' for stdin).")
    src.add_argument("--pr", type=int, help="GitHub PR number (uses `gh pr view`).")
    args = parser.parse_args()

    body = read_body(args)
    report = validate(body)
    print(render_report(report))
    return 0 if not report.fails else 1


if __name__ == "__main__":
    sys.exit(main())
