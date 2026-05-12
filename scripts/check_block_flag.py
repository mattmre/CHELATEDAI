#!/usr/bin/env python3
"""scripts/check_block_flag.py — Brutal Honesty Rulebook v3.3 §6.3 block-flag gate.

Reads `docs/next-session.md` (or a path passed via --file) and exits non-zero
when the Block flag is `BLOCKED`. This is the mechanism that turns the v3.1/v3.2
"BLOCKED forbids new feature work" rule from a doc-aspiration into an actual
gate CI/operators can wire into the merge flow.

Usage:
  python scripts/check_block_flag.py
  python scripts/check_block_flag.py --file path/to/next-session.md
  python scripts/check_block_flag.py --allow-debt-prs

What it checks:
  1. Locates the "Block flag" section header (case-insensitive, under any
     heading depth).
  2. Reads the immediately following paragraph for the canonical state token:
       - "CLEAR"  → exit 0
       - "BLOCKED" → exit 1 (unless --allow-debt-prs and the rest of the file
         shows the only Carried Debt items remaining are the ones this PR is
         trying to drain — that judgment is the operator's, not the script's;
         the flag still says BLOCKED until they edit the file)
  3. Counts Carried Debt rows in the table. Reports the count to make stale
     state obvious (e.g. flag says CLEAR but there are 12 debt items — the
     flag is lying).

Exit codes:
  0  Block flag is CLEAR (or --allow-debt-prs was passed).
  1  Block flag is BLOCKED.
  2  Could not parse the file (missing section, ambiguous flag value,
     unreadable file).

What this script is NOT:
  - It does not flip the flag automatically. The flag is set by the session-
    wrap step per §6.3 ("at the END of that cycle, any item still on the list
    flips the block flag to BLOCKED"). The script enforces it; humans set it.
  - It does not validate the Carried Debt schema beyond counting rows. The
    full schema check is the operator's review job.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Optional, Tuple

DEFAULT_FILE = Path("docs/next-session.md")

# Match a markdown heading whose text contains "block flag" (case-insensitive).
HEADING_RE = re.compile(r"^(#{1,6})\s*(.*?)\s*$", re.MULTILINE)
BLOCK_FLAG_HEADING_RE = re.compile(r"\bblock\s+flag\b", re.IGNORECASE)

# Canonical state tokens — searched verbatim (case-sensitive) so a typo'd
# "blocked" in lowercase prose doesn't trip the gate.
TOKEN_CLEAR = "CLEAR"
TOKEN_BLOCKED = "BLOCKED"

# Match the Carried Debt table heading.
CARRIED_DEBT_HEADING_RE = re.compile(r"\bcarried\s+debt\b", re.IGNORECASE)

# A markdown table row starts with `|` — we count rows after the header
# separator (`|---|---|...`).
TABLE_ROW_RE = re.compile(r"^\s*\|.+\|\s*$")
TABLE_SEPARATOR_RE = re.compile(r"^\s*\|[\s\-:|]+\|\s*$")


def _find_section(content: str, heading_re: re.Pattern) -> Optional[Tuple[int, int]]:
    """Return (section_start_offset, next_heading_offset_or_eof) for the first
    heading whose text matches `heading_re`."""
    found_start = None
    found_level = 0
    last_pos = len(content)
    for match in HEADING_RE.finditer(content):
        level = len(match.group(1))
        text = match.group(2)
        if found_start is None:
            if heading_re.search(text):
                found_start = match.end()
                found_level = level
            continue
        # Already past the target heading; the next heading at <= same level
        # ends the section.
        if level <= found_level:
            return (found_start, match.start())
    if found_start is None:
        return None
    return (found_start, last_pos)


def parse_block_flag(content: str) -> Tuple[str, str]:
    """Return (state, raw_excerpt). state ∈ {"CLEAR", "BLOCKED", "UNKNOWN"}.

    raw_excerpt is the section text used for the decision (for error messages).
    """
    section = _find_section(content, BLOCK_FLAG_HEADING_RE)
    if section is None:
        return ("UNKNOWN", "<no heading containing 'Block flag' found>")
    start, end = section
    excerpt = content[start:end].strip()

    # Look for the canonical tokens in the excerpt. We require uppercase to
    # avoid matching prose like "items will be blocked".
    has_clear = TOKEN_CLEAR in excerpt
    has_blocked = TOKEN_BLOCKED in excerpt

    if has_blocked and not has_clear:
        return ("BLOCKED", excerpt)
    if has_clear and not has_blocked:
        return ("CLEAR", excerpt)
    if has_clear and has_blocked:
        # Both present — common when the section explains the mechanics. Look
        # for the **Current**: pattern as the tiebreaker.
        m = re.search(
            r"(?:\*\*)?\s*current\s*(?:\*\*)?\s*:\s*[`\"']?(CLEAR|BLOCKED)\b",
            excerpt,
            re.IGNORECASE,
        )
        if m:
            return (m.group(1).upper(), excerpt)
        return ("UNKNOWN", excerpt)
    return ("UNKNOWN", excerpt)


def _split_row_cells(line: str) -> list[str]:
    """Split a markdown table row into its cell strings (stripped)."""
    return [c.strip() for c in line.strip().strip("|").split("|")]


def count_carried_debt_rows(content: str) -> int:
    """Count OPEN data rows in the Carried Debt table.

    If the table has a `Status` column, rows whose Status cell starts with
    `CLOSED` (case-insensitive, ignoring leading markdown bold markers) are
    treated as historical record of completed work and NOT counted as active
    debt. This matches §6.3 ("any item still on the list flips the block flag")
    — items marked CLOSED are no longer "still on the list" in the sense that
    matters for the block-flag math.

    If the table has no `Status` column, all non-placeholder data rows are
    counted (preserves the v3.2 baseline behavior for tables that pre-date the
    Status column).

    Returns 0 if no Carried Debt table is found.
    """
    section = _find_section(content, CARRIED_DEBT_HEADING_RE)
    if section is None:
        return 0
    start, end = section
    excerpt = content[start:end]

    rows = 0
    seen_separator = False
    header_cells: list[str] = []
    status_idx: int | None = None
    last_header_candidate: list[str] | None = None

    for line in excerpt.splitlines():
        if TABLE_SEPARATOR_RE.match(line):
            seen_separator = True
            # The line immediately before the separator is the header row.
            if last_header_candidate is not None:
                header_cells = last_header_candidate
                lowered = [h.lower() for h in header_cells]
                if "status" in lowered:
                    status_idx = lowered.index("status")
            continue
        if not seen_separator:
            # Track the most recent table-shaped line as a header candidate.
            if TABLE_ROW_RE.match(line):
                last_header_candidate = _split_row_cells(line)
            continue
        if not TABLE_ROW_RE.match(line):
            # Table ended.
            if rows > 0:
                break
            continue
        # Skip placeholder rows like "_none yet_".
        if "_none yet_" in line.lower() or "_no items_" in line.lower():
            continue
        cells = _split_row_cells(line)
        # Skip rows that are entirely italicized em-dashes (placeholder pattern).
        if all(c in ("_—_", "—", "-", "") for c in cells):
            continue
        # Status-aware filter: drop CLOSED rows when the column exists.
        if status_idx is not None and status_idx < len(cells):
            status = cells[status_idx].strip("*").strip().lower()
            if status.startswith("closed"):
                continue
        rows += 1
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "--file",
        default=str(DEFAULT_FILE),
        help=f"Path to the session-handoff file (default: {DEFAULT_FILE}).",
    )
    parser.add_argument(
        "--allow-debt-prs",
        action="store_true",
        help=(
            "Force exit 0 even if BLOCKED. Use ONLY for PRs whose entire "
            "purpose is draining Carried Debt items. The block-flag warning "
            "is still printed."
        ),
    )
    args = parser.parse_args()

    path = Path(args.file)
    if not path.exists():
        print(f"ERROR: file not found: {path}", file=sys.stderr)
        return 2
    try:
        content = path.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"ERROR: cannot read {path}: {exc}", file=sys.stderr)
        return 2

    state, excerpt = parse_block_flag(content)
    debt_count = count_carried_debt_rows(content)

    print("=" * 70)
    print("Brutal Honesty Rulebook v3.3 — §6.3 block-flag gate")
    print(f"File: {path}")
    print("=" * 70)
    print(f"Block flag state: {state}")
    print(f"Carried Debt row count: {debt_count}")

    # Stale-state detection (advisory only — doesn't change exit code by itself).
    # The script does not know the cycle-age of each row, so it can only flag
    # the *combination* (CLEAR + open rows > 0) as worth a human glance — it
    # cannot assert the flag is wrong. First-cycle items legitimately keep the
    # flag CLEAR per §6.3.
    if state == "CLEAR" and debt_count > 0:
        print(
            f"WARNING: flag is CLEAR but {debt_count} OPEN Carried Debt row(s) "
            "remain. Per §6.3, first-cycle items keep the flag CLEAR; items "
            "surviving a full cycle (TTL=0) flip the flag to BLOCKED. Verify "
            "each open row's cycle age in the table — if any have already "
            "survived a cycle, the session-wrap step should have flipped the "
            "flag. The script cannot determine cycle age automatically."
        )
    if state == "BLOCKED" and debt_count == 0:
        print(
            "WARNING: flag is BLOCKED but Carried Debt is empty. Either the flag "
            "is stale or the table is missing rows."
        )

    if state == "UNKNOWN":
        print(
            "ERROR: could not determine block flag state. Excerpt:\n---\n"
            + (excerpt[:500] if excerpt else "<empty>")
            + "\n---",
            file=sys.stderr,
        )
        return 2

    if state == "CLEAR":
        print("RESULT: PASS — block flag CLEAR. New feature work is allowed.")
        return 0

    # state == "BLOCKED"
    if args.allow_debt_prs:
        print(
            "RESULT: OVERRIDE — block flag BLOCKED, but --allow-debt-prs was "
            "passed. Exiting 0 so the debt-draining PR can proceed. The flag "
            "remains BLOCKED until session-wrap clears it."
        )
        return 0

    print(
        "RESULT: FAIL — block flag BLOCKED. Per §6.3, no new feature work may "
        "merge until the Carried Debt table is empty. If this PR's entire "
        "purpose is draining a Carried Debt item, re-run with --allow-debt-prs."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
