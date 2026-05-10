"""Unit tests for scripts/check_block_flag.py.

Covers the section-finder, the canonical state token detection (CLEAR /
BLOCKED / UNKNOWN), the Carried Debt row counter, the stale-state warning,
and the CLI exit codes including --allow-debt-prs.

Run with: python -m pytest tests/test_check_block_flag.py -v
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from scripts.check_block_flag import (
    count_carried_debt_rows,
    parse_block_flag,
)

SCRIPT = Path(__file__).resolve().parent.parent / "scripts" / "check_block_flag.py"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


CLEAR_BODY = textwrap.dedent(
    """\
    # Next Session

    ## Block flag

    **Current**: `CLEAR` — no debt items have expired.

    ## Carried Debt

    | ID | Item | Source | TTL | Blocking |
    |----|------|--------|-----|----------|
    | _none yet_ | _—_ | _—_ | _—_ | _—_ |
    """
)

BLOCKED_BODY = textwrap.dedent(
    """\
    # Next Session

    ## Block flag

    **Current**: `BLOCKED` — CD-001 expired without resolution.

    ## Carried Debt

    | ID | Item | Source | TTL | Blocking |
    |----|------|--------|-----|----------|
    | CD-001 | Add validator | PR #X | expired | YES |
    | CD-002 | Add smoke.sh  | PR #Y | expired | YES |
    """
)

CLEAR_WITH_DEBT_BODY = textwrap.dedent(
    """\
    # Next Session

    ## Block flag

    **Current**: `CLEAR` — first cycle for the items below.

    ## Carried Debt

    | ID | Item | Source | TTL | Blocking |
    |----|------|--------|-----|----------|
    | CD-001 | Real debt item | PR #X | 1 cycle | YES |
    """
)

BOTH_TOKENS_NO_CURRENT_BODY = textwrap.dedent(
    """\
    # Next Session

    ## Block flag

    The flag can be either CLEAR or BLOCKED depending on cycle state.
    """
)


def _write(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "next-session.md"
    p.write_text(text, encoding="utf-8")
    return p


def _run(path: Path, *extra: str):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--file", str(path), *extra],
        capture_output=True,
        text=True,
    )


# ---------------------------------------------------------------------------
# parse_block_flag()
# ---------------------------------------------------------------------------


class TestParseBlockFlag:
    def test_clear(self) -> None:
        state, _ = parse_block_flag(CLEAR_BODY)
        assert state == "CLEAR"

    def test_blocked(self) -> None:
        state, _ = parse_block_flag(BLOCKED_BODY)
        assert state == "BLOCKED"

    def test_both_tokens_without_current_returns_unknown(self) -> None:
        state, excerpt = parse_block_flag(BOTH_TOKENS_NO_CURRENT_BODY)
        assert state == "UNKNOWN"
        assert "CLEAR" in excerpt and "BLOCKED" in excerpt

    def test_no_block_flag_section_returns_unknown(self) -> None:
        state, _ = parse_block_flag("# Just a doc with no flag section.\n")
        assert state == "UNKNOWN"

    def test_lowercase_blocked_word_does_not_trigger(self) -> None:
        # Prose use of "blocked" lowercase should not flip the state.
        body = textwrap.dedent(
            """\
            ## Block flag

            **Current**: `CLEAR` — no items are blocked at this time.
            """
        )
        state, _ = parse_block_flag(body)
        assert state == "CLEAR"

    def test_current_pattern_is_tiebreaker(self) -> None:
        body = textwrap.dedent(
            """\
            ## Block flag

            Either CLEAR or BLOCKED applies during a cycle.

            **Current**: `BLOCKED` — debt expired.
            """
        )
        state, _ = parse_block_flag(body)
        assert state == "BLOCKED"


# ---------------------------------------------------------------------------
# count_carried_debt_rows()
# ---------------------------------------------------------------------------


class TestCountCarriedDebt:
    def test_placeholder_row_counts_as_zero(self) -> None:
        assert count_carried_debt_rows(CLEAR_BODY) == 0

    def test_two_real_rows(self) -> None:
        assert count_carried_debt_rows(BLOCKED_BODY) == 2

    def test_one_real_row(self) -> None:
        assert count_carried_debt_rows(CLEAR_WITH_DEBT_BODY) == 1

    def test_no_carried_debt_section(self) -> None:
        assert count_carried_debt_rows("# Just text\n") == 0

    def test_status_column_filters_closed_rows(self) -> None:
        # Mixed table: 8 CLOSED + 3 OPEN — should report 3 OPEN, not 11 total.
        # Mirrors the real docs/next-session.md schema as of v3.2.
        body = textwrap.dedent(
            """\
            ## Carried Debt

            | ID | Item | Source | TTL | Blocking | Status |
            |---|---|---|---|---|---|
            | CD-001 | A | x | — | — | **CLOSED** by PR #1 |
            | CD-002 | B | x | 1 cycle | NO | OPEN — first cycle |
            | CD-003 | C | x | — | — | **CLOSED** by PR #2 |
            | CD-004 | D | x | — | — | **CLOSED** by PR #3 |
            | CD-005 | E | x | — | — | CLOSED by PR #4 |
            | CD-006 | F | x | — | — | **closed** by PR #5 |
            | CD-007 | G | x | 1 cycle | NO | OPEN |
            | CD-008 | H | x | — | — | **CLOSED** by PR #6 |
            | CD-009 | I | x | — | — | **CLOSED** by PR #7 |
            | CD-010 | J | x | — | — | **CLOSED** by PR #8 |
            | CD-011 | K | x | 1 cycle | NO | OPEN |
            """
        )
        assert count_carried_debt_rows(body) == 3

    def test_status_column_absent_falls_back_to_count_all(self) -> None:
        # Older table format without a Status column: count every real row.
        body = textwrap.dedent(
            """\
            ## Carried Debt

            | ID | Item | Source | TTL | Blocking |
            |----|------|--------|-----|----------|
            | CD-001 | One   | PR | expired | YES |
            | CD-002 | Two   | PR | expired | YES |
            | CD-003 | Three | PR | expired | YES |
            """
        )
        assert count_carried_debt_rows(body) == 3


# ---------------------------------------------------------------------------
# CLI exit codes
# ---------------------------------------------------------------------------


class TestCliExitCodes:
    def test_clear_exits_zero(self, tmp_path: Path) -> None:
        result = _run(_write(tmp_path, CLEAR_BODY))
        assert result.returncode == 0
        assert "PASS" in result.stdout

    def test_blocked_exits_one(self, tmp_path: Path) -> None:
        result = _run(_write(tmp_path, BLOCKED_BODY))
        assert result.returncode == 1
        assert "FAIL" in result.stdout

    def test_blocked_with_allow_debt_exits_zero(self, tmp_path: Path) -> None:
        result = _run(_write(tmp_path, BLOCKED_BODY), "--allow-debt-prs")
        assert result.returncode == 0
        assert "OVERRIDE" in result.stdout

    def test_unknown_state_exits_two(self, tmp_path: Path) -> None:
        result = _run(_write(tmp_path, BOTH_TOKENS_NO_CURRENT_BODY))
        assert result.returncode == 2

    def test_missing_file_exits_two(self, tmp_path: Path) -> None:
        missing = tmp_path / "does_not_exist.md"
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--file", str(missing)],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 2

    def test_clear_with_debt_warns_about_stale_state(self, tmp_path: Path) -> None:
        result = _run(_write(tmp_path, CLEAR_WITH_DEBT_BODY))
        assert result.returncode == 0  # CLEAR still passes
        assert "WARNING" in result.stdout
