"""Unit tests for scripts/check_block_flag.py.

Covers:
- section parsing and canonical token detection (CLEAR / BLOCKED / UNKNOWN)
- Carried Debt table counting and status filtering
- CLI exit codes and warning behavior

Run with: python -m unittest tests.test_check_block_flag -v
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path
import textwrap
import unittest

from scripts.check_block_flag import (
    count_blocking_open_debt_rows,
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


def _run(path: Path, *extra: str):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--file", str(path), *extra],
        capture_output=True,
        text=True,
    )


def _run_with_tmp(content: str, *extra: str):
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "next-session.md"
        path.write_text(content, encoding="utf-8")
        return _run(path, *extra)


def _missing_file_path() -> Path:
    with tempfile.TemporaryDirectory() as td:
        return Path(td) / "does_not_exist.md"


# ---------------------------------------------------------------------------
# parse_block_flag()
# ---------------------------------------------------------------------------


class TestParseBlockFlag(unittest.TestCase):
    def test_clear(self) -> None:
        state, _ = parse_block_flag(CLEAR_BODY)
        self.assertEqual(state, "CLEAR")

    def test_blocked(self) -> None:
        state, _ = parse_block_flag(BLOCKED_BODY)
        self.assertEqual(state, "BLOCKED")

    def test_both_tokens_without_current_returns_unknown(self) -> None:
        state, excerpt = parse_block_flag(BOTH_TOKENS_NO_CURRENT_BODY)
        self.assertEqual(state, "UNKNOWN")
        self.assertIn("CLEAR", excerpt)
        self.assertIn("BLOCKED", excerpt)

    def test_no_block_flag_section_returns_unknown(self) -> None:
        state, _ = parse_block_flag("# Just a doc with no flag section.\n")
        self.assertEqual(state, "UNKNOWN")

    def test_lowercase_blocked_word_does_not_trigger(self) -> None:
        body = textwrap.dedent(
            """\
            ## Block flag

            **Current**: `CLEAR` — no items are blocked at this time.
            """
        )
        state, _ = parse_block_flag(body)
        self.assertEqual(state, "CLEAR")

    def test_current_pattern_is_tiebreaker(self) -> None:
        body = textwrap.dedent(
            """\
            ## Block flag

            Either CLEAR or BLOCKED applies during a cycle.

            **Current**: `BLOCKED` — debt expired.
            """
        )
        state, _ = parse_block_flag(body)
        self.assertEqual(state, "BLOCKED")


# ---------------------------------------------------------------------------
# count_carried_debt_rows()
# ---------------------------------------------------------------------------


class TestCountCarriedDebt(unittest.TestCase):
    def test_placeholder_row_counts_as_zero(self) -> None:
        self.assertEqual(count_carried_debt_rows(CLEAR_BODY), 0)

    def test_two_real_rows(self) -> None:
        self.assertEqual(count_carried_debt_rows(BLOCKED_BODY), 2)

    def test_one_real_row(self) -> None:
        self.assertEqual(count_carried_debt_rows(CLEAR_WITH_DEBT_BODY), 1)

    def test_no_carried_debt_section(self) -> None:
        self.assertEqual(count_carried_debt_rows("# Just text\n"), 0)

    def test_status_column_filters_closed_rows(self) -> None:
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
        self.assertEqual(count_carried_debt_rows(body), 3)

    def test_blocking_open_rows_count(self) -> None:
        body = textwrap.dedent(
            """\
            ## Carried Debt

            | ID | Item | Source | TTL | Blocking | Status |
            |----|------|--------|-----|----------|--------|
            | CD-A | a | x | 1 cycle | NO — reason | OPEN |
            | CD-B | b | x | 1 cycle | YES | OPEN |
            | CD-C | c | x | 1 cycle | YES — load-bearing | OPEN |
            | CD-D | d | x | 1 cycle | NO | **CLOSED** by PR #1 |
            """
        )
        self.assertEqual(count_carried_debt_rows(body), 3)
        self.assertEqual(count_blocking_open_debt_rows(body), 2)

    def test_blocking_column_missing_counts_open_rows_as_blocking(self) -> None:
        body = textwrap.dedent(
            """\
            ## Carried Debt

            | ID | Item | Source | TTL | Status |
            |---|---|---|---|---|
            | CD-A | a | x | 1 cycle | OPEN |
            | CD-B | b | x | 1 cycle | **OPEN** |
            | CD-C | c | x | 1 cycle | **CLOSED** by PR #1 |
            """
        )
        self.assertEqual(count_carried_debt_rows(body), 2)
        self.assertEqual(count_blocking_open_debt_rows(body), 2)

    def test_blank_line_inside_table_does_not_truncate(self) -> None:
        body = textwrap.dedent(
            """\
            ## Carried Debt

            | ID | Item | Source | TTL | Blocking | Status |
            |----|------|--------|-----|----------|--------|
            | CD-A | first | x | 1 cycle | NO | OPEN |

            | CD-B | second | x | 1 cycle | YES | OPEN |
            """
        )
        self.assertEqual(count_carried_debt_rows(body), 2)

    def test_status_column_absent_falls_back_to_count_all(self) -> None:
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
        self.assertEqual(count_carried_debt_rows(body), 3)


# ---------------------------------------------------------------------------
# CLI exit codes
# ---------------------------------------------------------------------------


class TestCliExitCodes(unittest.TestCase):
    def test_clear_exits_zero(self) -> None:
        result = _run_with_tmp(CLEAR_BODY)
        self.assertEqual(result.returncode, 0)
        self.assertIn("PASS", result.stdout)

    def test_blocked_exits_one(self) -> None:
        result = _run_with_tmp(BLOCKED_BODY)
        self.assertEqual(result.returncode, 1)
        self.assertIn("FAIL", result.stdout)

    def test_blocked_with_allow_debt_exits_zero(self) -> None:
        result = _run_with_tmp(BLOCKED_BODY, "--allow-debt-prs")
        self.assertEqual(result.returncode, 0)
        self.assertIn("OVERRIDE", result.stdout)

    def test_unknown_state_exits_two(self) -> None:
        result = _run_with_tmp(BOTH_TOKENS_NO_CURRENT_BODY)
        self.assertEqual(result.returncode, 2)

    def test_missing_file_exits_two(self) -> None:
        missing = _missing_file_path()
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--file", str(missing)],
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 2)

    def test_clear_with_debt_warns_about_stale_state(self) -> None:
        result = _run_with_tmp(CLEAR_WITH_DEBT_BODY)
        self.assertEqual(result.returncode, 0)
        self.assertIn("WARNING", result.stdout)


if __name__ == "__main__":
    unittest.main()
