"""Unit tests for scripts/validate_pr_brutal_honesty.py.

Exercises the §4 PR-body schema parser plus every cross-field rule
(R1–R9) defined in scripts/validate_pr_brutal_honesty.py.

Run with: python -m pytest tests/test_validate_pr_brutal_honesty.py -v
"""

from __future__ import annotations

import pytest

from scripts.validate_pr_brutal_honesty import (
    REQUIRED_FIELDS,
    SEVERITY_CAPS,
    VALID_SEVERITIES,
    parse_fields,
    validate,
)


# ---------------------------------------------------------------------------
# Helpers — keep test PR bodies readable
# ---------------------------------------------------------------------------


def _body(**overrides: str) -> str:
    """Build a clean §4 template body and apply field overrides.

    A field set to None is omitted entirely (so we can test R1-missing).
    """
    defaults = {
        "EVIDENCE": "scripts/smoke.sh ran locally; output below",
        "SMOKE": "scripts/smoke.sh exited 0",
        "BHS_SELF_DRAFT": "100",
        "BHS_SELF_DRAFT_AGENT": "session A, agent claude-sonnet-4.5",
        "BHS_TIER_B": "100",
        "BHS_TIER_B_AGENT": "session B, fresh reviewer agent",
        "BHS_TIER_B_SEVERITY": "none",
        "BHS_OFFICIAL": "100",
        "CARRY_FORWARD": "none",
        "DEFERRED_SCOPE": "none",
        "LOOP_ITERATIONS": "1",
        "OPERATOR_OVERRIDE": "empty",
    }
    defaults.update(overrides)
    lines = ["## Brutal Honesty", ""]
    for field in REQUIRED_FIELDS:
        value = defaults.get(field)
        if value is None:
            continue
        lines.append(f"{field}: {value}")
    return "\n".join(lines) + "\n"


def _fail_codes(report) -> list:
    return [v.rule for v in report.fails]


# ---------------------------------------------------------------------------
# parse_fields() — schema discovery
# ---------------------------------------------------------------------------


class TestParseFields:
    def test_extracts_all_required_fields(self) -> None:
        body = _body()
        fields = parse_fields(body)
        assert set(fields) == set(REQUIRED_FIELDS)

    def test_blockquoted_fields_are_recognized(self) -> None:
        body = "> EVIDENCE: trace\n> SMOKE: ok\n"
        fields = parse_fields(body)
        assert fields["EVIDENCE"] == "trace"
        assert fields["SMOKE"] == "ok"

    def test_last_occurrence_wins(self) -> None:
        body = "BHS_SELF_DRAFT: 50\nBHS_SELF_DRAFT: 100\n"
        assert parse_fields(body)["BHS_SELF_DRAFT"] == "100"

    def test_unknown_fields_ignored(self) -> None:
        body = "EVIDENCE: x\nNOT_A_FIELD: noise\n"
        fields = parse_fields(body)
        assert "NOT_A_FIELD" not in fields
        assert fields["EVIDENCE"] == "x"


# ---------------------------------------------------------------------------
# R1 — required fields present and not the raw template hint
# ---------------------------------------------------------------------------


class TestR1RequiredFields:
    def test_passing_body_emits_no_r1_failure(self) -> None:
        report = validate(_body())
        assert "R1" not in _fail_codes(report)

    @pytest.mark.parametrize("missing", REQUIRED_FIELDS)
    def test_each_missing_field_fires_r1(self, missing: str) -> None:
        report = validate(_body(**{missing: None}))
        # Expect at least one R1 violation naming the missing field.
        r1s = [v for v in report.fails if v.rule == "R1" and v.field == missing]
        assert r1s, f"R1 did not fire for missing field {missing}"

    def test_unfilled_template_hint_fires_r1(self) -> None:
        body = _body(BHS_SELF_DRAFT="<0–100, implementer's self-assessed score>")
        report = validate(body)
        r1s = [v for v in report.fails if v.rule == "R1" and v.field == "BHS_SELF_DRAFT"]
        assert r1s


# ---------------------------------------------------------------------------
# R2 — score parsing
# ---------------------------------------------------------------------------


class TestR2ScoreRange:
    def test_non_integer_score_fires_r2(self) -> None:
        report = validate(_body(BHS_SELF_DRAFT="not a number"))
        assert any(v.rule == "R2" and v.field == "BHS_SELF_DRAFT" for v in report.fails)

    def test_score_above_100_fires_r2(self) -> None:
        # 200 also breaks R5 (min mismatch); we just check R2 fires.
        report = validate(_body(BHS_SELF_DRAFT="200", BHS_OFFICIAL="100"))
        assert any(v.rule == "R2" and v.field == "BHS_SELF_DRAFT" for v in report.fails)

    def test_negative_score_fires_r2(self) -> None:
        report = validate(_body(BHS_TIER_B="-1"))
        assert any(v.rule == "R2" and v.field == "BHS_TIER_B" for v in report.fails)

    def test_score_with_trailing_justification_parses(self) -> None:
        # "100 — Tier B confirmed" should parse as 100.
        report = validate(
            _body(BHS_SELF_DRAFT="100 — Tier B confirmed", BHS_TIER_B="100", BHS_OFFICIAL="100")
        )
        assert "R2" not in _fail_codes(report)


# ---------------------------------------------------------------------------
# R3 — severity vocabulary
# ---------------------------------------------------------------------------


class TestR3SeverityValue:
    @pytest.mark.parametrize("severity", sorted(VALID_SEVERITIES))
    def test_valid_severities_accepted(self, severity: str) -> None:
        # Pick a Tier B score that satisfies the cap so R4 doesn't also fire.
        cap = SEVERITY_CAPS[severity]
        report = validate(
            _body(
                BHS_SELF_DRAFT=str(cap),
                BHS_TIER_B=str(cap),
                BHS_TIER_B_SEVERITY=severity,
                BHS_OFFICIAL=str(cap),
            )
        )
        assert "R3" not in _fail_codes(report)

    def test_invalid_severity_fires_r3(self) -> None:
        report = validate(_body(BHS_TIER_B_SEVERITY="urgent"))
        assert any(v.rule == "R3" for v in report.fails)


# ---------------------------------------------------------------------------
# R4 — severity caps on Tier B
# ---------------------------------------------------------------------------


class TestR4SeverityCaps:
    def test_critical_cap_at_70_fires(self) -> None:
        report = validate(
            _body(
                BHS_SELF_DRAFT="100",
                BHS_TIER_B="80",
                BHS_TIER_B_SEVERITY="critical",
                BHS_OFFICIAL="80",
            )
        )
        r4s = [v for v in report.fails if v.rule == "R4"]
        assert r4s, "critical severity did not cap Tier B at 70"

    def test_important_cap_at_90_fires(self) -> None:
        report = validate(
            _body(
                BHS_SELF_DRAFT="100",
                BHS_TIER_B="95",
                BHS_TIER_B_SEVERITY="important",
                BHS_OFFICIAL="95",
            )
        )
        assert any(v.rule == "R4" for v in report.fails)

    def test_critical_with_70_does_not_fire(self) -> None:
        report = validate(
            _body(
                BHS_SELF_DRAFT="100",
                BHS_TIER_B="70",
                BHS_TIER_B_SEVERITY="critical",
                BHS_OFFICIAL="70",
            )
        )
        assert "R4" not in _fail_codes(report)

    def test_cosmetic_does_not_cap(self) -> None:
        report = validate(
            _body(
                BHS_TIER_B_SEVERITY="cosmetic",
            )
        )
        assert "R4" not in _fail_codes(report)


# ---------------------------------------------------------------------------
# R5 — BHS_OFFICIAL = min(BHS_SELF_DRAFT, capped Tier B)
# ---------------------------------------------------------------------------


class TestR5OfficialMin:
    def test_official_matches_min_passes(self) -> None:
        report = validate(_body(BHS_SELF_DRAFT="80", BHS_TIER_B="100", BHS_OFFICIAL="80"))
        assert "R5" not in _fail_codes(report)

    def test_official_higher_than_min_fires(self) -> None:
        report = validate(_body(BHS_SELF_DRAFT="80", BHS_TIER_B="100", BHS_OFFICIAL="100"))
        assert any(v.rule == "R5" for v in report.fails)

    def test_official_after_severity_cap_fires(self) -> None:
        # Tier B raw 95 → capped at 90 (important) → expected official 90,
        # but the body declares 95.
        report = validate(
            _body(
                BHS_SELF_DRAFT="100",
                BHS_TIER_B="95",
                BHS_TIER_B_SEVERITY="important",
                BHS_OFFICIAL="95",
            )
        )
        # R4 fires for the raw cap and R5 fires for the official mismatch.
        codes = _fail_codes(report)
        assert "R4" in codes
        assert "R5" in codes


# ---------------------------------------------------------------------------
# R6 — agent independence
# ---------------------------------------------------------------------------


class TestR6Independence:
    def test_distinct_agents_pass(self) -> None:
        report = validate(_body())
        assert "R6" not in _fail_codes(report)

    def test_identical_agents_fire_and_force_official_zero(self) -> None:
        report = validate(
            _body(
                BHS_SELF_DRAFT_AGENT="session abc, agent X",
                BHS_TIER_B_AGENT="session abc, agent X",
                BHS_OFFICIAL="100",
            )
        )
        codes = _fail_codes(report)
        assert "R6" in codes
        # R5 should ALSO fire because the expected official is 0, not 100.
        assert "R5" in codes

    def test_case_insensitive_match(self) -> None:
        report = validate(
            _body(
                BHS_SELF_DRAFT_AGENT="Session ABC",
                BHS_TIER_B_AGENT="session abc",
            )
        )
        assert any(v.rule == "R6" for v in report.fails)


# ---------------------------------------------------------------------------
# R7 — LOOP_ITERATIONS bounds + scope-reduce note when capped
# ---------------------------------------------------------------------------


class TestR7Loops:
    @pytest.mark.parametrize("loops", ["0", "6", "-1", "abc"])
    def test_out_of_range_or_unparseable_fires(self, loops: str) -> None:
        report = validate(_body(LOOP_ITERATIONS=loops))
        assert any(v.rule == "R7" for v in report.fails)

    def test_loop_5_with_low_self_draft_requires_explanation(self) -> None:
        # No 'withdraw' or 'scope-reduce' phrase anywhere in the body.
        report = validate(
            _body(
                BHS_SELF_DRAFT="80",
                BHS_TIER_B="80",
                BHS_OFFICIAL="80",
                LOOP_ITERATIONS="5",
            )
        )
        assert any(v.rule == "R7" for v in report.fails)

    def test_loop_5_with_explicit_scope_reduce_passes_r7(self) -> None:
        body = (
            _body(
                BHS_SELF_DRAFT="80",
                BHS_TIER_B="80",
                BHS_OFFICIAL="80",
                LOOP_ITERATIONS="5",
            )
            + "\nThis PR was scope-reduced from the original lane after iteration 5.\n"
        )
        report = validate(body)
        assert "R7" not in _fail_codes(report)


# ---------------------------------------------------------------------------
# R8 — OPERATOR_OVERRIDE structural format
# ---------------------------------------------------------------------------


class TestR8OperatorOverride:
    def test_empty_override_passes(self) -> None:
        report = validate(_body(OPERATOR_OVERRIDE="empty"))
        assert "R8" not in _fail_codes(report)

    def test_three_field_override_fires_r8(self) -> None:
        report = validate(_body(OPERATOR_OVERRIDE="ship it, alex, 2026-05-09T12:00Z"))
        assert any(v.rule == "R8" for v in report.fails)

    def test_four_field_override_passes_r8_but_warns(self) -> None:
        report = validate(
            _body(
                OPERATOR_OVERRIDE=(
                    "small docs gap acceptable, alex, 2026-05-09T12:00Z, https://slack.example/thread/123"
                )
            )
        )
        # No FAIL for R8 (structure is good); still issues a WARN about BLOCKED.
        assert "R8" not in _fail_codes(report)
        assert any(v.rule == "R8" for v in report.warns)


# ---------------------------------------------------------------------------
# R9 — score-gaming auto-disclosure required when gap > 5
# ---------------------------------------------------------------------------


class TestR9ScoreGaming:
    def test_gap_above_5_without_disclosure_fires(self) -> None:
        report = validate(_body(BHS_SELF_DRAFT="100", BHS_TIER_B="80", BHS_OFFICIAL="80"))
        assert any(v.rule == "R9" for v in report.fails)

    def test_gap_above_5_with_disclosure_passes(self) -> None:
        body = (
            _body(BHS_SELF_DRAFT="100", BHS_TIER_B="80", BHS_OFFICIAL="80")
            + "\nL4 score-gaming detected: implementer over-scored draft by 20 points.\n"
        )
        report = validate(body)
        assert "R9" not in _fail_codes(report)

    def test_gap_at_5_does_not_fire(self) -> None:
        report = validate(_body(BHS_SELF_DRAFT="100", BHS_TIER_B="95", BHS_OFFICIAL="95"))
        assert "R9" not in _fail_codes(report)
