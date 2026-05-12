"""Regression tests for scripts/validate_v33_schema_drift.py.

The real v3.3 architecture tree is expected to stay validator-clean after
Phase C reconciliation. Planted drift in fixture copies still fails with the
expected exit-code class.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "validate_v33_schema_drift.py"
REAL_ARCH_DOC = ROOT / "docs/conventions/brutal-honesty-kit/v3.3-architecture.md"
REAL_ARTIFACT_DIR = ROOT / "docs/conventions/brutal-honesty-kit/v3.3"
BASELINE_OUTPUT = ROOT / "tests/fixtures/v33-validator-baseline-output.txt"
UUIDV7 = "018f6c2a-7b3c-7d00-8a00-111111111111"


LANE_HISTORY_EXAMPLES = """```jsonl
{"ts":"2026-05-10T14:32:00Z","kind":"lane_open","lane_id":"lane-test","plan_sha":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","plan_pr_count":3}
{"ts":"2026-05-10T14:33:00Z","kind":"pr_open","lane_id":"lane-test","pr":42,"phase":"Phase 2","intent":"implement validator fixture","branch":"docs/test-branch"}
{"ts":"2026-05-10T14:34:00Z","kind":"tier_a_iter","lane_id":"lane-test","pr":42,"iter":1,"bhs_self_draft":85,"bhs_self_draft_severity":"important"}
{"ts":"2026-05-10T14:35:00Z","kind":"tier_b_run","lane_id":"lane-test","pr":42,"agent":"gpt-5","family":"openai","bhs_tier_b":100,"severity":"none","bhs_official":100}
{"ts":"2026-05-10T14:36:00Z","kind":"pivot","lane_id":"lane-test","pr":42,"kind_of_pivot":"deferred-on-dependency","recovery_plan":"follow-up PR planned: unblock dependency","blocker_pr":47}
{"ts":"2026-05-10T14:37:00Z","kind":"merge","lane_id":"lane-test","pr":42,"bhs_official":100,"override":null,"merge_sha":"abcdef1"}
{"ts":"2026-05-10T14:38:00Z","kind":"lane_pause","lane_id":"lane-test","cause":"forward_progress:rolling_average_floor_violated","details":"rolling average below floor","last_event_index":6}
{"ts":"2026-05-10T14:39:00Z","kind":"lane_resume","lane_id":"lane-test","operator_signoff":"operator","paused_at_event_index":6}
```"""


CARRIED_DEBT_EXAMPLES = f"""```jsonl
{{"ts":"2026-05-10T14:32:05Z","kind":"cd_open","cd_id":"CD-014","lane_history_ref":"lane-history.jsonl:482","ttl_cycles_initial":1,"ttl_cycles_remaining":1,"blocking":true,"status":"open","item_text":"Resolve validator drift item","source_text":"Tier B found validator drift","opened_at_lane_history_index":483,"crc32":"deadbeef"}}
{{"ts":"2026-05-10T14:33:05Z","kind":"cd_tick","cd_id":"CD-014","ttl_cycles_remaining":0,"target_cycle_id":"{UUIDV7}","crc32":"deadbeef"}}
{{"ts":"2026-05-10T14:34:05Z","kind":"cd_close","cd_id":"CD-014","closed_at_lane_history_index":612,"close_reason":"operator_reverted_drift_in_PR_88","crc32":"deadbeef"}}
{{"ts":"2026-05-10T14:35:05Z","kind":"cd_block","cd_id":"CD-015","blocked_at_lane_history_index":700,"blocked_reason":"ttl_remaining_zero_and_status_still_open","target_cycle_id":"{UUIDV7}","crc32":"deadbeef"}}
```"""


def run_validator(
    architecture_doc: Path = REAL_ARCH_DOC,
    artifact_dir: Path = REAL_ARTIFACT_DIR,
    json_report: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    args = [
        sys.executable,
        str(SCRIPT),
        "--architecture-doc",
        str(architecture_doc),
        "--artifact-dir",
        str(artifact_dir),
    ]
    if json_report is not None:
        args.extend(["--json-report", str(json_report)])
    return subprocess.run(args, cwd=ROOT, capture_output=True, text=True)


def copy_real_tree(tmp_path: Path) -> tuple[Path, Path]:
    arch_doc = tmp_path / "v3.3-architecture.md"
    artifact_dir = tmp_path / "v3.3"
    shutil.copy2(REAL_ARCH_DOC, arch_doc)
    shutil.copytree(REAL_ARTIFACT_DIR, artifact_dir)
    return arch_doc, artifact_dir


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def replace_first_jsonl_blocks(text: str) -> str:
    first_start = text.index("```jsonl")
    first_end = text.index("```", first_start + len("```jsonl")) + 3
    text = text[:first_start] + LANE_HISTORY_EXAMPLES + text[first_end:]

    second_start = text.index("```jsonl", first_start + len(LANE_HISTORY_EXAMPLES))
    second_end = text.index("```", second_start + len("```jsonl")) + 3
    return text[:second_start] + CARRIED_DEBT_EXAMPLES + text[second_end:]


def replace_table_row(text: str, kind: str, replacement: str) -> str:
    lines = []
    prefix = f"| `{kind}` |"
    for line in text.splitlines():
        if line.startswith(prefix):
            lines.append(replacement)
        else:
            lines.append(line)
    return "\n".join(lines) + "\n"


def remove_table_row(text: str, kind: str) -> str:
    prefix = f"| `{kind}` |"
    return "\n".join(
        line for line in text.splitlines()
        if not line.startswith(prefix)
    ) + "\n"


def make_known_good_fixture(tmp_path: Path) -> tuple[Path, Path]:
    arch_doc, artifact_dir = copy_real_tree(tmp_path)

    (artifact_dir / "enums/pivot-kinds.txt").write_text(
        "deferred-on-dependency\nscope-reduced\nabandoned-with-replacement\n",
        encoding="utf-8",
    )
    (artifact_dir / "enums/state-recovery-reasons.txt").write_text(
        "crc_mismatch\ntruncated_json\nmissing_required_field\n"
        "heartbeat_gap_exceeded\npost_merge_drift_detected\n",
        encoding="utf-8",
    )

    lane_schema_path = artifact_dir / "schemas/lane-history.schema.json"
    lane_schema = load_json(lane_schema_path)
    pivot_branch = next(
        branch for branch in lane_schema["oneOf"] if branch["title"] == "pivot"
    )
    pivot_branch["allOf"] = [
        {
            "if": {"properties": {"kind_of_pivot": {"const": "deferred-on-dependency"}}},
            "then": {"required": ["blocker_pr"]},
        }
    ]
    write_json(lane_schema_path, lane_schema)

    cd_schema_path = artifact_dir / "schemas/carried-debt.schema.json"
    cd_schema = load_json(cd_schema_path)
    cd_open_branch = next(
        branch for branch in cd_schema["oneOf"] if branch["title"] == "cd_open"
    )
    cd_open_branch["properties"]["lane_history_ref"] = {
        "oneOf": [
            {"type": "string", "pattern": "^lane-history\\.jsonl:[0-9]+$"},
            {"const": "none"},
        ]
    }
    write_json(cd_schema_path, cd_schema)

    cd_table_path = artifact_dir / "tables/cd-state-machine.yaml"
    cd_table = cd_table_path.read_text(encoding="utf-8")
    cd_table = cd_table.replace("current_status == \"open\"", "status == \"open\"")
    cd_table = cd_table.replace("current_status == \"blocked\"", "status == \"blocked\"")
    cd_table_path.write_text(cd_table, encoding="utf-8")

    text = arch_doc.read_text(encoding="utf-8")
    text = replace_first_jsonl_blocks(text)
    text = replace_table_row(
        text,
        "cd_tick",
        "| `cd_tick` | `ttl_cycles_remaining` (int >= 0, the decremented value), `target_cycle_id` (UUIDv7 cycle id that caused the tick) | One `cd_tick` per cycle boundary while `status` remains `open`. |",
    )
    text = replace_table_row(
        text,
        "cd_block",
        "| `cd_block` | `blocked_at_lane_history_index` (int), `blocked_reason` (string, typically `ttl_remaining_zero_and_status_still_open` or `operator_explicit_block`), `target_cycle_id` (UUIDv7 cycle id that caused the block) | Indicates this `cd_id` has caused the block flag to flip. |",
    )
    text = text.replace("current_status == \"open\"", "status == \"open\"")
    text = text.replace("current_status == \"blocked\"", "status == \"blocked\"")
    arch_doc.write_text(text, encoding="utf-8")
    return arch_doc, artifact_dir


def issue_checks(output: str) -> set[str]:
    checks: set[str] = set()
    for line in output.splitlines():
        match = re.match(r"\s*-\s+\[([^\]\s]+)", line)
        if match:
            checks.add(match.group(1))
    return checks


def apply_known_iter8_drift(arch_doc: Path, artifact_dir: Path) -> None:
    (artifact_dir / "enums/pivot-kinds.txt").write_text(
        "phase_reassignment\nscope_reduction\nscope_substitution\n",
        encoding="utf-8",
    )
    (artifact_dir / "enums/state-recovery-reasons.txt").write_text(
        "corrupted_record_crc_mismatch\nmanual_log_edit_detected\n"
        "operator_initiated_rebuild\nschema_version_upgrade\n"
        "truncated_record_eof_mid_line\n",
        encoding="utf-8",
    )

    lane_schema_path = artifact_dir / "schemas/lane-history.schema.json"
    lane_schema = load_json(lane_schema_path)
    pivot_branch = next(
        branch for branch in lane_schema["oneOf"] if branch["title"] == "pivot"
    )
    pivot_branch.pop("allOf", None)
    write_json(lane_schema_path, lane_schema)

    cd_schema_path = artifact_dir / "schemas/carried-debt.schema.json"
    cd_schema = load_json(cd_schema_path)
    cd_open_branch = next(
        branch for branch in cd_schema["oneOf"] if branch["title"] == "cd_open"
    )
    cd_open_branch["properties"]["lane_history_ref"] = {
        "type": "string",
        "pattern": "^lane-history\\.jsonl:[0-9]+$",
    }
    write_json(cd_schema_path, cd_schema)

    text = arch_doc.read_text(encoding="utf-8")
    text = text.replace("`lane_id`; each line", "`lane_id`, and `crc32`; each line")
    text = replace_table_row(
        text,
        "cd_tick",
        "| `cd_tick` | `ttl_cycles_remaining` (int >= 0, the decremented value), `reason` (string, e.g. `cycle_boundary_2026-05-15`) | One `cd_tick` per cycle boundary while `status` remains `open`. |",
    )
    text = replace_table_row(
        text,
        "cd_block",
        "| `cd_block` | `blocked_at_lane_history_index` (int), `blocked_reason` (string, typically `ttl_remaining_zero_and_status_still_open` or `operator_explicit_block`) | Indicates this `cd_id` has caused the block flag to flip. |",
    )
    text = text.replace(
        '(status == "open" AND ttl_cycles_remaining >= 1) OR (status == "blocked")',
        '(current_status == "open" AND ttl_cycles_remaining >= 1) OR (current_status == "blocked")',
        1,
    )
    arch_doc.write_text(text, encoding="utf-8")


class TestValidatorV33(unittest.TestCase):
    def setUp(self) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self._tmpdir.name)

    def tearDown(self) -> None:
        self._tmpdir.cleanup()

    def test_current_tree_exits_zero_after_phase_c_reconciliation(self) -> None:
        report_path = self.tmp_path / "report.json"
        result = run_validator(json_report=report_path)

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("RESULT: PASS", result.stdout)
        payload = load_json(report_path)
        self.assertEqual(payload["issues"], [])

    def test_synthetic_known_good_fixture_exits_zero(self) -> None:
        arch_doc, artifact_dir = make_known_good_fixture(self.tmp_path)

        result = run_validator(arch_doc, artifact_dir)

        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertIn("RESULT: PASS", result.stdout)

    def test_known_iter8_drift_oracle_is_exercised(self) -> None:
        arch_doc, artifact_dir = make_known_good_fixture(self.tmp_path)
        apply_known_iter8_drift(arch_doc, artifact_dir)

        result = run_validator(arch_doc, artifact_dir)

        baseline_checks = issue_checks(BASELINE_OUTPUT.read_text(encoding="utf-8"))
        oracle_prefixes = (
            "prose.6.5.base_required_fields",
            "prose.6.5b.cd_block.required_fields",
            "prose.6.5b.cd_tick.required_fields",
            "prose.cd_open.lane_history_ref_none",
            "prose.cd_state_machine.audit_predicate.multiple_definitions",
            "prose.pivot.blocker_pr_conditional_required",
            "prose.pivot.kind_of_pivot",
            "prose.state_recovery.recovery_reason",
        )
        expected_checks = {
            check for check in baseline_checks
            if check.startswith(oracle_prefixes)
        }

        self.assertGreaterEqual(len(expected_checks), 10)
        self.assertEqual(result.returncode, 1)
        self.assertTrue(expected_checks <= issue_checks(result.stdout))

    def test_planted_prose_vs_artifact_drift_exits_one(self) -> None:
        drift_names = [
            "enum_plan_fault",
            "event_kind_sentence",
            "extra_lane_history_table_row",
            "missing_lane_history_table_row",
            "extra_carried_debt_table_row",
            "missing_carried_debt_table_row",
            "merge_authority_suffixes",
            "schema_required_field",
            "table_predicate",
            "lane_history_base_field",
        ]
        for drift_name in drift_names:
            with self.subTest(drift_name=drift_name):
                with tempfile.TemporaryDirectory() as td:
                    tmp = Path(td)
                    arch_doc, artifact_dir = make_known_good_fixture(tmp)

                    if drift_name == "enum_plan_fault":
                        text = arch_doc.read_text(encoding="utf-8")
                        text = text.replace(
                            "digest_missing}`",
                            "digest_missing,fixture_extra_fault}`",
                            1,
                        )
                        arch_doc.write_text(text, encoding="utf-8")
                        expected = "fixture_extra_fault"
                    elif drift_name == "event_kind_sentence":
                        text = arch_doc.read_text(encoding="utf-8")
                        text = text.replace(
                            "and `cycle_boundary`",
                            "and `cycle_boundary`, and `fake_event`",
                            1,
                        )
                        arch_doc.write_text(text, encoding="utf-8")
                        expected = "fake_event"
                    elif drift_name == "extra_lane_history_table_row":
                        text = arch_doc.read_text(encoding="utf-8")
                        text = text.replace(
                            "| `state_recovery` |",
                            "| `fake_event` | `fake_field` (string) | Fixture-only bogus row. |\n| `state_recovery` |",
                            1,
                        )
                        arch_doc.write_text(text, encoding="utf-8")
                        expected = "fake_event"
                    elif drift_name == "missing_lane_history_table_row":
                        text = remove_table_row(arch_doc.read_text(encoding="utf-8"), "lane_open")
                        arch_doc.write_text(text, encoding="utf-8")
                        expected = "lane_open"
                    elif drift_name == "extra_carried_debt_table_row":
                        text = arch_doc.read_text(encoding="utf-8")
                        text = text.replace(
                            "| `cd_block` |",
                            "| `cd_fake` | `fake_field` (string) | Fixture-only bogus row. |\n| `cd_block` |",
                            1,
                        )
                        arch_doc.write_text(text, encoding="utf-8")
                        expected = "cd_fake"
                    elif drift_name == "missing_carried_debt_table_row":
                        text = remove_table_row(arch_doc.read_text(encoding="utf-8"), "cd_open")
                        arch_doc.write_text(text, encoding="utf-8")
                        expected = "cd_open"
                    elif drift_name == "merge_authority_suffixes":
                        merge_path = artifact_dir / "enums/merge-authority-suffixes.txt"
                        text = merge_path.read_text(encoding="utf-8")
                        text = text.replace(
                            "rulebook_§6.1:bhs_official_100",
                            "rulebook_§6.1:bogus_valid_suffix",
                            1,
                        )
                        merge_path.write_text(text, encoding="utf-8")
                        expected = "bogus_valid_suffix"
                    elif drift_name == "schema_required_field":
                        schema_path = artifact_dir / "schemas/carried-debt.schema.json"
                        schema = load_json(schema_path)
                        cd_tick = next(branch for branch in schema["oneOf"] if branch["title"] == "cd_tick")
                        cd_tick["properties"]["fixture_required_field"] = {"type": "string"}
                        cd_tick["required"] = [
                            "ttl_cycles_remaining",
                            "target_cycle_id",
                            "fixture_required_field",
                        ]
                        write_json(schema_path, schema)
                        text = arch_doc.read_text(encoding="utf-8")
                        text = text.replace(
                            f'"target_cycle_id":"{UUIDV7}","crc32":"deadbeef"',
                            f'"target_cycle_id":"{UUIDV7}","fixture_required_field":"present","crc32":"deadbeef"',
                        )
                        arch_doc.write_text(text, encoding="utf-8")
                        expected = "fixture_required_field"
                    elif drift_name == "table_predicate":
                        table_path = artifact_dir / "tables/cd-state-machine.yaml"
                        table_text = table_path.read_text(encoding="utf-8")
                        table_text = table_text.replace('status == "blocked"', 'status == "parked"', 1)
                        table_path.write_text(table_text, encoding="utf-8")
                        expected = "audit_predicate"
                    else:
                        text = arch_doc.read_text(encoding="utf-8")
                        text = text.replace("`lane_id`; each line", "`lane_id`, and `crc32`; each line")
                        arch_doc.write_text(text, encoding="utf-8")
                        expected = "base_required_fields"

                    result = run_validator(arch_doc, artifact_dir)

                    self.assertEqual(result.returncode, 1, f"{drift_name}: {result.stdout + result.stderr}")
                    self.assertTrue(
                        expected in result.stdout or expected in result.stderr,
                        f"{drift_name}: expected '{expected}' in output",
                    )

    def test_planted_jsonl_example_drift_exits_two(self) -> None:
        arch_doc, artifact_dir = make_known_good_fixture(self.tmp_path)
        text = arch_doc.read_text(encoding="utf-8")
        text = text.replace('"crc32":"deadbeef"}', '"crc32":"nothex"}', 1)
        arch_doc.write_text(text, encoding="utf-8")

        result = run_validator(arch_doc, artifact_dir)

        self.assertEqual(result.returncode, 2)
        self.assertIn("crc32", result.stdout)


if __name__ == "__main__":
    unittest.main()


def test_planted_invalid_timestamp_exits_two(tmp_path: Path) -> None:
    arch_doc, artifact_dir = make_known_good_fixture(tmp_path)
    text = arch_doc.read_text(encoding="utf-8")
    text = text.replace('"ts":"2026-05-10T14:32:00Z"', '"ts":"not-a-date"', 1)
    arch_doc.write_text(text, encoding="utf-8")

    result = run_validator(arch_doc, artifact_dir)

    assert result.returncode == 2
    assert "jsonl.ts.utc_timestamp" in result.stdout


def test_planted_lane_history_crc32_json_body_exits_two(tmp_path: Path) -> None:
    arch_doc, artifact_dir = make_known_good_fixture(tmp_path)
    text = arch_doc.read_text(encoding="utf-8")
    text = text.replace('"plan_pr_count":3}', '"plan_pr_count":3,"crc32":"deadbeef"}', 1)
    arch_doc.write_text(text, encoding="utf-8")

    result = run_validator(arch_doc, artifact_dir)

    assert result.returncode == 2
    assert "crc32_in_json_body" in result.stdout


def test_planted_unknown_jsonl_kind_exits_three(tmp_path: Path) -> None:
    arch_doc, artifact_dir = make_known_good_fixture(tmp_path)
    text = arch_doc.read_text(encoding="utf-8")
    text = text.replace('"kind":"merge"', '"kind":"mystery_event"', 1)
    arch_doc.write_text(text, encoding="utf-8")

    result = run_validator(arch_doc, artifact_dir)

    assert result.returncode == 3
    assert "mystery_event" in result.stdout


def test_malformed_artifact_exits_four_with_line_number(tmp_path: Path) -> None:
    arch_doc, artifact_dir = make_known_good_fixture(tmp_path)
    plan_fault_path = artifact_dir / "enums/plan-faults.txt"
    plan_fault_path.write_text("valid_suffix\nBAD_SUFFIX\n", encoding="utf-8")

    result = run_validator(arch_doc, artifact_dir)

    assert result.returncode == 4
    assert "line 2" in result.stdout or "line 2" in result.stderr
    assert "BAD_SUFFIX" in result.stdout or "BAD_SUFFIX" in result.stderr


def test_malformed_merge_authority_suffix_exits_four(tmp_path: Path) -> None:
    arch_doc, artifact_dir = make_known_good_fixture(tmp_path)
    merge_path = artifact_dir / "enums/merge-authority-suffixes.txt"
    merge_path.write_text("rulebook_6.1:bhs_official_100\n", encoding="utf-8")

    result = run_validator(arch_doc, artifact_dir)

    assert result.returncode == 4
    assert "merge_authority_suffixes" in result.stdout
