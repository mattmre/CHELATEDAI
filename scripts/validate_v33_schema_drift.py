#!/usr/bin/env python3
"""Validate v3.3 Brutal Honesty Kit schema/prose/artifact drift.

This is the Phase 1 validator specified by
docs/conventions/brutal-honesty-kit/v3.3-architecture.md section 5.1.
It treats the v3.3 artifacts as machine-readable domains and checks that:

* enum/schema/table artifacts are parseable and internally well-formed;
* JSON Schema discriminator/enum values match the corresponding enum files;
* bounded prose sections in v3.3-architecture.md name the same identifier
  sets and required fields as the artifacts;
* fenced ```jsonl examples validate against the schema selected by `kind`.

Exit codes:
  0 clean
  1 prose-vs-artifact drift
  2 JSONL example schema-validation failure
  3 schema-vs-enum drift
  4 unreadable or malformed artifact
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import jsonschema
import yaml


DEFAULT_ARCHITECTURE_DOC = Path(
    "docs/conventions/brutal-honesty-kit/v3.3-architecture.md"
)
DEFAULT_ARTIFACT_DIR = Path("docs/conventions/brutal-honesty-kit/v3.3")

EXIT_PROSE_ARTIFACT = 1
EXIT_JSONL_EXAMPLE = 2
EXIT_SCHEMA_ENUM = 3
EXIT_MALFORMED_ARTIFACT = 4
EXIT_PRIORITY = (
    EXIT_MALFORMED_ARTIFACT,
    EXIT_PROSE_ARTIFACT,
    EXIT_SCHEMA_ENUM,
    EXIT_JSONL_EXAMPLE,
)

ENUM_FILES = {
    "lane_history_kinds": "lane-history-kinds.txt",
    "cd_record_kinds": "cd-record-kinds.txt",
    "lane_pause_causes": "lane-pause-causes.txt",
    "plan_faults": "plan-faults.txt",
    "merge_authority_suffixes": "merge-authority-suffixes.txt",
    "state_recovery_reasons": "state-recovery-reasons.txt",
    "pivot_kinds": "pivot-kinds.txt",
}

ENUM_PATTERNS = {
    "lane_history_kinds": re.compile(r"^[a-z][a-z0-9_]*$"),
    "cd_record_kinds": re.compile(r"^[a-z][a-z0-9_]*$"),
    "lane_pause_causes": re.compile(r"^(?:[a-z][a-z0-9_]*|F11)$"),
    "plan_faults": re.compile(r"^[a-z0-9_]+$"),
    "merge_authority_suffixes": re.compile(
        r"^(?:rulebook_\u00a7[0-9]+(?:\.[0-9]+)?|charter_\u00a7[0-9]+):[a-z0-9_]+$"
    ),
    "state_recovery_reasons": re.compile(r"^[a-z][a-z0-9_]*$"),
    "pivot_kinds": re.compile(r"^[a-z][a-z0-9_-]*$"),
}

UUIDV7_EXAMPLE = "018f6c2a-7b3c-7d00-8a00-111111111111"
UTC_TIMESTAMP_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?Z$"
)


@dataclass(frozen=True)
class Issue:
    code: int
    category: str
    check: str
    message: str
    artifact_path: str | None = None
    doc_section: str | None = None
    json_pointer: str | None = None
    line: int | None = None
    expected: list[str] | None = None
    actual: list[str] | None = None


@dataclass
class LoadedArtifacts:
    root: Path
    enums: dict[str, set[str]]
    schemas: dict[str, dict[str, Any]]
    tables: dict[str, dict[str, Any]]


def _rel(path: Path) -> str:
    return str(path).replace("\\", "/")


def _sorted(values: Iterable[str]) -> list[str]:
    return sorted(set(values))


def add_issue(
    issues: list[Issue],
    code: int,
    category: str,
    check: str,
    message: str,
    *,
    artifact_path: Path | str | None = None,
    doc_section: str | None = None,
    json_pointer: str | None = None,
    line: int | None = None,
    expected: Iterable[str] | None = None,
    actual: Iterable[str] | None = None,
) -> None:
    issues.append(
        Issue(
            code=code,
            category=category,
            check=check,
            message=message,
            artifact_path=_rel(Path(artifact_path)) if artifact_path else None,
            doc_section=doc_section,
            json_pointer=json_pointer,
            line=line,
            expected=_sorted(expected) if expected is not None else None,
            actual=_sorted(actual) if actual is not None else None,
        )
    )


def parse_enum_file(path: Path, name: str, issues: list[Issue]) -> set[str]:
    pattern = ENUM_PATTERNS[name]
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        add_issue(
            issues,
            EXIT_MALFORMED_ARTIFACT,
            "malformed_artifact",
            f"artifact.{name}.read",
            f"cannot read {path}: {exc}",
            artifact_path=path,
        )
        return set()

    values: set[str] = set()
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if not pattern.fullmatch(line):
            add_issue(
                issues,
                EXIT_MALFORMED_ARTIFACT,
                "malformed_artifact",
                f"artifact.{name}.line",
                f"{path} line {line_no} contains invalid identifier {line!r}",
                artifact_path=path,
                line=line_no,
            )
            continue
        if line in values:
            add_issue(
                issues,
                EXIT_MALFORMED_ARTIFACT,
                "malformed_artifact",
                f"artifact.{name}.duplicate",
                f"{path} line {line_no} duplicates identifier {line!r}",
                artifact_path=path,
                line=line_no,
            )
        values.add(line)
    return values


def load_json_schema(path: Path, issues: list[Issue]) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        add_issue(
            issues,
            EXIT_MALFORMED_ARTIFACT,
            "malformed_artifact",
            f"artifact.schema.{path.stem}.parse",
            f"cannot parse JSON schema {path}: {exc}",
            artifact_path=path,
        )
        return {}
    try:
        jsonschema.Draft202012Validator.check_schema(data)
    except jsonschema.SchemaError as exc:
        add_issue(
            issues,
            EXIT_MALFORMED_ARTIFACT,
            "malformed_artifact",
            f"artifact.schema.{path.stem}.schema",
            f"invalid JSON Schema {path}: {exc.message}",
            artifact_path=path,
            json_pointer="/" + "/".join(str(p) for p in exc.path),
        )
    return data


def load_yaml_table(path: Path, issues: list[Issue]) -> dict[str, Any]:
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        add_issue(
            issues,
            EXIT_MALFORMED_ARTIFACT,
            "malformed_artifact",
            f"artifact.table.{path.stem}.parse",
            f"cannot parse YAML table {path}: {exc}",
            artifact_path=path,
        )
        return {}
    if not isinstance(data, dict):
        add_issue(
            issues,
            EXIT_MALFORMED_ARTIFACT,
            "malformed_artifact",
            f"artifact.table.{path.stem}.shape",
            f"YAML table {path} must parse as a mapping",
            artifact_path=path,
        )
        return {}
    return data


def load_artifacts(artifact_dir: Path, issues: list[Issue]) -> LoadedArtifacts:
    enums_dir = artifact_dir / "enums"
    schemas_dir = artifact_dir / "schemas"
    tables_dir = artifact_dir / "tables"

    enums: dict[str, set[str]] = {}
    for name, filename in ENUM_FILES.items():
        enums[name] = parse_enum_file(enums_dir / filename, name, issues)

    schemas = {
        "lane_history": load_json_schema(schemas_dir / "lane-history.schema.json", issues),
        "carried_debt": load_json_schema(schemas_dir / "carried-debt.schema.json", issues),
    }

    tables = {
        "cd_state_machine": load_yaml_table(tables_dir / "cd-state-machine.yaml", issues),
        "cycle_clock": load_yaml_table(tables_dir / "cycle-clock.yaml", issues),
    }

    return LoadedArtifacts(root=artifact_dir, enums=enums, schemas=schemas, tables=tables)


def code_spans(text: str) -> list[str]:
    return re.findall(r"`([^`\n]+)`", text)


def split_pipe_values(text: str) -> set[str]:
    return {part.strip() for part in text.split("|") if part.strip()}


def normalize_required_token(token: str) -> str:
    return token.strip().strip("`").strip()


def schema_branch(schema: dict[str, Any], kind: str) -> dict[str, Any] | None:
    for branch in schema.get("oneOf", []):
        if not isinstance(branch, dict):
            continue
        const = (
            branch.get("properties", {})
            .get("kind", {})
            .get("const")
        )
        if const == kind:
            return branch
        if branch.get("title") == kind:
            return branch
    return None


def schema_kind_enum(schema: dict[str, Any]) -> set[str]:
    values = schema.get("properties", {}).get("kind", {}).get("enum", [])
    return set(values) if isinstance(values, list) else set()


def schema_branch_consts(schema: dict[str, Any]) -> set[str]:
    values: set[str] = set()
    for branch in schema.get("oneOf", []):
        if not isinstance(branch, dict):
            continue
        const = branch.get("properties", {}).get("kind", {}).get("const")
        if isinstance(const, str):
            values.add(const)
    return values


def schema_field_enum(schema: dict[str, Any], kind: str, field: str) -> set[str]:
    branch = schema_branch(schema, kind)
    if not branch:
        return set()
    values = (
        branch.get("properties", {})
        .get(field, {})
        .get("enum", [])
    )
    return set(values) if isinstance(values, list) else set()


def compare_sets(
    issues: list[Issue],
    *,
    check: str,
    left_name: str,
    left_values: set[str],
    right_name: str,
    right_values: set[str],
    code: int,
    category: str,
    artifact_path: Path | str | None = None,
    doc_section: str | None = None,
) -> None:
    left_extra = left_values - right_values
    right_extra = right_values - left_values
    if left_extra:
        add_issue(
            issues,
            code,
            category,
            f"{check}.extra_in_{left_name}",
            f"{left_name} names {_sorted(left_extra)} not in {right_name}",
            artifact_path=artifact_path,
            doc_section=doc_section,
            expected=right_values,
            actual=left_values,
        )
    if right_extra:
        add_issue(
            issues,
            code,
            category,
            f"{check}.extra_in_{right_name}",
            f"{right_name} declares {_sorted(right_extra)} not in {left_name}",
            artifact_path=artifact_path,
            doc_section=doc_section,
            expected=left_values,
            actual=right_values,
        )


def strip_fenced_blocks(text: str) -> str:
    return re.sub(r"^```.*?^```", "", text, flags=re.MULTILINE | re.DOTALL)


def get_section(text: str, heading: str, next_heading: str | None = None) -> str:
    start = re.search(rf"^###\s+{re.escape(heading)}\b.*$", text, re.MULTILINE)
    if not start:
        return ""
    if next_heading is None:
        end = re.search(r"^###\s+\d+\.\d+\b.*$", text[start.end():], re.MULTILINE)
        return text[start.end(): start.end() + end.start()] if end else text[start.end():]
    end = re.search(rf"^###\s+{re.escape(next_heading)}\b.*$", text, re.MULTILINE)
    return text[start.end(): end.start()] if end else text[start.end():]


def split_markdown_row(line: str) -> list[str]:
    """Split a markdown table row on pipes that are outside code spans."""
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]

    cells: list[str] = []
    current: list[str] = []
    in_code = False
    for char in stripped:
        if char == "`":
            in_code = not in_code
            current.append(char)
            continue
        if char == "|" and not in_code:
            cells.append("".join(current).strip())
            current = []
            continue
        current.append(char)
    cells.append("".join(current).strip())
    return cells


def markdown_rows(section: str) -> dict[str, tuple[str, str]]:
    rows: dict[str, tuple[str, str]] = {}
    for raw_line in section.splitlines():
        line = raw_line.strip()
        if not line.startswith("| `"):
            continue
        cells = split_markdown_row(line)
        if len(cells) < 3:
            continue
        kind = cells[0].strip("`")
        rows[kind] = (cells[1], cells[2])
    return rows


def prose_required_fields(required_cell: str, schema_props: set[str]) -> set[str]:
    fields: set[str] = set()
    for match in re.finditer(r"`([^`\n]+)`", required_cell):
        token = match.group(1)
        normalized = normalize_required_token(token)
        tail = required_cell[match.end(): match.end() + 8]
        looks_like_field_definition = tail.lstrip().startswith("(")
        if normalized in schema_props or (
            looks_like_field_definition and re.fullmatch(r"[a-z][a-z0-9_]*", normalized)
        ):
            fields.add(normalized)
    return fields


def find_jsonl_blocks(text: str) -> list[tuple[int, str]]:
    blocks: list[tuple[int, str]] = []
    pattern = re.compile(r"^```jsonl\s*\n(.*?)^```", re.MULTILINE | re.DOTALL)
    for match in pattern.finditer(text):
        line_no = text[: match.start()].count("\n") + 1
        blocks.append((line_no, match.group(1)))
    return blocks


def pointer_for_error(error: jsonschema.ValidationError) -> str:
    if error.path:
        return "$" + "".join(f"/{part}" for part in error.path)
    if error.schema_path:
        return "$ (schema " + "/".join(str(part) for part in error.schema_path) + ")"
    return "$"


def schema_accepts_literal(schema_fragment: dict[str, Any], value: Any) -> bool:
    try:
        jsonschema.Draft202012Validator(schema_fragment).validate(value)
        return True
    except jsonschema.ValidationError:
        return False


def example_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Return a stricter schema for documentation examples.

    The committed schemas intentionally describe the record contract but do not
    all set unevaluatedProperties. Examples are executable documentation, so
    they should not carry fields absent from the schema/prose contract.
    """
    strict = copy.deepcopy(schema)
    strict.setdefault("unevaluatedProperties", False)
    return strict


def conditional_requires(branch: dict[str, Any], field: str) -> bool:
    """Return true if a schema branch conditionally requires `field`.

    This detects standard JSON Schema shapes such as:
      {"if": ..., "then": {"required": [field]}}
    nested under the branch or inside any allOf/anyOf/oneOf entry.
    """
    stack = [branch]
    while stack:
        node = stack.pop()
        if not isinstance(node, dict):
            continue
        then = node.get("then")
        if isinstance(then, dict) and field in then.get("required", []):
            return True
        for key in ("allOf", "anyOf", "oneOf"):
            for child in node.get(key, []) if isinstance(node.get(key), list) else []:
                if isinstance(child, dict):
                    stack.append(child)
    return False


def normalized_predicate_expression(text: str) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    return text


def prose_predicate_candidates(doc_text: str) -> set[str]:
    candidates: set[str] = set()
    pattern = re.compile(
        r"\((?:current_)?status\s*==\s*\"open\"\s+AND\s+ttl_cycles_remaining\s*>=\s*1\)\s+OR\s+\((?:current_)?status\s*==\s*\"blocked\"\)",
        re.IGNORECASE,
    )
    for match in pattern.finditer(doc_text):
        candidates.add(normalized_predicate_expression(match.group(0)))
    return candidates


def cycle_clock_triggers_from_yaml(cycle_clock: dict[str, Any]) -> set[str]:
    boundary = str(cycle_clock.get("cycle", {}).get("boundary_definition", ""))
    triggers: set[str] = set()
    if "wrap_session" in boundary:
        triggers.add("wrap_session_skill")
    if "emit_cycle_boundary" in boundary:
        triggers.add("manual_emit_cycle_boundary")
    if "Charter-Phase" in boundary or "charter PR" in boundary:
        triggers.add("charter_phase_pr_merged")
    return triggers


def is_utc_timestamp(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    if not UTC_TIMESTAMP_PATTERN.fullmatch(value):
        return False
    try:
        datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    except ValueError:
        return False
    return True


def validate_schema_vs_enum(artifacts: LoadedArtifacts, issues: list[Issue]) -> None:
    lane_schema = artifacts.schemas["lane_history"]
    cd_schema = artifacts.schemas["carried_debt"]
    enum_dir = artifacts.root / "enums"
    schemas_dir = artifacts.root / "schemas"

    compare_sets(
        issues,
        check="schema.lane_history.kind_enum",
        left_name="lane-history.schema.json properties.kind.enum",
        left_values=schema_kind_enum(lane_schema),
        right_name="enums/lane-history-kinds.txt",
        right_values=artifacts.enums["lane_history_kinds"],
        code=EXIT_SCHEMA_ENUM,
        category="schema_vs_enum",
        artifact_path=schemas_dir / "lane-history.schema.json",
    )
    compare_sets(
        issues,
        check="schema.lane_history.oneof_consts",
        left_name="lane-history.schema.json oneOf kind consts",
        left_values=schema_branch_consts(lane_schema),
        right_name="enums/lane-history-kinds.txt",
        right_values=artifacts.enums["lane_history_kinds"],
        code=EXIT_SCHEMA_ENUM,
        category="schema_vs_enum",
        artifact_path=schemas_dir / "lane-history.schema.json",
    )
    compare_sets(
        issues,
        check="schema.carried_debt.kind_enum",
        left_name="carried-debt.schema.json properties.kind.enum",
        left_values=schema_kind_enum(cd_schema),
        right_name="enums/cd-record-kinds.txt",
        right_values=artifacts.enums["cd_record_kinds"],
        code=EXIT_SCHEMA_ENUM,
        category="schema_vs_enum",
        artifact_path=schemas_dir / "carried-debt.schema.json",
    )
    compare_sets(
        issues,
        check="schema.carried_debt.oneof_consts",
        left_name="carried-debt.schema.json oneOf kind consts",
        left_values=schema_branch_consts(cd_schema),
        right_name="enums/cd-record-kinds.txt",
        right_values=artifacts.enums["cd_record_kinds"],
        code=EXIT_SCHEMA_ENUM,
        category="schema_vs_enum",
        artifact_path=schemas_dir / "carried-debt.schema.json",
    )
    compare_sets(
        issues,
        check="schema.pivot.kind_of_pivot_enum",
        left_name="lane-history.schema.json pivot.kind_of_pivot enum",
        left_values=schema_field_enum(lane_schema, "pivot", "kind_of_pivot"),
        right_name="enums/pivot-kinds.txt",
        right_values=artifacts.enums["pivot_kinds"],
        code=EXIT_SCHEMA_ENUM,
        category="schema_vs_enum",
        artifact_path=enum_dir / "pivot-kinds.txt",
    )
    compare_sets(
        issues,
        check="schema.state_recovery.recovery_reason_enum",
        left_name="lane-history.schema.json state_recovery.recovery_reason enum",
        left_values=schema_field_enum(lane_schema, "state_recovery", "recovery_reason"),
        right_name="enums/state-recovery-reasons.txt",
        right_values=artifacts.enums["state_recovery_reasons"],
        code=EXIT_SCHEMA_ENUM,
        category="schema_vs_enum",
        artifact_path=enum_dir / "state-recovery-reasons.txt",
    )

    lane_pause_branch = schema_branch(lane_schema, "lane_pause") or {}
    cause_pattern = (
        lane_pause_branch.get("properties", {})
        .get("cause", {})
        .get("pattern", "")
    )
    namespaces: set[str] = set()
    m = re.search(r"^\^\(([^)]+)\):", cause_pattern)
    if m:
        namespaces = split_pipe_values(m.group(1))
    compare_sets(
        issues,
        check="schema.lane_pause.cause_namespaces",
        left_name="lane-history.schema.json lane_pause.cause namespaces",
        left_values=namespaces,
        right_name="enums/lane-pause-causes.txt",
        right_values=artifacts.enums["lane_pause_causes"],
        code=EXIT_SCHEMA_ENUM,
        category="schema_vs_enum",
        artifact_path=enum_dir / "lane-pause-causes.txt",
    )

    cycle_schema_triggers = schema_field_enum(lane_schema, "cycle_boundary", "trigger")
    yaml_triggers = cycle_clock_triggers_from_yaml(artifacts.tables["cycle_clock"])
    compare_sets(
        issues,
        check="schema.cycle_boundary.trigger_enum",
        left_name="lane-history.schema.json cycle_boundary.trigger enum",
        left_values=cycle_schema_triggers,
        right_name="tables/cycle-clock.yaml triggers",
        right_values=yaml_triggers,
        code=EXIT_SCHEMA_ENUM,
        category="schema_vs_enum",
        artifact_path=artifacts.root / "tables" / "cycle-clock.yaml",
    )


def prose_pivot_kinds(
    doc_text: str,
    section_61_without_fences: str,
    lane_rows: dict[str, tuple[str, str]],
) -> set[str]:
    values: set[str] = set()

    pivot_text = " ".join(lane_rows.get("pivot", ("", "")))
    pivot_match = re.search(r"kind_of_pivot`\s*\(`([^`]+)`\)", pivot_text)
    if pivot_match:
        values.update(split_pipe_values(pivot_match.group(1)))

    disclosure_match = re.search(
        r"PIVOT_DISCLOSURE:\s*```yaml\s*(.*?)^```",
        doc_text,
        re.MULTILINE | re.DOTALL,
    )
    if disclosure_match:
        kind_line = re.search(r"^\s*kind:\s*([^\n#]+)", disclosure_match.group(1), re.MULTILINE)
        if kind_line:
            values.update(split_pipe_values(kind_line.group(1)))

    allowed_values_match = re.search(
        r"allowed values\s*\((.*?)\)",
        section_61_without_fences,
        re.DOTALL,
    )
    if allowed_values_match:
        values.update(
            token for token in code_spans(allowed_values_match.group(1))
            if re.fullmatch(r"[a-z][a-z0-9_-]*", token)
        )

    return values


def validate_prose_vs_artifacts(
    doc_text: str,
    doc_path: Path,
    artifacts: LoadedArtifacts,
    issues: list[Issue],
) -> None:
    text_no_fences = strip_fenced_blocks(doc_text)
    section_60 = get_section(text_no_fences, "6.0", "6.1")
    section_61 = get_section(text_no_fences, "6.1", "6.2")
    section_65 = get_section(text_no_fences, "6.5", "6.5b")
    section_65b = get_section(text_no_fences, "6.5b", "6.6")
    lane_rows = markdown_rows(section_65)
    cd_rows = markdown_rows(section_65b)
    enum_dir = artifacts.root / "enums"

    event_para_match = re.search(
        r"Event kinds defined for Phase 4:(.*?)\*\*JSON Schema authority\*\*",
        section_65,
        re.DOTALL,
    )
    lane_row_kinds = set(lane_rows)
    event_tokens = set(lane_row_kinds)
    if event_para_match:
        prose_line_tokens = set(code_spans(event_para_match.group(1)))
        event_tokens.update(
            token for token in prose_line_tokens
            if re.fullmatch(r"[a-z][a-z0-9_]*", token)
            and token not in artifacts.enums["cd_record_kinds"]
        )
    compare_sets(
        issues,
        check="prose.lane_history.table_rows",
        left_name="doc section 6.5 kind table rows",
        left_values=lane_row_kinds,
        right_name="enums/lane-history-kinds.txt",
        right_values=artifacts.enums["lane_history_kinds"],
        code=EXIT_PROSE_ARTIFACT,
        category="prose_vs_artifact",
        artifact_path=enum_dir / "lane-history-kinds.txt",
        doc_section="6.5 per-event-kind table",
    )
    compare_sets(
        issues,
        check="prose.lane_history.event_kinds",
        left_name="doc section 6.5 event kinds",
        left_values=event_tokens,
        right_name="enums/lane-history-kinds.txt",
        right_values=artifacts.enums["lane_history_kinds"],
        code=EXIT_PROSE_ARTIFACT,
        category="prose_vs_artifact",
        artifact_path=enum_dir / "lane-history-kinds.txt",
        doc_section="6.5",
    )

    merge_authority_match = re.search(
        r"Merge-authority paths defined for Phase 4:(.*?)^\*\*Authority precedence\*\*",
        section_60,
        re.MULTILINE | re.DOTALL,
    )
    merge_authority_tokens = set()
    if merge_authority_match:
        merge_authority_tokens.update(
            token for token in code_spans(merge_authority_match.group(1))
            if ENUM_PATTERNS["merge_authority_suffixes"].fullmatch(token)
        )
    compare_sets(
        issues,
        check="prose.merge_authority.suffixes",
        left_name="doc section 6.0 merge-authority paths",
        left_values=merge_authority_tokens,
        right_name="enums/merge-authority-suffixes.txt",
        right_values=artifacts.enums["merge_authority_suffixes"],
        code=EXIT_PROSE_ARTIFACT,
        category="prose_vs_artifact",
        artifact_path=enum_dir / "merge-authority-suffixes.txt",
        doc_section="6.0",
    )

    record_para_match = re.search(
        r"\*\*Record kinds defined for Phase 4\*\*:(.*?)\.",
        section_65b,
        re.DOTALL,
    )
    cd_row_kinds = set(cd_rows)
    record_tokens = set()
    if record_para_match:
        record_tokens.update(
            token for token in code_spans(record_para_match.group(1))
            if token in artifacts.enums["cd_record_kinds"] or re.fullmatch(r"[a-z][a-z0-9_]*", token)
        )
    record_tokens.update(cd_row_kinds)
    compare_sets(
        issues,
        check="prose.carried_debt.table_rows",
        left_name="doc section 6.5b kind table rows",
        left_values=cd_row_kinds,
        right_name="enums/cd-record-kinds.txt",
        right_values=artifacts.enums["cd_record_kinds"],
        code=EXIT_PROSE_ARTIFACT,
        category="prose_vs_artifact",
        artifact_path=enum_dir / "cd-record-kinds.txt",
        doc_section="6.5b per-record-kind table",
    )
    compare_sets(
        issues,
        check="prose.carried_debt.record_kinds",
        left_name="doc section 6.5b record kinds",
        left_values=record_tokens,
        right_name="enums/cd-record-kinds.txt",
        right_values=artifacts.enums["cd_record_kinds"],
        code=EXIT_PROSE_ARTIFACT,
        category="prose_vs_artifact",
        artifact_path=enum_dir / "cd-record-kinds.txt",
        doc_section="6.5b",
    )

    lane_pause_text = " ".join(lane_rows.get("lane_pause", ("", "")))
    namespace_match = re.search(r"\^\(([^)]+)\):", lane_pause_text)
    doc_namespaces = split_pipe_values(namespace_match.group(1)) if namespace_match else set()
    compare_sets(
        issues,
        check="prose.lane_pause.namespaces",
        left_name="doc section 6.5 lane_pause namespaces",
        left_values=doc_namespaces,
        right_name="enums/lane-pause-causes.txt",
        right_values=artifacts.enums["lane_pause_causes"],
        code=EXIT_PROSE_ARTIFACT,
        category="prose_vs_artifact",
        artifact_path=enum_dir / "lane-pause-causes.txt",
        doc_section="6.5 lane_pause row",
    )

    plan_fault_tokens = set()
    plan_fault_row_match = re.search(r"plan_fault:\{([^}]+)\}", lane_pause_text)
    if plan_fault_row_match:
        plan_fault_tokens.update(
            token.strip() for token in plan_fault_row_match.group(1).split(",")
            if token.strip()
        )
    plan_fault_tokens.update(re.findall(r"`cause:\s*plan_fault:([a-z0-9_]+)`", text_no_fences))
    compare_sets(
        issues,
        check="prose.plan_fault.suffixes",
        left_name="doc plan_fault suffixes",
        left_values=plan_fault_tokens,
        right_name="enums/plan-faults.txt",
        right_values=artifacts.enums["plan_faults"],
        code=EXIT_PROSE_ARTIFACT,
        category="prose_vs_artifact",
        artifact_path=enum_dir / "plan-faults.txt",
        doc_section="6.5 lane_pause row + code spans",
    )

    doc_pivots = prose_pivot_kinds(doc_text, section_61, lane_rows)
    compare_sets(
        issues,
        check="prose.pivot.kind_of_pivot",
        left_name="doc pivot kind_of_pivot values",
        left_values=doc_pivots,
        right_name="enums/pivot-kinds.txt",
        right_values=artifacts.enums["pivot_kinds"],
        code=EXIT_PROSE_ARTIFACT,
        category="prose_vs_artifact",
        artifact_path=enum_dir / "pivot-kinds.txt",
        doc_section="6.1 + 6.5 pivot row",
    )

    state_text = " ".join(lane_rows.get("state_recovery", ("", "")))
    recovery_match = re.search(r"recovery_reason`\s*\(`([^`]+)`\)", state_text)
    doc_reasons = split_pipe_values(recovery_match.group(1)) if recovery_match else set()
    doc_reasons.update(re.findall(r"`recovery_reason:\s*([a-z0-9_]+)`", text_no_fences))
    compare_sets(
        issues,
        check="prose.state_recovery.recovery_reason",
        left_name="doc state_recovery recovery_reason values",
        left_values=doc_reasons,
        right_name="enums/state-recovery-reasons.txt",
        right_values=artifacts.enums["state_recovery_reasons"],
        code=EXIT_PROSE_ARTIFACT,
        category="prose_vs_artifact",
        artifact_path=enum_dir / "state-recovery-reasons.txt",
        doc_section="6.5 state_recovery row",
    )

    validate_required_field_tables(section_65, lane_rows, artifacts.schemas["lane_history"], "6.5", doc_path, issues)
    validate_required_field_tables(section_65b, cd_rows, artifacts.schemas["carried_debt"], "6.5b", doc_path, issues)
    validate_value_space_claims(lane_rows, cd_rows, artifacts, doc_path, issues)
    validate_predicate_match(doc_text, artifacts, issues)


def validate_required_field_tables(
    section_text: str,
    rows: dict[str, tuple[str, str]],
    schema: dict[str, Any],
    section_name: str,
    doc_path: Path,
    issues: list[Issue],
) -> None:
    if section_name == "6.5b":
        base_match = re.search(r"\*\*Always-required base fields per record\*\*:\s*(.*)", section_text)
    else:
        base_match = re.search(
            r"always-required (?:base|JSON) fields\s+([^;\n)]+)",
            section_text,
        )
    if base_match:
        base_text = " ".join(group for group in base_match.groups() if group)
        prose_base = {
            token for token in code_spans(base_text)
            if re.fullmatch(r"[a-z][a-z0-9_]*", token)
        }
        schema_base = set(schema.get("required", []))
        compare_sets(
            issues,
            check=f"prose.{section_name}.base_required_fields",
            left_name=f"doc section {section_name} base required fields",
            left_values=prose_base,
            right_name="schema top-level required",
            right_values=schema_base,
            code=EXIT_PROSE_ARTIFACT,
            category="prose_vs_artifact",
            artifact_path=doc_path,
            doc_section=section_name,
        )

    for kind, (required_cell, _notes) in rows.items():
        branch = schema_branch(schema, kind)
        if not branch:
            continue
        props = set(branch.get("properties", {}))
        prose_fields = prose_required_fields(required_cell, props)
        schema_fields = set(branch.get("required", []))
        compare_sets(
            issues,
            check=f"prose.{section_name}.{kind}.required_fields",
            left_name=f"doc section {section_name} {kind} required fields",
            left_values=prose_fields,
            right_name=f"schema {kind} required fields",
            right_values=schema_fields,
            code=EXIT_PROSE_ARTIFACT,
            category="prose_vs_artifact",
            artifact_path=doc_path,
            doc_section=f"{section_name} {kind} row",
        )


def validate_value_space_claims(
    lane_rows: dict[str, tuple[str, str]],
    cd_rows: dict[str, tuple[str, str]],
    artifacts: LoadedArtifacts,
    doc_path: Path,
    issues: list[Issue],
) -> None:
    lane_schema = artifacts.schemas["lane_history"]
    cd_schema = artifacts.schemas["carried_debt"]

    pivot_required, pivot_notes = lane_rows.get("pivot", ("", ""))
    pivot_branch = schema_branch(lane_schema, "pivot") or {}
    if "blocker_pr" in pivot_required + pivot_notes and "additionally" in pivot_notes:
        required = "blocker_pr" in pivot_branch.get("required", [])
        conditional = conditional_requires(pivot_branch, "blocker_pr")
        if not required and not conditional:
            add_issue(
                issues,
                EXIT_PROSE_ARTIFACT,
                "prose_vs_artifact",
                "prose.pivot.blocker_pr_conditional_required",
                "doc section 6.5 says pivot kind_of_pivot=deferred-on-dependency additionally requires blocker_pr, but lane-history.schema.json has no required or if/then requirement for blocker_pr",
                artifact_path=artifacts.root / "schemas" / "lane-history.schema.json",
                doc_section="6.5 pivot row",
                expected=["conditional required blocker_pr"],
                actual=["blocker_pr not conditionally required"],
            )

    cd_open_required, _cd_open_notes = cd_rows.get("cd_open", ("", ""))
    cd_open_branch = schema_branch(cd_schema, "cd_open") or {}
    lane_history_ref = (
        cd_open_branch.get("properties", {})
        .get("lane_history_ref", {})
    )
    if "`none`" in cd_open_required or "literal `none`" in cd_open_required:
        if not schema_accepts_literal(lane_history_ref, "none"):
            add_issue(
                issues,
                EXIT_PROSE_ARTIFACT,
                "prose_vs_artifact",
                "prose.cd_open.lane_history_ref_none",
                "doc section 6.5b says cd_open.lane_history_ref permits literal none, but carried-debt.schema.json rejects it",
                artifact_path=artifacts.root / "schemas" / "carried-debt.schema.json",
                doc_section="6.5b cd_open row",
                expected=["lane_history_ref accepts none"],
                actual=["lane_history_ref rejects none"],
            )


def validate_predicate_match(
    doc_text: str,
    artifacts: LoadedArtifacts,
    issues: list[Issue],
) -> None:
    cd_machine = artifacts.tables["cd_state_machine"]
    expression = (
        cd_machine.get("audit_predicate", {})
        .get("expression", "")
    )
    artifact_normalized = normalized_predicate_expression(str(expression))
    prose_candidates = prose_predicate_candidates(strip_fenced_blocks(doc_text))
    if len(prose_candidates) > 1:
        add_issue(
            issues,
            EXIT_PROSE_ARTIFACT,
            "prose_vs_artifact",
            "prose.cd_state_machine.audit_predicate.multiple_definitions",
            "architecture prose contains multiple distinct normalized (beta) predicate definitions",
            artifact_path=artifacts.root / "tables" / "cd-state-machine.yaml",
            doc_section="6.4 / 6.5 / 6.5b / 10.8",
            actual=prose_candidates,
        )
    if artifact_normalized and artifact_normalized not in prose_candidates:
        add_issue(
            issues,
            EXIT_PROSE_ARTIFACT,
            "prose_vs_artifact",
            "prose.cd_state_machine.audit_predicate",
            "tables/cd-state-machine.yaml audit_predicate.expression does not match any prose (beta) predicate definition after whitespace normalization",
            artifact_path=artifacts.root / "tables" / "cd-state-machine.yaml",
            doc_section="6.4 / 6.5 / 6.5b / 10.8",
            expected=prose_candidates,
            actual=[artifact_normalized],
        )


def validate_jsonl_examples(
    doc_text: str,
    doc_path: Path,
    artifacts: LoadedArtifacts,
    issues: list[Issue],
) -> None:
    lane_schema = example_schema(artifacts.schemas["lane_history"])
    cd_schema = example_schema(artifacts.schemas["carried_debt"])
    format_checker = jsonschema.FormatChecker()
    lane_validator = jsonschema.Draft202012Validator(
        lane_schema,
        format_checker=format_checker,
    )
    cd_validator = jsonschema.Draft202012Validator(
        cd_schema,
        format_checker=format_checker,
    )
    lane_kinds = artifacts.enums["lane_history_kinds"]
    cd_kinds = artifacts.enums["cd_record_kinds"]

    for block_line, block in find_jsonl_blocks(doc_text):
        for offset, raw_line in enumerate(block.splitlines(), start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            current_line = block_line + offset
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                add_issue(
                    issues,
                    EXIT_JSONL_EXAMPLE,
                    "jsonl_example",
                    "jsonl.parse",
                    f"{doc_path} line {current_line} is not valid JSON: {exc.msg}",
                    artifact_path=doc_path,
                    line=current_line,
                )
                continue
            if not isinstance(record, dict):
                add_issue(
                    issues,
                    EXIT_JSONL_EXAMPLE,
                    "jsonl_example",
                    "jsonl.shape",
                    f"{doc_path} line {current_line} JSONL record must be an object",
                    artifact_path=doc_path,
                    line=current_line,
                )
                continue
            kind = record.get("kind")
            if not isinstance(kind, str):
                add_issue(
                    issues,
                    EXIT_JSONL_EXAMPLE,
                    "jsonl_example",
                    "jsonl.kind_missing",
                    f"{doc_path} line {current_line} JSONL example lacks string kind field",
                    artifact_path=doc_path,
                    line=current_line,
                )
                continue
            if not is_utc_timestamp(record.get("ts")):
                add_issue(
                    issues,
                    EXIT_JSONL_EXAMPLE,
                    "jsonl_example",
                    "jsonl.ts.utc_timestamp",
                    f"{doc_path} line {current_line} JSONL example has invalid ts; expected ISO-8601 UTC timestamp ending in Z",
                    artifact_path=doc_path,
                    line=current_line,
                )
                continue
            if kind in lane_kinds:
                if "crc32" in record:
                    add_issue(
                        issues,
                        EXIT_JSONL_EXAMPLE,
                        "jsonl_example",
                        "jsonl.lane_history.crc32_in_json_body",
                        f"{doc_path} line {current_line} lane-history example includes crc32 inside the JSON object; §6.5 requires the CRC32 footer outside the JSON body",
                        artifact_path=doc_path,
                        line=current_line,
                    )
                    continue
                validator = lane_validator
                schema_name = "lane-history.schema.json"
            elif kind in cd_kinds:
                validator = cd_validator
                schema_name = "carried-debt.schema.json"
            else:
                add_issue(
                    issues,
                    EXIT_SCHEMA_ENUM,
                    "schema_vs_enum",
                    "jsonl.unknown_kind",
                    f"{doc_path} line {current_line} JSONL example kind {kind!r} appears in neither lane-history nor carried-debt kind enums",
                    artifact_path=doc_path,
                    line=current_line,
                    actual=[kind],
                )
                continue

            errors = list(validator.iter_errors(record))
            if errors:
                best = jsonschema.exceptions.best_match(errors)
                add_issue(
                    issues,
                    EXIT_JSONL_EXAMPLE,
                    "jsonl_example",
                    f"jsonl.{kind}.schema_validation",
                    f"{doc_path} line {current_line} {kind!r} example fails {schema_name}: {best.message}",
                    artifact_path=doc_path,
                    line=current_line,
                    json_pointer=pointer_for_error(best),
                )


def validate(
    architecture_doc: Path,
    artifact_dir: Path,
) -> tuple[int, list[Issue]]:
    issues: list[Issue] = []
    try:
        doc_text = architecture_doc.read_text(encoding="utf-8")
    except OSError as exc:
        add_issue(
            issues,
            EXIT_MALFORMED_ARTIFACT,
            "malformed_artifact",
            "architecture_doc.read",
            f"cannot read architecture doc {architecture_doc}: {exc}",
            artifact_path=architecture_doc,
        )
        return EXIT_MALFORMED_ARTIFACT, issues

    artifacts = load_artifacts(artifact_dir, issues)
    if any(issue.code == EXIT_MALFORMED_ARTIFACT for issue in issues):
        return choose_exit_code(issues), sorted_issues(issues)

    validate_schema_vs_enum(artifacts, issues)
    validate_prose_vs_artifacts(doc_text, architecture_doc, artifacts, issues)
    validate_jsonl_examples(doc_text, architecture_doc, artifacts, issues)
    return choose_exit_code(issues), sorted_issues(issues)


def choose_exit_code(issues: list[Issue]) -> int:
    if not issues:
        return 0
    codes = {issue.code for issue in issues}
    for code in EXIT_PRIORITY:
        if code in codes:
            return code
    return 1


def sorted_issues(issues: list[Issue]) -> list[Issue]:
    return sorted(
        issues,
        key=lambda issue: (
            issue.code,
            issue.category,
            issue.check,
            issue.artifact_path or "",
            issue.line or 0,
            issue.message,
        ),
    )


def render_human_report(
    architecture_doc: Path,
    artifact_dir: Path,
    exit_code: int,
    issues: list[Issue],
) -> str:
    lines = [
        "=" * 78,
        "BHS v3.3 schema drift validator",
        "=" * 78,
        f"Architecture doc: {_rel(architecture_doc)}",
        f"Artifact dir:     {_rel(artifact_dir)}",
        f"Exit code:        {exit_code}",
        "",
    ]
    if not issues:
        lines.append("RESULT: PASS - no schema/prose/artifact drift found.")
        return "\n".join(lines)

    by_code: dict[int, list[Issue]] = {}
    for issue in issues:
        by_code.setdefault(issue.code, []).append(issue)

    labels = {
        EXIT_PROSE_ARTIFACT: "prose-vs-artifact drift",
        EXIT_JSONL_EXAMPLE: "JSONL example schema failures",
        EXIT_SCHEMA_ENUM: "schema-vs-enum drift",
        EXIT_MALFORMED_ARTIFACT: "malformed artifacts",
    }
    for code in sorted(by_code):
        grouped = by_code[code]
        lines.append(f"{labels.get(code, 'issues')} ({len(grouped)}):")
        for issue in grouped:
            location_bits = []
            if issue.doc_section:
                location_bits.append(f"doc {issue.doc_section}")
            if issue.artifact_path:
                location_bits.append(issue.artifact_path)
            if issue.line is not None:
                location_bits.append(f"line {issue.line}")
            if issue.json_pointer:
                location_bits.append(issue.json_pointer)
            location = f" [{' | '.join(location_bits)}]" if location_bits else ""
            lines.append(f"  - [{issue.check}]{location} {issue.message}")
        lines.append("")

    lines.append("RESULT: FAIL - validator output is the Phase B/C punch list.")
    return "\n".join(lines)


def write_json_report(path: Path, exit_code: int, issues: list[Issue]) -> None:
    payload = {
        "schema_version": "1.0",
        "tool": "validate_v33_schema_drift",
        "exit_code": exit_code,
        "issue_count": len(issues),
        "issues": [asdict(issue) for issue in issues],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--architecture-doc",
        default=str(DEFAULT_ARCHITECTURE_DOC),
        help=f"Architecture markdown path (default: {DEFAULT_ARCHITECTURE_DOC})",
    )
    parser.add_argument(
        "--artifact-dir",
        default=str(DEFAULT_ARTIFACT_DIR),
        help=f"v3.3 artifact directory (default: {DEFAULT_ARTIFACT_DIR})",
    )
    parser.add_argument(
        "--json-report",
        help="Write machine-readable JSON report to this path.",
    )
    args = parser.parse_args(argv)

    architecture_doc = Path(args.architecture_doc)
    artifact_dir = Path(args.artifact_dir)
    exit_code, issues = validate(architecture_doc, artifact_dir)
    report = render_human_report(architecture_doc, artifact_dir, exit_code, issues)
    print(report)
    if issues:
        print(f"validate_v33_schema_drift: FAIL with {len(issues)} issue(s); exit {exit_code}", file=sys.stderr)
        for issue in issues[:10]:
            print(f"{issue.check}: {issue.message}", file=sys.stderr)
    if args.json_report:
        write_json_report(Path(args.json_report), exit_code, issues)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
