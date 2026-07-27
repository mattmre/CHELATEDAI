#!/usr/bin/env python3
"""Fail closed unless the complete v2 legacy nDCG quarantine is exact."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence

if __package__:
    from .build_metric_lineage_quarantine_v2 import (
        BACKEND_EVIDENCE,
        BLOCKED_STATE,
        FIRST_PASS_WARNING_DOCS,
        INDEX_ID,
        PROSE_TYPE,
        ROOT,
        SCHEMA_VERSION,
        SOURCE_HEAD,
        BuildError,
        build_documents,
    )
else:
    from build_metric_lineage_quarantine_v2 import (  # type: ignore[no-redef]
        BACKEND_EVIDENCE,
        BLOCKED_STATE,
        FIRST_PASS_WARNING_DOCS,
        INDEX_ID,
        PROSE_TYPE,
        ROOT,
        SCHEMA_VERSION,
        SOURCE_HEAD,
        BuildError,
        build_documents,
    )


DEFAULT_INDEX = ROOT / "artifacts" / "legacy-ndcg-quarantine-index-v2.json"
EXACT_INDEX_KEYS = {
    "schema_version",
    "index_id",
    "state",
    "authored_date",
    "source_head",
    "defect",
    "artifact_count",
    "artifact_counts",
    "artifact_shard_count",
    "artifact_shards",
    "raw_dispositions",
    "accepted_use_vocabulary",
    "prohibited_use_vocabulary",
    "immutability_policy",
    "preservation_contract",
    "legacy_v1",
    "backend_resolution_evidence",
    "tracked_debt",
}
EXACT_SHARD_KEYS = {
    "schema_version",
    "shard_id",
    "artifact_type",
    "artifact_count",
    "artifacts",
}
EXACT_ARTIFACT_KEYS = {
    "path",
    "artifact_type",
    "git_blob_sha1",
    "sha256",
    "defect_id",
    "state",
    "accepted_uses",
    "prohibited_uses",
    "raw_disposition_ids",
    "supersession",
}
EXACT_PROSE_ARTIFACT_KEYS = EXACT_ARTIFACT_KEYS | {"provenance_versions"}
CONTROL_SURFACE_MARKERS = {
    "CHANGELOG.md": (
        "LEGACY_METRIC_LINEAGE_BLOCKED",
        "legacy-ndcg-quarantine-index-v2.json",
        "113 affected tracked artifacts",
        "not a confirmed negative",
    ),
    "docs/ROADMAP_EXECUTION.md": (
        "LEGACY_METRIC_LINEAGE_BLOCKED",
        "legacy-ndcg-quarantine-index-v2.json",
        "113 affected tracked",
        "no accepted fail/win/rejection claim exists",
    ),
    "docs/next-session.md": (
        "CD-MLR-01",
        "DS-MLR-01",
        "legacy-ndcg-quarantine-index-v2.json",
        "113/113 affected tracked artifacts",
        "fail-closed validator",
    ),
    "docs/research/pr292-metric-lineage-reconditioning-2026-07.md": (
        "113 tracked artifacts",
        "ORIGINAL_PRE_RECONDITIONING_SOURCE",
        "RECONDITIONED_WARNING_SURFACE",
        "legacy-ndcg-quarantine-index-v2.json",
    ),
}


class ValidationError(ValueError):
    """Raised when quarantine state differs from the closed v2 contract."""


def _git(*args: str) -> bytes:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise ValidationError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout


def _safe_repo_path(value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{field} must be a non-empty string")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or "\\" in value:
        raise ValidationError(f"{field} must be a safe POSIX repository-relative path: {value!r}")
    return ROOT.joinpath(*pure.parts)


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _git_blob_sha1(content: bytes) -> str:
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content).hexdigest()  # noqa: S324 - Git object ID


def _require_exact_keys(
    payload: Mapping[str, Any],
    expected: set[str],
    *,
    field: str,
) -> None:
    observed = set(payload)
    if observed != expected:
        raise ValidationError(
            f"{field} key enum mismatch; "
            f"missing={sorted(expected - observed)}, "
            f"extra={sorted(observed - expected)}"
        )


def _assert_exact(actual: Any, expected: Any, *, field: str) -> None:
    if actual != expected:
        raise ValidationError(
            f"{field} does not match the exact v2 contract; " f"observed={actual!r}, expected={expected!r}"
        )


def _validate_warning_text(path: str, content: str) -> None:
    first_lines = "\n".join(content.splitlines()[:16])
    normalized = " ".join(line.lstrip("> ").strip() for line in first_lines.splitlines())
    if "LEGACY_METRIC_LINEAGE_BLOCKED" not in first_lines:
        raise ValidationError(f"{path}: quarantine warning must appear in the first 16 lines")
    if "not accepted evidence" not in normalized.lower():
        raise ValidationError(f"{path}: warning must say the retained claims are not accepted evidence")


def _validate_control_surface_text(path: str, content: str) -> None:
    markers = CONTROL_SURFACE_MARKERS.get(path)
    if markers is None:
        raise ValidationError(f"{path}: no closed control-surface marker enum")
    lowered = content.lower()
    for marker in markers:
        if marker.lower() not in lowered:
            raise ValidationError(f"{path}: missing required claim-boundary marker {marker!r}")


def _validate_prose_versions(entry: Mapping[str, Any]) -> None:
    path = str(entry["path"])
    versions = entry.get("provenance_versions")
    if not isinstance(versions, dict):
        raise ValidationError(f"{path}: provenance_versions must be an object")
    _require_exact_keys(
        versions,
        {"historical_source", "active_surface"},
        field=f"{path}.provenance_versions",
    )
    historical = versions["historical_source"]
    active = versions["active_surface"]
    if not isinstance(historical, dict) or not isinstance(active, dict):
        raise ValidationError(f"{path}: provenance version rows must be objects")
    _require_exact_keys(
        historical,
        {"role", "commit", "git_blob_sha1", "sha256"},
        field=f"{path}.historical_source",
    )
    _require_exact_keys(
        active,
        {
            "role",
            "warning_change_id",
            "first_warning_commit",
            "git_blob_sha1",
            "sha256",
        },
        field=f"{path}.active_surface",
    )
    if historical["role"] != "ORIGINAL_PRE_RECONDITIONING_SOURCE":
        raise ValidationError(f"{path}: invalid historical provenance role")
    if active["role"] != "RECONDITIONED_WARNING_SURFACE":
        raise ValidationError(f"{path}: invalid active provenance role")
    if active["warning_change_id"] != "PR292_V2_BLAST_RADIUS_PASS":
        raise ValidationError(f"{path}: invalid warning change id")
    expected_first_commit = "4d49d9ce45cb5234ed9f1d5d7af1085e23001975" if path in FIRST_PASS_WARNING_DOCS else None
    if active["first_warning_commit"] != expected_first_commit:
        raise ValidationError(f"{path}: incorrect first warning commit")

    commit = historical["commit"]
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValidationError(f"{path}: historical source commit must be full SHA")
    observed_blob = _git("rev-parse", f"{commit}:{path}").decode("ascii").strip()
    if historical["git_blob_sha1"] != observed_blob:
        raise ValidationError(f"{path}: historical source blob does not match commit")
    historical_content = _git("cat-file", "blob", observed_blob)
    if historical["sha256"] != _sha256(historical_content):
        raise ValidationError(f"{path}: historical source SHA-256 mismatch")
    if historical["git_blob_sha1"] == active["git_blob_sha1"]:
        raise ValidationError(f"{path}: original source and warning surface must be distinct blobs")
    if active["git_blob_sha1"] != entry["git_blob_sha1"]:
        raise ValidationError(f"{path}: active provenance blob differs from artifact")
    if active["sha256"] != entry["sha256"]:
        raise ValidationError(f"{path}: active provenance SHA differs from artifact")


def _validate_artifact_shape(entry: Any, artifact_type: str) -> str:
    if not isinstance(entry, dict):
        raise ValidationError("each artifact row must be an object")
    expected_keys = EXACT_PROSE_ARTIFACT_KEYS if artifact_type == PROSE_TYPE else EXACT_ARTIFACT_KEYS
    path = str(entry.get("path", "<missing>"))
    _require_exact_keys(entry, expected_keys, field=f"artifact[{path}]")
    if entry["artifact_type"] != artifact_type:
        raise ValidationError(f"{path}: artifact_type={entry['artifact_type']!r}, " f"expected {artifact_type!r}")
    if entry["state"] != BLOCKED_STATE:
        raise ValidationError(f"{path}: state must remain {BLOCKED_STATE}")
    if artifact_type == PROSE_TYPE:
        _validate_prose_versions(entry)
    return path


def _validate_backend_evidence(rows: Any) -> None:
    _assert_exact(rows, [BACKEND_EVIDENCE], field="backend_resolution_evidence")
    row = rows[0]
    source_path = row["source_path"]
    _safe_repo_path(source_path, field="backend source_path")
    commit = row["source_commit"]
    blob = row["git_blob_sha1"]
    observed_blob = _git("rev-parse", f"{commit}:{source_path}").decode("ascii").strip()
    if observed_blob != blob:
        raise ValidationError("backend evidence commit:path blob drift")
    ancestor = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if ancestor.returncode != 0:
        raise ValidationError(f"backend source commit {commit} is not reachable")
    content = _git("cat-file", "blob", blob).decode("utf-8", errors="replace")
    positions = []
    for marker in row["required_markers"]:
        position = content.find(marker)
        if position < 0:
            raise ValidationError(f"backend evidence blob is missing marker: {marker!r}")
        positions.append(position)
    if positions != sorted(positions) or len(set(positions)) != len(positions):
        raise ValidationError("backend evidence markers must appear once in causal order")


def _validate_repository_evidence(
    index: Mapping[str, Any],
    shards: Mapping[str, Mapping[str, Any]],
) -> None:
    preservation = index["preservation_contract"]
    anchor = preservation["anchor_commit"]
    preserved_types = set(preservation["byte_preserved_artifact_types"])
    preserved_count = 0
    for shard in shards.values():
        artifact_type = shard["artifact_type"]
        for entry in shard["artifacts"]:
            path = entry["path"]
            file_path = _safe_repo_path(path, field="artifact path")
            content = file_path.read_bytes()
            if entry["sha256"] != _sha256(content):
                raise ValidationError(f"{path}: active SHA-256 drift")
            if entry["git_blob_sha1"] != _git_blob_sha1(content):
                raise ValidationError(f"{path}: active Git blob drift")
            if artifact_type in preserved_types:
                anchor_blob = _git("rev-parse", f"{anchor}:{path}").decode("ascii").strip()
                if anchor_blob != entry["git_blob_sha1"]:
                    raise ValidationError(f"{path}: historical JSON/PNG changed from {anchor}")
                preserved_count += 1
            if artifact_type == PROSE_TYPE:
                _validate_warning_text(path, file_path.read_text(encoding="utf-8"))
    if preserved_count != preservation["byte_preserved_artifact_count"]:
        raise ValidationError("preservation artifact count differs from observed preserved rows")
    for path in CONTROL_SURFACE_MARKERS:
        content = _safe_repo_path(path, field="control surface").read_text(encoding="utf-8")
        _validate_control_surface_text(path, content)


def validate_payloads(
    index: Mapping[str, Any],
    shards: Mapping[str, Mapping[str, Any]],
    *,
    check_repository: bool = True,
) -> Sequence[str]:
    """Validate supplied payloads against the independently rebuilt contract."""

    try:
        expected_index, expected_shards = build_documents()
    except BuildError as exc:
        raise ValidationError(str(exc)) from exc

    _require_exact_keys(index, EXACT_INDEX_KEYS, field="index")
    _assert_exact(index.get("schema_version"), SCHEMA_VERSION, field="schema_version")
    _assert_exact(index.get("index_id"), INDEX_ID, field="index_id")
    _assert_exact(index.get("source_head"), SOURCE_HEAD, field="source_head")

    expected_shard_paths = set(expected_shards)
    if set(shards) != expected_shard_paths:
        raise ValidationError(
            "shard path enum mismatch; "
            f"missing={sorted(expected_shard_paths - set(shards))}, "
            f"extra={sorted(set(shards) - expected_shard_paths)}"
        )

    observed_paths = []
    for shard_row in index["artifact_shards"]:
        path = shard_row["path"]
        if path not in shards:
            raise ValidationError(f"index references missing shard: {path}")
        shard = shards[path]
        _require_exact_keys(shard, EXACT_SHARD_KEYS, field=f"shard[{path}]")
        artifact_type = shard["artifact_type"]
        rows = shard["artifacts"]
        if not isinstance(rows, list):
            raise ValidationError(f"{path}: artifacts must be a list")
        if shard["artifact_count"] != len(rows):
            raise ValidationError(f"{path}: fabricated shard artifact_count")
        for entry in rows:
            observed_paths.append(_validate_artifact_shape(entry, artifact_type))
        content = _json_bytes(shard)
        if shard_row["sha256"] != _sha256(content):
            raise ValidationError(f"{path}: shard SHA-256 mismatch")
        if shard_row["git_blob_sha1"] != _git_blob_sha1(content):
            raise ValidationError(f"{path}: shard Git blob mismatch")

    if len(observed_paths) != len(set(observed_paths)):
        raise ValidationError("duplicate artifact paths across shards")
    if len(observed_paths) != index["artifact_count"]:
        raise ValidationError("index artifact_count differs from observed paths")

    _assert_exact(index, expected_index, field="complete index")
    for path in expected_shard_paths:
        _assert_exact(shards[path], expected_shards[path], field=f"shard[{path}]")

    _validate_backend_evidence(index["backend_resolution_evidence"])
    if check_repository:
        _validate_repository_evidence(index, shards)
    return sorted(observed_paths)


def _read_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValidationError(f"{path}: top-level JSON must be an object")
    return payload


def validate(index_path: Path = DEFAULT_INDEX) -> Sequence[str]:
    index = _read_json(index_path)
    shards: Dict[str, Dict[str, Any]] = {}
    rows = index.get("artifact_shards")
    if not isinstance(rows, list):
        raise ValidationError("artifact_shards must be a list")
    for row in rows:
        if not isinstance(row, dict):
            raise ValidationError("artifact_shards rows must be objects")
        path_value = row.get("path")
        shard_path = _safe_repo_path(path_value, field="artifact_shards.path")
        if path_value in shards:
            raise ValidationError(f"duplicate shard path: {path_value}")
        shards[str(path_value)] = _read_json(shard_path)
    return validate_payloads(index, shards)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX,
        help="v2 quarantine index (default: repository canonical v2 index).",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parser().parse_args(argv)
    index_path = args.index if args.index.is_absolute() else ROOT / args.index
    try:
        paths = validate(index_path)
    except (
        OSError,
        json.JSONDecodeError,
        KeyError,
        TypeError,
        ValidationError,
    ) as exc:
        print(f"METRIC_LINEAGE_QUARANTINE: FAIL: {exc}", file=sys.stderr)
        return 1
    print(
        "METRIC_LINEAGE_QUARANTINE: PASS "
        f"(artifacts={len(paths)}/113, raw=89, aggregates=8, plots=6, "
        "prose=10, backend_evidence=1, debt=CD-MLR-01, deferred=DS-MLR-01)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
