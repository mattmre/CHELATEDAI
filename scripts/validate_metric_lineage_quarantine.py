#!/usr/bin/env python3
"""Validate the immutable legacy nDCG quarantine/supersession sidecar."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Optional


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INDEX = ROOT / "artifacts" / "legacy-ndcg-quarantine-index-v1.json"
EXPECTED_ARTIFACT_PATHS = {
    "docs/drift-recovery-h4-compound-cycles-ablation-2026-07.md",
    "docs/drift-recovery-post-bank-headtohead-nfcorpus-results-2026-06.md",
    "docs/drift-recovery-post-bank-headtohead-results-2026-06.md",
    "docs/drift-recovery-swap-nfcorpus-results-2026-06.md",
    "docs/drift-recovery-swap-results-2026-06.md",
    "experiment_runs/drift-recovery/h4-compound/C4a_compound0_seed42.json",
    "experiment_runs/drift-recovery/h4-compound/C4a_compound1_seed42.json",
    "experiment_runs/drift-recovery/post-bank-headtohead-nfcorpus/post-bank-headtohead-manifest-2026-06.json",
    "experiment_runs/drift-recovery/post-bank-headtohead/post-bank-headtohead-manifest-2026-06.json",
    "experiment_runs/drift-recovery/swap-nfcorpus/swap-campaign-manifest-2026-06.json",
    "experiment_runs/drift-recovery/swap/swap-campaign-manifest-2026-06.json",
}
REQUIRED_ACCEPTED_USES = {"historical_audit", "configuration_audit"}
REQUIRED_PROHIBITED_USES = {
    "scientific_performance_claim",
    "condition_or_comparator_ordering",
    "promotion_or_rejection_decision",
    "paper_table_abstract_or_release_claim",
}
MANIFEST_ROW_KEYS = ("rows", "main_rows", "budget_rows", "budget_confirm_rows")


class ValidationError(ValueError):
    """Raised when the quarantine contract does not match repository evidence."""


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


def _safe_repo_path(value: Any, *, field: str) -> tuple[str, Path]:
    if not isinstance(value, str) or not value:
        raise ValidationError(f"{field} must be a non-empty string")
    pure = PurePosixPath(value)
    if pure.is_absolute() or ".." in pure.parts or "\\" in value:
        raise ValidationError(f"{field} must be a safe POSIX repository-relative path: {value!r}")
    path = ROOT.joinpath(*pure.parts)
    return value, path


def _require_string_list(value: Any, *, field: str) -> list[str]:
    if not isinstance(value, list) or not value or not all(isinstance(item, str) and item for item in value):
        raise ValidationError(f"{field} must be a non-empty list of strings")
    if len(value) != len(set(value)):
        raise ValidationError(f"{field} contains duplicate values")
    return value


def _manifest_referenced_paths(payload: dict[str, Any]) -> list[str]:
    paths: set[str] = set()
    for key in MANIFEST_ROW_KEYS:
        rows = payload.get(key, [])
        if rows is None:
            continue
        if not isinstance(rows, list):
            raise ValidationError(f"manifest field {key!r} must be a list")
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("path"), str):
                raise ValidationError(f"manifest field {key!r} contains a row without a string path")
            paths.add(row["path"])
    return sorted(paths)


def _validate_raw_disposition(entry: dict[str, Any], artifact_path: Path) -> None:
    raw = entry.get("raw_disposition")
    if not isinstance(raw, dict):
        raise ValidationError(f"{entry['path']}: raw_disposition must be an object")
    for field in ("state", "referenced_path_count", "missing_from_tip_count", "source_paths"):
        if field not in raw:
            raise ValidationError(f"{entry['path']}: raw_disposition missing {field}")
    if not isinstance(raw["state"], str) or not raw["state"]:
        raise ValidationError(f"{entry['path']}: raw_disposition.state must be a non-empty string")
    if not isinstance(raw["referenced_path_count"], int) or raw["referenced_path_count"] < 0:
        raise ValidationError(f"{entry['path']}: referenced_path_count must be a non-negative integer")
    if not isinstance(raw["missing_from_tip_count"], int) or raw["missing_from_tip_count"] < 0:
        raise ValidationError(f"{entry['path']}: missing_from_tip_count must be a non-negative integer")
    if not isinstance(raw["source_paths"], list):
        raise ValidationError(f"{entry['path']}: raw_disposition.source_paths must be a list")
    for source in raw["source_paths"]:
        _source_value, source_path = _safe_repo_path(source, field=f"{entry['path']}.source_paths")
        if not source_path.is_file():
            raise ValidationError(f"{entry['path']}: retained source is missing: {source}")

    if entry["artifact_type"] != "historical_summary_manifest":
        return
    payload = json.loads(artifact_path.read_text(encoding="utf-8"))
    referenced = _manifest_referenced_paths(payload)
    missing = [path for path in referenced if not ROOT.joinpath(*PurePosixPath(path).parts).is_file()]
    if raw["referenced_path_count"] != len(referenced):
        raise ValidationError(
            f"{entry['path']}: referenced_path_count={raw['referenced_path_count']} "
            f"but manifest has {len(referenced)}"
        )
    if raw["missing_from_tip_count"] != len(missing):
        raise ValidationError(
            f"{entry['path']}: missing_from_tip_count={raw['missing_from_tip_count']} " f"but observed {len(missing)}"
        )


def _validate_artifact(entry: Any) -> str:
    if not isinstance(entry, dict):
        raise ValidationError("each artifacts entry must be an object")
    required = {
        "path",
        "artifact_type",
        "git_blob_sha1",
        "sha256",
        "defect_id",
        "status",
        "accepted_uses",
        "prohibited_uses",
        "raw_disposition",
        "supersession",
    }
    missing_fields = sorted(required - set(entry))
    if missing_fields:
        raise ValidationError(f"artifact entry missing fields: {', '.join(missing_fields)}")

    path_value, path = _safe_repo_path(entry["path"], field="artifacts.path")
    if not path.is_file():
        raise ValidationError(f"{path_value}: artifact is missing")
    if entry["defect_id"] != "MLR-NDCG-001":
        raise ValidationError(f"{path_value}: unexpected defect_id {entry['defect_id']!r}")
    if entry["status"] != "LEGACY_METRIC_LINEAGE_BLOCKED":
        raise ValidationError(f"{path_value}: status must be LEGACY_METRIC_LINEAGE_BLOCKED")

    content = path.read_bytes()
    observed_sha256 = hashlib.sha256(content).hexdigest()
    if entry["sha256"] != observed_sha256:
        raise ValidationError(f"{path_value}: SHA-256 drift " f"(index={entry['sha256']}, observed={observed_sha256})")
    observed_blob = _git("hash-object", "--no-filters", "--", path_value).decode("ascii").strip()
    if entry["git_blob_sha1"] != observed_blob:
        raise ValidationError(
            f"{path_value}: Git blob drift " f"(index={entry['git_blob_sha1']}, observed={observed_blob})"
        )

    accepted = set(_require_string_list(entry["accepted_uses"], field=f"{path_value}.accepted_uses"))
    prohibited = set(_require_string_list(entry["prohibited_uses"], field=f"{path_value}.prohibited_uses"))
    if not REQUIRED_ACCEPTED_USES.issubset(accepted):
        raise ValidationError(f"{path_value}: accepted_uses omits required audit uses")
    if not REQUIRED_PROHIBITED_USES.issubset(prohibited):
        raise ValidationError(f"{path_value}: prohibited_uses omits a required claim class")

    supersession = entry["supersession"]
    if not isinstance(supersession, dict):
        raise ValidationError(f"{path_value}: supersession must be an object")
    expected_supersession = {
        "state": "awaiting_corrected_regeneration",
        "replacement_path": None,
        "tracked_by": "CD-MLR-01",
    }
    if supersession != expected_supersession:
        raise ValidationError(f"{path_value}: supersession must remain pending CD-MLR-01")

    _validate_raw_disposition(entry, path)
    return path_value


def _validate_backend_evidence(rows: Any) -> None:
    if not isinstance(rows, list) or len(rows) != 1 or not isinstance(rows[0], dict):
        raise ValidationError("backend_resolution_evidence must contain exactly one bounded evidence row")
    row = rows[0]
    for field in (
        "claim",
        "scope_limit",
        "source_commit",
        "source_path",
        "git_blob_sha1",
        "required_markers",
        "production_path",
    ):
        if field not in row:
            raise ValidationError(f"backend evidence missing {field}")
    source_path, _unused = _safe_repo_path(row["source_path"], field="backend source_path")
    commit = row["source_commit"]
    blob = row["git_blob_sha1"]
    if not isinstance(commit, str) or len(commit) != 40:
        raise ValidationError("backend source_commit must be a full 40-character Git commit")
    if not isinstance(blob, str) or len(blob) != 40:
        raise ValidationError("backend git_blob_sha1 must be a full 40-character Git blob")
    observed_blob = _git("rev-parse", f"{commit}:{source_path}").decode("ascii").strip()
    if observed_blob != blob:
        raise ValidationError(f"backend evidence blob drift (index={blob}, commit-path={observed_blob})")
    completed = subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        raise ValidationError(f"backend source commit {commit} is not reachable from HEAD")
    content = _git("cat-file", "blob", blob).decode("utf-8", errors="replace")
    for marker in _require_string_list(row["required_markers"], field="backend required_markers"):
        if marker not in content:
            raise ValidationError(f"backend evidence blob is missing marker: {marker!r}")
    _require_string_list(row["production_path"], field="backend production_path")
    prohibited_scope_terms = ("does not prove", "metric", "performance")
    lowered_scope = str(row["scope_limit"]).lower()
    if not all(term in lowered_scope for term in prohibited_scope_terms):
        raise ValidationError("backend scope_limit must explicitly exclude metrics and performance")


def _validate_authoritative_prose() -> None:
    required_markers = {
        "CHANGELOG.md": (
            "LEGACY_METRIC_LINEAGE_BLOCKED",
            "this is not a confirmed negative",
            "do not currently prove",
            "procedural only",
        ),
        "docs/ROADMAP_EXECUTION.md": (
            "LEGACY_METRIC_LINEAGE_BLOCKED",
            "neither is a confirmed negative",
            "no accepted fail/win/rejection claim exists",
        ),
        "docs/next-session.md": (
            "CD-MLR-01",
            "DS-MLR-01",
            "procedural supersession only",
            "narrow procedural path proof only",
        ),
    }
    for relative, markers in required_markers.items():
        content = (ROOT / relative).read_text(encoding="utf-8").lower()
        for marker in markers:
            if marker.lower() not in content:
                raise ValidationError(f"{relative}: missing required claim-boundary marker {marker!r}")

    result_docs = [
        "docs/drift-recovery-h4-compound-cycles-ablation-2026-07.md",
        "docs/drift-recovery-post-bank-headtohead-nfcorpus-results-2026-06.md",
        "docs/drift-recovery-post-bank-headtohead-results-2026-06.md",
        "docs/drift-recovery-swap-nfcorpus-results-2026-06.md",
        "docs/drift-recovery-swap-results-2026-06.md",
    ]
    for relative in result_docs:
        first_lines = "\n".join((ROOT / relative).read_text(encoding="utf-8").splitlines()[:8])
        if "LEGACY_METRIC_LINEAGE_BLOCKED" not in first_lines:
            raise ValidationError(f"{relative}: quarantine warning must remain in the first eight lines")


def validate(index_path: Path = DEFAULT_INDEX) -> list[str]:
    payload = json.loads(index_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValidationError("schema_version must be exactly 1")
    if payload.get("index_id") != "legacy-ndcg-quarantine-v1":
        raise ValidationError("index_id must be legacy-ndcg-quarantine-v1")
    if payload.get("status") != "ACTIVE_QUARANTINE":
        raise ValidationError("status must remain ACTIVE_QUARANTINE")
    if payload.get("artifact_count") != 11:
        raise ValidationError("artifact_count must remain 11")
    defects = payload.get("defects")
    if not isinstance(defects, dict) or set(defects) != {"MLR-NDCG-001"}:
        raise ValidationError("defects must define exactly MLR-NDCG-001")
    if "all positive qrels" not in defects["MLR-NDCG-001"].get("summary", ""):
        raise ValidationError("MLR-NDCG-001 summary must name the qrels-complete IDCG defect")

    rows = payload.get("artifacts")
    if not isinstance(rows, list) or len(rows) != payload["artifact_count"]:
        raise ValidationError("artifacts length does not match artifact_count")
    observed_paths = [_validate_artifact(row) for row in rows]
    if len(observed_paths) != len(set(observed_paths)):
        raise ValidationError("artifacts contains duplicate paths")
    if set(observed_paths) != EXPECTED_ARTIFACT_PATHS:
        missing = sorted(EXPECTED_ARTIFACT_PATHS - set(observed_paths))
        extra = sorted(set(observed_paths) - EXPECTED_ARTIFACT_PATHS)
        raise ValidationError(f"artifact coverage mismatch; missing={missing}, extra={extra}")

    _validate_backend_evidence(payload.get("backend_resolution_evidence"))
    if payload.get("tracked_debt") != {
        "carried_debt_id": "CD-MLR-01",
        "deferred_scope_id": "DS-MLR-01",
    }:
        raise ValidationError("tracked_debt must bind CD-MLR-01 and DS-MLR-01")
    _validate_authoritative_prose()
    return observed_paths


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--index",
        type=Path,
        default=DEFAULT_INDEX,
        help="Versioned quarantine index to validate (default: repository v1 sidecar).",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parser().parse_args(argv)
    index_path = args.index if args.index.is_absolute() else ROOT / args.index
    try:
        paths = validate(index_path)
    except (OSError, json.JSONDecodeError, ValidationError) as exc:
        print(f"METRIC_LINEAGE_QUARANTINE: FAIL: {exc}", file=sys.stderr)
        return 1
    print(
        "METRIC_LINEAGE_QUARANTINE: PASS "
        f"(artifacts={len(paths)}, backend_evidence=1, "
        "debt=CD-MLR-01, deferred=DS-MLR-01)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
