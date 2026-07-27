#!/usr/bin/env python3
"""Build the deterministic v2 inventory for legacy drift-recovery nDCG evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple


ROOT = Path(__file__).resolve().parents[1]
SOURCE_HEAD = "835f61992b96412589dff0403b3b43f0af0e52f7"
PRE_WARNING_HEAD = "eb7509583e81b6a62d13b82587398562e4bba09a"
FIRST_WARNING_COMMIT = "4d49d9ce45cb5234ed9f1d5d7af1085e23001975"
DEFECT_ID = "MLR-NDCG-001"
BLOCKED_STATE = "LEGACY_METRIC_LINEAGE_BLOCKED"
SCHEMA_VERSION = "2.0.0"
INDEX_ID = "legacy-ndcg-quarantine-v2"

RAW_TYPE = "historical_raw_run_json"
AGGREGATE_TYPE = "historical_aggregate_json"
PLOT_TYPE = "historical_plot_png"
PROSE_TYPE = "historical_evidence_markdown"

EXPECTED_TYPE_COUNTS = {
    RAW_TYPE: 89,
    AGGREGATE_TYPE: 8,
    PLOT_TYPE: 6,
    PROSE_TYPE: 10,
}
EXPECTED_GROUP_COUNTS = {
    "base_matrix": (30, 0),
    "calibration": (39, 0),
    "knob_sweep": (18, 0),
    "swap_scifact": (0, 22),
    "swap_nfcorpus": (0, 22),
    "post_bank_scifact": (0, 15),
    "post_bank_nfcorpus": (0, 15),
    "h4_compound": (2, 0),
}

ACCEPTED_AUDIT_USES = [
    "historical_audit",
    "configuration_audit",
    "non_quantitative_runtime_structure",
]
ACCEPTED_REPRODUCTION_USES = [
    *ACCEPTED_AUDIT_USES,
    "metric_defect_reproduction",
]
PROHIBITED_USES = [
    "scientific_performance_claim",
    "condition_or_comparator_ordering",
    "promotion_or_rejection_decision",
    "paper_table_abstract_or_release_claim",
]
SUPERSESSION = {
    "state": "awaiting_qrels_complete_regeneration",
    "replacement_path": None,
    "tracked_by": "CD-MLR-01",
}

AGGREGATE_PATHS: Dict[str, Tuple[str, str]] = {
    "experiment_runs/drift-recovery/diagnostics-2026-06.json": (
        "drift_recovery_diagnostics",
        "base_matrix",
    ),
    "experiment_runs/drift-recovery/calibrated/calibration-manifest-2026-06.json": (
        "drift_recovery_calibration_campaign",
        "calibration",
    ),
    "experiment_runs/drift-recovery/calibrated/diagnostics-2026-06.json": (
        "drift_recovery_diagnostics",
        "calibration",
    ),
    "experiment_runs/drift-recovery/knob-sweep/knob-sweep-manifest-2026-06.json": (
        "drift_recovery_knob_sweep",
        "knob_sweep",
    ),
    "experiment_runs/drift-recovery/swap/swap-campaign-manifest-2026-06.json": (
        "drift_recovery_swap_campaign",
        "swap_scifact",
    ),
    "experiment_runs/drift-recovery/swap-nfcorpus/swap-campaign-manifest-2026-06.json": (
        "drift_recovery_swap_campaign",
        "swap_nfcorpus",
    ),
    "experiment_runs/drift-recovery/post-bank-headtohead/post-bank-headtohead-manifest-2026-06.json": (
        "post_bank_head_to_head_campaign",
        "post_bank_scifact",
    ),
    ("experiment_runs/drift-recovery/post-bank-headtohead-nfcorpus/" "post-bank-headtohead-manifest-2026-06.json"): (
        "post_bank_head_to_head_campaign",
        "post_bank_nfcorpus",
    ),
}

PLOT_GROUPS = {
    "experiment_runs/drift-recovery/scifact_noise_ndcg_diagnostics_mean.png": "base_matrix",
    "experiment_runs/drift-recovery/scifact_noise_ndcg_mean_std.png": "base_matrix",
    "experiment_runs/drift-recovery/scifact_rotation_ndcg_diagnostics_mean.png": "base_matrix",
    "experiment_runs/drift-recovery/scifact_rotation_ndcg_mean_std.png": "base_matrix",
    ("experiment_runs/drift-recovery/calibrated/" "scifact_noise_ndcg_diagnostics_mean.png"): "calibration",
    ("experiment_runs/drift-recovery/calibrated/" "scifact_rotation_ndcg_diagnostics_mean.png"): "calibration",
}

PROSE_GROUPS: Dict[str, List[str]] = {
    "docs/drift-recovery-results-2026-06.md": [
        "base_matrix",
        "calibration",
        "knob_sweep",
    ],
    "docs/drift-recovery-calibrated-results-2026-06.md": ["calibration"],
    "docs/drift-recovery-diagnostics-2026-06.md": ["base_matrix"],
    "docs/drift-recovery-knob-sweep-2026-06.md": ["knob_sweep"],
    "docs/drift-recovery-swap-results-2026-06.md": ["swap_scifact"],
    "docs/drift-recovery-swap-nfcorpus-results-2026-06.md": ["swap_nfcorpus"],
    "docs/drift-recovery-post-bank-headtohead-results-2026-06.md": ["post_bank_scifact"],
    "docs/drift-recovery-post-bank-headtohead-nfcorpus-results-2026-06.md": ["post_bank_nfcorpus"],
    "docs/drift-recovery-h4-compound-cycles-ablation-2026-07.md": ["h4_compound"],
    "experiment_runs/drift-recovery/calibrated/diagnostics-2026-06.md": ["calibration"],
}

FIRST_PASS_WARNING_DOCS = {
    "docs/drift-recovery-h4-compound-cycles-ablation-2026-07.md",
    "docs/drift-recovery-post-bank-headtohead-results-2026-06.md",
    "docs/drift-recovery-post-bank-headtohead-nfcorpus-results-2026-06.md",
    "docs/drift-recovery-swap-results-2026-06.md",
    "docs/drift-recovery-swap-nfcorpus-results-2026-06.md",
}

SHARD_SPECS = [
    (
        "raw-runs",
        RAW_TYPE,
        "artifacts/legacy-ndcg-quarantine-v2/raw-runs.json",
    ),
    (
        "aggregates",
        AGGREGATE_TYPE,
        "artifacts/legacy-ndcg-quarantine-v2/aggregates.json",
    ),
    (
        "plots",
        PLOT_TYPE,
        "artifacts/legacy-ndcg-quarantine-v2/plots.json",
    ),
    (
        "prose",
        PROSE_TYPE,
        "artifacts/legacy-ndcg-quarantine-v2/prose.json",
    ),
]

MANIFEST_ROW_KEYS = (
    "rows",
    "main_rows",
    "budget_rows",
    "budget_confirm_rows",
)

BACKEND_EVIDENCE = {
    "claim": (
        "The local all-mpnet-base-v2 backend resolved and reported vector size "
        "768 during the retained NFCorpus campaign."
    ),
    "scope_limit": (
        "This proves only one real backend resolution in the retained NFCorpus "
        "campaign. It does not prove an H2-specific rerun, metric correctness, "
        "performance, condition ordering, recovery, promotion, or rejection."
    ),
    "source_commit": "efac48daae5358599eeeb055906083e423eb5072",
    "source_path": ("experiment_runs/drift-recovery/swap-nfcorpus/campaign-completion.log"),
    "git_blob_sha1": "e5ebca641969ca20c813045f5f0898e2881272d6",
    "required_markers": [
        "Initializing local backend: all-mpnet-base-v2",
        "Local backend loaded. Vector size: 768",
    ],
    "production_path": [
        "run_drift_recovery_experiment._build_query_encoder_drift",
        "query_encoder_drift.QueryEncoderDrift.embed_queries",
        "query_encoder_drift.QueryEncoderDrift._backend",
        "embedding_backend.create_embedding_backend",
        "embedding_backend.LocalEmbeddingBackend",
    ],
}


class BuildError(ValueError):
    """Raised when the tracked tree cannot satisfy the closed v2 inventory."""


def _git_bytes(*args: str) -> bytes:
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=False,
        capture_output=True,
    )
    if completed.returncode != 0:
        detail = completed.stderr.decode("utf-8", errors="replace").strip()
        raise BuildError(f"git {' '.join(args)} failed: {detail}")
    return completed.stdout


def _git_text(*args: str) -> str:
    return _git_bytes(*args).decode("utf-8", errors="strict").strip()


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _git_blob_sha1(content: bytes) -> str:
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content).hexdigest()  # noqa: S324 - Git object ID


def _safe_repo_path(value: str) -> Path:
    pure = PurePosixPath(value)
    if not value or pure.is_absolute() or ".." in pure.parts or "\\" in value:
        raise BuildError(f"unsafe repository path: {value!r}")
    return ROOT.joinpath(*pure.parts)


def _active_identity(path: str) -> Dict[str, str]:
    content = _safe_repo_path(path).read_bytes()
    return {
        "git_blob_sha1": _git_blob_sha1(content),
        "sha256": _sha256(content),
    }


def _commit_identity(commit: str, path: str) -> Dict[str, str]:
    blob = _git_text("rev-parse", f"{commit}:{path}")
    content = _git_bytes("cat-file", "blob", blob)
    return {
        "commit": commit,
        "git_blob_sha1": blob,
        "sha256": _sha256(content),
    }


def _tracked_drift_paths() -> List[str]:
    lines = _git_text("ls-files", "experiment_runs/drift-recovery").splitlines()
    return sorted(line.replace("\\", "/") for line in lines if line)


def _load_json(path: str) -> Dict[str, Any]:
    payload = json.loads(_safe_repo_path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise BuildError(f"{path}: top-level JSON must be an object")
    return payload


def _raw_group(path: str) -> str:
    prefix = "experiment_runs/drift-recovery/"
    relative = path[len(prefix) :]
    if relative.startswith("calibrated/"):
        return "calibration"
    if relative.startswith("knob-sweep/"):
        return "knob_sweep"
    if relative.startswith("h4-compound/"):
        return "h4_compound"
    if "/" not in relative and relative.startswith("scifact_"):
        return "base_matrix"
    raise BuildError(f"unclassified drift_recovery_experiment path: {path}")


def _discover_json_paths() -> Tuple[List[str], List[str]]:
    tracked_json = [path for path in _tracked_drift_paths() if path.lower().endswith(".json")]
    raw_paths: List[str] = []
    aggregate_paths: List[str] = []
    for path in tracked_json:
        payload = _load_json(path)
        record_type = payload.get("record_type")
        if record_type == "drift_recovery_experiment":
            raw_paths.append(path)
            continue
        expected = AGGREGATE_PATHS.get(path)
        if expected is None:
            raise BuildError(f"{path}: unexpected non-raw drift-recovery JSON type {record_type!r}")
        if record_type != expected[0]:
            raise BuildError(f"{path}: record_type={record_type!r}, expected {expected[0]!r}")
        aggregate_paths.append(path)

    if len(raw_paths) != EXPECTED_TYPE_COUNTS[RAW_TYPE]:
        raise BuildError(f"raw run count={len(raw_paths)}, expected {EXPECTED_TYPE_COUNTS[RAW_TYPE]}")
    if set(aggregate_paths) != set(AGGREGATE_PATHS):
        raise BuildError("aggregate path enum does not match the tracked JSON tree")
    return sorted(raw_paths), sorted(aggregate_paths)


def _manifest_referenced_paths(path: str) -> List[str]:
    payload = _load_json(path)
    referenced = set()
    for key in MANIFEST_ROW_KEYS:
        rows = payload.get(key, [])
        if rows is None:
            continue
        if not isinstance(rows, list):
            raise BuildError(f"{path}: manifest field {key!r} must be a list")
        for row in rows:
            if not isinstance(row, dict) or not isinstance(row.get("path"), str):
                raise BuildError(f"{path}: {key!r} contains a row without path")
            normalized = row["path"].replace("\\", "/")
            _safe_repo_path(normalized)
            referenced.add(normalized)
    return sorted(referenced)


def _raw_dispositions(raw_paths: Sequence[str]) -> List[Dict[str, Any]]:
    retained_by_group: Dict[str, List[str]] = {group: [] for group in EXPECTED_GROUP_COUNTS}
    for path in raw_paths:
        retained_by_group[_raw_group(path)].append(path)

    missing_manifest_by_group = {
        "swap_scifact": ("experiment_runs/drift-recovery/swap/" "swap-campaign-manifest-2026-06.json"),
        "swap_nfcorpus": ("experiment_runs/drift-recovery/swap-nfcorpus/" "swap-campaign-manifest-2026-06.json"),
        "post_bank_scifact": (
            "experiment_runs/drift-recovery/post-bank-headtohead/" "post-bank-headtohead-manifest-2026-06.json"
        ),
        "post_bank_nfcorpus": (
            "experiment_runs/drift-recovery/post-bank-headtohead-nfcorpus/" "post-bank-headtohead-manifest-2026-06.json"
        ),
    }
    missing_by_group: Dict[str, List[str]] = {group: [] for group in EXPECTED_GROUP_COUNTS}
    for group, manifest in missing_manifest_by_group.items():
        referenced = _manifest_referenced_paths(manifest)
        present = [path for path in referenced if _safe_repo_path(path).is_file()]
        if present:
            raise BuildError(f"{manifest}: expected missing raw paths are now present: {present}")
        missing_by_group[group] = referenced

    dispositions: List[Dict[str, Any]] = []
    for group in EXPECTED_GROUP_COUNTS:
        retained = sorted(retained_by_group[group])
        missing = sorted(missing_by_group[group])
        expected_retained, expected_missing = EXPECTED_GROUP_COUNTS[group]
        if (len(retained), len(missing)) != (expected_retained, expected_missing):
            raise BuildError(
                f"{group}: raw disposition={(len(retained), len(missing))}, "
                f"expected={(expected_retained, expected_missing)}"
            )
        state = "ALL_REFERENCED_RAW_RETAINED" if not missing else "ALL_REFERENCED_RAW_MISSING_FROM_TIP"
        dispositions.append(
            {
                "id": group,
                "state": state,
                "referenced_raw_count": len(retained) + len(missing),
                "retained_raw_count": len(retained),
                "missing_from_tip_count": len(missing),
                "retained_paths": retained,
                "missing_paths": missing,
            }
        )
    return dispositions


def _base_entry(
    path: str,
    artifact_type: str,
    raw_disposition_ids: Sequence[str],
) -> Dict[str, Any]:
    identity = _active_identity(path)
    accepted = ACCEPTED_REPRODUCTION_USES if artifact_type in {RAW_TYPE, AGGREGATE_TYPE} else ACCEPTED_AUDIT_USES
    return {
        "path": path,
        "artifact_type": artifact_type,
        **identity,
        "defect_id": DEFECT_ID,
        "state": BLOCKED_STATE,
        "accepted_uses": list(accepted),
        "prohibited_uses": list(PROHIBITED_USES),
        "raw_disposition_ids": list(raw_disposition_ids),
        "supersession": dict(SUPERSESSION),
    }


def _raw_entries(raw_paths: Sequence[str]) -> List[Dict[str, Any]]:
    return [_base_entry(path, RAW_TYPE, [_raw_group(path)]) for path in raw_paths]


def _aggregate_entries(aggregate_paths: Sequence[str]) -> List[Dict[str, Any]]:
    return [_base_entry(path, AGGREGATE_TYPE, [AGGREGATE_PATHS[path][1]]) for path in aggregate_paths]


def _plot_entries() -> List[Dict[str, Any]]:
    return [_base_entry(path, PLOT_TYPE, [PLOT_GROUPS[path]]) for path in sorted(PLOT_GROUPS)]


def _prose_entries() -> List[Dict[str, Any]]:
    entries = []
    for path in sorted(PROSE_GROUPS):
        entry = _base_entry(path, PROSE_TYPE, PROSE_GROUPS[path])
        source_commit = PRE_WARNING_HEAD if path in FIRST_PASS_WARNING_DOCS else SOURCE_HEAD
        original = {
            "role": "ORIGINAL_PRE_RECONDITIONING_SOURCE",
            **_commit_identity(source_commit, path),
        }
        active = {
            "role": "RECONDITIONED_WARNING_SURFACE",
            "warning_change_id": "PR292_V2_BLAST_RADIUS_PASS",
            "git_blob_sha1": entry["git_blob_sha1"],
            "sha256": entry["sha256"],
        }
        if path in FIRST_PASS_WARNING_DOCS:
            active["first_warning_commit"] = FIRST_WARNING_COMMIT
        else:
            active["first_warning_commit"] = None
        if original["git_blob_sha1"] == active["git_blob_sha1"]:
            raise BuildError(f"{path}: reconditioned prose matches original blob")
        entry["provenance_versions"] = {
            "historical_source": original,
            "active_surface": active,
        }
        entries.append(entry)
    return entries


def _shard(
    shard_id: str,
    artifact_type: str,
    entries: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    expected_count = EXPECTED_TYPE_COUNTS[artifact_type]
    if len(entries) != expected_count:
        raise BuildError(f"{shard_id}: artifact count={len(entries)}, expected={expected_count}")
    return {
        "schema_version": SCHEMA_VERSION,
        "shard_id": shard_id,
        "artifact_type": artifact_type,
        "artifact_count": len(entries),
        "artifacts": list(entries),
    }


def _verify_preservation(
    entries_by_type: Mapping[str, Sequence[Mapping[str, Any]]],
) -> None:
    for artifact_type in (RAW_TYPE, AGGREGATE_TYPE, PLOT_TYPE):
        for entry in entries_by_type[artifact_type]:
            historical = _commit_identity(SOURCE_HEAD, str(entry["path"]))
            if historical["git_blob_sha1"] != entry["git_blob_sha1"]:
                raise BuildError(
                    f"{entry['path']}: historical JSON/PNG bytes changed from " f"preservation anchor {SOURCE_HEAD}"
                )


@lru_cache(maxsize=1)
def build_documents() -> Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]:
    """Return the exact v2 index payload and its four deterministic shards."""

    raw_paths, aggregate_paths = _discover_json_paths()
    tracked = set(_tracked_drift_paths())
    tracked_plots = {path for path in tracked if path.lower().endswith(".png")}
    if tracked_plots != set(PLOT_GROUPS):
        raise BuildError(
            "plot path enum mismatch; "
            f"missing={sorted(set(PLOT_GROUPS) - tracked_plots)}, "
            f"extra={sorted(tracked_plots - set(PLOT_GROUPS))}"
        )
    for path in PROSE_GROUPS:
        if not _safe_repo_path(path).is_file():
            raise BuildError(f"missing prose surface: {path}")

    entries_by_type = {
        RAW_TYPE: _raw_entries(raw_paths),
        AGGREGATE_TYPE: _aggregate_entries(aggregate_paths),
        PLOT_TYPE: _plot_entries(),
        PROSE_TYPE: _prose_entries(),
    }
    _verify_preservation(entries_by_type)

    shards: Dict[str, Dict[str, Any]] = {}
    shard_rows = []
    for shard_id, artifact_type, path in SHARD_SPECS:
        payload = _shard(shard_id, artifact_type, entries_by_type[artifact_type])
        shards[path] = payload
        content = _json_bytes(payload)
        shard_rows.append(
            {
                "path": path,
                "shard_id": shard_id,
                "artifact_type": artifact_type,
                "artifact_count": len(entries_by_type[artifact_type]),
                "git_blob_sha1": _git_blob_sha1(content),
                "sha256": _sha256(content),
            }
        )

    legacy_path = "artifacts/legacy-ndcg-quarantine-index-v1.json"
    legacy_identity = _active_identity(legacy_path)
    index = {
        "schema_version": SCHEMA_VERSION,
        "index_id": INDEX_ID,
        "state": "ACTIVE_QUARANTINE",
        "authored_date": "2026-07-27",
        "source_head": SOURCE_HEAD,
        "defect": {
            "id": DEFECT_ID,
            "summary": (
                "drift_recovery_metrics.ndcg_at_k forms IDCG only from retrieved "
                "top-k relevance instead of the complete positive qrels set"
            ),
            "claim_boundary": (
                "No exact nDCG value, ordering, selection, H1/H2/H3/H4/H5 "
                "verdict, promotion, rejection, or paper/release claim from this "
                "lineage is accepted before corrected regeneration."
            ),
        },
        "artifact_count": sum(EXPECTED_TYPE_COUNTS.values()),
        "artifact_counts": dict(EXPECTED_TYPE_COUNTS),
        "artifact_shard_count": len(SHARD_SPECS),
        "artifact_shards": shard_rows,
        "raw_dispositions": _raw_dispositions(raw_paths),
        "accepted_use_vocabulary": list(ACCEPTED_REPRODUCTION_USES),
        "prohibited_use_vocabulary": list(PROHIBITED_USES),
        "immutability_policy": (
            "Historical JSON and PNG artifacts remain byte-identical to "
            f"{SOURCE_HEAD}; prose keeps separate original-source and active "
            "warning-surface identities. Corrected results require new paths "
            "and an explicit old-to-new hash-linked supersession index."
        ),
        "preservation_contract": {
            "anchor_commit": SOURCE_HEAD,
            "byte_preserved_artifact_types": [
                RAW_TYPE,
                AGGREGATE_TYPE,
                PLOT_TYPE,
            ],
            "byte_preserved_artifact_count": (
                EXPECTED_TYPE_COUNTS[RAW_TYPE] + EXPECTED_TYPE_COUNTS[AGGREGATE_TYPE] + EXPECTED_TYPE_COUNTS[PLOT_TYPE]
            ),
        },
        "legacy_v1": {
            "path": legacy_path,
            "role": "INCOMPLETE_PREDECESSOR_PRESERVED_FOR_AUDIT",
            "artifact_count": 11,
            **legacy_identity,
        },
        "backend_resolution_evidence": [dict(BACKEND_EVIDENCE)],
        "tracked_debt": {
            "carried_debt_id": "CD-MLR-01",
            "deferred_scope_id": "DS-MLR-01",
        },
    }
    return index, shards


def write_documents(
    index: Mapping[str, Any],
    shards: Mapping[str, Mapping[str, Any]],
) -> None:
    for path, payload in shards.items():
        output = _safe_repo_path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(_json_bytes(payload))
    index_path = ROOT / "artifacts" / "legacy-ndcg-quarantine-index-v2.json"
    index_path.write_bytes(_json_bytes(index))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write",
        action="store_true",
        help="Write the deterministic v2 index and shards into artifacts/.",
    )
    return parser


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parser().parse_args(argv)
    try:
        index, shards = build_documents()
        if args.write:
            write_documents(index, shards)
    except (BuildError, OSError, json.JSONDecodeError) as exc:
        print(f"METRIC_LINEAGE_QUARANTINE_BUILD: FAIL: {exc}")
        return 1
    action = "WROTE" if args.write else "VERIFIED"
    print(f"METRIC_LINEAGE_QUARANTINE_BUILD: {action} " f"(artifacts={index['artifact_count']}, shards={len(shards)})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
