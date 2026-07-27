"""Hostile mutation tests for the complete legacy nDCG quarantine contract."""

from __future__ import annotations

import copy
import hashlib
import json
import unittest
from pathlib import Path
from typing import Any, Callable, Dict, Tuple

from scripts.build_metric_lineage_quarantine_v2 import (
    PROSE_TYPE,
    RAW_TYPE,
    ROOT,
    build_documents,
)
from scripts.validate_metric_lineage_quarantine import (
    ValidationError,
    _validate_control_surface_text,
    _validate_warning_text,
    validate,
    validate_payloads,
)


PayloadPair = Tuple[Dict[str, Any], Dict[str, Dict[str, Any]]]


def _json_bytes(payload: Any) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def _git_blob_sha1(content: bytes) -> str:
    header = f"blob {len(content)}\0".encode("ascii")
    return hashlib.sha1(header + content).hexdigest()  # noqa: S324 - Git ID


def _refresh_shard_binding(
    index: Dict[str, Any],
    shards: Dict[str, Dict[str, Any]],
    path: str,
) -> None:
    content = _json_bytes(shards[path])
    for row in index["artifact_shards"]:
        if row["path"] == path:
            row["artifact_count"] = shards[path]["artifact_count"]
            row["sha256"] = hashlib.sha256(content).hexdigest()
            row["git_blob_sha1"] = _git_blob_sha1(content)
            return
    raise AssertionError(f"missing shard row for {path}")


class TestMetricLineageQuarantineRepositoryGate(unittest.TestCase):
    def test_repository_contract_covers_all_113_artifacts(self) -> None:
        paths = validate()
        self.assertEqual(len(paths), 113)
        self.assertEqual(len(paths), len(set(paths)))

    def test_builder_matches_committed_index_and_shards(self) -> None:
        expected_index, expected_shards = build_documents()
        index_path = ROOT / "artifacts" / "legacy-ndcg-quarantine-index-v2.json"
        observed_index = json.loads(index_path.read_text(encoding="utf-8"))
        self.assertEqual(observed_index, expected_index)
        for relative, expected in expected_shards.items():
            observed = json.loads((ROOT / Path(relative)).read_text(encoding="utf-8"))
            self.assertEqual(observed, expected)


class TestMetricLineageQuarantineHostileMutations(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls._canonical_index, cls._canonical_shards = build_documents()

    def _payloads(self) -> PayloadPair:
        return (
            copy.deepcopy(self._canonical_index),
            copy.deepcopy(self._canonical_shards),
        )

    def _assert_rejected(
        self,
        mutate: Callable[[Dict[str, Any], Dict[str, Dict[str, Any]]], None],
    ) -> None:
        index, shards = self._payloads()
        mutate(index, shards)
        with self.assertRaises(ValidationError):
            validate_payloads(index, shards, check_repository=False)

    @staticmethod
    def _shard_path(index: Dict[str, Any], artifact_type: str) -> str:
        rows = [row["path"] for row in index["artifact_shards"] if row["artifact_type"] == artifact_type]
        if len(rows) != 1:
            raise AssertionError(f"expected one shard for {artifact_type}")
        return rows[0]

    def test_rejects_schema_version_drift(self) -> None:
        self._assert_rejected(lambda index, _shards: index.__setitem__("schema_version", "2.0"))

    def test_rejects_unknown_index_field(self) -> None:
        self._assert_rejected(lambda index, _shards: index.__setitem__("future_escape", True))

    def test_rejects_fabricated_total_count(self) -> None:
        self._assert_rejected(lambda index, _shards: index.__setitem__("artifact_count", 11))

    def test_rejects_fabricated_type_count(self) -> None:
        def mutate(index: Dict[str, Any], _shards: Dict[str, Any]) -> None:
            index["artifact_counts"][RAW_TYPE] = 2

        self._assert_rejected(mutate)

    def test_rejects_coordinated_missing_artifact_and_recount(self) -> None:
        def mutate(
            index: Dict[str, Any],
            shards: Dict[str, Dict[str, Any]],
        ) -> None:
            path = self._shard_path(index, RAW_TYPE)
            shards[path]["artifacts"].pop()
            shards[path]["artifact_count"] -= 1
            index["artifact_count"] -= 1
            index["artifact_counts"][RAW_TYPE] -= 1
            _refresh_shard_binding(index, shards, path)

        self._assert_rejected(mutate)

    def test_rejects_coordinated_extra_artifact_and_recount(self) -> None:
        def mutate(
            index: Dict[str, Any],
            shards: Dict[str, Dict[str, Any]],
        ) -> None:
            path = self._shard_path(index, RAW_TYPE)
            extra = copy.deepcopy(shards[path]["artifacts"][0])
            extra["path"] = "experiment_runs/drift-recovery/fabricated.json"
            shards[path]["artifacts"].append(extra)
            shards[path]["artifact_count"] += 1
            index["artifact_count"] += 1
            index["artifact_counts"][RAW_TYPE] += 1
            _refresh_shard_binding(index, shards, path)

        self._assert_rejected(mutate)

    def test_rejects_artifact_type_lie(self) -> None:
        def mutate(index: Dict[str, Any], shards: Dict[str, Any]) -> None:
            path = self._shard_path(index, RAW_TYPE)
            shards[path]["artifacts"][0]["artifact_type"] = "historical_raw"

        self._assert_rejected(mutate)

    def test_rejects_artifact_state_lie(self) -> None:
        def mutate(index: Dict[str, Any], shards: Dict[str, Any]) -> None:
            path = self._shard_path(index, RAW_TYPE)
            shards[path]["artifacts"][0]["state"] = "VALIDATED"

        self._assert_rejected(mutate)

    def test_rejects_artifact_path_substitution(self) -> None:
        def mutate(index: Dict[str, Any], shards: Dict[str, Any]) -> None:
            path = self._shard_path(index, RAW_TYPE)
            shards[path]["artifacts"][0]["path"] = "fabricated/result.json"

        self._assert_rejected(mutate)

    def test_rejects_hash_substitution(self) -> None:
        def mutate(index: Dict[str, Any], shards: Dict[str, Any]) -> None:
            path = self._shard_path(index, RAW_TYPE)
            shards[path]["artifacts"][0]["sha256"] = "0" * 64

        self._assert_rejected(mutate)

    def test_rejects_accepted_use_expansion(self) -> None:
        def mutate(index: Dict[str, Any], shards: Dict[str, Any]) -> None:
            path = self._shard_path(index, RAW_TYPE)
            shards[path]["artifacts"][0]["accepted_uses"].append("scientific_performance_claim")

        self._assert_rejected(mutate)

    def test_rejects_prohibited_use_removal(self) -> None:
        def mutate(index: Dict[str, Any], shards: Dict[str, Any]) -> None:
            path = self._shard_path(index, RAW_TYPE)
            shards[path]["artifacts"][0]["prohibited_uses"].pop()

        self._assert_rejected(mutate)

    def test_rejects_missing_raw_state_lie(self) -> None:
        def mutate(index: Dict[str, Any], _shards: Dict[str, Any]) -> None:
            row = next(item for item in index["raw_dispositions"] if item["id"] == "swap_scifact")
            row["state"] = "ALL_REFERENCED_RAW_RETAINED"

        self._assert_rejected(mutate)

    def test_rejects_missing_raw_count_lie(self) -> None:
        def mutate(index: Dict[str, Any], _shards: Dict[str, Any]) -> None:
            row = next(item for item in index["raw_dispositions"] if item["id"] == "post_bank_nfcorpus")
            row["missing_from_tip_count"] = 0

        self._assert_rejected(mutate)

    def test_rejects_missing_raw_path_removal(self) -> None:
        def mutate(index: Dict[str, Any], _shards: Dict[str, Any]) -> None:
            row = next(item for item in index["raw_dispositions"] if item["id"] == "swap_nfcorpus")
            row["missing_paths"].pop()
            row["missing_from_tip_count"] -= 1
            row["referenced_raw_count"] -= 1

        self._assert_rejected(mutate)

    def test_rejects_original_source_role_lie(self) -> None:
        def mutate(index: Dict[str, Any], shards: Dict[str, Any]) -> None:
            path = self._shard_path(index, PROSE_TYPE)
            versions = shards[path]["artifacts"][0]["provenance_versions"]
            versions["historical_source"]["role"] = "ACTIVE_SOURCE"

        self._assert_rejected(mutate)

    def test_rejects_original_source_blob_substitution(self) -> None:
        def mutate(index: Dict[str, Any], shards: Dict[str, Any]) -> None:
            path = self._shard_path(index, PROSE_TYPE)
            entry = shards[path]["artifacts"][0]
            entry["provenance_versions"]["historical_source"]["git_blob_sha1"] = entry["git_blob_sha1"]

        self._assert_rejected(mutate)

    def test_rejects_overbroad_backend_claim(self) -> None:
        def mutate(index: Dict[str, Any], _shards: Dict[str, Any]) -> None:
            index["backend_resolution_evidence"][0]["claim"] = "The H2 campaign and its metrics are valid."

        self._assert_rejected(mutate)

    def test_rejects_weakened_backend_scope_limit(self) -> None:
        def mutate(index: Dict[str, Any], _shards: Dict[str, Any]) -> None:
            index["backend_resolution_evidence"][0]["scope_limit"] = "This proves backend and campaign correctness."

        self._assert_rejected(mutate)

    def test_rejects_overbroad_backend_production_path(self) -> None:
        def mutate(index: Dict[str, Any], _shards: Dict[str, Any]) -> None:
            index["backend_resolution_evidence"][0]["production_path"].append(
                "run_drift_recovery_swap_campaign.H2_RESULT_CONFIRMED"
            )

        self._assert_rejected(mutate)

    def test_rejects_weak_backend_marker_evidence(self) -> None:
        def mutate(index: Dict[str, Any], _shards: Dict[str, Any]) -> None:
            index["backend_resolution_evidence"][0]["required_markers"] = ["Initializing local backend"]

        self._assert_rejected(mutate)

    def test_rejects_preservation_count_lie(self) -> None:
        def mutate(index: Dict[str, Any], _shards: Dict[str, Any]) -> None:
            index["preservation_contract"]["byte_preserved_artifact_count"] = 11

        self._assert_rejected(mutate)

    def test_rejects_debt_id_substitution(self) -> None:
        def mutate(index: Dict[str, Any], _shards: Dict[str, Any]) -> None:
            index["tracked_debt"]["carried_debt_id"] = "CLOSED"

        self._assert_rejected(mutate)

    def test_rejects_extra_shard(self) -> None:
        def mutate(
            _index: Dict[str, Any],
            shards: Dict[str, Dict[str, Any]],
        ) -> None:
            shards["artifacts/fabricated.json"] = {
                "schema_version": "2.0.0",
                "shard_id": "fabricated",
                "artifact_type": RAW_TYPE,
                "artifact_count": 0,
                "artifacts": [],
            }

        self._assert_rejected(mutate)

    def test_rejects_weak_prose_warning(self) -> None:
        with self.assertRaises(ValidationError):
            _validate_warning_text(
                "docs/fabricated.md",
                "# Historical report\n\nEverything below is fully validated.\n",
            )

    def test_rejects_weakened_control_surface(self) -> None:
        with self.assertRaises(ValidationError):
            _validate_control_surface_text(
                "docs/next-session.md",
                "CD-MLR-01 and DS-MLR-01 exist, but the old evidence is valid.",
            )


if __name__ == "__main__":
    unittest.main()
