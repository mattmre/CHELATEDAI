"""Tests for the artifact-first BCC-1 bridge-choice harness."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Optional
from unittest.mock import patch

import numpy as np

import bcc1_experiment as bcc1_module
from bcc1_experiment import (
    BCC1ValidationError,
    MANIFEST_SCHEMA,
    PACK_SCHEMA,
    analyze_bcc1 as _public_analyze_bcc1,
    canonical_json_bytes,
    decode_json_object,
    hierarchical_paired_bootstrap_mean_ci,
    paired_bootstrap_mean_ci,
    query_ids_sha256,
    run_bcc1_files,
    sha256_bytes,
)
from scripts.run_bcc1 import main as cli_main


def analyze_bcc1(pack: dict, manifest: dict) -> dict:
    """Exercise evaluator internals without weakening the public boundary."""

    if pack.get("evidence_mode") != "CONFIRMATORY":
        return _public_analyze_bcc1(pack, manifest)
    return bcc1_module._analyze_combined_blocked_protocol_diagnostic(pack, manifest)


def _metric_contract(pack_sha256: str, *, label: str) -> dict:
    qrels_sha256 = sha256_bytes(label.encode("utf-8"))
    binding_sha256 = sha256_bytes(
        canonical_json_bytes(
            {
                "schema_version": bcc1_module.METRIC_BINDING_SCHEMA,
                "pack_sha256": pack_sha256,
                "qrels_sha256": qrels_sha256,
            }
        )
    )
    return {
        "metric_name": "ndcg_at_10",
        "implementation_sha256": sha256_bytes((label + ":metric-implementation").encode("utf-8")),
        "gain": "binary_positive_qrel",
        "idcg_population": "all_positive_qrels_for_query",
        "cutoff": 10,
        "query_inclusion": "queries_with_at_least_one_positive_qrel",
        "qrels_sha256": qrels_sha256,
        "pack_qrels_binding_sha256": binding_sha256,
        "ranking_tie_break": "score_desc_document_id_asc",
    }


def _bind_manifest_to_pack(pack: dict, manifest: dict) -> None:
    pack_sha256 = sha256_bytes(canonical_json_bytes(pack))
    manifest["pack_sha256"] = pack_sha256
    manifest["metric_contract"]["pack_qrels_binding_sha256"] = sha256_bytes(
        canonical_json_bytes(
            {
                "schema_version": bcc1_module.METRIC_BINDING_SCHEMA,
                "pack_sha256": pack_sha256,
                "qrels_sha256": manifest["metric_contract"]["qrels_sha256"],
            }
        )
    )


def _row(
    query_id: str,
    block_id: str,
    feature: float,
    reverse: float,
    forward: float,
    *,
    native_old: float = 0.50,
    native_new: float = 0.80,
    fusion_scores: Optional[Dict[str, float]] = None,
    independence_group_id: Optional[str] = None,
) -> dict:
    row = {
        "query_id": query_id,
        "independence_group_id": independence_group_id or query_id,
        "block_id": block_id,
        "scores": {
            "reverse": reverse,
            "forward": forward,
            "native_old": native_old,
            "native_new": native_new,
        },
        "features": {"query_drift": feature},
    }
    if fusion_scores is not None:
        row["fusion_scores"] = fusion_scores
    return row


def _pack_and_manifest(*, soft_fusion: bool = False) -> tuple[dict, dict]:
    rows = [
        _row("s1", "select-a", -3.0, 0.20, 0.80),
        _row("s2", "select-a", -2.0, 0.25, 0.75),
        _row("s3", "select-a", -1.0, 0.30, 0.70),
        _row("s4", "select-a", 1.0, 0.85, 0.35),
        _row("s5", "select-a", 2.0, 0.90, 0.30),
        _row("s6", "select-a", 3.0, 0.95, 0.25),
        _row("r1", "report-a", -2.5, 0.20, 0.80),
        _row("r2", "report-a", -1.5, 0.30, 0.70),
        _row("r3", "report-a", 1.5, 0.80, 0.30),
        _row("r4", "report-a", 2.5, 0.90, 0.20),
    ]
    fusion_contract = None
    if soft_fusion:
        fusion_contract = {
            "source_kind": "precomputed_retrieval_score_fusion",
            "qrels_free_at_inference": True,
            "normalization": "per-query z-score before weighted sum",
            "candidates": [
                {"id": "alpha-025", "alpha": 0.25},
                {"id": "alpha-075", "alpha": 0.75},
            ],
        }
        for index, row in enumerate(rows):
            if index < 6:
                row["fusion_scores"] = {
                    "alpha-025": 0.60,
                    "alpha-075": 0.70,
                }
            else:
                row["fusion_scores"] = {
                    "alpha-025": 0.40,
                    "alpha-075": 0.72,
                }
    pack = {
        "schema_version": PACK_SCHEMA,
        "pack_id": "bcc1-fixture",
        "evidence_mode": "CONFIRMATORY",
        "sampled": False,
        "metric": {
            "name": "ndcg_at_10",
            "higher_is_better": True,
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "rows": rows,
    }
    pack_raw = canonical_json_bytes(pack)
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "pack_id": "bcc1-fixture",
        "evidence_mode": "CONFIRMATORY",
        "sampled": False,
        "pack_sha256": sha256_bytes(pack_raw),
        "metric_contract": _metric_contract(
            sha256_bytes(pack_raw),
            label="confirmatory-fixture-qrels",
        ),
        "reverse_route_role_validation": (bcc1_module.VALIDATED_QUERY_ROLE_REVERSE_BRIDGE),
        "features": [
            {
                "name": "query_drift",
                "qrels_free": True,
                "inference_available": True,
                "source": "norm difference between same-text query embeddings",
                "implementation_sha256": "1" * 64,
                "provenance_sha256": "2" * 64,
                "availability_stage": "PRE_SEARCH",
                "execution_cost_class": "zero_search",
            }
        ],
        "blocks": [
            {
                "block_id": "select-a",
                "dataset_family_id": "select-dataset",
                "transition_family_id": "select-transition",
                "role": "SELECT",
                "query_ids_sha256": query_ids_sha256(row["query_id"] for row in rows if row["block_id"] == "select-a"),
                "consumed": False,
            },
            {
                "block_id": "report-a",
                "dataset_family_id": "report-dataset",
                "transition_family_id": "report-transition",
                "role": "REPORT",
                "query_ids_sha256": query_ids_sha256(row["query_id"] for row in rows if row["block_id"] == "report-a"),
                "consumed": False,
            },
        ],
        "soft_fusion": fusion_contract,
        "confirmatory_gates": {
            "primary_method": "regularized_logistic",
            "minimum_worthwhile_effect": 0.005,
            "target_power": 0.80,
            "one_sided_alpha": 0.025,
            "minimum_report_queries": 2400,
            "minimum_report_blocks": 12,
            "native_new_noninferiority_margin": 0.02,
            "worst_block_noninferiority_margin": 0.02,
            "minimum_overall_coverage": 0.10,
            "minimum_block_coverage": 0.05,
            "power_replicates": 10000,
            "power_seed": 1730,
            "power_variance_inflation": 1.25,
        },
    }
    return pack, manifest


def _method_dev_pack_and_manifest() -> tuple[dict, dict]:
    rows = [
        _row("d1", "dev-a", -2.0, 0.70, 0.75),
        _row("d2", "dev-a", -1.0, 0.80, 0.72),
        _row("d3", "dev-a", 1.0, 0.60, 0.70),
        _row(
            "d4",
            "dev-b",
            2.0,
            0.90,
            0.85,
            independence_group_id="d1",
        ),
        _row(
            "d5",
            "dev-b",
            3.0,
            0.60,
            0.65,
            independence_group_id="d2",
        ),
        _row(
            "d6",
            "dev-b",
            4.0,
            0.75,
            0.75,
            independence_group_id="d3",
        ),
    ]
    for index, row in enumerate(rows):
        row["scores"]["mismatch"] = 0.20 + index * 0.01
        row["scores"].pop("native_old")
    pack = {
        "schema_version": PACK_SCHEMA,
        "pack_id": "bcc1-method-dev-fixture",
        "evidence_mode": "METHOD_DEV",
        "sampled": True,
        "metric": {
            "name": "ndcg_at_10",
            "higher_is_better": True,
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "rows": rows,
    }
    manifest = {
        "schema_version": MANIFEST_SCHEMA,
        "pack_id": pack["pack_id"],
        "evidence_mode": "METHOD_DEV",
        "sampled": True,
        "pack_sha256": sha256_bytes(canonical_json_bytes(pack)),
        "metric_contract": _metric_contract(
            sha256_bytes(canonical_json_bytes(pack)),
            label="method-dev-fixture-qrels",
        ),
        "reverse_route_role_validation": (bcc1_module.UNVALIDATED_DOCUMENT_ROLE_REVERSE_PROXY),
        "features": [
            {
                "name": "query_drift",
                "qrels_free": True,
                "inference_available": True,
                "source": "same-text embedding cycle residual",
                "implementation_sha256": "1" * 64,
                "provenance_sha256": "2" * 64,
                "availability_stage": "PRE_SEARCH",
                "execution_cost_class": "zero_search",
            }
        ],
        "blocks": [
            {
                "block_id": block_id,
                "dataset_family_id": "dev-dataset",
                "transition_family_id": block_id,
                "role": "METHOD_DEV",
                "query_ids_sha256": query_ids_sha256(row["query_id"] for row in rows if row["block_id"] == block_id),
                "consumed": False,
            }
            for block_id in ("dev-a", "dev-b")
        ],
        "soft_fusion": None,
    }
    return pack, manifest


def _confirmatory_universe_inputs(
    *,
    dataset_count: int = 8,
    transition_count: int = 6,
) -> tuple[dict, list[dict], list[dict]]:
    dataset_ids = ["dataset-{:02d}".format(index) for index in range(dataset_count)]
    transition_ids = ["transition-{:02d}".format(index) for index in range(transition_count)]
    dataset_roles = bcc1_module._mechanical_family_roles(
        dataset_ids,
        family_kind="dataset",
    )
    transition_roles = bcc1_module._mechanical_family_roles(
        transition_ids,
        family_kind="transition",
    )
    blocks = {}
    rows_by_role = {"SELECT": [], "REPORT": []}
    for role in ("SELECT", "REPORT"):
        role_datasets = sorted(family_id for family_id, assigned_role in dataset_roles.items() if assigned_role == role)
        role_transitions = sorted(
            family_id for family_id, assigned_role in transition_roles.items() if assigned_role == role
        )
        for dataset_id in role_datasets:
            group_ids = ["{}-group-{:03d}".format(dataset_id, index) for index in range(200)]
            for transition_id in role_transitions:
                block_id = "{}-{}-{}".format(
                    role.lower(),
                    dataset_id,
                    transition_id,
                )
                blocks[block_id] = {
                    "block_id": block_id,
                    "dataset_family_id": dataset_id,
                    "transition_family_id": transition_id,
                    "role": role,
                }
                rows_by_role[role].extend(
                    {
                        "block_id": block_id,
                        "query_id": "{}-{}-query".format(
                            group_id,
                            transition_id,
                        ),
                        "independence_group_id": group_id,
                    }
                    for group_id in group_ids
                )
    return blocks, rows_by_role["SELECT"], rows_by_role["REPORT"]


class TestValidation(unittest.TestCase):
    def test_valid_pack_is_deterministic_and_runs_supervised_baselines(self) -> None:
        pack, manifest = _pack_and_manifest()
        first = analyze_bcc1(pack, manifest)
        second = analyze_bcc1(copy.deepcopy(pack), copy.deepcopy(manifest))
        self.assertEqual(canonical_json_bytes(first), canonical_json_bytes(second))
        self.assertEqual(first["best_fixed"]["selected_direction"], "reverse")
        self.assertEqual(first["methods"]["regularized_logistic"]["status"], "evaluated")
        self.assertEqual(first["methods"]["knn"]["status"], "evaluated")
        self.assertEqual(
            first["methods"]["soft_score_fusion"]["status"],
            "not_run",
        )
        self.assertEqual(
            first["methods"]["knn"]["abstention_count"],
            first["data_contract"]["report_query_count"],
        )
        self.assertGreater(
            first["direction_oracle"]["descriptive_ex_post_choice_regret_upper_bound"],
            0.0,
        )
        self.assertFalse(first["evidence_classification"]["promotion_eligible"])
        self.assertEqual(first["promotion_gate"]["decision"], "BLOCKED")
        self.assertFalse(first["promotion_gate"]["checks"]["mwee_power_adequacy"])
        self.assertIn("mwee_power_adequacy", first["promotion_gate"]["failed_checks"])
        for check in (
            "separated_decision_outcome_boundary",
            "mechanical_universe_role_recomputation",
            "independent_feature_generation_attestation",
            "abi_operational_conformance_evidence",
            "canonical_replay_evidence",
        ):
            self.assertFalse(first["promotion_gate"]["checks"][check])
            self.assertIn(check, first["promotion_gate"]["failed_checks"])
        self.assertEqual(first["promotion_gate"]["protocol_status"], "BLOCKED_PROTOCOL")
        power = first["promotion_gate"]["power_and_precision"]
        power_scheme = first["promotion_gate"]["select_structural_transfer"]["policies"]["regularized_logistic"][
            "leave_transition_family_out"
        ]
        self.assertEqual(
            power["oof_residual_audit_sha256"],
            power_scheme["oof_residual_audit_sha256"],
        )
        upstream = first["promotion_gate"]["required_upstream_evidence"]
        self.assertEqual(upstream["status"], "unverified")
        self.assertTrue(upstream["promotion_gate_forced_blocked"])
        for required in (
            "representation_manifest_sha256",
            "external_anchor_validation_report_sha256",
            "environment_lock_sha256",
            "frozen_policy_sha256",
            "frozen_report_decisions_sha256",
            "resolver_adversarial_conformance_report_sha256",
        ):
            self.assertIn(required, upstream["artifacts"])

    def test_pack_hash_mismatch_fails(self) -> None:
        pack, manifest = _pack_and_manifest()
        pack["rows"][0]["scores"]["reverse"] = 0.21
        with self.assertRaisesRegex(BCC1ValidationError, "SHA-256 mismatch"):
            analyze_bcc1(pack, manifest)

    def test_metric_contract_binds_exact_lineage_and_pack_qrels_pair(self) -> None:
        pack, manifest = _pack_and_manifest()
        report = analyze_bcc1(pack, manifest)
        self.assertEqual(
            report["data_contract"]["metric_contract"],
            manifest["metric_contract"],
        )
        invalid_values = {
            "gain": "graded",
            "idcg_population": "retrieved_only",
            "cutoff": 20,
            "query_inclusion": "all_queries",
            "ranking_tie_break": "backend_default",
        }
        for field, value in invalid_values.items():
            with self.subTest(field=field):
                changed = copy.deepcopy(manifest)
                changed["metric_contract"][field] = value
                with self.assertRaisesRegex(
                    BCC1ValidationError,
                    "manifest.metric_contract.{}".format(field),
                ):
                    analyze_bcc1(pack, changed)
        changed = copy.deepcopy(manifest)
        changed["metric_contract"]["pack_qrels_binding_sha256"] = "0" * 64
        with self.assertRaisesRegex(BCC1ValidationError, "does not bind"):
            analyze_bcc1(pack, changed)

    def test_confirmatory_rejects_document_role_reverse_proxy(self) -> None:
        pack, manifest = _pack_and_manifest()
        manifest["reverse_route_role_validation"] = bcc1_module.UNVALIDATED_DOCUMENT_ROLE_REVERSE_PROXY
        with self.assertRaisesRegex(
            BCC1ValidationError,
            "validated query-role reverse bridge",
        ):
            analyze_bcc1(pack, manifest)

    def test_query_cannot_appear_in_select_and_report(self) -> None:
        pack, manifest = _pack_and_manifest()
        pack["rows"][-1]["query_id"] = "s1"
        _bind_manifest_to_pack(pack, manifest)
        with self.assertRaisesRegex(BCC1ValidationError, "more than one row/block"):
            analyze_bcc1(pack, manifest)

    def test_block_membership_hash_mismatch_fails(self) -> None:
        pack, manifest = _pack_and_manifest()
        manifest["blocks"][1]["query_ids_sha256"] = "0" * 64
        with self.assertRaisesRegex(BCC1ValidationError, "membership hash mismatch"):
            analyze_bcc1(pack, manifest)

    def test_consumed_report_fails_before_analysis(self) -> None:
        pack, manifest = _pack_and_manifest()
        manifest["blocks"][1]["consumed"] = True
        for field in (
            "consumption_report_sha256",
            "consumption_pack_sha256",
            "consumption_manifest_sha256",
            "consumption_output_identity_sha256",
            "consumption_transaction_id",
        ):
            manifest["blocks"][1][field] = "1" * 64
        with self.assertRaisesRegex(BCC1ValidationError, "already consumed"):
            analyze_bcc1(pack, manifest)

    def test_qrels_or_route_label_features_fail(self) -> None:
        for feature_name in ("qrels_count", "route_label", "ndcg_hint"):
            with self.subTest(feature_name=feature_name):
                pack, manifest = _pack_and_manifest()
                manifest["features"][0]["name"] = feature_name
                for row in pack["rows"]:
                    value = row["features"].pop("query_drift")
                    row["features"][feature_name] = value
                _bind_manifest_to_pack(pack, manifest)
                with self.assertRaisesRegex(BCC1ValidationError, "leakage token"):
                    analyze_bcc1(pack, manifest)

    def test_feature_provenance_must_be_qrels_free_and_inference_available(self) -> None:
        for field in ("qrels_free", "inference_available"):
            with self.subTest(field=field):
                pack, manifest = _pack_and_manifest()
                manifest["features"][0][field] = False
                with self.assertRaisesRegex(BCC1ValidationError, field):
                    analyze_bcc1(pack, manifest)

    def test_feature_contract_requires_bound_code_and_provenance_digests(self) -> None:
        for field in ("implementation_sha256", "provenance_sha256"):
            with self.subTest(field=field):
                pack, manifest = _pack_and_manifest()
                del manifest["features"][0][field]
                with self.assertRaisesRegex(BCC1ValidationError, field):
                    analyze_bcc1(pack, manifest)

    def test_confirmatory_policy_rejects_dual_read_features(self) -> None:
        pack, manifest = _pack_and_manifest()
        manifest["features"][0]["availability_stage"] = "DUAL_READ_METHOD_DEV"
        manifest["features"][0]["execution_cost_class"] = "dual_search_probe"
        with self.assertRaisesRegex(BCC1ValidationError, "forbids DUAL_READ"):
            analyze_bcc1(pack, manifest)

    def test_independence_group_cannot_leak_across_select_and_report(self) -> None:
        pack, manifest = _pack_and_manifest()
        pack["rows"][-1]["independence_group_id"] = pack["rows"][0]["independence_group_id"]
        _bind_manifest_to_pack(pack, manifest)
        with self.assertRaisesRegex(BCC1ValidationError, "spans dataset families|leaks across"):
            analyze_bcc1(pack, manifest)

    def test_numerically_renamed_route_label_fails(self) -> None:
        pack, manifest = _pack_and_manifest()
        for row in pack["rows"]:
            row["features"]["query_drift"] = 1.0 if row["scores"]["reverse"] > row["scores"]["forward"] else 0.0
        _bind_manifest_to_pack(pack, manifest)
        with self.assertRaisesRegex(BCC1ValidationError, "route label"):
            analyze_bcc1(pack, manifest)

    def test_undeclared_row_field_fails(self) -> None:
        pack, manifest = _pack_and_manifest()
        pack["rows"][0]["winner"] = "forward"
        _bind_manifest_to_pack(pack, manifest)
        with self.assertRaisesRegex(BCC1ValidationError, "undeclared"):
            analyze_bcc1(pack, manifest)

    def test_optional_native_old_can_be_absent_in_confirmatory_pack(self) -> None:
        pack, manifest = _pack_and_manifest()
        for row in pack["rows"]:
            row["scores"].pop("native_old")
        _bind_manifest_to_pack(pack, manifest)
        report = analyze_bcc1(pack, manifest)
        self.assertEqual(
            report["native_references"]["native_old_status"],
            "not_provided",
        )
        self.assertIsNone(report["native_references"]["native_old_report_mean"])

    def test_sampled_pack_cannot_claim_confirmatory_report(self) -> None:
        pack, manifest = _pack_and_manifest()
        pack["sampled"] = True
        manifest["sampled"] = True
        _bind_manifest_to_pack(pack, manifest)
        with self.assertRaisesRegex(BCC1ValidationError, "sampled packs"):
            analyze_bcc1(pack, manifest)

    def test_confirmatory_pack_requires_exact_preregistered_gates(self) -> None:
        pack, manifest = _pack_and_manifest()
        del manifest["confirmatory_gates"]
        with self.assertRaisesRegex(BCC1ValidationError, "confirmatory_gates"):
            analyze_bcc1(pack, manifest)

    def test_method_dev_pack_cannot_declare_confirmatory_gates(self) -> None:
        pack, manifest = _method_dev_pack_and_manifest()
        manifest["confirmatory_gates"] = {
            "primary_method": "regularized_logistic",
        }
        with self.assertRaisesRegex(BCC1ValidationError, "cannot declare"):
            analyze_bcc1(pack, manifest)

    def test_method_dev_pack_cannot_contain_report_block(self) -> None:
        pack, manifest = _method_dev_pack_and_manifest()
        manifest["blocks"][0]["role"] = "REPORT"
        with self.assertRaisesRegex(BCC1ValidationError, "REPORT block"):
            analyze_bcc1(pack, manifest)

    def test_method_dev_can_reuse_query_ids_across_condition_blocks(self) -> None:
        pack, manifest = _method_dev_pack_and_manifest()
        pack["rows"][3]["query_id"] = "d1"
        manifest["blocks"][1]["query_ids_sha256"] = query_ids_sha256(
            row["query_id"] for row in pack["rows"] if row["block_id"] == "dev-b"
        )
        _bind_manifest_to_pack(pack, manifest)
        report = analyze_bcc1(pack, manifest)
        self.assertEqual(
            report["method_dev_diagnostic"]["per_block"][1]["query_count"],
            3,
        )

    def test_duplicate_independence_group_within_structural_cell_fails(self) -> None:
        pack, manifest = _method_dev_pack_and_manifest()
        pack["rows"][1]["independence_group_id"] = pack["rows"][0]["independence_group_id"]
        _bind_manifest_to_pack(pack, manifest)
        with self.assertRaisesRegex(
            BCC1ValidationError,
            "independence_group_id .* occurs more than once in structural cell",
        ):
            analyze_bcc1(pack, manifest)

    def test_public_validator_has_no_confirmatory_universe_bypass(self) -> None:
        pack, manifest = _pack_and_manifest()
        with self.assertRaisesRegex(
            BCC1ValidationError,
            "at least 8 distinct dataset families",
        ):
            bcc1_module.validate_pack_and_manifest(pack, manifest)
        with self.assertRaisesRegex(TypeError, "unexpected keyword argument"):
            bcc1_module.validate_pack_and_manifest(
                pack,
                manifest,
                enforce_confirmatory_universe=False,
            )

    def test_duplicate_structural_cell_fails(self) -> None:
        pack, manifest = _method_dev_pack_and_manifest()
        duplicate = copy.deepcopy(manifest["blocks"][0])
        duplicate["block_id"] = "dev-a-duplicate"
        manifest["blocks"].append(duplicate)
        with self.assertRaisesRegex(
            BCC1ValidationError,
            "structural cell .* is declared by multiple block IDs",
        ):
            analyze_bcc1(pack, manifest)

    def test_full_mechanical_confirmatory_universe_is_accepted(self) -> None:
        blocks, select_rows, report_rows = _confirmatory_universe_inputs()
        bcc1_module._validate_confirmatory_universe(
            blocks=blocks,
            select_rows=select_rows,
            report_rows=report_rows,
        )

    def test_confirmatory_universe_rejects_axis_overlap(self) -> None:
        blocks, select_rows, report_rows = _confirmatory_universe_inputs()
        select_dataset = next(block["dataset_family_id"] for block in blocks.values() if block["role"] == "SELECT")
        report_block = next(block for block in blocks.values() if block["role"] == "REPORT")
        report_block["dataset_family_id"] = select_dataset
        with self.assertRaisesRegex(
            BCC1ValidationError,
            "SELECT and REPORT dataset-family axes overlap",
        ):
            bcc1_module._validate_confirmatory_universe(
                blocks=blocks,
                select_rows=select_rows,
                report_rows=report_rows,
            )

    def test_confirmatory_universe_rejects_missing_cross_product_cell(self) -> None:
        blocks, select_rows, report_rows = _confirmatory_universe_inputs()
        removed_block_id = next(block_id for block_id, block in blocks.items() if block["role"] == "SELECT")
        del blocks[removed_block_id]
        select_rows = [row for row in select_rows if row["block_id"] != removed_block_id]
        with self.assertRaisesRegex(
            BCC1ValidationError,
            "SELECT cells are not the complete mechanical cross-product",
        ):
            bcc1_module._validate_confirmatory_universe(
                blocks=blocks,
                select_rows=select_rows,
                report_rows=report_rows,
            )

    def test_confirmatory_universe_rejects_undersized_cell(self) -> None:
        blocks, select_rows, report_rows = _confirmatory_universe_inputs()
        target_block_id = select_rows[0]["block_id"]
        removed = False
        retained_rows = []
        for row in select_rows:
            if not removed and row["block_id"] == target_block_id:
                removed = True
                continue
            retained_rows.append(row)
        self.assertTrue(removed)
        with self.assertRaisesRegex(
            BCC1ValidationError,
            "SELECT requires at least 200 query rows per cell",
        ):
            bcc1_module._validate_confirmatory_universe(
                blocks=blocks,
                select_rows=retained_rows,
                report_rows=report_rows,
            )

    def test_confirmatory_universe_requires_repeated_group_membership(self) -> None:
        blocks, select_rows, report_rows = _confirmatory_universe_inputs()
        first_block_id = select_rows[0]["block_id"]
        first_block = blocks[first_block_id]
        sibling_block_id = next(
            block_id
            for block_id, block in blocks.items()
            if block["role"] == "SELECT"
            and block["dataset_family_id"] == first_block["dataset_family_id"]
            and block_id != first_block_id
        )
        sibling_row = next(row for row in select_rows if row["block_id"] == sibling_block_id)
        sibling_row["independence_group_id"] = first_block["dataset_family_id"] + "-replacement-group"
        with self.assertRaisesRegex(
            BCC1ValidationError,
            "must expose identical independence-group membership across transitions",
        ):
            bcc1_module._validate_confirmatory_universe(
                blocks=blocks,
                select_rows=select_rows,
                report_rows=report_rows,
            )

    def test_confirmatory_universe_enforces_mechanical_sha_parity(self) -> None:
        blocks, select_rows, report_rows = _confirmatory_universe_inputs()
        select_dataset = next(block["dataset_family_id"] for block in blocks.values() if block["role"] == "SELECT")
        report_dataset = next(block["dataset_family_id"] for block in blocks.values() if block["role"] == "REPORT")
        for block in blocks.values():
            if block["dataset_family_id"] == select_dataset:
                block["dataset_family_id"] = report_dataset
            elif block["dataset_family_id"] == report_dataset:
                block["dataset_family_id"] = select_dataset
        with self.assertRaisesRegex(
            BCC1ValidationError,
            "dataset-family role allocation differs from mechanical SHA parity",
        ):
            bcc1_module._validate_confirmatory_universe(
                blocks=blocks,
                select_rows=select_rows,
                report_rows=report_rows,
            )

    def test_duplicate_json_keys_are_rejected(self) -> None:
        with self.assertRaisesRegex(BCC1ValidationError, "duplicate JSON key"):
            decode_json_object(b'{"pack_id":"one","pack_id":"two"}', artifact_name="pack")


class TestEvaluationBoundary(unittest.TestCase):
    def test_knn_distance_ties_use_canonical_query_id(self) -> None:
        model = bcc1_module._KNNModel(
            mean=np.asarray([0.0]),
            scale=np.asarray([1.0]),
            x_train=np.zeros((6, 1), dtype=np.float64),
            y_train=np.asarray([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
            query_ids=("z", "a", "b", "c", "d", "e"),
        )
        probabilities = bcc1_module._predict_knn(
            model,
            np.asarray([[0.0]], dtype=np.float64),
        )
        self.assertEqual(probabilities.tolist(), [0.0])

    def test_confirmatory_in_memory_analysis_is_forbidden(self) -> None:
        pack, manifest = _pack_and_manifest()
        with self.assertRaisesRegex(BCC1ValidationError, "cannot be evaluated in memory"):
            _public_analyze_bcc1(pack, manifest)

    def test_method_dev_is_per_block_and_explicitly_non_promotional(self) -> None:
        pack, manifest = _method_dev_pack_and_manifest()
        report = analyze_bcc1(pack, manifest)
        classification = report["evidence_classification"]
        self.assertEqual(classification["evidence_mode"], "METHOD_DEV")
        self.assertFalse(classification["promotion_eligible"])
        self.assertFalse(classification["confirmatory_report_evaluated"])
        self.assertEqual(
            report["protocol"]["supervised_policy_fitting"],
            "exploratory_logistic_and_knn_diagnostics_allowed_but_non_promotional",
        )
        reverse_role = report["data_contract"]["reverse_route_role_contract"]
        self.assertEqual(
            reverse_role["validation_state"],
            bcc1_module.UNVALIDATED_DOCUMENT_ROLE_REVERSE_PROXY,
        )
        self.assertFalse(reverse_role["confirmatory_eligible"])
        self.assertEqual(
            reverse_role["independent_cut"],
            "CUT_UNVALIDATED_DOCUMENT_ROLE_REVERSE_PROXY",
        )
        diagnostic = report["method_dev_diagnostic"]
        self.assertEqual(
            diagnostic["classification"],
            "exploratory_non_promotional",
        )
        self.assertEqual(len(diagnostic["per_block"]), 2)
        first = diagnostic["per_block"][0]
        self.assertIn("descriptive_oracle_headroom", first)
        self.assertEqual(
            first["reverse_win_count"] + first["forward_win_count"] + first["tie_count"],
            first["query_count"],
        )
        self.assertEqual(first["native_old"]["status"], "not_provided")
        transfer = diagnostic["leave_one_structural_block_out"]
        self.assertEqual(transfer["status"], "evaluated")
        self.assertEqual(transfer["disposition"], "CUT_CURRENT_FEATURE_POLICY_SET")
        self.assertFalse(transfer["promotion_eligible"])
        self.assertEqual(transfer["generalizing_policies"], [])
        self.assertEqual(
            transfer["holdout_unit"],
            "one exact declared METHOD_DEV structural block",
        )
        axis_transfer = diagnostic["independence_safe_axis_transfer"]
        self.assertEqual(
            axis_transfer["disposition"],
            "CUT_UNVALIDATED_DOCUMENT_ROLE_REVERSE_PROXY",
        )
        self.assertTrue(axis_transfer["reverse_route_role_independent_cut"])
        transition = axis_transfer["policies"]["regularized_logistic"]["crossed_transition_family_x_query_fold"]
        self.assertTrue(transition["every_row_evaluated_exactly_once"])
        self.assertEqual(transition["query_group_fold_contract"]["fold_count"], 5)
        self.assertTrue(
            all(partition["independence_group_overlap_count"] == 0 for partition in transition["partition_audit"])
        )
        for method in report["methods"].values():
            self.assertEqual(method["status"], "not_run")

    def test_method_dev_dual_read_features_are_operationally_cut(self) -> None:
        pack, manifest = _method_dev_pack_and_manifest()
        manifest["features"][0]["availability_stage"] = "DUAL_READ_METHOD_DEV"
        manifest["features"][0]["execution_cost_class"] = "dual_search_probe"
        report = analyze_bcc1(pack, manifest)
        self.assertEqual(
            report["evidence_classification"]["operational_routeability"],
            "CUT_ROLE_MISMATCH_AND_DUAL_READ_FEATURE_SET",
        )
        boundary = report["data_contract"]["feature_execution_boundary"]
        self.assertEqual(boundary["status"], "DUAL_READ_METHOD_DEV")
        self.assertFalse(boundary["operational_one_route_eligible"])
        self.assertEqual(
            report["method_dev_diagnostic"]["independence_safe_axis_transfer"]["disposition"],
            "CUT_ROLE_MISMATCH_AND_DUAL_READ_FEATURE_SET",
        )

    def test_report_labels_do_not_change_frozen_policy_decisions(self) -> None:
        pack, manifest = _pack_and_manifest()
        original = analyze_bcc1(pack, manifest)
        changed_pack = copy.deepcopy(pack)
        for row in changed_pack["rows"]:
            if row["block_id"] == "report-a":
                row["scores"]["reverse"], row["scores"]["forward"] = (
                    row["scores"]["forward"],
                    row["scores"]["reverse"],
                )
        changed_manifest = copy.deepcopy(manifest)
        _bind_manifest_to_pack(changed_pack, changed_manifest)
        changed = analyze_bcc1(changed_pack, changed_manifest)
        self.assertEqual(
            original["promotion_gate"]["power_and_precision"],
            changed["promotion_gate"]["power_and_precision"],
        )
        for method in ("regularized_logistic", "knn"):
            original_directions = [item["direction"] for item in original["methods"][method]["decisions"]]
            changed_directions = [item["direction"] for item in changed["methods"][method]["decisions"]]
            self.assertEqual(original_directions, changed_directions)
            self.assertNotEqual(
                original["methods"][method]["mean_score"],
                changed["methods"][method]["mean_score"],
            )

    def test_degenerate_select_labels_use_best_fixed_fallback(self) -> None:
        pack, manifest = _pack_and_manifest()
        for row in pack["rows"]:
            if row["block_id"] == "select-a":
                row["scores"]["reverse"] = 0.9
                row["scores"]["forward"] = 0.2
        _bind_manifest_to_pack(pack, manifest)
        report = analyze_bcc1(pack, manifest)
        for method in ("regularized_logistic", "knn"):
            self.assertEqual(report["methods"][method]["status"], "fallback_only")
            self.assertEqual(
                report["methods"][method]["abstention_count"],
                report["data_contract"]["report_query_count"],
            )
            self.assertEqual(
                report["methods"][method]["mean_score"],
                report["best_fixed"]["report_mean_score"],
            )

    def test_fixed_fallback_uses_equal_structural_cell_macro(self) -> None:
        blocks = {
            "large": {
                "dataset_family_id": "dataset-large",
                "transition_family_id": "transition-a",
            },
            "small": {
                "dataset_family_id": "dataset-small",
                "transition_family_id": "transition-a",
            },
        }
        rows = [
            _row(
                "large-{}".format(index),
                "large",
                float(index),
                0.70,
                0.60,
            )
            for index in range(20)
        ]
        rows.append(_row("small-0", "small", 100.0, 0.0, 1.0))
        direction, means = bcc1_module._fixed_direction(rows, blocks)
        self.assertEqual(direction, "forward")
        self.assertEqual(means["reverse"], 0.35)
        self.assertEqual(means["forward"], 0.8)
        self.assertEqual(means["cell_count"], 2)

    def test_fixed_fallback_compares_unrounded_near_tie(self) -> None:
        blocks = {
            "cell": {
                "dataset_family_id": "dataset",
                "transition_family_id": "transition",
            }
        }
        rows = [_row("query", "cell", 0.0, 0.5, 0.5000000000004)]
        direction, means = bcc1_module._fixed_direction(rows, blocks)
        self.assertEqual(direction, "forward")
        self.assertEqual(means["reverse"], 0.5)
        self.assertEqual(means["forward"], 0.5)

    def test_noninferiority_bounds_are_strict_at_negative_margin(self) -> None:
        rows = [{"block_id": "block", "independence_group_id": "group"}]
        blocks = {
            "block": {
                "dataset_family_id": "dataset",
                "transition_family_id": "transition",
            }
        }
        superiority = {"mean_difference": 0.01, "ci_lower": 0.001}
        native_boundary = {"mean_difference": 0.0, "ci_lower": -0.02}
        block_boundary = {"mean_difference": 0.0, "ci_lower": -0.02}
        gates = {
            "minimum_worthwhile_effect": 0.005,
            "native_new_noninferiority_margin": 0.02,
            "worst_block_noninferiority_margin": 0.02,
            "minimum_overall_coverage": 0.10,
            "minimum_block_coverage": 0.05,
        }
        with (
            patch(
                "bcc1_experiment.crossed_hierarchical_paired_bootstrap_mean_ci",
                side_effect=[superiority, native_boundary],
            ),
            patch(
                "bcc1_experiment.grouped_paired_bootstrap_mean_ci",
                return_value=block_boundary,
            ),
        ):
            result = bcc1_module._confirmatory_promotion_gate(
                rows=rows,
                primary_method="regularized_logistic",
                primary_status="evaluated",
                primary_scores=np.asarray([0.6], dtype=np.float64),
                fixed_scores=np.asarray([0.5], dtype=np.float64),
                native_new_scores=np.asarray([0.6], dtype=np.float64),
                gates=gates,
                blocks=blocks,
                abstentions=[False],
                prereport_power={"pass": True},
                select_transfer={"pass": True},
            )
        self.assertFalse(result["checks"]["native_new_noninferiority"])
        self.assertFalse(result["checks"]["worst_block_safety"])

    def test_knn_fit_failure_invalidates_mixed_structural_partitions(self) -> None:
        blocks = {
            dataset: {
                "block_id": dataset,
                "dataset_family_id": dataset,
                "transition_family_id": "transition-a",
                "role": "SELECT",
            }
            for dataset in ("dataset-a", "dataset-b", "dataset-c")
        }
        rows = []
        for dataset, count in (
            ("dataset-a", 2),
            ("dataset-b", 2),
            ("dataset-c", 6),
        ):
            for index in range(count):
                reverse_wins = index % 2 == 0
                rows.append(
                    _row(
                        "{}-{}".format(dataset, index),
                        dataset,
                        float(index),
                        0.8 if reverse_wins else 0.2,
                        0.2 if reverse_wins else 0.8,
                    )
                )
        scheme = bcc1_module._select_structural_scheme(
            rows=rows,
            feature_names=("query_drift",),
            blocks=blocks,
            policy_name="knn",
            axis_field="dataset_family_id",
            gates={
                "minimum_worthwhile_effect": 0.005,
                "minimum_overall_coverage": 0.10,
                "minimum_block_coverage": 0.05,
            },
        )
        self.assertEqual(scheme["invalid_partition_count"], 1)
        self.assertFalse(scheme["checks"]["all_partitions_valid"])
        self.assertFalse(scheme["pass"])
        invalid = [item for item in scheme["partition_audit"] if not item["valid"]]
        self.assertEqual(invalid[0]["heldout_family_id"], "dataset-c")
        self.assertEqual(invalid[0]["failure_stage"], "fit")
        self.assertIn("at least 5 non-tied", invalid[0]["failure_reason"])
        self.assertTrue(any(item["valid"] for item in scheme["partition_audit"]))

    def test_empty_structural_folds_are_explicit_and_fail_closed(self) -> None:
        blocks = {}
        rows = []
        group_ids = ["group-{}".format(index) for index in range(4)]
        for transition_id in ("transition-a", "transition-b"):
            block_id = "dataset-{}".format(transition_id)
            blocks[block_id] = {
                "block_id": block_id,
                "dataset_family_id": "dataset",
                "transition_family_id": transition_id,
                "role": "SELECT",
            }
            for index, group_id in enumerate(group_ids):
                reverse_wins = index % 2 == 0
                rows.append(
                    _row(
                        "{}-{}".format(group_id, transition_id),
                        block_id,
                        float(index),
                        0.8 if reverse_wins else 0.2,
                        0.2 if reverse_wins else 0.8,
                        independence_group_id=group_id,
                    )
                )
        scheme = bcc1_module._select_structural_scheme(
            rows=rows,
            feature_names=("query_drift",),
            blocks=blocks,
            policy_name="regularized_logistic",
            axis_field="transition_family_id",
            gates={
                "minimum_worthwhile_effect": 0.005,
                "minimum_overall_coverage": 0.10,
                "minimum_block_coverage": 0.05,
            },
        )
        self.assertEqual(scheme["expected_partition_count"], 10)
        self.assertEqual(scheme["partition_count"], 8)
        self.assertEqual(scheme["empty_partition_count"], 2)
        self.assertGreaterEqual(scheme["invalid_partition_count"], 2)
        self.assertTrue(scheme["every_row_evaluated_exactly_once"])
        self.assertFalse(scheme["checks"]["all_partitions_valid"])
        self.assertFalse(scheme["pass"])
        empty_audits = [
            item for item in scheme["partition_audit"] if item["failure_reason"] == "empty evaluation partition"
        ]
        self.assertEqual(len(empty_audits), 2)
        self.assertTrue(
            all(
                item["failure_stage"] == "partition" and item["evaluation_observation_count"] == 0 and not item["valid"]
                for item in empty_audits
            )
        )

    def test_predict_and_logistic_solve_failures_are_audited_and_fail_closed(
        self,
    ) -> None:
        blocks = {
            dataset: {
                "block_id": dataset,
                "dataset_family_id": dataset,
                "transition_family_id": "transition-a",
                "role": "SELECT",
            }
            for dataset in ("dataset-a", "dataset-b", "dataset-c")
        }
        rows = [
            _row(
                "{}-{}".format(dataset, index),
                dataset,
                float(index),
                0.8 if index % 2 == 0 else 0.2,
                0.2 if index % 2 == 0 else 0.8,
            )
            for dataset in blocks
            for index in range(6)
        ]
        gates = {
            "minimum_worthwhile_effect": 0.005,
            "minimum_overall_coverage": 0.10,
            "minimum_block_coverage": 0.05,
        }
        with patch.object(
            bcc1_module,
            "_predict_knn",
            side_effect=RuntimeError("forced prediction failure"),
        ):
            predict_scheme = bcc1_module._select_structural_scheme(
                rows=rows,
                feature_names=("query_drift",),
                blocks=blocks,
                policy_name="knn",
                axis_field="dataset_family_id",
                gates=gates,
            )
        self.assertFalse(predict_scheme["pass"])
        self.assertTrue(all(item["failure_stage"] == "predict" for item in predict_scheme["partition_audit"]))
        self.assertTrue(
            all("forced prediction failure" in item["failure_reason"] for item in predict_scheme["partition_audit"])
        )
        with patch.object(
            bcc1_module,
            "_fit_logistic",
            return_value=(None, "regularized logistic solve was singular"),
        ):
            fit_scheme = bcc1_module._select_structural_scheme(
                rows=rows,
                feature_names=("query_drift",),
                blocks=blocks,
                policy_name="regularized_logistic",
                axis_field="dataset_family_id",
                gates=gates,
            )
        self.assertFalse(fit_scheme["pass"])
        self.assertTrue(all(item["failure_stage"] == "fit" for item in fit_scheme["partition_audit"]))
        self.assertTrue(
            all(
                item["failure_reason"] == "regularized logistic solve was singular"
                for item in fit_scheme["partition_audit"]
            )
        )
        with patch.object(bcc1_module, "LOGISTIC_MAX_ITERATIONS", 1):
            nonconverged_scheme = bcc1_module._select_structural_scheme(
                rows=rows,
                feature_names=("query_drift",),
                blocks=blocks,
                policy_name="regularized_logistic",
                axis_field="dataset_family_id",
                gates=gates,
            )
        self.assertFalse(nonconverged_scheme["pass"])
        self.assertTrue(
            all(
                item["failure_stage"] == "fit" and "did not converge within 1 iterations" in item["failure_reason"]
                for item in nonconverged_scheme["partition_audit"]
            )
        )

    def test_power_simulates_asymmetric_declared_report_shape(self) -> None:
        policy_name = "regularized_logistic"
        blocks = {}
        oof_rows = []
        for dataset_index in range(5):
            dataset_id = "select-dataset-{}".format(dataset_index)
            group_ids = ["{}-group-{}".format(dataset_id, group_index) for group_index in range(3)]
            for transition_index in range(4):
                transition_id = "select-transition-{}".format(transition_index)
                block_id = "{}-{}".format(dataset_id, transition_id)
                blocks[block_id] = {
                    "block_id": block_id,
                    "dataset_family_id": dataset_id,
                    "transition_family_id": transition_id,
                    "role": "SELECT",
                }
                for group_index, group_id in enumerate(group_ids):
                    oof_rows.append(
                        {
                            "block_id": block_id,
                            "query_id": "{}-query".format(group_id),
                            "independence_group_id": group_id,
                            "dataset_family_id": dataset_id,
                            "transition_family_id": transition_id,
                            "partition_valid": True,
                            "residual": (-0.01, 0.0, 0.01)[group_index],
                        }
                    )
        oof_rows.sort(key=lambda item: (item["block_id"], item["query_id"]))
        oof_sha256 = sha256_bytes(
            canonical_json_bytes(
                {
                    "schema_version": bcc1_module.OOF_RESIDUAL_AUDIT_SCHEMA,
                    "policy_name": policy_name,
                    "axis": "transition_family_id",
                    "rows": oof_rows,
                }
            )
        )
        report_rows = []
        for dataset_index in range(4):
            dataset_id = "report-dataset-{}".format(dataset_index)
            group_ids = ["{}-group-{}".format(dataset_id, group_index) for group_index in range(2)]
            for transition_index in range(3):
                transition_id = "report-transition-{}".format(transition_index)
                block_id = "{}-{}".format(dataset_id, transition_id)
                blocks[block_id] = {
                    "block_id": block_id,
                    "dataset_family_id": dataset_id,
                    "transition_family_id": transition_id,
                    "role": "REPORT",
                }
                for group_id in group_ids:
                    report_rows.append(
                        _row(
                            "{}-{}-query".format(group_id, transition_id),
                            block_id,
                            0.0,
                            0.5,
                            0.5,
                            independence_group_id=group_id,
                        )
                    )
        result = bcc1_module._select_power_gate(
            policy_name=policy_name,
            structural_scheme={
                "axis": "transition_family_id",
                "every_row_evaluated_exactly_once": True,
                "oof_residual_audit": oof_rows,
                "oof_residual_audit_sha256": oof_sha256,
            },
            report_rows=report_rows,
            blocks=blocks,
            gates={
                "power_variance_inflation": 1.25,
                "power_seed": 1730,
                "power_replicates": 100,
                "minimum_worthwhile_effect": 0.005,
                "target_power": 0.80,
                "minimum_report_queries": 24,
                "minimum_report_blocks": 12,
            },
        )
        self.assertEqual(result["status"], "evaluated")
        self.assertEqual(
            result["source_variance_library"]["dataset_family_count"],
            5,
        )
        self.assertEqual(
            result["source_variance_library"]["transition_family_count"],
            4,
        )
        self.assertEqual(result["report_design"]["dataset_family_count"], 4)
        self.assertEqual(result["report_design"]["transition_family_count"], 3)
        self.assertEqual(result["report_design"]["cell_count"], 12)
        self.assertTrue(
            all(
                cell["independence_group_count"] == 2 and cell["query_count"] == 2
                for cell in result["report_design"]["cells"]
            )
        )

    def test_power_couples_repeated_groups_across_transitions(self) -> None:
        policy_name = "regularized_logistic"
        blocks = {}
        oof_rows = []
        for transition_id in ("select-transition-a", "select-transition-b"):
            block_id = "select-dataset-{}".format(transition_id)
            blocks[block_id] = {
                "block_id": block_id,
                "dataset_family_id": "select-dataset",
                "transition_family_id": transition_id,
                "role": "SELECT",
            }
            for group_id, residual in (("negative", -1.0), ("positive", 1.0)):
                oof_rows.append(
                    {
                        "block_id": block_id,
                        "query_id": "{}-{}".format(group_id, transition_id),
                        "independence_group_id": group_id,
                        "dataset_family_id": "select-dataset",
                        "transition_family_id": transition_id,
                        "partition_valid": True,
                        "residual": residual,
                    }
                )
        oof_rows.sort(key=lambda item: (item["block_id"], item["query_id"]))
        oof_sha256 = sha256_bytes(
            canonical_json_bytes(
                {
                    "schema_version": bcc1_module.OOF_RESIDUAL_AUDIT_SCHEMA,
                    "policy_name": policy_name,
                    "axis": "transition_family_id",
                    "rows": oof_rows,
                }
            )
        )
        report_rows = []
        for transition_id in ("report-transition-a", "report-transition-b"):
            block_id = "report-dataset-{}".format(transition_id)
            blocks[block_id] = {
                "block_id": block_id,
                "dataset_family_id": "report-dataset",
                "transition_family_id": transition_id,
                "role": "REPORT",
            }
            report_rows.append(
                _row(
                    "report-query-{}".format(transition_id),
                    block_id,
                    0.0,
                    0.5,
                    0.5,
                    independence_group_id="repeated-report-group",
                )
            )
        result = bcc1_module._select_power_gate(
            policy_name=policy_name,
            structural_scheme={
                "axis": "transition_family_id",
                "every_row_evaluated_exactly_once": True,
                "oof_residual_audit": oof_rows,
                "oof_residual_audit_sha256": oof_sha256,
            },
            report_rows=report_rows,
            blocks=blocks,
            gates={
                "power_variance_inflation": 1.0,
                "power_seed": 1730,
                "power_replicates": 2000,
                "minimum_worthwhile_effect": 0.005,
                "target_power": 0.80,
                "minimum_report_queries": 2,
                "minimum_report_blocks": 2,
            },
        )
        self.assertEqual(result["status"], "evaluated")
        self.assertGreater(result["null_effect_standard_deviation"], 0.95)
        self.assertIn(
            "reuse one target-group to source-group mapping across all transitions",
            result["resampling"],
        )

    def test_oracle_headroom_is_computed_from_paired_report_queries(self) -> None:
        pack, manifest = _pack_and_manifest()
        report = analyze_bcc1(pack, manifest)
        self.assertEqual(report["best_fixed"]["report_mean_score"], 0.55)
        self.assertEqual(report["direction_oracle"]["report_mean_score"], 0.8)
        self.assertEqual(
            report["direction_oracle"]["descriptive_ex_post_choice_regret_upper_bound"],
            0.25,
        )

    def test_paired_bootstrap_is_deterministic(self) -> None:
        first = paired_bootstrap_mean_ci([0.4, 0.8, 0.6], [0.3, 0.7, 0.2])
        second = paired_bootstrap_mean_ci([0.4, 0.8, 0.6], [0.3, 0.7, 0.2])
        self.assertEqual(first, second)
        self.assertEqual(first["mean_difference"], 0.2)
        self.assertLessEqual(first["ci_lower"], first["mean_difference"])
        self.assertGreaterEqual(first["ci_upper"], first["mean_difference"])

    def test_hierarchical_bootstrap_equal_weights_structural_blocks(self) -> None:
        candidate = [1.0, 1.0, 0.0]
        comparator = [0.0, 0.0, 0.0]
        blocks = ["large", "large", "small"]
        first = hierarchical_paired_bootstrap_mean_ci(candidate, comparator, blocks, replicates=500, seed=7)
        second = hierarchical_paired_bootstrap_mean_ci(candidate, comparator, blocks, replicates=500, seed=7)
        self.assertEqual(first, second)
        self.assertEqual(first["block_count"], 2)
        self.assertEqual(first["query_count"], 3)
        self.assertEqual(first["mean_difference"], 0.5)

    def test_small_report_cannot_bypass_frozen_universe_gate(self) -> None:
        pack, manifest = _pack_and_manifest()
        select_rows = [
            _row(
                "sf{}".format(index),
                "select-a",
                -10.0 - index,
                0.10,
                0.90,
            )
            for index in range(6)
        ] + [
            _row(
                "sr{}".format(index),
                "select-a",
                10.0 + index,
                0.80,
                0.70,
            )
            for index in range(6)
        ]
        report_rows = [
            _row(
                "ra{}".format(index),
                "report-a",
                12.0 + index,
                0.80,
                0.70,
            )
            for index in range(4)
        ] + [
            _row(
                "rb{}".format(index),
                "report-b",
                12.5 + index,
                0.80,
                0.70,
            )
            for index in range(4)
        ]
        pack["rows"] = select_rows + report_rows
        manifest["blocks"] = [
            {
                "block_id": block_id,
                "dataset_family_id": ("select-dataset" if role == "SELECT" else block_id + "-dataset"),
                "transition_family_id": ("select-transition" if role == "SELECT" else "report-transition"),
                "role": role,
                "query_ids_sha256": query_ids_sha256(
                    row["query_id"] for row in pack["rows"] if row["block_id"] == block_id
                ),
                "consumed": False,
            }
            for block_id, role in (
                ("select-a", "SELECT"),
                ("report-a", "REPORT"),
                ("report-b", "REPORT"),
            )
        ]
        manifest["confirmatory_gates"]["primary_method"] = "knn"
        _bind_manifest_to_pack(pack, manifest)
        report = analyze_bcc1(pack, manifest)
        self.assertFalse(report["evidence_classification"]["promotion_eligible"])
        self.assertEqual(report["promotion_gate"]["decision"], "BLOCKED")
        self.assertFalse(report["promotion_gate"]["checks"]["mwee_power_adequacy"])

    def test_soft_fusion_runs_only_with_justified_frozen_inputs(self) -> None:
        pack, manifest = _pack_and_manifest(soft_fusion=True)
        report = analyze_bcc1(pack, manifest)
        fusion = report["methods"]["soft_score_fusion"]
        self.assertEqual(fusion["status"], "evaluated")
        self.assertEqual(
            fusion["selection"]["selected_candidate_id"],
            "alpha-075",
        )
        self.assertEqual(fusion["mean_score"], 0.72)

    def test_unjustified_soft_fusion_is_rejected(self) -> None:
        pack, manifest = _pack_and_manifest(soft_fusion=True)
        manifest["soft_fusion"]["qrels_free_at_inference"] = False
        with self.assertRaisesRegex(BCC1ValidationError, "qrels_free_at_inference"):
            analyze_bcc1(pack, manifest)


class TestFileRunner(unittest.TestCase):
    def test_method_dev_file_runner_writes_exact_report_without_consumption(self) -> None:
        pack, manifest = _method_dev_pack_and_manifest()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pack_path = root / "pack.json"
            manifest_path = root / "manifest.json"
            output_path = root / "report.json"
            pack_path.write_bytes(canonical_json_bytes(pack))
            original_manifest_raw = canonical_json_bytes(manifest)
            manifest_path.write_bytes(original_manifest_raw)

            report = run_bcc1_files(
                pack_path=pack_path,
                manifest_path=manifest_path,
                output_path=output_path,
            )
            self.assertEqual(
                output_path.read_bytes(),
                canonical_json_bytes(report, trailing_newline=True),
            )
            self.assertEqual(manifest_path.read_bytes(), original_manifest_raw)
            self.assertFalse(report["evidence_classification"]["confirmatory_report_evaluated"])
            claim = json.loads(manifest_path.with_name(manifest_path.name + ".bcc1.lock").read_text(encoding="utf-8"))
            self.assertEqual(claim["state"], "METHOD_DEV_COMPLETE")
            self.assertFalse(claim["report_outcomes_opened"])

    def test_confirmatory_single_pack_is_blocked_before_pack_read(self) -> None:
        _, manifest = _pack_and_manifest()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing_pack = root / "sealed-outcomes-must-not-open.json"
            manifest_path = root / "manifest.json"
            output_path = root / "report.json"
            manifest_path.write_bytes(canonical_json_bytes(manifest))
            with self.assertRaisesRegex(BCC1ValidationError, "BLOCKED_PROTOCOL_PREOPEN"):
                run_bcc1_files(
                    pack_path=missing_pack,
                    manifest_path=manifest_path,
                    output_path=output_path,
                )
            lock_path = manifest_path.with_name(manifest_path.name + ".bcc1.lock")
            claim = json.loads(lock_path.read_text(encoding="utf-8"))
            self.assertEqual(claim["state"], "BLOCKED_PROTOCOL_PREOPEN")
            self.assertFalse(claim["report_outcomes_opened"])
            self.assertFalse(missing_pack.exists())
            self.assertFalse(output_path.exists())

    def test_forged_method_dev_envelope_consumes_and_blocks_restricted_pack(
        self,
    ) -> None:
        pack, manifest = _pack_and_manifest()
        forged_manifest = copy.deepcopy(manifest)
        forged_manifest["evidence_mode"] = "METHOD_DEV"
        forged_manifest["sampled"] = True
        for block in forged_manifest["blocks"]:
            block["role"] = "METHOD_DEV"
            block["consumed"] = False

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pack_path = root / "restricted-pack.json"
            manifest_path = root / "forged-method-dev-manifest.json"
            output_path = root / "report.json"
            pack_path.write_bytes(canonical_json_bytes(pack))
            manifest_path.write_bytes(canonical_json_bytes(forged_manifest))

            with self.assertRaisesRegex(BCC1ValidationError, "consumed under the durable pre-open claim"):
                run_bcc1_files(
                    pack_path=pack_path,
                    manifest_path=manifest_path,
                    output_path=output_path,
                )

            lock_path = manifest_path.with_name(manifest_path.name + ".bcc1.lock")
            claim = json.loads(lock_path.read_text(encoding="utf-8"))
            self.assertEqual(claim["state"], "BLOCKED_AFTER_CLAIM")
            self.assertEqual(claim["consumption_state"], "CONSUMED_ON_CLAIM")
            self.assertTrue(claim["pack_bytes_opened"])
            self.assertTrue(claim["report_outcomes_opened"])
            self.assertFalse(output_path.exists())

            with self.assertRaisesRegex(BCC1ValidationError, "locked or protocol-blocked"):
                run_bcc1_files(
                    pack_path=pack_path,
                    manifest_path=manifest_path,
                    output_path=output_path,
                )

    def test_existing_lock_blocks_alternate_output_before_report_access(self) -> None:
        pack, manifest = _pack_and_manifest()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pack_path = root / "pack.json"
            manifest_path = root / "manifest.json"
            output_path = root / "report.json"
            pack_path.write_bytes(canonical_json_bytes(pack))
            manifest_path.write_bytes(canonical_json_bytes(manifest))
            lock_path = manifest_path.with_name(manifest_path.name + ".bcc1.lock")
            lock_path.write_text("held", encoding="utf-8")
            with self.assertRaisesRegex(BCC1ValidationError, "locked or protocol-blocked"):
                run_bcc1_files(
                    pack_path=pack_path,
                    manifest_path=manifest_path,
                    output_path=output_path,
                )
            self.assertFalse(output_path.exists())

    def test_missing_pack_fails_without_generating_data(self) -> None:
        _, safe_manifest = _method_dev_pack_and_manifest()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            missing = root / "missing-pack.json"
            manifest = root / "manifest.json"
            manifest.write_bytes(canonical_json_bytes(safe_manifest))
            with self.assertRaisesRegex(BCC1ValidationError, "does not exist"):
                run_bcc1_files(
                    pack_path=missing,
                    manifest_path=manifest,
                    output_path=root / "report.json",
                )
            self.assertFalse(missing.exists())
            self.assertFalse((root / "report.json").exists())

    def test_cli_returns_validation_error_for_missing_pack(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            result = cli_main(
                [
                    "--pack",
                    str(root / "missing.json"),
                    "--manifest",
                    str(root / "manifest.json"),
                    "--output",
                    str(root / "report.json"),
                ]
            )
            self.assertEqual(result, 2)
            self.assertFalse((root / "report.json").exists())


if __name__ == "__main__":
    unittest.main()
