"""Tests for the bounded PRW METHOD_DEV campaign runner."""

from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np

from run_prime_ring_waypoint_experiment import (
    CampaignConfig,
    CampaignValidationError,
    _RawSanityLiveAuthority,
    _RawSanityLiveSeal,
    _aggregate_raw_sanity_sweep,
    _build_carrier_templates,
    _build_payload_groups,
    _campaign_contract_manifest,
    _cell_id,
    _draw_flip_mask,
    _decode_native_oppw,
    _make_planted_query,
    _materialize_payload_collision_bank,
    _observe_and_decode_repeated_bit,
    _observe_native_oppw,
    _payload_query,
    _phase_codebook,
    _prepare_payload_controls,
    _primary_aggregate,
    _raw_type_accuracy_provenance,
    _raw_sanity_live_seal,
    _run_base_group,
    _stable_digest,
    _trial_truth,
    _unique_numpy_nbytes,
    clopper_pearson_upper,
    expected_cell_specs,
    run_campaign,
    validate_config,
)


def _tiny_config(**overrides) -> CampaignConfig:
    config = CampaignConfig(
        seeds=(7,),
        lengths=(31, 32),
        carrier_families=("legendre", "rademacher"),
        layers=(1, 8),
        planted_mask_families=("none", "typed16"),
        decoder_mask_policies=("none", "typed16", "free256"),
        bit_flip_rates=(0.35,),
        payload_noise_rates=(0.25,),
        type_count=3,
        waypoints_per_type=2,
        select_planted=1,
        select_unrelated=2,
        report_planted=1,
        report_unrelated=2,
        primary_select_planted=1,
        primary_select_unrelated=2,
        primary_report_planted=1,
        primary_report_unrelated=2,
    )
    return replace(config, **overrides)


def _without_timing(value):
    """Remove explicitly nondeterministic wall-clock fields recursively."""

    if isinstance(value, dict):
        return {
            key: _without_timing(item)
            for key, item in value.items()
            if "latency" not in key.lower()
            and "efficiency" not in key.lower()
        }
    if isinstance(value, list):
        return [_without_timing(item) for item in value]
    return value


def _threshold_map(artifact):
    return {
        cell["cell_id"]: cell["threshold"]
        for cell in artifact["cells"]
        if cell["status"] == "completed"
    }


def _fabricated_full_contract_config() -> CampaignConfig:
    return CampaignConfig(
        seeds=(7, 42, 1337),
        lengths=(4691,),
        carrier_families=("legendre",),
        layers=(8,),
        planted_mask_families=("typed16",),
        decoder_mask_policies=("typed16",),
        bit_flip_rates=(0.0, 0.20, 0.35, 0.45),
        payload_noise_rates=(0.25,),
        type_count=16,
        waypoints_per_type=8,
        select_planted=2,
        select_unrelated=4,
        report_planted=3,
        report_unrelated=8,
        primary_select_planted=512,
        primary_select_unrelated=512,
        primary_report_planted=1024,
        primary_report_unrelated=1024,
        run_label="FULL_METHOD_DEV",
    )


def _fabricated_raw_sanity_bundle(config=None, *, failed_key=None):
    if config is None:
        config = _fabricated_full_contract_config()
    campaign_digest = _campaign_contract_manifest(config)["digest"]
    live_authority = _RawSanityLiveAuthority(campaign_digest)
    cells = []
    live_seals = {}
    carrier_provenance = {
        "representation": "prime_legendre_ring",
        "template_factory": "make_template",
        "ring_observation_labels_manually_constructed": False,
        "base_shared_across_types": True,
        "composite_control": False,
        "template_dtype": "float64",
        "template_shape": [config.type_count, 8, 4691],
        "template_nbytes": config.type_count * 8 * 4691 * 8,
        "base_nbytes": 4691 * 8,
        "corrupted_queries_are_raw_array_copies": True,
        "payload_bank": {"fixture": "bounded_test_only"},
    }
    for seed in (7, 42, 1337):
        for rate in (0.0, 0.20, 0.35, 0.45):
            primary = rate == 0.45
            trial_count = 1024 if primary else 3
            correct_count = (
                trial_count - 1
                if failed_key == (seed, rate)
                else trial_count
            )
            spec = {
                "seed": seed,
                "length": 4691,
                "carrier_family": "legendre",
                "layers": 8,
                "planted_mask_family": "typed16",
                "decoder_mask_policy": "typed16",
                "bit_flip_rate": rate,
                "payload_noise_rate": 0.25,
            }
            rows = []
            for index in range(trial_count):
                truth = _trial_truth(
                    config,
                    seed,
                    "REPORT",
                    "planted",
                    index,
                    4691,
                )
                true_type = int(truth["true_type"])
                type_correct = index < correct_count
                rows.append(
                    {
                        "split": "REPORT",
                        "input_class": "planted",
                        "group_id": truth["group_id"],
                        "true_type": true_type,
                        "predicted_type": (
                            true_type
                            if type_correct
                            else (true_type + 1) % config.type_count
                        ),
                        "type_correct": type_correct,
                        "query_provenance": {
                            "representation": "raw_float64_ndarray",
                            "corruption": (
                                "independent_bernoulli_bit_flips"
                            ),
                            "noise_variant": "iid",
                            "declared_global_flip_probability_or_fraction": (
                                rate
                            ),
                        },
                    }
                )
            cell = {
                "cell_id": _cell_id(spec),
                "config": spec,
                "campaign_contract_digest": campaign_digest,
                "status": "completed",
                "primary_fur_seed_component": primary,
                "carrier_provenance": carrier_provenance,
                "threshold": {
                    "status": (
                        "FROZEN"
                        if primary
                        else "NO_ELIGIBLE_THRESHOLD_FAIL_CLOSED"
                    ),
                    "source_split": "SELECT",
                    "report_rows_consumed": 0,
                },
                "report": {
                    "planted_group_count": trial_count,
                    "raw_type_accuracy": correct_count / trial_count,
                    "raw_type_accuracy_provenance": _raw_type_accuracy_provenance(
                        config,
                        spec,
                        rows,
                    ),
                },
                "promotion_eligible": False,
            }
            cells.append(cell)
            live_seal = _raw_sanity_live_seal(
                config,
                spec,
                rows,
                carrier_provenance,
                authority=live_authority,
            )
            if live_seal is None:
                raise AssertionError("fabricated full contract must produce a seal")
            live_seals[cell["cell_id"]] = live_seal
    return cells, live_seals, live_authority


def _fabricated_raw_sanity_cells(config=None, *, failed_key=None):
    return _fabricated_raw_sanity_bundle(
        config,
        failed_key=failed_key,
    )[0]


def _fabricated_raw_sanity_sweep(
    config,
    cells,
    *,
    failed_key=None,
):
    _, live_seals, live_authority = _fabricated_raw_sanity_bundle(
        config,
        failed_key=failed_key,
    )
    return _aggregate_raw_sanity_sweep(
        config,
        cells,
        live_seals=live_seals,
        live_authority=live_authority,
    )


def _fabricated_primary_components(config, *, correct):
    resource_accounting = {
        "kind": "conservative_simultaneous_live_array_estimate",
        "standalone_estimated_peak_bytes": 1024,
        "harness_estimated_peak_bytes": 2048,
        "estimated_work_units": 100,
        "measured_process_peak": False,
    }
    native_contract = {
        "physical_channel_uses": 8 * 4691,
        "transmitted_energy": 8 * 4691,
        "dense_phase_estimates_consumed": False,
    }
    repeated_contract = {
        "physical_channel_uses": 8 * 4691,
        "transmitted_energy": 8 * 4691,
        "equal_channel_uses_to_dense_carrier": True,
    }

    def row(split, planted):
        row_correct = bool(correct and planted)
        score = 1.0 if row_correct else 0.0
        return {
            "split": split,
            "group_id": f"FABRICATED-{split}-{'P' if planted else 'U'}",
            "score": score,
            "margin": 1.0,
            "type_correct": row_correct,
            "native_oppw_control": {
                "status": "COMPLETED",
                "score": score,
                "margin": 1.0,
                "observation_contract": native_contract,
                "resource_accounting": resource_accounting,
            },
            "native_oppw_type_correct": row_correct,
            "native_oppw_phase_exact": row_correct,
            "repeated_bit_control": {
                "status": "COMPLETED",
                "score": score,
                "margin": 1.0,
                "contract": repeated_contract,
                "resource_accounting": resource_accounting,
            },
            "repeated_bit_type_correct": row_correct,
        }

    components = []
    for seed in config.seeds:
        components.append(
            {
                "seed": seed,
                "rows": {
                    "SELECT": {
                        "planted": [row("SELECT", True)] * 512,
                        "unrelated": [row("SELECT", False)] * 512,
                    },
                    "REPORT": {
                        "planted": [row("REPORT", True)] * 1024,
                        "unrelated": [row("REPORT", False)] * 1024,
                    },
                },
            }
        )
    return components


def _fabricated_frozen_threshold():
    return {
        "status": "FROZEN",
        "source_split": "SELECT",
        "score_threshold": 0.5,
        "margin_threshold": 0.5,
        "report_rows_consumed": 0,
    }


class TestPrimeRingWaypointCampaign(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.config = _tiny_config()
        cls.artifact = run_campaign(cls.config, write_artifact=False)

    def test_complete_grid_retains_completed_unavailable_and_error_cells(self):
        expected = expected_cell_specs(self.config)
        manifest = self.artifact["grid_manifest"]
        self.assertEqual(manifest["expected_cell_count"], len(expected))
        self.assertEqual(manifest["retained_cell_count"], len(expected))
        self.assertTrue(manifest["all_cells_retained"])
        self.assertTrue(manifest["cell_ids_unique"])
        self.assertNotIn("pending", manifest["status_counts"])
        self.assertEqual(len(self.artifact["cells"]), len(expected))
        self.assertTrue(
            all(
                cell["status"] in {"completed", "unavailable", "error"}
                for cell in self.artifact["cells"]
            )
        )

    def test_same_config_is_deterministic_after_timing_is_normalized(self):
        second = run_campaign(self.config, write_artifact=False)
        self.assertEqual(
            _without_timing(self.artifact),
            _without_timing(second),
        )

    def test_report_stream_cannot_change_select_thresholds(self):
        alternate = run_campaign(
            replace(self.config, report_stream_tag="REPORT-ALTERNATE"),
            write_artifact=False,
        )
        self.assertEqual(
            _threshold_map(self.artifact),
            _threshold_map(alternate),
        )
        self.assertNotEqual(
            self.artifact["rng_streams"]["stream_ids"]["REPORT"],
            alternate["rng_streams"]["stream_ids"]["REPORT"],
        )
        self.assertEqual(
            self.artifact["rng_streams"]["stream_ids"]["SELECT"],
            alternate["rng_streams"]["stream_ids"]["SELECT"],
        )
        for threshold in _threshold_map(self.artifact).values():
            self.assertEqual(threshold["source_split"], "SELECT")
            self.assertEqual(threshold["report_rows_consumed"], 0)

    def test_invalid_configs_fail_closed_before_execution(self):
        invalid = (
            replace(self.config, bit_flip_rates=(0.5,)),
            replace(self.config, seeds=(7, 7)),
            replace(self.config, decoder_mask_policies=("unknown",)),
            replace(self.config, type_count=True),
            replace(self.config, control_min_true_unlock_recall=0.0),
            replace(self.config, report_stream_tag=""),
        )
        for config in invalid:
            with self.subTest(config=config):
                with self.assertRaises(CampaignValidationError):
                    validate_config(config)
                with self.assertRaises(CampaignValidationError):
                    run_campaign(config, write_artifact=False)

    def test_artifact_schema_and_claim_boundaries(self):
        artifact = self.artifact
        self.assertEqual(
            artifact["record_type"],
            "prime_ring_waypoint_method_dev_campaign",
        )
        self.assertEqual(artifact["schema_version"], "1.1.0")
        self.assertEqual(artifact["evidence_mode"], "METHOD_DEV")
        self.assertEqual(artifact["run_label"], "SMOKE_NON_EVIDENTIARY")
        self.assertFalse(artifact["promotion_eligible"])
        self.assertFalse(artifact["rader_claim"])
        self.assertEqual(artifact["novelty_claim_status"], "unconfirmed")
        self.assertEqual(
            artifact["primary_aggregate"]["status"],
            "NOT_RUN_SMOKE_NON_EVIDENTIARY",
        )
        self.assertFalse(
            artifact["verdicts"]["PRW-4"]["special_status_for_4691"]
        )
        governance = artifact["control_claim_governance"]
        self.assertTrue(governance["native_sparse_oppw_required"])
        self.assertTrue(
            governance["equal_channel_use_repeated_bit_required"]
        )
        self.assertFalse(
            governance["matched_resource_advantage_claim_eligible"]
        )
        encoded = json.dumps(artifact, allow_nan=False)
        self.assertIn("SMOKE_NON_EVIDENTIARY", encoded)

    def test_primary_aggregate_closes_fabricated_pooled_control_gates(self):
        config = _fabricated_full_contract_config()
        components = _fabricated_primary_components(config, correct=True)
        frozen_threshold = _fabricated_frozen_threshold()
        (
            raw_sanity_cells,
            raw_sanity_live_seals,
            raw_sanity_live_authority,
        ) = _fabricated_raw_sanity_bundle()
        with patch(
            "run_prime_ring_waypoint_experiment.fit_select_thresholds",
            return_value=frozen_threshold,
        ):
            aggregate = _primary_aggregate(
                config,
                components,
                raw_sanity_cells=raw_sanity_cells,
                raw_sanity_live_seals=raw_sanity_live_seals,
                raw_sanity_live_authority=raw_sanity_live_authority,
            )
        self.assertEqual(aggregate["status"], "REPORT_EVALUATED")
        self.assertTrue(aggregate["fur_gate_pass"])
        self.assertTrue(
            aggregate["native_sparse_oppw_resource_control_closed"]
        )
        self.assertTrue(
            aggregate["equal_channel_use_repeated_bit_control_closed"]
        )
        self.assertTrue(aggregate["dense_primary_control_closed"])
        self.assertTrue(aggregate["dense_raw_type_sanity_gate_pass"])
        self.assertTrue(
            aggregate["through_q_0_45_raw_sanity_gate_complete"]
        )
        self.assertTrue(aggregate["all_mandatory_controls_closed"])
        self.assertFalse(aggregate["matched_noise_comparison_ready"])
        self.assertEqual(
            aggregate["controls"]["native_sparse_oppw"]["status"],
            "COMPLETED",
        )
        self.assertEqual(
            aggregate["controls"]["equal_channel_use_repeated_bit"]["status"],
            "COMPLETED",
        )
        self.assertFalse(aggregate["promotion_eligible"])
        native_resources = aggregate["controls"]["native_sparse_oppw"][
            "resource_accounting"
        ]
        self.assertEqual(
            native_resources["estimated_work_units"]["mean"],
            100.0,
        )
        self.assertIn("hard_max_work_units", native_resources)

        locked_components = _fabricated_primary_components(
            config, correct=False
        )
        with patch(
            "run_prime_ring_waypoint_experiment.fit_select_thresholds",
            return_value=frozen_threshold,
        ):
            locked = _primary_aggregate(
                config,
                locked_components,
                raw_sanity_cells=raw_sanity_cells,
                raw_sanity_live_seals=raw_sanity_live_seals,
                raw_sanity_live_authority=raw_sanity_live_authority,
            )
        self.assertTrue(locked["fur_gate_pass"])
        self.assertEqual(locked["report_true_unlock_recall"], 0.0)
        self.assertFalse(locked["dense_raw_type_sanity_gate_pass"])
        self.assertFalse(locked["dense_primary_control_closed"])
        self.assertFalse(
            locked["native_sparse_oppw_resource_control_closed"]
        )
        self.assertFalse(
            locked["equal_channel_use_repeated_bit_control_closed"]
        )
        self.assertFalse(locked["all_mandatory_controls_closed"])

    def test_4096_uses_true_generic_array_control_and_legendre_is_retained_unavailable(self):
        control_config = CampaignConfig(
            seeds=(7,),
            lengths=(4096,),
            carrier_families=("legendre", "rademacher"),
            layers=(8,),
            planted_mask_families=("none",),
            decoder_mask_policies=("none",),
            bit_flip_rates=(0.35,),
            payload_noise_rates=(0.25,),
            type_count=2,
            waypoints_per_type=2,
            select_planted=1,
            select_unrelated=1,
            report_planted=1,
            report_unrelated=1,
            primary_select_planted=1,
            primary_select_unrelated=1,
            primary_report_planted=1,
            primary_report_unrelated=1,
        )
        artifact = run_campaign(control_config, write_artifact=False)
        generic = [
            cell
            for cell in artifact["cells"]
            if cell["status"] == "completed"
            and cell["config"]["length"] == 4096
            and cell["config"]["carrier_family"] == "rademacher"
            and cell["config"]["layers"] == 8
        ]
        self.assertTrue(generic)
        self.assertTrue(
            all(
                cell["carrier_provenance"]["representation"]
                == "random_composite_control"
                for cell in generic
            )
        )
        unavailable = [
            cell
            for cell in artifact["cells"]
            if cell["config"]["length"] == 4096
            and cell["config"]["carrier_family"] == "legendre"
        ]
        self.assertTrue(unavailable)
        self.assertTrue(all(cell["status"] == "unavailable" for cell in unavailable))

    def test_smoke_threshold_exercises_wiring_but_never_passes_fur_gate(self):
        completed = [
            cell for cell in self.artifact["cells"] if cell["status"] == "completed"
        ]
        self.assertTrue(
            any(
                cell["threshold"]["status"]
                == "SMOKE_FROZEN_POINT_ESTIMATE_ONLY"
                for cell in completed
            )
        )
        self.assertTrue(
            all(not cell["report"]["fur_gate_pass"] for cell in completed)
        )
        self.assertTrue(all(not cell["promotion_eligible"] for cell in completed))
        crossed = next(
            cell
            for cell in completed
            if cell["config"]["layers"] == 8
            and cell["config"]["planted_mask_family"] == "typed16"
            and cell["config"]["decoder_mask_policy"] == "none"
        )
        diagnostic = crossed["construction_diagnostics"][
            "hamming_distance_and_bsc_reduction"
        ]
        self.assertFalse(diagnostic["bsc_observed_cell_applicable"])
        self.assertEqual(diagnostic["relative_mask_word_count"], 16)
        self.assertEqual(
            crossed["controls"]["equal_channel_use_repeated_bit"]["status"],
            "COMPLETED",
        )
        repeated_contract = crossed["controls"][
            "equal_channel_use_repeated_bit"
        ]["contract"]
        self.assertEqual(repeated_contract["physical_channel_uses"], 8 * 31)
        self.assertEqual(repeated_contract["transmitted_energy"], 8 * 31)
        self.assertTrue(
            repeated_contract["equal_channel_uses_to_dense_carrier"]
        )
        native = crossed["controls"]["native_sparse_oppw"]
        self.assertEqual(native["status"], "COMPLETED")
        self.assertFalse(native["dense_phase_estimates_consumed"])
        self.assertEqual(native["contract"]["physical_channel_uses"], 8 * 31)
        correlated = crossed["report"]["correlated_noise_variants"]
        realized_rate = round(0.35 * 8 * 31) / (8 * 31)
        for variant in ("block_correlated", "burst"):
            self.assertEqual(
                correlated[variant]["observed_flip_rate"]["mean"],
                realized_rate,
            )
        self.assertEqual(correlated["threshold_source"], "iid SELECT only")
        layer_one = next(
            cell
            for cell in completed
            if cell["config"]["layers"] == 1
        )
        self.assertEqual(
            layer_one["controls"]["equal_channel_use_repeated_bit"]["status"],
            "STRUCTURALLY_UNAVAILABLE",
        )
        resources = crossed["resources"]
        self.assertEqual(
            resources["logical_type_expanded_payload_bytes_float64"],
            resources["deduplicated_serialized_payload_bytes_float64"]
            * self.config.type_count,
        )
        self.assertEqual(
            crossed["descriptive_efficiency_denominator"],
            (
                "deduplicated_total_serialized_method_bytes * "
                "mean_route_latency_ms"
            ),
        )
        payload_working = crossed["report"]["waypoint"][
            "payload_decoder_working_array_bytes"
        ]
        for path in (
            "global_payload_only",
            "direct_known_type",
            "routed_frozen_shift",
        ):
            self.assertEqual(
                payload_working[path]["kind"],
                "conservative_simultaneous_live_peak_estimate",
            )
            self.assertFalse(payload_working[path]["measured_process_peak"])
            components = payload_working[path]["component_breakdown"]
            self.assertEqual(
                payload_working[path]["estimate_bytes"]["mean"],
                float(sum(components.values())),
            )
        self.assertIn(
            "fft_products",
            payload_working["global_payload_only"][
                "component_breakdown"
            ],
        )
        self.assertIn(
            "shifted_times_query_product",
            payload_working["routed_frozen_shift"][
                "component_breakdown"
            ],
        )
        recall_k_caveat = crossed["report"]["waypoint"][
            "payload_only_recall_at_k_caveat"
        ]
        self.assertTrue(recall_k_caveat["canonical_exact_tie_order_dependent"])
        self.assertTrue(recall_k_caveat["k_exceeds_type_count"])
        self.assertFalse(recall_k_caveat["evidence_of_type_retrieval"])

    def test_exact_clopper_pearson_zero_event_resolution(self):
        self.assertLessEqual(clopper_pearson_upper(0, 368), 0.01)
        self.assertGreater(clopper_pearson_upper(0, 367), 0.01)
        large = [
            clopper_pearson_upper(events, trials)
            for events, trials in (
                (400, 1536),
                (700, 1536),
                (1000, 3072),
            )
        ]
        self.assertTrue(all(0.0 < value < 1.0 for value in large))
        self.assertLess(
            clopper_pearson_upper(399, 1536),
            clopper_pearson_upper(400, 1536),
        )
        self.assertLess(
            clopper_pearson_upper(400, 1536),
            clopper_pearson_upper(401, 1536),
        )
        self.assertEqual(clopper_pearson_upper(1, 1), 1.0)
        with self.assertRaises(CampaignValidationError):
            clopper_pearson_upper(2, 1)

    def test_payload_collision_bank_is_type_ambiguous_and_deduplicated(self):
        config = replace(
            self.config,
            type_count=4,
            waypoints_per_type=2,
        )
        payloads, _manifest = _build_payload_groups(7, 2, 8, 31)
        bank = _materialize_payload_collision_bank(payloads, config.type_count)
        self.assertEqual(bank.shape, (2, 8, 31))
        self.assertEqual(bank.nbytes, payloads.nbytes)
        spectra = np.fft.rfft(bank, axis=2)
        norms = np.linalg.norm(bank, axis=2)
        for true_type in range(config.type_count):
            truth = {
                "group_id": f"PRW-REPORT-S7-PAYLOAD-{true_type:05d}",
                "true_type": true_type,
                "true_waypoint": 1,
                "global_shift": 5,
            }
            query = _payload_query(
                config, 7, "REPORT", truth, bank, 0.0
            )
            result = _prepare_payload_controls(
                config,
                query,
                bank,
                true_type,
                1,
                spectra,
                norms,
            )
            self.assertTrue(result["direct_known_type_recall_at_1"])
            self.assertEqual(
                result["payload_only_recall_at_1"],
                true_type == 0,
            )

    def test_corruption_severities_use_nested_base_draws(self):
        config = replace(self.config, type_count=3)
        signatures, _manifest = _phase_codebook(31, 8, 3, 7)
        templates, _base, _provenance, ring_templates = (
            _build_carrier_templates(
                31, 8, "legendre", signatures, 7
            )
        )
        truth = _trial_truth(
            config, 7, "REPORT", "planted", 0, 31
        )
        clean, _mask, _ = _make_planted_query(
            config,
            7,
            "REPORT",
            truth,
            templates,
            ring_templates,
            "typed16",
            0.0,
        )
        low, _mask, _ = _make_planted_query(
            config,
            7,
            "REPORT",
            truth,
            templates,
            ring_templates,
            "typed16",
            0.20,
        )
        high, _mask, _ = _make_planted_query(
            config,
            7,
            "REPORT",
            truth,
            templates,
            ring_templates,
            "typed16",
            0.35,
        )
        self.assertTrue(
            np.all(np.logical_or(low == clean, high != clean))
        )

        payloads, _ = _build_payload_groups(7, 2, 8, 31)
        bank = _materialize_payload_collision_bank(payloads, 3)
        payload_truth = {
            **truth,
            "true_waypoint": 1,
        }
        payload_clean = _payload_query(
            config, 7, "REPORT", payload_truth, bank, 0.0
        )
        payload_low = _payload_query(
            config, 7, "REPORT", payload_truth, bank, 0.10
        )
        payload_high = _payload_query(
            config, 7, "REPORT", payload_truth, bank, 0.25
        )
        low_cosine = np.sum(payload_clean * payload_low, axis=1)
        high_cosine = np.sum(payload_clean * payload_high, axis=1)
        self.assertTrue(np.all(low_cosine > high_cosine))

    def test_correlated_corruptions_are_exact_deterministic_and_nested(self):
        shape = (8, 31)
        low_rate = 0.20
        high_rate = 0.35
        high_masks = {}
        for variant in ("iid", "block_correlated", "burst"):
            low = _draw_flip_mask(
                shape,
                low_rate,
                variant,
                np.random.default_rng(12345),
            )
            high = _draw_flip_mask(
                shape,
                high_rate,
                variant,
                np.random.default_rng(12345),
            )
            repeated = _draw_flip_mask(
                shape,
                high_rate,
                variant,
                np.random.default_rng(12345),
            )
            if variant == "iid":
                self.assertEqual(low.dtype, np.dtype(np.bool_))
                self.assertLessEqual(
                    int(np.count_nonzero(low)),
                    int(np.count_nonzero(high)),
                )
            else:
                self.assertEqual(
                    int(np.count_nonzero(low)),
                    round(low_rate * np.prod(shape)),
                )
                self.assertEqual(
                    int(np.count_nonzero(high)),
                    round(high_rate * np.prod(shape)),
                )
            self.assertTrue(np.all(np.logical_or(~low, high)))
            np.testing.assert_array_equal(high, repeated)
            high_masks[variant] = high
        self.assertFalse(
            np.array_equal(
                high_masks["iid"], high_masks["block_correlated"]
            )
        )
        self.assertFalse(
            np.array_equal(high_masks["iid"], high_masks["burst"])
        )
        for layer in high_masks["burst"]:
            cyclic_transitions = np.count_nonzero(layer != np.roll(layer, 1))
            self.assertLessEqual(cyclic_transitions, 2)

    def test_native_sparse_oppw_uses_positions_and_recovers_noiseless_truth(self):
        config = replace(
            self.config,
            lengths=(31,),
            carrier_families=("legendre",),
            layers=(8,),
            type_count=3,
        )
        signatures, _manifest = _phase_codebook(31, 8, 3, 7)
        truth = _trial_truth(config, 7, "REPORT", "planted", 4, 31)
        positions, contract = _observe_native_oppw(
            config,
            7,
            "REPORT",
            truth,
            signatures,
            "planted",
            0.0,
        )
        decoded = _decode_native_oppw(positions, signatures, 31)
        self.assertEqual(positions.shape, (8,))
        self.assertEqual(positions.nbytes, 8 * np.dtype(np.int64).itemsize)
        self.assertEqual(decoded["predicted_type"], truth["true_type"])
        self.assertEqual(decoded["shared_shift"], truth["global_shift"])
        self.assertFalse(decoded["dense_phase_estimates_consumed"])
        self.assertEqual(contract["physical_channel_uses"], 8 * 31)
        self.assertEqual(contract["transmitted_energy"], 8 * 31)
        self.assertEqual(contract["query_nbytes"], positions.nbytes)
        observation_resource = contract["resource_accounting"]
        self.assertEqual(
            observation_resource["standalone_estimated_peak_bytes"],
            sum(observation_resource["component_breakdown"].values()),
        )
        decoder_resource = decoded["resource_accounting"]
        self.assertEqual(
            decoder_resource["standalone_estimated_peak_bytes"],
            sum(decoder_resource["component_breakdown"].values()),
        )
        self.assertFalse(decoder_resource["measured_process_peak"])
        noisy_a, provenance_a = _observe_native_oppw(
            config,
            7,
            "REPORT",
            truth,
            signatures,
            "planted",
            0.35,
        )
        noisy_b, provenance_b = _observe_native_oppw(
            config,
            7,
            "REPORT",
            truth,
            signatures,
            "planted",
            0.35,
        )
        np.testing.assert_array_equal(noisy_a, noisy_b)
        self.assertEqual(provenance_a, provenance_b)

    def test_equal_channel_repeated_bit_is_deterministic_and_fail_closed(self):
        config = replace(
            self.config,
            lengths=(31,),
            carrier_families=("legendre",),
            layers=(8,),
            type_count=3,
        )
        truth = _trial_truth(config, 7, "REPORT", "planted", 5, 31)
        clean = _observe_and_decode_repeated_bit(
            config,
            7,
            "REPORT",
            truth,
            "planted",
            31,
            8,
            0.0,
        )
        self.assertEqual(clean["status"], "COMPLETED")
        self.assertEqual(clean["predicted_type"], truth["true_type"])
        noisy_a = _observe_and_decode_repeated_bit(
            config,
            7,
            "REPORT",
            truth,
            "planted",
            31,
            8,
            0.20,
        )
        noisy_b = _observe_and_decode_repeated_bit(
            config,
            7,
            "REPORT",
            truth,
            "planted",
            31,
            8,
            0.20,
        )
        self.assertEqual(noisy_a, noisy_b)
        self.assertEqual(
            noisy_a["observed_flip_rate"],
            noisy_a["observed_flip_count"] / (8 * 31),
        )
        contract = noisy_a["contract"]
        self.assertEqual(contract["physical_channel_uses"], 8 * 31)
        self.assertEqual(contract["transmitted_energy"], 8 * 31)
        self.assertIn("independent Bernoulli", contract["noise_channel"])
        repeated_resource = noisy_a["resource_accounting"]
        self.assertEqual(
            repeated_resource["standalone_estimated_peak_bytes"],
            sum(repeated_resource["component_breakdown"].values()),
        )
        self.assertFalse(repeated_resource["measured_process_peak"])
        unavailable = _observe_and_decode_repeated_bit(
            config,
            7,
            "REPORT",
            truth,
            "planted",
            31,
            1,
            0.20,
        )
        self.assertEqual(unavailable["status"], "STRUCTURALLY_UNAVAILABLE")
        self.assertFalse(
            unavailable.get("matched_resource_advantage_claim_eligible", False)
        )

    def test_control_context_guard_refuses_before_control_allocation(self):
        config = replace(
            self.config,
            lengths=(31,),
            carrier_families=("legendre",),
            layers=(8,),
            planted_mask_families=("typed16",),
            decoder_mask_policies=("typed16",),
            type_count=3,
        )
        signatures, _manifest = _phase_codebook(31, 8, 3, 7)
        templates, base, _provenance, ring_templates = (
            _build_carrier_templates(
                31,
                8,
                "legendre",
                signatures,
                7,
            )
        )
        payload_groups, _ = _build_payload_groups(7, 2, 8, 31)
        payload_bank = _materialize_payload_collision_bank(
            payload_groups,
            config.type_count,
        )
        reference_ffts = np.fft.rfft(templates, axis=2)
        reference_norms = np.linalg.norm(templates, axis=2)
        payload_ffts = np.fft.rfft(payload_bank, axis=2)
        payload_norms = np.linalg.norm(payload_bank, axis=2)
        resident = _unique_numpy_nbytes(
            signatures,
            templates,
            base,
            ring_templates,
            payload_groups,
            payload_bank,
            reference_ffts,
            reference_norms,
            payload_ffts,
            payload_norms,
        )
        with (
            patch(
                "run_prime_ring_waypoint_experiment."
                "CONTROL_HARD_MAX_ESTIMATED_BYTES",
                1,
            ),
            patch(
                "run_prime_ring_waypoint_experiment._observe_native_oppw"
            ) as native_observer,
            patch(
                "run_prime_ring_waypoint_experiment."
                "_observe_and_decode_repeated_bit"
            ) as repeated_observer,
        ):
            with self.assertRaises(CampaignValidationError):
                _run_base_group(
                    config,
                    seed=7,
                    length=31,
                    carrier_family="legendre",
                    layers=8,
                    planted_mask_family="typed16",
                    bit_flip_rate=0.35,
                    signatures=signatures,
                    templates=templates,
                    ring_templates=ring_templates,
                    payload_type_bank=payload_bank,
                    payload_ffts=payload_ffts,
                    payload_norms=payload_norms,
                    reference_ffts=reference_ffts,
                    reference_norms=reference_norms,
                    resident_context_bytes=resident,
                )
        native_observer.assert_not_called()
        repeated_observer.assert_not_called()

    def test_json_artifact_is_written_with_every_cell(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tempdir:
            output = Path(tempdir) / "artifact.json"
            artifact = run_campaign(
                replace(self.config, output=str(output)),
                write_artifact=True,
            )
            self.assertTrue(output.is_file())
            loaded = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(
                loaded["grid_manifest"]["retained_cell_count"],
                artifact["grid_manifest"]["retained_cell_count"],
            )
            self.assertFalse(loaded["promotion_eligible"])


class TestRawSanitySweepContract(unittest.TestCase):
    def test_raw_provenance_emission_is_full_campaign_only(self):
        config = _fabricated_full_contract_config()
        smoke_config = replace(
            config,
            run_label="SMOKE_NON_EVIDENTIARY",
        )
        spec = expected_cell_specs(config)[0]
        self.assertIsNone(
            _raw_type_accuracy_provenance(smoke_config, spec, ())
        )

    def test_exact_twelve_cell_sweep_passes_without_lower_rate_inference(self):
        config = _fabricated_full_contract_config()
        with patch(
            "run_prime_ring_waypoint_experiment.fit_select_thresholds",
            side_effect=AssertionError(
                "raw sanity aggregation must not refit thresholds"
            ),
        ):
            sweep = _fabricated_raw_sanity_sweep(
                config,
                _fabricated_raw_sanity_cells(config),
            )
        self.assertEqual(sweep["status"], "COMPLETE_PASS")
        self.assertTrue(sweep["configuration_contract_pass"])
        self.assertTrue(sweep["contract_complete"])
        self.assertTrue(sweep["gate_pass"])
        self.assertEqual(
            sweep["campaign_contract_digest"],
            _campaign_contract_manifest(config)["digest"],
        )
        self.assertEqual(sweep["required_cell_count"], 12)
        self.assertEqual(
            sweep["required_bit_flip_rates"],
            [0.0, 0.20, 0.35, 0.45],
        )
        self.assertEqual(
            sweep["expected_report_planted_groups_by_rate"],
            {"0.00": 3, "0.20": 3, "0.35": 3, "0.45": 1024},
        )
        self.assertTrue(
            all(
                summary["gate_pass"]
                for summary in sweep["rate_summaries"].values()
            )
        )
        self.assertFalse(sweep["lower_rates_inferred_from_q_0_45"])
        self.assertFalse(sweep["threshold_refit_performed"])
        self.assertFalse(sweep["promotion_eligible"])
        self.assertEqual(
            sweep["live_source_contract"]["estimated_retained_bytes"],
            12 * 2048,
        )
        self.assertEqual(
            sweep["live_source_contract"][
                "total_live_attestation_estimated_bytes"
            ],
            13 * 2048,
        )
        self.assertEqual(
            sweep["live_source_contract"]["authentication"],
            "per_run_hmac_sha256",
        )
        self.assertTrue(sweep["live_source_contract"]["authority_present"])
        self.assertFalse(
            sweep["live_source_contract"]["authority_secret_serialized"]
        )
        self.assertFalse(
            sweep["live_source_contract"]["serialized_provenance_alone_sufficient"]
        )
        self.assertFalse(
            sweep["live_source_contract"][
                "additional_full_report_rows_retained_for_live_seal"
            ]
        )
        self.assertTrue(
            sweep["live_source_contract"][
                "serialized_raw_provenance_rows_retained_in_cells"
            ]
        )
        self.assertFalse(
            sweep["live_source_contract"]["live_seals_serialized_into_artifact"]
        )

    def test_serialized_provenance_without_live_seals_fails_closed(self):
        config = _fabricated_full_contract_config()
        cells = _fabricated_raw_sanity_cells(config)
        sweep = _aggregate_raw_sanity_sweep(config, cells)
        self.assertFalse(sweep["configuration_contract_pass"])
        self.assertFalse(sweep["contract_complete"])
        self.assertFalse(sweep["gate_pass"])
        self.assertEqual(sweep["live_source_contract"]["seal_count"], 0)
        self.assertTrue(
            all(
                "live_source_seal_missing_or_invalid"
                in summary.get("invalid_reasons", ())
                for summary in sweep["cell_summaries"]
            )
        )

    def test_incomplete_duplicate_invalid_and_accuracy_failure_fail_closed(self):
        config = _fabricated_full_contract_config()
        baseline = _fabricated_raw_sanity_cells()

        missing = _fabricated_raw_sanity_sweep(config, baseline[:-1])
        self.assertEqual(missing["status"], "INCOMPLETE_FAIL_CLOSED")
        self.assertFalse(missing["contract_complete"])
        self.assertFalse(missing["gate_pass"])
        self.assertEqual(missing["missing_cell_keys"], ["seed=1337|q=0.45"])

        duplicate = _fabricated_raw_sanity_sweep(
            config,
            baseline + [json.loads(json.dumps(baseline[0]))],
        )
        self.assertEqual(duplicate["status"], "INCOMPLETE_FAIL_CLOSED")
        self.assertFalse(duplicate["contract_complete"])
        self.assertEqual(duplicate["duplicate_cell_keys"], ["seed=7|q=0.00"])

        duplicate_id_cells = json.loads(json.dumps(baseline))
        duplicate_id_cells[1]["cell_id"] = duplicate_id_cells[0]["cell_id"]
        duplicate_ids = _fabricated_raw_sanity_sweep(
            config,
            duplicate_id_cells,
        )
        self.assertEqual(duplicate_ids["status"], "INCOMPLETE_FAIL_CLOSED")
        self.assertFalse(duplicate_ids["contract_complete"])
        self.assertEqual(
            duplicate_ids["duplicate_cell_ids"],
            [baseline[0]["cell_id"]],
        )

        invalid_cells = json.loads(json.dumps(baseline))
        invalid_cells[0]["report"]["planted_group_count"] = 4
        invalid = _fabricated_raw_sanity_sweep(config, invalid_cells)
        self.assertEqual(invalid["status"], "INCOMPLETE_FAIL_CLOSED")
        self.assertFalse(invalid["gate_pass"])
        self.assertEqual(invalid["invalid_cell_keys"], ["seed=7|q=0.00"])

        failed_cells = _fabricated_raw_sanity_cells(
            config,
            failed_key=(42, 0.20),
        )
        failed = _fabricated_raw_sanity_sweep(
            config,
            failed_cells,
            failed_key=(42, 0.20),
        )
        self.assertEqual(failed["status"], "COMPLETE_RAW_SANITY_FAILURE")
        self.assertTrue(failed["contract_complete"])
        self.assertFalse(failed["gate_pass"])
        self.assertEqual(
            failed["failed_raw_accuracy_cell_keys"],
            ["seed=42|q=0.20"],
        )

        missing_rate_config = replace(
            config,
            bit_flip_rates=(0.0, 0.35, 0.45),
        )
        bad_config = _fabricated_raw_sanity_sweep(
            missing_rate_config,
            baseline,
        )
        self.assertFalse(bad_config["configuration_contract_pass"])
        self.assertFalse(bad_config["contract_complete"])
        self.assertFalse(bad_config["gate_pass"])

    def test_rejects_arbitrary_id_string_accuracy_and_nonexact_coordinates(self):
        config = _fabricated_full_contract_config()

        arbitrary_id = _fabricated_raw_sanity_cells(config)
        arbitrary_id[0]["cell_id"] = "ARBITRARY-CELL-ID"
        rejected_id = _fabricated_raw_sanity_sweep(config, arbitrary_id)
        self.assertFalse(rejected_id["contract_complete"])
        self.assertIn(
            "cell_id_not_canonical",
            rejected_id["cell_summaries"][0]["invalid_reasons"],
        )

        string_accuracy = _fabricated_raw_sanity_cells(config)
        string_accuracy[0]["report"]["raw_type_accuracy"] = "1.0"
        rejected_accuracy = _fabricated_raw_sanity_sweep(
            config,
            string_accuracy,
        )
        self.assertFalse(rejected_accuracy["contract_complete"])
        self.assertIn(
            "raw_type_accuracy_not_finite_float",
            rejected_accuracy["cell_summaries"][0]["invalid_reasons"],
        )

        float_seed = _fabricated_raw_sanity_cells(config)
        float_seed[0]["config"]["seed"] = 7.0
        rejected_seed = _fabricated_raw_sanity_sweep(config, float_seed)
        self.assertFalse(rejected_seed["contract_complete"])
        self.assertEqual(
            rejected_seed["missing_cell_keys"],
            ["seed=7|q=0.00"],
        )

        near_rate = _fabricated_raw_sanity_cells(config)
        near_rate[1]["config"]["bit_flip_rate"] = 0.2000000000001
        rejected_rate = _fabricated_raw_sanity_sweep(config, near_rate)
        self.assertFalse(rejected_rate["contract_complete"])
        self.assertEqual(
            rejected_rate["missing_cell_keys"],
            ["seed=7|q=0.20"],
        )

    def test_rejects_report_sourced_or_report_consuming_thresholds(self):
        config = _fabricated_full_contract_config()
        for field, value, expected_reason in (
            ("source_split", "REPORT", "threshold_source_not_select"),
            (
                "report_rows_consumed",
                1,
                "threshold_report_rows_consumed_not_zero",
            ),
        ):
            with self.subTest(field=field):
                cells = _fabricated_raw_sanity_cells(config)
                cells[0]["threshold"][field] = value
                sweep = _fabricated_raw_sanity_sweep(config, cells)
                self.assertFalse(sweep["contract_complete"])
                self.assertIn(
                    expected_reason,
                    sweep["cell_summaries"][0]["invalid_reasons"],
                )

    def test_campaign_contract_binds_evidence_config_and_stream_manifest(self):
        config = _fabricated_full_contract_config()
        baseline_digest = _campaign_contract_manifest(config)["digest"]
        self.assertEqual(
            _campaign_contract_manifest(
                replace(config, output="another/artifact.json")
            )["digest"],
            baseline_digest,
        )
        changed_stream_config = replace(
            config,
            report_stream_tag="REPORT-v2",
        )
        self.assertNotEqual(
            _campaign_contract_manifest(changed_stream_config)["digest"],
            baseline_digest,
        )
        self.assertNotEqual(
            _campaign_contract_manifest(
                replace(config, confidence_level=0.95)
            )["digest"],
            baseline_digest,
        )
        stale_cells = _fabricated_raw_sanity_cells(config)
        rejected = _fabricated_raw_sanity_sweep(
            changed_stream_config,
            stale_cells,
        )
        self.assertFalse(rejected["contract_complete"])
        self.assertTrue(
            all(
                "campaign_contract_digest_mismatch"
                in summary.get("invalid_reasons", ())
                for summary in rejected["cell_summaries"]
            )
        )

    def test_select_report_stream_collision_fails_configuration_contract(self):
        config = replace(
            _fabricated_full_contract_config(),
            report_stream_tag="SELECT",
        )
        cells, live_seals, live_authority = _fabricated_raw_sanity_bundle(
            config
        )
        sweep = _aggregate_raw_sanity_sweep(
            config,
            cells,
            live_seals=live_seals,
            live_authority=live_authority,
        )
        self.assertFalse(
            sweep["configuration_checks"]["select_report_streams_disjoint"]
        )
        self.assertFalse(sweep["configuration_contract_pass"])
        self.assertFalse(sweep["contract_complete"])
        self.assertFalse(sweep["gate_pass"])

    def test_rejects_raw_metric_provenance_and_accuracy_mutations(self):
        config = _fabricated_full_contract_config()
        mutations = (
            ("campaign_contract_digest", "0" * 64),
            ("correct_count", 2),
            ("group_digest", "1" * 64),
            ("row_digest", "2" * 64),
            ("threshold_applied", True),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                cells = _fabricated_raw_sanity_cells(config)
                provenance = cells[0]["report"][
                    "raw_type_accuracy_provenance"
                ]
                provenance[field] = value
                sweep = _fabricated_raw_sanity_sweep(config, cells)
                self.assertFalse(sweep["contract_complete"])
                self.assertEqual(
                    sweep["invalid_cell_keys"],
                    ["seed=7|q=0.00"],
                )

        row_mutation = _fabricated_raw_sanity_cells(config)
        row_mutation[0]["report"]["raw_type_accuracy_provenance"][
            "rows"
        ][0]["threshold_applied"] = True
        rejected_row = _fabricated_raw_sanity_sweep(
            config,
            row_mutation,
        )
        self.assertFalse(rejected_row["contract_complete"])
        self.assertIn(
            "raw_provenance_row_contract_mismatch",
            rejected_row["cell_summaries"][0]["invalid_reasons"],
        )

        forged_truth = _fabricated_raw_sanity_cells(config)
        forged_provenance = forged_truth[0]["report"][
            "raw_type_accuracy_provenance"
        ]
        forged_row = forged_provenance["rows"][0]
        replacement_type = (forged_row["true_type"] + 1) % config.type_count
        forged_row["true_type"] = replacement_type
        forged_row["predicted_type"] = replacement_type
        forged_provenance["row_digest"] = _stable_digest(
            forged_provenance["rows"]
        )
        rejected_truth = _fabricated_raw_sanity_sweep(
            config,
            forged_truth,
        )
        self.assertFalse(rejected_truth["contract_complete"])
        self.assertIn(
            "raw_provenance_true_type_not_canonical",
            rejected_truth["cell_summaries"][0]["invalid_reasons"],
        )

        inconsistent_accuracy = _fabricated_raw_sanity_cells(config)
        inconsistent_accuracy[0]["report"]["raw_type_accuracy"] = 0.5
        rejected_consistency = _fabricated_raw_sanity_sweep(
            config,
            inconsistent_accuracy,
        )
        self.assertFalse(rejected_consistency["contract_complete"])
        self.assertIn(
            "raw_type_accuracy_provenance_inconsistent",
            rejected_consistency["cell_summaries"][0]["invalid_reasons"],
        )

    def test_live_seal_rejects_self_attested_prediction_rewrite(self):
        config = _fabricated_full_contract_config()
        cells, live_seals, live_authority = _fabricated_raw_sanity_bundle(
            config,
            failed_key=(7, 0.0),
        )
        provenance = cells[0]["report"]["raw_type_accuracy_provenance"]
        failed_row = next(
            row for row in provenance["rows"] if not row["type_correct"]
        )
        failed_row["predicted_type"] = failed_row["true_type"]
        failed_row["type_correct"] = True
        provenance["correct_count"] = provenance["trial_count"]
        provenance["row_digest"] = _stable_digest(provenance["rows"])
        cells[0]["report"]["raw_type_accuracy"] = 1.0
        for cell in cells:
            cell_id = cell["cell_id"]
            serialized_provenance = cell["report"][
                "raw_type_accuracy_provenance"
            ]
            old_seal = live_seals[cell_id]
            forged_fields = {
                "schema_version": old_seal.schema_version,
                "authority_id": old_seal.authority_id,
                "cell_id": cell_id,
                "campaign_contract_digest": cell[
                    "campaign_contract_digest"
                ],
                "carrier_provenance_digest": _stable_digest(
                    cell["carrier_provenance"]
                ),
                "raw_provenance_digest": _stable_digest(
                    serialized_provenance
                ),
                "trial_count": serialized_provenance["trial_count"],
                "correct_count": serialized_provenance["correct_count"],
                "raw_type_accuracy": cell["report"]["raw_type_accuracy"],
            }
            live_seals[cell_id] = _RawSanityLiveSeal(
                **forged_fields,
                authentication_tag=_stable_digest(forged_fields),
            )

        rejected = _aggregate_raw_sanity_sweep(
            config,
            cells,
            live_seals=live_seals,
            live_authority=live_authority,
        )
        self.assertFalse(rejected["contract_complete"])
        self.assertFalse(rejected["gate_pass"])
        self.assertIn(
            "live_source_seal_authentication_failed",
            rejected["cell_summaries"][0]["invalid_reasons"],
        )

    def test_live_seal_rejects_missing_carrier_provenance(self):
        config = _fabricated_full_contract_config()
        cells, live_seals, live_authority = _fabricated_raw_sanity_bundle(
            config
        )
        cells[0].pop("carrier_provenance")
        rejected = _aggregate_raw_sanity_sweep(
            config,
            cells,
            live_seals=live_seals,
            live_authority=live_authority,
        )
        self.assertFalse(rejected["contract_complete"])
        self.assertIn(
            "carrier_provenance_missing",
            rejected["cell_summaries"][0]["invalid_reasons"],
        )

    def test_live_seals_are_bound_to_one_per_run_authority(self):
        config = _fabricated_full_contract_config()
        cells, live_seals, _ = _fabricated_raw_sanity_bundle(config)
        replacement_authority = _RawSanityLiveAuthority(
            _campaign_contract_manifest(config)["digest"]
        )
        rejected = _aggregate_raw_sanity_sweep(
            config,
            cells,
            live_seals=live_seals,
            live_authority=replacement_authority,
        )
        self.assertFalse(rejected["contract_complete"])
        self.assertFalse(rejected["gate_pass"])
        self.assertTrue(
            all(
                "live_source_seal_authentication_failed"
                in summary.get("invalid_reasons", ())
                for summary in rejected["cell_summaries"]
            )
        )

    def test_primary_controls_require_the_complete_sweep(self):
        config = _fabricated_full_contract_config()
        components = _fabricated_primary_components(config, correct=True)
        frozen_threshold = _fabricated_frozen_threshold()
        (
            complete_cells,
            complete_seals,
            complete_authority,
        ) = _fabricated_raw_sanity_bundle(config)
        with patch(
            "run_prime_ring_waypoint_experiment.fit_select_thresholds",
            return_value=frozen_threshold,
        ):
            omitted = _primary_aggregate(
                config,
                components,
                raw_sanity_cells=(),
            )
            complete = _primary_aggregate(
                config,
                components,
                raw_sanity_cells=complete_cells,
                raw_sanity_live_seals=complete_seals,
                raw_sanity_live_authority=complete_authority,
            )
        self.assertTrue(omitted["dense_raw_type_sanity_gate_pass"])
        self.assertTrue(
            omitted["native_sparse_oppw_resource_control_closed"]
        )
        self.assertTrue(
            omitted["equal_channel_use_repeated_bit_control_closed"]
        )
        self.assertFalse(
            omitted["through_q_0_45_raw_sanity_contract_complete"]
        )
        self.assertFalse(
            omitted["through_q_0_45_raw_sanity_gate_complete"]
        )
        self.assertFalse(omitted["dense_primary_control_closed"])
        self.assertFalse(omitted["all_mandatory_controls_closed"])
        self.assertTrue(
            complete["through_q_0_45_raw_sanity_contract_complete"]
        )
        self.assertTrue(
            complete["through_q_0_45_raw_sanity_gate_complete"]
        )
        self.assertTrue(complete["dense_primary_control_closed"])
        self.assertTrue(complete["all_mandatory_controls_closed"])
        self.assertFalse(complete["promotion_eligible"])


if __name__ == "__main__":
    unittest.main()
