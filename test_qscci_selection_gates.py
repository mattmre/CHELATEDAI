import math
import inspect
import unittest

import qscci_experiment as qscci

from qscci_experiment import (
    QSCCIError,
    build_feature_records,
    choose_cell,
    compute_cell_endpoints,
    canonical_json,
    evaluate_disposition,
    run_worker,
    sha256_bytes,
)


def _select_activation_rows():
    rows = []
    for index in range(6):
        row_id = f"S{index + 1:02d}"
        y = 1 if index % 2 == 0 else -1
        canonical = [(feature_id + 2.0) * y if feature_id < 8 else 0.0 for feature_id in range(100)]
        values = {
            "canonical": canonical,
            "nuisance": [value - 0.5 if feature_id < 8 else value for feature_id, value in enumerate(canonical)],
            "material": [-value if feature_id < 8 else value for feature_id, value in enumerate(canonical)],
        }
        for variant in ("canonical", "nuisance", "material"):
            feature_ids = sorted(range(100), key=lambda feature_id: (-values[variant][feature_id], feature_id))
            body = {
                "prompt_id": f"{row_id}:{variant}",
                "feature_ids": feature_ids,
                "values": [values[variant][feature_id] for feature_id in feature_ids],
            }
            rows.append({**body, "row_sha256": sha256_bytes(canonical_json(body))})
    return rows


def _cell(k, alpha, *, contrast=0.2, mismatch=0.1, mean_kl=0.01):
    return {
        "cell_key": f"SELECT|grid|chelated|null|{k}|{alpha:g}",
        "split": "SELECT",
        "operating_point": "grid",
        "family": "chelated",
        "seed": None,
        "k": k,
        "alpha": alpha,
        "feature_ids": list(range(k)),
        "endpoints": {
            "bidirectional_causal_contrast": contrast,
            "canonical_nuisance_mismatch": mismatch,
            "mean_kl": mean_kl,
        },
    }


def _prompt_records():
    records = []
    target_ids = {1: 6572, -1: 7968}
    for row_index in range(6):
        row_id = f"S{row_index + 1:02d}"
        row_y = 1 if row_index % 2 == 0 else -1
        for variant in ("canonical", "nuisance", "material"):
            y = -row_y if variant == "material" else row_y
            for intervention_sign, gain in (("correct", 0.1), ("wrong", -0.1)):
                baseline_margin = 0.25 * y
                intervened_margin = (0.25 + gain) * y
                records.append(
                    {
                        "row_id": row_id,
                        "kind": variant,
                        "y": y,
                        "prompt_id": f"{row_id}:{variant}",
                        "intervention_sign": 1 if intervention_sign == "correct" else -1,
                        "baseline_margin": baseline_margin,
                        "intervened_margin": intervened_margin,
                        "gain": y * (intervened_margin - baseline_margin),
                        "kl": 0.01,
                        "baseline_top1_token_id": target_ids[y],
                        "intervened_top1_token_id": target_ids[y],
                        "shared_delta_norm": 0.5,
                        "cast_delta_norm": 0.5,
                        "realized_update_norm": 0.5,
                    }
                )
    kind_order = {"canonical": 0, "nuisance": 1, "material": 2}
    return sorted(
        records,
        key=lambda record: (
            0 if record["intervention_sign"] == 1 else 1,
            record["row_id"],
            kind_order[record["kind"]],
        ),
    )


def _passing_disposition_payload():
    return {
        "failures": [],
        "integrity_valid": True,
        "resources_valid": True,
        "service_lifecycle_valid": True,
        "gates": {
            "select_baseline_accuracy_at_least_0_75": True,
            "report_baseline_accuracy_at_least_0_75": True,
            "correct_positive_wrong_negative": True,
            "contrastive_advantage_at_least_0_05": True,
            "all_random_advantages_at_least_0_05": True,
            "mismatch_no_greater_than_controls": True,
            "material_completeness_at_least_5_6": True,
            "material_completeness_no_lower_than_controls": True,
            "mean_kl_at_most_0_05": True,
            "max_kl_at_most_0_25": True,
            "collateral_fraction_at_most_1_6": True,
            "treatment_separation": {"contrastive|null": True, "uniform_random|1701": True},
        },
    }


class TestQSCCIFeatureRecords(unittest.TestCase):
    def test_feature_records_reconstruct_signed_material_and_nuisance_scores(self):
        records = build_feature_records(_select_activation_rows(), width=32768)
        self.assertEqual(len(records), 32768)
        self.assertEqual([record["feature_id"] for record in records], list(range(32768)))
        feature = records[0]
        self.assertEqual(feature["active_count"], 12)
        self.assertEqual(feature["contrast"], 4.0)
        self.assertEqual(feature["material_score"], 4.0)
        self.assertEqual(feature["nuisance_score"], 0.5)
        self.assertEqual(feature["chelated_score"], 3.5)
        self.assertEqual(feature["sign"], 1)
        self.assertTrue(feature["eligible"])

    def test_exact_zero_contrast_is_ineligible_not_signed_positive(self):
        feature = build_feature_records(_select_activation_rows(), width=32768)[8]
        self.assertEqual(feature["contrast"], 0.0)
        self.assertFalse(feature["eligible"])
        self.assertEqual(feature["sign"], 0)
        self.assertEqual(feature["exclusion_reason"], "ZERO_CONTRAST")

    def test_build_feature_records_rejects_report_leakage(self):
        rows = _select_activation_rows()
        body = {"prompt_id": "R01:canonical", "feature_ids": rows[0]["feature_ids"], "values": rows[0]["values"]}
        rows[0] = {**body, "row_sha256": sha256_bytes(canonical_json(body))}
        with self.assertRaises(QSCCIError):
            build_feature_records(rows, width=32768)

    def test_build_feature_records_rejects_duplicate_missing_unsorted_and_nonfinite_sparse_rows(self):
        mutations = []
        duplicate = _select_activation_rows()
        duplicate[1] = duplicate[0]
        mutations.append(duplicate)
        missing = _select_activation_rows()[:-1]
        mutations.append(missing)
        digest_tamper = _select_activation_rows()
        digest_tamper[0]["values"][0] += 1.0
        mutations.append(digest_tamper)
        nonfinite = _select_activation_rows()
        nonfinite[0]["values"][0] = math.nan
        mutations.append(nonfinite)
        for rows in mutations:
            with self.subTest(count=len(rows)), self.assertRaises(QSCCIError):
                build_feature_records(rows, width=32768)


class TestQSCCICellChoice(unittest.TestCase):
    def test_choose_cell_uses_frozen_lexicographic_tuning_order(self):
        cells = [_cell(k, alpha) for k in (1, 4, 8) for alpha in (0.5, 1, 2, 4)]
        cells[-1] = _cell(8, 4, contrast=0.3, mismatch=0.9, mean_kl=0.9)
        chosen = choose_cell(cells)
        self.assertEqual((chosen["k"], chosen["alpha"]), (8, 4))

    def test_choose_cell_breaks_complete_tie_by_smaller_alpha_then_k(self):
        cells = [_cell(k, alpha) for k in (1, 4, 8) for alpha in (0.5, 1, 2, 4)]
        chosen = choose_cell(cells)
        self.assertEqual((chosen["k"], chosen["alpha"]), (1, 0.5))

    def test_choose_cell_quantizes_fully_computed_keys_before_comparison(self):
        cells = [_cell(k, alpha, contrast=0.2) for k in (1, 4, 8) for alpha in (0.5, 1, 2, 4)]
        cells[0]["endpoints"]["bidirectional_causal_contrast"] = 0.20000000040
        cells[1]["endpoints"]["bidirectional_causal_contrast"] = 0.20000000039
        chosen = choose_cell(cells)
        self.assertEqual((chosen["k"], chosen["alpha"]), (1, 0.5))

    def test_choose_cell_rejects_missing_duplicate_or_nonfinite_grid_cell(self):
        cells = [_cell(k, alpha) for k in (1, 4, 8) for alpha in (0.5, 1, 2, 4)]
        cases = (cells[:-1], cells[:-1] + [dict(cells[0])], cells[:-1] + [_cell(8, 4, mean_kl=math.nan)])
        for case in cases:
            with self.subTest(count=len(case)), self.assertRaises(QSCCIError):
                choose_cell(case)


class TestQSCCICellEndpoints(unittest.TestCase):
    def test_endpoints_recompute_bidirectional_material_and_collateral_metrics(self):
        endpoints = compute_cell_endpoints(_prompt_records())
        self.assertAlmostEqual(endpoints["mean_correct_gain"], 0.1)
        self.assertAlmostEqual(endpoints["mean_wrong_gain"], -0.1)
        self.assertAlmostEqual(endpoints["bidirectional_causal_contrast"], 0.2)
        self.assertAlmostEqual(endpoints["canonical_nuisance_mismatch"], 0.0)
        self.assertEqual(endpoints["material_response_completeness"], 1.0)
        self.assertAlmostEqual(endpoints["mean_kl"], 0.01)
        self.assertAlmostEqual(endpoints["max_kl"], 0.01)
        self.assertEqual(endpoints["outside_target_collateral_fraction"], 0.0)
        self.assertEqual(endpoints["baseline_accuracy"], 1.0)

    def test_endpoints_reject_missing_pair_duplicate_nonfinite_and_vanished_update(self):
        records = _prompt_records()
        cases = []
        cases.append(records[:-1])
        cases.append(records + [dict(records[0])])
        nonfinite = [dict(record) for record in records]
        nonfinite[0]["kl"] = math.inf
        cases.append(nonfinite)
        vanished = [dict(record) for record in records]
        vanished[0]["realized_update_norm"] = 0.0
        cases.append(vanished)
        sub_quantum_gain_forgery = [dict(record) for record in records]
        sub_quantum_gain_forgery[0]["gain"] += 1e-10
        cases.append(sub_quantum_gain_forgery)
        negative_shared_norm = [dict(record, shared_delta_norm=-1.0) for record in records]
        cases.append(negative_shared_norm)
        inconsistent_shared_norm = [dict(record) for record in records]
        inconsistent_shared_norm[0]["shared_delta_norm"] = 0.6
        cases.append(inconsistent_shared_norm)
        inconsistent_cast_norm = [dict(record) for record in records]
        inconsistent_cast_norm[0]["cast_delta_norm"] = 0.6
        cases.append(inconsistent_cast_norm)
        cases.append(list(reversed(records)))
        for case in cases:
            with self.subTest(count=len(case)), self.assertRaises(QSCCIError):
                compute_cell_endpoints(case)


class TestQSCCIDisposition(unittest.TestCase):
    def test_all_inclusive_boundary_gates_survive(self):
        payload = _passing_disposition_payload()
        self.assertEqual(evaluate_disposition(payload), "SURVIVES_SMALL_LABEL_ORACLE_FIXTURE")

    def test_integrity_failure_has_precedence_over_task_and_scientific_gates(self):
        payload = _passing_disposition_payload()
        payload["failures"] = [{"code": "RESTORATION_FAILED"}]
        payload["gates"]["select_baseline_accuracy_at_least_0_75"] = False
        payload["gates"]["correct_positive_wrong_negative"] = False
        self.assertEqual(evaluate_disposition(payload), "INVALID_RUN")

    def test_baseline_failure_has_precedence_over_scientific_failure(self):
        payload = _passing_disposition_payload()
        payload["gates"]["report_baseline_accuracy_at_least_0_75"] = False
        payload["gates"]["correct_positive_wrong_negative"] = False
        self.assertEqual(evaluate_disposition(payload), "INVALID_TASK")

    def test_dead_or_indiscriminately_disruptive_direction_does_not_survive(self):
        failed_gates = (
            "correct_positive_wrong_negative",
            "mean_kl_at_most_0_05",
            "collateral_fraction_at_most_1_6",
        )
        for gate in failed_gates:
            with self.subTest(gate=gate):
                payload = _passing_disposition_payload()
                payload["gates"][gate] = False
                self.assertEqual(
                    evaluate_disposition(payload), "DOES_NOT_SURVIVE_SMALL_LABEL_ORACLE_FIXTURE"
                )

    def test_no_treatment_separation_cannot_be_rescued_by_passing_gates(self):
        payload = _passing_disposition_payload()
        payload["gates"]["treatment_separation"]["contrastive|null"] = False
        self.assertEqual(evaluate_disposition(payload), "DOES_NOT_SURVIVE_SMALL_LABEL_ORACLE_FIXTURE")

    def test_malformed_or_nonboolean_gate_is_invalid_run(self):
        for gate, value in (("mean_kl_at_most_0_05", math.nan), ("max_kl_at_most_0_25", 1)):
            with self.subTest(gate=gate, value=value):
                payload = _passing_disposition_payload()
                payload["gates"][gate] = value
                self.assertEqual(evaluate_disposition(payload), "INVALID_RUN")


class TestQSCCIProducerContract(unittest.TestCase):
    def test_producer_verifier_and_gates_share_one_cosine_canonicalizer(self):
        for function in (qscci.run_worker, qscci._verify_artifact_impl, qscci.compute_gates):
            with self.subTest(function=function.__name__):
                self.assertIn("canonicalize_direction_cosine(", inspect.getsource(function))
        self.assertIn(
            "require_pinned_sae_provenance=True",
            inspect.getsource(qscci.verify_artifact),
        )

    def test_worker_excludes_chelated_candidate_from_control_cosine_map(self):
        source = inspect.getsource(run_worker)
        loop = source.index("for cell in primary.values():")
        assignment = source.index('candidate_cell["direction_cosines"]', loop)
        exclusion = source.index("if cell is candidate_cell:", loop, assignment)
        self.assertLess(exclusion, assignment)


if __name__ == "__main__":
    unittest.main()
