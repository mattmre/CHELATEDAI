import json
import math
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from prime_ring_rb10_contract import (
    ArtifactEnvelope,
    Deadline,
    ExperimentBudget,
    HARD_MAX_ASSIGNMENTS,
    HARD_MAX_ESTIMATED_BYTES,
    HARD_MAX_FACTOR_ARITY,
    HARD_MAX_FACTORS,
    HARD_MAX_NODES,
    HARD_MAX_OUTPUT_BYTES,
    HARD_MAX_SECONDS,
    HARD_MAX_WORK_UNITS,
    RB10IntegrityError,
    RB10ResourceError,
    RB10ValidationError,
    RunContract,
    STATUS_BOUNDARY,
    atomic_write_json,
    canonical_json,
    canonical_json_bytes,
    deterministic_digest,
    preflight_experiment,
    validate_artifact_envelope,
)


def _budget(**overrides):
    values = {
        "max_estimated_bytes": 1024,
        "max_work_units": 10_000,
        "max_seconds": 1.0,
        "max_assignments": 1000,
        "max_nodes": 4,
        "max_factors": 4,
        "max_factor_arity": 3,
        "max_output_bytes": 4096,
    }
    values.update(overrides)
    return ExperimentBudget(**values)


def _estimate(
    budget=None,
    *,
    streaming_assignments_not_materialized=True,
):
    if budget is None:
        budget = _budget()
    return preflight_experiment(
        assignment_count=343,
        node_count=3,
        factor_arities=(2, 3),
        component_bytes=(
            ("inputs", 128),
            ("streamed_state", 64),
        ),
        component_work=(
            ("normalization", 100),
            ("enumeration", 2400),
        ),
        budget=budget,
        streaming_assignments_not_materialized=(streaming_assignments_not_materialized),
    )


def _run_contract(budget=None):
    if budget is None:
        budget = _budget()
    return RunContract.create(
        stage_id="RB10-G2-ALGEBRA",
        hypothesis_id="PRW-G2",
        decision_rule_id="DOUBLE-CENTER-AND-MOBIUS-v1",
        channel_id="BSC-Q20-v1",
        tie_policy_id="CANONICAL-FIRST-v1",
        control_ids=("independent", "flat", "global_phase"),
        seeds=(7, 42),
        parameters={
            "modulus": 7,
            "nodes": 3,
            "elapsed_seconds": "a preregistered field name, not timing",
        },
        budget=budget,
    )


def _envelope(result=None, status="COMPLETE", refusal_reasons=()):
    if result is None:
        result = {
            "decision": "MECHANISM_SURVIVES_ALGEBRA_ONLY",
            "elapsed_seconds": 0.125,
            "nested": {"route_latency_ms": 1.5, "rank": 2},
        }
    return ArtifactEnvelope.create(
        stage_id="RB10-G2-ALGEBRA",
        status=status,
        run_contract=_run_contract(),
        resource_estimate=_estimate(),
        result=result,
        limitations=("tiny exact synthetic domain only",),
        refusal_reasons=refusal_reasons,
    )


class HostileInt(int):
    pass


class Bomb:
    def __iter__(self):
        raise AssertionError("hostile object was iterated")

    def __len__(self):
        raise AssertionError("hostile object length was inspected")


class RB10BudgetTests(unittest.TestCase):
    def test_hard_envelope_matches_the_registered_protocol(self):
        self.assertEqual(HARD_MAX_ESTIMATED_BYTES, 256 * 1024 * 1024)
        self.assertEqual(HARD_MAX_WORK_UNITS, 25_000_000)
        self.assertEqual(HARD_MAX_SECONDS, 30.0)
        self.assertEqual(HARD_MAX_ASSIGNMENTS, 200_000)
        self.assertEqual(HARD_MAX_NODES, 6)
        self.assertEqual(HARD_MAX_FACTORS, 8)
        self.assertEqual(HARD_MAX_FACTOR_ARITY, 3)

    def test_caller_can_only_lower_limits_beneath_hard_caps(self):
        accepted = _budget()
        self.assertEqual(accepted.max_assignments, 1000)
        for field, value in (
            ("max_estimated_bytes", HARD_MAX_ESTIMATED_BYTES + 1),
            ("max_work_units", HARD_MAX_WORK_UNITS + 1),
            ("max_seconds", HARD_MAX_SECONDS + 0.001),
            ("max_assignments", HARD_MAX_ASSIGNMENTS + 1),
            ("max_nodes", HARD_MAX_NODES + 1),
            ("max_factors", HARD_MAX_FACTORS + 1),
            ("max_factor_arity", HARD_MAX_FACTOR_ARITY + 1),
            ("max_output_bytes", HARD_MAX_OUTPUT_BYTES + 1),
        ):
            with self.subTest(field=field):
                with self.assertRaises(RB10ValidationError):
                    ExperimentBudget(**{field: value})

    def test_plain_types_reject_bools_subclasses_nan_and_infinity(self):
        with self.assertRaises(RB10ValidationError):
            ExperimentBudget(max_assignments=True)
        with self.assertRaises(RB10ValidationError):
            ExperimentBudget(max_assignments=HostileInt(3))
        with self.assertRaises(RB10ValidationError):
            ExperimentBudget(max_seconds=float("nan"))
        with self.assertRaises(RB10ValidationError):
            ExperimentBudget(max_seconds=float("inf"))


class RB10PreflightTests(unittest.TestCase):
    def test_componentized_preflight_is_exact_and_nonpromotional(self):
        estimate = _estimate()
        self.assertEqual(estimate.assignment_count, 343)
        self.assertEqual(estimate.node_count, 3)
        self.assertEqual(estimate.factor_count, 2)
        self.assertEqual(estimate.maximum_factor_arity, 3)
        self.assertEqual(estimate.estimated_peak_bytes, 192)
        self.assertEqual(estimate.estimated_work_units, 2500)
        self.assertEqual(
            estimate.estimated_peak_bytes,
            sum(value for _name, value in estimate.component_bytes),
        )
        self.assertEqual(
            estimate.estimated_work_units,
            sum(value for _name, value in estimate.component_work),
        )
        self.assertTrue(estimate.streaming_assignments_not_materialized)
        self.assertFalse(estimate.measured_process_peak)
        self.assertFalse(estimate.estimate_is_process_rss)

    def test_oversized_assignment_refuses_before_container_inspection(self):
        budget = _budget(max_assignments=10)
        with patch("prime_ring_rb10_contract._normalize_factor_arities") as arity_normalizer:
            with patch("prime_ring_rb10_contract._normalize_components") as component_normalizer:
                with self.assertRaises(RB10ResourceError):
                    preflight_experiment(
                        assignment_count=11,
                        node_count=3,
                        factor_arities=Bomb(),
                        component_bytes=Bomb(),
                        component_work=Bomb(),
                        budget=budget,
                    )
        arity_normalizer.assert_not_called()
        component_normalizer.assert_not_called()

    def test_oversized_node_count_refuses_before_container_inspection(self):
        budget = _budget(max_nodes=2)
        with patch("prime_ring_rb10_contract._normalize_factor_arities") as arity_normalizer:
            with self.assertRaises(RB10ResourceError):
                preflight_experiment(
                    assignment_count=7,
                    node_count=3,
                    factor_arities=Bomb(),
                    component_bytes=Bomb(),
                    component_work=Bomb(),
                    budget=budget,
                )
        arity_normalizer.assert_not_called()

    def test_streaming_assignment_mode_is_explicit_and_strict(self):
        self.assertTrue(_estimate().streaming_assignments_not_materialized)
        self.assertTrue(_estimate(streaming_assignments_not_materialized=True).streaming_assignments_not_materialized)
        self.assertFalse(_estimate(streaming_assignments_not_materialized=False).streaming_assignments_not_materialized)
        for invalid in (1, HostileInt(1), "false", None):
            with self.subTest(invalid=invalid):
                with self.assertRaises(RB10ValidationError):
                    _estimate(streaming_assignments_not_materialized=invalid)

    def test_byte_and_work_refusals_are_separate_and_fail_closed(self):
        with self.assertRaisesRegex(
            RB10ResourceError,
            "byte budget",
        ):
            preflight_experiment(
                assignment_count=7,
                node_count=1,
                factor_arities=(),
                component_bytes=(("oversized", 11),),
                component_work=(("small", 1),),
                budget=_budget(max_estimated_bytes=10),
            )
        with self.assertRaisesRegex(
            RB10ResourceError,
            "work exceeds budget",
        ):
            preflight_experiment(
                assignment_count=7,
                node_count=1,
                factor_arities=(),
                component_bytes=(("small", 1),),
                component_work=(("oversized", 11),),
                budget=_budget(max_work_units=10),
            )

    def test_factor_count_arity_and_components_are_strict(self):
        with self.assertRaises(RB10ResourceError):
            preflight_experiment(
                assignment_count=7,
                node_count=3,
                factor_arities=(1, 1, 1),
                component_bytes=(("bytes", 1),),
                component_work=(("work", 1),),
                budget=_budget(max_factors=2),
            )
        with self.assertRaises(RB10ResourceError):
            preflight_experiment(
                assignment_count=7,
                node_count=3,
                factor_arities=(3,),
                component_bytes=(("bytes", 1),),
                component_work=(("work", 1),),
                budget=_budget(max_factor_arity=2),
            )
        with self.assertRaises(RB10ValidationError):
            preflight_experiment(
                assignment_count=7,
                node_count=3,
                factor_arities=[2],
                component_bytes=(("bytes", 1),),
                component_work=(("work", 1),),
                budget=_budget(),
            )
        with self.assertRaises(RB10ValidationError):
            preflight_experiment(
                assignment_count=7,
                node_count=3,
                factor_arities=(2,),
                component_bytes=(("same", 1), ("same", 2)),
                component_work=(("work", 1),),
                budget=_budget(),
            )


class RB10DeadlineTests(unittest.TestCase):
    def test_deadline_uses_monotonic_clock_and_fails_closed(self):
        with patch(
            "prime_ring_rb10_contract.time.monotonic",
            side_effect=(10.0, 10.5, 11.1),
        ):
            deadline = Deadline.start(_budget(max_seconds=1.0))
            deadline.check("accepted_step")
            with self.assertRaisesRegex(
                RB10ResourceError,
                "expired_step",
            ):
                deadline.check("expired_step")

    def test_manual_deadline_cannot_use_nan_or_mismatched_expiry(self):
        with self.assertRaises(RB10ValidationError):
            Deadline(float("nan"), 1.0, 1.0)
        with self.assertRaises(RB10ValidationError):
            Deadline(1.0, 3.0, 1.0)


class RB10DeterminismTests(unittest.TestCase):
    def test_canonical_json_is_order_independent_and_returns_same_bytes(self):
        left = {"b": [2, 1], "a": {"z": 3}}
        right = {"a": {"z": 3}, "b": [2, 1]}
        self.assertEqual(canonical_json(left), canonical_json(right))
        self.assertEqual(
            canonical_json_bytes(left),
            canonical_json(left).encode("utf-8"),
        )
        self.assertEqual(
            deterministic_digest(left),
            deterministic_digest(right),
        )

    def test_declared_timing_fields_do_not_change_digest(self):
        first = {
            "decision": "same",
            "elapsed_seconds": 1.0,
            "nested": {"route_latency_ms": 2.0},
        }
        second = {
            "decision": "same",
            "elapsed_seconds": 999.0,
            "nested": {"route_latency_ms": 1000.0},
        }
        self.assertEqual(
            deterministic_digest(first),
            deterministic_digest(second),
        )
        second["decision"] = "changed"
        self.assertNotEqual(
            deterministic_digest(first),
            deterministic_digest(second),
        )

    def test_scientific_fields_cannot_be_declared_nondeterministic(self):
        with self.assertRaises(RB10ValidationError):
            deterministic_digest(
                {"candidate_signal": True},
                excluded_fields=("candidate_signal",),
            )

    def test_cyclic_nan_and_custom_json_values_fail_closed(self):
        cyclic = {}
        cyclic["self"] = cyclic
        with self.assertRaises(RB10ValidationError):
            canonical_json(cyclic)
        with self.assertRaises(RB10ValidationError):
            canonical_json({"bad": math.nan})
        with self.assertRaises(RB10ValidationError):
            canonical_json({"bad": Bomb()})


class RB10ArtifactTests(unittest.TestCase):
    def test_run_contract_is_immutable_canonical_and_seed_checked(self):
        first = _run_contract()
        second = _run_contract()
        self.assertEqual(first, second)
        self.assertEqual(first.digest, second.digest)
        self.assertEqual(first.as_dict()["parameters"]["modulus"], 7)
        with self.assertRaises(RB10ValidationError):
            RunContract.create(
                stage_id="x",
                hypothesis_id="h",
                decision_rule_id="d",
                channel_id="c",
                tie_policy_id="t",
                control_ids=("control",),
                seeds=(7, 7),
                parameters={},
            )

    def test_artifact_round_trip_validates_contract_and_claim_boundaries(self):
        envelope = _envelope()
        payload = envelope.as_dict()
        self.assertTrue(validate_artifact_envelope(envelope))
        self.assertTrue(validate_artifact_envelope(payload))
        rebuilt = ArtifactEnvelope.from_dict(payload)
        self.assertEqual(rebuilt, envelope)
        self.assertFalse(payload["promotion_eligible"])
        self.assertFalse(payload["novelty_claim"])
        self.assertEqual(payload["status_boundary"], STATUS_BOUNDARY)
        self.assertFalse(payload["execution_evidence_claim"])
        self.assertEqual(payload["status"], "COMPLETE")
        self.assertEqual(
            payload["run_contract_digest"],
            _run_contract().digest,
        )

    def test_result_and_final_envelope_are_bounded_by_contract(self):
        oversized_result_budget = _budget(max_output_bytes=2048)
        with self.assertRaisesRegex(
            RB10ResourceError,
            "canonical result",
        ):
            ArtifactEnvelope.create(
                stage_id="RB10-G2-ALGEBRA",
                status="COMPLETE",
                run_contract=_run_contract(oversized_result_budget),
                resource_estimate=_estimate(oversized_result_budget),
                result={"large": "x" * 5000},
            )

        envelope_overhead_budget = _budget(max_output_bytes=1792)
        self.assertLess(
            len(canonical_json({"ok": True}, excluded_fields=()).encode("utf-8")),
            envelope_overhead_budget.max_output_bytes,
        )
        with self.assertRaisesRegex(
            RB10ResourceError,
            "final artifact envelope",
        ):
            ArtifactEnvelope.create(
                stage_id="RB10-G2-ALGEBRA",
                status="COMPLETE",
                run_contract=_run_contract(envelope_overhead_budget),
                resource_estimate=_estimate(envelope_overhead_budget),
                result={"ok": True},
            )

        admitted = _envelope()
        final_bytes = (
            len(
                canonical_json(
                    admitted.as_dict(),
                    excluded_fields=(),
                ).encode("utf-8")
            )
            + 1
        )
        self.assertLessEqual(
            final_bytes,
            admitted.resource_estimate.max_output_bytes,
        )

    def test_artifact_digest_excludes_only_declared_timing_fields(self):
        first = _envelope(
            {
                "decision": "same",
                "elapsed_seconds": 1.0,
                "nested": {"route_latency_ms": 2.0},
            }
        )
        second = _envelope(
            {
                "decision": "same",
                "elapsed_seconds": 500.0,
                "nested": {"route_latency_ms": 900.0},
            }
        )
        self.assertNotEqual(first.result_json, second.result_json)
        self.assertEqual(first.artifact_digest, second.artifact_digest)
        changed = _envelope(
            {
                "decision": "different",
                "elapsed_seconds": 1.0,
                "nested": {"route_latency_ms": 2.0},
            }
        )
        self.assertNotEqual(first.artifact_digest, changed.artifact_digest)

    def test_tampered_contract_result_resource_and_claims_fail_closed(self):
        payload = _envelope().as_dict()
        tampered_result = json.loads(json.dumps(payload))
        tampered_result["result"]["decision"] = "PROMOTE"
        with self.assertRaises(RB10IntegrityError):
            validate_artifact_envelope(tampered_result)

        tampered_contract = json.loads(json.dumps(payload))
        tampered_contract["run_contract"]["channel_id"] = "different"
        with self.assertRaises(RB10IntegrityError):
            validate_artifact_envelope(tampered_contract)

        tampered_resource = json.loads(json.dumps(payload))
        tampered_resource["resource_estimate"]["estimated_work_units"] += 1
        with self.assertRaises(RB10IntegrityError):
            validate_artifact_envelope(tampered_resource)

        tampered_claim = json.loads(json.dumps(payload))
        tampered_claim["novelty_claim"] = True
        with self.assertRaises(RB10IntegrityError):
            validate_artifact_envelope(tampered_claim)

        tampered_status_boundary = json.loads(json.dumps(payload))
        tampered_status_boundary["status_boundary"] = "COMPLETE_PROVES_EXECUTION"
        with self.assertRaises(RB10IntegrityError):
            validate_artifact_envelope(tampered_status_boundary)

        tampered_execution_claim = json.loads(json.dumps(payload))
        tampered_execution_claim["execution_evidence_claim"] = True
        with self.assertRaises(RB10IntegrityError):
            validate_artifact_envelope(tampered_execution_claim)

    def test_deserialized_artifact_cannot_lower_its_output_budget_below_payload(self):
        payload = _envelope().as_dict()
        payload["run_contract"]["budget"]["max_output_bytes"] = 1792
        payload["resource_estimate"]["max_output_bytes"] = 1792
        with self.assertRaises(RB10IntegrityError):
            ArtifactEnvelope.from_dict(payload)

    def test_refusal_status_and_reasons_are_consistent(self):
        refused = _envelope(
            result={"executed": False},
            status="REFUSED",
            refusal_reasons=("assignment budget exceeded",),
        )
        self.assertTrue(validate_artifact_envelope(refused))
        with self.assertRaises(RB10ValidationError):
            _envelope(result={}, status="REFUSED")
        with self.assertRaises(RB10ValidationError):
            _envelope(
                result={},
                status="COMPLETE",
                refusal_reasons=("not permitted",),
            )


class RB10AtomicWriterTests(unittest.TestCase):
    def test_atomic_writer_emits_bounded_valid_json_with_newline(self):
        envelope = _envelope()
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "nested" / "artifact.json"
            with patch(
                "prime_ring_rb10_contract.os.fsync",
                wraps=os.fsync,
            ) as fsync:
                atomic_write_json(
                    output,
                    envelope,
                    max_encoded_bytes=64 * 1024,
                )
            fsync.assert_called_once()
            encoded = output.read_bytes()
            self.assertTrue(encoded.endswith(b"\n"))
            self.assertLessEqual(len(encoded), 64 * 1024)
            self.assertLessEqual(
                len(encoded),
                envelope.resource_estimate.max_output_bytes,
            )
            loaded = json.loads(encoded.decode("utf-8"))
            self.assertTrue(validate_artifact_envelope(loaded))
            self.assertEqual(
                list(output.parent.glob(f".{output.name}.*.tmp")),
                [],
            )

    def test_writer_uses_stricter_caller_cap_and_preserves_target(self):
        envelope = _envelope()
        encoded_size = (
            len(
                canonical_json(
                    envelope.as_dict(),
                    excluded_fields=(),
                ).encode("utf-8")
            )
            + 1
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artifact.json"
            output.write_text("preserve-me", encoding="utf-8")
            with self.assertRaises(RB10ResourceError):
                atomic_write_json(
                    output,
                    envelope,
                    max_encoded_bytes=encoded_size - 1,
                )
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                "preserve-me",
            )
            self.assertEqual(
                list(output.parent.glob(f".{output.name}.*.tmp")),
                [],
            )

    def test_size_refusal_preserves_existing_target_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "artifact.json"
            output.write_text("preserve-me", encoding="utf-8")
            with self.assertRaises(RB10ResourceError):
                atomic_write_json(
                    output,
                    {"large": "x" * 5000},
                    max_encoded_bytes=100,
                )
            self.assertEqual(
                output.read_text(encoding="utf-8"),
                "preserve-me",
            )
            self.assertEqual(
                list(output.parent.glob(f".{output.name}.*.tmp")),
                [],
            )

    def test_expired_deadline_refuses_before_tempfile_or_replace(self):
        deadline = Deadline(1.0, 2.0, 1.0)
        with tempfile.TemporaryDirectory() as directory:
            with patch(
                "prime_ring_rb10_contract.time.monotonic",
                return_value=3.0,
            ):
                with patch("prime_ring_rb10_contract.tempfile.NamedTemporaryFile") as temporary:
                    with patch("prime_ring_rb10_contract.os.replace") as replace:
                        with self.assertRaises(RB10ResourceError):
                            atomic_write_json(
                                Path(directory) / "artifact.json",
                                {"value": 1},
                                deadline=deadline,
                            )
        temporary.assert_not_called()
        replace.assert_not_called()

    def test_invalid_writer_inputs_fail_before_filesystem_work(self):
        with patch("prime_ring_rb10_contract.tempfile.NamedTemporaryFile") as temporary:
            with self.assertRaises(RB10ValidationError):
                atomic_write_json(Bomb(), {"value": 1})
            with self.assertRaises(RB10ValidationError):
                atomic_write_json(
                    "artifact.json",
                    {"value": 1},
                    max_encoded_bytes=HARD_MAX_OUTPUT_BYTES + 1,
                )
        temporary.assert_not_called()


if __name__ == "__main__":
    unittest.main()
