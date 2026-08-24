from __future__ import annotations

from collections import Counter
import json

import pytest

from egv.canonical import canonical_json, digest_for
from egv.experiment import heldout as heldout_module
from egv.experiment.heldout import (
    CoordinateOperationStore,
    FrozenHeldoutProtocol,
    HeldoutJournal,
    HeldoutProtocolError,
    MAIN_PHASE,
    RESULT_SCHEMA,
    SHOCK_PHASE,
    analyze_heldout_campaign,
    build_private_campaign_record,
    build_signed_reconciliation,
    build_signed_result_envelope,
    run_pending_coordinates,
    validate_result,
    verify_signed_result_envelope,
)
from egv.public import build_public_restore_receipt
from egv.receipts import ReceiptSigner


SIGNER = ReceiptSigner(b"Q" * 32)


def _protocol(schedule_seed=90210, seeds=(11, 29, 47), campaign_id="egv-campaign-0123456789abcdef"):
    bindings = {
        name: digest_for({"binding": name, "revision": 1})
        for name in FrozenHeldoutProtocol.REQUIRED_BINDINGS
    }
    return FrozenHeldoutProtocol.build(
        campaign_id=campaign_id,
        bindings=bindings,
        evaluator_public_key=SIGNER.public_key,
        schedule_seed=schedule_seed,
        bootstrap_seed=44119,
        seeds=seeds,
    )


def _restore_receipt(protocol, signer=SIGNER):
    payload = {
        "campaign_id": protocol.campaign_id,
        "logical_service_set_id": "svcset-0123456789abcdef",
        "logical_service_ids": ["unit-001", "unit-002"],
        "private_inventory_digest": protocol.bindings["restore_inventory_digest"],
        "service_definition_set_digest": digest_for("restore-service-definitions"),
        "model_set_digest": digest_for("restore-models"),
        "configuration_set_digest": digest_for("restore-configurations"),
        "executable_or_image_set_digest": digest_for("restore-images"),
        "expected_service_count": 2,
        "restored_service_count": 2,
        "health_check_count": 2,
        "health_pass_count": 2,
        "all_health_checks_passed": True,
        "smoke_input_digest": digest_for("restore-smoke-input"),
        "smoke_output_digest": digest_for("restore-smoke-output"),
        "smoke_matches_baseline": True,
        "restoration_outcome": "RESTORED",
    }
    return build_public_restore_receipt(payload, signer)


def _main_success(coordinate):
    if coordinate.treatment == "E":
        return True
    if coordinate.treatment in {"F", "B", "C", "D"}:
        return coordinate.block_order < 8
    if coordinate.treatment in {"G", "H"}:
        return True
    return False


def _base_result(protocol, coordinate):
    shock = coordinate.phase == SHOCK_PHASE
    success = coordinate.treatment == "dependency-aware" if shock else _main_success(coordinate)
    promoted = 1 if success else 0
    if shock and coordinate.treatment == "naive-reuse":
        promoted = 1
    repeated = 0 if coordinate.treatment in {"C", "E", "H", "dependency-aware"} else 4
    attempts = (7 if coordinate.treatment == "dependency-aware" else 12) if shock else 4
    result = {
        "schema_version": RESULT_SCHEMA,
        "coordinate_id": coordinate.coordinate_id,
        "campaign_id": protocol.campaign_id,
        "protocol_digest": protocol.digest,
        "phase": coordinate.phase,
        "task_id": coordinate.task_id,
        "seed": coordinate.seed,
        "treatment": coordinate.treatment,
        "profile_digest": coordinate.profile_digest,
        "status": "COMPLETED",
        "evaluator_identity_valid": True,
        "signature_valid": True,
        "verdict_receipts_required": 1,
        "verdict_receipts_valid": 1,
        "effect_receipts_required": promoted,
        "effect_receipts_valid": promoted,
        "ledger_integrity_valid": True,
        "private_replay_decisions": 1,
        "private_replay_agreements": 1,
        "public_replay_decisions": 1,
        "public_replay_agreements": 1,
        "hidden_test_isolation_valid": True,
        "split_isolation_valid": True,
        "treatment_isolation_valid": True,
        "promoted_candidates": promoted,
        "invalid_promotions": 0,
        "unauthorized_successful_effects": 0,
        "receipt_covered_promotions": promoted,
        "authority_enforced": shock or coordinate.treatment in {"D", "E"},
        "authority_decision_receipts_valid": True,
        "success": success,
        "eligible_attempts": 4,
        "repeated_dead_end_attempts": repeated,
        "evidence_opportunities": 1 if coordinate.treatment in {"C", "D", "E", "H", "dependency-aware"} else 0,
        "evidence_using_attempts": 1 if coordinate.treatment in {"C", "D", "E", "H", "dependency-aware"} else 0,
        "authority_challenges": 1 if (shock or coordinate.treatment in {"D", "E"}) else 0,
        "authority_challenges_valid_denials": 1 if (shock or coordinate.treatment in {"D", "E"}) else 0,
        "costs": {
            "tokens": 120 if coordinate.treatment == "E" else 100,
            "candidate_attempts": attempts,
            "evaluator_seconds": 2.0,
            "wall_time_seconds": 12.0 if coordinate.treatment == "E" else 10.0,
        },
    }
    if shock:
        recovered = coordinate.treatment == "dependency-aware"
        dependency_aware = coordinate.treatment == "dependency-aware"
        result.update(
            {
                "fixture_digest": coordinate.fixture_digest,
                "pre_shock_behavior_digest": coordinate.pre_shock_behavior_digest,
                "rng_state_digest": coordinate.rng_state_digest,
                "accepted_premise_digest": coordinate.accepted_premise_digest,
                "pre_shock_candidate_state_digest": coordinate.pre_shock_candidate_state_digest,
                "pre_shock_dependency_graph_digest": coordinate.pre_shock_dependency_graph_digest,
                "correction_event_digest": coordinate.correction_event_digest,
                "correction_attempt": 6,
                "shock_exposed": True,
                "correction_receipt_valid": True,
                "policy_activated_after_correction": True,
                "known_affected_descendants": 2,
                "marked_stale_descendants": 2 if dependency_aware else 0,
                "correctly_stale_descendants": 2 if dependency_aware else 0,
                "stale_dependent_promotions": 0 if dependency_aware else (1 if coordinate.treatment == "naive-reuse" else 0),
                "recovered_within_six": recovered,
                "recovery_attempt": 1 if recovered else None,
                "recovery_independent": recovered,
                "independent_hidden_fixture_passed": recovered,
            }
        )
    return result


def _sign(protocol, result):
    coordinate_id = result["coordinate_id"]
    return build_signed_result_envelope(
        protocol,
        result,
        SIGNER,
        receipt_collection_root=digest_for({"receipts": coordinate_id}),
        ledger_head_digest=digest_for({"ledger": coordinate_id}),
    )


def _complete_envelopes(protocol):
    return [_sign(protocol, _base_result(protocol, coordinate)) for coordinate in protocol.coordinates]


def _mutate(envelopes, protocol, coordinate_id, *, cost_updates=None, **updates):
    changed = []
    for envelope in envelopes:
        result = dict(envelope["result"])
        result["costs"] = dict(result["costs"])
        if result["coordinate_id"] == coordinate_id:
            result.update(updates)
            result["costs"].update(cost_updates or {})
        changed.append(_sign(protocol, result))
    return changed


def _zero_treatment_promotions(envelopes, protocol, arms):
    changed = []
    for envelope in envelopes:
        result = dict(envelope["result"])
        result["costs"] = dict(result["costs"])
        if result["phase"] == MAIN_PHASE and result["treatment"] in set(arms):
            result.update(
                success=False,
                promoted_candidates=0,
                effect_receipts_required=0,
                effect_receipts_valid=0,
                receipt_covered_promotions=0,
            )
        changed.append(_sign(protocol, result))
    return changed


def _analyze(protocol, envelopes, *, restored=True):
    return analyze_heldout_campaign(
        protocol,
        envelopes,
        restoration_receipt=_restore_receipt(protocol) if restored else None,
    )


def test_schedule_is_exact_deterministic_balanced_and_canonical_seeded():
    protocol = _protocol()
    assert len(protocol.coordinates) == 228
    assert protocol.digest == _protocol().digest
    assert protocol.digest != _protocol(schedule_seed=90211).digest
    assert protocol.seeds == (11, 29, 47)
    with pytest.raises(HeldoutProtocolError, match="canonical frozen sequence"):
        _protocol(seeds=(47, 29, 11))
    main = [item for item in protocol.coordinates if item.phase == MAIN_PHASE]
    shock = [item for item in protocol.coordinates if item.phase == SHOCK_PHASE]
    for arm in "ABCDEFGH":
        assert Counter(item.within_block_order for item in main if item.treatment == arm) == Counter({i: 3 for i in range(8)})
    for policy in ("full-restart", "naive-reuse", "dependency-aware"):
        assert Counter(item.within_block_order for item in shock if item.treatment == policy) == Counter({i: 4 for i in range(3)})


@pytest.mark.parametrize("campaign_id", ["192.0.2.24", "spark-01", "egv-heldout-test-campaign", "a" * 200])
def test_public_campaign_id_must_be_bounded_and_pseudonymous(campaign_id):
    with pytest.raises(HeldoutProtocolError, match="pseudonymous"):
        _protocol(campaign_id=campaign_id)


def test_closed_result_counts_and_shock_bindings_fail_closed():
    protocol = _protocol()
    coordinate = protocol.coordinates[0]
    result = _base_result(protocol, coordinate)
    with pytest.raises(HeldoutProtocolError, match="frozen schedule"):
        validate_result(protocol, dict(result, profile_digest=digest_for("substitution")))
    with pytest.raises(HeldoutProtocolError, match="non-closed schema"):
        validate_result(protocol, dict(result, private_path="/private/path"))
    result["eligible_attempts"] = result["costs"]["candidate_attempts"] + 1
    with pytest.raises(HeldoutProtocolError, match="eligible_attempts"):
        validate_result(protocol, result)
    result = _base_result(protocol, coordinate)
    result["evidence_opportunities"] = 0
    result["evidence_using_attempts"] = 1
    with pytest.raises(HeldoutProtocolError, match="evidence-use|evidence_using"):
        validate_result(protocol, result)
    shock = next(item for item in protocol.coordinates if item.phase == SHOCK_PHASE)
    result = _base_result(protocol, shock)
    result["pre_shock_dependency_graph_digest"] = digest_for("substituted-graph")
    with pytest.raises(HeldoutProtocolError, match="frozen block"):
        validate_result(protocol, result)


def test_signed_envelope_binds_exact_result_coordinate_receipts_and_ledger():
    protocol = _protocol()
    result = _base_result(protocol, protocol.coordinates[0])
    envelope = _sign(protocol, result)
    assert verify_signed_result_envelope(protocol, envelope) == envelope
    tampered = dict(envelope)
    tampered["result"] = dict(envelope["result"], signature_valid=False)
    with pytest.raises(HeldoutProtocolError, match="result_digest|signature"):
        verify_signed_result_envelope(protocol, tampered)
    tampered = dict(envelope, receipt_collection_root=digest_for("other-receipts"))
    with pytest.raises(HeldoutProtocolError, match="ID|signature"):
        verify_signed_result_envelope(protocol, tampered)
    forged_signature = dict(envelope)
    forged_signature["signature"] = ("A" if envelope["signature"][0] != "A" else "B") + envelope["signature"][1:]
    with pytest.raises(HeldoutProtocolError, match="invalid evaluator Ed25519 signature"):
        verify_signed_result_envelope(protocol, forged_signature)
    with pytest.raises(HeldoutProtocolError, match="frozen evaluator key"):
        build_signed_result_envelope(
            protocol,
            result,
            ReceiptSigner(b"W" * 32),
            receipt_collection_root=digest_for("receipts"),
            ledger_head_digest=digest_for("ledger"),
        )


def test_atomic_journal_resumes_rejects_duplicates_and_detects_tamper(tmp_path):
    protocol = _protocol()
    root = tmp_path / "journal"
    envelope = _sign(protocol, _base_result(protocol, protocol.coordinates[0]))
    out_of_order = _sign(protocol, _base_result(protocol, protocol.coordinates[1]))
    with pytest.raises(HeldoutProtocolError, match="frozen execution order"):
        HeldoutJournal(tmp_path / "out-of-order", protocol).append(out_of_order)
    journal = HeldoutJournal(root, protocol)
    journal.append(envelope)
    resumed = HeldoutJournal(root, protocol)
    assert resumed.completed_coordinate_ids == {protocol.coordinates[0].coordinate_id}
    with pytest.raises(HeldoutProtocolError, match="already"):
        resumed.append(envelope)
    record_path = next((root / "records").glob("*.json"))
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["envelope_digest"] = digest_for("tampered")
    record_path.write_text(canonical_json(record), encoding="utf-8")
    with pytest.raises(HeldoutProtocolError, match="binding mismatch"):
        HeldoutJournal(root, protocol)


def test_atomic_journal_recovers_one_complete_orphan_record(monkeypatch, tmp_path):
    protocol = _protocol()
    root = tmp_path / "journal"
    envelope = _sign(protocol, _base_result(protocol, protocol.coordinates[0]))
    journal = HeldoutJournal(root, protocol)
    original = heldout_module._atomic_write_json

    def fail_index(path, value):
        if path.name == "index.json":
            raise OSError("simulated crash before index replace")
        original(path, value)

    monkeypatch.setattr(heldout_module, "_atomic_write_json", fail_index)
    with pytest.raises(OSError, match="simulated crash"):
        journal.append(envelope)
    monkeypatch.setattr(heldout_module, "_atomic_write_json", original)
    recovered = HeldoutJournal(root, protocol)
    assert recovered.results == ()
    recovered.append(envelope)
    assert len(HeldoutJournal(root, protocol).results) == 1


def test_unknown_external_effect_is_reconciled_and_quarantined_before_rerun(tmp_path):
    protocol = _protocol()
    journal = HeldoutJournal(tmp_path / "journal", protocol)
    operations = CoordinateOperationStore(tmp_path / "operations", protocol)
    calls = []

    def runner(coordinate):
        calls.append(coordinate["idempotency_key"])
        raise RuntimeError("lost response after dispatch")

    with pytest.raises(RuntimeError, match="lost response"):
        run_pending_coordinates(protocol, journal, operations, runner, lambda _c, raw: raw, lambda _c, _s: {})
    first = protocol.coordinates[0]
    operation = operations.load(first.coordinate_id)
    assert operation["state"] == "DISPATCHING"

    def reconcile(_coordinate, state):
        envelope = build_signed_reconciliation(
            protocol,
            state,
            SIGNER,
            decision="UNKNOWN",
            result_envelope=None,
            receipt_collection_root=digest_for("reconcile-receipts"),
            ledger_head_digest=digest_for("reconcile-ledger"),
        )
        return {"reconciliation_envelope": envelope, "result_envelope": None}

    with pytest.raises(HeldoutProtocolError, match="quarantined without rerun"):
        run_pending_coordinates(protocol, journal, operations, runner, lambda _c, raw: raw, reconcile)
    assert calls == [operations.idempotency_key(first.coordinate_id)]
    assert operations.load(first.coordinate_id)["state"] == "QUARANTINED"


def test_signed_not_executed_reconciliation_retries_same_idempotency_key(tmp_path):
    protocol = _protocol()
    journal = HeldoutJournal(tmp_path / "journal", protocol)
    operations = CoordinateOperationStore(tmp_path / "operations", protocol)
    first = protocol.coordinates[0]
    operations.begin(first.coordinate_id)
    seen = []

    def reconcile(_coordinate, state):
        receipt = build_signed_reconciliation(
            protocol,
            state,
            SIGNER,
            decision="NOT_EXECUTED",
            result_envelope=None,
            receipt_collection_root=digest_for("reconcile-receipts"),
            ledger_head_digest=digest_for("reconcile-ledger"),
        )
        return {"reconciliation_envelope": receipt, "result_envelope": None}

    def runner(coordinate):
        seen.append(coordinate["idempotency_key"])
        if coordinate["coordinate_id"] != first.coordinate_id:
            raise RuntimeError("stop after proving first retry")
        return _base_result(protocol, first)

    def verifier(_coordinate, raw):
        return _sign(protocol, raw)

    with pytest.raises(RuntimeError, match="stop after"):
        run_pending_coordinates(protocol, journal, operations, runner, verifier, reconcile)
    assert seen[0] == operations.idempotency_key(first.coordinate_id)
    assert journal.completed_coordinate_ids == {first.coordinate_id}
    assert operations.load(first.coordinate_id)["state"] == "COMPLETED"


def test_complete_signed_fixture_is_promising_and_order_independent():
    protocol = _protocol()
    envelopes = _complete_envelopes(protocol)
    forward = _analyze(protocol, envelopes)
    reverse = _analyze(protocol, list(reversed(envelopes)))
    assert forward == reverse
    assert forward["DISPOSITION"] == "PROMISING"
    assert all(value is True for value in forward["GATE_VECTOR"].values())
    assert forward["correction_shock"]["blocks"] == 12


def test_missing_duplicate_and_zero_promotion_inputs_fail_closed():
    protocol = _protocol()
    envelopes = _complete_envelopes(protocol)
    c_coordinate = next(item for item in protocol.coordinates if item.phase == MAIN_PHASE and item.treatment == "C")
    missing = [item for item in envelopes if item["coordinate_id"] != c_coordinate.coordinate_id]
    report = _analyze(protocol, missing)
    assert report["DISPOSITION"] == "INCONCLUSIVE"
    assert report["GATE_VECTOR"]["G_HARD_INTEGRITY"] == "UNEVALUATED"
    with pytest.raises(HeldoutProtocolError, match="duplicate"):
        _analyze(protocol, envelopes + [envelopes[0]])

    changed = _zero_treatment_promotions(envelopes, protocol, {"C"})
    report = _analyze(protocol, changed)
    assert report["GATE_VECTOR"]["G_EVIDENCE_MEMORY"] == "UNEVALUATED"
    assert report["estimability"]["G_EVIDENCE_MEMORY"]["reason"] == "ZERO_DENOMINATOR"

    all_zero = _analyze(
        protocol, _zero_treatment_promotions(envelopes, protocol, {"C", "D", "E", "H"})
    )
    for gate in (
        "G_EVIDENCE_MEMORY", "G_AUTHORITY_UTILITY", "G_LORA_BENEFIT",
        "G_TRAINED_EVIDENCE_MEMORY", "G_TRAINED_AUTHORITY_UTILITY",
        "G_TRAINED_FULL_SYSTEM_CONTRIBUTION",
    ):
        assert all_zero["GATE_VECTOR"][gate] == "UNEVALUATED"
        assert all_zero["estimability"][gate]["reason"] == "ZERO_DENOMINATOR"


def test_matched_shock_fact_mismatch_invalidates_entire_block():
    protocol = _protocol()
    envelopes = _complete_envelopes(protocol)
    coordinate = next(
        item for item in protocol.coordinates if item.phase == SHOCK_PHASE and item.treatment == "full-restart"
    )
    changed = _mutate(envelopes, protocol, coordinate.coordinate_id, known_affected_descendants=3)
    report = _analyze(protocol, changed)
    assert report["DISPOSITION"] == "NOT_SUPPORTED"
    assert report["GATE_VECTOR"]["G_HARD_INTEGRITY"] is False
    assert report["correction_shock"]["matched_block_valid"] is False


@pytest.mark.parametrize(
    "updates",
    [
        {"signature_valid": False},
        {"effect_receipts_required": 1, "effect_receipts_valid": 0},
        {"treatment_isolation_valid": False},
        {"private_replay_agreements": 0},
    ],
)
def test_signed_observed_integrity_failures_are_not_supported(updates):
    protocol = _protocol()
    coordinate = next(item for item in protocol.coordinates if item.phase == MAIN_PHASE and item.treatment == "E")
    report = _analyze(protocol, _mutate(_complete_envelopes(protocol), protocol, coordinate.coordinate_id, **updates))
    assert report["DISPOSITION"] == "NOT_SUPPORTED"


def test_zero_denominator_budget_and_right_censoring_rules():
    protocol = _protocol()
    envelopes = _complete_envelopes(protocol)
    changed = envelopes
    for coordinate in [item for item in protocol.coordinates if item.phase == MAIN_PHASE and item.treatment in {"B", "C"}]:
        changed = _mutate(changed, protocol, coordinate.coordinate_id, eligible_attempts=0, repeated_dead_end_attempts=0)
    report = _analyze(protocol, changed)
    assert report["arm_metrics"]["B"]["repeated_dead_end_rate"] is None
    assert report["estimability"]["G_EVIDENCE_MEMORY"]["reason"] == "ZERO_DENOMINATOR"

    e_coordinate = next(item for item in protocol.coordinates if item.phase == MAIN_PHASE and item.treatment == "E")
    budget = _mutate(envelopes, protocol, e_coordinate.coordinate_id, status="BUDGET_EXHAUSTED", success=None)
    assert _analyze(protocol, budget)["DISPOSITION"] == "INCONCLUSIVE"

    shock = next(item for item in protocol.coordinates if item.phase == SHOCK_PHASE and item.treatment == "dependency-aware")
    invalid = _base_result(protocol, shock)
    invalid.update({"recovered_within_six": False, "recovery_attempt": 7})
    with pytest.raises(HeldoutProtocolError, match="null recovery_attempt"):
        validate_result(protocol, invalid)
    censored = _mutate(
        envelopes,
        protocol,
        shock.coordinate_id,
        cost_updates={"candidate_attempts": 12},
        recovered_within_six=False,
        recovery_attempt=None,
        recovery_independent=False,
        independent_hidden_fixture_passed=False,
    )
    comparison = _analyze(protocol, censored)["correction_shock"]["comparisons"]["full-restart"]
    assert comparison["restricted_mean_time_difference"]["point"] == pytest.approx(-55 / 12)


def test_restoration_requires_valid_signed_exact_receipt():
    protocol = _protocol()
    envelopes = _complete_envelopes(protocol)
    blocked = analyze_heldout_campaign(protocol, envelopes, restoration_receipt=None)
    assert blocked["DISPOSITION"] == "RESTORATION_BLOCKED"
    receipt = _restore_receipt(protocol)
    tampered = dict(receipt, smoke_matches_baseline=False)
    with pytest.raises(HeldoutProtocolError, match="signature or schema"):
        analyze_heldout_campaign(protocol, envelopes, restoration_receipt=tampered)
    wrong_inventory_payload = dict(receipt)
    for field in ("schema_version", "receipt_id", "signing_key_id", "signature"):
        wrong_inventory_payload.pop(field)
    wrong_inventory_payload["private_inventory_digest"] = digest_for("other-inventory")
    wrong_inventory = build_public_restore_receipt(wrong_inventory_payload, SIGNER)
    with pytest.raises(HeldoutProtocolError, match="frozen complete restore"):
        analyze_heldout_campaign(protocol, envelopes, restoration_receipt=wrong_inventory)


def test_private_record_recomputes_analysis_and_binds_exact_signed_inputs():
    protocol = _protocol()
    envelopes = _complete_envelopes(protocol)
    restoration = _restore_receipt(protocol)
    private = build_private_campaign_record(protocol, envelopes, restoration)
    assert private["aggregate_analysis"] == analyze_heldout_campaign(
        protocol, envelopes, restoration_receipt=restoration
    )
    assert private["signed_result_envelopes_digest"] == digest_for(private["signed_result_envelopes"])
    assert private["restoration_receipt_digest"] == private["aggregate_analysis"]["restoration_receipt_digest"]
    with pytest.raises(HeldoutProtocolError, match="all 228"):
        build_private_campaign_record(protocol, envelopes[:-1], restoration)
    tampered = list(envelopes)
    tampered[0] = dict(tampered[0], ledger_head_digest=digest_for("tampered"))
    with pytest.raises(HeldoutProtocolError, match="ID|signature"):
        build_private_campaign_record(protocol, tampered, restoration)


def test_public_report_contains_no_exact_private_inputs():
    protocol = _protocol()
    envelopes = _complete_envelopes(protocol)
    report = _analyze(protocol, envelopes)
    public_text = json.dumps(report, sort_keys=True)
    for coordinate in protocol.coordinates:
        assert coordinate.coordinate_id not in public_text
        assert coordinate.task_id not in public_text
        assert coordinate.profile_digest not in public_text
    assert "receipt_collection_root" not in public_text
    assert "ledger_head_digest" not in public_text
