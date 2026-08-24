from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from egv.canonical import digest_for
from egv.evaluation.dataset import EvaluationCorpus
from egv.evaluation.shock import DependencyGraph, SHOCK_TASK_IDS
from egv.experiment.heldout import FrozenHeldoutProtocol, HeldoutProtocolError, SHOCK_PHASE
from egv.experiment.runtime import (
    HeldoutTrainerInputs, HeldoutTrainerSources, build_trainer_evidence_package,
    ordered_public_heldout_task_records,
)
from egv.experiment.shock_runtime import (
    CorrectionShockCoordinateRunner, PreShockState, ShockAttemptObservation, ShockRuntimeContext,
    ShockVerificationEvidence,
)
from egv.receipts import ReceiptSigner


def _protocol(corpus):
    bindings = {name: digest_for({"binding": name}) for name in FrozenHeldoutProtocol.REQUIRED_BINDINGS}
    signer = ReceiptSigner(b"T" * 32)
    return FrozenHeldoutProtocol.build(
        campaign_id="egv-campaign-fedcba0987654321", bindings=bindings,
        evaluator_public_key=signer.public_key, schedule_seed=81, bootstrap_seed=82,
        heldout_task_records=ordered_public_heldout_task_records(corpus),
    )


class _Engine:
    def __init__(
        self, ledger, rng_state_digest, recover_at=None, fail_post=None,
        cache=None, effects=None, state_candidate="candidate",
    ):
        self.ledger = ledger
        self.recover_at = recover_at
        self.pre_calls = []
        self.post_calls = []
        self.action = None
        self.fail_post = fail_post
        self.rng_state_digest = rng_state_digest
        self.cache = cache if cache is not None else {}
        self.effects = effects if effects is not None else []
        self.state_candidate = state_candidate

    @staticmethod
    def observation(key, promoted=False, passed=False):
        return ShockAttemptObservation(
            promoted, "PASS" if promoted else "TEST_FAILURE", True, 7, 0.1,
            True, True, True, ("candidate",) if promoted else tuple(), passed,
            digest_for({"verdict": key}), digest_for({"effect": key}) if promoted else None,
            key,
        )

    def pre_correction_attempt(self, attempt, *, idempotency_key):
        self.pre_calls.append(attempt)
        if idempotency_key in self.cache:
            return self.cache[idempotency_key]
        observation = self.observation(idempotency_key, promoted=attempt == 2)
        self.cache[idempotency_key] = observation
        self.effects.append(idempotency_key)
        return observation

    def freeze_pre_shock_state(self):
        return PreShockState(
            "premise", {"candidate_id": self.state_candidate, "attempt": 6},
            DependencyGraph((("premise", "candidate"), ("candidate", "descendant"))),
            self.rng_state_digest,
        )

    def restore_from_journal(self, journal):
        self.pre_calls = list(range(1, journal["pre_attempts"] + 1))
        self.post_calls = list(range(1, journal["post_attempts"] + 1))

    def commit_correction(self, correction_event_digest, *, idempotency_key):
        assert len(correction_event_digest) == 64
        assert len(idempotency_key) == 64

    def full_restart(self, *, idempotency_key): self.action = "full-restart"
    def reuse_without_invalidation(self, *, idempotency_key): self.action = "naive-reuse"

    def invalidate_dependencies(self, node_ids, *, idempotency_key):
        assert tuple(node_ids) == ("candidate", "descendant")
        self.action = "dependency-aware"

    def post_correction_attempt(self, attempt, *, idempotency_key):
        self.post_calls.append(attempt)
        if idempotency_key in self.cache:
            return self.cache[idempotency_key]
        if attempt == self.fail_post:
            observation = self.observation(idempotency_key, promoted=False)
            self.cache[idempotency_key] = observation
            self.effects.append(idempotency_key)
            raise RuntimeError("injected crash after effect")
        recovered = attempt == self.recover_at
        observation = self.observation(idempotency_key, promoted=recovered, passed=recovered)
        self.cache[idempotency_key] = observation
        self.effects.append(idempotency_key)
        return observation

    def reconcile_attempt(self, idempotency_key):
        return self.cache.get(idempotency_key)

    def ledger_head_hash(self): return self.ledger.ledger_head_hash()
    def ledger_integrity(self): return self.ledger.verify_integrity()

    def verification_evidence(self):
        count = len(self.pre_calls) + len(self.post_calls)
        return ShockVerificationEvidence(
            True, True, count, count, count, count, 0,
            digest_for("correction-receipt"), digest_for("policy-receipt"),
        )


class ShockRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self):
        self.temporary.cleanup()

    def run_runtime(self, treatment, recover_at=None):
        seed_file = self.root / (treatment + "-seed.bin")
        seed_file.write_bytes(b"U" * 32)
        corpus = EvaluationCorpus.generate(secret_seed_file=seed_file)
        protocol = _protocol(corpus)
        raw_inputs, raw_sources = build_trainer_evidence_package(
            corpus, protocol, generation_profile_digest=digest_for("generation")
        )
        inputs = HeldoutTrainerInputs(raw_inputs, protocol=protocol)
        sources = HeldoutTrainerSources(raw_sources, trainer_inputs=inputs)
        engines = []

        def factory(**kwargs):
            self.assertEqual(kwargs["base_model_digest"], protocol.bindings["base_model_digest"])
            self.assertEqual(kwargs["adapter_digest"], protocol.bindings["adapter_digest"])
            engine = _Engine(
                kwargs["ledger"], kwargs["coordinate"].rng_state_digest,
                recover_at=recover_at,
            )
            engines.append(engine)
            return engine

        context = ShockRuntimeContext(protocol, inputs, sources, self.root / treatment, factory)
        coordinate = next(
            item for item in protocol.coordinates
            if item.phase == SHOCK_PHASE and item.treatment == treatment
        )
        return CorrectionShockCoordinateRunner(context)(coordinate), engines[0]

    def test_shock_addresses_are_real_frozen_public_heldout_tasks(self):
        self.assertEqual(SHOCK_TASK_IDS, (
            "egv-data_transform-heldout-1-v1",
            "egv-dependency_contract-heldout-1-v1",
            "egv-parser_edge-heldout-1-v1",
            "egv-state_transition-heldout-1-v1",
        ))

    def test_promotions_require_pass_receipts_effect_and_node_binding(self):
        verdict = digest_for("verdict")
        effect = digest_for("effect")
        with self.assertRaisesRegex(HeldoutProtocolError, "verified PASS"):
            ShockAttemptObservation(
                True, "TEST_FAILURE", True, 1, 0.1, True, True, True,
                ("candidate",), False, verdict, effect, digest_for("operation"),
            ).validate()
        with self.assertRaisesRegex(HeldoutProtocolError, "node IDs"):
            ShockAttemptObservation(
                True, "PASS", True, 1, 0.1, True, True, True,
                tuple(), False, verdict, effect, digest_for("operation"),
            ).validate()

    def test_dependency_aware_runs_through_six_then_recovers(self):
        result, engine = self.run_runtime("dependency-aware", recover_at=2)
        self.assertEqual(engine.pre_calls, [1, 2, 3, 4, 5, 6])
        self.assertEqual(engine.post_calls, [1, 2])
        self.assertEqual(engine.action, "dependency-aware")
        self.assertEqual(result["costs"]["candidate_attempts"], 8)
        self.assertEqual(result["recovery_attempt"], 2)
        self.assertEqual(result["marked_stale_descendants"], 2)

    def test_restart_and_naive_reuse_consume_full_right_censored_window(self):
        for policy in ("full-restart", "naive-reuse"):
            with self.subTest(policy=policy):
                result, engine = self.run_runtime(policy)
                self.assertEqual(engine.pre_calls, [1, 2, 3, 4, 5, 6])
                self.assertEqual(engine.post_calls, [1, 2, 3, 4, 5, 6])
                self.assertEqual(engine.action, policy)
                self.assertEqual(result["costs"]["candidate_attempts"], 12)
                self.assertIs(result["recovered_within_six"], False)

    def test_shock_journal_resumes_after_post_correction_crash(self):
        seed_file = self.root / "resume-seed.bin"
        seed_file.write_bytes(b"V" * 32)
        corpus = EvaluationCorpus.generate(secret_seed_file=seed_file)
        protocol = _protocol(corpus)
        raw_inputs, raw_sources = build_trainer_evidence_package(
            corpus, protocol, generation_profile_digest=digest_for("generation")
        )
        inputs = HeldoutTrainerInputs(raw_inputs, protocol=protocol)
        sources = HeldoutTrainerSources(raw_sources, trainer_inputs=inputs)
        coordinate = next(
            item for item in protocol.coordinates
            if item.phase == SHOCK_PHASE and item.treatment == "dependency-aware"
        )
        calls = 0
        cache = {}
        effects = []

        def factory(**kwargs):
            nonlocal calls
            calls += 1
            return _Engine(
                kwargs["ledger"], kwargs["coordinate"].rng_state_digest,
                recover_at=4, fail_post=3 if calls == 1 else None,
                cache=cache, effects=effects,
            )

        context = ShockRuntimeContext(protocol, inputs, sources, self.root / "resume", factory)
        runner = CorrectionShockCoordinateRunner(context)
        with self.assertRaisesRegex(RuntimeError, "injected crash after effect"):
            runner(coordinate)
        result = runner(coordinate)
        self.assertEqual(result["recovery_attempt"], 4)
        self.assertEqual(result["costs"]["candidate_attempts"], 10)
        self.assertEqual(len(effects), len(set(effects)))

    def test_three_policy_clones_must_match_actual_pre_shock_snapshot(self):
        seed_file = self.root / "matched-seed.bin"
        seed_file.write_bytes(b"W" * 32)
        corpus = EvaluationCorpus.generate(secret_seed_file=seed_file)
        protocol = _protocol(corpus)
        raw_inputs, raw_sources = build_trainer_evidence_package(
            corpus, protocol, generation_profile_digest=digest_for("generation")
        )
        inputs = HeldoutTrainerInputs(raw_inputs, protocol=protocol)
        sources = HeldoutTrainerSources(raw_sources, trainer_inputs=inputs)
        cache = {}

        def factory(**kwargs):
            treatment = kwargs["coordinate"].treatment
            return _Engine(
                kwargs["ledger"], kwargs["coordinate"].rng_state_digest,
                cache=cache,
                state_candidate="substituted" if treatment == "naive-reuse" else "candidate",
            )

        context = ShockRuntimeContext(protocol, inputs, sources, self.root / "matched", factory)
        block = next(item.block_id for item in protocol.coordinates if item.phase == SHOCK_PHASE)
        coordinates = {
            item.treatment: item for item in protocol.coordinates
            if item.phase == SHOCK_PHASE and item.block_id == block
        }
        runner = CorrectionShockCoordinateRunner(context)
        runner(coordinates["dependency-aware"])
        with self.assertRaisesRegex(HeldoutProtocolError, "differs across policy clones"):
            runner(coordinates["naive-reuse"])

    def test_shock_context_rederives_frozen_task_record_commitment(self):
        seed_file = self.root / "shock-admission-seed.bin"
        seed_file.write_bytes(b"X" * 32)
        corpus = EvaluationCorpus.generate(secret_seed_file=seed_file)
        protocol = _protocol(corpus)
        raw_inputs, raw_sources = build_trainer_evidence_package(
            corpus, protocol, generation_profile_digest=digest_for("generation")
        )
        inputs = HeldoutTrainerInputs(raw_inputs, protocol=protocol)
        sources = HeldoutTrainerSources(raw_sources, trainer_inputs=inputs)
        original = protocol.heldout_task_records_digest
        object.__setattr__(protocol, "heldout_task_records_digest", digest_for("substituted"))
        with self.assertRaisesRegex(HeldoutProtocolError, "admission|reconstruction|commitment"):
            ShockRuntimeContext(
                protocol, inputs, sources, self.root / "shock-admission",
                lambda **kwargs: None,
            )
        object.__setattr__(protocol, "heldout_task_records_digest", original)


if __name__ == "__main__":
    unittest.main()
