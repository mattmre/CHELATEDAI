from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from egv.canonical import digest_for
from egv.evaluation.dataset import EvaluationCorpus
from egv.experiment.heldout import FrozenHeldoutProtocol, HeldoutProtocolError, MAIN_PHASE
from egv.experiment.runtime import (
    HeldoutRuntimeContext, HeldoutRuntimeEvidence, HeldoutTrainerInputs, HeldoutTrainerSources,
    build_coordinate_public_projection, build_trainer_evidence_package,
    derive_verified_main_result, ordered_public_heldout_task_records,
)
from egv.receipts import ReceiptSigner
from egv.variation.loop import AttemptRecord, VariationReport


def _protocol(corpus):
    bindings = {name: digest_for({"binding": name}) for name in FrozenHeldoutProtocol.REQUIRED_BINDINGS}
    signer = ReceiptSigner(b"R" * 32)
    return FrozenHeldoutProtocol.build(
        campaign_id="egv-campaign-1234567890abcdef", bindings=bindings,
        evaluator_public_key=signer.public_key, schedule_seed=91, bootstrap_seed=92,
        heldout_task_records=ordered_public_heldout_task_records(corpus),
    )


class HeldoutRuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.seed = self.root / "seed.bin"
        self.seed.write_bytes(b"S" * 32)
        self.corpus = EvaluationCorpus.generate(secret_seed_file=self.seed)
        self.protocol = _protocol(self.corpus)

    def tearDown(self):
        self.temporary.cleanup()

    def package(self):
        raw_inputs, raw_sources = build_trainer_evidence_package(
            self.corpus, self.protocol, generation_profile_digest=digest_for("generation-profile")
        )
        inputs = HeldoutTrainerInputs(raw_inputs, protocol=self.protocol)
        sources = HeldoutTrainerSources(raw_sources, trainer_inputs=inputs)
        return raw_inputs, raw_sources, inputs, sources

    def report(self, coordinate):
        attempt = AttemptRecord(
            1, "candidate", digest_for("candidate"), digest_for("retrieval"), tuple(),
            "PASS", "SMALL", "PROMOTED", ("receipt",), digest_for("ledger-head"),
        )
        return VariationReport(
            self.protocol.campaign_id, "run", coordinate.treatment, coordinate.task_id,
            coordinate.seed, (attempt,), "PROMOTED", "checkpoint-abc.json",
            digest_for("ledger-head"), {"chain_valid": True},
            self.protocol.bindings["base_model_digest"], None, "SUCCESS_ONLY", False,
        )

    def test_public_package_is_exact_two_file_heldout_source_contract(self):
        raw_inputs, raw_sources, inputs, sources = self.package()
        self.assertEqual(raw_inputs["task_count"], 8)
        self.assertEqual(tuple(inputs.tasks), self.protocol.heldout_task_ids)
        self.assertTrue(all(
            [item["path"] for item in record["source_files"]] == ["README.md", "src/task.py"]
            for record in raw_sources["tasks"]
        ))
        self.assertTrue(all(sources.source_for(task_id) for task_id in self.protocol.heldout_task_ids))
        serialized = repr((raw_inputs, raw_sources))
        self.assertNotIn("hidden_spec", serialized)
        self.assertNotIn("expected_output", serialized)
        self.assertNotIn("golden_patch", serialized)

    def test_source_substitution_fails_even_if_outer_bundle_is_redigested(self):
        raw_inputs, raw_sources, _, _ = self.package()
        raw_sources["tasks"][0]["source_files"][1]["content_utf8"] += "\n# substituted\n"
        raw_sources.pop("trainer_sources_digest")
        raw_sources["trainer_sources_digest"] = digest_for(raw_sources)
        raw_inputs["trainer_sources_digest"] = raw_sources["trainer_sources_digest"]
        raw_inputs.pop("trainer_inputs_digest")
        raw_inputs["trainer_inputs_digest"] = digest_for(raw_inputs)
        inputs = HeldoutTrainerInputs(raw_inputs, protocol=self.protocol)
        with self.assertRaisesRegex(HeldoutProtocolError, "source bytes differ"):
            HeldoutTrainerSources(raw_sources, trainer_inputs=inputs)

    def test_fully_redigested_task_and_source_substitution_cannot_cross_protocol(self):
        raw_inputs, raw_sources, _, _ = self.package()
        replacement = raw_sources["tasks"][0]
        replacement["source_files"][1]["content_utf8"] += "\n# fully-redigested-substitution\n"
        source_map = {
            item["path"]: item["content_utf8"] for item in replacement["source_files"]
        }
        replacement_digest = digest_for(source_map)
        replacement["repository_source_digest"] = replacement_digest
        raw_sources.pop("trainer_sources_digest")
        raw_sources["trainer_sources_digest"] = digest_for(raw_sources)
        raw_inputs["tasks"][0]["source_digest"] = replacement_digest
        raw_inputs["trainer_sources_digest"] = raw_sources["trainer_sources_digest"]
        raw_inputs.pop("trainer_inputs_digest")
        raw_inputs["trainer_inputs_digest"] = digest_for(raw_inputs)
        with self.assertRaisesRegex(HeldoutProtocolError, "substituted"):
            HeldoutTrainerInputs(raw_inputs, protocol=self.protocol)

    def test_nested_task_scalars_are_strictly_typed_and_bounded(self):
        raw_inputs, _, _, _ = self.package()
        cases = (
            ("ordinal", {"not": "an integer"}),
            ("ordinal", True),
            ("ordinal", 0),
            ("family_id", ""),
            ("public_locus", ""),
            ("source_digest", "not-a-digest"),
        )
        for field, replacement in cases:
            with self.subTest(field=field, replacement=replacement):
                attack = json.loads(json.dumps(raw_inputs))
                attack["tasks"][0][field] = replacement
                attack.pop("trainer_inputs_digest")
                attack["trainer_inputs_digest"] = digest_for(attack)
                with self.assertRaises((HeldoutProtocolError, TypeError, ValueError)):
                    HeldoutTrainerInputs(attack, protocol=self.protocol)

    def test_validated_task_metadata_and_source_bytes_are_immutable(self):
        raw_inputs, raw_sources, inputs, sources = self.package()
        task_id = next(iter(inputs.tasks))
        with self.assertRaises(TypeError):
            inputs.tasks[task_id]["public_locus"] = "substituted"
        with self.assertRaises(TypeError):
            inputs.tasks[task_id] = {"substituted": True}
        with self.assertRaises(TypeError):
            sources.sources[task_id] = b"substituted"
        with self.assertRaises(AttributeError):
            inputs.tasks = {}
        with self.assertRaises(AttributeError):
            sources.sources = {}
        with self.assertRaises(AttributeError):
            del inputs.tasks
        with self.assertRaises(AttributeError):
            del inputs._sealed
        with self.assertRaises(AttributeError):
            del sources.sources

        # Mutating the caller-owned decoded objects after validation cannot
        # mutate the retained canonical package used at admission.
        raw_inputs["tasks"][0]["public_locus"] = "caller-substitution"
        raw_sources["tasks"][0]["source_files"][1]["content_utf8"] = "caller-substitution"
        inputs.validate_retained(self.protocol)
        sources.validate_retained(inputs)
        self.assertNotEqual(inputs.tasks[task_id]["public_locus"], "caller-substitution")
        self.assertNotEqual(sources.source_for(task_id), b"caller-substitution")

    def test_protocol_bindings_lookup_and_recomputed_digest_resist_mutation(self):
        with self.assertRaises(TypeError):
            self.protocol.bindings["base_model_digest"] = digest_for("substituted")
        with self.assertRaises(TypeError):
            self.protocol._coordinates_by_id["substituted"] = self.protocol.coordinates[0]
        original = self.protocol.heldout_task_records_digest
        object.__setattr__(self.protocol, "heldout_task_records_digest", digest_for("substituted"))
        with self.assertRaisesRegex(HeldoutProtocolError, "commitment|reconstruction"):
            self.protocol.validate_current()
        object.__setattr__(self.protocol, "heldout_task_records_digest", original)
        self.assertEqual(self.protocol.validate_current(), self.protocol.digest)

    def test_main_context_rederives_protocol_and_package_after_object_rebinding(self):
        _, _, inputs, sources = self.package()
        original_records = self.protocol.heldout_task_records
        substituted = [dict(record) for record in original_records]
        substituted[0]["public_locus"] = "substituted:locus"
        object.__setattr__(self.protocol, "heldout_task_records", tuple(substituted))
        with self.assertRaisesRegex(HeldoutProtocolError, "admission|reconstruction|commitment"):
            HeldoutRuntimeContext(
                self.protocol, inputs, sources, self.root / "protocol-attack",
                lambda **kwargs: None, lambda **kwargs: None,
                lambda report, store: None,
            )
        object.__setattr__(self.protocol, "heldout_task_records", original_records)

        original_digest = inputs._digest
        object.__setattr__(inputs, "_digest", digest_for("substituted-package"))
        with self.assertRaisesRegex(HeldoutProtocolError, "admission"):
            HeldoutRuntimeContext(
                self.protocol, inputs, sources, self.root / "package-attack",
                lambda **kwargs: None, lambda **kwargs: None,
                lambda report, store: None,
            )
        object.__setattr__(inputs, "_digest", original_digest)

    def test_main_result_keeps_base_model_and_adapter_identity_separate(self):
        coordinate = next(
            item for item in self.protocol.coordinates
            if item.phase == MAIN_PHASE and item.treatment == "A"
        )
        report = self.report(coordinate)
        evidence = HeldoutRuntimeEvidence(11, 0.2, 1, 1, 1, 1, 0, 0)
        result = derive_verified_main_result(
            self.protocol, coordinate, report, evidence, wall_time_seconds=0.3
        )
        self.assertIs(result["success"], True)
        projection = build_coordinate_public_projection(result)
        self.assertNotIn("task_id", projection)
        self.assertNotIn("candidate", repr(projection))
        bad = replace(report, model_digest=self.protocol.bindings["trained_model_digest"])
        with self.assertRaisesRegex(HeldoutProtocolError, "differs"):
            derive_verified_main_result(self.protocol, coordinate, bad, evidence, wall_time_seconds=0.3)

    def test_runtime_evidence_rejects_untruthful_denominators(self):
        with self.assertRaisesRegex(HeldoutProtocolError, "agreements exceed"):
            HeldoutRuntimeEvidence(1, 0.0, 1, 2, 0, 0, 0, 0).validate()
        with self.assertRaisesRegex(HeldoutProtocolError, "denials exceed"):
            HeldoutRuntimeEvidence(1, 0.0, 0, 0, 0, 0, 0, 1).validate()

    def test_runtime_context_rejects_inputs_from_another_protocol(self):
        _, _, inputs, sources = self.package()
        other = _protocol(self.corpus)
        object.__setattr__(other, "campaign_id", "egv-campaign-aaaaaaaaaaaaaaaa")
        with self.assertRaisesRegex(HeldoutProtocolError, "admission|rebound"):
            HeldoutRuntimeContext(
                other, inputs, sources, self.root / "runs",
                lambda **kwargs: None, lambda **kwargs: None,
                lambda report, store: None,
            )


if __name__ == "__main__":
    unittest.main()
