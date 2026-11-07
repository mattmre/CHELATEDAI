import hashlib
import json
import unittest
from copy import deepcopy
from pathlib import Path

import jsonschema

import qscci_experiment as qscci


ROOT = Path(__file__).resolve().parent
PROTOCOL_PATH = ROOT / "docs/research/qwen-scope-chelated-causal-intervention-delta-norm-erratum-v4-2026-08-16.md"
V1_PROTOCOL_PATH = ROOT / "docs/research/qwen-scope-chelated-causal-intervention-preregistration-2026-08-16.md"
V2_PROTOCOL_PATH = ROOT / "docs/research/qwen-scope-chelated-causal-intervention-infrastructure-addendum-v2-2026-08-16.md"
V3_PROTOCOL_PATH = ROOT / "docs/research/qwen-scope-chelated-causal-intervention-numerical-erratum-v3-2026-08-16.md"
FIXTURE_PATH = ROOT / "docs/research/qwen-scope-chelated-causal-intervention-fixture-v1.json"
SCHEMA_PATH = ROOT / "docs/research/qwen-scope-chelated-causal-intervention-artifact-schema-v4.json"
V1_SCHEMA_PATH = ROOT / "docs/research/qwen-scope-chelated-causal-intervention-artifact-schema-v1.json"
V2_SCHEMA_PATH = ROOT / "docs/research/qwen-scope-chelated-causal-intervention-artifact-schema-v2.json"
V3_SCHEMA_PATH = ROOT / "docs/research/qwen-scope-chelated-causal-intervention-artifact-schema-v3.json"

FIXTURE_SHA256 = "d9f873d5a0e00d0330b87e5ea053aaf0343d4c1636d7402020490719d1336f83"
SCHEMA_SHA256 = "a30fee50976ebc1d04c04d9ca7c1f1cc2c37cb1caed06ac56aae77e1d2a5eb46"
PROTOCOL_SHA256 = "b28e978fc61238df9439348355da92d95329092794b12b754bcb6c990e8aa27e"
V1_SCHEMA_SHA256 = "032f06fefc53aa04f13a410ce5aec93b7039c7c042bd0a2cedfe8f4745cb653a"
V1_PROTOCOL_SHA256 = "f7417b022dd93b96d523f6e8ca4a12c8a915288b5b4621e9ef130ac4f52848f1"
V2_SCHEMA_SHA256 = "9a1ec63343772955d3f7057beffaded5e64c10ccfaf57b7c36e14d02a65c5c80"
V2_PROTOCOL_SHA256 = "3bf01a1b2e140fa1ee57816dca15d204ad7fdc240804d421116261f207e6a467"
V3_SCHEMA_SHA256 = "237a3a678c2bf7cafc4c86b2ec22657f67aeab39dd49c2cc1bd687e02d38d9c8"
V3_PROTOCOL_SHA256 = "2d5375feb8f3b6b4c8da4d8640554d50f83f36fca63f696b8835f65fd3139a30"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestQSCCIFrozenFiles(unittest.TestCase):
    def test_fixture_bytes_have_frozen_digest(self):
        self.assertEqual(_sha256(FIXTURE_PATH), FIXTURE_SHA256)

    def test_schema_bytes_have_frozen_digest(self):
        self.assertEqual(_sha256(SCHEMA_PATH), SCHEMA_SHA256)

    def test_protocol_bytes_have_pinned_digest_for_artifact_binding(self):
        self.assertEqual(_sha256(PROTOCOL_PATH), PROTOCOL_SHA256)

    def test_v1_protocol_and_schema_bytes_remain_immutably_bound(self):
        self.assertEqual(_sha256(V1_PROTOCOL_PATH), V1_PROTOCOL_SHA256)
        self.assertEqual(_sha256(V1_SCHEMA_PATH), V1_SCHEMA_SHA256)

    def test_v4_schema_diff_is_only_version_identity_and_protocol_id(self):
        v3 = json.loads(V3_SCHEMA_PATH.read_text(encoding="utf-8"))
        v4 = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        v4["$id"] = v3["$id"]
        v4["properties"]["protocol_id"]["const"] = v3["properties"]["protocol_id"]["const"]
        self.assertEqual(v4, v3)

    def test_v4_erratum_binds_v3_v1_and_declares_only_delta_norm_canonicalization(self):
        addendum = PROTOCOL_PATH.read_text(encoding="utf-8")
        self.assertIn(V1_PROTOCOL_SHA256, addendum)
        self.assertIn(V1_SCHEMA_SHA256, addendum)
        self.assertIn(FIXTURE_SHA256, addendum)
        self.assertIn(V3_PROTOCOL_SHA256, addendum)
        self.assertIn(V3_SCHEMA_SHA256, addendum)
        self.assertIn("Sole erratum", addendum)
        self.assertIn("still constructs the actual per-prompt FP32 delta on the hook", addendum)
        self.assertIn("copies their exact components to detached CPU FP32 and BF16 tensors", addendum)
        self.assertIn("A component or cast disagreement invalidates the run", addendum)
        self.assertIn("remains\nmodel-execution-attested", addendum)
        self.assertIn("600-second restoration ceiling", addendum)
        for invariant in (
            "fixture and split", "selection and controls", "tuning", "model/SAE\nidentities",
            "endpoint formulas", "scientific gates", "statuses", "resources",
        ):
            self.assertIn(invariant, addendum)

    def test_unexecuted_v2_draft_bytes_remain_unchanged(self):
        self.assertEqual(_sha256(V2_PROTOCOL_PATH), V2_PROTOCOL_SHA256)
        self.assertEqual(_sha256(V2_SCHEMA_PATH), V2_SCHEMA_SHA256)

    def test_v3_erratum_and_schema_bytes_remain_unchanged(self):
        self.assertEqual(_sha256(V3_PROTOCOL_PATH), V3_PROTOCOL_SHA256)
        self.assertEqual(_sha256(V3_SCHEMA_PATH), V3_SCHEMA_SHA256)

    def test_fixture_has_exact_disjoint_split_and_label_balance(self):
        fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        self.assertEqual([row["id"] for row in fixture["select"]], [f"S{i:02d}" for i in range(1, 7)])
        self.assertEqual([row["id"] for row in fixture["report"]], [f"R{i:02d}" for i in range(1, 7)])
        self.assertEqual([row["y"] for row in fixture["select"]], [1, -1, 1, -1, 1, -1])
        self.assertEqual([row["y"] for row in fixture["report"]], [1, -1, 1, -1, 1, -1])
        self.assertTrue(
            {row["id"] for row in fixture["select"]}.isdisjoint({row["id"] for row in fixture["report"]})
        )

    def test_fixture_template_has_one_literal_review_marker(self):
        fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        self.assertEqual(fixture["prompt_template"].count("{review}"), 1)
        self.assertEqual(
            fixture["prompt_template"],
            "Review: The meal was wonderful.\n"
            "Sentiment: positive\n"
            "Review: The meal was awful.\n"
            "Sentiment: negative\n"
            "Review: {review}\n"
            "Sentiment:",
        )

    def test_fixture_rows_have_only_frozen_fields(self):
        fixture = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        for split in ("select", "report"):
            for row in fixture[split]:
                with self.subTest(split=split, row=row["id"]):
                    self.assertEqual(set(row), {"canonical", "id", "material", "nuisance", "y"})
                    self.assertNotEqual(row["canonical"], row["nuisance"])
                    self.assertNotEqual(row["canonical"], row["material"])

    def test_artifact_schema_is_fail_closed_at_envelope(self):
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.assertEqual(schema["$schema"], "https://json-schema.org/draft/2020-12/schema")
        self.assertEqual(schema["$id"], "CHELATEDAI-QSCCI-ARTIFACT-v4")
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["required"]), set(schema["properties"]))
        self.assertEqual(schema["properties"]["protocol_id"]["const"], "CHELATEDAI-QSCCI-v4")
        self.assertEqual(schema["properties"]["status"]["const"], "COMPLETE")
        self.assertEqual(schema["properties"]["scientific_claim_status"]["const"], "UNCONFIRMED")
        self.assertEqual(schema["properties"]["novelty_claim_status"]["const"], "UNCONFIRMED")
        self.assertEqual(schema["properties"]["feature_records"]["minItems"], 32768)
        self.assertEqual(schema["properties"]["feature_records"]["maxItems"], 32768)
        self.assertEqual(schema["properties"]["select_activation_rows"]["minItems"], 18)
        self.assertEqual(schema["properties"]["select_activation_rows"]["maxItems"], 18)

    def test_schema_disposition_is_exact_mutually_exclusive_set(self):
        schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        self.assertEqual(
            set(schema["properties"]["disposition"]["enum"]),
            {
                "SURVIVES_SMALL_LABEL_ORACLE_FIXTURE",
                "DOES_NOT_SURVIVE_SMALL_LABEL_ORACLE_FIXTURE",
                "INVALID_TASK",
                "INVALID_RUN",
            },
        )

    def test_production_constants_bind_exact_frozen_model_and_sae_identities(self):
        self.assertEqual(qscci.PROTOCOL_ID, "CHELATEDAI-QSCCI-v4")
        self.assertEqual(qscci.PROTOCOL_SHA256, PROTOCOL_SHA256)
        self.assertEqual(qscci.FIXTURE_SHA256, FIXTURE_SHA256)
        self.assertEqual(qscci.SCHEMA_SHA256, SCHEMA_SHA256)
        self.assertEqual(qscci.BASE_PROTOCOL_SHA256, V1_PROTOCOL_SHA256)
        self.assertEqual(qscci.BASE_SCHEMA_SHA256, V1_SCHEMA_SHA256)
        self.assertEqual(qscci.V3_PROTOCOL_SHA256, V3_PROTOCOL_SHA256)
        self.assertEqual(qscci.V3_SCHEMA_SHA256, V3_SCHEMA_SHA256)
        self.assertEqual(qscci.MODEL_REPO, "Qwen/Qwen3.5-2B-Base")
        self.assertEqual(qscci.MODEL_REVISION, "b1485b2fa6dfa1287294f269f5fb618e03d52d7c")
        self.assertEqual(qscci.SAE_REPO, "Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_100")
        self.assertEqual(qscci.SAE_REVISION, "027267657257a8d490296286e8fab41e1c1a1a3d")
        self.assertEqual(qscci.SAE_FILENAME, "layer11.sae.pt")
        self.assertEqual(qscci.SAE_SHA256, "d1828ace348b13cca9104f61fb47672e439e963d9d5fc5496f4c6b068a06499f")
        self.assertEqual((qscci.MODEL_LAYER, qscci.HIDDEN_SIZE, qscci.SAE_WIDTH, qscci.TOP_K), (11, 2048, 32768, 100))
        self.assertEqual((qscci.POSITIVE_TOKEN, qscci.NEGATIVE_TOKEN), (6572, 7968))
        self.assertEqual(qscci.TARGET_TOKEN_TEXT, {"positive": " positive", "negative": " negative"})


class TestQSCCIArtifactEnvelopeSchema(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))
        cls.valid = {
            "protocol_id": "CHELATEDAI-QSCCI-v4",
            "status": "COMPLETE",
            "scientific_claim_status": "UNCONFIRMED",
            "novelty_claim_status": "UNCONFIRMED",
            "protocol_file_sha256": PROTOCOL_SHA256,
            "fixture_sha256": FIXTURE_SHA256,
            "schema_file_sha256": SCHEMA_SHA256,
            "model_identity": {},
            "service_lifecycle": {},
            "resources": {},
            "select_activation_rows": [{} for _ in range(18)],
            "feature_records": [{} for _ in range(32768)],
            "choices": {},
            "prompt_records": [],
            "cells": [],
            "aggregates": {},
            "gates": {},
            "disposition": "INVALID_RUN",
            "failures": [],
            "artifact_digest": "0" * 64,
        }

    def test_schema_accepts_exact_envelope(self):
        jsonschema.Draft202012Validator(self.schema).validate(self.valid)

    def test_v4_schema_rejects_v1_v2_and_v3_protocol_identities(self):
        for protocol_id in ("CHELATEDAI-QSCCI-v1", "CHELATEDAI-QSCCI-v2", "CHELATEDAI-QSCCI-v3"):
            with self.subTest(protocol_id=protocol_id):
                payload = deepcopy(self.valid)
                payload["protocol_id"] = protocol_id
                with self.assertRaises(jsonschema.ValidationError):
                    jsonschema.Draft202012Validator(self.schema).validate(payload)

    def test_schema_rejects_unregistered_envelope_member(self):
        payload = deepcopy(self.valid)
        payload["stored_disposition_is_trusted"] = True
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.Draft202012Validator(self.schema).validate(payload)

    def test_schema_rejects_missing_required_member(self):
        payload = deepcopy(self.valid)
        del payload["service_lifecycle"]
        with self.assertRaises(jsonschema.ValidationError):
            jsonschema.Draft202012Validator(self.schema).validate(payload)

    def test_schema_rejects_tampered_protocol_and_digest_identifiers(self):
        mutations = {
            "protocol_id": "CHELATEDAI-QSCCI-v1",
            "status": "PASS",
            "scientific_claim_status": "CONFIRMED",
            "novelty_claim_status": "CONFIRMED",
            "fixture_sha256": "A" * 64,
            "artifact_digest": "0" * 63,
        }
        validator = jsonschema.Draft202012Validator(self.schema)
        for field, value in mutations.items():
            with self.subTest(field=field):
                payload = deepcopy(self.valid)
                payload[field] = value
                with self.assertRaises(jsonschema.ValidationError):
                    validator.validate(payload)

    def test_schema_rejects_wrong_full_feature_or_select_row_cardinality(self):
        mutations = {
            "feature_records": self.valid["feature_records"][:-1],
            "select_activation_rows": self.valid["select_activation_rows"] + [{}],
        }
        validator = jsonschema.Draft202012Validator(self.schema)
        for field, value in mutations.items():
            with self.subTest(field=field):
                payload = dict(self.valid)
                payload[field] = value
                with self.assertRaises(jsonschema.ValidationError):
                    validator.validate(payload)


if __name__ == "__main__":
    unittest.main()
