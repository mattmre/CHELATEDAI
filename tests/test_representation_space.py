"""Contract tests for the representation-space ABI."""

from __future__ import annotations

import hashlib
import unittest
from dataclasses import FrozenInstanceError, replace
from typing import Optional

from representation_space import (
    BridgeContract,
    BridgeContractError,
    BridgeCost,
    BridgeValidationState,
    DistanceMetric,
    Normalization,
    PoolingStrategy,
    RepresentationABIError,
    RepresentationDescriptor,
    RepresentationRole,
    ResidentIndexDescriptor,
    VectorDType,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _space(
    version: str = "old",
    *,
    role: RepresentationRole = RepresentationRole.SHARED,
    compatibility: Optional[str] = None,
    dimension: int = 8,
    normalization: Normalization = Normalization.L2,
    metric: DistanceMetric = DistanceMetric.COSINE,
    dtype: VectorDType = VectorDType.FLOAT32,
) -> RepresentationDescriptor:
    domain = compatibility or version
    return RepresentationDescriptor(
        encoder_id="acme/semantic-encoder",
        encoder_revision=f"refs/commits/{version}",
        encoder_weights_sha256=_sha(f"weights:{version}"),
        tokenizer_id="acme/tokenizer",
        tokenizer_sha256=_sha(f"tokenizer:{version}"),
        role=role,
        compatibility_domain_id=f"semantic-domain:{domain}",
        compatibility_domain_sha256=_sha(f"semantic-domain:{domain}"),
        instruction_sha256=_sha(f"instruction:{version}:{role.value}"),
        pooling=PoolingStrategy.MEAN,
        dimension=dimension,
        preprocessing_id="preprocess-v2",
        preprocessing_sha256=_sha("preprocess-v2"),
        projection_chain_id="identity-projection",
        projection_chain_sha256=_sha("identity-projection"),
        adapter_chain_id="no-adapter",
        adapter_chain_sha256=_sha("no-adapter"),
        quantization_id="none",
        quantization_sha256=_sha("none"),
        normalization=normalization,
        metric=metric,
        dtype=dtype,
    )


def _pair(version: str) -> tuple[RepresentationDescriptor, RepresentationDescriptor]:
    return (
        _space(version, role=RepresentationRole.QUERY, compatibility=version),
        _space(version, role=RepresentationRole.DOCUMENT, compatibility=version),
    )


def _index(
    version: str = "old",
    *,
    snapshot: str = "corpus-a",
    build: str = "build-a",
    count: int = 100,
) -> ResidentIndexDescriptor:
    query_space, document_space = _pair(version)
    return ResidentIndexDescriptor(
        index_id=f"index:{version}:{snapshot}:{build}",
        document_space=document_space,
        query_space=query_space,
        corpus_snapshot_id=snapshot,
        corpus_snapshot_sha256=_sha(snapshot),
        vector_build_id=build,
        vector_build_sha256=_sha(build),
        document_count=count,
    )


def _bridge(
    source: RepresentationDescriptor,
    target: RepresentationDescriptor,
    *,
    family: str = "ridge",
    state: BridgeValidationState = BridgeValidationState.VALIDATED,
    cost: Optional[BridgeCost] = None,
    confidence: Optional[float] = None,
) -> BridgeContract:
    if cost is None:
        cost = BridgeCost.MATERIALIZED_INDEX if source.role is RepresentationRole.DOCUMENT else BridgeCost.ZERO_WRITE
    if confidence is None:
        confidence = 0.95 if state is BridgeValidationState.VALIDATED else 0.0
    evidence_id = None if state is BridgeValidationState.UNVALIDATED else f"evidence:{family}"
    evidence_sha256 = None if state is BridgeValidationState.UNVALIDATED else _sha(f"evidence:{family}")
    return BridgeContract(
        source=source,
        target=target,
        role=source.role,
        transform_family=family,
        transform_id=f"transform:{family}",
        weights_sha256=_sha(f"weights:{family}"),
        hyperparameters_sha256=_sha(f"hyperparameters:{family}"),
        fit_anchor_manifest_sha256=_sha(f"anchors:{family}"),
        fit_count=64 if state is BridgeValidationState.VALIDATED else 0,
        validation_count=32 if state is BridgeValidationState.VALIDATED else 0,
        validation_state=state,
        cost=cost,
        evidence_id=evidence_id,
        evidence_sha256=evidence_sha256,
        validation_confidence=confidence,
    )


class TestRepresentationDescriptor(unittest.TestCase):
    def test_stable_id_is_deterministic_and_content_addressed(self):
        first = _space()
        second = RepresentationDescriptor.from_dict(first.to_dict())
        self.assertEqual(first, second)
        self.assertEqual(first.stable_id, second.stable_id)
        self.assertRegex(first.stable_id, r"^rep:v1:sha256:[0-9a-f]{64}$")
        self.assertEqual(first.canonical_json(), second.canonical_json())

    def test_every_material_pipeline_change_changes_stable_id(self):
        original = _space()
        variants = (
            replace(original, encoder_revision="refs/commits/new"),
            replace(original, encoder_weights_sha256=_sha("different weights")),
            replace(original, tokenizer_sha256=_sha("different tokenizer")),
            replace(original, role=RepresentationRole.QUERY),
            replace(original, compatibility_domain_sha256=_sha("different domain")),
            replace(original, instruction_sha256=_sha("different instruction")),
            replace(original, pooling=PoolingStrategy.CLS),
            replace(original, dimension=16),
            replace(original, preprocessing_sha256=_sha("different preprocessing")),
            replace(original, projection_chain_sha256=_sha("different projection")),
            replace(original, adapter_chain_sha256=_sha("different adapter")),
            replace(original, quantization_sha256=_sha("different quantization")),
            replace(original, normalization=Normalization.NONE),
            replace(original, metric=DistanceMetric.DOT_PRODUCT),
            replace(original, dtype=VectorDType.FLOAT16),
        )
        self.assertEqual(len({variant.stable_id for variant in variants}), len(variants))
        self.assertNotIn(original.stable_id, {variant.stable_id for variant in variants})

    def test_unicode_is_normalized_before_hashing(self):
        composed = _space()
        decomposed = replace(composed, encoder_id="cafe\u0301/encoder")
        recomposed = replace(composed, encoder_id="caf\u00e9/encoder")
        self.assertEqual(decomposed.encoder_id, recomposed.encoder_id)
        self.assertEqual(decomposed.stable_id, recomposed.stable_id)

    def test_descriptor_is_frozen(self):
        descriptor = _space()
        with self.assertRaises(FrozenInstanceError):
            descriptor.dimension = 99

    def test_invalid_sha_is_rejected(self):
        with self.assertRaisesRegex(RepresentationABIError, "64 hexadecimal"):
            replace(_space(), tokenizer_sha256="not-a-digest")

    def test_bool_dimension_is_rejected(self):
        with self.assertRaisesRegex(RepresentationABIError, "dimension must be an integer"):
            replace(_space(), dimension=True)

    def test_unsupported_abi_version_is_rejected(self):
        with self.assertRaisesRegex(RepresentationABIError, "unsupported representation ABI"):
            replace(_space(), abi_version=2)

    def test_leading_whitespace_is_rejected_in_identity_fields(self):
        with self.assertRaisesRegex(RepresentationABIError, "leading or trailing"):
            replace(_space(), encoder_id=" encoder")

    def test_unknown_serialized_field_is_rejected(self):
        raw = _space().to_dict()
        raw["surprise"] = True
        with self.assertRaisesRegex(RepresentationABIError, "unknown fields"):
            RepresentationDescriptor.from_dict(raw)

    def test_tampered_stable_id_is_rejected(self):
        raw = _space().to_dict()
        raw["stable_id"] = "rep:v1:sha256:" + ("0" * 64)
        with self.assertRaisesRegex(RepresentationABIError, "stable_id mismatch"):
            RepresentationDescriptor.from_dict(raw)

    def test_tampered_payload_with_old_id_is_rejected(self):
        raw = _space().to_dict()
        raw["dimension"] = 9
        with self.assertRaisesRegex(RepresentationABIError, "stable_id mismatch"):
            RepresentationDescriptor.from_dict(raw)


class TestResidentIndexDescriptor(unittest.TestCase):
    def test_round_trip_and_stable_id(self):
        index = _index()
        loaded = ResidentIndexDescriptor.from_dict(index.to_dict())
        self.assertEqual(index, loaded)
        self.assertRegex(index.stable_id, r"^index:v1:sha256:[0-9a-f]{64}$")

    def test_snapshot_build_and_count_are_all_identity_fields(self):
        base = _index()
        variants = (
            _index(snapshot="corpus-b"),
            _index(build="build-b"),
            _index(count=101),
        )
        for variant in variants:
            self.assertNotEqual(base.stable_id, variant.stable_id)

    def test_query_document_roles_are_bound_explicitly(self):
        index = _index()
        self.assertEqual(index.query_space.role, RepresentationRole.QUERY)
        self.assertEqual(index.document_space.role, RepresentationRole.DOCUMENT)
        self.assertNotEqual(index.query_space.stable_id, index.document_space.stable_id)

    def test_cross_domain_query_document_pair_is_rejected(self):
        query_space, _ = _pair("old")
        _, other_document_space = _pair("new")
        with self.assertRaisesRegex(RepresentationABIError, "compatibility domain"):
            ResidentIndexDescriptor(
                index_id="broken",
                document_space=other_document_space,
                query_space=query_space,
                corpus_snapshot_id="corpus",
                corpus_snapshot_sha256=_sha("corpus"),
                vector_build_id="build",
                vector_build_sha256=_sha("build"),
                document_count=10,
            )

    def test_same_domain_digest_with_different_domain_ids_is_rejected(self):
        query_space, document_space = _pair("old")
        aliased_document_space = replace(
            document_space,
            compatibility_domain_id="semantic-domain:unauthorized-alias",
        )
        self.assertEqual(
            query_space.compatibility_domain_sha256,
            aliased_document_space.compatibility_domain_sha256,
        )
        with self.assertRaisesRegex(RepresentationABIError, "domain ID and digest"):
            replace(
                _index(),
                query_space=query_space,
                document_space=aliased_document_space,
            )

    def test_deserialization_rejects_same_digest_different_domain_ids(self):
        index = _index()
        aliased_query_space = replace(
            index.query_space,
            compatibility_domain_id="semantic-domain:unauthorized-alias",
        )
        raw = index.to_dict()
        raw["query_space"] = aliased_query_space.to_dict()
        raw["query_space_id"] = aliased_query_space.stable_id
        raw.pop("stable_id")
        with self.assertRaisesRegex(RepresentationABIError, "domain ID and digest"):
            ResidentIndexDescriptor.from_dict(raw)

    def test_native_pair_with_384_and_768_dimensions_is_rejected(self):
        query_space = _space(
            "native",
            role=RepresentationRole.QUERY,
            compatibility="native",
            dimension=384,
        )
        document_space = _space(
            "native",
            role=RepresentationRole.DOCUMENT,
            compatibility="native",
            dimension=768,
        )
        with self.assertRaisesRegex(RepresentationABIError, "equal dimensions"):
            replace(
                _index(),
                query_space=query_space,
                document_space=document_space,
            )

    def test_native_pair_with_cosine_and_euclidean_metrics_is_rejected(self):
        query_space = _space(
            "native",
            role=RepresentationRole.QUERY,
            compatibility="native",
            metric=DistanceMetric.COSINE,
        )
        document_space = _space(
            "native",
            role=RepresentationRole.DOCUMENT,
            compatibility="native",
            metric=DistanceMetric.EUCLIDEAN,
        )
        with self.assertRaisesRegex(RepresentationABIError, "same distance metric"):
            replace(
                _index(),
                query_space=query_space,
                document_space=document_space,
            )

    def test_native_pair_with_incompatible_normalization_is_rejected(self):
        query_space = _space(
            "native",
            role=RepresentationRole.QUERY,
            compatibility="native",
            normalization=Normalization.L2,
        )
        document_space = _space(
            "native",
            role=RepresentationRole.DOCUMENT,
            compatibility="native",
            normalization=Normalization.NONE,
        )
        with self.assertRaisesRegex(RepresentationABIError, "same normalization"):
            replace(
                _index(),
                query_space=query_space,
                document_space=document_space,
            )

    def test_native_pair_with_incompatible_dtype_is_rejected(self):
        query_space = _space(
            "native",
            role=RepresentationRole.QUERY,
            compatibility="native",
            dtype=VectorDType.FLOAT32,
        )
        document_space = _space(
            "native",
            role=RepresentationRole.DOCUMENT,
            compatibility="native",
            dtype=VectorDType.FLOAT16,
        )
        with self.assertRaisesRegex(RepresentationABIError, "same vector dtype"):
            replace(
                _index(),
                query_space=query_space,
                document_space=document_space,
            )

    def test_deserialization_revalidates_all_native_pair_invariants(self):
        index = _index()
        incompatible_queries = {
            "dimension": replace(index.query_space, dimension=768),
            "metric": replace(index.query_space, metric=DistanceMetric.EUCLIDEAN),
            "normalization": replace(index.query_space, normalization=Normalization.NONE),
            "dtype": replace(index.query_space, dtype=VectorDType.FLOAT16),
        }
        expected_messages = {
            "dimension": "equal dimensions",
            "metric": "same distance metric",
            "normalization": "same normalization",
            "dtype": "same vector dtype",
        }
        for field, query_space in incompatible_queries.items():
            with self.subTest(field=field):
                raw = index.to_dict()
                raw["query_space"] = query_space.to_dict()
                raw["query_space_id"] = query_space.stable_id
                raw.pop("stable_id")
                with self.assertRaisesRegex(RepresentationABIError, expected_messages[field]):
                    ResidentIndexDescriptor.from_dict(raw)

    def test_document_count_rejects_bool(self):
        with self.assertRaisesRegex(RepresentationABIError, "document_count must be an integer"):
            replace(_index(), document_count=True)

    def test_nested_representation_tampering_is_rejected(self):
        raw = _index().to_dict()
        raw["document_space_id"] = _space("unrelated").stable_id
        with self.assertRaisesRegex(RepresentationABIError, "document_space_id"):
            ResidentIndexDescriptor.from_dict(raw)


class TestBridgeContract(unittest.TestCase):
    def test_valid_query_bridge_round_trip(self):
        old_query, _ = _pair("old")
        new_query, _ = _pair("new")
        bridge = _bridge(new_query, old_query)
        loaded = BridgeContract.from_dict(bridge.to_dict())
        self.assertEqual(bridge, loaded)
        self.assertTrue(loaded.is_usable)
        self.assertEqual(loaded.cost, BridgeCost.ZERO_WRITE)
        self.assertRegex(loaded.bridge_id, r"^bridge:v1:sha256:[0-9a-f]{64}$")

    def test_valid_document_bridge_requires_materialization(self):
        _, old_document = _pair("old")
        _, new_document = _pair("new")
        bridge = _bridge(old_document, new_document)
        self.assertEqual(bridge.role, RepresentationRole.DOCUMENT)
        self.assertEqual(bridge.cost, BridgeCost.MATERIALIZED_INDEX)

    def test_bridge_cannot_cross_roles(self):
        old_query, _ = _pair("old")
        _, new_document = _pair("new")
        with self.assertRaisesRegex(BridgeContractError, "role must exactly match"):
            _bridge(old_query, new_document)

    def test_query_bridge_cannot_claim_materialized_cost(self):
        old_query, _ = _pair("old")
        new_query, _ = _pair("new")
        with self.assertRaisesRegex(BridgeContractError, "query-role bridges"):
            _bridge(new_query, old_query, cost=BridgeCost.MATERIALIZED_INDEX)

    def test_document_bridge_cannot_claim_zero_write_cost(self):
        _, old_document = _pair("old")
        _, new_document = _pair("new")
        with self.assertRaisesRegex(BridgeContractError, "document-role bridges"):
            _bridge(old_document, new_document, cost=BridgeCost.ZERO_WRITE)

    def test_validated_bridge_requires_evidence(self):
        old_query, _ = _pair("old")
        new_query, _ = _pair("new")
        bridge = _bridge(new_query, old_query)
        with self.assertRaisesRegex(BridgeContractError, "evidence_id"):
            replace(bridge, evidence_id=None)
        with self.assertRaisesRegex(BridgeContractError, "evidence_sha256"):
            replace(bridge, evidence_sha256=None)

    def test_validated_bridge_requires_fit_and_validation_support(self):
        old_query, _ = _pair("old")
        new_query, _ = _pair("new")
        bridge = _bridge(new_query, old_query)
        with self.assertRaisesRegex(BridgeContractError, "fit_count"):
            replace(bridge, fit_count=0)
        with self.assertRaisesRegex(BridgeContractError, "validation_count"):
            replace(bridge, validation_count=0)

    def test_unvalidated_bridge_cannot_claim_confidence_or_evidence(self):
        old_query, _ = _pair("old")
        new_query, _ = _pair("new")
        bridge = _bridge(new_query, old_query, state=BridgeValidationState.UNVALIDATED)
        self.assertFalse(bridge.is_usable)
        with self.assertRaisesRegex(BridgeContractError, "must not claim"):
            replace(bridge, evidence_id="premature")
        with self.assertRaisesRegex(BridgeContractError, "must not claim"):
            replace(bridge, evidence_sha256=_sha("premature"))
        with self.assertRaisesRegex(BridgeContractError, "validation_confidence=0"):
            replace(bridge, validation_confidence=0.5)

    def test_rejected_bridge_requires_evidence_and_zero_confidence(self):
        old_query, _ = _pair("old")
        new_query, _ = _pair("new")
        bridge = _bridge(
            new_query,
            old_query,
            state=BridgeValidationState.REJECTED,
            confidence=0.0,
        )
        self.assertFalse(bridge.is_usable)
        with self.assertRaisesRegex(BridgeContractError, "must have validation_confidence=0"):
            replace(bridge, validation_confidence=0.1)

    def test_bridge_identity_binds_fit_manifest_weights_and_counts(self):
        old_query, _ = _pair("old")
        new_query, _ = _pair("new")
        bridge = _bridge(new_query, old_query)
        variants = (
            replace(bridge, weights_sha256=_sha("different weights")),
            replace(bridge, hyperparameters_sha256=_sha("different hyperparameters")),
            replace(bridge, fit_anchor_manifest_sha256=_sha("different anchors")),
            replace(bridge, evidence_sha256=_sha("different evidence")),
            replace(bridge, fit_count=65),
            replace(bridge, validation_count=33),
        )
        for variant in variants:
            self.assertNotEqual(bridge.stable_id, variant.stable_id)

    def test_bridge_nested_space_id_tampering_is_rejected(self):
        old_query, _ = _pair("old")
        new_query, _ = _pair("new")
        raw = _bridge(new_query, old_query).to_dict()
        raw["source_space_id"] = old_query.stable_id
        with self.assertRaisesRegex(BridgeContractError, "source_space_id"):
            BridgeContract.from_dict(raw)


if __name__ == "__main__":
    unittest.main()
