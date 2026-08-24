"""Fail-closed routing tests for semantic cache resolution."""

from __future__ import annotations

import hashlib
import json
import unittest
from dataclasses import replace
from collections.abc import Mapping
from typing import Optional

from representation_space import (
    BridgeContract,
    BridgeCost,
    BridgeValidationState,
    DistanceMetric,
    Normalization,
    PoolingStrategy,
    RepresentationDescriptor,
    RepresentationRole,
    ResidentIndexDescriptor,
    VectorDType,
)
from semantic_cache_resolver import (
    MaterializedIndexRef,
    ResolutionError,
    ResolutionMode,
    ResolutionReason,
    RouteFeatureDeclaration,
    RouteScorerContract,
    RouteScorerRuntimeAttestation,
    SemanticCacheResolver as _SemanticCacheResolver,
    route_confidence_rule_sha256,
    route_feature_manifest_sha256,
)


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


def _route_feature(
    name: str,
    value_type: str,
    *,
    qrels_free: bool = True,
    availability_stage: str = "PRE_SEARCH",
    execution_cost_class: str = "zero_search",
    implementation_label: Optional[str] = None,
    provenance_label: Optional[str] = None,
) -> RouteFeatureDeclaration:
    return RouteFeatureDeclaration(
        name=name,
        type=value_type,
        qrels_free=qrels_free,
        availability_stage=availability_stage,
        execution_cost_class=execution_cost_class,
        implementation_sha256=_sha(implementation_label or f"feature:{name}:implementation"),
        provenance_sha256=_sha(provenance_label or f"feature:{name}:provenance"),
    )


def _space(
    version: str,
    role: RepresentationRole,
    *,
    compatibility: Optional[str] = None,
    family: str = "encoder",
) -> RepresentationDescriptor:
    domain = compatibility or version
    return RepresentationDescriptor(
        encoder_id=f"acme/{family}",
        encoder_revision=f"revision:{version}",
        encoder_weights_sha256=_sha(f"weights:{version}:{family}"),
        tokenizer_id=f"acme/{family}-tokenizer",
        tokenizer_sha256=_sha(f"tokenizer:{version}:{family}"),
        role=role,
        compatibility_domain_id=f"domain:{domain}",
        compatibility_domain_sha256=_sha(f"domain:{domain}"),
        instruction_sha256=_sha(f"instruction:{version}:{role.value}"),
        pooling=PoolingStrategy.MEAN,
        dimension=8,
        preprocessing_id="preprocess-v1",
        preprocessing_sha256=_sha("preprocess-v1"),
        projection_chain_id="identity",
        projection_chain_sha256=_sha("identity-projection"),
        adapter_chain_id="none",
        adapter_chain_sha256=_sha("no-adapter"),
        quantization_id="none",
        quantization_sha256=_sha("no-quantization"),
        normalization=Normalization.L2,
        metric=DistanceMetric.COSINE,
        dtype=VectorDType.FLOAT32,
    )


def _pair(version: str) -> tuple[RepresentationDescriptor, RepresentationDescriptor]:
    return (
        _space(version, RepresentationRole.QUERY),
        _space(version, RepresentationRole.DOCUMENT),
    )


def _resident(
    version: str = "old",
    *,
    snapshot: str = "corpus-a",
    build: Optional[str] = None,
    count: int = 100,
) -> ResidentIndexDescriptor:
    query_space, document_space = _pair(version)
    build_name = build or f"build:{version}"
    return ResidentIndexDescriptor(
        index_id=f"index:{version}:{snapshot}",
        document_space=document_space,
        query_space=query_space,
        corpus_snapshot_id=snapshot,
        corpus_snapshot_sha256=_sha(snapshot),
        vector_build_id=build_name,
        vector_build_sha256=_sha(build_name),
        document_count=count,
    )


def _bridge(
    source: RepresentationDescriptor,
    target: RepresentationDescriptor,
    *,
    name: str,
    confidence: float = 0.95,
    state: BridgeValidationState = BridgeValidationState.VALIDATED,
    cost: Optional[BridgeCost] = None,
) -> BridgeContract:
    if cost is None:
        cost = BridgeCost.MATERIALIZED_INDEX if source.role is RepresentationRole.DOCUMENT else BridgeCost.ZERO_WRITE
    return BridgeContract(
        source=source,
        target=target,
        role=source.role,
        transform_family="ridge",
        transform_id=f"transform:{name}",
        weights_sha256=_sha(f"weights:{name}"),
        hyperparameters_sha256=_sha(f"hyperparameters:{name}"),
        fit_anchor_manifest_sha256=_sha(f"anchors:{name}"),
        fit_count=64 if state is BridgeValidationState.VALIDATED else 0,
        validation_count=32 if state is BridgeValidationState.VALIDATED else 0,
        validation_state=state,
        cost=cost,
        evidence_id=None if state is BridgeValidationState.UNVALIDATED else f"evidence:{name}",
        evidence_sha256=(None if state is BridgeValidationState.UNVALIDATED else _sha(f"evidence:{name}")),
        validation_confidence=confidence if state is BridgeValidationState.VALIDATED else 0.0,
    )


def SemanticCacheResolver(
    bridges=(),
    *,
    trusted_materializations=(),
    **kwargs,
):
    """Build a resolver whose test trust store is explicit and content-addressed."""

    bridges = tuple(bridges)
    trusted_materializations = tuple(trusted_materializations)
    kwargs.setdefault(
        "trusted_evidence_sha256s",
        tuple(bridge.evidence_sha256 for bridge in bridges if bridge.evidence_sha256 is not None),
    )
    kwargs.setdefault(
        "trusted_bridge_contract_ids",
        tuple(bridge.stable_id for bridge in bridges),
    )
    kwargs.setdefault(
        "trusted_materialization_contracts",
        {reference.stable_id: reference.content_sha256 for reference in trusted_materializations},
    )
    if kwargs.get("route_scorer") is not None:
        contract = kwargs.setdefault(
            "route_scorer_contract",
            _scorer_contract(
                minimum_route_confidence=kwargs.get(
                    "minimum_route_confidence",
                    0.8,
                )
            ),
        )
        kwargs.setdefault(
            "trusted_route_scorer_contract_sha256s",
            (contract.content_sha256,),
        )
    return _SemanticCacheResolver(bridges, **kwargs)


def _scorer_contract(
    policy: str = "semantic-route-policy-v1",
    feature_schema=(),
    minimum_route_confidence: float = 0.8,
) -> RouteScorerContract:
    feature_schema = tuple(feature_schema)
    implementation_sha256 = _sha("fixed-scorer-implementation-v1")
    return RouteScorerContract(
        policy_id=policy,
        scorer_stable_id=("route-scorer-implementation:v1:" f"sha256:{implementation_sha256}"),
        model_sha256=_sha("fixed-scorer-model-v1"),
        scaler_sha256=_sha("fixed-scorer-scaler-v1"),
        feature_manifest_sha256=route_feature_manifest_sha256(feature_schema),
        feature_schema=feature_schema,
        confidence_rule_sha256=route_confidence_rule_sha256(minimum_route_confidence),
        minimum_route_confidence=minimum_route_confidence,
        implementation_sha256=implementation_sha256,
        evidence_id=f"evidence:{policy}",
        evidence_sha256=_sha("fixed-scorer-evidence-v1"),
    )


def _materialized(
    bridge: BridgeContract,
    source: ResidentIndexDescriptor,
    target: ResidentIndexDescriptor,
    *,
    manifest_id: Optional[str] = None,
    manifest_sha256: Optional[str] = None,
) -> MaterializedIndexRef:
    manifest_id = manifest_id or f"manifest:{bridge.transform_id}"
    return MaterializedIndexRef(
        bridge_id=bridge.stable_id,
        source_index_id=source.stable_id,
        index=target,
        manifest_id=manifest_id,
        manifest_sha256=manifest_sha256 or _sha(manifest_id),
    )


class TestNativeAndAbstain(unittest.TestCase):
    def test_native_search_uses_declared_query_side_of_resident_store(self):
        resident = _resident()
        decision = SemanticCacheResolver().resolve(resident.query_space, resident)
        self.assertEqual(decision.mode, ResolutionMode.NATIVE)
        self.assertEqual(decision.reason, ResolutionReason.IDENTICAL_SPACE)
        self.assertTrue(decision.can_search)
        self.assertIs(decision.search_index, resident)
        self.assertNotEqual(resident.query_space.stable_id, resident.document_space.stable_id)
        decision.assert_search_authorized()

    def test_equal_dimensions_do_not_authorize_mismatched_spaces(self):
        resident = _resident("old")
        new_query, _ = _pair("new")
        self.assertEqual(new_query.dimension, resident.query_space.dimension)
        decision = SemanticCacheResolver().resolve(new_query, resident)
        self.assertEqual(decision.mode, ResolutionMode.ABSTAIN)
        self.assertEqual(decision.reason, ResolutionReason.NO_REPRESENTATION_BRIDGE)
        self.assertFalse(decision.can_search)
        self.assertIsNone(decision.search_index)

    def test_abstained_plan_raises_if_caller_attempts_search(self):
        resident = _resident("old")
        new_query, _ = _pair("new")
        decision = SemanticCacheResolver().resolve(new_query, resident)
        with self.assertRaisesRegex(ResolutionError, "search is not authorized"):
            decision.assert_search_authorized()

    def test_bare_representation_is_rejected_at_store_boundary(self):
        resident = _resident()
        with self.assertRaisesRegex(ResolutionError, "ResidentIndexDescriptor"):
            SemanticCacheResolver().resolve(resident.query_space, resident.document_space)

    def test_decision_serialization_exposes_store_and_authorization(self):
        resident = _resident()
        raw = SemanticCacheResolver().resolve(resident.query_space, resident).to_dict()
        self.assertEqual(raw["resident_index_id"], resident.stable_id)
        self.assertEqual(raw["search_index_id"], resident.stable_id)
        self.assertTrue(raw["can_search"])
        self.assertEqual(raw["corpus_write_cost"], "zero_write")


class TestReverseResolution(unittest.TestCase):
    def setUp(self):
        self.resident = _resident("old")
        self.new_query, _ = _pair("new")
        self.bridge = _bridge(
            self.new_query,
            self.resident.query_space,
            name="new-query-to-old-query",
        )

    def test_validated_query_bridge_authorizes_zero_write_reverse(self):
        decision = SemanticCacheResolver([self.bridge]).resolve(self.new_query, self.resident)
        self.assertEqual(decision.mode, ResolutionMode.REVERSE)
        self.assertEqual(decision.reason, ResolutionReason.VALIDATED_REVERSE_BRIDGE)
        self.assertTrue(decision.requires_query_transform)
        self.assertFalse(decision.requires_materialized_index)
        self.assertEqual(decision.corpus_write_cost, BridgeCost.ZERO_WRITE)
        self.assertIs(decision.search_index, self.resident)
        self.assertEqual(decision.bridge.evidence_id, self.bridge.evidence_id)
        self.assertEqual(decision.to_dict()["evidence_sha256"], self.bridge.evidence_sha256)

    def test_self_declared_validated_bridge_without_trusted_digest_abstains(self):
        decision = _SemanticCacheResolver([self.bridge]).resolve(
            self.new_query,
            self.resident,
        )
        self.assertEqual(decision.mode, ResolutionMode.ABSTAIN)
        self.assertEqual(decision.reason, ResolutionReason.UNTRUSTED_BRIDGE_EVIDENCE)

    def test_changed_bridge_weights_cannot_reuse_trusted_evidence(self):
        substituted = replace(
            self.bridge,
            weights_sha256=_sha("substituted-bridge-weights"),
        )
        self.assertEqual(
            substituted.evidence_sha256,
            self.bridge.evidence_sha256,
        )
        self.assertNotEqual(substituted.stable_id, self.bridge.stable_id)
        decision = _SemanticCacheResolver(
            [substituted],
            trusted_evidence_sha256s=(self.bridge.evidence_sha256,),
            trusted_bridge_contract_ids=(self.bridge.stable_id,),
        ).resolve(
            self.new_query,
            self.resident,
        )
        self.assertEqual(decision.mode, ResolutionMode.ABSTAIN)
        self.assertEqual(
            decision.reason,
            ResolutionReason.UNTRUSTED_BRIDGE_CONTRACT,
        )
        self.assertFalse(decision.can_search)

    def test_unvalidated_query_bridge_abstains(self):
        draft = _bridge(
            self.new_query,
            self.resident.query_space,
            name="draft",
            state=BridgeValidationState.UNVALIDATED,
        )
        decision = SemanticCacheResolver([draft]).resolve(self.new_query, self.resident)
        self.assertEqual(decision.mode, ResolutionMode.ABSTAIN)
        self.assertEqual(decision.reason, ResolutionReason.NO_VALIDATED_BRIDGE)

    def test_validation_confidence_below_policy_abstains(self):
        low = _bridge(
            self.new_query,
            self.resident.query_space,
            name="low-confidence",
            confidence=0.79,
        )
        decision = SemanticCacheResolver([low], minimum_route_confidence=0.8).resolve(
            self.new_query,
            self.resident,
        )
        self.assertEqual(decision.reason, ResolutionReason.LOW_ROUTE_CONFIDENCE)
        self.assertAlmostEqual(decision.confidence, 0.79)

    def test_per_query_confidence_is_capped_by_validation_confidence(self):
        contract = _scorer_contract(minimum_route_confidence=0.96)
        decision = SemanticCacheResolver(
            [self.bridge],
            minimum_route_confidence=0.96,
            route_scorer=_FixedScorer(
                {self.bridge.stable_id: 1.0},
                contract=contract,
            ),
            route_scorer_contract=contract,
        ).resolve(
            self.new_query,
            self.resident,
        )
        self.assertEqual(decision.reason, ResolutionReason.LOW_ROUTE_CONFIDENCE)
        self.assertAlmostEqual(decision.confidence, 0.95)
        self.assertRegex(decision.route_policy_sha256, r"^[0-9a-f]{64}$")
        self.assertRegex(decision.route_invocation_sha256, r"^[0-9a-f]{64}$")
        self.assertEqual(
            decision.route_candidate_ids,
            (self.bridge.stable_id,),
        )
        self.assertEqual(
            decision.route_scores,
            ((self.bridge.stable_id, 1.0),),
        )
        self.assertIsNone(decision.route_failure_class)

    def test_missing_per_query_score_fails_closed_as_router_failure(self):
        decision = SemanticCacheResolver(
            [self.bridge],
            route_scorer=_FixedScorer({}),
        ).resolve(
            self.new_query,
            self.resident,
        )
        self.assertEqual(decision.mode, ResolutionMode.ABSTAIN)
        self.assertEqual(decision.reason, ResolutionReason.ROUTER_FAILURE)

    def test_direct_route_scores_are_rejected_at_request_boundary(self):
        with self.assertRaisesRegex(ResolutionError, "direct route_scores"):
            SemanticCacheResolver([self.bridge]).resolve(
                self.new_query,
                self.resident,
                route_scores={self.bridge.stable_id: 0.9},
            )


class TestForwardResolution(unittest.TestCase):
    def setUp(self):
        self.resident = _resident("old", snapshot="corpus-a")
        self.new_query, self.new_document = _pair("new")
        self.bridge = _bridge(
            self.resident.document_space,
            self.new_document,
            name="old-document-to-new-document",
        )
        self.target = _resident("new", snapshot="corpus-a", build="bridge-build")

    def test_forward_bridge_without_materialized_index_abstains(self):
        decision = SemanticCacheResolver([self.bridge]).resolve(
            self.new_query,
            self.resident,
        )
        self.assertEqual(decision.mode, ResolutionMode.ABSTAIN)
        self.assertEqual(decision.reason, ResolutionReason.MATERIALIZED_INDEX_UNAVAILABLE)

    def test_valid_materialized_forward_index_authorizes_search(self):
        reference = _materialized(self.bridge, self.resident, self.target)
        decision = SemanticCacheResolver(
            [self.bridge],
            trusted_materializations=(reference,),
        ).resolve(
            self.new_query,
            self.resident,
            materialized_indexes=[reference],
        )
        self.assertEqual(decision.mode, ResolutionMode.FORWARD)
        self.assertEqual(decision.reason, ResolutionReason.VALIDATED_FORWARD_BRIDGE)
        self.assertTrue(decision.can_search)
        self.assertTrue(decision.requires_materialized_index)
        self.assertFalse(decision.requires_query_transform)
        self.assertEqual(decision.corpus_write_cost, BridgeCost.MATERIALIZED_INDEX)
        self.assertIs(decision.search_index, self.target)
        self.assertEqual(decision.search_space.stable_id, self.new_query.stable_id)
        serialized_contract = decision.to_dict()["materialized_index"]
        self.assertEqual(serialized_contract["stable_id"], reference.stable_id)
        self.assertEqual(
            serialized_contract["content_sha256"],
            reference.content_sha256,
        )

    def test_materialization_contract_binds_manifest_build_count_and_corpus(self):
        reference = _materialized(self.bridge, self.resident, self.target)
        raw = reference.to_dict()
        self.assertEqual(raw["bridge_stable_id"], self.bridge.stable_id)
        self.assertEqual(
            raw["source_resident_index_stable_id"],
            self.resident.stable_id,
        )
        self.assertEqual(
            raw["target_materialized_index_stable_id"],
            self.target.stable_id,
        )
        self.assertEqual(raw["manifest_sha256"], reference.manifest_sha256)
        self.assertEqual(raw["vector_build_id"], self.target.vector_build_id)
        self.assertEqual(
            raw["vector_build_sha256"],
            self.target.vector_build_sha256,
        )
        self.assertEqual(raw["document_count"], self.target.document_count)
        self.assertEqual(
            raw["corpus_snapshot_sha256"],
            self.target.corpus_snapshot_sha256,
        )
        self.assertEqual(raw["content_sha256"], reference.content_sha256)
        self.assertEqual(raw["stable_id"], reference.stable_id)

        changed_manifest = replace(
            reference,
            manifest_sha256=_sha("substituted-manifest"),
        )
        changed_count = replace(
            reference,
            index=replace(self.target, document_count=self.target.document_count + 1),
        )
        changed_corpus = replace(
            reference,
            index=replace(
                self.target,
                corpus_snapshot_sha256=_sha("substituted-corpus"),
            ),
        )
        for field_name, changed in (
            ("manifest", changed_manifest),
            ("count", changed_count),
            ("corpus", changed_corpus),
        ):
            with self.subTest(field=field_name):
                self.assertNotEqual(changed.content_sha256, reference.content_sha256)
                self.assertNotEqual(changed.stable_id, reference.stable_id)

    def test_altered_build_cannot_reuse_trusted_manifest(self):
        trusted = _materialized(self.bridge, self.resident, self.target)
        altered_target = replace(
            self.target,
            vector_build_sha256=_sha("forged-vector-build"),
        )
        forged = replace(trusted, index=altered_target)
        self.assertEqual(forged.manifest_id, trusted.manifest_id)
        self.assertEqual(forged.manifest_sha256, trusted.manifest_sha256)
        self.assertNotEqual(
            forged.index.vector_build_sha256,
            trusted.index.vector_build_sha256,
        )
        self.assertNotEqual(forged.stable_id, trusted.stable_id)

        decision = SemanticCacheResolver(
            [self.bridge],
            trusted_materializations=(trusted,),
        ).resolve(
            self.new_query,
            self.resident,
            materialized_indexes=[forged],
        )
        self.assertEqual(decision.mode, ResolutionMode.ABSTAIN)
        self.assertEqual(
            decision.reason,
            ResolutionReason.UNTRUSTED_MATERIALIZATION_CONTRACT,
        )
        self.assertFalse(decision.can_search)

    def test_self_declared_forged_materialization_is_not_trusted(self):
        forged = _materialized(
            self.bridge,
            self.resident,
            self.target,
            manifest_id="manifest:attacker-declared",
            manifest_sha256=_sha("attacker-declared-manifest"),
        )
        decision = SemanticCacheResolver([self.bridge]).resolve(
            self.new_query,
            self.resident,
            materialized_indexes=[forged],
        )
        self.assertEqual(decision.mode, ResolutionMode.ABSTAIN)
        self.assertEqual(
            decision.reason,
            ResolutionReason.UNTRUSTED_MATERIALIZATION_CONTRACT,
        )
        self.assertFalse(decision.can_search)

    def test_post_configuration_materialization_tampering_fails_closed(self):
        reference = _materialized(self.bridge, self.resident, self.target)
        resolver = SemanticCacheResolver(
            [self.bridge],
            trusted_materializations=(reference,),
        )
        trusted_stable_id = reference.stable_id
        object.__setattr__(
            reference,
            "manifest_sha256",
            _sha("post-configuration-forged-manifest"),
        )
        self.assertNotEqual(reference.stable_id, trusted_stable_id)

        decision = resolver.resolve(
            self.new_query,
            self.resident,
            materialized_indexes=[reference],
        )
        self.assertEqual(decision.mode, ResolutionMode.ABSTAIN)
        self.assertEqual(
            decision.reason,
            ResolutionReason.UNTRUSTED_MATERIALIZATION_CONTRACT,
        )
        self.assertFalse(decision.can_search)

    def test_materialization_trust_store_rejects_mismatched_id_and_digest(self):
        reference = _materialized(self.bridge, self.resident, self.target)
        with self.assertRaisesRegex(
            ResolutionError,
            "stable ID does not bind its content digest",
        ):
            SemanticCacheResolver(
                [self.bridge],
                trusted_materialization_contracts={
                    reference.stable_id: _sha("different-materialization"),
                },
            )

    def test_same_encoder_over_different_corpus_snapshot_is_rejected(self):
        wrong_snapshot_target = _resident("new", snapshot="corpus-b", build="bridge-build")
        reference = _materialized(self.bridge, self.resident, wrong_snapshot_target)
        with self.assertRaisesRegex(ResolutionError, "different corpus snapshot"):
            SemanticCacheResolver(
                [self.bridge],
                trusted_materializations=(reference,),
            ).resolve(
                self.new_query,
                self.resident,
                materialized_indexes=[reference],
            )

    def test_materialization_from_different_source_index_is_rejected(self):
        other_source = _resident("old", snapshot="corpus-a", build="different-old-build")
        reference = _materialized(self.bridge, other_source, self.target)
        with self.assertRaisesRegex(ResolutionError, "different resident index snapshot"):
            SemanticCacheResolver(
                [self.bridge],
                trusted_materializations=(reference,),
            ).resolve(
                self.new_query,
                self.resident,
                materialized_indexes=[reference],
            )

    def test_materialized_index_with_wrong_query_role_pair_is_rejected(self):
        unrelated_target = _resident("third", snapshot="corpus-a", build="bridge-build")
        reference = MaterializedIndexRef(
            bridge_id=self.bridge.stable_id,
            source_index_id=self.resident.stable_id,
            index=unrelated_target,
            manifest_id="manifest:wrong-target",
            manifest_sha256=_sha("manifest:wrong-target"),
        )
        with self.assertRaisesRegex(ResolutionError, "document space"):
            SemanticCacheResolver(
                [self.bridge],
                trusted_materializations=(reference,),
            ).resolve(
                self.new_query,
                self.resident,
                materialized_indexes=[reference],
            )

    def test_unregistered_materialization_is_rejected(self):
        reference = _materialized(self.bridge, self.resident, self.target)
        with self.assertRaisesRegex(ResolutionError, "unregistered bridges"):
            SemanticCacheResolver(
                trusted_materializations=(reference,),
            ).resolve(
                self.new_query,
                self.resident,
                materialized_indexes=[reference],
            )

    def test_duplicate_materialization_is_rejected(self):
        reference = _materialized(self.bridge, self.resident, self.target)
        with self.assertRaisesRegex(ResolutionError, "duplicate materialized index"):
            SemanticCacheResolver(
                [self.bridge],
                trusted_materializations=(reference,),
            ).resolve(
                self.new_query,
                self.resident,
                materialized_indexes=[reference, reference],
            )


class _FixedScorer:
    def __init__(
        self,
        scores,
        *,
        contract=None,
        implementation_label="fixed-scorer-implementation-v1",
        attestation_overrides=None,
    ):
        self.scores = scores
        self.calls = []
        contract = contract or _scorer_contract()
        attestation = contract.expected_runtime_attestation()
        if implementation_label != "fixed-scorer-implementation-v1":
            implementation_sha256 = _sha(implementation_label)
            attestation = replace(
                attestation,
                implementation_sha256=implementation_sha256,
                scorer_stable_id=("route-scorer-implementation:v1:" f"sha256:{implementation_sha256}"),
            )
        if attestation_overrides:
            attestation = replace(attestation, **attestation_overrides)
        self.runtime_attestation = attestation

    def score_routes(self, **kwargs):
        self.calls.append(kwargs)
        return self.scores


class _ExplodingScorer:
    runtime_attestation = _scorer_contract().expected_runtime_attestation()

    def score_routes(self, **kwargs):
        raise RuntimeError("model unavailable")


class _DuplicateItemsMapping(Mapping):
    def __init__(self, bridge_id):
        self.bridge_id = bridge_id

    def __getitem__(self, key):
        if key == self.bridge_id:
            return 0.9
        raise KeyError(key)

    def __iter__(self):
        return iter((self.bridge_id,))

    def __len__(self):
        return 1

    def items(self):
        return ((self.bridge_id, 0.9), (self.bridge_id, 0.8))


class TestRouteScorerContractAuthority(unittest.TestCase):
    def setUp(self):
        self.resident = _resident("old")
        self.new_query, _ = _pair("new")
        self.bridge = _bridge(
            self.new_query,
            self.resident.query_space,
            name="contract-authority",
        )
        self.scorer = _FixedScorer({self.bridge.stable_id: 0.9})
        self.contract = _scorer_contract()

    def test_contract_is_content_addressed_and_round_trips(self):
        raw = self.contract.to_dict()
        restored = RouteScorerContract.from_dict(json.loads(json.dumps(raw)))
        self.assertEqual(restored, self.contract)
        self.assertEqual(restored.stable_id, self.contract.stable_id)
        self.assertEqual(raw["content_sha256"], self.contract.content_sha256)
        attestation = self.contract.expected_runtime_attestation()
        self.assertIsInstance(attestation, RouteScorerRuntimeAttestation)
        self.assertEqual(
            attestation.contract_content_sha256,
            self.contract.content_sha256,
        )

    def test_mutated_serialized_contract_is_rejected(self):
        raw = self.contract.to_dict()
        raw["model_sha256"] = _sha("substituted-model")
        with self.assertRaisesRegex(ResolutionError, "content_sha256 does not match"):
            RouteScorerContract.from_dict(raw)

    def test_feature_manifest_digest_must_match_typed_schema(self):
        with self.assertRaisesRegex(
            ResolutionError,
            "feature_manifest_sha256 does not match",
        ):
            replace(
                self.contract,
                feature_schema=(_route_feature("query_length", "int"),),
            )

    def test_feature_manifest_binds_complete_declaration(self):
        declaration = _route_feature("query_length", "int")
        manifest_sha256 = route_feature_manifest_sha256((declaration,))
        mutations = {
            "name": replace(declaration, name="token_count"),
            "type": replace(declaration, type="float"),
            "qrels_free": replace(declaration, qrels_free=False),
            "availability_stage": replace(
                declaration,
                availability_stage="POST_SEARCH",
            ),
            "execution_cost_class": replace(
                declaration,
                execution_cost_class="one_search",
            ),
            "implementation_sha256": replace(
                declaration,
                implementation_sha256=_sha("substituted-feature-code"),
            ),
            "provenance_sha256": replace(
                declaration,
                provenance_sha256=_sha("substituted-feature-provenance"),
            ),
        }
        for field_name, mutated in mutations.items():
            with self.subTest(field=field_name):
                self.assertNotEqual(
                    route_feature_manifest_sha256((mutated,)),
                    manifest_sha256,
                )

        contract = _scorer_contract(feature_schema=(declaration,))
        self.assertEqual(
            set(contract.to_dict()["feature_schema"][0]),
            {
                "name",
                "type",
                "qrels_free",
                "availability_stage",
                "execution_cost_class",
                "implementation_sha256",
                "provenance_sha256",
            },
        )

    def test_trusted_manifest_cannot_allow_post_search_or_leaking_features(self):
        unsafe_declarations = (
            (
                _route_feature(
                    "ndcg_post_search_score",
                    "float",
                    availability_stage="POST_SEARCH",
                    execution_cost_class="one_search",
                ),
                "availability_stage PRE_SEARCH",
            ),
            (
                _route_feature("ndcg", "float"),
                "forbidden leakage token",
            ),
            (
                _route_feature(
                    "query_length",
                    "int",
                    qrels_free=False,
                ),
                "qrels_free=true",
            ),
            (
                _route_feature(
                    "query_length",
                    "int",
                    availability_stage="POST_SEARCH",
                ),
                "availability_stage PRE_SEARCH",
            ),
            (
                _route_feature(
                    "query_length",
                    "int",
                    execution_cost_class="one_search",
                ),
                "execution_cost_class zero_search",
            ),
        )
        for declaration, expected_error in unsafe_declarations:
            with self.subTest(
                feature=declaration.name,
                stage=declaration.availability_stage,
                cost=declaration.execution_cost_class,
            ):
                with self.assertRaisesRegex(ResolutionError, expected_error):
                    _scorer_contract(feature_schema=(declaration,))

    def test_configured_scorer_without_contract_is_rejected(self):
        with self.assertRaisesRegex(ResolutionError, "requires a RouteScorerContract"):
            _SemanticCacheResolver(
                [self.bridge],
                route_scorer=self.scorer,
                trusted_evidence_sha256s=(self.bridge.evidence_sha256,),
            )

    def test_untrusted_contract_is_rejected(self):
        with self.assertRaisesRegex(ResolutionError, "not present in the runtime trust store"):
            _SemanticCacheResolver(
                [self.bridge],
                route_scorer=self.scorer,
                route_scorer_contract=self.contract,
                trusted_evidence_sha256s=(self.bridge.evidence_sha256,),
            )

    def test_mutated_contract_does_not_reuse_prior_trust(self):
        mutated = replace(
            self.contract,
            model_sha256=_sha("mutated-model"),
        )
        self.assertNotEqual(mutated.content_sha256, self.contract.content_sha256)
        with self.assertRaisesRegex(ResolutionError, "not present in the runtime trust store"):
            _SemanticCacheResolver(
                [self.bridge],
                route_scorer=self.scorer,
                route_scorer_contract=mutated,
                trusted_evidence_sha256s=(self.bridge.evidence_sha256,),
                trusted_route_scorer_contract_sha256s=(self.contract.content_sha256,),
            )

    def test_scorer_swap_under_unchanged_contract_is_rejected(self):
        swapped = _FixedScorer(
            {self.bridge.stable_id: 0.99},
            implementation_label="substituted-scorer",
        )
        with self.assertRaisesRegex(
            ResolutionError,
            "mismatch for scorer_stable_id",
        ):
            _SemanticCacheResolver(
                [self.bridge],
                route_scorer=swapped,
                route_scorer_contract=self.contract,
                trusted_evidence_sha256s=(self.bridge.evidence_sha256,),
                trusted_route_scorer_contract_sha256s=(self.contract.content_sha256,),
            )

    def test_same_identity_with_substituted_artifact_digest_is_rejected(self):
        for field_name in (
            "model_sha256",
            "scaler_sha256",
            "feature_manifest_sha256",
            "confidence_rule_sha256",
            "evidence_sha256",
        ):
            with self.subTest(field_name=field_name):
                substituted = _FixedScorer(
                    {self.bridge.stable_id: 0.99},
                    contract=self.contract,
                    attestation_overrides={
                        field_name: _sha(f"substituted:{field_name}"),
                    },
                )
                self.assertEqual(
                    substituted.runtime_attestation.scorer_stable_id,
                    self.contract.scorer_stable_id,
                )
                self.assertEqual(
                    substituted.runtime_attestation.implementation_sha256,
                    self.contract.implementation_sha256,
                )
                with self.assertRaisesRegex(
                    ResolutionError,
                    f"mismatch for {field_name}",
                ):
                    _SemanticCacheResolver(
                        [self.bridge],
                        route_scorer=substituted,
                        route_scorer_contract=self.contract,
                        trusted_evidence_sha256s=(self.bridge.evidence_sha256,),
                        trusted_route_scorer_contract_sha256s=(self.contract.content_sha256,),
                    )

    def test_post_configuration_artifact_attestation_mutation_fails_closed(self):
        for field_name in (
            "model_sha256",
            "scaler_sha256",
            "feature_manifest_sha256",
            "confidence_rule_sha256",
            "evidence_sha256",
        ):
            with self.subTest(field_name=field_name):
                scorer = _FixedScorer(
                    {self.bridge.stable_id: 0.99},
                    contract=self.contract,
                )
                resolver = SemanticCacheResolver(
                    [self.bridge],
                    route_scorer=scorer,
                    route_scorer_contract=self.contract,
                )
                object.__setattr__(
                    scorer.runtime_attestation,
                    field_name,
                    _sha(f"post-configuration:{field_name}"),
                )
                decision = resolver.resolve(self.new_query, self.resident)
                self.assertEqual(decision.mode, ResolutionMode.ABSTAIN)
                self.assertEqual(
                    decision.reason,
                    ResolutionReason.ROUTER_FAILURE,
                )
                self.assertEqual(
                    decision.route_failure_class,
                    "ResolutionError",
                )
                self.assertEqual(scorer.calls, [])

    def test_resolver_threshold_must_match_trusted_policy(self):
        with self.assertRaisesRegex(
            ResolutionError,
            "minimum_route_confidence does not match",
        ):
            _SemanticCacheResolver(
                [self.bridge],
                minimum_route_confidence=0.9,
                route_scorer=self.scorer,
                route_scorer_contract=self.contract,
                trusted_evidence_sha256s=(self.bridge.evidence_sha256,),
                trusted_route_scorer_contract_sha256s=(self.contract.content_sha256,),
            )

    def test_post_configuration_scorer_swap_fails_closed_with_replay(self):
        resolver = SemanticCacheResolver(
            [self.bridge],
            route_scorer=self.scorer,
            route_scorer_contract=self.contract,
        )
        swapped = _FixedScorer(
            {self.bridge.stable_id: 0.99},
            implementation_label="substituted-scorer",
        )
        resolver._route_scorer = swapped
        decision = resolver.resolve(self.new_query, self.resident)
        self.assertEqual(decision.mode, ResolutionMode.ABSTAIN)
        self.assertEqual(decision.reason, ResolutionReason.ROUTER_FAILURE)
        self.assertEqual(decision.route_failure_class, "ResolutionError")
        self.assertEqual(decision.route_policy_id, self.contract.stable_id)
        self.assertRegex(decision.route_invocation_sha256, r"^[0-9a-f]{64}$")
        self.assertEqual(
            decision.route_candidate_ids,
            (self.bridge.stable_id,),
        )
        self.assertEqual(swapped.calls, [])

    def test_post_configuration_contract_tampering_fails_closed_at_resolve(self):
        resolver = SemanticCacheResolver(
            [self.bridge],
            route_scorer=self.scorer,
            route_scorer_contract=self.contract,
        )
        object.__setattr__(
            self.contract,
            "implementation_sha256",
            _sha("post-configuration-substitution"),
        )
        decision = resolver.resolve(self.new_query, self.resident)
        self.assertEqual(decision.mode, ResolutionMode.ABSTAIN)
        self.assertEqual(decision.reason, ResolutionReason.ROUTER_FAILURE)
        self.assertEqual(self.scorer.calls, [])

    def test_trusted_feature_declaration_tampering_fails_closed(self):
        contract = _scorer_contract(
            "nested-feature-tamper-policy",
            feature_schema=(_route_feature("query_length", "int"),),
        )
        scorer = _FixedScorer(
            {self.bridge.stable_id: 0.9},
            contract=contract,
        )
        resolver = SemanticCacheResolver(
            [self.bridge],
            route_scorer=scorer,
            route_scorer_contract=contract,
        )
        trusted_content_sha256 = contract.content_sha256
        declaration = contract.feature_schema[0]
        object.__setattr__(declaration, "name", "ndcg_post_search_score")
        object.__setattr__(declaration, "availability_stage", "POST_SEARCH")
        object.__setattr__(declaration, "execution_cost_class", "one_search")
        self.assertNotEqual(contract.content_sha256, trusted_content_sha256)

        decision = resolver.resolve(
            self.new_query,
            self.resident,
            query_context={
                "query_input": "alpha",
                "features": {"query_length": 5},
            },
        )
        self.assertEqual(decision.mode, ResolutionMode.ABSTAIN)
        self.assertEqual(decision.reason, ResolutionReason.ROUTER_FAILURE)
        self.assertEqual(decision.route_failure_class, "ResolutionError")
        self.assertEqual(scorer.calls, [])

    def test_contract_without_scorer_is_rejected(self):
        with self.assertRaisesRegex(ResolutionError, "requires a configured route_scorer"):
            SemanticCacheResolver(
                [self.bridge],
                route_scorer_contract=self.contract,
            )


class TestDeterminismAndPluggableRouting(unittest.TestCase):
    def setUp(self):
        self.resident = _resident("old")
        self.new_query, self.new_document = _pair("new")
        self.reverse_a = _bridge(
            self.new_query,
            self.resident.query_space,
            name="reverse-a",
            confidence=0.95,
        )
        self.reverse_b = _bridge(
            self.new_query,
            self.resident.query_space,
            name="reverse-b",
            confidence=0.95,
        )
        self.forward = _bridge(
            self.resident.document_space,
            self.new_document,
            name="forward",
            confidence=0.95,
        )
        self.target = _resident("new", build="bridge-build")
        self.materialization = _materialized(self.forward, self.resident, self.target)

    def test_bridge_registration_order_does_not_change_tie_result(self):
        scores = {
            self.reverse_a.stable_id: 0.95,
            self.reverse_b.stable_id: 0.95,
        }
        first = SemanticCacheResolver(
            [self.reverse_a, self.reverse_b],
            route_scorer=_FixedScorer(scores),
        ).resolve(
            self.new_query,
            self.resident,
        )
        second = SemanticCacheResolver(
            [self.reverse_b, self.reverse_a],
            route_scorer=_FixedScorer(scores),
        ).resolve(
            self.new_query,
            self.resident,
        )
        self.assertEqual(first.bridge.stable_id, second.bridge.stable_id)
        self.assertEqual(
            first.bridge.stable_id,
            min(self.reverse_a.stable_id, self.reverse_b.stable_id),
        )

    def test_equal_confidence_prefers_zero_write_reverse(self):
        decision = SemanticCacheResolver(
            [self.forward, self.reverse_a],
            trusted_materializations=(self.materialization,),
            route_scorer=_FixedScorer(
                {
                    self.forward.stable_id: 0.95,
                    self.reverse_a.stable_id: 0.95,
                }
            ),
        ).resolve(
            self.new_query,
            self.resident,
            materialized_indexes=[self.materialization],
        )
        self.assertEqual(decision.mode, ResolutionMode.REVERSE)

    def test_multiple_validated_routes_without_router_abstain(self):
        decision = SemanticCacheResolver(
            [self.forward, self.reverse_a],
            trusted_materializations=(self.materialization,),
        ).resolve(
            self.new_query,
            self.resident,
            materialized_indexes=[self.materialization],
        )
        self.assertEqual(decision.mode, ResolutionMode.ABSTAIN)
        self.assertEqual(decision.reason, ResolutionReason.ROUTER_REQUIRED)

    def test_higher_per_query_score_can_select_forward(self):
        decision = SemanticCacheResolver(
            [self.reverse_a, self.forward],
            trusted_materializations=(self.materialization,),
            minimum_route_confidence=0.8,
            route_scorer=_FixedScorer(
                {
                    self.reverse_a.stable_id: 0.80,
                    self.forward.stable_id: 0.94,
                }
            ),
        ).resolve(
            self.new_query,
            self.resident,
            materialized_indexes=[self.materialization],
        )
        self.assertEqual(decision.mode, ResolutionMode.FORWARD)

    def test_injected_scorer_is_called_with_candidate_contracts(self):
        scorer = _FixedScorer({self.reverse_a.stable_id: 0.9})
        decision = SemanticCacheResolver([self.reverse_a], route_scorer=scorer).resolve(
            self.new_query,
            self.resident,
            query_context={"query_input": "q-1", "features": {}},
        )
        self.assertEqual(decision.mode, ResolutionMode.REVERSE)
        self.assertEqual(len(scorer.calls), 1)
        self.assertEqual(
            scorer.calls[0]["query_context"],
            {"query_input": "q-1", "features": {}},
        )
        self.assertEqual(scorer.calls[0]["candidates"], (self.reverse_a,))

    def test_route_decision_serializes_policy_input_and_full_score_vector(self):
        scores = {
            self.reverse_a.stable_id: 0.94,
            self.reverse_b.stable_id: 0.81,
        }
        contract = _scorer_contract(
            "serialization-policy",
            feature_schema=(
                _route_feature("entropy", "float"),
                _route_feature("token_count", "int"),
            ),
        )
        decision = SemanticCacheResolver(
            [self.reverse_a, self.reverse_b],
            route_scorer=_FixedScorer(scores, contract=contract),
            route_scorer_contract=contract,
        ).resolve(
            self.new_query,
            self.resident,
            query_context={
                "query_input": "repair cache routing",
                "features": {"token_count": 3, "entropy": 0.25},
            },
        )
        raw = decision.to_dict()
        self.assertEqual(raw["route_policy_id"], contract.stable_id)
        self.assertEqual(raw["route_policy_sha256"], contract.content_sha256)
        self.assertEqual(raw["route_scorer_id"], contract.scorer_stable_id)
        self.assertEqual(
            raw["route_scorer_implementation_sha256"],
            contract.implementation_sha256,
        )
        self.assertEqual(
            raw["route_minimum_confidence"],
            contract.minimum_route_confidence,
        )
        self.assertEqual(
            raw["route_selection_rule_id"],
            contract.selection_rule_id,
        )
        self.assertEqual(
            raw["route_confidence_rule_sha256"],
            contract.confidence_rule_sha256,
        )
        self.assertRegex(raw["route_query_input_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(raw["route_invocation_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(raw["route_candidate_ids"], sorted(scores))
        self.assertEqual(
            raw["route_scores"],
            [{"bridge_id": bridge_id, "confidence": scores[bridge_id]} for bridge_id in sorted(scores)],
        )
        json.dumps(raw, allow_nan=False, sort_keys=True)

    def test_replay_digest_changes_when_query_input_changes(self):
        contract = _scorer_contract("query-replay-policy")
        kwargs = {
            "route_scorer": _FixedScorer(
                {self.reverse_a.stable_id: 0.9},
                contract=contract,
            ),
            "route_scorer_contract": contract,
        }
        first = SemanticCacheResolver([self.reverse_a], **kwargs).resolve(
            self.new_query,
            self.resident,
            query_context={"query_input": "alpha", "features": {}},
        )
        second = SemanticCacheResolver([self.reverse_a], **kwargs).resolve(
            self.new_query,
            self.resident,
            query_context={"query_input": "beta", "features": {}},
        )
        self.assertNotEqual(
            first.route_query_input_sha256,
            second.route_query_input_sha256,
        )
        self.assertNotEqual(
            first.route_invocation_sha256,
            second.route_invocation_sha256,
        )

    def test_invocation_digest_binds_eligible_candidate_action_set(self):
        contract = _scorer_contract("candidate-invocation-policy")
        one = SemanticCacheResolver(
            [self.reverse_a],
            route_scorer=_FixedScorer(
                {self.reverse_a.stable_id: 0.9},
                contract=contract,
            ),
            route_scorer_contract=contract,
        ).resolve(self.new_query, self.resident)
        two = SemanticCacheResolver(
            [self.reverse_a, self.reverse_b],
            route_scorer=_FixedScorer(
                {
                    self.reverse_a.stable_id: 0.9,
                    self.reverse_b.stable_id: 0.85,
                },
                contract=contract,
            ),
            route_scorer_contract=contract,
        ).resolve(self.new_query, self.resident)
        self.assertNotEqual(one.route_candidate_ids, two.route_candidate_ids)
        self.assertNotEqual(
            one.route_invocation_sha256,
            two.route_invocation_sha256,
        )

    def test_threshold_policy_changes_identity_and_route_outcome(self):
        low_contract = _scorer_contract(
            "threshold-low",
            minimum_route_confidence=0.8,
        )
        high_contract = _scorer_contract(
            "threshold-high",
            minimum_route_confidence=0.9,
        )
        scores = {self.reverse_a.stable_id: 0.85}
        routed = SemanticCacheResolver(
            [self.reverse_a],
            minimum_route_confidence=0.8,
            route_scorer=_FixedScorer(scores, contract=low_contract),
            route_scorer_contract=low_contract,
        ).resolve(self.new_query, self.resident)
        abstained = SemanticCacheResolver(
            [self.reverse_a],
            minimum_route_confidence=0.9,
            route_scorer=_FixedScorer(scores, contract=high_contract),
            route_scorer_contract=high_contract,
        ).resolve(self.new_query, self.resident)
        self.assertEqual(routed.mode, ResolutionMode.REVERSE)
        self.assertEqual(abstained.mode, ResolutionMode.ABSTAIN)
        self.assertEqual(
            abstained.reason,
            ResolutionReason.LOW_ROUTE_CONFIDENCE,
        )
        self.assertNotEqual(
            routed.route_policy_sha256,
            abstained.route_policy_sha256,
        )
        self.assertNotEqual(
            routed.route_invocation_sha256,
            abstained.route_invocation_sha256,
        )

    def test_invocation_digest_binds_query_and_resident_spaces(self):
        contract = _scorer_contract("space-invocation-policy")
        first = SemanticCacheResolver(
            [self.reverse_a],
            route_scorer=_FixedScorer(
                {self.reverse_a.stable_id: 0.9},
                contract=contract,
            ),
            route_scorer_contract=contract,
        ).resolve(self.new_query, self.resident)
        alternate_resident = _resident("alternate-old")
        alternate_bridge = _bridge(
            self.new_query,
            alternate_resident.query_space,
            name="alternate-space-route",
        )
        second = SemanticCacheResolver(
            [alternate_bridge],
            route_scorer=_FixedScorer(
                {alternate_bridge.stable_id: 0.9},
                contract=contract,
            ),
            route_scorer_contract=contract,
        ).resolve(self.new_query, alternate_resident)
        self.assertNotEqual(
            first.route_invocation_sha256,
            second.route_invocation_sha256,
        )

    def test_replay_vector_changes_when_unselected_score_changes(self):
        contract = _scorer_contract("score-replay-policy")
        first = SemanticCacheResolver(
            [self.reverse_a, self.reverse_b],
            route_scorer=_FixedScorer(
                {
                    self.reverse_a.stable_id: 0.95,
                    self.reverse_b.stable_id: 0.81,
                },
                contract=contract,
            ),
            route_scorer_contract=contract,
        ).resolve(self.new_query, self.resident)
        second = SemanticCacheResolver(
            [self.reverse_a, self.reverse_b],
            route_scorer=_FixedScorer(
                {
                    self.reverse_a.stable_id: 0.95,
                    self.reverse_b.stable_id: 0.82,
                },
                contract=contract,
            ),
            route_scorer_contract=contract,
        ).resolve(self.new_query, self.resident)
        self.assertEqual(first.bridge.stable_id, second.bridge.stable_id)
        self.assertNotEqual(first.route_scores, second.route_scores)

    def test_qrels_are_rejected_before_the_scorer_is_called(self):
        scorer = _FixedScorer({self.reverse_a.stable_id: 0.9})
        decision = SemanticCacheResolver(
            [self.reverse_a],
            route_scorer=scorer,
        ).resolve(
            self.new_query,
            self.resident,
            query_context={"query_input": "alpha", "qrels": {"doc-1": 1}},
        )
        self.assertEqual(decision.reason, ResolutionReason.ROUTER_FAILURE)
        self.assertEqual(scorer.calls, [])
        self.assertEqual(decision.route_failure_class, "ResolutionError")
        self.assertRegex(decision.route_invocation_sha256, r"^[0-9a-f]{64}$")

    def test_supervision_proxy_features_cannot_bypass_exact_allowlist(self):
        scorer = _FixedScorer({self.reverse_a.stable_id: 0.9})
        decision = SemanticCacheResolver(
            [self.reverse_a],
            route_scorer=scorer,
        ).resolve(
            self.new_query,
            self.resident,
            query_context={
                "query_input": "alpha",
                "features": {
                    "label": 1,
                    "ndcg": 1.0,
                    "relevance": 3,
                    "reward": 1.0,
                    "winner": "route-a",
                },
            },
        )
        self.assertEqual(decision.reason, ResolutionReason.ROUTER_FAILURE)
        self.assertEqual(scorer.calls, [])
        self.assertEqual(decision.route_failure_class, "ResolutionError")

    def test_feature_values_require_exact_contract_types(self):
        contract = _scorer_contract(
            "typed-feature-policy",
            feature_schema=(_route_feature("query_length", "int"),),
        )
        scorer = _FixedScorer(
            {self.reverse_a.stable_id: 0.9},
            contract=contract,
        )
        decision = SemanticCacheResolver(
            [self.reverse_a],
            route_scorer=scorer,
            route_scorer_contract=contract,
        ).resolve(
            self.new_query,
            self.resident,
            query_context={
                "query_input": "alpha",
                "features": {"query_length": True},
            },
        )
        self.assertEqual(decision.reason, ResolutionReason.ROUTER_FAILURE)
        self.assertEqual(scorer.calls, [])

    def test_scorer_exception_fails_closed_without_leaking_message(self):
        decision = SemanticCacheResolver(
            [self.reverse_a],
            route_scorer=_ExplodingScorer(),
        ).resolve(self.new_query, self.resident)
        self.assertEqual(decision.mode, ResolutionMode.ABSTAIN)
        self.assertEqual(decision.reason, ResolutionReason.ROUTER_FAILURE)
        self.assertIn("RuntimeError", decision.detail)
        self.assertNotIn("model unavailable", decision.detail)
        self.assertEqual(decision.route_failure_class, "RuntimeError")
        self.assertRegex(decision.route_policy_sha256, r"^[0-9a-f]{64}$")
        self.assertRegex(decision.route_invocation_sha256, r"^[0-9a-f]{64}$")
        self.assertEqual(
            decision.route_candidate_ids,
            (self.reverse_a.stable_id,),
        )
        self.assertEqual(decision.route_scores, ())

    def test_invalid_scorer_output_fails_closed(self):
        scorer = _FixedScorer({self.reverse_a.stable_id: float("nan")})
        decision = SemanticCacheResolver([self.reverse_a], route_scorer=scorer).resolve(
            self.new_query,
            self.resident,
        )
        self.assertEqual(decision.mode, ResolutionMode.ABSTAIN)
        self.assertEqual(decision.reason, ResolutionReason.ROUTER_FAILURE)

    def test_extra_route_score_fails_closed(self):
        scorer = _FixedScorer(
            {
                self.reverse_a.stable_id: 0.9,
                "bridge:v1:sha256:" + _sha("ineligible"): 0.9,
            }
        )
        decision = SemanticCacheResolver([self.reverse_a], route_scorer=scorer).resolve(
            self.new_query,
            self.resident,
        )
        self.assertEqual(decision.mode, ResolutionMode.ABSTAIN)
        self.assertEqual(decision.reason, ResolutionReason.ROUTER_FAILURE)

    def test_duplicate_route_score_fails_closed(self):
        scorer = _FixedScorer(_DuplicateItemsMapping(self.reverse_a.stable_id))
        decision = SemanticCacheResolver([self.reverse_a], route_scorer=scorer).resolve(
            self.new_query,
            self.resident,
        )
        self.assertEqual(decision.mode, ResolutionMode.ABSTAIN)
        self.assertEqual(decision.reason, ResolutionReason.ROUTER_FAILURE)

    def test_explicit_scores_and_injected_scorer_are_mutually_exclusive(self):
        scorer = _FixedScorer({self.reverse_a.stable_id: 0.9})
        with self.assertRaisesRegex(ResolutionError, "direct route_scores"):
            SemanticCacheResolver([self.reverse_a], route_scorer=scorer).resolve(
                self.new_query,
                self.resident,
                route_scores={self.reverse_a.stable_id: 0.9},
            )

    def test_duplicate_bridge_contract_is_rejected(self):
        with self.assertRaisesRegex(ResolutionError, "duplicate bridge"):
            SemanticCacheResolver([self.reverse_a, self.reverse_a])

    def test_shared_role_bridge_with_wrong_directional_cost_abstains(self):
        old_shared = _space("old", RepresentationRole.SHARED)
        new_shared = _space("new", RepresentationRole.SHARED)
        resident = ResidentIndexDescriptor(
            index_id="shared-index",
            document_space=old_shared,
            query_space=old_shared,
            corpus_snapshot_id="corpus-a",
            corpus_snapshot_sha256=_sha("corpus-a"),
            vector_build_id="shared-build",
            vector_build_sha256=_sha("shared-build"),
            document_count=100,
        )
        wrong_cost = _bridge(
            new_shared,
            old_shared,
            name="wrong-cost",
            cost=BridgeCost.MATERIALIZED_INDEX,
        )
        decision = SemanticCacheResolver([wrong_cost]).resolve(new_shared, resident)
        self.assertEqual(decision.mode, ResolutionMode.ABSTAIN)
        self.assertEqual(decision.reason, ResolutionReason.INVALID_BRIDGE_COST)


if __name__ == "__main__":
    unittest.main()
