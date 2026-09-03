"""Candidate-generator contracts for model and deterministic fixture paths."""

from __future__ import annotations

import ast
from dataclasses import dataclass
import json
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple

from ..canonical import digest_bytes, digest_for
from ..evaluation.diagnostics import REQUESTED_AUTHORITIES
from ..evaluation.prompts import PromptRegistry
from .adapter import SealedAdapterArtifact, validate_applied_peft_model
from .errors import VariationConfigurationError, VariationDependencyError
from .model import AdapterApplicationAttestation, LoadedPinnedModel, PinnedModelManifest, model_state_digest


_MODEL_GENERATOR_INIT_TOKEN = object()
_ORIGINAL_PROMPT_REGISTRY_VALIDATE = PromptRegistry.validate
_ORIGINAL_PROMPT_REGISTRY_RENDER = PromptRegistry.render
MODEL_RESPONSE_CONTRACTS = (
    "closed-json-v1",
    "source-only-v1",
    "source-only-prefill-v1",
)
MODEL_RESPONSE_CONTRACT_SCHEMA = "egv-model-response-contract-v1"
MODEL_GENERATION_PROFILE_SCHEMA = "egv-model-generation-profile-v1"


def model_response_contract_manifest(response_contract: str) -> Mapping[str, Any]:
    """Return the immutable host/model response boundary for one contract."""

    if response_contract not in MODEL_RESPONSE_CONTRACTS:
        raise VariationConfigurationError("unknown model response contract")
    common = {
        "schema_version": MODEL_RESPONSE_CONTRACT_SCHEMA,
        "response_contract": response_contract,
        "prompt_manifest_digest": PromptRegistry().manifest_digest(),
        "generation_mode": "deterministic-greedy-v1",
        "max_candidate_source_bytes": 256 * 1024,
    }
    if response_contract == "closed-json-v1":
        return {
            **common,
            "response_shape": "closed-json-candidate-envelope-v1",
            "host_binding": "model-declared-locus-authority-evidence-v1",
        }
    return {
        **common,
        "response_shape": "complete-python-module-v1",
        "host_binding": "trusted-locus-execute-authority-all-presented-evidence-v1",
        "source_input": "exact-src-task-py-utf8-v1",
        "python_admission": "whole-response-ast-and-public-locus-v1",
    }


def model_response_contract_digest(response_contract: str) -> str:
    return digest_for(model_response_contract_manifest(response_contract))


def model_generation_profile_manifest(
    response_contract: str,
    *,
    model_manifest_digest: str,
    chat_template_digest: str,
    max_new_tokens: int = 512,
) -> Mapping[str, Any]:
    if response_contract not in MODEL_RESPONSE_CONTRACTS:
        raise VariationConfigurationError("unknown model response contract")
    if (
        not isinstance(model_manifest_digest, str)
        or len(model_manifest_digest) != 64
        or not isinstance(chat_template_digest, str)
        or len(chat_template_digest) != 64
        or not isinstance(max_new_tokens, int)
        or isinstance(max_new_tokens, bool)
        or max_new_tokens <= 0
        or max_new_tokens > 2048
    ):
        raise VariationConfigurationError("model generation profile inputs are invalid")
    try:
        int(model_manifest_digest, 16)
        int(chat_template_digest, 16)
    except ValueError as exc:
        raise VariationConfigurationError("model generation profile digests are not hexadecimal") from exc
    return {
        "schema_version": MODEL_GENERATION_PROFILE_SCHEMA,
        "model_manifest_digest": model_manifest_digest,
        "response_contract": response_contract,
        "response_contract_digest": model_response_contract_digest(response_contract),
        "prompt_manifest_digest": PromptRegistry().manifest_digest(),
        "chat_template_digest": chat_template_digest,
        "max_new_tokens": max_new_tokens,
        "enable_thinking": False,
        "do_sample": False,
        "num_beams": 1,
        "skip_special_tokens": True,
        "source_only_chat_mode": (
            "assistant-continuation" if response_contract == "source-only-prefill-v1" else "generation-prompt"
        ),
    }


def model_generation_profile_digest(
    response_contract: str,
    *,
    model_manifest_digest: str,
    chat_template_digest: str,
    max_new_tokens: int = 512,
) -> str:
    return digest_for(
        model_generation_profile_manifest(
            response_contract,
            model_manifest_digest=model_manifest_digest,
            chat_template_digest=chat_template_digest,
            max_new_tokens=max_new_tokens,
        )
    )


CLOSED_JSON_RESPONSE_CONTRACT_DIGEST = model_response_contract_digest("closed-json-v1")
SOURCE_ONLY_RESPONSE_CONTRACT_DIGEST = model_response_contract_digest("source-only-v1")


@dataclass(frozen=True)
class CandidateContext:
    campaign_id: str
    run_id: str
    seed: int
    arm_id: str
    task_id: str
    family_id: str
    public_locus: str
    public_rule_id: str
    attempt_index: int
    parent_candidate_id: Optional[str]
    retrieval_records: Tuple[Mapping[str, Any], ...]
    retrieval_digest: str
    model_digest: str
    adapter_digest: Optional[str]
    prompt_digest: str
    task_statement: Optional[str] = None
    initial_source: Optional[str] = None
    initial_source_digest: Optional[str] = None
    response_contract: str = "closed-json-v1"
    response_contract_digest: str = CLOSED_JSON_RESPONSE_CONTRACT_DIGEST
    generation_profile_digest: Optional[str] = None


@dataclass(frozen=True)
class CandidateProposal:
    source: bytes
    declared_locus: str
    requested_authority: str
    evidence_ids: Tuple[str, ...]
    mutation_digest: str
    metadata: Mapping[str, Any]

    def validate(self, context: CandidateContext, *, source_limit: int) -> None:
        if not isinstance(self.source, bytes) or not self.source or len(self.source) > source_limit:
            raise VariationConfigurationError("candidate source is empty or exceeds the frozen byte ceiling")
        if self.declared_locus != context.public_locus:
            raise VariationConfigurationError("candidate proposal changes a locus outside the task contract")
        if self.requested_authority not in REQUESTED_AUTHORITIES:
            raise VariationConfigurationError("candidate proposal requests an unknown authority")
        if tuple(sorted(set(self.evidence_ids))) != self.evidence_ids:
            raise VariationConfigurationError("candidate evidence IDs must be sorted and unique")
        available = {str(record["event_id"]) for record in context.retrieval_records}
        if not set(self.evidence_ids).issubset(available):
            raise VariationConfigurationError("candidate cites an evidence event outside its retrieval view")
        if len(self.mutation_digest) != 64:
            raise VariationConfigurationError("candidate mutation digest is not SHA-256")
        try:
            int(self.mutation_digest, 16)
        except ValueError as exc:
            raise VariationConfigurationError("candidate mutation digest is not hexadecimal") from exc


@dataclass(frozen=True)
class CandidateGenerationEvidence:
    """Private raw generation paired with its validated trusted proposal."""

    proposal: CandidateProposal
    decoded_model_response: bytes
    decoded_model_response_digest: str
    contract_response: bytes
    contract_response_digest: str
    rendered_prompt: Optional[bytes]
    rendered_prompt_digest: Optional[str]
    response_contract: str

    def validate(self, context: CandidateContext) -> None:
        if self.response_contract not in MODEL_RESPONSE_CONTRACTS:
            raise VariationDependencyError("candidate generation evidence has an unknown response contract")
        if (
            context.response_contract != self.response_contract
            or context.response_contract_digest != model_response_contract_digest(self.response_contract)
        ):
            raise VariationDependencyError("candidate generation evidence differs from its frozen response contract")
        if not isinstance(self.decoded_model_response, bytes) or not self.decoded_model_response:
            raise VariationDependencyError("candidate generation evidence has an empty decoded model response")
        if digest_bytes(self.decoded_model_response) != self.decoded_model_response_digest:
            raise VariationDependencyError("candidate generation decoded-response digest mismatch")
        if not isinstance(self.contract_response, bytes) or not self.contract_response:
            raise VariationDependencyError("candidate generation evidence has an empty contract response")
        if digest_bytes(self.contract_response) != self.contract_response_digest:
            raise VariationDependencyError("candidate generation contract-response digest mismatch")
        if self.response_contract == "source-only-prefill-v1":
            if not isinstance(context.initial_source, str) or not context.initial_source.splitlines():
                raise VariationDependencyError("candidate generation prefill source is unavailable")
            expected_contract_response = (
                context.initial_source.splitlines()[0].encode("utf-8")
                + b"\n"
                + self.decoded_model_response
            )
        else:
            expected_contract_response = self.decoded_model_response
        if self.contract_response != expected_contract_response:
            raise VariationDependencyError(
                "candidate generation decoded response is not bound to its contract response"
            )
        if self.response_contract == "closed-json-v1":
            if self.rendered_prompt is not None or self.rendered_prompt_digest is not None:
                raise VariationDependencyError(
                    "closed-JSON generation does not expose rendered-prompt evidence"
                )
        elif (
            not isinstance(self.rendered_prompt, bytes)
            or not self.rendered_prompt
            or not isinstance(self.rendered_prompt_digest, str)
            or digest_bytes(self.rendered_prompt) != self.rendered_prompt_digest
            or self.rendered_prompt_digest != context.prompt_digest
        ):
            raise VariationDependencyError("candidate generation rendered-prompt digest mismatch")
        if type(self.proposal) is not CandidateProposal:
            raise VariationDependencyError("candidate generation evidence lacks the exact proposal type")
        self.proposal.validate(context, source_limit=256 * 1024)
        try:
            contract_text = self.contract_response.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise VariationDependencyError("candidate generation raw response is not UTF-8") from exc
        reparsed = ModelCandidateGenerator._parse_response(
            contract_text,
            context,
            response_contract=self.response_contract,
        )
        reparsed.validate(context, source_limit=256 * 1024)
        if reparsed != self.proposal:
            raise VariationDependencyError("candidate generation raw response differs from its proposal")
        if self.response_contract != "closed-json-v1" and (
            self.proposal.metadata.get("response_contract") != self.response_contract
            or self.proposal.metadata.get("contract_response_digest") != self.contract_response_digest
            or self.proposal.metadata.get("normalized_source_digest") != digest_bytes(self.proposal.source)
        ):
            raise VariationDependencyError("candidate generation proposal metadata is not evidence-bound")


@dataclass(frozen=True)
class CandidateGenerationFailureEvidence:
    """Private evidence for a deterministic generation/contract failure."""

    stage: str
    response_contract: str
    rendered_prompt: Optional[bytes]
    rendered_prompt_digest: Optional[str]
    decoded_model_response: Optional[bytes]
    decoded_model_response_digest: Optional[str]
    contract_response: Optional[bytes]
    contract_response_digest: Optional[str]
    error_code: str

    def validate(self, context: CandidateContext) -> None:
        if self.stage not in {
            "PROMPT_RENDER", "PROMPT_INTEGRITY", "MODEL_GENERATION", "RESPONSE_CONTRACT"
        }:
            raise VariationDependencyError("candidate generation failure stage is unknown")
        if (
            self.response_contract != context.response_contract
            or context.response_contract_digest != model_response_contract_digest(self.response_contract)
            or self.response_contract == "closed-json-v1"
        ):
            raise VariationDependencyError("candidate generation failure differs from its frozen response contract")
        if not isinstance(self.error_code, str) or not self.error_code or len(self.error_code) > 128:
            raise VariationDependencyError("candidate generation failure code is invalid")
        pairs = (
            (self.rendered_prompt, self.rendered_prompt_digest, "rendered prompt"),
            (self.decoded_model_response, self.decoded_model_response_digest, "decoded response"),
            (self.contract_response, self.contract_response_digest, "contract response"),
        )
        for raw, supplied, label in pairs:
            if raw is None:
                if supplied is not None:
                    raise VariationDependencyError("candidate generation failure {} digest is orphaned".format(label))
                continue
            if not isinstance(raw, bytes) or not isinstance(supplied, str) or digest_bytes(raw) != supplied:
                raise VariationDependencyError("candidate generation failure {} digest mismatch".format(label))
        if self.stage in {"MODEL_GENERATION", "RESPONSE_CONTRACT"}:
            if (
                not self.rendered_prompt
                or self.rendered_prompt_digest != context.prompt_digest
            ):
                raise VariationDependencyError("candidate generation failure is not bound to the rendered prompt")
        elif self.stage == "PROMPT_INTEGRITY" and (
            not self.rendered_prompt
            or self.rendered_prompt_digest == context.prompt_digest
        ):
            raise VariationDependencyError("prompt-integrity failure does not preserve the differing prompt bytes")
        if self.stage == "RESPONSE_CONTRACT" and (
            not self.decoded_model_response or not self.contract_response
        ):
            raise VariationDependencyError("response-contract failure lacks raw model evidence")


class CandidateGenerationFailure(VariationDependencyError):
    """Fail-closed generation error carrying only durable private evidence."""

    def __init__(self, evidence: CandidateGenerationFailureEvidence) -> None:
        super().__init__("pinned model candidate generation failed closed")
        self.evidence = evidence
        self.candidate_id: Optional[str] = None


class CandidateGenerator(Protocol):
    """Minimal generator boundary; it receives no hidden evaluator input."""

    model_digest: str
    adapter_digest: Optional[str]

    def propose(self, context: CandidateContext) -> CandidateProposal:
        ...


def render_candidate_prompt(
    context: CandidateContext,
    *,
    prompt_registry: Optional[PromptRegistry] = None,
    response_contract: str = "closed-json-v1",
) -> str:
    """Render the exact private candidate prompt from a frozen context.

    Keeping this pure lets the training-data builder reproduce and validate
    private prompts without constructing a model generator or reaching across
    the evaluator boundary.
    """

    if response_contract not in MODEL_RESPONSE_CONTRACTS:
        raise VariationConfigurationError("unknown model response contract")
    registry = prompt_registry or PromptRegistry()
    failure_families = sorted(
        {
            str(record.get("failure_family_root") or record.get("diagnostic_enum"))
            for record in context.retrieval_records
            if record.get("failure_family_root") or record.get("diagnostic_enum")
        }
    )
    corrections = sorted(
        str(record["event_id"])
        for record in context.retrieval_records
        if record.get("event_type") in {"CORRECTION", "RETRACTION"}
    )
    candidate = registry.render(
        "egv-candidate-v1",
        {
            "task_id": context.task_id,
            "family_id": context.family_id,
            "attempt_index": context.attempt_index,
            "public_locus": context.public_locus,
        },
    )
    evidence = registry.render(
        "egv-evidence-success-v1",
        {"evidence_ids": list(record["event_id"] for record in context.retrieval_records)},
    )
    failure = registry.render(
        "egv-evidence-failure-v1", {"failure_families": failure_families or ["none"]}
    )
    correction = registry.render(
        "egv-correction-v1", {"corrected_event_id": corrections[0] if corrections else "none"}
    )
    available_evidence_ids = sorted(str(record["event_id"]) for record in context.retrieval_records)
    if response_contract == "closed-json-v1":
        response_instruction = "".join((
            "Return exactly one JSON object with this closed field set: "
            '{"source": string, "declared_locus": string, "requested_authority": string, '
            '"evidence_ids": array[string], optional "metadata": object}. '
            "The source value must be the complete Python file encoded as a JSON string. "
            "declared_locus must equal ", json.dumps(context.public_locus),
            '. requested_authority must equal "EXECUTE_CANDIDATE". ',
            "evidence_ids must be a sorted unique subset of ", json.dumps(available_evidence_ids), ". "
            "Do not use Markdown fences, comments outside the object, or additional fields.",
        ))
    else:
        evidence = "".join((
            "Use the verified evidence context associated with ",
            json.dumps(available_evidence_ids),
            ". Do not emit evidence IDs or transport metadata in the Python file; "
            "the trusted host conservatively binds every presented evidence ID.",
        ))
        if (
            not isinstance(context.task_statement, str)
            or not context.task_statement.strip()
            or not isinstance(context.initial_source, str)
            or not context.initial_source.strip()
            or digest_bytes(context.initial_source.encode("utf-8")) != context.initial_source_digest
        ):
            raise VariationConfigurationError(
                "source-only response contract requires exact task statement and source bytes"
            )
        response_instruction = "".join((
            "Task statement: ", context.task_statement, "\n",
            "Current complete Python file:\n", context.initial_source, "\n",
            "Return only the complete Python file for ", json.dumps(context.public_locus), ". "
            "Begin with Python source and end with Python source. "
            "Do not return JSON, Markdown fences, explanations, prose, or transport metadata. "
            "The trusted host supplies the locus, authority, evidence binding, and digest after "
            "the entire response parses as one Python module.",
        ))
    return "\n".join(
        (
            registry.get("egv-system-v1").text,
            candidate,
            evidence,
            failure,
            correction,
            response_instruction,
        )
    )


class ModelCandidateGenerator:
    """Deterministic text-only generation with a closed JSON response contract."""

    def __setattr__(self, name: str, value: Any) -> None:
        if "_initialization_token" in self.__dict__ and name in {
            "loaded_model",
            "model",
            "tokenizer",
            "model_digest",
            "adapter_digest",
            "adapter_artifact",
            "adapter_attestation",
            "max_new_tokens",
            "prompt_registry",
            "prompt_manifest_digest",
            "chat_template_digest",
            "response_contract",
            "response_contract_digest",
            "generation_profile_digest",
        }:
            raise AttributeError("production model generator contract is immutable after construction")
        super().__setattr__(name, value)

    def __init__(
        self,
        loaded_model: Any,
        *,
        model_digest: str,
        adapter_digest: Optional[str] = None,
        adapter_artifact: Optional[SealedAdapterArtifact] = None,
        prompt_registry: Optional[PromptRegistry] = None,
        max_new_tokens: int = 512,
        response_contract: str = "closed-json-v1",
    ) -> None:
        if max_new_tokens <= 0 or max_new_tokens > 2048:
            raise VariationConfigurationError("model generation budget is outside the bounded contract")
        if response_contract not in MODEL_RESPONSE_CONTRACTS:
            raise VariationConfigurationError("unknown model response contract")
        if type(loaded_model) is not LoadedPinnedModel:
            raise VariationConfigurationError("ModelCandidateGenerator requires LoadedPinnedModel")
        if type(loaded_model.manifest) is not PinnedModelManifest:
            raise VariationDependencyError("LoadedPinnedModel manifest must be the frozen PinnedModelManifest")
        loaded_model.manifest.validate_contract()
        if loaded_model.manifest.digest() != loaded_model.manifest_digest or loaded_model.manifest_digest != model_digest:
            raise VariationConfigurationError("candidate generator model digest differs from the verified local manifest")
        if adapter_artifact is not None:
            if type(adapter_artifact) is not SealedAdapterArtifact:
                raise VariationDependencyError("LoRA adapter must be a sealed adapter artifact, not a digest-like object")
            adapter_artifact.verify()
            if adapter_digest is None:
                adapter_digest = adapter_artifact.digest
            if adapter_digest != adapter_artifact.digest:
                raise VariationConfigurationError("candidate generator adapter digest differs from the sealed artifact")
            if loaded_model.adapter_digest != adapter_artifact.digest:
                raise VariationDependencyError("sealed LoRA adapter was verified but not applied to the loaded model")
            attestation = loaded_model.adapter_attestation
            if type(attestation) is not AdapterApplicationAttestation:
                raise VariationDependencyError("sealed LoRA adapter lacks a loader-issued application attestation")
            attestation.validate()
            if (
                attestation.adapter_digest != adapter_artifact.digest
                or attestation.base_model_manifest_digest != loaded_model.manifest_digest
                or attestation.base_state_digest != loaded_model.base_state_digest
                or model_state_digest(loaded_model.model) != attestation.applied_model_state_digest
            ):
                raise VariationDependencyError("sealed LoRA adapter application attestation is not bound to the loaded model")
            validate_applied_peft_model(loaded_model.model, adapter_artifact)
        elif loaded_model.adapter_digest is not None or loaded_model.adapter_attestation is not None:
            raise VariationDependencyError("a loaded model with an applied adapter requires the matching sealed artifact")
        elif adapter_digest is not None:
            raise VariationDependencyError("a LoRA digest without a sealed adapter artifact is not accepted")
        self.loaded_model = loaded_model
        self.model = loaded_model.model
        self.tokenizer = loaded_model.tokenizer
        self.model_digest = model_digest
        self.adapter_digest = adapter_digest
        self.adapter_artifact = adapter_artifact
        self.adapter_attestation = loaded_model.adapter_attestation
        self.prompt_registry = prompt_registry or PromptRegistry()
        if response_contract != "closed-json-v1" and type(self.prompt_registry) is not PromptRegistry:
            raise VariationDependencyError("experimental source contract requires the exact PromptRegistry")
        self.prompt_registry.validate()
        self.prompt_manifest_digest = self.prompt_registry.manifest_digest()
        chat_template = getattr(self.tokenizer, "chat_template", None)
        if response_contract != "closed-json-v1" and (not isinstance(chat_template, str) or not chat_template):
            raise VariationDependencyError("experimental source contract requires pinned chat-template bytes")
        self.chat_template_digest = digest_bytes(chat_template.encode("utf-8")) if isinstance(chat_template, str) else None
        self.max_new_tokens = max_new_tokens
        self.response_contract = response_contract
        self.response_contract_digest = model_response_contract_digest(response_contract)
        if self.chat_template_digest is None:
            # closed-json-v1 keeps its historical tokenizing path and has no
            # exact rendered-prompt claim; a stable null-equivalent digest
            # still closes the generation profile.
            profile_chat_digest = digest_for("no-chat-template-digest-claim")
        else:
            profile_chat_digest = self.chat_template_digest
        self.generation_profile_digest = model_generation_profile_digest(
            response_contract,
            model_manifest_digest=self.model_digest,
            chat_template_digest=profile_chat_digest,
            max_new_tokens=self.max_new_tokens,
        )
        self._initialization_token = _MODEL_GENERATOR_INIT_TOKEN
        self._generator_contract = digest_for(
            {
                "loaded_model_id": id(self.loaded_model),
                "model_id": id(self.model),
                "tokenizer_id": id(self.tokenizer),
                "model_digest": self.model_digest,
                "adapter_digest": self.adapter_digest,
                "adapter_artifact_id": id(self.adapter_artifact) if self.adapter_artifact is not None else None,
                "max_new_tokens": self.max_new_tokens,
                "prompt_registry_id": id(self.prompt_registry),
                "prompt_manifest_digest": self.prompt_manifest_digest,
                "chat_template_digest": self.chat_template_digest,
                "response_contract": self.response_contract,
                "response_contract_digest": self.response_contract_digest,
                "generation_profile_digest": self.generation_profile_digest,
            }
        )

    def validate_production_integrity(self) -> None:
        """Revalidate the exact initialized model/adapter boundary."""

        if type(self) is not ModelCandidateGenerator:
            raise VariationDependencyError("production Variation requires the exact ModelCandidateGenerator type")
        if getattr(self, "_initialization_token", None) is not _MODEL_GENERATOR_INIT_TOKEN:
            raise VariationDependencyError("model generator initialization seal is missing")
        if any(
            name in self.__dict__
            for name in (
                "propose", "propose_with_evidence", "_generate_text", "_prompt",
                "_propose_with_evidence", "_render_chat", "prompt_digest_for",
            )
        ):
            raise VariationDependencyError("model generator prompt/propose cannot be overridden on an instance")
        if ModelCandidateGenerator.propose is not _ORIGINAL_MODEL_GENERATOR_PROPOSE:
            raise VariationDependencyError("model generator propose method was altered")
        if (
            ModelCandidateGenerator.propose_with_evidence is not _ORIGINAL_MODEL_GENERATOR_PROPOSE_EVIDENCE
            or ModelCandidateGenerator._generate_text is not _ORIGINAL_MODEL_GENERATOR_GENERATE_TEXT
            or ModelCandidateGenerator._propose_with_evidence is not _ORIGINAL_MODEL_GENERATOR_BUILD_EVIDENCE
            or ModelCandidateGenerator._parse_response is not _ORIGINAL_MODEL_GENERATOR_PARSE_RESPONSE
            or CandidateGenerationEvidence.validate is not _ORIGINAL_CANDIDATE_GENERATION_EVIDENCE_VALIDATE
        ):
            raise VariationDependencyError("model generator evidence-generation method was altered")
        if ModelCandidateGenerator._prompt is not _ORIGINAL_MODEL_GENERATOR_PROMPT:
            raise VariationDependencyError("model generator prompt method was altered")
        if (
            ModelCandidateGenerator._render_chat is not _ORIGINAL_MODEL_GENERATOR_RENDER_CHAT
            or ModelCandidateGenerator.prompt_digest_for is not _ORIGINAL_MODEL_GENERATOR_PROMPT_DIGEST
        ):
            raise VariationDependencyError("model generator chat rendering method was altered")
        if type(self.loaded_model) is not LoadedPinnedModel or type(self.loaded_model.manifest) is not PinnedModelManifest:
            raise VariationDependencyError("model generator loaded model identity is invalid")
        self.loaded_model.manifest.validate_contract()
        if self.loaded_model.manifest.digest() != self.loaded_model.manifest_digest:
            raise VariationDependencyError("model generator manifest digest is not self-consistent")
        if self.loaded_model.manifest_digest != self.model_digest:
            raise VariationDependencyError("model generator model digest binding changed")
        expected_contract = digest_for(
            {
                "loaded_model_id": id(self.loaded_model),
                "model_id": id(self.model),
                "tokenizer_id": id(self.tokenizer),
                "model_digest": self.model_digest,
                "adapter_digest": self.adapter_digest,
                "adapter_artifact_id": id(self.adapter_artifact) if self.adapter_artifact is not None else None,
                "max_new_tokens": self.max_new_tokens,
                "prompt_registry_id": id(self.prompt_registry),
                "prompt_manifest_digest": self.prompt_manifest_digest,
                "chat_template_digest": self.chat_template_digest,
                "response_contract": self.response_contract,
                "response_contract_digest": self.response_contract_digest,
                "generation_profile_digest": self.generation_profile_digest,
            }
        )
        if self._generator_contract != expected_contract:
            raise VariationDependencyError("model generator contract changed after construction")
        if self.response_contract_digest != model_response_contract_digest(self.response_contract):
            raise VariationDependencyError("model generator response-contract digest changed")
        profile_chat_digest = self.chat_template_digest or digest_for("no-chat-template-digest-claim")
        if self.generation_profile_digest != model_generation_profile_digest(
            self.response_contract,
            model_manifest_digest=self.model_digest,
            chat_template_digest=profile_chat_digest,
            max_new_tokens=self.max_new_tokens,
        ):
            raise VariationDependencyError("model generator generation profile changed")
        self.prompt_registry.validate()
        if self.prompt_registry.manifest_digest() != self.prompt_manifest_digest:
            raise VariationDependencyError("model generator prompt manifest binding changed")
        if self.response_contract != "closed-json-v1":
            if (
                type(self.prompt_registry) is not PromptRegistry
                or PromptRegistry.validate is not _ORIGINAL_PROMPT_REGISTRY_VALIDATE
                or PromptRegistry.render is not _ORIGINAL_PROMPT_REGISTRY_RENDER
            ):
                raise VariationDependencyError("experimental source prompt registry implementation changed")
            chat_template = getattr(self.tokenizer, "chat_template", None)
            if (
                not isinstance(chat_template, str)
                or digest_bytes(chat_template.encode("utf-8")) != self.chat_template_digest
            ):
                raise VariationDependencyError("experimental source chat-template bytes changed")
        if self.adapter_artifact is None:
            if self.adapter_digest is not None or self.loaded_model.adapter_digest is not None or self.loaded_model.adapter_attestation is not None:
                raise VariationDependencyError("base model generator carries unexpected adapter state")
            return
        if type(self.adapter_artifact) is not SealedAdapterArtifact:
            raise VariationDependencyError("model generator adapter identity is not SealedAdapterArtifact")
        self.adapter_artifact.verify()
        if self.adapter_digest != self.adapter_artifact.digest or self.loaded_model.adapter_digest != self.adapter_artifact.digest:
            raise VariationDependencyError("model generator adapter digest binding changed")
        attestation = self.adapter_attestation
        if type(attestation) is not AdapterApplicationAttestation:
            raise VariationDependencyError("model generator adapter attestation is not loader-issued")
        attestation.validate()
        if (
            attestation.adapter_digest != self.adapter_artifact.digest
            or attestation.base_model_manifest_digest != self.loaded_model.manifest_digest
            or attestation.base_state_digest != self.loaded_model.base_state_digest
            or model_state_digest(self.loaded_model.model) != attestation.applied_model_state_digest
        ):
            raise VariationDependencyError("model generator adapter application is not bound to current model state")
        validate_applied_peft_model(self.loaded_model.model, self.adapter_artifact)

    def _prompt(self, context: CandidateContext) -> str:
        return render_candidate_prompt(
            context,
            prompt_registry=self.prompt_registry,
            response_contract=self.response_contract,
        )

    def _render_chat(self, context: CandidateContext) -> str:
        apply_chat_template = getattr(self.tokenizer, "apply_chat_template", None)
        if not callable(apply_chat_template):
            raise VariationDependencyError("pinned Qwen tokenizer does not expose its official chat template")
        messages = [{"role": "user", "content": self._prompt(context)}]
        kwargs = {"tokenize": False, "enable_thinking": False}
        if self.response_contract == "source-only-prefill-v1":
            if not isinstance(context.initial_source, str) or not context.initial_source.splitlines():
                raise VariationConfigurationError("source-only prefill requires exact initial source")
            messages.append({"role": "assistant", "content": context.initial_source.splitlines()[0] + "\n"})
            kwargs.update({"add_generation_prompt": False, "continue_final_message": True})
        else:
            kwargs["add_generation_prompt"] = True
        rendered = apply_chat_template(messages, **kwargs)
        if not isinstance(rendered, str) or not rendered:
            raise VariationDependencyError("pinned Qwen chat template did not render exact text")
        return rendered

    def prompt_digest_for(self, context: CandidateContext) -> str:
        if (
            context.response_contract != self.response_contract
            or context.response_contract_digest != self.response_contract_digest
            or context.generation_profile_digest != self.generation_profile_digest
        ):
            raise VariationConfigurationError("candidate context differs from the frozen model generation profile")
        return digest_bytes(self._render_chat(context).encode("utf-8"))

    @staticmethod
    def _parse_response(
        text: str,
        context: CandidateContext,
        *,
        response_contract: str = "closed-json-v1",
    ) -> CandidateProposal:
        raw_response_bytes = text.encode("utf-8")
        stripped = text.strip()
        if response_contract not in MODEL_RESPONSE_CONTRACTS:
            raise VariationConfigurationError("unknown model response contract")
        if response_contract in {"source-only-v1", "source-only-prefill-v1"}:
            try:
                parsed_source = ast.parse(stripped)
            except (SyntaxError, ValueError) as exc:
                raise VariationDependencyError(
                    "pinned model did not emit the complete source-only candidate contract"
                ) from exc
            expected_function = context.public_locus.rsplit(":", 1)[-1]
            defined_functions = {
                node.name
                for node in parsed_source.body
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            if (
                not stripped
                or not expected_function.isidentifier()
                or expected_function not in defined_functions
            ):
                raise VariationDependencyError(
                    "pinned model did not emit the complete source-only candidate contract"
                )
            source_bytes = stripped.encode("utf-8")
            return CandidateProposal(
                source=source_bytes,
                declared_locus=context.public_locus,
                requested_authority="EXECUTE_CANDIDATE",
                evidence_ids=tuple(
                    sorted({str(record["event_id"]) for record in context.retrieval_records})
                ),
                mutation_digest=digest_bytes(source_bytes),
                metadata={
                    "response_contract": response_contract,
                    "contract_response_digest": digest_bytes(raw_response_bytes),
                    "normalized_source_digest": digest_bytes(source_bytes),
                },
            )
        try:
            value = json.loads(stripped)
        except ValueError as exc:
            # The base checkpoint can produce a valid bounded Python module while
            # omitting only the requested transport envelope. Repair exactly that
            # representation error without extracting Markdown, prose, or fragments.
            try:
                parsed_source = ast.parse(stripped)
            except (SyntaxError, ValueError) as source_exc:
                raise VariationDependencyError(
                    "pinned model did not emit the closed candidate JSON contract"
                ) from source_exc
            if not stripped or not any(
                isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) for node in parsed_source.body
            ):
                raise VariationDependencyError("pinned model did not emit the closed candidate JSON contract") from exc
            source_bytes = stripped.encode("utf-8")
            return CandidateProposal(
                source=source_bytes,
                declared_locus=context.public_locus,
                requested_authority="EXECUTE_CANDIDATE",
                evidence_ids=(),
                mutation_digest=digest_bytes(source_bytes),
                metadata={
                    "response_contract": "source-only-repair-v1",
                    "raw_response_digest": digest_bytes(source_bytes),
                },
            )
        if not isinstance(value, Mapping):
            raise VariationDependencyError("pinned model candidate response is not a JSON object")
        required = {"source", "declared_locus", "requested_authority", "evidence_ids"}
        if set(value) - required - {"metadata"} or not required.issubset(value):
            raise VariationDependencyError("pinned model candidate response has an unexpected field set")
        source = value["source"]
        declared_locus = value["declared_locus"]
        requested_authority = value["requested_authority"]
        evidence = value["evidence_ids"]
        metadata = value.get("metadata", {})
        if (
            not isinstance(source, str)
            or not isinstance(declared_locus, str)
            or not isinstance(requested_authority, str)
            or not isinstance(evidence, list)
            or not all(isinstance(item, str) for item in evidence)
            or not isinstance(metadata, Mapping)
            or not all(isinstance(key, str) for key in metadata)
        ):
            raise VariationDependencyError("pinned model candidate JSON has invalid source/evidence types")
        if requested_authority != "EXECUTE_CANDIDATE":
            raise VariationDependencyError("pinned model candidate JSON requests the wrong authority")
        source_bytes = source.encode("utf-8")
        return CandidateProposal(
            source=source_bytes,
            declared_locus=declared_locus,
            requested_authority=requested_authority,
            evidence_ids=tuple(evidence),
            mutation_digest=digest_bytes(source_bytes),
            metadata=dict(metadata),
        )

    def _generate_text(self, context: CandidateContext) -> Tuple[str, str, Optional[bytes]]:
        try:
            rendered_prompt = None
            if self.response_contract == "closed-json-v1":
                apply_chat_template = getattr(self.tokenizer, "apply_chat_template", None)
                if not callable(apply_chat_template):
                    raise VariationDependencyError("pinned Qwen tokenizer does not expose its official chat template")
                encoded = apply_chat_template(
                    [{"role": "user", "content": self._prompt(context)}],
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    enable_thinking=False,
                )
            else:
                rendered_chat = self._render_chat(context)
                rendered_prompt = rendered_chat.encode("utf-8")
                if digest_bytes(rendered_prompt) != context.prompt_digest:
                    raise VariationDependencyError("candidate prompt bytes differ from the frozen prompt digest")
                tokenize = getattr(self.tokenizer, "__call__", None)
                if not callable(tokenize):
                    raise VariationDependencyError("pinned Qwen tokenizer is not callable")
                encoded = tokenize(
                    rendered_chat,
                    add_special_tokens=False,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
            parameter = next(self.model.parameters())
            device = parameter.device
            if getattr(device, "type", str(device).split(":", 1)[0]) != "cuda":
                raise VariationDependencyError("production candidate generation requires a CUDA-resident model")
            if not isinstance(encoded, Mapping) or "input_ids" not in encoded:
                raise VariationDependencyError("pinned tokenizer did not return tensor input_ids")
            moved = {}
            for name, value in encoded.items():
                if not hasattr(value, "to"):
                    raise VariationDependencyError("pinned tokenizer returned a non-tensor generation input")
                tensor = value.to(device=device)
                if getattr(tensor, "device", None) != device:
                    raise VariationDependencyError("pinned tokenizer tensor did not move to the model device")
                moved[name] = tensor
            encoded = moved
            generated = self.model.generate(
                **encoded,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                num_beams=1,
                return_dict_in_generate=False,
            )
            input_length = int(encoded["input_ids"].shape[-1])
            tokens = generated[0][input_length:]
            decoded_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
            contract_text = decoded_text
            if self.response_contract == "source-only-prefill-v1":
                contract_text = context.initial_source.splitlines()[0] + "\n" + decoded_text
        except Exception as exc:
            raise VariationDependencyError("pinned model candidate generation failed") from exc
        return decoded_text, contract_text, rendered_prompt

    def _propose_with_evidence(self, context: CandidateContext) -> CandidateGenerationEvidence:
        decoded_text, contract_text, rendered_prompt = self._generate_text(context)
        proposal = self._parse_response(
            contract_text,
            context,
            response_contract=self.response_contract,
        )
        proposal.validate(context, source_limit=256 * 1024)
        evidence = CandidateGenerationEvidence(
            proposal=proposal,
            decoded_model_response=decoded_text.encode("utf-8"),
            decoded_model_response_digest=digest_bytes(decoded_text.encode("utf-8")),
            contract_response=contract_text.encode("utf-8"),
            contract_response_digest=digest_bytes(contract_text.encode("utf-8")),
            rendered_prompt=rendered_prompt,
            rendered_prompt_digest=(
                digest_bytes(rendered_prompt) if rendered_prompt is not None else None
            ),
            response_contract=self.response_contract,
        )
        evidence.validate(context)
        return evidence

    def propose(self, context: CandidateContext) -> CandidateProposal:
        return self._propose_with_evidence(context).proposal

    def propose_with_evidence(self, context: CandidateContext) -> CandidateGenerationEvidence:
        """Run the exact sealed generator while retaining private raw evidence."""

        if self.response_contract == "closed-json-v1":
            raise VariationConfigurationError(
                "generation evidence is available only for experimental source-only contracts"
            )
        self.validate_production_integrity()
        if (
            context.response_contract != self.response_contract
            or context.response_contract_digest != self.response_contract_digest
            or context.generation_profile_digest != self.generation_profile_digest
        ):
            raise VariationConfigurationError("candidate context differs from the frozen model generation profile")
        try:
            rendered_prompt = self._render_chat(context).encode("utf-8")
        except Exception as exc:
            failure = CandidateGenerationFailureEvidence(
                stage="PROMPT_RENDER",
                response_contract=self.response_contract,
                rendered_prompt=None,
                rendered_prompt_digest=None,
                decoded_model_response=None,
                decoded_model_response_digest=None,
                contract_response=None,
                contract_response_digest=None,
                error_code=type(exc).__name__,
            )
            failure.validate(context)
            raise CandidateGenerationFailure(failure) from exc
        if digest_bytes(rendered_prompt) != context.prompt_digest:
            failure = CandidateGenerationFailureEvidence(
                stage="PROMPT_INTEGRITY",
                response_contract=self.response_contract,
                rendered_prompt=rendered_prompt,
                rendered_prompt_digest=digest_bytes(rendered_prompt),
                decoded_model_response=None,
                decoded_model_response_digest=None,
                contract_response=None,
                contract_response_digest=None,
                error_code="PromptDigestMismatch",
            )
            failure.validate(context)
            raise CandidateGenerationFailure(failure)
        try:
            decoded_text, contract_text, generated_prompt = self._generate_text(context)
        except Exception as exc:
            failure = CandidateGenerationFailureEvidence(
                stage="MODEL_GENERATION",
                response_contract=self.response_contract,
                rendered_prompt=rendered_prompt,
                rendered_prompt_digest=digest_bytes(rendered_prompt),
                decoded_model_response=None,
                decoded_model_response_digest=None,
                contract_response=None,
                contract_response_digest=None,
                error_code=type(exc).__name__,
            )
            failure.validate(context)
            raise CandidateGenerationFailure(failure) from exc
        decoded = decoded_text.encode("utf-8")
        contract = contract_text.encode("utf-8")
        if generated_prompt != rendered_prompt:
            raise VariationDependencyError("candidate generation returned different rendered-prompt evidence")
        try:
            proposal = self._parse_response(
                contract_text,
                context,
                response_contract=self.response_contract,
            )
            proposal.validate(context, source_limit=256 * 1024)
        except Exception as exc:
            failure = CandidateGenerationFailureEvidence(
                stage="RESPONSE_CONTRACT",
                response_contract=self.response_contract,
                rendered_prompt=rendered_prompt,
                rendered_prompt_digest=digest_bytes(rendered_prompt),
                decoded_model_response=decoded,
                decoded_model_response_digest=digest_bytes(decoded),
                contract_response=contract,
                contract_response_digest=digest_bytes(contract),
                error_code=type(exc).__name__,
            )
            failure.validate(context)
            raise CandidateGenerationFailure(failure) from exc
        evidence = CandidateGenerationEvidence(
            proposal=proposal,
            decoded_model_response=decoded,
            decoded_model_response_digest=digest_bytes(decoded),
            contract_response=contract,
            contract_response_digest=digest_bytes(contract),
            rendered_prompt=rendered_prompt,
            rendered_prompt_digest=digest_bytes(rendered_prompt),
            response_contract=self.response_contract,
        )
        evidence.validate(context)
        return evidence


_ORIGINAL_MODEL_GENERATOR_PROPOSE = ModelCandidateGenerator.propose
_ORIGINAL_MODEL_GENERATOR_PROPOSE_EVIDENCE = ModelCandidateGenerator.propose_with_evidence
_ORIGINAL_MODEL_GENERATOR_GENERATE_TEXT = ModelCandidateGenerator._generate_text
_ORIGINAL_MODEL_GENERATOR_BUILD_EVIDENCE = ModelCandidateGenerator._propose_with_evidence
_ORIGINAL_MODEL_GENERATOR_PARSE_RESPONSE = ModelCandidateGenerator._parse_response
_ORIGINAL_MODEL_GENERATOR_PROMPT = ModelCandidateGenerator._prompt
_ORIGINAL_MODEL_GENERATOR_RENDER_CHAT = ModelCandidateGenerator._render_chat
_ORIGINAL_MODEL_GENERATOR_PROMPT_DIGEST = ModelCandidateGenerator.prompt_digest_for
_ORIGINAL_CANDIDATE_GENERATION_EVIDENCE_VALIDATE = CandidateGenerationEvidence.validate


class DeterministicFixtureGenerator:
    """Explicit smoke/test generator; never an authority or production model."""

    test_only = True

    def __init__(
        self,
        candidates: Mapping[str, Sequence[bytes]],
        *,
        public_locus: Mapping[str, str],
        model_digest: str = digest_for("fixture-model"),
        adapter_digest: Optional[str] = None,
        adapter_artifact: Optional[SealedAdapterArtifact] = None,
    ) -> None:
        if adapter_artifact is not None:
            if not isinstance(adapter_artifact, SealedAdapterArtifact):
                raise VariationDependencyError("LoRA adapter must be a sealed adapter artifact, not a digest-like object")
            adapter_artifact.verify()
            if adapter_digest is None:
                adapter_digest = adapter_artifact.digest
            if adapter_digest != adapter_artifact.digest:
                raise VariationConfigurationError("fixture adapter digest differs from the sealed artifact")
        elif adapter_digest is not None:
            raise VariationDependencyError("a LoRA digest without a sealed adapter artifact is not accepted")
        self.candidates = {task_id: tuple(bytes(source) for source in sources) for task_id, sources in candidates.items()}
        self.public_locus = dict(public_locus)
        self.model_digest = model_digest
        self.adapter_digest = adapter_digest
        self.adapter_artifact = adapter_artifact

    def propose(self, context: CandidateContext) -> CandidateProposal:
        try:
            sources = self.candidates[context.task_id]
            source = sources[min(context.attempt_index - 1, len(sources) - 1)]
            locus = self.public_locus[context.task_id]
        except (KeyError, IndexError) as exc:
            raise VariationDependencyError("fixture generator has no candidate for the requested task") from exc
        return CandidateProposal(
            source=source,
            declared_locus=locus,
            requested_authority="EXECUTE_CANDIDATE",
            evidence_ids=tuple(sorted(str(record["event_id"]) for record in context.retrieval_records)),
            mutation_digest=digest_bytes(source),
            metadata={"fixture_only": True},
        )


__all__ = [
    "CLOSED_JSON_RESPONSE_CONTRACT_DIGEST",
    "CandidateContext",
    "CandidateGenerationEvidence",
    "CandidateGenerationFailure",
    "CandidateGenerationFailureEvidence",
    "CandidateGenerator",
    "CandidateProposal",
    "DeterministicFixtureGenerator",
    "MODEL_GENERATION_PROFILE_SCHEMA",
    "MODEL_RESPONSE_CONTRACT_SCHEMA",
    "MODEL_RESPONSE_CONTRACTS",
    "ModelCandidateGenerator",
    "SOURCE_ONLY_RESPONSE_CONTRACT_DIGEST",
    "model_generation_profile_digest",
    "model_generation_profile_manifest",
    "model_response_contract_digest",
    "model_response_contract_manifest",
    "render_candidate_prompt",
]
