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
) -> str:
    """Render the exact private candidate prompt from a frozen context.

    Keeping this pure lets the training-data builder reproduce and validate
    private prompts without constructing a model generator or reaching across
    the evaluator boundary.
    """

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
    response_contract = "".join((
        "Return exactly one JSON object with this closed field set: "
        '{"source": string, "declared_locus": string, "requested_authority": string, '
        '"evidence_ids": array[string], optional "metadata": object}. '
        "The source value must be the complete Python file encoded as a JSON string. "
        "declared_locus must equal ", json.dumps(context.public_locus),
        '. requested_authority must equal "EXECUTE_CANDIDATE". ',
        "evidence_ids must be a sorted unique subset of ", json.dumps(available_evidence_ids), ". "
        "Do not use Markdown fences, comments outside the object, or additional fields.",
    ))
    return "\n".join(
        (
            registry.get("egv-system-v1").text,
            candidate,
            evidence,
            failure,
            correction,
            response_contract,
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
    ) -> None:
        if max_new_tokens <= 0 or max_new_tokens > 2048:
            raise VariationConfigurationError("model generation budget is outside the bounded contract")
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
        self.max_new_tokens = max_new_tokens
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
            }
        )

    def validate_production_integrity(self) -> None:
        """Revalidate the exact initialized model/adapter boundary."""

        if type(self) is not ModelCandidateGenerator:
            raise VariationDependencyError("production Variation requires the exact ModelCandidateGenerator type")
        if getattr(self, "_initialization_token", None) is not _MODEL_GENERATOR_INIT_TOKEN:
            raise VariationDependencyError("model generator initialization seal is missing")
        if "propose" in self.__dict__:
            raise VariationDependencyError("model generator propose cannot be overridden on an instance")
        if ModelCandidateGenerator.propose is not _ORIGINAL_MODEL_GENERATOR_PROPOSE:
            raise VariationDependencyError("model generator propose method was altered")
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
            }
        )
        if self._generator_contract != expected_contract:
            raise VariationDependencyError("model generator contract changed after construction")
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
        return render_candidate_prompt(context, prompt_registry=self.prompt_registry)

    @staticmethod
    def _parse_response(text: str, context: CandidateContext) -> CandidateProposal:
        stripped = text.strip()
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

    def propose(self, context: CandidateContext) -> CandidateProposal:
        prompt = self._prompt(context)
        try:
            apply_chat_template = getattr(self.tokenizer, "apply_chat_template", None)
            if not callable(apply_chat_template):
                raise VariationDependencyError("pinned Qwen tokenizer does not expose its official chat template")
            encoded = apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
                enable_thinking=False,
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
            text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        except Exception as exc:
            raise VariationDependencyError("pinned model candidate generation failed") from exc
        proposal = self._parse_response(text, context)
        proposal.validate(context, source_limit=256 * 1024)
        return proposal


_ORIGINAL_MODEL_GENERATOR_PROPOSE = ModelCandidateGenerator.propose


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
    "CandidateContext",
    "CandidateGenerator",
    "CandidateProposal",
    "DeterministicFixtureGenerator",
    "ModelCandidateGenerator",
    "render_candidate_prompt",
]
