"""Candidate-generator contracts for model and deterministic fixture paths."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Mapping, Optional, Protocol, Sequence, Tuple

from ..canonical import digest_bytes, digest_for
from ..evaluation.diagnostics import REQUESTED_AUTHORITIES
from ..evaluation.prompts import PromptRegistry
from .adapter import SealedAdapterArtifact
from .errors import VariationConfigurationError, VariationDependencyError


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


class ModelCandidateGenerator:
    """Deterministic text-only generation with a closed JSON response contract."""

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
        if not hasattr(loaded_model, "model") or not hasattr(loaded_model, "tokenizer"):
            raise VariationConfigurationError("ModelCandidateGenerator requires LoadedPinnedModel")
        loaded_manifest_digest = getattr(loaded_model, "manifest_digest", None)
        if loaded_manifest_digest is not None and loaded_manifest_digest != model_digest:
            raise VariationConfigurationError("candidate generator model digest differs from the verified local manifest")
        if adapter_artifact is not None:
            if not isinstance(adapter_artifact, SealedAdapterArtifact):
                raise VariationDependencyError("LoRA adapter must be a sealed adapter artifact, not a digest-like object")
            adapter_artifact.verify()
            if adapter_digest is None:
                adapter_digest = adapter_artifact.digest
            if adapter_digest != adapter_artifact.digest:
                raise VariationConfigurationError("candidate generator adapter digest differs from the sealed artifact")
            if getattr(loaded_model, "adapter_digest", None) != adapter_artifact.digest:
                raise VariationDependencyError("sealed LoRA adapter was verified but not applied to the loaded model")
        elif adapter_digest is not None:
            raise VariationDependencyError("a LoRA digest without a sealed adapter artifact is not accepted")
        self.loaded_model = loaded_model
        self.model = loaded_model.model
        self.tokenizer = loaded_model.tokenizer
        self.model_digest = model_digest
        self.adapter_digest = adapter_digest
        self.adapter_artifact = adapter_artifact
        self.prompt_registry = prompt_registry or PromptRegistry()
        self.max_new_tokens = max_new_tokens

    def _prompt(self, context: CandidateContext) -> str:
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
        candidate = self.prompt_registry.render(
            "egv-candidate-v1",
            {
                "task_id": context.task_id,
                "family_id": context.family_id,
                "attempt_index": context.attempt_index,
                "public_locus": context.public_locus,
            },
        )
        evidence = self.prompt_registry.render(
            "egv-evidence-success-v1",
            {"evidence_ids": list(record["event_id"] for record in context.retrieval_records)},
        )
        failure = self.prompt_registry.render(
            "egv-evidence-failure-v1", {"failure_families": failure_families or ["none"]}
        )
        correction = self.prompt_registry.render(
            "egv-correction-v1", {"corrected_event_id": corrections[0] if corrections else "none"}
        )
        return "\n".join(
            (
                self.prompt_registry.get("egv-system-v1").text,
                candidate,
                evidence,
                failure,
                correction,
                "Return JSON only.",
            )
        )

    @staticmethod
    def _parse_response(text: str, context: CandidateContext) -> CandidateProposal:
        stripped = text.strip()
        try:
            value = json.loads(stripped)
        except ValueError as exc:
            raise VariationDependencyError("pinned model did not emit the closed candidate JSON contract") from exc
        if not isinstance(value, Mapping):
            raise VariationDependencyError("pinned model candidate response is not a JSON object")
        required = {"source", "declared_locus", "requested_authority", "evidence_ids"}
        if set(value) - required - {"metadata"} or not required.issubset(value):
            raise VariationDependencyError("pinned model candidate response has an unexpected field set")
        source = value["source"]
        evidence = value["evidence_ids"]
        if not isinstance(source, str) or not isinstance(evidence, list) or not all(isinstance(item, str) for item in evidence):
            raise VariationDependencyError("pinned model candidate JSON has invalid source/evidence types")
        source_bytes = source.encode("utf-8")
        return CandidateProposal(
            source=source_bytes,
            declared_locus=value["declared_locus"],
            requested_authority=value["requested_authority"],
            evidence_ids=tuple(evidence),
            mutation_digest=digest_bytes(source_bytes),
            metadata=dict(value.get("metadata") or {}),
        )

    def propose(self, context: CandidateContext) -> CandidateProposal:
        prompt = self._prompt(context)
        try:
            encoded = self.tokenizer(prompt, return_tensors="pt")
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
]
