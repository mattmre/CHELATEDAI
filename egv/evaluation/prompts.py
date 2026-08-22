"""Immutable EGV Evaluation prompt templates."""

from __future__ import annotations

from dataclasses import dataclass
from string import Formatter
from typing import Any, Dict, Mapping
from types import MappingProxyType

from ..canonical import canonical_json, digest_for


PROMPT_IDS = (
    "egv-system-v1",
    "egv-candidate-v1",
    "egv-evidence-success-v1",
    "egv-evidence-failure-v1",
    "egv-correction-v1",
)

_PROMPT_TEXT = MappingProxyType({
    "egv-system-v1": (
        "You are an EGV candidate generator. Return only a bounded Python candidate. "
        "Do not claim authority, inspect hidden resources, or include free-form reasoning."
    ),
    "egv-candidate-v1": (
        "Task {task_id}\nFamily {family_id}\nAttempt {attempt_index}\n"
        "Repair only the declared public locus: {public_locus}."
    ),
    "egv-evidence-success-v1": (
        "Use only these verified evidence IDs: {evidence_ids}. Cite each used ID in the bounded mutation record."
    ),
    "egv-evidence-failure-v1": (
        "Known bounded failure families: {failure_families}. Do not treat a rejected result as a success."
    ),
    "egv-correction-v1": (
        "A signed correction invalidated premise {corrected_event_id}. Recompute dependencies before proposing a candidate."
    ),
})


@dataclass(frozen=True)
class PromptTemplate:
    template_id: str
    text: str
    template_digest: str


class PromptRegistry:
    """Read-only registry for the five protocol prompts."""

    def __init__(self) -> None:
        self._templates = tuple(
            PromptTemplate(template_id, _PROMPT_TEXT[template_id], digest_for(_PROMPT_TEXT[template_id]))
            for template_id in PROMPT_IDS
        )
        self.validate()

    def validate(self) -> None:
        if tuple(template.template_id for template in self._templates) != PROMPT_IDS:
            raise ValueError("prompt IDs differ from the frozen five-template contract")
        for template in self._templates:
            if digest_for(template.text) != template.template_digest:
                raise ValueError("prompt template digest mismatch")

    def get(self, template_id: str) -> PromptTemplate:
        for template in self._templates:
            if template.template_id == template_id:
                return template
        raise KeyError("unknown immutable EGV prompt template")

    def manifest(self) -> Dict[str, Any]:
        return {
            "schema_version": "egv-prompt-manifest-v1",
            "template_ids": list(PROMPT_IDS),
            "templates": [
                {"template_id": template.template_id, "template_digest": template.template_digest}
                for template in self._templates
            ],
        }

    def manifest_digest(self) -> str:
        return digest_for(self.manifest())

    def render(self, template_id: str, values: Mapping[str, Any]) -> str:
        template = self.get(template_id)
        fields = {field_name for _, field_name, _, _ in Formatter().parse(template.text) if field_name}
        supplied = set(values)
        if fields != supplied:
            raise ValueError("prompt rendering context must exactly match the immutable template fields")
        rendered = template.text.format(**{key: canonical_json(value) if isinstance(value, (dict, list)) else value for key, value in values.items()})
        if not rendered or "<SECRET>" in rendered or "PRIVATE_KEY" in rendered:
            raise ValueError("prompt rendering produced prohibited content")
        return rendered


__all__ = ["PROMPT_IDS", "PromptRegistry", "PromptTemplate"]
