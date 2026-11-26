"""W22-15 selective-trigger routing for the 0.5B fast claim extractor.

Two deterministic, label-free routers, preregistered in
`artifacts/pv7_w23b_w22_15/PREREGISTRATION.md` §3:

1. Suite A doc-level write router (`route_suite_a_case`): only documents that are either
   high-authority (`DEFAULT_CHANNEL_AUTHORITY >= 3`) or carry an amendment/conflict cue are sent to
   the extractor and written to the claim store; the rest go to a raw archive the read path does
   not query.
2. #62 / D13 reopen router (`SelectiveReopenRouter` + two backends): a reopened subtask is
   re-extracted only when its thread was amended (the cached citation left the candidate set, or a
   new cue-bearing / high-authority candidate appeared). Otherwise the job's own cached draft is
   reused with no model call.

Neither router reads gold answers, regime/case labels, stale-value lists, or the #62 structured
`MatterSourceDoc.field_value`.
"""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, List, Mapping, Optional, Sequence, Tuple

from chelated_pooled_agent_runtime import (
    LiveVLLMWorkerBackend,
    MatterSourceDoc,
    MatterTemplate,
    RecordedReplayWorkerBackend,
    SubtaskDraft,
)
from chelated_raw_text_exocortex import DEFAULT_CHANNEL_AUTHORITY, RawDocumentEnvelope

# Frozen in PREREGISTRATION.md §3.1 (derived from EXTRACTION_SYSTEM_PROMPT_V2 stance definitions).
AMENDMENT_CONFLICT_STEMS: Tuple[str, ...] = (
    "supersed",
    "replac",
    "revision",
    "revise",
    "restat",
    "updated guidance",
    "debunk",
    "refut",
    "denial",
    "denied",
    "deny",
    "counter-filing",
    "counter filing",
    "disput",
    "audit",
    "bulletin",
    "retract",
    "rescind",
    "withdraw",
    "amend",
    "correction",
)
AUTHORITY_TRIGGER_MIN: int = 3

_CUE_RE = re.compile("|".join(re.escape(s) for s in AMENDMENT_CONFLICT_STEMS))


def cue_match(text: str) -> Optional[str]:
    """Return the first amendment/conflict stem found in `text` (case-insensitive), else None."""
    m = _CUE_RE.search((text or "").lower())
    return m.group(0) if m else None


def doc_trigger_reason(text: str, authority: int) -> Optional[str]:
    """Preregistered doc trigger: `authority>=3` or a cue stem. Returns a reason string or None."""
    if int(authority) >= AUTHORITY_TRIGGER_MIN:
        return f"authority>={AUTHORITY_TRIGGER_MIN}"
    stem = cue_match(text)
    if stem is not None:
        return f"cue:{stem}"
    return None


@dataclass(frozen=True)
class SuiteARouteDecision:
    """Routing result for one Suite A thread (case)."""

    triggered_ids: Tuple[str, ...]
    archived_ids: Tuple[str, ...]
    reasons: Mapping[str, str]
    route_ms: float

    @property
    def thread_triggered(self) -> bool:
        return len(self.triggered_ids) > 0

    @property
    def n_docs(self) -> int:
        return len(self.triggered_ids) + len(self.archived_ids)


def route_suite_a_case(
    docs: Sequence[RawDocumentEnvelope],
    *,
    channel_authority: Optional[Mapping[str, int]] = None,
) -> SuiteARouteDecision:
    """Split one thread's docs into extract+index (triggered) vs raw archive, preserving input order."""
    t0 = time.perf_counter()
    auth_map = dict(DEFAULT_CHANNEL_AUTHORITY)
    if channel_authority:
        auth_map.update(channel_authority)
    trig: List[str] = []
    arch: List[str] = []
    reasons: Dict[str, str] = {}
    for d in docs:
        reason = doc_trigger_reason(d.text, auth_map.get(d.source_channel, 1))
        if reason is None:
            arch.append(d.doc_id)
        else:
            trig.append(d.doc_id)
            reasons[d.doc_id] = reason
    route_ms = (time.perf_counter() - t0) * 1000.0
    return SuiteARouteDecision(
        triggered_ids=tuple(trig),
        archived_ids=tuple(arch),
        reasons=reasons,
        route_ms=route_ms,
    )


def rule_reader(sel: Mapping[str, Any], exo: Any) -> str:
    """Preregistered deterministic reader (PREREGISTRATION.md §4.3) over an exocortex selection."""
    if str(sel.get("status")) == "CONTESTED_ABSTAIN":
        return "ABSTAIN"
    selected = [str(x) for x in (sel.get("selected_ids") or [])]
    if not selected:
        return ""
    slot = sel.get("inferred_slot") or {}
    key = (slot.get("tenant"), slot.get("entity"), slot.get("slot"))
    in_slot = [d for d in selected if (exo.events[d].tenant, exo.events[d].entity, exo.events[d].slot) == key]
    pool = in_slot or selected

    def _rank(did: str) -> Tuple[int, int, int]:
        ev = exo.events[did]
        return (int(exo.store.eff_auth.get(did, 0)), int(ev.valid_time), int(ev.tx_time))

    best = max(pool, key=_rank)
    return str(exo.events[best].claim_value)


# ---------------------------------------------------------------------------
# #62 / D13 selective reopen routing
# ---------------------------------------------------------------------------


@dataclass
class ReopenRouterStats:
    initial_calls: int = 0
    reopen_calls: int = 0
    reopen_reextracted: int = 0
    reopen_reused: int = 0
    reasons: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "initial_calls": self.initial_calls,
            "reopen_calls": self.reopen_calls,
            "reopen_reextracted": self.reopen_reextracted,
            "reopen_reused": self.reopen_reused,
            "reasons": dict(sorted(self.reasons.items())),
        }


@dataclass(frozen=True)
class _CachedDraft:
    draft: SubtaskDraft
    candidate_ids: FrozenSet[str]


class SelectiveReopenRouter:
    """Per-job draft cache + amendment trigger for reopened subtasks (PREREGISTRATION.md §3.2).

    Cache scope invariant: `PooledAgentCoordinator.run_cell` executes jobs sequentially and every job's
    initial (non-reopen) pass for `(matter, field)` overwrites the entry before that job's reopen.
    """

    def __init__(self) -> None:
        self._cache: Dict[Tuple[str, str], _CachedDraft] = {}
        self._lock = threading.Lock()
        self.stats = ReopenRouterStats()

    def record_initial(self, matter_id: str, field_name: str, draft: SubtaskDraft, candidate_docs: Sequence[MatterSourceDoc]) -> None:
        with self._lock:
            self._cache[(matter_id, field_name)] = _CachedDraft(
                draft=draft, candidate_ids=frozenset(d.doc_id for d in candidate_docs)
            )
            self.stats.initial_calls += 1

    def reopen_trigger(self, matter_id: str, field_name: str, candidate_docs: Sequence[MatterSourceDoc]) -> Optional[str]:
        """Return a re-extraction reason, or None when the cached draft may be reused."""
        with self._lock:
            cached = self._cache.get((matter_id, field_name))
        if cached is None:
            return "no_cached_draft"
        cand_ids = {d.doc_id for d in candidate_docs}
        cit = cached.draft.cited_source_id
        if cit is None or cit not in cand_ids:
            return "cited_source_not_admissible"
        for d in candidate_docs:
            if d.doc_id in cached.candidate_ids:
                continue
            if int(d.authority_tier) >= AUTHORITY_TRIGGER_MIN:
                return "new_high_authority_candidate"
            if cue_match(d.text) is not None:
                return "new_amendment_cue_candidate"
        return None

    def note_reopen(self, reason: Optional[str]) -> None:
        with self._lock:
            self.stats.reopen_calls += 1
            key = reason or "reused_no_amendment"
            self.stats.reasons[key] = self.stats.reasons.get(key, 0) + 1
            if reason is None:
                self.stats.reopen_reused += 1
            else:
                self.stats.reopen_reextracted += 1

    def cached(self, matter_id: str, field_name: str) -> SubtaskDraft:
        with self._lock:
            return self._cache[(matter_id, field_name)].draft


def _route_extract(
    router: SelectiveReopenRouter,
    inner_extract: Any,
    *,
    template: MatterTemplate,
    field_name: str,
    candidate_docs: Sequence[MatterSourceDoc],
    is_reopen: bool,
) -> SubtaskDraft:
    if not is_reopen:
        draft = inner_extract(template=template, field_name=field_name, candidate_docs=candidate_docs, is_reopen=False)
        router.record_initial(template.matter_id, field_name, draft, candidate_docs)
        return draft
    t0 = time.perf_counter()
    reason = router.reopen_trigger(template.matter_id, field_name, candidate_docs)
    router.note_reopen(reason)
    if reason is not None:
        return inner_extract(template=template, field_name=field_name, candidate_docs=candidate_docs, is_reopen=True)
    prev = router.cached(template.matter_id, field_name)
    route_ms = (time.perf_counter() - t0) * 1000.0
    return SubtaskDraft(
        field_name=field_name,
        field_value=prev.field_value,
        cited_source_id=prev.cited_source_id,
        exec_ms=route_ms,
        timed_out=False,
        live_call=False,
    )


class SelectiveTriggerReplayWorkerBackend(RecordedReplayWorkerBackend):
    """Stage 1 CPU recorded replay with selective reopen routing (no model calls)."""

    mode_name = "stage1-controlled-replay+selective-trigger"

    def __init__(self) -> None:
        super().__init__()
        self.router = SelectiveReopenRouter()

    def extract_field(
        self,
        *,
        template: MatterTemplate,
        field_name: str,
        candidate_docs: Sequence[MatterSourceDoc],
        is_reopen: bool = False,
    ) -> SubtaskDraft:
        return _route_extract(
            self.router,
            super().extract_field,
            template=template,
            field_name=field_name,
            candidate_docs=candidate_docs,
            is_reopen=is_reopen,
        )


class SelectiveTriggerLiveWorkerBackend(LiveVLLMWorkerBackend):
    """Live OpenAI-compatible worker; reopened subtasks only call the model when the thread was amended."""

    mode_name = "run-live-workers+selective-trigger"

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.router = SelectiveReopenRouter()

    def extract_field(
        self,
        *,
        template: MatterTemplate,
        field_name: str,
        candidate_docs: Sequence[MatterSourceDoc],
        is_reopen: bool = False,
    ) -> SubtaskDraft:
        return _route_extract(
            self.router,
            super().extract_field,
            template=template,
            field_name=field_name,
            candidate_docs=candidate_docs,
            is_reopen=is_reopen,
        )


__all__ = [
    "AMENDMENT_CONFLICT_STEMS",
    "AUTHORITY_TRIGGER_MIN",
    "ReopenRouterStats",
    "SelectiveReopenRouter",
    "SelectiveTriggerLiveWorkerBackend",
    "SelectiveTriggerReplayWorkerBackend",
    "SuiteARouteDecision",
    "cue_match",
    "doc_trigger_reason",
    "route_suite_a_case",
    "rule_reader",
]
