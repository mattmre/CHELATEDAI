"""Unit tests for selective_trigger_router (W22-15)."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any, List

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chelated_pooled_agent_runtime import (  # noqa: E402
    MatterSourceDoc,
    RecordedReplayWorkerBackend,
    SubtaskDraft,
    build_24_matter_templates,
    run_pooled_agent_experiment,
)
from chelated_raw_text_exocortex import DEFAULT_CHANNEL_AUTHORITY, RawDocumentEnvelope  # noqa: E402
from selective_trigger_router import (  # noqa: E402
    SelectiveReopenRouter,
    SelectiveTriggerReplayWorkerBackend,
    _route_extract,
    cue_match,
    doc_trigger_reason,
    route_suite_a_case,
    rule_reader,
)


def _channel_with_auth(pred: Any) -> str:
    for ch, a in sorted(DEFAULT_CHANNEL_AUTHORITY.items()):
        if pred(int(a)):
            return ch
    raise unittest.SkipTest("no channel with required authority")


def _env(doc_id: str, text: str, channel: str) -> RawDocumentEnvelope:
    return RawDocumentEnvelope(
        doc_id=doc_id, text=text, published_ts=1, ingested_ts=1, source_channel=channel, domain="test"
    )


class TestDocTrigger(unittest.TestCase):
    def test_cue_match_case_insensitive(self) -> None:
        self.assertEqual(cue_match("This filing SUPERSEDES the earlier one."), "supersed")
        self.assertEqual(cue_match("Party B lodged a counter-filing."), "counter-filing")
        self.assertIsNone(cue_match("Routine status update, nothing changed."))
        self.assertIsNone(cue_match(""))

    def test_authority_checked_before_cue(self) -> None:
        self.assertEqual(doc_trigger_reason("this refutes the rumor", 3), "authority>=3")
        self.assertEqual(doc_trigger_reason("this refutes the rumor", 1), "cue:refut")
        self.assertIsNone(doc_trigger_reason("plain echo of the news", 2))


class TestRouteSuiteA(unittest.TestCase):
    def test_order_preserved_and_partition(self) -> None:
        low = _channel_with_auth(lambda a: a < 3)
        high = _channel_with_auth(lambda a: a >= 3)
        docs = [
            _env("d0", "echo blog repeating the story", low),
            _env("d1", "official record", high),
            _env("d2", "a revision was published", low),
            _env("d3", "another echo", low),
        ]
        dec = route_suite_a_case(docs)
        self.assertEqual(dec.triggered_ids, ("d1", "d2"))
        self.assertEqual(dec.archived_ids, ("d0", "d3"))
        self.assertEqual(dec.n_docs, 4)
        self.assertTrue(dec.thread_triggered)
        self.assertEqual(dec.reasons["d1"], "authority>=3")
        self.assertEqual(dec.reasons["d2"], "cue:revision")
        self.assertGreaterEqual(dec.route_ms, 0.0)

    def test_unknown_channel_defaults_to_authority_one(self) -> None:
        dec = route_suite_a_case([_env("x", "plain text", "no_such_channel_xyz")])
        self.assertFalse(dec.thread_triggered)
        dec2 = route_suite_a_case([_env("x", "plain text", "no_such_channel_xyz")], channel_authority={"no_such_channel_xyz": 4})
        self.assertTrue(dec2.thread_triggered)


def _fake_exo(events: dict, auth: dict) -> Any:
    return SimpleNamespace(events=events, store=SimpleNamespace(eff_auth=auth))


def _ev(entity: str, slot: str, value: str, vt: int, tx: int) -> Any:
    return SimpleNamespace(tenant="t", entity=entity, slot=slot, claim_value=value, valid_time=vt, tx_time=tx)


class TestRuleReader(unittest.TestCase):
    def test_contested_abstains(self) -> None:
        self.assertEqual(rule_reader({"status": "CONTESTED_ABSTAIN", "selected_ids": ["a"]}, None), "ABSTAIN")

    def test_empty_selection(self) -> None:
        self.assertEqual(rule_reader({"status": "ADMIT_OK", "selected_ids": []}, None), "")

    def test_in_slot_max_authority_then_time(self) -> None:
        events = {
            "a": _ev("E", "s", "old", 1, 1),
            "b": _ev("E", "s", "new", 2, 2),
            "c": _ev("E", "s", "official", 1, 1),
            "z": _ev("OTHER", "s", "offslot", 9, 9),
        }
        exo = _fake_exo(events, {"a": 2, "b": 2, "c": 4, "z": 5})
        sel = {"status": "ADMIT_OK", "selected_ids": ["a", "b", "c", "z"], "inferred_slot": {"tenant": "t", "entity": "E", "slot": "s"}}
        self.assertEqual(rule_reader(sel, exo), "official")
        exo2 = _fake_exo(events, {"a": 2, "b": 2, "c": 1, "z": 5})
        self.assertEqual(rule_reader(sel, exo2), "new")

    def test_fallback_to_all_selected(self) -> None:
        events = {"a": _ev("E", "s", "v1", 1, 1), "b": _ev("F", "s", "v2", 1, 1)}
        exo = _fake_exo(events, {"a": 1, "b": 3})
        sel = {"status": "ADMIT_OK", "selected_ids": ["a", "b"], "inferred_slot": {"tenant": "t", "entity": "Q", "slot": "q"}}
        self.assertEqual(rule_reader(sel, exo), "v2")


def _doc(doc_id: str, text: str, tier: int) -> MatterSourceDoc:
    return MatterSourceDoc(
        doc_id=doc_id,
        matter_id="M01",
        field_name="f",
        field_value="UNREAD_BY_ROUTER",
        source_kind="memo",
        authority_tier=tier,
        tx_time=1,
        valid_from=1,
        text=text,
    )


class _CountingInner:
    def __init__(self) -> None:
        self.calls: List[bool] = []

    def __call__(self, *, template: Any, field_name: str, candidate_docs: Any, is_reopen: bool) -> SubtaskDraft:
        self.calls.append(is_reopen)
        cit = candidate_docs[0].doc_id if candidate_docs else None
        return SubtaskDraft(
            field_name=field_name, field_value=f"v{len(self.calls)}", cited_source_id=cit, exec_ms=5.0, timed_out=False, live_call=True
        )


class TestReopenRouter(unittest.TestCase):
    def setUp(self) -> None:
        self.tmpl = build_24_matter_templates()[0]
        self.d_a = _doc("A", "plain memo", 1)
        self.d_b = _doc("B", "another plain memo", 1)

    def _call(self, router: Any, inner: Any, docs: Any, reopen: bool) -> SubtaskDraft:
        return _route_extract(router, inner, template=self.tmpl, field_name="f", candidate_docs=docs, is_reopen=reopen)

    def test_reuse_when_unamended(self) -> None:
        r, inner = SelectiveReopenRouter(), _CountingInner()
        first = self._call(r, inner, [self.d_a, self.d_b], False)
        again = self._call(r, inner, [self.d_a, self.d_b], True)
        self.assertEqual(inner.calls, [False])
        self.assertEqual(again.field_value, first.field_value)
        self.assertEqual(again.cited_source_id, first.cited_source_id)
        self.assertFalse(again.live_call)
        self.assertEqual(r.stats.reopen_reused, 1)

    def test_reextract_when_citation_withdrawn(self) -> None:
        r, inner = SelectiveReopenRouter(), _CountingInner()
        self._call(r, inner, [self.d_a, self.d_b], False)
        self._call(r, inner, [self.d_b], True)
        self.assertEqual(inner.calls, [False, True])
        self.assertEqual(r.stats.reasons, {"cited_source_not_admissible": 1})

    def test_reextract_on_new_cue_or_authority_doc(self) -> None:
        r, inner = SelectiveReopenRouter(), _CountingInner()
        self._call(r, inner, [self.d_a], False)
        self._call(r, inner, [self.d_a, _doc("C", "this amends the prior memo", 1)], True)
        self._call(r, inner, [self.d_a], False)
        self._call(r, inner, [self.d_a, _doc("D", "plain", 4)], True)
        self._call(r, inner, [self.d_a], False)
        self._call(r, inner, [self.d_a, _doc("E", "plain", 1)], True)
        self.assertEqual(
            r.stats.reasons,
            {"new_amendment_cue_candidate": 1, "new_high_authority_candidate": 1, "reused_no_amendment": 1},
        )

    def test_no_cache_forces_reextract(self) -> None:
        r, inner = SelectiveReopenRouter(), _CountingInner()
        self._call(r, inner, [self.d_a], True)
        self.assertEqual(inner.calls, [True])
        self.assertEqual(r.stats.reasons, {"no_cached_draft": 1})

    def test_initial_pass_overwrites_cache_per_job(self) -> None:
        r, inner = SelectiveReopenRouter(), _CountingInner()
        self._call(r, inner, [self.d_a], False)
        second_initial = self._call(r, inner, [self.d_a], False)
        reused = self._call(r, inner, [self.d_a], True)
        self.assertEqual(reused.field_value, second_initial.field_value)
        self.assertEqual(r.stats.initial_calls, 2)


class TestReplayBackendEndToEnd(unittest.TestCase):
    def test_small_grid_sham_isolation(self) -> None:
        grid = dict(
            mode="stage1-controlled-replay",
            pool_sizes=(4,),
            schedules=("S1_steady",),
            branches=("A_derivative_only",),
            event_regimes=("no_change", "sham_subtask_reopen_interrupt", "withdrawal_only"),
        )
        base = run_pooled_agent_experiment(worker_backend=RecordedReplayWorkerBackend(), **grid)
        sel_backend = SelectiveTriggerReplayWorkerBackend()
        sel = run_pooled_agent_experiment(worker_backend=sel_backend, **grid)
        self.assertEqual(sel["worker_backend"], "stage1-controlled-replay+selective-trigger")
        self.assertTrue(sel["gates"]["sham_reopen_control_valid"])
        self.assertEqual(sel["zeromu_ek5_stale_commits_total"], 0)
        st = sel_backend.router.stats
        self.assertGreater(st.reopen_calls, 0)
        self.assertEqual(st.reopen_calls, st.reopen_reused + st.reopen_reextracted)
        self.assertGreater(st.reopen_reused, 0)
        self.assertEqual(
            base["sham_reopen_isolation"]["sham_on_time_cited_correct_jobs"],
            sel["sham_reopen_isolation"]["sham_on_time_cited_correct_jobs"],
        )


if __name__ == "__main__":
    unittest.main()
