"""Offline Wave-11 D7 tests. No Docker, SSH, GPU, Neo4j, or vLLM."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

import pytest
from pydantic import BaseModel

import graphiti_core.driver.neo4j_driver as neo4j_driver_mod
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.prompts.models import Message
from graphiti_core.search.search_filters import ComparisonOperator

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import chelated_graphiti_arena_v1 as arena  # noqa: E402
import scripts.run_wave11_d7_live_zep_graphiti_arena as runner  # noqa: E402

HOLM_GRAPHITI = "Chelated_v4_Stratified_WFS_minus_Live_Graphiti_Zep_v0_31_2"
HOLM_MEM0 = "Chelated_v4_Stratified_WFS_minus_Live_Mem0_v2_2_1_InferTrue"
HOLM_KEYS = {HOLM_GRAPHITI, HOLM_MEM0}


class _OkModel(BaseModel):
    ok: bool


class _Msg:
    def __init__(self, content: str) -> None:
        self.content = content


class _Choice:
    def __init__(self, content: str, finish_reason: Optional[str] = None) -> None:
        self.message = _Msg(content)
        self.finish_reason = finish_reason


class _Usage:
    def __init__(self, prompt: Optional[int], completion: Optional[int]) -> None:
        self.prompt_tokens = prompt
        self.completion_tokens = completion


class _Resp:
    def __init__(self, content: str, usage: Any, finish_reason: Optional[str] = None) -> None:
        self.choices = [_Choice(content, finish_reason)]
        self.usage = usage


class _Completions:
    def __init__(self) -> None:
        self.calls: List[Dict[str, Any]] = []
        self.script: List[Any] = []

    async def create(self, *args: Any, **kwargs: Any) -> Any:
        del args
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, BaseException):
            raise item
        return item


class _FakeAsyncOpenAI:
    def __init__(self) -> None:
        self.chat = type("Chat", (), {})()
        self.chat.completions = _Completions()


class _HTTP400(Exception):
    def __init__(self) -> None:
        super().__init__("schema rejected")
        self.status_code = 400


class _Doc:
    def __init__(self, doc_id: str, ingested_ts: int, published_ts: int = 0) -> None:
        self.doc_id = doc_id
        self.ingested_ts = ingested_ts
        self.published_ts = published_ts


class _Edge:
    def __init__(self, group_id: str, fact: str, invalid_at: Any, valid_at: Any) -> None:
        self.group_id = group_id
        self.fact = fact
        self.invalid_at = invalid_at
        self.valid_at = valid_at


class _SearchResult:
    def __init__(self, edges: List[Any]) -> None:
        self.edges = edges


class _CypherResult:
    def __init__(self, row: Dict[str, int]) -> None:
        self.records = [row]


def _client() -> tuple[_FakeAsyncOpenAI, arena.MeteredVllmOpenAIGenericClient]:
    fake = _FakeAsyncOpenAI()
    metered = arena.MeteredVllmOpenAIGenericClient(client=fake)
    return fake, metered


def _generate(metered: arena.MeteredVllmOpenAIGenericClient) -> Dict[str, Any]:
    return asyncio.run(
        metered._generate_response([Message(role="user", content="ping")], _OkModel, max_tokens=32)
    )


def _arm_row(
    item_id: str,
    *,
    chelated: int = 1,
    graphiti: int = 0,
    mem0: int = 0,
    abstain: bool = False,
    stale: Optional[List[str]] = None,
    fail_arm: Optional[str] = None,
) -> Dict[str, Any]:
    bits = {
        arena.ARM_CHELATED: chelated,
        arena.ARM_GRAPHITI: graphiti,
        arena.ARM_MEM0: mem0,
    }
    arms: Dict[str, Any] = {}
    for arm, bit in bits.items():
        slot = runner.new_arm_slot()
        slot["correct_settled"] = bool(bit)
        slot["correct_delta0s"] = bool(bit)
        slot["write_ack_ms"] = 3.0
        slot["consolidation_bg_ms"] = 4.0
        slot["write_prompt_tokens"] = 10
        slot["write_completion_tokens"] = 2
        slot["read_prompt_tokens"] = 6
        slot["read_completion_tokens"] = 1
        if fail_arm == arm:
            slot["arm_failure"] = True
            slot["correct_settled"] = False
            slot["graph_audit_settled"] = {"status": "ARM_FAILURE", "reason": "episode_only"}
        if stale:
            slot["stale_context_settled"] = arm != arena.ARM_CHELATED
            slot["stale_answer_settled"] = False
        arms[arm] = slot
    return {
        "item_id": item_id,
        "question": "q",
        "gold": "ABSTAIN" if abstain else "gold",
        "abstention_required": abstain,
        "stale_values": list(stale or []),
        "regime": "test",
        "input_sha256": "abc",
        "judge_complete": True,
        "arms": arms,
    }


def test_meter_forces_temperature_and_splits_phases() -> None:
    fake, metered = _client()
    fake.chat.completions.script.append(_Resp('{"ok": true}', _Usage(5, 2)))
    parsed = _generate(metered)
    assert parsed == {"ok": True}
    call = fake.chat.completions.calls[0]
    assert call["temperature"] == 0.0
    assert call["extra_body"]["chat_template_kwargs"]["enable_thinking"] is False
    assert call["response_format"]["type"] == "json_schema"
    assert "schema" in call["response_format"]["json_schema"]
    assert metered.write_prompt_tokens == 5
    assert metered.write_completion_tokens == 2
    assert metered.read_prompt_tokens == 0
    assert metered.llm_call_count == 1
    assert metered.json_repair_count == 0
    assert metered.schema_fallback_count == 0
    metered.reset_counters(phase="read")
    assert metered.llm_call_count == 0
    assert metered.write_prompt_tokens == 5
    assert metered.read_prompt_tokens == 0
    fake.chat.completions.script.append(_Resp('{"ok": true}', _Usage(9, 4)))
    _generate(metered)
    assert metered.read_prompt_tokens == 9
    assert metered.read_completion_tokens == 4
    assert metered.write_completion_tokens == 2
    assert metered.llm_call_count == 1


def test_missing_usage_does_not_increment() -> None:
    fake, metered = _client()
    fake.chat.completions.script.append(_Resp('{"ok": true}', None))
    with pytest.raises(RuntimeError, match="response.usage"):
        _generate(metered)
    assert metered.llm_call_count == 0
    assert metered.write_prompt_tokens == 0
    assert metered.write_completion_tokens == 0
    bare = _Usage(1, None)
    fake.chat.completions.script.append(_Resp('{"ok": true}', bare))
    with pytest.raises(RuntimeError, match="prompt_tokens or completion_tokens"):
        _generate(metered)
    assert metered.llm_call_count == 0


def test_http_400_falls_back_to_json_object_once() -> None:
    fake, metered = _client()
    fake.chat.completions.script.extend(
        [_HTTP400(), _Resp('{"ok": true}', _Usage(8, 3))]
    )
    parsed = _generate(metered)
    assert parsed == {"ok": True}
    assert metered.schema_fallback_count == 1
    assert fake.chat.completions.calls[0]["response_format"]["type"] == "json_schema"
    assert fake.chat.completions.calls[1]["response_format"] == {"type": "json_object"}
    assert metered.structured_output_mode == "json_object"
    fake.chat.completions.script.append(_Resp('{"ok": true}', _Usage(1, 1)))
    _generate(metered)
    assert metered.schema_fallback_count == 1
    assert fake.chat.completions.calls[-1]["response_format"]["type"] == "json_object"


def test_fenced_and_think_json_are_counted_and_not_recalled() -> None:
    fake, metered = _client()
    fenced = '```json\n{"ok": true}\n```'
    fake.chat.completions.script.append(_Resp(fenced, _Usage(2, 2)))
    assert _generate(metered) == {"ok": True}
    assert metered.json_repair_count == 1
    assert metered.schema_fallback_count == 1
    assert len(fake.chat.completions.calls) == 1
    assert metered.structured_output_mode == "json_object"

    fake2, metered2 = _client()
    think = '<think>secret</think>\n```json\n{"ok": true}\n```'
    fake2.chat.completions.script.append(_Resp(think, _Usage(2, 2)))
    assert _generate(metered2) == {"ok": True}
    assert metered2.json_repair_count == 2
    assert metered2.schema_fallback_count == 1
    assert len(fake2.chat.completions.calls) == 1


def test_reset_keeps_schema_counts_and_refuses_other_base_urls() -> None:
    fake, metered = _client()
    fake.chat.completions.script.append(_Resp('```json\n{"ok": true}\n```', _Usage(3, 1)))
    _generate(metered)
    assert metered.schema_fallback_count == 1
    metered.reset_counters(phase="write")
    assert metered.schema_fallback_count == 1
    assert metered.json_repair_count == 1
    assert metered.write_prompt_tokens == 0
    assert metered.llm_call_count == 0
    with pytest.raises(ValueError, match="phase"):
        metered.reset_counters(phase="judge")
    with pytest.raises(ValueError, match="8891"):
        arena.MeteredVllmOpenAIGenericClient(
            config=LLMConfig(
                api_key="x",
                model="nemotron-lightning",
                base_url="http://127.0.0.1:1/v1",
                temperature=0.0,
            ),
            client=_FakeAsyncOpenAI(),
        )


def test_embedder_shape_and_reranker_order() -> None:
    def encode(texts: List[str]) -> Any:
        import numpy as np

        return np.ones((len(texts), arena.EMBEDDING_DIM), dtype="float64")

    embedder = arena.LocalGpuBgeM3Embedder(encoder=encode)
    one = asyncio.run(embedder.create("solo"))
    listed = asyncio.run(embedder.create(["solo"]))
    assert len(one) == 1024 and len(listed) == 1024
    import numpy as np

    assert abs(float(np.linalg.norm(one)) - 1.0) < 1e-9
    with pytest.raises(TypeError):
        asyncio.run(embedder.create(["a", "b"]))
    batch = asyncio.run(embedder.create_batch(["a", "b"]))
    assert len(batch) == 2 and len(batch[0]) == 1024

    def short(_texts: List[str]) -> Any:
        return np.ones((1, 8), dtype="float64")

    with pytest.raises(RuntimeError, match="1024"):
        asyncio.run(arena.LocalGpuBgeM3Embedder(encoder=short).create("x"))

    def zeros(_texts: List[str]) -> Any:
        return np.zeros((1, 1024), dtype="float64")

    with pytest.raises(RuntimeError, match="zero or non-finite"):
        asyncio.run(arena.LocalGpuBgeM3Embedder(encoder=zeros).create("x"))

    seen: List[Any] = []

    def predict(pairs: List[List[str]]) -> List[float]:
        seen.append(pairs)
        return [0.1, 0.9]

    reranker = arena.LocalGpuBgeReranker(predict_fn=predict)
    assert asyncio.run(reranker.rank("q", [])) == []
    ranked = asyncio.run(reranker.rank("q", ["first", "second"]))
    assert ranked[0][0] == "second" and ranked[0][1] == 0.9
    assert ranked[1][0] == "first"
    assert seen == [[["q", "first"], ["q", "second"]]]


def test_single_database_pin_and_request_scope() -> None:
    client = type("Client", (), {})()
    client.sessions: List[Any] = []
    client.queries: List[Dict[str, Any]] = []

    def _session(database: Any = None) -> Any:
        client.sessions.append(database)
        return {"database": database}

    async def _execute(cypher: str, parameters_: Any = None, database_: Any = None, **kwargs: Any) -> Any:
        del kwargs
        client.queries.append({"cypher": cypher, "parameters_": parameters_, "database_": database_})
        return _CypherResult({"n": 1})

    client.session = _session
    client.execute_query = _execute

    class _Factory:
        @staticmethod
        def driver(*_args: Any, **_kwargs: Any) -> Any:
            return client

    original = neo4j_driver_mod.AsyncGraphDatabase
    neo4j_driver_mod.AsyncGraphDatabase = _Factory
    try:
        driver = arena.SingleDbCommunityNeo4jDriver(
            "bolt://127.0.0.1:17687",
            "neo4j",
            "pw",
            database="secret",
        )
    finally:
        neo4j_driver_mod.AsyncGraphDatabase = original
    assert driver._database == "neo4j"
    cloned = driver.clone("d7_case_00")
    assert cloned is not driver
    assert cloned._database == "neo4j"
    driver.session("d7_case_00")
    assert client.sessions == ["neo4j"]
    asyncio.run(
        driver.execute_query(
            "MATCH (n {group_id: $gid}) RETURN count(n) AS n_cnt",
            params={"gid": "d7_case_00"},
            database_="d7_case_00",
        )
    )
    assert client.queries[0]["database_"] == "neo4j"
    assert client.queries[0]["parameters_"] == {"gid": "d7_case_00"}

    class _Good:
        def __init__(self, database: str) -> None:
            self._database = database

        def clone(self, database: str) -> "_Good":
            return _Good(database)

    class _Clients:
        def __init__(self, driver_obj: Any) -> None:
            self.driver = driver_obj

        def model_copy(self, update: Optional[Dict[str, Any]] = None) -> "_Clients":
            return _Clients((update or {})["driver"])

    class _Graphiti:
        def __init__(self, driver_obj: Any) -> None:
            self.driver = driver_obj
            self.clients = _Clients(driver_obj)

        def _resolve_request_scope(self, group_id: Optional[str]) -> Any:
            return group_id, self.driver, self.clients

    good = _Graphiti(_Good("secret"))
    arena.pin_graphiti_request_scope(good)
    gid, pinned, clients = good._resolve_request_scope("d7_case_00")
    assert gid == "d7_case_00"
    assert pinned._database == "neo4j"
    assert clients.driver is pinned

    class _Ignore:
        def __init__(self) -> None:
            self._database = "secret"

        def clone(self, database: str) -> "_Ignore":
            del database
            cloned_driver = _Ignore()
            cloned_driver._database = "d7_case_00"
            return cloned_driver

    bad = _Graphiti(_Ignore())
    arena.pin_graphiti_request_scope(bad)
    with pytest.raises(RuntimeError, match="single-db pin failed"):
        bad._resolve_request_scope("d7_case_00")

    empty = _Graphiti(_Good("neo4j"))

    def _empty(group_id: Optional[str]) -> Any:
        del group_id
        return "", empty.driver, empty.clients

    empty._resolve_request_scope = _empty
    arena.pin_graphiti_request_scope(empty)
    with pytest.raises(RuntimeError, match="empty"):
        empty._resolve_request_scope(None)
    with pytest.raises(ValueError):
        arena.case_group_id(True)
    assert arena.case_group_id(0) == "d7_case_00"
    assert arena.case_group_id(0, prefix="d7b_ku") == "d7b_ku_00"
    with pytest.raises(ValueError):
        arena.require_group_id("neo4j")
    with pytest.raises(ValueError):
        arena.require_group_id("")


def test_active_edges_and_graph_audit() -> None:
    gid = "d7_case_03"
    live_at = datetime(2024, 1, 2, tzinfo=timezone.utc)
    edges = [
        _Edge(gid, "retracted claim", datetime(2024, 2, 1, tzinfo=timezone.utc), live_at),
        _Edge(gid, "live fact", None, live_at),
    ]

    class _Graph:
        def __init__(self) -> None:
            self.kwargs: Dict[str, Any] = {}

        async def search_(self, query: str, **kwargs: Any) -> _SearchResult:
            self.kwargs = {"query": query, **kwargs}
            return _SearchResult(edges)

    graph = _Graph()
    text = asyncio.run(arena.format_active_facts(graph, "what changed", gid, num_results=8))
    assert "retracted claim" not in text
    assert text.startswith("[fact | valid_at=2024-01-02T00:00:00+00:00] live fact")
    assert graph.kwargs["group_ids"] == [gid]
    assert graph.kwargs["config"].limit == 8
    operator = graph.kwargs["search_filter"].invalid_at[0][0].comparison_operator
    assert operator == ComparisonOperator.is_null

    class _Leak(_Graph):
        async def search_(self, query: str, **kwargs: Any) -> _SearchResult:
            del query, kwargs
            return _SearchResult([_Edge("other", "x", None, None)])

    with pytest.raises(RuntimeError, match="leaked"):
        asyncio.run(arena.format_active_facts(_Leak(), "q", gid))

    class _Bare:
        pass

    with pytest.raises(RuntimeError, match="search_"):
        asyncio.run(arena.format_active_facts(_Bare(), "q", gid))

    class _Empty(_Graph):
        async def search_(self, query: str, **kwargs: Any) -> _SearchResult:
            del query, kwargs
            return _SearchResult([])

    assert asyncio.run(arena.format_active_facts(_Empty(), "q", gid)) == "(No active facts.)"
    assert arena.NODE_COUNT_CYPHER == "MATCH (n {group_id: $gid}) RETURN count(n) AS n_cnt"
    assert arena.EDGE_COUNT_CYPHER == "MATCH ()-[r:RELATES_TO {group_id: $gid}]->() RETURN count(r) AS r_cnt"

    class _Audit:
        def __init__(self, n_cnt: int, r_cnt: int) -> None:
            self.n_cnt = n_cnt
            self.r_cnt = r_cnt
            self.calls: List[Any] = []

        async def execute_query(self, cypher: str, params: Optional[Dict[str, str]] = None) -> _CypherResult:
            self.calls.append((cypher, params))
            if "RELATES_TO" in cypher:
                return _CypherResult({"r_cnt": self.r_cnt})
            return _CypherResult({"n_cnt": self.n_cnt})

    episode = _Audit(4, 0)
    audit = asyncio.run(arena.verify_case_graph_populated(episode, gid))
    assert audit["status"] == "ARM_FAILURE"
    assert audit["reason"] == "episode_only"
    assert episode.calls[0][0] == arena.NODE_COUNT_CYPHER
    assert episode.calls[1][0] == arena.EDGE_COUNT_CYPHER
    assert episode.calls[0][1] == {"gid": gid}
    ok = asyncio.run(arena.verify_case_graph_populated(_Audit(2, 3), gid))
    assert ok["status"] == "OK" and ok["reason"] == "populated"
    observed = asyncio.run(arena.verify_case_graph_populated(_Audit(0, 0), gid, settled_gate=False))
    assert observed["status"] == "DELTA0_OBSERVED"
    assert observed["reason"] == "in_flight"

    class _Boom:
        async def execute_query(self, *_args: Any, **_kwargs: Any) -> Any:
            raise RuntimeError("cypher-down")

    with pytest.raises(RuntimeError, match="cypher-down"):
        asyncio.run(arena.verify_case_graph_populated(_Boom(), gid))

    class _Neo4jRecordLike(Mapping[str, int]):
        """Mimic neo4j 6.x Record where ``key in record`` checks values, not column names."""

        def __init__(self, key: str, value: int) -> None:
            self._data = {key: value}

        def keys(self) -> List[str]:  # type: ignore[override]
            return list(self._data.keys())

        def __getitem__(self, key: str) -> int:
            return self._data[key]

        def __iter__(self):
            return iter(self._data)

        def __len__(self) -> int:
            return len(self._data)

        def __contains__(self, item: object) -> bool:
            return item in self._data.values()

    assert arena._cypher_scalar(type("R", (), {"records": [_Neo4jRecordLike("n_cnt", 5)]})(), "n_cnt") == 5

    class _EmptyAttrs(BaseModel):
        pass

    fake_e, metered_e = _client()
    fake_e.chat.completions.script.append(_Resp("{}", _Usage(4, 1)))
    assert asyncio.run(
        metered_e._generate_response([Message(role="user", content="extract attrs")], _EmptyAttrs, max_tokens=16384)
    ) == {}
    assert fake_e.chat.completions.calls[0]["response_format"] == {"type": "json_object"}
    assert fake_e.chat.completions.calls[0]["max_tokens"] == 4096

    fake_r, metered_r = _client()
    metered_r.structured_output_mode = "json_object"
    fake_r.chat.completions.script.extend(
        [
            _Resp('{"$defs": {}, "properties": {"ok": {"type": "boolean"}}}', _Usage(6, 4)),
            _Resp('{"ok": true}', _Usage(7, 2)),
        ]
    )
    assert _generate(metered_r) == {"ok": True}
    assert fake_r.chat.completions.calls[1]["response_format"] == {"type": "json_object"}
    defs_retry = fake_r.chat.completions.calls[1]["messages"][-1]["content"]
    assert json.dumps(_OkModel.model_json_schema()) in defs_retry
    assert "never output '$defs'" in defs_retry
    assert metered_r.validation_retry_count == 1


def test_delta0_choreography_and_horizon_grades() -> None:
    searches: List[str] = []

    async def _write() -> str:
        await asyncio.sleep(0.05)
        return "retracted"

    async def _search() -> str:
        searches.append("search")
        return f"ctx-{len(searches)}"

    raced = asyncio.run(arena.run_delta0_then_settled(_write, _search))
    assert raced["task_done_at_delta0"] is False
    assert raced["task_done_after_settle"] is True
    assert raced["delta0"] == "ctx-1"
    assert raced["settled"] == "ctx-2"
    assert raced["synchronous_ek5_fence"] is False
    assert len(searches) == 2

    async def _bad_write() -> str:
        raise RuntimeError("write-broke")

    async def _bad_search() -> str:
        raise RuntimeError("search-broke")

    with pytest.raises(RuntimeError, match="write-broke"):
        asyncio.run(arena.run_delta0_then_settled(_bad_write, _bad_search))

    fenced = arena.synchronous_fence_horizons("same", write_ack_ms=1.0, consolidation_bg_ms=2.0, read_ms=3.0)
    assert fenced["delta0"] == fenced["settled"] == "same"
    assert fenced["task_done_at_delta0"] is True
    assert fenced["synchronous_ek5_fence"] is True

    failed = runner.new_arm_slot()
    failed["graph_audit_settled"] = {"status": "ARM_FAILURE"}
    clean = runner.new_arm_slot()
    row = {"arms": {arena.ARM_GRAPHITI: failed, arena.ARM_CHELATED: clean}}
    arena.apply_horizon_grades(
        row,
        {arena.ARM_GRAPHITI: {"correct": True}, arena.ARM_CHELATED: {"correct": True}},
        {arena.ARM_GRAPHITI: {"correct": True}, arena.ARM_CHELATED: {"correct": True}},
    )
    assert failed["correct_settled"] is False
    assert failed["correct_delta0s"] is True
    assert clean["correct_settled"] is True


def test_suite_statistics_keep_holm_to_two_contrasts() -> None:
    rows = [
        _arm_row(f"a{i:02d}", abstain=i < 2, stale=["oldvalue"] if i == 3 else None, fail_arm=arena.ARM_MEM0 if i == 4 else None)
        for i in range(8)
    ]
    report = arena.suite_a_horizon_inference(rows, "settled", n_bootstrap=40, seed=42)
    assert set(report["holm_bonferroni"]) == HOLM_KEYS
    comparison = report["comparisons"][HOLM_GRAPHITI]
    assert "lcb_975" in comparison
    assert "clopper_pearson_lcb95" in comparison
    assert "tango_score_lcb95" in comparison
    for name, holm in report["holm_bonferroni"].items():
        assert name in HOLM_KEYS
        assert holm["family_size"] == 2
        assert "holm_adjusted_p" in holm
    assert report["U_separate"]["in_holm_family"] is False
    assert report["U_separate"]["n"] == 2
    assert "U" not in report["holm_bonferroni"]
    assert report["arms"][arena.ARM_MEM0]["arm_failure_count"] == 1
    suite_b = arena.summarize_suite_b(rows[:3])
    assert suite_b["holm_bonferroni"] is None
    assert suite_b["unpooled"] is True
    assert suite_b["inferential_claim"] is False
    assert arena.N_BOOTSTRAP == 20000
    assert arena.MAX_COROUTINES == 6


def test_retraction_and_suite_b_selectors() -> None:
    for doc_id in ("x_official_supersede", "y_official_debunk", "z_v3_final", "w_d01_party_b"):
        chosen = arena.select_retraction_document([_Doc("archive_later", 99), _Doc(doc_id, 5)])
        assert chosen.doc_id == doc_id
        prior, retraction = arena.partition_retraction_documents([_Doc("archive_later", 99), _Doc(doc_id, 5)])
        assert retraction.doc_id == doc_id
        assert [doc.doc_id for doc in prior] == ["archive_later"]
    with pytest.raises(ValueError, match="no retraction"):
        arena.select_retraction_document([_Doc("archive_only", 3)])

    records = [
        {"question_id": "m", "question_type": "knowledge-update"},
        {"question_id": "z_abs", "question_type": "knowledge-update"},
        {"question_id": "c", "question_type": "single-session"},
        {"question_id": "a", "question_type": "knowledge-update"},
        {"question_id": "b", "question_type": "knowledge-update"},
    ]
    picked = arena.select_suite_b_items(records, n=2)
    assert [item["question_id"] for item in picked] == ["a", "b"]
    with pytest.raises(RuntimeError, match="expected 5"):
        arena.select_suite_b_items(records, n=5)

    item = {
        "question_id": "ku1",
        "answer_session_ids": ["s2"],
        "haystack_session_ids": ["s1", "s2"],
        "haystack_sessions": [
            [{"role": "user", "content": "old session", "has_answer": True}],
            [
                {"role": "user", "content": "first", "has_answer": True},
                {"role": "assistant", "content": "ignore assistant", "has_answer": True},
                {"role": "user", "content": "latest fact", "has_answer": True},
            ],
        ],
    }
    assert arena.knowledge_update_retraction_text(item) == "latest fact"
    chosen_message = arena.select_retraction_message(
        [
            {"content": "latest fact", "echo": True, "ts": 50, "session_id": "echo"},
            {"content": "latest fact", "echo": False, "ts": 10, "session_id": "early"},
            {"content": "latest fact", "echo": False, "ts": 30, "session_id": "late"},
            {"content": "other", "ts": 90, "session_id": "nope"},
        ],
        "latest fact",
    )
    assert chosen_message["session_id"] == "late"
    assert arena.check_stale_resurrection("see OLDvalue here", ["ab", "oldvalue"]) is True
    split = arena.split_stale_leakage("oldvalue", "clean answer", ["oldvalue"])
    assert split == {"stale_context": True, "stale_answer": False}
    assert arena.format_mem0_results({"results": []}) == "(No memories found.)"
    assert arena.format_mem0_results({"results": [{"memory": "alpha"}]}) == "- alpha"
    stamped = arena.reference_time_from_ts(3)
    assert stamped == datetime(2024, 1, 4, tzinfo=timezone.utc)


def test_reader_retries_and_requires_usage(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WAVE11_FAST_RETRY", "1")
    calls: List[Dict[str, Any]] = []

    class _Http:
        def __init__(self, status: int, body: Dict[str, Any]) -> None:
            self.status_code = status
            self.text = "nope"
            self._body = body

        def json(self) -> Dict[str, Any]:
            return self._body

    def _post(url: str, json: Optional[Dict[str, Any]] = None, timeout: float = 0) -> _Http:
        del timeout
        calls.append({"url": url, "json": json or {}})
        if len(calls) == 1:
            return _Http(500, {})
        return _Http(
            200,
            {
                "choices": [{"message": {"content": " ABSTAIN "}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 1},
            },
        )

    text, prompt, completion, _wall = arena.call_nemotron_reader(
        "http://127.0.0.1:8891/v1",
        "nemotron-lightning",
        "system",
        "user",
        post=_post,
    )
    assert (text, prompt, completion) == ("ABSTAIN", 3, 1)
    assert calls[0]["url"].endswith("/chat/completions")
    payload = calls[0]["json"]
    assert payload["temperature"] == 0.0
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert len(calls) == 2

    def _missing(url: str, json: Optional[Dict[str, Any]] = None, timeout: float = 0) -> _Http:
        del url, json, timeout
        return _Http(200, {"choices": [{"message": {"content": "x"}}], "usage": {}})

    with pytest.raises(RuntimeError, match="reader response.usage"):
        arena.call_nemotron_reader("http://127.0.0.1:8891/v1", "nemotron-lightning", "s", "u", post=_missing)


def test_jsonl_roundtrip_and_docker_contract(tmp_path: Path) -> None:
    path = tmp_path / "suite_a.jsonl"
    runner.append_jsonl(path, {"item_id": "a", "n": 1})
    raw = path.read_bytes()
    assert b"\r" not in raw and raw.endswith(b"\n")
    assert runner.load_jsonl_rows(path) == [{"item_id": "a", "n": 1}]
    with pytest.raises(RuntimeError, match="duplicate"):
        runner.append_jsonl(path, {"item_id": "a"})
    runner.rewrite_jsonl(path, [{"item_id": "b", "judge_complete": True}])
    assert not path.with_name(path.name + ".tmp").exists()
    assert runner.load_jsonl_rows(path)[0]["item_id"] == "b"
    path.write_bytes(b'{"item_id": "c"}\r\n')
    with pytest.raises(RuntimeError, match="CR"):
        runner.load_jsonl_rows(path)
    dup = tmp_path / "dup.jsonl"
    dup.write_text('{"item_id": "a"}\n{"item_id": "a"}\n', encoding="utf-8", newline="\n")
    with pytest.raises(RuntimeError, match="duplicate"):
        runner.load_jsonl_rows(dup)

    argv = runner.docker_run_argv()
    assert argv == [
        "docker",
        "run",
        "--detach",
        "--name",
        "wave11-d7-neo4j",
        "--memory=8g",
        "--publish",
        "127.0.0.1:17474:7474",
        "--publish",
        "127.0.0.1:17687:7687",
        "--env",
        "NEO4J_AUTH=neo4j/chelated_bench_2026",
        "neo4j:5.26-community",
    ]
    runner.assert_docker_argv_safe(argv)
    runner.assert_docker_argv_safe(list(runner.DOCKER_RM_ARGV))
    with pytest.raises(RuntimeError, match="nemotron"):
        runner.assert_docker_argv_safe(["docker", "rm", "-f", "nemotron-37c3"])
    seen: List[List[str]] = []

    class _Code:
        def __init__(self, code: int) -> None:
            self.returncode = code

    def _run(argv_in: List[str], check: bool = False) -> _Code:
        del check
        seen.append(list(argv_in))
        return _Code(0)

    assert runner.start_wave11_neo4j(run_cmd=_run) == argv
    assert seen[0] == ["docker", "rm", "-f", "wave11-d7-neo4j"]
    assert seen[1] == argv
    seen.clear()

    def _fail(argv_in: List[str], check: bool = False) -> _Code:
        del check
        seen.append(list(argv_in))
        return _Code(0 if argv_in[1] == "rm" else 1)

    with pytest.raises(RuntimeError, match="docker run"):
        runner.start_wave11_neo4j(run_cmd=_fail)
    stop_seen: List[List[str]] = []
    runner.stop_wave11_neo4j(run_cmd=lambda argv_in, check=False: stop_seen.append(list(argv_in)) or _Code(0))
    assert stop_seen == [["docker", "rm", "-f", "wave11-d7-neo4j"]]


def test_local_check_does_not_execute(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    def _boom(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("live path")

    monkeypatch.setattr(runner, "emit_wave11_artifacts", _boom)
    monkeypatch.setattr(runner, "start_wave11_neo4j", _boom)
    monkeypatch.setattr(runner, "stop_wave11_neo4j", _boom)
    before = set(sys.modules)
    assert runner.main(["--mode", "local-check"]) == 0
    added = set(sys.modules) - before
    assert not any(name == "mem0" or name.startswith("mem0.") or name == "torch" or name.startswith("torch.") for name in added)
    payload = json.loads(capsys.readouterr().out)
    assert payload["live_receipt_written"] is False
    assert payload["docker_executed"] is False
    assert payload["live_executed"] is False
    assert payload["max_coroutines"] == 6


def test_spark1_guards_refuse_docker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _start(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("docker")

    monkeypatch.setattr(runner, "start_wave11_neo4j", _start)
    with pytest.raises(SystemExit, match="osv-cache"):
        runner.main(["--mode", "spark1-live"])

    import prospective_realworld_drift_corpus as corpus

    def _builder(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("corpus builder")

    monkeypatch.setattr(corpus, "build_prospective_realworld_drift_suite", _builder)
    missing = tmp_path / "missing-osv.jsonl"
    lme = tmp_path / "lme.json"
    lme.write_text("[]", encoding="utf-8", newline="\n")
    args = runner.parse_args(
        ["--mode", "spark1-live", "--osv-cache", str(missing), "--lme-json", str(lme), "--output-dir", str(tmp_path / "out")]
    )
    with pytest.raises(FileNotFoundError):
        asyncio.run(runner.execute_spark1_live(args, start_container=_start, stop_container=_start))

    bad_model = runner.parse_args(
        [
            "--mode",
            "spark1-live",
            "--osv-cache",
            str(lme),
            "--lme-json",
            str(lme),
            "--vllm-model",
            "other",
            "--output-dir",
            str(tmp_path / "out2"),
        ]
    )
    with pytest.raises(ValueError, match="nemotron-lightning"):
        asyncio.run(runner.execute_spark1_live(bad_model, start_container=_start))

    osv = tmp_path / "osv.jsonl"
    osv.write_text("{}\n", encoding="utf-8", newline="\n")
    out = tmp_path / "corrupt"
    out.mkdir()
    (out / "suite_a.jsonl").write_text("{not json}\n", encoding="utf-8", newline="\n")
    corrupt_args = runner.parse_args(
        ["--mode", "spark1-live", "--osv-cache", str(osv), "--lme-json", str(lme), "--output-dir", str(out)]
    )
    with pytest.raises(RuntimeError, match="corrupt JSONL"):
        asyncio.run(runner.execute_spark1_live(corrupt_args, start_container=_start, stop_container=_start))
    dup_out = tmp_path / "dup"
    dup_out.mkdir()
    (dup_out / "suite_a.jsonl").write_text(
        '{"item_id": "a"}\n{"item_id": "a"}\n',
        encoding="utf-8",
        newline="\n",
    )
    dup_args = runner.parse_args(
        ["--mode", "spark1-live", "--osv-cache", str(osv), "--lme-json", str(lme), "--output-dir", str(dup_out)]
    )
    with pytest.raises(RuntimeError, match="duplicate"):
        asyncio.run(runner.execute_spark1_live(dup_args, start_container=_start, stop_container=_start))


def test_complete_checkpoint_emits_receipt_without_docker(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import prospective_realworld_drift_corpus as corpus

    def _builder(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("corpus builder")

    def _start() -> None:
        raise AssertionError("docker")

    monkeypatch.setattr(corpus, "build_prospective_realworld_drift_suite", _builder)
    osv = tmp_path / "osv.jsonl"
    lme = tmp_path / "lme.json"
    osv.write_text("{}\n", encoding="utf-8", newline="\n")
    lme.write_text("[]", encoding="utf-8", newline="\n")
    out = tmp_path / "arena"
    rows_a = [
        _arm_row(
            f"a{i:02d}",
            abstain=i < 2,
            stale=["oldvalue"] if i == 3 else None,
            fail_arm=arena.ARM_MEM0 if i == 4 else None,
        )
        for i in range(60)
    ]
    rows_b = [_arm_row(f"b{i:02d}") for i in range(20)]
    runner.rewrite_jsonl(out / "suite_a.jsonl", rows_a)
    runner.rewrite_jsonl(out / "suite_b.jsonl", rows_b)
    args = runner.parse_args(
        ["--mode", "spark1-live", "--osv-cache", str(osv), "--lme-json", str(lme), "--output-dir", str(out)]
    )
    manifest = asyncio.run(runner.execute_spark1_live(args, start_container=_start, stop_container=_start))
    assert manifest["suite_a_n"] == 60
    assert manifest["suite_b_n"] == 20
    assert manifest["suite_b_pooled"] is False
    assert manifest["max_coroutines"] == 6
    assert manifest["host"] == "mattmre@10.0.0.49"
    assert manifest["untouched_container"] == "nemotron-37c3"
    assert manifest["evidence_tier"] == "spark1-live"
    assert manifest["local_check_is_not_evidence"] is True
    assert manifest["suite_b_descriptive"]["holm_bonferroni"] is None
    assert manifest["suite_b_descriptive"]["inferential_claim"] is False
    holm = manifest["suite_a_inference"]["settled"]["holm_bonferroni"]
    assert set(holm) == HOLM_KEYS
    assert manifest["suite_a_inference"]["settled"]["U_separate"]["in_holm_family"] is False
    manifest_path = out / "wave11_d7_zep_graphiti_manifest.json"
    receipt_path = out / "report_receipt.json"
    manifest_bytes = manifest_path.read_bytes()
    receipt_bytes = receipt_path.read_bytes()
    assert b"\r" not in manifest_bytes and b"\r" not in receipt_bytes
    assert manifest_bytes.endswith(b"\n") and receipt_bytes.endswith(b"\n")
    receipt = json.loads(receipt_bytes.decode("utf-8"))
    assert receipt["manifest_sha256"] == hashlib.sha256(manifest_bytes).hexdigest()
    assert receipt["suite_b_pooled"] is False
    assert receipt["suite_a_n"] == 60


def test_repaired_wrong_fields_retry_as_json_object_with_schema() -> None:
    fake, metered = _client()
    fenced = '```json\n{"wrong": 1}\n```'
    fake.chat.completions.script.extend(
        [
            _Resp(fenced, _Usage(4, 2)),
            _Resp('{"ok": true}', _Usage(5, 1)),
        ]
    )
    assert _generate(metered) == {"ok": True}
    assert metered.structured_output_mode == "json_object"
    assert metered.validation_retry_count == 1
    assert metered.schema_fallback_count == 1
    assert metered.json_repair_count == 1
    assert len(fake.chat.completions.calls) == 2
    assert fake.chat.completions.calls[0]["response_format"]["type"] == "json_schema"
    assert fake.chat.completions.calls[1]["response_format"] == {"type": "json_object"}
    schema_text = json.dumps(_OkModel.model_json_schema())
    initial = fake.chat.completions.calls[0]["messages"][-1]["content"]
    retry_content = fake.chat.completions.calls[1]["messages"][-1]["content"]
    assert schema_text not in initial
    assert schema_text in retry_content
    assert "never output '$defs'" in retry_content
    assert "NOT the JSON Schema" in retry_content
    assert metered.token_snapshot()["validation_retry_count"] == 1
    metered.reset_counters(phase="read")
    assert metered.validation_retry_count == 1
    assert metered.schema_fallback_count == 1
    assert metered.json_repair_count == 1


def test_json_object_defs_echo_retries_with_json_object_not_json_schema() -> None:
    fake, metered = _client()
    metered.structured_output_mode = "json_object"
    echo = '{"$defs": {}, "properties": {"ok": {"type": "boolean"}}}'
    fake.chat.completions.script.extend(
        [
            _Resp(echo, _Usage(6, 4)),
            _Resp('{"ok": true}', _Usage(7, 2)),
        ]
    )
    assert _generate(metered) == {"ok": True}
    assert metered.validation_retry_count == 1
    assert metered.schema_fallback_count == 0
    assert metered.structured_output_mode == "json_object"
    assert len(fake.chat.completions.calls) == 2
    assert fake.chat.completions.calls[0]["response_format"] == {"type": "json_object"}
    assert fake.chat.completions.calls[1]["response_format"] == {"type": "json_object"}
    schema_text = json.dumps(_OkModel.model_json_schema())
    retry_content = fake.chat.completions.calls[1]["messages"][-1]["content"]
    assert schema_text in retry_content
    assert "never output '$defs'" in retry_content
    assert "properties" in retry_content


def test_length_truncation_counted_on_initial_and_retry() -> None:
    fake, metered = _client()
    fake.chat.completions.script.extend(
        [
            _Resp('{"wrong": 1}', _Usage(3, 2), finish_reason="length"),
            _Resp('{"ok": true}', _Usage(4, 1), finish_reason="length"),
        ]
    )
    assert _generate(metered) == {"ok": True}
    assert metered.length_truncation_count == 2
    assert metered.validation_retry_count == 1
    assert metered.llm_call_count == 2
    snap = metered.token_snapshot()
    assert snap["length_truncation_count"] == 2
    assert snap["validation_retry_count"] == 1
    metered.reset_counters(phase="write")
    assert metered.length_truncation_count == 2
    assert metered.validation_retry_count == 1
    assert metered.llm_call_count == 0

    fake_ok, metered_ok = _client()
    fake_ok.chat.completions.script.append(_Resp('{"ok": true}', _Usage(1, 1), finish_reason="length"))
    assert _generate(metered_ok) == {"ok": True}
    assert metered_ok.length_truncation_count == 1
    assert metered_ok.validation_retry_count == 0
    assert len(fake_ok.chat.completions.calls) == 1

    fake_stop, metered_stop = _client()
    fake_stop.chat.completions.script.append(_Resp('{"ok": true}', _Usage(1, 1), finish_reason="stop"))
    assert _generate(metered_stop) == {"ok": True}
    assert metered_stop.length_truncation_count == 0


def test_validation_retry_http_400_switches_to_json_object_with_schema() -> None:
    fake, metered = _client()
    fake.chat.completions.script.extend(
        [
            _Resp('{"wrong": 1}', _Usage(2, 2)),
            _HTTP400(),
            _Resp('{"ok": true}', _Usage(3, 1)),
        ]
    )
    assert _generate(metered) == {"ok": True}
    assert metered.validation_retry_count == 1
    assert metered.schema_fallback_count == 1
    assert metered.structured_output_mode == "json_object"
    assert metered.length_truncation_count == 0
    assert len(fake.chat.completions.calls) == 3
    assert fake.chat.completions.calls[0]["response_format"]["type"] == "json_schema"
    assert fake.chat.completions.calls[1]["response_format"]["type"] == "json_schema"
    assert fake.chat.completions.calls[2]["response_format"] == {"type": "json_object"}
    schema_text = json.dumps(_OkModel.model_json_schema())
    first_retry = fake.chat.completions.calls[1]["messages"][-1]["content"]
    fallback = fake.chat.completions.calls[2]["messages"][-1]["content"]
    assert "never output '$defs'" in first_retry
    assert schema_text not in first_retry
    assert schema_text in fallback
    assert "never output '$defs'" in fallback


def test_owned_sources_have_no_silent_except_pass() -> None:
    paths = [
        REPO / "chelated_graphiti_arena_v1.py",
        REPO / "scripts" / "run_wave11_d7_live_zep_graphiti_arena.py",
        REPO / "tests" / "test_wave11_d7_zep_graphiti.py",
    ]
    pattern = re.compile(
        r"^\s*except(?:\s+Exception(?:\s+as\s+\w+)?)?\s*:\s*(?:#.*)?\n\s*pass\s*$",
        re.MULTILINE,
    )
    for path in paths:
        text = path.read_text(encoding="utf-8")
        assert pattern.search(text) is None, path
    for path in paths[:2]:
        text = path.read_text(encoding="utf-8")
        assert "_pending_retraction" not in text
        assert "def _graphiti_race_search" not in text
