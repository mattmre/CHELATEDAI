"""Unit tests for Wave-6 K5 4-Way Competitor Arena stores, separated clocks, and Delta=0s freshness."""

from __future__ import annotations

import numpy as np

from scripts.run_wave6_pm3_v2_4way_arena import (
    Hindsight4NetEmulationStore,
    OpenSearchMemoryAsyncEmulation,
    check_stale_resurrection,
    summarize_suite,
)


def _toy_embed(texts: list[str]) -> np.ndarray:
    out = np.zeros((len(texts), 16), dtype=np.float32)
    for i, t in enumerate(texts):
        h = hash(t)
        for j in range(16):
            out[i, j] = float(((h >> j) & 7) + 1)
    norms = np.linalg.norm(out, axis=1, keepdims=True)
    return out / np.maximum(norms, 1e-6)


def test_delta0s_vs_post_consolidation_and_unpooled_summary(monkeypatch) -> None:
    import scripts.run_wave6_pm3_v2_4way_arena as arena

    def fake_vllm(_ep: str, _mdl: str, sys_p: str, _usr: str, *, max_tokens: int = 256):
        if "Retain" in sys_p:
            return (
                '{"world_facts": ["CVE-2024-3094 fixed in 5.6.2"], '
                '"entity_summaries": {"CVE-2024-3094": "Fixed in 5.6.2"}, '
                '"evolving_beliefs": [{"claim": "patch is 5.6.2", "confidence": 0.92}]}',
                120,
                15.0,
            )
        if "SEMANTIC" in sys_p:
            return '{"semantic_memories": ["CVE-2024-3094 fixed version is 5.6.2"]}', 100, 12.0
        return "- Summary: CVE-2024-3094 updated to 5.6.2", 90, 10.0

    monkeypatch.setattr(arena, "call_vllm_timed", fake_vllm)

    raw_events = [
        "Preliminary note: CVE-2024-3094 workaround is 5.6.0-beta",
        "Echo 1: CVE-2024-3094 workaround is 5.6.0-beta",
        "Official advisory: CVE-2024-3094 supersedes 5.6.0-beta with fixed version 5.6.2",
    ]
    osm = OpenSearchMemoryAsyncEmulation(_toy_embed)
    ack_ms = osm.write_events_immediate(raw_events)
    assert ack_ms >= 0.0
    ctx_d0, _ = osm.retrieve("What is the fixed version for CVE-2024-3094?", top_k=3, delta_zero_before_async=True)
    assert check_stale_resurrection(ctx_d0, ["5.6.0-beta"]) is True

    bg_ms, bg_toks = osm.run_async_strategies(["\n".join(raw_events)], "http://fake", "fake")
    assert bg_ms >= 0.0 and bg_toks == 190
    ctx_post, _ = osm.retrieve("What is the fixed version for CVE-2024-3094?", top_k=2, delta_zero_before_async=False)
    assert "[SEMANTIC]" in ctx_post

    hs = Hindsight4NetEmulationStore(_toy_embed)
    hs.write_raw_experiences_ack(raw_events)
    hs_bg, hs_toks = hs.retain_consolidate(["\n".join(raw_events)], "http://fake", "fake")
    assert hs_bg >= 0.0 and hs_toks == 120
    hs_ctx, _ = hs.recall("CVE-2024-3094", top_k=4, delta_zero_unconsolidated=False)
    assert "Belief" in hs_ctx or "WorldFact" in hs_ctx or "EntitySummary" in hs_ctx

    # Verify summarize_suite computes 4 separated clocks & distinct bootstrap quantiles
    fake_rows = []
    for idx in range(50):
        fake_rows.append({
            "item_id": f"id_{idx:03d}",
            "abstention_required": (idx % 5 == 0),
            "stale_values": ["5.6.0-beta"],
            "arms": {
                "Chelated_v4_Stratified_WFS": {
                    "correct": True,
                    "correct_delta0s": True,
                    "stale_resurrected_post": False,
                    "stale_resurrected_delta0s": False,
                    "write_ack_ms": 1.5,
                    "consolidation_bg_ms": 20.0,
                    "read_retrieval_ms": 1.2,
                    "reader_llm_ms": 80.0,
                    "write_llm_tokens": 150,
                    "read_prompt_tokens": 320,
                    "judge": {"schema_anomalies": {}},
                },
                "OpenSearch_Memory_Async_Emulation": {
                    "correct": (idx % 2 == 0),
                    "correct_delta0s": (idx % 4 == 0),
                    "stale_resurrected_post": (idx % 3 == 0),
                    "stale_resurrected_delta0s": True,
                    "write_ack_ms": 2.0,
                    "consolidation_bg_ms": 400.0,
                    "read_retrieval_ms": 1.8,
                    "reader_llm_ms": 95.0,
                    "write_llm_tokens": 2400,
                    "read_prompt_tokens": 680,
                    "judge": {"schema_anomalies": {}},
                },
            },
        })
    summ = summarize_suite(
        "TestSuite",
        fake_rows,
        ("Chelated_v4_Stratified_WFS", "OpenSearch_Memory_Async_Emulation"),
    )
    assert summ["arms"]["Chelated_v4_Stratified_WFS"]["stale_resurrection_delta0s_immediate"] == 0.0
    assert summ["arms"]["OpenSearch_Memory_Async_Emulation"]["stale_resurrection_delta0s_immediate"] == 1.0
    comp = summ["paired_comparisons_post_consolidation"]["Chelated_v4_Stratified_WFS_minus_OpenSearch_Memory_Async_Emulation"]
    assert comp["lcb_975"] > 0.0


def test_normalize_mem0_llm_json_repairs_bare_string_lists() -> None:
    import json
    from scripts.run_wave6_pm3_v2_4way_arena import _normalize_mem0_llm_json_str

    repaired, n_rep = _normalize_mem0_llm_json_str('{"memory": ["User likes matcha", {"fact": "Lives in Austin"}]}')
    assert n_rep >= 2
    parsed = json.loads(repaired)
    assert all(isinstance(x, dict) and x.get("text") and x.get("event") for x in parsed["memory"])

    rep_facts, nf = _normalize_mem0_llm_json_str('{"facts": [{"text": "Updated phone"}, "Prefers morning runs"]}')
    assert nf == 1
    pf = json.loads(rep_facts)
    assert pf["facts"] == ["Updated phone", "Prefers morning runs"]

