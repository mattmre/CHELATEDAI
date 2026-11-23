"""Unit tests for the workable live solution demonstration server (`demo_workbench/server.py`)."""

from __future__ import annotations

import json

import pytest

from demo_workbench.server import dispatch_api_request


def test_live_epistemic_wfs_retraction_bitemporal_and_contested() -> None:
    scenario = {
        "query": "CVE-2024-3094 fixed version",
        "events": [
            {
                "event_id": "v1_beta",
                "entity": "CVE-2024-3094",
                "slot": "fixed_version",
                "claim_value": "5.6.0-beta",
                "authority_tier": 3,
                "valid_time": 1,
                "tx_time": 1,
                "text": "CVE-2024-3094 fixed version is 5.6.0-beta",
            },
            {
                "event_id": "echo_1",
                "entity": "CVE-2024-3094",
                "slot": "fixed_version",
                "claim_value": "5.6.0-beta",
                "authority_tier": 1,
                "valid_time": 2,
                "tx_time": 2,
                "text": "CVE-2024-3094 fixed version is 5.6.0-beta in forum recap",
            },
            {
                "event_id": "v2_patch",
                "entity": "CVE-2024-3094",
                "slot": "fixed_version",
                "claim_value": "5.6.2",
                "authority_tier": 4,
                "valid_time": 3,
                "tx_time": 5,
                "text": "Official bulletin: CVE-2024-3094 fixed version is 5.6.2 superseding 5.6.0-beta",
            },
        ],
        "edges": [
            {"src_id": "echo_1", "dst_id": "v1_beta", "edge_type": "COPY_OF", "valid_time": 2, "tx_time": 2},
            {"src_id": "v2_patch", "dst_id": "v1_beta", "edge_type": "SUPERSEDE", "valid_time": 3, "tx_time": 5},
            {"src_id": "v2_patch", "dst_id": "echo_1", "edge_type": "REFUTE", "valid_time": 3, "tx_time": 5},
        ],
    }
    status, _ct, raw = dispatch_api_request("POST", "/api/v1/epistemic/evaluate", json.dumps(scenario).encode("utf-8"))
    assert status == 200
    body = json.loads(raw.decode("utf-8"))
    assert body["wfs_oracle_mismatches"] == 0
    assert body["chelated_admitted_topk_ids"] == ["v2_patch"]
    assert "v1_beta" in body["unmasked_delta0s_topk_ids"]

    scenario["as_of_tx"] = 2
    _, _, raw_past = dispatch_api_request("POST", "/api/v1/epistemic/evaluate", json.dumps(scenario).encode("utf-8"))
    past = json.loads(raw_past.decode("utf-8"))
    assert "v1_beta" in past["chelated_admitted_topk_ids"]


def test_live_migration_and_tco_arithmetic_identity() -> None:
    st_m, _, raw_m = dispatch_api_request(
        "POST",
        "/api/v1/migration/evaluate",
        json.dumps({"n_docs": 600, "d_low": 48, "d_high": 128, "backfill_fraction": 0.20, "k_os": 64}).encode("utf-8"),
    )
    assert st_m == 200
    m = json.loads(raw_m.decode("utf-8"))
    assert m["mode"] == "SYNTHETIC_NUMPY_MODE_B_SIMULATOR"
    assert m["stiefel_orthogonality_error"] < 1e-12
    assert m["shortlist_vs_exact_csls_recall_at_10"] >= 0.90

    st_t, _, raw_t = dispatch_api_request(
        "POST",
        "/api/v1/tco/model",
        json.dumps({"n_writes": 100_000_000, "n_reads": 500_000_000}).encode("utf-8"),
    )
    assert st_t == 200
    tco = json.loads(raw_t.decode("utf-8"))
    prov = tco["parameter_provenance"]
    assert "MEASURED_RECEIPT" in prov["mem0_infer_read_prompt_tokens_per_query"]
    assert "SENSITIVITY_ASSUMPTION" in prov["batched_write_tokens_620_740_580"]

    # Council M4: Assert exact parameterized arithmetic identity on every modeled row (never hardcode promotional rank)
    for arm_name, row in tco["systems"].items():
        expected_var = (
            row["write_llm_usd"]
            + row["read_llm_usd"]
            + row["anchor_fit_gpu_usd"]
            + row["backfill_or_reindex_gpu_usd"]
            + row["sidecar_or_extra_ram_annual_usd"]
            + row["retraction_compliance_ops_usd"]
        )
        assert row["variable_differentiation_subtotal_usd"] == pytest.approx(expected_var, abs=0.05), arm_name
        assert row["total_annualized_tco_usd"] == pytest.approx(
            row["variable_differentiation_subtotal_usd"] + row["equal_offset_base_infra_usd"], abs=0.05
        ), arm_name

    # Verify measured Mem0 Suite B read tokens (506.65 -> $151,995 at 500M reads * $0.60/1M) & OpenSearch compact default (2178.6)
    assert tco["systems"]["Mem0_InferTrue_BatchedWriteAssumption_MeasuredRead"]["read_llm_usd"] == pytest.approx(
        151_995.0, abs=1.0
    )
    assert tco["systems"]["OpenSearch_Memory_Emulation_CompactReadDefault"]["read_llm_usd"] == pytest.approx(
        653_580.0, abs=1.0
    )

    st_h, _, raw_h = dispatch_api_request("GET", "/api/v1/registry/status")
    assert st_h == 200
    reg_st = json.loads(raw_h.decode("utf-8"))
    assert all(b["sha256_verified_on_disk"] for b in reg_st["wave6_bundles"])

    st_w6, _, raw_w6 = dispatch_api_request("GET", "/api/v1/wave6/summary")
    assert st_w6 == 200
    w6 = json.loads(raw_w6.decode("utf-8"))
    assert w6["tracks_total"] == 5
    assert w6["tracks_completed"] == 5

    st_idx, ct_idx, raw_idx = dispatch_api_request("GET", "/index.html")
    assert st_idx == 200
    assert "text/html" in ct_idx
    assert b"SearchService.KnnQuery" in raw_idx
