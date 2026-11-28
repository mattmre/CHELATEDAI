"""Runner for Wave 18 / Bundle #50 (Item 3A): LOFT SciFact Pilot + Disjoint Zero-Write Exact Tango NI.

Supports:
  - `--mode local-drycheck` (default, CPU-only, `CUDA_VISIBLE_DEVICES=""`): runs deterministic
    canonical LOFT SciFact pilot (357 docs, 100 queries), derived zero-write migration pilot
    (`Rank1QueryResidual` + `ZeroMuPolarBridge`), `burn_ledger` rejection audit, power-sizing
    (`N >= 1,000`), underpowered pool check, and power-sized disjoint exact Tango NI vs `-0.0500`.
  - `--mode run-live-gpu`: evaluates live pre-encoded or external disjoint corpus JSON payloads
    against the locked `-0.0500` Tango NI gate (failing closed if required live inputs or
    complete historical fingerprint inventories are absent; never labels drycheck data as live).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chelated_loft_and_zero_write_ni import (  # noqa: E402
    BUNDLE_ID,
    HISTORICAL_PM2_TANGO_LCB,
    ITEM_ID,
    LOFT_SCIFACT_DOC_COUNT,
    LOFT_SCIFACT_QUERY_COUNT,
    MIN_DISJOINT_PAIRED_QUERIES,
    PREREGISTERED_NI_MARGIN,
    SPINE_DIGEST,
    audit_burned_corpora_rejection,
    build_deterministic_loft_scifact_fixture,
    build_synthetic_disjoint_ni_pool,
    evaluate_disjoint_zero_write_ni,
    run_canonical_loft_scifact_pilot,
    run_derived_zero_write_migration_pilot,
    size_disjoint_ni_sample,
)
from chelated_pv7_spine.burn_ledger import build_burn_ledger, text_fingerprint  # noqa: E402
from chelated_pv7_spine.claim_registry import REGISTRY_PATH, claim_registry_sha256  # noqa: E402
from chelated_pv7_spine.cpu_io_cage import assert_zero_cuda_context, build_cpu_only_env  # noqa: E402

MODES = ("local-drycheck", "run-live-gpu")


def _gate(name: str, passed: bool, detail: str) -> Dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _drycheck_historical_inventory() -> Dict[str, List[str]]:
    return {
        "documents": [text_fingerprint("pv7-b50-historical-baseline-doc-marker")],
        "queries": [text_fingerprint("pv7-b50-historical-baseline-query-marker")],
    }


def _run_live_gpu(
    *,
    disjoint_input_json: Optional[Path] = None,
    historical_fingerprints_json: Optional[Path] = None,
) -> Dict[str, Any]:
    if disjoint_input_json is None or historical_fingerprints_json is None:
        return {
            "bundle_id": BUNDLE_ID,
            "item": ITEM_ID,
            "mode": "run-live-gpu",
            "status": "awaiting-live-inputs",
            "passed": False,
            "reason": (
                "Live confirmatory mode requires --disjoint-input-json and "
                "--historical-fingerprints-json; refusing synthetic fallback."
            ),
            "ni_margin": PREREGISTERED_NI_MARGIN,
            "historical_pm2_tango_lcb": HISTORICAL_PM2_TANGO_LCB,
            "spine_digest": claim_registry_sha256(),
        }

    payload = json.loads(Path(disjoint_input_json).read_text(encoding="utf-8"))
    hist_fps = json.loads(Path(historical_fingerprints_json).read_text(encoding="utf-8"))
    weight_hex = payload.get("weight_bytes_hex")
    weight_bytes = bytes.fromhex(weight_hex) if isinstance(weight_hex, str) and weight_hex else None
    sizing = size_disjoint_ni_sample(
        assumed_p10=float(payload.get("assumed_p10", 0.032)),
        assumed_p01=float(payload.get("assumed_p01", 0.030)),
        ni_margin=PREREGISTERED_NI_MARGIN,
        min_n=MIN_DISJOINT_PAIRED_QUERIES,
    )
    ledger = build_burn_ledger(REPO_ROOT)
    ni_report = evaluate_disjoint_zero_write_ni(
        corpus_label=str(payload["corpus_label"]),
        doc_texts=payload["doc_texts"],
        query_texts=payload["query_texts"],
        zero_write_hits=payload["zero_write_hits"],
        reference_hits=payload["reference_hits"],
        sizing_plan=sizing,
        historical_fingerprints=hist_fps,
        ledger=ledger,
        ni_margin=PREREGISTERED_NI_MARGIN,
        is_live_confirmatory=True,
        weight_bytes=weight_bytes,
        weight_paths=payload.get("weight_paths"),
        provenance_manifest=payload.get("provenance_manifest"),
    )
    return {
        "bundle_id": BUNDLE_ID,
        "item": ITEM_ID,
        "mode": "run-live-gpu",
        "status": ni_report["status"],
        "passed": ni_report["status"] == "verified-pass",
        "disjoint_confirmatory_ni": ni_report,
        "spine_digest": claim_registry_sha256(),
    }


def execute(
    mode: str = "local-drycheck",
    *,
    disjoint_input_json: Optional[Path] = None,
    historical_fingerprints_json: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute Bundle #50 (`Item 3A`) in `local-drycheck` (CPU) or `run-live-gpu` mode."""
    if mode not in MODES:
        raise ValueError(f"Unsupported mode {mode!r}; expected one of {MODES}")
    if mode == "run-live-gpu":
        return _run_live_gpu(
            disjoint_input_json=disjoint_input_json,
            historical_fingerprints_json=historical_fingerprints_json,
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    cpu_env = build_cpu_only_env(os.environ)
    cuda_receipt = assert_zero_cuda_context(env=cpu_env, strict=True)
    registry_before = REGISTRY_PATH.read_bytes()

    # 1. Canonical gBench LOFT SciFact Pilot (strictly isolated, 357 docs, 100 queries)
    loft_fixture = build_deterministic_loft_scifact_fixture(seed=5001)
    canonical_pilot = run_canonical_loft_scifact_pilot(loft_fixture)

    # 2. Derived Zero-Write Migration Pilot (Rank1QueryResidual + ZeroMuPolarBridge)
    derived_pilot = run_derived_zero_write_migration_pilot(loft_fixture)

    # 3. Mechanical burn_ledger rejection audit
    ledger = build_burn_ledger(REPO_ROOT)
    burn_audit = audit_burned_corpora_rejection(ledger=ledger)

    # 4. Exact power-sizing from paired discordance variance (N >= 1,000)
    sizing = size_disjoint_ni_sample(
        assumed_p10=0.032,
        assumed_p01=0.030,
        ni_margin=PREREGISTERED_NI_MARGIN,
        min_n=MIN_DISJOINT_PAIRED_QUERIES,
    )
    hist_fps = _drycheck_historical_inventory()

    # 5. Underpowered disjoint pool check -> must emit status="underpowered", never loosen -0.0500
    underpowered_pool = build_synthetic_disjoint_ni_pool(
        corpus_label="pv7-b50-disjoint-underpowered-check",
        n_queries=250,
        n_docs=120,
        seed=5049,
    )
    underpowered_report = evaluate_disjoint_zero_write_ni(
        corpus_label=underpowered_pool["corpus_label"],
        doc_texts=underpowered_pool["doc_texts"],
        query_texts=underpowered_pool["query_texts"],
        zero_write_hits=underpowered_pool["zero_write_hits"],
        reference_hits=underpowered_pool["reference_hits"],
        sizing_plan=sizing,
        historical_fingerprints=hist_fps,
        ledger=ledger,
        ni_margin=PREREGISTERED_NI_MARGIN,
        is_live_confirmatory=False,
        allow_local_drycheck_smoke=True,  # synthetic CPU drycheck inventory is partial by construction
    )

    # 6. Power-sized disjoint corpus (N = 1,024 >= 1,000) one-shot exact Tango 95% LCB vs -0.0500
    disjoint_pool = build_synthetic_disjoint_ni_pool(
        corpus_label="pv7-b50-disjoint-power-sized-ni",
        n_queries=1024,
        n_docs=240,
        seed=5050,
    )
    disjoint_ni_report = evaluate_disjoint_zero_write_ni(
        corpus_label=disjoint_pool["corpus_label"],
        doc_texts=disjoint_pool["doc_texts"],
        query_texts=disjoint_pool["query_texts"],
        zero_write_hits=disjoint_pool["zero_write_hits"],
        reference_hits=disjoint_pool["reference_hits"],
        sizing_plan=sizing,
        historical_fingerprints=hist_fps,
        ledger=ledger,
        ni_margin=PREREGISTERED_NI_MARGIN,
        is_live_confirmatory=False,
        allow_local_drycheck_smoke=True,  # synthetic CPU drycheck inventory is partial by construction
    )

    registry_after = REGISTRY_PATH.read_bytes()
    actual_spine = claim_registry_sha256()

    gates: List[Dict[str, Any]] = [
        _gate("zero_cuda_cpu_cage", cuda_receipt.passed, f"cuda_visible={cuda_receipt.cuda_visible_devices!r}"),
        _gate(
            "canonical_loft_scifact_dimensions",
            canonical_pilot["doc_count"] == LOFT_SCIFACT_DOC_COUNT
            and canonical_pilot["query_count"] == LOFT_SCIFACT_QUERY_COUNT
            and canonical_pilot["confirmatory_ni_pooled"] is False,
            f"docs={canonical_pilot['doc_count']} queries={canonical_pilot['query_count']}",
        ),
        _gate(
            "derived_pilot_zero_write_and_separated",
            derived_pilot["separated_from_canonical_gbench"] is True
            and derived_pilot["resident_docs_unmodified"] is True
            and derived_pilot["resident_write_bytes"] == 0,
            f"arms={len(derived_pilot['arms'])} write_bytes={derived_pilot['resident_write_bytes']}",
        ),
        _gate(
            "burn_ledger_rejects_all_historical_corpora",
            burn_audit["all_burned_rejected"] is True,
            f"checked={len(burn_audit['checked_corpora'])}",
        ),
        _gate(
            "power_sizing_enforces_min_1000",
            sizing.required_n >= MIN_DISJOINT_PAIRED_QUERIES,
            f"required_n={sizing.required_n} var_disc={sizing.paired_discordance_variance:.6f}",
        ),
        _gate(
            "underpowered_pool_emits_underpowered_without_loosening",
            underpowered_report["status"] == "underpowered"
            and underpowered_report["ni_margin"] == PREREGISTERED_NI_MARGIN
            and underpowered_report["margin_loosened"] is False,
            f"status={underpowered_report['status']} margin={underpowered_report['ni_margin']}",
        ),
        _gate(
            "disjoint_tango_ni_clears_neg_0050",
            disjoint_ni_report["status"] == "local-drycheck-pass"
            and disjoint_ni_report["tango_ni"]["tango_score_lcb95"] > PREREGISTERED_NI_MARGIN
            and disjoint_ni_report["available_paired_queries"] >= MIN_DISJOINT_PAIRED_QUERIES,
            f"lcb95={disjoint_ni_report['tango_ni']['tango_score_lcb95']:.6f} margin={PREREGISTERED_NI_MARGIN:.4f}",
        ),
        _gate(
            "historical_pm2_neg_00572_preserved",
            disjoint_ni_report["historical_pm2_reference"]["tango_score_lcb95"] == HISTORICAL_PM2_TANGO_LCB
            and disjoint_ni_report["historical_pm2_reference"]["preserved_unmodified"] is True,
            f"pm2_lcb={HISTORICAL_PM2_TANGO_LCB}",
        ),
        _gate(
            "claim_registry_unmodified",
            registry_before == registry_after and actual_spine == SPINE_DIGEST,
            f"spine_digest={actual_spine}",
        ),
    ]

    passed = all(g["passed"] for g in gates)
    return {
        "bundle_id": BUNDLE_ID,
        "item": ITEM_ID,
        "mode": mode,
        "status": "local-drycheck-pass" if passed else "local-drycheck-fail",
        "passed": passed,
        "spine_digest": actual_spine,
        "gates": gates,
        "canonical_loft_scifact_pilot": canonical_pilot,
        "derived_zero_write_pilot": derived_pilot,
        "burn_ledger_audit": burn_audit,
        "power_sizing": sizing.to_dict(),
        "underpowered_probe": underpowered_report,
        "disjoint_zero_write_ni": disjoint_ni_report,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, default="local-drycheck")
    parser.add_argument("--disjoint-input-json", type=Path, default=None)
    parser.add_argument("--historical-fingerprints-json", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    report = execute(
        mode=args.mode,
        disjoint_input_json=args.disjoint_input_json,
        historical_fingerprints_json=args.historical_fingerprints_json,
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8", newline="\n")
    print(rendered)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
