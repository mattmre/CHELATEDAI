"""Runner for Wave 18 / Bundle #51 (Item 3B): gBench BEAM 128K Cluster Bootstrap + Paired Withdrawal Replay.

Supports:
  - `--mode local-drycheck` (default, CPU-only, `CUDA_VISIBLE_DEVICES=""`): runs deterministic
    20-conversation (400-question) BEAM 128K ingestion, conversation-level cluster bootstrap
    (`n=20`), stateful checkpoint replay, and paired derivative-only vs independent-corroboration
    withdrawal histories verifying 100% `incremental == full` `EK5` parity across every prefix.
  - `--mode run-live-gpu`: evaluates a live BEAM 128K JSON dataset path with `burn_ledger`
    verification (failing closed if required live inputs are absent; never labels drycheck data as live).
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

from chelated_beam_128k_replay import (  # noqa: E402
    BEAM_CONVERSATION_COUNT,
    BEAM_TOTAL_QUESTIONS,
    BUNDLE_ID,
    ITEM_ID,
    SPINE_DIGEST,
    ingest_beam_128k_conversations,
    run_beam_128k_paired_replay,
)
from chelated_pv7_spine.burn_ledger import build_burn_ledger, text_fingerprint  # noqa: E402
from chelated_pv7_spine.claim_registry import REGISTRY_PATH, claim_registry_sha256  # noqa: E402
from chelated_pv7_spine.cpu_io_cage import assert_zero_cuda_context, build_cpu_only_env  # noqa: E402

MODES = ("local-drycheck", "run-live-gpu")


def _gate(name: str, passed: bool, detail: str) -> Dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _drycheck_historical_inventory() -> Dict[str, List[str]]:
    return {
        "documents": [text_fingerprint("pv7-b51-historical-baseline-doc-marker")],
        "queries": [text_fingerprint("pv7-b51-historical-baseline-query-marker")],
    }


def _run_live_gpu(
    *,
    beam_source_json: Optional[Path] = None,
    historical_fingerprints_json: Optional[Path] = None,
) -> Dict[str, Any]:
    if beam_source_json is None or historical_fingerprints_json is None:
        return {
            "bundle_id": BUNDLE_ID,
            "item": ITEM_ID,
            "mode": "run-live-gpu",
            "status": "awaiting-live-inputs",
            "passed": False,
            "reason": (
                "Live BEAM 128K mode requires --beam-source-json and "
                "--historical-fingerprints-json; refusing synthetic fallback."
            ),
            "spine_digest": claim_registry_sha256(),
        }

    hist_fps = json.loads(Path(historical_fingerprints_json).read_text(encoding="utf-8"))
    ledger = build_burn_ledger(REPO_ROOT)
    convs, ingest_receipt = ingest_beam_128k_conversations(
        beam_source_json, historical_fingerprints=hist_fps, ledger=ledger
    )
    replay = run_beam_128k_paired_replay(convs, n_bootstrap=2000, seed=5102)
    provenance = ingest_receipt.get("weight_provenance") or {}
    parity = (
        replay["ek5_prefix_parity_rate"] == 1.0
        and replay["ek5_prefix_mismatches"] == 0
        and replay["checkpoint_replay_all_exact"] is True
    )
    authenticated = bool(provenance.get("authenticated"))
    superior = bool(replay.get("ek5_ni_vs_sql")) and bool(replay.get("ek5_ni_vs_withdrawal_aware_recency"))
    if not authenticated:
        status = "refused-unauthenticated-weights"
        passed = False
    elif parity and superior:
        status = "verified-pass"
        passed = True
    else:
        status = "verified-loss"
        passed = False
    return {
        "bundle_id": BUNDLE_ID,
        "item": ITEM_ID,
        "mode": "run-live-gpu",
        "status": status,
        "passed": passed,
        "ingestion": ingest_receipt,
        "replay": replay,
        "spine_digest": claim_registry_sha256(),
    }


def execute(
    mode: str = "local-drycheck",
    *,
    beam_source_json: Optional[Path] = None,
    historical_fingerprints_json: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute Bundle #51 (`Item 3B`) in `local-drycheck` (CPU) or `run-live-gpu` mode."""
    if mode not in MODES:
        raise ValueError(f"Unsupported mode {mode!r}; expected one of {MODES}")
    if mode == "run-live-gpu":
        return _run_live_gpu(
            beam_source_json=beam_source_json,
            historical_fingerprints_json=historical_fingerprints_json,
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    cpu_env = build_cpu_only_env(os.environ)
    cuda_receipt = assert_zero_cuda_context(env=cpu_env, strict=True)
    registry_before = REGISTRY_PATH.read_bytes()

    ledger = build_burn_ledger(REPO_ROOT)
    convs, ingest_receipt = ingest_beam_128k_conversations(
        None, historical_fingerprints=_drycheck_historical_inventory(), ledger=ledger, seed=5101
    )
    replay = run_beam_128k_paired_replay(convs, n_bootstrap=2000, seed=5102)
    registry_after = REGISTRY_PATH.read_bytes()
    actual_spine = claim_registry_sha256()

    joint_boot = replay["cluster_bootstrap"]["paired_joint"]
    vs_recency = replay["cluster_bootstrap"]["ek5_minus_recency_baseline"]

    gates: List[Dict[str, Any]] = [
        _gate("zero_cuda_cpu_cage", cuda_receipt.passed, f"cuda_visible={cuda_receipt.cuda_visible_devices!r}"),
        _gate(
            "beam_128k_20_conversations_400_questions",
            ingest_receipt["conversation_clusters"] == BEAM_CONVERSATION_COUNT
            and ingest_receipt["total_questions"] == BEAM_TOTAL_QUESTIONS
            and ingest_receipt["burn_ledger_verified"] is True,
            f"clusters={ingest_receipt['conversation_clusters']} questions={ingest_receipt['total_questions']}",
        ),
        _gate(
            "cluster_bootstrap_unit_is_20_conversations",
            joint_boot["sampling_unit"] == "conversation_cluster"
            and joint_boot["n_clusters"] == BEAM_CONVERSATION_COUNT
            and joint_boot["total_questions"] == BEAM_TOTAL_QUESTIONS
            and joint_boot["naive_question_n_inflated"] is False,
            f"n_clusters={joint_boot['n_clusters']} naive_inflated={joint_boot['naive_question_n_inflated']}",
        ),
        _gate(
            "ek5_incremental_equals_full_every_prefix",
            replay["ek5_prefix_parity_rate"] == 1.0
            and replay["ek5_prefix_mismatches"] == 0
            and replay["total_prefixes_checked"] > 0,
            f"exact={replay['exact_parity_prefixes']}/{replay['total_prefixes_checked']}",
        ),
        _gate(
            "stateful_checkpoint_replay_exact",
            replay["checkpoint_replay_all_exact"] is True,
            "all 20x2 conversation branches restored from checkpoint with exact EK5 digest match",
        ),
        _gate(
            "paired_withdrawal_beats_recency_baseline",
            joint_boot["mean"] == 1.0 and vs_recency["lcb_95"] > 0.0,
            f"joint_acc={joint_boot['mean']:.4f} ek5_minus_recency_lcb95={vs_recency['lcb_95']:.4f}",
        ),
        _gate(
            "sql_challenger_mode_parity_exact",
            replay["sql_challenger_mode_parity_exact"] is True,
            "recursive_cte == closure_table across all 400x2 paired queries",
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
        "ingestion": ingest_receipt,
        "replay": replay,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, default="local-drycheck")
    parser.add_argument("--beam-source-json", type=Path, default=None)
    parser.add_argument("--historical-fingerprints-json", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    report = execute(
        mode=args.mode,
        beam_source_json=args.beam_source_json,
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
