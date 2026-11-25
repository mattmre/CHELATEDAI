"""Runner for Wave 18 / Bundle #54 (Item 6B): Zero-Retune 12-Domain, 5-Encoder/4-Hop, Shift & Poisoning Gauntlet.

Supports:
  - `--mode local-drycheck` (default, CPU-only, `CUDA_VISIBLE_DEVICES=""`): runs deterministic
    12-domain unburned gauntlet with `burn_ledger` + SHA-256 overlap checks before scoring, full
    5x3 factorial (`bridge_merge_harness.run_factorial_5x3`), 20 frozen 500-anchor draws/domain,
    prompt-wrapper/tenant-prefix shift + shifted-anchor refit ceiling, 5-encoder / 4-hop `v1->v5`
    + direct `v1->v5`, `1%, 5%, 10%, 20%` anchor-poisoning sweep, and all 3 pre-registered hypotheses.
  - `--mode run-live-gpu`: evaluates precomputed live 12-domain x 5-encoder NPZ/JSON embeddings
    against the exact same zero-retune gauntlet (failing closed if required live inputs are absent;
    never labels synthetic drycheck outputs as live).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from chelated_12domain_4hop_gauntlet import (  # noqa: E402
    BUNDLE_ID,
    ENCODER_NAMES,
    ITEM_ID,
    LOCKED_12_DOMAINS,
    POISONING_SWEEP_RATES,
    SPINE_DIGEST,
    DomainGauntletBundle,
    build_deterministic_12domain_gauntlet_fixture,
    run_12domain_4hop_gauntlet,
)
from chelated_pv7_spine.bridge_merge_harness import (  # noqa: E402
    BRIDGE_NAMES,
    ITEM6B_RETIREMENT_FLOOR,
    ITEM6B_TIGHT_EQUIVALENCE_FLOOR,
    MERGE_NAMES,
)
from chelated_pv7_spine.burn_ledger import build_burn_ledger, text_fingerprint  # noqa: E402
from chelated_pv7_spine.claim_registry import REGISTRY_PATH, claim_registry_sha256  # noqa: E402
from chelated_pv7_spine.cpu_io_cage import assert_zero_cuda_context, build_cpu_only_env  # noqa: E402

MODES = ("local-drycheck", "run-live-gpu")


def _gate(name: str, passed: bool, detail: str) -> Dict[str, Any]:
    return {"name": name, "passed": bool(passed), "detail": detail}


def _drycheck_historical_inventory() -> Dict[str, List[str]]:
    return {
        "documents": [text_fingerprint("pv7-b54-historical-baseline-doc-marker")],
        "queries": [text_fingerprint("pv7-b54-historical-baseline-query-marker")],
    }


def _load_live_domain_bundles(domains_root: Path) -> List[DomainGauntletBundle]:
    loaded: List[DomainGauntletBundle] = []
    for desc in LOCKED_12_DOMAINS:
        meta_path = domains_root / f"{desc.burn_check_id}.json"
        npz_path = domains_root / f"{desc.burn_check_id}.npz"
        if not meta_path.is_file() or not npz_path.is_file():
            raise FileNotFoundError(
                f"Missing live domain artifacts for {desc.domain_id}: expected {meta_path.name} and {npz_path.name}"
            )
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        arrays = np.load(npz_path)
        doc_seq = tuple(np.asarray(arrays[f"docs_v{i}"], dtype=np.float64) for i in range(1, 6))
        query_seq = tuple(np.asarray(arrays[f"queries_v{i}"], dtype=np.float64) for i in range(1, 6))
        loaded.append(
            DomainGauntletBundle(
                descriptor=desc,
                doc_ids=tuple(meta["doc_ids"]),
                query_ids=tuple(meta["query_ids"]),
                doc_texts=tuple(meta["doc_texts"]),
                query_texts=tuple(meta["query_texts"]),
                qrels=meta["qrels"],
                doc_embeddings_seq=doc_seq,
                query_embeddings_seq=query_seq,
                shifted_queries_v1=np.asarray(arrays["shifted_queries_v1"], dtype=np.float64),
                shifted_anchor_queries_v1=np.asarray(arrays["shifted_anchor_queries_v1"], dtype=np.float64),
                shifted_anchor_targets_v2=np.asarray(arrays["shifted_anchor_targets_v2"], dtype=np.float64),
                backfill_mask=np.asarray(arrays["backfill_mask"], dtype=bool),
                prefix_displacement_mean_norm=float(meta["prefix_displacement_mean_norm"]),
                prefix_nonlinear_residual_mean_norm=float(meta["prefix_nonlinear_residual_mean_norm"]),
                prefix_linear_shift_fraction=float(meta["prefix_linear_shift_fraction"]),
            )
        )
    return loaded


def _run_live_gpu(
    *,
    domains_root: Optional[Path] = None,
    historical_fingerprints_json: Optional[Path] = None,
) -> Dict[str, Any]:
    if domains_root is None or historical_fingerprints_json is None:
        return {
            "bundle_id": BUNDLE_ID,
            "item": ITEM_ID,
            "mode": "run-live-gpu",
            "status": "awaiting-live-inputs",
            "passed": False,
            "reason": (
                "Live 12-domain gauntlet requires --domains-root and "
                "--historical-fingerprints-json; refusing synthetic fallback."
            ),
            "locked_12_domains": [d.domain_id for d in LOCKED_12_DOMAINS],
            "encoder_chain": list(ENCODER_NAMES),
            "spine_digest": claim_registry_sha256(),
        }

    hist_fps = json.loads(Path(historical_fingerprints_json).read_text(encoding="utf-8"))
    bundles = _load_live_domain_bundles(Path(domains_root))
    ledger = build_burn_ledger(REPO_ROOT)
    gauntlet = run_12domain_4hop_gauntlet(
        bundles,
        historical_fingerprints=hist_fps,
        ledger=ledger,
        is_live_confirmatory=True,
    )
    return {
        "bundle_id": BUNDLE_ID,
        "item": ITEM_ID,
        "mode": "run-live-gpu",
        "status": gauntlet["status"],
        "passed": gauntlet["hypotheses"]["overall_verdict"] == "verified-pass",
        "gauntlet": gauntlet,
        "spine_digest": claim_registry_sha256(),
    }


def execute(
    mode: str = "local-drycheck",
    *,
    domains_root: Optional[Path] = None,
    historical_fingerprints_json: Optional[Path] = None,
) -> Dict[str, Any]:
    """Execute Bundle #54 (`Item 6B`) in `local-drycheck` (CPU) or `run-live-gpu` mode."""
    if mode not in MODES:
        raise ValueError(f"Unsupported mode {mode!r}; expected one of {MODES}")
    if mode == "run-live-gpu":
        return _run_live_gpu(
            domains_root=domains_root,
            historical_fingerprints_json=historical_fingerprints_json,
        )

    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    cpu_env = build_cpu_only_env(os.environ)
    cuda_receipt = assert_zero_cuda_context(env=cpu_env, strict=True)
    registry_before = REGISTRY_PATH.read_bytes()

    ledger = build_burn_ledger(REPO_ROOT)
    bundles = build_deterministic_12domain_gauntlet_fixture(
        seed=5400,
        n_docs_per_domain=120,
        n_queries_per_domain=36,
        encoder_dims=(20, 20, 24, 28, 36),
    )
    gauntlet = run_12domain_4hop_gauntlet(
        bundles,
        historical_fingerprints=_drycheck_historical_inventory(),
        ledger=ledger,
        n_anchor_draws=20,
        anchors_per_draw=500,
        poisoning_rates=POISONING_SWEEP_RATES,
        n_bootstrap=2000,
        seed=5401,
        is_live_confirmatory=False,
    )
    registry_after = REGISTRY_PATH.read_bytes()
    actual_spine = claim_registry_sha256()

    hyps = gauntlet["hypotheses"]
    h_merge = hyps["H6B_merge"]
    h_static = hyps["H6B_bridge_static"]
    h_shift4 = hyps["H6B_bridge_shift_and_4hop"]
    expected_cells = len(BRIDGE_NAMES) * len(MERGE_NAMES)
    all_15_cells = all(
        len(dom_data["factorial_5x3"]) == expected_cells for dom_data in gauntlet["per_domain"].values()
    )
    all_4_hops = all(
        len(dom_data["four_hop_v1_to_v5"]["transitions_zeromu"]) == 4
        and abs(dom_data["four_hop_v1_to_v5"]["zeromu_final_nonzero_kappa"] - 1.0) < 1e-6
        for dom_data in gauntlet["per_domain"].values()
    )

    gates: List[Dict[str, Any]] = [
        _gate("zero_cuda_cpu_cage", cuda_receipt.passed, f"cuda_visible={cuda_receipt.cuda_visible_devices!r}"),
        _gate(
            "all_12_domains_unburned_and_decontaminated",
            gauntlet["burn_ledger_gate"]["all_12_unburned"] is True
            and gauntlet["burn_ledger_gate"]["n_domains_verified"] == 12
            and gauntlet["burn_ledger_gate"]["bioasq_nfcorpus_decontaminated"] is True,
            f"domains={gauntlet['burn_ledger_gate']['n_domains_verified']}",
        ),
        _gate(
            "full_5x3_factorial_per_domain_zero_retune",
            all_15_cells and gauntlet["retuned_on_eval"] is False,
            f"cells_per_domain={expected_cells} retuned={gauntlet['retuned_on_eval']}",
        ),
        _gate(
            "twenty_frozen_anchor_draws_variance_separated",
            gauntlet["anchor_draw_variance"]["n_draws_per_domain"] == 20
            and gauntlet["anchor_draw_variance"]["cross_domain_n"] == 12,
            f"draws=20 cross_domain_n={gauntlet['anchor_draw_variance']['cross_domain_n']}",
        ),
        _gate(
            "four_hop_v1_to_v5_polar_stiefel_exact",
            all_4_hops and len(gauntlet["encoder_chain"]) == 5,
            f"chain={'->'.join(ENCODER_NAMES)}",
        ),
        _gate(
            "poisoning_sweep_1_5_10_20_present",
            all(f"{r:.2f}" in gauntlet["poisoning_sweep"]["curve"] for r in (0.01, 0.05, 0.10, 0.20)),
            f"rates={gauntlet['poisoning_sweep']['rates']}",
        ),
        _gate(
            "h6b_merge_evaluated",
            h_merge["n_domains"] == 12 and "exact_binomial_p" in h_merge and "lcb_95" in h_merge,
            f"pos={h_merge['positive_domains']}/12 lcb95={h_merge['lcb_95']:.4f} p={h_merge['exact_binomial_p']:.4f}",
        ),
        _gate(
            "h6b_bridge_static_floors_and_centered_contrast",
            h_static["retirement_floor"] == ITEM6B_RETIREMENT_FLOOR
            and h_static["tight_equivalence_floor"] == ITEM6B_TIGHT_EQUIVALENCE_FLOOR
            and "zeromu_vs_centered_csls_contrast" in h_static
            and h_static["clears_retirement_floor"] is True,
            f"lcb95={h_static['zeromu_vs_uncentered_lcb_95']:.4f} vs_centered={h_static['zeromu_vs_centered_csls_contrast']['mean_delta']:.4f}",
        ),
        _gate(
            "h6b_bridge_shift_and_4hop_evaluated",
            h_shift4["prompt_shift"]["mean_prefix_displacement_norm"] is not None
            and h_shift4["four_hop_v1_to_v5"]["mean_direct_v1_v5_delta"] is not None
            and h_shift4["prompt_shift"]["lcb_95"] > 0.0
            and h_shift4["four_hop_v1_to_v5"]["lcb_95"] > 0.0,
            f"shift_lcb={h_shift4['prompt_shift']['lcb_95']:.4f} 4hop_lcb={h_shift4['four_hop_v1_to_v5']['lcb_95']:.4f}",
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
        "gauntlet": gauntlet,
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=MODES, default="local-drycheck")
    parser.add_argument("--domains-root", type=Path, default=None)
    parser.add_argument("--historical-fingerprints-json", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args(argv)

    report = execute(
        mode=args.mode,
        domains_root=args.domains_root,
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
