"""Build D3 frozen regimes, calibrated estimator artifacts, and validation."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import numpy as np

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from research.drift_recovery.artifacts import EmbeddingPack  # noqa: E402
from research.drift_recovery.estimator.calibration import fit_calibration  # noqa: E402
from research.drift_recovery.estimator.features import (  # noqa: E402
    extract_regime_features,
    fit_ridge_for_pack,
    fit_ridge_map,
)
from research.drift_recovery.estimator.validation import leave_one_regime_out  # noqa: E402
from research.drift_recovery.harness_bridge import (  # noqa: E402
    aggregate_ndcg,
    assert_harness_parity,
    encode_retrieval_case,
    load_retrieval_evalsplit,
    per_query_ndcg,
)
from research.drift_recovery.stats.paired_bootstrap import (  # noqa: E402
    BootstrapDraws,
    paired_query_bootstrap,
    strip_draws,
)


OLD_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
MPNET_MODEL = "all-mpnet-base-v2"
BGE_MODEL = "BAAI/bge-large-en-v1.5"
E5_MODEL = "intfloat/e5-base-v2"
DEFAULT_OUTPUT_DIR = Path("research/drift_recovery/out/estimator")
DEFAULT_D1_PACK = Path("research/drift_recovery/out/d1/scifact_evalsplit_pack")


@dataclass(frozen=True)
class RegimeSpec:
    name: str
    dataset: str
    swap_model: str
    anchor_fraction: float = 0.4
    seed: int = 42
    existing_pack: Optional[Path] = None


class RegimeUnavailableError(RuntimeError):
    """An explicitly skippable offline dataset/model-cache miss."""


REGIMES = (
    RegimeSpec(
        name="scifact_minilm_to_mpnet",
        dataset="SciFact",
        swap_model=MPNET_MODEL,
        existing_pack=DEFAULT_D1_PACK,
    ),
    RegimeSpec(
        name="nfcorpus_minilm_to_mpnet",
        dataset="NFCorpus",
        swap_model=MPNET_MODEL,
    ),
    RegimeSpec(
        name="scifact_minilm_to_bge_large",
        dataset="SciFact",
        swap_model=BGE_MODEL,
    ),
    RegimeSpec(
        name="nfcorpus_minilm_to_bge_large",
        dataset="NFCorpus",
        swap_model=BGE_MODEL,
    ),
)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def _safe_fit_index(
    doc_ids: np.ndarray,
    eval_qrels: Mapping[str, Mapping[str, float]],
    seed: int,
) -> np.ndarray:
    positives = {
        str(doc_id)
        for relevance in eval_qrels.values()
        for doc_id, score in relevance.items()
        if float(score) > 0.0
    }
    order = np.random.default_rng(int(seed)).permutation(len(doc_ids))
    safe_order = [index for index in order if str(doc_ids[index]) not in positives]
    # SciFact supports the D1 half-corpus count.  NFCorpus has much denser
    # relevance judgments, so its honest safe pool can be smaller.  Ridge is
    # regularized and remains well-defined below the embedding dimension; the
    # exact resulting n_anchor is a recorded regime axis rather than padded
    # with eval-positive pairs.
    fit_count = min(len(doc_ids) // 2, len(safe_order))
    if fit_count < 2:
        raise RuntimeError(
            f"only {fit_count} non-eval-positive documents remain for leakage-safe ridge fit"
        )
    selected = safe_order[:fit_count]
    return np.asarray(selected, dtype=np.int64)


def _recovery(
    floor_scores: Sequence[float],
    oracle_scores: Sequence[float],
    ridge_scores: Sequence[float],
) -> Dict[str, float]:
    floor = aggregate_ndcg(floor_scores)
    oracle = aggregate_ndcg(oracle_scores)
    ridge = aggregate_ndcg(ridge_scores)
    gap = oracle - floor
    if gap <= 1e-12:
        raise RuntimeError(f"oracle gap is non-positive ({gap:.6g})")
    return {
        "floor_ndcg": floor,
        "oracle_ndcg": oracle,
        "ridge_ndcg": ridge,
        "oracle_gap": gap,
        "ridge_recovery": (ridge - floor) / gap,
    }


def build_regime_pack(spec: RegimeSpec, prefix: Path, device: str) -> EmbeddingPack:
    """Build one additive D3 pack with a leakage-safe ridge headline."""

    try:
        case = load_retrieval_evalsplit(
            spec.dataset,
            seed=spec.seed,
            anchor_fraction=spec.anchor_fraction,
            max_queries=100,
            sample_docs=1200,
        )
    except RuntimeError as exc:
        if "offline" in str(exc).lower() and "cache is unavailable" in str(exc).lower():
            raise RegimeUnavailableError(str(exc)) from exc
        raise
    try:
        encoded = encode_retrieval_case(
            case,
            swap_model=spec.swap_model,
            seed=spec.seed,
            device=device,
            old_model=OLD_MODEL,
            batch_size=64,
        )
    except (FileNotFoundError, OSError) as exc:
        raise RegimeUnavailableError(
            f"offline model cache unavailable for {spec.swap_model}: {exc}"
        ) from exc
    doc_ids = np.asarray(case["doc_ids"], dtype=str)
    query_ids = np.asarray(case["eval_query_ids"], dtype=str)
    fit_idx = _safe_fit_index(doc_ids, case["eval_qrels"], spec.seed)
    ridge_map = fit_ridge_map(encoded["Do"], encoded["Dor"], fit_idx)
    ridge_documents = np.asarray(encoded["Do"], dtype=np.float64) @ ridge_map
    score_args = {
        "query_vectors": encoded["Qd"],
        "query_ids": query_ids,
        "doc_ids": doc_ids,
        "qrels": case["eval_qrels"],
        "k": 10,
    }
    floor_scores = per_query_ndcg(doc_vectors=encoded["Do"], **score_args)
    oracle_scores = per_query_ndcg(doc_vectors=encoded["Dor"], **score_args)
    ridge_scores = per_query_ndcg(doc_vectors=ridge_documents, **score_args)
    recovery = _recovery(floor_scores, oracle_scores, ridge_scores)
    aggregate = {
        "floor": recovery["floor_ndcg"],
        "oracle": recovery["oracle_ndcg"],
        "ridge": recovery["ridge_ndcg"],
    }
    pack = EmbeddingPack(
        Do=encoded["Do"],
        Dor=encoded["Dor"],
        Qd=encoded["Qd"],
        doc_ids=doc_ids,
        query_ids=query_ids,
        qrels=case["eval_qrels"],
        fit_idx=fit_idx,
        per_query_scores={
            "floor": floor_scores,
            "oracle": oracle_scores,
            "ridge": ridge_scores,
        },
        extra_arrays={
            "leakage_safe_fit_idx": fit_idx.copy(),
            "method_documents__ridge": ridge_documents,
        },
        metadata={
            "dataset": spec.dataset,
            "seed": spec.seed,
            "max_queries": 100,
            "sample_docs": 1200,
            "anchor_fraction": spec.anchor_fraction,
            "k": 10,
            "ndcg": "binary merged-harness ndcg_at_k",
            "models": {"old": encoded["old_model"], "swap": encoded["swap_model"]},
            "drift_manifest": encoded["drift_manifest"],
            "harness_aggregate_ndcg": aggregate,
            "recovery": {"ridge": recovery["ridge_recovery"]},
            "fit_contract": {
                "index": "fit_idx and extra_arrays.leakage_safe_fit_idx",
                "fit_count": int(len(fit_idx)),
                "eval_positive_docs_in_fit": 0,
                "regularization": 1.0,
                "eval_qrels_used_by_optimizer": False,
            },
        },
    )
    saved = pack.save(prefix)
    parity = assert_harness_parity(pack, atol=1e-12)
    if max(parity.values()) > 1e-12:
        raise AssertionError(f"harness parity exceeded tolerance: {parity}")
    print(f"[D3] froze {spec.name}: {saved['sha256']}", flush=True)
    return pack


def frozen_ridge_ground_truth(pack: EmbeddingPack) -> Dict[str, Any]:
    """Use the pack's frozen D1 ridge scores as the validation target.

    The estimator feature fit remains leakage-safe.  For the legacy D1 pack,
    the separate safe refit below is diagnostic only because the locked target
    contract calls for the pack's already-measured ridge recovery.
    """

    frozen_ridge_scores = np.asarray(pack.per_query_scores["ridge"], dtype=np.float64)
    metrics = _recovery(
        pack.per_query_scores["floor"],
        pack.per_query_scores["oracle"],
        frozen_ridge_scores,
    )
    fit = fit_ridge_for_pack(pack)
    safe_ridge_scores = per_query_ndcg(
        pack.Qd,
        pack.query_ids,
        fit.corrected_documents,
        pack.doc_ids,
        pack.qrels,
        k=int(pack.metadata.get("k", 10)),
    )
    safe_metrics = _recovery(
        pack.per_query_scores["floor"],
        pack.per_query_scores["oracle"],
        safe_ridge_scores,
    )
    try:
        draws = BootstrapDraws.create(len(pack.query_ids), draws=2000, seed=20260709)
        bootstrap = strip_draws(
            paired_query_bootstrap(
                {
                    "floor": pack.per_query_scores["floor"],
                    "oracle": pack.per_query_scores["oracle"],
                    "ridge": frozen_ridge_scores,
                },
                methods=("ridge",),
                bootstrap_draws=draws,
            )
        )
        bootstrap_error = None
    except (RuntimeError, ValueError) as exc:
        bootstrap = None
        bootstrap_error = f"{type(exc).__name__}: {exc}"
    return {
        **metrics,
        "ground_truth_source": "frozen pack per_query_scores.ridge",
        "ridge_per_query_scores": frozen_ridge_scores.tolist(),
        "paired_query_bootstrap": bootstrap,
        "bootstrap_error": bootstrap_error,
        "leakage_safe_feature_refit": {
            **safe_metrics,
            "fit_count": int(len(fit.fit_idx)),
            "eval_positive_docs_in_fit": 0,
        },
        "frozen_fit_audit": (
            pack.metadata.get("fit_contract")
            or pack.metadata.get("waypoint_audit")
            or {"status": "not recorded"}
        ),
    }


def _pack_digest(prefix: Path) -> str:
    return hashlib.sha256(prefix.with_suffix(".npz").read_bytes()).hexdigest()


def _render_validation(validation: Mapping[str, Any], manifest: Mapping[str, Any]) -> str:
    rho = validation["spearman"]
    accuracy = validation["bin_accuracy"]
    rho_text = "undefined" if rho is None else f"{rho:.3f}"
    accuracy_text = "undefined" if accuracy is None else f"{100.0 * accuracy:.1f}%"
    available_items = [
        item for item in manifest["regimes"] if item["status"] == "available"
    ]
    if available_items and all(
        item.get("predicted_inversion_rate") == 1.0 for item in available_items
    ):
        minimum_recovery = min(item["frozen_ridge_recovery"] for item in available_items)
        maximum_recovery = max(item["frozen_ridge_recovery"] for item in available_items)
        failure_note = (
            "Smallest concrete failure: the bound predicted a violation for **100% of "
            "queries in every available regime**, so its inversion-rate feature did not "
            f"separate observed recoveries ranging from {minimum_recovery:.3f} to "
            f"{maximum_recovery:.3f}."
        )
    else:
        failure_note = (
            "Smallest concrete failure: the held-out feature map misordered the regimes."
        )
    lines = [
        "# D3 recoverability estimator validation",
        "",
        "Held-out axis: **regime** (dataset / encoder family / anchor setting). "
        "No regime's own measured recovery trains its held-out prediction.",
        "",
        "| Held-out regime | Actual R | R_hat | Lower band | Actual bin | Predicted bin |",
        "|---|---:|---:|---:|---|---|",
    ]
    for row in validation["rows"]:
        estimate = "n/a" if row["R_hat"] is None else f"{row['R_hat']:.4f}"
        lower = "n/a" if row["lower_band"] is None else f"{row['lower_band']:.4f}"
        lines.append(
            f"| {row['regime']} | {row['actual_recovery']:.4f} | {estimate} | {lower} | "
            f"{row['actual_bin']} | {row['predicted_bin'] or 'n/a'} |"
        )
    lines.extend(
        [
            "",
            f"- Held-out regimes: **{validation['held_out_count']}**",
            f"- Spearman(R_hat, R): **{rho_text}**",
            f"- Three-way bin accuracy: **{accuracy_text}** "
            "(fixed bins: low <1/3, medium <2/3, high >=2/3)",
            f"- Verdict: **{validation['verdict']}**",
            "",
            validation["scoping_note"],
            "",
            failure_note,
            "",
            "Negative/scoping boundary: the per-query inequality only flags absence of an "
            "order-preservation guarantee for one oracle-best relevant/non-relevant pair. It "
            "does not prove an inversion, does not model the full top-k ordering, and Gram/margin "
            "distortion alone does not determine NDCG@10.",
            "",
            "Legacy D1 label note: `scifact_minilm_to_mpnet` uses the locked frozen-pack "
            "ridge target R=0.8434. Its estimator feature map is fitted only on "
            "`extra_arrays[\"leakage_safe_fit_idx\"]` (safe-refit diagnostic R=0.7891), "
            "because the legacy literal D1 target fit contains 28 eval-positive documents.",
            "",
            "## Regime availability",
            "",
        ]
    )
    for item in manifest["regimes"]:
        detail = item.get("reason") or item.get("pack_prefix") or ""
        lines.append(f"- `{item['name']}`: **{item['status']}** — {detail}")
    return "\n".join(lines) + "\n"


def _render_d3_report(validation: Mapping[str, Any], available_items: Sequence[Mapping[str, Any]]) -> str:
    rho = validation["spearman"]
    accuracy = validation["bin_accuracy"]
    rho_text = "undefined" if rho is None else f"{rho:.3f}"
    accuracy_text = "undefined" if accuracy is None else f"{100.0 * accuracy:.1f}%"
    available_names = [str(item["name"]) for item in available_items]
    inversion_saturated = bool(available_items) and all(
        item.get("predicted_inversion_rate") == 1.0 for item in available_items
    )
    if inversion_saturated:
        minimum_recovery = min(item["frozen_ridge_recovery"] for item in available_items)
        maximum_recovery = max(item["frozen_ridge_recovery"] for item in available_items)
        limitation = (
            "Most important limitation: the sufficient bound saturated at a 100% predicted "
            f"violation rate in all {len(available_items)} regimes despite measured recovery "
            f"spanning {minimum_recovery:.2f}-{maximum_recovery:.2f}, so the core "
            "inversion-rate signal did not discriminate regimes."
        )
    else:
        limitation = (
            "Most important limitation: this bound tracks one relevant/competitor order and "
            "cannot certify the full top-k NDCG ordering."
        )
    return "\n".join(
        [
            "# D3 report — recoverability estimator",
            "",
            "Built a pure-NumPy feature path over frozen `EmbeddingPack` data: oracle-space "
            "margin quantiles, leakage-safe ridge correction-error norms, and the per-query "
            "margin-bound violation rate. A strongly regularized linear map produces `R_hat`; "
            "training-only cross-fitted overprediction residuals produce its lower band.",
            "",
            f"Regimes used ({len(available_names)}): " + ", ".join(f"`{name}`" for name in available_names) + ".",
            "",
            f"Leave-one-regime-out result: Spearman **{rho_text}**, three-way bin accuracy "
            f"**{accuracy_text}**. Verdict: **{validation['verdict']}**.",
            "",
            validation["scoping_note"],
            "",
            limitation,
            "",
        ]
    )


def run_d3(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    device: str = "cuda",
    rebuild_packs: bool = False,
    only_existing: bool = False,
) -> Dict[str, Any]:
    output_dir = Path(output_dir)
    packs_dir = output_dir / "packs"
    packs_dir.mkdir(parents=True, exist_ok=True)
    manifest: Dict[str, Any] = {
        "record_type": "d3_estimator_regime_manifest",
        "environment": {
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", ""),
            "HF_DATASETS_OFFLINE": os.environ.get("HF_DATASETS_OFFLINE", ""),
            "device": device,
        },
        "parity_tolerance": 1e-12,
        "regimes": [],
    }
    validation_regimes = []
    for spec in REGIMES:
        if only_existing and spec.existing_pack is None:
            manifest["regimes"].append(
                {"name": spec.name, "status": "skipped", "reason": "--only-existing requested"}
            )
            continue
        prefix = Path(spec.existing_pack) if spec.existing_pack else packs_dir / f"{spec.name}_pack"
        try:
            if prefix.with_suffix(".npz").exists() and not rebuild_packs:
                print(f"[D3] reloading {spec.name}", flush=True)
                pack = EmbeddingPack.load(prefix)
            elif spec.existing_pack is not None:
                raise FileNotFoundError(f"required frozen D1 pack not found: {prefix}")
            else:
                print(f"[D3] building {spec.name} sequentially", flush=True)
                pack = build_regime_pack(spec, prefix, device=device)
        except RegimeUnavailableError as exc:
            reason = f"{type(exc).__name__}: {exc}"
            print(f"[D3] skipped {spec.name}: {reason}", flush=True)
            manifest["regimes"].append(
                {
                    "name": spec.name,
                    "dataset": spec.dataset,
                    "swap_model": spec.swap_model,
                    "anchor_fraction": spec.anchor_fraction,
                    "status": "skipped",
                    "reason": reason,
                }
            )
            continue

        # From here on every failure is a build failure, never an availability
        # skip.  In particular, parity/checksum/leakage/feature assertions must
        # terminate the run rather than manufacture a scoping-negative result.
        parity = assert_harness_parity(pack, atol=1e-12)
        ground_truth = frozen_ridge_ground_truth(pack)
        features = extract_regime_features(pack)
        feature_payload = {
            **features,
            "regime_name": spec.name,
            "ground_truth": ground_truth,
            "pack_prefix": str(prefix),
            "harness_parity": parity,
        }
        _write_json(output_dir / f"{spec.name}_features.json", feature_payload)
        validation_regimes.append(
            {
                "name": spec.name,
                "features": features,
                "recovery": ground_truth["ridge_recovery"],
            }
        )
        manifest["regimes"].append(
            {
                "name": spec.name,
                "dataset": spec.dataset,
                "swap_model": spec.swap_model,
                "anchor_fraction": spec.anchor_fraction,
                "status": "available",
                "pack_prefix": str(prefix),
                "pack_sha256": _pack_digest(prefix),
                "harness_parity": parity,
                "frozen_ridge_recovery": ground_truth["ridge_recovery"],
                "leakage_safe_feature_refit_recovery": ground_truth[
                    "leakage_safe_feature_refit"
                ]["ridge_recovery"],
                "document_count": int(features["summary"]["document_count"]),
                "query_count": int(features["summary"]["query_count"]),
                "safe_fit_count": int(features["summary"]["fit_count"]),
                "predicted_inversion_rate": float(
                    features["summary"]["predicted_inversion_rate"]
                ),
            }
        )

    validation = leave_one_regime_out(validation_regimes)
    _write_json(output_dir / "validation.json", validation)
    if validation_regimes:
        calibrator = fit_calibration(
            [item["features"] for item in validation_regimes],
            [item["recovery"] for item in validation_regimes],
            regime_names=[item["name"] for item in validation_regimes],
        )
        _write_json(output_dir / "calibration.json", calibrator.to_dict())
    _write_json(output_dir / "regime_manifest.json", manifest)
    (output_dir / "estimator_validation.md").write_text(
        _render_validation(validation, manifest), encoding="utf-8"
    )
    available_names = [item["name"] for item in validation_regimes]
    available_manifest_items = [
        item for item in manifest["regimes"] if item["status"] == "available"
    ]
    (output_dir / "D3_REPORT.md").write_text(
        _render_d3_report(validation, available_manifest_items), encoding="utf-8"
    )
    return {
        "output_dir": str(output_dir),
        "available_regimes": available_names,
        "validation": validation,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the D3 recoverability estimator")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--rebuild-packs", action="store_true")
    parser.add_argument("--only-existing", action="store_true")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    result = run_d3(
        output_dir=args.output_dir,
        device=args.device,
        rebuild_packs=args.rebuild_packs,
        only_existing=args.only_existing,
    )
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
