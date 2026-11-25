"""Bundle #63 (Item 5.1, T4 protocol-ready): Trial 1 Search-Operator Migration Kit & Offline Scorer.

Implements the buyer-owned offline evaluation toolkit for Partner Trial 1:
- Exact 5-arm comparison matrix:
    1. `UncenteredProcrustes + CalibratedZScore`
    2. `UncenteredProcrustes + CSLS`
    3. `ZeroMu + CalibratedZScore`
    4. `ZeroMu + CSLS`
    5. `PartnerIncumbentAlternative`
- Complete Migration Bill Worksheet (dual-index RAM, dual-index disk, backfill GPU
  hours, fallback compute hours, engineering hours, optional ZeroMu 4-byte sidecar).
- Pre-registered `+-0.010` `nDCG@10` + p95/p99 SLA retirement rule:
  Explicitly retires the `ZeroMu` bridge premium if `UncenteredProcrustes + CSLS`
  clears accuracy and SLA targets at `<=` total migration bill, while crediting `CSLS`.
- Offline row-level scorer exercised on 5 open `DEV` fixtures (`win`, `loss`, `tie`,
  `timeout`, `exact_fallback_refill`), strictly keeping timeouts in the denominator
  as `0.0` `nDCG@10` and sealing stage-1 deliverables strictly as `"status": "protocol-ready"`.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from benchmark_utils import canonicalize_id, ndcg_at_k, recall_at_k
from chelated_pv7_spine.bridge_merge_harness import (
    RequestSpanProfiler,
    paired_bootstrap_ci,
)


BUNDLE_ID = 63
TRACK_ID = "item_5_1_trial1_search_migration"
CLAIM_TIER = "T4"
STAGE1_STATUS = "protocol-ready"
EQUIVALENCE_BAND_NDCG10 = 0.010

ARM_UNCENTERED_ZSCORE = "UncenteredProcrustes+CalibratedZScore"
ARM_UNCENTERED_CSLS = "UncenteredProcrustes+CSLS"
ARM_ZEROMU_ZSCORE = "ZeroMu+CalibratedZScore"
ARM_ZEROMU_CSLS = "ZeroMu+CSLS"
ARM_PARTNER_INCUMBENT = "PartnerIncumbentAlternative"

MANDATORY_ARMS: Tuple[str, ...] = (
    ARM_UNCENTERED_ZSCORE,
    ARM_UNCENTERED_CSLS,
    ARM_ZEROMU_ZSCORE,
    ARM_ZEROMU_CSLS,
    ARM_PARTNER_INCUMBENT,
)

DEV_FIXTURE_SCENARIOS: Tuple[str, ...] = (
    "win",
    "loss",
    "tie",
    "timeout",
    "exact_fallback_refill",
)


def _sha256_json(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# 1. Migration Bill Worksheet (Full Operational Cost Accounting)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MigrationBillInputs:
    arm_id: str
    corpus_docs: int
    migration_window_hours: float
    dual_index_ram_gb: float
    dual_index_disk_gb: float
    backfill_gpu_hours: float
    fallback_cpu_hours: float
    engineering_hours: float
    zeromu_sidecar_bytes_per_doc: int = 0
    ram_gb_hour_usd: float = 0.012
    disk_gb_month_usd: float = 0.12
    gpu_hour_usd: float = 2.50
    cpu_hour_usd: float = 0.08
    engineering_hour_usd: float = 150.0


@dataclass(frozen=True)
class MigrationBillBreakdown:
    arm_id: str
    corpus_docs: int
    dual_index_ram_usd: float
    dual_index_disk_usd: float
    sidecar_ram_gb: float
    sidecar_ram_usd: float
    backfill_gpu_usd: float
    fallback_compute_usd: float
    engineering_usd: float
    total_bill_usd: float
    cost_per_million_docs_usd: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_migration_bill(inputs: MigrationBillInputs) -> MigrationBillBreakdown:
    """Compute itemized migration bill covering dual-index RAM/disk, GPU backfill, fallback, and engineering."""
    if inputs.corpus_docs <= 0:
        raise ValueError("corpus_docs must be positive")
    if inputs.migration_window_hours < 0.0:
        raise ValueError("migration_window_hours must be non-negative")

    ram_usd = float(inputs.dual_index_ram_gb) * float(inputs.ram_gb_hour_usd) * float(inputs.migration_window_hours)
    disk_months = float(inputs.migration_window_hours) / 720.0
    disk_usd = float(inputs.dual_index_disk_gb) * float(inputs.disk_gb_month_usd) * disk_months
    sidecar_gb = (float(inputs.corpus_docs) * float(inputs.zeromu_sidecar_bytes_per_doc)) / float(1024 ** 3)
    sidecar_usd = sidecar_gb * float(inputs.ram_gb_hour_usd) * float(inputs.migration_window_hours)
    gpu_usd = float(inputs.backfill_gpu_hours) * float(inputs.gpu_hour_usd)
    fallback_usd = float(inputs.fallback_cpu_hours) * float(inputs.cpu_hour_usd)
    eng_usd = float(inputs.engineering_hours) * float(inputs.engineering_hour_usd)
    total = ram_usd + disk_usd + sidecar_usd + gpu_usd + fallback_usd + eng_usd
    per_m = total / (float(inputs.corpus_docs) / 1_000_000.0)
    return MigrationBillBreakdown(
        arm_id=inputs.arm_id,
        corpus_docs=int(inputs.corpus_docs),
        dual_index_ram_usd=round(ram_usd, 4),
        dual_index_disk_usd=round(disk_usd, 4),
        sidecar_ram_gb=round(sidecar_gb, 6),
        sidecar_ram_usd=round(sidecar_usd, 4),
        backfill_gpu_usd=round(gpu_usd, 4),
        fallback_compute_usd=round(fallback_usd, 4),
        engineering_usd=round(eng_usd, 4),
        total_bill_usd=round(total, 4),
        cost_per_million_docs_usd=round(per_m, 4),
    )


# ---------------------------------------------------------------------------
# 2. Row-Level Offline Scorer & Pre-Registered Retirement Rule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Trial1RowRecord:
    query_id: str
    arm_id: str
    bridge_method: str
    merge_method: str
    ranked_doc_ids: Tuple[str, ...]
    qrels: Mapping[str, float]
    queue_us: float
    backend_knn_us: float
    prefilter_visited: int
    short_refill_count: int
    exact_fallback_triggered: bool
    gemv_us: float
    merge_us: float
    total_us: float
    timed_out: bool

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "Trial1RowRecord":
        return cls(
            query_id=canonicalize_id(raw["query_id"]),
            arm_id=str(raw["arm_id"]),
            bridge_method=str(raw["bridge_method"]),
            merge_method=str(raw["merge_method"]),
            ranked_doc_ids=tuple(canonicalize_id(d) for d in raw.get("ranked_doc_ids", ())),
            qrels={canonicalize_id(k): float(v) for k, v in dict(raw.get("qrels", {})).items()},
            queue_us=float(raw.get("queue_us", 0.0)),
            backend_knn_us=float(raw.get("backend_knn_us", 0.0)),
            prefilter_visited=int(raw.get("prefilter_visited", 0)),
            short_refill_count=int(raw.get("short_refill_count", 0)),
            exact_fallback_triggered=bool(raw.get("exact_fallback_triggered", False)),
            gemv_us=float(raw.get("gemv_us", 0.0)),
            merge_us=float(raw.get("merge_us", 0.0)),
            total_us=float(raw.get("total_us", 0.0)),
            timed_out=bool(raw.get("timed_out", False)),
        )


@dataclass(frozen=True)
class Trial1SLATargets:
    equivalence_band_ndcg10: float = EQUIVALENCE_BAND_NDCG10
    p95_latency_ms: float = 35.0
    p99_latency_ms: float = 75.0
    max_timeout_rate: float = 0.005


@dataclass(frozen=True)
class ArmEvaluationSummary:
    arm_id: str
    n_queries: int
    mean_ndcg_at_10: float
    mean_recall_at_10: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    timeout_count: int
    timeout_rate: float
    exact_fallback_count: int
    exact_fallback_rate: float
    total_short_refills: int
    delta_ndcg_vs_incumbent: float
    lcb95_delta_vs_incumbent: float
    ucb95_delta_vs_incumbent: float
    clears_ndcg_equivalence: bool
    clears_sla_targets: bool
    clears_all_targets: bool
    total_bill_usd: float
    bill_breakdown: Dict[str, Any]
    span_summary: Dict[str, Any]


def score_trial1_rows(
    rows: Sequence[Trial1RowRecord],
    bills: Mapping[str, MigrationBillInputs],
    *,
    sla: Trial1SLATargets = Trial1SLATargets(),
    incumbent_arm: str = ARM_PARTNER_INCUMBENT,
    fixture_label: str = "partner_offline_run",
    is_live_buyer_run: bool = False,
    k: int = 10,
    n_bootstrap: int = 1000,
    seed: int = 6301,
) -> Dict[str, Any]:
    """Score offline row-level logs across arms and evaluate the pre-registered retirement rule.

    Timeouts remain strictly in the denominator as `0.0` `nDCG@10` and `0.0` `Recall@10`.
    """
    if not rows:
        raise ValueError("rows must be non-empty")

    seen_queries: Set[Tuple[str, str]] = set()
    by_arm: Dict[str, List[Trial1RowRecord]] = {}
    for row in rows:
        key = (row.arm_id, row.query_id)
        if key in seen_queries:
            raise ValueError(f"Duplicate query_id {row.query_id!r} in arm {row.arm_id!r}")
        seen_queries.add(key)
        by_arm.setdefault(row.arm_id, []).append(row)

    missing_arms = [arm for arm in MANDATORY_ARMS if arm not in by_arm]
    if missing_arms:
        raise ValueError(f"Missing mandatory arms: {missing_arms}")
    if incumbent_arm not in by_arm:
        raise ValueError(f"Incumbent arm {incumbent_arm!r} missing from rows")

    per_arm_ndcg: Dict[str, Dict[str, float]] = {}
    per_arm_recall: Dict[str, Dict[str, float]] = {}
    per_arm_profilers: Dict[str, RequestSpanProfiler] = {}

    for arm_id, arm_rows in by_arm.items():
        profiler = RequestSpanProfiler()
        ndcg_scores: Dict[str, float] = {}
        recall_scores: Dict[str, float] = {}
        for row in arm_rows:
            if row.timed_out:
                q_ndcg = 0.0
                q_rec = 0.0
            else:
                q_ndcg = float(ndcg_at_k(row.ranked_doc_ids, row.qrels, k=k, binary=False))
                q_rec = float(recall_at_k(row.ranked_doc_ids, row.qrels, k=k))
            ndcg_scores[row.query_id] = q_ndcg
            recall_scores[row.query_id] = q_rec
            profiler.record_span(
                request_id=f"{arm_id}:{row.query_id}",
                queue_us=row.queue_us,
                backend_knn_us=row.backend_knn_us,
                prefilter_visited=row.prefilter_visited,
                short_refill_count=row.short_refill_count,
                exact_fallback_triggered=row.exact_fallback_triggered,
                gemv_us=row.gemv_us,
                merge_us=row.merge_us,
                total_us=row.total_us,
                timed_out=row.timed_out,
                ndcg_at_10=q_ndcg,
            )
        per_arm_ndcg[arm_id] = ndcg_scores
        per_arm_recall[arm_id] = recall_scores
        per_arm_profilers[arm_id] = profiler

    inc_ndcg_map = per_arm_ndcg[incumbent_arm]
    common_qids = sorted(inc_ndcg_map.keys())
    if not common_qids:
        raise ValueError("Incumbent arm has zero scored queries")

    arm_summaries: Dict[str, ArmEvaluationSummary] = {}
    for arm_id in sorted(by_arm.keys()):
        if arm_id not in bills:
            raise ValueError(f"Migration bill missing for arm {arm_id!r}")
        bill = compute_migration_bill(bills[arm_id])
        span = per_arm_profilers[arm_id].summarize()
        arm_ndcg_map = per_arm_ndcg[arm_id]
        arm_rec_map = per_arm_recall[arm_id]
        if sorted(arm_ndcg_map.keys()) != common_qids:
            raise ValueError(f"Arm {arm_id!r} query_id set does not match incumbent query_id set")

        paired_deltas = [arm_ndcg_map[q] - inc_ndcg_map[q] for q in common_qids]
        mean_delta, lcb, ucb = paired_bootstrap_ci(paired_deltas, n_resamples=n_bootstrap, alpha=0.05, seed=seed)
        p50_ms = span["p50_total_us"] / 1_000.0
        p95_ms = span["p95_total_us"] / 1_000.0
        p99_ms = span["p99_total_us"] / 1_000.0

        clears_ndcg = bool(mean_delta >= -float(sla.equivalence_band_ndcg10) and lcb >= -float(sla.equivalence_band_ndcg10))
        clears_sla = bool(
            p95_ms <= float(sla.p95_latency_ms)
            and p99_ms <= float(sla.p99_latency_ms)
            and span["timeout_rate"] <= float(sla.max_timeout_rate)
        )
        total_refills = sum(r.short_refill_count for r in by_arm[arm_id])

        arm_summaries[arm_id] = ArmEvaluationSummary(
            arm_id=arm_id,
            n_queries=len(common_qids),
            mean_ndcg_at_10=span["mean_ndcg_at_10_including_timeouts"],
            mean_recall_at_10=float(np.mean([arm_rec_map[q] for q in common_qids])),
            p50_latency_ms=p50_ms,
            p95_latency_ms=p95_ms,
            p99_latency_ms=p99_ms,
            timeout_count=int(span["timeout_count"]),
            timeout_rate=float(span["timeout_rate"]),
            exact_fallback_count=int(span["exact_fallback_count"]),
            exact_fallback_rate=float(span["exact_fallback_rate"]),
            total_short_refills=int(total_refills),
            delta_ndcg_vs_incumbent=mean_delta,
            lcb95_delta_vs_incumbent=lcb,
            ucb95_delta_vs_incumbent=ucb,
            clears_ndcg_equivalence=clears_ndcg,
            clears_sla_targets=clears_sla,
            clears_all_targets=bool(clears_ndcg and clears_sla),
            total_bill_usd=bill.total_bill_usd,
            bill_breakdown=bill.to_dict(),
            span_summary=span,
        )

    pair_lcb = float("-inf")
    if ARM_ZEROMU_CSLS in per_arm_ndcg and ARM_UNCENTERED_CSLS in per_arm_ndcg:
        zm_map = per_arm_ndcg[ARM_ZEROMU_CSLS]
        unc_map = per_arm_ndcg[ARM_UNCENTERED_CSLS]
        pair_qids = sorted(set(zm_map) & set(unc_map))
        if pair_qids:
            pair_deltas = [zm_map[qid] - unc_map[qid] for qid in pair_qids]
            _, pair_lcb, _ = paired_bootstrap_ci(
                pair_deltas, n_resamples=n_bootstrap, alpha=0.05, seed=seed + 7
            )
    decision = _apply_retirement_rule(arm_summaries, sla=sla, pair_lcb=pair_lcb)
    report: Dict[str, Any] = {
        "bundle_id": BUNDLE_ID,
        "track_id": TRACK_ID,
        "claim_tier": CLAIM_TIER,
        "status": STAGE1_STATUS if not is_live_buyer_run else "buyer-run-scored",
        "fixture_label": fixture_label,
        "sla_targets": asdict(sla),
        "arms": {k_arm: asdict(v_arm) for k_arm, v_arm in arm_summaries.items()},
        "decision": decision,
    }
    report["report_sha256"] = _sha256_json(report)
    return report


def _apply_retirement_rule(
    arms: Mapping[str, ArmEvaluationSummary],
    *,
    sla: Trial1SLATargets,
    pair_lcb: float = float("-inf"),
) -> Dict[str, Any]:
    """Enforce the pre-registered `+-0.010` nDCG@10 + SLA retirement rule.

    Explicitly retires the `ZeroMu` bridge premium whenever
    `UncenteredProcrustes + CSLS` clears accuracy and SLA targets at `<=` total bill,
    while crediting `CSLS`.
    """
    unc_csls = arms.get(ARM_UNCENTERED_CSLS)
    zm_csls = arms.get(ARM_ZEROMU_CSLS)
    unc_zs = arms.get(ARM_UNCENTERED_ZSCORE)
    inc = arms.get(ARM_PARTNER_INCUMBENT)

    csls_gain_over_zscore = (
        (unc_csls.mean_ndcg_at_10 - unc_zs.mean_ndcg_at_10)
        if (unc_csls is not None and unc_zs is not None)
        else 0.0
    )
    zeromu_gain_over_unc_at_csls = (
        (zm_csls.mean_ndcg_at_10 - unc_csls.mean_ndcg_at_10)
        if (zm_csls is not None and unc_csls is not None)
        else 0.0
    )

    zeromu_retired = False
    credit_csls = bool(csls_gain_over_zscore > 0.0)
    recommended_arm = ARM_PARTNER_INCUMBENT
    outcome = "loss"
    rationale = "Neither Chelated migration arm cleared the pre-registered nDCG@10 equivalence band and SLA targets."

    if unc_csls is not None and zm_csls is not None:
        unc_bill_le_zm = unc_csls.total_bill_usd <= zm_csls.total_bill_usd
        if unc_csls.clears_all_targets and unc_bill_le_zm:
            if not zm_csls.clears_all_targets or zeromu_gain_over_unc_at_csls <= float(sla.equivalence_band_ndcg10):
                zeromu_retired = True
                recommended_arm = ARM_UNCENTERED_CSLS
                outcome = "tie_retire_zeromu_credit_csls"
                rationale = (
                    "UncenteredProcrustes+CSLS cleared both the +-0.010 nDCG@10 equivalence band "
                    "and p95/p99 SLA targets at <= total migration bill vs ZeroMu+CSLS; "
                    "ZeroMu bridge premium is explicitly retired while CSLS calibration is credited."
                )
            elif pair_lcb > float(sla.equivalence_band_ndcg10):
                recommended_arm = ARM_ZEROMU_CSLS
                outcome = "win_zeromu_csls"
                rationale = (
                    "ZeroMu+CSLS cleared all targets and its paired LCB versus "
                    "UncenteredProcrustes+CSLS exceeded +0.010."
                )
            else:
                zeromu_retired = True
                recommended_arm = ARM_UNCENTERED_CSLS
                outcome = "tie_retire_zeromu_credit_csls"
                rationale = (
                    "The mean nDCG gap exceeded the band, but the paired LCB did not clear +0.010. "
                    "The ZeroMu bridge premium stays retired."
                )
        elif (
            zm_csls.clears_all_targets
            and not unc_csls.clears_all_targets
            and pair_lcb > float(sla.equivalence_band_ndcg10)
        ):
            recommended_arm = ARM_ZEROMU_CSLS
            outcome = "win_zeromu_csls"
            rationale = (
                "ZeroMu+CSLS cleared both the +-0.010 nDCG@10 equivalence band and p95/p99 SLA "
                "targets, and its paired LCB versus UncenteredProcrustes+CSLS exceeded +0.010."
            )
        elif unc_csls.clears_all_targets:
            zeromu_retired = True
            recommended_arm = ARM_UNCENTERED_CSLS
            outcome = "tie_retire_zeromu_credit_csls"
            rationale = "UncenteredProcrustes+CSLS cleared all targets; ZeroMu bridge premium retired."
    elif (
        zm_csls is not None
        and zm_csls.clears_all_targets
        and pair_lcb > float(sla.equivalence_band_ndcg10)
    ):
        recommended_arm = ARM_ZEROMU_CSLS
        outcome = "win_zeromu_csls"
        rationale = "ZeroMu+CSLS cleared all accuracy and SLA targets with paired LCB above +0.010."

    inc_bill = inc.total_bill_usd if inc is not None else 0.0
    rec_bill = arms[recommended_arm].total_bill_usd if recommended_arm in arms else inc_bill
    return {
        "outcome": outcome,
        "recommended_arm": recommended_arm,
        "zeromu_bridge_premium_retired": zeromu_retired,
        "csls_credited": credit_csls,
        "csls_ndcg_gain_over_zscore": round(csls_gain_over_zscore, 6),
        "zeromu_ndcg_gain_over_uncentered_at_csls": round(zeromu_gain_over_unc_at_csls, 6),
        "zeromu_vs_uncentered_csls_lcb95": (
            None if not math.isfinite(pair_lcb) else round(float(pair_lcb), 6)
        ),
        "recommended_bill_usd": rec_bill,
        "incumbent_bill_usd": inc_bill,
        "estimated_bill_savings_usd": round(inc_bill - rec_bill, 4),
        "rationale": rationale,
    }


# ---------------------------------------------------------------------------
# 3. Open DEV Fixtures (`win`, `loss`, `tie`, `timeout`, `exact_fallback_refill`)
# ---------------------------------------------------------------------------


def _default_bills(corpus_docs: int = 5_000_000, *, fallback_hours_override: Optional[float] = None) -> Dict[str, MigrationBillInputs]:
    fb = 12.0 if fallback_hours_override is None else float(fallback_hours_override)
    return {
        ARM_UNCENTERED_ZSCORE: MigrationBillInputs(
            arm_id=ARM_UNCENTERED_ZSCORE,
            corpus_docs=corpus_docs,
            migration_window_hours=168.0,
            dual_index_ram_gb=64.0,
            dual_index_disk_gb=200.0,
            backfill_gpu_hours=18.0,
            fallback_cpu_hours=fb,
            engineering_hours=16.0,
            zeromu_sidecar_bytes_per_doc=0,
        ),
        ARM_UNCENTERED_CSLS: MigrationBillInputs(
            arm_id=ARM_UNCENTERED_CSLS,
            corpus_docs=corpus_docs,
            migration_window_hours=168.0,
            dual_index_ram_gb=64.0,
            dual_index_disk_gb=200.0,
            backfill_gpu_hours=18.0,
            fallback_cpu_hours=fb,
            engineering_hours=16.0,
            zeromu_sidecar_bytes_per_doc=0,
        ),
        ARM_ZEROMU_ZSCORE: MigrationBillInputs(
            arm_id=ARM_ZEROMU_ZSCORE,
            corpus_docs=corpus_docs,
            migration_window_hours=168.0,
            dual_index_ram_gb=64.0,
            dual_index_disk_gb=200.0,
            backfill_gpu_hours=18.0,
            fallback_cpu_hours=fb,
            engineering_hours=18.0,
            zeromu_sidecar_bytes_per_doc=4,
        ),
        ARM_ZEROMU_CSLS: MigrationBillInputs(
            arm_id=ARM_ZEROMU_CSLS,
            corpus_docs=corpus_docs,
            migration_window_hours=168.0,
            dual_index_ram_gb=64.0,
            dual_index_disk_gb=200.0,
            backfill_gpu_hours=18.0,
            fallback_cpu_hours=fb,
            engineering_hours=18.0,
            zeromu_sidecar_bytes_per_doc=4,
        ),
        ARM_PARTNER_INCUMBENT: MigrationBillInputs(
            arm_id=ARM_PARTNER_INCUMBENT,
            corpus_docs=corpus_docs,
            migration_window_hours=168.0,
            dual_index_ram_gb=128.0,
            dual_index_disk_gb=400.0,
            backfill_gpu_hours=240.0,
            fallback_cpu_hours=0.0,
            engineering_hours=40.0,
            zeromu_sidecar_bytes_per_doc=0,
        ),
    }


def _ranked_list_for_quality(qid: str, quality: str) -> Tuple[Tuple[str, ...], Dict[str, float]]:
    qrels = {f"{qid}_rel1": 2.0, f"{qid}_rel2": 1.0}
    noise = [f"{qid}_n{j}" for j in range(1, 11)]
    if quality == "perfect":
        return (f"{qid}_rel1", f"{qid}_rel2", *noise[:8]), qrels
    if quality == "strong":
        return (f"{qid}_rel1", noise[0], f"{qid}_rel2", *noise[1:8]), qrels
    if quality == "medium":
        return (noise[0], f"{qid}_rel1", noise[1], f"{qid}_rel2", *noise[2:8]), qrels
    if quality == "weak":
        return (*noise[:4], f"{qid}_rel1", *noise[4:9]), qrels
    return tuple(noise[:10]), qrels


def build_open_dev_fixture(scenario: str, *, n_queries: int = 24) -> Tuple[List[Trial1RowRecord], Dict[str, MigrationBillInputs]]:
    """Build deterministic open `DEV` fixture rows and migration bills for one scenario."""
    if scenario not in DEV_FIXTURE_SCENARIOS:
        raise ValueError(f"Unknown DEV fixture scenario={scenario!r}; expected one of {DEV_FIXTURE_SCENARIOS}")

    arm_meta = {
        ARM_UNCENTERED_ZSCORE: ("UncenteredProcrustes", "CalibratedZScoreMerge"),
        ARM_UNCENTERED_CSLS: ("UncenteredProcrustes", "CSLSMerge"),
        ARM_ZEROMU_ZSCORE: ("ZeroMuPolarBridge", "CalibratedZScoreMerge"),
        ARM_ZEROMU_CSLS: ("ZeroMuPolarBridge", "CSLSMerge"),
        ARM_PARTNER_INCUMBENT: ("PartnerIncumbent", "PartnerIncumbent"),
    }

    rows: List[Trial1RowRecord] = []
    for i in range(int(n_queries)):
        qid = f"dev_{scenario}_q{i:03d}"
        for arm_id in MANDATORY_ARMS:
            bridge_m, merge_m = arm_meta[arm_id]
            timed_out = False
            short_refills = 0
            exact_fb = False
            prefilter_visited = 128 + (i % 32)
            queue_us = 350.0 + (i % 7) * 20.0
            knn_us = 9_500.0 + (i % 11) * 250.0
            gemv_us = 650.0 if "ZeroMu" in arm_id else 420.0
            merge_us = 520.0 if "CSLS" in arm_id else 310.0

            if scenario == "win":
                q_map = {
                    ARM_PARTNER_INCUMBENT: "strong",
                    ARM_ZEROMU_CSLS: "perfect",
                    ARM_ZEROMU_ZSCORE: "medium",
                    ARM_UNCENTERED_CSLS: "medium",
                    ARM_UNCENTERED_ZSCORE: "weak",
                }
                quality = q_map[arm_id]
            elif scenario == "loss":
                q_map = {
                    ARM_PARTNER_INCUMBENT: "perfect",
                    ARM_ZEROMU_CSLS: "weak",
                    ARM_ZEROMU_ZSCORE: "weak",
                    ARM_UNCENTERED_CSLS: "weak",
                    ARM_UNCENTERED_ZSCORE: "miss",
                }
                quality = q_map[arm_id]
            elif scenario == "tie":
                q_map = {
                    ARM_PARTNER_INCUMBENT: "strong",
                    ARM_ZEROMU_CSLS: "strong",
                    ARM_UNCENTERED_CSLS: "strong",
                    ARM_ZEROMU_ZSCORE: "medium",
                    ARM_UNCENTERED_ZSCORE: "medium",
                }
                quality = q_map[arm_id]
            elif scenario == "timeout":
                q_map = {
                    ARM_PARTNER_INCUMBENT: "strong",
                    ARM_ZEROMU_CSLS: "strong",
                    ARM_UNCENTERED_CSLS: "strong",
                    ARM_ZEROMU_ZSCORE: "medium",
                    ARM_UNCENTERED_ZSCORE: "medium",
                }
                quality = q_map[arm_id]
                if arm_id in {ARM_ZEROMU_CSLS, ARM_UNCENTERED_CSLS} and i in (2, 7, 13):
                    timed_out = True
                    knn_us = 92_000.0
            else:  # exact_fallback_refill
                q_map = {
                    ARM_PARTNER_INCUMBENT: "strong",
                    ARM_ZEROMU_CSLS: "perfect",
                    ARM_UNCENTERED_CSLS: "medium",
                    ARM_ZEROMU_ZSCORE: "medium",
                    ARM_UNCENTERED_ZSCORE: "weak",
                }
                quality = q_map[arm_id]
                if arm_id != ARM_PARTNER_INCUMBENT and i % 4 == 0:
                    short_refills = 2
                    exact_fb = True
                    prefilter_visited = 2_048
                    knn_us = 22_000.0

            ranked, qrels = _ranked_list_for_quality(qid, quality)
            total_us = queue_us + knn_us + gemv_us + merge_us
            rows.append(
                Trial1RowRecord(
                    query_id=qid,
                    arm_id=arm_id,
                    bridge_method=bridge_m,
                    merge_method=merge_m,
                    ranked_doc_ids=ranked,
                    qrels=qrels,
                    queue_us=queue_us,
                    backend_knn_us=knn_us,
                    prefilter_visited=prefilter_visited,
                    short_refill_count=short_refills,
                    exact_fallback_triggered=exact_fb,
                    gemv_us=gemv_us,
                    merge_us=merge_us,
                    total_us=total_us,
                    timed_out=timed_out,
                )
            )

    fb_hours = 48.0 if scenario == "exact_fallback_refill" else 12.0
    return rows, _default_bills(fallback_hours_override=fb_hours)


def run_offline_dev_fixture_suite(*, n_queries: int = 24) -> Dict[str, Any]:
    """Exercise the offline row-level scorer across all 5 open `DEV` fixtures."""
    scenarios: Dict[str, Any] = {}
    for scenario in DEV_FIXTURE_SCENARIOS:
        rows, bills = build_open_dev_fixture(scenario, n_queries=n_queries)
        scenarios[scenario] = score_trial1_rows(
            rows,
            bills,
            fixture_label=f"open_dev_{scenario}",
            is_live_buyer_run=False,
        )
    suite = {
        "bundle_id": BUNDLE_ID,
        "track_id": TRACK_ID,
        "claim_tier": CLAIM_TIER,
        "status": STAGE1_STATUS,
        "fixture_split": "DEV_OPEN",
        "mandatory_arms": list(MANDATORY_ARMS),
        "scenarios": scenarios,
    }
    suite["suite_sha256"] = _sha256_json(suite)
    return suite


def load_rows_jsonl(path: Path) -> List[Trial1RowRecord]:
    rows: List[Trial1RowRecord] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        rows.append(Trial1RowRecord.from_dict(json.loads(stripped)))
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Bundle #63 Trial 1 Search-Operator Migration Offline Scorer")
    parser.add_argument("--run-dev-fixtures", action="store_true", help="Run all 5 open DEV fixtures")
    parser.add_argument("--rows-jsonl", type=Path, default=None, help="Path to partner row-level JSONL log")
    parser.add_argument("--bills-json", type=Path, default=None, help="Path to partner migration bills JSON")
    parser.add_argument(
        "--buyer-attestation-sha256",
        default="",
        help="SHA-256 of the --bills-json file bytes. Required for --rows-jsonl buyer runs.",
    )
    parser.add_argument("--output", type=Path, default=None, help="Optional output JSON report path")
    args = parser.parse_args(argv)

    if args.rows_jsonl is not None:
        if args.bills_json is None or not str(args.buyer_attestation_sha256).strip():
            print(
                "buyer runs require --bills-json and --buyer-attestation-sha256",
                file=sys.stderr,
            )
            return 1
        bill_bytes = args.bills_json.read_bytes()
        actual_attestation = hashlib.sha256(bill_bytes).hexdigest()
        if actual_attestation != str(args.buyer_attestation_sha256).strip().lower():
            print("buyer attestation sha256 does not match --bills-json", file=sys.stderr)
            return 1
        rows = load_rows_jsonl(args.rows_jsonl)
        raw_bills = json.loads(bill_bytes.decode("utf-8"))
        bills = {k: MigrationBillInputs(**v) for k, v in raw_bills.items()}
        payload = score_trial1_rows(rows, bills, fixture_label=args.rows_jsonl.name, is_live_buyer_run=True)
    else:
        payload = run_offline_dev_fixture_suite()

    rendered = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


__all__ = [
    "ARM_PARTNER_INCUMBENT",
    "ARM_UNCENTERED_CSLS",
    "ARM_UNCENTERED_ZSCORE",
    "ARM_ZEROMU_CSLS",
    "ARM_ZEROMU_ZSCORE",
    "ArmEvaluationSummary",
    "BUNDLE_ID",
    "CLAIM_TIER",
    "DEV_FIXTURE_SCENARIOS",
    "EQUIVALENCE_BAND_NDCG10",
    "MANDATORY_ARMS",
    "MigrationBillBreakdown",
    "MigrationBillInputs",
    "STAGE1_STATUS",
    "TRACK_ID",
    "Trial1RowRecord",
    "Trial1SLATargets",
    "build_open_dev_fixture",
    "compute_migration_bill",
    "load_rows_jsonl",
    "main",
    "run_offline_dev_fixture_suite",
    "score_trial1_rows",
]


if __name__ == "__main__":
    raise SystemExit(main())
