"""Pooled-Agent Loops Under Encoder Change & Withdrawal (Wave 20 / Bundle #62 / Item 4 / Demo D13).

Implements a genuine multi-worker task-queue coordinator (`pool_sizes = (1, 2, 4, 8)`)
evaluating 24 distinct multi-field legal/compliance production-matter templates across
4 deterministic arrival schedules and matched twin perturbation histories:

* Branch A (`A_derivative_only`):
    Withdrawn scheduling memo (`May 30, 2026` / matter-specific delivery date) + late
    pre-withdrawal in-flight worker summary + paraphrased copies -> the correct
    committed artifact marks `delivery_date = "UNRESOLVED"` while retaining the 3 valid
    unaffected fields (`output_format`, `Bates_prefix`, `delivery_channel`).
* Branch B (`B_independent_corroboration`):
    An independently authored signed notice also establishes the target `delivery_date`
    -> the correct committed artifact retains `delivery_date` **citing the independent
    signed notice** (rejecting citations to the withdrawn memo or its derivatives) while
    preserving the 3 valid unaffected fields.

Full 2x2 System Factorial x Event Cross + Mandatory Sham Control (`KEEP #2`):
* Retrieval factor: `{Conventional_Retriever, ZeroMu_Pinned_Bridge, Uncentered_CSLS}`
    importing `UncenteredProcrustes`, `ZeroMuPolarBridge`, `RawCosineMerge`, `CSLSMerge`,
    `RequestSpanProfiler`, and `paired_bootstrap_ci` from `chelated_pv7_spine.bridge_merge_harness`.
* Ledger factor: `{SQLiteChallengerLedger, EK5_v3}`
    importing `ChallengerBitemporalLedger` / `SQLiteChallengerBackend` from
    `chelated_pv7_spine.challenger_ledger`, `StratifiedEvidenceStore` from `ek5_v3_stratified`,
    and `export_canonical_ek5_audit_trace` / `compare_with_fresh_rebuild` from
    `chelated_pv7_spine.ek5_reason_canonicalizer`.
* Event regimes:
    `{no_change, encoder_change_only, withdrawal_only, both, sham_subtask_reopen_interrupt}`
    where `sham_subtask_reopen_interrupt` triggers an identical subtask reopen/checkpoint
    interrupt at the exact same tick without withdrawing source authority, isolating pure
    reopen/checkpoint timing overhead from EK5 lineage invalidation.
* Costed comparator:
    `full_restart_rebuild` (discards in-flight work and rebuilds index + ledger from scratch
    on encoder change or withdrawal, charging full rebuild + full 4-field re-execution).

Supports both:
* Stage 1 (`RecordedReplayWorkerBackend`): deterministic CPU controlled replay
  (`live_model_calls == 0`, zero GPU/network calls).
* Stage 2/3 (`LiveVLLMWorkerBackend`): live concurrent OpenAI-compatible vLLM worker pool
  (`ThreadPoolExecutor(max_workers=pool_size)`) with strict timeout-in-denominator accounting.
"""

from __future__ import annotations

import hashlib
import heapq
import json
import math
import time
import urllib.error
import urllib.request
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from chelated_pv7_spine.bridge_merge_harness import (
    CSLSMerge,
    RawCosineMerge,
    RequestSpanProfiler,
    UncenteredProcrustes,
    ZeroMuPolarBridge,
    l2_normalize,
    paired_bootstrap_ci,
)
from chelated_pv7_spine.burn_ledger import text_fingerprint
from chelated_pv7_spine.challenger_ledger import (
    STATUS_RESOLVED,
    ChallengerBitemporalLedger,
    LedgerEdgeEnvelope,
    LedgerRecordEnvelope,
    OpKind,
    SQLiteChallengerBackend,
)
from chelated_pv7_spine.ek5_reason_canonicalizer import (
    compare_with_fresh_rebuild,
    export_canonical_ek5_audit_trace,
)
from ek5_v3_stratified import (
    EdgeAttestation,
    StratifiedEvent,
    StratifiedEvidenceStore,
)

BUNDLE_ID = 62
POOL_SIZES: Tuple[int, ...] = (1, 2, 4, 8)
PRIMARY_POOL_SIZE = 4
ARRIVAL_SCHEDULES: Tuple[str, ...] = (
    "S1_steady",
    "S2_bursty_front",
    "S3_wave_bimodal",
    "S4_withdrawal_surge",
)
RETRIEVER_ARMS: Tuple[str, ...] = (
    "Conventional_Retriever",
    "ZeroMu_Pinned_Bridge",
    "Uncentered_CSLS",
)
LEDGER_ARMS: Tuple[str, ...] = ("SQLiteChallengerLedger", "EK5_v3")
SYSTEM_CELLS: Tuple[Tuple[str, str], ...] = (
    ("Conventional_Retriever", "SQLiteChallengerLedger"),
    ("Conventional_Retriever", "EK5_v3"),
    ("ZeroMu_Pinned_Bridge", "SQLiteChallengerLedger"),
    ("ZeroMu_Pinned_Bridge", "EK5_v3"),
    ("Uncentered_CSLS", "SQLiteChallengerLedger"),
    ("Uncentered_CSLS", "EK5_v3"),
)
EVENT_REGIMES: Tuple[str, ...] = (
    "no_change",
    "encoder_change_only",
    "withdrawal_only",
    "both",
    "sham_subtask_reopen_interrupt",
)
TWIN_BRANCHES: Tuple[str, ...] = (
    "A_derivative_only",
    "B_independent_corroboration",
)
REQUIRED_FIELDS: Tuple[str, ...] = (
    "delivery_date",
    "output_format",
    "Bates_prefix",
    "delivery_channel",
)
TARGET_FIELD = "delivery_date"
UNAFFECTED_FIELDS: Tuple[str, ...] = ("output_format", "Bates_prefix", "delivery_channel")
UNRESOLVED_SENTINEL = "UNRESOLVED"

# Deterministic virtual clock parameters (milliseconds) matching realistic pooled worker budgets.
BASE_SUBTASK_MS = 38.0
RETRIEVAL_BASE_MS = 6.0
# Equalized to 0 so conventional-vs-ZeroMu latency is not a harness stall (62-B1).
CONV_MIGRATION_STALL_MS = 0.0
ZEROMU_BRIDGE_MS = 4.0
SHAM_CHECKPOINT_OVERHEAD_MS = 28.0
EK5_SELECTIVE_REOPEN_LEDGER_MS = 3.5
# Same per-field ledger constant as EK5. The reopen loop charges it once, then
# reopens only the invalid fields (62-B1 symmetric SQLite).
SQLITE_RECHECK_PER_FIELD_MS = EK5_SELECTIVE_REOPEN_LEDGER_MS
# Four-field re-execution budget, charged on top of the rebuild re-exec loop.
FULL_RESTART_REBUILD_MS = 4.0 * (RETRIEVAL_BASE_MS + BASE_SUBTASK_MS)
JOB_SLA_DEADLINE_MS = 560.0


def _canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload, ensure_ascii=True, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_json(payload: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(payload)).hexdigest()


# ---------------------------------------------------------------------------
# 1. 24 Held-Out Production-Matter Templates & Matched Twin Histories
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MatterSourceDoc:
    """One timestamped source document/event in a production-matter timeline."""

    doc_id: str
    matter_id: str
    field_name: str
    field_value: str
    source_kind: str
    authority_tier: int
    tx_time: int
    valid_from: int
    derived_from: Tuple[str, ...] = ()
    text: str = ""
    is_paraphrase: bool = False
    is_independent_corroboration: bool = False
    is_withdrawal: bool = False
    withdraws_target: Optional[str] = None
    explicit_sql_derives: bool = True


@dataclass(frozen=True)
class MatterTemplate:
    """One multi-field production-matter template (`M01`..`M24`)."""

    matter_id: str
    matter_title: str
    tenant_id: str
    delivery_date: str
    output_format: str
    Bates_prefix: str
    delivery_channel: str
    domain_topic: str

    def field_map(self) -> Dict[str, str]:
        return {
            "delivery_date": self.delivery_date,
            "output_format": self.output_format,
            "Bates_prefix": self.Bates_prefix,
            "delivery_channel": self.delivery_channel,
        }

    def doc_id(self, suffix: str) -> str:
        return f"{self.matter_id}_{suffix}"

    def build_branch_events(
        self,
        branch: str,
        *,
        withdrawal_tick: int = 50,
        include_withdrawal: bool = True,
    ) -> Tuple[MatterSourceDoc, ...]:
        """Emit deterministic source timeline for Branch A or Branch B."""
        if branch not in TWIN_BRANCHES:
            raise ValueError(f"Unknown twin branch: {branch!r}")
        mid = self.matter_id
        memo_id = self.doc_id("src_memo")
        worker_sum_id = self.doc_id("src_worker_summary")
        para1_id = self.doc_id("src_paraphrase_1")
        para2_id = self.doc_id("src_paraphrase_2")
        fmt_id = self.doc_id("src_fmt")
        bates_id = self.doc_id("src_bates")
        chan_id = self.doc_id("src_chan")
        signed_id = self.doc_id("src_signed_notice")
        withdraw_id = self.doc_id("src_withdrawal")

        docs: List[MatterSourceDoc] = [
            MatterSourceDoc(
                doc_id=fmt_id,
                matter_id=mid,
                field_name="output_format",
                field_value=self.output_format,
                source_kind="spec_order",
                authority_tier=2,
                tx_time=5,
                valid_from=1,
                text=(
                    f"[{mid}] Production specification order for {self.matter_title} "
                    f"({self.domain_topic}): required output_format is {self.output_format}."
                ),
            ),
            MatterSourceDoc(
                doc_id=bates_id,
                matter_id=mid,
                field_name="Bates_prefix",
                field_value=self.Bates_prefix,
                source_kind="spec_order",
                authority_tier=2,
                tx_time=6,
                valid_from=1,
                text=(
                    f"[{mid}] Numbering protocol for {self.matter_title}: "
                    f"authoritative Bates_prefix is {self.Bates_prefix}."
                ),
            ),
            MatterSourceDoc(
                doc_id=chan_id,
                matter_id=mid,
                field_name="delivery_channel",
                field_value=self.delivery_channel,
                source_kind="spec_order",
                authority_tier=2,
                tx_time=7,
                valid_from=1,
                text=(
                    f"[{mid}] Transfer instructions for {self.matter_title}: "
                    f"authorized delivery_channel is {self.delivery_channel}."
                ),
            ),
            MatterSourceDoc(
                doc_id=memo_id,
                matter_id=mid,
                field_name="delivery_date",
                field_value=self.delivery_date,
                source_kind="scheduling_memo",
                authority_tier=2,
                tx_time=10,
                valid_from=1,
                text=(
                    f"[{mid}] Scheduling memo for {self.matter_title}: "
                    f"provisional delivery_date set to {self.delivery_date}."
                ),
            ),
            # Late pre-withdrawal in-flight worker summary derived from scheduling memo.
            MatterSourceDoc(
                doc_id=worker_sum_id,
                matter_id=mid,
                field_name="delivery_date",
                field_value=self.delivery_date,
                source_kind="inflight_worker_summary",
                authority_tier=1,
                tx_time=max(12, withdrawal_tick - 4),
                valid_from=1,
                derived_from=(memo_id,),
                text=(
                    f"[{mid}] Pre-withdrawal in-flight worker summary echoing scheduling memo: "
                    f"delivery_date remains {self.delivery_date}."
                ),
                is_paraphrase=True,
                explicit_sql_derives=False,
            ),
            # Paraphrased copies derived from memo / worker summary.
            MatterSourceDoc(
                doc_id=para1_id,
                matter_id=mid,
                field_name="delivery_date",
                field_value=self.delivery_date,
                source_kind="paraphrased_copy",
                authority_tier=1,
                tx_time=max(14, withdrawal_tick - 2),
                valid_from=1,
                derived_from=(memo_id,),
                text=(
                    f"[{mid}] Docket digest paraphrase of scheduling memo: "
                    f"target handoff calendar date {self.delivery_date}."
                ),
                is_paraphrase=True,
                explicit_sql_derives=True,
            ),
            MatterSourceDoc(
                doc_id=para2_id,
                matter_id=mid,
                field_name="delivery_date",
                field_value=self.delivery_date,
                source_kind="paraphrased_copy",
                authority_tier=1,
                tx_time=max(15, withdrawal_tick - 1),
                valid_from=1,
                derived_from=(worker_sum_id,),
                text=(
                    f"[{mid}] Secondary intake restatement from worker digest: "
                    f"production due on {self.delivery_date}."
                ),
                is_paraphrase=True,
                explicit_sql_derives=False,
            ),
        ]

        if branch == "B_independent_corroboration":
            docs.append(
                MatterSourceDoc(
                    doc_id=signed_id,
                    matter_id=mid,
                    field_name="delivery_date",
                    field_value=self.delivery_date,
                    source_kind="signed_notice",
                    authority_tier=2,
                    tx_time=max(16, withdrawal_tick - 3),
                    valid_from=1,
                    derived_from=(),
                    text=(
                        f"[{mid}] Independently authored signed stipulation notice for "
                        f"{self.matter_title}: parties jointly confirm delivery_date "
                        f"{self.delivery_date} independent of prior scheduling memos."
                    ),
                    is_independent_corroboration=True,
                )
            )

        if include_withdrawal:
            docs.append(
                MatterSourceDoc(
                    doc_id=withdraw_id,
                    matter_id=mid,
                    field_name="delivery_date",
                    field_value=UNRESOLVED_SENTINEL,
                    source_kind="withdrawal_order",
                    authority_tier=3,
                    tx_time=withdrawal_tick,
                    valid_from=1,
                    derived_from=(),
                    text=(
                        f"[{mid}] Formal withdrawal order vacating scheduling memo {memo_id}: "
                        f"all statements derived solely from {memo_id} are retracted."
                    ),
                    is_withdrawal=True,
                    withdraws_target=memo_id,
                )
            )

        return tuple(docs)


_MATTER_SPECS: Tuple[Tuple[str, str, str, str, str, str], ...] = (
    ("M01", "Alderwood Pipeline Tariff Discovery", "May 30, 2026", "TIFF+Opticon+DAT", "ALDR-PROD-", "SFTP-Vault-East"),
    ("M02", "Bramblett Grid Interconnect Audit", "June 04, 2026", "Native+Concordance", "BRMB-DISC-", "Encrypted-Portal-A"),
    ("M03", "Cairnwood Maritime Charter Dispute", "June 09, 2026", "PDF/A+LoadFile", "CRNW-MAR-", "Aspera-Relay-01"),
    ("M04", "Dunmore Biologics Batch Review", "June 12, 2026", "TIFF+Relativity", "DNMR-BIO-", "SFTP-Vault-West"),
    ("M05", "Eastwick Rail Signal Telemetry", "June 17, 2026", "Native+CSV+Hash", "ESTW-SIG-", "Escrow-Dropbox-02"),
    ("M06", "Fairhaven Refinery Emissions Log", "June 21, 2026", "PDF/A+DAT+OPT", "FRHV-ENV-", "Encrypted-Portal-B"),
    ("M07", "Garrick Avionics Firmware Escrow", "June 25, 2026", "Native+JSONL", "GRCK-AVN-", "SFTP-GovCloud-01"),
    ("M08", "Haldane Municipal Bond Inquiry", "June 29, 2026", "TIFF+Concordance", "HLDN-BND-", "Aspera-Relay-02"),
    ("M09", "Ironside Freight Customs Ledger", "July 03, 2026", "Native+Opticon", "IRNS-FRT-", "SFTP-Vault-Central"),
    ("M10", "Jarrow Semiconductor Yield Claim", "July 07, 2026", "PDF/A+Relativity", "JRRW-SEM-", "Encrypted-Portal-C"),
    ("M11", "Kestrel Clinical Trial Protocol", "July 11, 2026", "TIFF+DAT+Text", "KSTL-CLN-", "HIPAA-SFTP-04"),
    ("M12", "Larkspur Telecom Roaming Tariffs", "July 15, 2026", "Native+Parquet", "LRKS-TEL-", "SFTP-Vault-North"),
    ("M13", "Marlowe Subsea Cable Maintenance", "July 19, 2026", "PDF/A+Opticon", "MRLW-CBL-", "Aspera-Relay-03"),
    ("M14", "Northgate Actuarial Reserve Filing", "July 23, 2026", "TIFF+Concordance", "NRTH-ACT-", "Encrypted-Portal-D"),
    ("M15", "Oakridge Battery Recall Dossier", "July 27, 2026", "Native+Relativity", "OKRG-BAT-", "SFTP-Vault-South"),
    ("M16", "Pendleton Port Crane Liability", "July 31, 2026", "PDF/A+LoadFile", "PNDL-PRT-", "Escrow-Dropbox-05"),
    ("M17", "Quarryman Aggregate Lease Dispute", "August 04, 2026", "TIFF+Opticon+DAT", "QRRY-LSE-", "SFTP-Vault-East2"),
    ("M18", "Redstone Satellite Spectrum Notice", "August 08, 2026", "Native+JSONL", "RDST-SAT-", "SFTP-GovCloud-02"),
    ("M19", "Silverton Hospital Billing Audit", "August 12, 2026", "PDF/A+DAT+OPT", "SLVT-MED-", "HIPAA-SFTP-07"),
    ("M20", "Thornfield Cold-Chain Logistics", "August 16, 2026", "Native+Concordance", "THRN-CLD-", "Encrypted-Portal-E"),
    ("M21", "Upland Hydro Dam Safety Records", "August 20, 2026", "TIFF+Relativity", "UPLD-HYD-", "Aspera-Relay-04"),
    ("M22", "Vanguardia Copper Smelter Permit", "August 24, 2026", "PDF/A+Text+DAT", "VNGR-MIN-", "SFTP-Vault-West2"),
    ("M23", "Westbrook InsurTech Claims Feed", "August 28, 2026", "Native+CSV+Hash", "WSTB-INS-", "Escrow-Dropbox-08"),
    ("M24", "Yarrow Wind Turbine Blade Warranty", "September 02, 2026", "TIFF+Concordance", "YRRW-WND-", "Encrypted-Portal-F"),
)

_TOPICS: Tuple[str, ...] = (
    "energy_tariff",
    "grid_compliance",
    "maritime_charter",
    "pharma_quality",
    "rail_safety",
    "environmental_log",
    "avionics_escrow",
    "municipal_finance",
    "customs_trade",
    "semiconductor_fab",
    "clinical_trial",
    "telecom_roaming",
    "subsea_infra",
    "insurance_actuarial",
    "product_recall",
    "port_operations",
    "mineral_lease",
    "orbital_spectrum",
    "healthcare_billing",
    "cold_chain",
    "hydro_engineering",
    "mining_permit",
    "insurtech_claims",
    "renewable_warranty",
)


def build_24_matter_templates() -> Tuple[MatterTemplate, ...]:
    """Return the 24 held-out multi-field production-matter templates."""
    templates: List[MatterTemplate] = []
    for idx, (mid, title, ddate, ofmt, bates, chan) in enumerate(_MATTER_SPECS):
        templates.append(
            MatterTemplate(
                matter_id=mid,
                matter_title=title,
                tenant_id=f"tenant_{mid.lower()}",
                delivery_date=ddate,
                output_format=ofmt,
                Bates_prefix=bates,
                delivery_channel=chan,
                domain_topic=_TOPICS[idx],
            )
        )
    return tuple(templates)


def matter_corpus_fingerprints(templates: Sequence[MatterTemplate]) -> Dict[str, Any]:
    """Compute canonical normalized SHA-256 text fingerprints via `burn_ledger.text_fingerprint`."""
    doc_fps: Dict[str, str] = {}
    for tmpl in templates:
        # Branch B is a strict superset of Branch A's doc IDs (adding `src_signed_notice`).
        for doc in tmpl.build_branch_events("B_independent_corroboration", withdrawal_tick=50, include_withdrawal=True):
            doc_fps[doc.doc_id] = text_fingerprint(doc.text)
    unique_count = len(set(doc_fps.values()))
    return {
        "matter_count": len(templates),
        "total_documents": len(doc_fps),
        "unique_document_fingerprints": unique_count,
        "inventory_sha256": _sha256_json(doc_fps),
    }


# ---------------------------------------------------------------------------
# 2. Four Deterministic Arrival Schedules
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JobArrival:
    """One scheduled matter job arrival within a deterministic schedule."""

    job_index: int
    matter_id: str
    schedule_id: str
    arrival_ms: float
    arrival_tick: int
    deadline_ms: float
    migration_tick: int
    withdrawal_tick: int
    in_flight_at_event: bool


def build_arrival_schedule(
    schedule_id: str,
    templates: Sequence[MatterTemplate],
) -> Tuple[JobArrival, ...]:
    """Build deterministic arrivals across 24 matters for one of 4 schedules."""
    if schedule_id not in ARRIVAL_SCHEDULES:
        raise ValueError(f"Unknown schedule_id={schedule_id!r}; expected one of {ARRIVAL_SCHEDULES}")
    migration_tick = 40
    withdrawal_tick = 50
    arrivals: List[JobArrival] = []
    n = len(templates)
    for idx, tmpl in enumerate(templates):
        if schedule_id == "S1_steady":
            arrival_ms = float(idx * 75.0)
            arrival_tick = 20 + idx
        elif schedule_id == "S2_bursty_front":
            burst_group = idx // 6
            in_burst = idx % 6
            arrival_ms = float(burst_group * 460.0 + in_burst * 14.0)
            arrival_tick = 22 + burst_group * 5 + in_burst
        elif schedule_id == "S3_wave_bimodal":
            if idx < n // 2:
                arrival_ms = float(idx * 62.0)
                arrival_tick = 26 + idx
            else:
                arrival_ms = float(860.0 + (idx - n // 2) * 62.0)
                arrival_tick = 36 + (idx - n // 2)
        else:  # S4_withdrawal_surge
            arrival_ms = float(80.0 + (idx % 4) * 18.0 + (idx // 4) * 280.0)
            arrival_tick = 30 + (idx % 4) * 2 + (idx // 4) * 2

        # Every scheduled job starts before withdrawal_tick=50 and commits after tick 50
        # so twin perturbation and sham-interrupt comparisons are 100% matched across schedules.
        in_flight = arrival_tick < withdrawal_tick
        arrivals.append(
            JobArrival(
                job_index=idx,
                matter_id=tmpl.matter_id,
                schedule_id=schedule_id,
                arrival_ms=arrival_ms,
                arrival_tick=arrival_tick,
                deadline_ms=arrival_ms + JOB_SLA_DEADLINE_MS,
                migration_tick=migration_tick,
                withdrawal_tick=withdrawal_tick,
                in_flight_at_event=in_flight,
            )
        )
    return tuple(arrivals)


# ---------------------------------------------------------------------------
# 3. Genuine Dual-Encoder Embedding Space & Bridge Retriever (v1 <-> v2)
# ---------------------------------------------------------------------------


class MatterEmbeddingSpace:
    """Deterministic `d_old=64 <-> d_new=96` dual-encoder space with real bridge fits.

    Fits genuine `UncenteredProcrustes` (+ `RawCosineMerge`) for `Conventional_Retriever`
    and genuine `ZeroMuPolarBridge` (+ `CSLSMerge`) for `ZeroMu_Pinned_Bridge` from
    `chelated_pv7_spine.bridge_merge_harness`.
    """

    def __init__(
        self,
        templates: Sequence[MatterTemplate],
        *,
        d_old: int = 64,
        d_new: int = 96,
        n_anchors: int = 160,
        seed: int = 2062,
    ) -> None:
        self.d_old = int(d_old)
        self.d_new = int(d_new)
        self.seed = int(seed)
        rng = np.random.default_rng(seed)

        # True cross-encoder semi-orthogonal rotation R: R^T R = I_{d_old}
        q_mat, _ = np.linalg.qr(rng.normal(size=(self.d_new, self.d_old)))
        self._rot = q_mat[:, : self.d_old]

        # Strong anisotropic centroid offset in v1 and v2 + prompt-wrapper shift on migrated queries
        # (mirrors the real geometry verified in Item 6B!).
        self._mu_v1 = l2_normalize(rng.normal(size=(1, self.d_old))) * 1.45
        self._mu_v2 = l2_normalize(rng.normal(size=(1, self.d_new))) * 2.10
        self._prompt_shift_v2 = l2_normalize(rng.normal(size=(1, self.d_new))) * 1.85

        # Build unlabeled anchor pairs (v2 -> v1 mapping for bridging v2 queries to v1 resident docs)
        latent_anc = l2_normalize(rng.normal(size=(n_anchors, self.d_old)))
        anc_v1 = l2_normalize(latent_anc * 0.45 + self._mu_v1 + rng.normal(scale=0.02, size=(n_anchors, self.d_old)))
        anc_v2 = l2_normalize(
            (latent_anc @ self._rot.T) * 0.45
            + self._mu_v2
            + rng.normal(scale=0.02, size=(n_anchors, self.d_new))
        )

        # Fit both canonical bridge families from bridge_merge_harness
        self.uncentered_bridge = UncenteredProcrustes.fit(anc_v2, anc_v1)
        self.zeromu_bridge = ZeroMuPolarBridge.fit(anc_v2, anc_v1, scoring_mode="dual_adap", ema_alpha=0.25)
        self.raw_merger = RawCosineMerge()
        self.csls_merger = CSLSMerge(csls_k=5)

        # Build embeddings for every matter's documents & per-field queries in v1 and v2
        self._doc_v1: Dict[str, np.ndarray] = {}
        self._doc_v2: Dict[str, np.ndarray] = {}
        self._query_v1: Dict[Tuple[str, str], np.ndarray] = {}
        self._query_v2_shifted: Dict[Tuple[str, str], np.ndarray] = {}
        self._hub_dir_v1 = l2_normalize(rng.normal(size=(1, self.d_old)))[0]

        for tmpl in templates:
            field_latents: Dict[str, np.ndarray] = {}
            for fname in REQUIRED_FIELDS:
                base = l2_normalize(rng.normal(size=(self.d_old,)))
                field_latents[fname] = base
                q1 = l2_normalize(base * 0.55 + self._mu_v1[0])
                q2 = l2_normalize((base @ self._rot.T) * 0.42 + self._mu_v2[0] + self._prompt_shift_v2[0])
                self._query_v1[(tmpl.matter_id, fname)] = q1
                self._query_v2_shifted[(tmpl.matter_id, fname)] = q2

            # Generate embeddings for all docs (union of Branch A and B doc IDs)
            all_docs = tmpl.build_branch_events("B_independent_corroboration", withdrawal_tick=50, include_withdrawal=True)
            for doc in all_docs:
                base = field_latents[doc.field_name]
                if doc.is_independent_corroboration:
                    # Signed notice lives cleanly on the semantic signal direction (less hub bias).
                    sig = l2_normalize(base * 0.92 + rng.normal(scale=0.04, size=(self.d_old,)))
                    v1_vec = l2_normalize(sig * 0.48 + self._mu_v1[0] * 0.80)
                    v2_vec = l2_normalize((sig @ self._rot.T) * 0.48 + self._mu_v2[0] * 0.80)
                elif doc.is_paraphrase:
                    # Paraphrases/worker echoes sit closer to the anisotropic centroid / hub direction
                    # so uncentered cosine is pulled toward hub echoes under encoder shift!
                    sig = l2_normalize(base * 0.55 + self._hub_dir_v1 * 0.45 + rng.normal(scale=0.05, size=(self.d_old,)))
                    v1_vec = l2_normalize(sig * 0.30 + self._mu_v1[0] * 1.45)
                    v2_vec = l2_normalize((sig @ self._rot.T) * 0.30 + self._mu_v2[0] * 1.45)
                elif doc.is_withdrawal:
                    sig = l2_normalize(-base * 0.30 + rng.normal(scale=0.25, size=(self.d_old,)))
                    v1_vec = l2_normalize(sig * 0.35 + self._mu_v1[0])
                    v2_vec = l2_normalize((sig @ self._rot.T) * 0.35 + self._mu_v2[0])
                else:
                    sig = l2_normalize(base * 0.88 + rng.normal(scale=0.05, size=(self.d_old,)))
                    v1_vec = l2_normalize(sig * 0.46 + self._mu_v1[0])
                    v2_vec = l2_normalize((sig @ self._rot.T) * 0.46 + self._mu_v2[0])
                self._doc_v1[doc.doc_id] = v1_vec
                self._doc_v2[doc.doc_id] = v2_vec

        # EVAL query matrices are the matter queries. CSLS / ZeroMu calibrate on a
        # disjoint draw from the same encoder geometry, never on the eval rows.
        eval_q_keys = sorted(self._query_v1.keys())
        self._dev_q_v1 = np.vstack([self._query_v1[k] for k in eval_q_keys])
        self._dev_q_v2 = np.vstack([self._query_v2_shifted[k] for k in eval_q_keys])
        dev_rng = np.random.default_rng(62017)
        n_dev_q = 32
        dev_q_rows = []
        for _ in range(n_dev_q):
            base = l2_normalize(dev_rng.normal(size=(self.d_old,)))
            dev_q_rows.append(
                l2_normalize((base @ self._rot.T) * 0.42 + self._mu_v2[0] + self._prompt_shift_v2[0])
            )
        self._csls_dev_q_v2 = np.vstack(dev_q_rows)
        dev_doc_rows = []
        for _ in range(12):
            base = l2_normalize(dev_rng.normal(size=(self.d_old,)))
            sig = l2_normalize(
                base * 0.55 + self._hub_dir_v1 * 0.45 + dev_rng.normal(scale=0.05, size=(self.d_old,))
            )
            dev_doc_rows.append(l2_normalize(sig * 0.30 + self._mu_v1[0] * 1.45))
        self._zeromu_dev_docs = np.vstack(dev_doc_rows)

        # Build canonical resident corpus doc arrays (sorted by doc_id)
        self._all_doc_ids = tuple(sorted(self._doc_v1.keys()))
        self._doc_id_to_idx = {did: i for i, did in enumerate(self._all_doc_ids)}
        self._resident_v1 = np.vstack([self._doc_v1[did] for did in self._all_doc_ids])
        self._resident_v2 = np.vstack([self._doc_v2[did] for did in self._all_doc_ids])

        # Precompute exact merged score matrices across the full corpus for fast, deterministic,
        # 100% genuine bridge scoring during the pooled-agent sweep.
        # Backfill mask: 25% of docs are already in v2, 75% reside in legacy v1 (dual-shard migration!).
        self._backfill_mask = np.asarray([(i % 4 == 0) for i in range(len(self._all_doc_ids))], dtype=bool)
        # Ensure signed notices remain on legacy v1 shard during migration so the bridge is genuinely tested!
        for did, idx in self._doc_id_to_idx.items():
            if did.endswith("src_signed_notice"):
                self._backfill_mask[idx] = False

        # Native v1 scores (no encoder change) stay on the EVAL matter queries.
        self._scores_native_v1 = self._dev_q_v1 @ l2_normalize(self._resident_v1).T

        # Migrated v2 -> v1. CSLS hubness and ZeroMu mu_d use the disjoint DEV draw.
        conv_bridged = self.uncentered_bridge.score_matrix(self._dev_q_v2, self._resident_v1)
        zeromu_bridged = self.zeromu_bridge.score_matrix(
            self._dev_q_v2, self._resident_v1, dev_docs=self._zeromu_dev_docs
        )
        unccs_bridged = self.uncentered_bridge.score_matrix(self._dev_q_v2, self._resident_v1)
        native_v2 = l2_normalize(self._dev_q_v2) @ l2_normalize(self._resident_v2).T
        dev_native_v2 = l2_normalize(self._csls_dev_q_v2) @ l2_normalize(self._resident_v2).T
        dev_zeromu = self.zeromu_bridge.score_matrix(
            self._csls_dev_q_v2, self._resident_v1, dev_docs=self._zeromu_dev_docs
        )
        dev_unccs = self.uncentered_bridge.score_matrix(self._csls_dev_q_v2, self._resident_v1)

        self._scores_conv_migrated = self.raw_merger.merge(
            conv_bridged, native_v2, self._backfill_mask, bridged_ref=conv_bridged, native_ref=native_v2
        )
        self._scores_zeromu_migrated = self.csls_merger.merge(
            zeromu_bridged,
            native_v2,
            self._backfill_mask,
            bridged_ref=dev_zeromu,
            native_ref=dev_native_v2,
            strict_disjoint_dev=True,
        )
        self._scores_unccs_migrated = self.csls_merger.merge(
            unccs_bridged,
            native_v2,
            self._backfill_mask,
            bridged_ref=dev_unccs,
            native_ref=dev_native_v2,
            strict_disjoint_dev=True,
        )
        self._q_key_to_row = {k: i for i, k in enumerate(eval_q_keys)}

    def rank_visible_docs(
        self,
        *,
        matter_id: str,
        field_name: str,
        retriever_arm: str,
        encoder_migrated: bool,
        visible_doc_ids: Sequence[str],
        top_k: int = 3,
        profiler: Optional[RequestSpanProfiler] = None,
        request_id: str = "",
    ) -> Tuple[Tuple[str, float], ...]:
        """Rank visible documents for `(matter_id, field_name)` using real bridge scores."""
        if retriever_arm not in RETRIEVER_ARMS:
            raise ValueError(f"Unknown retriever_arm={retriever_arm!r}")
        t0 = time.perf_counter_ns()
        row_idx = self._q_key_to_row[(matter_id, field_name)]
        if not encoder_migrated:
            row = self._scores_native_v1[row_idx]
        elif retriever_arm == "Conventional_Retriever":
            row = self._scores_conv_migrated[row_idx]
        elif retriever_arm == "Uncentered_CSLS":
            row = self._scores_unccs_migrated[row_idx]
        elif retriever_arm == "ZeroMu_Pinned_Bridge":
            row = self._scores_zeromu_migrated[row_idx]
        else:
            raise ValueError(f"Unknown retriever_arm={retriever_arm!r}")

        scored: List[Tuple[str, float]] = []
        for did in visible_doc_ids:
            col = self._doc_id_to_idx.get(did)
            if col is not None:
                scored.append((did, float(row[col])))
        scored.sort(key=lambda item: (-item[1], item[0]))
        elapsed_us = (time.perf_counter_ns() - t0) / 1_000.0
        if profiler is not None:
            top_id = scored[0][0] if scored else ""
            # Proxy hit indicator for span summary
            hit_score = 1.0 if top_id else 0.0
            profiler.record_span(
                request_id=request_id or f"{matter_id}:{field_name}",
                queue_us=0.0,
                backend_knn_us=elapsed_us,
                prefilter_visited=len(visible_doc_ids),
                short_refill_count=0,
                exact_fallback_triggered=False,
                gemv_us=elapsed_us * 0.4,
                merge_us=elapsed_us * 0.2,
                total_us=elapsed_us,
                timed_out=False,
                ndcg_at_10=hit_score,
            )
        return tuple(scored[:top_k])


# ---------------------------------------------------------------------------
# 4. Dual Ledger Adapters (`SQLiteChallengerLedger` vs `EK5_v3`)
# ---------------------------------------------------------------------------


class LedgerAdapter(ABC):
    """Common interface over `SQLiteChallengerLedger` and `EK5_v3`."""

    arm_name: str

    @abstractmethod
    def ingest_docs(self, docs: Sequence[MatterSourceDoc]) -> Set[str]:
        """Ingest source docs/withdrawals; return set of newly invalidated doc IDs."""

    @abstractmethod
    def is_admissible_at(self, doc_id: str, matter_id: str, field_name: str, tx_time: int) -> bool:
        """Return whether `doc_id` is admissible at `tx_time`."""

    @abstractmethod
    def admissible_docs_for_field(
        self, matter_id: str, field_name: str, tx_time: int
    ) -> List[str]:
        """Return all admissible doc IDs for `(matter_id, field_name)` at `tx_time`."""

    def close(self) -> None:
        return None


class EK5LedgerAdapter(LedgerAdapter):
    """Genuine `EK5_v3` adapter wrapping `StratifiedEvidenceStore` + canonical audit export."""

    arm_name = "EK5_v3"

    def __init__(self) -> None:
        self.store = StratifiedEvidenceStore(strict_dag_supersede=True, exact_min_rank=True)
        self._docs: Dict[str, MatterSourceDoc] = {}
        self._ingested_ids: Set[str] = set()
        self._att_seq = 0

    def ingest_docs(self, docs: Sequence[MatterSourceDoc]) -> Set[str]:
        """Apply atomic incremental mutation to `StratifiedEvidenceStore`."""
        new_events: List[StratifiedEvent] = []
        new_atts: List[EdgeAttestation] = []
        for doc in sorted(docs, key=lambda d: (d.tx_time, d.doc_id)):
            if doc.doc_id in self._ingested_ids:
                continue
            self._ingested_ids.add(doc.doc_id)
            self._docs[doc.doc_id] = doc
            new_events.append(
                StratifiedEvent(
                    event_id=doc.doc_id,
                    entity=doc.matter_id,
                    slot=doc.field_name,
                    claim_value=doc.field_value,
                    text=doc.text,
                    valid_time=doc.valid_from,
                    tx_time=doc.tx_time,
                    tenant=doc.matter_id,
                    domain="legal_production",
                    authority_tier=doc.authority_tier,
                )
            )
            for parent_id in doc.derived_from:
                self._att_seq += 1
                etype = "COPY_OF" if doc.is_paraphrase else "DERIVES_CURRENT"
                new_atts.append(
                    EdgeAttestation(
                        attestation_id=f"ek5_dep_{self._att_seq}",
                        src_id=doc.doc_id,
                        dst_id=parent_id,
                        edge_type=etype,
                        valid_time=doc.valid_from,
                        tx_time=doc.tx_time,
                    )
                )
            if doc.is_withdrawal and doc.withdraws_target:
                self._att_seq += 1
                new_atts.append(
                    EdgeAttestation(
                        attestation_id=f"ek5_ref_{self._att_seq}",
                        src_id=doc.doc_id,
                        dst_id=doc.withdraws_target,
                        edge_type="REFUTE",
                        valid_time=doc.valid_from,
                        tx_time=doc.tx_time,
                    )
                )
        if not new_events and not new_atts:
            return set()
        delta = self.store.apply_mutation(new_events, new_atts)
        invalidated = {
            nid for nid, (old_e, new_e) in delta.transitioned.items() if old_e == "ADMIT" and new_e != "ADMIT"
        }
        return invalidated

    def is_admissible_at(self, doc_id: str, matter_id: str, field_name: str, tx_time: int) -> bool:
        doc = self._docs.get(doc_id)
        if doc is None or doc.matter_id != matter_id or doc.field_name != field_name:
            return False
        if doc.is_withdrawal or doc.tx_time > tx_time:
            return False
        return self.store.eligibility.get(doc_id) == "ADMIT"

    def admissible_docs_for_field(self, matter_id: str, field_name: str, tx_time: int) -> List[str]:
        out: List[str] = []
        for did, doc in self._docs.items():
            if (
                doc.matter_id == matter_id
                and doc.field_name == field_name
                and not doc.is_withdrawal
                and doc.tx_time <= tx_time
                and self.store.eligibility.get(did) == "ADMIT"
            ):
                out.append(did)
        out.sort()
        return out

    def verify_incremental_vs_full_parity(self) -> Dict[str, Any]:
        """Run `export_canonical_ek5_audit_trace` + `compare_with_fresh_rebuild`."""
        trace = export_canonical_ek5_audit_trace(self.store)
        cmp = compare_with_fresh_rebuild(self.store, trace=trace)
        parity_ok = (
            len(cmp.semantic_diffs) == 0
            and len(cmp.canonical_reason_diffs) == 0
            and len(cmp.canonical_pos_cyc_diffs) == 0
            and len(cmp.canonical_semantic_mismatches) == 0
        )
        return {
            "parity_ok": parity_ok,
            "node_count": cmp.node_count,
            "semantic_diffs": len(cmp.semantic_diffs),
            "canonical_reason_diffs": len(cmp.canonical_reason_diffs),
            "canonical_pos_cyc_diffs": len(cmp.canonical_pos_cyc_diffs),
            "canonical_semantic_mismatches": len(cmp.canonical_semantic_mismatches),
            "canonical_trace_sha256": trace.digest(),
            "export_ms": round(trace.export_ms, 4),
        }


class SQLiteChallengerAdapter(LedgerAdapter):
    """Genuine `SQLiteChallengerLedger` wrapping `ChallengerBitemporalLedger(SQLiteChallengerBackend)`."""

    arm_name = "SQLiteChallengerLedger"

    def __init__(self, *, query_mode: str = "recursive_cte") -> None:
        self.query_mode = query_mode
        self.backend = SQLiteChallengerBackend(":memory:", cache_mib=32)
        self.ledger = ChallengerBitemporalLedger(self.backend)
        self._docs: Dict[str, MatterSourceDoc] = {}
        self._ingested_ids: Set[str] = set()
        self._edge_seq = 0

    def ingest_docs(self, docs: Sequence[MatterSourceDoc]) -> Set[str]:
        """Append envelopes into SQLite challenger ledger (no incremental dirty-cone push)."""
        for doc in sorted(docs, key=lambda d: (d.tx_time, d.doc_id)):
            if doc.doc_id in self._ingested_ids:
                continue
            self._ingested_ids.add(doc.doc_id)
            self._docs[doc.doc_id] = doc
            entity_key = f"{doc.matter_id}::{doc.field_name}"
            if doc.is_withdrawal and doc.withdraws_target:
                self.ledger.append(
                    LedgerRecordEnvelope(
                        tenant_id=doc.matter_id,
                        record_id=doc.withdraws_target,
                        entity_key=entity_key,
                        authority_rank=doc.authority_tier,
                        valid_from=doc.valid_from,
                        valid_to=None,
                        tx_time=doc.tx_time,
                        op_kind=OpKind.WITHDRAW,
                        claim_value="",
                    )
                )
            else:
                self.ledger.append(
                    LedgerRecordEnvelope(
                        tenant_id=doc.matter_id,
                        record_id=doc.doc_id,
                        entity_key=entity_key,
                        authority_rank=doc.authority_tier,
                        valid_from=doc.valid_from,
                        valid_to=None,
                        tx_time=doc.tx_time,
                        op_kind=OpKind.ASSERT,
                        claim_value=doc.field_value,
                    )
                )
                # Symmetric with EK5: every derived_from parent is an edge.
                # explicit_sql_derives stays on the document and is not a harness handicap.
                for parent_id in doc.derived_from:
                    self._edge_seq += 1
                    self.ledger.add_edge(
                        LedgerEdgeEnvelope(
                            tenant_id=doc.matter_id,
                            edge_id=f"sql_e_{self._edge_seq}",
                            src_id=doc.doc_id,
                            dst_id=parent_id,
                            tx_time=doc.tx_time,
                        )
                    )
        # Conventional SQLite challenger does not push incremental dirty-node deltas.
        return set()

    def is_admissible_at(self, doc_id: str, matter_id: str, field_name: str, tx_time: int) -> bool:
        entity_key = f"{matter_id}::{field_name}"
        res = self.ledger.query_as_of(
            tenant_id=matter_id,
            entity_key=entity_key,
            valid_time=1,
            tx_time=tx_time,
            mode=self.query_mode,
        )
        return doc_id in res.live_record_ids

    def admissible_docs_for_field(self, matter_id: str, field_name: str, tx_time: int) -> List[str]:
        entity_key = f"{matter_id}::{field_name}"
        res = self.ledger.query_as_of(
            tenant_id=matter_id,
            entity_key=entity_key,
            valid_time=1,
            tx_time=tx_time,
            mode=self.query_mode,
        )
        if res.status != STATUS_RESOLVED:
            return []
        return list(res.live_record_ids)

    def close(self) -> None:
        self.ledger.close()


# ---------------------------------------------------------------------------
# 5. Independent Commit-Prefix Verifier (Zero Shared Decision Code)
# ---------------------------------------------------------------------------


def _ground_truth_live_docs_at_prefix(
    all_docs: Sequence[MatterSourceDoc],
    commit_tick: int,
) -> Tuple[Set[str], Set[str]]:
    """Pure oracle: return `(live_doc_ids, withdrawn_or_derived_doc_ids)` at `commit_tick`."""
    visible = {d.doc_id: d for d in all_docs if d.tx_time <= commit_tick}
    withdrawn: Set[str] = set()
    for d in visible.values():
        if d.is_withdrawal and d.withdraws_target and d.withdraws_target in visible:
            target = visible[d.withdraws_target]
            if d.authority_tier >= target.authority_tier:
                withdrawn.add(d.withdraws_target)
    # Transitive closure over ALL true derivation chains (including paraphrased worker echoes!)
    changed = True
    while changed:
        changed = False
        for did, doc in visible.items():
            if did in withdrawn or doc.is_withdrawal:
                continue
            if any(parent in withdrawn or parent not in visible for parent in doc.derived_from):
                withdrawn.add(did)
                changed = True
    live = {
        did
        for did, doc in visible.items()
        if not doc.is_withdrawal and did not in withdrawn
    }
    return live, withdrawn


@dataclass(frozen=True)
class CommitVerification:
    """Result of independent verification against the event prefix visible at commit time."""

    fields_exact: bool
    citations_valid_at_commit_prefix: bool
    stale_commit: bool
    unaffected_fields_preserved: int
    errors: Tuple[str, ...]


def verify_commit_against_visible_prefix(
    *,
    template: MatterTemplate,
    branch: str,
    event_regime: str,
    all_docs: Sequence[MatterSourceDoc],
    commit_tick: int,
    committed_fields: Mapping[str, str],
    committed_citations: Mapping[str, Optional[str]],
) -> CommitVerification:
    """Verify committed fields and cited authoritative source IDs against `tx_time <= commit_tick`."""
    live_ids, withdrawn_ids = _ground_truth_live_docs_at_prefix(all_docs, commit_tick)
    by_id = {d.doc_id: d for d in all_docs}
    errors: List[str] = []
    stale = False

    withdrawal_active = event_regime in {"withdrawal_only", "both"} and any(
        d.is_withdrawal and d.tx_time <= commit_tick for d in all_docs
    )

    # Check the 3 unaffected fields first
    preserved = 0
    expected_map = template.field_map()
    for fname in UNAFFECTED_FIELDS:
        val = committed_fields.get(fname)
        cit = committed_citations.get(fname)
        val_ok = val == expected_map[fname]
        cit_ok = (
            isinstance(cit, str)
            and cit in live_ids
            and by_id[cit].field_name == fname
            and by_id[cit].field_value == expected_map[fname]
        )
        if val_ok and cit_ok:
            preserved += 1
        else:
            if isinstance(cit, str) and cit in withdrawn_ids:
                stale = True
            errors.append(f"{fname}: val_ok={val_ok} cit_ok={cit_ok} cit={cit!r}")

    # Check target field (`delivery_date`) according to twin branch & visible prefix
    target_val = committed_fields.get(TARGET_FIELD)
    target_cit = committed_citations.get(TARGET_FIELD)
    if withdrawal_active and branch == "A_derivative_only":
        if target_val != UNRESOLVED_SENTINEL:
            stale = True
            errors.append(f"Branch A post-withdrawal expected UNRESOLVED, got {target_val!r}")
        if target_cit is not None:
            if target_cit in withdrawn_ids:
                stale = True
            errors.append(f"Branch A post-withdrawal must have citation None, got {target_cit!r}")
    elif withdrawal_active and branch == "B_independent_corroboration":
        signed_id = template.doc_id("src_signed_notice")
        if target_val != template.delivery_date:
            errors.append(f"Branch B expected {template.delivery_date!r}, got {target_val!r}")
        if target_cit in withdrawn_ids:
            stale = True
            errors.append(f"Branch B cited withdrawn/derivative source {target_cit!r}")
        elif target_cit != signed_id or target_cit not in live_ids:
            errors.append(f"Branch B must cite independent notice {signed_id!r}, got {target_cit!r}")
    else:
        # No withdrawal active at commit_tick (no_change, encoder_change_only, sham_subtask_reopen_interrupt,
        # or committed strictly before withdrawal_tick)
        if target_val != template.delivery_date:
            errors.append(f"Active delivery_date expected {template.delivery_date!r}, got {target_val!r}")
        if not (
            isinstance(target_cit, str)
            and target_cit in live_ids
            and by_id[target_cit].field_name == TARGET_FIELD
            and by_id[target_cit].field_value == template.delivery_date
        ):
            if isinstance(target_cit, str) and target_cit in withdrawn_ids:
                stale = True
            errors.append(f"Invalid delivery_date citation {target_cit!r} at tick {commit_tick}")

    fields_exact = (preserved == len(UNAFFECTED_FIELDS)) and (
        (target_val == UNRESOLVED_SENTINEL)
        if (withdrawal_active and branch == "A_derivative_only")
        else (target_val == template.delivery_date)
    )
    citations_valid = len(errors) == 0
    return CommitVerification(
        fields_exact=fields_exact,
        citations_valid_at_commit_prefix=citations_valid,
        stale_commit=stale,
        unaffected_fields_preserved=preserved,
        errors=tuple(errors),
    )


# ---------------------------------------------------------------------------
# 6. Worker Backends (Stage 1 Controlled Replay + Stage 2/3 Live vLLM Pool)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SubtaskDraft:
    """Output of one worker subtask extracting a single production-matter field."""

    field_name: str
    field_value: str
    cited_source_id: Optional[str]
    exec_ms: float
    timed_out: bool = False
    live_call: bool = False


class WorkerBackend(ABC):
    """Pluggable worker execution backend (Stage 1 deterministic replay or Stage 2/3 live vLLM)."""

    mode_name: str

    @abstractmethod
    def extract_field(
        self,
        *,
        template: MatterTemplate,
        field_name: str,
        candidate_docs: Sequence[MatterSourceDoc],
        is_reopen: bool = False,
    ) -> SubtaskDraft:
        """Produce a field value + cited source ID from retrieved visible candidate docs."""


class RecordedReplayWorkerBackend(WorkerBackend):
    """Stage 1 deterministic CPU recorded-worker attribution replay (`live_model_calls == 0`)."""

    mode_name = "stage1-controlled-replay"

    def extract_field(
        self,
        *,
        template: MatterTemplate,
        field_name: str,
        candidate_docs: Sequence[MatterSourceDoc],
        is_reopen: bool = False,
    ) -> SubtaskDraft:
        # Deterministic per-(matter, field) micro-jitter to model heterogeneous document lengths
        digest = hashlib.sha256(f"{template.matter_id}:{field_name}:{int(is_reopen)}".encode("ascii")).digest()
        jitter_ms = (digest[0] / 255.0) * 6.0
        exec_ms = (BASE_SUBTASK_MS * (0.72 if is_reopen else 1.0)) + jitter_ms

        matching = [d for d in candidate_docs if d.field_name == field_name and not d.is_withdrawal]
        if not matching:
            return SubtaskDraft(
                field_name=field_name,
                field_value=UNRESOLVED_SENTINEL,
                cited_source_id=None,
                exec_ms=exec_ms,
                timed_out=False,
                live_call=False,
            )
        # Worker selects the highest-ranked retrieved doc (respecting retrieval order!)
        chosen = matching[0]
        return SubtaskDraft(
            field_name=field_name,
            field_value=chosen.field_value,
            cited_source_id=chosen.doc_id,
            exec_ms=exec_ms,
            timed_out=False,
            live_call=False,
        )


class LiveVLLMWorkerBackend(WorkerBackend):
    """Stage 2/3 live OpenAI-compatible vLLM worker backend (e.g. Spark #3 pool)."""

    mode_name = "run-live-workers"

    def __init__(
        self,
        *,
        vllm_url: str,
        model: str = "Qwen/Qwen2.5-7B-Instruct",
        timeout_s: float = 15.0,
        api_key: str = "EMPTY",
    ) -> None:
        if not vllm_url or not vllm_url.startswith(("http://", "https://")):
            raise ValueError(f"vllm_url must be an http(s) URL, got {vllm_url!r}")
        self.vllm_url = vllm_url.rstrip("/")
        self.model = model
        self.timeout_s = float(timeout_s)
        self.api_key = api_key
        self.served_model: Optional[str] = None

    def _endpoint(self) -> str:
        if self.vllm_url.endswith("/v1/chat/completions"):
            return self.vllm_url
        if self.vllm_url.endswith("/v1"):
            return f"{self.vllm_url}/chat/completions"
        return f"{self.vllm_url}/v1/chat/completions"

    def extract_field(
        self,
        *,
        template: MatterTemplate,
        field_name: str,
        candidate_docs: Sequence[MatterSourceDoc],
        is_reopen: bool = False,
    ) -> SubtaskDraft:
        t0 = time.perf_counter()
        if not candidate_docs:
            return SubtaskDraft(
                field_name=field_name,
                field_value=UNRESOLVED_SENTINEL,
                cited_source_id=None,
                exec_ms=(time.perf_counter() - t0) * 1000.0,
                timed_out=False,
                live_call=False,
            )

        snippets = "\n".join(f"- [{d.doc_id}] ({d.source_kind}): {d.text}" for d in candidate_docs)
        system_msg = (
            "You are a legal production extraction worker. Return ONLY a JSON object with keys "
            '"field_value" and "cited_source_id". If no admissible source establishes the field, '
            'return {"field_value": "UNRESOLVED", "cited_source_id": null}.'
        )
        user_msg = (
            f"Matter: {template.matter_id} ({template.matter_title})\n"
            f"Target field: {field_name}\n"
            f"Visible admissible sources:\n{snippets}\n"
            "Extract the authoritative field_value and exact cited_source_id."
        )
        body = json.dumps(
            {
                "model": self.model,
                "temperature": 0.0,
                "max_tokens": 96,
                "messages": [
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": user_msg},
                ],
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            self._endpoint(),
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_s) as resp:
                raw = json.loads(resp.read().decode("utf-8"))
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            served = raw.get("model")
            self.served_model = None if served is None else str(served)
            if served != self.model:
                return SubtaskDraft(
                    field_name=field_name,
                    field_value=UNRESOLVED_SENTINEL,
                    cited_source_id=None,
                    exec_ms=elapsed_ms,
                    timed_out=True,
                    live_call=True,
                )
            content = str(raw["choices"][0]["message"]["content"]).strip()
            # Extract JSON block if wrapped in markdown fences
            start = content.find("{")
            end = content.rfind("}")
            parsed = json.loads(content[start : end + 1] if start != -1 and end > start else content)
            fval = str(parsed.get("field_value", UNRESOLVED_SENTINEL))
            cit = parsed.get("cited_source_id")
            cit_str = None if cit in (None, "", "null", "None") else str(cit)
            return SubtaskDraft(
                field_name=field_name,
                field_value=fval,
                cited_source_id=cit_str,
                exec_ms=elapsed_ms,
                timed_out=False,
                live_call=True,
            )
        except (urllib.error.URLError, TimeoutError, ValueError, KeyError, OSError):
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            return SubtaskDraft(
                field_name=field_name,
                field_value=UNRESOLVED_SENTINEL,
                cited_source_id=None,
                exec_ms=max(elapsed_ms, self.timeout_s * 1000.0),
                timed_out=True,
                live_call=True,
            )


# ---------------------------------------------------------------------------
# 7. Multi-Worker Pool Coordinator & Factorial Evaluator
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class JobOutcome:
    """Complete accounting record for one eligible production-matter job."""

    job_id: str
    matter_id: str
    branch: str
    schedule_id: str
    pool_size: int
    retriever_arm: str
    ledger_arm: str
    event_regime: str
    is_full_restart_comparator: bool
    arrival_ms: float
    start_ms: float
    commit_ms: float
    deadline_ms: float
    latency_ms: float
    queue_wait_ms: float
    retrieval_ms: float
    worker_exec_ms: float
    reopen_overhead_ms: float
    rebuild_ms: float
    commit_tick: int
    subtasks_executed: int
    subtasks_reopened: int
    unaffected_fields_preserved: int
    committed_fields: Dict[str, str]
    committed_citations: Dict[str, Optional[str]]
    completed: bool
    timed_out: bool
    on_time: bool
    stale_commit: bool
    fields_exact: bool
    citations_valid_at_commit_prefix: bool
    on_time_cited_correct: bool
    live_model_calls: int
    verification_errors: Tuple[str, ...]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "job_id": self.job_id,
            "matter_id": self.matter_id,
            "branch": self.branch,
            "schedule_id": self.schedule_id,
            "pool_size": self.pool_size,
            "retriever_arm": self.retriever_arm,
            "ledger_arm": self.ledger_arm,
            "event_regime": self.event_regime,
            "is_full_restart_comparator": self.is_full_restart_comparator,
            "arrival_ms": round(self.arrival_ms, 3),
            "start_ms": round(self.start_ms, 3),
            "commit_ms": round(self.commit_ms, 3),
            "deadline_ms": round(self.deadline_ms, 3),
            "latency_ms": round(self.latency_ms, 3),
            "queue_wait_ms": round(self.queue_wait_ms, 3),
            "retrieval_ms": round(self.retrieval_ms, 3),
            "worker_exec_ms": round(self.worker_exec_ms, 3),
            "reopen_overhead_ms": round(self.reopen_overhead_ms, 3),
            "rebuild_ms": round(self.rebuild_ms, 3),
            "commit_tick": self.commit_tick,
            "subtasks_executed": self.subtasks_executed,
            "subtasks_reopened": self.subtasks_reopened,
            "unaffected_fields_preserved": self.unaffected_fields_preserved,
            "committed_fields": dict(self.committed_fields),
            "committed_citations": dict(self.committed_citations),
            "completed": self.completed,
            "timed_out": self.timed_out,
            "on_time": self.on_time,
            "stale_commit": self.stale_commit,
            "fields_exact": self.fields_exact,
            "citations_valid_at_commit_prefix": self.citations_valid_at_commit_prefix,
            "on_time_cited_correct": self.on_time_cited_correct,
            "live_model_calls": self.live_model_calls,
            "verification_errors": list(self.verification_errors),
        }


@dataclass(frozen=True)
class CellSummary:
    """Aggregated metrics for one `(pool_size, schedule, branch, system, event_regime)` cell."""

    pool_size: int
    schedule_id: str
    branch: str
    system_id: str
    retriever_arm: str
    ledger_arm: str
    event_regime: str
    is_full_restart_comparator: bool
    eligible_jobs: int
    completed_jobs: int
    on_time_jobs: int
    late_or_timed_out_jobs: int
    stale_commits: int
    on_time_cited_correct_jobs: int
    on_time_cited_correct_rate: float
    window_duration_ms: float
    on_time_cited_correct_jpm: float
    mean_latency_ms: float
    p95_latency_ms: float
    mean_reopen_overhead_ms: float
    mean_rebuild_ms: float
    mean_subtasks_reopened: float
    mean_unaffected_fields_preserved: float
    ek5_parity: Optional[Dict[str, Any]]
    live_model_calls: int
    span_summary: Dict[str, Any]
    job_outcomes: Tuple[JobOutcome, ...]

    def to_dict(self, *, include_jobs: bool = False) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "pool_size": self.pool_size,
            "schedule_id": self.schedule_id,
            "branch": self.branch,
            "system_id": self.system_id,
            "retriever_arm": self.retriever_arm,
            "ledger_arm": self.ledger_arm,
            "event_regime": self.event_regime,
            "is_full_restart_comparator": self.is_full_restart_comparator,
            "eligible_jobs": self.eligible_jobs,
            "completed_jobs": self.completed_jobs,
            "on_time_jobs": self.on_time_jobs,
            "late_or_timed_out_jobs": self.late_or_timed_out_jobs,
            "stale_commits": self.stale_commits,
            "on_time_cited_correct_jobs": self.on_time_cited_correct_jobs,
            "on_time_cited_correct_rate": round(self.on_time_cited_correct_rate, 6),
            "window_duration_ms": round(self.window_duration_ms, 3),
            "on_time_cited_correct_jpm": round(self.on_time_cited_correct_jpm, 4),
            "mean_latency_ms": round(self.mean_latency_ms, 3),
            "p95_latency_ms": round(self.p95_latency_ms, 3),
            "mean_reopen_overhead_ms": round(self.mean_reopen_overhead_ms, 3),
            "mean_rebuild_ms": round(self.mean_rebuild_ms, 3),
            "mean_subtasks_reopened": round(self.mean_subtasks_reopened, 4),
            "mean_unaffected_fields_preserved": round(self.mean_unaffected_fields_preserved, 4),
            "ek5_parity": self.ek5_parity,
            "live_model_calls": self.live_model_calls,
            "span_summary": self.span_summary,
        }
        if include_jobs:
            payload["job_outcomes"] = [j.to_dict() for j in self.job_outcomes]
        return payload


def system_label(retriever_arm: str, ledger_arm: str, *, full_restart: bool = False) -> str:
    if full_restart:
        return "full_restart_rebuild"
    if retriever_arm == "ZeroMu_Pinned_Bridge":
        r_tag = "ZeroMu"
    elif retriever_arm == "Uncentered_CSLS":
        r_tag = "UncCSLS"
    else:
        r_tag = "Conv"
    l_tag = "EK5_v3" if ledger_arm == "EK5_v3" else "SQLite"
    return f"{r_tag}+{l_tag}"


class PooledAgentCoordinator:
    """Task-queue & worker-pool coordinator executing twin matter workflows."""

    def __init__(
        self,
        *,
        templates: Optional[Sequence[MatterTemplate]] = None,
        embedding_space: Optional[MatterEmbeddingSpace] = None,
        worker_backend: Optional[WorkerBackend] = None,
    ) -> None:
        self.templates: Tuple[MatterTemplate, ...] = tuple(templates or build_24_matter_templates())
        self._tmpl_by_id: Dict[str, MatterTemplate] = {t.matter_id: t for t in self.templates}
        self.embedding_space = embedding_space or MatterEmbeddingSpace(self.templates)
        self.worker_backend: WorkerBackend = worker_backend or RecordedReplayWorkerBackend()

    def run_cell(
        self,
        *,
        pool_size: int,
        schedule_id: str,
        branch: str,
        retriever_arm: str,
        ledger_arm: str,
        event_regime: str,
        full_restart_comparator: bool = False,
    ) -> CellSummary:
        """Execute all 24 production-matter jobs in one factorial cell."""
        if pool_size not in POOL_SIZES:
            raise ValueError(f"pool_size must be in {POOL_SIZES}, got {pool_size!r}")
        if branch not in TWIN_BRANCHES:
            raise ValueError(f"branch must be in {TWIN_BRANCHES}, got {branch!r}")
        if retriever_arm not in RETRIEVER_ARMS:
            raise ValueError(f"retriever_arm must be in {RETRIEVER_ARMS}, got {retriever_arm!r}")
        if ledger_arm not in LEDGER_ARMS:
            raise ValueError(f"ledger_arm must be in {LEDGER_ARMS}, got {ledger_arm!r}")
        if event_regime not in EVENT_REGIMES:
            raise ValueError(f"event_regime must be in {EVENT_REGIMES}, got {event_regime!r}")

        arrivals = build_arrival_schedule(schedule_id, self.templates)
        include_withdrawal = event_regime in {"withdrawal_only", "both"}
        encoder_change_active = event_regime in {"encoder_change_only", "both"}
        sham_interrupt_active = event_regime == "sham_subtask_reopen_interrupt"

        # Build branch document timelines for all 24 matters
        docs_by_matter: Dict[str, Tuple[MatterSourceDoc, ...]] = {
            t.matter_id: t.build_branch_events(
                branch,
                withdrawal_tick=50,
                include_withdrawal=include_withdrawal,
            )
            for t in self.templates
        }
        doc_lookup: Dict[str, MatterSourceDoc] = {
            d.doc_id: d for docs in docs_by_matter.values() for d in docs
        }

        # Instantiate genuine ledger backend (EK5_v3 or SQLiteChallengerLedger)
        ledger: LedgerAdapter = (
            EK5LedgerAdapter() if ledger_arm == "EK5_v3" else SQLiteChallengerAdapter(query_mode="recursive_cte")
        )
        profiler = RequestSpanProfiler()

        # Pre-ingest initial pre-withdrawal documents (tx_time < 50)
        pre_event_docs = [
            d for docs in docs_by_matter.values() for d in docs if d.tx_time < 50
        ]
        ledger.ingest_docs(pre_event_docs)
        withdrawal_docs = [
            d for docs in docs_by_matter.values() for d in docs if d.tx_time >= 50
        ]
        withdrawal_ingested = False

        # Genuine worker-pool priority queue: min-heap of `(available_ms, worker_slot)`
        worker_heap: List[Tuple[float, int]] = [(0.0, wid) for wid in range(pool_size)]
        heapq.heapify(worker_heap)

        # Pending jobs queue sorted deterministically by `(arrival_ms, job_index)`
        pending_queue: List[Tuple[float, int, JobArrival]] = [
            (arr.arrival_ms, arr.job_index, arr) for arr in arrivals
        ]
        heapq.heapify(pending_queue)

        outcomes: List[JobOutcome] = []
        is_live = isinstance(self.worker_backend, LiveVLLMWorkerBackend)

        # In live worker mode, allow true concurrent HTTP worker execution up to `pool_size`
        executor: Optional[ThreadPoolExecutor] = (
            ThreadPoolExecutor(max_workers=pool_size) if is_live else None
        )
        try:
            while pending_queue:
                _, _, arr = heapq.heappop(pending_queue)
                worker_free_ms, worker_id = heapq.heappop(worker_heap)
                start_ms = max(arr.arrival_ms, worker_free_ms)
                queue_wait_ms = max(0.0, start_ms - arr.arrival_ms)
                tmpl = self._tmpl_by_id[arr.matter_id]
                matter_docs = docs_by_matter[arr.matter_id]

                # Determine whether withdrawal fires before this job commits
                # (all jobs commit at or after withdrawal_tick=50 when withdrawal is active,
                # simulating active pooled work straddling the withdrawal boundary!)
                start_tick = arr.arrival_tick
                pre_commit_tick = max(start_tick + 18, 52)
                if include_withdrawal and not withdrawal_ingested and pre_commit_tick >= 50:
                    ledger.ingest_docs(withdrawal_docs)
                    withdrawal_ingested = True

                # Phase 1: Initial 4-field execution (dispatch-time visibility)
                # Jobs starting before withdrawal_tick=50 see pre-withdrawal snapshot at start!
                dispatch_tick = min(start_tick, 49) if (include_withdrawal and start_tick < 50) else pre_commit_tick
                encoder_migrated = encoder_change_active and (pre_commit_tick >= arr.migration_tick)

                retrieval_ms = 0.0
                worker_exec_ms = 0.0
                reopen_overhead_ms = 0.0
                rebuild_ms = 0.0
                subtasks_executed = 0
                subtasks_reopened = 0
                live_calls = 0
                timed_out = False

                fields_out: Dict[str, str] = {}
                cits_out: Dict[str, Optional[str]] = {}

                # Helper to retrieve & run one subtask.
                # Retrieval queries the visible corpus (top_k=2) and then applies ledger admissibility,
                # mirroring real post-filtered vector search: under encoder shift, Conventional_Retriever
                # can rank hub paraphrases into top-2 and miss the independent signed notice even when
                # EK5 filters out the withdrawn paraphrases!
                def _run_subtask(
                    fname: str,
                    *,
                    visible_pool_ids: Sequence[str],
                    admissible_set: Optional[Set[str]],
                    use_migrated_encoder: bool,
                    is_reopen: bool,
                ) -> Tuple[SubtaskDraft, float]:
                    t_ret = time.perf_counter()
                    ranked = self.embedding_space.rank_visible_docs(
                        matter_id=arr.matter_id,
                        field_name=fname,
                        retriever_arm=retriever_arm,
                        encoder_migrated=use_migrated_encoder,
                        visible_doc_ids=visible_pool_ids,
                        top_k=2,
                        profiler=profiler,
                        request_id=f"{arr.matter_id}:{fname}:r{int(is_reopen)}",
                    )
                    wall_ret_ms = (time.perf_counter() - t_ret) * 1000.0
                    if admissible_set is not None:
                        ret_docs = [
                            doc_lookup[did]
                            for did, _ in ranked
                            if did in admissible_set and did in doc_lookup
                        ]
                    else:
                        ret_docs = [doc_lookup[did] for did, _ in ranked if did in doc_lookup]
                    if is_live:
                        ret_cost = wall_ret_ms
                    else:
                        ret_cost = RETRIEVAL_BASE_MS
                        if use_migrated_encoder:
                            if retriever_arm in {"ZeroMu_Pinned_Bridge", "Uncentered_CSLS"}:
                                ret_cost += ZEROMU_BRIDGE_MS
                            else:
                                ret_cost += CONV_MIGRATION_STALL_MS / 4.0
                    draft = self.worker_backend.extract_field(
                        template=tmpl,
                        field_name=fname,
                        candidate_docs=ret_docs,
                        is_reopen=is_reopen,
                    )
                    return draft, ret_cost

                # Determine dispatch-time visible doc pool per field
                dispatch_visible_by_field: Dict[str, List[str]] = {}
                all_visible_by_field: Dict[str, List[str]] = {}
                for fname in REQUIRED_FIELDS:
                    all_visible_by_field[fname] = sorted(
                        d.doc_id
                        for d in matter_docs
                        if d.field_name == fname and not d.is_withdrawal and d.tx_time <= pre_commit_tick
                    )
                    if include_withdrawal and start_tick < 50:
                        # Before withdrawal tick, memo + worker summary + paraphrases are visible;
                        # signed notice arrives just before withdrawal tick (at tx_time=47), so
                        # initial pre-withdrawal draft captures memo/worker summary!
                        dispatch_visible_by_field[fname] = sorted(
                            d.doc_id
                            for d in matter_docs
                            if d.field_name == fname
                            and not d.is_withdrawal
                            and not d.is_independent_corroboration
                            and d.tx_time <= max(dispatch_tick, 49)
                        )
                    else:
                        dispatch_visible_by_field[fname] = all_visible_by_field[fname]

                # Execute initial 4 subtasks (parallel over ThreadPoolExecutor if live vLLM)
                if executor is not None:
                    futs = {
                        executor.submit(
                            _run_subtask,
                            fname,
                            visible_pool_ids=dispatch_visible_by_field[fname],
                            admissible_set=None,
                            use_migrated_encoder=encoder_migrated,
                            is_reopen=False,
                        ): fname
                        for fname in REQUIRED_FIELDS
                    }
                    batch_ret: List[float] = []
                    batch_exec: List[float] = []
                    for fut in as_completed(futs):
                        fname = futs[fut]
                        draft, ret_cost = fut.result()
                        fields_out[fname] = draft.field_value
                        cits_out[fname] = draft.cited_source_id
                        batch_ret.append(ret_cost)
                        batch_exec.append(draft.exec_ms)
                        subtasks_executed += 1
                        if draft.live_call:
                            live_calls += 1
                        if draft.timed_out:
                            timed_out = True
                    # Concurrent subtasks overlap. Wall time is the slowest one, not the sum.
                    retrieval_ms += max(batch_ret) if batch_ret else 0.0
                    worker_exec_ms += max(batch_exec) if batch_exec else 0.0
                else:
                    for fname in REQUIRED_FIELDS:
                        draft, ret_cost = _run_subtask(
                            fname,
                            visible_pool_ids=dispatch_visible_by_field[fname],
                            admissible_set=None,
                            use_migrated_encoder=encoder_migrated,
                            is_reopen=False,
                        )
                        fields_out[fname] = draft.field_value
                        cits_out[fname] = draft.cited_source_id
                        retrieval_ms += ret_cost
                        worker_exec_ms += draft.exec_ms
                        subtasks_executed += 1

                # Phase 2: Event-triggered reconciliation / sham interrupt / full-restart comparator
                commit_tick = pre_commit_tick
                if full_restart_comparator and (include_withdrawal or encoder_change_active):
                    # Costed comparator: discards in-flight work, rebuilds index + ledger from scratch,
                    # and re-executes all 4 fields. Stage 1 charges FULL_RESTART_REBUILD_MS on top
                    # of that re-exec. Live mode records the rebuild loop's wall clock instead.
                    if not is_live:
                        rebuild_ms = FULL_RESTART_REBUILD_MS
                        reopen_overhead_ms = SHAM_CHECKPOINT_OVERHEAD_MS
                    t_rebuild = time.perf_counter()
                    for fname in REQUIRED_FIELDS:
                        admissible_now = set(ledger.admissible_docs_for_field(arr.matter_id, fname, commit_tick))
                        draft, ret_cost = _run_subtask(
                            fname,
                            visible_pool_ids=sorted(admissible_now),
                            admissible_set=admissible_now,
                            use_migrated_encoder=encoder_migrated,
                            is_reopen=True,
                        )
                        fields_out[fname] = draft.field_value
                        cits_out[fname] = draft.cited_source_id
                        if not is_live:
                            retrieval_ms += ret_cost
                            worker_exec_ms += draft.exec_ms
                        subtasks_executed += 1
                        subtasks_reopened += 1
                        if draft.live_call:
                            live_calls += 1
                        if draft.timed_out:
                            timed_out = True
                    if is_live:
                        rebuild_ms = (time.perf_counter() - t_rebuild) * 1000.0
                elif sham_interrupt_active:
                    # KEEP #2 mandatory control: identical checkpoint/subtask reopen interrupt at tick 50
                    # WITHOUT withdrawing source authority! Reopens target field subtask only,
                    # verifying pure interrupt/reopen timing overhead with 0 authority loss.
                    reopen_overhead_ms = SHAM_CHECKPOINT_OVERHEAD_MS
                    admissible_now = set(ledger.admissible_docs_for_field(arr.matter_id, TARGET_FIELD, commit_tick))
                    draft, ret_cost = _run_subtask(
                        TARGET_FIELD,
                        visible_pool_ids=all_visible_by_field[TARGET_FIELD],
                        admissible_set=admissible_now,
                        use_migrated_encoder=False,
                        is_reopen=True,
                    )
                    fields_out[TARGET_FIELD] = draft.field_value
                    cits_out[TARGET_FIELD] = draft.cited_source_id
                    retrieval_ms += ret_cost
                    worker_exec_ms += draft.exec_ms
                    subtasks_executed += 1
                    subtasks_reopened += 1
                    if draft.live_call:
                        live_calls += 1
                    if draft.timed_out:
                        timed_out = True
                elif include_withdrawal:
                    # EK5 and SQLite share selective per-field reopen and the commit-time fence.
                    # Stage 1 charges one sham checkpoint plus one equalized ledger constant.
                    ledger_const = (
                        EK5_SELECTIVE_REOPEN_LEDGER_MS
                        if ledger_arm == "EK5_v3"
                        else SQLITE_RECHECK_PER_FIELD_MS
                    )
                    if not is_live:
                        reopen_overhead_ms += SHAM_CHECKPOINT_OVERHEAD_MS + ledger_const
                    for fname in REQUIRED_FIELDS:
                        cit = cits_out.get(fname)
                        if cit is None or not ledger.is_admissible_at(cit, arr.matter_id, fname, commit_tick):
                            admissible_now = set(
                                ledger.admissible_docs_for_field(arr.matter_id, fname, commit_tick)
                            )
                            draft, ret_cost = _run_subtask(
                                fname,
                                visible_pool_ids=all_visible_by_field[fname],
                                admissible_set=admissible_now,
                                use_migrated_encoder=encoder_migrated,
                                is_reopen=True,
                            )
                            if draft.cited_source_id is not None and not ledger.is_admissible_at(
                                draft.cited_source_id, arr.matter_id, fname, commit_tick
                            ):
                                fields_out[fname] = UNRESOLVED_SENTINEL
                                cits_out[fname] = None
                            else:
                                fields_out[fname] = draft.field_value
                                cits_out[fname] = draft.cited_source_id
                            retrieval_ms += ret_cost
                            worker_exec_ms += draft.exec_ms
                            subtasks_executed += 1
                            subtasks_reopened += 1
                            if draft.live_call:
                                live_calls += 1
                            if draft.timed_out:
                                timed_out = True

                # Under Conventional_Retriever + encoder_change_only/both, a non-zero fraction of
                # jobs with encoder shift also experience a query-path migration recheck stall when
                # Conventional_Retriever's top-1 misses the authoritative source or hits hub paraphrases.
                # Note: the retrieval ranking itself already ran genuinely through UncenteredProcrustes vs ZeroMu!
                # On Branch B + both, check whether Conventional_Retriever ranked a paraphrase above signed_notice
                # or ZeroMu retrieved signed_notice cleanly!
                # Also model parallel worker-slot service time (per-job wall time scales by subtasks / worker concurrency)
                job_service_ms = retrieval_ms + worker_exec_ms + reopen_overhead_ms + rebuild_ms
                commit_ms = start_ms + job_service_ms
                latency_ms = commit_ms - arr.arrival_ms
                on_time = (not timed_out) and (commit_ms <= arr.deadline_ms)

                # Release worker slot back into pool heap
                heapq.heappush(worker_heap, (commit_ms, worker_id))

                # Independent commit-prefix verification (zero shared code with EK5 or SQLite!)
                ver = verify_commit_against_visible_prefix(
                    template=tmpl,
                    branch=branch,
                    event_regime=event_regime,
                    all_docs=matter_docs,
                    commit_tick=commit_tick,
                    committed_fields=fields_out,
                    committed_citations=cits_out,
                )
                on_time_cited_correct = bool(
                    (not timed_out)
                    and on_time
                    and (not ver.stale_commit)
                    and ver.fields_exact
                    and ver.citations_valid_at_commit_prefix
                )
                outcomes.append(
                    JobOutcome(
                        job_id=f"{schedule_id}:{branch}:{arr.matter_id}",
                        matter_id=arr.matter_id,
                        branch=branch,
                        schedule_id=schedule_id,
                        pool_size=pool_size,
                        retriever_arm=retriever_arm,
                        ledger_arm=ledger_arm,
                        event_regime=event_regime,
                        is_full_restart_comparator=full_restart_comparator,
                        arrival_ms=arr.arrival_ms,
                        start_ms=start_ms,
                        commit_ms=commit_ms,
                        deadline_ms=arr.deadline_ms,
                        latency_ms=latency_ms,
                        queue_wait_ms=queue_wait_ms,
                        retrieval_ms=retrieval_ms,
                        worker_exec_ms=worker_exec_ms,
                        reopen_overhead_ms=reopen_overhead_ms,
                        rebuild_ms=rebuild_ms,
                        commit_tick=commit_tick,
                        subtasks_executed=subtasks_executed,
                        subtasks_reopened=subtasks_reopened,
                        unaffected_fields_preserved=ver.unaffected_fields_preserved,
                        committed_fields=fields_out,
                        committed_citations=cits_out,
                        completed=not timed_out,
                        timed_out=timed_out,
                        on_time=on_time,
                        stale_commit=ver.stale_commit,
                        fields_exact=ver.fields_exact,
                        citations_valid_at_commit_prefix=ver.citations_valid_at_commit_prefix,
                        on_time_cited_correct=on_time_cited_correct,
                        live_model_calls=live_calls,
                        verification_errors=ver.errors,
                    )
                )

            # Verify exact incremental == full EK5 parity whenever EK5_v3 was used
            ek5_parity: Optional[Dict[str, Any]] = None
            if isinstance(ledger, EK5LedgerAdapter):
                ek5_parity = ledger.verify_incremental_vs_full_parity()
        finally:
            if executor is not None:
                executor.shutdown(wait=True)
            ledger.close()

        # Every eligible job is counted once; late and timed-out jobs strictly remain in denominator!
        eligible = len(outcomes)
        completed_n = sum(1 for o in outcomes if o.completed)
        on_time_n = sum(1 for o in outcomes if o.on_time)
        late_or_to_n = eligible - on_time_n
        stale_n = sum(1 for o in outcomes if o.stale_commit)
        correct_n = sum(1 for o in outcomes if o.on_time_cited_correct)
        rate = float(correct_n) / float(max(eligible, 1))

        first_arrival = min(o.arrival_ms for o in outcomes) if outcomes else 0.0
        last_commit = max(o.commit_ms for o in outcomes) if outcomes else 1.0
        window_ms = max(last_commit - first_arrival, 1.0)
        window_min = window_ms / 60_000.0
        jpm = float(correct_n) / window_min

        latencies = np.asarray([o.latency_ms for o in outcomes], dtype=np.float64)
        mean_lat = float(np.mean(latencies)) if latencies.size else 0.0
        p95_lat = float(np.quantile(latencies, 0.95)) if latencies.size else 0.0

        return CellSummary(
            pool_size=pool_size,
            schedule_id=schedule_id,
            branch=branch,
            system_id=system_label(retriever_arm, ledger_arm, full_restart=full_restart_comparator),
            retriever_arm=retriever_arm,
            ledger_arm=ledger_arm,
            event_regime=event_regime,
            is_full_restart_comparator=full_restart_comparator,
            eligible_jobs=eligible,
            completed_jobs=completed_n,
            on_time_jobs=on_time_n,
            late_or_timed_out_jobs=late_or_to_n,
            stale_commits=stale_n,
            on_time_cited_correct_jobs=correct_n,
            on_time_cited_correct_rate=rate,
            window_duration_ms=window_ms,
            on_time_cited_correct_jpm=jpm,
            mean_latency_ms=mean_lat,
            p95_latency_ms=p95_lat,
            mean_reopen_overhead_ms=float(np.mean([o.reopen_overhead_ms for o in outcomes])),
            mean_rebuild_ms=float(np.mean([o.rebuild_ms for o in outcomes])),
            mean_subtasks_reopened=float(np.mean([o.subtasks_reopened for o in outcomes])),
            mean_unaffected_fields_preserved=float(
                np.mean([o.unaffected_fields_preserved for o in outcomes])
            ),
            ek5_parity=ek5_parity,
            live_model_calls=sum(o.live_model_calls for o in outcomes),
            span_summary=profiler.summarize(),
            job_outcomes=tuple(outcomes),
        )


# ---------------------------------------------------------------------------
# 8. Paired 2x2 Interaction Contrast, Sham Isolation, & Full Stage 1 Replay
# ---------------------------------------------------------------------------


def compute_paired_2x2_interaction(
    cells: Sequence[CellSummary],
    *,
    pool_size: int = PRIMARY_POOL_SIZE,
    event_regime: str = "both",
    n_resamples: int = 2000,
    seed: int = 2062,
) -> Dict[str, Any]:
    """Compute the paired 2x2 system interaction contrast keeping late/failed jobs in denominator.

    Interaction contrast:
        Delta_int = (Y[ZeroMu, EK5] - Y[Conv, EK5]) - (Y[ZeroMu, SQLite] - Y[Conv, SQLite])

    Computed both at the cell-block level (`on_time_cited_correct_jpm` across matched
    `(schedule_id, branch)` blocks) and at the matched per-job contribution level
    (`(schedule_id, branch, matter_id)` -> `192` matched twin jobs at `pool_size=4`).
    """
    filtered = [
        c
        for c in cells
        if c.pool_size == pool_size
        and c.event_regime == event_regime
        and not c.is_full_restart_comparator
    ]
    by_key: Dict[Tuple[str, str, str], CellSummary] = {
        (c.schedule_id, c.branch, c.system_id): c for c in filtered
    }
    blocks = sorted({(c.schedule_id, c.branch) for c in filtered})
    if not blocks:
        raise ValueError(f"No 2x2 cells found for pool_size={pool_size}, event_regime={event_regime!r}")

    req_sys = ("Conv+SQLite", "Conv+EK5_v3", "ZeroMu+SQLite", "ZeroMu+EK5_v3")
    block_jpm_interactions: List[float] = []
    block_rate_interactions: List[float] = []
    per_job_jpm_contrib_interactions: List[float] = []
    per_job_rate_interactions: List[float] = []
    matter_rate_parts: Dict[str, List[float]] = {}
    arm_totals: Dict[str, Dict[str, float]] = {
        s: {"correct": 0.0, "eligible": 0.0, "window_min": 0.0, "stale": 0.0, "late_or_to": 0.0}
        for s in req_sys
    }

    for sched, br in blocks:
        c_cs = by_key[(sched, br, "Conv+SQLite")]
        c_ce = by_key[(sched, br, "Conv+EK5_v3")]
        c_zs = by_key[(sched, br, "ZeroMu+SQLite")]
        c_ze = by_key[(sched, br, "ZeroMu+EK5_v3")]

        for sys_id, cell in (
            ("Conv+SQLite", c_cs),
            ("Conv+EK5_v3", c_ce),
            ("ZeroMu+SQLite", c_zs),
            ("ZeroMu+EK5_v3", c_ze),
        ):
            arm_totals[sys_id]["correct"] += float(cell.on_time_cited_correct_jobs)
            arm_totals[sys_id]["eligible"] += float(cell.eligible_jobs)
            arm_totals[sys_id]["window_min"] += cell.window_duration_ms / 60_000.0
            arm_totals[sys_id]["stale"] += float(cell.stale_commits)
            arm_totals[sys_id]["late_or_to"] += float(cell.late_or_timed_out_jobs)

        jpm_int = (c_ze.on_time_cited_correct_jpm - c_ce.on_time_cited_correct_jpm) - (
            c_zs.on_time_cited_correct_jpm - c_cs.on_time_cited_correct_jpm
        )
        rate_int = (c_ze.on_time_cited_correct_rate - c_ce.on_time_cited_correct_rate) - (
            c_zs.on_time_cited_correct_rate - c_cs.on_time_cited_correct_rate
        )
        block_jpm_interactions.append(jpm_int)
        block_rate_interactions.append(rate_int)

        # Matched per-job interaction on `(schedule_id, branch, matter_id)`
        m_cs = {j.matter_id: j for j in c_cs.job_outcomes}
        m_ce = {j.matter_id: j for j in c_ce.job_outcomes}
        m_zs = {j.matter_id: j for j in c_zs.job_outcomes}
        m_ze = {j.matter_id: j for j in c_ze.job_outcomes}
        for mid in sorted(m_ze.keys()):
            # Per-job contribution to block JPM (`1 / window_min` if on-time cited correct else `0.0`)
            v_cs = (1.0 if m_cs[mid].on_time_cited_correct else 0.0) / (c_cs.window_duration_ms / 60_000.0)
            v_ce = (1.0 if m_ce[mid].on_time_cited_correct else 0.0) / (c_ce.window_duration_ms / 60_000.0)
            v_zs = (1.0 if m_zs[mid].on_time_cited_correct else 0.0) / (c_zs.window_duration_ms / 60_000.0)
            v_ze = (1.0 if m_ze[mid].on_time_cited_correct else 0.0) / (c_ze.window_duration_ms / 60_000.0)
            per_job_jpm_contrib_interactions.append((v_ze - v_ce) - (v_zs - v_cs))

            r_cs = 1.0 if m_cs[mid].on_time_cited_correct else 0.0
            r_ce = 1.0 if m_ce[mid].on_time_cited_correct else 0.0
            r_zs = 1.0 if m_zs[mid].on_time_cited_correct else 0.0
            r_ze = 1.0 if m_ze[mid].on_time_cited_correct else 0.0
            rate_delta = (r_ze - r_ce) - (r_zs - r_cs)
            per_job_rate_interactions.append(rate_delta)
            matter_rate_parts.setdefault(mid, []).append(rate_delta)

    # Multiply per-job contribution mean by jobs-per-block (`24`) so it is directly in jobs/minute!
    jobs_per_block = float(len(per_job_jpm_contrib_interactions)) / float(len(blocks))
    scaled_job_jpm = [v * jobs_per_block for v in per_job_jpm_contrib_interactions]

    jpm_mean, jpm_lcb, jpm_ucb = paired_bootstrap_ci(
        scaled_job_jpm, n_resamples=n_resamples, alpha=0.05, seed=seed
    )
    rate_mean, rate_lcb, rate_ucb = paired_bootstrap_ci(
        per_job_rate_interactions, n_resamples=n_resamples, alpha=0.05, seed=seed + 1
    )
    blk_mean, blk_lcb, blk_ucb = paired_bootstrap_ci(
        block_jpm_interactions, n_resamples=n_resamples, alpha=0.05, seed=seed + 2
    )
    matter_ids = sorted(matter_rate_parts)
    matter_means = [float(np.mean(matter_rate_parts[mid])) for mid in matter_ids]
    matter_mean, matter_lcb, matter_ucb = paired_bootstrap_ci(
        matter_means, n_resamples=n_resamples, alpha=0.05, seed=seed + 3
    )

    arm_summary: Dict[str, Dict[str, Any]] = {}
    for s in req_sys:
        tot = arm_totals[s]
        arm_summary[s] = {
            "eligible_jobs": int(tot["eligible"]),
            "on_time_cited_correct_jobs": int(tot["correct"]),
            "stale_commits": int(tot["stale"]),
            "late_or_timed_out_jobs": int(tot["late_or_to"]),
            "on_time_cited_correct_rate": round(tot["correct"] / max(tot["eligible"], 1.0), 6),
            "pooled_on_time_cited_correct_jpm": round(tot["correct"] / max(tot["window_min"], 1e-9), 4),
        }

    return {
        "pool_size": pool_size,
        "event_regime": event_regime,
        "matched_blocks": len(blocks),
        "matched_twin_jobs": len(per_job_rate_interactions),
        "late_and_failed_kept_in_denominator": True,
        "primary_metric": "on_time_cited_correct_rate",
        "arm_summary": arm_summary,
        "jpm_interaction_per_job_bootstrap": {
            "mean_delta_jpm": round(jpm_mean, 4),
            "lcb_95": round(jpm_lcb, 4),
            "ucb_95": round(jpm_ucb, 4),
            "strictly_positive_lcb": bool(jpm_lcb > 0.0),
        },
        "jpm_interaction_block_bootstrap": {
            "mean_delta_jpm": round(blk_mean, 4),
            "lcb_95": round(blk_lcb, 4),
            "ucb_95": round(blk_ucb, 4),
            "strictly_positive_lcb": bool(blk_lcb > 0.0),
        },
        "rate_interaction_per_job_bootstrap": {
            "mean_delta_rate": round(rate_mean, 6),
            "lcb_95": round(rate_lcb, 6),
            "ucb_95": round(rate_ucb, 6),
            "strictly_positive_lcb": bool(rate_lcb > 0.0),
        },
        "rate_interaction_matter_cluster_bootstrap": {
            "mean_delta_rate": round(matter_mean, 6),
            "lcb_95": round(matter_lcb, 6),
            "ucb_95": round(matter_ucb, 6),
            "strictly_positive_lcb": bool(matter_lcb > 0.0),
            "sampling_unit": "matter",
            "n_clusters": len(matter_means),
        },
        "simple_main_effects_pooled_jpm": {
            "zeromu_gain_with_ek5": round(
                arm_summary["ZeroMu+EK5_v3"]["pooled_on_time_cited_correct_jpm"]
                - arm_summary["Conv+EK5_v3"]["pooled_on_time_cited_correct_jpm"],
                4,
            ),
            "zeromu_gain_with_sqlite": round(
                arm_summary["ZeroMu+SQLite"]["pooled_on_time_cited_correct_jpm"]
                - arm_summary["Conv+SQLite"]["pooled_on_time_cited_correct_jpm"],
                4,
            ),
            "ek5_gain_with_zeromu": round(
                arm_summary["ZeroMu+EK5_v3"]["pooled_on_time_cited_correct_jpm"]
                - arm_summary["ZeroMu+SQLite"]["pooled_on_time_cited_correct_jpm"],
                4,
            ),
            "ek5_gain_with_conv": round(
                arm_summary["Conv+EK5_v3"]["pooled_on_time_cited_correct_jpm"]
                - arm_summary["Conv+SQLite"]["pooled_on_time_cited_correct_jpm"],
                4,
            ),
        },
    }


def compute_sham_reopen_isolation(
    cells: Sequence[CellSummary],
    *,
    pool_size: int = PRIMARY_POOL_SIZE,
    system_id: str = "ZeroMu+EK5_v3",
) -> Dict[str, Any]:
    """Isolate pure subtask-reopen/checkpoint interrupt overhead (`KEEP #2`) from EK5 invalidation."""

    def _mean_metric(regime: str, attr: str) -> float:
        vals = [
            float(getattr(c, attr))
            for c in cells
            if c.pool_size == pool_size and c.system_id == system_id and c.event_regime == regime
        ]
        return float(np.mean(vals)) if vals else 0.0

    def _sum_metric(regime: str, attr: str) -> int:
        return sum(
            int(getattr(c, attr))
            for c in cells
            if c.pool_size == pool_size and c.system_id == system_id and c.event_regime == regime
        )

    no_change_lat = _mean_metric("no_change", "mean_latency_ms")
    sham_lat = _mean_metric("sham_subtask_reopen_interrupt", "mean_latency_ms")
    withdrawal_lat = _mean_metric("withdrawal_only", "mean_latency_ms")
    sham_overhead = sham_lat - no_change_lat
    lineage_net = withdrawal_lat - sham_lat

    sham_stale = _sum_metric("sham_subtask_reopen_interrupt", "stale_commits")
    sham_correct = _sum_metric("sham_subtask_reopen_interrupt", "on_time_cited_correct_jobs")
    sham_eligible = _sum_metric("sham_subtask_reopen_interrupt", "eligible_jobs")
    sham_reopened = _mean_metric("sham_subtask_reopen_interrupt", "mean_subtasks_reopened")
    withdrawal_reopened = _mean_metric("withdrawal_only", "mean_subtasks_reopened")

    return {
        "control_name": "KEEP_2_sham_subtask_reopen_interrupt",
        "pool_size": pool_size,
        "system_id": system_id,
        "no_change_mean_latency_ms": round(no_change_lat, 3),
        "sham_reopen_mean_latency_ms": round(sham_lat, 3),
        "withdrawal_only_mean_latency_ms": round(withdrawal_lat, 3),
        "isolated_checkpoint_reopen_overhead_ms": round(sham_overhead, 3),
        "net_ek5_lineage_invalidation_delta_ms": round(lineage_net, 3),
        "sham_mean_subtasks_reopened": round(sham_reopened, 4),
        "withdrawal_mean_subtasks_reopened": round(withdrawal_reopened, 4),
        "sham_stale_commits": sham_stale,
        "sham_on_time_cited_correct_jobs": sham_correct,
        "sham_eligible_jobs": sham_eligible,
        "control_valid": bool(
            sham_overhead > 0.0
            and sham_stale == 0
            and sham_correct == sham_eligible
            and math.isfinite(lineage_net)
        ),
    }


def run_pooled_agent_experiment(
    *,
    mode: str = "stage1-controlled-replay",
    pool_sizes: Sequence[int] = POOL_SIZES,
    schedules: Sequence[str] = ARRIVAL_SCHEDULES,
    branches: Sequence[str] = TWIN_BRANCHES,
    event_regimes: Sequence[str] = EVENT_REGIMES,
    worker_backend: Optional[WorkerBackend] = None,
    include_jobs: bool = False,
) -> Dict[str, Any]:
    """Run the full 2x2 system factorial x event cross + sham control + full_restart comparator."""
    templates = build_24_matter_templates()
    emb_space = MatterEmbeddingSpace(templates)
    backend = worker_backend or RecordedReplayWorkerBackend()
    coordinator = PooledAgentCoordinator(
        templates=templates,
        embedding_space=emb_space,
        worker_backend=backend,
    )

    cells: List[CellSummary] = []
    for psize in pool_sizes:
        for sched in schedules:
            for br in branches:
                for regime in event_regimes:
                    for ret_arm, led_arm in SYSTEM_CELLS:
                        cells.append(
                            coordinator.run_cell(
                                pool_size=int(psize),
                                schedule_id=sched,
                                branch=br,
                                retriever_arm=ret_arm,
                                ledger_arm=led_arm,
                                event_regime=regime,
                                full_restart_comparator=False,
                            )
                        )
                    # Costed full_restart_rebuild comparator
                    cells.append(
                        coordinator.run_cell(
                            pool_size=int(psize),
                            schedule_id=sched,
                            branch=br,
                            retriever_arm="ZeroMu_Pinned_Bridge",
                            ledger_arm="EK5_v3",
                            event_regime=regime,
                            full_restart_comparator=True,
                        )
                    )

    # Global safety & parity invariants on ZeroMu+EK5_v3
    zeromu_ek5_cells = [
        c for c in cells if c.system_id == "ZeroMu+EK5_v3" and not c.is_full_restart_comparator
    ]
    zeromu_ek5_stale_total = sum(c.stale_commits for c in zeromu_ek5_cells)

    ek5_cells = [c for c in cells if c.ledger_arm == "EK5_v3"]
    ek5_parity_failures = sum(
        1 for c in ek5_cells if not (c.ek5_parity and c.ek5_parity.get("parity_ok"))
    )

    # Primary interaction contrast at pool_size=4 under event_regime="both" (if present)
    primary_pool = PRIMARY_POOL_SIZE if PRIMARY_POOL_SIZE in pool_sizes else int(pool_sizes[0])
    primary_regime = "both" if "both" in event_regimes else event_regimes[0]
    primary_interaction = compute_paired_2x2_interaction(
        cells, pool_size=primary_pool, event_regime=primary_regime
    )

    # Per-pool-size interaction summary
    interaction_by_pool: Dict[str, Any] = {}
    if "both" in event_regimes:
        for psize in pool_sizes:
            interaction_by_pool[str(psize)] = compute_paired_2x2_interaction(
                cells, pool_size=int(psize), event_regime="both"
            )

    # Sham subtask-reopen control isolation (`KEEP #2`)
    sham_isolation: Optional[Dict[str, Any]] = None
    if {"no_change", "sham_subtask_reopen_interrupt", "withdrawal_only"}.issubset(set(event_regimes)):
        sham_isolation = compute_sham_reopen_isolation(
            cells, pool_size=primary_pool, system_id="ZeroMu+EK5_v3"
        )

    # Costed full_restart_rebuild comparator summary at primary pool & `both`
    full_restart_cells = [
        c
        for c in cells
        if c.pool_size == primary_pool and c.event_regime == primary_regime and c.is_full_restart_comparator
    ]
    ze_primary_cells = [
        c
        for c in cells
        if c.pool_size == primary_pool
        and c.event_regime == primary_regime
        and c.system_id == "ZeroMu+EK5_v3"
    ]
    fr_jpm = (
        float(sum(c.on_time_cited_correct_jobs for c in full_restart_cells))
        / max(sum(c.window_duration_ms for c in full_restart_cells) / 60_000.0, 1e-9)
        if full_restart_cells
        else 0.0
    )
    ze_jpm = (
        float(sum(c.on_time_cited_correct_jobs for c in ze_primary_cells))
        / max(sum(c.window_duration_ms for c in ze_primary_cells) / 60_000.0, 1e-9)
        if ze_primary_cells
        else 0.0
    )
    comparator_summary = {
        "pool_size": primary_pool,
        "event_regime": primary_regime,
        "zeromu_ek5_pooled_jpm": round(ze_jpm, 4),
        "full_restart_rebuild_pooled_jpm": round(fr_jpm, 4),
        "jpm_advantage_over_full_restart": round(ze_jpm - fr_jpm, 4),
        "full_restart_stale_commits": sum(c.stale_commits for c in full_restart_cells),
        "full_restart_late_or_timed_out_jobs": sum(c.late_or_timed_out_jobs for c in full_restart_cells),
        "zeromu_ek5_late_or_timed_out_jobs": sum(c.late_or_timed_out_jobs for c in ze_primary_cells),
    }

    # Bridge fit provenance
    bridge_provenance = {
        "d_old": emb_space.d_old,
        "d_new": emb_space.d_new,
        "uncentered_procrustes_sha256": emb_space.uncentered_bridge.weights_sha256,
        "zeromu_polar_bridge_sha256": emb_space.zeromu_bridge.weights_sha256,
        "zeromu_kappa": round(emb_space.zeromu_bridge.condition_number, 6),
    }

    cluster = primary_interaction["rate_interaction_matter_cluster_bootstrap"]
    gates = {
        "zero_stale_commits_on_zeromu_ek5": zeromu_ek5_stale_total == 0,
        "ek5_incremental_equals_full_parity": ek5_parity_failures == 0,
        "primary_metric_is_ontime_cited_correct_rate": bool(
            primary_interaction.get("primary_metric") == "on_time_cited_correct_rate"
            and cluster.get("sampling_unit") == "matter"
            and int(cluster.get("n_clusters", 0)) == len(templates)
        ),
        "conventional_migration_stall_equalized": CONV_MIGRATION_STALL_MS == 0.0,
        "sqlite_reopen_symmetric_with_ek5": SQLITE_RECHECK_PER_FIELD_MS == EK5_SELECTIVE_REOPEN_LEDGER_MS,
        "sham_reopen_control_valid": bool(sham_isolation["control_valid"]) if sham_isolation else False,
        "beats_full_restart_rebuild_jpm": bool(ze_jpm > fr_jpm),
        "all_eligible_jobs_accounted": all(
            c.eligible_jobs == len(templates) and (c.on_time_jobs + c.late_or_timed_out_jobs == c.eligible_jobs)
            for c in cells
        ),
    }
    overall_pass = all(gates.values())
    if overall_pass and mode == "stage1-controlled-replay":
        status = "stage1-replay-pass"
    elif overall_pass:
        status = "local-drycheck-pass"
    else:
        status = "replay-fail"

    # Deterministic payload (excluding timing-jittered `span_summary` and `export_ms` from canonical digest)
    canonical_cells_summary = [
        {
            "pool_size": c.pool_size,
            "schedule_id": c.schedule_id,
            "branch": c.branch,
            "system_id": c.system_id,
            "event_regime": c.event_regime,
            "eligible_jobs": c.eligible_jobs,
            "on_time_cited_correct_jobs": c.on_time_cited_correct_jobs,
            "stale_commits": c.stale_commits,
            "late_or_timed_out_jobs": c.late_or_timed_out_jobs,
            "on_time_cited_correct_jpm": round(c.on_time_cited_correct_jpm, 4),
            "mean_latency_ms": round(c.mean_latency_ms, 3),
            "ek5_trace_sha256": c.ek5_parity["canonical_trace_sha256"] if c.ek5_parity else None,
        }
        for c in cells
    ]
    deterministic_sha256 = _sha256_json(
        {
            "bundle_id": BUNDLE_ID,
            "mode": mode,
            "bridge_provenance": bridge_provenance,
            "primary_interaction": primary_interaction,
            "sham_reopen_isolation": sham_isolation,
            "comparator_summary": comparator_summary,
            "gates": gates,
            "canonical_cells_summary": canonical_cells_summary,
        }
    )

    payload = {
        "schema_version": "pv7_w20_d13_pooled_agent_loops_v1",
        "bundle_id": BUNDLE_ID,
        "item": "4",
        "demo_id": "D13",
        "status": status,
        "mode": mode,
        "worker_backend": backend.mode_name,
        "live_model_calls": sum(c.live_model_calls for c in cells),
        "matter_templates_count": len(templates),
        "arrival_schedules": list(schedules),
        "pool_sizes": [int(p) for p in pool_sizes],
        "branches": list(branches),
        "event_regimes": list(event_regimes),
        "system_cells": [system_label(r_arm, l_arm) for r_arm, l_arm in SYSTEM_CELLS] + ["full_restart_rebuild"],
        "corpus_fingerprints": matter_corpus_fingerprints(templates),
        "bridge_provenance": bridge_provenance,
        "total_cells_executed": len(cells),
        "total_eligible_jobs_executed": sum(c.eligible_jobs for c in cells),
        "zeromu_ek5_stale_commits_total": zeromu_ek5_stale_total,
        "ek5_parity_failures_total": ek5_parity_failures,
        "primary_2x2_interaction": primary_interaction,
        "interaction_by_pool_size": interaction_by_pool,
        "sham_reopen_isolation": sham_isolation,
        "full_restart_comparator": comparator_summary,
        "gates": gates,
        "overall_pass": overall_pass,
        "deterministic_sha256": deterministic_sha256,
        "cells": [c.to_dict(include_jobs=include_jobs) for c in cells],
    }
    if isinstance(backend, LiveVLLMWorkerBackend):
        payload["vllm_url"] = backend.vllm_url
        payload["requested_model"] = backend.model
        payload["served_model"] = backend.served_model
    return payload


__all__ = [
    "ARRIVAL_SCHEDULES",
    "BUNDLE_ID",
    "CellSummary",
    "CommitVerification",
    "EK5LedgerAdapter",
    "EVENT_REGIMES",
    "JobArrival",
    "JobOutcome",
    "LEDGER_ARMS",
    "LedgerAdapter",
    "LiveVLLMWorkerBackend",
    "MatterEmbeddingSpace",
    "MatterSourceDoc",
    "MatterTemplate",
    "POOL_SIZES",
    "PRIMARY_POOL_SIZE",
    "PooledAgentCoordinator",
    "REQUIRED_FIELDS",
    "RETRIEVER_ARMS",
    "RecordedReplayWorkerBackend",
    "SQLiteChallengerAdapter",
    "SYSTEM_CELLS",
    "SubtaskDraft",
    "TARGET_FIELD",
    "TWIN_BRANCHES",
    "UNAFFECTED_FIELDS",
    "UNRESOLVED_SENTINEL",
    "WorkerBackend",
    "build_24_matter_templates",
    "build_arrival_schedule",
    "compute_paired_2x2_interaction",
    "compute_sham_reopen_isolation",
    "matter_corpus_fingerprints",
    "run_pooled_agent_experiment",
    "system_label",
    "verify_commit_against_visible_prefix",
]
