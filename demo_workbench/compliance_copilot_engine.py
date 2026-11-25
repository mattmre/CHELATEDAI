"""Regulated-Enterprise Compliance Copilot engine (Wave-11, Demo D8 / Wave-10 spec `L3`).

Three live comparison arms run over ONE real corpus per scenario, the same keyword index
(`scripts.night1_common.FastBM25Index`), the same deterministic vector similarity, and the
same reciprocal-rank fusion (k=60):

  * ``baseline_rag``      hybrid retrieval with a real delete-by-id (steelman baseline),
  * ``overwrite_policy``  newest-record-per-(entity, slot) overwrite (emulation, not a live product),
  * ``lineage_control``   `ek5_v3_stratified.StratifiedEvidenceStore` + independent oracle cross-check.

Nothing here calls a simulator; every number returned is computed from the three pipelines at
request time. Ground truth is the hand-authored ``expected_current`` set of each scenario step.

Presentation modes
------------------
``non_nda`` (default) builds every response from a typed public projection with a server-side
label map and then passes the serialized bytes through the fail-closed 56-pattern verifier
(`scripts/verify_non_nda_ip_clean.py`). ``nda`` mode additionally attaches an ``internal`` block and
requires a server-configured token (``X-Copilot-Nda-Token``); a bare query flag is never enough.

Integration
-----------
The orchestrator wires this engine into ``server.py`` with a forwarder that calls
``handle_copilot_get(path, query_params, headers)`` / ``handle_copilot_post(path, query_params,
headers, body)`` for every path where ``is_copilot_path(path)`` is true. Both return
``(status, content_type, body_bytes)``. Standalone: ``python -m demo_workbench.compliance_copilot_engine
--port 8788``.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import functools
import hashlib
import hmac
import json
import os
import re
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Sequence, Set, Tuple
from urllib.parse import parse_qs, urlparse

import numpy as np
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ek5_v3_stratified import (  # noqa: E402
    EdgeAttestation,
    StratifiedEvent,
    StratifiedEvidenceStore,
    SupersedeCycleError,
    V3_VALID_EDGE_TYPES,
    compute_alternating_wfs_oracle,
)
from scripts.night1_common import FastBM25Index  # noqa: E402
from scripts.verify_non_nda_ip_clean import (  # noqa: E402
    assert_non_nda_clean,
    scan_text_for_ip_leaks,
)

ENGINE_FILE_SHA256_SOURCE = REPO_ROOT / "ek5_v3_stratified.py"
HTML_PATH = Path(__file__).resolve().parent / "compliance_copilot.html"

TENANT = "tenant_demo"
TOP_K = 5
RRF_K = 60
DENSE_DIM = 512
MAX_BODY_BYTES = 64 * 1024
MAX_INJECT_EVENTS = 500
MAX_INJECT_EDGES = 2000
MAX_QUERY_CHARS = 400
API_PREFIX = "/api/v2/copilot"
PAGE_PATHS = ("/copilot", "/copilot/", "/copilot/index.html")
NDA_TOKEN_HEADER = "x-copilot-nda-token"
MODES = ("non_nda", "nda")

_ID_RE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")

# ---------------------------------------------------------------------------
# Server-side label maps (never shipped to the client as keys).
# ---------------------------------------------------------------------------
STATE_LABEL = {"ADMIT": "Current", "AUDIT_ONLY": "Withdrawn", "CONTESTED": "Disputed"}
REASON_LABEL = {
    "CURRENT": "Current",
    "REFUTED": "Retracted by a higher-authority notice",
    "SUPERSEDED": "Replaced by a newer record",
    "SUPERSEDED_ANCESTOR": "Derived from a replaced record",
    "MISSING_MANDATORY_PARENT": "Source record unavailable",
    "UNFOUNDED_CYCLE": "Circular support only",
    "EPISTEMIC_CONTESTED": "Disputed between equal-authority sources",
    "TEMPORAL_CONTESTED": "Currency disputed",
    "TEMPORAL_ANCESTOR_CONTESTED": "Derived from a disputed record",
}
EDGE_LABEL = {
    "REFUTE": "Retraction",
    "SUPERSEDE": "Replacement",
    "COPY_OF": "Echoed copy",
    "DERIVES_CURRENT": "Derived summary",
    "DERIVES_SNAPSHOT": "Point-in-time derivative",
    "AMENDS": "Amendment",
    "CITES_AUDIT": "Audit reference",
}
# Public edge vocabulary accepted by the custom-evidence endpoint (maps to engine edge types server-side).
PUBLIC_EDGE_INPUT = {
    "retracts": "REFUTE",
    "replaces": "SUPERSEDE",
    "copy": "COPY_OF",
    "derived": "DERIVES_CURRENT",
    "snapshot": "DERIVES_SNAPSHOT",
    "amends": "AMENDS",
    "cites": "CITES_AUDIT",
}
ARM_IDS = ("baseline_rag", "overwrite_policy", "lineage_control")
ARM_PUBLIC = {
    "baseline_rag": (
        "Hybrid retrieval with delete-by-id",
        "Keyword and vector scores fused by reciprocal rank over every ingested record. When a notice "
        "names a record, the operator really deletes that one record by id. This is the strongest simple baseline.",
    ),
    "overwrite_policy": (
        "Overwrite policy (emulation, not a live product)",
        "The newest record for each (entity, slot) replaces the previous one. Nothing else is withdrawn.",
    ),
    "lineage_control": (
        "Chelated control plane",
        "Withdrawal, replacement and dispute status propagate through the recorded lineage; an audience "
        "policy decides who may see withdrawn material.",
    ),
}
ARM_INTERNAL = {
    "baseline_rag": "FastBM25Index + hashed 3-gram dense + RRF k=60, dict delete-by-id",
    "overwrite_policy": "(entity, slot) last-write-wins over the same retrieval stack",
    "lineage_control": "ek5_v3_stratified.StratifiedEvidenceStore + compute_alternating_wfs_oracle cross-check",
}
READER_POLICY = (
    "One reader for all arms: take the best-ranked current record in the queried slot; otherwise the best-ranked "
    "current record about the same entity that carries a value; otherwise report no current evidence. Only the "
    "control plane may abstain (when the queried slot is in dispute)."
)
DISCLAIMER = "Not legal advice. This receipt records how a demonstration system processed synthetic records."
KEY_DISCLOSURE = (
    "The signing key is generated in memory when the server starts and is published only through the public-key "
    "endpoint. It does not survive a restart, so a receipt proves integrity since issuance, not the identity of a "
    "third-party auditor."
)
CLOCK_DISCLOSURE = "issued_at_us comes from the server clock and is not a trusted timestamp."
DENSE_PUBLIC = "Character n-gram vectors (hashed and deterministic; not a neural encoder)"
DENSE_INTERNAL = "hashed signed char-3gram, dim=512, L2-normalized cosine"
RETRIEVAL_PUBLIC = "Keyword index plus vector similarity fused by reciprocal rank (k=60)"
RETRIEVAL_INTERNAL = "FastBM25Index + hashed 3-gram dense, RRF k=60, top_k=5"


class CopilotError(Exception):
    """Request-level failure carrying an HTTP status and a neutral, non-echoing message."""

    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


# ---------------------------------------------------------------------------
# Scenario fixtures (synthetic, hand-authored; ground truth is authored, not engine-derived)
# ---------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class DocSpec:
    doc_id: str
    title: str
    system: str
    text: str
    entity: str
    slot: str
    claim_value: str = ""
    tier: int = 1


@dataclasses.dataclass(frozen=True)
class QuerySpec:
    query_id: str
    text: str
    entity: str
    slot: str
    expected_answer: str


@dataclasses.dataclass(frozen=True)
class RetractionSpec:
    retractor_id: str
    target_id: str
    irreversible: bool = False


@dataclasses.dataclass(frozen=True)
class StepSpec:
    title: str
    narrative: str
    docs: Tuple[DocSpec, ...]
    edges: Tuple[Tuple[str, str, str], ...]
    arm1_delete_ids: Tuple[str, ...]
    queries: Tuple[QuerySpec, ...]
    expected_current: FrozenSet[str]
    expected_contested: FrozenSet[str] = frozenset()
    retraction: Optional[RetractionSpec] = None
    gdpr_erasure: bool = False
    forbidden_substrings: Tuple[str, ...] = ()


@dataclasses.dataclass(frozen=True)
class ScenarioSpec:
    scenario_id: str
    title: str
    domain: str
    summary: str
    roles: Tuple[str, str]
    role_labels: Tuple[str, str]
    steps: Tuple[StepSpec, ...]


def _ids(text: str) -> FrozenSet[str]:
    return frozenset(text.split())


def _distractors(prefix: str, rows: Sequence[Tuple[str, str, str]]) -> Tuple[DocSpec, ...]:
    return tuple(
        DocSpec(f"{prefix}_x{i}", title, system, text, system, "notice", "", 0)
        for i, (title, system, text) in enumerate(rows, start=1)
    )


def _build_w1() -> ScenarioSpec:
    ent = "AX-7"
    distractors = _distractors(
        "w1",
        [
            ("Loading dock closure", "Facilities notice", "Facilities notice: the pharmacy loading dock is closed for repaving until Friday."),
            ("Vaccination clinics", "Staff bulletin", "Staff bulletin: flu vaccination clinics run every Tuesday in the east wing."),
            ("Hygiene refresher", "Training calendar", "Training calendar: hand hygiene refresher is scheduled for the first Monday of the month."),
        ],
    )
    base_docs = (
        DocSpec("w1_label", "Product label v1", "Product label",
                "Product label version 1 for compound AX-7 states the pediatric dose is 15 mg per kg per day, divided into three doses.",
                ent, "label_dose", "15 mg/kg/day", 1),
        DocSpec("w1_portal", "Patient portal post", "Patient portal",
                "Patient portal post: the AX-7 label gives the pediatric dose as 15 mg per kg per day.",
                ent, "portal_post", "15 mg/kg/day", 1),
        DocSpec("w1_pathway", "Care pathway summary", "Care pathway",
                "Care pathway summary: start AX-7 at 15 mg per kg per day for children under twelve.",
                ent, "care_pathway", "15 mg/kg/day", 1),
        DocSpec("w1_memo", "Safety audit memo", "Safety audit memo",
                "Safety audit memo: the 15 mg per kg per day pediatric figure in AX-7 label version 1 is under pharmacovigilance review.",
                ent, "audit_note", "", 2),
    ) + distractors
    base_edges = (
        ("w1_portal", "w1_label", "COPY_OF"),
        ("w1_pathway", "w1_label", "DERIVES_CURRENT"),
        ("w1_memo", "w1_label", "CITES_AUDIT"),
    )
    dose_q = QuerySpec("dose", "What is the pediatric dose of AX-7?", ent, "label_dose", "1.5 mg/kg/day")
    dose_q1 = dataclasses.replace(dose_q, expected_answer="15 mg/kg/day")
    contra_abstain = QuerySpec("renal", "Is AX-7 safe in renal impairment?", ent, "contraindication", "ABSTAIN")
    contra_ruled = dataclasses.replace(contra_abstain, expected_answer="contraindicated in renal impairment")
    bad = ("15 mg per kg per day",)
    steps = (
        StepSpec(
            "Ingest the label and its echoes",
            "The label, a patient-portal echo, a derived care-pathway summary and a safety audit memo are ingested. "
            "Everything agrees, so every arm answers the same way.",
            base_docs, base_edges, (), (dose_q1,),
            _ids("w1_label w1_portal w1_pathway w1_memo w1_x1 w1_x2 w1_x3"),
        ),
        StepSpec(
            "Regulator withdraws the dose",
            "A higher-authority regulatory notice retracts the published pediatric dose and supplies the corrected one. "
            "Delete-by-id removes only the label itself.",
            (DocSpec("w1_notice", "Regulatory notice", "Regulatory notice",
                     "Regulatory notice: the pediatric dose previously published for AX-7 was a transcription error and is withdrawn. "
                     "The correct pediatric dose is 1.5 mg per kg per day.",
                     ent, "label_dose", "1.5 mg/kg/day", 3),),
            (("w1_notice", "w1_label", "REFUTE"),),
            ("w1_label",), (dose_q,),
            _ids("w1_notice w1_memo w1_x1 w1_x2 w1_x3"),
            retraction=RetractionSpec("w1_notice", "w1_label"),
            forbidden_substrings=bad,
        ),
        StepSpec(
            "Two equal-authority auditors disagree",
            "Two auditors of equal authority publish opposite findings on a different slot. The dispute must not "
            "contaminate the dose slot, and the disputed slot must abstain.",
            (DocSpec("w1_audA", "Auditor A finding", "Auditor A",
                     "Auditor A finding: AX-7 is contraindicated in renal impairment because exposure doubles.",
                     ent, "contraindication", "contraindicated in renal impairment", 2),
             DocSpec("w1_audB", "Auditor B finding", "Auditor B",
                     "Auditor B finding: AX-7 is safe in mild renal impairment with routine monitoring.",
                     ent, "contraindication", "safe in mild renal impairment", 2)),
            (("w1_audA", "w1_audB", "REFUTE"), ("w1_audB", "w1_audA", "REFUTE")),
            (), (contra_abstain, dose_q),
            _ids("w1_notice w1_memo w1_x1 w1_x2 w1_x3"),
            expected_contested=_ids("w1_audA w1_audB"),
            forbidden_substrings=bad,
        ),
        StepSpec(
            "Higher authority rules on the dispute",
            "A chief medical officer ruling of higher authority retracts the finding it rejects; the dispute resolves.",
            (DocSpec("w1_ruling", "Chief medical officer ruling", "Medical ruling",
                     "Chief medical officer ruling: Auditor A is upheld and AX-7 is contraindicated in renal impairment.",
                     ent, "contraindication", "contraindicated in renal impairment", 4),),
            (("w1_ruling", "w1_audB", "REFUTE"),),
            (), (contra_ruled, dose_q),
            _ids("w1_notice w1_memo w1_audA w1_ruling w1_x1 w1_x2 w1_x3"),
            forbidden_substrings=bad,
        ),
    )
    return ScenarioSpec(
        "w1", "Pediatric dose correction", "Healthcare",
        "A published pediatric dose is withdrawn by the regulator. Echoes and derived summaries must stop being served; "
        "an audit memo that quotes the withdrawn dose must reach only the safety investigator; a dispute on another "
        "slot must abstain without poisoning the dose.",
        ("clinician", "safety_investigator"), ("Clinician (standard view)", "Safety investigator (privileged view)"),
        steps,
    )


def _build_w2() -> ScenarioSpec:
    ent = "Customer 4471"
    profile_text = ("CRM profile for Customer 4471: Jordan Example, email jordan.example@example.invalid, "
                    "phone 555-0142, premium plan since 2022.")
    distractors = _distractors(
        "w2",
        [
            ("Dark mode release", "Release notes", "Release notes: the mobile app now supports dark mode on all devices."),
            ("Invoice cadence", "Billing notice", "Billing notice: invoices are issued on the first business day of each month."),
            ("Holiday hours", "Holiday schedule", "Holiday schedule: the support desk closes early on the last Friday of December."),
        ],
    )
    q = QuerySpec("email", "What is the contact email for Customer 4471?", ent, "crm_profile", "ERASED")
    q1 = dataclasses.replace(q, expected_answer="jordan.example@example.invalid")
    pii = ("jordan.example@example.invalid", "555-0142", "Jordan Example")
    steps = (
        StepSpec(
            "Ingest the CRM profile and its echoes",
            "The profile, a derived account summary, a support ticket that quotes the profile and a privacy audit log are ingested.",
            (DocSpec("w2_profile", "CRM profile", "CRM profile", profile_text, ent, "crm_profile",
                     "jordan.example@example.invalid", 1),
             DocSpec("w2_summary", "Account summary", "Account summary",
                     "Account summary for Customer 4471: Jordan Example is a premium customer reachable at jordan.example@example.invalid.",
                     ent, "crm_summary", "jordan.example@example.invalid", 1),
             DocSpec("w2_ticket", "Support ticket", "Support ticket",
                     "Support ticket 88213 quotes the CRM profile for Customer 4471: Jordan Example, jordan.example@example.invalid, phone 555-0142.",
                     ent, "ticket_echo", "jordan.example@example.invalid", 1),
             DocSpec("w2_memo", "Privacy audit log", "Privacy audit log",
                     "Privacy audit log: the CRM profile for Customer 4471 was last reviewed for consent status in March.",
                     ent, "audit_note", "", 2)) + distractors,
            (("w2_summary", "w2_profile", "DERIVES_CURRENT"),
             ("w2_ticket", "w2_profile", "COPY_OF"),
             ("w2_memo", "w2_profile", "CITES_AUDIT")),
            (), (q1,),
            _ids("w2_profile w2_summary w2_ticket w2_memo w2_x1 w2_x2 w2_x3"),
        ),
        StepSpec(
            "Erasure order for Customer 4471",
            "A verified right-to-erasure order retracts the profile. Delete-by-id removes the profile only; the summary and the "
            "ticket quote the same personal data. The control plane physically replaces every withdrawn record's text with a hash tombstone.",
            (DocSpec("w2_erasure", "Erasure order", "Erasure order",
                     "Erasure order: a verified GDPR Article 17 request for Customer 4471 is complete and no personal data may be served.",
                     ent, "crm_profile", "ERASED", 3),),
            (("w2_erasure", "w2_profile", "REFUTE"),),
            ("w2_profile",), (q,),
            _ids("w2_erasure w2_memo w2_x1 w2_x2 w2_x3"),
            retraction=RetractionSpec("w2_erasure", "w2_profile", irreversible=True),
            gdpr_erasure=True,
            forbidden_substrings=pii,
        ),
        StepSpec(
            "Backup restore re-ingests the erased profile",
            "A nightly backup restore re-ingests the exact erased profile text. Delete-by-id and overwrite have no memory of the erasure; "
            "the control plane recognizes the erased-content fingerprint and withdraws the record on arrival.",
            (DocSpec("w2_restore", "Backup restore", "Backup restore", profile_text, ent, "crm_profile",
                     "jordan.example@example.invalid", 1),),
            (),
            (), (q,),
            _ids("w2_erasure w2_memo guard_w2_restore w2_x1 w2_x2 w2_x3"),
            forbidden_substrings=pii,
        ),
    )
    return ScenarioSpec(
        "w2", "Right-to-erasure order", "Customer data privacy",
        "An erasure order must remove a customer's personal data from every record derived from it, survive a backup restore, "
        "and leave an audit trail that contains hashes instead of the erased text.",
        ("support_agent", "privacy_officer"), ("Support agent (standard view)", "Privacy officer (privileged view)"),
        steps,
    )


def _build_w3() -> ScenarioSpec:
    ent = "Northwind Corp"
    distractors = _distractors(
        "w3",
        [
            ("Shareholder picnic", "Office notice", "Office notice: the annual shareholder picnic moves to the second Saturday of June."),
            ("Password rotation", "IT bulletin", "IT bulletin: password rotation is required every ninety days for all staff."),
            ("Badge reader swap", "Facilities notice", "Facilities notice: the north entrance badge reader will be replaced next week."),
        ],
    )
    rev_q1 = QuerySpec("revenue", "What was Northwind Corp third quarter revenue?", ent, "q3_revenue", "$412 million")
    rely_q = QuerySpec("reliance", "Can the Northwind Corp third quarter revenue figure still be relied upon?", ent, "filing_status", "NON_RELIANCE")
    rest_q = QuerySpec("restated", "What is the restated Northwind Corp third quarter revenue?", ent, "q3_revenue", "$398 million")
    bad = ("$412 million",)
    steps = (
        StepSpec(
            "File the quarterly report",
            "The filing, a board deck built from it, a press release that repeats it, an analyst model derived from it and a compliance memo are ingested.",
            (DocSpec("w3_10q", "Form 10-Q", "Form 10-Q",
                     "Northwind Corp Form 10-Q for the third quarter reports revenue of $412 million.",
                     ent, "q3_revenue", "$412 million", 2),
             DocSpec("w3_deck", "Board deck", "Board deck",
                     "Board deck presented on October 3 included third quarter figures drawn from the filed 10-Q.",
                     ent, "board_deck", "", 1),
             DocSpec("w3_press", "Press release", "Press release",
                     "Press release repeats Northwind Corp third quarter revenue of $412 million.",
                     ent, "press_release", "$412 million", 1),
             DocSpec("w3_note", "Analyst model", "Analyst model",
                     "Analyst model uses third quarter revenue of $412 million as its base case for Northwind Corp.",
                     ent, "analyst_note", "$412 million", 1),
             DocSpec("w3_memo", "Compliance audit memo", "Compliance memo",
                     "Compliance audit memo: the third quarter revenue of $412 million was cited during the filing review.",
                     ent, "audit_note", "", 2)) + distractors,
            (("w3_deck", "w3_10q", "DERIVES_SNAPSHOT"),
             ("w3_press", "w3_10q", "COPY_OF"),
             ("w3_note", "w3_10q", "DERIVES_CURRENT"),
             ("w3_memo", "w3_10q", "CITES_AUDIT")),
            (), (rev_q1,),
            _ids("w3_10q w3_deck w3_press w3_note w3_memo w3_x1 w3_x2 w3_x3"),
        ),
        StepSpec(
            "Non-reliance filing",
            "A current report declares the filed statements should no longer be relied upon. That retraction reaches the board deck, "
            "the press release and the analyst model through their lineage; delete-by-id removes only the filing.",
            (DocSpec("w3_8k", "Form 8-K", "Form 8-K",
                     "Form 8-K Item 4.02: Northwind Corp concludes that its previously issued third quarter financial statements should no longer be relied upon.",
                     ent, "filing_status", "NON_RELIANCE", 3),),
            (("w3_8k", "w3_10q", "REFUTE"),),
            ("w3_10q",), (rely_q,),
            _ids("w3_8k w3_memo w3_x1 w3_x2 w3_x3"),
            retraction=RetractionSpec("w3_8k", "w3_10q"),
            forbidden_substrings=bad,
        ),
        StepSpec(
            "Restated filing and amended notice",
            "An amended quarterly report restates the revenue and an amended current report replaces the non-reliance notice. "
            "Only the control plane knows the old figure's echoes are still withdrawn.",
            (DocSpec("w3_10qa", "Form 10-Q/A", "Form 10-Q/A",
                     "Form 10-Q/A restates Northwind Corp third quarter revenue to $398 million.",
                     ent, "q3_revenue", "$398 million", 2),
             DocSpec("w3_8ka", "Form 8-K/A", "Form 8-K/A",
                     "Form 8-K/A: Northwind Corp confirms the restated third quarter figures in the amended 10-Q may be relied upon.",
                     ent, "filing_status", "RELIANCE_RESTORED", 3)),
            (("w3_8ka", "w3_8k", "SUPERSEDE"),),
            ("w3_8k",), (rest_q,),
            _ids("w3_8ka w3_10qa w3_memo w3_x1 w3_x2 w3_x3"),
            forbidden_substrings=bad,
        ),
    )
    return ScenarioSpec(
        "w3", "Non-reliance and restatement", "Securities disclosure",
        "A non-reliance filing retracts a quarterly report. Every echo, derived model and point-in-time deck built on it must follow, "
        "and a later restatement must not resurrect the old figure.",
        ("analyst", "compliance_auditor"), ("Analyst (standard view)", "Compliance auditor (privileged view)"),
        steps,
    )


SCENARIOS: Dict[str, ScenarioSpec] = {s.scenario_id: s for s in (_build_w1(), _build_w2(), _build_w3())}


# ---------------------------------------------------------------------------
# Retrieval stack shared by all arms
# ---------------------------------------------------------------------------
def content_fingerprint(text: str) -> str:
    """SHA-256 of whitespace/case-normalized content (the erased-content blocklist key)."""
    return hashlib.sha256(" ".join(text.lower().split()).encode("utf-8")).hexdigest()


def tombstone_for(fingerprint_hex: str) -> str:
    return f"[REDACTED_GDPR_ART17:sha256:{fingerprint_hex}]"


@functools.lru_cache(maxsize=65536)
def _gram_slot(gram: str) -> Tuple[int, float]:
    h = int.from_bytes(hashlib.blake2b(gram.encode("utf-8"), digest_size=8).digest(), "big")
    return h % DENSE_DIM, (1.0 if (h >> 40) & 1 else -1.0)


def dense_embed(texts: Sequence[str]) -> np.ndarray:
    mat = np.zeros((len(texts), DENSE_DIM), dtype=np.float64)
    for i, text in enumerate(texts):
        norm = " " + " ".join(text.lower().split()) + " "
        for j in range(len(norm) - 2):
            slot, sign = _gram_slot(norm[j:j + 3])
            mat[i, slot] += sign
        n = float(np.linalg.norm(mat[i]))
        if n > 0.0:
            mat[i] /= n
    return mat


def hybrid_retrieve(docs: Sequence[Tuple[str, str]], query: str, top_k: int = TOP_K) -> List[Tuple[str, float]]:
    """Real keyword index + vector cosine fused by reciprocal rank (k=60). `docs` is [(doc_id, text)]."""
    if not docs:
        return []
    ids = [d[0] for d in docs]
    texts = [d[1] for d in docs]
    n = len(ids)
    bm = FastBM25Index(ids, texts).score_query(query)
    dense = dense_embed(texts) @ dense_embed([query])[0]
    fused = np.zeros(n, dtype=np.float64)
    for scores, require_positive in ((bm, True), (dense, False)):
        order = sorted(range(n), key=lambda i: (-float(scores[i]), ids[i]))
        for rank, i in enumerate(order, start=1):
            if require_positive and float(scores[i]) <= 0.0:
                continue
            fused[i] += 1.0 / (RRF_K + rank)
    final = sorted(range(n), key=lambda i: (-float(fused[i]), ids[i]))[:top_k]
    return [(ids[i], float(fused[i])) for i in final]


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class DocRecord:
    spec: DocSpec
    text: str
    tombstone_sha: Optional[str] = None
    synthetic_guard: bool = False


@dataclasses.dataclass
class StepApply:
    inserted: List[str]
    transitioned: Dict[str, Tuple[str, str]]
    tombstoned: List[str]
    guarded: List[str]
    work_wall_ms: float


class LineageArm:
    """Arm 3: the real EK5 v3 store plus the Copilot-layer policies (blocklist, tombstones, audience)."""

    def __init__(self) -> None:
        self.store = StratifiedEvidenceStore()
        self.clock = 0
        self.docs: Dict[str, DocRecord] = {}
        self.blocklist: Dict[str, str] = {}
        self.edge_log: List[Tuple[str, str, str]] = []
        self.step_applies: List[StepApply] = []

    def tick(self) -> int:
        self.clock += 1
        return self.clock

    def _event(self, spec: DocSpec, text: str) -> StratifiedEvent:
        t = self.tick()
        return StratifiedEvent(
            event_id=spec.doc_id, entity=spec.entity, slot=spec.slot, claim_value=spec.claim_value, text=text,
            valid_time=t, tx_time=t, tenant=TENANT, domain="general", authority_tier=spec.tier,
        )

    def _att(self, src: str, dst: str, etype: str) -> EdgeAttestation:
        t = self.tick()
        self.edge_log.append((src, dst, etype))
        return EdgeAttestation(attestation_id=f"att:{src}:{etype}:{dst}", src_id=src, dst_id=dst, edge_type=etype,
                               valid_time=t, tx_time=t)

    def apply_step(self, step: StepSpec) -> StepApply:
        events: List[StratifiedEvent] = []
        atts: List[EdgeAttestation] = []
        guarded: List[str] = []
        for spec in step.docs:
            fp = content_fingerprint(spec.text)
            if fp in self.blocklist:
                text = tombstone_for(fp)
                rec = DocRecord(spec, text, tombstone_sha=fp)
                guard_spec = DocSpec(
                    f"guard_{spec.doc_id}", f"Erasure guard for {spec.title}", "Erasure guard",
                    "Erasure guard: re-ingested content matched an erased-content fingerprint and was withdrawn on arrival.",
                    spec.entity, "erasure_guard", "", 9,
                )
                self.docs[spec.doc_id] = rec
                self.docs[guard_spec.doc_id] = DocRecord(guard_spec, guard_spec.text, synthetic_guard=True)
                events.append(self._event(spec, text))
                events.append(self._event(guard_spec, guard_spec.text))
                atts.append(self._att(guard_spec.doc_id, spec.doc_id, "REFUTE"))
                guarded.append(spec.doc_id)
            else:
                self.docs[spec.doc_id] = DocRecord(spec, spec.text)
                events.append(self._event(spec, spec.text))
        for src, dst, etype in step.edges:
            atts.append(self._att(src, dst, etype))
        delta = self.store.apply_mutation(events, atts)
        tombstoned: List[str] = []
        if step.gdpr_erasure:
            affected: Set[str] = set()
            if step.retraction is not None:
                affected.add(step.retraction.target_id)
            affected.update(nid for nid, (_, now) in delta.transitioned.items() if now != "ADMIT")
            for nid in sorted(affected):
                rec = self.docs.get(nid)
                if rec is None or rec.tombstone_sha is not None:
                    continue
                fp = content_fingerprint(rec.text)
                self.blocklist[fp] = nid
                rec.tombstone_sha = fp
                rec.text = tombstone_for(fp)
                ev = self.store.events[nid]
                self.store.events[nid] = dataclasses.replace(ev, text=rec.text)
                tombstoned.append(nid)
        result = StepApply(
            inserted=sorted(delta.inserted), transitioned=dict(delta.transitioned), tombstoned=tombstoned,
            guarded=guarded, work_wall_ms=float(delta.work_stats.get("wall_ms", 0.0)),
        )
        self.step_applies.append(result)
        return result

    def eligibility(self, doc_id: str) -> str:
        return self.store.eligibility.get(doc_id, "ADMIT")

    def citer_of_non_admit(self, doc_id: str) -> bool:
        return any(self.eligibility(c) != "ADMIT" for c in self.store.cites_audit.get(doc_id, ()))

    def oracle_check(self) -> Tuple[int, int]:
        oracle = compute_alternating_wfs_oracle(self.store.events.keys(), self.store.refs, self.store.deps_v)
        mismatches = sum(1 for nid, v in oracle.items() if self.store.val.get(nid) != v)
        mismatches += sum(1 for nid in self.store.events if nid not in oracle)
        return len(oracle), mismatches


class BaselineArms:
    """Arms 1 and 2 over the same documents. Neither knows about lineage."""

    def __init__(self) -> None:
        self.arm1: Dict[str, DocSpec] = {}
        self.arm2: Dict[Tuple[str, str], DocSpec] = {}
        self.arm2_order: List[Tuple[str, str]] = []

    def apply_step(self, step: StepSpec) -> None:
        for spec in step.docs:
            self.arm1[spec.doc_id] = spec
            key = (spec.entity, spec.slot)
            if key in self.arm2:
                self.arm2_order.remove(key)
            self.arm2[key] = spec
            self.arm2_order.append(key)
        for doc_id in step.arm1_delete_ids:
            self.arm1.pop(doc_id, None)

    def arm1_docs(self) -> List[DocSpec]:
        return list(self.arm1.values())

    def arm2_docs(self) -> List[DocSpec]:
        return [self.arm2[k] for k in self.arm2_order]


@dataclasses.dataclass
class ReplayState:
    scenario: ScenarioSpec
    upto: int
    lineage: LineageArm
    baselines: BaselineArms


def replay(scenario_id: str, upto_step: int) -> ReplayState:
    scenario = SCENARIOS[scenario_id]
    lineage = LineageArm()
    baselines = BaselineArms()
    for step in scenario.steps[:upto_step]:
        baselines.apply_step(step)
        lineage.apply_step(step)
    return ReplayState(scenario, upto_step, lineage, baselines)


_REPLAY_CACHE: Dict[Tuple[str, int], ReplayState] = {}
_REPLAY_LOCK = threading.Lock()


def get_replay(scenario_id: str, upto_step: int) -> ReplayState:
    """Cached read-only replay. Callers that mutate must use `replay()` directly."""
    key = (scenario_id, upto_step)
    with _REPLAY_LOCK:
        state = _REPLAY_CACHE.get(key)
        if state is None:
            state = replay(scenario_id, upto_step)
            _REPLAY_CACHE[key] = state
        return state


# ---------------------------------------------------------------------------
# Answer + metric computation
# ---------------------------------------------------------------------------
@dataclasses.dataclass
class Hit:
    doc_id: str
    title: str
    system: str
    entity: str
    slot: str
    claim_value: str
    text: str
    rrf: float
    current: bool
    state: str
    reason: str
    banner: Optional[str]
    stale: bool


def read_answer(hits: Sequence[Hit], query: QuerySpec) -> str:
    for h in hits:
        if h.current and h.slot == query.slot and h.claim_value:
            return h.claim_value
    for h in hits:
        if h.current and h.entity == query.entity and h.claim_value:
            return h.claim_value
    return "NO_CURRENT_EVIDENCE"


def _contains_forbidden(text: str, needles: Sequence[str]) -> bool:
    low = text.lower()
    return any(n.lower() in low for n in needles)


def _baseline_hits(
    docs: Sequence[DocSpec], query_text: str, step: StepSpec
) -> List[Hit]:
    by_id = {d.doc_id: d for d in docs}
    out: List[Hit] = []
    for doc_id, rrf in hybrid_retrieve([(d.doc_id, d.text) for d in docs], query_text):
        d = by_id[doc_id]
        out.append(Hit(d.doc_id, d.title, d.system, d.entity, d.slot, d.claim_value, d.text, rrf,
                       True, "Served", "No lineage status is tracked", None, d.doc_id not in step.expected_current))
    return out


def _lineage_hits(
    state: ReplayState, step: StepSpec, query_text: str, privileged: bool
) -> Tuple[List[Hit], int]:
    lin = state.lineage
    shown: List[Tuple[str, str]] = []
    hidden = 0
    for doc_id, rec in lin.docs.items():
        elig = lin.eligibility(doc_id)
        flagged = elig != "ADMIT" or lin.citer_of_non_admit(doc_id)
        if flagged and not privileged:
            hidden += 1
            continue
        shown.append((doc_id, rec.text))
    out: List[Hit] = []
    for doc_id, rrf in hybrid_retrieve(shown, query_text):
        rec = lin.docs[doc_id]
        elig = lin.eligibility(doc_id)
        banner: Optional[str] = None
        if rec.tombstone_sha is not None:
            banner = "Text physically erased; only the content hash remains"
        elif elig == "AUDIT_ONLY":
            banner = "WITHDRAWN - audit use only"
        elif elig == "CONTESTED":
            banner = "DISPUTED - sources of equal authority disagree"
        elif lin.citer_of_non_admit(doc_id):
            banner = "Cites a withdrawn record"
        out.append(Hit(
            doc_id, rec.spec.title, rec.spec.system, rec.spec.entity, rec.spec.slot, rec.spec.claim_value, rec.text,
            rrf, elig == "ADMIT", STATE_LABEL.get(elig, elig),
            REASON_LABEL.get(lin.store.reason.get(doc_id, ""), "Other"), banner, doc_id not in step.expected_current,
        ))
    return out, hidden


def _hit_public(h: Hit) -> Dict[str, Any]:
    return {
        "id": h.doc_id, "title": h.title, "system": h.system, "text": h.text, "state": h.state,
        "status_note": h.reason, "flag": h.banner, "rank_score_milli": int(round(h.rrf * 1e6)),
        "not_current_in_truth": h.stale,
    }


def _eval_query(
    arm_id: str, state: ReplayState, step: StepSpec, query: QuerySpec, text: str, privileged: bool,
    judge: bool,
) -> Dict[str, Any]:
    hidden = 0
    if arm_id == "baseline_rag":
        hits = _baseline_hits(state.baselines.arm1_docs(), text, step)
    elif arm_id == "overwrite_policy":
        hits = _baseline_hits(state.baselines.arm2_docs(), text, step)
    else:
        hits, hidden = _lineage_hits(state, step, text, privileged)
    slot_status = None
    answer = read_answer(hits, query)
    abstained = False
    if arm_id == "lineage_control":
        res = state.lineage.store.resolve_slot(TENANT, query.entity, query.slot)
        slot_status = res["status"]
        if res["status"] == "CONTESTED":
            answer, abstained = "ABSTAIN", True
    unflagged = [h for h in hits if h.banner is None]
    stale_unflagged = sorted({h.doc_id for h in unflagged if h.stale})
    stale_flagged = sorted({h.doc_id for h in hits if h.banner is not None and h.stale})
    forbidden = sorted({h.doc_id for h in unflagged if _contains_forbidden(h.text, step.forbidden_substrings)})
    result: Dict[str, Any] = {
        "query_id": query.query_id,
        "text": text,
        "answer": answer,
        "abstained": abstained,
        "retrieved": [_hit_public(h) for h in hits],
        "stale_unflagged_ids": stale_unflagged,
        "stale_flagged_ids": stale_flagged,
        "forbidden_text_ids": forbidden,
        "hidden_from_role": hidden,
        "_slot_status": slot_status,
    }
    if judge:
        result["expected_answer"] = query.expected_answer
        result["correct"] = bool(answer == query.expected_answer)
    else:
        result["expected_answer"] = None
        result["correct"] = None
    return result


def _arm_summary(queries: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    judged = [q for q in queries if q["correct"] is not None]
    stale = sorted({i for q in queries for i in q["stale_unflagged_ids"]})
    forb = sorted({i for q in queries for i in q["forbidden_text_ids"]})
    flagged = sorted({i for q in queries for i in q["stale_flagged_ids"]})
    return {
        "correct": sum(1 for q in judged if q["correct"]),
        "judged": len(judged),
        "stale_unflagged_count": len(stale),
        "stale_unflagged_ids": stale,
        "forbidden_text_count": len(forb),
        "withdrawn_shown_with_flag_count": len(flagged),
    }


def _require_scenario(scenario_id: Any) -> ScenarioSpec:
    if not isinstance(scenario_id, str) or scenario_id not in SCENARIOS:
        raise CopilotError(422, "unknown_scenario", "Unknown scenario.")
    return SCENARIOS[scenario_id]


def _require_step(scenario: ScenarioSpec, step: Any) -> int:
    try:
        k = int(step)
    except (TypeError, ValueError):
        raise CopilotError(422, "bad_step", "Step must be an integer.") from None
    if not 1 <= k <= len(scenario.steps):
        raise CopilotError(422, "bad_step", f"Step must be between 1 and {len(scenario.steps)}.")
    return k


def _require_role(scenario: ScenarioSpec, role: Any) -> Tuple[str, bool]:
    if role is None or role == "":
        role = scenario.roles[0]
    if role not in scenario.roles:
        raise CopilotError(422, "bad_role", "Unknown role for this scenario.")
    return str(role), role == scenario.roles[1]


def _scan_user_text(label: str, value: str, mode: str) -> None:
    if mode == "non_nda" and scan_text_for_ip_leaks(value, label):
        raise CopilotError(422, "reserved_term", "The text contains a reserved term and was rejected.")


def evaluate_step(
    scenario_id: str, step: Any, role: Any = None, mode: str = "non_nda", custom_query: Optional[str] = None
) -> Dict[str, Any]:
    scenario = _require_scenario(scenario_id)
    k = _require_step(scenario, step)
    role_name, privileged = _require_role(scenario, role)
    spec = scenario.steps[k - 1]
    state = get_replay(scenario.scenario_id, k)
    judge = custom_query is None
    if custom_query is not None:
        text = " ".join(str(custom_query).split())
        if not text or len(text) > MAX_QUERY_CHARS:
            raise CopilotError(422, "bad_query", f"Query must be 1-{MAX_QUERY_CHARS} characters.")
        _scan_user_text("query", text, mode)
        base_q = spec.queries[0]
        queries = [(QuerySpec("custom", text, base_q.entity, base_q.slot, ""), text)]
    else:
        queries = [(q, q.text) for q in spec.queries]

    arms_out: List[Dict[str, Any]] = []
    for arm_id in ARM_IDS:
        q_results = [_eval_query(arm_id, state, spec, q, t, privileged, judge) for q, t in queries]
        label, desc = ARM_PUBLIC[arm_id]
        arm = {"arm_id": arm_id, "label": label, "description": desc,
               "queries": [{k2: v for k2, v in qr.items() if k2 != "_slot_status"} for qr in q_results],
               "summary": _arm_summary(q_results)}
        if mode == "nda":
            arm["internal"] = {"implementation": ARM_INTERNAL[arm_id],
                               "slot_status_raw": [qr["_slot_status"] for qr in q_results]}
        arms_out.append(arm)

    lin = state.lineage
    n_checked, mismatches = lin.oracle_check()
    visible: List[Dict[str, Any]] = []
    hidden_count = 0
    for doc_id, rec in lin.docs.items():
        elig = lin.eligibility(doc_id)
        citer = lin.citer_of_non_admit(doc_id)
        if (elig != "ADMIT" or citer) and not privileged:
            hidden_count += 1
            continue
        row = {
            "id": doc_id, "title": rec.spec.title, "system": rec.spec.system, "state": STATE_LABEL.get(elig, elig),
            "status_note": REASON_LABEL.get(lin.store.reason.get(doc_id, ""), "Other"),
            "text": rec.text,
            "flag": ("Text physically erased; only the content hash remains" if rec.tombstone_sha is not None
                     else "Cites a withdrawn record" if citer and elig == "ADMIT" else None),
            "erased_content_sha256": rec.tombstone_sha,
            "added_by_policy": rec.synthetic_guard,
        }
        if mode == "nda":
            row["internal"] = {"eligibility": elig, "reason": lin.store.reason.get(doc_id), "val": lin.store.val.get(doc_id),
                               "cur": lin.store.cur.get(doc_id)}
        visible.append(row)
    links = [{"from": s, "to": d, "kind": EDGE_LABEL.get(t, t)} for s, d, t in lin.edge_log
             if privileged or (lin.eligibility(s) == "ADMIT" and lin.eligibility(d) == "ADMIT")]

    engine_ids = {d for d, _ in [(i, 0) for i in lin.docs]}
    truth_current = sorted(spec.expected_current)
    kernel_current = sorted(i for i in engine_ids if lin.eligibility(i) == "ADMIT")
    kernel_contested = sorted(i for i in engine_ids if lin.eligibility(i) == "CONTESTED")
    payload: Dict[str, Any] = {
        "mode": mode,
        "scenario": {"id": scenario.scenario_id, "title": scenario.title, "domain": scenario.domain,
                     "summary": scenario.summary, "n_steps": len(scenario.steps)},
        "step": {"index": k, "title": spec.title, "narrative": spec.narrative,
                 "has_retraction": spec.retraction is not None,
                 "retraction_reversible": bool(spec.retraction is not None and not spec.retraction.irreversible)},
        "role": {"id": role_name, "privileged": privileged,
                 "label": scenario.role_labels[1 if privileged else 0]},
        "custom_query": not judge,
        "arms": arms_out,
        "records": {"visible": visible, "hidden_from_this_role": hidden_count, "links": links},
        "ground_truth": {
            "current_ids": truth_current,
            "disputed_ids": sorted(spec.expected_contested),
            "matches_control_plane": bool(truth_current == kernel_current and sorted(spec.expected_contested) == kernel_contested),
        },
        "cross_check": {"nodes_checked": n_checked, "mismatches": mismatches},
        "reader_policy": READER_POLICY,
        "retrieval": RETRIEVAL_PUBLIC,
        "vectors": DENSE_PUBLIC,
        "disclosures": [
            "All records are synthetic and hand-authored for this demonstration.",
            "Ground truth for each step is authored in advance; custom queries report exposure only, with no accuracy claim.",
            "The overwrite arm is a policy emulation, not a live product.",
            "Latency is not reported here; use the benchmark endpoint, which times a warm store.",
        ],
    }
    if mode == "nda":
        payload["internal"] = {
            "engine": "ek5_v3_stratified.StratifiedEvidenceStore", "retrieval": RETRIEVAL_INTERNAL,
            "dense_backend": DENSE_INTERNAL, "edge_log": [list(e) for e in lin.edge_log],
            "transitioned_this_step": {k2: list(v) for k2, v in lin.step_applies[-1].transitioned.items()},
            "tombstoned_this_step": lin.step_applies[-1].tombstoned,
            "guarded_this_step": lin.step_applies[-1].guarded,
            "telemetry_blocked_edges": [list(e) for e in lin.store.get_telemetry().blocked_authority_edges],
        }
    return payload


def scoreboard(mode: str = "non_nda") -> Dict[str, Any]:
    """Aggregate every scenario x step (standard role) from live evaluations. No stored numbers."""
    totals = {a: {"correct": 0, "judged": 0, "stale_unflagged": 0, "forbidden_text": 0} for a in ARM_IDS}
    rows: List[Dict[str, Any]] = []
    mism = 0
    checked = 0
    truth_ok = True
    for sc in SCENARIOS.values():
        for k in range(1, len(sc.steps) + 1):
            ev = evaluate_step(sc.scenario_id, k, sc.roles[0], mode)
            mism += ev["cross_check"]["mismatches"]
            checked += ev["cross_check"]["nodes_checked"]
            truth_ok = truth_ok and ev["ground_truth"]["matches_control_plane"]
            row = {"scenario": sc.scenario_id, "step": k, "arms": {}}
            for arm in ev["arms"]:
                s = arm["summary"]
                t = totals[arm["arm_id"]]
                t["correct"] += s["correct"]
                t["judged"] += s["judged"]
                t["stale_unflagged"] += s["stale_unflagged_count"]
                t["forbidden_text"] += s["forbidden_text_count"]
                row["arms"][arm["arm_id"]] = s
            rows.append(row)
    return {
        "mode": mode,
        "arms": [{"arm_id": a, "label": ARM_PUBLIC[a][0], **totals[a]} for a in ARM_IDS],
        "rows": rows,
        "cross_check": {"nodes_checked": checked, "mismatches": mism},
        "ground_truth_matches_control_plane_everywhere": truth_ok,
        "note": "Computed live from the three pipelines at request time over synthetic fixtures; not a benchmark of any live product.",
    }


# ---------------------------------------------------------------------------
# Custom-evidence injection (oracle agreement only; no accuracy claim)
# ---------------------------------------------------------------------------
def inject_evidence(body: Mapping[str, Any], mode: str) -> Dict[str, Any]:
    events_in = body.get("events")
    edges_in = body.get("edges", [])
    if not isinstance(events_in, list) or not events_in:
        raise CopilotError(422, "bad_events", "events must be a non-empty list.")
    if not isinstance(edges_in, list):
        raise CopilotError(422, "bad_edges", "edges must be a list.")
    if len(events_in) > MAX_INJECT_EVENTS or len(edges_in) > MAX_INJECT_EDGES:
        raise CopilotError(422, "too_large", "Too many events or edges.")
    store = StratifiedEvidenceStore()
    events: List[StratifiedEvent] = []
    seen: Set[str] = set()
    for i, e in enumerate(events_in):
        if not isinstance(e, dict):
            raise CopilotError(422, "bad_events", "Each event must be an object.")
        eid = str(e.get("id", ""))
        if not _ID_RE.match(eid) or eid in seen:
            raise CopilotError(422, "bad_event_id", "Event ids must be unique and match [A-Za-z0-9_.:-]{1,64}.")
        seen.add(eid)
        text = str(e.get("text", ""))
        entity = str(e.get("entity", "entity"))
        slot = str(e.get("slot", "slot"))
        cv = str(e.get("claim_value", ""))
        for label, val in (("text", text), ("entity", entity), ("slot", slot), ("claim_value", cv), ("id", eid)):
            if len(val) > 2000:
                raise CopilotError(422, "too_large", "A field is too long.")
            _scan_user_text(label, val, mode)
        try:
            tier = int(e.get("tier", 1))
        except (TypeError, ValueError):
            raise CopilotError(422, "bad_tier", "tier must be an integer.") from None
        if not 0 <= tier <= 9:
            raise CopilotError(422, "bad_tier", "tier must be between 0 and 9.")
        events.append(StratifiedEvent(event_id=eid, entity=entity, slot=slot, claim_value=cv, text=text,
                                      valid_time=i + 1, tx_time=i + 1, tenant=TENANT, domain="general",
                                      authority_tier=tier))
    atts: List[EdgeAttestation] = []
    for j, ed in enumerate(edges_in):
        if not isinstance(ed, dict):
            raise CopilotError(422, "bad_edges", "Each edge must be an object.")
        src, dst, kind = str(ed.get("from", "")), str(ed.get("to", "")), str(ed.get("kind", ""))
        if src not in seen or dst not in seen:
            raise CopilotError(422, "dangling_edge", "Every edge must connect two supplied events.")
        etype = PUBLIC_EDGE_INPUT.get(kind)
        if etype is None and mode == "nda" and kind in V3_VALID_EDGE_TYPES:
            etype = kind
        if etype is None:
            raise CopilotError(422, "bad_edge_kind", "Unknown edge kind. Use: " + ", ".join(sorted(PUBLIC_EDGE_INPUT)) + ".")
        atts.append(EdgeAttestation(attestation_id=f"inj:{j}", src_id=src, dst_id=dst, edge_type=etype,
                                    valid_time=len(events) + j + 1, tx_time=len(events) + j + 1))
    try:
        delta = store.apply_mutation(events, atts)
    except SupersedeCycleError:
        raise CopilotError(422, "replacement_cycle", "The replacement links form a cycle and were rejected.") from None
    except ValueError:
        raise CopilotError(422, "invalid_graph", "The evidence graph is invalid.") from None
    oracle = compute_alternating_wfs_oracle(store.events.keys(), store.refs, store.deps_v)
    mism = sum(1 for nid, v in oracle.items() if store.val.get(nid) != v)
    telem = store.get_telemetry()
    records = []
    for ev in events:
        elig = store.eligibility.get(ev.event_id, "ADMIT")
        row = {"id": ev.event_id, "state": STATE_LABEL.get(elig, elig),
               "status_note": REASON_LABEL.get(store.reason.get(ev.event_id, ""), "Other")}
        if mode == "nda":
            row["internal"] = {"eligibility": elig, "reason": store.reason.get(ev.event_id), "val": store.val.get(ev.event_id)}
        records.append(row)
    slots: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for ev in events:
        key = (ev.entity, ev.slot)
        if key not in slots:
            res = store.resolve_slot(TENANT, ev.entity, ev.slot)
            slots[key] = {"entity": ev.entity, "slot": ev.slot,
                          "status": {"ADMIT": "Single current value", "CONTESTED": "In dispute (abstain)",
                                     "EMPTY": "No current evidence"}[res["status"]],
                          "value": res.get("claim_value")}
    out: Dict[str, Any] = {
        "mode": mode,
        "n_events": len(events), "n_edges": len(atts),
        "records": records,
        "slots": list(slots.values()),
        "blocked_by_authority_count": len(telem.blocked_authority_edges),
        "withdrawn_count": sum(1 for r in records if r["state"] == "Withdrawn"),
        "disputed_count": sum(1 for r in records if r["state"] == "Disputed"),
        "cross_check": {"nodes_checked": len(oracle), "mismatches": mism, "agrees": mism == 0},
        "claim_scope": "Oracle agreement only. No accuracy claim is made for custom evidence because there is no authored ground truth.",
    }
    if mode == "nda":
        out["internal"] = {"mutation_work_stats": delta.work_stats}
    return out


# ---------------------------------------------------------------------------
# Compensating-event undo (preserves bitemporal history)
# ---------------------------------------------------------------------------
def undo_retraction(scenario_id: str, step: Any, mode: str = "non_nda") -> Dict[str, Any]:
    scenario = _require_scenario(scenario_id)
    k = _require_step(scenario, step)
    spec = scenario.steps[k - 1]
    if spec.retraction is None:
        raise CopilotError(422, "no_retraction", "This step has no retraction to reverse.")
    if spec.retraction.irreversible:
        raise CopilotError(422, "irreversible", "Erased content is physically destroyed; this retraction cannot be reversed.")
    state = replay(scenario_id, k)
    lin = state.lineage
    before = {i: lin.eligibility(i) for i in lin.docs}
    retractor = lin.store.events[spec.retraction.retractor_id]
    rescind_id = f"{retractor.event_id}_rescinded"
    t_before = lin.clock
    ev = StratifiedEvent(
        event_id=rescind_id, entity=retractor.entity, slot="rescission", claim_value="",
        text=f"Compensating record: the retraction {retractor.event_id} was rescinded by a higher authority.",
        valid_time=lin.tick(), tx_time=lin.clock, tenant=TENANT, domain="general", authority_tier=retractor.authority_tier + 1,
    )
    att = lin._att(rescind_id, retractor.event_id, "REFUTE")
    delta = lin.store.apply_mutation([ev], [att])
    after = {i: lin.eligibility(i) for i in lin.store.events}
    snap = lin.store.as_of(t_before, t_before)
    replayed = {i: snap.eligibility.get(i) for i in before}
    restored = sorted(i for i in before if before[i] != after[i])
    out: Dict[str, Any] = {
        "mode": mode, "scenario_id": scenario_id, "step": k,
        "compensating_record_id": rescind_id,
        "mechanism": "A new higher-authority record was appended; nothing was deleted or edited.",
        "changed": [{"id": i, "before": STATE_LABEL.get(before[i], before[i]), "after": STATE_LABEL.get(after[i], after[i])} for i in restored],
        "history_preserved": bool(replayed == before),
        "history_check": "The store was re-materialized as of the moment before the compensating record and compared record by record.",
        "cross_check": {"nodes_checked": lin.oracle_check()[0], "mismatches": lin.oracle_check()[1]},
    }
    if mode == "nda":
        out["internal"] = {"transitioned": {k2: list(v) for k2, v in delta.transitioned.items()}}
    return out


# ---------------------------------------------------------------------------
# Warm-store latency benchmark (non-deterministic; never part of a signed payload)
# ---------------------------------------------------------------------------
def warm_benchmark(repeats: int = 30, scenario_id: str = "w1") -> Dict[str, Any]:
    scenario = _require_scenario(scenario_id)
    if not 5 <= repeats <= 500:
        raise CopilotError(422, "bad_repeats", "repeats must be between 5 and 500.")
    t0 = time.perf_counter()
    state = replay(scenario_id, len(scenario.steps))
    rebuild_ms = (time.perf_counter() - t0) * 1000.0
    lin = state.lineage
    anchor = scenario.steps[0].docs[0].doc_id
    work_ms: List[float] = []
    outer_ms: List[float] = []
    for i in range(repeats):
        child = StratifiedEvent(event_id=f"bench_child_{i}", entity="bench", slot=f"s{i}", claim_value="v", text="bench child",
                                valid_time=lin.tick(), tx_time=lin.clock, tenant=TENANT, authority_tier=1)
        retractor = StratifiedEvent(event_id=f"bench_ret_{i}", entity="bench", slot=f"r{i}", claim_value="", text="bench retraction",
                                    valid_time=lin.tick(), tx_time=lin.clock, tenant=TENANT, authority_tier=3)
        atts = [lin._att(child.event_id, anchor, "COPY_OF"), lin._att(retractor.event_id, child.event_id, "REFUTE")]
        t1 = time.perf_counter()
        delta = lin.store.apply_mutation([child, retractor], atts)
        outer_ms.append((time.perf_counter() - t1) * 1000.0)
        work_ms.append(float(delta.work_stats["wall_ms"]))
    arr = np.asarray(work_ms)
    outer = np.asarray(outer_ms)
    return {
        "repeats": repeats,
        "store_nodes_at_end": len(lin.store.events),
        "fixture_nodes_before": len(lin.docs),
        "incremental_update_ms": {"p50": round(float(np.percentile(arr, 50)), 4), "p95": round(float(np.percentile(arr, 95)), 4)},
        "incremental_update_outer_ms": {"p50": round(float(np.percentile(outer, 50)), 4), "p95": round(float(np.percentile(outer, 95)), 4)},
        "full_fixture_rebuild_ms": round(rebuild_ms, 4),
        "scope": ("Warm store, fixture-sized graph (tens of nodes), one child plus one retraction per repeat on one machine. "
                  "It says nothing about production-scale graphs. The rebuild figure is shown separately and is not the update latency."),
        "deterministic": False,
    }


# ---------------------------------------------------------------------------
# Receipts: canonical JSON (integers only), SHA-256 hash chain, Ed25519 signatures
# ---------------------------------------------------------------------------
def _reject_floats(obj: Any, path: str = "$") -> None:
    if isinstance(obj, float):
        raise ValueError(f"float not allowed in canonical payload at {path}")
    if isinstance(obj, dict):
        for k, v in obj.items():
            _reject_floats(v, f"{path}.{k}")
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            _reject_floats(v, f"{path}[{i}]")


def canonical_json(obj: Any) -> bytes:
    _reject_floats(obj)
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def engine_file_sha256() -> str:
    return hashlib.sha256(ENGINE_FILE_SHA256_SOURCE.read_bytes()).hexdigest()


GENESIS_HASH = "0" * 64


class CopilotService:
    """Process-level state: the receipt ledger, the ephemeral signing key and the NDA gate."""

    def __init__(
        self,
        *,
        nda_token: Optional[str] = None,
        clock_us: Optional[Callable[[], int]] = None,
        private_key: Optional[Ed25519PrivateKey] = None,
    ) -> None:
        self.nda_token = nda_token if nda_token else None
        self._clock_us = clock_us or (lambda: time.time_ns() // 1000)
        self._key = private_key or Ed25519PrivateKey.generate()
        raw = self._key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        self.public_key_b64 = base64.b64encode(raw).decode("ascii")
        self.key_id = hashlib.sha256(raw).hexdigest()[:16]
        self._lock = threading.Lock()
        self._ledgers: Dict[str, List[Dict[str, Any]]] = {"non_nda": [], "nda": []}

    @property
    def ledger(self) -> List[Dict[str, Any]]:
        return self._ledgers["non_nda"]

    # -- NDA gate -------------------------------------------------------
    def check_mode(self, requested: Any, headers: Mapping[str, str]) -> str:
        mode = "non_nda" if requested in (None, "") else str(requested)
        if mode not in MODES:
            raise CopilotError(422, "bad_mode", "Unknown mode.")
        if mode == "nda":
            supplied = headers.get(NDA_TOKEN_HEADER, "")
            if not self.nda_token or not hmac.compare_digest(supplied.encode("utf-8"), self.nda_token.encode("utf-8")):
                raise CopilotError(403, "mode_unavailable", "This mode is not available for this request.")
        return mode

    # -- receipts ---------------------------------------------------------
    def issue_receipt(self, scenario_id: str, step: Any, mode: str) -> Dict[str, Any]:
        scenario = _require_scenario(scenario_id)
        k = _require_step(scenario, step)
        spec = scenario.steps[k - 1]
        if spec.retraction is None:
            raise CopilotError(422, "no_retraction", "Receipts are issued for retraction steps only.")
        state = get_replay(scenario_id, k)
        lin = state.lineage
        applied = lin.step_applies[-1]
        withdrawn = sorted({spec.retraction.target_id} | {n for n, (_, now) in applied.transitioned.items() if now != "ADMIT"})
        n_checked, mism = lin.oracle_check()
        tombstones = [{"record_id": i, "erased_content_sha256": lin.docs[i].tombstone_sha}
                      for i in sorted(lin.docs) if lin.docs[i].tombstone_sha is not None]
        payload: Dict[str, Any] = {
            "receipt_kind": "retraction_audit_receipt",
            "schema": "copilot_receipt_v1",
            "mode": mode,
            "scenario_id": scenario_id,
            "scenario_title": scenario.title,
            "step": k,
            "step_title": spec.title,
            "retractor_id": spec.retraction.retractor_id,
            "target_id": spec.retraction.target_id,
            "withdrawn_lineage": withdrawn,
            "records_after": {i: STATE_LABEL.get(lin.eligibility(i), lin.eligibility(i)) for i in sorted(lin.docs)},
            "tombstones": tombstones,
            "cross_check": {"nodes_checked": n_checked, "mismatches": mism},
            "decision_engine_sha256": engine_file_sha256(),
            "vectors": DENSE_PUBLIC if mode == "non_nda" else DENSE_INTERNAL,
            "key_id": self.key_id,
            "key_disclosure": KEY_DISCLOSURE,
            "clock_disclosure": CLOCK_DISCLOSURE,
            "disclaimer": DISCLAIMER,
        }
        if mode == "nda":
            payload["internal"] = {
                "edge_types": sorted({t for _, _, t in lin.edge_log}),
                "eligibility_raw": {i: lin.eligibility(i) for i in sorted(lin.docs)},
            }
        with self._lock:
            chain_list = self._ledgers[mode]
            prev = chain_list[-1]["entry_hash"] if chain_list else GENESIS_HASH
            payload["seq"] = len(chain_list) + 1
            payload["prev_hash"] = prev
            payload["issued_at_us"] = int(self._clock_us())
            entry_hash = hashlib.sha256(prev.encode("ascii") + canonical_json(payload)).hexdigest()
            signature = self._key.sign(entry_hash.encode("ascii"))
            entry = {
                "payload": payload,
                "prev_hash": prev,
                "entry_hash": entry_hash,
                "signature_b64": base64.b64encode(signature).decode("ascii"),
                "public_key_b64": self.public_key_b64,
                "algorithm": "Ed25519 over the ASCII hex of SHA-256(prev_hash || canonical_json(payload))",
            }
            chain_list.append(entry)
        return entry

    def verify_receipt(self, entry: Any) -> Dict[str, Any]:
        reasons: List[str] = []
        checks = {"hash_ok": False, "signature_ok": False, "key_ok": False, "chain_link_ok": False}
        in_ledger = False
        try:
            payload = entry["payload"]
            prev = entry["prev_hash"]
            claimed_hash = entry["entry_hash"]
            sig = base64.b64decode(entry["signature_b64"], validate=True)
            pub_b64 = entry["public_key_b64"]
        except (KeyError, TypeError, ValueError):
            return {"valid": False, "checks": checks, "in_ledger": False, "reasons": ["malformed receipt"]}
        try:
            recomputed = hashlib.sha256(str(prev).encode("ascii") + canonical_json(payload)).hexdigest()
        except (ValueError, TypeError, UnicodeEncodeError):
            return {"valid": False, "checks": checks, "in_ledger": False, "reasons": ["payload is not canonicalizable"]}
        checks["hash_ok"] = hmac.compare_digest(recomputed, str(claimed_hash))
        if not checks["hash_ok"]:
            reasons.append("hash does not match payload")
        checks["key_ok"] = pub_b64 == self.public_key_b64 and isinstance(payload, dict) and payload.get("key_id") == self.key_id
        if not checks["key_ok"]:
            reasons.append("receipt was not signed by this server's key")
        try:
            Ed25519PublicKey.from_public_bytes(base64.b64decode(self.public_key_b64)).verify(sig, recomputed.encode("ascii"))
            checks["signature_ok"] = checks["hash_ok"]
        except InvalidSignature:
            reasons.append("signature invalid")
        checks["chain_link_ok"] = isinstance(payload, dict) and payload.get("prev_hash") == prev
        if not checks["chain_link_ok"]:
            reasons.append("payload prev_hash differs from entry prev_hash")
        with self._lock:
            seq_val = payload.get("seq") if isinstance(payload, dict) else None
            rec_mode = str(payload.get("mode", "non_nda")) if isinstance(payload, dict) else "non_nda"
            chain_list = self._ledgers.get(rec_mode, self._ledgers["non_nda"])
            if isinstance(seq_val, int) and not isinstance(seq_val, bool) and 1 <= seq_val <= len(chain_list):
                in_ledger = chain_list[seq_val - 1]["entry_hash"] == claimed_hash
        valid = all(checks.values())
        return {"valid": valid, "checks": checks, "in_ledger": in_ledger, "reasons": reasons}

    def verify_ledger(self, mode: str = "non_nda") -> Dict[str, Any]:
        with self._lock:
            entries = list(self._ledgers.get(mode, self._ledgers["non_nda"]))
        prev = GENESIS_HASH
        for i, e in enumerate(entries, start=1):
            res = self.verify_receipt(e)
            if not res["valid"] or e["prev_hash"] != prev or e["payload"]["seq"] != i:
                return {"intact": False, "length": len(entries), "first_bad_seq": i}
            prev = e["entry_hash"]
        return {"intact": True, "length": len(entries), "head_hash": prev}


_DEFAULT_SERVICE: Optional[CopilotService] = None
_DEFAULT_LOCK = threading.Lock()


def get_service() -> CopilotService:
    global _DEFAULT_SERVICE
    with _DEFAULT_LOCK:
        if _DEFAULT_SERVICE is None:
            _DEFAULT_SERVICE = CopilotService(nda_token=os.environ.get("CHELATED_COPILOT_NDA_TOKEN"))
        return _DEFAULT_SERVICE


def set_service(service: Optional[CopilotService]) -> None:
    global _DEFAULT_SERVICE
    with _DEFAULT_LOCK:
        _DEFAULT_SERVICE = service


# ---------------------------------------------------------------------------
# HTTP dispatch (pure functions; no sockets needed for tests)
# ---------------------------------------------------------------------------
Response = Tuple[int, str, bytes]
_JSON = "application/json; charset=utf-8"


def is_copilot_path(path: str) -> bool:
    clean = urlparse(path).path
    return clean in PAGE_PATHS or clean == API_PREFIX or clean.startswith(API_PREFIX + "/")


def _qp(query_params: Optional[Mapping[str, Any]], key: str, default: Optional[str] = None) -> Optional[str]:
    if not query_params:
        return default
    v = query_params.get(key)
    if isinstance(v, (list, tuple)):
        v = v[0] if v else None
    return default if v is None else str(v)


def _norm_headers(headers: Optional[Mapping[str, Any]]) -> Dict[str, str]:
    return {str(k).lower(): str(v) for k, v in (headers or {}).items()}


def _json_response(status: int, payload: Mapping[str, Any], mode: str = "non_nda") -> Response:
    text = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True)
    if mode == "non_nda":
        try:
            assert_non_nda_clean(text, "response")
        except ValueError:
            return 500, _JSON, json.dumps({"error": "projection_failed_closed", "message": "Response withheld."}).encode("utf-8")
    return status, _JSON, text.encode("utf-8")


def _error_response(err: CopilotError) -> Response:
    return _json_response(err.status, {"error": err.code, "message": err.message})


_HTML_SCAN_CACHE: Dict[Tuple[int, int], bool] = {}


def _serve_page() -> Response:
    try:
        data = HTML_PATH.read_bytes()
        st = HTML_PATH.stat()
    except OSError:
        return _error_response(CopilotError(404, "page_missing", "The page is not installed."))
    key = (st.st_mtime_ns, st.st_size)
    clean = _HTML_SCAN_CACHE.get(key)
    if clean is None:
        clean = not scan_text_for_ip_leaks(data.decode("utf-8"), "page")
        _HTML_SCAN_CACHE[key] = clean
    if not clean:
        return _error_response(CopilotError(500, "projection_failed_closed", "Response withheld."))
    return 200, "text/html; charset=utf-8", data


def _public_scenarios() -> Dict[str, Any]:
    return {"scenarios": [
        {"id": s.scenario_id, "title": s.title, "domain": s.domain, "summary": s.summary,
         "roles": [{"id": r, "label": lab, "privileged": i == 1} for i, (r, lab) in enumerate(zip(s.roles, s.role_labels))],
         "steps": [{"index": i, "title": st.title, "has_retraction": st.retraction is not None}
                   for i, st in enumerate(s.steps, start=1)]}
        for s in SCENARIOS.values()
    ]}


def handle_copilot_get(
    path: str, query_params: Optional[Mapping[str, Any]] = None, headers: Optional[Mapping[str, Any]] = None,
    service: Optional[CopilotService] = None,
) -> Response:
    svc = service or get_service()
    hdr = _norm_headers(headers)
    clean = urlparse(path).path
    mode = "non_nda"
    try:
        if clean in PAGE_PATHS:
            return _serve_page()
        if not (clean == API_PREFIX or clean.startswith(API_PREFIX + "/")):
            raise CopilotError(404, "not_found", "Not found.")
        sub = clean[len(API_PREFIX):].rstrip("/")
        if sub in ("", "/health"):
            return _json_response(200, {"status": "ok", "service": "compliance-copilot", "key_id": svc.key_id,
                                        "nda_mode_configured": bool(svc.nda_token)})
        mode = svc.check_mode(_qp(query_params, "mode"), hdr)
        if sub == "/scenarios":
            return _json_response(200, _public_scenarios(), mode)
        if sub == "/step":
            payload = evaluate_step(str(_qp(query_params, "scenario", "")), _qp(query_params, "step", "1"),
                                    _qp(query_params, "role"), mode)
            return _json_response(200, payload, mode)
        if sub == "/scoreboard":
            return _json_response(200, scoreboard(mode), mode)
        if sub == "/pubkey":
            return _json_response(200, {"algorithm": "Ed25519", "public_key_b64": svc.public_key_b64, "key_id": svc.key_id,
                                        "disclosure": KEY_DISCLOSURE}, mode)
        if sub == "/ledger":
            return _json_response(200, {"entries": list(svc._ledgers[mode]), "chain": svc.verify_ledger(mode)}, mode)
        if sub == "/benchmark":
            try:
                reps = int(_qp(query_params, "repeats", "30") or "30")
            except ValueError:
                raise CopilotError(422, "bad_repeats", "repeats must be an integer.") from None
            return _json_response(200, warm_benchmark(reps), mode)
        raise CopilotError(404, "not_found", "Not found.")
    except CopilotError as err:
        return _error_response(err)


def handle_copilot_post(
    path: str, query_params: Optional[Mapping[str, Any]] = None, headers: Optional[Mapping[str, Any]] = None,
    body: bytes = b"", service: Optional[CopilotService] = None,
) -> Response:
    svc = service or get_service()
    hdr = _norm_headers(headers)
    clean = urlparse(path).path
    try:
        if not (clean == API_PREFIX or clean.startswith(API_PREFIX + "/")):
            raise CopilotError(404, "not_found", "Not found.")
        if body is not None and len(body) > MAX_BODY_BYTES:
            raise CopilotError(413, "too_large", "Request body too large.")
        try:
            data = json.loads(body.decode("utf-8")) if body else {}
        except (UnicodeDecodeError, ValueError):
            raise CopilotError(400, "invalid_json", "Request body is not valid JSON.") from None
        if not isinstance(data, dict):
            raise CopilotError(400, "invalid_json", "Request body must be a JSON object.")
        sub = clean[len(API_PREFIX):].rstrip("/")
        mode = svc.check_mode(data.get("mode", _qp(query_params, "mode")), hdr)
        if sub == "/query":
            payload = evaluate_step(str(data.get("scenario", "")), data.get("step", 1), data.get("role"), mode,
                                    custom_query=str(data.get("text", "")))
            return _json_response(200, payload, mode)
        if sub == "/inject":
            return _json_response(200, inject_evidence(data, mode), mode)
        if sub == "/undo":
            return _json_response(200, undo_retraction(str(data.get("scenario", "")), data.get("step", 1), mode), mode)
        if sub == "/receipt":
            return _json_response(200, svc.issue_receipt(str(data.get("scenario", "")), data.get("step", 1), mode), mode)
        if sub == "/verify-receipt":
            return _json_response(200, svc.verify_receipt(data.get("receipt", data)), mode)
        raise CopilotError(404, "not_found", "Not found.")
    except CopilotError as err:
        return _error_response(err)


# ---------------------------------------------------------------------------
# Standalone server (same-origin only: no CORS headers are ever emitted)
# ---------------------------------------------------------------------------
def make_handler(service: CopilotService) -> type:
    class CopilotRequestHandler(BaseHTTPRequestHandler):
        server_version = "ComplianceCopilot/1.0"

        def _send(self, status: int, ctype: str, data: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy",
                             "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; base-uri 'none'; form-action 'none'")
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            if url.path in ("/", ""):
                self.send_response(302)
                self.send_header("Location", "/copilot")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self._send(*handle_copilot_get(url.path, parse_qs(url.query), dict(self.headers.items()), service))

        def do_POST(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            try:
                length = int(self.headers.get("Content-Length", "0") or "0")
            except ValueError:
                length = -1
            if length < 0 or length > MAX_BODY_BYTES:
                self._send(*_error_response(CopilotError(413, "too_large", "Request body too large.")))
                return
            body = self.rfile.read(length) if length else b""
            self._send(*handle_copilot_post(url.path, parse_qs(url.query), dict(self.headers.items()), body, service))

        def do_OPTIONS(self) -> None:  # noqa: N802
            self._send(*_error_response(CopilotError(405, "same_origin_only", "Cross-origin requests are not supported.")))

        def log_message(self, fmt: str, *args: Any) -> None:  # silence default stderr access log
            return

    return CopilotRequestHandler


def build_server(host: str, port: int, service: Optional[CopilotService] = None) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(service or get_service()))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the standalone Compliance Copilot server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8788)
    args = parser.parse_args(argv)
    srv = build_server(args.host, args.port)
    print(f"Compliance Copilot listening on http://{args.host}:{srv.server_address[1]}/copilot "
          f"(key_id={get_service().key_id}; nda mode {'enabled' if get_service().nda_token else 'disabled'})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
