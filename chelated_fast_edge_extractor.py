"""Wave-11 Demo D9 (Part A, Guardrail L4):
Disjoint Synthetic Claim-Extraction Corpus, LoRA SFT Trainer (`0.5B` / `1.5B`),
and Batched Persistent GPU Inference Engine (`FastEdgeClaimExtractor`).

Zero-stub production module plugging directly into `AuthorityDominatedExocortexV2`
(`ground_and_clean_claims` + `compile_claims_to_ek5_v2`) without modifying any
existing codebase file.

Key L4 invariants enforced fail-closed:
  1. 4 non-Suite-A domains (`pharma_clinical`, `supply_chain`, `telecom_bgp`, `energy_scada`)
     with balanced `150`/stance (`N=600` train) and `30`/stance (`N=120` dev) across
     `ASSERT`, `SUPERSEDE`, `REFUTE`, and `DEBUNK_RUMOR`.
  2. Hard-negative two-value contrasts:
       - Tier-1 "despite official denials" rumor echoes -> `ASSERT` (rumor value),
       - Authoritative debunks quoting a false rumor alongside the verified value ->
         `DEBUNK_RUMOR` (target = VERIFIED value),
       - Equal-tier Party-B counter-filings rejecting Party A's value ->
         `REFUTE` (target = Party B's DISTINCT counter-value so `compile_claims_to_ek5_v2`
         registers a genuine mutual conflict).
  3. Fail-closed `assert_extractor_corpus_disjoint_from_suite_a`:
       - `0` overlap on canonical entities, slots, claim values, and domains,
       - Max word 5-gram Jaccard `< 0.25` against ALL `1,020` raw documents in
         `build_prospective_realworld_drift_suite()` AND `_ECHO_VARIATION_TEMPLATES`.
  4. LoRA SFT (`r=16, lora_alpha=32`, completion-only `-100` prompt masking) with
     `merge_and_unload()` `bf16` disk-reload logit parity check `< 1e-4`.
  5. Persistent GPU `FastEdgeClaimExtractor` running all 17 docs of a Suite A case in a
     SINGLE left-padded `batch_size=17` `model.generate(..., max_new_tokens=64, do_sample=False)`
     call and logging `ms_per_case` (`p50/p95/mean`), `json_valid_rate`, and `tokens_generated`.
"""

from __future__ import annotations

import gc
import hashlib
import json
import random
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from chelated_epistemic_proposer_v2 import (
    EXTRACTION_SYSTEM_PROMPT_V2,
    AuthorityDominatedExocortexV2,
)
from chelated_raw_text_exocortex import (
    ExtractedClaimRecord,
    RawDocumentEnvelope,
    canonicalize_key,
    canonicalize_value,
    parse_extracted_claim_json,
)
from prospective_realworld_drift_corpus import (
    _ECHO_VARIATION_TEMPLATES,
    build_prospective_realworld_drift_suite,
)

REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OSV_CACHE_PATH = (
    REPO_ROOT / "artifacts" / "phase2_d1_4node_local_fleet_proof" / "_work" / "osv_cve_seeds.json"
)

VALID_STANCES: Tuple[str, ...] = ("ASSERT", "SUPERSEDE", "REFUTE", "DEBUNK_RUMOR")
SYNTHETIC_DOMAINS: Tuple[str, ...] = (
    "pharma_clinical",
    "supply_chain",
    "telecom_bgp",
    "energy_scada",
)

_WORD_RE = re.compile(r"[a-z0-9]+")
_QUERY_ATTR_RE = re.compile(
    r"What is the verified current (.+?) for ([^?]+)\?",
    re.IGNORECASE,
)
_JSON_OBJ_RE = re.compile(r"\{[^{}]*\}")


@dataclass(frozen=True)
class SyntheticExtractorExample:
    """One synthetic document + gold structured extraction target."""

    doc: RawDocumentEnvelope
    target_entity: str
    target_slot: str
    target_value: str
    target_stance: str
    domain: str
    archetype: str
    target_json: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "doc_id": self.doc.doc_id,
            "text": self.doc.text,
            "published_ts": self.doc.published_ts,
            "ingested_ts": self.doc.ingested_ts,
            "source_channel": self.doc.source_channel,
            "tenant": self.doc.tenant,
            "domain": self.domain,
            "archetype": self.archetype,
            "target_entity": self.target_entity,
            "target_slot": self.target_slot,
            "target_value": self.target_value,
            "target_stance": self.target_stance,
            "target_json": self.target_json,
        }


# ---------------------------------------------------------------------------
# Domain Blueprints (Strictly Disjoint from Suite A cybersec/sec/cloud_sre)
# ---------------------------------------------------------------------------
# Each domain defines:
#   - entity prefixes & numeric ranges (separate ranges for train vs dev!)
#   - concrete snake_case slots paired with natural-language phrases
#   - diverse value formats: version/firmware/dose strings, numeric/currency/rate
#     figures, CLI/protocol config flags, and SCREAMING_SNAKE_CASE status verdicts
_DOMAIN_BLUEPRINTS: Dict[str, Dict[str, Any]] = {
    "pharma_clinical": {
        "official_channels": ("official_nvd_advisory", "signed_security_bulletin", "sec_edgar_filing"),
        "verified_channels": ("vendor_advisory", "maintainer_release_note", "postmortem_final", "independent_security_lab"),
        "working_channels": ("internal_ops_chat", "support_ticket", "analyst_note", "stackoverflow_mirror"),
        "rumor_channels": ("rumor_feed", "social_repost", "community_forum", "unverified_blog"),
        "entity_prefixes": ("CLN-ONCO", "BIO-TRIAL", "DRUG-PK", "REG-COHORT", "PHARM-DOSE"),
        "slots": (
            ("dose_escalation_ceiling", "maximum tolerated dose ceiling", "dose"),
            ("lyophilized_vial_expiry", "stability shelf-life specification", "duration"),
            ("hepatotoxicity_signal_grade", "liver enzyme safety verdict", "code"),
            ("placebo_unblinding_protocol", "dsmb unblinding trigger policy", "flag"),
            ("assay_potency_release_floor", "batch potency acceptance floor", "metric"),
            ("sterile_fill_iso_class", "aseptic fill line clearance status", "code"),
            ("pharmacokinetic_half_life", "observed plasma clearance half-life", "metric"),
            ("cohort_enrollment_threshold", "pivotal arm accrual minimum", "metric"),
        ),
        "values": {
            "dose": (
                "12.5mg-q12h", "18.0mg-q12h", "25.0mg-q24h", "37.5mg-q24h",
                "50.0mg-iv-bolus", "7.5mg-subq-wk", "15.0mg-oral-bid", "42.0mg-infusion",
                "6.25mg-loading", "30.0mg-maintenance", "9.5mg-pediatric", "64.0mg-maxcap",
            ),
            "duration": (
                "18-months-2c", "24-months-2c", "36-months-minus20c", "12-months-ambient",
                "9-months-refrigerated", "30-months-cryo", "6-months-reconstituted", "48-months-frozen",
                "15-months-coldroom", "21-months-validated", "42-months-vault", "72-hours-thawed",
            ),
            "metric": (
                "94.2%-potency", "97.8%-potency", "91.5%-potency", "98.6%-potency",
                "6.4hr-t12", "11.8hr-t12", "14.3hr-t12", "8.9hr-t12",
                "420_PATIENTS", "580_PATIENTS", "710_PATIENTS", "845_PATIENTS",
            ),
            "flag": (
                "--dsmb-stop-alpha=0.005", "--dsmb-stop-alpha=0.012", "--interim-futility-p=0.18",
                "--protocol-amendment=rev4b", "--protocol-amendment=rev5c", "--washout-window-days=28",
                "--washout-window-days=14", "--stratify-ecog-score=true", "--central-lab-recheck=strict",
                "--blinded-audit-lock=enforced", "--adverse-triage-sla=24h", "--endpoint-adjudicate=dual",
            ),
            "code": (
                "GRADE1_TRANSIENT_ELEVATION_ONLY", "GRADE3_DRUG_INDUCED_HEPATITIS",
                "ASEPTIC_LINE_CLEARED_ISO5", "MEDIA_FILL_CONTAMINATION_HALT",
                "NO_QT_PROLONGATION_OBSERVED", "CARDIAC_REPOLARIZATION_SIGNAL_ACTIVE",
                "BIOEQUIVALENCE_MET_90CI", "DISSOLUTION_PROFILE_OUT_OF_SPEC",
                "PLACEBO_ARM_CONFOUNDED_BASELINE", "STATISTICAL_SUPERIORITY_CONFIRMED",
                "STERILITY_TEST_PASS_VERIFIED", "ENDOTOXIN_EXCURSION_QUARANTINED",
            ),
        },
    },
    "supply_chain": {
        "official_channels": ("official_nvd_advisory", "signed_security_bulletin", "sec_edgar_filing"),
        "verified_channels": ("vendor_advisory", "maintainer_release_note", "postmortem_final", "independent_security_lab"),
        "working_channels": ("internal_ops_chat", "support_ticket", "analyst_note", "stackoverflow_mirror"),
        "rumor_channels": ("rumor_feed", "social_repost", "community_forum", "unverified_blog"),
        "entity_prefixes": ("SCM-PORT", "COLD-SHIP", "LOT-CELL", "WHSE-SKU", "FRT-LANE"),
        "slots": (
            ("reefer_thermal_band", "cold-chain container temperature band", "metric"),
            ("customs_hs_tariff_code", "binding customs tariff classification", "version"),
            ("tier1_foundry_lead_time", "contracted component lead time", "duration"),
            ("container_berth_allocation", "port berth priority allocation", "code"),
            ("lithium_cell_lot_disposition", "battery cell lot release verdict", "code"),
            ("air_cargo_surcharge_rate", "contracted air-freight fuel rate", "metric"),
            ("edi_dispatch_routing_flag", "warehouse erp routing directive", "flag"),
            ("safety_stock_reorder_floor", "regional depot minimum buffer", "metric"),
        ),
        "values": {
            "version": (
                "HS-8542.31.00", "HS-8507.60.40", "HS-9018.90.80", "HS-8471.50.15",
                "HS-3004.90.92", "HS-8541.40.60", "HS-8486.20.00", "HS-8504.40.95",
                "INCOTERM-DDP-2026", "INCOTERM-FOB-ROTTERDAM", "INCOTERM-CIF-BUSAN", "INCOTERM-EXW-OSAKA",
            ),
            "duration": (
                "14-weeks-fob", "22-weeks-fob", "9-weeks-expedited", "31-weeks-constrained",
                "6-days-drayage", "11-days-transshipment", "19-days-suez-reroute", "27-days-cape-route",
                "48-hours-customs-hold", "5-days-bonded-clear", "16-weeks-tooling", "10-weeks-dual-source",
            ),
            "metric": (
                "2.0C-to-8.0C", "minus18C-to-minus15C", "15.0C-to-25.0C", "minus70C-dryice",
                "$4.85/kg-air", "$6.30/kg-air", "$3.40/kg-air", "$7.15/kg-air",
                "18,500_UNITS", "26,000_UNITS", "34,200_UNITS", "12,400_UNITS",
            ),
            "flag": (
                "--route-via-kaohsiung=primary", "--route-via-antwerp=backup", "--split-bill-lading=enabled",
                "--bonded-crossdock-hold=off", "--bonded-crossdock-hold=strict", "--reefer-telemetry-poll=5m",
                "--pallet-rfid-verify=mandatory", "--carrier-cap-teu=450", "--customs-preclear-edi=856",
                "--hazmat-class9-segregate=yes", "--fifo-lot-lock=enforced", "--backorder-auto-divert=true",
            ),
            "code": (
                "LOT_RELEASED_ZERO_DENDRITE", "THERMAL_EXCURSION_QUARANTINE",
                "BERTH_4_CONFIRMED_NO_STRIKE", "TERMINAL_CRANE_LOCKOUT_ACTIVE",
                "CUSTOMS_INSPECTION_CLEARED", "FORCED_LABOR_DETENTION_ISSUED",
                "DUAL_SUPPLIER_QUALIFIED_OK", "SINGLE_SOURCE_ANODE_SHORTAGE",
                "SEAL_INTACT_MANIFEST_VERIFIED", "CONTAINER_TAMPER_DISCREPANCY",
                "NORMAL_VESSEL_CADENCE_RESUMED", "TYPHOON_ANCHORAGE_BACKLOG",
            ),
        },
    },
    "telecom_bgp": {
        "official_channels": ("official_nvd_advisory", "signed_security_bulletin", "sec_edgar_filing"),
        "verified_channels": ("vendor_advisory", "maintainer_release_note", "postmortem_final", "independent_security_lab"),
        "working_channels": ("internal_ops_chat", "support_ticket", "analyst_note", "stackoverflow_mirror"),
        "rumor_channels": ("rumor_feed", "social_repost", "community_forum", "unverified_blog"),
        "entity_prefixes": ("ASN-EDGE", "IXP-LINK", "CORE-RTR", "MPLS-PE", "RAN-CELL"),
        "slots": (
            ("rpki_roa_validation_mode", "bgp rpki origin enforcement mode", "code"),
            ("peering_max_prefix_cap", "inbound session max-prefix limit", "metric"),
            ("optical_transponder_firmware", "certified coherent dsp firmware", "version"),
            ("bgp_hold_timer_policy", "configured bgp keepalive policy", "flag"),
            ("subsea_cable_fault_attribution", "backbone fiber attenuation fault", "code"),
            ("fiveg_core_upf_release", "production 5g upf software build", "version"),
            ("qos_dscp_marking_profile", "voice trunk dscp priority class", "flag"),
            ("packet_jitter_sla_ceiling", "measured metro ring jitter bound", "metric"),
        ),
        "values": {
            "version": (
                "DSP-7.4.12-r2", "DSP-7.4.15-ga", "DSP-8.0.3-lts", "DSP-6.9.21-legacy",
                "UPF-24.11.4", "UPF-25.02.1", "UPF-25.05.3-fips", "UPF-24.08.9",
                "IOSXR-7.11.2", "JUNOS-23.4R2-S3", "EOS-4.32.2F", "SR-OS-24.7.R1",
            ),
            "metric": (
                "850_PREFIXES", "1,400_PREFIXES", "2,250_PREFIXES", "600_PREFIXES",
                "1.8ms-p99", "3.4ms-p99", "0.9ms-p99", "6.2ms-p99",
                "400Gbps-QPSK", "800Gbps-16QAM", "200Gbps-BPSK", "600Gbps-8QAM",
            ),
            "flag": (
                "--bgp-hold-timer=90s", "--bgp-hold-timer=30s", "--bfd-multiplier=3-interval-50ms",
                "--rpki-drop-invalid=strict", "--rpki-drop-invalid=warn-only", "--graceful-restart-stale=120s",
                "--dscp-ef-46-strict-priority", "--dscp-af41-weighted-fair", "--irr-filter-as-set=AS64512",
                "--dampening-half-life=15m", "--ecmp-hash-l4-symmetric=on", "--mpls-frr-ti-lfa=enabled",
            ),
            "code": (
                "RPKI_INVALID_DROPPED_AT_EDGE", "ROUTE_LEAK_PROPAGATING_GLOBAL",
                "NO_FIBER_CUT_OPTICAL_PATCH_LOOSE", "SHARK_BITE_SHALLOW_SEGMENT_BREAK",
                "TRANSIT_PEER_AS_PATH_PREPEND_OK", "MALICIOUS_ORIGIN_HIJACK_ACTIVE",
                "COHERENT_FEC_LOCKED_HEALTHY", "EDFA_PUMP_LASER_DEGRADATION",
                "DUAL_HOMED_FAILOVER_CLEAN", "BGP_SESSION_FLAP_MD5_MISMATCH",
                "SUBSCRIBER_PLANE_UNAFFECTED", "CONTROL_PLANE_DIAMETER_OVERLOAD",
            ),
        },
    },
    "energy_scada": {
        "official_channels": ("official_nvd_advisory", "signed_security_bulletin", "sec_edgar_filing"),
        "verified_channels": ("vendor_advisory", "maintainer_release_note", "postmortem_final", "independent_security_lab"),
        "working_channels": ("internal_ops_chat", "support_ticket", "analyst_note", "stackoverflow_mirror"),
        "rumor_channels": ("rumor_feed", "social_repost", "community_forum", "unverified_blog"),
        "entity_prefixes": ("SUB-HVDC", "SCADA-RTU", "TURB-WIND", "BESS-INV", "GEN-SYNC"),
        "slots": (
            ("relay_underfrequency_trip", "protective relay trip setpoint", "metric"),
            ("plc_firmware_baseline", "signed substation plc firmware", "version"),
            ("inverter_ride_through_mode", "grid-forming inverter fault mode", "flag"),
            ("transformer_dissolved_gas_verdict", "main transformer dga inspection verdict", "code"),
            ("bess_state_of_charge_floor", "battery reserve dispatch floor", "metric"),
            ("scada_dnp3_auth_profile", "dnp3 secure authentication level", "flag"),
            ("feeder_islanding_root_cause", "confirmed feeder trip root cause", "code"),
            ("hvdc_bipole_transfer_limit", "authorized hvdc corridor limit", "metric"),
        ),
        "values": {
            "version": (
                "PLC-FW-4.18.2", "PLC-FW-4.19.0-signed", "PLC-FW-4.17.6", "PLC-FW-5.0.1-nerc",
                "RTU-OS-3.8.4", "RTU-OS-3.9.1-cip", "INV-CTRL-11.2.7", "INV-CTRL-11.4.0",
                "REL-670-v2.2.5", "SEL-421-R308", "SIPROTEC-5-v9.40", "GE-UR-v8.31",
            ),
            "metric": (
                "59.30Hz-0.4s", "59.15Hz-0.2s", "58.80Hz-0.1s", "59.50Hz-1.0s",
                "18.5%-SOC-MIN", "25.0%-SOC-MIN", "12.0%-SOC-MIN", "30.0%-SOC-MIN",
                "1,450MW-BIPOLE", "1,820MW-BIPOLE", "980MW-MONOPOLE", "1,640MW-THERMAL",
            ),
            "flag": (
                "--lvrt-reactive-kfactor=2.5", "--lvrt-reactive-kfactor=1.5", "--dnp3-sa-v5-hmac-sha256=on",
                "--dnp3-sa-v5-hmac-sha256=off", "--synchrocheck-angle-max=15deg", "--synchrocheck-angle-max=8deg",
                "--governor-droop-pct=4.0", "--governor-droop-pct=5.0", "--auto-reclose-deadtime=1.2s",
                "--modbus-tcp-readonly=strict", "--iec61850-goose-vlan=402", "--island-detect-rocof=0.5hzs",
            ),
            "code": (
                "ACETYLENE_ZERO_INSULATION_SOUND", "INTERNAL_ARCING_TAP_CHANGER_FAULT",
                "VEGETATION_CONTACT_PHASE_B", "PT_FUSE_BLOWN_FALSE_DIFFERENTIAL",
                "ZERO_CYBER_INTRUSION_SENSOR_DRIFT", "UNAUTHORIZED_PLC_LOGIC_MODIFIED",
                "COOLING_PUMP_BACKUP_ENGAGED", "BUSHING_POWER_FACTOR_EXCEEDED",
                "CAPACITOR_BANK_HARMONIC_RESONANCE", "LIGHTNING_ARRESTER_CLAMP_OK",
                "SYNCHRONOUS_CONDENSER_ONLINE", "EXCITATION_LIMITER_OVERFLUX_TRIP",
            ),
        },
    },
}


# ---------------------------------------------------------------------------
# Diverse Synthetic Prose Templates (0.00-0.15 5-gram Jaccard vs Suite A!)
# ---------------------------------------------------------------------------
_ASSERT_BASELINE_TEMPLATES: Tuple[str, ...] = (
    "Baseline dossier logged for {ent}: engineers initialized the {attr} at {v1} prior to field validation.",
    "Kickoff engineering brief covering {ent}. Working specification designates {v1} as the {attr} during phase launch.",
    "Initial commissioning memo ({ent}): operations recorded {v1} for {attr} in the starter ledger.",
    "Standing catalog entry for {ent} establishes {attr} at {v1} pending scheduled periodic review.",
)

_ASSERT_ECHO_TEMPLATES: Tuple[str, ...] = (
    "Shift turnover log mentions {ent}: crews still cite {v1} under {attr} across regional clipboards.",
    "Desk chatter recap on {ent} indicates specialists kept copying {v1} as the {attr} without checking newer filings.",
    "Forwarded mail digest for {ent} repeats {v1} for {attr} verbatim from last month's briefing sheet.",
    "Operator bulletin board ({ent}): technicians keep entering {v1} when logging the {attr} field.",
    "Unverified channel mirror concerning {ent} circulates {v1} for the {attr} among third-party observers.",
)

_ASSERT_RUMOR_SEED_TEMPLATES: Tuple[str, ...] = (
    "Anonymous wire tip alleging on {ent} that the {attr} shifted to {v1}; corroboration has not arrived.",
    "Unconfirmed street chatter targeting {ent} claims {v1} now governs the {attr} ahead of formal signoff.",
    "External rumor tracker flagged {ent}: unnamed insiders assert {v1} represents the {attr}.",
)

# HARD NEGATIVE: Tier-1 rumor echo referencing official denials while STILL asserting the rumor (`ASSERT`, target=v_rumor!)
_ASSERT_DESPITE_DENIAL_TEMPLATES: Tuple[str, ...] = (
    "Syndicated rumor commentary on {ent}: even after regulators rejected the report, chatter persists that {attr} actually stands at {v1}.",
    "Fringe blog rebuttal about {ent}: ignoring formal statements from inspectors, commenters keep insisting {v1} is the real {attr}.",
    "Viral repost thread on {ent}: skeptical observers dismiss the signed denial and repeat that {attr} moved to {v1}.",
    "Unverified social post covering {ent}: contradicting headquarters notices, anonymous accounts claim {v1} remains the true {attr}.",
)

# HARD NEGATIVE TWO-VALUE SUPERSEDE (old -> new, target=v_new!)
_SUPERSEDE_TWO_VAL_TEMPLATES: Tuple[str, ...] = (
    "Binding directive for {ent}: retires the obsolete {v_old} setting and mandates {v_new} as the active {attr}.",
    "Formal engineering revision ({ent}): replaces earlier {v_old} parameters after qualification testing; set {attr} to {v_new}.",
    "Regulatory amendment on {ent}: overrides the legacy {v_old} figure and enforces {v_new} for {attr} immediately.",
    "Approved change order regarding {ent}: deprecates {v_old} completely and promotes {v_new} as the governing {attr}.",
)

# HARD NEGATIVE THREE-VALUE LINEAGE SUPERSEDE (v1 -> v2 -> v3, target=v3!)
_SUPERSEDE_THREE_VAL_TEMPLATES: Tuple[str, ...] = (
    "Final release bulletin for {ent}: cancels both interim {v_mid} and original {v_old} schedules; the authoritative {attr} is now {v_new}.",
    "Concluding compliance restatement ({ent}): displaces prior {v_mid} and legacy {v_old} benchmarks, locking {attr} at {v_new}.",
    "Master specification update on {ent}: supersedes intermediate {v_mid} (plus initial {v_old}) and fixes {attr} at {v_new}.",
)

# HARD NEGATIVE TWO-VALUE DEBUNK_RUMOR (false rumor quoted + verified ground truth, target=v_verified!)
_DEBUNK_RUMOR_TEMPLATES: Tuple[str, ...] = (
    "Verified forensic bulletin for {ent}: explicitly rejects viral rumors alleging {v_false}; laboratory inspection proves {attr} is {v_true}.",
    "Signed compliance rebuttal ({ent}): cites and refutes fabricated claims of {v_false} — audited telemetry confirms {attr} equals {v_true}.",
    "Authoritative oversight notice on {ent}: the circulating story asserting {v_false} is false; documented evidence establishes {attr} as {v_true}.",
    "Official inspector determination for {ent}: verified {attr} is {v_true}, directly debunking uncorroborated posts claiming {v_false}.",
    "Executive audit clearance ({ent}): dismantles the unverified {v_false} speculation and certifies {v_true} as the validated {attr}.",
)

# HARD NEGATIVE PARTY-A & PARTY-B TWO-VALUE COUNTER-FILING REFUTE (Party B quotes Party A's value and asserts Party B's distinct value!)
_REFUTE_PARTY_A_TEMPLATES: Tuple[str, ...] = (
    "Primary adversarial submission (Side Alpha) on {ent}: challenges rival findings and argues {attr} equals {v_self}, awaiting neutral arbitration.",
    "Independent engineering brief (Group 1) for {ent}: disputes opposing determinations and maintains {v_self} as the {attr} pending board review.",
    "Formal contention filing (Reviewer A) covering {ent}: rejects conflicting interpretations and records {v_self} for {attr} without final settlement.",
)

_REFUTE_PARTY_B_TWO_VAL_TEMPLATES: Tuple[str, ...] = (
    "Equal-standing rebuttal dossier (Side Beta) on {ent}: rejects Side Alpha's {v_other} position and instead certifies {v_self} for {attr}, leaving arbitration unresolved.",
    "Peer counter-submission (Group 2) regarding {ent}: directly refutes the competing {v_other} conclusion and enters {v_self} as the {attr} with no umpire ruling.",
    "Parallel opposing filing (Reviewer B) for {ent}: disputes the rival {v_other} finding and asserts {attr} is {v_self}; neither side has yielded.",
    "Joint-tier contestation brief (Team Two) on {ent}: counters the {v_other} claim and places {v_self} on record for {attr} pending independent review.",
)


def _format_target_json(entity: str, slot: str, value: str, stance: str) -> str:
    return json.dumps(
        {
            "entity": entity,
            "slot": slot,
            "value": value,
            "stance": stance,
        },
        ensure_ascii=False,
    )


def build_disjoint_extractor_splits(
    n_train: int = 600,
    n_dev: int = 120,
    seed: int = 20261123,
) -> Dict[str, Any]:
    """Generate balanced, Suite-A-disjoint synthetic training (`N=600`) and dev (`N=120`) splits.

    Balances stances (`150`/stance train, `30`/stance dev) across 4 non-Suite-A domains
    (`pharma_clinical`, `supply_chain`, `telecom_bgp`, `energy_scada`) with strict
    train-vs-dev entity & value separation and hard-negative two-value contrasts.
    """
    if n_train <= 0 or n_dev <= 0:
        raise ValueError("n_train and n_dev must be positive")
    if n_train % len(VALID_STANCES) != 0:
        raise ValueError(f"n_train={n_train} must be divisible by {len(VALID_STANCES)} stances")
    if n_dev % len(VALID_STANCES) != 0:
        raise ValueError(f"n_dev={n_dev} must be divisible by {len(VALID_STANCES)} stances")

    rng = random.Random(seed)

    # Partition each domain's value pools strictly between train (first 8/12) and dev (last 4/12)
    # so DEV tests genuine unseen entities AND unseen claim values!
    split_val_pools: Dict[str, Dict[str, Dict[str, Tuple[str, ...]]]] = {"train": {}, "dev": {}}
    for dom, bp in _DOMAIN_BLUEPRINTS.items():
        split_val_pools["train"][dom] = {}
        split_val_pools["dev"][dom] = {}
        for vtype, vals in bp["values"].items():
            ordered = list(vals)
            rng.shuffle(ordered)
            cut = max(4, (len(ordered) * 2) // 3)
            split_val_pools["train"][dom][vtype] = tuple(ordered[:cut])
            split_val_pools["dev"][dom][vtype] = tuple(ordered[cut:])

    def _build_one_split(split_name: str, total_n: int, id_base: int) -> List[SyntheticExtractorExample]:
        per_stance = total_n // len(VALID_STANCES)
        examples: List[SyntheticExtractorExample] = []
        seq = 0

        for stance in VALID_STANCES:
            for idx_in_stance in range(per_stance):
                seq += 1
                dom = SYNTHETIC_DOMAINS[idx_in_stance % len(SYNTHETIC_DOMAINS)]
                bp = _DOMAIN_BLUEPRINTS[dom]
                prefix = bp["entity_prefixes"][(idx_in_stance // len(SYNTHETIC_DOMAINS)) % len(bp["entity_prefixes"])]
                ent_num = id_base + seq
                suffix = rng.choice(("A", "B", "C", "D", "X", "Z"))
                entity = f"{prefix}-{ent_num:04d}-{suffix}"

                slot_key, attr_phrase, vtype = bp["slots"][rng.randrange(len(bp["slots"]))]
                pool = list(split_val_pools[split_name][dom][vtype])
                if len(pool) < 3:
                    raise RuntimeError(f"Value pool too small for {split_name}/{dom}/{vtype}")
                chosen_vals = rng.sample(pool, 3)

                if stance == "ASSERT":
                    mod = idx_in_stance % 10
                    if mod in (0, 1, 2):
                        # Hard-negative Tier-1 "despite official denials" rumor echo -> ASSERT!
                        archetype = "tier1_rumor_despite_denial_assert"
                        target_val = chosen_vals[0]
                        text = rng.choice(_ASSERT_DESPITE_DENIAL_TEMPLATES).format(
                            ent=entity, attr=attr_phrase, v1=target_val
                        )
                        channel = rng.choice(("unverified_blog", "social_repost", "rumor_feed"))
                        pub_ts = rng.randint(15, 24)
                    elif mod in (3, 4):
                        archetype = "rumor_seed_assert"
                        target_val = chosen_vals[0]
                        text = rng.choice(_ASSERT_RUMOR_SEED_TEMPLATES).format(
                            ent=entity, attr=attr_phrase, v1=target_val
                        )
                        channel = "rumor_feed"
                        pub_ts = rng.randint(2, 6)
                    elif mod in (5, 6, 7):
                        archetype = "secondary_echo_assert"
                        target_val = chosen_vals[0]
                        text = rng.choice(_ASSERT_ECHO_TEMPLATES).format(
                            ent=entity, attr=attr_phrase, v1=target_val
                        )
                        channel = rng.choice(bp["working_channels"] + bp["rumor_channels"])
                        pub_ts = rng.randint(3, 14)
                    else:
                        archetype = "baseline_prelim_assert"
                        target_val = chosen_vals[0]
                        text = rng.choice(_ASSERT_BASELINE_TEMPLATES).format(
                            ent=entity, attr=attr_phrase, v1=target_val
                        )
                        channel = rng.choice(bp["verified_channels"])
                        pub_ts = rng.randint(1, 5)

                elif stance == "SUPERSEDE":
                    if idx_in_stance % 3 == 0:
                        archetype = "three_value_lineage_supersede"
                        v_old, v_mid, v_new = chosen_vals[0], chosen_vals[1], chosen_vals[2]
                        target_val = v_new
                        text = rng.choice(_SUPERSEDE_THREE_VAL_TEMPLATES).format(
                            ent=entity, attr=attr_phrase, v_old=v_old, v_mid=v_mid, v_new=v_new
                        )
                        channel = rng.choice(bp["official_channels"])
                        pub_ts = rng.randint(16, 25)
                    else:
                        archetype = "two_value_revision_supersede"
                        v_old, v_new = chosen_vals[0], chosen_vals[1]
                        target_val = v_new
                        text = rng.choice(_SUPERSEDE_TWO_VAL_TEMPLATES).format(
                            ent=entity, attr=attr_phrase, v_old=v_old, v_new=v_new
                        )
                        channel = rng.choice(bp["official_channels"] + bp["verified_channels"])
                        pub_ts = rng.randint(8, 22)

                elif stance == "DEBUNK_RUMOR":
                    archetype = "two_value_authoritative_debunk"
                    v_false, v_true = chosen_vals[0], chosen_vals[1]
                    target_val = v_true  # CRITICAL: verified value, NOT the quoted false rumor!
                    text = rng.choice(_DEBUNK_RUMOR_TEMPLATES).format(
                        ent=entity, attr=attr_phrase, v_false=v_false, v_true=v_true
                    )
                    channel = rng.choice(bp["official_channels"])
                    pub_ts = rng.randint(12, 22)

                else:  # REFUTE
                    if idx_in_stance % 3 == 0:
                        archetype = "party_a_adversarial_refute"
                        target_val = chosen_vals[0]
                        text = rng.choice(_REFUTE_PARTY_A_TEMPLATES).format(
                            ent=entity, attr=attr_phrase, v_self=target_val
                        )
                        channel = rng.choice(bp["official_channels"] + bp["verified_channels"])
                        pub_ts = rng.randint(10, 16)
                    else:
                        archetype = "party_b_two_value_counter_filing_refute"
                        v_other, v_self = chosen_vals[0], chosen_vals[1]
                        target_val = v_self  # CRITICAL: Party B's distinct counter-value!
                        text = rng.choice(_REFUTE_PARTY_B_TWO_VAL_TEMPLATES).format(
                            ent=entity, attr=attr_phrase, v_other=v_other, v_self=v_self
                        )
                        channel = rng.choice(bp["official_channels"] + bp["verified_channels"])
                        pub_ts = rng.randint(10, 16)

                ing_ts = pub_ts + rng.randint(0, 2)
                doc = RawDocumentEnvelope(
                    doc_id=f"synth_{split_name}_{dom}_{stance.lower()}_{seq:04d}",
                    text=text,
                    published_ts=pub_ts,
                    ingested_ts=ing_ts,
                    source_channel=channel,
                    tenant="synth_d9",
                    domain=dom,
                )
                examples.append(
                    SyntheticExtractorExample(
                        doc=doc,
                        target_entity=entity,
                        target_slot=slot_key,
                        target_value=target_val,
                        target_stance=stance,
                        domain=dom,
                        archetype=archetype,
                        target_json=_format_target_json(entity, slot_key, target_val, stance),
                    )
                )

        rng.shuffle(examples)
        return examples

    train_examples = _build_one_split("train", n_train, id_base=1000)
    dev_examples = _build_one_split("dev", n_dev, id_base=8000)

    def _bundle_sha(items: Sequence[SyntheticExtractorExample]) -> str:
        payload = json.dumps(
            [ex.to_dict() for ex in items],
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    return {
        "train": train_examples,
        "dev": dev_examples,
        "seed": int(seed),
        "domains": list(SYNTHETIC_DOMAINS),
        "train_stance_counts": dict(sorted(Counter(e.target_stance for e in train_examples).items())),
        "dev_stance_counts": dict(sorted(Counter(e.target_stance for e in dev_examples).items())),
        "train_domain_counts": dict(sorted(Counter(e.domain for e in train_examples).items())),
        "dev_domain_counts": dict(sorted(Counter(e.domain for e in dev_examples).items())),
        "train_archetype_counts": dict(sorted(Counter(e.archetype for e in train_examples).items())),
        "dev_archetype_counts": dict(sorted(Counter(e.archetype for e in dev_examples).items())),
        "train_sha256": _bundle_sha(train_examples),
        "dev_sha256": _bundle_sha(dev_examples),
    }


# ---------------------------------------------------------------------------
# Fail-Closed Suite-A Disjointness & 5-Gram Jaccard Verifier
# ---------------------------------------------------------------------------
def _word_ngrams(text: str, n: int = 5) -> Set[Tuple[str, ...]]:
    toks = _WORD_RE.findall((text or "").lower())
    if len(toks) < n:
        return {tuple(toks)} if toks else set()
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def _set_jaccard(a: Set[Any], b: Set[Any]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    if inter == 0:
        return 0.0
    return float(inter) / float(len(a) + len(b) - inter)


def assert_extractor_corpus_disjoint_from_suite_a(
    synthetic_examples: Sequence[SyntheticExtractorExample],
    *,
    osv_cache_path: Optional[Path] = None,
    max_5gram_jaccard_cap: float = 0.25,
) -> Dict[str, Any]:
    """Fail-closed verifier proving zero canonical overlap and `< 0.25` 5-gram Jaccard vs Suite A.

    Checks EVERY synthetic document against:
      - ALL canonical entities, slots, claim values, and domains across `Suite A (N=60)`,
      - ALL `1,020` raw documents in `build_prospective_realworld_drift_suite()`,
      - ALL `_ECHO_VARIATION_TEMPLATES` in `prospective_realworld_drift_corpus.py`.
    """
    cache_p = osv_cache_path or DEFAULT_OSV_CACHE_PATH
    suite_cases, suite_docs, suite_sha = build_prospective_realworld_drift_suite(cache_p)

    suite_entities: Set[str] = set()
    suite_slots: Set[str] = set()
    suite_values: Set[str] = set()
    suite_domains: Set[str] = set()
    for c in suite_cases:
        suite_domains.add(canonicalize_key(str(c["domain"])))
        suite_entities.add(canonicalize_key(str(c["subject"])))
        m_attr = _QUERY_ATTR_RE.search(str(c["query_text"]))
        if m_attr:
            suite_slots.add(canonicalize_key(m_attr.group(1)))
        gold = str(c["gold"]).strip()
        if gold.upper() != "ABSTAIN":
            suite_values.add(canonicalize_value(gold).casefold())
            suite_values.add(canonicalize_key(gold))
        for sv in c.get("stale_or_false_values") or ():
            suite_values.add(canonicalize_value(str(sv)).casefold())
            suite_values.add(canonicalize_key(str(sv)))

    synth_entities: Set[str] = set()
    synth_slots: Set[str] = set()
    synth_values: Set[str] = set()
    synth_domains: Set[str] = set()
    for ex in synthetic_examples:
        synth_domains.add(canonicalize_key(ex.domain))
        synth_entities.add(canonicalize_key(ex.target_entity))
        synth_slots.add(canonicalize_key(ex.target_slot))
        synth_values.add(canonicalize_value(ex.target_value).casefold())
        synth_values.add(canonicalize_key(ex.target_value))

    ent_inter = synth_entities & suite_entities
    slot_inter = synth_slots & suite_slots
    val_inter = synth_values & suite_values
    dom_inter = synth_domains & suite_domains

    if ent_inter or slot_inter or val_inter or dom_inter:
        raise AssertionError(
            "Synthetic extractor corpus overlaps Suite A: "
            f"domains={sorted(dom_inter)[:5]}, entities={sorted(ent_inter)[:5]}, "
            f"slots={sorted(slot_inter)[:5]}, values={sorted(val_inter)[:5]}"
        )

    # Exact inverted-index 5-gram Jaccard check across ALL 1,020 Suite A docs + echo templates
    suite_ngrams: List[Tuple[str, Set[Tuple[str, ...]]]] = []
    inv_index: Dict[Tuple[str, ...], List[int]] = defaultdict(list)

    for d in suite_docs:
        ngs = _word_ngrams(d.text, n=5)
        idx = len(suite_ngrams)
        suite_ngrams.append((d.doc_id, ngs))
        for ng in ngs:
            inv_index[ng].append(idx)

    echo_start_idx = len(suite_ngrams)
    for t_i, tmpl in enumerate(_ECHO_VARIATION_TEMPLATES):
        cleaned = re.sub(r"\{[^}]+\}", " ", tmpl)
        ngs = _word_ngrams(cleaned, n=5)
        idx = len(suite_ngrams)
        suite_ngrams.append((f"_ECHO_VARIATION_TEMPLATE_{t_i:02d}", ngs))
        for ng in ngs:
            inv_index[ng].append(idx)

    max_jac = 0.0
    max_pair = ("", "")
    max_echo_jac = 0.0
    max_echo_pair = ("", "")

    for ex in synthetic_examples:
        s_ngs = _word_ngrams(ex.doc.text, n=5)
        cand_counts: Dict[int, int] = defaultdict(int)
        for ng in s_ngs:
            for ref_idx in inv_index.get(ng, ()):
                cand_counts[ref_idx] += 1
        for ref_idx, inter_cnt in cand_counts.items():
            ref_id, ref_ngs = suite_ngrams[ref_idx]
            union_cnt = len(s_ngs) + len(ref_ngs) - inter_cnt
            jac = float(inter_cnt) / float(union_cnt) if union_cnt > 0 else 0.0
            if jac > max_jac:
                max_jac = jac
                max_pair = (ex.doc.doc_id, ref_id)
            if ref_idx >= echo_start_idx and jac > max_echo_jac:
                max_echo_jac = jac
                max_echo_pair = (ex.doc.doc_id, ref_id)

    if max_jac >= max_5gram_jaccard_cap:
        raise AssertionError(
            f"5-gram Jaccard violation: max_jaccard={max_jac:.4f} >= {max_5gram_jaccard_cap} "
            f"between {max_pair[0]} and {max_pair[1]}"
        )

    return {
        "disjoint_passed": True,
        "suite_a_corpus_sha256": suite_sha,
        "n_synthetic_docs": len(synthetic_examples),
        "n_suite_a_docs_checked": len(suite_docs),
        "n_echo_templates_checked": len(_ECHO_VARIATION_TEMPLATES),
        "domain_overlap": 0,
        "entity_overlap": 0,
        "slot_overlap": 0,
        "value_overlap": 0,
        "max_5gram_jaccard": round(max_jac, 6),
        "max_5gram_jaccard_cap": float(max_5gram_jaccard_cap),
        "max_5gram_jaccard_pair": list(max_pair),
        "max_echo_template_5gram_jaccard": round(max_echo_jac, 6),
        "max_echo_template_5gram_pair": list(max_echo_pair),
    }


def assert_train_dev_disjoint(
    train_examples: Sequence[SyntheticExtractorExample],
    dev_examples: Sequence[SyntheticExtractorExample],
) -> Dict[str, Any]:
    """Verify synthetic train and dev splits share zero entities, claim values, or doc texts."""
    tr_ents = {canonicalize_key(e.target_entity) for e in train_examples}
    dv_ents = {canonicalize_key(e.target_entity) for e in dev_examples}
    tr_vals = {canonicalize_value(e.target_value).casefold() for e in train_examples}
    dv_vals = {canonicalize_value(e.target_value).casefold() for e in dev_examples}
    tr_txts = {e.doc.text for e in train_examples}
    dv_txts = {e.doc.text for e in dev_examples}

    ent_ov = tr_ents & dv_ents
    val_ov = tr_vals & dv_vals
    txt_ov = tr_txts & dv_txts
    if ent_ov or val_ov or txt_ov:
        raise AssertionError(
            f"Train/Dev leakage: entities={len(ent_ov)}, values={len(val_ov)}, texts={len(txt_ov)}"
        )
    return {
        "train_dev_disjoint_passed": True,
        "train_n": len(train_examples),
        "dev_n": len(dev_examples),
        "entity_overlap": 0,
        "value_overlap": 0,
        "text_overlap": 0,
    }


# ---------------------------------------------------------------------------
# Prompt Formatting & Strict JSON Validity Audit
# ---------------------------------------------------------------------------
def format_extractor_user_message(doc: RawDocumentEnvelope) -> str:
    """Exact user prompt schema used by `chelated_epistemic_proposer_v2` & Wave-6 PM3 v2."""
    return f"Source channel: {doc.source_channel}\nPublished day: {doc.published_ts}\nDocument:\n{doc.text}"


def is_strict_valid_claim_json(raw_output: str) -> Tuple[bool, Optional[Dict[str, str]]]:
    """Return `(is_valid, parsed_dict)` requiring valid JSON with non-empty entity/slot/value and valid stance."""
    txt = (raw_output or "").strip()
    if not txt:
        return False, None
    obj: Optional[Dict[str, Any]] = None
    try:
        loaded = json.loads(txt)
        if isinstance(loaded, dict):
            obj = loaded
    except Exception:
        m = _JSON_OBJ_RE.search(txt)
        if m:
            try:
                loaded = json.loads(m.group(0))
                if isinstance(loaded, dict):
                    obj = loaded
            except Exception:
                obj = None
    if not obj:
        return False, None

    ent = str(obj.get("entity") or "").strip()
    slot = str(obj.get("slot") or "").strip()
    val = str(obj.get("value") or "").strip()
    stance = str(obj.get("stance") or "").strip().upper()
    if not ent or not slot or not val or stance not in VALID_STANCES:
        return False, None
    if ent.lower() == "unknown_entity" or val.upper() == "UNKNOWN":
        return False, None
    return True, {"entity": ent, "slot": slot, "value": val, "stance": stance}


def build_extractor_sft_features(
    tokenizer: Any,
    examples: Sequence[SyntheticExtractorExample],
    *,
    max_length: int = 640,
) -> Tuple[List[Dict[str, List[int]]], Dict[str, Any]]:
    """Build completion-only SFT features with `-100` on ALL prompt tokens."""
    rows: List[Dict[str, List[int]]] = []
    prompt_lens: List[int] = []
    comp_lens: List[int] = []
    unmasked_prompt_tokens = 0

    for ex in examples:
        prompt_msgs = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT_V2},
            {"role": "user", "content": format_extractor_user_message(ex.doc)},
        ]
        full_msgs = prompt_msgs + [{"role": "assistant", "content": ex.target_json}]

        prompt_str = tokenizer.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        full_str = tokenizer.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)

        p_ids: List[int] = list(tokenizer(prompt_str, add_special_tokens=False)["input_ids"])
        f_ids: List[int] = list(tokenizer(full_str, add_special_tokens=False)["input_ids"])

        if len(f_ids) > max_length:
            raise ValueError(f"Extractor SFT sequence length {len(f_ids)} > max_length={max_length} on {ex.doc.doc_id}")
        if f_ids[: len(p_ids)] != p_ids:
            raise RuntimeError(f"Chat template prefix mismatch on {ex.doc.doc_id}")
        c_ids = f_ids[len(p_ids) :]
        if not c_ids:
            raise ValueError(f"Empty completion tokens on {ex.doc.doc_id}")

        labels = [-100] * len(p_ids) + c_ids
        unmasked_prompt_tokens += sum(1 for tok_lbl in labels[: len(p_ids)] if tok_lbl != -100)
        prompt_lens.append(len(p_ids))
        comp_lens.append(len(c_ids))
        rows.append(
            {
                "input_ids": f_ids,
                "attention_mask": [1] * len(f_ids),
                "labels": labels,
            }
        )

    audit = {
        "n_examples": len(rows),
        "unmasked_prompt_token_count": int(unmasked_prompt_tokens),
        "prompt_tokens_min": int(min(prompt_lens)) if prompt_lens else 0,
        "prompt_tokens_max": int(max(prompt_lens)) if prompt_lens else 0,
        "completion_tokens_min": int(min(comp_lens)) if comp_lens else 0,
        "completion_tokens_max": int(max(comp_lens)) if comp_lens else 0,
    }
    return rows, audit


def _hash_directory(dir_path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in sorted(dir_path.rglob("*")):
        if p.is_file():
            h = hashlib.sha256()
            with p.open("rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            out[str(p.relative_to(dir_path)).replace("\\", "/")] = h.hexdigest()
    return out


# ---------------------------------------------------------------------------
# LoRA SFT Trainer (`0.5B` & `1.5B`) + `merge_and_unload()` bf16 Parity Check
# ---------------------------------------------------------------------------
def train_fast_claim_extractor_lora(
    *,
    base_model_id: str,
    train_examples: Sequence[SyntheticExtractorExample],
    output_dir: Path,
    probe_examples: Optional[Sequence[SyntheticExtractorExample]] = None,
    epochs: int = 3,
    learning_rate: float = 2e-4,
    per_device_train_batch_size: int = 8,
    gradient_accumulation_steps: int = 2,
    lora_r: int = 16,
    lora_alpha: int = 32,
    seed: int = 42,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Train completion-only LoRA SFT claim extractor, merge in `bf16`, and verify reload parity `< 1e-4`."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

    _tf_major = int(str(transformers.__version__).split(".", 1)[0])

    def _load_causal_lm(model_ref: str, dt: Any) -> Any:
        kw = {"dtype": dt} if _tf_major >= 5 else {"torch_dtype": dt}
        return AutoModelForCausalLM.from_pretrained(model_ref, **kw)

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for train_fast_claim_extractor_lora but unavailable")

    output_dir.mkdir(parents=True, exist_ok=True)
    adapter_dir = output_dir / "lora_adapter"
    merged_dir = output_dir / "merged_bf16"

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, padding_side="left")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    sft_rows, mask_audit = build_extractor_sft_features(tokenizer, train_examples)
    if mask_audit["unmasked_prompt_token_count"] != 0:
        raise AssertionError("Completion-only masking violated: prompt tokens were unmasked")

    def _left_pad_collator(features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        in_ids, attn, lbls = [], [], []
        for f in features:
            pad_n = max_len - len(f["input_ids"])
            in_ids.append([tokenizer.pad_token_id] * pad_n + f["input_ids"])
            attn.append([0] * pad_n + f["attention_mask"])
            lbls.append([-100] * pad_n + f["labels"])
        return {
            "input_ids": torch.tensor(in_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(lbls, dtype=torch.long),
        }

    # Build deterministic probe batch for bf16 merge/reload parity verification
    probe_src = list(probe_examples[:4]) if probe_examples else list(train_examples[:4])
    probe_texts = [
        tokenizer.apply_chat_template(
            [
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT_V2},
                {"role": "user", "content": format_extractor_user_message(ex.doc)},
            ],
            tokenize=False,
            add_generation_prompt=True,
        )
        for ex in probe_src
    ]
    probe_enc = tokenizer(probe_texts, return_tensors="pt", padding=True, add_special_tokens=False)

    def _probe_logits(mdl: Any) -> torch.Tensor:
        mdl.eval()
        dev = next(mdl.parameters()).device
        with torch.inference_mode():
            out = mdl(
                input_ids=probe_enc["input_ids"].to(dev),
                attention_mask=probe_enc["attention_mask"].to(dev),
                use_cache=False,
            )
            return out.logits[:, -6:, :].detach().float().cpu()

    dtype = torch.bfloat16 if (device.startswith("cuda") and torch.cuda.is_bf16_supported()) else torch.float16
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats()
    torch.manual_seed(seed)
    np.random.seed(seed)

    base_model = _load_causal_lm(base_model_id, dtype).to(device)
    base_logits = _probe_logits(base_model)

    base_model.config.use_cache = False
    base_model.gradient_checkpointing_enable()
    lora_cfg = LoraConfig(
        r=int(lora_r),
        lora_alpha=int(lora_alpha),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    peft_model = get_peft_model(base_model, lora_cfg)
    trainable_params = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in peft_model.parameters())

    est_steps = max(
        1,
        (len(sft_rows) * int(epochs))
        // max(1, int(per_device_train_batch_size) * int(gradient_accumulation_steps)),
    )
    train_args = TrainingArguments(
        output_dir=str(output_dir / "trainer_out"),
        num_train_epochs=int(epochs),
        per_device_train_batch_size=int(per_device_train_batch_size),
        gradient_accumulation_steps=int(gradient_accumulation_steps),
        learning_rate=float(learning_rate),
        lr_scheduler_type="cosine",
        warmup_steps=max(1, int(round(0.08 * est_steps))),
        bf16=(dtype == torch.bfloat16),
        fp16=(dtype == torch.float16),
        logging_steps=5,
        save_strategy="no",
        report_to="none",
        seed=int(seed),
        data_seed=int(seed),
        remove_unused_columns=False,
        gradient_checkpointing=True,
    )
    trainer = Trainer(
        model=peft_model,
        args=train_args,
        train_dataset=Dataset.from_list(sft_rows),
        data_collator=_left_pad_collator,
    )

    sample_batch = next(iter(trainer.get_train_dataloader()))
    sup_frac = float((sample_batch["labels"] != -100).float().mean().item())
    mask_audit["dataloader_supervised_token_fraction"] = round(sup_frac, 4)
    if not (0.02 < sup_frac < 0.50):
        raise AssertionError(f"Unexpected supervised token fraction in extractor SFT batch: {sup_frac:.4f}")

    t0 = time.perf_counter()
    train_out = trainer.train()
    train_sec = round(time.perf_counter() - t0, 2)
    train_loss = round(float(train_out.training_loss), 5)

    peft_model.save_pretrained(str(adapter_dir))
    peak_vram_mib = (
        round(float(torch.cuda.max_memory_allocated()) / (1024.0**2), 1)
        if device.startswith("cuda")
        else 0.0
    )
    del trainer, peft_model, base_model
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # Merge from a clean unhooked base + saved adapter so Trainer/Accelerate gradient_checkpointing hooks do not alter logits
    clean_base = _load_causal_lm(base_model_id, dtype).to(device)
    clean_peft = PeftModel.from_pretrained(clean_base, str(adapter_dir)).to(device)
    merged_model = clean_peft.merge_and_unload()
    # Cast parameters only (never downcast float32 RoPE inv_freq buffers via .to(dtype)!)
    for _p in merged_model.parameters():
        if _p.dtype != dtype:
            _p.data = _p.data.to(dtype=dtype)
    merged_model.config.use_cache = True
    merged_model.eval()
    merged_model.save_pretrained(str(merged_dir))
    tokenizer.save_pretrained(str(merged_dir))

    merged_logits = _probe_logits(merged_model)
    vs_base_shift = float((merged_logits - base_logits).abs().max().item())

    del merged_model, clean_peft, clean_base
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    # Fail-closed disk reload parity check (`< 1e-4`)
    reloaded = _load_causal_lm(str(merged_dir), dtype).to(device)
    reloaded_logits = _probe_logits(reloaded)
    reload_max_abs_diff = float((reloaded_logits - merged_logits).abs().max().item())
    del reloaded
    gc.collect()
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    if reload_max_abs_diff >= 1e-4:
        raise AssertionError(
            f"Merged bf16 extractor reload logit parity failed for {base_model_id}: {reload_max_abs_diff} >= 1e-4"
        )

    return {
        "base_model_id": base_model_id,
        "merged_model_dir": str(merged_dir),
        "adapter_dir": str(adapter_dir),
        "lora_r": int(lora_r),
        "lora_alpha": int(lora_alpha),
        "epochs": int(epochs),
        "learning_rate": float(learning_rate),
        "seed": int(seed),
        "dtype": str(dtype),
        "trainable_params": int(trainable_params),
        "total_params": int(total_params),
        "mask_audit": mask_audit,
        "train_seconds": train_sec,
        "train_loss": train_loss,
        "vs_base_max_abs_logit_shift": round(vs_base_shift, 6),
        "reload_max_abs_logit_diff": reload_max_abs_diff,
        "reload_parity_below_1e4": bool(reload_max_abs_diff < 1e-4),
        "peak_vram_mib": peak_vram_mib,
        "merged_weights_sha256": _hash_directory(merged_dir),
    }


# ---------------------------------------------------------------------------
# Batched Persistent GPU Inference Engine (`FastEdgeClaimExtractor`)
# ---------------------------------------------------------------------------
class FastEdgeClaimExtractor:
    """Persistent GPU claim extractor running `batch_size=17` Suite A cases in one `generate()` call."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str = "cuda",
        max_input_length: int = 512,
        default_max_new_tokens: int = 64,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.model_path = str(model_path)
        self.device = device
        self.max_input_length = int(max_input_length)
        self.default_max_new_tokens = int(default_max_new_tokens)

        t0 = time.perf_counter()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, padding_side="left")
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "left"

        self.dtype = (
            torch.bfloat16
            if (device.startswith("cuda") and torch.cuda.is_available() and torch.cuda.is_bf16_supported())
            else (torch.float16 if device.startswith("cuda") else torch.float32)
        )
        if device.startswith("cuda"):
            torch.cuda.reset_peak_memory_stats()
        import transformers as _tf_mod
        _dt_kw = {"dtype": self.dtype} if int(str(_tf_mod.__version__).split(".", 1)[0]) >= 5 else {"torch_dtype": self.dtype}
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, **_dt_kw).to(device)
        self.model.eval()
        self.load_seconds = round(time.perf_counter() - t0, 3)

    def warmup(self, sample_doc: Optional[RawDocumentEnvelope] = None) -> float:
        """Run one short untimed warmup pass to initialize CUDA kernels & KV allocators."""
        dummy = sample_doc or RawDocumentEnvelope(
            doc_id="warmup_0",
            text="Baseline dossier logged for CLN-WARM-0001-A: initialized dosage at 10.0mg-q24h.",
            published_ts=1,
            ingested_ts=1,
            source_channel="vendor_advisory",
            domain="pharma_clinical",
        )
        t0 = time.perf_counter()
        self.extract_batch([dummy], batch_size=1, max_new_tokens=32)
        return round((time.perf_counter() - t0) * 1000.0, 2)

    def _render_prompts(self, docs: Sequence[RawDocumentEnvelope]) -> List[str]:
        rendered: List[str] = []
        for d in docs:
            msgs = [
                {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT_V2},
                {"role": "user", "content": format_extractor_user_message(d)},
            ]
            rendered.append(
                self.tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            )
        return rendered

    def extract_batch(
        self,
        docs: Sequence[RawDocumentEnvelope],
        *,
        batch_size: int = 17,
        max_new_tokens: Optional[int] = None,
    ) -> Tuple[List[ExtractedClaimRecord], Dict[str, Any]]:
        """Extract claims for `docs` using left-padded batched greedy generation (`batch_size=17`)."""
        import torch

        if not docs:
            return [], {
                "n_docs": 0,
                "wall_ms": 0.0,
                "json_valid_count": 0,
                "json_valid_rate": 1.0,
                "tokens_generated": 0,
                "prompt_tokens": 0,
                "generate_calls": 0,
            }

        gen_cap = int(max_new_tokens if max_new_tokens is not None else self.default_max_new_tokens)
        prompts = self._render_prompts(docs)
        records: List[ExtractedClaimRecord] = []
        valid_count = 0
        total_gen_tokens = 0
        total_prompt_tokens = 0
        generate_calls = 0

        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        with torch.inference_mode():
            for start in range(0, len(prompts), int(batch_size)):
                b_prompts = prompts[start : start + int(batch_size)]
                b_docs = docs[start : start + int(batch_size)]
                enc = self.tokenizer(
                    b_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_input_length,
                    add_special_tokens=False,
                ).to(self.device)
                in_len = int(enc["input_ids"].shape[1])
                total_prompt_tokens += int(enc["attention_mask"].sum().item())

                gen = self.model.generate(
                    **enc,
                    max_new_tokens=gen_cap,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    top_k=None,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    use_cache=True,
                )
                generate_calls += 1
                gen_only = gen[:, in_len:]

                for b_i, d in enumerate(b_docs):
                    row_ids = gen_only[b_i]
                    non_pad = int((row_ids != self.tokenizer.pad_token_id).sum().item())
                    total_gen_tokens += non_pad
                    raw_txt = self.tokenizer.decode(row_ids, skip_special_tokens=True).strip()
                    ok_json, _ = is_strict_valid_claim_json(raw_txt)
                    if ok_json:
                        valid_count += 1
                    records.append(parse_extracted_claim_json(d.doc_id, raw_txt))

        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        wall_ms = (time.perf_counter() - t0) * 1000.0

        stats = {
            "n_docs": len(docs),
            "batch_size": int(batch_size),
            "max_new_tokens": gen_cap,
            "wall_ms": round(wall_ms, 3),
            "ms_per_doc": round(wall_ms / max(1, len(docs)), 3),
            "json_valid_count": int(valid_count),
            "json_valid_rate": round(float(valid_count) / float(len(docs)), 6),
            "tokens_generated": int(total_gen_tokens),
            "prompt_tokens": int(total_prompt_tokens),
            "generate_calls": int(generate_calls),
        }
        return records, stats

    def extract_cases_batched(
        self,
        case_doc_lists: Sequence[Sequence[RawDocumentEnvelope]],
        *,
        batch_size: int = 17,
        max_new_tokens: Optional[int] = None,
    ) -> Tuple[List[List[ExtractedClaimRecord]], Dict[str, Any]]:
        """Run one `batch_size=17` call per Suite A case and compute `p50/p95/mean` `ms_per_case`."""
        import torch

        all_claims: List[List[ExtractedClaimRecord]] = []
        case_ms: List[float] = []
        total_docs = 0
        total_valid = 0
        total_gen_toks = 0
        total_prompt_toks = 0
        total_calls = 0

        for docs in case_doc_lists:
            claims, st = self.extract_batch(docs, batch_size=batch_size, max_new_tokens=max_new_tokens)
            all_claims.append(claims)
            case_ms.append(float(st["wall_ms"]))
            total_docs += int(st["n_docs"])
            total_valid += int(st["json_valid_count"])
            total_gen_toks += int(st["tokens_generated"])
            total_prompt_toks += int(st["prompt_tokens"])
            total_calls += int(st["generate_calls"])

        peak_vram_mib = (
            round(float(torch.cuda.max_memory_allocated()) / (1024.0**2), 1)
            if self.device.startswith("cuda") and torch.cuda.is_available()
            else 0.0
        )
        ms_arr = np.asarray(case_ms, dtype=np.float64) if case_ms else np.zeros(1, dtype=np.float64)
        summary = {
            "model_path": self.model_path,
            "n_cases": len(case_doc_lists),
            "n_docs": int(total_docs),
            "batch_size": int(batch_size),
            "generate_calls": int(total_calls),
            "ms_per_case_p50": round(float(np.quantile(ms_arr, 0.50)), 3),
            "ms_per_case_p95": round(float(np.quantile(ms_arr, 0.95)), 3),
            "ms_per_case_mean": round(float(np.mean(ms_arr)), 3),
            "ms_per_doc_mean": round(float(np.sum(ms_arr)) / max(1, total_docs), 3),
            "json_valid_rate": round(float(total_valid) / max(1, total_docs), 6),
            "tokens_generated": int(total_gen_toks),
            "tokens_generated_mean_per_doc": round(float(total_gen_toks) / max(1, total_docs), 2),
            "tokens_generated_mean_per_case": round(float(total_gen_toks) / max(1, len(case_doc_lists)), 2),
            "prompt_tokens_mean_per_case": round(float(total_prompt_toks) / max(1, len(case_doc_lists)), 2),
            "peak_vram_mib": peak_vram_mib,
            "per_case_ms": [round(x, 3) for x in case_ms],
        }
        return all_claims, summary

    def evaluate_on_synthetic_dev(
        self,
        dev_examples: Sequence[SyntheticExtractorExample],
        *,
        batch_size: int = 17,
        max_new_tokens: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Score extractor fidelity & latency on the unseen synthetic DEV split (`N=120`)."""
        docs = [ex.doc for ex in dev_examples]
        # Chunk into 17-doc batches to measure realistic Suite-A-equivalent batch timing on DEV too
        batches = [docs[i : i + int(batch_size)] for i in range(0, len(docs), int(batch_size))]
        batched_claims, bench = self.extract_cases_batched(
            batches, batch_size=batch_size, max_new_tokens=max_new_tokens
        )
        preds = [c for grp in batched_claims for c in grp]

        ent_ok = 0
        stance_ok = 0
        val_ok = 0
        full_ok = 0
        by_stance_ok: Dict[str, List[float]] = defaultdict(list)
        by_archetype_ok: Dict[str, List[float]] = defaultdict(list)
        two_val_ok: List[float] = []

        for ex, pred in zip(dev_examples, preds):
            e_hit = canonicalize_key(pred.entity) == canonicalize_key(ex.target_entity)
            s_hit = pred.stance.upper() == ex.target_stance.upper()
            v_hit = canonicalize_value(pred.claim_value).casefold() == canonicalize_value(ex.target_value).casefold()
            all_hit = bool(e_hit and s_hit and v_hit)
            ent_ok += int(e_hit)
            stance_ok += int(s_hit)
            val_ok += int(v_hit)
            full_ok += int(all_hit)
            by_stance_ok[ex.target_stance].append(1.0 if all_hit else 0.0)
            by_archetype_ok[ex.archetype].append(1.0 if all_hit else 0.0)
            if ex.archetype in {
                "two_value_authoritative_debunk",
                "party_b_two_value_counter_filing_refute",
                "two_value_revision_supersede",
                "three_value_lineage_supersede",
                "tier1_rumor_despite_denial_assert",
            }:
                two_val_ok.append(1.0 if all_hit else 0.0)

        n = max(1, len(dev_examples))
        return {
            "n_dev": len(dev_examples),
            "json_valid_rate": bench["json_valid_rate"],
            "entity_accuracy": round(ent_ok / n, 4),
            "stance_accuracy": round(stance_ok / n, 4),
            "value_exact_accuracy": round(val_ok / n, 4),
            "full_tuple_accuracy": round(full_ok / n, 4),
            "hard_negative_contrast_accuracy": round(float(np.mean(two_val_ok)) if two_val_ok else 0.0, 4),
            "by_stance_full_accuracy": {k: round(float(np.mean(v)), 4) for k, v in sorted(by_stance_ok.items())},
            "by_archetype_full_accuracy": {k: round(float(np.mean(v)), 4) for k, v in sorted(by_archetype_ok.items())},
            "ms_per_17doc_batch_p50": bench["ms_per_case_p50"],
            "ms_per_17doc_batch_p95": bench["ms_per_case_p95"],
            "ms_per_17doc_batch_mean": bench["ms_per_case_mean"],
            "ms_per_doc_mean": bench["ms_per_doc_mean"],
            "tokens_generated": bench["tokens_generated"],
            "tokens_generated_mean_per_doc": bench["tokens_generated_mean_per_doc"],
            "peak_vram_mib": bench["peak_vram_mib"],
        }

    def close(self) -> None:
        import torch

        if hasattr(self, "model") and self.model is not None:
            del self.model
            self.model = None
        gc.collect()
        if self.device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

    def __enter__(self) -> "FastEdgeClaimExtractor":
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        self.close()


def build_exocortex_v2_from_claims(
    docs: Sequence[RawDocumentEnvelope],
    raw_claims: Sequence[ExtractedClaimRecord],
    doc_embeddings: np.ndarray,
    slot_embeddings: np.ndarray,
    *,
    channel_authority: Optional[Mapping[str, int]] = None,
) -> AuthorityDominatedExocortexV2:
    """Direct plug-in constructor connecting `FastEdgeClaimExtractor` outputs to `AuthorityDominatedExocortexV2`."""
    return AuthorityDominatedExocortexV2(
        docs=docs,
        raw_claims=raw_claims,
        doc_embeddings=doc_embeddings,
        slot_embeddings=slot_embeddings,
        channel_authority=channel_authority,
    )
