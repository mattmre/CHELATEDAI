"""Prospective Real-World Multi-Domain Raw-Text Enterprise Drift Corpus (`N=60` Timelines, `1,020` Global Docs).

Constructs a realistic, ungameable 3-domain enterprise drift benchmark grounded in live public
OSV.dev / NVD CVE vulnerability metadata, SEC EDGAR disclosure patterns, and Cloud SRE postmortems:
  - Domain 1 (`cybersec_cve`, `N=20`): Grounded in live-fetched OSV.dev CVE advisories + realistic
    DevOps Slack/forum/blog workaround echoes, false-positive rumor laundering, and dual-vendor PoC disputes.
  - Domain 2 (`sec_financial`, `N=20`): Corporate Form 8-K Item 4.02 restatements, preliminary earnings
    wire echoes, M&A/executive rumor debunks, multi-quarter guidance chains, and contested Schedule 14A filings.
  - Domain 3 (`cloud_sre`, `N=20`): Production incident runbooks, dangerous mitigation rollbacks echoed across
    on-call threads, multi-hop root-cause re-attributions, and split-brain contested telemetry.

Anti-shortcut invariants:
  1. Every document is 100% natural-language prose (no `[vt=..., tier=...]` inline metadata tags!).
  2. NO pre-extracted `StratifiedEvent` or `EdgeAttestation` edges are exposed to any evaluation arm;
     every arm receives ONLY `List[RawDocumentEnvelope]` + natural-language `query_text`.
  3. All `1,020` documents (`60 cases x 17 docs/case`, with `10` distinct natural-language echoes per case
     exceeding `top_k = 5`) reside in ONE shared global multi-entity store.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests

from chelated_raw_text_exocortex import RawDocumentEnvelope

REAL_OSV_CVE_IDS: Tuple[str, ...] = (
    "CVE-2024-3094",   # XZ Utils liblzma backdoor
    "CVE-2024-6387",   # OpenSSH RegreSSHion
    "CVE-2024-21626",  # runc container breakout
    "CVE-2024-3400",   # Palo Alto PAN-OS GlobalProtect remediation revision
    "CVE-2024-4577",   # PHP-CGI argument injection
    "CVE-2023-4863",   # libwebp heap buffer overflow
    "CVE-2023-44487",  # HTTP/2 Rapid Reset
    "CVE-2024-24790",  # Go net/netip IsPrivate
    "CVE-2024-27351",  # Django regular expression DoS
    "CVE-2024-6232",   # Python tarfile header parsing
)

_ECHO_VARIATION_TEMPLATES: Tuple[str, ...] = (
    "Quick note from our internal review: for {subject}, the {attr_phrase} remains {val}. Teams are still referencing this baseline across active tickets.",
    "Mirroring the earlier operational guidance on {subject} — everyone is tracking {val} as the {attr_phrase} in today's handoff digest.",
    "Per the circulating engineering thread regarding {subject}, {val} continues to be cited for the {attr_phrase} during triage.",
    "Community roundup for {subject}: multiple operators report relying on {val} when asked about the {attr_phrase}.",
    "Ticket sync summary ({subject}): no change flagged in our chat notes yet; {val} is listed as the {attr_phrase}.",
    "Reposting the pinned FAQ entry on {subject}: set or record the {attr_phrase} as {val} until told otherwise.",
    "Weekly digest mention — {subject} still shows {val} under {attr_phrase} across shared runbook copies.",
    "Field support escalation note on {subject}: analysts copied {val} forward as the {attr_phrase} in client trackers.",
    "Automated Slack recap for {subject}: consensus in the channel still points to {val} for the {attr_phrase}.",
    "Secondary forum mirror ({subject}): contributors reiterated {val} regarding the {attr_phrase} this morning.",
)


def fetch_or_load_osv_cve_seeds(cache_path: Path) -> Dict[str, Dict[str, Any]]:
    """Fetch real CVE advisory metadata from `api.osv.dev` once and cache deterministically."""
    if cache_path.is_file():
        return json.loads(cache_path.read_text(encoding="utf-8"))

    seeds: Dict[str, Dict[str, Any]] = {}
    for cve_id in REAL_OSV_CVE_IDS:
        summary = f"Security advisory record for {cve_id}"
        fixed_ver: Optional[str] = None
        pkg_name: Optional[str] = None
        try:
            resp = requests.get(f"https://api.osv.dev/v1/vulns/{cve_id}", timeout=6.0)
            if resp.status_code == 200:
                payload = resp.json()
                summary = str(payload.get("summary") or payload.get("details") or summary)[:220].replace("\n", " ")
                for aff in payload.get("affected") or []:
                    pkg = (aff.get("package") or {}).get("name")
                    if pkg and not pkg_name:
                        pkg_name = str(pkg)
                    for rng in aff.get("ranges") or []:
                        for ev in rng.get("events") or []:
                            if "fixed" in ev and not fixed_ver:
                                fixed_ver = str(ev["fixed"])
        except Exception:
            pass
        seeds[cve_id] = {
            "cve_id": cve_id,
            "osv_summary": summary,
            "osv_package": pkg_name or "upstream-package",
            "osv_fixed": fixed_ver,
        }

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(seeds, indent=2, sort_keys=True), encoding="utf-8")
    return seeds


def _make_echo_docs(
    case_id: str,
    domain: str,
    subject: str,
    attr_phrase: str,
    stale_val: str,
    start_pub_day: int,
    start_ing_day: int,
    channels: Sequence[str],
    n_echoes: int = 10,
) -> List[RawDocumentEnvelope]:
    out: List[RawDocumentEnvelope] = []
    for i in range(n_echoes):
        tmpl = _ECHO_VARIATION_TEMPLATES[i % len(_ECHO_VARIATION_TEMPLATES)]
        ch = channels[i % len(channels)]
        pub = start_pub_day + (i % 4)
        ing = start_ing_day + i
        out.append(
            RawDocumentEnvelope(
                doc_id=f"{case_id}_echo_{i:02d}",
                text=tmpl.format(subject=subject, attr_phrase=attr_phrase, val=stale_val),
                published_ts=pub,
                ingested_ts=ing,
                source_channel=ch,
                domain=domain,
            )
        )
    return out


def build_prospective_realworld_drift_suite(osv_cache_path: Path) -> Tuple[List[Dict[str, Any]], List[RawDocumentEnvelope], str]:
    """Build `N=60` real-world raw-text drift cases (`1,020` global documents) across 3 industries."""
    osv_seeds = fetch_or_load_osv_cve_seeds(osv_cache_path)

    # 3 domains x 4 epistemic regimes x 5 distinct real-world entities = 60 cases (17 docs each = 1,020 docs)
    cve_entities = [
        ("CVE-2024-3094", "remediated xz-utils release", "5.4.6- hotfix-disable-ifunc", "5.6.1+really5.4.5-1"),
        ("CVE-2024-6387", "safe openssh-server version", "9.6p1-grace-timer-env", "9.8p1"),
        ("CVE-2024-21626", "patched runc release", "1.1.10-fd-ulimit-cap", "1.1.12"),
        ("CVE-2024-3400", "remediated pan-os release", "11.1.2-telemetry-off", "11.1.2-h3"),
        ("CVE-2023-4863", "safe libwebp version", "1.3.1-crop-filter", "1.3.2"),
    ]
    cve_rumor_entities = [
        ("CVE-2024-4577-LINUX-MOD", "linux apache prefork exposure status", "RCE_EXPLOITABLE_ON_LINUX", "NOT_VULNERABLE_WINDOWS_ONLY"),
        ("CVE-2023-44487-QUIC-CORE", "http3 quic transport vulnerability status", "CRITICAL_STREAM_RESET_RCE", "UNIMPACTED_TCP_HTTP2_ONLY"),
        ("CVE-2024-24790-IPV4-PURE", "pure ipv4 acl bypass status", "BYPASS_ACTIVE_ON_IPV4_NATIVE", "SAFE_IPV4_MAPPED_IPV6_ONLY"),
        ("CVE-2024-27351-ORM-SQLI", "django sql injection impact status", "UNAUTHENTICATED_SQL_INJECTION", "REDOS_ONLY_NO_SQL_INJECTION"),
        ("CVE-2024-6232-ZIPFILE-STD", "python zipfile module vulnerability status", "ZIPFILE_PATH_TRAVERSAL_CONFIRMED", "UNIMPACTED_TARFILE_ONLY"),
    ]
    cve_lineage_entities = [
        ("GHSA-K8S-INGRESS-2026A", "approved ingress-nginx patched build", "v1.10.1", "v1.11.2", "v1.11.5"),
        ("GHSA-PY-CRYPTO-2026B", "minimum compliant cryptography wheel", "42.0.2", "42.0.5", "43.0.1"),
        ("GHSA-RUSTLS-TLS-2026C", "certified rustls crate version", "0.22.2", "0.23.5", "0.23.14"),
        ("GHSA-CONTAINERD-2026D", "hardened containerd daemon release", "1.7.13", "1.7.18", "1.7.23"),
        ("GHSA-POSTGRES-WAL-2026E", "mandated postgresql minor release", "16.1", "16.3", "16.6"),
    ]
    cve_disputed_entities = [
        ("CVE-2026-8841-REDIS-LUA", "sandbox escape exploitability verdict", "SANDBOX_ESCAPE_EXPLOITABLE", "INTENDED_TRUSTED_ADMIN_FEATURE"),
        ("CVE-2026-8842-SQLITE-CLI", "memory corruption boundary classification", "REMOTE_BOUNDARY_VULNERABILITY", "LOCAL_DEBUG_BUILD_ASSERT_ONLY"),
        ("CVE-2026-8843-NGINX-HTTP3", "quic 0-rtt replay vulnerability verdict", "PROTOCOL_AUTH_BYPASS_DEFECT", "RFC9001_EXPECTED_IDEMPOTENCY"),
        ("CVE-2026-8844-CURL-SOCKS", "proxy hostname truncation severity verdict", "HEAP_OVERFLOW_EXPLOITABLE", "UNREACHABLE_UNDER_RFC_LIMITS"),
        ("CVE-2026-8845-SYSTEMD-NS", "mount namespace race exploitability verdict", "UNPRIVILEGED_ROOT_ESCALATION", "REQUIRES_EXISTING_CAP_SYS_ADMIN"),
    ]

    sec_supersede_entities = [
        ("ACME-SEMI-FY26Q2", "verified fy26 q2 diluted eps", "$1.48", "$1.19"),
        ("NORTHSTAR-BIO-FY26Q1", "recognized fy26 q1 product revenue", "$412M", "$364M"),
        ("VANGUARD-CLOUD-FY26Q3", "certified fy26 q3 operating margin", "24.6%", "18.9%"),
        ("HELIOS-GRID-FY26Q2", "restated fy26 q2 free cash flow", "$290M", "$215M"),
        ("APEX-ROBOTICS-FY26Q1", "audited fy26 q1 backlog value", "$1.85B", "$1.42B"),
    ]
    sec_rumor_entities = [
        ("ORION-FINTECH-MERGER", "alleged acquisition buyout status", "DEFINITIVE_BUYOUT_AT_72_USD", "NO_MERGER_INDEPENDENT_OPERATIONS"),
        ("TITAN-ENERGY-CEO", "chief executive leadership status", "CEO_RESIGNED_EFFECTIVE_IMMEDIATELY", "CEO_RETAINED_FULL_BOARD_BACKING"),
        ("MERIDIAN-HEALTH-FDA", "phase3 clinical hold status", "MANDATORY_FDA_CLINICAL_HOLD", "TRIAL_ACTIVE_NO_FDA_HOLD"),
        ("COBALT-AERO-DOD", "defense contract cancellation status", "DOD_CONTRACT_TERMINATED", "CONTRACT_ACTIVE_ON_SCHEDULE"),
        ("SUMMIT-REIT-DIVIDEND", "quarterly dividend suspension status", "DIVIDEND_SUSPENDED_ZERO_PAYOUT", "DIVIDEND_DECLARED_0_85_USD"),
    ]
    sec_lineage_entities = [
        ("STRATOS-AI-FY26-GUIDE", "full-year fy26 revenue guidance midpoint", "$920M", "$985M", "$1.06B"),
        ("PACIFIC-LOGISTICS-CAPEX", "authorized fy26 capital expenditure budget", "$310M", "$355M", "$410M"),
        ("QUANTUM-FAB-WAFER-CAP", "committed q4 monthly wafer output target", "42,000_WAFERS", "48,000_WAFERS", "55,000_WAFERS"),
        ("ATLAS-PHARMA-RD-GUIDE", "approved fy26 r&d expense guidance", "$640M", "$705M", "$760M"),
        ("NOVA-PAYMENTS-TPV-GUIDE", "projected fy26 total payment volume", "$84B", "$91B", "$99B"),
    ]
    sec_disputed_entities = [
        ("SILVERLAKE-RETAIL-14A", "contested store-fleet impairment valuation", "$680M_IMPAIRMENT_REQUIRED", "$190M_CARRYING_VALUE_SOUND"),
        ("IRONCLAD-MINING-14A", "contested proven lithium reserve estimate", "1.1M_TONNES_OVERSTATED", "2.4M_TONNES_JORC_COMPLIANT"),
        ("ZENITH-MEDIA-14A", "contested related-party licensing margin", "BELOW_MARKET_4_PERCENT_ROYALTY", "ARMS_LENGTH_11_PERCENT_EFFECTIVE"),
        ("HARBOR-SHIPPING-14A", "contested vessel charter renewal rate", "$18,500_PER_DAY_LOCKED", "$27,200_PER_DAY_INDEXED"),
        ("CRESTVIEW-BANK-14A", "contested commercial real estate loss reserve", "$540M_RESERVE_DEFICIT", "$310M_RESERVE_FULLY_ADEQUATE"),
    ]

    sre_supersede_entities = [
        ("INC-US-EAST-KAFKA-401", "approved consumer-lag remediation flag", "--unsafe-reset-offsets-earliest", "--throttle-rebalance-max-inflight=2"),
        ("INC-EU-AUTH-REDIS-402", "verified failover mitigation parameter", "cluster-allow-reads-when-down=yes", "min-replicas-to-write=2"),
        ("INC-APAC-ETCD-403", "safe quorum recovery command", "etcdctl-force-new-cluster-live", "etcdutl-snapshot-restore-quorum"),
        ("INC-GLOBAL-BGP-404", "validated traffic drain policy", "withdraw-all-anycast-prefixes", "graceful-shutdown-community-65535:0"),
        ("INC-GPU-NCCL-405", "production hang remediation env var", "NCCL_P2P_DISABLE=1", "NCCL_IB_TIMEOUT=22"),
    ]
    sre_rumor_entities = [
        ("CLOUD-KMS-KEY-LEAK-501", "production root hsm compromise status", "ROOT_HSM_KEYS_EXFILTRATED", "ZERO_KEY_COMPROMISE_TEST_FIXTURE_ONLY"),
        ("OBJ-STORE-SILENT-CORRUPT-502", "s3-west checksum corruption status", "SILENT_BITROT_ON_PARITY_SHARDS", "DATA_INTEGRITY_INTACT_CLIENT_SDK_BUG"),
        ("DNS-CONTROL-HIJACK-503", "authoritative nameserverbgp hijack status", "EXTERNAL_BGP_HIJACK_ACTIVE", "INTERNAL_TTL_CACHE_CONFIG_ONLY"),
        ("BILLING-LEDGER-WIPE-504", "production invoice database loss status", "PRIMARY_AND_REPLICA_WIPED", "ZERO_DATA_LOSS_READ_REPLICA_LAG_ONLY"),
        ("IAM-TOKEN-FORGERY-505", "oidc jwt signature bypass status", "UNAUTHENTICATED_JWT_FORGERY", "REJECTED_BY_GATEWAY_DEV_SANDBOX_ONLY"),
    ]
    sre_lineage_entities = [
        ("INC-CDN-POSTMORTEM-601", "confirmed primary outage root cause", "UPSTREAM_FIBER_CUT_ASHBURN", "TLS_OCSP_STAPLING_TIMEOUT", "WAF_REGEX_CPU_STARVATION"),
        ("INC-DB-POSTMORTEM-602", "verified primary latency root cause", "NVME_FIRMWARE_GC_STALL", "CONNECTION_POOL_SYN_FLOOD", "AUTOVACUUM_XID_WRAPAROUND_LOCK"),
        ("INC-MESH-POSTMORTEM-603", "confirmed envoy crash root cause", "HTTP2_HPACK_OOM_SPIKE", "XDS_SNAPSHOT_RACE_CONDITION", "MTLS_CRL_PARSER_STACK_OVERFLOW"),
        ("INC-SCHED-POSTMORTEM-604", "confirmed kube-scheduler stall root cause", "API_SERVER_LIST_PAGINATION", "POD_TOPOLOGY_SPREAD_BACKTRACK", "WEBHOOK_ADMISSION_DEADLOCK"),
        ("INC-INFER-POSTMORTEM-605", "confirmed vllm p99 spike root cause", "CUDA_GRAPH_RECAPTURE_THRASH", "KV_CACHE_PREFIX_EVICTION_STORM", "TOKENIZER_IPC_SOCKET_CONTENTION"),
    ]
    sre_disputed_entities = [
        ("INC-SPLIT-TRANSIT-701", "contested transatlantic packet-loss fault point", "CARRIER_A_SUBSEA_AMPLIFIER_FAULT", "CARRIER_B_EDGE_PEERING_POLICER_DROP"),
        ("INC-SPLIT-STORAGE-702", "contested dual-controller san fencing verdict", "CONTROLLER_0_SCSI_RESERVATION_FAULT", "CONTROLLER_1_HEARTBEAT_NIC_DROP"),
        ("INC-SPLIT-PAYMENT-703", "contested settlement timeout responsibility", "ISSUER_SWITCH_ISO8583_DELAY", "ACQUIRER_GATEWAY_TLS_RENEGOTIATION"),
        ("INC-SPLIT-FABRIC-704", "contested infiniband crc flap hardware source", "SPINE_SWITCH_OPTICAL_TRANSCEIVER", "HOST_CX7_PCIE_RETIMER_LANE_DEGRADE"),
        ("INC-SPLIT-CLOCK-705", "contested ptp time-drift grandmaster fault", "GPS_ANTENNA_MULTIPATH_SPOOF", "BOUNDARY_CLOCK_FPGA_PLL_UNLOCK"),
    ]

    cases: List[Dict[str, Any]] = []
    all_docs: List[RawDocumentEnvelope] = []

    def _add_supersede_family(domain: str, items: List[Tuple[str, str, str, str]], official_ch: str, prelim_ch: str) -> None:
        for ent, attr, stale_v, gold_v in items:
            cid = f"{domain}_sup_{ent.lower().replace('-', '_')}"
            osv_note = ""
            if ent in osv_seeds:
                osv_note = f" Upstream context: {osv_seeds[ent]['osv_summary']}."
            docs: List[RawDocumentEnvelope] = [
                RawDocumentEnvelope(
                    doc_id=f"{cid}_d00_prelim",
                    text=(
                        f"Preliminary release regarding {ent}:{osv_note} Initial guidance sets the {attr} to {stale_v}. "
                        f"Downstream teams logged {stale_v} for immediate deployment tracking."
                    ),
                    published_ts=2,
                    ingested_ts=2,
                    source_channel=prelim_ch,
                    domain=domain,
                ),
            ]
            docs.extend(
                _make_echo_docs(
                    cid, domain, ent, attr, stale_v,
                    start_pub_day=3, start_ing_day=3,
                    channels=("internal_ops_chat", "support_ticket", "stackoverflow_mirror", "community_forum", "analyst_note"),
                    n_echoes=11,
                )
            )
            docs.append(
                RawDocumentEnvelope(
                    doc_id=f"{cid}_d12_official_supersede",
                    text=(
                        f"Official authoritative update for {ent}: supersedes and replaces the earlier {stale_v} guidance "
                        f"after validation showed it was incomplete or unsafe. The verified current {attr} is {gold_v}."
                    ),
                    published_ts=15,
                    ingested_ts=15,
                    source_channel=official_ch,
                    domain=domain,
                )
            )
            # Add 4 late low-tier stale echoes ingested AFTER day 15 ( referencing pre-patch day 6 notes ) + noise
            for k_i in range(4):
                docs.append(
                    RawDocumentEnvelope(
                        doc_id=f"{cid}_d{13+k_i}_late_archive",
                        text=(
                            f"Archived regional mirror from last week's checklist on {ent}: operators had recorded "
                            f"{stale_v} as the {attr} prior to the formal bulletin."
                        ),
                        published_ts=6 + k_i,
                        ingested_ts=16 + k_i,
                        source_channel="community_forum",
                        domain=domain,
                    )
                )
            q_txt = f"What is the verified current {attr} for {ent}?"
            cases.append({
                "case_id": cid,
                "domain": domain,
                "regime": "supersession_under_echo_flood",
                "subject": ent,
                "query_text": q_txt,
                "gold": gold_v,
                "stale_or_false_values": [stale_v],
                "doc_ids": [d.doc_id for d in docs],
            })
            all_docs.extend(docs)

    def _add_rumor_laundering_family(domain: str, items: List[Tuple[str, str, str, str]], official_ch: str) -> None:
        for ent, attr, false_rumor, gold_v in items:
            cid = f"{domain}_rumor_{ent.lower().replace('-', '_')}"
            docs: List[RawDocumentEnvelope] = [
                RawDocumentEnvelope(
                    doc_id=f"{cid}_d00_seed_rumor",
                    text=(
                        f"Unverified external post regarding {ent}: anonymous sources claim the {attr} is {false_rumor}. "
                        f"Sharing here before official verification."
                    ),
                    published_ts=3,
                    ingested_ts=3,
                    source_channel="rumor_feed",
                    domain=domain,
                ),
            ]
            docs.extend(
                _make_echo_docs(
                    cid, domain, ent, attr, false_rumor,
                    start_pub_day=4, start_ing_day=4,
                    channels=("social_repost", "community_forum", "unverified_blog", "internal_ops_chat"),
                    n_echoes=10,
                )
            )
            docs.append(
                RawDocumentEnvelope(
                    doc_id=f"{cid}_d11_official_debunk",
                    text=(
                        f"Signed authoritative audit bulletin for {ent}: explicitly cites and refutes the circulating "
                        f"unverified rumor claiming {false_rumor}. Forensic audit confirms the {attr} is {gold_v}."
                    ),
                    published_ts=14,
                    ingested_ts=14,
                    source_channel=official_ch,
                    domain=domain,
                )
            )
            # 5 late low-tier rumor-laundering posts published AFTER day 14 trying to re-assert `false_rumor`!
            for k_i in range(5):
                docs.append(
                    RawDocumentEnvelope(
                        doc_id=f"{cid}_d{12+k_i}_late_launder",
                        text=(
                            f"Follow-up blog commentary on {ent}: despite official denials, social threads still insist "
                            f"the {attr} has shifted to {false_rumor} based on the earlier forum post."
                        ),
                        published_ts=16 + k_i,
                        ingested_ts=16 + k_i,
                        source_channel="unverified_blog" if k_i % 2 == 0 else "social_repost",
                        domain=domain,
                    )
                )
            q_txt = f"What is the verified current {attr} for {ent}?"
            cases.append({
                "case_id": cid,
                "domain": domain,
                "regime": "rumor_laundering_and_audit_trap",
                "subject": ent,
                "query_text": q_txt,
                "gold": gold_v,
                "stale_or_false_values": [false_rumor],
                "doc_ids": [d.doc_id for d in docs],
            })
            all_docs.extend(docs)

    def _add_lineage_family(domain: str, items: List[Tuple[str, str, str, str, str]], official_ch: str) -> None:
        for ent, attr, v1, v2, v3_gold in items:
            cid = f"{domain}_lin_{ent.lower().replace('-', '_')}"
            docs: List[RawDocumentEnvelope] = [
                RawDocumentEnvelope(
                    doc_id=f"{cid}_d00_v1",
                    text=f"Initial baseline notice for {ent}: establishes the {attr} at {v1} for initial planning.",
                    published_ts=1,
                    ingested_ts=1,
                    source_channel="vendor_advisory",
                    domain=domain,
                ),
                RawDocumentEnvelope(
                    doc_id=f"{cid}_d01_v2",
                    text=(
                        f"Mid-cycle revision for {ent}: supersedes the initial {v1} baseline and updates the {attr} to {v2}."
                    ),
                    published_ts=6,
                    ingested_ts=6,
                    source_channel="vendor_advisory",
                    domain=domain,
                ),
            ]
            docs.extend(
                _make_echo_docs(
                    cid, domain, ent, attr, v2,
                    start_pub_day=7, start_ing_day=7,
                    channels=("analyst_note", "internal_ops_chat", "support_ticket", "stackoverflow_mirror"),
                    n_echoes=12,
                )
            )
            docs.append(
                RawDocumentEnvelope(
                    doc_id=f"{cid}_d14_v3_final",
                    text=(
                        f"Final authoritative revision for {ent}: supersedes the interim {v2} figure (and prior {v1} baseline). "
                        f"The current binding {attr} is {v3_gold}."
                    ),
                    published_ts=20,
                    ingested_ts=20,
                    source_channel=official_ch,
                    domain=domain,
                )
            )
            for k_i in range(2):
                docs.append(
                    RawDocumentEnvelope(
                        doc_id=f"{cid}_d{15+k_i}_v1_echo",
                        text=f"Legacy onboarding archive ({ent}): early Q1 slides listed {v1} as the {attr}.",
                        published_ts=2,
                        ingested_ts=21 + k_i,
                        source_channel="community_forum",
                        domain=domain,
                    )
                )
            q_txt = f"What is the verified current {attr} for {ent}?"
            cases.append({
                "case_id": cid,
                "domain": domain,
                "regime": "multihop_version_lineage_drift",
                "subject": ent,
                "query_text": q_txt,
                "gold": v3_gold,
                "stale_or_false_values": [v1, v2],
                "doc_ids": [d.doc_id for d in docs],
            })
            all_docs.extend(docs)

    def _add_contested_family(domain: str, items: List[Tuple[str, str, str, str]], peer_ch_a: str, peer_ch_b: str) -> None:
        for ent, attr, claim_a, claim_b in items:
            cid = f"{domain}_contest_{ent.lower().replace('-', '_')}"
            docs: List[RawDocumentEnvelope] = [
                RawDocumentEnvelope(
                    doc_id=f"{cid}_d00_party_a",
                    text=(
                        f"Formal technical filing (Party A) on {ent}: directly refutes competing assessments and asserts "
                        f"that the {attr} is {claim_a}, pending independent adjudication."
                    ),
                    published_ts=12,
                    ingested_ts=12,
                    source_channel=peer_ch_a,
                    domain=domain,
                ),
                RawDocumentEnvelope(
                    doc_id=f"{cid}_d01_party_b",
                    text=(
                        f"Formal technical counter-filing (Party B) on {ent}: directly refutes Party A's {claim_a} claim "
                        f"and asserts that the {attr} is {claim_b}, with no third-party resolution issued."
                    ),
                    published_ts=12,
                    ingested_ts=13,
                    source_channel=peer_ch_b,
                    domain=domain,
                ),
            ]
            docs.extend(
                _make_echo_docs(
                    cid, domain, ent, attr, claim_a,
                    start_pub_day=13, start_ing_day=14,
                    channels=("community_forum", "analyst_note", "social_repost"),
                    n_echoes=8,
                )
            )
            docs.extend(
                _make_echo_docs(
                    cid + "_b", domain, ent, attr, claim_b,
                    start_pub_day=13, start_ing_day=22,
                    channels=("internal_ops_chat", "support_ticket", "unverified_blog"),
                    n_echoes=7,
                )
            )
            # Normalize doc_ids on the 7 B-side echoes so case has exact 17 docs
            fixed_tail: List[RawDocumentEnvelope] = []
            for idx_t, d_tail in enumerate(docs[10:]):
                fixed_tail.append(
                    RawDocumentEnvelope(
                        doc_id=f"{cid}_echo_b_{idx_t:02d}",
                        text=d_tail.text,
                        published_ts=d_tail.published_ts,
                        ingested_ts=d_tail.ingested_ts,
                        source_channel=d_tail.source_channel,
                        domain=d_tail.domain,
                    )
                )
            docs = docs[:10] + fixed_tail
            q_txt = f"What is the verified current {attr} for {ent}?"
            cases.append({
                "case_id": cid,
                "domain": domain,
                "regime": "mutual_peer_conflict_abstain",
                "subject": ent,
                "query_text": q_txt,
                "gold": "ABSTAIN",
                "stale_or_false_values": [claim_a, claim_b],
                "doc_ids": [d.doc_id for d in docs],
            })
            all_docs.extend(docs)

    _add_supersede_family("cybersec_cve", cve_entities, "official_nvd_advisory", "vendor_advisory")
    _add_rumor_laundering_family("cybersec_cve", cve_rumor_entities, "official_nvd_advisory")
    _add_lineage_family("cybersec_cve", cve_lineage_entities, "official_nvd_advisory")
    _add_contested_family("cybersec_cve", cve_disputed_entities, "independent_security_lab", "vendor_advisory")

    _add_supersede_family("sec_financial", sec_supersede_entities, "sec_edgar_filing", "vendor_advisory")
    _add_rumor_laundering_family("sec_financial", sec_rumor_entities, "sec_edgar_filing")
    _add_lineage_family("sec_financial", sec_lineage_entities, "sec_edgar_filing")
    _add_contested_family("sec_financial", sec_disputed_entities, "sec_edgar_filing", "sec_edgar_filing")

    _add_supersede_family("cloud_sre", sre_supersede_entities, "signed_security_bulletin", "postmortem_final")
    _add_rumor_laundering_family("cloud_sre", sre_rumor_entities, "signed_security_bulletin")
    _add_lineage_family("cloud_sre", sre_lineage_entities, "signed_security_bulletin")
    _add_contested_family("cloud_sre", sre_disputed_entities, "postmortem_final", "vendor_advisory")

    digest_payload = json.dumps(
        [{"id": d.doc_id, "txt": d.text, "p": d.published_ts, "i": d.ingested_ts, "ch": d.source_channel} for d in all_docs],
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    corpus_sha256 = hashlib.sha256(digest_payload).hexdigest()
    return cases, all_docs, corpus_sha256
