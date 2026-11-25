"""Network & Orbital Fabric control plane engine (Wave-20, Demo D12 `/fabric`, Bundle #61).

Autonomous DPU/switch and multi-tranche LEO control plane, *emulated in software*. Seven arms are
evaluated over the SAME synthetic traffic, the SAME contact schedule and the SAME authored ground
truth:

  1. ``uncalibrated_fleet_rag``              raw padded cosine over per-tranche vectors, last-seen rule
  2. ``chained_ridge_zscore``                chained ridge adapters + z-score merge, last-seen rule
  3. ``unsigned_popularity_correlator``      emulation (ours): majority claim per class, signature lookup
  4. ``overwrite_last_arrival``              last arrival per class wins, signature lookup
  5. ``spine_bitemporal_ledger_raw``         SQLiteChallengerBackend ledger, raw vectors
  6. ``procrustes_csls_plus_spine_ledger``   semi-orthogonal chain + CSLS merge + ledger
  7. ``chelated_fabric_control``             bridge-health gate + sidecar + EK5 v3 lineage + 2-bit register

Honesty contract
----------------
Every payload and the UI banner carry ``hardware_asic_p4_silicon_in_loop_claimed: false`` and
``orbital_flight_hardware_claimed: false``. Nothing here ran on a P4/ASIC target or on a spacecraft:
the TLS flows are synthetic (CESNET-TLS / CIC-IDS *schema*), the TLE elements are synthetic in the
CelesTrak two-line *format*, and propagation is two-body Keplerian (not SGP4). The dry-check encoders
are deterministic text-derived stand-ins; ``--mode run-live-encoders`` swaps in real checkpoints.

Presentation modes
------------------
``non_nda`` (default) projects every response through public labels and the fail-closed 56-pattern
verifier. ``nda`` mode adds an ``internal`` block and needs ``X-Copilot-Nda-Token``.
"""

from __future__ import annotations

import argparse
import base64
import dataclasses
import functools
import hashlib
import hmac
import json
import math
import os
import random
import re
import sqlite3  # noqa: F401  (ledger backend is SQLite; imported for the documented dependency)
import struct
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

from chelated_pv7_spine.bridge_merge_harness import (  # noqa: E402
    DEFAULT_CSLS_K,
    CalibratedZScoreMerge,
    CSLSMerge,
    RawCosineMerge,
    RidgeDriftAdapter,
    UncenteredProcrustes,
    ZeroMuPolarBridge,
    compose_bridge_chain,
    compute_tra1_anchor_diagnostics,
    l2_normalize,
)
from chelated_pv7_spine.burn_ledger import BurnedCorpusError, assert_corpus_unburned  # noqa: E402
from chelated_pv7_spine.challenger_ledger import (  # noqa: E402
    ChallengerBitemporalLedger,
    LedgerEdgeEnvelope,
    LedgerRecordEnvelope,
    SQLiteChallengerBackend,
)
from chelated_pv7_spine.ek5_reason_canonicalizer import export_canonical_ek5_audit_trace  # noqa: E402
from ek5_v3_stratified import (  # noqa: E402
    EdgeAttestation,
    StratifiedEvent,
    StratifiedEvidenceStore,
)
from scripts.verify_non_nda_ip_clean import assert_non_nda_clean, scan_text_for_ip_leaks  # noqa: E402

HTML_PATH = Path(__file__).resolve().parent / "network_orbital.html"
API_PREFIX = "/api/v2/fabric"
PAGE_PATHS = ("/fabric", "/fabric/", "/fabric/index.html")
NDA_TOKEN_HEADER = "x-copilot-nda-token"
MODES = ("non_nda", "nda")
MAX_BODY_BYTES = 64 * 1024
TENANT = "tenant_fabric"
BUNDLE_ID = 61

HONESTY_FLAGS: Dict[str, bool] = {
    "hardware_asic_p4_silicon_in_loop_claimed": False,
    "orbital_flight_hardware_claimed": False,
}
HONESTY_BANNER = (
    "Software emulation only. No ASIC, P4 or DPU silicon was in the loop and no flight hardware was used. "
    "Traffic and orbital elements are synthetic; propagation is two-body Keplerian."
)

VERDICT_FORWARD = "FORWARD"
VERDICT_DROP = "DROP"
VERDICT_DEFER = "DEFER_INSPECT"
VERDICT_INVALID = "INVALID_FAIL_CLOSED"
VERDICTS = (VERDICT_FORWARD, VERDICT_DROP, VERDICT_DEFER)
REG_CODE = {VERDICT_DROP: 0b00, VERDICT_FORWARD: 0b01, VERDICT_DEFER: 0b10, VERDICT_INVALID: 0b11}
REG_DECODE = {v: k for k, v in REG_CODE.items()}

# ---------------------------------------------------------------------------
# Arms, public ids and labels (the internal ids never leave the server in non_nda mode)
# ---------------------------------------------------------------------------
ARMS: Tuple[str, ...] = (
    "uncalibrated_fleet_rag",
    "chained_ridge_zscore",
    "unsigned_popularity_correlator",
    "overwrite_last_arrival",
    "spine_bitemporal_ledger_raw",
    "procrustes_csls_plus_spine_ledger",
    "chelated_fabric_control",
)
ARM_PUBLIC_ID = {
    "uncalibrated_fleet_rag": "fleet_rag",
    "chained_ridge_zscore": "chained_ridge",
    "unsigned_popularity_correlator": "popularity_correlator",
    "overwrite_last_arrival": "overwrite_last",
    "spine_bitemporal_ledger_raw": "ledger_raw",
    "procrustes_csls_plus_spine_ledger": "bridge_plus_ledger",
    "chelated_fabric_control": "fabric_control",
}
ARM_PUBLIC_LABEL = {
    "fleet_rag": "Fleet RAG over raw per-tranche vectors",
    "chained_ridge": "Chained linear adapters with score normalisation",
    "popularity_correlator": "Unsigned popularity correlator (emulation, ours)",
    "overwrite_last": "Overwrite by last arrival",
    "ledger_raw": "Bitemporal ledger over raw vectors",
    "bridge_plus_ledger": "Orthogonal bridge chain plus bitemporal ledger",
    "fabric_control": "Chelated fabric control plane",
}
ARM_PUBLIC_NOTE = {
    "fleet_rag": "Each tranche vector is zero-padded to the ground width and compared with raw cosine; the last rule seen for the matched class is applied.",
    "chained_ridge": "Four pairwise ridge adapters are composed and scores are z-normalised; the last rule seen for the matched class is applied.",
    "popularity_correlator": "Authored emulation: the claim repeated by most records for a class wins, whatever their authority. Exact class-signature lookup, no vectors.",
    "overwrite_last": "The most recently delivered record per class replaces the previous one. Exact class-signature lookup, no vectors.",
    "ledger_raw": "Records live in a bitemporal ledger with authority and withdrawal; vectors are raw and unbridged. Echoes are separate records.",
    "bridge_plus_ledger": "Same ledger, but queries cross a semi-orthogonal bridge chain with hub-bias-corrected scores.",
    "fabric_control": "Bridge-health gate, compact per-flow sidecar, lineage engine with observable echo collapse, and an epoch-pinned two-bit register.",
}
ARM_INTERNAL = {
    "uncalibrated_fleet_rag": "zero-pad raw cosine + RawCosineMerge, last-writer rule",
    "chained_ridge_zscore": "compose_bridge_chain(RidgeDriftAdapter x4) + CalibratedZScoreMerge",
    "unsigned_popularity_correlator": "plurality claim per class, exact signature lookup (authored emulation)",
    "overwrite_last_arrival": "last-arrival-wins per class, exact signature lookup (authored emulation)",
    "spine_bitemporal_ledger_raw": "ChallengerBitemporalLedger(SQLiteChallengerBackend) + raw zero-pad cosine",
    "procrustes_csls_plus_spine_ledger": "UncenteredProcrustes chain + CSLSMerge + ChallengerBitemporalLedger(SQLite)",
    "chelated_fabric_control": "TRA-1 gate + ZeroMuPolarBridge chain + CSLS 8 B sidecar + EK5 v3 WFS + 2-bit register",
}

ENCODER_CHAIN: Tuple[Tuple[str, str, int], ...] = (
    ("enc_a", "sentence-transformers/all-MiniLM-L6-v2", 384),
    ("enc_b", "BAAI/bge-small-en-v1.5", 384),
    ("enc_c", "sentence-transformers/all-mpnet-base-v2", 768),
    ("enc_d", "BAAI/bge-m3", 1024),
    ("enc_e", "Qwen/Qwen3-Embedding-8B", 4096),
)
ENCODER_PUBLIC_LABEL = {"enc_a": "Encoder A (384-d)", "enc_b": "Encoder B (384-d)", "enc_c": "Encoder C (768-d)",
                        "enc_d": "Encoder D (1024-d)", "enc_e": "Ground index encoder (4096-d)"}
NATIVE_DIM = 4096
LATENT_DIM = 1024

# ---------------------------------------------------------------------------
# Flow classes, TLS-flow serializer (CESNET-TLS / CIC-IDS style schema)
# ---------------------------------------------------------------------------
ALPN_CODE = {"": 0, "h2": 1, "http/1.1": 2, "h3": 3, "mqtt": 4, "imap": 5, "dot": 6}


@dataclasses.dataclass(frozen=True)
class FlowClass:
    cid: str
    public: str
    bases: Tuple[str, ...]
    ciphers: Tuple[int, ...]
    alpn: str
    size_mu: int
    size_sd: int
    iat_mu_us: int
    bursty: bool
    cert_days: int
    n_pkts: int


_C = lambda base, k: tuple(0x1301 + ((base * 7 + i * 3) % 40) for i in range(k))  # noqa: E731
FLOW_CLASSES: Tuple[FlowClass, ...] = (
    FlowClass("cls_video_cdn", "Class 01 streaming video", ("videohost.example", "streamnet.example"), _C(1, 9), "h2", 1180, 160, 3500, False, 90, 48),
    FlowClass("cls_web_browse", "Class 02 interactive web", ("pagehost.example", "newsdaily.example"), _C(2, 12), "h2", 620, 260, 12000, True, 365, 36),
    FlowClass("cls_ota_telemetry", "Class 03 firmware and telemetry", ("otafeed.example", "vehiclehub.example"), _C(3, 7), "http/1.1", 410, 70, 90000, False, 730, 24),
    FlowClass("cls_bulk_backup", "Class 04 bulk backup", ("coldstore.example", "vaultsync.example"), _C(4, 8), "http/1.1", 1420, 60, 800, False, 180, 64),
    FlowClass("cls_voip", "Class 05 voice and conferencing", ("voicemesh.example", "meetroom.example"), _C(5, 10), "h3", 210, 40, 20000, False, 60, 40),
    FlowClass("cls_iot_mqtt", "Class 06 sensor messaging", ("sensorbus.example", "metergrid.example"), _C(6, 6), "mqtt", 140, 30, 250000, False, 1095, 20),
    FlowClass("cls_dns_over_tls", "Class 07 name resolution", ("resolver.example", "namehub.example"), _C(7, 6), "dot", 180, 35, 150000, False, 120, 12),
    FlowClass("cls_mail_submission", "Class 08 mail submission", ("mailrelay.example", "inboxco.example"), _C(8, 9), "imap", 560, 190, 40000, True, 400, 28),
    FlowClass("cls_game_stream", "Class 09 interactive streaming", ("playfield.example", "arcadecloud.example"), _C(9, 11), "h3", 880, 140, 6000, False, 75, 52),
    FlowClass("cls_c2_beacon", "Class 10 periodic beaconing", ("pagehost.example", "cdnfront.example"), _C(2, 12), "h2", 640, 40, 60000, False, 15, 16),
    FlowClass("cls_exfil_tunnel", "Class 11 sustained outbound transfer", ("coldstore.example", "tunnelgate.example"), _C(4, 8), "http/1.1", 1380, 90, 1500, True, 20, 60),
    FlowClass("cls_ephemeral_cert", "Class 12 short-lived certificate services", ("edgefn.example", "flashcert.example"), _C(11, 8), "h2", 520, 120, 9000, False, 7, 30),
)
CLASS_IDS: Tuple[str, ...] = tuple(c.cid for c in FLOW_CLASSES)
CLASS_BY_ID: Dict[str, FlowClass] = {c.cid: c for c in FLOW_CLASSES}
CLASS_PUBLIC: Dict[str, str] = {c.cid: c.public for c in FLOW_CLASSES}


def _h(*parts: Any, mod: int) -> int:
    data = "\x1f".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "big") % mod


@dataclasses.dataclass(frozen=True)
class TlsFlow:
    """One TLS flow in the CESNET-TLS22 / CIC-IDS style: handshake metadata plus packet sequences."""

    flow_id: str
    cls: str
    tranche: int
    sni: str
    ja3: str
    ja4: str
    alpn: str
    cipher_family_key: str
    pkt_sizes: Tuple[int, ...]
    iat_us: Tuple[int, ...]
    cert_valid_days: int


def _quantile(values: Sequence[int], q_ppm: int) -> int:
    ordered = sorted(values)
    if not ordered:
        return 0
    idx = min(len(ordered) - 1, (q_ppm * (len(ordered) - 1) + 500_000) // 1_000_000)
    return int(ordered[idx])


def serialize_tls_flow(flow: TlsFlow) -> Dict[str, int]:
    """Serialize a flow to the integer feature record the fabric consumes (no floats, deterministic)."""
    domain = ".".join(flow.sni.split(".")[-2:])
    sizes = list(flow.pkt_sizes)
    iats = list(flow.iat_us)
    n = max(len(iats), 1)
    mean = sum(iats) / n
    var = sum((x - mean) ** 2 for x in iats) / n
    sd = math.sqrt(var)
    # Goh-Barabasi burstiness B=(sd-mu)/(sd+mu) in [-1, 1] -> shifted ppm in [0, 2_000_000].
    burst = 0 if mean + sd == 0 else (sd - mean) / (sd + mean)
    ja4_mid = flow.ja4.split("_")[1] if "_" in flow.ja4 else flow.ja4
    return {
        "sni_class": _h("sni", domain, mod=64),
        "sni_len_bucket": min(len(flow.sni) // 8, 15),
        "ja3_family": _h("ja3f", flow.cipher_family_key, mod=4096),
        "ja4_family": int(ja4_mid[:3], 16) % 4096,
        "alpn": ALPN_CODE.get(flow.alpn, 0),
        "cert_valid_days": int(flow.cert_valid_days),
        "n_pkts": len(sizes),
        "pkt_q10": _quantile(sizes, 100_000),
        "pkt_q50": _quantile(sizes, 500_000),
        "pkt_q90": _quantile(sizes, 900_000),
        "iat_burst_ppm": int(round((burst + 1.0) * 1_000_000)),
    }


def flow_text(feat: Mapping[str, int]) -> str:
    """Canonical token string of a serialized flow; the retrieval encoders embed this text."""
    return (
        f"sni{feat['sni_class']} len{feat['sni_len_bucket']} ja3f{feat['ja3_family']} ja4f{feat['ja4_family']} "
        f"alpn{feat['alpn']} cert{feat['cert_valid_days'] // 30} npk{feat['n_pkts'] // 8} "
        f"pq10b{feat['pkt_q10'] // 64} pq50b{feat['pkt_q50'] // 64} pq90b{feat['pkt_q90'] // 64} "
        f"burst{feat['iat_burst_ppm'] // 150_000}"
    )


def class_signature(feat: Mapping[str, int]) -> str:
    """Exact lookup key used by the signature-lookup arms (SNI bucket, cipher family, ALPN)."""
    return f"{feat['sni_class']}:{feat['ja3_family']}:{feat['alpn']}"


def generate_flows(seed: int, per_class: int, *, tranche_cycle: int = 3) -> List[TlsFlow]:
    """Deterministic synthetic flows (CESNET-TLS22 / CIC-IDS schema; not captured traffic)."""
    rng = random.Random(seed)
    flows: List[TlsFlow] = []
    for ci, fc in enumerate(FLOW_CLASSES):
        for k in range(per_class):
            base = fc.bases[rng.randrange(len(fc.bases))]
            sni = f"{rng.choice(('a', 'edge', 'node', 'www', 'cdn'))}{rng.randrange(1, 90)}.{base}"
            sibling = rng.random() < 0.08
            ciphers = list(fc.ciphers)
            if sibling:
                ciphers[-1] = 0x1301 + ((ciphers[-1] + 5) % 60)
            exts = list(range(10, 22))
            rng.shuffle(exts)
            ja3 = hashlib.md5(
                f"771,{'-'.join(map(str, ciphers))},{'-'.join(map(str, exts))},29-23-24,0".encode("ascii")
            ).hexdigest()
            fam_key = ",".join(str(c) for c in sorted(ciphers))
            ja4 = (
                f"t13d{len(ciphers):02d}{len(exts):02d}{fc.alpn[:2] or '00'}_"
                f"{hashlib.sha256(fam_key.encode()).hexdigest()[:12]}_"
                f"{hashlib.sha256(','.join(map(str, sorted(exts))).encode()).hexdigest()[:12]}"
            )
            sizes = tuple(max(40, min(1500, int(rng.gauss(fc.size_mu, fc.size_sd)))) for _ in range(fc.n_pkts))
            if fc.bursty:
                iats = tuple(int(rng.expovariate(1.0 / fc.iat_mu_us) * (0.05 if rng.random() < 0.7 else 5.0)) for _ in range(fc.n_pkts))
            else:
                iats = tuple(max(1, int(rng.gauss(fc.iat_mu_us, fc.iat_mu_us * 0.2))) for _ in range(fc.n_pkts))
            cert = max(1, int(rng.gauss(fc.cert_days, max(1, fc.cert_days // 10))))
            flows.append(TlsFlow(f"f{seed}_{ci:02d}_{k:04d}", fc.cid, (ci + k) % tranche_cycle, sni, ja3, ja4,
                                 fc.alpn, fam_key, sizes, iats, cert))
    return flows


# ---------------------------------------------------------------------------
# CelesTrak-format 2-line elements and a Keplerian contact schedule
# ---------------------------------------------------------------------------
MU_KM3_S2 = 398600.4418
R_EARTH_KM = 6378.137
OMEGA_EARTH = 7.2921159e-5
STATIONS: Tuple[Tuple[str, float, float], ...] = (
    ("GS-NORTH", 64.8, -147.7), ("GS-EUROPE", 52.2, 5.2), ("GS-SOUTH-AFRICA", -25.9, 27.7), ("GS-AUSTRALIA", -31.8, 115.9),
)
OCEAN_BOXES: Tuple[Tuple[float, float, float, float], ...] = (  # lat_min, lat_max, lon_min, lon_max (coarse)
    (-50.0, 50.0, -175.0, -110.0), (-45.0, 45.0, -40.0, -10.0), (-45.0, 0.0, 55.0, 95.0),
)
TLE_EPOCH_YY = 26
TLE_EPOCH_DAY = 100.0
SCHEDULE_HORIZON_S = 6 * 3600
SCHEDULE_STEP_S = 30
OISL_MAX_RANGE_KM = 5500.0
ELEV_MASK_DEG = 10.0


def _tle_checksum(line: str) -> int:
    total = 0
    for ch in line[:68]:
        if ch.isdigit():
            total += int(ch)
        elif ch == "-":
            total += 1
    return total % 10


def build_tle(name: str, satnum: int, inc: float, raan: float, ecc: float, argp: float, ma: float,
              mean_motion: float, elset: int = 1) -> Tuple[str, str, str]:
    """Format a CelesTrak-style 2-line element set (synthetic values, real column layout)."""
    l1 = (f"1 {satnum:05d}U {'26001A':<8} {TLE_EPOCH_YY:02d}{TLE_EPOCH_DAY:012.8f} {' .00000000':>10} "
          f"{' 00000-0':>8} {' 00000-0':>8} 0 {elset:4d}")
    l2 = (f"2 {satnum:05d} {inc:8.4f} {raan:8.4f} {int(round(ecc * 1e7)):07d} {argp:8.4f} {ma:8.4f} "
          f"{mean_motion:11.8f}{1:5d}")
    l1 += str(_tle_checksum(l1))
    l2 += str(_tle_checksum(l2))
    return name, l1, l2


def parse_tle(name: str, l1: str, l2: str) -> Dict[str, Any]:
    """Parse and checksum-verify a 2-line element set."""
    for line, first in ((l1, "1"), (l2, "2")):
        if len(line) != 69 or line[0] != first:
            raise ValueError("malformed two-line element set")
        if int(line[68]) != _tle_checksum(line):
            raise ValueError("two-line element checksum mismatch")
    if int(l1[2:7]) != int(l2[2:7]):
        raise ValueError("catalog number mismatch between lines")
    return {
        "name": name, "satnum": int(l1[2:7]), "epoch_yy": int(l1[18:20]), "epoch_day": float(l1[20:32]),
        "inc": math.radians(float(l2[8:16])), "raan": math.radians(float(l2[17:25])),
        "ecc": int(l2[26:33]) / 1e7, "argp": math.radians(float(l2[34:42])), "ma": math.radians(float(l2[43:51])),
        "mean_motion_rev_day": float(l2[52:63]),
    }


def _kepler_e(m: float, ecc: float) -> float:
    e_anom = m
    for _ in range(12):
        e_anom = e_anom - (e_anom - ecc * math.sin(e_anom) - m) / (1.0 - ecc * math.cos(e_anom))
    return e_anom


def _gmst0_rad() -> float:
    """Greenwich sidereal angle at the element epoch (IAU-82 polynomial, year 20YY)."""
    year = 2000 + TLE_EPOCH_YY
    jd = 1721028.5 + 365 * (year - 1) + (year - 1) // 4 - (year - 1) // 100 + (year - 1) // 400 + (TLE_EPOCH_DAY - 1.0)
    t = (jd - 2451545.0) / 36525.0
    gmst_s = 67310.54841 + (876600 * 3600 + 8640184.812866) * t + 0.093104 * t * t - 6.2e-6 * t ** 3
    return math.radians((gmst_s % 86400.0) / 240.0) % (2 * math.pi)


def propagate_ecef(elems: Mapping[str, Any], times_s: np.ndarray) -> np.ndarray:
    """Two-body Keplerian propagation to ECEF km (documented approximation; not SGP4)."""
    n = float(elems["mean_motion_rev_day"]) * 2 * math.pi / 86400.0
    a = (MU_KM3_S2 / (n * n)) ** (1.0 / 3.0)
    ecc = float(elems["ecc"])
    out = np.empty((len(times_s), 3))
    ci, si = math.cos(elems["inc"]), math.sin(elems["inc"])
    co, so = math.cos(elems["raan"]), math.sin(elems["raan"])
    cw, sw = math.cos(elems["argp"]), math.sin(elems["argp"])
    g0 = _gmst0_rad()
    for i, t in enumerate(times_s):
        m = (elems["ma"] + n * float(t)) % (2 * math.pi)
        e_anom = _kepler_e(m, ecc)
        x_p = a * (math.cos(e_anom) - ecc)
        y_p = a * math.sqrt(1 - ecc * ecc) * math.sin(e_anom)
        x1 = cw * x_p - sw * y_p
        y1 = sw * x_p + cw * y_p
        xe = co * x1 - so * ci * y1
        ye = so * x1 + co * ci * y1
        ze = si * y1
        th = g0 + OMEGA_EARTH * float(t)
        out[i] = (math.cos(th) * xe + math.sin(th) * ye, -math.sin(th) * xe + math.cos(th) * ye, ze)
    return out


def _station_ecef(lat: float, lon: float) -> np.ndarray:
    la, lo = math.radians(lat), math.radians(lon)
    return R_EARTH_KM * np.array([math.cos(la) * math.cos(lo), math.cos(la) * math.sin(lo), math.sin(la)])


def _windows(mask: np.ndarray, step: int) -> List[Tuple[int, int]]:
    wins: List[Tuple[int, int]] = []
    start: Optional[int] = None
    for i, v in enumerate(mask):
        if v and start is None:
            start = i
        if not v and start is not None:
            wins.append((start * step, i * step))
            start = None
    if start is not None:
        wins.append((start * step, len(mask) * step))
    return wins


@functools.lru_cache(maxsize=1)
def build_contact_schedule() -> Dict[str, Any]:
    """Tranche 0..2 -> ground / inter-tranche (OISL) contacts and coarse-ocean blackout windows."""
    layout = (  # (tranche, name, satnum, raan, ma)
        (0, "FABRIC-T0-A", 90001, 10.0, 0.0), (0, "FABRIC-T0-B", 90002, 10.0, 180.0),
        (1, "FABRIC-T1-A", 90011, 130.0, 60.0), (1, "FABRIC-T1-B", 90012, 130.0, 240.0),
        (2, "FABRIC-T2-A", 90021, 250.0, 120.0), (2, "FABRIC-T2-B", 90022, 250.0, 300.0),
    )
    tles = [build_tle(nm, sn, 53.0, raan, 0.0001, 90.0, ma, 15.06) for (_t, nm, sn, raan, ma) in layout]
    elems = [parse_tle(*t) for t in tles]
    tranche_of = [t for (t, *_rest) in layout]
    times = np.arange(0, SCHEDULE_HORIZON_S, SCHEDULE_STEP_S, dtype=float)
    pos = [propagate_ecef(e, times) for e in elems]
    st = [_station_ecef(la, lo) for (_n, la, lo) in STATIONS]
    ground_vis = np.zeros((3, len(times)), dtype=bool)
    sub_ocean = np.zeros((3, len(times)), dtype=bool)
    for si, p in enumerate(pos):
        tr = tranche_of[si]
        for s in st:
            rel = p - s
            rng = np.linalg.norm(rel, axis=1)
            sin_el = (rel @ (s / np.linalg.norm(s))) / rng
            ground_vis[tr] |= np.degrees(np.arcsin(np.clip(sin_el, -1, 1))) >= ELEV_MASK_DEG
        lat = np.degrees(np.arcsin(p[:, 2] / np.linalg.norm(p, axis=1)))
        lon = np.degrees(np.arctan2(p[:, 1], p[:, 0]))
        ocean = np.zeros(len(times), dtype=bool)
        for (la0, la1, lo0, lo1) in OCEAN_BOXES:
            ocean |= (lat >= la0) & (lat <= la1) & (lon >= lo0) & (lon <= lo1)
        sub_ocean[tr] |= ocean

    def _los_ok(p: np.ndarray, q: np.ndarray) -> np.ndarray:
        d = q - p
        t = np.clip(-np.sum(p * d, axis=1) / np.maximum(np.sum(d * d, axis=1), 1e-9), 0.0, 1.0)
        closest = np.linalg.norm(p + d * t[:, None], axis=1)
        return (closest > R_EARTH_KM + 80.0) & (np.linalg.norm(d, axis=1) <= OISL_MAX_RANGE_KM)

    oisl: Dict[Tuple[int, int], np.ndarray] = {}
    for a_t, b_t in ((0, 1), (1, 2)):
        m = np.zeros(len(times), dtype=bool)
        for ia, ta in enumerate(tranche_of):
            for ib, tb in enumerate(tranche_of):
                if ta == a_t and tb == b_t:
                    m |= _los_ok(pos[ia], pos[ib])
        oisl[(a_t, b_t)] = m
    blackout = {t: (~ground_vis[t]) & sub_ocean[t] for t in range(3)}
    sched = {
        "tle_lines": [list(t) for t in tles],
        "tle_checksums_verified": True,
        "propagator": "two-body Keplerian (not SGP4); synthetic elements in CelesTrak 2-line format",
        "step_s": SCHEDULE_STEP_S,
        "horizon_s": SCHEDULE_HORIZON_S,
        "stations": [n for (n, _a, _b) in STATIONS],
        "ground_windows": {str(t): _windows(ground_vis[t], SCHEDULE_STEP_S) for t in range(3)},
        "oisl_windows": {f"{a}-{b}": _windows(m, SCHEDULE_STEP_S) for (a, b), m in oisl.items()},
        "oceanic_blackout_windows": {str(t): _windows(blackout[t], SCHEDULE_STEP_S) for t in range(3)},
    }
    sched["summary"] = {
        "ground_window_count": {k: len(v) for k, v in sched["ground_windows"].items()},
        "oisl_window_count": {k: len(v) for k, v in sched["oisl_windows"].items()},
        "oceanic_blackout_count": {k: len(v) for k, v in sched["oceanic_blackout_windows"].items()},
        "oceanic_blackout_seconds": {k: int(sum(b - a for a, b in v)) for k, v in sched["oceanic_blackout_windows"].items()},
    }
    sched["schedule_sha256"] = hashlib.sha256(canonical_json_for_hash(sched)).hexdigest()
    return sched


def canonical_json_for_hash(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def dtn_delay_s(tranche: int, t_s: int) -> int:
    """Seconds until a message handed to the ground at ``t_s`` can reach ``tranche`` (next ground window)."""
    for (a, b) in build_contact_schedule()["ground_windows"][str(tranche)]:
        if b > t_s:
            return max(0, a - t_s)
    return SCHEDULE_HORIZON_S - t_s


# ---------------------------------------------------------------------------
# Deterministic text-derived encoders (dry-check) and optional live encoders
# ---------------------------------------------------------------------------
_NOISE_SIGMA = {0: 0.30, 1: 0.26, 2: 0.22, 3: 0.14, 4: 0.04}
_LATENT_CACHE: Dict[str, np.ndarray] = {}
_PROJ_CACHE: Dict[int, np.ndarray] = {}


def _latent(text: str) -> np.ndarray:
    hit = _LATENT_CACHE.get(text)
    if hit is not None:
        return hit
    vec = np.zeros(LATENT_DIM)
    low = text.lower()
    tokens = low.split() + [low[i:i + 3] for i in range(len(low) - 2)]
    for tok in tokens:
        d = hashlib.blake2b(tok.encode("utf-8"), digest_size=8).digest()
        v = int.from_bytes(d, "big")
        vec[v % LATENT_DIM] += 1.0 if (v >> 40) & 1 else -1.0
    n = np.linalg.norm(vec)
    vec = vec / n if n > 0 else vec
    _LATENT_CACHE[text] = vec
    return vec


def _proj(enc_index: int) -> np.ndarray:
    hit = _PROJ_CACHE.get(enc_index)
    if hit is not None:
        return hit
    d = ENCODER_CHAIN[enc_index][2]
    g = np.random.default_rng(7000 + enc_index).standard_normal((max(LATENT_DIM, d), min(LATENT_DIM, d)))
    q, _ = np.linalg.qr(g)
    mat = q[:, : min(LATENT_DIM, d)]
    mat = mat if d <= LATENT_DIM else mat.T
    _PROJ_CACHE[enc_index] = mat if mat.shape == (LATENT_DIM, d) else mat.reshape(LATENT_DIM, d)
    return _PROJ_CACHE[enc_index]


def dry_encode(texts: Sequence[str], enc_index: int) -> np.ndarray:
    """Deterministic stand-in encoder #enc_index (shared latent + per-encoder rotation + per-text noise)."""
    d = ENCODER_CHAIN[enc_index][2]
    lat = np.stack([_latent(t) for t in texts])
    out = lat @ _proj(enc_index)
    sigma = _NOISE_SIGMA[enc_index]
    for i, t in enumerate(texts):
        seed = int.from_bytes(hashlib.blake2b(f"{enc_index}|{t}".encode(), digest_size=8).digest(), "big")
        out[i] += np.random.default_rng(seed).standard_normal(d) * (sigma / math.sqrt(d))
    return l2_normalize(out)


def live_encode(texts: Sequence[str], enc_index: int) -> np.ndarray:  # pragma: no cover - needs weights
    """Real checkpoint embeddings (offline/CPU); raises if the checkpoint is not cached locally."""
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    model = _live_model(ENCODER_CHAIN[enc_index][1])
    return l2_normalize(np.asarray(model.encode(list(texts), normalize_embeddings=True, show_progress_bar=False), dtype=np.float64))


@functools.lru_cache(maxsize=8)
def _live_model(name: str) -> Any:  # pragma: no cover - needs weights
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(name, device="cpu")


def live_encoder_available(enc_index: int) -> Tuple[bool, str]:
    """Whether a real checkpoint can be loaded offline on CPU (never downloads, never touches a GPU)."""
    try:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        from huggingface_hub import try_to_load_from_cache  # noqa: F401
        import sentence_transformers  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - optional dependency probe
        return False, f"sentence-transformers unavailable: {type(exc).__name__}"
    name = ENCODER_CHAIN[enc_index][1]
    if enc_index == 4:
        return False, "4096-d ground encoder (Qwen3-8B) is not loaded on CPU in this harness; dry stand-in retained"
    try:
        _live_model(name)
    except Exception as exc:  # noqa: BLE001 - any load failure means "not available offline"
        return False, f"checkpoint not loadable offline: {type(exc).__name__}"
    return True, "loaded"


# ---------------------------------------------------------------------------
# Control-plane event model, authored ground truth, scenarios A (6 steps) and B (5 steps)
# ---------------------------------------------------------------------------
KIND_RULE, KIND_ECHO, KIND_RULING, KIND_DERIVED, KIND_AUDIT, KIND_RETRACT = (
    "RULE", "ECHO", "RULING", "DERIVED", "AUDIT", "RETRACT")
CLAIM_KINDS = (KIND_RULE, KIND_ECHO, KIND_RULING)

BASE_VERDICT: Dict[str, str] = {c: VERDICT_FORWARD for c in CLASS_IDS}
BASE_VERDICT.update({"cls_c2_beacon": VERDICT_DROP, "cls_exfil_tunnel": VERDICT_DROP,
                     "cls_ephemeral_cert": VERDICT_DEFER})

CLS_X, CLS_Y, CLS_W, CLS_Z = "cls_video_cdn", "cls_web_browse", "cls_voip", "cls_ota_telemetry"
CLS_P, CLS_R = "cls_bulk_backup", "cls_iot_mqtt"
DERIVED_CLASSES = (CLS_X, CLS_Y, CLS_Z)  # X, Y get invalidated parents; Z is the untouched control


@dataclasses.dataclass(frozen=True)
class FEvent:
    kind: str
    eid: str
    cls: str
    claim: str
    tier: int
    tranche: int
    target: str   # RETRACT: eid retracted; RULING: eid refuted ("" = none); DERIVED/AUDIT: parent eid; else ""
    text: str
    extra_edges: Tuple[Tuple[str, str], ...] = ()  # (edge_type, dst_eid) beyond the structural ones


@dataclasses.dataclass
class Step:
    sid: str
    title: str
    events: List[FEvent]
    truth: Dict[str, str]
    invalid_parents: FrozenSet[str]
    epoch: int = 0


def _rep_text(cls: str, rep: Mapping[str, str]) -> str:
    return rep[cls]


def representative_texts() -> Dict[str, str]:
    """One representative serialized flow text per class (from an independent generator seed)."""
    flows = generate_flows(31337, 1)
    return {f.cls: flow_text(serialize_tls_flow(f)) for f in flows}


def _baseline_events(rep: Mapping[str, str]) -> List[FEvent]:
    out: List[FEvent] = []
    for ci, c in enumerate(CLASS_IDS):
        out.append(FEvent(KIND_RULE, f"r_{c}", c, BASE_VERDICT[c], 1, 0, "", rep[c]))
    for ci, c in enumerate(CLASS_IDS):
        for hop in (1, 2):
            out.append(FEvent(KIND_ECHO, f"e{hop}_{c}", c, BASE_VERDICT[c], 1, hop, f"r_{c}", f"{rep[c]} hop{hop}"))
    for c in DERIVED_CLASSES:
        out.append(FEvent(KIND_DERIVED, f"d_{c}", c, "", 1, 0, f"r_{c}", f"summary of {c} policy"))
    out.append(FEvent(KIND_AUDIT, f"a_{CLS_Y}", CLS_Y, "", 1, 0, f"r_{CLS_Y}", f"audit memo cites {CLS_Y} policy"))
    return out


def build_scenarios() -> Dict[str, List[Step]]:
    rep = representative_texts()
    base = _baseline_events(rep)
    truth0 = dict(BASE_VERDICT)
    inv0: FrozenSet[str] = frozenset()

    # ---- Scenario A: conflicting authority, equal-tier dispute, late echo after retraction, epoch roll
    a_steps: List[Step] = []
    a_steps.append(Step("A1", "Ingest baseline policy, replica echoes, derived summaries, audit memo", base, dict(truth0), inv0))
    t2 = dict(truth0)
    t2[CLS_X] = VERDICT_DROP
    t2[CLS_Y] = VERDICT_DEFER
    inv2 = frozenset({f"r_{CLS_X}", f"r_{CLS_Y}"})
    a_steps.append(Step("A2", "Authoritative ruling overrides class X; class Y rule is retracted with no replacement", [
        FEvent(KIND_RULING, "g_x_drop", CLS_X, VERDICT_DROP, 3, 0, f"r_{CLS_X}", f"ruling drop {rep[CLS_X]}"),
        FEvent(KIND_RETRACT, "g_y_retract", CLS_Y, "", 3, 1, f"r_{CLS_Y}", f"retract {rep[CLS_Y]}"),
    ], t2, inv2))
    t3 = dict(t2)
    t3[CLS_W] = VERDICT_DEFER
    a_steps.append(Step("A3", "Two equal-authority tranches disagree on class W", [
        FEvent(KIND_RULE, "w_fwd", CLS_W, VERDICT_FORWARD, 2, 1, "", f"tranche one view {rep[CLS_W]}"),
        FEvent(KIND_RULE, "w_drop", CLS_W, VERDICT_DROP, 2, 2, "", f"tranche two view {rep[CLS_W]}"),
    ], t3, inv2))
    t4 = dict(t3)
    t4[CLS_W] = VERDICT_DROP
    a_steps.append(Step("A4", "Authoritative ruling resolves the class W dispute", [
        FEvent(KIND_RULING, "g_w_drop", CLS_W, VERDICT_DROP, 3, 0, "w_fwd", f"ruling drop {rep[CLS_W]}",
               extra_edges=(("REFUTE", "w_drop"),)),
    ], t4, inv2))
    a_steps.append(Step("A5", "A delayed replica of the retracted class Y rule arrives late", [
        FEvent(KIND_ECHO, "e3_late_y", CLS_Y, VERDICT_FORWARD, 1, 2, f"r_{CLS_Y}", f"{rep[CLS_Y]} hop3"),
    ], dict(t4), inv2))
    a_steps.append(Step("A6", "Policy epoch rolls and the register is recompiled under epoch pinning", [], dict(t4), inv2, epoch=1))

    # ---- Scenario B: retraction before target, replica-lagged retraction, reserved code, cycle-only support
    b_steps: List[Step] = []
    b_steps.append(Step("B1", "Ingest baseline policy, replica echoes, derived summaries, audit memo", list(base), dict(truth0), inv0))
    b_steps.append(Step("B2", "A retraction arrives BEFORE the rule it retracts", [
        FEvent(KIND_RETRACT, "g_p2_retract", CLS_P, "", 3, 0, "r_p2", f"retract tier two {rep[CLS_P]}"),
        FEvent(KIND_RULE, "r_p2", CLS_P, VERDICT_DROP, 2, 1, "", f"tier two view {rep[CLS_P]}"),
    ], dict(truth0), frozenset()))
    tb3 = dict(truth0)
    tb3[CLS_P] = VERDICT_DEFER
    b_steps.append(Step("B3", "Uplink retracts the class P baseline rule; replicas still carry it", [
        FEvent(KIND_RETRACT, "g_p_retract", CLS_P, "", 3, 0, f"r_{CLS_P}", f"retract {rep[CLS_P]}"),
    ], tb3, frozenset({f"r_{CLS_P}"})))
    b_steps.append(Step("B4", "A single-event upset sets the reserved register code for class P", [], dict(tb3), frozenset({f"r_{CLS_P}"})))
    b_steps.append(Step("B5", "A cycle-only support pair claims DROP for class R at maximum authority", [
        FEvent(KIND_RULE, "s1_r", CLS_R, VERDICT_DROP, 3, 1, "", f"support one {rep[CLS_R]}", extra_edges=(("DERIVES_FROM", "s2_r"),)),
        FEvent(KIND_RULE, "s2_r", CLS_R, VERDICT_DROP, 3, 2, "", f"support two {rep[CLS_R]}", extra_edges=(("DERIVES_FROM", "s1_r"),)),
    ], dict(tb3), frozenset({f"r_{CLS_P}"})))
    return {"A": a_steps, "B": b_steps}


# ---------------------------------------------------------------------------
# 2-bit register compiler: epoch header | 2 bits per class | epoch trailer (torn reads fenced)
# ---------------------------------------------------------------------------
REG_HDR = 4
REG_TRL = 4


def register_payload_len(n_classes: int) -> int:
    return (n_classes * 2 + 7) // 8


def compile_register(epoch: int, verdicts: Mapping[str, str]) -> bytes:
    """Epoch-pinned register image. ``01=FORWARD``, ``00=DROP``, ``10=DEFER_INSPECT``, ``11`` reserved."""
    payload = bytearray(register_payload_len(len(CLASS_IDS)))
    for i, c in enumerate(CLASS_IDS):
        code = REG_CODE[verdicts.get(c, VERDICT_DEFER)]
        payload[i // 4] |= code << ((i % 4) * 2)
    return struct.pack(">I", epoch) + bytes(payload) + struct.pack(">I", epoch)


def decode_register_fenced(image: bytes, pinned_epoch: int, last_good: Mapping[str, str]) -> Tuple[Dict[str, str], Dict[str, int]]:
    """Fenced reader: header==trailer==pinned else last good snapshot; reserved 11 fails closed (never FORWARD)."""
    stats = {"torn_reads": 0, "reserved_codes": 0}
    hdr = struct.unpack(">I", image[:REG_HDR])[0]
    trl = struct.unpack(">I", image[-REG_TRL:])[0]
    if not (hdr == trl == pinned_epoch):
        stats["torn_reads"] = 1
        return dict(last_good), stats
    out: Dict[str, str] = {}
    for i, c in enumerate(CLASS_IDS):
        code = (image[REG_HDR + i // 4] >> ((i % 4) * 2)) & 0b11
        if code == 0b11:
            stats["reserved_codes"] += 1
            out[c] = VERDICT_INVALID
        else:
            out[c] = REG_DECODE[code]
    return out, stats


def decode_register_naive(image: bytes) -> Dict[str, str]:
    """Unfenced reader: ignores epochs; treats the low bit as the forward bit (the textbook bug)."""
    out: Dict[str, str] = {}
    for i, c in enumerate(CLASS_IDS):
        code = (image[REG_HDR + i // 4] >> ((i % 4) * 2)) & 0b11
        out[c] = VERDICT_FORWARD if code & 1 else VERDICT_DROP
    return out


def decode_register_reference(image: bytes, pinned_epoch: int, last_good: Mapping[str, str]) -> Dict[str, str]:
    """Second, independently written decoder used only by the oracle (bit arithmetic via integer big-number)."""
    n = len(CLASS_IDS)
    body = image[REG_HDR:-REG_TRL]
    hdr = int.from_bytes(image[:REG_HDR], "big")
    trl = int.from_bytes(image[-REG_TRL:], "big")
    if hdr != pinned_epoch or trl != pinned_epoch:
        return dict(last_good)
    big = int.from_bytes(body, "little")
    out: Dict[str, str] = {}
    names = {0: VERDICT_DROP, 1: VERDICT_FORWARD, 2: VERDICT_DEFER, 3: VERDICT_INVALID}
    for i in range(n):
        out[CLASS_IDS[i]] = names[(big >> (2 * i)) & 3]
    return out


def _write_orders(n_total: int, n_payload_end: int, rng: random.Random) -> List[Tuple[str, List[int]]]:
    payload = list(range(REG_HDR, n_payload_end))
    hdr = list(range(0, REG_HDR))
    trl = list(range(n_payload_end, n_total))
    orders = [("payload_header_trailer", payload + hdr + trl), ("trailer_header_payload", trl + hdr + payload),
              ("header_payload_trailer", hdr + payload + trl)]
    for k in range(6):
        perm = list(range(n_total))
        rng.shuffle(perm)
        orders.append((f"shuffled_{k}", perm))
    return orders


def torn_read_experiment(old: Mapping[str, str], new: Mapping[str, str], old_epoch: int, new_epoch: int,
                         seed: int = 11) -> Dict[str, Any]:
    """In-place register roll under several byte-write orders; read after EVERY byte; count mixed snapshots.

    A snapshot is *mixed* when it equals neither the pure old nor the pure new decode of the same reader.
    """
    new_img = compile_register(new_epoch, new)
    pure_old_naive = decode_register_naive(compile_register(old_epoch, old))
    pure_new_naive = decode_register_naive(new_img)
    rng = random.Random(seed)
    naive_mixed = positions = 0
    fenced_contract = fenced_noncontract = 0
    by_order: Dict[str, Dict[str, int]] = {}
    trl_positions = set(range(len(new_img) - REG_TRL, len(new_img)))
    for name, order in _write_orders(len(new_img), len(new_img) - REG_TRL, rng):
        contract = max(i for i, p in enumerate(order) if p not in trl_positions) < min(i for i, p in enumerate(order) if p in trl_positions)
        img = bytearray(compile_register(old_epoch, old))
        last_good = dict(old)
        o_naive = o_fenced = 0
        for pos in order:
            img[pos] = new_img[pos]
            positions += 1
            nv = decode_register_naive(bytes(img))
            if nv != pure_old_naive and nv != pure_new_naive:
                naive_mixed += 1
                o_naive += 1
            fv, _stats = decode_register_fenced(bytes(img), new_epoch, last_good)
            if fv != dict(old) and fv != dict(new):
                o_fenced += 1
                if contract:
                    fenced_contract += 1
                else:
                    fenced_noncontract += 1
            elif fv == dict(new):
                last_good = dict(new)
        by_order[name] = {"trailer_written_last": int(contract), "naive_mixed": o_naive, "fenced_mixed": o_fenced}
    return {"write_positions": positions, "write_orders": len(by_order), "naive_mixed_reads": naive_mixed,
            "fenced_mixed_reads": fenced_contract, "fenced_mixed_reads_noncontract_orders": fenced_noncontract,
            "by_order": by_order}


def seu_flip_to_reserved(image: bytes, cls: str) -> bytes:
    """Single-event upset: force the two register bits of ``cls`` to the reserved code 0b11."""
    i = CLASS_IDS.index(cls)
    buf = bytearray(image)
    buf[REG_HDR + i // 4] |= 0b11 << ((i % 4) * 2)
    return bytes(buf)


# ---------------------------------------------------------------------------
# 8 B/flow sidecar (doc_den, hubness r_d) and the TRA-1 bridge-health gate
# ---------------------------------------------------------------------------
def build_sidecar(doc_den: np.ndarray, r_d: np.ndarray) -> bytes:
    return b"".join(struct.pack("<ff", float(a), float(b)) for a, b in zip(doc_den, r_d))


def read_sidecar(blob: bytes) -> Tuple[np.ndarray, np.ndarray]:
    n = len(blob) // 8
    arr = np.array([struct.unpack_from("<ff", blob, 8 * i) for i in range(n)], dtype=np.float64)
    return arr[:, 0], arr[:, 1]


def tra1_gate(src_anchors: np.ndarray, tgt_anchors: np.ndarray, bridge: Any) -> Dict[str, Any]:
    """Fail-closed bridge-health gate; any status other than PASS refuses the bridge for control decisions."""
    diag = compute_tra1_anchor_diagnostics(src_anchors, tgt_anchors, bridge)
    status = str(diag.tra1_health_status)
    return {"status": status, "admit": status == "PASS"}


# ---------------------------------------------------------------------------
# Echo-cluster inference (observable, uses only class, claim, tier and text)
# ---------------------------------------------------------------------------
_HOP_RE = re.compile(r"\s+hop\d+\b")


def echo_fingerprint(text: str) -> str:
    return _HOP_RE.sub("", text).strip()


def infer_echo_clusters(events: Sequence[FEvent], arrival: Mapping[str, int]) -> Dict[str, str]:
    """Map every claim event that repeats an earlier identical-claim record to that record (the cluster root)."""
    roots: Dict[Tuple[str, str, int, str], str] = {}
    out: Dict[str, str] = {}
    for ev in sorted((e for e in events if e.kind in CLAIM_KINDS), key=lambda e: arrival[e.eid]):
        key = (ev.cls, ev.claim, ev.tier, echo_fingerprint(ev.text))
        root = roots.get(key)
        if root is None:
            roots[key] = ev.eid
        else:
            out[ev.eid] = root
    return out


# ---------------------------------------------------------------------------
# Retrieval path: per-hop 4096-d cross-encoder retrieval for the fleet arms
# ---------------------------------------------------------------------------
INDEX_SEED = 7
DOCS_PER_CLASS = 12
ANCHOR_SEED = 99
ANCHORS_PER_CLASS = 300  # oversampled; unique texts (>1024) are kept so every cross-covariance can be full rank
DEV_REF_COUNT = 288
KAPPA_TOL = 1e-5
SIDECAR_BYTES_PER_FLOW = 8
FAMILIES = ("ridge", "procrustes", "zeromu")


def unique_flows(seed: int, per_class: int, pool_per_class: int, exclude: Sequence[str] = ()) -> List[TlsFlow]:
    """First ``per_class`` flows per class whose serialized text is unique and not in ``exclude``."""
    seen = set(exclude)
    out: List[TlsFlow] = []
    counts: Dict[str, int] = {}
    for f in generate_flows(seed, pool_per_class):
        txt = flow_text(serialize_tls_flow(f))
        if txt in seen or counts.get(f.cls, 0) >= per_class:
            continue
        seen.add(txt)
        counts[f.cls] = counts.get(f.cls, 0) + 1
        out.append(f)
    return out


class EncoderSet:
    """Resolves an encoder callable per index; live checkpoints are used only when requested AND loadable."""

    def __init__(self, mode: str) -> None:
        self.mode = mode
        self.status: Dict[str, str] = {}
        self._live: Dict[int, bool] = {}
        for i, (eid, name, dim) in enumerate(ENCODER_CHAIN):
            if mode == "run-live-encoders":
                ok, why = live_encoder_available(i)
                self._live[i] = ok
                self.status[eid] = ("live:" + name) if ok else ("stand_in (" + why + ")")
            else:
                self._live[i] = False
                self.status[eid] = "stand_in (deterministic text-derived)"

    def encode(self, texts: Sequence[str], i: int) -> np.ndarray:
        return live_encode(texts, i) if self._live[i] else dry_encode(texts, i)


@dataclasses.dataclass
class HopResult:
    hop: int
    src_enc: int
    top5: np.ndarray            # [n_probe, 5] doc indices
    pred_class: List[str]       # top-1 class
    recall1: int                # ppm
    recall5: int                # ppm
    class_acc: int              # ppm
    kappa_ppm_dev: int          # |kappa - 1| in 1e-9 units (integer)
    nonzero_kappa: float


class FleetRetrieval:
    """Builds the 4096-d index, anchor sets, pairwise bridges (3 families) and per-hop retrieval outputs."""

    def __init__(self, mode: str = "local-drycheck") -> None:
        self.mode = mode
        self.enc = EncoderSet(mode)
        self.index_flows = unique_flows(INDEX_SEED, DOCS_PER_CLASS, 80)
        self.index_texts = [flow_text(serialize_tls_flow(f)) for f in self.index_flows]
        self.index_feats = [serialize_tls_flow(f) for f in self.index_flows]
        self.doc_class = [f.cls for f in self.index_flows]
        self.doc_tranche = [f.tranche for f in self.index_flows]
        anchors = unique_flows(ANCHOR_SEED, ANCHORS_PER_CLASS, ANCHORS_PER_CLASS, exclude=self.index_texts)
        self.anchor_texts = [flow_text(serialize_tls_flow(f)) for f in anchors]
        self.docs = self.enc.encode(self.index_texts, 4)                         # native ground index
        self.anchor_vecs = [self.enc.encode(self.anchor_texts, i) for i in range(5)]
        self.probe_vecs = [self.enc.encode(self.index_texts, i) for i in range(4)]  # tranche/hop sources
        self.burn_check = self._burn_check()
        self.bridges: Dict[str, List[Any]] = {}
        for fam in FAMILIES:
            self.bridges[fam] = [self._fit(fam, self.anchor_vecs[k], self.anchor_vecs[k + 1]) for k in range(4)]
        self.hops: Dict[str, Dict[int, HopResult]] = {a: {} for a in ("raw", "ridge", "procrustes", "zeromu")}
        self.sidecar: Dict[str, Any] = {}
        self._run_hops()
        self.native_ceiling = self._native_ceiling()
        self.tra1 = self._tra1()

    # -- helpers
    @staticmethod
    def _fit(fam: str, x: np.ndarray, y: np.ndarray) -> Any:
        if fam == "ridge":
            return RidgeDriftAdapter.fit(x, y, regularization=1e-2)
        if fam == "procrustes":
            return UncenteredProcrustes.fit(x, y)
        return ZeroMuPolarBridge.fit(x, y, scoring_mode="dual_fit")

    def _burn_check(self) -> Dict[str, Any]:
        try:
            assert_corpus_unburned("fabric_synthetic_tls_v1", doc_texts=self.index_texts[:8], query_texts=self.anchor_texts[:8])
            return {"checked": True, "burned": False}
        except BurnedCorpusError as exc:
            msg = str(exc)
            if "BURNED as" in msg or "collision" in msg.lower():
                return {"checked": True, "burned": True, "detail": msg[:120]}
            return {"checked": False, "burned": False,
                    "detail": "disjointness not established: no historical fingerprint inventory supplied (corpus is synthetic, generated in-process)"}
        except Exception as exc:  # noqa: BLE001 - the ledger may demand inventories we do not have
            return {"checked": False, "burned": False, "detail": type(exc).__name__}

    def chain(self, fam: str, hop: int) -> Any:
        src = 4 - hop
        return compose_bridge_chain(self.bridges[fam][src:4], polar_reproject=False)

    def _classify(self, scores: np.ndarray, hop: int, src: int, nk: float) -> HopResult:
        order = np.argsort(-scores, axis=1)[:, :5]
        n = scores.shape[0]
        pred = [self.doc_class[int(order[i, 0])] for i in range(n)]
        r1 = sum(1 for i in range(n) if int(order[i, 0]) == i)
        r5 = sum(1 for i in range(n) if i in {int(j) for j in order[i]})
        ca = sum(1 for i in range(n) if pred[i] == self.doc_class[i])
        return HopResult(hop, src, order, pred, r1 * 1_000_000 // n, r5 * 1_000_000 // n, ca * 1_000_000 // n,
                         int(round(abs(nk - 1.0) * 1e9)), float(nk))

    def _run_hops(self) -> None:
        docs = self.docs
        dev_idx = np.random.default_rng(5).choice(len(self.anchor_texts), DEV_REF_COUNT, replace=False)
        for hop in (1, 2, 3, 4):
            src = 4 - hop
            q = self.probe_vecs[src]
            dev_q = self.anchor_vecs[src][dev_idx]
            # arm 1/5: raw zero-padded cosine (no bridge)
            pad = np.zeros((q.shape[0], NATIVE_DIM))
            pad[:, : q.shape[1]] = q
            raw = RawCosineMerge().merge(l2_normalize(pad) @ docs.T, l2_normalize(pad) @ docs.T,
                                         np.zeros(docs.shape[0], dtype=bool))
            self.hops["raw"][hop] = self._classify(raw, hop, src, 1.0)
            # arm 2: ridge chain + z-score merge
            rc = self.chain("ridge", hop)
            rs = rc.transform(q, normalize=True) @ docs.T
            rref = rc.transform(dev_q, normalize=True) @ docs.T
            rz = CalibratedZScoreMerge().merge(rs, rs, np.zeros(docs.shape[0], dtype=bool), bridged_ref=rref, native_ref=rref)
            self.hops["ridge"][hop] = self._classify(rz, hop, src, float(rc.nonzero_condition_number))
            # arm 6: orthogonal chain + CSLS
            pc = self.chain("procrustes", hop)
            ps = pc.transform(q, normalize=True) @ docs.T
            pref = pc.transform(dev_q, normalize=True) @ docs.T
            pz = CSLSMerge().merge(ps, ps, np.zeros(docs.shape[0], dtype=bool), bridged_ref=pref, native_ref=pref)
            self.hops["procrustes"][hop] = self._classify(pz, hop, src, float(pc.nonzero_condition_number))
            # arm 7: zero-mu chain + sidecar scoring + CSLS
            zc = self.chain("zeromu", hop)
            mu_x, mu_y, mat, _ = zc.linear_components()
            dn = l2_normalize(docs)
            mu = mu_y.reshape(-1)
            den = np.sqrt(np.maximum(1.0 + float(mu @ mu) - 2.0 * (dn @ mu), 1e-12))
            qh = l2_normalize((q - mu_x) @ mat)
            zs = (qh @ dn.T - (qh @ mu)[:, None]) / den[None, :]
            dh = l2_normalize((dev_q - mu_x) @ mat)
            zref = (dh @ dn.T - (dh @ mu)[:, None]) / den[None, :]
            k = max(1, min(DEFAULT_CSLS_K, zref.shape[0]))
            r_d = np.partition(zref, zref.shape[0] - k, axis=0)[-k:, :].mean(axis=0)
            den32 = den.astype(np.float32).astype(np.float64)
            rd32 = r_d.astype(np.float32).astype(np.float64)
            blob = build_sidecar(den32, rd32)
            sc_den, sc_rd = read_sidecar(blob)
            s_side = (qh @ dn.T - (qh @ mu)[:, None]) / sc_den[None, :]
            kq = max(1, min(DEFAULT_CSLS_K, s_side.shape[1]))
            r_q = np.partition(s_side, s_side.shape[1] - kq, axis=1)[:, -kq:].mean(axis=1)
            side_final = 2.0 * s_side - sc_rd[None, :] - r_q[:, None]
            harness_final = CSLSMerge().merge(zs, zs, np.zeros(docs.shape[0], dtype=bool), bridged_ref=zref, native_ref=zref)
            self.hops["zeromu"][hop] = self._classify(harness_final, hop, src, float(zc.nonzero_condition_number))
            parity = float(np.max(np.abs(side_final - harness_final)))
            argmax_same = bool(np.all(np.argmax(side_final, axis=1) == np.argmax(harness_final, axis=1)))
            if hop == 1:
                direct = self.bridges["zeromu"][3].score_matrix(q, docs)
                direct_parity = float(np.max(np.abs(direct - zs)))
            else:
                direct_parity = -1.0
            self.sidecar[str(hop)] = {"bytes_per_flow": len(blob) // docs.shape[0], "total_bytes": len(blob),
                                      "max_abs_parity_ppm": int(parity * 1e6), "argmax_identical": argmax_same,
                                      "direct_harness_parity_ppm": int(direct_parity * 1e6) if direct_parity >= 0 else -1}

    def _native_ceiling(self) -> Dict[str, int]:
        q = self.enc.encode(self.index_texts, 4)  # same texts through the native encoder again (different object, same noise seed)
        sims = q @ self.docs.T
        order = np.argmax(sims, axis=1)
        n = len(order)
        return {"recall1_ppm": sum(1 for i in range(n) if order[i] == i) * 1_000_000 // n,
                "class_acc_ppm": sum(1 for i in range(n) if self.doc_class[order[i]] == self.doc_class[i]) * 1_000_000 // n}

    def _tra1(self) -> Dict[str, Any]:
        """TRA-1 over the real chain bridges and over a deliberately degenerate anchor set."""
        out: Dict[str, Any] = {"chain": {}}
        for k, b in enumerate(self.bridges["zeromu"]):
            g = tra1_gate(self.anchor_vecs[k], self.anchor_vecs[k + 1], b)
            out["chain"][f"b{k}{k + 1}"] = g["status"]
        out["chain_all_pass"] = all(v == "PASS" for v in out["chain"].values())
        x = np.repeat(self.anchor_vecs[0][:1], 64, axis=0) + 1e-9 * np.random.default_rng(1).standard_normal((64, self.anchor_vecs[0].shape[1]))
        y = np.repeat(self.anchor_vecs[1][:1], 64, axis=0) + 1e-9 * np.random.default_rng(2).standard_normal((64, self.anchor_vecs[1].shape[1]))
        try:
            bad = ZeroMuPolarBridge.fit(x, y, scoring_mode="dual_fit")
            gate = tra1_gate(x, y, bad)
        except Exception as exc:  # noqa: BLE001 - a bridge that cannot even be fitted is refused too
            gate = {"status": "FAIL_CLOSED_DEGENERATE", "admit": False, "fit_error": type(exc).__name__}
        out["degenerate_anchor_status"] = gate["status"]
        out["degenerate_anchor_refused"] = not gate["admit"]
        return out

    # -- reporting
    def probe_hop(self, i: int) -> int:
        return 4 - self.doc_tranche[i]

    def pred_class_for(self, arm_key: str, i: int) -> str:
        return self.hops[arm_key][self.probe_hop(i)].pred_class[i]

    def hop_table(self) -> List[Dict[str, Any]]:
        rows = []
        for hop in (1, 2, 3, 4):
            row: Dict[str, Any] = {"hop": hop, "source_encoder": f"enc_{'abcde'[4 - hop]}"}
            for key in ("raw", "ridge", "procrustes", "zeromu"):
                r = self.hops[key][hop]
                row[key] = {"recall1_ppm": r.recall1, "recall5_ppm": r.recall5, "class_acc_ppm": r.class_acc,
                            "kappa_dev_nano": r.kappa_ppm_dev}
            rows.append(row)
        return rows


@functools.lru_cache(maxsize=4)
def get_retrieval(mode: str = "local-drycheck") -> FleetRetrieval:
    return FleetRetrieval(mode)


# ---------------------------------------------------------------------------
# Control-plane arms (state machines). Arms 1/2 and 5/6 share a state machine; they differ in retrieval.
# ---------------------------------------------------------------------------
LEDGER_T = 10 ** 6
SLOT_CLAIM = "policy"
SLOT_MEMO = "memo"


class LastArrivalArm:
    """Arms 1 and 2: the last claim seen for the matched class wins; tiers and retractions are ignored."""

    def __init__(self) -> None:
        self.last: Dict[str, str] = {}
        self.derived: Dict[str, str] = {}
        self.audit: Dict[str, str] = {}

    def ingest(self, ev: FEvent) -> None:
        if ev.kind in CLAIM_KINDS:
            self.last[ev.cls] = ev.claim
        elif ev.kind == KIND_DERIVED:
            self.derived[ev.eid] = ev.target
        elif ev.kind == KIND_AUDIT:
            self.audit[ev.eid] = ev.target

    def verdicts(self) -> Dict[str, str]:
        return {c: self.last.get(c, VERDICT_DEFER) for c in CLASS_IDS}

    def derived_served(self) -> Set[str]:
        return set(self.derived)

    def audit_retained(self) -> Set[str]:
        return set(self.audit)


class PopularityArm:
    """Arm 3 (authored emulation): plurality claim among non-retracted records, whatever their authority."""

    def __init__(self) -> None:
        self.records: List[Tuple[str, str, str]] = []
        self.retracted: Set[str] = set()
        self.derived: Dict[str, str] = {}
        self.audit: Dict[str, str] = {}

    def ingest(self, ev: FEvent) -> None:
        if ev.kind in CLAIM_KINDS:
            self.records.append((ev.eid, ev.cls, ev.claim))
        elif ev.kind == KIND_RETRACT:
            self.retracted.add(ev.target)
        elif ev.kind == KIND_DERIVED:
            self.derived[ev.eid] = ev.target
        elif ev.kind == KIND_AUDIT:
            self.audit[ev.eid] = ev.target

    def verdicts(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for c in CLASS_IDS:
            counts: Dict[str, int] = {}
            for eid, cls, claim in self.records:
                if cls == c and eid not in self.retracted:
                    counts[claim] = counts.get(claim, 0) + 1
            if not counts:
                out[c] = VERDICT_DEFER
                continue
            top = max(counts.values())
            winners = sorted(k for k, v in counts.items() if v == top)
            out[c] = winners[0] if len(winners) == 1 else VERDICT_DEFER
        return out

    def derived_served(self) -> Set[str]:
        return set(self.derived)

    def audit_retained(self) -> Set[str]:
        return set(self.audit)


class OverwriteArm:
    """Arm 4: last arrival replaces the previous record; a retraction is a class tombstone (served as DEFER)."""

    def __init__(self) -> None:
        self.state: Dict[str, Optional[str]] = {}
        self.derived: Dict[str, str] = {}
        self.audit: Dict[str, str] = {}

    def ingest(self, ev: FEvent) -> None:
        if ev.kind in CLAIM_KINDS:
            self.state[ev.cls] = ev.claim
        elif ev.kind == KIND_RETRACT:
            self.state[ev.cls] = None
        elif ev.kind == KIND_DERIVED:
            self.derived[ev.eid] = ev.target
        elif ev.kind == KIND_AUDIT:
            self.audit[ev.eid] = ev.target

    def verdicts(self) -> Dict[str, str]:
        return {c: (self.state.get(c) or VERDICT_DEFER) for c in CLASS_IDS}

    def derived_served(self) -> Set[str]:
        return set(self.derived)

    def audit_retained(self) -> Set[str]:
        return set(self.audit)


class LedgerArm:
    """Arms 5 and 6: records live in the spine bitemporal ledger (authority, withdrawal, DERIVES edges)."""

    def __init__(self) -> None:
        self.ledger = ChallengerBitemporalLedger(SQLiteChallengerBackend(":memory:"))
        self.tick = 0
        self.derived: Dict[str, str] = {}
        self.audit: Dict[str, str] = {}
        self._edge_n = 0
        self._cls_of: Dict[str, str] = {}

    def _assert(self, rid: str, entity: str, tier: int, claim: str, t: int) -> None:
        self.ledger.append(LedgerRecordEnvelope(TENANT, rid, entity, tier, t, None, t, "ASSERT", 0, claim))

    def ingest(self, ev: FEvent) -> None:
        self.tick += 1
        t = self.tick
        if ev.kind in CLAIM_KINDS:
            self._assert(ev.eid, ev.cls, ev.tier, ev.claim, t)
        elif ev.kind == KIND_RETRACT:
            self.ledger.append(LedgerRecordEnvelope(TENANT, ev.target, ev.cls, ev.tier, t, None, t, "WITHDRAW", 0, ""))
        elif ev.kind in (KIND_DERIVED, KIND_AUDIT):
            suffix = "#derived" if ev.kind == KIND_DERIVED else "#audit"
            self._cls_of[ev.eid] = ev.cls
            self._assert(ev.eid, ev.cls + suffix, ev.tier, "memo", t)
            (self.derived if ev.kind == KIND_DERIVED else self.audit)[ev.eid] = ev.target
            self._edge(ev.eid, ev.target, t)
        for et, dst in ev.extra_edges:
            if et == "DERIVES_FROM":
                self._edge(ev.eid, dst, t)

    def _edge(self, src: str, dst: str, t: int) -> None:
        self._edge_n += 1
        self.ledger.add_edge(LedgerEdgeEnvelope(TENANT, f"le{self._edge_n}", src, dst, t))

    def _q(self, entity: str) -> Any:
        return self.ledger.query_as_of(TENANT, entity, LEDGER_T, LEDGER_T)

    def verdicts(self) -> Dict[str, str]:
        out: Dict[str, str] = {}
        for c in CLASS_IDS:
            res = self._q(c)
            out[c] = res.winner_claim_value if (res.status == "RESOLVED" and res.winner_claim_value) else VERDICT_DEFER
        return out

    def derived_served(self) -> Set[str]:
        return {e for e in self.derived if self._q(self._ent(e, "#derived")).status == "RESOLVED"}

    def audit_retained(self) -> Set[str]:
        return {e for e in self.audit if self._q(self._ent(e, "#audit")).status == "RESOLVED"}

    def _ent(self, eid: str, suffix: str) -> str:
        return self._cls_of[eid] + suffix



# ---------------------------------------------------------------------------
# EK5 feed (production side of the oracle comparison): events + edges with a pending-edge queue
# ---------------------------------------------------------------------------
class EK5Pipe:
    """Feeds the EK5 v3 store; an edge whose endpoints have not both arrived is held until they do."""

    def __init__(self) -> None:
        self.store = StratifiedEvidenceStore(strict_dag_supersede=True, buffer_pending=False)
        self.queue: List[EdgeAttestation] = []
        self.seen_edge_ids: Set[str] = set()
        self.ops: List[Tuple[Any, ...]] = []

    def event(self, ev: StratifiedEvent) -> None:
        self.ops.append(("event", ev))
        present = set(self.store.events) | {ev.event_id}
        ready = [a for a in self.queue if a.src_id in present and a.dst_id in present]
        self.queue = [a for a in self.queue if a not in ready]
        self.store.apply_mutation([ev], ready)

    def edge(self, att_id: str, etype: str, src: str, dst: str, t: int = 0) -> None:
        if att_id in self.seen_edge_ids:
            return
        self.seen_edge_ids.add(att_id)
        self.ops.append(("edge", att_id, etype, src, dst))
        att = EdgeAttestation(att_id, src, dst, etype, valid_time=t, tx_time=t)
        if src in self.store.events and dst in self.store.events:
            self.store.apply_mutation([], [att])
        else:
            self.queue.append(att)


def read_verdicts_from(items: Sequence[Tuple[str, str, int, str]]) -> Dict[str, str]:
    """Verdict reader over (class, claim, tier, eligibility) rows of claim events."""
    out: Dict[str, str] = {}
    for c in CLASS_IDS:
        adm = [(t, cl) for (k, cl, t, el) in items if k == c and el == "ADMIT"]
        con = [(t, cl) for (k, cl, t, el) in items if k == c and el == "CONTESTED"]
        a_top = max((t for t, _ in adm), default=-1)
        c_top = max((t for t, _ in con), default=-1)
        if con and (not adm or c_top >= a_top):
            out[c] = VERDICT_DEFER
        elif adm:
            claims = {cl for t, cl in adm if t == a_top}
            out[c] = next(iter(claims)) if len(claims) == 1 else VERDICT_DEFER
        else:
            out[c] = VERDICT_DEFER
    return out


class FabricArm:
    """Arm 7: echo-cluster collapse + conflict detection + explicit refutation over the EK5 lineage store."""

    def __init__(self) -> None:
        self.pipe = EK5Pipe()
        self.events: Dict[str, FEvent] = {}
        self.arrival: Dict[str, int] = {}
        self.clock = 0
        self.echo_root: Dict[str, str] = {}
        self.wanted: Dict[str, Tuple[str, str, str]] = {}  # att_id -> (type, src, dst)

    def ingest(self, ev: FEvent) -> None:
        self.clock += 1
        t = self.clock
        if ev.eid in self.events:
            return
        self.events[ev.eid] = ev
        self.arrival[ev.eid] = t
        slot = SLOT_CLAIM if ev.kind in CLAIM_KINDS else SLOT_MEMO
        self.pipe.event(StratifiedEvent(event_id=ev.eid, entity=ev.cls, slot=slot, claim_value=ev.claim, text=ev.text,
                                        valid_time=t, tx_time=t, tenant=TENANT, domain="fabric", authority_tier=ev.tier))
        self._edges_for(ev, t)

    def _want(self, att_id: str, etype: str, src: str, dst: str, t: int) -> None:
        if att_id not in self.wanted:
            self.wanted[att_id] = (etype, src, dst)
            self.pipe.edge(att_id, etype, src, dst, t)

    def _edges_for(self, ev: FEvent, t: int) -> None:
        if ev.kind in CLAIM_KINDS:
            roots = infer_echo_clusters(list(self.events.values()), self.arrival)
            if ev.eid in roots:
                self.echo_root[ev.eid] = roots[ev.eid]
                self._want(f"echo:{ev.eid}", "COPY_OF", ev.eid, roots[ev.eid], t)
            # control-plane conflict detector: same class, different claim, equal tier -> mutual refutation
            for other in self.events.values():
                if (other.eid != ev.eid and other.kind in CLAIM_KINDS and other.cls == ev.cls
                        and other.tier == ev.tier and other.claim != ev.claim):
                    a, b = sorted((ev.eid, other.eid))
                    self._want(f"conf:{a}>{b}", "REFUTE", a, b, t)
                    self._want(f"conf:{b}>{a}", "REFUTE", b, a, t)
            if ev.kind == KIND_RULING and ev.target:
                self._want(f"ref:{ev.eid}>{ev.target}", "REFUTE", ev.eid, ev.target, t)
        elif ev.kind == KIND_RETRACT:
            self._want(f"ref:{ev.eid}>{ev.target}", "REFUTE", ev.eid, ev.target, t)
        elif ev.kind == KIND_DERIVED:
            self._want(f"der:{ev.eid}", "DERIVES_CURRENT", ev.eid, ev.target, t)
        elif ev.kind == KIND_AUDIT:
            self._want(f"aud:{ev.eid}", "CITES_AUDIT", ev.eid, ev.target, t)
        for et, dst in ev.extra_edges:
            self._want(f"x:{et}:{ev.eid}>{dst}", et, ev.eid, dst, t)

    @property
    def store(self) -> StratifiedEvidenceStore:
        return self.pipe.store

    def claim_rows(self) -> List[Tuple[str, str, int, str]]:
        rows = []
        for eid, ev in self.events.items():
            if ev.kind in CLAIM_KINDS:
                rows.append((ev.cls, ev.claim, int(self.store.eff_auth[eid]), self.store.eligibility[eid]))
        return rows

    def verdicts(self) -> Dict[str, str]:
        return read_verdicts_from(self.claim_rows())

    def derived_served(self) -> Set[str]:
        return {e for e, ev in self.events.items() if ev.kind == KIND_DERIVED and self.store.eligibility[e] == "ADMIT"}

    def audit_retained(self) -> Set[str]:
        return {e for e, ev in self.events.items() if ev.kind == KIND_AUDIT and e in self.store.events}

    def audit_in_trace(self) -> Set[str]:
        trace = export_canonical_ek5_audit_trace(self.store)
        ids = set(trace.by_id())
        return {e for e, ev in self.events.items() if ev.kind == KIND_AUDIT and e in ids}

    def echo_collapse(self) -> Dict[str, int]:
        n_echo = len(self.echo_root)
        n_clusters = len(set(self.echo_root.values()))
        return {"echo_records": n_echo, "echo_clusters": n_clusters}


# ---------------------------------------------------------------------------
# Independent state-transition prefix oracle: 0 shared decision code with EK5
# ---------------------------------------------------------------------------
class RefOracle:
    """Re-derives val / cur / eff_auth / eligibility from the raw op stream with its own graph code.

    Only the *semantics* documented for EK5 are mirrored: dependency edges (src depends on dst), authority-qualified
    refutation, positive cycles forced false, and the alternating fixpoint (K0 = {}, U = G(K), K' = G(U)).
    """

    DV = ("COPY_OF", "DERIVES_CURRENT", "DERIVES_FROM", "AMENDS", "DERIVES_SNAPSHOT")
    DC = ("COPY_OF", "DERIVES_CURRENT", "DERIVES_FROM")

    def __init__(self) -> None:
        self.tier: Dict[str, int] = {}
        self.held: List[Tuple[str, str, str]] = []
        self.edges: Set[Tuple[str, str, str]] = set()

    def event(self, eid: str, tier: int) -> None:
        if eid not in self.tier:
            self.tier[eid] = tier
            self._release()

    def edge(self, typ: str, src: str, dst: str) -> None:
        self.held.append((typ, src, dst))
        self._release()

    def _release(self) -> None:
        keep = []
        for e in self.held:
            if e[1] in self.tier and e[2] in self.tier:
                self.edges.add(e)
            else:
                keep.append(e)
        self.held = keep

    def solve(self) -> Dict[str, Tuple[str, str, int, str]]:
        nodes = sorted(self.tier)
        par_v: Dict[str, Set[str]] = {n: set() for n in nodes}
        par_c: Dict[str, Set[str]] = {n: set() for n in nodes}
        refs: Dict[str, Set[str]] = {n: set() for n in nodes}
        for typ, s, d in self.edges:
            if typ in self.DV:
                par_v[s].add(d)
            if typ in self.DC:
                par_c[s].add(d)
            if typ == "REFUTE":
                refs[d].add(s)

        def reaches_self(n: str) -> bool:
            seen: Set[str] = set()
            stack = list(par_v[n])
            while stack:
                x = stack.pop()
                if x == n:
                    return True
                if x in seen:
                    continue
                seen.add(x)
                stack.extend(par_v[x])
            return False

        cyc = {n for n in nodes if reaches_self(n)}
        eff_memo: Dict[str, int] = {}

        def eff(n: str) -> int:
            if n in eff_memo:
                return eff_memo[n]
            if n in cyc:
                eff_memo[n] = 0
                return 0
            v = self.tier[n]
            for p in par_v[n]:
                v = min(v, eff(p))
            eff_memo[n] = v
            return v

        for n in nodes:
            eff(n)
        active: Dict[str, Set[str]] = {n: {u for u in refs[n] if eff_memo[u] >= eff_memo[n]} for n in nodes}

        def gamma(j: Set[str]) -> Set[str]:
            s: Set[str] = set()
            changed = True
            while changed:
                changed = False
                for n in nodes:
                    if n in s or n in cyc:
                        continue
                    if any(r in j for r in active[n]):
                        continue
                    if all(p in s for p in par_v[n]):
                        s.add(n)
                        changed = True
            return s

        k: Set[str] = set()
        while True:
            u = gamma(k)
            k2 = gamma(u)
            if k2 == k:
                break
            k = k2
        over = gamma(k)
        val = {n: ("T" if n in k else ("U" if n in over else "F")) for n in nodes}
        cur_memo: Dict[str, str] = {}

        def cur(n: str, guard: Set[str]) -> str:
            if n in cur_memo:
                return cur_memo[n]
            if val[n] != "T":
                cur_memo[n] = val[n]
                return val[n]
            if n in guard:
                cur_memo[n] = "F"
                return "F"
            guard.add(n)
            res = "T"
            for p in par_c[n]:
                c = cur(p, guard)
                if c == "F":
                    res = "F"
                    break
                if c == "U":
                    res = "U"
            guard.discard(n)
            cur_memo[n] = res
            return res

        out: Dict[str, Tuple[str, str, int, str]] = {}
        for n in nodes:
            c = cur(n, set())
            if val[n] == "T" and c == "T":
                el = "ADMIT"
            elif val[n] == "F" or c == "F":
                el = "AUDIT_ONLY"
            else:
                el = "CONTESTED"
            out[n] = (val[n], c, eff_memo[n], el)
        return out


def oracle_verdicts(solved: Mapping[str, Tuple[str, str, int, str]], meta: Mapping[str, Tuple[str, str, bool]]) -> Dict[str, str]:
    """Independently written verdict reader: ``meta[eid] = (class, claim, is_claim_event)``."""
    out: Dict[str, str] = {}
    for c in CLASS_IDS:
        live = [(solved[e][2], meta[e][1]) for e in solved if meta[e][2] and meta[e][0] == c and solved[e][3] == "ADMIT"]
        held = [solved[e][2] for e in solved if meta[e][2] and meta[e][0] == c and solved[e][3] == "CONTESTED"]
        best = max((t for t, _ in live), default=-1)
        if held and max(held) >= best:
            out[c] = VERDICT_DEFER
        elif live:
            top = {cl for t, cl in live if t == best}
            out[c] = sorted(top)[0] if len(top) == 1 else VERDICT_DEFER
        else:
            out[c] = VERDICT_DEFER
    return out


@dataclasses.dataclass
class OracleReport:
    traces: int = 0
    prefixes: int = 0
    disagreements: int = 0
    register_checks: int = 0
    register_disagreements: int = 0
    reserved_injected: int = 0
    reserved_forward_leaks_fenced: int = 0
    reserved_forward_leaks_naive: int = 0
    torn_injected: int = 0
    torn_unfenced_mixed: int = 0
    first_mismatches: List[str] = dataclasses.field(default_factory=list)

    def merge(self, other: "OracleReport") -> None:
        for f in dataclasses.fields(self):
            if f.name == "first_mismatches":
                self.first_mismatches.extend(other.first_mismatches)
                self.first_mismatches = self.first_mismatches[:8]
            else:
                setattr(self, f.name, getattr(self, f.name) + getattr(other, f.name))


def check_ops_against_oracle(ops: Sequence[Tuple[Any, ...]], label: str, rng: Optional[random.Random] = None,
                             register_probe: bool = True) -> OracleReport:
    """Replays ``ops`` into a FRESH EK5 store and the oracle; compares every prefix; probes the register at every prefix."""
    rep = OracleReport(traces=1)
    pipe = EK5Pipe()
    orc = RefOracle()
    meta: Dict[str, Tuple[str, str, bool]] = {}
    last_good: Dict[str, str] = {c: VERDICT_DEFER for c in CLASS_IDS}
    epoch = 1
    rng = rng or random.Random(0)
    for idx, op in enumerate(ops):
        if op[0] == "event":
            ev: StratifiedEvent = op[1]
            pipe.event(ev)
            orc.event(ev.event_id, ev.authority_tier)
            if ev.event_id not in meta:
                meta[ev.event_id] = (ev.entity, ev.claim_value, ev.slot == SLOT_CLAIM and bool(ev.claim_value))
        else:
            _, att_id, etype, src, dst = op
            pipe.edge(att_id, etype, src, dst)
            orc.edge(etype, src, dst)
        solved = orc.solve()
        rep.prefixes += 1
        st = pipe.store
        bad = []
        for eid, (v, c, ea, el) in solved.items():
            got = (st.val[eid], st.cur[eid], int(st.eff_auth[eid]), st.eligibility[eid])
            if (v, c, ea, el) != got:
                bad.append(f"{label}#{idx} {eid}: oracle={(v, c, ea, el)} ek5={got}")
        if set(st.events) != set(solved):
            bad.append(f"{label}#{idx} node-set differs")
        rows = [(meta[e][0], meta[e][1], int(st.eff_auth[e]), st.eligibility[e]) for e in st.events if meta[e][2]]
        v_ek5 = read_verdicts_from(rows)
        v_orc = oracle_verdicts(solved, meta)
        if v_ek5 != v_orc:
            bad.append(f"{label}#{idx} verdict reader differs")
        if bad:
            rep.disagreements += len(bad)
            if len(rep.first_mismatches) < 8:
                rep.first_mismatches.extend(bad[:2])
        if register_probe:
            _probe_register(rep, v_ek5, last_good, epoch, rng)
            last_good = dict(v_ek5)
    return rep


def _probe_register(rep: OracleReport, verdicts: Mapping[str, str], last_good: Mapping[str, str], epoch: int,
                    rng: random.Random) -> None:
    img = compile_register(epoch, verdicts)
    mode = rng.choice(("clean", "clean", "torn", "reserved"))
    want = dict(verdicts)
    stats_expect_last_good = False
    if mode == "torn":
        rep.torn_injected += 1
        bad_epoch = epoch - 1 if rng.random() < 0.5 else epoch + 7
        buf = bytearray(img)
        buf[-REG_TRL:] = struct.pack(">I", bad_epoch)  # trailer disagrees with header: a torn write
        img = bytes(buf)
        stats_expect_last_good = True
        if decode_register_naive(img) != decode_register_naive(compile_register(epoch, verdicts)):
            rep.torn_unfenced_mixed += 1
    elif mode == "reserved":
        rep.reserved_injected += 1
        victim = rng.choice(CLASS_IDS)
        img = seu_flip_to_reserved(img, victim)
        want[victim] = VERDICT_INVALID
        if decode_register_naive(img)[victim] == VERDICT_FORWARD:
            rep.reserved_forward_leaks_naive += 1
    got, _ = decode_register_fenced(img, epoch, last_good)
    ref = decode_register_reference(img, epoch, last_good)
    rep.register_checks += 1
    expect = dict(last_good) if stats_expect_last_good else want
    if got != ref or got != expect:
        rep.register_disagreements += 1
    if mode == "reserved" and any(got[c] == VERDICT_FORWARD and want[c] == VERDICT_INVALID for c in CLASS_IDS):
        rep.reserved_forward_leaks_fenced += 1


def gen_fuzz_ops(seed: int) -> List[Tuple[Any, ...]]:
    """Seeded randomized trace: out-of-order delivery, duplicate events, edges before endpoints, cycles."""
    rng = random.Random(seed)
    classes = rng.sample(list(CLASS_IDS), rng.randint(2, 4))
    n_ev = rng.randint(7, 15)
    ids = [f"z{seed}_{i}" for i in range(n_ev)]
    events: List[Tuple[Any, ...]] = []
    for i, eid in enumerate(ids):
        cls = rng.choice(classes)
        memo = rng.random() < 0.2
        events.append(("event", StratifiedEvent(
            event_id=eid, entity=cls, slot=SLOT_MEMO if memo else SLOT_CLAIM,
            claim_value="" if memo else rng.choice((VERDICT_FORWARD, VERDICT_DROP)), text=f"fuzz {i}",
            valid_time=i, tx_time=i, tenant=TENANT, domain="fabric", authority_tier=rng.randint(1, 3))))
    edges: List[Tuple[Any, ...]] = []
    n_edge = rng.randint(n_ev // 2, n_ev + 4)
    for j in range(n_edge):
        a, b = rng.sample(ids, 2)
        et = rng.choice(("REFUTE", "REFUTE", "COPY_OF", "DERIVES_CURRENT", "CITES_AUDIT", "DERIVES_FROM"))
        edges.append(("edge", f"fa{seed}_{j}", et, a, b))
    if rng.random() < 0.35 and n_ev >= 3:  # positive cycle injection (2- or 3-cycle)
        k = rng.choice((2, 3))
        cyc = rng.sample(ids, k)
        for q in range(k):
            edges.append(("edge", f"fc{seed}_{q}", "DERIVES_FROM", cyc[q], cyc[(q + 1) % k]))
    if rng.random() < 0.5 and edges:  # duplicate edge under a new attestation id
        e = rng.choice(edges)
        edges.append(("edge", f"fd{seed}_0", e[2], e[3], e[4]))
    ops = events + edges
    if events and rng.random() < 0.6:  # duplicate event insertion (re-insert is ignored by both sides)
        ops.append(rng.choice(events))
    rng.shuffle(ops)
    return ops


def run_fuzz(n_traces: int = 260, base_seed: int = 1000) -> OracleReport:
    total = OracleReport()
    for i in range(n_traces):
        ops = gen_fuzz_ops(base_seed + i)
        total.merge(check_ops_against_oracle(ops, f"fuzz{base_seed + i}", random.Random(base_seed + i)))
    return total


# ---------------------------------------------------------------------------
# Scenario runner
# ---------------------------------------------------------------------------
def action_of(verdict: str) -> str:
    return VERDICT_DEFER if verdict == VERDICT_INVALID else verdict


def _make_arm(key: str) -> Any:
    if key in ("uncalibrated_fleet_rag", "chained_ridge_zscore"):
        return LastArrivalArm()
    if key == "unsigned_popularity_correlator":
        return PopularityArm()
    if key == "overwrite_last_arrival":
        return OverwriteArm()
    if key in ("spine_bitemporal_ledger_raw", "procrustes_csls_plus_spine_ledger"):
        return LedgerArm()
    return FabricArm()


ARM_HOP_KEY = {"uncalibrated_fleet_rag": "raw", "chained_ridge_zscore": "ridge", "spine_bitemporal_ledger_raw": "raw",
               "procrustes_csls_plus_spine_ledger": "procrustes", "chelated_fabric_control": "zeromu"}
SIGNATURE_ARMS = ("unsigned_popularity_correlator", "overwrite_last_arrival")


def signature_table(fr: "FleetRetrieval") -> Tuple[Dict[str, str], int]:
    """Exact-lookup table (signature -> first class registered); returns collisions for disclosure."""
    table: Dict[str, str] = {}
    collisions: Set[str] = set()
    for c in CLASS_IDS:
        for i, f in enumerate(fr.index_feats):
            if fr.doc_class[i] != c:
                continue
            s = class_signature(f)
            if s not in table:
                table[s] = c
            elif table[s] != c:
                collisions.add(s)
    return table, len(collisions)


def flow_pred_class(fr: "FleetRetrieval", arm: str, i: int, sig: Mapping[str, str]) -> str:
    if arm in SIGNATURE_ARMS:
        return sig.get(class_signature(fr.index_feats[i]), "")
    return fr.pred_class_for(ARM_HOP_KEY[arm], i)


def run_scenario(name: str, fr: "FleetRetrieval", steps: Sequence[Step]) -> Dict[str, Any]:
    sig, n_collisions = signature_table(fr)
    n_docs = len(fr.doc_class)
    out: Dict[str, Any] = {"scenario": name, "steps": [], "arms": {}, "signature_collisions": n_collisions}
    arms = {k: _make_arm(k) for k in ARMS}
    per_arm: Dict[str, Dict[str, Any]] = {k: {"state_correct": 0, "state_total": 0, "flow_correct": 0, "flow_total": 0,
                                                 "steps": []} for k in ARMS}
    fabric: FabricArm = arms["chelated_fabric_control"]  # type: ignore[assignment]
    last_good = {c: VERDICT_DEFER for c in CLASS_IDS}
    register_log: List[Dict[str, Any]] = []
    seu_demo: Dict[str, Any] = {}
    first_verdicts: Dict[str, str] = dict(last_good)
    for si, step in enumerate(steps):
        for ev in step.events:
            for a in arms.values():
                a.ingest(ev)
        step_row: Dict[str, Any] = {"sid": step.sid, "title": step.title, "epoch": step.epoch,
                                    "event_ids": [e.eid for e in step.events], "truth": dict(step.truth)}
        # the fabric control plane compiles an epoch-pinned register and serves the data plane from the FENCED read
        fv = fabric.verdicts()
        image = compile_register(step.epoch, fv)
        data_plane: Dict[str, str]
        alarms = 0
        if step.sid == "B4":
            image = seu_flip_to_reserved(image, CLS_P)
            naive = decode_register_naive(image)
            fenced, stats = decode_register_fenced(image, step.epoch, last_good)
            alarms = stats["reserved_codes"]
            seu_demo = {"class": CLS_P, "verdict_state": fv[CLS_P], "naive_action": naive[CLS_P],
                        "fenced_decoded": fenced[CLS_P], "fenced_action": action_of(fenced[CLS_P]),
                        "forward_leak_naive": naive[CLS_P] == VERDICT_FORWARD,
                        "forward_leak_fenced": action_of(fenced[CLS_P]) == VERDICT_FORWARD,
                        "alarms": alarms, "register_hex": image.hex()}
            data_plane = fenced
        else:
            data_plane, stats = decode_register_fenced(image, step.epoch, last_good)
            alarms = stats["reserved_codes"]
        if si == 0:
            first_verdicts = dict(fv)
        last_good = {c: data_plane[c] for c in CLASS_IDS}
        register_log.append({"sid": step.sid, "epoch": step.epoch, "alarms": alarms, "register_hex": image.hex()})
        for key in ARMS:
            arm = arms[key]
            verdicts = arm.verdicts()
            plane = data_plane if key == "chelated_fabric_control" else verdicts
            sc = sum(1 for c in CLASS_IDS if action_of(plane[c]) == step.truth[c])
            per_arm[key]["state_correct"] += sc
            per_arm[key]["state_total"] += len(CLASS_IDS)
            fc = 0
            for i in range(n_docs):
                pc = flow_pred_class(fr, key, i, sig)
                dec = action_of(plane.get(pc, VERDICT_DEFER)) if pc else VERDICT_DEFER
                if dec == step.truth[fr.doc_class[i]]:
                    fc += 1
            per_arm[key]["flow_correct"] += fc
            per_arm[key]["flow_total"] += n_docs
            per_arm[key]["steps"].append({
                "sid": step.sid,
                "verdicts": {c: action_of(plane[c]) for c in CLASS_IDS},
                "state_correct": sc, "flow_correct": fc,
                "derived_served": sorted(arm.derived_served()),
                "audit_retained": sorted(arm.audit_retained()),
            })
        out["steps"].append(step_row)
    torn_demo = torn_read_experiment(first_verdicts, fabric.verdicts(), 0, 1)
    last = steps[-1]
    invalid = set(last.invalid_parents)
    derived_parent = {ev.eid: ev.target for s in steps for ev in s.events if ev.kind == KIND_DERIVED}
    audit_parent = {ev.eid: ev.target for s in steps for ev in s.events if ev.kind == KIND_AUDIT}
    for key in ARMS:
        arm = arms[key]
        served = set(arm.derived_served())
        stale = sum(1 for e in served if derived_parent[e] in invalid)
        suppressed = sum(1 for e in derived_parent if e not in served and derived_parent[e] not in invalid)
        retained = set(arm.audit_retained())
        pa = per_arm[key]
        pa["derived_total"] = len(derived_parent)
        pa["stale_served"] = stale
        pa["wrongly_suppressed"] = suppressed
        pa["derived_hygiene_ppm"] = (len(derived_parent) - stale - suppressed) * 1_000_000 // max(1, len(derived_parent))
        pa["audit_total"] = len(audit_parent)
        pa["audit_survived"] = len(retained & set(audit_parent))
        pa["audit_invalid_parent_total"] = sum(1 for e, p in audit_parent.items() if p in invalid)
        pa["audit_invalid_parent_survived"] = sum(1 for e, p in audit_parent.items() if p in invalid and e in retained)
        pa["state_acc_ppm"] = pa["state_correct"] * 1_000_000 // pa["state_total"]
        pa["flow_acc_ppm"] = pa["flow_correct"] * 1_000_000 // pa["flow_total"]
    trace_audit = sorted(fabric.audit_in_trace())
    out["arms"] = per_arm
    out["fabric"] = {
        "echo": fabric.echo_collapse(),
        "audit_in_canonical_trace": trace_audit,
        "register_log": register_log,
        "seu": seu_demo,
        "torn": torn_demo,
        "ek5_ops": len(fabric.pipe.ops),
        "event_count": len(fabric.events),
        "pending_edges_unresolved": len(fabric.pipe.queue),
    }
    out["oracle"] = check_ops_against_oracle(fabric.pipe.ops, f"scenario{name}", random.Random(7 if name == "A" else 8))
    return out


# ---------------------------------------------------------------------------
# Part 4: evaluation, public/internal projection, receipts, TCO calculator, HTTP surface
# ---------------------------------------------------------------------------
ENGINE_FILE_SHA256_SOURCE = Path(__file__).resolve()
ENCODER_MODE_DEFAULT = "local-drycheck"
_ENCODER_MODE = {"value": ENCODER_MODE_DEFAULT}
FUZZ_TRACES = 260
KAPPA_TOL_NANO = int(KAPPA_TOL * 1e9)  # |kappa - 1| < 1e-5  ==  < 10_000 in 1e-9 units

CLASS_PUB_ID: Dict[str, str] = {c: f"class_{i + 1:02d}" for i, c in enumerate(CLASS_IDS)}
FAMILY_PUBLIC = {"raw": "raw_padded", "ridge": "linear_chain", "procrustes": "orthogonal_chain", "zeromu": "fabric_chain"}
STATE_LABEL_PUBLIC = {"ADMIT": "admitted", "AUDIT_ONLY": "audit_only", "CONTESTED": "contested"}
KEY_DISCLOSURE = "Ephemeral Ed25519 key generated at server start; it proves integrity of this server's ledger, not the identity of any party."
CLOCK_DISCLOSURE = "issued_at_us is the server wall clock in microseconds and is not an independent time source."
DISCLAIMER = ("Software emulation. No ASIC, P4 or DPU silicon and no flight hardware were used; flows and orbital elements "
              "are synthetic. Receipts attest to this emulation's state only.")
PUBLIC_DISCLOSURES: Tuple[str, ...] = (
    "Traffic is synthetic and follows the public CESNET-TLS / CIC-IDS feature schema; no captured traffic is used.",
    "Orbital elements are synthetic in the CelesTrak two-line format and are propagated two-body Keplerian, not SGP4.",
    "In the default check mode the encoders are deterministic text-derived stand-ins; real checkpoints run only with --mode run-live-encoders.",
    "Arms 3 and 4 are authored emulations (exact class-signature lookup); they are not reproductions of any named product.",
    "Arms 5 and 6 share one ledger state machine and differ only in the retrieval path that selects the matched class.",
    "The two-bit register is torn-read safe only under a trailer-last writer; other write orders can still expose mixed images.",
    "The bridge-health gate admits a warned bridge with an alarm and refuses only a degenerate anchor set; retrieval uses all four hops.",
    "Disjointness of the synthetic corpus from historical fingerprint inventories was not established (no inventory supplied).",
    "Echo clusters are inferred from record text fingerprints; no embedding-similarity threshold is used.",
)


def set_encoder_mode(mode: str) -> None:
    if mode not in ("local-drycheck", "run-live-encoders"):
        raise ValueError("unknown encoder mode")
    _ENCODER_MODE["value"] = mode


def encoder_mode() -> str:
    return _ENCODER_MODE["value"]


class FabricError(Exception):
    def __init__(self, status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.status, self.code, self.message = status, code, message


CopilotError = FabricError  # same shape as the Copilot service errors


@functools.lru_cache(maxsize=1)
def scenario_book() -> Dict[str, List[Step]]:
    return build_scenarios()


@functools.lru_cache(maxsize=4)
def get_eval(mode: str = ENCODER_MODE_DEFAULT) -> Dict[str, Any]:
    """Full evaluation: both scenarios over the real retrieval layer, plus the >=250-trace fuzz oracle sweep."""
    fr = get_retrieval(mode)
    runs = {name: run_scenario(name, fr, steps) for name, steps in scenario_book().items()}
    fuzz = run_fuzz(FUZZ_TRACES)
    merged = OracleReport()
    for r in runs.values():
        merged.merge(r["oracle"])
    scen_only = dataclasses.replace(merged, first_mismatches=list(merged.first_mismatches))
    merged.merge(fuzz)
    return {"fr": fr, "runs": runs, "fuzz": fuzz, "oracle": merged, "oracle_scenarios_only": scen_only}


def _require_scenario(name: Any) -> List[Step]:
    steps = scenario_book().get(str(name))
    if steps is None:
        raise FabricError(404, "unknown_scenario", "Unknown scenario.")
    return steps


def _require_step(steps: Sequence[Step], step: Any) -> int:
    try:
        k = int(step)
    except (TypeError, ValueError):
        raise FabricError(422, "bad_step", "step must be an integer.") from None
    if not 1 <= k <= len(steps):
        raise FabricError(422, "bad_step", "step is out of range.")
    return k


def replay_arms(name: str, k: int, *, drop_kinds: Sequence[str] = (), extra: Sequence[FEvent] = ()) -> Dict[str, Any]:
    """Cheap state-only replay (no retrieval): every arm ingests scenario events up to step ``k``."""
    steps = _require_scenario(name)
    arms = {key: _make_arm(key) for key in ARMS}
    for st in steps[:k]:
        for ev in st.events:
            if ev.kind in drop_kinds:
                continue
            for a in arms.values():
                a.ingest(ev)
    for ev in extra:
        for a in arms.values():
            a.ingest(ev)
    return arms


def _pub_verdicts(v: Mapping[str, str]) -> Dict[str, str]:
    return {CLASS_PUB_ID[c]: action_of(v.get(c, VERDICT_DEFER)) for c in CLASS_IDS}


def _class_table() -> List[Dict[str, Any]]:
    return [{"id": CLASS_PUB_ID[c], "label": CLASS_PUBLIC[c], "baseline_action": BASE_VERDICT[c]} for c in CLASS_IDS]


def _stamp(payload: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(payload)
    for k, v in HONESTY_FLAGS.items():
        out[k] = v
    out.setdefault("banner", HONESTY_BANNER)
    return out


# ---------------------------------------------------------------------------
# Bridge-health policy (admit PASS; admit WARN with alarm; refuse FAIL_CLOSED_DEGENERATE)
# ---------------------------------------------------------------------------
def bridge_policy(tra1: Mapping[str, Any]) -> Dict[str, Any]:
    chain = dict(tra1["chain"])
    admitted = sorted(k for k, v in chain.items() if v != "FAIL_CLOSED_DEGENERATE")
    alarms = sorted(k for k, v in chain.items() if v == "WARN_CONTAMINATED")
    refused = sorted(k for k, v in chain.items() if v == "FAIL_CLOSED_DEGENERATE")
    return {
        "chain_status": chain,
        "admitted": admitted,
        "alarms": alarms,
        "refused": refused,
        "degenerate_anchor_status": tra1["degenerate_anchor_status"],
        "degenerate_anchor_refused": bool(tra1["degenerate_anchor_refused"]),
        "policy": "admit PASS; admit WARN_CONTAMINATED with an alarm; refuse FAIL_CLOSED_DEGENERATE",
    }


# ---------------------------------------------------------------------------
# Scoreboard / retrieval table / oracle report projection
# ---------------------------------------------------------------------------
def _arm_metrics(pa: Mapping[str, Any]) -> Dict[str, Any]:
    keys = ("state_acc_ppm", "flow_acc_ppm", "derived_hygiene_ppm", "stale_served", "wrongly_suppressed", "audit_total",
            "audit_survived", "audit_invalid_parent_total", "audit_invalid_parent_survived")
    return {k: int(pa[k]) for k in keys}


def _combined(a: Mapping[str, Any], b: Mapping[str, Any]) -> Dict[str, Any]:
    state = (a["state_correct"] + b["state_correct"]) * 1_000_000 // (a["state_total"] + b["state_total"])
    flow = (a["flow_correct"] + b["flow_correct"]) * 1_000_000 // (a["flow_total"] + b["flow_total"])
    return {"state_acc_ppm": int(state), "flow_acc_ppm": int(flow)}


def retrieval_report(fr: FleetRetrieval, mode: str) -> Dict[str, Any]:
    rows = []
    for hop in (1, 2, 3, 4):
        row: Dict[str, Any] = {"hop": hop, "source_encoder": f"enc_{'abcde'[4 - hop]}",
                               "source_label": ENCODER_PUBLIC_LABEL[f"enc_{'abcde'[4 - hop]}"], "families": {}}
        for fam in ("raw", "ridge", "procrustes", "zeromu"):
            r = fr.hops[fam][hop]
            entry = {"recall1_ppm": int(r.recall1), "recall5_ppm": int(r.recall5), "class_acc_ppm": int(r.class_acc),
                     "kappa_dev_nano": int(r.kappa_ppm_dev),
                     "kappa_within_tol": bool(r.kappa_ppm_dev < KAPPA_TOL_NANO) if fam != "raw" else None}
            row["families"][FAMILY_PUBLIC[fam]] = entry
        rows.append(row)
    kappa_ok = {FAMILY_PUBLIC[f]: all(fr.hops[f][h].kappa_ppm_dev < KAPPA_TOL_NANO for h in (1, 2, 3, 4))
                for f in ("ridge", "procrustes", "zeromu")}
    out: Dict[str, Any] = {
        "ground_dim": NATIVE_DIM,
        "doc_count": len(fr.doc_class),
        "kappa_tolerance_nano": KAPPA_TOL_NANO,
        "kappa_within_tol_all_hops": kappa_ok,
        "native_ceiling": {k: int(v) for k, v in fr.native_ceiling.items()},
        "per_hop": rows,
        "encoder_status": {ENCODER_PUBLIC_LABEL[k]: ("live" if v.startswith("live:") else "stand_in")
                           for k, v in fr.enc.status.items()},
        "sidecar": {h: {"bytes_per_flow": int(s["bytes_per_flow"]), "total_bytes": int(s["total_bytes"]),
                        "max_abs_parity_ppm": int(s["max_abs_parity_ppm"]), "argmax_identical": bool(s["argmax_identical"]),
                        "direct_harness_parity_ppm": int(s["direct_harness_parity_ppm"])} for h, s in fr.sidecar.items()},
        "bridge_health": bridge_policy(fr.tra1),
        "burn_check": {"checked": bool(fr.burn_check.get("checked")), "burned": bool(fr.burn_check.get("burned")),
                       "disjointness_established": False},
    }
    if mode == "nda":
        out["internal"] = {"burn_detail": fr.burn_check.get("detail", ""), "tra1_raw": fr.tra1,
                           "encoders": {eid: {"checkpoint": nm, "dim": d, "status": fr.enc.status[eid]}
                                        for (eid, nm, d) in ENCODER_CHAIN}}
    return out


def oracle_report_dict(rep: OracleReport) -> Dict[str, Any]:
    return {
        "traces": int(rep.traces), "prefixes": int(rep.prefixes), "oracle_disagreements": int(rep.disagreements),
        "register_checks": int(rep.register_checks), "register_disagreements": int(rep.register_disagreements),
        "reserved_code_injected": int(rep.reserved_injected),
        "reserved_forward_leaks_fenced": int(rep.reserved_forward_leaks_fenced),
        "reserved_forward_leaks_naive": int(rep.reserved_forward_leaks_naive),
        "torn_injected": int(rep.torn_injected),
        "shared_decision_code_with_engine": False,
    }


def scoreboard(mode: str = "non_nda") -> Dict[str, Any]:
    ev = get_eval(encoder_mode())
    fr: FleetRetrieval = ev["fr"]
    runs = ev["runs"]
    arms_out: List[Dict[str, Any]] = []
    for key in ARMS:
        pid = ARM_PUBLIC_ID[key]
        a, b = runs["A"]["arms"][key], runs["B"]["arms"][key]
        row: Dict[str, Any] = {
            "id": pid, "label": ARM_PUBLIC_LABEL[pid], "note": ARM_PUBLIC_NOTE[pid],
            "emulation_ours": key in SIGNATURE_ARMS,
            "scenario_a": _arm_metrics(a), "scenario_b": _arm_metrics(b), "combined": _combined(a, b),
        }
        if mode == "nda":
            row["internal"] = {"technique": ARM_INTERNAL[key], "internal_id": key}
        arms_out.append(row)
    fab_a, fab_b = runs["A"]["fabric"], runs["B"]["fabric"]
    seu = fab_b["seu"]
    torn = fab_a["torn"]
    payload: Dict[str, Any] = {
        "schema": "fabric_scoreboard_v1", "mode": mode, "encoder_mode": fr.mode,
        "arms": arms_out,
        "retrieval": retrieval_report(fr, mode),
        "oracle": oracle_report_dict(ev["oracle"]),
        "oracle_scenario_a_steps": len(runs["A"]["steps"]), "oracle_scenario_b_steps": len(runs["B"]["steps"]),
        "oracle_fuzz_traces": int(ev["fuzz"].traces),
        "echo_collapse": {"scenario_a": fab_a["echo"], "scenario_b": fab_b["echo"]},
        "register": {
            "encoding": {"FORWARD": "01", "DROP": "00", "DEFER_INSPECT": "10", "INVALID_FAIL_CLOSED": "11 (reserved)"},
            "reserved_code_demo": {
                "class": CLASS_PUB_ID[seu["class"]], "naive_action": seu["naive_action"], "fenced_action": seu["fenced_action"],
                "forward_leak_naive": bool(seu["forward_leak_naive"]), "forward_leak_fenced": bool(seu["forward_leak_fenced"]),
                "alarms": int(seu["alarms"]),
            },
            "torn_read_demo": {k: (int(v) if isinstance(v, (int, bool)) and not isinstance(v, bool) else v)
                               for k, v in torn.items() if k in ("write_positions", "write_orders", "naive_mixed_reads",
                                                                  "fenced_mixed_reads", "fenced_mixed_reads_noncontract_orders")},
            "writer_contract": "Fenced reads are exact only when the epoch trailer is written last.",
        },
        "audit_survival": {"scenario_a": sorted(fab_a["audit_in_canonical_trace"]),
                           "scenario_b": sorted(fab_b["audit_in_canonical_trace"])},
        "signature_collisions": {"scenario_a": int(runs["A"]["signature_collisions"])},
        "disclosures": list(PUBLIC_DISCLOSURES),
        "classes": _class_table(),
    }
    if mode == "nda":
        payload["internal"] = {"fuzz": dataclasses.asdict(ev["fuzz"]),
                               "first_mismatches": list(ev["oracle"].first_mismatches),
                               "ek5_ops": {"A": fab_a["ek5_ops"], "B": fab_b["ek5_ops"]},
                               "pending_edges_unresolved": {"A": fab_a["pending_edges_unresolved"],
                                                            "B": fab_b["pending_edges_unresolved"]}}
    return _stamp(payload)


# ---------------------------------------------------------------------------
# Per-step views, custom query, inject, undo
# ---------------------------------------------------------------------------
def evaluate_step(name: str, step: Any, mode: str, custom_query: Optional[str] = None) -> Dict[str, Any]:
    steps = _require_scenario(name)
    k = _require_step(steps, step)
    ev = get_eval(encoder_mode())
    run = ev["runs"][name]
    srow = run["steps"][k - 1]
    arms_out = []
    for key in ARMS:
        rec = run["arms"][key]["steps"][k - 1]
        pid = ARM_PUBLIC_ID[key]
        item: Dict[str, Any] = {
            "id": pid, "label": ARM_PUBLIC_LABEL[pid], "verdicts": _pub_verdicts(rec["verdicts"]),
            "state_correct": int(rec["state_correct"]), "state_total": len(CLASS_IDS),
            "flow_correct": int(rec["flow_correct"]), "flow_total": len(ev["fr"].doc_class),
            "derived_served": list(rec["derived_served"]), "audit_retained": list(rec["audit_retained"]),
        }
        arms_out.append(item)
    reg = run["fabric"]["register_log"][k - 1]
    payload: Dict[str, Any] = {
        "schema": "fabric_step_v1", "mode": mode, "scenario": name, "step": k, "step_count": len(steps),
        "sid": f"step_{k}", "title": srow["title"], "epoch": int(srow["epoch"]), "event_ids": list(srow["event_ids"]),
        "truth": _pub_verdicts(srow["truth"]),
        "invalid_parents": sorted(steps[k - 1].invalid_parents),
        "arms": arms_out,
        "register": {"image_hex": reg["register_hex"], "alarms": int(reg["alarms"]), "epoch": int(reg["epoch"])},
        "classes": _class_table(),
    }
    if custom_query is not None:
        payload["query"] = _custom_query(ev["fr"], run, k, custom_query)
    if mode == "nda":
        arms_replay = replay_arms(name, k)
        fab: FabricArm = arms_replay["chelated_fabric_control"]
        payload["internal"] = {
            "eligibility_raw": {e: fab.store.eligibility[e] for e in sorted(fab.events)},
            "effective_authority": {e: int(fab.store.eff_auth[e]) for e in sorted(fab.events)},
            "edge_ops": [list(o[:4]) for o in fab.pipe.ops if o and o[0] == "edge"][:200],
        }
    return payload


def _fabric_scores(fr: FleetRetrieval, q: np.ndarray, hop: int = 4) -> np.ndarray:
    zc = fr.chain("zeromu", hop)
    mu_x, mu_y, mat, _ = zc.linear_components()
    dn = l2_normalize(fr.docs)
    mu = mu_y.reshape(-1)
    den = np.sqrt(np.maximum(1.0 + float(mu @ mu) - 2.0 * (dn @ mu), 1e-12))
    qh = l2_normalize((q - mu_x) @ mat)
    return (qh @ dn.T - (qh @ mu)[:, None]) / den[None, :]


def _custom_query(fr: FleetRetrieval, run: Mapping[str, Any], k: int, text: str) -> Dict[str, Any]:
    if not text or len(text) > 2000:
        raise FabricError(422, "bad_query", "text must be 1..2000 characters.")
    q = fr.enc.encode([text], 0)
    scores = _fabric_scores(fr, q, 4)
    order = np.argsort(-scores[0])[:5]
    plane = run["arms"]["chelated_fabric_control"]["steps"][k - 1]["verdicts"]
    top = []
    for rank, di in enumerate(order, start=1):
        c = fr.doc_class[int(di)]
        top.append({"rank": rank, "class": CLASS_PUB_ID[c], "label": CLASS_PUBLIC[c],
                    "tranche": int(fr.doc_tranche[int(di)]), "action": plane[c]})
    c1 = fr.doc_class[int(order[0])]
    return {"text_length": len(text), "source_encoder": ENCODER_PUBLIC_LABEL["enc_a"], "top5": top,
            "decision": plane[c1], "decision_class": CLASS_PUB_ID[c1],
            "scoring": "bridged cosine through the full chain, without score-normalisation references (single query)"}


def inject_evidence(data: Mapping[str, Any], mode: str) -> Dict[str, Any]:
    name = str(data.get("scenario", ""))
    steps = _require_scenario(name)
    k = _require_step(steps, data.get("step", 1))
    cid = {v: c for c, v in CLASS_PUB_ID.items()}.get(str(data.get("class", "")))
    if cid is None:
        raise FabricError(422, "bad_class", "class must be one of the public class ids.")
    claim = str(data.get("claim", ""))
    if claim not in VERDICTS:
        raise FabricError(422, "bad_claim", "claim must be FORWARD, DROP or DEFER_INSPECT.")
    try:
        tier = int(data.get("tier", 1))
        tranche = int(data.get("tranche", 1))
    except (TypeError, ValueError):
        raise FabricError(422, "bad_value", "tier and tranche must be integers.") from None
    if not 1 <= tier <= 3 or not 0 <= tranche <= 2:
        raise FabricError(422, "bad_value", "tier must be 1..3 and tranche 0..2.")
    base = replay_arms(name, k)
    before = {key: _pub_verdicts(a.verdicts()) for key, a in base.items()}
    inj = FEvent(KIND_RULING if bool(data.get("ruling")) else KIND_RULE, "inj_1", cid, claim, tier, tranche,
                 "", f"injected evidence for {cid}")
    after_arms = replay_arms(name, k, extra=[inj])
    after = {key: _pub_verdicts(a.verdicts()) for key, a in after_arms.items()}
    changed = {ARM_PUBLIC_ID[key]: sorted(c for c in before[key] if before[key][c] != after[key][c]) for key in ARMS}
    fab: FabricArm = after_arms["chelated_fabric_control"]
    return {
        "schema": "fabric_inject_v1", "mode": mode, "scenario": name, "step": k,
        "injected": {"class": CLASS_PUB_ID[cid], "claim": claim, "tier": tier, "tranche": tranche},
        "verdicts_after": {ARM_PUBLIC_ID[key]: after[key] for key in ARMS},
        "changed_classes": changed,
        "register_image_hex": compile_register(steps[k - 1].epoch, fab.verdicts()).hex(),
        "note": "State-only replay; the injected record is not part of the authored ground truth.",
    }


def undo_retraction(name: str, step: Any, mode: str) -> Dict[str, Any]:
    steps = _require_scenario(name)
    k = _require_step(steps, step)
    rets = [e for st in steps[:k] for e in st.events if e.kind == KIND_RETRACT]
    if not rets:
        raise FabricError(422, "no_retraction", "No retraction has been applied up to this step.")
    withR = replay_arms(name, k)
    without = replay_arms(name, k, drop_kinds=(KIND_RETRACT,))
    out: Dict[str, Any] = {"schema": "fabric_undo_v1", "mode": mode, "scenario": name, "step": k,
                           "retractions_removed": [e.eid for e in rets], "arms": []}
    for key in ARMS:
        a, b = withR[key], without[key]
        va, vb = _pub_verdicts(a.verdicts()), _pub_verdicts(b.verdicts())
        out["arms"].append({
            "id": ARM_PUBLIC_ID[key], "with_retraction": va, "without_retraction": vb,
            "changed_classes": sorted(c for c in va if va[c] != vb[c]),
            "derived_served_with": sorted(a.derived_served()), "derived_served_without": sorted(b.derived_served()),
        })
    return out


# ---------------------------------------------------------------------------
# Parameterized TCO / OISL calculator (MEASURED vs SENSITIVITY_ASSUMPTION)
# ---------------------------------------------------------------------------
MEASURED = "MEASURED"
ASSUMED = "SENSITIVITY_ASSUMPTION"
TCO_INPUTS: Dict[str, Tuple[int, int, int, str]] = {   # name -> (default, lo, hi, unit)
    "n_flows": (1_000_000, 1, 10_000_000_000, "flows"),
    "oisl_rate_mbps": (1000, 1, 1_000_000, "Mbit/s"),
    "link_efficiency_ppm": (700_000, 1, 1_000_000, "ppm"),
    "vector_bytes_per_dim": (2, 1, 8, "bytes"),
    "downlink_cents_per_gb": (2000, 0, 10_000_000, "cents/GB"),
}


def tco_calc(params: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    vals: Dict[str, int] = {}
    for name, (dflt, lo, hi, _unit) in TCO_INPUTS.items():
        raw = (params or {}).get(name)
        if raw in (None, ""):
            vals[name] = dflt
            continue
        try:
            v = int(str(raw))
        except ValueError:
            raise FabricError(422, "bad_param", f"{name} must be an integer.") from None
        if not lo <= v <= hi:
            raise FabricError(422, "bad_param", f"{name} must be within {lo}..{hi}.")
        vals[name] = v
    fr = get_retrieval(encoder_mode())
    side_bpf = int(fr.sidecar["1"]["bytes_per_flow"])
    sched = build_contact_schedule()
    oisl_s = {k: int(sum(b - a for a, b in v)) for k, v in sched["oisl_windows"].items()}
    contact_s = int(min(oisl_s.values())) if oisl_s else 0
    n = vals["n_flows"]
    side_bytes = n * side_bpf
    full_bytes = n * NATIVE_DIM * vals["vector_bytes_per_dim"]
    eff_bps = vals["oisl_rate_mbps"] * 1_000_000 * vals["link_efficiency_ppm"] // 1_000_000
    xfer_side_ms = (side_bytes * 8 * 1000 + eff_bps - 1) // eff_bps
    xfer_full_ms = (full_bytes * 8 * 1000 + eff_bps - 1) // eff_bps
    gb = 1_000_000_000
    cost_side = (side_bytes * vals["downlink_cents_per_gb"] + gb - 1) // gb
    cost_full = (full_bytes * vals["downlink_cents_per_gb"] + gb - 1) // gb

    def out(value: Any, unit: str, label: str, basis: str) -> Dict[str, Any]:
        return {"value": value, "unit": unit, "label": label, "basis": basis}

    outputs = {
        "sidecar_bytes_per_flow": out(side_bpf, "bytes", MEASURED, "read from the built sidecar blob"),
        "sidecar_total_bytes": out(side_bytes, "bytes", ASSUMED, "n_flows is an assumption"),
        "full_vector_total_bytes": out(full_bytes, "bytes", ASSUMED, "n_flows and bytes per dimension are assumptions"),
        "reduction_x1000": out(full_bytes * 1000 // max(1, side_bytes), "x1000", ASSUMED, "depends on bytes per dimension"),
        "oisl_contact_seconds_min_pair": out(contact_s, "s", MEASURED, "synthetic schedule, two-body propagation, 6 h horizon"),
        "oisl_transfer_ms_sidecar": out(int(xfer_side_ms), "ms", ASSUMED, "link rate and efficiency are assumptions"),
        "oisl_transfer_ms_full_vectors": out(int(xfer_full_ms), "ms", ASSUMED, "link rate and efficiency are assumptions"),
        "sidecar_fits_in_shortest_pair_contact": out(bool(xfer_side_ms <= contact_s * 1000), "bool", ASSUMED,
                                                     "compares an assumed transfer time with a measured contact total"),
        "full_vectors_fit_in_shortest_pair_contact": out(bool(xfer_full_ms <= contact_s * 1000), "bool", ASSUMED,
                                                         "compares an assumed transfer time with a measured contact total"),
        "downlink_cost_cents_sidecar": out(int(cost_side), "cents", ASSUMED, "price per GB is an assumption"),
        "downlink_cost_cents_full_vectors": out(int(cost_full), "cents", ASSUMED, "price per GB is an assumption"),
    }
    return {
        "schema": "fabric_tco_v1",
        "inputs": {k: {"value": vals[k], "unit": TCO_INPUTS[k][3], "label": ASSUMED} for k in TCO_INPUTS},
        "outputs": outputs,
        "oisl_contact_seconds": oisl_s,
        "note": "Arithmetic only. Every input is a user-adjustable sensitivity assumption; nothing here is a price quote or a link budget.",
    }


def schedule_report(mode: str) -> Dict[str, Any]:
    s = build_contact_schedule()
    out = {
        "schema": "fabric_schedule_v1", "mode": mode, "tle_lines": s["tle_lines"], "propagator": s["propagator"],
        "step_s": int(s["step_s"]), "horizon_s": int(s["horizon_s"]), "stations": list(s["stations"]),
        "ground_windows": s["ground_windows"], "oisl_windows": s["oisl_windows"],
        "oceanic_blackout_windows": s["oceanic_blackout_windows"], "summary": s["summary"],
        "schedule_sha256": s["schedule_sha256"],
        "dtn_delay_s_at_t0": {str(t): int(dtn_delay_s(t, 0)) for t in range(3)},
        "synthetic": True,
    }
    return json.loads(json.dumps(out))


def warm_benchmark(repeats: int = 30) -> Dict[str, Any]:
    if not 1 <= repeats <= 200:
        raise FabricError(422, "bad_repeats", "repeats must be 1..200.")
    get_eval(encoder_mode())
    steps = scenario_book()["A"]
    samples = {"state_replay_us": [], "register_compile_decode_us": []}
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        arms = replay_arms("A", len(steps))
        t1 = time.perf_counter_ns()
        fv = arms["chelated_fabric_control"].verdicts()
        img = compile_register(1, fv)
        decode_register_fenced(img, 1, {c: VERDICT_DEFER for c in CLASS_IDS})
        t2 = time.perf_counter_ns()
        samples["state_replay_us"].append((t1 - t0) // 1000)
        samples["register_compile_decode_us"].append((t2 - t1) // 1000)

    def q(v: List[int], pct: int) -> int:
        s = sorted(v)
        return int(s[min(len(s) - 1, (pct * (len(s) - 1) + 50) // 100)])

    return {"schema": "fabric_benchmark_v1", "repeats": repeats, "label": MEASURED,
            "host": "software emulation on the build host; not switch or DPU latency",
            "metrics": {k: {"p50_us": q(v, 50), "p95_us": q(v, 95), "max_us": int(max(v))} for k, v in samples.items()}}


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


class FabricService:
    """Process-level state: the receipt ledger, the ephemeral signing key and the NDA gate."""

    def __init__(self, *, nda_token: Optional[str] = None, clock_us: Optional[Callable[[], int]] = None,
                 private_key: Optional[Ed25519PrivateKey] = None) -> None:
        self.nda_token = nda_token if nda_token else None
        self._clock_us = clock_us or (lambda: time.time_ns() // 1000)
        self._key = private_key or Ed25519PrivateKey.generate()
        raw = self._key.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
        self.public_key_b64 = base64.b64encode(raw).decode("ascii")
        self.key_id = hashlib.sha256(raw).hexdigest()[:16]
        self._lock = threading.Lock()
        self.ledger: List[Dict[str, Any]] = []

    def check_mode(self, requested: Any, headers: Mapping[str, str]) -> str:
        mode = "non_nda" if requested in (None, "") else str(requested)
        if mode not in MODES:
            raise FabricError(422, "bad_mode", "Unknown mode.")
        if mode == "nda":
            supplied = headers.get(NDA_TOKEN_HEADER, "")
            if not self.nda_token or not hmac.compare_digest(supplied.encode("utf-8"), self.nda_token.encode("utf-8")):
                raise FabricError(403, "mode_unavailable", "This mode is not available for this request.")
        return mode

    def issue_receipt(self, scenario_id: str, step: Any, mode: str) -> Dict[str, Any]:
        steps = _require_scenario(scenario_id)
        k = _require_step(steps, step)
        spec = steps[k - 1]
        arms = replay_arms(scenario_id, k)
        fab: FabricArm = arms["chelated_fabric_control"]
        verdicts = fab.verdicts()
        image = compile_register(spec.epoch, verdicts)
        trace = export_canonical_ek5_audit_trace(fab.store)
        rep = check_ops_against_oracle(fab.pipe.ops, f"receipt{scenario_id}{k}", random.Random(k), register_probe=True)
        truth_ok = sum(1 for c in CLASS_IDS if action_of(verdicts[c]) == spec.truth[c])
        payload: Dict[str, Any] = {
            "receipt_kind": "fabric_register_receipt", "schema": "fabric_receipt_v1", "mode": mode,
            "scenario_id": scenario_id, "step": k, "step_title": spec.title, "epoch": int(spec.epoch),
            "register_hex": image.hex(),
            "register_codes": {CLASS_PUB_ID[c]: REG_CODE[verdicts[c]] for c in CLASS_IDS},
            "truth_match": {"correct": truth_ok, "total": len(CLASS_IDS)},
            "derived_served": sorted(fab.derived_served()),
            "audit_in_canonical_trace": sorted(fab.audit_in_trace()),
            "canonical_trace_sha256": str(trace.digest()),
            "oracle_check": {"prefixes": int(rep.prefixes), "disagreements": int(rep.disagreements)},
            "schedule_sha256": build_contact_schedule()["schedule_sha256"],
            "decision_engine_sha256": engine_file_sha256(),
            "hardware_asic_p4_silicon_in_loop_claimed": False,
            "orbital_flight_hardware_claimed": False,
            "key_id": self.key_id, "key_disclosure": KEY_DISCLOSURE, "clock_disclosure": CLOCK_DISCLOSURE,
            "disclaimer": DISCLAIMER,
        }
        if mode == "nda":
            payload["internal"] = {"eligibility_raw": {e: fab.store.eligibility[e] for e in sorted(fab.events)}}
        with self._lock:
            prev = self.ledger[-1]["entry_hash"] if self.ledger else GENESIS_HASH
            payload["seq"] = len(self.ledger) + 1
            payload["prev_hash"] = prev
            payload["issued_at_us"] = int(self._clock_us())
            entry_hash = hashlib.sha256(prev.encode("ascii") + canonical_json(payload)).hexdigest()
            signature = self._key.sign(entry_hash.encode("ascii"))
            entry = {"payload": payload, "prev_hash": prev, "entry_hash": entry_hash,
                     "signature_b64": base64.b64encode(signature).decode("ascii"),
                     "public_key_b64": self.public_key_b64,
                     "algorithm": "Ed25519 over the ASCII hex of SHA-256(prev_hash || canonical_json(payload))"}
            self.ledger.append(entry)
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
            if isinstance(payload, dict) and isinstance(payload.get("seq"), int) and 1 <= payload["seq"] <= len(self.ledger):
                in_ledger = self.ledger[payload["seq"] - 1]["entry_hash"] == claimed_hash
        return {"valid": all(checks.values()), "checks": checks, "in_ledger": in_ledger, "reasons": reasons}

    def verify_ledger(self) -> Dict[str, Any]:
        with self._lock:
            entries = list(self.ledger)
        prev = GENESIS_HASH
        for i, e in enumerate(entries, start=1):
            res = self.verify_receipt(e)
            if not res["valid"] or e["prev_hash"] != prev or e["payload"]["seq"] != i:
                return {"intact": False, "length": len(entries), "first_bad_seq": i}
            prev = e["entry_hash"]
        return {"intact": True, "length": len(entries), "head_hash": prev}


_DEFAULT_SERVICE: Optional[FabricService] = None
_DEFAULT_LOCK = threading.Lock()


def get_service() -> FabricService:
    global _DEFAULT_SERVICE
    with _DEFAULT_LOCK:
        if _DEFAULT_SERVICE is None:
            _DEFAULT_SERVICE = FabricService(nda_token=os.environ.get("CHELATED_COPILOT_NDA_TOKEN"))
        return _DEFAULT_SERVICE


def set_service(service: Optional[FabricService]) -> None:
    global _DEFAULT_SERVICE
    with _DEFAULT_LOCK:
        _DEFAULT_SERVICE = service


# ---------------------------------------------------------------------------
# HTTP dispatch (pure functions; no sockets needed for tests)
# ---------------------------------------------------------------------------
Response = Tuple[int, str, bytes]
_JSON = "application/json; charset=utf-8"


def is_fabric_path(path: str) -> bool:
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
    text = json.dumps(_stamp(dict(payload)), indent=2, sort_keys=True, ensure_ascii=True)
    if mode == "non_nda":
        try:
            assert_non_nda_clean(text, "response")
        except ValueError:
            return 500, _JSON, json.dumps(_stamp({"error": "projection_failed_closed", "message": "Response withheld."})).encode("utf-8")
    return status, _JSON, text.encode("utf-8")


def _error_response(err: FabricError) -> Response:
    return _json_response(err.status, {"error": err.code, "message": err.message})


_HTML_SCAN_CACHE: Dict[Tuple[int, int], bool] = {}


def _serve_page() -> Response:
    try:
        data = HTML_PATH.read_bytes()
        st = HTML_PATH.stat()
    except OSError:
        return _error_response(FabricError(404, "page_missing", "The page is not installed."))
    key = (st.st_mtime_ns, st.st_size)
    clean = _HTML_SCAN_CACHE.get(key)
    if clean is None:
        clean = not scan_text_for_ip_leaks(data.decode("utf-8"), "page")
        _HTML_SCAN_CACHE[key] = clean
    if not clean:
        return _error_response(FabricError(500, "projection_failed_closed", "Response withheld."))
    return 200, "text/html; charset=utf-8", data


def _public_scenarios() -> Dict[str, Any]:
    book = scenario_book()
    return {"schema": "fabric_scenarios_v1", "classes": _class_table(), "scenarios": [
        {"id": name, "title": "Conflicting authority, late echo and epoch roll" if name == "A"
         else "Early retraction, reserved register code and cycle-only support",
         "steps": [{"index": i, "sid": f"step_{i}", "title": st.title, "epoch": st.epoch,
                    "has_retraction": any(e.kind == KIND_RETRACT for e in st.events)}
                   for i, st in enumerate(steps, start=1)]}
        for name, steps in book.items()]}


def handle_fabric_get(path: str, query_params: Optional[Mapping[str, Any]] = None,
                      headers: Optional[Mapping[str, Any]] = None, service: Optional[FabricService] = None) -> Response:
    svc = service or get_service()
    hdr = _norm_headers(headers)
    clean = urlparse(path).path
    mode = "non_nda"
    try:
        if clean in PAGE_PATHS:
            return _serve_page()
        if not (clean == API_PREFIX or clean.startswith(API_PREFIX + "/")):
            raise FabricError(404, "not_found", "Not found.")
        sub = clean[len(API_PREFIX):].rstrip("/")
        if sub in ("", "/health"):
            return _json_response(200, {"status": "ok", "service": "network-orbital-fabric", "key_id": svc.key_id,
                                        "nda_mode_configured": bool(svc.nda_token), "bundle_id": BUNDLE_ID})
        mode = svc.check_mode(_qp(query_params, "mode"), hdr)
        if sub == "/scenarios":
            return _json_response(200, {**_public_scenarios(), "mode": mode}, mode)
        if sub == "/step":
            return _json_response(200, evaluate_step(str(_qp(query_params, "scenario", "")), _qp(query_params, "step", "1"), mode), mode)
        if sub == "/scoreboard":
            return _json_response(200, scoreboard(mode), mode)
        if sub == "/schedule":
            return _json_response(200, schedule_report(mode), mode)
        if sub == "/tco":
            tco = tco_calc({k: _qp(query_params, k) for k in TCO_INPUTS})
            return _json_response(200, {**tco, "mode": mode}, mode)
        if sub == "/pubkey":
            return _json_response(200, {"algorithm": "Ed25519", "public_key_b64": svc.public_key_b64, "key_id": svc.key_id,
                                        "disclosure": KEY_DISCLOSURE}, mode)
        if sub == "/ledger":
            return _json_response(200, {"entries": svc.ledger, "chain": svc.verify_ledger()}, mode)
        if sub == "/benchmark":
            try:
                reps = int(_qp(query_params, "repeats", "30") or "30")
            except ValueError:
                raise FabricError(422, "bad_repeats", "repeats must be an integer.") from None
            return _json_response(200, warm_benchmark(reps), mode)
        raise FabricError(404, "not_found", "Not found.")
    except FabricError as err:
        return _error_response(err)


def handle_fabric_post(path: str, query_params: Optional[Mapping[str, Any]] = None,
                       headers: Optional[Mapping[str, Any]] = None, body: bytes = b"",
                       service: Optional[FabricService] = None) -> Response:
    svc = service or get_service()
    hdr = _norm_headers(headers)
    clean = urlparse(path).path
    try:
        if not (clean == API_PREFIX or clean.startswith(API_PREFIX + "/")):
            raise FabricError(404, "not_found", "Not found.")
        if body is not None and len(body) > MAX_BODY_BYTES:
            raise FabricError(413, "too_large", "Request body too large.")
        try:
            data = json.loads(body.decode("utf-8")) if body else {}
        except (UnicodeDecodeError, ValueError):
            raise FabricError(400, "invalid_json", "Request body is not valid JSON.") from None
        if not isinstance(data, dict):
            raise FabricError(400, "invalid_json", "Request body must be a JSON object.")
        sub = clean[len(API_PREFIX):].rstrip("/")
        mode = svc.check_mode(data.get("mode", _qp(query_params, "mode")), hdr)
        if sub == "/query":
            return _json_response(200, evaluate_step(str(data.get("scenario", "")), data.get("step", 1), mode,
                                                     custom_query=str(data.get("text", ""))), mode)
        if sub == "/inject":
            return _json_response(200, inject_evidence(data, mode), mode)
        if sub == "/undo":
            return _json_response(200, undo_retraction(str(data.get("scenario", "")), data.get("step", 1), mode), mode)
        if sub == "/receipt":
            return _json_response(200, svc.issue_receipt(str(data.get("scenario", "")), data.get("step", 1), mode), mode)
        if sub == "/verify-receipt":
            return _json_response(200, svc.verify_receipt(data.get("receipt", data)), mode)
        raise FabricError(404, "not_found", "Not found.")
    except FabricError as err:
        return _error_response(err)


# ---------------------------------------------------------------------------
# Standalone server (same-origin only: no CORS headers are ever emitted)
# ---------------------------------------------------------------------------
def make_handler(service: FabricService) -> type:
    class FabricRequestHandler(BaseHTTPRequestHandler):
        server_version = "NetworkOrbitalFabric/1.0"

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
                self.send_header("Location", "/fabric")
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
            self._send(*handle_fabric_get(url.path, parse_qs(url.query), dict(self.headers.items()), service))

        def do_POST(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            try:
                length = int(self.headers.get("Content-Length", "0") or "0")
            except ValueError:
                length = -1
            if length < 0 or length > MAX_BODY_BYTES:
                self._send(*_error_response(FabricError(413, "too_large", "Request body too large.")))
                return
            body = self.rfile.read(length) if length else b""
            self._send(*handle_fabric_post(url.path, parse_qs(url.query), dict(self.headers.items()), body, service))

        def do_OPTIONS(self) -> None:  # noqa: N802
            self._send(*_error_response(FabricError(405, "same_origin_only", "Cross-origin requests are not supported.")))

        def log_message(self, fmt: str, *args: Any) -> None:  # silence default stderr access log
            return

    return FabricRequestHandler


def build_server(host: str, port: int, service: Optional[FabricService] = None) -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), make_handler(service or get_service()))


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Run the standalone Network & Orbital Fabric server.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8791)
    parser.add_argument("--mode", choices=("local-drycheck", "run-live-encoders"), default=ENCODER_MODE_DEFAULT)
    args = parser.parse_args(argv)
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    set_encoder_mode(args.mode)
    srv = build_server(args.host, args.port)
    print(f"Fabric control plane listening on http://{args.host}:{srv.server_address[1]}/fabric "
          f"(encoders={args.mode}; key_id={get_service().key_id}; "
          f"nda mode {'enabled' if get_service().nda_token else 'disabled'})", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        srv.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
