"""Wave 18 / Bundle #50 (Item 3A): Canonical gBench LOFT SciFact Pilot + Disjoint Zero-Write Exact Tango NI.

Implements the locked Post-v7 R2 protocol for Item 3A (#50):
  1. Strict separation between the canonical `gBench` (`d3e42831...`) `LOFT SciFact` pilot
     (`357` documents, `100` queries) and the derived zero-write migration pilot evaluating
     `Rank1QueryResidual` (`u(v^T q)`) and `ZeroMuPolarBridge` from `bridge_merge_harness`.
  2. Exact sample-size / power-sizing from paired discordance variance (`N >= 1,000` paired queries).
  3. Mechanical `burn_ledger` disjointness enforcement (`assert_corpus_unburned`) failing closed on
     historical burned corpora (`msmarco`, `quora`, `fiqa`, `scidocs`, `arguana`, `scifact`,
     `nfcorpus`, `cosqa`, `longmemeval`, `locomo`, `osv_cve_seeds`) and exact SHA-256 text overlaps.
  4. One-shot exact Tango (1998) asymptotic score 95% LCB vs `-0.0500` on the disjoint corpus:
     - Emits `"status": "underpowered"` if a candidate disjoint pool lacks required power rather than
       ever loosening `-0.0500`.
     - Preserves the historical `PM-2` `-0.0572` Tango LCB strictly untouched.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import NormalDist
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import numpy as np

from chelated_pv7_spine.bridge_merge_harness import (
    CSLSMerge,
    CenteredProcrustes,
    Rank1QueryResidual,
    RawCosineMerge,
    UncenteredProcrustes,
    ZeroMuPolarBridge,
    evaluate_score_matrix,
    l2_normalize,
)
from chelated_pv7_spine.burn_ledger import (
    BURNED_CORPORA,
    BurnLedger,
    BurnedCorpusError,
    assert_corpus_unburned,
    build_burn_ledger,
    normalize_dataset_id,
    text_fingerprint,
)
from chelated_pv7_spine.claim_registry import claim_registry_sha256
from chelated_pv7_spine.tra1_weight_sha_guard import (
    EMPTY_ADAPTER_SHA256,
    TRA1StateIdentityContract,
    sha256_digest,
)

BUNDLE_ID = 50
ITEM_ID = "3A"
SPINE_DIGEST = "32693f54e0ce7ad3d401ceb752ae4723b0eee251f6d49b95cdfb26a6201971e9"
GBENCH_CANONICAL_COMMIT = "d3e42831"
LOFT_SCIFACT_DOC_COUNT = 357
LOFT_SCIFACT_QUERY_COUNT = 100

PREREGISTERED_NI_MARGIN = -0.0500
HISTORICAL_PM2_TANGO_LCB = -0.0572
MIN_DISJOINT_PAIRED_QUERIES = 1000
DEFAULT_ALPHA = 0.05
DEFAULT_TARGET_POWER = 0.80

CANONICAL_BURNED_AUDIT_LIST: Tuple[str, ...] = tuple(BURNED_CORPORA)

_STD_NORMAL = NormalDist(0.0, 1.0)


def _sha256_json(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _array_sha256(arr: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(np.asarray(arr, dtype=np.float64))
    h = hashlib.sha256()
    h.update(str(contiguous.shape).encode("ascii"))
    h.update(contiguous.tobytes())
    return h.hexdigest()


_WEIGHT_SUFFIXES = {".safetensors", ".bin", ".pt", ".pth", ".npy", ".onnx", ".gguf", ".msgpack"}
_UNAUTHENTICATED_SUFFIXES = {".json", ".npz"}


def _manifest_signature_ok(manifest: Mapping[str, Any]) -> bool:
    digests = manifest.get("weight_file_digests")
    payload = manifest.get("payload_sha256")
    signature = manifest.get("signature_sha256")
    if not isinstance(digests, Mapping) or not digests:
        return False
    if not isinstance(payload, str) or len(payload) != 64:
        return False
    if not isinstance(signature, str) or len(signature) != 64:
        return False
    body = {
        "payload_sha256": payload,
        "weight_file_digests": {str(key): str(value) for key, value in sorted(digests.items())},
    }
    expected = sha256_digest(json.dumps(body, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8"))
    return signature == expected


def resolve_weight_provenance(
    *,
    is_live: bool,
    weight_bytes: Optional[bytes] = None,
    weight_paths: Optional[Sequence[str]] = None,
    provenance_manifest: Optional[Mapping[str, Any]] = None,
    content_bytes: bytes = b"",
) -> Dict[str, Any]:
    """Hash real weight bytes before a live confirmatory stamp is allowed."""
    file_digests: Dict[str, str] = {}
    saw_unauth = False
    for raw_path in weight_paths or ():
        path = Path(str(raw_path))
        if not path.is_file():
            continue
        if path.suffix.lower() in _UNAUTHENTICATED_SUFFIXES:
            saw_unauth = True
            continue
        file_digests[str(path)] = sha256_digest(path.read_bytes())
    if file_digests:
        blob = json.dumps(file_digests, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
        return {
            "authenticated": True,
            "reason": "weight-file-digests",
            "base_weights_sha256": sha256_digest(blob),
            "weight_file_digests": file_digests,
        }
    if weight_bytes:
        return {
            "authenticated": True,
            "reason": "weight-bytes",
            "base_weights_sha256": sha256_digest(bytes(weight_bytes)),
            "weight_file_digests": {},
        }
    if provenance_manifest is not None and _manifest_signature_ok(provenance_manifest):
        return {
            "authenticated": True,
            "reason": "signed-provenance-manifest",
            "base_weights_sha256": str(provenance_manifest["payload_sha256"]),
            "weight_file_digests": {str(k): str(v) for k, v in provenance_manifest["weight_file_digests"].items()},
        }
    content_sha = sha256_digest(content_bytes)
    if not is_live:
        return {
            "authenticated": False,
            "reason": "drycheck-content-bytes",
            "base_weights_sha256": content_sha,
            "weight_file_digests": {},
        }
    reason = "unauthenticated-json-or-npz" if saw_unauth or weight_paths else "refused-unauthenticated-weights"
    return {
        "authenticated": False,
        "reason": reason,
        "base_weights_sha256": content_sha,
        "weight_file_digests": {},
    }


# ---------------------------------------------------------------------------
# 1. Exact Tango (1998) Asymptotic Score CI & Non-Inferiority Test
# ---------------------------------------------------------------------------


def _clopper_pearson_two_sided(k: int, n: int, alpha: float = 0.05) -> Tuple[float, float]:
    """Exact two-sided Clopper-Pearson interval using regularized beta inversion via binomial CDF."""
    if n <= 0 or k < 0 or k > n:
        raise ValueError(f"Invalid binomial counts k={k}, n={n}")

    def binom_cdf(x: int, trials: int, p: float) -> float:
        if x < 0:
            return 0.0
        if x >= trials or p <= 0.0:
            return 1.0
        if p >= 1.0:
            return 0.0
        if x >= (trials + 1) * p:
            return 1.0 - binom_cdf(trials - x - 1, trials, 1.0 - p)
        log_p = math.log(p)
        log_q = math.log1p(-p)
        lead = math.exp(
            math.lgamma(trials + 1)
            - math.lgamma(x + 1)
            - math.lgamma(trials - x + 1)
            + x * log_p
            + (trials - x) * log_q
        )
        odds = (1.0 - p) / p
        scale = term = 1.0
        for idx in range(x, 0, -1):
            term *= idx * odds / (trials - idx + 1)
            scale += term
            if term <= scale * 1e-16:
                break
        return min(1.0, lead * scale)

    def invert(x: int, tail: float) -> float:
        lo, hi = 0.0, 1.0
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            if binom_cdf(x, n, mid) > tail:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)

    tail = alpha / 2.0
    low = invert(k - 1, 1.0 - tail) if k > 0 else 0.0
    high = invert(k, tail) if k < n else 1.0
    return float(low), float(high)


def _tango_constrained_q01(n: int, n10: int, n01: int, d: float) -> float:
    """Closed-form Tango (1998) constrained MLE q01(d) under H0: p10 - p01 = d."""
    a_coef = 2.0 * n
    b_coef = (2.0 * n - n10 + n01) * d - (n10 + n01)
    c_coef = -n01 * d * (1.0 - d)
    disc = max(0.0, b_coef * b_coef - 4.0 * a_coef * c_coef)
    q01 = (-b_coef + math.sqrt(disc)) / (2.0 * a_coef)
    return max(max(0.0, -d), min(0.5 * (1.0 - d), q01))


def tango_score_z(n: int, n10: int, n01: int, d: float) -> float:
    """Signed Tango (1998) asymptotic score statistic Z(d) for H0: p10 - p01 = d."""
    if n <= 0 or n10 < 0 or n01 < 0 or (n10 + n01) > n:
        raise ValueError(f"Invalid matched-pair counts n={n}, n10={n10}, n01={n01}")
    p10 = n10 / float(n)
    p01 = n01 / float(n)
    q01 = _tango_constrained_q01(n, n10, n01, d)
    var_num = max(2.0 * q01 + d * (1.0 - d), 1e-15)
    return float((p10 - p01 - d) * math.sqrt(n / var_num))


@dataclass(frozen=True)
class TangoNIResult:
    """One-shot exact Tango (1998) asymptotic score non-inferiority result."""

    n_pairs: int
    n11: int
    n10: int
    n01: int
    n00: int
    zero_write_rate: float
    reference_rate: float
    delta: float
    paired_discordance_rate: float
    paired_discordance_variance: float
    ni_margin: float
    alpha: float
    tango_score_lcb95: float
    tango_score_ucb95: float
    tango_z_at_margin: float
    tango_one_sided_ni_p: float
    clopper_pearson_lcb95: float
    clopper_pearson_ucb95: float
    clears_ni_margin: bool
    margin_loosened: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def compute_exact_tango_ni(
    zero_write_hits: Sequence[int],
    reference_hits: Sequence[int],
    *,
    ni_margin: float = PREREGISTERED_NI_MARGIN,
    alpha: float = DEFAULT_ALPHA,
) -> TangoNIResult:
    """Compute exact Tango (1998) matched-pair 95% score LCB and one-sided NI test vs `ni_margin`.

    Valid even at zero discordance (`n10 == n01 == 0`) without collapsing to `[0.0, 0.0]`.
    Enforces strict inequality `tango_score_lcb95 > ni_margin` per claim registry.
    """
    if len(zero_write_hits) != len(reference_hits) or not zero_write_hits:
        raise ValueError("zero_write_hits and reference_hits must be non-empty and equal length")
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    if not math.isfinite(ni_margin) or not (-1.0 < ni_margin < 1.0):
        raise ValueError("ni_margin must be finite and in (-1, 1)")

    n11 = n10 = n01 = n00 = 0
    for a_raw, b_raw in zip(zero_write_hits, reference_hits):
        if type(a_raw) not in (int, bool) or type(b_raw) not in (int, bool):
            raise ValueError("Binary outcomes must be integers/booleans in {0, 1}")
        a, b = int(a_raw), int(b_raw)
        if a not in (0, 1) or b not in (0, 1):
            raise ValueError("Binary outcomes must be in {0, 1}")
        if a == 1 and b == 1:
            n11 += 1
        elif a == 1 and b == 0:
            n10 += 1
        elif a == 0 and b == 1:
            n01 += 1
        else:
            n00 += 1

    n = len(zero_write_hits)
    delta = (n10 - n01) / float(n)
    p_disc = (n10 + n01) / float(n)
    var_disc = max(0.0, p_disc - delta * delta)

    z_crit = _STD_NORMAL.inv_cdf(1.0 - alpha / 2.0)
    z2 = z_crit * z_crit

    def tango_stat(d: float) -> float:
        z_val = tango_score_z(n, n10, n01, d)
        return z_val * z_val

    lo, hi = -1.0 + 1e-12, delta
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        if tango_stat(mid) > z2:
            lo = mid
        else:
            hi = mid
    tango_low = float(lo)

    lo, hi = delta, 1.0 - 1e-12
    for _ in range(64):
        mid = 0.5 * (lo + hi)
        if tango_stat(mid) > z2:
            hi = mid
        else:
            lo = mid
    tango_high = float(hi)

    z_margin = tango_score_z(n, n10, n01, ni_margin)
    ni_p = float(max(0.0, min(1.0, 1.0 - _STD_NORMAL.cdf(z_margin))))

    lw, uw = _clopper_pearson_two_sided(n10, n, alpha=alpha / 2.0)
    ll, ul = _clopper_pearson_two_sided(n01, n, alpha=alpha / 2.0)
    cp_low = max(-1.0, lw - ul)
    cp_high = min(1.0, uw - ll)

    return TangoNIResult(
        n_pairs=n,
        n11=n11,
        n10=n10,
        n01=n01,
        n00=n00,
        zero_write_rate=(n11 + n10) / float(n),
        reference_rate=(n11 + n01) / float(n),
        delta=float(delta),
        paired_discordance_rate=float(p_disc),
        paired_discordance_variance=float(var_disc),
        ni_margin=float(ni_margin),
        alpha=float(alpha),
        tango_score_lcb95=tango_low,
        tango_score_ucb95=tango_high,
        tango_z_at_margin=z_margin,
        tango_one_sided_ni_p=ni_p,
        clopper_pearson_lcb95=float(cp_low),
        clopper_pearson_ucb95=float(cp_high),
        clears_ni_margin=bool(tango_low > float(ni_margin)),
        margin_loosened=bool(ni_margin < PREREGISTERED_NI_MARGIN),
    )


# ---------------------------------------------------------------------------
# 2. Exact Power-Sizing from Paired Discordance Variance (N >= 1,000)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PowerSizingPlan:
    """Pre-registered paired sample-size plan derived from paired discordance variance."""

    assumed_p10: float
    assumed_p01: float
    assumed_delta: float
    paired_discordance_rate: float
    paired_discordance_variance: float
    ni_margin: float
    alpha: float
    target_power: float
    asymptotic_required_n: int
    floor_min_n: int
    required_n: int
    achieved_asymptotic_power: float

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def size_disjoint_ni_sample(
    assumed_p10: float,
    assumed_p01: float,
    *,
    ni_margin: float = PREREGISTERED_NI_MARGIN,
    alpha: float = DEFAULT_ALPHA,
    target_power: float = DEFAULT_TARGET_POWER,
    min_n: int = MIN_DISJOINT_PAIRED_QUERIES,
) -> PowerSizingPlan:
    """Size paired disjoint sample `N >= 1,000` from paired discordance variance `sigma^2_disc`."""
    if not (0.0 <= assumed_p10 <= 1.0 and 0.0 <= assumed_p01 <= 1.0 and assumed_p10 + assumed_p01 <= 1.0):
        raise ValueError("assumed_p10 and assumed_p01 must be valid probabilities with sum <= 1")
    if not (0.0 < alpha < 1.0) or not (0.0 < target_power < 1.0):
        raise ValueError("alpha and target_power must be in (0, 1)")
    if min_n < MIN_DISJOINT_PAIRED_QUERIES:
        raise ValueError(f"min_n must be >= {MIN_DISJOINT_PAIRED_QUERIES}")

    delta_0 = float(assumed_p10 - assumed_p01)
    gap = delta_0 - float(ni_margin)
    if gap <= 0.0:
        raise ValueError(
            f"Assumed true delta ({delta_0:.4f}) must exceed ni_margin ({ni_margin:.4f}) to size NI power"
        )
    p_disc = float(assumed_p10 + assumed_p01)
    var_disc = max(p_disc - delta_0 * delta_0, 1e-6)

    z_alpha = _STD_NORMAL.inv_cdf(1.0 - alpha / 2.0)
    z_beta = _STD_NORMAL.inv_cdf(target_power)
    n_asymp = int(math.ceil(((z_alpha + z_beta) ** 2) * var_disc / (gap * gap)))
    required_n = max(int(min_n), n_asymp)

    # Achieved asymptotic power at required_n using paired discordance variance
    se = math.sqrt(var_disc / float(required_n))
    achieved_power = float(_STD_NORMAL.cdf(gap / se - z_alpha))

    return PowerSizingPlan(
        assumed_p10=float(assumed_p10),
        assumed_p01=float(assumed_p01),
        assumed_delta=delta_0,
        paired_discordance_rate=p_disc,
        paired_discordance_variance=float(var_disc),
        ni_margin=float(ni_margin),
        alpha=float(alpha),
        target_power=float(target_power),
        asymptotic_required_n=n_asymp,
        floor_min_n=int(min_n),
        required_n=required_n,
        achieved_asymptotic_power=achieved_power,
    )


# ---------------------------------------------------------------------------
# 3. Mechanical burn_ledger Admission & Audit Gate
# ---------------------------------------------------------------------------


def audit_burned_corpora_rejection(ledger: Optional[BurnLedger] = None) -> Dict[str, Any]:
    """Verify `burn_ledger` fails closed on all canonical burned corpora."""
    active_ledger = build_burn_ledger() if ledger is None else ledger
    per_corpus: Dict[str, bool] = {}
    known_names = []
    for entry in active_ledger.datasets:
        # BurnLedger.datasets is (dataset_id, source_origins), not a bare name list.
        known_names.append(entry if isinstance(entry, str) else entry[0])
    known = {normalize_dataset_id(name) for name in known_names}
    for corpus in CANONICAL_BURNED_AUDIT_LIST:
        rejected = normalize_dataset_id(corpus) in known
        if not rejected:
            try:
                assert_corpus_unburned(corpus, ledger=active_ledger)
            except BurnedCorpusError:
                rejected = True
        per_corpus[corpus] = rejected
    return {
        "all_burned_rejected": all(per_corpus.values()),
        "burn_ledger_sha256": active_ledger.sha256,
        "checked_corpora": per_corpus,
    }


def verify_disjoint_corpus_admission(
    corpus_label: str,
    doc_texts: Sequence[str],
    query_texts: Sequence[str],
    *,
    historical_fingerprints: Mapping[str, Sequence[str]],
    ledger: Optional[BurnLedger] = None,
    allow_local_drycheck_smoke: Optional[bool] = None,
) -> Dict[str, Any]:
    """Fail-closed `burn_ledger` + SHA-256 overlap gate for candidate disjoint NI corpora.

    ``allow_local_drycheck_smoke=True`` admits a partial inventory as non-confirmatory local-drycheck smoke.
    """
    active_ledger = build_burn_ledger() if ledger is None else ledger
    assert_corpus_unburned(
        corpus_label,
        doc_texts=doc_texts,
        query_texts=query_texts,
        historical_fingerprints=historical_fingerprints,
        ledger=active_ledger,
        allow_local_drycheck_smoke=allow_local_drycheck_smoke,
    )
    doc_hashes = sorted({text_fingerprint(t) for t in doc_texts})
    query_hashes = sorted({text_fingerprint(t) for t in query_texts})
    return {
        "corpus_label": str(corpus_label),
        "admitted_unburned": True,
        "doc_count": len(doc_texts),
        "query_count": len(query_texts),
        "unique_doc_fingerprints": len(doc_hashes),
        "unique_query_fingerprints": len(query_hashes),
        "burn_ledger_sha256": active_ledger.sha256,
        "admission_sha256": _sha256_json(
            {
                "corpus_label": corpus_label,
                "doc_count": len(doc_texts),
                "query_count": len(query_texts),
                "first_doc_fp": doc_hashes[0] if doc_hashes else "",
                "first_query_fp": query_hashes[0] if query_hashes else "",
            }
        ),
    }


# ---------------------------------------------------------------------------
# 4. Canonical gBench LOFT SciFact Pilot (357 docs, 100 queries) vs Derived Zero-Write Pilot
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class LoftSciFactFixture:
    """Strictly pilot-scoped gBench LOFT SciFact representation (357 docs, 100 queries)."""

    gbench_commit: str
    doc_ids: Tuple[str, ...]
    query_ids: Tuple[str, ...]
    doc_texts: Tuple[str, ...]
    query_texts: Tuple[str, ...]
    qrels: Dict[str, Dict[str, float]]
    docs_old: np.ndarray
    docs_new: np.ndarray
    queries_old: np.ndarray
    queries_new: np.ndarray
    anchor_indices: np.ndarray
    is_pilot_only: bool = True
    confirmatory_eligible: bool = False
    dataset_origin: str = "synthetic_schema_drycheck_fixture"


def build_deterministic_loft_scifact_fixture(
    *,
    seed: int = 5001,
    d_old: int = 32,
    d_new: int = 48,
) -> LoftSciFactFixture:
    """Build deterministic 357-doc, 100-query LOFT SciFact pilot fixture (CPU drycheck)."""
    rng = np.random.default_rng(seed)
    n_docs = LOFT_SCIFACT_DOC_COUNT
    n_queries = LOFT_SCIFACT_QUERY_COUNT

    # Latent semantic space + rectangular rotation + realistic encoder anisotropy centroid + rank-1 skew
    latent_dim = min(d_old, d_new)
    latent_docs = l2_normalize(rng.normal(size=(n_docs, latent_dim)))
    rot_old, _ = np.linalg.qr(rng.normal(size=(latent_dim, d_old)).T)
    rot_old = rot_old[:d_old, :latent_dim].T  # [latent_dim, d_old]
    rot_new, _ = np.linalg.qr(rng.normal(size=(latent_dim, d_new)).T)
    rot_new = rot_new[:d_new, :latent_dim].T  # [latent_dim, d_new]

    mu_old = l2_normalize(rng.normal(size=(1, d_old))) * 0.45
    mu_new = l2_normalize(rng.normal(size=(1, d_new))) * 0.55
    skew_u = l2_normalize(rng.normal(size=(d_old, 1)))
    skew_v = l2_normalize(rng.normal(size=(1, d_new)))

    raw_old = latent_docs @ rot_old + mu_old + 0.04 * rng.normal(size=(n_docs, d_old))
    docs_old = l2_normalize(raw_old)
    raw_new = (
        latent_docs @ rot_new
        + mu_new
        + 0.12 * ((raw_old - mu_old) @ skew_u) @ skew_v
        + 0.04 * rng.normal(size=(n_docs, d_new))
    )
    docs_new = l2_normalize(raw_new)

    doc_ids = tuple(f"loft_sf_doc_{i:04d}" for i in range(n_docs))
    query_ids = tuple(f"loft_sf_q_{i:03d}" for i in range(n_queries))
    doc_texts = tuple(
        f"Synthetic schema drycheck passage {i:04d} biomedical slot evidence token {i}" for i in range(n_docs)
    )
    query_texts = tuple(
        f"Synthetic schema drycheck question {i:03d} biomedical slot probe {i}" for i in range(n_queries)
    )

    # Each query targets 1-2 relevant documents deterministically
    qrels: Dict[str, Dict[str, float]] = {}
    q_latent = np.empty((n_queries, latent_dim), dtype=np.float64)
    for qi in range(n_queries):
        primary_doc = (qi * 3 + 7) % n_docs
        secondary_doc = (qi * 3 + 11) % n_docs
        qrels[query_ids[qi]] = {doc_ids[primary_doc]: 2.0, doc_ids[secondary_doc]: 1.0}
        q_latent[qi] = l2_normalize(
            0.78 * latent_docs[primary_doc] + 0.22 * latent_docs[secondary_doc] + 0.08 * rng.normal(size=(latent_dim,))
        )

    queries_old = l2_normalize(q_latent @ rot_old + mu_old + 0.05 * rng.normal(size=(n_queries, d_old)))
    queries_new = l2_normalize(
        q_latent @ rot_new
        + mu_new
        + 0.12 * (((q_latent @ rot_old) @ skew_u) @ skew_v)
        + 0.05 * rng.normal(size=(n_queries, d_new))
    )
    anchor_indices = np.sort(rng.choice(n_docs, size=180, replace=False))

    return LoftSciFactFixture(
        gbench_commit="",
        doc_ids=doc_ids,
        query_ids=query_ids,
        doc_texts=doc_texts,
        query_texts=query_texts,
        qrels=qrels,
        docs_old=docs_old,
        docs_new=docs_new,
        queries_old=queries_old,
        queries_new=queries_new,
        anchor_indices=anchor_indices,
    )


def run_canonical_loft_scifact_pilot(fixture: LoftSciFactFixture) -> Dict[str, Any]:
    """Evaluate canonical gBench LOFT SciFact (357 docs, 100 queries) strictly as an isolated pilot."""
    if len(fixture.doc_ids) != LOFT_SCIFACT_DOC_COUNT:
        raise ValueError(f"Canonical LOFT SciFact requires {LOFT_SCIFACT_DOC_COUNT} docs, got {len(fixture.doc_ids)}")
    if len(fixture.query_ids) != LOFT_SCIFACT_QUERY_COUNT:
        raise ValueError(
            f"Canonical LOFT SciFact requires {LOFT_SCIFACT_QUERY_COUNT} queries, got {len(fixture.query_ids)}"
        )

    old_scores = l2_normalize(fixture.queries_old) @ l2_normalize(fixture.docs_old).T
    new_scores = l2_normalize(fixture.queries_new) @ l2_normalize(fixture.docs_new).T
    old_ndcg, old_rec = evaluate_score_matrix(old_scores, fixture.query_ids, fixture.doc_ids, fixture.qrels, k=10)
    new_ndcg, new_rec = evaluate_score_matrix(new_scores, fixture.query_ids, fixture.doc_ids, fixture.qrels, k=10)

    return {
        "track_role": "schema_drycheck_dimension_pilot",
        "dataset_origin": fixture.dataset_origin,
        "gbench_commit": fixture.gbench_commit,
        "doc_count": len(fixture.doc_ids),
        "query_count": len(fixture.query_ids),
        "confirmatory_ni_pooled": False,
        "burned_for_confirmatory_ni": True,
        "native_old_mean_ndcg_at_10": float(np.mean(list(old_ndcg.values()))),
        "native_old_mean_recall_at_10": float(np.mean(list(old_rec.values()))),
        "native_new_mean_ndcg_at_10": float(np.mean(list(new_ndcg.values()))),
        "native_new_mean_recall_at_10": float(np.mean(list(new_rec.values()))),
    }


def run_derived_zero_write_migration_pilot(fixture: LoftSciFactFixture) -> Dict[str, Any]:
    """Derived zero-write migration pilot testing Rank1QueryResidual u(v^T q) + ZeroMuPolarBridge.

    Strictly separated from the canonical gBench LOFT SciFact score AND from the confirmatory NI test.
    Verifies zero resident document vector writes (`resident_docs_sha256_before == after`).
    """
    docs_resident = np.array(fixture.docs_old, dtype=np.float64, copy=True)
    docs_sha_before = _array_sha256(docs_resident)

    src_anc = fixture.docs_new[fixture.anchor_indices]
    tgt_anc = docs_resident[fixture.anchor_indices]

    # Fit bridges mapping new-encoder queries -> unmodified old-encoder resident index (zero writes!)
    bridges = {
        "UncenteredProcrustes": UncenteredProcrustes.fit(src_anc, tgt_anc),
        "CenteredProcrustes": CenteredProcrustes.fit(src_anc, tgt_anc),
        "Rank1QueryResidual": Rank1QueryResidual.fit(src_anc, tgt_anc),
        "ZeroMuPolarBridge_dual_adap": ZeroMuPolarBridge.fit(src_anc, tgt_anc, scoring_mode="dual_adap"),
        "ZeroMuPolarBridge_query_only": ZeroMuPolarBridge.fit(src_anc, tgt_anc, scoring_mode="query_only"),
    }

    zero_mask = np.zeros(len(fixture.doc_ids), dtype=bool)
    mergers = {
        "RawCosineMerge": RawCosineMerge(),
        "CSLSMerge": CSLSMerge(csls_k=10),
    }
    n_dev = max(4, int(fixture.queries_new.shape[0]) // 5)
    if n_dev >= int(fixture.queries_new.shape[0]):
        n_dev = int(fixture.queries_new.shape[0]) - 1
    dev_queries = fixture.queries_new[:n_dev]
    eval_queries = fixture.queries_new[n_dev:]
    eval_query_ids = fixture.query_ids[n_dev:]
    dev_docs = l2_normalize(np.random.default_rng(5001).normal(size=(12, docs_resident.shape[1])))

    arm_metrics: Dict[str, Dict[str, Any]] = {}
    for b_name, bridge in bridges.items():
        score_kwargs: Dict[str, Any] = {}
        if b_name.startswith("ZeroMu"):
            score_kwargs["dev_docs"] = dev_docs
        raw_scores = bridge.score_matrix(eval_queries, docs_resident, **score_kwargs)
        dummy_native = np.zeros_like(raw_scores)
        for m_name, merger in mergers.items():
            if m_name == "CSLSMerge":
                dev_scores = bridge.score_matrix(dev_queries, docs_resident, **score_kwargs)
                fused = merger.merge(
                    raw_scores,
                    dummy_native,
                    zero_mask,
                    bridged_ref=dev_scores,
                    native_ref=dummy_native,
                    strict_disjoint_dev=True,
                )
            else:
                fused = merger.merge(
                    raw_scores, dummy_native, zero_mask, bridged_ref=raw_scores, native_ref=dummy_native
                )
            pq_n, pq_r = evaluate_score_matrix(fused, eval_query_ids, fixture.doc_ids, fixture.qrels, k=10)
            key = f"{b_name}__{m_name}"
            arm_metrics[key] = {
                "bridge": b_name,
                "merge": m_name,
                "mean_ndcg_at_10": float(np.mean(list(pq_n.values()))),
                "mean_recall_at_10": float(np.mean(list(pq_r.values()))),
                "kappa": float(bridge.condition_number),
                "nonzero_kappa": float(bridge.nonzero_condition_number),
                "weights_sha256": bridge.weights_sha256,
            }

    rank1_bridge = bridges["Rank1QueryResidual"]
    assert isinstance(rank1_bridge, Rank1QueryResidual)
    docs_sha_after = _array_sha256(docs_resident)

    return {
        "track_role": "derived_zero_write_migration_pilot",
        "separated_from_canonical_gbench": True,
        "confirmatory_ni_pooled": False,
        "anchor_count": int(fixture.anchor_indices.size),
        "resident_docs_unmodified": bool(docs_sha_before == docs_sha_after),
        "resident_docs_sha256_before": docs_sha_before,
        "resident_docs_sha256_after": docs_sha_after,
        "resident_write_bytes": 0,
        "rank1_query_residual_diagnostics": {
            "alpha": float(rank1_bridge.alpha),
            "u1_norm": float(np.linalg.norm(rank1_bridge.u1)),
            "v1_norm": float(np.linalg.norm(rank1_bridge.v1)),
            "effective_rank": int(rank1_bridge.effective_rank),
        },
        "arms": arm_metrics,
    }


# ---------------------------------------------------------------------------
# 5. Power-Sized Disjoint Corpus One-Shot Exact Tango NI (-0.0500)
# ---------------------------------------------------------------------------


def build_synthetic_disjoint_ni_pool(
    *,
    corpus_label: str = "pv7-b50-disjoint-synthetic-ni-pool",
    n_queries: int = 1024,
    n_docs: int = 240,
    seed: int = 5050,
    p_both_hit: float = 0.84,
    p10: float = 0.032,
    p01: float = 0.030,
) -> Dict[str, Any]:
    """Deterministic CPU-drycheck disjoint corpus + paired binary hit outcomes (never labeled live)."""
    rng = np.random.default_rng(seed)
    if p_both_hit + p10 + p01 > 1.0:
        raise ValueError("Joint outcome probabilities must sum to <= 1.0")
    p00 = 1.0 - (p_both_hit + p10 + p01)
    draws = rng.choice(4, size=n_queries, p=[p_both_hit, p10, p01, p00])
    zw_hits = [1 if d in (0, 1) else 0 for d in draws]
    ref_hits = [1 if d in (0, 2) else 0 for d in draws]
    doc_texts = [f"{corpus_label} unburned resident passage {i:05d} seed {seed}" for i in range(n_docs)]
    query_texts = [f"{corpus_label} unburned paired verification query {i:05d} seed {seed}" for i in range(n_queries)]
    return {
        "corpus_label": corpus_label,
        "doc_texts": doc_texts,
        "query_texts": query_texts,
        "zero_write_hits": zw_hits,
        "reference_hits": ref_hits,
    }


def evaluate_disjoint_zero_write_ni(
    *,
    corpus_label: str,
    doc_texts: Sequence[str],
    query_texts: Sequence[str],
    zero_write_hits: Sequence[int],
    reference_hits: Sequence[int],
    sizing_plan: PowerSizingPlan,
    historical_fingerprints: Mapping[str, Sequence[str]],
    ledger: Optional[BurnLedger] = None,
    ni_margin: float = PREREGISTERED_NI_MARGIN,
    alpha: float = DEFAULT_ALPHA,
    is_live_confirmatory: bool = False,
    weight_bytes: Optional[bytes] = None,
    weight_paths: Optional[Sequence[str]] = None,
    provenance_manifest: Optional[Mapping[str, Any]] = None,
    allow_local_drycheck_smoke: Optional[bool] = None,
) -> Dict[str, Any]:
    """Run the one-shot disjoint zero-write exact Tango NI gate vs `-0.0500`.

    Enforces:
      1. Mechanical `burn_ledger` + SHA-256 disjointness gate BEFORE scoring.
      2. Hard refusal to loosen `ni_margin` below `-0.0500`.
      3. Underpowered pools (`n < sizing_plan.required_n` or `n < 1000`) emit `"status": "underpowered"`
         instead of loosening `-0.0500` or declaring confirmatory pass.
      4. Preserves historical `PM-2` `-0.0572` untouched.
    """
    if float(ni_margin) < PREREGISTERED_NI_MARGIN - 1e-12:
        raise ValueError(
            f"Refusing to loosen pre-registered NI margin {PREREGISTERED_NI_MARGIN:.4f} to {ni_margin:.4f}"
        )

    admission = verify_disjoint_corpus_admission(
        corpus_label,
        doc_texts,
        query_texts,
        historical_fingerprints=historical_fingerprints,
        ledger=ledger,
        allow_local_drycheck_smoke=allow_local_drycheck_smoke,
    )

    if len(zero_write_hits) != len(query_texts) or len(reference_hits) != len(query_texts):
        raise ValueError("zero_write_hits and reference_hits lengths must equal len(query_texts)")

    n_pairs = len(zero_write_hits)
    tango = compute_exact_tango_ni(zero_write_hits, reference_hits, ni_margin=ni_margin, alpha=alpha)
    underpowered = n_pairs < max(sizing_plan.required_n, MIN_DISJOINT_PAIRED_QUERIES)

    content_bytes = "\n".join(doc_texts).encode("utf-8")
    provenance = resolve_weight_provenance(
        is_live=bool(is_live_confirmatory),
        weight_bytes=weight_bytes,
        weight_paths=weight_paths,
        provenance_manifest=provenance_manifest,
        content_bytes=content_bytes,
    )
    if underpowered:
        status = "underpowered"
        ni_decision = "underpowered"
        one_shot_consumed = False
    elif is_live_confirmatory and not provenance["authenticated"]:
        status = "refused-unauthenticated-weights"
        ni_decision = "refused"
        one_shot_consumed = False
    else:
        one_shot_consumed = True
        ni_decision = "pass" if tango.clears_ni_margin else "loss"
        if is_live_confirmatory:
            status = "verified-pass" if tango.clears_ni_margin else "verified-loss"
        else:
            status = "local-drycheck-pass" if tango.clears_ni_margin else "local-drycheck-loss"

    weight_sha = str(provenance["base_weights_sha256"])
    contract = TRA1StateIdentityContract(
        base_weights_sha256=weight_sha,
        adapter_weights_sha256=EMPTY_ADAPTER_SHA256,
        config_sha256=_sha256_json({"bundle_id": BUNDLE_ID, "ni_margin": ni_margin, "alpha": alpha}),
        tokenizer_sha256=sha256_digest(weight_sha.encode("utf-8")),
        kv_or_state_dtype="float32",
        rope_freq_base=10000.0,
        rope_scaling_hash=sha256_digest(content_bytes[:64] or b"\x00"),
    )

    return {
        "bundle_id": BUNDLE_ID,
        "item": ITEM_ID,
        "status": status,
        "ni_decision": ni_decision,
        "one_shot_consumed": one_shot_consumed,
        "is_live_confirmatory": bool(is_live_confirmatory),
        "weight_provenance": provenance,
        "underpowered": bool(underpowered),
        "available_paired_queries": n_pairs,
        "required_paired_queries": sizing_plan.required_n,
        "ni_margin": float(ni_margin),
        "margin_loosened": False,
        "historical_pm2_reference": {
            "tango_score_lcb95": HISTORICAL_PM2_TANGO_LCB,
            "preserved_unmodified": True,
        },
        "burn_admission": admission,
        "power_sizing": sizing_plan.to_dict(),
        "tango_ni": tango.to_dict(),
        "tra1_state_identity_digest": contract.digest(),
        "spine_digest": claim_registry_sha256(),
    }


__all__ = [
    "BUNDLE_ID",
    "CANONICAL_BURNED_AUDIT_LIST",
    "DEFAULT_ALPHA",
    "DEFAULT_TARGET_POWER",
    "GBENCH_CANONICAL_COMMIT",
    "HISTORICAL_PM2_TANGO_LCB",
    "ITEM_ID",
    "LOFT_SCIFACT_DOC_COUNT",
    "LOFT_SCIFACT_QUERY_COUNT",
    "LoftSciFactFixture",
    "MIN_DISJOINT_PAIRED_QUERIES",
    "PREREGISTERED_NI_MARGIN",
    "PowerSizingPlan",
    "SPINE_DIGEST",
    "TangoNIResult",
    "audit_burned_corpora_rejection",
    "build_deterministic_loft_scifact_fixture",
    "build_synthetic_disjoint_ni_pool",
    "compute_exact_tango_ni",
    "evaluate_disjoint_zero_write_ni",
    "run_canonical_loft_scifact_pilot",
    "run_derived_zero_write_migration_pilot",
    "size_disjoint_ni_sample",
    "tango_score_z",
    "verify_disjoint_corpus_admission",
]
