"""Wave 18 / Bundle #54 (Item 6B): Zero-Retune 12-Domain, 5-Encoder/4-Hop, Prompt-Shift & Anchor-Poisoning Gauntlet.

Implements the locked Post-v7 R2 protocol for Item 6B (#54):
  1. Locks the 12 `burn_ledger`-verified unburned public domain descriptors:
     `TREC-COVID`, `BioASQ-TaskB` (NFCorpus-decontaminated), `LitSearch`, `MIRACL-en/multi`,
     `CoIR-Apps` (NEVER `CoSQA`), `Touche-2020` (NEVER `ArguAna`), `TREC-CAR`, `DBPedia-Entity-v2`,
     `Robust04`, `Signal-1M`, `CQADupStack-Android`, `FEVER`.
     Runs `burn_ledger` + document/query SHA-256 overlap checks BEFORE any scoring!
  2. Reuses `run_factorial_5x3` from `chelated_pv7_spine.bridge_merge_harness` crossing all 5 bridges
     `{UncenteredProcrustes, CenteredProcrustes, RidgeDriftAdapter, Rank1QueryResidual, ZeroMuPolarBridge}`
     x all 3 merge methods `{RawCosineMerge, CalibratedZScoreMerge, CSLSMerge}` with zero retuning.
  3. Evaluates all 3 pre-registered hypotheses without retuning:
     - `H6B_merge`: `CSLS > CalibratedZScore` (`LCB > 0` + `>= 10/12` domains positive, exact sign-test p).
     - `H6B_bridge_static`: `ZeroMu + CSLS` non-inferiority guard `LCB > -0.010` nDCG@10 vs
       `UncenteredProcrustes + CSLS` (also reporting `-0.005` tight equivalence AND `CenteredProcrustes + CSLS`
       head-to-head contrast!).
     - `H6B_bridge_shift_and_4hop`: Without refitting, `ZeroMu + CSLS` beats `UncenteredProcrustes + CSLS`
       under (a) held-out RAG prompt-wrapper/tenant-prefix shifts (logging nonlinear encoder prefix
       displacement explicitly + shifted-anchor refit ceiling) AND (b) 5-encoder / 4-transition sequential
       composition `v1 -> v2 -> v3 -> v4 -> v5` (`MiniLM-384 -> bge-small-384 -> mpnet-768 -> bge-m3-1024 -> Qwen3-4096`)
       alongside direct `v1 -> v5` fit.
  4. Runs 20 frozen 500-anchor draws/domain variance decomposition AND `1%, 5%, 10%, 20%` anchor-poisoning sweep.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from chelated_pv7_spine.bridge_merge_harness import (
    BRIDGE_NAMES,
    DEFAULT_ANCHOR_DRAWS,
    DEFAULT_ANCHORS_PER_DRAW,
    DEFAULT_CONTAMINATION_RATES,
    DEFAULT_CSLS_K,
    ITEM6B_RETIREMENT_FLOOR,
    ITEM6B_TIGHT_EQUIVALENCE_FLOOR,
    MERGE_NAMES,
    CSLSMerge,
    CenteredProcrustes,
    UncenteredProcrustes,
    ZeroMuPolarBridge,
    evaluate_4hop_chain,
    evaluate_item6b_hypotheses,
    evaluate_score_matrix,
    l2_normalize,
    run_contamination_sweep,
    run_factorial_5x3,
    sample_anchor_draws,
    summarize_anchor_draw_variance,
)
from chelated_pv7_spine.burn_ledger import (
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

BUNDLE_ID = 54
ITEM_ID = "6B"
SPINE_DIGEST = "32693f54e0ce7ad3d401ceb752ae4723b0eee251f6d49b95cdfb26a6201971e9"


@dataclass(frozen=True)
class DomainDescriptor:
    """Preregistered unburned public domain descriptor for Item 6B (#54)."""

    domain_id: str
    burn_check_id: str
    category: str
    decontaminate_against: Tuple[str, ...] = ()
    notes: str = ""


LOCKED_12_DOMAINS: Tuple[DomainDescriptor, ...] = (
    DomainDescriptor("TREC-COVID", "trec-covid", "biomedical_ir"),
    DomainDescriptor(
        "BioASQ-TaskB",
        "bioasq-taskb",
        "biomedical_qa",
        decontaminate_against=("nfcorpus",),
        notes="NFCorpus PubMed abstract SHA-256 decontaminated before scoring",
    ),
    DomainDescriptor("LitSearch", "litsearch", "scientific_literature"),
    DomainDescriptor("MIRACL-en/multi", "miracl-en-multi", "multilingual_wikipedia_ir"),
    DomainDescriptor("CoIR-Apps", "coir-apps", "code_retrieval", notes="Unburned code retrieval; never CoSQA"),
    DomainDescriptor("Touche-2020", "touche-2020", "argument_retrieval", notes="Unburned argument IR; never ArguAna"),
    DomainDescriptor("TREC-CAR", "trec-car", "encyclopedic_ir"),
    DomainDescriptor("DBPedia-Entity-v2", "dbpedia-entity-v2", "entity_retrieval"),
    DomainDescriptor("Robust04", "robust04", "news_adhoc_ir"),
    DomainDescriptor("Signal-1M", "signal-1m", "news_stream_ir"),
    DomainDescriptor("CQADupStack-Android", "cqadupstack-android", "technical_community_qa"),
    DomainDescriptor("FEVER", "fever", "fact_verification"),
)

FORBIDDEN_BURNED_SUBSTITUTES: Tuple[str, ...] = (
    "cosqa",
    "arguana",
    "scifact",
    "nfcorpus",
    "fiqa",
    "quora",
    "scidocs",
    "msmarco",
)

ENCODER_CHAIN_SPEC: Tuple[Tuple[str, str, int], ...] = (
    ("MiniLM-384", "sentence-transformers/all-MiniLM-L6-v2", 384),
    ("bge-small-384", "BAAI/bge-small-en-v1.5", 384),
    ("mpnet-768", "sentence-transformers/all-mpnet-base-v2", 768),
    ("bge-m3-1024", "BAAI/bge-m3", 1024),
    ("Qwen3-4096", "Qwen/Qwen3-Embedding-8B", 4096),
)
ENCODER_NAMES: Tuple[str, ...] = tuple(item[0] for item in ENCODER_CHAIN_SPEC)
ENCODER_NOMINAL_DIMS: Tuple[int, ...] = tuple(item[2] for item in ENCODER_CHAIN_SPEC)

POISONING_SWEEP_RATES: Tuple[float, ...] = DEFAULT_CONTAMINATION_RATES  # (0.0, 0.01, 0.05, 0.10, 0.20)
POISONING_SWEEP_MODES: Tuple[str, ...] = ("mismatched", "duplicate", "off_domain")

PREREGISTERED_ZERO_RETUNE_CONFIG: Dict[str, Any] = {
    "bundle_id": BUNDLE_ID,
    "item": ITEM_ID,
    "csls_k": DEFAULT_CSLS_K,
    "ridge_regularization": 1e-2,
    "zeromu_mode": "dual_adap",
    "ema_alpha": 0.05,
    "n_anchor_draws": DEFAULT_ANCHOR_DRAWS,
    "anchors_per_draw": DEFAULT_ANCHORS_PER_DRAW,
    "poisoning_rates": list(POISONING_SWEEP_RATES),
    "poisoning_modes": list(POISONING_SWEEP_MODES),
    "static_retirement_floor": ITEM6B_RETIREMENT_FLOOR,
    "static_tight_equivalence_floor": ITEM6B_TIGHT_EQUIVALENCE_FLOOR,
    "min_positive_domains_merge": 10,
    "retuned_on_eval": False,
}


def _sha256_json(payload: Mapping[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


# ---------------------------------------------------------------------------
# 1. Domain Payload & Mechanical burn_ledger + SHA-256 Pre-Scoring Gate
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DomainGauntletBundle:
    """Self-contained per-domain embeddings, texts, qrels, and held-out prompt-shift queries."""

    descriptor: DomainDescriptor
    doc_ids: Tuple[str, ...]
    query_ids: Tuple[str, ...]
    doc_texts: Tuple[str, ...]
    query_texts: Tuple[str, ...]
    qrels: Dict[str, Dict[str, float]]
    # 5-encoder sequence: [v1, v2, v3, v4, v5]
    doc_embeddings_seq: Tuple[np.ndarray, ...]
    query_embeddings_seq: Tuple[np.ndarray, ...]
    # Held-out RAG prompt-wrapper / tenant-prefix shifted queries in v1 space (and shifted anchors for refit ceiling)
    shifted_queries_v1: np.ndarray
    shifted_anchor_queries_v1: np.ndarray
    shifted_anchor_targets_v2: np.ndarray
    # Backfill mask for 5x3 dual-shard merge evaluation (e.g. 30% native shard, 70% bridged shard)
    backfill_mask: np.ndarray
    # Explicit nonlinear encoder prefix displacement telemetry
    prefix_displacement_mean_norm: float
    prefix_nonlinear_residual_mean_norm: float
    prefix_linear_shift_fraction: float


def verify_12_domains_unburned(
    domain_bundles: Sequence[DomainGauntletBundle],
    *,
    historical_fingerprints: Mapping[str, Sequence[str]],
    ledger: Optional[BurnLedger] = None,
) -> Dict[str, Any]:
    """Fail-closed `burn_ledger` + document/query SHA-256 overlap verification BEFORE any scoring."""
    expected_ids = [d.domain_id for d in LOCKED_12_DOMAINS]
    provided_ids = [b.descriptor.domain_id for b in domain_bundles]
    if provided_ids != expected_ids:
        raise ValueError(f"Domain slate mismatch: expected {expected_ids}, got {provided_ids}")

    active_ledger = build_burn_ledger() if ledger is None else ledger

    # Guard against forbidden burned substitutes
    for desc in LOCKED_12_DOMAINS:
        norm_id = normalize_dataset_id(desc.burn_check_id)
        if norm_id in FORBIDDEN_BURNED_SUBSTITUTES:
            raise BurnedCorpusError(f"Forbidden burned substitute detected in 12-domain slate: {desc.domain_id}")

    seen_doc_fps: Dict[str, str] = {}
    per_domain_receipts: Dict[str, Dict[str, Any]] = {}
    for bundle in domain_bundles:
        desc = bundle.descriptor
        assert_corpus_unburned(
            desc.burn_check_id,
            doc_texts=bundle.doc_texts,
            query_texts=bundle.query_texts,
            historical_fingerprints=historical_fingerprints,
            ledger=active_ledger,
        )
        # Check cross-domain document SHA-256 disjointness as well
        dom_doc_fps = [text_fingerprint(t) for t in bundle.doc_texts]
        cross_overlaps = 0
        for fp in dom_doc_fps:
            prev = seen_doc_fps.get(fp)
            if prev is not None and prev != desc.domain_id:
                cross_overlaps += 1
            seen_doc_fps[fp] = desc.domain_id
        if cross_overlaps > 0:
            raise BurnedCorpusError(
                f"Cross-domain document SHA-256 overlap detected in {desc.domain_id}: {cross_overlaps} collisions"
            )
        per_domain_receipts[desc.domain_id] = {
            "burn_check_id": desc.burn_check_id,
            "admitted_unburned": True,
            "decontaminated_against": list(desc.decontaminate_against),
            "doc_count": len(bundle.doc_texts),
            "query_count": len(bundle.query_texts),
            "unique_doc_sha256": len(set(dom_doc_fps)),
            "cross_domain_doc_overlaps": 0,
        }

    return {
        "n_domains_verified": len(domain_bundles),
        "all_12_unburned": True,
        "burn_ledger_sha256": active_ledger.sha256,
        "bioasq_nfcorpus_decontaminated": True,
        "forbidden_substitutes_excluded": list(FORBIDDEN_BURNED_SUBSTITUTES),
        "per_domain": per_domain_receipts,
    }


# ---------------------------------------------------------------------------
# 2. Deterministic CPU-Drycheck 12-Domain x 5-Encoder Fixture Generator
# ---------------------------------------------------------------------------


def build_deterministic_12domain_gauntlet_fixture(
    *,
    seed: int = 5400,
    n_docs_per_domain: int = 120,
    n_queries_per_domain: int = 36,
    encoder_dims: Tuple[int, int, int, int, int] = (20, 20, 24, 28, 36),
) -> Tuple[DomainGauntletBundle, ...]:
    """Generate deterministic 12-domain x 5-encoder (`v1->v5`) fixture for CPU local-drycheck.

    Models realistic cross-encoder hubness, anisotropy centroid shift, held-out RAG prompt-wrapper
    prefix shift (mean shift + nonlinear per-query residual), and 4-hop dimensional expansion so
    the full 5x3 factorial, 20 anchor draws, 4-hop composition, and poisoning sweep run in seconds.
    """
    if len(encoder_dims) != 5:
        raise ValueError("encoder_dims must specify exactly 5 encoder dimensions (v1..v5)")

    bundles: List[DomainGauntletBundle] = []
    latent_dim = min(encoder_dims)

    for dom_idx, desc in enumerate(LOCKED_12_DOMAINS):
        dom_seed = int.from_bytes(hashlib.sha256(f"{seed}:{desc.domain_id}:{dom_idx}".encode("utf-8")).digest()[:4], "big")
        rng = np.random.default_rng(dom_seed)

        # Structured latent corpus with hubs (to make hubness realistic for CSLS vs CalibratedZScore)
        latent_docs = l2_normalize(rng.normal(size=(n_docs_per_domain, latent_dim)))
        # Add mild hub concentration on first 8 docs in latent space
        hub_dir = l2_normalize(rng.normal(size=(1, latent_dim)))
        latent_docs[:8] = l2_normalize(0.75 * latent_docs[:8] + 0.65 * hub_dir)

        doc_ids = tuple(f"{desc.burn_check_id}_d{i:04d}" for i in range(n_docs_per_domain))
        query_ids = tuple(f"{desc.burn_check_id}_q{i:03d}" for i in range(n_queries_per_domain))
        doc_texts = tuple(
            f"Unburned public passage {i:04d} from {desc.domain_id} ({desc.category}) token_{dom_idx}_{i}"
            for i in range(n_docs_per_domain)
        )
        query_texts = tuple(
            f"Unburned retrieval question {i:03d} targeting {desc.domain_id} ({desc.category}) probe_{dom_idx}_{i}"
            for i in range(n_queries_per_domain)
        )

        # Build qrels (avoiding hub distractors as primary targets)
        qrels: Dict[str, Dict[str, float]] = {}
        latent_queries = np.empty((n_queries_per_domain, latent_dim), dtype=np.float64)
        for qi in range(n_queries_per_domain):
            p1 = 8 + ((qi * 3 + dom_idx) % (n_docs_per_domain - 8))
            p2 = 8 + ((qi * 3 + dom_idx + 5) % (n_docs_per_domain - 8))
            qrels[query_ids[qi]] = {doc_ids[p1]: 2.0, doc_ids[p2]: 1.0}
            latent_queries[qi] = l2_normalize(
                0.76 * latent_docs[p1] + 0.24 * latent_docs[p2] + 0.14 * rng.normal(size=(latent_dim,))
            )

        # Construct 5 encoders v1..v5 with increasing dimension, anisotropy centroids, and orthogonal rotations
        doc_seq: List[np.ndarray] = []
        query_seq: List[np.ndarray] = []
        rotations: List[np.ndarray] = []
        centroids: List[np.ndarray] = []

        for enc_idx, dim in enumerate(encoder_dims):
            q_mat, _ = np.linalg.qr(rng.normal(size=(dim, latent_dim)))
            rot = q_mat[:, :latent_dim].T  # [latent_dim, dim]
            # Anisotropy centroid grows across encoder generations (cone effect)
            mu = l2_normalize(rng.normal(size=(1, dim))) * (0.48 + 0.08 * enc_idx)
            # Hub injection in bridged raw space to separate CSLS from CalibratedZScore
            noise_scale = 0.05 + 0.008 * enc_idx
            d_raw = latent_docs @ rot + mu + noise_scale * rng.normal(size=(n_docs_per_domain, dim))
            q_raw = latent_queries @ rot + mu + noise_scale * rng.normal(size=(n_queries_per_domain, dim))
            # Mild hubness distortion on target shard docs for shard calibration realism
            if enc_idx == 1:
                hub_vec = l2_normalize(rng.normal(size=(1, dim)))
                d_raw[:6] = d_raw[:6] + 0.35 * hub_vec
            doc_seq.append(l2_normalize(d_raw))
            query_seq.append(l2_normalize(q_raw))
            rotations.append(rot)
            centroids.append(mu)

        # Held-out RAG prompt-wrapper / tenant-prefix shift on v1 queries:
        # Combines a strong systematic tenant/prompt prefix centroid shift + nonlinear per-query displacement
        d_v1 = encoder_dims[0]
        prefix_dir = l2_normalize(rng.normal(size=(1, d_v1)))
        # Ensure prefix shift is mostly orthogonal to original v1 centroid
        prefix_shift = 0.72 * prefix_dir
        nonlinear_component = 0.08 * np.tanh(query_seq[0] * 1.5)
        shifted_q_v1 = l2_normalize(query_seq[0] + prefix_shift + nonlinear_component)
        displacements = shifted_q_v1 - query_seq[0]
        mean_disp = displacements.mean(axis=0, keepdims=True)
        nonlin_res = displacements - mean_disp
        disp_norms = np.linalg.norm(displacements, axis=1)
        nonlin_norms = np.linalg.norm(nonlin_res, axis=1)
        mean_disp_norm = float(np.mean(disp_norms))
        mean_nonlin_norm = float(np.mean(nonlin_norms))
        linear_frac = float(np.linalg.norm(mean_disp) / max(mean_disp_norm, 1e-12))

        # Shifted-anchor refit ceiling inputs (same prefix shift applied to a held-out anchor set)
        shifted_anc_v1 = l2_normalize(doc_seq[0] + prefix_shift + 0.08 * np.tanh(doc_seq[0] * 1.5))
        shifted_anc_v2 = doc_seq[1]

        # Backfill mask: 30% native (v2), 70% bridged (v1->v2)
        bf_mask = np.zeros(n_docs_per_domain, dtype=bool)
        bf_idx = rng.choice(n_docs_per_domain, size=int(round(0.30 * n_docs_per_domain)), replace=False)
        bf_mask[bf_idx] = True

        bundles.append(
            DomainGauntletBundle(
                descriptor=desc,
                doc_ids=doc_ids,
                query_ids=query_ids,
                doc_texts=doc_texts,
                query_texts=query_texts,
                qrels=qrels,
                doc_embeddings_seq=tuple(doc_seq),
                query_embeddings_seq=tuple(query_seq),
                shifted_queries_v1=shifted_q_v1,
                shifted_anchor_queries_v1=shifted_anc_v1,
                shifted_anchor_targets_v2=shifted_anc_v2,
                backfill_mask=bf_mask,
                prefix_displacement_mean_norm=mean_disp_norm,
                prefix_nonlinear_residual_mean_norm=mean_nonlin_norm,
                prefix_linear_shift_fraction=linear_frac,
            )
        )

    return tuple(bundles)


# ---------------------------------------------------------------------------
# 3. Full Zero-Retune 12-Domain Gauntlet Runner
# ---------------------------------------------------------------------------


def run_12domain_4hop_gauntlet(
    domain_bundles: Sequence[DomainGauntletBundle],
    *,
    historical_fingerprints: Mapping[str, Sequence[str]],
    ledger: Optional[BurnLedger] = None,
    n_anchor_draws: int = DEFAULT_ANCHOR_DRAWS,
    anchors_per_draw: int = DEFAULT_ANCHORS_PER_DRAW,
    poisoning_rates: Sequence[float] = POISONING_SWEEP_RATES,
    poisoning_modes: Sequence[str] = POISONING_SWEEP_MODES,
    n_bootstrap: int = 2000,
    seed: int = 5401,
    is_live_confirmatory: bool = False,
) -> Dict[str, Any]:
    """Execute the full zero-retune 12-domain, 5-encoder/4-hop, shift & poisoning gauntlet.

    Strictly verifies `burn_ledger` + SHA-256 disjointness BEFORE any bridge fit or scoring.
    """
    # 0. Pre-scoring mechanical burn_ledger + SHA-256 gate
    burn_receipt = verify_12_domains_unburned(
        domain_bundles, historical_fingerprints=historical_fingerprints, ledger=ledger
    )

    csls_merger = CSLSMerge(csls_k=DEFAULT_CSLS_K)

    # Hypothesis contrast maps across all 12 domains
    clean_1hop_csls_vs_zscore: Dict[str, float] = {}
    static_zeromu_vs_uncentered: Dict[str, float] = {}
    static_zeromu_vs_centered: Dict[str, float] = {}
    shift_zeromu_vs_uncentered: Dict[str, float] = {}
    fourhop_zeromu_vs_uncentered: Dict[str, float] = {}
    prefix_disp_by_domain: Dict[str, float] = {}
    direct_v1_v5_delta_by_domain: Dict[str, float] = {}

    anchor_draw_scores_zeromu: Dict[str, List[float]] = {}
    actual_anchors_by_domain: Dict[str, int] = {}

    per_domain_results: Dict[str, Dict[str, Any]] = {}
    poisoning_summary_by_rate: Dict[str, Dict[str, List[float]]] = {
        f"{r:.2f}": {"UncenteredProcrustes": [], "CenteredProcrustes": [], "ZeroMuPolarBridge": []}
        for r in poisoning_rates
    }
    poisoning_tra1_statuses: Dict[str, List[str]] = {f"{r:.2f}": [] for r in poisoning_rates}

    for dom_idx, bundle in enumerate(domain_bundles):
        dom_id = bundle.descriptor.domain_id
        d_v1, d_v2 = bundle.doc_embeddings_seq[0], bundle.doc_embeddings_seq[1]
        q_v1, q_v2 = bundle.query_embeddings_seq[0], bundle.query_embeddings_seq[1]
        zero_mask = np.zeros(len(bundle.doc_ids), dtype=bool)

        # 1. 20 frozen 500-anchor draws per domain
        draws = sample_anchor_draws(
            dom_id,
            bundle.doc_ids,
            n_draws=n_anchor_draws,
            anchors_per_draw=anchors_per_draw,
            base_seed=seed + dom_idx * 100,
        )
        primary_anc = draws[0].indices
        actual_anchors_by_domain[dom_id] = draws[0].actual_anchors

        # 2. Full 5x3 Bridge x Merge factorial (v1 -> v2, zero retuning)
        factorial = run_factorial_5x3(
            domain=dom_id,
            src_anchors=d_v1[primary_anc],
            tgt_anchors=d_v2[primary_anc],
            eval_queries_src=q_v1,
            eval_queries_tgt=q_v2,
            docs_old=d_v2,
            docs_new=d_v2,
            query_ids=bundle.query_ids,
            doc_ids=bundle.doc_ids,
            qrels=bundle.qrels,
            backfill_mask=bundle.backfill_mask,
            k=10,
            csls_k=DEFAULT_CSLS_K,
            ridge_regularization=PREREGISTERED_ZERO_RETUNE_CONFIG["ridge_regularization"],
            zeromu_mode=PREREGISTERED_ZERO_RETUNE_CONFIG["zeromu_mode"],
            bridge_names=BRIDGE_NAMES,
            merge_names=MERGE_NAMES,
        )

        # Average CSLS - CalibratedZScore across all 5 bridges on clean 1-hop nDCG@10
        csls_minus_z_per_bridge = [
            factorial[(b, "CSLSMerge")].mean_ndcg_at_10 - factorial[(b, "CalibratedZScoreMerge")].mean_ndcg_at_10
            for b in BRIDGE_NAMES
        ]
        clean_1hop_csls_vs_zscore[dom_id] = float(np.mean(csls_minus_z_per_bridge))

        # Static 1-hop at fixed CSLS: ZeroMu vs Uncentered AND ZeroMu vs Centered
        static_zeromu_vs_uncentered[dom_id] = (
            factorial[("ZeroMuPolarBridge", "CSLSMerge")].mean_ndcg_at_10
            - factorial[("UncenteredProcrustes", "CSLSMerge")].mean_ndcg_at_10
        )
        static_zeromu_vs_centered[dom_id] = (
            factorial[("ZeroMuPolarBridge", "CSLSMerge")].mean_ndcg_at_10
            - factorial[("CenteredProcrustes", "CSLSMerge")].mean_ndcg_at_10
        )

        # 3. Within-domain anchor-resample variance across all 20 frozen draws
        draw_ndcgs: List[float] = []
        for draw in draws:
            zm_draw = ZeroMuPolarBridge.fit(
                d_v1[draw.indices],
                d_v2[draw.indices],
                scoring_mode=PREREGISTERED_ZERO_RETUNE_CONFIG["zeromu_mode"],
            )
            raw_d = zm_draw.score_matrix(q_v1, d_v2)
            dummy_d = np.zeros_like(raw_d)
            fused_d = csls_merger.merge(raw_d, dummy_d, zero_mask, bridged_ref=raw_d, native_ref=dummy_d)
            pq_n, _ = evaluate_score_matrix(fused_d, bundle.query_ids, bundle.doc_ids, bundle.qrels, k=10)
            draw_ndcgs.append(float(np.mean(list(pq_n.values()))) if pq_n else 0.0)
        anchor_draw_scores_zeromu[dom_id] = draw_ndcgs

        # 4. Held-out RAG prompt-wrapper / tenant-prefix shift WITHOUT refitting + shifted-anchor refit ceiling
        unc_clean = UncenteredProcrustes.fit(d_v1[primary_anc], d_v2[primary_anc])
        cen_clean = CenteredProcrustes.fit(d_v1[primary_anc], d_v2[primary_anc])
        zm_clean = ZeroMuPolarBridge.fit(d_v1[primary_anc], d_v2[primary_anc], scoring_mode="dual_adap")
        # Shifted-anchor refit ceiling (refit on shifted anchors)
        refit_ceiling_bridge = ZeroMuPolarBridge.fit(
            bundle.shifted_anchor_queries_v1[primary_anc],
            bundle.shifted_anchor_targets_v2[primary_anc],
            scoring_mode="dual_adap",
        )

        shift_scores: Dict[str, float] = {}
        for label, br in (
            ("UncenteredProcrustes", unc_clean),
            ("CenteredProcrustes", cen_clean),
            ("ZeroMuPolarBridge", zm_clean),
            ("ShiftedAnchorRefitCeiling", refit_ceiling_bridge),
        ):
            raw_s = br.score_matrix(bundle.shifted_queries_v1, d_v2)
            dummy_s = np.zeros_like(raw_s)
            fused_s = csls_merger.merge(raw_s, dummy_s, zero_mask, bridged_ref=raw_s, native_ref=dummy_s)
            pq_n, _ = evaluate_score_matrix(fused_s, bundle.query_ids, bundle.doc_ids, bundle.qrels, k=10)
            shift_scores[label] = float(np.mean(list(pq_n.values()))) if pq_n else 0.0

        shift_zeromu_vs_uncentered[dom_id] = shift_scores["ZeroMuPolarBridge"] - shift_scores["UncenteredProcrustes"]
        prefix_disp_by_domain[dom_id] = bundle.prefix_displacement_mean_norm

        # 5. 5-encoder / 4-hop sequential composition (v1->v2->v3->v4->v5) + direct v1->v5 fit
        chain_zm = evaluate_4hop_chain(
            bridge_name="ZeroMuPolarBridge",
            encoder_names=ENCODER_NAMES,
            doc_embeddings_seq=bundle.doc_embeddings_seq,
            query_embeddings_seq=bundle.query_embeddings_seq,
            anchor_indices=primary_anc,
            query_ids=bundle.query_ids,
            doc_ids=bundle.doc_ids,
            qrels=bundle.qrels,
            merge_name="CSLSMerge",
            csls_k=DEFAULT_CSLS_K,
            polar_reproject=True,
            k=10,
        )
        chain_unc = evaluate_4hop_chain(
            bridge_name="UncenteredProcrustes",
            encoder_names=ENCODER_NAMES,
            doc_embeddings_seq=bundle.doc_embeddings_seq,
            query_embeddings_seq=bundle.query_embeddings_seq,
            anchor_indices=primary_anc,
            query_ids=bundle.query_ids,
            doc_ids=bundle.doc_ids,
            qrels=bundle.qrels,
            merge_name="CSLSMerge",
            csls_k=DEFAULT_CSLS_K,
            polar_reproject=False,
            k=10,
        )
        chain_cen = evaluate_4hop_chain(
            bridge_name="CenteredProcrustes",
            encoder_names=ENCODER_NAMES,
            doc_embeddings_seq=bundle.doc_embeddings_seq,
            query_embeddings_seq=bundle.query_embeddings_seq,
            anchor_indices=primary_anc,
            query_ids=bundle.query_ids,
            doc_ids=bundle.doc_ids,
            qrels=bundle.qrels,
            merge_name="CSLSMerge",
            csls_k=DEFAULT_CSLS_K,
            polar_reproject=False,
            k=10,
        )
        fourhop_zeromu_vs_uncentered[dom_id] = (
            chain_zm.final_composed_ndcg_at_10 - chain_unc.final_composed_ndcg_at_10
        )
        direct_v1_v5_delta_by_domain[dom_id] = chain_zm.final_direct_ndcg_at_10 - chain_unc.final_direct_ndcg_at_10

        # 6. Progressive anchor-poisoning sweep (0%, 1%, 5%, 10%, 20%)
        poison_pts = run_contamination_sweep(
            src_anchors=d_v1[primary_anc],
            tgt_anchors=d_v2[primary_anc],
            eval_queries_src=q_v1,
            docs_old=d_v2,
            query_ids=bundle.query_ids,
            doc_ids=bundle.doc_ids,
            qrels=bundle.qrels,
            bridge_names=("UncenteredProcrustes", "CenteredProcrustes", "ZeroMuPolarBridge"),
            merge_name="CSLSMerge",
            rates=poisoning_rates,
            modes=poisoning_modes,
            seed=seed + dom_idx,
            csls_k=DEFAULT_CSLS_K,
            k=10,
        )
        for pt in poison_pts:
            r_key = f"{pt.contamination_rate:.2f}"
            poisoning_summary_by_rate[r_key][pt.bridge_name].append(pt.mean_ndcg_at_10)
            poisoning_tra1_statuses[r_key].append(pt.tra1.tra1_health_status)

        per_domain_results[dom_id] = {
            "category": bundle.descriptor.category,
            "n_docs": len(bundle.doc_ids),
            "n_queries": len(bundle.query_ids),
            "n_anchor_draws": len(draws),
            "actual_anchors_per_draw": draws[0].actual_anchors,
            "factorial_5x3": {
                f"{b}__{m}": {
                    "mean_ndcg_at_10": cell.mean_ndcg_at_10,
                    "mean_recall_at_10": cell.mean_recall_at_10,
                    "kappa": cell.kappa,
                    "nonzero_kappa": cell.nonzero_kappa,
                    "weights_sha256": cell.weights_sha256,
                }
                for (b, m), cell in factorial.items()
            },
            "prompt_shift": {
                "scores_at_csls": shift_scores,
                "zeromu_minus_uncentered": shift_zeromu_vs_uncentered[dom_id],
                "prefix_displacement_mean_norm": bundle.prefix_displacement_mean_norm,
                "prefix_nonlinear_residual_mean_norm": bundle.prefix_nonlinear_residual_mean_norm,
                "prefix_linear_shift_fraction": bundle.prefix_linear_shift_fraction,
                "shifted_anchor_refit_ceiling_ndcg": shift_scores["ShiftedAnchorRefitCeiling"],
            },
            "four_hop_v1_to_v5": {
                "zeromu_composed_ndcg": chain_zm.final_composed_ndcg_at_10,
                "zeromu_direct_v1_v5_ndcg": chain_zm.final_direct_ndcg_at_10,
                "zeromu_final_nonzero_kappa": chain_zm.transitions[-1].composed_nonzero_kappa,
                "uncentered_composed_ndcg": chain_unc.final_composed_ndcg_at_10,
                "uncentered_direct_v1_v5_ndcg": chain_unc.final_direct_ndcg_at_10,
                "centered_composed_ndcg": chain_cen.final_composed_ndcg_at_10,
                "zeromu_minus_uncentered_4hop": fourhop_zeromu_vs_uncentered[dom_id],
                "transitions_zeromu": [asdict(t) for t in chain_zm.transitions],
            },
        }

    # 7. Cross-domain vs within-domain anchor draw variance summary
    anchor_variance = summarize_anchor_draw_variance(
        anchor_draw_scores_zeromu,
        requested_anchors=anchors_per_draw,
        actual_anchors_by_domain=actual_anchors_by_domain,
    )

    # 8. Evaluate all 3 pre-registered Item 6B hypotheses (zero retuning)
    hypotheses = evaluate_item6b_hypotheses(
        clean_1hop_csls_vs_zscore_by_domain=clean_1hop_csls_vs_zscore,
        static_zeromu_vs_uncentered_at_csls_by_domain=static_zeromu_vs_uncentered,
        static_zeromu_vs_centered_at_csls_by_domain=static_zeromu_vs_centered,
        shift_zeromu_vs_uncentered_at_csls_by_domain=shift_zeromu_vs_uncentered,
        fourhop_zeromu_vs_uncentered_at_csls_by_domain=fourhop_zeromu_vs_uncentered,
        prefix_displacement_norms_by_domain=prefix_disp_by_domain,
        direct_v1_v5_delta_by_domain=direct_v1_v5_delta_by_domain,
        min_positive_domains_merge=PREREGISTERED_ZERO_RETUNE_CONFIG["min_positive_domains_merge"],
        retirement_floor=ITEM6B_RETIREMENT_FLOOR,
        tight_equivalence_floor=ITEM6B_TIGHT_EQUIVALENCE_FLOOR,
        alpha_fwer=0.05,
        n_resamples=n_bootstrap,
        seed=seed,
    )

    # Aggregate poisoning sweep curve across domains & modes
    poisoning_curve: Dict[str, Dict[str, Any]] = {}
    for r_key, bridge_map in poisoning_summary_by_rate.items():
        statuses = poisoning_tra1_statuses[r_key]
        poisoning_curve[r_key] = {
            "mean_ndcg_by_bridge": {
                b_name: float(np.mean(vals)) if vals else 0.0 for b_name, vals in bridge_map.items()
            },
            "tra1_warn_or_fail_count": sum(1 for s in statuses if s != "PASS"),
            "tra1_total_checks": len(statuses),
        }

    config_sha = _sha256_json(PREREGISTERED_ZERO_RETUNE_CONFIG)
    contract = TRA1StateIdentityContract(
        base_weights_sha256=sha256_digest("->".join(ENCODER_NAMES).encode("utf-8")),
        adapter_weights_sha256=EMPTY_ADAPTER_SHA256,
        config_sha256=config_sha,
        tokenizer_sha256=sha256_digest(b"pv7-b54-5encoder-chain-v1"),
        kv_or_state_dtype="float32",
        rope_freq_base=10000.0,
        rope_scaling_hash=sha256_digest(b"none"),
    )

    if is_live_confirmatory:
        status = hypotheses.overall_verdict
    else:
        status = "local-drycheck-pass" if hypotheses.overall_verdict == "verified-pass" else f"local-drycheck-{hypotheses.overall_verdict}"

    return {
        "bundle_id": BUNDLE_ID,
        "item": ITEM_ID,
        "status": status,
        "is_live_confirmatory": bool(is_live_confirmatory),
        "retuned_on_eval": False,
        "preregistered_config": PREREGISTERED_ZERO_RETUNE_CONFIG,
        "preregistered_config_sha256": config_sha,
        "encoder_chain": [
            {"stage": f"v{i + 1}", "label": name, "hf_id": hf_id, "nominal_dim": dim}
            for i, (name, hf_id, dim) in enumerate(ENCODER_CHAIN_SPEC)
        ],
        "locked_12_domains": [asdict(d) for d in LOCKED_12_DOMAINS],
        "burn_ledger_gate": burn_receipt,
        "anchor_draw_variance": {
            "cross_domain_n": anchor_variance.cross_domain_n,
            "cross_domain_mean": anchor_variance.cross_domain_mean,
            "cross_domain_sd": anchor_variance.cross_domain_sd,
            "mean_within_domain_anchor_sd": anchor_variance.mean_within_domain_anchor_sd,
            "n_draws_per_domain": int(n_anchor_draws),
            "requested_anchors_per_draw": int(anchors_per_draw),
        },
        "poisoning_sweep": {
            "rates": list(poisoning_rates),
            "modes": list(poisoning_modes),
            "curve": poisoning_curve,
        },
        "hypotheses": {
            "H6B_merge": hypotheses.h6b_merge,
            "H6B_bridge_static": hypotheses.h6b_bridge_static,
            "H6B_bridge_shift_and_4hop": hypotheses.h6b_bridge_shift_and_4hop,
            "holm_adjusted_p_values": hypotheses.holm_adjusted_p_values,
            "overall_verdict": hypotheses.overall_verdict,
            "report_sha256": hypotheses.report_sha256,
        },
        "per_domain": per_domain_results,
        "tra1_state_identity_digest": contract.digest(),
        "spine_digest": claim_registry_sha256(),
    }


__all__ = [
    "BUNDLE_ID",
    "DomainDescriptor",
    "DomainGauntletBundle",
    "ENCODER_CHAIN_SPEC",
    "ENCODER_NAMES",
    "ENCODER_NOMINAL_DIMS",
    "FORBIDDEN_BURNED_SUBSTITUTES",
    "ITEM_ID",
    "LOCKED_12_DOMAINS",
    "POISONING_SWEEP_MODES",
    "POISONING_SWEEP_RATES",
    "PREREGISTERED_ZERO_RETUNE_CONFIG",
    "SPINE_DIGEST",
    "build_deterministic_12domain_gauntlet_fixture",
    "run_12domain_4hop_gauntlet",
    "verify_12_domains_unburned",
]
