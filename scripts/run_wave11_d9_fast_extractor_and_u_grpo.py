r"""Wave 11 Demo D9 (Guardrail `L4`):
Part A — Fast Disjoint `0.5B/1.5B` LoRA Claim Extractor + `AuthorityDominatedExocortexV2` (`Suite A N=60`)
Part B — Pre-Sealed Virgin-Holdout (`EVAL_VIRGIN_V2 N=120`) + Curriculum Anti-Collapse `Epistemic-WFS-GRPO` (`7B`, 3 seeds)

CLI Modes:
  - `--mode verify-splits`:
      Fast CPU verification of Part A synthetic corpus disjointness (`0` overlap, `< 0.25` 5-gram Jaccard vs
      all 1,020 Suite A docs + `_ECHO_VARIATION_TEMPLATES`), Part B 5-way RLVR split disjointness
      (`SFT_WARMUP N=128`, `GRPO_RL N=240`, `DEV N=80`, `EVAL_WAVE6_LOOK2 N=120`, `EVAL_VIRGIN_V2 N=120`,
      plus `D1 N=60`), byte-identical Wave-6 `EVAL` SHA-256 lock, 3-way `audit_episode` zero-mismatch proof,
      untruncated 4-step prompt-only CoT accuracy, and anti-collapse reward math (`E[R_always_abstain] < 0`).
  - `--mode run-part-a`:
      Local GPU pipeline (Desktop RTX 3090): trains `Qwen/Qwen2.5-0.5B-Instruct` & `Qwen/Qwen2.5-1.5B-Instruct`
      completion-only LoRA (`r=16, alpha=32`, `merge_and_unload()` bf16 parity `< 1e-4`), selects the best
      checkpoint strictly on synthetic `DEV (N=120)` BEFORE touching `Suite A (N=60)`, then runs batched
      (`batch_size=17`) persistent GPU extraction -> `AuthorityDominatedExocortexV2` -> `Nemotron-30B` reader ->
      `BlindCrossFamilySemanticJudge` (`Qwen2.5-7B`) AND deterministic exact-match, computing paired bootstrap
      `LCB97.5`, Tango, and Clopper-Pearson CIs vs live `Mem0 81.67%` and Wave-6 `30B Extractor 96.67%`.
  - `--mode run-part-b`:
      Spark #2 (`.venv-rlvr`) GPU pipeline (`Protocol v2`, pre-registered in `docs/research/d9b_amendment_v1.json`):
      trains completion-only SFT (`N=128`, untruncated 4-step WFS CoT) -> `merge_and_unload()` reload parity `< 1e-4` ->
      3-seed confirmatory `(137, 2026, 31415)` `Epistemic-WFS-GRPO` (`seed_42` retired as DEV-diagnostic;
      `max_steps=60, grad_accum=8, batch=8, num_gen=8, lora_r=16, lora_alpha=32, lr=5e-6, beta=0.05,
      scale_rewards="none", loss_type="dr_grpo", R_false_abstain=R_wrong_det=-1.00, R_false_commit_u=-1.15`),
      verifies `float32` `merge_and_unload()` parity `< 1e-4` across 8 DEV probes, scores ONLY `DEV` inside the
      training loop, freezes `merged_sft_dir` + all GRPO adapter SHA-256s in `selection_frozen.json`, and then
      executes a single digest-verified post-freeze evaluation of `Always_Abstain_Baseline`, `Base 7B`, `SFT 7B`,
      `GRPO 7B` (fixed `final_step@60` 3-seed mean PRIMARY with 2-seed `(137, 2026)` mean reported alongside;
      `best_dev` descriptive SECONDARY), and `Bare Nemotron-30B` on `EVAL_VIRGIN_V2 (PRIMARY, virgin for GRPO arm only)`,
      `EVAL_WAVE6_LOOK2 (Look #2)`, and `D1 (N=60)` with `paired_bootstrap_wave6`, `holm_bonferroni_stepdown`,
      and `T/F` non-inferiority gate.
  - `--mode assemble-manifest`:
      Combines Part A + Part B JSON outputs & CSVs into
      `artifacts/wave11_d9_fast_extractor_and_u_grpo/wave11_d9_extractor_grpo_manifest.json` + `report_receipt.json`.
"""

from __future__ import annotations

import os

if os.name != "nt":
    os.environ.setdefault("HF_HOME", "/home/mattmre/chelated-night1/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import argparse  # noqa: E402
import concurrent.futures  # noqa: E402
import contextlib  # noqa: E402
import csv  # noqa: E402
import gc  # noqa: E402
import hashlib  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter, defaultdict  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple  # noqa: E402

import numpy as np  # noqa: E402
import requests  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import ek5_wfs_rlvr_env as rlvr_env  # noqa: E402
from chelated_epistemic_proposer_v2 import AuthorityDominatedExocortexV2  # noqa: E402
from chelated_fast_edge_extractor import (  # noqa: E402
    FastEdgeClaimExtractor,
    assert_extractor_corpus_disjoint_from_suite_a,
    assert_train_dev_disjoint,
    build_disjoint_extractor_splits,
    train_fast_claim_extractor_lora,
)
from chelated_raw_text_exocortex import RawDocumentEnvelope  # noqa: E402
from ek5_v3_stratified import (  # noqa: E402
    DEP_ALL_TYPES,
    DEP_CUR_TYPES,
    StratifiedEvent,
    compute_alternating_wfs_oracle,
)
from ek5_wfs_rlvr_env import (  # noqa: E402
    ABSTAIN,
    LABEL_F,
    LABEL_T,
    LABEL_U,
    EpistemicEpisode,
    _SlotCtx,
    audit_episode,
    build_episode_store,
)
from prospective_realworld_drift_corpus import (  # noqa: E402
    build_prospective_realworld_drift_suite,
)
from scripts.night1_common import (  # noqa: E402
    FastBM25Index,
    current_git_sha,
    write_json_atomic,
)
from scripts.run_wave6_d4_trained_epistemic_grpo import (  # noqa: E402
    D1_PROSE_TRANSFER_SYSTEM_PROMPT,
    EXTRA_SCENARIOS as WAVE6_EXTRA_SCENARIOS,
    SPLIT_DISTRACTOR_RANGE as WAVE6_SPLIT_DISTRACTOR_RANGE,
    SPLIT_RETRACTOR_SPECS as WAVE6_SPLIT_RETRACTOR_SPECS,
    SPLIT_SCENARIOS as WAVE6_SPLIT_SCENARIOS,
    SPLIT_SEEDS as WAVE6_SPLIT_SEEDS,
    SPLIT_SLOT_POOLS as WAVE6_SPLIT_SLOT_POOLS,
    _HEADER_TARGET_RE,
    _EV_LINE_RE,
    _PROPOSED_LINE_RE,
    _REL_LINE_RE,
    _VERB_TO_EDGE,
    _kosaraju_sccs,
    _summarize_d1_completions,
    build_d1_matched_evidence_prompts,
    score_d1_prose_prediction,
)
from spherical_migration_bridge import l2_normalize  # noqa: E402
from wave6_common import (  # noqa: E402
    BlindCrossFamilySemanticJudge,
    holm_bonferroni_stepdown,
    paired_bootstrap_wave6,
)

# ---------------------------------------------------------------------------
# L4 Constants, Pre-Registered Seeds, & Anti-Collapse Reward Spec
# ---------------------------------------------------------------------------
DEFAULT_OUTPUT_DIR = REPO_ROOT / "artifacts" / "wave11_d9_fast_extractor_and_u_grpo"
DEFAULT_OSV_CACHE = (
    REPO_ROOT / "artifacts" / "phase2_d1_4node_local_fleet_proof" / "_work" / "osv_cve_seeds.json"
)
DEFAULT_WAVE6_PM3_MANIFEST = (
    REPO_ROOT / "artifacts" / "wave6_pm3_v2_competitor_arena" / "wave6_pm3_v2_arena_manifest.json"
)
DEFAULT_WAVE6_PM3_ROWS_CSV = (
    REPO_ROOT / "artifacts" / "wave6_pm3_v2_competitor_arena" / "wave6_pm3_v2_arena_rows.csv"
)

WAVE6_EVAL_EXPECTED_SHA256 = "6b08233ce70b69fdd3ec5e1fabfbda8b0934e37acc48b4db96ccb89df078a4a3"
DEFAULT_D9B_AMENDMENT_PATH = REPO_ROOT / "docs" / "research" / "d9b_amendment_v1.json"

# Protocol v2 Confirmatory Seeds (`seed_42` retired as DEV-diagnostic seed)
RETIRED_DEV_DIAGNOSTIC_SEED: int = 42
PREREGISTERED_SEEDS: Tuple[int, ...] = (137, 2026, 31415)
PRIMARY_TWO_SEED_SUBSET: Tuple[int, ...] = (137, 2026)

# Protocol v2 Bundled Hyperparameters (zero single-knob causal attribution claimed)
PROTOCOL_V2_VERSION = "d9b_protocol_v2"
PROTOCOL_V2_GRPO_STEPS = 60
PROTOCOL_V2_LR = 5e-6
PROTOCOL_V2_BETA = 0.05
PROTOCOL_V2_LORA_R = 16
PROTOCOL_V2_LORA_ALPHA = 32
PROTOCOL_V2_BATCH_SIZE = 8
PROTOCOL_V2_GRAD_ACCUM = 8
PROTOCOL_V2_NUM_GENERATIONS = 8
PROTOCOL_V2_TEMPERATURE = 1.0
PROTOCOL_V2_SCALE_REWARDS = "none"
PROTOCOL_V2_LOSS_TYPE = "dr_grpo"

WAVE11_SPLIT_SIZES: Dict[str, int] = {
    "SFT_WARMUP": 128,
    "GRPO_RL": 240,
    "DEV": 80,
    "EVAL_WAVE6_LOOK2": 120,
    "EVAL_VIRGIN_V2": 120,
}
MAX_PROMPT_LENGTH = 1664
MAX_COMPLETION_LENGTH = 384

# Protocol v2 Anti-Collapse 3-Valued Reward (L4):
# Setting FALSE_ABSTAIN_REWARD == WRONG_DETERMINATE_REWARD == -1.00 guarantees:
#   1. On balanced 1/3 T, 1/3 F, 1/3 U:
#      E[R_always_abstain] = (1/3)*(+1.05) + (2/3)*(-1.00) = -0.95/3 = -0.316667 < 0!
#   2. On any 0-correct determinate (T/F) rollout group mixing ABSTAIN and wrong answers,
#      every rollout ties at -1.00 (group std == 0.0 -> zero abstention gradient!).
EXACT_CORE_REWARD = 1.00
FORMAT_BONUS_REWARD = 0.05
PARTIAL_VERDICT_MISMATCH_REWARD = 0.75
FALSE_ABSTAIN_REWARD = -1.00
FALSE_COMMIT_ON_U_REWARD = -1.15
WRONG_DETERMINATE_REWARD = -1.00
DETERMINATE_NON_INFERIORITY_MARGIN = 0.03

GRPO_SYSTEM_PROMPT_L4 = (
    "You are an epistemic reasoning model grounded in 3-valued Well-Founded Semantics (T=Admitted, "
    "F=Refuted/Superseded, U=Contested). Given the multi-agent evidence log, reason systematically "
    "inside <think>...</think> across 4 steps: (1) raw claims & D_V effective authority ceilings, "
    "(2) qualified REFUTE/SUPERSEDE/CITES_AUDIT edges, (3) alternating WFS fixpoint & cycle/contested-parent "
    "propagation, (4) canonical verdict. Then output EXACTLY two lines after </think>:\n"
    "VERDICT: T | F | U\n"
    "ANSWER: <exact unique current value> | ABSTAIN"
)

ENTERPRISE_READER_SYSTEM = (
    "You are an enterprise verification assistant. Answer the question concisely using ONLY the provided records.\n"
    "- Higher-authority official filings, signed advisories, and postmortems supersede preliminary notes and unverified rumors.\n"
    "- If equal-authority official sources mutually conflict with no authoritative resolution (CONTESTED), or if the "
    "verified answer is absent from the records, reply ONLY: ABSTAIN."
)

_THINK_CLOSED_RE = re.compile(r"^\s*<think>(.+?)</think>\s*(.+)$", re.IGNORECASE | re.DOTALL)
_VERDICT_LINE_RE = re.compile(r"^\s*VERDICT\s*:\s*(T|F|U)\s*$", re.IGNORECASE | re.MULTILINE)
_ANSWER_LINE_RE = re.compile(r"^\s*ANSWER\s*:\s*([^\r\n]+?)\s*$", re.IGNORECASE | re.MULTILINE)
_NEGATION_RE = re.compile(
    r"\b(?:not|never|no\s+longer|other\s+than|except|cannot\s+be|isn't|is\s+not|neither|nor)\b",
    re.IGNORECASE,
)
_CLOSED_ABSTAIN_TOKENS = {"abstain"}
_NON_CANONICAL_REFUSALS = {
    "unknown",
    "unresolved",
    "none",
    "n/a",
    "contested",
    "undetermined",
    "unsure",
    "cannot determine",
    "null",
}
_FORBIDDEN_ORACLE_COT_SUBSTRINGS = (
    "store_slot_status",
    "store_slot_reason",
    "store_query_reason",
    "unique_admitted",
    "audit_only",
    "slot_contested",
)


# ---------------------------------------------------------------------------
# Pre-Sealed Virgin Holdout (`SLOT_POOL_VIRGIN_V2`) + New Disjoint Topologies
# ---------------------------------------------------------------------------
SLOT_POOL_VIRGIN_V2: Tuple[Tuple[str, str, Tuple[str, ...]], ...] = (
    (
        "ingress-mesh-gw",
        "rate_limit_profile",
        (
            "rl-500rps-burst",
            "rl-1200rps-steady",
            "rl-250rps-strict",
            "rl-5000rps-bypass",
            "rl-800rps-adaptive",
            "rl-100rps-lockdown",
        ),
    ),
    (
        "telemetry-bus",
        "export_cipher_suite",
        (
            "chacha20-poly1305",
            "aes256-gcm-sha384",
            "aes128-gcm-sha256",
            "null-plaintext-debug",
            "xchacha20-ietf-v2",
            "camellia256-cbc-legacy",
        ),
    ),
    (
        "ledger-attestor",
        "quorum_threshold",
        (
            "quorum-5of7-hsm",
            "quorum-3of5-hsm",
            "quorum-2of3-soft",
            "quorum-1of1-single",
            "quorum-7of9-cold",
            "quorum-4of6-mpc",
        ),
    ),
    (
        "artifact-mirror",
        "sbom_signing_mode",
        (
            "slsa-l3-enforced",
            "slsa-l2-advisory",
            "unsigned-exempt",
            "in-toto-strict-v1",
            "cosign-keyless-oidc",
            "gpg-legacy-v3",
        ),
    ),
)

WAVE11_SPLIT_SLOT_POOLS: Dict[str, Tuple[Tuple[str, str, Tuple[str, ...]], ...]] = {
    "SFT_WARMUP": WAVE6_SPLIT_SLOT_POOLS["SFT_WARMUP"],
    "GRPO_RL": WAVE6_SPLIT_SLOT_POOLS["GRPO_RL"],
    "DEV": WAVE6_SPLIT_SLOT_POOLS["DEV"],
    "EVAL_WAVE6_LOOK2": WAVE6_SPLIT_SLOT_POOLS["EVAL"],
    "EVAL_VIRGIN_V2": SLOT_POOL_VIRGIN_V2,
}

WAVE11_SPLIT_RETRACTOR_SPECS: Dict[str, Tuple[str, str, str]] = {
    "SFT_WARMUP": WAVE6_SPLIT_RETRACTOR_SPECS["SFT_WARMUP"],
    "GRPO_RL": WAVE6_SPLIT_RETRACTOR_SPECS["GRPO_RL"],
    "DEV": WAVE6_SPLIT_RETRACTOR_SPECS["DEV"],
    "EVAL_WAVE6_LOOK2": WAVE6_SPLIT_RETRACTOR_SPECS["EVAL"],
    "EVAL_VIRGIN_V2": (
        "noise-retractor-virgin-v2",
        "retract_notes_virgin_v2",
        "withdrawn-virgin-v2-marker",
    ),
}

WAVE11_SPLIT_SEEDS: Dict[str, int] = {
    "SFT_WARMUP": WAVE6_SPLIT_SEEDS["SFT_WARMUP"],
    "GRPO_RL": WAVE6_SPLIT_SEEDS["GRPO_RL"],
    "DEV": WAVE6_SPLIT_SEEDS["DEV"],
    "EVAL_WAVE6_LOOK2": WAVE6_SPLIT_SEEDS["EVAL"],
    "EVAL_VIRGIN_V2": 550505,
}

WAVE11_SPLIT_DISTRACTOR_RANGE: Dict[str, Tuple[int, int]] = {
    "SFT_WARMUP": WAVE6_SPLIT_DISTRACTOR_RANGE["SFT_WARMUP"],
    "GRPO_RL": WAVE6_SPLIT_DISTRACTOR_RANGE["GRPO_RL"],
    "DEV": WAVE6_SPLIT_DISTRACTOR_RANGE["DEV"],
    "EVAL_WAVE6_LOOK2": WAVE6_SPLIT_DISTRACTOR_RANGE["EVAL"],
    "EVAL_VIRGIN_V2": (0, 2),
}


# --- New Disjoint Training Motifs (`SFT_WARMUP` & `GRPO_RL`) ---
def _sc_derived_ceiling_refute(s: _SlotCtx, want: str) -> str:
    """SFT motif: nominal tier-4 claim derives from tier-1 root (`eff=1`), tries to refute standing tier-2/3 truth (blocked)."""
    rng, b = s.b.rng, s.b
    truth_v, trap_v = s.take(2)
    poison = s.ev(trap_v, 1, role="external-web-scraper")
    laund = s.ev(trap_v, 4, role="security-auditor")
    b.add_edge(laund, poison, rng.choice(("DERIVES_CURRENT", "COPY_OF")))
    truth = s.ev(truth_v, rng.choice((2, 3)), role="sre-operator")
    b.add_edge(truth, poison, "REFUTE")
    b.add_edge(laund, truth, "REFUTE")  # blocked by eff_tier=1 < 2/3
    if rng.random() < 0.5:
        echo = s.ev(trap_v, 3, role="sre-operator")
        b.add_edge(echo, laund, "COPY_OF")
        b.add_edge(echo, truth, "REFUTE")  # also blocked (eff_tier=1)
    return truth if want == LABEL_T else laund


def _sc_five_node_refutation_ring(s: _SlotCtx, want: str) -> str:
    """GRPO motif (`U`): 5-node odd refutation cycle at equal authority tier (disjoint from EVAL's 3-node cycle)."""
    _ = want
    rng, b = s.b.rng, s.b
    tier = rng.choice((2, 3, 4))
    ps = [s.ev(v, tier) for v in s.take(5)]
    for i in range(5):
        b.add_edge(ps[i], ps[(i + 1) % 5], "REFUTE")
    return rng.choice(ps)


def _sc_derived_contested_chain(s: _SlotCtx, want: str) -> str:
    """GRPO motif (`U`): downstream `DERIVES_CURRENT` chain inheriting `val=U/cur=U` from deadlocked peers."""
    _ = want
    rng, b = s.b.rng, s.b
    v1, v2, v3 = s.take(3)
    tier = rng.choice((2, 3))
    p1 = s.ev(v1, tier)
    p2 = s.ev(v2, tier)
    b.add_edge(p1, p2, "REFUTE")
    b.add_edge(p2, p1, "REFUTE")
    child = s.ev(v3, min(4, tier + 1))
    b.add_edge(child, p1, "DERIVES_CURRENT")
    cands = [child, p1, p2]
    if rng.random() < 0.5:
        grandchild = s.ev(v3, min(4, tier + 1))
        b.add_edge(grandchild, child, "DERIVES_CURRENT")
        cands.append(grandchild)
    if rng.random() < 0.5:
        sup = s.ev(v1, 1)
        b.add_edge(sup, p1, "SUPPORT")
    return rng.choice(cands)


# --- Brand-New Unseen Topological Scenarios (`EVAL_VIRGIN_V2`) ---
def _sc_three_tier_withdrawn_override(s: _SlotCtx, want: str) -> str:
    """Virgin V2 (`T/F`): 3-tier chain + derived decoy blocked by eff_tier ceiling + tier-4 CITES_AUDIT override."""
    rng, b = s.b.rng, s.b
    v1, v2, v3, trap_v = s.take(4)
    poison = s.ev(trap_v, 1, role="external-web-scraper")
    e1 = s.ev(v1, 2, role="triage-summarizer")
    e2 = s.ev(v2, 3, role="sre-operator")
    b.add_edge(e2, e1, "SUPERSEDE")
    decoy = s.ev(v1, 3, role="sre-operator")
    b.add_edge(decoy, e1, "DERIVES_CURRENT")  # eff_tier=2, cur cascades F from e1
    b.add_edge(decoy, e2, "REFUTE")  # blocked: eff_tier(decoy)=2 < 3
    aud = s.ev(v3, 4, domain="general", role="security-auditor")
    b.add_edge(aud, poison, "CITES_AUDIT")
    b.add_edge(aud, poison, "REFUTE")
    b.add_edge(aud, e2, "SUPERSEDE")
    b.add_edge(poison, aud, "REFUTE")  # blocked
    return aud if want == LABEL_T else rng.choice((decoy, e2, poison))


def _sc_cascading_snapshot_vs_current(s: _SlotCtx, want: str) -> str:
    """Virgin V2 (`T/F`): bitemporal cutoff around a `DERIVES_CURRENT` child whose parent is later superseded."""
    rng, b = s.b.rng, s.b
    v1, v2 = s.take(2)
    t1 = b.tx + 1
    base = s.ev(v1, 2, tx=t1, role="triage-summarizer")
    t2 = t1 + rng.randint(1, 2)
    child = s.ev(v1, 3, tx=t2, role="sre-operator")
    b.add_edge(child, base, "DERIVES_CURRENT")
    t3 = t2 + rng.randint(3, 5)
    newer = s.ev(v2, 3, tx=t3, role="sre-operator")
    b.add_edge(newer, base, "SUPERSEDE")
    cut = rng.randint(t2, t3 - 1) if want == LABEL_T else rng.randint(t3, t3 + 3)
    s.as_of = (cut, cut)
    return child


def _sc_diamond_peer_contest(s: _SlotCtx, want: str) -> str:
    """Virgin V2 (`U`): dual deadlocked peers each feeding a derived consumer (`DERIVES_CURRENT`) -> diamond `U`."""
    _ = want
    rng, b = s.b.rng, s.b
    v1, v2, v3, v4 = s.take(4)
    tier = rng.choice((2, 3))
    p1 = s.ev(v1, tier)
    p2 = s.ev(v2, tier)
    b.add_edge(p1, p2, "REFUTE")
    b.add_edge(p2, p1, "REFUTE")
    d1 = s.ev(v3, min(4, tier + 1))
    d2 = s.ev(v4, min(4, tier + 1))
    b.add_edge(d1, p1, "DERIVES_CURRENT")
    b.add_edge(d2, p2, "DERIVES_CURRENT")
    if rng.random() < 0.5:
        sup = s.ev(v3, 1)
        b.add_edge(sup, d1, "SUPPORT")
    return rng.choice((d1, d2, p1, p2))


def _sc_mutual_cycle_with_observer(s: _SlotCtx, want: str) -> str:
    """Virgin V2 (`U`): 3-node refutation ring superseding baseline alongside an unattacked tier-4 `CITES_AUDIT` observer."""
    _ = want
    rng, b = s.b.rng, s.b
    v0, v1, v2, v3, v_obs = s.take(5)
    base = s.ev(v0, 2, role="triage-summarizer")
    ps = [s.ev(v, 3, role="sre-operator") for v in (v1, v2, v3)]
    for i in range(3):
        b.add_edge(ps[i], ps[(i + 1) % 3], "REFUTE")
    b.add_edge(ps[0], base, "SUPERSEDE")
    obs = s.ev(v_obs, 4, domain="general", role="security-auditor")
    b.add_edge(obs, ps[0], "CITES_AUDIT")
    return rng.choice((obs, ps[0], ps[1], base))


WAVE11_EXTRA_SCENARIOS: Dict[str, Callable[[_SlotCtx, str], str]] = {
    **WAVE6_EXTRA_SCENARIOS,
    "derived_ceiling_refute": _sc_derived_ceiling_refute,
    "five_node_refutation_ring": _sc_five_node_refutation_ring,
    "derived_contested_chain": _sc_derived_contested_chain,
    "three_tier_withdrawn_override": _sc_three_tier_withdrawn_override,
    "cascading_snapshot_vs_current": _sc_cascading_snapshot_vs_current,
    "diamond_peer_contest": _sc_diamond_peer_contest,
    "mutual_cycle_with_observer": _sc_mutual_cycle_with_observer,
}

WAVE11_SPLIT_SCENARIOS: Dict[str, Tuple[Tuple[str, Tuple[str, ...]], ...]] = {
    "SFT_WARMUP": (
        ("conflict_refute", (LABEL_T, LABEL_F)),
        ("supersede_chain", (LABEL_T, LABEL_F)),
        ("derived_ceiling_refute", (LABEL_T, LABEL_F)),
        ("peer_deadlock", (LABEL_U,)),
        ("silent_multi_current", (LABEL_U,)),
    ),
    "GRPO_RL": (
        ("sybil_corroboration", (LABEL_T, LABEL_F)),
        ("laundered_supersede_takeover", (LABEL_T, LABEL_F)),
        ("audit_cite_chain_override", (LABEL_T, LABEL_F)),
        ("asof_postcutoff_reversal", (LABEL_T, LABEL_F)),
        ("laundered_peer_standoff", (LABEL_U,)),
        ("dual_branch_unlinked_fork", (LABEL_U,)),
        ("five_node_refutation_ring", (LABEL_U,)),
        ("derived_contested_chain", (LABEL_U,)),
    ),
    "DEV": WAVE6_SPLIT_SCENARIOS["DEV"],
    "EVAL_WAVE6_LOOK2": WAVE6_SPLIT_SCENARIOS["EVAL"],
    "EVAL_VIRGIN_V2": (
        ("three_tier_withdrawn_override", (LABEL_T, LABEL_F)),
        ("cascading_snapshot_vs_current", (LABEL_T, LABEL_F)),
        ("diamond_peer_contest", (LABEL_U,)),
        ("mutual_cycle_with_observer", (LABEL_U,)),
    ),
}


def _rewrite_wave11_noise_retractor(ep: EpistemicEpisode, split_name: str) -> EpistemicEpisode:
    ret_ent, ret_slot, ret_val = WAVE11_SPLIT_RETRACTOR_SPECS[split_name]
    changed = False
    new_events: List[StratifiedEvent] = []
    roles_map = dict(ep.roles)
    for ev in ep.events:
        if ev.entity == "noise-retractor":
            changed = True
            role = roles_map.get(ev.event_id, "compliance-auditor")
            new_events.append(
                StratifiedEvent(
                    event_id=ev.event_id,
                    entity=ret_ent,
                    slot=ret_slot,
                    claim_value=ret_val,
                    text=f"{role} (tier {ev.authority_tier}, domain {ev.domain}) reports {ret_ent}.{ret_slot} = {ret_val}",
                    valid_time=ev.valid_time,
                    tx_time=ev.tx_time,
                    tenant=ev.tenant,
                    domain=ev.domain,
                    authority_tier=ev.authority_tier,
                    valid_end=ev.valid_end,
                )
            )
        else:
            new_events.append(ev)
    if not changed:
        return ep
    withdrawn = set(ep.withdrawn_attestation_ids)
    visible_atts = [a for a in ep.attestations if a.attestation_id not in withdrawn]
    new_prompt = rlvr_env._render_prompt(
        new_events,
        visible_atts,
        roles_map,
        ep.entity,
        ep.slot,
        ep.as_of,
        ep.query_event_id,
        ep.proposed_value,
    )
    rebuilt = EpistemicEpisode(**{**ep.__dict__, "events": tuple(new_events), "prompt": new_prompt})
    _, view = build_episode_store(rebuilt)
    res = view.resolve_slot(rlvr_env.TENANT, rebuilt.entity, rebuilt.slot)
    if res["status"] != rebuilt.store_slot_status:
        raise RuntimeError(f"Retractor rename altered store status in {rebuilt.episode_id}")
    return rebuilt


@contextlib.contextmanager
def _scoped_wave11_split(split_name: str) -> Iterator[List[str]]:
    orig_pool = rlvr_env.SLOT_POOL
    orig_fams = rlvr_env.FAMILIES
    orig_dist = rlvr_env._DISTRACTOR_SCENARIOS
    orig_scenarios = dict(rlvr_env._SCENARIOS)
    orig_add_distractors = rlvr_env._add_distractors

    split_fams: List[Tuple[str, str, str]] = []
    schedule_names: List[str] = []
    det_fams: List[str] = []
    u_fams: List[str] = []
    for sc, labs in WAVE11_SPLIT_SCENARIOS[split_name]:
        for lab in labs:
            fname = f"{sc}:{lab}"
            split_fams.append((fname, sc, lab))
            if lab == LABEL_U:
                u_fams.append(fname)
            else:
                det_fams.append(fname)

    t_fams = [f for f in det_fams if f.endswith(":T")]
    f_fams = [f for f in det_fams if f.endswith(":F")]
    period = max(len(t_fams), len(f_fams), len(u_fams), 1)
    for k in range(period):
        if t_fams:
            schedule_names.append(t_fams[k % len(t_fams)])
        if f_fams:
            schedule_names.append(f_fams[k % len(f_fams)])
        if u_fams:
            schedule_names.append(u_fams[k % len(u_fams)])

    asof_scenarios = {"bitemporal_as_of", "asof_postcutoff_reversal", "cascading_snapshot_vs_current"}
    split_dist = tuple(sc for sc, _labs in WAVE11_SPLIT_SCENARIOS[split_name] if sc not in asof_scenarios)
    lo_d, hi_d = WAVE11_SPLIT_DISTRACTOR_RANGE[split_name]

    def _scoped_add_distractors(b: Any, used_slots: Set[Tuple[str, str]]) -> None:
        rng = b.rng
        for _ in range(rng.randint(lo_d, hi_d)):
            free = [p for p in rlvr_env.SLOT_POOL if (p[0], p[1]) not in used_slots]
            if not free:
                return
            entity, slot, vals = rng.choice(free)
            used_slots.add((entity, slot))
            ctx = _SlotCtx(b, entity, slot, vals, rng.choice(("general", "ops")))
            rlvr_env._SCENARIOS[rng.choice(rlvr_env._DISTRACTOR_SCENARIOS)](
                ctx, rng.choice((LABEL_T, LABEL_F, LABEL_U))
            )
            if rng.random() < 0.25:
                x = ctx.b.add_event(entity, slot, rng.choice(vals), 1, ctx.domain)
                b.add_edge(x, x, "DERIVES_CURRENT")

    try:
        rlvr_env._SCENARIOS.update(WAVE11_EXTRA_SCENARIOS)
        rlvr_env.SLOT_POOL = WAVE11_SPLIT_SLOT_POOLS[split_name]
        rlvr_env.FAMILIES = tuple(split_fams)
        rlvr_env._DISTRACTOR_SCENARIOS = split_dist
        rlvr_env._add_distractors = _scoped_add_distractors
        yield schedule_names
    finally:
        rlvr_env._SCENARIOS.clear()
        rlvr_env._SCENARIOS.update(orig_scenarios)
        rlvr_env.SLOT_POOL = orig_pool
        rlvr_env.FAMILIES = orig_fams
        rlvr_env._DISTRACTOR_SCENARIOS = orig_dist
        rlvr_env._add_distractors = orig_add_distractors


def build_wave11_disjoint_wfs_split(
    split_name: str,
    n_episodes: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[EpistemicEpisode]:
    """Generate 3-way oracle-audited WFS episodes strictly confined to `split_name`'s disjoint subspace."""
    if split_name not in WAVE11_SPLIT_SLOT_POOLS:
        raise KeyError(f"Unknown split {split_name!r}; expected one of {sorted(WAVE11_SPLIT_SLOT_POOLS)}")
    count = WAVE11_SPLIT_SIZES[split_name] if n_episodes is None else int(n_episodes)
    use_seed = WAVE11_SPLIT_SEEDS[split_name] if seed is None else int(seed)
    with _scoped_wave11_split(split_name) as schedule_names:
        episodes: List[EpistemicEpisode] = []
        offset = 0
        max_attempts = max(600, count * 25)
        while len(episodes) < count:
            if offset >= max_attempts:
                raise RuntimeError(f"Exceeded {max_attempts} generation attempts for split {split_name}")
            fam_name = schedule_names[len(episodes) % len(schedule_names)]
            raw_ep = rlvr_env.generate_episode(offset, seed=use_seed, family=fam_name)
            offset += 1
            ep = _rewrite_wave11_noise_retractor(raw_ep, split_name)
            if len(ep.prompt) > 3100:
                continue
            aud = audit_episode(ep)
            mismatches = sum(v for k, v in aud.items() if k != "nodes_checked")
            if mismatches != 0:
                raise RuntimeError(f"WFS 3-way oracle mismatch in {split_name}/{ep.episode_id}: {aud}")
            episodes.append(ep)
    return episodes


def build_all_five_disjoint_splits() -> Dict[str, List[EpistemicEpisode]]:
    return {
        name: build_wave11_disjoint_wfs_split(name, WAVE11_SPLIT_SIZES[name])
        for name in ("SFT_WARMUP", "GRPO_RL", "DEV", "EVAL_WAVE6_LOOK2", "EVAL_VIRGIN_V2")
    }


def _episode_footprint(episodes: Sequence[EpistemicEpisode]) -> Dict[str, Set[str]]:
    entities: Set[str] = set()
    slots: Set[str] = set()
    values: Set[str] = set()
    scenarios: Set[str] = set()
    families: Set[str] = set()
    seeds: Set[str] = set()
    prompt_shas: Set[str] = set()
    retractor_entities: Set[str] = set()
    for ep in episodes:
        scenarios.add(ep.scenario)
        families.add(ep.family)
        seeds.add(ep.seed)
        prompt_shas.add(hashlib.sha256(ep.prompt.encode("utf-8")).hexdigest())
        for ev in ep.events:
            entities.add(ev.entity.casefold())
            slots.add(ev.slot.casefold())
            values.add(ev.claim_value.strip().casefold())
            if "retractor" in ev.entity.casefold():
                retractor_entities.add(ev.entity.casefold())
    return {
        "entities": entities,
        "slots": slots,
        "values": values,
        "scenarios": scenarios,
        "families": families,
        "seeds": seeds,
        "prompt_shas": prompt_shas,
        "retractor_entities": retractor_entities,
    }


def assert_five_way_disjoint_splits(
    splits: Mapping[str, Sequence[EpistemicEpisode]],
    d1_cases: Optional[Sequence[Mapping[str, Any]]] = None,
    *,
    verify_wave6_eval_sha: bool = True,
) -> Dict[str, Any]:
    """Fail-closed 5-way pairwise disjointness proof (`SFT_WARMUP`, `GRPO_RL`, `DEV`, `EVAL_WAVE6_LOOK2`,
    `EVAL_VIRGIN_V2`) plus `D1` overlap check, `SLOT_POOL_VIRGIN_V2` vs `ek5_wfs_rlvr_env.SLOT_POOL` check,
    and byte-identical Wave-6 `EVAL` SHA-256 verification.
    """
    required_splits = {"SFT_WARMUP", "GRPO_RL", "DEV", "EVAL_WAVE6_LOOK2", "EVAL_VIRGIN_V2"}
    missing = required_splits - set(splits.keys())
    if missing:
        raise AssertionError(f"Missing required splits for 5-way disjointness check: {sorted(missing)}")

    footprints = {k: _episode_footprint(v) for k, v in splits.items()}
    names = sorted(footprints.keys())
    dims = ("entities", "slots", "values", "scenarios", "families", "seeds", "prompt_shas")
    checked_pairs: List[str] = []
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            checked_pairs.append(f"{a}__vs__{b}")
            for dim in dims:
                inter = footprints[a][dim] & footprints[b][dim]
                if inter:
                    raise AssertionError(f"5-way split leakage between {a} and {b} on {dim}: {sorted(inter)[:5]}")

    # Verify `SLOT_POOL_VIRGIN_V2` is completely disjoint from ALL 8 `ek5_wfs_rlvr_env.SLOT_POOL` entries
    base_pool_ents = {e.casefold() for e, _s, _vs in rlvr_env.SLOT_POOL}
    base_pool_slots = {s.casefold() for _e, s, _vs in rlvr_env.SLOT_POOL}
    base_pool_vals = {v.strip().casefold() for _e, _s, vs in rlvr_env.SLOT_POOL for v in vs}
    v2_ents = {e.casefold() for e, _s, _vs in SLOT_POOL_VIRGIN_V2}
    v2_slots = {s.casefold() for _e, s, _vs in SLOT_POOL_VIRGIN_V2}
    v2_vals = {v.strip().casefold() for _e, _s, vs in SLOT_POOL_VIRGIN_V2 for v in vs}
    if (v2_ents & base_pool_ents) or (v2_slots & base_pool_slots) or (v2_vals & base_pool_vals):
        raise AssertionError("SLOT_POOL_VIRGIN_V2 overlaps ek5_wfs_rlvr_env.SLOT_POOL!")

    d1_entity_overlap = 0
    d1_value_overlap = 0
    if d1_cases is not None:
        d1_ents = {str(c["subject"]).strip().casefold() for c in d1_cases}
        d1_vals: Set[str] = set()
        for c in d1_cases:
            g = str(c["gold"]).strip().casefold()
            if g != "abstain":
                d1_vals.add(g)
            for sv in c.get("stale_or_false_values", ()):
                d1_vals.add(str(sv).strip().casefold())
        all_rlvr_ents = set().union(*(fp["entities"] for fp in footprints.values()))
        all_rlvr_vals = set().union(*(fp["values"] for fp in footprints.values()))
        d1_entity_overlap = len(all_rlvr_ents & d1_ents)
        d1_value_overlap = len(all_rlvr_vals & d1_vals)
        if d1_entity_overlap != 0 or d1_value_overlap != 0:
            raise AssertionError(
                f"RLVR vs D1 overlap detected: entity_overlap={d1_entity_overlap}, value_overlap={d1_value_overlap}"
            )

    summary: Dict[str, Any] = {
        "pairwise_checked_pairs": checked_pairs,
        "noise_retractor_included_in_hash": True,
        "virgin_v2_vs_base_slot_pool_overlap": 0,
        "d1_entity_overlap": d1_entity_overlap,
        "d1_value_overlap": d1_value_overlap,
        "splits": {},
    }
    for k, fp in footprints.items():
        lbls = Counter(ep.gold_label for ep in splits[k])
        bundle_sha = hashlib.sha256("\n".join(sorted(fp["prompt_shas"])).encode("utf-8")).hexdigest()
        summary["splits"][k] = {
            "n_episodes": len(splits[k]),
            "label_counts": dict(sorted(lbls.items())),
            "n_entities": len(fp["entities"]),
            "n_slots": len(fp["slots"]),
            "n_values": len(fp["values"]),
            "retractor_entities": sorted(fp["retractor_entities"]),
            "scenarios": sorted(fp["scenarios"]),
            "families": sorted(fp["families"]),
            "split_prompt_bundle_sha256": bundle_sha,
        }

    wave6_sha = summary["splits"]["EVAL_WAVE6_LOOK2"]["split_prompt_bundle_sha256"]
    summary["wave6_eval_sha256_matches_byte_identical"] = bool(wave6_sha == WAVE6_EVAL_EXPECTED_SHA256)
    summary["eval_virgin_v2_presealed_sha256"] = summary["splits"]["EVAL_VIRGIN_V2"]["split_prompt_bundle_sha256"]
    if verify_wave6_eval_sha and len(splits["EVAL_WAVE6_LOOK2"]) == 120:
        if wave6_sha != WAVE6_EVAL_EXPECTED_SHA256:
            raise AssertionError(
                f"EVAL_WAVE6_LOOK2 SHA mismatch: got {wave6_sha}, expected {WAVE6_EVAL_EXPECTED_SHA256}"
            )
    return summary


# ---------------------------------------------------------------------------
# Untruncated 4-Step WFS CoT Trace (Strictly `prompt_text: str` ONLY!)
# ---------------------------------------------------------------------------
def derive_untruncated_wfs_cot_from_prompt(prompt_text: str) -> str:
    """Derive untruncated 4-step WFS `<think>...</think>\\nVERDICT: ...\\nANSWER: ...` from `prompt_text: str` ONLY.

    Zero oracle metadata reads, zero truncation on `>6` target nodes or `>4` relations:
      (1) Raw claims & D_V effective authority ceiling for ALL target nodes
      (2) Qualified REFUTE / SUPERSEDE / CITES_AUDIT edges (plus blocked tier/domain attempts)
      (3) Alternating WFS fixpoint & cycle / contested-parent (`cur=U`/`val=U`) propagation
      (4) Canonical VERDICT + ANSWER
    """
    hdr_m = _HEADER_TARGET_RE.search(prompt_text)
    if not hdr_m:
        raise ValueError("Could not parse target entity.slot from prompt header")
    target_ent, target_slot = hdr_m.group(1), hdr_m.group(2)
    t_cut = int(hdr_m.group(3)) if hdr_m.group(3) is not None else None

    prop_m = _PROPOSED_LINE_RE.search(prompt_text)
    if not prop_m:
        raise ValueError("Could not parse Proposed answer line from prompt")
    proposed_val, query_id = prop_m.group(1), prop_m.group(2)

    raw_evs: Dict[str, Dict[str, Any]] = {}
    raw_rels: List[Tuple[int, str, str, str]] = []
    post_cutoff_ignored = 0

    for raw_ln in prompt_text.splitlines():
        ln = raw_ln.strip()
        em = _EV_LINE_RE.match(ln)
        if em:
            eid, tx_s, role, tier_s, dom, ent, slt, val = em.groups()
            tx = int(tx_s)
            if t_cut is not None and tx > t_cut:
                post_cutoff_ignored += 1
                continue
            raw_evs[eid] = {
                "id": eid,
                "tx": tx,
                "role": role,
                "tier": int(tier_s),
                "domain": dom.strip(),
                "entity": ent,
                "slot": slt,
                "val": val,
            }
            continue
        rm = _REL_LINE_RE.match(ln)
        if rm:
            rtx_s, src, verb, dst = rm.groups()
            rtx = int(rtx_s)
            if t_cut is not None and rtx > t_cut:
                post_cutoff_ignored += 1
                continue
            raw_rels.append((rtx, src, _VERB_TO_EDGE[verb], dst))

    nodes = sorted(raw_evs.keys())
    vis_rels = [(rtx, s, et, d) for (rtx, s, et, d) in raw_rels if s in raw_evs and d in raw_evs]

    deps_v: Dict[str, Set[str]] = defaultdict(set)
    deps_c: Dict[str, Set[str]] = defaultdict(set)
    raw_ref: Dict[str, Set[str]] = defaultdict(set)
    raw_sup: Dict[str, Set[str]] = defaultdict(set)
    cites_audit_pairs: List[Tuple[str, str]] = []

    for _rtx, src, et, dst in vis_rels:
        if et in DEP_ALL_TYPES:
            deps_v[src].add(dst)
            if et in DEP_CUR_TYPES:
                deps_c[src].add(dst)
        elif et == "REFUTE":
            raw_ref[dst].add(src)
        elif et == "SUPERSEDE":
            raw_sup[dst].add(src)
        elif et == "CITES_AUDIT":
            cites_audit_pairs.append((src, dst))

    cyclic: Set[str] = set()
    for comp in _kosaraju_sccs(nodes, deps_v):
        if len(comp) > 1 or comp[0] in deps_v.get(comp[0], ()):
            cyclic.update(comp)

    eff: Dict[str, int] = {}

    def _get_eff(u: str) -> int:
        if u in eff:
            return eff[u]
        if u in cyclic:
            eff[u] = 0
            return 0
        best = int(raw_evs[u]["tier"])
        for p in deps_v.get(u, ()):
            best = min(best, _get_eff(p))
        eff[u] = best
        return best

    for u in nodes:
        _get_eff(u)

    def _qualifies(src: str, dst: str) -> bool:
        sd, td = raw_evs[src]["domain"], raw_evs[dst]["domain"]
        if sd != "general" and td != "general" and sd != td:
            return False
        return eff[src] >= eff[dst]

    refs_q = {w: {a for a in raw_ref.get(w, ()) if _qualifies(a, w)} for w in nodes}
    sup_q = {w: {a for a in raw_sup.get(w, ()) if _qualifies(a, w)} for w in nodes}
    val = compute_alternating_wfs_oracle(nodes, refs_q, deps_v)

    cur: Dict[str, str] = {}
    for v in nodes:
        if val[v] == LABEL_F:
            cur[v] = LABEL_F
            continue
        b_ext = LABEL_F
        for j in sup_q[v]:
            if val[j] == LABEL_T:
                b_ext = LABEL_T
            elif val[j] == LABEL_U and b_ext != LABEL_T:
                b_ext = LABEL_U
        if b_ext == LABEL_T:
            cur[v] = LABEL_F
        elif val[v] == LABEL_U or b_ext == LABEL_U:
            cur[v] = LABEL_U
        else:
            cur[v] = LABEL_T

    for comp in reversed(_kosaraju_sccs(nodes, {u: set(deps_c.get(u, ())) for u in nodes})):
        if len(comp) > 1 or comp[0] in deps_c.get(comp[0], ()):
            for u in comp:
                cur[u] = LABEL_F
            continue
        u = comp[0]
        if cur[u] == LABEL_F:
            continue
        for p in deps_c.get(u, ()):
            if cur[p] == LABEL_F:
                cur[u] = LABEL_F
                break
            if cur[p] == LABEL_U and cur[u] == LABEL_T:
                cur[u] = LABEL_U

    target_ids = [
        u for u in nodes if raw_evs[u]["entity"] == target_ent and raw_evs[u]["slot"] == target_slot
    ]
    target_set = set(target_ids)
    standing_current = [u for u in target_ids if val[u] == LABEL_T and cur[u] == LABEL_T]
    defeated_or_old = [u for u in target_ids if val[u] == LABEL_F or cur[u] == LABEL_F]
    unresolved_claims = [
        u
        for u in target_ids
        if not (val[u] == LABEL_F or cur[u] == LABEL_F) and not (val[u] == LABEL_T and cur[u] == LABEL_T)
    ]
    surviving_vals = sorted({str(raw_evs[u]["val"]) for u in standing_current})

    # Step (1): ALL target claims & D_V effective authority ceilings (ZERO truncation!)
    step1_items: List[str] = []
    for u in target_ids:
        own_t = raw_evs[u]["tier"]
        u_val = raw_evs[u]["val"]
        if u in cyclic:
            step1_items.append(f"{u}=`{u_val}`(circ->eff0)")
        elif eff[u] < own_t:
            ps = ",".join(sorted(deps_v[u]))
            step1_items.append(f"{u}=`{u_val}`(t{own_t}->eff{eff[u]}<-{ps})")
        else:
            step1_items.append(f"{u}=`{u_val}`(eff{eff[u]})")
    cutoff_note = f" at t<={t_cut} (drop {post_cutoff_ignored} post-cutoff)" if t_cut is not None else ""
    step1 = f"(1) Target `{target_ent}.{target_slot}`{cutoff_note} ({len(target_ids)} claims): " + ", ".join(step1_items) + "."

    # Step (2): ALL qualified REFUTE / SUPERSEDE / CITES_AUDIT edges + blocked attempts (ZERO truncation!)
    edge_notes: List[str] = []
    for src, dst in cites_audit_pairs:
        if src in target_set or dst in target_set:
            edge_notes.append(f"{src} CITES_AUDIT {dst}(keeps eff{eff[src]})")
    for u in target_ids:
        q_refs = sorted(refs_q[u])
        q_sups = sorted(sup_q[u])
        blk_refs = sorted(raw_ref.get(u, set()) - refs_q[u])
        blk_sups = sorted(raw_sup.get(u, set()) - sup_q[u])
        if q_refs:
            edge_notes.append(f"{'+'.join(q_refs)} REF {u}")
        if q_sups:
            edge_notes.append(f"{'+'.join(q_sups)} SUP {u}")
        if blk_refs:
            edge_notes.append(f"blocked REF {'+'.join(blk_refs)}->{u}")
        if blk_sups:
            edge_notes.append(f"blocked SUP {'+'.join(blk_sups)}->{u}")
    step2 = "(2) Edges: " + ("; ".join(edge_notes) if edge_notes else "none") + "."

    # Step (3): Alternating WFS fixpoint & cycle / contested-parent propagation
    u_reasons: List[str] = []
    for u in unresolved_claims:
        if val[u] == LABEL_U and any(val[p] == LABEL_U for p in deps_v.get(u, ())):
            u_reasons.append(f"{u}(val=U via parent)")
        elif val[u] == LABEL_U:
            u_reasons.append(f"{u}(val=U cycle/peer)")
        else:
            u_reasons.append(f"{u}(cur=U contested sup/parent)")
    step3_parts = [f"stand={standing_current or '[]'}", f"defeated={defeated_or_old or '[]'}"]
    if u_reasons:
        step3_parts.append("unresolved=[" + ", ".join(u_reasons) + "]")
    else:
        step3_parts.append("unresolved=[]")
    step3 = "(3) WFS fixpoint: " + "; ".join(step3_parts) + "."

    # Step (4): Canonical Verdict + Answer
    if unresolved_claims or len(surviving_vals) != 1:
        verdict = LABEL_U
        ans = ABSTAIN
        if unresolved_claims:
            step4 = f"(4) Unresolved claims {unresolved_claims} prevent unique admission -> VERDICT U, ANSWER ABSTAIN."
        else:
            step4 = f"(4) Multiple distinct surviving values {surviving_vals} -> VERDICT U, ANSWER ABSTAIN."
    else:
        ans = surviving_vals[0]
        q_stands = query_id in standing_current
        verdict = LABEL_T if q_stands else LABEL_F
        step4 = (
            f"(4) Unique surviving `{ans}`; proposed `{proposed_val}` [{query_id}] "
            f"{'stands -> T' if q_stands else 'defeated/superseded -> F'}."
        )

    cot = f"<think>{step1} {step2} {step3} {step4}</think>\nVERDICT: {verdict}\nANSWER: {ans}"
    low_cot = cot.lower()
    for bad in _FORBIDDEN_ORACLE_COT_SUBSTRINGS:
        if bad in low_cot:
            raise AssertionError(f"Forbidden internal token {bad!r} leaked into observable CoT")
    return cot


# ---------------------------------------------------------------------------
# Calibrated Anti-Collapse 3-Valued Reward (`score_epistemic_completion`)
# ---------------------------------------------------------------------------
def _norm_val(s: str) -> str:
    return s.strip().strip("*`\"' ").rstrip(".;,").strip("*`\"' ").casefold()


def score_epistemic_completion(completion_text: Any, episode_payload: Any) -> Dict[str, Any]:
    """L4 Protocol v2 Anti-Collapse 3-Valued WFS Verifier:
      - Exact canonical match (`T/F` exact value + matching verdict, or `U` + `ABSTAIN` + `VERDICT: U`):
        `core_reward = +1.00`, plus `format_bonus = +0.05` ONLY when `core_reward == 1.00` AND clean `<think>` 2-line format.
      - False abstention (`ABSTAIN` on determinate `T/F` item, REGARDLESS of verdict `T`, `F`, or `U`):
        `core_reward = -1.00` (`== WRONG_DETERMINATE_REWARD`), `format_bonus = 0.0` (ensuring
        `E[R_always_abstain] = -0.95/3 = -0.3167 < 0` on 1/3 T, 1/3 F, 1/3 U AND exact `-1.00` tie
        with `std == 0.0` on any 0-correct determinate `T/F` rollout group!).
      - False commitment on `U` (committing a non-ABSTAIN value when slot is contested `U`):
        `core_reward = -1.15`, `format_bonus = 0.0`.
      - Right exact value on `T/F` (or `ABSTAIN` on `U`) with mismatched verdict:
        `core_reward = +0.75`, `format_bonus = 0.0`, `exact_match = 0.0`.
      - Wrong determinate value / malformed / unclosed `<think>` / negation / mixed commitment / non-canonical refusal:
        `core_reward = -1.00`, `format_bonus = 0.0`.
    """
    ep = rlvr_env._coerce_episode(episode_payload)
    raw = (
        "".join(str(m.get("content", "")) for m in completion_text if isinstance(m, Mapping))
        if isinstance(completion_text, list)
        else str(completion_text or "")
    )
    low = raw.lower()
    if "<think>" in low and "</think>" not in low:
        return {
            "core_reward": WRONG_DETERMINATE_REWARD,
            "format_bonus": 0.0,
            "total_reward": WRONG_DETERMINATE_REWARD,
            "exact_match": 0.0,
            "verdict_match": 0.0,
            "is_abstain": False,
            "is_false_abstain": False,
            "is_false_commit_on_u": False,
            "valid_format": False,
            "outcome": "unclosed_think",
        }

    post = raw[low.rfind("</think>") + len("</think>") :].strip() if "</think>" in low else raw.strip()
    non_empty_post_lines = [ln.strip() for ln in post.splitlines() if ln.strip()]
    v_lines = _VERDICT_LINE_RE.findall(post)
    a_lines = _ANSWER_LINE_RE.findall(post)
    if len(v_lines) != 1 or len(a_lines) != 1 or len(non_empty_post_lines) != 2:
        return {
            "core_reward": WRONG_DETERMINATE_REWARD,
            "format_bonus": 0.0,
            "total_reward": WRONG_DETERMINATE_REWARD,
            "exact_match": 0.0,
            "verdict_match": 0.0,
            "is_abstain": False,
            "is_false_abstain": False,
            "is_false_commit_on_u": False,
            "valid_format": False,
            "outcome": "malformed_lines",
        }

    verdict = v_lines[0].upper()
    ans_raw = a_lines[0].strip().strip("*`\"' ").rstrip(".;,").strip("*`\"' ")
    ans_lo = ans_raw.casefold()
    if not ans_lo or _NEGATION_RE.search(ans_raw):
        return {
            "core_reward": WRONG_DETERMINATE_REWARD,
            "format_bonus": 0.0,
            "total_reward": WRONG_DETERMINATE_REWARD,
            "exact_match": 0.0,
            "verdict_match": 0.0,
            "is_abstain": False,
            "is_false_abstain": False,
            "is_false_commit_on_u": False,
            "valid_format": False,
            "outcome": "negated_or_empty",
        }

    cand_hits = [cv for cv in ep.candidate_values if cv.casefold() in ans_lo]
    cand_hits = [
        v for v in cand_hits if not any(v != w and v.casefold() in w.casefold() for w in cand_hits)
    ]
    has_abstain_word = "abstain" in ans_lo
    if (
        (has_abstain_word and ans_lo not in _CLOSED_ABSTAIN_TOKENS)
        or (has_abstain_word and cand_hits)
        or len(cand_hits) > 1
    ):
        return {
            "core_reward": WRONG_DETERMINATE_REWARD,
            "format_bonus": 0.0,
            "total_reward": WRONG_DETERMINATE_REWARD,
            "exact_match": 0.0,
            "verdict_match": 0.0,
            "is_abstain": False,
            "is_false_abstain": False,
            "is_false_commit_on_u": False,
            "valid_format": False,
            "outcome": "mixed_commitment",
        }
    if ans_lo in _NON_CANONICAL_REFUSALS:
        return {
            "core_reward": WRONG_DETERMINATE_REWARD,
            "format_bonus": 0.0,
            "total_reward": WRONG_DETERMINATE_REWARD,
            "exact_match": 0.0,
            "verdict_match": 0.0,
            "is_abstain": False,
            "is_false_abstain": False,
            "is_false_commit_on_u": False,
            "valid_format": False,
            "outcome": "non_canonical_abstain",
        }

    is_abstain = ans_lo == "abstain"
    verdict_ok = verdict == ep.gold_label
    has_clean_think = bool(_THINK_CLOSED_RE.match(raw))

    if ep.gold_label == LABEL_U:
        if not is_abstain:
            return {
                "core_reward": FALSE_COMMIT_ON_U_REWARD,
                "format_bonus": 0.0,
                "total_reward": FALSE_COMMIT_ON_U_REWARD,
                "exact_match": 0.0,
                "verdict_match": float(verdict_ok),
                "is_abstain": False,
                "is_false_abstain": False,
                "is_false_commit_on_u": True,
                "valid_format": has_clean_think,
                "outcome": "false_commit_on_u",
            }
        core_r = EXACT_CORE_REWARD if verdict_ok else PARTIAL_VERDICT_MISMATCH_REWARD
        exact_m = 1.0 if verdict_ok else 0.0
        fmt_b = FORMAT_BONUS_REWARD if (exact_m == 1.0 and has_clean_think) else 0.0
        return {
            "core_reward": core_r,
            "format_bonus": fmt_b,
            "total_reward": round(core_r + fmt_b, 6),
            "exact_match": exact_m,
            "verdict_match": float(verdict_ok),
            "is_abstain": True,
            "is_false_abstain": False,
            "is_false_commit_on_u": False,
            "valid_format": has_clean_think,
            "outcome": "correct_abstention" if verdict_ok else "correct_abstention_wrong_verdict",
        }

    # Determinate episode (`ep.gold_label in {"T", "F"}`)
    if is_abstain:
        return {
            "core_reward": FALSE_ABSTAIN_REWARD,
            "format_bonus": 0.0,
            "total_reward": FALSE_ABSTAIN_REWARD,
            "exact_match": 0.0,
            "verdict_match": float(verdict_ok),
            "is_abstain": True,
            "is_false_abstain": True,
            "is_false_commit_on_u": False,
            "valid_format": has_clean_think,
            "outcome": "false_abstention_on_determinate",
        }

    gold_norm = _norm_val(ep.gold_value or "")
    if ans_lo == gold_norm:
        core_r = EXACT_CORE_REWARD if verdict_ok else PARTIAL_VERDICT_MISMATCH_REWARD
        exact_m = 1.0 if verdict_ok else 0.0
        fmt_b = FORMAT_BONUS_REWARD if (exact_m == 1.0 and has_clean_think) else 0.0
        return {
            "core_reward": core_r,
            "format_bonus": fmt_b,
            "total_reward": round(core_r + fmt_b, 6),
            "exact_match": exact_m,
            "verdict_match": float(verdict_ok),
            "is_abstain": False,
            "is_false_abstain": False,
            "is_false_commit_on_u": False,
            "valid_format": has_clean_think,
            "outcome": "correct_value" if verdict_ok else "correct_value_wrong_verdict",
        }

    is_trap = ep.gold_label == LABEL_F and ans_lo == _norm_val(ep.proposed_value)
    return {
        "core_reward": WRONG_DETERMINATE_REWARD,
        "format_bonus": 0.0,
        "total_reward": WRONG_DETERMINATE_REWARD,
        "exact_match": 0.0,
        "verdict_match": float(verdict_ok),
        "is_abstain": False,
        "is_false_abstain": False,
        "is_false_commit_on_u": False,
        "valid_format": has_clean_think,
        "outcome": "accepted_trap" if is_trap else "wrong_value",
    }


def compute_expected_always_abstain_reward(
    p_t: float = 1.0 / 3.0,
    p_f: float = 1.0 / 3.0,
    p_u: float = 1.0 / 3.0,
) -> float:
    """Analytical expected reward of `Always_Abstain` with clean `<think>` format."""
    return float(
        p_u * (EXACT_CORE_REWARD + FORMAT_BONUS_REWARD) + (p_t + p_f) * FALSE_ABSTAIN_REWARD
    )


def evaluate_always_abstain_wfs_baseline(eps: Sequence[EpistemicEpisode]) -> Dict[str, Any]:
    """Explicit `Always_Abstain_Baseline` evaluation row on any WFS split."""
    const_comp = "<think>Always abstaining baseline.</think>\nVERDICT: U\nANSWER: ABSTAIN"
    return _score_wfs_completions(eps, [const_comp] * len(eps), lat_s=0.0)


def evaluate_always_abstain_d1_baseline(cases: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Explicit `Always_Abstain_Baseline` evaluation row on D1 enterprise prose."""
    const_comp = "<think>Always abstaining baseline.</think>\nVERDICT: U\nANSWER: ABSTAIN"
    return _summarize_d1_completions(cases, [const_comp] * len(cases), wall_s=0.0)


class Wave11RolloutTelemetryTracker:
    """In-process step-level rollout telemetry recorder for `GRPOTrainer` (L4 anti-collapse reward)."""

    def __init__(self, num_generations: int = 8) -> None:
        self.num_generations = int(num_generations)
        self.records: List[Dict[str, float]] = []
        self.log_metric_errors: List[str] = []

    def make_reward_fn(self) -> Callable[..., List[float]]:
        tracker = self

        def wave11_ek5_wfs_reward_func(
            *,
            prompts: Sequence[Any],
            completions: Sequence[Any],
            episode: Sequence[Any],
            **kwargs: Any,
        ) -> List[float]:
            assert completions is not prompts, "completions must not alias prompts"
            if prompts is None or completions is None or episode is None:
                raise ValueError("prompts, completions, and episode must all be provided by keyword")
            if not (len(prompts) == len(completions) == len(episode) and len(completions) > 0):
                raise ValueError(
                    f"Cardinality mismatch: prompts={len(prompts)}, completions={len(completions)}, episode={len(episode)}"
                )
            for p_item, c_item in zip(prompts, completions):
                if p_item == c_item:
                    raise AssertionError("Reward received prompt text inside completions!")

            scored = [score_epistemic_completion(c, ep) for c, ep in zip(completions, episode)]
            totals = [float(s["total_reward"]) for s in scored]

            g = tracker.num_generations
            zero_std_groups = 0
            n_groups = 0
            if len(totals) % g == 0:
                for start in range(0, len(totals), g):
                    grp = totals[start : start + g]
                    n_groups += 1
                    if float(np.std(grp)) < 1e-8:
                        zero_std_groups += 1

            rec = {
                "mean_total_reward": float(np.mean(totals)),
                "mean_core_reward": float(np.mean([s["core_reward"] for s in scored])),
                "exact_match_rate": float(np.mean([s["exact_match"] for s in scored])),
                "abstain_rate": float(np.mean([1.0 if s["is_abstain"] else 0.0 for s in scored])),
                "false_abstain_rate": float(np.mean([1.0 if s["is_false_abstain"] else 0.0 for s in scored])),
                "false_commit_on_u_rate": float(
                    np.mean([1.0 if s["is_false_commit_on_u"] else 0.0 for s in scored])
                ),
                "valid_format_rate": float(np.mean([1.0 if s["valid_format"] else 0.0 for s in scored])),
                "group_zero_std_frac": float(zero_std_groups / max(1, n_groups)),
            }
            tracker.records.append(rec)
            log_metric = kwargs.get("log_metric")
            if callable(log_metric):
                for m_k in ("abstain_rate", "false_abstain_rate", "false_commit_on_u_rate", "exact_match_rate"):
                    try:
                        log_metric(m_k, rec[m_k])
                    except Exception as exc:
                        msg = f"log_metric({m_k!r}) failed: {type(exc).__name__}: {exc}"
                        tracker.log_metric_errors.append(msg)
                        print(f"[Wave11RolloutTelemetryTracker] WARNING: {msg}", file=sys.stderr, flush=True)
            return totals

        return wave11_ek5_wfs_reward_func


def _score_wfs_completions(
    eps: Sequence[EpistemicEpisode],
    comps: Sequence[str],
    lat_s: float,
) -> Dict[str, Any]:
    per_id_exact: Dict[str, float] = {}
    per_id_reward: Dict[str, float] = {}
    details: List[Dict[str, Any]] = []
    by_sc: Dict[str, List[float]] = defaultdict(list)
    det_false_abs: List[float] = []
    u_false_comm: List[float] = []
    u_exact: List[float] = []
    det_exact: List[float] = []

    for ep, comp in zip(eps, comps):
        sc = score_epistemic_completion(comp, ep)
        per_id_exact[ep.episode_id] = float(sc["exact_match"])
        per_id_reward[ep.episode_id] = float(sc["total_reward"])
        by_sc[ep.scenario].append(float(sc["exact_match"]))
        if ep.gold_label == LABEL_U:
            u_false_comm.append(1.0 if sc["is_false_commit_on_u"] else 0.0)
            u_exact.append(float(sc["exact_match"]))
        else:
            det_false_abs.append(1.0 if sc["is_false_abstain"] else 0.0)
            det_exact.append(float(sc["exact_match"]))
        details.append(
            {
                "episode_id": ep.episode_id,
                "scenario": ep.scenario,
                "gold_label": ep.gold_label,
                "gold_value": ep.gold_value or ABSTAIN,
                "exact_match": float(sc["exact_match"]),
                "total_reward": float(sc["total_reward"]),
                "core_reward": float(sc["core_reward"]),
                "outcome": sc["outcome"],
                "is_abstain": bool(sc["is_abstain"]),
                "valid_format": bool(sc["valid_format"]),
                "completion": comp,
            }
        )
    return {
        "exact_match": round(float(np.mean(list(per_id_exact.values()))), 4),
        "mean_total_reward": round(float(np.mean(list(per_id_reward.values()))), 4),
        "mean_core_reward": round(float(np.mean([d["core_reward"] for d in details])), 4),
        "determinate_exact_match": round(float(np.mean(det_exact)) if det_exact else 0.0, 4),
        "u_abstention_exact_match": round(float(np.mean(u_exact)) if u_exact else 0.0, 4),
        "abstain_rate": round(float(np.mean([1.0 if d["is_abstain"] else 0.0 for d in details])), 4),
        "false_abstain_rate_on_determinate": round(float(np.mean(det_false_abs)) if det_false_abs else 0.0, 4),
        "false_commit_rate_on_u": round(float(np.mean(u_false_comm)) if u_false_comm else 0.0, 4),
        "valid_format_rate": round(float(np.mean([1.0 if d["valid_format"] else 0.0 for d in details])), 4),
        "by_scenario_exact_match": {k: round(float(np.mean(v)), 4) for k, v in sorted(by_sc.items())},
        "eval_seconds": round(lat_s, 2),
        "per_id_exact": per_id_exact,
        "per_id_reward": per_id_reward,
        "details": details,
    }


def _strip_heavy_eval(res: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in res.items() if k not in ("per_id_exact", "per_id_reward", "details")}


def _strip_heavy_d1(res: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in res.items() if k not in ("per_id_strict", "per_id_frozen", "details")}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_directory_files(dir_path: Path) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for p in sorted(dir_path.rglob("*")):
        if p.is_file():
            out[str(p.relative_to(dir_path)).replace("\\", "/")] = sha256_file(p)
    return out


# ---------------------------------------------------------------------------
# Shared Helpers (vLLM Reader, RRF, Completion-Only SFT)
# ---------------------------------------------------------------------------
def _rrf_combine(bm25_scores: np.ndarray, dense_scores: np.ndarray, k: int = 60) -> np.ndarray:
    n = bm25_scores.shape[0]
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    out = np.zeros(n, dtype=np.float64)
    for arr in (bm25_scores, dense_scores):
        order = np.argsort(-arr, kind="mergesort")
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(1, n + 1, dtype=np.float64)
        out += 1.0 / (float(k) + ranks)
    return out


def call_vllm_timed(
    endpoint: str,
    model_id: str,
    system: str,
    user: str,
    *,
    max_tokens: int = 128,
) -> Tuple[str, int, float]:
    t0 = time.perf_counter()
    for attempt in range(5):
        try:
            resp = requests.post(
                f"{endpoint.rstrip('/')}/chat/completions",
                json={
                    "model": model_id,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": 0.0,
                    "max_tokens": int(max_tokens),
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout=180.0,
            )
            if resp.status_code in {408, 409, 429, 500, 502, 503, 504}:
                raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:160]}")
            resp.raise_for_status()
            body = resp.json()
            txt = (body["choices"][0]["message"].get("content") or "").strip()
            ptoks = int(body.get("usage", {}).get("prompt_tokens") or 0)
            return txt, ptoks, (time.perf_counter() - t0) * 1000.0
        except Exception:
            if attempt == 4:
                raise
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError("Unreachable")


def load_wave6_pm3_suite_a_baselines(
    manifest_path: Path = DEFAULT_WAVE6_PM3_MANIFEST,
    rows_csv_path: Path = DEFAULT_WAVE6_PM3_ROWS_CSV,
) -> Dict[str, Any]:
    """Load exact per-item Suite A accuracy maps & latency baselines for live Mem0 (81.67%) and Wave-6 30B (96.67%)."""
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    suite_a_meta = manifest["suite_a_enterprise_n60"]["arms"]
    mem0_manifest_acc = float(suite_a_meta["Mem0_v2_2_1_Live_InferTrue"]["accuracy"])
    w6_30b_manifest_acc = float(suite_a_meta["Chelated_v4_Stratified_WFS"]["accuracy"])
    w6_30b_extract_ms = float(suite_a_meta["Chelated_v4_Stratified_WFS"]["clocks_ms"]["consolidation_bg_mean_ms"])

    mem0_per_id: Dict[str, float] = {}
    w6_30b_per_id: Dict[str, float] = {}
    with rows_csv_path.open("r", encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("suite") != "SuiteA_Enterprise_N60":
                continue
            item_id = str(row["item_id"])
            val = 1.0 if str(row["correct_post"]).strip().lower() == "true" else 0.0
            if row["arm"] == "Mem0_v2_2_1_Live_InferTrue":
                mem0_per_id[item_id] = val
            elif row["arm"] == "Chelated_v4_Stratified_WFS":
                w6_30b_per_id[item_id] = val

    if len(mem0_per_id) != 60 or len(w6_30b_per_id) != 60:
        raise AssertionError(
            f"Expected 60 Suite A baseline rows per arm, got mem0={len(mem0_per_id)}, w6_30b={len(w6_30b_per_id)}"
        )
    if abs(float(np.mean(list(mem0_per_id.values()))) - mem0_manifest_acc) > 1e-6:
        raise AssertionError("Mem0 CSV accuracy disagrees with wave6_pm3_v2_arena_manifest.json")
    if abs(float(np.mean(list(w6_30b_per_id.values()))) - w6_30b_manifest_acc) > 1e-6:
        raise AssertionError("Wave-6 30B CSV accuracy disagrees with wave6_pm3_v2_arena_manifest.json")

    return {
        "manifest_path": str(manifest_path),
        "rows_csv_path": str(rows_csv_path),
        "mem0_live_infer_true_accuracy": mem0_manifest_acc,
        "wave6_30b_extractor_accuracy": w6_30b_manifest_acc,
        "wave6_30b_extractor_mean_ms_per_case": w6_30b_extract_ms,
        "mem0_live_infer_true_per_id": mem0_per_id,
        "wave6_30b_extractor_per_id": w6_30b_per_id,
    }


def select_best_extractor_on_dev(candidates: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Pre-registered DEV-only checkpoint selection (zero peeking at Suite A).

    Promotes the smallest candidate meeting `json_valid_rate >= 0.99`, `stance_accuracy >= 0.92`,
    and `value_exact_accuracy >= 0.90`; otherwise picks highest `full_tuple_accuracy`, breaking ties
    by `value_exact_accuracy` then lower `ms_per_17doc_batch_mean`.
    """
    if not candidates:
        raise ValueError("No extractor candidates provided for DEV selection")

    qualifying = [
        c
        for c in candidates
        if float(c["dev_metrics"]["json_valid_rate"]) >= 0.99
        and float(c["dev_metrics"]["stance_accuracy"]) >= 0.92
        and float(c["dev_metrics"]["value_exact_accuracy"]) >= 0.90
    ]
    if qualifying:
        winner = min(
            qualifying,
            key=lambda c: (
                int(c["train_summary"]["total_params"]),
                float(c["dev_metrics"]["ms_per_17doc_batch_mean"]),
            ),
        )
        reason = "smallest_model_passing_preregistered_dev_gate"
    else:
        winner = max(
            candidates,
            key=lambda c: (
                float(c["dev_metrics"]["full_tuple_accuracy"]),
                float(c["dev_metrics"]["value_exact_accuracy"]),
                -float(c["dev_metrics"]["ms_per_17doc_batch_mean"]),
            ),
        )
        reason = "highest_dev_full_tuple_accuracy_fallback"

    return {
        "selected_model_id": winner["base_model_id"],
        "selected_tag": winner["tag"],
        "merged_model_dir": winner["train_summary"]["merged_model_dir"],
        "selection_reason": reason,
        "selected_dev_metrics": winner["dev_metrics"],
    }


# ---------------------------------------------------------------------------
# Mode 1: `--mode verify-splits` (Fast CPU Verification)
# ---------------------------------------------------------------------------
def run_verify_splits(output_dir: Path, osv_cache_path: Path) -> Dict[str, Any]:
    t0 = time.perf_counter()

    # Part A verification
    ext_splits = build_disjoint_extractor_splits(n_train=600, n_dev=120, seed=20261123)
    train_dev_proof = assert_train_dev_disjoint(ext_splits["train"], ext_splits["dev"])
    suite_a_disjoint_proof = assert_extractor_corpus_disjoint_from_suite_a(
        ext_splits["train"] + ext_splits["dev"],
        osv_cache_path=osv_cache_path,
        max_5gram_jaccard_cap=0.25,
    )

    # Part B verification
    wfs_splits = build_all_five_disjoint_splits()
    d1_cases, d1_docs, d1_sha = build_prospective_realworld_drift_suite(osv_cache_path)
    five_way_proof = assert_five_way_disjoint_splits(wfs_splits, d1_cases=d1_cases, verify_wave6_eval_sha=True)

    # Verify untruncated 4-step CoT on ALL 5 splits (0 oracle reads, 100% agreement!)
    cot_stats: Dict[str, Dict[str, Any]] = {}
    for s_name, eps in wfs_splits.items():
        exact_hits = 0
        max_words = 0
        for ep in eps:
            cot = derive_untruncated_wfs_cot_from_prompt(ep.prompt)
            sc = score_epistemic_completion(cot, ep)
            if sc["exact_match"] == 1.0 and sc["total_reward"] == pytest_approx_val(1.05):
                exact_hits += 1
            else:
                raise AssertionError(f"Untruncated CoT failed on {s_name}/{ep.episode_id}: {sc}")
            max_words = max(max_words, len(cot.split()))
        cot_stats[s_name] = {
            "n_episodes": len(eps),
            "exact_matches": exact_hits,
            "agreement_rate": round(exact_hits / max(1, len(eps)), 4),
            "max_words": max_words,
        }

    always_abstain_virgin = _strip_heavy_eval(evaluate_always_abstain_wfs_baseline(wfs_splits["EVAL_VIRGIN_V2"]))
    always_abstain_wave6 = _strip_heavy_eval(evaluate_always_abstain_wfs_baseline(wfs_splits["EVAL_WAVE6_LOOK2"]))
    e_abstain = compute_expected_always_abstain_reward()
    if e_abstain >= 0.0:
        raise AssertionError(f"Anti-collapse reward violated: E[R_always_abstain]={e_abstain} >= 0")

    report = {
        "status": "VERIFY_SPLITS_OK",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "wall_seconds": round(time.perf_counter() - t0, 2),
        "part_a_extractor_splits": {
            "train_n": len(ext_splits["train"]),
            "dev_n": len(ext_splits["dev"]),
            "train_stance_counts": ext_splits["train_stance_counts"],
            "dev_stance_counts": ext_splits["dev_stance_counts"],
            "train_domain_counts": ext_splits["train_domain_counts"],
            "dev_domain_counts": ext_splits["dev_domain_counts"],
            "train_sha256": ext_splits["train_sha256"],
            "dev_sha256": ext_splits["dev_sha256"],
            "train_dev_disjoint_proof": train_dev_proof,
            "suite_a_disjoint_proof": suite_a_disjoint_proof,
        },
        "part_b_wfs_splits": {
            "five_way_disjoint_proof": five_way_proof,
            "d1_cases": len(d1_cases),
            "d1_docs": len(d1_docs),
            "d1_corpus_sha256": d1_sha,
            "untruncated_cot_verification": cot_stats,
            "expected_always_abstain_reward_balanced": round(e_abstain, 6),
            "always_abstain_eval_virgin_v2": always_abstain_virgin,
            "always_abstain_eval_wave6_look2": always_abstain_wave6,
        },
    }
    work_dir = output_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(work_dir / "verify_splits_report.json", report)
    print(
        f"[Wave11-D9] VERIFY_SPLITS_OK in {report['wall_seconds']}s | "
        f"Suite-A max 5-gram Jaccard={suite_a_disjoint_proof['max_5gram_jaccard']:.4f} | "
        f"EVAL_WAVE6 SHA match={five_way_proof['wave6_eval_sha256_matches_byte_identical']} | "
        f"EVAL_VIRGIN_V2 SHA={five_way_proof['eval_virgin_v2_presealed_sha256'][:12]}... | "
        f"E[R_abstain]={e_abstain:.4f}",
        flush=True,
    )
    return report


def pytest_approx_val(target: float, tol: float = 1e-6) -> Any:
    class _Approx:
        def __eq__(self, other: object) -> bool:
            return isinstance(other, (int, float)) and abs(float(other) - target) <= tol

    return _Approx()


# ---------------------------------------------------------------------------
# Mode 2: `--mode run-part-a` (Desktop RTX 3090 Fast Extractor Pipeline)
# ---------------------------------------------------------------------------
def run_part_a_fast_extractor(
    *,
    output_dir: Path,
    osv_cache_path: Path,
    extractor_models: Sequence[str] = ("Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"),
    vllm_endpoint: str = "http://10.0.0.194:8890/v1",
    vllm_model: str = "nemotron-lightning",
    judge_hf_model: str = "Qwen/Qwen2.5-7B-Instruct",
    wave6_pm3_manifest: Path = DEFAULT_WAVE6_PM3_MANIFEST,
    wave6_pm3_rows_csv: Path = DEFAULT_WAVE6_PM3_ROWS_CSV,
    epochs: int = 3,
    max_new_tokens: int = 64,
    seed: int = 42,
    device: str = "cuda",
) -> Dict[str, Any]:
    """Train 0.5B & 1.5B LoRA extractors, select on synthetic DEV only, and evaluate Suite A end-to-end."""
    import torch
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t_start = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "_work" / "part_a"
    work_dir.mkdir(parents=True, exist_ok=True)

    # 1. Build & prove disjointness of synthetic train (N=600) + dev (N=120) splits
    ext_splits = build_disjoint_extractor_splits(n_train=600, n_dev=120, seed=20261123)
    train_dev_proof = assert_train_dev_disjoint(ext_splits["train"], ext_splits["dev"])
    suite_a_disjoint_proof = assert_extractor_corpus_disjoint_from_suite_a(
        ext_splits["train"] + ext_splits["dev"],
        osv_cache_path=osv_cache_path,
        max_5gram_jaccard_cap=0.25,
    )
    print(
        f"[Part-A] Synthetic corpus verified disjoint from Suite A (max 5-gram Jaccard="
        f"{suite_a_disjoint_proof['max_5gram_jaccard']:.4f} < 0.25).",
        flush=True,
    )

    # 2. Train each candidate extractor & benchmark strictly on synthetic DEV (N=120) BEFORE touching Suite A!
    candidates: List[Dict[str, Any]] = []
    for model_id in extractor_models:
        tag = re.sub(r"[^a-z0-9]+", "_", model_id.lower()).strip("_")
        m_dir = work_dir / f"ckpt_{tag}"
        print(f"[Part-A] Training completion-only LoRA SFT (r=16, alpha=32) for {model_id}...", flush=True)
        train_summary = train_fast_claim_extractor_lora(
            base_model_id=model_id,
            train_examples=ext_splits["train"],
            output_dir=m_dir,
            probe_examples=ext_splits["dev"][:4],
            epochs=epochs,
            seed=seed,
            device=device,
        )
        print(
            f"[Part-A] Trained & merged {model_id} in {train_summary['train_seconds']}s "
            f"(loss={train_summary['train_loss']}, reload_diff={train_summary['reload_max_abs_logit_diff']:.2e}). "
            "Evaluating on synthetic DEV (N=120)...",
            flush=True,
        )
        with FastEdgeClaimExtractor(
            train_summary["merged_model_dir"],
            device=device,
            default_max_new_tokens=max_new_tokens,
        ) as extractor:
            extractor.warmup(ext_splits["dev"][0].doc)
            dev_metrics = extractor.evaluate_on_synthetic_dev(
                ext_splits["dev"],
                batch_size=17,
                max_new_tokens=max_new_tokens,
            )
        print(
            f"[Part-A] DEV ({tag}): json_valid={dev_metrics['json_valid_rate']:.4f}, "
            f"stance={dev_metrics['stance_accuracy']:.4f}, val={dev_metrics['value_exact_accuracy']:.4f}, "
            f"full={dev_metrics['full_tuple_accuracy']:.4f}, ms/17docs={dev_metrics['ms_per_17doc_batch_mean']:.1f}",
            flush=True,
        )
        candidates.append(
            {
                "tag": tag,
                "base_model_id": model_id,
                "train_summary": train_summary,
                "dev_metrics": dev_metrics,
            }
        )

    selection = select_best_extractor_on_dev(candidates)
    print(
        f"[Part-A] Promoted on DEV ONLY: {selection['selected_model_id']} ({selection['selection_reason']})",
        flush=True,
    )

    # 3. Evaluate ONLY the DEV-promoted extractor on Suite A (N=60, 17 docs/case in ONE batch_size=17 call!)
    cases_a, all_docs_a, suite_a_sha = build_prospective_realworld_drift_suite(osv_cache_path)
    docs_by_id = {d.doc_id: d for d in all_docs_a}
    case_doc_lists: List[List[RawDocumentEnvelope]] = [
        sorted(
            [docs_by_id[did] for did in c["doc_ids"]],
            key=lambda d: (d.ingested_ts, d.published_ts, d.doc_id),
        )
        for c in cases_a
    ]

    with FastEdgeClaimExtractor(
        selection["merged_model_dir"],
        device=device,
        default_max_new_tokens=max_new_tokens,
    ) as promoted_extractor:
        promoted_extractor.warmup(ext_splits["dev"][0].doc)
        suite_a_claims_by_case, suite_a_extract_bench = promoted_extractor.extract_cases_batched(
            case_doc_lists,
            batch_size=17,
            max_new_tokens=max_new_tokens,
        )

    print(
        f"[Part-A] Suite A extraction complete: p50={suite_a_extract_bench['ms_per_case_p50']:.1f}ms, "
        f"p95={suite_a_extract_bench['ms_per_case_p95']:.1f}ms, mean={suite_a_extract_bench['ms_per_case_mean']:.1f}ms, "
        f"json_valid={suite_a_extract_bench['json_valid_rate']:.4f}",
        flush=True,
    )

    # 4. Plug directly into AuthorityDominatedExocortexV2 + Nemotron-30B reader
    bge = SentenceTransformer("BAAI/bge-m3", device=device)
    bge.max_seq_length = 384

    def _bge_encode(texts: Sequence[str], _enc: Any = bge) -> np.ndarray:
        return np.asarray(
            _enc.encode(list(texts), batch_size=64, normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False),
            dtype=np.float32,
        )

    case_rows: List[Dict[str, Any]] = []
    for idx_c, (case, c_docs, raw_claims) in enumerate(zip(cases_a, case_doc_lists, suite_a_claims_by_case)):
        cid = str(case["case_id"])
        q_txt = str(case["query_text"])
        gold = str(case["gold"])
        stale_vals = list(case.get("stale_or_false_values") or [])

        t_ack0 = time.perf_counter()
        d_vecs = _bge_encode([d.text for d in c_docs])
        s_vecs = _bge_encode([f"{cl.entity} {cl.slot}" for cl in raw_claims])
        exo = AuthorityDominatedExocortexV2(c_docs, raw_claims, d_vecs, s_vecs)
        bm_exo = FastBM25Index([d.doc_id for d in c_docs], [d.text for d in c_docs])
        write_ack_ms = (time.perf_counter() - t_ack0) * 1000.0

        t_read0 = time.perf_counter()
        qv = _bge_encode([q_txt])[0]
        hyb = _rrf_combine(bm_exo.score_query(q_txt), l2_normalize(d_vecs) @ qv)
        sel = exo.select_context_for_query(hyb, q_txt, top_k=5)
        read_ms = (time.perf_counter() - t_read0) * 1000.0
        ctx_txt = str(sel["formatted_context"])

        if sel["status"] == "CONTESTED_ABSTAIN":
            pred, r_toks, r_ms = "ABSTAIN", 0, 0.0
        else:
            pred, r_toks, r_ms = call_vllm_timed(
                vllm_endpoint,
                vllm_model,
                ENTERPRISE_READER_SYSTEM,
                f"Records:\n{ctx_txt}\n\nQuestion: {q_txt}\nAnswer:",
                max_tokens=128,
            )

        det_sc = score_d1_prose_prediction(pred, gold, stale_vals)
        case_rows.append(
            {
                "item_id": cid,
                "domain": str(case["domain"]),
                "regime": str(case["regime"]),
                "question": q_txt,
                "gold": gold,
                "abstention_required": (gold.strip().upper() == "ABSTAIN"),
                "stale_values": stale_vals,
                "exocortex_status": str(sel["status"]),
                "pred": pred,
                "extract_ms": float(suite_a_extract_bench["per_case_ms"][idx_c]),
                "write_ack_ms": round(write_ack_ms, 3),
                "read_retrieval_ms": round(read_ms, 3),
                "reader_llm_ms": round(r_ms, 3),
                "read_prompt_tokens": int(r_toks),
                "strict_exact_match": int(det_sc["strict_exact_match"]),
                "frozen_d1_match": int(det_sc["frozen_d1_match"]),
                "leaked_stale_value": bool(det_sc["leaked_stale_value"]),
            }
        )

    del bge
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 5. Blind Cross-Family Semantic Judge (`Qwen/Qwen2.5-7B-Instruct` judging `nvidia-nemotron` reader!)
    print(f"[Part-A] Loading BlindCrossFamilySemanticJudge ({judge_hf_model})...", flush=True)
    j_tok = AutoTokenizer.from_pretrained(judge_hf_model, padding_side="left")
    if j_tok.pad_token_id is None:
        j_tok.pad_token_id = j_tok.eos_token_id
    j_dtype = torch.bfloat16 if (device.startswith("cuda") and torch.cuda.is_bf16_supported()) else torch.float16
    import transformers as _tf_j
    _j_kw = {"dtype": j_dtype} if int(str(_tf_j.__version__).split(".", 1)[0]) >= 5 else {"torch_dtype": j_dtype}
    j_mdl = AutoModelForCausalLM.from_pretrained(judge_hf_model, **_j_kw).to(device)
    j_mdl.eval()

    def _qwen_judge(sys_p: str, usr_p: str, _mdl: Any = j_mdl) -> str:
        p_str = j_tok.apply_chat_template(
            [{"role": "system", "content": sys_p}, {"role": "user", "content": usr_p}],
            tokenize=False,
            add_generation_prompt=True,
        )
        enc = j_tok([p_str], return_tensors="pt").to(device)
        with torch.inference_mode():
            out = _mdl.generate(**enc, max_new_tokens=20, do_sample=False, pad_token_id=j_tok.pad_token_id)
        return j_tok.decode(out[0, enc.input_ids.shape[1] :], skip_special_tokens=True).strip()

    blind_judge = BlindCrossFamilySemanticJudge(
        judge_family="alibaba-qwen2.5",
        generator_family="nvidia-nemotron",
        completion_fn=_qwen_judge,
        salt="wave11-d9-part-a-judge-v1",
    )

    arm_key = f"FastEdge_{selection['selected_tag']}_Chelated_v2"
    fast_semantic_per_id: Dict[str, float] = {}
    fast_strict_per_id: Dict[str, float] = {}
    for row in case_rows:
        graded = blind_judge.grade_arms_blind(
            item_id=f"w11_partA:{row['item_id']}",
            question=row["question"],
            gold_reference=row["gold"],
            arm_predictions={arm_key: row["pred"]},
            abstention_required=row["abstention_required"],
        )[arm_key]
        row["blind_judge"] = graded
        row["semantic_correct"] = int(bool(graded["correct"]))
        fast_semantic_per_id[row["item_id"]] = float(row["semantic_correct"])
        fast_strict_per_id[row["item_id"]] = float(row["strict_exact_match"])

    del j_mdl
    gc.collect()
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 6. Paired Statistical Comparisons vs Live Mem0 (81.67%) & Wave-6 30B Extractor (96.67%)
    baselines = load_wave6_pm3_suite_a_baselines(wave6_pm3_manifest, wave6_pm3_rows_csv)
    item_ids = [r["item_id"] for r in case_rows]
    paired_vs_mem0 = paired_bootstrap_wave6(
        fast_semantic_per_id,
        baselines["mem0_live_infer_true_per_id"],
        seed=42,
        expected_ids=item_ids,
    )
    paired_vs_w6_30b = paired_bootstrap_wave6(
        fast_semantic_per_id,
        baselines["wave6_30b_extractor_per_id"],
        seed=43,
        expected_ids=item_ids,
    )
    part_a_holm = holm_bonferroni_stepdown(
        {
            "FastEdge_vs_Mem0_Live_InferTrue": paired_vs_mem0["p_value_two_sided"],
            "FastEdge_vs_Wave6_30B_Extractor": paired_vs_w6_30b["p_value_two_sided"],
        }
    )

    det_rows = [r for r in case_rows if not r["abstention_required"]]
    abs_rows = [r for r in case_rows if r["abstention_required"]]
    speedup_vs_30b = round(
        baselines["wave6_30b_extractor_mean_ms_per_case"] / max(1e-6, suite_a_extract_bench["ms_per_case_mean"]),
        2,
    )

    # Export per-case CSV
    csv_path = output_dir / "wave11_d9_part_a_suite_a_n60_rows.csv"
    csv_rows = [
        {
            "item_id": r["item_id"],
            "domain": r["domain"],
            "regime": r["regime"],
            "gold": r["gold"],
            "abstention_required": r["abstention_required"],
            "exocortex_status": r["exocortex_status"],
            "semantic_correct": r["semantic_correct"],
            "strict_exact_match": r["strict_exact_match"],
            "frozen_d1_match": r["frozen_d1_match"],
            "leaked_stale_value": int(r["leaked_stale_value"]),
            "extract_ms": r["extract_ms"],
            "write_ack_ms": r["write_ack_ms"],
            "read_retrieval_ms": r["read_retrieval_ms"],
            "reader_llm_ms": r["reader_llm_ms"],
            "read_prompt_tokens": r["read_prompt_tokens"],
            "pred": r["pred"][:240],
        }
        for r in case_rows
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)

    part_a_payload = {
        "schema_version": "wave11_d9_part_a_fast_extractor_v1",
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": current_git_sha(REPO_ROOT),
        "synthetic_corpus_audit": {
            "train_n": len(ext_splits["train"]),
            "dev_n": len(ext_splits["dev"]),
            "train_stance_counts": ext_splits["train_stance_counts"],
            "dev_stance_counts": ext_splits["dev_stance_counts"],
            "train_domain_counts": ext_splits["train_domain_counts"],
            "dev_domain_counts": ext_splits["dev_domain_counts"],
            "train_sha256": ext_splits["train_sha256"],
            "dev_sha256": ext_splits["dev_sha256"],
            "train_dev_disjoint_proof": train_dev_proof,
            "suite_a_disjoint_proof": suite_a_disjoint_proof,
        },
        "trained_candidates_dev_only": candidates,
        "dev_only_selection": selection,
        "suite_a_evaluation_n60": {
            "suite_a_corpus_sha256": suite_a_sha,
            "promoted_arm": arm_key,
            "reader_model": vllm_model,
            "blind_judge_model": judge_hf_model,
            "blind_judge_family": "alibaba-qwen2.5",
            "generator_family": "nvidia-nemotron",
            "n_cases": len(case_rows),
            "semantic_accuracy": round(float(np.mean([r["semantic_correct"] for r in case_rows])), 4),
            "semantic_determinate_accuracy": round(float(np.mean([r["semantic_correct"] for r in det_rows])), 4),
            "semantic_abstention_recall": round(float(np.mean([r["semantic_correct"] for r in abs_rows])), 4),
            "deterministic_strict_exact_match": round(float(np.mean([r["strict_exact_match"] for r in case_rows])), 4),
            "deterministic_frozen_d1_accuracy": round(float(np.mean([r["frozen_d1_match"] for r in case_rows])), 4),
            "stale_leak_rate": round(float(np.mean([1.0 if r["leaked_stale_value"] else 0.0 for r in case_rows])), 4),
            "extraction_benchmark": {k: v for k, v in suite_a_extract_bench.items() if k != "per_case_ms"},
            "speedup_vs_wave6_30b_extractor": speedup_vs_30b,
            "wave6_baselines": {
                "mem0_live_infer_true_accuracy": baselines["mem0_live_infer_true_accuracy"],
                "wave6_30b_extractor_accuracy": baselines["wave6_30b_extractor_accuracy"],
                "wave6_30b_extractor_mean_ms_per_case": baselines["wave6_30b_extractor_mean_ms_per_case"],
            },
            "paired_bootstrap_vs_mem0_live": paired_vs_mem0,
            "paired_bootstrap_vs_wave6_30b": paired_vs_w6_30b,
            "holm_bonferroni": part_a_holm,
            "l4_gates": {
                "ms_per_case_below_1500": bool(suite_a_extract_bench["ms_per_case_mean"] < 1500.0),
                "json_valid_rate_ge_099": bool(suite_a_extract_bench["json_valid_rate"] >= 0.99),
                "beats_mem0_lcb975_gt_0": bool(paired_vs_mem0["lcb_975"] > 0.0),
                "reload_parity_below_1e4": all(
                    bool(c["train_summary"]["reload_parity_below_1e4"]) for c in candidates
                ),
            },
        },
        "csv_artifact": {
            "filename": csv_path.name,
            "sha256": sha256_file(csv_path),
        },
        "wall_seconds": round(time.perf_counter() - t_start, 2),
    }

    part_a_path = output_dir / "part_a_fast_extractor_results.json"
    write_json_atomic(part_a_path, part_a_payload)
    print(
        f"[Part-A] DONE in {part_a_payload['wall_seconds']}s | "
        f"Suite-A sem_acc={part_a_payload['suite_a_evaluation_n60']['semantic_accuracy']:.4f}, "
        f"strict_em={part_a_payload['suite_a_evaluation_n60']['deterministic_strict_exact_match']:.4f}, "
        f"mean_ms/case={suite_a_extract_bench['ms_per_case_mean']:.1f} ({speedup_vs_30b}x vs 30B)",
        flush=True,
    )
    return part_a_payload


# ---------------------------------------------------------------------------
# Mode 3: `--mode run-part-b` (Spark #2 Virgin-Holdout Curriculum U-GRPO)
# ---------------------------------------------------------------------------
def build_wave11_completion_only_sft_features(
    tok: Any,
    episodes: Sequence[EpistemicEpisode],
    *,
    max_prompt_len: int = MAX_PROMPT_LENGTH,
    max_completion_len: int = MAX_COMPLETION_LENGTH,
) -> Tuple[List[Dict[str, List[int]]], Dict[str, Any]]:
    rows: List[Dict[str, List[int]]] = []
    prompt_lens: List[int] = []
    comp_lens: List[int] = []
    unmasked_prompt_tokens = 0
    cot_oracle_matches = 0

    for ep in episodes:
        cot = derive_untruncated_wfs_cot_from_prompt(ep.prompt)
        sc = score_epistemic_completion(cot, ep)
        if sc["exact_match"] != 1.0 or sc["format_bonus"] != FORMAT_BONUS_REWARD:
            raise RuntimeError(f"Untruncated CoT failed oracle verification on {ep.episode_id}: {sc}")
        cot_oracle_matches += 1

        prompt_msgs = [
            {"role": "system", "content": GRPO_SYSTEM_PROMPT_L4},
            {"role": "user", "content": ep.prompt},
        ]
        full_msgs = prompt_msgs + [{"role": "assistant", "content": cot}]
        prompt_str = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        full_str = tok.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)

        p_ids: List[int] = list(tok(prompt_str, add_special_tokens=False)["input_ids"])
        f_ids: List[int] = list(tok(full_str, add_special_tokens=False)["input_ids"])
        if len(p_ids) > max_prompt_len:
            raise ValueError(f"SFT prompt length {len(p_ids)} > {max_prompt_len} on {ep.episode_id}")
        if f_ids[: len(p_ids)] != p_ids:
            raise RuntimeError(f"Chat template prefix mismatch on {ep.episode_id}")
        c_ids = f_ids[len(p_ids) :]
        if len(c_ids) == 0 or len(c_ids) > max_completion_len:
            raise ValueError(f"SFT completion length {len(c_ids)} out of (0, {max_completion_len}] on {ep.episode_id}")

        labels = [-100] * len(p_ids) + c_ids
        unmasked_prompt_tokens += sum(1 for x in labels[: len(p_ids)] if x != -100)
        prompt_lens.append(len(p_ids))
        comp_lens.append(len(c_ids))
        rows.append({"input_ids": f_ids, "attention_mask": [1] * len(f_ids), "labels": labels})

    return rows, {
        "n_sft_examples": len(rows),
        "cot_oracle_exact_matches": cot_oracle_matches,
        "cot_uses_prompt_text_only": True,
        "cot_untruncated_4step": True,
        "store_slot_status_leaked": False,
        "unmasked_prompt_token_count": int(unmasked_prompt_tokens),
        "prompt_tokens_min": int(min(prompt_lens)),
        "prompt_tokens_max": int(max(prompt_lens)),
        "completion_tokens_min": int(min(comp_lens)),
        "completion_tokens_max": int(max(comp_lens)),
    }


def _call_nemotron_one(
    endpoint: str,
    model_id: str,
    sys_prompt: str,
    user_prompt: str,
    max_tokens: int = MAX_COMPLETION_LENGTH,
) -> str:
    url = f"{endpoint.rstrip('/')}/chat/completions"
    for attempt in range(3):
        try:
            r = requests.post(
                url,
                json={
                    "model": model_id,
                    "messages": [
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0.0,
                    "max_tokens": int(max_tokens),
                    "chat_template_kwargs": {"enable_thinking": False},
                },
                timeout=90.0,
            )
            r.raise_for_status()
            msg = r.json()["choices"][0]["message"]
            return str(msg.get("content") or "").strip()
        except Exception as exc:
            if attempt == 2:
                return f"ERROR: {exc}"
            time.sleep(1.0)
    return ""


def _eval_nemotron_wfs_l4(
    endpoint: str,
    model_id: str,
    eps: Sequence[EpistemicEpisode],
    *,
    max_tokens: int = MAX_COMPLETION_LENGTH,
    workers: int = 12,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        comps = list(
            pool.map(
                lambda ep: _call_nemotron_one(endpoint, model_id, GRPO_SYSTEM_PROMPT_L4, ep.prompt, max_tokens),
                eps,
            )
        )
    return _score_wfs_completions(eps, comps, time.perf_counter() - t0)


def _eval_nemotron_d1_l4(
    endpoint: str,
    model_id: str,
    cases: Sequence[Mapping[str, Any]],
    user_prompts: Sequence[str],
    *,
    max_tokens: int = MAX_COMPLETION_LENGTH,
    workers: int = 12,
) -> Dict[str, Any]:
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        comps = list(
            pool.map(
                lambda up: _call_nemotron_one(endpoint, model_id, D1_PROSE_TRANSFER_SYSTEM_PROMPT, up, max_tokens),
                user_prompts,
            )
        )
    return _summarize_d1_completions(cases, comps, time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# Protocol v2 Helpers: TRL Startup Check, DEV Selection, Config Digests, Freeze
# ---------------------------------------------------------------------------
def compute_dev_eval_steps(run_steps: int) -> Tuple[int, ...]:
    """Return DEV evaluation step schedule: every 10 steps for `run_steps >= 10`,
    or `every max(1, run_steps // 6)` in short smoke tests, always including `run_steps`.
    """
    steps_i = int(run_steps)
    if steps_i <= 0:
        raise ValueError(f"run_steps must be positive, got {run_steps}")
    if steps_i >= 10:
        pts = set(range(10, steps_i + 1, 10))
        pts.add(steps_i)
        return tuple(sorted(pts))
    stride = max(1, steps_i // 6)
    pts = set(range(stride, steps_i + 1, stride))
    pts.add(steps_i)
    return tuple(sorted(pts))


class BestDevSelector:
    """DEV-only checkpoint selector with `step=0` SFT baseline & `det_em >= sft_det_em - margin` guard.

    - Records `step=0` SFT DEV metrics as the initial baseline in `history`.
    - A `step > 0` checkpoint is eligible ONLY if
      `dev_determinate_exact_match >= sft_dev_det_em - non_inferiority_margin`.
    - Promotes an eligible `step > 0` checkpoint ONLY if it strictly beats the current best
      (starting from `step=0` SFT) on `dev_exact_match` (tie-break: `dev_mean_total_reward`).
    - If NO `step > 0` checkpoint is eligible, records
      `best_dev_status = "none_eligible_fell_back_to_step0_sft"` and `fell_back_to_step0_sft = True`
      (using `merged_sft_dir`, never copying `final_step` into `best_dev_dir`).
    - If eligible `step > 0` checkpoints exist but none beat `step=0` SFT, records
      `best_dev_status = "step0_sft_retained_no_eligible_step_improved"` and `fell_back_to_step0_sft = True`.
    """

    def __init__(
        self,
        sft_dev_metrics: Mapping[str, Any],
        *,
        non_inferiority_margin: float = DETERMINATE_NON_INFERIORITY_MARGIN,
    ) -> None:
        self.margin = float(non_inferiority_margin)
        self.sft_dev_exact_match = float(sft_dev_metrics["exact_match"])
        self.sft_dev_det_em = float(sft_dev_metrics["determinate_exact_match"])
        self.sft_dev_mean_reward = float(sft_dev_metrics["mean_total_reward"])
        self.det_em_floor = self.sft_dev_det_em - self.margin

        self.best_step: int = 0
        self.best_exact_match: float = self.sft_dev_exact_match
        self.best_mean_reward: float = self.sft_dev_mean_reward
        self.eligible_grpo_steps_count: int = 0

        sft_abstain = float(sft_dev_metrics["abstain_rate"])
        self.history: List[Dict[str, Any]] = [
            {
                "step": 0,
                "source": "step0_sft_baseline",
                "eligible": True,
                "det_em_floor": round(self.det_em_floor, 4),
                "dev_exact_match": round(self.sft_dev_exact_match, 4),
                "dev_determinate_exact_match": round(self.sft_dev_det_em, 4),
                "dev_u_abstention_exact_match": round(float(sft_dev_metrics["u_abstention_exact_match"]), 4),
                "dev_mean_total_reward": round(self.sft_dev_mean_reward, 4),
                "dev_abstain_rate": round(sft_abstain, 4),
                "dev_abstain_in_band_025_045_diagnostic": bool(0.25 <= sft_abstain <= 0.45),
                "dev_false_abstain_rate": round(float(sft_dev_metrics["false_abstain_rate_on_determinate"]), 4),
                "dev_false_commit_on_u": round(float(sft_dev_metrics["false_commit_rate_on_u"]), 4),
            }
        ]

    def record_step(self, step: int, dev_metrics: Mapping[str, Any]) -> Tuple[Dict[str, Any], bool]:
        """Record a `step > 0` DEV evaluation and return `(record, should_save_best_dev)`."""
        step_i = int(step)
        if step_i <= 0:
            raise ValueError(f"BestDevSelector.record_step requires step > 0, got {step_i}")

        dev_em = float(dev_metrics["exact_match"])
        dev_det_em = float(dev_metrics["determinate_exact_match"])
        dev_rew = float(dev_metrics["mean_total_reward"])
        dev_abstain = float(dev_metrics["abstain_rate"])

        eligible = bool(dev_det_em >= self.det_em_floor - 1e-12)
        if eligible:
            self.eligible_grpo_steps_count += 1

        should_save = False
        if eligible:
            is_better = (dev_em > self.best_exact_match + 1e-12) or (
                abs(dev_em - self.best_exact_match) <= 1e-12 and dev_rew > self.best_mean_reward + 1e-12
            )
            if is_better:
                self.best_step = step_i
                self.best_exact_match = dev_em
                self.best_mean_reward = dev_rew
                should_save = True

        rec = {
            "step": step_i,
            "source": "grpo_checkpoint",
            "eligible": eligible,
            "promoted_to_best_dev": should_save,
            "det_em_floor": round(self.det_em_floor, 4),
            "dev_exact_match": round(dev_em, 4),
            "dev_determinate_exact_match": round(dev_det_em, 4),
            "dev_u_abstention_exact_match": round(float(dev_metrics["u_abstention_exact_match"]), 4),
            "dev_mean_total_reward": round(dev_rew, 4),
            "dev_abstain_rate": round(dev_abstain, 4),
            "dev_abstain_in_band_025_045_diagnostic": bool(0.25 <= dev_abstain <= 0.45),
            "dev_false_abstain_rate": round(float(dev_metrics["false_abstain_rate_on_determinate"]), 4),
            "dev_false_commit_on_u": round(float(dev_metrics["false_commit_rate_on_u"]), 4),
        }
        self.history.append(rec)
        return rec, should_save

    def summary(self) -> Dict[str, Any]:
        if self.best_step > 0:
            status = "selected_eligible_grpo_step"
            fell_back = False
        elif self.eligible_grpo_steps_count == 0:
            status = "none_eligible_fell_back_to_step0_sft"
            fell_back = True
        else:
            status = "step0_sft_retained_no_eligible_step_improved"
            fell_back = True
        return {
            "best_dev_step": int(self.best_step),
            "best_dev_status": status,
            "fell_back_to_step0_sft": fell_back,
            "eligible_grpo_steps_count": int(self.eligible_grpo_steps_count),
            "sft_dev_determinate_exact_match": round(self.sft_dev_det_em, 4),
            "det_em_eligibility_floor": round(self.det_em_floor, 4),
            "best_dev_exact_match": round(self.best_exact_match, 4),
            "best_dev_mean_total_reward": round(self.best_mean_reward, 4),
            "dev_trajectory": list(self.history),
        }


def compute_unscaled_group_advantages(
    rewards: Sequence[float],
    *,
    scale_rewards: Any,
) -> np.ndarray:
    """Compute group-mean-centered advantages respecting `scale_rewards` (`'none'` / `False` skips std division)."""
    arr = np.asarray(rewards, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError(f"Expected non-empty 1-D reward vector, got shape {arr.shape}")
    centered = arr - float(np.mean(arr))
    if scale_rewards in ("none", False):
        return centered
    std_val = float(np.std(arr, ddof=1 if arr.size > 1 else 0))
    return centered / (std_val + 1e-4)


def verify_unscaled_group_advantages(
    scale_rewards_value: Any,
    grpo_trainer_cls: Any = None,
) -> Dict[str, Any]:
    """Deterministic startup self-check verifying unequal-spread group rewards mean-center
    without dividing by std (`[1.0, -1.0, -1.0, -1.0] -> [+1.5, -0.5, -0.5, -0.5]`, and
    two unequal-spread groups preserve their 5:1 spread ratio without std normalization).
    """
    if scale_rewards_value not in ("none", False):
        raise RuntimeError(
            f"Refusing to run with std-scaled GRPO advantages: scale_rewards={scale_rewards_value!r}"
        )

    probe_primary = np.array([1.0, -1.0, -1.0, -1.0], dtype=np.float64)
    expected_primary = np.array([1.5, -0.5, -0.5, -0.5], dtype=np.float64)
    adv_primary = compute_unscaled_group_advantages(probe_primary, scale_rewards=scale_rewards_value)
    if not np.allclose(adv_primary, expected_primary, atol=1e-9):
        raise RuntimeError(
            f"Unscaled advantage self-check failed on [1,-1,-1,-1]: got {adv_primary.tolist()}, expected {expected_primary.tolist()}"
        )
    pop_std_scaled = (probe_primary - float(np.mean(probe_primary))) / float(np.std(probe_primary, ddof=0))
    if np.allclose(adv_primary, pop_std_scaled, atol=1e-3):
        raise RuntimeError("Unscaled advantage check unexpectedly matched population-std-divided advantages")

    # Second unequal-spread probe where std != 1.0 under BOTH ddof=0 and ddof=1:
    probe_wide = np.array([1.05, -1.0, -1.0, -1.0], dtype=np.float64)
    probe_narrow = np.array([0.2, -0.2, -0.2, -0.2], dtype=np.float64)
    adv_wide = compute_unscaled_group_advantages(probe_wide, scale_rewards=scale_rewards_value)
    adv_narrow = compute_unscaled_group_advantages(probe_narrow, scale_rewards=scale_rewards_value)
    if not np.allclose(adv_wide, np.array([1.5375, -0.5125, -0.5125, -0.5125]), atol=1e-9):
        raise RuntimeError(f"Unscaled wide-group check failed: {adv_wide.tolist()}")
    if not np.allclose(adv_narrow, np.array([0.3, -0.1, -0.1, -0.1]), atol=1e-9):
        raise RuntimeError(f"Unscaled narrow-group check failed: {adv_narrow.tolist()}")
    if abs(float(adv_wide[0]) - float(adv_narrow[0])) < 1.0:
        raise RuntimeError(
            f"Unequal-spread groups collapsed to identical advantage magnitude: wide={adv_wide[0]}, narrow={adv_narrow[0]}"
        )

    trainer_source_checked = False
    if grpo_trainer_cls is not None:
        gen_fn = getattr(grpo_trainer_cls, "_generate_and_score_completions", None)
        comp_fn = getattr(grpo_trainer_cls, "_compute_advantages", None)
        src_parts: List[str] = []
        for fn in (gen_fn, comp_fn):
            if fn is not None:
                src_parts.append(inspect.getsource(fn))
        if not src_parts:
            raise RuntimeError("Could not locate GRPOTrainer advantage computation source for verification")
        combined_src = "\n".join(src_parts)
        if "scale_rewards" not in combined_src:
            raise RuntimeError("GRPOTrainer source does not reference 'scale_rewards'; cannot certify unscaled advantages")
        trainer_source_checked = True

    return {
        "passed": True,
        "scale_rewards_value": scale_rewards_value,
        "primary_probe_rewards": probe_primary.tolist(),
        "primary_probe_advantages": [round(float(x), 6) for x in adv_primary],
        "wide_vs_narrow_max_advantage": [round(float(adv_wide[0]), 6), round(float(adv_narrow[0]), 6)],
        "trainer_source_checked": trainer_source_checked,
    }


def resolve_and_verify_trl_grpo_config_kwargs(
    grpo_config_cls: Any,
    grpo_trainer_cls: Any = None,
) -> Dict[str, Any]:
    """Fail-closed startup check on TRL `GRPOConfig` signature:
    - Requires `"scale_rewards"` in `GRPOConfig.__init__`.
    - Passes `scale_rewards="none"` for string-typed parameter (TRL 1.18.2: `scale_rewards: str = 'group'`),
      or `scale_rewards=False` ONLY if default is bool-typed.
    - Passes `loss_type="dr_grpo"` if `"loss_type"` is supported by `GRPOConfig`.
    - Runs `verify_unscaled_group_advantages` self-check.
    """
    sig = inspect.signature(grpo_config_cls.__init__)
    params = sig.parameters
    if "scale_rewards" not in params:
        raise RuntimeError(
            "TRL GRPOConfig.__init__ is missing required 'scale_rewards' parameter; "
            "Protocol v2 fails closed rather than silently running std-scaled GRPO."
        )

    sr_param = params["scale_rewards"]
    if isinstance(sr_param.default, bool):
        scale_rewards_val: Any = False
    else:
        scale_rewards_val = PROTOCOL_V2_SCALE_REWARDS  # "none"

    loss_type_supported = "loss_type" in params
    loss_type_val: Optional[str] = PROTOCOL_V2_LOSS_TYPE if loss_type_supported else None

    self_check = verify_unscaled_group_advantages(scale_rewards_val, grpo_trainer_cls=grpo_trainer_cls)
    extra_kwargs: Dict[str, Any] = {"scale_rewards": scale_rewards_val}
    if loss_type_supported and loss_type_val is not None:
        extra_kwargs["loss_type"] = loss_type_val
    if "max_prompt_length" in params:
        extra_kwargs["max_prompt_length"] = MAX_PROMPT_LENGTH

    return {
        "scale_rewards": scale_rewards_val,
        "loss_type": loss_type_val,
        "loss_type_supported": loss_type_supported,
        "grpo_config_has_beta": "beta" in params,
        "grpo_config_has_max_prompt_length": "max_prompt_length" in params,
        "extra_kwargs": extra_kwargs,
        "unscaled_advantage_self_check": self_check,
    }


def compute_script_sha256(script_path: Optional[Path] = None) -> str:
    return sha256_file(script_path or Path(__file__).resolve())


def compute_stage1_config_digest(
    *,
    base_model_id: str,
    sft_split_sha256: str,
    dev_split_sha256: str,
    grpo_split_sha256: str,
    amendment_sha256: str,
    false_abstain_reward: float,
    script_sha256: str,
) -> str:
    payload = {
        "protocol_version": PROTOCOL_V2_VERSION,
        "base_model_id": str(base_model_id),
        "sft_split_sha256": str(sft_split_sha256),
        "dev_split_sha256": str(dev_split_sha256),
        "grpo_split_sha256": str(grpo_split_sha256),
        "amendment_sha256": str(amendment_sha256),
        "false_abstain_reward": round(float(false_abstain_reward), 6),
        "wrong_determinate_reward": round(float(WRONG_DETERMINATE_REWARD), 6),
        "false_commit_on_u_reward": round(float(FALSE_COMMIT_ON_U_REWARD), 6),
        "script_sha256": str(script_sha256),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def compute_grpo_seed_config_digest(
    *,
    seed: int,
    run_steps: int,
    lr: float,
    beta: float,
    lora_r: int,
    lora_alpha: int,
    grad_accum: int,
    scale_rewards: Any,
    loss_type: Any,
    false_abstain_reward: float,
    script_sha256: str,
    merged_sft_sha256: str = "",
    amendment_sha256: str = "",
    grpo_split_sha256: str = "",
    dev_split_sha256: str = "",
) -> str:
    """Bind per-seed resume keys to full Protocol v2 hyperparameters + code + SFT + amendment digests."""
    payload = {
        "protocol_version": PROTOCOL_V2_VERSION,
        "seed": int(seed),
        "run_steps": int(run_steps),
        "lr": f"{float(lr):.12g}",
        "beta": f"{float(beta):.12g}",
        "lora_r": int(lora_r),
        "lora_alpha": int(lora_alpha),
        "grad_accum": int(grad_accum),
        "scale_rewards": str(scale_rewards),
        "loss_type": str(loss_type),
        "false_abstain_reward": round(float(false_abstain_reward), 6),
        "script_sha256": str(script_sha256),
        "merged_sft_sha256": str(merged_sft_sha256),
        "amendment_sha256": str(amendment_sha256),
        "grpo_split_sha256": str(grpo_split_sha256),
        "dev_split_sha256": str(dev_split_sha256),
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def can_resume_seed_checkpoint(
    seed_ckpt_path: Path,
    expected_config_digest: str,
    final_adapter_dir: Path,
    best_dev_dir: Path,
) -> Tuple[bool, Optional[Dict[str, Any]], str]:
    """Validate a Protocol v2 per-seed training checkpoint strictly against its config digest and file hashes."""
    if not seed_ckpt_path.is_file():
        return False, None, "checkpoint_missing"
    try:
        ck = json.loads(seed_ckpt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return False, None, f"checkpoint_unreadable:{type(exc).__name__}"
    if ck.get("protocol_version") != PROTOCOL_V2_VERSION:
        return False, None, "protocol_version_mismatch"
    if ck.get("config_digest") != expected_config_digest:
        return False, None, "config_digest_mismatch"
    rec = ck.get("seed_train_record")
    if not isinstance(rec, dict):
        return False, None, "missing_seed_train_record"
    if not (final_adapter_dir / "adapter_model.safetensors").is_file():
        return False, None, "final_adapter_missing"
    expected_final_hashes = rec.get("final_adapter_sha256")
    if not isinstance(expected_final_hashes, dict) or hash_directory_files(final_adapter_dir) != expected_final_hashes:
        return False, None, "final_adapter_hash_mismatch"
    fell_back = bool(rec.get("fell_back_to_step0_sft"))
    if not fell_back:
        expected_best_hashes = rec.get("best_dev_adapter_sha256")
        if (
            not best_dev_dir.is_dir()
            or not isinstance(expected_best_hashes, dict)
            or hash_directory_files(best_dev_dir) != expected_best_hashes
        ):
            return False, None, "best_dev_adapter_hash_mismatch"
    return True, rec, "ok"


def build_and_write_selection_frozen(
    *,
    output_dir: Path,
    merged_sft_dir: Path,
    seed_train_records: Mapping[str, Mapping[str, Any]],
    amendment_path: Path,
    script_path: Optional[Path] = None,
    presealed_splits_sha256: Mapping[str, str],
) -> Tuple[Path, Dict[str, Any]]:
    """Atomically write `selection_frozen.json` locking SHA-256 digests of `merged_sft_dir`,
    every seed's `final_step` & eligible `best_dev` adapter files, `d9b_amendment_v1.json`,
    and the script BEFORE any Stage 3 post-freeze holdout evaluation runs.
    """
    if not merged_sft_dir.is_dir():
        raise FileNotFoundError(f"merged_sft_dir does not exist: {merged_sft_dir}")
    if not amendment_path.is_file():
        raise FileNotFoundError(f"d9b_amendment_v1.json does not exist: {amendment_path}")
    resolved_script = (script_path or Path(__file__)).resolve()

    merged_sft_hashes = hash_directory_files(merged_sft_dir)
    if not merged_sft_hashes:
        raise AssertionError(f"merged_sft_dir has no files to freeze: {merged_sft_dir}")

    frozen_seeds: Dict[str, Dict[str, Any]] = {}
    for sk in sorted(seed_train_records.keys(), key=lambda s: int(s.split("_")[1])):
        rec = seed_train_records[sk]
        final_dir = Path(str(rec["final_adapter_dir"]))
        if not final_dir.is_dir():
            raise FileNotFoundError(f"Missing final_step_adapter for {sk}: {final_dir}")
        final_hashes = hash_directory_files(final_dir)
        if not final_hashes:
            raise AssertionError(f"Empty final_step_adapter for {sk}: {final_dir}")

        fell_back = bool(rec["fell_back_to_step0_sft"])
        best_step = int(rec["best_dev_step"])
        best_status = str(rec["best_dev_status"])
        if fell_back:
            if best_step != 0:
                raise AssertionError(f"Inconsistent step0 fallback on {sk}: best_dev_step={best_step} != 0")
            best_dir_str: Optional[str] = None
            best_hashes: Optional[Dict[str, str]] = None
            best_source = "merged_sft_dir"
        else:
            best_dir = Path(str(rec["best_dev_adapter_dir"]))
            if not best_dir.is_dir():
                raise FileNotFoundError(f"Missing eligible best_dev_adapter for {sk}: {best_dir}")
            best_hashes = hash_directory_files(best_dir)
            if not best_hashes:
                raise AssertionError(f"Empty eligible best_dev_adapter for {sk}: {best_dir}")
            best_dir_str = str(best_dir.resolve())
            best_source = "best_dev_adapter_dir"

        frozen_seeds[sk] = {
            "seed": int(rec["seed"]),
            "config_digest": str(rec["config_digest"]),
            "final_step_adapter_dir": str(final_dir.resolve()),
            "final_step_adapter_sha256": final_hashes,
            "best_dev_step": best_step,
            "best_dev_status": best_status,
            "fell_back_to_step0_sft": fell_back,
            "best_dev_source": best_source,
            "best_dev_adapter_dir": best_dir_str,
            "best_dev_adapter_sha256": best_hashes,
        }

    frozen_payload: Dict[str, Any] = {
        "schema_version": "wave11_d9b_selection_frozen_v2",
        "protocol_version": PROTOCOL_V2_VERSION,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "eval_virgin_v2_scope": "virgin_for_grpo_arm_only",
        "in_loop_holdout_evaluations_performed": 0,
        "script_path": str(resolved_script),
        "script_sha256": sha256_file(resolved_script),
        "amendment_path": str(amendment_path.resolve()),
        "amendment_sha256": sha256_file(amendment_path),
        "merged_sft_dir": str(merged_sft_dir.resolve()),
        "merged_sft_dir_sha256": merged_sft_hashes,
        "presealed_splits_sha256": dict(sorted(presealed_splits_sha256.items())),
        "seeds": frozen_seeds,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    frozen_path = output_dir / "selection_frozen.json"
    write_json_atomic(frozen_path, frozen_payload)
    return frozen_path, frozen_payload


def verify_selection_frozen_manifest(frozen_path: Path) -> Dict[str, Any]:
    """Fail-closed byte-for-byte digest verification of `selection_frozen.json` before Stage 3 scoring.

    Refuses to proceed if ANY file in `merged_sft_dir`, any seed's `final_step_adapter` or
    eligible `best_dev_adapter`, `d9b_amendment_v1.json`, or the script changed on disk.
    """
    if not frozen_path.is_file():
        raise FileNotFoundError(f"Refusing to run Stage 3 evaluation: missing {frozen_path}")
    payload = json.loads(frozen_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != "wave11_d9b_selection_frozen_v2":
        raise AssertionError(f"Unexpected selection_frozen schema_version: {payload.get('schema_version')!r}")

    script_p = Path(str(payload["script_path"]))
    if not script_p.is_file() or sha256_file(script_p) != payload["script_sha256"]:
        raise AssertionError(f"selection_frozen.json script SHA-256 mismatch on {script_p}")

    amend_p = Path(str(payload["amendment_path"]))
    if not amend_p.is_file() or sha256_file(amend_p) != payload["amendment_sha256"]:
        raise AssertionError(f"selection_frozen.json amendment SHA-256 mismatch on {amend_p}")

    sft_p = Path(str(payload["merged_sft_dir"]))
    if not sft_p.is_dir() or hash_directory_files(sft_p) != payload["merged_sft_dir_sha256"]:
        raise AssertionError(f"selection_frozen.json merged_sft_dir SHA-256 mismatch on {sft_p}")

    seeds_map = payload.get("seeds")
    if not isinstance(seeds_map, dict) or not seeds_map:
        raise AssertionError("selection_frozen.json has no frozen seeds")

    for sk, entry in seeds_map.items():
        fin_p = Path(str(entry["final_step_adapter_dir"]))
        if not fin_p.is_dir() or hash_directory_files(fin_p) != entry["final_step_adapter_sha256"]:
            raise AssertionError(f"selection_frozen.json final_step_adapter SHA-256 mismatch for {sk} at {fin_p}")
        if bool(entry["fell_back_to_step0_sft"]):
            if int(entry["best_dev_step"]) != 0 or entry.get("best_dev_adapter_dir") is not None:
                raise AssertionError(f"Invalid step0 fallback record in selection_frozen.json for {sk}")
        else:
            best_p = Path(str(entry["best_dev_adapter_dir"]))
            if not best_p.is_dir() or hash_directory_files(best_p) != entry["best_dev_adapter_sha256"]:
                raise AssertionError(f"selection_frozen.json best_dev_adapter SHA-256 mismatch for {sk} at {best_p}")

    return payload


def _verify_fp32_merge_and_unload_parity(
    merged_sft_dir: Path,
    adapter_dir: Path,
    probe_inputs: Mapping[str, Any],
    *,
    min_probes: int = 8,
) -> Dict[str, Any]:
    """Verify pure `float32`-vs-`float32` `merge_and_unload()` parity (`math.isfinite(diff) and diff < 1e-4`)
    on CPU across `min(8, n_available)` probe sequences (asserting `>= min_probes`), casting
    `peft_fp32 = peft_fp32.float()` explicitly and logging parameter dtypes.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    n_avail = int(probe_inputs["input_ids"].shape[0])
    n_probes = min(max(8, int(min_probes)), n_avail)
    if n_probes < int(min_probes):
        raise AssertionError(
            f"_verify_fp32_merge_and_unload_parity requires >= {min_probes} probe sequences, got {n_avail}"
        )

    base_fp32 = AutoModelForCausalLM.from_pretrained(
        str(merged_sft_dir), dtype=torch.float32, local_files_only=True
    )
    peft_fp32 = PeftModel.from_pretrained(
        base_fp32, str(adapter_dir), is_trainable=False, autocast_adapter_dtype=False
    )
    peft_fp32 = peft_fp32.float()
    peft_fp32.eval()
    unmerged_param_dtypes = sorted({str(p.dtype) for p in peft_fp32.parameters()})
    if unmerged_param_dtypes != ["torch.float32"]:
        raise AssertionError(f"Expected pure torch.float32 unmerged PEFT params, got {unmerged_param_dtypes}")

    unmerged_chunks: List[Any] = []
    with torch.inference_mode():
        for idx in range(n_probes):
            in_ids = probe_inputs["input_ids"][idx : idx + 1]
            attn = probe_inputs["attention_mask"][idx : idx + 1]
            unmerged_chunks.append(
                peft_fp32(input_ids=in_ids, attention_mask=attn, use_cache=False).logits[:, -8:, :].detach()
            )

    merged_fp32 = peft_fp32.merge_and_unload()
    merged_fp32.eval()
    merged_param_dtypes = sorted({str(p.dtype) for p in merged_fp32.parameters()})
    if merged_param_dtypes != ["torch.float32"]:
        raise AssertionError(f"Expected pure torch.float32 merged params, got {merged_param_dtypes}")

    max_diff = 0.0
    with torch.inference_mode():
        for idx in range(n_probes):
            in_ids = probe_inputs["input_ids"][idx : idx + 1]
            attn = probe_inputs["attention_mask"][idx : idx + 1]
            merged_logits = merged_fp32(input_ids=in_ids, attention_mask=attn, use_cache=False).logits[:, -8:, :].detach()
            probe_diff = float((merged_logits - unmerged_chunks[idx]).abs().max().item())
            if not math.isfinite(probe_diff):
                raise AssertionError(f"Non-finite fp32 merge_and_unload() parity diff on probe {idx}: {probe_diff}")
            if probe_diff > max_diff:
                max_diff = probe_diff

    del merged_fp32, peft_fp32, base_fp32, unmerged_chunks
    gc.collect()

    is_finite = math.isfinite(max_diff)
    passed = bool(is_finite and max_diff < 1e-4)
    report = {
        "max_abs_logit_diff": max_diff,
        "n_probes": n_probes,
        "positions_per_probe": 8,
        "unmerged_param_dtypes": unmerged_param_dtypes,
        "merged_param_dtypes": merged_param_dtypes,
        "is_finite": is_finite,
        "passed": passed,
    }
    if not passed:
        raise AssertionError(f"float32 merge_and_unload() parity failed: {report}")
    return report


def run_part_b_u_grpo_on_spark2(
    *,
    output_dir: Path,
    osv_cache_path: Path,
    base_model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    nemotron_endpoint: str = "http://127.0.0.1:8890/v1",
    nemotron_model: str = "nemotron-lightning",
    grpo_steps: int = PROTOCOL_V2_GRPO_STEPS,
    seeds: Sequence[int] = PREREGISTERED_SEEDS,
    lr: float = PROTOCOL_V2_LR,
    beta: float = PROTOCOL_V2_BETA,
    lora_r: int = PROTOCOL_V2_LORA_R,
    lora_alpha: int = PROTOCOL_V2_LORA_ALPHA,
    grad_accum: int = PROTOCOL_V2_GRAD_ACCUM,
    amendment_path: Path = DEFAULT_D9B_AMENDMENT_PATH,
) -> Dict[str, Any]:
    """Execute Part B (`Protocol v2`) on Spark #2 with strict out-of-loop holdout lock:
      - Fail-closed TRL `GRPOConfig` check (`scale_rewards="none"`, `loss_type="dr_grpo"`) + synthetic advantage self-check.
      - Stage 1 (DEV-only): Base 7B DEV eval + Completion-only SFT (`N=128`, or verified `merged_sft_dir` reuse).
      - Stage 2 (DEV-only): 3-seed confirmatory `(137, 2026, 31415)` `Epistemic-WFS-GRPO` (`DEV` every 10 steps,
        `step=0` SFT baseline, `det_em >= sft_det_em - 0.03` guard, pure `fp32` 8-probe parity `< 1e-4`).
      - Selection Freeze: Write & verify `selection_frozen.json` byte-for-byte before ANY holdout scoring.
      - Stage 3 (One-Shot Post-Freeze): Score `Always_Abstain`, `Base 7B`, `SFT 7B`, all 3 GRPO seeds
        (`final_step` PRIMARY + `best_dev` SECONDARY), and `Bare Nemotron-30B` on `EVAL_VIRGIN_V2`,
        `EVAL_WAVE6_LOOK2`, and `D1 (k=17, k=5)`.
      - Stage 4: Report 3-seed `(137, 2026, 31415)` AND 2-seed `(137, 2026)` means, paired bootstrap +
        Holm-Bonferroni (`alpha=0.05`), and honest hypothesis outcome (`"K4 stands under Protocol v2"` on null).
    """
    import peft
    import torch
    import transformers
    import trl
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments
    from trl import GRPOConfig, GRPOTrainer

    t_pipeline0 = time.perf_counter()
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = output_dir / "_work" / "part_b"
    work_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Spark #2 Part B GPU training pipeline")
    if not amendment_path.is_file():
        raise FileNotFoundError(
            f"Pre-launch Protocol v2 amendment file is required before running Part B: {amendment_path}"
        )

    amendment_sha256 = sha256_file(amendment_path)
    script_sha256 = compute_script_sha256()

    grpo_trn_params = list(inspect.signature(GRPOTrainer.__init__).parameters.keys())
    calc_rew_src = inspect.getsource(GRPOTrainer._calculate_rewards)
    if "prompts=prompts, completions=completions" not in calc_rew_src:
        raise RuntimeError("Unexpected TRL GRPOTrainer._calculate_rewards call signature")

    # Fail-closed TRL GRPOConfig + unscaled advantage startup self-check
    trl_cfg_check = resolve_and_verify_trl_grpo_config_kwargs(GRPOConfig, GRPOTrainer)
    resolved_scale_rewards = trl_cfg_check["scale_rewards"]
    resolved_loss_type = trl_cfg_check["loss_type"]

    env_versions = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "trl": trl.__version__,
        "hf_home": os.environ.get("HF_HOME", ""),
        "grpo_trainer_accepts_peft_config": "peft_config" in grpo_trn_params,
        "grpo_config_has_beta": trl_cfg_check["grpo_config_has_beta"],
        "scale_rewards": resolved_scale_rewards,
        "loss_type": resolved_loss_type,
        "loss_type_supported": trl_cfg_check["loss_type_supported"],
        "unscaled_advantage_self_check_passed": trl_cfg_check["unscaled_advantage_self_check"]["passed"],
        "script_sha256": script_sha256,
        "amendment_sha256": amendment_sha256,
    }
    print(f"[Part-B-GPU] Protocol v2 Environment verified: {json.dumps(env_versions)}", flush=True)

    # 1. Pre-seal ALL 5 disjoint splits (`EVAL_VIRGIN_V2` locked BEFORE any training!) + D1 corpus metadata
    splits = build_all_five_disjoint_splits()
    d1_cases, d1_docs, d1_corpus_sha = build_prospective_realworld_drift_suite(osv_cache_path)
    disjoint_proof = assert_five_way_disjoint_splits(splits, d1_cases=d1_cases, verify_wave6_eval_sha=True)
    write_json_atomic(work_dir / "presealed_splits_lock.json", disjoint_proof)
    split_shas = {k: str(v["split_prompt_bundle_sha256"]) for k, v in disjoint_proof["splits"].items()}
    print(
        f"[Part-B-GPU] 5-way disjoint splits pre-sealed: "
        f"EVAL_VIRGIN_V2={disjoint_proof['eval_virgin_v2_presealed_sha256'][:12]}..., "
        f"EVAL_WAVE6_LOOK2_MATCH={disjoint_proof['wave6_eval_sha256_matches_byte_identical']}",
        flush=True,
    )

    tok = AutoTokenizer.from_pretrained(base_model_id, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"

    split_prompt_token_max: Dict[str, int] = {}
    for s_name, s_eps in splits.items():
        lens = [
            len(
                tok(
                    tok.apply_chat_template(
                        [{"role": "system", "content": GRPO_SYSTEM_PROMPT_L4}, {"role": "user", "content": ep.prompt}],
                        tokenize=False,
                        add_generation_prompt=True,
                    ),
                    add_special_tokens=False,
                )["input_ids"]
            )
            for ep in s_eps
        ]
        split_prompt_token_max[s_name] = int(max(lens))
        if max(lens) > MAX_PROMPT_LENGTH:
            raise AssertionError(f"{s_name} prompt max tokens {max(lens)} > {MAX_PROMPT_LENGTH}")

    def generate_batched_greedy(
        mdl: Any,
        chat_pairs: Sequence[Tuple[str, str]],
        *,
        batch_size: int = 16,
        max_new_tokens: int = MAX_COMPLETION_LENGTH,
    ) -> List[str]:
        mdl.eval()
        tok.padding_side = "left"
        rendered = [
            tok.apply_chat_template(
                [{"role": "system", "content": sp}, {"role": "user", "content": up}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for sp, up in chat_pairs
        ]
        outputs: List[str] = []
        dev = next(mdl.parameters()).device
        for start in range(0, len(rendered), batch_size):
            batch_txt = rendered[start : start + batch_size]
            enc = tok(batch_txt, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            in_len = enc["input_ids"].shape[1]
            with torch.inference_mode():
                gen = mdl.generate(
                    **enc,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                    pad_token_id=tok.pad_token_id,
                    eos_token_id=tok.eos_token_id,
                    use_cache=True,
                )
            for row in gen[:, in_len:]:
                outputs.append(tok.decode(row, skip_special_tokens=True))
        return outputs

    def eval_wfs(mdl: Any, eps: Sequence[EpistemicEpisode], batch_size: int = 24) -> Dict[str, Any]:
        pairs = [(GRPO_SYSTEM_PROMPT_L4, ep.prompt) for ep in eps]
        t0 = time.perf_counter()
        comps = generate_batched_greedy(mdl, pairs, batch_size=batch_size, max_new_tokens=MAX_COMPLETION_LENGTH)
        return _score_wfs_completions(eps, comps, time.perf_counter() - t0)

    def eval_d1(mdl: Any, user_prompts: Sequence[str], batch_size: int = 30) -> Dict[str, Any]:
        pairs = [(D1_PROSE_TRANSFER_SYSTEM_PROMPT, up) for up in user_prompts]
        t0 = time.perf_counter()
        comps = generate_batched_greedy(mdl, pairs, batch_size=batch_size, max_new_tokens=MAX_COMPLETION_LENGTH)
        return _summarize_d1_completions(d1_cases, comps, time.perf_counter() - t0)

    # 8 DEV probe prompts for pure fp32 merge_and_unload parity & logit shift checks
    n_probe_prompts = min(8, len(splits["DEV"]))
    probe_texts = [
        tok.apply_chat_template(
            [{"role": "system", "content": GRPO_SYSTEM_PROMPT_L4}, {"role": "user", "content": splits["DEV"][i].prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for i in range(n_probe_prompts)
    ]
    probe_inputs = tok(probe_texts, return_tensors="pt", padding=True, add_special_tokens=False)

    def compute_probe_logits(mdl: Any) -> torch.Tensor:
        mdl.eval()
        dev = next(mdl.parameters()).device
        with torch.inference_mode():
            out = mdl(
                input_ids=probe_inputs["input_ids"].to(dev),
                attention_mask=probe_inputs["attention_mask"].to(dev),
                use_cache=False,
            )
            return out.logits[:, -8:, :].detach().float().cpu()

    peak_mem_gib_by_stage: Dict[str, float] = {}

    def record_stage_mem(stage: str) -> float:
        torch.cuda.synchronize()
        gib = round(float(torch.cuda.max_memory_allocated()) / (1024.0**3), 3)
        peak_mem_gib_by_stage[stage] = gib
        if gib > 48.0:
            raise RuntimeError(f"Stage {stage} exceeded 48 GB peak memory guardrail: {gib:.2f} GiB")
        return gib

    # -----------------------------------------------------------------------
    # Stage 1 (DEV-ONLY): Base 7B DEV Eval + Untruncated 4-Step SFT (`N=128`)
    #   ZERO holdout (`EVAL_VIRGIN_V2` / `EVAL_WAVE6_LOOK2` / `D1`) evaluations!
    # -----------------------------------------------------------------------
    merged_sft_dir = work_dir / "sft_merged_model"
    stage1_digest = compute_stage1_config_digest(
        base_model_id=base_model_id,
        sft_split_sha256=split_shas["SFT_WARMUP"],
        dev_split_sha256=split_shas["DEV"],
        grpo_split_sha256=split_shas["GRPO_RL"],
        amendment_sha256=amendment_sha256,
        false_abstain_reward=FALSE_ABSTAIN_REWARD,
        script_sha256=script_sha256,
    )
    stage1_ckpt_path = work_dir / f"_ckpt_stage1_w11_v2_{stage1_digest[:12]}.json"
    merged_sft_reused_from_disk = False

    stage1_resumed = False
    if stage1_ckpt_path.is_file() and (merged_sft_dir / "config.json").is_file():
        st1 = json.loads(stage1_ckpt_path.read_text(encoding="utf-8"))
        if (
            st1.get("protocol_version") == PROTOCOL_V2_VERSION
            and st1.get("config_digest") == stage1_digest
            and st1.get("merged_sft_sha256") == hash_directory_files(merged_sft_dir)
        ):
            print(f"[Part-B-GPU] Resuming Stage 1 (DEV-only) from {stage1_ckpt_path}...", flush=True)
            sft_mask_audit = st1["sft_mask_audit"]
            sft_train_sec = float(st1["sft_train_seconds"])
            sft_train_loss_val = float(st1["sft_train_loss"])
            sft_vs_base_logit_shift = float(st1["sft_vs_base_logit_shift"])
            sft_reload_logit_diff = float(st1["sft_reload_logit_diff"])
            sft_grpo_split_res = st1["sft_grpo_split_res"]
            base_dev_res = st1["base_dev_res"]
            sft_dev_res = st1["sft_dev_res"]
            merged_sft_reused_from_disk = bool(st1.get("merged_sft_reused_from_disk", False))
            peak_mem_gib_by_stage.update(st1.get("peak_mem_gib_by_stage") or {})
            sft_rel = AutoModelForCausalLM.from_pretrained(
                str(merged_sft_dir), dtype=torch.bfloat16, local_files_only=True
            ).to("cuda")
            sft_merged_probe_logits = compute_probe_logits(sft_rel)
            del sft_rel
            gc.collect()
            torch.cuda.empty_cache()
            stage1_resumed = True

    if not stage1_resumed and (merged_sft_dir / "config.json").is_file():
        # Reuse existing merged_sft_dir weights ONLY if reload parity < 1e-4 & shift vs base > 1e-4
        print(
            f"[Part-B-GPU] Existing merged_sft_dir found at {merged_sft_dir}; verifying reload parity < 1e-4 "
            "and recomputing DEV-only Stage 1 checkpoint (zero holdout access)...",
            flush=True,
        )
        torch.cuda.reset_peak_memory_stats()
        base_mdl = AutoModelForCausalLM.from_pretrained(
            base_model_id, dtype=torch.bfloat16, local_files_only=True
        ).to("cuda")
        base_probe_logits = compute_probe_logits(base_mdl)
        base_dev_res = eval_wfs(base_mdl, splits["DEV"])
        record_stage_mem("base_7b_dev_eval")
        del base_mdl
        gc.collect()
        torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats()
        sft_rel_1 = AutoModelForCausalLM.from_pretrained(
            str(merged_sft_dir), dtype=torch.bfloat16, local_files_only=True
        ).to("cuda")
        sft_merged_probe_logits = compute_probe_logits(sft_rel_1)
        sft_vs_base_logit_shift = float((sft_merged_probe_logits - base_probe_logits).abs().max().item())
        del sft_rel_1
        gc.collect()
        torch.cuda.empty_cache()

        sft_rel_2 = AutoModelForCausalLM.from_pretrained(
            str(merged_sft_dir), dtype=torch.bfloat16, local_files_only=True
        ).to("cuda")
        sft_reload_logit_diff = float((compute_probe_logits(sft_rel_2) - sft_merged_probe_logits).abs().max().item())
        if not (math.isfinite(sft_reload_logit_diff) and sft_reload_logit_diff < 1e-4):
            raise AssertionError(
                f"Existing merged_sft_dir failed reload parity check: {sft_reload_logit_diff} >= 1e-4"
            )
        if not (math.isfinite(sft_vs_base_logit_shift) and sft_vs_base_logit_shift > 1e-4):
            raise AssertionError(
                f"Existing merged_sft_dir shows no logit shift vs base: {sft_vs_base_logit_shift}"
            )

        sft_grpo_split_res = eval_wfs(sft_rel_2, splits["GRPO_RL"])
        sft_dev_res = eval_wfs(sft_rel_2, splits["DEV"])
        record_stage_mem("sft_reload_and_dev_eval")
        del sft_rel_2
        gc.collect()
        torch.cuda.empty_cache()

        _, sft_mask_audit = build_wave11_completion_only_sft_features(tok, splits["SFT_WARMUP"])
        sft_train_sec = 0.0
        sft_train_loss_val = 0.0
        legacy_st1_path = work_dir / "_ckpt_stage1_w11.json"
        if legacy_st1_path.is_file():
            legacy_st1 = json.loads(legacy_st1_path.read_text(encoding="utf-8"))
            # Read training provenance scalars ONLY (NEVER holdout scores!)
            if "sft_train_seconds" in legacy_st1:
                sft_train_sec = float(legacy_st1["sft_train_seconds"])
            if "sft_train_loss" in legacy_st1:
                sft_train_loss_val = float(legacy_st1["sft_train_loss"])
            legacy_audit = legacy_st1.get("sft_mask_audit") or {}
            if "dataloader_batch_supervised_token_fraction" in legacy_audit:
                sft_mask_audit["dataloader_batch_supervised_token_fraction"] = float(
                    legacy_audit["dataloader_batch_supervised_token_fraction"]
                )
        merged_sft_reused_from_disk = True
        merged_sft_hashes_st1 = hash_directory_files(merged_sft_dir)
        write_json_atomic(
            stage1_ckpt_path,
            {
                "protocol_version": PROTOCOL_V2_VERSION,
                "config_digest": stage1_digest,
                "merged_sft_reused_from_disk": True,
                "merged_sft_sha256": merged_sft_hashes_st1,
                "sft_mask_audit": sft_mask_audit,
                "sft_train_seconds": sft_train_sec,
                "sft_train_loss": sft_train_loss_val,
                "sft_vs_base_logit_shift": sft_vs_base_logit_shift,
                "sft_reload_logit_diff": sft_reload_logit_diff,
                "sft_grpo_split_res": sft_grpo_split_res,
                "base_dev_res": base_dev_res,
                "sft_dev_res": sft_dev_res,
                "peak_mem_gib_by_stage": peak_mem_gib_by_stage,
            },
        )
        print(
            f"[Part-B-GPU] Reused verified merged_sft_dir (reload_diff={sft_reload_logit_diff:.2e}, "
            f"shift={sft_vs_base_logit_shift:.4f}) | SFT DEV={sft_dev_res['exact_match']:.4f} "
            f"(det={sft_dev_res['determinate_exact_match']:.4f}, u_em={sft_dev_res['u_abstention_exact_match']:.4f})",
            flush=True,
        )
    elif not stage1_resumed:
        torch.cuda.reset_peak_memory_stats()
        base_mdl = AutoModelForCausalLM.from_pretrained(
            base_model_id, dtype=torch.bfloat16, local_files_only=True
        ).to("cuda")
        base_probe_logits = compute_probe_logits(base_mdl)
        base_dev_res = eval_wfs(base_mdl, splits["DEV"])
        record_stage_mem("base_7b_dev_eval")

        sft_rows, sft_mask_audit = build_wave11_completion_only_sft_features(tok, splits["SFT_WARMUP"])
        sft_ds = Dataset.from_list(sft_rows)

        def _left_pad_collator(features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
            max_len = max(len(f["input_ids"]) for f in features)
            in_ids, attn, lbls = [], [], []
            for f in features:
                pad_n = max_len - len(f["input_ids"])
                in_ids.append([tok.pad_token_id] * pad_n + f["input_ids"])
                attn.append([0] * pad_n + f["attention_mask"])
                lbls.append([-100] * pad_n + f["labels"])
            return {
                "input_ids": torch.tensor(in_ids, dtype=torch.long),
                "attention_mask": torch.tensor(attn, dtype=torch.long),
                "labels": torch.tensor(lbls, dtype=torch.long),
            }

        torch.manual_seed(42)
        base_mdl.config.use_cache = False
        base_mdl.gradient_checkpointing_enable()
        sft_lora_cfg = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        sft_peft_mdl = get_peft_model(base_mdl, sft_lora_cfg)
        sft_args = TrainingArguments(
            output_dir=str(work_dir / "sft_warmup_r16" / "trainer_out"),
            num_train_epochs=3,
            per_device_train_batch_size=4,
            gradient_accumulation_steps=2,
            learning_rate=8e-5,
            lr_scheduler_type="cosine",
            warmup_steps=4,
            bf16=True,
            logging_steps=4,
            save_strategy="no",
            report_to="none",
            seed=42,
            data_seed=42,
            remove_unused_columns=False,
            gradient_checkpointing=True,
        )
        sft_trainer = Trainer(
            model=sft_peft_mdl,
            args=sft_args,
            train_dataset=sft_ds,
            data_collator=_left_pad_collator,
        )
        sample_batch = next(iter(sft_trainer.get_train_dataloader()))
        sup_frac = float((sample_batch["labels"] != -100).float().mean().item())
        sft_mask_audit["dataloader_batch_supervised_token_fraction"] = round(sup_frac, 4)
        if not (0.02 < sup_frac < 0.50):
            raise AssertionError(f"Unexpected SFT supervised fraction: {sup_frac}")

        t_sft0 = time.perf_counter()
        sft_out = sft_trainer.train()
        sft_train_sec = round(time.perf_counter() - t_sft0, 2)
        sft_train_loss_val = round(float(sft_out.training_loss), 5)

        merged_sft_mdl = sft_peft_mdl.merge_and_unload()
        merged_sft_mdl.save_pretrained(str(merged_sft_dir))
        tok.save_pretrained(str(merged_sft_dir))
        sft_merged_probe_logits = compute_probe_logits(merged_sft_mdl)
        sft_vs_base_logit_shift = float((sft_merged_probe_logits - base_probe_logits).abs().max().item())
        record_stage_mem("sft_train_and_merge")

        del sft_trainer, merged_sft_mdl, sft_peft_mdl, base_mdl
        gc.collect()
        torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats()
        sft_reloaded = AutoModelForCausalLM.from_pretrained(
            str(merged_sft_dir), dtype=torch.bfloat16, local_files_only=True
        ).to("cuda")
        sft_reload_logit_diff = float((compute_probe_logits(sft_reloaded) - sft_merged_probe_logits).abs().max().item())
        if not (math.isfinite(sft_reload_logit_diff) and sft_reload_logit_diff < 1e-4):
            raise AssertionError(f"SFT disk reload logit diff {sft_reload_logit_diff} >= 1e-4")

        sft_grpo_split_res = eval_wfs(sft_reloaded, splits["GRPO_RL"])
        sft_dev_res = eval_wfs(sft_reloaded, splits["DEV"])
        record_stage_mem("sft_reload_and_dev_eval")
        del sft_reloaded
        gc.collect()
        torch.cuda.empty_cache()

        merged_sft_hashes_st1 = hash_directory_files(merged_sft_dir)
        write_json_atomic(
            stage1_ckpt_path,
            {
                "protocol_version": PROTOCOL_V2_VERSION,
                "config_digest": stage1_digest,
                "merged_sft_reused_from_disk": False,
                "merged_sft_sha256": merged_sft_hashes_st1,
                "sft_mask_audit": sft_mask_audit,
                "sft_train_seconds": sft_train_sec,
                "sft_train_loss": sft_train_loss_val,
                "sft_vs_base_logit_shift": sft_vs_base_logit_shift,
                "sft_reload_logit_diff": sft_reload_logit_diff,
                "sft_grpo_split_res": sft_grpo_split_res,
                "base_dev_res": base_dev_res,
                "sft_dev_res": sft_dev_res,
                "peak_mem_gib_by_stage": peak_mem_gib_by_stage,
            },
        )
        print(
            f"[Part-B-GPU] SFT 7B (DEV-only Stage 1) -> DEV={sft_dev_res['exact_match']:.4f} "
            f"(det={sft_dev_res['determinate_exact_match']:.4f}, u_em={sft_dev_res['u_abstention_exact_match']:.4f})",
            flush=True,
        )

    merged_sft_hashes = hash_directory_files(merged_sft_dir)
    merged_sft_dir_digest = hashlib.sha256(
        json.dumps(merged_sft_hashes, sort_keys=True).encode("utf-8")
    ).hexdigest()

    # -----------------------------------------------------------------------
    # Stage 2 (DEV-ONLY): 3-Seed Confirmatory `(137, 2026, 31415)` `Epistemic-WFS-GRPO`
    #   Protocol v2: `max_steps=60, grad_accum=8 -> 480 prompt-groups/seed`,
    #   `r=16, alpha=32, lr=5e-6, beta=0.05, scale_rewards="none", loss_type="dr_grpo"`,
    #   `FALSE_ABSTAIN_REWARD = WRONG_DETERMINATE_REWARD = -1.00`.
    #   STRICT ZERO non-DEV evaluations inside Stage 2!
    # -----------------------------------------------------------------------
    grpo_rows = [
        {
            "prompt": [
                {"role": "system", "content": GRPO_SYSTEM_PROMPT_L4},
                {"role": "user", "content": ep.prompt},
            ],
            "episode": ep.to_json(),
        }
        for ep in splits["GRPO_RL"]
    ]
    grpo_ds = Dataset.from_list(grpo_rows)

    grpo_lora_cfg = LoraConfig(
        r=int(lora_r),
        lora_alpha=int(lora_alpha),
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    run_steps = int(grpo_steps)
    grad_accum_i = int(grad_accum)
    eval_ckpts = compute_dev_eval_steps(run_steps)
    seed_train_records: Dict[str, Dict[str, Any]] = {}

    for seed_val in seeds:
        seed_i = int(seed_val)
        seed_key = f"seed_{seed_i}"
        seed_digest = compute_grpo_seed_config_digest(
            seed=seed_i,
            run_steps=run_steps,
            lr=lr,
            beta=beta,
            lora_r=lora_r,
            lora_alpha=lora_alpha,
            grad_accum=grad_accum_i,
            scale_rewards=resolved_scale_rewards,
            loss_type=resolved_loss_type,
            false_abstain_reward=FALSE_ABSTAIN_REWARD,
            script_sha256=script_sha256,
            merged_sft_sha256=merged_sft_dir_digest,
            amendment_sha256=amendment_sha256,
            grpo_split_sha256=split_shas["GRPO_RL"],
            dev_split_sha256=split_shas["DEV"],
        )
        seed_dir = work_dir / f"v2_{seed_key}_{seed_digest[:12]}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        final_adapter_dir = seed_dir / "final_step_adapter"
        best_dev_dir = seed_dir / "best_dev_adapter"
        seed_ckpt_path = work_dir / f"_ckpt_v2_{seed_key}_steps{run_steps}_{seed_digest[:12]}.json"

        can_resume, cached_rec, resume_reason = can_resume_seed_checkpoint(
            seed_ckpt_path, seed_digest, final_adapter_dir, best_dev_dir
        )
        if can_resume and cached_rec is not None:
            print(
                f"[Part-B-GPU] Resuming {seed_key} training state (DEV-only, digest={seed_digest[:12]}) "
                f"from {seed_ckpt_path}...",
                flush=True,
            )
            seed_train_records[seed_key] = cached_rec
            peak_mem_gib_by_stage[f"grpo_train_{seed_key}"] = float(cached_rec["peak_cuda_allocated_gib"])
            continue
        elif seed_ckpt_path.is_file():
            print(
                f"[Part-B-GPU] Refusing stale/mismatched checkpoint {seed_ckpt_path} ({resume_reason}); retraining {seed_key}...",
                flush=True,
            )

        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(seed_i)
        np.random.seed(seed_i)

        sft_base_for_grpo = AutoModelForCausalLM.from_pretrained(
            str(merged_sft_dir), dtype=torch.bfloat16, local_files_only=True
        )
        sft_base_for_grpo.config.use_cache = False

        tracker = Wave11RolloutTelemetryTracker(num_generations=PROTOCOL_V2_NUM_GENERATIONS)
        reward_fn = tracker.make_reward_fn()
        dev_selector = BestDevSelector(sft_dev_res, non_inferiority_margin=DETERMINATE_NON_INFERIORITY_MARGIN)

        class DevTrajectoryCallback(TrainerCallback):
            def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
                if state.global_step in eval_ckpts:
                    mdl_cur = kwargs["model"]
                    was_training = mdl_cur.training
                    torch.cuda.empty_cache()
                    dev_m = eval_wfs(mdl_cur, splits["DEV"], batch_size=8)
                    torch.cuda.empty_cache()
                    mdl_cur.config.use_cache = False
                    if was_training:
                        mdl_cur.train()
                    rec_dev, should_save = dev_selector.record_step(int(state.global_step), dev_m)
                    print(f"[Part-B-GPU] {seed_key} step={state.global_step} DEV: {rec_dev}", flush=True)
                    if should_save:
                        mdl_cur.save_pretrained(str(best_dev_dir))

        grpo_extra_kw = dict(trl_cfg_check["extra_kwargs"])
        grpo_cfg = GRPOConfig(
            output_dir=str(seed_dir / "trainer_out"),
            max_steps=run_steps,
            per_device_train_batch_size=PROTOCOL_V2_BATCH_SIZE,
            gradient_accumulation_steps=grad_accum_i,
            num_generations=PROTOCOL_V2_NUM_GENERATIONS,
            max_completion_length=MAX_COMPLETION_LENGTH,
            **grpo_extra_kw,
            temperature=PROTOCOL_V2_TEMPERATURE,
            learning_rate=float(lr),
            lr_scheduler_type="cosine",
            warmup_steps=min(6, max(1, run_steps // 10)),
            beta=float(beta),
            bf16=True,
            gradient_checkpointing=True,
            use_cache=False,
            disable_dropout=True,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            seed=seed_i,
            data_seed=seed_i,
        )
        if getattr(grpo_cfg, "scale_rewards", None) != resolved_scale_rewards:
            raise AssertionError(
                f"GRPOConfig did not retain scale_rewards={resolved_scale_rewards!r}: "
                f"got {getattr(grpo_cfg, 'scale_rewards', None)!r}"
            )
        if resolved_loss_type is not None and getattr(grpo_cfg, "loss_type", None) != resolved_loss_type:
            raise AssertionError(
                f"GRPOConfig did not retain loss_type={resolved_loss_type!r}: "
                f"got {getattr(grpo_cfg, 'loss_type', None)!r}"
            )

        grpo_trainer = GRPOTrainer(
            model=sft_base_for_grpo,
            processing_class=tok,
            reward_funcs=[reward_fn],
            args=grpo_cfg,
            train_dataset=grpo_ds,
            peft_config=grpo_lora_cfg,
            callbacks=[DevTrajectoryCallback()],
        )
        if grpo_trainer.ref_model is not None:
            raise AssertionError("Expected PEFT adapter-disabled reference (`ref_model is None`)")

        t_g0 = time.perf_counter()
        grpo_trainer.train()
        grpo_wall_sec = round(time.perf_counter() - t_g0, 2)

        trained_peft_mdl = grpo_trainer.model
        trained_peft_mdl.save_pretrained(str(final_adapter_dir))
        grpo_peak_gib = record_stage_mem(f"grpo_train_{seed_key}")

        trained_active_logits = compute_probe_logits(trained_peft_mdl)
        with trained_peft_mdl.disable_adapter():
            disabled_ref_logits = compute_probe_logits(trained_peft_mdl)
        kl_ref_vs_sft_diff = float((disabled_ref_logits - sft_merged_probe_logits).abs().max().item())
        active_vs_sft_shift = float((trained_active_logits - sft_merged_probe_logits).abs().max().item())
        if not (math.isfinite(kl_ref_vs_sft_diff) and kl_ref_vs_sft_diff < 1e-4):
            raise AssertionError(f"KL reference drifted from frozen SFT: {kl_ref_vs_sft_diff}")

        # CRITICAL Protocol v2 Guard (`Neon Δ3`): NEVER overwrite missing `best_dev_dir` with `final_step`!
        dev_summary = dev_selector.summary()
        fell_back_to_step0_sft = bool(dev_summary["fell_back_to_step0_sft"])
        final_hashes_before = hash_directory_files(final_adapter_dir)
        best_dev_hashes: Optional[Dict[str, str]] = (
            None if fell_back_to_step0_sft else hash_directory_files(best_dev_dir)
        )

        log_hist = list(grpo_trainer.state.log_history)
        trl_zero_std = [float(x["frac_reward_zero_std"]) for x in log_hist if "frac_reward_zero_std" in x]
        trl_kl = [float(x["kl"]) for x in log_hist if "kl" in x]
        trl_entropy = [float(x["entropy"]) for x in log_hist if "entropy" in x]
        trl_grad_norm = [float(x["grad_norm"]) for x in log_hist if "grad_norm" in x]
        mean_zero_std = (
            float(np.mean(trl_zero_std))
            if trl_zero_std
            else float(np.mean([r["group_zero_std_frac"] for r in tracker.records]))
        )

        del grpo_trainer, trained_peft_mdl, sft_base_for_grpo
        gc.collect()
        torch.cuda.empty_cache()

        final_hashes_after = hash_directory_files(final_adapter_dir)
        if final_hashes_after != final_hashes_before or "adapter_model.safetensors" not in final_hashes_after:
            raise AssertionError(f"Checkpoint hash mismatch on {final_adapter_dir}")

        # Gate pure fp32-vs-fp32 `merge_and_unload()` parity `< 1e-4` across 8 DEV probes
        fp32_parity_report = _verify_fp32_merge_and_unload_parity(
            merged_sft_dir, final_adapter_dir, probe_inputs, min_probes=n_probe_prompts
        )
        fp32_merge_diff = float(fp32_parity_report["max_abs_logit_diff"])

        # Ungated bf16 reload telemetry (assert finite only; normal bf16 rounding never crashes!)
        reload_base = AutoModelForCausalLM.from_pretrained(
            str(merged_sft_dir), dtype=torch.bfloat16, local_files_only=True
        ).to("cuda")
        reloaded_final_peft = PeftModel.from_pretrained(
            reload_base, str(final_adapter_dir), is_trainable=False, autocast_adapter_dtype=False
        ).to("cuda")
        bf16_reload_max_abs_logit_diff = float(
            (compute_probe_logits(reloaded_final_peft) - trained_active_logits).abs().max().item()
        )
        if not math.isfinite(bf16_reload_max_abs_logit_diff):
            raise AssertionError(f"Non-finite bf16 reload logit diff on {seed_key}: {bf16_reload_max_abs_logit_diff}")

        # DEV-only verification of reloaded final_step adapter (ZERO holdout evaluation inside Stage 2!)
        primary_dev = eval_wfs(reloaded_final_peft, splits["DEV"])
        print(
            f"[Part-B-GPU] {seed_key} Stage-2 DONE (DEV-only, step={run_steps}) -> "
            f"DEV={primary_dev['exact_match']:.4f} (det={primary_dev['determinate_exact_match']:.4f}, "
            f"u_em={primary_dev['u_abstention_exact_match']:.4f}, abstain={primary_dev['abstain_rate']:.4f}) | "
            f"best_dev={dev_summary['best_dev_status']} (step={dev_summary['best_dev_step']}) | "
            f"fp32_parity={fp32_merge_diff:.2e}, bf16_reload_telemetry={bf16_reload_max_abs_logit_diff:.2e}",
            flush=True,
        )

        del reloaded_final_peft, reload_base
        gc.collect()
        torch.cuda.empty_cache()

        seed_train_records[seed_key] = {
            "seed": seed_i,
            "config_digest": seed_digest,
            "preregistered_steps": run_steps,
            "gradient_accumulation_steps": grad_accum_i,
            "prompt_groups_seen": run_steps * grad_accum_i,
            "train_wall_seconds": grpo_wall_sec,
            "peak_cuda_allocated_gib": grpo_peak_gib,
            "frac_reward_zero_std_mean": round(mean_zero_std, 4),
            "mean_kl": round(float(np.mean(trl_kl)) if trl_kl else 0.0, 6),
            "final_kl": round(float(trl_kl[-1]) if trl_kl else 0.0, 6),
            "mean_entropy": round(float(np.mean(trl_entropy)) if trl_entropy else 0.0, 4),
            "mean_grad_norm": round(float(np.mean(trl_grad_norm)) if trl_grad_norm else 0.0, 4),
            "rollout_abstain_rate_mean": round(float(np.mean([r["abstain_rate"] for r in tracker.records])), 4),
            "rollout_false_abstain_rate_mean": round(
                float(np.mean([r["false_abstain_rate"] for r in tracker.records])), 4
            ),
            "rollout_false_commit_on_u_mean": round(
                float(np.mean([r["false_commit_on_u_rate"] for r in tracker.records])), 4
            ),
            "kl_ref_disabled_adapter_vs_frozen_sft_max_abs_logit_diff": kl_ref_vs_sft_diff,
            "active_grpo_adapter_vs_frozen_sft_max_abs_logit_shift": active_vs_sft_shift,
            "bf16_reload_max_abs_logit_diff": bf16_reload_max_abs_logit_diff,
            "bf16_reload_diff_gated": False,
            "reload_max_abs_logit_diff": bf16_reload_max_abs_logit_diff,
            "fp32_merge_and_unload_max_abs_logit_diff": fp32_merge_diff,
            "fp32_merge_and_unload_parity_report": fp32_parity_report,
            "final_adapter_dir": str(final_adapter_dir.resolve()),
            "best_dev_adapter_dir": None if fell_back_to_step0_sft else str(best_dev_dir.resolve()),
            "final_adapter_sha256": final_hashes_after,
            "best_dev_adapter_sha256": best_dev_hashes,
            "dev_trajectory": dev_summary["dev_trajectory"],
            "best_dev_step": dev_summary["best_dev_step"],
            "best_dev_status": dev_summary["best_dev_status"],
            "fell_back_to_step0_sft": fell_back_to_step0_sft,
            "eligible_grpo_steps_count": dev_summary["eligible_grpo_steps_count"],
            "primary_final_step_dev": primary_dev,
        }
        write_json_atomic(
            seed_ckpt_path,
            {
                "protocol_version": PROTOCOL_V2_VERSION,
                "config_digest": seed_digest,
                "seed_train_record": seed_train_records[seed_key],
            },
        )

    # -----------------------------------------------------------------------
    # Freeze Boundary: Write & Byte-for-Byte Verify `selection_frozen.json`
    #   BEFORE touching `EVAL_VIRGIN_V2`, `EVAL_WAVE6_LOOK2`, or `D1`!
    # -----------------------------------------------------------------------
    frozen_path, frozen_manifest = build_and_write_selection_frozen(
        output_dir=output_dir,
        merged_sft_dir=merged_sft_dir,
        seed_train_records=seed_train_records,
        amendment_path=amendment_path,
        presealed_splits_sha256=split_shas,
    )
    verify_selection_frozen_manifest(frozen_path)
    frozen_sha256 = sha256_file(frozen_path)
    print(
        f"[Part-B-GPU] Selection frozen & verified byte-for-byte at {frozen_path} "
        f"(sha256={frozen_sha256[:12]}...). Starting Stage 3 One-Shot Post-Freeze Evaluation...",
        flush=True,
    )

    # -----------------------------------------------------------------------
    # Stage 3: One-Shot Post-Freeze Holdout Evaluation (Digest-Locked!)
    #   Scores Always_Abstain, Base 7B, SFT 7B, all 3 GRPO seeds (`final_step` +
    #   `best_dev`), and Bare Nemotron-30B on `EVAL_VIRGIN_V2`, `EVAL_WAVE6_LOOK2`,
    #   `d1_user_prompts_k17`, and `d1_user_prompts_k5`.
    # -----------------------------------------------------------------------
    verify_selection_frozen_manifest(frozen_path)

    abstain_virgin = evaluate_always_abstain_wfs_baseline(splits["EVAL_VIRGIN_V2"])
    abstain_wave6 = evaluate_always_abstain_wfs_baseline(splits["EVAL_WAVE6_LOOK2"])
    abstain_d1 = evaluate_always_abstain_d1_baseline(d1_cases)

    # Precompute BAAI/bge-m3 D1 prompts post-freeze
    emb_model = SentenceTransformer("BAAI/bge-m3", device="cuda")
    d1_doc_embs = np.asarray(
        emb_model.encode([d.text for d in d1_docs], batch_size=64, normalize_embeddings=True, show_progress_bar=False),
        dtype=np.float32,
    )
    d1_q_embs = np.asarray(
        emb_model.encode([c["query_text"] for c in d1_cases], batch_size=64, normalize_embeddings=True, show_progress_bar=False),
        dtype=np.float32,
    )
    del emb_model
    gc.collect()
    torch.cuda.empty_cache()
    d1_user_prompts_k17, d1_retrieval_k17_stats = build_d1_matched_evidence_prompts(
        d1_cases, d1_docs, d1_doc_embs, d1_q_embs, top_k=17
    )
    d1_user_prompts_k5, d1_retrieval_k5_stats = build_d1_matched_evidence_prompts(
        d1_cases, d1_docs, d1_doc_embs, d1_q_embs, top_k=5
    )

    # 3a. Base 7B post-freeze holdout evaluation
    torch.cuda.reset_peak_memory_stats()
    base_eval_mdl = AutoModelForCausalLM.from_pretrained(
        base_model_id, dtype=torch.bfloat16, local_files_only=True
    ).to("cuda")
    base_virgin_res = eval_wfs(base_eval_mdl, splits["EVAL_VIRGIN_V2"])
    base_wave6_res = eval_wfs(base_eval_mdl, splits["EVAL_WAVE6_LOOK2"])
    base_d1_k17_res = eval_d1(base_eval_mdl, d1_user_prompts_k17)
    base_d1_k5_res = eval_d1(base_eval_mdl, d1_user_prompts_k5)
    record_stage_mem("stage3_base_7b_holdout_eval")
    del base_eval_mdl
    gc.collect()
    torch.cuda.empty_cache()

    # 3b. SFT 7B post-freeze holdout evaluation
    torch.cuda.reset_peak_memory_stats()
    sft_eval_mdl = AutoModelForCausalLM.from_pretrained(
        str(merged_sft_dir), dtype=torch.bfloat16, local_files_only=True
    ).to("cuda")
    sft_virgin_res = eval_wfs(sft_eval_mdl, splits["EVAL_VIRGIN_V2"])
    sft_wave6_res = eval_wfs(sft_eval_mdl, splits["EVAL_WAVE6_LOOK2"])
    sft_d1_k17_res = eval_d1(sft_eval_mdl, d1_user_prompts_k17)
    sft_d1_k5_res = eval_d1(sft_eval_mdl, d1_user_prompts_k5)
    record_stage_mem("stage3_sft_7b_holdout_eval")
    del sft_eval_mdl
    gc.collect()
    torch.cuda.empty_cache()
    print(
        f"[Part-B-GPU] Stage-3 SFT 7B -> VIRGIN_V2={sft_virgin_res['exact_match']:.4f} "
        f"(det={sft_virgin_res['determinate_exact_match']:.4f}, u_fc={sft_virgin_res['false_commit_rate_on_u']:.4f}), "
        f"WAVE6_LOOK2={sft_wave6_res['exact_match']:.4f}",
        flush=True,
    )

    # 3c. All confirmatory GRPO seeds (`final_step` PRIMARY + `best_dev` SECONDARY)
    seed_keys = sorted(seed_train_records.keys(), key=lambda s: int(s.split("_")[1]))
    seed_runs: Dict[str, Dict[str, Any]] = {}
    for seed_key in seed_keys:
        tr_rec = seed_train_records[seed_key]
        fin_dir = Path(str(tr_rec["final_adapter_dir"]))

        torch.cuda.reset_peak_memory_stats()
        rb_fin = AutoModelForCausalLM.from_pretrained(
            str(merged_sft_dir), dtype=torch.bfloat16, local_files_only=True
        ).to("cuda")
        fin_peft = PeftModel.from_pretrained(
            rb_fin, str(fin_dir), is_trainable=False, autocast_adapter_dtype=False
        ).to("cuda")
        primary_virgin = eval_wfs(fin_peft, splits["EVAL_VIRGIN_V2"])
        primary_wave6 = eval_wfs(fin_peft, splits["EVAL_WAVE6_LOOK2"])
        primary_d1_k17 = eval_d1(fin_peft, d1_user_prompts_k17)
        primary_d1_k5 = eval_d1(fin_peft, d1_user_prompts_k5)
        record_stage_mem(f"stage3_grpo_final_{seed_key}")
        del fin_peft, rb_fin
        gc.collect()
        torch.cuda.empty_cache()

        print(
            f"[Part-B-GPU] Stage-3 {seed_key} PRIMARY (step={run_steps}) -> "
            f"VIRGIN_V2={primary_virgin['exact_match']:.4f} (det={primary_virgin['determinate_exact_match']:.4f}, "
            f"u_fc={primary_virgin['false_commit_rate_on_u']:.4f}), "
            f"WAVE6_LOOK2={primary_wave6['exact_match']:.4f} (u_fc={primary_wave6['false_commit_rate_on_u']:.4f}), "
            f"D1_k17={primary_d1_k17['strict_accuracy']:.4f}",
            flush=True,
        )

        if bool(tr_rec["fell_back_to_step0_sft"]):
            sec_virgin = sft_virgin_res
            sec_wave6 = sft_wave6_res
            sec_reused_from: Optional[str] = "step0_sft_merged_sft_dir"
        elif (
            int(tr_rec["best_dev_step"]) == run_steps
            and tr_rec.get("best_dev_adapter_sha256") == tr_rec.get("final_adapter_sha256")
        ):
            sec_virgin = primary_virgin
            sec_wave6 = primary_wave6
            sec_reused_from = "final_step_adapter"
        else:
            best_dir_p = Path(str(tr_rec["best_dev_adapter_dir"]))
            rb_b = AutoModelForCausalLM.from_pretrained(
                str(merged_sft_dir), dtype=torch.bfloat16, local_files_only=True
            ).to("cuda")
            best_peft = PeftModel.from_pretrained(
                rb_b, str(best_dir_p), is_trainable=False, autocast_adapter_dtype=False
            ).to("cuda")
            sec_virgin = eval_wfs(best_peft, splits["EVAL_VIRGIN_V2"])
            sec_wave6 = eval_wfs(best_peft, splits["EVAL_WAVE6_LOOK2"])
            del best_peft, rb_b
            gc.collect()
            torch.cuda.empty_cache()
            sec_reused_from = None

        seed_runs[seed_key] = {
            **tr_rec,
            "primary_final_step_eval_virgin_v2": primary_virgin,
            "primary_final_step_eval_wave6_look2": primary_wave6,
            "primary_final_step_d1_k17": primary_d1_k17,
            "primary_final_step_d1_k5": primary_d1_k5,
            "secondary_best_dev_reused_from": sec_reused_from,
            "secondary_best_dev_eval_virgin_v2": _strip_heavy_eval(sec_virgin),
            "secondary_best_dev_eval_wave6_look2": _strip_heavy_eval(sec_wave6),
        }

    # 3d. Bare Nemotron-30B on `EVAL_VIRGIN_V2`, `EVAL_WAVE6_LOOK2`, & `D1`
    print(f"[Part-B-GPU] Stage-3 Evaluating Bare Nemotron-30B at {nemotron_endpoint}...", flush=True)
    nem_virgin_res = _eval_nemotron_wfs_l4(nemotron_endpoint, nemotron_model, splits["EVAL_VIRGIN_V2"])
    nem_wave6_res = _eval_nemotron_wfs_l4(nemotron_endpoint, nemotron_model, splits["EVAL_WAVE6_LOOK2"])
    nem_d1_k17_res = _eval_nemotron_d1_l4(nemotron_endpoint, nemotron_model, d1_cases, d1_user_prompts_k17)
    nem_d1_k5_res = _eval_nemotron_d1_l4(nemotron_endpoint, nemotron_model, d1_cases, d1_user_prompts_k5)

    # -----------------------------------------------------------------------
    # Stage 4: 3-Seed + 2-Seed `(137, 2026)` Aggregation, Paired Bootstrap, Holm
    # -----------------------------------------------------------------------
    two_seed_keys = [f"seed_{s}" for s in PRIMARY_TWO_SEED_SUBSET if f"seed_{s}" in seed_runs]

    def _summarize_seed_subset_wfs(subset_keys: Sequence[str], seed_field: str) -> Dict[str, Any]:
        eids_local = sorted(sft_virgin_res["per_id_exact"].keys()) if "virgin" in seed_field else sorted(sft_wave6_res["per_id_exact"].keys())
        exact_m = {
            eid: float(np.mean([seed_runs[sk][seed_field]["per_id_exact"][eid] for sk in subset_keys]))
            for eid in eids_local
        }
        rew_m = {
            eid: float(np.mean([seed_runs[sk][seed_field]["per_id_reward"][eid] for sk in subset_keys]))
            for eid in eids_local
        }
        return {
            "seeds": [int(seed_runs[sk]["seed"]) for sk in subset_keys],
            "exact_match": round(float(np.mean(list(exact_m.values()))), 4),
            "exact_match_seed_std": round(
                float(np.std([seed_runs[sk][seed_field]["exact_match"] for sk in subset_keys])), 4
            ),
            "determinate_exact_match": round(
                float(np.mean([seed_runs[sk][seed_field]["determinate_exact_match"] for sk in subset_keys])), 4
            ),
            "u_abstention_exact_match": round(
                float(np.mean([seed_runs[sk][seed_field]["u_abstention_exact_match"] for sk in subset_keys])), 4
            ),
            "mean_total_reward": round(float(np.mean(list(rew_m.values()))), 4),
            "abstain_rate_mean": round(
                float(np.mean([seed_runs[sk][seed_field]["abstain_rate"] for sk in subset_keys])), 4
            ),
            "false_abstain_rate_on_determinate_mean": round(
                float(np.mean([seed_runs[sk][seed_field]["false_abstain_rate_on_determinate"] for sk in subset_keys])), 4
            ),
            "false_commit_rate_on_u_mean": round(
                float(np.mean([seed_runs[sk][seed_field]["false_commit_rate_on_u"] for sk in subset_keys])), 4
            ),
            "_per_id_exact": exact_m,
            "_per_id_reward": rew_m,
        }

    def _aggregate_wfs_holdout(
        split_label: str,
        seed_field: str,
        secondary_field: str,
        base_res: Mapping[str, Any],
        sft_res: Mapping[str, Any],
        nem_res: Mapping[str, Any],
        abstain_res: Mapping[str, Any],
        csv_filename: str,
        seed_offset: int,
    ) -> Tuple[Dict[str, Any], str]:
        eids = sorted(sft_res["per_id_exact"].keys())
        mean_3s = _summarize_seed_subset_wfs(seed_keys, seed_field)
        grpo_exact_3s = mean_3s.pop("_per_id_exact")
        grpo_rew_3s = mean_3s.pop("_per_id_reward")

        contrasts: Dict[str, Any] = {
            "GRPO_3SeedMean_minus_SFT_7B_exact": paired_bootstrap_wave6(
                grpo_exact_3s, sft_res["per_id_exact"], seed=seed_offset + 1, expected_ids=eids
            ),
            "GRPO_3SeedMean_minus_SFT_7B_reward": paired_bootstrap_wave6(
                grpo_rew_3s, sft_res["per_id_reward"], seed=seed_offset + 2, expected_ids=eids
            ),
            "GRPO_3SeedMean_minus_Base_7B_exact": paired_bootstrap_wave6(
                grpo_exact_3s, base_res["per_id_exact"], seed=seed_offset + 3, expected_ids=eids
            ),
            "GRPO_3SeedMean_minus_Bare_Nemotron_30B_exact": paired_bootstrap_wave6(
                grpo_exact_3s, nem_res["per_id_exact"], seed=seed_offset + 4, expected_ids=eids
            ),
        }

        two_seed_summary: Optional[Dict[str, Any]] = None
        if len(two_seed_keys) == 2:
            mean_2s = _summarize_seed_subset_wfs(two_seed_keys, seed_field)
            grpo_exact_2s = mean_2s.pop("_per_id_exact")
            grpo_rew_2s = mean_2s.pop("_per_id_reward")
            two_seed_summary = mean_2s
            contrasts["GRPO_2SeedMean_137_2026_minus_SFT_7B_exact"] = paired_bootstrap_wave6(
                grpo_exact_2s, sft_res["per_id_exact"], seed=seed_offset + 21, expected_ids=eids
            )
            contrasts["GRPO_2SeedMean_137_2026_minus_SFT_7B_reward"] = paired_bootstrap_wave6(
                grpo_rew_2s, sft_res["per_id_reward"], seed=seed_offset + 22, expected_ids=eids
            )

        for idx_s, sk in enumerate(seed_keys):
            contrasts[f"GRPO_{sk}_minus_SFT_7B_exact"] = paired_bootstrap_wave6(
                seed_runs[sk][seed_field]["per_id_exact"],
                sft_res["per_id_exact"],
                seed=seed_offset + 100 + idx_s,
                expected_ids=eids,
            )

        det_mean = float(mean_3s["determinate_exact_match"])
        sft_det = float(sft_res["determinate_exact_match"])

        csv_p = output_dir / csv_filename
        rows: List[Dict[str, Any]] = []
        sources = [
            ("Always_Abstain_Baseline", abstain_res["details"]),
            ("Base_Qwen2.5_7B", base_res["details"]),
            ("SFT_Qwen2.5_7B", sft_res["details"]),
            ("Bare_Nemotron_30B", nem_res["details"]),
        ] + [(f"SFT_plus_Epistemic_GRPO_7B_{sk}", seed_runs[sk][seed_field]["details"]) for sk in seed_keys]
        for arm_n, dets in sources:
            for d in dets:
                rows.append(
                    {
                        "holdout": split_label,
                        "episode_id": d["episode_id"],
                        "scenario": d["scenario"],
                        "gold_label": d["gold_label"],
                        "gold_value": d["gold_value"],
                        "arm": arm_n,
                        "exact_match": d["exact_match"],
                        "total_reward": d["total_reward"],
                        "core_reward": d["core_reward"],
                        "outcome": d["outcome"],
                        "is_abstain": int(d["is_abstain"]),
                        "valid_format": int(d["valid_format"]),
                    }
                )
        with csv_p.open("w", encoding="utf-8", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

        secondary_best_dev_3s = {
            "role": "descriptive_secondary_only",
            "exact_match_mean": round(
                float(np.mean([seed_runs[sk][secondary_field]["exact_match"] for sk in seed_keys])), 4
            ),
            "determinate_exact_match_mean": round(
                float(np.mean([seed_runs[sk][secondary_field]["determinate_exact_match"] for sk in seed_keys])), 4
            ),
            "u_abstention_exact_match_mean": round(
                float(np.mean([seed_runs[sk][secondary_field]["u_abstention_exact_match"] for sk in seed_keys])), 4
            ),
            "false_commit_rate_on_u_mean": round(
                float(np.mean([seed_runs[sk][secondary_field]["false_commit_rate_on_u"] for sk in seed_keys])), 4
            ),
            "n_seeds_fell_back_to_step0_sft": sum(1 for sk in seed_keys if bool(seed_runs[sk]["fell_back_to_step0_sft"])),
            "per_seed": {
                sk: {
                    **seed_runs[sk][secondary_field],
                    "best_dev_step": seed_runs[sk]["best_dev_step"],
                    "best_dev_status": seed_runs[sk]["best_dev_status"],
                    "fell_back_to_step0_sft": seed_runs[sk]["fell_back_to_step0_sft"],
                    "secondary_best_dev_reused_from": seed_runs[sk]["secondary_best_dev_reused_from"],
                }
                for sk in seed_keys
            },
        }

        block: Dict[str, Any] = {
            "Always_Abstain_Baseline": _strip_heavy_eval(abstain_res),
            "Base_Qwen2.5_7B": _strip_heavy_eval(base_res),
            "SFT_Qwen2.5_7B": _strip_heavy_eval(sft_res),
            "Bare_Nemotron_30B": _strip_heavy_eval(nem_res),
            "SFT_plus_Epistemic_GRPO_7B_per_seed_primary": {
                sk: _strip_heavy_eval(seed_runs[sk][seed_field]) for sk in seed_keys
            },
            "SFT_plus_Epistemic_GRPO_7B_3SeedMean_primary": mean_3s,
            "SFT_plus_Epistemic_GRPO_7B_2SeedMean_137_2026_primary": two_seed_summary,
            "SFT_plus_Epistemic_GRPO_7B_secondary_best_dev": secondary_best_dev_3s,
            "determinate_non_inferiority_gate": {
                "sft_determinate_exact_match": sft_det,
                "grpo_3seed_determinate_exact_match": det_mean,
                "margin": DETERMINATE_NON_INFERIORITY_MARGIN,
                "passed": bool(det_mean >= sft_det - DETERMINATE_NON_INFERIORITY_MARGIN),
            },
            "paired_bootstrap_wave6": contrasts,
        }
        return block, sha256_file(csv_p)

    virgin_block, virgin_csv_sha = _aggregate_wfs_holdout(
        "EVAL_VIRGIN_V2",
        "primary_final_step_eval_virgin_v2",
        "secondary_best_dev_eval_virgin_v2",
        base_virgin_res,
        sft_virgin_res,
        nem_virgin_res,
        abstain_virgin,
        "wave11_d9_part_b_eval_virgin_v2_n120_rows.csv",
        seed_offset=40,
    )
    wave6_block, wave6_csv_sha = _aggregate_wfs_holdout(
        "EVAL_WAVE6_LOOK2",
        "primary_final_step_eval_wave6_look2",
        "secondary_best_dev_eval_wave6_look2",
        base_wave6_res,
        sft_wave6_res,
        nem_wave6_res,
        abstain_wave6,
        "wave11_d9_part_b_eval_wave6_look2_n120_rows.csv",
        seed_offset=80,
    )

    # D1 aggregation (3-seed + 2-seed `(137, 2026)`)
    d1_ids = sorted(sft_d1_k17_res["per_id_strict"].keys())
    grpo_d1_k17_3s = {
        cid: float(np.mean([seed_runs[sk]["primary_final_step_d1_k17"]["per_id_strict"][cid] for sk in seed_keys]))
        for cid in d1_ids
    }
    grpo_d1_k5_3s = {
        cid: float(np.mean([seed_runs[sk]["primary_final_step_d1_k5"]["per_id_strict"][cid] for sk in seed_keys]))
        for cid in d1_ids
    }
    grpo_d1_k17_2s_mean: Optional[float] = None
    grpo_d1_k5_2s_mean: Optional[float] = None
    if len(two_seed_keys) == 2:
        grpo_d1_k17_2s = {
            cid: float(np.mean([seed_runs[sk]["primary_final_step_d1_k17"]["per_id_strict"][cid] for sk in two_seed_keys]))
            for cid in d1_ids
        }
        grpo_d1_k5_2s = {
            cid: float(np.mean([seed_runs[sk]["primary_final_step_d1_k5"]["per_id_strict"][cid] for sk in two_seed_keys]))
            for cid in d1_ids
        }
        grpo_d1_k17_2s_mean = round(float(np.mean(list(grpo_d1_k17_2s.values()))), 4)
        grpo_d1_k5_2s_mean = round(float(np.mean(list(grpo_d1_k5_2s.values()))), 4)

    d1_contrasts = {
        "D1_k17_GRPO_3SeedMean_minus_SFT_7B_strict": paired_bootstrap_wave6(
            grpo_d1_k17_3s, sft_d1_k17_res["per_id_strict"], seed=121, expected_ids=d1_ids
        ),
        "D1_k17_GRPO_3SeedMean_minus_Base_7B_strict": paired_bootstrap_wave6(
            grpo_d1_k17_3s, base_d1_k17_res["per_id_strict"], seed=122, expected_ids=d1_ids
        ),
        "D1_k17_GRPO_3SeedMean_minus_Bare_Nemotron_30B_strict": paired_bootstrap_wave6(
            grpo_d1_k17_3s, nem_d1_k17_res["per_id_strict"], seed=123, expected_ids=d1_ids
        ),
        "D1_k5_GRPO_3SeedMean_minus_Bare_Nemotron_30B_strict": paired_bootstrap_wave6(
            grpo_d1_k5_3s, nem_d1_k5_res["per_id_strict"], seed=124, expected_ids=d1_ids
        ),
    }

    holm_primary_virgin = holm_bonferroni_stepdown(
        {
            "VIRGIN_V2_GRPO_vs_SFT_7B": virgin_block["paired_bootstrap_wave6"]["GRPO_3SeedMean_minus_SFT_7B_exact"][
                "p_value_two_sided"
            ],
            "VIRGIN_V2_GRPO_vs_Base_7B": virgin_block["paired_bootstrap_wave6"]["GRPO_3SeedMean_minus_Base_7B_exact"][
                "p_value_two_sided"
            ],
            "VIRGIN_V2_GRPO_vs_Bare_Nemotron_30B": virgin_block["paired_bootstrap_wave6"][
                "GRPO_3SeedMean_minus_Bare_Nemotron_30B_exact"
            ]["p_value_two_sided"],
        }
    )
    holm_secondary_wave6_d1 = holm_bonferroni_stepdown(
        {
            "WAVE6_LOOK2_GRPO_vs_SFT_7B": wave6_block["paired_bootstrap_wave6"]["GRPO_3SeedMean_minus_SFT_7B_exact"][
                "p_value_two_sided"
            ],
            "WAVE6_LOOK2_GRPO_vs_Base_7B": wave6_block["paired_bootstrap_wave6"]["GRPO_3SeedMean_minus_Base_7B_exact"][
                "p_value_two_sided"
            ],
            "WAVE6_LOOK2_GRPO_vs_Bare_Nemotron_30B": wave6_block["paired_bootstrap_wave6"][
                "GRPO_3SeedMean_minus_Bare_Nemotron_30B_exact"
            ]["p_value_two_sided"],
            "D1_Prose_GRPO_vs_SFT_7B": d1_contrasts["D1_k17_GRPO_3SeedMean_minus_SFT_7B_strict"]["p_value_two_sided"],
            "D1_Prose_GRPO_vs_Base_7B": d1_contrasts["D1_k17_GRPO_3SeedMean_minus_Base_7B_strict"]["p_value_two_sided"],
            "D1_Prose_GRPO_vs_Bare_Nemotron_30B": d1_contrasts[
                "D1_k17_GRPO_3SeedMean_minus_Bare_Nemotron_30B_strict"
            ]["p_value_two_sided"],
        }
    )

    primary_diff = float(virgin_block["paired_bootstrap_wave6"]["GRPO_3SeedMean_minus_SFT_7B_exact"]["diff"])
    primary_p_raw = float(virgin_block["paired_bootstrap_wave6"]["GRPO_3SeedMean_minus_SFT_7B_exact"]["p_value_two_sided"])
    primary_holm_entry = holm_primary_virgin["VIRGIN_V2_GRPO_vs_SFT_7B"]
    primary_p_holm = float(
        primary_holm_entry["p_adjusted"]
        if isinstance(primary_holm_entry, Mapping) and "p_adjusted" in primary_holm_entry
        else primary_holm_entry.get("adjusted_p", primary_p_raw)
        if isinstance(primary_holm_entry, Mapping)
        else primary_p_raw
    )
    grpo_sig_over_sft = bool(primary_diff > 0.0 and primary_p_holm < 0.05)
    hypothesis_verdict = (
        "GRPO_IMPROVES_OVER_SFT_UNDER_PROTOCOL_V2"
        if grpo_sig_over_sft
        else "K4 stands under Protocol v2"
    )

    seed_summaries = {
        sk: {
            k: v
            for k, v in sr.items()
            if k
            not in (
                "primary_final_step_dev",
                "primary_final_step_eval_virgin_v2",
                "primary_final_step_eval_wave6_look2",
                "primary_final_step_d1_k17",
                "primary_final_step_d1_k5",
            )
        }
        for sk, sr in seed_runs.items()
    }
    for sk in seed_keys:
        seed_summaries[sk]["primary_final_step_dev"] = _strip_heavy_eval(seed_runs[sk]["primary_final_step_dev"])
        seed_summaries[sk]["primary_final_step_eval_virgin_v2"] = _strip_heavy_eval(
            seed_runs[sk]["primary_final_step_eval_virgin_v2"]
        )
        seed_summaries[sk]["primary_final_step_eval_wave6_look2"] = _strip_heavy_eval(
            seed_runs[sk]["primary_final_step_eval_wave6_look2"]
        )
        seed_summaries[sk]["primary_final_step_d1_k17"] = _strip_heavy_d1(seed_runs[sk]["primary_final_step_d1_k17"])
        seed_summaries[sk]["primary_final_step_d1_k5"] = _strip_heavy_d1(seed_runs[sk]["primary_final_step_d1_k5"])

    part_b_payload = {
        "schema_version": "wave11_d9_part_b_virgin_u_grpo_v2",
        "protocol_version": PROTOCOL_V2_VERSION,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": current_git_sha(REPO_ROOT),
        "compute_node": "DGX Spark #2 (mattmre@10.0.0.194)",
        "base_model": base_model_id,
        "environment_versions": env_versions,
        "protocol_v2_hypothesis_test": {
            "primary_endpoint": (
                f"final_step@{run_steps} 3-seed {tuple(int(s) for s in seeds)} mean exact_match on "
                "EVAL_VIRGIN_V2 vs SFT_Qwen2.5_7B (paired_bootstrap_wave6 + holm_bonferroni_stepdown at alpha=0.05)"
            ),
            "grpo_minus_sft_diff": round(primary_diff, 6),
            "grpo_minus_sft_p_two_sided": round(primary_p_raw, 6),
            "grpo_minus_sft_holm_p_adj": round(primary_p_holm, 6),
            "grpo_minus_sft_significant_alpha_005": grpo_sig_over_sft,
            "verdict": hypothesis_verdict,
            "null_result_fails_receipt": False,
            "protocol_v2_single_knob_attribution_claimed": False,
            "eval_virgin_v2_scope": "virgin_for_grpo_arm_only",
            "retired_dev_diagnostic_seed": RETIRED_DEV_DIAGNOSTIC_SEED,
            "confirmatory_seeds": [int(s) for s in seeds],
            "two_seed_sensitivity_seeds": list(PRIMARY_TWO_SEED_SUBSET),
            "subsequent_retune_policy": "Any further hyperparameter or selection change requires minting EVAL_VIRGIN_V3",
            "selection_frozen_file": frozen_path.name,
            "selection_frozen_sha256": frozen_sha256,
            "amendment_file": str(amendment_path),
            "amendment_sha256": amendment_sha256,
        },
        "l4_part_b_audit": {
            "five_way_disjoint_splits": disjoint_proof,
            "prompt_max_tokens_by_split": split_prompt_token_max,
            "max_prompt_length_cap": MAX_PROMPT_LENGTH,
            "max_completion_length_matched": MAX_COMPLETION_LENGTH,
            "out_of_loop_holdout_lock": {
                "selection_frozen_file": frozen_path.name,
                "selection_frozen_sha256": frozen_sha256,
                "in_loop_holdout_evaluations_performed": 0,
                "post_freeze_digest_verified": True,
                "merged_sft_reused_from_disk": merged_sft_reused_from_disk,
            },
            "sft_completion_only_and_untruncated_4step_cot": sft_mask_audit,
            "sft_vs_base_logit_shift": sft_vs_base_logit_shift,
            "sft_reload_max_abs_logit_diff": sft_reload_logit_diff,
            "sft_on_grpo_rl": _strip_heavy_eval(sft_grpo_split_res),
            "grpo_hyperparameters": {
                "protocol_version": PROTOCOL_V2_VERSION,
                "max_steps": run_steps,
                "gradient_accumulation_steps": grad_accum_i,
                "prompt_groups_per_seed": run_steps * grad_accum_i,
                "per_device_train_batch_size": PROTOCOL_V2_BATCH_SIZE,
                "num_generations": PROTOCOL_V2_NUM_GENERATIONS,
                "lora_r": int(lora_r),
                "lora_alpha": int(lora_alpha),
                "learning_rate": float(lr),
                "temperature": PROTOCOL_V2_TEMPERATURE,
                "beta": float(beta),
                "scale_rewards": resolved_scale_rewards,
                "loss_type": resolved_loss_type,
                "dev_eval_steps": list(eval_ckpts),
                "retired_dev_diagnostic_seed": RETIRED_DEV_DIAGNOSTIC_SEED,
                "preregistered_seeds": [int(s) for s in seeds],
                "two_seed_sensitivity_subset": list(PRIMARY_TWO_SEED_SUBSET),
            },
            "trl_unscaled_advantage_startup_check": trl_cfg_check["unscaled_advantage_self_check"],
            "anti_collapse_reward_spec": {
                "exact_core_reward": EXACT_CORE_REWARD,
                "format_bonus_reward": FORMAT_BONUS_REWARD,
                "false_abstain_reward": FALSE_ABSTAIN_REWARD,
                "false_commit_on_u_reward": FALSE_COMMIT_ON_U_REWARD,
                "wrong_determinate_reward": WRONG_DETERMINATE_REWARD,
                "expected_always_abstain_reward_balanced": round(compute_expected_always_abstain_reward(), 6),
            },
            "max_bf16_reload_logit_diff_across_seeds_ungated_telemetry": max(
                seed_runs[sk]["bf16_reload_max_abs_logit_diff"] for sk in seed_keys
            ),
            "max_reload_logit_diff_across_seeds": max(
                seed_runs[sk]["bf16_reload_max_abs_logit_diff"] for sk in seed_keys
            ),
            "max_fp32_merge_and_unload_diff_across_seeds": max(
                seed_runs[sk]["fp32_merge_and_unload_max_abs_logit_diff"] for sk in seed_keys
            ),
            "peak_cuda_memory_gib_by_stage": peak_mem_gib_by_stage,
        },
        "primary_holdout_eval_virgin_v2_n120": virgin_block,
        "secondary_look2_eval_wave6_n120": wave6_block,
        "d1_zero_shot_enterprise_prose_n60": {
            "corpus_sha256": d1_corpus_sha,
            "retrieval_k17": d1_retrieval_k17_stats,
            "retrieval_k5": d1_retrieval_k5_stats,
            "Always_Abstain_Baseline": _strip_heavy_d1(abstain_d1),
            "Base_Qwen2.5_7B": _strip_heavy_d1(base_d1_k17_res),
            "SFT_Qwen2.5_7B": _strip_heavy_d1(sft_d1_k17_res),
            "Bare_Nemotron_30B": _strip_heavy_d1(nem_d1_k17_res),
            "SFT_plus_Epistemic_GRPO_7B_3SeedMean_k17_strict": round(float(np.mean(list(grpo_d1_k17_3s.values()))), 4),
            "SFT_plus_Epistemic_GRPO_7B_2SeedMean_137_2026_k17_strict": grpo_d1_k17_2s_mean,
            "secondary_retrieval_starved_k5": {
                "Base_Qwen2.5_7B": _strip_heavy_d1(base_d1_k5_res),
                "SFT_Qwen2.5_7B": _strip_heavy_d1(sft_d1_k5_res),
                "Bare_Nemotron_30B": _strip_heavy_d1(nem_d1_k5_res),
                "SFT_plus_Epistemic_GRPO_7B_3SeedMean_k5_strict": round(float(np.mean(list(grpo_d1_k5_3s.values()))), 4),
                "SFT_plus_Epistemic_GRPO_7B_2SeedMean_137_2026_k5_strict": grpo_d1_k5_2s_mean,
            },
            "paired_bootstrap_wave6": d1_contrasts,
        },
        "holm_bonferroni_primary_virgin_v2": holm_primary_virgin,
        "holm_bonferroni_secondary_wave6_and_d1": holm_secondary_wave6_d1,
        "grpo_seed_runs": seed_summaries,
        "csv_artifacts": {
            "wave11_d9_part_b_eval_virgin_v2_n120_rows.csv": virgin_csv_sha,
            "wave11_d9_part_b_eval_wave6_look2_n120_rows.csv": wave6_csv_sha,
        },
        "pipeline_wall_seconds": round(time.perf_counter() - t_pipeline0, 2),
    }

    part_b_path = output_dir / "part_b_u_grpo_results.json"
    write_json_atomic(part_b_path, part_b_payload)
    print(
        f"[Part-B-GPU] Full Protocol v2 Part B completed in {part_b_payload['pipeline_wall_seconds']}s | "
        f"verdict={hypothesis_verdict} (diff={primary_diff:+.4f}, p_holm={primary_p_holm:.4f})",
        flush=True,
    )
    return part_b_payload


# ---------------------------------------------------------------------------
# Mode 4: `--mode assemble-manifest` (Combine Part A + Part B + Receipt)
# ---------------------------------------------------------------------------
def _extract_part_a_gate_diagnostics(part_a: Mapping[str, Any]) -> Dict[str, Any]:
    """Extract transparent Part A gate diagnostics, reporting latency >1,500 ms honestly as
    `latency_gate_not_met` without failing `report_receipt.json` `COMPLETED` status when
    accuracy, JSON-validity, and leakage/parity gates pass.
    """
    suite_a = part_a.get("suite_a_evaluation_n60") or {}
    l4_gates = suite_a.get("l4_gates") or {}
    bench = suite_a.get("extraction_benchmark") or {}
    ms_mean = bench.get("ms_per_case_mean")
    speedup = suite_a.get("speedup_vs_wave6_30b_extractor")

    if "ms_per_case_below_1500" in l4_gates:
        ms_below_1500: Optional[bool] = bool(l4_gates["ms_per_case_below_1500"])
    elif ms_mean is not None:
        ms_below_1500 = bool(float(ms_mean) < 1500.0)
    else:
        ms_below_1500 = None

    if ms_below_1500 is True:
        latency_status = "latency_gate_met"
    elif ms_below_1500 is False:
        latency_status = "latency_gate_not_met"
    else:
        latency_status = "not_reported"

    disclosure: Optional[str] = None
    if ms_below_1500 is False and ms_mean is not None:
        sp_txt = f"; {speedup}x faster than 30B" if speedup is not None else ""
        disclosure = f"latency gate not met ({float(ms_mean):.1f} ms vs 1500.0 ms gate{sp_txt})"
    elif ms_below_1500 is True and ms_mean is not None:
        disclosure = f"latency gate met ({float(ms_mean):.1f} ms < 1500.0 ms gate)"

    non_blocking_variances: List[str] = []
    if ms_below_1500 is False:
        non_blocking_variances.append("part_a_ms_per_case_below_1500: latency_gate_not_met")

    blocking_failed: List[str] = []
    for gk in ("json_valid_rate_ge_099", "beats_mem0_lcb975_gt_0", "reload_parity_below_1e4"):
        if gk in l4_gates and not bool(l4_gates[gk]):
            blocking_failed.append(gk)
    disjoint_proof = (part_a.get("synthetic_corpus_audit") or {}).get("suite_a_disjoint_proof") or {}
    if "disjoint_passed" in disjoint_proof and not bool(disjoint_proof["disjoint_passed"]):
        blocking_failed.append("suite_a_disjoint_passed")

    return {
        "part_a_ms_per_case_mean": float(ms_mean) if ms_mean is not None else None,
        "part_a_latency_target_ms": 1500.0,
        "part_a_ms_per_case_below_1500": ms_below_1500,
        "part_a_latency_status": latency_status,
        "part_a_latency_disclosure": disclosure,
        "part_a_blocking_gates_passed": len(blocking_failed) == 0,
        "part_a_blocking_failed_gates": blocking_failed,
        "non_blocking_gate_variances": non_blocking_variances,
    }


def assemble_wave11_d9_manifest(
    *,
    output_dir: Path,
    part_a_json: Optional[Path] = None,
    part_b_json: Optional[Path] = None,
) -> Dict[str, Any]:
    """Combine Part A & Part B outputs into `wave11_d9_extractor_grpo_manifest.json` + `report_receipt.json`."""
    output_dir.mkdir(parents=True, exist_ok=True)
    p_a = part_a_json or (output_dir / "part_a_fast_extractor_results.json")
    p_b = part_b_json or (output_dir / "part_b_u_grpo_results.json")

    if not p_a.is_file():
        raise FileNotFoundError(f"Missing Part A results JSON: {p_a}")
    if not p_b.is_file():
        raise FileNotFoundError(f"Missing Part B results JSON: {p_b}")

    part_a = json.loads(p_a.read_text(encoding="utf-8"))
    part_b = json.loads(p_b.read_text(encoding="utf-8"))

    part_a_diag = _extract_part_a_gate_diagnostics(part_a)
    part_b_hyp = part_b.get("protocol_v2_hypothesis_test") or {}

    git_sha = current_git_sha(REPO_ROOT)
    now_iso = datetime.now(timezone.utc).isoformat()
    manifest: Dict[str, Any] = {
        "schema_version": "wave11_d9_fast_extractor_and_u_grpo_l4_v1",
        "track": "Wave 11 Demo D9 (Guardrail L4): Fast 0.5B/1.5B LoRA Extractor + Virgin-Holdout U-GRPO (Protocol v2)",
        "git_sha": git_sha,
        "timestamp_utc": now_iso,
        "part_a_gate_diagnostics": part_a_diag,
        "part_b_hypothesis_summary": {
            "verdict": part_b_hyp.get("verdict"),
            "eval_virgin_v2_scope": part_b_hyp.get("eval_virgin_v2_scope"),
            "protocol_v2_single_knob_attribution_claimed": part_b_hyp.get(
                "protocol_v2_single_knob_attribution_claimed"
            ),
            "null_result_fails_receipt": False,
        },
        "part_a_fast_edge_extractor": part_a,
        "part_b_virgin_holdout_u_grpo": part_b,
        "source_part_hashes": {
            p_a.name: sha256_file(p_a),
            p_b.name: sha256_file(p_b),
        },
    }

    manifest_path = output_dir / "wave11_d9_extractor_grpo_manifest.json"
    manifest_sha = write_json_atomic(manifest_path, manifest)

    artifact_hashes: Dict[str, str] = {
        manifest_path.name: manifest_sha,
    }
    for src_p in (p_a, p_b):
        if src_p.parent.resolve() == output_dir.resolve():
            artifact_hashes[src_p.name] = sha256_file(src_p)

    # Hash any emitted CSVs and selection_frozen.json inside `output_dir`
    csv_a = (part_a.get("csv_artifact") or {}).get("filename")
    if csv_a and (output_dir / csv_a).is_file():
        artifact_hashes[csv_a] = sha256_file(output_dir / csv_a)
    for csv_b in (part_b.get("csv_artifacts") or {}).keys():
        if (output_dir / csv_b).is_file():
            artifact_hashes[csv_b] = sha256_file(output_dir / csv_b)
    frozen_in_out = output_dir / "selection_frozen.json"
    if frozen_in_out.is_file():
        artifact_hashes[frozen_in_out.name] = sha256_file(frozen_in_out)

    # Status remains COMPLETED when accuracy/JSON-validity/leakage gates pass;
    # Part A latency >1,500 ms is disclosed transparently as `latency_gate_not_met` (non-blocking),
    # and Part B null GRPO-vs-SFT (`K4 stands under Protocol v2`) never fails receipt completion.
    receipt_status = "COMPLETED" if part_a_diag["part_a_blocking_gates_passed"] else "FAILED_BLOCKING_GATES"
    receipt = {
        "schema_version": "report_receipt_v4_0",
        "bundle_id": "wave11_d9_fast_extractor_and_u_grpo",
        "status": receipt_status,
        "git_sha": git_sha,
        "generated_at_utc": now_iso,
        "timestamp_utc": now_iso,
        "manifest_file": manifest_path.name,
        "manifest_sha256": manifest_sha,
        "diagnostics": {
            "part_a_ms_per_case_below_1500": part_a_diag["part_a_ms_per_case_below_1500"],
            "part_a_latency_status": part_a_diag["part_a_latency_status"],
            "part_a_latency_disclosure": part_a_diag["part_a_latency_disclosure"],
            "non_blocking_gate_variances": part_a_diag["non_blocking_gate_variances"],
            "part_b_hypothesis_verdict": part_b_hyp.get("verdict"),
        },
        "artifact_hashes": artifact_hashes,
    }
    receipt_sha = write_json_atomic(output_dir / "report_receipt.json", receipt)
    manifest["_receipt_sha256"] = receipt_sha
    print(
        f"[Wave11-D9] Assembled {manifest_path} (sha256={manifest_sha[:12]}...) + report_receipt.json "
        f"[status={receipt_status}, part_a_latency={part_a_diag['part_a_latency_status']}]",
        flush=True,
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Wave 11 Demo D9 (Guardrail L4): Fast 0.5B/1.5B LoRA Extractor + Virgin-Holdout U-GRPO (Protocol v2)"
    )
    parser.add_argument(
        "--mode",
        choices=("verify-splits", "run-part-a", "run-part-b", "assemble-manifest"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--osv-cache", type=Path, default=DEFAULT_OSV_CACHE)
    # Part A options
    parser.add_argument(
        "--extractor-models",
        nargs="+",
        default=["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-1.5B-Instruct"],
    )
    parser.add_argument("--vllm-endpoint", type=str, default="http://10.0.0.194:8890/v1")
    parser.add_argument("--vllm-model", type=str, default="nemotron-lightning")
    parser.add_argument("--judge-hf-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--wave6-pm3-manifest", type=Path, default=DEFAULT_WAVE6_PM3_MANIFEST)
    parser.add_argument("--wave6-pm3-rows-csv", type=Path, default=DEFAULT_WAVE6_PM3_ROWS_CSV)
    parser.add_argument("--extractor-epochs", type=int, default=3)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    # Part B options (`Protocol v2` defaults)
    parser.add_argument("--grpo-base-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--nemotron-endpoint", type=str, default="http://127.0.0.1:8890/v1")
    parser.add_argument("--grpo-steps", type=int, default=PROTOCOL_V2_GRPO_STEPS)
    parser.add_argument("--amendment-path", type=Path, default=DEFAULT_D9B_AMENDMENT_PATH)
    # Assemble options
    parser.add_argument("--part-a-json", type=Path, default=None)
    parser.add_argument("--part-b-json", type=Path, default=None)
    args = parser.parse_args()

    if args.mode == "verify-splits":
        run_verify_splits(args.output_dir, args.osv_cache)
    elif args.mode == "run-part-a":
        run_part_a_fast_extractor(
            output_dir=args.output_dir,
            osv_cache_path=args.osv_cache,
            extractor_models=args.extractor_models,
            vllm_endpoint=args.vllm_endpoint,
            vllm_model=args.vllm_model,
            judge_hf_model=args.judge_hf_model,
            wave6_pm3_manifest=args.wave6_pm3_manifest,
            wave6_pm3_rows_csv=args.wave6_pm3_rows_csv,
            epochs=args.extractor_epochs,
            max_new_tokens=args.max_new_tokens,
        )
    elif args.mode == "run-part-b":
        run_part_b_u_grpo_on_spark2(
            output_dir=args.output_dir,
            osv_cache_path=args.osv_cache,
            base_model_id=args.grpo_base_model,
            nemotron_endpoint=args.nemotron_endpoint,
            nemotron_model=args.vllm_model,
            grpo_steps=args.grpo_steps,
            amendment_path=args.amendment_path,
        )
    else:
        assemble_wave11_d9_manifest(
            output_dir=args.output_dir,
            part_a_json=args.part_a_json,
            part_b_json=args.part_b_json,
        )


if __name__ == "__main__":
    main()
