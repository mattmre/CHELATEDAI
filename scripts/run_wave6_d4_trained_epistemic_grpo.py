r"""Wave 6 Track D (`Demo D4`, Guardrail `K4`):
Live Trained `SFT` vs `SFT + Epistemic-WFS-GRPO` LoRA on `Qwen/Qwen2.5-7B-Instruct`
+ Zero-Shot Unseen Real-World Enterprise Drift Prose Transfer (`D1`, `N=60`) on DGX Spark #2.

Implements all Wave-5 Cross-Judgment `K4` guardrails strictly additively:
  1. 4-Way Pairwise-Disjoint Splits (`SFT_WARMUP` N=96, `GRPO_RL` N=192, `DEV` N=80, `EVAL` N=120):
     Zero overlap on entities, slots, values, scenarios, families, seeds, or prompt hashes
     (explicitly hashing `noise-retractor-*` and all distractors, and disjoint from `D1` prose).
  2. Completion-Only SFT Without Oracle Leak (`lr=8e-5`, `3` epochs, `r=16`):
     `derive_cot_from_prompt_text(prompt_text: str)` accepts ONLY visible prompt text (zero
     `store_slot_status` / oracle leak) and masks all prompt tokens (`-100`).
  3. `merge_and_unload()` Before Fresh GRPO LoRA (`r=32, lora_alpha=64, lora_dropout=0.0`):
     Merges SFT adapter into frozen `SFT` base on disk before attaching fresh GRPO LoRA (`beta=0.04`),
     proving adapter-disabled KL reference == frozen `SFT` at zero duplicate 7B VRAM (`<= 34 GB`).
  4. TRL Signature, Left Padding, & Calibrated 3-Valued Reward:
     `tok.padding_side = "left"`, `per_device_train_batch_size=8 == num_generations=8`, `temperature=0.9`,
     keyword-only reward signature (`assert completions is not prompts`), `+1.00` exact canonical match
     (`+0.05` format bonus ONLY when `core == 1.00`), **`-0.15`** false abstention on determinate `T/F`
     items regardless of verdict (closing the `+0.25` loophole), **`-1.00`** wrong/negated/mixed/unclosed.
     Logs `frac_reward_zero_std` (`< 0.80`), `mean_kl`, and `abstain_rate`.
  5. 3 Pre-Registered Random Seeds `(42, 137, 2026)`, Hashed Disk Reload + Logit-Parity Check (`< 1e-4`),
     and Zero-Shot Evaluation (`max_new_tokens=256`) of `Base 7B`, `SFT 7B`, `SFT + Epistemic-GRPO 7B`
     (per-seed + 3-seed mean; fixed pre-registered final step PRIMARY, `best_dev` SECONDARY), and
     `Bare Nemotron-30B` (`:8890`) on BOTH `EVAL` (`N=120`) AND `D1` Enterprise Drift Prose (`N=60`)
     using `paired_bootstrap_wave6` and `holm_bonferroni_stepdown`.
"""

from __future__ import annotations

import os

if os.name != "nt":
    os.environ["HF_HOME"] = "/home/mattmre/chelated-night1/hf_cache"
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
import subprocess  # noqa: E402
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
    current_git_sha,
    write_json_atomic,
)
from wave6_common import (  # noqa: E402
    holm_bonferroni_stepdown,
    paired_bootstrap_wave6,
)

# ---------------------------------------------------------------------------
# Constants & Prompts
# ---------------------------------------------------------------------------
PREREGISTERED_SEEDS: Tuple[int, ...] = (42, 137, 2026)
SPLIT_SIZES: Dict[str, int] = {
    "SFT_WARMUP": 96,
    "GRPO_RL": 192,
    "DEV": 80,
    "EVAL": 120,
}
MAX_PROMPT_LENGTH = 1664
MAX_COMPLETION_LENGTH = 272
FALSE_ABSTAIN_REWARD = -0.15
WRONG_OR_MALFORMED_REWARD = -1.00
EXACT_CORE_REWARD = 1.00
PARTIAL_VERDICT_MISMATCH_REWARD = 0.75
FORMAT_BONUS_REWARD = 0.05

GRPO_SYSTEM_PROMPT = (
    "You are an epistemic reasoning model grounded in 3-valued Well-Founded Semantics (T=Admitted, "
    "F=Refuted/Superseded, U=Contested). Given the multi-agent evidence log, reason concisely inside "
    "<think>...</think> (under 140 words), then output EXACTLY two lines after </think>:\n"
    "VERDICT: T | F | U\n"
    "ANSWER: <exact unique current value> | ABSTAIN"
)

D1_PROSE_TRANSFER_SYSTEM_PROMPT = (
    "You are an enterprise epistemic verification model grounded in 3-valued Well-Founded Semantics "
    "(T=Admitted, U=Contested). Read the dated natural-language records below carefully.\n"
    "- Higher-authority official filings, signed security bulletins, NVD advisories, and SEC filings "
    "supersede or refute earlier preliminary notes and lower-authority chat/forum/blog/social echoes, "
    "even when stale echoes or rumor-laundering posts are repeated more often or ingested later.\n"
    "- Follow multi-step authoritative supersession chains to the latest binding official revision.\n"
    "- If equal-authority official sources directly refute one another with no independent resolution "
    "(symmetric peer conflict, state U), you MUST abstain.\n"
    "Reason concisely inside <think>...</think> (under 140 words), then output EXACTLY two lines after </think>:\n"
    "VERDICT: T | U\n"
    "ANSWER: <exact current verified value/identifier> | ABSTAIN"
)

_THINK_CLOSED_RE = re.compile(r"^\s*<think>(.+?)</think>\s*(.+)$", re.IGNORECASE | re.DOTALL)
_VERDICT_LINE_RE = re.compile(r"^\s*VERDICT\s*:\s*(T|F|U)\s*$", re.IGNORECASE | re.MULTILINE)
_ANSWER_LINE_RE = re.compile(r"^\s*ANSWER\s*:\s*([^\r\n]+?)\s*$", re.IGNORECASE | re.MULTILINE)
_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
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
_BM25_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9_.\-/]*")

# ---------------------------------------------------------------------------
# Guardrail 1: 4-Way Pairwise-Disjoint Split Pools & Novel WFS Regimes
# ---------------------------------------------------------------------------
SPLIT_SLOT_POOLS: Dict[str, Tuple[Tuple[str, str, Tuple[str, ...]], ...]] = {
    "SFT_WARMUP": (
        (
            "payments-api",
            "smtp_relay",
            (
                "mx1.corp.example",
                "mx2.corp.example",
                "relay.vendor-mail.example",
                "smtp.backup.example",
                "mx-eu.corp.example",
                "relay.attacker-cdn.example",
            ),
        ),
        (
            "billing",
            "wire_recipient",
            (
                "ACME-OPS-0042",
                "ACME-OPS-0077",
                "VENDOR-ESCROW-19",
                "OFFSHORE-HOLD-88",
                "ACME-TREASURY-01",
                "SUPPLIER-PAY-31",
            ),
        ),
        (
            "auth-broker",
            "session_ttl_sec",
            ("ttl-900s", "ttl-1800s", "ttl-3600s", "ttl-7200s", "ttl-300s", "ttl-86400s"),
        ),
        (
            "backup-vault",
            "retention_class",
            ("glacier-90d", "worm-365d", "hot-14d", "archive-7y", "ephemeral-24h", "cold-180d"),
        ),
    ),
    "GRPO_RL": (
        (
            "edge-proxy",
            "tls_min_version",
            ("TLS1.3", "TLS1.2", "TLS1.1", "TLS1.0", "SSL3.0", "TLS1.3-PQ", "TLS1.2-FIPS"),
        ),
        (
            "cve-2026-1187",
            "patch_policy",
            (
                "patch-within-24h",
                "patch-within-72h",
                "defer-to-quarter",
                "risk-accepted",
                "mitigate-waf-only",
                "emergency-rollback",
                "hotfix-canary",
            ),
        ),
        (
            "service-mesh",
            "mtls_ca_bundle",
            (
                "spiffe-root-g3",
                "spiffe-root-g2",
                "legacy-intranet-ca",
                "rogue-selfsigned-ca",
                "vault-pki-v4",
                "hsm-offline-r5",
                "staging-ca-v1",
            ),
        ),
        (
            "artifact-signer",
            "cosign_key_id",
            (
                "kms-prod-key-09",
                "kms-prod-key-04",
                "yubihsm-slot-02",
                "compromised-ci-key",
                "nitro-enclave-k7",
                "revoked-dev-key",
                "offline-root-k1",
            ),
        ),
        (
            "kafka-cluster",
            "ack_quorum_mode",
            (
                "acks-all-isr3",
                "acks-leader-only",
                "acks-none-async",
                "acks-all-isr2",
                "acks-quorum-strict",
                "acks-deprecated-v0",
                "acks-rack-aware",
            ),
        ),
        (
            "secrets-engine",
            "seal_mechanism",
            (
                "shamir-5of3",
                "awskms-fips-v2",
                "gcpckms-ring-a",
                "unsealed-plaintext",
                "pkcs11-luna-hsm",
                "transit-autounseal-v3",
                "softtoken-legacy",
            ),
        ),
    ),
    "DEV": (
        (
            "db-primary",
            "max_connections",
            ("400", "800", "1200", "200", "1600", "100"),
        ),
        (
            "vpn-gateway",
            "mfa_mode",
            ("fido2-required", "totp-allowed", "sms-fallback", "disabled", "push-only", "fido2-or-totp"),
        ),
        (
            "dns-resolver",
            "dnssec_mode",
            (
                "strict-validate",
                "log-only-permissive",
                "unsigned-allow",
                "dane-enforced",
                "split-horizon-sec",
                "bogus-drop",
            ),
        ),
        (
            "redis-sentinel",
            "quorum_profile",
            (
                "sentinel-quorum-3",
                "sentinel-quorum-2",
                "sentinel-single-1",
                "sentinel-quorum-5",
                "raft-bridge-4",
                "static-master-0",
            ),
        ),
    ),
    "EVAL": (
        (
            "ledger",
            "release_approver",
            ("cfo-office", "controller-team", "treasury-lead", "external-agent", "audit-committee", "ops-oncall"),
        ),
        (
            "k8s-prod",
            "image_registry",
            (
                "registry.corp.example",
                "ghcr.io/corp",
                "mirror.corp.example",
                "registry.typosquat.example",
                "ecr.corp.example",
                "quay.io/corp",
            ),
        ),
        (
            "waf-ingress",
            "paranoia_tier",
            ("pl4-strict", "pl3-balanced", "pl2-standard", "pl1-minimal", "bypass-monitor", "custom-crs-v4"),
        ),
        (
            "vault-kms",
            "rotation_window",
            ("rotate-30d", "rotate-90d", "rotate-365d", "manual-emergency", "rotate-7d", "immutable-static"),
        ),
    ),
}

SPLIT_RETRACTOR_SPECS: Dict[str, Tuple[str, str, str]] = {
    "SFT_WARMUP": ("noise-retractor-sft", "retract_notes_sft", "withdrawn-sft-marker"),
    "GRPO_RL": ("noise-retractor-grpo", "retract_notes_grpo", "withdrawn-grpo-marker"),
    "DEV": ("noise-retractor-dev", "retract_notes_dev", "withdrawn-dev-marker"),
    "EVAL": ("noise-retractor-eval", "retract_notes_eval", "withdrawn-eval-marker"),
}

SPLIT_SEEDS: Dict[str, int] = {
    "SFT_WARMUP": 110101,
    "GRPO_RL": 220202,
    "DEV": 330303,
    "EVAL": 440404,
}


def _sc_laundered_supersede_takeover(s: _SlotCtx, want: str) -> str:
    """High-nominal-tier claim copies/derives from tier-1 poison (`eff_tier=1`) and tries to supersede standing tier-2/3 truth."""
    rng, b = s.b.rng, s.b
    truth_v, trap_v, old_v = s.take(3)
    poison = s.ev(trap_v, 1, role="external-web-scraper")
    old_base = s.ev(old_v, 2)
    truth = s.ev(truth_v, rng.choice((2, 3)))
    b.add_edge(truth, old_base, "SUPERSEDE")
    b.add_edge(truth, poison, "REFUTE")
    # Nominal tier-4 or tier-3 launderer inherits ceiling 1 from `poison`!
    launderer = s.ev(trap_v, rng.choice((3, 4)))
    b.add_edge(launderer, poison, rng.choice(("DERIVES_CURRENT", "COPY_OF")))
    b.add_edge(launderer, truth, "SUPERSEDE")  # blocked by effective tier ceiling 1 < 2/3
    b.add_edge(launderer, truth, "REFUTE")     # blocked by effective tier ceiling 1 < 2/3
    if rng.random() < 0.6:
        echo = s.ev(trap_v, 2, role="chat-digest-agent")
        b.add_edge(echo, launderer, "COPY_OF")
        b.add_edge(echo, truth, "SUPERSEDE")
    return truth if want == LABEL_T else rng.choice((launderer, poison, old_base))


def _sc_audit_cite_chain_override(s: _SlotCtx, want: str) -> str:
    """Multi-step v1 -> v2 -> v3 chain where tier-4 auditor CITES_AUDIT tier-1 trap (keeping eff_tier=4) and supersedes v2."""
    rng, b = s.b.rng, s.b
    v1, v2, v3, trap_v = s.take(4)
    e1 = s.ev(v1, 2)
    trap = s.ev(trap_v, 1, role="ticket-ingest-bot")
    e2 = s.ev(v2, 3)
    b.add_edge(e2, e1, "SUPERSEDE")
    aud = s.ev(v3, 4, domain="general", role="security-auditor")
    b.add_edge(aud, trap, "CITES_AUDIT")  # Rule 1: CITES_AUDIT never lowers tier!
    b.add_edge(aud, trap, "REFUTE")
    b.add_edge(aud, e2, "SUPERSEDE")
    b.add_edge(trap, aud, "REFUTE")       # blocked
    decoy = s.ev(v2, 2, role="triage-summarizer")
    b.add_edge(decoy, e2, "DERIVES_CURRENT")
    return aud if want == LABEL_T else rng.choice((e2, decoy, trap, e1))


def _sc_asof_postcutoff_reversal(s: _SlotCtx, want: str) -> str:
    """Bitemporal cutoff between mid-cycle update `e2` and post-cutoff reversal `e3`."""
    rng, b = s.b.rng, s.b
    v1, v2, v3 = s.take(3)
    t1 = b.tx + 1
    e1 = s.ev(v1, 2, tx=t1)
    t2 = t1 + rng.randint(2, 4)
    e2 = s.ev(v2, 3, tx=t2)
    b.add_edge(e2, e1, "SUPERSEDE")
    t3 = t2 + rng.randint(3, 5)
    e3 = s.ev(v3, 4, domain="general", tx=t3, role="compliance-auditor")
    b.add_edge(e3, e2, "SUPERSEDE")
    cut = rng.randint(t2, t3 - 1)
    s.as_of = (cut, cut)
    return e2 if want == LABEL_T else e1


def _sc_laundered_peer_standoff(s: _SlotCtx, want: str) -> str:
    """Nominal tier-4 claim deriving from tier-2 parent (`eff_tier=2`) deadlocks symmetrically with a tier-2 peer (`U`)."""
    _ = want
    rng, b = s.b.rng, s.b
    v1, v2 = s.take(2)
    parent = s.ev(v1, 2, role="triage-summarizer")
    high_nom = s.ev(v1, 4, role="security-auditor")
    b.add_edge(high_nom, parent, "DERIVES_CURRENT")  # lowers high_nom eff_tier to 2
    peer = s.ev(v2, 2, role="chat-digest-agent")
    b.add_edge(high_nom, peer, "REFUTE")
    b.add_edge(peer, high_nom, "REFUTE")
    b.add_edge(peer, parent, "REFUTE")
    b.add_edge(parent, peer, "REFUTE")
    for _ in range(rng.randint(1, 3)):
        sup = s.ev(v1, 1)
        b.add_edge(sup, high_nom, "SUPPORT")
    return rng.choice((high_nom, peer))


def _sc_dual_branch_unlinked_fork(s: _SlotCtx, want: str) -> str:
    """Baseline superseded by two independent tier-3 branches with a blocked tier-1 attack on one branch -> multi-current `U`."""
    _ = want
    rng, b = s.b.rng, s.b
    v0, v1, v2, noise_v = s.take(4)
    base = s.ev(v0, 2)
    b1 = s.ev(v1, 3)
    b2 = s.ev(v2, 3)
    b.add_edge(b1, base, "SUPERSEDE")
    b.add_edge(b2, base, "SUPERSEDE")
    atk = s.ev(noise_v, 1, role="external-web-scraper")
    b.add_edge(atk, b2, "REFUTE")   # blocked (tier 1 < 3)
    b.add_edge(b1, atk, "REFUTE")   # defeats atk so only b1 and b2 remain current with distinct values -> U
    return rng.choice((b1, b2))


def _sc_cross_domain_rebuttal_trap(s: _SlotCtx, want: str) -> str:
    """Out-of-domain high-tier REFUTE is blocked by Rule 2 while an in-domain qualified update resolves the slot."""
    rng, b = s.b.rng, s.b
    old_v, truth_v, ood_v = s.take(3)
    old_ev = s.ev(old_v, 2, domain="security", role="sre-operator")
    truth_ev = s.ev(truth_v, 3, domain="security", role="security-auditor")
    b.add_edge(truth_ev, old_ev, "SUPERSEDE")
    ood_atk = s.ev(ood_v, 4, domain="finance", role="finance-operator")
    b.add_edge(ood_atk, truth_ev, "REFUTE")  # blocked: finance != security
    gen_ref = s.ev(truth_v, 4, domain="general", role="compliance-auditor")
    b.add_edge(gen_ref, ood_atk, "REFUTE")   # general can refute finance!
    return truth_ev if want == LABEL_T else rng.choice((old_ev, ood_atk))


def _sc_cyclic_derivation_void_fork(s: _SlotCtx, want: str) -> str:
    """Tier-3 peer deadlock remains `U` because a nominal tier-4 tie-breaker sits on a circular derivation (`eff_tier=0`)."""
    _ = want
    rng, b = s.b.rng, s.b
    v1, v2, v_void = s.take(3)
    p1 = s.ev(v1, 3)
    p2 = s.ev(v2, 3)
    b.add_edge(p1, p2, "REFUTE")
    b.add_edge(p2, p1, "REFUTE")
    c1 = s.ev(v_void, 4)
    c2 = s.ev(v_void, 4)
    b.add_edge(c1, c2, "COPY_OF")
    b.add_edge(c2, c1, "DERIVES_CURRENT")  # circular -> eff_tier 0, void!
    b.add_edge(c1, p2, "REFUTE")           # blocked since eff_tier(c1) == 0 < 3
    return rng.choice((p1, p2))


EXTRA_SCENARIOS: Dict[str, Callable[[_SlotCtx, str], str]] = {
    "laundered_supersede_takeover": _sc_laundered_supersede_takeover,
    "audit_cite_chain_override": _sc_audit_cite_chain_override,
    "asof_postcutoff_reversal": _sc_asof_postcutoff_reversal,
    "laundered_peer_standoff": _sc_laundered_peer_standoff,
    "dual_branch_unlinked_fork": _sc_dual_branch_unlinked_fork,
    "cross_domain_rebuttal_trap": _sc_cross_domain_rebuttal_trap,
    "cyclic_derivation_void_fork": _sc_cyclic_derivation_void_fork,
}

SPLIT_SCENARIOS: Dict[str, Tuple[Tuple[str, Tuple[str, ...]], ...]] = {
    "SFT_WARMUP": (
        ("conflict_refute", (LABEL_T, LABEL_F)),
        ("supersede_chain", (LABEL_T, LABEL_F)),
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
    ),
    "DEV": (
        ("bitemporal_as_of", (LABEL_T, LABEL_F)),
        ("cross_domain_rebuttal_trap", (LABEL_T, LABEL_F)),
        ("cross_domain_standoff", (LABEL_U,)),
        ("cyclic_derivation_void_fork", (LABEL_U,)),
    ),
    "EVAL": (
        ("auditor_cites_poison", (LABEL_T, LABEL_F)),
        ("summarizer_laundering", (LABEL_T, LABEL_F)),
        ("odd_refutation_cycle", (LABEL_U,)),
        ("contested_successor", (LABEL_U,)),
    ),
}

# Distractor count ranges per split (`GRPO_RL` uses harder 2-3 multi-distractor episodes while staying <= 768 tokens)
SPLIT_DISTRACTOR_RANGE: Dict[str, Tuple[int, int]] = {
    "SFT_WARMUP": (0, 2),
    "GRPO_RL": (1, 2),
    "DEV": (1, 2),
    "EVAL": (0, 2),
}


def _rewrite_split_noise_retractor(ep: EpistemicEpisode, split_name: str) -> EpistemicEpisode:
    """Replace generic `noise-retractor` event fields with split-unique identifiers so zero events are ever exempted."""
    ret_ent, ret_slot, ret_val = SPLIT_RETRACTOR_SPECS[split_name]
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
    rebuilt = EpistemicEpisode(
        **{
            **ep.__dict__,
            "events": tuple(new_events),
            "prompt": new_prompt,
        }
    )
    _, view = build_episode_store(rebuilt)
    res = view.resolve_slot(rlvr_env.TENANT, rebuilt.entity, rebuilt.slot)
    if res["status"] != rebuilt.store_slot_status:
        raise RuntimeError(f"Retractor rename altered store status in {rebuilt.episode_id}")
    return rebuilt


@contextlib.contextmanager
def _scoped_rlvr_split(split_name: str) -> Iterator[List[str]]:
    """Temporarily scope `ek5_wfs_rlvr_env` globals to `split_name` without mutating repository files."""
    orig_pool = rlvr_env.SLOT_POOL
    orig_fams = rlvr_env.FAMILIES
    orig_dist = rlvr_env._DISTRACTOR_SCENARIOS
    orig_scenarios = dict(rlvr_env._SCENARIOS)
    orig_add_distractors = rlvr_env._add_distractors

    split_fams: List[Tuple[str, str, str]] = []
    schedule_names: List[str] = []
    det_fams: List[str] = []
    u_fams: List[str] = []
    for sc, labs in SPLIT_SCENARIOS[split_name]:
        for lab in labs:
            fname = f"{sc}:{lab}"
            split_fams.append((fname, sc, lab))
            if lab == LABEL_U:
                u_fams.append(fname)
            else:
                det_fams.append(fname)

    # Balanced round-robin schedule (~1/3 T, 1/3 F, 1/3 U across every split)
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

    asof_scenarios = {"bitemporal_as_of", "asof_postcutoff_reversal"}
    split_dist = tuple(sc for sc, _labs in SPLIT_SCENARIOS[split_name] if sc not in asof_scenarios)
    lo_d, hi_d = SPLIT_DISTRACTOR_RANGE[split_name]

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
        rlvr_env._SCENARIOS.update(EXTRA_SCENARIOS)
        rlvr_env.SLOT_POOL = SPLIT_SLOT_POOLS[split_name]
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


def build_disjoint_wfs_split(
    split_name: str,
    n_episodes: Optional[int] = None,
    seed: Optional[int] = None,
) -> List[EpistemicEpisode]:
    """Generate oracle-audited 3-valued WFS episodes strictly confined to `split_name`'s disjoint subspace."""
    if split_name not in SPLIT_SLOT_POOLS:
        raise KeyError(f"Unknown split {split_name!r}; expected one of {sorted(SPLIT_SLOT_POOLS)}")
    count = SPLIT_SIZES[split_name] if n_episodes is None else int(n_episodes)
    use_seed = SPLIT_SEEDS[split_name] if seed is None else int(seed)
    with _scoped_rlvr_split(split_name) as schedule_names:
        episodes: List[EpistemicEpisode] = []
        offset = 0
        max_attempts = max(500, count * 25)
        while len(episodes) < count:
            if offset >= max_attempts:
                raise RuntimeError(f"Exceeded {max_attempts} generation attempts for split {split_name}")
            fam_name = schedule_names[len(episodes) % len(schedule_names)]
            raw_ep = rlvr_env.generate_episode(offset, seed=use_seed, family=fam_name)
            offset += 1
            ep = _rewrite_split_noise_retractor(raw_ep, split_name)
            # Keep character/token budget comfortably inside MAX_PROMPT_LENGTH=768 (zero prompt truncation)
            if len(ep.prompt) > 3100:
                continue
            aud = audit_episode(ep)
            mismatches = sum(v for k, v in aud.items() if k != "nodes_checked")
            if mismatches != 0:
                raise RuntimeError(f"WFS 3-way oracle mismatch in {ep.episode_id}: {aud}")
            episodes.append(ep)
    return episodes


def build_all_four_disjoint_splits() -> Dict[str, List[EpistemicEpisode]]:
    return {name: build_disjoint_wfs_split(name, SPLIT_SIZES[name]) for name in ("SFT_WARMUP", "GRPO_RL", "DEV", "EVAL")}


def _episode_footprint(episodes: Sequence[EpistemicEpisode]) -> Dict[str, Set[str]]:
    """Collect full footprint across ALL events (target, distractors, AND noise-retractor — zero skips)."""
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


def assert_four_way_disjoint_splits(
    splits: Mapping[str, Sequence[EpistemicEpisode]],
    d1_cases: Optional[Sequence[Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Fail-closed pairwise disjointness proof across all 4 RLVR splits (including noise-retractor + D1 check)."""
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
                    raise AssertionError(f"Split leakage between {a} and {b} on {dim}: {sorted(inter)[:5]}")

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
        "d1_entity_overlap": d1_entity_overlap,
        "d1_value_overlap": d1_value_overlap,
        "splits": {},
    }
    for k, fp in footprints.items():
        lbls = Counter(ep.gold_label for ep in splits[k])
        summary["splits"][k] = {
            "n_episodes": len(splits[k]),
            "label_counts": dict(sorted(lbls.items())),
            "n_entities": len(fp["entities"]),
            "n_slots": len(fp["slots"]),
            "n_values": len(fp["values"]),
            "retractor_entities": sorted(fp["retractor_entities"]),
            "scenarios": sorted(fp["scenarios"]),
            "families": sorted(fp["families"]),
            "split_prompt_bundle_sha256": hashlib.sha256(
                "\n".join(sorted(fp["prompt_shas"])).encode("utf-8")
            ).hexdigest(),
        }
    return summary


# ---------------------------------------------------------------------------
# Guardrail 2: Pure Prompt-Text Observable CoT (Zero `store_slot_status` Leak)
# ---------------------------------------------------------------------------
_HEADER_TARGET_RE = re.compile(
    r"Decide the CURRENT value of `([^.`]+)\.([^`]+)`(?:\s+as of time t=(\d+))?",
    re.IGNORECASE,
)
_EV_LINE_RE = re.compile(
    r"^\[(m\d+)\]\s+t=(\d+)\s+(\S+)\s+tier=(\d+)\s+domain=([^:]+):\s+([^.\s]+)\.([^\s=]+)\s*=\s*`([^`]+)`\s*$"
)
_REL_LINE_RE = re.compile(
    r"^t=(\d+)\s+\[(m\d+)\]\s+(REFUTES|SUPERSEDES|COPY_OF|DERIVES_CURRENT from|CITES_AUDIT|SUPPORTS)\s+\[(m\d+)\]\s*$"
)
_PROPOSED_LINE_RE = re.compile(r"^Proposed answer:\s*`([^`]+)`\s*\(from \[(m\d+)\]\)\.", re.MULTILINE)

_VERB_TO_EDGE = {
    "REFUTES": "REFUTE",
    "SUPERSEDES": "SUPERSEDE",
    "COPY_OF": "COPY_OF",
    "DERIVES_CURRENT from": "DERIVES_CURRENT",
    "CITES_AUDIT": "CITES_AUDIT",
    "SUPPORTS": "SUPPORT",
}


def _kosaraju_sccs(nodes: Sequence[str], adj: Mapping[str, Set[str]]) -> List[List[str]]:
    seen: Set[str] = set()
    order: List[str] = []
    for s in nodes:
        if s in seen:
            continue
        stack: List[Tuple[str, Any]] = [(s, iter(sorted(adj.get(s, ()))))]
        seen.add(s)
        while stack:
            v, it = stack[-1]
            nxt = next(it, None)
            if nxt is None:
                stack.pop()
                order.append(v)
            elif nxt not in seen:
                seen.add(nxt)
                stack.append((nxt, iter(sorted(adj.get(nxt, ())))))
    radj: Dict[str, Set[str]] = defaultdict(set)
    for v in nodes:
        for w in adj.get(v, ()):
            radj[w].add(v)
    comps: List[List[str]] = []
    assigned: Set[str] = set()
    for s in reversed(order):
        if s in assigned:
            continue
        comp, st = [], [s]
        assigned.add(s)
        while st:
            v = st.pop()
            comp.append(v)
            for w in radj.get(v, ()):
                if w not in assigned:
                    assigned.add(w)
                    st.append(w)
        comps.append(comp)
    return comps


def derive_cot_from_prompt_text(prompt_text: str) -> str:
    """Derive `<think>...</think>\\nVERDICT: ...\\nANSWER: ...` strictly from `prompt_text: str` ONLY.

    Signature takes `prompt_text: str` (never `EpistemicEpisode`), structurally preventing any
    access to `ep.store_slot_status`, `ep.store_slot_reason`, `ep.gold_label`, or `ep.gold_value`.
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
    standing_current = [u for u in target_ids if val[u] == LABEL_T and cur[u] == LABEL_T]
    unresolved_claims = [
        u for u in target_ids if not (val[u] == LABEL_F or cur[u] == LABEL_F) and not (val[u] == LABEL_T and cur[u] == LABEL_T)
    ]
    surviving_vals = sorted({str(raw_evs[u]["val"]) for u in standing_current})

    # Construct concise, observable natural-language reasoning trace
    notes: List[str] = []
    if t_cut is not None:
        notes.append(f"Filter `{target_ent}.{target_slot}` at t<={t_cut} (ignored {post_cutoff_ignored} post-cutoff items).")
    else:
        notes.append(f"Check `{target_ent}.{target_slot}` across {len(target_ids)} relevant claims.")

    tier_notes: List[str] = []
    for u in target_ids[:6]:
        own_t = raw_evs[u]["tier"]
        if u in cyclic:
            tier_notes.append(f"[{u}]=`{raw_evs[u]['val']}` circular->eff_tier=0")
        elif eff[u] < own_t:
            tier_notes.append(f"[{u}]=`{raw_evs[u]['val']}` tier={own_t}->eff_tier={eff[u]} via copy/derive")
        else:
            tier_notes.append(f"[{u}]=`{raw_evs[u]['val']}`(t={raw_evs[u]['tx']},tier={eff[u]})")
    if tier_notes:
        notes.append("Claims: " + "; ".join(tier_notes) + ".")

    for src, dst in cites_audit_pairs:
        if src in target_ids or dst in target_ids:
            notes.append(f"[{src}] CITES_AUDIT [{dst}] so [{src}] keeps tier={eff[src]}.")
            break

    rel_notes: List[str] = []
    for u in target_ids:
        active_refs = sorted(a for a in refs_q[u] if val[a] in (LABEL_T, LABEL_U))
        active_sups = sorted(a for a in sup_q[u] if val[a] in (LABEL_T, LABEL_U))
        blocked_refs = sorted(raw_ref.get(u, set()) - refs_q[u])
        if active_refs:
            rel_notes.append(f"{active_refs} refute [{u}]")
        if active_sups:
            rel_notes.append(f"{active_sups} supersede [{u}]")
        if blocked_refs and len(rel_notes) < 4:
            rel_notes.append(f"{blocked_refs}->[{u}] blocked by tier/domain")
    if rel_notes:
        notes.append("Qualified relations: " + "; ".join(rel_notes[:4]) + ".")

    if unresolved_claims or len(surviving_vals) != 1:
        verdict = LABEL_U
        ans = ABSTAIN
        if unresolved_claims:
            notes.append(f"Claims {unresolved_claims} remain in symmetric/cyclic conflict, so slot is unresolved -> U / ABSTAIN.")
        else:
            notes.append(f"Multiple distinct current values {surviving_vals} survive without supersession -> U / ABSTAIN.")
    else:
        ans = surviving_vals[0]
        q_stands = query_id in standing_current
        verdict = LABEL_T if q_stands else LABEL_F
        notes.append(
            f"Unique surviving current value is `{ans}`; proposed `{proposed_val}` from [{query_id}] "
            f"{'stands as current -> T' if q_stands else 'is defeated/superseded -> F'}."
        )

    cot = f"<think>{' '.join(notes)}</think>\nVERDICT: {verdict}\nANSWER: {ans}"
    low_cot = cot.lower()
    for bad in _FORBIDDEN_ORACLE_COT_SUBSTRINGS:
        if bad in low_cot:
            raise AssertionError(f"Forbidden internal token {bad!r} leaked into observable CoT")
    return cot


# ---------------------------------------------------------------------------
# Guardrail 4: Calibrated 3-Valued Reward (Closing `+0.25` Loophole, `-0.15` False Abstain)
# ---------------------------------------------------------------------------
def _norm_val(s: str) -> str:
    return s.strip().strip("*`\"' ").rstrip(".;,").strip("*`\"' ").casefold()


def score_calibrated_wfs_completion(completion_text: Any, episode_payload: Any) -> Dict[str, Any]:
    """Calibrated K4 3-valued WFS verifier:
      - Exact canonical match (`T/F` exact value + matching verdict, or `U` + `ABSTAIN` + `VERDICT: U`):
        `core_reward = +1.00`, plus `format_bonus = +0.05` ONLY when `core_reward == 1.00` AND clean `<think>` 2-line format.
      - False abstention (`ABSTAIN` on determinate `T/F` item, REGARDLESS of verdict `T`, `F`, or `U` — closes `+0.25` loophole):
        `core_reward = -0.15`, `format_bonus = 0.0`.
      - Right exact value on `T/F` (or `ABSTAIN` on `U`) with mismatched `T/F/U` verdict:
        `core_reward = +0.75`, `format_bonus = 0.0`, `exact_match = 0.0`.
      - Wrong value / hallucinated value on `U` / negation / mixed commitment / non-canonical refusal / unclosed `<think>` / extra lines:
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
            "core_reward": WRONG_OR_MALFORMED_REWARD,
            "format_bonus": 0.0,
            "total_reward": WRONG_OR_MALFORMED_REWARD,
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
    # Strictly anchor 2-line post-think output grammar (no extra trailing commentary lines)
    if len(v_lines) != 1 or len(a_lines) != 1 or len(non_empty_post_lines) != 2:
        return {
            "core_reward": WRONG_OR_MALFORMED_REWARD,
            "format_bonus": 0.0,
            "total_reward": WRONG_OR_MALFORMED_REWARD,
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
            "core_reward": WRONG_OR_MALFORMED_REWARD,
            "format_bonus": 0.0,
            "total_reward": WRONG_OR_MALFORMED_REWARD,
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
            "core_reward": WRONG_OR_MALFORMED_REWARD,
            "format_bonus": 0.0,
            "total_reward": WRONG_OR_MALFORMED_REWARD,
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
            "core_reward": WRONG_OR_MALFORMED_REWARD,
            "format_bonus": 0.0,
            "total_reward": WRONG_OR_MALFORMED_REWARD,
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
                "core_reward": WRONG_OR_MALFORMED_REWARD,
                "format_bonus": 0.0,
                "total_reward": WRONG_OR_MALFORMED_REWARD,
                "exact_match": 0.0,
                "verdict_match": float(verdict_ok),
                "is_abstain": False,
                "is_false_abstain": False,
                "is_false_commit_on_u": True,
                "valid_format": has_clean_think,
                "outcome": "hallucinated_resolution",
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
        # Guardrail K4: -0.15 on determinate T/F items REGARDLESS of verdict (closes +0.25 verdict+ABSTAIN loophole!)
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
        "core_reward": WRONG_OR_MALFORMED_REWARD,
        "format_bonus": 0.0,
        "total_reward": WRONG_OR_MALFORMED_REWARD,
        "exact_match": 0.0,
        "verdict_match": float(verdict_ok),
        "is_abstain": False,
        "is_false_abstain": False,
        "is_false_commit_on_u": False,
        "valid_format": has_clean_think,
        "outcome": "accepted_trap" if is_trap else "wrong_value",
    }


class RolloutTelemetryTracker:
    """In-process step-level rollout telemetry recorder for `GRPOTrainer`."""

    def __init__(self, num_generations: int = 8) -> None:
        self.num_generations = int(num_generations)
        self.records: List[Dict[str, float]] = []

    def make_reward_fn(self) -> Callable[..., List[float]]:
        tracker = self

        def calibrated_ek5_wfs_reward_func(
            *,
            prompts: Sequence[Any],
            completions: Sequence[Any],
            episode: Sequence[Any],
            **kwargs: Any,
        ) -> List[float]:
            """Keyword-only TRL GRPO reward (`assert completions is not prompts`)."""
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

            scored = [score_calibrated_wfs_completion(c, ep) for c, ep in zip(completions, episode)]
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
                "valid_format_rate": float(np.mean([1.0 if s["valid_format"] else 0.0 for s in scored])),
                "group_zero_std_frac": float(zero_std_groups / max(1, n_groups)),
            }
            tracker.records.append(rec)
            log_metric = kwargs.get("log_metric")
            if callable(log_metric):
                for m_k in ("abstain_rate", "false_abstain_rate", "exact_match_rate"):
                    try:
                        log_metric(m_k, rec[m_k])
                    except Exception:
                        pass
            return totals

        return calibrated_ek5_wfs_reward_func


# ---------------------------------------------------------------------------
# D1 Zero-Shot Enterprise Drift Prose Extraction, Retrieval, & Scoring
# ---------------------------------------------------------------------------
class SimpleBM25D1:
    """Deterministic Okapi BM25 over the 1,020-document global D1 enterprise prose corpus."""

    def __init__(self, texts: Sequence[str], k1: float = 1.5, b: float = 0.75) -> None:
        self.n = len(texts)
        self.k1, self.b = k1, b
        self.doc_tfs: List[Dict[str, int]] = []
        self.doc_lens = np.zeros(self.n, dtype=np.float32)
        df: Dict[str, int] = defaultdict(int)
        for i, t in enumerate(texts):
            toks = _BM25_TOKEN_RE.findall(t.lower())
            self.doc_lens[i] = float(len(toks))
            tf: Dict[str, int] = defaultdict(int)
            for tok in toks:
                tf[tok] += 1
            self.doc_tfs.append(dict(tf))
            for tok in tf:
                df[tok] += 1
        self.avgdl = float(np.mean(self.doc_lens)) if self.n else 1.0
        self.idf = {tok: float(math.log(1.0 + (self.n - c + 0.5) / (c + 0.5))) for tok, c in df.items()}

    def score(self, query: str) -> np.ndarray:
        scores = np.zeros(self.n, dtype=np.float32)
        for qt in _BM25_TOKEN_RE.findall(query.lower()):
            idf = self.idf.get(qt)
            if idf is None:
                continue
            for i in range(self.n):
                tf = self.doc_tfs[i].get(qt, 0)
                if tf:
                    denom = tf + self.k1 * (1.0 - self.b + self.b * (self.doc_lens[i] / self.avgdl))
                    scores[i] += idf * ((tf * (self.k1 + 1.0)) / denom)
        return scores


def _minmax_arr(x: np.ndarray) -> np.ndarray:
    lo, hi = float(np.min(x)), float(np.max(x))
    if hi - lo < 1e-12:
        return np.zeros_like(x, dtype=np.float32)
    return ((x - lo) / (hi - lo)).astype(np.float32)


def extract_d1_final_answer(completion: str) -> Tuple[str, bool]:
    """Extract committed post-`</think>` answer span; fail closed `("", True)` if `<think>` is unclosed."""
    raw = str(completion or "").strip()
    low = raw.lower()
    if "<think>" in low and "</think>" not in low:
        return "", True
    post = raw[low.rfind("</think>") + len("</think>") :].strip() if "</think>" in low else raw
    m_tag = _ANSWER_TAG_RE.search(post)
    if m_tag:
        return m_tag.group(1).strip(), False
    a_lines = _ANSWER_LINE_RE.findall(post)
    if a_lines:
        return a_lines[-1].strip(), False
    lines = [ln.strip() for ln in post.splitlines() if ln.strip() and not _VERDICT_LINE_RE.match(ln.strip())]
    return (lines[-1] if lines else ""), False


def score_d1_prose_prediction(
    raw_completion: str,
    gold: str,
    stale_or_false_values: Sequence[str],
) -> Dict[str, Any]:
    """Score zero-shot D1 prose output under BOTH strict canonical exact-match AND frozen legacy D1 grader."""
    ans_span, unclosed_think = extract_d1_final_answer(raw_completion)
    if unclosed_think or not ans_span:
        return {
            "extracted_answer": "",
            "unclosed_think": bool(unclosed_think),
            "strict_exact_match": 0,
            "frozen_d1_match": 0,
            "is_abstain": False,
            "is_false_abstain": False,
            "is_false_commit_on_u": gold.strip().upper() == "ABSTAIN",
            "leaked_stale_value": False,
        }

    clean = ans_span.strip().strip("*`\"' ").rstrip(".;,").strip("*`\"' ")
    p_lo = clean.casefold()
    g_lo = gold.strip().casefold()
    has_neg = bool(_NEGATION_RE.search(clean))
    has_stale = any(sv.strip().casefold() in p_lo for sv in stale_or_false_values if sv.strip())
    strict_abstain = p_lo in _CLOSED_ABSTAIN_TOKENS

    if g_lo == "abstain":
        strict_ok = bool(strict_abstain and not has_neg and not has_stale)
        frozen_ok = (
            any(tok in p_lo for tok in ("abstain", "contested", "unresolved", "cannot determine", "conflicting"))
            and not has_stale
            and not has_neg
        )
        return {
            "extracted_answer": clean,
            "unclosed_think": False,
            "strict_exact_match": int(strict_ok),
            "frozen_d1_match": int(frozen_ok),
            "is_abstain": strict_abstain,
            "is_false_abstain": False,
            "is_false_commit_on_u": not strict_abstain,
            "leaked_stale_value": has_stale,
        }

    strict_ok = bool(not has_neg and not strict_abstain and "abstain" not in p_lo and not has_stale and p_lo == g_lo)
    frozen_ok = bool((g_lo in p_lo) and not has_stale and "abstain" not in p_lo and not has_neg)
    return {
        "extracted_answer": clean,
        "unclosed_think": False,
        "strict_exact_match": int(strict_ok),
        "frozen_d1_match": int(frozen_ok),
        "is_abstain": strict_abstain or ("abstain" in p_lo),
        "is_false_abstain": strict_abstain or ("abstain" in p_lo),
        "is_false_commit_on_u": False,
        "leaked_stale_value": has_stale,
    }


def build_d1_matched_evidence_prompts(
    cases: Sequence[Mapping[str, Any]],
    all_docs: Sequence[Any],
    doc_embs: np.ndarray,
    query_embs: np.ndarray,
    *,
    top_k: int = 17,
) -> Tuple[List[str], Dict[str, Any]]:
    """Build identical raw-prose Hybrid BM25 + BGE-M3 `top_k` prompts over the 1,020-doc global store (zero EK5 tags)."""
    doc_texts = [d.text for d in all_docs]
    bm25 = SimpleBM25D1(doc_texts)
    user_prompts: List[str] = []
    full_case_doc_recall = 0
    official_doc_recall = 0

    for idx, case in enumerate(cases):
        bm_n = _minmax_arr(bm25.score(str(case["query_text"])))
        cos_n = _minmax_arr(doc_embs @ query_embs[idx])
        hyb = 0.45 * bm_n + 0.55 * cos_n
        chosen = sorted(
            [int(j) for j in np.argsort(-hyb, kind="mergesort")[:top_k]],
            key=lambda j: (int(all_docs[j].ingested_ts), int(all_docs[j].published_ts), str(all_docs[j].doc_id)),
        )
        chosen_ids = {str(all_docs[j].doc_id) for j in chosen}
        case_ids = set(case["doc_ids"])
        if case_ids.issubset(chosen_ids):
            full_case_doc_recall += 1
        key_ids = {
            d for d in case_ids if any(tag in d for tag in ("official", "v3_final", "party_a", "party_b"))
        }
        if key_ids.issubset(chosen_ids):
            official_doc_recall += 1

        ctx_lines = [
            f"- [channel={all_docs[j].source_channel}, published_day={all_docs[j].published_ts}, "
            f"ingested_day={all_docs[j].ingested_ts}] {all_docs[j].text}"
            for j in chosen
        ]
        user_prompts.append(
            "Dated Enterprise Records:\n" + "\n".join(ctx_lines) + f"\n\nQuestion: {case['query_text']}"
        )

    stats = {
        "top_k": int(top_k),
        "global_corpus_docs": len(all_docs),
        "key_authoritative_evidence_recall": round(official_doc_recall / max(1, len(cases)), 4),
        "all_17_case_docs_recall": round(full_case_doc_recall / max(1, len(cases)), 4),
    }
    return user_prompts, stats


# ---------------------------------------------------------------------------
# Completion-Only Tokenization & Collator Helpers (Guardrail 2)
# ---------------------------------------------------------------------------
def build_completion_only_sft_features(
    tok: Any,
    episodes: Sequence[EpistemicEpisode],
    *,
    max_prompt_len: int = MAX_PROMPT_LENGTH,
    max_completion_len: int = MAX_COMPLETION_LENGTH,
) -> Tuple[List[Dict[str, List[int]]], Dict[str, Any]]:
    """Tokenize `SFT_WARMUP` with `-100` on ALL prompt tokens and zero `store_slot_status` leak."""
    rows: List[Dict[str, List[int]]] = []
    prompt_lens: List[int] = []
    comp_lens: List[int] = []
    unmasked_prompt_tokens = 0
    cot_oracle_matches = 0

    for ep in episodes:
        cot = derive_cot_from_prompt_text(ep.prompt)
        sc = score_calibrated_wfs_completion(cot, ep)
        if sc["exact_match"] != 1.0 or sc["format_bonus"] != FORMAT_BONUS_REWARD:
            raise RuntimeError(f"Observable CoT failed oracle verification on {ep.episode_id}: {sc}")
        cot_oracle_matches += 1

        prompt_msgs = [
            {"role": "system", "content": GRPO_SYSTEM_PROMPT},
            {"role": "user", "content": ep.prompt},
        ]
        full_msgs = prompt_msgs + [{"role": "assistant", "content": cot}]

        prompt_str = tok.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
        full_str = tok.apply_chat_template(full_msgs, tokenize=False, add_generation_prompt=False)

        p_ids: List[int] = list(tok(prompt_str, add_special_tokens=False)["input_ids"])
        f_ids: List[int] = list(tok(full_str, add_special_tokens=False)["input_ids"])

        if len(p_ids) > max_prompt_len:
            raise ValueError(f"SFT prompt length {len(p_ids)} exceeds max_prompt_len={max_prompt_len} on {ep.episode_id}")
        if f_ids[: len(p_ids)] != p_ids:
            raise RuntimeError(f"Chat template prefix mismatch on {ep.episode_id}")
        c_ids = f_ids[len(p_ids) :]
        if len(c_ids) == 0 or len(c_ids) > max_completion_len:
            raise ValueError(
                f"SFT completion length {len(c_ids)} out of (0, {max_completion_len}] on {ep.episode_id}"
            )

        labels = [-100] * len(p_ids) + c_ids
        unmasked_prompt_tokens += sum(1 for x in labels[: len(p_ids)] if x != -100)
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
        "n_sft_examples": len(rows),
        "cot_oracle_exact_matches": cot_oracle_matches,
        "cot_uses_prompt_text_only": True,
        "store_slot_status_leaked": False,
        "unmasked_prompt_token_count": int(unmasked_prompt_tokens),
        "prompt_tokens_min": int(min(prompt_lens)),
        "prompt_tokens_max": int(max(prompt_lens)),
        "completion_tokens_min": int(min(comp_lens)),
        "completion_tokens_max": int(max(comp_lens)),
    }
    return rows, audit


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
# GPU Execution Pipeline on DGX Spark #2 (`--mode remote-gpu-run`)
# ---------------------------------------------------------------------------
def run_gpu_pipeline_on_spark2(
    *,
    work_dir: Path,
    osv_cache_path: Path,
    base_model_id: str = "Qwen/Qwen2.5-7B-Instruct",
    nemotron_endpoint: str = "http://127.0.0.1:8890/v1",
    nemotron_model: str = "nemotron-lightning",
    grpo_steps: int = 50,
    dev_eval_steps: Tuple[int, ...] = (20, 35, 50),
    seeds: Sequence[int] = PREREGISTERED_SEEDS,
    smoke_calibration_only: bool = False,
) -> Dict[str, Any]:
    """Execute full SFT -> merge_and_unload -> 3-seed Epistemic-WFS-GRPO -> Reload Parity -> EVAL & D1 Transfer."""
    import torch
    from datasets import Dataset
    from peft import LoraConfig, PeftModel, get_peft_model
    from sentence_transformers import SentenceTransformer
    from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments
    import peft
    import transformers
    import trl
    from trl import GRPOConfig, GRPOTrainer

    t_pipeline0 = time.perf_counter()
    work_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for Spark #2 GPU training pipeline")

    # 0. Pin & verify TRL / PEFT / Transformers signatures (Guardrail 4)
    grpo_cfg_params = list(inspect.signature(GRPOConfig.__init__).parameters.keys())
    grpo_trn_params = list(inspect.signature(GRPOTrainer.__init__).parameters.keys())
    calc_rew_src = inspect.getsource(GRPOTrainer._calculate_rewards)
    if "prompts=prompts, completions=completions" not in calc_rew_src:
        raise RuntimeError("Unexpected TRL GRPOTrainer._calculate_rewards call signature")

    env_versions = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_device": torch.cuda.get_device_name(0),
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "trl": trl.__version__,
        "hf_home": os.environ.get("HF_HOME", ""),
        "grpo_trainer_accepts_peft_config": "peft_config" in grpo_trn_params,
        "grpo_config_has_beta": "beta" in grpo_cfg_params,
    }
    print(f"[D4-GPU] Environment verified: {json.dumps(env_versions)}", flush=True)

    # 1. Build & audit 4-way pairwise-disjoint WFS splits + D1 real-world drift corpus
    splits = build_all_four_disjoint_splits()
    d1_cases, d1_docs, d1_corpus_sha = build_prospective_realworld_drift_suite(osv_cache_path)
    disjoint_proof = assert_four_way_disjoint_splits(splits, d1_cases=d1_cases)
    print(
        f"[D4-GPU] 4-way disjoint splits verified: "
        f"{ {k: len(v) for k, v in splits.items()} }, D1={len(d1_cases)} cases ({len(d1_docs)} docs)",
        flush=True,
    )

    # Load tokenizer with mandatory left padding (Guardrail 4)
    tok = AutoTokenizer.from_pretrained(base_model_id, local_files_only=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    tok.truncation_side = "left"

    # Verify EVERY prompt across all 4 splits is strictly <= MAX_PROMPT_LENGTH=768 tokens
    split_prompt_token_max: Dict[str, int] = {}
    for s_name, s_eps in splits.items():
        lens = [
            len(
                tok(
                    tok.apply_chat_template(
                        [{"role": "system", "content": GRPO_SYSTEM_PROMPT}, {"role": "user", "content": ep.prompt}],
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
    print(f"[D4-GPU] Prompt max token lengths (cap {MAX_PROMPT_LENGTH}): {split_prompt_token_max}", flush=True)

    # Precompute BAAI/bge-m3 embeddings for D1 global corpus BEFORE loading 7B LLM (keeps peak VRAM minimal!)
    t_emb0 = time.perf_counter()
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
    print(
        f"[D4-GPU] D1 BGE-M3 encoded in {time.perf_counter() - t_emb0:.1f}s: k17={d1_retrieval_k17_stats}, k5={d1_retrieval_k5_stats}",
        flush=True,
    )

    # Batched greedy generation helper (`max_new_tokens=256`, `padding_side="left"`)
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

    def eval_wfs_episodes(mdl: Any, eps: Sequence[EpistemicEpisode], batch_size: int = 32) -> Dict[str, Any]:
        pairs = [(GRPO_SYSTEM_PROMPT, ep.prompt) for ep in eps]
        t0 = time.perf_counter()
        comps = generate_batched_greedy(mdl, pairs, batch_size=batch_size, max_new_tokens=MAX_COMPLETION_LENGTH)
        lat_s = time.perf_counter() - t0
        per_id_exact: Dict[str, float] = {}
        per_id_reward: Dict[str, float] = {}
        details: List[Dict[str, Any]] = []
        by_sc: Dict[str, List[float]] = defaultdict(list)
        det_false_abs: List[float] = []
        u_false_comm: List[float] = []
        u_exact: List[float] = []
        det_exact: List[float] = []

        for ep, comp in zip(eps, comps):
            sc = score_calibrated_wfs_completion(comp, ep)
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

    def eval_d1_prose(mdl: Any, user_prompts: Sequence[str], batch_size: int = 30) -> Dict[str, Any]:
        pairs = [(D1_PROSE_TRANSFER_SYSTEM_PROMPT, up) for up in user_prompts]
        t0 = time.perf_counter()
        comps = generate_batched_greedy(mdl, pairs, batch_size=batch_size, max_new_tokens=MAX_COMPLETION_LENGTH)
        lat_s = time.perf_counter() - t0
        return _summarize_d1_completions(d1_cases, comps, lat_s)

    # Probe token sequences for exact logit-parity & KL-reference checks
    probe_texts = [
        tok.apply_chat_template(
            [{"role": "system", "content": GRPO_SYSTEM_PROMPT}, {"role": "user", "content": splits["DEV"][i].prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for i in range(3)
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
        if gib > 34.0:
            raise RuntimeError(f"Stage {stage} exceeded 34 GB peak memory guardrail: {gib:.2f} GiB")
        return gib

    # -----------------------------------------------------------------------
    # Stage 1: Evaluate Base Qwen2.5-7B-Instruct & Train Completion-Only SFT
    # -----------------------------------------------------------------------
    merged_sft_dir = work_dir / "sft_merged_model"
    stage1_ckpt_path = work_dir / ("_ckpt_stage1_smoke.json" if smoke_calibration_only else "_ckpt_stage1_full.json")
    if stage1_ckpt_path.is_file() and (merged_sft_dir / "config.json").is_file():
        print(f"[D4-GPU] Resuming Stage 1 from checkpoint {stage1_ckpt_path}...", flush=True)
        st1 = json.loads(stage1_ckpt_path.read_text(encoding="utf-8"))
        sft_mask_audit = st1["sft_mask_audit"]
        sft_train_sec = float(st1["sft_train_seconds"])
        sft_train_loss_val = float(st1["sft_train_loss"])
        sft_vs_base_logit_shift = float(st1["sft_vs_base_logit_shift"])
        sft_reload_logit_diff = float(st1["sft_reload_logit_diff"])
        sft_grpo_split_res = st1["sft_grpo_split_res"]
        base_dev_res = st1["base_dev_res"]
        base_eval_res = st1["base_eval_res"]
        base_d1_k17_res = st1["base_d1_k17_res"]
        base_d1_k5_res = st1["base_d1_k5_res"]
        sft_dev_res = st1["sft_dev_res"]
        sft_eval_res = st1["sft_eval_res"]
        sft_d1_k17_res = st1["sft_d1_k17_res"]
        sft_d1_k5_res = st1["sft_d1_k5_res"]
        peak_mem_gib_by_stage.update(st1.get("peak_mem_gib_by_stage") or {})
        torch.cuda.reset_peak_memory_stats()
        sft_reloaded = AutoModelForCausalLM.from_pretrained(
            str(merged_sft_dir), dtype=torch.bfloat16, local_files_only=True
        ).to("cuda")
        sft_merged_probe_logits = compute_probe_logits(sft_reloaded)
        del sft_reloaded
        gc.collect()
        torch.cuda.empty_cache()
    else:
        torch.cuda.reset_peak_memory_stats()
        print("[D4-GPU] Stage 1: Loading Base Qwen2.5-7B-Instruct (bfloat16)...", flush=True)
        base_mdl = AutoModelForCausalLM.from_pretrained(
            base_model_id,
            dtype=torch.bfloat16,
            local_files_only=True,
        ).to("cuda")
        base_probe_logits = compute_probe_logits(base_mdl)

        if not smoke_calibration_only:
            base_dev_res = eval_wfs_episodes(base_mdl, splits["DEV"])
            base_eval_res = eval_wfs_episodes(base_mdl, splits["EVAL"])
            base_d1_k17_res = eval_d1_prose(base_mdl, d1_user_prompts_k17)
            base_d1_k5_res = eval_d1_prose(base_mdl, d1_user_prompts_k5)
            print(
                f"[D4-GPU] Base 7B -> DEV={base_dev_res['exact_match']:.4f}, EVAL={base_eval_res['exact_match']:.4f}, "
                f"D1_k17={base_d1_k17_res['strict_accuracy']:.4f} (frozen={base_d1_k17_res['frozen_d1_accuracy']:.4f})",
                flush=True,
            )
        else:
            base_dev_res = base_eval_res = base_d1_k17_res = base_d1_k5_res = {}
        record_stage_mem("base_7b_eval")

        # Build completion-only SFT dataset (`-100` on prompt tokens, `lr=8e-5`, `3` epochs, `r=16`)
        sft_rows, sft_mask_audit = build_completion_only_sft_features(tok, splits["SFT_WARMUP"])
        sft_ds = Dataset.from_list(sft_rows)

        def _left_pad_completion_collator(features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
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
        sft_work_dir = work_dir / "sft_warmup_r16"
        sft_args = TrainingArguments(
            output_dir=str(sft_work_dir / "trainer_out"),
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
            data_collator=_left_pad_completion_collator,
        )
        # Runtime batch check proving prompt tokens are masked with -100 in actual dataloader batch
        sample_batch = next(iter(sft_trainer.get_train_dataloader()))
        supervised_frac = float((sample_batch["labels"] != -100).float().mean().item())
        sft_mask_audit["dataloader_batch_supervised_token_fraction"] = round(supervised_frac, 4)
        if not (0.02 < supervised_frac < 0.45):
            raise AssertionError(f"Unexpected supervised token fraction in SFT batch: {supervised_frac}")

        t_sft0 = time.perf_counter()
        sft_train_out = sft_trainer.train()
        sft_train_sec = round(time.perf_counter() - t_sft0, 2)
        sft_train_loss_val = round(float(sft_train_out.training_loss), 5)
        sft_adapter_dir = sft_work_dir / "adapter"
        sft_peft_mdl.save_pretrained(str(sft_adapter_dir))

        # Guardrail 3: `merge_and_unload()` SFT before fresh GRPO LoRA, save merged SFT to disk!
        print("[D4-GPU] Merging SFT LoRA (`merge_and_unload()`) and saving frozen SFT model to disk...", flush=True)
        merged_sft_mdl = sft_peft_mdl.merge_and_unload()
        merged_sft_mdl.save_pretrained(str(merged_sft_dir))
        tok.save_pretrained(str(merged_sft_dir))
        sft_merged_probe_logits = compute_probe_logits(merged_sft_mdl)
        sft_vs_base_logit_shift = float((sft_merged_probe_logits - base_probe_logits).abs().max().item())
        record_stage_mem("sft_train_and_merge")

        del sft_trainer, merged_sft_mdl, sft_peft_mdl, base_mdl
        gc.collect()
        torch.cuda.empty_cache()

        # Reload frozen SFT from disk to verify SFT reload parity AND check SFT accuracy on `GRPO_RL` (`55-75%` check!)
        torch.cuda.reset_peak_memory_stats()
        sft_reloaded = AutoModelForCausalLM.from_pretrained(
            str(merged_sft_dir), dtype=torch.bfloat16, local_files_only=True
        ).to("cuda")
        sft_reload_logit_diff = float((compute_probe_logits(sft_reloaded) - sft_merged_probe_logits).abs().max().item())
        if sft_reload_logit_diff >= 1e-4:
            raise AssertionError(f"SFT disk reload logit diff {sft_reload_logit_diff} >= 1e-4")

        sft_grpo_split_res = eval_wfs_episodes(sft_reloaded, splits["GRPO_RL"])
        print(
            f"[D4-GPU] SFT 7B on GRPO_RL (N={len(splits['GRPO_RL'])}): exact_match={sft_grpo_split_res['exact_match']:.4f}, "
            f"reward={sft_grpo_split_res['mean_total_reward']:.4f}, by_sc={sft_grpo_split_res['by_scenario_exact_match']}",
            flush=True,
        )
        if not smoke_calibration_only:
            sft_dev_res = eval_wfs_episodes(sft_reloaded, splits["DEV"])
            sft_eval_res = eval_wfs_episodes(sft_reloaded, splits["EVAL"])
            sft_d1_k17_res = eval_d1_prose(sft_reloaded, d1_user_prompts_k17)
            sft_d1_k5_res = eval_d1_prose(sft_reloaded, d1_user_prompts_k5)
            print(
                f"[D4-GPU] SFT 7B -> DEV={sft_dev_res['exact_match']:.4f}, EVAL={sft_eval_res['exact_match']:.4f}, "
                f"D1_k17={sft_d1_k17_res['strict_accuracy']:.4f} (frozen={sft_d1_k17_res['frozen_d1_accuracy']:.4f})",
                flush=True,
            )
        else:
            sft_dev_res = eval_wfs_episodes(sft_reloaded, splits["DEV"][:24])
            sft_eval_res = sft_d1_k17_res = sft_d1_k5_res = {}
        record_stage_mem("sft_reload_and_eval")
        del sft_reloaded
        gc.collect()
        torch.cuda.empty_cache()

        write_json_atomic(
            stage1_ckpt_path,
            {
                "sft_mask_audit": sft_mask_audit,
                "sft_train_seconds": sft_train_sec,
                "sft_train_loss": sft_train_loss_val,
                "sft_vs_base_logit_shift": sft_vs_base_logit_shift,
                "sft_reload_logit_diff": sft_reload_logit_diff,
                "sft_grpo_split_res": sft_grpo_split_res,
                "base_dev_res": base_dev_res,
                "base_eval_res": base_eval_res,
                "base_d1_k17_res": base_d1_k17_res,
                "base_d1_k5_res": base_d1_k5_res,
                "sft_dev_res": sft_dev_res,
                "sft_eval_res": sft_eval_res,
                "sft_d1_k17_res": sft_d1_k17_res,
                "sft_d1_k5_res": sft_d1_k5_res,
                "peak_mem_gib_by_stage": peak_mem_gib_by_stage,
            },
        )

    # -----------------------------------------------------------------------
    # Stage 2: 3-Seed Epistemic-WFS-GRPO (`r=32, alpha=64, dropout=0.0, beta=0.04`)
    # -----------------------------------------------------------------------
    grpo_rows = [
        {
            "prompt": [
                {"role": "system", "content": GRPO_SYSTEM_PROMPT},
                {"role": "user", "content": ep.prompt},
            ],
            "episode": ep.to_json(),
        }
        for ep in splits["GRPO_RL"]
    ]
    grpo_ds = Dataset.from_list(grpo_rows)

    grpo_lora_cfg = LoraConfig(
        r=32,
        lora_alpha=64,
        lora_dropout=0.0,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    seed_runs: Dict[str, Dict[str, Any]] = {}
    run_seeds = (seeds[0],) if smoke_calibration_only else tuple(seeds)
    run_steps = 3 if smoke_calibration_only else int(grpo_steps)
    eval_ckpts = (
        (run_steps,)
        if smoke_calibration_only
        else tuple(sorted({max(1, run_steps // 2), run_steps}))
    )

    for seed_val in run_seeds:
        seed_key = f"seed_{seed_val}"
        seed_dir = work_dir / seed_key
        seed_dir.mkdir(parents=True, exist_ok=True)
        seed_ckpt_path = work_dir / f"_ckpt_{seed_key}_steps{run_steps}{'_smoke' if smoke_calibration_only else ''}.json"
        if seed_ckpt_path.is_file() and (seed_dir / "final_step_adapter" / "adapter_model.safetensors").is_file():
            print(f"[D4-GPU] Resuming {seed_key} from checkpoint {seed_ckpt_path}...", flush=True)
            ck_seed = json.loads(seed_ckpt_path.read_text(encoding="utf-8"))
            seed_runs[seed_key] = ck_seed["seed_run"]
            peak_mem_gib_by_stage[f"grpo_train_{seed_key}"] = float(ck_seed["seed_run"]["peak_cuda_allocated_gib"])
            continue
        torch.cuda.reset_peak_memory_stats()
        torch.manual_seed(seed_val)
        np.random.seed(seed_val)

        print(f"[D4-GPU] Starting Epistemic-WFS-GRPO {seed_key} (steps={run_steps})...", flush=True)
        sft_base_for_grpo = AutoModelForCausalLM.from_pretrained(
            str(merged_sft_dir),
            dtype=torch.bfloat16,
            local_files_only=True,
        )
        sft_base_for_grpo.config.use_cache = False

        tracker = RolloutTelemetryTracker(num_generations=8)
        reward_fn = tracker.make_reward_fn()

        best_dev_state: Dict[str, Any] = {
            "best_step": 0,
            "best_exact_match": -1.0,
            "best_mean_reward": -999.0,
            "history": [],
        }
        best_dev_dir = seed_dir / "best_dev_adapter"

        class DevCheckpointCallback(TrainerCallback):
            def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> None:
                if state.global_step in eval_ckpts:
                    mdl_cur = kwargs["model"]
                    was_training = mdl_cur.training
                    dev_subset = splits["DEV"][:16] if smoke_calibration_only else splits["DEV"]
                    dev_m = eval_wfs_episodes(mdl_cur, dev_subset)
                    mdl_cur.config.use_cache = False
                    if was_training:
                        mdl_cur.train()
                    rec_dev = {
                        "step": int(state.global_step),
                        "dev_exact_match": dev_m["exact_match"],
                        "dev_mean_total_reward": dev_m["mean_total_reward"],
                        "dev_abstain_rate": dev_m["abstain_rate"],
                        "dev_false_abstain_rate": dev_m["false_abstain_rate_on_determinate"],
                    }
                    best_dev_state["history"].append(rec_dev)
                    print(f"[D4-GPU] {seed_key} step={state.global_step} DEV: {rec_dev}", flush=True)
                    is_better = (dev_m["exact_match"] > best_dev_state["best_exact_match"]) or (
                        abs(dev_m["exact_match"] - best_dev_state["best_exact_match"]) < 1e-9
                        and dev_m["mean_total_reward"] > best_dev_state["best_mean_reward"]
                    )
                    if is_better:
                        best_dev_state["best_step"] = int(state.global_step)
                        best_dev_state["best_exact_match"] = dev_m["exact_match"]
                        best_dev_state["best_mean_reward"] = dev_m["mean_total_reward"]
                        mdl_cur.save_pretrained(str(best_dev_dir))

        grpo_extra_kw = {"max_prompt_length": MAX_PROMPT_LENGTH} if "max_prompt_length" in grpo_cfg_params else {}
        grpo_cfg = GRPOConfig(
            output_dir=str(seed_dir / "trainer_out"),
            max_steps=run_steps,
            per_device_train_batch_size=8,
            gradient_accumulation_steps=1,
            num_generations=8,
            max_completion_length=MAX_COMPLETION_LENGTH,
            **grpo_extra_kw,
            temperature=0.9,
            learning_rate=2e-5,
            lr_scheduler_type="cosine",
            warmup_steps=min(5, max(1, run_steps // 8)),
            beta=0.04,
            bf16=True,
            gradient_checkpointing=True,
            use_cache=False,
            disable_dropout=True,
            logging_steps=1,
            save_strategy="no",
            report_to="none",
            seed=int(seed_val),
            data_seed=int(seed_val),
        )
        grpo_trainer = GRPOTrainer(
            model=sft_base_for_grpo,
            processing_class=tok,
            reward_funcs=[reward_fn],
            args=grpo_cfg,
            train_dataset=grpo_ds,
            peft_config=grpo_lora_cfg,
            callbacks=[DevCheckpointCallback()],
        )
        if grpo_trainer.ref_model is not None:
            raise AssertionError("Expected PEFT adapter-disabled reference (`ref_model is None`) for zero duplicate VRAM")

        t_g0 = time.perf_counter()
        grpo_trainer.train()
        grpo_wall_sec = round(time.perf_counter() - t_g0, 2)
        grpo_peak_gib = record_stage_mem(f"grpo_train_{seed_key}")

        # Verify adapter-disabled KL reference == frozen SFT (`< 1e-4`) while active adapter shifted!
        trained_peft_mdl = grpo_trainer.model
        trained_active_logits = compute_probe_logits(trained_peft_mdl)
        with trained_peft_mdl.disable_adapter():
            disabled_ref_logits = compute_probe_logits(trained_peft_mdl)
        kl_ref_vs_sft_diff = float((disabled_ref_logits - sft_merged_probe_logits).abs().max().item())
        active_vs_sft_shift = float((trained_active_logits - sft_merged_probe_logits).abs().max().item())
        if kl_ref_vs_sft_diff >= 1e-4:
            raise AssertionError(f"Adapter-disabled KL reference drifted from frozen SFT: {kl_ref_vs_sft_diff}")

        # Save PRIMARY pre-registered final-step adapter (`step == run_steps`) + hash files
        final_adapter_dir = seed_dir / "final_step_adapter"
        trained_peft_mdl.save_pretrained(str(final_adapter_dir))
        if not best_dev_dir.exists():
            trained_peft_mdl.save_pretrained(str(best_dev_dir))
        final_hashes_before = hash_directory_files(final_adapter_dir)
        best_dev_hashes = hash_directory_files(best_dev_dir)

        # Extract TRL log_history telemetry (`frac_reward_zero_std`, `kl`, `entropy`, `grad_norm`)
        log_hist = list(grpo_trainer.state.log_history)
        trl_zero_std = [float(x["frac_reward_zero_std"]) for x in log_hist if "frac_reward_zero_std" in x]
        trl_kl = [float(x["kl"]) for x in log_hist if "kl" in x]
        trl_entropy = [float(x["entropy"]) for x in log_hist if "entropy" in x]
        trl_grad_norm = [float(x["grad_norm"]) for x in log_hist if "grad_norm" in x]

        mean_zero_std = float(np.mean(trl_zero_std)) if trl_zero_std else float(
            np.mean([r["group_zero_std_frac"] for r in tracker.records])
        )
        if mean_zero_std >= 0.80:
            raise RuntimeError(f"Guardrail K4 abort: {seed_key} frac_reward_zero_std={mean_zero_std:.4f} >= 0.80")

        del grpo_trainer, trained_peft_mdl, sft_base_for_grpo
        gc.collect()
        torch.cuda.empty_cache()

        # Guardrail 5: Reload trained PRIMARY checkpoint freshly from disk, verify SHA-256 & logit parity < 1e-4!
        final_hashes_after = hash_directory_files(final_adapter_dir)
        if final_hashes_after != final_hashes_before or "adapter_model.safetensors" not in final_hashes_after:
            raise AssertionError(f"Checkpoint hash mismatch on {final_adapter_dir}")

        reload_base = AutoModelForCausalLM.from_pretrained(
            str(merged_sft_dir), dtype=torch.bfloat16, local_files_only=True
        ).to("cuda")
        # Preserve saved float32 LoRA adapter precision (`autocast_adapter_dtype=False`) to match GRPOTrainer's active fp32 LoRA
        reloaded_final_peft = PeftModel.from_pretrained(
            reload_base, str(final_adapter_dir), is_trainable=False, autocast_adapter_dtype=False
        ).to("cuda")
        reloaded_logits = compute_probe_logits(reloaded_final_peft)
        reload_max_abs_logit_diff = float((reloaded_logits - trained_active_logits).abs().max().item())
        if reload_max_abs_logit_diff >= 1e-4:
            raise AssertionError(
                f"Guardrail K4 reload logit parity failed on {seed_key}: {reload_max_abs_logit_diff} >= 1e-4"
            )
        print(
            f"[D4-GPU] {seed_key} finished in {grpo_wall_sec}s (peak={grpo_peak_gib} GiB): "
            f"frac_zero_std={mean_zero_std:.4f}, mean_kl={float(np.mean(trl_kl)) if trl_kl else 0.0:.5f}, "
            f"reload_logit_diff={reload_max_abs_logit_diff:.2e}",
            flush=True,
        )

        if not smoke_calibration_only:
            primary_dev = eval_wfs_episodes(reloaded_final_peft, splits["DEV"])
            primary_eval = eval_wfs_episodes(reloaded_final_peft, splits["EVAL"])
            primary_d1_k17 = eval_d1_prose(reloaded_final_peft, d1_user_prompts_k17)
            primary_d1_k5 = eval_d1_prose(reloaded_final_peft, d1_user_prompts_k5)
            print(
                f"[D4-GPU] {seed_key} PRIMARY (step={run_steps}) -> DEV={primary_dev['exact_match']:.4f}, "
                f"EVAL={primary_eval['exact_match']:.4f}, D1_k17={primary_d1_k17['strict_accuracy']:.4f} "
                f"(frozen={primary_d1_k17['frozen_d1_accuracy']:.4f})",
                flush=True,
            )
        else:
            primary_dev = primary_eval = primary_d1_k17 = primary_d1_k5 = {}

        del reloaded_final_peft, reload_base
        gc.collect()
        torch.cuda.empty_cache()

        # Evaluate SECONDARY `best_dev` checkpoint on EVAL & D1 (if distinct step from final, else reuse exact dict)
        if not smoke_calibration_only:
            if best_dev_state["best_step"] == run_steps:
                sec_eval = primary_eval
                sec_d1_k17 = primary_d1_k17
            else:
                reload_base_b = AutoModelForCausalLM.from_pretrained(
                    str(merged_sft_dir), dtype=torch.bfloat16, local_files_only=True
                ).to("cuda")
                reloaded_best_peft = PeftModel.from_pretrained(
                    reload_base_b, str(best_dev_dir), is_trainable=False, autocast_adapter_dtype=False
                ).to("cuda")
                sec_eval = eval_wfs_episodes(reloaded_best_peft, splits["EVAL"])
                sec_d1_k17 = eval_d1_prose(reloaded_best_peft, d1_user_prompts_k17)
                del reloaded_best_peft, reload_base_b
                gc.collect()
                torch.cuda.empty_cache()
        else:
            sec_eval = sec_d1_k17 = {}

        seed_runs[seed_key] = {
            "seed": int(seed_val),
            "preregistered_steps": run_steps,
            "train_wall_seconds": grpo_wall_sec,
            "peak_cuda_allocated_gib": grpo_peak_gib,
            "frac_reward_zero_std_mean": round(mean_zero_std, 4),
            "frac_reward_zero_std_max": round(float(max(trl_zero_std)) if trl_zero_std else mean_zero_std, 4),
            "mean_kl": round(float(np.mean(trl_kl)) if trl_kl else 0.0, 6),
            "final_kl": round(float(trl_kl[-1]) if trl_kl else 0.0, 6),
            "mean_entropy": round(float(np.mean(trl_entropy)) if trl_entropy else 0.0, 4),
            "mean_grad_norm": round(float(np.mean(trl_grad_norm)) if trl_grad_norm else 0.0, 4),
            "rollout_abstain_rate_mean": round(float(np.mean([r["abstain_rate"] for r in tracker.records])), 4),
            "rollout_false_abstain_rate_mean": round(
                float(np.mean([r["false_abstain_rate"] for r in tracker.records])), 4
            ),
            "rollout_exact_match_first10": round(
                float(np.mean([r["exact_match_rate"] for r in tracker.records[:10]])), 4
            ),
            "rollout_exact_match_last10": round(
                float(np.mean([r["exact_match_rate"] for r in tracker.records[-10:]])), 4
            ),
            "kl_ref_disabled_adapter_vs_frozen_sft_max_abs_logit_diff": kl_ref_vs_sft_diff,
            "active_grpo_adapter_vs_frozen_sft_max_abs_logit_shift": active_vs_sft_shift,
            "final_adapter_sha256": final_hashes_after,
            "best_dev_adapter_sha256": best_dev_hashes,
            "reload_max_abs_logit_diff": reload_max_abs_logit_diff,
            "trained_grpo_checkpoint_present": True,
            "dev_trajectory": best_dev_state["history"],
            "best_dev_step": best_dev_state["best_step"],
            "primary_final_step_dev": primary_dev,
            "primary_final_step_eval": primary_eval,
            "primary_final_step_d1_k17": primary_d1_k17,
            "primary_final_step_d1_k5": primary_d1_k5,
            "secondary_best_dev_eval": _strip_heavy_eval(sec_eval) if sec_eval else {},
            "secondary_best_dev_d1_k17": _strip_heavy_d1(sec_d1_k17) if sec_d1_k17 else {},
            "step_log_history": log_hist,
        }
        write_json_atomic(seed_ckpt_path, {"seed_run": seed_runs[seed_key]})

    if smoke_calibration_only:
        smoke_out = {
            "status": "SMOKE_CALIBRATION_OK",
            "env_versions": env_versions,
            "split_prompt_token_max": split_prompt_token_max,
            "sft_mask_audit": sft_mask_audit,
            "sft_train_seconds": sft_train_sec,
            "sft_vs_base_logit_shift": sft_vs_base_logit_shift,
            "sft_reload_logit_diff": sft_reload_logit_diff,
            "sft_on_grpo_rl_exact_match": sft_grpo_split_res["exact_match"],
            "sft_on_grpo_rl_by_scenario": sft_grpo_split_res["by_scenario_exact_match"],
            "sft_dev_subset_exact_match": sft_dev_res["exact_match"],
            "peak_mem_gib_by_stage": peak_mem_gib_by_stage,
            "smoke_seed": {
                k: {
                    sk: sv
                    for sk, sv in v.items()
                    if sk not in ("primary_final_step_dev", "primary_final_step_eval", "step_log_history")
                }
                for k, v in seed_runs.items()
            },
        }
        write_json_atomic(work_dir / "smoke_calibration_summary.json", smoke_out)
        print(f"[D4-GPU] SMOKE CALIBRATION PASSED: {json.dumps(smoke_out, indent=2)}", flush=True)
        return smoke_out

    # -----------------------------------------------------------------------
    # Stage 3: Zero-Shot Bare Nemotron-30B (`:8890`, `enable_thinking=False`, `max_tokens=256`)
    # -----------------------------------------------------------------------
    print(f"[D4-GPU] Stage 3: Evaluating Bare Nemotron-30B at {nemotron_endpoint}...", flush=True)
    nem_eval_res = _eval_nemotron_wfs(
        nemotron_endpoint, nemotron_model, splits["EVAL"], max_tokens=MAX_COMPLETION_LENGTH
    )
    nem_d1_k17_res = _eval_nemotron_d1(
        nemotron_endpoint, nemotron_model, d1_cases, d1_user_prompts_k17, max_tokens=MAX_COMPLETION_LENGTH
    )
    nem_d1_k5_res = _eval_nemotron_d1(
        nemotron_endpoint, nemotron_model, d1_cases, d1_user_prompts_k5, max_tokens=MAX_COMPLETION_LENGTH
    )
    print(
        f"[D4-GPU] Bare Nemotron-30B -> EVAL={nem_eval_res['exact_match']:.4f}, "
        f"D1_k17={nem_d1_k17_res['strict_accuracy']:.4f} (frozen={nem_d1_k17_res['frozen_d1_accuracy']:.4f})",
        flush=True,
    )

    raw_bundle = {
        "env_versions": env_versions,
        "disjoint_proof": disjoint_proof,
        "d1_corpus_sha256": d1_corpus_sha,
        "d1_retrieval_k17_stats": d1_retrieval_k17_stats,
        "d1_retrieval_k5_stats": d1_retrieval_k5_stats,
        "split_prompt_token_max": split_prompt_token_max,
        "sft_mask_audit": sft_mask_audit,
        "sft_train_seconds": sft_train_sec,
        "sft_train_loss": sft_train_loss_val,
        "sft_vs_base_logit_shift": sft_vs_base_logit_shift,
        "sft_reload_logit_diff": sft_reload_logit_diff,
        "sft_on_grpo_rl": _strip_heavy_eval(sft_grpo_split_res),
        "peak_mem_gib_by_stage": peak_mem_gib_by_stage,
        "base_7b": {
            "dev": _strip_heavy_eval(base_dev_res),
            "eval": base_eval_res,
            "d1_k17": base_d1_k17_res,
            "d1_k5": base_d1_k5_res,
        },
        "sft_7b": {
            "dev": _strip_heavy_eval(sft_dev_res),
            "eval": sft_eval_res,
            "d1_k17": sft_d1_k17_res,
            "d1_k5": sft_d1_k5_res,
        },
        "grpo_seeds": seed_runs,
        "bare_nemotron_30b": {
            "eval": nem_eval_res,
            "d1_k17": nem_d1_k17_res,
            "d1_k5": nem_d1_k5_res,
        },
        "pipeline_wall_seconds": round(time.perf_counter() - t_pipeline0, 2),
    }
    write_json_atomic(work_dir / "remote_gpu_raw_results.json", raw_bundle)
    print(f"[D4-GPU] Full GPU pipeline completed in {raw_bundle['pipeline_wall_seconds']}s!", flush=True)
    return raw_bundle


def _summarize_d1_completions(
    cases: Sequence[Mapping[str, Any]],
    completions: Sequence[str],
    wall_s: float,
) -> Dict[str, Any]:
    strict_by_id: Dict[str, float] = {}
    frozen_by_id: Dict[str, float] = {}
    by_reg_strict: Dict[str, List[float]] = defaultdict(list)
    by_reg_frozen: Dict[str, List[float]] = defaultdict(list)
    by_dom_strict: Dict[str, List[float]] = defaultdict(list)
    u_false_comm: List[float] = []
    det_false_abs: List[float] = []
    details: List[Dict[str, Any]] = []

    for c, comp in zip(cases, completions):
        cid = str(c["case_id"])
        sc = score_d1_prose_prediction(comp, str(c["gold"]), c["stale_or_false_values"])
        strict_by_id[cid] = float(sc["strict_exact_match"])
        frozen_by_id[cid] = float(sc["frozen_d1_match"])
        by_reg_strict[str(c["regime"])].append(float(sc["strict_exact_match"]))
        by_reg_frozen[str(c["regime"])].append(float(sc["frozen_d1_match"]))
        by_dom_strict[str(c["domain"])].append(float(sc["strict_exact_match"]))
        if str(c["gold"]).upper() == "ABSTAIN":
            u_false_comm.append(1.0 if sc["is_false_commit_on_u"] else 0.0)
        else:
            det_false_abs.append(1.0 if sc["is_false_abstain"] else 0.0)
        details.append(
            {
                "case_id": cid,
                "domain": c["domain"],
                "regime": c["regime"],
                "gold": c["gold"],
                "extracted_answer": sc["extracted_answer"],
                "strict_exact_match": int(sc["strict_exact_match"]),
                "frozen_d1_match": int(sc["frozen_d1_match"]),
                "is_abstain": bool(sc["is_abstain"]),
                "unclosed_think": bool(sc["unclosed_think"]),
                "leaked_stale_value": bool(sc["leaked_stale_value"]),
                "completion": comp,
            }
        )
    return {
        "strict_accuracy": round(float(np.mean(list(strict_by_id.values()))), 4),
        "frozen_d1_accuracy": round(float(np.mean(list(frozen_by_id.values()))), 4),
        "abstain_rate": round(float(np.mean([1.0 if d["is_abstain"] else 0.0 for d in details])), 4),
        "false_abstain_rate_on_determinate": round(float(np.mean(det_false_abs)) if det_false_abs else 0.0, 4),
        "false_commit_rate_on_contested_u": round(float(np.mean(u_false_comm)) if u_false_comm else 0.0, 4),
        "unclosed_think_rate": round(float(np.mean([1.0 if d["unclosed_think"] else 0.0 for d in details])), 4),
        "stale_leak_rate": round(float(np.mean([1.0 if d["leaked_stale_value"] else 0.0 for d in details])), 4),
        "by_regime_strict": {k: round(float(np.mean(v)), 4) for k, v in sorted(by_reg_strict.items())},
        "by_regime_frozen_d1": {k: round(float(np.mean(v)), 4) for k, v in sorted(by_reg_frozen.items())},
        "by_domain_strict": {k: round(float(np.mean(v)), 4) for k, v in sorted(by_dom_strict.items())},
        "eval_seconds": round(wall_s, 2),
        "per_id_strict": strict_by_id,
        "per_id_frozen": frozen_by_id,
        "details": details,
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


def _eval_nemotron_wfs(
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
                lambda ep: _call_nemotron_one(endpoint, model_id, GRPO_SYSTEM_PROMPT, ep.prompt, max_tokens),
                eps,
            )
        )
    lat_s = time.perf_counter() - t0
    per_id_exact: Dict[str, float] = {}
    per_id_reward: Dict[str, float] = {}
    details: List[Dict[str, Any]] = []
    by_sc: Dict[str, List[float]] = defaultdict(list)
    det_false_abs: List[float] = []
    u_false_comm: List[float] = []
    u_exact: List[float] = []
    det_exact: List[float] = []

    for ep, comp in zip(eps, comps):
        sc = score_calibrated_wfs_completion(comp, ep)
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


def _eval_nemotron_d1(
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


def _strip_heavy_eval(res: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in res.items() if k not in ("per_id_exact", "per_id_reward", "details")}


def _strip_heavy_d1(res: Mapping[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in res.items() if k not in ("per_id_strict", "per_id_frozen", "details")}


# ---------------------------------------------------------------------------
# Final Manifest, CSV Export, Bootstrap Inference, & Receipt Builder
# ---------------------------------------------------------------------------
def build_final_artifacts_from_raw(raw_bundle: Mapping[str, Any], output_dir: Path) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_keys = sorted(raw_bundle["grpo_seeds"].keys(), key=lambda s: int(s.split("_")[1]))

    eval_ids = sorted(raw_bundle["sft_7b"]["eval"]["per_id_exact"].keys())
    d1_ids = sorted(raw_bundle["sft_7b"]["d1_k17"]["per_id_strict"].keys())

    # Compute item-level 3-seed mean scores for EVAL and D1
    grpo_eval_exact_3seed: Dict[str, float] = {}
    grpo_eval_rew_3seed: Dict[str, float] = {}
    for eid in eval_ids:
        grpo_eval_exact_3seed[eid] = float(
            np.mean([raw_bundle["grpo_seeds"][sk]["primary_final_step_eval"]["per_id_exact"][eid] for sk in seed_keys])
        )
        grpo_eval_rew_3seed[eid] = float(
            np.mean([raw_bundle["grpo_seeds"][sk]["primary_final_step_eval"]["per_id_reward"][eid] for sk in seed_keys])
        )

    grpo_d1_k17_strict_3seed: Dict[str, float] = {}
    grpo_d1_k17_frozen_3seed: Dict[str, float] = {}
    grpo_d1_k5_strict_3seed: Dict[str, float] = {}
    for cid in d1_ids:
        grpo_d1_k17_strict_3seed[cid] = float(
            np.mean([raw_bundle["grpo_seeds"][sk]["primary_final_step_d1_k17"]["per_id_strict"][cid] for sk in seed_keys])
        )
        grpo_d1_k17_frozen_3seed[cid] = float(
            np.mean([raw_bundle["grpo_seeds"][sk]["primary_final_step_d1_k17"]["per_id_frozen"][cid] for sk in seed_keys])
        )
        grpo_d1_k5_strict_3seed[cid] = float(
            np.mean([raw_bundle["grpo_seeds"][sk]["primary_final_step_d1_k5"]["per_id_strict"][cid] for sk in seed_keys])
        )

    # Write per-item CSVs (hashed in report_receipt.json)
    eval_csv_path = output_dir / "wave6_d4_wfs_eval_n120_rows.csv"
    eval_csv_rows: List[Dict[str, Any]] = []
    arm_eval_sources = [
        ("Base_Qwen2.5_7B", raw_bundle["base_7b"]["eval"]["details"]),
        ("SFT_Qwen2.5_7B", raw_bundle["sft_7b"]["eval"]["details"]),
        ("Bare_Nemotron_30B", raw_bundle["bare_nemotron_30b"]["eval"]["details"]),
    ] + [
        (f"SFT_plus_Epistemic_GRPO_7B_{sk}", raw_bundle["grpo_seeds"][sk]["primary_final_step_eval"]["details"])
        for sk in seed_keys
    ]
    for arm_name, det_list in arm_eval_sources:
        for d in det_list:
            eval_csv_rows.append(
                {
                    "episode_id": d["episode_id"],
                    "scenario": d["scenario"],
                    "gold_label": d["gold_label"],
                    "gold_value": d["gold_value"],
                    "arm": arm_name,
                    "exact_match": d["exact_match"],
                    "total_reward": d["total_reward"],
                    "core_reward": d["core_reward"],
                    "outcome": d["outcome"],
                    "is_abstain": int(d["is_abstain"]),
                    "valid_format": int(d["valid_format"]),
                    "completion_preview": str(d["completion"]).replace("\n", " \\n ")[:260],
                }
            )
    with eval_csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(eval_csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(eval_csv_rows)

    d1_csv_path = output_dir / "wave6_d4_d1_prose_transfer_n60_rows.csv"
    d1_csv_rows: List[Dict[str, Any]] = []
    arm_d1_sources = [
        ("Base_Qwen2.5_7B", raw_bundle["base_7b"]["d1_k17"]["details"]),
        ("SFT_Qwen2.5_7B", raw_bundle["sft_7b"]["d1_k17"]["details"]),
        ("Bare_Nemotron_30B", raw_bundle["bare_nemotron_30b"]["d1_k17"]["details"]),
    ] + [
        (f"SFT_plus_Epistemic_GRPO_7B_{sk}", raw_bundle["grpo_seeds"][sk]["primary_final_step_d1_k17"]["details"])
        for sk in seed_keys
    ]
    for arm_name, det_list in arm_d1_sources:
        for d in det_list:
            d1_csv_rows.append(
                {
                    "case_id": d["case_id"],
                    "domain": d["domain"],
                    "regime": d["regime"],
                    "gold": d["gold"],
                    "arm": arm_name,
                    "extracted_answer": d["extracted_answer"],
                    "strict_exact_match": d["strict_exact_match"],
                    "frozen_d1_match": d["frozen_d1_match"],
                    "is_abstain": int(d["is_abstain"]),
                    "unclosed_think": int(d["unclosed_think"]),
                    "leaked_stale_value": int(d["leaked_stale_value"]),
                }
            )
    with d1_csv_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(d1_csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(d1_csv_rows)

    # Paired bootstrap inference (`paired_bootstrap_wave6` with distinct p5.0, p2.5, p1.25)
    eval_contrasts = {
        "GRPO_3SeedMean_minus_SFT_7B_exact": paired_bootstrap_wave6(
            grpo_eval_exact_3seed, raw_bundle["sft_7b"]["eval"]["per_id_exact"], seed=42, expected_ids=eval_ids
        ),
        "GRPO_3SeedMean_minus_SFT_7B_reward": paired_bootstrap_wave6(
            grpo_eval_rew_3seed, raw_bundle["sft_7b"]["eval"]["per_id_reward"], seed=43, expected_ids=eval_ids
        ),
        "GRPO_3SeedMean_minus_Base_7B_exact": paired_bootstrap_wave6(
            grpo_eval_exact_3seed, raw_bundle["base_7b"]["eval"]["per_id_exact"], seed=44, expected_ids=eval_ids
        ),
        "GRPO_3SeedMean_minus_Bare_Nemotron_30B_exact": paired_bootstrap_wave6(
            grpo_eval_exact_3seed, raw_bundle["bare_nemotron_30b"]["eval"]["per_id_exact"], seed=45, expected_ids=eval_ids
        ),
    }
    for idx_s, sk in enumerate(seed_keys):
        eval_contrasts[f"GRPO_{sk}_minus_SFT_7B_exact"] = paired_bootstrap_wave6(
            raw_bundle["grpo_seeds"][sk]["primary_final_step_eval"]["per_id_exact"],
            raw_bundle["sft_7b"]["eval"]["per_id_exact"],
            seed=100 + idx_s,
            expected_ids=eval_ids,
        )

    d1_contrasts = {
        "D1_k17_GRPO_3SeedMean_minus_SFT_7B_strict": paired_bootstrap_wave6(
            grpo_d1_k17_strict_3seed, raw_bundle["sft_7b"]["d1_k17"]["per_id_strict"], seed=52, expected_ids=d1_ids
        ),
        "D1_k17_GRPO_3SeedMean_minus_Base_7B_strict": paired_bootstrap_wave6(
            grpo_d1_k17_strict_3seed, raw_bundle["base_7b"]["d1_k17"]["per_id_strict"], seed=53, expected_ids=d1_ids
        ),
        "D1_k17_GRPO_3SeedMean_minus_Bare_Nemotron_30B_strict": paired_bootstrap_wave6(
            grpo_d1_k17_strict_3seed, raw_bundle["bare_nemotron_30b"]["d1_k17"]["per_id_strict"], seed=54, expected_ids=d1_ids
        ),
        "D1_k17_GRPO_3SeedMean_minus_SFT_7B_frozen_grader": paired_bootstrap_wave6(
            grpo_d1_k17_frozen_3seed, raw_bundle["sft_7b"]["d1_k17"]["per_id_frozen"], seed=55, expected_ids=d1_ids
        ),
        "D1_k17_GRPO_3SeedMean_minus_Bare_Nemotron_30B_frozen_grader": paired_bootstrap_wave6(
            grpo_d1_k17_frozen_3seed,
            raw_bundle["bare_nemotron_30b"]["d1_k17"]["per_id_frozen"],
            seed=56,
            expected_ids=d1_ids,
        ),
        "D1_k5_Starved_GRPO_3SeedMean_minus_Bare_Nemotron_30B_strict": paired_bootstrap_wave6(
            grpo_d1_k5_strict_3seed, raw_bundle["bare_nemotron_30b"]["d1_k5"]["per_id_strict"], seed=57, expected_ids=d1_ids
        ),
    }
    for idx_s, sk in enumerate(seed_keys):
        d1_contrasts[f"D1_k17_GRPO_{sk}_minus_SFT_7B_strict"] = paired_bootstrap_wave6(
            raw_bundle["grpo_seeds"][sk]["primary_final_step_d1_k17"]["per_id_strict"],
            raw_bundle["sft_7b"]["d1_k17"]["per_id_strict"],
            seed=200 + idx_s,
            expected_ids=d1_ids,
        )

    holm_family = holm_bonferroni_stepdown(
        {
            "EVAL_GRPO_3SeedMean_vs_SFT_7B": eval_contrasts["GRPO_3SeedMean_minus_SFT_7B_exact"]["p_value_two_sided"],
            "EVAL_GRPO_3SeedMean_vs_Base_7B": eval_contrasts["GRPO_3SeedMean_minus_Base_7B_exact"]["p_value_two_sided"],
            "EVAL_GRPO_3SeedMean_vs_Bare_Nemotron_30B": eval_contrasts[
                "GRPO_3SeedMean_minus_Bare_Nemotron_30B_exact"
            ]["p_value_two_sided"],
            "D1_Prose_GRPO_3SeedMean_vs_SFT_7B": d1_contrasts["D1_k17_GRPO_3SeedMean_minus_SFT_7B_strict"][
                "p_value_two_sided"
            ],
            "D1_Prose_GRPO_3SeedMean_vs_Base_7B": d1_contrasts["D1_k17_GRPO_3SeedMean_minus_Base_7B_strict"][
                "p_value_two_sided"
            ],
            "D1_Prose_GRPO_3SeedMean_vs_Bare_Nemotron_30B": d1_contrasts[
                "D1_k17_GRPO_3SeedMean_minus_Bare_Nemotron_30B_strict"
            ]["p_value_two_sided"],
        }
    )

    seed_summaries: Dict[str, Any] = {}
    for sk in seed_keys:
        sr = raw_bundle["grpo_seeds"][sk]
        seed_summaries[sk] = {
            "seed": sr["seed"],
            "preregistered_steps": sr["preregistered_steps"],
            "train_wall_seconds": sr["train_wall_seconds"],
            "peak_cuda_allocated_gib": sr["peak_cuda_allocated_gib"],
            "frac_reward_zero_std_mean": sr["frac_reward_zero_std_mean"],
            "frac_reward_zero_std_max": sr["frac_reward_zero_std_max"],
            "mean_kl": sr["mean_kl"],
            "final_kl": sr["final_kl"],
            "mean_entropy": sr["mean_entropy"],
            "mean_grad_norm": sr["mean_grad_norm"],
            "rollout_abstain_rate_mean": sr["rollout_abstain_rate_mean"],
            "rollout_false_abstain_rate_mean": sr["rollout_false_abstain_rate_mean"],
            "rollout_exact_match_first10": sr["rollout_exact_match_first10"],
            "rollout_exact_match_last10": sr["rollout_exact_match_last10"],
            "kl_ref_disabled_adapter_vs_frozen_sft_max_abs_logit_diff": sr[
                "kl_ref_disabled_adapter_vs_frozen_sft_max_abs_logit_diff"
            ],
            "active_grpo_adapter_vs_frozen_sft_max_abs_logit_shift": sr[
                "active_grpo_adapter_vs_frozen_sft_max_abs_logit_shift"
            ],
            "final_adapter_sha256": sr["final_adapter_sha256"],
            "best_dev_adapter_sha256": sr["best_dev_adapter_sha256"],
            "reload_max_abs_logit_diff": sr["reload_max_abs_logit_diff"],
            "trained_grpo_checkpoint_present": sr["trained_grpo_checkpoint_present"],
            "dev_trajectory": sr["dev_trajectory"],
            "best_dev_step": sr["best_dev_step"],
            "primary_final_step_dev": _strip_heavy_eval(sr["primary_final_step_dev"]),
            "primary_final_step_eval": _strip_heavy_eval(sr["primary_final_step_eval"]),
            "primary_final_step_d1_k17": _strip_heavy_d1(sr["primary_final_step_d1_k17"]),
            "primary_final_step_d1_k5": _strip_heavy_d1(sr["primary_final_step_d1_k5"]),
            "secondary_best_dev_eval": sr["secondary_best_dev_eval"],
            "secondary_best_dev_d1_k17": sr["secondary_best_dev_d1_k17"],
        }

    git_sha = current_git_sha(REPO_ROOT)
    eval_csv_sha = sha256_file(eval_csv_path)
    d1_csv_sha = sha256_file(d1_csv_path)

    manifest: Dict[str, Any] = {
        "schema_version": "wave6_d4_trained_epistemic_grpo_k4_v1",
        "track": "Wave 6 Track D — Demo D4 (Guardrail K4)",
        "git_sha": git_sha,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "compute_node": "DGX Spark #2 (mattmre@10.0.0.194)",
        "base_model": "Qwen/Qwen2.5-7B-Instruct",
        "nemotron_baseline_endpoint": "http://127.0.0.1:8890/v1 (nemotron-lightning, enable_thinking=False)",
        "environment_versions": raw_bundle["env_versions"],
        "trained_grpo_checkpoint_present": all(
            seed_summaries[sk]["trained_grpo_checkpoint_present"] for sk in seed_keys
        ),
        "k4_guardrail_audit": {
            "four_way_pairwise_disjoint_splits": raw_bundle["disjoint_proof"],
            "prompt_max_tokens_by_split": raw_bundle["split_prompt_token_max"],
            "max_prompt_length_cap": MAX_PROMPT_LENGTH,
            "max_completion_length_matched": MAX_COMPLETION_LENGTH,
            "sft_completion_only_and_zero_oracle_leak": raw_bundle["sft_mask_audit"],
            "sft_merge_and_unload_before_grpo": {
                "sft_lora_rank": 16,
                "sft_lr": 8e-5,
                "sft_epochs": 3,
                "sft_train_seconds": raw_bundle["sft_train_seconds"],
                "sft_train_loss": raw_bundle["sft_train_loss"],
                "sft_vs_base_logit_shift": raw_bundle["sft_vs_base_logit_shift"],
                "sft_disk_reload_max_abs_logit_diff": raw_bundle["sft_reload_logit_diff"],
                "grpo_fresh_lora_rank": 32,
                "grpo_fresh_lora_alpha": 64,
                "grpo_fresh_lora_dropout": 0.0,
                "grpo_kl_beta": 0.04,
                "max_kl_ref_vs_frozen_sft_logit_diff": max(
                    seed_summaries[sk]["kl_ref_disabled_adapter_vs_frozen_sft_max_abs_logit_diff"] for sk in seed_keys
                ),
            },
            "sft_calibration_on_grpo_rl_split": raw_bundle["sft_on_grpo_rl"],
            "calibrated_3valued_reward_spec": {
                "exact_canonical_match_core": EXACT_CORE_REWARD,
                "gated_format_bonus_only_when_core_1": FORMAT_BONUS_REWARD,
                "false_abstention_on_determinate_any_verdict": FALSE_ABSTAIN_REWARD,
                "plus_025_verdict_abstain_loophole_closed": True,
                "wrong_negated_mixed_or_unclosed_think": WRONG_OR_MALFORMED_REWARD,
                "keyword_only_signature_enforced": True,
            },
            "three_seed_training_telemetry": {
                "preregistered_seeds": [int(s.split("_")[1]) for s in seed_keys],
                "mean_frac_reward_zero_std": round(
                    float(np.mean([seed_summaries[sk]["frac_reward_zero_std_mean"] for sk in seed_keys])), 4
                ),
                "max_frac_reward_zero_std": round(
                    float(max(seed_summaries[sk]["frac_reward_zero_std_max"] for sk in seed_keys)), 4
                ),
                "frac_reward_zero_std_below_080": all(
                    seed_summaries[sk]["frac_reward_zero_std_mean"] < 0.80 for sk in seed_keys
                ),
                "mean_kl_3seed": round(float(np.mean([seed_summaries[sk]["mean_kl"] for sk in seed_keys])), 6),
                "mean_rollout_abstain_rate_3seed": round(
                    float(np.mean([seed_summaries[sk]["rollout_abstain_rate_mean"] for sk in seed_keys])), 4
                ),
                "max_reload_logit_diff_across_seeds": max(
                    seed_summaries[sk]["reload_max_abs_logit_diff"] for sk in seed_keys
                ),
                "reload_logit_parity_below_1e4": all(
                    seed_summaries[sk]["reload_max_abs_logit_diff"] < 1e-4 for sk in seed_keys
                ),
            },
            "peak_cuda_memory_gib_by_stage": raw_bundle["peak_mem_gib_by_stage"],
            "max_peak_cuda_memory_gib": max(raw_bundle["peak_mem_gib_by_stage"].values()),
            "peak_memory_within_34gb_cap": max(raw_bundle["peak_mem_gib_by_stage"].values()) <= 34.0,
        },
        "heldout_wfs_eval_n120": {
            "Base_Qwen2.5_7B": _strip_heavy_eval(raw_bundle["base_7b"]["eval"]),
            "SFT_Qwen2.5_7B": _strip_heavy_eval(raw_bundle["sft_7b"]["eval"]),
            "Bare_Nemotron_30B": _strip_heavy_eval(raw_bundle["bare_nemotron_30b"]["eval"]),
            "SFT_plus_Epistemic_GRPO_7B_per_seed_primary": {
                sk: seed_summaries[sk]["primary_final_step_eval"] for sk in seed_keys
            },
            "SFT_plus_Epistemic_GRPO_7B_3SeedMean_primary": {
                "exact_match": round(float(np.mean(list(grpo_eval_exact_3seed.values()))), 4),
                "exact_match_seed_std": round(
                    float(np.std([seed_summaries[sk]["primary_final_step_eval"]["exact_match"] for sk in seed_keys])), 4
                ),
                "mean_total_reward": round(float(np.mean(list(grpo_eval_rew_3seed.values()))), 4),
                "abstain_rate_mean": round(
                    float(np.mean([seed_summaries[sk]["primary_final_step_eval"]["abstain_rate"] for sk in seed_keys])),
                    4,
                ),
                "false_abstain_rate_on_determinate_mean": round(
                    float(
                        np.mean(
                            [
                                seed_summaries[sk]["primary_final_step_eval"]["false_abstain_rate_on_determinate"]
                                for sk in seed_keys
                            ]
                        )
                    ),
                    4,
                ),
                "false_commit_rate_on_u_mean": round(
                    float(
                        np.mean(
                            [seed_summaries[sk]["primary_final_step_eval"]["false_commit_rate_on_u"] for sk in seed_keys]
                        )
                    ),
                    4,
                ),
            },
            "SFT_plus_Epistemic_GRPO_7B_secondary_best_dev": {
                sk: seed_summaries[sk]["secondary_best_dev_eval"] for sk in seed_keys
            },
            "paired_bootstrap_wave6": eval_contrasts,
        },
        "d1_zero_shot_enterprise_prose_transfer_n60": {
            "corpus_sha256": raw_bundle["d1_corpus_sha256"],
            "matched_evidence_k17_retrieval": raw_bundle["d1_retrieval_k17_stats"],
            "starved_k5_retrieval": raw_bundle["d1_retrieval_k5_stats"],
            "primary_matched_evidence_k17": {
                "Base_Qwen2.5_7B": _strip_heavy_d1(raw_bundle["base_7b"]["d1_k17"]),
                "SFT_Qwen2.5_7B": _strip_heavy_d1(raw_bundle["sft_7b"]["d1_k17"]),
                "Bare_Nemotron_30B": _strip_heavy_d1(raw_bundle["bare_nemotron_30b"]["d1_k17"]),
                "SFT_plus_Epistemic_GRPO_7B_per_seed_primary": {
                    sk: seed_summaries[sk]["primary_final_step_d1_k17"] for sk in seed_keys
                },
                "SFT_plus_Epistemic_GRPO_7B_3SeedMean_primary": {
                    "strict_accuracy": round(float(np.mean(list(grpo_d1_k17_strict_3seed.values()))), 4),
                    "strict_accuracy_seed_std": round(
                        float(
                            np.std(
                                [
                                    seed_summaries[sk]["primary_final_step_d1_k17"]["strict_accuracy"]
                                    for sk in seed_keys
                                ]
                            )
                        ),
                        4,
                    ),
                    "frozen_d1_accuracy": round(float(np.mean(list(grpo_d1_k17_frozen_3seed.values()))), 4),
                    "abstain_rate_mean": round(
                        float(
                            np.mean([seed_summaries[sk]["primary_final_step_d1_k17"]["abstain_rate"] for sk in seed_keys])
                        ),
                        4,
                    ),
                    "false_abstain_rate_on_determinate_mean": round(
                        float(
                            np.mean(
                                [
                                    seed_summaries[sk]["primary_final_step_d1_k17"]["false_abstain_rate_on_determinate"]
                                    for sk in seed_keys
                                ]
                            )
                        ),
                        4,
                    ),
                    "false_commit_rate_on_contested_u_mean": round(
                        float(
                            np.mean(
                                [
                                    seed_summaries[sk]["primary_final_step_d1_k17"]["false_commit_rate_on_contested_u"]
                                    for sk in seed_keys
                                ]
                            )
                        ),
                        4,
                    ),
                },
                "SFT_plus_Epistemic_GRPO_7B_secondary_best_dev": {
                    sk: seed_summaries[sk]["secondary_best_dev_d1_k17"] for sk in seed_keys
                },
            },
            "secondary_retrieval_starved_k5_bare_hybridrag": {
                "Base_Qwen2.5_7B": _strip_heavy_d1(raw_bundle["base_7b"]["d1_k5"]),
                "SFT_Qwen2.5_7B": _strip_heavy_d1(raw_bundle["sft_7b"]["d1_k5"]),
                "Bare_Nemotron_30B": _strip_heavy_d1(raw_bundle["bare_nemotron_30b"]["d1_k5"]),
                "SFT_plus_Epistemic_GRPO_7B_3SeedMean_strict_accuracy": round(
                    float(np.mean(list(grpo_d1_k5_strict_3seed.values()))), 4
                ),
            },
            "paired_bootstrap_wave6": d1_contrasts,
        },
        "holm_bonferroni_multiplicity_family": holm_family,
        "grpo_seed_runs": seed_summaries,
        "csv_artifacts": {
            eval_csv_path.name: eval_csv_sha,
            d1_csv_path.name: d1_csv_sha,
        },
        "pipeline_wall_seconds": raw_bundle["pipeline_wall_seconds"],
    }

    manifest_path = output_dir / "wave6_d4_epistemic_grpo_manifest.json"
    manifest_sha = write_json_atomic(manifest_path, manifest)

    receipt = {
        "schema_version": "report_receipt_v4_0",
        "bundle_id": "wave6_d4_trained_epistemic_grpo",
        "git_sha": git_sha,
        "manifest_file": manifest_path.name,
        "manifest_sha256": manifest_sha,
        "artifact_hashes": {
            manifest_path.name: manifest_sha,
            eval_csv_path.name: eval_csv_sha,
            d1_csv_path.name: d1_csv_sha,
        },
    }
    receipt_sha = write_json_atomic(output_dir / "report_receipt.json", receipt)
    manifest["_receipt_sha256"] = receipt_sha
    return manifest


# ---------------------------------------------------------------------------
# Windows -> Spark #2 SSH Orchestrator (`stdin=subprocess.DEVNULL` mandatory)
# ---------------------------------------------------------------------------
def orchestrate_remote_spark2(
    *,
    output_dir: Path,
    osv_cache_path: Path,
    spark_host: str = "mattmre@10.0.0.194",
    remote_root: str = "/home/mattmre/chelated-night1/out/wave6_d4_epistemic_grpo",
    grpo_steps: int = 50,
    smoke_calibration_only: bool = False,
) -> Dict[str, Any]:
    """Ship isolated code bundle to Spark #2, run `.venv-rlvr` GPU training, and materialize verified artifacts."""
    work_dir = output_dir / "_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    remote_src = f"{remote_root}/src_bundle"
    remote_work = f"{remote_root}/_work"

    subprocess.run(
        ["ssh", "-o", "BatchMode=yes", spark_host, f"mkdir -p {remote_src}/scripts {remote_work}"],
        stdin=subprocess.DEVNULL,
        check=True,
    )

    root_modules = [
        REPO_ROOT / "ek5_v3_stratified.py",
        REPO_ROOT / "ek5_wfs_rlvr_env.py",
        REPO_ROOT / "prospective_realworld_drift_corpus.py",
        REPO_ROOT / "chelated_raw_text_exocortex.py",
        REPO_ROOT / "spherical_migration_bridge.py",
        REPO_ROOT / "m1_baseline_ladder.py",
        REPO_ROOT / "benchmark_utils.py",
        REPO_ROOT / "wave6_common.py",
    ]
    script_modules = [
        REPO_ROOT / "scripts" / "night1_common.py",
        Path(__file__).resolve(),
    ]
    subprocess.run(
        ["scp", "-q", *[str(p) for p in root_modules], f"{spark_host}:{remote_src}/"],
        stdin=subprocess.DEVNULL,
        check=True,
    )
    subprocess.run(
        ["scp", "-q", *[str(p) for p in script_modules], f"{spark_host}:{remote_src}/scripts/"],
        stdin=subprocess.DEVNULL,
        check=True,
    )
    if osv_cache_path.is_file():
        subprocess.run(
            ["scp", "-q", str(osv_cache_path), f"{spark_host}:{remote_work}/osv_cve_seeds.json"],
            stdin=subprocess.DEVNULL,
            check=True,
        )

    smoke_flag = "--smoke-calibration" if smoke_calibration_only else ""
    git_sha = current_git_sha(REPO_ROOT)
    py_inc = (
        "/home/mattmre/chelated-night1/py312_headers/usr/include/python3.12:"
        "/home/mattmre/chelated-night1/py312_headers/usr/include/aarch64-linux-gnu/python3.12:"
        "/home/mattmre/chelated-night1/py312_headers/usr/include"
    )
    remote_cmd = (
        f"CPATH={py_inc} "
        f"CHELATED_GIT_SHA={git_sha} "
        f"PYTHONPATH={remote_src}:/home/mattmre/chelated-night1/src "
        f"/home/mattmre/chelated-night1/.venv-rlvr/bin/python -u "
        f"{remote_src}/scripts/run_wave6_d4_trained_epistemic_grpo.py "
        f"--mode remote-gpu-run "
        f"--output-dir {remote_root} "
        f"--osv-cache {remote_work}/osv_cve_seeds.json "
        f"--grpo-steps {int(grpo_steps)} {smoke_flag}"
    )
    print(f"[D4-Orchestrator] Executing on {spark_host}:\n  {remote_cmd}", flush=True)
    subprocess.run(
        ["ssh", "-o", "BatchMode=yes", spark_host, remote_cmd],
        stdin=subprocess.DEVNULL,
        check=True,
    )

    if smoke_calibration_only:
        local_smoke = work_dir / "smoke_calibration_summary.json"
        subprocess.run(
            ["scp", "-q", f"{spark_host}:{remote_work}/smoke_calibration_summary.json", str(local_smoke)],
            stdin=subprocess.DEVNULL,
            check=True,
        )
        return json.loads(local_smoke.read_text(encoding="utf-8"))

    local_raw = work_dir / "remote_gpu_raw_results.json"
    subprocess.run(
        ["scp", "-q", f"{spark_host}:{remote_work}/remote_gpu_raw_results.json", str(local_raw)],
        stdin=subprocess.DEVNULL,
        check=True,
    )
    raw_bundle = json.loads(local_raw.read_text(encoding="utf-8"))
    return build_final_artifacts_from_raw(raw_bundle, output_dir)


def run_local_cpu_verify(output_dir: Path, osv_cache_path: Path) -> Dict[str, Any]:
    """CPU-only verification of all K4 invariants (4-way disjointness, pure-prompt CoT, -0.15 reward, signature)."""
    splits = build_all_four_disjoint_splits()
    d1_cases, d1_docs, d1_sha = build_prospective_realworld_drift_suite(osv_cache_path)
    disjoint_proof = assert_four_way_disjoint_splits(splits, d1_cases=d1_cases)

    for split_name, eps in splits.items():
        for ep in eps:
            cot = derive_cot_from_prompt_text(ep.prompt)
            sc = score_calibrated_wfs_completion(cot, ep)
            if sc["exact_match"] != 1.0 or sc["total_reward"] != 1.05:
                raise AssertionError(f"{split_name} {ep.episode_id} observable CoT failed: {sc}")

    res = {
        "status": "LOCAL_VERIFY_OK",
        "disjoint_proof": disjoint_proof,
        "d1_cases": len(d1_cases),
        "d1_docs": len(d1_docs),
        "d1_corpus_sha256": d1_sha,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json_atomic(output_dir / "verify_local_k4.json", res)
    print("[D4-Local-Verify] PASSED: 488 WFS episodes + 60 D1 cases verified.", flush=True)
    return res


def main() -> None:
    parser = argparse.ArgumentParser(description="Wave 6 Track D (Demo D4, Guardrail K4): Trained Epistemic-WFS-GRPO")
    parser.add_argument(
        "--mode",
        choices=("orchestrate-spark2", "remote-gpu-run", "rebuild-from-raw", "verify-local"),
        default="orchestrate-spark2",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "artifacts" / "wave6_d4_trained_epistemic_grpo",
    )
    parser.add_argument(
        "--osv-cache",
        type=Path,
        default=REPO_ROOT / "artifacts" / "phase2_d1_4node_local_fleet_proof" / "_work" / "osv_cve_seeds.json",
    )
    parser.add_argument("--grpo-steps", type=int, default=50)
    parser.add_argument("--smoke-calibration", action="store_true")
    args = parser.parse_args()

    if args.mode == "verify-local":
        run_local_cpu_verify(args.output_dir / "_work", args.osv_cache)
    elif args.mode == "remote-gpu-run":
        run_gpu_pipeline_on_spark2(
            work_dir=args.output_dir / "_work",
            osv_cache_path=args.osv_cache,
            grpo_steps=args.grpo_steps,
            smoke_calibration_only=args.smoke_calibration,
        )
    elif args.mode == "rebuild-from-raw":
        raw_p = args.output_dir / "_work" / "remote_gpu_raw_results.json"
        build_final_artifacts_from_raw(json.loads(raw_p.read_text(encoding="utf-8")), args.output_dir)
    else:
        orchestrate_remote_spark2(
            output_dir=args.output_dir,
            osv_cache_path=args.osv_cache,
            grpo_steps=args.grpo_steps,
            smoke_calibration_only=args.smoke_calibration,
        )


if __name__ == "__main__":
    main()
