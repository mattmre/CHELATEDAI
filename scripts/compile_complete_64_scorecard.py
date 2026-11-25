"""Deterministically compile the Brutal-Honesty Rulebook v3.3 64/64 Complete Evidence Scorecard.

Reads:
  1. docs/research/verified_registry.json (Bundles #1-#46, Parts I-VIII, v7.0)
  2. docs/research/registry.d/47..64*.json (Bundles #47-#64, Part IX, Waves 16-21)

Enforces anti-L4 rules:
  - Never conflates design `tier` (T0-T4) or `status=protocol-ready` / verifier exit 0 with live confirmatory evidence.
  - Preserves `evidence_tier` and `evidence_disclosure` verbatim from `registry.d/*.json`.
  - Explicitly surfaces Bundle #46 (PM-6) as `empirically-mixed-r5-fpr-failed`, Bundle #58 as `verified-mixed`
    (scattered -720 B loss vs floor), Bundle #62 live worker run as `FAILED` (`replay-fail`, `overall_pass=false`),
    Bundles #49/#50/#51/#52/#54/#55/#56/#59/#60 as `local-drycheck-pass` (`live_model_calls=0` / synthetic smoke),
    Bundle #53 as observational `live-shadow-pass` (`live_pm6_gateway_executed=false`), and Bundles #63/#64 as
    `protocol-ready` (`DEV_OPEN` offline scorers, 0 partner runs).

Writes:
  - docs/research/COMPLETE_64_BUNDLE_EVIDENCE_SCORECARD.json
  - docs/research/WAVES_1_TO_21_MASTER_EVIDENCE_AND_PR_INDEX.md
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[1]
VERIFIED_REG = REPO_ROOT / "docs/research/verified_registry.json"
REGISTRY_D = REPO_ROOT / "docs/research/registry.d"
OUT_JSON = REPO_ROOT / "docs/research/COMPLETE_64_BUNDLE_EVIDENCE_SCORECARD.json"
OUT_MD = REPO_ROOT / "docs/research/WAVES_1_TO_21_MASTER_EVIDENCE_AND_PR_INDEX.md"


def classify_base_bundle(b: Dict[str, Any]) -> str:
    bid = str(b["id"])
    role = str(b.get("role", ""))
    if bid == "iq01":
        return "falsifier-negative"
    if bid == "f2_longmemeval":
        return "superseded-historical"
    if bid == "s3":
        return "verified-mixed-needs-review"
    if bid == "r5b_s1b_pm2_hybrid_coupled":
        return "verified-partial"
    if bid == "wave6_pm1b_v4_reranked_sota":
        return "verified-scoped-open-gate"
    if bid == "wave11_d9_fast_extractor_and_u_grpo":
        return "verified-mixed-latency-miss-and-grpo-null"
    if bid == "wave11_d10_5m_real_cutover_and_70b_scale":
        return "verified-mixed-with-quora-loss"
    if bid == "wave14_d11_pm6_takedown_carryforward":
        return "verified-mixed-r5-fpr-failed"
    if "DIAGNOSTIC" in role or "PRE_FIX" in role or "BASELINE" in role or "AGGREGATE_ONLY" in role:
        return "diagnostic-or-baseline-sealed"
    return "verified-sealed"


def build_scorecard() -> Dict[str, Any]:
    reg = json.loads(VERIFIED_REG.read_text(encoding="utf-8"))
    base_bundles = reg["bundles"]
    if len(base_bundles) != 46:
        raise RuntimeError(f"Expected 46 base bundles in verified_registry.json, found {len(base_bundles)}")

    rows: List[Dict[str, Any]] = []
    tier_counts: Counter[str] = Counter()

    for idx, b in enumerate(base_bundles, 1):
        ev_class = classify_base_bundle(b)
        tier_counts[ev_class] += 1
        rows.append(
            {
                "ordinal": idx,
                "bundle_id": b["id"],
                "source": "verified_registry_v7_0",
                "part": b["part"],
                "role_or_design_tier": b["role"],
                "registry_status": b["status"],
                "evidence_tier": ev_class,
                "manifest_sha256": b["manifest_sha256"],
                "verified_side_artifacts_count": len(b.get("verified_side_artifacts", {})),
                "summary_and_mandatory_disclosures": b.get("notes", ""),
            }
        )

    frag_files = sorted(REGISTRY_D.glob("*.json"))
    if len(frag_files) != 18:
        raise RuntimeError(f"Expected 18 fragments in {REGISTRY_D}, found {len(frag_files)}")

    for fp in frag_files:
        d = json.loads(fp.read_text(encoding="utf-8"))
        bid = int(d["id"])
        ev_tier = str(d["evidence_tier"])
        disc = d.get("evidence_disclosure", {})
        if bid == 62:
            ev_key = "stage1-replay-pass-live-worker-FAILED"
        else:
            ev_key = ev_tier
        tier_counts[ev_key] += 1
        rows.append(
            {
                "ordinal": bid,
                "bundle_id": f"pv7_{fp.stem}",
                "fragment_id": bid,
                "fragment_file": f"docs/research/registry.d/{fp.name}",
                "source": "pv7_registry_d",
                "part": "PART_IX_POST_V7_ADDITIVE_WAVES_16_TO_21",
                "role_or_design_tier": str(d.get("tier") or "UNSPECIFIED"),
                "registry_status": str(d.get("status")),
                "evidence_tier": ev_key,
                "spine_digest": d.get("spine_digest"),
                "evidence_disclosure": disc,
            }
        )

    if len(rows) != 64:
        raise RuntimeError(f"Expected 64 total rows, got {len(rows)}")

    payload: Dict[str, Any] = {
        "schema_version": "complete_64_bundle_evidence_scorecard_v1",
        "release_status": reg["release_status"],
        "base_verified_registry_sha256": reg["registry_content_sha256"],
        "pv7_spine_digest": rows[-1]["spine_digest"],
        "totals": {
            "total_bundles_retained_additively": 64,
            "base_v7_bundles_1_to_46": 46,
            "post_v7_fragment_bundles_47_to_64": 18,
            "vacated_or_pruned_bundle_ids": 0,
            "by_honest_evidence_classification": dict(sorted(tier_counts.items())),
        },
        "anti_l4_governance_note": (
            "Passing an offline verifier (exit 0) or carrying registry status 'protocol-ready' does NOT make a "
            "Post-v7.0 bundle live confirmatory evidence. Among #47-#64: 3 are live-pass on synthetic/emulated "
            "fixtures (#48 synthetic 12k Gaussian across live Qdrant/OpenSearch/pgvector; #57 synthetic .cspg "
            "cartridges in Spark #3 6GiB cgroup; #61 CPU software emulation with 3/5 live encoders & 2 dry stand-ins), "
            "1 is observational live-shadow-pass (#53, live_pm6_gateway_executed=false), 1 is verified-mixed (#58 "
            "symmetric WAL: clustered +18.9k..+22.5k B saved, scattered 0 B tie / -720 B loss vs floor), 1 is "
            "stage1-replay-pass whose live vLLM worker run FAILED (#62 overall_pass=false, 0/1680 on-time jobs under "
            "560 ms SLA), 10 are local-drycheck-pass synthetic/smoke fixtures (#47, #49, #50, #51, #52, #54 [H6B_merge "
            "6/12 false], #55 [superiority_is_wild_evidence=false], #56, #59, #60), and 2 are protocol-ready open DEV "
            "scorers (#63, #64, 0 partner runs)."
        ),
        "bundles": rows,
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    payload["scorecard_sha256"] = digest
    return payload


def render_markdown(sc: Dict[str, Any]) -> str:
    reg_sha12 = sc["base_verified_registry_sha256"][:12]
    sc_sha12 = sc["scorecard_sha256"][:12]
    lines: List[str] = [
        "# Master Multi-Wave Evidence, Architecture, & Complete `64/64` Honest Scorecard (`Waves 1–21`)",
        "",
        "> **Status (`2026-11-25`):** **`PRE-RELEASE / HUMAN AUDIT & CLEAN-RERUN GATES PENDING`**  ",
        f"> Base `v7.0` Registry (`#1–#46`): `docs/research/verified_registry.json` (`{reg_sha12}`) · Complete `64/64` Machine Scorecard: `docs/research/COMPLETE_64_BUNDLE_EVIDENCE_SCORECARD.json` (`{sc_sha12}`).  ",
        "> Unqualified production release remains **BLOCKED (`PRE-RELEASE`)** pending **`PM-4b`** (`100`-item independent human grading of `pm4_human_audit_100_blind_sheet.csv`), **`PM-1b`** (`assignment_gates_pass = false`), and **`PM-6`** empirical benign false-refusal calibration (`6.25%/11.76% > 2.0%` target). Zero `Ship-Ready` claims are made anywhere in the repository, and per Brutal-Honesty Rulebook v3.3 (`L4` guard), `local-drycheck-pass` / `protocol-ready` / verifier `rc=0` cells are **never** counted as live confirmatory completions.",
        "",
        "---",
        "",
        "## 1. Stacked Branch & Pull Request Architecture (`mattmre/CHELATEDAI`)",
        "",
        "| Stack Layer | PR | Branch | Base Branch | Scope & Honest Verification Summary |",
        "|---|---|---|---|---|",
        "| **Layer 1 (`v6.0`)** | `#364` | `fix/stage0-metric-lineage-and-m1-ladder` (`tag: v6.0-wave13-sealed-20261125`) | `main` | **Parts I–VII (`Waves 1–13`, Bundles `#1–#45`, Registry `01738dbf94e6`):** Stage-0 `mlr-v1` Quarantine (`113/113`), Flagships #1 & #2, 4-Node Fleet (`D1–D5`), & Tier-1/2 Demos (`D6–D10`, including `D9` `2,104 ms > 1,500 ms` & `K4` GRPO null, `D10` `9.47M` `Quora -6.92 pp` Holm loss & `7B = 70B NF4` parity). Unanimous `2/2 APPROVED_FOR_V6_SEAL`. |",
        f"| **Layer 2 (`v7.0`)** | `#365` | `feat/v7-formal-pm6` (merged into `fix/stage0-metric-lineage-and-m1-ladder`) | `fix/stage0-metric-lineage-and-m1-ladder` | **Part VIII (`Waves 14–15`, Bundle `#46` + `/formal`, Registry `{reg_sha12}`):** Lean 4 `Spec.lean` `63fe3e644bde` (`68` theorems, `0 sorry/native_decide`, `nanoda_bin 0.4.32` `25,473` decls, `7,722`-ckpt conformance w/ `153` Pitfall-B reason diffs & incremental-deletions conformance-only) + Live `Demo D11 / PM-6` (`7248d05a02ba` + `pm6_posthoc_statistical_reanalysis.json`: **EMPIRICALLY MIXED** — beats `Arm 3` `+0.474/+0.476` AUROC & `DE` `+0.152`, `100%` Gate-1 `R4`, **FAILS `r5_benign_fpr_le_002` at `6.25%/11.76% > 2%`**, parity/loss vs `Arm 6/8` on EN `R3`, `15.6%/17.6%` `R2b` breach). |",
        "| **Layer 3 (`Post-v7.0`)** | `#366` | `pv7/w16-core` | `fix/stage0-metric-lineage-and-m1-ladder` (`v7.0`) | **Part IX (`Waves 16–21`, Additive Bundles `#47–#64`, Spine `32693f54e0ce`):** Declarative `registry.d`/`routes.d`/`verifiers.d` loaders (`18/18` offline verifiers `rc=0`) + honest breakdown: `3` synthetic/emulated `live-pass` (`#48`, `#57`, `#61`), `1` observational `live-shadow-pass` (`#53`), `1` `verified-mixed` (`#58` clustered win / scattered `-720 B` loss vs floor), `1` `stage1-replay-pass` whose **live vLLM worker run FAILED** (`#62` `overall_pass=false`, `0/1,680` on-time jobs), `10` `local-drycheck-pass` synthetic/smoke fixtures (`#47, #49–#52, #54–#56, #59, #60`), and `2` `protocol-ready` open DEV scorers (`#63, #64`). |",
        "",
        "---",
        "",
        "## 2. Honest `64/64` Classification Breakdown (`0` Vacated IDs, `0` Partial-as-Complete Inflation)",
        "",
        "| Evidence Classification | Count | Bundle Ordinals / IDs |",
        "|---|---:|---|",
    ]

    by_class: Dict[str, List[str]] = {}
    for r in sc["bundles"]:
        c = r["evidence_tier"]
        by_class.setdefault(c, []).append(f"`#{r['ordinal']}` (`{r['bundle_id']}`)")
    for c, items in sorted(by_class.items()):
        lines.append(f"| `{c}` | **{len(items)}** | {', '.join(items)} |")

    lines.extend(
        [
            "",
            "---",
            "",
            "## 3. Part IX (`Bundles #47–#64`, Waves 16–21) Honest Per-Bundle Evidence Scorecard",
            "",
            "| # | Fragment | Design Tier | Registry `status` | Honest `evidence_tier` | Verified Scope & Mandatory Brutal-Honesty Disclosures |",
            "|---:|---|---|---|---|---|",
        ]
    )

    for r in sc["bundles"][46:]:
        disc = r["evidence_disclosure"]
        disc_summary = "; ".join(f"**{k}:** {v}" for k, v in disc.items())
        lines.append(
            f"| **`#{r['ordinal']}`** | `{Path(r['fragment_file']).name}` | `{r['role_or_design_tier']}` | `{r['registry_status']}` | **`{r['evidence_tier']}`** | {disc_summary} |"
        )

    lines.extend(
        [
            "",
            "---",
            "",
            f"## 4. Parts I–VIII (`Bundles #1–#46`, Base `v7.0` Verified Registry `{reg_sha12}`)",
            "",
            "| # | Part | Bundle ID | Honest `evidence_tier` | `manifest_sha256` (Prefix) |",
            "|---:|---|---|---|---|",
        ]
    )
    for r in sc["bundles"][:46]:
        lines.append(
            f"| `#{r['ordinal']}` | `{r['part']}` | `{r['bundle_id']}` | `{r['evidence_tier']}` | `{r['manifest_sha256'][:12]}` |"
        )

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    sc = build_scorecard()
    OUT_JSON.write_text(json.dumps(sc, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    OUT_MD.write_text(render_markdown(sc), encoding="utf-8", newline="\n")
    print(
        f"WROTE {OUT_JSON.name} & {OUT_MD.name} "
        f"(bundles={len(sc['bundles'])}/64, scorecard_sha={sc['scorecard_sha256'][:12]})"
    )


if __name__ == "__main__":
    main()
