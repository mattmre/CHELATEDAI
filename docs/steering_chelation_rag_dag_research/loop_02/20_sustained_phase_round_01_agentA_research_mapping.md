# Sustained Phase Round Plan (Agent A — Research & Mapping) — First Long-Running 10-Agent Round
**Round ID**: Sustained-01 (under SUSTAINED_PHASE_ROUND_DRIVER.md + 60min scheduler context 019e6ab0e6d0; post-deletion of old 3min 019e6a78debf)  
**Date**: 2026-05-27 (first sustained round)  
**Agent A Role**: Research & Mapping — deep audit of governing north star + harness substrate + prior pivot work; concrete executable slice selection + mapping to 10-agent roles + harness entrypoints.  
**Governing North Star**: FULL_SHIM_LOOP_PHASE_PLAN.md (Phases 0-9) + SUSTAINED_PHASE_ROUND_DRIVER.md (explicit first recommended target) + 10_AGENT_SAFE_MERGE_COORDINATION_PROTOCOL.md (full §1-8 for dispatch)  

**Brutal Honesty Header (per all prior artifacts + rulebook v3.3 + goal success defs)**:  
This round plan and all work under it remain 100% research-only (docs/steering_chelation_rag_dag_research/artifacts/ + loop_02/). 0 substrate advance on goal success definition #1 (no real SIP wired to tts_pipeline.py:47-80 or antigravity_engine.py:2452-2600/2566-2600; no prod-path runtime deltas; no SHIM-CD-01 closure). Program score remains 10/100 flat. BLOCKED count:2 (FAIL via check_block_flag.py). OVERRIDE: NONE. 5-vs-10 L4/L9/L13 gap persists at scheduler/runtime level. All deliverables L3/L4 on synthetic harness only. Does NOT satisfy goal #1-3. Human §128 intervention or explicit OVERRIDE still required for any Phase 3 movement. This round tests *sustained 10-agent model fidelity* + produces measurable synthetic substrate deltas as Phase 1/2/5 proxy evidence.

---

## 1. Full Re-Read Citations (Mandatory §1 Protocol Compliance — Tool-Grounded, No Drift)

**Core re-reads performed 2026-05-27 (via list_dir, read_file, grep, run_terminal on absolute paths)**:

1. **FULL_SHIM_LOOP_PHASE_PLAN.md** (full 1-229 lines):
   - Phase 2 (lines 73-88): "Status: Recently Added. ... **Current Status**: Mechanism exists. Demonstration is partial (mostly documentation of the rule itself). **Needs real usage**." Objective: "Concrete examples of successful pivots (alternative slices advanced while #1 remains blocked)." Suggested focus: J/D/E. Primary risks: "the mechanism exists on paper but is never actually used (L9)."
   - Phase 5 (lines 136-148): "OPSD Trace Integration & Precomputed Shims". "Current Status: Basic synthetic trace generation exists (from G work in Cycle-011). Needs significant deepening and realism." "Can be advanced in parallel with Phase 3/4 as long as it stays research-only." Key deliverables: "High-quality synthetic privileged trace generator... At least one experiment showing that training on these traces produces better MTP predictors..."
   - Phase 1 (55-71): "Mostly Complete... Unblocked." Focus on "Full MinMaxBlockRelevanceScorer integration... Improved MTP de-mock... High-quality synthetic OPSD-style trace generation... Clear attribution fields." Suggested: B/C/I/G/J.
   - Phase 8 (181-193): "Low. This phase can and should run in parallel with others." Literature cross-pollination unblocked.
   - Phase 3 (91-118): "Core Blocker — Primary Workstream" at "0% complete. This is the single largest open item (SHIM-CD-01)". Explicit: "Work on this phase should normally be the highest priority... When blocked, the loop must explicitly pivot (see Phase 2)".
   - "How the Loop Should Use" (218-223): "When the highest-priority unblocked phase is not Phase 3, the loop should explicitly say 'We are in Pivot Mode, working on Phase X because Phase 3 is blocked by Y.' Progress is measured by movement across phases with supporting BHS evidence..."
   - Success criteria (20-30): #1 requires "At least one real (non-research-only) SIP... with before/after runtime evidence... BHS score ≥ 70". Until met: "the loop continues in either normal or Troubleshooting/Pivot mode."
   - Version: "2026-05-27: Initial creation as the synthesized north star..."

2. **SUSTAINED_PHASE_ROUND_DRIVER.md** (full 1-66):
   - "This replaces the previous 3-minute fragmentation loop... Goal is to demonstrate sustained, fully-implemented 10-agent work on unblocked phases."
   - "A 'Round' = one focused, fully-implemented development cycle on 1-2 high-leverage unblocked slices from the Phase Plan."
   - "Every Round **must** dispatch and collect **all 10 agents (A-J)** with independent artifacts before synthesis (enforce the original 10-agent model that the short loop never delivered at runtime)."
   - "Use the existing 10_AGENT_SAFE_MERGE_COORDINATION_PROTOCOL.md for all parallel work (mandatory re-reads, safe edit order...)"
   - "Research guard + BLOCKED + SHIM-CDs remain in force until human changes them (OVERRIDE or debt clearance). No prod SIP wiring."
   - "**First Recommended Long Round Target (as of 2026-05-27)**: Advance Phase 2 ('real usage' of pivot + resilience) + Phase 1/5 (MTP synthetic signal + MinMax correlation + trace generator variance work) with a full 10-agent wave. This is completely unblocked, has existing harness substrate from prior pivot work, and directly tests whether the new longer model can deliver the 10/10 fidelity the old loop never achieved at runtime."
   - 10-agent roles (26-37): Explicit A (Research & Mapping), B (Build), C (Test & Evidence), ... G (OPSD / Trace Work), I (MTP Prototype), J (Meta Auditor — "fidelity of the 10-agent round itself, protocol health, Phase 2 'real usage' vs L9 theater assessment").
   - BHS invariants: "Explicit '0 substrate / does not satisfy goal success def #1' while SHIM-CD-01 + BLOCKED + research guard are active." "10-agent fidelity is now load-bearing (0/10 = automatic L4 + score cap)."

3. **10_AGENT_SAFE_MERGE_COORDINATION_PROTOCOL.md** (key excerpts via reads):
   - §1: Mandatory 9-file re-reads + block FAIL + 0-prod "exactly 2 research files" + scheduler + loop_02/ list before any action.
   - §2: Safe edit order (A/D audit first → B narrow guarded → C evidence → distinct NN_ loop_02/ files). Append-only coord notes on shared (harness/shim_node).
   - Pivot Rule (236+ referenced in prior): Must not repeat failing patterns; use Phase Plan for intelligent pivots; log "We are in Pivot Mode..."; 0/10 fidelity triggers L4.
   - Troubleshooting Mode when OVERRIDE: NONE + BLOCKED.

4. **Harness substrate (shim_collapse_benchmark_extension.py — 2828 lines; exact mappings)**:
   - Synthetic substrate: imports build_synthetic_collapse_fixture / evaluate from synthetic_collapse_benchmark (216-224); ShimCollapseBenchmark + simulate_sip_effect etc. (core metrics e.g. noise_reduction ~0.78863193 stable across cycles).
   - Trace generator (Phase 5 / G): `generate_successful_synthetic_shim_cascade_traces` (1022-1147): produces json traces (context/cascade/outcome with forced high success_rate~1.0, low cost, rollback_proof via temp_experiment + record_shim_activation). CLI --family traces. Backlog #4 (Cycle-010 Agent6). L4 synthetic only.
   - MTP (Phase 5/1 / I): `Cycle011_MTPShimLookahead` (627-703) + `synthetic_eval_on_gtraces` (705-777): uses traces, MinMaxBlockRelevanceScorer (745-749 toy blocks for variance), usage_stats, predicts or "no cascade". Returns hit_rate/precision_at_k (historically flat ~0.2; post-17 alt ~0.33 with injected feature variance). CLI --research-mtp + --family traces. L3 mock / 0 real head (explicit in 774, SHIM-CD-03).
   - MinMaxBlockRelevanceScorer (854-992): compute/filter/partition for relevance/variance proxy. Wired into MTP eval post-17 pivot alt (harness:742 "derive *varying* features... instead of constants").
   - Recent pivot hygiene/variance/correlation (explicit notes):
     - Pivot Fire (162-181): Phase 2 "Needs real usage" + protocol Pivot Rule; hygiene for MTP/G substrate; 00_pivot md + bhs_pivot_mtp_gtraces_20260527.json. Results: flat 0.2.
     - PIVOT ALT (592-613): "make synthetic_eval_on_gtraces derive varying min_max via MinMax... + usage deltas from trace['outcome']". "first measurable delta on this substrate" (hit 0.3333). L-tax: L1/L3/L4/L9. 17_pivot_alt_mtp_variance_20260527.md + json.
     - Correlation fire (19_): Post-alt analysis on 60 traces: "Mean min_max=0.8335 (std 0.1379 — good variance... but generator construction... leaves zero outcome variance for correlation." "High-mm vs low-mm success delta=0.0". Diagnosis: need generator outcome variance for real corr signal.
   - 0-prod invariant (repeated in notes 148, 193 etc.): Active Shim*/MinMax/MockMTP/Cycle011_MTP only in exactly 2 research files (shim_collapse... + shim_node.py); prod files have only "Wired? NO" comments.
   - CLI families (2149): traces, --research-mtp, --minmax-blocks (all gated).
   - L-taxonomy / CANNOT PROVE (2556-2829): Explicit L1 (scaffold), L3 (mocks), L4 (partial + "while #1 0%"), L5 (synthetic fixture only), L13 (prose vs mech); "does not satisfy goal success def #1"; HARD REQUIREMENTS for promotion list real SIP + Tier B + non-synthetic fixture etc.

5. **Recent loop_02/ pivot artifacts (direct substrate state)**:
   - 00_pivot_fire_20260527_mtp_gtraces_phase2_demo.md: "First concrete demonstration of the Pivot Rule... Phase 2 'real usage'". Post-hygiene eval still weak 0.2; "Correlation observation: ... flat/weak... no visible strong relationship". "Proposed next: Increase variance in the synthetic trace generator".
   - 17_pivot_alt_mtp_variance_20260527.md: "Post-edit (this alt...): hit_rate=0.3333... first measurable delta". "The 17 alt successfully injected min_max variance... but generator... leaves zero outcome variance".
   - 19_fire_019e6a78debf_pivot_mtp_correlation.md: Explicit "We are in Pivot Mode, advancing Phase 2... + Phase 1/5...". 60-trace analysis confirming diagnosis. Repro on harness.
   - Prior Cycle-011: 01_cycle011_agentA..., 02_...B, 03_C, 07_G (traces), 09_I (MTP), etc. (10/10 collection gate in 00_).
   - 18_fire..._stats.md referenced in 19_.

6. **Supporting**:
   - BHS_SHIM_LOOP_DASHBOARD.md: Program 10/100 flat; repeated "0 substrate"; 5-vs-10 L4/L13; §128 recs; historical cycle rows all capped for fidelity failures + OPEN SHIM-CDs.
   - scripts/check_block_flag.py + next-session.md:22 (BLOCKED count:2 + "Carried Debt row count: 2" + "RESULT: FAIL" + SHIM-CD-01..09 OPEN incl. #3 on MTP L3 pure sim + #9 L9 meta).
   - shim_node.py: ShimRegistry / apply_shim_cascade / usage_stats (exercised by generator; research only).
   - OPERATOR_OVERRIDE.md: NONE.
   - scheduler_list (historical context): old 3min active until deletion; new 60min per driver/task.

**No VR drift / context rot**: All citations are fresh tool outputs (read_file offsets/lines, grep matches with -B/-A, list_dir, wc -l=229 for plan). Absolute paths used. Pre-edit protocol followed for any future harness touches.

---

## 2. Concrete Round Plan: 1-2 Narrow Executable Sub-Slices (Phase 2 + Phase 5)

**Selection Rationale (narrow, high-value, one sustained round ~60min soft timebox, measurable synthetic deltas, unblocked, harness-mapped, 10/10 fidelity test)**:
- Directly executes the **DRIVER's explicit "First Recommended Long Round Target"** (Phase 2 "real usage" + Phase 1/5 MTP synthetic/MinMax correlation/trace generator variance).
- Addresses **PLAN Phase 2 "Needs real usage" gap** (prior 00/17/19 were partial single-fire or alt; this round delivers full 10-agent coordinated pivot usage with J fidelity audit + independent artifacts).
- Addresses **PLAN Phase 5 "Needs significant deepening and realism"** + post-19 diagnosis (variance in features achieved; outcome variance missing → no corr signal possible; generator forces success~1.0).
- **Prioritizes measurable runtime deltas on synthetic substrate** (harness synthetic_eval_on_gtraces + generate_... + MinMax toy paths): e.g., hit_rate/prec std across seeds >0 (vs prior flat 0.2), reported Pearson/spearman corrs between mm_scores and outcomes, ablation deltas, wall-time attribution, before/after json diffs.
- Feasible in one round: All changes narrow guarded appends to *existing* harness paths (no new files except mandated per-agent loop_02/NN_*.md + artifacts/ bhs_*.json). Uses --family traces / --research-mtp entrypoints. Protocol safe order + distinct artifacts enforced.
- **NOT in scope (over-scope forbidden)**: Any Phase 3 SIP (0%), prod touches, real OPSD data, training loops, new test files, dashboard/plan edits beyond E synthesis, literature deep-dives (Phase 8 parallel only if spare), MicroSLM/H policy closure.
- Pivot Mode declaration (per PLAN:221 + DRIVER + protocol): "We are in Pivot Mode, advancing Phase 2 (full 10-agent 'real usage' of resilience machinery via variance/corr experiment) + Phase 5/1 (MTP synthetic signal + trace generator outcome variance + MinMax/usage correlation) because Phase 3 is blocked by SHIM-CD-01 + BLOCKED count:2 + research guard + OVERRIDE: NONE."

**The 1-2 Sub-Slices**:
1. **Phase 2 Primary — Full 10-Agent Pivot "Real Usage" Fidelity Round on MTP Synthetic Variance + Correlation Substrate**: Orchestrated dispatch of A-J (A: this plan; B/I/G/C as below + D L-tax/J fidelity audit/E synthesis). Produce 10 independent artifacts + 1+ bhs json(s) quantifying: (a) 10/10 collection gate success (vs historical 0/5 or 5/10 gaps in cycles 1-11), (b) runtime harness deltas from variance work (hit/prec movement + corr stats on synthetic), (c) J's process audit ("real usage" vs L9 theater; protocol health; 5-vs-10 gap status for *this* round). Maps to harness MTP eval + generator + recent 17/19 alts. Success: 10 distinct loop_02/ files + measurable synthetic numbers + J "fidelity: 10/10 achieved in sustained model" (or honest gap disclosure).
2. **Phase 5 (Harness/Phase1 support) — Trace Generator Outcome Variance + MTP/MinMax Correlation Surface**: The technical payload enabling #1. G leads generator extension; I leads MTP eval deepening for corr/ablation/variance reporting; B narrow plumbing if needed; C owns all measurement + persisted evidence proving deltas. Produces the "better MTP predictors" experiment signal per PLAN Phase 5 deliverable (synthetic only).

**Expected Measurable Runtime Deltas (synthetic substrate priority)**:
- Generator runs with outcome_variance>0: traces show success_rate distribution (e.g. mean 0.85-0.95, std>0) vs forced 1.0.
- MTP synthetic_eval (multiple seeds/runs pre/post): hit_rate/prec_at_k now vary (std reported >0.05-0.1); corr(mm_mean, success_rate) computed and non-trivial (e.g. |r|>0.1 or ablation delta >5% relative).
- Attribution in bhs json: "delta from G variance injection: +X hit_rate std"; "I corr surface: r=0.XX"; wall times; rollback proofs intact.
- All survive `git clean -fdx && python -B <exact smoke>` on research paths only.

---

## 3. Specific Deliverables Expected from B, I, G, C Agents (Mapped to Harness + Protocol)

All agents: Full §1 re-reads (citations in their mds), append coord note (safe order: A plan first provides clearance), distinct loop_02/ files (e.g. 02_sustained_round_b_*.md, 03_...c_..., 07_...g_..., 09_...i_...), bhs_*.json where applicable, explicit L-tax + "0 substrate / does not satisfy goal #1" + Pivot Mode declaration + repro SMOKE. No shared file overwrites. Post-work: C/D/J/E gates before any synthesis.

- **Agent G (OPSD / Trace Work — primary on Phase 5 generator deepening)**: 
  - Narrow guarded extension to `generate_successful_synthetic_shim_cascade_traces` (harness ~1022-1147; new optional param e.g. `outcome_variance: float = 0.0` default for backward compat; when >0 use seeded rng to set probabilistic `was_success` + jitter `token_cost_delta` / quality_proxy in record + outcome). Keep "successful" filter tunable or add variant.
  - Update traces-family CLI path + comments/samples (harness ~2180+).
  - Independent artifact: loop_02/07_sustained_round_g_traces_variance.md (with before/after generator snippets, example trace json diff showing variance, EVIDENCE/SMOKE).
  - Contributes to shared bhs json (attribution fields).
  - Citations: harness generator docstring/backlog#4, 17/19 pivot diagnosis ("zero outcome variance"), PLAN Phase 5 "high-quality synthetic privileged trace generator".
  - BHS: L3 (synthetic generator), L4 (while #1 0%).

- **Agent I (MTP Prototype — primary on Phase 5/1 MTP synthetic signal + correlation)**:
  - Enhance `Cycle011_MTPShimLookahead.synthetic_eval_on_gtraces` (harness ~705-777) and/or add helper: (a) multi-seed loop (e.g. 5 seeds) reporting hit/prec mean/std; (b) extract per-trace mm_scores + outcome success_rate → np.corrcoef / simple stats; (c) ablation (run predict with mm only / usage only / both; delta hit rates); (d) logging of "no cascade" drivers.
  - Optional: light CLI surface under --research-mtp (no default change).
  - Independent artifact: loop_02/09_sustained_round_i_mtp_correlation.md (numbers vs 17/19 baselines e.g. "pre: flat 0.2; post-variance: hit std=0.08, corr(mm,success)=0.XX, ablation +12% with mm", json payload).
  - Citations: harness 627 (class), 742 (prior alt variance note), 19_ correlation diagnosis, Cycle011_MTP notes, PLAN Phase 5 "experiment showing... better MTP predictors".
  - BHS: L3 (mock + heuristic), L4 (partial deepening while BLOCKED).

- **Agent B (Build — narrow guarded support plumbing for above)**:
  - Only as needed post-A clearance + I/G design: minimal append (e.g. private helper `_derive_varying_outcome(...)` or context builder for eval using new generator variance; or MinMax integration tweak for corr surface). Strictly behind CHELATED_SHIM_RESEARCH / --research-* flags. Harness only (shim_collapse... or shim_node compat if registry usage_stats impacted).
  - Full coord note (pre-grep, safe order citing this A plan as clearance).
  - Independent artifact: loop_02/02_sustained_round_b_harness_plumbing.md (diff summary, EVIDENCE of no core metric regression e.g. sip_effect noise~0.7886 unchanged, rollback).
  - 0 claims of substrate advance outside synthetic.
  - Citations: harness MinMax 854+, generator 1022, prior Cycle-010/011 B notes (e.g. 131+), protocol §2.
  - BHS: L4 (partial while #1 0% + BLOCKED).

- **Agent C (Test & Evidence — owner of all measurement + persisted deltas)**:
  - Comprehensive execution: pre-round baseline re-runs (exact 17/19 repros), post G/I/B changes runs (multiple n_traces/seeds/families: traces + --research-mtp + --minmax-blocks where relevant), sweeps.
  - Persist 1+ new artifacts/bhs_sustained_round_mtp_variance_correlation_*.json (full eval outputs + deltas + attribution to G/I edits + runtimes + hashes + bhs_evidence blocks with "0 substrate / does not satisfy #1").
  - Independent artifact: loop_02/03_sustained_round_c_evidence.md (SMOKE commands e.g. `CHELATED_SHIM_RESEARCH=1 python -B ... --family traces --research-mtp ...`; exact before/after tables; proof all survive fresh checkout; full CAN PROVE on deltas + CANNOT on anything prod/real).
  - Cross-verify 0-prod / block gates post any harness edit.
  - Citations: harness main 2145+ CLI, 2180 traces/MTP branch, prior C 03_ files, EVIDENCE banners throughout py.
  - BHS: L3/L4 (evidence on L3/L4 substrate); explicit "Visible = Verified".

**Other agents (for completeness; A dispatches per protocol)**: D (full L1-13 on round + round score cap), E (cross-synth + dashboard/phase status update + 4Qs), F (optional targeted lit if unblocked time), H (N/A this slice), J (Phase 2 "real usage" vs L9 theater + 10-agent fidelity audit: "10/10 artifacts collected in sustained round; process delta vs cycles 1-11"; protocol health).

**Round Gates (per DRIVER + protocol)**: All 10 artifacts + jsons present + independent before E/J synthesis. 0-prod + block re-check. Quantified synthetic deltas in summary.

---

## 4. BHS L-Taxonomy for This Round Plan + Expected Work (Research-Only Expectation: L3/L4 Dominant)

**On the plan itself (A output)**:
- L1 (Scaffold): This md is research mapping artifact only.
- L3 (Mock-ate-real): All mappings are to synthetic harness mocks (MTP L3 per SHIM-CD-03/harness:2199, generator L4 synthetic per 1020).
- L4 (Partial + claim risk while #1 0%): Explicit "first full 10-agent sustained" language bounded by "tests the model"; "measurable synthetic deltas" only; full "0 substrate / does not satisfy goal #1" + Pivot Mode + BLOCKED disclosures repeated. Severity cap.
- L9 (Doc-as-impl / meta volume): Mitigated — this is *one* mandated A deliverable per DRIVER round structure; no repeated failing pattern; focuses on executable slices with C evidence required. J will audit for theater.
- L13 (Soft-prose as mechanical): No claim this "advances Phase 3" or "closes SHIM-CDs"; explicit "proxy evidence on synthetic", "human intervention still required".
- No L2/L5(new)/L8/L10/L11/L12 from this doc (no code, no tests, no broad claims).

**On expected B/I/G/C deliverables (synthetic harness only)**:
- L1: Any new helpers/scorer calls (harness-local).
- L3: Core MTP eval, trace gen, predictions (explicit mocks per class docs + 774 note).
- L4: All "deepening"/"variance injection"/"correlation surface"/"deltas" while SHIM-CD-01/BLOCKED/0 SIPs + research guard (disclosed in every artifact + coord notes + json "research_guard" fields). "Partial" on Phase 2 "real usage" (full dispatch is new for sustained model; prior was partial fires).
- L5: All on synthetic_collapse fixture + toy blocks (harness partition 884+, generator 1122 fixture).
- L9 (bounded): Mitigated by protocol (A first, distinct files, C runtime proof, J audit); this round produces *actual harness runtime substrate deltas* (not pure doc).
- L13: Bounded by repeated "synthetic only", "harness simulation", "no real OPSD/head", "does not satisfy #1" + HARD REQUIREMENTS section in py (2710+).
- Process: Adding Phase 5/1 work while #1 open = disclosed L4/L9 risk (per PLAN 162 + goal §157 + prior 17/19 notes); tracked in artifacts.

**Round Score Self-Draft Expectation (capped)**: 35-45/100 possible for 10/10 fidelity + synthetic deltas + Phase 2 "real usage" execution + honest L/0-substrate language. Heavy caps for BLOCKED + 0 on goal #1 + 5-vs-10 history + program 10/100. D/J will finalize adversarial.

**Full Disclosure**: Every artifact in this round must contain (or link) the py's "HARD REQUIREMENTS FOR ANY FUTURE PROMOTION" (2710-2728) + "does not satisfy goal success def #1".

---

## 5. Execution Notes for Orchestrator / 10-Agent Dispatch (This Round)

- **Start**: This A md + todo_write (1 item: "Execute Sustained-01 per plan") as Round Start (DRIVER 0-5min).
- **Dispatch**: Spawn A-J (orchestrator coordinates per protocol; A already executing). Enforce collection gate (all 10 + jsons) before E/J.
- **Duration**: Soft 60min; log overrun only if zero output. Background-friendly.
- **Artifacts Location**: loop_02/20_sustained... (this) + 02_b_..., 03_c_..., 07_g_..., 09_i_... + others (distinct naming); artifacts/bhs_sustained_round_*.json.
- **Post-Round**: E updates dashboard + phase plan status (synthetic deltas noted under Phase 1/2/5); J fidelity report; explicit "next round recommendation" (continue variance/corr or new unblocked slice).
- **SMOKE for round success**: 10 distinct loop_02/ files present; at least one bhs json with "hit_rate std" or "corr" numbers + "synthetic delta" attribution; all agents cite this plan + full re-reads + "0 substrate on #1"; block still FAIL; 0-prod exactly 2 research files.

**References for all agents (embed in headers)**: This file + FULL_SHIM... (Phase 2:83 "Needs real usage", Phase 5:145 "basic synthetic... deepening", Phase 1:60 MinMax/MTP/traces, driver:57 "First Recommended... Phase 2 + Phase 1/5 MTP... variance + correlation"), harness exact lines above, 00_/17_/19_ pivot mds (diagnosis + prior deltas), protocol §1-2 + Pivot Rule, check_block_flag live FAIL.

---

**End of Agent A Research & Mapping Round Plan**.  
This is narrow, executable, harness-mapped, BHS-honest, and directly advances the new sustained 10-agent model on unblocked phases while producing synthetic runtime deltas. 0 over-scope. Ready for dispatch.

**Visible = Verified** (citations tool-grounded; no synthesis claims here).  
**0 substrate on goal #1** (repeated for emphasis). 

Next action (orchestrator): todo + full 10-agent spawn using this plan.