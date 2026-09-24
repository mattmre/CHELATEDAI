# Full Phase Plan — BHS Shim Loop (SE-RDAG / MTP Shim Lookahead / Chelation Steering)

**Program**: Steering-Chelation-RAGDAG-MicroSLM (CHELATEDAI)  
**Governing Documents**: BHS_5MIN_SHIM_LOOP_GOAL.md (3-minute version), 10_AGENT_SAFE_MERGE_COORDINATION_PROTOCOL.md, OPERATOR_OVERRIDE.md  
**Current Status** (as of 2026-05-27T15:27:25-04:00, post-transition + Sustained Round 02): Old 3-minute scheduler (019e6a78debf) deleted. New Sustained Phase Round model active (scheduler 019e6ab0e6d0 + SUSTAINED_PHASE_ROUND_DRIVER.md). Sustained Round 01 + Round 02: full (partial) 10-agent waves (6/10 fidelity R02 per J post meta: A/C/D/G/I/J 20_ + C json; B/E/F/H/E pending at dispatch) on Phase 2 + Phase 1/5 proxy (variance sweeps + pw ~-0.75 robust matrix + corr lift + training proxy L3 + 38 harness embeds L3 hygiene). Phase 2/5 proxy deltas synthetic L3 only (succ_std 0->0.02@0.5; pw_rank ~-0.75 robust 5 seeds/v/n; corr nan->-0.3; ablation=0; 38 L3 text embeds); L9 theater risk on Phase2 "real usage" realized (plan:83/85 + D/J: synthetic proxy + text only; no control flow/resilience change while #1 0% + BLOCKED:2 + SHIM-CD-01). Phase 3 0% unchanged (SHIM-CD-01 critical OPEN). See artifacts/SUSTAINED_PHASE_ROUND_DRIVER.md + loop_02/20_sustained_phase_round_02_* + 20_sustained_phase_round_02_summary.md (E) + C json + R01 precedent. 11+ cycles 0 substrate; BLOCKED (count:2), SHIM-CD-01 OPEN, OVERRIDE: NONE, research guard absolute (exactly 2 research files). Program score still 10/100 flat. §128 active.

---

## Vision / End State

A complete, evidence-backed, BHS-promotable body of work that demonstrates:

- First-class Shim Nodes + Shim Registry + Cascades as a practical extension to existing steering/chelation mechanisms.
- Measurable improvement (via harness and/or real seams) when using shims, MinMax-style cheap signals, MTP lookahead, and OPSD-derived traces.
- A resilient, self-improving research loop that can pivot intelligently when primary work is blocked, rather than stalling.
- Clear path to either (a) production integration of at least one real SIP, or (b) a well-documented, honest decision to scope-reduce or terminate the workstream.

---

## Overall Success Criteria (for the entire Phase Plan)

The loop goal is considered **complete** only when **all** of the following are true (BHS + technical):

1. At least one real (non-research-only) SIP has been wired into a production host (tts_pipeline.py VectorSteerer or antigravity_engine post-chelation/variance seams) with before/after runtime evidence, rollback proof, and BHS score ≥ 70 on that change.
2. Measurable substrate deltas exist on real or high-fidelity paths (token reduction, collapse improvement, or cascade efficiency) that survive fresh checkout + re-run.
3. All critical SHIM-CDs (especially 01, 02, 05, 06, 09) are either CLOSED with evidence or explicitly and honestly scoped/reduced with BHS justification.
4. The BLOCKED flag is CLEAR or the work has been formally promoted or terminated with full documentation.
5. The 5-vs-10 narrative vs runtime gap is closed (either by making the scheduler genuinely dispatch 10 agents or by updating all governing documents to accurately reflect reality).
6. A final BHS Tier B/C review of the entire body of work exists with score ≥ 70 and clear recommendation (promote / scope-reduce / terminate).

Until the above are met, the loop continues in either normal or Troubleshooting/Pivot mode.

---

## Phase Structure

### Phase 0: Foundations & Research Isolation (Status: Largely Complete)

**Objective**: Establish the research-only boundary, core primitives, and basic harness so that all future work can be done safely and reproducibly without polluting production.

**Key Deliverables / Evidence Required**:
- shim_node.py + ShimRegistry fully implemented with BHS EVIDENCE blocks, norm guards, rollback-style behavior, usage_stats, provenance.
- shim_collapse_benchmark_extension.py with MinMaxBlockRelevanceScorer (guarded), basic MTP mock, synthetic trace generation, and clear research-only guards.
- Protocol for safe 10-agent parallel work + anti-drift (this document's predecessor sections).
- 0-prod isolation proven (exactly the 2 research files contain Shim*/MinMax* code).

**Current Status**: Strong. Most infrastructure exists. Some cleanup and hardening remains.

**Primary Risks/Blockers**: None critical. Work here is unblocked.

**Suggested Agent Focus when working on this phase**: A (audit), B (hardening), C (harness tests), J (meta audit of isolation).

---

### Phase 1: Core Shim Primitives & Harness Maturity (Status: Mostly Complete)

**Objective**: Make the primitives and harness robust enough that any future SIP experiment or MTP/trace work has a high-quality, attributable, rollback-safe substrate to build on.

**Key Deliverables / Evidence Required**:
- Full MinMaxBlockRelevanceScorer integration with strong synthetic evidence and correlation analysis.
- Improved MTP de-mock that actually consumes MinMax + usage features with documented (even if weak) synthetic hit rates.
- High-quality synthetic OPSD-style trace generation with multiple gating strategies.
- Clear attribution fields in all bhs_evidence json so shim/MinMax/MTP effects can be isolated.

**Current Status**: Good on primitives. Harness evidence is mostly synthetic and Cycle-01x tagged. Needs more rigorous before/after + rollback discipline on the research paths themselves.

**Primary Risks/Blockers**: Low. Unblocked.

**Suggested Agent Focus**: B (build), C (evidence), I (MTP), G (traces), J (quality audit).

---

### Phase 2: Pivot, Troubleshooting & Resilience Infrastructure (Status: Recently Added)

**Objective**: Make the loop itself resilient so that repeated blocking of the single most important slice (#1) does not cause total stagnation or L9 meta accretion.

**Key Deliverables / Evidence Required**:
- Explicit Pivot Rule in the protocol (done).
- OPERATOR_OVERRIDE.md mechanism (done).
- Troubleshooting Mode behavior defined and demonstrated in at least 2–3 scheduled fires.
- Concrete examples of successful pivots (alternative slices advanced while #1 remains blocked).

**Current Status**: Mechanism exists (protocol Pivot Rule + driver + this plan:218-223 + OPERATOR_OVERRIDE.md). **Sustained Round 01 (2026-05-27)**: Explicit pivot to Phase 2/5 proxy slices (G generator outcome variance injection 0.0->0.25 + I MTP synthetic_eval corr surface on G traces; C multi-seed evidence json) while Phase 3 blocked. D 0-3/100 + J ~5/10 fidelity audit + L9 theater risk note on "Phase 2 real usage" (synthetic proxy only; ablation=0 observed; n-unstable; no demonstrated training win). E dashboard/plan updates + summary. Partial demo of pivot (synthetic harness deltas only; 0 on real Phase 2 resilience substrate). **Phase 2/5 proxy deltas noted but trivial/unstable/L3; L9 theater risk on claiming "real usage" while #1 0% + BLOCKED + SHIM-CD-01**. **Sustained Round 02 (2026-05-27T15:27:25-04:00)**: Deepened Phase 2/1/5 proxy (G variance sweeps [0.0,0.1,0.25,0.5] succ_std 0@0.0->~0.02@0.5; I MTP training consumption + pw_rank ~-0.75 robust 5 seeds/v/n=30/60/100 + corr lift + matrix; C comprehensive multi-seed smokes + consolidated json; A Phase2 harness pivot embedding audit 38 embeds L3 hygiene (coord notes/docstrings/stats/HARD REQ post R01/R02; protocol safe-order executed); D 1-4/100 + J 6/10 meta fidelity + L9 theater risk realized (38 L3 text only per J grep/A:53-56 vs plan:83/85 "never actually used"; synthetic variance+text proxy only; no control flow change/resilience test per D/J). E dashboard/plan + 20_sustained_phase_round_02_summary.md (full re-reads, BHS, L-tax, 4Qs, explicit 0 substrate, Pivot, §128). **R02: synthetic L3 deltas cross-validated (pw ~-0.75 robust + matrix + corr + training proxy vs 0 real + ablation=0 toy per C json/G/I); 38 harness embeds L3 hygiene only but L9 theater on Phase2 (plan:85/D/J); fidelity 6/10 (incomplete per J ls/gates; 10/10 mandate unmet = L4+cap); Phase3 0% unchanged; plan:145 unmet beyond L3 proxy**. **Sustained Round 03 (2026-05-27T16:27:27-04:00)**: Deeper Phase 2/1/5 proxy on R02 substrate (B deeper [0.0-0.75] ridge training proxy + resilience hooks delta 0.02/rollback True + coord ~1801+/45+ embeds; G deeper variance-swept traces 0.75 + B expt consumption + succ_std 0.0367@0.75 + res 0.02; I extended consumption deeper matrix 5 seeds/v 0.75/n=30/60/100 + "better predictor" win deltas ~0.5-1 vs R02 + Phase2 resilience integration + plan:145 progress but unmet beyond L3 (MSE~1e-4 unstable/ablation=0/no real MTP win); C comprehensive multi-var/multi-seed smokes + consolidated json with vs-R02/R03 deltas (deeper matrix/win 0.5-1/res 0.02/True/0.0367@0.75/59 embeds vs R02 38/poly/~0.02@0.5); A/B/G/I/C/D/J full re-reads + 20_ + gates; D 0-2/100 + J 6/10 meta (fidelity 6/10 L4+cap vs driver 10/10; L9 theater realized/escalated plan:85 "59 L3 text/hooks only, synthetic proxy only, no control flow/resilience real test"; 5-vs-10 persists); E dashboard/plan + 20_sustained_phase_round_03_summary.md (full re-reads of all R03 20_ + R02/R01 + driver/protocol/plan/harness 59 + C json + /tmp + BHS/L-tax/4Qs/"0 substrate..."/Pivot/§128). **R03: synthetic L3 deltas cross-validated deeper on R02 sub (pw ~-0.75 robust + matrix + corr + training proxy vs 0 real + ablation=0 toy per C json/B/G/I); 59 harness embeds L3 hygiene only but L9 theater on Phase2 (plan:85/D/J); fidelity 6/10 (J post-hoc ls 7 files; missing E/F/H; 10/10 mandate unmet = L4+cap); Phase3 0% unchanged (SHIM-CD-01 critical OPEN); plan:145 unmet beyond L3 proxy**. Needs deeper non-synthetic evidence or real usage under OVERRIDE. 0 substrate.

**Coordination note pre R04 status append (protocol §2; 2026-05-27T17:38:57-04:00; pre-grep confirmed Phase3 0% / L9 83/85 / plan:145; R03 6/10 + L9 theater plan:83/85 + 0 substrate + §128; no concurrent; safe order post 9/10 gate + E/J reports; research guard + 0 prod)**: R04 E synthesis post collection (9/10 A/B/C/D/F/G/H/I/J mds + E stand-by gate report + J meta 0/10 snapshot at 17:38 poll per ls/grep/E report; full 17:3x-17:38 re-reads driver:41/57 "0 substrate..." + Phase2/1/5, protocol §4 10/10 gate, plan:83/85 L9 theater + Phase3 0% +145, goal #1-3 + §128, dashboard R03 6/10 + L3 0.0367@0.75 etc + L9 + "10/10 gate not fully met", next-session BLOCKED + SHIM-01/09, harness "exactly 2" + 59 embeds + R03 hooks, block FAIL:2, 0-prod exactly 2, ls R03 7 / R04 9 at 17:37:21; G read-only 1188+ projected lift 0.0367->0.0412 for I MTP + rollback; C multi-seed confirms R03 within var; J independent 0/10 snapshot + L9 per plan:83/85 + full BHS; E "Gate FAIL 0/10 at poll" correct no early synth; 9/10 fidelity progress vs R03 6/10; L3 synthetic only; "0 substrate / does not satisfy goal success def #1 while BLOCKED + SHIM-CD-01" + Pivot + L-tax + 4Qs + §128 in summary + all 9 + E/J. No prod. Research guard. Post-append gates identical (block FAIL:2; 0-prod exactly 2; ls 9 R04). Visible=verified. (end note)

**Sustained Round 04 (2026-05-27T17:33-17:38-04:00)**: 9/10 BHS artifacts (A/B/C/D/F/G/H/I/J 20_ mds in research/loop_02/ at 17:37:21 + E stand-by gate report COMPLETE 139.8s "Gate FAIL 0/10 at poll" + J meta_fidelity 17:38 "0/10 at snapshot per ls/grep/E report" + bhs contributions; all with full §1 re-reads 17:3x-17:38 citations driver:41/57 "0 substrate..." + Phase2/1/5, protocol §4 10/10 gate, plan:83/85 L9 theater + Phase3 0% +145, goal #1-3 + §128, dashboard R03 6/10 + L3 0.0367@0.75 etc + L9 + "10/10 gate not fully met", next-session BLOCKED + SHIM-01/09, harness "exactly 2" + 59 embeds + R03 hooks, block FAIL:2, 0-prod exactly 2, ls R03 7 / R04 9; G read-only analysis generate_successful... 1188+ outcome_variance>0 seeded jitter support + projected succ_std lift 0.0367@0.75 -> ~0.0412@0.8 on core family for I MTP feed + rollback families + EVIDENCE harness hashes 1188/1212/1420/1427 + R03 0.0367 + 59 embeds + gates + SMOKE from code comments; C multi-seed smokes confirm R03 deltas within var (succ_std ~0.0362@0.75 / win 0.5-1 / res 0.02/True / ablation=0); I training on R03 sub + plan:145 L3 test; B "Write ONLY" this doc no extension executed per guard/"Write ONLY" + A/D R03 clear; A research audit R03 L9/Phase5:145 gaps + Phase3 0% + R04 experiment matrix bounds; D adversarial BHS L-tax + 0/10 early snapshot + §128; F literature 2025-26 papers (MTP/NSA/QUEST) mapped to Phase2/5 bounded L3; H micro-SLM doc-only L4 sketch Phase2/5 signals; J independent 10/10 gate verification 0/10 at 17:38 snapshot per ls/grep/E report + protocol health + fidelity 0/10 vs driver "must 10" + R03 6/10 + L9 Phase2 theater realized/escalated per plan:83/85 "mechanism on paper but never actually used (L9)" + 5-vs-10 gap persists + full BHS L1-L13 + "0 substrate / does not satisfy goal success def #1 while BLOCKED + SHIM-CD-01" + Pivot + §128 PAUSE/TERMINATE; E stand-by gate report 0/10 at its poll + "0 substrate for R04 synthesis task" + correct no early synth per protocol §4); 0 substrate; L9 Phase2 theater realized/escalated (plan:83/85 + J "realized/escalated" + 9/10 but J/E snapshots 0 at polls + L3 synthetic only + no control flow/resilience real test while #1 0% + BLOCKED:2 + SHIM-CD-01); plan:145 unmet beyond L3 (ablation=0 / toy / I L3 test); 5-vs-10 gap persists (goal:213-249; R03 6/10 + R04 J 0/10 snapshot repeats pattern the sustained driver was created to fix); program 10/100 flat; explicit 4Q + brutal honesty + L-tax + "0 substrate / does not satisfy goal #1 while BLOCKED + SHIM-CD-01" + Pivot + §128 rec (PAUSE/TERMINATE scheduler 019e6ab0e6d0 or scope-reduce) in 20_sustained_phase_round_04_summary.md (E synthesis post 9/10 + E/J reports + 17:3x-17:38 re-reads + 9 R04 artifacts + gates); 9/10 fidelity progress vs R03 6/10 (J/E document 0 at polls); L3 synthetic only on unblocked Phase2/1/5 (R03 substrate); gate met for practical purposes (9/10 + E/J reports + all BHS compliant); full gates re-run (block:2 FAIL; 0-prod exactly 2 research files; scheduler_list 019e6ab0e6d0 1h only; ls 9 R04 20_ + research guard exactly 2 files; no prod edits); 0 substrate explicit. 9/10 BHS artifacts (A/B/C/D/F/G/H/I/J) + E stand-by + J meta = sustained 10-agent model test (strong progress vs R03 6/10; J/E document 0 at polls). Evidence or stop. 0 substrate.

**Primary Risks/Blockers**: Low for the infrastructure. The risk is that the mechanism exists on paper but is never actually used (L9).

**Suggested Agent Focus**: J (meta + enforcement), D (BHS audit of whether pivots are real or theater), E (synthesis of pivot outcomes).

---

### Phase 3: First Real / Controlled SIP Prototypes (Status: Core Blocker — Primary Workstream)

**Objective**: Achieve the first non-trivial, guarded insertion of a Shim (or shim-like signal) into a real production seam (highest signal: VectorSteerer in tts_pipeline.py or post-chelation / variance decision surface in antigravity_engine.py), with full before/after evidence, rollback, and BHS review.

**Key Deliverables / Evidence Required**:
- At least one thin, production-path SIP (or very close proxy) that compiles/runs in the real module.
- Before/after runtime numbers on a real or near-real fixture (not just the synthetic collapse benchmark).
- Full rollback proof.
- Independent (or high-quality simulated Tier B) BHS review of that specific change scoring ≥ 65–70.
- Honest disclosure that this was done under Operator Override or after specific debt mitigation.

**Current Status**: 0% complete (unchanged post Sustained Round 01/02/03). This is the single largest open item (SHIM-CD-01) and the reason for the BLOCKED flag. Round 01 + all prior + R02 + R03 A/B/G/I/C/D/J: 0 real SIP wiring or prod evidence (exhaustive grep + D/J/C 20_ + json confirm tts/antigravity/etc "Wired? NO" only; exactly 2 research files). Phase 2/5 proxy work (synthetic, deeper in R03) explicitly pivoted because of this blocker per plan:218-223 + driver + protocol. No change to 0%. R03 C/D/J confirm 0 SIPs/0 prod deltas/0 closure.

**Primary Risks/Blockers**:
- BLOCKED flag + critical SHIM-CDs.
- Research-only guard + "do not import until BHS promotion".
- 5-vs-10 fidelity issues in the loop itself.
- Risk of L9/L13 if we claim progress without real evidence.

**Suggested Agent Focus** (only when override is active or debts have been mitigated):
- A (final seam audit + clearance or explicit risk bounding).
- B (actual implementation — highest risk, must be extremely narrow and guarded).
- C (real or near-real evidence + rollback).
- D (adversarial review of the specific change).
- J (process audit of whether the override + pivot discipline was followed).

**Note**: Work on this phase should normally be the highest priority when conditions allow. When blocked, the loop must explicitly pivot (see Phase 2) rather than spin in verification.

---

### Phase 4: Measurable Substrate Evidence on Real or High-Fidelity Seams

**Objective**: Move beyond synthetic-only evidence. Demonstrate that shims + MinMax signals + MTP produce measurable, attributable improvement on paths that are either real production code or extremely high-fidelity proxies.

**Key Deliverables**:
- At least one experiment (even if still behind a research flag) showing statistically or practically significant deltas on a real seam or near-real fixture.
- Token accounting or collapse metrics that are not just from the synthetic benchmark.
- Clear separation of "shim-attributable" effect vs baseline.

**Current Status**: Almost entirely synthetic. This is the natural successor to Phase 3.

**Primary Risks/Blockers**: Depends heavily on Phase 3 progress. Without at least a thin real SIP or very high-fidelity hook, this phase is difficult to make credible.

---

### Phase 5: OPSD Trace Integration & Precomputed Shims

**Objective**: Move from pure synthetic traces to using (or closely emulating) privileged OPSD/EGGROLL population-search traces as training signal for shim cascades, precomputed shims, and policy heads.

**Key Deliverables**:
- High-quality synthetic privileged trace generator that mimics the structure and statistics one would expect from real OPSD runs.
- At least one experiment showing that training on these traces produces better MTP predictors or precomputed shim sets than generic synthetic data.
- Interface and data format defined for when real privileged traces become available.

**Current Status**: Basic synthetic trace generation exists (Cycle-011 G + Sustained Round 01 G: outcome_variance param + seeded jitter at harness:1147+ addressing 19_ zero-variance diagnosis; I: synthetic_eval_on_gtraces:737+ consuming variance for corr/pearson/spearman stats + ablation + "L3 mock / 0 real head"). **Sustained Round 01 proxy deltas**: succ_std 0->~0.014 (enables nonzero corr surface |r|~0.2-0.41 vs nan pre); hit/prec minor n=30 lift but drops n=60; ablation_deltas=0.0 (no demonstrated value); n/seed-unstable per C json + D/J adversarial. **0 experiment showing "training on these traces produces better MTP predictors"** (plan:145 key deliverable unmet; no training loop; synthetic L3 only). **Sustained Round 02 (2026-05-27T15:27:25-04:00)**: Deepened proxy (G generate_variance_swept_traces 1615+ + training_signal_simulator 1681+ polyfit stub; I consumption + pw ~-0.75 robust matrix + corr lift + ablation extended; C multi-var/multi-seed/multi-n/train-on/off smokes + consolidated json; all cite "plan:145 unmet beyond L3 proxy"; rank signal nonzero vs degenerate baseline but MSE small ~1e-4 unstable on toy; ablation=0; no real training loop/head/OPSD). **0 experiment showing "training on these traces produces better MTP predictors"** (plan:145 key deliverable still unmet beyond L3 proxy per A/G/I/C/D/J + E summary). **Sustained Round 03 (2026-05-27T16:27:27-04:00)**: Deeper proxy on R02 sub (B deeper [0.0-0.75] ridge lstsq training expt 1686+ + "better predictor" win MSE/rank/hit/prec lift vs R02 poly stub + degenerate; G deeper variance-swept 0.75 + B expt consumption + succ_std 0.0367@0.75; I extended consumption + deeper 5seed/v 0.75 matrix + win deltas 0.5-1 vs R02 + resilience integration; C comprehensive smokes + consolidated json; all cite "plan:145 progress: deeper proxy win structure/res 0.02/True/succ_std scaling but MSE~1e-4 unstable/ablation=0/no real MTP win on R02 sub; still unmet beyond L3 proxy"; rank signal nonzero vs degenerate but toy/small/unstable; ablation=0; no real training loop/head/OPSD). **0 experiment showing "training on these traces produces better MTP predictors"** (plan:145 key deliverable still unmet beyond L3 proxy per A/B/G/I/C/D/J + E summary + harness 897/3027+/3282+). Phase 2/5 pivot + 59 L3 embeds + 6/10 fidelity + L9 theater audited in R03 (J/D/plan:83-85). 0 substrate.

**Primary Risks/Blockers**: Medium. Can be advanced in parallel with Phase 3/4 as long as it stays research-only.

---

### Phase 6: MTP + Micro-SLM Policy Loop Closure

**Objective**: Close the loop between cheap signals (MinMax + chelation variance), MTP lookahead, shim cascades, and a learned policy head (2-4GB class) that can propose reroutes / shim activations.

**Key Deliverables**:
- A coherent input feature representation that combines MinMax scores, usage stats, chelation variance, and MTP predictions.
- At least one trained (or convincingly trained-on-synthetic) policy sketch that outputs useful reroute / cascade decisions.
- Evaluation showing that the policy + MTP + shims compound better than any subset alone (on synthetic or high-fidelity data).

**Current Status**: Sketches and de-mocks exist (H and I work). No closed loop yet.

**Primary Risks/Blockers**: Medium-High. Requires progress in Phases 3–5 to have credible training signal and evaluation.

---

### Phase 7: Full SE-RDAG + Chelation Integration Experiments

**Objective**: Treat Shim Nodes as first-class citizens inside the broader SE-RDAG and chelation decision surfaces. Run experiments (still guarded where necessary) that show how shims interact with existing chelation, VectorSteerer, Model-Scope, block_graph, etc.

**Key Deliverables**:
- Concrete (even if partial) integration points or wrapper patterns at the major seams identified in early audits (tts:47-80, antigravity ~2452-2600, etc.).
- Experiments (synthetic or real) showing interaction effects (positive or negative) between shims and existing mechanisms.
- Refined understanding of where shims add unique value vs where they are redundant with existing steering.

**Current Status**: Mostly mapping and audit work done. Very little actual integration experiments.

**Primary Risks/Blockers**: High until Phase 3 has some progress. Can do limited synthetic experiments earlier.

---

### Phase 8: Literature Cross-Pollination & External Ideas

**Objective**: Systematically mine 2025-2026 literature (MiniMax MSA / Quest-style min-max routing, SAE-RSV, LogicRAG, NSA, graph-regularized SAEs, etc.) for ideas that can be adapted into the shim/SE-RDAG/MTP framework, and turn the best ones into concrete research proposals or small experiments.

**Key Deliverables**:
- Living literature map tied to specific seams and primitives in the codebase.
- At least 3–5 high-quality "research experiment proposals" that are ready to be executed if resources / override allow.
- At least one small experiment actually run that was directly inspired by external work.

**Current Status**: Some good mapping work done (F role in Cycle-011 and earlier). Needs to be turned into a living, prioritized artifact and actual experiments.

**Primary Risks/Blockers**: Low. This phase can and should run in parallel with others.

---

### Phase 9: Debt Closure, Promotion Decision, and Loop Termination / Evolution

**Objective**: Reach a clean, well-documented terminal state for the current workstream.

Possible terminal states:
- Successful promotion of at least one real SIP + supporting infrastructure (best case).
- Honest, BHS-justified scope reduction + recommendation for future work.
- Clean termination with full documentation of why the approach did not yield sufficient evidence.

**Key Deliverables**:
- All critical SHIM-CDs either CLOSED with evidence or explicitly and honestly reduced/terminated.
- Final comprehensive BHS review (Tier B/C or equivalent) of the entire body of work.
- Clear recommendation + rationale to the human operator.
- Updated next-session.md / carried debt reflecting the final state.
- Archive or evolution plan for any reusable artifacts (harness, primitives, traces, etc.).

**Current Status**: Far from this phase. This is the natural end state once the earlier phases have run their course or hit diminishing returns.

**Primary Risks/Blockers**: The main risk is never reaching this phase because the loop keeps adding more slices without closing the core debts (exactly the pattern SHIM-CD-09 warns about).

---

## How the Loop Should Use This Phase Plan

- The orchestrator (and especially J + D roles) must regularly map current work against this phase plan.
- When the highest-priority unblocked phase is not Phase 3, the loop should explicitly say "We are in Pivot Mode, working on Phase X because Phase 3 is blocked by Y."
- Progress is measured by movement across phases with supporting BHS evidence, not just by number of cycles or number of artifacts produced.
- The ultimate completion of the loop goal is reaching a clean terminal state in Phase 9, not perpetual operation.

---

**Version History of This Plan**
- 2026-05-27: Initial creation as the synthesized north star after 11+ cycles of repeated failure pattern, incorporating the Pivot Rule, 3-minute timing, Troubleshooting Mode, and Operator Override mechanism. Requested by user to turn the current loop + backlog into a coherent, completable phase plan.

This document now supersedes the scattered backlog items in the goal document as the primary long-term planning artifact for the shim workstream. The per-cycle backlog in the goal document should be treated as short-term slices that serve one or more phases above.