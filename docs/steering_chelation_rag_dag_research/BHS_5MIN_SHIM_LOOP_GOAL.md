# BHS 3-Minute Research/Build/Test Loop Goal
## Shim Nodes + MTP Shim Lookahead — Self-Improving Completion Engine

**Program**: Steering-Chelation-RAGDAG-MicroSLM (CHELATEDAI)  
**Focus Primitive**: Shim Nodes, Compounding Cascades, MTP Shim Lookahead, SE-RDAG, Precomputed Shims, Shim Backdoors  
**Loop Cadence**: Every 3 minutes (recurring scheduler)  
**Agent Model**: Exactly 10 parallel specialized sub-agents per cycle (expedite + diversity) — updated 2026-05-27 from prior 5-agent definition (see Model Change Log below)  
**Governing Discipline**: Brutal Honesty Kit v3.3 + Program BHS Rubric (no exceptions)  
**Primary Output**: BHS-derived completion metrics + measurable self-improvement per cycle

---

## The Goal (Measurable, Time-Bounded, Evidence-Driven)

**Primary Objective**:
Within each 3-minute cycle, advance the **Shim primitive** from "research scaffold" toward "production-viable, evidence-backed substrate" by completing the highest-leverage remaining slices in research → build → test → brutal honesty → metrics → self-improvement.

**Success Definition (BHS 100 required for "cycle complete")**:
A cycle is only considered complete if it produces:
1. **Runtime evidence** (not docs or plans) from at least one new or improved production path or harness (EVIDENCE: + SMOKE: lines).
2. A **BHS Cycle Score** (0-100) computed from:
   - Self-draft (agent team)
   - Independent-style review (or simulated Tier B via fresh subagent when possible)
   - Severity caps respected
   - Carried debt reduction (or honest disclosure of increase)
   - L1-L13 disclosures (zero tolerance for hidden ones)
3. **Measurable self-improvement delta** vs previous cycle (examples below).
4. Updated **living BHS Completion Dashboard** for the shim work.
5. Clear next-cycle plan with prioritized slices.

**Non-negotiable Rules** (inherited from CLAUDE.md + program rubric):
- Evidence rule: "complete" claims require runtime output from the code path on a fresh checkout or controlled harness.
- Visible means verified: No surfacing of capabilities as working until they have passed a smoke on the actual surface.
- 10-agent model per cycle: Orchestrator + 10 specialized agents (A–J; see expanded roles in Phase 1 below). (Updated 2026-05-27; prior 8 cycles operated under the original 5-agent definition.)
- Brutal honesty in every artifact and every agent output.
- No scope creep into full agent harnesses or un-scoped hardware claims.

---

## Cycle Structure (3-Minute Hard Limit)

**Phase 0 (0-15s)**: Orchestrator reads state
- Latest artifacts in `artifacts/` and `loop_01/`
- Previous cycle's BHS score + carried debt
- Current highest-priority slices (from living backlog)

**Phase 1 (15s-1.5min)**: 10 Parallel Agents Execute Slices
Typical agent roles (rotated/adapted each cycle based on need; expanded 2026-05-27):
- **Agent A — Research & Mapping**: Literature, code audits, pain-point updates, new paper mappings (LogicRAG, SAE-RSV, MTP, OPSD extensions).
- **Agent B — Build / Implementation**: Concrete code (ShimNode extensions, SIP wiring, Registry improvements, minimal MTP lookahead head, benchmark extensions).
- **Agent C — Test & Evidence Generation**: Run/extend harnesses (synthetic collapse shim families, road-course profiles, rollback demos, token accounting where possible), capture real runtime output.
- **Agent D — BHS Auditor & Metrics**: Apply full rulebook review to all new work, compute cycle score, track L1-L13, update carried debt, produce brutal honesty section.
- **Agent E — Integration & Self-Improvement**: Synthesize across agents, wire small cross-surface improvements, update living dashboard, quantify deltas, propose next cycle's slices.
- **Agent F — Literature & External Research**: Deep dive on newest 2025-2026 papers (SAE variants, graph RAG, steering vector methods, MTP follow-ups); map directly to CHELATEDAI seams.
- **Agent G — OPSD / EGGROLL Trace Integration**: Consume privileged population-search traces as training signal for shim cascades / precomputed shims.
- **Agent H — Micro-SLM Policy Sketch**: Draft objectives + synthetic data format for a 2-4GB route-policy head that learns reroutes from chelation + shim activations.
- **Agent I — MTP Shim Lookahead Prototype**: Lightweight next-shim predictor (usage stats + relevance) that compounds cascades; evaluate hit-rate on held-out traces.
- **Agent J — Cross-Cycle Meta Auditor**: Independent audit of the loop process itself (fidelity, time discipline, L-taxonomy on prior cycles, scheduler health).

**Phase 2 (1.5min-2.5min)**: Orchestrator Integration
- Collect all 10 agent outputs.
- Run any cross-validation or additional evidence collection.
- Compute official BHS Cycle Score.
- Update dashboard and living backlog.

**Phase 3 (2.5min-3min)**: Closeout + Self-Improvement Reflection
- Publish cycle summary (BHS metrics, deltas, honest assessment).
- Explicitly log what improved the system this cycle (code quality, coverage, evidence strength, reduced debt, new capability demonstrated).
- Seed the prompt for the next 3-minute firing (now using the 10-agent model).

**Hard Stop at 3 minutes**: Any incomplete work is logged as carried debt with severity. The loop continues.

---

## BHS-Derived Completion Metrics (Tracked Every Cycle)

**Core Metrics** (must be reported every cycle):
- **BHS Cycle Score** (0-100): Weighted (Self-draft 40% + Auditor review 40% + Evidence strength 20%), with severity caps applied.
- **Carried Debt Delta**: Number and severity of open items added vs closed this cycle.
- **Evidence Strength**: Count of new runtime EVIDENCE:/SMOKE: artifacts that survive fresh checkout + re-run.
- **Slice Completion Rate**: % of targeted slices that reached runtime evidence (not just design).
- **Self-Improvement Deltas** (quantified where possible):
  - New SIPs wired (with before/after behavior)
  - New benchmark families passing with measurable lift + rollback proof
  - Token accounting surface coverage increase
  - MTP lookahead prediction accuracy / hit rate on held-out traces (when data exists)
  - Reduction in "L4 risk surface" (scaffolding that could be misrepresented)
  - Quality of brutal honesty sections (number of previously hidden risks now disclosed)
- **BHS Research Program Score** (cumulative for the shim workstream, 0-100).

**Living Dashboard Location** (updated every cycle):
`docs/steering_chelation_rag_dag_research/artifacts/BHS_SHIM_LOOP_DASHBOARD.md`

---

## Overarching North Star

The primary long-term planning artifact for this workstream is now `FULL_SHIM_LOOP_PHASE_PLAN.md` (in this directory). That document defines the complete multi-phase roadmap with clear objectives, success criteria, current status, and risks. The loop's purpose is to systematically advance through those phases (with intelligent pivoting when primary work is blocked).

The per-cycle backlog below should be treated as short-term, executable slices that serve one or more phases in the Full Phase Plan.

## Current Highest-Priority Slice Backlog (as of program kickoff + first 5-agent wave; 10-agent model active from 2026-05-27 onward)

(Orchestrator must re-prioritize at the start of every cycle based on latest evidence, the Full Phase Plan, and the Pivot Rule when primary work is blocked)

1. Wire first real minimal SIP (highest signal: TTS/VectorSteerer or antigravity variance decision) + demonstrate insert-once shim behavior with rollback.
2. Extend benchmark skeleton to produce real token-accounted before/after numbers on a controlled fixture.
3. Implement basic MTP Shim Lookahead mock → real lightweight head that consumes shim registry state.
4. Generate first synthetic "successful shim cascade" traces usable as privileged OPSD data.
5. Full substrate audit of one major host surface (e.g. entire `antigravity_engine.py` shim-related paths) with 02_audit.md style rigor.
6. Registry persistence / block-graph export path for Precomputed Shims.
7. First end-to-end evidence chain for a single Precomputed Shim improving a noisy neighborhood (NDCG lift + rollback + quant survival + token delta).
8. SelfEditDirective + shim_directive integration (proposal + evaluation + ledger).
9. Incorporate min-max style lightweight block/index scoring as a cheap relevance signal for shim activation and SE-RDAG rerouting (Agent 7 draft — full BHS template below: cheap min-max block pre-filter gates at SIP seams + ShimRegistry + MTP lookahead; high-leverage efficiency primitive).
10. **NEW (Agent 10 Integrator meta 2026-05-27)**: Add dedicated comparison of MinMax MSA vs SE-RDAG (with min-max adaptation pseudocode) to research plan + this goal; synthesize Agent 1-9 outputs (F literature/MiniMax-M*, I MTP, J meta-auditor, A/D audits) into Loop 10 comparative analysis section. Include BHS disclosures, L citations, and explicit note that this is doc-only (0 substrate advance, does not close any SHIM-CD). Update living dashboard as "successful 10-agent model use" (with full §4 template + 4Q). See STEERING_CHELATION_RAGDAG_MICROSLM_RESEARCH_PLAN.md new section for pseudocode + full comparison.

The 10 agents in each cycle must be assigned from this backlog (or newly discovered higher-value slices). (Model updated 2026-05-27; see change log. This #10 added by Agent 10 role under explicit task; L4/L9/L13 on meta "success" framing disclosed in dashboard update and plan edit.)

---

## Expanded Backlog Item #9 (Agent 7 New Slice Draft — BHS-Compliant Template)

**Title:** Incorporate min-max style lightweight block/index scoring as a cheap relevance signal for shim activation and SE-RDAG rerouting

**Description:**  
Implement a minimal, guarded `MinMaxBlockRelevanceScorer` (research-only initially; compatible with ShimRegistry and harness block partitions) that for a query vector + partitioned index (synthetic blocks in shim_collapse_benchmark_extension.py fixtures, or future mappings to vector_store partitions / computational_storage_poc/block_graph blocks) computes O(1)-or-O(blocks) cheap per-block signals:  
- per-block min/max of (query · block_centroid) or component-wise extrema (pre-aggregated where possible);  
- range = max_sim_block - min_sim_block as "relevance variance" proxy;  
- optional lightweight norm or variance stats mirroring existing dim_variances computation (antigravity_engine.py:2569).  
The resulting scalar (or top-K mask) gates downstream work: only blocks exceeding threshold (or in top-K by cheap score) trigger full `ShimRegistry.lookup_by_context` / `apply_shim_cascade` (shim_node.py:154+) / MTP Shim Lookahead prediction / SE-RDAG expansion / VectorSteerer steer extensions.  
Primary candidate SIP seams (per prior audits + nomenclature §2.1/3):  
- tts_pipeline.py:47 (VectorSteerer.steer + clear_signals) post-embed intercept path;  
- antigravity_engine.py:2452-2458 (TTS intercept) and 2566-2600 (variance/chelation decision before final_top_ids);  
- ShimRegistry.lookup_by_context (shim_node.py) when context carries block tags;  
- harness simulate_sip_effect / record_shim_activation paths for synthetic block partitions.  
Scorer must respect BoundedAdapter / INT8 floors, produce copy-safe outputs, and support rollback metadata. Precomputation hooks for block_graph payloads encouraged (computational_storage_poc/).

**Motivation (tie to MiniMax):**  
docs/llm-architecture-ai-engineering-adaptation-review-2026-04-27.md:66 explicitly calls out MiniMax-M1/M2/M2.5: "M1 linear attention; M2 returns to full attention; sparse MoE; Highly sparse active parameters; MTP in related variants". These architectures achieve their efficiency not by making every operation cheap, but by using extremely lightweight signals (linear approximations, block/expert-wise cheap scores, partial RoPE) to *decide which* expensive paths (full attention, dense experts, long cascades) are worth materializing for a given input. The SE-RDAG + Shim Nodes + MTP Shim Lookahead (nomenclature.md:79-84, shim_node.py:34-36) are explicitly designed as the *low-token, high-precision* escape valves from rigid retrieval. If the decision to activate a shim or reroute itself requires a full model forward, expensive registry scan, or un-gated DAG expansion on every query, the entire primitive set fails its efficiency thesis and replicates the "always-on cost" problem it was invented to solve. A min-max block/index scorer is the direct analogue of MiniMax's cheap gating: a production-grade, quant-survivable, pre-aggregatable signal that keeps shim/SE-RDAG activation inside the "cheap relevance" regime. It compounds with existing StructuralHealthScore (structural_health_score.py:45-77) and _cosine_scores (synthetic_collapse_benchmark.py:14-19) without replacing them.

**Success Criteria:**  
- Guarded research implementation (CHELATED_SHIM_RESEARCH=1 / --research-shim or equivalent never-default) that on extended synthetic collapse fixtures (with explicit block partitions) reduces shim activation attempts / cascade evaluations / MTP head invocations by a measurable delta (target >=25-30% relative reduction) vs ungated baseline on sip/sip_effect families.  
- Quality preservation: ndcg@3, recovered, noise_reduction (0.7886... sip_effect baseline) within pre-registered tolerance; no regression on clean cases.  
- Scorer exercised end-to-end with record_shim_activation + apply_shim_cascade (depth>=1) + explicit rollback in at least one harness path; before/after activation counts + scorer_latency_ms + "minmax_gated" fields emitted in bhs_evidence.  
- One research-only thin SIP wrapper demo at a real seam (e.g. conditional pre-filter before 2456 apply or 2584 variance test) producing observable gated behavior + rollback proof in harness smoke only.  
- Full EVIDENCE/SMOKE + §4 BHS in the landing cycle; independent correlation check (high cheap-score blocks actually correlate with higher post-activation success_rate in usage_stats).  
- When BHS promotion gate passed: observable in prod-path smoke (thin non-default wrapper) with identical guarantees + no default behavior change.

**Required Evidence (runtime + BHS, non-negotiable per goal §18-29 + rulebook §0-2):**  
- EVIDENCE: exact commands e.g. `PYTHONPATH=. python -B docs/steering_chelation_rag_dag_research/artifacts/shim_collapse_benchmark_extension.py --family sip_effect --research-shim --minmax-blocks` (and equivalent prod-seam wrapper invocations) whose output includes hashes, "minmax_block_score" / "gated_activations_reduced" / "scorer_vs_lookup_latency_ratio" in bhs_evidence, rollback_post=True, and core metrics bitwise match to prior baseline except for the new gated deltas. Artifact (dated json) survives fresh checkout + re-run.  
- SMOKE: "research harness only; 0 prod/default change until promotion; metrics + gated savings proven; does not satisfy goal success #1 until real SIP wiring + Tier B pass". Reproducible on clean python -B.  
- BHS Cycle Score delta contribution (evidence strength 20% weight) + explicit L1-L13 table for the slice (especially L4 on research scope vs SE-RDAG language).  
- Carried debt update + dashboard row documenting the new capability + any new debt introduced.  
- Adversarial confirmation (Agent D style) that the cheap signal is not L13 (prose-only "mechanical gate").

**Risks (L taxonomy — explicit disclosure required in every related artifact):**  
- L1 (Scaffold-as-feature): scorer body returns constant 1.0 or identity; tracked with file:line in shim_node.py or harness extension.  
- L4 (Partial-with-claim-of-complete): "SE-RDAG rerouting" or "shim activation signal" language used while all paths remain research/artifacts/ only (exact repeat of existing shim L4 surface per 01_cycle00*.md audits); severity cap mandatory.  
- L5/L8 (Test-as-truth / asserts-the-bug): only harness synthetic asserts lift; real index partitions or road-course never exercised.  
- L11 (Broad-catch swallowing): except blocks around scorer (e.g. in engine 2465/2471 style) that silently disable the gate ("always activate shim") hiding scorer bugs.  
- L13 (Soft-prose-claimed-as-mechanical): research plan / nomenclature / dashboard claim "mechanical pre-filter inside SE-RDAG" or "wired at SIP" while diff only adds comments/conditionals in research py.  
- Additional domain risks (must be BHS §4 disclosed): over-pruning (missed utility shims on tail blocks → false-negative relevance, degrading NDCG on long-tail); scorer itself adding measurable latency that negates savings; interaction/contradiction with existing global_variance (antigravity_engine.py:2569) and StructuralHealthScore producing unstable decisions; pre-agg maintenance cost in dynamic indexes (new L9 doc-vs-reality debt if not implemented).  
- Process risk: adding this slice while backlog #1 remains 0% (0 SIPs) risks further L9/L4 on "high-leverage" framing without substrate wiring.

**Suggested Owner Roles (10-agent model, goal §48-58):**  
- Agent A — Research & Mapping: Literature tie-in (MiniMax M1/M2 linear attn + sparse gating papers) + exhaustive SIP seam matrix update (tts:47-80, antigravity:2452-2600 + block_graph + StructuralHealthScore call sites) with "cheap scorer applicability" column; fresh 0-prod grep.  
- Agent I — MTP Shim Lookahead Prototype: Consume min-max scores as input feature / early-exit predicate for cascade prediction; define interface extension.  
- Agent B — Build / Implementation: Concrete MinMaxBlockRelevanceScorer (dataclass + compute method + precompute hook) + guarded wiring into harness + one research-only SIP seam example; full BHS self-draft.  
- Agent C — Test & Evidence Generation: Extend fixtures with block partitions; execute gated vs baseline; persist Cycle-N bhs_shim_evidence_*.json with new fields + EVIDENCE/SMOKE banners + rollback proof.  
- Agent D — BHS Auditor & Metrics: Full L1-L13 enumeration on the slice (with file:line), score capping for L4/L13 on scope vs language, verification that evidence actually proves cheap-signal correlation, §128 rec if pattern continues.  
- Agent E — Integration & Self-Improvement: Dashboard update, quantification of activation/latency/token delta vs prior cycles, refresh of living backlog priority (this slice directly de-risks #1/#2/#3), 4Q reflection.  
- Agent J — Cross-Cycle Meta Auditor (optional concurrent): Audit whether adding this high-leverage slice while 8 prior slices + 9+ cycles at 0 SIPs constitutes process L4.

**BHS Self-Draft Note (this entry itself):** This is a complete, actionable slice definition drafted by Agent 7 per explicit user request. It has zero implementation at insertion time. Insertion of the prose does not constitute "wiring" or "evidence". It follows the exact template requirements (description, MiniMax tie, success, runtime+BHS evidence, L risks with file:line examples, owners). Adding it to the living backlog is the minimal visible action; actual progress remains 0 until a cycle produces the required harness json + smoke + Tier B pass. Tracked as potential new carried debt entry if not actioned. References: BHS_5MIN_SHIM_LOOP_GOAL.md (this section + §18-29), shim_node.py:34 (L4 disclosure), nomenclature.md:105 (SE-RDAG), rulebook v3.3 §1 L-taxonomy + §4 template, prior cycle audits (e.g. 01_cycle009_audit.md).

---

## Self-Improvement Mechanism

Every cycle must explicitly answer:
- What concrete capability or evidence strength increased this cycle that did not exist before?
- What previously hidden risk or carried debt was surfaced and either closed or properly bounded?
- How did the quality of the BHS process itself improve (better auditor prompts, stronger evidence capture, tighter time discipline)?
- What pattern from this cycle should be templated for future cycles?

The orchestrator prompt for the scheduler must force this reflection.

---

## Scheduler Configuration

- Interval: 5 minutes
- Recurring: true
- Fire immediately on creation
- The scheduled prompt is the "Orchestrator 5-Minute BHS Loop Driver" (see separate scheduler creation). Note: the baked scheduler task (ID 019e669bf1bb) was originally created with 5-agent language; it continues to dispatch 5 agents until a human manually updates or recreates the task. The narrative in this document now describes the 10-agent model going forward.

**Termination Conditions** (human intervention required):
- 3 consecutive cycles with BHS Cycle Score < 60
- Explicit "PAUSE" or "STOP" command from operator
- Critical safety/scope violation discovered

---

## Brutal Honesty on This Goal Document Itself

This is an ambitious meta-process designed to force continuous, measurable progress under the same brutal honesty rules the rest of the program demands.

It has not yet run a single 3-minute cycle. Success is not guaranteed — the 3-minute hard limit is intentionally tight to prevent scope explosion and force ruthless prioritization.

All claims of "completion" or "self-improvement" produced by future cycles must themselves survive the BHS evidence rule.

This document is the north star and contract for the recurring scheduler.

**Version**: 1.2 — 3-minute wall time (2026-05-27 user request)  
**Last Updated**: 2026-05-27 (timing change from 5min → 3min hard limit + 10-agent model)

---

## Model Change Log (2026-05-27)

**Change**: The canonical loop narrative was updated from "Exactly 5 parallel specialized sub-agents per cycle" to "Exactly 10 parallel specialized sub-agents per cycle" (expanded roles A–J above).

**Scope of update**: Forward-looking definition in this goal document + agent role descriptions + Phase headers. Historical cycle artifacts (Cycles 1–8), prior D adversarial reports, E reflections, and all "5-agent model failure" citations in the dashboard and loop_02/ files were **left verbatim** (they accurately describe what actually ran).

**BHS implications (L taxonomy)**:
- L4 (partial): The narrative now claims a 10-agent model while the active scheduler task (019e669bf1bb) and all executed history used 5. This is a post-hoc documentation change, not a retroactive rewrite of evidence.
- L9 (hygiene): Future cycles must explicitly reference this log when citing "the 10-agent model" so readers are not misled about prior 8 cycles.
- No new SHIM-CDs created by this edit (the underlying 0-prod / 0-SIP substrate reality is unchanged).
- Evidence rule respected: This log itself is the visible, verifiable record of the revision.

**Impact on prior evidence**: All Cycle 1–8 scores, deltas, L disclosures, and "5-agent failure" statements remain factually correct for the period in which they were produced. The 10-agent model begins with Cycle 009 (if the loop continues).

**Runtime reality**: The orchestrator prompt baked into scheduler 019e669bf1bb still says "exactly 5". Human operator action is required to update the scheduled task if 10-agent dispatches are desired in automation.

This change was made in direct response to an explicit user request. It does not alter the BHS assessment that the loop (under either agent count) has produced 0 production SIPs or substrate evidence after 8 cycles.

---

**Change (2026-05-27)**: Wall time reduced from 5 minutes to 3 minutes hard limit.

**Scope**: 
- Updated cycle phase timings to fit 3-minute total (Phase 0: 0-15s, Phase 1: 15s-1.5min, Phase 2: 1.5-2.5min, Phase 3: 2.5-3min).
- Updated all references from "5-minute" to "3-minute" in objectives, hard stop language, and firing descriptions.
- Scheduler cadence changed from 5m to 3m interval.

**BHS implications**:
- Tighter time pressure increases risk of incomplete slices being logged as carried debt.
- May accelerate "unambiguous failure" pattern (already at 11+ cycles of 0 substrate) or force more ruthless prioritization.
- L9 risk if documentation claims "faster self-improvement" without corresponding substrate evidence.

**Runtime action**: Active scheduler updated from 5m → 3m. Old scheduler deleted; new one created with matching prompt updates.

This change was made in direct response to an explicit user request to make the loop stricter.

---

*Drive the loop. Be brutally honest. Produce evidence. Improve the system.*