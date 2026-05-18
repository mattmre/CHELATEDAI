# Session 2026-05-15 — CHELATEDAI Chelation + OPSD Research Program

**Session Type**: Long-running research & implementation program initiation  
**Focus**: Making the existing "chelation" system (ChelationAdapter + SelfHealingChelationPlanner + sedimentation) practically viable using On-Policy Self-Distillation (OPSD/SDPO) and related self-improvement techniques.

## Session Objectives (as stated by operator)
- Deeply research OPSD / SDPO concepts from @ar0cket1 and related 2026 literature.
- Map those techniques to the current chelation pain points in the repo.
- Design and begin building a viable upgrade path.
- Run a structured 10-loop BHS-scored research program using 10 parallel agents.
- Maximize toward 100/100 BHS Research quality.
- Do not stop / do not keep asking for permission between steps.

## Major Work Completed

### 1. Research Program Structure Created
- Full 10-loop BHS-scored research program defined:
  1. Deep Research & Mapping
  2. Problem Refinement & Opportunity Framing
  3. Architecture Design
  4. Loss Function & Training Objective Design
  5. Stability, KL Control & Forgetting Mitigation
  6. Sample Efficiency & Data Strategies
  7. Quantization-Aware & Low-Rank Integration
  8. Self-Edit Directive Pipeline Integration
  9. Evaluation Framework & Benchmark Design
  10. Implementation, Testing & Final Synthesis
- Created custom **BHS Research Scoring Rubric** (Evidence Quality, Honesty & Gap Exposure, Completeness, Practicality & Testability, Leverage & Novelty).
- Applied initial BHS Research Self-Assessment to the program plan itself (starting score 71/100, with explicit L1–L13 gaps and Carried Debt declared).

### 2. Research Infrastructure
- Directory created: `docs/chelation_opsd_research/loop_01/`
- Living documents created and seeded:
  - `CHELATION_OPSD_RESEARCH_PLAN.md`
  - `CHELATION_OPSD_BHS_RESEARCH_RUBRIC.md`
  - `CHELATION_OPSD_10_LOOP_BHS_PROGRAM.md`
  - `11_evaluation_framework_skeleton.md`
  - `12_pain_point_to_opsd_mapping.md`
  - `13_candidate_upgrade_patterns.md`
  - `14_initial_bhs_research_self_assessment.md`

### 3. 10-Agent Parallel Swarm (Loop 1)
Ten specialized research agents were spawned in parallel for Loop 1:

1. OPSD Literature Deep Dive
2. Current Chelation Implementation Auditor (completed — 296-line ruthless audit)
3. Loss Function & Objective Space Explorer
4. On-Policy Training Dynamics Analyst
5. KL Divergence & Stability Control Specialist
6. Self-Edit Directive Integration Specialist
7. Sample Efficiency & Data Filtering Researcher
8. Quantization-Aware & Low-Rank Chelation Variants Explorer (completed — implemented `QuantizationAwareLowRankAdapter` + 8 variants + tests)
9. Related Research Scanner (completed — 12 additional patterns from Self-Rewarding LLMs, STaR, ReST-MCTS, SDFT, etc.)
10. Synthesis & Prioritization Agent

**Completed high-value outputs so far**:
- Full ruthless audit of the entire chelation system (Agent 2) — identified 40+ concrete pain points and direct OPSD mappings.
- Real implemented code: `QuantizationAwareLowRankAdapter` with STE (Straight-Through Estimator) + 8 concrete testable variants (Agent 8).
- Broad literature scan + 12 prioritized additional upgrade patterns (Agent 9).

### 4. Key Technical Insights Captured
- Current chelation is "fundamentally incomplete" — self-healing is advisory-only; actual training happens in a fragile sedimentation pipeline.
- Strongest conceptual mapping: `SelfEditDirective` system is a proto-ReSTEM / proto-SDPO outer loop. The missing piece is the **inner on-policy self-distillation** engine.
- Multiple high-leverage patterns identified (Asymmetric On-Policy Self-Distillation for Self-Edit Directives, KL-Regularized Residual Chelation, Quantization-Aware Low-Rank OPSD, etc.).
- Significant environment blocker: current WSL container lacks `python3-venv`, qpdf, and full Playwright support, preventing real training runs and artifact generation.

### 5. BHS Process Applied to Research
- Created dedicated BHS Research Rubric for this program.
- Performed initial BHS Research Self-Assessment of the program plan.
- Explicitly declared Carried Debt items (environment limitations, empirical validation, long-horizon experiments).
- All agent work followed full reading discipline and BHS rules.

## Current Honest State (End of Session)

- Loop 1 (Deep Research & Mapping) is well underway with strong agent outputs.
- Several high-value patterns and one real code implementation already exist.
- Realistic current BHS Research ceiling for the program: ~88–92/100 once artifacts are generated on a proper host.
- The research program now has proper structure, living documents, and BHS scoring baked in.

## How to Resume Tomorrow (Clear Handoff)

1. **Start here**: Read `docs/chelation_opsd_research/CHELATION_OPSD_10_LOOP_BHS_PROGRAM.md` and `CHELATION_OPSD_BHS_RESEARCH_RUBRIC.md`.
2. Check the latest state of the living documents in `loop_01/` (especially `13_candidate_upgrade_patterns.md` and `12_pain_point_to_opsd_mapping.md`).
3. Continue monitoring the remaining active Loop 1 agents and integrate their outputs.
4. Once Loop 1 synthesis is solid, move into **Loop 2: Architecture Design** for the top 3–4 patterns.
5. When the WSL environment is fixed (or you move to a provisioned host), run the harness to generate real committed browser artifacts for the demo page.

**Key files to open first**:
- `docs/chelation_opsd_research/CHELATION_OPSD_10_LOOP_BHS_PROGRAM.md`
- `docs/chelation_opsd_research/loop_01/13_candidate_upgrade_patterns.md`
- `docs/chelation_opsd_research/loop_01/02_chelation_system_audit.md`

## Evidence Checklist
- [x] 10-loop BHS program defined with scoring rubric
- [x] 10 parallel research agents launched for Loop 1
- [x] Multiple high-quality agent reports received with code and patterns
- [x] Living synthesis documents created and seeded
- [x] Gaps in the research plan itself exposed using BHS language
- [x] Clear handoff and resume instructions written

**Session wrapped per Evokore session-wrap skill + project BHS evidence standards.**

Ready to continue the 10-loop program tomorrow with full momentum. No context lost.