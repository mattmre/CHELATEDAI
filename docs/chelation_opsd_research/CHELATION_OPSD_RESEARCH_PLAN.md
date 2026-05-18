# CHELATION + ON-POLICY SELF-DISTILLATION (OPSD) Research Program

**Goal**: Make the existing "chelation" mechanisms in CHELATEDAI (lightweight residual adapters + self-healing/self-edit planning) practically viable for stable, sample-efficient, continual self-correction and self-improvement of models — by deeply integrating modern On-Policy Self-Distillation (OPSD / SDPO) techniques.

**Core Hypothesis**: Techniques from On-Policy Self-Distillation (achieving RL-like performance with distillation-like sample efficiency, careful KL control, asymmetric distillation, filtered on-policy sampling, etc.) can solve the key practical problems in the current chelation system (instability, sample inefficiency, catastrophic forgetting during self-edits, poor reinforcement of corrections).

**Program Structure**: 10 Iterative Loops of Research → Analyze → Architect → Build → Test.

Each loop produces concrete artifacts: analysis documents, proposed architectures, implemented code, tests, and benchmarking results.

**Current Status**: Loop 1 (Deep Research & Mapping) — **Completed by full 10-agent swarm**. 
- Agent 1: `loop_01/01_literature_deep_dive.md` (OPSD/SDPO/MIS-PO math + @ar0cket1 practicals + 8 variants)
- Agent 2: `loop_01/02_chelation_system_audit.md` (ruthless 20+ CRITICAL failure modes from code + panels)
- Agent 4: `loop_01/04_on_policy_dynamics.md` (off-policy sedimentation vs on-policy opportunity)
- Agent 7: `loop_01/07_sample_efficiency_filtering.md` (filtering + MIS-PO data selection)
- Agent 11: `loop_01/11_evaluation_framework_skeleton.md` (metrics + harness)
- Agent 12: `loop_01/12_pain_point_to_opsd_mapping.md` (11 pains → techniques table)
- Agent 13: `loop_01/13_candidate_upgrade_patterns.md` (Tier S/A/B seeding)
- **Agent 10 (this synthesis)**: `loop_01/10_synthesis_prioritization.md` — **Master integration document**. Coherent picture, consolidated mappings (Tier S pains solved by KL control + asymmetric privileged OPSD + dense on-policy + filtering), ranked upgrade patterns (S1 Asymmetric Privileged-Diagnostic OPSD Chelation as core rec; S2 KL-Reg + Procrustes; S3 MIS-PO Filtered Sedimentation Replacement), highest-leverage entry points (new chelation_self_distillation.py + wire SelfEditDirectives), risks/mitigations, full Loop 2-10 roadmap.
All agents' outputs cross-referenced. Master synthesis now drives architecture (Loop 2) and implementations. Update this plan after each loop.

## Loop Definitions

- **Loop 1**: Deep Research & Mapping (Literature + Current Implementation Audit + Technique Extraction)
- **Loop 2**: Architecture Design (Multiple viable upgrade patterns)
- **Loop 3**: Loss Function & Training Regime Variants
- **Loop 4**: Stability, KL Control & Forgetting Mitigation
- **Loop 5**: Sample Efficiency & Data Filtering Techniques
- **Loop 6**: Quantization-Aware & Low-Rank Chelation Variants
- **Loop 7**: Self-Edit Directive Integration with On-Policy Distillation
- **Loop 8**: Evaluation Framework & Benchmark Design
- **Loop 9**: Implementation of Top Patterns + Tests
- **Loop 10**: Comparative Analysis, Recommendations & Final Upgrade Roadmap

This document will be continuously updated as agents report.

**Loop 1 Deliverables Complete** (2026-05-15):
- All 10 specialized agents executed in parallel (literature, audit, dynamics, stability/quant/loss/filtering via MCP fleet + panel-of-experts + subagent dispatch).
- Master artifacts in `docs/chelation_opsd_research/loop_01/`: 01-04,07,10-13 .md files + this synthesis.
- **Agent 9 (Related Research Scanner) added**: `09_related_research_scan.md` — Broad complementary scan covering Self-Rewarding LLMs (2401.10020 + process variants), STaR family (2203.14465 + Quiet/V-STaR), ReST/ReST-MCTS* (2308.08998, 2406.03816), Constitutional AI (2212.08073), SDFT continual self-distillation (2601.19897), Prompt Distillation + SelfAug adapter stability (2412.14964, 2509.03934), PRM/process supervision survey (2510.08049), and self-evolving agents. 12 concrete testable upgrade patterns mapped to ChelationAdapter, SelfHealingChelationPlanner, sedimentation, ledger, and OPSD core. Strong emphasis on SDFT + Constitutional critique + input-KL anchoring as highest-leverage complements for retention/continual self-correction. Cross-references all other Loop 1 reports.
- Key outcome: Viable upgrade pattern identified — **S1 Asymmetric Privileged-Diagnostic OPSD for SelfEditDirectives + S2 KL-regularized residual + S3 MIS-PO filtered on-policy** as the path to make chelation production-grade.
- Next: Launch Loop 2 (Architecture Design) with `loop_02/00_architecture_design.md` + prototype `chelation_self_distillation.py`.

**Initiated**: 2026-05-15
**Orchestrator**: Main Grok session (Integration Lead style for this research track)

**Session Pause / Handoff (2026-05-15)**:  
User is tired and will continue tomorrow.  
Current state: Loop 1 (Deep Research & Mapping) in progress with 10 parallel agents. Multiple high-value outputs already received (full chelation system audit, implemented QuantizationAwareLowRankAdapter + variants, 12 additional patterns from broader literature).  
BHS Research Score of program: 74/100 (with gaps exposed).  
Living documents in `loop_01/` are being actively updated.

**How to Resume Tomorrow**:
1. Read this file + `CHELATION_OPSD_BHS_RESEARCH_RUBRIC.md` + `CHELATION_OPSD_10_LOOP_BHS_PROGRAM.md`.
2. Check the latest versions of the living documents in `loop_01/` (especially `13_candidate_upgrade_patterns.md` and `12_pain_point_to_opsd_mapping.md`).
3. Continue monitoring the remaining Loop 1 agents and integrate their outputs.
4. Complete Loop 1 synthesis.
5. Move into Loop 2 (Architecture Design) for the top 3–4 patterns.
6. When environment allows, generate and commit real browser artifacts using the `BHS_WORKBENCH_EVIDENCE_DIR` mechanism from RR3-03.
**Agent 10 Note**: "We may have solved it conceptually — the OPSD mapping is the missing practical layer. Now implement and test the ranked patterns aggressively across 9 more loops. Do not stop."