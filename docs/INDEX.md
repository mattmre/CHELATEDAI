# Documentation Index

## Core Documentation

| Document | Description |
|----------|-------------|
| [Docs Home](README.md) | Canonical starting point plus legacy-to-canonical doc map |
| [Repository Overview](../README.md) | High-level repo summary, quick start, and validation commands |
| [System Blueprint](SYSTEM_BLUEPRINT.md) | Architecture, stack, CI surfaces, and information-flow diagrams |
| [Module Guide](MODULE_GUIDE.md) | Module-by-module inventory across runtime, evaluation, and storage POC files |
| [Research Tracks](RESEARCH_TRACKS.md) | Current research themes, maturity, and open questions |
| [Evolution Strategies Hyperscale Comparison](evolution-strategies-hyperscale-chelatedai-analysis.md) | Comparison of the EGGROLL / Evolution Strategies hyperscale paper with ChelatedAI tuning, quantization, and storage-node strategy |
| [Self-Adapting Chelation: SEAL + EGGROLL Adaptation](self-adapting-chelation-seal-eggroll-analysis-2026-04-28.md) | Mapping of SEAL self-edits and EGGROLL low-rank ES into ChelatedAI's self-healing chelation implementation |
| [SEAL + EGGROLL Multi-Panel Architecture Review](seal-eggroll-multipanel-architecture-2026-04-28.md) | Multi-panel synthesis, claim boundary, and implementation sequence for self-healing adapter-only chelation |
| [Current EGGROLL/ChelatedAI Research Validation](current-research-eggroll-chelatedai-2026-04-27.md) | Current research scan and design validation for low-rank ES, quantized ZO, adaptive retrieval, and near-data storage scoring |
| [EGGROLL Implementation Expert Panel Review](eggroll-implementation-panel-review-2026-04-27.md) | Multi-panel expert review of missing coverage, risks, and next priorities after the EGGROLL-inspired implementation |
| [EGGROLL Strategic Analysis Plan](eggroll-strategic-analysis-plan-2026-04-27.md) | Three-loop strategic plan for retrieval-native ES, candidate promotion gates, storage-backed fitness, and follow-up platform functionality |
| [LLM Architecture And AI Engineering Adaptation Review](llm-architecture-ai-engineering-adaptation-review-2026-04-27.md) | Review of modern LLM architecture features and practical AI-engineering operations mapped to ChelatedAI subcomponent adaptation opportunities |
| [Live-Fire Diagnostics And Calibration](live-fire-diagnostics-2026-04-27.md) | Deterministic end-to-end diagnostics harness results, known-good value guidance, and next benchmark campaign priorities |
| [Safety Testbed Road-Course Campaign Plan](safety-testbed-road-course-plan.md) | Project-car safety testbed status, default-promotion gate, road-course campaign evidence requirements, and documentation refresh criteria |
| [Road-Course Results And Default Threshold Decision](road-course-results-2026-04-27.md) | Small-model SciFact/NFCorpus road-course evidence supporting the safer `0.01` default chelation threshold guardrail |
| [Golden Default And Autopilot Roadmap](golden-default-roadmap-2026-04-29.md) | Two-day analysis of road-course, six-path validation, masking/reformulation branches, and the next autopilot search path |
| [Qwen-Scope Engine Mapping](qwen-scope-engine-mapping-2026-04-30.md) | Loop-fix summary plus the "Engine-Scope" roadmap for internal feature gating, coverage analysis, and synthetic hard-negative generation |
| [Model-Scope Steering Architecture](model-scope-steering-architecture-2026-05-01.md) | Architecture translation from Qwen-Scope-style model hooking into a bounded ChelatedAI Model-Scope program with segmented memory and overlay-first promotion |
| [Model-Scope Overlay Bundle Schema](model-scope-overlay-bundle-schema-2026-05-05.md) | Operator contract for Model-Scope campaign reports, overlay sidecars, validation summaries, and promotion-linkage audit fields |
| [ADR-0001: Evidence-Governed Variation Agent](architecture/adr-0001-evidence-governed-variation-agent.md) | Proposed small-model architecture combining an immutable evidence ledger, disposable Qdrant projection, LoRA-only training, enforceable authority, and dual-Spark evaluation |
| [Evidence-Governed Variation Dual-Spark Runbook](runbooks/evidence-governed-variation-dual-spark.md) | Documentation-only activation, resumability, service restoration, evidence packaging, and matched-ablation plan for the proposed EGV campaign |
| [Default-Promotion Evidence Runbook](default-promotion-evidence-runbook-2026-05-06.md) | Operator workflow for linked validation, promotion-linkage audit, repeat-seed decision, and fail-closed preflight evidence |
| [Evidence Dashboard Runbook](evidence-dashboard-runbook-2026-05-06.md) | Operator guide for dashboard evidence panels, source APIs, artifact regeneration, and fail-closed interpretation |
| [Evidence Artifact Retention Policy](evidence-artifact-retention-policy-2026-05-06.md) | Retention, regeneration, CI artifact, and safe-deletion policy for generated evidence outputs |
| [Evidence Chain And Index Schema](evidence-chain-index-schema-2026-05-06.md) | JSON contract for evidence-chain summaries, cross-artifact evidence indexes, and freshness audits |
| [Evidence Cleanup Plan Schema](evidence-cleanup-plan-schema-2026-05-06.md) | JSON contract for generated evidence cleanup dry-run plans and dashboard cleanup-plan responses |
| [HeavySkill Engine Adaptation Review](heavyskill-engine-adaptation-2026-05-05.md) | HeavySkill paper and repo review mapped into typed trajectory evidence, deliberation records, heavy-thinking metrics, and fail-closed engine promotion phases |
| [Computational Storage And Drive Nodes](COMPUTATIONAL_STORAGE_DRIVE_NODES.md) | Canonical summary of hard-drive / storage-node experiments and scope limits |
| [Computational Storage Scope Decision](computational-storage-transport-scope-decision.md) | Formal claim boundary for the RP2040 transport path |
| [Computational Storage Hardware Evidence Runbook](computational-storage-hardware-evidence-runbook.md) | Operator workflow for real hardware evidence capture |
| [Revised Roadmap: Disk-First CPU And Retrieval Program (2026-03-28)](revised-roadmap-disk-first-program-2026-03-28.md) | Active phase-gated program roadmap with dependencies, review loops, and scope controls |
| [Disk-Resident LLM Feasibility (2026-03-28)](disk-resident-llm-feasibility-2026-03-28.md) | Feasibility memo, hardware sizing, and repo improvement plan for SSD-resident CPU inference |
| [Disk-Resident LLM Addendum: REAP And TurboQuant (2026-03-28)](disk-resident-llm-addendum-reap-turboquant-2026-03-28.md) | Follow-up research memo on MoE pruning, KV compression, and CPU / retrieval-first systems |
| [Phase 7 Promotion Review (2026-03-29)](phase7-promotion-review-2026-03-29.md) | Final research-baseline promotion call, defer conditions, and next-branch recommendations for the disk-first CPU / retrieval program |
| [Roadmap Audit And Weight Refinement Plan (2026-03-06)](roadmap-audit-and-weight-refinement-plan-2026-03-06.md) | Current conclusion that non-hardware development phases are complete plus the next evaluation plan |
| [Weight Refinement Campaign Results (Session 28)](weight-refinement-campaign-results-2026-03-06-session28.md) | Durable summary of the partial bounded campaign, recovered findings, and promotion guidance |
| [Weight Refinement Campaign Results (Session 32)](weight-refinement-campaign-results-2026-04-25-session32.md) | Durable summary of the Session 32 partial bounded campaign and explicit no-promotion outcome |
| [2026-05-15 Desktop Reconciliation & Reimplementation Backlog](planning/2026-05-reconciliation/reconciliation-2026-05-15-reimplementation-backlog.md) | Full post-merge reconciliation of desktop machine with laptop work, multi-agent gap analysis, and prioritized reimplementation backlog |
| [AttnRes Adapter Implementation (2026-05-04)](attnres-adapter-implementation-2026-05-04.md) | BlockAttnResAdapter and LayerAttentionAggregator — MoonshotAI Attention Residuals adapted to the ChelatedAI adapter framework |
| [BHS Scope B Audit (2026-05-16)](bhs-scope-b-audit-2026-05-16.md) | End-to-end Brutal Honesty Score audit — 13 fresh Tier B adversarial agents covering Engine-Scope (5), Model-Scope (6), and TTS (2) phases; per-phase scores, critical cross-phase findings, and 9 new Carried Debt rows |

## Agentic Engineering And Planning

| Document | Description |
|----------|-------------|
| [ARCH-AEP Overview](ARCH%20AGENTIC%20ENGINEERING%20AND%20PLANNING/README.md) | Narrative overview for ARCH-AEP |
| [ARCH-AEP Orchestrator Briefing](ARCH%20AGENTIC%20ENGINEERING%20AND%20PLANNING/orchestrator-briefing.md) | Session-start narrative and folder index |
| [ARCH-AEP Workflow Spec](ARCH%20AGENTIC%20ENGINEERING%20AND%20PLANNING/workflow.md) | End-to-end workflow specification |
| [ARCH-AEP Templates](ARCH%20AGENTIC%20ENGINEERING%20AND%20PLANNING/templates.md) | ID, branch, and tracker conventions |
| [ARCH-AEP Next Session](ARCH%20AGENTIC%20ENGINEERING%20AND%20PLANNING/next-session.md) | Session handoff checklist |
| [ARCH-AEP Phase Planning](ARCH%20AGENTIC%20ENGINEERING%20AND%20PLANNING/phase-planning.md) | Long-running planning record |
| [ARCH-AEP Engine-Scope Roadmap](ARCH%20AGENTIC%20ENGINEERING%20AND%20PLANNING/architecture-2026-04-30-engine-scope-roadmap.md) | Phase-by-phase architecture and implementation roadmap for the Engine-Scope autonomous search cycle |
| [ARCH-AEP Model-Scope Roadmap](ARCH%20AGENTIC%20ENGINEERING%20AND%20PLANNING/architecture-2026-05-01-model-scope-roadmap.md) | Phase-by-phase architecture and implementation roadmap for the true model-hook steering cycle |
| [ARCH-AEP Schedule And Tracking](ARCH%20AGENTIC%20ENGINEERING%20AND%20PLANNING/schedule-and-tracking.md) | Cadence and gates |
| [ARCH-AEP Tier Close Checklist](ARCH%20AGENTIC%20ENGINEERING%20AND%20PLANNING/tier-close-checklist.md) | Tier close audit checklist |
| [ARCH-AEP Cycle Summary Template](ARCH%20AGENTIC%20ENGINEERING%20AND%20PLANNING/cycle-summary-template.md) | End-of-cycle summary template |
| [ARCH-AEP Cycle Summaries](ARCH%20AGENTIC%20ENGINEERING%20AND%20PLANNING/cycle-summaries/README.md) | Cycle summary storage location |
| [ARCH-AEP Backlog Template](ARCH%20AGENTIC%20ENGINEERING%20AND%20PLANNING/backlog-template.md) | Master backlog template |
| [ARCH-AEP Backlog Index](ARCH%20AGENTIC%20ENGINEERING%20AND%20PLANNING/backlog-index.md) | Backlog index across cycles |
| [ARCH-AEP Tracker Pointer](ARCH%20AGENTIC%20ENGINEERING%20AND%20PLANNING/tracker-pointer.md) | Active tracker link |
| [ARCH-AEP Tracker Index](ARCH%20AGENTIC%20ENGINEERING%20AND%20PLANNING/tracker-index.md) | Tracker index across cycles |
| [ARCH-AEP Verification Log](ARCH%20AGENTIC%20ENGINEERING%20AND%20PLANNING/verification-log.md) | Test/build evidence log |
| [ARCH-AEP Phase Summary Template](ARCH%20AGENTIC%20ENGINEERING%20AND%20PLANNING/phase-summary-template.md) | Per-PR or per-phase summary template |
| [ARCH-AEP Phase Summaries](ARCH%20AGENTIC%20ENGINEERING%20AND%20PLANNING/phase-summaries/README.md) | Phase summary storage location |
| [ARCH-AEP Risk Memo Template](ARCH%20AGENTIC%20ENGINEERING%20AND%20PLANNING/risk-memo-template.md) | Critical/High risk memo template |
| [ARCH-AEP Risk Memos](ARCH%20AGENTIC%20ENGINEERING%20AND%20PLANNING/risk-memos/README.md) | Risk memo storage location |
| [ARCH-AEP Risk Memo Archive](ARCH%20AGENTIC%20ENGINEERING%20AND%20PLANNING/risk-memos/closed/README.md) | Closed risk memo archive |
| [ARCH-AEP Agent Learning](ARCH%20AGENTIC%20ENGINEERING%20AND%20PLANNING/agent-learning.md) | Cross-session learnings |
| [ARCH-AEP Change Log](ARCH%20AGENTIC%20ENGINEERING%20AND%20PLANNING/change-log.md) | Scope/defer decision log |
| [ARCH-AEP Scope Lock Template](ARCH%20AGENTIC%20ENGINEERING%20AND%20PLANNING/scope-lock-template.md) | Scope lock record |
| [ARCH-AEP Glossary](ARCH%20AGENTIC%20ENGINEERING%20AND%20PLANNING/glossary.md) | Shared terminology |
| [ARCH-AEP Test Matrix](ARCH%20AGENTIC%20ENGINEERING%20AND%20PLANNING/test-matrix.md) | Test selection guidance |
| [ARCH-AEP Active Tracker Template](ARCH%20AGENTIC%20ENGINEERING%20AND%20PLANNING/active-tracker-template.md) | Tracker template with lock block |
| [ARCH-AEP Cycle Layout Template](ARCH%20AGENTIC%20ENGINEERING%20AND%20PLANNING/cycle-layout-template.md) | Optional parallel cycle layout |

## Session Logs & Research

| Document | Description |
|----------|-------------|
| [Session Log 1 (2026-02-13)](ARCH%20AGENTIC%20ENGINEERING%20AND%20PLANNING/session-log-2026-02-13-impl.md) | Initial analysis cycle -- 55 findings discovered |
| [Session Log 2 (2026-02-13)](ARCH%20AGENTIC%20ENGINEERING%20AND%20PLANNING/session-log-2026-02-13-impl-2.md) | Implementation cycle -- 7 findings resolved, 5 PRs merged, 134 tests |
| [Session Log 3 (2026-02-17)](ARCH%20AGENTIC%20ENGINEERING%20AND%20PLANNING/session-log-2026-02-17-impl-3.md) | Remediation cycle -- 13 findings resolved, 345 tests passing locally |
| [Session Log 34 (2026-04-29)](ARCH%20AGENTIC%20ENGINEERING%20AND%20PLANNING/session-log-2026-04-29-session34.md) | Road-course, self-healing, and attribution batch with 1259-test wrap validation |
| [Backlog (2026-02-13)](ARCH%20AGENTIC%20ENGINEERING%20AND%20PLANNING/backlog-2026-02-13.md) | Master backlog with 55 prioritized findings |
| [Research: F-006 Config Mapping](ARCH%20AGENTIC%20ENGINEERING%20AND%20PLANNING/research-f006-config-mapping.md) | Hardcoded value -> ChelationConfig mapping |
| [Research: F-010 Logger Migration](ARCH%20AGENTIC%20ENGINEERING%20AND%20PLANNING/research-f010-logger-migration.md) | print() -> ChelationLogger migration plan |
| [Research: F-002/F-003 Test Plan](ARCH%20AGENTIC%20ENGINEERING%20AND%20PLANNING/research-f002-f003-test-plan.md) | Test coverage plan for benchmark + checkpoint |
| [Research: Tier2/Tier3 Plan (2026-02-17)](ARCH%20AGENTIC%20ENGINEERING%20AND%20PLANNING/research-2026-02-17-tier2-tier3-plan.md) | Multi-finding implementation architecture and sequencing notes |
| [Proposed Trials and Validation Round (2026-02-17)](proposed-trials-validation-round-2026-02-17.md) | Adaptive weighting + Eagan delta-z comparative validation plan |
## Analysis

| Document | Description |
|----------|-------------|
| [RLM Analysis](rlm-analysis.md) | Detailed analysis of the RLM paper source code |

---
