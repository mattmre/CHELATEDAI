# STEERING + CHELATION + ADAPTIVE RAG-DAG + MICROS LM
## 10-Loop BHS Research Program Definition

**Version**: Kickoff (extends the successful CHELATION_OPSD_10_LOOP_BHS_PROGRAM structure)

## Program Identity
**Full Name**: Steering Node Chelation for Adaptive RAG-DAGs with Hyperscale Graph Convergence and MicroSLM Proof-of-Concept  
**Short**: Steering-Chelation-RAGDAG-MicroSLM Program  
**Repository Home**: `docs/steering_chelation_rag_dag_research/`  
**Governance**: BHS v3.3 (see `STEERING_CHELATION_BHS_RESEARCH_RUBRIC.md` and parent conventions)  
**Cross-References**: All prior OPSD Loop 01 artifacts, Model-Scope steering architecture (2026-05-01), EGGROLL hyperscale analysis, Computational Storage Drive Nodes doc, TTS pipeline, full AEP archive.

## The 10 Loops (Canonical)

1. **Deep Research & Mapping**  
   Literature (LogicRAG dynamic DAGs, SAE-RSV + Matryoshka SAEs for steering, latest OPSD/SDPO variants, spectral embedding methods, hyperscale ES extensions) + ruthless audit of the *five connected substrates* (Chelation, TTS Steering Nodes, Model-Scope, Comp-Storage graphs/drive nodes, EGGROLL/OPSD optimizer surfaces). Produce master pain-point → technique mapping and Tier S/A/B upgrade patterns.

2. **Architecture Design**  
   Concrete specs for `RerouteDAG` abstraction, multi-cast steering node interface, micro-SLM I/O contract (chelation signals + sparse features → route proposals), drive-node dispatch contract for candidate routes, artifact card + promotion schema extensions. Multiple viable patterns sketched with pseudocode and dependency DAGs.

3. **Loss Function & Training Regime Variants**  
   OPSD asymmetric privileged-diagnostic for reroute traces, route-cohesion auxiliary losses, structural health regularizers, quantization-aware objectives, low-rank population (EGGROLL-style) vs gradient hybrids. Target: stable on-policy self-improvement of the route policy without KL shocks in embedding or DAG space.

4. **Stability, KL Control & Route Forgetting Mitigation**  
   The hardest practical problem. Mechanisms to prevent accepted reroutes from destroying previously reliable retrieval facts or base model compatibility. Includes retention replay families specific to route histories, anchoring to privileged successful traces, and bounded actuator constraints extended to DAG mutations.

5. **Sample Efficiency & Data Filtering**  
   MIS-PO / hard-negative / attribution-style filtering applied to reroute proposal traces. How to decide which noisy neighborhoods or failed routes are worth spending micro-SLM capacity and multi-path speculation budget on. Budget-aware collection policy as first-class citizen.

6. **Quantization-Aware & Low-Rank / Bounded Variants**  
   Everything (micro-SLM head, steering actuators, route proposal generators) must survive the same INT8/BoundedAdapter floor that production chelation already targets. Low-rank route deltas, Matryoshka-style nested steering features, block-graph friendly representations for drive-node paths.

7. **Self-Edit Directive + Steering Node + DAG Mutation Integration**  
   Extend `SelfEditDirective` (and the self-healing ledger) so that generated directives can propose *live DAG topology changes* and multi-vector reroute sets, not just embedding adapter corrections. Close the loop between diagnostics → directive → steering node execution → outcome fitness → distillation back into the policy.

8. **Evaluation Framework & Benchmark Design**  
   Extend synthetic collapse fixture and road-course harnesses to DAG/reroute tasks. New "route acceptance under noise" family of benchmarks. Drive-node latency parity surfaces (where in scope). Full transfer testing (BEIR + new DAG reasoning tasks). Artifact card + verifier card automation for every candidate.

9. **Implementation of Top Patterns + Tests + MicroSLM Smoke**  
   Ship the 2-3 highest-ranked patterns from Loops 2-8 as working, evidence-backed slices. First real training of a 2-4 GB class micro-SLM (or its steering head) against the substrate. Full BHS evidence packages. Smoke pipeline that exercises the entire chain on a fresh checkout.

10. **Comparative Analysis, Recommendations & Final Upgrade Roadmap**  
    What actually delivered lift under BHS gates. What was rejected and why (with data). Concrete ship/no-ship decisions for main. Updated productionization plan for the winning substrate (how it wires into AntigravityEngine / Model-Scope / existing vector store). Clear statement of remaining carried debt and recommended next program (if any).

## Loop Execution Rules (identical spirit to OPSD program)
- Each loop is executed by one or more specialized agents (literature, substrate audit, architecture, loss design, etc.) + integration lead for synthesis.
- Every loop ends with a `loop_N/NN_synthesis_and_prioritization.md` (or equivalent) that contains the ranked patterns, BHS self-assessment, updated program score, and explicit carried-debt re-audit.
- Code or architecture artifacts from a loop only become "official" for the program after the synthesis document is written and the loop is closed.
- Parallel swarm execution is encouraged for speed (as proven in OPSD Loop 01), but integration/synthesis is serial and owned by the orchestrator.
- No loop may be declared complete until the BHS rubric items for that loop's deliverables are satisfied (evidence, brutal honesty, no L1-L13 violations).

## Entry Criteria for Starting Loop N
- Loop N-1 synthesis + all supporting agent docs committed.
- Explicit "Loop N Kickoff Brief" (one-pager) that names the 3-5 concrete questions the loop must answer and the minimal evidence surface required to close it.
- Carried debt items from prior loops either closed or explicitly carried with mitigation plan.

## Exit Criteria for the Whole Program (Loop 10 close)
- At least one pattern or micro-SLM configuration has a complete, independently reviewed BHS evidence chain (artifact card, replay, holdout, quant gate, rollback demo, route-specific metrics) scoring 100.
- The program can state with evidence which of the original thesis claims are supported, refuted, or still open.
- Clear recommendation: "Ship X to main as the new default reroute substrate", "Retain Y as experimental in `computational_storage_poc/` or a feature branch", "Pivot Z because fundamental blocker discovered".
- Updated docs for consumers (engineers who will actually use the new surfaces).

## Relationship to Prior Work (explicit, non-duplicative)
This program *does not* restart chelation or OPSD research. It treats the 2026-05 OPSD Loop 01 outputs as *input substrate* (the pain-point mappings and candidate upgrade patterns for the chelation/self-edit layer are directly reusable). It adds the missing "graph + steering node + micro SLM + drive dispatch + hyperscale ES" dimensions that turn a per-vector correction system into a live reroutable reasoning graph.

It also does not duplicate the Model-Scope steering architecture doc or the computational-storage POC; those are the *raw material* being converged.

## Scope Locks (non-negotiable for this tranche)
- Full agent harnesses / long-horizon planning loops: out of scope (inspiration only, per frontier adaptive overlay decisions).
- Claiming real hardware LLM inference on spinning rust or SSDs: scope-locked to existing computational-storage transport + dispatch contracts + emulation. New RP2040 evidence only if actually captured on hardware.
- Mutating base 2-4 GB model weights in the proof: forbidden without explicit exception + full rollback evidence. Adapters, steering heads, and feature banks only.
- "It worked in simulation therefore it ships": never. Every surface must have a smoke/replay path that a fresh checkout can run.

**This document is the canonical program definition.** Update it only at loop closeouts with a new version note and diff summary.

*Program kickoff — 2026-05*