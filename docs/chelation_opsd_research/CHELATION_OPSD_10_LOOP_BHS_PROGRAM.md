# CHELATION + OPSD: 10-Loop BHS-Scored Research Program

**Program Owner**: Integration Lead (main session)  
**Goal**: Maximize the BHS Research Score of the upgrade plan for making CHELATEDAI's chelation system practically viable using On-Policy Self-Distillation and related techniques. Target: as close to 100/100 as possible through rigorous iteration.

**Core Philosophy**: Apply the exact same Brutal Honesty v3.3 standards (evidence rule, L1–L13 taxonomy, Carried Debt, independent Tier B review, no overclaiming) to the research process itself that we apply to code.

---

## The 10-Loop Structure (with BHS Evaluation)

Each loop ends with a formal **BHS Research Self-Assessment** (using the rubric) and an updated Gap Tracker. Only after honest scoring and gap exposure do we proceed.

**Loop 1: Deep Research & Mapping** (Current)
- Exhaustive literature review (OPSD, SDPO, related self-distillation, self-rewarding, process supervision).
- Ruthless audit of current chelation implementation.
- Extraction of all relevant techniques and failure modes.
- **BHS Evaluation Gate**: Full integration of all agent outputs + honest scoring of completeness and evidence quality.

**Loop 2: Architecture Design**
- Define 4–6 coherent, comparable upgrade architectures.
- For each: interfaces, data flow, training/inference changes, risks.
- **BHS Evaluation Gate**: Are the architectures described at a level that can actually be implemented and tested? Are weaknesses exposed?

**Loop 3: Loss Function & Training Objective Design**
- Design and compare multiple concrete loss functions (KL-regularized residual correction, asymmetric privileged distillation, filtered self-distillation, multi-objective, quantization-aware, etc.).
- **BHS Evaluation Gate**: Are the losses specified with enough mathematical and implementation detail to code?

**Loop 4: Stability, KL Control & Forgetting Mitigation**
- Detailed mechanisms for KL scheduling, shock mitigation, retention regularization, and long-horizon stability.
- **BHS Evaluation Gate**: Do we have concrete, testable methods or just high-level ideas?

**Loop 5: Sample Efficiency & Data Strategies**
- On-policy data collection, filtering (MIS-PO style), curriculum learning, replay for retention.
- **BHS Evaluation Gate**: Are the data pipelines and filtering logic defined enough to implement?

**Loop 6: Quantization-Aware & Low-Rank Variants**
- Deep integration of quantization into the learning objectives (STE, fake-quant, etc.).
- Low-rank + self-distillation hybrids.
- **BHS Evaluation Gate**: Do we have working code sketches or only theory?

**Loop 7: Self-Edit Directive Pipeline Integration**
- Turn the existing `SelfHealingChelationPlanner` + `SelfEditDirective` system into a true on-policy self-distillation loop.
- **BHS Evaluation Gate**: Is the new pipeline defined end-to-end with clear ownership of training vs advisory roles?

**Loop 8: Evaluation Framework & Benchmark Design**
- Finalize metrics, success criteria, benchmarks, retention tests, and harness.
- **BHS Evaluation Gate**: Is the evaluation framework rigorous enough that a negative result would be believable?

**Loop 9: Implementation & Empirical Testing**
- Implement the top 3 patterns.
- Run controlled experiments (correction gain, retention over cycles, sample efficiency, quantization survival, stability).
- **BHS Evaluation Gate**: Do we have real numbers, honest negative results, and reproducible experiments?

**Loop 10: Synthesis, Final Roadmap & BHS Research Score**
- Cross-pattern comparison.
- Final honest BHS Research Self-Assessment of the entire program.
- Production-ready upgrade recommendation with explicit Carried Debt.
- **BHS Evaluation Gate**: What is the final BHS Research Score of the program? What work remains at 100/100 quality?

---

## BHS Research Scoring at the End of Each Loop

At the end of every loop, we produce:

- **BHS Research Self-Assessment** (0–100) using the rubric.
- Honest strengths and weaknesses.
- Updated Carried Debt list (with TTL).
- Specific actions to improve the score in the next loop.
- Decision: Proceed / Remediate current loop / Reduce scope.

The explicit goal is to drive the overall research program as close to **100/100 BHS Research Score** as possible through this self-correcting loop, exactly as we demand of code changes.

---

**Current Status**: Loop 1 in progress. Multiple high-quality agent outputs already received. Initial BHS Research Self-Assessment of the program: **71/100** (detailed in separate document).

This structure is now locked. We will execute all 10 loops with BHS scoring. No more vague progress — only scored, gap-exposed iteration.

**Next**: Complete Loop 1 synthesis from all 10 agents, score it using the rubric, expose remaining gaps, then move into Loop 2 architecture with the lessons applied.