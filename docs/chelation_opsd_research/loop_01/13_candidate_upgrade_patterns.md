# Candidate Upgrade Patterns for CHELATION + OPSD (Living Document) — Updated Post Loop 1 Agents

**Loop:** 1 (Refined after Agent 2, 8, 9 outputs)  
**Last Updated:** 2026-05-15  
**BHS Research Self-Assessment of this document:** 82/100 (strong synthesis from completed agents, but still early — full empirical validation pending Loops 9-10)

---

## Refined Tier S Patterns (Highest Priority)

**Pattern S1: Asymmetric On-Policy Self-Distillation for Self-Edit Directives (Top Priority)**
- Teacher = model with privileged diagnostic context (structural health, quant gate, retrieval anomalies, chelation_log signals).
- Student = normal model + ChelationAdapter (and low-rank variants).
- Train the adapter(s) to match the teacher's output distribution on the student's own on-policy generations, using dense per-token KL/JSD + pointwise clipping.
- Add MIS-PO-style filtering on self-generated "mistakes" or high-correction-value examples.
- Use the existing `SelfEditDirective` generation as the outer loop (proto-ReSTEM), with the inner loop being this OPSD distillation.
- **Why it fits:** Directly turns advisory directives into actually reinforced, stable updates. Matches @ar0cket1's SDPO/MIS-PO work and the Self-Distilled Reasoner paper. Highest leverage for closing the "advisory-only" + "no persistent reinforcement" gap (Agent 2 audit).

**Pattern S2: KL-Regularized Residual Chelation Training**
- Add explicit KL divergence term (to frozen base model) during training of ChelationAdapter (MLP, LowRankAffine, QuantizationAwareLowRank, etc.).
- Combine with existing residual structure (`x + δ(x)`) and L2 normalization.
- Use adaptive KL scheduling and "shock" mitigation from SDPO literature to prevent forgetting.
- **Why it fits:** Directly attacks the catastrophic forgetting risk explicitly noted in the SEAL/EGGROLL docs and multiple panel findings (F-ML- series). Complements Pattern S1.

**Pattern S3: Quantization-Aware Low-Rank OPSD (Agent 8 Implementation + OPSD Extension)**
- Build directly on the `QuantizationAwareLowRankAdapter` delivered by Agent 8 (STE fake-quant in train mode, fallback simulation in eval, nestable with BoundedAdapter).
- Train using OPSD-style asymmetric distillation where the teacher is quantization-aware (privileged quant simulation during teacher forward pass).
- Per-dim scaling (from DSM) + per-token importance clipping mapped to embedding dimensions with highest post-quant impact.
- **Why it fits:** Turns the current post-hoc BoundedAdapter + QuantizationPromotionGate (scaffolding only) into corrections that survive INT8 by construction. Highest practicality for immediate testing (code already exists).

**Pattern S4: Filtered On-Policy Data for Sedimentation Replacement / Augmentation**
- Replace or heavily augment the current sedimentation training (InfoNCE on collapse counts) with filtered on-policy self-distillation data from the model's own generations.
- Use entropy/divergence-based filtering (inspired by @ar0cket1's low-entropy/high-divergence analysis and MIS-PO).
- **Why it fits:** Current sedimentation is one of the weakest and most unstable parts of the system (mode collapse, hard-coded losses, no gradient clipping). OPSD provides the dense, stable replacement.

---

## Strong Supporting Patterns (Tier A)

- SDFT-Style Continual Retention for Chelation (from Agent 9): Use SDFT (Self-Distillation Fine-Tuning) techniques for long-horizon retention during repeated self-edits.
- Constitutional Chelation Critic (from Agent 9): Add a chelation-specific "constitution" for self-critique/revision of directives before distillation (Constitutional AI style).
- ReST-MCTS over Directives (from Agent 9): Use ReST-MCTS* with Process Reward Models for process-dense filtering of SelfEditDirectives.
- Input-Anchored + Generic-KL Regularized Adapters (PD/SelfAug style): Anchor corrections to input while using KL for stability.

---

## BHS Research Self-Assessment of Current Patterns (Post Agent 8 + 9)

**BHS Research Score for this document: 82/100**

**Strengths**: Strong mapping from completed agents. Real code already exists for Pattern S3. Multiple high-leverage patterns identified with clear ties to both the audit pain points and OPSD literature.

**Weaknesses Exposed**:
- L5 risk: Still mostly "patterns" — no full training runs yet.
- L4 risk: Some patterns (especially S1 and S4) require new training loop infrastructure that doesn't fully exist yet.
- Environment Carried Debt: Full empirical validation blocked until WSL is fixed or we move to a provisioned host.
- Missing depth: Interaction with computational storage POC and true multi-hundred-cycle retention not yet deeply mapped.

**Carried Debt**: Empirical validation of all patterns (Loops 9-10). Deep computational storage integration (Loop 6/10).

---

**Next**: I am now moving into Loop 2 (Architecture Design) for the top 3 patterns (S1, S2, S3) and will produce initial architecture documents + interface sketches without further pausing.

The 10-agent swarm for Loop 1 continues. I will integrate remaining agent outputs as they land and continue the program.

The research continues. No more stopping. The 10-loop BHS-scored program is now driving forward.