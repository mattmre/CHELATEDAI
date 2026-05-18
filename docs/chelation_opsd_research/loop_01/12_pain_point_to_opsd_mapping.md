# Pain Point → OPSD Technique Mapping (Living Document)

**Loop:** 1 (Initial Version)  
**Purpose:** Map every major documented weakness in the current CHELATEDAI chelation/self-healing system to specific techniques from On-Policy Self-Distillation (OPSD), SDPO, Self-Distilled Reasoner, and related work.

This will become the foundation for architecture and loss function design in later loops.

---

## 1. Major Pain Points (from Audit + Panel Reviews + findings.md)

### High-Severity / CRITICAL

**P1: InfoNCE temperature too low → mode collapse risk**  
- Evidence: F-ML-001 in panel review (temp=0.07 described as "dangerously small")
- Current impact: Sedimentation training frequently collapses or produces low-entropy representations.

**P2: No KL / divergence control during adaptation**  
- Self-edits and sedimentation updates can cause large distribution shifts with no mechanism to bound them.
- Result: "KL shocks", instability, and unpredictable behavior.

**P3: Self-healing is advisory-only — no actual training loop for accepted directives**  
- `SelfEditDirective`s are generated and evaluated but the system does not have a robust way to turn accepted directives into persistent, high-quality parameter updates on the ChelationAdapter.

**P4: Catastrophic forgetting during repeated self-edits**  
- Explicitly noted in SEAL/EGGROLL analysis: "repeated edits can cause catastrophic forgetting".
- No retention mechanism strong enough for long-horizon self-healing.

**P5: Data leakage in `chelation_log`**  
- The adapter is sometimes trained on data that will later be used for evaluation (F-ML-040).

**P6: Training/serving parity broken (norm drift, etc.)**  
- F-ML-023: stored vectors may have varying norms after adaptation.

**P7: Extremely poor sample efficiency of current self-correction loops**  
- Current sedimentation + self-edit process is closer to naive RL-style updates than to modern dense self-distillation.

### Medium / Important

**P8: No proper experiment tracking or safe checkpointing**  
- Checkpoints get overwritten; no experiment trail.

**P9: Weak or missing regularization on adapters during training**  
- Many adapter variants have `regularization_loss()` returning 0.0.

**P10: Quantization survival is hacky (BoundedAdapter with hardcoded 0.01 threshold)**  
- Not deeply integrated with training objectives.

**P11: Chelation adapter and disk-first / computational storage path have never been properly integrated**

---

## 2. Mapping to OPSD / SDPO Techniques

| Pain Point | Closest OPSD / Related Technique | How It Helps | Specific Papers / Ideas | Priority for CHELATEDAI |
|------------|----------------------------------|--------------|-------------------------|-------------------------|
| P1 (InfoNCE collapse) | Temperature annealing + entropy-aware objectives in self-distillation | Prevents over-sharpening | Self-Distilled Reasoner, SDPO loss variants | High |
| P2 (No KL control) | Explicit KL regularization + "KL shock" mitigation schedules | Bounds distribution shift during self-correction | @ar0cket1 threads, SDPO papers, standard RLHF KL penalty | **Very High** |
| P3 (No training loop for directives) | On-policy self-distillation from privileged diagnostic context | Turns diagnostic signals into dense training signal for the adapter | Asymmetric distillation (teacher sees diagnostics, student does not) | **Very High** |
| P4 (Catastrophic forgetting) | Retention via KL to base model + replay of previous on-policy data | Prevents overwriting useful base behavior | Mix of SFT-style retention + on-policy distillation | High |
| P5 (Data leakage) | Proper train/eval split on on-policy rollouts + held-out diagnostic sets | Standard best practice, easier with on-policy filtering | ReST, Self-Distilled Reasoner evaluation protocols | Medium |
| P6 (Training/serving parity) | Consistency regularization between training and inference distributions | Reduces norm drift and representation shift | Techniques from online distillation literature | Medium-High |
| P7 (Sample efficiency) | Dense token-level self-distillation instead of sparse rewards | Much better sample efficiency than current sedimentation-style updates | Core promise of OPSD vs RL | **Very High** |
| P8 (No experiment tracking) | Not directly solved by OPSD, but can be combined with modern experiment tracking + checkpointing best practices | - | - | Medium |
| P9 (Weak regularization) | Add KL-to-base or Frobenius regularization on adapter during self-distillation | Prevents adapter from growing too large too fast | Common in LoRA + distillation setups | High |
| P10 (Quantization survival) | Quantization-aware self-distillation objectives + BoundedAdapter integrated into loss | Make survival a first-class training objective | Quantization-aware distillation papers + BoundedAdapter work | High |
| P11 (No integration with disk-first path) | Opportunity to do near-data self-distillation | Could be a unique advantage of CHELATEDAI architecture | Computational storage + on-device distillation ideas | Medium (strategic) |

---

## 3. Highest-Leverage Starting Points (Initial Ranking)

Based on current knowledge (will be refined as more agents report):

**Tier S (Must address early):**
- KL control + stability during self-correction training
- Turning SelfEditDirectives into on-policy training data (asymmetric distillation with privileged diagnostics)
- Sample-efficient dense supervision instead of current sedimentation

**Tier A:**
- Regularization of ChelationAdapter during self-distillation
- Quantization-aware objectives
- Better data filtering / selection for self-correction training data

**Tier B:**
- Integration with computational storage / disk-first path
- Advanced variants (low-rank + self-distillation, multi-agent self-distillation, etc.)

---

**Status:** Initial mapping. Will be heavily expanded and ranked as literature agents and code auditors finish. This document will feed directly into Loop 2 (Architecture Design) and Loop 3 (Loss Functions).