# Evaluation Framework Skeleton for CHELATION + OPSD

**Loop:** 1 (Early Draft)  
**Owner:** Integration Lead + Agent 10 (Synthesis)  
**Purpose:** Define the metrics, success criteria, and benchmark harness that will be used across all 10 loops to evaluate whether OPSD-augmented chelation actually works in practice.

---

## 1. Core Goals of the Evaluation Framework

We need to answer:  
**"Does adding on-policy self-distillation techniques make our chelation/self-healing system meaningfully better at stable, sample-efficient, persistent self-correction?"**

Key dimensions to measure:

- Correction Effectiveness
- Retention / Stability (anti-forgetting)
- Sample Efficiency
- Training Stability (KL shocks, collapse, gradient health)
- Quantization Survival
- Online / Continual Learning Capability
- Computational & Memory Overhead

---

## 2. Proposed Metric Categories (Draft)

### 2.1 Correction Gain
- Delta in retrieval quality (NDCG@10, Recall@K, MRR) on held-out sets after applying chelation/self-edits.
- "Self-healing success rate": % of generated SelfEditDirectives that produce statistically significant improvement when applied.

### 2.2 Retention / Forgetting
- Performance on original training distribution after multiple rounds of self-correction.
- Catastrophic forgetting index (drop in performance on base tasks).

### 2.3 Sample Efficiency
- Number of on-policy examples / tokens needed to achieve a target correction gain.
- Comparison vs current sedimentation baseline and vs pure RL-style methods.

### 2.4 Stability
- Frequency and magnitude of "KL shocks" during self-distillation training.
- Gradient norm statistics.
- Incidence of mode collapse or entropy collapse.

### 2.5 Quantization Robustness
- Performance delta when moving from FP16 → INT8 after chelation training (with and without BoundedAdapter).

### 2.6 Online Adaptation
- Ability to improve over time from streaming diagnostic signals without full retraining.
- Wall-clock time and token budget to reach a target improvement in a simulated online loop.

---

## 3. Benchmark Tasks (Initial Proposal)

**Core Retrieval Benchmarks (must-run):**
- BEIR suite (SciFact, NFCorpus, etc.) — already used in the repo
- Additional domain-specific sets if available

**Self-Correction Specific Tasks:**
- Synthetic "drift injection" benchmarks (intentionally corrupt embeddings and measure recovery)
- Iterative self-correction loops on fixed query sets
- Long-horizon retention tests (apply 10–20 rounds of self-edits and measure degradation)

**Stability & Efficiency Micro-benchmarks:**
- KL divergence curves during self-distillation training
- Sample complexity curves (performance vs number of on-policy examples)

---

## 4. Harness Requirements

- Reproducible experiment tracking (no more overwriting checkpoints)
- Held-out evaluation sets that are never seen during self-edit generation or training
- Ability to run both "with OPSD-style objectives" and "baseline sedimentation" under identical conditions
- Support for `BHS_WORKBENCH_EVIDENCE_DIR` style artifact capture (for future browser/UX proof if needed)
- Easy comparison across the 5–8 loss variants we will design in Loop 3

---

## 5. Next Steps for This Document

- Refine metrics after Agent 2’s full audit is integrated
- Get input from Agent 6 (Self-Edit Directive Integration) and Agent 7 (Sample Efficiency)
- Lock v1.0 of the harness before Loop 8 begins

---

**Status:** Early skeleton — will be heavily updated as Loop 1 agents report.