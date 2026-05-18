# Research Agent 9: Related Research Scan Report
**Loop 1 — CHELATION + OPSD Upgrade Program**

**Agent**: Research Agent 9 - Related Research Scanner  
**Date**: 2026-05-15  
**Repo Root**: /mnt/d/GITHUB/CHELATEDAI  
**Output Location**: `docs/chelation_opsd_research/loop_01/09_related_research_scan.md`  
**Role Mandate**: Broadly scan research areas beyond core OPSD/SDPO (Self-Rewarding LLMs, Iterative/Self-DPO, Constitutional AI, STaR, ReST, online RLHF variants, continual learning with self-distillation, process supervision, etc.). Identify additional techniques/papers that complement or enhance application of on-policy self-distillation to chelation/self-healing systems. Surface novel ideas in stability, data efficiency, or self-correction not heavily covered in core @ar0cket1 / OPSD work. Produce actionable mappings and testable variants for the CHELATEDAI upgrade.

**Dependencies**: This report complements and cross-references:
- `01_literature_deep_dive.md` (Agent 1: core OPSD 2601.18734, SDPO 2601.20802, MIS-PO, KL shocks, privileged-context teacher-student, @ar0cket1 Hermes practicals)
- `02_chelation_system_audit.md` (Agent 2: ChelationAdapter family, SelfHealingChelationPlanner + SelfEditDirective, advisory nature, sedimentation training, AntigravityEngine spectral masking + chelation_log, retention/quantization gates, failure modes)
- `05_kl_stability_control.md`, `06_selfedit_directive_integration.md`, `04_on_policy_dynamics.md`, `07_sample_efficiency_filtering.md`, `12_pain_point_to_opsd_mapping.md`, `13_candidate_upgrade_patterns.md` (other agents' Loop 1 outputs)
- Core source: `chelation_adapter.py` (residual MLP + OrthogonalProcrustesAdapter/Cayley, near-identity init, L2 normalize), `self_healing_chelation.py` (SEAL/EGGROLL-inspired planner, sandbox executor, CandidateProvenanceLedger, retention/structural/quant gates), `sedimentation*.py`, `antigravity_engine.py`, `stability_tracker.py`, `online_updater.py`

**Core Thesis of This Scan**: The CHELATEDAI chelation system (lightweight residual embedding correction `x + δ(x)` + normalize, diagnostic-driven SelfEditDirectives, retention/quantization/structural-health filters, sedimentation training on collapse logs) is a **natural fit** for OPSD-style on-policy self-distillation. However, core OPSD papers focus heavily on privileged-context teacher vs. student rollouts for reasoning/math with reverse-KL token-level objectives. Complementary literatures provide:
- Mechanisms for **generating higher-quality self-edits** (self-critique/revision, reflection, process supervision).
- Stronger **continual/retention guarantees** across multiple self-healing cycles (SDFT on-policy from demonstrations/diagnostics).
- **Adapter/LoRA-specific stability** (input-logit KL alignment, generic-task KL replay, distribution-gap bridging self-distillation).
- **Denser, more reliable fitness signals** for directive acceptance and sedimentation (process vs. outcome PRMs, ReST-MCTS filtering).
- **Efficient synthetic data loops** without external teachers or massive human labels.

These close critical gaps identified in the audit (decoupled advisory planner vs. actual training, off-policy sedimentation risks, retention collapse over iterations, sparse credit assignment, lack of process-level self-correction).

---

## 1. Self-Rewarding LLMs and Iterative / Self-DPO Family

### Foundational Paper
- **Self-Rewarding Language Models** (Yuan, Pang, Cho, Li, Sukhbaatar, Xu, Weston et al., Meta FAIR + NYU), arXiv:2401.10020 (Jan 2024, v3 Mar 2025).  
  https://arxiv.org/abs/2401.10020 | PDF: https://arxiv.org/pdf/2401.10020

**Core Technique**: The LLM simultaneously generates responses (policy) **and** acts as its own judge/reward model via LLM-as-a-Judge prompting on its own outputs. Creates synthetic preference pairs (chosen/rejected) from self-evaluations. Trains with **Iterative DPO** (or variants). Over 3 iterations on Llama-2-70B, the model improves both instruction-following **and** its own reward-modeling ability. Outperforms Claude 2 / Gemini Pro / GPT-4-0613 on AlpacaEval 2.0.

**Key Innovations Relevant to Chelation**:
- **Closed self-improvement loop** without external reward models or human labels (directly maps to SelfEditDirective generation + fitness/reward from internal diagnostics in `SelfHealingChelationPlanner`).
- **Iterative refinement**: Each cycle uses improved model for both generation and judging → virtuous cycle. Matches the "self-healing" aspiration (multiple AdaptiveValidationRound loops).
- **LLM-as-Judge for preferences** → can be adapted to embedding-space "preference" (better vs. worse chelation correction on synthetic probes or retrieval pairs).

**Follow-ups**:
- **Process-based Self-Rewarding Language Models** (Zhang, Liu et al., arXiv:2503.03746, Mar 2025) — Extends to **step-wise** LLM-as-Judge (pairwise comparisons per reasoning step) + step-wise DPO. Addresses degradation from pure outcome self-rewarding. Introduces segmented instruction data + PRM initialization.
- **Temporal Self-Rewarding Language Models** (arXiv:2508.06026).

**Mapping to CHELATEDAI + OPSD Upgrade**:
- Enhance `generate_directives()` in `self_healing_chelation.py` with self-rewarding judge prompts over synthetic_examples + diagnostics (structural_health, retrieval_anomaly, quantization_failure).
- Turn accepted `SelfEditEvaluation` into preference pairs for **Iterative DPO on the ChelationAdapter** (or low-rank deltas) instead of (or in addition to) current ES/sedimentation.
- Combine with OPSD privileged-context: teacher (with full diagnostics + "correct" correction trace) judges student (normal context) rollouts.
- **Testable Variance**: "Self-Rewarding Chelation DPO" — run 3-5 iterative cycles where each cycle's accepted directives generate new preference data for DPO on adapter weights (using `online_updater.py` micro-updates or full sedimentation_trainer). Metric: retention score delta + NDCG lift after each iteration vs. baseline single-cycle.

This family provides the **outer-loop iteration engine** missing from current advisory self-healing.

---

## 2. STaR: Self-Taught Reasoner and Bootstrapping Reasoning Family

### Foundational Paper
- **STaR: Bootstrapping Reasoning With Reasoning** (Zelikman, Wu, Mu, Goodman, Stanford), arXiv:2203.14465 (Mar 2022).  
  https://arxiv.org/abs/2203.14465

**Core Technique**: Iterative bootstrapping for chain-of-thought (CoT) reasoning without massive rationale datasets.
1. Few-shot prompt with rationales.
2. Generate rationales + answers on unlabeled questions.
3. For incorrect final answers: **rationalize** by prompting with correct answer + generate rationale that leads to it.
4. Fine-tune **only on rationales that led to correct answers**.
5. Repeat — improved model solves harder problems.

**Extensions**:
- **Quiet-STaR** (Zelikman et al., arXiv:2403.09629, 2024) — Internal "thinking" tokens before output (hidden CoT).
- **V-STaR** (arXiv:2402.06457) — Train verifier on both correct and incorrect traces (positive + negative).
- Later: START, B-STaR, RL-STaR, STaR-SQL, etc.

**Why Complementary to OPSD/Chelation**:
- STaR's **rationalization trick** (retry conditioned on ground truth/correct outcome) is a powerful data-efficiency hack for self-correction. Directly applicable to "failed" SelfEditDirectives or sedimentation collapse events: re-generate correction traces conditioned on successful retention/quantization outcome.
- Focus on **filtering only successful reasoning traces** for training → maps perfectly to `CandidateProvenanceLedger`, `SelfEditEvaluation.accepted`, retention_replay_guard, and quantization_gate filtering.
- Addresses **credit assignment** in long reasoning/self-edit sequences (relevant for multi-step diagnostic → directive pipelines).

**Mapping & Testable Patterns**:
- In `SelfHealingChelationPlanner.generate_directives()` or sandbox executor: for directives that fail retention/structural_health, trigger a "rationalization" pass — condition generation on "correct" (high-fitness) outcome and synthesize improved synthetic_examples.
- **STaR-style Chelation Bootstrapping**: Collect "failed chelation traces" (queries where adapter caused collapse or low retention) + "successful correction" from ledger. Fine-tune adapter (or generate new directives) only on successful rationalized corrections.
- Synergy with OPSD: Use on-policy student trajectories, but filter/re-weight with STaR-style success rationales before distillation loss.
- **Variance for Testing**: "Rationalized Self-Edit Replay" — augment retention_replay_guard with STaR rationalized versions of borderline directives. Measure sample efficiency (fewer directives needed for same retention gain) and stability over 10+ self-healing cycles.

This fills the **data synthesis + filtering gap** for high-quality self-edit examples.

---

## 3. ReST: Reinforced Self-Training and Variants (ReST-MCTS*, etc.)

### Foundational Paper
- **Reinforced Self-Training (ReST) for Language Modeling** (Gulcehre, Paine et al., Google DeepMind), arXiv:2308.08998 (Aug 2023).  
  https://arxiv.org/abs/2308.08998

**Core Technique**: Growing-batch RL-inspired self-training.
- **Grow phase**: Sample from current policy to build/expand offline dataset.
- **Improve phase**: Score/filter samples with reward model (or human prefs), then update policy with offline RL or SFT on high-quality subset.
- Data reuse across iterations; more efficient than pure online RLHF.

**Major Extensions**:
- **ReST-MCTS*** (Zhang et al., arXiv:2406.03816, 2024) — Integrates **Process Reward Model (PRM)** guidance + Monte Carlo Tree Search (MCTS) for collecting high-quality reasoning traces. Infers per-step process rewards from final-answer oracles via rollouts (no manual step annotations). Iterative EM-style self-training. Strong gains on math/reasoning over outcome-only ReST^EM.
- **Re-ReST** (Reflection-Reinforced), ReST^EM, ReST meets ReAct (tool use), ReST-RL (code), etc.
- Later works use MCTS + self-generated PRMs for fully automated process supervision.

**Relevance to Chelation**:
- **Grow/Improve** directly analogous to current `run_sedimentation_cycle()` + directive evaluation: generate candidate corrections (grow), apply retention/quant/fitness gates (improve/filter), reinforce only survivors.
- **MCTS + PRM** for process-level guidance: Instead of sparse final "fitness" on whole directive, break SelfEditDirective into steps (e.g., anomaly detection → synthetic example synthesis → optimization param choice → adapter delta application) and assign process rewards. This gives **denser signals** for the planner and for distillation loss (better than pure outcome in current `FitnessEvaluation`).
- ReST's offline dataset reuse → perfect for `CandidateProvenanceLedger` + chelation_log replay buffer (avoid catastrophic sample waste in current system).

**Mapping & Variants**:
- Integrate MCTS-style search **over directive generation** in SelfHealingChelationPlanner (multiple synthetic_example branches, tree of adaptation_modes).
- **Process-supervised sedimentation**: Extend `sedimentation_loss.py` (InfoNCE/Hybrid) with PRM-style step rewards derived from intermediate diagnostics (isomer_detector, structural_health_score, topology_analyzer outputs).
- **Testable Pattern**: "ReST-MCTS Chelation" — Wrap directive generation in lightweight MCTS (expand with different strategies: implication_synthesis, retrieval_ttt, eggroll_es, structural_repair). Use PRM (could be a small head on diagnostics or another LLM-as-judge) to score partial paths. Only fully successful leaves go to adapter distillation. Expected: higher acceptance rate + better long-horizon stability than flat generation.

This family provides **search + process-dense filtering** to make self-healing more reliable and data-efficient.

---

## 4. Constitutional AI: Self-Critique and Self-Revision Loops

### Foundational Paper
- **Constitutional AI: Harmlessness from AI Feedback** (Bai, Askell, Chen, et al., Anthropic), arXiv:2212.08073 (Dec 2022).  
  https://arxiv.org/abs/2212.08073

**Core Technique** (two phases):
- **SL-CAI (Supervised)**: Sample response → **self-critique** against a "constitution" (list of principles) → **self-revise** the response to fix violations → SFT on revised outputs. Chain-of-thought in critique/revision improves quality. Can iterate revisions.
- **RL-CAI (RLAIF)**: Generate response pairs → AI feedback model (same LLM + CoT) scores which better follows constitution principles → train preference/reward model on AI labels → PPO (or DPO) on policy.

Principles are human-written but explicit/inspectable (harmlessness, honesty, etc.). Produces non-evasive assistants that explain objections.

**Follow-ups**: Collective/ public constitutional AI, inverses, multi-constitution ensembling.

**Why Powerful for Chelation Upgrade**:
- The **critique → revise** pattern is a perfect **meta-layer** for `SelfEditDirective` generation and evaluation.
- Current directives are generated heuristically from diagnostics. A Constitutional-style loop inside the planner would let the system **critique its own proposed correction** ("Does this delta violate retention? Will it survive quantization? Does it address the root structural_health issue?") and **revise** the directive/synthetic_examples/optimization_params before even sandbox execution.
- "Constitution" for chelation = explicit principles: "Preserve base retrieval knowledge (min_retention_score >= 0.8)", "Minimize ||delta|| to avoid KL shock", "Ensure quantization survival", "Maintain structural health and topology", "Improve fitness on self-generated probes without regression on prior ledger entries".
- Self-critique makes the **advisory** planner much smarter and self-consistent before costly training.

**Mapping & Testable Patterns**:
- Add `generate_critique_and_revise_directive()` method to `SelfHealingChelationPlanner` or a new `ConstitutionalChelationCritic` class. Use few-shot constitution prompts (or in-context diagnostics) to critique/revise candidate directives.
- Chain multiple revisions (like paper's monotonic harmlessness improvement with more revisions).
- For OPSD integration: Use the revised (higher-quality) directive as the **privileged teacher context** for on-policy distillation of the adapter correction.
- **Variance**: "Constitutional Self-Healing Chelation" — Every generate_directives() call runs 1-3 critique-revise rounds against a chelation constitution. Only revised directives enter the ledger and distillation. Expected win: dramatically higher acceptance rate in `SelfEditEvaluation`, fewer wasted sedimentation cycles, better long-term retention (fewer "regret" corrections).

This supplies the **self-reflective / meta-correction** capability that pure OPSD privileged-context distillation lacks (OPSD assumes good privileged traces; Constitutional generates/refines them).

---

## 5. Continual / Lifelong Learning via Self-Distillation (SDFT and Extensions)

### Key Paper
- **Self-Distillation Enables Continual Learning** (Shenfeld, Damani, Hübotter, Agrawal; MIT/Improbable/ETH), arXiv:2601.19897 (Jan 2026).  
  https://arxiv.org/abs/2601.19897 (closely related to the SDPO authors)

**Core Technique — SDFT (Self-Distillation Fine-Tuning)**:
- Exploit in-context learning: Same model acts as:
  - **Teacher**: Conditioned on prompt `x` + expert demonstration `c` → produces demonstration-aware distribution `Q`.
  - **Student**: Conditioned only on `x` (normal context) → produces `P`.
- Train by minimizing **reverse KL** (`D_KL(P || Q)`) on **on-policy trajectories** generated by the student itself.
- Often uses EMA of student params for teacher stability.
- Results: Learns new skills/knowledge while **substantially reducing catastrophic forgetting** vs. standard SFT. Enables true sequential/continual multi-skill accumulation without regression on prior capabilities or unrelated benchmarks (MMLU, etc.). Better generalization (in/out-of-distribution).

**Related**:
- **Self-Distillation as a Performance Recovery Mechanism for LLMs: Counteracting Compression and Catastrophic Forgetting** (arXiv:2604.15794, Apr 2026) — Explicitly uses self-distillation to recover from compression-induced forgetting.
- **Self-Evolving LLM Agents through an Experience-Driven Lifecycle** (arXiv:2510.16079) — Offline experience self-distillation (distill trajectories into reusable strategic principles) + online interaction for lifelong agents.
- Surveys on continual LLM learning (arXiv:2603.12658) discuss self-distillation + replay constraints.

**Direct Relevance to Chelation Pain Points** (from audit: retention collapse over cycles, off-policy sedimentation, catastrophic forgetting of base embeddings):
- Chelation's core problem is **continual self-correction without destroying prior knowledge** (base model retrieval, previous successful adapters, structural health).
- `SelfEditDirective` synthetic_examples + diagnostics can serve as the "expert demonstration `c`".
- The ChelationAdapter (or its low-rank/Procrustes variant) is the perfect low-capacity student being distilled.
- On-policy nature + reverse KL keeps the correction close to the "demonstration-aware" (privileged diagnostic) behavior while updating only on the model's own current generations → exactly what is needed for stable multi-cycle self-healing.
- EMA teacher or fixed reference adapter (base + identity) prevents drift.

**Mapping & High-Priority Variants**:
- **SDFT-Chelation Adapter Training**: In `sedimentation_trainer.py` or new `opsd_chelation_trainer.py`, for each accepted directive:
  - Teacher forward: condition (via prompt engineering or auxiliary input) on full diagnostics + "successful correction example".
  - Student: normal query + current adapter.
  - Loss: reverse KL (per "token" or per embedding dimension / retrieval pair) + task loss (InfoNCE or fitness).
  - Regularize with EMA of previous successful adapter state.
- Use in `online_updater.py` for micro continual updates.
- **Testable Pattern "SDFT Retention Guard"**: Replace or augment current `retention_replay_guard` + simple replay with full SDFT loop. Run sequential self-healing campaigns (new knowledge batches) and measure: (a) new-task gain, (b) retention on old ledger entries / BEIR / structural health after N cycles. Expected: SDFT variant maintains >0.85 retention where baseline sedimentation drops below 0.7.
- Synergy with ProcrustesAdapter: The orthogonal constraint + SDFT KL provides double protection against forgetting.

This is one of the **highest-leverage complementary papers** — SDFT was published in the same Jan 2026 wave as OPSD/SDPO and directly solves the "continual self-correction" problem the chelation system was designed for but has not yet achieved.

---

## 6. Stability-Focused Self-Distillation with Adapters / LoRA / PEFT

Key papers in this family (strong practical guidance for the lightweight ChelationAdapter):

1. **Prompt Distillation (PD): Efficient Knowledge Injection via Self-Distillation** (Kujanpää et al., arXiv:2412.14964, 2024/2025).
   - Teacher (with new knowledge context `c` + query) generates answers. Student (base + LoRA, zero-init) sees only query. KL (high-temp soft logits) between teacher/student on answer tokens.
   - Explicit **regularization KL** on generic/unrelated instruction-response pairs (from Tülu 3 or similar) to prevent forgetting.
   - LoRA + PD outperforms SFT at every rank; better resistance to MMLU/general capability drop.
   - **Mapping**: The "new knowledge" = diagnostic context / successful correction trace. Generic KL replay = retention_replay on previous successful queries or base model outputs. Perfect for ChelationAdapter (already residual + near-zero init, like LoRA).

2. **SelfAug: Self-Distribution Alignment via Input Logits** (Huang et al., arXiv:2509.03934, 2025).
   - During fine-tuning (LoRA), add **KL loss aligning fine-tuned model's logits on the input sequence** to the original frozen model's logits on the same inputs.
   - Directly targets distribution shift (strong predictor of forgetting severity). Works especially well in RAG/long-context (highly relevant to retrieval-focused CHELATEDAI).
   - Often beats orthogonal-loss or replay baselines.

3. **SDFT (arXiv:2402.13669 variant)** — Self-distilled rewriting of task responses to match model's own distribution before fine-tuning (bridges distribution gap, reduces forgetting even with raw LoRA).

4. Other: BA-LoRA (KL consistency on LoRA), O-LoRA / C-LoRA (orthogonal task-specific LoRAs + proximal terms), PESO with softmax-KL, KL replay buffers in continual LoRA.

**Common Stability Toolkit Extracted** (not all in core OPSD papers):
- KL on **input / prefix / generic context** (not just response tokens) — anchors the model.
- High-temperature softening for softer targets.
- Explicit replay of "safe" previous behaviors (retention probes).
- EMA teachers or fixed reference models.
- Distribution-gap bridging (rewrite data in model's voice).
- Proximal / trust-region terms (aligns with MIS-PO).

**Mapping to CHELATEDAI**:
- In `ChelationAdapter.forward` or a wrapper loss: compute **input-logit (or embedding) KL** between (base model without adapter) and (base + current adapter) on the raw input embeddings/queries. Add as regularizer (weight scheduled like OPSD KL).
- Use Prompt Distillation style: privileged diagnostic context only for teacher path; normal for student + adapter.
- Extend `regularization_loss()` in ChelationAdapter (currently returns 0.0 for MLP) and implement for Procrustes variant.
- **High-Priority Testable Pattern**: "Input-Anchored + Generic-KL Chelation". Combine with existing Procrustes (already orthogonal) + new input-KL term + generic retention replay. This directly attacks the "adapter drift" and "KL shock on non-target queries" failure modes from the audit.

These papers give **concrete, implementable regularizers** that pair beautifully with the existing near-identity init and L2-normalize in ChelationAdapter.

---

## 7. Process vs. Outcome Supervision and PRM Self-Training

Survey: **A Survey of Process Reward Models** (Zheng, Zhu et al., arXiv:2510.08049 v3, Oct 2025 / Apr 2026). Excellent overview of automated/self-supervised PRM methods.

Key works:
- **Process-based Self-Rewarding** (arXiv:2503.03746) — step-wise LLM-as-Judge + step DPO.
- **ReST-MCTS*** (arXiv:2406.03816) — PRM-guided MCTS for process rewards inferred from outcomes + self-training.
- **PRL: Process Reward Learning** (arXiv:2601.10201, 2026) — Theoretical decomposition of outcome RL into process signals.
- Unsupervised PRMs, generative PRMs ("PRMs that think"), FreePRM, SP-PRM, rStar-Math, ThinkPRM.

**Value for Chelation**:
- Current fitness (`FitnessEvaluation`) and directive acceptance are largely **outcome-based** (final retrieval fitness, retention score, quant gate pass/fail after full directive execution).
- Process supervision provides **intermediate credit** (e.g., "was the anomaly detection accurate?", "did the synthetic_examples address the specific collapse type?", "was the low-rank ES population diverse enough?").
- This leads to better directive filtering, better loss shaping in distillation, and more sample-efficient learning (credit not only at end of long self-edit chain).
- Self-supervised PRM training (from own rollouts + inferred process labels) fits the "no external teacher" OPSD ethos.

**Mapping**:
- Extend `FitnessFunctionInterface` and `SelfGeneratedEvalProbe` with process-level annotations (step-wise structural health, partial retention on sub-probes, topology repair success at each stage).
- Train a lightweight PRM head (or use LLM-as-judge on diagnostic traces) on ledger data.
- Use process rewards to shape the distillation loss (token/embedding-step level weighting) or to filter in ReST-MCTS style during directive search.
- **Variance**: "PRM-Guided Chelation Self-Training" — integrate into `SelfEditSandboxExecutor` and sedimentation. Measure improvement in "self-healing success rate" (accepted directives per generated) and reduction in wasted training steps.

This addresses the **sparse reward / poor credit assignment** pain point identified across multiple agent audits.

---

## 8. Additional Complementary Directions & Novel Ideas

- **Experience-Driven Self-Evolving Agents** (arXiv:2510.16079): Offline distillation of interaction trajectories into reusable "strategic principles" + online refinement. Maps to distilling successful `SelfEditDirective` patterns + provenance ledger into a persistent "chelation principle bank" or feature_direction_bank that seeds future generations.
- **Reflection-Augmented Loops** (Re-ReST, Reflexion-style): Add explicit reflection step after failed directives ("Why did retention drop? What was the root cause in diagnostics?") before next generation. Can be cheap LLM call or structured analysis using existing `isomer_detector.py`, `topology_analyzer.py`, `structural_health_score.py`.
- **Unsupervised / Generative PRMs**: Fully self-supervised process reward models (no human step labels). Useful for CHELATEDAI's retrieval/embedding domain where step annotations are expensive.
- **Orthogonal / Procrustes + Self-Distillation Hybrids**: The repo already has `OrthogonalProcrustesAdapter` (Cayley param, inspired by Drift-Adapter arXiv:2509.23471). Combine with SDFT/PD input-KL for "doubly stable" adapters (geometric constraint + distributional anchor). Novel: "Procrustes-SDFT Chelation".
- **Quantization-Aware Self-Training**: Explicitly mentioned in EGGROLL/SEAL inspirations. Look for papers on int8/4-bit aware self-distillation or LoRA + QAT in continual settings (complements the `QuantizationPromotionGate`).
- **Multi-Model / Ensemble Self-Critique**: Beyond single-model constitutional; use small ensemble or router (adapter_router.py already exists) for more robust judging of directives.
- **Synthetic Collapse Benchmark Synergies** (`synthetic_collapse_benchmark.py`): Use generated hard negative collapse cases as the "expert demonstrations" for SDFT teacher paths.
- **KL Control Nuances from Broader Lit**: Per-position clipping, JSD_β (already in some OPSD follow-ups), unbiased estimators, high-divergence region filtering (from @ar0cket1), plus input-sequence anchoring and generic replay from PD/SelfAug papers.

**Novel Idea Not Heavily in Core OPSD**: **"Diagnostic-Privileged Process Self-Distillation"** — Break the chelation correction itself into process steps (anomaly classification → probe synthesis → delta proposal → normalization check → retention simulation). Apply on-policy self-distillation at each process step using privileged intermediate diagnostics. This gives ultra-dense signals tailored to the embedding/retrieval domain.

---

## 9. Synthesis: Most Promising Additional Directions & Why They Matter for Viable Chelation

| Research Area              | Key Paper(s)                  | Primary Gap It Fills in Current Chelation + Core OPSD                          | Expected Impact on Upgrade Viability |
|----------------------------|-------------------------------|--------------------------------------------------------------------------------|--------------------------------------|
| Self-Rewarding + Iterative DPO | 2401.10020, 2503.03746       | Outer-loop iteration + self-judge for directive quality & preference data     | High — turns advisory into true iterative self-improver |
| STaR Bootstrapping         | 2203.14465 + Quiet/V-STaR    | High-quality synthetic data synthesis + rationalization of failures            | High — sample efficiency for self-edit examples |
| ReST + MCTS/PRM            | 2308.08998, 2406.03816       | Process-dense filtering, search over correction strategies, data reuse         | Very High — better acceptance rate, credit assignment |
| Constitutional Self-Critique/Revision | 2212.08073                | Meta self-reflection on proposed edits before training                         | Very High — fewer bad directives, self-consistent constitution |
| SDFT Continual Self-Distillation | 2601.19897 + 2604.15794     | On-policy continual learning from "demonstrations" (diagnostics) w/o forgetting | **Critical** — solves multi-cycle retention collapse |
| Adapter/LoRA Stability (PD, SelfAug, input KL) | 2412.14964, 2509.03934     | Input-anchored KL + generic replay for distribution control in PEFT            | High — directly stabilizes ChelationAdapter drift |
| Process Supervision / PRM  | 2503.03746, 2510.08049 survey | Dense step-level fitness instead of sparse outcome                             | High — better directive eval & loss shaping |

These are **not redundant** with core OPSD; they supply the "scaffolding" (generation, filtering, reflection, continual anchoring, process credit) around the privileged-teacher on-policy distillation core.

---

## 10. Concrete Testable Upgrade Patterns & Variances Emerging from This Scan (Prioritized for Loops 2-10)

1. **SDFT-Privileged-Diagnostic Chelation (Highest Priority)**: Implement SDFT-style teacher (diagnostics + successful correction trace in context) vs student (normal + adapter) reverse-KL on on-policy embedding trajectories. Wire into sedimentation_trainer + online_updater. Use EMA adapter reference. Test in sequential knowledge-injection campaigns + retention tracking. (Direct from 2601.19897 + OPSD synergy).

2. **Constitutional Chelation Critic**: New `ConstitutionalChelationCritic` module with chelation-specific constitution (retention, quant survival, structural health, minimal delta). Critique-revise loop inside `generate_directives`. Feed revised directives to OPSD distillation. Measure acceptance rate lift and long-cycle stability.

3. **ReST-MCTS Directive Search + PRM Fitness**: Lightweight MCTS over adaptation strategies + synthetic example variants. Lightweight PRM (on existing diagnostics or small head) for process scoring. Only high-process-reward leaves enter ledger/distillation. (ReST-MCTS 2406.03816).

4. **Input-Anchored + Generic-KL Regularized Adapter**: Extend `ChelationAdapter.regularization_loss()` and Procrustes variant with SelfAug-style input embedding KL (base vs adapted on raw queries) + PD-style generic retention replay KL. Combine with existing L2 normalize and near-zero init. (2412.14964 + 2509.03934).

5. **STaR-Rationalized Self-Edit Replay**: For failed/low-reward directives, trigger rationalization pass (condition on high-fitness outcome) to synthesize improved examples. Add to retention_replay_guard. (STaR 2203.14465).

6. **Process-Supervised Sedimentation Loss**: Extend InfoNCE/Hybrid losses with step/process rewards (partial structural health, probe success, isomer repair). Weight distillation gradients accordingly. (Process Self-Rewarding + survey).

7. **Self-Rewarding Iterative DPO on Adapters**: After each self-healing round, convert accepted vs. rejected directive outcomes (or teacher vs student rollouts) into preference pairs. Run 2-3 iterations of DPO (or SDPO variant) on ChelationAdapter params (low-rank or full). Use `adapter_router.py` for multi-adapter management.

8. **Experience-Principle Distillation Bank**: Periodically distill high-success ledger entries + provenance into a compact "chelation strategy memory" (principles or few-shot bank) that seeds future `generate_directives`. (Self-evolving agents 2510.16079). Integrate with `feature_direction_bank.py` or model_scope.

9. **Quantization-Aware Process PRM**: Train PRM explicitly on quantization survival outcomes at multiple bit-widths. Use during filtering and loss. (Complements EGGROLL + new quant papers).

10. **Hybrid Reflection + OPSD**: After directive execution (success or fail), explicit reflection step (using existing analyzers) produces improved "next-cycle privileged context". Use for subsequent on-policy distillation.

11. **Multi-Constitution Ensemble Judge**: Router or small ensemble for more robust self-reward / critique (leverage existing `adapter_router.py`).

12. **Unsupervised PRM for Embedding Corrections**: Fully self-supervised process reward model derived from consistency across multiple probes / topology metrics (no external labels).

**Implementation Recommendations**:
- Start with SDFT (Pattern 1) + Constitutional (Pattern 2) + Input-KL (Pattern 4) as they have the most direct code hooks (`self_healing_chelation.py`, `chelation_adapter.py` regularization, `sedimentation_trainer`).
- Create `opsd_chelation_extensions.py` or augment existing trainers with a `SelfDistillationChelationTrainer` class that accepts teacher_context_fn (diagnostics/ledger) and student_adapter.
- All patterns should expose flags for A/B testing in `run_road_course_campaign.py`, `run_sweep.py`, `test_self_healing_chelation.py`.
- Metrics to track (extend existing): correction_gain, retention_score_delta over cycles, structural_health trajectory, quantization_survival_rate, sample_efficiency (directives or tokens per unit fitness lift), "self_healing_persistence" (performance after 10+ autonomous cycles without external intervention), KL shock frequency.

---

## 11. Risks, Open Questions, and Synergies

**Risks**:
- Over-engineering the outer loop (critique, MCTS, PRM) could add latency/complexity; start lightweight (1-2 critique iterations, shallow MCTS).
- Teacher conditioning in SDFT/PD must not leak privileged info into student at inference (use strict separation; only during training).
- Process PRM hacking / reward model collapse — mitigate with OPSD-style filtering of high-KL regions + human/ledger review in early loops.
- Quantization interaction: all new losses must be tested through the existing `QuantizationPromotionGate`.
- Compute: Iterative self-loops (Self-Rewarding, ReST grow/improve) are expensive; leverage existing `compute_budget_policy.py` and `evolution_strategies_optimizer.py`.

**Open Questions for Later Loops**:
- How to best represent "privileged diagnostic context" for embedding models (aux tokens? separate encoder head? in-context via model_scope?).
- Optimal temperature / KL beta scheduling when distilling residual adapters (vs. full LLM policy).
- Can the existing `CandidateProvenanceLedger` serve as a high-quality replay buffer for SDFT/generic KL without extra storage?
- Interaction with computational storage / disk-resident aspects (many retention issues may be I/O or serialization related).

**Synergies with Other Agents**:
- Agent 2 audit + Agent 6 self-edit integration: The constitutional critic and SDFT patterns can be wired directly into the "conversion layer" they proposed.
- KL stability (Agent 5): All new regularizers (input KL, generic replay, process weighting) should be combined with their JSD_β / clipping / high-divergence filtering.
- Sample efficiency (Agent 7): STaR rationalization + ReST-MCTS + PRM filtering dramatically reduce wasted directives.
- Evaluation harness (Agent 8/11): All patterns need the multi-cycle continual retention + self-healing success rate benchmarks they are designing.

---

## 12. Recommendations for Subsequent Loops

- **Loop 2 (Architecture)**: Prioritize SDFT + Constitutional + Input-KL-anchored adapter as the "core three" upgrade patterns. Design the `SelfDistillationChelationTrainer` and `ConstitutionalChelationCritic` modules with clean interfaces to existing `FitnessFunctionInterface`, ledger, and AntigravityEngine.
- **Loop 3 (Losses)**: Prototype the reverse-KL on-policy + input-anchored KL + process-weighted hybrid losses. Include ablations on temperature, beta scheduling, EMA strength.
- **Loop 4+**: Stress-test all patterns on long-horizon continual self-healing (new document batches over 20+ cycles) using the synthetic_collapse_benchmark and road-course campaigns. Include quantization promotion gates at every step.
- Prototype in a new `chelation_self_distillation.py` or extend `self_healing_chelation.py` and `sedimentation_trainer.py`.
- Cross-pollinate with @ar0cket1 Hermes repo patterns (MIS-PO filtering, online LoRA self-improvement) as noted in Agent 1.
- Update `task_plan.md`, `findings.md`, and `CHELATION_OPSD_RESEARCH_PLAN.md` with these new patterns and references.
- Consider adding key papers to `REFERENCES.md`.

---

## References (Key Papers Cited)

1. Self-Rewarding Language Models — arXiv:2401.10020
2. Process-based Self-Rewarding Language Models — arXiv:2503.03746
3. STaR: Bootstrapping Reasoning With Reasoning — arXiv:2203.14465
4. Quiet-STaR — arXiv:2403.09629
5. Reinforced Self-Training (ReST) — arXiv:2308.08998
6. ReST-MCTS*: LLM Self-Training via Process Reward Guided Tree Search — arXiv:2406.03816
7. Constitutional AI — arXiv:2212.08073
8. Self-Distillation Enables Continual Learning (SDFT) — arXiv:2601.19897
9. Self-Distillation as Performance Recovery... — arXiv:2604.15794
10. Prompt Distillation (PD) — arXiv:2412.14964
11. SelfAug: Self-Distribution Alignment — arXiv:2509.03934
12. A Survey of Process Reward Models — arXiv:2510.08049
13. Self-Evolving LLM Agents... — arXiv:2510.16079
14. PRL: Process Reward Learning — arXiv:2601.10201
15. Drift-Adapter (existing repo inspiration) — arXiv:2509.23471
16. Core OPSD (for cross-ref) — arXiv:2601.18734
17. SDPO — arXiv:2601.20802

Full PDFs and follow-up citations available via arXiv. Many have code (SDFT project page, SDPO TRL integration, ReST-MCTS repos, etc.).

---

**Conclusion**: The broader literature provides a rich, immediately actionable toolkit that, when layered on top of core OPSD privileged-context on-policy distillation and the existing CHELATEDAI chelation primitives (residual adapters, SelfEditDirectives, retention/quant gates, ledger, sedimentation), creates multiple **viable, testable paths** to stable, sample-efficient, continual self-healing chelation. SDFT + Constitutional critique + process PRM + adapter-specific input-KL stability stand out as the strongest complements.

This report, combined with the other Loop 1 artifacts, gives the program a complete research foundation. Ready for architecture design (Loop 2) and implementation of the top patterns.

**Next Action for Swarm**: Orchestrator to synthesize all 10+ agent reports into a unified "Loop 1 Synthesis & Prioritized Upgrade Patterns" document, then task Loop 2 agents with concrete module designs and pseudocode for the top 4-5 patterns.

---

*Report generated autonomously as Research Agent 9. All claims traceable to cited arXiv papers. All mappings grounded in direct code audit of chelation_adapter.py, self_healing_chelation.py, and related modules.*