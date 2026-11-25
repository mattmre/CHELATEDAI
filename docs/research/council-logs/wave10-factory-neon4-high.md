# Wave 10 Factory Engineering Memo (`L4` Demo D9 & `L5b` 70B Scale Sweep)

- **`panel_type`:** `architecture-planning`
- **`artifact`:** `docs/research/council-logs/wave10-r0-tier1-tier2-master-spec.md`
- **`run_date`:** `2026-11-23`
- **`run_id`:** `wave10-factory-l4-l5b-neon4-high`
- **`challenge_depth`:** `1` (Factory Audit — `L4` & `L5b`)
- **`active_experts`:** `["gemini-4-neon (Factory, High)"]`
- **`persist_version`:** `1.1.0`

**Reviewed Scope:** `wave10-r0-tier1-tier2-master-spec.md`, `scripts/run_wave6_d4_trained_epistemic_grpo.py`, `ek5_wfs_rlvr_env.py`, `chelated_epistemic_proposer_v2.py`, `prospective_realworld_drift_corpus.py`, `artifacts/wave6_d4_trained_epistemic_grpo/wave6_d4_epistemic_grpo_manifest.json`.

---

## 1. `L4` Root-Cause Diagnosis: Wave-6 `D4` `U`-Abstention Failure & Zero-Leak GRPO

Wave-6 `D4` stalled at `EVAL false_commit_on_u = 0.875` (`contested_successor = 0.00–0.05`, `odd_refutation_cycle = 0.20–0.30`) and non-significant `GRPO - SFT` (`+2.22 pp`, raw `p = 0.2248`, Holm `p = 0.4495`) due to four defects:

1. **Severe Under-Training (`16` Prompts):** `run_wave6_d4_trained_epistemic_grpo.py:2102-2124` used `per_device_train_batch_size=8`, `num_generations=8`, `grad_accum=1`, and `16` steps — only **1 prompt/step** (`16/192` prompts, ~5 `U` episodes), with low rollout entropy (`0.082–0.094`) and `mean_kl = 0.0013`.
2. **Truncated SFT Trace:** `derive_cot_from_prompt_text` (`:997-1033`) slices `target_ids[:6]` and `rel_notes[:4]` without tracing `eff_tier` drops or `cur=U` propagation. `SFT` started at `0.2344` on `GRPO_RL` (`dual_branch_unlinked_fork = 0.0312`, `false_commit_on_u = 0.9219`), starving GRPO of positive `U` rollouts.
3. **Topology Gap (`SPLIT_SCENARIOS:486-513`):** `SFT_WARMUP` only saw `peer_deadlock` and `silent_multi_current`; `GRPO_RL` only saw `laundered_peer_standoff` and `dual_branch_unlinked_fork`. Neither exposed >2-node cycles or indirect `U`-propagation.
4. **Abstention-Collapse Trap:** The spec's `-0.20` false-abstain penalty incentivizes **100% abstention** whenever determinate accuracy `p_det < 0.39` (`0.3125*1.05 - 0.6875*1.00 = -0.359 < -0.20`).

**Zero-Leak Fix & Verification:**
- Preserve `EVAL` (`N=120`, SHA `6b08233c...`) and `D1` (`N=60`) byte-identical (`assert_four_way_disjoint_splits`).
- Replace `derive_cot_from_prompt_text` with an untruncated 4-step WFS trace (full `eff_tier` table, edge qualification, fixpoint cycle/propagated-`U` check, verdict). Add disjoint motifs: `derived_ceiling_refute` (`SFT_WARMUP`), plus `five_node_refutation_ring` (5-cycle vs `EVAL`'s 3-cycle) and `derived_contested_chain` (`DERIVES_CURRENT` child of deadlocked peers vs `EVAL`'s `contested_successor`) in `GRPO_RL`.
- Run `160` steps with `grad_accum=2` (320 stratified `T/F/U` prompt-groups), `temperature=1.0`, `beta=0.02`, `R(U, commit)=-1.25`, and **`R(T/F, ABSTAIN)=-0.45`** (`E[attempt] > -0.45` for `p_det > 0.27`). Gate on `DEV abstain_rate <= 0.55`, `determinate_exact_match >= SFT - 0.02`, and Holm `p < 0.05` (`LCB97.5 > 0`) for 3-seed `GRPO > SFT` on `EVAL`.

---

## 2. `L4` Fast `0.5B/1.5B` LoRA Extractor & `AuthorityDominatedExocortexV2` Integration

- **Disjoint Synthetic Corpus (`N=600`):** Synthesize `150` docs/stance (`ASSERT`, `SUPERSEDE`, `REFUTE`, `DEBUNK_RUMOR`) across four non-Suite-A domains (`pharma_clinical`, `supply_chain`, `telecom_bgp`, `energy_scada`). Include hard-negative contrasts: Tier-1 "despite denials" echoes (`ASSERT`), debunks quoting a rumor alongside the confirmed value (`DEBUNK_RUMOR`, extracting the **confirmed** value), and Party-B counter-filings (`REFUTE`, extracting Party B's value — vital because `compile_claims_to_ek5_v2:272` skips equal-value pairs). Require `0` entity/slot/value overlap and `<0.25` 5-gram Jaccard vs `prospective_realworld_drift_corpus.py`.
- **Batched GPU Inference:** Build persistent `FastEdgeClaimExtractor` (`chelated_fast_edge_extractor.py`) with `merge_and_unload()` `bf16` weights on RTX 3090, `batch_size=17` (one Suite A case per `generate()` call), `max_new_tokens=48`, greedy decoding — targeting `<300 ms/case` (`0.5B`) / `<600 ms/case` (`1.5B`) vs `9,690 ms/case`, plugging unchanged into `AuthorityDominatedExocortexV2`. Benchmark `0.5B` and `1.5B` on a synthetic dev split first; promote the smallest checkpoint achieving `>=99%` JSON validity and `>=93.3%` Suite A accuracy (`LCB97.5 > 0` vs Mem0 `81.67%`).

---

## 3. `L5b` Fair Cross-Family Protocol: `Llama-3.3-70B-Instruct-bnb-4bit` vs `3B/7B + Chelated`

Evaluating `unsloth/Llama-3.3-70B-Instruct-bnb-4bit` on Spark #2 (`~40–44 GB` RAM, `NF4`/`bf16`) eliminates same-family bias against `Qwen2.5-3B/7B`:
- **Matched Contexts:** Run bare `70B` at BOTH `k=5` (HybridRAG parity) and `k=17` (`100%` authoritative-doc recall control) against `3B/7B + Chelated (k=5)` on `Suite A (N=60)` using Llama-3.3's native chat template, greedy decoding, and matched token caps.
- **Unbiased Judging & Statistics:** Primary metrics are deterministic `strict_exact_match`, `stale_leak_rate`, and `false_commit_on_u`. Secondary semantic judging uses **dual cross-family blind judges** (`Nemotron-30B` — disjoint from Llama and Qwen — plus `Qwen2.5-7B`, reporting Cohen's `kappa`), paired bootstrap `LCB97.5`, Tango score CI, exact Clopper–Pearson discordant CIs, Holm–Bonferroni correction, plus VRAM/latency/active-parameter ratios.

**Unknowns / Next Checks:** Verify Spark #2 has `>=50 GiB` free unified RAM before loading `70B-bnb-4bit`, and confirm `0.5B` two-value disambiguation fidelity on synthetic dev before locking extractor size.
