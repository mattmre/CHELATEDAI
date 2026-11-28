# W22-16 — QSCCI singleton probe audited against its kill criterion

Date: 2026-11-26. This is a CPU-only audit of evidence that already exists. No new GPU run was made.

**Kill criterion (from `wave22_queue_additions_2026-11-25.json`, W22-16):**
- If the selection methods pick identical winners, there is no treatment separation.
- Kill the nuisance-ranking advantage if the k=1 native neuron or the SAE latent ties ordinary contrastive
  selection on all three of these, measured on held-out data:
  - answer margin,
  - hard-negative misfires,
  - collateral KL.

## Evidence examined

| Run | Where | Model / SAE | Verdict recorded in the run |
|---|---|---|---|
| QSCCI v4 run-001 (2026-08-16) | `D:\GITHUB\CHELATEDAI\artifacts\method-dev\qscci-v4\run-001\qscci.json`, untracked in the main checkout. The artifact sha256 is `043f5b39…`. | Qwen3.5-2B-Base L11 with the official `Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_100` | `DOES_NOT_SURVIVE_SMALL_LABEL_ORACLE_FIXTURE` |
| W22C b52 live (34125969) | `artifacts/pv7_w22c_b52_qscci_live/w22c_b52_qscci_live_manifest.json`, run on Colab L4 | Qwen2.5-1.5B and Qwen3.5-2B-Base. The SAE was self-trained: TopK, 4,096 latents, 120 steps. | QSCCI beats CAA, SAE and sham with a strictly positive LCB |

## Finding 1 — the kill criterion already fired in v4 (the run with the official SAE)

- In `qscci.json` under `choices`, `chelated|null` (the nuisance-penalized selection) and `contrastive|null`
  (ordinary contrastive selection) chose the **same** feature set, `[15620, 17083, 3067, 18817]`, at k=4 and α=4.
- The gate `treatment_separation["contrastive|null"]` is `false`.
- The gates `contrastive_advantage_at_least_0_05`, `material_completeness_at_least_5_6` and
  `mismatch_no_greater_than_controls` are also `false`.
- Identical winners is the first clause of the kill criterion. **The nuisance-ranking advantage is killed in the
  SAE basis.**
- **The kill fired at the criterion's own k=1.** The v4 selection grid scored both selectors at k = 1, 4 and 8 for
  α = 0.5, 1, 2 and 4 (`aggregates.cell_endpoints["SELECT|grid|{chelated,contrastive}|null|k|α"]`):
  - At **k=1**, the chelated and contrastive endpoints are identical for every α. For example, at α=4 both have
    `bidirectional_causal_contrast` 0.4826 and `canonical_nuisance_mismatch` 0.2188.
  - At **k=4** they are identical for every α too.
  - They first differ at **k=8**: at α=0.5, for example, `mean_correct_gain` is +0.0174 for chelated and −0.0069 for
    contrastive.
  - The frozen choice (k=4, α=4) is therefore one of the identical cells, not a special case.
- **Correction history:** round 1 of this audit said the selection ran "at k=4, not the k=1 named in the criterion".
  That was wrong, and a round-2 Tier B reviewer caught it: the grid includes k=1.
- **The scope that remains:** v4 is a small label-oracle fixture with one model, one layer and one SAE. The kill
  does not show that nuisance-penalized selection can never separate from contrastive selection on a larger fixture
  or another model.

## Finding 2 — the W22C "QSCCI win" is selected on its own outcome and does not test W22-16

`scripts/run_colab_l4_w22c_b52_qscci_live_gpu.py`:

- **How the direction is chosen (lines 507–517).** `qscci_dir = unit(mean_SELECT(d margin / d h_11))`. This is the
  first-order steepest-ascent direction of the logit margin (` positive` minus ` negative`).
- **What REPORT measures (lines 546–556; the arms are called at 566–570).** `_eval_report_arm` injects every arm at the same L2 dose and scores the
  family-mean change in **that same margin**.
- **SELECT and REPORT are separate splits, so this is not data leakage.** The direction is estimated on SELECT
  prompts and scored on held-out REPORT prompts. The problem is that the selection *objective* is the same quantity
  as the reported *endpoint*. Held-out evaluation does not remove that advantage.
- **Why it wins by construction.** At a fixed norm, the gradient direction maximizes the first-order change in the
  margin. CAA (the class-mean difference) and the SAE latent are selected for class separation, not for margin
  ascent, so they are not competing on the scored quantity. The Qwen3.5 result (QSCCI 1.131 vs CAA 0.008, cosine
  0.002 between the two directions) is what that construction predicts. It is not evidence of a mechanism.
- **The push is toward " positive" regardless of class.** The mean change is +1.106 on positive prompts and +1.156
  on negative prompts for Qwen3.5. For Qwen2.5-1.5B it is +0.602 and +1.164, so it is larger on negatives. This is
  a bias shift, not class-conditional steering.
- **There is no causal-mechanism signal.** Two-pass restoration is 0.016 for QSCCI vs −0.005 for the sham on
  Qwen3.5, and −0.015 vs −0.008 on Qwen2.5.
- **The "native" arm is not a neuron.** `k1_native_mlp_write_site` is a dense direction over 2,048 dimensions. The
  literal single-coordinate neuron (`k1_single_coordinate_mlp_neuron`, also chosen by argmax |grad|) gets 0.109,
  vs 0.027 for the SAE and 0.037 for the sham.
- **The SAE comparator is a self-trained SAE, not the official W32K one** (`topk_sae_training`: 120 steps, k_active 16).
  How weak it is depends on the model:
  - **Qwen3.5-2B-Base:** the selected latent's `selected_latent_select_diff` is **−0.164**, the wrong sign. It still
    scores 0.027 on REPORT, below the sham's 0.037.
  - **Qwen2.5-1.5B:** the select difference is **+0.384**, the right sign, so the "wrong-sign latent" excuse does not
    apply. It scores **−0.012** on REPORT (CI [−0.033, +0.010]), against +0.012 for the sham and −0.020 for CAA.
  - So on Qwen2.5 a correctly signed SAE latent fails on the scored margin as well. That fits the
    selection-on-outcome reading above: neither SAE nor CAA was selected for margin ascent.
- **Parts of the kill criterion were never measured in W22C:**
  - hard-negative misfires;
  - collateral KL per arm (only the QSCCI hook is measured);
  - a nuisance-penalized vs ordinary selection contrast.

## Disposition

- **Nuisance-ranking advantage: KILLED** on the v4 fixture. The identical-winner clause fired with the official SAE
  at k=1 and at k=4, and the selectors diverge only at k=8. See Finding 1 for the remaining scope (one small fixture,
  one model, layer and SAE).
- **Tier B history:**
  - Round 1 scored this audit 93 and asked for two additions. SELECT and REPORT being separate splits is folded in.
    Its "fixture scope is k=4" request was itself wrong.
  - Round 2 scored it 85. It found that k=4 error, the Qwen2.5 SAE numbers, and a line citation off by two. All three
    are fixed above.
- **W22C bundle #52's "QSCCI beats CAA/SAE" headline: should be SUSPENDED** as selection-on-outcome.
  - What survives is a narrower statement: a gradient-chosen write direction moves the target margin more than
    non-gradient directions at an equal dose, with low collateral KL (0.0069 nats on Qwen3.5).
  - That is expected first-order behavior, not a mechanistic finding.
  - Status: PROPOSED, pending a fresh adversarial Tier B review.
- **A W22-16 live rerun is not justified on the current evidence.** It would be worth running only with all of
  these:
  - the official W32K SAE;
  - selection on a SELECT objective that differs from the REPORT endpoint;
  - class-conditional scoring (correct-side gain minus wrong-side gain);
  - hard-negative misfires;
  - per-arm collateral KL.

  The Qwen3.5-2B-Base weights and the official SAE are not in this machine's HF cache, so a 3090 rerun would also
  need a download of about 5 GB. This is carried as an optional side bet. It does not block anything.
