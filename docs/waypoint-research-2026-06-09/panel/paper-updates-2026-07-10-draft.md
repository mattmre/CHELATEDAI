# Paper updates draft — 2026-07-10 (chair applies; paper not touched here)

**Status:** Chair-ready paste blocks only. Do **not** apply to `paper-draft/main.md` from this note without chair review.

**Frozen sources (every numeric claim below):**

| Block | Artifact |
|---|---|
| Estimator (§7 subsection + Limitations) | `research/drift_recovery/out/estimator/objA_validation.md`, `objA_REPORT.md`, `objA_validation.json` (`generated_at_utc` 2026-07-10T19:56:45Z; `verdict` = `PROMISING-BUT-UNDERPOWERED`; `validated_positive` = false; `success_bar_met` = true) |
| Home-turf preflight (§7 paragraph) | `research/drift_recovery/out/d2b_preflight/preflight_report.md` (verdict **CLOSE**; 144 seed-runs; two families) |
| Synthetic kill-screen context (Limitations) | `research/drift_recovery/out/d2/D2_REPORT.md` (procedural `NO_GLOBAL_G3_VERDICT_…`; β=0.10 regime non-discriminating) |

**Hard bans enforced in all three blocks:** no “ceiling” claims; no “irreducible”; no “ridge ≈ MLP proves linearity”; no upgrade of **PROMISING-BUT-UNDERPOWERED** to validated.

**Rounding used in prose:** Spearman/partial to three decimals as requested by panel brief (0.8857→0.886, 0.8286→0.829, 0.7897→0.79); residual max rounded as 0.033 from strongest admitted-failure cells 0.0295 / 0.0328. Exact table values stay in artifacts.

---

## Block 1 — New §7 subsection (estimator)

**WHERE in `paper-draft/main.md`:** Insert as a new subsection under `## 7. Open questions and remaining pre-registration`, after the existing open-question bullets (after the “Larger scale” paragraph, before `## 8. Limitations`). Suggested heading level: `###` (sibling to none currently; §7 is flat bullets — either convert this to a bold lead-in paragraph block or add `### 7.1` / keep unnumbered bold title matching §7 style).

**Chair note (not paper text):** Primary feature was pre-registered as univariate `oracle_margin_mean` (`prereg_objA` id `objA-oracle-margin-block-loo-v1`). Nulls: `oracle_gap_ols`, `mean_R`. Offline scope: 6 independent dataset×encoder-family cells; **3** whole-dataset holdout units (SciFact, NFCorpus, FiQA2018); **2** encoder-family units (mpnet, bge-large). Do not call this validated.

### Proposed paper text (paste-ready)

**A margin-based recoverability estimator — promising but underpowered.**
Beyond reporting observed recovery fractions, we asked whether a cheap, leakage-safe scalar could rank regimes by residual recoverability *R* (ridge recovery of the oracle NDCG gap). We pre-registered a **univariate** predictor, `oracle_margin_mean` (mean per-query oracle margin on a leakage-safe fit index), and evaluated it with **block leave-one-out** at the independent-pair grain (dataset×encoder-family cells; no hyperparameter pseudo-replicates). On the offline six-cell inventory, block-LOO Spearman for `oracle_margin_mean` was **0.886** under whole-dataset holdout (MAE 0.0252) and **0.829** under whole-encoder-family holdout (MAE 0.0324), strictly beating the gap-only OLS null on both Spearman and MAE in each scheme (dataset gap-only Spearman **−0.086**, MAE 0.0773; encoder gap-only Spearman **0.429**, MAE 0.0696). Partial Spearman of margin with *R* after controlling for oracle gap was **0.79**, so the ranking signal is not a restatement of gap size alone. We still label the result **PROMISING-BUT-UNDERPOWERED**, not validated: the offline cache supplies only **three** dataset-level holdout blocks and two encoder-family blocks, so a met AND-bar is descriptive at small *n*, not a powered generalization claim. Within the two-cell FiQA2018 holdout alone, margin and gap-only both achieve perfect rank order (Spearman 1.0 on *n*=2) — a **tie** on the ranking metric, even though margin MAE is lower (0.0265 vs 0.0321) — so the headline LOO win must not be oversold as a clean third-dataset validation. A powered test would need a pre-registered expansion of independent holdout units (many more whole datasets and encoder families, still at true block grain, not anchor-fraction or seed cells), with the same strict “beats gap-only on Spearman *and* MAE under both block schemes” bar and an a priori sample-size target for the partial-correlation increment; until then we treat `oracle_margin_mean` as a promising ranking feature, not a deployed recoverability certificate.

---

## Block 2 — §7 paragraph closing the “home-turf” question

**WHERE in `paper-draft/main.md`:** Insert under `## 7. Open questions and remaining pre-registration`, as a new paragraph (or bold lead-in) **after** Block 1’s estimator subsection if both are applied, and **before** `## 8. Limitations`. If the chair prefers thematic grouping, place immediately after the existing **“Milder drift magnitudes.”** bullet (synthetic / severity-calibration theme) rather than after the estimator.

**Chair note (not paper text):** Obj B CPU preflight verdict is **CLOSE**. Grid = 2 warp families × 2 scale (`s`) × 4 strength (`gamma`) × 3 cluster counts (`n`) = **48 cells**; protocol audit **144 seed-runs**; residual gate **≥ 0.05** after **dev-selected, CV-λ gated local ridge** (dense not handicapped; low-rank alternative same λ candidates; method chosen on anchor-dev MSE only). Strongest residual cells: quadratic 0.0295, soft_fold 0.0328 — **no cell admitted**. Framing must **not** claim local ridge dominates chelation.

### Proposed paper text (paste-ready)

**Home-turf residual (synthetic sparse-local non-affine preflight) — CLOSE: no admitting residual, no second arena.**
A natural follow-up to the encoder-upgrade results is whether a *local*, detector-gated corrector could still win on a constructed “home-turf” residual: sparse non-affine warp of synthetic Gaussian clusters where a fair local map might leave recoverable NDCG after a strong local baseline. We ran a CPU-only preflight (no GPU, no embedding model; NumPy closed-form fits; routing inferred from corrupted vectors) over **48 cells** and **144 seed-runs** across two warp families (`quadratic`, `soft_fold`). Admission required residual NDCG after an anchor-dev-selected, **CV-λ gated local ridge** (fair λ / rank selection; eval queries select no hyperparameter) of **≥ 0.05**, plus displacement and purity sanity gates. **No cell admitted** (strongest residual left 0.0295 / 0.0328 by family — max residual ≈ **0.033** &lt; 0.05). The residuals are small primarily because **absolute oracle–floor gaps are small** and the gated local ridge **often fails to beat the no-op floor**, not because we demonstrated that local ridge dominates a bounded/annealed chelation corrector. The preflight therefore shows that **no discriminating home-turf residual was constructible** under these small-magnitude non-affine warps; it does **not** establish local-ridge superiority over chelation, and it supplies **no second arena** in which to re-argue the corrector. (Separately, the β=0.10 synthetic-collapse kill-screen in the D2 protocol is non-discriminating—recoverable oracle gaps ≈0 or negative—so it neither kills nor revives detector-gated chelation; that screen needs severity calibration before any G3 verdict.)

---

## Block 3 — §8 Limitations additions (1–2 sentences)

**WHERE in `paper-draft/main.md`:** Append inside `## 8. Limitations`, as new enumerated threats after existing (v) fit-set leakage (or as (vi)/(vii) if the chair keeps the numbered list style). Do not rewrite (iv) statistical power about ridge−MLP; these are **additional** limitations from Obj A / Obj B / D2 freezes.

### Proposed paper text (paste-ready)

(vi) **Recoverability estimator underpowered.** The pre-registered `oracle_margin_mean` block-LOO result (dataset Spearman 0.886; encoder 0.829; partial Spearman vs *R* controlling for oracle gap 0.79) meets its AND-bar against gap-only nulls but rests on only three dataset-level holdout units and is labeled **PROMISING-BUT-UNDERPOWERED**, not validated—including a disclosed FiQA-only holdout rank-order tie with gap-only. (vii) **Synthetic-only home-turf preflight.** The 48-cell / 144-run sparse-local non-affine CPU preflight admitted no residual ≥ 0.05 after fair gated local ridge (max residual ≈ 0.033) and remains synthetic Gaussian-cluster only; together with the mis-calibrated mild β=0.10 D2 regime, we claim neither a constructive home-turf win for local correction nor a powered synthetic kill of detector-gated chelation.

---

## Chair application checklist

- [ ] Paste Block 1 into §7; keep verdict wording **PROMISING-BUT-UNDERPOWERED** (never “validated”).
- [ ] Paste Block 2 into §7; keep corrected residual framing (small gaps + floor losses; **not** ridge≻chelation).
- [ ] Paste Block 3 into §8 as (vi)/(vii) or equivalent prose.
- [ ] Scan applied text for banned phrases: ceiling / irreducible / ridge≈MLP⇒linear / validated estimator.
- [ ] Optional figure/table pointers (artifacts only; not required for prose ship):
  - Estimator LOO table: `objA_validation.md` “Frozen block-LOO results”
  - Preflight grid: `preflight_report.md` full 48-row table
  - D2 cell table: `D2_REPORT.md` (context only; no G3 win/kill claim)

## Number audit (quick)

| Claim in draft | Source |
|---|---|
| `oracle_margin_mean` primary, univariate | `objA_validation.json` → `preregistration.primary_feature_names` |
| Dataset LOO Spearman 0.8857 / MAE 0.0252 | `objA_validation.md` table; JSON `block_loo.dataset.metrics.oracle_margin_mean` |
| Dataset gap-only Spearman −0.0857 / MAE 0.0773 | same |
| Encoder LOO Spearman 0.8286 / MAE 0.0324 | same, `encoder_family` |
| Encoder gap-only Spearman 0.4286 / MAE 0.0696 | same |
| Partial Spearman 0.7897 | JSON `partial_spearman.partial_spearman_margin_R_controlling_oracle_gap` |
| FiQA holdout: both Spearman 1.0; margin MAE 0.0265; gap 0.0321; beats_on_both false | JSON `fiqa_held_out` |
| 3 dataset / 2 encoder holdout units; cell-n 6 | validation.md “Achieved scale”; JSON `counts` |
| Verdict PROMISING-BUT-UNDERPOWERED; not validated | validation.md L3; JSON `verdict`, `validated_positive: false` |
| 48 cells, 144 seed-runs, two families | preflight grid length + “Protocol audit: 144 seed-runs”; families quadratic / soft_fold |
| residual gate ≥ 0.05; max residual 0.0295 / 0.0328 | preflight “Per-family admission” + frozen rules |
| CLOSE; small absolute gaps; local often ≤ floor; not ridge≻chelation | preflight verdict callout |
| D2 no G3; β=0.10 non-discriminating | `D2_REPORT.md` headline + bottom line |
