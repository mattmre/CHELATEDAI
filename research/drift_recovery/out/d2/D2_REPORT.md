# D2 kill-screen report (with C3 adversarial-review honesty fixes)

**Procedural verdict:** `NO_GLOBAL_G3_VERDICT_UNDERPOWERED_INVALID_OR_UNAVAILABLE`.
**Scientific verdict:** *inconclusive because the β=0.10 synthetic-collapse regime was mis-calibrated
(drift too mild), not a clean detector-gated-chelation kill.*

The only judged correction is `chelation_primary` (unbounded displacement, detector-gated, unpaired).
The α=0.05 arm is a separate design-premise stress test; the paired local adapter is diagnostic only.
`NO_G3_VERDICT` is **not** a corrector-kill claim and **not** a corrector-win claim.

## The headline caveat (C3): the regime is degenerate at β=0.10
Every cell's recoverable oracle gap (mean oracle NDCG − mean floor NDCG) is ≈0 or **negative** —
i.e. pulling 8/16 clusters 10% toward their own clean centroid barely hurts retrieval, and on
NFCorpus it sometimes *helps* the floor. With no recoverable gap, **G3 was a priori unpassable**: the
locked `min oracle gap ≥ 0.05 per seed` precondition fails everywhere by an order of magnitude, and
24–76% of bootstrap gap-draws are non-positive (invalid). This is a **protocol miss** — β=0.10 was
hardcoded with no severity/discriminating-band calibration (unlike D1 §4.4). A confirmatory D2 needs a
preregistered β-calibration step that sets β so the oracle gap lands in a discriminating band *before*
running; until then this crossover cannot render a G3 verdict.

| Cell | G2 half-width | min oracle gap (per seed) | point gap (mean) | invalid gap draws | max\|chel−floor\| NDCG | Harm AUPRC (pos/N) | verdict |
|---|---:|---:|---:|---:|---:|---:|---|
| SciFact/8  | 0.0190 (FAIL) | +0.0000 | +0.0019 | 1.53%  | **0.0e+00 (identity)** | 0.225 (5/160) | NO_G3_VERDICT |
| SciFact/16 | 0.0148 (pass) | −0.0046 | +0.0017 | 23.76% (INVALID) | **0.0e+00 (identity)** | 0.151 (12/160) | NO_G3_VERDICT |
| NFCorpus/8 | 0.0275 (FAIL) | −0.0078 | −0.0022 | 76.15% (INVALID) | 2.4e−04 | 0.269 (14/160) | NO_G3_VERDICT |
| NFCorpus/16| 0.0255 (FAIL) | −0.0091 | −0.0010 | 60.86% (INVALID) | 4.7e−03 | 0.307 (33/160) | NO_G3_VERDICT |

## What the columns do and do not say
- **Chelation does essentially nothing here.** On *both* SciFact cells the detector-gated primary arm
  leaves eval NDCG **bit-identical to the floor** (max|chel−floor| = 0.0 across all 5 seeds); on
  NFCorpus it moves NDCG by ≤5e−3. Any positive Δ vs CBIE/hubness is "doing nothing beats a harmful
  adapter," not recovery. (An earlier "Primary movement" column reported the *vector displacement
  fraction*, not an NDCG change — dropped here to avoid implying action.)
- **AUPRC 0.15–0.31 is not an independent detector kill.** Harm labels are counterfactual per-cluster
  NDCG loss (not injection membership), verified. But harm barely exists: positive-cluster prevalence
  is 3–21%, median positive harm ~3e−4–2e−3 (near the 1e−8 label floor), and SciFact/8 has 2
  single-class seeds. AUPRC < 0.80 here is a **symptom of regime mildness**, not evidence the detector
  can't rank real harm. We therefore do **not** invoke the locked "AUPRC<0.80 → KILL_CORRECTOR" path;
  the G2/invalid-gap gates short-circuit first and that is the honest stopping point.
- **No hidden chelation win.** NFCorpus/16 `chel−ridge` looks flattering (ΔNDCG +0.022, CI
  [0.0001, 0.048], 5/5 positive seeds) but fails the dual-CI rule: CBIE **beats** chelation on both
  NFCorpus cells (all seed Δ negative), `chel−hubness` CI includes 0, Holm does not reject
  (adj p≈0.14), and the min-oracle-gap≥0.05 precondition fails. Correctly not claimed as a win.

## Locked-rule compliance (C3-spot-checked in code, not test names)
G2-hard-gates-G3, dual-CI (≥0.05 recovery ∧ ≥0.02 NDCG ∧ CI>0 ∧ 5 same-sign seeds ∧ min gap ≥0.05),
Holm family frozen to `{cbie, hubness, ridge}`, α=0.05 excluded from the decision family, harm =
counterfactual NDCG loss, detector train/eval query partitions disjoint, no eval-qrel leakage into
fit (audits show 0 contaminated), invalid-gap-fraction>1% blocks G3 — all verified present in code.
23/23 unit tests pass.

## Secondary caveat: the excluded pilot
An earlier SciFact pilot was excluded because it issued an *invalid* `KILL_CORRECTOR` (it lacked the
invalid-gap gate and fired at min_oracle_gap −0.0046 with ~24% non-positive gap draws). Protocol v2
added the identity-gate, clean-space routing, tied-AUPRC, parity, and invalid-gap-blocking fixes and
was rerun. The exclusion is honest (real implementation bugs, no valid win was buried), but it is a
*secondary* caveat — the **primary** caveat is the β=0.10 regime degeneracy above.

## Bottom line
This is an honest procedural "no G3 verdict." The scientific truth is that the synthetic-collapse
crossover at β=0.10 is non-discriminating (no recoverable gap), so it neither kills nor revives
detector-gated chelation. A decisive kill-screen requires a severity-calibrated β that produces a real
oracle gap; that is the one experiment still owed before chelation's home-turf claim can be settled.
