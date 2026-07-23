# Round 1 decisions (architect: Claude) — 2026-07-10

Swarm: Grok-4.5-low design red-team + Codex-gpt5.6-low offline-cache scope. Purpose: de-risk before GPU.

## Grok design red-team verdict (out_r1_grok_designredteam.txt)

### Obj A — `oracle_margin_mean` at scale → PROCEED-WITH-CHANGES
Binding changes before any pack is built:
1. **Preregister the gap-only null.** Primary = univariate `oracle_margin_mean` OLS **block-LOO**
   Spearman; it MUST significantly beat both `oracle_gap→R` and `mean-R` nulls on **rank AND MAE**.
   Margin is pure oracle geometry, so absent this it just re-labels "oracle ranking ease" as
   recoverability (oracle-gap→R already scores +0.60 on n=4).
2. **Block-LOO, not cell-LOO.** Hold out a whole dataset OR whole encoder family. Effective n = #
   independent (dataset × encoder-family) blocks, NOT hyperparam cells. Report block-n; claim power
   only at block-n. Drift-magnitude/anchor-fraction expansion of the same pair = pseudo-replication.
3. **Freeze everything before building:** single primary metric, feature list, bins, success
   inequality. No q50/sign_rate fallback "if mean fails." Optional locked partial-correlation test for
   increment over gap.
Prior: expect regression from +1.0 toward ~gap-level (~0.6) or noise unless it survives block-LOO.

### Obj B — β-calibrated D2 kill-screen → **CUT**
Unwinnable-by-construction. Collapse `x←x+β(c−x)` is cluster-affine + clean-target-aligned → ridge/CBIE
are the natural solution class; bounded LOCAL chelation's premise (sparse residual harm after the best
global map) is not what the process generates. β-calibrating to a gap band selects regimes where
oracle≫floor → global maps a priori strong → rebuilds the oracle hazard. No β is both discriminating
AND fair to local chelation. **Do not spend GPU on the β-sweep.**

## Architect decisions
- **Obj B β-sweep: CUT.** Keep β=0.10 artifacts as regime-calibration-failure documentation only.
- **Obj B PIVOT:** the only meaningful home-turf test is a **non-affine residual regime** — a
  corruption that leaves residual local harm the best global ridge-on-anchors map CANNOT fit, with the
  recovery object = **oracle-vs-ridge residual gap** (not oracle-vs-floor). This must be red-teamed
  before any build. Candidate generators (Grok): sparse doc-level adversarial moves, query-conditioned
  local swaps, label-sensitive neighborhood collapse. If it can't be made fair+discriminating, Obj B
  stays cut and is documented as "chelation home-turf is not constructible with our tools."
- **Obj A: PROCEED** under the tightened block-LOO + must-beat-gap protocol, sized by the cache scope.
