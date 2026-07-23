# Chair edit-list for the paper

This file is an application list only; the git-excluded paper was not modified.

## Replace all ceiling claims

Use **observed plateau under this protocol**. This wording does not claim a mathematical or universal ceiling.

| Source line | Section | Current text containing `ceiling` | Required edit |
|---:|---|---|---|
| 26 | Abstract (draft — methodology framing) | **highest recovery we observe under this protocol**, not a proven ceiling: on 60 queries the study is | Replace each `ceiling` noun phrase with `observed plateau under this protocol`; recast grammar where needed. |
| 81 | 1. Introduction | ceiling. The lesson: always benchmark the trivial linear map before claiming a drift corrector | Replace each `ceiling` noun phrase with `observed plateau under this protocol`; recast grammar where needed. |
| 495 | 5.5 The trivial-baseline hazard: the loop is beaten ~4.2× by a least-squares map | observed recovery under this protocol, not a proven ceiling. The tested S1 doc2query self-pairs | Replace each `ceiling` noun phrase with `observed plateau under this protocol`; recast grammar where needed. |
| 501 | 5.5 The trivial-baseline hazard: the loop is beaten ~4.2× by a least-squares map | **(ii) ~84% is the highest recovery we observe under this protocol — not a proven ceiling.** A | Replace each `ceiling` noun phrase with `observed plateau under this protocol`; recast grammar where needed. |
| 505 | 5.5 The trivial-baseline hazard: the loop is beaten ~4.2× by a least-squares map | post-hoc ceiling, "no nonlinear benefit", or an information-theoretic bound: ridge−MLP | Replace each `ceiling` noun phrase with `observed plateau under this protocol`; recast grammar where needed. |
| 617 | 7. Open questions and remaining pre-registration | ceiling — the nonlinear MLP is statistically indistinguishable, §5.5). The teacher-supervised variant C3b (full re-embeddings of anchor docs as | Replace each `ceiling` noun phrase with `observed plateau under this protocol`; recast grammar where needed. |
| 675 | 8. Limitations | tight method-equality or "linear ceiling" claim — a ridge−MLP half-width ≤0.015 would require ≈800+ | Replace each `ceiling` noun phrase with `observed plateau under this protocol`; recast grammar where needed. |

## Mandatory leakage-sensitivity disclosure

The literal waypoint 600-document fit contains 28 of 62 eval-positive documents (45.2%). The leakage-safe full-600 fit excludes all of them. The 84.3% headline is therefore a leaky-fit point estimate.

| Method | Literal waypoint recovery | Leakage-safe full-600 recovery | Inflation (pp) |
|---|---:|---:|---:|
| ridge | 84.3% | 78.9% | +5.4 |
| closed-form orthogonal Procrustes | 82.3% | 73.9% | +8.4 |
| residual MLP | 81.6% | 67.5% | +14.1 |

State explicitly that the primary ladder/bootstrap is a literal-waypoint reproduction, while the full-600 sensitivity and document learning curves are leakage-safe. Do not equate the safe n=128 learning-curve regime with the literal 600-document headline.

## Mandatory location-agnostic inference-ban block

Paste this prohibition wherever the ridge−MLP result or plateau interpretation is discussed:

> Do not interpret ridge−MLP non-significance as evidence of a linear post-hoc ceiling or the absence of nonlinear benefit. The 95% CI on ΔNDCG(ridge−MLP) is [−0.0333, 0.0773] — it permits MLP better by up to ~4 recovery points and ridge better by up to ~10. G2 FAILED (primary ridge−MLP half-width 0.055 ≫ 0.015; the secondary median across both preregistered contrasts is 0.089), so this study is underpowered for the equality contrast on 60 queries. Non-significance here is absence of evidence, not evidence of absence.

### Forbidden claims

- No **linear post-hoc ceiling**.
- No **no nonlinear benefit** or **capacity is linear**.
- No **84% is the recoverable upper bound**.
- No **irreducible 16%**.
- No G2-powered precision on method contrasts.

### Permitted claims

- Report point estimates plus 95% CIs on the stated protocol.
- Ridge is the highest-observed method here.
- Ridge ≫ C3a is Holm-significant.
- Ridge versus MLP is not distinguishable at α=0.05 under this n.
- Prefer **observed plateau under this protocol**.

## Replace point-only D1 numbers with uncertainty-aware values

Apply these at every matching abstract, Introduction, §4.2, §5.5, §6/§7, and Limitations occurrence; keep the full precision in tables and round prose to one recovery point / three NDCG decimals.

| Quantity | Chair-ready replacement |
|---|---|
| Ridge | NDCG 0.6812 (95% CI 0.5838–0.7731); recovery 84.3% (95% CI 77.0%–91.0%) |
| Residual MLP | NDCG 0.6593 (95% CI 0.5619–0.7512); recovery 81.6% (95% CI 74.2%–88.6%) |
| C3a (per-query, not scalar) | NDCG 0.1570 (95% CI 0.0749–0.2488); recovery 19.4% (95% CI 9.4%–30.6%) |
| Ridge − MLP | ΔNDCG 0.0219 (95% CI -0.0333–0.0773); Holm-adjusted p=0.430957 |
| Ridge − C3a | ΔNDCG 0.5242 (95% CI 0.3968–0.6411); Holm-adjusted p=0.00039996 |

## Location-specific instructions

- Abstract and §1: replace the point-only 84%/20% comparison with ridge and C3a recovery plus their CIs; describe ridge as the best observed plateau under this protocol.
- §4.2: define recovery exactly as already written, but replace the quoted `~85% linear ceiling` interpretation with the observed ridge plateau and its CI.
- §5.5 table/caption/body: replace all ridge, residual MLP, closed-form orthogonal Procrustes, and C3a entries from `ladder.json`; add the paired contrast CIs and Holm results. State that the ladder's closed-form SVD map is distinct from the trainable Cayley adapter, and that C3a was rescored per query through the merged NDCG implementation.
- Remove the old C3a `0.1609` scalar wherever it is presented as this fixed split. The actual seed-42 per-query mean is 0.1570; `0.1609` was the mean across three different seeded splits.
- §6 and §7: do not call the unexplained residual an irreducible 16%; the bootstrap measures sampling uncertainty, not an information-theoretic bound.
- §8: replace `fixed ... ceiling` with `observed plateau under this protocol` and add the fixed 60-query eval-split limitation.
- Learning-curve text/figure: show two panels. Ridge/MLP x-axis is paired document embeddings; C3a x-axis is distinct non-eval query anchors. Do not overlay them on a shared `n anchors` axis.

G2 is **failed**. The primary ridge−MLP half-width is 0.0553; the secondary median across both preregistered contrasts is 0.0887.
