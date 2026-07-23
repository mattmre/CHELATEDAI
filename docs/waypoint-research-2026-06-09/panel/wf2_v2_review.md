# Adversarial review — Obj A v2 powered validation returned NEGATIVE. Disprove it (both directions).

Read `research/drift_recovery/out/estimator/objA_v2_REPORT.md`, `objA_v2_validation.md` + `.json`,
`estimator/prereg_objA_v2.md`/`.json`, and the new pack metadata under `out/estimator/packs/`
(arguana_*, *_e5_*). The powered run says: frozen-label **NEGATIVE** — at 12 cells / 4 dataset-blocks /
3 encoder-blocks, margin block-LOO rho = −0.7063 (dataset) / −0.6853 (encoder), LOSES to gap-only;
ArguAna and e5 holdouts individually agree. This reverses the 6-cell PROMISING result (0.886/0.829).

Attack BOTH ways:
1. **Is the NEGATIVE an artifact?** A rank INVERSION (+0.89 → −0.71) is drastic. Check the 6 new packs
   for construction bugs: ArguAna oracle gaps positive and plausible? R values in a sane range? qrels
   wired correctly (ArguAna is 1-positive-per-query — does the slice recipe keep each query's positive
   doc in the 1,200-doc slice, as the other datasets' recipe does)? e5 packs sane (768-d, projection
   recipe identical to mpnet's)? harness parity actually 1e-12 on all 6? Was prereg_v2 frozen BEFORE
   packs (check file ordering/hashes if recorded)? Recompute the block-LOO from the frozen features
   yourself and confirm the −0.7063/−0.6853 and the gap-only comparisons.
2. **Is the NEGATIVE real?** If packs are sound, confirm the honest story: margin's n=4/n=6 rank was
   inventory-overfit; on novel dataset+encoder blocks the relation inverts, so `oracle_margin_mean` is
   NOT a generalizing recoverability predictor. Check whether the v1 PROMISING label + this NEGATIVE
   are consistent artifacts (same code path, same frozen metrics) or whether v2 changed anything
   besides the regime set.
3. **Per-regime sanity:** list the 12 (regime, R, margin) pairs and eyeball the relation. If one pack is
   a wild outlier driving the inversion (e.g. ArguAna R near 0 or 1 from a qrel bug), name it — an
   artifact-driven NEGATIVE must be flagged, not shipped.

Deliver: PASS (negative is real) / PASS-WITH-FIXES / FAIL (artifact) + file:line specifics + a
one-sentence bottom line the paper can quote. Recompute, don't trust prose.
