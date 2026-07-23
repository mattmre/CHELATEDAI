# Obj B pivot — design proposal: sparse-local non-affine drift ("chelation's honest home turf")

Architect (Claude) proposal, for Grok adversarial red-team BEFORE any GPU. β-collapse was CUT
(cluster-affine → global ridge trivially wins). This proposes the only regime class where bounded,
detector-gated, *local* chelation could plausibly beat a global map — and confronts the harder
question of whether it also beats a per-cluster *local* ridge (else chelation is redundant).

## The four properties a fair+discriminating chelation home turf must have
1. **Sparse harm** — only a small fraction s of clusters are corrupted; the rest stay clean. (This is
   where **detector-gating** earns its keep: a blanket correction damages healthy clusters.)
2. **Local & non-affine** — no single global linear map W inverts the harm, and it is non-affine even
   within a cluster (so global ridge/Procrustes provably fail; the oracle-vs-ridge residual is real).
3. **Low supervision** — few held-out anchors per harmed cluster (this is where a bounded near-identity
   correction's regularization can beat an *overfit* per-cluster local ridge).
4. **Small magnitude** — the per-doc harm is bounded, so a near-identity correction is in-class.

## Generative model (proposed)
- k-means the clean corpus; select a sparse fraction s∈{0.15,0.25} of clusters as "harmed".
- Within each harmed cluster apply a **distinct non-affine local warp**: a per-cluster smooth radial
  distortion (r ↦ r·(1+γ·f(r))) composed with a small per-cluster rotation, parameters drawn per
  cluster so (a) no single global W inverts all harmed clusters and (b) the radial term is non-affine.
- Calibrate γ so mean(oracle NDCG − floor) lands in a discriminating band [0.05, 0.20] — but here the
  gap is genuinely non-globally-linear, not ridge-invertible (unlike β-collapse).

## Baselines chelation MUST beat (the real bar — not just global ridge)
- Global ridge / orthogonal Procrustes  → expected to FAIL (per-cluster-varying non-affine warp).
- **Per-cluster local ridge, parameter-matched**  → THE HARD BASELINE.
- CBIE / ZCA / all-but-top / hubness  → isotropy/score baselines.
- Oracle = clean re-embed.
- Chelation = bounded (calibrated α) + harm-detector-gated, unpaired.

## Recovery object (report both)
- oracle − floor recovery (headline), AND
- **oracle − best-global-map(ridge) residual recovery** — the residual ridge leaves on the table is
  what a local corrector must close.

## The decisive question for the red-team (answer before we spend GPU)
Does bounded detector-gated chelation have a **non-tautological edge over the per-cluster local ridge**
in the low-supervision limit — via (i) the bound's regularization when anchors/cluster are scarce, and
(ii) detector-gating that spares healthy clusters — OR does the per-cluster local ridge (also
detector-gatable, also parameter-matched) simply win, making chelation redundant?

- If chelation has NO edge the local ridge doesn't also get → **Obj B CLOSES** as "no constructible
  regime where chelation's specific mechanism beats a simpler local corrector," which is a valuable,
  publishable conclusion and needs NO GPU.
- If chelation has a plausible edge (low-supervision regularization + sparse-harm gating) that the
  baselines don't trivially capture → build it, with the per-cluster local ridge as the co-primary
  contrast under the locked dual-CI gates.

## Questions for Grok
1. Is this regime fair to local chelation AND discriminating, or does it still smuggle in an advantage
   for some baseline (e.g. does the per-cluster local ridge fit the radial warp given enough anchors,
   making low-supervision the ONLY axis where chelation could win — and is that axis real or a knob to
   hand chelation a rigged win)?
2. Is "bounded near-identity beats overfit local ridge at low supervision" a real phenomenon or wishful
   — would a ridge with a properly tuned λ (also a form of bound) erase chelation's edge?
3. Is detector-gating on sparse harm a genuine chelation-specific advantage, or does every baseline get
   the same gate for free?
4. Verdict: **BUILD / BUILD-WITH-CHANGES / CLOSE**. If CLOSE, state the one-line honest conclusion for
   the paper. If BUILD, the single most important design change + the co-primary baseline to beat.
