# Grok design red-team — HI-1 pre-registration (BEFORE any GPU, before freezing)

Read `docs/waypoint-research-2026-06-09/panel/prereg-harmonic-invariance-HI1-draft.md` in full, plus
the repo context it reuses (`research/drift_recovery/` packs/harness if present, `evidence_dag.py`,
`adapter_router.py`, the rung-16 results doc `docs/rung16-quant-aware-routing-results-2026-07.md`).

This is a pre-registration for the ONE genuinely-unpursued cell from a deep-research pass: *does a
gradient-free nonlinear geometric operator (anchor-relative + diffusion-harmonic) beat a strong linear
map (ridge/Procrustes) SPECIFICALLY in measurable unknown/OOD regions of a frozen embedding cloud?*

Context you must weigh: this program has produced FIVE consecutive fail-closeds where elaborate
structure lost to a trivial global linear baseline (drift corrector beaten 4.2x; recoverability
estimator inverted at powered scale; sparse-local home-turf preflight admitted nothing; H5 living bank
tied a frozen static bank; rung-16 routing plane FAIL-CLOSED on both arenas including a multi-domain
one). The honest prior is that HI-1 also fails.

## Attack it from BOTH sides — a false negative is as bad as a false positive

### A. Rigged-to-WIN (would manufacture a false positive)
1. Is the UNKNOWN-region estimator (k-NN mean cosine distance to FIT docs, 70th-percentile CAL
   threshold) gameable, circular, or correlated with the outcome by construction? Could selecting the
   "least-supported 30%" mechanically favor a smoother/harmonic method regardless of real structure?
2. Is the baseline L genuinely strong? "Best of ridge/Procrustes chosen on CAL" — is that the honest
   strong linear map, or should it include the leakage-safe full-fit / a tuned low-rank affine?
   Name the strongest linear baseline we are obliged to beat.
3. Parameter/。budget parity between G and L: is the stated parity real, or does G get more effective
   capacity (anchor count + eigenvector count + Nyström) than L's single λ?
4. Multiple comparisons: one primary hypothesis, but two datasets, KNOWN/UNKNOWN strata, and two
   estimators — is the frozen decision rule actually immune to shopping?

### B. Rigged-to-FAIL (would manufacture a false negative — equally disqualifying)
5. Does the design give the harmonic operator a REAL chance? Diffusion maps need adequate sampling
   density; the UNKNOWN region is BY DEFINITION low-density. Is Nyström out-of-sample extension into a
   low-density region doomed a priori — i.e. is HI-1 unwinnable-by-construction the way the beta=0.10
   collapse screen was? If so, say so now and propose the fix (e.g. build the graph on the union of
   FIT+drifted points, or use a density-adaptive kernel) or recommend CUT.
6. Is the +0.01 absolute NDCG lower-bound win threshold too strict given how few REPORT UNKNOWN queries
   there will be (30% of ~60-100 queries = ~20-30 queries)? Compute the achievable CI half-width at
   that n — if the study is underpowered to ever clear the bar, the design is a guaranteed fail-closed
   and must be re-scoped (more queries, or a paired-difference metric with more power).
7. Is "G must not lose on KNOWN (>= -0.005)" a fair constraint or an unnecessary handicap?

### C. Falsifiability and scope
8. The deep research flagged the SCOPE GAP: the proven superposition/compressed-sensing wedge is about
   feature readout WITHIN one space; HI-1 tests CROSS-space alignment. Does HI-1 actually test the
   theory, or a weaker analogy? If it only tests the analogy, is there a cheaper WITHIN-space test that
   would be more decisive (e.g. can a gradient-free harmonic readout beat a linear probe at recovering a
   known superposed feature)? Recommend whichever is more decisive per GPU-hour.
9. Any part of the hypothesis still unfalsifiable as written? Name the exact sentence.

## Deliver
`PROCEED / PROCEED-WITH-CHANGES / CUT`, plus:
- the single highest-value change,
- the strongest linear baseline we must beat,
- a power estimate at the realistic REPORT-UNKNOWN n and whether the win threshold is achievable,
- an explicit verdict on whether HI-1 is unwinnable-by-construction (the low-density/Nyström concern),
- and a one-paragraph bottom line: is this worth ~2 GPU-hours, or is it dominated by a known result
  and should be cut in favor of the within-space test (or cut entirely)?

Be brutally honest. Cutting a doomed experiment before GPU is a win — you already did that once this
program (the beta-sweep) and it was correct.
