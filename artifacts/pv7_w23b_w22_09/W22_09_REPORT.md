# W22-09 Retraction boundary: report

- Queue item `W22-09`. Branch `pv7/w23b-w22-09-retraction-state`, cut from spine `e5e0b037`.
- Preregistration: `PREREGISTRATION.md`, committed first as `c72921e6`.
- Code:
  - `retraction_boundary.py`
  - `scripts/run_w22_09_retraction_boundary.py`
  - `tests/test_retraction_boundary.py`
- Results: `w22_09_results.json`. Every number below is copied from that file, from the single full run at
  2026-11-26 03:33 local time.
  - Total runtime 630.8 s (A 115.7 s, B 422.4 s, C 79.4 s, Crouzeix 13.3 s).
  - CPU only: Python 3.11.9, numpy 2.4.5, Windows 11.

## Verdict: KILLED (proposed; Tier B round 1 scored 85, important)

**Corrected framing after Tier B round 1.**
- **K2 on #60 carries the verdict.** A fresh reviewer re-ran Part A three times. The B-FLAT/treatment latency ratio
  was 0.59, 0.39 and 0.41 at N = 16, and 0.78, 0.77 and 0.71 at N = 32, so K2 fired every time. Absolute latencies
  were about 3× lower than in the reported run, which points to a loaded machine during that run. Limit 2's claim
  that N = 16 "would plausibly flip" was wrong in direction. The result is robust, but close to tautological: replay
  matches replay, as the preregistration predicted.
- **The two K1 firings below are a certificate-coverage gap and a semantics note.** Neither is a way for a bundle to
  serve a wrong answer.
  - **`P51-DIGEST` cannot produce a wrong answer in #51.** #51 scores its answers from `inc_store`
    (`chelated_beam_128k_replay.py:891`), not from the resumed store. The replay feeds back the same turn objects
    (`:872`, `:879`), and an untampered control gives 0/20 differing answers. The checkpoint restore is separately
    checked by `tra1_state_digest`, whose rows do include `claim_value` (`:721-734`, `:783-786`). What the probe
    shows is that the `checkpoint_replay_exact` *metric* is weaker evidence than its name suggests. It collides two
    different input histories through the digest, and it involves no retraction.
  - **`PEK5-ASOF` fires only under the bitemporal reading.** The preregistration defines deletion as "the state that
    would exist had item `k` never been in the history", and under that definition H1 = ADMIT is exact. The firing
    depends on the dated-event reading the probe adopted.
- **The K1 bullets that follow are the original text.** Read them with the framing above.

- **K1 fires on a path the bundle uses.**
  - #51 certifies checkpoint-replay exactness by canonical-digest equality (`chelated_beam_128k_replay.py:885`).
  - That digest omits `claim_value`.
  - A replay whose post-checkpoint turn `aux_v2` carries a different `claim_value` gets the **same digest**. #51's
    criterion would report it exact, yet 3 of the 20 #51 questions get a different answer: `budget_125k` vs
    `budget_125k_DIVERGED`.
  - That is a state-collision pair that breaks exactness without a certificate.
- **K1 also fires on the native EK5 deletion API** (`PEK5-ASOF`), which #51/#62 do not call.
- **K2 fires for #60.**
  - B-FLAT means one flat KV file plus a prefix-slice restore and suffix replay.
  - At N = 32 blocks it matches the bundle treatment's deletion error (both ≤ 7e-13) with median latency
    0.93× and 0.89× the bytes.
  - At N = 16 it fires narrowly (latency 1.067×, inside the 1.10 threshold). Timing noise there is large; see
    Limits.
- **K2 does not fire for EK5 (#51/#62).** The geometric-checkpoint baseline is 2.8× to 418× slower at 1.68× to 1.86×
  the treatment's bytes.

What this means:

- Where the bundles actually do retraction, it is replay. The #60 KV cache is prefix reuse plus suffix replay, and a
  plain flat checkpoint matches it.
- Where they do incremental maintenance (the EK5 WFS), it does beat checkpoint-plus-replay at matched bytes.
- But the certificate #51 uses to call a replay "exact" does not cover the fields its answers read.

The item as titled cannot be run. No compressed neural state from which an item is retracted exists in #51, #60 or
#62 (preregistration §"plain finding"). Everything below is the scoped version.

## Kill-rule readout

| Component | Fires | Where | Counted in verdict |
|---|---|---|---|
| K1 `P51-DIGEST` | yes | #51 bundle path (`checkpoint_replay_exact`) | yes |
| K1 `PEK5-ASOF` | yes | native `withdraw_attestation_ids` + `as_of` (not called by #51/#62) | yes (prereg: both count) |
| K1 A3 order (i), logits | no | #60 | yes |
| K1 A3 order (ii), logits | no | #60 | yes |
| K1 A3 order (ii), request-prefix reading of invariant (b) | yes | #60 | **no**, see Departures D1 |
| K2 #60 B-FLAT, N=8 | no (latency 1.95×) | #60 | yes |
| K2 #60 B-FLAT, N=16 | yes (1.067×, 0.94× bytes) | #60 | yes |
| K2 #60 B-FLAT, N=32 | yes (0.93×, 0.89× bytes) | #60 | yes |
| K2 EK5 B-GEO-CKPT, all 6 workloads | no | #51/#62 | yes |

## Part A: #60 SSD KV cache (treatment = `SSDKVCache` + `serve_chelated`, unmodified)

- Setup: 3 model seeds (6001 to 6003) × N blocks; every block was deleted once, independently.
- Latency is one `perf_counter` measurement per deletion, in ms. Bytes are the mean over seeds.
- Error is the max abs difference of the last-position logits and of every KV entry vs `full_prefill` of the
  counterfactual prompt.

| N | Arm | median ms | p90 ms | max err | retained bytes |
|---|---|---|---|---|---|
| 8 | T60 treatment | 10.02 | 17.28 | 4.5e-13 | 121,294 |
| 8 | B-FLAT | 19.55 | 43.05 | 2.3e-13 | 117,429 |
| 8 | B-REPLAY (context) | 3.44 | 7.37 | 0 (is the reference) | 1,802 |
| 16 | T60 treatment | 33.64 | 59.47 | 7.0e-13 | 251,638 |
| 16 | B-FLAT | 35.88 | 95.65 | 3.5e-13 | 236,509 |
| 16 | B-REPLAY (context) | 43.63 | 84.01 | 0 | 3,634 |
| 32 | T60 treatment | 113.83 | 191.44 | 7.0e-13 | 526,390 |
| 32 | B-FLAT | 105.94 | 171.04 | 6.1e-13 | 467,909 |
| 32 | B-REPLAY (context) | 156.29 | 227.74 | 0 | 7,194 |

- **Treatment bytes are larger than B-FLAT** because every `.kvblk` header carries the full causal-prefix token list
  as JSON, which is O(N²) tokens in total.
- **The EK5 retraction mutation is excluded from the treatment timing, as preregistered.** It costs 0.084 to
  0.093 ms (median). Adding it would lower the B-FLAT/treatment latency ratio by 0.016 at N = 8 and by under 0.003 at
  N = 16 and 32.
- **The treatment refused 0 cached blocks on these deletions.** That is expected: the blocks before `k` do not
  contain the retracted version, and the blocks after it get new chain keys.
- **Per-seed medians at N = 16 (treatment / B-FLAT, ms):** 39.3 / 35.3, 32.7 / 17.4, 37.1 / 65.7. The pooled ratio
  of 1.067 sits inside the run-to-run spread.

**A1, splice without replay (the dirty cone).**

- Method: drop block `k`'s KV rows and keep every other row.
- Prefix rows stay exact in all 168 deletions.
- **100% of suffix KV rows are wrong** in both layers, in every deletion that has a suffix (21 / 45 / 93 deletions
  for N = 8 / 16 / 32).
- The max abs error reaches 32 to 39.
- Most of this is positional: with absolute sinusoidal positions every suffix token's position shifts, so even the
  layer-0 K/V change. With this model, no KV row after a deleted block can be reused without recomputation.

**A3, chain-key collision.** `v1 ≠ v2` with identical tokens at the same position have colliding chain keys (verified
`true`).

- Order (i): `v1` cached, then `v1` retracted, then the `v2` prompt is served.
  - The colliding block is refused (`refused_retracted_blocks = 1`) and recomputed. Logits error is 0.
  - This fails closed, but it is over-conservative: a clean `v2` is denied the cache.
- Order (ii): `v2` cached, then `v1` retracted, then the **`v1` prompt** is served.
  - All 3 blocks are served from the cache, with 0 refusals and `stale_served_blocks = 0`. Logits error is 0.
  - The request contains a retracted version, and a cached block is served for it without any flag.
  - The logits are exact because the content is identical. Provenance is not exact: the cache's `sources` record
    says `v2` and the request says `v1`.
  - The bundle never refuses a request that contains a retracted version, even with no cache involved. The
    refusal applies only to cached blocks whose recorded sources were retracted.

## Part B: #51/#62 EK5 store (treatment = `StratifiedEvidenceStore.apply_mutation(withdraw_attestation_ids=[a])`, unmodified)

- Store configuration: as #51 (`strict_dag_supersede=False, buffer_pending=False`).
- Deletions: 32 per rng seed × seeds {0, 1, 2} = 96 per workload.
- Error is the count of `val`/`cur`/`eligibility` diffs vs `reference_fresh_rebuild`.
- The warmed incremental store showed 0 diffs vs a fresh rebuild on every workload.

| Workload | events | T median / p90 ms | T bytes | GEO median / p90 ms | GEO bytes (ckpt positions) | REBUILD ms (context) | max diffs T/GEO/REB | GEO lat × | GEO bytes × |
|---|---|---|---|---|---|---|---|---|---|
| W-BEAM R=1 | 140 | 0.146 / 0.257 | 69,275 | 3.97 / 7.16 | 128,997 (70, 105) | 4.36 | 0/0/0 | 27.2 | 1.86 |
| W-BEAM R=4 | 560 | 0.178 / 0.591 | 237,410 | 17.39 / 32.74 | 434,612 (280, 420) | 20.73 | 0/0/0 | 97.9 | 1.83 |
| W-BEAM R=16 | 2,240 | 0.215 / 0.399 | 913,812 | 89.68 / 195.01 | 1,647,778 (1120, 1680) | 112.93 | 0/0/0 | 417.9 | 1.80 |
| W-CHAIN n=256 | 257 | 3.72 / 9.13 | 76,790 | 10.34 / 22.48 | 129,018 (128, 192) | 10.65 | 0/0/0 | 2.78 | 1.68 |
| W-CHAIN n=1024 | 1,025 | 18.74 / 44.07 | 313,533 | 59.57 / 91.62 | 530,697 (512, 768) | 58.79 | 0/0/0 | 3.18 | 1.69 |
| W-CHAIN n=4096 | 4,097 | 87.33 / 243.11 | 1,278,366 | 247.26 / 436.65 | 2,177,945 (2048, 3072) | 285.30 | 0/0/0 | 2.83 | 1.70 |

- **Dirty cone.** Median nodes transitioned per withdrawal:
  - W-BEAM: 1 at every size, so latency stays flat in R.
  - W-CHAIN: 137, 508 and 1,780. Treatment latency grows roughly linearly with n (3.7, 18.7, 87.3 ms).
  - That is the suffix blow-up. Even so, it beats checkpoint plus batch replay at ≤ 2× bytes, because the batch
    replays everything after the checkpoint, not just the cone.
- **Geometric checkpoints are coarse.** Only the first two ladder points fit under 2× bytes, so the median
  GEO deletion replays 257, 513 and 2,049 turns on W-CHAIN. On W-CHAIN n = 256, more than half the deletions fall
  before the first checkpoint and rebuild from scratch.
- **The bundle-native `REFUTE`-append (context, not K2) takes 0.12 to 0.22 ms median.**

**`P51-DIGEST` (K1, bundle path).**

- Setup: conversation `beam128k_conv_00`, checkpoint at step 3 (the #51 default), with the `claim_value` of the
  replayed turn `beam128k_conv_00:aux_v2` changed.
- Digest equal: `true`. Questions with different answers: 3 of 20 (q17, q18, q19).
- The divergence is **injected** by this probe; no #51 run was observed to diverge. What is shown is that the
  certificate cannot detect such a divergence.
- `CanonicalNode` holds `event_id, val, cur, eligibility, reason, cause, pos_cyc, witness`. It has no
  `claim_value`, times or tier, while `_resolve_ek5_question` (`chelated_beam_128k_replay.py:792-822`) reads
  `claim_value`, `tx_time` and `authority_tier`.

**`PEK5-ASOF` (K1, native API only).**

- H1: `C COPY_OF R` at tx 2, `R` refuted by tier 5 at tx 5, then the `COPY_OF` attestation is withdrawn natively.
- H2: the same, but the edge never existed.
- `as_of(∞, 7)` returns `C = ADMIT` for both.
- The bitemporal truth for H1 at tx 7 is `AUDIT_ONLY`. That is also H1's own state before the withdrawal.
- The withdrawn record has no withdrawal time. Its fields are `active, attestation_id, dst_id, edge_type,
  lineage_id, src_id, tx_time, valid_time`, and `as_of` filters on `att.active` (`ek5_v3_stratified.py:1298`).
- Nothing flags this.
- Under a pure "erase as if never present" reading, H1's answer equals the counterfactual and is correct. The defect
  is that the store cannot represent "withdrawn at tx 10" and "never existed" as different histories. #51 calls
  `as_of` (line 798) but never calls `withdraw_attestation_ids`, so it does not reach this path.

## Part C: constructed affine state `x_{t+1} = A_t x_t + b_t` (secondary; cannot rescue or kill the item)

- Grid: 4 regimes × d ∈ {8, 32} × T ∈ {256, 2048} × seeds {0, 1, 2} × 64 deletion positions = 192 deletions per
  config.
- Error is relative 2-norm vs a float64 full replay of the counterfactual. Latency is in µs (median).
- Two deletion semantics: "input" sets `b_k = 0`, "skip" sets `(A_k, b_k) = (I, 0)`.

| Config | SUB-TV in | SUB-LTI in | INV-cert skip (answered/192, max err) | INV-nocert skip max err | ST skip | CK-geo skip | CK-matched skip | RP skip |
|---|---|---|---|---|---|---|---|---|
| normal d8 T256 | 4.9 | - | 180.6 (192, 1.2e-15) | 1.5e-15 | 90.1 | 812.5 | 486.1 | 896.0 |
| normal d8 T2048 | 5.0 | - | 174.3 (192, 4.4e-15) | 4.4e-15 | 113.0 | 4,793.5 | 3,169.2 | 6,612.1 |
| normal d32 T256 | 5.8 | - | 239.6 (192, 2.2e-15) | 2.2e-15 | 138.2 | 834.5 | 529.1 | 1,013.2 |
| normal d32 T2048 | 6.1 | - | 241.3 (192, 3.4e-15) | 3.4e-15 | 179.7 | 5,238.3 | 3,798.5 | 8,248.9 |
| nonnormal-TV d8 T256 | 4.7 | - | 148.9 (107, 2.7e-12) | 2.8e-11 | 85.8 | 632.4 | 407.2 | 820.1 |
| nonnormal-TV d8 T2048 | 5.1 | - | 130.2 (13, 3.2e-12) | 2.7e-9 | 112.9 | 4,634.4 | 3,330.7 | 6,437.9 |
| nonnormal-TV d32 T256 | 6.0 | - | 195.2 (29, 4.4e-13) | 2.5e-6 | 140.0 | 841.7 | 530.6 | 1,011.5 |
| nonnormal-TV d32 T2048 | 6.2 | - | 187.3 (0, -) | 3.1e-6 | 175.5 | 5,169.3 | 4,075.2 | 8,135.9 |
| nonnormal-LTI d8 T256 | 4.8 | 15.1 | 134.3 (29, 1.3e-15) | 8.1e-14 | 84.8 | 607.2 | 447.3 | 812.0 |
| nonnormal-LTI d8 T2048 | 4.9 | 19.5 | 133.0 (5, 3.9e-16) | 6.7e-14 | 113.7 | 5,289.6 | 3,378.9 | 6,550.1 |
| nonnormal-LTI d32 T256 | 6.0 | 19.8 | 195.4 (10, 7.6e-16) | 4.3e-7 | 137.8 | 830.8 | 497.3 | 1,009.2 |
| nonnormal-LTI d32 T2048 | 6.0 | 25.8 | 193.6 (0, -) | 1.9e-8 | 177.8 | 5,773.3 | 4,224.7 | 8,407.4 |
| singular d8 T256 | 5.0 | - | 135.5 (10, 9.0e-16) | **1.8e-1** | 90.0 | 709.9 | 444.2 | 847.7 |
| singular d8 T2048 | 5.1 | - | 130.4 (4, 8.6e-16) | **1.6e-1** | 109.8 | 4,844.2 | 3,153.9 | 6,407.0 |
| singular d32 T256 | 5.7 | - | 180.5 (13, 1.5e-15) | **1.5e-1** | 131.7 | 663.0 | 465.5 | 950.5 |
| singular d32 T2048 | 5.8 | - | 182.8 (0, -) | **7.9e-2** | 165.9 | 4,191.8 | 3,256.6 | 7,296.7 |

**Errors.**

- SUB-TV, SUB-LTI and ST stay ≤ 4.9e-14 relative error in every config.
- CK and RP are bit-exact, since they are the reference computation started from a stored state.

**Retained bytes, d = 32, T = 2048:**

| Arm | Bytes |
|---|---|
| SUB-LTI | 98,560 |
| SUB-TV | 16,777,472 |
| RP | 17,301,760 |
| CK-geo | 17,304,832 |
| CK-matched | 17,826,048 |
| ST | 34,594,816 |
| INV | 34,603,264 |

**What Part C maps (the "boundary").**

1. **Exact deletion without replay and without storing the history needs both an LTI transition and input
   semantics.**
   - It also needs the retraction request to supply the deleted input `b_k`.
   - Under those conditions, SUB-LTI holds `x_T` plus log₂T powers of A: 58× to 380× fewer bytes than ST.
   - It deletes in 15 to 26 µs, with ≤ 1.9e-14 relative error.
   - Nothing else in the table is a compressed state.
   - SUB-TV is exact and fast (5 to 6 µs), but its retained bytes equal the whole transition log.
2. **Skip-deletion by inversion is uncertifiable on non-normal or singular dynamics.**
   - INV-cert refuses 85 to 192 of the 192 deletions in every non-normal or singular config, and all 192 in the
     T = 2048, d = 32 configs.
   - Without the certificate, INV answers everything, with relative errors up to 2.5e-6 (non-normal) and 0.18
     (singular).
3. **Collision pairs (K1-style, singular-TV d8 T256).**
   - 9 pairs, 3 per seed. Each pair has `x_T` equal to ≤ 7.8e-16 relative and identical retained deletion inputs.
   - Their true skip-counterfactuals differ by 3.0e-5 to 2.6e-4 relative.
   - Uncertified INV returns the same answer for both (outputs differ ≤ 3.8e-13), so at least one is wrong (errors
     1.9e-6 to 1.3e-3). The pair is indistinguishable from the retained state.
   - The certificate (`cond ≥ 1e8`; observed 2.3e16 to 6.1e17) refuses all 9.
   - **Part C K1 therefore fires for INV without the certificate, and not with it.**
4. **Part C K2.**
   - The ordered segment tree (ST) matches the certified INV treatment: same error class, 0.50× to 0.87× its latency,
     1.00× its bytes. K2 fires in the 13 configs where INV-cert answered anything.
   - Against SUB-TV, ST fails on bytes (2.06× to 2.25×) and on latency (13× to 21×).
   - CK-geo fails on latency against every treatment: 98× to 1,065× vs SUB-TV, and 3.5× to 40× vs INV-cert.
   - Against SUB-LTI, every baseline fails on bytes (29× to 380×).

## Crouzeix (#325)

- **Method.** W(A) boundary = `v*Av` at 720 angles, with v the top eigenvector of the Hermitian part of `e^{iθ}A`.
- **Polynomials.** `z^n` for n = 1 to 64, plus 32 random complex polynomials of degree ≤ 8.
- **Matrices (74):**
  - 2 nonnormal-LTI;
  - 8 normal-TV per d (×2);
  - 8 nonnormal-TV per d (×2);
  - 8 singular-TV per d (×2);
  - 24 #60 TinyModel weight matrices (`wq/wk/wv/wo`, 2 layers, seeds 6001 to 6003). These are labelled as weights,
    not transitions.
- **Constant used: 1+√2 ≈ 2.414 (the Crouzeix–Palencia 2017 theorem).** 2 is reported as the conjecture only.

| | value |
|---|---|
| max over all matrices of `‖p(A)‖ / max_{∂W}|p|` | **1.557** (TinyModel 6001 `wv[1]`) |
| holds with 1+√2 | all 74 |
| holds with 2 (conjecture) | all 74 |
| max ratio in transitions only | 1.477 (nonnormal-TV d32, t = 218) |
| max `‖A^n‖ / ρ(A)^n` (n ≤ 64) | **5.2e10** (nonnormal-TV d32) |
| boundary refinement triggered | never (no ratio within 0.1% of 2) |

Transient growth:

- **The non-normal transitions show `‖A^n‖` up to 6.1e7 while ρ = 0.9.** The spectral radius gives no usable bound
  on retraction error amplification.
  - The numerical range does not give a usable bound here either. This sentence was corrected after Tier B round 1.
  - The radius is w(A) ≈ 1.99 to 2.27, so ‖Aⁿ‖ ≤ 2.414·wⁿ grows like 2ⁿ and is vacuous for large n.
- **Normal matrices sit at a ratio of exactly 1.000, as theory requires.**
- **This is suggestive, not a proof, that INV fails on non-normal regimes.** `Φ(T,k)` is a product of different
  transitions, each with W-radius above 1. Crouzeix's theorem applies to a single matrix, so it does not carry over to
  these time-varying products.

## Limits

1. **No bundle holds a compressed neural state.** The titled comparison does not exist. Part C is a constructed
   stand-in; it is not #51, #60 or #62 code.
2. **#60 latencies are disk-write dominated and noisy.**
   - p90/median is up to 2.7 (B-FLAT, N = 16), with one measurement per deletion as preregistered.
   - This run's N = 16 K2 firing was 1.067× against the 1.10 threshold. Tier B round 1's three re-runs gave
     0.39–0.59× at N = 16 and 0.71–0.78× at N = 32, so the firing is robust. The reported run was slow, probably
     because the machine was loaded.
   - The verdict rests on this K2 leg. The K1 legs are a coverage gap and a semantics note (see Verdict).
   - The Part B table is labelled "#51/#62", but `withdraw_attestation_ids` has 0 hits in either bundle. The
     treatment is the shared store's native API, not the bundles' own path.
3. **#60 runs a tiny 2-layer, d = 16, float64 numpy model with absolute positions (a floor-tier mechanism, as the
   bundle itself discloses).** The 100% dirty-cone result depends on absolute positions. A RoPE model would also have
   every suffix row dirty through attention in layer ≥ 1, but layer-0 K/V could differ.
4. **#62 is represented only through the shared `StratifiedEvidenceStore` class, with #51's configuration.**
   - #62 constructs it with `strict_dag_supersede=True, exact_min_rank=True` (`chelated_pooled_agent_runtime.py:804`).
     That configuration was not run.
   - #62's worker-reopen layer is out of scope.
5. **EK5 bytes are `pickle` sizes, so they include Python object overhead.** Baselines and treatment are measured the
   same way.
6. **`P51-DIGEST` shows a certificate-coverage gap using an injected divergence.** It is not an observed failure in
   a #51 run.
7. **`PEK5-ASOF` depends on reading withdrawal as a bitemporal event.** Under the counterfactual-erasure reading it is
   not a defect (see Part B).
8. **Crouzeix checks are empirical.** The boundary is sampled, which can only underestimate `max_W |p|`. No ratio came
   near a bound, so no refinement was needed. The weights are not transition operators of #60 (it has none).

## Departures from the preregistration

- **D1. A3 invariant (b) is ambiguous for order (ii).**
  - The preregistration defines #60 exactness as (a) logits AND (b) "no cached block whose causal prefix contains a
    retracted version is served".
  - In order (ii), the served block's *recorded* causal prefix (`sources = [P0, v2]`) contains no retracted version,
    but the *request's* prefix (`P0, v1`) does.
  - I counted only (a) for K1 and report the request-prefix reading separately (it fires).
  - The verdict is the same either way.
- **D2. The P51 tamper target.**
  - The first smoke run tampered `copy_summary_2`, which is before the step-3 checkpoint and therefore not
    replayed. That was a bug: the probe changed nothing.
  - It was moved to `aux_v2`, a replayed turn, before the full run. The preregistration specifies "a replayed turn's
    `claim_value`", which `aux_v2` satisfies.
- **D3. The Crouzeix LTI matrix is a different draw.**
  - The nonnormal-LTI matrix used for Crouzeix is drawn with T = 8, seed 0. The generator is keyed on T, so this is a
    different draw from the same distribution as the Part C LTI matrices (T = 256/2048).
  - TV transitions are sampled at seed 0, T = 256.
  - For singular-TV, the 8 sampled A_t are the first 4 singular steps plus 4 regular steps. The preregistration says
    only "8 sampled A_t".
- **D4. Part A K2 is evaluated per N, pooled over the 3 model seeds.** Per-seed medians are in the JSON
  (`part_A.per_seed`).
- **D5. Part C INV latency medians include refused calls**, which cost about the same as answered ones (cond
  computation). Configs where INV-cert answered 0 deletions have no K2 entry for INV: there is no working treatment
  to compare.
- **D6. The optional W-BEAM R = 64 was not run.**
  - The measured total was 631 s; R = 64 would likely have pushed it near the 20-minute cap.
  - At R = 16 the GEO/treatment latency ratio is already 418×, so R = 64 could not change the EK5 K2 outcome.
- **D7. W-BEAM replicas also prefix `lineage_id`** (as well as event, parent and entity ids), so that replicas do not
  share version lineages.
- **D8. Part C collision pairs are limited.** They use singular-TV, d = 8, T = 256, and the first 3 singular steps
  per seed (9 pairs). The preregistration did not fix a count.
- **D9. One full run failed and was re-run once.**
  - The first full run died in Part A on a Windows MAX_PATH error (a long scratch subfolder name), before any K2
    numbers were produced.
  - Scratch names were shortened and the run was repeated once; the reported numbers come from that second run.
  - Between the runs the code changed only by lint renames (`I` → `eye`) and the scratch naming.

## Brutal-honesty disclosures (L1 to L13, rulebook §1)

- **L1 (scaffold-as-feature): none.** No `pass` or `NotImplementedError` bodies. One field is a constant rather
  than computed: `asof_collision_probe` returns `flag_raised: False`. The reason is that `as_of` returns only a store
  and has no warning channel.
- **L2 (conditional escape hatch): two guards, both preregistered or disclosed.**
  - `crouzeix_check` refines only above 0.999 × 2, and that never triggered.
  - Part C K2 skips INV comparisons when INV answered nothing (D5).
- **L3 (mock ate the real code): none.** The bundle code is called unmodified: `serve_chelated`, `SSDKVCache`,
  `apply_mutation`, `reference_fresh_rebuild`, `_resolve_ek5_question`, `_snapshot_checkpoint`.
- **L4 (partial with claim of complete): applies, and is disclosed.**
  - The titled comparison (compressed neural-state retraction) was not run, because it does not exist.
  - #62's own store configuration was not run (Limit 4).
  - The optional R = 64 was not run (D6).
- **L5 (test as truth): avoided.** The evidence is the full run's JSON, not the unit tests.
- **L6 (aggregated-claim drift): checked.** The kill table lists every component, including the ones that do not
  fire and the one not counted (D1).
- **L7 (re-summarization decay): one risk, and it was checked.** This work spans a context compaction. The probe
  designs were re-checked against the code after it: the D2 bug was found that way, and the `withdraw_attestation_ids`
  usage was confirmed by grep (0 hits in the three bundle files).
- **L8 (test asserting the bug): none known.** The Jordan-block test pins the known extremal value of 2 from theory,
  not from the code's output.
- **L9 (doc as implementation): none.** Every file the report names exists in the commit. Every number was copied
  by hand from `w22_09_results.json`; there is no automated table generation, so a transcription slip is possible.
- **L10 (dependency phantom): none.** All imports resolve, and the full run executed every path.
- **L11 (broad-catch swallowing): none** in the new code.
- **L12 (status-permissive test): one weak test.** `test_asof_probe_shape` only checks keys and one boolean. The
  runner's Part A and probe orchestration has no unit test; it is exercised only by the `--quick` smoke (run before
  the D2 fix) and the full run.
- **L13 (soft prose claimed as mechanical): two statements are reasoning, not runs.** "The verdict is the same either
  way" (D1) and "R = 64 could not change K2" (D6).

## Scratch receipt (bulk data; gitignored; removable with owner approval)

| Full path | Size |
|---|---|
| `C:\Users\mattm\AppData\Local\Temp\claude\439e695c92b40043\c9f4318c-455f-427a-844a-07562a0a3c42\scratchpad\w22-09\artifacts\pv7_w23b_w22_09\scratch\r1126033316` | 37,630,149 bytes (the reported run) |
| `C:\Users\mattm\AppData\Local\Temp\claude\439e695c92b40043\c9f4318c-455f-427a-844a-07562a0a3c42\scratchpad\w22-09\artifacts\pv7_w23b_w22_09\scratch\run_20261126T033012` | 37,553,787 bytes (failed first run, D9) |

Nothing was deleted.

## Self-score

- `BHS_SELF_DRAFT: 90`. Why it is not 100:
  - the N = 16 K2 firing is inside timing noise;
  - #62's own store configuration was not exercised;
  - the runner's Part A orchestration has no unit test;
  - the Crouzeix LTI matrix is not the Part C draw (D3).
- **`BHS_TIER_B: 85` (important).** The reviewer was fresh agent `a44533c80b1ac471f`. Re-check output is in
  `C:\Users\mattm\AppData\Local\Temp\claude\439e695c92b40043\c9f4318c-455f-427a-844a-07562a0a3c42\scratchpad\tierb_w22_09\out`.
  It found:
  - K1 `P51-DIGEST` was framed as a bundle-path break when it is a metric-coverage gap;
  - `PEK5-ASOF` contradicts the preregistration's own definition of deletion;
  - the N = 16 noise hedge pointed the wrong way;
  - the Crouzeix prose overclaimed;
  - the Part B "#51/#62" label was misleading.

  All of these are corrected above. It confirmed:
  - the preregistration came first;
  - the kill rule was applied as preregistered;
  - the report numbers match the JSON.
- `BHS_OFFICIAL: 85`.

## Reproduce

```
python scripts/run_w22_09_retraction_boundary.py --out-dir artifacts/pv7_w23b_w22_09
python -m unittest tests.test_retraction_boundary
```
