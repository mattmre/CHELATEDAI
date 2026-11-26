# W22-10 — #60 KV-Guard Scope Bound: report

Branch `pv7/w23b-w22-10-kv-guard-bound` (base `2c408925`). Preregistration:
`artifacts/pv7_w23b_w22_10/PREREGISTRATION.md`, committed on its own as `03f92d9f`
before any full-run results existed. First report: `1a3334bd`. This revision folds in
Tier B round 1 (see the section near the end).

## Verdict

**T-K STAYS KILLED under both acceptance criteria below.** The receiver-activation bound
is sound: 0 violations in 13,200 rows (3 model seeds × 100 cases × 44 revision instances).
It is **vacuous on every realistic revision class** (R3 layer-0 adapter, R4 ε ≥ 1e-4,
R5 cross-model, R6 fp16). R_bound = 0 and seeds_passing = 0/3 for each of them, so it cannot
beat selective top-layer re-prefill (TLR-h).

The *reason* it fails depends on which acceptance criterion is used. The first report
overstated this (see Brutal Honesty, L13/L4).

1. **Logit criterion, τ = 1e-4 (the preregistered rule, `LOGIT_TOL` of #60).**
   - An oracle that knew the true error would also accept 0% on every realistic class, on
     every seed (`stale_within_tau_rate` = 0.0). Reusing the stale KV really exceeds τ there.
   - The smallest stale error seen is 0.0198 (R4 ε=1e-4) and 0.0458 (R6). That is about
     2.3 to 2.7 orders of magnitude over τ, and the largest errors are up to about 6.3 orders
     (182, R5).
   - At this τ, a bound's looseness is *not* what kills it. No sound guard could accept these rows.
2. **Argmax criterion: every suffix next-token argmax is preserved (receiver-level).**
   - Here the oracle ceiling is **high** on several realistic classes. Stale-KV reuse keeps
     every suffix argmax on most rows (rows kept, seeds 6001/6002/6003):

     | class | rows kept (out of) |
     |---|---|
     | R4 ε=1e-4 | 490 / 489 / 493 (of 500) |
     | R6 fp16 | 93 / 95 / 95 (of 100) |
     | R4 ε=1e-3 | 429 / 383 / 394 (of 500) |
     | R4 ε=1e-2 | 92 / 73 / 71 (of 500) |

   - The bound certifies **0** argmax positions on all of these classes, because B is
     about 4e2 to 8e3 against top-2 margins of order 1.
   - At the argmax level, **the bound's looseness is what kills it**. A much tighter
     receiver-level certificate could in principle accept most R4 ε≤1e-3 and R6 rows.
   - T-K still stays killed. The kill rule asks whether *this* bound beats TLR-h, and it
     accepts nothing. No tighter bound was built or tested here, and I do not claim one exists.

**N3 half was not run.** It is gated off because W22-07 killed T-D
(`artifacts/pv7_w23a_seal/W23A_DISPOSITION.md`). No N3 code, data or claims are in this slice.

## What the "weight-bound KV guard" actually is

#60 (`chelated_ssd_kv_repo_cache.py`) has no bound or tolerance. Its guard is a
**binary identity check**: the TRA-1 contract digest binds the SHA of every weight, and
any weight change raises `StaleWeightRevisionError`. The "bound" is a research instrument
built beside it in `kv_guard_scope_bound.py`. **#60's code is unchanged.**

## Key numbers (primary model seed 6001; 100 cases, prefix 131–253 tokens)

Two columns are new in this revision. The **argmax oracle** is the number of rows where
stale reuse keeps every suffix argmax. **Bound cert** is the number of argmax positions the
bound certified.

| class | A0 refuses | R_cone | R_bound | bound median | stale err median / max | argmax oracle rows | bound cert pos | TLR-h max | TLR-r max | tightness median |
|---|---|---|---|---|---|---|---|---|---|---|
| R0 identity | 0.0 | 1.00 | 1.00 | 0 | 0 / 0 | 100/100 | 2936/2936 | 1.9e-13 | 7.8e-13 | n/a |
| R1 patched_base | 1.0 | 0.02 | 0.02 | 5.18e2 | 2.1e-2 / 3.7e-1 | 99/100 | 68/2936 | 3.7e-1 | 3.7e-1 | 2.4e4 |
| R2 last-layer adapter | 1.0 | 1.00 | 1.00 | 0 | 0 / 0 | 100/100 | 2936/2936 | 1.9e-13 | 7.6e-13 | n/a |
| R3 layer-0 adapter | 1.0 | 0 | 0 | 6.07e2 | 52.8 / 126 | 3/500 | 0 | 126 | **1.6e-12** | 11.6 |
| R4 ε=1e-1 | 1.0 | 0 | 0 | 2.60e3 | 68.2 / 140 | 0/500 | 0 | 141 | 141 | 37.4 |
| R4 ε=1e-2 | 1.0 | 0 | 0 | 2.22e3 | 13.2 / 92.4 | 92/500 | 0 | 92.8 | 79.6 | 169 |
| R4 ε=1e-3 | 1.0 | 0 | 0 | 1.19e3 | 1.38 / 13.3 | 429/500 | 0 | 12.9 | 8.76 | 860 |
| R4 ε=1e-4 | 1.0 | 0 | 0 | 5.31e2 | 0.138 / 1.34 | 490/500 | 0 | 1.30 | 0.905 | 3.77e3 |
| R4 ε=1e-6 | 1.0 | 0 | 0 | 4.78e2 | 1.4e-3 / 1.3e-2 | 500/500 | 0 | 1.3e-2 | 9.1e-3 | 3.42e5 |
| R4 ε=1e-8 | 1.0 | 0 | 0 | 5.32 | 1.4e-5 / 1.3e-4 | 500/500 | 1662/14680 | 1.3e-4 | 9.1e-5 | 3.71e5 |
| R5 cross-model | 1.0 | 0 | 0 | 7.71e3 | 100 / 171 | 0/500 | 0 | 185 | 177 | 76.5 |
| R6 fp16 | 1.0 | 0 | 0 | 6.13e2 | 0.345 / 3.47 | 93/100 | 0 | 2.95 | 1.98 | 1.81e3 |

**τ-level oracle ceiling (`stale_within_tau_rate`):**
- 0.0 for R3, R4 ε≥1e-6, R5 and R6, on all 3 seeds.
- R4 ε=1e-8: 0.996 / 0.998 / 0.998. The bound there is 3.11 to 14.04 (per-seed min/max
  over all three seeds), so it certifies nothing at τ.
- R1: 0.02 / 0.07 / 0.20.

**Median tightness (B / actual)** ranges from 11.6 (R3, seed 6001) to 1.04e6 (R1, seed 6003).

Seeds 6002 and 6003 give the same qualitative picture. The original numbers are in
`w22_10_results.json`. The argmax-oracle and cap-hit columns are in `w22_10_results_tierb_r1.json`.
That file came from a re-run with the diagnostic code. Every deterministic row field was
bit-identical to the original run (0 of 13,200 rows differ; only the wall-time fields differ).

**Cost per case (seed 6001):**

| arm | mean FLOPs/case | extra storage vs stored K,V |
|---|---|---|
| full re-prefill | 1,617,547 | 0 |
| TLR-h | 183,234 | **+25%**: stores the sender's layer-1 input h₁ (P×d) next to 2 layers × (K,V) = 4·P×d |
| TLR-r | 1,434,312 | 0 |
| A2 guard, rejecting (realistic; amortized / unamortized) | 1,688,412 / 2,830,413 | 0 |
| A2 guard, accepting (R0/R2) | 70,865 / 1,212,865 | 0 |

When A2 rejects, it costs more than the full re-prefill. Whether it beats TLR-h on cost
is moot, because R_bound = 0.

**Integrity checks:**
- Stale-KV bit mismatches between the #60 on-disk `.kvblk` and the instrumented prefill: 0 on all seeds.
- Argmax-certificate violations: 0.
- `K_SOUND` holds.

**Wall time (seed 6001, R4 ε=1e-8, medians):**

| step | seconds |
|---|---|
| full prefill | 2.68e-3 |
| bound, per-case part | 2.31e-4 |
| TLR-h KV | 2.3e-5 |

These are dominated by Python overhead and are not part of the decision.

## How much of the bound derivation was exercised (Tier B F2)

`_soft(δ) = min(e^{2δ}−1, 2)` is applied in three steps: prefix layer 0 (`cap0`), suffix
layer 0 (`cap_s0`) and suffix layer 1 (`cap_s1`). At its cap of 2 the step is trivial: it
only says the attention distribution moved by at most its full L1 diameter. Cap-hit rates
are identical on all three seeds.

| classes | cap0 | cap_s0 | cap_s1 |
|---|---|---|---|
| R4 ε=1e-2, R4 ε=1e-1, R5 | 1.0 | 1.0 | 1.0 |
| R3, R4 ε=1e-3/1e-4/1e-6, R6 | 0 | 0 | 1.0 |
| R1 | 0 | 0 | 0.98 |
| R0, R2, R4 ε=1e-8 | 0 | 0 | 0 |

So the **non-trivial softmax branch is fully exercised only on R0, R2 and R4 ε=1e-8**, and
B = 0 on R0 and R2. The 0-violation soundness result is therefore **weak evidence** that the
derivation is right. On every realistic row at least one step is capped, and B sits 11× to
1e6× above the actual error. A constant-factor error (say a missing √d or factor 2)
in a non-capped term would very likely still pass. The 200-trial numeric test of the softmax
lemma (`tests/test_kv_guard_scope_bound.py`) and the hand derivation in the prereg are the
real support. Tier B re-derived the bound by hand and could not break it.

## Scope map for #60's guard (the useful output)

- **R0 identity** and **R2 last-layer output-projection change**: the stale KV is *exactly*
  valid (error 0), because layer K/V does not depend on `Wo_1`. The cone rule (A1) and the
  bound (A2) both accept 100%. The shipped guard refuses R2.
- **R1 patched_base**: the stale KV is safe only when the patched token is absent from the
  prefix (2% of cases). A1 catches this exactly. **TLR-h is identical to naive stale reuse
  on R1** (300/300 rows with the same error). `Wk1`/`Wv1` are unchanged and h₁ comes from
  the sender, so TLR-h recomputes the same layer-1 K/V.
- **R3 layer-0 adapter on `Wo_0`**:
  - **TLR-h is likewise identical to naive stale reuse** (1500/1500 rows; max error 125.7
    on seed 6001, 128.9 over all seeds).
  - TLR-r (keep stale layer-0 K/V, recompute layer 1 from the receiver's own layer-0 pass)
    is **exact** (≤ 1.6e-12).
  - This is an **unpreregistered observation**: the structural cone is finer than "layers ≥
    first changed layer".
- **Global perturbations ε ≥ 1e-6, cross-model, fp16**: stale reuse exceeds τ on every row.
  At τ, refusing is correct. At the argmax level, many of these rows would have been fine
  (R4 ε≤1e-3, R6), but this bound cannot tell which ones.

## Departures from the preregistration

1. I changed `_soft` after the full run had started (overflow clamp). Tier B confirmed it
   is a no-op: rows were bit-identical. Tier B's re-run is under `scratch/tierb`, and my
   Tier B round-1 re-run is also bit-identical.
2. Each R4 ε is scored as its own class.
3. I ran a `--quick` smoke run (2 repos, seed 6001) after the prereg commit and before the full run.
4. A0 was checked with `cache.get` on the first prefix block only.
5. TLR-r is computed single-shot over the whole prefix.
6. The per-case guard cost includes the Kmax/R and observed-run norm passes.
7. The `nonvacuous` flag is false for R0 and R2 by definition. They are not realistic classes.
8. Some metrics were not preregistered: the τ-oracle `stale_within_tau_rate`,
   `tlrh_within_tau_rate`, the TLR-r columns, argmax-mismatch counts, and (this revision)
   the argmax-oracle ceiling, `_soft` cap-hit rates and the TLR-h storage overhead.
9. The R3 TLR-r exactness finding is unpreregistered.
10. Ruff was run as `python -m ruff check`.
11. **Tier B round 1 re-run.**
    - After the first report, I re-ran the full campaign with diagnostic-only additions:
      `_soft` deltas and cap flags in the bound's intermediates, plus argmax-oracle and
      cap-rate aggregates.
    - It used the new `--out scratch/full_tierb_r1` and `--results w22_10_results_tierb_r1.json`,
      so the original evidence was not overwritten. It ran in 81 s.
    - B and every decision field are unchanged (bit-identical rows apart from timing). The
      verdict is unchanged.
12. **Runner CLI change (F5).** I added `--out` and `--results` (defaults unchanged) and
    `parse_args` / `scratch_dir` / `results_path` helpers. Three tests were added (2 for args,
    1 for the `_soft` cap), for 14 tests in total.

## Brutal Honesty

- **L13 / L4 overclaim in the first report (`1a3334bd`), caught by Tier B.**
  - The verdict there said an oracle "with perfect tightness would also fail" and that "no
    sound guard of any tightness may accept it". That is true **only at τ = 1e-4 on logits**.
  - Stated without the τ qualifier, it read as a general statement about receiver-level
    guards. At the argmax level the oracle ceiling is 77–99% of rows on R4 ε∈{1e-3, 1e-4}
    and R6, and the bound's looseness is the cause of failure.
  - The first report also mis-stated three ranges:
    - "3 to 6 orders over τ" should be about 2.3 to 6.3.
    - The R4 ε=1e-8 bound "5–9" should be 3.11–14.04.
    - Median tightness "up to 5.8e5" should be up to 1.04e6.
  - It also omitted TLR-h's +25% storage overhead and its identity with stale reuse on R1/R3.
  - All of these are corrected above. The verdict (T-K stays killed) did not change, but its
    stated reason was too strong.
- **L1 stubs:** none.
- **L2 escape conditionals:**
  - `_soft` returns 2.0 at the lemma's cap. That cap fires on most realistic rows; see the F2 section.
  - `bound_is_sound` allows `1e-12 + 1e-9·B` of slack. This is disclosed.
- **L3 mocks in production paths:** none. The sender is the real #60 `SSDKVCache` /
  `serve_chelated`. `read_block_unchecked` bypasses the contract on purpose, as a research reader.
- **L4 partial-as-complete:**
  - Floor tier on #60's 2-layer, d=16, float64 TinyModel. It is not an LLM.
  - The FLOP model is approximate.
  - Soundness evidence is weak in the sense given in the F2 section.
- **L5/L8/L12:** the runner's `main` has no end-to-end unit test. Its argument handling is now
  tested, and the two full runs are its runtime evidence.
- **L11:** none.
- **Not done:** I did not push and did not open a PR. No tighter (argmax-level) bound was attempted.

```
EVIDENCE: artifacts/pv7_w23b_w22_10/w22_10_results.json (original full run, 13,200 rows) + artifacts/pv7_w23b_w22_10/w22_10_results_tierb_r1.json (diagnostic re-run, bit-identical deterministic fields)
SMOKE: floor tier — python scripts/run_w22_10_kv_guard_bound.py --quick --out <fresh dir>; ceiling = full runs above, on #60's TinyModel only
TESTS: python -m unittest tests.test_kv_guard_scope_bound -> 14 tests OK; python -m ruff check (3 files) clean
BHS_SELF_DRAFT: 85
BHS_SELF_DRAFT_AGENT: claude-opus-5-5 (implementer subagent, W22-10)
BHS_TIER_B: 88 (round 1, on 1a3334bd)
BHS_TIER_B_AGENT: a51c8a69e592899a4 (fresh adversarial agent)
BHS_TIER_B_SEVERITY: important
BHS_OFFICIAL: 85 (= min(85, 88)); does not merge — needs Tier B round 2 on this revision
CARRY_FORWARD: Tier B round 2; whether the cone/TLR-r structural relaxation is worth a separate #60 slice; whether an argmax-level certificate is worth building (would reopen nothing by itself — kill rule is about this bound)
DEFERRED_SCOPE: N3 half (gated off by W22-07 T-D kill)
LOOP_ITERATIONS: 2
OPERATOR_OVERRIDE: none
```

**Why the self-score is lower than the first draft (92 → 85):** my own review let a material
overclaim and three wrong numeric ranges through, and Tier B caught them. A self-score above
Tier B's would be score-gaming. Three items still stand: the weak soundness evidence (F2),
the approximate FLOP model, and the lack of a round-2 Tier B review.

## Tier B round 1

- **Agent:** a51c8a69e592899a4 (fresh adversarial reviewer), on commit `1a3334bd`.
- **Score:** BHS_TIER_B = 88. Severity: important.

| # | sev | finding | addressed by |
|---|---|---|---|
| F1 | important | "Perfect bound accepts 0%" holds only at τ=1e-4; at the argmax level the stale oracle keeps every argmax on most R4 ε≤1e-3 / R6 rows and the bound certifies 0, so looseness kills it there | Recomputed from the original `scratch/full/rows.jsonl`; my counts match Tier B's on every cell. Added `argmax_oracle_*` and `argmax_bound_certified_position_rate` to the aggregate (in `w22_10_results_tierb_r1.json`) and as report columns. Verdict rewritten to be τ-specific, with both criteria. L13/L4 self-disclosure added. |
| F2 | nit | `_soft` hits its cap on realistic rows; the derivation is exercised non-trivially only on R0/R2/R4 ε=1e-8; 0 violations is weak evidence | Measured cap-hit rates per step (new `soft_cap_rate`, `soft_cap_any_rate`, `soft_cap_all_rate`) and added the F2 section. |
| F3 | nit | Three wrong numeric ranges | Verified against the JSON and rows: min stale errors 0.01984 (R4 1e-4) / 0.04577 (R6); R4 ε=1e-8 B ∈ [3.109, 14.044]; max median tightness 1.0356e6 (R1, seed 6003). Fixed. |
| F4 | nit | TLR-h +25% storage; TLR-h ≡ stale on R1/R3 | Added a storage column to the cost table and the identity to the scope map (300/300 R1 and 1500/1500 R3 rows equal; R3 max 125.733 on seed 6001, 128.866 over all seeds). |
| F5 | nit | Runner hard-codes the scratch path, so `--quick` overwrites earlier evidence | Added `--out` / `--results` (defaults unchanged), helper functions, and 2 argparse tests. Ruff clean, 14/14 tests pass. |

Claims Tier B tried and **failed to disprove**:
- The bound's soundness, by hand re-derivation.
- That the `_soft` edit is a no-op (bit-identical rows on re-run).
- That the preregistration was committed before results (`03f92d9f` precedes `1a3334bd`).
- That the report's numbers match the JSON. The three F3 ranges were the exception; they were prose summaries, not table cells.

## Scratch receipt (removable, owner approval required)

- Path: `C:\Users\mattm\AppData\Local\Temp\claude\439e695c92b40043\c9f4318c-455f-427a-844a-07562a0a3c42\scratchpad\w22-10\artifacts\pv7_w23b_w22_10\scratch`
- Total: 66,530,420 bytes across 2,697 files:

| path | bytes | files | what it is |
|---|---|---|---|
| `full\` | 31,514,952 | 1,312 | original full run |
| `full_tierb_r1\` | 33,304,177 | 1,312 | Tier B round-1 diagnostic re-run |
| `quick\` | 871,656 | 36 | smoke run |
| `tierb\` | 837,300 | 35 | written by the Tier B agent, not by me |
| `full_run_stdout.txt` | 1,339 | 1 | stdout of the original run |
| `full_tierb_r1_stdout.txt` | 996 | 1 | stdout of the re-run |

- Gitignored by `.gitignore` (`artifacts/**/scratch/`). Nothing was deleted.
