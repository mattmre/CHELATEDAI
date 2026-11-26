# W22-10 — #60 KV-Guard Scope Bound: report

Branch `pv7/w23b-w22-10-kv-guard-bound` (base `2c408925`). Preregistration:
`artifacts/pv7_w23b_w22_10/PREREGISTRATION.md`, committed on its own as `03f92d9f`
before any full-run results existed.

## Verdict

**T-K STAYS KILLED.** The receiver-activation bound is sound: 0 violations across
13,200 rows (3 model seeds × 100 cases × 44 revision instances). But it is **vacuous
on every realistic revision class** (R3 layer-0 adapter, R4 ε ≥ 1e-4, R5 cross-model,
R6 fp16). R_bound = 0 on all 3 seeds for each of them, so it cannot beat selective
top-layer re-prefill (TLR-h). seeds_passing = 0/3 for every realistic class.

The bound's looseness does not decide this. An **oracle** with perfect tightness would
also fail. `stale_within_tau_rate` (actual stale-KV logit error ≤ τ = 1e-4) is 0.0 for
every realistic class on every seed. On those classes reusing the stale KV is wrong by
3 to 6 orders of magnitude over τ, so no sound guard of any tightness may accept it.
TLR-h is also never within τ on those classes (0.0).

**N3 half was not run.** It is gated off because W22-07 killed T-D
(`artifacts/pv7_w23a_seal/W23A_DISPOSITION.md`). No N3 code, data or claims are in
this slice.

## What the "weight-bound KV guard" actually is

#60 (`chelated_ssd_kv_repo_cache.py`) has no bound or tolerance. Its guard is a
**binary identity check**. The TRA-1 contract digest binds the SHA of every weight,
and any weight change raises `StaleWeightRevisionError`. The "bound" here is a
research instrument built beside it, in `kv_guard_scope_bound.py`. **#60's code is unchanged.**
The question answered is whether that binary refusal could be relaxed to a certified
tolerance. The answer: only where it is already exactly safe structurally (below).

## Key numbers (primary model seed 6001; 100 cases, prefix 131–253 tokens)

| class | A0 refuses | R_cone | R_bound | bound median | stale err median / max | TLR-h max err | TLR-r max err | tightness median |
|---|---|---|---|---|---|---|---|---|
| R0 identity | 0.0 | 1.00 | 1.00 | 0 | 0 / 0 | 1.9e-13 | 7.8e-13 | n/a |
| R1 patched_base | 1.0 | 0.02 | 0.02 | 5.18e2 | 2.1e-2 / 3.7e-1 | 3.7e-1 | 3.7e-1 | 2.4e4 |
| R2 last-layer adapter | 1.0 | 1.00 | 1.00 | 0 | 0 / 0 | 1.9e-13 | 7.6e-13 | n/a |
| R3 layer-0 adapter | 1.0 | 0 | 0 | 6.07e2 | 52.8 / 126 | 126 | **1.6e-12** | 11.6 |
| R4 ε=1e-1 | 1.0 | 0 | 0 | 2.60e3 | 68.2 / 140 | 141 | 141 | 37.4 |
| R4 ε=1e-2 | 1.0 | 0 | 0 | 2.22e3 | 13.2 / 92.4 | 92.8 | 79.6 | 169 |
| R4 ε=1e-3 | 1.0 | 0 | 0 | 1.19e3 | 1.38 / 13.3 | 12.9 | 8.76 | 860 |
| R4 ε=1e-4 | 1.0 | 0 | 0 | 5.31e2 | 0.138 / 1.34 | 1.30 | 0.905 | 3.77e3 |
| R4 ε=1e-6 | 1.0 | 0 | 0 | 4.78e2 | 1.4e-3 / 1.3e-2 | 1.3e-2 | 9.1e-3 | 3.42e5 |
| R4 ε=1e-8 | 1.0 | 0 | 0 | 5.32 | 1.4e-5 / 1.3e-4 | 1.3e-4 | 9.1e-5 | 3.71e5 |
| R5 cross-model | 1.0 | 0 | 0 | 7.71e3 | 100 / 171 | 185 | 177 | 76.5 |
| R6 fp16 | 1.0 | 0 | 0 | 6.13e2 | 0.345 / 3.47 | 2.95 | 1.98 | 1.81e3 |

The oracle ceiling (`stale_within_tau_rate`) is 0.0 for R3, R4 ε≥1e-6, R5 and R6 on all 3 seeds.
For R4 ε=1e-8 it is 0.996 / 0.998 / 0.998 (seeds 6001/6002/6003), but the bound is 5–9 there and
certifies nothing. For R1 it is 0.02 / 0.07 / 0.20.

Seeds 6002 and 6003 give the same qualitative picture. Every number is in `w22_10_results.json`.

**Mean cost (FLOPs per case, model formulas from the prereg):**

| arm | FLOPs |
|---|---|
| full re-prefill | 1,617,547 |
| TLR-h | 183,234 |
| TLR-r | 1,434,312 |
| A2 guard, rejecting (realistic classes; amortized / unamortized) | 1,688,412 / 2,830,413 |
| A2 guard, accepting (R0/R2) | 70,865 / 1,212,865 |

When A2 rejects, it costs more than the full re-prefill (guard plus fallback). The
cost of beating TLR-h is moot because R_bound = 0.

**Integrity checks:**
- Stale-KV bit mismatches between the #60 on-disk `.kvblk` and the instrumented prefill: 0 on all seeds.
- Argmax-certificate violations: 0.
- `K_SOUND` holds, with 0 total soundness violations.

**Wall time (seed 6001, R4 ε=1e-8, medians):**

| step | seconds |
|---|---|
| full prefill | 2.68e-3 |
| bound, per-case part | 2.31e-4 |
| TLR-h KV | 2.3e-5 |

These are dominated by Python overhead and are not part of the decision. The full run took 113 s.

## Scope map for #60's guard (the useful output)

- **R0 identity** and **R2 last-layer output-projection change**: the stale KV is
  *exactly* valid (error 0), because layer-K/V does not depend on `Wo_1`. The
  dependency-cone rule (A1) and the bound (A2) both accept 100%. The shipped guard
  refuses R2 (refusal 1.0), so it is conservative there.
- **R1 patched_base**: safe only when the patched token is absent from the prefix
  (2% of cases). A1 catches this exactly, and the bound adds nothing beyond A1.
- **R3 layer-0 adapter on `Wo_0`**: TLR-r (keep stale layer-0 K/V, recompute layer 1 from
  the receiver) is **exact** (max error ≤ 1.6e-12 on all seeds). TLR-h is wrong (126). This shows
  the structural cone is finer than "layers ≥ first changed layer". It is an **unpreregistered
  observation** and does not enter the decision.
- **Everything else** (global perturbations ε ≥ 1e-6, cross-model, fp16): stale reuse is
  numerically wrong beyond τ. Refusing it is correct, so the shipped guard is right to refuse.

The practical takeaway is a structural (cone) relaxation of the binary guard, not a
numeric bound. That relaxation is not implemented in #60, and this slice does not propose shipping it.

## Departures from the preregistration

1. I changed `_soft` (soft-softmax lemma helper) after the full run had started. I added
   `if 2δ ≥ ln 3: return 2.0` before `expm1` to silence an overflow RuntimeWarning.
   The output is the same, because `min(e^{2δ}-1, 2)` = 2 in that region. The run used the
   old code; its stdout (`scratch/full_run_stdout.txt`) shows the warning. I did not re-run.
2. Each R4 ε is treated as its own class in the decision table. The prereg listed R4
   ε ≥ 1e-4 as realistic, without stating whether to pool or split. Every split fails, so pooling cannot change the verdict.
3. I ran a `--quick` smoke run (2 repos, seed 6001; outputs in `scratch/quick`) after
   the prereg commit and before the full run.
4. A0 (the shipped guard) is checked with `cache.get` on the **first** prefix block only.
   Every block is bound to the same contract digest, so this is representative but not exhaustive.
5. TLR-r is computed single-shot over the whole prefix, not block by block. This is
   equivalent for a causal model, but it is not how #60 prefills.
6. The per-case guard cost includes the Kmax/R and observed-run norm passes. The prereg
   lumped them into "per-case" without itemizing them.
7. The `nonvacuous` flag is false for R0 and R2 by definition, because R_bound is not > R_cone.
   They are not realistic classes and do not enter the verdict.
8. These metrics were not in the prereg: the oracle `stale_within_tau_rate`, `tlrh_within_tau_rate`,
   TLR-r error columns, and the argmax-mismatch counts. They are reported, and do not drive the decision.
9. The R3 TLR-r exactness finding is unpreregistered (see the scope map).
10. Ruff was invoked as `python -m ruff check`, because bare `ruff` is not on PATH.

## Brutal Honesty

- **L1 stubs:** none. `grep` for `TODO`/`pass`-bodies in the three new files finds no placeholder logic.
- **L2 escape conditionals:** `_soft` returns 2.0 for `2δ ≥ ln 3`. This is the lemma's
  `min(·,2)` cap, not an escape. `bound_is_sound` allows a slack of `1e-12 + 1e-9·B` for float
  round-off (`kv_guard_scope_bound.py`, `bound_is_sound`). The slack is disclosed. It is ≥3 orders
  smaller than every realistic B, so it cannot flip a verdict.
- **L3 mocks in production paths:** none. The sender uses the real #60 `SSDKVCache` /
  `serve_chelated` and real on-disk `.kvblk` files. `read_block_unchecked` deliberately
  bypasses the contract check (it is a research reader). It verifies prefix tokens and body
  length and is not wired into #60.
- **L4 partial-as-complete:**
  - This is a **floor-tier** result on #60's 2-layer, d=16, float64 TinyModel. It is not an LLM.
    Nothing here licenses claims about real-model KV transport.
  - The FLOP model is approximate: SVD is estimated at 20·d³ per spectral norm, and softmax/exp is ignored.
  - The bound is very loose (median tightness 11.6 to 5.8e5). The verdict does not hinge on
    that, because the oracle ceiling is also 0 (see Verdict).
- **L5/L8/L12 untested production paths:** the runner script itself has no unit test. The library
  has 11 `unittest` tests, all passing. The full run is the runner's runtime evidence.
- **L11 broad-catch swallowing:** none. `read_block_unchecked` raises `ValueError` and does not swallow it.
- **L13 soft prose claimed as mechanical:** the "scope map" is an empirical summary of this
  run. It is not a proven theorem, except the R2 independence of `Wo_1`, which holds structurally.
- **Not done:** I ran no Tier B adversarial review, did not push, and opened no PR.

```
EVIDENCE: artifacts/pv7_w23b_w22_10/w22_10_results.json (full run, 13,200 rows, 113 s, exit 0)
SMOKE: floor tier — python scripts/run_w22_10_kv_guard_bound.py --quick (2 repos); ceiling = full run above, on #60's TinyModel only
TESTS: python -m unittest tests.test_kv_guard_scope_bound -> 11 tests OK
BHS_SELF_DRAFT: 92
BHS_SELF_DRAFT_AGENT: claude-opus-5-5 (implementer subagent, W22-10)
BHS_TIER_B: PENDING
BHS_TIER_B_AGENT: PENDING (must be a fresh agent, not the implementer)
BHS_TIER_B_SEVERITY: PENDING
BHS_OFFICIAL: PENDING (= min(self, tier B); cannot merge until 100)
CARRY_FORWARD: Tier B review; decide whether the cone/TLR-r structural relaxation is worth a separate #60 slice
DEFERRED_SCOPE: N3 half (gated off by W22-07 T-D kill)
LOOP_ITERATIONS: 1
OPERATOR_OVERRIDE: none
```

The self-score is held below 100 for three reasons: departure 1 (code changed during the run,
verified only by argument), the approximate FLOP model, and the missing Tier B.

## Scratch receipt (removable, owner approval required)

- Path: `C:\Users\mattm\AppData\Local\Temp\claude\439e695c92b40043\c9f4318c-455f-427a-844a-07562a0a3c42\scratchpad\w22-10\artifacts\pv7_w23b_w22_10\scratch`
- Size: 32,387,947 bytes across 1,349 files:

| path | bytes |
|---|---|
| `full\` | 31,514,952 |
| `quick\` | 871,656 |
| `full_run_stdout.txt` | 1,339 |

- Gitignored by `.gitignore` (`artifacts/**/scratch/`). Nothing was deleted.
