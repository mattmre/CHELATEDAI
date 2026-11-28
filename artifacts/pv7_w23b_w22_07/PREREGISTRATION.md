# W22-07 Preregistration: Blinded Hidden-Copy Twins (Graded Leakage) & Priced Provenance-Check Policy

Status: **frozen before any W22-07 result was computed.** This file is committed alone, before the
module, the runner, the tests and any results file. Any later edit shows up in `git log`.

- Queue item: `W22-07` in `docs/research/wave22_queue_additions_2026-11-25.json` (gates track `T-D`,
  depends on `W22-05`). Handoff packet row 7 of
  `docs/research/waves-handoff-wave22-breakthrough-audit-and-queue-additions-2026-11-25.md`.
- Compute class: CPU only. No LLM, no embedding model, no GPU. Network is not used: the one generator
  that would fetch (#38's OSV seeds) is given a local fallback cache written under `scratch/`.
- Spine commit: `e5e0b037`. Branch `pv7/w23b-w22-07-copy-twins`.

## What I already knew before freezing (not blind)

- W22-05 (PR #370) found EK5 and a plain SQL ledger tie on clean graphs. Its #55 fixture had no
  COPY/DERIVES edges after `echo_of` was stripped.
- The legacy copy fixtures are visibly templated. I looked at their generators and at a few records
  before writing this file:
  - **#55** (`artifacts/pv7_w22e_b55_wild_live/frozen_snapshots/`): mirror and wire records carry the
    prefixes "Syndicated mirror A:", "Press wire recap:" and "Analyst digest:".
  - **#20** (`scripts/run_overnight_r5_spark6724.build_zero_shortcut_v3_flooded_cases`): copies read
    "Commissioning archive mirror N", "Derived rollup DN", "Synced cache CN copies ...".
  - **#38** (`prospective_realworld_drift_corpus._ECHO_VARIATION_TEMPLATES`): 10 fixed echo templates.
  - I expect the legacy leakage audit to be close to 100%. That number is therefore descriptive and is
    not a prediction.
- **#35** (`phase2_d3_multiagent_poisoning`) and **#46** (`wave14_d11_pm6_takedown_carryforward`)
  have only manifests and aggregates on disk. They have no per-record copy fixture and no generator I
  can run on CPU without models. I can't audit them, and they are named as out of scope (Departure D1,
  declared in advance).
- **My prior expectation, recorded as a prediction: `KILL_T_D`.** On exact twins the T-D policy and
  the conventional Bayesian copy model share one posterior. T-D's only additions are pricing (skipping
  checks that are not worth their price) and check targeting, and on exact twins targeting has no
  signal to use.

## Hypotheses

- **H1 (fixture validity).** On exact blinded twins (leakage grade λ = 0), a text-only bag-of-words
  classifier cannot separate withdrawn-root-copy families from independent-corroboration families.
  - The family-level and witness-level balanced accuracy must both be ≤ 0.55.
  - **Positive control.** At λ = 1.0 the same classifier must exceed 0.55. If it doesn't, the
    leakage test is reported as `UNPOWERED`.
- **H2 (no-check ceiling).** Every no-check policy (no-check, cosine dedup, EK5_v3, the copy-aware
  ledger) makes the same decision for both members of an exact twin pair. So it is right on at most 1
  of 2 members of every pair in which the two members' truths differ.
- **H3 (T-D premium).** The T-D priced-check policy has lower prevalence-weighted harm than the best
  conventional provenance-aware copy model at the same check cap. The kill rule below decides this.

## Fixture construction (blinded twins)

- **Source texts.** Real public records from the #55 frozen snapshots: 4 strata × 60 families = 240.
  - In each family, the first ASSERT/CONFLICT record is the **root**. Its `text`, `entity_key` and
    `claim_value` are real.
  - Nothing else from the legacy family is reused. In particular, no mirror, wire or digest text is
    reused.
- **One twin pair per source family.** Each pair has a member C (withdrawn-root-copy) and a member I
  (independent corroboration).
- **Public records, identical in C and I except blinded ids:**
  - **Root.** tx = valid = 10, authority 5, tenant = the family's blinded tenant.
  - **k witnesses.** k ∈ {2, 3}, seeded. Each witness:
    - ASSERTs the same `claim_value` at tx = valid = 20 + i, authority 5, same tenant and entity;
    - gets a `source` label drawn by seed from a neutral pool (`feed-1` … `feed-6`);
    - gets a text made from the root text by a **surface transform** drawn by seed from
      T0 verbatim, T1 key-field (`"<entity>: <claim_value>"`), T2 leading-60%-of-tokens,
      T3 colon-swap (clauses on either side of the first `:` swapped; verbatim if there is no colon),
      and T4 lowercase with punctuation stripped.
  - **d declared witnesses.** d ∈ {0 .. k−1}, seeded. A declared witness carries
    `explicit_parent_ids = (root,)`, which is a public citation, and it is a copy in both twins.
    Declared witnesses are always the earliest d. That leaves m = k − d ≥ 1 undeclared witnesses.
  - **Withdrawal.** `WITHDRAW` of the root at tx = valid = 30, authority 5.
  - **Query cut** at tx = valid = 40. A secondary cut at 25 is reported but is not part of any decision.
- **Hidden truth.** It is stored only in `echo_of`, which every policy input strips.
  - In C, every undeclared witness has `echo_of = root`. The truth at cut 40 is `RETRACT`.
  - In I, j ~ U{1..m} undeclared witnesses (chosen by seed) are independent, with no `echo_of`. The
    rest are copies. The truth at cut 40 is `SUPPORT` (the claim value).
  - Ground truth comes from `chelated_wild_revision_streams.oracle_at_cut` on the full, unstripped
    records.
- **Blinding.**
  - `record_id` = `sha256(salt ‖ pair ‖ member ‖ slot)[:12]`. The tenant id is blinded the same way.
  - Member order inside each pair is shuffled by seed.
  - No generator role words appear in any public field. A test asserts that the strings "copy",
    "mirror", "echo", "indep" and "syndicat" are absent from public fields at λ = 0. The check is
    case-insensitive, except where the word appears in the real root text itself; that is reported.
- **Graded leakage λ ∈ {0, 0.25, 0.5, 1.0}.** With probability λ per pair (seeded), the pair is
  "leaky":
  - every hidden-copy undeclared witness gets the legacy #55 cue: the text becomes
    `"Syndicated mirror: " + root text`;
  - every independent witness gets T1 or T3;
  - only λ = 0 is "exact twins". λ > 0 levels are secondary.
- **Seeds.** Fixture seeds 20261126, 20261127 and 20261128. For each seed, the dev/test split is 50/50
  by pair (seed + 1).
  - Dev pairs calibrate τ (cosine) and the posterior model.
  - Test pairs are scored.
  - Classifier CV uses seed 1601. The bootstrap uses seed 1601 with 2,000 resamples, clustered by pair.

## Leakage test (fixture gate)

- **Classifier.** `sklearn` `CountVectorizer(binary=True, token_pattern=r"[A-Za-z0-9_]+")` +
  `LogisticRegression(C=1.0, max_iter=2000)`. sklearn is allowed only in this isolated validation flow.
- **Evaluation.** Grouped 5-fold CV, with groups = twin pair, so both members are always in the same
  fold. The metric is balanced accuracy.
- **Family level.** Input is the serialized public view: every public field of every record (texts,
  sources, op kinds, tx/valid times as tokens, declared-parent flags, blinded ids). Label = C vs I.
- **Witness level.** Input is the text + source + position of each undeclared witness. Label = copy vs
  independent.
- **Gate.** λ = 0 is `INVALID` if either balanced accuracy is > 0.55 for any seed.
- **Structural note, declared now.** Exact twins have identical public features, and both members sit
  in the same fold. So the family-level number is 0.5 by construction. What it guards against is
  generator bugs that leak the label through some field. The witness-level test and the λ > 0
  positive control are where the test has any power.
- **Legacy leakage audit (descriptive).** Record-level text-only classifier (copy role vs not), grouped
  CV by family/case, on:
  - **#55:** frozen; the label is `echo_of` non-empty.
  - **#20:** ZS v3 flooded, `n_per_family=30`, rebuilt offline with `HF_HUB_OFFLINE=1`; the label is
    the generator role: event id prefix `hist_m`, `copy_`, `copy_orig_`, `syn_d_` or `syn_u_`.
  - **#38:** arena suite, rebuilt offline with a fallback OSV seed cache in `scratch/`; the label is
    `_echo` in the doc id.
  - #35 and #46 are not audited (D1).

## Policies (all receive the same stripped public input; cap B = 1 check per family for the primary)

| id | policy | checks |
|---|---|---|
| A0 | **no-check corroboration**: SUPPORT if any non-withdrawn assertion survives | 0 |
| A1 | **cosine dedup + earliest**: single-link clusters on term-frequency cosine ≥ τ; a cluster whose earliest member is withdrawn is dead; SUPPORT iff any cluster is live. τ is chosen on dev from {0.05, 0.10, …, 0.95} to minimize the primary harm at π = 0.5 | 0 |
| A2 | **EK5_v3** (existing engine): `project_ek5` + `build_ek5_store` + `ek5_prediction` from `scripts/run_pv7_w18_6c_wild_revisions.py`. claim → SUPPORT, retracted → RETRACT, disputed → ABSTAIN | 0 |
| A3 | **copy-aware ledger**: `chelated_pv7_spine.challenger_ledger` via that script's `sql_envelopes` (declared DERIVES). Both modes, `recursive_cte` and `closure_table`, must agree. RESOLVED → SUPPORT, RETRACTED/ABSENT → RETRACT, DISPUTED → ABSTAIN | 0 |
| A4 | **ledger + earliest-first check** (conventional): check undeclared witnesses earliest-first, up to B. An independent result → SUPPORT (stop). A copy result → append DERIVES(witness → root) to the ledger. Then query the ledger as in A3 | ≤ B |
| A5 | **conventional Bayesian copy model**, no abstain: posterior below; checks earliest-first up to B until an independent witness is found or none are left unchecked; decides argmin{SUPPORT, RETRACT} of expected harm | ≤ B |
| A5a | **A5 + reject option** (Chow): as A5, but ABSTAIN is also allowed | ≤ B |
| A6 | **T-D priced provenance check**: the same posterior as A5. Exact finite-horizon expected-harm dynamic program: at each step either stop (min over SUPPORT/RETRACT/ABSTAIN) or check whichever unchecked undeclared witness minimizes `c + E[next value]`. It checks only when that lowers expected harm | ≤ B |

**Posterior shared by A5/A5a/A6.**
- The hidden configuration is h ∈ {0, 1}^m over undeclared witnesses (1 = independent).
- **Prior.** P(h | m) is the dev count over both twins with +0.5 smoothing over all 2^m configurations.
- **Text likelihood.** For each witness, the feature is its term-frequency cosine to the root, cut into
  5 equal-width bins on [0, 1]. P(bin | copy) and P(bin | independent) are dev counts with +1
  smoothing.
- P(h | obs) ∝ P(h | m) ∏ P(bin_i | h_i), conditioned on every check result.
- SUPPORT is true iff some h_i = 1. Declared witnesses are copies with certainty.

## Pricing, harm and coverage

- **Decision costs.** False SUPPORT on C (a withdrawn claim propagates): w_FS = 1.0. False RETRACT on
  I: w_FR = 1.0. ABSTAIN: w_A = 0.35. Each check: c = 0.10. Correct decision: 0.
- **Item harm** = decision cost + c × checks used.
- **Prevalence-weighted harm**, for π ∈ {0.05, 0.25, 0.50}, where π is the share of
  withdrawn-root claims whose survivors are all hidden copies:
  `H(π) = π·mean_harm(C) + (1−π)·mean_harm(I)`.
- **Coverage(π)** = π·answered(C) + (1−π)·answered(I), where answered means not ABSTAIN.
- **Sensitivity grid (reported, not part of the decision).** c ∈ {0.02, 0.10, 0.25}, w_A ∈ {0.20,
  0.35, 0.50}, w_FR ∈ {0.5, 1.0}, B ∈ {1, 2}, λ ∈ {0, 0.25, 0.5, 1.0}.

## Kill / advance rule (frozen)

The rule is evaluated on the **test pairs of all 3 seeds, pooled**, with λ = 0, B = 1 and primary prices.

1. **Fixture gate.** If H1 fails (any λ = 0 balanced accuracy > 0.55), the verdict is
   `FIXTURE_INVALID_LEAKAGE` and no T-D verdict is issued.
2. **Conventional comparator set.** CONV = {A3, A4, A5, A5a}. A6 is the T-D candidate.
3. **`KILL_T_D`** if at least one arm X ∈ CONV satisfies **both**, at every π ∈ {0.05, 0.25, 0.50}:
   - `H_X(π) ≤ H_A6(π) + 0.005`
   - `Coverage_X(π) ≥ Coverage_A6(π) − 0.005`

   The check cap is equal (B = 1) for both, and the harm already charges every check at c.
4. **Otherwise `ADVANCE_T_D`.** The report then lists the π values and the per-pair outcomes that
   carry A6's margin.
5. The pair-clustered bootstrap CI of `H_A6(0.5) − min_X H_X(0.5)` is reported but is not part of the
   decision.

## Secondary outputs (not decisive)

- EK5_v3 vs the ledger on twins: the disagreement count, and per-pair accuracy (H2 check).
- The graded-leakage curve: classifier accuracy and each policy's harm vs λ.
- The sensitivity grid.
- Cut 25 accuracy for A2/A3.
- The legacy leakage audit.

## Planned artifacts

- `artifacts/pv7_w23b_w22_07/w22_07_results.json` and `report.md`.
- Bulk fixture dumps go to `<out>/scratch/`, which is gitignored.
- Module `w22_07_copy_twins.py`.
- Runner `scripts/run_pv7_w23b_w22_07_copy_twins.py --out-dir`.
- Tests `test_w22_07_copy_twins.py` (unittest).

## Declared departures from the spec text (made in advance)

- **D1.** #35 and #46 legacy fixtures are absent on disk, or need models to rebuild. They are not
  replaced or audited here. Their blinded-twin replacement is the generic fixture above, which is
  built from #55 real roots.
- **D2.** The structure of the twins (witness counts, declared links, which witnesses are independent)
  is simulated. Only the root texts and claims are real. Independent witnesses reuse transforms of the
  root text, by design, so that text cannot reveal provenance. That is the hidden-copy regime, and it
  is not a claim about how often hidden copies occur in the wild.
- **D3.** The "T-D" policy is my own instantiation (A6). No T-D implementation exists in the repo.
