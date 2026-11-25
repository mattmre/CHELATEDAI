# Partner Charter — Trial 2: Records-Owner Governance (Bundle #64)

**Status:** `protocol-ready` (claim tier **T4**). Nothing in this charter is a result. The scorer
(`partner_kits/trial2_records_governance_kit.py`) has only been exercised on **synthetic open DEV
fixtures** (`data_origin = "open_dev_fixture"`); no partner records have been scored.
**Wave / branch:** Wave 21 · `pv7/t5b-partner-records` · Item 5.2.
**Anti-"Muddy-Waters" seal:** a tie or a miss is sealed permanently as `verified-loss`; the premium
claim is never re-argued after unblinding.

## Three questions every partner reads first

1. **What would the partner deploy without us?**
   The records owner's current resolution path for conflicting statements about the same fact across
   closed matters (their incumbent system, or the independent bitemporal SQL `ChallengerLedger`
   run as the baseline arm). That is the comparator — not "no system".
2. **What is the smallest difference that would change that choice?**
   A lower-confidence-bound advantage of **at least 0.25 loss units per natural-conflict matter**,
   in *both* the Attribution Pass and the Operational Pass, against *both* comparators (see the joint
   rule below). Anything smaller is a tie and the partner keeps the incumbent.
3. **What result retires our premium claim?**
   Failing *either* comparison, *either* pass, the explanation-faithfulness non-inferiority check, or
   the timeout ceiling retires the EK5 premium. There is no partial credit and no re-run.

## Arms (identical matters, identical order)

| Arm | What it is |
|---|---|
| `ek5_on` | EK5 stratified evidence engine with explanations enabled |
| `ek5_off` | Same pipeline, EK5 strata disabled (ablation of the premium itself) |
| `challenger_ledger` | Independent bitemporal SQL ledger (`chelated_pv7_spine/challenger_ledger.py`) |

## Natural-conflict base-rate sizing (locked before data access)

The partner declares the base rate `b` (0 < b ≤ 1) of **naturally occurring** conflicts among their closed
matters. The sample is the **`N ≥ ceil(30 / b)` consecutive closed matters** ending at an agreed cut-off
date — no gaps, no hand-picked matters. `N` is computed with exact rational arithmetic.

* Fewer than **30 natural conflicts** (or fewer than `N` matters, or a `matter_seq` gap) ⇒ verdict
  **`INSUFFICIENT_EVIDENCE`**. No win, loss, or tie may be claimed. Gaps are rejected outright as
  selection bias.
* Injected/synthetic conflicts never count toward the 30.

## Pre-locked owner loss weights (loss units per matter; lower is better)

| Outcome | Loss |
|---|---|
| `correct` | 0 |
| `abstain` (escalated to a human) | 2 |
| `timeout` | 6 |
| `wrong` (confidently incorrect resolution) | 10 |

Labor rates (Operational Pass only): extraction **0.05 / min**, review **0.20 / min**. The owner may
nominate different weights **only before data access**; the SHA-256 of the locked constants
(`protocol_digest`) is stamped on every report so any later change is visible.

## Two passes

* **Attribution Pass** — decision loss only. Isolates what the engine *decided*.
* **Operational Pass** — decision loss **plus** extraction and review labor. What the owner *pays*.

An arm that is more accurate but needs far more reviewer time can win the Attribution Pass and lose the
Operational Pass; the premium requires both.

## Blinded explanation-faithfulness rubric

For every natural-conflict matter a reviewer who cannot see arm names scores each arm's explanation
under an opaque code (`X1`–`X3`) on four binary criteria:

1. `names_deciding_record` — the explanation identifies the record that decided the outcome;
2. `cause_matches_ground_truth` — the stated cause class (superseded / refuted / unfounded cycle /
   peer conflict / temporal) matches the adjudicated ground truth;
3. `no_unsupported_claim` — nothing asserted that the records do not support;
4. `reviewer_can_act_without_rerun` — the reviewer can act without re-executing the system.

Faithfulness is the mean fraction of criteria met. `ek5_on` must be **non-inferior to the
ChallengerLedger within 0.05**. The code→arm unblinding key is held by the trial coordinator and applied
by the scorer only after all scores are fixed.

## Joint retirement rule

The EK5 premium is **retained only if all hold**:

1. `EK5-on vs ChallengerLedger`: lower 95% one-sided confidence bound of the mean per-matter loss
   advantage ≥ 0.25 in the Attribution Pass **and** the Operational Pass;
2. `EK5-on vs EK5-off`: the same two bounds ≥ 0.25;
3. explanation faithfulness non-inferior to the ChallengerLedger (margin 0.05);
4. `ek5_on` timeout rate ≤ 10%.

Otherwise the premium is **retired**, and the report lists each failing condition in
`retirement_causes`. Outcome classes: `win` (retained), `timeout`, `loss` (mean advantage < 0 vs the
challenger in either pass), `tie` (retired without being worse), or `insufficient_natural_conflicts`.
The lower bound is `mean − 1.645·sd/√n` over natural-conflict matters (normal approximation; paired
per-matter differences).

## Data handling

Partner records stay on partner infrastructure. The kit consumes **row-level outcomes only** (one JSONL
row per matter: outcome, minutes, blinded rubric scores) — never record text. Run it offline:

```
python partner_kits/trial2_records_governance_kit.py --run-dev-fixtures            # synthetic DEV only
python partner_kits/trial2_records_governance_kit.py --rows-jsonl rows.jsonl \
    --unblinding-key-json key.json --declared-base-rate 0.3 --output report.json
```

## Known limits (disclosed up front)

* One-sided normal-approximation bound; with n≈30 natural conflicts and heavy-tailed losses it is
  approximate. A bootstrap or exact test is deferred scope, not silently assumed.
* The 4-criterion rubric is binary and unweighted; inter-rater agreement is not computed by this kit.
* Loss weights are owner-set; a different owner can legitimately reverse the verdict.
* DEV fixture output demonstrates that the scorer *runs and separates* win/loss/tie/timeout/insufficient
  cases. It is not evidence about any real records corpus.
