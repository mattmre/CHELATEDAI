# W22-07: blinded hidden-copy twins and a priced provenance-check policy

| | |
|---|---|
| Preregistration | `b5096e50`, committed alone before any result |
| Code | `879acac5` |
| Results | `w22_07_results.json`, committed in `28b2ad4c` and produced from `879acac5` (`git_sha_at_run`) |
| Snapshot sha256 | `a3593820…d1a7aa` |

Command: `python scripts/run_pv7_w23b_w22_07_copy_twins.py --legacy38-osv-cache <W22-05 scratch>/osv_cve_seeds.json`.
It ran on CPU with no network, in about 3.5–7 minutes. Three full runs gave identical values for every key the verdict
depends on.

The implementer sub-agent's harness blocked it from writing this file, so the orchestrator transcribed it from the
sub-agent's final report. Every number below comes from `w22_07_results.json`.

## Verdict: `KILL_T_D` (PROPOSED; Tier B round 1 scored 88, important)

**Scope of the kill (Tier B round 1).** The verdict holds only where public evidence carries **zero provenance
signal**. At λ=0 the text likelihood is uninformative (P(bin | copy) ≈ P(bin | indep) ≈ 0.8 in the top bin). At the
primary price c=0.10 the pricing never binds: A6 checks exactly 1.0 witness per family and never abstains. So the
primary cell could not exercise either of A6's distinguishing mechanisms (targeting and priced stopping). A KILL was
close to guaranteed by construction, and ADVANCE was reachable only through noise. The preregistration expected this,
but it bounds what the kill means. Under mild signal (λ=0.25, where the classifier scores only 0.57–0.61), A6 beats
every conventional arm in all 18 B=1 cells. Do **not** cite this as a general kill of T-D. W22-10 and W22-12 should
read it as "no T-D premium without provenance signal".

### The fixture gate is valid

- **No leakage at λ=0.** On exact twins (λ=0), a text-only classifier is at chance. The highest balanced accuracy over
  3 seeds and both levels (family and witness) is 0.513, below the 0.55 threshold.
- **The test can detect leakage.** The same classifier reaches 1.000 at λ=1, and it already crosses the threshold at
  λ=0.25 (0.568–0.606).

| λ | Seed 26 (family / witness) | Seed 27 (family / witness) | Seed 28 (family / witness) |
|---|---|---|---|
| 0 | 0.500 / 0.485 | 0.500 / 0.513 | 0.500 / 0.490 |
| 0.25 | 0.588 / 0.599 | 0.585 / 0.568 | 0.590 / 0.606 |
| 0.5 | 0.677 / 0.749 | 0.685 / 0.700 | 0.692 / 0.763 |
| 1.0 | 0.879 / 1.000 | 0.854 / 1.000 | 0.858 / 1.000 |

Role-word violations: 0. Oracle mismatches: 0.

### Primary result

The primary setting is 360 test pairs, λ=0, B=1 and primary prices. H(π) is prevalence-weighted harm at π = 0.05, 0.25
and 0.5.

| Arm | H(0.05) | H(0.25) | H(0.5) | Coverage |
|---|---|---|---|---|
| A0 no-check | 0.050 | 0.250 | 0.500 | 1.0 |
| A1 cosine dedup | 0.753 | 0.640 | 0.500 | 1.0 |
| A2 EK5_v3 | 0.613 | 0.563 | 0.500 | 1.0 |
| A3 copy-aware ledger | 0.050 | 0.250 | 0.500 | 1.0 |
| A4 ledger + 1 check | 0.128 | 0.238 | 0.375 | 1.0 |
| **A5 Bayesian copy model (conventional)** | **0.237** | **0.208** | **0.172** | 1.0 |
| A5a A5 + abstain | 0.228 | 0.204 | 0.174 | 0.98 / 0.98 / 0.97 |
| **A6 T-D priced check** | 0.245 | 0.215 | 0.176 | 1.0 |

### Why T-D is killed

- A5, the conventional provenance-aware copy model, has lower harm than A6 at every π, with the same coverage.
- The bootstrap difference H_A6(0.5) − best conventional H is +0.0042, with 95% CI [−0.021, +0.028]. A6 is no better.
- Per seed, A5 vs A6 at π=0.5 is 0.154/0.171, 0.183/0.163 and 0.179/0.196. The sign flips between seeds, which fits
  noise.
- **The verdict is a pooled coin-flip (Tier B round 1).**
  - Under the frozen rule, seed 20261127 on its own gives **ADVANCE_T_D**, with no conventional arm matching it.
    Seeds 20261126 and 20261128 give KILL.
  - A5 and A6 decide differently on 77 of 720 families, all I members: 40 go one way and 37 the other, a net of 3.
  - Which witness A6 checks follows noise in the dev-fitted prior. For seed 26, m=2, the prior puts '01' at 0.105
    and '10' at 0.068, so A6 checks the *later* witness (`w22_07_copy_twins.py:536, 596-608`).
  - The true difference is about 0, so KILL is the right answer in expectation. It is not robust evidence.
- **The bootstrap CI is slightly too narrow.** The 360 pairs pool 3 seeds over the same 240 roots, so roots recur.
  Resampling clusters by `seed:pair` (`scripts/run_pv7_w23b_w22_07_copy_twins.py:52`), not by root. This does not
  change the decision.

### H2: only a check separates the twins

- A0–A3 give the same decision to both twins in 360 of 360 pairs.
- Policies that use only text or only the graph therefore sit at 0.5 accuracy.

### Graded leakage

A6 pulls ahead of A5 only once the fixture leaks. The table gives H(0.5).

| λ | A1 | A2 | A3 | A4 | A5 | A5a | A6 |
|---|---|---|---|---|---|---|---|
| 0 | 0.500 | 0.500 | 0.500 | 0.375 | 0.172 | 0.174 | 0.176 |
| 0.25 | 0.435 | 0.532 | 0.500 | 0.375 | 0.174 | 0.171 | 0.160 |
| 0.5 | 0.365 | 0.550 | 0.500 | 0.375 | 0.171 | 0.162 | 0.149 |
| 1 | 0.235 | 0.604 | 0.500 | 0.375 | 0.139 | 0.139 | 0.108 |

At λ=1, A6 also uses 28% fewer checks. On a templated fixture like the legacy ones, T-D would have looked like a win.

### The legacy fixtures leak completely

A text-only classifier scores 1.000 on each legacy fixture:

| Bundle | Records | Notes |
|---|---|---|
| #55 | 900 | |
| #20 | 3,840 | Scored with both the preregistered label and a widened copy-role label |
| #38 | 1,020 | The corpus hash matches the W22-05 gate |

### Sensitivity grid

- The kill condition holds in 20 of the 36 λ=0 cells.
- In 13 of the 16 cells where it fails, A6 is worse than the no-check ledger at π=0.05. (Corrected after Tier B
  round 1. An earlier version said all 16, which was a transcription error.) The other 3 cells are λ=0, B=2, c=0.02,
  w_FR=1.0, at all three w_A values:
  - A6 scores 0.042 against A3's 0.050, so A6 beats the no-check ledger and loses only to A4 (0.0304).
  - The kill fails there by a floating-point margin: A5 at π=0.05 is 0.047 against a bound of 0.047.
- At c=0.25 and w_A=0.20 (4 cells), A6's win at π=0.5 comes from not checking at all (0.00 checks). That is a
  stopping rule, not provenance targeting. In the other failing c=0.25 cells, A6 still checks (0.08, 0.88 or 0.99
  witnesses on average) and still beats A5, for example 0.314 vs 0.322.

| λ | Kill cells, B=1 (of 18) | Kill cells, B=2 (of 18) |
|---|---|---|
| 0 | 11 | 9 |
| 0.25 | 0 | 9 |
| 0.5 | 0 | 9 |
| 1 | 0 | 3 |

### Secondary result

- After withdrawal, EK5 and the ledger disagree on 450 of 720 families, and neither beats chance.
- Before withdrawal, both are at 1.000.

## Departures from the preregistration

- **D1 (declared):** #35 and #46 are not audited or replaced; only their manifests exist on disk.
- **D2 (declared):** the twin structure is simulated. Only the 240 root texts are real.
- **D3 (declared):** A6 is my own version of T-D.
- **D4:** the #38 audit used a byte copy of W22-05's OSV seed cache rather than the fallback the preregistration
  named, so that the corpus hash matches the W22-05 gate. No network either way. A one-seed smoke run on the fallback
  also scored 1.000; that run is not committed.
- **D5:** the #20 audit reports the preregistered label (the headline) plus a widened copy-role label. Both score
  1.000.
- **D6:** the preregistration did not fix the tie-break rules. These were used:
  - Decisions break ties in the order SUPPORT, then RETRACT, then ABSTAIN.
  - The dynamic program stops on a tie.
  - Target ties go to the lowest position.
  - Cosine-threshold ties go to the smaller value. At λ=0 every threshold tied, so it was 0.05.
- **D7:** the cosine threshold and the posterior were fitted once per (λ, seed) at primary prices and then held fixed
  across the sensitivity grid.
- **D8:** there is a bug in the shared `build_sql_ledger` in `scripts/run_pv7_w18_6c_wild_revisions.py` (about lines
  495–501).
  - It passes edge envelopes to `append`, which raises on them.
  - #55 never had edges, so the bug never fired there.
  - This module routes edges correctly. The shared script is left unfixed.
- **D9 (undeclared; found by Tier B round 1):** `CHECK_EDGE_TICK = 35` (`w22_07_copy_twins.py:74`), the timestamp on
  the DERIVES edge A4 adds after a check. It is not in the preregistration. Tier B judged it harmless.

## Tier B round 1

A fresh adversarial agent, `add5b29e07181830c`, scored this **88** (important). It could not disprove KILL_T_D:
- The preregistration came first.
- The arms and prices match it.
- A re-run into `scratch/tierb/` reproduced every key.
- The λ=0 twins are identical apart from blinded hashes.
- A5 gets no information A6 lacks, and the budget is equal: 1.0 check per family each.

It found:
- the false "all 16 cells" sentence and the over-broad c=0.25 sentence (both corrected above);
- that seed robustness was under-disclosed, and that the verdict needs scoping to zero provenance signal (both added
  above);
- the undeclared D9;
- the bootstrap clustering issue.

## Brutal Honesty (BHS_SELF_DRAFT 92; BHS_TIER_B 88; BHS_OFFICIAL 88)

- **L4:**
  - There is one generic twin fixture, built from #55 roots, rather than a replacement for each of #20/#35/#38/#46/#55.
  - The legacy fixtures were audited, not rewritten.
- **L5:** the runner's `main()` and the #20/#38 audits run end to end but have no unit tests. Everything else has unit
  tests: 23 tests, all OK, and ruff is clean.
- **L11:** `_git_sha` catches broad exceptions but records any failure in the JSON. It did not trigger.
- **L1/L3:** there are no stubs or mocks. The real EK5_v3 and the SQLite ledger are used, and both query modes are
  asserted equal.
- **Offline:** #20 imports `sentence_transformers` with `HF_HUB_OFFLINE=1`.
- **Fresh-checkout reproducibility:** the #38 cache comes from W22-05's gitignored scratch folder. On a fresh checkout
  the #38 audit falls back to the default values and produces a different hash. The verdict does not depend on it.
- **Limits:**
  - The policies are fitted on 50/50 C/I data and do not adapt to prevalence, so the no-check arms win at π=0.05.
  - No conventional arm stops when checks are costly.
  - With n=360, effects smaller than about 0.02 cannot be resolved.
- **CARRY_FORWARD:**
  - Fix the edge routing in `build_sql_ledger`.
  - Add a cost-aware stopping arm before any attempt to revive T-D.
  - Build twins for #35/#46 if their generators become runnable.
- **Scratch:** `artifacts/pv7_w23b_w22_07/scratch/` holds 6.1 MB of bulk data. It is gitignored and can be removed after
  review.
