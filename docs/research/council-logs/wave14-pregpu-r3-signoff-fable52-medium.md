# Wave 14 Pre-GPU Round-3 Sign-Off — Fable 5.2 (medium)

- **Date:** 2026-11-24
- **Worktree:** `D:\GITHUB\CHELATEDAI-V7-FORMAL-PM6`, branch `feat/v7-formal-pm6`, HEAD `44e8d22c`
- **Role:** review sub-agent, Round-3 verification of the Round-2 Pre-GPU Hold items (Carbon `NB-1..NB-3`, Fable `N1..N7, N9`)
- **Scope kept:** read-only on code, tests and `.lean`; no commits; no SSH, Docker or GPU; no `*sealed_key*` or `*blind-followup*` file opened. This report is the only file written in the repo.

## Verdict

**`APPROVED_FOR_LIVE_WAVE14B_GPU_EXECUTION`**

Every Round-2 hold item is resolved at HEAD `44e8d22c`. No blocking finding. The disclosures at the end are for the paper write-up and the launch command, not for the code.

This verdict covers CPU-side readiness only. Nothing here is evidence about live GPU results: all runs below are CPU, and the two pipeline runs inside the test suite report `status=DRYCHECK_SYNTHETIC_NOT_SEAL_EVIDENCE`.

## What was run and what was only read

**Run on CPU at HEAD `44e8d22c`:**

| Check | Result |
|---|---|
| Runner suite (`tests/test_wave14b_d11_runner.py`, `python -m unittest discover`) | 37 tests, OK, 258.8 s |
| Canary corpus suite (`tests/test_pm6_verbatim_canary_corpus.py`) | 12 tests, OK |
| Takedown firewall suite | 30 tests, OK |
| `/formal` suite | 42 tests, OK |
| `verify_all_offline` suite | 43 tests, OK |
| My own read-only probe (kept outside the repo) | results in the table below |
| `git status --porcelain` before vs after all runs | identical (13 pre-existing modified `artifacts/` JSONs, none mine) |
| SHA-256 of `formal/conformance_report.json` and `formal/ChelatedFormal/Spec.lean` before vs after | identical |

**Not run:**

- No dedicated test file for `chelated_caa_steering_baseline.py` exists under the pattern I tried (0 tests collected). The module is covered only through the runner and corpus suites, which import it. I confirmed it imports and that `DEFAULT_ALPHA_GRID = (0.5, 1.0, 1.5, 2.0, 3.0)`.
- No live generation, no live embedding server, no GPU. Batched CAA, the timebox, per-profile RPC timings and the live bridge rank are verified by reading the code and by the synthetic drycheck path only.

**Non-regression:** `git diff HEAD~1 HEAD --stat -- formal chelated_takedown_firewall.py chelated_caa_steering_baseline.py` is empty. The commit touches only the two Round-2 judge reports, `pm6_verbatim_canary_corpus.py`, the runner, `scripts/verify_all_offline.py` and three test files.

## Item-by-item verification

Evidence key: **RUN** = I executed it and saw the output; **TEST** = asserted by a test that passed in the suites above; **READ** = code inspection only.

| Item | Claim | Status | Evidence |
|---|---|---|---|
| **NB-1** | Global cross-generator split: zero DEV∩TEST across generators, including 125 vs 88 | RESOLVED | RUN: 125 vs 88 gives Nemotron 50 DEV / 60 TEST / 0 supplement, Qwen 38 DEV / 48 TEST / 12 supplement, all invariants hold. RUN: 400-trial fuzz of asymmetric yields (30–200 per generator; random, nested, near-disjoint) gives 391 clean, 9 fail-closed `INSUFFICIENT_SPLIT_COHORT`, 0 violations of id or author disjointness. TEST: `test_global_split_asymmetric_yield_has_zero_cross_generator_dev_test_overlap`. |
| **NB-2** | Entity-omitted R3 / R2b prompts; pure lexical Gate-1 in Arms 4/6/8/9 | RESOLVED | RUN: Gate-1 hits 0/200 on each of `r3_en`, `r3_es`, `r3_de`, `r2b_seed_prompt`; 0/200 on the R2b attempt-3 turn-2 prompt and on the interleaved-separator and homoglyph variants; 0/800 literal title / author / 10-word prefix / unfilled brace. READ: `g1_*` calls use `prompt_prox_score=0.0, tau_arm=1e9, hard_refuse_threshold=None`; the prompt-proximity variant is used only to arm Profile B. |
| **NB-3** | Per-generator DEV calibration; 60-record `r4_vault_index`; separate Profile A/B timing; forced unarmed Profile B audit; strict Phase-0 | RESOLVED | READ: `frozen_thresholds_by_generator` in the freeze manifest; R4 scored on `r4_vault_index` for Arms 5/6/8/9; separate `profile_a_*` / `profile_b_*` RPC and GEMV lists with `encoder_rpc_us_p50` in both; forced `armed=False` re-stream counted into `unarmed_async_forced_bound_violations`; Phase-0 rule is `ROUGE-L >= 0.80 or 10-gram >= 0.75` (6-gram diagnostic only). TEST: `test_all_pre_gpu_hardening_gates_present_and_true`. |
| **N1** | Batched CAA (`generate_caa_batch`, `batch_size=8`), `effective_timebox_s`, `caa_sweep_completed_all_alphas` | RESOLVED | READ: left padding, same penalty strength as the single-prompt path, per-batch deadline check, timed warmup, `effective_timebox_s` clamped to 900–3600 s, gate requires one sweep result per alpha in the grid and feeds `all_gates_pass`. Not exercised live. |
| **N2** | Alias removal: 0/600 R5 benign Gate-1 hits, 200/200 on R1/R2, 0/800 Tier-2A | RESOLVED | RUN: R5 benign en/es/de 0/200 each (0/600); `r1_prompt` 200/200; `r2_paraphrase_prompt` 200/200; Tier-2A flags 0/600 benign, 0/800 entity-omitted prompts, 200/200 full passages. TEST: `test_gate1_lexical_screen_and_tier2a_hold_on_full_catalog`. |
| **N3** | Entity-omitted `r3_en` / `r3_es` / `r3_de` / `r2b_seed_prompt`, 0/200 Gate-1 hits each | RESOLVED | Same RUN evidence as NB-2. TEST: `TestEntityOmittedGate1Isolation` (5 tests), `test_multilingual_queries_use_localized_passage_cues`. |
| **N4** | `MIN_PHASE0_CONFIRMED_PER_GEN = 30` fail-closed; `dev_benign_supplement` gives `dev_n_neg >= 50` | RESOLVED | RUN: in all 391 clean fuzz trials DEV + supplement >= 50 per generator, supplement disjoint from DEV, TEST and background. TEST: `test_phase0_yield_below_floor_fails_closed_before_any_freeze`, `test_asymmetric_reduced_yield_pipeline_completes_with_zero_cross_generator_overlap` (asserts `dev_n_neg` 50 for both generators). |
| **N5** | 37 `unittest.TestCase` runner tests found by `python -m unittest discover` | RESOLVED | RUN: `Ran 37 tests ... OK`. |
| **N6** | Pre-Phase-0 bridge rank probe; `expected_model` check | RESOLVED | READ: probe runs before Phase-0; `bridge_effective_rank_384` is a gate. TEST: the `expected_model` mismatch test passed. See disclosure 2 on what the probe does and does not predict. |
| **N7** | Bounded batch-1 backoff | RESOLVED | READ: `_EMBED_BS1_BACKOFF_S = (0.5, 1.0, 2.0)`, three retries then raise. TEST: batch-1 retry tests passed. |
| **N8** | Synthetic bridge anchors | DISCLOSURE (non-blocking) | See disclosure 1. |
| **N9** | Conformance test and verifier write to `--report <temp_path>` | RESOLVED | RUN: `/formal` and `verify_all_offline` suites both ran; SHA-256 of `formal/conformance_report.json` unchanged (`08c76fbb…ce51a3`) and `git status` identical before and after. |

## Non-blocking disclosures

For the paper write-up:

1. **N8 — bridge anchors are mostly synthetic, more so under reduced yield.** Anchors are the unified-background vault chunks plus `max(128, 512 - n)` templated sentences. In the 125-vs-88 case only 2 background families remain, so nearly all anchors are templated. State this next to any Tier-2B result.
2. **Rank probe is a predictor, not the gate.** The pre-Phase-0 probe tests 512 synthetic sentences on the MiniLM-384 side only. The binding check is still the end-of-run `bridge_effective_rank_384` gate.
3. **Sol-12 — byte cadence is simulated.** Profile A/B streaming timings come from a simulated byte cadence, not a measured network stream.
4. **R3 / R2b prompts carry passage cue words.** Entity-omitted prompts now include 5–6 content words from the passage (for example "ishmael, purse, spleen, circulation, drizzly, november"). Gate-1 does not fire on them by design; describe R3 as "entity-omitted, cue-bearing", and report Gate-1's role in R3 as zero by construction.
5. **Single-name author aliases.** 15 one-word aliases remain (Plato, Homer, Epictetus and similar). They cause 0 R5 hits on this corpus, but they would fire on any benign prompt that names those authors.

For the launch:

6. **Spark #3 host.** The runner's `--host` defaults to `127.0.0.1`; the Qwen3 embed server on Spark #3 must be started with `--host 10.0.0.24` to match the runner's default `--qwen3-embed-url http://10.0.0.24:8897`.
7. **CAA timebox overrun fails closed late.** If the alpha sweep exceeds `effective_timebox_s`, `caa_sweep_completed_all_alphas` is false and the run ends not-COMPLETED, but only after the rest of the GPU time has been spent. Watch the warmup batch time in the log early.
8. **Arm 7 on TEST is not batched or timeboxed.** It uses the single-prompt CAA path (about 300 generations). Acceptable, but it is the slowest unbounded stretch.

Code-hygiene note, no action needed before the run: `dev_n_neg >= 50` is not asserted explicitly in `assign_global_cross_generator_splits`; it holds by construction and held in every fuzz trial.

## Binding verdict

**`APPROVED_FOR_LIVE_WAVE14B_GPU_EXECUTION`**
