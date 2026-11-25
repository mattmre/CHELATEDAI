# Wave 14 Pre-GPU Judge Verification (Fable 5.2, medium)

**Verdict: `HOLD_WITH_FIXES`**

- Worktree: `D:\GITHUB\CHELATEDAI-V7-FORMAL-PM6`, branch `feat/v7-formal-pm6`, HEAD `50a0bc02`.
- Date: 2026-11-24. Role: independent pre-GPU judge. No code, `.lean`, commit, SSH, Docker or GPU action taken.
- Round-1 inputs: `wave14-code-xval-{sonnet55-high,sol62-high,neon4-high,grok47-xhigh}.md`.

The remediation is real. Of 37 Round-1 BLOCKERs, 30 are RESOLVED, 7 are PARTIAL (the code defect is fixed, a
named residual remains), and none is fully OPEN. I found no stub, modulo arithmetic or fabricated arm in the live
path. The hold rests on four new issues (N1-N4 below) that would either waste Spark hours or make a headline
number uninterpretable. All four are small fixes.

Round-1 reports use only BLOCKER and MINOR tiers; there is no MAJOR tier to map.

## 1. What I ran and what I only read

| Evidence | Result |
|---|---|
| `sha256sum formal/ChelatedFormal/Spec.lean` | `63fe3e64...cdc6eb`, matches `formal/spec_sha256.json` |
| `python -m unittest tests.test_wave14a_formal_conformance` | 42 tests OK (7.6 s), includes the 1,000-sequence conformance against the built `ek5_ref_cli.exe` |
| `python -m unittest tests.test_wave14b_takedown_firewall` | 30 tests OK (8.6 s) |
| `python -m unittest tests.test_wave14b_d11_runner` | **Ran 0 tests** (file is pytest-style, see N5) |
| `python -m pytest tests/test_wave14b_d11_runner.py` | 17 passed (128.7 s) |
| Tier-2A false-positive probe (my script, production scanner, runner settings `ngram_n=10`, `lcs_min_tokens=14`, ratio 0.72, 16-token windows) | 0 / 800 R5 benign texts flagged (4,613 windows); 0 / 100 held-out literary passages flagged against a vault of the other 100 (6,145 windows); 40 / 40 positive controls flagged; 1.5-2.3 ms per window with 200 works |
| Gate-1 alias probe (production `TakedownLineageFence.screen_prompt`, `aliases_by_work`) | **7 / 200 English R5 benign prompts hard-blocked** (see N2); 200 / 200 `r3_en` prompts hard-blocked |
| Corpus load (`pm6_verbatim_canary_corpus`) | 200 works, 200 distinct `author_group_id`, 200 distinct R5 queries per language and references, 60 fixtures over 60 distinct orgs, 60 distinct secret tokens; passages 65-80 words, continuations >= 55 words |

Not run: `scripts/verify_all_offline.py --full-formal` (it rebuilds and writes files), `lake build`, nanoda, any
live endpoint, any GPU path. Every live-path statement below is from code reading only.

Read in full or near-full: the runner lines 300-4979 except `SyntheticDrycheckSpace` (1744-1980); firewall
scanner, vault index, alias screen and stream gate; CAA tokenisation, pooling, calibration and generation;
verifier formal tier; `Statement.lean`; `Theorem2`. Not read: runner 1-300 beyond constants, firewall
canonicaliser (107-262), `TakedownLineageFence` ingestion (896-1025), `TakedownReceiptLedger` (1596-1719),
CAA direction extraction and hooks (253-516), `Theorem1a/1b/3/4` proofs, `Ek5RefCli.lean`, most of
`run_conformance.py`. I did not re-run Sol's Unicode-format probe for blocker 13.

**Side effect I caused and reverted.** Running the Wave-14A unittest rewrote the tracked
`formal/conformance_report.json`. The only changed line was `elapsed_seconds` (3.93 to 5.04); every count and
gate reproduced exactly. I restored the file with `git checkout -- formal/conformance_report.json`; `git status`
is back to its prior 13 entries (all older `artifacts/` JSONs that were already modified before I started).

Paths below: `runner` = `scripts/run_wave14b_d11_pm6_takedown_carryforward.py`, `fw` =
`chelated_takedown_firewall.py`, `caa` = `chelated_caa_steering_baseline.py`, `verifier` =
`scripts/verify_all_offline.py`.

## 2. Round-1 BLOCKER checklist

### Sonnet 5.5

| ID | Finding | Status | Evidence |
|---|---|---|---|
| S-B1 | Arms 1/2/7 fabricated | RESOLVED | Real per-arm generation in `_evaluate_generator_on_test`, runner:3213-3743; CAA arm calls the hooked model, runner:3399-3404 and 3550-3552. Residual: Arm 7 can still be degenerate, see N1. |
| S-B2 | Profile A/B "0 leak" vacuous | RESOLVED | Leak counters come from real streamed outputs, runner:4721-4731, gated at 4741-4743; delay test needs an abort on all 16 offsets plus a client-byte check, runner:2940-3054. Residual: N3, N6. |
| S-B3 | Tier-2A LCS throughput, needs prefilter and >= 60-work timing test | RESOLVED | Multiset prefilter before exact LCS, fw:471-488; 60-work test `tests/test_wave14b_takedown_firewall.py:143-162` passes. Its works are synthetic tokens; my probe on 200 real passages measured 1.5-2.3 ms per window, at or slightly above the 2 ms bar. |

### Sol 6.2

| ID | Finding | Status | Evidence |
|---|---|---|---|
| Sol-1 | EK5 differential passes with 153 incremental diffs | RESOLVED (adjudicated) | 153 diffs are confined to `pos_cyc` (104) and `reason` (49); `semantic_divergences_total: 0` in `formal/conformance_report.json`; scope labelled "conformance-tested, unproven" in `formal/ChelatedFormal/Statement.lean:50-58`. Reproduced locally. |
| Sol-2 | Harness shares Stratum-0 state; oracle reads `store.refs` | RESOLVED | Independent `_Model`, `formal/run_conformance.py:93`; snapshots from the model, :321; oracle from the raw snapshot, :490-500; 7,722 materialisation checkpoints, 0 mismatches. Residual: the oracle function itself is `ek5.compute_alternating_wfs_oracle` in the production module (:498). |
| Sol-3 | No external-kernel provenance | PARTIAL | Fixed: logical bin names and pinned revs in `formal/nanoda_manifest.json`; resolvers, verifier:984-1021; staleness check, verifier:1049-1060; N > 0 and N >= expected, verifier:1024-1036; fail-closed `NOT_RUN` rc=2, verifier:1184-1196 and 1306-1307. Still open: pinned revs are never checked against the resolved binary (no digest); staleness is mtime-based; the runner consumes no formal receipt (no reference to the verifier or conformance report anywhere in the runner). |
| Sol-4 | Theorem 2 omits runtime quantities | PARTIAL | 2x CSLS scale convention documented, `Statement.lean:36-41`; non-vacuity witness, `Theorem2_AdjointZeroMuPerturbation.lean:108-120`. The locked statement still does not bind `γ_d` to the centred document norm, and I did not find or verify the BEIR clearance evidence. Not a GPU blocker; a seal-claim limit. |
| Sol-5 | Nemotron contract | RESOLVED | `nemotron-lightning`, runner:126; `enable_thinking: False`, :2059; Bearer, :2061; `finish_reason` and truncated-empty handling, :2074-2080. Residual: preflight passes on an empty served-model list, :2020. |
| Sol-6 | Fabricated memory target, cohort union | RESOLVED | Template markers rejected and length floors, runner:488-534; enterprise fixtures separate, :561-604; Qwen chat template, :2386-2406; per-generator confirmation with intersection, :3909. |
| Sol-7 | Weak leakage tripwire | RESOLVED | Family, author group, author entity, node ID, normalised-text hash, both vector spaces required, runner:882-1029; fail-closed vector validation, :847-879. Note: returned overlap counts are literal zeros (:1019-1023), acceptable only because any overlap raises. |
| Sol-8 | Ref scoring depends on TEST; freeze incomplete | RESOLVED | `doc_mean` defaults to the ref mean, runner:1366-1404; both pool hashes, :4247-4252; freeze before TEST, :4536-4580, enforced by `assert_calibration_frozen`, :1032-1047; DEV vault padded to TEST node count, :4002-4020. |
| Sol-9 | Nine-arm matrix simulated | RESOLVED | runner:3213-3743, real requests per generator, outputs scored. |
| Sol-10 | Matched-FPR and cluster inference incomplete | PARTIAL | Tie-safe threshold with pre-triggers in the FP budget, runner:1072-1146; cluster bootstrap, :1191-1280. Still open: `dev_met_target_fpr` (:1143) and `caa_met_target_fpr` (:4572) are recorded but not in `all_gates_pass` (:4734-4750); see N2 for a measured reason the target can be missed. |
| Sol-11 | Cross-lingual ablation confounded | PARTIAL | Native 384 and 4096 means, runner:4162-4167. ES/DE benign queries now exist in the corpus. I did not verify that positives and negatives are language-matched in the ablation. |
| Sol-12 | Transport benchmark vacuous | PARTIAL | Real 0.5 s sleep at all 16 offsets with positive control and client-byte inspection, runner:2940-3054. Profile B async is still a two-pass simulation at a synthetic cadence (runner:2853-2934) and Nemotron is not consumed as a hosted byte stream. Must be reported as simulated transport. |
| Sol-13 | Tier-2A flags bytes already emitted | RESOLVED (code read) | Constructor invariants, fw:1134-1145; release rule, fw:1529-1544; strict-mode SSE rejection aborts, fw:1216-1218; `H0_HOLDBACK_TOKENS=16`, runner:150. Firewall suite passes. Sol's probe not re-run. |
| Sol-14 | R2b static, pre-freeze | PARTIAL | R2b is evaluated post-freeze on model outputs inside the test evaluator. Variants are still three fixed transforms (runner:373-420); I did not find attempt selection that adapts to prior results. |
| Sol-15 | Assembly trusts booleans | RESOLVED | Independent chain, sequence, Ed25519 and Non-NDA re-verification, runner:1654-1738. Residuals: public key comes from the same bundle; file-hash compare is skipped when the reference field is absent, runner:4857 and 4864. |
| Sol-16 | Encoder identity, GPU, row validity unchecked | PARTIAL | CUDA required, runner:2312-2316; lock, :2218; response validation in the client, :2109-2198. `expected_model` is never passed, so a same-width substitute encoder is still accepted. |
| Sol-17 | Completion gate admits drycheck or incomplete evidence | RESOLVED | Drycheck/production guard, runner:3768-3774; `COMPLETED` needs live mode and real sleep, :4752-4761. Residual: reduced-yield runs can still be `COMPLETED`, see N4. |

### Neon 4

| ID | Finding | Status | Evidence |
|---|---|---|---|
| N-B1 | Tier-2A 10-char n-grams, 100% R5 FPR | RESOLVED | Word n-gram index, fw:341-349 and 441-457; runner disables char shingles, runner:1486-1488. Measured 0 / 800. |
| N-B2 | Ref pool centred on TEST `doc_mean` | RESOLVED | fw:596-601; runner:1366-1404. |
| N-B3 | `screen_prompt` alias inversion | RESOLVED | fw:1026-1064 with `aliases_by_work`; runner:1492-1510. New measured side effect: N2. |
| N-B4 | Profile B exposure cap; `h0 < lcs_min_tokens` | RESOLVED | fw:1144-1145 and 1521-1523; runner:150. |
| N-B5 | Phase-0 crash, modulo arms | RESOLVED | runner:488-534, 3909, 3213-3743. |
| N-B6 | MiniLM arm centred with projected Qwen means | RESOLVED | runner:4162-4167. |
| N-B7 | Unsynchronised CUDA, no `pad_token` | RESOLVED | runner:2218 and 2321-2323; caa:143-151. |
| N-B8 | Hardcoded manifest paths, no pins, no staleness check | RESOLVED | `formal/nanoda_manifest.json` (schema 2); verifier:984-1021 and 1049-1060. |

### Grok 4.7

| ID | Finding | Status | Evidence |
|---|---|---|---|
| G-B1 | Catalog cannot yield >= 110 memorisations | RESOLVED at corpus level | 200 genuine passages (spot-checked: Moby-Dick opening), 200 author groups. Actual yield is unknown until Phase 0. |
| G-B2 | Phase-0 union | RESOLVED | runner:3909. |
| G-B3 | Nemotron omits `enable_thinking: False` | RESOLVED | runner:2059. |
| G-B4 | Qwen CAA no pad token | RESOLVED | caa:143-151; pooling correct for either padding side, caa:172-182. |
| G-B5 | Embed server OOM/race | RESOLVED | runner:2218, 2272-2277, 2292-2296. |
| G-B6 | Split not author-disjoint | RESOLVED | runner:727-844, hard fail at :818-825. |
| G-B7 | Delay test passes without flagging | RESOLVED | runner:2940-3054; `test_delay_safety_test_fails_closed_if_windows_do_not_abort` passes. |
| G-B8 | Live run skips matrix; alpha returned after missing FPR | RESOLVED | Matrix as above; infeasible sweep returns alpha 0.0 with `met_target_fpr=False`, caa:615-627. Residual: N1. |
| G-B9 | Unit tests certify dry path only | PARTIAL | Live components now have tests (chat client, embed server, CAA hooks, delay fail-closed). End-to-end is still drycheck only, and the 17 runner tests are not executed by the repo's `unittest` CI command, see N5. |

Round-1 MINORs spot-checked as fixed: `PASS_PARTIAL` exit 0 (Grok M3; the constant survives unused at
verifier:73), freeze hash naming (Grok M4, Neon MINOR-7), `s_2b > w_2b` (Neon MINOR-3, fw:1134-1135), UTF-8
wedge (Neon MINOR-2, fw:1286-1305), CSLS factor of 2 (Neon MINOR-8). The remaining MINORs were not individually
re-verified.

## 3. New risks before live GPU execution

**N1. CAA calibration timebox is too small for the work inside it (fix before run).** The live evaluator does
three Qwen generations per DEV family per alpha (96, 96 and 64 new tokens; runner:4489-4514) over a five-value
grid (caa:30) inside one 900 s budget (runner:4522). At full yield that is roughly 150 generations per alpha
and 750 in total, so the sweep completes only if a generation averages about 1.2 s. I have not timed Qwen on
the target GPU; if it is slower, the sweep truncates to the lowest alphas or to none, and the fallback is
alpha 0.0 with null rates. `caa_met_target_fpr` is not a completion gate, so the run can end `COMPLETED` with a
non-functional Arm 7. Fix: time one generation first and size the budget from it, or subsample DEV, and gate or
loudly label Arm 7 when no alpha was fully evaluated.

**N2. Gate-1 alias "Essays" hard-blocks 3.5% of English benign prompts (measured; fix before run).** The
Montaigne work carries the one-word alias `Essays`. Seven R5 benign queries of the form "How did epistolary
moral essays ..." match it and are hard-blocked (fw:1048-1050). That is 7 / 200 English R5 prompts, above the
2% target, from Gate 1 alone and independent of any threshold. Pre-triggers count against the calibration FP
budget (runner:1101-1104), so matched-FPR may be unattainable for every Gate-1 arm. There are 45 one-word
aliases in total; others (`Ethics`, `Republic`, `Histories`, `Confessions`, `Meditations`) did not collide
with the current R5 set but are the same hazard. Fix: drop or qualify generic one-word title aliases, or
rewrite the seven queries, then re-run the probe.

**N3. Gate 1 decides every R3 positive.** All 200 `r3_en` prompts name the author and are hard-blocked by alias
match. In any arm that includes Gate 1, R1/R2/R3 leak rates are zero by construction and say nothing about
Tier-2B. Only the arm-isolation rows without Gate 1 carry semantic evidence. This is a reporting constraint,
not a code defect; the write-up must not present full-stack zero leak as a Tier-2B result.

**N4. Reduced-yield path is likely to burn Phase 0 and then fail, or to complete on a tiny cohort.** When
either generator confirms fewer than 110 works, the primary split is the Qwen split (runner:3923-3940), so
Nemotron TEST families can sit in Qwen DEV and the leakage audit (runner:4203-4242) should then fail closed
after the Phase-0 spend. Separately, the fallback accepts as few as 4 confirmed works in 4 groups
(runner:787-802), and neither yield nor the reduced-yield flag is in `all_gates_pass`. With fewer than 50 DEV
negatives the FP budget is zero (runner:1072-1146). Fix: decide the reduced-yield policy explicitly (one split
built from the intersection, a minimum cohort for `COMPLETED`), and stop right after Phase 0 if it is not met.

**N5. The runner test file is not run by the repo's test command.** `tests/test_wave14b_d11_runner.py` uses
module-level functions with `pytest.MonkeyPatch` and `tmp_path`; `python -m unittest` reports "Ran 0 tests,
OK". The CI workflow uses `unittest discover` and the repo convention says CI does not install pytest. The 17
tests pass under pytest locally, but a green CI is not evidence for them.

**N6. Late hard gates that depend on live embeddings.** `effective_rank == 384` (runner:4735) is an exact
equality on a bridge fitted to templated, near-duplicate anchor sentences (runner:610-705), and the delay test
requires the frozen arm-9 threshold to flag a lexical paraphrase at all 16 offsets. Both are evaluated only at
the end. Either can turn a multi-hour run into `FAILED`. The bridge anchors need only the two encoders, so
compute the rank before Phase 0 and abort early if it is short.

**N7. Embedding RPC has no resilience at batch size 1.** Per-window 4096-d scoring uses batch size 1
(runner:4172-4174); a single transient failure raises, there is no retry at that size, no checkpoint and no
resume. The embed server binds 127.0.0.1 by default (runner:4925) while the runner's default URL is
`10.0.0.24:8897`, so it must be started with an explicit host.

**N8. Templated background, reference and language-mean texts.** The reference pool, the ES/DE mean texts and
the bridge anchors are generated sentences from 20 topics by 16 regions in a civic/technical register
(runner:610-705). They are far from the literary canary domain and close to each other, which weakens the CSLS
hubness term and the language means as stand-ins for real traffic. Disclose it; it also feeds N6.

**N9. Smaller items.**
- Thresholds are calibrated on Qwen outputs and R3 positives only (runner:4333-4387) and then applied to
  Nemotron.
- R2/R3 semantic-leak scoring fires at content overlap 0.18 with the denominator capped at 25, about five
  shared tokens (runner:2748-2771); expect some false leaks.
- `generate_caa_drycheck` discards its hook result and its docstring says otherwise (runner:2692-2742);
  drycheck only.
- The Wave-14A unittest rewrites the tracked `formal/conformance_report.json` on every run, so a test run
  dirties a sealed evidence file.
- The assembled manifest is unsigned and the receipt public key is self-attested.

Retracted: I expected Tier-2A LCS to produce benign false positives from function-word overlap. The probe
found none in 10,758 windows, so that concern is withdrawn.

## 4. Fixes required to lift the hold

1. N1: size the CAA calibration budget from a measured generation time, and make an unevaluated sweep visible
   in the completion status.
2. N2: remove the generic one-word alias collision and re-measure benign Gate-1 hits to 0 / 600.
3. N4: define and enforce the reduced-yield policy immediately after Phase 0.
4. N6 and N7: move the bridge-rank check ahead of generation; add retry or resume to the batch-1 embed path;
   start the embed server with the right host and pass `expected_model`.

Recommended, not blocking: N5 (make the runner tests run under the CI command), record `--full-formal` output
as a receipt the runner checks (Sol-3), and state N3, N8 and the Sol-12 simulation limit in the results
write-up.

Once items 1-4 are in and a drycheck plus the two probes above are re-run, I would expect to approve.

**Verdict: `HOLD_WITH_FIXES`**
