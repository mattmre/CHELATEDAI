# Wave 15 Post-GPU Cross-Validation — Adversarial Review (Opus 5.5, high)

- Reviewer: Claude Opus 5.5 (`claude-opus-5-5`), a fresh adversarial review sub-agent under Brutal-Honesty Rulebook v3.3. Its job was to disprove completion, not to verify it.
- Date: 2026-11-25
- Range reviewed: `5f15c731..60ffc1f7`. Commits:
  - `f140e23b`: rank floor.
  - `f6a82cd2`: 100 public-domain canaries and the R1 span extractor.
  - `64ec9fcf`: stratified reduced-yield split.
  - `f357e4ba`: benign deduplication.
  - `60ffc1f7`: seal.
- Bundle: `artifacts/wave14_d11_pm6_takedown_carryforward/` (sealed at `60ffc1f7`; the receipt's `git_sha` is `f357e4ba`).
- Constraints honored:
  - No file was modified except this memo.
  - No `*sealed_key*` or `*blind-followup*` file was opened.
  - No GPU work was run. Tests ran with `CUDA_VISIBLE_DEVICES=""`.
  - The 13 unstaged CRLF `artifacts/...` files were left untouched.

---

## (d) Verdict, stated first

**BLOCK** (applies to the seal as currently framed).

The sealed bundle is internally consistent, the hashes verify, and the raw numbers are reported without fabrication. The data are honest. The **claims attached to the data are not**:

- The seal states `status=COMPLETED, all_gates_pass=True`.
- None of the six gates checks a TEST outcome.
- The headline arm (Arm 9) misses the pre-registered `TARGET_BENIGN_FPR = 0.02` on TEST for both generators.
- The required Holm–Bonferroni correction is absent from the script and the manifest.
- The bundle shows no measured advantage of the chelated mechanism over trivial baselines.

Presenting this as a passing seal is L13 (procedural gates presented as an efficacy proof) and L4 (partial presented as complete).

Separately, the brief's CPU suite command is red: 102 passed, 9 errors. The cause is order-dependent; the drycheck passes in isolation (B5, non-blocking, under "CPU tests" in §(b)).

**Unblock path. No GPU re-run is needed.** The verdict converts to **PASS_WITH_MANDATORY_DISCLOSURES** once both of these are true:

- B1–B4 below are fixed by a post-hoc analysis addendum computed from the already-sealed manifest. The sealed bundle itself stays unchanged.
- Every disclosure in §(c) appears in Whitepaper 7.0.

Nothing here requires re-opening the sealed key.

---

## (a) Blocking defects

### B1 — `all_gates_pass=True` is a procedural pass, not an efficacy pass (L13)

The gate set in `wave14_d11_pm6_manifest.json` is:

- `phase0_min_yield_met`
- `all_dev_calibrations_met_target_fpr`
- `bridge_effective_rank_384`
- `caa_sweep_completed_all_alphas`
- `unarmed_async_forced_bound_violations_zero`
- `zero_test_leakage_verified`

**None of these reads a TEST-cohort suppression or benign-FPR outcome.** The pre-registered constant is `TARGET_BENIGN_FPR = 0.02` (runner L139). Arm 9 (chelated full dual gate) on TEST:

| Generator | TEST families | R5 benign FP | FPR | Clopper–Pearson 95% upper | Target |
|---|---|---|---|---|---|
| Qwen2.5-7B | 17 | 2 | 0.118 | 0.326 | 0.02, **missed** |
| Nemotron-30B | 32 | 2 | 0.0625 | 0.184 | 0.02, **missed** |

DEV met the target at exactly 1/50 = 0.02 (`dev_met_target_fpr`). The calibration therefore does not transfer from DEV to TEST.

At n=17, even 0 false positives gives a CP upper bound of 0.162. At n=32 it gives 0.109. Neither TEST cohort can certify a 2% FPR, whatever the outcome. The commit-message phrase "all_gates_pass=True" must not be repeated in Whitepaper 7.0 as evidence that the firewall works.

### B2 — Holm–Bonferroni correction is absent

A search of the runner and the manifest for `holm`, `bonferroni`, `adjusted` and `fwer` found nothing. All reported p-values are raw. I applied Holm by hand to the 12 unique paired tests in the manifest (sorted raw p: 0.0001, 0.0001, 0.0029, 0.0163, …):

- Family of 12, α=0.05. Ranks 1–3 survive (0.0001 ≤ 0.00417, 0.0001 ≤ 0.00455, 0.0029 ≤ 0.005). **Rank 4 (p=0.0163) fails** (threshold 0.05/9 = 0.0056). Holm stops there.
- The 6-test cross-lingual sub-family gives the same result: **Spanish `c_vs_a` (AUC 0.958 vs 0.858, p=0.0163) does not survive**. German `c_vs_a` (p=0.0029) survives.
- `c_vs_b` (the effect of μ_lang centering) is non-significant for both languages before any correction (de p=0.867, es p=0.609). **The bundle shows no measurable contribution from language-mean centering.**
- A further caveat on the floor values: p=0.0001 is the bootstrap resolution floor (1/10 000). It is not an exact p-value.

### B3 — The headline mechanism is weakly dominated by its own baselines

Qwen TEST (n=17):

| Arm | R1 | R2 | R3 | R4 | R5 FPR | AUC |
|---|---|---|---|---|---|---|
| 9 chelated dual gate | 1.0 | 1.0 | 0.882 | 1.0 | 2/17 | 0.9758 |
| 8 linear bridge | 1.0 | 1.0 | 0.882 | 1.0 | **0/17** | — |
| 6 raw cosine | — | — | **1.0** | — | 1/17 | **0.9965** |
| 4 Gate-1 lexical only | 1.0 | 1.0 | — | 1.0 | — | — |

Paired AUC differences, none of them significant:

| Comparison | Qwen | Nemotron |
|---|---|---|
| Arm 9 − Arm 6 | −0.0208 (p=0.177) | −0.0107 (p=0.154) |
| Arm 9 − Arm 8 | −0.0138 (p=0.392) | −0.0039 (p=0.731) |

On Nemotron, Arm 9 has FPR 2/32 against 0/32 for Arm 8.

Arm 9 and Arm 5 share the frozen threshold `0.4627838706292536` and identical metrics. **The second gate of the "dual gate" therefore contributes nothing observable in this run.**

The only significant comparison favouring Arm 9 is against Arm 3, and `claim_scope_statement` compares against Arm 3 alone. Choosing the single beaten baseline for the scope statement is selective reporting (L4). The honest reading is this: at n=17/32, **the chelated/zeromu-CSLS mechanism is statistically indistinguishable from raw cosine and from a linear bridge, and numerically worse on FPR.**

### B4 — `bridge_effective_rank_384` is a misnomer and partly vacuous (L13)

What `f140e23b` changed:

- The gate now passes when the effective rank is **≥ 380**. The live value was 382.
- The gate passes when `stiefel_column_rank == 384`.
- The gate passes when the orthogonality error is below 1e-6.

The last two checks always hold by construction. In `CenteredProcrustesBridge.fit` (`spherical_migration_bridge.py`), `Q = U @ Vᵀ` comes from a thin SVD, so Q always has 384 orthonormal columns, whatever the data. The substantive check is therefore effective rank ≥ 380, and the gate's name claims 384. The 2–4 null directions of the target manifold are **unidentified** by the anchors: Q's action on them is arbitrary rotation, not a learned map.

Fix: rename the gate (for example `bridge_target_manifold_spanned_ge380`). Report `effective_rank=382/384` alongside `relative_frobenius_residual=0.650` and `captured_energy=0.903`. A residual of 0.65 is a weak linear fit and must be stated as such. The rename can live in the addendum. The sealed key name does not need to change.

---

## (b) Methodological and statistical critique, per commit

### `f140e23b` — the MiniLM "intrinsic manifold rank 381/382" claim

- **The rank is numerical, not intrinsic.** The commit and the constant comment (L143) assert an intrinsic LayerNorm manifold rank of "381 (f32) / 382 (f64)". The value changing with dtype shows it is tolerance-dependent. That makes it a numerical-rank artifact of the 1e-12 relative cutoff and the anchor sample, not an established property of MiniLM-L6-v2. LayerNorm's centering removes one direction per token before the affine map and the mean-pooling. A deficiency of 2–3 directions is plausible, but the bundle contains no artifact that proves it: no singular-value spectrum and no probe receipt.
- **Does it weaken the geometry?** Mildly. The effective rank of 382/384 is not a concern for retrieval. The concerns are these:
  - The gate floor was relaxed **after** the live run aborted on the old `!= 384` probe. That is a forking path. It is low-risk, because it happened before any TEST outcome existed, but it must be disclosed.
  - The "Stiefel 384" language in the commit title overstates what was enforced (see B4).
  - The residual of 0.65 is the bigger geometric weakness. The Procrustes map explains about 90% of the energy, but a third of the per-anchor Frobenius norm is unexplained.
- **Whitepaper 7.0 must disclose this:** "The 4096→384 bridge spans 382 of 384 target directions; 2 directions are numerically null in the anchor set and the map is unidentified on them; the relative residual is 0.65."

### `f6a82cd2` — 100 public-domain canaries and the R1 span extractor

- **DEV→TEST leakage: none found.**
  - All 100 rows have unique `work_family_id`, `author_group_id`, `title` and `author_or_entity`, and unique R1/R2/R3/R5 prompts.
  - The partition is author-disjoint, and `CROSS_GENERATOR_LEAKAGE` checks (work id, author group, author_or_entity) report 0 overlaps.
  - `zero_test_leakage_audit` reports 0 overlaps on all keys.
- **Distribution shift is real and undisclosed.** The new canaries are the most memorized texts in English (KJV passages, Amazing Grace, The Star-Spangled Banner, A Visit from St. Nicholas, constitutional amendments, nursery rhymes). Phase-0 confirmation selects exactly these: in the Nemotron samples, Moby Dick reached ROUGE-L 1.0.
  - The TEST cohort is therefore enriched for near-perfect, high-frequency regurgitation. That is the easiest case for an n-gram or alias gate (Arm 4 reaches R1=R2=R4=1.0 on its own).
  - None of these works is a copyrighted work under a live takedown, so the result does not transfer to a real takedown notice.
  - The manifest also does not list the per-source composition of the DEV/TEST cohorts (built-in versus external JSONL). The TEST shift cannot be quantified from the bundle. This is a reporting gap.
- **Takedown separation is made easier.** Yes, in two ways:
  - (i) The famous-text enrichment described above.
  - (ii) The `f357e4ba` interaction described below.
- **The span extractor loosens confirmation.**
  - `extract_r1_continuation_span` searches positions 1–19 for the first 4 **ground-truth** tokens and truncates to `len(gt)`. `is_memorization_confirmed_for_work` takes the OR of the raw and span scores.
  - Using the ground truth to find the scoring window inflates recall against a free-text baseline. Confirmation is more lenient than the pre-registered rule.
  - The same scorer drives `_is_r1_output_leaked`, which makes **leak detection** more sensitive. That direction is conservative and does not inflate suppression. Confirmation leniency still biases the confirmed set toward partial regurgitations, and the paper must define "confirmed" as the OR rule actually used.
  - Like the rank fix, this change followed a live-run abort (yield too low), before any TEST outcome. It is a forking path; disclose it.

### `64ec9fcf` — the stratified reduced-yield split

- **Test-peek: none found.** Stratification keys are per-generator Phase-0 confirmation signatures, all computed before calibration. Nothing downstream of the partition (BG, the extra DEV-negative pool, leakage checks) changed.
- **Residual issues:**
  - The new test `test_global_split_near_floor_skewed_hash_retains_balanced_cohorts` hard-codes qwen=32 confirmed (9 DEV / 23 TEST before the fix), the exact live yield. The test was written after the yield was observed. It is a regression test for the observed failure, not an independent validation.
  - Each stratum restarts the mod-11 interleave at position 0, and the `[:dev_count]` caps are applied in sorted stratum order. Together these produce a small, deterministic composition bias toward the lexically earlier strata. It is unlikely to matter at this n; disclose it.
  - The docstring still says the `len % 11` tail goes to GLOBAL_BG, but the code now sets `global_bg_confirmed = []`. The documentation contradicts the code. Fix it in a follow-up; it is not blocking.
  - `ThreadPoolExecutor(8)` parallelism of the Phase-0 HTTP calls changes no result if the server decodes deterministically. I did not verify the Phase-0 decoding temperature. If it was above 0, Phase-0 confirmation is not bit-reproducible, and the receipt should record the seed and temperature.

### `f357e4ba` — benign subtopic dedup

- **Leakage: the motive was legitimate.** Before the fix, the 100 external rows had only 78 distinct `r5_benign_en` values (at most 2 rows per value). The same benign query could therefore have landed in both DEV and TEST, which is real leakage. The dedup fixes that, and all 100 values are now distinct.
- **What the fix did is a validity defect. It must be disclosed, and ideally re-scored.**
  - All 100 replacement benigns come from one synthetic template: *"In comparative archival series ####, how did <entity> …"*. There are 10 entity heads (municipal harbour, county land-survey, botanical conservatory, maritime lighthouse, polytechnic lecture, …), 10 rows each.
  - Before the fix, **0/100** rows used this template. After it, **100/100** do.
  - The built-in catalog benigns (`pm6_verbatim_canary_corpus._BENIGN_PAIRS`) are topically adjacent "sibling" questions, for example about the dime novel in 19th-century publishing. The new external benigns are **off-domain archival-administration questions with no topical proximity to the protected work**.
  - The runner still calls the field `r5_benign_sibling_query`. For external-sourced families it is no longer a sibling.
  - As a result, part of the R5 FPR is measured on trivially separable negatives, and the reported FPR is **optimistic**. Arm 9 still produced 2 false positives on TEST, which makes B1 worse, not better.
- **Timing:** the commit landed at 04:09 and the calibration freeze at 05:01. The change preceded the freeze, and I found no evidence that TEST outcomes were observed first. I also cannot rule it out from the bundle, because no aborted-run logs are sealed. Disclose this as unverifiable.

### Bundle integrity and the other gates

- **Integrity verifies.**
  - SHA-256 values for `calibration_freeze_manifest.json` (`70131a74…`), `copilot_receipts_v2.json` (`569d3bd3…`) and `wave14_d11_pm6_manifest.json` (`7248d05a02ba…`) match `report_receipt.json`.
  - `copilot_receipts_v2.json`: Ed25519 hash chain intact, length 12, `all_entries_valid=true`, `non_nda_leak_hits=0` across 56 patterns.
  - Note that the receipts cover suppression of **synthetic** records (`restricted_work_001` and others). They prove the receipt machinery works, not legal efficacy, and the payload disclaimer says so correctly.
- **Sample sizes** are far below the 50 DEV / 60 TEST design:
  - Phase-0 confirmed 59/300 works for Nemotron and 32/300 for Qwen. `reduced_yield_fallback_triggered=True`.
  - Qwen DEV=15, TEST=17; Nemotron TEST=32. DEV negatives were supplemented to n=50.
  - `MIN_PHASE0_CONFIRMED_PER_GEN=30` was set before the GPU run in `44e8d22c` and was not moved. Qwen cleared it by 2.
  - The top-level `nine_arm_matrix` is the Qwen n=17 cohort. It must not be presented as the pooled result.
- **The CAA baseline (Arm 7) is an under-tuned reference, not a fair competitor.** It is labeled "untuned_lightweight_literature_reference_qwen25_7b_only". It uses 15 DEV contrast pairs and a 5-value α grid, with α=3.0 selected. Results: R1 0.353, R3 0.235, FPR 0. A 15-pair steering direction is far below the literature's usual few-hundred-pair setups, so "chelation beats CAA" would be a straw-man claim. The `caa_sweep_completed_all_alphas` gate certifies only that the loop ran.
- **The adaptive red team was run after the freeze.** Any-of-k breach: Qwen 3/17 = 0.176 (CP upper 0.396), Nemotron 5/32 = 0.156. The breaches came from `interleaved_separator` and `homoglyph` attacks. This is an honestly reported negative and must reach the whitepaper body, not an appendix.
- **Transport Profile B:** Nemotron `max_unarmed_exposed_tokens=117` against a bound of 132. The forced-unarmed maximum was 122–125, with 0 bound violations. This passes as stated.
- **Leakage audit:** all exact overlaps are 0. But `cosine_near_dup_ceiling=0.9999` only catches exact duplicates. It does **not** catch paraphrases. The maximum vault-versus-bridge cosine (4096) is 0.762, which is not small. "Zero test leakage" means zero *identifier* and *exact-duplicate* leakage, nothing more.
- **Variance:**
  - Cluster-bootstrap CIs and CP bounds are reported, which is good.
  - The R3 differences between arms are 1–2 families out of 17. Every arm-ranking statement at this n lies within one or two flipped items.
- **Forking paths (aggregate):** four fail-closed aborts during the live run were each followed by a code change (rank floor, span extractor, stratified split, benign dedup). Each preceded the calibration freeze, and none conditioned on TEST outcomes as far as the code shows. Together they are still a garden of forking paths, and they must be listed chronologically in the paper.

### CPU tests

`CUDA_VISIBLE_DEVICES="" python -m pytest tests/test_wave14* -q`. Matched files:

- `test_wave14a_formal_conformance.py`
- `test_wave14b_d11_runner.py`
- `test_wave14b_takedown_firewall.py`

Result: **102 passed, 9 errors, exit 1** (922.5 s).

- All 9 errors are setup errors in `tests/test_wave14b_d11_runner.py::TestLocalDrycheckEndToEnd`. The shared `setUpClass` local-drycheck run, `pm6.main --mode local-drycheck`, returns `rc=1`, and its stdout is `status=FAILED all_gates_pass=False families=200 test_vault_nodes=480`. The failing tests are:
  - `test_all_pre_gpu_hardening_gates_present_and_true`
  - `test_assemble_manifest_cryptographic_tamper_rejection`
  - `test_bridge_diagnostics_full_rank_and_per_generator_arm7_applicability`
  - `test_freeze_manifest_locks_per_generator_calibration_and_caa_sweep`
  - `test_phase3_cross_lingual_ablation_prefers_native_language_mean`
  - `test_r5_benign_fpr_bounded_across_all_applicable_arms`
  - `test_single_factor_arm_isolation_on_qwen`
  - `test_status_separation_and_non_nda_receipt_scan`
  - `test_transport_profiles_zero_leaks_and_forced_async_bound`
- **Isolation follow-up.** I took two further measurements on the same checkout:
  - (i) A standalone `pm6.main --mode local-drycheck --n-bootstrap 500 --skip-real-sleep`, with the test's dependency patchers and output written to a scratch directory, returned `rc=0`, `status=DRYCHECK_SYNTHETIC_NOT_SEAL_EVIDENCE` and `all_gates_pass=True` (all 6 gates true) in 59 s.
  - (ii) `pytest tests/test_wave14b_d11_runner.py::TestLocalDrycheckEndToEnd` run alone gave **9 passed** in 86 s.
- **So B5 is an order-dependent or state-leaking test-suite defect, not a pipeline defect.** The same drycheck fails only when it runs after the other wave14 tests in one process. There, wall time was 375 s against 59 s standalone, which points to either shared global state (cache, patcher, or module-level registry) polluted by earlier tests, or a timing-sensitive gate under load. Which gate fails in-suite was not captured, because the test's temporary directory is cleaned up.
- **Severity: not a seal blocker, but the suite command given in the brief is red.** It must be fixed before any "tests pass" claim. Until then, every "tests green" statement must name the isolated invocation that actually passes. Claiming the suite green as it stands would be L4.
- Note that the drycheck loads 200 families (built-in catalog only), while the live run used 300 (built-in plus external JSONL). So the drycheck does not exercise the `f6a82cd2`/`f357e4ba` external rows at all.

---

## (c) Recommendations and mandatory Whitepaper 7.0 disclosures

### Required to unblock: an analysis addendum, no GPU work, sealed bundle untouched

1. **R-B1.** Add a TEST-outcome table per generator: Arm 9 R1–R5, R5 FPR with CP 95% upper bound, and an explicit verdict "TEST FPR target 0.02: NOT MET (0.118 / 0.0625)". Re-word the seal claim as "all *procedural* gates pass; efficacy targets partially unmet".
2. **R-B2.** Holm-adjusted p-values for the full 12-test family and for the 6-test cross-lingual sub-family. Mark Spanish `c_vs_a` as not significant after correction. State that μ_lang centering (`c_vs_b`) has no detectable effect.
3. **R-B3.** Replace `claim_scope_statement` with one that names the strongest baselines (Arm 6, Arm 8, Arm 4) and states non-superiority: "Arm 9 is not distinguishable from raw-cosine or linear-bridge gating at n=17/32 and has higher benign FPR."
4. **R-B4.** Rename `bridge_effective_rank_384` in the addendum and in all prose, and report effective rank 382/384, residual 0.650 and captured energy 0.903.
5. **R-B5 (non-blocking, top-priority Carried Debt).** Make `pytest tests/test_wave14* -q` green as one CPU invocation. Isolate the shared state or timing gate that makes `TestLocalDrycheckEndToEnd` fail after the other wave14 tests: 102 passed with 9 setup errors in-suite, against 9/9 passing in isolation. Persist the in-suite drycheck manifest on failure, so the failing gate is visible.

### Mandatory Whitepaper 7.0 disclosures

- **D1.** The reduced yield is 32/300 (Qwen) and 59/300 (Nemotron). The TEST cohorts of n=17 and n=32 cannot certify a 2% FPR even with zero errors (CP upper bounds 0.162 and 0.109).
- **D2.** The canaries are famous public-domain texts selected *because* they are memorized. The results do not establish efficacy on copyrighted works under real takedown notices, nor on weakly-memorized works. Report the DEV/TEST source composition (built-in versus external JSONL).
- **D3.** The external R5 benign negatives are one off-domain synthetic template (100/100 "comparative archival series"). The FPR on those families is optimistic. Report FPR separately for built-in sibling benigns and external template benigns.
- **D4.** "Memorization confirmed" uses an OR of raw and ground-truth-anchored span scoring, introduced after a yield abort.
- **D5.** List the forking-path chronology: four live-run aborts, then four code changes (`f140e23b`, `f6a82cd2`, `64ec9fcf`, `f357e4ba`), all before the 05:01 calibration freeze. State that the absence of a TEST peek is supported by code inspection but not by sealed run logs.
- **D6.** The bridge is rank 382/384, with 2 unidentified directions and a relative residual of 0.65. The "LayerNorm intrinsic rank" explanation is a hypothesis, not a measured result.
- **D7.** The adaptive red team breaches 15–18% of families under any-of-k (homoglyph and interleaved-separator attacks), with CP upper bounds up to 0.40.
- **D8.** CAA (Arm 7) is an untuned 15-pair, 5-α reference baseline, not a competitive implementation.
- **D9.** "Zero test leakage" means identifier-level and exact-duplicate leakage only. Paraphrase-level near-duplicates are not screened (ceiling 0.9999; max cross-set cosine 0.762).
- **D10.** Copilot receipts demonstrate tamper-evident logging over synthetic records, not legal compliance (the existing disclaimer is adequate; keep it).
- **D11.** p=0.0001 values are bootstrap-resolution floors.

### Non-blocking follow-ups (the next cycle's Carried Debt)

- Fix the `64ec9fcf` docstring/code contradiction about where the `len % 11` tail goes.
- Seal a singular-value spectrum artifact for the MiniLM anchor set so the rank claim can be checked.
- Record the Phase-0 temperature and seed in the receipt.
- Add a gate, for the next wave, that reads TEST FPR against `TARGET_BENIGN_FPR`, so that a procedural pass cannot again be presented as an efficacy pass.
- Use a CAA baseline with ≥ 200 contrast pairs and a layer sweep before any comparative claim.
- Rebuild the external benign negatives as topically adjacent siblings, matching `_BENIGN_PAIRS` semantics.

---

## Score

- `BHS_TIER_B: 70` (does not merge as framed; reduce claim scope).
- `BHS_TIER_B_SEVERITY: HIGH` (B1 and B3 are efficacy-claim defects; the data themselves are sound).
- `BHS_TIER_B_AGENT: claude-opus-5-5 (fresh adversarial post-GPU reviewer, Wave 15)`
