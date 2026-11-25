# Wave 15 Final Judge Sign-Off — v7.0 PRE-RELEASE Seal (46/46 Bundles + `/formal`)

- **Judge:** Claude Fable 5.2 (medium effort), independent Final Judge #2, `$panel-of-experts` + Brutal-Honesty Rulebook v3.3
- **Date:** 2026-11-25
- **Checkout:** `D:\GITHUB\CHELATEDAI-STAGE0-NIGHT1`, branch `fix/stage0-metric-lineage-and-m1-ladder`, HEAD `c8ea34f26c43e2d84bf0b5bd4727c340f6dd6b5a`
- **Scope kept:** read-only; no GPU job (`CUDA_VISIBLE_DEVICES=""` on every Python call); no `*sealed_key*` / `*blind-followup*` file opened; `pm4_human_audit_100_blind_sheet.csv` untouched; no existing file edited; no commit. This memo is the only file written.

## VERDICT: `APPROVED_FOR_V7_SEAL`

Approved as a **PRE-RELEASE** seal only. The approval rests on two conditions I could not satisfy myself:

1. **Full-formal tier is unattested by me.** I ran `--fast`, which reports `formal_tier=SKIPPED_USE_FULL_FORMAL`. The whitepaper (line 429) claims `--full-formal` gives `formal_tier=PASS` with `0` skipped checks. That claim stands or falls on Judge #1's `lake build` / nanoda run. If Judge #1 does not report a clean full-formal PASS, this approval is void.
2. **Finding F1 below should be fixed before the investor deck is sent to anyone outside.** It does not change any sealed artifact or registry hash input that I checked, so I do not hold the seal for it.

BHS (my independent score for the v7.0 framing as committed): **92**. The primary documents are honest; the deductions are F1–F3. This is a judge's score on the seal framing, not a PR Tier-B score.

## 1. Runtime evidence (commands I ran)

| Gate | Result |
|---|---|
| `python scripts/verify_non_nda_ip_clean.py` | exit 0. `ZERO_IP_LEAK_AUDIT: PASS (0/56 forbidden method patterns matched across 2 file(s))` |
| `python scripts/build_whitepaper_assets.py --check` | exit 0. `WHITEPAPER_ASSETS_CHECK: PASS (bundles=46/46, side_hashes=61, unhashed_hist_raw=20, pristine=15/15, registry_sha=6ef7eba4f414)` |
| `python scripts/verify_all_offline.py --fast` | exit 0. `PASS mode=fast formal_tier=SKIPPED_USE_FULL_FORMAL wall=23.1s`. Skipped: `lean_lake_build`, `print_axioms_allowlist`, `nanoda_external_kernel`, `ek5_1000_mutation_conformance` |
| `python -m unittest tests.test_verify_all_offline` | `Ran 71 tests in 21.326s OK` |
| `sha256sum formal/ChelatedFormal/Spec.lean` | `63fe3e644bdef93160c5d612bfc7d7086af03c3d15cf34417402a89a86cdc6eb`, equal to `formal/spec_sha256.json` (`locked_before_proofs: true`) |
| D11 artifact hashes recomputed by hand | manifest `7248d05a02ba4901aace9fd960c2a4d04d0046f83bcd1846426159e5cab9ffc8` and the two sibling hashes match `report_receipt.json` |
| `git status --short` after all gates | only untracked paths that existed before I started; no tracked file modified |

Relevant `--fast` lines: `bundle.wave14_d11_pm6_takedown_carryforward: manifest 7248d05a02ba ok`; `d11.ed25519_and_hash_chain` (12 receipts); `metrics.d11` (6 gates process-only, `r5_benign_fpr_le_002` recomputed and disclosed FAILED, 12-test Holm recomputed, D1–D11 bound to the manifest sha); `d11.reanalysis_recompute` byte-identical to a fresh recomputation; `formal.spec_sha256_lock`; `registry.content_sha256 6ef7eba4f414`; `registry.matches_spec` (46 bundles); `release_status.honesty` PRE-RELEASE retained; `r5b_s4_pm4_cross_family_2x2` `sealed_unread=1`.

Inspected, not executed: `formal/conformance_report.json` (`ok: true`; 1,000 sequences; 7,722 checkpoints; 1,531/1,531 faults caught; `inc_vs_fresh` 153, `lean_vs_inc` 153, `lean_vs_fresh` 0, semantic 0; `incremental_with_deletions_status = conformance_tested_unproven_in_lean`), `formal/nanoda_manifest.json` (nanoda 0.4.32, 27 export roots, axioms `propext` / `Classical.choice` / `Quot.sound`). A source grep finds 68 `theorem`/`lemma` declarations under `formal/ChelatedFormal`.

## 2. Reconciliation of the two peer memos

### Opus 5.5 (BLOCK as framed, convertible via post-hoc addendum)

| Item | Status | Evidence |
|---|---|---|
| B1 process gates presented against TEST FPR | CLOSED | Reanalysis JSON classifies the six gates `PROCESS_ONLY`; `r5_benign_fpr_le_002` verdict FAILED, 2/32 = 6.25% (CP upper 0.1839) and 2/17 = 11.76% (CP upper 0.3262); min N = 149. Whitepaper lines 349–353 and the presentation say the same. |
| B2 no multiplicity control | CLOSED | 12-test Holm family, 3 survivors. I re-derived the adjusted values by hand: 12 × 0.0001 = 0.0012; 10 × 0.0029 = 0.029; 9 × 0.0163 = 0.1467 (ES fails). Raw p-values and AUCs match the sealed manifest. |
| B3 Arm 9 weakly dominated by Arms 6 / 8 | CLOSED in whitepaper and presentation; OPEN in investor deck (F1) | Arm 9 vs Arm 6: −0.0107 (p 0.1543), −0.0208 (p 0.1767). Vs Arm 8: −0.0039 (p 0.7311), −0.0138 (p 0.3921). R4 = 100% attributed to the shared Gate-1 path of arms 4/6/8/9. |
| B4 `bridge_effective_rank_384` misnomer | CLOSED | Reanalysis reports rank 382/384, vacuous orthogonality, LayerNorm explanation marked `HYPOTHESIS_NOT_VERIFIED`. |
| D1–D11 | CLOSED | All eleven in the reanalysis JSON, bound to the manifest sha by the verifier. |
| B5 order-dependent in-suite test errors (non-blocking) | NOT RE-VERIFIED | I did not run `tests/test_wave14*`. The whitepaper says it was addressed; I have no evidence either way. |

### Neon 4 (CONDITIONAL GO / HOLD MERGE)

| Item | Status | Evidence |
|---|---|---|
| 1. v6.0 sealed first | CLOSED | Per the commit chronology recorded in the reanalysis and the diff base `7aefda2f`. |
| 2. Verifier D11 blind spot | CLOSED | `metrics.d11`, `d11.ed25519_and_hash_chain`, `d11.reanalysis_recompute`, `formal.spec_sha256_lock` all execute and pass; 71 unit tests pass. |
| 3. Spec / registry bumped to v7.0 46/46 | CLOSED | `bundle_registry_spec_v7_0`, `verified_registry_v7_0`, content sha `6ef7eba4f414…`, 61 side hashes. |
| 4. Section-c disclosures in v7.0 docs | CLOSED for every item in the Final Judge brief; PARTIAL for Neon 4's finer formal items (F3) | See F3. |
| MemT-1.0 never "retracted" | CLOSED | Whitepaper line 374 cites `arXiv:2610.26825v5` as an active preprint, not retracted, with the superposed-state reason and `_state_identity()` omitting weight hashes. A grep of `docs`, `scripts` and root markdown finds no "retracted MemT" phrasing outside Neon 4's own quotation of the old error. |
| Hosted gateway only, no unlearning | CLOSED | Manifest `governance_header`: `unlearning_claimed=false`, `parametric_recall_neutering_claimed=false`, `hosted_gateway_scope_only=true`. |

### Required-framing check, by document

| Framing | Whitepaper | Presentation | Teaser (md) | Deck (html) |
|---|---|---|---|---|
| PRE-RELEASE, PM-4b pending, PM-1b `assignment_gates_pass=false`, zero Ship-Ready | yes | yes | yes | yes (PRE-RELEASE, 46/46) |
| D11 empirically mixed, FPR fail 6.25% / 11.76% | yes | yes | yes | yes |
| N>=149 floor, 3/12 Holm, ES 0.1467, mu_lang null | yes | yes | no | no |
| English R3 parity/loss vs Arm 6 and Arm 8 | yes | yes | soft ("matching simpler vector baselines") | **no** |
| R4 = 100% shared across arms 4/6/8/9 | yes | not checked literally | no | no |
| R2b breach 15.63% / 17.65% | yes | yes | yes | yes |
| Hosted gateway only, no weight erasure | yes | yes | yes | yes |
| MemT-1.0 active preprint, `_state_identity()` | yes | active preprint yes; `_state_identity` no | not cited | not cited |
| `/formal`: 68 theorems, 0 sorry/native_decide, nanoda 0.4.32 / 25,473 decls, 7,722 checkpoints / 153 diffs, incremental-deletions conformance-tested | yes | yes | 68 proofs + external kernel, **no conformance caveat** | 68-theorem suite, no caveat |

## 3. Findings

**F1 (medium, fix before external distribution of the deck) — selective baseline in the investor deck.** `external-investor-teaser-deck-non-nda.html` reports PM-6 against string filters only (+47.4 to +47.6 AUROC points) and omits that the simpler vector baselines (raw cosine, linear bridge) match or beat Arm 9 on English paraphrase detection. This is Opus B3 surviving in the one document most likely to be read alone. The md teaser includes the clause, but its "matching" is generous: Arm 9 is numerically lower on both generators, not significantly. The deck does disclose the FPR miss, the red-team bypass and the hosted-gateway scope, which is why I rate this medium rather than blocking. One added clause fixes it.

**F2 (medium, unverified claim) — "68 machine-checked formal proofs … verified by an independent external proof kernel" in the teaser.** The nanoda manifest lists 27 export roots, one of them the umbrella `all_wave14a_formal_theorems_verified`. Whether the 25,473-declaration export closure of those roots contains all 68 theorems is something I did not establish, and Neon 4 reported 3 lemmas outside the SHA-locked umbrella. All 68 are checked by the Lean kernel on `lake build` (Judge #1's tier); the external-kernel claim for all 68 is unproven by my review. The same sentence also carries no caveat that incremental maintenance under deletions is conformance-tested rather than proved. Suggested wording: "68 Lean-checked theorems; core statements re-checked by an independent external kernel; incremental deletion handling is conformance-tested, not proved."

**F3 (low) — finer formal-scope disclosures from Neon 4 are not in the whitepaper.** The whitepaper's mandatory `/formal` disclosure sentence (line 349) covers (a) incremental-with-deletions and (b) the 153 diagnostic diffs. It does not state that Theorem 4 covers hard-projection rho-ball invariance rather than EMA convergence (0 hits for "hard projection"), the CSLS 2x scale convention (0 hits), that only Stratum 2 is kernel-bound while Strata 0/1/3 are conformance-only, or which lemmas sit outside the Spec lock. None of these were in the Final Judge brief's required list, so they do not gate the seal, but a reader could take Theorem 4 for more than it proves.

**F4 (low, pre-existing) — teaser line 37 "Enforces synchronous right-to-erasure".** Unchanged from v6.0 (not in the `7aefda2f..HEAD` diff). It describes retrieval-layer suppression. Next to a v7.0 scope line that says the system does not modify model weights, "right-to-erasure" reads stronger than the evidence. Suggest "retrieval-layer suppression".

**F5 (low) — presentation omits `_state_identity()` and the literal `D1–D11` reference.** It does say MemT-1.0 is an active preprint and cited accurately, so the substantive requirement is met.

**F6 (informational) — manifest still carries `all_gates_pass=True` and a `claim_scope_statement` of "Arm 9 exceeds Arm 3 on R1/R2 beyond pre-registered tolerance alongside R3/R4 and carry-forward."** The sealed manifest cannot be changed, and the reanalysis addendum plus verifier correctly reclassify the gates as process-only. Anyone quoting the manifest alone would get the pre-remediation picture; the addendum must travel with it.

## 4. Not verified by me

- Full-formal tier: `lake build`, `#print axioms` allowlist, nanoda external kernel run, EK5 1,000-mutation conformance re-run. Left to Judge #1 to avoid `.lake/build` lock contention.
- Wave-14 test suites other than `tests.test_verify_all_offline` (Opus B5).
- Whether the nanoda export closure covers all 68 theorems (F2).
- D11 check code in `scripts/verify_all_offline.py` was judged by its output and its 71 passing unit tests, not read line by line.
- PM-4b human audit and PM-1b clean rerun remain pending by design; nothing here speaks to them.

## 5. Sign-off

The v7.0 suite keeps PRE-RELEASE status, makes no Ship-Ready claim, reports D11/PM-6 as empirically mixed with the FPR failure and the Holm result stated, cites MemT-1.0 as an active preprint, and carries the required `/formal` disclosures in the whitepaper and presentation. The registry (`6ef7eba4f414…`, 46/46, 61/61) and the D11 reanalysis reproduce from the committed artifacts on CPU.

`APPROVED_FOR_V7_SEAL` — PRE-RELEASE only, conditional on Judge #1's full-formal PASS, with F1 and F2 to be corrected before the Non-NDA investor files leave the building.

— Final Judge #2, Claude Fable 5.2 (medium)
