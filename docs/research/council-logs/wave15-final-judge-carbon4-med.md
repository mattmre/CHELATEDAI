# Wave 15 Final Judge Sign-Off Memo (`v7.0` Pre-Release Seal, `46/46` Bundles + NDA `/formal` Lean 4)

- **`panel_type`:** `brutal-honesty-v3.3-final-adjudication` (`$panel-of-experts`)
- **`artifact`:** `docs/research/{bundle_registry_spec.json,verified_registry.json,doctoral-whitepaper-chelated-memory-and-spherical-migration-2026-11-21.md,chelated-memory-spherical-migration-whitepaper-presentation.html,external-investor-teaser-non-nda.md,external-investor-teaser-deck-non-nda.html}`, `artifacts/wave14_d11_pm6_takedown_carryforward/pm6_posthoc_statistical_reanalysis.json`, `scripts/{compute_wave14b_pm6_posthoc_reanalysis.py,verify_all_offline.py,build_whitepaper_assets.py}`, `formal/`
- **`run_date`:** `2026-11-25`
- **`run_id`:** `wave15-final-judge-carbon4-med`
- **`challenge_depth`:** `1`
- **`active_experts`:** `["gemini-4-carbon (Medium — Independent Final Judge; Statistical Inference, Cryptographic Provenance, Formal Mechanization & Brutal-Honesty v3.3 Governance)"]`
- **`persist_version`:** `1.1.0`
- **`repo_worktree`:** `D:\GITHUB\CHELATEDAI-STAGE0-NIGHT1`
- **`branch`:** `fix/stage0-metric-lineage-and-m1-ladder`
- **`repo_head_audited`:** `c8ea34f26c43e2d84bf0b5bd4727c340f6dd6b5a` (`Merge: 7aefda2f eddec2a5`)
- **`hard_safety_compliance`:** `D:\GITHUB\CHELATEDAI` never touched; zero GPU jobs executed (`CUDA_VISIBLE_DEVICES=""` on all Python invocations); zero `*sealed_key*` or `*blind-followup*` files opened; `pm4_human_audit_100_blind_sheet.csv` untouched; zero edits to existing repository files; no `git commit`; sole file written is this LF-only (`\n`, `0` CR) memo.

---

## Final Verdict

**`APPROVED_FOR_V7_SEAL`**

*(Scoped strictly to the **PRE-RELEASE** `Draft v7.0` seal at commit `c8ea34f26c43e2d84bf0b5bd4727c340f6dd6b5a`: `release_status = "PRE-RELEASE / HUMAN AUDIT & CLEAN-RERUN GATES PENDING"` and `PM-1b` `assignment_gates_pass = false` remain mandatory and active while `PM-4b` independent `100`-item human grading is pending and `Demo D11 / PM-6` remains **EMPIRICALLY MIXED** due to held-out `R5` benign FPR `6.25%` / `11.76% > 2.0%` target, English `R3` parity/loss vs. `Arm 6` / `Arm 8`, and `15.63%` / `17.65%` `R2b` adaptive red-team breach rates. Zero `Ship-Ready` claims exist anywhere in the suite.)*

---

## 1. Reconciliation of Wave-15 Post-GPU Adversarial Peer Audits (`Opus 5.5 High` & `Neon 4 High`)

Both post-GPU adversarial peer-audit memos (`docs/research/council-logs/wave15-postgpu-xval-opus55-high.md` and `docs/research/council-logs/wave15-postgpu-xval-neon4-high.md`) were audited item-by-item against commits `7aefda2f` (`v6.0` seal), `eddec2a5` (Wave-15 remediation), and `c8ea34f2` (`v7.0` merge + publication synthesis):

### 1.1 Closure of `Opus 5.5 High` Blockers (`B1–B4`), CPU Suite Issue (`B5`), & Disclosures (`D1–D11`)

| Peer Finding | Required Remediation | Verified On-Disk Resolution at `c8ea34f2` | Verdict |
|---|---|---|---|
| **`B1` (`L13`): `all_gates_pass=True` is procedural, not empirical efficacy; TEST `R5` benign FPR misses `<= 2.0%`** | Keep sealed manifest untouched; add CPU post-hoc addendum classifying the 6 gates as process-only and reporting TEST `R5` FPR failure + CP95 bounds across all `v7.0` surfaces. | `artifacts/wave14_d11_pm6_takedown_carryforward/pm6_posthoc_statistical_reanalysis.json` (`schema_version = "wave15_pm6_posthoc_statistical_reanalysis_v1"`, `source_manifest_sha256 = "7248d05a02ba4901aace9fd960c2a4d04d0046f83bcd1846426159e5cab9ffc8"`, `sealed_files_mutated = false`, `gpu_jobs_executed = 0`) classifies all 6 manifest gates as `"PROCESS_ONLY_GATE"` (`:1047-1080`) and records `empirical_test_targets.r5_benign_fpr_le_002.verdict = "FAILED"` (`passed_all_generators = false`, `:1008-1045`): `nemotron_30b = 2/32 = 0.0625` (one-sided CP95 `<= 0.183943`, 0-error CP95 `= 0.089368`) and `qwen25_7b = 2/17 = 0.117647` (one-sided CP95 `<= 0.326193`, 0-error CP95 `= 0.161566`), with `min_n_to_certify_target_at_zero_errors = 149`. *(Independent check: Opus `:67,227` hand-computed `0.109` at `n=32` by mixing two-sided 95% `1 - 0.025^(1/32) = 0.1089` with one-sided 95% `1 - 0.05^(1/17) = 0.1616`; `compute_wave14b_pm6_posthoc_reanalysis.py:91-114` uses exact consistent one-sided 95% Clopper–Pearson bounds throughout.)* | **CLOSED** |
| **`B2`: Holm–Bonferroni step-down absent; `ES` `c_vs_a` (`p=0.0163`) fails Holm; `mu_lang` vs `mu_en` (`c_vs_b`) null; `1e-4` bootstrap floor** | Compute exact Holm step-down for the 12-test family and both 6-test sub-families; disclose `ES` non-survival, `mu_lang = mu_en` null, and `1/10,000` floor. | `pm6_posthoc_statistical_reanalysis.json:417-851` computes exact step-down (`familywise_alpha = 0.05`, monotone running-max `holm_adjusted_p`): **3/12 survive** — `nemotron_30b:arm9_vs_arm3` (`+0.473633`, raw `p=0.0001` floor, `Holm p=0.0012`), `qwen25_7b:arm9_vs_arm3` (`+0.475779`, raw `p=0.0001` floor, `Holm p=0.0012`), and `phase3:de:paired_bootstrap_c_vs_a` (`+0.152249`, raw `p=0.0029`, `Holm p=0.0290` full / `0.0174` 6-test sub-family). **9/12 fail**, led by `phase3:es:paired_bootstrap_c_vs_a` (`+0.100346`, raw `p=0.0163 > 0.005556` rank-4 threshold, `Holm p=0.1467` full / `0.0815` sub-family); `c_vs_b` (`mu_lang` vs `mu_en`) is explicitly marked `"NULL"` (`DE p=0.8667`, `ES p=0.6093`). Independently recomputed in `scripts/verify_all_offline.py:775-798,858-878`. | **CLOSED** |
| **`B3` (`L4`): `Arm 9` weakly dominated on English `R3` / `R5` FPR by `Arm 6` (Raw Cosine) & `Arm 8` (Linear Bridge); `R4=100%` comes from Gate-1 `EK5`** | Tabulate full 9-arm deltas; state English parity/loss vs `Arm 6`/`Arm 8` with higher FPR; attribute `R4=100%` to shared Gate-1 `EK5` (`Arms 4, 6, 8, 9`). | `pm6_posthoc_statistical_reanalysis.json:30-415` tabulates all 9 arms and `arm9_deltas` per generator: `Arm 9` trails `Arm 6` (`-0.010742` Nemotron `p=0.1543`, `-0.020761` Qwen `p=0.1767`) and `Arm 8` (`-0.003906` Nemotron `p=0.7311`, `-0.013841` Qwen `p=0.3921`) on English `R3-vs-R5` AUROC, trails `Arm 6` on `R3` (`93.75%` vs `100%`; `88.24%` vs `100%`), and has higher `R5` FPR than both (`6.25%/11.76%` vs `3.13%/5.88%` `Arm 6` and `0.00%/0.00%` `Arm 8`). `r4_attribution_note` (`:37`) and `arms_with_r4_eq_1 = ["arm4", "arm6", "arm8", "arm9"]` (`:223,406`) isolate `100%` (`60/60`) `R4` suppression to Gate-1 `EK5` lineage (vs `0%–5%` without Gate-1). Disclosed verbatim in Whitepaper `v7.0:4,352,410`, HTML `v7.0:402,1177,1236`, Spec `:457`, and Teaser `:42`. | **CLOSED** |
| **`B4` (`L13`): `bridge_effective_rank_384` checks `>= 380`; live rank `382/384`; `0.6501` residual; vacuous thin-SVD orthogonality** | Disclose `>= 380` actual floor, `382/384` rank (`2` unidentified directions), `0.650082` residual (`0.902834` captured energy), vacuous SVD orthogonality, and unverified LayerNorm hypothesis. | Recorded in `pm6_posthoc_statistical_reanalysis.json:3-29`, enforced fail-closed in `verify_all_offline.py:879-886` (`test_bridge_audit_must_disclose_380_floor`), and disclosed in Whitepaper `v7.0:351,353`, HTML `v7.0:1177`, and Spec `:457`. | **CLOSED** |
| **`B5` (Non-blocking): In-suite `TestLocalDrycheckEndToEnd` CAA sweep timebox failure under full CPU load** | Fix shared/timing sensitivity in drycheck CAA sweep without weakening gates. | Commit `eddec2a5` updated `scripts/run_wave14b_d11_pm6_takedown_carryforward.py` and `tests/test_wave14b_d11_runner.py:1642-1647` to size the drycheck CAA sweep timebox dynamically from measured warmup cost (`[DRYCHECK_CAA_TIMEBOX_FLOOR_S, DRYCHECK_CAA_TIMEBOX_CAP_S]`) rather than truncating mid-sweep under full-process CPU load. | **CLOSED** |
| **`D1–D11` Mandatory Disclosures** | Carry all 11 methodological, corpus, chronology, geometry, red-team, leakage, receipt, and p-floor disclosures into `v7.0`. | All 11 disclosures (`D1`..`D11`) and the 6-step pre-freeze chronology (`f140e23b -> f6a82cd2 -> 64ec9fcf -> f357e4ba -> 05:01 freeze -> 60ffc1f7`) are bound in `pm6_posthoc_statistical_reanalysis.json:852-970`, enforced by `verify_all_offline.py:888-892`, and written into Whitepaper `v7.0:350-353,429,494` and HTML `v7.0:402,1177,1236`. | **CLOSED** |

### 1.2 Closure of `Neon 4 High` Merge, Verifier, Registry, `/formal`, & `MemT-1.0` Requirements

1. **Ordered Merge Sequencing (`v6.0` Sealed Before `v7.0` Merge):**
   - Verified in git history: `7aefda2f` sealed `v6.0` (`45/45` bundles, `01738dbf94e6`, preserved byte-exact in `archive/whitepaper-v6.0-pre-v7-backup/`) on `fix/stage0-metric-lineage-and-m1-ladder` before `c8ea34f2` merged `eddec2a5` (`feat/v7-formal-pm6` + Wave-15 remediation) cleanly with zero conflicts and preserved `check_d7_headlines`, `check_d9_headlines`, and `check_d10_headlines` (`scripts/verify_all_offline.py:1174-1254`).
2. **Offline Verifier `D11` & `/formal` Lock Gates (`scripts/verify_all_offline.py` & `tests/test_verify_all_offline.py`):**
   - `PART7_OPTIONAL_PREFIXES` (`:112`) now includes `"wave14_d11_"`, and `wave14_d11_pm6_takedown_carryforward` is registered as required Bundle `#46` in `docs/research/bundle_registry_spec.json:450-458`.
   - `d11.ed25519_and_hash_chain` (`:668-772, 931-952`) recomputes all `12/12` `copilot_receipt_v2` `entry_hash` values from `prev_hash || canonical_json(payload)`, verifies Ed25519 signatures, `seq`/`prev_hash` chain linkage, `head_hash == b214649ccadd14bf8ea680a5efbef7d40478bd8f46c94c4084a94812645cc263` across ledger, `report_receipt.json`, and `wave14_d11_pm6_manifest.json`, enforces integer-only payloads (`_contains_float == False`), and rescans all entries against the 56 Non-NDA forbidden IP patterns (`0` hits).
   - `metrics.d11` (`:801-895, 954-970`) and `d11.reanalysis_recompute` (`:972-990`) recompute `r5_benign_fpr_le_002` (`verdict == "FAILED"`), the 12-test Holm step-down, bridge rank floor `380`, `D1–D11` disclosures, pre-freeze commit order, governance flags (`unlearning_claimed == False`, `parametric_recall_neutering_claimed == False`), and verify that on-disk `pm6_posthoc_statistical_reanalysis.json` is byte-identical to a fresh CPU recomputation (`scripts/compute_wave14b_pm6_posthoc_reanalysis.py`).
   - `formal.spec_sha256_lock` (`:997-1050`) verifies `sha256(formal/ChelatedFormal/Spec.lean) == 63fe3e644bdef93160c5d612bfc7d7086af03c3d15cf34417402a89a86cdc6eb`, `locked_before_proofs is True`, and `formal/conformance_report.json["incremental_with_deletions_status"] == "conformance_tested_unproven_in_lean"`.
   - `tests/test_verify_all_offline.py` (`71/71` tests passing) covers positive execution plus negative tamper mutations for signatures, hash chains, float injection, flipped FPR verdict, promoted Holm non-survivors, missing disclosures, altered `Spec.lean`, and altered conformance status.
3. **`MemT-1.0` (`arXiv:2610.26825v5`) Factual & Architectural Framing:**
   - Verified across Whitepaper `v7.0` (`:18, 374, 494`), Presentation HTML `v7.0` (`:411, 1177`), `bundle_registry_spec.json` (`:457`), and `docs/research/WAVES_1_TO_21_MASTER_EVIDENCE_AND_PR_INDEX.md` (`:68-69`): `MemT-1.0` (`arXiv:2610.26825v5`) is cited accurately as an **active recurrent-state memory preprint (explicitly noted as NOT retracted)** whose superposed hidden state precludes selective post-hoc retraction/clawback without external lineage gating and whose `_state_identity()` (`modeling_memt.py:876`) omits model weight SHA-256 digests.
4. **NDA `/formal` Lean 4 Scope & Conformance Disclosures:**
   - Verified across Whitepaper `v7.0` (`:4, 18, 349, 410, 429`), HTML `v7.0` (`:400, 1177, 1236, 1248`), `formal/ChelatedFormal/Statement.lean` (`:28-57`), and `formal/conformance_report.json` (`:217-297`): `68` Lean 4 theorems on `{propext, Classical.choice, Quot.sound}`, `0` `sorry`/`native_decide`, `nanoda_bin 0.4.32` (`rev 0da9f98f`) checking `25,473` declarations (`>= 27` exported roots, `0` errors), `7,722` conformance checkpoints (`6,000` mutation + `1,722` `as_of` incl. `951` missing-parent; `1,531/1,531` injected faults caught; `0` semantic diffs on `eff_auth`/`val`/`cur`/`eligibility`/`node_set`), `153` non-semantic Pitfall-B pruning telemetry diffs (`104` `pos_cyc` + `49` `reason` on `inc_vs_fresh` and `153` on `lean_vs_inc`), and `incremental_with_deletions_status = conformance_tested_unproven_in_lean`.

---

## 2. Publication & Non-NDA Investor Surface Audit (`v7.0`)

| Target File | Key `v7.0` Governance, Metric, & Honesty Checks | Verified Status |
|---|---|---|
| `docs/research/bundle_registry_spec.json` | `schema_version = "bundle_registry_spec_v7_0"`; `release_status = "PRE-RELEASE / HUMAN AUDIT & CLEAN-RERUN GATES PENDING"`; `46` bundles (`#46` `wave14_d11_pm6_takedown_carryforward` in `PART_VIII_FORMAL_AND_PM6_TAKEDOWN` with full mixed-verdict, Holm `3/12`, `6.25%/11.76%` FPR FAIL, `N>=149`, `153` Pitfall-B diffs, `D1–D11`, and active `MemT-1.0` `arXiv:2610.26825v5` notes). | **PASS** |
| `docs/research/verified_registry.json` | `schema_version = "verified_registry_v7_0"`; `registry_content_sha256 = 6ef7eba4f4147de1ea7e51f0ee5ade00711227f3e2a8d4b0c77e5ba7c0511542`; `46/46` bundles `SHA256_VERIFIED`; `61/61` receipt-bound side hashes (`2+6+4+13+16+8+10+2`); `20` unhashed historical Part I raw files inventoried; `15/15` pristine mirror byte-exact; `observations.wave14_d11_pm6_takedown_carryforward` binds `reanalysis_sha256` and `FAILED` `r5_benign_fpr_le_002`. | **PASS** |
| `docs/research/doctoral-whitepaper-chelated-memory-and-spherical-migration-2026-11-21.md` | `Draft v7.0`; `PRE-RELEASE` banner at `:3-4` (`0` `Ship-Ready`); `PM-4b` human sheet pending & `PM-1b` `assignment_gates_pass = false`; `D11 / PM-6` framed as **EMPIRICALLY MIXED** (`:4, 350-353, 410, 480`); `MemT-1.0` (`arXiv:2610.26825v5`) active/non-retracted at `:18, 374`; `/formal` Lean 4 + `nanoda_bin` + `153` Pitfall-B + `conformance_tested_unproven_in_lean` at `:4, 349, 410`; all `46/46` manifest SHA prefixes & `6ef7eba4f414` present. | **PASS** |
| `docs/research/chelated-memory-spherical-migration-whitepaper-presentation.html` | `v7.0`; print-visible `.release-status-banner` (`PRE-RELEASE`, `0` `Ship-Ready`, `:396-406`); `PM-4b` & `PM-1b` open; `D11 / PM-6` **empirically mixed** (`R5 FPR 6.25%/11.76% > 2% FAIL`, `Arm 6/8` EN `R3` parity/loss, `ES` Holm fail `0.1467`, `mu_lang=mu_en` null, `R2b` `15.6%/17.6%`, `382/384` rank & `0.650` residual, hosted gateway only, `:402, 1177, 1236, 1294`); active `MemT-1.0` (`:411, 1177`); WCAG AAA `--warm: #7d2f0a`, MathML, ARIA `tablist`, `12` inline SVGs, all `46/46` SHA prefixes & `6ef7eba4f414`. | **PASS** |
| `docs/research/external-investor-teaser-non-nda.md` | `PRE-RELEASE` diligence banner (`:3, 6-7`, `0` `Ship-Ready`); `46/46` bundles + `68` formal proofs; explicitly discloses `PM-4b` human audit pending, `PM-1b` (`assignment_gates_pass = false`), and `PM-6` missing `<= 2.0%` benign false-refusal target at `6.25%–11.76%`, matching simpler vector baselines on English paraphrase detection, `15.6%–17.6%` adaptive red-team bypass, and hosted-gateway-only scope (`:3, 42`); **`0/56` Non-NDA forbidden IP pattern hits**. | **PASS** |
| `docs/research/external-investor-teaser-deck-non-nda.html` | `PRE-RELEASE` header & Slide 1/4 disclosures (`:144-145, 152, 278-279`, `0` `Ship-Ready`); `46/46` bundles + `68` formal proofs; explicitly discloses `PM-4b`, `PM-1b`, and `PM-6` (`6.25%–11.76% > 2.0%` benign false-refusal miss, `15.6%–17.6%` adaptive red-team bypass, hosted gateway only, plus `D9`/`D10` latency & `Quora -6.92 pts` loss); **`0/56` Non-NDA forbidden IP pattern hits**. | **PASS** |

---

## 3. Executed Read-Only CPU Gate Receipts (`CUDA_VISIBLE_DEVICES=""`)

All required gates and supplementary deterministic checks were executed inside `D:\GITHUB\CHELATEDAI-STAGE0-NIGHT1` at `HEAD = c8ea34f26c43e2d84bf0b5bd4727c340f6dd6b5a` with `CUDA_VISIBLE_DEVICES=""`:

1. **`python scripts/verify_non_nda_ip_clean.py`**
   - **Output:** `ZERO_IP_LEAK_AUDIT: PASS (0/56 forbidden method patterns matched across 2 file(s))`
   - **Exit Code:** `0`
2. **`python scripts/build_whitepaper_assets.py --check`**
   - **Output:** `WHITEPAPER_ASSETS_CHECK: PASS (bundles=46/46, side_hashes=61, unhashed_hist_raw=20, pristine=15/15, registry_sha=6ef7eba4f414)`
   - **Exit Code:** `0`
3. **`python scripts/verify_all_offline.py --full-formal`**
   - **Output:** `verify_all_offline: PASS mode=full-formal formal_tier=PASS wall=75.9s cpu=0.781s platform=Windows-10-10.0.26300-SP0 python=3.11.9 skipped_checks=none`
   - **Exit Code:** `0`
   - **Key Verified Gates (`68/68` report rows PASS, `0` FAIL, `0` SKIP/WARN/NOT_RUN):**
     - `pristine_mirror` (`15/15` byte-exact) & `bundle.*` (`46/46` PASS; `r5b_s4` `sealed_unread=1` never opened)
     - `d8.ed25519_and_hash_chain`, `d8.sealed_file_hashes`, `d8.live_replay` (`3` receipts, head `820705ef47fc`)
     - `d11.ed25519_and_hash_chain` (`12` payload-recomputed SHA-256 + Ed25519 receipts, `0` floats, `0/56` Non-NDA rescan hits), `metrics.d11` (`r5_benign_fpr_le_002` recomputed `FAILED`, `12`-test Holm recomputed, `D1–D11` bound), `d11.reanalysis_recompute` (byte-identical to live CPU recomputation)
     - `formal.spec_sha256_lock` (`Spec.lean` `63fe3e644bde` + `conformance_tested_unproven_in_lean`)
     - `non_nda.*` (`0/56` across teaser `.md`, teaser `.html`, and `compliance_copilot.html`)
     - `metrics.d6`, `metrics.d8`, `metrics.d7`, `metrics.d9`, `metrics.d10` (`5/5` PASS)
     - `registry.content_sha256` (`6ef7eba4f414`), `registry.matches_spec` (`46` bundles), `release_status.honesty` (`PRE-RELEASE` retained)
     - `formal.lake_build` (`PASS`), `formal.no_sorry_native_decide` (`0` across `9` `.lean` files), `formal.print_axioms_allowlist` (`68` theorems on `['Classical.choice', 'Quot.sound', 'propext']`), `formal.negative_control` (`3/3` rejected), `formal.lean4export_prepare` (`PASS`), `formal.nanoda_external_kernel` (`25,473` declarations checked, `>= 27` exported, `0` errors), `formal.ek5_1000_mutation_conformance` (`PASS`).
4. **Supplementary Read-Only CPU Checks:**
   - `python scripts/compute_wave14b_pm6_posthoc_reanalysis.py --check` -> `OK: ... matches recomputation` (`exit_code = 0`)
   - `python -m pytest tests/test_verify_all_offline.py -q` -> `71 passed in 23.55s` (`exit_code = 0`)

---

## 4. Non-Blocking Residual Hygiene Items Logged for Post-`v7.0` (`v7.1` / `PV7`) Cleanup

None of the four observations below affect any sealed SHA-256 digest, verifier gate, statistical conclusion, or Brutal-Honesty claim boundary:

1. **`NH-1` (Inline `/formal` `Statement.lean` Companion Notes in Whitepaper §4.9.6):** `formal/ChelatedFormal/Statement.lean:28-56` clearly documents that `all_wave14a_formal_theorems_verified` conjoins the `10` SHA-locked `Spec.lean` `*_Stmt` propositions while `ConformalChain.norm_mulVec`, `theorem4_chordal_cosine_drift`, and `evalStratum2Wfs_sound` sit outside that conjunction (though still kernel-checked and verified by `nanoda_bin`), that only Stratum 2 is kernel-bound (Strata 0/1/3 conformance-tested), that `Spec.lean` Theorem 2 scales runtime CSLS by `0.5×`, and that `Theorem 4` (`Theorem4_AnchoredClippedEMA.lean:16-20`) proves hard radial-projection `ρ`-ball invariance (`PM-6` freezes `α=0`). Whitepaper `v7.0:349` makes zero overclaim ("anchored-clipped EMA drift bound" + `conformance_tested_unproven_in_lean` + `153` Pitfall-B diffs), and `v7.1` can optionally surface those `Statement.lean` details inline.
2. **`NH-2` (Per-Generator `Arm 3` `R2` & Transport Latency Detail in Prose):** `pm6_posthoc_statistical_reanalysis.json` (`:143, 213`) records that `Arm 3` `R2` is `96.88%` (`31/32`) on `Nemotron-30B` vs `41.18%` (`7/17`) on `Qwen2.5-7B` (against `100.0%` `Arm 9` on both), and `wave14_d11_pm6_manifest.json` logs `Profile A` (`157.6 / 179.1 ms` p50 TTFT, `0` client-byte leaks) vs. `Profile B` (`3.34 / 5.08 ms` p50 TTFT, `<= 117` unarmed tokens exposed on `3/32` Nemotron works, `0` bound violations). Whitepaper `v7.0:352` already qualifies the `41.18%` `R2` parenthetical explicitly as `"Qwen R2 100.0% vs 41.18%"`.
3. **`NH-3` (Auxiliary Session-Log & Section-Header Hygiene Outside Publication Ledgers):** An older internal handoff log (`docs/next-session.md:229`, outside the `v7.0` publication suite) still contains the stale phrase `"retracted MemT-1.0 weights"` (already corrected in `WAVES_1_TO_21_MASTER_EVIDENCE_AND_PR_INDEX.md:68` and across all `v7.0` publication files); `WAVES_1_TO_21_MASTER_EVIDENCE_AND_PR_INDEX.md:4,12,26-27` retains pre-merge snapshot status strings alongside its updated `:51` summary; and Whitepaper Chapter 8 (`:397`) / HTML Section 4 (`:1218`) retain `"PM-1..PM-5"` / `"Post-Wave-8"` headings above tables that already include the `PM-6 & /formal` row.
4. **`NH-4` (Optional Defense-in-Depth Tokens in `build_whitepaper_assets.py`):** While `verify_all_offline.py` (`metrics.d11` + `formal.spec_sha256_lock`) already enforces `r5_benign_fpr_le_002 == FAILED` and `conformance_tested_unproven_in_lean` on the JSON artifacts and `build_whitepaper_assets.py:532-535` enforces `2610.26825`, `pm6_posthoc_statistical_reanalysis.json`, `63fe3e64`, and `7248d05a02ba` in the prose/HTML, adding `6.25%`, `11.76%`, and `conformance_tested_unproven_in_lean` to `validate_publication_documents()` would lock those prose strings against future accidental edits as well.

**Sign-Off:** **Independent Final Judge (`gemini-4-carbon` Medium) — `APPROVED_FOR_V7_SEAL`.**
