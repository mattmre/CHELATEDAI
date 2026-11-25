---
panel_type: wave15-postgpu-adversarial-xval
reviewer: gemini-4-neon (high)
rulebook: Brutal-Honesty Rulebook v3.3
target_worktree: D:\GITHUB\CHELATEDAI-V7-FORMAL-PM6
target_branch: feat/v7-formal-pm6
target_head: 60ffc1f7e4c392fc6ad96d90aadf0d104f474985
merge_base_vs_stage0_night1: d6e701854dff97cda392e31f740a0b65025932bd
stage0_night1_head: 2a922f29
audit_date: "2026-11-25"
verdict: CONDITIONAL GO — HOLD MERGE UNTIL VERIFIER/REGISTRY GAPS & MANDATORY V7.0 DISCLOSURES ARE CLOSED
---

# Wave 15 Post-GPU Adversarial Review (`v7.0` Merge & Formal-Gate Readiness)

**Overall Verdict:** **`CONDITIONAL GO (HOLD MERGE)`**

The `/formal` Lean 4 + `nanoda_bin 0.4.32` + `7,722`-checkpoint `EK5` conformance gate passes end-to-end (`exit 0`, `71.0s` CPU wall-clock) and the live `Demo D11 / PM-6` bundle (`wave14_d11_pm6_manifest.json`, SHA-256 `7248d05a02ba...`) is cryptographically intact (`12/12` Ed25519 receipts verified, `0/56` Non-NDA hits, `0` test leakage). **However, merging `feat/v7-formal-pm6` (`60ffc1f7`) onto `STAGE0-NIGHT1` (`2a922f29`) as-is has two hard verification blind spots and five mandatory statistical/methodological disclosures that will fail an adversarial Whitepaper v7.0 audit if left unaddressed:**

1. **`verify_all_offline.py --full-formal` currently NEVER verifies Bundle `#46` (`wave14_d11_pm6_takedown_carryforward`)** because `PART7_OPTIONAL_PREFIXES` searches for `"wave11_d11_"` instead of `"wave14_d11_"` and `bundle_registry_spec.json` is still frozen at v5.0 (`40` bundles).
2. **Arm 9 (`ZeroMu+CSLS`) does NOT beat Arm 6 (Raw Cosine) or Arm 8 (Linear Bridge) on English `R3-vs-R5` AUROC or test benign FPR** (`11.76%` Qwen / `6.25%` Nemotron vs `2%` DEV target), **Phase-3 cross-lingual `μ_lang` vs `μ_en` is statistically null** (`p=0.61–0.87`), **post-freeze `R2b` adaptive red-team breaches reach `15.6%–17.6%`** under separator/homoglyph attacks, and **top-level manifest fields silently alias `qwen25_7b` (`N=17`) rather than `nemotron_30b` (`N=32`)**.

---

## (a) `--full-formal` Live Verification Receipt

Executed in `D:\GITHUB\CHELATEDAI-V7-FORMAL-PM6` at `HEAD = 60ffc1f7e4c392fc6ad96d90aadf0d104f474985` with `CUDA_VISIBLE_DEVICES=""`:

```text
Command:   CUDA_VISIBLE_DEVICES="" python scripts/verify_all_offline.py --full-formal
Result:    PASS  (exit_code=0, mode=full-formal, formal_tier=PASS, skipped_checks=none)
Timing:    wall=71.0s (71.13s stopwatch), cpu=0.422s
Platform:  Windows-10-10.0.26300-SP0 | Python 3.11.9
```

### Live Check-by-Check Summary

| Gate Group | Result | Observed Evidence | Adversarial Finding |
|---|---|---|---|
| `pristine_mirror` | **PASS** | `15/15` files byte-exact | Clean. |
| `bundle.*` (`v5.0` spec) | **PASS** | `40/40` bundles SHA-256 verified (`sealed_unread=1` on `r5b_s4_pm4_cross_family_2x2`, never opened) | `13` unstaged `M artifacts/**/*.json` files in working tree are pure CRLF drift (`git diff --stat --ignore-cr-at-eol` empty) and still match receipt hashes on disk. |
| `part7.*` standalone | **PASS (INCOMPLETE)** | Verified `wave11_d6` (`0578d56b715d`) & `wave11_d8` (`3fea611fe16a`) only | **BLOCKER:** `artifacts/wave14_d11_pm6_takedown_carryforward/` (`7248d05a02ba`) was **completely skipped** (`PART7_OPTIONAL_PREFIXES` has `"wave11_d11_"`, not `"wave14_d11_"`). |
| `d8.*` crypto + replay | **PASS** | `3/3` Ed25519 + SHA-256 chain + file hashes + live subprocess replay (`head 820705ef47fc`) | `D8` was re-sealed on `STAGE0-NIGHT1` (`2a922f29`); re-verify post-merge. |
| `non_nda.*` | **PASS** | `0/56` hits across teaser `.md`, teaser `.html`, `compliance_copilot.html` | Clean. |
| `metrics.d6` / `metrics.d8` | **PASS** | D6 cross-engine parity + `0.00%` stale @ `0/10/50/90%`; D8 `38/38` checks | **GAP:** Missing `check_d7/d9/d10` (present on `STAGE0-NIGHT1`) AND missing `check_d11_headlines` (absent on BOTH branches). |
| `registry.*` & `release_status` | **PASS** | `verified_registry.json` (`6e6e9a156f38`, `40` bundles); `PRE-RELEASE` retained (`PM-4b` pending) | Stale relative to v6.0 (`45`) and v7.0 (`46`) bundles. |
| `formal.lake_build` | **PASS** | `leanprover/lean4:v4.36.0-rc1` + Mathlib `v4.36.0-rc1` compiled cleanly | Clean. |
| `formal.no_sorry_native_decide` | **PASS** | `0` `sorry` / `native_decide` across `9` `.lean` files | Clean. |
| `formal.print_axioms_allowlist` | **PASS** | `68` theorems depend solely on `['Classical.choice', 'Quot.sound', 'propext']` | Clean. |
| `formal.negative_control` | **PASS** | `3/3` injected violations (`bogusAxiom`, `Lean.ofReduceBool`, `sorryAx`) rejected | Clean. |
| `formal.lean4export_prepare` | **PASS** | `.lake/nanoda/chelated_formal_all.export` newer than all `formal/**/*.lean` and manifest | Re-test cold regeneration (`--prepare-nanoda`) in CI on clean checkout. |
| `formal.nanoda_external_kernel` | **PASS** | `nanoda_bin 0.4.32` (`0da9f98f`) checked `25,473` declarations (`>= 27` exported) with `0` typechecker errors | Clean independent Rust kernel verification. |
| `formal.ek5_1000_mutation_conformance` | **PASS** | `1,000` seqs (`seed=20261124`), `7,722` checkpoints (`6,000` mutation + `1,722` `as_of` incl. `951` missing-parent), `46,967` nodes, `41,709` edges; `1,531/1,531` injected comparator & materialization faults caught; `0` oracle mismatches; `0` `lean_vs_fresh` diffs; `0` semantic diffs across `eff_auth`, `val`, `cur`, `eligibility`, `node_set` | **`153` Pitfall-B pruning telemetry diffs** (`104` `pos_cyc` + `49` `reason` on `inc_vs_fresh` and `153` on `lean_vs_inc`, `306` total) — MUST be disclosed verbatim (see §c.1). |

### Independent D11 Cryptographic Verification (Executed Separately Because Verifier Skipped D11)

Verified manually offline via Python (`cryptography` Ed25519 + SHA-256):
- `report_receipt.json` (`git_sha = f357e4ba5780...`, `schema_version = report_receipt_v4_1`, `status = COMPLETED`, `seal_eligible = true`):
  - `wave14_d11_pm6_manifest.json`: SHA-256 `7248d05a02ba4901...` (**MATCH**, `123,279` bytes)
  - `calibration_freeze_manifest.json`: SHA-256 `70131a74faa63e68...` (**MATCH**)
  - `copilot_receipts_v2.json`: SHA-256 `569d3bd3d3309875...` (**MATCH**)
- `copilot_receipts_v2.json`: **`12/12` Ed25519 signatures and SHA-256 hash chain VALID** (`head_hash = b214649ccadd14bf...`, matches `report_receipt.receipts_head_hash`, `zero_floats_verified = true`, `non_nda_leak_hits = 0`).

---

## (b) Merge Conflict, Verifier, & Registry Integration Checklist (`STAGE0-NIGHT1` -> `v7.0`)

### 1. Git Merge Topology (`2a922f29` vs `60ffc1f7`)
- **Merge base:** `d6e701854dff97cda392e31f740a0b65025932bd`.
- **Dry-run 3-way merge (`git merge-tree --write-tree 2a922f29 60ffc1f7`):** Exit `0`, clean merged tree `061db820b966e8d84d07a679f4ebdfd1bb891078` — **ZERO textual merge conflicts**.
- **Overlapping files across both branches (only 2):**
  1. `tests/test_verify_all_offline.py`: **byte-identical** between `2a922f29` and `60ffc1f7` (`git diff 2a922f29..60ffc1f7` is empty).
  2. `scripts/verify_all_offline.py`: Disjoint hunks — `STAGE0-NIGHT1` added `check_d7_headlines`, `check_d9_headlines`, `check_d10_headlines` (`lines 718–795`), while `feat/v7-formal-pm6` added fail-closed `nanoda_bin` / `lean4export` / `run_conformance.py` gates (`lines 32–40`, `966–1240`).
- **Pre-merge sequencing gate (Master Plan §1.1):** `D:\GITHUB\CHELATEDAI-STAGE0-NIGHT1` currently has untracked `artifacts/wave11_d10_5m_real_cutover_and_70b_scale/` and `bundle_registry_spec.json` is still at `v5.0` (`40` bundles). **Seal & commit `v6.0` (`45/45` bundles including `D10`) on `STAGE0-NIGHT1` BEFORE merging `feat/v7-formal-pm6`.**

### 2. Exact Additions Required in `scripts/verify_all_offline.py`
1. **Fix D11 bundle discovery (`line 98`):** Replace `"wave11_d11_"` with `"wave14_d11_"` in `PART7_OPTIONAL_PREFIXES` (or promote `D6–D11` into `bundle_registry_spec.json` as required spec entries `#41–#46` so `verify_spec_bundles` always enforces them).
2. **Add `check_d11_headlines(manifest)` and wire `("d11", "wave14_d11_", check_d11_headlines, ...)` into `verify_headline_metrics()`** (preserving `STAGE0-NIGHT1`'s `check_d7/d9/d10_headlines`):
   - Assert `status == "COMPLETED"`, `all_gates_pass is True`, `seal_eligible is True`, `execution_mode == "run-live"`.
   - Assert governance scope locks: `unlearning_claimed is False`, `parametric_recall_neutering_claimed is False`, `hosted_gateway_scope_only is True`, `cites_audit_is_auditable_suppression_not_physical_erasure is True`.
   - Assert `zero_test_leakage_audit.verified is True` and `0` family/node/author/text overlaps.
   - Assert `bridge_diagnostics`: `stiefel_column_rank == 384`, `full_target_manifold_spanned is True`, `effective_rank >= 380`, `stiefel_orthogonality_frobenius_err < 1e-6`.
   - Assert `ek5_v3_r4_lineage_audit`: `primary_leaked_general == 0`, `copy_of_leaked_general == 0`, `derives_current_leaked_general == 0`, `cites_audit_preservation_rate == 1.0`, `n_enterprise_fixtures == 60`.
   - Assert transport safety on **both** generators (`nemotron_30b`, `qwen25_7b`): `profile_a.flagged_window_client_byte_leaks_per_work == 0`, `injected_500ms_delay_test.zero_byte_leak_pass is True` & `real_sleep_all_strides is True`, `sse_sidechannel_audit.strict_reject_raised_value_error is True`, `profile_b.tier2a_literal_byte_leaks == 0`, `unarmed_exposure_bound_violations == 0`, `unarmed_async_forced_bound_violations == 0`.
   - Assert honesty locks so disclosures cannot be silently dropped: `reduced_yield_fallback_any is True`, `arm7_caa_qwen25_7b_reference.scope_disclosure == "untuned_lightweight_literature_reference_qwen25_7b_only"`, and `post_freeze_r2b_adaptive_redteam_by_generator` reports non-zero breach counts (`3/17` Qwen, `5/32` Nemotron).
3. **Add `verify_d11_receipt_crypto(repo_root, report)`:** Independently verify all `12/12` Ed25519 signatures, SHA-256 hash chain, `zero_floats_verified == True`, `non_nda_leak_hits == 0`, and `head_hash == report_receipt.receipts_head_hash` on `artifacts/wave14_d11_pm6_takedown_carryforward/copilot_receipts_v2.json` (parallel to `verify_d8_receipt_crypto`).
4. **Add `formal/spec_sha256.json` lock check in `--fast` AND `--full-formal`:** Verify `sha256(formal/ChelatedFormal/Spec.lean) == formal/spec_sha256.json["spec_sha256"]` (`63fe3e644bde...`) and confirm `formal/conformance_report.json["incremental_with_deletions_status"] == "conformance_tested_unproven_in_lean"`.
5. **Unit tests (`tests/test_verify_all_offline.py`):** Add positive + negative mutation tests for `check_d7/d9/d10/d11_headlines`, `verify_d11_receipt_crypto`, and `wave14_d11_` discovery.

### 3. Exact Additions Required in `scripts/build_whitepaper_assets.py` & `docs/research/bundle_registry_spec.json`
Both files are identical across `STAGE0-NIGHT1` and `feat/v7-formal-pm6` and remain at **`v5.0` (`40/40` bundles)**. For `v7.0` (`46/46` bundles):
1. **`docs/research/bundle_registry_spec.json`:**
   - Bump `schema_version` to `"bundle_registry_spec_v7_0"` (keeping `release_status: "PRE-RELEASE / HUMAN AUDIT & CLEAN-RERUN GATES PENDING"` while `PM-4b` human grading is pending).
   - Register `#41–#45` (`D6–D10`, Part VII) and `#46` `wave14_d11_pm6_takedown_carryforward` (Part VIII) with `part_vii_code_provenance` and `part_viii_code_provenance` (pinning `formal/spec_sha256.json`, `nanoda_bin 0.4.32` rev `0da9f98f`, `lean4export` rev `df81913e`, `run_wave14b_d11_pm6_takedown_carryforward.py`, and `chelated_takedown_firewall.py` SHA-256s).
2. **`scripts/build_whitepaper_assets.py`:**
   - Bump `schema_version` to `"verified_registry_v7_0"`.
   - Add `extract_key_metrics()` extractors for `D6–D10` and **`wave14_d11_pm6_takedown_carryforward`** (summarizing `gates`, `catalog_summary`, `bridge_diagnostics`, `phase0_memorization_confirmation` counts, `nine_arm_matrix_by_generator` minus raw ROC arrays, `paired_arm_auc_comparisons_by_generator`, `phase3_cross_lingual_4way_ablation` AUCs/bootstrap CIs, `post_freeze_r2b_adaptive_redteam_by_generator`, `transport_profiles_by_generator`, `ek5_v3_r4_lineage_audit`, and `copilot_receipts_v2_ref`).
   - Extend `validate_publication_documents()` with mandatory `v7.0` honesty/citation tokens: `7248d05a02ba`, `63fe3e644bde`, `nanoda`, `conformance_tested_unproven_in_lean`, `153`, `382`, `0.1176` / `0.0625` (Arm 9 test FPR), `15.6` / `17.6` (R2b breach), `reduced_yield`, and `2610.26825` (`MemT-1.0`).
   - Run `python scripts/build_whitepaper_assets.py` then `python scripts/build_whitepaper_assets.py --check --allow-gitignored-raw-missing` after updating Whitepaper v7.0 + HTML v7.0.

---

## (c) Mandatory `v7.0` Disclosures, Statistical Audit, & Related Work Framing

### 1. `/formal` Lean 4 Scope & Conformance Disclosures
1. **153 Pitfall-B Pruning Telemetry Diffs (`conformance_report.json`):** `ek5_v3_stratified.py:665` skips re-solving already-`F` nodes protected by a positive-cycle or `sigma` shield, producing `104` `pos_cyc` + `49` `reason` (`REFUTED` vs `UNFOUNDED_CYCLE`) diffs on `inc_vs_fresh` and `153` identical diffs on `lean_vs_inc` (`306` non-semantic diffs total, `0` semantic diffs on `eff_auth`/`val`/`cur`/`eligibility`/`node_set`). State explicitly that **incremental maintenance with deletions is `"conformance_tested_unproven_in_lean"`**.
2. **Deductive vs. Conformance Boundary & Umbrella Split (`Statement.lean:28-56`):**
   - `all_wave14a_formal_theorems_verified` locks **10 `*_Stmt` conjuncts** (`formal/spec_sha256.json` = `63fe3e644bde...`). Three exported lemmas (`ConformalChain.norm_mulVec`, `theorem4_chordal_cosine_drift`, `evalStratum2Wfs_sound`) are kernel-checked and verified by `nanoda_bin` (`27` exported declarations) but sit **outside** the SHA-locked `Spec.lean` umbrella.
   - Only **Stratum 2** (`evalStratum2Wfs_sound`) is kernel-bound; **Strata 0, 1, and 3** are conformance-tested only, unproven in Lean.
   - **CSLS 2× Scale Convention:** `Spec.lean` Theorem 2 bounds `cos - 0.5*(r_q + r_d)`, while runtime computes `2*cos - r_q - r_d`; bounds transferred to runtime scores scale by `2×`.
3. **Theorem 4 (`Anchored-Clipped EMA`) Strength Honesty (`Theorem4_AnchoredClippedEMA.lean:16-20`):** The Lean proof is a 4-line induction whose inductive step depends **only on hard radial projection** (`radialClip_norm_sub_le`), ignoring `α`, `λ`, and `q t`. Do **not** describe Theorem 4 as an EMA convergence/contraction proof — describe it accurately as **hard-projection `ρ`-ball invariance** paired with `2ρ/γ` chordal / `1 - 2(ρ/γ)²` cosine stability (`theorem4_chordal_cosine_drift`) and the `100`-query empirical burst test (`max_clipped_drift = 0.05` vs `0.999761` unclipped at `d=384`). Also note `PM-6` freezes `μ_q = μ_0` (`α = 0`).

### 2. Statistical & Reproducibility Audit of `wave14_d11_pm6_manifest.json` (`7248d05a02ba`) & `git diff 5f15c731..60ffc1f7`

| Audit Item | Exact Evidence in `7248d05a02ba` / Diff | Mandatory Whitepaper v7.0 Framing |
|---|---|---|
| **1. Arm 9 vs Arm 6 (Raw Cosine) & Arm 8 (Linear Bridge) on English R3** | **Qwen (`n=17`):** Arm 9 AUROC `0.9758` (`88.24%` R3, **`11.76%` (`2/17`) R5 FPR**) trails Arm 6 (`0.9965`, `100%` R3, `5.88%` FPR; `Δ=-0.0208`, `p=0.1767`) & Arm 8 (`0.9896`, `88.24%` R3, `0.00%` FPR; `Δ=-0.0138`, `p=0.3921`). **Nemotron (`n=32`):** Arm 9 AUROC `0.9893` (`93.75%` R3, **`6.25%` (`2/32`) R5 FPR**) trails Arm 6 (`1.0000`, `100%` R3, `3.13%` FPR; `Δ=-0.0107`, `p=0.1543`) & Arm 8 (`0.9932`, `81.25%` R3, `0.00%` FPR; `Δ=-0.0039`, `p=0.7311`). | **NEVER claim `ZeroMu+CSLS` beats Raw Cosine or Linear Bridge on English R3 AUROC or benign FPR.** Disclose honestly that Arm 9's held-out R5 benign FPR (`6.25%` Nemotron / `11.76%` Qwen) exceeded the `2.0%` DEV calibration target (`CP95` upper `18.39%` / `32.62%`), and scope Arm 9's verified gains strictly to: (i) **`+0.474/+0.476` AUROC over Arm 3 lexical-only (`p=0.0001`)**, (ii) **`100%` (`60/60`) R4 derivative suppression vs `0%` (`Arms 1–3, 7`) and `0–5%` (`Arm 5` Gate-2-only)**, and (iii) **cross-lingual `4096→384` carry-forward**. |
| **2. Arm 3 Narrowing Nuance (Nemotron vs Qwen)** | Qwen R2: Arm 9 `100%` vs Arm 3 `41.18%` (`+58.8 pp`). **Nemotron R2: Arm 9 `100%` vs Arm 3 `96.88%` (`31/32`)** — only `3.13 pp` (`1` work) above the `2.0 pp` pre-registered narrowing tolerance. | Report **both** generators for the Arm 3 comparison; do not cite Qwen's `41.18%` as representative of Nemotron (`96.88%`). |
| **3. Phase-3 Cross-Lingual `μ_lang` vs `μ_en` Null** | `Arm c` (`4096→384`, `μ_lang`) beats `Arm a` (`MiniLM-384`) on **`DE` (`+0.1522`, `p=0.0029`)** and **`ES` (`+0.1003`, `p=0.0163`)**, **BUT `c vs b` (`μ_lang` vs `μ_en`) is null**: `DE +0.0069 (p=0.8667)`, `ES +0.0104 (p=0.6093)`. Native `4096-d` oracle (`Arm d`) hits `1.0000` (`d vs c`: `DE p=0.6983`, `ES p=0.2301`, underpowered at `n=17`). | Attribute the cross-lingual carry-forward win honestly to the **multilingual `4096→384` Stiefel bridge** (`c vs a`), NOT to unlabeled `μ_lang` re-centering (`c vs b` null). |
| **4. Post-Freeze `R2b` Adaptive Red-Team Breaches** | **Qwen (`n=17`):** `17.65%` (`3/17`, `CP95` upper `39.56%`: `2` interleaved-separator, `1` homoglyph, `0` multi-turn). **Nemotron (`n=32`):** `15.63%` (`5/32`, `CP95` upper `30.08%`: `2` interleaved-separator, `3` homoglyph, `0` multi-turn). | Spotlight `15.6%–17.6%` any-of-3 breach rates prominently as an **open adversarial surface** against character-level obfuscation (separator insertion & Unicode confusables). |
| **5. Top-Level Manifest Generator Aliasing** | Top-level `nine_arm_matrix`, `paired_arm_auc_comparisons`, `arm3_pre_registered_narrowing_rule`, `post_freeze_r2b_adaptive_redteam`, `catalog_summary` (`dev=15, test=17`), and `phase3` (`n=17`) all alias **`qwen25_7b`**, NOT `nemotron_30b` (`dev=27, test=32`) or the union (`38` unique test families in `zero_test_leakage_audit`). | Always tabulate **both generators explicitly** (`Nemotron-30B N=32`, `Qwen2.5-7B N=17`, union `N=38`); never cite top-level fields without generator & `N` labels. |
| **6. Post-Signoff Amendments (`5f15c731..60ffc1f7`)** | Five commits landed after R3 pre-GPU judge sign-off (`5f15c731`): (i) `f140e23b` relaxed MiniLM intrinsic manifold rank gate `384 -> 380` (`effective_rank=382`, `stiefel_column_rank=384`, while keeping misleading gate key name `"bridge_effective_rank_384": true`), (ii) `f6a82cd2` + `f357e4ba` added `100` external pre-1929 works (`200 -> 300`) and `extract_r1_continuation_span` (`max(raw, span)` for Phase-0 yield), (iii) `64ec9fcf` added confirmation-signature stratified fallback against mod-11 skew (`reduced_yield_fallback_any=true`: `32/300` Qwen, `59/300` Nemotron vs planned `>=110`). | Disclose all post-signoff amendments explicitly in the v7.0 protocol ledger. **Crucial corpus caveat:** Inspecting `pm6_external_public_domain_canaries_v1.jsonl` shows all `100` added `r5_benign_*` prompts are **templated archival-ledger queries** (`"In comparative archival series 1810..1909, how did municipal harbour ledgers..."`, with minor synthetic German inflection artifacts like `"Wie haben ... vereinheitlichten?"`) rather than topic-matched literary benign siblings — disclose that this makes R5 negatives easier on the added third of the catalog. |
| **7. Transport Latency, Profile B Bound Telemetry, & CAA Hash Naming** | **Profile A:** `0/17` & `0/32` flagged client-byte leaks (`16/16` strides `0` bytes under `500.11 ms` real sleep), benign p50 TTFT = **`157.6 ms` (Qwen) / `179.1 ms` (Nemotron)** (`~130 ms` encoder RPC). **Profile B:** benign p50 TTFT = **`3.34 ms` / `5.08 ms`**, `0` Tier-2A literal leaks, `0` bound violations, **but** Nemotron had `3/32` unarmed works (`max_unarmed_exposed_tokens = 117 <= 132`), and Qwen reports `unarmed_async_forced_max_exposed_tokens = 122 > max_theoretical_exposure_bound = 119` with `0` violations (because `119` aggregates normal runs while forced-unarmed checks each run's own live `L_2B_actual`). Also note 3 distinct `"dev_ids_sha256"` digests across `calibration_freeze_manifest.json` (`4754d48a`) and `arm7` (`bbfadab4` / `0f538643`). | Disclose Profile A's `158–179 ms` encoder-RPC TTFT cost vs Profile B's `<=117`-token unarmed semantic exposure window; explain the forced-unarmed `122` vs `119` bound denominator (or log `unarmed_async_forced_max_theoretical_bound` in future runs) and label the `arm7` prompt-subset `dev_ids_sha256` keys clearly. |

### 3. Required `MemT-1.0` (`arXiv:2610.26825v5`) Related Work Framing for Whitepaper v7.0
`MemT-1.0` (`arXiv:2610.26825v5`) currently has **0 citations** in `docs/research/doctoral-whitepaper-chelated-memory-and-spherical-migration-2026-11-21.md`. Add a dedicated Related Work subsection in Whitepaper v7.0 covering:
1. **Accurate Technical Summary & Factual Correction:** *MemT-1.0: Towards Memory-Native Evolution* (`arXiv:2610.26825v5`, MemTensor; `Nano 36B`, `Flash 288B`, `Pro 554B` built on NVIDIA Nemotron-3-Ultra) internalizes long-horizon memory via **Gated DeltaNet-2** recurrent updates + **Mixture-of-Storage** native memory blocks (~`66×` KV-cache storage reduction and ~`1.86×` 128K-context inference speedup). **CRITICAL FACTUAL FIX:** `docs/research/WAVES_1_TO_21_MASTER_EVIDENCE_AND_PR_INDEX.md:68` erroneously writes *"retracted `MemT-1.0` weights"*. As of `2026-11-25`, **`arXiv:2610.26825v5` is active and NOT retracted**, and MemT-1.0 weights remain live on Hugging Face. **Do NOT repeat "retracted" in Whitepaper v7.0** — state factually that v7.0 executes `0` GPU runs on MemT-1.0 weights by architectural scope design.
2. **Orthogonal Architectural Boundaries vs. Chelated Memory:**
   - *Opaque Recurrent State vs. Selective Retraction & Audit (`EK5 v3` + `PM-6`):* Folding history into fixed-size GDN-2/MoS recurrent matrices improves context compression but cannot perform deterministic $O(|\text{dirty cone}|)$ selective retraction, cycle-safe three-valued `WFS` abstention, non-destructive `CITES_AUDIT` legal holds, or cross-generation hosted-gateway takedown carry-forward (`Demo D11`) without full history replay or weight-level unlearning.
   - *Weight-Bound State Identity (`TRA-1` vs. `modeling_memt.py:876`):* MemT-1.0's `_state_identity()` omits base and adapter weight hashes (creating a silent cross-revision state-reuse risk when weights/adapters change while config and tokenizer stay unchanged); Chelated `TRA-1` cryptographically binds vector/cache/state spaces to model weights, adapters, and epochs.
   - *Symmetric Evaluation Protocol:* Note the asymmetric reranker comparison caveat in `arXiv:2610.26825v5` (motivating Chelated's shared-read-function rule in post-v7.0 work) and position MemT-1.0 cleanly as a **complementary memory-native backbone** whose hosted outputs still require externalized epistemic governance (`EK5 v3` + `ZeroMu` + `PM-6`).

---

## (d) Verdict & Go/No-Go Release Gate

- **Verdict:** **`CONDITIONAL GO — HOLD MERGE OF `feat/v7-formal-pm6` (`60ffc1f7`) INTO `STAGE0-NIGHT1` (`2a922f29`)`** until the following **4 concrete items** are completed in order:
  1. **Seal `v6.0` (`45/45` bundles incl. `D10`) on `STAGE0-NIGHT1` first** (per Master Plan §1.1), then merge `feat/v7-formal-pm6` (`0` textual git conflicts verified).
  2. **Close the D11 verifier blind spot in `scripts/verify_all_offline.py`:** add `"wave14_d11_"` / spec registration, `check_d11_headlines`, `verify_d11_receipt_crypto` (`12/12` Ed25519 + chain), `formal/spec_sha256.json` lock check, and unit tests while preserving `STAGE0-NIGHT1`'s `check_d7/d9/d10_headlines`; re-run `python scripts/verify_all_offline.py --full-formal` (`exit 0`).
  3. **Upgrade `docs/research/bundle_registry_spec.json` & `scripts/build_whitepaper_assets.py` to `v7.0` (`46/46` bundles)** with `D6–D11` metric extractors and v7.0 mandatory disclosure tokens, and pass `build_whitepaper_assets.py --check`.
  4. **Incorporate every §c disclosure verbatim in Whitepaper v7.0 / HTML v7.0** (Pitfall-B `153` telemetry diffs, `conformance_tested_unproven_in_lean`, Theorem 4 projection-only scope, Arm 9 vs Arm 6/Arm 8 English AUROC & `6.25%`/`11.76%` test FPR, `μ_lang` vs `μ_en` null, `15.6%–17.6%` R2b breaches, `5f15c731..60ffc1f7` post-signoff amendments + archival R5 template caveat, both-generator reporting, accurate non-"retracted" `MemT-1.0` `arXiv:2610.26825v5` Related Work framing, and retained `PRE-RELEASE` status while human `PM-4b` grading remains open).
