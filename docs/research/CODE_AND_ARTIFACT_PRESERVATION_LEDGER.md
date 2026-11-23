# Code, Baseline, & Experimental Artifact Preservation Ledger

**Non-Destructive Engineering Policy:**  
No historical baseline, code version, whitepaper revision, or experimental artifact in `CHELATEDAI-STAGE0-NIGHT1` is ever deleted or overwritten without first locking a byte-exact SHA-256 backup snapshot. Every experimental result in `docs/research/verified_registry.json` (`40/40` bundles across Parts I–VI, `49/49` receipt-bound side-artifact SHA-256 digests) maps back to an exact retrievable code snapshot below.

---

## 1. Git Tags & Checkpoints (Pushed to `origin`)

| Checkpoint / Tag | Commit SHA | Scope Preserved |
|---|---|---|
| `baseline/pre-fix-pristine-2026-11-21` | `b37e6a5523f8647c0d682178628db1f1545dc1ed` | Untouched pre-Council Stage-0 / R1–R3 codebase & historical manifests (`Part I` & `Part II`). Also mirrored byte-for-byte in git at `archive/pristine-b37e6a55-exact/` (`15/15` files). |
| `checkpoint/r4-complete-pre-overnight-r5-2026-11-22` | `ceb67cfad9a69d6252150ff73fbee43127150488` | Post-Council R4 complete codebase, `f1_r4_comprehensive` + `f2_ek5_v3_1m_powerlaw` + `f2_r4_zero_shortcut_v2_and_locomo` (`18/18` bundles), Whitepaper v2, and Presentation HTML v2 (`registry_sha: bd1577d750bd`). |
| R5 Pre-Overnight Code Commit | `13943aae30acb4331644b7ea85887047974dae43` | Additive R5 `em4_soft_telemetry_routing.py` (`SHA-256: 761fdcc41115...`), unit tests, and initial overnight Spark harnesses prior to R5b completion scripts. |

---

## 2. Local Byte-Exact Backup Vaults (`D:\GITHUB\CHELATEDAI-STAGE0-NIGHT1\archive\`)

*(Note: Only `archive/pristine-b37e6a55-exact/` is tracked in git to keep repository size lean; all vaults below reside intact on local storage with `MANIFEST.json` SHA-256 verification, plus remote Spark vaults at `/home/mattmre/chelated-night1/backups_r4_ceb67cfa/` on `10.0.0.194` and `10.0.0.49`.)*

| Local Vault Directory | Files | Contents & Purpose |
|---|---:|---|
| `archive/pristine-b37e6a55-exact/` | `15` | Read-only byte-exact mirror of commit `b37e6a55` (`PRISTINE_MANIFEST.json`, verified on every `build_whitepaper_assets.py --check`). |
| `archive/pristine-b37e6a55/` | `15` | Earlier PowerShell copy exhibiting BOM/CRLF drift—retained untouched as forensic audit evidence (`B-15`). |
| `archive/r4-complete-ceb67cfa-backup/` | `24` | Complete R4 code, manifests, receipts, Whitepaper v2, and HTML v2 snapshot before overnight R5. |
| `archive/r5-overnight-s123-backup/` | `12` | Raw CSVs, manifests, and receipts from overnight R5 (`r5_s1_zs_v3_flooded`, `r5_s2_locomo_300_em4`, `r5_s3_longmemeval_n500`, `pm4` sealed key). |
| `archive/r5b-spark37c3-pm2-backup/` | `15` | Raw CSVs, manifests, and receipts from `gx10-37c3` (`r5_s1_pm2`, `r5b_s1b_pm2_hybrid`, `r5b_s2_pm3_default`, `r5b_s2b_pm3_steelmanned`). |
| `archive/r5b-spark6724-pm1a-pm4-backup/` | `8` | Raw CSVs, manifests, and receipts from `gx10-6724` (`r5b_s1_pm1a_closure`, `r5b_s4_pm4_2x2`). |
| `archive/r5-r5b-complete-vault-2026-11-22/` | `39` | Consolidated master vault of all R5 + R5b CSVs, manifests, receipts, and sealed audit keys. |
| `archive/r5-exact-code-snapshot-2026-11-22/` | `10` | **Exact Python modules & scripts (`R5_EXACT_CODE_MANIFEST.json`)** as executed during R5 & R5b prior to 4-Model Panel edge-case fixes (`em4_soft_telemetry_routing.py` `761fdcc41115...`, `ek5_v3_stratified.py` `b54dadb51e1f...`, `spherical_migration_bridge.py` `437b4db73431...`, and all 6 R5/R5b runners). |
| `archive/whitepaper-v2-pre-r5-update-backup/` | `5` | Exact v2 `bundle_registry_spec.json`, `verified_registry.json`, Whitepaper v2, HTML v2, and `build_whitepaper_assets.py` prior to Draft v3 update. |
| `archive/whitepaper-v3-pre-panel-edits-backup/` | `9` | Exact Draft v3 (`b57b05660edf`) files as audited by the 4-Model Frontier Panel (`GPT-6.2 Astra Ultra`, `Claude Opus 5.5 Ultrathink`, `Grok 4.7 XHigh`, `Gemini 4 Neon XHigh`) prior to post-panel refinements, plus `phase2-roadmap-6week-v1.md` (`SHA-256: 4f620d90b75a0c40e933cb944170c0d821be6e8508c15b92c6737676cdd8f7b0`) prior to 10-day sprint acceleration. |
| `archive/whitepaper-v3.1-pre-v4-backup/` | `6` | Exact Draft v3.1 (`BACKUP_MANIFEST.json`) before Draft v4.0 integration: Whitepaper v3.1 (`ecadaa26c03f...`), Presentation HTML v3.1 (`acbebb5e6ca9...`), `bundle_registry_spec.json` (`74398a5dd458...`), `verified_registry.json` (`3fef4bb716af...`), and `build_whitepaper_assets.py` (`65f38a9f7d03...`). |

---

## 3. Exact R5/R5b Execution Code SHA-256 Ledger (`Part IV` Bundles)

Because Part IV runs were synced via `scp` onto DGX Spark worktrees whose base git `HEAD` sat at `18973789615ecab1268a77ebd3c69a283607241c`, each Part IV manifest's `git_sha` records that base commit while the exact executed Python files had the following SHA-256 digests (archived verbatim in `archive/r5-exact-code-snapshot-2026-11-22/`):

- `ek5_v3_stratified.py`: `b54dadb51e1f54576f39e22808fa9c7cd9209fe3de86bf62139eded10b13dfd0` (`69,467` bytes)
- `spherical_migration_bridge.py`: `437b4db7343176c75300c1a4c31855ea7d6e5a5f703857e92858cafbbe855948` (`61,450` bytes)
- `em4_soft_telemetry_routing.py` *(R5 exact; post-panel fixes retain `legacy_r5_compat=True`)*: `761fdcc4111503b223157f4ceb19bfc1fe03ae72229ed43fb7f9c7933da3b7eb` (`16,789` bytes)
- `scripts/run_overnight_r5_spark6724.py`: `10a19831d385534a605a5fb8a2bda05e5d0d8b2772ea54070f417fd02acf7995` (`72,902` bytes)
- `scripts/run_overnight_r5_spark37c3.py`: `6c3b9450f06f6c6b609aa66966ab6470569137c512af36bb1dceef050b66359b` (`52,592` bytes)
- `scripts/run_r5b_spark6724_completion.py`: `d4239fe2be2a2a865b3e1730a33043cb6f20d74a0a500128cb350d8dcdf2607e` (`23,464` bytes)
- `scripts/run_r5b_spark37c3_part2_and_hybrid_pm2.py`: `76ebe7da47ee06cdfd294480cf3ff573503309a4846a1efadda6b1f0328d1832` (`31,089` bytes)
- `scripts/run_r5b_steelmanned_mem0_pm3.py`: `3d2f0ff4334a78a630b47f8c974f904ff74dd78092fc504b6a5703eddc12fda0` (`20,200` bytes)


---

## 4. Phase-2 Demo D1 (`4-Node Fleet Proof`) & Encrypted Google Drive Master Vault (`2026-11-22`)

- **Encrypted Google Drive Master Vault (`O:\backups\CHELATEDAI-COMPLETE-VAULT-2026-11-22\`):**
  - `CHELATEDAI-COMPLETE-VAULT-2026-11-22.zip.001`: `1,992,294,400` bytes (`SHA-256: d4c8fb8b46644c4aa1509eb1c59d7b3c82a028c9931fea2459397b970ddf7530`)
  - `CHELATEDAI-COMPLETE-VAULT-2026-11-22.zip.002`: `1,992,294,400` bytes (`SHA-256: 304111a9f4186fe595107aa72c52d9d974d7967cd81dc7a648ad1527ab34b6e0`)
  - `CHELATEDAI-COMPLETE-VAULT-2026-11-22.zip.003`: `1,873,870,726` bytes (`SHA-256: d809d263f991cc7c8878f4e62579a4d5f2dc4cc9d525d34336dfa492e961e76e`)
  - Seals `37,767` files (`17.942 GiB` uncompressed -> `5,587.06 MiB` WinZip/7-Zip AES-256): `293`-ref `--all` Git bundle, all `21` `CHELATEDAI*` directories + nested `.claude/worktrees`, remote DGX Spark `gx10-37c3` & `gx10-6724` `out/` tarballs, `27/27` receipt bundles, Phase-2 D1 4-node fleet proof, and `CHELATEDAI` logs across T3 (`statev2.sqlite` + logs), Antigravity/Gemini (`19` brains), Claude Code (`3` project trees), OpenAI Codex (`25` sessions), and 4-Model Frontier Panel reviews. Verified via `7z t` (`Everything is Ok`).
- **Additive Phase-2 Demo D1 Modules (Zero Mutation of R1–R5b Files):**
  - `chelated_raw_text_exocortex.py`, `prospective_realworld_drift_corpus.py`, `scripts/run_phase2_d1_local_fleet_proof.py`, `tests/test_chelated_raw_text_exocortex.py`, and `artifacts/phase2_d1_4node_local_fleet_proof/phase2_d1_4node_fleet_manifest.json` (`rows_csv_sha256: c908d82f2a45bd0a666f7d5e118ab3f4fdf00772380ffe5b54b70b1de7abc3b5`).


---

## 5. Stage R6 (`PM-2` Label-Free Closure) & Phase-2 Wave 2 (`D1 v2` Full `N=60` + `D2` `1M`-Vector Migration) (`2026-11-22`)

- **Stage R6 `PM-2` Label-Free & Causal Streaming Closure:**
  - `scripts/run_r6_pm2_labelfree_closure.py` & `artifacts/r6_pm2_labelfree_closure/r6_pm2_labelfree_manifest.json` (`rows_csv_sha256: d99cd138b30c510ead0f7dfc9260831f6ec6c80c18384f7f68a64628a82d4bc6`, `7/7` gate checks passed, `0` `c["family"]` labels, causal EMA `mu_q = 0.9750` vs `Native = 1.0000`, `p95 = 7.589 ms` on the full 3,200-record hybrid workload).
- **Phase-2 Wave 2 (`D1 v2` Full `N=60` 4-Node Fleet Proof + `D2` `1,000,000`-Vector `Drift-Adapter` vs `ZeroMu` + Semantic Smear):**
  - `chelated_epistemic_proposer_v2.py`, `tests/test_chelated_epistemic_proposer_v2.py`, `scripts/run_phase2_d1_v2_and_d2_1m_scale.py`, `docs/research/phase2-wave2-results-dashboard.html`, and `artifacts/phase2_wave2_d1v2_and_d2_1m/phase2_wave2_d1v2_and_d2_1m_manifest.json` (`rows_csv_sha256: 3fcbab523376d14e806c9d65e9abddd02ff9e2dd700138809e1fc846937e769c`).

---

## 6. Wave-4 Frontier Cross-Judgment Pre-Adjudication Vaults (`2026-11-22`)

- `archive/r6-pm1b-v1-spark6724-backup/`: Exact v1 `PM-1b` (`3,534` rows) snapshot before DualGran v2 (`3,768` rows) and Wave-4 `J4` held-out architectural split.
- `archive/r6-pm3-v1-aborted-seq-3rows/`: Archived 3-row sequential `PM-3` pilot before `ChelatedConversationalV3` sharded `N=40` execution.
- `archive/wave4-pre-adjudication-v1-backups/`: Complete SHA-256 vault (`BACKUP_MANIFEST.json`) preserving exact v1 manifests, CSVs, and JSONL logs across `r6_pm1b_conversational_sota`, `r6_pm2_labelfree_closure`, `r6_pm4_conversational_kappa_n200`, `phase2_d2_qdrant_hnsw_csls_backfill`, `phase2_d3_multiagent_poisoning`, and `phase2_wave2_d1v2_and_d2_1m`, plus `docs_v3_1/` byte-exact pre-Wave-4 `v3.1` copies of `bundle_registry_spec.json`, `verified_registry.json`, `doctoral-whitepaper-chelated-memory-and-spherical-migration-2026-11-21.md`, and `chelated-memory-spherical-migration-whitepaper-presentation.html`, prior to executing Wave-4 unanimous cross-judgment (`J1–J7`) updates and compiling v4.0.

---

## 7. Waves 5–9 (`Part VI` Closure & Gateway, Investor Package, Live Solution Workbench, & `v5.0` Vault — `2026-11-23`)

- **Pre-`v5.0` Byte-Exact Snapshot (`archive/whitepaper-v4.0-pre-v5-backup/BACKUP_MANIFEST.json`, `6/6` Files Verified):**
  - Preserves exact Draft `v4.0` (`5a9bd46f2eeae2522a4a8d81017edb3c2407f936`) copies of `bundle_registry_spec.json`, `verified_registry.json` (`5603e923c5e3...`, `35/35` bundles), `doctoral-whitepaper-chelated-memory-and-spherical-migration-2026-11-21.md`, `chelated-memory-spherical-migration-whitepaper-presentation.html`, `build_whitepaper_assets.py`, and `CODE_AND_ARTIFACT_PRESERVATION_LEDGER.md` prior to `v5.0` synthesis.
- **Additive `Part VI` Experimental Receipt Bundles (`5/5` New Bundles -> `40/40` Total, Zero Mutation of `35/35` `v4.0` Bundles):**
  1. `artifacts/wave6_pm1b_v4_reranked_sota/` (`manifest_sha256: 5fcfab2b4c1c62e7babe4d8225f16b745ec61112efc1218eaeb0fc555957f6f3`, `4/4` side hashes)
  2. `artifacts/wave6_d2_4096d_3beir_complete/` (`manifest_sha256: dcb84026a15c4863fb4c2a0028cd14387c3ffe0d3c0087c581f28d717943f547`, `1/1` side hash)
  3. `artifacts/wave6_pm3_v2_competitor_arena/` (`manifest_sha256: 05442a849c199ad32955ce2145c1540b7bf0f78c3e56bb6392a69100894c5773`, `1/1` side hash)
  4. `artifacts/wave6_d4_trained_epistemic_grpo/` (`manifest_sha256: f20433dce33cde8b47c303ddf834046e9374d04b3a77288a8fb81f586135f28a`, `2/2` side hashes)
  5. `artifacts/wave6_d5_grpc_interceptor_gateway/` (`manifest_sha256: cba30b931812ae3f13133ff2def304cb81d363a0a7adb8ce8cba481d457bc743`, manifest-complete; `qdrant/qdrant@sha256:d277592721d2e3b73f9b797b53cf05e6a5bb2134f6d13fdbda4ca51782afbcac`)
- **Exact `Part VI` Producer Modules & Scripts (Committed to Git & SHA-256 Locked in `bundle_registry_spec.json`):**
  - `wave6_common.py`: `238352f8b802787d587a52cbdc4139aaa5f978ea5cf237f08a9034271dc4aef3`
  - `chelated_conversational_v4.py`: `8534c485dbb35e30789f6d1e44e4fb098436d26c09cc1f51a8cb1a7a1c70f1f4`
  - `virgin_conversational_holdout.py`: `a804140c537a538fe1d6cf53430e0dfd2b461306b0863b78bf26fbb8d66c56af`
  - `chelated_grpc_gateway.py`: `901008c848ce70a3a6e486cb817ff67e896b6400ebc743332def373a1c7d58e9`
  - `scripts/run_wave6_pm1b_v4_and_virgin_holdout.py`: `f926a5b355501203d114e63d9f78340d83ef3d41e328f73896ac9c5deed13e1b`
  - `scripts/run_wave6_d2_4096d_both_modes_3beir.py`: `0b74c7fa318361db817785e0f185094ed4021f8be249c9040f7ffcf0a774d6e0`
  - `scripts/run_wave6_pm3_v2_4way_arena.py`: `93c3edc3b6b83957efd3f7d5997b361082ab5901910e95972eadd78678910154`
  - `scripts/run_wave6_d4_trained_epistemic_grpo.py`: `5fd78fb5967535c8306742e66b23564db95bd979fdcad6ff5e0cb410c94e7691`
  - `scripts/run_wave6_d5_grpc_interceptor_gateway.py`: `b68468f16fb51c41e04be177c716d24d69b897bbedda1fbb738e1965c9868121`
- **16-Memo Frontier Council Audit Trail (`Waves 5 & 8`) & Wave-7 Investor / Solution Workbench Package:**
  - `docs/research/council-logs/wave5-*.md` (`K1–K6`) & `docs/research/council-logs/wave8-*.md` (`M1–M5`)
  - `docs/research/investor-thesis-and-competitive-moat-2026-11-23.md`, `docs/research/investor-solution-architecture-deck.html`, `demo_workbench/server.py`, `demo_workbench/index.html`, and `tests/test_workbench_server.py`.
  - Encrypted Google Drive vault addendum: `O:\backups\CHELATEDAI-COMPLETE-VAULT-2026-11-22\WAVE5-V5-OVERNIGHT-COMPLETE-ADDENDUM-2026-11-23.7z` (AES-256 + `-mhe=on`, `7z t` verified).



