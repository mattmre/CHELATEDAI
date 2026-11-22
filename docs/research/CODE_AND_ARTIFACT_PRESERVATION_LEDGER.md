# Code, Baseline, & Experimental Artifact Preservation Ledger

**Non-Destructive Engineering Policy:**  
No historical baseline, code version, whitepaper revision, or experimental artifact in `CHELATEDAI-STAGE0-NIGHT1` is ever deleted or overwritten without first locking a byte-exact SHA-256 backup snapshot. Every experimental result in `docs/research/verified_registry.json` (`27/27` bundles) maps back to an exact retrievable code snapshot below.

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
