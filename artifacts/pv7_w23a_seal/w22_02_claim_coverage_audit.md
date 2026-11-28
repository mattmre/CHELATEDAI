# W22-02: Claim-coverage audit of the 31 `verified-sealed` bundles

- Worktree: `D:\GITHUB\CHELATEDAI-PV7-SPINE`, branch `pv7/w16-core`, HEAD `7ffbd0df`. This audit was read-only. It did NOT run `verify_all_offline.py`: a concurrent run owns `artifacts/pv7_w23a_seal/verify_all_offline_7ffbd0df.log`, which was 0 bytes at audit time.
- Source of truth: `docs/research/COMPLETE_64_BUNDLE_EVIDENCE_SCORECARD.json`, where `totals.by_honest_evidence_classification["verified-sealed"] = 31`. That matches the packet's 31.
- All 31 have `source = verified_registry_v7_0` and `registry_status = SHA256_VERIFIED`, with ordinals 1-43. None of the PV7 fragment bundles #47-#64 is `verified-sealed`.
- Audit date: 2026-11-26.

## How the sealed bundles are verified

This section is read from the code, not from what the scorecard claims.

`scripts/verify_all_offline.py::run_fast_checks` (:1953) is the only entry point. These are the calls that touch the 31 sealed bundles:

1. **`verify_spec_bundles`** (:392, called at :1972) runs `verify_receipt_dir` (:290-389) on every spec bundle. It checks:
   - the receipt exists;
   - `sha256(manifest) == receipt.manifest_sha256` (:332-337);
   - the ancillary JSON, `artifact_sha256` and `artifact_hashes` sidecars match (:341-379).

   Gitignored raw `.csv`/`.jsonl` files that are **absent are silently skipped as optional** (:373-374). Raw files that are present but not listed in the receipt are only counted (:388). **No number in any manifest is recomputed.**
2. **`verify_headline_metrics`** (:1426) runs only for D6/D7/D8 (and for D9/D10/D11, which are not sealed). These checks **re-read stored manifest fields**:
   - D6 checks the stored stale-rate is 0 and that the *stored* `ndcg_at_10.means` agree across engines (:1308-1342).
   - D7 checks the stored accuracy is at least 0.90 and reads the stored Holm `reject_null` flags (:1368-1387).
   - D8 checks the stored `checks_total == 38` and similar fields (:1345-1365).
3. **`verify_d8`** (:769) is the only genuine recomputation that touches a sealed bundle. It has two parts:
   - an Ed25519 signature check plus SHA-256 chain linkage over the sealed rows (:673-727);
   - a **live subprocess replay of `demo_workbench/compliance_copilot_engine`**, which must reproduce the 3 sealed entry hashes and the head (:729-767, compared at :832).
4. **`verify_non_nda`** (:1264) does a live 56-pattern rescan of 3 files: the teaser markdown, the teaser deck and `compliance_copilot.html` (:1287). It covers only D8's "0/56 hits across HTML" sub-claim. The "20 scenario/step/role responses" part of that claim is not rescanned.
5. **Nothing else verifies these bundles:**
   - `scripts/verifiers.d/47..64_verifier.py` cover fragments #47-#64 only. A grep found **no reference to any sealed bundle directory** in them or in `scripts/audit_findings.py`.
   - `scripts/score_pm4_human_audit.py` references the PM-4 sheet, but `verify_all_offline.py` does not invoke it (PM-4b human grading is still pending).
   - `verify_part7` (`scripts/verify_all_offline.py:621`) and `verify_pristine_mirror` (:636) are also hash-only: they compare file sha256 values against a manifest and recompute nothing.
   - `tests/test_verify_all_offline.py` does load the D6 and D8 sealed manifests (copied in at :37-39, used at :244-250), but only to check that tampering is detected. No test recomputes a headline from raw rows. (An earlier draft of this report wrongly said no test loads a sealed manifest. A Tier B reviewer caught it.)

### Raw-row availability

Checked on disk in both `CHELATEDAI-PV7-SPINE` and `D:\GITHUB\CHELATEDAI`:
- **Ordinals 1-12 (9 historical bundles):** raw CSV/JSONL is **present** but **not hashed by the receipt**. The scorecard itself discloses "historical CSV unhashed". The headline could be recomputed, but the rows are not bound to the seal. They are also **untracked in git**: `is_raw_gitignored` (`scripts/verify_all_offline.py:245`) treats every raw suffix as gitignored, and `git ls-files artifacts` lists 0 `.csv`/`.jsonl` files. A fresh checkout therefore has no raw rows for any ordinal, so these files are only "recoverable" on this machine.
- **Ordinals 16-42 (20 bundles):** the receipt hashes 1-2 raw row files each, but **none of them is present in either checkout**. The verifier skips them as optional, so the headline cannot be recomputed offline.
- **Ordinals 17, 40, 41 and 43** reference no raw rows at all.

## Table

| bundle (ordinal, id) | headline (scorecard `summary_and_mandatory_disclosures`, truncated at 220 chars) | verifier file:line | class | recommended `evidence_tier` change | raw rows |
|---|---|---|---|---|---|
| 1 `s1` | Stage-0 M1-M3 frontier; 2/2 ancillary JSON hashes verified; historical CSV/JSONL unhashed in receipt. | scripts/verify_all_offline.py:409 -> :290-389 (manifest sha :332-337; sidecars :358-379; raw-absent skip :373-374) | HASH_REPLAY_ONLY | verified-sealed -> hash-sealed-unrecomputed (DOWNGRADE) | raw on disk (2), NOT hashed by receipt |
| 2 `s2` | 3-hop spectral condition blow-up vs Procrustes; historical CSV unhashed in receipt. | scripts/verify_all_offline.py:409 -> :290-389 (manifest sha :332-337; sidecars :358-379; raw-absent skip :373-374) | HASH_REPLAY_ONLY | verified-sealed -> hash-sealed-unrecomputed (DOWNGRADE) | raw on disk (1), NOT hashed by receipt |
| 4 `spherical` | 2x2 spherical centering/whitening ablation & B1-B3 backfill study. | scripts/verify_all_offline.py:409 -> :290-389 (manifest sha :332-337; sidecars :358-379; raw-absent skip :373-374) | HASH_REPLAY_ONLY | verified-sealed -> hash-sealed-unrecomputed (DOWNGRADE) | raw on disk (3), NOT hashed by receipt |
| 5 `precipitated` | P4/P6 manifold-precipitated anchor selection study. | scripts/verify_all_offline.py:409 -> :290-389 (manifest sha :332-337; sidecars :358-379; raw-absent skip :373-374) | HASH_REPLAY_ONLY | verified-sealed -> hash-sealed-unrecomputed (DOWNGRADE) | raw on disk (2), NOT hashed by receipt |
| 6 `f1_expanded` | 6-BEIR expanded Flagship #1 single-hop, low-anchor, and 4-hop suite. | scripts/verify_all_offline.py:409 -> :290-389 (manifest sha :332-337; sidecars :358-379; raw-absent skip :373-374) | HASH_REPLAY_ONLY | verified-sealed -> hash-sealed-unrecomputed (DOWNGRADE) | raw on disk (3), NOT hashed by receipt |
| 7 `f1_conformal_4hop` | 6-BEIR 4-hop conformal radius-matching validation. | scripts/verify_all_offline.py:409 -> :290-389 (manifest sha :332-337; sidecars :358-379; raw-absent skip :373-374) | HASH_REPLAY_ONLY | verified-sealed -> hash-sealed-unrecomputed (DOWNGRADE) | raw on disk (1), NOT hashed by receipt |
| 8 `f2_iq04` | IQ-04 1,500-lifecycle synthetic temporal evidence suite + ScaleMAS surrogate. | scripts/verify_all_offline.py:409 -> :290-389 (manifest sha :332-337; sidecars :358-379; raw-absent skip :373-374) | HASH_REPLAY_ONLY | verified-sealed -> hash-sealed-unrecomputed (DOWNGRADE) | raw on disk (2), NOT hashed by receipt |
| 9 `f2_hard` | Hard stress suite (5 families, Nemotron-3.5-Lightning). | scripts/verify_all_offline.py:409 -> :290-389 (manifest sha :332-337; sidecars :358-379; raw-absent skip :373-374) | HASH_REPLAY_ONLY | verified-sealed -> hash-sealed-unrecomputed (DOWNGRADE) | raw on disk (1), NOT hashed by receipt |
| 12 `f2_longmemeval_v2` | LongMemEval-S N=50 KU subset; MUST disclose Regime A clean 42/50 M4b vs 46/50 M1 beside Regime B stress. | scripts/verify_all_offline.py:409 -> :290-389 (manifest sha :332-337; sidecars :358-379; raw-absent skip :373-374) | HASH_REPLAY_ONLY | verified-sealed -> hash-sealed-unrecomputed (DOWNGRADE) | raw on disk (1), NOT hashed by receipt |
| 16 `f1_r4_comprehensive` | Post-council Flagship #1 side-by-side matrix (all original baselines + OOS/EM-2/EM-5b/A2g/Polar/A3-2x2/B4; 100% SHA-256). | scripts/verify_all_offline.py:409 -> :290-389 (manifest sha :332-337; sidecars :358-379; raw-absent skip :373-374) | HASH_REPLAY_ONLY | verified-sealed -> hash-sealed-unrecomputed (DOWNGRADE) | raw hashed in receipt (2), ABSENT on disk -> skipped |
| 17 `f2_ek5_v3_1m_powerlaw` | Post-council EK5 v3 K=4 differential oracle proofs (353,105 independent checks: 256+65,536+262,144+169 bitemporal snapshots+25,000 random ops, plus 28,561 cached pair checks), 1M power-law/hubs, & 100k service. | scripts/verify_all_offline.py:409 -> :290-389 (manifest sha :332-337; sidecars :358-379; raw-absent skip :373-374) | HASH_REPLAY_ONLY | verified-sealed -> hash-sealed-unrecomputed (DOWNGRADE) | no raw rows referenced or present |
| 18 `f2_r4_zero_shortcut_v2_and_locomo` | Post-council Zero-Shortcut v2 (N=240=8x30, EK5 v3 K=4 + Astra 4-line) + 10-conv balanced LoCoMo (N=300=10x30, blind echoes, 0 judge races, EM-1 M4e). | scripts/verify_all_offline.py:409 -> :290-389 (manifest sha :332-337; sidecars :358-379; raw-absent skip :373-374) | HASH_REPLAY_ONLY | verified-sealed -> hash-sealed-unrecomputed (DOWNGRADE) | raw hashed in receipt (2), ABSENT on disk -> skipped |
| 20 `r5b_s1_zs_v3_pm1a_closure` | PM-1a closure on flooded Zero-Shortcut v3 (N=240=8x30, 16.0 events/case > top_k=5): C4_v4_ek5_v3_blocked_telemetry=1.0000 (240/240, 8/8 families 100%) at 162.1 tokens; LCB95 = +0.4375 vs C4_v2, +0.6041 vs C2b, +0.7875... | scripts/verify_all_offline.py:409 -> :290-389 (manifest sha :332-337; sidecars :358-379; raw-absent skip :373-374) | HASH_REPLAY_ONLY | verified-sealed -> hash-sealed-unrecomputed (DOWNGRADE) | raw hashed in receipt (1), ABSENT on disk -> skipped |
| 21 `r5_s2_locomo_300_em4` | EM-4 soft telemetry routing on 10-conv balanced LoCoMo (N=300=10x30): Clean Recall@5 restored 0.610->0.630 (100% M1 parity); Clean acc M4f=0.3867 (+0.0000 vs M1, Tango -0.0126), M4g=0.4000 (+0.0133 vs M1); Blind Echo ... | scripts/verify_all_offline.py:409 -> :290-389 (manifest sha :332-337; sidecars :358-379; raw-absent skip :373-374) | HASH_REPLAY_ONLY | verified-sealed -> hash-sealed-unrecomputed (DOWNGRADE) | raw hashed in receipt (1), ABSENT on disk -> skipped |
| 22 `r5_s3_longmemeval_n500` | Full unmodified ICLR 2025 LongMemEval-S (N=500) + 78-item KU echo stress: Clean M4f=0.5480 (+0.0000 vs M1, Tango -0.0076; single-session-user N=70 0.9143 == M1), M4g=0.5580 (+0.0100 vs M1, SSU 0.9429), M4b=0.5140 (-0.... | scripts/verify_all_offline.py:409 -> :290-389 (manifest sha :332-337; sidecars :358-379; raw-absent skip :373-374) | HASH_REPLAY_ONLY | verified-sealed -> hash-sealed-unrecomputed (DOWNGRADE) | raw hashed in receipt (1), ABSENT on disk -> skipped |
| 26 `r5b_s2b_pm3_steelmanned_mem0` | PM-3 steelmanned mem0ai 2.2.1 (enable_thinking=False, 80/80 non-empty extractions): Suite A ZS v3 (N=80) C4_v4=0.9750 (0.48ms) vs U1c_steelmanned_infer=0.4750 (2,987.95ms; +0.5000, LCB95=+0.3875) & U1b_direct_qdrant=0... | scripts/verify_all_offline.py:409 -> :290-389 (manifest sha :332-337; sidecars :358-379; raw-absent skip :373-374) | HASH_REPLAY_ONLY | verified-sealed -> hash-sealed-unrecomputed (DOWNGRADE) | raw hashed in receipt (2), ABSENT on disk -> skipped |
| 27 `r5b_s4_pm4_cross_family_2x2` | PM-4a 2x2 cross-family {Nemotron-3.5-30B, Qwen2.5-7B-Instruct} generator x judge matrix: C4_v4=1.0000 in all 4 cells; ZS 800-row Cohen kappa=0.9847, combined 900-row kappa=0.9570 (CI95=[0.9376, 0.9751] >= 0.85 PASS), ... | scripts/verify_all_offline.py:409 -> :290-389 (manifest sha :332-337; sidecars :358-379; raw-absent skip :373-374) | HASH_REPLAY_ONLY | verified-sealed -> hash-sealed-unrecomputed (DOWNGRADE) | raw hashed in receipt (1), ABSENT on disk -> skipped |
| 28 `r6_pm1b_conversational_sota` | Wave-4 J4 single frozen held-out shot (LoCoMo convs 5..9 N=150 + LME SHA-256 test half N=251, 7,080 total rows, 0 API errors): held-out LoCoMo clean J4_Arch=0.4400 (+6.00 pp vs Chelated_v3=0.3800, +4.00 pp vs Legacy_M... | scripts/verify_all_offline.py:409 -> :290-389 (manifest sha :332-337; sidecars :358-379; raw-absent skip :373-374) | HASH_REPLAY_ONLY | verified-sealed -> hash-sealed-unrecomputed (DOWNGRADE) | raw hashed in receipt (1), ABSENT on disk -> skipped |
| 29 `r6_pm2_labelfree_closure` | Wave-4 J2 zero-oracle causal coupled PM-2 (N=200, 3,200 records, 0 oracle metadata reads verified from runtime trace, unified selector across all arms): CausalEMA_EK5v3=0.9750 (5-seed shuffle mean=0.9750, std=0.0; ful... | scripts/verify_all_offline.py:409 -> :290-389 (manifest sha :332-337; sidecars :358-379; raw-absent skip :373-374) | HASH_REPLAY_ONLY | verified-sealed -> hash-sealed-unrecomputed (DOWNGRADE) | raw hashed in receipt (1), ABSENT on disk -> skipped |
| 30 `r6_pm3_fair_rawtext_n100` | Stage R6 PM-3 fair identical-raw-text N=100 (60 enterprise raw-prose + 40 live LongMemEval-S full-timeline items): Chelated=0.7300 (73/100; N=60 0.9000, N=40 0.4750 at 4,311 tokens) vs mem0ai 2.2.1 infer=True=0.6900 (... | scripts/verify_all_offline.py:409 -> :290-389 (manifest sha :332-337; sidecars :358-379; raw-absent skip :373-374) | HASH_REPLAY_ONLY | verified-sealed -> hash-sealed-unrecomputed (DOWNGRADE) | raw hashed in receipt (2), ABSENT on disk -> skipped |
| 31 `r6_pm4_conversational_kappa_n200` | Wave-4 J5 isolated N=200 conversational cross-family Cohen kappa (Nemotron-3.5-30B vs Qwen2.5-7B-Instruct under v8 rubric, dual YES+NO rejected, 200/200 clean parses): kappa=0.8400 (raw agreement 92.0%, iid CI95=[0.75... | scripts/verify_all_offline.py:409 -> :290-389 (manifest sha :332-337; sidecars :358-379; raw-absent skip :373-374) | HASH_REPLAY_ONLY | verified-sealed -> hash-sealed-unrecomputed (DOWNGRADE) | raw hashed in receipt (1), ABSENT on disk -> skipped |
| 33 `phase2_wave2_d1v2_and_d2_1m` | Wave-2 D1 v2 N=60 4-node fleet (Small-3B Chelated v2=0.9333 at 95.6 ms, Mid-7B=0.9167, Hetero ZeroMu=0.9000 vs 30B Bare=0.7167 [LCB95=+8.33 pp] & Mem0 LiveInfer FullN60=0.8500 [non-significant LCB95=-6.67/-5.00 pp]) +... | scripts/verify_all_offline.py:409 -> :290-389 (manifest sha :332-337; sidecars :358-379; raw-absent skip :373-374) | HASH_REPLAY_ONLY | verified-sealed -> hash-sealed-unrecomputed (DOWNGRADE) | raw hashed in receipt (1), ABSENT on disk -> skipped |
| 34 `phase2_d2_qdrant_hnsw_csls_backfill` | Wave-4 J3 D2-4096d SciFact 384->4096 Qwen3-8B live Qdrant HNSW-DOT + calibrated backfill (FiQA/NFCorpus MISSING_4096; row identity verified min_doc_cos=0.99999994, min_q_cos=1.0, nn_top1=1.0): in-domain ZeroMu=0.5252 ... | scripts/verify_all_offline.py:409 -> :290-389 (manifest sha :332-337; sidecars :358-379; raw-absent skip :373-374) | HASH_REPLAY_ONLY | verified-sealed -> hash-sealed-unrecomputed (DOWNGRADE) | raw hashed in receipt (2), ABSENT on disk -> skipped |
| 35 `phase2_d3_multiagent_poisoning` | Wave-4 J6 Demo D3 controlled co-designed 4-agent (384d+768d) anti-poisoning bus (N=80=4x20, trusted signed tiers) + offline 1,000-episode (10,827-node) EK5-WFS-RLVR verifier audit: B4 ASR=0.0 (0/80, Clopper-Pearson UC... | scripts/verify_all_offline.py:409 -> :290-389 (manifest sha :332-337; sidecars :358-379; raw-absent skip :373-374) | HASH_REPLAY_ONLY | verified-sealed -> hash-sealed-unrecomputed (DOWNGRADE) | raw hashed in receipt (2), ABSENT on disk -> skipped |
| 37 `wave6_d2_4096d_3beir_complete` | Wave-6 Track B (M2): 0 MISSING_4096 across SciFact (5,183), NFCorpus (3,633), and FiQA2018 (57,638) (duplicate_aware_nn_top1=1.0; bf16 doc self-cosine minima 0.99947-0.99976 < 0.9999 floor). Primary Mode B (4096->384,... | scripts/verify_all_offline.py:409 -> :290-389 (manifest sha :332-337; sidecars :358-379; raw-absent skip :373-374) | HASH_REPLAY_ONLY | verified-sealed -> hash-sealed-unrecomputed (DOWNGRADE) | raw hashed in receipt (1), ABSENT on disk -> skipped |
| 38 `wave6_pm3_v2_competitor_arena` | Wave-6 Track C (M1/M4, Qwen2.5-7B blind judge): Suite A Regulated Drift (N=60; 9.69s pre-ack extraction, 6,882.7 write toks; 0.7833 [SUPERSEDED] context-substring scan): Chelated_v4=96.67% (93.33% conflict abstention)... | scripts/verify_all_offline.py:409 -> :290-389 (manifest sha :332-337; sidecars :358-379; raw-absent skip :373-374) | HASH_REPLAY_ONLY | verified-sealed -> hash-sealed-unrecomputed (DOWNGRADE) | raw hashed in receipt (1), ABSENT on disk -> skipped |
| 39 `wave6_d4_trained_epistemic_grpo` | Wave-6 Track D (M3, Spark #2 GB10, 25.375 GiB peak <= 34 GiB, 4-way disjoint splits w/ noise-retractor hashed, 0 unmasked prompt tokens, merge_and_unload KL diff 0.0, reload logit diff 0.0, matched max_new_tokens=272)... | scripts/verify_all_offline.py:409 -> :290-389 (manifest sha :332-337; sidecars :358-379; raw-absent skip :373-374) | HASH_REPLAY_ONLY | verified-sealed -> hash-sealed-unrecomputed (DOWNGRADE) | raw hashed in receipt (2), ABSENT on disk -> skipped |
| 40 `wave6_d5_grpc_interceptor_gateway` | Wave-6 Track E (M2, 43,040 open-loop queries, 0 errors, project-defined SearchService.KnnQuery Protobuf over live containerized Qdrant :16334 sha256:d277592721d2..., live_opensearch_java_cluster_claimed=false): 250 Hz... | scripts/verify_all_offline.py:409 -> :290-389 (manifest sha :332-337; sidecars :358-379; raw-absent skip :373-374) | HASH_REPLAY_ONLY | verified-sealed -> hash-sealed-unrecomputed (DOWNGRADE) | no raw rows referenced or present |
| 41 `wave11_d6_opensearch_pgvector_qdrant` | Wave-11 Tier-1 Demo D6 (L1, Spark #1 live Docker OpenSearch 3.9.0 :19200 + pgvector 0.8.7/PG17 :15432 + Qdrant 1.20.2 :16334 across SciFact 5,183 & FiQA2018 57,638): identical cross-engine nDCG@10 at 0% retraction (Fi... | scripts/verify_all_offline.py:409 -> :290-389 (manifest sha :332-337; sidecars :358-379; raw-absent skip :373-374); + stored-mean re-read scripts/verify_all_offline.py:1308-1342 (reads stored means :1326, stale==0 :1321, parity :1341); headline 0.3068/0.2563 NOT checked | HASH_REPLAY_ONLY | verified-sealed -> hash-sealed-unrecomputed (DOWNGRADE) | no raw rows referenced or present |
| 42 `wave11_d7_live_zep_graphiti_arena` | Wave-11 Tier-1 Demo D7 (L2, Spark #1 live Neo4j 5.26 + graphiti-core 0.31.2 + mem0ai 2.2.1, single seed=42, single Nemotron reader, single Qwen2.5-7B blind cross-family judge): Primary settled Suite A (N=60) Chelated_... | scripts/verify_all_offline.py:409 -> :290-389 (manifest sha :332-337; sidecars :358-379; raw-absent skip :373-374); + stored-field re-read scripts/verify_all_offline.py:1368-1387 (stored acc>=0.90 :1382, stored Holm flags) | HASH_REPLAY_ONLY | verified-sealed -> hash-sealed-unrecomputed (DOWNGRADE) | raw hashed in receipt (2), ABSENT on disk -> skipped |
| 43 `wave11_d8_compliance_copilot` | Wave-11 Tier-1 Demo D8 (L3, Compliance Copilot + Non-NDA Black-Box Mode, re-sealed at git_sha 4b1520cd after closing Sol 6.2 High's seal_receipts & --public NDA-routing blockers): 38/38 deterministic & loopback HTTP c... | scripts/verify_all_offline.py:673-727 (Ed25519 + chain over sealed rows), :729-767 + :832 (live engine replay reproduces 3 entry hashes + head), :1264-1291 (live 56-pattern rescan of HTML, :1287); 38/38 & 0/178 WFS only re-read/unchecked :1345-1365 | RECOMPUTE (partial) | keep verified-sealed for receipt-replay + Non-NDA-HTML sub-claims ONLY; annotate 38/38 checks and 0/178 WFS as hash-replay (scope-reduce headline) | no raw rows referenced or present |

## Summary counts

| class | count | bundles |
|---|---|---|
| RECOMPUTE (full) | 0 | none |
| RECOMPUTE (partial) | 1 | 43 `wave11_d8_compliance_copilot`: receipt chain, live engine replay and HTML Non-NDA rescan are recomputed; the 38/38 checks and 0/178 WFS are only re-read |
| HASH_REPLAY_ONLY | 30 | 1, 2, 4, 5, 6, 7, 8, 9, 12, 16, 17, 18, 20, 21, 22, 26, 27, 28, 29, 30, 31, 33, 34, 35, 37, 38, 39, 40, 41, 42 |
| NONE | 0 | none; every sealed bundle at least gets the receipt/manifest hash check |
| **total** | **31** | |

Two of the 30 HASH_REPLAY_ONLY bundles also get a stored-field consistency re-read: #41 (D6) and #42 (D7). That re-read is still not a recompute. D6 never checks its headline numbers at all (CSLS 0.3068 vs DEV_ZScore 0.2563 vs Raw 0.1962).

## Kill criterion applied

### Downgrade: headline backed only by hash replay (30 bundles)

`s1`, `s2`, `spherical`, `precipitated`, `f1_expanded`, `f1_conformal_4hop`, `f2_iq04`, `f2_hard`, `f2_longmemeval_v2`, `f1_r4_comprehensive`, `f2_ek5_v3_1m_powerlaw`, `f2_r4_zero_shortcut_v2_and_locomo`, `r5b_s1_zs_v3_pm1a_closure`, `r5_s2_locomo_300_em4`, `r5_s3_longmemeval_n500`, `r5b_s2b_pm3_steelmanned_mem0`, `r5b_s4_pm4_cross_family_2x2`, `r6_pm1b_conversational_sota`, `r6_pm2_labelfree_closure`, `r6_pm3_fair_rawtext_n100`, `r6_pm4_conversational_kappa_n200`, `phase2_wave2_d1v2_and_d2_1m`, `phase2_d2_qdrant_hnsw_csls_backfill`, `phase2_d3_multiagent_poisoning`, `wave6_d2_4096d_3beir_complete`, `wave6_pm3_v2_competitor_arena`, `wave6_d4_trained_epistemic_grpo`, `wave6_d5_grpc_interceptor_gateway`, `wave11_d6_opensearch_pgvector_qdrant`, `wave11_d7_live_zep_graphiti_arena`.

The suggested tier name is `hash-sealed-unrecomputed`. This is a proposal only: the name is not in the current vocabulary, and the owner chooses the label.

How each group could get back to `verified-sealed`:
- **Ordinals 1-12:** write a verifier that recomputes the headline from the raw CSV/JSONL already on disk, **and** re-seal the receipt so those rows are hashed. Today the rows are not bound to the seal.
- **Ordinals 16-42:** the hashed raw rows are missing from both checkouts. **I don't know** whether they still exist on the Spark hosts or anywhere else. Without them the headline cannot be recomputed, and these bundles should stay downgraded.

### Scope-reduce instead of downgrade (1 bundle)

`wave11_d8_compliance_copilot`: the receipt replay and the HTML Non-NDA scan are genuinely recomputed. The "38/38 checks" and "0/178 WFS oracle mismatches" sub-claims are hash-replay only and should be labelled that way.

### Pre-fix (MLR-NDCG-001) headlines needing restatement among the sealed 31

None identified.
- The sealed nDCG bundles (6, 7, 16, 34, 37, 41) are not part of the quarantine lineage.
- #41's runner uses the complete-qrels `benchmark_utils.ndcg_at_k(ranked, qrels)` (`scripts/run_wave11_d6_opensearch_pgvector_qdrant.py:28,353`).
- For 6, 7, 16, 34 and 37 I did not trace every runner's nDCG call. Whether they all used the complete-qrels form: **I don't know**. No verifier recomputes any of their nDCG values, so this is unproven either way.

## Legacy nDCG quarantine inventory (`artifacts/legacy-ndcg-quarantine-index-v2.json`)

Where the index is: it is in the PV7 worktree only. It is not on `origin/main` and not under `D:\GITHUB\CHELATEDAI\artifacts`.

What it records:
- `state = ACTIVE_QUARANTINE`.
- Defect `MLR-NDCG-001`: IDCG is built from the retrieved top-k only.
- Anchor commit `e40c0883`.
- 113 artifacts: 89 raw-run JSON, 8 aggregate JSON, 6 PNG plots and 10 markdown files.

The 8 dispositions reference 163 raw runs in total. 89 are retained and on disk; 74 are missing from the tip.

| disposition | referenced | retained & on disk | missing |
|---|---|---|---|
| base_matrix | 30 | 30 | 0 |
| calibration | 39 | 39 | 0 |
| knob_sweep | 18 | 18 | 0 |
| h4_compound | 2 | 2 | 0 |
| swap_scifact | 22 | 0 | 22 |
| swap_nfcorpus | 22 | 0 | 22 |
| post_bank_scifact | 15 | 0 | 15 |
| post_bank_nfcorpus | 15 | 0 | 15 |

### Recoverability was checked by recomputing, not assumed

The retained raw runs contain everything needed:
- Every retained run stores per-query `ranked_ids` and `relevant_ids` at baseline **and** at every trajectory cycle (`recovery.trajectory[*].metadata.query_ndcg`, present in 1,068 of 1,068 cycles).
- `relevant_ids` is the complete positive set: 965 of the 8,820 baseline queries list relevant docs outside the retrieved top-10.

Recomputing baseline nDCG@10 from those rows gave:
- **Legacy retrieved-only formula:** reproduces the stored value to within 3.3e-16 on all 89 runs, so the defect reproduces exactly.
- **Complete-qrels IDCG:** moves the stored baseline by up to 0.0037 for base/calibration, 0.0048 for knob_sweep and 0.0041 for h4. Example: 0.8276 becomes 0.8239.

### Artifact-level split

The aggregate, prose and plot mapping below is inferred from file paths, not traced through code.

- **Recoverable: 105 artifacts.**
  - all 89 raw runs;
  - 4 aggregates: `calibrated/calibration-manifest`, `calibrated/diagnostics`, `diagnostics`, `knob-sweep-manifest`;
  - all 6 plots (base and calibrated);
  - 6 prose files: `calibrated-results`, `diagnostics`, `calibrated/diagnostics.md`, `h4-compound-cycles-ablation`, `knob-sweep`, `results`.
- **Unrecoverable: 8 artifacts.** These are the 4 aggregates and 4 prose files for swap-scifact, swap-nfcorpus, post-bank-scifact and post-bank-nfcorpus, whose 74 raw runs are missing from the tip.
  - These lineages carry the "A4 real-data headline" (C3a 0.159 etc.) and the "H5 living-bank hard negative" cited in project memory.
  - They cannot be restated; they can only be withdrawn or regenerated.
- **Needs restating:** every recoverable pre-fix headline above (drift-recovery base matrix, calibration, knob sweep, H4 compound-cycles 0.005258 vs 0.236297). Each must be restated under complete qrels before any use outside `accepted_use_vocabulary`. I recomputed only the baseline; trajectory and recovery-cycle values may shift more.

### Separate finding: the defect on main is wider than the quarantine

- `origin/main:benchmark_utils.py:161` `ndcg_at_k(r, k)` still builds IDCG from `sorted(r)` of the retrieved gains.
- `origin/main:drift_recovery_metrics.py:11-20` routes through it.
- On main it is also called by `benchmark_comparative.py`, `benchmark_distillation.py`, `benchmark_evolution.py`, `benchmark_rlm.py` and `retrieval_fitness_evaluator.py`.

So the defect on main reaches beyond the 113 quarantined artifacts. How many other historical claims it affects: **I don't know**; that is outside this index's scope.

## PR #364 / `mlr-v1` triage

### Current state

- **`gh pr view 364`:**
  - title "feat(waves1-15): v6.0 + v7.0 (46/46 bundles, Lean 4 /formal + nanoda, Demos D1-D11, PRE-RELEASE)";
  - `state=OPEN`, `mergeable=MERGEABLE`, base `main`, head `fix/stage0-metric-lineage-and-m1-ladder` at `0134d59e`;
  - `isDraft=true`, `mergeStateStatus=BLOCKED`, `reviewDecision=REVIEW_REQUIRED`;
  - 55 commits, 430 files, +209,054/-256.
- **`git log --oneline -3 origin/main`** (local remote-tracking ref, not re-fetched): `e40c0883` (#353), `2bac5d3f` (#331), `46a5a538` (#337).
- **`git branch -r --contains e40c0883`:** includes `origin/main` (where it is the tip), `origin/pv7/w16-core` and `origin/fix/stage0-metric-lineage-and-m1-ladder`, among others.
- **Branch relationships:** #364's head is 55 commits ahead of `origin/main` and 0 behind. It is also an ancestor of the `pv7/w16-core` HEAD.
- **`mlr-v1` is not on main:**
  - commit `09184c83` ("canonical mlr-v1 metrics, quarantine v2, math fixes") exists only on the #364 branch;
  - `origin/main` still has the defective `ndcg_at_k`;
  - `artifacts/legacy-ndcg-quarantine-index-v2.json` is absent from `origin/main`.

### What blocks the merge

1. **Draft PR.** It must be marked ready for review first.
2. **Ruleset "MATT ONLY MAIN MERGE" (active).** It requires 1 approving review **including code-owner review**, plus resolved threads. `reviewDecision` is `REVIEW_REQUIRED`.
3. **`required_signatures`.** All 55 commits report `verification.verified=false`. A merge needs re-signed commits or a bypass (RepositoryRole 5, admin, has `bypass_mode: always`).
4. **`required_linear_history`.** The branch contains 1 merge commit, so a merge-commit merge is disallowed; squash or rebase would be needed.
5. **CI is red, but not on the merits.**
   - All 17 check runs failed within about 2 s (2026-11-25, 17:53 and 18:00 UTC).
   - The `lint` annotation reads: "The job was not started because recent account payments have failed or your spending limit needs to be increased."
   - No test, lint or BHS gate actually ran. GitGuardian also reports FAILURE.
   - The PR therefore has **no runtime CI evidence**, and the BHS Rule 5 and §4 gates have not run.

### Verdict

**#364 is not mergeable as-is.** It has no conflicts (`MERGEABLE`), but it is blocked by:
- draft status;
- missing code-owner approval;
- unsigned commits;
- the linear-history rule;
- a CI layer that never ran because of GitHub billing.

An admin bypass is technically possible. It would land 209k lines and 46 bundle claims with zero executed CI. By this audit, 30 of the 31 sealed bundles also have no recompute verifier. Under the rulebook's evidence rule that is an L4/L5 exposure.

Recommended order:
1. Fix GitHub billing.
2. Re-run CI.
3. Split the `mlr-v1` metric fix (`09184c83`) into its own small PR, so the complete-qrels fix reaches main without waiting for the 46-bundle payload.

Whether the GitGuardian failure is billing-related or a real secret finding: **I don't know**; I did not inspect it.
