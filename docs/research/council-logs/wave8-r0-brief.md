# Wave-8 Round-0 Council Audit Brief — Verification of All 5 `Part VI` Bundles, Investor Thesis, & Live Solution Workbench

- **Date:** 2026-11-23
- **Workspace:** `D:\GITHUB\CHELATEDAI-STAGE0-NIGHT1` (branch `fix/stage0-metric-lineage-and-m1-ladder`, base `v4.0` commit `5a9bd46f2eeae2522a4a8d81017edb3c2407f936`)
- **Pre-`v5.0` Backup Vault:** `archive/whitepaper-v4.0-pre-v5-backup/BACKUP_MANIFEST.json` (`6/6` `v4.0` files byte-exact SHA-256 verified; all `35/35` existing `v4.0` bundles untouched)
- **Binding Upstream Contract:** `docs/research/council-logs/wave5-r2-synthesis.md` (`K1–K6`)

---

## 1. Summary of Completed Wave-6 (`Part VI`) Receipt Bundles (`5/5` Complete & SHA-256 Verified)

1. **Track A — `artifacts/wave6_pm1b_v4_reranked_sota/` (`manifest_sha256: 5fcfab2b4c1c62e7babe4d8225f16b745ec61112efc1218eaeb0fc555957f6f3`):**
   - **Code:** `chelated_conversational_v4.py`, `virgin_conversational_holdout.py`, `scripts/run_wave6_pm1b_v4_and_virgin_holdout.py`.
   - **Primary Confirmatory Virgin Holdout (`Virgin-Conversational-120`, `24` disjoint conversations, cross-family `Qwen2.5-7B` blind semantic judge, `100.0%` clean JSON parses):**
     - `Chelated_v4_Reranked_EK5v3 = 95.00%` (`114/120`, `1,028` reader tokens) vs `Legacy_MiniLM_Flat_Top5 = 86.67%` (`+8.33 pp`, `lcb_975 = +1.67 pp`, cluster `lcb_975 = +4.17 pp`, `raw_p = 0.01915`, `4`-test Holm `p = 0.0766`).
     - Pre-registered `knowledge_update_echo_flood` slice (`N=24`): **`95.83%` vs `54.17%`** (`+41.67 pp`, `lcb_975 = +20.83 pp`, `p = 5e-5`).
     - Transparent null disclosures (`K2`/`K6`): `DualGran_Reranked_NoEK5 = 95.83%` (`100.0%` on `KU` slice — cross-encoder reranking absorbs turn-level echo distractions on short 14–16 turn sessions without needing `EK5`), and `Flat_BGEM3_TokenMatched = 91.67%` at `2,488` tokens (`2.42x` higher prompt cost, `+3.33 pp`, Holm `p = 0.772`).
   - **Post-`J4` Look #2 (`alpha = 0.025`, `p0125_bonf2`) & Full `LME-500` Secondary:**
     - `LME-500` clean (`secondary_pooled_full_dataset`): `Chelated_v4 = 60.60%` (`303/500`, `KU = 82.05%`) at **`2,190.9` reader prompt tokens** (`2.78x` fewer reader tokens than `v3/J4`'s `6,088`).
     - `LoCoMo 5–9` (`N=150`, Look #2): `44.67%` clean (`+6.00 pp` vs `Flat_BGEM3_TM`, `raw_p = 0.03395`, Holm `p = 0.1358`) and `42.67%` echo stress; `LME_KU_echo_stress_r5_full_n78`: `78.21%` (`+5.13 pp` over `DualGran_NoEK5 = 73.08%` [`p = 0.3025`, non-sig], **`+29.49 pp`** over `Legacy_MiniLM = 48.72%`, Holm `p = 0.00025`).
     - `PM-1b` historical R5/J4 clean `EK5` gate (`assignment_gates_pass = false`) remains openly disclosed as `SCOPED / OPEN` on clean unstructured chat.

2. **Track B — `artifacts/wave6_d2_4096d_3beir_complete/` (`manifest_sha256: dcb84026a15c4863fb4c2a0028cd14387c3ffe0d3c0087c581f28d717943f547`):**
   - **Code:** `scripts/run_wave6_d2_4096d_both_modes_3beir.py` + `tests/test_wave6_d2_both_modes.py`.
   - **Row-Identity Proof:** `0` `MISSING_4096` domains across `SciFact` (`5,183`), `NFCorpus` (`3,633`), and `FiQA2018` (`57,638`); `duplicate_aware_nn_top1 = 1.0` on all 3 domains.
   - **Primary Mode B (`4096 -> 384` True Legacy Upgrade, `n = 500` qrels-blind anchors = `0.87%` of `FiQA2018`, physical dual-shard Qdrant gRPC `20%` backfill, `n = 371` pooled `REPORT` queries):**
     - `Causal_DEV_CSLS = 0.3779` beats `DEV_ZScore = 0.3510` (**`+2.69 pp`**, `lcb_975 = +1.32 pp`, `p0125_bonf2 = +1.15 pp`, `p = 5e-5`) and `RRF(k=60) = 0.3237` (**`+5.42 pp`**, `lcb_975 = +3.81 pp`, `p = 5e-5`), with `100.0%` exact top-10 HNSW recall at `4.38–7.21 ms` p50.
   - **Secondary Mode A (`384 -> 4096` Edge-to-Cloud Up-Bridge):**
     - Beats `QueryAffine` (`Drift-Adapter`) by **`+12.36 pp` on `FiQA2018`** (`lcb_975 = +9.33 pp`, `p = 5e-5`) and **`+3.07 pp` on `NFCorpus`** (`lcb_975 = +0.21 pp`, `p = 0.033`), while matching `Uncentered Procrustes` (`LCB <= 0` disclosed).

3. **Track C — `artifacts/wave6_pm3_v2_competitor_arena/` (`manifest_sha256: 05442a849c199ad32955ce2145c1540b7bf0f78c3e56bb6392a69100894c5773`):**
   - **Code:** `scripts/run_wave6_pm3_v2_4way_arena.py` + `tests/test_wave6_pm3_v2_4way_arena.py`.
   - **Suite A (`N=60` Regulated Enterprise Drift & Retraction, cross-family `Qwen2.5-7B` blind judge):**
     - `Chelated_v4 = 96.67%` (`58/60`, `93.33%` peer-conflict abstention) statistically significantly beats **all 4 comparators** (`lcb_975 > 0`, `p0125_bonf2 > 0`, `Tango LCB95 > 0`): live `mem0ai 2.2.1 infer=True = 81.67%` (`+15.00 pp`, `lcb_975 = +5.00 pp`, `p = 0.00465`; `33.33%` abstention), `OpenSearch Memory (emulation) = 80.00%` (`+16.67 pp`, `lcb_975 = +6.67 pp`, `p = 0.00195`), `Hindsight 4-Net (emulation) = 78.33%` (`+18.33 pp`, `lcb_975 = +8.33 pp`, `p = 0.00105`; `73.33%` at `Delta=0s`), and live `mem0ai 2.2.1 infer=False = 61.67%` (`+35.00 pp`, `p = 5e-5`).
     - Explicitly disclosed in Investor Thesis: on unstructured raw prose (`AuthorityDominatedExocortexV2`), `Chelated_v4` performs one-time bounded claim extraction (`6,882.7` write prompt tokens/case vs `9,258.5` in `Mem0 infer=True`) and `formatted_context` retains `[SUPERSEDED]` audit notes (`0.7833` raw context substring scan vs `96.67%` reader decision accuracy).
   - **Suite B (`N=40` Live `LongMemEval-S` Conversational Timelines):**
     - `Chelated_v4 = 62.50%` (`80.0%` determinate) at **`0.0` write LLM tokens** (`1,756.4` total tokens) vs `mem0ai infer=False = 57.50%` (`2,743` toks), `Hindsight (emulation) = 55.00%` (`122,908` toks = **`70.0x`**), `mem0ai infer=True = 52.50%` (`364,222` toks = **`207.4x`**, `236.1s` write-ack p50), and `OpenSearch Memory (emulation) = 45.00%` (`238,167` toks = **`135.6x`**). Two-sided `lcb_975 <= 0` on `N=40` accuracy deltas is explicitly disclosed (`K6`).

4. **Track D — `artifacts/wave6_d4_trained_epistemic_grpo/` (`manifest_sha256: f20433dce33cde8b47c303ddf834046e9374d04b3a77288a8fb81f586135f28a`):**
   - **Code:** `scripts/run_wave6_d4_trained_epistemic_grpo.py` + `tests/test_wave6_d4_epistemic_grpo.py`.
   - **All `K4` Guardrails Verified on `DGX Spark #2` (`peak VRAM = 25.375 GiB <= 34.0 GiB`):** 4-way pairwise-disjoint splits (`noise-retractor-*` hashed, `0` `D1` overlap), completion-only SFT (`unmasked_prompt_token_count = 0`, `store_slot_status_leaked = false`, `merge_and_unload()` KL ref logit diff `0.0`), 3 pre-registered seeds `(42, 137, 2026)` (`16` steps, `mean_frac_reward_zero_std = 0.1667 < 0.80`, `max_reload_logit_diff_across_seeds = 0.0 < 1e-4` via `autocast_adapter_dtype=False`).
   - **Results & Honest Attribution:**
     - `EVAL` (`N=120` unseen scenarios): `Base 7B = 19.17%`, `Bare Nemotron-30B = 30.00%`, `SFT 7B = 42.50%`, **`SFT + Epistemic-GRPO 7B` (3-seed primary mean) = `44.72% ± 1.42%`** (`+14.72 pp` vs `Bare 30B`, `lcb_975 = +4.17 pp`, **Holm `p = 0.02175`**; `+25.56 pp` vs `Base 7B`, **Holm `p = 0.0014`**; incremental `GRPO - SFT` = `+2.22 pp` exact / `+0.0550` reward, `lcb_975 = -1.11 pp`, **Holm `p = 0.4495` — non-significant**).
     - Zero-Shot `D1` Enterprise Prose (`N=60`, `k=17` raw prose without `EK5` tags): `Base 7B = 8.33%`, `Bare Nemotron-30B = 38.33%` (`41.67%` frozen), `SFT 7B = 100.00%`, **`SFT + Epistemic-GRPO 7B` (3-seed mean) = `99.44% ± 0.79%`** (`best_dev = 100.0%` on all 3 seeds; **`+61.11 pp` vs `Bare 30B`, Holm `p = 0.0003`**; parity with `SFT 7B`, `-0.56 pp`, Holm `p = 0.733`). Starved `k=5`: `71.67%` vs `38.33%` `Bare 30B` (`+33.33 pp`, `lcb_975 = +21.11 pp`, `p = 5e-5`).

5. **Track E — `artifacts/wave6_d5_grpc_interceptor_gateway/` (`manifest_sha256: cba30b931812ae3f13133ff2def304cb81d363a0a7adb8ce8cba481d457bc743`):**
   - **Code:** `chelated_grpc_gateway.py`, `scripts/run_wave6_d5_grpc_interceptor_gateway.py` (`43,040` open-loop queries, `0` errors).
   - **Flagship `N=10,000`/arm `250 Hz` LAN Tail Proof (`Desktop -> 10.0.0.24`):** `Chelated_Gateway` (`:16335`, `384 -> 4096` `SearchService.KnnQuery` Protobuf over live containerized Qdrant gRPC `:16334`) achieves **`7.90 ms` service p50 / `21.19 ms` p95** (`0.22 ms` p50 egress rescore), **`8.71x` smaller ingress wire payload** (`1,888 B` vs `16,435 B`), `0 B/doc` vector egress, `8 B/doc` RAM sidecar (`100.0%` certified shortlist stop), and **`0.00%` stale resurrection across `0/10/50/90%` retraction densities** (`99.42%–100.0%` admitted Recall@10 vs `33.37%` naive post-filter starvation and `95.35%` lagged-replica stale leak at `90%`).

---

## 2. Wave-7 Investor & Workable Solution Deliverables Under Audit

1. `docs/research/investor-thesis-and-competitive-moat-2026-11-23.md`
2. `docs/research/investor-solution-architecture-deck.html`
3. `demo_workbench/server.py` + `demo_workbench/index.html` + `tests/test_workbench_server.py`

Please audit `W8-Q1` through `W8-Q5` rigorously and surface any subtle numerical, statistical, or framing defect before we seal `verified_registry.json` (`40/40` bundles), Whitepaper `Draft v5.0`, and Presentation HTML `v5.0`.
