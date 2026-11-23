# Wave-8 Round-2 Cross-Judgment Synthesis (`M1–M5` Binding Adjudication)

- **Date:** 2026-11-23
- **Panelists (`4/4` Completed):**
  1. `GPT-6.2 Sol XHigh` — [`wave8-r1-sol62-xhigh.md`](wave8-r1-sol62-xhigh.md), [`wave8-r2-judgment-sol62-xhigh.md`](wave8-r2-judgment-sol62-xhigh.md)
  2. `Claude Opus 5.5 High` — [`wave8-r1-opus55-high.md`](wave8-r1-opus55-high.md), [`wave8-r2-judgment-opus55-high.md`](wave8-r2-judgment-opus55-high.md)
  3. `Grok 4.7 XHigh` — [`wave8-r1-grok47-xhigh.md`](wave8-r1-grok47-xhigh.md), [`wave8-r2-judgment-grok47-xhigh.md`](wave8-r2-judgment-grok47-xhigh.md)
  4. `Gemini 4 Neon High` — [`wave8-r1-neon-high.md`](wave8-r1-neon-high.md), [`wave8-r2-judgment-neon-high.md`](wave8-r2-judgment-neon-high.md)

---

## Vote Tally (`M1–M5`)

| Adjudication Item | `Sol 6.2` | `Opus 5.5` | `Grok 4.7` | `Neon` | Binding Resolution |
|---|---|---|---|---|---|
| **`M1` (`Part VI` Registry `40/40` & Gate Labels)** | `APPROVE_WITH_GUARDRAILS` | `MODIFY` | `MODIFY` | `MODIFY` | **ADOPT WITH MANDATORY GUARDRAILS** |
| **`M2` (`Demo D5` Gateway & `Track B` 3-BEIR)** | `MODIFY` | `MODIFY` | `MODIFY` | `MODIFY` | **UNANIMOUS `MODIFY` (ALL CAVEATS BOUND)** |
| **`M3` (`Demo D4` SFT vs GRPO Attribution)** | `MODIFY` | `APPROVE_WITH_GUARDRAILS` | `MODIFY` | `MODIFY` | **UNANIMOUS SFT-DOMINATED ATTRIBUTION** |
| **`M4` (Investor Thesis, `100M` TCO, Deck, Workbench)** | `MODIFY` | `MODIFY (blocking)` | `MODIFY` | `BLOCK` | **REMEDIATE ALL P0 TCO & UI DEFECTS BEFORE SEAL** |
| **`M5` (`v5.0` Sign-Off & `PRE-RELEASE` Retention)** | `BLOCK` *(until M1–M4 land)* | `MODIFY` | `APPROVE_WITH_GUARDRAILS` | `APPROVE_WITH_GUARDRAILS` | **LAND ALL `M1–M5` EDITS, COMMIT PRODUCERS, KEEP `PRE-RELEASE`** |

---

## Binding Resolutions Applied Across Code, Registry `v5.0`, Whitepaper `v5.0`, Investor Package, & Workbench

1. **`M1` — `40/40` Verified Registry & Harmonized `PM-1..PM-5` Gate Status:**
   - Expand `bundle_registry_spec.json` and `verified_registry.json` from `35/35` (`41` side hashes) to **`40/40` bundles (`49` receipt-bound side-artifact SHA-256 hashes)**, recording `part_vi_code_provenance` (pre-run base `git_sha: 5a9bd46f2eeae2522a4a8d81017edb3c2407f936`, exact producer script SHA-256s, Qdrant image digest, and documented `K1–K5` variances) and committing all Wave-6/7 producer scripts to git.
   - **`PM-1a`, `PM-2`, `PM-5`:** Carried unchanged (`PM-1a` structured `240/240` + `D1 v2` `93.33%`; `PM-2` zero-oracle bootstrap `-5.00 pp` PASS / exact Tango `-0.0572` miss; `PM-5` `0.904 ms` mut p95 @ `100k`, benchmark-scoped).
   - **`PM-1b`:** Remains **`SCOPED / OPEN` (`assignment_gates_pass = false`)**. `Virgin-120` (`95.00%`, `1,028` toks) trails `DualGran_NoEK5` (`95.83%`, `100%` KU), does not beat `Flat_BGEM3_TM` (`91.67%`, `2,488` toks, Holm `p = 0.772`), has `4`-test Holm `p = 0.0766` vs `Legacy_MiniLM` (`86.67%`), and its `+41.67 pp` `KU` gain (`95.83%` vs `54.17%`, raw `p = 5e-5` outside the 4-test family) is matched by `Flat_BGEM3_TM` (`95.83%`, BGE-M3 embedder effect). `LME-500` (`60.60%` vs `60.80%` `DualGran`, `2,190.9` toks vs `6,088` in historical `J4` `TEST` `N=251` receipt `54a9fc475fb9`) is explicitly labeled `SECONDARY_POOLED_CONTAINS_DEV`; `LME_KU_echo_stress_r5_full_n78` (`78.21%`) beats `MiniLM` (`48.72%`) by `+29.49 pp` (`Holm p = 0.00025`) while `+5.13 pp` vs `DualGran_NoEK5` (`p = 0.3025`) is non-significant; `34/101` discrete bootstrap quantile ties are disclosed.
   - **`PM-3`:** `Suite A` (`58/60 = 96.67%`) has paired bootstrap `LCB97.5 > 0` and Tango `LCB95 > 0` vs all 4 arms while exact conservative Clopper–Pearson `LCB95 <= 0` on 3/4 arms (`-2.74 pp` `Mem0 True`, `-1.57 pp` `OpenSearch`, `-0.38 pp` `Hindsight`, `+20.55 pp` `Mem0 False`); `0.7833` is explicitly labeled a `formatted_context` `[SUPERSEDED]` provenance-substring scan (not `D5` resurrection), alongside `6,882.7` write tokens / `9.69 s` synchronous pre-ack extraction. `Suite B` (`25/40 = 62.50%`) reports **`1.56x` (`mem0 infer=False`) and `70.0x / 135.6x / 207.4x` (LLM-write arms)** fewer tokens, `521.1 ms` cross-encoder rerank read p50 (vs `4.6–13.9 ms`), and descriptive accuracy parity (`LCB97.5 <= 0`, equivalence untested).
   - **`PM-4`:** `PM-4a` `N=200` conv $\kappa = 0.8400$ (`cluster CI95 [0.7535, 0.9122]`); **`PM-4b` (`100`-item human sheet) remains the sole open human gate** keeping `PRE-RELEASE` active.

2. **`M2` — `Demo D5` Gateway & `Track B` 3-BEIR Claims:**
   - **Backend & Scope:** Explicitly state everywhere: *"project-defined OpenSearch-style `SearchService.KnnQuery` Protobuf over live containerized Qdrant gRPC `:16334` (`live_opensearch_java_cluster_claimed = false`; native Java OpenSearch cluster interoperability unverified)."*
   - **Latency & Retraction Variances:** Report `250 Hz` LAN `service_rtt` (`7.90 / 21.19 / 115.82 ms` p50/p95/p99,certified stop disabled) beside **`open_loop_e2e` (`153.75 / 251.86 ms` p95/p99)**; note direct `4096-d` baseline saturated at `218.79 Hz` (`e2e p50 = 3,309 ms`) and report the matched-grid worst-case **`+18.00 ms` incremental service p95 (`<= 25 ms` gate)**; disclose `50%` retraction Recall@10 **`0.9942` (`< 0.995` `K1` variance)**, `90%` density **`100%` exact fallback**, `k_os = 10` `0.8872` vs `k_os >= 32` `0.9988–1.000`, exact-prefix certificate condition, and Track B live `20%` uncalibrated loss (`-22.14 pp` Mode B / `-25.93 pp` Mode A alongside Wave-3 `-26.90 pp` `8edbc4df0e76`).
   - **Track B Mode B & Mode A:** Cite **`n = 500` anchors (`0.87%` `FiQA2018` / `9.65%` `SciFact` / `13.76%` `NFCorpus`)**, note pooled `Mode B` `CSLS` (`0.3779`, `+2.69 pp` vs `ZScore`) is `FiQA`-carried (`SciFact p = 0.0815`, `NFCorpus p = 0.463`), live Qdrant covers `0%/20%` (`100%` `CSLS` trails raw cosine), `4096-d` `bf16` minima `0.99947–0.99976 < 0.9999` (`K3` variance), `Mode A` `SciFact` null (`p = 0.892`), `NFCorpus` Look #2 `p0125 = -0.19 pp`, and `FiQA` positive-excluded `-2.08 pp` vs `Procrustes` (`p = 0.00745`).

3. **`M3` — `Demo D4` 3-Seed `SFT` vs `SFT + Epistemic-WFS-GRPO` Attribution:**
   - Lead with **completion-only `EK5` SFT** as the driver of the `7B` gain over `Base 7B` (`19.17%`) and `Bare Nemotron-30B` (`30.00%` `EVAL`, `0.0%` valid format; `38.33%` `D1`): `SFT 7B` scores `42.50%` `EVAL` and `100.0%` `D1`, while `16`-step `GRPO` (`44.72% ± 1.42%` `EVAL`, `99.44% ± 0.79%` `D1`) adds **no statistically significant increment over `SFT`** (`+2.22 pp` `EVAL`, `LCB97.5 = -1.11 pp`, Holm `p = 0.4495`; `-0.56 pp` `D1`, Holm `p = 0.733`; KL `0.0016`, `mean_frac_zero_std = 0.1667`, `max = 1.0`).
   - Prominently disclose `EVAL` `false_commit_rate_on_u = 87.5%` (matching `SFT` and worse than `Bare 30B`'s `80.0%`; `contested_successor = 0%–5%`), `max_new_tokens = 272` (`K4` `256` variance), `"zero entity/value overlap, shared scenario grammar"` on `D1`, and scope Moat #3 to determinate-slot `EK5` SFT supervision.

4. **`M4` & `M5` — Unified TCO Identity, Provenance Tags, & Zero Emulation Over-Attribution:**
   - Replace hardcoded `1,900` `Mem0` read tokens across Thesis, Deck, `server.py`, and `index.html` with measured **`506.65` (`Suite B`)** (showing both batched-write `ASSUMPTION` `$377k` — explicitly noting where `Mem0` is cheaper than Regime I `$527k` on `5:1` read-heavy chat! — and unbatched Suite B `4,546.4` write tokens/turn `MEASURED` `$612k`), default `OpenSearch` to **`2,178.6`** (`13,500` sensitivity toggle), include Regime II Raw Prose (with non-zero contested-review ops), add equal-offset base embedding/ANN/storage costs, separate `500`-anchor fit from `20%` backfill, specify `8 B/doc + 16 KB centroid`, tag every parameter `MEASURED_RECEIPT` vs `SENSITIVITY_ASSUMPTION`, remove unqualified `85%+` / vendor-behavior claims, re-verify SHA-256 on disk in `server.py`, and assert the arithmetic identity in `test_workbench_server.py`.
