# Wave-8 Round-1 Peer Audit Memo (`neon-high`)

- **Date:** 2026-11-23 | **Scope:** Wave-6 `Part VI` (`5/5` bundles), Wave-7 deliverables, `K1–K6`
- **Verdict:** **CONDITIONAL GO — HOLD `v5.0` SEALING PENDING P0 FIXES.** All five Wave-6 manifests satisfy core `K1–K5` guardrails, but Wave-7 deliverables contain a blocking `K6` TCO defect (`Mem0` read tokens) and disclosure gaps requiring surgical fixes before sealing `verified_registry.json` (`40/40`) and `v5.0`.

## W8-Q1 (`K1` & `K3`): `Demo D5` & `3-BEIR` Dual-Mode

**Demo D5 (`wave6_d5_grpc_gateway_manifest.json`, `chelated_grpc_gateway.py`):** All `K1` gates pass across `43,040` open-loop queries (`0` errors): real `grpcio`/`protoc` binary Protobuf, `0 B/doc` vector egress (`with_vectors=False`), `8 B/doc` RAM sidecar (`4 B` norm + `4 B` CSLS), `100.0%` shortlist certification (`k_os>=32`), and `0.00%` stale resurrection across `0/10/50/90%` retractions and lagged replicas (`99.42%–100.0%` admitted Recall@10 vs `33.37%` naive starvation and `95.35%` stale leak at `90%`).

**D5 Gaps:** (1) `investor-solution-architecture-deck.html` (Slide 2) and `demo_workbench/index.html` (Tab 3) omit `live_opensearch_java_cluster_claimed: false` (Protobuf runs over live Qdrant `:16334`, not Java OpenSearch). (2) Flagship `250 Hz` LAN reporting cites `service_rtt` (`7.90/21.19 ms` p50/p95 vs `43.55/53.20 ms` direct `4096-d`) without noting `8.71x` ingress compression (`384-d` vs saturated `4096-d`), gateway `service p99=115.82 ms`, `open_loop_e2e p95/p99=153.75/251.86 ms`, or same-dim `4096-d` `+18.00 ms` service p95 overhead (`<=25 ms` gate passes). (3) `index.html` Tab 2 hardcodes stale Wave-3 SciFact numbers (`0.6082` vs `0.5399`; `k_os=10` `86.40%`) instead of Wave-6 3-BEIR (`0.3779` vs `0.3510`) and D5 (`88.72%`).

**Track B (`wave6_d2_3beir_manifest.json`):** `0` `MISSING_4096` domains; `duplicate_aware_nn_top1=1.0` across `SciFact/NFCorpus/FiQA2018` (`bf16` `mean cos>=0.99989`, `min=0.99947`). Mode B (`0.87%` FiQA anchors, `20%` Qdrant gRPC, `100.0%` exact top-10 HNSW) pooled `n=371` `CSLS=0.3779` beats `DEV_ZScore` (`+2.69 pp`, `lcb_975=+1.32 pp`, `p0125=+1.15 pp`, `p=5e-5`) and `RRF` (`+5.42 pp`, `p=5e-5`). **Missing caveats:** (a) per-domain `20%` `CSLS-ZScore` significance is driven by `FiQA2018` (`+3.35 pp`, `p=0.00015`; `SciFact` `+3.74 pp`, `p=0.0815` and `NFCorpus` `+0.43 pp`, `p=0.463` cross zero); (b) at `100%`, `CSLS` trails raw cosine (`-0.51` to `-0.90 pp`; disable post-migration); (c) Mode A omits `SciFact` null (`+0.29 pp`, `p=0.892`) and `FiQA2018` positive-excluded regression vs Uncentered Procrustes (`-2.08 pp`, `p=0.00745`).

## W8-Q2 (`K2`): `PM-1b v4` & Virgin Holdout

All 8 `K2` flags pass (`frozen_dev_config.json` locked pre-Look #2). `p1.25 < p2.5 < p5.0` separates on well-powered contrasts (`Virgin` `v4 vs Flat_BGEM3`: `-2.50% < -1.67% < -0.83%`; `v4 vs MiniLM`: `+0.83% < +1.67% < +2.50%`), with expected `1/N` ties on tiny-discordance slices. Secondary `LME-500` hits `60.60%` (`KU=82.05%`) at `2,190.9` tokens (`2.78x` below `v3/J4`'s `6,088`), while held-out `LME_TEST` Look #2 (`N=251`) is parity (`60.16%` vs `60.96%`, `p=0.787`) and `LoCoMo 5–9` (`+6.00 pp`, Holm `p=0.1358`) is non-significant. On `LME_KU_echo_stress_r5_full_n78`, `Chelated_v4` (`78.21%`) beats `MiniLM` (`48.72%`) by **`+29.49 pp`** (fix `wave8-r0-brief.md` L20's `+28.21 pp` typo).

On `Virgin-Conversational-120` (blind `Qwen2.5-7B`, `100%` parses), `Chelated_v4=95.00%` (`1,028` toks) trails `DualGran_NoEK5=95.83%` and does not beat `Flat_BGEM3_TM=91.67%` (`2,488` toks, Holm `p=0.772`) or overall `MiniLM=86.67%` after 4-test Holm (`p=0.0766`), though `KU` echo-flood (`95.83%` vs `54.17%`, `+41.67 pp`, `p=5e-5`) is decisive. **Verdict:** Keep `PM-1b` **`SCOPED / OPEN` (`assignment_gates_pass=false`) in `v5.0`**—claim `2.42x–2.78x` compaction and echo-flood resilience only.

## W8-Q3 (`K4`): `Demo D4` 3-Seed `Epistemic-WFS-GRPO` + `D1`

All `K4` gates pass (`25.375 GiB` VRAM): 4-way split disjointness (`noise-retractor-*` hashed, `0` `D1` overlap), completion-only SFT (`unmasked_prompt_token_count=0`, `store_slot_status_leaked=false`), `merge_and_unload()` KL ref diff `0.0`, 3 seeds `(42,137,2026)`, `mean_frac_reward_zero_std=0.1667 < 0.80` (batch `max=1.0`), and `0.0` reload logit diff (`autocast_adapter_dtype=False`). `SFT+GRPO 7B` beats `Bare 30B` on unseen `EVAL` (`44.72%±1.42%` vs `30.00%`, `+14.72 pp`, Holm `p=0.02175`) and zero-shot `D1` (`99.44%±0.79%` vs `38.33%`, `+61.11 pp`, Holm `p=0.0003`; `k=5`: `71.67%`, `+33.33 pp`), while **incremental `GRPO-SFT` is non-significant** (`+2.22 pp` `EVAL`, Holm `p=0.4495`; `-0.56 pp` `D1`, Holm `p=0.733`) and `EVAL` `false_commit_rate_on_u_mean=87.5%` (`contested_successor=0%–5%`). **Defect:** `investor-thesis` L75 cites `max_new_tokens=256`, but code (`run_wave6_d4_trained_epistemic_grpo.py` L101) and manifest matched all arms at **`272`**.

## W8-Q4 (`K5` & `K6`): `PM-3 v2` & Wave-7 Deliverables

Suite A (`N=60`) passes `K5`: `Chelated_v4=96.67%` (`93.33%` abstention) beats all 4 arms (`LCB97.5>0`). Suite B (`N=40`) hits `62.50%` at `0.0` write tokens (`70x–207x` fewer tokens) with `K6` parity disclosed (`LCB97.5<=0`); footnote Suite B ran `ChelatedConversationalV3` + inline reranker (`6800` chars), not `V4` (`8400`).

**Blocking `K6` Defects:**
1. **Wrong `Mem0` read-token input:** Thesis (L98), Deck (L494), `server.py` (L309), and `index.html` (L1044) hardcode **`1,900`** read tokens (`$570k/yr`), contradicting Track C's measured **`506.65` (Suite B) / `371.35` (Suite A)**. At `506.65`, `Mem0` read COGS drops to `$152k/yr` and total COGS falls from `$795k` to **`$377k`** under batched `620` write tokens—below Chelated Regime I (`$527k`) unless unbatched Suite B writes (`4,546/turn`) or Regime II (`$75k`) are compared.
2. **Untagged heuristics & missing lines:** Mark `620/740/580` write tokens and `15%/10%` incident rates illustrative; show (or label equal-offset) embedding, ANN compute, and base storage.
3. **Workbench strawman/staleness:** `server.py` (L313) and `index.html` (L1046) default OpenSearch to `13,500` read tokens (`$4.21M`) instead of Deck's `2,178.6`; `index.html` L632 cites stale `"46.7% D1 abstention"` vs `33.33%`.
4. **Margin contradiction:** Thesis L120 claims `85%+` gross margin vs Section 4.1's `70.7%` Regime I bundled-LLM margin—split BYO-LLM software (`85%+`) from bundled-LLM (`70.7%`/`95.8%`).
5. Deck Slide 3 omits Suite A's raw-prose extraction cost (`6,883` toks) and `[SUPERSEDED]` audit-note disclosure.

## W8-Q5: Pre-Seal Remediation Checklist

- **P0 (Blocking):** (1) Recompute TCO across Thesis, Deck, `server.py`, `index.html` with `Mem0=506.65` read tokens, batched/unbatched write sensitivity, equal-offset infra lines, and split BYO-LLM/bundled margins. (2) Default OpenSearch in `server.py`/`index.html` to `2,178.6` (optional `13,500` toggle); refresh `index.html` Tab 4 (`33.33%`) and Tab 2 (Wave-6 3-BEIR/D5). (3) Add OpenSearch-over-Qdrant disclosure to Deck Slide 2 and `index.html` Tab 3, plus Suite A footnote to Slide 3. (4) Fix `256->272` (Thesis L75) and `+28.21->+29.49 pp`.
- **P1 (`v5.0`):** Keep `PM-1b` `SCOPED / OPEN`; include D5 `open_loop_e2e` p95/p99, same-dim `+18.00 ms`, Track B per-domain/Mode A caveats, `D4` `87.5%` false-commit rate, and exact Track E SHA (`cba30b...457bc743`) in `verified_registry.json`.
