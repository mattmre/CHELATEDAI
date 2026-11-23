# Wave-8 Round-1 Audit Memo — Grok 4.7 (xhigh)

**Date:** 2026-11-23. Read-only audit of the five Wave-6 Part VI manifests and the Wave-7 thesis, deck, and workbench, against `K1–K6`. No GPU, ssh, or sealed-key file. Manifest SHA-256 was not recomputed.

**Verdict:** The paired tests match the receipts. Do not seal the registry or whitepaper `v5.0` until the edits below land. `PM-1b` stays `SCOPED / OPEN`.

## W8-Q1 — K1 and K3

Egress holds: `0` vector bytes and `8` bytes per document. The gateway rescores raw dots plus the norm sidecar; Qdrant uses `with_vectors=False`. Epoch-fenced stale resurrection is `0.0` at `0/10/50/90%`. Contested-as-admitted is `0.0`. Unfenced lagged-replica rates are `0 / 37.21% / 55.81% / 95.35%`. Naive `k=32, ef=32` admitted recall at `90%` is `0.3337`. `live_opensearch_java_cluster_claimed` is false: `SearchService.KnnQuery` Protobuf over containerized Qdrant `:16334`. The thesis states that. The deck headline still reads as a live OpenSearch 3.9 placement.

`Recall@10 >= 0.995` is the `k_os=64, ef=64` cell (`1.0`). All four `k_os=10` cells are `0.887` through `ef=256`. At `90%`, `p_admit=0.0961` and `exact_fallback_trigger_rate=1.0`, so the `100%` is that fallback. Flagship service RTT is `7.90/21.19` ms p50/p95; ingress is `1,888` vs `16,435` bytes (`8.71x`). Open-loop e2e p95 is `153.7` ms, service p99 `115.8` ms, and `4096-d` batch-32 service p95 about `632` ms. The `<= 25 ms` gate is incremental versus bare Qdrant. `43,040 = 11,520 + 11,520 + 20,000`, counting each batch-32 RPC as 32 queries. Live dual-shard loss is `+22.14` pp (Mode B) and `+25.93` pp (Mode A). The thesis `26.90` pp is the K1 prior.

Mode B primary, pooled `n=371`: CSLS `0.3779` vs z-score `0.3510` (`+2.69` pp, `lcb_975 +1.32` pp, `p0125 +1.15` pp, `p=5e-5`) and vs RRF `0.3237` (`+5.42` pp, `lcb_975 +3.81` pp). `0.87%` is FiQA `500/57,638`. The same 500 anchors are `9.65%` of SciFact and `13.76%` of NFCorpus. Exact top-10 is `1.0` at p=20 (`4.38–7.21` ms p50). p=0 controls are `0.9947` (FiQA) and `0.9989` (NFCorpus). Uncalibrated gaps `+4.70 / +8.82 / +11.44` pp are p=20 only. At p=50% they are not significant.

Mode A vs QueryAffine: FiQA `+12.36` pp (`lcb_975 +9.33`, `p=5e-5`); NFCorpus `+3.07` pp (`lcb_975 +0.21`, `p=0.033`, `p0125_bonf2 -0.19`, Look #2 fails); SciFact is null (`+0.29` pp, `lcb_975 -3.80`, `p=0.89`). Vs Procrustes, primary deltas are `-0.46` to `-0.75` pp with intervals through zero. FiQA positive-excluded sensitivity loses (`-2.08` pp, `ucb_975 -0.56`, `p=0.00745`). NN-top1 is `1.0`. Minimum `4096-d` bf16 cosine is `0.999467 / 0.999680 / 0.999764`, under `0.9999`; FiQA mean is `0.999885`. `16/18` and `51.2x` CAR are absent from this manifest.

## W8-Q2 — K2

Virgin-120: Chelated `95.00%` (`114/120`). Vs DualGran-NoEK5 `95.83%`, delta `-0.83` pp, `p=0.83`. Vs Flat `91.67%`, `+3.33` pp, `lcb_975 -1.67` pp, Holm `p=0.772`. Vs Legacy MiniLM, `+8.33` pp, `lcb_975 +1.67` pp, raw `p=0.01915`; 4-test Holm `p=0.0766` does not reject. LME-500 clean is `303/500 = 60.60%` at `2,190.86` tokens vs DualGran `60.80%`. Wave-6 does not store `6,088`. Post-J4 Look #2 is labeled `alpha=0.025`. One contrast passes `p0125 < lcb_975 < lcb_95`. Across 108 contrasts: 0 reversals, 7 zero-variance ties, 34 ties with `boot_std>0`. `assignment_gates_pass` is absent and stays false on the R6 receipt. `PM-1b` remains `SCOPED / OPEN`.

## W8-Q3 — K4

Four splits are pairwise disjoint; `noise-retractor-*` is hashed into each; D1 overlap is 0. Completion-only SFT: `unmasked_prompt_token_count=0`, `store_slot_status_leaked=false`. `merge_and_unload()` logit diff is `0.0`; disk reload is `0.0`. Mean `frac_reward_zero_std` is `0.1667`; the per-step max is `1.0`. Training mean KL `0.001582` is not the parity check. EVAL `N=120`: Base `19.17%`, Bare 30B `30.00%`, SFT `42.50%`, GRPO mean `44.72% ± 1.42%`. Holm-6: vs 30B `0.02175`, vs Base `0.0014`, vs SFT `0.4495` (`lcb_975 -1.11` pp). D1 `k=17`: SFT `100%`, GRPO `99.44% ± 0.79%` (seed 42 primary `98.33%`), vs 30B Holm `0.0003`, vs SFT Holm `0.733`. Starved `k=5` is `71.67%` vs `38.33%`, outside that family. `false_commit_rate_on_u` is `0.875`. `contested_successor` exact match is `0.00–0.05`.

## W8-Q4 — K5 and K6

Suite A is `58/60 = 96.67%`, determinate `97.78%`, abstention `14/15`, write tokens `6,882.67`, read `170.5`. Paired `lcb_975` is positive vs all four arms. `stale_resurrection_delta0s_immediate=0.7833` is a substring scan of `formatted_context` (length `>= 3`, including a `[SUPERSEDED]` note), not the D5 rate. Suite B is `25/40 = 62.50%`, `0` write tokens, `1,756.425` total, ratios `70.0x / 207.4x / 135.6x`. All four Suite B `lcb_975` values are `<= 0`.

Printed TCO at `$0.60/1M`, `$18/1M` embeds, `$6.50/GiB-month`, and `$25` incidents reproduces: `$526,927` (`70.7%` of `$1.8M`), `$527,014`, and `$75,457` (`95.8%`). Write rates `620 / 740 / 580` and Mem0 read `1,900` are not Suite A (`9,258.5 / 1,215.5 / 2,219`) or Suite B (`363,716 / 120,822 / 235,988`). `405` is about `6,882.7/17`. `server.py` calls those defaults measured. The workbench OpenSearch arm is only `13,500` tokens; the deck defaults to `2,178.6`. Scaling `0.0087` onto `10M` vectors treats the FiQA anchor fraction as universal. The close says `85%+` and `8B-sidecar`; the table is `70.7%` or `95.8%`, and the sidecar is `8` bytes/doc plus a `16,384`-byte centroid. `index.html` still denies an accuracy edge vs Mem0 `infer=True`. Thesis p0125 and Tango ranges were not re-checked on every arm.

## W8-Q5 — Edits before seal

1. Copy the Q1 backend, service-versus-e2e latency, `k_os>=32` recall, `90%` fallback, and `22.14/25.93` pp lines onto the thesis, deck, workbench, and whitepaper.
2. Print `0.87/9.65/13.76%` at fixed `n=500`, p=20-only uncalibrated wins, the NFCorpus Look #2 miss, the SciFact null, the FiQA sensitivity loss, and the three bf16 minima. Drop `16/18` and `51.2x`.
3. Keep `assignment_gates_pass=false` beside virgin `95%`. Print Holm `0.0766` and LME `60.60%` vs DualGran `60.80%`. Cite `6,088` from the J4 receipt.
4. Lead D4 with SFT. State GRPO−SFT `lcb_975<0`, `false_commit_on_U=0.875`, `contested_successor≈0`, and max zero-std fraction `1.0`.
5. Rename `stale_resurrection_*` before registry seal. Mark `620/1900/740/580/13500` as scenarios beside Suite B tokens. Delete `85%+`. Write `8 bytes/doc`. Default the workbench to compact OpenSearch `2,178.6` and keep `13,500`. Correct the Mem0 disclaimer. Leave `PM-4b` open.
