# Wave-8 Round-2 Cross-Judgment — Grok 4.7 (xhigh)

**Date:** 2026-11-23. Four Round-1 memos, checked against the Wave-6 manifests and `K1–K6`. Suite B's arm is `Chelated_v4_Stratified_WFS`.

| Item | Vote |
|---|---|
| M1 Gate labels | **MODIFY** |
| M2 D5 and Track B | **MODIFY** |
| M3 D4 attribution | **MODIFY** |
| M4 Thesis, TCO, deck, workbench | **MODIFY** |
| M5 v5.0 checklist | **APPROVE_WITH_GUARDRAILS** |

## M1 — MODIFY

`PM-1b` stays `SCOPED / OPEN`. Virgin-120 is `95.00%` (`114/120`), behind DualGran-NoEK5 `95.83%` and not above Flat `91.67%` (Holm `p=0.772`). Versus MiniLM, Holm `p=0.0766` does not reject (`lcb_975=+1.67 pp`). KU `+41.67 pp` equals Flat BGE-M3 at `95.83%`; do not credit EK5. Holm `0.00025` is R5 KU stress versus MiniLM. LME-500 `60.60%` is `DO_NOT_CITE_AS_HELD_OUT`. Tokens `2,190.9` are compaction only; `6,088` is historical J4.

Suite A (`58/60`) keeps `LCB97.5>0` versus all four arms and must show Clopper–Pearson `lcb95≤0` on three. Suite B (`25/40`) keeps `LCB97.5≤0`. Print `1.56×` (Mem0 `infer=False`) and `70.0× / 135.6× / 207.4×` (LLM-write arms). Read p50 is `521 ms` versus `5–14 ms`. The `0.7833` field is a Suite A substring, including `[SUPERSEDED]`, not the D5 rate. Leave `PM-1a`, `PM-2`, and `PM-5` as carried. `PM-4b` is the only open human gate.

## M2 — MODIFY

Binding line: `SearchService.KnnQuery` Protobuf over Qdrant gRPC `:16334`, not a live Java cluster; `qdrant:latest` is unpinned. Named rescore passes at `k_os=64, ef=64` (`1.0`). Disclose `50%` admitted recall `0.994186`. At `90%`, `1.0` is full exact fallback. `250 Hz` is dispatch (`250.78 Hz`). Service RTT is `7.90/21.19/115.82 ms`; e2e is `153.75/251.86 ms` p95/p99. Incremental `−32.01 ms` is a saturated `4096-d` queue (`218.79 Hz`, e2e p50 `3,309 ms`). The honest `≤25 ms` gate is `+18.00 ms`.

Anchors `n=500` are `0.87% / 9.65% / 13.76%` of FiQA, SciFact, and NFCorpus. Pooled CSLS `0.3779` beats z-score and RRF (`lcb_975 +1.32 / +3.81 pp`) and is FiQA-carried; uncalibrated gaps are p=20 only. Mode A FiQA `+12.36 pp` stands; NFCorpus Look #2 fails; SciFact is null. Positive-excluded FiQA loses `2.08 pp` to Procrustes. bf16 minima miss `0.9999`. Drop `16/18` and `51.2×` without a receipt ID.

## M3 — MODIFY

SFT, not GRPO, carries the lift. EVAL `N=120`: SFT `42.50%`, GRPO `44.72%`, `lcb_975=−1.11 pp` (Holm `p=0.4495`), against Base `19.17%` and Bare 30B `30.00%`. D1 `N=60`: SFT `100%`, GRPO `99.44%`, `lcb_975=−1.67 pp`. State each SFT gap on its own interval. `false_commit_on_u=0.875` matches SFT and exceeds Bare `0.80`, whose `valid_format_rate=0.0`. `contested_successor` is about `0–5%`. Mean zero-std `0.1667` passes; the max is `1.0`. Length is `272`, not `256`. D1 is zero entity overlap under a shared grammar.

## M4 — MODIFY

Cell sums check: `$526,927` (`70.7%`; `$526,920` is read), `$527,014`, `$75,457` (`95.8%`), Mem0 `$794,880`, OpenSearch-setup `$4,209,980`. Rates `620/740/580` and reads `1,900` and `13,500` are not Suite measurements. Suite B reads are `1,756.4 / 506.65 / 2,743 / 2,178.6` (Chelated, Mem0 true, Mem0 false, OpenSearch). Mem0-true writes are `363,716`, not `620`. Using `506.65` on the `500M`-read column cuts Mem0 read COGS to about `$152k` and can put batched-write Mem0 under Regime I. Show that case. `620` stays assumed, and embed, ANN, storage, and bridge stay required. Tag the printed coefficients `ASSUMPTION`. Default the workbench to `2,178.6` and keep `13,500` labeled. Drop unqualified `85%+`. Sidecar is `8` bytes/doc plus `16,384` centroid bytes. State both Mem0 intervals.

## M5 — APPROVE_WITH_GUARDRAILS

Stay `PRE-RELEASE` until `PM-4b` grading. Before seal: add five Part VI bundles to the current 35, bind side hashes, and copy M1–M4 into the whitepaper, HTML, and registry. Keep `PM-1b` open. Rename the Suite A stale field. Record `272`, the D5 latency split, Mode A/B caveats, and assumption-versus-measured TCO. Do not seal on `git_sha 5a9bd46f` until that commit contains the Wave-6 producers. No drop-in OpenSearch or production-margin sentence remains.
