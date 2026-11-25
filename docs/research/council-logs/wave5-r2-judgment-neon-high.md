# Wave-5 Round-2 Cross-Judgment (`gemini-4-neon`)

Read-only adjudication of Wave-5 R1 memos (`Sol`, `Opus`, `Grok`, `Neon`) plus `d2_4096d_qdrant_manifest.json:597,606` check.

## K1: Demo D5 Bidirectional gRPC Gateway & Latency — **MODIFY**
- **Contract:** Opus rightly proves request-only $\hat{q}$ over raw `DOT` omits per-doc $1/\|d_j-\mu_d\|_2$ (constant $\mu_d^\top\hat{q}$ preserves rank), reducing to *Procrustes*, not `QueryAffine`. Neon/Grok/Sol rightly show skipping egress rescoring drops raw-DOT `k_os=10` Recall@10 to `0.864` and uncalibrated mixed-shard nDCG@10 by `−26.9 pp`. **Binding:** ingress enforces fail-closed `TRA-1` checks, causal EMA $\hat{q}$, and decoupled `(k_os,hnsw_ef)`. Egress returns zero vectors—only IDs, DOT, `shard_tag`, `4B` $\|d_j-\mu_d\|_2$, `8B` causal DEV CSLS $r_d$—with Opus's certified stop $s-c<(s_{10}-c)\cdot n_{\min}/n_{\max}$ (bounded refill) before per-shard CSLS merge.
- **EK5 Fence:** Pre-filter indexed `admitted` bitsets with adaptive `ef_eff=min(ef_max,ceil(k_os/(1−r_local)))` plus exact fallback; keep `CONTESTED` distinct from `AUDIT_ONLY`; enforce egress `EK5 v3` epoch-pinned fencing (`wait=true`) across `0/10/50/90%` retraction densities.
- **Transport:** Unanimously, `2,290.8` vs `2,281.9 ms` (`+8.9 ms`) is Nemotron-30B decode. Isolate transport+gateway from decode locally and `3090→Spark` (pooled HTTP/2, no SSH in timer): REST/JSON vs Qdrant gRPC `:6334` and pinned OpenSearch `SearchService.KnnQuery` Protobuf (drop unverified "Uber" claims), `384/1024/4096-d`. Gate: Recall@10 `>=0.995`, p95 overhead `<=25 ms`.

## K2: PM-1b v4 Reranker, Percentiles & Holdouts — **ADOPT**
- **Binding:** Collapse `EK5` echoes *before* session budgeting; drop `max_sessions=6` on `KU`; preserve chronology on `SUPERSEDE` chains. Tune `bge-reranker-v2-m3` compaction (`1.2k–2.0k` tokens) strictly on `LoCoMo 0–4` and frozen `LME DEV` vs token-matched controls; SHA-256 freeze pre-eval.
- Fix `lcb_975==lcb_95` by reporting `p5.0`, `p2.5`, `p1.25`. Label `LoCoMo 5–9`/`LME TEST` **Post-`J4` Look #2 (`alpha=0.025`, burned)** beside original `J4` scores. Require a virgin, lineage-disjoint conversational holdout with a cross-family blind judge (off Nemotron).

## K3: D2-4096d 3-Domain BEIR & Dual Modes — **MODIFY**
- **Fact Check:** `d2_4096d_qdrant_manifest.json:597,606` confirms `max_abs_dot_vs_numpy=2.682209014892578e-07` (`2.68e-7`, float32 parity); Opus misread `2.68`. `D4096` row identity remains unverified.
- **Binding:** Complete `D384`+`D4096` caches for `SciFact (5,183)`, `NFCorpus (3,633)`, `FiQA2018 (57,638)` with pinned `Qwen3-Embedding-8B` SHA, left-padding, final nonpad pooling, query-only `Instruct:`, normalized `float32`, embedded text/ID hashes, and independent re-encode verification (`cos>=0.99999`, all queries + `>=256` docs, both widths).
- Test **both** **Mode A** (`384→4096` Up-Bridge) and Opus's **Mode B** (`4096→384` Legacy Upgrade, `n=500` = `0.87%` of `FiQA2018`; unmigrated docs in `384-d` HNSW, backfilled `p%` in `4096-d` HNSW), fused via causal DEV-only `CSLS` vs `DEV z-score` vs `RRF(k=60)` (primary `CSLS` vs `DEV z-score` at `p=0.2`, Holm-corrected; `qrels`-blind primary + positive-excluded sensitivity).

## K4: Demo D4 Epistemic-WFS-GRPO LoRA & D1 Transfer — **ADOPT**
- **Binding:** (1) `merge_and_unload()` SFT before fresh GRPO LoRA (`beta=0.04` anchors to SFT, dropout off); (2) disjoint `SFT_WARMUP`/`GRPO_RL` episodes (hash `noise-retractor`), logging `frac_zero_std` (abort `>0.8`), KL, `abstain_rate`; (3) `padding_side="left"`, `batch=8==num_generations=8`, pinned TRL/PEFT, explicit reward-signature check; (4) completion-only SFT without `store_slot_status` leak; (5) matched `max_new_tokens=256`, Part-1 cap `>48` tokens, AWQ `fp16`; (6) `-0.15` false-abstention penalty; (7) zero-shot `Base 7B`, `SFT 7B`, `SFT+GRPO 7B`, `Bare 30B` on unseen `D1` prose (`>=3` seeds, DEV never picks checkpoint).

## K5: PM-3 v2 Zero-Strawman 4-Way Arena — **ADOPT**
- **Binding:** Use contention-free GPU encoding; never pool `N=60` and `N=40` clocks. Separate write-ack, consolidation-lag (`Delta=0s` vs `{1,10,60}s`), read, and LLM clocks/tokens. Replace `ABSTAIN_MARKERS` with a blind cross-family semantic abstention judge (auditing `"I found N matching records"`/`"ABSTRACT"` leakage). Label `Chelated v4`, live `mem0ai 2.2.1` (`infer=True`), `Hindsight` (`Retain/Recall/Reflect`), and `OpenSearch Memory`-parity async `SEMANTIC+SUMMARY` consolidation honestly across shared external regulated-drift (OSV/NVD, SEC 10-K/A) and conversational streams.

## K6: Institutional Valuation, 100M-Event TCO & Moat — **ADOPT**
- **Binding:** Drop accuracy claims where `LCB<=0`. Publish a parameterized `100M`-event COGS/margin model from measured token/embed/ANN/re-encode rates (no invented pricing). Anchor positioning on deterministic 3-valued `WFS` compliance/auditability (`0.00%` masked resurrection, `as_of` replay, zero-LLM writes) and dual-mode migration control plane, disclosing cloud-vendor replication risk (1–2 quarter `CONTESTED` replication), bridge-fit cost, and paid-pilot conversion gates.
