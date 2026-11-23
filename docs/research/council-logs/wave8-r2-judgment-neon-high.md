# Wave-8 Round-2 Cross-Judgment Memo (`neon-high`)

**Peers:** `sol62-xhigh`, `opus55-high`, `grok47-xhigh`, `neon-high` | **Verdict:** **HOLD `v5.0` SEALING.** Five Wave-6 manifests match receipt SHA-256s, but code is untracked against `git_sha 5a9bd46f`, `wave8-r0-brief.md` misprints Track B/E hashes and `+28.21` (`+29.49 pp`), and `M4` has blocking `K6` defects.

## M1 — Part VI Registry & Gate Table (`v5.0`): `MODIFY`
Expand `verified_registry.json` (`35->40`, binding side hashes) and update `build_whitepaper_assets.py --check`. Lock gates:
- **`PM-1a`:** `PASS` (structured WFS).
- **`PM-1b`:** **`SCOPED / OPEN`** (`assignment_gates_pass=false`, R5/J4 clean `EK5` delta). Claim `2.42x–2.78x` compaction and echo-flood resilience only. Report `Virgin-120` (`95.00%`, `1,028` toks) beside `NoEK5` (`95.83%`, `100%` KU), `Flat_BGEM3` (`91.67%`, `2,488` toks, Holm `p=0.772`), and `MiniLM` (`86.67%`, Holm `p=0.0766`; `KU +41.67 pp`, raw `p=5e-5` outside 4-test family, matched by `Flat_BGEM3` `95.83%`). Name `LME-500` (`60.60%`, `2,190.9` toks, `SECONDARY_POOLED_CONTAINS_DEV`) vs J4 `TEST` (`6,088`); strike misattributed `+5.13 pp` Holm `0.00025` (`v4-NoEK5 p=0.3025`); narrow K2's tie flag (`34/101`).
- **`PM-2`:** `PASS (Bootstrap NI) / SCOPED (Tango -0.0572)`.
- **`PM-3`:** `PASS (Suite A) / PARITY + SAVINGS (Suite B)`. Suite A (`96.67%`) beats all 4 arms (bootstrap `LCB97.5>0`, Tango `LCB95>0`), disclosing Clopper–Pearson `LCB95<=0` on 3/4 arms, `6,882.7` write toks/`9.69s` extraction, and renaming `0.7833` `[SUPERSEDED]` scan. Suite B (`62.50%`) saves `70x–207x` tokens vs LLM-write arms (`1.56x` vs `infer=False`) at `LCB97.5<=0` parity (`521 ms` read p50 vs `5–14 ms`).
- **`PM-4`:** `PM-4a PASS / PM-4b OPEN` (sole human gate).
- **`PM-5`:** `PASS` (`0.904 ms` p95 @ `100k`).

## M2 — `Demo D5` & `Track B` 3-BEIR: `MODIFY`
- **D5:** Mandate `live_opensearch_java_cluster_claimed: false` everywhere (`SearchService.KnnQuery` Protobuf over containerized Qdrant `:16334`; pin digest; Java OpenSearch unverified). Frame `7.90/21.19 ms` (`250 Hz` LAN) as **service RTT** (`8.71x` smaller `384-d` ingress vs saturated `4096-d` @ `218.79 Hz`, e2e `3,309 ms`) beside `open_loop_e2e` p95/p99 (`153.75/251.86 ms`), `service p99` (`115.82 ms`), and same-dim `4096-d` incremental p95 (`+18.00 ms<=25 ms`). Disclose `50%` Recall@10 `0.9942` (K1 `0.995` variance), `90%` exact fallback (`1.0`), `k_os>=32` (`k_os=10` `0.887`), and exact-prefix stopping.
- **Track B:** Scope `n=500` anchors (`0.87%` FiQA / `9.65%` SciFact / `13.76%` NFCorpus). Disclose Mode B `CSLS` (`0.3779`, `LCB97.5 +1.32 pp` vs `ZScore`) is FiQA-carried (`SciFact`/`NFCorpus` cross zero), live Qdrant covers `0%/20%` (`100%` `CSLS` trails cosine), bf16 minima (`0.99947–0.99976<0.9999`, K3 variance), Mode A `SciFact` null (`p=0.89`), and `FiQA` positive-excluded `-2.08 pp` vs Procrustes (`p=0.00745`).

## M3 — `Demo D4` `Epistemic-WFS-GRPO`: `MODIFY`
Attribute gains over `Bare 30B` (`EVAL` `44.72%±1.42%` vs `30.00%`, Holm `p=0.0218`; `D1 k=17` `99.44%±0.79%` vs `38.33%`, Holm `p=0.0003`) **primarily to completion-only `EK5` SFT** (`42.50%` `EVAL`, `100.0%` `D1`). State `GRPO-SFT` is non-significant (`+2.22 pp` `EVAL`, `LCB97.5 -1.11 pp`, Holm `p=0.4495`; `-0.56 pp` `D1`, Holm `p=0.733`; KL `0.0016`, `max_frac_zero_std=1.0` vs `0.1667`). Disclose `EVAL` `false_commit_on_u=0.875` (vs 30B `0.80`; `contested_successor=0–5%`), 30B `valid_format_rate=0.0`, `"zero entity/value overlap, shared grammar"`, and `max_new_tokens=272` (fix `256`, K4 variance). Scope Moat #3 to determinate/SFT transfer.

## M4 — Thesis, TCO, Deck, & Workbench: `BLOCK`
Peers confirmed blocking `K6` defects: `Mem0` reads hardcode `1,900` (`$570k`) vs measured `506.65` (`Suite B`, `$152k`, cutting batched Mem0 COGS to `$377k`—below Chelated Regime I `$527k`), while `620/740/580` writes and `15%/10%` audit rates are assumptions mislabeled `"Measured"` (`server.py:305`). Fixes:
1. Align Thesis/Deck/`server.py`/`index.html`: use `506.65` Mem0 reads (batched vs `4,546` unbatched writes), tag assumptions, add equal-offset embed/ANN/storage lines, default OpenSearch to `2,178.6` (`13,500` toggle), add Regime II (`server.py`, drop `$0 ops`), specify `8 B/doc`+`16 KB` centroid, split BYO-LLM (`85%+`) vs bundled (`70.7%/95.8%`) margins, and cut `"multi-hundred-thousand-dollar"` (`<=$180` GPU migration).
2. Assert arithmetic identity (`test_workbench_server.py`), rehash before `"verified"`, label NumPy simulator/legacy-shard stop check, fix stale `index.html` metrics (Tab 2 Wave-3, Tab 4 `46.7%->33.33%`, Mem0 disclaimer), and add Deck Slide 2/3 disclosures.

## M5 — `v5.0` Sign-Off Checklist: `APPROVE_WITH_GUARDRAILS`
Retain **`PRE-RELEASE`** until `PM-4b` closes. Seal `v5.0` only after: (1) committing Wave-6/7 code, logging K1–K5 variances, and passing `build_whitepaper_assets.py --check` (`40/40`); (2) landing `M1–M4` fixes; (3) removing vendor-behavior prose on `emulation (ours)` arms and binding out-of-bundle counts (`353,105`, `16/18`, `20/20`, `TRA-1`) to receipts; and (4) keeping `PM-1b` (`SCOPED/OPEN`) and `PM-4b` (`OPEN`) in all banners.
