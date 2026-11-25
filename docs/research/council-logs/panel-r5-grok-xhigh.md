# Final 4-Model Panel Audit — Grok 4.7 (`reasoningEffort: xhigh`)

- **Thread:** `thread://v2/c5bb8594-9c79-4948-9251-bd83e8527cf6/thread:delegated-task:command%3Amcp%3A3b84a70c-9824-441c-8750-b9902399a932%3Adelegate-task%3Apanel-r5-grok-xhigh-20261122`
- **Model:** `grok` / `grok-4.7` (`reasoningEffort="xhigh"`)
- **Date:** 2026-11-22

---

**PRE-RELEASE / HUMAN AUDIT PENDING stays the correct label.** The headline receipts I re-read match the registry. The draft still over-reads three of them: the 0.718 ms figure, the “positive” clean `M4g` gains, and the `6,225×` mem0 comparison. I re-hashed seven Part IV manifests against `verified_registry.json`. All seven match. I did not re-run `scripts/build_whitepaper_assets.py --check`, and I did not re-hash the other 20 bundles or the side-artifact files. I did not open the human-audit sheet or the blind-followup fixture.

Registry content digest in the draft is `b57b05660edf7d8ca47f8dfc916e8baa76efa9f5c68a0bf4f4b29dd92e25e0dc`. That is the pre-image hash inside the JSON, not the SHA-256 of the file bytes (`3a8aac94…`).

## 1. Claim-to-receipt parity

Numbers that match the closure receipts:

| Gate | Receipt | Draft |
|---|---|---|
| PM-1a | `C4_v4 = 1.000` on `240/240`, `162.1` tokens. Closure baselines: `C0 0.0208`, `C1 0.1208`, `C2/C2c 0.1625`, `C2b 0.3375` (delta `+0.6625`, LCB `+0.6041`), `C3 0.0375`, `C4_v2 0.500`, `C4_v3 0.875`. Diagnostic `C2b` is a different run: `0.3417`. | Both figures are in the whitepaper, on different runs. The abstract uses the closure `0.3375`. |
| PM-1b | LoCoMo `M4f 0.3867` / `M4g 0.4000` vs `M1 0.3867`. LME-500 `0.5480` / `0.5580` vs `0.5480`. SSU-70 `0.9143` / `0.9429`. KU stress `M4g 0.7564` vs `M1 0.5128` (LCB `+0.1282`) and `M1c 0.7692`. | Table 4C matches. |
| PM-2 QA | `0.965` vs native `0.985` (delta `-0.020`, LCB `-0.045`), CAR `0.695` (`+0.270`, LCB `+0.200`), no-EK5 `0.170` (`+0.795`, LCB `+0.735`). | Matches `r5b_s1b`. |
| PM-3 | Default mem0 `0.0000` at `6,034.2 ms` vs `C4_v4 0.950` at `0.477 ms`. Steelman `N=80`: `0.975` vs `0.475` at `2,987.95 ms` (LCB `+0.3875`) and raw Qdrant `0.1125`. Suite B `N=30`: `M4g 0.700`, `M4f 0.6667`, mem0 `0.600` (LCB `-0.0333`), `M1 0.400`, `6,399.0 ms`. | Matches. |
| PM-4a | `C4_v4 = 1.0` in all 4 cells. κ `0.9847` / `0.957` with CI `[0.9376, 0.9751]` / open-conv `0.7414` at `87%` agreement. JSON stores `0.957`; four-decimal `0.9570` is the same rounded value. | Matches. |
| PM-5 | Oracle mismatches `0` on `256+65,536+262,144+28,561+25,000 = 381,497`. `100k` mutation p95 `0.9043 ms`. `1M` sparse p95 `7.9463 ms`, power-law p95 `7.936 ms`, power-law p99 `2,622.6 ms`. | `0.904` / `7.95` / `7.94` / `2.62 s` are fair roundings. |

Overclaims, in the order they would mislead a reader:

1. **The `0.718 ms` number is a numpy microbenchmark, and the timed path does not run EK5.** In `scripts/run_r5b_spark37c3_part2_and_hybrid_pm2.py` lines 83–101, queries are already encoded, corpus norms are precomputed, and the mask is `admit_mask[::7] = False` (`n_queries = 86`). Encoder time, BM25, `StratifiedEvidenceStore`, and the `N=200` QA path are outside that timer. Do not extrapolate `0.718 ms` to `10^8–10^10` chunks.
2. **`M4g`’s clean “gains” have negative lower bounds.** LoCoMo `M4g` vs `M1`: mean `+0.0133`, LCB95 `-0.010`, cluster low `-0.010`, Tango `-0.0142` (`M3` clean is `0.4133`; `M4g` Recall@5 is `0.615` vs `M1` `0.630`). LME-500 `M4g` vs `M1`: mean `+0.010`, LCB95 `-0.006`, Tango `-0.0069`.
3. **The pre-registered PM-1b stress bar failed on all three clauses.** Point `+24.36 pp` (`< +25 pp`), LCB `+12.82 pp` (`< +15 pp`), `M1c 0.7692 > M4g 0.7564`.
4. **`C4_v4 = 100%` is one draw, not invariant across draws.** PM-4 cells are `N=80` (`8×10`); two other `N=80` draws scored `0.950` and `0.975`.
5. **`6,225×` compares a CPU graph walk (`retrieve_c4_v4_blocked_telemetry` on pre-structured edges) to an LLM write (`Memory.add` on flattened text).** Suite B is `ku_items[:30]`, mem0 sees top-20 hybrid slice, LCB `-3.33 pp`. Stored default speedup in JSON is `12,644.2×`.
6. **`12/12 TRA-1` is an in-process ABI unit check**; dense `12/12` CAR cells are inherited from `r5_s1`; coupled QA abstain keys off `c["family"] == "mutual_refutation_contested_abstain"`.

## 2. EM-4 (`M4f` / `M4g` / `C4_v4`)
- No gold-id leak inside `route_soft_telemetry_top_k`.
- Disclose explicitly: thresholds were tuned on the three named R4 LoCoMo misses and re-scored on that same 300; `_EXTENDED_RECAP_RE` matches the six harness prefixes; same-session pairs are skipped (`sess_ids[a] == sess_ids[b]`); missing `session_id` defaults to row index; lines 296–298 are a dead loop; `C4_v4` (structured attestations) is distinct from conversational `M4f`/`M4g` (text-only).

## 3. Next Demonstration & Training Strategy
- Do not pretrain a 30B/70B base model.
- Run frozen Nemotron-3.5-30B + Qwen2.5-7B on the two GB10 boxes + one 70B-class instruct model on short H100/A100 burst as a third judge & third encoder family on an unseen raw-text corpus (same raw text for mem0, graphiti-class, and EK5; paraphrases generated by a model distinct from the judge; real encode + eligibility + top-k latency at `>= 10^6` chunks).
- **Two high-leverage 7B LoRA/RLVR adapters worth GPU hours:**
  1. **7B Edge-Proposer LoRA** emitting `REFUTE` / `SUPERSEDE` / authority edges from raw text (retiring the pre-authored attestation gap).
  2. **7B Epistemic Verifier RLVR adapter** trained on the `κ = 0.7414` disagreement slice seeded by `PM-4b` human grades.
