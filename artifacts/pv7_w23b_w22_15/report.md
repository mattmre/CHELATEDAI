# W22-15 Report: Selective-Trigger 0.5B Extraction

(Report text written by the W22-15 implementer sub-agent; transcribed into this file by the orchestrator because the harness blocks sub-agents from writing `.md` files.)

Commits on `pv7/w23b-w22-15-selective-extractor`:
- `582ff59d`: preregistration, committed alone.
- `f2afa488`: router, runner, tests, and the CPU-replay and dev-recall stage JSONs.
- `c5d4e3c9`: runner fixes, `stage_train.json`, `stage_suite_a.json` and `w22_15_results.json`.
  - The fixes: load from local snapshots, take bge-m3 from refs/pr/130, use full branch names, and correct the G3 reason text.
  - The G3 reason previously said "GPU not available", which was false. It now says the run was stopped before `stage_live62.json` was written.

Checks: ruff is clean and the 14 unit tests pass.

## Verdict: KILL

The preregistered rule returns KILL if any measured gate fails. G1 failed on measured numbers, so the result is KILL regardless of G3. The verdict is computed by `decide()` from `w22_15_results.json`.

| Gate | Requirement | Measured | Result |
|---|---|---|---|
| G1 | SELECTIVE mean e2e ≤ 1500 ms | 3607.264 ms | **FAIL** |
| G2 | SELECTIVE ≥ 57/60 and ≥ FULL − 1 | SELECTIVE 57/60 (0.95); FULL 56/60 (0.9333) | **PASS** |
| G3 | Live sham cells: on-time cited correct > 0 and zero stale commits | not measured | **UNMEASURED** |

### G1 latency (SELECTIVE)
- **Distribution:** mean 3607.264 ms, p50 3530.927 ms, p95 4560.197 ms.
- **Comparators:**
  - It misses the 500 ms stretch target.
  - It is slower than the recorded Bare 70B k5 figure of 3257.82 ms.
  - It is 28% faster than FULL (4995.807 ms).
- **Components:**

  | Component | ms |
  |---|---|
  | extract | 3473.039 (FULL: 4729.97) |
  | write_ack | 95.099 |
  | read | 38.992 |
  | route | 0.126 |
  | reader | 0.007 |

- **Why G1 failed:** SELECTIVE indexes 72% fewer documents (288 against 1020), but extraction is only 27% faster. With HF bf16 generation on the RTX 3090, extraction time is dominated by per-step overhead, so removing documents barely shortens it.

### G2 correctness
- **Stale leaks:** SELECTIVE 2, FULL 3.
- **Wrong cases:** SELECTIVE 604, 703 and 704; FULL 604, 605, 703 and 704.
- **Router:** triggered on all 60 threads, so it never skipped a whole thread. It skipped individual documents only.

### G3: unmeasured. Status of the live62 stage
- **What ran:** the live server loaded base Qwen2.5-0.5B-Instruct from the local snapshot on CUDA in bf16. torch.compile fell back to eager mode because Triton is not installed. The server handled requests.
- **Last progress line:** `reqs_served=402 last_bs=4 batch_p50_ms=2713.4 batch_p95_ms=4450.8`.
- **Outputs:** no `[live62]` arm summary line was written, and `stage_live62.json` does not exist. How far through the grid it got is therefore unknown. The log was last written at 2026-11-26 05:05:14 -0500 (`scratch/live62.log`).
- **Why it stopped:** the implementer stopped it on the orchestrator's request so the GPU would be free, since G3 cannot change the verdict.
- **GPU afterwards:** 221 MiB used, 0% utilisation, and the server process gone.

## Other stage evidence (committed)
- **CPU replay over the full #62 grid, using recorded backends:** both arms pass stage 1.
  - Sham overhead: baseline 71.994 ms, selective 36.837 ms.
  - Sham cells correct: 192/192 in each arm.
  - Router: 107,520 initial routes and 23,808 reopens. Of the reopens, 10,752 re-extracted and 13,056 reused.
- **Dev recall:** non-ASSERT trigger recall 1.0 (90/90). ASSERT trigger rate 0.2667, overall 0.8167.
- **Train:** DEV json_valid 1.0, full-tuple 0.9917, wall time 123.792 s.

## Departures
- **D1:** the LoRA was retrained, so it is not byte-identical to D9-A.
- **D2:** bf16 HF inference on an RTX 3090.
- **D3:** a deterministic rule reader replaces Nemotron.
- **D4:** Bare 70B is compared against recorded numbers only.
- **D5:** D10-B and D7 were not re-executed.
- **D6:** the #62 worker is base Qwen2.5-0.5B-Instruct.
- **D7:** the live grid was reduced to 30 cells × 24 jobs.
- **D8:** the router also acts on the full-restart comparator.
- **D9:** the existing latency definitions are kept.
- **D10:** the hardware differs from the recorded runs.
- **D11 (not preregistered):**
  - The bge-m3 weights come from `model.safetensors` on refs/pr/130 (sha256 993b2248…1e38e7e). They were combined with the main branch's config and tokenizer (5617a9f6) into `scratch/bge_m3_local`, because torch 2.5.1 cannot load the main-branch `.bin`.
  - Qwen loads from its local snapshot path because of an offline-mode bug in transformers 4.57.3.
  - The first train attempt and the first Suite A attempt failed for these reasons. Their logs are kept as `scratch/train.log` and `scratch/suite_a.log`.
- **D12 (not preregistered):** live62 was stopped before it finished, so G3 is unmeasured.
- **Contamination:** disclosed in prereg §6. The implementer had seen the Suite A text before writing the router.

## Brutal Honesty
- **L1:** none in the router or runner.
- **L2:** `decide()` returns KILL when any measured gate fails, even if another gate is unmeasured. This is the preregistered rule.
- **L3:** the CPU replay uses recorded backends by design and is labelled as replay.
- **L4:**
  - G3 is unmeasured, and this is declared.
  - The extractor is not byte-identical to D9-A (D1).
- **L5/L8/L12:**
  - `SelectiveTriggerLiveWorkerBackend` has no completed live run.
  - The unit tests use recorded replay only.
  - `scripts/verifiers.d/62_verifier.py` only checks the committed W20 bundle. It does not check this slice's outputs.
- **L11:** the existing W22-B server code catches broadly when torch.compile falls back to eager. The fallback is logged, not silent.
- **Contamination:** the router's trigger stems were written after seeing the Suite A text. G2 is therefore biased upward, which only strengthens the KILL.
- **L13:** none known. The verdict is computed mechanically.

BHS_SELF_DRAFT: 90
BHS_SELF_DRAFT_AGENT: claude-opus-5-5 (W22-15 implementer sub-agent)
BHS_TIER_B: pending
BHS_OFFICIAL: pending; at most 90, so this cannot merge
CARRY_FORWARD: finish G3 or formally drop it. The verdict does not depend on it.
DEFERRED_SCOPE:
- D10-B and D7 were not re-executed. The comparators are the recorded 26,596.47 ms and 10,373.31 ms.
- The G3 live sham-controlled #62 replay is incomplete.

LOOP_ITERATIONS: 1
OPERATOR_OVERRIDE: none

## Scratch receipt (removable, gitignored)
- **Path:** `C:\Users\mattm\AppData\Local\Temp\claude\439e695c92b40043\c9f4318c-455f-427a-844a-07562a0a3c42\scratchpad\w22-15\artifacts\pv7_w23b_w22_15\scratch`
- **Size:** 3,426,028,984 bytes (3.2 GB).
- **Contents:**
  - `bge_m3_local/`
  - `lora/` (the LoRA adapter and the merged model)
  - 2 CPU-replay manifests
  - `osv_cve_seeds.json`
  - logs: `train.log`, `train_attempt2.log`, `suite_a.log`, `suite_a_attempt2.log`, `live62.log`
  - `suite_a_dump.txt`
  - an empty file `x`

Nothing was deleted.
