# W22-15 Report: Selective-Trigger 0.5B Extraction

(Report text written by the W22-15 implementer sub-agent; transcribed into this file by the orchestrator because the harness blocks sub-agents from writing `.md` files.)

Commits on `pv7/w23b-w22-15-selective-extractor`:
- `582ff59d`: preregistration, committed alone.
- `f2afa488`: router, runner, tests, and the CPU-replay and dev-recall stage JSONs.
- `c5d4e3c9`: runner fixes, `stage_train.json`, `stage_suite_a.json` and `w22_15_results.json`.
  - The fixes: load from local snapshots, take bge-m3 from refs/pr/130, use full branch names, and correct the G3 reason text.
  - The G3 reason previously said "GPU not available", which was false. It now says the run was stopped before `stage_live62.json` was written.

Checks: ruff is clean and the 14 unit tests pass.

## Verdict: KILL under the D2 serving stack (bf16 HF `transformers`, greedy, one batched generate per case). The queue's vLLM/FP8 batched-serving premise is untested.

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
  - It ran 28% faster than FULL (4995.807 ms) in this run. **Tier B found the gap comes from block order, not from routing.** See the Tier B section.
- **Components:**

  | Component | ms |
  |---|---|
  | extract | 3473.039 (FULL: 4729.97) |
  | write_ack | 95.099 |
  | read | 38.992 |
  | route | 0.126 |
  | reader | 0.007 |

- **Why G1 failed:** SELECTIVE indexes 72% fewer documents (288 against 1020). Extraction time is dominated by per-step decode overhead, which removing documents barely touches.
  - Tier B's fit for SELECTIVE extraction is 3174 ms plus 62 ms per extracted doc.
  - Tier B re-measured with the two arms alternating case by case. Routing gave **no measurable extraction saving**: SELECTIVE 3490 ms, FULL 3441 ms.
  - The host was also slow (Tier B finding 2), so the miss looks larger than it is. Tier B estimates SELECTIVE at about 1.9–2.0 s at D9-A's recorded speed, which still fails.

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

## Tier B round 1 (fresh adversarial sub-agent a31e384fae8b7c25a; findings folded in by the orchestrator)

**What Tier B reproduced**
- All G1 and G2 numbers, from the 60 per-case rows. The components sum to the stored e2e in every case.
- All other stage numbers, and the recorded comparators.
- The preregistration has not been edited since `582ff59d`. KILL-first was already in `decide()` at `f2afa488`.
- 14 tests pass and ruff is clean.

**GPU re-measure.** About 3 minutes on an idle 3090, covering 15 cases. SELECTIVE extraction took 3490 ms, against 3560 ms originally for the same cases, so G1 reproduces.

**Important findings (folded into this report)**

1. **The "27% / 28% faster than FULL" claim comes from block order.**
   - FULL ran as a block before SELECTIVE, and extraction time drifted within each block. FULL rose from 4383 to 5077 ms; SELECTIVE fell from 3826 to 3120 ms.
   - Within SELECTIVE, each extracted doc cost only 62 ms.
   - The same 17 docs took 3856 ms in SELECTIVE and 6250 ms in FULL.
   - With the arms interleaved there is no saving. **Routing gave no measurable extraction saving on this setup.**
2. **This run was about 2.25× slower than D9-A on the same code and corpus.**
   - FULL took 4730 ms per 17-doc case; D9-A recorded 2104 ms. On DEV the gap is only 13%.
   - The GPU guard does not record CPU load. During review the host CPU was at 76%, with FastCopy and other heavy jobs running.
   - So the margin of the G1 miss is inflated. Tier B estimates about 1.9–2.0 s at D9-A's speed, which still fails.
   - A clean, unloaded host coming close to 1500 ms cannot be ruled out.
3. **The KILL was stated without scope.**
   - The queue premise is vLLM/FP8 batched serving, which D2 replaced. The per-step overhead behind G1 belongs to D2.
   - The verdict now reads "KILL under the D2 serving stack", and D2 is added to DEFERRED_SCOPE.
4. **A preregistration claim is false (L13-style).** PREREGISTRATION.md:40/124 says the cue stems are "taken from the stance definitions in EXTRACTION_SYSTEM_PROMPT_V2".
   - That is false for 9 of the 21 stems: revise, denial, denied, counter filing, retract, rescind, withdraw, amend and correction.
   - "denial" is the most-fired cue on Suite A (75 triggers), and "withdraw" fired 15 times.
   - These stems were chosen with Suite A in view, so G2's pass at exactly the 57/60 floor is optimistically biased.
   - The preregistration is not edited after the fact; this disclosure replaces that.

**Nits (folded in)**

5. `w22_15_results.json` records `git_sha` `f2afa488`. But the Suite A stage needed the runner fixes that were then uncommitted and were committed later in `c5d4e3c9`. The code at `f2afa488` could not have produced this evidence offline.
6. Prereg §5 lists the KILL and INCONCLUSIVE clauses without an order. "KILL when any measured gate fails" is one reading. It is a reasonable one, coded before any results, but it is still an interpretation.
7. The live62 server's batch p50 was 2713 ms, far above the 560 ms SLA. G3 would probably have timed out, as the earlier 0/1680 run did. "Unmeasured" should not be read as a likely pass.

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
- **L13:** the preregistration says the cue stems come from the extractor prompt. That is false for 9 of the 21 stems (Tier B finding 4). The verdict itself is computed mechanically.
- **L4 (Tier B findings 1–3):** the original report claimed a routing compute saving that disappears when the arms are interleaved. It also stated the KILL without its scope. Both are corrected above.

BHS_SELF_DRAFT: 90
BHS_SELF_DRAFT_AGENT: claude-opus-5-5 (W22-15 implementer sub-agent)
BHS_TIER_B: 85
BHS_TIER_B_AGENT: fresh general-purpose sub-agent a31e384fae8b7c25a
BHS_TIER_B_SEVERITY: important
BHS_OFFICIAL: 85 = min(90, 85). This does not merge.
CARRY_FORWARD: finish G3 or formally drop it. The verdict does not depend on it.
DEFERRED_SCOPE:
- D10-B and D7 were not re-executed. The comparators are the recorded 26,596.47 ms and 10,373.31 ms.
- The G3 live sham-controlled #62 replay is incomplete.
- The queue's vLLM/FP8 batched-serving premise is untested. D2 replaced it, and that departure decides G1.
- G1 has not been re-measured on a clean, unloaded host with CPU load recorded.

LOOP_ITERATIONS: 2
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
