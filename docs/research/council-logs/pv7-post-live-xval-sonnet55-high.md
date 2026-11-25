# PV7 Post-Live Cross-Validation: Wave 19 (Bundles #56-#60) and Fragment Integration

Reviewer: Claude Sonnet 5.5 (adversarial BH Rulebook v3.3 review sub-agent, effort high)
Target: `D:\GITHUB\CHELATEDAI-PV7-SPINE`, HEAD `25ae2067`, branch `pv7/w16-core`
Date: 2026-11-25
Scope rules honoured: the parent workspace was not modified; no repo file other than this memo was written; no `*sealed_key*` or `*blind-followup*` file was opened; every run used `CUDA_VISIBLE_DEVICES=""`. Scratch work (probe scripts, a throwaway clone for the rebase trial) lives only in the session scratchpad.

Premise: every completion claim is false until runtime evidence says otherwise.

---

## 0. Verdict

| Item | Verdict |
|---|---|
| #58 Cold-OSD "scattered" win | **Not supported as written.** The write-byte win depends on an asymmetric journal codec. Under a symmetric codec the scattered layout loses by 720 B. |
| #56, #59, #60 | Honest local-drycheck / floor-tier. No live receipt. Registry says `protocol-ready`. |
| #57 | A real `run-live` receipt exists (Spark, Linux aarch64, cgroup enforced). The registry fragment still says `protocol-ready`, so status text lags the evidence. |
| #59 burn gate | Passes on a **partial, self-derived inventory** and goes through the legacy deprecated path. |
| Fragment loaders | The three named loaders **do not load the 47..64 fragments at all**. "64/64" is not true for those loaders. |
| Rebase onto STAGE0-NIGHT1 + V7-FORMAL-PM6 | Textually clean in a scratch trial. No PV7-attributable regression found. |
| Tests | `pytest tests/test_pv7_w19* tests/test_pv7_w16* -q`: **89 passed, 3 warnings**. |

**BHS_REVIEWER (Tier B) = 82/100. Does not merge as a "complete" claim (BHS_OFFICIAL must be 100).** The score is capped by one L4/L2-class finding (#58 metric dependence) and two L13-class findings (loader/spec drift, registry status drift). All are cheap to fix. See section 9.

---

## 1. Bundle #58 Cold-OSD: the WAL format choice

### 1.1 What the author flagged

Remediation commit `d68f2d40` admits that the scattered layout "only passes the new write-byte gate because of a WAL format choice I made." I traced it to `chelated_cold_osd_scanner.py`.

### 1.2 The two journals are not the same codec

Cold-OSD (`EK5ExtentFence._log`) writes `_fence_wal_record(kind, epoch+1, ids)`. The record format is:

`EK5FWAL1` header (8 B once) | `kind` (1 B) | `epoch` u64 | `count` u32 | `ids` u32[] | `crc32`

It is fsynced per record. `reinstate` calls `_log(b"R")` with **no ids**. This is the cheap record.

The SQ8 challenger (`MatchedIVFSQ8._journal`) writes `kind + len(ids) u32 + ids u32[]`, fsynced. It is called with ids for **W, C and R**, and `active_ids != set(ids)` is enforced. It has no CRC, no epoch and no recovery path.

So the arms differ in three ways:
1. Cold-OSD's R and C records carry no ids; SQ8 carries them. That is a modelling difference (Cold-OSD keeps reinstatement in the fence, SQ8 must journal the id set to be replayable), but it is the whole source of the margin.
2. Cold-OSD pays an 8 B header; `ColdOSD.__init__` sets `written = 0 if fence.recovery is not None else fence.wal_bytes`. On the recovered path the header is excluded from the accounting. On a fresh path it is included. The accounting rule depends on which constructor path ran.
3. Cold-OSD records carry CRC + epoch (more durable, so it should cost more). SQ8 has neither, so it is cheaper per record and not equivalently durable. The comparison is not durability-matched.

### 1.3 Probe results (scratchpad, reproducible: `sym_probe.py`)

I re-ran `drycheck()` with a symmetric codec (R/C records carrying no ids on both arms):

| Case | Cold-OSD total | SQ8 total | Saved | LCB95 | Gate |
|---|---|---|---|---|---|
| Baseline (as committed), clustered | 2940 B | 25443 B | 22503 B | 20136 B | pass |
| Baseline, scattered | 2940 B | 4140 B | 1200 B | 1200 B | pass |
| Symmetric codec, clustered | n/a | n/a | 18855 B | 16680 B | pass |
| Symmetric codec, scattered | n/a | n/a | **-720 B** | **< 0** | **FAIL** |

Under a symmetric codec the clustered layout still passes, and the overall drycheck fails because of the scattered layout. The committed `compare_traces` pass condition is `saved > 0 and lcb > 0 and matched and zero_stale and visibility`, so a 1200 B margin on 16-id traces rides entirely on the codec choice.

### 1.4 The bootstrap is vacuous on scattered

The scattered LCB95 equals the point estimate (1200 = 1200). Every trace pair has the same byte count, so the 10,000-resample paired bootstrap has zero variance. The statistical gate adds no evidence there. The pass condition is a deterministic arithmetic comparison, not a distributional one. Describing it as "LCB95 > 0" overstates what was tested.

### 1.5 Is it legitimate engineering or metric gaming?

Both parts are true, and the disclosure is what separates them:

- Legitimate: a compact binary record is a genuinely better design than verbose JSON per-id fsync. Not journalling ids for a reinstatement is defensible when an in-memory fence holds the denial bits.
- Not legitimate as presented: the write-byte win is attributed to the *architecture* (immutable extents + denial-bit fence) when on the scattered layout it is attributable to a *record-format choice* that was not applied to the challenger. The challenger was also given a weaker durability contract (no CRC, no epoch, no recovery), which makes its records cheaper while making it slower to be safe in a real deployment. These two effects pull in opposite directions and nothing in the manifest nets them out.

L-mapping: **L4 (partial-as-complete)** for the unqualified "strictly fewer write bytes on both layouts" claim; **L2-adjacent (escape conditional)** because the gate passes only under a specific codec choice; **L13** for the "LCB95" label on a zero-variance sample.

### 1.6 Live-mode caveat (not yet evidenced)

The live runner (`scripts/run_pv7_w19_2c_cold_osd_9m.py`) accounts in 512-byte device sectors through optimizer and device quiescence (`BlockWriteProbe`). At 512 B granularity a record of 8-30 B is rounded up to a whole sector. The per-record difference that decides the scattered result (tens of bytes) will be **quantised away**: both arms will write one sector per fsynced record. The live result will be decided by *record count*, not record size. That is a different experiment from the drycheck. There is no live #58 receipt in the repo (`artifacts/pv7_w19_2c_cold_osd_9m/` holds only `pv7_w19_2c_drycheck_manifest.json`; the registry status is `protocol-ready`). The live result is unknown, and the drycheck margin should not be extrapolated to it.

### 1.7 Recommended fix (do all three)

1. **Symmetric accounting.** Give both arms the same record schema (kind + epoch + count + ids + crc32) and the same fsync discipline, or give both a shared `journal_record(kind, ids)` function in one module. Then report three rows: as-built Cold-OSD vs as-built SQ8, symmetric-codec Cold-OSD vs symmetric-codec SQ8, and a "minimum-viable-durability" row. Gate on the symmetric row. Report the as-built row as supplementary.
2. **Mandatory disclosure in the manifest and registry fragment.** Add `"codec_symmetry": {"as_built": ..., "symmetric": ...}`, plus the sentence "scattered-layout write-byte advantage is codec-dependent: -720 B under symmetric records" in `tier_disclosure`. Without it, the manifest overstates.
3. **Replace the LCB95 on deterministic traces** with either a per-trace randomised id set (so the bootstrap has variance) or an honest label ("deterministic margin, no sampling variance"). Add a header-inclusion rule that applies on both the fresh and recovered constructor paths.

Until (1) and (2) land, the #58 claim must be worded as "fewer write bytes on the clustered layout under both codecs; scattered layout parity not established."

---

## 2. Bundle #56 Disk-MoE

- `artifacts/pv7_w19_2a_disk_moe/pv7_w19_2a_drycheck_manifest.json`: `mode = local-drycheck`, `passed` true, `exact_fixture_parity` recorded, `io_method`, `extent_count`, `rungs`, accepted vs unaccepted draft tokens.
- It carries **no `tier` and no `tier_disclosure` field**, unlike #57, #59, #60. A reader cannot tell floor from ceiling without opening the runner. L8-adjacent: add `tier: "floor"` and a disclosure matching #60's.
- Registry: `protocol-ready`. No live receipt. Honest as local-drycheck, not honest if surfaced as a performance result.
- Test: `tests/test_pv7_w19_2a_2c_storage.py` passes (an 8-row memory-budget assertion at line 74 checks a budget partition, not throughput).

## 3. Bundle #57 250-payload cartridge array and live receipt

- `artifacts/pv7_w19_2b_cartridges/pv7_w19_2b_manifest.json` and `report_receipt.json`: `mode = run-live`, `overall_pass = true`, tier `ceiling-candidate`, generated `2026-11-25T11:32:15Z`, platform `Linux-6.17.0-1029-nvidia-aarch64` (Spark), `cuda_visible_devices = ""`.
- Cage is genuinely enforced: `MemoryMax=6442450944` (6 GiB), `MemorySwapMax=0`, `enforced: true`, "limits match".
- Swap evidence: `cold_posix_fadvise` tier, measured bandwidth 2.29 GB/s, integrity failures 0, overhead ratio vs bound 3.72 (gate requires <= 50, so the threshold is loose), median first usable inference 13.6 ms.
- Revocation: 300 distinct payloads raced, Clopper-Pearson one-sided 95% upper bound 0.00125 for zero events (gate < 1%), EK5 semantic diffs 0.
- **Findings:**
  - The registry fragment `registry.d/57-t2b-cartridge-array.json` still says `protocol-ready`. A live receipt exists. This is status drift (L13). Either the fragment is promoted with a receipt reference, or it states why the receipt does not promote it.
  - The overhead gate (<= 50x the physical bound) is a very loose ceiling for a 3.7x measured result. A reviewer should ask who chose 50 and when relative to the first run. It does not invalidate the result; it weakens "ceiling-candidate."
  - `cuda_hidden` and `cp_upper_bound_below_1pct` are the strongest gates; the 1% CP bound with only 300 zero-event trials is the honest ceiling on the revocation claim (about 0.12%).
  - The live receipt is for #57 only. It does not transfer to #58.

## 4. Bundle #59 Sub-1GiB IQ-04 + MemOS

- Manifest: `mode = local-drycheck`; cage `enforced_on_this_host: false` ("local-drycheck: no cage is applied"); `strictly_under_one_gib: true` is a claim about the **configured** limit (1,006,632,960 B), not a measurement under enforcement. The traced peak is the only measured number, and it is a Python allocation trace (`traced_peak_bytes`), not RSS. L5: the memory claim is untested on the production path.
- **Burn gate (`chelated_cpu_memory_appliance.py` around line 617).** `assert_families_burn_clean` calls `burn_ledger.assert_corpus_unburned(...)` without `allow_local_drycheck_smoke`. That selects the legacy `None` path, which emits a `DeprecationWarning` (this is the warning visible in the pytest run) and stamps the result `local_drycheck_smoke`, never confirmatory-eligible.
  - The "historical inventory" is built by `worktree_local_inventory` from the *string values in `registry.d/*.json`* with sentinel fallbacks. The manifest shows `inventory_documents: 85`, `inventory_queries: 10`, against `documents_checked: 750`, `queries_checked: 200`. So the gate checked 750 documents against an 85-document inventory the worktree generated about itself. A burn check against an inventory the checked project derives from its own fragments cannot detect corpus reuse from earlier campaigns. It is a self-consistency check, correctly stamped smoke.
  - `burn_clean: true` in the manifest is therefore not evidence of an unburned corpus. Label it `burn_clean_local_inventory_only`.
  - Fix: pass `allow_local_drycheck_smoke=True` explicitly in drycheck mode (removes the warning and states the intent) and `False` in `run-live` so run-live fails closed if the historical inventory is partial. `run-live` already requires `--historical-inventory`; assert that the file's document count is >= the ledger's recorded count rather than trusting its presence.
- No live Spark receipt for #59. Registry `protocol-ready`. Honest.

## 5. Bundle #60 Causal-prefix + weight-SHA SSD KV cache

- Manifest: `mode = local-drycheck`, `overall_pass = true`, tier `floor`, disclosure "synthetic git histories and a tiny deterministic 2-layer numpy model, not an LLM on real repos; this checks the binding/invalidation mechanism, not production speedup". That wording is exactly right. Keep it, and do not let release notes drop it.
- **Weight-SHA omission test (MemT-1.0 `modeling_memt.py:876` pattern).** The test `test_stale_weight_revision_error_on_patch_and_adapter` (line 334) and `test_direct_reuse_raises_stale_weight_revision` (line 343) confirm that the cache key binds `base_weights_sha256` and `adapter_weights_sha256` (`chelated_ssd_kv_repo_cache.py:222-223`) and that a patched base or adapter raises `StaleWeightRevisionError`. `weight_revision_probe` reports `base_weights_changed` and `adapter_changed` (lines 584-585).
  - Gap: the tests prove the key *changes* and a raised error on a reuse path. They do not prove that a cache **written without** a weight digest (the MemT-1.0 omission case) is rejected at read time. Add a negative test that hand-builds a cache entry lacking the digest fields and asserts it is refused, not treated as a miss that silently repopulates.
  - The challenger-control row ("serves stale after retraction control") confirms the challenger fails where Cold prefix invalidation holds, which is the intended discriminator. It uses the same synthetic model, so the 2-layer numpy scope applies to both arms.
- No live Spark receipt for #60. Honest.

## 6. `computational_storage_poc/*.py` EXPERIMENTAL flag

- `computational_storage_poc/_experimental.py` provides `mark_experimental`, called at import time. It reads each module's `EXPERIMENTAL` constant and emits one `ExperimentalPOCWarning` (a `DeprecationWarning` subclass).
- Modules that import and call it: `cpu_backends`, `integrated_repo_runtime`, `moe_reap`, `packed_cpu_inference`, `packed_graph`, `phase7_system_evaluation`, `pool_shard`, `repo_graph_memory`, `sparse_cpu_inference`, plus the helper itself. That is **10 of 32** `.py` files in the directory matched `EXPERIMENTAL`. The other 22 do not carry it. Confirm which of the remaining 22 are in scope (the older Session 22-27 POC files may be intended to be flagged too). If the intent is "all POC files," this is partial (L4).
- Because it is a `DeprecationWarning` subclass, Python's default filters hide it outside `__main__` and test runners. A user importing a POC module from library code will not see it. Consider `UserWarning`/`FutureWarning` so the marker is visible by default.

## 7. Fragment loaders and zero-collision claim

- `scripts/verify_all_offline.py`, `scripts/build_whitepaper_assets.py` and `demo_workbench/server.py` contain **no reference** to the 47..64 fragments. `verify_all_offline.py` reads `docs/research/bundle_registry_spec.json` (40 bundles with string ids; no PV7 bundles). `build_whitepaper_assets.py` reads the same via `SPEC_PATH`. `server.py` has no `routes.d` handling, so `demo_workbench/routes.d/61-fabric.json` (`/fabric`, `/api/v2/fabric/*`) is never served.
- The only fragment consumer is `chelated_pv7_spine/fragment_compiler.py` (`FRAGMENT_DIRS`: `docs/research/registry.d`, `demo_workbench/routes.d`, `scripts/verifiers.d`; `BUNDLE_IDS = range(47, 65)`; pinned `SPINE_DIGEST`), reached by the PV7 verifiers themselves (for example `47_verifier.py` calls `compile_repo_fragments`).
- Consequences:
  1. "Zero-collision" is trivially true for these loaders: they read none of the fragments, so there is nothing to collide with. It is not evidence of a working integration (L13: prose claiming a mechanical property).
  2. `46 base bundles + 18 fragments = 64/64` holds as a **count of files**, not as a count a single loader reports. No script I ran emits 64.
  3. Schema mismatch is latent: spec ids are strings, fragment `bundle_id` values are integers (47..64). The first loader that merges both will either mis-sort or collide on type. Decide the id type now.
- Recommendation: add one `scripts/` function (`load_all_bundles()`) that merges spec + fragments with an explicit id-normalisation rule, have the three loaders call it, and add a test asserting `len == 64`, id uniqueness and schema equality of required keys.

## 8. Rebase readiness (pv7/w16-core onto STAGE0-NIGHT1 + V7-FORMAL-PM6)

Scratch trial in a throwaway clone (branches `formal`, `stage0`, `integ`; session scratchpad):
- Replaying the 9 PV7 commits onto stage0 + formal-pm6 is **textually clean** (no conflicts).
- `verify_all_offline --fast` on the integrated tree shows extra FAILs vs real HEAD (sidecar `.md`/`.txt` hashes, `d8.sealed_file_hashes`, `d8.live_replay`). I re-ran at the pre-PV7 merge commit `d9a17b0` and got the **identical failure set**, so those failures are caused by the stage0 base plus Windows CRLF conversion in the clone, not by PV7. This is a caveat, not an accepted verdict: I could not rule out a CRLF contribution on the real merge host. `.gitattributes` is modified and unstaged in the parent workspace, so receipt-hashed files need an explicit `-text`/`eol=lf` rule before the rebase is trusted.
- My attempt to run the `tests/test_pv7_w19*` suite in the scratch integrated tree reported "file or directory not found" (those tests live on `pv7/w16-core` only, not in the stage0/formal histories), so **I have no integrated-tree test evidence**. Re-run it after the real rebase.

## 9. Test run and scoring

Command: `CUDA_VISIBLE_DEVICES="" pytest tests/test_pv7_w19* tests/test_pv7_w16* -q` on the real HEAD: **89 passed, 3 warnings**. The warnings include the `DeprecationWarning` from section 4. A passing count is not evidence for the #58 codec finding: the tests assert the gate passes, and the gate is what is in question.

L-mapping and deductions:

| Finding | L-class | Points |
|---|---|---|
| #58 scattered win is codec-dependent; claim unqualified | L4, L2-adjacent | -8 |
| #58 LCB95 on zero-variance sample | L13 | -2 |
| #59 burn gate: partial self-derived inventory, legacy deprecated path | L2/L5 | -3 |
| Loaders do not consume fragments; "64/64" is a file count | L13 | -2 |
| #57 live receipt exists, registry still `protocol-ready` | L13 | -1 |
| #56 missing tier/disclosure fields | L8 | -1 |
| #60 weight-digest-omitted negative test missing | L5 | -1 |

**BHS_REVIEWER = 82. BHS_OFFICIAL = min(self, 82) < 100: does not merge.** CARRY_FORWARD entries are needed for all seven rows.

## 10. Ordered action list

1. #58: unify the journal codec (one shared function), gate on the symmetric row, add disclosure fields, replace or relabel the LCB95. Re-run the drycheck and expect the scattered gate to **fail** unless the architecture genuinely wins at equal codec; if it fails, say so and reword the claim.
2. #58: do not extrapolate drycheck bytes to live 512 B sector accounting; pre-register that live results are decided by record count.
3. #59: explicit `allow_local_drycheck_smoke=True/False`; rename `burn_clean` to say what it checked; assert the historical inventory size in `run-live`.
4. #57: promote or annotate the registry fragment against the live receipt; record who set the 50x gate and when.
5. #56: add `tier` and `tier_disclosure`.
6. #60: add the digest-omitted negative test.
7. Loaders: add `load_all_bundles()` with explicit id normalisation and a 64-count test; wire `routes.d` into `server.py` or stop calling `61-fabric` a served route.
8. Rebase: add an LF rule for receipt-hashed files, rebase, then re-run `tests/test_pv7_w19*` and `verify_all_offline --fast` on the real result.
9. `computational_storage_poc/`: state which of the 32 files must carry `EXPERIMENTAL`; use a default-visible warning class.

EVIDENCE: `artifacts/pv7_w19_2c_cold_osd_9m/pv7_w19_2c_drycheck_manifest.json`; scratchpad probe `sym_probe.py` (symmetric-codec re-run: scattered -720 B).
SMOKE: floor tier only. `pytest` 89 passed; no live #58/#56/#59/#60 receipt exists to run.
