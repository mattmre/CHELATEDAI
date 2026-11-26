# W22-04 Report: real-data bridge x merge factorial at equal total cost

- Preregistration: `PREREGISTRATION.md`, committed alone as `dc670217` before any result existed.
- Runner: `scripts/run_pv7_w22_04_bridge_merge_factorial.py`, at `48dd4075` plus the offline-path fix `c3a569cb`.
- Results: `w22_04_results.json` (commit `543646ed`). Every number below is copied from that file.
- Hardware: RTX 3090, torch 2.5.1+cu121, Python 3.11.9, Windows 11.
- Run 2 wall time: 1252.9 s, offline (`HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`).

(Report text written by the W22-04 implementer sub-agent; transcribed into this file by the orchestrator because the harness blocks sub-agents from writing `.md` files.)

## Verdict: KILL (the unique ZeroMu bridge claim is killed; CSLS is retained as the core finding)

| Test | WITHIN | Threshold | Verdict |
|---|---|---|---|
| Primary (abs(delta) <= 0.005 or paired 95% CI contains 0, against either Procrustes) | **10/12** | >= 10 | **KILL** |
| Secondary (marginal-CI overlap) | 12/12 | >= 10 | KILL |

The cost clause holds by construction (prereg section 3). Both Procrustes variants persist less state than ZeroMu. All bridges share the same anchors, backfill and DEV data.

## Per-domain result: ZeroMu+CSLS minus comparator+CSLS (nDCG@10, paired bootstrap, seed 1601)

| Domain | n_q | vs Centered [CI] | vs Uncentered [CI] | WITHIN via |
|---|---|---|---|---|
| climate-fever | 260 | -0.0548 [-0.0745, -0.0365] | -0.0157 [-0.0326, +0.0007] | Uncentered only |
| cqadupstack-android | 240 | -0.0091 [-0.0246, +0.0065] | +0.0280 [+0.0037, +0.0512] | Centered |
| cqadupstack-english | 260 | +0.0176 [+0.00004, +0.0367] | +0.0972 [+0.0669, +0.1298] | NOT WITHIN (ZeroMu better) |
| cqadupstack-gis | 260 | -0.0122 [-0.0269, +0.0007] | +0.0314 [+0.0080, +0.0551] | Centered |
| cqadupstack-physics | 240 | +0.0093 [-0.0050, +0.0243] | +0.0672 [+0.0390, +0.0943] | Centered |
| cqadupstack-programmers | 240 | +0.0087 [-0.0046, +0.0233] | +0.0873 [+0.0588, +0.1165] | Centered |
| cqadupstack-stats | 220 | +0.0180 [+0.00007, +0.0361] | +0.0965 [+0.0620, +0.1304] | NOT WITHIN (ZeroMu better) |
| cqadupstack-tex | 260 | -0.0045 [-0.0198, +0.0108] | +0.0529 [+0.0253, +0.0814] | Centered |
| cqadupstack-unix | 240 | +0.0133 [-0.0014, +0.0277] | +0.0721 [+0.0472, +0.0970] | Centered |
| dbpedia | 260 | +0.0043 [-0.0066, +0.0156] | +0.0634 [+0.0479, +0.0805] | Centered |
| touche-2020 | 34 | +0.0105 [-0.0129, +0.0332] | +0.0431 [+0.0047, +0.0813] | Centered |
| trec-covid | 35 | -0.0357 [-0.0846, +0.0165] | +0.0204 [-0.0498, +0.0929] | Both |

- Against Centered Procrustes, ZeroMu is unresolved on 9/12 domains. It is resolved worse on climate-fever and barely better on english and stats.
- Against Uncentered Procrustes, ZeroMu is significantly better on 10/12 domains. Centering recovers almost all of that advantage.

## Cell means over 12 domains (nDCG@10)

| Bridge | Raw | ZScore | CSLS | RRF |
|---|---|---|---|---|
| UncenteredProcrustes | 0.3889 | 0.4818 | 0.6115 | 0.4112 |
| CenteredProcrustes | 0.4116 | 0.6111 | **0.6680** | 0.5179 |
| RidgeDriftAdapter | 0.4861 | 0.6044 | 0.6580 | 0.5150 |
| Rank1QueryResidual | 0.4253 | 0.5983 | 0.6632 | 0.5102 |
| ZeroMuPolarBridge | 0.3764 | 0.6401 | 0.6651 | 0.5409 |

## CSLS retention (CSLS against the best non-CSLS merge, paired CI, seed 1602)

| Bridge | CSLS is best merge | CI above | CI below |
|---|---|---|---|
| Uncentered | 12/12 | 12 | 0 |
| Centered | 12/12 | 11 | 0 |
| Ridge | 11/12 | 11 | 0 |
| Rank1 | 12/12 | 11 | 0 |
| ZeroMu | 11/12 | 8 | 1 |

## Fidelity (prereg section 6)

- Dataset sha256 matches both manifests on all 12 repos.
- Eval doc and query counts match on all 12 domains (dbpedia has 954 docs). The scrub removed 0 rows.
- Largest difference from recorded values, per domain:
  - climate-fever 0.0019
  - android 0.0056
  - english 0.0028
  - gis 0.0037
  - physics 0.0058
  - programmers 0.0102
  - stats 0.0090
  - tex 0.0050
  - unix 0.0069
  - dbpedia 0.0032
  - touche 0.0115
  - trec-covid 0.0133
- No domain is flagged (threshold 0.02). The run is close to the manifests but not bit-exact.

## Departures

1. Pre-declared: RTX 3090 instead of L4; torch 2.5.1; 2 of 5 encoders encoded; Qwen batch size 32; one fetch per missing public resource from the original HF source; remaining5 shift texts encoded only to match batch composition.
2. Not pre-declared: run 1 failed while loading Qwen (`LocalEntryNotFoundError`, no `refs/main` for a snapshot fetched by revision). Fixed in `c3a569cb` with no change to analysis logic. Its log is kept at `scratch/run.log`.
3. The factorial has 5 bridges: the harness's `ALL_BRIDGES` includes Rank1QueryResidual. Only the two Procrustes variants enter the kill rule.
4. The preregistration commit message says "4x4"; the preregistration text and the run are 5x4.

## Brutal Honesty

- KILL lands exactly on the threshold (10/12).
  - english and stats miss WITHIN by CI lower bounds about 5e-5 above 0, and gis is WITHIN by a CI upper bound of +0.0007.
  - A different bootstrap seed could flip the verdict. The secondary rule (12/12) agrees.
- climate-fever is WITHIN only through Uncentered Procrustes. Centered Procrustes is resolved better there, by 0.055.
- "CI contains 0" is not an equivalence test. touche-2020 (n=34) and trec-covid (n=35) are WITHIN largely through low power.
- ZeroMu is not useless: ZeroMu+ZScore (0.640) beats Centered+ZScore (0.611). Its edge disappears under CSLS.
- Not done:
  - no multi-seed repeat of anchor sampling (single seed 20261125);
  - no bootstrap-seed sweep;
  - nDCG@10 only;
  - the other 3 encoders were not rerun.
- No mocks or stubs in the production path. The unit tests are synthetic and are not evidence for the verdict.
- No Tier B review has been run.

DEFERRED_SCOPE: none. All 240 cells were run.
BHS_SELF_DRAFT: 95
BHS_SELF_DRAFT_AGENT: claude-opus-5-5 (W22-04 implementer subagent)
BHS_TIER_B: not run
BHS_OFFICIAL: pending

## Scratch receipt (removable)

- Path: `C:\Users\mattm\AppData\Local\Temp\claude\439e695c92b40043\c9f4318c-455f-427a-844a-07562a0a3c42\scratchpad\w22-04\artifacts\pv7_w23a_w22_04\scratch`
- Size: 380,227,812 bytes (about 362.6 MiB), gitignored.
- Contents: 4 embedding caches (`.npy` with `.json` sidecars), `run.log`, `run2.log`.
- Outside the repo: the HF cache `C:\Users\mattm\.cache\huggingface\hub` grew by about 15 GB (Qwen3-Embedding-8B) and about 6.3 GB (datasets).
- Removing either follows the full-path deletion rule. Nothing was deleted.
