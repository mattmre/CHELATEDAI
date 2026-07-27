# PR #292 metric-lineage reconditioning — offline report

Date: 2026-07-27

Mode: local-only; no GitHub mutation

Starting reconditioning head: `4d49d9ce45cb5234ed9f1d5d7af1085e23001975`

Second-pass preservation anchor:
`835f61992b96412589dff0403b3b43f0af0e52f7`

## Narrow claim

This reconditioning does not repair nDCG, regenerate campaigns, or establish a
new quantitative drift-recovery result. It preserves and quarantines the
historical evidence, removes promotion/rejection language from authoritative
current-status prose, records exact artifact identities and missing-raw
dispositions, separates original report blobs from later warning-surface blobs,
and makes the remaining metric-lineage work explicit.

## Finding disposition

- `MLR-NDCG-001`: `drift_recovery_metrics.ndcg_at_k()` converts only the
  retrieved top-k IDs into a binary relevance vector and delegates IDCG to
  `benchmark_utils.ndcg_at_k()`. The latter sorts that retrieved vector; it
  does not receive the count or grades of all positive qrels. Exact values,
  ordering, effect sizes, and derived gates are therefore
  `LEGACY_METRIC_LINEAGE_BLOCKED`.
- H5 remains non-promoted because there is no valid positive evidence. It is
  not a confirmed negative.
- H4 compounding remains non-promoted. The retained single-seed artifacts do
  not currently prove collapse, rejection, or correctness of the alternative.
- H1 debt closure is procedural only: the pre-H1 reports were superseded after
  the H1 code change. No metric equality or effect-size statement is retained.
- CD-A2-01 remains narrowly closed because a pre-existing committed log blob
  proves that the real `all-mpnet-base-v2` local backend resolved. This is not
  reconstructed evidence and does not validate H2-specific execution or any
  metric.

## Durable backend-resolution evidence

- Source commit: `efac48daae5358599eeeb055906083e423eb5072`
- Source path:
  `experiment_runs/drift-recovery/swap-nfcorpus/campaign-completion.log`
- Git blob: `e5ebca641969ca20c813045f5f0898e2881272d6`
- Required immutable markers:
  `Initializing local backend: all-mpnet-base-v2` and
  `Local backend loaded. Vector size: 768`
- Production path:
  `run_drift_recovery_experiment._build_query_encoder_drift()` →
  `QueryEncoderDrift.embed_queries()` → `QueryEncoderDrift._backend()` →
  `create_embedding_backend()` → `LocalEmbeddingBackend`

The validator resolves the commit:path pair to the stated blob, verifies that
the source commit is reachable from the current branch, and checks both
markers. This proves a real load, not quantitative correctness.

## Complete blast radius and preservation

Producer/caller tracing, followed by content parsing of every tracked file
under `experiment_runs/drift-recovery`, establishes a closed affected set of
**113 tracked artifacts**:

| Artifact class | Count | Provenance |
|---|---:|---|
| Direct raw JSON | 89 | `record_type=drift_recovery_experiment`, emitted by `run_drift_recovery_experiment.run_experiment()` through the two faulty evaluation paths |
| Derived aggregate JSON | 8 | two diagnostics, two swap manifests, two post-bank manifests, one calibration manifest, and one knob-sweep manifest |
| Generated PNG | 6 | plots emitted from the affected diagnostics/trajectory values |
| Metric-claim Markdown | 10 | result, diagnostics, selection, H2/H4/H5, and generated diagnostics surfaces |

The prior `artifacts/legacy-ndcg-quarantine-index-v1.json` covers only 11 of
these files. It is preserved byte-for-byte as an incomplete audit predecessor;
it is not the authoritative coverage map.

The authoritative v2 map is
`artifacts/legacy-ndcg-quarantine-index-v2.json`, with four deterministic
hash-bound shards under `artifacts/legacy-ndcg-quarantine-v2/`. It covers
113/113 artifacts with exact Git blob and SHA-256 identities, closed
artifact-type/path/state enums, allowed and prohibited use vocabularies, raw
disposition group IDs, and pending supersession state.

All 89 raw JSONs, 8 aggregate JSONs, and 6 PNGs are byte-identical to the
second-pass preservation anchor `835f6199` (103 files total). No historical
JSON or image was edited. The five previously un-warned result/diagnostics
Markdown files received warning blocks only; their tables and historical
verdict prose were not rewritten. All 10 metric-claim Markdown surfaces now
carry a prominent `LEGACY_METRIC_LINEAGE_BLOCKED` warning.

The four retained late-campaign manifests reference 74 raw paths in total:
22 swap SciFact, 22 swap NFCorpus, 15 post-bank SciFact, and 15 post-bank
NFCorpus. All 74 are absent from the branch tip and are enumerated exactly in
the v2 raw-disposition records. The earlier base/calibration/knob/H4 groups
retain all 89 of their referenced raw artifacts.

## Original report provenance versus warning surfaces

The first pass retroactively added warnings to five historical reports. A
post-edit working-tree hash is not an original-source identity. V2 therefore
records two separate roles for every one of the 10 prose surfaces:

- `ORIGINAL_PRE_RECONDITIONING_SOURCE`: an exact reachable commit, Git blob,
  and SHA-256. The five first-pass reports bind their original form at
  `eb7509583e81b6a62d13b82587398562e4bba09a`; the five second-pass reports
  bind their pre-warning form at `835f61992b96412589dff0403b3b43f0af0e52f7`.
- `RECONDITIONED_WARNING_SURFACE`: the current distinct Git blob and SHA-256,
  used only as the active fail-closed reader surface.

The validator resolves every historical `commit:path` pair and rejects any
attempt to relabel a warning-surface blob as the original evidence.

## Deferred scope and exit criteria

CD-MLR-01 and DS-MLR-01 remain open and track:

1. a qrels-complete graded/binary nDCG contract;
2. migration of every drift-recovery caller;
3. a regression where positive qrels exceed relevant retrieved hits;
4. corrected regeneration or explicit archival retirement for every affected
   quantitative campaign whose claims are to be restored, including H2/H4/H5;
5. retained environment, code, dataset, model, seed, and command provenance;
6. an exact 113-entry old-artifact → corrected/retired-artifact supersession
   map.

Until all six are closed, the old exact values and rankings remain prohibited
for scientific, promotion, rejection, paper, release, or roadmap claims.

## Offline PR-body field proposal

Do not publish these fields or merge until a fresh independent Tier-B pass
reviews the final diff and runtime evidence:

```text
CARRY_FORWARD: CD-MLR-01 — qrels-complete metric contract, caller migration, corrected regeneration or explicit retirement for affected quantitative campaigns (including H2/H4/H5), and a 113-entry hash-linked supersession/disposition map; TTL=1 cycle.
DEFERRED_SCOPE: DS-MLR-01 — valid quantitative H2/H4/H5 conclusions; owner research-validity; target next remediation cycle.
LOOP_ITERATIONS: 3 — iteration 1 added report warnings and conservative disposition; iteration 2 removed authoritative hard-negative/rejection claims, added the bounded v1 sidecar, narrowed CD-A2 proof, and made CD-H1 procedural; iteration 3 expanded to exact 113/113 coverage, separated original and warning-surface provenance, added hostile fail-closed tests, and wired the validator into CI.
BHS_TIER_B: PENDING — must be assigned by a fresh independent reviewer after this local commit.
BHS_OFFICIAL: PENDING — not merge-ready on this report alone.
OPERATOR_OVERRIDE:
```

## Local validation

- `python -m unittest -v test_run_drift_recovery_swap_campaign.py
  test_run_drift_recovery_experiment.py test_post_bank_head_to_head.py
  test_post_bank_conditions.py test_post_bank_correction.py
  test_post_bank_runtime.py test_query_encoder_drift.py` — **PASS**, 70 tests
  run, 1 skipped. The skip is the explicit
  `CHELATED_RUN_REAL_MODEL_TESTS=1` integration gate; these unit tests do not
  validate historical nDCG values.
- `python scripts/validate_metric_lineage_quarantine.py` — **PASS**,
  113/113 artifacts (89 raw JSON, 8 aggregate JSON, 6 PNG, 10 prose), one
  durable bounded backend-evidence anchor, CD-MLR-01 and DS-MLR-01 linked.
- `python -m unittest -v test_validate_metric_lineage_quarantine.py` —
  **PASS**, 28 tests. The hostile matrix rejects fabricated total/type/shard
  counts, coordinated missing/extra artifacts, path/type/state/hash drift,
  allowed/prohibited-use drift, missing-raw state/count/path lies,
  original-source role/blob substitution, over-broad backend claim/scope/path,
  weak marker evidence, preservation-count drift, debt substitution, extra
  shards, weak prose warnings, and weakened authoritative control surfaces.
- `python scripts/check_block_flag.py` — **PASS / CLEAR with warning**,
  one open first-cycle debt row. If CD-MLR-01 survives a cycle, the operator
  must mark it expired and flip the flag to BLOCKED.
- `python scripts/validate_v33_schema_drift.py --json-report
  C:\tmp\pr292-recondition-v33-schema.json` — **PASS**, exit 0, no issues.
- `python -c "from scripts.smoke_pipeline import run_floor_smoke; raise
  SystemExit(run_floor_smoke())"` — **PASS, floor tier only**. It imported the
  production engine and four core dependencies and verified the
  `AntigravityEngine` entry point. Ceiling smoke was deliberately not run; no
  model was initialized or downloaded for this reconditioning.
- `python -m ruff check .` — **PASS**. The first
  `ruff format --check` reported that the new validator required formatting;
  `python -m ruff format scripts/validate_metric_lineage_quarantine.py` was
  applied, after which focused lint and format checks passed.
- Python 3.9 AST parse for the new builder, validator, and hostile test module —
  **PASS**, 3 files.
- `python -m json.tool artifacts/legacy-ndcg-quarantine-index-v2.json` and all
  four v2 shards — **PASS**. The v1 JSON remains unchanged and parseable.
- `git diff --check` — **PASS**.
- `gitleaks detect --source . --no-git --redact --exit-code 1` — **PASS**,
  approximately 71.15 MB scanned, no leaks found.

An attempted combined `--help` probe was not acceptance evidence:
`smoke_pipeline.py` ignores `--help` and started its normal path, so the
combined command timed out after 34 seconds. Static entry-point inspection
identified the correct bounded floor function; the transient Python process
had exited by the time its exact PID and command were rechecked.
