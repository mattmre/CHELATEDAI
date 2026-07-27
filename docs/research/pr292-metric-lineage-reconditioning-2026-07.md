# PR #292 metric-lineage reconditioning — offline report

Date: 2026-07-27

Mode: local-only; no GitHub mutation

Starting reconditioning head: `4d49d9ce45cb5234ed9f1d5d7af1085e23001975`

## Narrow claim

This reconditioning does not repair nDCG, regenerate campaigns, or establish a
new H2/H4/H5 result. It preserves and quarantines the historical evidence,
removes promotion/rejection language from authoritative current-status prose,
records exact artifact identities and missing-raw dispositions, and makes the
remaining metric-lineage work explicit.

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

## Historical evidence preservation

The first reconditioning pass remains intact:

- all five result reports retain a top-of-file
  `LEGACY_METRIC_LINEAGE_BLOCKED` warning;
- `docs/next-session.md` retains a conservative H5 non-promotion disposition;
- none of the historical JSON or manifest artifacts was edited by this pass.

The versioned machine-readable sidecar is
`artifacts/legacy-ndcg-quarantine-index-v1.json`. It covers all 11 retained
affected result reports, manifests, and raw H4 artifacts with exact Git blob
and SHA-256 identities, accepted/prohibited use classes, and a raw-data
disposition. The four retained summary manifests reference 74 raw paths in
total (15 + 15 + 22 + 22); none is present at the branch tip. Historical blobs
at reused paths are not accepted substitutes without a digest/provenance map.

## Deferred scope and exit criteria

CD-MLR-01 and DS-MLR-01 track:

1. a qrels-complete graded/binary nDCG contract;
2. migration of every drift-recovery caller;
3. a regression where positive qrels exceed relevant retrieved hits;
4. corrected H2/H4/H5 regeneration;
5. retained environment, code, dataset, model, seed, and command provenance;
6. an exact old-artifact → corrected-artifact supersession map.

Until all six are closed, the old exact values and rankings remain prohibited
for scientific, promotion, rejection, paper, release, or roadmap claims.

## Offline PR-body field proposal

Do not publish these fields or merge until a fresh independent Tier-B pass
reviews the final diff and runtime evidence:

```text
CARRY_FORWARD: CD-MLR-01 — qrels-complete metric contract, caller migration, corrected H2/H4/H5 regeneration, and hash-linked supersession provenance; TTL=1 cycle.
DEFERRED_SCOPE: DS-MLR-01 — valid quantitative H2/H4/H5 conclusions; owner research-validity; target next remediation cycle.
LOOP_ITERATIONS: 2 — iteration 1 added report warnings and conservative disposition; iteration 2 removed authoritative hard-negative/rejection claims, added immutable sidecar validation, narrowed CD-A2 proof, and made CD-H1 procedural.
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
  11 artifacts, one durable backend-evidence anchor, CD-MLR-01 and DS-MLR-01
  linked.
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
- Python 3.9 AST parse for the new validator — **PASS**.
- `python -m json.tool artifacts/legacy-ndcg-quarantine-index-v1.json` —
  **PASS**.
- `git diff --check` — **PASS**.

An attempted combined `--help` probe was not acceptance evidence:
`smoke_pipeline.py` ignores `--help` and started its normal path, so the
combined command timed out after 34 seconds. Static entry-point inspection
identified the correct bounded floor function; the transient Python process
had exited by the time its exact PID and command were rechecked.
