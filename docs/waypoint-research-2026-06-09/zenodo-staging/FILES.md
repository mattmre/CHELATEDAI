# Files to include in the Zenodo upload bundle

Zip these from a **clean checkout at the chosen tag** (paths relative to repo root).
All are public, committed repo files — this manifest just scopes the bundle.

## Harness (core)
- `drift_injector.py`
- `drift_recovery_metrics.py`
- `run_drift_recovery_experiment.py`
- `run_drift_recovery_swap_campaign.py`
- `query_encoder_drift.py`
- `benchmark_utils.py`            # ndcg/eval helpers used by the harness
- `chelation_adapter.py`          # adapter family incl. BoundedAdapter
- `antigravity_engine.py`         # engine (frozen-base + adapter retrieval path)
- `vector_store.py`

## Tests (evidence the harness behaves as claimed)
- `test_drift_injector.py`
- `test_drift_recovery_metrics.py`
- `test_run_drift_recovery_experiment.py`
- `test_engine_telemetry_cuda_guard.py`  # ⚠️ NOT yet on main — lands with the H2 merge
  (the `antigravity_engine` `device_count()>0` CUDA-telemetry guard + its test live on the
  GPU-deferred H2 branch). OMIT from a pre-H2 bundle; include once H2 lands.

## Results (the worked example — use the REGENERATED versions post-H1)
- `docs/drift-recovery-swap-results-2026-06.md`            # SciFact
- `docs/drift-recovery-swap-nfcorpus-results-2026-06.md`   # NFCorpus
- `experiment_runs/drift-recovery/swap/swap-campaign-manifest-2026-06.json`
- `experiment_runs/drift-recovery/swap-nfcorpus/swap-campaign-manifest-2026-06.json`

## Provenance
- `requirements.txt`, `pyproject.toml`
- This `README.md` + `.zenodo.json`
- Reference the public repo URL + the exact commit/tag the bundle was cut from.

## NOT included (local-only strategy — never upload)
- Anything under `docs/waypoint-research-2026-06-09/` (the paper draft, plans, this
  staging dir itself). Only the public harness + committed result docs go in the bundle.
