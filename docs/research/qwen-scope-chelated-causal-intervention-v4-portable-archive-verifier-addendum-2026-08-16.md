# QSCCI v4 portable archive-verifier addendum

**Applies to:** copied official `CHELATEDAI-QSCCI-v4` run-001 evidence  
**Scientific protocol:** unchanged  
**Scope:** custody and cache-independent verification only

The official result was published and verified on its Spark execution host.
The copied archive is retained at
`artifacts/method-dev/qscci-v4/run-001`. This addendum does not recondition a
scientific fixture, selection, intervention, endpoint, gate, disposition,
resource ceiling, service contract, model identity, SAE identity, or protocol.

## Frozen custody roots

The portable verifier pins the complete official three-file set:

- `qscci.json` SHA-256:
  `043f5b395549f383b21f39c5fdbd16572ea78b763b423a0314d2d48d378dbad6`.
- `manifest.json` SHA-256:
  `dd0cc08b5f3e3833b28a2ada70f0f78fe4d86ac099cc03fbdc38afb90538e991`.
- `COMMIT.json` SHA-256:
  `c16438aeaeceee03d638f2eea6a3f5c14727937590160cdf6ebf63e976189c20`.
- Retained internal artifact digest:
  `fe75a63bb2e2ef14d911c8be457f01044d329377c8982ccf598a7a19ac71d1e2`.

These SHA-256 values are code-distribution custody anchors. They are not
signatures, signer identity, execution-host attestation, or proof against a
chosen-prefix/collision break in SHA-256. A coherent edit and reseal changes at
least one pinned file root and is rejected.

## Mechanical checks

Archive verification requires an ordinary directory with exactly three direct
ordinary regular-file members named above. Symlinks, junctions, Windows reparse
points, hidden/stage/quarantine ancestry, path escapes, nested directories,
special files, and unexpected entries are rejected. Lexical and resolved paths
must remain inside the declared archive root.

All three members must be strict UTF-8 canonical JSON with no duplicate keys or
nonfinite constants. Their complete file SHA-256 roots must match this addendum.
The artifact must validate against the frozen v4 JSON Schema distribution and
retain the exact v4 source hashes, COMPLETE status, UNCONFIRMED scientific and
novelty claims, empty failure list, and official
`DOES_NOT_SURVIVE_SMALL_LABEL_ORACLE_FIXTURE` disposition. Its internal digest
is recomputed. The retained service-lifecycle evidence is checked against the
same closed typed evidence contract used by the live verifier. The manifest
and commit receipt must exactly cross-link the pinned artifact, internal digest,
and manifest roots. The shared cache-independent validator also reconstructs
the sparse-activation feature records, selections, prompt/cell identities,
ordered endpoints, tuning choices, REPORT grid, baseline cross-links, gates,
disposition, aggregate endpoint index, retained decoder-column membership and
per-column digests, resources, and failure/digest boundary.

## Deliberately false claims

The portable verifier does not load the pinned model or SAE cache and does not
reexecute model leaves, prompts, interventions, decoder-column provenance, or
the Torch-version-dependent decoder direction, delta-norm, and cosine floating
reductions. Those remain live-verifier-only checks. It does recompute the
platform-portable retained-leaf relationships listed above. It does not probe
DeepSeek, establish current service health, replay restoration, prove the
copied directory is the current Spark publication target, or replace the live
verifier. Its success mapping therefore sets `custody_only=true` and sets `reexecution_verified`,
`current_target_verified`, `live_service_lifecycle_verified`, and
`pinned_model_sae_cache_verified` to false.

Live `verify_result_dir` remains unchanged and continues to require the pinned
SAE cache and its full exact scientific reconstruction. Archive success uses the
distinct status `ARCHIVED_QSCCI_V4_VERIFIED`; it never emits live `verified` or
`committed_result_dir` claims.

## Command

```text
python run_qscci.py --verify-v4-archive artifacts/method-dev/qscci-v4/run-001
```
