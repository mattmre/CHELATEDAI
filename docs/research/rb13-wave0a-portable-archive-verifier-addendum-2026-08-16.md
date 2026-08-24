# RB-13 Wave 0A portable archive-verifier addendum

**Applies to:** completed official v8 SELECT and REPORT campaign archives  
**Scientific protocol:** unchanged `CHELATEDAI-RB13-WAVE0A-v8`  
**Scientific protocol SHA-256:**
`2fa9b950be0c68cea82e0922dbeb452015773e42efc352d8681f5dc88b9026aa`  
**Scope:** verification portability only; no new fixture, seed, endpoint,
disposition, null, or scientific result.

The original campaigns verified exactly and cross-phase on their GBA1 Linux
execution host. Their canonical JSON bytes, artifact/file/manifest digests,
schemas, claims, resources, masks, dispositions, nulls, and platform history
remain immutable. A copied Windows archive failed only the live verifier's
artifact-independent shared-code regeneration equality: floating operations
reassociated at the final machine-precision bits.

Measured across all copied official artifacts before defining archive mode:

- VAR1 maximum absolute regenerated numeric difference:
  `1.7763568394002505e-15`.
- SPU0 maximum absolute regenerated numeric difference:
  `1.9721522630525295e-31`.

The frozen portable archive bounds are deliberately just above those observed
cross-platform differences:

- VAR1 absolute tolerance: `2.0e-15`.
- SPU0 absolute tolerance: `2.5e-31`.
- Relative tolerance: zero. No scale-dependent widening is permitted.

The verifier distribution pins the exact official manifest file roots before
any tolerant semantic regeneration:

- SELECT manifest file SHA-256:
  `f17defbae0c6ac7a49a5eba295604ead4a70e66ee5dae749c7f2289153339ab2`.
- REPORT manifest file SHA-256:
  `e03f090d632307f58f5316f2dbec9a2b214ecc8bcbcff5cc8910bbd17a371a2c`.

These are code-distribution custody anchors. They are not signatures,
cryptographic identity assertions, or host attestation. Changing an artifact
and coherently recomputing every internal digest still changes the pinned
manifest file root and is rejected.

Archive mode applies these tolerances only to float leaves regenerated for
SPU0 and VAR1. Types, keys, list lengths, integer values, strings, booleans,
masks, dispositions, nulls, feasibility, claims, resources, artifact bytes,
and every digest remain exact. Observed floats, regenerated floats, and their
computed absolute differences must each be finite before comparison; NaN and
positive or negative infinity always fail. A coherent reseal beyond either bound fails.
Live `verify_campaign` and `verify_cross_phase` retain exact equality and are
unchanged.

The archive regular-file set must be exactly `manifest.json` plus every
manifest entry path. Paths must be canonical safe POSIX-relative names.
Unexpected regular files, symlinked files, symlinked directories, junctions,
and other nonregular files are rejected. Windows reparse points are detected
from `os.lstat(...).st_file_attributes & FILE_ATTRIBUTE_REPARSE_POINT`, so the
check does not depend on newer `Path.is_junction()` availability. Resolved
real paths must remain under the non-reparse archive root. Empty ordinary
directories carry no evidence and may be ignored.

Portable success is intentionally distinct:

- Campaign: `ARCHIVED_CAMPAIGN_VERIFIED`.
- Cross phase: `ARCHIVED_CROSS_PHASE_VERIFIED`.
- `current_platform_exact_regeneration=false` whenever a tolerated difference
  was required. Archive mode never emits live `VERIFIED`.

Commands:

```text
python run_rb13_wave0a.py --verify-archived-campaign artifacts/method-dev/rb13-wave0a/select
python run_rb13_wave0a.py --verify-archived-campaign artifacts/method-dev/rb13-wave0a/report
python run_rb13_wave0a.py --verify-archived-cross-phase artifacts/method-dev/rb13-wave0a/select artifacts/method-dev/rb13-wave0a/report
```
