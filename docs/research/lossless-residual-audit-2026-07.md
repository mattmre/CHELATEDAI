# CHELATEDAI Lossless Residual Audit

Date: 2026-07-27
Cycle: `AEP-20260727-7`
Status: `PRESERVED_LOCAL_PRIVATE_REVIEW`; cleanup is not authorized.

## Decision

Ignored, untracked, worktree-only, stashed, binary, and recovery material is
not equivalent to disposable output. It was inventoried and hash-bound before
publication or merge decisions were made. This public-safe summary contains
aggregate counts and recovery proofs; the file-level manifests remain private
because relative filenames alone can reveal local or sensitive context.

No item in this audit was deleted, restored, regenerated, applied from a
stash, dropped, pruned, garbage-collected, or published.

## Private inventory binding

| Snapshot | Scope | File SHA-256 |
| --- | --- | --- |
| V1 | initial ignored/untracked inventory | `de539255ad4dde8c1df2a4e592cf4ba8e1444efc5be599031d9ab020b586d3fc` |
| V2 | V1-bound inventory plus publication dispositions and tracked-dirty state | `4c1192a5fd77f75c5cd1309f2398755453c2fcaab376fc7571d2f85a4599e98b` |
| V3 | V2-bound post-reconciliation snapshot at `8c7446fb` | `108efa75aeeda94d6ae8fad1210ee1c27d50b81d161148204e7f354c33ee5cf0` |

V2 records:

- 10 total worktrees (the primary plus 9 linked worktrees);
- 2 stashes;
- 2,584 ignored or untracked files;
- 1,678,118,521 total bytes;
- zero inventory/read/hash errors;
- entry-manifest SHA-256
  `31e443916f471abcdc6c1284e07f28b30d5389b470f45a34397aad885a69fe08`;
- worktree-state SHA-256
  `14b46c4fe641b7446b5edd27c8fe6d26404ae393ddac3e01d0f673a143975181`;
- `cleanup_authorized: false`.

An independent validator recomputed the schema, aggregate counts and bytes,
unique `(worktree, relative path)` keys, entry digest, worktree-state digest,
prior-manifest binding, and dirty-state structure. All checks passed.

V3 additionally records 2,854 entries, 1,829,320,221 bytes, ten total
worktrees, two stashes, and zero errors, with:

- entry-manifest SHA-256
  `ee3148af9015c423ef182c16995a2127b95f592439cbdfa9413e222b0e62630f`;
- worktree-state SHA-256
  `d6f8bd1833328376897fe26609b5a5b71fef2710b49709fd8e8c6e21e685f3c0`;
- `cleanup_authorized: false`.

This document-only V3 acceptance record postdates the immutable snapshot and
does not change its source/evidence or residual payload.

## V2 publication dispositions

| Disposition | Files | Bytes | Public-repository action |
| --- | ---: | ---: | --- |
| `DO_NOT_PUBLISH_SENSITIVE` | 14 | 692,906,577 | Never push in current form |
| `KEEP_PRIVATE_RECOVERY_DO_NOT_PUBLISH` | 30 | 344,070,467 | Keep as local recovery packages and manifests |
| `PRIVATE_ARCHIVE_OR_EXTERNAL_STORAGE_REVIEW` | 348 | 456,661,426 | Models/databases require license, provenance, deserialization, and storage review |
| `CURATE_PROVENANCE_BEFORE_PUBLICATION` | 205 | 87,344,649 | Publish only a reviewed, source-bound evidence subset |
| `REPRODUCIBLE_DO_NOT_PUBLISH` | 1,764 | 33,059,268 | Rebuild from source; do not commit caches/build output |
| `PRIVATE_LOCAL_STATE_DO_NOT_PUBLISH` | 9 | 21,450 | Keep local |
| `REVIEW_BEFORE_PUBLICATION` | 213 | 64,042,605 | Resolve exact duplication, privacy, and license status first |
| `SOURCE_RECONDITION_AND_REVIEW` | 1 | 12,079 | Preserve now; format, test, and review in a separate PR |

The byte total includes private recovery copies by design. It is an inventory
of what exists, not a claim that every byte is unique or should be uploaded.

## V3 publication dispositions

| Disposition | Files | Bytes | Public-repository action |
| --- | ---: | ---: | --- |
| `DO_NOT_PUBLISH_SENSITIVE` | 14 | 693,043,982 | Never push in current form |
| `KEEP_PRIVATE_RECOVERY_DO_NOT_PUBLISH` | 36 | 489,918,549 | Keep as local recovery packages and manifests |
| `PRIVATE_ARCHIVE_OR_EXTERNAL_STORAGE_REVIEW` | 348 | 456,661,426 | Models/databases require license, provenance, deserialization, and storage review |
| `CURATE_PROVENANCE_BEFORE_PUBLICATION` | 205 | 87,344,649 | Publish only a reviewed, source-bound evidence subset |
| `REPRODUCIBLE_DO_NOT_PUBLISH` | 2,028 | 38,275,481 | Rebuild from source; do not commit caches/build output |
| `PRIVATE_LOCAL_STATE_DO_NOT_PUBLISH` | 9 | 21,450 | Keep local |
| `REVIEW_BEFORE_PUBLICATION` | 213 | 64,042,605 | Resolve exact duplication, privacy, and license status first |
| `SOURCE_RECONDITION_AND_REVIEW` | 1 | 12,079 | Preserve now; format, test, and review in a separate PR |

## Verified preservation results

### Git histories

The primary research line, drift-recovery line, waypoint corpus, semantic/BCC1
line, H2 guard, PR-repair candidates, and both stashes have exact local refs or
verified bundles. Incremental bundles record their prerequisite commits. The
complete-history stash bundle advertises both archive refs exactly.

Rejected repair histories remain preserved rather than silently overwritten:

- PR #292 candidate `835f6199` failed fresh Tier B at 70/Critical because its
  quarantine covered only 11 of 113 affected artifacts and its validator was
  fail-open; its rejected head has an exact local archive ref and a complete
  replacement is committed at `f2e41d42`. The replacement reports 113/113
  coverage, preserves all 103 historical JSON/PNG bytes, passes 28 hostile
  validator cases, and has a verified incremental bundle. A second independent
  exact-candidate review scored its content at local Tier B 100, making it
  content-ready for owner-approved publication/CI. Current public-state BHS
  remains 70/Critical because #292 is still at `eb750958` with a stale body and
  no `f2e41d42` hosted runs.
- PR #293 candidate `454e4a32` failed at 70/Critical after six iterations and
  has an exact local archive ref; no seventh repair is permitted.
- PR #295 candidate `730b305e` failed at 60/Critical after five iterations and
  has an exact local archive ref; no sixth repair is permitted. Its four
  consumed lock/REPORT records plus the historical manifest remain
  byte-preserved fail-closed evidence and cannot be overwritten or relabeled as
  exact-current-head evidence. Withdraw current #295; any archival replacement
  is a separate nine-file PR from final #292.
- PR #294 candidate `6e78cf41` reached 100 only on its original stacked/
  pre-transplant head. Its code is independent of rejected #293 and its old
  head has an exact local archive ref, but it must use a new transplant branch/
  replacement PR without rejected ancestry and be reviewed anew. Do not
  force-push or retarget public #294.

### Raw-evidence archives

- PR #292 raw drift evidence: all 76 live files stream-hashed byte-identically
  to the 76 ZIP file entries (plus three directory records); zero missing,
  mismatched, or unexpected file entries.
- Session-29 weight-refinement evidence: all 24 live files stream-hashed
  byte-identically to the 24 ZIP file entries (plus two directory records);
  zero missing, mismatched, or unexpected file entries.
- Remaining primary/H2 experiment evidence and execution logs: 140
  secret-scanned live files totaling 28,802,844 bytes were first matched to V2
  and then stream-hashed byte-identically into a private ZIP with internal
  manifest. Archive SHA-256:
  `c41ef68a14e25b4d1c458270eb3a643b88886a16229546228cc34735b517b7eb`.

### Waypoint corpus

All 201 ignored/untracked files present in the primary waypoint directory
match tracked files at recovery commit
`65ae99a4ae9b90c56febc6d8b84e1447c522ea3f` byte-for-byte. The branch is a
durable local recovery, but it is not approved for public upload: it contains
large agent-output transcripts that need privacy/license review. A redacted
Gitleaks result also flagged a JavaScript object property whose value was
independently parsed as the internal item identifier `R3-h5-campaign`, not a
credential. The false positive does not waive the broader transcript review.

### Unique untracked source

`scripts/scaffold_brain_dossier.py` is not present in any ordinary local or
remote branch history. Its 12,079 bytes are preserved in a verified one-entry
private ZIP:

- source SHA-256:
  `f6a1625fee982b1061b9cefaf1e26d6ff462ea9b5fc4cb96e4830df8d0cef7fc`;
- archive SHA-256:
  `1c0608755ec2dabf3d6cad2cbc016f36467a43a8c182da4ab1822e55b9fa9ffa`.

Python 3.9 parsing, Ruff lint, a read-only tier scan, and a secret scan passed.
Ruff formatting failed and no dedicated tests exist. It is valuable source,
but not merge-ready source; it requires a separate reconditioning PR.

### Models and checkpoints

The V1 model/checkpoint set contained 344 files and 334,948,786 bytes, but only
177 distinct content hashes. Repeated hashes account for 167 duplicate file
instances and 193,143,042 redundant bytes. The analysis used hashes and file
sizes only; serialized tensor files were not loaded because pickle-backed
formats are an execution boundary. These objects should not be committed to
ordinary Git. A later owner-approved phase can select unique, licensed,
source-bound objects for private artifact storage or Git LFS.

## Tracked worktree state

At the V2 snapshot, nine worktrees were clean. The only tracked-dirty worktree
was `h2-rerun`, with the known 24 unstaged experiment-file deletions. The H2
source fix is independently committed and recovery-bundled. Those deletions
were not staged, restored, committed, or used in its review.

## Git object-integrity boundary

With commit-graph acceleration enabled, `git fsck --full` reports 17 unreadable
commit IDs, each repeated as a commit-graph parse message. With
`core.commitGraph=false`, full fsck exits zero with no missing object, broken
link, invalid ref, or structural error. The defect is isolated to stale
commit-graph metadata, not live history.

The current split commit-graph files were hash-fingerprinted privately. No
commit-graph rewrite, garbage collection, object pruning, or dangling-object
cleanup is authorized before itemized owner signoff.

## Safe upload rule

Before an initial public push or PR edit, an exact local Git commit must pass
content/privacy review, object-size checks, Gitleaks, repository gates, local
independent review, and explicit public-destination/payload approval. That
approved push may then trigger hosted checks. Merge additionally requires
successful hosted checks and a fresh independent review of the exact published
head, body, and check evidence. The same explicit-mutation gate applies to PR
title/body/base edits, close/withdraw actions, replacement PR creation, thread
resolution on the owner's behalf, and merges. Private manifests, recovery
bundles/ZIPs, sensitive logs, agent transcripts, learned weights, databases,
generated caches, and uncurated raw experiment directories are excluded from
the public push.

After all accepted branches are merged and their remote reachability is
proved, every remaining worktree, ref, stash, archive, ignored path, and Git
metadata repair will be presented separately with its hash/size, containment
proof, recovery procedure, and deletion consequence. No batch cleanup approval
is inferred.
