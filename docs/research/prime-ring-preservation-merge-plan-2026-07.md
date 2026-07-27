# CHELATEDAI Lossless Preservation and Merge Plan

Date: 2026-07-27
Cycle: `AEP-20260727-7`
Status: `IN_PROGRESS_PRESERVATION_FIRST`; cleanup is not authorized.

## 1. Decision

The current work is recoverable, but it is not yet safe to call merged or
clean. The immediate safe action is to finish the reviewed primary research
tree, obtain explicit approval for its public destination/payload, and publish
the exact commit history. The former linear lattice PR stack is no longer a
valid merge chain: PR #293 exhausted its repair budget with reproducible
critical integrity defects and must be withdrawn. PR #294 must use a new
transplant branch/replacement PR. PR #295 must be withdrawn; only a separate
evidence-only archive PR is presently defensible. These actions remain local
plans until their exact public mutations are explicitly approved.

No worktree, stash, ignored artifact, recovery package, branch, or unreachable
object will be deleted, pruned, rewritten, or consolidated in this cycle.
Cleanup begins only after merged-state verification and explicit owner approval
for each residual item.

## 2. Live Git and GitHub boundary

- Public repository: `mattmre/CHELATEDAI`.
- Default branch: `main`.
- Refreshed `origin/main`: `34ce4b5632e0d9cd2a16c29e0e1acc42e645b9c2`.
- Pre-reconciliation primary durability anchor after the evidence and first
  bookkeeping commits:
  `codex/prime-ring-onion-method-dev` at
  `7dec564de77b6efb03a82cb87405820284c7a8aa`, exactly 13 commits ahead and
  zero behind.
- Commits 1-4 (`2b0fa044` through `eb750958`) are already the exact published
  head of open PR #292.
- Commits 5-13 (`11d548bd` through `7dec564d`) have no GitHub ref and are the
  first publication priority.
- No remote ref contains the primary preservation head.

### Existing dependency stack

| PR | Head | Dependency | Current disposition |
| --- | --- | --- | --- |
| #292 | `eb750958` public; `835f6199` rejected; `f2e41d42` complete local candidate | `main` | Local evidence is sound: 113/113 covered, all 103 historical JSON/PNG bytes preserved, 28 hostile cases pass, and second independent local candidate Tier B is 100, making it content-ready for owner-approved publication/CI. Current public-state BHS remains 70/Critical because GitHub is still at `eb750958`, the body publishes invalid legacy claims, and `f2e41d42` has no hosted runs. Public update/check/re-review requires explicit approval. |
| #293 | `98e9dec4` public; `454e4a32` rejected local repair | #292 | Tier B 70/Critical at iteration 6: BaseException rollback corruption, deterministic concurrent lost updates, and forged/aliased provenance. No seventh repair; withdraw the current PR and open no replacement absent an authorized real consumer. |
| #294 | `12aa1be6` public; `6e78cf41` local repair | formerly #293 | Tier B 100 on the original stacked/pre-transplant head, now archive-ref preserved. Its pool-shard code is independent; use a new transplant branch/replacement PR from final #292, with fresh exact-head gates. Do not force-push or retarget public #294. |
| #295 | `2c78195d` public; `730b305e` rejected local iteration-5 candidate | formerly #294 | Tier B 60/Critical reproduced state-publication races, invalid no-global promotion, false promotion with incomplete REPORT qrels, non-finite/duplicate-ID metrics, publication-time provenance TOCTOU, and dimension-mismatch enablement. Withdraw current #295; optionally open a separate nine-file evidence-only PR from final #292. |

The unpublished primary stack has no forecast conflict with #293 or #294. It
has one expected textual conflict with #295 in the last-session line of
`docs/next-session.md`; resolution must preserve both the lattice history and
the newer AEP/prime-ring truth.

## 3. Worktree and branch inventory

| Worktree | Branch/head | Preservation status |
| --- | --- | --- |
| primary | `codex/prime-ring-onion-method-dev` / V3 snapshot `8c7446fb` (pre-reconciliation anchor `7dec564d`; evidence `186590c8`) | reviewed 38-path allowlist and documentation reconciliation are committed locally; exact-head and 64-surface all-refs bundles plus V2-bound V3 validation pass; this acceptance-record update postdates V3 without changing source/evidence; push and exact remote equality proof remain |
| `agent-build` | `codex/recover-drift-research-20260722` / `b831493a` | clean unique 209-file commit; verified full-history bundle; archive ref only until public-content review |
| `h2-rerun` | `feat/h2-swap-rerun-clean` / `6c3e1847` | unique CUDA-guard commit; 24 local deletions remain untouched; five ignored outputs are hash-bound and privately archived |
| `relaxed-wozniak-271e04` | `bf23a47f` | commit already contained by `main`; ignored raw evidence is covered by a verified ZIP |
| `semantic-cache-h1` | `codex/crsv-onion-method-dev` / `59378814` | fully contained by the primary branch; no separate merge is required |
| `waypoint-recovery` | `codex/recover-waypoint-research-20260722` / `65ae99a4` | clean unique 203-file archival commit; 201 duplicate live files match the branch byte-for-byte; verified full-history bundle; raw agent-output transcripts require privacy/license review |
| `pr292-recondition` | `codex/pr292-metric-lineage-recondition` / `f2e41d42` (`835f6199` rejected) | clean complete local candidate; 113/113 coverage and verified incremental bundle pass; local candidate Tier B 100 and content-ready for owner-approved publication/CI, current public state 70/Critical |
| `pr293-fix` | `lattice/rung13-disintegration-20260714` / `454e4a32` | retained rejected worktree; full history is bundle/archive-ref preserved; no further repair on the current PR |
| `pr294-fix` | `lattice/rung17-diskpool-20260714` / `6e78cf41` | clean original stacked/pre-transplant Tier-B-100 candidate; exact incremental bundle/archive ref; use a new transplant branch/replacement PR |
| `pr295-fix` | `lattice/rung16-routing-20260714` / `730b305e` | clean rejected iteration-5 candidate; exact incremental bundle and local archive ref verified; withdraw current PR, optionally replace with evidence-only archive |

No worktree is currently marked locked or prunable.

## 4. Primary Git allowlist

The following current-tree classes are approved for Git preservation after the
pre-commit gates in Section 9:

1. Canonical research and handoff records:
   - `docs/next-session.md`;
   - `docs/research/prime-ring-remaining-hypotheses-2026-07.md`;
   - `docs/research/prime-ring-waypoint-method-dev-protocol-2026-07.md`;
   - `docs/research/prime-ring-leading-shell-proof-2026-07.md`;
   - `docs/research/lossless-residual-audit-2026-07.md`;
   - `docs/research/prime-ring-orbit-plank-mesh-protocol-2026-07.md`;
   - `docs/research/prime-ring-t1r-primary-claim-chart-2026-07.md`;
   - this plan;
   - `findings.md`, `progress.md`, and `task_plan.md`.
2. RB-10 implementation:
   - `prime_ring_action_nondegeneracy.py`;
   - `prime_ring_irreducible_factors.py`;
   - `prime_ring_joint_orbit_spectrum.py`;
   - `prime_ring_leading_shell.py`;
   - `prime_ring_learned_quotient.py`;
   - `prime_ring_rb10_contract.py`;
   - `prime_ring_transcript_cost.py`;
   - `run_prime_ring_rb10_experiments.py`.
3. The eight corresponding `tests/test_*.py` modules.
4. The four immutable RB-10 JSON evidence files in
   `artifacts/method-dev/prime-ring/`.
5. The AEP cycle scope lock, backlog, tracker, verification log, tracker
   pointer, and indexes.
6. `pyproject.toml`, limited to registering every root module added by the
   unpublished stack so an installed package does not silently omit them.

Staging must use this explicit allowlist. `git add -A`, force-adding ignored
directories, and wildcard staging of `.claude`, logs, weights, checkpoints,
databases, or recovery packages are forbidden.

## 5. Immutable RB-10 evidence ledger

The four files below were not regenerated during RB-12. Raw hashes were
recomputed independently; the three envelope semantic digests and the manifest
rebinding also pass.

| File | Bytes | Raw SHA-256 |
| --- | ---: | --- |
| `rb10-bounded-experiment-manifest.json` | 2,822 | `fd50d8292b1054d25408a41e720dd7e1cef64866c9b794a7505bc6f94ada9160` |
| `rb10-prw-a1-nondegeneracy-p11.json` | 5,838 | `b91c6445eb9b60cfc6221d2fb47e45f8e139879424f155979142e0211d9b16e2` |
| `rb10-prw-g2-algebraic-p7-n3.json` | 9,059 | `257acee3c62662e1935ea1ffa9665dda922205979f04ccb810f0e9405f9af86e` |
| `rb10-prw-jo1-exact-p11-k2.json` | 10,578 | `dc79226dd44acb43ff18afa132b504fbf4eab7e942674178c6f03ba9ad5dc19d` |

Recorded stage order is `jo1`, `a1`, `g2-p7`, each in a fresh child process.
The reconstructable runner invocation is:

```powershell
python run_prime_ring_rb10_experiments.py --stages jo1 a1 g2-p7
```

That exact historical command line was not written to the original artifact,
so it is recorded as reconstructable, not asserted as independently proven
producer history. The current audit environment is Python 3.11.9 on Windows
10 build 26200, NumPy 2.4.5, and psutil 7.2.2. It matches the present validation
environment; it is not retroactively claimed as the complete 2026-07-25
producer environment.

The Git commit containing source, tests, and these exact bytes is
`186590c8bde8311f39c17167110d5ba30b13a4fd`. This bookkeeping update binds that
immutable evidence commit without creating a self-referential hash.

## 6. Ignored and external-only residuals

### Preserve outside ordinary Git with a checksum manifest

- Private V2 inventory: 2,584 ignored/untracked files / 1,678,118,521 bytes
  across ten worktrees, two stashes, zero errors, and explicit publication
  dispositions. File SHA-256:
  `4c1192a5fd77f75c5cd1309f2398755453c2fcaab376fc7571d2f85a4599e98b`.
- Private recovery includes exact/incremental Git bundles, complete-history
  stash refs/bundle, two older evidence ZIPs, and a new primary/H2 evidence ZIP
  with 140 source payload files plus one internal manifest; all 141 ZIP entries
  were verified.
- Models/databases/checkpoints remain private/external-storage candidates.
  Hash-only analysis found 344 model/checkpoint files but only 177 unique
  contents, with 193,143,042 duplicate bytes; serialized tensors were not
  loaded.
- `scripts/scaffold_brain_dossier.py` is uniquely preserved in a verified
  one-entry ZIP. It is secret-clean and parses/lints, but fails formatting and
  lacks dedicated tests; recondition in a separate PR before publication.

The 120,804,181-byte drift bundle is above GitHub's ordinary 100 MiB blob
limit. Publishing the contained branch is preferable to committing the bundle,
but the branch remains private-to-local until its public-content scan passes.

### Sensitive; do not publish to this public repository

- Fourteen V2 entries totaling 692,906,577 bytes are classified
  `DO_NOT_PUBLISH_SENSITIVE`; they include free-form debug/terminal/local-state
  surfaces that must not enter the public repository.
- Local `.claude`/`.vscode` settings, terminal captures, credentials, and
  learned weights until separately reviewed.
- Recovery bundles or raw agent transcripts before privacy/content-license
  review.

### Generated and reproducible

- `__pycache__`, Ruff/Pytest caches, ordinary build directories, `nul`, lock
  files, and uncited prime-ring runtime/test logs.
- Two ignored RLM clones, both clean and byte-identical at upstream
  `alexzhang13/rlm` commit `ee149fc7c691f59fbedadfcbb6b3ba27b82bba20`.

No residual in these categories is deleted in RB-12.

## 7. Stashes and unreachable-object lane

- `stash@{0}` (`02b96bb6`) and `stash@{1}` (`25e50b9b`) now have exact local
  archive refs and a verified complete-history two-ref bundle. Neither stash
  was applied or dropped. Seven dossier blobs remain unique to the first stash.

Reachable refs pass `git fsck` when stale commit-graph use is disabled.
With commit-graph acceleration enabled, 17 missing commit IDs each emit two
stale commit-graph errors. With `core.commitGraph=false`, full fsck exits zero
with no missing objects, broken links, or invalid refs; dangling objects remain
unpruned. The three split commit-graph files are privately hash-fingerprinted.
No GC, prune, commit-graph rewrite, worktree cleanup, or stash drop is allowed
without itemized owner signoff.

## 8. GitHub governance boundary

Ruleset `13081256` requires a PR, one approval, code-owner review, resolved
threads, signed commits, and linear history. It permits the owner to bypass.
The repository has no `CODEOWNERS` file and only one collaborator, so an
ordinary independent GitHub approval is currently impossible.

The ruleset does not mechanically require the repository's §4 validator,
block-flag, smoke, schema-drift, lint, or test jobs. Describing those jobs as
branch-rule-enforced merge gates is an L13 process overclaim. This campaign
will still treat every one as mandatory evidence and will not merge a failed
or pending exact-head check.

Original commits will be retained on explicit archive refs before any
policy-compliant squash merge. If an admin bypass is used, the PR record must
state the exact reason: preservation of reviewed work despite the
single-collaborator approval deadlock. No failing test or unresolved review
thread will be bypassed.

Explicit approval is required before every public GitHub mutation, not merely
a branch push. This includes push/force-with-lease, PR title/body/base edits,
closing or withdrawing a PR, opening a replacement PR, resolving threads on
behalf of the owner, and merging. Local branches, bundles, tests, plans, and
read-only GitHub inspection do not imply approval for those mutations.

The repository's root `LICENSE` is Apache-2.0 while `README.md` and
`pyproject.toml` say MIT. This conflict predates RB-12. Changing the project
license is an owner/legal decision and is not silently folded into a
preservation PR. The PR must disclose the ambiguity and make no package-release
or relicensing claim.

## 9. Required gates before the primary preservation commit

1. Stop concurrent writers and rebuild the exact status/allowlist.
2. Recompute all four raw hashes and manifest/envelope binding without
   regeneration.
3. Run Gitleaks on `origin/main..HEAD` and the staged allowlist; do not scan or
   publish the known-sensitive ignored logs.
4. Run all focused RB-10/RB-11 and inherited prime-ring/CRSV/BCC1 tests.
5. Run Ruff check/format, Python 3.9 AST compatibility, block-flag, schema-drift,
   smoke, whitespace, and object-size checks.
6. Build an isolated wheel without dependencies and import every newly
   registered module from outside the source tree.
7. Stage only the allowlist and inspect `git diff --cached`, file modes, object
   sizes, and candidate hashes.
8. Commit the evidence/source tree, then update this plan/tracker with that
   commit SHA in a second commit.
9. Push with an explicit source/destination refspec and prove equality using
   both `git ls-remote` and the GitHub API.

### Gate execution snapshot before staging

- Four immutable RB-10 raw hashes and manifest/envelope bindings: pass without
  regeneration.
- Focused prime-ring: `334/334` pass. Inherited semantic-cache/BCC1/CRSV:
  `202/202` pass.
- Full workflow-equivalent unittest discovery with the existing offline cache:
  `3330/3330` pass, 11 skips, 144.059 seconds.
- `python -m ruff check .`: pass. Ruff format check on the 16 new allowlisted
  source/test files: pass. Repository-wide format check is a known non-gate
  baseline failure across 258 historical files and will not be repaired by an
  unrelated preservation rewrite.
- Python 3.9 AST parse: 500 applicable files pass. Block flag: clear.
  Schema-drift: pass. `git diff --check`: pass.
- Isolated wheel: 626,146 bytes; SHA-256
  `053de47bba3ee76835459f2d6dd2e63238174759388b3ab9fdabce46f829925d`;
  all 26 newly registered modules imported from the external install target.
- Online smoke: floor pass; ceiling locally blocked by Hugging Face TLS
  certificate verification. TLS was not disabled. Clean hosted-runner coverage
  remains required before merge.
- Committed-range and staged Gitleaks: pass. The final index matched the
  38-path allowlist exactly, all staged objects were mode `100644`, and no
  object approached 100 MB. Evidence/source commit
  `186590c8bde8311f39c17167110d5ba30b13a4fd` and bookkeeping commit
  `7dec564de77b6efb03a82cb87405820284c7a8aa` are local. A verified private
  full-history bundle exists. Public push and remote equality proof remain
  blocked pending explicit approval of the public destination and exact
  payload.

## 10. Dependency-aware merge order

1. Finish the primary bookkeeping update, obtain explicit approval for the
   exact public destination/payload, then publish and verify the primary branch;
   do not open a duplicate independent PR for the four #292 commits.
2. Local review of complete #292 candidate `f2e41d42` scored its content at
   Tier B 100, making it content-ready for owner-approved publication/CI, while
   the current public state remains 70/Critical. After explicit approval,
   update the PR title/body, fast-forward its public branch to the exact SHA,
   run hosted checks, and obtain a fresh exact-public-head review. Archive the
   accepted head and merge only at 100.
3. Land the small H2 CUDA-guard commit as a separate reviewed PR without the
   H2 worktree's 24 deletions.
4. Archive and withdraw current #293. Do not open a replacement now: a pure
   detector-output normalizer would have no authorized production consumer and
   would be scaffolding. If a real consumer is later authorized, start a fresh
   reduced PR that excludes DAG mutation, re-annealing, trusted provenance,
   transactional guarantees, and `DONE` claims; record the removed scope in
   Deferred Scope and Carried Debt.
5. Create a new #294 transplant branch/replacement PR from the final accepted
   #292, preserving its old head on the archive ref. Do not force-push or
   retarget public #294. Transplant only its own code commits rather than replaying
   rejected #293 ancestry: `e12b7668`, `d8ccf70c`, `9d211613`, then
   `6e78cf41`, all with `cherry-pick -x`. Do not transplant the stale
   #293-dependent `12aa1be6` `DONE` documentation commit. Rebuild the roadmap
   from final #292, then rerun exact-head tests, hosted checks, and Tier B.
6. Archive and withdraw current #295. Iteration five
   scored Tier B 60/Critical after reproducing state-publication races, invalid
   no-global promotion, false promotion with incomplete REPORT qrels,
   non-finite/duplicate-ID metric failures, publication-time provenance TOCTOU,
   and dimension-mismatched engine enablement; no sixth repair loop is
   permitted. A separate replacement may be evidence-only, or a later fresh
   component PR may contain only isolated router-freeze and finite-arithmetic
   work on a fresh branch. Code
   independence is not safety acceptance: do not transplant the rejected
   promotion-plane/engine chain. Preserve the four consumed lock/REPORT
   artifacts byte-for-byte as historical fail-closed evidence; any new
   campaign must use new identifiers and paths. The defensible archival
   replacement is independent of #294: from final accepted #292, extract eight
   exact preregistration/manifest/lock/marker/reconciliation blobs and write
   one new narrowed disposition document. Do not cherry-pick any #295 commit
   wholesale. Full withdrawal remains safer if the archival prose cannot avoid
   implying accepted source attribution or metric validity.
7. Reconcile `main` into the unpublished research branch, preserving the newer
   next-session/AEP state and all original commits on the archive ref.
8. Split the large unpublished research history into reviewable PR themes if
   required by ARCH-AEP: semantic/BCC1, CRSV, waypoint code/evidence, bounded
   controls/falsifiers, cascade gates, and RB-10/RB-11 preservation.
9. Run exact-head Tier B and GitHub checks on every PR, merge in dependency
   order, and verify every merged/ref-only commit is reachable from either
   refreshed `main` or an explicit remote archive ref.
10. Reconcile the private V2 inventory into a final post-merge residual ledger.
    Stop and request owner approval separately for each proposed cleanup action.

## 11. Cleanup approval contract

For each future cleanup candidate, the owner will receive:

- exact worktree/branch/stash/path;
- current byte count and hash or commit;
- remote containment proof;
- recovery procedure;
- whether it is duplicate, superseded, generated, sensitive, or unresolved;
- precise consequence of deletion/removal.

No batch approval will be inferred from this merge campaign.
