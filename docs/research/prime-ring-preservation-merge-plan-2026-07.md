# CHELATEDAI Lossless Preservation and Merge Plan

Date: 2026-07-27
Cycle: `AEP-20260727-7`
Status: `IN_PROGRESS_PRESERVATION_FIRST`; cleanup is not authorized.

## 1. Decision

The current work is recoverable, but it is not yet safe to call merged or
clean. The immediate safe action is to publish the reviewed primary research
tree and exact commit history. The existing lattice PR stack must then be
repaired and merged in dependency order before the unpublished research stack
can target `main`.

No worktree, stash, ignored artifact, recovery package, branch, or unreachable
object will be deleted, pruned, rewritten, or consolidated in this cycle.
Cleanup begins only after merged-state verification and explicit owner approval
for each residual item.

## 2. Live Git and GitHub boundary

- Public repository: `mattmre/CHELATEDAI`.
- Default branch: `main`.
- Refreshed `origin/main`: `34ce4b5632e0d9cd2a16c29e0e1acc42e645b9c2`.
- Primary local branch before the preservation commit:
  `codex/prime-ring-onion-method-dev` at
  `41779be47442fc74fcd93e0c68bd7ca9dc115b99`, exactly 11 commits ahead and
  zero behind.
- Commits 1-4 (`2b0fa044` through `eb750958`) are already the exact published
  head of open PR #292.
- Commits 5-11 (`11d548bd` through `41779be4`) have no GitHub ref and are the
  first publication priority.
- No remote ref contains the primary pre-preservation head.

### Existing dependency stack

| PR | Head | Dependency | Current disposition |
| --- | --- | --- | --- |
| #292 | `eb750958` | `main` | exact-head CI green; no unresolved review thread; owner/admin merge path required because ordinary approval is impossible |
| #293 | `98e9dec4` remote; `b0d72d12` first local repair | #292 | fresh Tier B rejected the unpushed repair for three callback/state-integrity gaps; second repair active |
| #294 | `12aa1be6` remote; `9d211613` first local repair | #293 | fresh Tier B rejected the unpushed repair for strict numeric typing and unbounded padding; second repair active |
| #295 | `2c78195d` remote; `7a79b0ec` first local repair | #294 | fresh Tier B found finite-number and whole-SELECT/REPORT concurrency gaps; unpushed and not merge-ready |

The unpublished primary stack has no forecast conflict with #293 or #294. It
has one expected textual conflict with #295 in the last-session line of
`docs/next-session.md`; resolution must preserve both the lattice history and
the newer AEP/prime-ring truth.

## 3. Worktree and branch inventory

| Worktree | Branch/head | Preservation status |
| --- | --- | --- |
| primary | `codex/prime-ring-onion-method-dev` / evidence commit `186590c8bde8311f39c17167110d5ba30b13a4fd` | reviewed 38-path allowlist is committed locally; bookkeeping commit, push, and exact remote equality proof remain |
| `agent-build` | `codex/recover-drift-research-20260722` / `b831493a` | clean unique 209-file commit; verified full-history bundle; archive ref only until public-content review |
| `h2-rerun` | `feat/h2-swap-rerun-clean` / `6c3e1847` | unique CUDA-guard commit; 24 local deletions must not be committed; five ignored outputs need a separate manifest |
| `relaxed-wozniak-271e04` | `bf23a47f` | commit already contained by `main`; ignored raw evidence is covered by a verified ZIP |
| `semantic-cache-h1` | `codex/crsv-onion-method-dev` / `59378814` | fully contained by the primary branch; no separate merge is required |
| `waypoint-recovery` | `codex/recover-waypoint-research-20260722` / `65ae99a4` | clean unique 203-file archival commit; verified full-history bundle; archive ref only until transcript/privacy review |
| `pr293-fix` | local #293 branch / first repair `b0d72d12` | retained repair worktree; fresh Tier B rejected this head and second repair is active; do not publish until re-reviewed |
| `pr294-fix` | local #294 branch / first repair `9d211613` | retained repair worktree; fresh Tier B rejected this head and second repair is active; do not publish until re-reviewed |
| `pr295-fix` | local #295 branch / first repair `7a79b0ec` | retained repair worktree; fresh Tier B rejected this head; do not publish until reconditioned and re-reviewed |

No worktree is currently marked locked or prunable.

## 4. Primary Git allowlist

The following current-tree classes are approved for Git preservation after the
pre-commit gates in Section 9:

1. Canonical research and handoff records:
   - `docs/next-session.md`;
   - `docs/research/prime-ring-remaining-hypotheses-2026-07.md`;
   - `docs/research/prime-ring-waypoint-method-dev-protocol-2026-07.md`;
   - `docs/research/prime-ring-leading-shell-proof-2026-07.md`;
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

- `.claude/recovery/`: 10 files / 166,517,869 bytes, including three verified
  full-history bundles and two verified evidence ZIPs.
- Main `experiment_runs/`: 240 files / 183,607,122 bytes. At least 104
  non-Git files outside the session-29 archive still need a source-path/hash
  manifest.
- `checkpoints/`: 315 files / 295,258,480 bytes; 169 distinct hashes.
- Unique adapter-weight contents, `db_scifact_evolution`, four retired-session
  notes, H2's five unique ignored outputs, and the unique agent checkpoint
  metadata.
- `scripts/scaffold_brain_dossier.py` until its hard-coded local path is
  sanitized and its intended repository role is approved.

The 120,804,181-byte drift bundle is above GitHub's ordinary 100 MiB blob
limit. Publishing the contained branch is preferable to committing the bundle,
but the branch remains private-to-local until its public-content scan passes.

### Sensitive; do not publish to this public repository

- Five `chelation_debug.jsonl` files totaling approximately 691.7 MB; their
  schema permits free-form query/message content and contains machine/runtime
  metadata.
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

- `stash@{0}` (`02b96bb6`):
  eleven added implementation/test blobs already match current HEAD; seven
  dossier versions are unique. Preserve only those seven on a dedicated rescue
  ref after content review; do not apply the whole stash.
- `stash@{1}` (`25e50b9b`):
  one historical swap-results document is older than the current corrected
  version. Retain audit-only and do not overwrite current HEAD.

Reachable refs pass `git fsck` when stale commit-graph use is disabled.
The object database nevertheless contains 53,283 dangling commits, 1,706
dangling trees, 30 dangling blobs, and 17 unreadable commit-graph entries.
No GC, prune, commit-graph rewrite, worktree cleanup, or stash drop is allowed
until this lane is separately catalogued or snapshotted.

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
  `186590c8bde8311f39c17167110d5ba30b13a4fd` is local. The bookkeeping commit
  and remote equality proof remain open.

## 10. Dependency-aware merge order

1. Publish and verify the primary branch for durability; do not open a
   duplicate independent PR for the four #292 commits.
2. Create an archive ref for #292, then merge #292 after its body and exact head
   are rechecked.
3. Land the small H2 CUDA-guard commit as a separate reviewed PR without the
   H2 worktree's 24 deletions.
4. Repair, re-test, re-review, and merge #293.
5. Repair, re-test, re-review, and merge #294.
6. Repair the freeze/register race, re-test, re-review, and merge #295.
7. Reconcile `main` into the unpublished research branch, preserving the newer
   next-session/AEP state and all original commits on the archive ref.
8. Split the large unpublished research history into reviewable PR themes if
   required by ARCH-AEP: semantic/BCC1, CRSV, waypoint code/evidence, bounded
   controls/falsifiers, cascade gates, and RB-10/RB-11 preservation.
9. Run exact-head Tier B and GitHub checks on every PR, merge in dependency
   order, and verify every merged/ref-only commit is reachable from either
   refreshed `main` or an explicit remote archive ref.
10. Generate the post-merge residual ledger. Stop and request owner approval
    separately for each proposed cleanup action.

## 11. Cleanup approval contract

For each future cleanup candidate, the owner will receive:

- exact worktree/branch/stash/path;
- current byte count and hash or commit;
- remote containment proof;
- recovery procedure;
- whether it is duplicate, superseded, generated, sensitive, or unresolved;
- precise consequence of deletion/removal.

No batch approval will be inferred from this merge campaign.
