# T1 A2 AVO and integration receipt

No audit agent wrote these worktrees. Two owner lanes changed during read-only inspection, so no stable defect claim was made.

## Moving snapshots

```text
CHELATEDAI-EGV-AVO-CORPUS-RUNNER
HEAD=c459d1599a83ec88d96981db07052c3ef4009a27
22:02:31-04:00 four tracked files, 319+/13-
22:08:55-04:00 five tracked files plus docs/egv-public-expression-dsl-v1.md
final patch-id=5d74cf352d7b82d1bc0482aa130af2bd9f5c23fd

CHELATEDAI-EGV-AVO-RUNTIME
HEAD=88b0f389fa56c9c879c0c6a04e7f3fceef9dc243
22:02:31-04:00 scheduler deleted, 0+/1349-
22:05:41-04:00 23726 bytes, sha256=4CC897DDDEB949BD34834DE56D28BE23882937DB10784CF64976E0A2B5912A13, 264+/972-
22:08:55-04:00 71317 bytes, sha256=C566351D4007C38068036C60E6EABDF7D8F706C686433CD7F8EFA0715CC544B6, 937+/654-
final patch-id=8e02f3f4aa35b022cc2d7ca43a9d1ee64a7c908f
```

Disposition for both: `MOVING_TARGET / BLOCKED`. Dependency: pause the owning writer and take one new immutable worktree, porcelain, and patch snapshot.

## Stable residue checks

```text
NEMOTRON HEAD=dc295442af7623acfd69b1c487a59b80cd4dc78f
.gitleaks-egv-avo.json entries=11 rule=generic-api-key unique-secret=REDACTED references=0
sha256=CB8038923A290E6411653CE7740E2B2718779FDCAAA2AD7D051FD2F1717461BC

INTEGRATION HEAD=841b314b3f1e1524fab035a2bf2cb14cd0865c3b
source.bin size=28 sha256=6E371B4DA6A16D28A34A4DD3601EE8179717975D2EB771C9C2A1DCF101316B04
target.bin size=28 sha256=6E371B4DA6A16D28A34A4DD3601EE8179717975D2EB771C9C2A1DCF101316B04
references=0
```

These are untracked operational residue with no FE/backend product wiring. They are not treated as product defects or test evidence under the requested no-cosmetic-nits boundary.
