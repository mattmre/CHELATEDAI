# T0 A1 worktree receipt

Snapshot: local `2026-08-29T21:53:45.4563772-04:00` through `21:53:48.0866294-04:00`; UTC `2026-08-30T01:53:45.4613103Z` through `01:53:48.0867486Z`.

## Commands

```powershell
git remote -v
git remote get-url origin
git branch --show-current
git rev-parse HEAD
git rev-parse --abbrev-ref --symbolic-full-name '@{upstream}'
git rev-parse '@{upstream}'
git rev-list --left-right --count '@{upstream}...HEAD'
git status --porcelain=v1 --untracked-files=all
git worktree list --porcelain
git -C <each-exact-worktree-path> status --porcelain=v1 --untracked-files=all
gh pr list --repo mattmre/CHELATEDAI --state open --limit 100 --json number,headRefName,url
```

## Root receipt

```text
origin=https://github.com/mattmre/CHELATEDAI.git
branch=codex/prime-ring-onion-method-dev
HEAD=65c9085cd048e8a7351a53e87666fdd5639e612b
upstream=origin/codex/prime-ring-onion-method-dev
upstream_HEAD=65c9085cd048e8a7351a53e87666fdd5639e612b
upstream...HEAD left/right=0 0
ROOT_PORCELAIN_PATH_COUNT=150
ROOT_PORCELAIN_UTF8_LF_SHA256=1d072986182f83a75a07bee0b9adc3a2e8f77e22a7386ead6823aa565f100b1b
```

First ten porcelain rows:

```text
 M .gitattributes
 M docs/next-session.md
 M findings.md
 M paired_intervention_experiments.py
 M progress.md
 M run_paired_intervention_sanity.py
 M tests/test_paired_intervention_experiments.py
?? .codex-egv-8f-review-followup.md
?? .codex-egv-aa8671e-review-prompt.md
?? .codex-egv-canary-r2-execution.json
```

Last ten porcelain rows:

```text
?? run_qscci.py
?? test_qscci_blind_followup_fixture.py
?? test_qscci_followup_fixture_boundary.py
?? test_qscci_frozen_contract.py
?? test_qscci_numerics.py
?? test_qscci_selection_gates.py
?? test_qscci_supervisor.py
?? test_qscci_v4_archive.py
?? validate_qscci_blind_followup_fixture.py
?? verify_paired_intervention_sanity_v2_archive.py
```

## Worktree metadata stream

Format: `path | branch-or-detached | HEAD | porcelain-path-count`.

```text
D:/GITHUB/CHELATEDAI | codex/prime-ring-onion-method-dev | 65c9085cd048e8a7351a53e87666fdd5639e612b | 150
D:/GITHUB/CHELATEDAI/.claude/worktrees/agent-build | codex/recover-drift-research-20260722 | b831493a325b4df93615c8e5e4f0989c112eaaa6 | 0
D:/GITHUB/CHELATEDAI/.claude/worktrees/h2-rerun | feat/h2-swap-rerun-clean | 6c3e1847830ec231157a421735a5a34d7d71a658 | 24
D:/GITHUB/CHELATEDAI/.claude/worktrees/pr292-recondition | codex/pr292-metric-lineage-recondition | f2e41d420c4e6100ada8508635c40d81c5319866 | 0
D:/GITHUB/CHELATEDAI/.claude/worktrees/pr293-fix | lattice/rung13-disintegration-20260714 | 454e4a32453b29040eda9c286ed120a158a265b4 | 0
D:/GITHUB/CHELATEDAI/.claude/worktrees/pr294-fix | lattice/rung17-diskpool-20260714 | 6e78cf419c59ada7d4dd3ad2dace8f8721552451 | 0
D:/GITHUB/CHELATEDAI/.claude/worktrees/pr295-fix | lattice/rung16-routing-20260714 | 730b305e8352b1c7f41e8b77062c4c7cba543dc6 | 0
D:/GITHUB/CHELATEDAI/.claude/worktrees/relaxed-wozniak-271e04 | detached | bf23a47f754241e77fdc10b91c3f9deb6dfb8943 | 0
D:/GITHUB/CHELATEDAI/.claude/worktrees/semantic-cache-h1 | codex/crsv-onion-method-dev | 593788145c05e5634bab52915af6e9ae4212d73e | 0
D:/GITHUB/CHELATEDAI/.claude/worktrees/waypoint-recovery | codex/recover-waypoint-research-20260722 | 65ae99a4ae9b90c56febc6d8b84e1447c522ea3f | 0
D:/GITHUB/CHELATEDAI-EGV-ARCH | codex/egv-bootstrap-replay-closeout-20260821 | ebf6ebd78ac8f0a1dd952661e4c57a488f69e496 | 0
D:/GITHUB/CHELATEDAI-EGV-AVO-CORPUS-PROOF | codex/egv-avo-corpus-proof-20260829 | c459d1599a83ec88d96981db07052c3ef4009a27 | 0
D:/GITHUB/CHELATEDAI-EGV-AVO-CORPUS-RUNNER | codex/egv-avo-corpus-runner-20260830 | c459d1599a83ec88d96981db07052c3ef4009a27 | 3
D:/GITHUB/CHELATEDAI-EGV-AVO-INTEGRATION-20260829 | codex/egv-avo-integration-20260829 | bfcb1d45dbcbc08409e1166e66a2575600c4b043 | 0
D:/GITHUB/CHELATEDAI-EGV-AVO-NEMOTRON | codex/egv-avo-nemotron-protocol-20260825 | dc295442af7623acfd69b1c487a59b80cd4dc78f | 1
D:/GITHUB/CHELATEDAI-EGV-AVO-PILOT | codex/egv-avo-pilot-20260829 | 1a14d5484da2a4f98eaa062d4cc8b426f32abae0 | 0
D:/GITHUB/CHELATEDAI-EGV-AVO-RUNTIME | codex/egv-avo-runtime-20260829 | 88b0f389fa56c9c879c0c6a04e7f3fceef9dc243 | 1
D:/GITHUB/CHELATEDAI-EGV-BD-CAMPAIGN | codex/egv-bd-campaign-20260824 | 64e60c3fe0264a26415ec9cdd83a2b548ff1eaff | 0
D:/GITHUB/CHELATEDAI-EGV-CAMPAIGN | main | 0cb5c858cc52cd3b4fb575caf76c09895dbb2d97 | 0
D:/GITHUB/CHELATEDAI-EGV-EVAL | codex/egv-evaluation-20260822 | 1d1371ce10b6798482590cbd6e26175edf84e5f5 | 0
D:/GITHUB/CHELATEDAI-EGV-HELDOUT-PROD | codex/egv-heldout-production-20260824 | d72edd6d572f9d41be48f31f6b4ae8224dbb4891 | 0
D:/GITHUB/CHELATEDAI-EGV-INTEGRATION | codex/egv-live-integration-20260822 | 841b314b3f1e1524fab035a2bf2cb14cd0865c3b | 2
D:/GITHUB/CHELATEDAI-EGV-ORCHESTRATOR | codex/egv-commissioning-runner-20260822 | 6db80e86d5ab9be1ad5b771bd0bdb122665356c1 | 0
D:/GITHUB/CHELATEDAI-EGV-TRAINING | codex/egv-training-20260822 | 97a8237e3f1b92425b6802895c1fcbdeafeb2681 | 3898
D:/GITHUB/CHELATEDAI-RHPC1 | codex/rhpc1-method-dev-20260825 | 305986427850dcb393a6078900201a7aa84c5b33 | 13
```

Dirty worktrees: root, `h2-rerun`, AVO corpus runner, AVO Nemotron, AVO runtime, EGV integration, EGV training, and RHPC1. Open-head matches: root/#296, PR293, PR294, PR295, and AVO Nemotron/#308. Excluding base `main`, eighteen branch-attached worktrees have no open PR; one worktree is detached.
