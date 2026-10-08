# T0 A2 PR metadata receipt

Timestamp local: `2026-08-29T21:52:40.4647595-04:00`.

## Commands

```powershell
gh --version
gh auth status --active --hostname github.com --json hosts
git remote get-url origin
gh pr list --repo mattmre/CHELATEDAI --state open --limit 1000 --json 'number,title,headRefName,baseRefName,isDraft,mergeable,statusCheckRollup' --jq 'map({number,title,headRefName,baseRefName,isDraft,mergeable,checkRollupTotal:((.statusCheckRollup // []) | length),checkRollupCounts:(([ (.statusCheckRollup // [])[] | (.conclusion // .state // .status // "UNKNOWN") ] | group_by(.) | map({key:.[0],value:length}) | from_entries))})'
```

## CLI/account output

```text
gh version 2.96.0 (2026-07-02)
https://github.com/cli/cli/releases/tag/v2.96.0
origin=https://github.com/mattmre/CHELATEDAI.git
```

```json
{"hosts":{"github.com":[{"state":"success","active":true,"host":"github.com","login":"mattmre","tokenSource":"keyring","scopes":"gist, read:org, repo, workflow","gitProtocol":"https"}]}}
```

## Quoted PR metadata output

```json
[
  {"baseRefName":"main","checkRollupCounts":{"FAILURE":7,"SUCCESS":17},"checkRollupTotal":24,"headRefName":"codex/egv-avo-nemotron-protocol-20260825","isDraft":true,"mergeable":"MERGEABLE","number":308,"title":"docs(research): preregister EGV AVO/Nemotron recommissioning"},
  {"baseRefName":"main","checkRollupCounts":{"FAILURE":1,"SUCCESS":11},"checkRollupTotal":12,"headRefName":"autoresearch/maintain-and-grow-a-repo-grounded-related-works-20260815","isDraft":true,"mergeable":"MERGEABLE","number":297,"title":"record: autoresearch related-works 20260815 (gx10-eba1)"},
  {"baseRefName":"main","checkRollupCounts":{"FAILURE":1,"NEUTRAL":1,"SUCCESS":10},"checkRollupTotal":12,"headRefName":"codex/prime-ring-onion-method-dev","isDraft":true,"mergeable":"CONFLICTING","number":296,"title":"research: preserve consolidated method-development evidence"},
  {"baseRefName":"lattice/rung17-diskpool-20260714","checkRollupCounts":{"SUCCESS":1},"checkRollupTotal":1,"headRefName":"lattice/rung16-routing-20260714","isDraft":false,"mergeable":"MERGEABLE","number":295,"title":"lattice rung 16: quant-aware routing plane - FAIL-CLOSED on both arenas"},
  {"baseRefName":"lattice/rung13-disintegration-20260714","checkRollupCounts":{"SUCCESS":1},"checkRollupTotal":1,"headRefName":"lattice/rung17-diskpool-20260714","isDraft":false,"mergeable":"MERGEABLE","number":294,"title":"lattice rung 17: disk pool slice - block-graph read with host parity"},
  {"baseRefName":"lattice/phase2-continue-20260713","checkRollupCounts":{"SUCCESS":1},"checkRollupTotal":1,"headRefName":"lattice/rung13-disintegration-20260714","isDraft":false,"mergeable":"MERGEABLE","number":293,"title":"lattice rung 13: detector-driven Evidence-DAG disintegration loop"},
  {"baseRefName":"main","checkRollupCounts":{"FAILURE":1,"NEUTRAL":1,"SUCCESS":14},"checkRollupTotal":16,"headRefName":"lattice/phase2-continue-20260713","isDraft":false,"mergeable":"CONFLICTING","number":292,"title":"lattice(phase2): H2/H5/H4 verdicts + close CD-H1-01/CD-A2-01 + docs-truth sync"},
  {"baseRefName":"main","checkRollupCounts":{"FAILURE":2,"SUCCESS":21},"checkRollupTotal":23,"headRefName":"feat/brain-file-map-b0-b1","isDraft":false,"mergeable":"MERGEABLE","number":278,"title":"docs(brain): file-map dossiers B0+B1 (7 Tier A modules)"},
  {"baseRefName":"main","checkRollupCounts":{"CANCELLED":1,"FAILURE":1,"SUCCESS":10},"checkRollupTotal":12,"headRefName":"feat/live-progress-tracker-20260606","isDraft":true,"mergeable":"CONFLICTING","number":257,"title":"feat(progress): Live ChelatedAI progress branch with README and changelog"},
  {"baseRefName":"main","checkRollupCounts":{"FAILURE":2,"SUCCESS":10},"checkRollupTotal":12,"headRefName":"feat/shim-cd01-unblock-first-probe-design","isDraft":true,"mergeable":"MERGEABLE","number":256,"title":"feat(unblock): sustained 10min zero-wall loop + 10-agent SHIM-CD-01 first thin SIP probe design (VectorSteerer minimal guarded diff + full review package)"}
]
```

The first unquoted field-list attempt failed during local PowerShell argument parsing before an API request. No body, diff, file list, comment, review, thread, or log was opened in Tier 0.
