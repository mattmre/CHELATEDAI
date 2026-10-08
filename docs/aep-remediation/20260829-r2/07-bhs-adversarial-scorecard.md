# Round 2 BHS v3.7.1 scorecard — audit artifact

## Tier A self-assessment

- `BHS_SELF_DRAFT: 90`
- `BHS_SELF_DRAFT_AGENT: /root (A4 coordinator)`
- `BHS_SELF_SEVERITY: important`
- `TIER_A_ITERATION: 1 of 5`
- `BHS_PASS_STATUS: near-complete audit artifact; not shippable`
- `PRODUCT_STATUS: BLOCKED / plan-only / zero fixes claimed`
- `OPERATOR_OVERRIDE: none`
- `PLATEAU: none in Round 2`

### Weighted scope

| Audit-artifact item | Weight | Earned | Evidence |
|---|---:|---:|---|
| strict Tier 0→1→2→3 chronology and exact inventory | 10 | 10 | `00`, `01`, `02` indexes and verification chronology |
| all dirty worktrees and all open PR discussion/check surfaces opened | 15 | 15 | 6 dirty worktrees; 10 PRs; 27 unresolved threads |
| FE/BE/boundary source and runtime-contract coverage | 15 | 15 | `03-full-repo-map.md`, panel convergence, twelve active findings |
| cold finding packet completeness and deduplication | 20 | 20 | 13 packets: 12 active, 1 explicit duplicate; normalized by A1–A3 |
| fresh local evidence gates | 15 | 15 | 3,633-test full suite, lint, focused modules, failure probes |
| tracker/playbook/panel/summary output contract | 15 | 15 | all required Round 2 outputs present before Tier B |
| durable fresh-checkout/canonical registration | 5 | 0 | entire Round 2 corpus is untracked and not registered in the canonical ARCH-AEP tracker |
| served-page browser integration gate | 5 | 0 | no checked-in repository browser harness; source/unit claims only |
| **Total** | **100** | **90** | only 100 ships |

## Tier A disclosures

- `SKIPPED_GATES: SERVED_PAGE_BROWSER`
- `CARRY_FORWARD: 21 active Round 1 product findings; none silently closed`
- `DEFERRED_SCOPE: product fixes; staging/commit/push; PR mutation; deployment; official Spark; hardware/model opt-in runs`
- `DURABILITY_GAP: docs/aep-remediation/20260829-r2 is untracked and therefore absent from a fresh checkout`
- `CANONICAL_REGISTRATION_GAP: cycle is not selected by the tracked ARCH-AEP tracker pointer`
- `PROCESS_VARIANCE: first A3 Tier 3 assignment stopped at prompt classification; revised local source/unit assignment completed and the root-cause record is preserved`

The self-score is capped at 90 rather than caveated upward. The full suite and lint cannot compensate for missing durability or the absent served-page gate. Product defects are not counted as audit-artifact defects because this is explicitly a remediation plan; they remain open in the controlling backlog.

Tier A freeze: `tier-a-candidate.sha256` contains 26 entries; independent local recomputation found zero mismatches. Manifest SHA-256: `821ba1350b91c059c144cc39d522865b884bae7fcac72bf1b76ba02ca7bbd5cf`. This scorecard and the manifest file itself are excluded so the independent verdict can be appended without changing the reviewed corpus.

## Fresh Tier B

- `BHS_TIER_B: 70`
- `BHS_TIER_B_AGENT: /root/r2_tier_b`
- `BHS_TIER_B_SEVERITY: critical`
- `BHS_OFFICIAL: 70` (`min(90, 70)`)
- `ONLY_100_PASSES: yes`
- `BHS_PASS_STATUS: fail / return to Tier A`
- `PRODUCT_STATUS: BLOCKED / plan-only / unfixed`
- `MANIFEST_VERIFICATION: PASS — 26/26 unique entries, zero missing or mismatched`
- `MANIFEST_SHA256: 821ba1350b91c059c144cc39d522865b884bae7fcac72bf1b76ba02ca7bbd5cf`

The fresh reviewer opened all required surfaces, validated 13/13 packet structure, confirmed the exact 5 ready / 7 blocked / 1 duplicate disposition, recomputed Round 2 and cumulative severity totals, found zero broken local links, and checked every packet's current reproduction/source command locally. The packet mechanisms and defect boundaries held.

### Tier B Critical gap

The manifest-bound corpus does not independently preserve the load-bearing command streams behind several audit-completeness claims. Inventory/PR/thread chronology and the 3,633-test/lint results are represented by normalized receipts and summary rows rather than raw command streams with timestamps. The reviewer therefore accepted internal consistency but rejected independent proof of the claimed complete audit execution. This caps Tier B at 70.

### Tier B important gaps

1. The Round 2 corpus is wholly untracked; `git ls-files` selects zero cycle files, and the canonical ARCH-AEP tracker/backlog still selects `AEP-20260727-7`. The disclosed durability caveat earns no completion points.
2. Repository-local BHS authority is v3.3/L1–L13 while this cycle correctly follows the operator-specified v3.7.1/L1–L16 authority from the kit's `v3.5/` path. A cold offline agent cannot resolve that authority solely from this checkout.
3. Three cold-command precision issues remain in the frozen corpus: PR295's PowerShell wrapper can finish exit 0 after the expected Python failure; REPO-009's combined source scan can match the static client and hide its intended no-test marker; some future post-fix gates describe the test to add rather than naming a currently executable command.
4. Owners are exact functional roles rather than assigned agents/branches, and a few ready packets leave a bounded transaction/config decision to the implementing agent.

### L4 score-gap disclosure

The 20-point `90 → 70` self/Tier-B gap is a material L4 audit-closure miss. Tier A correctly disclosed untracked/canonical and browser gaps but still overcredited normalized summaries as independently durable execution evidence and overestimated cold command precision. No caveat restores those points.

### Terminal Round 2 disposition

Round 2 closes at `BHS_OFFICIAL: 70/Critical`; only 100 passes. A second Tier A rewrite is not started because the controlling gap requires durable tracked/canonical authority outside the requested output directory and original raw chronology cannot be manufactured retroactively. Another untracked summary rewrite would repeat the same failure class. Round 3 must preserve raw receipts from Tier 0 onward, sharpen every cold command before freeze, and retain the durability limitation explicitly.
