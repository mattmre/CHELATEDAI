export const meta = {
  name: 'lattice-recon-swarm',
  description: 'Grok recon swarm: extract exact runnable commands + gotchas for H2 re-run, CD-A2-01 closure, and H5 verdict campaign before GPU spend',
  phases: [
    { title: 'Recon', detail: '3 parallel Grok agents read the merged drivers' },
    { title: 'Synthesize', detail: 'merge into one GPU execution runbook' },
  ],
}

const WT = 'D:\\GITHUB\\CHELATEDAI\\.claude\\worktrees\\relaxed-wozniak-271e04'
const PANEL = 'D:\\GITHUB\\CHELATEDAI\\docs\\waypoint-research-2026-06-09\\panel'
const GROK = 'C:\\Users\\mattm\\.grok\\bin\\grok.exe'

function driveGrok(fileKey, promptBody) {
  const promptPath = `${PANEL}\\wf3_${fileKey}.md`
  const outPath = `${PANEL}\\out_wf3_${fileKey}.txt`
  return [
    `You are driving an external Grok CLI agent. Do exactly this and nothing else:`,
    `1. Write the following text verbatim to \`${promptPath}\` with your Write tool:`,
    `-----BEGIN PROMPT-----`,
    promptBody,
    `-----END PROMPT-----`,
    `2. Run this PowerShell command with tool timeout 570000 ms and wait:`,
    `   & "${GROK}" -m grok-4.5 --effort low --cwd "${WT}" --prompt-file "${promptPath}" *> "${outPath}"`,
    `3. Read \`${outPath}\` and return its full contents prefixed "GROK ${fileKey}:".`,
    `   If it errored/timed out, return "GROK ${fileKey}: CLI_FAILED" plus partial output.`,
  ].join('\n')
}

const COMMON = [
  `Repo root is the cwd (branch lattice/phase2-continue-20260713 = origin/main @ 34ce4b56).`,
  `Machine: RTX 3090, HF cache has all-MiniLM-L6-v2 / all-mpnet-base-v2 / bge-large / e5-base-v2;`,
  `SciFact + NFCorpus + FiQA2018 + ArguAna datasets cached. Offline flags normally set; real-model`,
  `tests gate on CHELATED_RUN_REAL_MODEL_TESTS=1. Deliver EXACT PowerShell commands, not prose.`,
].join(' ')

const R1 = [
  `# Recon: H2 re-run commands (closes CD-H1-01)`, ``, COMMON, ``,
  `CD-H1-01 says the committed swap-campaign result docs (docs/drift-recovery-swap-results-2026-06.md`,
  `+ docs/drift-recovery-swap-nfcorpus-results-2026-06.md) carry pre-H1-fix C3a numbers; closure = re-run`,
  `both campaigns (SciFact + NFCorpus) on post-#276 main and regenerate both auto-generated docs.`,
  `Read the campaign driver (PR #273 added it — likely run_swap_campaign.py or similar; find it), the H1`,
  `fix (#276, commit d5fc7dfa) to know what changed, and the doc-regeneration path (the docs say`,
  `auto-generated — find the generator). Deliver: (1) the exact command(s) to re-run BOTH campaigns with`,
  `the right seeds/conditions to reproduce the doc tables (C0/C2/C2O/C3a/C4a, 3 seeds, budget sweep if`,
  `included), (2) expected runtime + GPU footprint, (3) which files the run regenerates vs which need`,
  `manual number refresh (paper Section 5 mention), (4) gotchas (env vars, offline flags, output dirs`,
  `like experiment_runs/, whether the driver commits artifacts or writes gitignored dirs).`,
].join('\n')

const R2 = [
  `# Recon: CD-A2-01 closure requirements`, ``, COMMON, ``,
  `CD-A2-01: the real swap-backend path (query_encoder_drift.QueryEncoderDrift._backend ->`,
  `embedding_backend.create_embedding_backend loading real all-mpnet-base-v2) is not exercised in default`,
  `CI; closure = "the PR-A4 real-model campaign actually running the all-mpnet-base-v2 swap path on a`,
  `networked/cached machine (CHELATED_RUN_REAL_MODEL_TESTS=1) and recording the artifact".`,
  `Read the gated tests (#271 made them opt-in) and the A4 campaign driver. Deliver: (1) the exact`,
  `command(s) that satisfy this closure (is the H2 campaign re-run sufficient by itself, or must the`,
  `gated unittest also run? name the test module), (2) what artifact must be recorded and where it should`,
  `live so the debt row can cite it, (3) the exact next-session.md row edit that closes it honestly.`,
].join('\n')

const R3 = [
  `# Recon: H5/H4 verdict campaign commands`, ``, COMMON, ``,
  `PRs: #279 H5a steering-post bank + prune/re-anneal lifecycle; #281-283 H5b/S2 post-bank-from-anchors`,
  `builder + runtime glue + conditions C5/C5s/C5r; #290 H5 post-bank head-to-head campaign driver`,
  `(C5/C5s/C5r verdict); #291 H4 compound_cycles knob. No results(h5) commit exists — the verdict`,
  `campaign has not been run. Read the #290 driver. Deliver: (1) the exact command(s) to run the full`,
  `head-to-head (which conditions, seeds, datasets it covers; does it include H4 compound_cycles arms),`,
  `(2) expected runtime/GPU, (3) what verdict artifact/doc it writes and whether a results doc generator`,
  `exists, (4) what the preregistered decision rule is (what makes C5/C5s/C5r a win/loss vs C3a/C4a),`,
  `(5) gotchas. If the driver expects artifacts from the H2 re-run (fresh baselines), say so explicitly —`,
  `that decides run ordering.`,
].join('\n')

const ITEMS = [
  { key: 'R1-h2-rerun', prompt: R1 },
  { key: 'R2-a2-closure', prompt: R2 },
  { key: 'R3-h5-campaign', prompt: R3 },
]

phase('Recon')
const results = await parallel(ITEMS.map(item => () =>
  agent(driveGrok(item.key, item.prompt), {
    label: `grok-recon:${item.key}`, phase: 'Recon', agentType: 'general-purpose',
  }).then(r => ({ key: item.key, report: r }))
))

phase('Synthesize')
const runbook = await agent(
  [
    `Merge these three Grok recon reports into ONE GPU execution runbook for the operator (Fable chair).`,
    `Sections: (1) run order with rationale (especially whether H5 needs H2's fresh artifacts first),`,
    `(2) exact PowerShell commands in sequence with env vars, (3) expected artifacts + where they land,`,
    `(4) the exact debt-row closures + doc regenerations each run unlocks, (5) risks/gotchas. Be terse`,
    `and mechanical — commands must be copy-paste runnable.`,
    ``,
    ...results.filter(Boolean).map(r => `## ${r.key}\n${r.report}\n`),
  ].join('\n'),
  { label: 'runbook-synthesis', phase: 'Synthesize' }
)

return { recon: results.filter(Boolean).map(r => r.key), runbook }
