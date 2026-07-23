export const meta = {
  name: 'l23-docs-swarm',
  description: 'Grok swarm: draft L2 CHANGELOG+decision and L3 docs-truth sync, each adversarially reviewed',
  phases: [
    { title: 'Draft', detail: 'Grok drafts L2 + L3 docs' },
    { title: 'Review', detail: 'fresh Grok adversarially reviews each draft vs artifacts' },
  ],
}

const WT = 'D:\\GITHUB\\CHELATEDAI\\.claude\\worktrees\\relaxed-wozniak-271e04'
const PANEL = 'D:\\GITHUB\\CHELATEDAI\\docs\\waypoint-research-2026-06-09\\panel'
const GROK = 'C:\\Users\\mattm\\.grok\\bin\\grok.exe'

function driveGrok(fileKey, promptBody) {
  const promptPath = `${PANEL}\\wf4_${fileKey}.md`
  const outPath = `${PANEL}\\out_wf4_${fileKey}.txt`
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

const L2_DRAFT = [
  `# Draft L2 docs: H5 living-bank verdict + H4 ablation (chair applies; draft to panel only)`,
  ``,
  `Write TWO chair-ready paste blocks to \`${PANEL}\\l2-changelog-decision-draft.md\`. Ground every`,
  `number in the frozen result docs — read first:`,
  `- docs/drift-recovery-post-bank-headtohead-results-2026-06.md (SciFact H5)`,
  `- docs/drift-recovery-post-bank-headtohead-nfcorpus-results-2026-06.md (NFCorpus H5)`,
  `- docs/drift-recovery-h4-compound-cycles-ablation-2026-07.md (H4)`,
  `Block A — a CHANGELOG.md entry (under [Unreleased]) for the H5 head-to-head verdict + H4 ablation:`,
  `  SciFact C5==C5s (0.131135, bit-identical) and C5r one-shot 0.180862 BEATS living; NFCorpus C5==C5s`,
  `  (0.046389) and living edges one-shot (0.045817) but ties static; BOTH datasets LIVING BANK WINS =`,
  `  False. H4: compound_cycles True 0.005258 vs False 0.236297 (~45x collapse). State it as an honest`,
  `  negative: the living/annealed lifecycle adds nothing over a frozen static bank; a one-shot router is`,
  `  competitive-or-better; compounding is catastrophic.`,
  `Block B — a decision paragraph for docs/next-session.md (a NEW Carried Debt or Deferred-Scope row is`,
  `  NOT needed unless you judge one is; instead propose the honest disposition of the "living post-bank"`,
  `  line): park the living-bank/annealed-post-bank corrector line as NON-PROMOTED per its own`,
  `  preregistered gate (C5 must beat BOTH C5s and C5r; it beat neither on SciFact and only one on`,
  `  NFCorpus). Note this is rung-13/14 evidence and that the frozen static bank (C5s) is the honest`,
  `  baseline to keep if any post-bank is used at all.`,
  `HARD BANS: no "wins"/"promising"/"validated" for the living bank; no ceiling claims; label single-seed`,
  `H4 as single-seed. Every number must appear in the three source docs.`,
].join('\n')

const L2_REVIEW = [
  `# Adversarially review the L2 CHANGELOG+decision draft`,
  ``,
  `Read \`${PANEL}\\l2-changelog-decision-draft.md\` and try to DISPROVE it. Verify EVERY number against`,
  `docs/drift-recovery-post-bank-headtohead-results-2026-06.md,`,
  `docs/drift-recovery-post-bank-headtohead-nfcorpus-results-2026-06.md, and`,
  `docs/drift-recovery-h4-compound-cycles-ablation-2026-07.md. Any number not in a source doc = FAIL.`,
  `Check the verdict logic: LIVING BANK WINS requires C5 > C5s AND C5 > C5r; confirm the draft reports`,
  `False for both datasets with the right per-dataset detail (SciFact loses both; NFCorpus ties static,`,
  `beats one-shot). Hunt any upgrade of the living bank toward a win, any softening of the C5==C5s tie,`,
  `or an H4 magnitude not labeled single-seed. Return PASS/PASS-WITH-FIXES/FAIL + exact replacement text.`,
].join('\n')

const L3_DRAFT = [
  `# Draft L3 docs-truth sync (chair applies; draft to panel only)`,
  ``,
  `Main advanced 33 commits; ROADMAP_EXECUTION.md Phase II step statuses are stale. Read`,
  `docs/ROADMAP_EXECUTION.md and CHANGELOG.md, and confirm delivery from git log (use git show/log on the`,
  `merged PRs). Write chair-ready edits to \`${PANEL}\\l3-docs-truth-draft.md\` proposing:`,
  `1. For EACH Phase II step 9-17, its true status with the PR that delivered it, verified from git:`,
  `   rung 10 SHIM DoD (#284/#285/#286, DoD-correction #289), rung 11 annealing controller +`,
  `   temperature schedule (#260, #280), rung 12 Evidence DAG schema (#277), H5a lifecycle (#279),`,
  `   H5b post-bank builder/runtime/conditions (#281/#282/#283), H3 teacher-supervised (#287/#288),`,
  `   H4 compound knob (#291), H5 head-to-head driver (#290), and now the H5 VERDICT run (this slice).`,
  `   Steps still genuinely open: 15 (GNN prototype), 16 (quant-aware shim routing —`,
  `   adapter_router.py + QuantizationPromotionGate integration), 17 (disk pool slice). Verify these`,
  `   three are actually not-yet-delivered (grep for integration; do not assume).`,
  `2. A short CHANGELOG note that the lattice rung series (10-14 apparatus) is delivered and the`,
  `   living-bank verdict (H5) closed the post-bank question as a negative.`,
  `Propose EXACT text edits (old -> new) for ROADMAP_EXECUTION.md status column. Do not claim a step`,
  `delivered unless a merged PR shows it. If a step is partial, say partial with what remains.`,
].join('\n')

const L3_REVIEW = [
  `# Adversarially review the L3 docs-truth draft`,
  ``,
  `Read \`${PANEL}\\l3-docs-truth-draft.md\`. For EVERY "delivered by #NNN" claim, verify against git`,
  `(git log --oneline, git show --stat #NNN's merge). Any step marked delivered without a real merged PR`,
  `= FAIL. For steps 15/16/17 marked open, confirm they are actually not implemented (grep adapter_router`,
  `+ QuantizationPromotionGate wiring for 16; GNN modules for 15; block_graph pool-shard read for 17).`,
  `Flag any over-claim of delivery or any real delivery the draft missed. Return PASS/PASS-WITH-FIXES/FAIL`,
  `+ per-step corrections.`,
].join('\n')

const ITEMS = [
  { key: 'L2-changelog', draft: L2_DRAFT, review: L2_REVIEW },
  { key: 'L3-docs-truth', draft: L3_DRAFT, review: L3_REVIEW },
]

phase('Draft')
const results = await pipeline(
  ITEMS,
  item => agent(driveGrok(`${item.key}-draft`, item.draft), {
    label: `grok-draft:${item.key}`, phase: 'Draft', agentType: 'general-purpose',
  }),
  (draftReport, item) => agent(driveGrok(`${item.key}-review`, item.review), {
    label: `grok-review:${item.key}`, phase: 'Review', agentType: 'general-purpose',
  }).then(review => ({ key: item.key, draft: draftReport, review })),
)

return {
  items: results.filter(Boolean).map(r => ({
    key: r.key, draftTail: String(r.draft).slice(-1500), reviewTail: String(r.review).slice(-2500),
  })),
}
