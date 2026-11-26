import json
import sys

R = sys.argv[1]
D = "docs/research/council-logs/"
F = {"w8_grok": D + "wave8-r1-grok47-xhigh.md", "w8_neon": D + "wave8-r1-neon-high.md",
     "w8_opus": D + "wave8-r1-opus55-high.md", "w8_sol": D + "wave8-r1-sol62-xhigh.md"}
rec = [
    # ---- B40 (D5 gateway) ----
    ("w8_grok", 9, "The deck headline still reads as a live OpenSearch 3.9 placement", ["B40"], "C03", "deck overstates backend as live OpenSearch"),
    ("w8_grok", 33, "Copy the Q1 backend, service-versus-e2e latency, `k_os>=32` recall, `90%` fallback", ["B40"], "C03", "headline latency/recall/fallback caveats missing from deliverables"),
    ("w8_neon", 10, "omit `live_opensearch_java_cluster_claimed: false`", ["B40"], "C03", "deck/workbench omit OpenSearch-over-Qdrant disclosure"),
    ("w8_neon", 10, "cites `service_rtt` (`7.90/21.19 ms` p50/p95 vs `43.55/53.20 ms` direct `4096-d`) without noting", ["B40"], "C03", "service RTT headline without e2e p95/p99 caveat"),
    ("w8_neon", 38, "exact Track E SHA (`cba30b...457bc743`) in `verified_registry.json`", ["B40"], "C07", "Track E digest must be bound exactly (judgement: aligned with brief-digest misprint)"),
    ("w8_opus", 12, "is a load artifact. The baseline was saturated", ["B40"], "C08", "saturated direct-Qdrant baseline makes latency comparison unfair"),
    ("w8_opus", 12, "The gateway's own open-loop e2e p95/p99 (153.75 / 251.86 ms) is omitted.", ["B40"], "C03", "e2e tail omitted from latency headline"),
    ("w8_opus", 13, "admitted Recall@10 = **0.994186 < the K1 0.995 gate**", ["B40"], "C05", "50% density recall misses K1 gate"),
    ("w8_opus", 14, "Its recall of 1.0 is 100% exact fallback, with latency unreported.", ["B40"], "C03", "90% recall headline is fallback"),
    ("w8_opus", 15, "*Naive baseline.* It is not budget-matched", ["B40"], "C08", "naive baseline not budget-matched"),
    ("w8_opus", 16, "*Image.* `qdrant:latest` is unpinned.", ["B40"], "C07", "unpinned container image"),
    ("w8_opus", 7, "In the brief, the Track B hash has 65 hex characters and the Track E short hash is garbled.", ["B37", "B40"], "C07", "brief digest misprints for Track B and Track E"),
    ("w8_sol", 3, "The Round 0 brief misprints Track B’s digest", ["B37"], "C07", "brief misprints Track B digest"),
    ("w8_sol", 3, "gives an abbreviated erroneous Track E digest before the correct one", ["B40"], "C07", "brief misprints Track E digest"),
    ("w8_sol", 7, "At 50% retraction, admitted recall is 99.42%, below 99.5%.", ["B40"], "C05", "50% density recall misses K1 gate"),
    ("w8_sol", 7, "The 21.19 ms LAN p95 measures service RTT with certified stopping disabled.", ["B40"], "C03", "LAN p95 headline caveat (service RTT, certification off)"),
    ("w8_sol", 7, "Direct Qdrant achieved 218.79 Hz at the shared 250 Hz target.", ["B40"], "C08", "saturated baseline (judgement: neutral phrasing but in the timing-boundary corrections)"),
    ("w8_sol", 7, "The 99.611% D1 number divides two reader means and cannot isolate decode.", ["B40"], "C14", "ratio of means cannot isolate decode"),
    ("w8_sol", 9, "Native OpenSearch compatibility remains unverified.", ["B40"], "C03", "OpenSearch compatibility claim overstated"),
    ("w8_sol", 9, "The ANN stop bound assumes an exact dot prefix; D5 permits an approximate prefix.", ["B40"], "C16", "certificate stop bound invalid under approximate prefix"),
    # ---- B37 (D2 3-BEIR) ----
    ("w8_grok", 13, "`0.87%` is FiQA `500/57,638`. The same 500 anchors are `9.65%` of SciFact", ["B37"], "C03", "anchor fraction denominator FiQA-only"),
    ("w8_grok", 34, "the NFCorpus Look #2 miss, the SciFact null, the FiQA sensitivity loss, and the three bf16 minima", ["B37"], "C05", "undisclosed nulls/losses and bf16 minima below K3 floor"),
    ("w8_neon", 12, "**Missing caveats:**", ["B37"], "C05", "per-domain nulls, CSLS trails cosine at 100%, SciFact null, FiQA regression undisclosed"),
    ("w8_opus", 21, "**\"0.87%\" is FiQA-only.**", ["B37"], "C03", "anchor fraction denominator FiQA-only"),
    ("w8_opus", 22, "Mode A against QueryAffine is null on SciFact (p 0.89), which the thesis omits.", ["B37"], "C05", "SciFact null omitted"),
    ("w8_opus", 23, "\"Matching Procrustes\" hides the FiQA sensitivity result", ["B37"], "C05", "FiQA sensitivity loss hidden"),
    ("w8_opus", 17, "*SciFact.* ZeroMu (0.5252) < Procrustes (0.5318), and it is n.s. against QueryAffine", ["B37"], "C05", "SciFact Mode A null (listed under D5 heading; content is Track B)"),
    ("w8_sol", 11, "The 500 anchors represent 0.87% only on FiQA.", ["B37"], "C03", "anchor fraction denominator FiQA-only"),
    ("w8_sol", 11, "All three 4096-d document minima (0.999467–0.999764) miss K3’s 0.9999 floor.", ["B37"], "C05", "bf16 minima miss K3 floor"),
    ("w8_sol", 11, "The runner uses 0.9990, and its duplicate-aware", ["B37"], "C10", "runner threshold drifts from K3 spec"),
    ("w8_sol", 11, "can credit cosine agreement despite a different neighbor", ["B37"], "C08", "metric depends on implementation choice"),
    # ---- B36 (PM-1b v4) ----
    ("w8_grok", 35, "Keep `assignment_gates_pass=false` beside virgin `95%`. Print Holm `0.0766`", ["B36"], "C05", "failed Holm / gate / LME loss must be disclosed"),
    ("w8_grok", 35, "Cite `6,088` from the J4 receipt.", ["B36"], "C03", "6,088 denominator is cross-run (aligned with 2.78x ratio defect; grok frames it as provenance)"),
    ("w8_neon", 16, "fix `wave8-r0-brief.md` L20's `+28.21 pp` typo", ["B36"], "C10", "brief number misprint"),
    ("w8_neon", 18, "does not beat `Flat_BGEM3_TM=91.67%`", ["B36"], "C05", "primary virgin contrasts fail Holm; claims must be restricted"),
    ("w8_opus", 28, "**Primary virgin test fails Holm.**", ["B36"], "C05", "primary virgin test fails Holm"),
    ("w8_opus", 29, "**KU gain is the embedder.**", ["B36"], "C03", "KU gain misattributed to EK5"),
    ("w8_opus", 29, "Its p=5e-5 lies outside the Holm family.", ["B36"], "C14", "KU p-value outside multiplicity family"),
    ("w8_opus", 30, "is misattributed.** That Holm p is v4 vs **MiniLM**.", ["B36"], "C03", "+5.13pp Holm p misattributed"),
    ("w8_opus", 31, "**LME-500 carries a do-not-cite label.**", ["B36"], "C03", "do-not-cite pooled DEV number headlined"),
    ("w8_opus", 32, "**2.78× tokens** is a cross-run ratio.", ["B36"], "C03", "cross-run ratio"),
    ("w8_opus", 33, "is contradicted by ties", ["B36"], "C10", "K2 compliance flag true but artifact shows ties"),
    ("w8_sol", 15, "The compliance flag checks only the primary Flat BGE-M3 comparison.", ["B36"], "C10", "K2 compliance flag broader than what it checks"),
    ("w8_sol", 15, "LME-500 averages 2,191 tokens versus 6,088 on historical J4 TEST N=251. Name both denominators.", ["B36"], "C03", "2.78x cross-run ratio denominators"),
    ("w8_sol", 17, "so the nominal token control is loose", ["B36"], "C08", "token-matched control not matched"),
    ("w8_sol", 17, "sits outside the four-test Holm family", ["B36"], "C14", "KU p-value outside multiplicity family"),
    # ---- B39 (D4 GRPO) ----
    ("w8_grok", 36, "State GRPO−SFT `lcb_975<0`, `false_commit_on_U=0.875`, `contested_successor≈0`, and max zero-std fraction `1.0`.", ["B39"], "C05", "adverse endpoints (false commits, zero-std max) undisclosed"),
    ("w8_grok", 36, "Lead D4 with SFT. State GRPO", ["B39"], "C03", "lift credited to GRPO rather than SFT (split from same line)"),
    ("w8_neon", 22, "**Defect:** `investor-thesis` L75 cites `max_new_tokens=256`", ["B39"], "C10", "thesis 256 vs code/manifest 272"),
    ("w8_neon", 38, "`D4` `87.5%` false-commit rate", ["B39"], "C05", "false-commit rate must be disclosed"),
    ("w8_opus", 40, "but the abort flag tests the mean (0.1667). Disclose both.", ["B39"], "C05", "max zero-std 1.0 undisclosed"),
    ("w8_opus", 41, "**Critical — false commits on U.**", ["B39"], "C05", "false_commit_on_U 0.875 undisclosed"),
    ("w8_opus", 43, "Moat claim #3 (\"verifiable 3-valued reward\") is unsupported.", ["B39"], "C03", "moat claim overstated"),
    ("w8_opus", 44, "**Format confound.** Bare 30B `valid_format_rate = 0.0`", ["B39"], "C08", "format-confounded comparator"),
    ("w8_opus", 45, "**D1.** SFT is 100% on every D1 regime, so D1 is an SFT result.", ["B39"], "C03", "D1 credited to GRPO / zero-shot wording"),
    ("w8_sol", 21, "K4 requires 256 completion tokens and the thesis advertises 256.", ["B39"], "C10", "thesis/spec 256 vs run 272"),
    ("w8_sol", 23, "Attribute the observed lift primarily to SFT.", ["B39"], "C03", "lift credited to GRPO rather than SFT"),
    # ---- B38 (PM-3 arena) ----
    ("w8_grok", 27, "is a substring scan of `formatted_context`", ["B38"], "C10", "stale_resurrection metric name vs computation"),
    ("w8_grok", 37, "Rename `stale_resurrection_*` before registry seal.", ["B38"], "C10", "rename mislabelled metric"),
    ("w8_neon", 26, "footnote Suite B ran `ChelatedConversationalV3` + inline reranker", ["B38"], "C10", "Suite B ran V3 not V4 as labelled"),
    ("w8_neon", 33, "Deck Slide 3 omits Suite A's raw-prose extraction cost (`6,883` toks)", ["B38"], "C03", "raw-prose extraction cost omitted"),
    ("w8_neon", 33, "and `[SUPERSEDED]` audit-note disclosure", ["B38"], "C10", "stale-resurrection 0.7833 audit-note disclosure (aligned with metric-label defect)"),
    ("w8_opus", 50, "But the manifest's Clopper-Pearson lcb₉₅ is **≤ 0 against 3 of 4 arms**", ["B38"], "C05", "conservative intervals include zero, undisclosed"),
    ("w8_opus", 51, "The margin is 15 U items in a self-authored corpus.", ["B38"], "C03", "Suite A margin overstated"),
    ("w8_opus", 52, "Chelated `stale_resurrection = 0.7833` contradicts Regime II's \"$0 ops.\"", ["B38"], "C10", "metric vs prose contradiction"),
    ("w8_opus", 56, "The \"70×–207×\" slogan omits Mem0 infer=False at **1.56×**.", ["B38"], "C03", "token-ratio slogan omits weakest comparator"),
    ("w8_opus", 57, "Chelated read p50 is **521 ms vs 5–14 ms**, which is omitted.", ["B38"], "C05", "adverse read latency omitted"),
    ("w8_opus", 59, "assigns our emulations' behaviour to real vendors", ["B38"], "C03", "emulation behaviour presented as vendor behaviour"),
    ("w8_sol", 27, "Three conservative Clopper–Pearson intervals include zero.", ["B38"], "C05", "conservative intervals include zero (judgement: stated in a corrections memo)"),
    ("w8_sol", 27, "excluded from 123.1 ms median ack", ["B38"], "C03", "extraction cost excluded from ack latency headline"),
    ("w8_sol", 27, "Disclose live Mem0’s MiniLM encoder, wrapper repairs and uncounted retry failures.", ["B38"], "C08", "comparator configuration/repairs undisclosed"),
    ("w8_sol", 27, "Visibility clocks cover immediate and post-consolidation states, with no intermediate checkpoints.", ["B38"], "C11", "coverage gap in visibility checkpoints"),
    # ---- XC (Wave-7 deliverables, TCO, cross-bundle) ----
    ("w8_grok", 29, "Write rates `620 / 740 / 580` and Mem0 read `1,900` are not Suite A", ["XC"], "C13", "TCO coefficients presented as measured but unsourced"),
    ("w8_grok", 29, "`server.py` calls those defaults measured.", ["XC"], "C13", "assumed coefficients labelled measured"),
    ("w8_grok", 29, "The workbench OpenSearch arm is only `13,500` tokens; the deck defaults to `2,178.6`.", ["XC"], "C10", "workbench vs deck default drift"),
    ("w8_grok", 29, "The close says `85%+` and `8B-sidecar`", ["XC"], "C03", "margin and sidecar size overstated in close"),
    ("w8_grok", 29, "`index.html` still denies an accuracy edge vs Mem0 `infer=True`.", ["XC"], "C04", "stale workbench disclaimer"),
    ("w8_grok", 15, "`16/18` and `51.2x` CAR are absent from this manifest.", ["XC"], "C07", "out-of-bundle citations without receipt"),
    ("w8_neon", 29, "**Wrong `Mem0` read-token input:**", ["XC"], "C13", "Mem0 1,900 read tokens contradict measured"),
    ("w8_neon", 30, "Mark `620/740/580` write tokens and `15%/10%` incident rates illustrative", ["XC"], "C13", "assumptions must be labelled"),
    ("w8_neon", 30, "show (or label equal-offset) embedding, ANN compute, and base storage", ["XC"], "C11", "TCO omits embed/ANN/storage lines"),
    ("w8_neon", 31, "default OpenSearch to `13,500` read tokens (`$4.21M`) instead of Deck's `2,178.6`", ["XC"], "C10", "workbench vs deck default drift"),
    ("w8_neon", 31, "cites stale `\"46.7% D1 abstention\"` vs `33.33%`", ["XC"], "C04", "stale workbench number"),
    ("w8_neon", 10, "`index.html` Tab 2 hardcodes stale Wave-3 SciFact numbers", ["XC"], "C04", "stale workbench numbers"),
    ("w8_neon", 32, "**Margin contradiction:**", ["XC"], "C10", "85% margin prose contradicts table"),
    ("w8_opus", 6, "is **untracked**, so the SHA does not reproduce the run", ["XC"], "C07", "producing code untracked; git_sha not reproducing"),
    ("w8_opus", 61, "The **coefficients are not measured**:", ["XC"], "C13", "TCO coefficients not measured"),
    ("w8_opus", 64, "`server.py:305` labels them \"Measured\".", ["XC"], "C13", "assumed coefficients labelled measured"),
    ("w8_opus", 65, "`test_workbench_server.py:85-90` asserts the ranking.", ["XC"], "C12", "test asserts the ranking rather than arithmetic"),
    ("w8_opus", 66, "K6's embed/ANN terms are missing.", ["XC"], "C11", "TCO omits embed/ANN"),
    ("w8_opus", 67, "contradicting the thesis's \"multi-hundred-thousand-dollar wall.\"", ["XC"], "C10", "migration cost prose contradicts measured"),
    ("w8_opus", 82, "Give out-of-bundle citations (353,105 WFS checks, 16/18 OOD", ["XC"], "C07", "out-of-bundle citations without receipt"),
    ("w8_sol", 29, "Competitor batching, manual-audit shares and prices are assumptions.", ["XC"], "C13", "TCO coefficients are assumptions"),
    ("w8_sol", 29, "Recurring embedding, ANN, base storage, output tokens and bridge fitting are omitted.", ["XC"], "C11", "TCO omits cost lines"),
    ("w8_sol", 33, "remove production margin and drop-in guarantees", ["XC"], "C03", "margin and drop-in guarantee overstated"),
    ("w8_sol", 34, "The API omits raw prose and defaults to 13,500 OpenSearch read tokens.", ["XC"], "C10", "workbench vs deck default drift"),
    ("w8_sol", 34, "Label the NumPy migration simulator and its legacy-shard-only stop check explicitly.", ["XC"], "C03", "simulator not labelled as simulator"),
]

lines = {k: open(R + "/" + v, encoding="utf-8").read().splitlines() for k, v in F.items()}
findings, bad, cnt = [], 0, {}
for lid, ln, q, ords, cls, note in rec:
    if not (len(q) >= 20 and q in lines[lid][ln - 1]):
        bad += 1
        print("BAD", lid, ln, q)
    cnt[lid] = cnt.get(lid, 0) + 1
    findings.append({"finding_id": "%s-%02d" % (lid, cnt[lid]), "list_id": lid, "source_line": ln, "quote": q,
                     "ordinals": ords, "defect_class": cls, "alignment_note": note})
scope = {
    "w8_grok": (3, "Read-only audit of the five Wave-6 Part VI manifests and the Wave-7 thesis, deck, and workbench"),
    "w8_neon": (3, "**Scope:** Wave-6 `Part VI` (`5/5` bundles), Wave-7 deliverables, `K1–K6`"),
    "w8_opus": (3, "All 5 manifest SHA-256 values match their receipts."),
    "w8_sol": (3, "All five Wave 6 manifests and all eight receipt-listed side files matched their SHA-256 digests."),
}
lists = []
for lid, path in F.items():
    sl, sq = scope[lid]
    if sq not in lines[lid][sl - 1]:
        bad += 1
        print("BAD SCOPE", lid)
    lists.append({"list_id": lid, "source_path": path, "declared_scope": ["B36", "B37", "B38", "B39", "B40", "XC"],
                  "scope_quote": sq, "scope_line": sl, "reads_other_lists": False,
                  "independence_quote": "none found", "independence_line": None})
out = {
    "family": "F-W8",
    "extractor": "implementer (main agent), single classifier for all four lists",
    "brief_path": D + "wave8-r0-brief.md", "brief_preseeds_defects": False,
    "brief_preseed_quote": None, "brief_preseed_line": None,
    "lists": lists, "findings": findings,
    "splits_and_judgement_calls": [
        "Ordinal mapping from wave8-r0-brief.md L12-L48 and scorecard bundle_id: Track A=B36 wave6_pm1b_v4, Track B=B37 wave6_d2, Track C=B38 wave6_pm3_v2, Track D=B39 wave6_d4, Track E=B40 wave6_d5.",
        "The brief is a claim summary with no prior defect list, so brief_preseeds_defects=false. All four memos answer the same brief; none cites another memo.",
        "Wave-7 deliverable defects that misstate one bundle's receipt map to that bundle; TCO/workbench/margin/untracked-code defects map to XC (coding_rules.md rule 3).",
        "Brief digest misprints (opus L6, sol L3) map to B37:C07 and B40:C07; neon L38 'exact Track E SHA in verified_registry.json' aligned to B40:C07 (judgement call: registry binding, not the brief).",
        "grok L35 'Cite 6,088 from the J4 receipt' aligned to B36:C03 with opus L32 and sol L15 (same 2.78x cross-run ratio), though grok frames it as provenance.",
        "neon L33 [SUPERSEDED] disclosure aligned with grok/opus stale_resurrection metric-label defect as B38:C10.",
        "sol L7 'Direct Qdrant achieved 218.79 Hz ...' counted as B40:C08 saturated baseline (judgement: neutral phrasing inside the timing-boundary corrections).",
        "sol L27 Clopper-Pearson counted as B38:C05 (judgement, same reason).",
        "Excluded under coding rule 1 (neutral restatement): sol L11 SciFact null and FiQA -2.08pp; neon L22 bold 'incremental GRPO-SFT is non-significant'; neon L16 '1/N ties' (explicitly excused); grok L19 tie counts; sol L17 Holm .0766; sol L21 and neon L22 zero-std max.",
        "Coarse class IDs collapse distinct defects of one class in one bundle (B40:C03 holds the OpenSearch label, service-RTT headline and 90% fallback). This is the preregistered unit; it raises pairwise overlap.",
    ],
}
json.dump(out, open(R + "/artifacts/pv7_w23b_w22_08/extraction/F-W8.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("records", len(findings), "bad", bad, cnt)
