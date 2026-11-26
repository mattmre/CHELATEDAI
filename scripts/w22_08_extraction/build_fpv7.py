"""Build artifacts/pv7_w23b_w22_08/extraction/F-PV7.json (W22-08 F-PV7 family).

Hand-coded finding records for the four PV7 council memos plus the W23A disposition (V2 only).
Every quote is checked against the cited source line before the file is written.

Usage: python scripts/w22_08_extraction/build_fpv7.py <repo_root>
"""
import json
import sys

R = sys.argv[1] if len(sys.argv) > 1 else "."
D = "docs/research/council-logs/"
W23A = "artifacts/pv7_w23a_seal/W23A_DISPOSITION.md"
F = {
    "pv7_grok": D + "pv7-post-live-xval-grok47-xhigh.md",
    "pv7_sonnet": D + "pv7-post-live-xval-sonnet55-high.md",
    "pv7_carbon": D + "pv7-final-judge-carbon4-med.md",
    "pv7_fable": D + "pv7-final-judge-fable52-med.md",
    "w23a": W23A,
}
BURN = ["B50", "B51", "B52", "B53", "B54", "B55"]
rec = [
    # ---------------- grok (S_A) ----------------
    ("pv7_grok", 13, "Three of them (#49, #55, #63) have no `__main__`, so the CLI exit 0 is a no-op.", ["B49", "B55", "B63"], "C02", "CLI verifiers without __main__ exit 0 vacuously"),
    ("pv7_grok", 105, "Counting the 129 ms CLI exit as a cold-start proof is L4.", ["B49"], "C02", "#49 vacuous CLI exit"),
    ("pv7_grok", 69, "It cannot validate this file.", ["B48"], "C01", "#48 verifier runs drycheck, cannot load live receipt"),
    ("pv7_grok", 119, "`61_verifier.py` calls `execute(\"local-drycheck\")` and does not open this report.", ["B61"], "C01", "#61 verifier does not open live report"),
    ("pv7_grok", 134, "A green verifier beside a failed live receipt is the scorecard bug.", ["B62"], "C01", "#62 verifier replays stage-1, ignores failed live receipt"),
    ("pv7_grok", 224, "Do not count #62 live as a pass.", ["B62"], "C05", "#62 failed live run must not be counted as pass"),
    ("pv7_grok", 179, "`53_verifier.py` never loads either file.", ["B53"], "C01", "#53 verifier loads neither shadow file"),
    ("pv7_grok", 202, "The verifier does not open:", ["B47"], "C01", "#47 verifier ignores the two host A/A receipts"),
    ("pv7_grok", 11, "There is no 64/64 Complete Evidence Scorecard in this tree.", ["XC"], "C11", "no evidence_tier / complete scorecard across registry rows"),
    ("pv7_grok", 219, "Add `evidence_tier` and a receipt path to every `docs/research/registry.d/*.json` row.", ["XC"], "C11", "registry rows lack evidence_tier"),
    ("pv7_grok", 17, "does not record the index-build knobs that were raised in the same seal commit", ["B48"], "C06", "#48 knob lift undisclosed"),
    ("pv7_grok", 95, "Checking out the stamped commit cannot reproduce the measurement.", ["B48"], "C07", "#48 spine_commit predates module; no source hash"),
    ("pv7_grok", 85, "and still `telemetry_passed=true`", ["B48"], "C16", "#48 telemetry None counts as pass"),
    ("pv7_grok", 223, "Change `telemetry_passed` so `graph_telemetry_complete is None` is not a pass.", ["B48"], "C16", "#48 telemetry predicate"),
    ("pv7_grok", 225, "Do not count #49 family names as live models.", ["B49"], "C03", "#49 family names read as live models"),
    ("pv7_grok", 115, "The name overclaims fail-closed behavior.", ["B61"], "C10", "#61 gate name vs behaviour"),
    ("pv7_grok", 117, "The passing gate does not require that family.", ["B61"], "C05", "#61 scale gate passes with linear_chain false"),
    ("pv7_grok", 224, "Log the chat-template body, including `enable_thinking`, in that receipt.", ["B62"], "C07", "#62 enable_thinking not recorded"),
    ("pv7_grok", 140, "Current call sites that omit the flag:", BURN, "C11", "burn-ledger partial-inventory implicit smoke path at #50-#55 call sites (loft=50, beam=51, qscci=52, pm6=53, 12domain=54, wild=55)"),
    ("pv7_grok", 226, "Pass `allow_local_drycheck_smoke` explicitly at loft:482", BURN, "C11", "same burn-ledger defect, repair item"),
    ("pv7_grok", 227, "Stop setting `one_shot_consumed=true` on a non-live #50 run", ["B50"], "C16", "#50 one-shot flag consumed by drycheck"),
    ("pv7_grok", 233, "stop reading `token_length_est` as a measured 128k context", ["B51"], "C13", "#51 token length is a label"),
    ("pv7_grok", 178, "Gate `live_shadow_executed` passes on the detail string `manifest-shadow-observed`.", ["B53"], "C03", "#53 live-shadow gate passes on a string"),
    ("pv7_grok", 234, "stop reading `run-live-shadow` as a live gateway", ["B53"], "C03", "#53 run-live-shadow overstates"),
    ("pv7_grok", 186, "says `>=10/12` positive. The receipt requires `11/12`.", ["B54"], "C10", "#54 registry vs receipt threshold"),
    ("pv7_grok", 187, "It does not require `H6B_merge.passed`.", ["B54"], "C05", "#54 gate passes while H6B_merge failed"),
    ("pv7_grok", 188, "`admitted_unburned` against that fixture is not a BioASQ run", ["B54"], "C03", "#54 real dataset names on synthetic fixture"),
    ("pv7_grok", 196, "The defect is the sibling `passed` bit.", ["B55"], "C16", "#55 passed=true on drycheck"),
    ("pv7_grok", 229, "Keep `preregistered_conjunction_met` next to `family_count=4`", ["B55"], "C03", "#55 conjunction shown without 4-family caveat"),
    ("pv7_grok", 207, "Registry status `verified-pass` is a stronger word than the verifier's `local-drycheck-pass`", ["B47"], "C03", "#47 registry status outruns verifier"),
    # ---------------- sonnet (S_B + XC) ----------------
    ("pv7_sonnet", 16, "The write-byte win depends on an asymmetric journal codec.", ["B58"], "C08", "#58 codec asymmetry"),
    ("pv7_sonnet", 46, "The accounting rule depends on which constructor path ran.", ["B58"], "C08", "#58 header accounting asymmetry"),
    ("pv7_sonnet", 64, "Describing it as \"LCB95 > 0\" overstates what was tested.", ["B58"], "C14", "#58 zero-variance bootstrap LCB"),
    ("pv7_sonnet", 73, "for the unqualified \"strictly fewer write bytes on both layouts\" claim", ["B58"], "C03", "#58 unqualified claim"),
    ("pv7_sonnet", 77, "the drycheck margin should not be extrapolated to it", ["B58"], "C03", "#58 live extrapolation caveat"),
    ("pv7_sonnet", 92, "It carries **no `tier` and no `tier_disclosure` field**", ["B56"], "C03", "#56 floor-tier caveat missing"),
    ("pv7_sonnet", 103, "This is status drift (L13).", ["B57"], "C04", "#57 registry status lags live receipt"),
    ("pv7_sonnet", 104, "A reviewer should ask who chose 50 and when relative to the first run.", ["B57"], "C06", "#57 loose 50x gate possibly set after first run (judgement)"),
    ("pv7_sonnet", 110, "L5: the memory claim is untested on the production path.", ["B59"], "C03", "#59 under-1GiB is configured limit, not measurement"),
    ("pv7_sonnet", 113, "`burn_clean: true` in the manifest is therefore not evidence of an unburned corpus.", ["B59"], "C03", "#59 burn_clean label overstates"),
    ("pv7_sonnet", 112, "cannot detect corpus reuse from earlier campaigns", ["B59"], "C11", "#59 partial self-derived burn inventory"),
    ("pv7_sonnet", 121, "Add a negative test that hand-builds a cache entry lacking the digest fields", ["B60"], "C12", "#60 digest-omitted negative test missing"),
    ("pv7_sonnet", 136, "\"Zero-collision\" is trivially true for these loaders", ["XC"], "C10", "loaders ignore 47..64 fragments"),
    ("pv7_sonnet", 137, "holds as a **count of files**, not as a count a single loader reports", ["XC"], "C10", "64/64 is a file count"),
    ("pv7_sonnet", 133, "(`/fabric`, `/api/v2/fabric/*`) is never served.", ["XC"], "C10", "routes.d never served"),
    ("pv7_sonnet", 138, "Schema mismatch is latent: spec ids are strings", ["XC"], "C16", "id type mismatch between spec and fragments"),
    # ---------------- carbon (dependent judge) ----------------
    ("pv7_carbon", 107, "`#48`'s receipt retains ancestor `spine_commit = d4454e2d`", ["B48"], "C07", "residual: #48 spine_commit"),
    ("pv7_carbon", 107, "`graph_telemetry_complete = null` on OpenSearch/pgvector (`telemetry_passed = true` on `is not False`)", ["B48"], "C16", "residual: #48 telemetry None pass"),
    ("pv7_carbon", 107, "`#50`'s synthetic drycheck sets `one_shot_consumed = true`", ["B50"], "C16", "residual: #50 one-shot flag"),
    ("pv7_carbon", 107, "gate checks evaluation completeness while `H6B_merge.passed = false`", ["B54"], "C05", "residual: #54 gate vs failed endpoint"),
    ("pv7_carbon", 107, "`6/12` positive vs `11/12` code / `>=10/12` registry prose", ["B54"], "C10", "residual: #54 threshold drift"),
    ("pv7_carbon", 107, "`#55`'s drycheck carries `passed = true` alongside `status = \"DRYCHECK\"`", ["B55"], "C16", "residual: #55 passed bit"),
    ("pv7_carbon", 107, "retains legacy gate name `live_encoders_fail_closed_on_dry_fallback`", ["B61"], "C10", "residual: #61 gate name"),
    # ---------------- fable (dependent judge) ----------------
    ("pv7_fable", 15, "That change is not disclosed anywhere in the row, the scorecard or the receipt.", ["B48"], "C06", "B1: #48 knob lift undisclosed"),
    ("pv7_fable", 93, "Its `spine_commit` is `d4454e2d`, and that commit does not contain `hnsw_prefilter_pushdown.py` at all.", ["B48"], "C07", "#48 provenance"),
    ("pv7_fable", 16, "The Markdown index never mentions it.", ["B54"], "C05", "B2: #54 failed endpoint hidden"),
    ("pv7_fable", 101, "The gate `h6b_merge_evaluated` passes because it only checks that the statistic was computed.", ["B54"], "C05", "#54 gate vs endpoint"),
    ("pv7_fable", 111, "still lets `None` pass for OpenSearch and pgvector", ["B48"], "C16", "#48 telemetry predicate"),
    ("pv7_fable", 112, "still passes with two dry stand-ins allowed", ["B61"], "C10", "#61 gate naming"),
    ("pv7_fable", 112, "The scale gate still does not require `linear_chain`.", ["B61"], "C05", "#61 scale gate"),
    ("pv7_fable", 113, "Registry `status` is still `verified-pass` beside `evidence_tier=local-drycheck-pass`.", ["B47"], "C03", "#47 status word"),
    ("pv7_fable", 115, "No committed #57 or #62 receipt contains the string \"spark\"", ["B57", "B62"], "C07", "host attribution from commit messages only"),
    ("pv7_fable", 47, "the registry row does not say in plain words that no live measurement exists", ["B58"], "C03", "#58 no-live-measurement caveat"),
]
# V2 only: W23A later audit (not an estimating list)
v2 = [
    ("w23a", 23, "says `>=10/12` domains must be positive.", ["B54"], "C10", "#54 threshold wording nit"),
    ("w23a", 70, "#53's best arm is ZeroMu with raw cosine", ["B53"], "C08", "W22-03 cost-matched rescore: comparator ordering changes"),
    ("w23a", 71, "The conventional arms (62.97 and 62.87 JPM) beat Chelated (59.38).", ["B62"], "C08", "W22-03 cost-matched comparator beats Chelated"),
    ("w23a", 71, "and the sham control is invalid.", ["B62"], "C09", "sham control invalid (overlaps grok's failed sham gate; class differs)"),
    ("w23a", 79, "so a claim that ZeroMu is a uniquely better bridge is not supported against the conventional", ["B54"], "C08", "W22-04 bridge tie vs conventional baseline"),
    ("w23a", 87, "per-domain CIs, which the strict kill rule needs.", ["B54"], "C11", "W22-04 missing cells"),
]

lines = {k: open(R + "/" + v, encoding="utf-8").read().splitlines() for k, v in F.items()}
bad = 0


def mk(rows):
    global bad
    out, cnt = [], {}
    for lid, ln, q, ords, cls, note in rows:
        if not (len(q) >= 20 and q in lines[lid][ln - 1]):
            bad += 1
            print("BAD", lid, ln, q)
        cnt[lid] = cnt.get(lid, 0) + 1
        out.append({"finding_id": "%s-%02d" % (lid, cnt[lid]), "list_id": lid, "source_line": ln, "quote": q,
                    "ordinals": ords, "defect_class": cls, "alignment_note": note})
    return out


findings = mk(rec)
v2f = mk(v2)
SA = ["B47", "B48", "B49", "B50", "B51", "B52", "B53", "B54", "B55", "B61", "B62", "B63", "B64"]
SB = ["B56", "B57", "B58", "B59", "B60"]
meta = {
    "pv7_grok": (SA + ["XC"], 23, "Bundles #56–#60 were not audited.", False, "none found", None),
    "pv7_sonnet": (SB + ["XC"], 1, "Wave 19 (Bundles #56-#60) and Fragment Integration", False, "none found", None),
    "pv7_carbon": (SA + SB + ["XC"], 1, "Additive Bundles `#47–#64`", True,
                   "Both Post-`v7.0` adversarial peer-audit memos", 27),
    "pv7_fable": (SA + SB + ["XC"], 1, "Bundles #47-#64", True,
                  "I am blocking on two items from the grok-4.7 peer memo", 13),
}
lists = []
for lid, (sc, sl, sq, dep, iq, il) in meta.items():
    if sq not in lines[lid][sl - 1]:
        bad += 1
        print("BAD SCOPE", lid)
    if il is not None and iq not in lines[lid][il - 1]:
        bad += 1
        print("BAD INDEP", lid)
    lists.append({"list_id": lid, "source_path": F[lid], "declared_scope": sc, "scope_quote": sq, "scope_line": sl,
                  "reads_other_lists": dep, "independence_quote": iq, "independence_line": il})
out = {
    "family": "F-PV7",
    "extractor": "implementer (main agent), single classifier for all four lists and the W23A V2 records",
    "brief_path": None, "brief_preseeds_defects": False, "brief_preseed_quote": None, "brief_preseed_line": None,
    "strata": {"S_A": SA, "S_B": SB},
    "lists": lists, "findings": findings,
    "v2_external_list": {"list_id": "w23a", "source_path": W23A, "findings": v2f},
    "splits_and_judgement_calls": [
        "Strata are defined by ordinal only. XC findings belong to no stratum; they are reported separately and do not enter any stratum's estimators.",
        "Carbon section 1 rows restate peer defects and mark them CLOSED. Under the preregistered unit those are 'already fixed' restatements and are not counted. Only NH-PV7-1 residual items (logged for v7.1+ cleanup) are counted.",
        "Carbon NH-PV7-1 #56 placement note excluded (the disclosure exists in the registry; not a defect). NH-PV7-2 (wave6 server summary, pre-existing v6.0) and NH-PV7-3 (computational_storage_poc warning scope) are outside #47-#64 and dropped. NH-PV7-4 describes intended behaviour; not a defect.",
        "Sonnet section 6 (computational_storage_poc EXPERIMENTAL flag) and section 8 (rebase/CRLF) name no scorecard bundle and are outside the family scope: dropped.",
        "Grok L140 'brief's line numbers are stale' concerns review-brief call-site line numbers, not bundle evidence: not counted (coding rule 4 covers digest/number misprints only).",
        "Grok L211-215 and L232 (#63/#64 keep protocol-ready, DEV scorers) are status confirmations (coding rule 2): not counted. Grok L211 'do not carry an evidence tier' folds into XC:C11.",
        "Burn-ledger implicit smoke path at #50-#55 call sites (grok) and #59 partial self-derived inventory (sonnet) are both classed C11 so the same defect type aligns across strata.",
        "Sonnet L104 (who chose the 50x gate and when) counted as B57:C06 although it is framed as a question; borderline.",
        "Sonnet L110 'strictly_under_one_gib is configured, not measured' classed C03 (label overstates) under the tie-break order rather than C13.",
        "Fable L46 (gate moved to clustered-only after scattered tie/loss) is explicitly accepted by the judge as disclosed: not counted. Fable L54 is a brief wording correction: not counted. Fable L114 (#50 one-shot not re-verified) is a scope limitation, not a defect claim.",
        "W23A V2 records: the #62 ABSENT nit is explicitly not a defect in W23A and is excluded. W22-03 #53/#62 and W22-04 #54 cost-matched results are classed C08 (comparator). The W23A 'sham control is invalid' is C09 while grok/fable's failed sham gate is C05 in this coding; the coarse class split may overstate V2 misses by one.",
        "Both judges read the peer memos (Carbon reconciles both; Fable cites the grok memo). They fail the procedural-independence filter for every stratum.",
    ],
}
dst = R + "/artifacts/pv7_w23b_w22_08/extraction/F-PV7.json"
json.dump(out, open(dst, "w", encoding="utf-8"), indent=1, ensure_ascii=False)
print("records", len(findings), "v2", len(v2f), "bad", bad)
