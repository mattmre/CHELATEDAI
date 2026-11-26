"""W22-08 runner: verify extractions, run capture-recapture per family/stratum, write results + report.

Usage: python scripts/run_w22_08_capture_recapture.py [repo_root]
"""
from __future__ import annotations

import json
import math
import os
import sys

ROOT = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

import capture_recapture_audit as cra  # noqa: E402

OUT = os.path.join(ROOT, "artifacts", "pv7_w23b_w22_08")
EXT = os.path.join(OUT, "extraction")

DEPARTURES = [
    "Extraction: F-W14 (as permitted), F-R5 and F-W4 were extracted by separate sub-agents; the preregistration only "
    "named F-W14 as delegable. F-PV7 and F-W8 were extracted by the implementer. The F-W14 sub-agent did not receive "
    "the coding rules (extraction/CODING_RULES.md); the F-R5/F-W4 sub-agents did.",
    "Scope rule: 'declared scope equals the family's shared scope' is implemented as: the unit scope is the "
    "intersection of the candidate lists' declared scopes (or the F-PV7 stratum), a list enters if its declared "
    "scope covers the unit scope, and findings outside the unit scope are dropped and logged.",
    "F-PV7 strata contain bundle ordinals only. XC (cross-cutting) findings are reported descriptively and enter no "
    "stratum estimate.",
    "F-R5 scopes are textual ('Draft v3 whole, 27 bundles'); all four lists are assigned B1-B27 + XC. r5_grok "
    "reviewed the whole draft but re-hashed only 7 manifests; that is treated as a method limit, not a scope change.",
    "F-W14 IDs are component-level ('46/runner:C01'), coarser than bundle IDs; this inflates overlap. Unit scope is "
    "the intersection runner+formal; w14_sonnet's partial runner coverage is accepted as covering it.",
    "Chao2 variance uses the A=(k-1)/k scaled incidence form (Chao 2005); the preregistration names only 'Chao's "
    "(1987) log-normal interval'. For K3 a non-converged log-linear model, or one with a boundary estimate (a "
    "parameter |beta| > 15, i.e. the MLE does not exist because a capture-history margin is empty), is treated as "
    "non-finite. This changes K3 for F-PV7 non-gating units and for F-W14 (boundary on the grok*sol, neon*sol and sol*sonnet interaction models; without this rule F-W14 K3 would pass); F-W14 already fails K2, so no verdict changes.",
    "No R5 or W4 round brief exists on disk, so brief pre-seeding for those families is 'no evidence', not verified.",
]


def fmt(x, nd=2):
    if x is None:
        return "n/a"
    if isinstance(x, float):
        if math.isinf(x):
            return "inf"
        return ("%." + str(nd) + "f") % x
    return str(x)


def load_family(name):
    fam = cra.load_json(os.path.join(EXT, name + ".json"))
    src = cra.read_source_lines(ROOT, fam["lists"])
    ok, bad = cra.verify_findings(fam["findings"], src)
    return fam, ok, bad


def scope_by_list_for(name, fam):
    if name == "F-R5":
        s = {"B%d" % i for i in range(1, 28)} | {"XC"}
        return {lst["list_id"]: set(s) for lst in fam["lists"]}
    if name == "F-W14":
        return {lst["list_id"]: {"46/" + c for c in lst["declared_scope"]} for lst in fam["lists"]}
    return {lst["list_id"]: set(lst["declared_scope"]) for lst in fam["lists"]}


def main():
    results = {"preregistration": "artifacts/pv7_w23b_w22_08/PREREGISTRATION.md", "prereg_commit": "3911ad1c",
               "departures": DEPARTURES, "families": {}}
    tables = {}
    for name in ("F-PV7", "F-W8", "F-R5", "F-W4", "F-W14"):
        fam, ok, bad = load_family(name)
        sbl = scope_by_list_for(name, fam)
        entry = {"extraction_records": len(fam["findings"]), "verified": len(ok), "extraction_errors": bad,
                 "brief_preseeds_defects": fam.get("brief_preseeds_defects"),
                 "judgement_calls": fam.get("splits_and_judgement_calls", []),
                 "list_meta": [{k: lst.get(k) for k in ("list_id", "source_path", "scope_quote", "scope_line",
                                                       "reads_other_lists", "independence_quote",
                                                       "independence_line")} for lst in fam["lists"]],
                 "units": {}}
        if name == "F-PV7":
            units = {s: set(v) for s, v in fam["strata"].items()}
        else:
            units = {"all": set.intersection(*sbl.values())}
        for uname, scope in units.items():
            entry["units"][uname] = cra.build_unit(fam, ok, scope, sbl)
        if name == "F-PV7":
            xc_cap, _ = cra.captures_by_list(ok, [lst["list_id"] for lst in fam["lists"]], {"XC"})
            entry["xc_descriptive"] = {lid: sorted(v) for lid, v in xc_cap.items()}
            v2 = fam["v2_external_list"]
            v2src = cra.read_source_lines(ROOT, [{"list_id": v2["list_id"], "source_path": v2["source_path"]}])
            v2ok, v2bad = cra.verify_findings(v2["findings"], v2src)
            entry["V2"] = {"extraction_errors": v2bad, "strata": {}}
            for uname, scope in units.items():
                union = set().union(*entry["units"][uname]["captures_all_lists"].values())
                w23 = set(i for f in v2ok for i in cra.canonical_ids(f) if i.split(":")[0] in scope)
                u = entry["units"][uname]
                ref = u.get("as_if_independent_non_gating") or u["gating"]
                ch = ref.get("chao2")
                entry["V2"]["strata"][uname] = {
                    "w23a_ids_in_stratum": sorted(w23), "missed_by_fpv7_union": sorted(w23 - union),
                    "n_missed": len(w23 - union),
                    "compare_chao2_f0": ch["f0"] if ch else None,
                    "compare_chao2_f0_upper95": ch["f0_upper95"] if ch else None,
                    "compare_basis": "as-if-independent (non-gating)" if u.get("as_if_independent_non_gating")
                    else "gating estimating set"}
        # aligned table
        rows = {}
        for f in ok:
            for cid in cra.canonical_ids(f):
                rows.setdefault(cid, {}).setdefault(f["list_id"], []).append(f["finding_id"])
        tables[name] = (fam, rows, entry)
        results["families"][name] = entry
    passing = [n for n, e in results["families"].items() if any(u["gating"]["passes"] for u in e["units"].values())]
    results["verdict"] = "ADVANCE" if passing else "KILLED"
    results["passing_families"] = passing
    with open(os.path.join(OUT, "w22_08_results.json"), "w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=1, ensure_ascii=False, default=lambda o: None)
    write_report(results, tables)
    print("verdict", results["verdict"], "passing", passing)
    for n, e in results["families"].items():
        print(n, "errors", len(e["extraction_errors"]),
              {u: (v["gating"]["K1"]["pass"], v["gating"]["K2"]["pass"], v["gating"]["K3"]["pass"],
                   v["gating"]["K4"]["pass"]) for u, v in e["units"].items()})


def kline(res):
    parts = []
    for k in ("K1", "K2", "K3", "K4"):
        r = res[k]
        if "not_evaluated" in r:
            parts.append("%s n/e" % k)
        else:
            parts.append("%s %s" % (k, "PASS" if r["pass"] else "FAIL"))
    return ", ".join(parts)


def unit_md(res, label):
    L = ["**%s** lists=%s, captures=%s, S_obs=%d. %s. Unit %s." % (
        label, ", ".join(res["lists"]) or "none", res["n_captures"], res["S_obs"], kline(res),
        "PASSES" if res["passes"] else "fails")]
    if res["pairwise"]:
        L.append("")
        L.append("| pair | n1 | n2 | m | LP | Chapman (95% CI) | overlap | Jaccard |")
        L.append("|---|---|---|---|---|---|---|---|")
        for p in res["pairwise"]:
            c = p["chapman"]
            L.append("| %s | %d | %d | %d | %s | %s (%s, %s) | %s | %s |" % (
                " / ".join(p["pair"]), p["n1"], p["n2"], p["m"], fmt(p["lincoln_petersen"], 1), fmt(c["N"], 1),
                fmt(c["ci95"][0], 1), fmt(c["ci95"][1], 1), fmt(p["overlap"]), fmt(p["jaccard"])))
    if res.get("chao2"):
        c = res["chao2"]
        L.append("")
        L.append("Chao2: k=%d, f1=%d, f2=%d, f0=%s, N=%s, 95%% CI (%s, %s), f0 upper95=%s." % (
            c["k"], c["f1"], c["f2"], fmt(c["f0"]), fmt(c["N"]), fmt(c["N_ci95"][0]), fmt(c["N_ci95"][1]),
            fmt(c["f0_upper95"])))
        L.append("")
        L.append("| log-linear model | converged | boundary | f0 (all-zero cell) |")
        L.append("|---|---|---|---|")
        for m in res["loglinear"]:
            L.append("| %s | %s | %s | %s |" % (m["model"], m["converged"], m["boundary_estimate"], fmt(m["f0"])))
        k3 = res["K3"]
        L.append("")
        L.append("K3: f0 ratio max/max(min,1)=%s (a %s); theta f0 = %s vs S_obs=%d (b %s)." % (
            fmt(k3["f0_ratio"]), "FAIL" if k3["a_fail"] else "pass",
            {t: fmt(v) for t, v in k3["f0_theta"].items()}, k3["S_obs"], "FAIL" if k3["b_fail"] else "pass"))
        L.append("")
        L.append("K4: " + "; ".join("%s u=%d vs upper95(others)=%s%s" % (
            r["held_out"], r["u_i"], fmt(r["chao2_others_f0_upper95"]), " FAIL" if r["fail"] else "")
            for r in res["K4"]["rows"]))
    return L


def write_report(results, tables):
    R = ["# W22-08 Capture-Recapture Audit Residual Pilot: report", "",
         "Preregistration: `artifacts/pv7_w23b_w22_08/PREREGISTRATION.md` (commit `3911ad1c`). Generated by "
         "`scripts/run_w22_08_capture_recapture.py`; numbers in `w22_08_results.json`.", "",
         "## Verdict", "",
         "**%s**. Families passing K1-K4: %s." % (results["verdict"], results["passing_families"] or "none"), ""]
    R.append("| family | unit | estimating set | K1 | K2 | K3 | K4 | passes |")
    R.append("|---|---|---|---|---|---|---|---|")
    for n, e in results["families"].items():
        for u, v in e["units"].items():
            g = v["gating"]
            R.append("| %s | %s | %s | %s |" % (n, u, ", ".join(v["estimating_set"]) or "none",
                                                 " | ".join("n/e" if "not_evaluated" in g[k] else
                                                            ("PASS" if g[k]["pass"] else "FAIL")
                                                            for k in ("K1", "K2", "K3", "K4")))
                     + (" %s |" % ("yes" if g["passes"] else "no")))
    R += ["", "Max pairwise overlap per gating unit (K2 threshold 0.80): " + "; ".join(
        "%s/%s %s" % (n, u, fmt(v["gating"]["K2"]["max_overlap"])) for n, e in results["families"].items()
        for u, v in e["units"].items()) + ". The overlaps depend on how coarse the canonical IDs are; a different "
        "coding could change them, but the coding was frozen before counting and is not re-tuned here."]
    R += ["", "## Departures from the preregistration", ""] + ["- " + d for d in results["departures"]]
    for n, (fam, rows, e) in tables.items():
        R += ["", "## %s" % n, "",
              "Extraction: %d records, %d verified, %d extraction errors. Extractor: %s." % (
                  e["extraction_records"], e["verified"], len(e["extraction_errors"]), fam.get("extractor")), ""]
        R.append("| list | scope quote (line) | reads other lists | independence quote (line) |")
        R.append("|---|---|---|---|")
        for m in e["list_meta"]:
            R.append("| %s | %s (%s) | %s | %s (%s) |" % (
                m["list_id"], (m["scope_quote"] or "").replace("|", "\\|"), m["scope_line"], m["reads_other_lists"],
                (m["independence_quote"] or "").replace("|", "\\|"), m["independence_line"]))
        for u, v in e["units"].items():
            R += ["", "### %s / %s" % (n, u), "",
                  "Scope: %s. In scope: %s. Out of scope: %s. Dependent (removed): %s. Out-of-scope drops: %d." % (
                      ", ".join(v["scope"]) or "EMPTY (no common scope)", v["lists_in_scope"] or "none",
                      v["lists_out_of_scope"] or "none", v["lists_dependent"] or "none",
                      len(v["dropped_out_of_scope"])), ""]
            R += unit_md(v["gating"], "Gating")
            if v.get("as_if_independent_non_gating"):
                R += ["", "As-if-independent (NON-GATING; includes dependent lists):", ""]
                R += unit_md(v["as_if_independent_non_gating"], "Non-gating")
        if n == "F-PV7":
            R += ["", "### F-PV7 XC (descriptive only)", ""]
            for lid, ids in e["xc_descriptive"].items():
                R.append("- %s: %s" % (lid, ", ".join(ids) or "none"))
            R += ["", "### V2: W23A later-miss check (non-gating)", ""]
            for u, s in e["V2"]["strata"].items():
                R.append("- %s: W23A IDs in stratum %s; missed by F-PV7 union: %s (n=%d). Chao2 f0=%s, "
                         "upper95=%s (%s)." % (u, s["w23a_ids_in_stratum"], s["missed_by_fpv7_union"], s["n_missed"],
                                                 fmt(s["compare_chao2_f0"]), fmt(s["compare_chao2_f0_upper95"]),
                                                 s["compare_basis"]))
        lids = [m["list_id"] for m in e["list_meta"]]
        R += ["", "### %s full aligned defect table" % n, "",
              "| canonical ID | " + " | ".join(lids) + " | n lists |", "|---" * (len(lids) + 2) + "|"]
        for cid in sorted(rows):
            R.append("| %s | %s | %d |" % (cid, " | ".join(", ".join(rows[cid].get(lid, [])) or "-" for lid in lids),
                                           len(rows[cid])))
        R += ["", "Judgement calls (from extraction):", ""] + ["- " + j for j in e["judgement_calls"]]
    with open(os.path.join(OUT, "W22_08_REPORT.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(R) + "\n")


if __name__ == "__main__":
    main()
