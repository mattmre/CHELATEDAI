"""Capture-recapture estimators and kill rules for the W22-08 audit residual pilot.

Preregistration: artifacts/pv7_w23b_w22_08/PREREGISTRATION.md (commit 3911ad1c).

A "list" is one auditor's memo. A capture is a canonical defect ID ``<ordinal>:<class>``.
The module verifies extraction quotes against their source lines, filters lists by scope and
procedural independence, and applies kill rules K1-K4 to one analysis unit (a family, or a
stratum of F-PV7).
"""
from __future__ import annotations

import itertools
import json
import math
import os
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

Z95 = 1.959963984540054
MIN_QUOTE_LEN = 20
K2_OVERLAP_MAX = 0.80
K3_RATIO_MAX = 3.0
K3_THETA = 2.0
THETAS = (1.0, 2.0, 4.0)


# --------------------------------------------------------------------------- extraction
def load_json(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def verify_findings(findings: Sequence[dict], source_lines: Dict[str, List[str]]
                    ) -> Tuple[List[dict], List[dict]]:
    """Split findings into (verified, errors).

    ``source_lines`` maps list_id -> the source file's lines. A finding is verified when its quote
    is at least 20 characters and occurs verbatim on its cited (1-based) line.
    """
    ok, bad = [], []
    for f in findings:
        lines = source_lines.get(f["list_id"])
        ln = int(f["source_line"])
        reason = None
        if lines is None:
            reason = "no source for list"
        elif not 1 <= ln <= len(lines):
            reason = "line out of range"
        elif len(f["quote"]) < MIN_QUOTE_LEN:
            reason = "quote shorter than %d chars" % MIN_QUOTE_LEN
        elif f["quote"] not in lines[ln - 1]:
            reason = "quote not on cited line"
        if reason is None:
            ok.append(f)
        else:
            bad.append({"finding_id": f.get("finding_id"), "list_id": f["list_id"], "reason": reason})
    return ok, bad


def read_source_lines(repo_root: str, lists: Iterable[dict]) -> Dict[str, List[str]]:
    out = {}
    for lst in lists:
        with open(os.path.join(repo_root, lst["source_path"]), encoding="utf-8") as fh:
            out[lst["list_id"]] = fh.read().splitlines()
    return out


def canonical_ids(finding: dict) -> List[str]:
    return ["%s:%s" % (o, finding["defect_class"]) for o in finding["ordinals"]]


def captures_by_list(findings: Sequence[dict], list_ids: Sequence[str], scope: Set[str]
                     ) -> Tuple[Dict[str, Set[str]], List[dict]]:
    """Collapse findings to per-list sets of canonical IDs inside ``scope``.

    Returns (captures, dropped) where ``dropped`` records each (finding, ordinal) outside scope.
    """
    cap = {lid: set() for lid in list_ids}
    dropped = []
    for f in findings:
        if f["list_id"] not in cap:
            continue
        for o in f["ordinals"]:
            if o in scope:
                cap[f["list_id"]].add("%s:%s" % (o, f["defect_class"]))
            else:
                dropped.append({"finding_id": f["finding_id"], "list_id": f["list_id"], "ordinal": o})
    return cap, dropped


# --------------------------------------------------------------------------- pairwise
def lincoln_petersen(n1: int, n2: int, m: int) -> float:
    if m == 0:
        return math.inf
    return n1 * n2 / m


def chapman(n1: int, n2: int, m: int) -> dict:
    n_hat = (n1 + 1) * (n2 + 1) / (m + 1) - 1
    var = (n1 + 1) * (n2 + 1) * (n1 - m) * (n2 - m) / ((m + 1) ** 2 * (m + 2))
    se = math.sqrt(var)
    return {"N": n_hat, "var": var, "ci95": [n_hat - Z95 * se, n_hat + Z95 * se]}


def pairwise(cap: Dict[str, Set[str]]) -> List[dict]:
    out = []
    for a, b in itertools.combinations(sorted(cap), 2):
        n1, n2 = len(cap[a]), len(cap[b])
        m = len(cap[a] & cap[b])
        union = len(cap[a] | cap[b])
        mn = min(n1, n2)
        out.append({
            "pair": [a, b], "n1": n1, "n2": n2, "m": m,
            "lincoln_petersen": lincoln_petersen(n1, n2, m),
            "chapman": chapman(n1, n2, m),
            "overlap": (m / mn) if mn > 0 else None,
            "jaccard": (m / union) if union > 0 else None,
        })
    return out


# --------------------------------------------------------------------------- Chao2
def incidence_frequencies(cap: Dict[str, Set[str]]) -> Dict[str, int]:
    freq = {}
    for ids in cap.values():
        for i in ids:
            freq[i] = freq.get(i, 0) + 1
    return freq


def chao2(cap: Dict[str, Set[str]]) -> dict:
    """Chao2 incidence estimator with Chao's log-normal 95% interval.

    f0 = A f1^2 / (2 f2), A = (k-1)/k; bias-corrected A f1 (f1-1) / 2 when f2 = 0.
    Variance (Chao 1987 / Chao 2005 incidence form, A-scaled):
      f2 > 0:  f2 [A/2 r^2 + A^2 r^3 + A^2/4 r^4],  r = f1/f2
      f2 = 0:  A f1(f1-1)/2 + A^2 f1 (2 f1 - 1)^2 / 4 - A^4 f1^4 / (4 f0)
    Upper 95% bound on the undetected count = f0 * C, C = exp(z sqrt(log(1 + var/f0^2))).
    """
    k = len(cap)
    freq = incidence_frequencies(cap)
    s_obs = len(freq)
    f1 = sum(1 for v in freq.values() if v == 1)
    f2 = sum(1 for v in freq.values() if v == 2)
    a = (k - 1) / k if k > 0 else 0.0
    if f2 > 0:
        f0 = a * f1 * f1 / (2 * f2)
        r = f1 / f2
        var = f2 * (a / 2 * r ** 2 + a * a * r ** 3 + a * a / 4 * r ** 4)
    else:
        f0 = a * f1 * (f1 - 1) / 2
        if f0 > 0:
            var = a * f1 * (f1 - 1) / 2 + a * a * f1 * (2 * f1 - 1) ** 2 / 4 - a ** 4 * f1 ** 4 / (4 * f0)
        else:
            var = 0.0
    var = max(var, 0.0)
    if f0 > 0:
        c = math.exp(Z95 * math.sqrt(math.log(1 + var / (f0 * f0))))
        f0_upper = f0 * c
        lower = s_obs + f0 / c
    else:
        f0_upper = 0.0
        lower = float(s_obs)
    return {"k": k, "S_obs": s_obs, "f1": f1, "f2": f2, "f0": f0, "var_f0": var,
            "N": s_obs + f0, "N_ci95": [lower, s_obs + f0_upper], "f0_upper95": f0_upper,
            "degenerate_f0_zero": f0 == 0}


# --------------------------------------------------------------------------- log-linear
def capture_history_counts(cap: Dict[str, Set[str]]) -> Tuple[List[str], List[Tuple[int, ...]], List[int]]:
    """Return (list order, observed histories excluding all-zero, counts incl. zero cells)."""
    order = sorted(cap)
    k = len(order)
    counts = {}
    for i in set().union(*cap.values()) if cap else set():
        h = tuple(1 if i in cap[lid] else 0 for lid in order)
        counts[h] = counts.get(h, 0) + 1
    hist = [h for h in itertools.product((0, 1), repeat=k) if any(h)]
    return order, hist, [counts.get(h, 0) for h in hist]


def fit_poisson_irls(x: np.ndarray, y: np.ndarray, max_iter: int = 100, tol: float = 1e-10) -> dict:
    """Poisson GLM (log link) by IRLS with R-style deviance convergence."""
    mu = y + 0.5
    eta = np.log(mu)
    dev_old = math.inf
    beta = np.zeros(x.shape[1])
    converged = False
    for it in range(1, max_iter + 1):
        z = eta + (y - mu) / mu
        w = mu
        xtw = x.T * w
        try:
            beta = np.linalg.solve(xtw @ x, xtw @ z)
        except np.linalg.LinAlgError:
            beta, *_ = np.linalg.lstsq(xtw @ x, xtw @ z, rcond=None)
        eta = x @ beta
        if not np.all(np.isfinite(eta)) or np.max(eta) > 700:
            return {"beta": beta.tolist(), "converged": False, "iterations": it, "deviance": math.inf}
        mu = np.exp(eta)
        with np.errstate(divide="ignore", invalid="ignore"):
            term = np.where(y > 0, y * np.log(y / mu), 0.0)
        dev = float(2 * np.sum(term - (y - mu)))
        if abs(dev - dev_old) / (abs(dev) + 0.1) < tol:
            converged = True
            break
        dev_old = dev
    return {"beta": beta.tolist(), "converged": converged, "iterations": it, "deviance": dev}


def loglinear_models(cap: Dict[str, Set[str]]) -> List[dict]:
    """Independence model plus each single two-way-interaction model; predict the all-zero cell."""
    order, hist, counts = capture_history_counts(cap)
    k = len(order)
    y = np.array(counts, dtype=float)
    h = np.array(hist, dtype=float)
    base = np.column_stack([np.ones(len(hist)), h])
    specs = [("independence", None)] + [("%s*%s" % (order[i], order[j]), (i, j))
                                        for i, j in itertools.combinations(range(k), 2)]
    out = []
    for name, pair in specs:
        x = base if pair is None else np.column_stack([base, h[:, pair[0]] * h[:, pair[1]]])
        fit = fit_poisson_irls(x, y)
        f0 = math.exp(fit["beta"][0]) if fit["converged"] and math.isfinite(fit["beta"][0]) else math.inf
        boundary = any(abs(b) > 15 for b in fit["beta"][1:])
        finite = fit["converged"] and math.isfinite(f0)
        out.append({"model": name, "converged": fit["converged"], "iterations": fit["iterations"],
                    "deviance": fit["deviance"], "f0": f0 if finite else None, "finite": finite,
                    "boundary_estimate": boundary, "params": len(x[0]), "cells": len(hist)})
    return out


# --------------------------------------------------------------------------- kill rules
def k1(n_lists: int) -> dict:
    return {"pass": n_lists >= 3, "n_lists": n_lists}


def k2(pairs: Sequence[dict]) -> dict:
    worst = max((p["overlap"] for p in pairs if p["overlap"] is not None), default=None)
    fails = [p["pair"] for p in pairs if p["overlap"] is not None and p["overlap"] > K2_OVERLAP_MAX]
    return {"pass": not fails, "max_overlap": worst, "failing_pairs": fails}


def k3(models: Sequence[dict], s_obs: int) -> dict:
    # A boundary estimate (a parameter diverging, so the MLE does not exist) is treated like a
    # non-finite prediction: the all-zero cell is then an artefact of the divergence.
    vals = [m["f0"] for m in models]
    nonfinite = [m["model"] for m in models if not m["finite"] or m["boundary_estimate"]]
    if nonfinite:
        ratio = None
        a_fail = True
    else:
        ratio = max(vals) / max(min(vals), 1.0)
        a_fail = ratio > K3_RATIO_MAX
    indep = models[0]["f0"]
    if indep is None or models[0]["boundary_estimate"]:
        b_fail = True
        theta = {str(t): (None if indep is None else t * indep) for t in THETAS}
    else:
        b_fail = K3_THETA * indep > s_obs
        theta = {str(t): t * indep for t in THETAS}
    return {"pass": not (a_fail or b_fail), "a_fail": a_fail, "b_fail": b_fail, "f0_ratio": ratio,
            "nonfinite_models": nonfinite, "f0_independence": indep, "S_obs": s_obs, "f0_theta": theta}


def k4(cap: Dict[str, Set[str]]) -> dict:
    rows = []
    for lid in sorted(cap):
        others = {o: cap[o] for o in cap if o != lid}
        seen_else = set().union(*others.values()) if others else set()
        u = len(cap[lid] - seen_else)
        ch = chao2(others)
        rows.append({"held_out": lid, "u_i": u, "chao2_others_f0": ch["f0"],
                     "chao2_others_f0_upper95": ch["f0_upper95"], "fail": u > ch["f0_upper95"]})
    return {"pass": not any(r["fail"] for r in rows), "rows": rows}


def analyse_unit(cap: Dict[str, Set[str]]) -> dict:
    """Run every estimator and K1-K4 on one estimating set."""
    n = len(cap)
    res = {"lists": sorted(cap), "n_captures": {lid: len(v) for lid, v in sorted(cap.items())},
           "S_obs": len(set().union(*cap.values())) if cap else 0, "pairwise": pairwise(cap)}
    res["K1"] = k1(n)
    res["K2"] = k2(res["pairwise"])
    if n >= 3:
        res["chao2"] = chao2(cap)
        res["loglinear"] = loglinear_models(cap)
        res["K3"] = k3(res["loglinear"], res["S_obs"])
        res["K4"] = k4(cap)
    else:
        res["chao2"] = res["loglinear"] = None
        res["K3"] = res["K4"] = {"pass": False, "not_evaluated": "fewer than 3 lists"}
    res["passes"] = all(res[k]["pass"] for k in ("K1", "K2", "K3", "K4"))
    return res


def build_unit(family: dict, findings: Sequence[dict], scope: Set[str],
               scope_by_list: Optional[Dict[str, Set[str]]] = None) -> dict:
    """Apply the scope and independence filters, then analyse.

    A list is in scope when its declared scope covers the unit scope. A list is independent unless
    ``reads_other_lists`` is true or the family brief pre-seeded defects. When K1 fails only because
    of dependence, the as-if-independent analysis on all in-scope lists is attached (non-gating).
    """
    lists = family["lists"]
    preseed = bool(family.get("brief_preseeds_defects", False))
    if scope_by_list is None:
        scope_by_list = {lst["list_id"]: set(lst["declared_scope"]) for lst in lists}
    in_scope = [lst["list_id"] for lst in lists if scope and scope <= scope_by_list[lst["list_id"]]]
    out_scope = [lst["list_id"] for lst in lists if lst["list_id"] not in in_scope]
    dependent = [lst["list_id"] for lst in lists if lst["list_id"] in in_scope
                 and (lst.get("reads_other_lists") or preseed)]
    est = [lid for lid in in_scope if lid not in dependent]
    cap_all, dropped = captures_by_list(findings, [lst["list_id"] for lst in lists], scope)
    unit = {"scope": sorted(scope), "lists_in_scope": in_scope, "lists_out_of_scope": out_scope,
            "lists_dependent": dependent, "estimating_set": est,
            "dropped_out_of_scope": dropped,
            "gating": analyse_unit({lid: cap_all[lid] for lid in est})}
    if not unit["gating"]["K1"]["pass"] and dependent and len(in_scope) >= 3:
        unit["as_if_independent_non_gating"] = analyse_unit({lid: cap_all[lid] for lid in in_scope})
    unit["captures_all_lists"] = {lid: sorted(v) for lid, v in cap_all.items()}
    return unit
