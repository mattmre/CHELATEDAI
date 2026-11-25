"""Wave-6 shared statistical inference, Holm-Bonferroni multiplicity, and cross-family blind semantic judging.

Implements Wave-5 Cross-Judgment guardrails K2, K3, and K5 without mutating historical R1-R6 helpers:
  - `paired_bootstrap_wave6`: emits distinct `lcb_95` (`p5.0` one-sided 95%), `lcb_975` (`p2.5` two-sided 95% lower),
    `p0125_bonf2` (`p1.25` two-sided 97.5% Bonferroni Look #2 lower), paired cluster bootstrap (`cluster_by_id`),
    exact Clopper-Pearson, Tango 95% (`z=1.959964`), and Tango Look #2 97.5% (`z=2.241403`).
    Fails closed if `p0125_bonf2 < lcb_975 < lcb_95` is violated when bootstrap variance is positive.
  - `holm_bonferroni_stepdown`: family-wise error control over a family of hypotheses using paired bootstrap
    two-sided p-values and adjusted lower bounds.
  - `BlindCrossFamilySemanticJudge`: deterministic, SHA-256-logged blind semantic QA & abstention judge
    supporting a cross-family LLM endpoint or local HuggingFace model (off the generator family),
    separating semantic correctness from strict schema compliance and eliminating substring `ABSTAIN_MARKERS`.
"""

from __future__ import annotations

import hashlib
import re
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from scripts.night1_common import exact_paired_binary_ci

Z_TWO_SIDED_95 = 1.959963984540054
Z_TWO_SIDED_975_BONF2 = 2.241402727604947

BLIND_SEMANTIC_JUDGE_SYSTEM = (
    "You are an impartial, blinded evaluation judge grading a candidate response against a reference answer.\n"
    "Evaluate strictly according to the item mode:\n"
    "1) FACTUAL_QA mode:\n"
    "   - Reply VERDICT: YES if the candidate response states the core factual value, entity, count, or date in the "
    "reference answer without contradicting it or asserting an outdated superseded value.\n"
    "   - Equivalent numeric/date/unit formatting (e.g., '5' vs 'five', 'May 2026' vs '2026-05') is YES.\n"
    "   - Reply VERDICT: NO if the candidate abstains, outputs a record count instead of answering, states a stale/wrong value, "
    "or misses a required part of the reference.\n"
    "2) ABSTENTION_REQUIRED mode (unanswerable, false premise, or unresolved symmetric peer conflict):\n"
    "   - Reply VERDICT: YES if and ONLY if the candidate clearly declines to answer, states the requested entity/detail/event "
    "is not mentioned / insufficient / unknown / contested, or outputs 'ABSTAIN', WITHOUT fabricating an answer to the "
    "unmentioned premise.\n"
    "   - Correctly noting what WAS mentioned in the records while refusing the unmentioned premise (e.g., 'Records mention "
    "Harajuku, not Shinjuku, so this cannot be answered') is VERDICT: YES.\n"
    "   - Hallucinating a value for the false premise, guessing a number, outputting bare 'ABSTRACT', or outputting "
    "'I found N matching records' without refusing the question is VERDICT: NO.\n"
    "Output EXACTLY one line: VERDICT: YES or VERDICT: NO"
)

_VERDICT_YES_RE = re.compile(r"\bVERDICT\s*:\s*YES\b", re.IGNORECASE)
_VERDICT_NO_RE = re.compile(r"\bVERDICT\s*:\s*NO\b", re.IGNORECASE)
_BARE_YES_RE = re.compile(r"^\s*YES\b", re.IGNORECASE)
_BARE_NO_RE = re.compile(r"^\s*NO\b", re.IGNORECASE)
_RECORD_COUNT_LEAK_RE = re.compile(r"\bi\s+found\s+\d+\s+matching\s+records?\b", re.IGNORECASE)


def audit_schema_anomalies(text: str) -> Dict[str, bool]:
    """Audit known prompt/template anomalies separately from semantic correctness (Guardrail K5)."""
    stripped = (text or "").strip()
    return {
        "empty_output": len(stripped) == 0,
        "bare_abstract_token": stripped.upper() == "ABSTRACT",
        "found_n_matching_records_template": bool(_RECORD_COUNT_LEAK_RE.search(stripped)),
        "strict_abstain_token": stripped.upper() == "ABSTAIN",
    }


def parse_strict_judge_verdict(raw_judge_text: str) -> Tuple[bool, bool]:
    """Return `(is_yes, clean_parse)`. Rejects dual YES+NO outputs fail-closed."""
    txt = (raw_judge_text or "").strip()
    has_v_yes = bool(_VERDICT_YES_RE.search(txt))
    has_v_no = bool(_VERDICT_NO_RE.search(txt))
    if has_v_yes and has_v_no:
        return False, False
    if has_v_yes:
        return True, True
    if has_v_no:
        return False, True
    has_yes = bool(_BARE_YES_RE.search(txt))
    has_no = bool(_BARE_NO_RE.search(txt))
    if has_yes and not has_no and "NO" not in txt.upper().split()[:3]:
        return True, True
    if has_no and not has_yes and "YES" not in txt.upper().split()[:3]:
        return False, True
    return False, False


def paired_bootstrap_wave6(
    scores_a: Mapping[str, float],
    scores_b: Mapping[str, float],
    *,
    n_bootstrap: int = 20_000,
    seed: int = 42,
    expected_ids: Optional[Sequence[str]] = None,
    cluster_by_id: Optional[Mapping[str, str]] = None,
    strict_ids: bool = True,
) -> Dict[str, float]:
    """Compute paired item-level (and optional cluster-level) bootstrap quantiles with distinct p5.0, p2.5, p1.25."""
    keys_a = set(scores_a.keys())
    keys_b = set(scores_b.keys())
    if strict_ids and keys_a != keys_b:
        raise ValueError(
            f"Paired bootstrap ID mismatch: only_in_a={len(keys_a - keys_b)}, only_in_b={len(keys_b - keys_a)}"
        )
    if expected_ids is not None:
        if len(set(expected_ids)) != len(expected_ids):
            raise ValueError("expected_ids contains duplicate IDs")
        exp_set = set(expected_ids)
        if strict_ids and (keys_a != exp_set or keys_b != exp_set):
            raise ValueError(
                f"Paired bootstrap IDs do not match expected_ids: expected={len(exp_set)}, "
                f"a={len(keys_a)}, b={len(keys_b)}"
            )
        qids = sorted(exp_set & keys_a & keys_b)
    else:
        qids = sorted(keys_a & keys_b)
    if not qids:
        raise ValueError("No overlapping query IDs for paired bootstrap")

    vals_a = np.asarray([float(scores_a[q]) for q in qids], dtype=np.float64)
    vals_b = np.asarray([float(scores_b[q]) for q in qids], dtype=np.float64)
    if not (np.all(np.isfinite(vals_a)) and np.all(np.isfinite(vals_b))):
        raise ValueError("Non-finite score encountered in paired_bootstrap_wave6")

    diffs = vals_a - vals_b
    rng = np.random.default_rng(seed)
    n_q = len(qids)
    boot = np.empty(n_bootstrap, dtype=np.float64)
    chunk = 2000
    for start in range(0, n_bootstrap, chunk):
        end = min(n_bootstrap, start + chunk)
        idx = rng.integers(0, n_q, size=(end - start, n_q))
        boot[start:end] = diffs[idx].mean(axis=1)

    p0125 = float(np.quantile(boot, 0.0125))
    p025 = float(np.quantile(boot, 0.025))
    p050 = float(np.quantile(boot, 0.05))
    p500 = float(np.quantile(boot, 0.50))
    p950 = float(np.quantile(boot, 0.95))
    p975 = float(np.quantile(boot, 0.975))
    p9875 = float(np.quantile(boot, 0.9875))

    boot_std = float(np.std(boot))
    if boot_std > 1e-9 and n_q >= 40 and len(np.unique(diffs)) >= 4:
        if not (p0125 < p025 < p050):
            raise RuntimeError(
                f"Quantile separation check failed on non-degenerate continuous/large sample: "
                f"p1.25={p0125}, p2.5={p025}, p5.0={p050}"
            )
    elif boot_std > 1e-9:
        if not (p0125 <= p025 <= p050):
            raise RuntimeError(f"Monotonicity violated: p1.25={p0125}, p2.5={p025}, p5.0={p050}")

    # Two-sided bootstrap p-value against H0: mean_delta == 0
    le_zero = float(np.mean(boot <= 0.0))
    ge_zero = float(np.mean(boot >= 0.0))
    p_two_sided = min(1.0, 2.0 * min(le_zero, ge_zero) + (1.0 / float(n_bootstrap)))

    out: Dict[str, float] = {
        "n_queries": int(n_q),
        "mean_delta": float(diffs.mean()),
        "median_delta": p500,
        "boot_std": boot_std,
        "lcb_95": p050,           # 5.0th percentile (one-sided 95% LCB)
        "ucb_95_one_sided": p950, # 95.0th percentile (one-sided 95% UCB)
        "lcb_975": p025,          # 2.5th percentile (two-sided 95% CI lower / one-sided 97.5% LCB)
        "ucb_975": p975,          # 97.5th percentile (two-sided 95% CI upper)
        "p0125_bonf2": p0125,     # 1.25th percentile (two-sided 97.5% Bonferroni Look #2 lower)
        "p9875_bonf2": p9875,     # 98.75th percentile (two-sided 97.5% Bonferroni Look #2 upper)
        "p_value_two_sided": p_two_sided,
    }

    if cluster_by_id is not None:
        clusters: Dict[str, List[int]] = {}
        for pos, qid in enumerate(qids):
            if qid not in cluster_by_id:
                raise KeyError(f"Missing cluster_by_id mapping for {qid!r}")
            clusters.setdefault(str(cluster_by_id[qid]), []).append(pos)
        c_keys = sorted(clusters.keys())
        n_c = len(c_keys)
        if n_c < 2:
            raise ValueError(f"Cluster bootstrap requires >= 2 clusters, got {n_c}")
        c_sums = np.asarray([float(diffs[clusters[ck]].sum()) for ck in c_keys], dtype=np.float64)
        c_counts = np.asarray([float(len(clusters[ck])) for ck in c_keys], dtype=np.float64)
        rng_c = np.random.default_rng(seed + 10007)
        c_boot = np.empty(n_bootstrap, dtype=np.float64)
        for start in range(0, n_bootstrap, chunk):
            end = min(n_bootstrap, start + chunk)
            c_idx = rng_c.integers(0, n_c, size=(end - start, n_c))
            num = c_sums[c_idx].sum(axis=1)
            den = np.maximum(c_counts[c_idx].sum(axis=1), 1.0)
            c_boot[start:end] = num / den
        out["n_clusters"] = int(n_c)
        out["cluster_lcb_95_p050"] = float(np.quantile(c_boot, 0.05))
        out["cluster_lcb_975_p025"] = float(np.quantile(c_boot, 0.025))
        out["cluster_ucb_975_p975"] = float(np.quantile(c_boot, 0.975))
        out["cluster_p0125_bonf2"] = float(np.quantile(c_boot, 0.0125))
        out["cluster_p9875_bonf2"] = float(np.quantile(c_boot, 0.9875))

    if np.all(np.isin(vals_a, (0.0, 1.0))) and np.all(np.isin(vals_b, (0.0, 1.0))):
        a_int = vals_a.astype(int).tolist()
        b_int = vals_b.astype(int).tolist()
        exact95 = exact_paired_binary_ci(a_int, b_int, z=Z_TWO_SIDED_95)
        exact_bonf2 = exact_paired_binary_ci(a_int, b_int, z=Z_TWO_SIDED_975_BONF2)
        out["n10"] = float(exact95["n10"])
        out["n01"] = float(exact95["n01"])
        out["clopper_pearson_lcb95"] = exact95["clopper_pearson_low"]
        out["clopper_pearson_ucb95"] = exact95["clopper_pearson_high"]
        out["tango_score_lcb95"] = exact95["tango_score_low"]
        out["tango_score_ucb95"] = exact95["tango_score_high"]
        out["tango_score_lcb975_bonf2"] = exact_bonf2["tango_score_low"]
        out["tango_score_ucb975_bonf2"] = exact_bonf2["tango_score_high"]
    return out


def holm_bonferroni_stepdown(
    named_p_values: Mapping[str, float],
    *,
    alpha: float = 0.05,
) -> Dict[str, Dict[str, Any]]:
    """Apply Holm-Bonferroni step-down multiplicity correction across a family of tests."""
    items = sorted(named_p_values.items(), key=lambda kv: (float(kv[1]), kv[0]))
    m = len(items)
    results: Dict[str, Dict[str, Any]] = {}
    running_max = 0.0
    still_rejecting = True
    for rank_idx, (name, raw_p) in enumerate(items):
        k_rem = m - rank_idx
        adj_p = min(1.0, max(running_max, float(raw_p) * k_rem))
        running_max = adj_p
        threshold = float(alpha) / float(k_rem)
        if float(raw_p) > threshold:
            still_rejecting = False
        results[name] = {
            "rank": rank_idx + 1,
            "family_size": m,
            "raw_p": float(raw_p),
            "holm_adjusted_p": adj_p,
            "step_alpha_threshold": threshold,
            "reject_null_at_alpha": bool(still_rejecting),
        }
    return results


class BlindCrossFamilySemanticJudge:
    """Blinded semantic QA & abstention judge using a model family distinct from the answer generator."""

    def __init__(
        self,
        *,
        judge_family: str,
        generator_family: str,
        completion_fn: Callable[[str, str], str],
        salt: str = "wave6-blind-judge-v1",
    ) -> None:
        if judge_family.strip().lower() == generator_family.strip().lower():
            raise ValueError(
                f"Cross-family judge violation: judge_family={judge_family!r} matches generator_family={generator_family!r}"
            )
        self.judge_family = judge_family
        self.generator_family = generator_family
        self._completion_fn = completion_fn
        self._salt = salt
        self.audit_log: List[Dict[str, Any]] = []

    def grade_arms_blind(
        self,
        *,
        item_id: str,
        question: str,
        gold_reference: str,
        arm_predictions: Mapping[str, str],
        abstention_required: bool = False,
    ) -> Dict[str, Dict[str, Any]]:
        """Shuffle arm order deterministically per item so the judge sees only anonymous candidate texts."""
        mode = "ABSTENTION_REQUIRED" if abstention_required else "FACTUAL_QA"
        arm_names = sorted(arm_predictions.keys())
        order = sorted(
            arm_names,
            key=lambda name: hashlib.sha256(f"{self._salt}|{item_id}|{name}".encode("utf-8")).hexdigest(),
        )
        results: Dict[str, Dict[str, Any]] = {}
        for blind_slot, arm_name in enumerate(order):
            pred = str(arm_predictions[arm_name] or "").strip()
            anomalies = audit_schema_anomalies(pred)
            # Fail closed immediately on known broken template leaks or empty strings
            if anomalies["empty_output"] or anomalies["bare_abstract_token"] or anomalies["found_n_matching_records_template"]:
                is_correct = False
                clean_parse = True
                raw_verdict = "AUTO_REJECT_SCHEMA_ANOMALY"
            else:
                user_prompt = (
                    f"Evaluation Mode: {mode}\n"
                    f"Question: {question}\n"
                    f"Reference Ground Truth: {gold_reference}\n"
                    f"Anonymous Candidate Response: {pred}\n\n"
                    "Output EXACTLY: VERDICT: YES or VERDICT: NO"
                )
                raw_verdict = self._completion_fn(BLIND_SEMANTIC_JUDGE_SYSTEM, user_prompt)
                is_correct, clean_parse = parse_strict_judge_verdict(raw_verdict)
            blind_hash = hashlib.sha256(f"{self._salt}|{item_id}|slot{blind_slot}|{pred}".encode("utf-8")).hexdigest()[:16]
            entry = {
                "correct": bool(is_correct),
                "clean_judge_parse": bool(clean_parse),
                "mode": mode,
                "blind_slot": blind_slot,
                "blind_ticket": blind_hash,
                "judge_raw": raw_verdict.strip()[:120],
                "schema_anomalies": anomalies,
            }
            results[arm_name] = entry
            self.audit_log.append({"item_id": item_id, "arm": arm_name, **entry})
        return results
