"""Bundle #64 (Item 5.2, T4 protocol-ready): Trial 2 Records-Owner Governance Kit & Offline Scorer.

Buyer-owned offline evaluation toolkit for Partner Trial 2 (a records owner who resolves
conflicting statements about the same fact across closed matters).  It scores three arms
on the same matters:

    ``ek5_on``            the EK5 stratified evidence engine, explanations enabled
    ``ek5_off``           the same pipeline with the EK5 strata disabled (ablation)
    ``challenger_ledger`` the independent bitemporal SQL ChallengerLedger

Pre-registered, locked before any matter is unblinded
-----------------------------------------------------
* **Natural conflict base-rate sizing.**  The partner declares the base rate ``b`` of
  naturally-occurring conflicts among closed matters.  The sample must be ``N >=
  ceil(30 / b)`` *consecutive* closed matters (no gaps, no cherry-picked matters) and must
  contain at least 30 natural conflicts.  Fewer than 30 natural conflicts yields
  ``INSUFFICIENT_EVIDENCE`` -- never a win or a loss.
* **Owner loss weights** (``OWNER_LOSS_WEIGHTS``) and labor cost rates
  (``LABOR_COST_PER_MINUTE``) are frozen constants; their digest is part of every report.
* **Two passes.**  The *Attribution Pass* scores decision loss only (isolates what the
  engine decided).  The *Operational Pass* adds extraction + review labor (what the
  owner actually pays).
* **Blinded explanation-faithfulness rubric** (``RUBRIC_CRITERIA``).  Reviewers score
  explanations under opaque codes; the arm <-> code key is applied by the scorer only
  after all scores are fixed.
* **Joint retirement rule.**  The EK5 premium is retained only if BOTH ``EK5-on vs
  ChallengerLedger`` AND ``EK5-on vs EK5-off`` clear their pre-registered margins (lower
  confidence bound of mean per-matter loss advantage >= margin in both passes; plus
  non-inferior faithfulness against the ChallengerLedger).  Otherwise the premium is
  retired.

Evidence tier
-------------
This module is the *protocol and scorer*.  Output on the built-in open DEV fixtures is
synthetic, labelled ``data_origin = "open_dev_fixture"``, and says nothing about real
records.  Stage-1 deliverables seal strictly as ``"status": "protocol-ready"`` until a
partner supplies real matter rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

BUNDLE_ID = 64
TRACK_ID = "item_5_2_trial2_records_governance"
CLAIM_TIER = "T4"
STAGE1_STATUS = "protocol-ready"

ARM_EK5_ON = "ek5_on"
ARM_EK5_OFF = "ek5_off"
ARM_CHALLENGER = "challenger_ledger"
MANDATORY_ARMS: Tuple[str, ...] = (ARM_EK5_ON, ARM_EK5_OFF, ARM_CHALLENGER)

OUTCOME_CORRECT = "correct"
OUTCOME_WRONG = "wrong"
OUTCOME_ABSTAIN = "abstain"
OUTCOME_TIMEOUT = "timeout"
OUTCOMES: Tuple[str, ...] = (OUTCOME_CORRECT, OUTCOME_WRONG, OUTCOME_ABSTAIN, OUTCOME_TIMEOUT)

DEV_FIXTURE_SCENARIOS: Tuple[str, ...] = (
    "win",
    "loss",
    "tie",
    "timeout",
    "insufficient_natural_conflicts",
)

#: Minimum natural conflicts and the numerator of the sizing rule N >= ceil(30 / base_rate).
MIN_NATURAL_CONFLICTS = 30

#: Owner loss per matter outcome (loss units; lower is better).  A confidently wrong
#: resolution is the costliest; an abstain/escalation costs reviewer attention; a timeout
#: leaves the matter unresolved and is charged between the two.
OWNER_LOSS_WEIGHTS: Mapping[str, float] = {
    OUTCOME_CORRECT: 0.0,
    OUTCOME_ABSTAIN: 2.0,
    OUTCOME_TIMEOUT: 6.0,
    OUTCOME_WRONG: 10.0,
}

#: Labor cost per minute (loss units) charged in the Operational Pass only.
LABOR_COST_PER_MINUTE: Mapping[str, float] = {"extraction": 0.05, "review": 0.20}

#: Required lower-confidence-bound advantage (loss units per natural-conflict matter).
PREMIUM_MARGIN_ATTRIBUTION = 0.25
PREMIUM_MARGIN_OPERATIONAL = 0.25
#: One-sided 95% normal quantile used for the lower confidence bound.
Z_ONE_SIDED_95 = 1.645
#: Explanation-faithfulness non-inferiority tolerance vs the ChallengerLedger.
FAITHFULNESS_NONINFERIORITY = 0.05
#: An arm whose timeout rate exceeds this cannot support a premium claim.
MAX_TIMEOUT_RATE = 0.10

#: Blinded rubric: each criterion is scored 0/1 by a reviewer who sees only an opaque code.
RUBRIC_CRITERIA: Tuple[str, ...] = (
    "names_deciding_record",
    "cause_matches_ground_truth",
    "no_unsupported_claim",
    "reviewer_can_act_without_rerun",
)

_BLIND_CODE_RE = re.compile(r"^X[0-9]$")


class TrialProtocolError(ValueError):
    """The submitted rows violate the pre-registered protocol (not a scored verdict)."""


def _sha256_json(payload: Any) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def locked_protocol() -> Dict[str, Any]:
    """The frozen pre-registration; its digest is stamped on every report."""
    return {
        "bundle_id": BUNDLE_ID,
        "min_natural_conflicts": MIN_NATURAL_CONFLICTS,
        "owner_loss_weights": dict(OWNER_LOSS_WEIGHTS),
        "labor_cost_per_minute": dict(LABOR_COST_PER_MINUTE),
        "premium_margin_attribution": PREMIUM_MARGIN_ATTRIBUTION,
        "premium_margin_operational": PREMIUM_MARGIN_OPERATIONAL,
        "z_one_sided_95": Z_ONE_SIDED_95,
        "faithfulness_noninferiority": FAITHFULNESS_NONINFERIORITY,
        "max_timeout_rate": MAX_TIMEOUT_RATE,
        "rubric_criteria": list(RUBRIC_CRITERIA),
        "arms": list(MANDATORY_ARMS),
    }


PROTOCOL_DIGEST = _sha256_json(locked_protocol())


# ---------------------------------------------------------------------------
# 1. Natural base-rate sizing
# ---------------------------------------------------------------------------
def required_consecutive_matters(base_rate: float) -> int:
    """``ceil(30 / base_rate)`` computed exactly (no float rounding at the boundary)."""
    if isinstance(base_rate, bool) or not isinstance(base_rate, (int, float, Fraction)):
        raise TrialProtocolError(f"base_rate must be numeric, got {type(base_rate).__name__}")
    frac = base_rate if isinstance(base_rate, Fraction) else Fraction(str(base_rate))
    if not (0 < frac <= 1):
        raise TrialProtocolError(f"base_rate must satisfy 0 < b <= 1, got {base_rate!r}")
    return math.ceil(Fraction(MIN_NATURAL_CONFLICTS) / frac)


# ---------------------------------------------------------------------------
# 2. Row types
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ArmResult:
    outcome: str
    extraction_minutes: float = 0.0
    review_minutes: float = 0.0

    def __post_init__(self) -> None:
        if self.outcome not in OUTCOMES:
            raise TrialProtocolError(f"outcome must be one of {OUTCOMES}, got {self.outcome!r}")
        for name in ("extraction_minutes", "review_minutes"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0 or math.isnan(value):
                raise TrialProtocolError(f"{name} must be a non-negative number, got {value!r}")

    def attribution_loss(self) -> float:
        return float(OWNER_LOSS_WEIGHTS[self.outcome])

    def operational_loss(self) -> float:
        return (
            self.attribution_loss()
            + self.extraction_minutes * LABOR_COST_PER_MINUTE["extraction"]
            + self.review_minutes * LABOR_COST_PER_MINUTE["review"]
        )


@dataclass(frozen=True)
class MatterRow:
    """One closed matter.  ``rubric`` is keyed by *opaque blind code*, never by arm name."""

    matter_seq: int
    matter_id: str
    natural_conflict: bool
    arms: Mapping[str, ArmResult]
    rubric: Mapping[str, Mapping[str, int]]

    def __post_init__(self) -> None:
        if isinstance(self.matter_seq, bool) or not isinstance(self.matter_seq, int):
            raise TrialProtocolError("matter_seq must be an int")
        if not self.matter_id:
            raise TrialProtocolError("matter_id must be non-empty")
        if set(self.arms) != set(MANDATORY_ARMS):
            raise TrialProtocolError(
                f"matter {self.matter_id!r} must carry exactly the arms {MANDATORY_ARMS}, got {sorted(self.arms)}"
            )
        for code, scores in self.rubric.items():
            if not _BLIND_CODE_RE.match(code):
                raise TrialProtocolError(f"rubric key {code!r} is not an opaque blind code (X0-X9)")
            if set(scores) != set(RUBRIC_CRITERIA):
                raise TrialProtocolError(f"rubric for {code!r} must score exactly {RUBRIC_CRITERIA}")
            for criterion, score in scores.items():
                if score not in (0, 1) or isinstance(score, bool):
                    raise TrialProtocolError(f"rubric score {criterion}={score!r} must be 0 or 1")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "MatterRow":
        arms = {name: ArmResult(**spec) for name, spec in raw["arms"].items()}
        return cls(
            matter_seq=int(raw["matter_seq"]),
            matter_id=str(raw["matter_id"]),
            natural_conflict=bool(raw["natural_conflict"]),
            arms=arms,
            rubric={code: dict(scores) for code, scores in raw.get("rubric", {}).items()},
        )


def blind_codes(matter_id: str, salt: str) -> Dict[str, str]:
    """Deterministic arm -> opaque-code assignment for one matter (order hidden by the salt)."""
    ranked = sorted(MANDATORY_ARMS, key=lambda arm: hashlib.sha256(f"{salt}|{matter_id}|{arm}".encode()).hexdigest())
    return {arm: f"X{i + 1}" for i, arm in enumerate(ranked)}


def make_unblinding_key(matter_ids: Sequence[str], salt: str) -> Dict[str, Dict[str, str]]:
    """``{matter_id: {code: arm}}`` -- held by the trial coordinator, applied after scoring."""
    return {mid: {code: arm for arm, code in blind_codes(mid, salt).items()} for mid in matter_ids}


# ---------------------------------------------------------------------------
# 3. Statistics
# ---------------------------------------------------------------------------
def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def lower_confidence_bound(diffs: Sequence[float]) -> float:
    """One-sided 95% normal lower bound of the mean of paired per-matter advantages."""
    n = len(diffs)
    if n < 2:
        raise TrialProtocolError("need at least 2 paired matters for a confidence bound")
    mu = _mean(diffs)
    var = sum((d - mu) ** 2 for d in diffs) / (n - 1)
    return mu - Z_ONE_SIDED_95 * math.sqrt(var / n)


def _row_loss(row: MatterRow, arm: str, pass_name: str) -> float:
    result = row.arms[arm]
    return result.attribution_loss() if pass_name == "attribution" else result.operational_loss()


def _paired(rows: Sequence[MatterRow], better: str, worse: str, pass_name: str) -> Dict[str, Any]:
    diffs = [_row_loss(r, worse, pass_name) - _row_loss(r, better, pass_name) for r in rows]
    margin = PREMIUM_MARGIN_ATTRIBUTION if pass_name == "attribution" else PREMIUM_MARGIN_OPERATIONAL
    lcb = lower_confidence_bound(diffs)
    return {
        "mean_advantage": round(_mean(diffs), 6),
        "lower_confidence_bound": round(lcb, 6),
        "margin": margin,
        "clears_margin": bool(lcb >= margin),
    }


def _faithfulness(
    rows: Sequence[MatterRow], unblinding_key: Mapping[str, Mapping[str, str]]
) -> Dict[str, float]:
    per_arm: Dict[str, List[float]] = {arm: [] for arm in MANDATORY_ARMS}
    for row in rows:
        key = unblinding_key.get(row.matter_id)
        if key is None:
            raise TrialProtocolError(f"unblinding key missing matter {row.matter_id!r}")
        if not row.rubric:
            raise TrialProtocolError(f"natural-conflict matter {row.matter_id!r} has no blinded rubric scores")
        seen = set()
        for code, scores in row.rubric.items():
            arm = key.get(code)
            if arm not in per_arm:
                raise TrialProtocolError(f"blind code {code!r} of {row.matter_id!r} maps to no arm")
            seen.add(arm)
            per_arm[arm].append(sum(scores.values()) / len(RUBRIC_CRITERIA))
        if seen != set(MANDATORY_ARMS):
            raise TrialProtocolError(f"matter {row.matter_id!r} rubric must cover all three arms exactly once")
    return {arm: round(_mean(vals), 6) for arm, vals in per_arm.items()}


# ---------------------------------------------------------------------------
# 4. Scorer
# ---------------------------------------------------------------------------
def score_trial2_rows(
    rows: Sequence[MatterRow],
    *,
    declared_base_rate: float,
    unblinding_key: Mapping[str, Mapping[str, str]],
    data_origin: str,
    fixture_label: str = "",
) -> Dict[str, Any]:
    """Score closed-matter rows under the locked protocol.  Pure and deterministic."""
    if data_origin not in ("open_dev_fixture", "partner"):
        raise TrialProtocolError("data_origin must be 'open_dev_fixture' or 'partner'")
    ordered = sorted(rows, key=lambda r: r.matter_seq)
    if not ordered:
        raise TrialProtocolError("no matters supplied")
    seqs = [r.matter_seq for r in ordered]
    if len(set(seqs)) != len(seqs):
        raise TrialProtocolError("duplicate matter_seq: matters must be unique")
    if seqs[-1] - seqs[0] != len(seqs) - 1:
        raise TrialProtocolError("matter_seq must be consecutive closed matters (gap = selection bias)")
    if len({r.matter_id for r in ordered}) != len(ordered):
        raise TrialProtocolError("duplicate matter_id")

    required_n = required_consecutive_matters(declared_base_rate)
    natural = [r for r in ordered if r.natural_conflict]
    sizing = {
        "declared_base_rate": declared_base_rate,
        "required_consecutive_matters": required_n,
        "consecutive_matters": len(ordered),
        "natural_conflicts": len(natural),
        "observed_base_rate": round(len(natural) / len(ordered), 6),
        "min_natural_conflicts": MIN_NATURAL_CONFLICTS,
        "sizing_met": bool(len(ordered) >= required_n and len(natural) >= MIN_NATURAL_CONFLICTS),
    }
    report: Dict[str, Any] = {
        "bundle_id": BUNDLE_ID,
        "track_id": TRACK_ID,
        "claim_tier": CLAIM_TIER,
        "status": STAGE1_STATUS,
        "data_origin": data_origin,
        "fixture_label": fixture_label,
        "protocol_digest": PROTOCOL_DIGEST,
        "sizing": sizing,
    }

    if not sizing["sizing_met"]:
        report.update(
            {
                "outcome": "insufficient_natural_conflicts",
                "verdict": "INSUFFICIENT_EVIDENCE",
                "ek5_premium": "undetermined",
                "retirement_causes": [],
                "rationale": (
                    f"Need >= {required_n} consecutive closed matters containing >= {MIN_NATURAL_CONFLICTS} "
                    f"natural conflicts; have {len(ordered)} matters / {len(natural)} natural conflicts. "
                    "No win or loss may be claimed."
                ),
            }
        )
        report["report_sha256"] = _sha256_json(report)
        return report

    timeout_rates = {
        arm: round(sum(1 for r in natural if r.arms[arm].outcome == OUTCOME_TIMEOUT) / len(natural), 6)
        for arm in MANDATORY_ARMS
    }
    faith = _faithfulness(natural, unblinding_key)
    passes = {}
    for label, (better, worse) in {
        "ek5_on_vs_challenger": (ARM_EK5_ON, ARM_CHALLENGER),
        "ek5_on_vs_ek5_off": (ARM_EK5_ON, ARM_EK5_OFF),
    }.items():
        passes[label] = {
            "attribution_pass": _paired(natural, better, worse, "attribution"),
            "operational_pass": _paired(natural, better, worse, "operational"),
        }
    faith_ok = faith[ARM_EK5_ON] >= faith[ARM_CHALLENGER] - FAITHFULNESS_NONINFERIORITY

    causes: List[str] = []
    if timeout_rates[ARM_EK5_ON] > MAX_TIMEOUT_RATE:
        causes.append("EK5_ON_TIMEOUT_RATE_EXCEEDS_LIMIT")
    for label, tag in (("ek5_on_vs_challenger", "VS_CHALLENGER"), ("ek5_on_vs_ek5_off", "VS_EK5_OFF")):
        if not passes[label]["attribution_pass"]["clears_margin"]:
            causes.append(f"{tag}_ATTRIBUTION_MARGIN_NOT_CLEARED")
        if not passes[label]["operational_pass"]["clears_margin"]:
            causes.append(f"{tag}_OPERATIONAL_MARGIN_NOT_CLEARED")
    if not faith_ok:
        causes.append("FAITHFULNESS_INFERIOR_TO_CHALLENGER")

    retained = not causes
    adv_c = passes["ek5_on_vs_challenger"]
    if retained:
        outcome = "win"
    elif timeout_rates[ARM_EK5_ON] > MAX_TIMEOUT_RATE:
        outcome = "timeout"
    elif adv_c["attribution_pass"]["mean_advantage"] < 0 or adv_c["operational_pass"]["mean_advantage"] < 0:
        outcome = "loss"
    else:
        outcome = "tie"

    report.update(
        {
            "outcome": outcome,
            "verdict": "EK5_PREMIUM_RETAINED" if retained else "EK5_PREMIUM_RETIRED",
            "ek5_premium": "retained" if retained else "retired",
            "retirement_causes": causes,
            "comparisons": passes,
            "timeout_rate_by_arm": timeout_rates,
            "faithfulness_by_arm": faith,
            "faithfulness_noninferior_to_challenger": faith_ok,
            "rationale": (
                "Both EK5-on vs ChallengerLedger and EK5-on vs EK5-off cleared the pre-registered margins."
                if retained
                else "Joint retirement rule not satisfied; the EK5 premium is retired: " + ", ".join(causes)
            ),
        }
    )
    report["report_sha256"] = _sha256_json(report)
    return report


# ---------------------------------------------------------------------------
# 5. Open DEV fixtures (synthetic; deterministic)
# ---------------------------------------------------------------------------
_FIXTURE_SALT = "pv7-trial2-open-dev"
_FIXTURE_BASE_RATE = 0.3


def _pattern(scenario: str, arm: str, i: int) -> str:
    """Outcome of ``arm`` on natural-conflict matter ``i`` for a fixture scenario."""
    if scenario == "win":
        if arm == ARM_EK5_ON:
            return OUTCOME_CORRECT
        return OUTCOME_WRONG if i % 2 == 0 else OUTCOME_ABSTAIN
    if scenario == "loss":
        if arm == ARM_EK5_ON:
            return OUTCOME_WRONG if i % 2 == 0 else OUTCOME_ABSTAIN
        return OUTCOME_CORRECT
    if scenario == "tie":
        return OUTCOME_WRONG if i % 5 == 0 else OUTCOME_CORRECT
    if scenario == "timeout":
        if arm == ARM_EK5_ON:
            return OUTCOME_TIMEOUT if i % 4 == 0 else OUTCOME_CORRECT
        return OUTCOME_WRONG if i % 2 == 0 else OUTCOME_ABSTAIN
    return OUTCOME_CORRECT  # insufficient_natural_conflicts: outcomes are irrelevant


def _labor(scenario: str, arm: str) -> Tuple[float, float]:
    if scenario == "win":
        return (2.0, 1.0) if arm == ARM_EK5_ON else (3.0, 4.0)
    if scenario == "loss":
        return (4.0, 5.0) if arm == ARM_EK5_ON else (2.0, 1.0)
    return (2.0, 2.0)


def _rubric_scores(scenario: str, arm: str, i: int) -> Dict[str, int]:
    good = {c: 1 for c in RUBRIC_CRITERIA}
    if scenario == "loss" and arm == ARM_EK5_ON:
        return {c: (1 if k == 0 else 0) for k, c in enumerate(RUBRIC_CRITERIA)}
    if scenario == "win" and arm != ARM_EK5_ON:
        return {c: (1 if (k + i) % 2 == 0 else 0) for k, c in enumerate(RUBRIC_CRITERIA)}
    return good


def build_open_dev_fixture(scenario: str) -> Tuple[List[MatterRow], float, Dict[str, Dict[str, str]]]:
    """Return ``(rows, declared_base_rate, unblinding_key)`` for one synthetic DEV scenario.

    Matters ``seq % 10 in (0, 3, 6)`` are natural conflicts: a declared 30% base rate gives
    ``ceil(30 / 0.3) = 100`` consecutive matters containing exactly 30 natural conflicts
    (the minimum).  The ``insufficient_natural_conflicts`` fixture has the same 100 matters
    but only 20 natural conflicts (``seq % 10 in (0, 5)``).
    """
    if scenario not in DEV_FIXTURE_SCENARIOS:
        raise ValueError(f"Unknown DEV fixture scenario={scenario!r}; expected one of {DEV_FIXTURE_SCENARIOS}")
    n_matters = 100  # ceil(30 / 0.3) == 100
    natural_slots = {0, 3, 6} if scenario != "insufficient_natural_conflicts" else {0, 5}
    rows: List[MatterRow] = []
    nat_index = 0
    for seq in range(n_matters):
        matter_id = f"dev_{scenario}_m{seq:03d}"
        natural = (seq % 10) in natural_slots
        arms: Dict[str, ArmResult] = {}
        rubric: Dict[str, Dict[str, int]] = {}
        codes = blind_codes(matter_id, _FIXTURE_SALT)
        for arm in MANDATORY_ARMS:
            if natural:
                extraction, review = _labor(scenario, arm)
                arms[arm] = ArmResult(_pattern(scenario, arm, nat_index), extraction, review)
                rubric[codes[arm]] = _rubric_scores(scenario, arm, nat_index)
            else:
                arms[arm] = ArmResult(OUTCOME_CORRECT, 1.0, 0.0)
        if natural:
            nat_index += 1
        rows.append(MatterRow(seq, matter_id, natural, arms, rubric))
    key = make_unblinding_key([r.matter_id for r in rows], _FIXTURE_SALT)
    return rows, _FIXTURE_BASE_RATE, key


def run_offline_dev_fixture_suite() -> Dict[str, Any]:
    """Exercise the offline row-level scorer across all 5 open DEV fixtures."""
    scenarios: Dict[str, Any] = {}
    for scenario in DEV_FIXTURE_SCENARIOS:
        rows, base_rate, key = build_open_dev_fixture(scenario)
        scenarios[scenario] = score_trial2_rows(
            rows,
            declared_base_rate=base_rate,
            unblinding_key=key,
            data_origin="open_dev_fixture",
            fixture_label=f"open_dev_{scenario}",
        )
    suite: Dict[str, Any] = {
        "bundle_id": BUNDLE_ID,
        "track_id": TRACK_ID,
        "claim_tier": CLAIM_TIER,
        "status": STAGE1_STATUS,
        "fixture_split": "DEV_OPEN",
        "data_origin": "open_dev_fixture",
        "protocol_digest": PROTOCOL_DIGEST,
        "mandatory_arms": list(MANDATORY_ARMS),
        "scenarios": scenarios,
    }
    suite["suite_sha256"] = _sha256_json(suite)
    return suite


# ---------------------------------------------------------------------------
# 6. CLI
# ---------------------------------------------------------------------------
def load_rows_jsonl(path: Path) -> List[MatterRow]:
    rows: List[MatterRow] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped:
            rows.append(MatterRow.from_dict(json.loads(stripped)))
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Bundle #64 Trial 2 Records-Owner Offline Scorer")
    parser.add_argument("--run-dev-fixtures", action="store_true", help="Run all 5 open DEV fixtures")
    parser.add_argument("--rows-jsonl", type=Path, default=None, help="Partner closed-matter row log (JSONL)")
    parser.add_argument("--unblinding-key-json", type=Path, default=None, help="{matter_id: {code: arm}} JSON")
    parser.add_argument("--declared-base-rate", type=float, default=None, help="Pre-registered natural base rate")
    parser.add_argument("--output", type=Path, default=None, help="Optional output JSON report path")
    args = parser.parse_args(argv)

    if args.rows_jsonl is not None:
        if args.unblinding_key_json is None or args.declared_base_rate is None:
            parser.error("--rows-jsonl requires --unblinding-key-json and --declared-base-rate")
        payload = score_trial2_rows(
            load_rows_jsonl(args.rows_jsonl),
            declared_base_rate=args.declared_base_rate,
            unblinding_key=json.loads(args.unblinding_key_json.read_text(encoding="utf-8")),
            data_origin="partner",
            fixture_label=args.rows_jsonl.name,
        )
    else:
        payload = run_offline_dev_fixture_suite()

    rendered = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


__all__ = [
    "ARM_CHALLENGER",
    "ARM_EK5_OFF",
    "ARM_EK5_ON",
    "ArmResult",
    "BUNDLE_ID",
    "CLAIM_TIER",
    "DEV_FIXTURE_SCENARIOS",
    "LABOR_COST_PER_MINUTE",
    "MANDATORY_ARMS",
    "MIN_NATURAL_CONFLICTS",
    "MatterRow",
    "OWNER_LOSS_WEIGHTS",
    "PROTOCOL_DIGEST",
    "RUBRIC_CRITERIA",
    "STAGE1_STATUS",
    "TRACK_ID",
    "TrialProtocolError",
    "blind_codes",
    "build_open_dev_fixture",
    "locked_protocol",
    "lower_confidence_bound",
    "main",
    "make_unblinding_key",
    "required_consecutive_matters",
    "run_offline_dev_fixture_suite",
    "score_trial2_rows",
]


if __name__ == "__main__":
    raise SystemExit(main())
