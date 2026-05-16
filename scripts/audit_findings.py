#!/usr/bin/env python3
"""scripts/audit_findings.py — operator-driven semantic audit of AEP findings.

## Why this script exists (CD-245-01 Part B)

``scripts/bhs_validator.py`` is a **structural** rubric: it checks field
presence, evidence pointers, severity enum, content length, and content
quality (entropy / unique tokens / dominant-token ratio). It does NOT make
a semantic judgement about whether a finding's prose is *meaningful*.

A motivated operator can compose prose that passes every structural signal
while saying nothing useful (``"foo bar baz qux at handler.py:42"`` — 4
unique tokens, entropy ~3.4, evidence pointer present, no lie markers →
structural score 100). The rulebook v3.3 § Closing Note acknowledges this
class of gap and the proposal accepted on 2026-05-16 (Candidate A + B,
per CD-245-01) is to mitigate it with a periodic human sample-grade
rather than escalating to non-deterministic LLM-based scoring (which would
conflict with Session Rule #1).

## What this script does

Reads recent AEP findings from one or more JSON artifacts, presents each
to a human operator with the structural BHS score already attached, and
asks the operator for a 1–10 semantic grade (or `s` to skip). Writes the
audit result to ``artifacts/findings_audit_YYYY-MM-DD.json`` and prints
aggregate sample statistics:

  - n graded, n skipped
  - sample mean of semantic grades
  - delta between mean structural score (already in the input) and
    mean semantic grade × 10 (scaled to the same 0–100 axis)
  - per-finding rows where structural - semantic > 30 (potential
    structural-passes-but-semantically-empty cases)

The artifact path + latest-audit timestamp is also written to
``artifacts/findings_audit_latest.json`` so the AEP closure summary can
surface "Last audit: 2026-05-16 (12 findings, mean semantic 7.3 vs
structural 92, delta -19)".

## How operators use it

Weekly (or per AEP cycle, whichever is shorter):

  $ python scripts/audit_findings.py --input artifacts/aep_findings.json --sample 12
  Finding F-001 [HIGH] structural=92
    impact: "..."
    recommended_fix: "..."
  Semantic grade (1-10, s=skip, q=quit): 8
  ...

  Audit complete: 12 graded, 0 skipped.
  Sample mean semantic grade: 7.3
  Sample mean structural score: 92
  Delta (structural - semantic*10): 19
  Wrote artifacts/findings_audit_2026-05-16.json

## Non-goals

- This script does NOT block any merge gate. It is an audit tool, not
  a gate. If the operator finds repeated structural-passes-but-empty
  findings, the rulebook recommends adding a Carried Debt entry for
  rubric drift; this script does not auto-file that entry.
- This script does NOT use any LLM or external API. Per Session Rule #1
  there is no semantic substitute for human judgement here.
- This script is NOT run in CI. CI has no operator to grade.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Repo root is one level above /scripts.
_REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_REPO_ROOT))

from scripts.bhs_validator import validate_pr_brutal_honesty  # noqa: E402


def _load_findings(input_paths: List[Path]) -> List[Dict[str, Any]]:
    """Load findings from one or more JSON files.

    Each file may contain either a list of findings, or a dict with a
    ``findings`` key holding the list. Other shapes raise ``ValueError``.
    """
    out: List[Dict[str, Any]] = []
    for path in input_paths:
        text = path.read_text(encoding="utf-8")
        data = json.loads(text)
        if isinstance(data, list):
            out.extend(data)
        elif isinstance(data, dict) and isinstance(data.get("findings"), list):
            out.extend(data["findings"])
        else:
            raise ValueError(
                f"{path}: expected a list of findings or {{'findings': [...]}}; "
                f"got {type(data).__name__}"
            )
    return out


def _sample(findings: List[Dict[str, Any]], n: int, seed: Optional[int]) -> List[Dict[str, Any]]:
    """Return a random sample of up to n findings, deterministic given seed."""
    if n >= len(findings):
        return list(findings)
    rng = random.Random(seed)
    return rng.sample(findings, n)


def _present_finding(idx: int, total: int, finding: Dict[str, Any], structural_score: float) -> None:
    print()
    print("=" * 72)
    print(f"Finding {idx + 1}/{total} — id={finding.get('id', '?')!r} "
          f"severity={finding.get('severity', '?')!r} structural={structural_score:.0f}")
    print("=" * 72)
    for key in ("impact", "recommended_fix", "title", "description"):
        if key in finding and finding[key]:
            print(f"  {key}: {finding[key]}")


def _prompt_grade(input_fn=input) -> Optional[int]:
    """Prompt for a 1-10 grade; return None to skip; raise SystemExit on quit.

    ``input_fn`` injected for unit-test reachability — tests pass a fake.
    """
    while True:
        try:
            raw = input_fn("Semantic grade (1-10, s=skip, q=quit): ").strip().lower()
        except EOFError:
            raise SystemExit("\nEOF on stdin; aborting audit.")
        if raw == "q":
            raise SystemExit("\nOperator quit; partial audit not written.")
        if raw == "s":
            return None
        try:
            grade = int(raw)
        except ValueError:
            print(f"  invalid: expected 1-10, 's', or 'q'; got {raw!r}")
            continue
        if not 1 <= grade <= 10:
            print(f"  invalid: grade must be in [1, 10]; got {grade}")
            continue
        return grade


def run_audit(
    findings: List[Dict[str, Any]],
    sample_size: int,
    seed: Optional[int],
    output_dir: Path,
    input_fn=input,
) -> Dict[str, Any]:
    """Run an interactive audit over a sample of findings.

    Returns the audit result dict (also written to disk).
    """
    today = dt.date.today().isoformat()
    sample = _sample(findings, sample_size, seed)
    rows: List[Dict[str, Any]] = []

    for i, finding in enumerate(sample):
        structural = validate_pr_brutal_honesty(finding_dict=finding)
        _present_finding(i, len(sample), finding, structural.score)
        grade = _prompt_grade(input_fn=input_fn)
        rows.append({
            "finding_id": finding.get("id"),
            "severity": finding.get("severity"),
            "structural_score": structural.score,
            "semantic_grade": grade,
            "structural_flags": structural.optimism_flags,
        })

    graded = [r for r in rows if r["semantic_grade"] is not None]
    n_graded = len(graded)
    n_skipped = len(rows) - n_graded

    mean_semantic = (
        sum(r["semantic_grade"] for r in graded) / n_graded if n_graded else 0.0
    )
    mean_structural = (
        sum(r["structural_score"] for r in graded) / n_graded if n_graded else 0.0
    )
    delta = mean_structural - mean_semantic * 10.0
    structural_passes_but_low_semantic = [
        r for r in graded if r["structural_score"] - r["semantic_grade"] * 10.0 > 30.0
    ]

    audit = {
        "date": today,
        "sample_size_requested": sample_size,
        "sample_size_actual": len(sample),
        "n_graded": n_graded,
        "n_skipped": n_skipped,
        "mean_semantic_grade": round(mean_semantic, 2),
        "mean_structural_score": round(mean_structural, 2),
        "delta_structural_minus_semantic_scaled": round(delta, 2),
        "structural_passes_but_low_semantic": [
            {"finding_id": r["finding_id"], "structural": r["structural_score"], "semantic": r["semantic_grade"]}
            for r in structural_passes_but_low_semantic
        ],
        "rows": rows,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"findings_audit_{today}.json"
    out_path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    latest_path = output_dir / "findings_audit_latest.json"
    try:
        artifact_path_str = str(out_path.relative_to(_REPO_ROOT))
    except ValueError:
        # output_dir lives outside the repo (e.g. tempdir in tests); use absolute.
        artifact_path_str = str(out_path)
    latest_path.write_text(
        json.dumps({
            "date": today,
            "artifact_path": artifact_path_str,
            "n_graded": n_graded,
            "mean_semantic_grade": round(mean_semantic, 2),
            "mean_structural_score": round(mean_structural, 2),
            "delta_structural_minus_semantic_scaled": round(delta, 2),
        }, indent=2),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print(f"Audit complete: {n_graded} graded, {n_skipped} skipped.")
    if n_graded:
        print(f"  Sample mean semantic grade: {mean_semantic:.2f} (out of 10)")
        print(f"  Sample mean structural score: {mean_structural:.2f} (out of 100)")
        print(f"  Delta (structural - semantic*10): {delta:+.2f}")
        if structural_passes_but_low_semantic:
            print(f"  WARNING: {len(structural_passes_but_low_semantic)} finding(s) "
                  f"scored structurally >= semantic*10 + 30 — potential rubric-gap "
                  f"(see {out_path}).")
    print(f"  Wrote {out_path}")
    print(f"  Updated {latest_path}")
    return audit


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__.split("\n\n", 1)[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--input", "-i", action="append", type=Path, required=True,
        help="JSON file with findings list. Can be passed multiple times.",
    )
    parser.add_argument(
        "--sample", "-n", type=int, default=12,
        help="Sample size (default 12). If fewer findings exist, audits all.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Random seed for sample selection (default: nondeterministic).",
    )
    parser.add_argument(
        "--output-dir", "-o", type=Path,
        default=_REPO_ROOT / "artifacts",
        help="Directory for audit JSON output (default: artifacts/).",
    )
    args = parser.parse_args(argv)

    for path in args.input:
        if not path.exists():
            print(f"ERROR: input file does not exist: {path}", file=sys.stderr)
            return 1

    findings = _load_findings(args.input)
    if not findings:
        print("ERROR: no findings loaded from input files.", file=sys.stderr)
        return 1

    run_audit(findings, args.sample, args.seed, args.output_dir)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
