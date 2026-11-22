"""Standalone offline Cohen's kappa scorer for the PM-4 100-item blind human audit sheet.

Usage:
  python scripts/score_pm4_human_audit.py \
      --blind-sheet artifacts/flagship1-night1/20261122-overnight-r5/spark6724/r5_s4_pm4_cross_family_2x2_and_audit/pm4_human_audit_100_blind_sheet.csv \
      --sealed-key artifacts/flagship1-night1/20261122-overnight-r5/spark6724/r5_s4_pm4_cross_family_2x2_and_audit/pm4_human_audit_100_sealed_key.json
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Set, Tuple


def cohens_kappa_safe(y1: List[int], y2: List[int]) -> Tuple[float, float, bool]:
    """Return `(kappa, raw_agreement, degenerate_marginals)` with fail-safe zero-denominator guard."""
    if len(y1) != len(y2) or len(y1) == 0:
        raise ValueError("y1 and y2 must be non-empty lists of equal length")
    n = float(len(y1))
    po = sum(1 for a, b in zip(y1, y2) if a == b) / n
    p1 = sum(y1) / n
    p2 = sum(y2) / n
    pe = p1 * p2 + (1.0 - p1) * (1.0 - p2)
    if abs(1.0 - pe) <= 1e-12:
        # Degenerate constant marginals: kappa is undefined unless exact 100% agreement; flag explicitly!
        return (1.0 if abs(po - 1.0) <= 1e-12 else 0.0), po, True
    return (po - pe) / (1.0 - pe), po, False


def main() -> None:
    parser = argparse.ArgumentParser(description="Score PM-4 100-item human audit against sealed LLM judge grades.")
    parser.add_argument("--blind-sheet", type=Path, required=True)
    parser.add_argument("--sealed-key", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=100)
    args = parser.parse_args()

    key_data = json.loads(args.sealed_key.read_text(encoding="utf-8"))
    by_id: Dict[str, Dict] = {it["audit_id"]: it for it in key_data["items"]}
    if len(by_id) != args.expected_count:
        raise ValueError(f"Sealed key contains {len(by_id)} unique IDs, expected {args.expected_count}")

    human_grades: List[int] = []
    nem_grades: List[int] = []
    qwen_grades: List[int] = []
    seen_ids: Set[str] = set()

    with args.blind_sheet.open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            aid = row["audit_id"].strip()
            if aid in seen_ids:
                raise ValueError(f"Duplicate audit_id='{aid}' in blind sheet")
            if aid not in by_id:
                raise ValueError(f"Unknown audit_id='{aid}' not in sealed key")
            seen_ids.add(aid)
            raw = row["human_yes_or_no"].strip().upper()
            if raw not in ("YES", "NO", "1", "0"):
                raise ValueError(f"Missing or invalid human_yes_or_no='{raw}' on {aid}")
            h = 1 if raw in ("YES", "1") else 0
            rec = by_id[aid]
            human_grades.append(h)
            nem_grades.append(int(rec.get("nemotron_fresh_judge", rec["nemotron_judge_grade"])))
            if "qwen_judge" in rec:
                qwen_grades.append(int(rec["qwen_judge"]))

    if seen_ids != set(by_id.keys()):
        missing = sorted(set(by_id.keys()) - seen_ids)
        raise ValueError(f"Incomplete human audit sheet: {len(seen_ids)}/{len(by_id)} graded (missing {missing[:5]})")

    k_nem, raw_nem, deg_nem = cohens_kappa_safe(human_grades, nem_grades)
    print(f"Graded unique items: {len(human_grades)}/{args.expected_count}")
    print(
        f"Human vs Nemotron Judge: raw_acc={raw_nem:.4f}, kappa={k_nem:.4f}, "
        f"degenerate={deg_nem} (PM-4b pass >= 0.85 & not degenerate: {k_nem >= 0.85 and not deg_nem})"
    )
    if len(qwen_grades) == len(human_grades):
        k_qw, raw_qw, deg_qw = cohens_kappa_safe(human_grades, qwen_grades)
        print(
            f"Human vs Qwen2.5 Judge:  raw_acc={raw_qw:.4f}, kappa={k_qw:.4f}, "
            f"degenerate={deg_qw} (PM-4b pass >= 0.85 & not degenerate: {k_qw >= 0.85 and not deg_qw})"
        )


if __name__ == "__main__":
    main()
