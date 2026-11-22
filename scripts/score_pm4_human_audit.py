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
from typing import Dict, List


def cohens_kappa(y1: List[int], y2: List[int]) -> float:
    n = float(len(y1))
    po = sum(1 for a, b in zip(y1, y2) if a == b) / n
    p1 = sum(y1) / n
    p2 = sum(y2) / n
    pe = p1 * p2 + (1.0 - p1) * (1.0 - p2)
    return 1.0 if abs(1.0 - pe) <= 1e-12 else (po - pe) / (1.0 - pe)


def main() -> None:
    parser = argparse.ArgumentParser(description="Score PM-4 100-item human audit against sealed LLM judge grades.")
    parser.add_argument("--blind-sheet", type=Path, required=True)
    parser.add_argument("--sealed-key", type=Path, required=True)
    args = parser.parse_args()

    key_data = json.loads(args.sealed_key.read_text(encoding="utf-8"))
    by_id: Dict[str, Dict] = {it["audit_id"]: it for it in key_data["items"]}

    human_grades: List[int] = []
    nem_grades: List[int] = []
    qwen_grades: List[int] = []

    with args.blind_sheet.open("r", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            raw = row["human_yes_or_no"].strip().upper()
            if raw not in ("YES", "NO", "1", "0"):
                raise ValueError(f"Missing or invalid human_yes_or_no='{raw}' on {row['audit_id']}")
            h = 1 if raw in ("YES", "1") else 0
            rec = by_id[row["audit_id"]]
            human_grades.append(h)
            nem_grades.append(int(rec.get("nemotron_fresh_judge", rec["nemotron_judge_grade"])))
            if "qwen_judge" in rec:
                qwen_grades.append(int(rec["qwen_judge"]))

    k_nem = cohens_kappa(human_grades, nem_grades)
    print(f"Graded items: {len(human_grades)}")
    print(f"Human vs Nemotron Judge Cohen's kappa: {k_nem:.4f} (PM-4 pass >= 0.85: {k_nem >= 0.85})")
    if len(qwen_grades) == len(human_grades):
        k_qw = cohens_kappa(human_grades, qwen_grades)
        print(f"Human vs Qwen2.5 Judge Cohen's kappa:  {k_qw:.4f} (PM-4 pass >= 0.85: {k_qw >= 0.85})")


if __name__ == "__main__":
    main()
