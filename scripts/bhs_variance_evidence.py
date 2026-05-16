#!/usr/bin/env python3
"""Runtime evidence script for CD-244-02.

Drives ``aep_orchestrator.AEPOrchestrator`` through synthesis + closure with
two distinct finding sets (rich vs. sparse) and writes the resulting
``avg_bhs_score`` values to a JSON artifact. This is the deterministic
end-to-end exercise that proves the BHS gate no longer returns a hardcoded
0.0 for every cycle.

Usage:
    python scripts/bhs_variance_evidence.py [--out path/to/evidence.json]

Default output: ``artifacts/bhs_variance_evidence.json``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from aep_orchestrator import AEPOrchestrator  # noqa: E402


def _run(findings):
    orch = AEPOrchestrator()
    f = orch.discovery(findings, pr_number=999)
    orch.synthesis(f)
    return orch.closure()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out",
        default=str(REPO_ROOT / "artifacts" / "bhs_variance_evidence.json"),
        help="Output JSON path (default: artifacts/bhs_variance_evidence.json)",
    )
    args = parser.parse_args()

    rich = [
        {
            "title": "Production endpoint regression",
            "severity": "HIGH",
            "impact": "Endpoint at server.py:120 returns wrong shape",
            "recommended_fix": "Patch handler at api.py:55; artifact: logs/outage.json",
            "effort": "S",
        },
        {
            "title": "Auth bypass",
            "severity": "CRITICAL",
            "impact": "Auth bypass at auth.py:42 leaks tokens; commit a1b2c3d4e5f",
            "recommended_fix": "Rewrite flow; reference PR #100",
            "effort": "M",
        },
    ]
    sparse = [
        {"title": "Vague A", "severity": "MEDIUM", "effort": "S"},
        {"title": "Vague B", "severity": "LOW", "effort": "S"},
    ]

    rich_summary = _run(rich)
    sparse_summary = _run(sparse)

    artifact = {
        "task": "CD-244-02",
        "purpose": "verify summary['avg_bhs_score'] varies with finding content",
        "rich_avg_bhs_score": rich_summary["avg_bhs_score"],
        "rich_bhs_samples": rich_summary["bhs_samples"],
        "sparse_avg_bhs_score": sparse_summary["avg_bhs_score"],
        "sparse_bhs_samples": sparse_summary["bhs_samples"],
        "differ": rich_summary["avg_bhs_score"] != sparse_summary["avg_bhs_score"],
        "rich_higher": rich_summary["avg_bhs_score"] > sparse_summary["avg_bhs_score"],
    }

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(artifact, indent=2), encoding="utf-8")
    print(json.dumps(artifact, indent=2))
    print(f"\nWrote {out}")

    if not artifact["differ"]:
        print("FAIL: avg_bhs_score did not vary — CD-244-02 NOT closed.", file=sys.stderr)
        return 1
    if not artifact["rich_higher"]:
        print("FAIL: rich findings did not score higher than sparse.", file=sys.stderr)
        return 1
    print("PASS: avg_bhs_score varies; rich > sparse. CD-244-02 verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
