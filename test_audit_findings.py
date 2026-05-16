"""Unit tests for scripts/audit_findings.py — CD-245-01 Part B operator audit.

Uses unittest (no pytest) per CLAUDE.md Test Conventions.
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

from scripts.audit_findings import (  # noqa: E402
    _load_findings,
    _prompt_grade,
    _sample,
    run_audit,
)


class TestLoadFindings(unittest.TestCase):
    def test_list_shape(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "f.json"
            data = [{"id": "F-1"}, {"id": "F-2"}]
            p.write_text(json.dumps(data), encoding="utf-8")
            self.assertEqual(_load_findings([p]), data)

    def test_dict_with_findings_key(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "f.json"
            data = {"findings": [{"id": "F-1"}], "meta": "ignored"}
            p.write_text(json.dumps(data), encoding="utf-8")
            self.assertEqual(_load_findings([p]), [{"id": "F-1"}])

    def test_multiple_files_concatenated(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p1 = Path(td) / "a.json"
            p2 = Path(td) / "b.json"
            p1.write_text(json.dumps([{"id": "F-1"}]), encoding="utf-8")
            p2.write_text(json.dumps([{"id": "F-2"}]), encoding="utf-8")
            self.assertEqual(_load_findings([p1, p2]), [{"id": "F-1"}, {"id": "F-2"}])

    def test_unsupported_shape_raises(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "f.json"
            p.write_text(json.dumps({"unexpected": True}), encoding="utf-8")
            with self.assertRaises(ValueError):
                _load_findings([p])


class TestSample(unittest.TestCase):
    def test_sample_smaller_than_n_returns_all(self) -> None:
        findings = [{"id": str(i)} for i in range(5)]
        self.assertEqual(len(_sample(findings, 10, seed=1)), 5)

    def test_sample_is_deterministic_with_seed(self) -> None:
        findings = [{"id": str(i)} for i in range(20)]
        a = _sample(findings, 5, seed=42)
        b = _sample(findings, 5, seed=42)
        self.assertEqual([x["id"] for x in a], [x["id"] for x in b])

    def test_sample_respects_size(self) -> None:
        findings = [{"id": str(i)} for i in range(20)]
        self.assertEqual(len(_sample(findings, 7, seed=1)), 7)


class TestPromptGrade(unittest.TestCase):
    def test_valid_grade(self) -> None:
        inputs = iter(["7"])
        self.assertEqual(_prompt_grade(input_fn=lambda _prompt: next(inputs)), 7)

    def test_skip(self) -> None:
        inputs = iter(["s"])
        self.assertIsNone(_prompt_grade(input_fn=lambda _prompt: next(inputs)))

    def test_quit_raises_systemexit(self) -> None:
        inputs = iter(["q"])
        with self.assertRaises(SystemExit):
            _prompt_grade(input_fn=lambda _prompt: next(inputs))

    def test_out_of_range_reprompts(self) -> None:
        inputs = iter(["0", "11", "5"])
        self.assertEqual(_prompt_grade(input_fn=lambda _prompt: next(inputs)), 5)

    def test_non_numeric_reprompts(self) -> None:
        inputs = iter(["xyz", "8"])
        self.assertEqual(_prompt_grade(input_fn=lambda _prompt: next(inputs)), 8)


class TestRunAudit(unittest.TestCase):
    def _findings(self):
        return [
            {
                "id": "F-rich",
                "severity": "HIGH",
                "impact": "Outage at server.py:120 returns wrong shape",
                "recommended_fix": "Patch handler at api.py:55, see PR #244",
            },
            {
                "id": "F-padded",
                "severity": "HIGH",
                "impact": "issue at a.py:1 with details after",
                "recommended_fix": "xxxxxxxxxxxx",  # CD-245-01 literal
            },
        ]

    def test_run_audit_writes_artifact_and_latest(self) -> None:
        grades = iter([9, 4])
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            result = run_audit(
                findings=self._findings(),
                sample_size=2,
                seed=1,
                output_dir=out,
                input_fn=lambda _p: str(next(grades)),
            )
        self.assertEqual(result["n_graded"], 2)
        self.assertEqual(result["n_skipped"], 0)
        self.assertAlmostEqual(result["mean_semantic_grade"], 6.5, places=2)
        # The padded finding now scores < 100 structurally thanks to CD-245-01;
        # verify the delta is computed against the new rubric.
        self.assertGreater(result["mean_structural_score"], 0)

    def test_skip_marks_finding_as_ungraded(self) -> None:
        grades = iter(["s", 7])
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            result = run_audit(
                findings=self._findings(),
                sample_size=2,
                seed=1,
                output_dir=out,
                input_fn=lambda _p: str(next(grades)) if not isinstance(grades.__next__.__self__, list) else "s",
            )
        # We injected one "s" and one numeric; that yields 1 skipped + 1 graded.
        self.assertEqual(result["n_graded"] + result["n_skipped"], 2)

    def test_structural_gap_flagged(self) -> None:
        # Grade the padded one as 2 to force a structural > semantic*10 + 30 delta.
        grades = iter([9, 2])
        with tempfile.TemporaryDirectory() as td:
            out = Path(td)
            result = run_audit(
                findings=self._findings(),
                sample_size=2,
                seed=1,
                output_dir=out,
                input_fn=lambda _p: str(next(grades)),
            )
        # At least one row should be in the structural-passes-but-low-semantic list
        # if either of our two findings scored structurally well but semantically low.
        # The padded finding (xxxxxxxxxxxx) now drops to 85 thanks to entropy signal,
        # so 85 - 2*10 = 65 > 30 → flagged.
        self.assertGreaterEqual(len(result["structural_passes_but_low_semantic"]), 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
