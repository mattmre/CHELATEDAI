"""SHIM-CD-06: scheduler verification evidence collection."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "scripts" / "record_shim_scheduler_evidence.py"


def _latest_artifact_for_day(
    artifact_dir: Path,
    suffix: str = "scheduler_evidence_",
) -> Path:
    matches = sorted(artifact_dir.glob(f"bhs_shim_{suffix}*.json"))
    if not matches:
        raise AssertionError("expected scheduler evidence artifact to exist")
    return sorted(matches, key=lambda p: p.stat().st_mtime)[-1]


class TestShimSchedulerEvidence(unittest.TestCase):
    def _run_script(
        self,
        scheduler_id: str,
        expected_agents: int,
        fixture: str | None = None,
        strict: bool = False,
        monkey_env: dict[str, str] | None = None,
    ) -> tuple[subprocess.CompletedProcess[str], dict[str, object]]:
        tmpdir = tempfile.mkdtemp()
        artifact_dir = Path(tmpdir)
        try:
            fixture_path = artifact_dir / "fixture.json"

            if fixture is not None:
                fixture_path.write_text(fixture, encoding="utf-8")

            env = os.environ.copy()
            env["CHELATED_SHIM_SCHEDULER_FIXTURE"] = (
                str(fixture_path) if fixture is not None else ""
            )
            env["CHELATED_SHIM_EVIDENCE_DIR"] = str(artifact_dir)
            if monkey_env:
                env.update(monkey_env)

            result = subprocess.run(
                [
                    "python",
                    str(SCRIPT),
                    "--scheduler-id",
                    scheduler_id,
                    "--expected-agents",
                    str(expected_agents),
                ]
                + (["--strict"] if strict else []),
                cwd=str(ROOT),
                env=env,
                capture_output=True,
                text=True,
                timeout=120,
            )

            artifact = _latest_artifact_for_day(artifact_dir)
            return result, json.loads(artifact.read_text(encoding="utf-8"))
        finally:
            shutil.rmtree(artifact_dir, ignore_errors=True)

    def test_scheduler_script_verifies_agent_count_from_fixture(self) -> None:
        result, data = self._run_script(
            "019e669bf1bb",
            10,
            fixture=json.dumps(
                {
                    "schedulers": [
                        {"id": "019e669bf1bb", "agents": 10, "name": "bhs loop"},
                        {"id": "other", "agents": 5},
                    ]
                }
            ),
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(data["scheduler_id"], "019e669bf1bb")
        self.assertTrue(data["verified"])
        self.assertTrue(data["found"])
        self.assertEqual(data["details"]["discovered_agent_count"], 10)

    def test_scheduler_script_fails_when_below_expected_agents(self) -> None:
        result, data = self._run_script(
            "019e669bf1bb",
            10,
            fixture=json.dumps(
                {"id": "019e669bf1bb", "agents": 5, "prompt": "exactly 5"}
            ),
            strict=True,
        )
        self.assertNotEqual(result.returncode, 0, result.stdout)
        self.assertFalse(data["verified"])
        self.assertTrue(data["found"])
        self.assertEqual(data["details"]["discovered_agent_count"], 5)
        self.assertEqual(data["details"]["expected_agent_count"], 10)

    def test_scheduler_script_uses_cli_probe_when_no_fixture(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            probe_path = Path(tmpdir) / "probe.sh"
            probe_path.write_text(
                "#!/usr/bin/env bash\n"
                "cat <<'EOF'\n"
                "019e669bf1bb: exactly 5 workers\n"
                "EOF\n",
                encoding="utf-8",
            )
            probe_path.chmod(0o755)

            result, data = self._run_script(
                "019e669bf1bb",
                5,
                fixture=None,
                monkey_env={"CHELATED_SHIM_SCHEDULER_CMD": str(probe_path)},
            )
            self.assertEqual(result.returncode, 0, result.stdout)
            self.assertTrue(data["found"])
            self.assertTrue(data["verified"])
            self.assertTrue(any("found with at least 5 agents" in note for note in data["notes"]))
            self.assertEqual(data["details"]["discovered_agent_count"], 5)
            self.assertEqual(data["details"]["expected_agent_count"], 5)

    def test_scheduler_script_strict_flag_fails_without_deterministic_probe(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            bad_cmd = path / "missing_probe"
            result, data = self._run_script(
                "019e669bf1bb",
                10,
                fixture=None,
                strict=True,
                monkey_env={
                    "CHELATED_SHIM_SCHEDULER_CMD": str(bad_cmd),
                    "CHELATED_SHIM_SCHEDULER_FIXTURE": "",
                },
            )
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertFalse(data["verified"])
            self.assertTrue(data["strict"])
            self.assertIn("scheduler id not present", " ".join(data["notes"]).lower())

    def test_scheduler_script_respects_scheduler_require_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir)
            bad_cmd = path / "missing_probe"
            result, data = self._run_script(
                "019e669bf1bb",
                10,
                fixture=None,
                monkey_env={
                    "CHELATED_SHIM_SCHEDULER_CMD": str(bad_cmd),
                    "CHELATED_SHIM_SCHEDULER_REQUIRE": "1",
                    "CHELATED_SHIM_SCHEDULER_FIXTURE": "",
                },
            )
            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertTrue(data["strict"])
            self.assertIn("not present", " ".join(data["notes"]).lower())


if __name__ == "__main__":
    unittest.main()
