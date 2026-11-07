import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from qscci_experiment import QSCCIError, canonical_json, sha256_bytes
from run_qscci import (
    ARTIFACT_NAME,
    DEEPSEEK_MODEL_ID,
    PlanServiceController,
    RESTORATION_SECONDS,
    SubprocessChildLauncher,
    RUNNING_METRIC,
    SupervisorConfig,
    WAITING_METRIC,
    _probe_health,
    _probe_idle,
    _probe_models,
    _redacted_utf8_tail,
    run_supervised,
    verify_result_dir,
)


class FakeProcess:
    def __init__(self, *, returncode=0, communicate_effect=None):
        self.returncode = returncode
        self.pid = 12345
        self._communicate_effect = communicate_effect
        self.communicate_calls = 0

    def communicate(self, timeout=None):
        self.communicate_calls += 1
        if self._communicate_effect is not None:
            return self._communicate_effect(self.communicate_calls, timeout)
        return b"worker stdout", b"worker stderr"

    def poll(self):
        return self.returncode


class FakeServiceController:
    def __init__(self, *, fail_restore=False, restoration_verified=True):
        self.fail_restore = fail_restore
        self.restoration_verified = restoration_verified
        self.events = []

    def observe_before(self):
        self.events.append("observe")
        return {"status": "AVAILABLE", "service_stopped": False, "container_ids": ["a", "b"]}

    def stop_if_required(self, before):
        self.events.append("stop")
        return {"service_stopped": True, "same_before": before["container_ids"]}

    def restore_and_verify(self, before, stopped, deadline):
        self.events.append("restore")
        if self.fail_restore:
            raise QSCCIError("simulated restoration failure")
        return {
            "status": "RESTORED",
            "service_stopped": stopped["service_stopped"],
            "same_container_ids": before["container_ids"],
            "restoration_verified": self.restoration_verified,
        }


class FakeChildLauncher:
    def __init__(
        self,
        *,
        process=None,
        launch_error=None,
        extra_file=False,
        invalid_run_collision=False,
        invalid_run_symlink_target=None,
        invalid_run_fifo=False,
        replace_stage_with_symlink=None,
        create_public_target=False,
    ):
        self.process = process or FakeProcess()
        self.launch_error = launch_error
        self.extra_file = extra_file
        self.invalid_run_collision = invalid_run_collision
        self.invalid_run_symlink_target = invalid_run_symlink_target
        self.invalid_run_fifo = invalid_run_fifo
        self.replace_stage_with_symlink = replace_stage_with_symlink
        self.create_public_target = create_public_target
        self.lifecycle = None

    def launch(self, config, stage_dir, lifecycle):
        self.lifecycle = lifecycle
        if self.launch_error is not None:
            raise self.launch_error
        artifact = {
            "service_lifecycle": lifecycle,
            "artifact_digest": "0" * 64,
            "disposition": "INVALID_RUN",
        }
        (stage_dir / ARTIFACT_NAME).write_bytes(canonical_json(artifact))
        if self.extra_file:
            (stage_dir / "unregistered.txt").write_text("tamper", encoding="utf-8")
        if self.invalid_run_collision:
            (stage_dir / "INVALID_RUN.json").write_text("child-secret-untrusted", encoding="utf-8")
        if self.invalid_run_symlink_target is not None:
            (stage_dir / "INVALID_RUN.json").symlink_to(self.invalid_run_symlink_target, target_is_directory=True)
        if self.invalid_run_fifo:
            os.mkfifo(stage_dir / "INVALID_RUN.json")
        if self.replace_stage_with_symlink is not None:
            displaced = stage_dir.with_name(f"{stage_dir.name}-child-displaced")
            stage_dir.rename(displaced)
            stage_dir.symlink_to(self.replace_stage_with_symlink, target_is_directory=True)
        if self.create_public_target:
            config.output_dir.mkdir()
            (config.output_dir / "child-sentinel.txt").write_text("child-created", encoding="utf-8")
        return self.process


def _config(parent):
    return SupervisorConfig(output_dir=Path(parent) / "result", repo_root=Path(__file__).resolve().parent)


def _service_plan(*, stop_required=False):
    model_id = DEEPSEEK_MODEL_ID
    return {
        "stop_required": stop_required,
        "container_identity_commands": [["identity", "node-a"], ["identity", "node-b"]],
        "idle_endpoint": {
            "url": "http://fixture/metrics",
            "method": "GET",
            "expected_status": 200,
            "running_metric": RUNNING_METRIC,
            "waiting_metric": WAITING_METRIC,
        },
        "stop_commands": [["stop", "node-a"], ["stop", "node-b"]] if stop_required else [],
        "start_commands": [["start", "cluster"]] if stop_required else [],
        "health_endpoint": {
            "url": "http://fixture/health",
            "method": "GET",
            "expected_status": 200,
            "expected_body": "",
        },
        "models_endpoint": {
            "url": "http://fixture/v1/models",
            "method": "GET",
            "expected_status": 200,
            "expected_model_id": model_id,
        },
        "completion_endpoint": {
            "url": "http://fixture/v1/completions",
            "method": "POST",
            "json": {"model": model_id, "prompt": "Return READY", "temperature": 0, "max_tokens": 4},
            "expected_status": 200,
            "expected_model_id": model_id,
            "expected_text": "READY",
        },
        "after_idle_endpoint": {
            "url": "http://fixture/metrics",
            "method": "GET",
            "expected_status": 200,
            "running_metric": RUNNING_METRIC,
            "waiting_metric": WAITING_METRIC,
        },
    }


class FakeHTTPResponse:
    def __init__(self, body, status=200):
        self.body = body
        self.status = status

    def read(self, _limit):
        return self.body

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False


def _typed_urlopen(request, timeout=None):
    del timeout
    url = request.full_url
    if url.endswith("/health"):
        body = b""
    elif url.endswith("/v1/models"):
        body = json.dumps(
            {"object": "list", "data": [{"id": DEEPSEEK_MODEL_ID}]}, separators=(",", ":")
        ).encode("utf-8")
    elif url.endswith("/v1/completions"):
        body = json.dumps(
            {"model": DEEPSEEK_MODEL_ID, "choices": [{"text": "READY"}]}, separators=(",", ":")
        ).encode("utf-8")
    elif "/metrics" in url:
        body = (
            f'{RUNNING_METRIC}{{model_name="{DEEPSEEK_MODEL_ID}"}} 0\n'
            f'{WAITING_METRIC}{{model_name="{DEEPSEEK_MODEL_ID}"}} 0\n'
        ).encode("utf-8")
    else:
        raise AssertionError(f"unexpected fixture URL: {url}")
    return FakeHTTPResponse(body)


def _identity_run(argv, **_kwargs):
    identity = argv[-1].encode("utf-8") if argv and argv[0] == "identity" else b""
    return subprocess.CompletedProcess(argv, 0, stdout=identity, stderr=b"")


class TestQSCCISupervisor(unittest.TestCase):
    def test_v4_inherits_exact_v1_restoration_ceiling(self):
        with tempfile.TemporaryDirectory() as temporary:
            self.assertEqual(RESTORATION_SECONDS, 600.0)
            self.assertEqual(_config(temporary).restoration_seconds, 600.0)
            with self.assertRaisesRegex(QSCCIError, "frozen at 600 seconds"):
                SupervisorConfig(
                    output_dir=Path(temporary) / "result",
                    repo_root=Path(__file__).resolve().parent,
                    restoration_seconds=2400.0,
                ).validate()

    def test_multiline_process_context_causes_conservative_whole_tail_redaction(self):
        hostile_tails = (
            b'{\n  "argv": [\n    "python",\n    "--lifecycle-b64",\n    "c2VjcmV0LWZyb20tcHJldHR5LWpzb24"\n  ]\n}',
            b'{"env": {\n "CUSTOM_CREDENTIAL": "not-covered-private-value"}}',
            b'ordinary prefix\ncommand:\n python worker.py\n --lifecycle-b64\\\n secret-value',
        )
        for raw in hostile_tails:
            with self.subTest(raw=raw):
                redacted, labels = _redacted_utf8_tail(raw)
                self.assertEqual(redacted, "")
                self.assertEqual(labels, ["all_stderr_text_withheld"])
                for private in (
                    "python", "c2VjcmV0LWZyb20tcHJldHR5LWpzb24",
                    "not-covered-private-value", "secret-value",
                ):
                    self.assertNotIn(private, redacted)

    def test_ordinary_traceback_text_is_withheld_but_digest_remains_available_in_carrier(self):
        raw = b'Traceback (most recent call last):\n  File "worker.py", line 17\nValueError: tensor shape mismatch'
        redacted, labels = _redacted_utf8_tail(raw)
        self.assertEqual(redacted, "")
        self.assertEqual(labels, ["all_stderr_text_withheld"])

    def test_multiline_quoted_secret_is_not_retained_from_traceback(self):
        raw = b'Traceback (most recent call last):\nValueError: PASSWORD="quoted\nsecret on next line"'
        redacted, labels = _redacted_utf8_tail(raw)
        self.assertEqual(redacted, "")
        self.assertEqual(labels, ["all_stderr_text_withheld"])
        self.assertNotIn("quoted", redacted)
        self.assertNotIn("secret", redacted)

    def test_attacker_controlled_traceback_identifiers_are_never_retained(self):
        hostile = (
            'Traceback (most recent call last):\n'
            '  File "/private", line 7, in APIKEY_sk_live_SUPERSECRET\n'
            'APIKEY_sk_live_SUPERSECRETError: msg\n'
        ).encode("utf-8")
        redacted, _labels = _redacted_utf8_tail(hostile)
        self.assertEqual(redacted, "")
        self.assertNotIn("SUPERSECRET", redacted)
        self.assertNotIn("APIKEY", redacted)

    def test_control_ansi_and_confusable_marker_evasion_cannot_enter_fixed_traceback_tokens(self):
        hostile = (
            "\x1b[31margv\x1b[0m: hidden\n"
            "еnv: hidden\n"
            "PASSWORD_supersecretError: details\n"
        ).encode("utf-8")
        redacted, _labels = _redacted_utf8_tail(hostile)
        self.assertEqual(redacted, "")
        for private in ("hidden", "supersecret", "PASSWORD", "argv", "еnv"):
            self.assertNotIn(private, redacted)

    def test_official_launcher_rejects_non_linux_before_service_touch(self):
        with tempfile.TemporaryDirectory() as temporary:
            service = FakeServiceController()
            with mock.patch("run_qscci._official_platform_supported", return_value=False), self.assertRaisesRegex(
                QSCCIError, "Linux/POSIX"
            ):
                run_supervised(_config(temporary), service, SubprocessChildLauncher())
            self.assertEqual(service.events, [])

    def test_success_restores_before_atomic_publication_and_retains_after_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = _config(temporary)
            service = FakeServiceController()
            launcher = FakeChildLauncher()
            with mock.patch("run_qscci.verify_artifact", return_value={"verified": True}):
                result = run_supervised(config, service, launcher)
            self.assertTrue(result["published"])
            self.assertEqual(service.events, ["observe", "stop", "restore"])
            self.assertEqual(launcher.lifecycle["after"], {"status": "PENDING"})
            retained = json.loads((config.output_dir / ARTIFACT_NAME).read_text(encoding="utf-8"))
            self.assertTrue(retained["service_lifecycle"]["after"]["restoration_verified"])
            self.assertEqual(sorted(path.name for path in config.output_dir.iterdir()), ["COMMIT.json", "manifest.json", "qscci.json"])
            self.assertEqual(list(Path(temporary).glob(".result.stage-*")), [])

    def test_restoration_failure_quarantines_successful_child_and_never_publishes(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = _config(temporary)
            with self.assertRaisesRegex(QSCCIError, "restoration failure"):
                run_supervised(config, FakeServiceController(fail_restore=True), FakeChildLauncher())
            self.assertFalse(config.output_dir.exists())
            quarantines = list(Path(temporary).glob(".result.quarantine-*"))
            self.assertEqual(len(quarantines), 1)
            self.assertTrue((quarantines[0] / ARTIFACT_NAME).is_file())

    def test_unverified_restoration_quarantines_and_never_publishes(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = _config(temporary)
            service = FakeServiceController(restoration_verified=False)
            with self.assertRaisesRegex(QSCCIError, "restoration did not verify"):
                run_supervised(config, service, FakeChildLauncher())
            self.assertFalse(config.output_dir.exists())
            self.assertEqual(len(list(Path(temporary).glob(".result.quarantine-*"))), 1)

    def test_unchanged_service_plan_reverifies_typed_health_models_completion_and_idle(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = _config(temporary)
            launcher = FakeChildLauncher()
            controller = PlanServiceController(_service_plan(stop_required=False))
            with mock.patch("run_qscci.urllib.request.urlopen", side_effect=_typed_urlopen), mock.patch(
                "run_qscci.subprocess.run", side_effect=_identity_run
            ), mock.patch("run_qscci.verify_artifact", return_value={"verified": True}):
                run_supervised(config, controller, launcher)
            retained = json.loads((config.output_dir / ARTIFACT_NAME).read_text(encoding="utf-8"))
            lifecycle = retained["service_lifecycle"]
            self.assertEqual(lifecycle["before"]["status"], "AVAILABLE")
            self.assertFalse(lifecycle["stop"]["service_stopped"])
            self.assertEqual(lifecycle["after"]["status"], "NOT_APPLICABLE")
            observations = lifecycle["after"]["verification_observations"]
            self.assertEqual(observations["health"]["body"], "")
            self.assertEqual(observations["models"]["model_id"], DEEPSEEK_MODEL_ID)
            self.assertEqual(observations["completion"]["text"], "READY")
            self.assertEqual(
                observations["idle"]["metric_sums"],
                {RUNNING_METRIC: 0.0, WAITING_METRIC: 0.0},
            )

    def test_empty_200_is_accepted_only_by_typed_health_probe(self):
        plan = _service_plan()
        with mock.patch("run_qscci.urllib.request.urlopen", return_value=FakeHTTPResponse(b"")):
            health = _probe_health(plan["health_endpoint"], 1.0)
            self.assertEqual(health["body"], "")
            with self.assertRaises(QSCCIError):
                _probe_models(plan["models_endpoint"], 1.0)

    def test_arbitrary_json_and_missing_endpoint_expectations_are_rejected(self):
        plan = _service_plan()
        malformed = dict(plan)
        malformed["models_endpoint"] = {"url": "http://fixture/v1/models", "method": "GET"}
        with self.assertRaises(QSCCIError):
            PlanServiceController(malformed)
        with mock.patch(
            "run_qscci.urllib.request.urlopen", return_value=FakeHTTPResponse(b'{"status":"ok"}')
        ), self.assertRaises(QSCCIError):
            _probe_models(plan["models_endpoint"], 1.0)
        wrong_request_model = _service_plan()
        wrong_request_model["completion_endpoint"]["json"]["model"] = "arbitrary-model"
        with self.assertRaises(QSCCIError):
            PlanServiceController(wrong_request_model)
        wrong_metric = _service_plan()
        wrong_metric["idle_endpoint"]["running_metric"] = "arbitrary_zero_metric"
        with self.assertRaises(QSCCIError):
            PlanServiceController(wrong_metric)
        duplicate_identity_command = _service_plan()
        duplicate_identity_command["container_identity_commands"][1] = list(
            duplicate_identity_command["container_identity_commands"][0]
        )
        with self.assertRaises(QSCCIError):
            PlanServiceController(duplicate_identity_command)

    def test_duplicate_container_identity_output_is_rejected(self):
        controller = PlanServiceController(_service_plan())
        with mock.patch("run_qscci._run_argv", return_value={"stdout": "same-container"}), self.assertRaises(
            QSCCIError
        ):
            controller.observe_before()

    def test_negative_nonfinite_and_missing_idle_metrics_are_rejected(self):
        specification = _service_plan()["idle_endpoint"]
        bodies = (
            f"{RUNNING_METRIC} -1\n{WAITING_METRIC} 0\n".encode("utf-8"),
            f"{RUNNING_METRIC} NaN\n{WAITING_METRIC} 0\n".encode("utf-8"),
            f"{RUNNING_METRIC} Inf\n{WAITING_METRIC} 0\n".encode("utf-8"),
            f"{RUNNING_METRIC} 0\n".encode("utf-8"),
        )
        for body in bodies:
            with self.subTest(body=body), mock.patch(
                "run_qscci.urllib.request.urlopen", return_value=FakeHTTPResponse(body)
            ), self.assertRaises(QSCCIError):
                _probe_idle(specification, 1.0)

    def test_partial_stop_failure_attempts_start_before_quarantine(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = _config(temporary)
            controller = PlanServiceController(_service_plan(stop_required=True))
            before = {
                "status": "AVAILABLE",
                "service_stopped": False,
                "container_identities": [{"identity": "a"}, {"identity": "b"}],
                "observations": {},
            }
            calls = []

            def command(argv, *_args, **_kwargs):
                calls.append(tuple(argv))
                if argv == ["stop", "node-b"]:
                    raise QSCCIError("partial stop failure")
                if argv[0] == "identity":
                    return before["container_identities"][0 if argv[-1] == "node-a" else 1]
                return {"returncode": 0}

            with mock.patch.object(controller, "observe_before", return_value=before), mock.patch(
                "run_qscci._run_argv", side_effect=command
            ), mock.patch("run_qscci._probe_health", return_value={}), mock.patch(
                "run_qscci._probe_models", return_value={}
            ), mock.patch("run_qscci._probe_completion", return_value={}), mock.patch(
                "run_qscci._probe_idle", return_value={}
            ):
                with self.assertRaisesRegex(QSCCIError, "partial stop failure"):
                    run_supervised(config, controller, FakeChildLauncher())
            self.assertIn(("start", "cluster"), calls)
            self.assertFalse(config.output_dir.exists())
            self.assertEqual(len(list(Path(temporary).glob(".result.quarantine-*"))), 1)

    def test_post_restore_artifact_verifier_failure_quarantines(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = _config(temporary)
            with mock.patch(
                "run_qscci.verify_artifact", side_effect=QSCCIError("post-restore verifier failure")
            ):
                with self.assertRaisesRegex(QSCCIError, "post-restore verifier failure"):
                    run_supervised(config, FakeServiceController(), FakeChildLauncher())
            self.assertFalse(config.output_dir.exists())
            quarantines = list(Path(temporary).glob(".result.quarantine-*"))
            self.assertEqual(len(quarantines), 1)
            self.assertTrue((quarantines[0] / ARTIFACT_NAME).is_file())

    def test_parent_fsync_failure_after_rename_quarantines_visible_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = _config(temporary)
            calls = 0

            def fail_parent_after_stage(_path):
                nonlocal calls
                calls += 1
                if calls >= 2:
                    raise OSError("simulated parent fsync failure")

            with mock.patch("run_qscci.verify_artifact", return_value={"verified": True}), mock.patch(
                "run_qscci._fsync_directory", side_effect=fail_parent_after_stage
            ):
                with self.assertRaisesRegex(OSError, "parent fsync failure"):
                    run_supervised(config, FakeServiceController(), FakeChildLauncher())
            self.assertFalse(config.output_dir.exists())
            quarantines = list(Path(temporary).glob(".result.quarantine-*"))
            self.assertEqual(len(quarantines), 1)
            self.assertTrue((quarantines[0] / ARTIFACT_NAME).is_file())

    @unittest.skipUnless(os.name == "nt", "Windows junction semantics")
    def test_output_parent_junction_is_rejected_before_service_touch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            real = root / "real"
            junction = root / "junction"
            real.mkdir()
            created = subprocess.run(
                ["cmd", "/c", "mklink", "/J", str(junction), str(real)],
                capture_output=True,
                check=False,
            )
            if created.returncode != 0:
                self.skipTest("junction creation unavailable")
            service = FakeServiceController()
            with self.assertRaisesRegex(QSCCIError, "reparse|junction"):
                run_supervised(
                    SupervisorConfig(junction / "result", Path(__file__).resolve().parent),
                    service,
                    FakeChildLauncher(),
                )
            self.assertEqual(service.events, [])

    def test_child_failure_still_restores_then_quarantines(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = _config(temporary)
            service = FakeServiceController()
            launcher = FakeChildLauncher(process=FakeProcess(returncode=7))
            with self.assertRaisesRegex(QSCCIError, "child failed"):
                run_supervised(config, service, launcher)
            self.assertEqual(service.events, ["observe", "stop", "restore"])
            self.assertFalse(config.output_dir.exists())
            quarantines = list(Path(temporary).glob(".result.quarantine-*"))
            self.assertEqual(len(quarantines), 1)
            evidence = json.loads((quarantines[0] / "INVALID_RUN.json").read_text(encoding="utf-8"))
            self.assertEqual(evidence["failure_category"], "CHILD_NONZERO")
            self.assertEqual(evidence["returncode"], 7)
            with self.assertRaisesRegex(QSCCIError, "ancestry"):
                verify_result_dir(quarantines[0], config.repo_root)

    def test_child_failure_evidence_is_bounded_utf8_safe_redacted_and_digest_bound(self):
        stderr = (
            b"\xffbroken\n" + b"x" * (32 * 1024)
            + b"\nargv=python run_qscci.py --lifecycle-b64 c2VjcmV0\n"
            + b"OPENAI_API_KEY=sk-openai\nHF_TOKEN=hf-secret\n"
            + b"Authorization: Bearer bearer-secret\nAWS_SECRET_ACCESS_KEY=aws-secret\n"
            + b"  env: PATH=private-process-environment\n"
            + b"{argv:python--foo}\n{\"environment\":\"private-json-env\"}\n"
            + b"PASSWORD=\"quoted secret with spaces\"\n"
        )

        def communicate_effect(_call, _timeout):
            return b"bounded stdout", stderr

        with tempfile.TemporaryDirectory() as temporary:
            config = _config(temporary)
            process = FakeProcess(returncode=9, communicate_effect=communicate_effect)
            with self.assertRaisesRegex(QSCCIError, "child failed"):
                run_supervised(config, FakeServiceController(), FakeChildLauncher(process=process))
            quarantine = next(Path(temporary).glob(".result.quarantine-*"))
            raw = (quarantine / "INVALID_RUN.json").read_bytes()
            evidence = json.loads(raw.decode("utf-8"))
            self.assertEqual(raw, canonical_json(evidence))
            self.assertEqual(evidence["stderr_byte_count"], len(stderr))
            self.assertEqual(evidence["stderr_full_sha256"], sha256_bytes(stderr))
            self.assertTrue(evidence["diagnostic_text_withheld"])
            self.assertNotIn("stderr_tail_utf8", evidence)
            self.assertNotIn("stderr_tail_sha256", evidence)
            serialized = raw.decode("utf-8")
            self.assertNotIn("c2VjcmV0", serialized)
            for secret in (
                "sk-openai", "hf-secret", "bearer-secret", "aws-secret",
                "private-process-environment", "python--foo", "private-json-env",
                "quoted secret with spaces",
            ):
                self.assertNotIn(secret, serialized)
            self.assertNotIn('"argv"', serialized)

    def test_diagnostic_write_failure_preserves_original_child_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = _config(temporary)
            with mock.patch("run_qscci._write_private_child_failure", side_effect=OSError("diagnostic disk failure")):
                with self.assertRaisesRegex(QSCCIError, "child failed"):
                    run_supervised(config, FakeServiceController(), FakeChildLauncher(process=FakeProcess(returncode=8)))
            self.assertFalse(config.output_dir.exists())
            self.assertEqual(len(list(Path(temporary).glob(".result.quarantine-*"))), 1)

    def test_child_precreated_invalid_run_is_atomically_replaced_by_supervisor(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = _config(temporary)
            launcher = FakeChildLauncher(process=FakeProcess(returncode=8), invalid_run_collision=True)
            with self.assertRaisesRegex(QSCCIError, "child failed"):
                run_supervised(config, FakeServiceController(), launcher)
            quarantine = next(Path(temporary).glob(".result.quarantine-*"))
            raw = (quarantine / "INVALID_RUN.json").read_bytes()
            evidence = json.loads(raw.decode("utf-8"))
            self.assertEqual(raw, canonical_json(evidence))
            self.assertNotIn(b"child-secret-untrusted", raw)

    @unittest.skipIf(os.name == "nt", "POSIX symlink semantics regression")
    def test_child_symlinked_invalid_run_is_replaced_without_touching_external_directory(self):
        with tempfile.TemporaryDirectory() as temporary:
            external = Path(temporary) / "external"
            external.mkdir()
            sentinel = external / "sentinel.txt"
            sentinel.write_text("must remain", encoding="utf-8")
            config = _config(temporary)
            launcher = FakeChildLauncher(
                process=FakeProcess(returncode=8), invalid_run_symlink_target=external
            )
            with self.assertRaisesRegex(QSCCIError, "child failed"):
                run_supervised(config, FakeServiceController(), launcher)
            quarantine = next(Path(temporary).glob(".result.quarantine-*"))
            evidence_path = quarantine / "INVALID_RUN.json"
            self.assertFalse(evidence_path.is_symlink())
            self.assertEqual(json.loads(evidence_path.read_text(encoding="utf-8"))["returncode"], 8)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "must remain")

    @unittest.skipIf(os.name == "nt", "POSIX FIFO semantics regression")
    def test_child_fifo_invalid_run_is_replaced_by_canonical_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = _config(temporary)
            launcher = FakeChildLauncher(process=FakeProcess(returncode=8), invalid_run_fifo=True)
            with self.assertRaisesRegex(QSCCIError, "child failed"):
                run_supervised(config, FakeServiceController(), launcher)
            quarantine = next(Path(temporary).glob(".result.quarantine-*"))
            evidence_path = quarantine / "INVALID_RUN.json"
            self.assertTrue(evidence_path.is_file())
            self.assertEqual(json.loads(evidence_path.read_text(encoding="utf-8"))["returncode"], 8)

    @unittest.skipIf(os.name == "nt", "POSIX symlink semantics regression")
    def test_child_stage_symlink_replacement_never_writes_external_and_preserves_failure(self):
        with tempfile.TemporaryDirectory() as temporary:
            external = Path(temporary) / "external"
            external.mkdir()
            sentinel = external / "sentinel.txt"
            sentinel.write_text("must remain", encoding="utf-8")
            config = _config(temporary)
            launcher = FakeChildLauncher(
                process=FakeProcess(returncode=8), replace_stage_with_symlink=external
            )
            with self.assertRaisesRegex(QSCCIError, "child failed"):
                run_supervised(config, FakeServiceController(), launcher)
            self.assertEqual(sentinel.read_text(encoding="utf-8"), "must remain")
            self.assertFalse((external / "INVALID_RUN.json").exists())
            self.assertFalse(config.output_dir.exists())
            self.assertFalse(any(path.is_symlink() for path in Path(temporary).glob(".result.stage-*")))

    def test_child_created_public_target_is_quarantined_on_nonzero_exit(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = _config(temporary)
            launcher = FakeChildLauncher(process=FakeProcess(returncode=8), create_public_target=True)
            with self.assertRaisesRegex(QSCCIError, "child failed"):
                run_supervised(config, FakeServiceController(), launcher)
            self.assertFalse(config.output_dir.exists())
            quarantines = list(Path(temporary).glob(".result.quarantine-*"))
            self.assertEqual(len(quarantines), 2)
            self.assertTrue(any((path / "child-sentinel.txt").is_file() for path in quarantines))
            self.assertTrue(any((path / "INVALID_RUN.json").is_file() for path in quarantines))

    def test_child_created_public_target_blocks_success_before_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = _config(temporary)
            launcher = FakeChildLauncher(create_public_target=True)
            with self.assertRaisesRegex(QSCCIError, "before supervisor publication"):
                run_supervised(config, FakeServiceController(), launcher)
            self.assertFalse(config.output_dir.exists())
            quarantines = list(Path(temporary).glob(".result.quarantine-*"))
            self.assertEqual(len(quarantines), 2)
            self.assertTrue(any((path / "child-sentinel.txt").is_file() for path in quarantines))

    def test_public_target_race_immediately_before_promotion_is_quarantined(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = _config(temporary)

            def fsync_with_race(path):
                if ".stage-" in path.name and not config.output_dir.exists():
                    config.output_dir.mkdir()
                    (config.output_dir / "racing-sentinel.txt").write_text("race", encoding="utf-8")

            with mock.patch("run_qscci.verify_artifact", return_value={"verified": True}), mock.patch(
                "run_qscci._fsync_directory", side_effect=fsync_with_race
            ):
                with self.assertRaisesRegex(QSCCIError, "publication target appeared"):
                    run_supervised(config, FakeServiceController(), FakeChildLauncher())
            self.assertFalse(config.output_dir.exists())
            quarantines = list(Path(temporary).glob(".result.quarantine-*"))
            self.assertEqual(len(quarantines), 2)
            self.assertTrue(any((path / "racing-sentinel.txt").is_file() for path in quarantines))

    def test_launcher_exception_still_attempts_restoration(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = _config(temporary)
            service = FakeServiceController()
            with self.assertRaisesRegex(OSError, "launch failed"):
                run_supervised(config, service, FakeChildLauncher(launch_error=OSError("launch failed")))
            self.assertEqual(service.events, ["observe", "stop", "restore"])
            self.assertFalse(config.output_dir.exists())

    def test_timeout_terms_then_kills_only_child_group_and_quarantines(self):
        def communicate_effect(call, timeout):
            if call in (1, 2):
                raise subprocess.TimeoutExpired("fixture", timeout)
            return b"late stdout", b"late stderr"

        with tempfile.TemporaryDirectory() as temporary:
            config = _config(temporary)
            service = FakeServiceController()
            process = FakeProcess(communicate_effect=communicate_effect)
            launcher = FakeChildLauncher(process=process)
            with mock.patch("run_qscci._signal_group") as signal_group:
                with self.assertRaisesRegex(QSCCIError, "external ceiling"):
                    run_supervised(config, service, launcher)
            self.assertEqual(signal_group.call_args_list, [mock.call(process, kill=False), mock.call(process, kill=True)])
            self.assertEqual(service.events[-1], "restore")
            self.assertFalse(config.output_dir.exists())

    def test_timeout_after_kill_still_writes_canonical_private_evidence(self):
        def communicate_effect(_call, timeout):
            raise subprocess.TimeoutExpired(
                "fixture", timeout, output=b"partial stdout", stderr=b"partial stderr"
            )

        with tempfile.TemporaryDirectory() as temporary:
            config = _config(temporary)
            process = FakeProcess(communicate_effect=communicate_effect)
            with mock.patch("run_qscci._signal_group") as signal_group:
                with self.assertRaisesRegex(QSCCIError, "external ceiling"):
                    run_supervised(
                        config,
                        FakeServiceController(),
                        FakeChildLauncher(process=process),
                    )
            self.assertEqual(process.communicate_calls, 3)
            self.assertEqual(signal_group.call_args_list, [mock.call(process, kill=False), mock.call(process, kill=True)])
            quarantine = next(Path(temporary).glob(".result.quarantine-*"))
            raw = (quarantine / "INVALID_RUN.json").read_bytes()
            evidence = json.loads(raw.decode("utf-8"))
            self.assertEqual(raw, canonical_json(evidence))
            self.assertEqual(evidence["failure_category"], "CHILD_TIMEOUT")
            self.assertEqual(evidence["stdout_sha256"], sha256_bytes(b"partial stdout"))
            self.assertEqual(evidence["stderr_full_sha256"], sha256_bytes(b"partial stderr"))

    def test_unexpected_child_file_is_quarantined_not_published(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = _config(temporary)
            with self.assertRaisesRegex(QSCCIError, "unexpected file set"):
                run_supervised(config, FakeServiceController(), FakeChildLauncher(extra_file=True))
            self.assertFalse(config.output_dir.exists())
            self.assertEqual(len(list(Path(temporary).glob(".result.quarantine-*"))), 1)

    def test_existing_output_is_refused_without_modification_or_service_touch(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = _config(temporary)
            config.output_dir.mkdir()
            marker = config.output_dir / "owned.txt"
            marker.write_text("preserve", encoding="utf-8")
            service = FakeServiceController()
            with self.assertRaisesRegex(QSCCIError, "overwrite is forbidden"):
                run_supervised(config, service, FakeChildLauncher())
            self.assertEqual(marker.read_text(encoding="utf-8"), "preserve")
            self.assertEqual(service.events, [])

    def test_publication_verifier_rejects_artifact_manifest_receipt_and_extra_file_tamper(self):
        mutations = ("artifact", "manifest", "receipt", "extra")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                config = _config(temporary)
                with mock.patch("run_qscci.verify_artifact", return_value={"verified": True}):
                    run_supervised(config, FakeServiceController(), FakeChildLauncher())
                    if mutation == "extra":
                        (config.output_dir / "extra").write_text("x", encoding="utf-8")
                    else:
                        name = {"artifact": "qscci.json", "manifest": "manifest.json", "receipt": "COMMIT.json"}[mutation]
                        path = config.output_dir / name
                        path.write_bytes(path.read_bytes() + b" ")
                    with self.assertRaises(QSCCIError):
                        verify_result_dir(config.output_dir, config.repo_root)

    def test_public_verifier_rejects_hidden_stage_or_quarantine_ancestry(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for private_parent in (".hidden-parent", "quarantine-launder", "stage-launder"):
                visible = root / private_parent / "visible-result"
                visible.mkdir(parents=True)
                with self.subTest(parent=private_parent), self.assertRaisesRegex(
                    QSCCIError, "ancestry"
                ):
                    verify_result_dir(visible, Path(__file__).resolve().parent)

    def test_cli_refuses_real_run_without_explicit_opt_in_before_child_launch(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "result"
            result = subprocess.run(
                [sys.executable, "run_qscci.py", "run", "--output-dir", str(target)],
                cwd=Path(__file__).resolve().parent,
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("explicit --allow-real-model opt-in", result.stderr)
            self.assertFalse(target.exists())


if __name__ == "__main__":
    unittest.main()
