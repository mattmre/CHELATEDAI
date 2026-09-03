from __future__ import annotations

import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout

from egv.canonical import GENESIS_HASH, canonical_bytes, canonical_json, digest_bytes, digest_for
from egv.cli import main
from egv.campaign.trajectories import GenerationRequest, GenerationResponse, validate_accepted_response
from egv.evaluation.authority import AuthorityPolicy
from egv.evaluation.dataset import EvaluationCorpus
from egv.evaluation.sandbox import DockerCandidateSandbox, DockerSandboxConfig, SandboxResult
from egv.ledger import EvidenceLedger
from egv.receipts import ReceiptSigner, receipt_hash
from egv.variation.errors import VariationConfigurationError, VariationDependencyError
from egv.variation.arms import arm_policy
from egv.variation.generator import CandidateProposal, model_generation_profile_digest
from egv.variation.loop import BoundedCandidateLoop, VARIATION_PROTOCOL_DIGEST, VariationTask
from egv.variation.retrieval import RetrievalResult
from egv.variation.remote import (
    REMOTE_VARIATION_SERVICE_SCHEMA,
    RemoteControllerEvaluationGateway,
    RemoteEvaluatorServiceManifest,
    _decode_b64,
    _bounded_process_exited_without_reap,
    _durable_remote_response,
    _encode_b64,
    _operation_digest,
    build_remote_evaluator_service_manifest,
    run_remote_evaluator_once,
    pinned_python_invocation,
    _run_bounded_command,
    _terminate_bounded_process_tree,
)


RESPONDER = r'''from __future__ import annotations
import json, os, sys
from egv.canonical import GENESIS_HASH, canonical_bytes, canonical_json, content_id, digest_bytes, digest_for
from egv.receipts import ReceiptSigner, receipt_hash

request = json.loads(sys.stdin.read())
capture = os.environ.get("EGV_REMOTE_TEST_CAPTURE")
if capture:
    open(capture, "w", encoding="utf-8").write(canonical_json(request))
signer = ReceiptSigner(bytes.fromhex("__PRIVATE_KEY__"))
common = {
    "campaign_id": request["campaign_id"], "run_id": request["run_id"],
    "task_id": request["task_id"], "candidate_id": request["candidate_id"],
    "candidate_artifact_digest": request["candidate_artifact_digest"],
    "protocol_digest": request["protocol_digest"], "policy_digest": request["policy_digest"],
    "arm_policy_digest": request["arm_policy_digest"],
    "evaluator_digest": request["service_manifest_digest"],
    "task_family": request["public_task_binding"]["family_id"],
    "normalized_public_locus": request["public_task_binding"]["public_locus"],
    "public_rule_id": request["public_task_binding"]["public_rule_id"],
}
sequence = request["receipt_sequence_start"]
previous = request["previous_receipt_hash"]
receipts = []
for kind, fields in (
    ("AUTHORITY", {"request_id": "request-authority-" + request["candidate_id"], "decision": "ALLOW"}),
    ("VERDICT", {"request_id": "request-verdict-" + request["candidate_id"], "decision": "PASS",
        "diagnostic_enum": "PASS", "resource_bucket": "UNDER_25", "exit_status_class": "SUCCESS",
        "input_digest": digest_for("evaluator-private"), "output_digest": digest_bytes(b"ok")}),
    ("EFFECT", {"request_id": "request-effect-" + request["candidate_id"], "decision": "ALLOW",
        "diagnostic_enum": "PASS", "normalized_action_hash": digest_for({"action":"execute_candidate","locus":request["declared_locus"]}),
        "sandbox_id": "sealed-sandbox", "started_at": "2026-08-22T00:00:00Z",
        "finished_at": "2026-08-22T00:00:01Z", "exit_status_class": "SUCCESS",
        "output_digest": digest_bytes(b"ok"), "environment_diff_digest": digest_for({})}),
):
    receipt = signer.sign_receipt({**common, "receipt_type": kind, **fields}, sequence=sequence,
        previous_receipt_hash=previous, idempotency_key=content_id("remote-test", {"request":request["request_digest"],"kind":kind}))
    receipts.append(receipt); previous = receipt_hash(receipt); sequence += 1
result = {
    "candidate_id": request["candidate_id"], "task_id": request["task_id"],
    "candidate_artifact_digest": request["candidate_artifact_digest"], "diagnostic_enum": "PASS",
    "resource_bucket": "UNDER_25", "disposition": "PROMOTED", "infrastructure_loss": False,
    "receipt_ids": [item["receipt_id"] for item in receipts], "output_digest": digest_bytes(b"ok"),
}
unsigned = {"schema_version":"egv-remote-variation-response-v1", "operation_digest":request["operation_digest"],
    "request_digest":request["request_digest"],
    "service_manifest_digest":request["service_manifest_digest"], "result":result, "receipts":receipts,
    "signing_key_id":signer.key_id}
mode = os.environ.get("EGV_REMOTE_TEST_MODE")
if mode == "stale-request": unsigned["request_digest"] = digest_for("stale")
response = {**unsigned, "signature":signer.sign_bytes(canonical_bytes(unsigned))}
if mode == "tamper-result": response["result"]["candidate_artifact_digest"] = digest_for("substitute")
print(canonical_json(response))
'''


class RemoteEvaluatorCLITests(unittest.TestCase):
    def test_remote_response_signature_requires_canonical_ed25519_base64url(self) -> None:
        encoded = ReceiptSigner(b"R" * 32).sign_bytes(b"remote-envelope")
        self.assertEqual(len(_decode_b64(encoded, "response signature", exact_length=64)), 64)
        tail_alias = {"A": "B", "Q": "R", "g": "h", "w": "x"}
        with self.assertRaisesRegex(VariationConfigurationError, "canonical"):
            _decode_b64(encoded[:-1] + tail_alias[encoded[-1]], "response signature", exact_length=64)

    def test_evaluator_once_dispatches_without_nonexistent_adapter_store_argument(self) -> None:
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as temporary, patch(
            "egv.cli.run_remote_evaluator_once", return_value={"status": "PASS"}
        ) as remote, patch("sys.stdin", io.StringIO("{}")), redirect_stdout(output):
            root = Path(temporary)
            code = main([
                "variation", "evaluator-once",
                "--service-manifest", str(root / "service.json"),
                "--evaluator-seed", str(root / "seed.bin"),
                "--private-key", str(root / "key.bin"),
                "--workspace", str(root / "workspace"),
                "--state-root", str(root / "state"),
            ])
        self.assertEqual(code, 0)
        self.assertEqual(json.loads(output.getvalue()), {"status": "PASS"})
        self.assertNotIn("adapter_store", remote.call_args.kwargs)

    def test_freeze_service_command_option_does_not_overwrite_top_level_dispatch(self) -> None:
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            seed = root / "seed.bin"
            key = root / "evaluator.pub"
            command = root / "endpoint.py"
            destination = root / "service.json"
            seed.write_bytes(b"S" * 32)
            key.write_bytes(b"K" * 32)
            command.write_text("print('endpoint')\n", encoding="utf-8")
            frozen = {"service_manifest_digest": digest_for("service"), "task_bindings": []}
            with patch("egv.cli.build_remote_evaluator_service_manifest", return_value=frozen) as build, patch(
                "egv.evaluation.sandbox.DockerSandboxConfig.from_environment", return_value=object()
            ), redirect_stdout(output):
                code = main([
                    "variation", "freeze-evaluator-service",
                    "--campaign-id", "campaign-cli",
                    "--model-digest", digest_for("model"),
                    "--evaluator-revision", "revision-cli",
                    "--evaluator-seed", str(seed),
                    "--public-key", str(key),
                    "--command", str(command),
                    "--output", str(destination),
                ])
            frozen_on_disk = json.loads(destination.read_text(encoding="utf-8"))
        self.assertEqual(code, 0)
        self.assertEqual(build.call_args.kwargs["command"], command)
        self.assertEqual(frozen_on_disk, frozen)


class RemoteVariationGatewayTests(unittest.TestCase):

    @unittest.skipUnless(os.name == "nt", "Windows unassigned suspended-child cleanup canary")
    def test_windows_job_assignment_failure_directly_kills_and_reaps_suspended_child(self) -> None:
        started = []
        real_popen = subprocess.Popen

        def capture_start(*args, **kwargs):
            process = real_popen(*args, **kwargs)
            started.append(process)
            return process

        with patch("egv.variation.remote.subprocess.Popen", side_effect=capture_start), patch(
            "egv.variation.remote._assign_process_to_windows_job",
            side_effect=OSError("fault-injected assignment failure"),
        ):
            with self.assertRaisesRegex(VariationDependencyError, "invocation failed"):
                _run_bounded_command([sys.executable, "-c", "import time; time.sleep(60)"], "{}")
        self.assertEqual(len(started), 1)
        self.assertIsNotNone(started[0].poll())

    @unittest.skipUnless(os.name == "nt", "Windows failed-start cleanup propagation canary")
    def test_windows_unassigned_cleanup_failures_remain_attached_to_start_error(self) -> None:
        process = unittest.mock.Mock(pid=12345)
        process._handle = 12345
        process.stdin = unittest.mock.Mock()
        process.stdout = unittest.mock.Mock()
        process.stderr = unittest.mock.Mock()
        process.kill.side_effect = OSError("fault-injected direct kill failure")
        process.wait.side_effect = subprocess.TimeoutExpired(["remote"], 5)
        with patch(
            "egv.variation.remote.subprocess.Popen",
            return_value=process,
        ), patch(
            "egv.variation.remote._assign_process_to_windows_job",
            side_effect=OSError("fault-injected assignment failure"),
        ):
            with self.assertRaisesRegex(VariationDependencyError, "invocation failed") as raised:
                _run_bounded_command([sys.executable, "-c", "pass"], "{}")
        startup_error = raised.exception.__cause__
        self.assertIsInstance(startup_error, OSError)
        self.assertEqual(
            getattr(startup_error, "cleanup_context", ()),
            (
                "fault-injected direct kill failure",
                "Command '['remote']' timed out after 5 seconds",
            ),
        )
        process.kill.assert_called_once()
        process.wait.assert_called_once()

    @unittest.skipUnless(os.name == "nt", "Windows owned-resource cleanup fault matrix")
    def test_windows_failed_start_cleanup_reports_wait_stream_and_handle_failures(self) -> None:
        from egv.variation import remote as remote_module

        for wait_error in (
            subprocess.TimeoutExpired(["remote"], 5),
            OSError("fault-injected wait failure"),
        ):
            with self.subTest(wait_error=type(wait_error).__name__):
                process = unittest.mock.Mock(pid=12345)
                process.stdin = unittest.mock.Mock()
                process.stdout = unittest.mock.Mock()
                process.stderr = unittest.mock.Mock()
                process.wait.side_effect = wait_error
                process.stderr.close.side_effect = OSError(
                    "fault-injected stderr close failure"
                )
                close_handle = unittest.mock.Mock(return_value=False)
                errors = remote_module._cleanup_failed_windows_start(
                    process,
                    9876,
                    job_assigned=False,
                    close_handle=close_handle,
                    deadline=time.monotonic() + 1,
                )
                self.assertIn(wait_error, errors)
                self.assertTrue(
                    any("stderr close failure" in str(error) for error in errors)
                )
                self.assertTrue(
                    any("Job Object close failed" in str(error) for error in errors)
                )
                process.kill.assert_called_once()
                process.wait.assert_called_once()
                close_handle.assert_called_once_with(9876)

    @unittest.skipUnless(os.name == "nt", "Windows assigned-tree cleanup fault canary")
    def test_windows_failed_start_propagates_assigned_tree_termination_failure(self) -> None:
        from egv.variation import remote as remote_module

        process = unittest.mock.Mock(pid=12345)
        process.stdin = unittest.mock.Mock()
        process.stdout = unittest.mock.Mock()
        process.stderr = unittest.mock.Mock()
        close_handle = unittest.mock.Mock(return_value=True)
        tree_error = VariationDependencyError("fault-injected process-tree cleanup failure")
        with patch(
            "egv.variation.remote._terminate_bounded_process_tree",
            side_effect=tree_error,
        ) as terminate_tree:
            errors = remote_module._cleanup_failed_windows_start(
                process,
                9876,
                job_assigned=True,
                close_handle=close_handle,
                deadline=time.monotonic() + 1,
            )
        self.assertIn(tree_error, errors)
        terminate_tree.assert_called_once()
        process.kill.assert_not_called()
        process.wait.assert_not_called()
        close_handle.assert_called_once_with(9876)

    def test_late_native_startup_fails_as_timeout_before_request_io(self) -> None:
        from egv.variation import remote as remote_module

        real_start = remote_module._start_bounded_process

        def delayed_start(*args, **kwargs):
            process = real_start(*args, **kwargs)
            time.sleep(0.1)
            return process

        with patch("egv.variation.remote.REMOTE_VARIATION_TIMEOUT_SECONDS", 0.05), patch(
            "egv.variation.remote._start_bounded_process", side_effect=delayed_start
        ):
            with self.assertRaisesRegex(VariationDependencyError, "timed out"):
                _run_bounded_command([sys.executable, "-c", "import time; time.sleep(60)"], "{}")

    @unittest.skipIf(os.name == "nt", "POSIX waitid ordering canary")
    def test_posix_observes_exit_without_reap_then_kills_group_before_wait(self) -> None:
        process = unittest.mock.Mock(pid=12345)
        process.poll.side_effect = AssertionError("poll must not reap the POSIX leader")
        with patch("egv.variation.remote.os.waitid", return_value=object()) as waitid:
            self.assertTrue(_bounded_process_exited_without_reap(process))
        waitid.assert_called_once()
        order = []
        process.wait.side_effect = lambda timeout: order.append(("wait", timeout))
        with patch(
            "egv.variation.remote.os.waitid",
            side_effect=lambda *args: order.append(("waitid", args[1])) or object(),
        ), patch(
            "egv.variation.remote.os.killpg",
            side_effect=lambda pid, sig: order.append(("killpg", pid)),
        ):
            _terminate_bounded_process_tree(process, deadline=time.monotonic() + 1)
        self.assertEqual([item[0] for item in order], ["waitid", "killpg", "wait"])

    @unittest.skipIf(os.name == "nt", "POSIX lost-anchor cleanup canary")
    def test_posix_lost_leader_anchor_never_signals_recycled_process_group(self) -> None:
        process = unittest.mock.Mock(pid=12345)
        process.wait.return_value = 0
        with patch(
            "egv.variation.remote.os.waitid",
            side_effect=ChildProcessError("fault-injected lost leader"),
        ), patch("egv.variation.remote.os.killpg") as killpg:
            with self.assertRaisesRegex(VariationDependencyError, "cleanup failed"):
                _terminate_bounded_process_tree(process, deadline=time.monotonic() + 1)
        killpg.assert_not_called()
        process.wait.assert_called_once()

    def test_process_tree_cleanup_retains_simultaneous_termination_and_wait_failures(self) -> None:
        process = unittest.mock.Mock(pid=12345)
        process.wait.side_effect = OSError("fault-injected cleanup wait failure")
        if os.name == "nt":
            process.poll.return_value = None
            process.kill.side_effect = OSError(
                "fault-injected cleanup termination failure"
            )
            with self.assertRaisesRegex(
                VariationDependencyError,
                "process-tree cleanup failed",
            ) as raised:
                _terminate_bounded_process_tree(
                    process,
                    deadline=time.monotonic() + 1,
                )
        else:
            with patch(
                "egv.variation.remote.os.waitid",
                return_value=object(),
            ), patch(
                "egv.variation.remote.os.killpg",
                side_effect=OSError(
                    "fault-injected cleanup termination failure"
                ),
            ):
                with self.assertRaisesRegex(
                    VariationDependencyError,
                    "process-tree cleanup failed",
                ) as raised:
                    _terminate_bounded_process_tree(
                        process,
                        deadline=time.monotonic() + 1,
                    )
        self.assertEqual(
            getattr(raised.exception, "cleanup_context", ()),
            (
                "fault-injected cleanup termination failure",
                "fault-injected cleanup wait failure",
            ),
        )
        process.wait.assert_called_once()

    def test_late_startup_failure_remains_primary_with_flattened_cleanup_evidence(self) -> None:
        startup_error = OSError("fault-injected native startup failure")
        setattr(
            startup_error,
            "cleanup_context",
            (
                "fault-injected termination cleanup failure",
                "fault-injected wait cleanup failure",
            ),
        )
        with patch(
            "egv.variation.remote.REMOTE_VARIATION_TIMEOUT_SECONDS",
            0.05,
        ), patch(
            "egv.variation.remote.time.monotonic",
            side_effect=(0.0, 1.0),
        ), patch(
            "egv.variation.remote._start_bounded_process",
            side_effect=startup_error,
        ):
            with self.assertRaisesRegex(
                VariationDependencyError,
                "timed out",
            ) as raised:
                _run_bounded_command([sys.executable, "-c", "pass"], "{}")
        self.assertIs(raised.exception.__cause__, startup_error)
        self.assertEqual(
            getattr(raised.exception, "cleanup_context", ()),
            (
                "fault-injected native startup failure",
                "fault-injected termination cleanup failure",
                "fault-injected wait cleanup failure",
            ),
        )

    @unittest.skipUnless(os.name == "nt", "Windows owned-handle cleanup canary")
    def test_owned_windows_handle_preserves_close_failure_on_primary_error(self) -> None:
        from egv.variation import remote as remote_module

        close_handle = unittest.mock.Mock(return_value=False)
        primary = OSError("fault-injected primary startup failure")
        with self.assertRaises(OSError) as raised:
            with remote_module._owned_windows_handle(
                close_handle,
                12345,
                "evaluator primary thread",
            ):
                raise primary
        self.assertIs(raised.exception, primary)
        self.assertTrue(
            any(
                "primary thread close failed" in item
                for item in getattr(primary, "cleanup_context", ())
            )
        )

    @unittest.skipUnless(os.name == "nt", "Windows pinned-identity raw-handle canary")
    def test_pinned_python_raw_handle_close_failure_remains_attached(self) -> None:
        kernel32 = unittest.mock.Mock()
        kernel32.CreateFileW.return_value = 12345
        kernel32.CloseHandle.return_value = False
        conversion_error = OSError("fault-injected descriptor conversion failure")
        executable = Path(sys.executable).resolve()
        executable_digest = hashlib.sha256(executable.read_bytes()).hexdigest()
        with patch("ctypes.WinDLL", return_value=kernel32), patch(
            "msvcrt.open_osfhandle",
            side_effect=conversion_error,
        ):
            with self.assertRaisesRegex(
                VariationDependencyError,
                "identity is unavailable",
            ) as raised:
                with pinned_python_invocation(executable, executable_digest):
                    self.fail("pinned identity unexpectedly yielded")
        self.assertIs(raised.exception.__cause__, conversion_error)
        self.assertTrue(
            any(
                "pinned Python executable close failed" in item
                for item in getattr(raised.exception, "cleanup_context", ())
            )
        )
        kernel32.CloseHandle.assert_called_once_with(12345)

    def test_primary_overflow_preserves_cleanup_context_without_exception_notes(self) -> None:
        from egv.variation import remote as remote_module

        real_cleanup = remote_module._terminate_bounded_process_tree

        def cleanup_then_report(*args, **kwargs):
            real_cleanup(*args, **kwargs)
            raise VariationDependencyError("fault-injected secondary cleanup error")

        command = [sys.executable, "-c", "import sys; sys.stdout.write('X'*4096); sys.stdout.flush()"]
        with patch("egv.variation.remote.REMOTE_VARIATION_RESPONSE_LIMIT", 128), patch(
            "egv.variation.remote._terminate_bounded_process_tree", side_effect=cleanup_then_report
        ):
            with self.assertRaisesRegex(VariationDependencyError, "stdout exceeded") as raised:
                _run_bounded_command(command, "{}")
        self.assertEqual(
            getattr(raised.exception, "cleanup_context", ()),
            ("fault-injected secondary cleanup error",),
        )

    def test_cleanup_only_failure_preserves_every_secondary_cleanup_error(self) -> None:
        process = unittest.mock.Mock(pid=12345)
        process.returncode = 0
        process.stdin = unittest.mock.Mock()
        process.stdout = unittest.mock.Mock()
        process.stderr = unittest.mock.Mock()
        process.stdin.write.side_effect = lambda value: len(value)
        process.stdout.read.return_value = b""
        process.stderr.read.return_value = b""
        process.stdout.close.side_effect = OSError(
            "fault-injected stdout close failure"
        )
        process.stderr.close.side_effect = OSError(
            "fault-injected stderr close failure"
        )
        tree_error = VariationDependencyError(
            "fault-injected process-tree cleanup wrapper"
        )
        setattr(
            tree_error,
            "cleanup_context",
            (
                "fault-injected tree termination failure",
                "fault-injected tree wait failure",
            ),
        )

        with patch(
            "egv.variation.remote._start_bounded_process",
            return_value=(process, None),
        ), patch(
            "egv.variation.remote._bounded_process_exited_without_reap",
            return_value=True,
        ), patch(
            "egv.variation.remote._terminate_bounded_process_tree",
            side_effect=tree_error,
        ):
            with self.assertRaisesRegex(
                VariationDependencyError,
                "process-tree cleanup failed",
            ) as raised:
                _run_bounded_command(["fault-injected-remote"], "{}")

        self.assertIs(raised.exception.__cause__, tree_error)
        self.assertEqual(
            getattr(raised.exception, "cleanup_context", ()),
            (
                "fault-injected process-tree cleanup wrapper",
                "fault-injected tree termination failure",
                "fault-injected tree wait failure",
                "fault-injected stdout close failure",
                "fault-injected stderr close failure",
            ),
        )
        process.stdin.close.assert_called_once()
        process.stdout.close.assert_called_once()
        process.stderr.close.assert_called_once()

    @unittest.skipUnless(os.name == "nt", "Windows Job Object containment canary")
    def test_bounded_command_windows_job_contains_remote_descendant(self) -> None:
        import ctypes
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        open_process = kernel32.OpenProcess
        open_process.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
        open_process.restype = wintypes.HANDLE
        wait_for_single = kernel32.WaitForSingleObject
        wait_for_single.argtypes = (wintypes.HANDLE, wintypes.DWORD)
        wait_for_single.restype = wintypes.DWORD
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = (wintypes.HANDLE,)
        close_handle.restype = wintypes.BOOL

        with tempfile.TemporaryDirectory() as temporary:
            pid_file = Path(temporary) / "windows-descendant.pid"
            child = "import time; time.sleep(60)"
            script = (
                "import pathlib,subprocess,sys\n"
                "p=subprocess.Popen([sys.executable,'-c',%r])\n" % child
                + "pathlib.Path(%r).write_text(str(p.pid),encoding='ascii')\n" % str(pid_file)
                + "sys.stdout.write('{}'); sys.stdout.flush()\n"
            )
            code, stdout, _stderr = _run_bounded_command([sys.executable, "-c", script], "{}")
            self.assertEqual(code, 0)
            self.assertEqual(stdout, b"{}")
            self.assertTrue(pid_file.exists())
            pid = int(pid_file.read_text(encoding="ascii"))
            handle = open_process(0x00100000, False, pid)  # SYNCHRONIZE
            if handle:
                try:
                    self.assertEqual(wait_for_single(handle, 3000), 0)  # WAIT_OBJECT_0
                finally:
                    close_handle(handle)
            else:
                self.assertEqual(ctypes.get_last_error(), 87)  # ERROR_INVALID_PARAMETER: PID is gone

    def test_bounded_command_times_out_when_child_never_reads_maximum_request(self) -> None:
        command = [sys.executable, "-c", "import time; time.sleep(60)"]
        request = "X" * (512 * 1024)
        started = time.monotonic()
        with patch("egv.variation.remote.REMOTE_VARIATION_TIMEOUT_SECONDS", 0.2):
            with self.assertRaisesRegex(VariationDependencyError, "timed out"):
                _run_bounded_command(command, request)
        self.assertLess(time.monotonic() - started, 6.0)

    @unittest.skipIf(os.name == "nt", "POSIX process-group containment canary")
    def test_bounded_command_contains_descendants_on_timeout_overflow_and_success(self) -> None:
        def wait_gone(pid_file: Path) -> None:
            deadline = time.monotonic() + 3
            while not pid_file.exists() and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(pid_file.exists())
            pid = int(pid_file.read_text(encoding="ascii"))
            while time.monotonic() < deadline:
                try:
                    os.kill(pid, 0)
                except ProcessLookupError:
                    return
                time.sleep(0.02)
            self.fail("descendant process survived containment cleanup")

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for mode in ("timeout", "overflow", "success"):
                with self.subTest(mode=mode):
                    pid_file = root / (mode + ".pid")
                    child = "import time; time.sleep(60)"
                    tail = {
                        "timeout": "time.sleep(60)",
                        "overflow": "sys.stdout.write('X'*4096); sys.stdout.flush()",
                        "success": "sys.stdout.write('{}'); sys.stdout.flush()",
                    }[mode]
                    script = (
                        "import pathlib,subprocess,sys,time\n"
                        "p=subprocess.Popen([sys.executable,'-c',%r])\n" % child
                        + "pathlib.Path(%r).write_text(str(p.pid),encoding='ascii')\n" % str(pid_file)
                        + tail
                        + "\n"
                    )
                    command = [sys.executable, "-c", script]
                    timeout_value = 0.2 if mode == "timeout" else 5
                    response_limit = 128 if mode == "overflow" else 1024 * 1024
                    with patch("egv.variation.remote.REMOTE_VARIATION_TIMEOUT_SECONDS", timeout_value), patch(
                        "egv.variation.remote.REMOTE_VARIATION_RESPONSE_LIMIT", response_limit
                    ):
                        if mode == "success":
                            code, stdout, _stderr = _run_bounded_command(command, "{}")
                            self.assertEqual(code, 0)
                            self.assertEqual(stdout, b"{}")
                        else:
                            with self.assertRaises(VariationDependencyError):
                                _run_bounded_command(command, "{}")
                    wait_gone(pid_file)

    @unittest.skipIf(os.name == "nt", "sealed memfd execution is POSIX-specific")
    def test_pinned_python_uses_sealed_immutable_copy_and_launches_it(self) -> None:
        import fcntl

        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "python-copy"
            shutil.copy2(Path(sys.executable).resolve(), source)
            expected_bytes = source.read_bytes()
            expected_digest = hashlib.sha256(expected_bytes).hexdigest()
            with pinned_python_invocation(source.resolve(), expected_digest) as (executable, kwargs):
                fd = kwargs["pass_fds"][0]
                self.assertFalse(os.get_inheritable(fd))
                source.write_bytes(b"X" * len(expected_bytes))
                os.lseek(fd, 0, os.SEEK_SET)
                self.assertEqual(hashlib.sha256(os.read(fd, len(expected_bytes))).hexdigest(), expected_digest)
                with self.assertRaises(OSError):
                    os.write(fd, b"mutation")
                seals = fcntl.fcntl(fd, getattr(fcntl, "F_GET_SEALS", 1034))
                required = 0x0001 | 0x0002 | 0x0004 | 0x0008
                self.assertEqual(seals & required, required)
                completed = subprocess.run(
                    [executable, "-c", "print('sealed-launch-ok')"],
                    pass_fds=(fd,), capture_output=True, text=True, check=True,
                )
                self.assertEqual(completed.stdout.strip(), "sealed-launch-ok")

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="egv-remote-variation-")
        self.root = Path(self.temporary.name)
        self._old_pythonpath = os.environ.get("PYTHONPATH")
        repository_root = str(Path(__file__).resolve().parents[1])
        os.environ["PYTHONPATH"] = repository_root + (os.pathsep + self._old_pythonpath if self._old_pythonpath else "")
        self.signer = ReceiptSigner(b"V" * 32)
        self.key_path = self.root / "evaluator.pub"
        self.key_path.write_bytes(self.signer.public_key_raw)
        self.command = self.root / "remote-endpoint.py"
        self.command.write_text(RESPONDER.replace("__PRIVATE_KEY__", self.signer.private_key_raw.hex()), encoding="utf-8")
        self.task = {
            "template_id": "task-heldout-001",
            "family_id": "family-test",
            "split": "train",
            "ordinal": 1,
            "source_digest": digest_for("source"),
            "public_rule_id": "rule-test",
            "public_locus": "module:solve",
        }
        policy_digest = AuthorityPolicy.candidate_execution().digest
        unsigned = {
            "schema_version": REMOTE_VARIATION_SERVICE_SCHEMA,
            "campaign_id": "campaign-remote-test",
            "model_digest": digest_for("model"),
            "protocol_digest": VARIATION_PROTOCOL_DIGEST,
            "policy_digest": policy_digest,
            "data_manifest_digest": digest_for("data"),
            "task_manifest_digest": digest_for([self.task]),
            "task_bindings": [self.task],
            "evaluator_revision": "remote-evaluator-test-v1",
            "evaluator_digest": digest_for("remote-evaluator-test-v1"),
            "docker_image_digest": DockerSandboxConfig().pinned_image_id,
            "docker_config_digest": digest_for(dict(DockerSandboxConfig().__dict__)),
            "authority_policy_digest": policy_digest,
            "command_digest": hashlib.sha256(self.command.read_bytes()).hexdigest(),
            "evaluator_key_id": self.signer.key_id,
            "evaluator_public_key_digest": hashlib.sha256(self.key_path.read_bytes()).hexdigest(),
        }
        self.manifest_value = {**unsigned, "service_manifest_digest": digest_for(unsigned)}
        self.manifest_path = self.root / "service-manifest.json"
        self.manifest_path.write_text(canonical_json(self.manifest_value) + "\n", encoding="utf-8")
        self.ledger = EvidenceLedger(self.root / "ledger.sqlite", clock=lambda: "2026-08-22T00:00:00Z")

    def tearDown(self) -> None:
        self.ledger.close()
        os.environ.pop("EGV_REMOTE_TEST_CAPTURE", None)
        os.environ.pop("EGV_REMOTE_TEST_MODE", None)
        if self._old_pythonpath is None:
            os.environ.pop("PYTHONPATH", None)
        else:
            os.environ["PYTHONPATH"] = self._old_pythonpath
        self.temporary.cleanup()

    def gateway(self) -> RemoteControllerEvaluationGateway:
        return RemoteControllerEvaluationGateway(
            ledger=self.ledger,
            manifest_path=self.manifest_path,
            public_key_path=self.key_path,
            command=self.command,
        )

    def register_commissioning_candidate(
        self, request: GenerationRequest, candidate_id: str, source: bytes
    ) -> None:
        self.ledger.create_campaign(
            request.campaign_id,
            protocol_hash=request.variation_protocol_digest,
            source_commit="commissioning-test",
            model_revision="test-model",
            data_manifest_hash=request.corpus_manifest_digest,
            evaluator_hash=self.manifest_value["service_manifest_digest"],
            policy_hash=AuthorityPolicy.candidate_execution().digest,
            seed_set=(0, 1),
        )
        self.ledger.create_run(
            request.run_id,
            campaign_id=request.campaign_id,
            arm=request.arm_id,
            task_id=request.task_id,
            seed=request.seed,
            parent_checkpoint=None,
            start_state="READY",
            host_role="spark_trainer",
            software_manifest_hash=digest_for("test-software"),
        )
        self.ledger.append_candidate(
            candidate_id,
            campaign_id=request.campaign_id,
            run_id=request.run_id,
            task_id=request.task_id,
            parent_candidate_id=None,
            mutation_family=request.task_family,
            patch_hash=digest_for(source.decode("utf-8")),
            requested_authority="EXECUTE_CANDIDATE",
            prompt_hash=digest_for("prompt"),
            model_hash=request.model_manifest_digest,
            metadata={"arm_id": request.arm_id, "candidate_artifact_digest": digest_bytes(source)},
        )

    def durable_request(self, label: str) -> dict:
        body = {
            "operation_digest": digest_for({"operation": label}),
            "receipt_sequence_start": 1,
            "previous_receipt_hash": GENESIS_HASH,
        }
        return {**body, "request_digest": digest_for(body)}

    def durable_builder(self, request: dict):
        def build() -> dict:
            receipt = self.signer.sign_receipt(
                {
                    "receipt_type": "AUTHORITY",
                    "campaign_id": "campaign-remote-test",
                    "run_id": "run-evaluation",
                    "task_id": self.task["template_id"],
                    "request_id": "request-authority-durable",
                    "candidate_id": "candidate-durable",
                    "decision": "DENY",
                },
                sequence=request["receipt_sequence_start"],
                previous_receipt_hash=request["previous_receipt_hash"],
                idempotency_key="durable:" + request["operation_digest"],
            )
            unsigned = {
                "schema_version": "egv-remote-variation-response-v1",
                "operation_digest": request["operation_digest"],
                "request_digest": request["request_digest"],
                "service_manifest_digest": self.manifest_value["service_manifest_digest"],
                "result": {},
                "receipts": [receipt],
                "signing_key_id": self.signer.key_id,
            }
            return {**unsigned, "signature": self.signer.sign_bytes(canonical_bytes(unsigned))}

        return build

    def test_remote_gateway_ingests_verified_chain_without_private_input_or_path(self) -> None:
        capture = self.root / "request.json"
        os.environ["EGV_REMOTE_TEST_CAPTURE"] = str(capture)
        result = self.gateway().evaluate(
            candidate_id="candidate-001",
            task_id=self.task["template_id"],
            source=b"def solve(value):\n    return value\n",
            opaque_input=None,
            requested_authority="EXECUTE_CANDIDATE",
            declared_locus=self.task["public_locus"],
            candidate_source_path=str(self.root / "private" / "candidate.py"),
        )
        self.assertEqual(result.disposition, "PROMOTED")
        self.assertEqual(len(self.ledger.receipts()), 3)
        replay = self.ledger.receipts()
        with self.assertRaises(Exception):
            self.ledger.ingest_receipts_atomic(replay, self.signer.public_key)
        self.assertEqual(self.ledger.receipts(), replay)
        request = json.loads(capture.read_text(encoding="utf-8"))
        serialized = canonical_json(request)
        self.assertNotIn("never", serialized)
        self.assertNotIn("send-this-private-value", serialized)
        self.assertNotIn("candidate.py", serialized)
        self.assertEqual(set(request), {
            "schema_version", "operation_digest", "request_digest", "service_manifest_digest", "campaign_id", "model_digest",
            "protocol_digest", "policy_digest", "data_manifest_digest", "task_manifest_digest",
            "run_id", "arm_policy_digest",
            "evaluator_digest", "docker_image_digest", "candidate_id", "task_id", "public_task_binding",
            "candidate_artifact_digest", "candidate_source_b64", "requested_authority", "declared_locus",
            "receipt_sequence_start", "previous_receipt_hash",
        })

    def test_remote_promotions_bind_commissioning_run_arm_policy_and_materialized_disposition(self) -> None:
        for index, arm_id in enumerate(("B", "D")):
            with self.subTest(arm_id=arm_id):
                request = GenerationRequest.build(
                    campaign_id=self.manifest_value["campaign_id"],
                    task_record=self.task,
                    corpus_manifest_digest=self.manifest_value["data_manifest_digest"],
                    arm_id=arm_id,
                    seed=index,
                    model_manifest_digest=self.manifest_value["model_digest"],
                    variation_protocol_digest=self.manifest_value["protocol_digest"],
                    generation_profile_digest=model_generation_profile_digest(
                        "source-only-v1",
                        model_manifest_digest=self.manifest_value["model_digest"],
                        chat_template_digest=digest_for("remote-test-chat-template"),
                    ),
                )
                source = "def solve(value):\n    return value + {}\n".format(index).encode("utf-8")
                candidate_id = "candidate-commissioning-" + arm_id.lower()
                self.register_commissioning_candidate(request, candidate_id, source)
                result = self.gateway().evaluate(
                    candidate_id=candidate_id,
                    task_id=request.task_id,
                    source=source,
                    opaque_input=None,
                    requested_authority="EXECUTE_CANDIDATE",
                    declared_locus=self.task["public_locus"],
                )
                receipts = [self.ledger.receipt_by_id(item)["receipt"] for item in result.receipt_ids]
                payload = {
                    "schema_version": "egv-commissioning-generation-response-v1",
                    "request_id": request.request_id,
                    "candidate_id": candidate_id,
                    "candidate_artifact_digest": result.candidate_artifact_digest,
                    "model_output_digest": result.candidate_artifact_digest,
                    "output_byte_count": len(source),
                    "disposition": "PROMOTED",
                    "receipts": receipts,
                }
                from egv.canonical import content_id

                payload["response_id"] = content_id("genresp", payload)
                accepted = validate_accepted_response(
                    request,
                    GenerationResponse.from_mapping(payload),
                    evaluator_public_key=self.signer.public_key,
                    evaluator_digest=self.manifest_value["service_manifest_digest"],
                    expected_first_sequence=receipts[0]["sequence"],
                    expected_previous_receipt_hash=receipts[0]["previous_receipt_hash"],
                )
                self.assertEqual(accepted["arm_id"], arm_id)
                self.assertTrue(all(item["run_id"] == request.run_id for item in receipts))
                self.assertTrue(all(item["arm_policy_digest"] == arm_policy(arm_id).digest for item in receipts))
                self.assertTrue(all(item["policy_digest"] == AuthorityPolicy.candidate_execution().digest for item in receipts))

                # The remote gateway owns authenticated receipt admission. The
                # bounded loop owns the verdict/effect projections used for a
                # durable candidate disposition. Exercise that production
                # handoff explicitly so a valid remote PASS chain cannot remain
                # ABSTAINED merely because the gateway itself is projection-free.
                runner = object.__new__(BoundedCandidateLoop)
                runner.ledger = self.ledger
                runner.evaluator = self.gateway()
                runner.policy = arm_policy(arm_id)
                runner.campaign_id = request.campaign_id
                task = VariationTask.from_public_record(self.task)
                proposal = CandidateProposal(
                    source=source,
                    declared_locus=task.public_locus,
                    requested_authority="EXECUTE_CANDIDATE",
                    evidence_ids=(),
                    mutation_digest=digest_for({"candidate": candidate_id}),
                    metadata={"arm_id": arm_id},
                )
                retrieval = RetrievalResult(
                    policy=runner.policy.retrieval_policy,
                    arm_id=arm_id,
                    task_id=task.task_id,
                    records=(),
                    evidence_digest=digest_for([]),
                )
                attempt = runner._materialize_result(
                    task=task,
                    run_id=request.run_id,
                    candidate_id=candidate_id,
                    proposal=proposal,
                    retrieval=retrieval,
                    result=result,
                    attempt_index=1,
                )
                self.assertEqual(attempt.disposition, "PROMOTED")
                self.assertEqual(self.ledger.candidate_disposition(candidate_id), "PROMOTED")

    def test_response_tamper_and_stale_binding_fail_before_ingestion(self) -> None:
        for mode in ("stale-request", "tamper-result"):
            with self.subTest(mode=mode):
                os.environ["EGV_REMOTE_TEST_MODE"] = mode
                with self.assertRaises(VariationConfigurationError):
                    self.gateway().evaluate(
                        candidate_id="candidate-" + mode,
                        task_id=self.task["template_id"],
                        source=b"def solve(value):\n    return value\n",
                        opaque_input=None,
                        requested_authority="EXECUTE_CANDIDATE",
                        declared_locus=self.task["public_locus"],
                    )
                self.assertEqual(self.ledger.receipts(), [])

    def test_command_tamper_and_manifest_mutation_fail_closed(self) -> None:
        gateway = self.gateway()
        self.command.write_text(self.command.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        with self.assertRaises(VariationDependencyError):
            gateway.validate_runtime()
        manifest = RemoteEvaluatorServiceManifest(self.manifest_value)
        manifest._value["campaign_id"] = "stale"  # adversarial mutation of an internal container
        with self.assertRaises(VariationDependencyError):
            manifest.validate_integrity()

    def test_wrong_key_closed_manifest_and_timeout_fail_closed(self) -> None:
        wrong_key = self.root / "wrong.pub"
        wrong_key.write_bytes(ReceiptSigner(b"W" * 32).public_key_raw)
        with self.assertRaises(VariationConfigurationError):
            RemoteControllerEvaluationGateway(
                ledger=self.ledger,
                manifest_path=self.manifest_path,
                public_key_path=wrong_key,
                command=self.command,
            )
        malformed = dict(self.manifest_value)
        malformed["unexpected"] = "field"
        with self.assertRaises(VariationConfigurationError):
            RemoteEvaluatorServiceManifest(malformed)
        gateway = self.gateway()
        with patch("egv.variation.remote._run_bounded_command", side_effect=VariationDependencyError("timeout")):
            with self.assertRaises(VariationDependencyError):
                gateway.evaluate(
                    candidate_id="candidate-timeout",
                    task_id=self.task["template_id"],
                    source=b"def solve(value):\n    return value\n",
                    opaque_input=None,
                    requested_authority="EXECUTE_CANDIDATE",
                    declared_locus=self.task["public_locus"],
                )
        self.assertEqual(self.ledger.receipts(), [])

    def test_public_task_registry_accepts_train_and_heldout_but_rejects_dev(self) -> None:
        for split in ("train", "heldout"):
            with self.subTest(split=split):
                task = {**self.task, "split": split}
                unsigned = {
                    key: value
                    for key, value in self.manifest_value.items()
                    if key != "service_manifest_digest"
                }
                unsigned["task_bindings"] = [task]
                unsigned["task_manifest_digest"] = digest_for([task])
                manifest = RemoteEvaluatorServiceManifest(
                    {**unsigned, "service_manifest_digest": digest_for(unsigned)}
                )
                self.assertEqual(manifest.public_record(task["template_id"]), task)
        task = {**self.task, "split": "dev"}
        unsigned = {
            key: value
            for key, value in self.manifest_value.items()
            if key != "service_manifest_digest"
        }
        unsigned["task_bindings"] = [task]
        unsigned["task_manifest_digest"] = digest_for([task])
        with self.assertRaises(VariationConfigurationError):
            RemoteEvaluatorServiceManifest({**unsigned, "service_manifest_digest": digest_for(unsigned)})

    def test_service_freeze_omits_private_seed_path_and_dev_records(self) -> None:
        seed = self.root / "evaluator-private-seed.bin"
        seed.write_bytes(b"S" * 32)
        corpus = EvaluationCorpus.generate(secret_seed_file=seed)
        config = DockerSandboxConfig()
        with patch.object(DockerSandboxConfig, "verify_image", return_value=config.pinned_image_id):
            manifest = build_remote_evaluator_service_manifest(
                campaign_id="campaign-remote-test",
                model_digest=digest_for("model"),
                protocol_digest=VARIATION_PROTOCOL_DIGEST,
                policy_digest=AuthorityPolicy.candidate_execution().digest,
                corpus=corpus,
                evaluator_revision="remote-evaluator-test-v1",
                public_key_path=self.key_path,
                command=self.command,
                docker_config=config,
            )
        serialized = canonical_json(manifest)
        self.assertNotIn(str(seed), serialized)
        self.assertNotIn("evaluator-private-seed", serialized)
        self.assertEqual({task["split"] for task in manifest["task_bindings"]}, {"train", "heldout"})
        self.assertNotIn("dev", {task["split"] for task in manifest["task_bindings"]})

    def test_evaluator_once_signs_real_train_controller_receipts(self) -> None:
        # The production evaluator is a separate receipt-only process. Close
        # this test fixture's trainer-side ledger before exercising that exact
        # controller cohosting guard in-process.
        self.ledger.close()
        seed = self.root / "sealed-seed.bin"
        private_key = self.root / "sealed-private-key.bin"
        command = self.root / "sealed-endpoint.py"
        manifest_path = self.root / "sealed-service.json"
        seed.write_bytes(b"T" * 32)
        private_key.write_bytes(self.signer.private_key_raw)
        command.write_text("# evaluator endpoint identity\n", encoding="utf-8")
        corpus = EvaluationCorpus.generate(secret_seed_file=seed)
        repo = corpus.split("train")[0]
        config = DockerSandboxConfig()
        with patch.object(DockerSandboxConfig, "verify_image", return_value=config.pinned_image_id):
            manifest_value = build_remote_evaluator_service_manifest(
                campaign_id="campaign-real-train",
                model_digest=digest_for("model-real-train"),
                protocol_digest=VARIATION_PROTOCOL_DIGEST,
                policy_digest=AuthorityPolicy.candidate_execution().digest,
                corpus=corpus,
                evaluator_revision="remote-real-train-v1",
                public_key_path=self.key_path,
                command=command,
                docker_config=config,
            )
        manifest_path.write_text(canonical_json(manifest_value) + "\n", encoding="utf-8")
        source = b"def main(value):\n    return value\n"
        stable = {
            "schema_version": "egv-remote-variation-request-v1",
            "service_manifest_digest": manifest_value["service_manifest_digest"],
            "campaign_id": manifest_value["campaign_id"],
            "model_digest": manifest_value["model_digest"],
            "protocol_digest": manifest_value["protocol_digest"],
            "policy_digest": manifest_value["policy_digest"],
            "run_id": "run-real-train",
            "arm_policy_digest": arm_policy("B").digest,
            "data_manifest_digest": manifest_value["data_manifest_digest"],
            "task_manifest_digest": manifest_value["task_manifest_digest"],
            "evaluator_digest": manifest_value["evaluator_digest"],
            "docker_image_digest": manifest_value["docker_image_digest"],
            "candidate_id": "candidate-real-train",
            "task_id": repo.template_id,
            "public_task_binding": repo.public_manifest_record(),
            "candidate_artifact_digest": digest_bytes(source),
            "candidate_source_b64": _encode_b64(source),
            "requested_authority": "EXECUTE_CANDIDATE",
            "declared_locus": repo.public_locus,
        }
        body = {
            **stable,
            "operation_digest": _operation_digest(stable),
            "receipt_sequence_start": 1,
            "previous_receipt_hash": GENESIS_HASH,
        }
        request = {**body, "request_digest": digest_for(body)}
        sandbox_result = SandboxResult(
            digest_for("sandbox-real-train"),
            digest_bytes(source),
            "PASS",
            "UNDER_25",
            canonical_bytes(repo.expected_output) + b"\n",
            "SUCCESS",
            1,
            {"backend": "docker-enforced-v1", "candidate_contract": "pure-return-v1"},
        )

        def mock_verify_image(_config):
            return config.pinned_image_id

        def mock_execute(_sandbox, _source, _opaque_input, **_kwargs):
            return sandbox_result

        # Preserve the gateway's frozen-method identity checks while replacing
        # only the Docker boundary. The controller, private oracle comparison,
        # signing, and response validation all remain real.
        with patch.object(DockerSandboxConfig, "from_environment", return_value=config), patch.object(
            DockerSandboxConfig, "verify_image", mock_verify_image
        ), patch("egv.variation.loop._ORIGINAL_DOCKER_CONFIG_VERIFY_IMAGE", mock_verify_image), patch.object(
            # Match the sandbox's own immutable identity sentinel as well.
            # Both sentinels still reject any uncoordinated method mutation.
            DockerCandidateSandbox, "execute", mock_execute
        ), patch("egv.evaluation.sandbox._ORIGINAL_CONFIG_VERIFY_IMAGE", mock_verify_image), patch(
            "egv.variation.loop._ORIGINAL_DOCKER_SANDBOX_EXECUTE", mock_execute
        ), patch(
            "egv.evaluation.sandbox._ORIGINAL_DOCKER_EXECUTE", mock_execute
        ):
            response = run_remote_evaluator_once(
                request,
                service_manifest=manifest_path,
                evaluator_seed=seed,
                evaluator_private_key=private_key,
                workspace=self.root / "remote-workspace",
                state_root=self.root / "remote-state",
            )
        self.assertEqual(response["result"]["diagnostic_enum"], "PASS")
        self.assertEqual(response["result"]["disposition"], "PROMOTED")
        self.assertEqual(len(response["receipts"]), 3)
        for receipt in response["receipts"]:
            self.assertEqual(receipt["run_id"], "run-real-train")
            self.assertEqual(receipt["task_id"], repo.template_id)
            self.assertEqual(receipt["task_family"], repo.family_id)
            self.assertEqual(receipt["normalized_public_locus"], repo.public_locus)
            self.assertEqual(receipt["public_rule_id"], repo.public_rule_id)
            self.assertEqual(receipt["arm_policy_digest"], arm_policy("B").digest)
        serialized = canonical_json(response)
        self.assertNotIn(str(seed), serialized)
        self.assertNotIn(canonical_json(repo.hidden_spec), serialized)

    def test_atomic_receipt_chain_rolls_back_if_later_signature_is_invalid(self) -> None:
        first = self.signer.sign_receipt(
            {
                "receipt_type": "AUTHORITY", "campaign_id": "campaign", "run_id": "run", "task_id": "task",
                "request_id": "request-a", "candidate_id": "candidate", "decision": "ALLOW",
            },
            sequence=1,
            previous_receipt_hash=GENESIS_HASH,
            idempotency_key="atomic-a",
        )
        second = self.signer.sign_receipt(
            {
                "receipt_type": "VERDICT", "campaign_id": "campaign", "run_id": "run", "task_id": "task",
                "request_id": "request-v", "candidate_id": "candidate", "decision": "PASS",
                "diagnostic_enum": "PASS",
            },
            sequence=2,
            previous_receipt_hash=receipt_hash(first),
            idempotency_key="atomic-v",
        )
        second["signature"] = ("A" if second["signature"][0] != "A" else "B") + second["signature"][1:]
        with self.assertRaises(Exception):
            self.ledger.ingest_receipts_atomic([first, second], self.signer.public_key)
        self.assertEqual(self.ledger.receipts(), [])

    def test_durable_evaluator_returns_byte_identical_cached_response_after_restart(self) -> None:
        manifest = RemoteEvaluatorServiceManifest(self.manifest_value)
        state_root = self.root / "durable-state"
        request = self.durable_request("retry")
        first = _durable_remote_response(
            request,
            manifest=manifest,
            signer=self.signer,
            state_root=state_root,
            build_response=self.durable_builder(request),
        )
        called = []
        resumed = {**request, "receipt_sequence_start": 2, "previous_receipt_hash": receipt_hash(first["receipts"][-1])}
        resumed["request_digest"] = digest_for({key: value for key, value in resumed.items() if key != "request_digest"})
        second = _durable_remote_response(
            resumed,
            manifest=manifest,
            signer=self.signer,
            state_root=state_root,
            build_response=lambda: called.append(True),
        )
        self.assertEqual(canonical_bytes(first), canonical_bytes(second))
        self.assertEqual(called, [])

    def test_durable_evaluator_rejects_uncached_operation_at_stale_anchor_before_effect(self) -> None:
        manifest = RemoteEvaluatorServiceManifest(self.manifest_value)
        state_root = self.root / "stale-anchor-state"
        first_request = self.durable_request("first-operation")
        _durable_remote_response(
            first_request,
            manifest=manifest,
            signer=self.signer,
            state_root=state_root,
            build_response=self.durable_builder(first_request),
        )
        uncached_request = self.durable_request("uncached-operation")
        effects = []
        with self.assertRaisesRegex(
            VariationConfigurationError,
            "stale or forked receipt anchor",
        ):
            _durable_remote_response(
                uncached_request,
                manifest=manifest,
                signer=self.signer,
                state_root=state_root,
                build_response=lambda: effects.append("executed"),
            )
        self.assertEqual(effects, [])

    def test_crash_after_execution_intent_quarantines_retry_without_reexecution(self) -> None:
        manifest = RemoteEvaluatorServiceManifest(self.manifest_value)
        state_root = self.root / "crash-state"
        request = self.durable_request("crash")
        executions = []

        def crash_after_effect() -> dict:
            executions.append("executed")
            raise RuntimeError("simulated process loss after sandbox effect")

        with self.assertRaises(RuntimeError):
            _durable_remote_response(
                request,
                manifest=manifest,
                signer=self.signer,
                state_root=state_root,
                build_response=crash_after_effect,
            )
        with self.assertRaises(VariationDependencyError):
            _durable_remote_response(
                request,
                manifest=manifest,
                signer=self.signer,
                state_root=state_root,
                build_response=crash_after_effect,
            )
        self.assertEqual(executions, ["executed"])

    def test_concurrent_requests_cannot_sign_receipt_forks(self) -> None:
        manifest = RemoteEvaluatorServiceManifest(self.manifest_value)
        state_root = self.root / "fork-state"
        requests = [self.durable_request("fork-a"), self.durable_request("fork-b")]
        outcomes = []

        def worker(request: dict) -> None:
            try:
                _durable_remote_response(
                    request,
                    manifest=manifest,
                    signer=self.signer,
                    state_root=state_root,
                    build_response=self.durable_builder(request),
                )
                outcomes.append("complete")
            except VariationConfigurationError:
                outcomes.append("stale")

        threads = [threading.Thread(target=worker, args=(request,)) for request in requests]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertCountEqual(outcomes, ["complete", "stale"])


if __name__ == "__main__":
    unittest.main()
