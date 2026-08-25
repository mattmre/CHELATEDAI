"""Sealed command bridge for an independent production Variation evaluator."""

from __future__ import annotations

import base64
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

from ..canonical import (
    GENESIS_HASH,
    canonical_bytes,
    canonical_json,
    content_id,
    digest_bytes,
    digest_for,
    failure_family_root,
    validate_sha256,
)
from ..evaluation.controller import EvaluationResult, EvaluatorController, HiddenEvaluatorRunner
from ..evaluation.diagnostics import Diagnostic, validate_diagnostic, validate_disposition, validate_resource_bucket
from ..ledger import EvidenceLedger
from ..receipts import ReceiptJournal, ReceiptSigner, key_id_for_public_key, load_public_key, receipt_hash, verify_receipt
from .errors import VariationConfigurationError, VariationDependencyError


REMOTE_VARIATION_SERVICE_SCHEMA = "egv-remote-variation-service-v1"
REMOTE_VARIATION_REQUEST_SCHEMA = "egv-remote-variation-request-v1"
REMOTE_VARIATION_RESPONSE_SCHEMA = "egv-remote-variation-response-v1"
REMOTE_VARIATION_TIMEOUT_SECONDS = 600
REMOTE_VARIATION_REQUEST_LIMIT = 512 * 1024
REMOTE_VARIATION_RESPONSE_LIMIT = 1024 * 1024
REMOTE_VARIATION_STATE_SCHEMA = "egv-remote-variation-state-v1"

_TASK_FIELDS = frozenset({"template_id", "family_id", "split", "ordinal", "source_digest", "public_rule_id", "public_locus"})
_MANIFEST_FIELDS = frozenset(
    {
        "schema_version",
        "campaign_id",
        "model_digest",
        "protocol_digest",
        "policy_digest",
        "data_manifest_digest",
        "task_manifest_digest",
        "task_bindings",
        "evaluator_revision",
        "evaluator_digest",
        "docker_image_digest",
        "docker_config_digest",
        "authority_policy_digest",
        "command_digest",
        "evaluator_key_id",
        "evaluator_public_key_digest",
        "service_manifest_digest",
    }
)
_REQUEST_FIELDS = frozenset(
    {
        "schema_version",
        "operation_digest",
        "request_digest",
        "service_manifest_digest",
        "campaign_id",
        "model_digest",
        "protocol_digest",
        "policy_digest",
        "run_id",
        "arm_policy_digest",
        "data_manifest_digest",
        "task_manifest_digest",
        "evaluator_digest",
        "docker_image_digest",
        "candidate_id",
        "task_id",
        "public_task_binding",
        "candidate_artifact_digest",
        "candidate_source_b64",
        "requested_authority",
        "declared_locus",
        "receipt_sequence_start",
        "previous_receipt_hash",
    }
)
_RESPONSE_FIELDS = frozenset(
    {
        "schema_version",
        "operation_digest",
        "request_digest",
        "service_manifest_digest",
        "result",
        "receipts",
        "signing_key_id",
        "signature",
    }
)


def _require_digest(value: Any, name: str) -> str:
    try:
        return validate_sha256(value, name)
    except Exception as exc:
        raise VariationConfigurationError(str(exc)) from exc


def _decode_b64(value: Any, name: str, *, exact_length: Optional[int] = None) -> bytes:
    if not isinstance(value, str) or not value:
        raise VariationConfigurationError("{} must be non-empty base64url".format(name))
    try:
        padded = value + "=" * (-len(value) % 4)
        decoded = base64.b64decode(padded.encode("ascii"), altchars=b"-_", validate=True)
    except (ValueError, UnicodeError) as exc:
        raise VariationConfigurationError("{} is not valid base64url".format(name)) from exc
    if _encode_b64(decoded) != value or (exact_length is not None and len(decoded) != exact_length):
        raise VariationConfigurationError("{} is not canonical base64url".format(name))
    return decoded


def _encode_b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


@contextmanager
def _exclusive_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        if os.name == "nt":
            import msvcrt

            handle.seek(0, os.SEEK_END)
            if handle.tell() == 0:
                handle.write("0")
                handle.flush()
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_LOCK, 1)
        else:
            import fcntl

            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            if os.name == "nt":
                import msvcrt

                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _path_has_link_or_reparse_component(path: Path) -> bool:
    """Return whether an absolute path traverses a link/reparse component."""

    cursor = Path(path)
    while True:
        metadata = os.lstat(cursor)
        attributes = int(getattr(metadata, "st_file_attributes", 0))
        if stat.S_ISLNK(metadata.st_mode) or attributes & 0x400:
            return True
        if cursor.parent == cursor:
            return False
        cursor = cursor.parent


def _read_open_executable(fd: int, path: Path, expected_digest: str) -> None:
    """Verify executable bytes and pathname identity through one open file."""

    opened_before = os.fstat(fd)
    named = os.stat(path, follow_symlinks=False)
    if (
        not stat.S_ISREG(opened_before.st_mode)
        or int(getattr(opened_before, "st_nlink", 0)) != 1
        or (opened_before.st_dev, opened_before.st_ino) != (named.st_dev, named.st_ino)
    ):
        raise VariationDependencyError("pinned Python executable identity is invalid")
    os.lseek(fd, 0, os.SEEK_SET)
    digest = hashlib.sha256()
    observed = 0
    while True:
        chunk = os.read(fd, 1024 * 1024)
        if not chunk:
            break
        observed += len(chunk)
        digest.update(chunk)
    opened_after = os.fstat(fd)
    named_after = os.stat(path, follow_symlinks=False)
    if (
        (opened_after.st_dev, opened_after.st_ino, opened_after.st_size)
        != (opened_before.st_dev, opened_before.st_ino, opened_before.st_size)
        or (named_after.st_dev, named_after.st_ino)
        != (opened_before.st_dev, opened_before.st_ino)
        or observed != opened_before.st_size
        or digest.hexdigest() != expected_digest
    ):
        raise VariationDependencyError("pinned Python executable changed or differs from its manifest")
    os.lseek(fd, 0, os.SEEK_SET)


@contextmanager
def pinned_python_invocation(path: Path, expected_digest: str):
    """Hold one verified Python identity through process creation.

    POSIX copies the verified bytes into a sealed anonymous executable and
    launches that object through ``/proc/self/fd``.  Windows holds
    a read-only handle that denies write/delete sharing while ``CreateProcess``
    opens the verified pathname.  This boundary addresses pathname replacement;
    it is not a defense against same-user process injection or memory tampering.
    """

    supplied = Path(path)
    target = Path(os.path.abspath(os.fspath(supplied)))
    try:
        expected = validate_sha256(expected_digest, "pinned Python executable digest")
        if not supplied.is_absolute() or _path_has_link_or_reparse_component(target):
            raise VariationDependencyError("pinned Python executable path is not direct")
    except (OSError, ValueError) as exc:
        raise VariationDependencyError("pinned Python executable identity is unavailable") from exc

    if os.name == "nt":
        import ctypes
        import msvcrt
        from ctypes import wintypes

        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        create_file = kernel32.CreateFileW
        create_file.argtypes = (
            wintypes.LPCWSTR,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.LPVOID,
            wintypes.DWORD,
            wintypes.DWORD,
            wintypes.HANDLE,
        )
        create_file.restype = wintypes.HANDLE
        close_handle = kernel32.CloseHandle
        close_handle.argtypes = (wintypes.HANDLE,)
        close_handle.restype = wintypes.BOOL
        handle = create_file(
            str(target),
            0x80000000,  # GENERIC_READ
            0x00000001,  # FILE_SHARE_READ: deny write and delete sharing
            None,
            3,  # OPEN_EXISTING
            0x00200000,  # FILE_FLAG_OPEN_REPARSE_POINT
            None,
        )
        invalid_handle = ctypes.c_void_p(-1).value
        if handle == invalid_handle:
            raise VariationDependencyError("pinned Python executable could not be locked") from ctypes.WinError(
                ctypes.get_last_error()
            )
        fd: Optional[int] = None
        try:
            fd = msvcrt.open_osfhandle(int(handle), os.O_RDONLY | getattr(os, "O_BINARY", 0))
            handle = None
            _read_open_executable(fd, target, expected)
        except BaseException as exc:
            cleanup_errors = []
            if fd is not None:
                try:
                    os.close(fd)
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
                fd = None
            elif handle not in (None, invalid_handle):
                try:
                    _close_windows_handle(
                        close_handle,
                        handle,
                        "pinned Python executable",
                    )
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
                handle = None
            if isinstance(exc, OSError):
                error = VariationDependencyError(
                    "pinned Python executable identity is unavailable"
                )
                _preserve_cleanup_context(error, cleanup_errors)
                raise error from exc
            _preserve_cleanup_context(exc, cleanup_errors)
            raise
        try:
            yield str(target), {}
        except BaseException as primary:
            cleanup_errors = []
            if fd is not None:
                try:
                    os.close(fd)
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
            elif handle not in (None, invalid_handle):
                try:
                    _close_windows_handle(
                        close_handle,
                        handle,
                        "pinned Python executable",
                    )
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
            _preserve_cleanup_context(primary, cleanup_errors)
            raise
        else:
            if fd is not None:
                os.close(fd)
            elif handle not in (None, invalid_handle):
                _close_windows_handle(
                    close_handle,
                    handle,
                    "pinned Python executable",
                )
        return

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        fd = os.open(target, flags)
    except OSError as exc:
        raise VariationDependencyError("pinned Python executable could not be opened") from exc
    memfd: Optional[int] = None
    try:
        try:
            _read_open_executable(fd, target, expected)
        except OSError as exc:
            raise VariationDependencyError("pinned Python executable identity is unavailable") from exc
        if not hasattr(os, "memfd_create") or not Path("/proc/self/fd").is_dir():
            raise VariationDependencyError("immutable pinned Python execution is unavailable")
        import fcntl

        memfd_flags = getattr(os, "MFD_CLOEXEC", 0x0001) | getattr(os, "MFD_ALLOW_SEALING", 0x0002)
        mfd_exec = getattr(os, "MFD_EXEC", 0x0010)
        try:
            memfd = os.memfd_create("egv-pinned-python", flags=memfd_flags | mfd_exec)
        except OSError as exc:
            if exc.errno not in {getattr(os, "EINVAL", 22), getattr(os, "ENOSYS", 38)}:
                raise VariationDependencyError("immutable pinned Python object could not be created") from exc
            try:
                memfd = os.memfd_create("egv-pinned-python", flags=memfd_flags)
            except OSError as fallback_exc:
                raise VariationDependencyError("immutable pinned Python object could not be created") from fallback_exc
        os.lseek(fd, 0, os.SEEK_SET)
        copied_digest = hashlib.sha256()
        copied_size = 0
        while True:
            chunk = os.read(fd, 1024 * 1024)
            if not chunk:
                break
            view = memoryview(chunk)
            while view:
                written = os.write(memfd, view)
                if written <= 0:
                    raise VariationDependencyError("immutable pinned Python copy was incomplete")
                view = view[written:]
            copied_digest.update(chunk)
            copied_size += len(chunk)
        if copied_digest.hexdigest() != expected or copied_size != os.fstat(fd).st_size:
            raise VariationDependencyError("immutable pinned Python copy differs from its manifest")
        os.fchmod(memfd, 0o500)
        required_seals = (
            getattr(fcntl, "F_SEAL_WRITE", 0x0008)
            | getattr(fcntl, "F_SEAL_GROW", 0x0004)
            | getattr(fcntl, "F_SEAL_SHRINK", 0x0002)
            | getattr(fcntl, "F_SEAL_SEAL", 0x0001)
        )
        try:
            fcntl.fcntl(memfd, getattr(fcntl, "F_ADD_SEALS", 1033), required_seals)
            observed_seals = fcntl.fcntl(memfd, getattr(fcntl, "F_GET_SEALS", 1034))
        except OSError as exc:
            raise VariationDependencyError("immutable pinned Python object could not be sealed") from exc
        if observed_seals & required_seals != required_seals:
            raise VariationDependencyError("immutable pinned Python object seals are incomplete")
        os.lseek(memfd, 0, os.SEEK_SET)
        sealed_digest = hashlib.sha256()
        sealed_size = 0
        while True:
            chunk = os.read(memfd, 1024 * 1024)
            if not chunk:
                break
            sealed_digest.update(chunk)
            sealed_size += len(chunk)
        if sealed_digest.hexdigest() != expected or sealed_size != copied_size:
            raise VariationDependencyError("sealed pinned Python object failed verification")
        os.lseek(memfd, 0, os.SEEK_SET)
        yield "/proc/self/fd/{}".format(memfd), {"pass_fds": (memfd,)}
    except BaseException as primary:
        cleanup_errors = []
        for descriptor in (memfd, fd):
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
        _preserve_cleanup_context(primary, cleanup_errors)
        raise
    else:
        cleanup_errors = []
        for descriptor in (memfd, fd):
            if descriptor is not None:
                try:
                    os.close(descriptor)
                except BaseException as cleanup_error:
                    cleanup_errors.append(cleanup_error)
        if cleanup_errors:
            error = VariationDependencyError(
                "pinned Python executable identity cleanup failed"
            )
            _preserve_cleanup_context(error, cleanup_errors)
            raise error from cleanup_errors[0]


def _terminate_bounded_process_tree(
    process: subprocess.Popen[Any], windows_job: Any = None, *, deadline: Optional[float] = None
) -> None:
    """Terminate and reap the complete evaluator process tree."""

    cleanup_errors = []
    if os.name == "nt":
        if windows_job is not None:
            try:
                import ctypes
                from ctypes import wintypes

                kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
                terminate_job = kernel32.TerminateJobObject
                terminate_job.argtypes = (wintypes.HANDLE, wintypes.UINT)
                terminate_job.restype = wintypes.BOOL
                if not terminate_job(windows_job, 1):
                    raise ctypes.WinError(ctypes.get_last_error())
            except BaseException as exc:
                cleanup_errors.append(exc)
        elif process.poll() is None:
            try:
                process.kill()
            except BaseException as exc:
                cleanup_errors.append(exc)
    else:
        if not all(
            hasattr(os, name)
            for name in ("waitid", "P_PID", "WEXITED", "WNOHANG", "WNOWAIT")
        ):
            cleanup_errors.append(
                VariationDependencyError("pid-safe evaluator cleanup anchor is unavailable")
            )
        else:
            try:
                os.waitid(  # type: ignore[attr-defined]
                    os.P_PID,  # type: ignore[attr-defined]
                    process.pid,
                    os.WEXITED | os.WNOHANG | os.WNOWAIT,  # type: ignore[attr-defined]
                )
            except ChildProcessError as exc:
                primary = VariationDependencyError(
                    "remote Variation evaluator leader anchor was lost before group cleanup"
                )
                primary.__cause__ = exc
                cleanup_errors.append(primary)
            except BaseException as exc:
                cleanup_errors.append(exc)
            else:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                except BaseException as exc:
                    cleanup_errors.append(exc)
    cleanup_deadline = deadline if deadline is not None else time.monotonic() + 5
    try:
        process.wait(timeout=max(0.0, cleanup_deadline - time.monotonic()))
    except BaseException as exc:
        cleanup_errors.append(exc)
    if cleanup_errors:
        error = VariationDependencyError(
            "remote Variation evaluator process-tree cleanup failed"
        )
        _preserve_cleanup_context(error, cleanup_errors)
        raise error from cleanup_errors[0]


def _bounded_process_exited_without_reap(process: subprocess.Popen[Any]) -> bool:
    """Observe POSIX leader exit while retaining its PID/process-group anchor."""

    if os.name == "nt":
        return process.poll() is not None
    if not all(hasattr(os, name) for name in ("waitid", "P_PID", "WEXITED", "WNOHANG", "WNOWAIT")):
        raise VariationDependencyError("pid-safe evaluator exit observation is unavailable")
    try:
        return os.waitid(  # type: ignore[attr-defined]
            os.P_PID, process.pid, os.WEXITED | os.WNOHANG | os.WNOWAIT  # type: ignore[attr-defined]
        ) is not None
    except ChildProcessError as exc:
        raise VariationDependencyError("remote Variation evaluator leader was reaped outside containment") from exc


def _preserve_cleanup_context(error: BaseException, cleanup_errors: Sequence[BaseException]) -> None:
    """Attach secondary cleanup evidence without relying on Python 3.11 notes."""

    if cleanup_errors:
        existing = tuple(getattr(error, "cleanup_context", ()))
        flattened = []
        for cleanup_error in cleanup_errors:
            flattened.append(str(cleanup_error))
            flattened.extend(tuple(getattr(cleanup_error, "cleanup_context", ())))
        setattr(
            error,
            "cleanup_context",
            existing + tuple(flattened),
        )


def _assign_process_to_windows_job(assign_job: Any, windows_job: Any, process_handle: Any) -> None:
    """Assign a still-suspended process so fault injection can prove fallback cleanup."""

    if not assign_job(windows_job, process_handle):
        import ctypes

        raise ctypes.WinError(ctypes.get_last_error())


def _close_windows_handle(close_handle: Any, handle: Any, label: str) -> None:
    """Close one owned Windows handle and fail if ownership was not released."""

    if not close_handle(handle):
        import ctypes

        native_error = ctypes.WinError(ctypes.get_last_error())
        raise OSError("{} close failed: {}".format(label, native_error)) from native_error


@contextmanager
def _owned_windows_handle(close_handle: Any, handle: Any, label: str):
    """Preserve a primary error while validating cleanup of a temporary handle."""

    try:
        yield handle
    except BaseException as primary:
        try:
            _close_windows_handle(close_handle, handle, label)
        except BaseException as cleanup_error:
            _preserve_cleanup_context(primary, (cleanup_error,))
        raise
    else:
        _close_windows_handle(close_handle, handle, label)


def _cleanup_failed_windows_start(
    process: Optional[subprocess.Popen[Any]],
    windows_job: Any,
    *,
    job_assigned: bool,
    close_handle: Any,
    deadline: float,
) -> Tuple[BaseException, ...]:
    """Clean only the owned failed-start process and Job Object resources."""

    cleanup_errors = []
    if process is not None:
        if job_assigned:
            try:
                _terminate_bounded_process_tree(process, windows_job, deadline=deadline)
            except BaseException as exc:
                cleanup_errors.append(exc)
        else:
            try:
                # The process is still suspended and unassigned. Popen.kill uses
                # its owned process handle, not a reusable ambient PID/PGID.
                process.kill()
            except BaseException as exc:
                cleanup_errors.append(exc)
            try:
                process.wait(timeout=max(0.0, deadline - time.monotonic()))
            except BaseException as exc:
                cleanup_errors.append(exc)
        for stream in (process.stdin, process.stdout, process.stderr):
            if stream is not None:
                try:
                    stream.close()
                except BaseException as exc:
                    cleanup_errors.append(exc)
    if windows_job is not None:
        try:
            _close_windows_handle(close_handle, windows_job, "evaluator Job Object")
        except BaseException as exc:
            cleanup_errors.append(exc)
    return tuple(cleanup_errors)


def _start_bounded_process(
    invocation: Sequence[str], popen_kwargs: Mapping[str, Any], *, deadline: float
) -> Tuple[subprocess.Popen[Any], Any]:
    """Start an evaluator in a containment boundary before it can execute."""

    options = dict(popen_kwargs)
    windows_job = None
    if os.name != "nt":
        if time.monotonic() >= deadline:
            raise OSError("remote Variation evaluator startup deadline elapsed")
        options["start_new_session"] = True
        return subprocess.Popen(
            list(invocation), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **options
        ), None

    import ctypes
    from ctypes import wintypes

    creationflags = int(options.pop("creationflags", 0)) | 0x00000200 | 0x00000004
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_job = kernel32.CreateJobObjectW
    create_job.argtypes = (wintypes.LPVOID, wintypes.LPCWSTR)
    create_job.restype = wintypes.HANDLE
    set_job = kernel32.SetInformationJobObject
    set_job.argtypes = (wintypes.HANDLE, ctypes.c_int, wintypes.LPVOID, wintypes.DWORD)
    set_job.restype = wintypes.BOOL
    assign_job = kernel32.AssignProcessToJobObject
    assign_job.argtypes = (wintypes.HANDLE, wintypes.HANDLE)
    assign_job.restype = wintypes.BOOL
    create_snapshot = kernel32.CreateToolhelp32Snapshot
    create_snapshot.argtypes = (wintypes.DWORD, wintypes.DWORD)
    create_snapshot.restype = wintypes.HANDLE
    open_thread = kernel32.OpenThread
    open_thread.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    open_thread.restype = wintypes.HANDLE
    resume_thread = kernel32.ResumeThread
    resume_thread.argtypes = (wintypes.HANDLE,)
    resume_thread.restype = wintypes.DWORD
    terminate_job = kernel32.TerminateJobObject
    terminate_job.argtypes = (wintypes.HANDLE, wintypes.UINT)
    terminate_job.restype = wintypes.BOOL
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL

    windows_job = create_job(None, None)
    if not windows_job:
        raise ctypes.WinError(ctypes.get_last_error())

    class _Basic(ctypes.Structure):
        _fields_ = (("PerProcessUserTimeLimit", ctypes.c_longlong), ("PerJobUserTimeLimit", ctypes.c_longlong),
                    ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD), ("SchedulingClass", wintypes.DWORD))

    class _Io(ctypes.Structure):
        _fields_ = tuple((name, ctypes.c_ulonglong) for name in (
            "ReadOperationCount", "WriteOperationCount", "OtherOperationCount",
            "ReadTransferCount", "WriteTransferCount", "OtherTransferCount"))

    class _Extended(ctypes.Structure):
        _fields_ = (("BasicLimitInformation", _Basic), ("IoInfo", _Io),
                    ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t))

    info = _Extended()
    info.BasicLimitInformation.LimitFlags = 0x00002000
    if not set_job(windows_job, 9, ctypes.byref(info), ctypes.sizeof(info)):
        primary = ctypes.WinError(ctypes.get_last_error())
        try:
            _close_windows_handle(close_handle, windows_job, "evaluator Job Object")
        except BaseException as cleanup_error:
            _preserve_cleanup_context(primary, (cleanup_error,))
        raise primary
    if time.monotonic() >= deadline:
        primary = OSError("remote Variation evaluator startup deadline elapsed")
        try:
            _close_windows_handle(close_handle, windows_job, "evaluator Job Object")
        except BaseException as cleanup_error:
            _preserve_cleanup_context(primary, (cleanup_error,))
        raise primary
    process = None
    job_assigned = False
    try:
        process = subprocess.Popen(
            list(invocation), stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            creationflags=creationflags, **options
        )
        _assign_process_to_windows_job(assign_job, windows_job, wintypes.HANDLE(process._handle))
        job_assigned = True
        # Python closes the primary thread handle, so locate and resume it.
        snapshot = create_snapshot(0x00000004, 0)
        if snapshot == ctypes.c_void_p(-1).value:
            raise ctypes.WinError(ctypes.get_last_error())

        class _ThreadEntry(ctypes.Structure):
            _fields_ = (("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                        ("th32ThreadID", wintypes.DWORD), ("th32OwnerProcessID", wintypes.DWORD),
                        ("tpBasePri", ctypes.c_long), ("tpDeltaPri", ctypes.c_long), ("dwFlags", wintypes.DWORD))

        thread_first = kernel32.Thread32First
        thread_first.argtypes = (wintypes.HANDLE, ctypes.POINTER(_ThreadEntry))
        thread_first.restype = wintypes.BOOL
        thread_next = kernel32.Thread32Next
        thread_next.argtypes = (wintypes.HANDLE, ctypes.POINTER(_ThreadEntry))
        thread_next.restype = wintypes.BOOL

        with _owned_windows_handle(
            close_handle,
            snapshot,
            "evaluator thread snapshot",
        ):
            entry = _ThreadEntry()
            entry.dwSize = ctypes.sizeof(entry)
            found = []
            present = thread_first(snapshot, ctypes.byref(entry))
            while present:
                if int(entry.th32OwnerProcessID) == process.pid:
                    found.append(int(entry.th32ThreadID))
                present = thread_next(snapshot, ctypes.byref(entry))
        if len(found) != 1:
            raise OSError("suspended evaluator has no unique primary thread")
        thread_handle = open_thread(0x0002, False, found[0])
        if not thread_handle:
            raise ctypes.WinError(ctypes.get_last_error())
        with _owned_windows_handle(
            close_handle,
            thread_handle,
            "evaluator primary thread",
        ):
            if resume_thread(thread_handle) != 1:
                raise OSError("suspended evaluator thread has an invalid suspend count")
        return process, windows_job
    except BaseException as primary:
        cleanup_errors = _cleanup_failed_windows_start(
            process,
            windows_job,
            job_assigned=job_assigned,
            close_handle=close_handle,
            deadline=time.monotonic() + 5,
        )
        _preserve_cleanup_context(primary, cleanup_errors)
        raise


def _run_bounded_command(
    invocation: Sequence[str],
    request_text: str,
    *,
    popen_kwargs: Optional[Mapping[str, Any]] = None,
) -> Tuple[int, bytes, bytes]:
    """Execute with hard in-memory stdout/stderr caps and a frozen timeout."""

    started_at = time.monotonic()
    deadline = started_at + REMOTE_VARIATION_TIMEOUT_SECONDS
    try:
        process, windows_job = _start_bounded_process(
            invocation, dict(popen_kwargs or {}), deadline=deadline
        )
    except OSError as exc:
        error: VariationDependencyError
        if time.monotonic() >= deadline:
            error = VariationDependencyError("remote Variation evaluator timed out")
        else:
            error = VariationDependencyError(
                "remote Variation evaluator invocation failed"
            )
        _preserve_cleanup_context(error, (exc,))
        raise error from exc
    stdout = bytearray()
    stderr = bytearray()
    overflow = []
    cleanup_errors = []
    timeout_error: Optional[subprocess.TimeoutExpired] = None
    writer_errors: list[BaseException] = []
    reader_errors: list[BaseException] = []
    request_complete = threading.Event()

    def drain(stream: Any, sink: bytearray, label: str) -> None:
        try:
            while True:
                chunk = stream.read(65536)
                if not chunk:
                    break
                if len(sink) + len(chunk) > REMOTE_VARIATION_RESPONSE_LIMIT:
                    overflow.append(label)
                    break
                sink.extend(chunk)
        except BaseException as exc:
            reader_errors.append(exc)

    def write_request() -> None:
        try:
            assert process.stdin is not None
            request = request_text.encode("utf-8")
            written = process.stdin.write(request)
            process.stdin.flush()
            if written != len(request):
                raise OSError("remote Variation evaluator consumed an incomplete request")
            request_complete.set()
        except BaseException as exc:
            writer_errors.append(exc)
        finally:
            try:
                if process.stdin is not None:
                    process.stdin.close()
            except BaseException as exc:
                writer_errors.append(exc)

    startup_timed_out = time.monotonic() >= deadline
    if startup_timed_out:
        timeout_error = subprocess.TimeoutExpired(list(invocation), REMOTE_VARIATION_TIMEOUT_SECONDS)
        reader_threads: Tuple[threading.Thread, ...] = ()
        threads: Tuple[threading.Thread, ...] = ()
    else:
        reader_threads = (
            threading.Thread(target=drain, args=(process.stdout, stdout, "stdout"), daemon=True),
            threading.Thread(target=drain, args=(process.stderr, stderr, "stderr"), daemon=True),
        )
        writer_thread = threading.Thread(target=write_request, daemon=True)
        threads = (*reader_threads, writer_thread)
    for thread in threads:
        thread.start()
    try:
        while timeout_error is None and not overflow and not writer_errors and not reader_errors:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                timeout_error = subprocess.TimeoutExpired(list(invocation), REMOTE_VARIATION_TIMEOUT_SECONDS)
                break
            if _bounded_process_exited_without_reap(process):
                break
            time.sleep(min(remaining, 0.05))
    finally:
        # Always terminate the containment boundary: a successful or failed
        # parent may leave descendants alive or holding inherited pipes.
        cleanup_deadline = time.monotonic() + 5
        try:
            _terminate_bounded_process_tree(process, windows_job, deadline=cleanup_deadline)
        except VariationDependencyError as exc:
            cleanup_errors.append(exc)
        for thread in threads:
            thread.join(timeout=max(0.0, cleanup_deadline - time.monotonic()))
        if any(thread.is_alive() for thread in threads):
            cleanup_errors.append(VariationDependencyError("remote Variation evaluator I/O cleanup did not complete"))
        for thread, stream in zip(reader_threads, (process.stdout, process.stderr)):
            if not thread.is_alive() and stream is not None:
                try:
                    stream.close()
                except BaseException as exc:
                    cleanup_errors.append(exc)
        if startup_timed_out:
            for stream in (process.stdin, process.stdout, process.stderr):
                if stream is not None:
                    try:
                        stream.close()
                    except BaseException as exc:
                        cleanup_errors.append(exc)
        if os.name == "nt" and windows_job is not None:
            import ctypes
            from ctypes import wintypes

            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            close_handle = kernel32.CloseHandle
            close_handle.argtypes = (wintypes.HANDLE,)
            close_handle.restype = wintypes.BOOL
            try:
                _close_windows_handle(
                    close_handle,
                    windows_job,
                    "remote Variation evaluator Job Object",
                )
            except BaseException as exc:
                cleanup_errors.append(exc)
    if timeout_error is not None:
        error = VariationDependencyError("remote Variation evaluator timed out")
        _preserve_cleanup_context(error, cleanup_errors)
        raise error from timeout_error
    if overflow:
        error = VariationDependencyError(
            "remote Variation evaluator {} exceeded the bounded output limit".format(overflow[0])
        )
        _preserve_cleanup_context(error, cleanup_errors)
        raise error
    if writer_errors or not request_complete.is_set():
        error = VariationDependencyError("remote Variation evaluator request write did not complete")
        _preserve_cleanup_context(error, cleanup_errors)
        if writer_errors:
            raise error from writer_errors[0]
        raise error
    if reader_errors:
        error = VariationDependencyError("remote Variation evaluator response read did not complete")
        _preserve_cleanup_context(error, cleanup_errors)
        raise error from reader_errors[0]
    if cleanup_errors:
        error = VariationDependencyError(
            "remote Variation evaluator process-tree cleanup failed"
        )
        _preserve_cleanup_context(error, cleanup_errors)
        raise error from cleanup_errors[0]
    return int(process.returncode), bytes(stdout), bytes(stderr)


def _validate_task_binding(value: Any) -> Dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != _TASK_FIELDS:
        raise VariationConfigurationError("remote evaluator public task binding is not closed")
    result = dict(value)
    for field in ("template_id", "family_id", "split", "public_rule_id", "public_locus"):
        if not isinstance(result[field], str) or not result[field]:
            raise VariationConfigurationError("remote evaluator public task binding is malformed")
    if result["split"] not in {"train", "heldout"}:
        raise VariationConfigurationError("remote evaluator may expose only frozen train or held-out public task bindings")
    if not isinstance(result["ordinal"], int) or isinstance(result["ordinal"], bool) or result["ordinal"] < 1:
        raise VariationConfigurationError("remote evaluator public task ordinal is invalid")
    _require_digest(result["source_digest"], "task source digest")
    return result


class RemoteEvaluatorServiceManifest:
    """Closed, self-digesting trust manifest for one evaluator command."""

    def __init__(self, value: Mapping[str, Any]) -> None:
        if not isinstance(value, Mapping) or set(value) != _MANIFEST_FIELDS:
            raise VariationConfigurationError("remote evaluator service manifest is not closed")
        normalized = dict(value)
        supplied = normalized.pop("service_manifest_digest")
        if normalized.get("schema_version") != REMOTE_VARIATION_SERVICE_SCHEMA or digest_for(normalized) != supplied:
            raise VariationConfigurationError("remote evaluator service manifest digest is invalid")
        for field in (
            "model_digest",
            "protocol_digest",
            "policy_digest",
            "data_manifest_digest",
            "task_manifest_digest",
            "evaluator_digest",
            "docker_config_digest",
            "authority_policy_digest",
            "command_digest",
            "evaluator_public_key_digest",
        ):
            _require_digest(value[field], field)
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", str(value["docker_image_digest"])):
            raise VariationConfigurationError("remote evaluator Docker image is not pinned by SHA-256 ID")
        if not isinstance(value["campaign_id"], str) or not value["campaign_id"]:
            raise VariationConfigurationError("remote evaluator campaign ID is missing")
        if not isinstance(value["evaluator_revision"], str) or not value["evaluator_revision"]:
            raise VariationConfigurationError("remote evaluator revision is missing")
        if value["evaluator_digest"] != digest_for(value["evaluator_revision"]):
            raise VariationConfigurationError("remote evaluator revision digest is invalid")
        tasks = value["task_bindings"]
        if not isinstance(tasks, list) or not tasks:
            raise VariationConfigurationError("remote evaluator task registry is empty")
        bindings = [_validate_task_binding(item) for item in tasks]
        if [item["template_id"] for item in bindings] != sorted(set(item["template_id"] for item in bindings)):
            raise VariationConfigurationError("remote evaluator task registry is not unique and sorted")
        if digest_for(bindings) != value["task_manifest_digest"]:
            raise VariationConfigurationError("remote evaluator task registry digest is invalid")
        self._value = dict(value)
        self._tasks = {item["template_id"]: item for item in bindings}

    @classmethod
    def from_path(cls, path: Path) -> "RemoteEvaluatorServiceManifest":
        target = Path(path)
        if target.is_symlink() or not target.is_file():
            raise VariationDependencyError("remote evaluator service manifest must be a regular file")
        try:
            value = json.loads(target.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            raise VariationConfigurationError("remote evaluator service manifest cannot be decoded") from exc
        return cls(value)

    @property
    def digest(self) -> str:
        return self._value["service_manifest_digest"]

    def __getitem__(self, name: str) -> Any:
        return self._value[name]

    def public_record(self, task_id: str) -> Optional[Dict[str, Any]]:
        value = self._tasks.get(task_id)
        return dict(value) if value is not None else None

    def validate_integrity(self) -> None:
        current = dict(self._value)
        supplied = current.pop("service_manifest_digest", None)
        if supplied != digest_for(current):
            raise VariationDependencyError("remote evaluator service manifest changed after validation")
        bindings = [_validate_task_binding(item) for item in current["task_bindings"]]
        if digest_for(bindings) != current["task_manifest_digest"]:
            raise VariationDependencyError("remote evaluator task registry changed after validation")


def build_remote_evaluator_service_manifest(
    *,
    campaign_id: str,
    model_digest: str,
    protocol_digest: str,
    policy_digest: str,
    corpus: Any,
    evaluator_revision: str,
    public_key_path: Path,
    command: Path,
    docker_config: Any,
) -> Dict[str, Any]:
    """Freeze a public-only service manifest from evaluator-owned authority."""

    from ..evaluation.authority import AuthorityPolicy
    from ..evaluation.dataset import EvaluationCorpus
    from ..evaluation.sandbox import DockerSandboxConfig

    if type(corpus) is not EvaluationCorpus:
        raise VariationDependencyError("remote service freeze requires the exact evaluator corpus")
    if type(docker_config) is not DockerSandboxConfig:
        raise VariationDependencyError("remote service freeze requires the exact Docker configuration")
    authority_policy = AuthorityPolicy.candidate_execution()
    if policy_digest != authority_policy.digest:
        raise VariationConfigurationError("remote service policy differs from candidate-execution authority")
    public_key = Path(public_key_path)
    endpoint = Path(command)
    if any(path.is_symlink() or not path.is_file() for path in (public_key, endpoint)):
        raise VariationDependencyError("remote service public key and command must be regular files")
    image_id = docker_config.verify_image()
    if image_id != docker_config.pinned_image_id:
        raise VariationConfigurationError("remote service Docker inspection differs from its pinned image")
    task_bindings = sorted(
        (
            repo.public_manifest_record()
            for repo in corpus.repositories
            if repo.split in {"train", "heldout"}
        ),
        key=lambda item: item["template_id"],
    )
    public_key_bytes = public_key.read_bytes()
    endpoint_bytes = endpoint.read_bytes()
    unsigned = {
        "schema_version": REMOTE_VARIATION_SERVICE_SCHEMA,
        "campaign_id": campaign_id,
        "model_digest": _require_digest(model_digest, "model digest"),
        "protocol_digest": _require_digest(protocol_digest, "protocol digest"),
        "policy_digest": policy_digest,
        "data_manifest_digest": corpus.manifest_digest(),
        "task_manifest_digest": digest_for(task_bindings),
        "task_bindings": task_bindings,
        "evaluator_revision": evaluator_revision,
        "evaluator_digest": digest_for(evaluator_revision),
        "docker_image_digest": docker_config.pinned_image_id,
        "docker_config_digest": digest_for(dict(docker_config.__dict__)),
        "authority_policy_digest": authority_policy.digest,
        "command_digest": hashlib.sha256(endpoint_bytes).hexdigest(),
        "evaluator_key_id": key_id_for_public_key(public_key_bytes),
        "evaluator_public_key_digest": hashlib.sha256(public_key_bytes).hexdigest(),
    }
    result = {**unsigned, "service_manifest_digest": digest_for(unsigned)}
    RemoteEvaluatorServiceManifest(result)
    return result


def _operation_digest(request: Mapping[str, Any]) -> str:
    stable = dict(request)
    for field in ("request_digest", "operation_digest", "receipt_sequence_start", "previous_receipt_hash"):
        stable.pop(field, None)
    return digest_for(stable)


def _verify_response_envelope(
    response: Mapping[str, Any], manifest: RemoteEvaluatorServiceManifest, public_key: Any
) -> Dict[str, Any]:
    if set(response) != _RESPONSE_FIELDS or response.get("schema_version") != REMOTE_VARIATION_RESPONSE_SCHEMA:
        raise VariationConfigurationError("remote Variation response is not closed")
    if response.get("service_manifest_digest") != manifest.digest:
        raise VariationConfigurationError("remote Variation response has a stale service binding")
    if response.get("signing_key_id") != manifest["evaluator_key_id"]:
        raise VariationConfigurationError("remote Variation response uses the wrong evaluator key")
    unsigned = dict(response)
    signature = _decode_b64(unsigned.pop("signature"), "response signature", exact_length=64)
    try:
        load_public_key(public_key).verify(signature, canonical_bytes(unsigned))
    except Exception as exc:
        raise VariationConfigurationError("remote Variation response signature is invalid") from exc
    return dict(response)


def _empty_remote_state(manifest: RemoteEvaluatorServiceManifest) -> Dict[str, Any]:
    body = {
        "schema_version": REMOTE_VARIATION_STATE_SCHEMA,
        "service_manifest_digest": manifest.digest,
        "evaluator_key_id": manifest["evaluator_key_id"],
        "next_sequence": 1,
        "receipt_head": GENESIS_HASH,
        "operation_order": [],
        "responses": {},
        "pending_operation": None,
    }
    return {**body, "state_digest": digest_for(body)}


def _load_remote_state(
    path: Path, manifest: RemoteEvaluatorServiceManifest, public_key: Any
) -> Dict[str, Any]:
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        raise VariationDependencyError("remote evaluator has an ambiguous interrupted state commit")
    if not path.exists():
        return _empty_remote_state(manifest)
    if path.is_symlink() or not path.is_file():
        raise VariationDependencyError("remote evaluator state must be a regular file")
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise VariationDependencyError("remote evaluator state cannot be decoded") from exc
    fields = {
        "schema_version", "service_manifest_digest", "evaluator_key_id", "next_sequence",
        "receipt_head", "operation_order", "responses", "pending_operation", "state_digest",
    }
    if not isinstance(state, Mapping) or set(state) != fields:
        raise VariationDependencyError("remote evaluator state is not closed")
    body = dict(state)
    supplied = body.pop("state_digest")
    if supplied != digest_for(body):
        raise VariationDependencyError("remote evaluator state digest is invalid")
    if (
        body["schema_version"] != REMOTE_VARIATION_STATE_SCHEMA
        or body["service_manifest_digest"] != manifest.digest
        or body["evaluator_key_id"] != manifest["evaluator_key_id"]
    ):
        raise VariationDependencyError("remote evaluator state authority binding is stale")
    order = body["operation_order"]
    responses = body["responses"]
    if not isinstance(order, list) or order != list(dict.fromkeys(order)) or not isinstance(responses, Mapping):
        raise VariationDependencyError("remote evaluator operation cache is malformed")
    if set(order) != set(responses):
        raise VariationDependencyError("remote evaluator operation cache index differs from its responses")
    pending = body["pending_operation"]
    if pending is not None:
        pending_fields = {
            "operation_digest", "request_digest", "receipt_sequence_start", "previous_receipt_hash"
        }
        if not isinstance(pending, Mapping) or set(pending) != pending_fields:
            raise VariationDependencyError("remote evaluator pending operation is malformed")
        _require_digest(pending["operation_digest"], "pending operation digest")
        _require_digest(pending["request_digest"], "pending request digest")
        if (
            pending["receipt_sequence_start"] != body["next_sequence"]
            or pending["previous_receipt_hash"] != body["receipt_head"]
            or pending["operation_digest"] in responses
        ):
            raise VariationDependencyError("remote evaluator pending operation anchor is inconsistent")
    sequence = 1
    previous = GENESIS_HASH
    for operation in order:
        response = _verify_response_envelope(responses[operation], manifest, public_key)
        if response["operation_digest"] != operation:
            raise VariationDependencyError("remote evaluator cached operation binding is invalid")
        receipts = response["receipts"]
        if not isinstance(receipts, list) or not receipts:
            raise VariationDependencyError("remote evaluator cached receipt suffix is empty")
        for receipt in receipts:
            previous = verify_receipt(
                receipt,
                public_key,
                expected_key_id=manifest["evaluator_key_id"],
                expected_sequence=sequence,
                expected_previous_hash=previous,
            )
            sequence += 1
    if body["next_sequence"] != sequence or body["receipt_head"] != previous:
        raise VariationDependencyError("remote evaluator cached receipt head is inconsistent")
    return dict(state)


def _write_remote_state(path: Path, state: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(canonical_json(state) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))
    if os.name != "nt":
        directory = os.open(str(path.parent), os.O_RDONLY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)


def _durable_remote_response(
    request: Mapping[str, Any],
    *,
    manifest: RemoteEvaluatorServiceManifest,
    signer: ReceiptSigner,
    state_root: Path,
    build_response: Any,
    validate_response: Any = None,
) -> Mapping[str, Any]:
    root = Path(state_root)
    root.mkdir(parents=True, exist_ok=True)
    state_path = root / "remote-evaluator-state.json"
    with _exclusive_lock(root / "remote-evaluator-state.lock"):
        state = _load_remote_state(state_path, manifest, signer.public_key)
        operation = request["operation_digest"]
        cached = state["responses"].get(operation)
        if cached is not None:
            if validate_response is not None:
                validate_response(cached)
            return dict(cached)
        pending = state["pending_operation"]
        if pending is not None:
            raise VariationDependencyError(
                "remote evaluator is quarantined after an ambiguous interrupted execution"
            )
        if (
            request["receipt_sequence_start"] != state["next_sequence"]
            or request["previous_receipt_hash"] != state["receipt_head"]
        ):
            raise VariationConfigurationError("remote evaluator rejected a stale or forked receipt anchor")
        pending_body = {key: value for key, value in state.items() if key != "state_digest"}
        pending_body["pending_operation"] = {
            "operation_digest": operation,
            "request_digest": request["request_digest"],
            "receipt_sequence_start": request["receipt_sequence_start"],
            "previous_receipt_hash": request["previous_receipt_hash"],
        }
        pending_state = {**pending_body, "state_digest": digest_for(pending_body)}
        # The execution intent is durable before invoking Docker. A crash after this
        # point deliberately quarantines the evaluator instead of re-executing an
        # action whose effects cannot be proven absent.
        _write_remote_state(state_path, pending_state)
        response = _verify_response_envelope(build_response(), manifest, signer.public_key)
        if response["operation_digest"] != operation or response["request_digest"] != request["request_digest"]:
            raise VariationConfigurationError("remote evaluator response is not bound to the current operation")
        sequence = state["next_sequence"]
        previous = state["receipt_head"]
        for receipt in response["receipts"]:
            previous = verify_receipt(
                receipt,
                signer.public_key,
                expected_key_id=manifest["evaluator_key_id"],
                expected_sequence=sequence,
                expected_previous_hash=previous,
            )
            sequence += 1
        if validate_response is not None:
            validate_response(response)
        next_body = {key: value for key, value in state.items() if key != "state_digest"}
        next_body["next_sequence"] = sequence
        next_body["receipt_head"] = previous
        next_body["operation_order"] = list(next_body["operation_order"]) + [operation]
        next_body["responses"] = {**dict(next_body["responses"]), operation: response}
        next_body["pending_operation"] = None
        next_state = {**next_body, "state_digest": digest_for(next_body)}
        _write_remote_state(state_path, next_state)
        return response


def _validate_remote_result_semantics(
    *,
    result_value: Any,
    receipts: Sequence[Mapping[str, Any]],
    manifest: RemoteEvaluatorServiceManifest,
    task_binding: Mapping[str, Any],
    candidate_id: str,
    task_id: str,
    artifact_digest: str,
    declared_locus: str,
    run_id: str,
    arm_policy_digest: str,
) -> EvaluationResult:
    """Validate the complete signed controller contract without mutating the ledger."""

    if not isinstance(result_value, Mapping):
        raise VariationConfigurationError("remote Variation result is not an object")
    try:
        normalized = dict(result_value)
        normalized["receipt_ids"] = tuple(normalized["receipt_ids"])
        result = EvaluationResult(**normalized)
        validate_diagnostic(result.diagnostic_enum)
        validate_resource_bucket(result.resource_bucket)
        validate_disposition(result.disposition)
        _require_digest(result.candidate_artifact_digest, "result candidate artifact digest")
        _require_digest(result.output_digest, "result output digest")
    except (KeyError, TypeError, ValueError) as exc:
        raise VariationConfigurationError("remote Variation result contract is invalid") from exc
    if type(result.infrastructure_loss) is not bool:
        raise VariationConfigurationError("remote Variation infrastructure flag is invalid")
    receipt_ids = tuple(receipt["receipt_id"] for receipt in receipts)
    if (
        result.candidate_id != candidate_id
        or result.task_id != task_id
        or result.candidate_artifact_digest != artifact_digest
        or result.receipt_ids != receipt_ids
    ):
        raise VariationConfigurationError("remote Variation result differs from its signed request or receipts")
    expected_common = {
        "campaign_id": manifest["campaign_id"],
        "run_id": run_id,
        "task_id": task_id,
        "candidate_id": candidate_id,
        "candidate_artifact_digest": artifact_digest,
        "protocol_digest": manifest["protocol_digest"],
        "policy_digest": manifest["policy_digest"],
        "arm_policy_digest": arm_policy_digest,
        "evaluator_digest": manifest.digest,
        "task_family": task_binding["family_id"],
        "normalized_public_locus": task_binding["public_locus"],
        "public_rule_id": task_binding["public_rule_id"],
    }
    for receipt in receipts:
        for field, expected in expected_common.items():
            if receipt.get(field) != expected:
                raise VariationConfigurationError("remote Variation receipt {} binding is invalid".format(field))
    types = [receipt["receipt_type"] for receipt in receipts]
    if types not in (["AUTHORITY"], ["AUTHORITY", "VERDICT", "EFFECT"]):
        raise VariationConfigurationError("remote Variation receipt chain has an invalid type order")
    authority = receipts[0]
    if authority.get("request_id") != "request-authority-{}".format(candidate_id):
        raise VariationConfigurationError("remote Variation authority request identity is invalid")
    empty_digest = digest_bytes(b"")
    if len(receipts) == 1:
        allowed = {
            Diagnostic.PROTOCOL_VIOLATION.value: "REJECTED",
            Diagnostic.MUTATION_LOCUS_VIOLATION.value: "REJECTED",
            Diagnostic.AUTHORITY_DENIED.value: "ABSTAINED",
            Diagnostic.INTERNAL_ERROR.value: "ABSTAINED",
        }
        if (
            authority.get("decision") != "DENY"
            or result.diagnostic_enum not in allowed
            or result.disposition != allowed[result.diagnostic_enum]
            or result.resource_bucket != "UNDER_25"
            or result.output_digest != empty_digest
        ):
            raise VariationConfigurationError("remote Variation authority-only result is inconsistent")
        receipt_diagnostic = authority.get("diagnostic_enum")
        if result.diagnostic_enum == Diagnostic.AUTHORITY_DENIED.value:
            if receipt_diagnostic is not None:
                raise VariationConfigurationError("authority denial receipt disclosed an invalid diagnostic")
        elif receipt_diagnostic != result.diagnostic_enum:
            raise VariationConfigurationError("authority denial diagnostic differs from its result")
    else:
        verdict, effect = receipts[1:]
        expected_action = digest_for({"action": "execute_candidate", "locus": declared_locus})
        diagnostic = result.diagnostic_enum
        infrastructure = diagnostic == Diagnostic.INTERNAL_ERROR.value
        expected_verdict = "ERROR" if infrastructure else ("PASS" if diagnostic == Diagnostic.PASS.value else "FAIL")
        expected_effect = "ERROR" if infrastructure else "ALLOW"
        expected_disposition = (
            "ABSTAINED" if infrastructure else ("PROMOTED" if diagnostic == Diagnostic.PASS.value else "REJECTED")
        )
        if (
            authority.get("decision") != "ALLOW"
            or verdict.get("request_id") != "request-verdict-{}".format(candidate_id)
            or effect.get("request_id") != "request-effect-{}".format(candidate_id)
            or verdict.get("decision") != expected_verdict
            or effect.get("decision") != expected_effect
            or verdict.get("diagnostic_enum") != diagnostic
            or effect.get("diagnostic_enum") != diagnostic
            or verdict.get("resource_bucket") != result.resource_bucket
            or verdict.get("output_digest") != result.output_digest
            or effect.get("normalized_action_hash") != expected_action
            or result.disposition != expected_disposition
        ):
            raise VariationConfigurationError("remote Variation result disagrees with its signed controller chain")
    incident = result.infrastructure_incident_id
    root = result.failure_family_root
    if result.diagnostic_enum == Diagnostic.INTERNAL_ERROR.value:
        if not result.infrastructure_loss or not incident or not root:
            raise VariationConfigurationError("remote Variation infrastructure loss is incomplete")
        expected_root = failure_family_root(
            task_binding["family_id"],
            result.diagnostic_enum,
            task_binding["public_locus"],
            task_binding["public_rule_id"],
            infrastructure_incident_id=incident,
        )
        if root != expected_root:
            raise VariationConfigurationError("remote Variation infrastructure root is invalid")
        for receipt in receipts:
            if receipt.get("infrastructure_incident_id") != incident or receipt.get("failure_family_root") != root:
                raise VariationConfigurationError("remote Variation receipt incident binding is inconsistent")
    elif result.infrastructure_loss or incident is not None or root is not None:
        raise VariationConfigurationError("remote Variation non-infrastructure result carries an incident")
    return result


class RemoteControllerEvaluationGateway:
    """Exact production client for a separately administered evaluator command."""

    enforceable = True

    def __init__(
        self,
        *,
        ledger: EvidenceLedger,
        manifest_path: Path,
        public_key_path: Path,
        command: Path,
        python_executable: Optional[Path] = None,
        python_digest: Optional[str] = None,
        timeout_seconds: int = REMOTE_VARIATION_TIMEOUT_SECONDS,
    ) -> None:
        if type(self) is not RemoteControllerEvaluationGateway:
            raise VariationDependencyError("remote Variation evaluator type is not frozen")
        if type(ledger) is not EvidenceLedger:
            raise VariationDependencyError("remote Variation evaluator requires the authoritative EvidenceLedger")
        paths = tuple(Path(item) for item in (manifest_path, public_key_path, command))
        if any(path.is_symlink() or not path.is_file() for path in paths):
            raise VariationDependencyError("remote evaluator manifest, key, and command must be regular files")
        if timeout_seconds != REMOTE_VARIATION_TIMEOUT_SECONDS:
            raise VariationConfigurationError("remote evaluator timeout differs from the frozen boundary")
        if (python_executable is None) != (python_digest is None):
            raise VariationConfigurationError("remote evaluator Python path and digest must be bound together")
        manifest = RemoteEvaluatorServiceManifest.from_path(paths[0])
        public_key = paths[1].read_bytes()
        if hashlib.sha256(public_key).hexdigest() != manifest["evaluator_public_key_digest"]:
            raise VariationConfigurationError("remote evaluator public key differs from the service manifest")
        if key_id_for_public_key(public_key) != manifest["evaluator_key_id"]:
            raise VariationConfigurationError("remote evaluator key ID differs from its public key")
        command_bytes = paths[2].read_bytes()
        if hashlib.sha256(command_bytes).hexdigest() != manifest["command_digest"]:
            raise VariationConfigurationError("remote evaluator command differs from the service manifest")
        self.ledger = ledger
        self.manifest = manifest
        self._public_key = public_key
        self._command = paths[2].resolve()
        self._command_bytes = command_bytes
        self._command_suffix = paths[2].suffix.lower()
        self._python_executable = None if python_executable is None else Path(python_executable)
        self._python_digest = None
        if self._python_executable is not None:
            self._python_digest = _require_digest(python_digest, "remote evaluator Python executable digest")
            with pinned_python_invocation(self._python_executable, self._python_digest):
                pass
        self.evaluator_revision = manifest["evaluator_revision"]
        self.evaluator_digest = manifest["service_manifest_digest"]
        self.task_registry = manifest
        self._frozen_contract = digest_for(
            {
                "manifest": manifest.digest,
                "key": hashlib.sha256(public_key).hexdigest(),
                "command": hashlib.sha256(command_bytes).hexdigest(),
                "python": self._python_digest,
            }
        )
        self._pinned_ledger = ledger
        self._frozen = True

    def __setattr__(self, name: str, value: Any) -> None:
        if getattr(self, "_frozen", False):
            raise AttributeError("RemoteControllerEvaluationGateway is immutable")
        object.__setattr__(self, name, value)

    def validate_campaign_bindings(
        self,
        *,
        campaign_id: str,
        model_digest: str,
        protocol_digest: str,
        policy_digest: str,
        data_manifest_digest: str,
    ) -> None:
        expected = {
            "campaign_id": campaign_id,
            "model_digest": model_digest,
            "protocol_digest": protocol_digest,
            "policy_digest": policy_digest,
            "data_manifest_digest": data_manifest_digest,
        }
        for field, value in expected.items():
            if self.manifest[field] != value:
                raise VariationConfigurationError("remote evaluator has a stale {} binding".format(field))

    def validate_runtime(self) -> None:
        if type(self) is not RemoteControllerEvaluationGateway:
            raise VariationDependencyError("remote Variation evaluator type changed")
        if "evaluate" in self.__dict__ or "validate_runtime" in self.__dict__:
            raise VariationDependencyError("remote Variation evaluator methods cannot be overridden")
        if type(self).evaluate is not _ORIGINAL_REMOTE_EVALUATE or type(self).validate_runtime is not _ORIGINAL_REMOTE_VALIDATE:
            raise VariationDependencyError("remote Variation evaluator implementation changed")
        if type(self)._invoke is not _ORIGINAL_REMOTE_INVOKE:
            raise VariationDependencyError("remote Variation evaluator command boundary changed")
        if self.ledger is not self._pinned_ledger or type(self.ledger) is not EvidenceLedger:
            raise VariationDependencyError("remote Variation evaluator ledger binding changed")
        self.manifest.validate_integrity()
        current = self._command.read_bytes()
        if current != self._command_bytes or hashlib.sha256(current).hexdigest() != self.manifest["command_digest"]:
            raise VariationDependencyError("remote evaluator command changed after construction")
        if self._python_executable is not None and self._python_digest is not None:
            with pinned_python_invocation(self._python_executable, self._python_digest):
                pass
        if digest_for(
            {
                "manifest": self.manifest.digest,
                "key": hashlib.sha256(self._public_key).hexdigest(),
                "command": hashlib.sha256(current).hexdigest(),
                "python": self._python_digest,
            }
        ) != self._frozen_contract:
            raise VariationDependencyError("remote evaluator frozen contract changed")

    def _invoke(self, request: Mapping[str, Any]) -> Mapping[str, Any]:
        self.validate_runtime()
        request_text = canonical_json(request)
        if len(request_text.encode("utf-8")) > REMOTE_VARIATION_REQUEST_LIMIT:
            raise VariationConfigurationError("remote Variation request exceeds the bounded input limit")
        with tempfile.TemporaryDirectory(prefix="egv-pinned-variation-endpoint-") as temporary:
            endpoint = Path(temporary) / (self.manifest["command_digest"] + self._command_suffix)
            with endpoint.open("xb") as handle:
                handle.write(self._command_bytes)
                handle.flush()
                os.fsync(handle.fileno())
            endpoint.chmod(0o500)
            if hashlib.sha256(endpoint.read_bytes()).hexdigest() != self.manifest["command_digest"]:
                raise VariationDependencyError("content-addressed remote evaluator copy failed verification")
            if self._python_executable is not None and self._python_digest is not None:
                with pinned_python_invocation(self._python_executable, self._python_digest) as (
                    pinned_python,
                    popen_kwargs,
                ):
                    returncode, stdout, _stderr = _run_bounded_command(
                        [pinned_python, "-I", "-S", str(endpoint)],
                        request_text,
                        popen_kwargs=popen_kwargs,
                    )
            else:
                invocation = [sys.executable, str(endpoint)] if self._command_suffix == ".py" else [str(endpoint)]
                returncode, stdout, _stderr = _run_bounded_command(invocation, request_text)
        if returncode != 0:
            raise VariationDependencyError("remote Variation evaluator returned a nonzero exit status")
        try:
            response = json.loads(stdout.decode("utf-8"))
        except (UnicodeError, ValueError) as exc:
            raise VariationDependencyError("remote Variation evaluator returned invalid JSON") from exc
        if not isinstance(response, Mapping):
            raise VariationDependencyError("remote Variation evaluator returned no result object")
        return response

    def evaluate(
        self,
        *,
        candidate_id: str,
        task_id: str,
        source: bytes,
        opaque_input: Any,
        requested_authority: str,
        declared_locus: str,
        candidate_source_path: Optional[str] = None,
    ) -> EvaluationResult:
        del candidate_source_path
        if opaque_input is not None:
            raise VariationConfigurationError("remote evaluator boundary cannot receive evaluator-private input")
        task_binding = self.manifest.public_record(task_id)
        if task_binding is None:
            raise VariationConfigurationError("remote evaluator task is outside the frozen public registry")
        if task_binding["public_locus"] != declared_locus:
            raise VariationConfigurationError("remote evaluator task locus differs from the public registry")
        source_bytes = bytes(source)
        from .loop import CANDIDATE_SOURCE_LIMIT

        if not source_bytes or len(source_bytes) > CANDIDATE_SOURCE_LIMIT:
            raise VariationConfigurationError("remote Variation candidate source exceeds the frozen byte ceiling")
        artifact_digest = digest_bytes(source_bytes)
        candidate = self.ledger.connection.execute(
            "SELECT campaign_id,run_id,task_id,candidate_json FROM candidates WHERE candidate_id=?",
            (candidate_id,),
        ).fetchone()
        if candidate is None:
            run_id = "run-evaluation"
            arm_policy_digest = digest_for("unbound-remote-evaluation-arm")
        else:
            try:
                candidate_value = json.loads(candidate["candidate_json"])
                if candidate["campaign_id"] != self.manifest["campaign_id"] or candidate["task_id"] != task_id:
                    raise ValueError("candidate authority binding differs")
                if candidate_value["metadata"]["candidate_artifact_digest"] != artifact_digest:
                    raise ValueError("candidate artifact binding differs")
                arm_id = candidate_value["metadata"]["arm_id"]
                from .arms import arm_policy

                arm_policy_digest = arm_policy(str(arm_id)).digest
            except (KeyError, TypeError, ValueError) as exc:
                raise VariationConfigurationError("remote Variation candidate arm binding is invalid") from exc
            run_id = str(candidate["run_id"])
        stable_body = {
            "schema_version": REMOTE_VARIATION_REQUEST_SCHEMA,
            "service_manifest_digest": self.manifest.digest,
            "campaign_id": self.manifest["campaign_id"],
            "model_digest": self.manifest["model_digest"],
            "protocol_digest": self.manifest["protocol_digest"],
            "policy_digest": self.manifest["policy_digest"],
            "run_id": run_id,
            "arm_policy_digest": arm_policy_digest,
            "data_manifest_digest": self.manifest["data_manifest_digest"],
            "task_manifest_digest": self.manifest["task_manifest_digest"],
            "evaluator_digest": self.manifest["evaluator_digest"],
            "docker_image_digest": self.manifest["docker_image_digest"],
            "candidate_id": candidate_id,
            "task_id": task_id,
            "public_task_binding": task_binding,
            "candidate_artifact_digest": artifact_digest,
            "candidate_source_b64": _encode_b64(source_bytes),
            "requested_authority": requested_authority,
            "declared_locus": declared_locus,
        }
        operation = _operation_digest(stable_body)
        body = {
            **stable_body,
            "operation_digest": operation,
            "receipt_sequence_start": self.ledger.receipt_next_sequence(),
            "previous_receipt_hash": self.ledger.receipt_head(),
        }
        request = {**body, "request_digest": digest_for(body)}
        response = dict(self._invoke(request))
        response = _verify_response_envelope(response, self.manifest, self._public_key)
        if response["operation_digest"] != operation:
            raise VariationConfigurationError("remote Variation response has a stale operation binding")
        receipts = response["receipts"]
        if not isinstance(receipts, list) or not receipts:
            raise VariationConfigurationError("remote Variation response receipt chain is empty")
        first = receipts[0]
        if not isinstance(first, Mapping):
            raise VariationConfigurationError("remote Variation receipt is not an object")
        sequence = first.get("sequence")
        previous = first.get("previous_receipt_hash")
        if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 1:
            raise VariationConfigurationError("remote Variation receipt sequence is invalid")
        if previous != GENESIS_HASH:
            _require_digest(previous, "remote Variation previous receipt hash")
        receipt_values = []
        for receipt in receipts:
            if not isinstance(receipt, Mapping):
                raise VariationConfigurationError("remote Variation receipt is not an object")
            receipt_value = dict(receipt)
            complete = verify_receipt(
                receipt_value,
                self._public_key,
                expected_key_id=self.manifest["evaluator_key_id"],
                expected_sequence=sequence,
                expected_previous_hash=previous,
            )
            receipt_values.append(receipt_value)
            previous = complete
            sequence += 1
        result = _validate_remote_result_semantics(
            result_value=response["result"],
            receipts=receipt_values,
            manifest=self.manifest,
            task_binding=task_binding,
            candidate_id=candidate_id,
            task_id=task_id,
            artifact_digest=artifact_digest,
            declared_locus=declared_locus,
            run_id=run_id,
            arm_policy_digest=arm_policy_digest,
        )
        stored = [self.ledger.receipt_by_id(receipt["receipt_id"]) for receipt in receipt_values]
        present = [item is not None for item in stored]
        if any(present) and not all(present):
            raise VariationConfigurationError("remote Variation receipt suffix is only partially present")
        if all(present):
            for existing, receipt in zip(stored, receipt_values):
                assert existing is not None
                if canonical_bytes(existing["receipt"]) != canonical_bytes(receipt):
                    raise VariationConfigurationError("remote Variation cached receipt differs from the ledger")
            return result
        if response["request_digest"] != request["request_digest"]:
            raise VariationConfigurationError("remote Variation response has a stale request binding")
        if (
            receipt_values[0]["previous_receipt_hash"] != request["previous_receipt_hash"]
            or receipt_values[0]["sequence"] != request["receipt_sequence_start"]
            or self.ledger.receipt_head() != request["previous_receipt_hash"]
            or self.ledger.receipt_next_sequence() != request["receipt_sequence_start"]
        ):
            raise VariationConfigurationError("authoritative receipt head changed during remote evaluation")
        self.ledger.ingest_receipts_atomic(receipt_values, self._public_key)
        return result


_ORIGINAL_REMOTE_EVALUATE = RemoteControllerEvaluationGateway.evaluate
_ORIGINAL_REMOTE_VALIDATE = RemoteControllerEvaluationGateway.validate_runtime
_ORIGINAL_REMOTE_INVOKE = RemoteControllerEvaluationGateway._invoke


def run_remote_evaluator_once(
    request: Mapping[str, Any],
    *,
    service_manifest: Path,
    evaluator_seed: Path,
    evaluator_private_key: Path,
    workspace: Path,
    state_root: Path,
) -> Mapping[str, Any]:
    """Run one request with evaluator-owned hidden inputs and Docker authority."""

    if not isinstance(request, Mapping) or set(request) != _REQUEST_FIELDS:
        raise VariationConfigurationError("remote Variation request is not closed")
    request_value = dict(request)
    if len(canonical_bytes(request_value)) > REMOTE_VARIATION_REQUEST_LIMIT:
        raise VariationConfigurationError("remote Variation request exceeds the bounded input limit")
    if request_value.get("operation_digest") != _operation_digest(request_value):
        raise VariationConfigurationError("remote Variation operation digest is invalid")
    unsigned_request = dict(request_value)
    supplied_request_digest = unsigned_request.pop("request_digest")
    if digest_for(unsigned_request) != supplied_request_digest:
        raise VariationConfigurationError("remote Variation request digest is invalid")
    manifest = RemoteEvaluatorServiceManifest.from_path(service_manifest)
    for field in (
        "service_manifest_digest",
        "campaign_id",
        "model_digest",
        "protocol_digest",
        "policy_digest",
        "data_manifest_digest",
        "task_manifest_digest",
        "evaluator_digest",
        "docker_image_digest",
    ):
        expected = manifest.digest if field == "service_manifest_digest" else manifest[field]
        if request_value[field] != expected:
            raise VariationConfigurationError("remote Variation request {} binding is stale".format(field))
    if not isinstance(request_value["run_id"], str) or not request_value["run_id"]:
        raise VariationConfigurationError("remote Variation run binding is invalid")
    _require_digest(request_value["arm_policy_digest"], "remote Variation arm policy digest")
    task_binding = manifest.public_record(str(request_value["task_id"]))
    if task_binding is None or request_value["public_task_binding"] != task_binding:
        raise VariationConfigurationError("remote Variation request task binding is invalid")
    from .loop import CANDIDATE_SOURCE_LIMIT, ControllerEvaluationGateway

    encoded_source = request_value["candidate_source_b64"]
    encoded_limit = ((CANDIDATE_SOURCE_LIMIT + 2) // 3) * 4
    if not isinstance(encoded_source, str) or len(encoded_source) > encoded_limit:
        raise VariationConfigurationError("remote Variation encoded candidate exceeds the frozen byte ceiling")
    source = _decode_b64(encoded_source, "candidate source")
    if len(source) > CANDIDATE_SOURCE_LIMIT or digest_bytes(source) != request_value["candidate_artifact_digest"]:
        raise VariationConfigurationError("remote Variation candidate bytes violate the sealed request")
    seed_path = Path(evaluator_seed)
    key_path = Path(evaluator_private_key)
    if seed_path.is_symlink() or not seed_path.is_file() or key_path.is_symlink() or not key_path.is_file():
        raise VariationDependencyError("remote evaluator seed and private key must be regular local files")
    signer = ReceiptSigner(key_path.read_bytes())
    if signer.key_id != manifest["evaluator_key_id"]:
        raise VariationConfigurationError("remote evaluator private key differs from the service authority")
    from ..evaluation.authority import AuthorityBroker, AuthorityPolicy, DockerEnforcedRuntime
    from ..evaluation.dataset import EvaluationCorpus
    from ..evaluation.sandbox import DockerCandidateSandbox, DockerSandboxConfig

    corpus = EvaluationCorpus.generate(secret_seed_file=seed_path)
    if corpus.manifest_digest() != manifest["data_manifest_digest"]:
        raise VariationConfigurationError("remote evaluator private corpus differs from the frozen manifest")
    repo = corpus.get(str(request_value["task_id"]))
    if repo.public_manifest_record() != task_binding or repo.split not in {"train", "heldout"}:
        raise VariationConfigurationError("remote evaluator private task differs from its public binding")
    config = DockerSandboxConfig.from_environment()
    if config.pinned_image_id != manifest["docker_image_digest"] or digest_for(dict(config.__dict__)) != manifest["docker_config_digest"]:
        raise VariationConfigurationError("remote evaluator Docker configuration differs from the service manifest")
    policy = AuthorityPolicy.candidate_execution()
    if policy.digest != manifest["authority_policy_digest"] or policy.digest != manifest["policy_digest"]:
        raise VariationConfigurationError("remote evaluator authority policy differs from the frozen campaign")
    runtime = DockerEnforcedRuntime(config)
    sandbox = DockerCandidateSandbox(Path(workspace), config=config)
    # Construct the oracle runner from the evaluator-owned sealed corpus for
    # exactly this public task address. HiddenEvaluatorRunner.from_corpus is the
    # held-out campaign runner and intentionally omits TRAIN; the remote service
    # must support TRAIN trajectories without widening that held-out API.
    try:
        runner = HiddenEvaluatorRunner.for_remote_task(
            corpus,
            repo.template_id,
            evaluator_revision=manifest["evaluator_revision"],
        )
    except ValueError as exc:
        raise VariationConfigurationError("remote evaluator could not bind its private task oracle") from exc
    sequence = request_value["receipt_sequence_start"]
    if not isinstance(sequence, int) or isinstance(sequence, bool) or sequence < 1:
        raise VariationConfigurationError("remote Variation receipt sequence anchor is invalid")
    previous = request_value["previous_receipt_hash"]
    if previous != GENESIS_HASH:
        _require_digest(previous, "previous receipt hash")
    def build_response() -> Mapping[str, Any]:
        collected = []
        with tempfile.TemporaryDirectory(prefix="egv-remote-receipts-") as journal_root:
            journal = ReceiptJournal(Path(journal_root) / "receipts.jsonl", signer.public_key)
            controller = EvaluatorController(
                sandbox=sandbox,
                hidden_runner=runner,
                broker=AuthorityBroker(runtime, policy),
                signer=signer,
                journal=journal,
                ingest=lambda receipt: collected.append(dict(receipt)),
                campaign_id=manifest["campaign_id"],
                protocol_digest=manifest["protocol_digest"],
                policy_digest=manifest["policy_digest"],
            )
            local_result = ControllerEvaluationGateway(controller).evaluate(
                candidate_id=str(request_value["candidate_id"]),
                task_id=repo.template_id,
                source=source,
                opaque_input=repo.evaluator_input,
                requested_authority=str(request_value["requested_authority"]),
                declared_locus=str(request_value["declared_locus"]),
            )
        signed_receipts = []
        current_sequence = sequence
        current_previous = previous
        for original in collected:
            fields = dict(original)
            for key in (
                "schema_version", "sequence", "previous_receipt_hash", "idempotency_key",
                "signing_key_id", "receipt_id", "signature",
            ):
                fields.pop(key, None)
            fields["evaluator_digest"] = manifest.digest
            fields["run_id"] = request_value["run_id"]
            fields["arm_policy_digest"] = request_value["arm_policy_digest"]
            receipt = signer.sign_receipt(
                fields,
                sequence=current_sequence,
                previous_receipt_hash=current_previous,
                idempotency_key=content_id(
                    "remote-variation-receipt",
                    {"operation_digest": request_value["operation_digest"], "receipt_type": fields["receipt_type"]},
                ),
            )
            signed_receipts.append(receipt)
            current_previous = receipt_hash(receipt)
            current_sequence += 1
        result_value = local_result.to_dict()
        result_value["receipt_ids"] = [receipt["receipt_id"] for receipt in signed_receipts]
        response_unsigned = {
            "schema_version": REMOTE_VARIATION_RESPONSE_SCHEMA,
            "operation_digest": request_value["operation_digest"],
            "request_digest": supplied_request_digest,
            "service_manifest_digest": manifest.digest,
            "result": result_value,
            "receipts": signed_receipts,
            "signing_key_id": signer.key_id,
        }
        return {**response_unsigned, "signature": signer.sign_bytes(canonical_bytes(response_unsigned))}

    return _durable_remote_response(
        request_value,
        manifest=manifest,
        signer=signer,
        state_root=state_root,
        build_response=build_response,
        validate_response=lambda response: _validate_remote_result_semantics(
            result_value=response["result"],
            receipts=response["receipts"],
            manifest=manifest,
            task_binding=task_binding,
            candidate_id=str(request_value["candidate_id"]),
            task_id=repo.template_id,
            artifact_digest=str(request_value["candidate_artifact_digest"]),
            declared_locus=str(request_value["declared_locus"]),
            run_id=str(request_value["run_id"]),
            arm_policy_digest=str(request_value["arm_policy_digest"]),
        ),
    )


__all__ = [
    "REMOTE_VARIATION_REQUEST_SCHEMA",
    "REMOTE_VARIATION_RESPONSE_SCHEMA",
    "REMOTE_VARIATION_SERVICE_SCHEMA",
    "RemoteControllerEvaluationGateway",
    "RemoteEvaluatorServiceManifest",
    "build_remote_evaluator_service_manifest",
    "run_remote_evaluator_once",
]
