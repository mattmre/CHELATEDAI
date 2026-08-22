"""Docker-enforced candidate isolation and explicitly test-only local helper.

Production Evaluation never uses the local helper.  The Docker adapter copies
one content-addressed source file into an ephemeral container, sends one opaque
JSON input on stdin, and receives only bounded stdout.  Hidden evaluator files,
host paths, credentials, the Docker socket, and the ledger are not mounted.
"""

from __future__ import annotations

import ast
from dataclasses import dataclass, field
import hashlib
import hmac
import json
import os
from pathlib import Path
import re
import secrets
import shutil
import subprocess
import sys
import tempfile
import time
from typing import Any, Dict, List, Mapping, Optional, Union
import uuid

from ..canonical import canonical_bytes, content_id, digest_bytes, digest_for
from .artifacts import ContentAddressedArtifactStore
from .diagnostics import Diagnostic
from .errors import ArtifactError, DockerConfigurationError, InfrastructureFailure


DEFAULT_DOCKER_IMAGE = "edc-backbone:live"
DEFAULT_DOCKER_IMAGE_ID = "sha256:06f0c2fa1954db8df5399ff935362563f0a74d5641e3240c42b2ff0aedcce156"
DOCKER_USER = "65534:65534"
DOCKER_PIDS_LIMIT = 64
DOCKER_MEMORY_LIMIT = "128m"
RESOURCE_BOUND_MEMORY_LIMIT = "64m"
DOCKER_TMPFS = "16m"
DOCKER_TIMEOUT_SECONDS = 2.0
DOCKER_OUTPUT_LIMIT = 65536

# The outer profile must permit the pinned Python runtime to load the source
# artifact and start the runner.  It is still deny-default: capabilities not
# named here are rejected by Docker before the candidate starts.  The inner
# candidate-phase filter below removes the EGV-dangerous subset after source
# and opaque input are loaded.
_DOCKER_STARTUP_ALLOWLIST = (
    "read", "write", "close", "fstat", "newfstatat", "lseek", "mmap", "mprotect", "munmap", "brk",
    "rt_sigaction", "rt_sigprocmask", "rt_sigreturn", "ioctl", "pread64", "readv", "writev",
    "faccessat", "faccessat2", "pipe2", "fcntl", "dup", "dup3", "futex", "set_tid_address",
    "set_robust_list", "rseq", "prlimit64", "getrandom", "clock_gettime", "clock_getres",
    "clock_nanosleep", "getpid", "getppid", "gettid", "getuid", "geteuid", "getgid", "getegid",
    "uname", "sched_getaffinity", "exit", "exit_group", "wait4", "getcwd", "statx", "getdents64",
    "getrlimit", "getrusage", "sysinfo", "times", "madvise", "mremap", "gettimeofday", "nanosleep",
    "sigaltstack", "prctl", "chdir", "fchdir", "fstatfs", "fsync", "fdatasync", "epoll_create1",
    "epoll_ctl", "epoll_pwait", "eventfd2", "getgroups", "getitimer", "getpeername", "getsid",
    "getsockname", "getsockopt", "ppoll", "pselect6", "setresuid", "setresgid", "setgroups",
    "setpgid", "setsid", "setuid", "setgid", "setreuid", "setregid", "umask", "utimensat",
    "openat", "openat2", "execve", "execveat", "clone", "clone3", "socket", "socketpair", "connect",
    "bind", "listen", "accept", "accept4", "renameat", "renameat2", "linkat", "unlinkat", "truncate",
    "ftruncate", "fchmod", "fchmodat", "fchmodat2", "mkdirat", "symlinkat", "mknodat", "kill", "tkill",
    "tgkill", "ptrace", "process_vm_readv", "process_vm_writev", "readlinkat", "open_by_handle_at",
    "mount", "umount2", "pivot_root", "unshare", "setns", "io_uring_setup", "io_uring_enter",
    "io_uring_register", "memfd_create", "memfd_secret",
)


@dataclass(frozen=True)
class DockerSandboxConfig:
    """Pinned, deterministic Docker policy; no pull is ever attempted."""

    image_ref: str = DEFAULT_DOCKER_IMAGE
    pinned_image_id: str = DEFAULT_DOCKER_IMAGE_ID
    docker_binary: str = "docker"
    user: str = DOCKER_USER
    pids_limit: int = DOCKER_PIDS_LIMIT
    memory_limit: str = DOCKER_MEMORY_LIMIT
    tmpfs_size: str = DOCKER_TMPFS
    timeout_seconds: float = DOCKER_TIMEOUT_SECONDS
    output_limit: int = DOCKER_OUTPUT_LIMIT

    @classmethod
    def from_environment(cls) -> "DockerSandboxConfig":
        return cls(
            image_ref=os.environ.get("EGV_DOCKER_IMAGE", DEFAULT_DOCKER_IMAGE),
            pinned_image_id=os.environ.get("EGV_DOCKER_IMAGE_ID", DEFAULT_DOCKER_IMAGE_ID),
        )

    def validate(self) -> None:
        if not self.image_ref or any(character.isspace() for character in self.image_ref) or self.image_ref.startswith("-"):
            raise DockerConfigurationError("Docker image reference is invalid")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", self.pinned_image_id):
            raise DockerConfigurationError("Docker image must be pinned to a SHA-256 image ID")
        if self.user != DOCKER_USER:
            raise DockerConfigurationError("Docker candidate must run as uid/gid 65534")
        if (
            self.pids_limit != DOCKER_PIDS_LIMIT
            or self.memory_limit != DOCKER_MEMORY_LIMIT
            or self.tmpfs_size != DOCKER_TMPFS
            or self.timeout_seconds != DOCKER_TIMEOUT_SECONDS
            or self.output_limit != DOCKER_OUTPUT_LIMIT
        ):
            raise DockerConfigurationError("Docker resource limits differ from the frozen Evaluation contract")

    def verify_image(self) -> str:
        self.validate()
        binary = shutil.which(self.docker_binary)
        if binary is None:
            raise DockerConfigurationError("Docker executable is unavailable")
        try:
            completed = subprocess.run(
                [binary, "image", "inspect", self.image_ref],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise DockerConfigurationError("Docker image inspection failed") from exc
        if completed.returncode != 0:
            raise DockerConfigurationError("pinned Docker image is not cached; pull is forbidden")
        try:
            records = json.loads(completed.stdout)
            actual = str(records[0]["Id"])
        except (ValueError, IndexError, KeyError, TypeError) as exc:
            raise DockerConfigurationError("Docker image inspection returned an invalid record") from exc
        if actual != self.pinned_image_id:
            raise DockerConfigurationError("cached Docker image ID does not match the pinned contract")
        return actual

    def seccomp_profile(self, root: Path) -> Path:
        """Create a deterministic host-side profile; it is never mounted."""

        profile_path = root / "docker-seccomp-egv-v1.json"
        profile = {
            "defaultAction": "SCMP_ACT_ERRNO",
            "architectures": ["SCMP_ARCH_AARCH64", "SCMP_ARCH_X86_64"],
            "syscalls": [
                {
                    "names": list(_DOCKER_STARTUP_ALLOWLIST),
                    "action": "SCMP_ACT_ALLOW",
                }
            ],
        }
        encoded = (json.dumps(profile, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8")
        if profile_path.exists() and profile_path.read_bytes() != encoded:
            raise DockerConfigurationError("Docker seccomp profile was modified")
        if not profile_path.exists():
            profile_path.write_bytes(encoded)
            profile_path.chmod(0o444)
        return profile_path


@dataclass(frozen=True)
class SandboxResult:
    sandbox_id: str
    artifact_digest: str
    diagnostic_enum: str
    resource_bucket: str
    output_bytes: bytes
    exit_status: str
    elapsed_millis: int
    environment_diff: Mapping[str, Any] = field(default_factory=dict)
    incident_id: Optional[str] = None
    # The low-level adversarial contract is an evidence transport only.  It
    # is intentionally represented in the result type so callers cannot
    # mistake a successful kernel probe for a public candidate decision.
    evidence_only: bool = False

    @property
    def output_digest(self) -> str:
        return digest_bytes(self.output_bytes)

    @property
    def environment_diff_digest(self) -> str:
        return digest_for(dict(self.environment_diff))

    @property
    def exit_status_class(self) -> str:
        """Return the closed public status class, never raw runtime text."""

        if self.diagnostic_enum == Diagnostic.TIMEOUT.value:
            return "TIMEOUT"
        if self.exit_status == "OUTPUT_LIMIT":
            return "OUTPUT_LIMIT"
        if self.diagnostic_enum == Diagnostic.RESOURCE_LIMIT.value:
            return "RESOURCE_LIMIT"
        if self.diagnostic_enum == Diagnostic.INTERNAL_ERROR.value:
            return "INFRASTRUCTURE_LOSS"
        if self.exit_status == "SUCCESS" and self.diagnostic_enum == Diagnostic.PASS.value:
            return "SUCCESS"
        if self.exit_status.startswith("SIGNAL") or self.exit_status in {"SIGKILL", "SIGTERM"}:
            return "SIGNAL"
        return "NONZERO"


def _incident(sandbox_id: str, reason: str) -> str:
    return content_id("incident", {"sandbox_id": sandbox_id, "reason": reason, "nonce": uuid.uuid4().hex})


# The public production path uses a deliberately narrow executable contract.
# It is the last trusted check before arbitrary candidate bytes reach Docker:
# generated Evaluation candidates are pure functions of their opaque input,
# with no imports, ambient capability lookup, frame access, process control,
# or direct output.  The low-level Docker adapter has an explicitly named
# adversarial mode so kernel controls can still be exercised, but no such
# source can receive a production PASS.
_PURE_RETURN_ALLOWED_NODES = frozenset(
    {
        "Add", "And", "Assign", "Attribute", "BinOp", "BoolOp", "Call", "Compare", "Constant", "Dict",
        "Eq", "Expr", "For", "FunctionDef", "If", "IfExp", "IsNot", "Lambda", "List", "ListComp", "Load",
        "Module", "Name", "NotEq", "NotIn", "Return", "Slice", "Store", "Subscript", "Tuple", "arg",
        "arguments", "comprehension", "keyword",
    }
)
_PURE_RETURN_ALLOWED_BUILTINS = frozenset(
    {
        "all", "any", "bool", "dict", "enumerate", "filter", "float", "int", "len", "list", "map", "max",
        "min", "range", "repr", "round", "set", "sorted", "str", "sum", "tuple", "type", "zip",
    }
)
_PURE_RETURN_ALLOWED_ATTRIBUTES = frozenset(
    {"append", "format", "get", "items", "join", "split", "startswith", "update"}
)
_PURE_RETURN_FORBIDDEN_NAMES = frozenset(
    {
        "__import__", "breakpoint", "compile", "dir", "eval", "exec", "getattr", "globals", "help", "input",
        "locals", "memoryview", "open", "setattr", "vars",
    }
)


def validate_candidate_source_contract(source: bytes) -> Optional[str]:
    """Return a bounded reason when source is outside ``pure-return-v1``.

    This is a production admission and decision precondition for the narrow
    Evaluation source shape, not an isolation boundary. Docker is the
    enforceable isolation boundary. Docker return codes, stdout, and
    low-level probe output remain untrusted evidence; the evaluator-private
    hidden oracle is the decision authority after this check.
    """

    try:
        text = source.decode("utf-8")
        tree = ast.parse(text, mode="exec")
    except (UnicodeDecodeError, SyntaxError):
        return "source is not valid UTF-8 Python"
    function_names = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
    if "main" not in function_names:
        return "pure-return-v1 requires a main(value) function"
    for node in ast.walk(tree):
        node_name = type(node).__name__
        if node_name not in _PURE_RETURN_ALLOWED_NODES:
            return "pure-return-v1 rejects AST node {}".format(node_name)
        if isinstance(node, ast.Name):
            if node.id in _PURE_RETURN_FORBIDDEN_NAMES or node.id.startswith("__"):
                return "pure-return-v1 rejects capability name {}".format(node.id[:48])
        elif isinstance(node, ast.Attribute):
            if node.attr not in _PURE_RETURN_ALLOWED_ATTRIBUTES:
                return "pure-return-v1 rejects attribute {}".format(node.attr[:48])
        elif isinstance(node, ast.Call):
            function = node.func
            if isinstance(function, ast.Name):
                if function.id not in _PURE_RETURN_ALLOWED_BUILTINS and function.id not in function_names:
                    return "pure-return-v1 rejects call {}".format(function.id[:48])
            elif isinstance(function, ast.Attribute):
                if function.attr not in _PURE_RETURN_ALLOWED_ATTRIBUTES:
                    return "pure-return-v1 rejects call attribute {}".format(function.attr[:48])
            else:
                return "pure-return-v1 rejects dynamic call target"
        elif isinstance(node, ast.FunctionDef):
            if node.decorator_list or node.returns is not None or node.type_comment is not None:
                return "pure-return-v1 rejects function metadata"
        elif isinstance(node, ast.arguments):
            if node.vararg or node.kwarg or node.kwonlyargs or node.posonlyargs:
                return "pure-return-v1 rejects variadic arguments"
    return None


_DOCKER_BOOTSTRAP = r'''
import ctypes
import errno
import json
import os
import signal
import socket
import subprocess
import sys

_REASON_BYTES = 192

def _write_all(fd, payload):
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        view = view[written:]

def _control(fd, prefix, reason=""):
    if prefix == b"READY" or prefix == b"SYNTAX":
        payload = prefix
    else:
        encoded = str(reason).encode("utf-8", errors="replace")[:_REASON_BYTES]
        payload = b"FAIL" + encoded.ljust(_REASON_BYTES, b"\0")
    try:
        _write_all(fd, payload)
    finally:
        try:
            os.close(fd)
        except OSError:
            pass

def _fail(fd, reason):
    _control(fd, b"FAIL", reason)
    os._exit(44)

def _install_inner_filter():
    """Install the candidate-phase deny-default allowlist after preload."""
    try:
        seccomp = ctypes.CDLL("libseccomp.so.2", use_errno=True)
    except BaseException as exc:
        raise RuntimeError("candidate-phase libseccomp is unavailable") from exc
    seccomp.seccomp_init.restype = ctypes.c_void_p
    seccomp.seccomp_syscall_resolve_name.argtypes = [ctypes.c_char_p]
    seccomp.seccomp_syscall_resolve_name.restype = ctypes.c_int
    seccomp.seccomp_rule_add.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int, ctypes.c_uint]
    seccomp.seccomp_rule_add.restype = ctypes.c_int
    seccomp.seccomp_load.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_load.restype = ctypes.c_int
    seccomp.seccomp_release.argtypes = [ctypes.c_void_p]
    seccomp.seccomp_release.restype = None
    # SCMP_ACT_ERRNO(EPERM) is the default.  Resolve every configured name
    # inside the pinned image; a missing name is infrastructure loss rather
    # than a silently weaker candidate policy.
    context = seccomp.seccomp_init(0x00050000 | errno.EPERM)
    if not context:
        raise RuntimeError("libseccomp initialization failed")
    try:
        candidate_allowlist = (
            "read", "write", "close", "fstat", "newfstatat", "lseek", "mmap", "mprotect", "munmap", "brk",
            "rt_sigaction", "rt_sigprocmask", "rt_sigreturn", "ioctl", "pread64", "readv", "writev",
            "faccessat", "faccessat2", "pipe2", "fcntl", "dup", "dup3", "futex", "set_tid_address",
            "set_robust_list", "rseq", "prlimit64", "getrandom", "clock_gettime", "clock_getres",
            "clock_nanosleep", "getpid", "getppid", "gettid", "getuid", "geteuid", "getgid", "getegid",
            "uname", "sched_getaffinity", "exit", "exit_group", "wait4", "getcwd", "statx", "getdents64",
            "getrlimit", "getrusage", "sysinfo", "times", "madvise", "mremap", "gettimeofday", "nanosleep",
            "sigaltstack", "rt_sigsuspend", "prctl", "chdir", "fchdir", "fstatfs", "fsync", "fdatasync", "epoll_create1",
            "epoll_ctl", "epoll_pwait", "eventfd2", "getgroups", "getitimer", "getpeername", "getsid",
            "getsockname", "getsockopt", "ppoll", "pselect6", "setresuid", "setresgid", "setgroups",
            "setpgid", "setsid", "setuid", "setgid", "setreuid", "setregid", "umask", "utimensat",
            "openat", "openat2", "execve", "execveat", "clone", "clone3", "socket", "socketpair", "connect",
            "bind", "listen", "accept", "accept4", "io_uring_setup", "io_uring_enter", "io_uring_register",
            "memfd_create", "memfd_secret", "userfaultfd", "bpf", "mount", "umount2", "pivot_root", "unshare",
            "setns", "renameat", "renameat2", "linkat", "unlinkat", "truncate", "ftruncate", "fchmod",
            "fchmodat", "fchmodat2", "mkdirat", "symlinkat", "mknodat", "kill", "tkill", "tgkill", "ptrace",
            "process_vm_readv", "process_vm_writev", "readlinkat", "getdents64", "open_by_handle_at",
        )
        denied = frozenset(
            {
                "openat", "openat2", "execve", "execveat", "clone", "clone3", "socket", "socketpair",
                "connect", "bind", "listen", "accept", "accept4", "io_uring_setup", "io_uring_enter",
                "io_uring_register", "memfd_create", "memfd_secret", "userfaultfd", "bpf", "mount", "umount2",
                "pivot_root", "unshare", "setns", "renameat", "renameat2", "linkat", "unlinkat", "truncate",
                "ftruncate", "fchmod", "fchmodat", "fchmodat2", "mkdirat", "symlinkat", "mknodat", "kill",
                "exit", "exit_group", "tkill", "tgkill", "ptrace", "process_vm_readv", "process_vm_writev", "readlinkat",
                "getdents64", "open_by_handle_at",
            }
        )
        for name in candidate_allowlist:
            syscall_number = seccomp.seccomp_syscall_resolve_name(name.encode("ascii"))
            if syscall_number < 0:
                raise RuntimeError("unresolved candidate syscall " + name)
            if name in denied:
                continue
            if seccomp.seccomp_rule_add(context, 0x7FFF0000, syscall_number, 0) != 0:
                raise RuntimeError("libseccomp rule installation failed for " + name)
        if seccomp.seccomp_load(context) != 0:
            raise RuntimeError("libseccomp load failed")
    finally:
        seccomp.seccomp_release(context)

_FRAME_MAGIC = b"EGV-WORKER-V1\n"
_MAX_FRAME_BYTES = 65537

def _read_exact(fd, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = os.read(fd, remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)

def _read_line(fd, limit):
    value = bytearray()
    while len(value) <= limit:
        chunk = os.read(fd, 1)
        if not chunk:
            return bytes(value) if value else None
        value.extend(chunk)
        if chunk == b"\n":
            return bytes(value[:-1])
    return None

def _read_frame(fd):
    magic = _read_exact(fd, len(_FRAME_MAGIC))
    if not magic:
        return None
    if magic != _FRAME_MAGIC:
        return (b"INVALID", b"bad worker frame magic")
    kind = _read_line(fd, 64)
    length_text = _read_line(fd, 32)
    if kind is None or length_text is None:
        return (b"INVALID", b"truncated worker frame")
    try:
        length = int(length_text.decode("ascii"))
    except (UnicodeDecodeError, ValueError):
        return (b"INVALID", b"invalid worker frame length")
    if length < 0 or length > _MAX_FRAME_BYTES:
        return (b"INVALID", b"worker frame exceeds bounded channel")
    payload = _read_exact(fd, length)
    if len(payload) != length:
        return (b"INVALID", b"truncated worker frame payload")
    return kind, payload

try:
    source_path = sys.argv[1]
    control_fd = int(sys.argv[2])
    try:
        source_text = open(source_path, "r", encoding="utf-8").read()
        source_code = compile(source_text, source_path, "exec")
    except SyntaxError:
        _control(control_fd, b"SYNTAX")
        os._exit(42)
    except BaseException as exc:
        _fail(control_fd, "source load failed:" + type(exc).__name__)
    try:
        value = json.load(sys.stdin)
    except BaseException as exc:
        _fail(control_fd, "opaque input failed:" + type(exc).__name__)

    # These bindings live in the bootstrap parent, never in candidate code.
    # Candidate mutation of os._exit/sys.exit/_status in the worker cannot
    # alter how this parent terminates or classifies the worker.
    libc = ctypes.CDLL(None, use_errno=True)
    hard_exit = libc._exit
    hard_exit.argtypes = [ctypes.c_int]
    hard_exit.restype = None
    raw_write = libc.write
    raw_write.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
    raw_write.restype = ctypes.c_ssize_t
    raw_close = libc.close
    raw_close.argtypes = [ctypes.c_int]
    raw_close.restype = ctypes.c_int
    raw_kill = libc.kill
    raw_kill.argtypes = [ctypes.c_int, ctypes.c_int]
    raw_kill.restype = ctypes.c_int
    raw_pause = libc.pause
    raw_pause.argtypes = []
    raw_pause.restype = ctypes.c_int

    def raw_write_all(fd, payload):
        buffer = ctypes.create_string_buffer(payload)
        offset = 0
        while offset < len(payload):
            written = raw_write(fd, ctypes.byref(buffer, offset), len(payload) - offset)
            if written <= 0:
                raise OSError(ctypes.get_errno(), "worker result channel write failed")
            offset += written

    def send_frame(fd, kind, payload=b""):
        if len(payload) > _MAX_FRAME_BYTES:
            payload = payload[:_MAX_FRAME_BYTES]
        header = _FRAME_MAGIC + kind + b"\n" + str(len(payload)).encode("ascii") + b"\n"
        raw_write_all(fd, header + payload)

    def block_worker(fd):
        raw_close(fd)
        while True:
            raw_pause()

    def parent_status(code, name):
        sys.stderr.write("EGV_SANDBOX_STATUS=" + name + "\n")
        sys.stderr.flush()
        hard_exit(code)

    def terminate_and_reap(pid):
        if raw_kill(pid, 15) != 0 and ctypes.get_errno() != 3:
            raw_kill(pid, 9)
        while True:
            try:
                waited_pid, waited_status = os.waitpid(pid, 0)
                if waited_pid == pid:
                    return waited_status
            except InterruptedError:
                continue

    worker_read, worker_write = os.pipe()
    worker_pid = os.fork()
    if worker_pid == 0:
        # The worker has no bootstrap control descriptor or Docker stdout.
        # It can report only bounded frames, then blocks until this parent
        # terminates and reaps it.
        os.close(worker_read)
        os.close(control_fd)
        del control_fd
        devnull = os.open("/dev/null", os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        if devnull > 2:
            raw_close(devnull)
        send = send_frame
        block = block_worker
        try:
            try:
                _install_inner_filter()
            except BaseException as exc:
                send(worker_write, b"FILTER_FAILURE", ("filter setup:" + str(exc)[:160]).encode("utf-8"))
                block(worker_write)
            send(worker_write, b"FILTER_READY")
            json_dumps = json.dumps
            namespace = {"__name__": "__candidate__", "__file__": source_path}
            try:
                exec(source_code, namespace, namespace)
                candidate_main = namespace.get("main")
                if not callable(candidate_main):
                    raise RuntimeError("candidate must define main(value)")
                precheck = namespace.get("__egv_precheck__")
                if callable(precheck):
                    namespace["__egv_precheck_result__"] = precheck(value)
                result = candidate_main(value)
                rendered = (json_dumps(result, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
                send(worker_write, b"NORMAL", rendered)
            except SyntaxError:
                send(worker_write, b"SYNTAX")
            except BaseException as exc:
                send(worker_write, b"RUNTIME_EXCEPTION", ("RUNTIME_EXCEPTION:" + type(exc).__name__).encode("utf-8"))
            block(worker_write)
        except BaseException as exc:
            try:
                send(worker_write, b"RUNTIME_EXCEPTION", ("worker failure:" + type(exc).__name__).encode("utf-8"))
            except BaseException:
                pass
            block(worker_write)
    os.close(worker_write)
    setup = _read_frame(worker_read)
    if setup is None:
        waited = terminate_and_reap(worker_pid)
        if os.WIFSIGNALED(waited) and os.WTERMSIG(waited) == 9:
            hard_exit(137)
        _fail(control_fd, "worker exited before filter-ready")
    setup_kind, setup_payload = setup
    if setup_kind == b"FILTER_FAILURE":
        terminate_and_reap(worker_pid)
        _control(control_fd, b"FAIL", setup_payload.decode("utf-8", errors="replace"))
        hard_exit(44)
    if setup_kind == b"SYNTAX":
        terminate_and_reap(worker_pid)
        _control(control_fd, b"SYNTAX")
        hard_exit(42)
    if setup_kind != b"FILTER_READY":
        terminate_and_reap(worker_pid)
        _control(control_fd, b"FAIL", "worker filter handshake was invalid")
        hard_exit(44)
    _control(control_fd, b"READY")
    frame = _read_frame(worker_read)
    if frame is None:
        raw_close(worker_read)
        waited = terminate_and_reap(worker_pid)
        if os.WIFSIGNALED(waited) and os.WTERMSIG(waited) == 9:
            parent_status(137, "RESOURCE_LIMIT")
        parent_status(43, "RUNTIME_EXCEPTION:WORKER_NO_RESULT")
    kind, payload = frame
    if kind == b"NORMAL":
        live_pid, live_status = os.waitpid(worker_pid, os.WNOHANG)
        if live_pid != 0:
            parent_status(43, "RUNTIME_EXCEPTION:WORKER_COMPLETION_NOT_LIVE")
        terminate_and_reap(worker_pid)
        raw_close(worker_read)
        sys.stdout.buffer.write(payload)
        sys.stdout.buffer.flush()
        hard_exit(0)
    raw_close(worker_read)
    if kind == b"RUNTIME_EXCEPTION" or kind == b"SYNTAX":
        terminate_and_reap(worker_pid)
        parent_status(43, kind.decode("ascii", errors="replace"))
    if kind == b"RESOURCE_LIMIT":
        terminate_and_reap(worker_pid)
        parent_status(137, "RESOURCE_LIMIT")
    terminate_and_reap(worker_pid)
    parent_status(43, "RUNTIME_EXCEPTION:WORKER_PROTOCOL")
except SystemExit:
    raise
except BaseException:
    sys.stderr.write("EGV_SANDBOX_STATUS=RUNTIME_EXCEPTION\n")
    sys.stderr.flush()
    try:
        hard_exit(43)
    except NameError:
        os._exit(43)
'''


_DOCKER_RUNNER = r'''
import hashlib
import hmac
import json
import os
import sys

_BOOTSTRAP = __EGV_BOOTSTRAP_LITERAL__
_REASON_BYTES = 192

def _status(code, name):
    sys.stderr.write("EGV_SANDBOX_STATUS=" + name + "\n")
    sys.stderr.flush()
    os._exit(code)

def _read_exact(fd, size):
    chunks = []
    remaining = size
    while remaining:
        chunk = os.read(fd, remaining)
        if not chunk:
            break
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)

def _write_all(fd, payload):
    view = memoryview(payload)
    while view:
        written = os.write(fd, view)
        view = view[written:]

def _runner_failure(token, reason):
    message = "RUNNER_FILTER_SETUP_FAILED:" + str(reason)[:160]
    mac = hmac.new(token, message.encode("utf-8"), hashlib.sha256).hexdigest()
    sys.stderr.write("EGV_SANDBOX_RUNNER_SENTINEL=" + message + ":" + mac + "\n")
    sys.stderr.flush()
    os._exit(44)

try:
    envelope = json.load(sys.stdin)
    if not isinstance(envelope, dict) or not isinstance(envelope.get("egv_runner_auth"), str):
        raise RuntimeError("runner authentication envelope is invalid")
    try:
        runner_auth = bytes.fromhex(envelope["egv_runner_auth"])
    except ValueError as exc:
        raise RuntimeError("runner authentication token is invalid") from exc
    if len(runner_auth) != 32 or "opaque_input" not in envelope:
        raise RuntimeError("runner authentication envelope is incomplete")
    input_bytes = (json.dumps(envelope["opaque_input"], sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode("utf-8")
    del envelope
    control_read, control_write = os.pipe()
    input_read, input_write = os.pipe()
    try:
        child_pid = os.fork()
    except BaseException as exc:
        for fd in (control_read, control_write, input_read, input_write):
            try:
                os.close(fd)
            except OSError:
                pass
        _runner_failure(runner_auth, "runner fork failed:" + type(exc).__name__)
    if child_pid == 0:
        try:
            os.close(control_read)
            os.close(input_write)
            os.dup2(input_read, 0)
            os.close(input_read)
            os.set_inheritable(0, True)
            os.set_inheritable(control_write, True)
            env = {
                "HOME": "/tmp",
                "PATH": "/usr/local/bin:/usr/bin:/bin",
                "PYTHONNOUSERSITE": "1",
                "PYTHONPATH": "",
                "LC_ALL": "C",
            }
            os.execve(
                sys.executable,
                [sys.executable, "-S", "-c", _BOOTSTRAP, "/candidate/egv-source.py", str(control_write)],
                env,
            )
        except BaseException as exc:
            try:
                _write_all(control_write, b"FAIL" + ("bootstrap exec failed:" + type(exc).__name__).encode("utf-8")[:_REASON_BYTES].ljust(_REASON_BYTES, b"\0"))
            except BaseException:
                pass
            os._exit(44)
    os.close(control_write)
    os.close(input_read)
    try:
        _write_all(input_write, input_bytes)
    except BaseException as exc:
        os.close(input_write)
        _runner_failure(runner_auth, "runner input pipe failed:" + type(exc).__name__)
    os.close(input_write)
    handshake = _read_exact(control_read, 4 + _REASON_BYTES)
    os.close(control_read)
    try:
        _, wait_status = os.waitpid(child_pid, 0)
    except BaseException as exc:
        _runner_failure(runner_auth, "runner wait failed:" + type(exc).__name__)
    if handshake == b"READY":
        if os.WIFEXITED(wait_status):
            child_code = os.WEXITSTATUS(wait_status)
            if child_code == 0:
                os._exit(0)
            if child_code in {137, 143}:
                _status(137, "RESOURCE_LIMIT")
            _status(43, "RUNTIME_EXCEPTION:CANDIDATE_EXIT_{}".format(child_code))
        child_signal = os.WTERMSIG(wait_status)
        if child_signal == 9:
            _status(137, "RESOURCE_LIMIT")
        _status(43, "RUNTIME_EXCEPTION:SIGNAL_{}".format(child_signal))
    if handshake == b"SYNTAX":
        _status(42, "SYNTAX_OR_IMPORT")
    if handshake.startswith(b"FAIL"):
        reason = handshake[4:].rstrip(b"\0").decode("utf-8", errors="replace")
    else:
        reason = "runner bootstrap handshake failed"
    _runner_failure(runner_auth, reason)
except BaseException as exc:
    try:
        token = runner_auth
    except NameError:
        token = None
    if token is not None and len(token) == 32:
        _runner_failure(token, "runner supervisor failure:" + type(exc).__name__)
    os._exit(44)
'''


def _docker_runner_source() -> str:
    placeholder = "__EGV_BOOTSTRAP_LITERAL__"
    if placeholder not in _DOCKER_RUNNER:
        raise DockerConfigurationError("Docker runner bootstrap placeholder is missing")
    return _DOCKER_RUNNER.replace(placeholder, repr(_DOCKER_BOOTSTRAP), 1)


class DockerCandidateSandbox:
    """The only enforceable candidate sandbox for the production smoke."""

    enforceable = True
    backend_name = "docker-enforced-v1"

    def __init__(
        self,
        workspace: Union[Path, str],
        *,
        config: Optional[DockerSandboxConfig] = None,
        timeout_seconds: float = DOCKER_TIMEOUT_SECONDS,
        output_limit: int = DOCKER_OUTPUT_LIMIT,
    ) -> None:
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.config = config or DockerSandboxConfig.from_environment()
        if (
            timeout_seconds != DOCKER_TIMEOUT_SECONDS
            or output_limit != DOCKER_OUTPUT_LIMIT
            or self.config.timeout_seconds != DOCKER_TIMEOUT_SECONDS
            or self.config.output_limit != DOCKER_OUTPUT_LIMIT
        ):
            raise DockerConfigurationError("Docker timeout/output limits differ from the frozen Evaluation contract")
        self.timeout_seconds = DOCKER_TIMEOUT_SECONDS
        self.output_limit = DOCKER_OUTPUT_LIMIT
        self.image_id = self.config.verify_image()
        self.docker_binary = shutil.which(self.config.docker_binary) or self.config.docker_binary
        self.artifacts = ContentAddressedArtifactStore(self.workspace / "source-artifacts")
        self.seccomp_path = self.config.seccomp_profile(self.workspace)
        self.last_environment_diff: Mapping[str, Any] = {}
        self.last_negative_control_evidence: Dict[str, Mapping[str, Any]] = {}

    def runtime_report(self) -> Dict[str, Any]:
        return {
            "backend": self.backend_name,
            "enforceable": True,
            "image_pinned": True,
            "image_id": self.image_id,
            "network": "none",
            "read_only_root": True,
            "cap_drop": "ALL",
            "no_new_privileges": True,
            "user": self.config.user,
            "pids_limit": self.config.pids_limit,
            "memory_limit": self.config.memory_limit,
            "tmpfs": "/tmp:ro,noexec,nosuid,nodev,size={}".format(self.config.tmpfs_size),
            "python_isolated": True,
            "candidate_environment": ["HOME", "PATH", "PYTHONNOUSERSITE", "PYTHONPATH", "LC_ALL"],
            "candidate_phase_inner_seccomp": "deny-open-exec-process-network-mutation-topology",
            "candidate_contract": (
                "pure-return-v1 (production AST admission; Docker isolation; "
                "hidden-oracle decision; untrusted adapter evidence otherwise)"
            ),
            "mutable_mounts": [],
            "runtime_inputs": ["pinned-image", "content-addressed-source", "opaque-stdin-json"],
            "host_paths_mounted": False,
        }

    def _create_args(self, name: str, source_volume: str, *, memory_limit: Optional[str] = None) -> List[str]:
        effective_memory = memory_limit or self.config.memory_limit
        if effective_memory not in {self.config.memory_limit, RESOURCE_BOUND_MEMORY_LIMIT}:
            raise DockerConfigurationError("candidate memory limit is outside the frozen resource contract")
        return [
            self.docker_binary,
            "create",
            "--name",
            name,
            "--interactive",
            "--network",
            "none",
            "--read-only",
            "--cap-drop",
            "ALL",
            "--security-opt",
            "no-new-privileges:true",
            "--security-opt",
            "seccomp={}".format(self.seccomp_path),
            "--user",
            self.config.user,
            "--pids-limit",
            str(self.config.pids_limit),
            "--memory",
            effective_memory,
            "--memory-swap",
            effective_memory,
            "--tmpfs",
            "/tmp:ro,noexec,nosuid,nodev,size={}".format(self.config.tmpfs_size),
            "--tmpfs",
            "/app:ro,noexec,nosuid,nodev,size=8m",
            "--tmpfs",
            "/etc:ro,noexec,nosuid,nodev,size=4m",
            "--tmpfs",
            "/home:ro,noexec,nosuid,nodev,size=4m",
            "--tmpfs",
            "/root:ro,noexec,nosuid,nodev,size=4m",
            "--tmpfs",
            "/var/run:ro,noexec,nosuid,nodev,size=4m",
            "--mount",
            "type=volume,source={},target=/candidate,readonly".format(source_volume),
            "--entrypoint",
            "/usr/bin/env",
            self.image_id,
            "-i",
            "HOME=/tmp",
            "PATH=/usr/local/bin:/usr/bin:/bin",
            "PYTHONNOUSERSITE=1",
            "PYTHONPATH=",
            "LC_ALL=C",
            "/usr/local/bin/python",
            "-S",
            "-c",
            _docker_runner_source(),
        ]

    def _stage_source(self, helper_name: str, source_volume: str, source_path: Path) -> None:
        """Stage the CAS bytes in a Docker volume before the read-only mount.

        Docker refuses ``docker cp`` into a read-only container.  The short
        lived staging container is infrastructure, not the candidate: it has
        no candidate input, no host bind, no network, and writes only the
        content-addressed source into the volume that is mounted read-only for
        the candidate.
        """

        volume_created = False
        helper_created = False
        staging_succeeded = False
        try:
            volume = subprocess.run(
                [self.docker_binary, "volume", "create", source_volume],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=10,
            )
            volume_created = volume.returncode == 0
            if volume.returncode != 0 or volume.stdout.strip() != source_volume:
                raise InfrastructureFailure("Docker source volume creation failed")
            create = subprocess.run(
                [
                    self.docker_binary,
                    "create",
                    "--name",
                    helper_name,
                    "--network",
                    "none",
                    "--read-only",
                    "--cap-drop",
                    "ALL",
                    "--security-opt",
                    "no-new-privileges:true",
                    "--security-opt",
                    "seccomp={}".format(self.seccomp_path),
                    "--user",
                    self.config.user,
                    "--pids-limit",
                    str(self.config.pids_limit),
                    "--memory",
                    self.config.memory_limit,
                    "--memory-swap",
                    self.config.memory_limit,
                    "--mount",
                    "type=volume,source={},target=/candidate".format(source_volume),
                    "--entrypoint",
                    "/bin/true",
                    self.image_id,
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=10,
            )
            helper_created = create.returncode == 0
            if create.returncode != 0 or not create.stdout.strip():
                raise InfrastructureFailure("Docker source staging container creation failed")
            copy = subprocess.run(
                [self.docker_binary, "cp", str(source_path), "{}:/candidate/egv-source.py".format(helper_name)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=10,
            )
            if copy.returncode != 0:
                raise InfrastructureFailure("Docker source artifact staging failed")
            staging_succeeded = True
        finally:
            cleanup_errors = []
            if helper_created:
                try:
                    self._cleanup_container(helper_name, required=True)
                except Exception as exc:
                    cleanup_errors.append(str(exc))
            if volume_created and (not staging_succeeded or cleanup_errors):
                try:
                    self._remove_volume(source_volume, required=True)
                except Exception as exc:
                    cleanup_errors.append(str(exc))
            if cleanup_errors:
                raise InfrastructureFailure("Docker staging cleanup failed: {}".format("; ".join(cleanup_errors)))

    def _start_args(self, name: str) -> List[str]:
        return [self.docker_binary, "start", "--attach", "--interactive", name]

    def _inspect(self, name: str) -> Dict[str, Any]:
        completed = subprocess.run(
            [self.docker_binary, "inspect", name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=10,
        )
        if completed.returncode != 0:
            raise InfrastructureFailure("Docker container inspection failed")
        try:
            return dict(json.loads(completed.stdout)[0])
        except (ValueError, IndexError, TypeError) as exc:
            raise InfrastructureFailure("Docker container inspection was invalid") from exc

    @staticmethod
    def _mount_inventory(inspect: Mapping[str, Any]) -> List[Dict[str, Any]]:
        inventory = []
        for mount in inspect.get("Mounts", []) or []:
            inventory.append(
                {
                    "type": mount.get("Type"),
                    "target": mount.get("Destination"),
                    "read_only": not bool(mount.get("RW", False)),
                    "rw": bool(mount.get("RW", False)),
                    "name_present": bool(mount.get("Name")),
                }
            )
        return sorted(inventory, key=lambda item: str(item.get("target")))

    def _container_snapshot(self, inspect: Mapping[str, Any]) -> Dict[str, Any]:
        config = inspect.get("Config", {}) or {}
        host_config = inspect.get("HostConfig", {}) or {}
        env_names = sorted(str(item).split("=", 1)[0] for item in (config.get("Env", []) or []))
        return {
            "image": config.get("Image"),
            "user": config.get("User"),
            "env_names": env_names,
            "network_mode": host_config.get("NetworkMode"),
            "readonly_rootfs": bool(host_config.get("ReadonlyRootfs", False)),
            "cap_drop": sorted(host_config.get("CapDrop") or []),
            "no_new_privileges": bool(host_config.get("SecurityOpt") and any("no-new-privileges" in item for item in host_config.get("SecurityOpt", []))),
            "mount_inventory": self._mount_inventory(inspect),
        }

    def _source_digest_from_container(self, name: str) -> str:
        destination = self.workspace / ("source-after-{}-{}.py".format(name[-12:], uuid.uuid4().hex))
        try:
            copied = subprocess.run(
                [self.docker_binary, "cp", "{}:/candidate/egv-source.py".format(name), str(destination)],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=10,
            )
            if copied.returncode != 0:
                raise InfrastructureFailure("Docker source after-state copy failed")
            return digest_bytes(destination.read_bytes())
        finally:
            if destination.exists():
                try:
                    destination.unlink()
                except OSError as exc:
                    raise InfrastructureFailure("Docker source after-state cleanup failed") from exc

    def _environment_diff(self, name: str, before: Mapping[str, Any], *, exit_code: Optional[int]) -> Dict[str, Any]:
        after_inspect = self._inspect(name)
        diff_result = subprocess.run(
            [self.docker_binary, "diff", name],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=10,
        )
        if diff_result.returncode != 0:
            raise InfrastructureFailure("Docker container diff failed")
        changed = sorted(line for line in diff_result.stdout.splitlines() if line)
        state = after_inspect.get("State", {}) or {}
        after_snapshot = self._container_snapshot(after_inspect)
        source_digest_after = self._source_digest_from_container(name)
        mutable_mounts = [item for item in after_snapshot["mount_inventory"] if item["rw"]]
        source_unchanged = source_digest_after == before["source_digest"]
        expected_mounts = tuple(str(item) for item in before.get("expected_mount_targets", ()))
        unexpected_changed = [
            path for path in changed
            if not any(path == "A {}".format(target) or path.endswith(" {}".format(target)) for target in expected_mounts)
        ]
        docker_diff_empty = not unexpected_changed
        return {
            "before": dict(before),
            "after": {
                "status": state.get("Status"),
                "exit_code": state.get("ExitCode", exit_code),
                "oom_killed": bool(state.get("OOMKilled", False)),
                "container_snapshot": after_snapshot,
                "source_digest_after": source_digest_after,
                "source_digest_before": before["source_digest"],
                "source_unchanged": source_unchanged,
                "mutable_mounts": mutable_mounts,
                "mutable_mounts_absent": not mutable_mounts,
                "docker_diff": changed[:128],
                "changed_paths": changed[:128],
                "changed_path_count": len(changed),
                "unexpected_changed_paths": unexpected_changed[:128],
                "docker_diff_empty": docker_diff_empty,
                "verified_unchanged_state": source_unchanged and docker_diff_empty and not mutable_mounts,
            },
        }

    @staticmethod
    def _authenticated_runner_failure(stderr: str, runner_auth: str) -> bool:
        """Accept infrastructure exit 44 only with the host-known runner MAC."""

        try:
            token = bytes.fromhex(runner_auth)
        except ValueError:
            return False
        marker = "EGV_SANDBOX_RUNNER_SENTINEL="
        for line in stderr.splitlines():
            if not line.startswith(marker):
                continue
            signed = line[len(marker):]
            try:
                message, mac = signed.rsplit(":", 1)
            except ValueError:
                continue
            expected = hmac.new(token, message.encode("utf-8"), hashlib.sha256).hexdigest()
            if message.startswith("RUNNER_FILTER_SETUP_FAILED:") and hmac.compare_digest(mac, expected):
                return True
        return False

    def _cleanup_container(self, name: str, *, required: bool) -> None:
        try:
            completed = subprocess.run(
                [self.docker_binary, "rm", "--force", name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise InfrastructureFailure("Docker container cleanup failed") from exc
        if completed.returncode != 0:
            missing = "no such container" in completed.stderr.lower()
            if not (not required and missing):
                raise InfrastructureFailure("Docker container cleanup failed")

    def _remove_volume(self, name: str, *, required: bool) -> None:
        try:
            completed = subprocess.run(
                [self.docker_binary, "volume", "rm", "--force", name],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                check=False,
                timeout=10,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise InfrastructureFailure("Docker volume cleanup failed") from exc
        if completed.returncode != 0:
            missing = "no such volume" in completed.stderr.lower()
            if not (not required and missing):
                raise InfrastructureFailure("Docker volume cleanup failed")

    def execute(
        self,
        source: bytes,
        opaque_input: Any,
        *,
        artifact_digest: str,
        candidate_id: str,
        source_path: Optional[Union[Path, str]] = None,
        memory_limit: Optional[str] = None,
        candidate_contract: str = "pure-return-v1",
    ) -> SandboxResult:
        if digest_bytes(source) != artifact_digest:
            raise ArtifactError("candidate artifact digest does not match source bytes")
        if source_path is not None:
            candidate_path = Path(source_path)
            if not candidate_path.is_file() or candidate_path.read_bytes() != source:
                raise ArtifactError("candidate source path does not match the content-addressed artifact")
        stored = self.artifacts.put(source, media_type="text/x-python", role="private-candidate-source")
        if stored.digest != artifact_digest:
            raise ArtifactError("candidate source CAS digest mismatch")
        sandbox_id = digest_for({"candidate_id": candidate_id, "artifact_digest": artifact_digest, "input": opaque_input})
        started = time.monotonic()
        if candidate_contract not in {"pure-return-v1", "untrusted-adversarial-v1"}:
            raise DockerConfigurationError("unknown candidate execution contract")
        if candidate_contract == "pure-return-v1":
            contract_reason = validate_candidate_source_contract(source)
            if contract_reason is not None:
                return SandboxResult(
                    sandbox_id,
                    artifact_digest,
                    Diagnostic.PROTOCOL_VIOLATION.value,
                    "UNDER_25",
                    b"",
                    "PROTOCOL_VIOLATION",
                    int((time.monotonic() - started) * 1000),
                    {
                        "backend": self.backend_name,
                        "candidate_contract": "pure-return-v1",
                        "contract_status": "REJECTED",
                        "contract_reason": contract_reason,
                    },
                )
        name = "egv-evaluation-{}".format(sandbox_id[-24:])
        source_volume = "egv-evaluation-source-{}-{}".format(artifact_digest[:16], sandbox_id[-8:])
        before = {
            "backend": self.backend_name,
            "image_id": self.image_id,
            "user": self.config.user,
            "network": "none",
            "read_only_root": True,
            "cap_drop": "ALL",
            "no_new_privileges": True,
            "pids_limit": self.config.pids_limit,
            "memory_limit": memory_limit or self.config.memory_limit,
            "candidate_contract": candidate_contract,
            "source_digest": artifact_digest,
            "expected_mount_targets": ["/candidate"],
        }
        candidate_created = False
        volume_created = False
        completed: Optional[subprocess.CompletedProcess] = None
        runner_auth = secrets.token_hex(32)
        authenticated_runner_failure = False
        environment_diff: Mapping[str, Any] = {"before": before, "after": {"status": "not-created"}}
        execution_error: Optional[BaseException] = None
        timed_out = False
        cleanup_errors: List[str] = []
        try:
            self._stage_source(name + "-stage", source_volume, self.artifacts.root / stored.relative_path)
            volume_created = True
            create = subprocess.run(
                self._create_args(name, source_volume, memory_limit=memory_limit),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=10,
            )
            candidate_created = create.returncode == 0
            if create.returncode != 0 or not create.stdout.strip():
                raise InfrastructureFailure("Docker candidate container creation failed")
            before_inspect = self._inspect(name)
            before_snapshot = self._container_snapshot(before_inspect)
            before = {
                **before,
                "container_snapshot": before_snapshot,
                "source_digest_before": artifact_digest,
                "mutable_mounts_before": [item for item in before_snapshot["mount_inventory"] if item["rw"]],
            }
            completed = subprocess.run(
                self._start_args(name),
                input=canonical_bytes({"egv_runner_auth": runner_auth, "opaque_input": opaque_input}),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
                timeout=self.timeout_seconds,
            )
            environment_diff = self._environment_diff(name, before, exit_code=completed.returncode)
            runner_status = completed.stderr.decode("utf-8", errors="replace")[:512]
            authenticated_runner_failure = self._authenticated_runner_failure(runner_status, runner_auth)
            if runner_status:
                environment_diff = {
                    **environment_diff,
                    "after": {**dict(environment_diff.get("after", {})), "runner_status": runner_status},
                }
            self.last_environment_diff = environment_diff
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            execution_error = exc
            try:
                environment_diff = self._environment_diff(name, before, exit_code=None) if candidate_created else environment_diff
                self.last_environment_diff = environment_diff
            except Exception as diff_exc:
                environment_diff = {"before": before, "after": {"status": "timeout", "diff_error": type(diff_exc).__name__}}
                self.last_environment_diff = environment_diff
        except (OSError, InfrastructureFailure) as exc:
            execution_error = exc
        finally:
            try:
                self._cleanup_container(name, required=candidate_created)
            except Exception as exc:
                cleanup_errors.append(str(exc))
            try:
                self._remove_volume(source_volume, required=volume_created)
            except Exception as exc:
                cleanup_errors.append(str(exc))

        if cleanup_errors:
            environment_diff = {
                **dict(environment_diff),
                "cleanup": {"ok": False, "errors": list(cleanup_errors)},
                "quarantine": {
                    "required": True,
                    "reason": "cleanup-failed",
                    "container_name": name,
                    "source_volume": source_volume,
                },
            }
            self.last_environment_diff = environment_diff
            return SandboxResult(
                sandbox_id,
                artifact_digest,
                Diagnostic.INTERNAL_ERROR.value,
                "UNDER_25",
                b"",
                "CLEANUP_FAILED",
                int((time.monotonic() - started) * 1000),
                environment_diff,
                _incident(sandbox_id, "docker-cleanup-failed"),
            )
        if execution_error is not None:
            self.last_environment_diff = environment_diff
            return SandboxResult(
                sandbox_id,
                artifact_digest,
                Diagnostic.TIMEOUT.value if timed_out else Diagnostic.INTERNAL_ERROR.value,
                "LIMIT_REACHED" if timed_out else "UNDER_25",
                b"",
                "TIMEOUT" if timed_out else "INTERNAL_ERROR",
                int((time.monotonic() - started) * 1000),
                environment_diff,
                _incident(sandbox_id, "docker-timeout" if timed_out else type(execution_error).__name__),
            )

        assert completed is not None
        self.last_environment_diff = environment_diff
        elapsed = int((time.monotonic() - started) * 1000)
        if not environment_diff.get("after", {}).get("verified_unchanged_state", False):
            return SandboxResult(
                sandbox_id,
                artifact_digest,
                Diagnostic.INTERNAL_ERROR.value,
                "UNDER_25",
                b"",
                "STATE_CHANGED",
                elapsed,
                environment_diff,
                _incident(sandbox_id, "docker-state-changed"),
            )
        output = bytes(completed.stdout)
        if len(output) > self.output_limit:
            return SandboxResult(
                sandbox_id,
                artifact_digest,
                Diagnostic.RESOURCE_LIMIT.value,
                "OUTPUT_LIMIT",
                b"",
                "OUTPUT_LIMIT",
                elapsed,
                environment_diff,
            )
        if completed.returncode == 0:
            try:
                json.loads(output.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                return SandboxResult(
                    sandbox_id,
                    artifact_digest,
                    Diagnostic.PROTOCOL_VIOLATION.value,
                    "UNDER_25",
                    b"",
                    "INVALID_OUTPUT",
                    elapsed,
                    environment_diff,
                )
            if candidate_contract == "untrusted-adversarial-v1":
                # Arbitrary adversarial Python is deliberately exercised in
                # the Docker adapter for negative controls, but it shares no
                # trusted decision state with the wrapper.  Even a normal
                # return is therefore protocol evidence, never PASS.
                evidence_diff = {
                    **dict(environment_diff),
                    "evidence_only": True,
                    "observed_diagnostic": Diagnostic.PASS.value,
                    "decision_eligible": False,
                }
                return SandboxResult(
                    sandbox_id,
                    artifact_digest,
                    Diagnostic.PROTOCOL_VIOLATION.value,
                    "UNDER_25",
                    output,
                    "EVIDENCE_ONLY",
                    elapsed,
                    evidence_diff,
                    None,
                    True,
                )
            return SandboxResult(
                sandbox_id,
                artifact_digest,
                Diagnostic.PASS.value,
                "UNDER_25",
                output,
                "SUCCESS",
                elapsed,
                environment_diff,
            )
        if completed.returncode == 42:
            diagnostic = Diagnostic.SYNTAX_OR_IMPORT.value
            status = "SYNTAX_OR_IMPORT"
        elif completed.returncode == 44:
            if authenticated_runner_failure:
                diagnostic = Diagnostic.INTERNAL_ERROR.value
                status = "INFRASTRUCTURE_LOSS"
            else:
                diagnostic = Diagnostic.RUNTIME_EXCEPTION.value
                status = "RUNTIME_EXCEPTION"
        elif completed.returncode == 41:
            # This status is reserved for a broker/runtime-tagged denial.  A
            # candidate-raised PermissionError exits 43 and is a model runtime
            # failure, never AUTHORITY_DENIED.
            diagnostic = Diagnostic.AUTHORITY_DENIED.value
            status = "DENIED"
        elif completed.returncode == 43:
            diagnostic = Diagnostic.RUNTIME_EXCEPTION.value
            status = "RUNTIME_EXCEPTION"
        elif completed.returncode in {137, 143} or environment_diff.get("after", {}).get("oom_killed", False):
            # OOM/cgroup enforcement is a resource result even when Docker
            # reports it as a generic signal-like exit code.
            diagnostic = Diagnostic.RESOURCE_LIMIT.value
            status = "RESOURCE_LIMIT"
        elif completed.returncode < 0:
            diagnostic = Diagnostic.RUNTIME_EXCEPTION.value
            status = "SIGNAL({})".format(-completed.returncode)
        else:
            diagnostic = Diagnostic.INTERNAL_ERROR.value
            status = "INFRASTRUCTURE_LOSS"
        return SandboxResult(
            sandbox_id,
            artifact_digest,
            diagnostic,
            "LIMIT_REACHED" if diagnostic == Diagnostic.RESOURCE_LIMIT.value else "UNDER_25",
            b"",
            status,
            elapsed,
            environment_diff,
            _incident(sandbox_id, status) if diagnostic == Diagnostic.INTERNAL_ERROR.value else None,
        )

    def _probe_source(self, expression: str, *, precheck: Optional[str] = None) -> bytes:
        source = (
            "import errno\n"
            "import ctypes\n"
            "import os\n"
            "import subprocess\n"
            "def __egv_precheck__(value):\n"
            "    path = {!r}\n"
            "    if path is None:\n"
            "        return 'READY'\n"
            "    try:\n"
            "        os.lstat(path)\n"
            "    except (FileNotFoundError, NotADirectoryError):\n"
            "        return 'ABSENT'\n"
            "    except PermissionError as exc:\n"
            "        return {'status': 'DENIED', 'errno': getattr(exc, 'errno', None)}\n"
            "    except OSError as exc:\n"
            "        return {'status': 'ERROR', 'errno': getattr(exc, 'errno', None)}\n"
            "    return 'PRESENT'\n"
            "def __egv_observe(result, errno_value):\n"
            "    return {'__egv_observed__': True, 'return_value': result, 'errno': errno_value}\n"
            "def __egv_libc_open(path):\n"
            "    libc = ctypes.CDLL(None, use_errno=True)\n"
            "    function = libc.open\n"
            "    function.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]\n"
            "    function.restype = ctypes.c_int\n"
            "    return function(path, os.O_RDONLY, 0)\n"
            "def __egv_libc_openat(path):\n"
            "    libc = ctypes.CDLL(None, use_errno=True)\n"
            "    function = libc.openat\n"
            "    function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_int]\n"
            "    function.restype = ctypes.c_int\n"
            "    return function(-100, path, os.O_RDONLY, 0)\n"
            "def __egv_syscall_number(name):\n"
            "    numbers = {'aarch64': {'userfaultfd': 282, 'bpf': 280}, 'arm64': {'userfaultfd': 282, 'bpf': 280}, 'x86_64': {'userfaultfd': 323, 'bpf': 321}}\n"
            "    return numbers[os.uname().machine][name]\n"
            "def main(value):\n"
            "    precheck = globals().get('__egv_precheck_result__')\n"
            "    if precheck == 'ABSENT':\n"
            "        return {'probe_status': 'ABSENT'}\n"
            "    try:\n"
            "        result = __EGV_PROBE_EXPRESSION__\n"
            "    except (FileNotFoundError, NotADirectoryError) as exc:\n"
            "        return {'probe_status': 'ABSENT', 'errno': getattr(exc, 'errno', None), 'exception': type(exc).__name__}\n"
            "    except PermissionError as exc:\n"
            "        return {'probe_status': 'DENIED', 'errno': getattr(exc, 'errno', None), 'exception': type(exc).__name__}\n"
            "    except OSError as exc:\n"
            "        status = 'ABSENT' if getattr(exc, 'errno', None) in (errno.ENOENT, errno.ENOTDIR) else 'DENIED'\n"
            "        return {'probe_status': status, 'errno': getattr(exc, 'errno', None), 'exception': type(exc).__name__}\n"
            "    except (RuntimeError, subprocess.SubprocessError) as exc:\n"
            "        return {'probe_status': 'FAILED', 'exception': type(exc).__name__}\n"
            "    if isinstance(result, dict) and result.get('__egv_observed__'):\n"
            "        return {'probe_status': 'OBSERVED', 'return_value': result.get('return_value'), 'errno': result.get('errno')}\n"
            "    return {'probe_status': 'ALLOWED', 'value_type': type(result).__name__, 'value': repr(result)}\n"
        )
        source = source.replace("{!r}", repr(precheck), 1)
        source = source.replace("__EGV_PROBE_EXPRESSION__", expression, 1)
        return source.encode("utf-8")

    def negative_controls(self, hidden_resource: Union[Path, str]) -> Dict[str, bool]:
        """Run production controls while distinguishing absence from denial."""

        del hidden_resource  # The hidden fixture is intentionally never mounted or named.
        hidden = "/egv/hidden/fixture.json"
        hidden_dir = "/egv/hidden"
        probes = {
            "hidden_read": ("open({!r}, 'rb').read()".format(hidden), hidden, "ABSENT"),
            "hidden_list": ("os.listdir({!r})".format(hidden_dir), hidden_dir, "ABSENT"),
            "hidden_walk": ("bool(list(os.walk({!r})))".format(hidden_dir), hidden_dir, "ABSENT"),
            "hidden_scandir": ("list(os.scandir({!r}))".format(hidden_dir), hidden_dir, "ABSENT"),
            "hidden_stat": ("os.stat({!r})".format(hidden), hidden, "ABSENT"),
            "hidden_chdir": ("os.chdir({!r})".format(hidden_dir), hidden_dir, "ABSENT"),
            "hidden_readlink": ("os.readlink({!r})".format(hidden), hidden, "ABSENT"),
            "hidden_rename": ("os.rename({!r}, '/tmp/egv-hidden-steal')".format(hidden), hidden, "ABSENT"),
            "hidden_link": ("os.link({!r}, '/tmp/egv-hidden-link')".format(hidden), hidden, "ABSENT"),
            "hidden_unlink": ("os.unlink({!r})".format(hidden), hidden, "ABSENT"),
            "hidden_truncate": ("os.truncate({!r}, 0)".format(hidden), hidden, "ABSENT"),
            "hidden_chmod": ("os.chmod({!r}, 0o777)".format(hidden), hidden, "ABSENT"),
            "source_unlink": ("os.unlink('/candidate/egv-source.py')", "/candidate/egv-source.py", "DENIED"),
            "source_truncate": ("os.truncate('/candidate/egv-source.py', 0)", "/candidate/egv-source.py", "DENIED"),
            "source_chmod": ("os.chmod('/candidate/egv-source.py', 0o777)", "/candidate/egv-source.py", "DENIED"),
            "source_link": ("os.link('/candidate/egv-source.py', '/tmp/egv-source-link')", "/candidate/egv-source.py", "DENIED"),
            "process": ("subprocess.Popen(['true'])", None, "DENIED"),
            "network": ("__import__('socket').socket()", None, "DENIED"),
            "same_uid_kill": ("os.kill(os.getppid(), 0)", None, "DENIED"),
            "mount": ("__egv_observe(ctypes.CDLL(None, use_errno=True).mount(b'none', b'/tmp', b'tmpfs', 0, None), ctypes.get_errno())", None, "OBSERVED_DENIED"),
            "umount": ("__egv_observe(ctypes.CDLL(None, use_errno=True).umount2(b'/tmp', 0), ctypes.get_errno())", None, "OBSERVED_DENIED"),
            "ctypes_open": ("__egv_observe(__egv_libc_open(b'/candidate/egv-source.py'), ctypes.get_errno())", None, "OBSERVED_DENIED"),
            "ctypes_openat": ("__egv_observe(__egv_libc_openat(b'/candidate/egv-source.py'), ctypes.get_errno())", None, "OBSERVED_DENIED"),
            "ctypes_socket": ("__egv_observe(ctypes.CDLL(None, use_errno=True).socket(2, 1, 0), ctypes.get_errno())", None, "OBSERVED_DENIED"),
            "ctypes_system": ("__egv_observe(ctypes.CDLL(None, use_errno=True).system(b'exit 0'), ctypes.get_errno())", None, "OBSERVE_NONZERO"),
            "execve": ("os.execve('/bin/true', ['true'], {})", None, "DENIED"),
            "io_uring": ("__egv_observe(ctypes.CDLL(None, use_errno=True).syscall(425, 0, 0, 0), ctypes.get_errno())", None, "OBSERVED_DENIED"),
            "memfd_create": ("os.memfd_create('egv-negative-control', 0)", None, "DENIED"),
            "memfd_secret": ("__egv_observe(ctypes.CDLL(None, use_errno=True).syscall(447, 0), ctypes.get_errno())", None, "OBSERVED_DENIED"),
            "userfaultfd": ("__egv_observe(ctypes.CDLL(None, use_errno=True).syscall(__egv_syscall_number('userfaultfd'), 0), ctypes.get_errno())", None, "OBSERVED_DENIED"),
            "bpf": ("__egv_observe(ctypes.CDLL(None, use_errno=True).syscall(__egv_syscall_number('bpf'), 0, 0, 0), ctypes.get_errno())", None, "OBSERVED_DENIED"),
            "mountinfo_read": ("open('/proc/self/mountinfo', 'rb').read()", None, "DENIED"),
            "proc_environ_read": ("open('/proc/1/environ', 'rb').read()", None, "DENIED"),
            "credentials_file": ("open('/run/secrets/egv-credential', 'rb').read()", "/run/secrets/egv-credential", "ABSENT"),
            "credentials_mount": ("os.path.exists('/run/secrets')", "/run/secrets", "ABSENT"),
            "credentials_proc": ("open('/proc/1/root/root/.aws/credentials', 'rb').read()", None, "DENIED"),
            "credentials_env": ("('PRESENT' if any(os.environ.get(name) for name in ('AWS_ACCESS_KEY_ID', 'AWS_SECRET_ACCESS_KEY', 'OPENAI_API_KEY', 'EGV_EVALUATOR_SECRET', 'GPG_KEY')) else 'ABSENT')", None, "VALUE:ABSENT"),
            "host_path_read": ("open('/host/etc/passwd', 'rb').read()", "/host/etc/passwd", "ABSENT"),
            "host_socket_absent": ("open('/var/run/docker.sock', 'rb').read()", "/var/run/docker.sock", "ABSENT"),
        }
        results: Dict[str, bool] = {}
        evidence: Dict[str, Mapping[str, Any]] = {}
        denial_errnos = {1, 30}  # EPERM / EROFS, including libseccomp's EPERM.
        for name, (expression, precheck, expected) in probes.items():
            source = self._probe_source(expression, precheck=precheck)
            digest = digest_bytes(source)
            outcome = self.execute(
                source,
                {},
                artifact_digest=digest,
                candidate_id="negative-" + name,
                candidate_contract="untrusted-adversarial-v1",
            )
            actual: Dict[str, Any] = {"diagnostic": outcome.diagnostic_enum}
            if outcome.evidence_only and outcome.diagnostic_enum == Diagnostic.PROTOCOL_VIOLATION.value:
                actual["evidence_only"] = True
            if outcome.output_bytes and (outcome.diagnostic_enum == Diagnostic.PASS.value or outcome.evidence_only):
                try:
                    actual.update(json.loads(outcome.output_bytes.decode("utf-8")))
                except (UnicodeDecodeError, ValueError):
                    actual["probe_status"] = "INVALID_OUTPUT"
            else:
                actual["probe_status"] = "EXECUTION_FAILURE"
            status = actual.get("probe_status")
            if expected == "ABSENT":
                valid = status == "ABSENT"
            elif expected == "DENIED":
                valid = status == "DENIED" and actual.get("errno") in denial_errnos
            elif expected == "OBSERVED_DENIED":
                valid = (
                    status == "OBSERVED"
                    and actual.get("return_value") == -1
                    and actual.get("errno") in denial_errnos
                )
            elif expected == "OBSERVE_NONZERO":
                # libc.system returns a wait status; errno is not a reliable
                # representation of the child exec result.  Do not invent it.
                valid = status == "OBSERVED" and actual.get("return_value") not in (None, 0, False)
            elif expected == "ABSENT_ERRNO":
                valid = status == "OBSERVED" and actual.get("return_value") == -1 and actual.get("errno") in {2, 20}
            else:
                valid = status == "ALLOWED" and actual.get("value") == "'ABSENT'"
            actual = {**actual, "expected": expected, "valid": valid}
            evidence[name] = actual
            results[name] = valid

        identity_source = (
            "import os\n"
            "def main(value):\n"
            "    env_names = sorted(os.environ)\n"
            "    forbidden = ('AWS_', 'OPENAI_', 'SSH_', 'EGV_EVALUATOR', 'EGV_EVALUATION_SECRET', 'DOCKER_', 'GPG_KEY')\n"
            "    return {'uid': os.getuid(), 'gid': os.getgid(), 'pid': os.getpid(), 'ppid': os.getppid(),\n"
            "            'source_mode': os.stat('/candidate/egv-source.py').st_mode & 0o777,\n"
            "            'source_present': os.path.isfile('/candidate/egv-source.py'),\n"
            "            'hidden_namespace_absent': not os.path.exists('/egv'),\n"
            "            'credential_leak': any(any(name.startswith(prefix) for prefix in forbidden) for name in env_names)}\n"
        ).encode("utf-8")
        identity_digest = digest_bytes(identity_source)
        identity = self.execute(
            identity_source,
            {},
            artifact_digest=identity_digest,
            candidate_id="negative-identity",
            candidate_contract="untrusted-adversarial-v1",
        )
        identity_value: Dict[str, Any] = {}
        if identity.output_bytes and (identity.diagnostic_enum == Diagnostic.PASS.value or identity.evidence_only):
            try:
                identity_value = json.loads(identity.output_bytes.decode("utf-8"))
            except (UnicodeDecodeError, ValueError):
                identity_value = {}
        # Docker PID 1 owns the outer runner; the bootstrap parent then owns
        # the filtered worker. The candidate must remain inside that isolated
        # process tree and never report the host/container init as its parent.
        results["pid_namespace"] = identity_value.get("pid", 1) > 1 and isinstance(identity_value.get("ppid"), int) and identity_value.get("ppid") > 1
        results["uid_gid"] = (identity_value.get("uid"), identity_value.get("gid")) == (65534, 65534)
        results["source_read_only"] = identity_value.get("source_mode") == 0o444
        results["hidden_inode_absent"] = identity_value.get("source_present") is True and identity_value.get("hidden_namespace_absent") is True
        after_snapshot = self.last_environment_diff.get("after", {}).get("container_snapshot", {})
        mounts = after_snapshot.get("mount_inventory", [])
        results["host_path_absent"] = all(item.get("type") != "bind" and item.get("target") != "/var/run/docker.sock" for item in mounts)
        results["credentials_absent"] = identity_value.get("credential_leak") is False and evidence.get("credentials_env", {}).get("valid") is True
        evidence["identity"] = {
            "value": identity_value,
            "diagnostic": identity.diagnostic_enum,
            "evidence_only": identity.evidence_only,
        }
        self.last_negative_control_evidence = evidence
        if not all(results.values()):
            raise InfrastructureFailure("Docker sandbox negative controls failed: {}".format(results))
        return results


_LOCAL_RUNNER = r'''
import builtins
import json
import os
import socket
import subprocess
import sys

class _LocalAuthorityDenied(PermissionError):
    pass

ROOT = os.path.abspath(sys.argv[1])

def _inside(path):
    try:
        resolved = os.path.abspath(os.fspath(path))
    except (OSError, TypeError, ValueError):
        return False
    return resolved == ROOT or resolved.startswith(ROOT + os.sep)

def _deny(message='capability'):
    raise _LocalAuthorityDenied('[EGV_TEST_DENIED] ' + str(message))

def _audit(event, args):
    if event == 'open' and args and not isinstance(args[0], int) and not _inside(args[0]):
        _deny('file access')
    if event in {'os.listdir', 'os.scandir', 'glob.glob', 'shutil.copyfile'} and args and not _inside(args[0]):
        _deny('directory discovery')
    if event.startswith('socket') or event in {'subprocess.Popen', 'os.system', 'os.fork', 'os.exec'}:
        _deny('process or network capability')

sys.addaudithook(_audit)
socket.socket = _deny
socket.create_connection = _deny
subprocess.Popen = _deny
os.system = _deny
if hasattr(os, 'fork'):
    os.fork = _deny

try:
    namespace = {'__name__': '__candidate__', '__file__': sys.argv[2]}
    source_text = open(sys.argv[2], encoding='utf-8').read()
    exec(compile(source_text, sys.argv[2], 'exec'), namespace, namespace)
    main = namespace.get('main')
    if not callable(main):
        raise RuntimeError('candidate must define main(value)')
    result = main(json.load(sys.stdin))
    sys.stdout.write(json.dumps(result, sort_keys=True, separators=(',', ':')) + '\n')
except _LocalAuthorityDenied as exc:
    sys.stderr.write(str(exc) + '\n')
    sys.exit(41)
except SyntaxError:
    sys.exit(42)
except PermissionError:
    sys.exit(43)
except BaseException:
    sys.exit(43)
'''


class LocalTestSandbox:
    """AST/audit helper for unit tests only; it is not an authority boundary."""

    enforceable = False
    backend_name = "local-test-helper"

    def __init__(self, workspace: Union[Path, str], *, timeout_seconds: float = 2.0, output_limit: int = 65536) -> None:
        self.workspace = Path(workspace)
        self.workspace.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = timeout_seconds
        self.output_limit = output_limit

    def execute(
        self,
        source: bytes,
        opaque_input: Any,
        *,
        artifact_digest: str,
        candidate_id: str,
        source_path: Optional[Union[Path, str]] = None,
    ) -> SandboxResult:
        if digest_bytes(source) != artifact_digest:
            raise ArtifactError("candidate artifact digest does not match source bytes")
        sandbox_id = digest_for({"candidate_id": candidate_id, "artifact_digest": artifact_digest, "input": opaque_input})
        with tempfile.TemporaryDirectory(prefix="sandbox-", dir=str(self.workspace)) as directory:
            root = Path(directory)
            source_path_local = root / "candidate.py"
            runner_path = root / "runner.py"
            source_path_local.write_bytes(source)
            runner_path.write_text(_LOCAL_RUNNER, encoding="utf-8")
            started = time.monotonic()
            try:
                completed = subprocess.run(
                    [sys.executable, str(runner_path), str(root), str(source_path_local)],
                    input=canonical_bytes(opaque_input),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    cwd=str(root),
                    env={"PYTHONNOUSERSITE": "1"},
                    timeout=self.timeout_seconds,
                    check=False,
                )
            except subprocess.TimeoutExpired:
                return SandboxResult(sandbox_id, artifact_digest, Diagnostic.TIMEOUT.value, "LIMIT_REACHED", b"", "TIMEOUT", int((time.monotonic() - started) * 1000), {"backend": self.backend_name})
            except OSError as exc:
                return SandboxResult(sandbox_id, artifact_digest, Diagnostic.INTERNAL_ERROR.value, "UNDER_25", b"", "INTERNAL_ERROR", int((time.monotonic() - started) * 1000), {"backend": self.backend_name}, _incident(sandbox_id, type(exc).__name__))
            elapsed = int((time.monotonic() - started) * 1000)
            output = bytes(completed.stdout)
            if len(output) > self.output_limit:
                return SandboxResult(sandbox_id, artifact_digest, Diagnostic.RESOURCE_LIMIT.value, "OUTPUT_LIMIT", b"", "OUTPUT_LIMIT", elapsed, {"backend": self.backend_name})
            if completed.returncode == 0:
                try:
                    json.loads(output.decode("utf-8"))
                except (UnicodeDecodeError, ValueError):
                    return SandboxResult(sandbox_id, artifact_digest, Diagnostic.PROTOCOL_VIOLATION.value, "UNDER_25", b"", "INVALID_OUTPUT", elapsed, {"backend": self.backend_name})
                return SandboxResult(sandbox_id, artifact_digest, Diagnostic.PASS.value, "UNDER_25", output, "SUCCESS", elapsed, {"backend": self.backend_name})
            if completed.returncode == 41:
                diagnostic = Diagnostic.AUTHORITY_DENIED.value
                status = "DENIED"
            elif completed.returncode == 42:
                diagnostic = Diagnostic.SYNTAX_OR_IMPORT.value
                status = "SYNTAX_OR_IMPORT"
            else:
                diagnostic = Diagnostic.RUNTIME_EXCEPTION.value
                status = "RUNTIME_EXCEPTION"
            return SandboxResult(sandbox_id, artifact_digest, diagnostic, "UNDER_25", b"", status, elapsed, {"backend": self.backend_name})

    def negative_controls(self, hidden_resource: Union[Path, str]) -> Dict[str, bool]:
        """Legacy helper controls; never used as production evidence."""

        hidden = str(hidden_resource)
        probes = {
            "hidden_read": "open({!r}, 'rb').read()".format(hidden),
            "hidden_list": "__import__('os').listdir({!r})".format(hidden),
            "process": "__import__('subprocess').Popen(['true'])",
            "network": "__import__('socket').socket()",
            "credentials": "__import__('os').environ.get('EGV_EVALUATION_SECRET', 'UNEXPECTED_ALLOW')",
            "evaluator_mutation": "open({!r}, 'wb').write(b'x')".format(hidden),
        }
        results: Dict[str, bool] = {}
        for name, expression in probes.items():
            if name == "credentials":
                source = b"def main(value):\n    return 'DENIED' if 'EGV_EVALUATION_SECRET' not in __import__('os').environ else 'UNEXPECTED_ALLOW'\n"
            else:
                source = (
                    "def main(value):\n"
                    "    try:\n"
                    "        {}\n"
                    "    except (PermissionError, KeyError):\n"
                    "        return 'DENIED'\n"
                    "    return 'UNEXPECTED_ALLOW'\n"
                ).format(expression).encode("utf-8")
            outcome = self.execute(source, {}, artifact_digest=digest_bytes(source), candidate_id="negative-" + name)
            try:
                value = json.loads(outcome.output_bytes.decode("utf-8")) if outcome.output_bytes else None
            except (UnicodeDecodeError, ValueError):
                value = None
            results[name] = outcome.diagnostic_enum == Diagnostic.PASS.value and value == "DENIED"
        if not all(results.values()):
            raise ArtifactError("test-only sandbox control failed: {}".format(results))
        return results


__all__ = [
    "DOCKER_MEMORY_LIMIT",
    "DOCKER_OUTPUT_LIMIT",
    "DOCKER_PIDS_LIMIT",
    "DOCKER_TIMEOUT_SECONDS",
    "DOCKER_TMPFS",
    "DockerCandidateSandbox",
    "DockerSandboxConfig",
    "LocalTestSandbox",
    "SandboxResult",
    "validate_candidate_source_contract",
]
