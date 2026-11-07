"""Out-of-process supervisor and CLI for frozen CHELATEDAI-QSCCI-v4.

The supervisor owns service observation/restoration, child process-group
timeouts, and no-replace publication.  The research worker can write only to a
hidden stage and can never publish a result directory itself.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import math
import os
import secrets
import signal
import stat
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

from qscci_experiment import (
    FIXTURE_SHA256,
    PROTOCOL_ID,
    PROTOCOL_SHA256,
    QSCCIError,
    SCHEMA_PATH,
    SCHEMA_SHA256,
    artifact_digest,
    canonical_json,
    run_worker,
    sha256_bytes,
    strict_json,
    verify_artifact,
    verify_artifact_cache_independent,
)


ARTIFACT_NAME = "qscci.json"
MANIFEST_NAME = "manifest.json"
COMMIT_NAME = "COMMIT.json"
INVALID_RUN_NAME = "INVALID_RUN.json"
CHILD_TERM_SECONDS = 1900.0
CHILD_KILL_GRACE_SECONDS = 30.0
RESTORATION_SECONDS = 600.0
DEEPSEEK_MODEL_ID = "deepseek-v4-flash-0731"
RUNNING_METRIC = "vllm:num_requests_running"
WAITING_METRIC = "vllm:num_requests_waiting"
V4_ARCHIVE_ARTIFACT_SHA256 = "043f5b395549f383b21f39c5fdbd16572ea78b763b423a0314d2d48d378dbad6"
V4_ARCHIVE_MANIFEST_SHA256 = "dd0cc08b5f3e3833b28a2ada70f0f78fe4d86ac099cc03fbdc38afb90538e991"
V4_ARCHIVE_COMMIT_SHA256 = "c16438aeaeceee03d638f2eea6a3f5c14727937590160cdf6ebf63e976189c20"
V4_ARCHIVE_ARTIFACT_DIGEST = "fe75a63bb2e2ef14d911c8be457f01044d329377c8982ccf598a7a19ac71d1e2"
V4_ARCHIVE_DISPOSITION = "DOES_NOT_SURVIVE_SMALL_LABEL_ORACLE_FIXTURE"
V4_ARCHIVE_ADDENDUM_PATH = Path(
    "docs/research/qwen-scope-chelated-causal-intervention-v4-portable-archive-verifier-addendum-2026-08-16.md"
)
# Filled only from the immutable addendum bytes; changing the addendum requires
# an explicit new custody-verifier distribution.
V4_ARCHIVE_ADDENDUM_SHA256 = "f76a89035ce0fc6f16c8d1f1d1a91b36760f60ee8e7004d7cb66b7c06d3ae495"


class ChildExecutionFailure(QSCCIError):
    """Private carrier for bounded child diagnostics; never serialized wholesale."""

    def __init__(self, category: str, returncode: int | None, stdout: bytes, stderr: bytes, message: str) -> None:
        super().__init__(message)
        self.category = category
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


@dataclass(frozen=True)
class SupervisorConfig:
    output_dir: Path
    repo_root: Path
    child_term_seconds: float = CHILD_TERM_SECONDS
    child_kill_grace_seconds: float = CHILD_KILL_GRACE_SECONDS
    restoration_seconds: float = RESTORATION_SECONDS

    def validate(self) -> None:
        if not self.output_dir.is_absolute() or not self.repo_root.is_absolute():
            raise QSCCIError("supervisor paths must be absolute")
        if os.path.lexists(self.output_dir):
            raise QSCCIError("output directory already exists; overwrite is forbidden")
        if not self.repo_root.is_dir():
            raise QSCCIError("repo_root must be an existing directory")
        _assert_safe_path(self.repo_root, must_exist=True, directory=True)
        _assert_safe_path(self.output_dir.parent, must_exist=True, directory=True)
        _assert_public_result_ancestry(self.output_dir)
        if self.child_term_seconds != CHILD_TERM_SECONDS or self.child_kill_grace_seconds != CHILD_KILL_GRACE_SECONDS:
            raise QSCCIError("child TERM/KILL ceilings are frozen at 1900/30 seconds")
        if self.restoration_seconds != RESTORATION_SECONDS:
            raise QSCCIError("restoration ceiling is frozen at 600 seconds")


class ServiceController(Protocol):
    def observe_before(self) -> Mapping[str, Any]: ...

    def stop_if_required(self, observation: Mapping[str, Any]) -> Mapping[str, Any]: ...

    def restore_and_verify(
        self, before: Mapping[str, Any], stop_state: Mapping[str, Any], deadline: float
    ) -> Mapping[str, Any]: ...


class ChildLauncher(Protocol):
    def launch(self, config: SupervisorConfig, stage_dir: Path, lifecycle: Mapping[str, Any]) -> subprocess.Popen[bytes]: ...


def _http_bytes(specification: Mapping[str, Any], timeout: float) -> bytes:
    method = specification.get("method", "GET")
    if method not in ("GET", "POST"):
        raise QSCCIError("health endpoint method must be GET or POST")
    body = specification.get("json")
    data = None if body is None else canonical_json(body).rstrip(b"\n")
    request = urllib.request.Request(
        specification["url"], data=data, method=method,
        headers={"Content-Type": "application/json"} if data is not None else {},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            raw = response.read(1024 * 1024 + 1)
            if response.status != specification["expected_status"] or len(raw) > 1024 * 1024:
                raise QSCCIError(f"endpoint {specification['url']} failed bounded HTTP contract")
    except (OSError, urllib.error.URLError) as exc:
        raise QSCCIError(f"endpoint {specification['url']} failed: {exc}") from exc
    return raw


def _probe_health(specification: Mapping[str, Any], timeout: float) -> dict[str, Any]:
    expected = {"url", "method", "expected_status", "expected_body"}
    if set(specification) != expected or specification["method"] != "GET" or specification["expected_status"] != 200 or specification["expected_body"] != "":
        raise QSCCIError("health endpoint requires exact GET/200/empty-body expectation")
    raw = _http_bytes(specification, timeout)
    if raw != b"":
        raise QSCCIError("health endpoint body is not exactly empty")
    return {"url": specification["url"], "method": "GET", "status": 200, "body": "", "sha256": sha256_bytes(raw)}


def _json_payload(specification: Mapping[str, Any], timeout: float) -> tuple[bytes, Any]:
    raw = _http_bytes(specification, timeout)
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise QSCCIError(f"endpoint {specification['url']} returned duplicate key {key!r}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise QSCCIError(f"endpoint {specification['url']} returned nonfinite JSON constant {value}")

    try:
        return raw, json.loads(raw.decode("utf-8"), object_pairs_hook=pairs, parse_constant=reject_constant)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QSCCIError(f"endpoint {specification['url']} returned invalid JSON") from exc


def _probe_models(specification: Mapping[str, Any], timeout: float) -> dict[str, Any]:
    if set(specification) != {"url", "method", "expected_status", "expected_model_id"} or specification["method"] != "GET" or specification["expected_status"] != 200 or not isinstance(specification["expected_model_id"], str) or not specification["expected_model_id"]:
        raise QSCCIError("models endpoint contract is incomplete")
    raw, payload = _json_payload(specification, timeout)
    if not isinstance(payload, dict) or set(payload) != {"object", "data"} or payload["object"] != "list" or not isinstance(payload["data"], list):
        raise QSCCIError("models endpoint payload is outside the exact OpenAI list shape")
    identities = []
    for item in payload["data"]:
        if not isinstance(item, dict) or item.get("id") is None:
            raise QSCCIError("models endpoint contains an invalid model record")
        identities.append(item["id"])
    if identities != [specification["expected_model_id"]]:
        raise QSCCIError("models endpoint does not expose exactly the pinned model identity")
    return {"url": specification["url"], "method": "GET", "status": 200, "model_id": identities[0], "sha256": sha256_bytes(raw)}


def _probe_completion(specification: Mapping[str, Any], timeout: float) -> dict[str, Any]:
    expected_keys = {"url", "method", "json", "expected_status", "expected_model_id", "expected_text"}
    if set(specification) != expected_keys or specification["method"] != "POST" or specification["expected_status"] != 200:
        raise QSCCIError("completion endpoint contract is incomplete")
    if not isinstance(specification["json"], dict) or not specification["json"] or not isinstance(specification["expected_text"], str) or not specification["expected_text"]:
        raise QSCCIError("completion request and semantic expectation must be non-empty")
    raw, payload = _json_payload(specification, timeout)
    if not isinstance(payload, dict) or payload.get("model") != specification["expected_model_id"]:
        raise QSCCIError("completion response model identity differs")
    choices = payload.get("choices")
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        raise QSCCIError("completion response must contain exactly one choice")
    choice = choices[0]
    text = choice.get("text")
    if text is None and isinstance(choice.get("message"), dict):
        text = choice["message"].get("content")
    if text != specification["expected_text"]:
        raise QSCCIError("completion semantic predicate failed")
    return {"url": specification["url"], "method": "POST", "status": 200, "model_id": payload["model"], "text": text, "sha256": sha256_bytes(raw)}


def _probe_idle(specification: Mapping[str, Any], timeout: float) -> dict[str, Any]:
    expected_keys = {"url", "method", "expected_status", "running_metric", "waiting_metric"}
    if set(specification) != expected_keys or specification["method"] != "GET" or specification["expected_status"] != 200:
        raise QSCCIError("idle endpoint contract is incomplete")
    names = (specification["running_metric"], specification["waiting_metric"])
    if any(not isinstance(name, str) or not name for name in names) or names[0] == names[1]:
        raise QSCCIError("idle metric names must be two distinct non-empty strings")
    raw = _http_bytes(specification, timeout)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise QSCCIError("Prometheus endpoint is not UTF-8") from exc
    sums = {name: [] for name in names}
    pattern = re.compile(r"^([a-zA-Z_:][a-zA-Z0-9_:]*)(?:\{[^\r\n]*\})?\s+([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)$")
    for line in text.splitlines():
        match = pattern.match(line.strip())
        if match and match.group(1) in sums:
            value = float(match.group(2))
            if not math.isfinite(value) or value < 0.0:
                raise QSCCIError("idle metric is nonfinite")
            sums[match.group(1)].append(value)
    if any(not values or math.fsum(values) != 0.0 for values in sums.values()):
        raise QSCCIError("DeepSeek running/waiting request counters are not both exactly zero")
    return {"url": specification["url"], "method": "GET", "status": 200, "metric_sums": {name: math.fsum(sums[name]) for name in names}, "sha256": sha256_bytes(raw)}


def _run_argv(argv: Sequence[str], timeout: float, *, require_stdout: bool = False) -> dict[str, Any]:
    if not isinstance(argv, list) or not argv or any(not isinstance(item, str) or not item for item in argv):
        raise QSCCIError("service command must be a non-empty argv string array")
    try:
        completed = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise QSCCIError(f"service command failed: {type(exc).__name__}: {exc}") from exc
    evidence = {
        "argv_sha256": sha256_bytes(canonical_json(list(argv))), "returncode": completed.returncode,
        "stdout_sha256": sha256_bytes(completed.stdout), "stderr_sha256": sha256_bytes(completed.stderr),
    }
    if completed.returncode != 0:
        raise QSCCIError(f"service command returned {completed.returncode}; evidence={evidence}")
    if len(completed.stdout) > 64 * 1024 or len(completed.stderr) > 64 * 1024:
        raise QSCCIError("service command output exceeds the 64 KiB evidence bound")
    try:
        evidence["stdout"] = completed.stdout.decode("utf-8").strip()
    except UnicodeDecodeError as exc:
        raise QSCCIError("service identity command stdout must be UTF-8") from exc
    if require_stdout and not evidence["stdout"]:
        raise QSCCIError("service identity command produced empty identity evidence")
    return evidence


class PlanServiceController:
    """Exact argv/HTTP lifecycle controller loaded from a local operator plan."""

    def __init__(self, plan: Mapping[str, Any]) -> None:
        expected = {
            "stop_required", "container_identity_commands", "idle_endpoint", "stop_commands", "start_commands",
            "health_endpoint", "models_endpoint", "completion_endpoint", "after_idle_endpoint",
        }
        if set(plan) != expected:
            raise QSCCIError("service plan keys differ from the closed lifecycle interface")
        if len(plan["container_identity_commands"]) != 2:
            raise QSCCIError("service plan must identify exactly two containers")
        if plan["container_identity_commands"][0] == plan["container_identity_commands"][1]:
            raise QSCCIError("service plan must use two distinct container identity commands")
        if type(plan["stop_required"]) is not bool:
            raise QSCCIError("service plan stop_required must be boolean")
        if plan["stop_required"] is False and (plan["stop_commands"] or plan["start_commands"]):
            raise QSCCIError("unchanged-service plan must not contain stop/start commands")
        if plan["stop_required"] is True and (not plan["stop_commands"] or not plan["start_commands"]):
            raise QSCCIError("stopped-service plan requires stop and start commands")
        # Validate every expectation at construction, before any service touch.
        if set(plan["health_endpoint"]) != {"url", "method", "expected_status", "expected_body"}:
            raise QSCCIError("health endpoint expectation is incomplete")
        if set(plan["models_endpoint"]) != {"url", "method", "expected_status", "expected_model_id"}:
            raise QSCCIError("models endpoint expectation is incomplete")
        if set(plan["completion_endpoint"]) != {"url", "method", "json", "expected_status", "expected_model_id", "expected_text"}:
            raise QSCCIError("completion endpoint expectation is incomplete")
        for key in ("idle_endpoint", "after_idle_endpoint"):
            if set(plan[key]) != {"url", "method", "expected_status", "running_metric", "waiting_metric"}:
                raise QSCCIError(f"{key} expectation is incomplete")
        model_id = plan["models_endpoint"]["expected_model_id"]
        if plan["completion_endpoint"]["expected_model_id"] != model_id:
            raise QSCCIError("models and completion endpoints must expect the same pinned model")
        if plan["health_endpoint"] != {
            "url": plan["health_endpoint"]["url"], "method": "GET", "expected_status": 200, "expected_body": ""
        }:
            raise QSCCIError("health expectation must be exactly GET/200/empty")
        if plan["models_endpoint"]["method"] != "GET" or plan["models_endpoint"]["expected_status"] != 200 or model_id != DEEPSEEK_MODEL_ID:
            raise QSCCIError("models expectation must require one non-empty pinned model ID")
        completion = plan["completion_endpoint"]
        if completion["method"] != "POST" or completion["expected_status"] != 200 or not isinstance(completion["json"], dict) or not completion["json"] or not isinstance(completion["expected_text"], str) or not completion["expected_text"]:
            raise QSCCIError("completion expectation must require an exact non-empty semantic response")
        if set(completion["json"]) != {"model", "prompt", "temperature", "max_tokens"} or completion["json"]["model"] != DEEPSEEK_MODEL_ID or not isinstance(completion["json"]["prompt"], str) or not completion["json"]["prompt"] or completion["json"]["temperature"] != 0 or type(completion["json"]["max_tokens"]) is not int or not 1 <= completion["json"]["max_tokens"] <= 32:
            raise QSCCIError("completion request must bind the frozen model and deterministic bounded inference")
        for key in ("idle_endpoint", "after_idle_endpoint"):
            idle = plan[key]
            if idle["method"] != "GET" or idle["expected_status"] != 200 or any(not isinstance(idle[field], str) or not idle[field] for field in ("running_metric", "waiting_metric")) or idle["running_metric"] == idle["waiting_metric"]:
                raise QSCCIError(f"{key} must require two distinct zero-valued Prometheus counters")
            if (idle["running_metric"], idle["waiting_metric"]) != (RUNNING_METRIC, WAITING_METRIC):
                raise QSCCIError(f"{key} must use the frozen vLLM running/waiting counters")
        for key in ("health_endpoint", "models_endpoint", "completion_endpoint", "idle_endpoint", "after_idle_endpoint"):
            if not isinstance(plan[key]["url"], str) or not plan[key]["url"].startswith(("http://", "https://")):
                raise QSCCIError(f"{key} URL must be explicit HTTP(S)")
        if plan["idle_endpoint"]["url"] != plan["after_idle_endpoint"]["url"]:
            raise QSCCIError("before/after idle counters must come from the same endpoint")
        self.stop_required = plan["stop_required"]
        self.plan = plan

    def evidence_contract(self) -> Mapping[str, Any]:
        return json.loads(canonical_json(self.plan).decode("utf-8"))

    def observe_before(self) -> Mapping[str, Any]:
        identities = [_run_argv(command, 30.0, require_stdout=True) for command in self.plan["container_identity_commands"]]
        if identities[0]["stdout"] == identities[1]["stdout"]:
            raise QSCCIError("service identity probes resolved to the same container")
        observations = {
            "health": _probe_health(self.plan["health_endpoint"], 30.0),
            "models": _probe_models(self.plan["models_endpoint"], 30.0),
            "idle": _probe_idle(self.plan["idle_endpoint"], 30.0),
        }
        return {"status": "AVAILABLE", "service_stopped": False, "container_identities": identities, "observations": observations}

    def stop_if_required(self, observation: Mapping[str, Any]) -> Mapping[str, Any]:
        if self.plan["stop_required"] is False:
            return {"service_stopped": False, "reason": "not_required"}
        commands = [_run_argv(command, 60.0) for command in self.plan["stop_commands"]]
        return {"service_stopped": True, "commands": commands}

    def restore_and_verify(
        self, before: Mapping[str, Any], stop_state: Mapping[str, Any], deadline: float
    ) -> Mapping[str, Any]:
        commands = []
        if stop_state.get("service_stopped") is True:
            for command in self.plan["start_commands"]:
                commands.append(_run_argv(command, max(1.0, deadline - time.monotonic())))
        elif self.plan["stop_required"] is True:
            raise QSCCIError("planned service lifecycle did not record its required stop")
        after_identities = [
            _run_argv(command, max(1.0, deadline - time.monotonic()), require_stdout=True)
            for command in self.plan["container_identity_commands"]
        ]
        if after_identities != before["container_identities"]:
            raise QSCCIError("restored container identities differ from before-state evidence")
        observations = {
            "health": _probe_health(self.plan["health_endpoint"], max(1.0, deadline - time.monotonic())),
            "models": _probe_models(self.plan["models_endpoint"], max(1.0, deadline - time.monotonic())),
            "completion": _probe_completion(self.plan["completion_endpoint"], max(1.0, deadline - time.monotonic())),
            "idle": _probe_idle(self.plan["after_idle_endpoint"], max(1.0, deadline - time.monotonic())),
        }
        if time.monotonic() > deadline:
            raise QSCCIError("service restoration exceeded 600 seconds")
        return {
            "status": "RESTORED" if stop_state.get("service_stopped") else "NOT_APPLICABLE",
            "service_stopped": bool(stop_state.get("service_stopped")), "restoration_verified": True,
            "start_commands": commands, "verification_observations": observations,
            "same_container_identity_evidence": after_identities,
        }


class SubprocessChildLauncher:
    def launch(self, config: SupervisorConfig, stage_dir: Path, lifecycle: Mapping[str, Any]) -> subprocess.Popen[bytes]:
        encoded = base64.urlsafe_b64encode(canonical_json(lifecycle)).decode("ascii")
        command = [sys.executable, str(config.repo_root / "run_qscci.py"), "worker", "--stage-dir", str(stage_dir), "--repo-root", str(config.repo_root), "--lifecycle-b64", encoded]
        options: dict[str, Any] = {"cwd": config.repo_root, "stdin": subprocess.DEVNULL, "stdout": subprocess.PIPE, "stderr": subprocess.PIPE}
        if os.name == "nt":
            options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            options["start_new_session"] = True
        return subprocess.Popen(command, **options)


def _official_platform_supported() -> bool:
    return os.name == "posix" and sys.platform.startswith("linux")


def _is_reparse(path: Path) -> bool:
    try:
        stat_result = path.lstat()
    except OSError as exc:
        raise QSCCIError(f"cannot inspect path safety for {path}: {exc}") from exc
    attributes = getattr(stat_result, "st_file_attributes", 0)
    return path.is_symlink() or bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))


def _assert_safe_path(path: Path, *, must_exist: bool, directory: bool | None = None) -> None:
    candidate = path.absolute()
    if must_exist and not candidate.exists():
        raise QSCCIError(f"required path does not exist: {candidate}")
    current = candidate if candidate.exists() else candidate.parent
    while True:
        if _is_reparse(current):
            raise QSCCIError(f"symlink/reparse/junction path is forbidden: {current}")
        if current.parent == current:
            break
        current = current.parent
    if candidate.exists() and directory is True and not candidate.is_dir():
        raise QSCCIError(f"path must be a directory: {candidate}")
    if candidate.exists() and directory is False and not candidate.is_file():
        raise QSCCIError(f"path must be a regular file: {candidate}")


def _assert_member(root: Path, member: Path) -> None:
    _assert_safe_path(root, must_exist=True, directory=True)
    _assert_safe_path(member, must_exist=True, directory=False)
    if member.parent.absolute() != root.absolute() or member.resolve(strict=True).parent != root.resolve(strict=True):
        raise QSCCIError("result member escapes its declared root")


def _assert_public_result_ancestry(path: Path) -> None:
    def private_component(component: str) -> bool:
        lowered = component.lower()
        return component.startswith(".") or "quarantine" in lowered or "stage" in lowered

    lexical = path.absolute()
    resolved = path.resolve(strict=path.exists())
    for label, candidate in (("lexical", lexical), ("resolved", resolved)):
        for component in candidate.parts:
            if component in (candidate.anchor, os.sep, ""):
                continue
            if private_component(component):
                raise QSCCIError(
                    f"public result {label} ancestry contains a hidden/stage/quarantine component"
                )


def _quarantine(stage: Path, quarantine: Path) -> None:
    if not stage.exists():
        return
    _assert_safe_path(stage, must_exist=True, directory=True)
    if stage.parent.absolute() != quarantine.parent.absolute() or quarantine.exists():
        raise QSCCIError("quarantine target is unsafe or already exists")
    os.rename(stage, quarantine)


def _fresh_quarantine_path(output_dir: Path) -> Path:
    while True:
        candidate = output_dir.parent / f".{output_dir.name}.quarantine-{secrets.token_hex(16)}"
        if not os.path.lexists(candidate):
            return candidate


def _quarantine_entry_no_follow(entry: Path, quarantine: Path) -> None:
    """Move one unexpected direct child entry without resolving its contents."""
    if not os.path.lexists(entry):
        return
    if (
        entry.absolute().parent != quarantine.absolute().parent
        or os.path.lexists(quarantine)
    ):
        raise QSCCIError("unexpected publication target quarantine is unsafe")
    _promote_noreplace(entry, quarantine)


def _signal_group(process: subprocess.Popen[bytes], *, kill: bool) -> None:
    if os.name == "nt":
        if process.poll() is not None:
            return
        if kill:
            process.kill()
        else:
            process.send_signal(signal.CTRL_BREAK_EVENT)
    else:
        try:
            os.killpg(process.pid, signal.SIGKILL if kill else signal.SIGTERM)
        except ProcessLookupError:
            pass


def _wait_child(process: subprocess.Popen[bytes], config: SupervisorConfig) -> tuple[bytes, bytes]:
    try:
        return process.communicate(timeout=config.child_term_seconds)
    except subprocess.TimeoutExpired:
        _signal_group(process, kill=False)
        try:
            stdout, stderr = process.communicate(timeout=config.child_kill_grace_seconds)
        except subprocess.TimeoutExpired:
            _signal_group(process, kill=True)
            try:
                stdout, stderr = process.communicate(timeout=30.0)
            except subprocess.TimeoutExpired as final_timeout:
                stdout = final_timeout.output if isinstance(final_timeout.output, bytes) else b""
                stderr = final_timeout.stderr if isinstance(final_timeout.stderr, bytes) else b""
        raise ChildExecutionFailure(
            "CHILD_TIMEOUT",
            process.returncode if type(process.returncode) is int else None,
            stdout,
            stderr,
            f"research child exceeded external ceiling; stdout_sha256={sha256_bytes(stdout)} stderr_sha256={sha256_bytes(stderr)}",
        )


def _redacted_utf8_tail(raw: bytes) -> tuple[str, list[str]]:
    # Child stderr is attacker-controlled and can encode secrets in arbitrary
    # identifiers, Unicode confusables, control sequences, or multiline
    # structures.  Retain only its byte count and full digest in the carrier.
    del raw
    return "", ["all_stderr_text_withheld"]


def _write_private_child_failure(stage: Path, failure: ChildExecutionFailure) -> None:
    # The child can mutate its pathname while it runs.  Revalidate the whole
    # stage ancestry immediately before any supervisor-authored diagnostic.
    _assert_safe_path(stage, must_exist=True, directory=True)
    target = stage / INVALID_RUN_NAME
    if (
        target.parent.absolute() != stage.absolute()
        or target.parent.resolve(strict=True) != stage.resolve(strict=True)
    ):
        raise QSCCIError("private INVALID_RUN target escaped the supervisor stage")
    if os.path.lexists(target) and (target.is_symlink() or _is_reparse(target) or not target.is_file()):
        _remove_no_follow(target)
    _withheld_text, redactions = _redacted_utf8_tail(failure.stderr)
    payload = {
        "schema_version": "qscci-private-invalid-run.v1",
        "protocol_id": PROTOCOL_ID,
        "status": "INVALID_RUN",
        "failure_category": failure.category,
        "returncode": failure.returncode,
        "stdout_byte_count": len(failure.stdout),
        "stdout_sha256": sha256_bytes(failure.stdout),
        "stderr_byte_count": len(failure.stderr),
        "stderr_full_sha256": sha256_bytes(failure.stderr),
        "diagnostic_text_withheld": True,
        "redactions_applied": redactions,
        "retention_scope": "PRIVATE_QUARANTINE_ONLY_NOT_PUBLIC_EVIDENCE",
    }
    _atomic_write(target, payload)


def _remove_no_follow(path: Path) -> None:
    """Remove one supervisor-reserved child collision without following links."""
    if not os.path.lexists(path):
        return
    # Test the symlink bit before any operation that follows it.  In
    # particular, Path.is_dir() is true for a link to a directory, while
    # rmdir(link) fails on POSIX and would leave child-controlled evidence.
    if path.is_symlink():
        path.unlink()
        return
    if _is_reparse(path):
        if path.is_dir():
            os.rmdir(path)
        else:
            path.unlink()
        return
    if path.is_dir():
        for entry in os.scandir(path):
            _remove_no_follow(Path(entry.path))
        os.rmdir(path)
        return
    # The name is reserved to the supervisor.  Unlink every other node type
    # (regular file, FIFO, socket, or device) without opening or following it.
    path.unlink()


def _atomic_write(path: Path, payload: Any) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_json(payload))
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _fsync_directory(path: Path) -> None:
    if os.name != "posix":
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _promote_noreplace(stage: Path, target: Path) -> None:
    if os.path.lexists(target):
        raise QSCCIError("publication target appeared before promotion")
    if os.name == "posix":
        import ctypes
        import errno

        libc = ctypes.CDLL(None, use_errno=True)
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise QSCCIError("renameat2(RENAME_NOREPLACE) is required")
        if renameat2(-100, os.fsencode(stage), -100, os.fsencode(target), 1) != 0:
            code = ctypes.get_errno()
            if code in (errno.EEXIST, errno.ENOTEMPTY):
                raise QSCCIError("publication target appeared before promotion")
            raise OSError(code, os.strerror(code))
    else:
        os.rename(stage, target)


def run_supervised(
    config: SupervisorConfig, service_controller: ServiceController, child_launcher: ChildLauncher
) -> dict[str, Any]:
    """Run, restore, verify, and atomically publish one experiment result."""
    config.validate()
    if isinstance(child_launcher, SubprocessChildLauncher) and not _official_platform_supported():
        raise QSCCIError("official real worker supervision requires Linux/POSIX process-group semantics on Spark")
    config.output_dir.parent.mkdir(parents=True, exist_ok=True)
    stage = config.output_dir.parent / f".{config.output_dir.name}.stage-{secrets.token_hex(16)}"
    quarantine = _fresh_quarantine_path(config.output_dir)
    before: Mapping[str, Any] = {}
    stopped: Mapping[str, Any] = {}
    restoration: Mapping[str, Any] = {}
    process: subprocess.Popen[bytes] | None = None
    stage.mkdir()
    _assert_safe_path(stage, must_exist=True, directory=True)
    if stage.parent.absolute() != config.output_dir.parent.absolute():
        raise QSCCIError("hidden stage escaped the output parent")
    failure: BaseException | None = None
    child_failure: ChildExecutionFailure | None = None
    contract = service_controller.evidence_contract() if hasattr(service_controller, "evidence_contract") else {}
    try:
        before = service_controller.observe_before()
        if getattr(service_controller, "stop_required", False) is True:
            # Conservative state passed to finally even if a multi-command stop
            # raises after its first side effect.
            stopped = {"service_stopped": True, "commands": [], "stop_attempted": True}
        stopped = service_controller.stop_if_required(before)
        lifecycle_for_child = {"contract": contract, "before": dict(before), "stop": dict(stopped), "after": {"status": "PENDING"}}
        process = child_launcher.launch(config, stage, lifecycle_for_child)
        stdout, stderr = _wait_child(process, config)
        if process.returncode != 0:
            raise ChildExecutionFailure(
                "CHILD_NONZERO",
                int(process.returncode),
                stdout,
                stderr,
                f"research child failed rc={process.returncode}; stdout_sha256={sha256_bytes(stdout)} stderr_sha256={sha256_bytes(stderr)}",
            )
    except BaseException as exc:
        failure = exc
        if isinstance(exc, ChildExecutionFailure):
            child_failure = exc
    finally:
        deadline = time.monotonic() + config.restoration_seconds
        try:
            restoration = service_controller.restore_and_verify(before, stopped, deadline)
            if restoration.get("restoration_verified") is not True:
                raise QSCCIError("service restoration did not verify")
        except BaseException as exc:
            failure = QSCCIError(f"service restoration failure: {type(exc).__name__}: {exc}")
    if os.path.lexists(config.output_dir):
        # The output name was absent before launch and is exclusively owned by
        # this run.  A child-created entry is private failed-run material, not
        # a publication; rename the entry itself without following it.
        try:
            _quarantine_entry_no_follow(config.output_dir, quarantine)
            quarantine = _fresh_quarantine_path(config.output_dir)
        except (OSError, QSCCIError) as exc:
            if failure is None:
                failure = QSCCIError(f"unexpected publication target could not be quarantined: {exc}")
        if failure is None:
            failure = QSCCIError("publication target appeared before supervisor publication")
    if failure is not None:
        if child_failure is not None:
            try:
                _write_private_child_failure(stage, child_failure)
            except (OSError, QSCCIError, ValueError, TypeError):
                # Diagnostic evidence is subordinate to restoration and the
                # original execution failure; never replace that failure.
                pass
        try:
            _quarantine(stage, quarantine)
        except (OSError, QSCCIError):
            # A hostile child may have replaced the stage name itself.  Never
            # follow that replacement during cleanup and never mask the
            # original execution/restoration failure with diagnostic cleanup.
            try:
                if os.path.lexists(stage) and (stage.is_symlink() or _is_reparse(stage)):
                    _remove_no_follow(stage)
            except (OSError, QSCCIError):
                pass
        raise failure
    promoted = False
    try:
        artifact_path = stage / ARTIFACT_NAME
        if sorted(path.name for path in stage.iterdir()) != [ARTIFACT_NAME]:
            raise QSCCIError("child stage has an incomplete or unexpected file set")
        _assert_member(stage, artifact_path)
        artifact = strict_json(artifact_path)
        artifact["service_lifecycle"] = {"contract": contract, "before": dict(before), "stop": dict(stopped), "after": dict(restoration)}
        artifact["artifact_digest"] = artifact_digest(artifact)
        _atomic_write(artifact_path, artifact)
        _assert_member(stage, artifact_path)
        verify_artifact(artifact, root=config.repo_root)
        manifest = {
            "protocol_id": PROTOCOL_ID, "status": "COMPLETE", "artifact": ARTIFACT_NAME,
            "artifact_sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
            "artifact_digest": artifact["artifact_digest"], "publication": "atomic_no_replace",
        }
        _atomic_write(stage / MANIFEST_NAME, manifest)
        receipt = {
            "protocol_id": PROTOCOL_ID, "status": "COMMITTED", "manifest": MANIFEST_NAME,
            "manifest_sha256": hashlib.sha256((stage / MANIFEST_NAME).read_bytes()).hexdigest(),
            "artifact_sha256": manifest["artifact_sha256"], "commit_is_last_write": True,
        }
        _atomic_write(stage / COMMIT_NAME, receipt)
        for name in (ARTIFACT_NAME, MANIFEST_NAME, COMMIT_NAME):
            _assert_member(stage, stage / name)
        _fsync_directory(stage)
        _promote_noreplace(stage, config.output_dir)
        promoted = True
        _fsync_directory(config.output_dir.parent)
        return {"published": True, "output_dir": str(config.output_dir), "disposition": artifact["disposition"]}
    except BaseException:
        if not promoted and os.path.lexists(config.output_dir):
            _quarantine_entry_no_follow(config.output_dir, quarantine)
            quarantine = _fresh_quarantine_path(config.output_dir)
        _quarantine(config.output_dir if promoted else stage, quarantine)
        if promoted:
            try:
                _fsync_directory(config.output_dir.parent)
            except OSError:
                # The run is already failed and the visible target has been
                # removed; a second parent-fsync failure cannot be recovered
                # inside this process.
                pass
        raise


def verify_result_dir(path: Path, repo_root: Path) -> dict[str, Any]:
    _assert_public_result_ancestry(path)
    _assert_safe_path(path, must_exist=True, directory=True)
    _assert_safe_path(repo_root, must_exist=True, directory=True)
    if sorted(item.name for item in path.iterdir()) != sorted([ARTIFACT_NAME, MANIFEST_NAME, COMMIT_NAME]):
        raise QSCCIError("result directory is not the exact committed three-file set")
    for item in path.iterdir():
        _assert_member(path, item)
    artifact, manifest, receipt = (strict_json(path / name) for name in (ARTIFACT_NAME, MANIFEST_NAME, COMMIT_NAME))
    if (path / ARTIFACT_NAME).read_bytes() != canonical_json(artifact) or (path / MANIFEST_NAME).read_bytes() != canonical_json(manifest) or (path / COMMIT_NAME).read_bytes() != canonical_json(receipt):
        raise QSCCIError("result files are not canonical JSON")
    artifact_sha = hashlib.sha256((path / ARTIFACT_NAME).read_bytes()).hexdigest()
    manifest_sha = hashlib.sha256((path / MANIFEST_NAME).read_bytes()).hexdigest()
    if manifest != {"protocol_id": PROTOCOL_ID, "status": "COMPLETE", "artifact": ARTIFACT_NAME, "artifact_sha256": artifact_sha, "artifact_digest": artifact["artifact_digest"], "publication": "atomic_no_replace"}:
        raise QSCCIError("manifest does not bind the artifact")
    if receipt != {"protocol_id": PROTOCOL_ID, "status": "COMMITTED", "manifest": MANIFEST_NAME, "manifest_sha256": manifest_sha, "artifact_sha256": artifact_sha, "commit_is_last_write": True}:
        raise QSCCIError("commit receipt does not bind manifest and artifact")
    result = verify_artifact(artifact, root=repo_root)
    return {**result, "committed_result_dir": True}


def _verify_archive_distribution_file(repo_root: Path, relative_path: Path, expected_sha256: str) -> None:
    path = repo_root / relative_path
    _assert_safe_path(repo_root, must_exist=True, directory=True)
    _assert_safe_path(path, must_exist=True, directory=False)
    if repo_root.resolve(strict=True) not in path.resolve(strict=True).parents:
        raise QSCCIError("archive-verifier distribution file escapes repository root")
    if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha256:
        raise QSCCIError(f"archive-verifier distribution digest mismatch: {relative_path}")


def _strict_canonical_archive_member(path: Path) -> tuple[bytes, Any]:
    raw = path.read_bytes()
    value = strict_json(path)
    try:
        canonical = canonical_json(value)
    except (TypeError, ValueError) as exc:
        raise QSCCIError(f"archive member contains a noncanonical JSON value: {path.name}") from exc
    if raw != canonical:
        raise QSCCIError(f"archive member is not canonical JSON: {path.name}")
    return raw, value


def verify_v4_archive(result_directory: Path) -> dict[str, object]:
    """Verify custody and cache-independent semantics of official v4 run-001.

    This deliberately does not call or weaken ``verify_result_dir``.  Archive
    success proves the copied three-file evidence matches pinned official bytes;
    it does not prove a current publication target, service state, or scientific
    reexecution on this machine.
    """

    root = Path(result_directory)
    _assert_public_result_ancestry(root)
    _assert_safe_path(root, must_exist=True, directory=True)
    resolved_root = root.resolve(strict=True)
    expected_names = {ARTIFACT_NAME, MANIFEST_NAME, COMMIT_NAME}
    entries = list(root.iterdir())
    if {entry.name for entry in entries} != expected_names or len(entries) != len(expected_names):
        raise QSCCIError("v4 archive is not the exact committed three-file set")
    for entry in entries:
        _assert_member(root, entry)
        if entry.resolve(strict=True).parent != resolved_root:
            raise QSCCIError("v4 archive member escapes archive root")

    repo_root = Path(__file__).resolve().parent
    _verify_archive_distribution_file(repo_root, SCHEMA_PATH, SCHEMA_SHA256)
    _verify_archive_distribution_file(
        repo_root, V4_ARCHIVE_ADDENDUM_PATH, V4_ARCHIVE_ADDENDUM_SHA256
    )

    artifact_raw, artifact = _strict_canonical_archive_member(root / ARTIFACT_NAME)
    manifest_raw, manifest = _strict_canonical_archive_member(root / MANIFEST_NAME)
    commit_raw, receipt = _strict_canonical_archive_member(root / COMMIT_NAME)
    observed_roots = {
        "artifact": hashlib.sha256(artifact_raw).hexdigest(),
        "manifest": hashlib.sha256(manifest_raw).hexdigest(),
        "commit": hashlib.sha256(commit_raw).hexdigest(),
    }
    expected_roots = {
        "artifact": V4_ARCHIVE_ARTIFACT_SHA256,
        "manifest": V4_ARCHIVE_MANIFEST_SHA256,
        "commit": V4_ARCHIVE_COMMIT_SHA256,
    }
    if observed_roots != expected_roots:
        raise QSCCIError("v4 archive custody root mismatch")

    try:
        import jsonschema
    except ImportError as exc:
        raise QSCCIError("jsonschema is required for portable archive verification") from exc
    schema = strict_json(repo_root / SCHEMA_PATH)
    try:
        jsonschema.Draft202012Validator.check_schema(schema)
        jsonschema.Draft202012Validator(schema).validate(artifact)
    except jsonschema.exceptions.SchemaError as exc:
        raise QSCCIError("frozen v4 artifact schema is invalid") from exc
    except jsonschema.exceptions.ValidationError as exc:
        raise QSCCIError("archived artifact fails the frozen v4 schema") from exc

    if (
        artifact.get("protocol_id") != PROTOCOL_ID
        or artifact.get("status") != "COMPLETE"
        or artifact.get("scientific_claim_status") != "UNCONFIRMED"
        or artifact.get("novelty_claim_status") != "UNCONFIRMED"
        or artifact.get("protocol_file_sha256") != PROTOCOL_SHA256
        or artifact.get("fixture_sha256") != FIXTURE_SHA256
        or artifact.get("schema_file_sha256") != SCHEMA_SHA256
        or artifact.get("disposition") != V4_ARCHIVE_DISPOSITION
        or artifact.get("failures") != []
    ):
        raise QSCCIError("archived artifact identity, claim boundary, or disposition differs")
    if (
        artifact.get("artifact_digest") != V4_ARCHIVE_ARTIFACT_DIGEST
        or artifact_digest(artifact) != V4_ARCHIVE_ARTIFACT_DIGEST
    ):
        raise QSCCIError("archived artifact internal digest mismatch")
    verify_artifact_cache_independent(artifact, root=repo_root)

    expected_manifest = {
        "protocol_id": PROTOCOL_ID,
        "status": "COMPLETE",
        "artifact": ARTIFACT_NAME,
        "artifact_sha256": V4_ARCHIVE_ARTIFACT_SHA256,
        "artifact_digest": V4_ARCHIVE_ARTIFACT_DIGEST,
        "publication": "atomic_no_replace",
    }
    if manifest != expected_manifest:
        raise QSCCIError("archived manifest does not exactly bind the official artifact")
    expected_receipt = {
        "protocol_id": PROTOCOL_ID,
        "status": "COMMITTED",
        "manifest": MANIFEST_NAME,
        "manifest_sha256": V4_ARCHIVE_MANIFEST_SHA256,
        "artifact_sha256": V4_ARCHIVE_ARTIFACT_SHA256,
        "commit_is_last_write": True,
    }
    if receipt != expected_receipt:
        raise QSCCIError("archived commit receipt does not exactly bind manifest and artifact")

    return {
        "status": "ARCHIVED_QSCCI_V4_VERIFIED",
        "protocol_id": PROTOCOL_ID,
        "disposition": V4_ARCHIVE_DISPOSITION,
        "scientific_claim_status": "UNCONFIRMED",
        "novelty_claim_status": "UNCONFIRMED",
        "artifact_sha256": V4_ARCHIVE_ARTIFACT_SHA256,
        "manifest_sha256": V4_ARCHIVE_MANIFEST_SHA256,
        "commit_sha256": V4_ARCHIVE_COMMIT_SHA256,
        "artifact_digest": V4_ARCHIVE_ARTIFACT_DIGEST,
        "custody_only": True,
        "reexecution_verified": False,
        "current_target_verified": False,
        "live_service_lifecycle_verified": False,
        "pinned_model_sae_cache_verified": False,
    }


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-v4-archive", type=Path)
    commands = parser.add_subparsers(dest="command")
    worker = commands.add_parser("worker")
    worker.add_argument("--stage-dir", required=True, type=Path)
    worker.add_argument("--repo-root", required=True, type=Path)
    worker.add_argument("--lifecycle-b64", required=True)
    run = commands.add_parser("run")
    run.add_argument("--output-dir", required=True, type=Path)
    run.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent)
    run.add_argument("--service-plan", type=Path)
    run.add_argument("--allow-real-model", action="store_true")
    verify = commands.add_parser("verify")
    verify.add_argument("--result-dir", required=True, type=Path)
    verify.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args(argv)
    if (args.verify_v4_archive is None) == (args.command is None):
        parser.error("choose exactly one command or --verify-v4-archive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        if args.verify_v4_archive is not None:
            print(json.dumps(verify_v4_archive(args.verify_v4_archive.absolute()), sort_keys=True))
        elif args.command == "worker":
            lifecycle = json.loads(base64.urlsafe_b64decode(args.lifecycle_b64.encode("ascii")).decode("utf-8"))
            stage_dir, repo_root = args.stage_dir.absolute(), args.repo_root.absolute()
            _assert_safe_path(stage_dir, must_exist=True, directory=True)
            _assert_safe_path(repo_root, must_exist=True, directory=True)
            artifact = run_worker(stage_dir, repo_root, lifecycle)
            print(json.dumps({"status": artifact["status"], "disposition": artifact["disposition"]}, sort_keys=True))
        elif args.command == "verify":
            print(json.dumps(verify_result_dir(args.result_dir.absolute(), args.repo_root.absolute()), sort_keys=True))
        else:
            if not args.allow_real_model:
                raise QSCCIError("run requires explicit --allow-real-model opt-in")
            controller: ServiceController
            if args.service_plan is None:
                raise QSCCIError("run requires --service-plan so before/after DeepSeek availability is real evidence")
            service_plan = args.service_plan.absolute()
            _assert_safe_path(service_plan, must_exist=True, directory=False)
            controller = PlanServiceController(strict_json(service_plan))
            result = run_supervised(
                SupervisorConfig(args.output_dir.absolute(), args.repo_root.absolute()), controller, SubprocessChildLauncher()
            )
            print(json.dumps(result, sort_keys=True))
        return 0
    except (QSCCIError, OSError, ValueError, TypeError) as exc:
        print(f"INVALID_RUN: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
