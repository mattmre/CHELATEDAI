"""Run a public-safe deterministic control matrix against an OpenAI-compatible model.

The endpoint remains operator-private and is deliberately omitted from the
report. These fixtures are capability/admission controls, not research tasks
and not evidence that an EGV campaign arm passed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import tempfile
import time
from typing import Any
from urllib import error, request


SCHEMA_VERSION = "egv-model-control-matrix-v1"
SEED = 1729
DEFAULT_MAX_TOKENS = 1024
SYSTEM_MESSAGE = (
    "Follow the user instruction exactly. Return only the JSON object required "
    "by the supplied strict response schema."
)


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False) + "\n").encode(
        "utf-8"
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _http_json(url: str, *, body: Any | None = None, timeout: float) -> tuple[int, Any]:
    encoded = None if body is None else _canonical(body)
    headers = {} if encoded is None else {"Content-Type": "application/json"}
    call = request.Request(url, data=encoded, headers=headers, method="GET" if body is None else "POST")
    try:
        with request.urlopen(call, timeout=timeout) as response:
            return response.status, json.loads(response.read().decode("utf-8"))
    except error.HTTPError as exc:
        payload = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError("model control request failed with HTTP {}: {}".format(exc.code, payload)) from exc


def _case(
    case_id: str,
    instruction: str,
    schema: dict[str, Any],
    expected: Any,
) -> dict[str, Any]:
    return {
        "case_id": case_id,
        "instruction": instruction,
        "schema": schema,
        "expected": expected,
    }


CASES = (
    _case(
        "integer-arithmetic",
        "Compute 19 + 23. Set answer to the resulting integer.",
        {
            "type": "object",
            "properties": {"answer": {"type": "integer"}},
            "required": ["answer"],
            "additionalProperties": False,
        },
        {"answer": 42},
    ),
    _case(
        "stable-sort",
        "Sort the integers [3, 1, 2, 1] in ascending order, preserving duplicates. Set values to the result.",
        {
            "type": "object",
            "properties": {
                "values": {"type": "array", "items": {"type": "integer"}}
            },
            "required": ["values"],
            "additionalProperties": False,
        },
        {"values": [1, 1, 2, 3]},
    ),
    _case(
        "preserve-empty-token",
        "Split the exact text alpha||||omega at every exact || delimiter. Preserve the empty token. Set tokens to the result.",
        {
            "type": "object",
            "properties": {
                "tokens": {"type": "array", "items": {"type": "string"}}
            },
            "required": ["tokens"],
            "additionalProperties": False,
        },
        {"tokens": ["alpha", "", "omega"]},
    ),
    _case(
        "state-transition",
        "A state machine is in integer state 7. The event is advance and the public advance amount is 4. Set state to the new state.",
        {
            "type": "object",
            "properties": {"state": {"type": "integer"}},
            "required": ["state"],
            "additionalProperties": False,
        },
        {"state": 11},
    ),
    _case(
        "mapping-default",
        "The mapping is {\"present\": \"yes\"}. Look up absent and use the public default missing:fixture when absent. Set value to the result.",
        {
            "type": "object",
            "properties": {"value": {"type": "string"}},
            "required": ["value"],
            "additionalProperties": False,
        },
        {"value": "missing:fixture"},
    ),
)


def _run_case(
    base_url: str,
    model: str,
    case: dict[str, Any],
    timeout: float,
    max_tokens: int,
) -> dict[str, Any]:
    messages = [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": case["instruction"]},
    ]
    body = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "seed": SEED,
        "max_tokens": max_tokens,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": case["case_id"].replace("-", "_"),
                "schema": case["schema"],
                "strict": True,
            },
        },
    }
    started = time.monotonic()
    status, payload = _http_json(
        base_url.rstrip("/") + "/v1/chat/completions", body=body, timeout=timeout
    )
    elapsed = time.monotonic() - started
    message = payload["choices"][0]["message"]
    content = message.get("content")
    usage = payload.get("usage", {})
    reasoning = message.get("reasoning_content") or message.get("reasoning") or ""
    parse_error = None
    observed = None
    if not isinstance(content, str):
        parse_error = "missing-text-content"
    else:
        try:
            observed = json.loads(content)
        except json.JSONDecodeError:
            parse_error = "content-not-json"
    return {
        "case_id": case["case_id"],
        "request_digest": _digest(body),
        "http_status": status,
        "elapsed_seconds": round(elapsed, 6),
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_returned_separately": bool(reasoning),
        "response_error": parse_error,
        "expected": case["expected"],
        "observed": observed,
        "exact_pass": observed == case["expected"],
    }


def _atomic_write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", delete=False, dir=path.parent) as handle:
        temporary = Path(handle.name)
        handle.write(_canonical(value))
        handle.flush()
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--engine-version", required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--timeout", default=180.0, type=float)
    parser.add_argument("--max-tokens", default=DEFAULT_MAX_TOKENS, type=int)
    args = parser.parse_args()
    if not 1 <= args.max_tokens <= 8192:
        parser.error("--max-tokens must be between 1 and 8192")

    health_started = time.monotonic()
    health_call = request.Request(args.base_url.rstrip("/") + "/health", method="GET")
    with request.urlopen(health_call, timeout=args.timeout) as response:
        health_status = response.status
    health_seconds = time.monotonic() - health_started
    models_status, models_payload = _http_json(
        args.base_url.rstrip("/") + "/v1/models", timeout=args.timeout
    )
    observed_ids = sorted(
        item.get("id") for item in models_payload.get("data", []) if isinstance(item.get("id"), str)
    )
    cases = []
    for case in CASES:
        try:
            cases.append(
                _run_case(
                    args.base_url,
                    args.model,
                    case,
                    args.timeout,
                    args.max_tokens,
                )
            )
        except (RuntimeError, error.URLError, TimeoutError):
            cases.append(
                {
                    "case_id": case["case_id"],
                    "response_error": "request-failed",
                    "expected": case["expected"],
                    "observed": None,
                    "exact_pass": False,
                }
            )
    report: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "evidence_scope": "bounded-public-fixture-control-not-task-bearing-research",
        "endpoint_disclosure": "omitted-operator-private",
        "model": {
            "repository": args.model,
            "revision": args.revision,
            "engine_version": args.engine_version,
            "precision": args.precision,
        },
        "generation": {
            "temperature": 0,
            "seed": SEED,
            "max_tokens": args.max_tokens,
            "strict_json_schema": True,
        },
        "service": {
            "health_http_status": health_status,
            "health_elapsed_seconds": round(health_seconds, 6),
            "models_http_status": models_status,
            "exact_model_id_observed": args.model in observed_ids,
            "model_count": len(observed_ids),
        },
        "cases": cases,
        "summary": {
            "case_count": len(cases),
            "exact_pass_count": sum(case["exact_pass"] for case in cases),
            "all_exact_passed": all(case["exact_pass"] for case in cases),
        },
    }
    report["report_digest"] = _digest(report)
    _atomic_write(args.output, report)
    print(json.dumps(report["summary"], sort_keys=True))
    return 0 if report["summary"]["all_exact_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
