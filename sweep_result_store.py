"""Append-only storage for large-sweep results.

Each result is one JSON object on its own line. The JSON array file is
rewritten once, from that log, instead of being read and rewritten on every
iteration.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def append_jsonl(path: str | Path, entry: dict[str, Any]) -> None:
    """Append one result. This does not read the existing log or the array."""
    destination = Path(path)
    line = json.dumps(entry, separators=(",", ":"), sort_keys=True) + "\n"
    with destination.open("a", encoding="utf-8") as handle:
        handle.write(line)
        handle.flush()
        os.fsync(handle.fileno())


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL log once. A bad line fails closed."""
    source = Path(path)
    records: list[dict[str, Any]] = []
    with source.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number} is not a JSON object") from exc
            if not isinstance(payload, dict):
                raise ValueError(f"{source}:{line_number} is not a JSON object")
            records.append(payload)
    return records


def materialize_json_array(jsonl_path: str | Path, json_path: str | Path) -> int:
    """Write the JSON array once, replacing it only after the temp file is complete."""
    records = read_jsonl(jsonl_path)
    destination = Path(json_path)
    temporary = destination.with_name(destination.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, destination)
    return len(records)


def migrate_json_array_to_jsonl(json_path: str | Path, jsonl_path: str | Path) -> int:
    """Copy an existing JSON array into the log once.

    If the log already exists, it is left unchanged. Later sweep steps append
    to the log and do not read the array again.
    """
    log_path = Path(jsonl_path)
    if log_path.exists():
        return 0
    source = Path(json_path)
    if not source.exists():
        return 0
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"{source} is not a JSON array")
    for entry in payload:
        if not isinstance(entry, dict):
            raise ValueError(f"{source} contains a non-object result")
        append_jsonl(log_path, entry)
    return len(payload)
