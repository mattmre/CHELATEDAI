"""Process-boundary controls for the production evaluator.

The evaluator is a receipt producer, never a SQLite writer.  This hook is
installed before evaluator construction and is intentionally irreversible for
the lifetime of the process: native ``sqlite3.connect`` calls remain denied
even if application code tries to replace a Python helper.
"""

from __future__ import annotations

import sys
from typing import Any, Dict, List, Optional, Sequence

_EVALUATOR_PROCESS_ENFORCEMENT = False


def mark_evaluator_process() -> None:
    """Bind this interpreter to the receipt-only evaluator role.

    The role is set by the spawned evaluator entry point before importing or
    constructing evaluator state.  Direct unit helpers may still construct a
    controller without poisoning the parent test interpreter with a global
    SQLite denial hook; they are not production evaluator processes.
    """

    global _EVALUATOR_PROCESS_ENFORCEMENT
    _EVALUATOR_PROCESS_ENFORCEMENT = True


def install_evaluator_sqlite_jail(counter: Optional[List[int]] = None) -> Dict[str, Any]:
    """Install the immutable SQLite audit boundary and return probe metadata."""

    calls = counter if counter is not None else [0]
    enforcing = _EVALUATOR_PROCESS_ENFORCEMENT

    def deny_sqlite(event: str, _args: Sequence[Any]) -> None:
        if enforcing and event == "sqlite3.connect":
            calls[0] += 1
            raise PermissionError("EGV_EVALUATOR_SQLITE_DENIED")

    sys.addaudithook(deny_sqlite)
    return {"event": "sqlite3.connect", "hook_installed": True, "enforcing": enforcing}


__all__ = ["install_evaluator_sqlite_jail", "mark_evaluator_process"]
