"""Public surface of the zero-services chelated-kernel wheel."""

from __future__ import annotations

import sys
from importlib.abc import MetaPathFinder

from chelated_kernel.errors import UnsupportedCapabilityError
from chelated_kernel.event_ledger import EventLedger
from chelated_kernel.index_wrapper import wrap_index
from chelated_kernel.profiles import negotiate_capability
from chelated_kernel.stream_guard import wrap_stream

__version__ = "1.0.0"


class _PrivateImportBlocker(MetaPathFinder):
    """Fail closed on chelated_kernel._* without blocking the host repository."""

    def find_spec(self, fullname: str, path: object, target: object = None) -> None:  # type: ignore[override]
        if fullname == "chelated_kernel" or not fullname.startswith("chelated_kernel."):
            return None
        parts = fullname.split(".")
        if any(part.startswith("_") for part in parts):
            raise ImportError(f"private import blocked: {fullname}")
        return None


def _install_private_import_blocker() -> None:
    if any(isinstance(finder, _PrivateImportBlocker) for finder in sys.meta_path):
        return
    sys.meta_path.insert(0, _PrivateImportBlocker())


_install_private_import_blocker()
negotiate_capability.UnsupportedCapabilityError = UnsupportedCapabilityError  # type: ignore[attr-defined]

__all__ = ["wrap_index", "EventLedger", "wrap_stream", "negotiate_capability"]
