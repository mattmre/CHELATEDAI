"""Shared marker for experimental computational-storage POC modules.

Each POC module that is NOT consumed by a production code path imports
``mark_experimental`` and calls it at module import time. The call:

  - reads the module's own ``EXPERIMENTAL`` constant (so the constant is
    actually load-bearing, not just a static comment), and
  - emits a single ``ExperimentalPOCWarning`` (a ``DeprecationWarning``
    subclass) per module per process, naming the module.

This closes the L13 / L4-light gap flagged by Tier B on PR #247: prior to
this helper, the ``EXPERIMENTAL = True`` constants were read by nothing in
the repo, so they were effectively comments-with-extra-steps. Now the
constants are part of an actual runtime check — flip one to ``False`` and
the warning stops; flip it to a non-bool and the helper raises ``TypeError``.

This module itself is part of the POC scaffolding; production code should
not import from it.
"""

from __future__ import annotations

import warnings


class ExperimentalPOCWarning(DeprecationWarning):
    """Raised once per experimental POC module at import time.

    Subclasses ``DeprecationWarning`` so it is visible under ``-W default``
    but not under ``-W ignore::DeprecationWarning`` (production CI). Operators
    who explicitly want to see these can run with ``-W default``.
    """


_ALREADY_WARNED: set[str] = set()


def mark_experimental(module_name: str, flag: bool) -> None:
    """Read the module's EXPERIMENTAL flag and emit a warning if True.

    Parameters
    ----------
    module_name : str
        Pass ``__name__`` from the caller — used in the warning text and to
        deduplicate within a single process.
    flag : bool
        Pass the module's own ``EXPERIMENTAL`` constant. If ``False``, no
        warning is emitted; if ``True``, the warning fires once. Any non-bool
        value raises ``TypeError`` immediately so a typo in the constant
        cannot silently downgrade an experimental module to "looks-stable".

    Raises
    ------
    TypeError
        If ``flag`` is not a bool — protects against ``EXPERIMENTAL = "yes"``
        or ``EXPERIMENTAL = 1`` accidentally counting as truthy without being
        machine-checkable.
    """
    if not isinstance(flag, bool):
        raise TypeError(
            f"{module_name}.EXPERIMENTAL must be a bool, got {type(flag).__name__}"
        )
    if not flag:
        return
    if module_name in _ALREADY_WARNED:
        return
    _ALREADY_WARNED.add(module_name)
    warnings.warn(
        f"{module_name} is a research-stage POC and not consumed by any "
        f"production code path. Do not depend on its API in operator-facing "
        f"surfaces. See computational_storage_poc/README.md (## Status).",
        ExperimentalPOCWarning,
        stacklevel=3,
    )
