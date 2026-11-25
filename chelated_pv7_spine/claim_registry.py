"""Permanent bundle reservations and confirmatory Holm adjudication."""

from collections.abc import Mapping, Sequence
from copy import deepcopy
from math import isfinite
from pathlib import Path

from . import _json_sha256, _read_json

REGISTRY_PATH = Path(__file__).resolve().parents[1] / "docs/research/pv7_claim_registry_v1.json"
BUNDLE_IDS = frozenset(range(47, 65))
STATUSES = frozenset({"verified-pass", "verified-loss", "unsupported", "protocol-ready", "reserved-pending"})
HOLM_ALPHA = 0.05
HOLM_FAMILY = ("3A", "H6B_merge", "H6B_bridge_shift_and_4hop", "6C", "6D")
BRIDGE_COMPONENTS = frozenset({"prompt_shift", "four_hop"})
_RESERVATION_SHA256 = "61350baa1c7e0ba36ee4beb07066f88241f447bee4f5acd72c3bef68b678ff24"
_FINAL_STATUSES = frozenset({"verified-pass", "verified-loss", "unsupported"})


def validate_claim_registry(registry):
    """Require every permanent reservation and check its immutable fields."""
    if not isinstance(registry, dict) or not isinstance(registry.get("bundles"), list):
        raise ValueError("Expected a claim registry object with a bundles list")
    canonical = deepcopy(registry)
    seen = set()
    for row in canonical["bundles"]:
        bundle_id = row.get("id") if isinstance(row, dict) else None
        if type(bundle_id) is not int or bundle_id not in BUNDLE_IDS or bundle_id in seen:
            raise ValueError(f"Invalid or duplicate permanent bundle ID: {bundle_id!r}")
        status = row.pop("status", None)
        if not isinstance(status, str) or status not in STATUSES:
            raise ValueError(f"Bundle {bundle_id} has an invalid or vacated status: {status!r}")
        seen.add(bundle_id)
    if seen != BUNDLE_IDS:
        raise ValueError(f"Missing permanent bundle IDs: {sorted(BUNDLE_IDS - seen)}")
    canonical["bundles"].sort(key=lambda row: row["id"])
    if _json_sha256(canonical) != _RESERVATION_SHA256:
        raise ValueError("Permanent claim registry metadata has changed")


def load_claim_registry(path=REGISTRY_PATH):
    """Read and validate the complete version-one reservation table."""
    registry = _read_json(Path(path).read_text(encoding="utf-8"))
    validate_claim_registry(registry)
    return registry


def claim_registry_sha256(registry=None):
    """Hash the complete validated registry, including current statuses."""
    registry = load_claim_registry() if registry is None else registry
    validate_claim_registry(registry)
    canonical = {**registry, "bundles": sorted(registry["bundles"], key=lambda row: row["id"])}
    return _json_sha256(canonical)


def with_bundle_status(registry, bundle_id, status):
    """Return an updated copy while preserving final adjudications."""
    validate_claim_registry(registry)
    if type(bundle_id) is not int or bundle_id not in BUNDLE_IDS:
        raise ValueError(f"Unknown permanent bundle ID: {bundle_id!r}")
    if not isinstance(status, str) or status not in STATUSES:
        raise ValueError(f"Invalid bundle status: {status!r}")
    updated = deepcopy(registry)
    row = next(row for row in updated["bundles"] if row["id"] == bundle_id)
    previous = row["status"]
    if previous != status and (previous in _FINAL_STATUSES or status == "reserved-pending"):
        raise ValueError(f"Bundle {bundle_id} cannot change from {previous} to {status}")
    row["status"] = status
    return updated


def _probability(value):
    if type(value) not in (int, float) or not 0 <= value <= 1 or not isfinite(value):
        raise ValueError(f"Expected a finite probability in [0, 1], got {value!r}")
    return float(value)


def _claim_p_value(hypothesis, supplied):
    if hypothesis == "H6B_bridge_shift_and_4hop":
        if not isinstance(supplied, Mapping) or supplied.keys() != BRIDGE_COMPONENTS:
            raise ValueError("The bridge claim requires named prompt_shift and four_hop p-values")
    if isinstance(supplied, Mapping):
        supplied = tuple(supplied.values())
    if isinstance(supplied, Sequence) and not isinstance(supplied, (str, bytes, bytearray)):
        if not supplied:
            raise ValueError(f"No p-values supplied for {hypothesis}")
        return max(_probability(component) for component in supplied)
    return _probability(supplied)


def adjudicate_holm(p_values):
    """Adjudicate the five fixed T1 tests in step-down order at family alpha 0.05.

    A mapping or sequence represents a conjunction. For 6C, any p-value for an
    alternative accuracy-or-cost rule must already account for that choice.
    Results are statistical decisions. Bundle status additionally requires the
    endpoint and engineering gates in the registry, including the static 6B guard.
    """
    if not isinstance(p_values, Mapping) or p_values.keys() != set(HOLM_FAMILY):
        raise ValueError(f"Holm requires the complete T1 family: {', '.join(HOLM_FAMILY)}")
    ordered = sorted(
        (_claim_p_value(hypothesis, p_values[hypothesis]), hypothesis)
        for hypothesis in HOLM_FAMILY
    )
    results = {}
    still_rejecting = True
    adjusted = 0.0
    for index, (p_value, hypothesis) in enumerate(ordered):
        remaining = len(ordered) - index
        threshold = HOLM_ALPHA / remaining
        adjusted = min(1.0, max(adjusted, p_value * remaining))
        still_rejecting = still_rejecting and p_value <= threshold
        results[hypothesis] = {
            "p_value": p_value,
            "adjusted_p_value": adjusted,
            "threshold": threshold,
            "reject": still_rejecting,
        }
    return results
