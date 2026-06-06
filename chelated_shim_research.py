"""Research-only shim insertion helpers (default OFF).

Env: CHELATED_SHIM_RESEARCH=1 enables guarded preflight metadata at production seams.
Does not import docs/steering_chelation_rag_dag_research artifacts until BHS promotion.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import numpy as np


def research_enabled() -> bool:
    return os.environ.get("CHELATED_SHIM_RESEARCH") == "1"


def promoted_enabled() -> bool:
    """BHS-promoted ShimRegistry at repo root (see scripts/promote_shim_primitives.py)."""
    return os.environ.get("CHELATED_SHIM_PROMOTED") == "1"


_PROMOTED_START_ID = "chelated-sip-probe-v1"
_PROMOTED_MAX_DELTA = 0.05


def promoted_registry_probe(dim: int = 8) -> Optional[Dict[str, Any]]:
    """Register a deterministic probe shim when promotion env is on (metadata only)."""
    if not (research_enabled() and promoted_enabled()):
        return None
    from shim_node_promoted import ShimRegistry  # noqa: WPS433 — promoted copy

    registry = ShimRegistry(dim=dim)
    registry.register_seeded("chelated-sip-probe-v1", tier=0)
    got = registry.get("chelated-sip-probe-v1")
    return {
        "promoted_shim_registry": True,
        "probe_shim_registered": got is not None,
        "registry_count": registry.count(),
    }


def promoted_sip_apply(
    v: np.ndarray,
    *,
    max_delta: float = _PROMOTED_MAX_DELTA,
) -> Tuple[np.ndarray, Optional[Dict[str, Any]]]:
    """Bounded insert-once SIP: apply promoted cascade composite to v (env-guarded).

    Uses ShimRegistry.apply_shim_cascade; does not mutate registry storage.
    Returns (v_out, metadata) or (v.copy(), None) when guards off or no composite.
    """
    if not (research_enabled() and promoted_enabled()):
        return np.array(v, dtype=float).copy(), None

    dim = int(v.shape[0]) if v.size else 8
    from shim_node_promoted import ShimRegistry  # noqa: WPS433

    registry = ShimRegistry(dim=dim)
    if registry.get(_PROMOTED_START_ID) is None:
        registry.register_seeded(_PROMOTED_START_ID, tier=0)

    application = registry.apply_shim_cascade(
        _PROMOTED_START_ID,
        max_depth=1,
        max_fanout=2,
        include_composite=True,
    )
    composite = application.composite_vector
    if composite is None or composite.size != dim:
        return np.array(v, dtype=float).copy(), {
            "promoted_sip_applied": False,
            "reason": "no_composite",
            "cascade_ids": list(application.cascade_ids),
        }

    delta = np.array(composite, dtype=float)
    dn = float(np.linalg.norm(delta))
    if dn > 1e-12 and dn > max_delta:
        delta = delta * (max_delta / dn)
        dn = max_delta

    v_out = np.array(v, dtype=float) + delta
    return v_out, {
        "promoted_sip_applied": True,
        "promoted_sip_insert_once": True,
        "cascade_ids": list(application.cascade_ids),
        "composite_delta_norm": dn,
        "registry_count": registry.count(),
        "sip_seam": "VectorSteerer.steer.promoted_apply",
    }


def research_preflight_metadata(
    *,
    seam: str,
    stall_count: int,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    meta: Dict[str, Any] = {
        "research_shim_guard": True,
        "research_stall_count": stall_count,
        "sip_seam": seam,
    }
    if extra:
        meta.update(extra)
    return meta


def bump_stall_counter(counter: int, *, has_work: bool) -> int:
    if has_work:
        return 0
    return counter + 1


def attach_research_meta(diagnostics: dict, meta: Optional[Dict[str, Any]]) -> dict:
    """Include last research SIP preflight in runtime diagnostics when present."""
    if meta:
        diagnostics["research_shim"] = meta
    return diagnostics


def collect_research_probe_from_tts_metadata(
    steering_meta: Optional[Dict[str, Any]],
    seam: str = "tts_pipeline.VectorSteerer.steer",
    cycle_tag: str = "research-probe-VectorSteerer-first-sip-C",
) -> Dict[str, Any]:
    """Harvest guarded SIP metadata from TTS steering_meta (research paths only)."""
    if steering_meta is None or not isinstance(steering_meta, dict):
        return {
            "probe_hit": False,
            "reason": "no steering_meta (steering disabled, no signals, or non-TTS path)",
            "seam": seam,
            "cycle_tag": cycle_tag,
            "research_guard": "CHELATED_SHIM_RESEARCH=1 required for keys to appear",
        }
    activated = bool(
        steering_meta.get("research_shim_probe_activated")
        or steering_meta.get("research_shim_guard")
    )
    return {
        "probe_hit": activated,
        "seam": steering_meta.get("sip_seam", seam),
        "probe_count": steering_meta.get(
            "research_shim_probe_count",
            steering_meta.get("research_stall_count", 0),
        ),
        "activation_record": steering_meta.get("research_activation_record", {}),
        "base_signals_applied": steering_meta.get("signals_applied"),
        "base_total_delta_norm": steering_meta.get("total_delta_norm"),
        "base_was_steered": steering_meta.get("was_steered"),
        "cycle_tag": cycle_tag,
        "all_meta_keys_present": list(steering_meta.keys()),
        "research_guard": "CHELATED_SHIM_RESEARCH or --research-shim",
    }