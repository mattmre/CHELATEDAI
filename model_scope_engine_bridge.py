"""Phase 6: Bridge connecting AntigravityEngine to Model-Scope runtime, features, and memory."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
import numpy as np
from pathlib import Path
from typing import TYPE_CHECKING, Optional
from uuid import uuid4

from model_scope_artifacts import ArtifactStore, summarize_model_scope_artifact
from model_scope_features import FeatureExtractor
from model_scope_memory import MemoryManager
from model_scope_steering import SteeringActuator
from steering_policy import PolicyRegistry, PolicyStatus, SteeringMode, SteeringPolicyConfig

try:
    from chelated_shim_research import (
        promoted_sip_apply,
        bump_stall_counter,
        research_enabled,
        research_preflight_metadata,
    )
except ModuleNotFoundError:
    def promoted_sip_apply(v):
        return np.array(v, dtype=float).copy(), None

    def research_enabled() -> bool:
        return False

    def bump_stall_counter(counter: int, *, has_work: bool) -> int:
        return counter

    def research_preflight_metadata(
        *,
        seam: str,
        stall_count: int,
        extra: dict | None = None,
    ) -> dict:
        return {
            "research_shim_guard": False,
            "research_stall_count": stall_count,
            "sip_seam": seam,
            **(dict(extra or {})),
        }

if TYPE_CHECKING:
    from model_scope_runtime import LocalModelRuntime


@dataclass
class ModelScopeBridgeConfig:
    """Configuration for the Model-Scope engine bridge."""

    artifact_dir: str = "experiment_runs/model_scope"
    enable_feature_extraction: bool = True
    enable_steering: bool = False
    max_events_in_memory: int = 100
    observation_tag: str = "engine_bridge"
    max_total_interventions: int = 100


@dataclass
class ModelScopeObservationResult:
    """Result of one bridge observation pass."""

    run_id: str
    query: str
    activation_event: Optional[object] = None
    feature_event: Optional[object] = None
    intervention_count: int = 0
    artifact_path: Optional[str] = None
    elapsed_ms: float = 0.0
    error: Optional[str] = None


class ModelScopeEngineBridge:
    """Connects AntigravityEngine to the Model-Scope runtime, features, and memory."""

    def __init__(self, config: Optional[ModelScopeBridgeConfig] = None) -> None:
        self._config = config or ModelScopeBridgeConfig()
        artifact_path = Path(self._config.artifact_dir)
        artifact_path.mkdir(parents=True, exist_ok=True)
        self._artifact_store = ArtifactStore(base_dir=artifact_path)
        self._memory_manager = MemoryManager(
            base_dir=Path(self._config.artifact_dir) / "memory",
            working_max_entries=self._config.max_events_in_memory,
        )
        self._policy_registry = PolicyRegistry()
        self._default_policy_id: str | None = None
        self._extractor = FeatureExtractor(adapter=None)
        self._actuator = SteeringActuator(self._policy_registry, max_total_interventions=self._config.max_total_interventions)
        self._observation_count = 0
        self._error_count = 0
        self._research_observe_stall_count: int = 0
        self._last_research_shim_meta: dict | None = None

    @staticmethod
    def _apply_shim_to_sparse_features(feature_event: object) -> tuple[object | None, dict | None]:
        """Apply a promoted shim perturbation to a feature-event feature vector."""
        if feature_event is None:
            return None, None
        try:
            features = dict(getattr(feature_event, "features", {}))
        except Exception:
            return feature_event, None
        if not isinstance(features, dict):
            return feature_event, None
        keys = list(features.keys())
        vector = np.array([float(features[k]) for k in keys], dtype=float)
        out, promoted_meta = promoted_sip_apply(vector)
        if promoted_meta is None:
            return feature_event, None
        if len(out):
            try:
                updated = dict(features)
                for i, key in enumerate(keys):
                    if i < len(out):
                        updated[key] = float(out[i])
                feature_event.features = updated
                feature_event.nonzero_count = sum(1 for value in updated.values() if float(value) != 0.0)
            except Exception:
                return feature_event, promoted_meta
        return feature_event, promoted_meta

    def observe(
        self,
        query: str,
        runtime: "LocalModelRuntime",
    ) -> ModelScopeObservationResult:
        """Capture an activation event from the runtime and extract sparse features."""
        import time

        run_id = f"obs_{uuid4().hex[:8]}"
        t_start = time.monotonic()

        if not runtime.is_loaded():
            self._error_count += 1
            if research_enabled():
                self._research_observe_stall_count = bump_stall_counter(
                    self._research_observe_stall_count,
                    has_work=False,
                )
                _, promoted_meta = promoted_sip_apply(np.zeros(8, dtype=float))
                self._last_research_shim_meta = research_preflight_metadata(
                    seam="ModelScopeEngineBridge.observe",
                    stall_count=self._research_observe_stall_count,
                    extra={
                        "runtime_loaded": False,
                        "error": "runtime_not_loaded",
                        **({"promoted_sip_apply": promoted_meta} if promoted_meta else {}),
                    },
                )
            else:
                self._last_research_shim_meta = None
            return ModelScopeObservationResult(
                run_id=run_id,
                query=query,
                error="runtime_not_loaded",
                elapsed_ms=(time.monotonic() - t_start) * 1000.0,
            )

        activation_events = runtime.get_events()
        if not activation_events:
            self._observation_count += 1
            if research_enabled():
                self._research_observe_stall_count = bump_stall_counter(
                    self._research_observe_stall_count,
                    has_work=False,
                )
                _, promoted_meta = promoted_sip_apply(np.zeros(8, dtype=float))
                self._last_research_shim_meta = research_preflight_metadata(
                    seam="ModelScopeEngineBridge.observe",
                    stall_count=self._research_observe_stall_count,
                    extra={
                        "activation_events": 0,
                        "runtime_id": run_id,
                        **({"promoted_sip_apply": promoted_meta} if promoted_meta else {}),
                    },
                )
            else:
                self._last_research_shim_meta = None
            return ModelScopeObservationResult(
                run_id=run_id,
                query=query,
                activation_event=None,
                elapsed_ms=(time.monotonic() - t_start) * 1000.0,
            )

        activation_event = activation_events[-1]
        feature_event: Optional[object] = None
        artifact_path_str: Optional[str] = None
        intervention_count = 0
        feature_shim_meta: dict | None = None

        if self._config.enable_feature_extraction:
            feature_event = self._extractor.extract(activation_event)
            if research_enabled():
                feature_event, feature_shim_meta = self._apply_shim_to_sparse_features(feature_event)
            saved_path = self._artifact_store.save_feature_event(feature_event)
            artifact_path_str = str(saved_path)
            self._memory_manager.episodic.store(
                key=run_id,
                value={"run_id": run_id, "query": query, "artifact_path": artifact_path_str},
                tags=[self._config.observation_tag],
            )

        if self._config.enable_steering and feature_event is not None:
            _, records = self._actuator.apply_all_active(feature_event)
            intervention_count = sum(1 for r in records if r.applied)
            if records:
                intervention_path = self._artifact_store._base_dir / f"intervention_{run_id}.json"
                from model_scope_steering import intervention_record_to_dict as _irtd
                import warnings as _warnings
                try:
                    with open(intervention_path, "w", encoding="utf-8") as fh:
                        json.dump([_irtd(r) for r in records], fh)
                except OSError as _write_err:
                    _warnings.warn(
                        f"Failed to persist intervention record {run_id}: {_write_err!r}",
                        stacklevel=2,
                    )
                else:
                    # Enforce cap to prevent unbounded accumulation.
                    # Files are removed until count reaches the cap; sort order is
                    # lexicographic, not temporal (UUID-based names have no time
                    # correlation, so "oldest" is meaningless here).
                    existing = sorted(self._artifact_store._base_dir.glob("intervention_*.json"))
                    if len(existing) > 50:
                        for old_file in existing[:-50]:
                            try:
                                old_file.unlink()
                            except OSError as _del_err:
                                import warnings as _warnings
                                _warnings.warn(
                                    f"Failed to remove old intervention file during cap cleanup: {_del_err!r}",
                                    UserWarning,
                                    stacklevel=2,
                                )

        self._observation_count += 1
        if research_enabled():
            self._research_observe_stall_count = bump_stall_counter(
                self._research_observe_stall_count,
                has_work=True,
            )
            _, promoted_meta = promoted_sip_apply(np.zeros(8, dtype=float))
            self._last_research_shim_meta = research_preflight_metadata(
                seam="ModelScopeEngineBridge.observe",
                stall_count=self._research_observe_stall_count,
                extra={
                    "activation_events": len(activation_events),
                    "intervention_count": intervention_count,
                    "feature_event_present": bool(feature_event),
                    "artifact_path": artifact_path_str,
                    **({"promoted_sip_apply": promoted_meta} if promoted_meta else {}),
                    **({"feature_shim_applied": feature_shim_meta} if feature_shim_meta else {}),
                },
            )
        else:
            self._last_research_shim_meta = None
        elapsed_ms = (time.monotonic() - t_start) * 1000.0
        return ModelScopeObservationResult(
            run_id=run_id,
            query=query,
            activation_event=activation_event,
            feature_event=feature_event,
            intervention_count=intervention_count,
            artifact_path=artifact_path_str,
            elapsed_ms=elapsed_ms,
        )

    def register_shadow_policy(
        self,
        *,
        name: str = "model_scope_shadow_policy",
        feature_space: str = "raw_stats",
        target_features: list[str] | tuple[str, ...] | None = None,
        max_interventions: int = 0,
        status: PolicyStatus = PolicyStatus.ACTIVE,
        policy_id: str | None = None,
    ) -> str:
        """Register the default advisory shadow policy and return policy_id.

        The bridge uses this policy for observation-mode steering by default. Policy
        entries are immutable after registration except status transitions handled by
        the registry.
        """
        if target_features is None:
            target_features = ["mean_activation", "norm_activation", "token_count", "shape_0"]
        config = SteeringPolicyConfig(
            name=name,
            mode=SteeringMode.SHADOW,
            target_features=list(target_features),
            max_interventions=max_interventions,
            status=status,
            description=f"feature_space={feature_space}",
        )
        if policy_id is not None:
            config.policy_id = policy_id
        self._actuator._registry.register(config)
        self._default_policy_id = config.policy_id
        return config.policy_id


    def get_last_shadow_policy_id(self) -> str | None:
        """Return the most recently registered model-scope policy id, if any."""
        return self._default_policy_id


    def get_active_policy_ids(self) -> list[str]:
        """Return ACTIVE policy ids currently registered in the bridge."""
        return [p.policy_id for p in self._actuator._registry.list_active()]


    def get_last_research_shim_meta(self) -> dict | None:
        """Return the latest research shim metadata emitted by observe()."""
        return self._last_research_shim_meta

    def get_telemetry(self) -> dict:
        """Return bridge telemetry snapshot."""
        return {
            "observation_count": self._observation_count,
            "error_count": self._error_count,
            "artifact_dir": self._config.artifact_dir,
            "memory_snapshot": self._memory_manager.snapshot(),
        }

    def list_recent_observations(self, limit: int = 20) -> list:
        """Return up to `limit` recent feature event summaries from the artifact store."""
        paths = self._artifact_store.list_artifacts(pattern="feature_event_*.json")[-limit:]
        items = []
        for p in reversed(paths):
            try:
                with open(p, encoding="utf-8") as fh:
                    loaded = json.load(fh)
                items.append({"path": str(p), "artifact": summarize_model_scope_artifact(loaded)})
            except Exception as exc:
                items.append({"path": str(p), "error": str(exc)})
        return items

    def get_summary_for_diagnostics(self) -> dict:
        """Return a dict suitable for passing as `model_scope` in diagnostics reports."""
        paths = self._artifact_store.list_artifacts(pattern="feature_event_*.json")
        last_artifact_summary = None
        if paths:
            try:
                with open(paths[-1], encoding="utf-8") as fh:
                    last_data = json.load(fh)
                last_artifact_summary = summarize_model_scope_artifact(last_data)
            except Exception as _diag_err:
                import warnings as _warnings
                _warnings.warn(
                    f"get_summary_for_diagnostics: failed to read last artifact: {_diag_err!r}",
                    UserWarning,
                    stacklevel=2,
                )
                last_artifact_summary = None
        return {
            "observation_count": self._observation_count,
            "error_count": self._error_count,
            "last_artifact": last_artifact_summary,
            "default_policy_id": self._default_policy_id,
            "active_policy_ids": self.get_active_policy_ids(),
            "bridge_config": dataclasses.asdict(self._config),
            "intervention_summary": {
                "total_applied": self._actuator.total_applied(),
                "total_shadow": self._actuator.total_shadow(),
            },
        }
