"""Phase 6: Bridge connecting AntigravityEngine to Model-Scope runtime, features, and memory."""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Optional
from uuid import uuid4

from model_scope_artifacts import ArtifactStore, summarize_model_scope_artifact
from model_scope_features import FeatureExtractor
from model_scope_memory import MemoryManager
from model_scope_steering import SteeringActuator
from steering_policy import PolicyRegistry

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
        self._extractor = FeatureExtractor(adapter=None)
        registry = PolicyRegistry()
        self._actuator = SteeringActuator(registry, max_total_interventions=self._config.max_total_interventions)
        self._observation_count = 0
        self._error_count = 0

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
            return ModelScopeObservationResult(
                run_id=run_id,
                query=query,
                error="runtime_not_loaded",
                elapsed_ms=(time.monotonic() - t_start) * 1000.0,
            )

        activation_events = runtime.get_events()
        if not activation_events:
            self._observation_count += 1
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

        if self._config.enable_feature_extraction:
            feature_event = self._extractor.extract(activation_event)
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
                from model_scope_steering import intervention_record_to_dict
                with open(intervention_path, "w", encoding="utf-8") as fh:
                    json.dump([intervention_record_to_dict(r) for r in records], fh)

        self._observation_count += 1
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
            except Exception:
                last_artifact_summary = None
        return {
            "observation_count": self._observation_count,
            "error_count": self._error_count,
            "last_artifact": last_artifact_summary,
            "bridge_config": dataclasses.asdict(self._config),
            "intervention_summary": {
                "total_applied": self._actuator.total_applied(),
                "total_shadow": self._actuator.total_shadow(),
            },
        }
