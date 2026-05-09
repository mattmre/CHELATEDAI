"""
Translation-Transport-Steering (TTS) pipeline for inference-time vector relocation.

Chains VectorTranslator → VectorTransport → VectorSteerer in sequence. Each stage
is independently enableable. The pipeline intercepts query/document embeddings after
the base model produces them and before they enter the vector store, allowing semantic
relocation without backpropagation.

Architecture (per Matt's tweet, 2026-05-09):
  "Steering nodes that disjoint [the backpropagation] framework and allow for vector
   relocation downstream at inference. Translation, Transport, and Steering."
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from chelation_logger import get_logger
from vector_translator import TranslationConfig, TranslationResult, VectorTranslator
from vector_transport import TransportConfig, TransportResult, VectorTransport


@dataclass
class SteeringSignal:
    direction: np.ndarray  # unit vector in embedding space
    strength: float  # magnitude of the force (0.0 to 1.0)
    source: str  # e.g. "sparse_feature_high_mean_activation"


class VectorSteerer:
    def __init__(self, max_strength: float = 0.3, enabled: bool = True) -> None:
        self._max_strength = max_strength
        self._enabled = enabled
        self._signals: List[SteeringSignal] = []

    def add_signal(self, signal: SteeringSignal) -> None:
        """Append a steering signal to the queue."""
        self._signals.append(signal)

    def clear_signals(self) -> None:
        """Remove all accumulated steering signals."""
        self._signals.clear()

    def steer(self, v: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Apply accumulated steering signals to v.

        Returns (steered_v, metadata_dict).
        metadata includes: signals_applied, total_delta_norm, was_steered.
        Clamps total steering delta norm to max_strength.
        """
        v = np.array(v, dtype=float)

        if not self._enabled or not self._signals:
            return v.copy(), {
                "signals_applied": 0,
                "total_delta_norm": 0.0,
                "was_steered": False,
            }

        total_delta = np.zeros_like(v)
        for sig in self._signals:
            d = np.array(sig.direction, dtype=float)
            dn = float(np.linalg.norm(d))
            if dn > 1e-12:
                d = d / dn
            total_delta += sig.strength * d

        delta_norm = float(np.linalg.norm(total_delta))
        if delta_norm > self._max_strength and delta_norm > 0:
            total_delta = total_delta * (self._max_strength / delta_norm)
            delta_norm = self._max_strength

        return v + total_delta, {
            "signals_applied": len(self._signals),
            "total_delta_norm": delta_norm,
            "was_steered": True,
        }

    @classmethod
    def from_sparse_feature_event(
        cls, feature_event: Any, strength_scale: float = 0.1
    ) -> "VectorSteerer":
        """Build a VectorSteerer from a SparseFeatureEvent.

        Each active feature becomes a steering signal. Direction is computed as:
        a unit vector with nonzero values at the feature's corresponding dimensions
        (using feature_id hash mod dim as a proxy until real SAE decoder is available).
        Strength is feature_value * strength_scale, clamped to max_strength.
        This is an approximation — real implementation would use the SAE decoder matrix.
        """
        max_strength = 0.3
        steerer = cls(max_strength=max_strength)
        features = getattr(feature_event, "features", [])
        dim = getattr(feature_event, "dim", 384)

        for feat in features:
            feature_id = str(getattr(feat, "feature_id", id(feat)))
            feature_value = float(getattr(feat, "value", 1.0))
            dim_idx = abs(hash(feature_id)) % dim
            direction = np.zeros(dim)
            direction[dim_idx] = 1.0
            strength = min(feature_value * strength_scale, max_strength)
            steerer.add_signal(
                SteeringSignal(
                    direction=direction,
                    strength=strength,
                    source=f"sparse_feature_{feature_id}",
                )
            )
        return steerer


@dataclass
class TTSConfig:
    translation_enabled: bool = True
    transport_enabled: bool = True
    steering_enabled: bool = True
    record_intermediates: bool = True


@dataclass
class TTSResult:
    original: np.ndarray
    after_translation: np.ndarray
    after_transport: np.ndarray
    after_steering: np.ndarray  # = final
    translation_result: Optional[TranslationResult]
    transport_result: Optional[TransportResult]
    steering_meta: Optional[Dict[str, Any]]
    total_delta_norm: float  # L2 distance from original to final
    stages_applied: List[str]  # which stages actually modified the vector


class TTSPipeline:
    def __init__(
        self,
        translator: VectorTranslator,
        transport: VectorTransport,
        steerer: VectorSteerer,
        config: TTSConfig,
    ) -> None:
        self._translator = translator
        self._transport = transport
        self._steerer = steerer
        self._config = config
        self.logger = get_logger()

    def apply(
        self,
        v: np.ndarray,
        cluster_id: Optional[str] = None,
        cluster_confidence: float = 0.0,
        feature_event: Any = None,
    ) -> TTSResult:
        """Run Translation → Transport → Steering in sequence.

        If feature_event is provided, updates VectorSteerer signals from it before steering.
        Each stage uses the output of the prior stage as input.
        """
        v = np.array(v, dtype=float)
        original = v.copy()
        stages_applied: List[str] = []

        # ── Stage 1: Translation ─────────────────────────────────────────────
        current = v.copy()
        translation_result: Optional[TranslationResult] = None
        if self._config.translation_enabled:
            tr = self._translator.translate(
                current,
                cluster_id=cluster_id,
                cluster_confidence=cluster_confidence,
            )
            translation_result = tr
            if tr.mode != "passthrough":
                stages_applied.append("translation")
            current = tr.translated
        after_translation = current.copy()

        # ── Stage 2: Transport ───────────────────────────────────────────────
        transport_result: Optional[TransportResult] = None
        if self._config.transport_enabled:
            tr2 = self._transport.transport(current)
            transport_result = tr2
            if tr2.was_transported:
                stages_applied.append("transport")
            current = tr2.transported
        after_transport = current.copy()

        # ── Stage 3: Steering ────────────────────────────────────────────────
        steering_meta: Optional[Dict[str, Any]] = None
        if self._config.steering_enabled:
            if feature_event is not None:
                # feature_event signals are transient per-inference: clear accumulated
                # state from prior calls before loading this inference's signals.
                # Signals added via steerer.add_signal() (external/persistent) only
                # persist when feature_event=None.
                self._steerer.clear_signals()
                for sig in VectorSteerer.from_sparse_feature_event(feature_event)._signals:
                    self._steerer.add_signal(sig)
            steered, meta = self._steerer.steer(current)
            steering_meta = meta
            if meta.get("was_steered", False):
                stages_applied.append("steering")
            current = steered
        after_steering = current.copy()

        total_delta_norm = float(np.linalg.norm(after_steering - original))

        return TTSResult(
            original=original,
            after_translation=after_translation,
            after_transport=after_transport,
            after_steering=after_steering,
            translation_result=translation_result,
            transport_result=transport_result,
            steering_meta=steering_meta,
            total_delta_norm=total_delta_norm,
            stages_applied=stages_applied,
        )

    @classmethod
    def build_default(
        cls, dim: int, phase_c_results_path: Optional[str] = None
    ) -> "TTSPipeline":
        """Build a default-configured TTSPipeline.

        - Translator: loads from phase_c_results_path if provided, else passthrough
        - Transport: no targets, passthrough
        - Steerer: max_strength=0.3, no initial signals
        - Config: all stages enabled
        """
        t_config = TranslationConfig(offset_dim=dim)
        if phase_c_results_path is not None:
            translator = VectorTranslator.from_phase_c_results(phase_c_results_path, t_config)
        else:
            translator = VectorTranslator(t_config)
        transport = VectorTransport(TransportConfig())
        steerer = VectorSteerer(max_strength=0.3)
        return cls(translator, transport, steerer, TTSConfig())

    def get_diagnostics(self) -> Dict[str, Any]:
        """Return pipeline config summary for dashboard/reporting."""
        return {
            "translation_enabled": self._config.translation_enabled,
            "transport_enabled": self._config.transport_enabled,
            "steering_enabled": self._config.steering_enabled,
            "record_intermediates": self._config.record_intermediates,
            "translator_has_offset": self._translator._learned_offset is not None,
            "translator_cluster_count": len(self._translator._cluster_offsets),
            "transport_target_count": len(self._transport._targets),
            "steerer_signal_count": len(self._steerer._signals),
            "steerer_max_strength": self._steerer._max_strength,
        }
