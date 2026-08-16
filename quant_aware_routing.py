"""Leakage-safe, quant-aware adapter routing promotion plane (Rung 16).

The plane deliberately keeps fitting, selection, and reporting as separate
state transitions:

* ANCHOR fits the per-route and single-global document adapters and mines their
  serving centroids.
* SELECT is the only promotion stage.  It freezes the best-route ablation,
  paired query bootstrap CI, no-route floor check, and honest quantization
  gates for every adapter the frozen router could use on REPORT or in service.
* REPORT is a one-shot evaluation of the frozen candidate.  It can downgrade a
  candidate to DEGENERATE when the multi-route binding is not exercised, but it
  can never promote or tune the candidate.

Adapters are trained and served in one coherent geometry: cached document
vectors are transformed into the query coordinate system and retained as one
precomputed document view per route.  Routing never swaps a document-trained
adapter onto a query vector.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import asdict, dataclass
from threading import Lock
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from adapter_router import AdapterRouter
from benchmark_utils import canonicalize_id
from quantization_promotion_gate import QuantizationPromotionGate


def _positive_builtin_int(value: Any, name: str) -> int:
    """Require an exact built-in integer, excluding bool and coercible values."""

    if type(value) is not int:
        raise TypeError(f"{name} must be a built-in int")
    if value < 1:
        raise ValueError(f"{name} must be >= 1")
    return value


@dataclass(frozen=True)
class ThreeWaySplit:
    """Frozen query IDs for the ANCHOR/SELECT/REPORT protocol."""

    anchor_ids: Tuple[str, ...]
    select_ids: Tuple[str, ...]
    report_ids: Tuple[str, ...]
    seed: int

    def validate(self) -> None:
        anchor = set(self.anchor_ids)
        select = set(self.select_ids)
        report = set(self.report_ids)
        if anchor & select or anchor & report or select & report:
            raise ValueError("ANCHOR, SELECT, and REPORT query IDs must be pairwise disjoint")
        if not anchor or not select or not report:
            raise ValueError("ANCHOR, SELECT, and REPORT must each contain at least one query")

    def to_dict(self, include_ids: bool = True) -> Dict[str, Any]:
        result = {
            "seed": int(self.seed),
            "anchor_count": len(self.anchor_ids),
            "select_count": len(self.select_ids),
            "report_count": len(self.report_ids),
            "anchor_sha256": _ids_sha256(self.anchor_ids),
            "select_sha256": _ids_sha256(self.select_ids),
            "report_sha256": _ids_sha256(self.report_ids),
        }
        if include_ids:
            result.update(
                {
                    "anchor_ids": list(self.anchor_ids),
                    "select_ids": list(self.select_ids),
                    "report_ids": list(self.report_ids),
                }
            )
        return result


@dataclass(frozen=True)
class RoutingPlaneConfig:
    """Promotion-critical values loaded from the frozen preregistration."""

    k: int = 10
    split_seed: int = 1616
    anchor_fraction: float = 0.4
    select_fraction: float = 0.3
    report_fraction: float = 0.3
    route_k: int = 3
    min_cluster_documents: int = 5
    margin_delta: float = 0.02
    min_report_routes: int = 2
    min_report_route_fraction: float = 0.10
    confidence_level: float = 0.95
    bootstrap_resamples: int = 5000
    bootstrap_seed: int = 1617
    min_lift: float = 0.005
    max_floor_loss: float = 0.005
    quant_retained_gain: float = 0.8
    quant_minimum_fp32_gain: float = 0.01
    quant_levels: int = 127
    quantile: float = 0.99
    adapter_steps: int = 2000
    adapter_learning_rate: float = 0.1
    adapter_batch_size: int = 64
    adapter_min_correction: float = 0.01
    adapter_max_correction: float = 0.5

    def validate(self) -> None:
        floating_values = {
            "anchor_fraction": self.anchor_fraction,
            "select_fraction": self.select_fraction,
            "report_fraction": self.report_fraction,
            "margin_delta": self.margin_delta,
            "min_report_route_fraction": self.min_report_route_fraction,
            "confidence_level": self.confidence_level,
            "min_lift": self.min_lift,
            "max_floor_loss": self.max_floor_loss,
            "quant_retained_gain": self.quant_retained_gain,
            "quant_minimum_fp32_gain": self.quant_minimum_fp32_gain,
            "quantile": self.quantile,
            "adapter_learning_rate": self.adapter_learning_rate,
            "adapter_min_correction": self.adapter_min_correction,
            "adapter_max_correction": self.adapter_max_correction,
        }
        nonfinite = sorted(
            name
            for name, value in floating_values.items()
            if not math.isfinite(float(value))
        )
        if nonfinite:
            raise ValueError(
                "routing promotion floating parameters must be finite: "
                + ", ".join(nonfinite)
            )
        fractions = (self.anchor_fraction, self.select_fraction, self.report_fraction)
        if any(value <= 0.0 for value in fractions) or not math.isclose(sum(fractions), 1.0, abs_tol=1e-9):
            raise ValueError("three-way split fractions must be positive and sum to 1")
        _positive_builtin_int(self.k, "k")
        _positive_builtin_int(self.route_k, "route_k")
        if self.min_cluster_documents < 1:
            raise ValueError("min_cluster_documents must be >= 1")
        if self.margin_delta < 0.0:
            raise ValueError("margin_delta must be non-negative")
        if self.min_report_routes < 2:
            raise ValueError("Rung 16 requires at least two specialist routes")
        if not 0.0 < self.min_report_route_fraction <= 1.0:
            raise ValueError("min_report_route_fraction must be in (0, 1]")
        if not 0.0 < self.confidence_level < 1.0:
            raise ValueError("confidence_level must be in (0, 1)")
        if self.bootstrap_resamples < 1:
            raise ValueError("bootstrap_resamples must be >= 1")
        if self.min_lift < 0.0 or self.max_floor_loss < 0.0:
            raise ValueError("lift and floor-loss thresholds must be non-negative")
        if self.quant_retained_gain < 0.0:
            raise ValueError("quant_retained_gain must be non-negative")
        if self.quant_minimum_fp32_gain <= 0.0:
            raise ValueError("Rung 16 quant_minimum_fp32_gain must be strictly positive")
        if self.quant_levels < 2 or not 0.0 < self.quantile <= 1.0:
            raise ValueError("invalid simulated INT8 configuration")
        if self.adapter_steps < 1 or self.adapter_learning_rate <= 0.0 or self.adapter_batch_size < 1:
            raise ValueError("adapter training budget must be positive")
        if (
            self.adapter_min_correction < 0.0
            or self.adapter_max_correction <= 0.0
            or self.adapter_min_correction > self.adapter_max_correction
        ):
            raise ValueError("adapter correction bounds must satisfy 0 <= min <= max")

    @classmethod
    def from_preregistration(cls, preregistration: Mapping[str, Any]) -> "RoutingPlaneConfig":
        split = preregistration["split"]
        routing = preregistration["routing"]
        promotion = preregistration["promotion"]
        quant = preregistration["quantization"]
        adapter = preregistration["adapter"]
        config = cls(
            split_seed=int(split["seed"]),
            anchor_fraction=float(split["anchor_fraction"]),
            select_fraction=float(split["select_fraction"]),
            report_fraction=float(split["report_fraction"]),
            route_k=routing["k"],
            min_cluster_documents=int(routing["minimum_cluster_documents"]),
            margin_delta=float(routing["margin_delta"]),
            min_report_routes=int(routing["minimum_report_routes"]),
            min_report_route_fraction=float(routing["minimum_report_fraction_per_route"]),
            confidence_level=float(promotion["confidence_level"]),
            bootstrap_resamples=int(promotion["bootstrap_resamples"]),
            bootstrap_seed=int(promotion["bootstrap_seed"]),
            min_lift=float(promotion["minimum_ci_lower_bound_lift"]),
            max_floor_loss=float(promotion["maximum_material_loss_vs_no_route"]),
            quant_retained_gain=float(quant["retained_gain_threshold"]),
            quant_minimum_fp32_gain=float(quant["minimum_fp32_gain_over_no_route"]),
            quant_levels=int(quant["levels"]),
            quantile=float(quant["quantile"]),
            adapter_steps=int(adapter["training_steps"]),
            adapter_learning_rate=float(adapter["learning_rate"]),
            adapter_batch_size=int(adapter["batch_size"]),
            adapter_min_correction=float(adapter["min_correction"]),
            adapter_max_correction=float(adapter["max_correction"]),
        )
        config.validate()
        return config


@dataclass(frozen=True)
class BootstrapCI:
    """Paired query-level bootstrap result."""

    estimate: float
    lower: float
    upper: float
    confidence_level: float
    resamples: int
    seed: int
    query_count: int

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PlaneEvaluation:
    """One policy's aggregate and per-query graded NDCG."""

    policy: str
    mean_ndcg: float
    rows: List[Dict[str, Any]]

    @property
    def scores_by_query(self) -> Dict[str, float]:
        return {str(row["query_id"]): float(row["ndcg"]) for row in self.rows}

    def to_dict(self, include_rows: bool = True) -> Dict[str, Any]:
        result = {
            "policy": self.policy,
            "mean_ndcg": float(self.mean_ndcg),
            "evaluated_queries": len(self.rows),
        }
        if include_rows:
            result["rows"] = list(self.rows)
        return result


def three_way_seeded_split(
    query_ids: Iterable[Any],
    *,
    seed: int,
    anchor_fraction: float = 0.4,
    select_fraction: float = 0.3,
    report_fraction: float = 0.3,
    strata: Optional[Mapping[Any, str]] = None,
) -> ThreeWaySplit:
    """Return an exhaustive deterministic split, optionally stratified."""

    fractions = (float(anchor_fraction), float(select_fraction), float(report_fraction))
    if any(value <= 0.0 for value in fractions) or not math.isclose(sum(fractions), 1.0, abs_tol=1e-9):
        raise ValueError("split fractions must be positive and sum to 1")
    canonical_ids = sorted({canonicalize_id(query_id) for query_id in query_ids})
    if len(canonical_ids) < 3:
        raise ValueError("three-way split requires at least three unique query IDs")
    groups: Dict[str, List[str]] = {}
    if strata is None:
        groups["__all__"] = canonical_ids
    else:
        canonical_strata = {canonicalize_id(key): str(value) for key, value in strata.items()}
        missing = [query_id for query_id in canonical_ids if query_id not in canonical_strata]
        if missing:
            raise ValueError(f"missing strata for query IDs: {missing[:5]}")
        for query_id in canonical_ids:
            groups.setdefault(canonical_strata[query_id], []).append(query_id)

    anchor: List[str] = []
    select: List[str] = []
    report: List[str] = []
    for stratum, members in sorted(groups.items()):
        if len(members) < 3:
            raise ValueError(f"stratum {stratum!r} needs at least three query IDs")
        stable_seed = int(seed) + int(hashlib.sha256(stratum.encode("utf-8")).hexdigest()[:8], 16)
        shuffled = list(sorted(members))
        random.Random(stable_seed).shuffle(shuffled)
        anchor_n = int(round(len(shuffled) * fractions[0]))
        select_n = int(round(len(shuffled) * fractions[1]))
        anchor_n = max(1, min(anchor_n, len(shuffled) - 2))
        select_n = max(1, min(select_n, len(shuffled) - anchor_n - 1))
        anchor.extend(shuffled[:anchor_n])
        select.extend(shuffled[anchor_n : anchor_n + select_n])
        report.extend(shuffled[anchor_n + select_n :])

    split = ThreeWaySplit(tuple(sorted(anchor)), tuple(sorted(select)), tuple(sorted(report)), int(seed))
    split.validate()
    if set(split.anchor_ids) | set(split.select_ids) | set(split.report_ids) != set(canonical_ids):
        raise ValueError("three-way split is not exhaustive")
    return split


def paired_query_bootstrap_ci(
    candidate_scores: Mapping[Any, float],
    comparator_scores: Mapping[Any, float],
    *,
    confidence_level: float,
    resamples: int,
    seed: int,
) -> BootstrapCI:
    """Bootstrap paired per-query deltas; independent resampling is forbidden."""

    candidate = {canonicalize_id(key): float(value) for key, value in candidate_scores.items()}
    comparator = {canonicalize_id(key): float(value) for key, value in comparator_scores.items()}
    if set(candidate) != set(comparator) or not candidate:
        raise ValueError("paired bootstrap requires identical non-empty query ID sets")
    ordered_ids = sorted(candidate)
    deltas = np.asarray([candidate[query_id] - comparator[query_id] for query_id in ordered_ids], dtype=float)
    if not np.all(np.isfinite(deltas)):
        raise ValueError("paired bootstrap scores must be finite")
    if not 0.0 < confidence_level < 1.0 or resamples < 1:
        raise ValueError("invalid bootstrap configuration")
    rng = np.random.default_rng(int(seed))
    indices = rng.integers(0, len(deltas), size=(int(resamples), len(deltas)))
    sampled_means = np.mean(deltas[indices], axis=1)
    alpha = 1.0 - float(confidence_level)
    lower, upper = np.quantile(sampled_means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return BootstrapCI(
        estimate=float(np.mean(deltas)),
        lower=float(lower),
        upper=float(upper),
        confidence_level=float(confidence_level),
        resamples=int(resamples),
        seed=int(seed),
        query_count=len(deltas),
    )


def _graded_gain(score: float) -> float:
    try:
        gain = math.pow(2.0, score) - 1.0
    except OverflowError as exc:
        raise ValueError("derived relevance gains must be finite") from exc
    if not math.isfinite(gain):
        raise ValueError("derived relevance gains must be finite")
    return gain


def _positive_relevance(relevance: Mapping[Any, float]) -> Dict[str, float]:
    qrels: Dict[str, float] = {}
    for doc_id, raw_score in relevance.items():
        score = float(raw_score)
        if not math.isfinite(score):
            raise ValueError("relevance scores must be finite")
        if score > 0.0:
            _graded_gain(score)
            qrels[canonicalize_id(doc_id)] = score
    return qrels


def graded_ndcg_at_k(ranked_ids: Sequence[Any], relevance: Mapping[Any, float], k: int = 10) -> float:
    """Correct graded NDCG whose IDCG uses the full query qrels."""

    validated_k = _positive_builtin_int(k, "k")
    qrels = _positive_relevance(relevance)
    if not qrels:
        return 0.0

    dcg = 0.0
    for rank, doc_id in enumerate(list(ranked_ids)[:validated_k], start=1):
        term = _graded_gain(qrels.get(canonicalize_id(doc_id), 0.0)) / math.log2(rank + 1.0)
        dcg += term
        if not math.isfinite(term) or not math.isfinite(dcg):
            raise ValueError("derived relevance DCG must be finite")
    ideal = sorted(qrels.values(), reverse=True)[:validated_k]
    idcg = 0.0
    for rank, score in enumerate(ideal, start=1):
        term = _graded_gain(score) / math.log2(rank + 1.0)
        idcg += term
        if not math.isfinite(term) or not math.isfinite(idcg):
            raise ValueError("derived relevance IDCG must be finite")
    if idcg <= 0.0:
        return 0.0
    ndcg = dcg / idcg
    if not math.isfinite(ndcg):
        raise ValueError("derived NDCG must be finite")
    return float(ndcg)


class QuantAwareRoutingPlane:
    """Stateful ANCHOR-fit / SELECT-promote / REPORT-once routing plane."""

    def __init__(
        self,
        *,
        router: AdapterRouter,
        adapters: Mapping[str, Any],
        document_ids: Sequence[Any],
        document_embeddings: np.ndarray,
        qrels: Mapping[Any, Mapping[Any, float]],
        split: ThreeWaySplit,
        config: RoutingPlaneConfig,
        preregistration: Mapping[str, Any],
        anchor_query_ids_used: Iterable[Any],
        anchor_document_ids_used: Iterable[Any],
        oracle_document_embeddings: Optional[np.ndarray] = None,
    ):
        config.validate()
        split.validate()
        self.router = router
        self.adapters = dict(adapters)
        if "global" not in self.adapters:
            raise ValueError("plane requires a fitted single-global adapter")
        self.document_ids = tuple(canonicalize_id(doc_id) for doc_id in document_ids)
        self.document_embeddings = _as_matrix(document_embeddings, "document_embeddings")
        if len(self.document_ids) != self.document_embeddings.shape[0]:
            raise ValueError("document_ids and document_embeddings row count differ")
        self.oracle_document_embeddings = None
        if oracle_document_embeddings is not None:
            oracle = _as_matrix(oracle_document_embeddings, "oracle_document_embeddings")
            if oracle.shape != self.document_embeddings.shape:
                raise ValueError("oracle document matrix shape must match base document matrix")
            self.oracle_document_embeddings = oracle
        self.qrels = {
            canonicalize_id(query_id): _positive_relevance(relevance)
            for query_id, relevance in qrels.items()
        }
        self.split = split
        self.config = config
        self.preregistration = json.loads(json.dumps(preregistration, sort_keys=True))
        self.preregistration_sha256 = _json_sha256(self.preregistration)
        anchor_used = {canonicalize_id(query_id) for query_id in anchor_query_ids_used}
        if not anchor_used or not anchor_used <= set(split.anchor_ids):
            raise ValueError("fit provenance must contain only ANCHOR query IDs")
        self._phase_query_ids: Dict[str, set] = {
            "ANCHOR": set(anchor_used),
            "SELECT": set(),
            "REPORT": set(),
        }
        self._anchor_document_ids_used = tuple(sorted({canonicalize_id(doc_id) for doc_id in anchor_document_ids_used}))
        self._view_cache: Dict[Tuple[str, bool], np.ndarray] = {}
        self._view_cache_checksums: Optional[Dict[Tuple[str, bool], str]] = None
        self._adapter_checksums = {key: _adapter_checksum(adapter) for key, adapter in self.adapters.items()}
        self._document_checksum = hashlib.sha256(np.ascontiguousarray(self.document_embeddings).tobytes()).hexdigest()
        self._document_ids_checksum = hashlib.sha256(
            json.dumps(self.document_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        self._oracle_checksum = (
            hashlib.sha256(np.ascontiguousarray(self.oracle_document_embeddings).tobytes()).hexdigest()
            if self.oracle_document_embeddings is not None
            else None
        )
        self._qrels_checksum = _json_sha256(self.qrels)
        self._split_checksum = _json_sha256(self.split.to_dict(include_ids=True))
        self._config_checksum = _json_sha256(asdict(self.config))
        self._router_checksum: Optional[str] = None
        self._selection_frozen = False
        self._report_consumed = False
        self._select_decision: Optional[Dict[str, Any]] = None
        self._select_decision_checksum: Optional[str] = None
        self._report_result: Optional[Dict[str, Any]] = None
        self._report_result_checksum: Optional[str] = None
        self._final_state_checksum: Optional[str] = None
        self._final_verdict = "UNSELECTED"
        self._state_lock = Lock()
        self._state = "ANCHOR_READY"

    @property
    def select_gate_passed(self) -> bool:
        self._require_stable_state()
        self._assert_decisions_frozen()
        self._assert_adapters_frozen()
        return bool(self._select_decision and self._select_decision.get("select_gate_passed"))

    @property
    def verdict(self) -> str:
        self._require_stable_state()
        self._assert_decisions_frozen()
        self._assert_adapters_frozen()
        return self._final_verdict

    @property
    def promoted(self) -> bool:
        return self.verdict == "PROMOTED"

    @property
    def report_consumed(self) -> bool:
        with self._state_lock:
            return self._report_consumed

    @property
    def state(self) -> str:
        with self._state_lock:
            return self._state

    def _require_stable_state(self) -> str:
        state = self.state
        if state in {"SELECTING", "REPORTING"}:
            raise RuntimeError(f"routing plane transition is in progress ({state})")
        return state

    def _assert_select_frozen(self) -> None:
        if self._selection_frozen:
            if self._select_decision is None or self._select_decision_checksum is None:
                raise RuntimeError("frozen SELECT decision is missing")
            if _json_sha256(self._select_decision) != self._select_decision_checksum:
                raise RuntimeError("frozen SELECT promotion decision changed")

    def _assert_decisions_frozen(self) -> None:
        self._assert_select_frozen()
        if self._report_consumed:
            if self._report_result is None:
                raise RuntimeError("consumed REPORT result is missing")
            if self._report_result_checksum is None or self._final_state_checksum is None:
                raise RuntimeError("frozen REPORT result checksum is missing")
            if _json_sha256(self._report_result) != self._report_result_checksum:
                raise RuntimeError("frozen REPORT result changed")
            final_state = {
                "select_decision_sha256": self._select_decision_checksum,
                "report_result_sha256": self._report_result_checksum,
                "report_consumed": self._report_consumed,
                "verdict": self._final_verdict,
                "router_state_sha256": self._router_checksum,
                "plane_state": self.state,
            }
            if _json_sha256(final_state) != self._final_state_checksum:
                raise RuntimeError("frozen final promotion state changed")

    def _assert_adapters_frozen(self) -> None:
        if _json_sha256(self.preregistration) != self.preregistration_sha256:
            raise RuntimeError("preregistration changed after plane construction")
        if _json_sha256(asdict(self.config)) != self._config_checksum:
            raise RuntimeError("routing configuration changed after plane construction")
        if _json_sha256(self.split.to_dict(include_ids=True)) != self._split_checksum:
            raise RuntimeError("three-way split changed after plane construction")
        if _json_sha256(self.qrels) != self._qrels_checksum:
            raise RuntimeError("qrels changed after plane construction")
        current = {key: _adapter_checksum(adapter) for key, adapter in self.adapters.items()}
        if current != self._adapter_checksums:
            raise RuntimeError("adapter state changed after ANCHOR fitting")
        checksum = hashlib.sha256(np.ascontiguousarray(self.document_embeddings).tobytes()).hexdigest()
        if checksum != self._document_checksum:
            raise RuntimeError("base document embeddings changed after ANCHOR fitting")
        document_ids_checksum = hashlib.sha256(
            json.dumps(self.document_ids, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        if document_ids_checksum != self._document_ids_checksum:
            raise RuntimeError("document IDs changed after plane construction")
        oracle_checksum = (
            hashlib.sha256(np.ascontiguousarray(self.oracle_document_embeddings).tobytes()).hexdigest()
            if self.oracle_document_embeddings is not None
            else None
        )
        if oracle_checksum != self._oracle_checksum:
            raise RuntimeError("oracle document embeddings changed after plane construction")
        if self._router_checksum is not None:
            if not self.router.frozen or self.router.state_checksum() != self._router_checksum:
                raise RuntimeError("frozen router state changed after SELECT promotion")
        if self._view_cache_checksums is not None:
            if set(self._view_cache) != set(self._view_cache_checksums):
                raise RuntimeError("frozen adapted document-view membership changed")
            for key, expected in self._view_cache_checksums.items():
                view = self._view_cache[key]
                checksum = hashlib.sha256(np.ascontiguousarray(view).tobytes()).hexdigest()
                if checksum != expected or view.flags.writeable:
                    raise RuntimeError(f"frozen adapted document view changed: {key}")

    def _freeze_views(self) -> Dict[str, str]:
        expected_keys = {
            (adapter_key, quantized)
            for adapter_key in self.adapters
            for quantized in (False, True)
        }
        if set(self._view_cache) != expected_keys:
            raise RuntimeError("SELECT did not evaluate every retained FP32 and quantized document view")
        checksums: Dict[Tuple[str, bool], str] = {}
        for key, view in self._view_cache.items():
            view.setflags(write=False)
            checksums[key] = hashlib.sha256(np.ascontiguousarray(view).tobytes()).hexdigest()
        self._view_cache_checksums = checksums
        return {
            f"{key[0]}:{'quantized' if key[1] else 'fp32'}": checksum
            for key, checksum in sorted(checksums.items())
        }

    def _view(self, adapter_key: str, quantized: bool) -> np.ndarray:
        cache_key = (str(adapter_key), bool(quantized))
        if cache_key not in self._view_cache:
            adapted = _apply_adapter_numpy(self.adapters[adapter_key], self.document_embeddings)
            if quantized:
                adapted = _simulate_int8_numpy(adapted, self.config.quant_levels, self.config.quantile)
            self._view_cache[cache_key] = _normalize_rows(adapted)
        return self._view_cache[cache_key]

    def _base_view(self) -> np.ndarray:
        return _normalize_rows(self.document_embeddings)

    def _oracle_view(self) -> np.ndarray:
        if self.oracle_document_embeddings is None:
            raise ValueError("C2O oracle document embeddings were not supplied")
        return _normalize_rows(self.oracle_document_embeddings)

    def _evaluate(
        self,
        query_vectors: Mapping[Any, np.ndarray],
        query_ids: Sequence[str],
        *,
        policy: str,
        quantized: bool = False,
        usage_scope: Optional[str] = None,
        fixed_route: Optional[str] = None,
    ) -> PlaneEvaluation:
        rows: List[Dict[str, Any]] = []
        base_view = self._base_view() if policy == "no_route" else None
        oracle_view = self._oracle_view() if policy == "c2o_oracle" else None
        for query_id in query_ids:
            query = np.asarray(query_vectors[query_id], dtype=np.float32)
            if query.ndim != 1 or query.shape[0] != self.document_embeddings.shape[1] or not np.all(np.isfinite(query)):
                raise ValueError(f"invalid query vector for {query_id}")
            route_key = None
            route_metadata = None
            if policy == "plane":
                route = self.router.select(query, usage_scope=usage_scope)
                route_key = route.key
                route_metadata = route.to_dict()
                view = self._view(route.key, quantized)
            elif policy == "single_global":
                route_key = "global"
                view = self._view("global", quantized)
            elif policy == "single_best_route":
                if fixed_route is None or fixed_route not in self.adapters or fixed_route == "global":
                    raise ValueError("single_best_route requires a frozen specialist route")
                route_key = fixed_route
                view = self._view(fixed_route, quantized)
            elif policy == "no_route":
                view = base_view
            elif policy == "c2o_oracle":
                view = oracle_view
            else:
                raise ValueError(f"unsupported routing policy: {policy}")
            ranked_ids = _rank_document_ids(query, view, self.document_ids, self.config.k)
            relevance = self.qrels.get(query_id, {})
            if not relevance:
                continue
            score = graded_ndcg_at_k(ranked_ids, relevance, self.config.k)
            rows.append(
                {
                    "query_id": query_id,
                    "ndcg": float(score),
                    "ranked_ids": ranked_ids,
                    "route_key": route_key,
                    "route": route_metadata,
                }
            )
        mean_ndcg = float(np.mean([row["ndcg"] for row in rows])) if rows else 0.0
        return PlaneEvaluation(policy=policy, mean_ndcg=mean_ndcg, rows=rows)

    def select(self, query_vectors: Mapping[Any, np.ndarray]) -> Dict[str, Any]:
        """Run and freeze the only promotion decision on SELECT."""

        with self._state_lock:
            if self._state != "ANCHOR_READY":
                raise RuntimeError(f"SELECT is unavailable in plane state {self._state}")
        canonical_vectors = _canonical_query_vectors(query_vectors)
        expected = set(self.split.select_ids)
        _validate_phase_query_vectors(
            canonical_vectors,
            expected,
            self.document_embeddings.shape[1],
            "SELECT",
        )
        if expected & set(self.split.report_ids):
            raise RuntimeError("split corruption: SELECT overlaps REPORT")
        with self._state_lock:
            if self._state != "ANCHOR_READY":
                raise RuntimeError(f"SELECT is unavailable in plane state {self._state}")
            self._state = "SELECTING"
        try:
            return self._run_select(canonical_vectors, expected)
        except BaseException:
            self._rollback_select_transition()
            raise

    def _run_select(
        self,
        canonical_vectors: Mapping[str, np.ndarray],
        expected: set,
    ) -> Dict[str, Any]:
        """Execute one reserved SELECT transition."""

        self._assert_adapters_frozen()
        router_checksum = self.router.freeze()
        if self._router_checksum is None:
            self._router_checksum = router_checksum
        elif router_checksum != self._router_checksum:
            raise RuntimeError("router state changed between SELECT attempts")
        self._assert_adapters_frozen()
        self.router.reset_usage("SELECT")

        no_route = self._evaluate(canonical_vectors, self.split.select_ids, policy="no_route")
        global_fp32 = self._evaluate(canonical_vectors, self.split.select_ids, policy="single_global")
        plane_fp32 = self._evaluate(
            canonical_vectors,
            self.split.select_ids,
            policy="plane",
            usage_scope="SELECT",
        )
        plane_quant = self._evaluate(canonical_vectors, self.split.select_ids, policy="plane", quantized=True)
        ci = paired_query_bootstrap_ci(
            plane_fp32.scores_by_query,
            global_fp32.scores_by_query,
            confidence_level=self.config.confidence_level,
            resamples=self.config.bootstrap_resamples,
            seed=self.config.bootstrap_seed,
        )

        specialist_routes = sorted(key for key in self.adapters if key != "global")
        route_ablation_scores: Dict[str, float] = {}
        route_fp32_evaluations: Dict[str, PlaneEvaluation] = {}
        route_quant_evaluations: Dict[str, PlaneEvaluation] = {}
        for route_key in specialist_routes:
            route_fp32 = self._evaluate(
                canonical_vectors,
                self.split.select_ids,
                policy="single_best_route",
                fixed_route=route_key,
            )
            route_quant = self._evaluate(
                canonical_vectors,
                self.split.select_ids,
                policy="single_best_route",
                fixed_route=route_key,
                quantized=True,
            )
            route_fp32_evaluations[route_key] = route_fp32
            route_quant_evaluations[route_key] = route_quant
            route_ablation_scores[route_key] = route_fp32.mean_ndcg
        best_route = None
        if route_ablation_scores:
            best_route = sorted(route_ablation_scores, key=lambda key: (-route_ablation_scores[key], key))[0]

        no_route_by_query = no_route.scores_by_query
        fp32_by_query = plane_fp32.scores_by_query
        quant_by_query = plane_quant.scores_by_query
        global_quant = self._evaluate(
            canonical_vectors,
            self.split.select_ids,
            policy="single_global",
            quantized=True,
        )
        selected_keys = {str(row["query_id"]): str(row["route_key"]) for row in plane_fp32.rows}
        quant_gate = QuantizationPromotionGate(
            retained_gain_threshold=self.config.quant_retained_gain,
            minimum_fp32_gain=self.config.quant_minimum_fp32_gain,
        )
        quant_decisions: Dict[str, Dict[str, Any]] = {}
        for adapter_key in sorted(self.adapters):
            subset = sorted(query_id for query_id, key in selected_keys.items() if key == adapter_key)
            selected_on_select = bool(subset)
            if not subset:
                # A retained route that SELECT happened not to choose can still
                # win a REPORT/serving query.  Gate it on all SELECT queries so
                # no frozen route can bypass quant survival.
                subset = list(self.split.select_ids)
            if selected_on_select:
                adapter_fp32_by_query = fp32_by_query
                adapter_quant_by_query = quant_by_query
                baseline_source = "no_route_same_routed_select_subset"
            elif adapter_key == "global":
                adapter_fp32_by_query = global_fp32.scores_by_query
                adapter_quant_by_query = global_quant.scores_by_query
                baseline_source = "no_route_full_select_for_unseen_retained_adapter"
            else:
                adapter_fp32_by_query = route_fp32_evaluations[adapter_key].scores_by_query
                adapter_quant_by_query = route_quant_evaluations[adapter_key].scores_by_query
                baseline_source = "no_route_full_select_for_unseen_retained_adapter"
            baseline = float(np.mean([no_route_by_query[query_id] for query_id in subset]))
            fp32 = float(np.mean([adapter_fp32_by_query[query_id] for query_id in subset]))
            quantized = float(np.mean([adapter_quant_by_query[query_id] for query_id in subset]))
            decision = quant_gate.evaluate(fp32, quantized, baseline_fitness=baseline)
            quant_decisions[adapter_key] = {
                "query_ids": subset,
                "query_count": len(subset),
                "selected_on_select": selected_on_select,
                "baseline_source": baseline_source,
                **decision.to_dict(),
            }

        quant_passed = bool(quant_decisions) and all(item["passed"] for item in quant_decisions.values())
        ci_passed = bool(ci.lower > self.config.min_lift)
        floor_passed = bool(plane_quant.mean_ndcg >= no_route.mean_ndcg - self.config.max_floor_loss)
        select_gate_passed = bool(ci_passed and quant_passed and floor_passed)
        reasons: List[str] = []
        if not ci_passed:
            reasons.append("plane_ci_lower_bound_not_above_min_lift")
        if not quant_passed:
            reasons.append("one_or_more_retained_adapters_failed_quant_survival")
        if not floor_passed:
            reasons.append("quantized_plane_materially_below_no_route")

        self._phase_query_ids["SELECT"] = set(expected)
        self._select_decision = {
            "record_type": "rung16_select_promotion_decision",
            "frozen": True,
            "select_gate_passed": select_gate_passed,
            "reasons": reasons,
            "ci_passed": ci_passed,
            "paired_bootstrap_ci": ci.to_dict(),
            "min_lift": self.config.min_lift,
            "floor_passed": floor_passed,
            "max_floor_loss": self.config.max_floor_loss,
            "quant_passed": quant_passed,
            "quant_pass_rate": (
                float(sum(1 for item in quant_decisions.values() if item["passed"]) / len(quant_decisions))
                if quant_decisions
                else 0.0
            ),
            "quant_decisions": quant_decisions,
            "usage": self.router.get_usage_summary("SELECT"),
            "metrics": {
                "no_route": no_route.to_dict(),
                "single_global_fp32": global_fp32.to_dict(),
                "single_global_quantized": global_quant.to_dict(),
                "plane_fp32": plane_fp32.to_dict(),
                "plane_quantized": plane_quant.to_dict(),
            },
            "single_best_route": best_route,
            "single_route_select_scores": route_ablation_scores,
            "router_state_sha256": self._router_checksum,
        }
        self._assert_adapters_frozen()
        frozen_view_checksums = self._freeze_views()
        self._select_decision["adapted_document_view_sha256"] = frozen_view_checksums
        self._select_decision_checksum = _json_sha256(self._select_decision)
        self._selection_frozen = True
        self._final_verdict = "SELECT-PASSED" if select_gate_passed else "FAIL-CLOSED-PENDING-REPORT"
        self._assert_decisions_frozen()
        self._assert_adapters_frozen()
        with self._state_lock:
            if self._state != "SELECTING":
                raise RuntimeError(f"invalid SELECT finalization state {self._state}")
            self._state = "SELECT_FROZEN"
        return json.loads(json.dumps(self._select_decision))

    def _rollback_select_transition(self) -> None:
        """Rollback transient SELECT data while retaining the frozen router."""

        self.router.reset_usage("SELECT")
        self._phase_query_ids["SELECT"] = set()
        self._view_cache.clear()
        self._view_cache_checksums = None
        self._selection_frozen = False
        self._select_decision = None
        self._select_decision_checksum = None
        self._report_consumed = False
        self._report_result = None
        self._report_result_checksum = None
        self._final_state_checksum = None
        self._final_verdict = "UNSELECTED"
        with self._state_lock:
            self._state = "ANCHOR_READY"

    def report(self, query_vectors: Mapping[Any, np.ndarray]) -> Dict[str, Any]:
        """Evaluate the frozen plane once; REPORT can only downgrade degeneracy."""

        with self._state_lock:
            if self._state != "SELECT_FROZEN":
                raise RuntimeError(f"REPORT is unavailable in plane state {self._state}")
        self._assert_decisions_frozen()
        canonical_vectors = _canonical_query_vectors(query_vectors)
        expected = set(self.split.report_ids)
        _validate_phase_query_vectors(
            canonical_vectors,
            expected,
            self.document_embeddings.shape[1],
            "REPORT",
        )
        self._assert_adapters_frozen()
        with self._state_lock:
            if self._state != "SELECT_FROZEN":
                raise RuntimeError(f"REPORT is unavailable in plane state {self._state}")
            self._state = "REPORTING"
            self._report_consumed = True
            self._phase_query_ids["REPORT"] = set(expected)
        try:
            return self._run_report(canonical_vectors)
        except BaseException as exc:
            self._finalize_report_failure(exc)
            raise

    def _run_report(
        self,
        canonical_vectors: Mapping[str, np.ndarray],
    ) -> Dict[str, Any]:
        """Execute one atomically reserved REPORT transition."""

        # REPORT is already atomically reserved and therefore intentionally has
        # no result yet.  Verify the frozen SELECT decision here; public readers
        # remain blocked while the plane is in REPORTING.
        self._assert_select_frozen()
        self._assert_adapters_frozen()
        self.router.reset_usage("REPORT")

        no_route = self._evaluate(canonical_vectors, self.split.report_ids, policy="no_route")
        global_fp32 = self._evaluate(canonical_vectors, self.split.report_ids, policy="single_global")
        plane_fp32 = self._evaluate(
            canonical_vectors,
            self.split.report_ids,
            policy="plane",
            usage_scope="REPORT",
        )
        plane_quant = self._evaluate(canonical_vectors, self.split.report_ids, policy="plane", quantized=True)
        best_route = self._select_decision.get("single_best_route")
        best_route_eval = None
        if best_route is not None:
            best_route_eval = self._evaluate(
                canonical_vectors,
                self.split.report_ids,
                policy="single_best_route",
                fixed_route=str(best_route),
            )
        oracle = None
        if self.oracle_document_embeddings is not None:
            oracle = self._evaluate(canonical_vectors, self.split.report_ids, policy="c2o_oracle")
        report_ci = paired_query_bootstrap_ci(
            plane_fp32.scores_by_query,
            global_fp32.scores_by_query,
            confidence_level=self.config.confidence_level,
            resamples=self.config.bootstrap_resamples,
            seed=self.config.bootstrap_seed,
        )

        usage = self.router.get_usage_summary("REPORT")
        binding_routes = sorted(
            key
            for key, fraction in usage["route_p_k"].items()
            if float(fraction) >= self.config.min_report_route_fraction
        )
        binding_passed = len(binding_routes) >= self.config.min_report_routes
        if not binding_passed:
            verdict = "DEGENERATE"
        elif bool(self._select_decision.get("select_gate_passed")):
            verdict = "PROMOTED"
        else:
            verdict = "FAIL-CLOSED"
        self._assert_adapters_frozen()
        self._final_verdict = verdict
        self._report_result = {
            "record_type": "rung16_frozen_report",
            "verdict": verdict,
            "report_is_descriptive_not_promotional": True,
            "multi_route_binding": {
                "passed": binding_passed,
                "minimum_routes": self.config.min_report_routes,
                "minimum_fraction_per_route": self.config.min_report_route_fraction,
                "qualifying_routes": binding_routes,
            },
            "usage": usage,
            "plane_vs_single_global_delta": float(plane_fp32.mean_ndcg - global_fp32.mean_ndcg),
            "paired_bootstrap_ci": report_ci.to_dict(),
            "select_quant_pass_rate": self._select_decision["quant_pass_rate"],
            "metrics": {
                "no_route": no_route.to_dict(),
                "single_global_fp32": global_fp32.to_dict(),
                "plane_fp32": plane_fp32.to_dict(),
                "plane_quantized": plane_quant.to_dict(),
                "single_best_route": best_route_eval.to_dict() if best_route_eval is not None else None,
                "c2o_oracle": oracle.to_dict() if oracle is not None else None,
            },
        }
        self._report_result_checksum = _json_sha256(self._report_result)
        self._final_state_checksum = _json_sha256(
            {
                "select_decision_sha256": self._select_decision_checksum,
                "report_result_sha256": self._report_result_checksum,
                "report_consumed": self._report_consumed,
                "verdict": self._final_verdict,
                "router_state_sha256": self._router_checksum,
                "plane_state": "FINALIZED",
            }
        )
        with self._state_lock:
            if self._state != "REPORTING":
                raise RuntimeError(f"invalid REPORT finalization state {self._state}")
            self._state = "FINALIZED"
        self._assert_decisions_frozen()
        return json.loads(json.dumps(self._report_result))

    def _finalize_report_failure(self, exc: BaseException) -> None:
        """Consume a failed REPORT attempt and preserve a terminal fail-closed record."""

        self.router.reset_usage("REPORT")
        self._final_verdict = "FAIL-CLOSED"
        self._report_result = {
            "record_type": "rung16_report_failure",
            "verdict": "FAIL-CLOSED",
            "report_is_descriptive_not_promotional": True,
            "reason": "report_evaluation_exception",
            "error_type": type(exc).__name__,
        }
        self._report_result_checksum = _json_sha256(self._report_result)
        self._final_state_checksum = _json_sha256(
            {
                "select_decision_sha256": self._select_decision_checksum,
                "report_result_sha256": self._report_result_checksum,
                "report_consumed": self._report_consumed,
                "verdict": self._final_verdict,
                "router_state_sha256": self._router_checksum,
                "plane_state": "REPORT_FAILED_CLOSED",
            }
        )
        with self._state_lock:
            self._state = "REPORT_FAILED_CLOSED"

    def retrieve(
        self,
        query_vector: Iterable[float],
        *,
        k: Optional[int] = None,
        quantized: bool = True,
        usage_scope: Optional[str] = "SERVE",
    ) -> Dict[str, Any]:
        """Retrieve through the final promoted plane for engine integration."""

        if not self.promoted:
            raise RuntimeError(f"quant-aware routing plane is not promoted (verdict={self.verdict})")
        self._assert_decisions_frozen()
        self._assert_adapters_frozen()
        retrieval_k = (
            self.config.k
            if k is None
            else _positive_builtin_int(k, "k")
        )
        query = np.asarray(list(query_vector), dtype=np.float32)
        route = self.router.select(query, usage_scope=usage_scope)
        view = self._view(route.key, bool(quantized))
        result_ids = _rank_document_ids(query, view, self.document_ids, retrieval_k)
        return {
            "ids": result_ids,
            "route": route.to_dict(),
            "quantized": bool(quantized),
            "verdict": self.verdict,
        }

    def provenance(self) -> Dict[str, Any]:
        """Return JSON-safe immutable promotion and leakage provenance."""

        state = self._require_stable_state()
        self._assert_decisions_frozen()
        self._assert_adapters_frozen()
        leakage_safe = (
            self._phase_query_ids["ANCHOR"] <= set(self.split.anchor_ids)
            and self._phase_query_ids["SELECT"] <= set(self.split.select_ids)
            and self._phase_query_ids["REPORT"] <= set(self.split.report_ids)
            and not (self._phase_query_ids["ANCHOR"] & self._phase_query_ids["SELECT"])
            and not (self._phase_query_ids["ANCHOR"] & self._phase_query_ids["REPORT"])
            and not (self._phase_query_ids["SELECT"] & self._phase_query_ids["REPORT"])
        )
        provenance = {
            "record_type": "rung16_quant_aware_routing_provenance",
            "preregistration_sha256": self.preregistration_sha256,
            "split": self.split.to_dict(include_ids=False),
            "fit": {
                "anchor_query_count": len(self._phase_query_ids["ANCHOR"]),
                "anchor_query_sha256": _ids_sha256(self._phase_query_ids["ANCHOR"]),
                "anchor_document_count": len(self._anchor_document_ids_used),
                "anchor_document_sha256": _ids_sha256(self._anchor_document_ids_used),
                "adapter_checksums": dict(self._adapter_checksums),
                "router_checksum": self._router_checksum,
            },
            "select": self._select_decision,
            "report": self._report_result,
            "report_consumed": self._report_consumed,
            "leakage_safe": bool(leakage_safe),
            "verdict": self._final_verdict,
            "plane_state": state,
        }
        return json.loads(json.dumps(provenance))


def fit_quant_aware_plane(
    *,
    document_ids: Sequence[Any],
    document_embeddings: np.ndarray,
    qrels: Mapping[Any, Mapping[Any, float]],
    split: ThreeWaySplit,
    anchor_pairs: Sequence[Mapping[str, Any]],
    config: RoutingPlaneConfig,
    preregistration: Mapping[str, Any],
    oracle_document_embeddings: Optional[np.ndarray] = None,
    device: str = "cuda",
) -> QuantAwareRoutingPlane:
    """Fit global and specialist adapters from ANCHOR pairs only.

    Each pair must contain ``query_id``, ``doc_id``, ``doc_vector`` (the
    relevance-weighted positive-document centroid), ``query_vector`` (the
    swapped query embedding), and ``route_key``.  The function rejects any
    non-ANCHOR query before constructing an adapter.
    """

    config.validate()
    split.validate()
    if not anchor_pairs:
        raise ValueError("ANCHOR produced no training pairs")
    anchor_ids = set(split.anchor_ids)
    pair_query_ids = {canonicalize_id(pair["query_id"]) for pair in anchor_pairs}
    if not pair_query_ids or not pair_query_ids <= anchor_ids:
        raise ValueError("adapter fitting may consume only ANCHOR query IDs")
    if pair_query_ids & (set(split.select_ids) | set(split.report_ids)):
        raise ValueError("SELECT or REPORT query leaked into adapter fitting")

    vectors = _as_matrix(document_embeddings, "document_embeddings")
    input_dim = int(vectors.shape[1])
    grouped: Dict[str, List[Mapping[str, Any]]] = {}
    for pair in anchor_pairs:
        grouped.setdefault(str(pair["route_key"]), []).append(pair)
    eligible = {
        key: rows
        for key, rows in grouped.items()
        if len({doc_id for row in rows for doc_id in _pair_source_doc_ids(row)}) >= config.min_cluster_documents
    }

    global_adapter = _fit_adapter(anchor_pairs, input_dim, config, device=device)
    adapters: Dict[str, Any] = {"global": global_adapter}
    router = AdapterRouter(margin_delta=config.margin_delta)
    global_docs = _unique_pair_vectors(anchor_pairs)
    global_centroid = np.mean(_apply_adapter_numpy(global_adapter, global_docs), axis=0)
    router.register_global(global_centroid, global_adapter)
    for route_key, rows in sorted(eligible.items()):
        adapter = _fit_adapter(rows, input_dim, config, device=device)
        route_docs = _unique_pair_vectors(rows)
        centroid = np.mean(_apply_adapter_numpy(adapter, route_docs), axis=0)
        adapters[route_key] = adapter
        router.register(route_key, centroid, adapter)

    return QuantAwareRoutingPlane(
        router=router,
        adapters=adapters,
        document_ids=document_ids,
        document_embeddings=vectors,
        qrels=qrels,
        split=split,
        config=config,
        preregistration=preregistration,
        anchor_query_ids_used=pair_query_ids,
        anchor_document_ids_used=[
            doc_id
            for pair in anchor_pairs
            for doc_id in _pair_source_doc_ids(pair)
        ],
        oracle_document_embeddings=oracle_document_embeddings,
    )


def _fit_adapter(
    pairs: Sequence[Mapping[str, Any]],
    input_dim: int,
    config: RoutingPlaneConfig,
    *,
    device: str,
):
    import torch

    from chelation_adapter import create_adapter
    from sedimentation_loss import SedimentationInfoNCELoss

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA is required by the frozen Rung 16 preregistration")
    torch.manual_seed(int(config.split_seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(config.split_seed))
    adapter = create_adapter(
        "mlp",
        input_dim=input_dim,
        bounded=True,
        min_correction=config.adapter_min_correction,
        max_correction=config.adapter_max_correction,
    ).to(device)
    inputs = torch.tensor(
        np.asarray([pair["doc_vector"] for pair in pairs], dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    targets = torch.tensor(
        np.asarray([pair["query_vector"] for pair in pairs], dtype=np.float32),
        dtype=torch.float32,
        device=device,
    )
    if inputs.ndim != 2 or targets.shape != inputs.shape:
        raise ValueError("ANCHOR adapter inputs and query targets must have equal 2D shapes")
    loss_fn = SedimentationInfoNCELoss()
    optimizer = torch.optim.Adam(adapter.parameters(), lr=config.adapter_learning_rate)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(config.split_seed))
    batch_size = min(config.adapter_batch_size, len(pairs))
    order: List[int] = []
    adapter.train()
    for _ in range(config.adapter_steps):
        if len(order) < batch_size:
            order.extend(torch.randperm(len(pairs), generator=generator).tolist())
        indices = order[:batch_size]
        del order[:batch_size]
        index_tensor = torch.tensor(indices, dtype=torch.long, device=device)
        optimizer.zero_grad()
        outputs = adapter(inputs.index_select(0, index_tensor))
        loss = loss_fn(outputs, targets.index_select(0, index_tensor))
        loss.backward()
        optimizer.step()
    adapter.eval()
    return adapter.to("cpu")


def _unique_pair_vectors(pairs: Sequence[Mapping[str, Any]]) -> np.ndarray:
    by_doc: Dict[str, np.ndarray] = {}
    for pair in pairs:
        by_doc.setdefault(canonicalize_id(pair["doc_id"]), np.asarray(pair["doc_vector"], dtype=np.float32))
    return _as_matrix(np.asarray([by_doc[key] for key in sorted(by_doc)], dtype=np.float32), "anchor_doc_vectors")


def _pair_source_doc_ids(pair: Mapping[str, Any]) -> List[str]:
    source = pair.get("source_doc_ids")
    if source is None:
        source = [pair["doc_id"]]
    return [canonicalize_id(doc_id) for doc_id in source]


def _apply_adapter_numpy(adapter: Any, matrix: np.ndarray, batch_size: int = 2048) -> np.ndarray:
    import torch

    source = _as_matrix(matrix, "adapter_input")
    try:
        first_parameter = next(adapter.parameters())
        device = first_parameter.device
    except (AttributeError, StopIteration):
        device = torch.device("cpu")
    chunks = []
    if hasattr(adapter, "eval"):
        adapter.eval()
    with torch.no_grad():
        for start in range(0, len(source), batch_size):
            tensor = torch.tensor(source[start : start + batch_size], dtype=torch.float32, device=device)
            output = adapter(tensor)
            if not isinstance(output, torch.Tensor):
                output = torch.as_tensor(output)
            chunks.append(output.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0) if chunks else np.empty_like(source)


def _simulate_int8_numpy(matrix: np.ndarray, levels: int, quantile: float) -> np.ndarray:
    import torch

    from evolution_strategies_optimizer import simulate_int8_quantization

    with torch.no_grad():
        quantized = simulate_int8_quantization(
            torch.tensor(matrix, dtype=torch.float32),
            levels=int(levels),
            quantile=float(quantile),
        )
    return quantized.detach().cpu().numpy().astype(np.float32)


def _rank_document_ids(
    query_vector: np.ndarray,
    normalized_documents: np.ndarray,
    document_ids: Sequence[str],
    k: int,
) -> List[str]:
    validated_k = _positive_builtin_int(k, "k")
    query = np.asarray(query_vector, dtype=np.float32)
    query_unit = _finite_unit_vector(query, "query vector")
    if query_unit is None:
        raise ValueError("query vector must be non-zero")
    scores = normalized_documents @ query_unit
    if not np.all(np.isfinite(scores)):
        raise ValueError("derived document scores must be finite")
    order = np.argsort(-scores, kind="mergesort")[
        : min(validated_k, len(document_ids))
    ]
    return [str(document_ids[index]) for index in order]


def _normalize_rows(matrix: np.ndarray) -> np.ndarray:
    values = _as_matrix(matrix, "embedding_matrix")
    scales = np.max(np.abs(values), axis=1, keepdims=True)
    safe_scales = np.where(scales > 0.0, scales, 1.0)
    scaled = values / safe_scales
    scaled_norms = np.linalg.norm(scaled.astype(np.float64), axis=1, keepdims=True)
    if not np.all(np.isfinite(scaled_norms)):
        raise ValueError("embedding row normalization must remain finite")
    safe_norms = np.where(scaled_norms > 0.0, scaled_norms, 1.0)
    normalized = scaled.astype(np.float64) / safe_norms
    if not np.all(np.isfinite(normalized)):
        raise ValueError("embedding row normalization must remain finite")
    return normalized.astype(np.float32)


def _finite_unit_vector(vector: np.ndarray, name: str) -> Optional[np.ndarray]:
    values = np.asarray(vector, dtype=np.float32)
    if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
        raise ValueError(f"{name} must be a non-empty finite 1D vector")
    scale = float(np.max(np.abs(values)))
    if scale == 0.0:
        return None
    scaled = values / scale
    scaled_norm = float(np.linalg.norm(scaled.astype(np.float64)))
    if not math.isfinite(scaled_norm) or scaled_norm <= 0.0:
        raise ValueError(f"{name} normalization must remain finite")
    unit = scaled.astype(np.float64) / scaled_norm
    if not np.all(np.isfinite(unit)):
        raise ValueError(f"{name} normalization must remain finite")
    return unit.astype(np.float32)


def _as_matrix(value: Any, name: str) -> np.ndarray:
    matrix = np.asarray(value, dtype=np.float32)
    if matrix.ndim != 2 or matrix.shape[0] < 1 or matrix.shape[1] < 1:
        raise ValueError(f"{name} must be a non-empty 2D matrix")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{name} must contain only finite values")
    return matrix


def _canonical_query_vectors(query_vectors: Mapping[Any, np.ndarray]) -> Dict[str, np.ndarray]:
    return {canonicalize_id(query_id): np.asarray(vector, dtype=np.float32) for query_id, vector in query_vectors.items()}


def _validate_phase_query_vectors(
    query_vectors: Mapping[str, np.ndarray],
    expected_ids: set,
    dimension: int,
    phase: str,
) -> None:
    if set(query_vectors) != expected_ids:
        raise ValueError(
            f"{phase} evaluation requires exactly the frozen {phase} query IDs"
        )
    for query_id in sorted(expected_ids):
        vector = np.asarray(query_vectors[query_id], dtype=np.float32)
        if (
            vector.ndim != 1
            or vector.shape[0] != int(dimension)
            or not np.all(np.isfinite(vector))
        ):
            raise ValueError(f"invalid {phase} query vector for {query_id}")
        if _finite_unit_vector(vector, f"{phase} query vector") is None:
            raise ValueError(f"{phase} query vector must be non-zero for {query_id}")


def _ids_sha256(ids: Iterable[Any]) -> str:
    payload = "\n".join(sorted(canonicalize_id(value) for value in ids)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _json_sha256(value: Mapping[str, Any]) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _adapter_checksum(adapter: Any) -> str:
    digest = hashlib.sha256()
    if not hasattr(adapter, "state_dict"):
        digest.update(repr(adapter).encode("utf-8"))
        return digest.hexdigest()
    for name, tensor in sorted(adapter.state_dict().items()):
        digest.update(str(name).encode("utf-8"))
        digest.update(np.ascontiguousarray(tensor.detach().cpu().numpy()).tobytes())
    return digest.hexdigest()


__all__ = [
    "BootstrapCI",
    "PlaneEvaluation",
    "QuantAwareRoutingPlane",
    "RoutingPlaneConfig",
    "ThreeWaySplit",
    "fit_quant_aware_plane",
    "graded_ndcg_at_k",
    "paired_query_bootstrap_ci",
    "three_way_seeded_split",
]
