"""METHOD_DEV analysis primitives for conditional replacement composition.

This module implements the falsifiable quantities in the CRSV/LIR/SRS
research protocol:

* CRSV: serving-subdomain x replacement-provenance interaction;
* LIR: layered inversion reduction, reversal burden, set interaction, and
  replacement-order sensitivity;
* SRS: subspace coherence, non-normal transient amplification, and useful
  signal atrophy.

The functions are analysis machinery, not evidence that any scientific
hypothesis is true.  They accept already-frozen observations and never train a
router or inspect qrels to make a serving decision.

Only NumPy and the Python standard library are required.  The implementation
is compatible with Python 3.9.
"""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Sequence, Tuple, TypeVar

import numpy as np


TIE_TOLERANCE = 1e-12
DEFAULT_EPSILON = 1e-12
DEFAULT_PERMUTATION_SEED = 271828
DEFAULT_MAX_PERMUTATIONS = 10000
_HEX_64_RE = re.compile(r"^[0-9a-f]{64}$")
_T = TypeVar("_T")


class CRSVValidationError(ValueError):
    """Raised when a METHOD_DEV input violates its declared contract."""


@dataclass(frozen=True)
class SpaceContract:
    """Exact representation-space contract at one replacement boundary."""

    role: str
    metric: str
    dimension: int
    normalization: str
    dtype: str
    lineage_id: str
    compatibility_domain_id: str

    def __post_init__(self) -> None:
        for name in (
            "role",
            "metric",
            "normalization",
            "dtype",
            "lineage_id",
            "compatibility_domain_id",
        ):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise CRSVValidationError("SpaceContract.{} must be a non-empty string".format(name))
        if isinstance(self.dimension, bool) or not isinstance(self.dimension, int):
            raise CRSVValidationError("SpaceContract.dimension must be an integer")
        if self.dimension <= 0:
            raise CRSVValidationError("SpaceContract.dimension must be positive")


@dataclass(frozen=True)
class ReplacementStep:
    """One replacement with declared provenance metadata."""

    step_id: str
    layer_id: str
    input_contract: SpaceContract
    output_contract: SpaceContract
    evidence_sha256: str
    evidence_state: str
    evidence_expires_at: str

    def __post_init__(self) -> None:
        if not isinstance(self.input_contract, SpaceContract):
            raise CRSVValidationError("ReplacementStep.input_contract must be a SpaceContract")
        if not isinstance(self.output_contract, SpaceContract):
            raise CRSVValidationError("ReplacementStep.output_contract must be a SpaceContract")
        if not isinstance(self.step_id, str) or not self.step_id.strip():
            raise CRSVValidationError("ReplacementStep.step_id must be a non-empty string")
        if not isinstance(self.layer_id, str) or not self.layer_id.strip():
            raise CRSVValidationError("ReplacementStep.layer_id must be a non-empty string")
        if not _HEX_64_RE.fullmatch(self.evidence_sha256):
            raise CRSVValidationError("ReplacementStep.evidence_sha256 must be lowercase SHA-256")
        if self.evidence_state != "VALID":
            raise CRSVValidationError("ReplacementStep.evidence_state must be VALID")
        _parse_utc_timestamp(self.evidence_expires_at, "evidence_expires_at")


def _parse_utc_timestamp(value: str, field: str) -> datetime:
    if not isinstance(value, str) or not value:
        raise CRSVValidationError("{} must be a non-empty timestamp".format(field))
    normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise CRSVValidationError("{} must be ISO-8601".format(field)) from exc
    if parsed.tzinfo is None:
        raise CRSVValidationError("{} must include a timezone".format(field))
    return parsed.astimezone(timezone.utc)


def validate_declared_replacement_path(
    initial_contract: SpaceContract,
    steps: Sequence[ReplacementStep],
    *,
    evaluated_at: str,
) -> Tuple[SpaceContract, ...]:
    """Validate declared path shape/time metadata and return prefix contracts.

    The first returned element is ``initial_contract``.  Each later element is
    the exact output contract after the corresponding step.  Declared evidence
    metadata must say ``VALID`` and be unexpired at ``evaluated_at``;
    input/output contracts must chain exactly.

    This function does not fetch evidence bytes, authenticate an issuer, or
    consult a revocation registry.  It is therefore a fail-closed structural
    preflight, not evidence authentication or a certificate gate.
    """

    if not isinstance(initial_contract, SpaceContract):
        raise CRSVValidationError("initial_contract must be a SpaceContract")
    evaluated = _parse_utc_timestamp(evaluated_at, "evaluated_at")
    current = initial_contract
    prefixes: List[SpaceContract] = [current]
    seen_steps = set()
    seen_layers = set()
    for index, step in enumerate(steps):
        if not isinstance(step, ReplacementStep):
            raise CRSVValidationError("steps[{}] must be a ReplacementStep".format(index))
        if step.step_id in seen_steps:
            raise CRSVValidationError("replacement step IDs must be unique")
        if step.layer_id in seen_layers:
            raise CRSVValidationError("replacement layer IDs must be unique within one path")
        expires = _parse_utc_timestamp(step.evidence_expires_at, "evidence_expires_at")
        if expires <= evaluated:
            raise CRSVValidationError("replacement evidence is expired at evaluation time")
        if step.input_contract != current:
            raise CRSVValidationError(
                "replacement prefix {} input contract does not match the prior output".format(index)
            )
        seen_steps.add(step.step_id)
        seen_layers.add(step.layer_id)
        current = step.output_contract
        prefixes.append(current)
    return tuple(prefixes)


def deterministic_ranking(scores: Mapping[str, float]) -> Tuple[str, ...]:
    """Return score-descending IDs with identifier-ascending tie resolution."""

    if not isinstance(scores, Mapping) or not scores:
        raise CRSVValidationError("scores must be a non-empty mapping")
    rows = []
    for identifier, raw_score in scores.items():
        if not isinstance(identifier, str) or not identifier:
            raise CRSVValidationError("ranking identifiers must be non-empty strings")
        if isinstance(raw_score, bool) or not isinstance(raw_score, (int, float)):
            raise CRSVValidationError("ranking scores must be numeric")
        score = float(raw_score)
        if not math.isfinite(score):
            raise CRSVValidationError("ranking scores must be finite")
        rows.append((identifier, score))
    rows.sort(key=lambda item: (-item[1], item[0]))
    return tuple(identifier for identifier, _ in rows)


def _validate_ranking(ranking: Sequence[str], label: str) -> Tuple[str, ...]:
    if isinstance(ranking, (str, bytes)) or not isinstance(ranking, Sequence):
        raise CRSVValidationError("{} must be a sequence of identifiers".format(label))
    normalized = tuple(ranking)
    if len(normalized) < 2:
        raise CRSVValidationError("{} must contain at least two identifiers".format(label))
    if any(not isinstance(item, str) or not item for item in normalized):
        raise CRSVValidationError("{} identifiers must be non-empty strings".format(label))
    if len(set(normalized)) != len(normalized):
        raise CRSVValidationError("{} identifiers must be unique".format(label))
    return normalized


def _pair_key(left: str, right: str) -> Tuple[str, str]:
    return (left, right) if left < right else (right, left)


def weighted_kendall_inversion_loss(
    reference_ranking: Sequence[str],
    candidate_ranking: Sequence[str],
    pair_weights: Optional[Mapping[Tuple[str, str], float]] = None,
) -> float:
    """Compute normalized weighted Kendall disagreement in ``[0, 1]``.

    Rankings must contain the same unique identifiers.  Score ties should be
    resolved before calling this function with :func:`deterministic_ranking`.
    ``pair_weights`` may omit pairs, which then receive unit weight.
    """

    reference = _validate_ranking(reference_ranking, "reference_ranking")
    candidate = _validate_ranking(candidate_ranking, "candidate_ranking")
    if set(reference) != set(candidate):
        raise CRSVValidationError("reference and candidate rankings must contain identical IDs")
    weights: Dict[Tuple[str, str], float] = {}
    if pair_weights is not None:
        if not isinstance(pair_weights, Mapping):
            raise CRSVValidationError("pair_weights must be a mapping")
        for raw_pair, raw_weight in pair_weights.items():
            if not isinstance(raw_pair, tuple) or len(raw_pair) != 2:
                raise CRSVValidationError("pair_weights keys must be two-item tuples")
            left, right = raw_pair
            if left == right or left not in reference or right not in reference:
                raise CRSVValidationError("pair_weights contains an invalid ranking pair")
            if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
                raise CRSVValidationError("pair weights must be numeric")
            weight = float(raw_weight)
            if not math.isfinite(weight) or weight < 0.0:
                raise CRSVValidationError("pair weights must be finite and non-negative")
            key = _pair_key(left, right)
            if key in weights:
                raise CRSVValidationError("pair_weights contains a duplicate undirected pair")
            weights[key] = weight

    candidate_position = {identifier: index for index, identifier in enumerate(candidate)}
    weight_scale = max(1.0, max(weights.values(), default=1.0))
    disagreement = 0.0
    total_weight = 0.0
    for left_index, left in enumerate(reference):
        for right in reference[left_index + 1 :]:
            weight = weights.get(_pair_key(left, right), 1.0) / weight_scale
            total_weight += weight
            if candidate_position[left] > candidate_position[right]:
                disagreement += weight
    if not math.isfinite(total_weight) or not math.isfinite(disagreement) or total_weight <= 0.0:
        raise CRSVValidationError("weighted ranking pair universe must have positive mass")
    result = float(disagreement / total_weight)
    if not math.isfinite(result):
        raise CRSVValidationError("weighted inversion loss is not finite")
    return result


def onion_path_metrics(
    inversion_losses: Sequence[float],
    *,
    shell_weights: Optional[Sequence[float]] = None,
    epsilon: float = DEFAULT_EPSILON,
) -> Dict[str, object]:
    """Measure prefix reductions and later-shell reversals for one path."""

    losses = _finite_vector(inversion_losses, "inversion_losses", minimum_length=2)
    if np.any(losses < 0.0) or np.any(losses > 1.0):
        raise CRSVValidationError("inversion_losses must lie in [0, 1]")
    if not math.isfinite(epsilon) or epsilon <= 0.0:
        raise CRSVValidationError("epsilon must be finite and positive")
    layer_count = losses.size - 1
    if shell_weights is None:
        weights = np.ones(layer_count, dtype=np.float64)
    else:
        weights = _finite_vector(shell_weights, "shell_weights", minimum_length=layer_count)
        if weights.size != layer_count:
            raise CRSVValidationError("shell_weights must match the number of path layers")
        if np.any(weights < 0.0):
            raise CRSVValidationError("shell_weights must be non-negative")
    differentials = losses[:-1] - losses[1:]
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            contraction = np.log(losses[:-1] + epsilon) - np.log(losses[1:] + epsilon)
        except FloatingPointError as exc:
            raise CRSVValidationError("inversion contraction is not numerically identifiable") from exc
    negative_contraction = np.maximum(-contraction, 0.0)
    absolute_differential = np.abs(differentials)
    reversal_mass = float(np.sum(np.maximum(-differentials, 0.0)))
    gross_movement = float(np.sum(absolute_differential))
    result = {
        "baseline_inversion_loss": float(losses[0]),
        "final_inversion_loss": float(losses[-1]),
        "endpoint_benefit": float(losses[0] - losses[-1]),
        "marginal_differentials": [float(value) for value in differentials],
        "log_contraction_differentials": [float(value) for value in contraction],
        "endpoint_log_contraction": float(np.sum(contraction)),
        "reversal_burden": float(np.sum(weights * negative_contraction)),
        "reversal_mass": reversal_mass,
        "reversal_fraction": 0.0 if gross_movement == 0.0 else reversal_mass / gross_movement,
        "telescoping_error": float(abs(np.sum(differentials) - (losses[0] - losses[-1]))),
    }
    return _assert_finite_tree(result, "onion_path_metrics")


def _normalize_subset(raw_subset: Iterable[str]) -> FrozenSet[str]:
    if isinstance(raw_subset, (str, bytes)):
        raise CRSVValidationError("a replacement subset must be an iterable of layer IDs")
    items = tuple(raw_subset)
    subset = frozenset(items)
    if len(subset) != len(items):
        raise CRSVValidationError("replacement subsets must not contain duplicate layer IDs")
    if any(not isinstance(item, str) or not item for item in subset):
        raise CRSVValidationError("replacement subset IDs must be non-empty strings")
    return subset


def _normalized_benefits(
    benefits: Mapping[Iterable[str], float],
) -> Dict[FrozenSet[str], float]:
    if not isinstance(benefits, Mapping):
        raise CRSVValidationError("benefits must be a mapping")
    normalized: Dict[FrozenSet[str], float] = {}
    for raw_subset, raw_value in benefits.items():
        subset = _normalize_subset(raw_subset)
        if subset in normalized:
            raise CRSVValidationError("benefits contains a duplicate normalized subset")
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise CRSVValidationError("benefit values must be numeric")
        value = float(raw_value)
        if not math.isfinite(value):
            raise CRSVValidationError("benefit values must be finite")
        normalized[subset] = value
    return normalized


def _all_subsets(items: FrozenSet[str]) -> Iterable[FrozenSet[str]]:
    ordered = sorted(items)
    for size in range(len(ordered) + 1):
        for combination in itertools.combinations(ordered, size):
            yield frozenset(combination)


def mobius_interaction(
    benefits: Mapping[Iterable[str], float],
    subset: Iterable[str],
) -> float:
    """Return the finite Möbius interaction coefficient for ``subset``."""

    target = _normalize_subset(subset)
    if not target:
        raise CRSVValidationError("Möbius interaction requires a non-empty subset")
    normalized = _normalized_benefits(benefits)
    missing = [candidate for candidate in _all_subsets(target) if candidate not in normalized]
    if missing:
        raise CRSVValidationError("benefits is missing subsets required by the Möbius transform")
    terms = []
    for candidate in _all_subsets(target):
        sign = -1.0 if (len(target) - len(candidate)) % 2 else 1.0
        terms.append(sign * normalized[candidate])
    return _stable_signed_sum(terms, "Möbius interaction")


def additivity_gap(
    benefits: Mapping[Iterable[str], float],
    subset: Iterable[str],
) -> float:
    """Return combined benefit minus the sum of isolated benefits."""

    target = _normalize_subset(subset)
    if len(target) < 2:
        raise CRSVValidationError("additivity_gap requires at least two layers")
    normalized = _normalized_benefits(benefits)
    if frozenset() not in normalized:
        raise CRSVValidationError("benefits must include an explicit empty baseline")
    empty = normalized[frozenset()]
    if target not in normalized:
        raise CRSVValidationError("benefits is missing the target subset")
    singleton_values = []
    for item in target:
        singleton = frozenset((item,))
        if singleton not in normalized:
            raise CRSVValidationError("benefits is missing a singleton subset")
        singleton_values.append(normalized[singleton])
    terms = [normalized[target]]
    terms.extend(-value for value in singleton_values)
    terms.extend(empty for _ in range(len(target) - 1))
    return _stable_signed_sum(terms, "additivity gap")


def classify_interaction(value: float, *, tolerance: float = TIE_TOLERANCE) -> str:
    """Classify a signed benefit interaction."""

    if not math.isfinite(value):
        raise CRSVValidationError("interaction value must be finite")
    if not math.isfinite(tolerance) or tolerance < 0.0:
        raise CRSVValidationError("tolerance must be finite and non-negative")
    if value > tolerance:
        return "SUPERADDITIVE"
    if value < -tolerance:
        return "SUBADDITIVE"
    return "ADDITIVE"


def order_spread(
    losses_by_order: Mapping[Tuple[str, ...], float],
    *,
    admissible_order_count: Optional[int] = None,
) -> Dict[str, object]:
    """Return worst-minus-best inversion loss over complete admissible orders."""

    if not isinstance(losses_by_order, Mapping) or len(losses_by_order) < 2:
        raise CRSVValidationError("losses_by_order must contain at least two orders")
    normalized: Dict[Tuple[str, ...], float] = {}
    expected_layers: Optional[FrozenSet[str]] = None
    for raw_order, raw_loss in losses_by_order.items():
        if not isinstance(raw_order, tuple) or not raw_order:
            raise CRSVValidationError("order keys must be non-empty tuples")
        if len(set(raw_order)) != len(raw_order):
            raise CRSVValidationError("each order must contain unique layer IDs")
        if any(not isinstance(item, str) or not item for item in raw_order):
            raise CRSVValidationError("order layer IDs must be non-empty strings")
        layer_set = frozenset(raw_order)
        if expected_layers is None:
            expected_layers = layer_set
        elif layer_set != expected_layers:
            raise CRSVValidationError("all order keys must permute the same layer set")
        if isinstance(raw_loss, bool) or not isinstance(raw_loss, (int, float)):
            raise CRSVValidationError("order losses must be numeric")
        loss = float(raw_loss)
        if not math.isfinite(loss) or loss < 0.0 or loss > 1.0:
            raise CRSVValidationError("order losses must be finite and lie in [0, 1]")
        normalized[raw_order] = loss
    unconstrained_count = math.factorial(len(expected_layers))
    if admissible_order_count is None:
        declared_admissible_count = unconstrained_count
    else:
        if (
            isinstance(admissible_order_count, bool)
            or not isinstance(admissible_order_count, int)
            or admissible_order_count < len(normalized)
            or admissible_order_count > unconstrained_count
        ):
            raise CRSVValidationError(
                "admissible_order_count must cover tested orders and not exceed the factorial universe"
            )
        declared_admissible_count = admissible_order_count
    best_order = min(normalized, key=lambda order: (normalized[order], order))
    worst_order = max(normalized, key=lambda order: (normalized[order], order))
    return {
        "best_order": list(best_order),
        "best_loss": normalized[best_order],
        "worst_order": list(worst_order),
        "worst_loss": normalized[worst_order],
        "spread": float(normalized[worst_order] - normalized[best_order]),
        "tested_order_count": len(normalized),
        "declared_admissible_order_count": declared_admissible_count,
        "admissible_order_coverage": (len(normalized) / declared_admissible_count),
    }


def commutator_distance(
    left_operator: Sequence[Sequence[float]],
    right_operator: Sequence[Sequence[float]],
    vector: Optional[Sequence[float]] = None,
) -> float:
    """Measure noncommutativity in operator norm or at one frozen vector."""

    left = _square_matrix(left_operator, "left_operator")
    right = _square_matrix(right_operator, "right_operator")
    if left.shape != right.shape:
        raise CRSVValidationError("operators must have identical shapes")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            commutator = left @ right - right @ left
        except FloatingPointError as exc:
            raise CRSVValidationError("commutator is not numerically finite") from exc
    if vector is None:
        return _finite_norm(commutator, "commutator", order=2)
    point = _finite_vector(vector, "vector", minimum_length=left.shape[0])
    if point.size != left.shape[0]:
        raise CRSVValidationError("vector dimension must match the operators")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            displaced = commutator @ point
        except FloatingPointError as exc:
            raise CRSVValidationError("commutator displacement is not numerically finite") from exc
    return _finite_norm(displaced, "commutator displacement", order=None)


def _orthonormal_basis(raw_basis: Sequence[Sequence[float]], label: str) -> np.ndarray:
    basis = _finite_matrix(raw_basis, label)
    if basis.shape[0] < 1 or basis.shape[1] < 1:
        raise CRSVValidationError("{} must be non-empty".format(label))
    try:
        left, singular_values, _ = np.linalg.svd(basis, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise CRSVValidationError("{} SVD did not converge".format(label)) from exc
    if singular_values.size == 0:
        raise CRSVValidationError("{} has no singular values".format(label))
    tolerance = max(basis.shape) * np.finfo(np.float64).eps * singular_values[0]
    rank = int(np.sum(singular_values > tolerance))
    if rank == 0:
        raise CRSVValidationError("{} must span a non-zero subspace".format(label))
    return left[:, :rank]


def subspace_attunement(
    left_basis: Sequence[Sequence[float]],
    right_basis: Sequence[Sequence[float]],
) -> Dict[str, object]:
    """Return principal-angle coherence between two column subspaces."""

    left = _orthonormal_basis(left_basis, "left_basis")
    right = _orthonormal_basis(right_basis, "right_basis")
    if left.shape[0] != right.shape[0]:
        raise CRSVValidationError("subspace ambient dimensions must match")
    try:
        singular_values = np.linalg.svd(left.T @ right, compute_uv=False)
    except np.linalg.LinAlgError as exc:
        raise CRSVValidationError("subspace SVD did not converge") from exc
    singular_values = np.clip(singular_values, 0.0, 1.0)
    angles = np.arccos(singular_values)
    result = {
        "maximum_coherence": float(np.max(singular_values)),
        "minimum_principal_angle_radians": float(np.min(angles)),
        "principal_cosines": [float(value) for value in singular_values],
        "normalized_overlap_energy": float(np.sum(np.square(singular_values)) / min(left.shape[1], right.shape[1])),
    }
    return _assert_finite_tree(result, "subspace_attunement")


def spectral_diagnostics(
    operator: Sequence[Sequence[float]],
    *,
    horizon: int,
    unit_boundary_tolerance: float = 1e-8,
) -> Dict[str, object]:
    """Diagnose one explicitly declared discrete-time transition operator.

    Horizon zero is included with gain one.  This makes
    ``peak_transient_gain > 1`` an actual amplification relative to the
    initial state rather than merely the largest post-transition norm.
    """

    matrix = _square_matrix(operator, "operator")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise CRSVValidationError("horizon must be a positive integer")
    if not math.isfinite(unit_boundary_tolerance) or unit_boundary_tolerance <= 0.0 or unit_boundary_tolerance >= 1.0:
        raise CRSVValidationError("unit_boundary_tolerance must be finite and lie in (0, 1)")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            eigenvalues = np.linalg.eigvals(matrix)
            spectral_radius = float(np.max(np.abs(eigenvalues)))
            normality_residual = matrix.conj().T @ matrix - matrix @ matrix.conj().T
            denominator = float(np.linalg.norm(matrix, ord="fro") ** 2)
            departure = (
                0.0 if denominator == 0.0 else float(np.linalg.norm(normality_residual, ord="fro") / denominator)
            )
            gains = [1.0]
            current = np.eye(matrix.shape[0], dtype=np.float64)
            for _ in range(1, horizon + 1):
                current = current @ matrix
                gains.append(float(np.linalg.norm(current, ord=2)))
        except (FloatingPointError, np.linalg.LinAlgError) as exc:
            raise CRSVValidationError("operator diagnostics are not numerically finite") from exc
    peak_index = int(np.argmax(gains))
    stability_margin = 1.0 - spectral_radius
    near_unit_boundary = abs(stability_margin) <= unit_boundary_tolerance
    result = {
        "spectral_radius": spectral_radius,
        "stability_margin": stability_margin,
        "unit_boundary_tolerance": unit_boundary_tolerance,
        "near_unit_boundary": near_unit_boundary,
        "asymptotically_contractive": (spectral_radius < 1.0 - unit_boundary_tolerance),
        "departure_from_normality": departure,
        "finite_horizon_gains": gains,
        "peak_transient_gain": gains[peak_index],
        "peak_horizon": peak_index,
        "strict_transient_amplification": (gains[peak_index] > 1.0 + TIE_TOLERANCE),
    }
    return _assert_finite_tree(result, "spectral_diagnostics")


def _dominant_left_subspace(
    operator: np.ndarray,
    *,
    rank: int,
    label: str,
) -> np.ndarray:
    try:
        left, singular_values, _ = np.linalg.svd(operator, full_matrices=False)
    except np.linalg.LinAlgError as exc:
        raise CRSVValidationError("{} SVD did not converge".format(label)) from exc
    if singular_values.size == 0:
        raise CRSVValidationError("{} has no singular values".format(label))
    tolerance = max(operator.shape) * np.finfo(np.float64).eps * singular_values[0]
    numerical_rank = int(np.sum(singular_values > tolerance))
    if numerical_rank < rank:
        raise CRSVValidationError(
            "{} has numerical rank {}, below requested subspace rank {}".format(
                label,
                numerical_rank,
                rank,
            )
        )
    return left[:, :rank]


def additive_operator_diagnostics(
    base_operator: Sequence[Sequence[float]],
    additives: Mapping[str, Sequence[Sequence[float]]],
    *,
    coordinate_space_id: str,
    horizon: int,
    weights: Optional[Mapping[str, float]] = None,
    subspace_rank: int = 1,
) -> Dict[str, object]:
    """Diagnose ``base + sum(weight[k] * additive[k])``.

    All matrices must already be expressed in the one declared coordinate
    space.  The triangle slack is reported only as geometric dispersion; true
    destructive cancellation is the negative portion of signed interference.
    Pairwise subspace, alignment, and commutator quantities remain separate so
    unlike mechanisms cannot silently compensate for one another.
    """

    if not isinstance(coordinate_space_id, str) or not coordinate_space_id.strip():
        raise CRSVValidationError("coordinate_space_id must be a non-empty string")
    if not isinstance(additives, Mapping) or len(additives) < 2:
        raise CRSVValidationError("additives must contain at least two named operators")
    if isinstance(subspace_rank, bool) or not isinstance(subspace_rank, int):
        raise CRSVValidationError("subspace_rank must be an integer")
    if subspace_rank < 1:
        raise CRSVValidationError("subspace_rank must be positive")
    if isinstance(horizon, bool) or not isinstance(horizon, int) or horizon < 1:
        raise CRSVValidationError("horizon must be a positive integer")

    names = tuple(sorted(additives))
    if any(not isinstance(name, str) or not name for name in names):
        raise CRSVValidationError("operator IDs must be non-empty strings")
    if weights is None:
        normalized_weights = {name: 1.0 for name in names}
    else:
        if not isinstance(weights, Mapping) or set(weights) != set(names):
            raise CRSVValidationError("weights must name every operator exactly once")
        normalized_weights = {}
        for name in names:
            raw_weight = weights[name]
            if isinstance(raw_weight, bool) or not isinstance(raw_weight, (int, float)):
                raise CRSVValidationError("operator weights must be numeric")
            weight = float(raw_weight)
            if not math.isfinite(weight) or weight == 0.0:
                raise CRSVValidationError("operator weights must be finite and non-zero")
            normalized_weights[name] = weight

    base = _square_matrix(base_operator, "base_operator")
    matrices: Dict[str, np.ndarray] = {}
    expected_shape: Tuple[int, int] = base.shape
    if subspace_rank > base.shape[0]:
        raise CRSVValidationError("subspace_rank cannot exceed operator dimension")
    for name in names:
        matrix = _square_matrix(additives[name], "additives[{}]".format(name))
        if matrix.shape != expected_shape:
            raise CRSVValidationError("all additive operators must have identical shapes")
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                matrices[name] = normalized_weights[name] * matrix
            except FloatingPointError as exc:
                raise CRSVValidationError("weighted additive operator is not numerically finite") from exc

    aggregate = base.copy()
    component_frobenius_norms: Dict[str, float] = {}
    component_subspaces: Dict[str, np.ndarray] = {}
    for name in names:
        matrix = matrices[name]
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                aggregate += matrix
            except FloatingPointError as exc:
                raise CRSVValidationError("additive aggregate is not numerically finite") from exc
        component_frobenius_norms[name] = _finite_norm(
            matrix,
            "additives[{}]".format(name),
            order="fro",
        )
        component_subspaces[name] = _dominant_left_subspace(
            matrix,
            rank=subspace_rank,
            label="operators[{}]".format(name),
        )

    additive_sum = aggregate - base
    norm_sum = float(sum(component_frobenius_norms.values()))
    additive_sum_norm = _finite_norm(
        additive_sum,
        "weighted additive sum",
        order="fro",
    )
    triangle_alignment_ratio = additive_sum_norm / norm_sum
    triangle_slack = min(1.0, max(0.0, 1.0 - triangle_alignment_ratio))
    individual_energy = float(sum(value * value for value in component_frobenius_norms.values()))
    additive_sum_energy = additive_sum_norm * additive_sum_norm
    if not math.isfinite(individual_energy) or individual_energy <= 0.0 or not math.isfinite(additive_sum_energy):
        raise CRSVValidationError("additive interference energy is not numerically identifiable")
    signed_interference = additive_sum_energy - individual_energy
    pairwise = []
    for left_name, right_name in itertools.combinations(names, 2):
        left = matrices[left_name]
        right = matrices[right_name]
        left_norm = component_frobenius_norms[left_name]
        right_norm = component_frobenius_norms[right_name]
        with np.errstate(over="raise", invalid="raise", divide="raise"):
            try:
                signed_alignment = float(np.sum((left / left_norm) * (right / right_norm)))
            except FloatingPointError as exc:
                raise CRSVValidationError("additive alignment is not numerically finite") from exc
        attunement = subspace_attunement(
            component_subspaces[left_name],
            component_subspaces[right_name],
        )
        raw_commutator = commutator_distance(left, right)
        operator_norm_product = float(
            _finite_norm(left, "left additive operator", order=2)
            * _finite_norm(right, "right additive operator", order=2)
        )
        if not math.isfinite(operator_norm_product) or operator_norm_product <= 0.0:
            raise CRSVValidationError("normalized commutator denominator is not identifiable")
        pairwise.append(
            {
                "left": left_name,
                "right": right_name,
                "signed_frobenius_alignment": signed_alignment,
                "dominant_subspace_coherence": attunement["maximum_coherence"],
                "principal_cosines": attunement["principal_cosines"],
                "normalized_overlap_energy": attunement["normalized_overlap_energy"],
                "commutator_distance": raw_commutator,
                "normalized_commutator_distance": (raw_commutator / operator_norm_product),
            }
        )

    aggregate_spectral = spectral_diagnostics(aggregate, horizon=horizon)
    leave_one_out = []
    for name in names:
        diagnostics = spectral_diagnostics(aggregate - matrices[name], horizon=horizon)
        leave_one_out.append(
            {
                "omitted": name,
                "spectral_radius": diagnostics["spectral_radius"],
                "peak_transient_gain": diagnostics["peak_transient_gain"],
                "peak_gain_change": (diagnostics["peak_transient_gain"] - aggregate_spectral["peak_transient_gain"]),
            }
        )
    result = {
        "coordinate_space_id": coordinate_space_id,
        "operator_ids": list(names),
        "weights": normalized_weights,
        "base_frobenius_norm": _finite_norm(
            base,
            "base_operator",
            order="fro",
        ),
        "component_frobenius_norms": component_frobenius_norms,
        "aggregate_transition_frobenius_norm": _finite_norm(
            aggregate,
            "aggregate transition",
            order="fro",
        ),
        "additive_sum_frobenius_norm": additive_sum_norm,
        "triangle_alignment_ratio": triangle_alignment_ratio,
        "triangle_slack": triangle_slack,
        "signed_interference_energy": signed_interference,
        "normalized_signed_interference": (signed_interference / individual_energy),
        "net_destructive_interference_scalar": (max(0.0, -signed_interference) / individual_energy),
        "pairwise": pairwise,
        "aggregate_transition_spectral": aggregate_spectral,
        "leave_one_out_perturbations": leave_one_out,
    }
    return _assert_finite_tree(result, "additive_operator_diagnostics")


def ordered_prefix_operator_diagnostics(
    ordered_operators: Sequence[Tuple[str, Sequence[Sequence[float]]]],
    *,
    coordinate_space_id: str,
) -> Dict[str, object]:
    """Measure gains of the exact ordered prefix products.

    For column-vector semantics and order ``(T1, T2, ...)``, prefix ``k`` is
    ``Tk @ ... @ T2 @ T1``.  This is distinct from repeatedly applying one
    state-transition operator and from adding residual operators.
    """

    if not isinstance(coordinate_space_id, str) or not coordinate_space_id.strip():
        raise CRSVValidationError("coordinate_space_id must be a non-empty string")
    if (
        isinstance(ordered_operators, (str, bytes))
        or not isinstance(ordered_operators, Sequence)
        or not ordered_operators
    ):
        raise CRSVValidationError("ordered_operators must be a non-empty sequence")
    names = []
    matrices = []
    expected_shape: Optional[Tuple[int, int]] = None
    for index, raw_item in enumerate(ordered_operators):
        if not isinstance(raw_item, tuple) or len(raw_item) != 2:
            raise CRSVValidationError("ordered_operators[{}] must be an (ID, operator) tuple".format(index))
        name, raw_matrix = raw_item
        if not isinstance(name, str) or not name:
            raise CRSVValidationError("ordered operator IDs must be non-empty strings")
        if name in names:
            raise CRSVValidationError("ordered operator IDs must be unique")
        matrix = _square_matrix(
            raw_matrix,
            "ordered_operators[{}]".format(index),
        )
        if expected_shape is None:
            expected_shape = matrix.shape
        elif matrix.shape != expected_shape:
            raise CRSVValidationError("all ordered operators must have identical shapes")
        names.append(name)
        matrices.append(matrix)

    current = np.eye(expected_shape[0], dtype=np.float64)
    prefixes = [
        {
            "prefix_length": 0,
            "last_operator_id": None,
            "cumulative_gain": 1.0,
            "cumulative_spectral_radius": 1.0,
        }
    ]
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            for index, (name, matrix) in enumerate(zip(names, matrices), start=1):
                current = matrix @ current
                prefixes.append(
                    {
                        "prefix_length": index,
                        "last_operator_id": name,
                        "cumulative_gain": float(np.linalg.norm(current, ord=2)),
                        "cumulative_spectral_radius": float(np.max(np.abs(np.linalg.eigvals(current)))),
                    }
                )
        except (FloatingPointError, np.linalg.LinAlgError) as exc:
            raise CRSVValidationError("ordered prefix diagnostics are not numerically finite") from exc
    peak = max(prefixes, key=lambda item: item["cumulative_gain"])
    result = {
        "coordinate_space_id": coordinate_space_id,
        "operator_order": names,
        "prefixes": prefixes,
        "peak_prefix_gain": peak["cumulative_gain"],
        "peak_prefix_length": peak["prefix_length"],
    }
    return _assert_finite_tree(result, "ordered_prefix_operator_diagnostics")


def useful_signal_atrophy(
    native_activations: Sequence[Sequence[float]],
    candidate_activations: Sequence[Sequence[float]],
    signal_basis: Sequence[Sequence[float]],
    *,
    minimum_native_signal_energy: float = DEFAULT_EPSILON,
) -> Dict[str, float]:
    """Measure energy change in one frozen useful-signal subspace.

    The ratio is undefined when native signal energy is below the
    preregistered identification floor, so that case fails closed rather than
    adding a stabilizer that could manufacture atrophy.
    """

    native = _finite_matrix(native_activations, "native_activations")
    candidate = _finite_matrix(candidate_activations, "candidate_activations")
    if native.shape != candidate.shape:
        raise CRSVValidationError("native and candidate activations must have identical shapes")
    basis = _orthonormal_basis(signal_basis, "signal_basis")
    if basis.shape[0] != native.shape[1]:
        raise CRSVValidationError("signal_basis ambient dimension must match activation width")
    if not math.isfinite(minimum_native_signal_energy) or minimum_native_signal_energy <= 0.0:
        raise CRSVValidationError("minimum_native_signal_energy must be finite and positive")
    native_energy = (
        _finite_norm(
            native @ basis,
            "native useful-signal projection",
            order="fro",
        )
        ** 2
    )
    candidate_energy = (
        _finite_norm(
            candidate @ basis,
            "candidate useful-signal projection",
            order="fro",
        )
        ** 2
    )
    if native_energy <= minimum_native_signal_energy:
        raise CRSVValidationError("native useful-signal energy is below the identification floor")
    retention = candidate_energy / native_energy
    result = {
        "native_signal_energy": native_energy,
        "candidate_signal_energy": candidate_energy,
        "signal_retention": retention,
        "signed_signal_change": candidate_energy - native_energy,
        "signal_atrophy": max(0.0, 1.0 - retention),
    }
    return _assert_finite_tree(result, "useful_signal_atrophy")


def crsv_interaction_energy(scores: Sequence[Sequence[Sequence[float]]]) -> Dict[str, object]:
    """Estimate balanced serving x replacement interaction across groups.

    ``scores`` has shape ``(independence_groups, serving_subdomains,
    replacement_subdomains)``.  Every group is double-centered first.  The
    cross-group U-statistic estimates squared population interaction without
    the positive sampling-noise bias of squaring the sample interaction mean.
    It can be negative in finite samples and is not itself a significance test.
    """

    cube = _finite_array(scores, "scores", ndim=3)
    groups, serving_count, replacement_count = cube.shape
    if groups < 2 or serving_count < 2 or replacement_count < 2:
        raise CRSVValidationError("scores requires at least 2 groups, 2 serving subdomains, and 2 replacements")
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            group_serving_means = np.mean(cube, axis=2, keepdims=True)
            group_replacement_means = np.mean(cube, axis=1, keepdims=True)
            group_grand_means = np.mean(cube, axis=(1, 2), keepdims=True)
            group_residuals = cube - group_serving_means - group_replacement_means + group_grand_means
            residual = np.mean(group_residuals, axis=0)
            squared_mean_energy = float(np.mean(np.square(residual)))
            residual_sum = np.sum(group_residuals, axis=0)
            cross_group_inner_sum = float(np.sum(np.square(residual_sum)) - np.sum(np.square(group_residuals)))
            unbiased_energy = cross_group_inner_sum / (groups * (groups - 1) * serving_count * replacement_count)
            group_interaction_dispersion = float(np.mean(np.var(group_residuals, axis=0, ddof=1)))
            cell_means = np.mean(cube, axis=0)
        except FloatingPointError as exc:
            raise CRSVValidationError("CRSV interaction diagnostics are not numerically finite") from exc
    result = {
        "interaction_energy": unbiased_energy,
        "unbiased_cross_group_interaction_energy": unbiased_energy,
        "squared_sample_mean_interaction_energy": squared_mean_energy,
        "group_interaction_dispersion": group_interaction_dispersion,
        "cell_means": cell_means.tolist(),
        "interaction_residuals": residual.tolist(),
        "group_interaction_residuals": group_residuals.tolist(),
    }
    return _assert_finite_tree(result, "crsv_interaction_energy")


def _home_effect(
    cell_means: np.ndarray,
    serving_labels: Sequence[str],
    replacement_labels: Sequence[str],
) -> float:
    contributions = []
    for serving_index, serving_label in enumerate(serving_labels):
        home_indices = [
            index for index, replacement_label in enumerate(replacement_labels) if replacement_label == serving_label
        ]
        if len(home_indices) != 1:
            raise CRSVValidationError("each serving label must have exactly one matching replacement label")
        home_index = home_indices[0]
        cross_indices = [index for index in range(len(replacement_labels)) if index != home_index]
        contributions.append(
            float(cell_means[serving_index, home_index] - np.mean(cell_means[serving_index, cross_indices]))
        )
    return float(np.mean(contributions))


def _label_permutations_by_block(
    labels: Sequence[str],
    blocks: Sequence[str],
) -> Iterable[Tuple[str, ...]]:
    block_to_indices: Dict[str, List[int]] = {}
    for index, block in enumerate(blocks):
        block_to_indices.setdefault(block, []).append(index)
    block_names = sorted(block_to_indices)
    block_permutations = []
    for block in block_names:
        indices = block_to_indices[block]
        block_permutations.append(list(itertools.permutations([labels[index] for index in indices])))
    for choices in itertools.product(*block_permutations):
        candidate = list(labels)
        for block_index, block in enumerate(block_names):
            indices = block_to_indices[block]
            for local_index, global_index in enumerate(indices):
                candidate[global_index] = choices[block_index][local_index]
        yield tuple(candidate)


def home_correspondence_permutation_test(
    scores: Sequence[Sequence[Sequence[float]]],
    serving_labels: Sequence[str],
    replacement_labels: Sequence[str],
    *,
    exchangeability_blocks: Optional[Sequence[str]] = None,
    max_permutations: int = DEFAULT_MAX_PERMUTATIONS,
    seed: int = DEFAULT_PERMUTATION_SEED,
) -> Dict[str, object]:
    """Test semantic home correspondence while preserving replacement profiles.

    Replacement *labels*, not outcomes, are permuted.  Labels move only within
    design-justified exchangeability blocks, which are mandatory.  This is
    finite-label randomization inference: it does not establish population
    generalization to unseen replacement identities or subdomains.  The
    statistic is home quality minus mean cross quality.
    """

    cube = _finite_array(scores, "scores", ndim=3)
    if cube.shape[0] < 2:
        raise CRSVValidationError("scores requires at least two independence groups")
    serving = tuple(serving_labels)
    replacement = tuple(replacement_labels)
    if len(serving) != cube.shape[1] or len(replacement) != cube.shape[2]:
        raise CRSVValidationError("label counts must match the serving and replacement axes")
    if any(not isinstance(label, str) or not label for label in serving + replacement):
        raise CRSVValidationError("subdomain labels must be non-empty strings")
    if len(set(serving)) != len(serving) or len(set(replacement)) != len(replacement):
        raise CRSVValidationError("subdomain labels must be unique on each axis")
    if set(serving) != set(replacement):
        raise CRSVValidationError("serving and replacement labels must name the same subdomains")
    if isinstance(max_permutations, bool) or not isinstance(max_permutations, int) or max_permutations < 2:
        raise CRSVValidationError("max_permutations must be an integer of at least 2")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise CRSVValidationError("seed must be a non-negative integer")
    if exchangeability_blocks is None:
        raise CRSVValidationError("exchangeability_blocks must be explicitly design-justified")
    blocks = tuple(exchangeability_blocks)
    if len(blocks) != len(replacement):
        raise CRSVValidationError("exchangeability_blocks must match replacement labels")
    if any(not isinstance(block, str) or not block for block in blocks):
        raise CRSVValidationError("exchangeability block IDs must be non-empty strings")

    total_permutations = 1
    for block in set(blocks):
        total_permutations *= math.factorial(sum(value == block for value in blocks))
    if total_permutations < 2:
        raise CRSVValidationError("exchangeability blocks admit no non-identity permutation")

    cell_means = np.mean(cube, axis=0)
    observed = _home_effect(cell_means, serving, replacement)
    exact = total_permutations <= max_permutations
    if exact:
        assignments = list(_label_permutations_by_block(replacement, blocks))
    else:
        rng = np.random.default_rng(seed)
        assignments = []
        block_to_indices: Dict[str, List[int]] = {}
        for index, block in enumerate(blocks):
            block_to_indices.setdefault(block, []).append(index)
        for _ in range(max_permutations):
            candidate = list(replacement)
            for indices in block_to_indices.values():
                permuted = rng.permutation([replacement[index] for index in indices])
                for local_index, global_index in enumerate(indices):
                    candidate[global_index] = str(permuted[local_index])
            assignments.append(tuple(candidate))
    null_statistics = [_home_effect(cell_means, serving, assignment) for assignment in assignments]
    exceedances = sum(value >= observed - TIE_TOLERANCE for value in null_statistics)
    if exact:
        p_value = exceedances / len(null_statistics)
        monte_carlo_standard_error = None
    else:
        p_value = (exceedances + 1.0) / (len(null_statistics) + 1.0)
        monte_carlo_standard_error = math.sqrt(p_value * (1.0 - p_value) / len(null_statistics))
    result = {
        "observed_home_effect": observed,
        "one_sided_p_value": float(p_value),
        "exact": exact,
        "sampling_mode": "EXACT" if exact else "MONTE_CARLO",
        "permutation_count": len(null_statistics),
        "total_permutation_universe": total_permutations,
        "seed": None if exact else seed,
        "monte_carlo_standard_error": monte_carlo_standard_error,
        "exchangeability_block_count": len(set(blocks)),
        "null_min": float(min(null_statistics)),
        "null_max": float(max(null_statistics)),
    }
    return _assert_finite_tree(result, "home_correspondence_permutation_test")


def _finite_array(raw: object, label: str, *, ndim: int) -> np.ndarray:
    try:
        array = np.asarray(raw, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise CRSVValidationError("{} must be numeric".format(label)) from exc
    if array.ndim != ndim:
        raise CRSVValidationError("{} must be {}-dimensional".format(label, ndim))
    if array.size == 0 or not np.all(np.isfinite(array)):
        raise CRSVValidationError("{} must be non-empty and finite".format(label))
    return array


def _finite_norm(
    array: np.ndarray,
    label: str,
    *,
    order: object,
) -> float:
    with np.errstate(over="raise", invalid="raise", divide="raise"):
        try:
            value = float(np.linalg.norm(array, ord=order))
        except (FloatingPointError, np.linalg.LinAlgError) as exc:
            raise CRSVValidationError("{} norm is not numerically finite".format(label)) from exc
    if not math.isfinite(value):
        raise CRSVValidationError("{} norm is not finite".format(label))
    return value


def _stable_signed_sum(values: Sequence[float], label: str) -> float:
    if not values:
        return 0.0
    scale = max(abs(value) for value in values)
    if scale == 0.0:
        return 0.0
    try:
        scaled_sum = math.fsum(value / scale for value in values)
        result = scaled_sum * scale
    except (OverflowError, ValueError) as exc:
        raise CRSVValidationError("{} is not numerically finite".format(label)) from exc
    if not math.isfinite(result):
        raise CRSVValidationError("{} is not numerically finite".format(label))
    return float(result)


def _assert_finite_tree(value: _T, label: str) -> _T:
    if isinstance(value, Mapping):
        for key, child in value.items():
            _assert_finite_tree(child, "{}.{}".format(label, key))
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_finite_tree(child, "{}[{}]".format(label, index))
    elif isinstance(value, np.ndarray):
        if not np.all(np.isfinite(value)):
            raise CRSVValidationError("{} contains non-finite output".format(label))
    elif isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            raise CRSVValidationError("{} is non-finite".format(label))
    return value


def _finite_matrix(raw: object, label: str) -> np.ndarray:
    return _finite_array(raw, label, ndim=2)


def _square_matrix(raw: object, label: str) -> np.ndarray:
    matrix = _finite_matrix(raw, label)
    if matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1:
        raise CRSVValidationError("{} must be a non-empty square matrix".format(label))
    return matrix


def _finite_vector(raw: object, label: str, *, minimum_length: int) -> np.ndarray:
    array = _finite_array(raw, label, ndim=1)
    if array.size < minimum_length:
        raise CRSVValidationError("{} must contain at least {} values".format(label, minimum_length))
    return array
