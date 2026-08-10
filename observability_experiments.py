"""Bounded, dependency-light observability experiments for RB-14.

This module is deliberately separate from the production retrieval path and
from the prime-ring experiment contracts.  It implements the first executable
RB-14 gates:

* ``PRW-OBS1``: an invariant-span/off-span identifiability experiment;
* ``PRW-COA1``: a lifted interaction-space/top-k observability experiment.

The experiments use streaming Gram matrices rather than retaining a sample
matrix.  Their outputs are deterministic for a seed and can be written as
content-addressed JSON envelopes.  A result is evidence about the frozen
synthetic fixture only; it is not a production or novelty claim.

Only NumPy and the Python standard library are required.  The module is
compatible with Python 3.9.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import math
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


OBSERVABILITY_PROTOCOL_ID = "CHELATEDAI-RB14-OBSERVABILITY-v1"
OBS1_STAGE_ID = "PRW-OBS1"
COA1_STAGE_ID = "PRW-COA1"
CTX1_STAGE_ID = "PRW-CTX1"
TRANSPORT_STAGE_ID = "PRW-SPU1-TRANSPORT"
EK7_COALITION_STAGE_ID = "PRW-EK7-COALITION"
DEFAULT_MAX_DIMENSION = 64
DEFAULT_MAX_PROBES = 20_000
DEFAULT_MAX_FEATURES = 2_000
DEFAULT_MAX_ESTIMATED_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_WORK_UNITS = 5_000_000
DEFAULT_MAX_SECONDS = 10.0
DEFAULT_MAX_OUTPUT_BYTES = 8 * 1024 * 1024
_RANK_RELATIVE_TOLERANCE = 1e-10


class ObservabilityValidationError(ValueError):
    """Raised when a fixture or experiment contract is malformed."""


class ObservabilityResourceError(RuntimeError):
    """Raised before or during work when an immutable resource limit is crossed."""


def _plain_int(value: object, name: str, minimum: int, maximum: int) -> int:
    if type(value) is not int:
        raise ObservabilityValidationError(f"{name} must be a plain integer")
    if value < minimum or value > maximum:
        raise ObservabilityValidationError(f"{name} must be in [{minimum}, {maximum}]")
    return value


def _plain_float(value: object, name: str, minimum: float, maximum: float) -> float:
    if type(value) not in (int, float) or isinstance(value, bool):
        raise ObservabilityValidationError(f"{name} must be a plain real number")
    result = float(value)
    if not math.isfinite(result) or result < minimum or result > maximum:
        raise ObservabilityValidationError(f"{name} must be finite and in [{minimum}, {maximum}]")
    return result


@dataclass(frozen=True)
class ObservabilityBudget:
    """Caller-lowerable limits under the immutable RB-14 envelope."""

    max_dimension: int = DEFAULT_MAX_DIMENSION
    max_probes: int = DEFAULT_MAX_PROBES
    max_features: int = DEFAULT_MAX_FEATURES
    max_estimated_bytes: int = DEFAULT_MAX_ESTIMATED_BYTES
    max_work_units: int = DEFAULT_MAX_WORK_UNITS
    max_seconds: float = DEFAULT_MAX_SECONDS
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES

    def __post_init__(self) -> None:
        for name, value, maximum in (
            ("max_dimension", self.max_dimension, DEFAULT_MAX_DIMENSION),
            ("max_probes", self.max_probes, DEFAULT_MAX_PROBES),
            ("max_features", self.max_features, DEFAULT_MAX_FEATURES),
            ("max_estimated_bytes", self.max_estimated_bytes, DEFAULT_MAX_ESTIMATED_BYTES),
            ("max_work_units", self.max_work_units, DEFAULT_MAX_WORK_UNITS),
            ("max_output_bytes", self.max_output_bytes, DEFAULT_MAX_OUTPUT_BYTES),
        ):
            checked = _plain_int(value, name, 1, maximum)
            object.__setattr__(self, name, checked)
        object.__setattr__(self, "max_seconds", _plain_float(self.max_seconds, "max_seconds", 0.001, DEFAULT_MAX_SECONDS))


@dataclass(frozen=True)
class ObservabilityResourceEstimate:
    """Allocation-free estimate retained with every bounded result."""

    dimension: int
    feature_count: int
    probe_count: int
    estimated_bytes: int
    work_units: int
    max_seconds: float

    def as_dict(self) -> Dict[str, object]:
        return asdict(self)


class _Deadline:
    def __init__(self, seconds: float) -> None:
        self._deadline = time.monotonic() + seconds

    def check(self, stage: str) -> None:
        if time.monotonic() > self._deadline:
            raise ObservabilityResourceError(f"deadline exceeded at {stage}")


def estimate_observability_resources(
    *,
    dimension: int,
    feature_count: int,
    probe_count: int,
    budget: ObservabilityBudget = ObservabilityBudget(),
) -> ObservabilityResourceEstimate:
    """Preflight a streaming Gramian without constructing experiment arrays."""

    d = _plain_int(dimension, "dimension", 1, budget.max_dimension)
    features = _plain_int(feature_count, "feature_count", 1, budget.max_features)
    probes = _plain_int(probe_count, "probe_count", 1, budget.max_probes)
    gramian_bytes = features * features * 8
    vector_bytes = features * 8 * 3
    estimated_bytes = gramian_bytes + vector_bytes + 64 * 1024
    work_units = probes * features * 4
    if estimated_bytes > budget.max_estimated_bytes:
        raise ObservabilityResourceError(
            f"modeled allocation {estimated_bytes} exceeds {budget.max_estimated_bytes} bytes"
        )
    if work_units > budget.max_work_units:
        raise ObservabilityResourceError(
            f"modeled work {work_units} exceeds {budget.max_work_units} units"
        )
    return ObservabilityResourceEstimate(
        dimension=d,
        feature_count=features,
        probe_count=probes,
        estimated_bytes=estimated_bytes,
        work_units=work_units,
        max_seconds=budget.max_seconds,
    )


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest_payload(value: Mapping[str, object]) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ObservabilityArtifact:
    """Content-addressed JSON envelope for one bounded experiment suite."""

    stage_id: str
    protocol_id: str
    status: str
    seed: int
    budget: Dict[str, object]
    resource_estimate: Dict[str, object]
    result: Dict[str, object]
    limitations: Tuple[str, ...]
    artifact_digest: str

    @classmethod
    def create(
        cls,
        *,
        stage_id: str,
        seed: int,
        budget: ObservabilityBudget,
        resource_estimate: ObservabilityResourceEstimate,
        result: Mapping[str, object],
        limitations: Sequence[str],
    ) -> "ObservabilityArtifact":
        if stage_id not in (
            OBS1_STAGE_ID,
            COA1_STAGE_ID,
            CTX1_STAGE_ID,
            TRANSPORT_STAGE_ID,
            EK7_COALITION_STAGE_ID,
        ):
            raise ObservabilityValidationError("unknown observability stage")
        checked_seed = _plain_int(seed, "seed", 0, (1 << 63) - 1)
        checked_limitations = tuple(str(item) for item in limitations)
        payload: Dict[str, object] = {
            "stage_id": stage_id,
            "protocol_id": OBSERVABILITY_PROTOCOL_ID,
            "status": "COMPLETE",
            "seed": checked_seed,
            "budget": asdict(budget),
            "resource_estimate": resource_estimate.as_dict(),
            "result": dict(result),
            "limitations": list(checked_limitations),
        }
        return cls(
            stage_id=stage_id,
            protocol_id=OBSERVABILITY_PROTOCOL_ID,
            status="COMPLETE",
            seed=checked_seed,
            budget=asdict(budget),
            resource_estimate=resource_estimate.as_dict(),
            result=dict(result),
            limitations=checked_limitations,
            artifact_digest=_digest_payload(payload),
        )

    def as_dict(self) -> Dict[str, object]:
        return {
            "stage_id": self.stage_id,
            "protocol_id": self.protocol_id,
            "status": self.status,
            "seed": self.seed,
            "budget": self.budget,
            "resource_estimate": self.resource_estimate,
            "result": self.result,
            "limitations": list(self.limitations),
            "artifact_digest": self.artifact_digest,
        }

    def write_json(self, output: object) -> Path:
        if isinstance(output, Path):
            path = output
        elif type(output) is str and output:
            path = Path(output)
        else:
            raise ObservabilityValidationError("output must be a non-empty path")
        encoded = (_canonical_json(self.as_dict()) + "\n").encode("utf-8")
        if len(encoded) > int(self.budget["max_output_bytes"]):
            raise ObservabilityResourceError("artifact exceeds max_output_bytes")
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Optional[Path] = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="wb",
                dir=str(path.parent),
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                handle.write(encoded)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, path)
            temporary_path = None
        finally:
            if temporary_path is not None:
                try:
                    temporary_path.unlink()
                except FileNotFoundError:
                    pass
        return path


def _validate_seed(seed: object) -> int:
    return _plain_int(seed, "seed", 0, (1 << 63) - 1)


def _validate_policy(policy: str, allowed: Iterable[str]) -> str:
    if type(policy) is not str or policy not in tuple(allowed):
        raise ObservabilityValidationError("unknown policy: {}".format(policy))
    return policy


def _unit_vector(index: int, dimension: int) -> np.ndarray:
    vector = np.zeros(dimension, dtype=np.float64)
    vector[index] = 1.0
    return vector


def _off_span_probe(
    policy: str,
    step: int,
    *,
    dimension: int,
    initial_rank: int,
    rng: np.random.Generator,
) -> np.ndarray:
    if policy == "in_span":
        return _unit_vector(step % initial_rank, dimension)
    if policy == "bounded_full_rank":
        return _unit_vector(step % dimension, dimension)
    if policy == "orthogonal_frontier":
        if step < initial_rank:
            return _unit_vector(step, dimension)
        return _unit_vector(initial_rank + ((step - initial_rank) % (dimension - initial_rank)), dimension)
    if policy == "bounded_isotropic":
        vector = rng.normal(size=dimension)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            vector[0] = 1.0
            norm = 1.0
        return vector / norm
    raise ObservabilityValidationError(f"unknown off-span policy: {policy}")


def _stream_rank(gramian: np.ndarray) -> Tuple[int, float, np.ndarray]:
    eigenvalues = np.linalg.eigvalsh(gramian)
    scale = max(float(eigenvalues[-1]), 1.0)
    threshold = _RANK_RELATIVE_TOLERANCE * scale
    rank = int(np.count_nonzero(eigenvalues > threshold))
    positive = eigenvalues[eigenvalues > threshold]
    minimum = float(positive[0]) if positive.size else 0.0
    return rank, minimum, eigenvalues


def run_obs1_case(
    *,
    policy: str,
    dimension: int = 16,
    initial_rank: int = 4,
    probes: int = 32,
    seed: int = 7,
    effect_size: float = 1.0,
    noise_std: float = 0.0,
    budget: ObservabilityBudget = ObservabilityBudget(),
) -> Dict[str, object]:
    """Run one streaming off-span case without retaining the probe matrix."""

    allowed = ("in_span", "bounded_full_rank", "orthogonal_frontier", "bounded_isotropic")
    selected_policy = _validate_policy(policy, allowed)
    d = _plain_int(dimension, "dimension", 2, budget.max_dimension)
    k = _plain_int(initial_rank, "initial_rank", 1, d - 1)
    n = _plain_int(probes, "probes", 1, budget.max_probes)
    effect = _plain_float(effect_size, "effect_size", 0.0, 1_000.0)
    noise = _plain_float(noise_std, "noise_std", 0.0, 1_000.0)
    resource = estimate_observability_resources(
        dimension=d,
        feature_count=d,
        probe_count=n,
        budget=budget,
    )
    deadline = _Deadline(budget.max_seconds)
    rng = np.random.default_rng(_validate_seed(seed))
    hidden_index = k
    theta = np.zeros(d, dtype=np.float64)
    theta[hidden_index] = effect
    gramian = np.zeros((d, d), dtype=np.float64)
    response_vector = np.zeros(d, dtype=np.float64)
    hidden_energy = 0.0
    for step in range(n):
        deadline.check("obs1_probe")
        probe = _off_span_probe(
            selected_policy,
            step,
            dimension=d,
            initial_rank=k,
            rng=rng,
        )
        response = float(np.dot(theta, probe))
        if noise:
            response += float(rng.normal(scale=noise))
        gramian += np.outer(probe, probe)
        response_vector += probe * response
        hidden_energy += float(probe[hidden_index] ** 2)
    rank, minimum_eigenvalue, eigenvalues = _stream_rank(gramian)
    estimate = np.linalg.pinv(gramian, rcond=_RANK_RELATIVE_TOLERANCE) @ response_vector
    hidden_estimate = float(estimate[hidden_index])
    discovered = bool(
        hidden_energy > 0.0
        and abs(hidden_estimate) >= max(effect * 0.5, 1e-9)
    )
    return {
        "stage_id": OBS1_STAGE_ID,
        "policy": selected_policy,
        "dimension": d,
        "initial_rank": k,
        "probe_count": n,
        "seed": _validate_seed(seed),
        "hidden_index": hidden_index,
        "hidden_energy": hidden_energy,
        "design_rank": rank,
        "minimum_positive_eigenvalue": minimum_eigenvalue,
        "largest_eigenvalue": float(eigenvalues[-1]),
        "hidden_effect_estimate": hidden_estimate,
        "discovered": discovered,
        "support_preserves_initial_span": bool(hidden_energy == 0.0),
        "effect_size": effect,
        "noise_std": noise,
        "resource_estimate": resource.as_dict(),
    }


def run_obs1_suite(
    *,
    dimension: int = 16,
    initial_rank: int = 4,
    probes: int = 32,
    seeds: Sequence[int] = (7, 11),
    budget: ObservabilityBudget = ObservabilityBudget(),
) -> Dict[str, object]:
    """Run the bounded OBS1 scout/confirmation matrix."""

    checked_seeds = tuple(_validate_seed(seed) for seed in seeds)
    if not checked_seeds:
        raise ObservabilityValidationError("seeds cannot be empty")
    rows: List[Dict[str, object]] = []
    for seed in checked_seeds:
        for policy in ("in_span", "bounded_full_rank", "orthogonal_frontier", "bounded_isotropic"):
            rows.append(
                run_obs1_case(
                    policy=policy,
                    dimension=dimension,
                    initial_rank=initial_rank,
                    probes=probes,
                    seed=seed,
                    budget=budget,
                )
            )
    return {
        "stage_id": OBS1_STAGE_ID,
        "protocol_id": OBSERVABILITY_PROTOCOL_ID,
        "evidence_state": "VALIDATED",
        "scientific_claim_status": "UNCONFIRMED",
        "rows": rows,
        "sanity": {
            "in_span_never_discovers": all(
                not bool(row["discovered"])
                for row in rows
                if row["policy"] == "in_span"
            ),
            "bounded_full_rank_discovers": all(
                bool(row["discovered"])
                for row in rows
                if row["policy"] == "bounded_full_rank"
            ),
        },
    }


def _interaction_subsets(dimension: int, order: int) -> Tuple[Tuple[int, ...], ...]:
    subsets: List[Tuple[int, ...]] = [()]
    for degree in range(1, order + 1):
        subsets.extend(itertools.combinations(range(dimension), degree))
    return tuple(subsets)


def _interaction_features(mask: np.ndarray, subsets: Sequence[Tuple[int, ...]]) -> np.ndarray:
    return np.asarray(
        [1.0 if not subset else float(np.prod(mask[list(subset)])) for subset in subsets],
        dtype=np.float64,
    )


def _coalition_masks(
    policy: str,
    *,
    dimension: int,
    max_active: int,
    probes: int,
    rng: np.random.Generator,
) -> Iterable[np.ndarray]:
    if policy == "singleton":
        for step in range(probes):
            mask = np.zeros(dimension, dtype=np.float64)
            mask[step % dimension] = 1.0
            yield mask
        return
    if policy == "top_k":
        candidates: List[Tuple[int, ...]] = []
        for degree in range(1, max_active + 1):
            candidates.extend(itertools.combinations(range(dimension), degree))
        if not candidates:
            raise ObservabilityValidationError("top_k requires max_active >= 1")
        for step in range(probes):
            mask = np.zeros(dimension, dtype=np.float64)
            mask[list(candidates[step % len(candidates)])] = 1.0
            yield mask
        return
    if policy == "balanced_factorial":
        candidates = list(itertools.combinations(range(dimension), max_active))
        if not candidates:
            raise ObservabilityValidationError("balanced_factorial has no masks")
        for step in range(probes):
            mask = np.zeros(dimension, dtype=np.float64)
            selected = candidates[step % len(candidates)]
            mask[list(selected)] = 1.0
            yield mask
        return
    if policy == "random_masks":
        for _ in range(probes):
            mask = (rng.random(dimension) < 0.5).astype(np.float64)
            if not np.any(mask):
                mask[int(rng.integers(0, dimension))] = 1.0
            yield mask
        return
    raise ObservabilityValidationError(f"unknown coalitional policy: {policy}")


def run_coa1_case(
    *,
    policy: str,
    dimension: int = 6,
    order: int = 2,
    max_active: int = 1,
    probes: int = 64,
    seed: int = 7,
    budget: ObservabilityBudget = ObservabilityBudget(),
) -> Dict[str, object]:
    """Run one pure interaction case in the lifted feature space."""

    allowed = ("singleton", "top_k", "balanced_factorial", "random_masks")
    selected_policy = _validate_policy(policy, allowed)
    d = _plain_int(dimension, "dimension", 2, budget.max_dimension)
    q = _plain_int(order, "order", 1, 3)
    active = _plain_int(max_active, "max_active", 1, d)
    n = _plain_int(probes, "probes", 1, budget.max_probes)
    subsets = _interaction_subsets(d, q)
    feature_count = len(subsets)
    if feature_count > budget.max_features:
        raise ObservabilityResourceError("lifted interaction feature count exceeds budget")
    resource = estimate_observability_resources(
        dimension=d,
        feature_count=feature_count,
        probe_count=n,
        budget=budget,
    )
    deadline = _Deadline(budget.max_seconds)
    rng = np.random.default_rng(_validate_seed(seed))
    hidden_subset = tuple(range(q))
    hidden_column = subsets.index(hidden_subset)
    gramian = np.zeros((feature_count, feature_count), dtype=np.float64)
    response_vector = np.zeros(feature_count, dtype=np.float64)
    hidden_energy = 0.0
    for mask in _coalition_masks(
        selected_policy,
        dimension=d,
        max_active=active,
        probes=n,
        rng=rng,
    ):
        deadline.check("coa1_probe")
        features = _interaction_features(mask, subsets)
        response = float(np.prod(mask[list(hidden_subset)]))
        gramian += np.outer(features, features)
        response_vector += features * response
        hidden_energy += float(features[hidden_column] ** 2)
    rank, minimum_eigenvalue, eigenvalues = _stream_rank(gramian)
    estimate = np.linalg.pinv(gramian, rcond=_RANK_RELATIVE_TOLERANCE) @ response_vector
    hidden_estimate = float(estimate[hidden_column])
    support_below_degree = active < q or selected_policy == "singleton"
    discovered = bool(
        hidden_energy > 0.0
        and abs(hidden_estimate) >= 0.5
    )
    return {
        "stage_id": COA1_STAGE_ID,
        "policy": selected_policy,
        "dimension": d,
        "order": q,
        "max_active": active,
        "probe_count": n,
        "seed": _validate_seed(seed),
        "hidden_subset": list(hidden_subset),
        "feature_count": feature_count,
        "hidden_energy": hidden_energy,
        "design_rank": rank,
        "minimum_positive_eigenvalue": minimum_eigenvalue,
        "largest_eigenvalue": float(eigenvalues[-1]),
        "hidden_interaction_estimate": hidden_estimate,
        "support_below_degree": support_below_degree,
        "discovered": discovered,
        "required_sanity_holds": (not discovered) if support_below_degree else discovered,
        "resource_estimate": resource.as_dict(),
    }


def run_coa1_suite(
    *,
    dimension: int = 6,
    probes: int = 64,
    seeds: Sequence[int] = (7, 11),
    budget: ObservabilityBudget = ObservabilityBudget(),
) -> Dict[str, object]:
    """Run bounded pair and triad top-k observability cases."""

    checked_seeds = tuple(_validate_seed(seed) for seed in seeds)
    if not checked_seeds:
        raise ObservabilityValidationError("seeds cannot be empty")
    rows: List[Dict[str, object]] = []
    for seed in checked_seeds:
        rows.append(
            run_coa1_case(
                policy="singleton",
                dimension=dimension,
                order=2,
                max_active=1,
                probes=probes,
                seed=seed,
                budget=budget,
            )
        )
        rows.append(
            run_coa1_case(
                policy="top_k",
                dimension=dimension,
                order=2,
                max_active=2,
                probes=probes,
                seed=seed,
                budget=budget,
            )
        )
        rows.append(
            run_coa1_case(
                policy="top_k",
                dimension=dimension,
                order=3,
                max_active=2,
                probes=probes,
                seed=seed,
                budget=budget,
            )
        )
        rows.append(
            run_coa1_case(
                policy="top_k",
                dimension=dimension,
                order=3,
                max_active=3,
                probes=probes,
                seed=seed,
                budget=budget,
            )
        )
    return {
        "stage_id": COA1_STAGE_ID,
        "protocol_id": OBSERVABILITY_PROTOCOL_ID,
        "evidence_state": "VALIDATED",
        "scientific_claim_status": "UNCONFIRMED",
        "rows": rows,
        "sanity": {
            "below_degree_remains_unidentified": all(
                bool(row["required_sanity_holds"])
                for row in rows
                if bool(row["support_below_degree"])
            ),
            "eligible_pair_recovers": all(
                bool(row["discovered"])
                for row in rows
                if row["order"] == 2 and not bool(row["support_below_degree"])
            ),
            "eligible_triad_recovers": all(
                bool(row["discovered"])
                for row in rows
                if row["order"] == 3 and not bool(row["support_below_degree"])
            ),
        },
    }


def run_ctx1_case(
    *,
    case: str,
    seed: int = 7,
    minimum_effect: float = 0.5,
    budget: ObservabilityBudget = ObservabilityBudget(),
) -> Dict[str, object]:
    """Run a deterministic provenance/context qualification fixture.

    The fixture is intentionally small and explicit.  It separates a naive
    ``max singleton + max singleton`` promotion from a context- and support-
    aware decision.  ``UNIDENTIFIED`` is the correct result when the required
    shared context or propensity support is absent; that absence is not a
    negative effect.
    """

    selected_case = _validate_policy(
        case,
        ("incompatible_witness", "compatible_positive", "missing_propensity"),
    )
    checked_seed = _validate_seed(seed)
    threshold = _plain_float(minimum_effect, "minimum_effect", 0.0, 1_000.0)
    resource = estimate_observability_resources(
        dimension=4,
        feature_count=8,
        probe_count=12,
        budget=budget,
    )
    fixtures = {
        "incompatible_witness": (
            {
                "context": "ctx_a",
                "a_singleton": 1.0,
                "b_singleton": -0.2,
                "joint_effect": -0.8,
                "propensity_overlap": 0.0,
                "lineage_compatible": True,
            },
            {
                "context": "ctx_b",
                "a_singleton": -0.2,
                "b_singleton": 1.0,
                "joint_effect": -0.8,
                "propensity_overlap": 0.0,
                "lineage_compatible": True,
            },
        ),
        "compatible_positive": (
            {
                "context": "ctx_shared",
                "a_singleton": 0.8,
                "b_singleton": 0.7,
                "joint_effect": 1.2,
                "propensity_overlap": 0.6,
                "lineage_compatible": True,
            },
            {
                "context": "ctx_replicate",
                "a_singleton": 0.75,
                "b_singleton": 0.65,
                "joint_effect": 1.0,
                "propensity_overlap": 0.5,
                "lineage_compatible": True,
            },
        ),
        "missing_propensity": (
            {
                "context": "ctx_shared",
                "a_singleton": 0.8,
                "b_singleton": 0.7,
                "joint_effect": 1.2,
                "propensity_overlap": 0.0,
                "lineage_compatible": True,
            },
        ),
    }
    rows = [dict(row) for row in fixtures[selected_case]]
    a_witnesses = [row for row in rows if row["a_singleton"] >= threshold]
    b_witnesses = [row for row in rows if row["b_singleton"] >= threshold]
    naive_promotes = bool(a_witnesses and b_witnesses)
    shared_positive_contexts = [
        row
        for row in rows
        if row["a_singleton"] >= threshold and row["b_singleton"] >= threshold
    ]
    qualified_contexts = [
        row
        for row in shared_positive_contexts
        if row["propensity_overlap"] > 0.0 and row["lineage_compatible"]
    ]
    if not shared_positive_contexts or not qualified_contexts:
        aware_decision = "UNIDENTIFIED"
        qualified_joint_effect = None
    else:
        qualified_joint_effect = float(
            sum(row["joint_effect"] for row in qualified_contexts)
            / len(qualified_contexts)
        )
        aware_decision = "PROMOTE" if qualified_joint_effect >= threshold else "REJECT"
    required_sanity_holds = (
        aware_decision != "PROMOTE"
        if selected_case in ("incompatible_witness", "missing_propensity")
        else aware_decision == "PROMOTE"
    )
    return {
        "stage_id": CTX1_STAGE_ID,
        "case": selected_case,
        "seed": checked_seed,
        "minimum_effect": threshold,
        "context_count": len(rows),
        "singleton_a_witness_count": len(a_witnesses),
        "singleton_b_witness_count": len(b_witnesses),
        "naive_promotes": naive_promotes,
        "shared_positive_contexts": [row["context"] for row in shared_positive_contexts],
        "qualified_contexts": [row["context"] for row in qualified_contexts],
        "qualified_joint_effect": qualified_joint_effect,
        "aware_decision": aware_decision,
        "required_sanity_holds": required_sanity_holds,
        "resource_estimate": resource.as_dict(),
    }


def run_ctx1_suite(
    *,
    seeds: Sequence[int] = (7, 11),
    budget: ObservabilityBudget = ObservabilityBudget(),
) -> Dict[str, object]:
    """Run compatible, incompatible, and missing-support context cells."""

    checked_seeds = tuple(_validate_seed(seed) for seed in seeds)
    if not checked_seeds:
        raise ObservabilityValidationError("seeds cannot be empty")
    rows: List[Dict[str, object]] = []
    for seed in checked_seeds:
        for case in ("incompatible_witness", "compatible_positive", "missing_propensity"):
            rows.append(run_ctx1_case(case=case, seed=seed, budget=budget))
    return {
        "stage_id": CTX1_STAGE_ID,
        "protocol_id": OBSERVABILITY_PROTOCOL_ID,
        "evidence_state": "VALIDATED",
        "scientific_claim_status": "UNCONFIRMED",
        "rows": rows,
        "sanity": {
            "incompatible_witness_not_promoted": all(
                row["required_sanity_holds"]
                for row in rows
                if row["case"] == "incompatible_witness"
            ),
            "compatible_positive_promoted": all(
                row["aware_decision"] == "PROMOTE"
                for row in rows
                if row["case"] == "compatible_positive"
            ),
            "missing_propensity_abstains": all(
                row["aware_decision"] == "UNIDENTIFIED"
                for row in rows
                if row["case"] == "missing_propensity"
            ),
        },
    }


def _pairwise_cosine(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(norms <= 0.0):
        raise ObservabilityValidationError("cosine matrix received a zero vector")
    normalized = matrix / norms[:, None]
    return normalized @ normalized.T


def _off_diagonal_mean(matrix: np.ndarray) -> float:
    if matrix.shape[0] < 2:
        return 0.0
    mask = ~np.eye(matrix.shape[0], dtype=bool)
    return float(np.mean(matrix[mask]))


def _top_one_agreement(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape or left.shape[0] < 2:
        raise ObservabilityValidationError("top-one inputs must be matching square matrices")
    left_scores = left.copy()
    right_scores = right.copy()
    np.fill_diagonal(left_scores, -np.inf)
    np.fill_diagonal(right_scores, -np.inf)
    return float(np.mean(np.argmax(left_scores, axis=1) == np.argmax(right_scores, axis=1)))


def run_transport_case(
    *,
    dimension: int = 8,
    points: int = 48,
    seed: int = 7,
    scale_min: float = 0.25,
    scale_max: float = 2.5,
    budget: ObservabilityBudget = ObservabilityBudget(),
) -> Dict[str, object]:
    """Measure a known coordinate transport confound without labels or models."""

    d = _plain_int(dimension, "dimension", 2, budget.max_dimension)
    n = _plain_int(points, "points", 2, budget.max_probes)
    checked_seed = _validate_seed(seed)
    low = _plain_float(scale_min, "scale_min", 0.001, 100.0)
    high = _plain_float(scale_max, "scale_max", low, 100.0)
    resource = estimate_observability_resources(
        dimension=d,
        feature_count=d,
        probe_count=n,
        budget=budget,
    )
    deadline = _Deadline(budget.max_seconds)
    rng = np.random.default_rng(checked_seed)
    latent = rng.normal(size=(n, d))
    rotation_source = rng.normal(size=(d, d))
    rotation, _ = np.linalg.qr(rotation_source)
    scales = np.linspace(low, high, d)
    transport = rotation @ np.diag(scales)
    represented = latent @ transport
    aligned = represented @ np.linalg.inv(transport)
    deadline.check("transport_metrics")
    latent_cosine = _pairwise_cosine(latent)
    raw_cosine = _pairwise_cosine(represented)
    aligned_cosine = _pairwise_cosine(aligned)
    raw_error = _off_diagonal_mean(np.abs(raw_cosine - latent_cosine))
    aligned_error = _off_diagonal_mean(np.abs(aligned_cosine - latent_cosine))
    raw_top_one = _top_one_agreement(raw_cosine, latent_cosine)
    aligned_top_one = _top_one_agreement(aligned_cosine, latent_cosine)
    return {
        "stage_id": TRANSPORT_STAGE_ID,
        "dimension": d,
        "point_count": n,
        "seed": checked_seed,
        "scale_min": low,
        "scale_max": high,
        "raw_cosine_error": raw_error,
        "aligned_cosine_error": aligned_error,
        "raw_top_one_agreement": raw_top_one,
        "aligned_top_one_agreement": aligned_top_one,
        "transport_recovers_metric": bool(aligned_error <= 1e-10),
        "raw_drift_visible": bool(raw_error > 1e-3),
        "resource_estimate": resource.as_dict(),
    }


def run_transport_suite(
    *,
    dimension: int = 8,
    points: int = 48,
    seeds: Sequence[int] = (7, 11),
    budget: ObservabilityBudget = ObservabilityBudget(),
) -> Dict[str, object]:
    checked_seeds = tuple(_validate_seed(seed) for seed in seeds)
    if not checked_seeds:
        raise ObservabilityValidationError("seeds cannot be empty")
    rows = [
        run_transport_case(
            dimension=dimension,
            points=points,
            seed=seed,
            budget=budget,
        )
        for seed in checked_seeds
    ]
    return {
        "stage_id": TRANSPORT_STAGE_ID,
        "protocol_id": OBSERVABILITY_PROTOCOL_ID,
        "evidence_state": "VALIDATED",
        "scientific_claim_status": "UNCONFIRMED",
        "rows": rows,
        "sanity": {
            "transport_recovers_metric": all(row["transport_recovers_metric"] for row in rows),
            "raw_drift_visible": all(row["raw_drift_visible"] for row in rows),
        },
    }


def _rag_documents() -> Tuple[Tuple[str, Tuple[str, ...]], ...]:
    return (
        ("d_ab_1", ("alpha", "beta")),
        ("d_ab_2", ("alpha", "beta")),
        ("d_gamma", ("gamma",)),
        ("d_delta", ("delta",)),
        ("d_noise", ("omega",)),
    )


def _coverage(document_ids: Sequence[str], documents: Mapping[str, Sequence[str]]) -> List[str]:
    tokens = set()
    for document_id in document_ids:
        tokens.update(documents[document_id])
    return sorted(tokens)


def run_coalition_rag_case(
    *,
    query_tokens: Sequence[str] = ("alpha", "beta", "gamma", "delta"),
    document_budget: int = 3,
    seed: int = 7,
    budget: ObservabilityBudget = ObservabilityBudget(),
) -> Dict[str, object]:
    """Run a matched-token coalition surrogate for a conjunctive query."""

    checked_seed = _validate_seed(seed)
    if not query_tokens or any(type(token) is not str or not token for token in query_tokens):
        raise ObservabilityValidationError("query_tokens must contain non-empty strings")
    k = _plain_int(document_budget, "document_budget", 1, 5)
    documents_tuple = _rag_documents()
    documents = {document_id: tokens for document_id, tokens in documents_tuple}
    query = set(query_tokens)
    resource = estimate_observability_resources(
        dimension=len(query),
        feature_count=len(documents_tuple),
        probe_count=len(documents_tuple),
        budget=budget,
    )
    overlaps = {
        document_id: len(query.intersection(tokens))
        for document_id, tokens in documents_tuple
    }
    individual_selected = [
        document_id
        for document_id, _ in sorted(
            documents_tuple,
            key=lambda item: (-overlaps[item[0]], item[0]),
        )[:k]
    ]
    coalition_selected: List[str] = []
    covered = set()
    remaining = {document_id for document_id, _ in documents_tuple}
    for _ in range(k):
        if not remaining:
            break
        selected = min(
            remaining,
            key=lambda document_id: (
                -len((query.intersection(documents[document_id])) - covered),
                -overlaps[document_id],
                document_id,
            ),
        )
        coalition_selected.append(selected)
        covered.update(query.intersection(documents[selected]))
        remaining.remove(selected)
    individual_coverage = _coverage(individual_selected, documents)
    coalition_coverage = _coverage(coalition_selected, documents)
    return {
        "stage_id": EK7_COALITION_STAGE_ID,
        "seed": checked_seed,
        "query_tokens": sorted(query),
        "document_budget": k,
        "document_count": len(documents_tuple),
        "individual_selected": individual_selected,
        "coalition_selected": coalition_selected,
        "individual_coverage": individual_coverage,
        "coalition_coverage": coalition_coverage,
        "individual_conjunctive_success": query.issubset(individual_coverage),
        "coalition_conjunctive_success": query.issubset(coalition_coverage),
        "resource_estimate": resource.as_dict(),
    }


def run_coalition_rag_suite(
    *,
    seeds: Sequence[int] = (7, 11),
    budget: ObservabilityBudget = ObservabilityBudget(),
) -> Dict[str, object]:
    checked_seeds = tuple(_validate_seed(seed) for seed in seeds)
    if not checked_seeds:
        raise ObservabilityValidationError("seeds cannot be empty")
    rows = [run_coalition_rag_case(seed=seed, budget=budget) for seed in checked_seeds]
    return {
        "stage_id": EK7_COALITION_STAGE_ID,
        "protocol_id": OBSERVABILITY_PROTOCOL_ID,
        "evidence_state": "VALIDATED",
        "scientific_claim_status": "UNCONFIRMED",
        "rows": rows,
        "sanity": {
            "individual_top_k_misses_conjunction": all(
                not row["individual_conjunctive_success"] for row in rows
            ),
            "coalition_recovers_conjunction": all(
                row["coalition_conjunctive_success"] for row in rows
            ),
        },
    }


def make_obs1_artifact(
    *,
    seed: int = 7,
    budget: ObservabilityBudget = ObservabilityBudget(),
) -> ObservabilityArtifact:
    result = run_obs1_suite(seeds=(seed,), budget=budget)
    resource = estimate_observability_resources(
        dimension=16,
        feature_count=16,
        probe_count=32,
        budget=budget,
    )
    return ObservabilityArtifact.create(
        stage_id=OBS1_STAGE_ID,
        seed=_validate_seed(seed),
        budget=budget,
        resource_estimate=resource,
        result=result,
        limitations=(
            "synthetic linear response only",
            "no semantic ontology generation",
            "no production retrieval or model evidence",
        ),
    )


def make_coa1_artifact(
    *,
    seed: int = 7,
    budget: ObservabilityBudget = ObservabilityBudget(),
) -> ObservabilityArtifact:
    result = run_coa1_suite(seeds=(seed,), budget=budget)
    resource = estimate_observability_resources(
        dimension=6,
        feature_count=len(_interaction_subsets(6, 3)),
        probe_count=64,
        budget=budget,
    )
    return ObservabilityArtifact.create(
        stage_id=COA1_STAGE_ID,
        seed=_validate_seed(seed),
        budget=budget,
        resource_estimate=resource,
        result=result,
        limitations=(
            "synthetic Boolean polynomial response only",
            "interaction existence is not utility advantage",
            "no RAG corpus or model evidence",
        ),
    )


def make_ctx1_artifact(
    *,
    seed: int = 7,
    budget: ObservabilityBudget = ObservabilityBudget(),
) -> ObservabilityArtifact:
    result = run_ctx1_suite(seeds=(seed,), budget=budget)
    resource = estimate_observability_resources(
        dimension=4,
        feature_count=8,
        probe_count=12,
        budget=budget,
    )
    return ObservabilityArtifact.create(
        stage_id=CTX1_STAGE_ID,
        seed=_validate_seed(seed),
        budget=budget,
        resource_estimate=resource,
        result=result,
        limitations=(
            "deterministic context/effect fixture only",
            "no causal estimate from a live corpus",
            "UNIDENTIFIED denotes missing compatible support, not a negative effect",
        ),
    )


def make_transport_artifact(
    *,
    seed: int = 7,
    budget: ObservabilityBudget = ObservabilityBudget(),
) -> ObservabilityArtifact:
    result = run_transport_suite(seeds=(seed,), budget=budget)
    resource = estimate_observability_resources(
        dimension=8,
        feature_count=8,
        probe_count=48,
        budget=budget,
    )
    return ObservabilityArtifact.create(
        stage_id=TRANSPORT_STAGE_ID,
        seed=_validate_seed(seed),
        budget=budget,
        resource_estimate=resource,
        result=result,
        limitations=(
            "known synthetic coordinate transport only",
            "transport correction is a confound control, not a discovery mechanism",
            "no learned alignment or production representation evidence",
        ),
    )


def make_coalition_rag_artifact(
    *,
    seed: int = 7,
    budget: ObservabilityBudget = ObservabilityBudget(),
) -> ObservabilityArtifact:
    result = run_coalition_rag_suite(seeds=(seed,), budget=budget)
    resource = estimate_observability_resources(
        dimension=4,
        feature_count=5,
        probe_count=5,
        budget=budget,
    )
    return ObservabilityArtifact.create(
        stage_id=EK7_COALITION_STAGE_ID,
        seed=_validate_seed(seed),
        budget=budget,
        resource_estimate=resource,
        result=result,
        limitations=(
            "matched-token document surrogate only",
            "no corpus, embedding model, generator, or answer-quality evaluation",
            "optional EK7 survivor ablation remains gated by the unchanged EK7 entry contract",
        ),
    )


__all__ = [
    "COA1_STAGE_ID",
    "CTX1_STAGE_ID",
    "EK7_COALITION_STAGE_ID",
    "OBS1_STAGE_ID",
    "OBSERVABILITY_PROTOCOL_ID",
    "ObservabilityArtifact",
    "ObservabilityBudget",
    "ObservabilityResourceEstimate",
    "ObservabilityResourceError",
    "ObservabilityValidationError",
    "TRANSPORT_STAGE_ID",
    "estimate_observability_resources",
    "make_coalition_rag_artifact",
    "make_coa1_artifact",
    "make_ctx1_artifact",
    "make_obs1_artifact",
    "make_transport_artifact",
    "run_coalition_rag_case",
    "run_coalition_rag_suite",
    "run_coa1_case",
    "run_coa1_suite",
    "run_ctx1_case",
    "run_ctx1_suite",
    "run_obs1_case",
    "run_obs1_suite",
    "run_transport_case",
    "run_transport_suite",
]
