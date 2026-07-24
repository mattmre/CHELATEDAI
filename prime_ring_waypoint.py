"""Prime-ring waypoint (PRW) METHOD_DEV primitives.

This module isolates a falsifiable synchronization mechanism:

* a bipolar Legendre carrier estimates one cyclic shift shared by all layers;
* a separate payload carries waypoint identity;
* an interlayer phase vector lives in ``Z_p**L`` modulo one global rotation;
* constrained carrier-orientation policies are compared with deliberately
  permissive negative controls.

Nothing here is a production router or evidence of retrieval utility.  The
implementation is NumPy-only, deterministic, and fail-closed on malformed or
non-finite inputs.  It uses generic direct/FFT correlation and makes no Rader
transform claim.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple

import numpy as np


PRW_ARITHMETIC_ID = "PRW-ARITH-LEGENDRE-001"
PRW_SHARED_SHIFT_ID = "PRW-SYNC-SHARED-SHIFT-001"
PRW_INDEPENDENT_CONTROL_ID = "PRW-NEG-INDEPENDENT-LAYERS-001"
PRW_TYPED16_ID = "PRW-MASK-TYPED16-001"
PRW_FREE256_CONTROL_ID = "PRW-NEG-FREE256-001"
PRW_WAYPOINT_SCORE_ID = "PRW-WAYPOINT-SEPARATED-001"
PRW_PACKED_BYTES_ID = "PRW-PACKED-BYTES-001"

MASK_POLICIES = ("none", "shared", "typed16", "free256")
CARRIER_FAMILIES = ("legendre", "rademacher", "custom")
PAYLOAD_ENCODINGS = ("float64", "bitpacked")
_PRW_WAYPOINT_ID_RE = re.compile(r"^PRW-WP-[A-Z0-9][A-Z0-9_-]{0,63}$")


class PRWValidationError(ValueError):
    """Raised when a PRW METHOD_DEV input violates its declared contract."""


@dataclass(frozen=True)
class RingObservation:
    """Legacy carrier and payload observation encoded on one cyclic ring.

    ``relative_phase`` is the canonical quotient representative whose first
    coordinate is zero.  ``global_shift`` is kept separately.  The carrier and
    payload remain separate arrays so identity evidence cannot leak into phase
    estimation.  ``orientation_mask`` acts on the carrier/type key, never the
    identity payload.

    This backward-compatible object may represent a synthetically corrupted
    observation, so its latent phase/mask labels are not asserted to reconstruct
    its observed arrays.  New code should use :class:`RingTemplate` for a clean,
    internally consistent candidate and :class:`ObservedRing` for a query.
    """

    waypoint_id: str
    p: int
    carrier: np.ndarray
    payload: np.ndarray
    relative_phase: Tuple[int, ...]
    global_shift: int
    orientation_mask: Tuple[int, ...]
    carrier_family: str = "legendre"

    def __post_init__(self) -> None:
        _validate_waypoint_id(self.waypoint_id)
        _require_ring_length(self.p)
        carrier_family = _carrier_family(self.carrier_family)
        if carrier_family == "legendre":
            _require_ideal_prime(self.p)
        carrier = _finite_layers(self.carrier, "carrier", expected_p=self.p)
        payload = _finite_layers(self.payload, "payload", expected_p=self.p)
        if carrier.shape != payload.shape:
            raise PRWValidationError("carrier and payload must have identical shapes")
        if not np.all(np.logical_or(carrier == -1.0, carrier == 1.0)):
            raise PRWValidationError("carrier must be bipolar (-1 or +1)")
        relative_phase = _phase_vector(
            self.relative_phase,
            carrier.shape[0],
            self.p,
            "relative_phase",
        )
        if relative_phase[0] != 0:
            raise PRWValidationError("relative_phase must be a canonical quotient with first coordinate zero")
        global_shift = _ring_integer(self.global_shift, "global_shift", self.p, canonical=True)
        orientation_mask = _polarity_mask(
            self.orientation_mask,
            carrier.shape[0],
            "orientation_mask",
        )

        carrier = np.array(carrier, dtype=np.float64, copy=True)
        payload = np.array(payload, dtype=np.float64, copy=True)
        carrier.setflags(write=False)
        payload.setflags(write=False)
        object.__setattr__(self, "carrier", carrier)
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "relative_phase", relative_phase)
        object.__setattr__(self, "global_shift", global_shift)
        object.__setattr__(self, "orientation_mask", orientation_mask)
        object.__setattr__(self, "carrier_family", carrier_family)

    @property
    def layers(self) -> int:
        """Number of interlocked ring layers."""

        return int(self.carrier.shape[0])


@dataclass(frozen=True)
class ObservedRing:
    """Structurally valid query arrays with no embedded latent ground truth.

    Observed carriers may be real-valued after a declared corruption process.
    Phase, orientation, corruption, and source-template truth belongs in the
    experiment record, not in this scorer input.
    """

    waypoint_id: str
    p: int
    carrier: np.ndarray
    payload: np.ndarray

    def __post_init__(self) -> None:
        _validate_waypoint_id(self.waypoint_id)
        ring_length = _require_ring_length(self.p)
        carrier = _finite_layers(self.carrier, "carrier", expected_p=ring_length)
        payload = _finite_layers(self.payload, "payload", expected_p=ring_length)
        if carrier.shape != payload.shape:
            raise PRWValidationError("carrier and payload must have identical shapes")

        carrier = np.array(carrier, dtype=np.float64, copy=True)
        payload = np.array(payload, dtype=np.float64, copy=True)
        carrier.setflags(write=False)
        payload.setflags(write=False)
        object.__setattr__(self, "p", ring_length)
        object.__setattr__(self, "carrier", carrier)
        object.__setattr__(self, "payload", payload)

    @property
    def layers(self) -> int:
        """Number of interlocked ring layers."""

        return int(self.carrier.shape[0])


@dataclass(frozen=True)
class RingTemplate:
    """Clean candidate whose carrier is exactly reconstructible from metadata."""

    waypoint_id: str
    p: int
    base_carrier: np.ndarray
    carrier: np.ndarray
    payload: np.ndarray
    relative_phase: Tuple[int, ...]
    orientation_mask: Tuple[int, ...]
    carrier_family: str

    def __post_init__(self) -> None:
        _validate_waypoint_id(self.waypoint_id)
        ring_length = _require_ring_length(self.p)
        carrier_family = _carrier_family(self.carrier_family)
        base_carrier = _bipolar_carrier(
            self.base_carrier,
            ring_length,
            "base_carrier",
        )
        if carrier_family == "legendre":
            _require_ideal_prime(ring_length)
            if not np.array_equal(base_carrier, legendre_carrier(ring_length)):
                raise PRWValidationError(
                    "legendre template base_carrier must equal the canonical Legendre carrier"
                )

        carrier = _finite_layers(self.carrier, "carrier", expected_p=ring_length)
        payload = _finite_layers(self.payload, "payload", expected_p=ring_length)
        if carrier.shape != payload.shape:
            raise PRWValidationError("carrier and payload must have identical shapes")
        if not np.all(np.logical_or(carrier == -1.0, carrier == 1.0)):
            raise PRWValidationError("template carrier must be bipolar (-1 or +1)")
        relative_phase = _phase_vector(
            self.relative_phase,
            carrier.shape[0],
            ring_length,
            "relative_phase",
        )
        if relative_phase[0] != 0:
            raise PRWValidationError(
                "relative_phase must be a canonical quotient with first coordinate zero"
            )
        orientation_mask = _polarity_mask(
            self.orientation_mask,
            carrier.shape[0],
            "orientation_mask",
        )

        expected = np.vstack(
            [
                orientation_mask[layer]
                * np.roll(base_carrier, relative_phase[layer])
                for layer in range(carrier.shape[0])
            ]
        )
        if not np.array_equal(carrier, expected):
            raise PRWValidationError(
                "template carrier is inconsistent with base_carrier, relative_phase, or orientation_mask"
            )

        base_carrier = np.array(base_carrier, dtype=np.float64, copy=True)
        carrier = np.array(carrier, dtype=np.float64, copy=True)
        payload = np.array(payload, dtype=np.float64, copy=True)
        base_carrier.setflags(write=False)
        carrier.setflags(write=False)
        payload.setflags(write=False)
        object.__setattr__(self, "p", ring_length)
        object.__setattr__(self, "base_carrier", base_carrier)
        object.__setattr__(self, "carrier", carrier)
        object.__setattr__(self, "payload", payload)
        object.__setattr__(self, "relative_phase", relative_phase)
        object.__setattr__(self, "orientation_mask", orientation_mask)
        object.__setattr__(self, "carrier_family", carrier_family)

    @property
    def layers(self) -> int:
        """Number of interlocked ring layers."""

        return int(self.carrier.shape[0])


@dataclass(frozen=True)
class AlignmentScore:
    """Result of a shared-shift or factorized negative-control search."""

    mechanism_id: str
    policy: str
    score: float
    shared_shift: Optional[int]
    layer_shifts: Tuple[int, ...]
    mask: Tuple[int, ...]
    per_layer_scores: Tuple[float, ...]
    candidate_count: int
    negative_control: bool


@dataclass(frozen=True)
class WaypointScore:
    """Separated carrier synchronization and synchronized payload score."""

    mechanism_id: str
    query_id: str
    candidate_id: str
    policy: str
    carrier_score: float
    carrier_shift: int
    payload_score: float
    orientation_mask: Tuple[int, ...]
    negative_control: bool


@dataclass(frozen=True)
class PackedByteAccounting:
    """Byte counts for a declared payload encoding and packed control bound."""

    mechanism_id: str
    p: int
    layers: int
    payload_planes: int
    policy: str
    bits_per_residue: int
    canonical_carrier_bytes: int
    layer_plane_bytes: int
    payload_bytes: int
    relative_phase_bytes: int
    orientation_bytes: int
    total_bytes: int
    payload_encoding: str = "float64"
    payload_bits_per_value: int = 64
    theoretical_packed_layer_plane_bytes: int = 0
    theoretical_packed_payload_bytes: int = 0
    theoretical_packed_total_bytes: int = 0


def _integer(value: object, name: str, *, minimum: Optional[int] = None) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise PRWValidationError("{} must be an integer".format(name))
    result = int(value)
    if minimum is not None and result < minimum:
        raise PRWValidationError("{} must be >= {}".format(name, minimum))
    return result


def _ring_integer(value: object, name: str, p: int, *, canonical: bool) -> int:
    result = _integer(value, name)
    if canonical and not 0 <= result < p:
        raise PRWValidationError("{} must be a canonical residue in [0, p)".format(name))
    return result % p


def _validate_waypoint_id(value: object) -> str:
    if not isinstance(value, str) or not _PRW_WAYPOINT_ID_RE.fullmatch(value):
        raise PRWValidationError(
            "waypoint_id must match PRW-WP-[A-Z0-9][A-Z0-9_-]{0,63}"
        )
    return value


def _require_ring_length(p: int) -> int:
    """Return a cyclic ring length usable by generic METHOD_DEV controls."""

    return _integer(p, "p", minimum=3)


def _carrier_family(value: object) -> str:
    if not isinstance(value, str) or value not in CARRIER_FAMILIES:
        raise PRWValidationError(
            "carrier_family must be one of {}".format(", ".join(CARRIER_FAMILIES))
        )
    return value


def _bipolar_carrier(value: object, p: int, name: str = "base_carrier") -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise PRWValidationError(
            "{} must be a one-dimensional bipolar array".format(name)
        ) from exc
    if array.ndim != 1 or array.shape[0] != p:
        raise PRWValidationError("{} must have shape ({},)".format(name, p))
    if array.dtype.kind not in "iuf" or array.dtype.kind == "b":
        raise PRWValidationError("{} must be a real numeric array".format(name))
    try:
        result = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PRWValidationError("{} must be convertible to float64".format(name)) from exc
    if not np.all(np.isfinite(result)):
        raise PRWValidationError("{} must contain only finite values".format(name))
    if not np.all(np.logical_or(result == -1.0, result == 1.0)):
        raise PRWValidationError("{} must be bipolar (-1 or +1)".format(name))
    result = np.array(result, dtype=np.float64, copy=True)
    result.setflags(write=False)
    return result


def is_prime_exact(n: int) -> bool:
    """Return exact primality by deterministic trial division.

    This intentionally favors a transparent exact check over a probabilistic
    test.  It is suitable for the experiment-scale ring lengths considered by
    PRW (including 4091 and 4691), not cryptographic-size integers.
    """

    value = _integer(n, "n")
    if value < 2:
        return False
    if value in (2, 3):
        return True
    if value % 2 == 0 or value % 3 == 0:
        return False
    limit = math.isqrt(value)
    divisor = 5
    while divisor <= limit:
        if value % divisor == 0 or value % (divisor + 2) == 0:
            return False
        divisor += 6
    return True


def _require_ideal_prime(p: int) -> int:
    value = _integer(p, "p", minimum=3)
    if value % 4 != 3 or not is_prime_exact(value):
        raise PRWValidationError("p must be an odd prime congruent to 3 modulo 4")
    return value


def legendre_carrier(p: int) -> np.ndarray:
    """Return the canonical length-``p`` bipolar Legendre carrier.

    Coordinate zero is +1.  For the required primes ``p == 3 (mod 4)``, its
    periodic autocorrelation is exactly ``p`` at zero and ``-1`` elsewhere.
    """

    prime = _require_ideal_prime(p)
    carrier = np.ones(prime, dtype=np.int8)
    exponent = (prime - 1) // 2
    for coordinate in range(1, prime):
        carrier[coordinate] = 1 if pow(coordinate, exponent, prime) == 1 else -1
    carrier.setflags(write=False)
    return carrier


def rademacher_carrier(p: int, seed: int) -> np.ndarray:
    """Return a deterministic bipolar control carrier for any cyclic length.

    A METHOD_DEV campaign should create one carrier per ``(seed, p)`` and share
    it across every type so the random carrier cannot become an extra type key.
    """

    ring_length = _require_ring_length(p)
    random_seed = _integer(seed, "seed", minimum=0)
    rng = np.random.default_rng(random_seed)
    carrier = rng.choice(
        np.array([-1, 1], dtype=np.int8),
        size=ring_length,
        replace=True,
    )
    carrier.setflags(write=False)
    return carrier


def exact_periodic_autocorrelation(sequence: Sequence[int]) -> Tuple[int, ...]:
    """Compute integer periodic autocorrelation without floating-point FFTs."""

    array = np.asarray(sequence)
    if array.ndim != 1 or array.size == 0:
        raise PRWValidationError("sequence must be a non-empty one-dimensional array")
    if array.dtype.kind not in "iu" or array.dtype.kind == "b":
        raise PRWValidationError("sequence must contain exact integers")
    values = tuple(int(value) for value in array.tolist())
    length = len(values)
    return tuple(
        sum(values[index] * values[(index - shift) % length] for index in range(length))
        for shift in range(length)
    )


def has_ideal_legendre_autocorrelation(p: int) -> bool:
    """Verify the exact PRW Legendre autocorrelation identity."""

    prime = _require_ideal_prime(p)
    observed = exact_periodic_autocorrelation(legendre_carrier(prime))
    return observed == (prime,) + (-1,) * (prime - 1)


def factor_integer_exact(n: int) -> Tuple[int, ...]:
    """Return the exact prime factorization of a positive integer.

    Factors include multiplicity and are returned in ascending order.
    """

    remainder = _integer(n, "n", minimum=1)
    if remainder == 1:
        return ()
    factors = []
    divisor = 2
    while divisor * divisor <= remainder:
        while remainder % divisor == 0:
            factors.append(divisor)
            remainder //= divisor
        divisor = 3 if divisor == 2 else divisor + 2
    if remainder > 1:
        factors.append(remainder)
    return tuple(factors)


def is_primitive_root_exact(candidate: int, p: int) -> bool:
    """Return whether ``candidate`` generates the multiplicative group mod p."""

    prime = _integer(p, "p", minimum=2)
    if not is_prime_exact(prime):
        raise PRWValidationError("p must be prime for a primitive-root check")
    root = _ring_integer(candidate, "candidate", prime, canonical=False)
    if root == 0:
        return False
    distinct_factors = set(factor_integer_exact(prime - 1))
    return all(pow(root, (prime - 1) // factor, prime) != 1 for factor in distinct_factors)


def _crt_moduli(moduli: Sequence[int]) -> Tuple[int, ...]:
    if isinstance(moduli, (str, bytes)):
        raise PRWValidationError("moduli must be a non-empty sequence")
    try:
        raw_moduli = tuple(moduli)
    except TypeError as exc:
        raise PRWValidationError("moduli must be a non-empty sequence") from exc
    if not raw_moduli:
        raise PRWValidationError("moduli must be a non-empty sequence")
    normalized = tuple(
        _integer(modulus, "moduli[{}]".format(index), minimum=2)
        for index, modulus in enumerate(raw_moduli)
    )
    for left_index, left in enumerate(normalized):
        for right in normalized[left_index + 1 :]:
            if math.gcd(left, right) != 1:
                raise PRWValidationError("CRT moduli must be pairwise coprime")
    return normalized


def crt_encode_integer(value: int, moduli: Sequence[int]) -> Tuple[int, ...]:
    """Encode a multiplicative-group exponent into pairwise-coprime residues.

    This is an exponent-space CRT helper.  It is intentionally separate from
    the additive ring rotations used by carrier synchronization.
    """

    integer = _integer(value, "value")
    normalized_moduli = _crt_moduli(moduli)
    modulus_product = math.prod(normalized_moduli)
    canonical = integer % modulus_product
    return tuple(canonical % modulus for modulus in normalized_moduli)


def crt_reconstruct_integer(
    residues: Sequence[int],
    moduli: Sequence[int],
) -> int:
    """Reconstruct the canonical exponent from pairwise-coprime residues."""

    normalized_moduli = _crt_moduli(moduli)
    if isinstance(residues, (str, bytes)):
        raise PRWValidationError("residues must be a sequence matching moduli")
    try:
        raw_residues = tuple(residues)
    except TypeError as exc:
        raise PRWValidationError("residues must be a sequence matching moduli") from exc
    if len(raw_residues) != len(normalized_moduli):
        raise PRWValidationError("residues must contain exactly one value per modulus")
    normalized_residues = tuple(
        _ring_integer(
            residue,
            "residues[{}]".format(index),
            modulus,
            canonical=True,
        )
        for index, (residue, modulus) in enumerate(zip(raw_residues, normalized_moduli))
    )
    modulus_product = math.prod(normalized_moduli)
    reconstruction = 0
    for residue, modulus in zip(normalized_residues, normalized_moduli):
        partial = modulus_product // modulus
        inverse = pow(partial, -1, modulus)
        reconstruction += residue * partial * inverse
    return reconstruction % modulus_product


def _phase_vector(
    phases: Sequence[int],
    layers: int,
    p: int,
    name: str = "phases",
) -> Tuple[int, ...]:
    if isinstance(phases, (str, bytes)):
        raise PRWValidationError("{} must be a sequence of integer residues".format(name))
    try:
        values = tuple(phases)
    except TypeError as exc:
        raise PRWValidationError("{} must be a sequence of integer residues".format(name)) from exc
    if len(values) != layers:
        raise PRWValidationError("{} must contain exactly one residue per layer".format(name))
    return tuple(_ring_integer(value, "{}[{}]".format(name, index), p, canonical=False) for index, value in enumerate(values))


def relative_phase_signature(phases: Sequence[int], p: int) -> Tuple[int, ...]:
    """Return a representative of ``Z_p**L`` modulo one global rotation.

    The quotient construction is valid for any cyclic modulus.  Legendre
    carrier construction imposes the stronger prime-length restriction.
    """

    ring_length = _require_ring_length(p)
    if isinstance(phases, (str, bytes)):
        raise PRWValidationError("phases must be a non-empty sequence")
    try:
        layer_count = len(phases)
    except TypeError as exc:
        raise PRWValidationError("phases must be a non-empty sequence") from exc
    if layer_count < 1:
        raise PRWValidationError("phases must be a non-empty sequence")
    normalized = _phase_vector(phases, layer_count, ring_length)
    origin = normalized[0]
    return tuple((phase - origin) % ring_length for phase in normalized)


def _finite_layers(
    value: object,
    name: str,
    *,
    expected_p: Optional[int] = None,
) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise PRWValidationError(
            "{} must be a rectangular numeric array".format(name)
        ) from exc
    if array.ndim != 2 or array.shape[0] < 1 or array.shape[1] < 3:
        raise PRWValidationError("{} must have shape (layers, p) with both dimensions non-empty".format(name))
    if expected_p is not None and array.shape[1] != expected_p:
        raise PRWValidationError("{} ring length does not match p".format(name))
    if array.dtype.kind not in "iuf" or array.dtype.kind == "b":
        raise PRWValidationError("{} must be a real numeric array".format(name))
    try:
        result = np.asarray(array, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise PRWValidationError(
            "{} must be convertible to float64".format(name)
        ) from exc
    if not np.all(np.isfinite(result)):
        raise PRWValidationError("{} must contain only finite values".format(name))
    norms = np.linalg.norm(result, axis=1)
    if not np.all(np.isfinite(norms)) or np.any(norms <= 0.0):
        raise PRWValidationError("{} must have positive finite norm in every layer".format(name))
    return result


def _polarity_mask(value: Sequence[int], layers: int, name: str = "mask") -> Tuple[int, ...]:
    if isinstance(value, (str, bytes)):
        raise PRWValidationError("{} must be a sequence of -1/+1 values".format(name))
    try:
        values = tuple(value)
    except TypeError as exc:
        raise PRWValidationError("{} must be a sequence of -1/+1 values".format(name)) from exc
    if len(values) != layers:
        raise PRWValidationError("{} must contain exactly one polarity per layer".format(name))
    normalized = []
    for index, item in enumerate(values):
        integer = _integer(item, "{}[{}]".format(name, index))
        if integer not in (-1, 1):
            raise PRWValidationError("{} values must be exactly -1 or +1".format(name))
        normalized.append(integer)
    return tuple(normalized)


def _resolve_base_carrier(
    p: int,
    carrier_family: str,
    base_carrier: Optional[object],
) -> Tuple[str, np.ndarray]:
    family = _carrier_family(carrier_family)
    if base_carrier is None:
        if family != "legendre":
            raise PRWValidationError(
                "base_carrier is required for rademacher and custom carrier families"
            )
        return family, legendre_carrier(p).astype(np.float64)

    resolved = _bipolar_carrier(base_carrier, p)
    if family == "legendre":
        _require_ideal_prime(p)
        if not np.array_equal(resolved, legendre_carrier(p)):
            raise PRWValidationError(
                "legendre base_carrier must equal the canonical Legendre carrier"
            )
    return family, resolved


def make_observation(
    waypoint_id: str,
    payload: object,
    phases: Sequence[int],
    *,
    global_shift: int = 0,
    orientation_mask: Optional[Sequence[int]] = None,
    carrier_family: str = "legendre",
    base_carrier: Optional[object] = None,
) -> RingObservation:
    """Encode a backward-compatible observation on a cyclic ring.

    A common offset in ``phases`` is folded into ``global_shift``.  The
    orientation mask acts only on the carrier/type key.  The separate identity
    payload is phase-rotated but never polarity-masked by this function.  With
    no explicit base carrier, this retains the legacy prime Legendre behavior.
    Composite-length controls must pass an explicit bipolar base carrier and
    declare ``carrier_family="rademacher"`` or ``"custom"``.
    """

    _validate_waypoint_id(waypoint_id)
    raw_payload = _finite_layers(payload, "payload")
    layers, p = raw_payload.shape
    _require_ring_length(p)
    family, resolved_base = _resolve_base_carrier(
        p,
        carrier_family,
        base_carrier,
    )
    normalized_phases = _phase_vector(phases, layers, p)
    quotient = relative_phase_signature(normalized_phases, p)
    shared = (_ring_integer(global_shift, "global_shift", p, canonical=False) + normalized_phases[0]) % p
    mask = (
        (1,) * layers
        if orientation_mask is None
        else _polarity_mask(orientation_mask, layers, "orientation_mask")
    )

    carrier_layers = np.empty((layers, p), dtype=np.float64)
    payload_layers = np.empty((layers, p), dtype=np.float64)
    for layer, relative_shift in enumerate(quotient):
        total_shift = (relative_shift + shared) % p
        carrier_layers[layer] = mask[layer] * np.roll(resolved_base, total_shift)
        payload_layers[layer] = np.roll(raw_payload[layer], total_shift)
    return RingObservation(
        waypoint_id=waypoint_id,
        p=p,
        carrier=carrier_layers,
        payload=payload_layers,
        relative_phase=quotient,
        global_shift=shared,
        orientation_mask=mask,
        carrier_family=family,
    )


def make_template(
    waypoint_id: str,
    payload: object,
    phases: Sequence[int],
    *,
    orientation_mask: Optional[Sequence[int]] = None,
    carrier_family: str = "legendre",
    base_carrier: Optional[object] = None,
) -> RingTemplate:
    """Create a canonical clean template with global phase fixed to zero."""

    _validate_waypoint_id(waypoint_id)
    raw_payload = _finite_layers(payload, "payload")
    layers, p = raw_payload.shape
    _require_ring_length(p)
    family, resolved_base = _resolve_base_carrier(
        p,
        carrier_family,
        base_carrier,
    )
    quotient = relative_phase_signature(phases, p)
    if len(quotient) != layers:
        raise PRWValidationError("phases must contain exactly one residue per layer")
    mask = (
        (1,) * layers
        if orientation_mask is None
        else _polarity_mask(orientation_mask, layers, "orientation_mask")
    )

    carrier_layers = np.vstack(
        [
            mask[layer] * np.roll(resolved_base, quotient[layer])
            for layer in range(layers)
        ]
    )
    payload_layers = np.vstack(
        [
            np.roll(raw_payload[layer], quotient[layer])
            for layer in range(layers)
        ]
    )
    return RingTemplate(
        waypoint_id=waypoint_id,
        p=p,
        base_carrier=resolved_base,
        carrier=carrier_layers,
        payload=payload_layers,
        relative_phase=quotient,
        orientation_mask=mask,
        carrier_family=family,
    )


def observe_template(
    template: RingTemplate,
    *,
    waypoint_id: Optional[str] = None,
    global_shift: int = 0,
    orientation_mask: Optional[Sequence[int]] = None,
) -> ObservedRing:
    """Apply one shared shift and relative carrier mask to a clean template."""

    if not isinstance(template, RingTemplate):
        raise PRWValidationError("template must be a RingTemplate")
    query_id = template.waypoint_id if waypoint_id is None else _validate_waypoint_id(waypoint_id)
    shift = _ring_integer(
        global_shift,
        "global_shift",
        template.p,
        canonical=False,
    )
    mask = (
        (1,) * template.layers
        if orientation_mask is None
        else _polarity_mask(
            orientation_mask,
            template.layers,
            "orientation_mask",
        )
    )
    carrier = np.vstack(
        [
            mask[layer] * np.roll(template.carrier[layer], shift)
            for layer in range(template.layers)
        ]
    )
    payload = np.roll(template.payload, shift, axis=1)
    return ObservedRing(
        waypoint_id=query_id,
        p=template.p,
        carrier=carrier,
        payload=payload,
    )


def _sylvester_hadamard(order: int) -> np.ndarray:
    matrix = np.ones((1, 1), dtype=np.int8)
    while matrix.shape[0] < order:
        matrix = np.block([[matrix, matrix], [matrix, -matrix]])
    return matrix


def policy_masks(policy: str, layers: int) -> np.ndarray:
    """Return allowed layer-polarity masks for a declared policy.

    ``typed16`` is the eight rows of ``H_8`` plus their negatives.
    ``free256`` is deliberately unconstrained and exists only as a
    multiple-comparison/false-unlock negative control.
    """

    if policy not in MASK_POLICIES:
        raise PRWValidationError("policy must be one of {}".format(", ".join(MASK_POLICIES)))
    layer_count = _integer(layers, "layers", minimum=1)
    if policy == "none":
        masks = np.ones((1, layer_count), dtype=np.int8)
    elif policy == "shared":
        masks = np.vstack(
            (np.ones(layer_count, dtype=np.int8), -np.ones(layer_count, dtype=np.int8))
        )
    elif policy == "typed16":
        if layer_count != 8:
            raise PRWValidationError("typed16 is defined only for exactly 8 layers")
        hadamard = _sylvester_hadamard(8)
        masks = np.vstack((hadamard, -hadamard))
    else:
        if layer_count != 8:
            raise PRWValidationError("free256 is defined only for exactly 8 layers")
        masks = np.array(
            [
                [1 if (mask_index >> layer) & 1 == 0 else -1 for layer in range(8)]
                for mask_index in range(256)
            ],
            dtype=np.int8,
        )
    masks.setflags(write=False)
    return masks


def layer_circular_correlations(
    query: object,
    reference: object,
    *,
    method: str = "fft",
) -> np.ndarray:
    """Return normalized per-layer scores for every reference rotation.

    Entry ``[layer, shift]`` is the cosine-like normalized dot product between
    ``query[layer]`` and ``roll(reference[layer], shift)``.
    """

    query_layers = _finite_layers(query, "query")
    reference_layers = _finite_layers(reference, "reference")
    if query_layers.shape != reference_layers.shape:
        raise PRWValidationError("query and reference must have identical shapes")
    if method not in ("direct", "fft"):
        raise PRWValidationError("method must be 'direct' or 'fft'")

    normalization = np.linalg.norm(query_layers, axis=1) * np.linalg.norm(reference_layers, axis=1)
    if method == "direct":
        layer_count, p = query_layers.shape
        correlations = np.empty((layer_count, p), dtype=np.float64)
        for shift in range(p):
            correlations[:, shift] = np.sum(
                query_layers * np.roll(reference_layers, shift, axis=1),
                axis=1,
            ) / normalization
    else:
        query_fft = np.fft.rfft(query_layers, axis=1)
        reference_fft = np.fft.rfft(reference_layers, axis=1)
        correlations = np.fft.irfft(
            query_fft * np.conjugate(reference_fft),
            n=query_layers.shape[1],
            axis=1,
        )
        correlations = correlations / normalization[:, np.newaxis]
    if not np.all(np.isfinite(correlations)):
        raise PRWValidationError("correlation produced non-finite values")
    return np.clip(correlations, -1.0, 1.0)


def _best_mask_at_scores(
    layer_scores: np.ndarray,
    policy: str,
) -> Tuple[float, Tuple[int, ...], int]:
    layers = int(layer_scores.size)
    if policy == "free256":
        if layers != 8:
            raise PRWValidationError("free256 is defined only for exactly 8 layers")
        # max_m sum_i m_i*c_i factorizes into sum_i |c_i|.  That algebraic
        # freedom is precisely why free256 is a negative control.
        mask = tuple(1 if value >= 0.0 else -1 for value in layer_scores)
        score = float(np.sum(np.abs(layer_scores)) / layers)
        return score, mask, 256
    masks = policy_masks(policy, layers)
    objectives = np.matmul(masks.astype(np.float64), layer_scores) / layers
    mask_index = int(np.argmax(objectives))
    return (
        float(objectives[mask_index]),
        tuple(int(value) for value in masks[mask_index]),
        int(masks.shape[0]),
    )


def score_shared_shift(
    query: object,
    reference: object,
    *,
    policy: str = "none",
    method: str = "fft",
) -> AlignmentScore:
    """Maximize a joint score over one shift shared by every layer."""

    correlations = layer_circular_correlations(query, reference, method=method)
    layers, p = correlations.shape
    # Vectorize over all shifts so the 4091/4691 controls remain tractable.
    # Ties retain the scalar implementation's order: first shift, then first
    # mask in the declared codebook.
    masks = policy_masks(policy, layers)
    policy_count = int(masks.shape[0])
    if policy == "free256":
        best_by_shift = np.mean(np.abs(correlations), axis=0)
        best_shift = int(np.argmax(best_by_shift))
        best_score = float(best_by_shift[best_shift])
        best_mask = tuple(
            1 if value >= 0.0 else -1
            for value in correlations[:, best_shift]
        )
    else:
        mask_scores = np.matmul(masks.astype(np.float64), correlations) / layers
        best_mask_by_shift = np.argmax(mask_scores, axis=0)
        best_by_shift = mask_scores[best_mask_by_shift, np.arange(p)]
        best_shift = int(np.argmax(best_by_shift))
        best_mask_index = int(best_mask_by_shift[best_shift])
        best_score = float(best_by_shift[best_shift])
        best_mask = tuple(int(value) for value in masks[best_mask_index])
    signed = tuple(
        float(best_mask[layer] * correlations[layer, best_shift])
        for layer in range(layers)
    )
    if not math.isfinite(best_score):
        raise PRWValidationError("shared-shift score is not finite")
    if policy == "free256":
        mechanism_id = PRW_FREE256_CONTROL_ID
    elif policy == "typed16":
        mechanism_id = PRW_TYPED16_ID
    else:
        mechanism_id = PRW_SHARED_SHIFT_ID
    return AlignmentScore(
        mechanism_id=mechanism_id,
        policy=policy,
        score=float(best_score),
        shared_shift=best_shift,
        layer_shifts=(best_shift,) * layers,
        mask=best_mask,
        per_layer_scores=signed,
        candidate_count=p * policy_count,
        negative_control=policy == "free256",
    )


def score_independent_layers(
    query: object,
    reference: object,
    *,
    allow_polarity: bool = True,
    method: str = "fft",
) -> AlignmentScore:
    """Factorize phase search per layer as an explicit degeneracy control.

    This control discards the interlayer relative-phase signature.  A perfect
    result here does not support the shared-shift PRW hypothesis.
    """

    if not isinstance(allow_polarity, (bool, np.bool_)):
        raise PRWValidationError("allow_polarity must be boolean")
    correlations = layer_circular_correlations(query, reference, method=method)
    layers, p = correlations.shape
    shifts = []
    masks = []
    scores = []
    for layer in range(layers):
        candidates = np.abs(correlations[layer]) if allow_polarity else correlations[layer]
        shift = int(np.argmax(candidates))
        raw_score = float(correlations[layer, shift])
        mask = 1 if not allow_polarity or raw_score >= 0.0 else -1
        shifts.append(shift)
        masks.append(mask)
        scores.append(mask * raw_score)
    score = float(np.mean(scores))
    if not math.isfinite(score):
        raise PRWValidationError("independent-layer score is not finite")
    return AlignmentScore(
        mechanism_id=PRW_INDEPENDENT_CONTROL_ID,
        policy="independent_polarity" if allow_polarity else "independent_phase",
        score=score,
        shared_shift=None,
        layer_shifts=tuple(shifts),
        mask=tuple(masks),
        per_layer_scores=tuple(float(value) for value in scores),
        candidate_count=(p * (2 if allow_polarity else 1)) ** layers,
        negative_control=True,
    )


def score_waypoint(
    query: object,
    candidate: object,
    *,
    policy: str = "none",
    method: str = "fft",
) -> WaypointScore:
    """Synchronize on carrier only, then score payload at that frozen shift."""

    if not isinstance(query, (RingObservation, ObservedRing)):
        raise PRWValidationError(
            "query must be an ObservedRing or legacy RingObservation"
        )
    if not isinstance(candidate, (RingTemplate, RingObservation)):
        raise PRWValidationError(
            "candidate must be a clean RingTemplate or legacy RingObservation"
        )
    if isinstance(query, ObservedRing) and not isinstance(candidate, RingTemplate):
        raise PRWValidationError(
            "ObservedRing queries require a clean RingTemplate candidate"
        )
    if query.p != candidate.p or query.layers != candidate.layers:
        raise PRWValidationError("query and candidate ring contracts must match")

    synchronization = score_shared_shift(
        query.carrier,
        candidate.carrier,
        policy=policy,
        method=method,
    )
    if synchronization.shared_shift is None:
        raise PRWValidationError("carrier synchronization did not produce a shared shift")
    payload_correlations = layer_circular_correlations(
        query.payload,
        candidate.payload,
        method=method,
    )
    payload_score = float(
        np.mean(payload_correlations[:, synchronization.shared_shift])
    )
    if not math.isfinite(payload_score):
        raise PRWValidationError("payload score is not finite")
    return WaypointScore(
        mechanism_id=PRW_WAYPOINT_SCORE_ID,
        query_id=query.waypoint_id,
        candidate_id=candidate.waypoint_id,
        policy=policy,
        carrier_score=synchronization.score,
        carrier_shift=synchronization.shared_shift,
        payload_score=payload_score,
        orientation_mask=synchronization.mask,
        negative_control=policy == "free256",
    )


def packed_byte_accounting(
    p: int,
    layers: int,
    *,
    payload_planes: int = 1,
    policy: str = "none",
    payload_encoding: str = "float64",
) -> PackedByteAccounting:
    """Account for a declared payload encoding plus a one-bit lower bound.

    The canonical bipolar base carrier is stored once.  Each payload plane contains
    ``layers * p`` values.  The default ``float64`` encoding matches the arrays
    actually retained by :class:`RingObservation`, :class:`ObservedRing`, and
    :class:`RingTemplate`.  ``bitpacked`` must be requested explicitly and is a
    serialized binary-plane model, not a claim that arbitrary real payloads
    have already been quantized without loss.

    ``theoretical_packed_*`` fields always expose the one-bit-per-value lower
    bound separately.  Only ``layers - 1`` phase residues are stored because
    the shared global rotation is quotiented out.
    """

    prime = _require_ring_length(p)
    layer_count = _integer(layers, "layers", minimum=1)
    plane_count = _integer(payload_planes, "payload_planes", minimum=1)
    mask_count = int(policy_masks(policy, layer_count).shape[0])
    if not isinstance(payload_encoding, str) or payload_encoding not in PAYLOAD_ENCODINGS:
        raise PRWValidationError(
            "payload_encoding must be one of {}".format(
                ", ".join(PAYLOAD_ENCODINGS)
            )
        )

    bits_per_residue = max(1, (prime - 1).bit_length())
    canonical_carrier_bytes = (prime + 7) // 8
    layer_plane_values = layer_count * prime
    theoretical_packed_layer_plane_bytes = (layer_plane_values + 7) // 8
    theoretical_packed_payload_bytes = (
        plane_count * layer_plane_values + 7
    ) // 8
    if payload_encoding == "float64":
        payload_bits_per_value = 64
        layer_plane_bytes = layer_plane_values * 8
        payload_bytes = plane_count * layer_plane_bytes
    else:
        payload_bits_per_value = 1
        layer_plane_bytes = theoretical_packed_layer_plane_bytes
        payload_bytes = theoretical_packed_payload_bytes
    relative_phase_bits = (layer_count - 1) * bits_per_residue
    relative_phase_bytes = (relative_phase_bits + 7) // 8
    orientation_bits = 0 if mask_count == 1 else (mask_count - 1).bit_length()
    orientation_bytes = (orientation_bits + 7) // 8
    total_bytes = (
        canonical_carrier_bytes
        + payload_bytes
        + relative_phase_bytes
        + orientation_bytes
    )
    theoretical_packed_total_bytes = (
        canonical_carrier_bytes
        + theoretical_packed_payload_bytes
        + relative_phase_bytes
        + orientation_bytes
    )
    return PackedByteAccounting(
        mechanism_id=PRW_PACKED_BYTES_ID,
        p=prime,
        layers=layer_count,
        payload_planes=plane_count,
        policy=policy,
        bits_per_residue=bits_per_residue,
        canonical_carrier_bytes=canonical_carrier_bytes,
        layer_plane_bytes=layer_plane_bytes,
        payload_bytes=payload_bytes,
        relative_phase_bytes=relative_phase_bytes,
        orientation_bytes=orientation_bytes,
        total_bytes=total_bytes,
        payload_encoding=payload_encoding,
        payload_bits_per_value=payload_bits_per_value,
        theoretical_packed_layer_plane_bytes=theoretical_packed_layer_plane_bytes,
        theoretical_packed_payload_bytes=theoretical_packed_payload_bytes,
        theoretical_packed_total_bytes=theoretical_packed_total_bytes,
    )


__all__ = [
    "AlignmentScore",
    "CARRIER_FAMILIES",
    "MASK_POLICIES",
    "ObservedRing",
    "PAYLOAD_ENCODINGS",
    "PRWValidationError",
    "PackedByteAccounting",
    "RingObservation",
    "RingTemplate",
    "WaypointScore",
    "exact_periodic_autocorrelation",
    "crt_encode_integer",
    "crt_reconstruct_integer",
    "factor_integer_exact",
    "has_ideal_legendre_autocorrelation",
    "is_primitive_root_exact",
    "is_prime_exact",
    "layer_circular_correlations",
    "legendre_carrier",
    "make_observation",
    "make_template",
    "observe_template",
    "packed_byte_accounting",
    "policy_masks",
    "relative_phase_signature",
    "rademacher_carrier",
    "score_independent_layers",
    "score_shared_shift",
    "score_waypoint",
]
