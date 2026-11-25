"""Paired identical-arm calibration with exact binomial acceptance bounds."""

import argparse
import hashlib
import hmac
import json
import math
import random
import sys
from collections.abc import Mapping, Sequence

from . import _canonical_json, _json_sha256, _read_json

ALPHA = 0.05
CONFIDENCE = 0.95
SCHEMA_VERSION = "pv7_aa_calibration_v1"


def _number(value, *, maximum=math.inf):
    if type(value) not in (int, float) or not 0 <= value <= maximum:
        raise ValueError(f"Expected a numeric observation in [0, {maximum}], got {value!r}")
    try:
        numeric = float(value)
    except OverflowError as exc:
        raise ValueError(f"Observation is not representable: {value!r}") from exc
    if not math.isfinite(numeric):
        raise ValueError(f"Observation must be finite: {value!r}")
    return numeric


def ndcg_at_10(relevance, ideal_relevance):
    """Compute graded nDCG@10 from ranked relevance and all judged relevance."""
    ranked = list(map(_number, relevance))[:10]
    ideal = sorted(map(_number, ideal_relevance), reverse=True)[:10]
    if any(value > (ideal[index] if index < len(ideal) else 0) for index, value in enumerate(
        sorted(ranked, reverse=True)
    )):
        raise ValueError("Ranked relevance is inconsistent with ideal relevance")
    if not ideal or ideal[0] == 0:
        return 0.0

    # Exp2 gains scaled by exp2(max relevance); the factor cancels in the ratio.
    scale = ideal[0]

    def dcg(grades):
        return math.fsum(
            math.exp2(value - scale) * -math.expm1(-math.log(2) * value)
            / math.log2(index + 2)
            for index, value in enumerate(grades)
        )

    return min(1.0, dcg(ranked) / dcg(ideal))


def _retrieval_observation(value):
    if isinstance(value, Mapping):
        if value.keys() != {"relevance", "ideal_relevance"}:
            raise ValueError("Ranked observations require relevance and ideal_relevance")
        return ndcg_at_10(value["relevance"], value["ideal_relevance"])
    return _number(value, maximum=1)


def _binary(value):
    if type(value) not in (bool, int) or value not in (0, 1):
        raise ValueError(f"Expected a binary disposition, got {value!r}")
    return int(value)


def _pairs(observations, transform):
    pairs = []
    for pair in observations:
        if not isinstance(pair, Sequence) or isinstance(pair, (str, bytes)) or len(pair) != 2:
            raise ValueError("Observations must contain exactly two aligned values per sampling unit")
        pairs.append((transform(pair[0]), transform(pair[1])))
    if len(pairs) < 2:
        raise ValueError("A/A calibration requires at least two paired sampling units per metric")
    return pairs


def _binomial_cdf(k, n, p):
    """Evaluate a binomial CDF by summing the tails from the mode outwards."""
    if k < 0:
        return 0.0
    if k >= n or p == 0:
        return 1.0
    if p == 1:
        return 0.0
    if k >= (n + 1) * p:
        return 1.0 - _binomial_cdf(n - k - 1, n, 1.0 - p)
    log_p, log_q = math.log(p), math.log1p(-p)
    leading = math.exp(
        math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)
        + k * log_p + (n - k) * log_q
    )
    odds = (1.0 - p) / p
    scale = term = 1.0
    for index in range(k, 0, -1):
        term *= index * odds / (n - index + 1)
        scale += term
        if term <= scale * 1e-16:
            break
    return min(1.0, leading * scale)


def _invert_binomial_cdf(k, n, tail):
    low, high = 0.0, 1.0
    for _ in range(56):
        midpoint = (low + high) / 2
        if _binomial_cdf(k, n, midpoint) > tail:
            low = midpoint
        else:
            high = midpoint
    return (low + high) / 2


def clopper_pearson(rejections, trials, confidence=CONFIDENCE):
    """Return the two-sided exact binomial confidence interval."""
    if type(trials) is not int or type(rejections) is not int or not 0 <= rejections <= trials or trials < 1:
        raise ValueError("Expected integer counts with 0 <= rejections <= trials and trials > 0")
    if type(confidence) not in (int, float) or not 0 < confidence < 1:
        raise ValueError("Confidence must be in (0, 1)")
    tail = (1 - confidence) / 2
    low = _invert_binomial_cdf(rejections - 1, trials, 1 - tail) if rejections else 0.0
    high = _invert_binomial_cdf(rejections, trials, tail) if rejections < trials else 1.0
    return low, high


def mcnemar_exact(pairs):
    """Calculate the two-sided exact McNemar p-value from paired binary outcomes."""
    wins = losses = 0
    for a, b in pairs:
        a, b = _binary(a), _binary(b)
        wins += a > b
        losses += a < b
    return min(1.0, 2 * _binomial_cdf(min(wins, losses), wins + losses, 0.5))


def _orient(pairs, mask):
    return [(b, a) if mask & (1 << index) else (a, b) for index, (a, b) in enumerate(pairs)]


def _mean_difference(pairs):
    return abs(math.fsum(a - b for a, b in pairs) / len(pairs))


def _p95_difference(pairs):
    index = (95 * len(pairs) + 99) // 100 - 1  # Nearest rank, zero-based.
    a_values, b_values = zip(*pairs)
    return abs(sorted(a_values)[index] - sorted(b_values)[index])


def _paired_randomization_p(pairs, statistic, permutations, randomizer):
    observed = statistic(pairs)
    if observed == 0:
        return 1.0
    # Swapping both arms gives the same absolute statistic. Fix the last slot.
    bits = len(pairs) - 1
    distinct = 1 << bits
    cutoff = math.nextafter(observed, -math.inf)
    if distinct <= permutations:
        count = sum(statistic(_orient(pairs, mask)) >= cutoff for mask in range(distinct))
        return count / distinct
    count = sum(
        statistic(_orient(pairs, randomizer.getrandbits(bits))) >= cutoff for _ in range(permutations)
    )
    return (count + 1) / (permutations + 1)


def _calibrate_metric(pairs, statistic, split_size, trials, permutations, seed):
    # Sample independently from the fixed same-harness pool. Keep observed
    # arm labels so drift between identical configurations is still detectable.
    split_rng = random.Random(seed)
    test_rng = random.Random(f"{seed}:permutations")
    rejections = 0
    for _ in range(trials):
        split = split_rng.sample(pairs, split_size)
        p_value = (
            mcnemar_exact(split) if statistic is None
            else _paired_randomization_p(split, statistic, permutations, test_rng)
        )
        rejections += p_value <= ALPHA
    lower, upper = clopper_pearson(rejections, trials)
    return {
        "source_pairs": len(pairs),
        "identical_measured_pairs": sum(a == b for a, b in pairs),
        "split_size": split_size,
        "rejections": rejections,
        "false_rejection_rate": rejections / trials,
        "clopper_pearson_interval": [lower, upper],
        "alpha_inside_interval": lower <= ALPHA <= upper,
    }


def run_aa_calibration(
    retrieval_pairs, latency_pairs, storage_pairs, disposition_pairs,
    *, host_id, trials=200, split_size=None, permutations=999, seed=0, signing_key=None,
):
    """Calibrate four metrics using independently measured pairs from identical arm configurations.

    Retrieval arms accept nDCG@10 values or graded relevance objects. Latency and
    storage use paired nearest-rank p95, and disposition uses exact McNemar.
    Inference is conditional on the supplied host/harness pool.
    """
    if not isinstance(host_id, str) or not host_id.strip():
        raise ValueError("host_id is required")
    if type(seed) is not int:
        raise ValueError("seed must be an integer")
    if type(trials) is not int or trials < 200:
        raise ValueError("A/A calibration requires at least 200 resampled splits")
    if type(permutations) is not int or permutations < 99:
        raise ValueError("At least 99 Monte Carlo permutations are required")
    if split_size is not None and (type(split_size) is not int or split_size < 2):
        raise ValueError("split_size must be at least two")
    if signing_key is not None and (not isinstance(signing_key, bytes) or not signing_key):
        raise ValueError("signing_key must be non-empty bytes")

    pools = {
        "retrieval_ndcg_at_10": _pairs(retrieval_pairs, _retrieval_observation),
        "latency_p95": _pairs(latency_pairs, _number),
        "storage_p95": _pairs(storage_pairs, _number),
        "disposition_mcnemar": _pairs(disposition_pairs, _binary),
    }
    statistics = (_mean_difference, _p95_difference, _p95_difference, None)
    metrics = {}
    for (metric, pairs), statistic in zip(pools.items(), statistics, strict=True):
        size = max(2, len(pairs) // 2) if split_size is None else split_size
        if size > len(pairs):
            raise ValueError(f"split_size exceeds the paired pool for {metric}")
        metrics[metric] = _calibrate_metric(pairs, statistic, size, trials, permutations, f"{seed}:{metric}")

    body = {
        "schema_version": SCHEMA_VERSION,
        "host_id": host_id,
        "input_sha256": _json_sha256(pools),
        "nominal_alpha": ALPHA,
        "confidence": CONFIDENCE,
        "trials": trials,
        "monte_carlo_permutations": permutations,
        "seed": seed,
        "metrics": metrics,
        "passed": all(metric["alpha_inside_interval"] for metric in metrics.values()),
    }
    receipt = {**body, "receipt_sha256": _json_sha256(body)}
    if signing_key is not None:
        receipt["hmac_sha256"] = hmac.new(signing_key, _canonical_json(receipt), hashlib.sha256).hexdigest()
    return receipt


def verify_aa_calibration_receipt(receipt, *, signing_key=None):
    """Check the receipt hash and, for a signed receipt, its keyed authentication."""
    if not isinstance(receipt, Mapping) or receipt.get("schema_version") != SCHEMA_VERSION:
        return False
    payload = dict(receipt)
    signature = payload.pop("hmac_sha256", None)
    digest = payload.get("receipt_sha256")
    if not isinstance(digest, str) or not digest.isascii():
        return False
    try:
        if not hmac.compare_digest(digest, _json_sha256(
            {key: value for key, value in payload.items() if key != "receipt_sha256"}
        )):
            return False
        if signing_key is not None:
            if not isinstance(signing_key, bytes) or not signing_key:
                return False
            expected = hmac.new(signing_key, _canonical_json(payload), hashlib.sha256).hexdigest()
            return isinstance(signature, str) and signature.isascii() and hmac.compare_digest(signature, expected)
        return signature is None
    except (TypeError, ValueError):
        return False


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host-id", required=True)
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--split-size", type=int)
    parser.add_argument("--permutations", type=int, default=999)
    parser.add_argument("--seed", type=int, default=0)
    options = parser.parse_args(argv)
    try:
        observations = _read_json(sys.stdin.read())
        receipt = run_aa_calibration(
            **observations, host_id=options.host_id, trials=options.trials,
            split_size=options.split_size, permutations=options.permutations, seed=options.seed,
        )
    except (TypeError, ValueError) as exc:
        parser.exit(2, f"Invalid calibration input: {exc}\n")
    print(json.dumps({"aa_calibration_receipt": receipt}, indent=2, sort_keys=True))
    return 0 if receipt["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
