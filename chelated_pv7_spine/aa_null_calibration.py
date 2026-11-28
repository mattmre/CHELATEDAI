"""Paired identical-arm (A/A) calibration with a one-sided exact binomial acceptance rule.

Under the null hypothesis that both arms are identical, each disjoint split's
test rejects with probability at most ``ALPHA``. Conservative calibration (few
or zero rejections) therefore passes; only anti-conservative inflation fails:
the exact upper-tail binomial p-value of the rejection count against
``H0: rate <= ALPHA`` must not fall below ``INFLATION_P`` (equivalently the
one-sided ``1 - INFLATION_P`` Clopper-Pearson lower bound must not exceed
``ALPHA``). Splits are disjoint, so the trial outcomes are independent.
"""

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
INFLATION_P = 0.01
SCHEMA_VERSION = "pv7_aa_calibration_v2"


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


def ndcg_at_10_from_qrels(ranked_doc_ids, qrels):
    """Compute nDCG@10 with the IDCG taken from every judged document in ``qrels``.

    ``ranked_doc_ids`` is the system ranking; unjudged documents have grade 0.
    The ideal ranking uses the full qrels, not only the retrieved documents, so a
    run cannot inflate nDCG by omitting relevant documents from its candidates.
    """
    if not isinstance(qrels, Mapping) or not qrels:
        raise ValueError("Full-qrels nDCG requires a non-empty qrels mapping")
    if isinstance(ranked_doc_ids, (str, bytes, Mapping)):
        raise ValueError("ranked_doc_ids must be a sequence of document IDs")
    ranked = list(ranked_doc_ids)
    if len(set(ranked)) != len(ranked):
        raise ValueError("ranked_doc_ids contains duplicate documents")
    grades = {doc: _number(grade) for doc, grade in qrels.items()}
    return ndcg_at_10([grades.get(doc, 0.0) for doc in ranked], list(grades.values()))


def _retrieval_observation(value):
    if isinstance(value, Mapping):
        if value.keys() == {"ranked_doc_ids", "qrels"}:
            return ndcg_at_10_from_qrels(value["ranked_doc_ids"], value["qrels"])
        if value.keys() != {"relevance", "ideal_relevance"}:
            raise ValueError("Ranked observations require ranked_doc_ids and qrels, or relevance and ideal_relevance")
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


def binomial_upper_tail_p(rejections, trials, rate=ALPHA):
    """Exact one-sided p-value P(X >= rejections) for X ~ Binomial(trials, rate)."""
    if type(trials) is not int or type(rejections) is not int or not 0 <= rejections <= trials or trials < 1:
        raise ValueError("Expected integer counts with 0 <= rejections <= trials and trials > 0")
    if type(rate) not in (int, float) or not 0 < rate < 1:
        raise ValueError("rate must be in (0, 1)")
    return max(0.0, min(1.0, 1.0 - _binomial_cdf(rejections - 1, trials, rate)))


def clopper_pearson_lower_one_sided(rejections, trials, level=1 - INFLATION_P):
    """One-sided exact lower confidence bound for the binomial rate."""
    if type(trials) is not int or type(rejections) is not int or not 0 <= rejections <= trials or trials < 1:
        raise ValueError("Expected integer counts with 0 <= rejections <= trials and trials > 0")
    if type(level) not in (int, float) or not 0 < level < 1:
        raise ValueError("level must be in (0, 1)")
    return _invert_binomial_cdf(rejections - 1, trials, level) if rejections else 0.0


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
    # One seeded shuffle, then `trials` pairwise-disjoint splits: no sampling
    # unit appears in two trials, so trial outcomes are independent and the
    # rejection count is Binomial(trials, rate) under the null. Observed arm
    # labels are kept so drift between identical configurations is detectable.
    order = list(range(len(pairs)))
    random.Random(seed).shuffle(order)
    test_rng = random.Random(f"{seed}:permutations")
    rejections = 0
    for trial in range(trials):
        split = [pairs[index] for index in order[trial * split_size:(trial + 1) * split_size]]
        p_value = (
            mcnemar_exact(split) if statistic is None
            else _paired_randomization_p(split, statistic, permutations, test_rng)
        )
        rejections += p_value <= ALPHA
    lower, upper = clopper_pearson(rejections, trials)
    inflation_p = binomial_upper_tail_p(rejections, trials, ALPHA)
    one_sided_lower = clopper_pearson_lower_one_sided(rejections, trials, 1 - INFLATION_P)
    identical = sum(a == b for a, b in pairs)
    anti_conservative = inflation_p < INFLATION_P
    degenerate = identical == len(pairs)
    return {
        "source_pairs": len(pairs),
        "identical_measured_pairs": identical,
        "split_size": split_size,
        "disjoint_splits": True,
        "pairs_used": trials * split_size,
        "rejections": rejections,
        "false_rejection_rate": rejections / trials,
        "clopper_pearson_interval": [lower, upper],
        "clopper_pearson_lower_one_sided": one_sided_lower,
        "one_sided_confidence": 1 - INFLATION_P,
        "inflation_p_value": inflation_p,
        "anti_conservative": anti_conservative,
        "degenerate_identical_measurements": degenerate,
        "passed": not anti_conservative and not degenerate,
    }


def run_aa_calibration(
    retrieval_pairs, latency_pairs, storage_pairs, disposition_pairs,
    *, host_id, trials=200, split_size=None, permutations=999, seed=0, signing_key=None,
):
    """Calibrate four metrics using independently measured pairs from identical arm configurations.

    Retrieval arms accept nDCG@10 values, ``{"ranked_doc_ids", "qrels"}``
    objects (full-qrels IDCG) or graded relevance objects. Latency and storage
    use paired nearest-rank p95, and disposition uses exact McNemar. Each of
    the ``trials`` splits is disjoint, so every pool needs at least
    ``trials * split_size`` pairs (``split_size`` defaults to
    ``len(pool) // trials``). A metric passes unless its rejection count is
    anti-conservative (exact one-sided binomial p < ``INFLATION_P`` against
    ``rate <= ALPHA``) or every measured pair is identical (a duplicated arm,
    not an independent re-measurement). Inference is conditional on the
    supplied host/harness pool.
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
        size = len(pairs) // trials if split_size is None else split_size
        if size < 2 or size * trials > len(pairs):
            raise ValueError(
                f"{metric} needs {trials} disjoint splits of at least two pairs "
                f"({trials * max(size, 2)} pairs); got {len(pairs)}"
            )
        metrics[metric] = _calibrate_metric(pairs, statistic, size, trials, permutations, f"{seed}:{metric}")

    body = {
        "schema_version": SCHEMA_VERSION,
        "host_id": host_id,
        "input_sha256": _json_sha256(pools),
        "nominal_alpha": ALPHA,
        "confidence": CONFIDENCE,
        "acceptance_rule": "fail iff exact upper-tail binomial p(rejections | trials, alpha) < inflation_p",
        "inflation_p": INFLATION_P,
        "trials": trials,
        "monte_carlo_permutations": permutations,
        "seed": seed,
        "metrics": metrics,
        "passed": all(metric["passed"] for metric in metrics.values()),
    }
    receipt = {**body, "receipt_sha256": _json_sha256(body)}
    if signing_key is not None:
        receipt["hmac_sha256"] = hmac.new(signing_key, _canonical_json(receipt), hashlib.sha256).hexdigest()
    return receipt


DRYCHECK_HOST_ID = "pv7-w16-local-drycheck-synthetic-identical-arms"


def deterministic_drycheck_pools(seed=47, *, trials=200, split_size=10):
    """Build seeded synthetic identical-arm pools (both arms drawn from one distribution).

    This is a local drycheck of the calibration machinery, not a host
    measurement: retrieval uses full-qrels nDCG@10 over independently shuffled
    candidate rankings, latency/storage independent exponential draws, and
    disposition independent Bernoulli(0.3) outcomes.
    """
    rng = random.Random(f"aa-drycheck:{seed}")
    count = trials * split_size
    retrieval = []
    for query in range(count):
        qrels = {f"q{query}-d{doc}": rng.choice((1, 2, 3)) for doc in range(rng.randint(1, 4))}
        candidates = [*qrels, *(f"q{query}-n{doc}" for doc in range(16))]
        arms = []
        for _ in range(2):
            ranking = candidates[:]
            rng.shuffle(ranking)
            arms.append({"ranked_doc_ids": ranking[:10], "qrels": qrels})
        retrieval.append(tuple(arms))
    latency = [(100 * rng.expovariate(1), 100 * rng.expovariate(1)) for _ in range(count)]
    storage = [(4096 * rng.expovariate(1), 4096 * rng.expovariate(1)) for _ in range(count)]
    disposition = [(int(rng.random() < 0.3), int(rng.random() < 0.3)) for _ in range(count)]
    return retrieval, latency, storage, disposition


def run_deterministic_drycheck(seed=47, *, trials=200, split_size=10, permutations=99):
    """Run the A/A calibration on the deterministic synthetic pools; returns the hashed receipt."""
    pools = deterministic_drycheck_pools(seed, trials=trials, split_size=split_size)
    return run_aa_calibration(
        *pools, host_id=DRYCHECK_HOST_ID, trials=trials, split_size=split_size,
        permutations=permutations, seed=seed,
    )


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
