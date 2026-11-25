"""CPU-only research controls for the post-v7 spine."""

import hashlib
import json
import math


def _canonical_json(value):
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _json_sha256(value):
    return hashlib.sha256(_canonical_json(value)).hexdigest()


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _invalid_json_constant(value):
    raise ValueError(f"Invalid JSON constant: {value}")


def _finite_json_float(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"JSON number is out of range: {value}")
    return result


def _read_json(source):
    return json.loads(
        source, object_pairs_hook=_unique_object, parse_constant=_invalid_json_constant,
        parse_float=_finite_json_float,
    )
