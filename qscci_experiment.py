"""Frozen Qwen-Scope chelated causal-intervention experiment.

The production backend is deliberately lazy: importing this module never loads
Torch, a model, or a network client.  The public verifier treats model-produced
activations, margins, KL values, and token IDs as attested leaves and
reconstructs every deterministic downstream selection, endpoint, tuning choice,
gate, and disposition.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import shutil
import stat
import struct
import sys
import time
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_EVEN
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


PROTOCOL_ID = "CHELATEDAI-QSCCI-v4"
PROTOCOL_PATH = Path("docs/research/qwen-scope-chelated-causal-intervention-delta-norm-erratum-v4-2026-08-16.md")
V3_PROTOCOL_PATH = Path("docs/research/qwen-scope-chelated-causal-intervention-numerical-erratum-v3-2026-08-16.md")
BASE_PROTOCOL_PATH = Path("docs/research/qwen-scope-chelated-causal-intervention-preregistration-2026-08-16.md")
FIXTURE_PATH = Path("docs/research/qwen-scope-chelated-causal-intervention-fixture-v1.json")
SCHEMA_PATH = Path("docs/research/qwen-scope-chelated-causal-intervention-artifact-schema-v4.json")
V3_SCHEMA_PATH = Path("docs/research/qwen-scope-chelated-causal-intervention-artifact-schema-v3.json")
BASE_SCHEMA_PATH = Path("docs/research/qwen-scope-chelated-causal-intervention-artifact-schema-v1.json")
PROTOCOL_SHA256 = "b28e978fc61238df9439348355da92d95329092794b12b754bcb6c990e8aa27e"
V3_PROTOCOL_SHA256 = "2d5375feb8f3b6b4c8da4d8640554d50f83f36fca63f696b8835f65fd3139a30"
BASE_PROTOCOL_SHA256 = "f7417b022dd93b96d523f6e8ca4a12c8a915288b5b4621e9ef130ac4f52848f1"
FIXTURE_SHA256 = "d9f873d5a0e00d0330b87e5ea053aaf0343d4c1636d7402020490719d1336f83"
SCHEMA_SHA256 = "a30fee50976ebc1d04c04d9ca7c1f1cc2c37cb1caed06ac56aae77e1d2a5eb46"
V3_SCHEMA_SHA256 = "237a3a678c2bf7cafc4c86b2ec22657f67aeab39dd49c2cc1bd687e02d38d9c8"
BASE_SCHEMA_SHA256 = "032f06fefc53aa04f13a410ce5aec93b7039c7c042bd0a2cedfe8f4745cb653a"
MODEL_REPO = "Qwen/Qwen3.5-2B-Base"
MODEL_REVISION = "b1485b2fa6dfa1287294f269f5fb618e03d52d7c"
SAE_REPO = "Qwen/SAE-Res-Qwen3.5-2B-Base-W32K-L0_100"
SAE_REVISION = "027267657257a8d490296286e8fab41e1c1a1a3d"
SAE_FILENAME = "layer11.sae.pt"
SAE_SHA256 = "d1828ace348b13cca9104f61fb47672e439e963d9d5fc5496f4c6b068a06499f"
MODEL_LAYER = 11
HIDDEN_SIZE = 2048
SAE_WIDTH = 32768
TOP_K = 100
POSITIVE_TOKEN = 6572
NEGATIVE_TOKEN = 7968
TARGET_TOKEN_TEXT = {"positive": " positive", "negative": " negative"}
SEEDS = (1701, 1709, 1721)
KS = (1, 4, 8)
ALPHAS = (0.5, 1.0, 2.0, 4.0)
MAX_WALL_SECONDS = 1800.0
MAX_RSS_BYTES = 24 * 1024**3
MIN_DISK_BYTES = 12 * 1024**3
MIN_CUDA_FREE_BYTES = 12 * 1024**3
MAX_CUDA_ALLOCATED_BYTES = 8 * 1024**3
MAX_CUDA_RESERVED_BYTES = 10 * 1024**3
QUANTUM = Decimal("0.000000001")
COSINE_TOLERANCE = 1e-12
FAMILIES = ("chelated", "contrastive", "activation_matched_random", "uniform_random")
RANDOM_FAMILIES = ("activation_matched_random", "uniform_random")
DEEPSEEK_MODEL_ID = "deepseek-v4-flash-0731"
RUNNING_METRIC = "vllm:num_requests_running"
WAITING_METRIC = "vllm:num_requests_waiting"


class QSCCIError(RuntimeError):
    """Fail-closed experiment or artifact validation error."""


def canonical_json(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def strict_json(path: Path) -> Any:
    def pairs(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise QSCCIError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    raw = path.read_bytes()
    try:
        value = json.loads(raw.decode("utf-8"), object_pairs_hook=pairs)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise QSCCIError(f"invalid strict JSON in {path}: {exc}") from exc
    return value


def q(value: float) -> Decimal:
    finite(value, "quantized scalar")
    return Decimal.from_float(float(value)).quantize(QUANTUM, rounding=ROUND_HALF_EVEN)


def q9(value: float) -> Decimal:
    """Public name for the protocol's sole nine-decimal comparison helper."""
    return q(value)


def finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise QSCCIError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise QSCCIError(f"{label} must be finite")
    return result


def canonicalize_direction_cosine(value: Any) -> float:
    """Canonicalize binary64 cosine roundoff under the frozen v3 envelope."""
    raw = finite(value, "direction cosine")
    if raw < -1.0 - COSINE_TOLERANCE or raw > 1.0 + COSINE_TOLERANCE:
        raise QSCCIError("direction cosine exceeds the frozen numerical envelope")
    if raw < -1.0:
        return -1.0
    if raw > 1.0:
        return 1.0
    return raw


def float32_leaf(value: Any, label: str) -> float:
    result = finite(value, label)
    try:
        rounded = struct.unpack("!f", struct.pack("!f", result))[0]
    except (OverflowError, struct.error) as exc:
        raise QSCCIError(f"{label} is outside FP32") from exc
    if rounded != result:
        raise QSCCIError(f"{label} is not an exact retained FP32 value")
    return result


def exact_object(value: Any, keys: set[str], label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or set(value) != keys:
        actual = sorted(value) if isinstance(value, Mapping) else type(value).__name__
        raise QSCCIError(f"{label} keys differ: expected {sorted(keys)}, got {actual}")
    return value


def plain_int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if type(value) is not int or (minimum is not None and value < minimum):
        raise QSCCIError(f"{label} must be an integer" + (f" >= {minimum}" if minimum is not None else ""))
    return value


def mean(values: Iterable[float], label: str = "mean") -> float:
    sequence = [finite(value, label) for value in values]
    if not sequence:
        raise QSCCIError(f"{label} cannot be empty")
    return math.fsum(sequence) / len(sequence)


def _assert_frozen_sources(root: Path) -> dict[str, Any]:
    def unsafe(path: Path) -> bool:
        attributes = getattr(path.lstat(), "st_file_attributes", 0)
        return path.is_symlink() or bool(attributes & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400))

    if not root.is_dir() or unsafe(root):
        raise QSCCIError("repository root must be a regular non-reparse directory")
    resolved_root = root.resolve(strict=True)
    paths = {
        "protocol": root / PROTOCOL_PATH,
        "v3_protocol": root / V3_PROTOCOL_PATH,
        "base_protocol": root / BASE_PROTOCOL_PATH,
        "fixture": root / FIXTURE_PATH,
        "schema": root / SCHEMA_PATH,
        "v3_schema": root / V3_SCHEMA_PATH,
        "base_schema": root / BASE_SCHEMA_PATH,
    }
    expected = {
        "protocol": PROTOCOL_SHA256,
        "v3_protocol": V3_PROTOCOL_SHA256,
        "base_protocol": BASE_PROTOCOL_SHA256,
        "fixture": FIXTURE_SHA256,
        "schema": SCHEMA_SHA256,
        "v3_schema": V3_SCHEMA_SHA256,
        "base_schema": BASE_SCHEMA_SHA256,
    }
    for name, path in paths.items():
        if not path.is_file() or path.is_symlink():
            raise QSCCIError(f"frozen {name} must be a regular file")
        if unsafe(path) or resolved_root not in path.resolve(strict=True).parents:
            raise QSCCIError(f"frozen {name} escapes through a reparse/symlink path")
        observed = sha256_file(path)
        if observed != expected[name]:
            raise QSCCIError(f"frozen {name} digest mismatch: {observed}")
    fixture = strict_json(paths["fixture"])
    if set(fixture) != {"prompt_template", "select", "report"}:
        raise QSCCIError("fixture root differs from frozen structure")
    if fixture["prompt_template"].count("{review}") != 1:
        raise QSCCIError("fixture prompt template must contain one literal marker")
    for split, prefix in (("select", "S"), ("report", "R")):
        rows = fixture[split]
        if len(rows) != 6:
            raise QSCCIError(f"fixture {split} must have six rows")
        for index, row in enumerate(rows, 1):
            if set(row) != {"id", "y", "canonical", "nuisance", "material"}:
                raise QSCCIError(f"fixture row {split}[{index}] keys differ")
            if row["id"] != f"{prefix}{index:02d}" or row["y"] not in (-1, 1):
                raise QSCCIError(f"fixture row {split}[{index}] identity differs")
    return fixture


def prompt_instances(fixture: Mapping[str, Any], split: str) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in fixture[split]:
        for kind in ("canonical", "nuisance", "material"):
            result.append(
                {
                    "prompt_id": f"{row['id']}:{kind}",
                    "row_id": row["id"],
                    "kind": kind,
                    "y": -row["y"] if kind == "material" else row["y"],
                    "prompt": fixture["prompt_template"].replace("{review}", row[kind]),
                }
            )
    return result


def deterministic_topk(values: Sequence[float], k: int = TOP_K) -> tuple[tuple[int, float], ...]:
    if type(k) is not int or not 0 < k <= len(values):
        raise QSCCIError("TopK count must be a positive integer no larger than the input")
    checked = [finite(value, f"preactivation[{index}]") for index, value in enumerate(values)]
    ranked = sorted(range(len(values)), key=lambda feature_id: (-checked[feature_id], feature_id))[:k]
    return tuple((feature_id, checked[feature_id]) for feature_id in ranked)


def feature_digest(family: str, seed: int, k: int, feature_id: int) -> bytes:
    """Return the frozen library-independent random-control digest."""
    if type(seed) is not int or type(k) is not int or family not in RANDOM_FAMILIES or seed not in SEEDS or k not in KS:
        raise QSCCIError("feature digest identity is outside the frozen grid")
    if type(feature_id) is not int or feature_id < 0:
        raise QSCCIError("feature_id must be a non-negative integer")
    return hashlib.sha256(f"{family}|{seed}|{k}|{feature_id}".encode("ascii")).digest()


def build_feature_records(rows: Sequence[Mapping[str, Any]], width: int = SAE_WIDTH) -> list[dict[str, Any]]:
    if len(rows) != 18:
        raise QSCCIError("SELECT activation row count must be 18")
    prompt_map: dict[str, dict[int, float]] = {}
    observed_prompt_order: list[str] = []
    for row in rows:
        if set(row) != {"prompt_id", "feature_ids", "values", "row_sha256"}:
            raise QSCCIError("SELECT activation row keys differ")
        prompt_id = row["prompt_id"]
        if not isinstance(prompt_id, str) or prompt_id in prompt_map:
            raise QSCCIError("SELECT activation prompt IDs must be unique strings")
        ids, values = row["feature_ids"], row["values"]
        if len(ids) != TOP_K or len(values) != TOP_K:
            raise QSCCIError("SELECT activation rows must contain exactly TopK entries")
        if len(set(ids)) != TOP_K or any(type(item) is not int or not 0 <= item < width for item in ids):
            raise QSCCIError("SELECT activation feature IDs are invalid")
        checked_values = [float32_leaf(value, "SELECT activation") for value in values]
        expected_topk_order = sorted(zip(ids, checked_values), key=lambda item: (-item[1], item[0]))
        if list(zip(ids, checked_values)) != expected_topk_order:
            raise QSCCIError("SELECT activation row is not ordered by descending FP32 value then feature ID")
        expected_digest = sha256_bytes(canonical_json({"prompt_id": prompt_id, "feature_ids": ids, "values": checked_values}))
        if row["row_sha256"] != expected_digest:
            raise QSCCIError("SELECT activation row digest mismatch")
        prompt_map[prompt_id] = dict(zip(ids, checked_values))
        observed_prompt_order.append(prompt_id)
    expected_ids = [f"S{index:02d}:{kind}" for index in range(1, 7) for kind in ("canonical", "nuisance", "material")]
    if observed_prompt_order != expected_ids:
        raise QSCCIError("SELECT activation rows are outside frozen order")

    records: list[dict[str, Any]] = []
    for feature_id in range(width):
        contrasts: list[float] = []
        nuisances: list[float] = []
        active_count = 0
        absolute_values: list[float] = []
        for index in range(1, 7):
            row_id = f"S{index:02d}"
            y = 1 if index in (1, 3, 5) else -1
            canonical = prompt_map[f"{row_id}:canonical"].get(feature_id, 0.0)
            nuisance = prompt_map[f"{row_id}:nuisance"].get(feature_id, 0.0)
            material = prompt_map[f"{row_id}:material"].get(feature_id, 0.0)
            contrasts.append(y * (canonical - material))
            nuisances.append(abs(canonical - nuisance))
            absolute_values.extend((abs(canonical), abs(material)))
            active_count += int(feature_id in prompt_map[f"{row_id}:canonical"])
            active_count += int(feature_id in prompt_map[f"{row_id}:material"])
        contrast = mean(contrasts, "contrast")
        material_score = abs(contrast)
        nuisance_score = mean(nuisances, "nuisance")
        eligible = active_count >= 3 and contrast != 0.0
        exclusion = None if eligible else ("ZERO_CONTRAST" if contrast == 0.0 else "ACTIVE_COUNT_BELOW_3")
        records.append(
            {
                "feature_id": feature_id,
                "active_count": active_count,
                "contrast": contrast,
                "material_score": material_score,
                "nuisance_score": nuisance_score,
                "eligible": eligible,
                "exclusion_reason": exclusion,
                "mean_absolute_activation": mean(absolute_values, "mean absolute activation"),
                "activation_decile": None,
                "sign": 1 if contrast > 0.0 else (-1 if contrast < 0.0 else 0),
                "contrastive_score": material_score,
                "chelated_score": material_score - nuisance_score,
            }
        )
    eligible = sorted(
        (record for record in records if record["eligible"]),
        key=lambda record: (record["mean_absolute_activation"], record["feature_id"]),
    )
    total = len(eligible)
    for rank, record in enumerate(eligible):
        record["activation_decile"] = min(9, (10 * rank) // total)
    if total < 8:
        raise QSCCIError("fewer than eight eligible features")
    return records


def reconstruct_feature_records(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Backward-compatible internal spelling for the frozen-width builder."""
    return build_feature_records(rows, SAE_WIDTH)


def ranked_features(records: Sequence[Mapping[str, Any]], family: str, k: int) -> list[int]:
    if family not in ("chelated", "contrastive"):
        raise QSCCIError("ranked family must be chelated or contrastive")
    score = f"{family}_score"
    eligible = [record for record in records if record["eligible"]]
    return [
        record["feature_id"]
        for record in sorted(eligible, key=lambda record: (-q(record[score]), record["feature_id"]))[:k]
    ]


def random_features(
    records: Sequence[Mapping[str, Any]], family: str, seed: int, k: int, chelated_ids: Sequence[int]
) -> tuple[list[int], list[dict[str, Any]]]:
    if family not in RANDOM_FAMILIES or seed not in SEEDS or k not in KS:
        raise QSCCIError("random draw identity is outside frozen grid")
    eligible = [record for record in records if record["eligible"]]

    def digest(feature_id: int) -> bytes:
        return feature_digest(family, seed, k, feature_id)

    selected: list[Mapping[str, Any]] = []
    if family == "uniform_random":
        selected = sorted(eligible, key=lambda record: (digest(record["feature_id"]), record["feature_id"]))[:k]
    else:
        required: dict[int, int] = {}
        by_id = {record["feature_id"]: record for record in records}
        for feature_id in chelated_ids:
            decile = by_id[feature_id]["activation_decile"]
            required[decile] = required.get(decile, 0) + 1
        for decile in sorted(required):
            pool = [record for record in eligible if record["activation_decile"] == decile]
            count = required[decile]
            if len(pool) < count:
                raise QSCCIError(f"activation decile {decile} has a sampling shortfall")
            selected.extend(sorted(pool, key=lambda record: (digest(record["feature_id"]), record["feature_id"]))[:count])
    selected = sorted(selected, key=lambda record: (digest(record["feature_id"]), record["feature_id"]))
    ranks = [
        {"feature_id": record["feature_id"], "digest": digest(record["feature_id"]).hex(), "decile": record["activation_decile"]}
        for record in selected
    ]
    return [record["feature_id"] for record in selected], ranks


def selection_grid(records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for k in KS:
        chelated = ranked_features(records, "chelated", k)
        contrastive = ranked_features(records, "contrastive", k)
        result[f"chelated|null|{k}"] = {"feature_ids": chelated, "digest_ranks": []}
        result[f"contrastive|null|{k}"] = {"feature_ids": contrastive, "digest_ranks": []}
        for family in RANDOM_FAMILIES:
            for seed in SEEDS:
                ids, ranks = random_features(records, family, seed, k, chelated)
                result[f"{family}|{seed}|{k}"] = {"feature_ids": ids, "digest_ranks": ranks}
    return result


def cell_key(family: str, seed: int | None, k: int, alpha: float, split: str, operating_point: str) -> str:
    seed_text = "null" if seed is None else str(seed)
    return f"{split}|{operating_point}|{family}|{seed_text}|{k}|{alpha:g}"


def aggregate_cell(prompt_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(prompt_records) != 36:
        raise QSCCIError("a complete cell requires 18 correct-sign and 18 wrong-sign prompt records")
    base_keys = {
        "row_id", "kind", "y", "prompt_id", "intervention_sign", "baseline_margin",
        "intervened_margin", "gain", "kl", "baseline_top1_token_id", "intervened_top1_token_id",
        "shared_delta_norm", "cast_delta_norm", "realized_update_norm",
    }
    enriched_keys = base_keys | {"cell_key", "family", "seed", "k", "alpha"}
    prefixes: set[str] = set()
    for record in prompt_records:
        if not isinstance(record, Mapping) or set(record) not in (base_keys, enriched_keys):
            raise QSCCIError("prompt record keys differ from the exact endpoint schema")
        if not isinstance(record["prompt_id"], str) or len(record["prompt_id"]) < 5:
            raise QSCCIError("prompt ID is invalid")
        prefix = record["prompt_id"][0]
        prefixes.add(prefix)
        if prefix not in {"S", "R"}:
            raise QSCCIError("prompt ID must belong to SELECT or REPORT")
        try:
            row_number = int(record["prompt_id"][1:3])
            prompt_row, kind = record["prompt_id"].split(":", 1)
        except (ValueError, IndexError) as exc:
            raise QSCCIError("prompt ID is malformed") from exc
        if row_number not in range(1, 7) or prompt_row != f"{prefix}{row_number:02d}" or record["row_id"] != prompt_row or kind not in {"canonical", "nuisance", "material"} or record["kind"] != kind:
            raise QSCCIError("prompt row/kind cross-link is invalid")
        canonical_y = 1 if row_number in (1, 3, 5) else -1
        expected_y = -canonical_y if kind == "material" else canonical_y
        if type(record["y"]) is not int or record["y"] != expected_y:
            raise QSCCIError("prompt label differs from the frozen balanced fixture")
        if type(record["intervention_sign"]) is not int or record["intervention_sign"] not in (-1, 1):
            raise QSCCIError("prompt intervention sign must be exact integer -1 or +1")
        plain_int(record["baseline_top1_token_id"], "baseline top-1 token", minimum=0)
        plain_int(record["intervened_top1_token_id"], "intervened top-1 token", minimum=0)
    if len(prefixes) != 1:
        raise QSCCIError("one cell cannot mix SELECT and REPORT prompts")
    prefix = next(iter(prefixes))
    frozen_prompt_order = [f"{prefix}{index:02d}:{kind}" for index in range(1, 7) for kind in ("canonical", "nuisance", "material")]
    expected_record_order = [(prompt_id, sign) for sign in (1, -1) for prompt_id in frozen_prompt_order]
    if [(record["prompt_id"], record["intervention_sign"]) for record in prompt_records] != expected_record_order:
        raise QSCCIError("prompt records are outside frozen prompt/intervention-sign order")
    correct = [record for record in prompt_records if record["intervention_sign"] == 1]
    wrong = [record for record in prompt_records if record["intervention_sign"] == -1]
    if len(correct) != 18 or len(wrong) != 18:
        raise QSCCIError("cell intervention signs are incomplete")
    correct_by_id = {record["prompt_id"]: record for record in correct}
    wrong_by_id = {record["prompt_id"]: record for record in wrong}
    if len(correct_by_id) != 18 or len(wrong_by_id) != 18 or set(correct_by_id) != set(wrong_by_id):
        raise QSCCIError("cell prompt IDs are incomplete or duplicated")
    if len({record["shared_delta_norm"] for record in prompt_records}) != 1:
        raise QSCCIError("shared FP32 delta norm must be singular within a cell")
    if len({record["cast_delta_norm"] for record in prompt_records}) != 1:
        raise QSCCIError("cast BF16 delta norm must be invariant within a cell")
    for record in prompt_records:
        for field in ("baseline_margin", "intervened_margin", "gain", "kl", "shared_delta_norm", "cast_delta_norm", "realized_update_norm"):
            finite(record[field], f"prompt record {field}")
        expected_gain = record["y"] * (record["intervened_margin"] - record["baseline_margin"])
        if record["gain"] != expected_gain:
            raise QSCCIError("prompt gain contradicts retained margins")
        if record["kl"] < 0.0 or record["shared_delta_norm"] <= 0.0 or record["cast_delta_norm"] <= 0.0 or record["realized_update_norm"] <= 0.0:
            raise QSCCIError("prompt KL/update norm violates frozen range")
    correct_gain = mean((record["gain"] for record in correct), "correct gain")
    wrong_gain = mean((record["gain"] for record in wrong), "wrong gain")
    mismatches: list[float] = []
    complete_rows = 0
    for index in range(1, 7):
        prefix = correct[0]["prompt_id"][0]
        canonical = correct_by_id[f"{prefix}{index:02d}:canonical"]["gain"]
        nuisance = correct_by_id[f"{prefix}{index:02d}:nuisance"]["gain"]
        material = correct_by_id[f"{prefix}{index:02d}:material"]["gain"]
        wrong_canonical = wrong_by_id[f"{prefix}{index:02d}:canonical"]["gain"]
        wrong_material = wrong_by_id[f"{prefix}{index:02d}:material"]["gain"]
        mismatches.append(abs(canonical - nuisance))
        row_bidirectional = ((canonical + material) - (wrong_canonical + wrong_material)) / 2.0
        complete_rows += int(q(canonical) >= q(0.02) and q(material) >= q(0.02) and q(row_bidirectional) >= q(0.04))
    baseline_accuracy = mean(
        (int(record["baseline_top1_token_id"] == (POSITIVE_TOKEN if record["y"] == 1 else NEGATIVE_TOKEN)) for record in correct),
        "baseline accuracy",
    )
    intervened_accuracy = mean(
        (int(record["intervened_top1_token_id"] == (POSITIVE_TOKEN if record["y"] == 1 else NEGATIVE_TOKEN)) for record in correct),
        "intervened accuracy",
    )
    collateral = mean(
        (
            int(
                record["baseline_top1_token_id"] not in (POSITIVE_TOKEN, NEGATIVE_TOKEN)
                and record["intervened_top1_token_id"] not in (POSITIVE_TOKEN, NEGATIVE_TOKEN)
                and record["baseline_top1_token_id"] != record["intervened_top1_token_id"]
            )
            for record in correct
        ),
        "collateral fraction",
    )
    return {
        "mean_correct_gain": correct_gain,
        "mean_wrong_gain": wrong_gain,
        "bidirectional_causal_contrast": correct_gain - wrong_gain,
        "baseline_accuracy": baseline_accuracy,
        "intervened_accuracy": intervened_accuracy,
        "canonical_nuisance_mismatch": mean(mismatches, "canonical nuisance mismatch"),
        "material_response_completeness": complete_rows / 6.0,
        "mean_kl": mean((record["kl"] for record in correct), "mean KL"),
        "max_kl": max(finite(record["kl"], "KL") for record in correct),
        "outside_target_collateral_fraction": collateral,
    }


def compute_cell_endpoints(prompt_records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Public endpoint reducer; all reductions use the frozen retained leaves."""
    return aggregate_cell(prompt_records)


def tuning_choice(cells: Sequence[Mapping[str, Any]], family: str, seed: int | None) -> dict[str, Any]:
    candidates = [
        cell for cell in cells
        if cell["split"] == "SELECT" and cell["operating_point"] == "grid" and cell["family"] == family and cell["seed"] == seed
    ]
    if len(candidates) != 12:
        raise QSCCIError(f"{family}/{seed} tuning grid must contain 12 cells")
    chosen = min(
        candidates,
        key=lambda cell: (
            -q(cell["endpoints"]["bidirectional_causal_contrast"]),
            q(cell["endpoints"]["canonical_nuisance_mismatch"]),
            q(cell["endpoints"]["mean_kl"]),
            cell["alpha"],
            cell["k"],
        ),
    )
    return {"k": chosen["k"], "alpha": chosen["alpha"], "cell_key": chosen["cell_key"], "feature_ids": chosen["feature_ids"]}


def choose_cell(cells: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Choose one cell from a single-family 12-cell SELECT grid."""
    if not cells:
        raise QSCCIError("tuning grid cannot be empty")
    family = cells[0].get("family")
    seed = cells[0].get("seed")
    if any(cell.get("family") != family or cell.get("seed") != seed for cell in cells):
        raise QSCCIError("choose_cell accepts exactly one family/seed grid")
    identities = [(cell.get("k"), cell.get("alpha")) for cell in cells]
    if len(cells) != 12 or set(identities) != {(k, alpha) for k in KS for alpha in ALPHAS}:
        raise QSCCIError("choose_cell requires every frozen k/alpha cell exactly once")
    return tuning_choice(cells, family, seed)


def evaluate_disposition(payload: Mapping[str, Any]) -> str:
    """Apply frozen fail-closed status precedence to a reconstructed payload."""
    failures = payload.get("failures", [])
    integrity = payload.get("integrity_valid", True)
    resources = payload.get("resources_valid", True)
    lifecycle = payload.get("service_lifecycle_valid", True)
    if failures or integrity is not True or resources is not True or lifecycle is not True:
        return "INVALID_RUN"
    gates = payload.get("gates")
    if not isinstance(gates, Mapping):
        return "INVALID_RUN"
    expected_gate_names = {
        "select_baseline_accuracy_at_least_0_75", "report_baseline_accuracy_at_least_0_75",
        "correct_positive_wrong_negative", "contrastive_advantage_at_least_0_05",
        "all_random_advantages_at_least_0_05", "mismatch_no_greater_than_controls",
        "material_completeness_at_least_5_6", "material_completeness_no_lower_than_controls",
        "mean_kl_at_most_0_05", "max_kl_at_most_0_25", "collateral_fraction_at_most_1_6",
        "treatment_separation",
    }
    if set(gates) != expected_gate_names:
        return "INVALID_RUN"
    if not gates.get("select_baseline_accuracy_at_least_0_75") or not gates.get("report_baseline_accuracy_at_least_0_75"):
        if any(type(gates.get(name)) is not bool for name in expected_gate_names - {"treatment_separation"}):
            return "INVALID_RUN"
        return "INVALID_TASK"
    separation = gates.get("treatment_separation")
    if not isinstance(separation, Mapping) or not separation or any(type(value) is not bool for value in separation.values()):
        return "INVALID_RUN"
    scientific = [value for key, value in gates.items() if key not in {"select_baseline_accuracy_at_least_0_75", "report_baseline_accuracy_at_least_0_75", "treatment_separation"}]
    if any(type(value) is not bool for value in scientific):
        return "INVALID_RUN"
    if not all(value is True for value in scientific) or not all(value is True for value in separation.values()):
        return "DOES_NOT_SURVIVE_SMALL_LABEL_ORACLE_FIXTURE"
    return "SURVIVES_SMALL_LABEL_ORACLE_FIXTURE"


def reference_delta_components(direction: Any, alpha: float, scale: float) -> tuple[Any, Any]:
    """Construct verifier reference components for the frozen v1 delta formula."""
    import torch

    if direction.dtype != torch.float32 or direction.ndim != 1:
        raise QSCCIError("direction must be a rank-1 FP32 tensor")
    alpha_value, scale_value = finite(alpha, "alpha"), finite(scale, "scale")
    if alpha_value <= 0.0 or scale_value <= 0.0 or not torch.isfinite(direction).all():
        raise QSCCIError("canonical delta norm inputs must be finite and positive")
    direction_cpu = direction.detach().to(device="cpu", dtype=torch.float32)
    alpha_cpu = torch.tensor(alpha_value, dtype=torch.float32, device="cpu")
    scale_cpu = torch.tensor(scale_value, dtype=torch.float32, device="cpu")
    delta_cpu = alpha_cpu * scale_cpu * direction_cpu
    return delta_cpu, delta_cpu.to(torch.bfloat16)


def canonical_delta_norms(delta_fp32: Any, delta_bf16: Any) -> tuple[float, float]:
    """Reduce exact actual/reference delta components on the v4 CPU path."""
    import torch

    if (
        delta_fp32.dtype != torch.float32
        or delta_bf16.dtype != torch.bfloat16
        or delta_fp32.ndim != 1
        or delta_bf16.ndim != 1
        or delta_fp32.shape != delta_bf16.shape
    ):
        raise QSCCIError("canonical delta norm vectors have invalid shape or dtype")
    fp32_cpu = delta_fp32.detach().to(device="cpu", dtype=torch.float32)
    bf16_cpu = delta_bf16.detach().to(device="cpu", dtype=torch.bfloat16)
    if not torch.isfinite(fp32_cpu).all() or not torch.isfinite(bf16_cpu.float()).all():
        raise QSCCIError("canonical delta norm vectors must be finite")
    shared = float(torch.linalg.vector_norm(fp32_cpu))
    cast = float(torch.linalg.vector_norm(bf16_cpu.float()))
    if not math.isfinite(shared) or not math.isfinite(cast) or shared <= 0.0 or cast <= 0.0:
        raise QSCCIError("canonical delta norms must be finite and positive")
    return shared, cast


def _same_tensor_bits(left: Any, right: Any) -> bool:
    """Compare frozen floating tensors without collapsing signed zero."""
    import torch

    if left.dtype != right.dtype or left.shape != right.shape:
        return False
    integer_dtype = torch.int32 if left.dtype == torch.float32 else torch.int16
    return bool(torch.equal(left.contiguous().view(integer_dtype), right.contiguous().view(integer_dtype)))


def apply_bf16_update(
    hidden: Any, direction: Any, labels: Any, alpha: float, scale: float, last_token_indices: Any
) -> tuple[Any, dict[str, Any]]:
    """Apply the exact BF16-preserving hook update and return norm evidence."""
    import torch

    if hidden.dtype != torch.bfloat16 or hidden.ndim != 3:
        raise QSCCIError("hidden must be a rank-3 BF16 tensor")
    if direction.dtype != torch.float32 or direction.ndim != 1 or direction.shape[0] != hidden.shape[-1]:
        raise QSCCIError("direction must be a hidden-width FP32 vector")
    labels = torch.as_tensor(labels, device=hidden.device, dtype=torch.float32)
    positions = torch.as_tensor(last_token_indices, device=hidden.device, dtype=torch.long)
    if tuple(labels.shape) != (hidden.shape[0],) or tuple(positions.shape) != (hidden.shape[0],):
        raise QSCCIError("labels and last-token indices must match batch size")
    if not bool(torch.all((labels == 1) | (labels == -1))):
        raise QSCCIError("labels must contain only -1 and +1")
    if bool(torch.any(positions < 0)) or bool(torch.any(positions >= hidden.shape[1])):
        raise QSCCIError("last-token indices are outside the sequence")
    alpha_value, scale_value = finite(alpha, "alpha"), finite(scale, "scale")
    if alpha_value <= 0.0 or scale_value <= 0.0 or not torch.isfinite(direction).all():
        raise QSCCIError("update parameters must be finite and positive")
    direction_device = direction.to(hidden.device)
    delta_fp32 = labels[:, None] * alpha_value * scale_value * direction_device[None, :]
    delta_bf16 = delta_fp32.to(dtype=hidden.dtype)
    reference_fp32, reference_bf16 = reference_delta_components(direction, alpha_value, scale_value)
    signs_cpu = labels.detach().to(device="cpu")[:, None]
    expected_fp32 = signs_cpu * reference_fp32[None, :]
    expected_bf16 = expected_fp32.to(torch.bfloat16)
    actual_fp32 = delta_fp32.detach().to(device="cpu")
    actual_bf16 = delta_bf16.detach().to(device="cpu")
    if not _same_tensor_bits(actual_fp32, expected_fp32) or not _same_tensor_bits(actual_bf16, expected_bf16):
        raise QSCCIError("actual CUDA delta components differ from verifier reconstruction")
    canonical_norms = [
        canonical_delta_norms(actual_fp32[index], actual_bf16[index]) for index in range(hidden.shape[0])
    ]
    if len({item[0] for item in canonical_norms}) != 1 or len({item[1] for item in canonical_norms}) != 1:
        raise QSCCIError("signed delta norms must be invariant across prompts")
    shared_delta_norm = canonical_norms[0][0]
    batch = torch.arange(hidden.shape[0], device=hidden.device)
    before = hidden[batch, positions]
    after = before + delta_bf16
    modified = hidden.clone()
    modified[batch, positions] = after
    evidence = {
        "shared_delta_norm": shared_delta_norm,
        "cast_delta_norms": [item[1] for item in canonical_norms],
        "realized_update_norms": [float(value) for value in torch.linalg.vector_norm((after - before).float(), dim=1).cpu().tolist()],
    }
    if any(not math.isfinite(value) or value <= 0.0 for key in ("cast_delta_norms", "realized_update_norms") for value in evidence[key]):
        raise QSCCIError("BF16 update vanished or became nonfinite")
    return modified, evidence


def _primary_report_cells(cells: Sequence[Mapping[str, Any]], choices: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    by_key = {cell["cell_key"]: cell for cell in cells}
    result: dict[str, Mapping[str, Any]] = {}
    for choice_id, choice in choices.items():
        family, seed_text = choice_id.split("|", 1)
        seed = None if seed_text == "null" else int(seed_text)
        key = cell_key(family, seed, choice["k"], choice["alpha"], "REPORT", "primary")
        if key not in by_key:
            raise QSCCIError(f"missing primary REPORT cell {key}")
        result[choice_id] = by_key[key]
    return result


def compute_gates(cells: Sequence[Mapping[str, Any]], choices: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    primary = _primary_report_cells(cells, choices)
    candidate = primary["chelated|null"]
    contrastive = primary["contrastive|null"]
    controls = [contrastive] + [cell for key, cell in primary.items() if key.startswith(RANDOM_FAMILIES)]
    candidate_ep = candidate["endpoints"]
    task_select = next(cell["endpoints"]["baseline_accuracy"] for cell in cells if cell["split"] == "SELECT")
    task_report = candidate_ep["baseline_accuracy"]
    gates = {
        "select_baseline_accuracy_at_least_0_75": q(task_select) >= q(0.75),
        "report_baseline_accuracy_at_least_0_75": q(task_report) >= q(0.75),
        "correct_positive_wrong_negative": q(candidate_ep["mean_correct_gain"]) > q(0.0) and q(candidate_ep["mean_wrong_gain"]) < q(0.0),
        "contrastive_advantage_at_least_0_05": q(candidate_ep["bidirectional_causal_contrast"] - contrastive["endpoints"]["bidirectional_causal_contrast"]) >= q(0.05),
        "all_random_advantages_at_least_0_05": all(q(candidate_ep["bidirectional_causal_contrast"] - control["endpoints"]["bidirectional_causal_contrast"]) >= q(0.05) for control in controls[1:]),
        "mismatch_no_greater_than_controls": all(q(candidate_ep["canonical_nuisance_mismatch"] - control["endpoints"]["canonical_nuisance_mismatch"]) <= q(0.0) for control in controls),
        "material_completeness_at_least_5_6": q(candidate_ep["material_response_completeness"]) >= q(5.0 / 6.0),
        "material_completeness_no_lower_than_controls": all(q(candidate_ep["material_response_completeness"] - control["endpoints"]["material_response_completeness"]) >= q(0.0) for control in controls),
        "mean_kl_at_most_0_05": q(candidate_ep["mean_kl"]) <= q(0.05),
        "max_kl_at_most_0_25": q(candidate_ep["max_kl"]) <= q(0.25),
        "collateral_fraction_at_most_1_6": q(candidate_ep["outside_target_collateral_fraction"]) <= q(1.0 / 6.0),
    }
    expected_control_keys = {
        "contrastive|null",
        *(f"{family}|{seed}" for family in RANDOM_FAMILIES for seed in SEEDS),
    }
    cosines = candidate.get("direction_cosines")
    if not isinstance(cosines, Mapping) or set(cosines) != expected_control_keys:
        raise QSCCIError("candidate direction cosines must contain the exact seven control keys")
    separation: dict[str, bool] = {}
    for key, control in primary.items():
        if key == "chelated|null":
            continue
        same = set(candidate["feature_ids"]) == set(control["feature_ids"])
        cosine = abs(canonicalize_direction_cosine(cosines[key]))
        separation[key] = not (same or q(cosine) > q(0.999))
    gates["treatment_separation"] = separation
    task_valid = gates["select_baseline_accuracy_at_least_0_75"] and gates["report_baseline_accuracy_at_least_0_75"]
    science_valid = all(value for key, value in gates.items() if key != "treatment_separation") and all(separation.values())
    return gates, ("SURVIVES_SMALL_LABEL_ORACLE_FIXTURE" if task_valid and science_valid else ("INVALID_TASK" if not task_valid else "DOES_NOT_SURVIVE_SMALL_LABEL_ORACLE_FIXTURE"))


def artifact_digest(artifact: Mapping[str, Any]) -> str:
    payload = dict(artifact)
    payload["artifact_digest"] = "0" * 64
    return sha256_bytes(canonical_json(payload))


def _validate_command_evidence(value: Any, label: str, *, require_stdout: bool) -> None:
    exact_object(value, {"argv_sha256", "returncode", "stdout_sha256", "stderr_sha256", "stdout"}, label)
    if value["returncode"] != 0 or any(not isinstance(value[field], str) or len(value[field]) != 64 or any(character not in "0123456789abcdef" for character in value[field]) for field in ("argv_sha256", "stdout_sha256", "stderr_sha256")):
        raise QSCCIError(f"{label} is not successful digest-bound command evidence")
    if not isinstance(value["stdout"], str) or (require_stdout and not value["stdout"]):
        raise QSCCIError(f"{label} stdout evidence is invalid")


def _validate_evidence_sha(value: Mapping[str, Any], label: str) -> None:
    digest = value.get("sha256")
    if not isinstance(digest, str) or len(digest) != 64 or any(character not in "0123456789abcdef" for character in digest):
        raise QSCCIError(f"{label} SHA-256 evidence is invalid")


def _validate_service_lifecycle(lifecycle: Any) -> None:
    exact_object(lifecycle, {"contract", "before", "stop", "after"}, "service lifecycle")
    contract = exact_object(
        lifecycle["contract"],
        {
            "stop_required", "container_identity_commands", "idle_endpoint", "stop_commands", "start_commands",
            "health_endpoint", "models_endpoint", "completion_endpoint", "after_idle_endpoint",
        },
        "service contract",
    )
    if type(contract["stop_required"]) is not bool or not isinstance(contract["container_identity_commands"], list) or len(contract["container_identity_commands"]) != 2:
        raise QSCCIError("service contract stop/container identity fields are invalid")
    if contract["container_identity_commands"][0] == contract["container_identity_commands"][1]:
        raise QSCCIError("service contract repeats one container identity command")
    exact_object(contract["health_endpoint"], {"url", "method", "expected_status", "expected_body"}, "health contract")
    exact_object(contract["models_endpoint"], {"url", "method", "expected_status", "expected_model_id"}, "models contract")
    exact_object(contract["completion_endpoint"], {"url", "method", "json", "expected_status", "expected_model_id", "expected_text"}, "completion contract")
    for key in ("idle_endpoint", "after_idle_endpoint"):
        exact_object(contract[key], {"url", "method", "expected_status", "running_metric", "waiting_metric"}, f"{key} contract")
    if contract["health_endpoint"]["method"] != "GET" or contract["health_endpoint"]["expected_status"] != 200 or contract["health_endpoint"]["expected_body"] != "":
        raise QSCCIError("health contract is not exact GET/200/empty")
    expected_model_id = contract["models_endpoint"]["expected_model_id"]
    if contract["models_endpoint"]["method"] != "GET" or contract["models_endpoint"]["expected_status"] != 200 or expected_model_id != DEEPSEEK_MODEL_ID or contract["completion_endpoint"]["expected_model_id"] != expected_model_id:
        raise QSCCIError("service model identity contract is invalid")
    if contract["completion_endpoint"]["method"] != "POST" or contract["completion_endpoint"]["expected_status"] != 200 or not isinstance(contract["completion_endpoint"]["expected_text"], str) or not contract["completion_endpoint"]["expected_text"]:
        raise QSCCIError("service completion semantic contract is invalid")
    request = contract["completion_endpoint"]["json"]
    if not isinstance(request, Mapping) or set(request) != {"model", "prompt", "temperature", "max_tokens"} or request["model"] != DEEPSEEK_MODEL_ID or not isinstance(request["prompt"], str) or not request["prompt"] or request["temperature"] != 0 or type(request["max_tokens"]) is not int or not 1 <= request["max_tokens"] <= 32:
        raise QSCCIError("service completion request is not deterministic and model-bound")
    for command_group in ("container_identity_commands", "stop_commands", "start_commands"):
        if not isinstance(contract[command_group], list) or any(not isinstance(command, list) or not command or any(not isinstance(item, str) or not item for item in command) for command in contract[command_group]):
            raise QSCCIError(f"service {command_group} is not an exact argv-array list")
    if contract["stop_required"] != bool(contract["stop_commands"] and contract["start_commands"]):
        raise QSCCIError("service stop-required flag contradicts stop/start commands")
    for key in ("idle_endpoint", "after_idle_endpoint"):
        if (contract[key]["running_metric"], contract[key]["waiting_metric"]) != (RUNNING_METRIC, WAITING_METRIC):
            raise QSCCIError("service idle contract uses non-frozen metric identities")
    if contract["idle_endpoint"]["url"] != contract["after_idle_endpoint"]["url"]:
        raise QSCCIError("before/after idle evidence must use the same endpoint")
    before, stop, after = lifecycle["before"], lifecycle["stop"], lifecycle["after"]
    exact_object(before, {"status", "service_stopped", "container_identities", "observations"}, "service before")
    if before["status"] != "AVAILABLE" or before["service_stopped"] is not False:
        raise QSCCIError("service before-state must be available and unstopped")
    identities = before["container_identities"]
    if not isinstance(identities, list) or len(identities) != 2:
        raise QSCCIError("service before-state must retain exactly two container identities")
    for identity in identities:
        _validate_command_evidence(identity, "container identity", require_stdout=True)
    if identities[0]["stdout"] == identities[1]["stdout"]:
        raise QSCCIError("service evidence does not identify two distinct containers")
    for command, evidence in zip(contract["container_identity_commands"], identities):
        if evidence["argv_sha256"] != sha256_bytes(canonical_json(command)):
            raise QSCCIError("container identity evidence does not bind the contracted argv")
    observations = exact_object(before["observations"], {"health", "models", "idle"}, "before observations")
    exact_object(observations["health"], {"url", "method", "status", "body", "sha256"}, "before health")
    _validate_evidence_sha(observations["health"], "before health")
    if observations["health"]["method"] != "GET" or observations["health"]["status"] != 200 or observations["health"]["body"] != "":
        raise QSCCIError("before health evidence is invalid")
    if observations["health"]["url"] != contract["health_endpoint"]["url"]:
        raise QSCCIError("before health URL differs from contract")
    exact_object(observations["models"], {"url", "method", "status", "model_id", "sha256"}, "before models")
    _validate_evidence_sha(observations["models"], "before models")
    if observations["models"]["method"] != "GET" or observations["models"]["status"] != 200 or not isinstance(observations["models"]["model_id"], str) or not observations["models"]["model_id"]:
        raise QSCCIError("before model evidence is invalid")
    if observations["models"]["model_id"] != expected_model_id:
        raise QSCCIError("before model evidence differs from the contracted identity")
    if observations["models"]["url"] != contract["models_endpoint"]["url"]:
        raise QSCCIError("before models URL differs from contract")
    exact_object(observations["idle"], {"url", "method", "status", "metric_sums", "sha256"}, "before idle")
    _validate_evidence_sha(observations["idle"], "before idle")
    if observations["idle"]["method"] != "GET" or observations["idle"]["status"] != 200 or not isinstance(observations["idle"]["metric_sums"], Mapping) or len(observations["idle"]["metric_sums"]) != 2 or any(value != 0.0 for value in observations["idle"]["metric_sums"].values()):
        raise QSCCIError("before idle counters are not exactly zero")
    if observations["idle"]["url"] != contract["idle_endpoint"]["url"] or set(observations["idle"]["metric_sums"]) != {RUNNING_METRIC, WAITING_METRIC}:
        raise QSCCIError("before idle evidence differs from the contracted endpoint/metrics")
    stopped = stop.get("service_stopped") if isinstance(stop, Mapping) else None
    if stopped is True:
        exact_object(stop, {"service_stopped", "commands"}, "service stop")
        if not isinstance(stop["commands"], list) or not stop["commands"]:
            raise QSCCIError("stopped lifecycle lacks stop-command evidence")
        for command in stop["commands"]:
            _validate_command_evidence(command, "stop command", require_stdout=False)
        if len(stop["commands"]) != len(contract["stop_commands"]) or any(evidence["argv_sha256"] != sha256_bytes(canonical_json(command)) for command, evidence in zip(contract["stop_commands"], stop["commands"])):
            raise QSCCIError("stop evidence does not bind every contracted argv")
    elif stopped is False:
        exact_object(stop, {"service_stopped", "reason"}, "service no-stop")
        if stop["reason"] != "not_required":
            raise QSCCIError("no-stop reason differs from the closed contract")
    else:
        raise QSCCIError("service stop state is invalid")
    exact_object(
        after,
        {"status", "service_stopped", "restoration_verified", "start_commands", "verification_observations", "same_container_identity_evidence"},
        "service after",
    )
    expected_status = "RESTORED" if stopped else "NOT_APPLICABLE"
    if after["status"] != expected_status or after["service_stopped"] is not stopped or after["restoration_verified"] is not True:
        raise QSCCIError("service after-state contradicts stop/restoration state")
    if not isinstance(after["start_commands"], list) or bool(after["start_commands"]) is not stopped:
        raise QSCCIError("service start-command evidence contradicts stop state")
    for command in after["start_commands"]:
        _validate_command_evidence(command, "start command", require_stdout=False)
    if len(after["start_commands"]) != len(contract["start_commands"]) or any(evidence["argv_sha256"] != sha256_bytes(canonical_json(command)) for command, evidence in zip(contract["start_commands"], after["start_commands"])):
        raise QSCCIError("start evidence does not bind every contracted argv")
    if after["same_container_identity_evidence"] != identities:
        raise QSCCIError("after container identities differ from before-state identities")
    verified = exact_object(after["verification_observations"], {"health", "models", "completion", "idle"}, "after observations")
    exact_object(verified["health"], {"url", "method", "status", "body", "sha256"}, "after health")
    _validate_evidence_sha(verified["health"], "after health")
    if verified["health"]["method"] != "GET" or verified["health"]["status"] != 200 or verified["health"]["body"] != "":
        raise QSCCIError("after health evidence is invalid")
    if verified["health"]["url"] != contract["health_endpoint"]["url"]:
        raise QSCCIError("after health URL differs from contract")
    exact_object(verified["models"], {"url", "method", "status", "model_id", "sha256"}, "after models")
    exact_object(verified["completion"], {"url", "method", "status", "model_id", "text", "sha256"}, "after completion")
    _validate_evidence_sha(verified["models"], "after models")
    _validate_evidence_sha(verified["completion"], "after completion")
    model_id = observations["models"]["model_id"]
    if verified["models"].get("model_id") != model_id or verified["completion"].get("model_id") != model_id or verified["completion"].get("method") != "POST" or verified["completion"].get("status") != 200 or not isinstance(verified["completion"].get("text"), str) or not verified["completion"]["text"]:
        raise QSCCIError("after model/completion semantic evidence differs from pinned before-state")
    if verified["completion"]["text"] != contract["completion_endpoint"]["expected_text"]:
        raise QSCCIError("completion evidence differs from the contracted semantic result")
    if verified["models"]["url"] != contract["models_endpoint"]["url"] or verified["completion"]["url"] != contract["completion_endpoint"]["url"]:
        raise QSCCIError("after models/completion URL differs from contract")
    exact_object(verified["idle"], {"url", "method", "status", "metric_sums", "sha256"}, "after idle")
    _validate_evidence_sha(verified["idle"], "after idle")
    if verified["idle"]["method"] != "GET" or verified["idle"]["status"] != 200 or verified["idle"]["metric_sums"] != observations["idle"]["metric_sums"] or any(value != 0.0 for value in verified["idle"]["metric_sums"].values()):
        raise QSCCIError("after idle counters are not exactly the zero before-state counters")
    if verified["idle"]["url"] != contract["after_idle_endpoint"]["url"] or set(verified["idle"]["metric_sums"]) != {RUNNING_METRIC, WAITING_METRIC}:
        raise QSCCIError("after idle evidence differs from the contracted endpoint/metrics")


def _verify_artifact_impl(
    artifact: Mapping[str, Any], *, root: Path, require_pinned_sae_provenance: bool
) -> dict[str, Any]:
    fixture = _assert_frozen_sources(root)
    required = {
        "protocol_id", "status", "scientific_claim_status", "novelty_claim_status", "protocol_file_sha256",
        "fixture_sha256", "schema_file_sha256", "model_identity", "service_lifecycle", "resources",
        "select_activation_rows", "feature_records", "choices", "prompt_records", "cells", "aggregates",
        "gates", "disposition", "failures", "artifact_digest",
    }
    if set(artifact) != required:
        raise QSCCIError("artifact envelope keys differ from schema")
    if artifact["protocol_id"] != PROTOCOL_ID or artifact["status"] != "COMPLETE":
        raise QSCCIError("artifact protocol/status differs")
    if artifact["scientific_claim_status"] != "UNCONFIRMED" or artifact["novelty_claim_status"] != "UNCONFIRMED":
        raise QSCCIError("claim status must remain UNCONFIRMED")
    if (artifact["protocol_file_sha256"], artifact["fixture_sha256"], artifact["schema_file_sha256"]) != (PROTOCOL_SHA256, FIXTURE_SHA256, SCHEMA_SHA256):
        raise QSCCIError("artifact source digests differ")
    identity = artifact["model_identity"]
    expected_identity = {
        "model_repo": MODEL_REPO, "model_revision": MODEL_REVISION, "sae_repo": SAE_REPO,
        "sae_revision": SAE_REVISION, "sae_filename": SAE_FILENAME, "sae_sha256": SAE_SHA256,
        "layer_index": MODEL_LAYER, "hidden_size": HIDDEN_SIZE, "sae_width": SAE_WIDTH, "top_k": TOP_K,
        "positive_token_text": " positive", "negative_token_text": " negative",
        "positive_token_id": POSITIVE_TOKEN, "negative_token_id": NEGATIVE_TOKEN,
        "offline": True, "dtype": "bfloat16", "attention_implementation": "sdpa",
    }
    if identity != expected_identity:
        raise QSCCIError("model identity differs from frozen identity")
    _validate_service_lifecycle(artifact["service_lifecycle"])
    expected_prompt_metadata: dict[tuple[str, str], tuple[str, str, int]] = {}
    for split_name in ("SELECT", "REPORT"):
        for prompt in prompt_instances(fixture, split_name.lower()):
            expected_prompt_metadata[(split_name, prompt["prompt_id"])] = (prompt["row_id"], prompt["kind"], prompt["y"])
    reconstructed = reconstruct_feature_records(artifact["select_activation_rows"])
    if reconstructed != artifact["feature_records"]:
        raise QSCCIError("feature records do not reconstruct from sparse SELECT activations")
    selections = selection_grid(reconstructed)
    cells = artifact["cells"]
    prompt_records = artifact["prompt_records"]
    if not isinstance(cells, list) or not isinstance(prompt_records, list):
        raise QSCCIError("cells and prompt_records must be arrays")
    cell_keys = [cell.get("cell_key") for cell in cells]
    if len(cell_keys) != len(set(cell_keys)):
        raise QSCCIError("cell keys must be unique")
    expected_select = {
        cell_key(family, seed, k, alpha, "SELECT", "grid")
        for family, seed in (("chelated", None), ("contrastive", None), *((family, seed) for family in RANDOM_FAMILIES for seed in SEEDS))
        for k in KS for alpha in ALPHAS
    }
    observed_select = {cell["cell_key"] for cell in cells if cell.get("split") == "SELECT"}
    if observed_select != expected_select:
        raise QSCCIError("SELECT cell grid is incomplete or contains extras")
    prompt_by_cell: dict[str, list[Mapping[str, Any]]] = {}
    for record in prompt_records:
        exact_object(
            record,
            {
                "prompt_id", "row_id", "kind", "y", "intervention_sign", "baseline_margin",
                "intervened_margin", "gain", "kl", "baseline_top1_token_id",
                "intervened_top1_token_id", "shared_delta_norm", "cast_delta_norm",
                "realized_update_norm", "cell_key", "family", "seed", "k", "alpha",
            },
            "prompt record",
        )
        if not isinstance(record["cell_key"], str):
            raise QSCCIError("prompt record cell_key must be a string")
        prompt_by_cell.setdefault(record["cell_key"], []).append(record)
    if set(prompt_by_cell) != set(cell_keys):
        raise QSCCIError("prompt records and cells are not exactly cross-linked")
    for cell in cells:
        exact_object(
            cell,
            {
                "cell_key", "split", "operating_point", "family", "seed", "k", "alpha",
                "feature_ids", "digest_ranks", "aggregate_direction_norm", "direction_cosines",
                "residual_rms_scale", "endpoints",
            },
            "cell",
        )
        if cell["split"] not in {"SELECT", "REPORT"} or cell["family"] not in FAMILIES:
            raise QSCCIError("cell split/family is outside the frozen closed set")
        if cell["seed"] is not None and (type(cell["seed"]) is not int or cell["seed"] not in SEEDS):
            raise QSCCIError("cell seed is outside the frozen set")
        if (cell["family"] in RANDOM_FAMILIES) != (cell["seed"] is not None):
            raise QSCCIError("cell family and seed identity disagree")
        if type(cell["k"]) is not int or cell["k"] not in KS or type(cell["alpha"]) not in (int, float) or float(cell["alpha"]) not in ALPHAS:
            raise QSCCIError("cell k/alpha is outside the frozen grid")
        if cell["operating_point"] not in {"grid", "primary", "fixed_k4_alpha2"}:
            raise QSCCIError("cell operating point is outside the frozen set")
        if cell["split"] == "SELECT" and cell["operating_point"] != "grid":
            raise QSCCIError("SELECT cell must be a grid cell")
        if cell["split"] == "REPORT" and cell["operating_point"] == "grid":
            raise QSCCIError("REPORT cell cannot be a tuning grid cell")
        expected_selection = selections[f"{cell['family']}|{'null' if cell['seed'] is None else cell['seed']}|{cell['k']}"]
        if cell["feature_ids"] != expected_selection["feature_ids"] or cell["digest_ranks"] != expected_selection["digest_ranks"]:
            raise QSCCIError("cell feature selection differs from frozen reconstruction")
        if cell["cell_key"] != cell_key(cell["family"], cell["seed"], cell["k"], cell["alpha"], cell["split"], cell["operating_point"]):
            raise QSCCIError("cell key is noncanonical")
        records_for_cell = prompt_by_cell.get(cell["cell_key"], [])
        for record in records_for_cell:
            if (record["family"], record["seed"], record["k"], float(record["alpha"])) != (
                cell["family"], cell["seed"], cell["k"], float(cell["alpha"])
            ):
                raise QSCCIError("prompt record intervention identity disagrees with its cell")
            expected_metadata = expected_prompt_metadata.get((cell["split"], record["prompt_id"]))
            if expected_metadata is None or (record["row_id"], record["kind"], record["y"]) != expected_metadata:
                raise QSCCIError("prompt record identity/label differs from the frozen fixture")
            if type(record["intervention_sign"]) is not int or record["intervention_sign"] not in (-1, 1):
                raise QSCCIError("prompt intervention sign must be -1 or +1")
            plain_int(record["baseline_top1_token_id"], "baseline top-1 token", minimum=0)
            plain_int(record["intervened_top1_token_id"], "intervened top-1 token", minimum=0)
            for field in ("baseline_margin", "intervened_margin", "kl", "shared_delta_norm", "cast_delta_norm", "realized_update_norm"):
                float32_leaf(record[field], f"prompt record {field}")
        endpoints = aggregate_cell(records_for_cell)
        if endpoints != cell["endpoints"]:
            raise QSCCIError("cell endpoints do not reconstruct")
        exact_object(
            cell["endpoints"],
            {
                "mean_correct_gain", "mean_wrong_gain", "bidirectional_causal_contrast",
                "baseline_accuracy", "intervened_accuracy", "canonical_nuisance_mismatch",
                "material_response_completeness", "mean_kl", "max_kl",
                "outside_target_collateral_fraction",
            },
            "cell endpoints",
        )
        finite(cell["aggregate_direction_norm"], "aggregate direction norm")
        if cell["aggregate_direction_norm"] <= 0.0:
            raise QSCCIError("aggregate direction norm must be positive")
        if float32_leaf(cell["residual_rms_scale"], "residual RMS scale") <= 0.0:
            raise QSCCIError("residual RMS scale must be positive")
    reconstructed_choices: dict[str, Any] = {}
    for split_name in ("SELECT", "REPORT"):
        if len({cell["residual_rms_scale"] for cell in cells if cell["split"] == split_name}) != 1:
            raise QSCCIError(f"{split_name} cells must share one frozen residual RMS scale")
    for family in ("chelated", "contrastive"):
        reconstructed_choices[f"{family}|null"] = tuning_choice(cells, family, None)
    for family in RANDOM_FAMILIES:
        for seed in SEEDS:
            reconstructed_choices[f"{family}|{seed}"] = tuning_choice(cells, family, seed)
    if reconstructed_choices != artifact["choices"]:
        raise QSCCIError("SELECT tuning choices do not reconstruct")
    expected_choice_keys = {
        "chelated|null", "contrastive|null",
        *(f"{family}|{seed}" for family in RANDOM_FAMILIES for seed in SEEDS),
    }
    if set(artifact["choices"]) != expected_choice_keys:
        raise QSCCIError("choice identities differ from the frozen eight pipelines")
    for choice in artifact["choices"].values():
        exact_object(choice, {"k", "alpha", "cell_key", "feature_ids"}, "choice")
    expected_report: set[str] = set()
    for choice_id, choice in reconstructed_choices.items():
        family, seed_text = choice_id.split("|", 1)
        seed = None if seed_text == "null" else int(seed_text)
        expected_report.add(cell_key(family, seed, choice["k"], choice["alpha"], "REPORT", "primary"))
        if (choice["k"], choice["alpha"]) != (4, 2.0):
            expected_report.add(cell_key(family, seed, 4, 2.0, "REPORT", "fixed_k4_alpha2"))
    observed_report = {cell["cell_key"] for cell in cells if cell.get("split") == "REPORT"}
    if observed_report != expected_report:
        raise QSCCIError("REPORT primary/fixed operating-point cells differ from frozen contract")
    baseline_attestations: dict[tuple[str, str], tuple[Any, Any]] = {}
    for record in prompt_records:
        split = record["cell_key"].split("|", 1)[0]
        key = (split, record["prompt_id"])
        attestation = (record["baseline_margin"], record["baseline_top1_token_id"])
        if key in baseline_attestations and baseline_attestations[key] != attestation:
            raise QSCCIError("baseline attested leaves changed across intervention cells")
        baseline_attestations[key] = attestation
    gates, disposition = compute_gates(cells, reconstructed_choices)
    if gates != artifact["gates"] or disposition != artifact["disposition"]:
        raise QSCCIError("gates or disposition do not reconstruct")
    aggregate_bundle = exact_object(artifact["aggregates"], {"cell_endpoints", "decoder_columns"}, "aggregates")
    endpoint_index = aggregate_bundle["cell_endpoints"]
    if not isinstance(endpoint_index, Mapping) or set(endpoint_index) != set(cell_keys) or endpoint_index != {cell["cell_key"]: cell["endpoints"] for cell in cells}:
        raise QSCCIError("aggregate endpoint index differs from cell endpoints")
    required_decoder_ids = sorted({feature_id for selection in selections.values() for feature_id in selection["feature_ids"]})
    columns = aggregate_bundle["decoder_columns"]
    if not isinstance(columns, list) or [column.get("feature_id") for column in columns if isinstance(column, Mapping)] != required_decoder_ids or len(columns) != len(required_decoder_ids):
        raise QSCCIError("retained decoder columns differ from the exact selected-feature union")
    torch: Any = None
    if require_pinned_sae_provenance:
        try:
            import torch
        except ImportError as exc:
            raise QSCCIError("Torch is required to reconstruct frozen FP32 directions") from exc
    decoder_by_id: dict[int, Any] = {}
    for column in columns:
        exact_object(column, {"feature_id", "values", "sha256"}, "decoder column")
        plain_int(column["feature_id"], "decoder feature_id", minimum=0)
        if not isinstance(column["values"], list) or len(column["values"]) != HIDDEN_SIZE:
            raise QSCCIError("decoder column must retain exactly 2048 FP32 values")
        values = [float32_leaf(value, "decoder column value") for value in column["values"]]
        body = {"feature_id": column["feature_id"], "values": values}
        if column["sha256"] != sha256_bytes(canonical_json(body)):
            raise QSCCIError("decoder column digest mismatch")
        if require_pinned_sae_provenance:
            decoder_by_id[column["feature_id"]] = torch.tensor(values, dtype=torch.float32)
    if require_pinned_sae_provenance:
        try:
            from huggingface_hub import hf_hub_download

            pinned_sae_path = Path(
                hf_hub_download(
                    repo_id=SAE_REPO,
                    filename=SAE_FILENAME,
                    revision=SAE_REVISION,
                    local_files_only=True,
                )
            )
        except (ImportError, OSError) as exc:
            raise QSCCIError("pinned SAE cache is required to verify decoder-column provenance") from exc
        if not pinned_sae_path.is_file() or sha256_file(pinned_sae_path) != SAE_SHA256:
            raise QSCCIError("cached SAE checkpoint identity/digest differs from the frozen source")
        try:
            state = torch.load(pinned_sae_path, map_location="cpu", weights_only=True)
        except (RuntimeError, EOFError, OSError) as exc:
            raise QSCCIError("cached SAE checkpoint could not be loaded safely") from exc
        if not isinstance(state, dict) or "W_dec" not in state or tuple(state["W_dec"].shape) != (HIDDEN_SIZE, SAE_WIDTH):
            raise QSCCIError("cached SAE decoder contract is invalid")
        pinned_decoder = state["W_dec"].float()
        for feature_id, retained_column in decoder_by_id.items():
            if not torch.equal(retained_column, pinned_decoder[:, feature_id]):
                raise QSCCIError("retained decoder column differs from the pinned checkpoint")
        del state, pinned_decoder
    if require_pinned_sae_provenance:
        reconstructed_directions: dict[tuple[str, int | None, int], Any] = {}
        reconstructed_norms: dict[tuple[str, int | None, int], float] = {}
        for selection_key, selection in selections.items():
            family, seed_text, k_text = selection_key.split("|")
            seed = None if seed_text == "null" else int(seed_text)
            k = int(k_text)
            selected_columns = torch.stack([decoder_by_id[item] for item in selection["feature_ids"]], dim=1)
            norms = torch.linalg.vector_norm(selected_columns, dim=0)
            if not torch.isfinite(norms).all() or bool((norms <= 0).any()):
                raise QSCCIError("retained decoder column norm is invalid")
            signs = torch.tensor([reconstructed[item]["sign"] for item in selection["feature_ids"]], dtype=torch.float32)
            direction_sum = torch.sum((selected_columns / norms) * signs.unsqueeze(0), dim=1, dtype=torch.float32)
            norm = torch.linalg.vector_norm(direction_sum)
            if not torch.isfinite(norm) or float(norm) <= 0.0:
                raise QSCCIError("reconstructed aggregate direction is invalid")
            reconstructed_norms[(family, seed, k)] = float(norm)
            reconstructed_directions[(family, seed, k)] = direction_sum / norm
        for cell in cells:
            if cell["aggregate_direction_norm"] != reconstructed_norms[(cell["family"], cell["seed"], cell["k"])]:
                raise QSCCIError("aggregate direction norm does not reconstruct from retained decoder columns")
            direction = reconstructed_directions[(cell["family"], cell["seed"], cell["k"])]
            reference_fp32, reference_bf16 = reference_delta_components(
                direction, cell["alpha"], cell["residual_rms_scale"]
            )
            expected_shared_norm, expected_cast_norm = canonical_delta_norms(reference_fp32, reference_bf16)
            for record in prompt_by_cell[cell["cell_key"]]:
                if record["shared_delta_norm"] != expected_shared_norm or record["cast_delta_norm"] != expected_cast_norm:
                    raise QSCCIError("retained delta norms do not reconstruct from scale/direction/alpha")
        primary_cells = _primary_report_cells(cells, reconstructed_choices)
        candidate_primary = primary_cells["chelated|null"]
        candidate_direction = reconstructed_directions[("chelated", None, candidate_primary["k"])]
        expected_cosines: dict[str, float] = {}
        for control_key, control in primary_cells.items():
            if control_key == "chelated|null":
                continue
            control_direction = reconstructed_directions[(control["family"], control["seed"], control["k"])]
            expected_cosines[control_key] = canonicalize_direction_cosine(float(
                torch.sum(candidate_direction.double() * control_direction.double())
                / (torch.linalg.vector_norm(candidate_direction.double()) * torch.linalg.vector_norm(control_direction.double()))
            ))
        if candidate_primary["direction_cosines"] != expected_cosines:
            raise QSCCIError("primary direction cosines do not reconstruct from retained decoder columns")
        for cell in cells:
            if cell is not candidate_primary and cell["direction_cosines"] != {}:
                raise QSCCIError("only the primary chelated cell may retain comparison cosines")
    resources = artifact["resources"]
    exact_object(
        resources,
        {
            "free_cuda_bytes", "total_cuda_bytes", "free_disk_bytes", "device", "wall_seconds",
            "peak_rss_bytes", "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes", "python",
            "torch", "transformers",
        },
        "resources",
    )
    for field in ("free_cuda_bytes", "total_cuda_bytes", "free_disk_bytes", "peak_rss_bytes", "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"):
        plain_int(resources[field], f"resources.{field}", minimum=0)
    if resources["free_cuda_bytes"] < MIN_CUDA_FREE_BYTES or resources["free_disk_bytes"] < MIN_DISK_BYTES or resources["total_cuda_bytes"] < resources["free_cuda_bytes"]:
        raise QSCCIError("retained preflight resources violate the frozen contract")
    if resources["python"] != "3.12" and not str(resources["python"]).startswith("3.12."):
        raise QSCCIError("retained Python identity is not 3.12")
    if resources["torch"] != "2.13.0+cu130" or resources["transformers"] != "5.15.0":
        raise QSCCIError("retained library identities differ from the frozen versions")
    if not isinstance(resources["device"], str) or not resources["device"]:
        raise QSCCIError("retained CUDA device identity must be non-empty")
    for field in ("wall_seconds", "peak_rss_bytes", "cuda_peak_allocated_bytes", "cuda_peak_reserved_bytes"):
        finite(resources[field], f"resources.{field}")
    if resources["wall_seconds"] < 0.0 or resources["wall_seconds"] > MAX_WALL_SECONDS or resources["peak_rss_bytes"] > MAX_RSS_BYTES:
        raise QSCCIError("retained CPU resource ceiling failed")
    if resources["cuda_peak_allocated_bytes"] > MAX_CUDA_ALLOCATED_BYTES or resources["cuda_peak_reserved_bytes"] > MAX_CUDA_RESERVED_BYTES:
        raise QSCCIError("retained CUDA resource ceiling failed")
    if artifact["failures"]:
        raise QSCCIError("COMPLETE artifact cannot retain failures")
    if artifact["artifact_digest"] != artifact_digest(artifact):
        raise QSCCIError("artifact digest mismatch")
    return {"verified": True, "protocol_id": PROTOCOL_ID, "disposition": disposition, "attested_model_leaves_reexecuted": False}


def verify_artifact(artifact: Mapping[str, Any], *, root: Path) -> dict[str, Any]:
    """Live verifier; pinned SAE decoder provenance is always mandatory."""
    return _verify_artifact_impl(
        artifact, root=root, require_pinned_sae_provenance=True
    )


def verify_artifact_cache_independent(
    artifact: Mapping[str, Any], *, root: Path
) -> dict[str, Any]:
    """Check portable retained-leaf semantics without decoder floating replay."""
    result = _verify_artifact_impl(
        artifact, root=root, require_pinned_sae_provenance=False
    )
    return {
        "status": "CACHE_INDEPENDENT_RETAINED_LEAF_SEMANTICS_VERIFIED",
        "protocol_id": result["protocol_id"],
        "disposition": result["disposition"],
        "retained_leaf_semantics_verified": True,
        "pinned_model_sae_cache_verified": False,
        "decoder_floating_replay_verified": False,
        "reexecution_verified": False,
        "attested_model_leaves_reexecuted": False,
    }


@dataclass
class TorchBackend:
    """Pinned offline CUDA/BF16 backend used only by an explicitly opted-in worker."""

    root: Path

    def __post_init__(self) -> None:
        self.torch: Any = None
        self.tokenizer: Any = None
        self.model: Any = None
        self.w_enc: Any = None
        self.b_enc: Any = None
        self.w_dec: Any = None

    def load(self, output_parent: Path) -> dict[str, Any]:
        import torch
        from huggingface_hub import hf_hub_download
        from transformers import AutoModelForCausalLM, AutoTokenizer, __version__ as transformers_version

        if sys.version_info[:2] != (3, 12):
            raise QSCCIError("official worker requires Python 3.12")
        if torch.__version__ != "2.13.0+cu130":
            raise QSCCIError(f"PyTorch identity mismatch: {torch.__version__}")
        if transformers_version != "5.15.0":
            raise QSCCIError(f"Transformers identity mismatch: {transformers_version}")
        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        if not torch.cuda.is_available():
            raise QSCCIError("CUDA is required")
        free_cuda, total_cuda = torch.cuda.mem_get_info()
        free_disk = shutil.disk_usage(output_parent).free
        if free_cuda < MIN_CUDA_FREE_BYTES or free_disk < MIN_DISK_BYTES:
            raise QSCCIError("preflight free CUDA or disk is below frozen minimum")
        tokenizer = AutoTokenizer.from_pretrained(MODEL_REPO, revision=MODEL_REVISION, local_files_only=True)
        if tokenizer.padding_side != "right":
            tokenizer.padding_side = "right"
        if tokenizer.pad_token_id is None:
            if tokenizer.eos_token_id is None:
                raise QSCCIError("tokenizer has neither pad nor EOS token")
            tokenizer.pad_token_id = tokenizer.eos_token_id
        for text, expected in ((" positive", POSITIVE_TOKEN), (" negative", NEGATIVE_TOKEN)):
            encoded = tokenizer.encode(text, add_special_tokens=False)
            if encoded != [expected]:
                raise QSCCIError(f"target token identity mismatch for {text!r}: {encoded}")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_REPO, revision=MODEL_REVISION, local_files_only=True, torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
        ).to("cuda").eval()
        if len(model.model.layers) <= MODEL_LAYER or int(model.config.hidden_size) != HIDDEN_SIZE:
            raise QSCCIError("model layer/hidden identity mismatch")
        sae_path = Path(hf_hub_download(repo_id=SAE_REPO, filename=SAE_FILENAME, revision=SAE_REVISION, local_files_only=True))
        if sha256_file(sae_path) != SAE_SHA256:
            raise QSCCIError("SAE checkpoint digest mismatch")
        state = torch.load(sae_path, map_location="cpu", weights_only=True)
        shapes = {"W_enc": (SAE_WIDTH, HIDDEN_SIZE), "W_dec": (HIDDEN_SIZE, SAE_WIDTH), "b_enc": (SAE_WIDTH,), "b_dec": (HIDDEN_SIZE,)}
        if set(state) != set(shapes) or any(tuple(state[key].shape) != shape for key, shape in shapes.items()):
            raise QSCCIError("SAE four-tensor contract mismatch")
        self.torch, self.tokenizer, self.model = torch, tokenizer, model
        self.w_enc, self.b_enc, self.w_dec = state["W_enc"].float(), state["b_enc"].float(), state["W_dec"].float()
        torch.cuda.reset_peak_memory_stats()
        return {"free_cuda_bytes": free_cuda, "total_cuda_bytes": total_cuda, "free_disk_bytes": free_disk, "device": torch.cuda.get_device_name()}

    def baseline(self, prompts: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], Any, float]:
        torch = self.torch
        encoded = self.tokenizer([item["prompt"] for item in prompts], return_tensors="pt", padding=True)
        encoded = {key: value.to("cuda") for key, value in encoded.items()}
        positions = encoded["attention_mask"].sum(dim=1) - 1
        captured: dict[str, Any] = {}

        def hook(_module: Any, _inputs: Any, output: Any) -> None:
            hidden = output[0] if isinstance(output, tuple) else output
            captured["last"] = hidden[torch.arange(hidden.shape[0], device=hidden.device), positions].detach().cpu().float()

        handle = self.model.model.layers[MODEL_LAYER].register_forward_hook(hook)
        try:
            torch.cuda.synchronize()
            with torch.inference_mode():
                output = self.model(**encoded)
            torch.cuda.synchronize()
        finally:
            handle.remove()
        residual = captured.get("last")
        if residual is None or tuple(residual.shape) != (18, HIDDEN_SIZE) or not torch.isfinite(residual).all():
            raise QSCCIError("captured residual is missing, nonfinite, or wrong-shaped")
        logits = output.logits[torch.arange(18, device="cuda"), positions].float()
        if not torch.isfinite(logits).all():
            raise QSCCIError("baseline logits are nonfinite")
        pre = residual @ self.w_enc.t() + self.b_enc
        if not torch.isfinite(pre).all():
            raise QSCCIError("SAE preactivations are nonfinite")
        rows: list[dict[str, Any]] = []
        for index, prompt in enumerate(prompts):
            values = [float(value) for value in pre[index].tolist()]
            topk = deterministic_topk(values)
            ids = [item[0] for item in topk]
            selected = [item[1] for item in topk]
            body = {"prompt_id": prompt["prompt_id"], "feature_ids": ids, "values": selected}
            rows.append({**body, "row_sha256": sha256_bytes(canonical_json(body))})
        sorted_rms = sorted(float(torch.sqrt(torch.sum(row * row, dtype=torch.float32) / HIDDEN_SIZE)) for row in residual)
        scale = float(torch.tensor((float(sorted_rms[8]) + float(sorted_rms[9])) / 2.0, dtype=torch.float32))
        if not math.isfinite(scale) or scale <= 0.0:
            raise QSCCIError("residual RMS median is invalid")
        return rows, logits, scale

    def direction(self, feature_ids: Sequence[int], records: Sequence[Mapping[str, Any]]) -> tuple[Any, float]:
        torch = self.torch
        signs = torch.tensor([records[item]["sign"] for item in feature_ids], dtype=torch.float32)
        columns = self.w_dec[:, list(feature_ids)]
        norms = torch.linalg.vector_norm(columns, dim=0)
        if not torch.isfinite(columns).all() or not torch.isfinite(norms).all() or bool((norms <= 0).any()):
            raise QSCCIError("decoder direction column is invalid")
        direction_sum = torch.sum((columns / norms) * signs.unsqueeze(0), dim=1, dtype=torch.float32)
        norm = torch.linalg.vector_norm(direction_sum)
        if not torch.isfinite(norm) or float(norm) <= 0.0:
            raise QSCCIError("aggregate direction is invalid")
        return direction_sum / norm, float(norm)

    def intervene(
        self, prompts: Sequence[Mapping[str, Any]], baseline_logits: Any, direction: Any, scale: float,
        alpha: float, intervention_sign: int,
    ) -> list[dict[str, Any]]:
        torch = self.torch
        encoded = self.tokenizer([item["prompt"] for item in prompts], return_tensors="pt", padding=True)
        encoded = {key: value.to("cuda") for key, value in encoded.items()}
        positions = encoded["attention_mask"].sum(dim=1) - 1
        labels = torch.tensor([item["y"] for item in prompts], device="cuda", dtype=torch.float32)
        direction_cuda = direction.to("cuda")
        retained_norms: dict[str, Any] = {}

        def hook(_module: Any, _inputs: Any, output: Any) -> Any:
            hidden = output[0] if isinstance(output, tuple) else output
            modified, evidence = apply_bf16_update(
                hidden, direction_cuda.float(), intervention_sign * labels, alpha, scale, positions
            )
            retained_norms["shared"] = evidence["shared_delta_norm"]
            retained_norms["cast"] = evidence["cast_delta_norms"]
            retained_norms["realized"] = evidence["realized_update_norms"]
            return (modified, *output[1:]) if isinstance(output, tuple) else modified

        handle = self.model.model.layers[MODEL_LAYER].register_forward_hook(hook)
        try:
            torch.cuda.synchronize()
            with torch.inference_mode():
                output = self.model(**encoded)
            torch.cuda.synchronize()
        finally:
            handle.remove()
        changed = output.logits[torch.arange(18, device="cuda"), positions].float()
        if not torch.isfinite(changed).all():
            raise QSCCIError("intervened logits are nonfinite")
        logp0 = torch.log_softmax(baseline_logits.float(), dim=-1)
        logp1 = torch.log_softmax(changed.float(), dim=-1)
        p0 = torch.exp(logp0)
        kl = torch.sum(p0 * (logp0 - logp1), dim=-1, dtype=torch.float32)
        base_margin = baseline_logits[:, POSITIVE_TOKEN] - baseline_logits[:, NEGATIVE_TOKEN]
        changed_margin = changed[:, POSITIVE_TOKEN] - changed[:, NEGATIVE_TOKEN]
        base_top = torch.argmax(baseline_logits, dim=-1)
        changed_top = torch.argmax(changed, dim=-1)
        records = []
        for index, prompt in enumerate(prompts):
            cast_norm = float(retained_norms["cast"][index])
            realized_norm = float(retained_norms["realized"][index])
            if not math.isfinite(cast_norm) or not math.isfinite(realized_norm) or cast_norm <= 0.0 or realized_norm <= 0.0:
                raise QSCCIError("BF16 intervention vanished or became nonfinite")
            baseline_margin = float(base_margin[index])
            intervened_margin = float(changed_margin[index])
            records.append(
                {
                    "prompt_id": prompt["prompt_id"], "row_id": prompt["row_id"], "kind": prompt["kind"], "y": prompt["y"],
                    "intervention_sign": intervention_sign, "baseline_margin": baseline_margin,
                    "intervened_margin": intervened_margin, "gain": prompt["y"] * (intervened_margin - baseline_margin),
                    "kl": float(kl[index]), "baseline_top1_token_id": int(base_top[index]),
                    "intervened_top1_token_id": int(changed_top[index]), "shared_delta_norm": retained_norms["shared"],
                    "cast_delta_norm": cast_norm, "realized_update_norm": realized_norm,
                }
            )
        return records


def peak_rss_bytes() -> int:
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [("cb", wintypes.DWORD), ("page_fault", wintypes.DWORD), ("peak_working_set", ctypes.c_size_t), ("working_set", ctypes.c_size_t), ("qpp", ctypes.c_size_t), ("qp", ctypes.c_size_t), ("qnpp", ctypes.c_size_t), ("qnp", ctypes.c_size_t), ("pagefile", ctypes.c_size_t), ("peak_pagefile", ctypes.c_size_t)]

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not ctypes.windll.psapi.GetProcessMemoryInfo(ctypes.windll.kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            raise QSCCIError("GetProcessMemoryInfo failed")
        return int(counters.peak_working_set)
    import resource

    raw = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(raw if sys.platform == "darwin" else raw * 1024)


def run_worker(stage_dir: Path, root: Path, service_lifecycle: Mapping[str, Any]) -> dict[str, Any]:
    """Execute the official worker into an existing supervisor-owned hidden stage."""
    started = time.monotonic()
    if not stage_dir.is_dir() or not stage_dir.name.startswith(".") or any(stage_dir.iterdir()):
        raise QSCCIError("worker requires an empty hidden supervisor-owned stage")
    fixture = _assert_frozen_sources(root)
    backend = TorchBackend(root)
    preflight = backend.load(stage_dir.parent)
    split_data: dict[str, tuple[list[dict[str, Any]], Any, float, list[dict[str, Any]]]] = {}
    select_rows: list[dict[str, Any]] = []
    for split_name in ("select", "report"):
        prompts = prompt_instances(fixture, split_name)
        rows, logits, scale = backend.baseline(prompts)
        split_data[split_name.upper()] = (rows, logits, scale, prompts)
        if split_name == "select":
            select_rows = rows
    feature_records = reconstruct_feature_records(select_rows)
    selections = selection_grid(feature_records)
    decoder_columns: list[dict[str, Any]] = []
    for feature_id in sorted({item for selection in selections.values() for item in selection["feature_ids"]}):
        values = [float(value) for value in backend.w_dec[:, feature_id].tolist()]
        body = {"feature_id": feature_id, "values": values}
        decoder_columns.append({**body, "sha256": sha256_bytes(canonical_json(body))})
    cells: list[dict[str, Any]] = []
    prompt_records: list[dict[str, Any]] = []
    directions: dict[tuple[str, int | None, int], Any] = {}

    def execute_cell(split: str, operating_point: str, family: str, seed: int | None, k: int, alpha: float) -> None:
        if time.monotonic() - started > MAX_WALL_SECONDS or peak_rss_bytes() > MAX_RSS_BYTES:
            raise QSCCIError("cooperative wall/RSS ceiling exceeded")
        selection = selections[f"{family}|{'null' if seed is None else seed}|{k}"]
        direction, direction_norm = backend.direction(selection["feature_ids"], feature_records)
        directions[(family, seed, k)] = direction
        _rows, baseline_logits, scale, prompts = split_data[split]
        key = cell_key(family, seed, k, alpha, split, operating_point)
        records: list[dict[str, Any]] = []
        for sign in (1, -1):
            for record in backend.intervene(prompts, baseline_logits, direction, scale, alpha, sign):
                records.append({**record, "cell_key": key, "family": family, "seed": seed, "k": k, "alpha": alpha})
        endpoints = aggregate_cell(records)
        cells.append(
            {
                "cell_key": key, "split": split, "operating_point": operating_point, "family": family,
                "seed": seed, "k": k, "alpha": alpha, "feature_ids": selection["feature_ids"],
                "digest_ranks": selection["digest_ranks"], "aggregate_direction_norm": direction_norm,
                "direction_cosines": {}, "residual_rms_scale": scale, "endpoints": endpoints,
            }
        )
        prompt_records.extend(records)

    identities = [("chelated", None), ("contrastive", None)] + [(family, seed) for family in RANDOM_FAMILIES for seed in SEEDS]
    for family, seed in identities:
        for k in KS:
            for alpha in ALPHAS:
                execute_cell("SELECT", "grid", family, seed, k, alpha)
    choices = {f"{family}|{'null' if seed is None else seed}": tuning_choice(cells, family, seed) for family, seed in identities}
    for family, seed in identities:
        choice = choices[f"{family}|{'null' if seed is None else seed}"]
        execute_cell("REPORT", "primary", family, seed, choice["k"], choice["alpha"])
        if (choice["k"], choice["alpha"]) != (4, 2.0):
            execute_cell("REPORT", "fixed_k4_alpha2", family, seed, 4, 2.0)

    primary = _primary_report_cells(cells, choices)
    candidate_cell = primary["chelated|null"]
    candidate_direction = directions[("chelated", None, candidate_cell["k"])]
    torch = backend.torch
    for cell in primary.values():
        if cell is candidate_cell:
            continue
        direction = directions[(cell["family"], cell["seed"], cell["k"])]
        cosine = canonicalize_direction_cosine(float(torch.sum(candidate_direction.double() * direction.double()) / (torch.linalg.vector_norm(candidate_direction.double()) * torch.linalg.vector_norm(direction.double()))))
        candidate_cell["direction_cosines"][f"{cell['family']}|{'null' if cell['seed'] is None else cell['seed']}"] = cosine
    gates, disposition = compute_gates(cells, choices)
    wall = time.monotonic() - started
    resources = {
        **preflight, "wall_seconds": wall, "peak_rss_bytes": peak_rss_bytes(),
        "cuda_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "cuda_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "python": platform.python_version(), "torch": torch.__version__, "transformers": "5.15.0",
    }
    artifact: dict[str, Any] = {
        "protocol_id": PROTOCOL_ID, "status": "COMPLETE", "scientific_claim_status": "UNCONFIRMED",
        "novelty_claim_status": "UNCONFIRMED", "protocol_file_sha256": PROTOCOL_SHA256,
        "fixture_sha256": FIXTURE_SHA256, "schema_file_sha256": SCHEMA_SHA256,
        "model_identity": {
            "model_repo": MODEL_REPO, "model_revision": MODEL_REVISION, "sae_repo": SAE_REPO,
            "sae_revision": SAE_REVISION, "sae_filename": SAE_FILENAME, "sae_sha256": SAE_SHA256,
            "layer_index": MODEL_LAYER, "hidden_size": HIDDEN_SIZE, "sae_width": SAE_WIDTH, "top_k": TOP_K,
            "positive_token_text": " positive", "negative_token_text": " negative",
            "positive_token_id": POSITIVE_TOKEN, "negative_token_id": NEGATIVE_TOKEN,
            "offline": True, "dtype": "bfloat16", "attention_implementation": "sdpa",
        },
        "service_lifecycle": dict(service_lifecycle), "resources": resources,
        "select_activation_rows": select_rows, "feature_records": feature_records, "choices": choices,
        "prompt_records": prompt_records, "cells": cells,
        "aggregates": {
            "cell_endpoints": {cell["cell_key"]: cell["endpoints"] for cell in cells},
            "decoder_columns": decoder_columns,
        },
        "gates": gates, "disposition": disposition, "failures": [], "artifact_digest": "0" * 64,
    }
    artifact["artifact_digest"] = artifact_digest(artifact)
    (stage_dir / "qscci.json").write_bytes(canonical_json(artifact))
    return artifact
