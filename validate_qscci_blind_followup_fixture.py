"""Dependency-free contract validator for the blind QSCCI follow-up draft."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from itertools import combinations
from pathlib import Path
from typing import Iterable


FIXTURE_PATH = Path("docs/research/qscci-blind-followup-fixture-review-draft-v1.json")
NEAR_DUPLICATE_THRESHOLD = 0.86
PARTITION_ORDER = ("DEVELOPMENT", "SELECT", "REPORT")
LOGIC_FAMILIES = (
    "TRANSITIVE_CHAIN", "CONJUNCTIVE_RULE", "DISJUNCTIVE_ELIMINATION",
    "TEMPORAL_ORDER", "CARDINALITY", "MUTUAL_EXCLUSION",
)
LOGIC_SUBTYPES = {
    "TRANSITIVE_CHAIN": ("TWO_HOP",),
    "CONJUNCTIVE_RULE": ("BOTH_REQUIRED",),
    "DISJUNCTIVE_ELIMINATION": ("EXCLUSIVE_OR",),
    "TEMPORAL_ORDER": ("TWO_STEP_ORDER",),
    "CARDINALITY": ("EXACT_COUNT", "THRESHOLD"),
    "MUTUAL_EXCLUSION": ("DISJOINT_CATEGORIES",),
}
EXPECTED_IDS = {
    "DEVELOPMENT": (
        "QBF-7K2M9A", "QBF-P4X8LC", "QBF-3VQ6ND", "QBF-H9R2WT",
        "QBF-B5J7YU", "QBF-N8C3KF", "QBF-4ZL6PS", "QBF-T2G9XE",
        "QBF-M7W4AR", "QBF-C9F2VH", "QBF-X6D8BJ", "QBF-R3Y5QN",
    ),
    "SELECT": (
        "QBF-J8P4CZ", "QBF-2N7VKM", "QBF-W5R9DL", "QBF-6YB3QF",
        "QBF-L2X8SN", "QBF-A9M5TG", "QBF-V4C7HP", "QBF-8D3KWR",
        "QBF-F6Q2NJ", "QBF-Z7L4BX", "QBF-3T9GMC", "QBF-K5V8YA",
        "QBF-R2H6PD", "QBF-C8W3LF", "QBF-9N4XQJ", "QBF-G7B2VK",
        "QBF-M3S8ZR", "QBF-5P9DTH", "QBF-Y2K6CW", "QBF-H8F4AN",
        "QBF-4L7JMX", "QBF-X9C3RG", "QBF-B6V2QP", "QBF-T5N8KD",
    ),
    "REPORT": (
        "QBF-D7Q4YL", "QBF-2W9KFH", "QBF-S5M8VC", "QBF-8R3NPJ",
        "QBF-L6T2XB", "QBF-A4V7QK", "QBF-Y9C5GM", "QBF-3H8WDR",
        "QBF-P2K6ZN", "QBF-F7B4XT", "QBF-9J5LCV", "QBF-C3N8QS",
        "QBF-V6R2MY", "QBF-5X9DGH", "QBF-K4P7WB", "QBF-Z8T3NF",
        "QBF-B2M6RJ", "QBF-7C5QXL", "QBF-N9V4KA", "QBF-G3Y8PD",
        "QBF-4F7SWH", "QBF-X2L6TC", "QBF-R8K5BZ", "QBF-M4D9QG",
    ),
}
EXPECTED_LABELS = {
    "DEVELOPMENT": ("YES", "NO", "NO", "YES", "NO", "YES", "YES", "NO", "YES", "NO", "NO", "YES"),
    "SELECT": (
        "NO", "YES", "YES", "NO", "YES", "NO", "NO", "YES",
        "YES", "NO", "YES", "NO", "NO", "NO", "YES", "YES",
        "NO", "NO", "YES", "NO", "NO", "YES", "YES", "YES",
    ),
    "REPORT": (
        "YES", "YES", "NO", "NO", "YES", "NO", "YES", "NO",
        "NO", "YES", "NO", "NO", "YES", "NO", "YES", "YES",
        "NO", "YES", "NO", "YES", "NO", "YES", "NO", "YES",
    ),
}
TOP_LEVEL_FIELDS = {
    "fixture_id", "schema_version", "status", "task", "partition_order",
    "label_order_policy", "near_duplicate_policy", "structural_policy", "rows",
}
ROW_FIELDS = {
    "id", "partition", "logic_family", "logic_subtype", "surface_signature",
    "canonical_text", "canonical_label", "nuisance_text", "nuisance_label",
    "material_text", "material_label",
}
LABEL_ORDER_POLICY = {
    "method": "fixed_pinned_shuffle_per_partition",
    "ordinal_label_leakage": False,
    "expected_yes_fraction": 0.5,
}
NEAR_DUPLICATE_POLICY = {
    "normalization": "NFKC_then_casefold",
    "tokenization": "unicode_L_or_N_with_following_M",
    "metric": "multiset_sorensen_dice",
    "threshold": NEAR_DUPLICATE_THRESHOLD,
    "comparison": "all_variants_across_distinct_base_rows",
    "threshold_rule": "reject_greater_than_or_equal",
}
STRUCTURAL_POLICY = {
    "families": list(LOGIC_FAMILIES),
    "subtypes": {family: list(subtypes) for family, subtypes in LOGIC_SUBTYPES.items()},
    "surface_signatures": ["AFFIRMATIVE_CONCLUSION", "NEGATED_CONCLUSION"],
    "quota_per_family": {"DEVELOPMENT": 2, "SELECT": 4, "REPORT": 4},
    "label_balance_within_family_and_partition": True,
    "subtype_balance_across_labels": True,
    "surface_signature_balance_across_labels": True,
    "conjunctive_canonical_forbidden_tokens": ["but"],
    "cardinality_subtype_markers": {
        "EXACT_COUNT": ["exactly"],
        "THRESHOLD": ["at least", "at most"],
    },
}


def normalize_and_tokenize(text: str) -> tuple[str, ...]:
    """Apply NFKC, casefold, then tokenize Unicode letters/numbers plus marks."""
    normalized = unicodedata.normalize("NFKC", text).casefold()
    tokens: list[str] = []
    current: list[str] = []
    for character in normalized:
        category = unicodedata.category(character)
        if category[0] in {"L", "N"}:
            current.append(character)
        elif category[0] == "M" and current:
            current.append(character)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tuple(tokens)


def multiset_dice(left: Iterable[str], right: Iterable[str]) -> float:
    """Return multiset Sorensen-Dice similarity; two empty streams score 1."""
    left_counts = Counter(left)
    right_counts = Counter(right)
    denominator = sum(left_counts.values()) + sum(right_counts.values())
    if denominator == 0:
        return 1.0
    return 2.0 * sum((left_counts & right_counts).values()) / denominator


def _field_drift(actual: set[str], expected: set[str], location: str) -> list[str]:
    errors: list[str] = []
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing:
        errors.append(f"{location}: missing fields {missing}")
    if unknown:
        errors.append(f"{location}: unknown fields {unknown}")
    return errors


def _contiguous_pair_indices(tokens: tuple[str, ...], pair: tuple[str, str]) -> list[int]:
    return [
        index for index in range(len(tokens) - 1)
        if tokens[index:index + 2] == pair
    ]


def _derive_cardinality_subtype(canonical_text: str) -> str | None:
    """Derive subtype only from closed syntax in the first rule sentence."""
    rule_sentence = canonical_text.split(".", 1)[0]
    rule_tokens = normalize_and_tokenize(rule_sentence)
    exact_hits = _contiguous_pair_indices(rule_tokens, ("exactly", "when"))
    threshold_hits = (
        _contiguous_pair_indices(rule_tokens, ("at", "least"))
        + _contiguous_pair_indices(rule_tokens, ("at", "most"))
    )
    if len(exact_hits) == 1 and not threshold_hits:
        return "EXACT_COUNT"
    if len(threshold_hits) == 1 and not exact_hits:
        return "THRESHOLD"
    return None


def validate_document(document: object) -> list[str]:
    """Validate a parsed document and return all safely detectable errors."""
    if not isinstance(document, dict):
        return ["document root must be an object"]
    errors = _field_drift(set(document), TOP_LEVEL_FIELDS, "root")
    if errors:
        return errors

    exact_values = {
        "fixture_id": "QSCCI-BF-RD1",
        "schema_version": 2,
        "status": "review-draft",
        "task": "Answer YES exactly when the proposed conclusion is logically forced by all stated facts; otherwise answer NO.",
        "partition_order": list(PARTITION_ORDER),
        "label_order_policy": LABEL_ORDER_POLICY,
        "near_duplicate_policy": NEAR_DUPLICATE_POLICY,
        "structural_policy": STRUCTURAL_POLICY,
    }
    for field, expected in exact_values.items():
        if document.get(field) != expected:
            errors.append(f"{field} does not match the review-draft contract")

    rows = document.get("rows")
    if not isinstance(rows, list):
        return errors + ["rows must be a list"]
    expected_total = sum(len(ids) for ids in EXPECTED_IDS.values())
    if len(rows) != expected_total:
        errors.append(f"expected {expected_total} rows, found {len(rows)}")

    seen_ids: set[str] = set()
    variants: list[tuple[str, str, tuple[str, ...]]] = []
    actual_ids = {partition: [] for partition in PARTITION_ORDER}
    actual_labels = {partition: [] for partition in PARTITION_ORDER}
    family_counts: Counter[tuple[str, str, str]] = Counter()
    subtype_counts: Counter[tuple[str, str, str, str]] = Counter()
    signature_counts: Counter[tuple[str, str, str, str]] = Counter()
    for index, row in enumerate(rows):
        location = f"row {index}"
        if not isinstance(row, dict):
            errors.append(f"{location} must be an object")
            continue
        field_errors = _field_drift(set(row), ROW_FIELDS, location)
        errors.extend(field_errors)
        if field_errors:
            continue

        row_id = row["id"]
        partition = row["partition"]
        family = row["logic_family"]
        subtype = row["logic_subtype"]
        signature = row["surface_signature"]
        canonical_label = row["canonical_label"]
        if not isinstance(row_id, str) or not re.fullmatch(r"QBF-[A-Z0-9]{6}", row_id):
            errors.append(f"{location}: id must match QBF-[A-Z0-9]{{6}}")
            row_key = location
        else:
            row_key = row_id
            if row_id in seen_ids:
                errors.append(f"duplicate id {row_id}")
            seen_ids.add(row_id)
        if partition not in PARTITION_ORDER:
            errors.append(f"{row_key}: invalid partition {partition!r}")
        else:
            actual_ids[partition].append(row_id)
            actual_labels[partition].append(canonical_label)
        if family not in LOGIC_FAMILIES:
            errors.append(f"{row_key}: invalid logic_family {family!r}")
        elif subtype not in LOGIC_SUBTYPES[family]:
            errors.append(f"{row_key}: invalid logic_subtype {subtype!r} for {family}")
        if signature not in ("AFFIRMATIVE_CONCLUSION", "NEGATED_CONCLUSION"):
            errors.append(f"{row_key}: invalid surface_signature {signature!r}")
        if canonical_label not in ("YES", "NO"):
            errors.append(f"{row_key}: invalid canonical_label {canonical_label!r}")
        if partition in PARTITION_ORDER and family in LOGIC_FAMILIES and canonical_label in ("YES", "NO"):
            family_counts[(partition, family, canonical_label)] += 1
            if subtype in LOGIC_SUBTYPES[family]:
                subtype_counts[(partition, family, canonical_label, subtype)] += 1
            if signature in ("AFFIRMATIVE_CONCLUSION", "NEGATED_CONCLUSION"):
                signature_counts[(partition, family, canonical_label, signature)] += 1
        if row["nuisance_label"] != canonical_label:
            errors.append(f"{row_key}: nuisance transform must preserve the label")
        inverse = "NO" if canonical_label == "YES" else "YES"
        if row["material_label"] != inverse:
            errors.append(f"{row_key}: material transform must invert the label")

        sibling_tokens: list[tuple[str, ...]] = []
        for field in ("canonical_text", "nuisance_text", "material_text"):
            text = row[field]
            if not isinstance(text, str) or not text.strip():
                errors.append(f"{row_key}: {field} must be non-empty text")
                continue
            tokens = normalize_and_tokenize(text)
            if len(tokens) < 8:
                errors.append(f"{row_key}: {field} has fewer than 8 tokens")
            sibling_tokens.append(tokens)
            variants.append((row_key, field, tokens))
            if field == "canonical_text":
                marker_indices = _contiguous_pair_indices(tokens, ("proposed", "conclusion"))
                if len(marker_indices) != 1:
                    errors.append(
                        f"{row_key}: canonical_text must contain contiguous 'proposed conclusion' exactly once"
                    )
                else:
                    marker_index = marker_indices[0]
                    conclusion_tokens = tokens[marker_index + 2:]
                    derived = (
                        "NEGATED_CONCLUSION" if "not" in conclusion_tokens
                        else "AFFIRMATIVE_CONCLUSION"
                    )
                    if signature != derived:
                        errors.append(
                            f"{row_key}: surface_signature {signature!r} does not match derived {derived!r}"
                        )
                if family == "CONJUNCTIVE_RULE" and "but" in tokens:
                    errors.append(f"{row_key}: conjunctive canonical_text contains forbidden token 'but'")
                if family == "CARDINALITY" and subtype in LOGIC_SUBTYPES[family]:
                    derived_subtype = _derive_cardinality_subtype(text)
                    if derived_subtype != subtype:
                        errors.append(
                            f"{row_key}: logic_subtype {subtype!r} does not match first-rule-sentence "
                            f"syntax {derived_subtype!r}"
                        )
        if len(sibling_tokens) == 3 and len(set(sibling_tokens)) != 3:
            errors.append(f"{row_key}: canonical, nuisance, and material texts must be distinct")

    for partition in PARTITION_ORDER:
        if tuple(actual_ids[partition]) != EXPECTED_IDS[partition]:
            errors.append(f"{partition}: opaque ID order or ID-to-partition mapping drifted")
        if tuple(actual_labels[partition]) != EXPECTED_LABELS[partition]:
            errors.append(f"{partition}: fixed pinned label shuffle drifted")
        quota = STRUCTURAL_POLICY["quota_per_family"][partition]
        for family in LOGIC_FAMILIES:
            yes = family_counts[(partition, family, "YES")]
            no = family_counts[(partition, family, "NO")]
            if yes + no != quota or yes != no:
                errors.append(
                    f"{partition}/{family}: expected quota {quota} split equally, found YES={yes}, NO={no}"
                )
            yes_subtypes = Counter({
                subtype: subtype_counts[(partition, family, "YES", subtype)]
                for subtype in LOGIC_SUBTYPES[family]
            })
            no_subtypes = Counter({
                subtype: subtype_counts[(partition, family, "NO", subtype)]
                for subtype in LOGIC_SUBTYPES[family]
            })
            if yes_subtypes != no_subtypes:
                errors.append(
                    f"{partition}/{family}: logic_subtype distribution differs by label: "
                    f"YES={dict(yes_subtypes)}, NO={dict(no_subtypes)}"
                )
            signatures = ("AFFIRMATIVE_CONCLUSION", "NEGATED_CONCLUSION")
            yes_signatures = Counter({
                signature: signature_counts[(partition, family, "YES", signature)]
                for signature in signatures
            })
            no_signatures = Counter({
                signature: signature_counts[(partition, family, "NO", signature)]
                for signature in signatures
            })
            if yes_signatures != no_signatures:
                errors.append(
                    f"{partition}/{family}: surface_signature distribution differs by label: "
                    f"YES={dict(yes_signatures)}, NO={dict(no_signatures)}"
                )

    for left, right in combinations(variants, 2):
        left_id, left_field, left_tokens = left
        right_id, right_field, right_tokens = right
        if left_id == right_id:
            continue
        similarity = multiset_dice(left_tokens, right_tokens)
        if similarity >= NEAR_DUPLICATE_THRESHOLD:
            errors.append(
                f"near duplicate at {similarity:.6f}: {left_id}.{left_field} vs "
                f"{right_id}.{right_field} (inclusive threshold {NEAR_DUPLICATE_THRESHOLD:.2f})"
            )
    return errors


def validate_fixture(path: Path = FIXTURE_PATH) -> list[str]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        return [f"cannot read fixture: {exc}"]
    return validate_document(document)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", nargs="?", type=Path, default=FIXTURE_PATH)
    args = parser.parse_args()
    errors = validate_fixture(args.path)
    if errors:
        for error in errors:
            print(f"ERROR: {error}")
        return 1
    print(f"PASS: {args.path} satisfies the blind follow-up review-draft contract")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
