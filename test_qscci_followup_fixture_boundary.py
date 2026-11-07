from __future__ import annotations

import hashlib
import json
import unittest
from collections import Counter
from fractions import Fraction
from pathlib import Path

from validate_qscci_blind_followup_fixture import (
    multiset_dice,
    normalize_and_tokenize,
)


ROOT = Path(__file__).resolve().parent
NEW_FIXTURE = ROOT / "docs/research/qscci-blind-followup-fixture-review-draft-v1.json"
OLD_FIXTURE = ROOT / "docs/research/qwen-scope-chelated-causal-intervention-fixture-v1.json"
PREREG = (
    ROOT
    / "docs/research/qwen-scope-nuisance-rank-separation-preregistration-draft-2026-08-16.md"
)
NEW_FIXTURE_SHA256 = "877e2a27ec42afeb2a0b3100f76576aa45fee1c435d08e4748c71efb6338820c"
OLD_FIXTURE_SHA256 = "d9f873d5a0e00d0330b87e5ea053aaf0343d4c1636d7402020490719d1336f83"
FROZEN_PARTITIONS = {
    "DEVELOPMENT": (
        ROOT / "docs/research/qscci-blind-followup-fixture-v1-development.json",
        "6ea3a5f927be266074a3875d5e793b1a801688970c3ce0cd6f7ca549422c88f5",
        12,
    ),
    "SELECT": (
        ROOT / "docs/research/qscci-blind-followup-fixture-v1-select.json",
        "18197b51af39d81639ab730675330507c2b78fb58f74c3ae7dc36096550a59df",
        24,
    ),
    "REPORT": (
        ROOT / "docs/research/qscci-blind-followup-fixture-v1-report.json",
        "26263306fbcfedbe106d6335d434f960b6128c8e2b6894b31d188b0b505de64e",
        24,
    ),
}
PROMPT_TEMPLATE = (
    "Determine whether the proposed conclusion is logically forced by all stated facts.\n"
    "Respond with positive if it is forced, otherwise respond with negative.\n\n"
    "{text}\n\n"
    "Answer:"
)
PROMPT_TEMPLATE_SHA256 = "54bc90ed5be052328c82dadae8fb86279668f8be51958b887f2bb5ecb1c5c013"
NEAR_DUPLICATE_THRESHOLD = 0.86


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TestQSCCIFollowupFixtureBoundary(unittest.TestCase):
    def test_exact_fixture_and_prompt_bindings(self) -> None:
        self.assertEqual(_sha256(NEW_FIXTURE), NEW_FIXTURE_SHA256)
        self.assertEqual(_sha256(OLD_FIXTURE), OLD_FIXTURE_SHA256)
        prompt_bytes = PROMPT_TEMPLATE.encode("utf-8")
        self.assertEqual(len(prompt_bytes), 171)
        self.assertEqual(hashlib.sha256(prompt_bytes).hexdigest(), PROMPT_TEMPLATE_SHA256)
        preregistration = PREREG.read_text(encoding="utf-8")
        self.assertIn(f"```text\n{PROMPT_TEMPLATE}\n```", preregistration)
        self.assertNotIn("<BLOCKING: BLIND FIXTURE PENDING>", preregistration)

    def test_partition_counts_and_label_mapping(self) -> None:
        payload = json.loads(NEW_FIXTURE.read_text(encoding="utf-8"))
        expected = {"DEVELOPMENT": 12, "SELECT": 24, "REPORT": 24}
        for partition, count in expected.items():
            rows = [row for row in payload["rows"] if row["partition"] == partition]
            self.assertEqual(len(rows), count)
            self.assertEqual(
                {label: sum(row["canonical_label"] == label for row in rows) for label in ("YES", "NO")},
                {"YES": count // 2, "NO": count // 2},
            )
            for row in rows:
                self.assertEqual(row["nuisance_label"], row["canonical_label"])
                self.assertNotEqual(row["material_label"], row["canonical_label"])

    def test_frozen_partition_files_are_exact_and_capability_isolated(self) -> None:
        source = json.loads(NEW_FIXTURE.read_text(encoding="utf-8"))
        expected_mapping = {
            "NO": {"numeric_label": -1, "target_string": " negative", "token_id": 7968},
            "YES": {"numeric_label": 1, "target_string": " positive", "token_id": 6572},
        }
        seen_ids: set[str] = set()
        for index, (partition, (path, digest, count)) in enumerate(FROZEN_PARTITIONS.items()):
            self.assertEqual(_sha256(path), digest)
            raw = path.read_bytes()
            self.assertFalse(raw.startswith(b"\xef\xbb\xbf"))
            self.assertTrue(raw.endswith(b"\n"))
            self.assertNotIn(b"\r", raw)
            payload = json.loads(raw)
            self.assertEqual(payload["status"], "FROZEN")
            self.assertEqual(payload["fixture_id"], f"QSCCI-BF-v1-{partition}")
            self.assertEqual(payload["partition"], partition)
            self.assertEqual(payload["partition_order_index"], index)
            self.assertEqual(payload["row_count"], count)
            self.assertEqual(payload["source_fixture_sha256"], NEW_FIXTURE_SHA256)
            self.assertEqual(payload["label_mapping"], expected_mapping)
            self.assertEqual(len(payload["rows"]), count)
            self.assertTrue(all(row["partition"] == partition for row in payload["rows"]))
            expected_rows = [row for row in source["rows"] if row["partition"] == partition]
            self.assertEqual(payload["rows"], expected_rows)
            partition_ids = {row["id"] for row in payload["rows"]}
            self.assertTrue(seen_ids.isdisjoint(partition_ids))
            seen_ids.update(partition_ids)
        self.assertEqual(len(seen_ids), 60)

    def test_new_fixture_is_not_near_duplicate_of_exhausted_v4_fixture(self) -> None:
        new_payload = json.loads(NEW_FIXTURE.read_text(encoding="utf-8"))
        old_payload = json.loads(OLD_FIXTURE.read_text(encoding="utf-8"))
        new_variants = [
            normalize_and_tokenize(row[key])
            for row in new_payload["rows"]
            for key in ("canonical_text", "nuisance_text", "material_text")
        ]
        old_variants = [
            normalize_and_tokenize(row[key])
            for partition in ("select", "report")
            for row in old_payload[partition]
            for key in ("canonical", "nuisance", "material")
        ]
        similarities = [
            multiset_dice(new_variant, old_variant)
            for new_variant in new_variants
            for old_variant in old_variants
        ]
        self.assertEqual(len(similarities), 6_480)
        self.assertLess(max(similarities), NEAR_DUPLICATE_THRESHOLD)
        exact_similarities = []
        for new_variant in new_variants:
            new_counts = Counter(new_variant)
            for old_variant in old_variants:
                old_counts = Counter(old_variant)
                numerator = 2 * sum((new_counts & old_counts).values())
                denominator = sum(new_counts.values()) + sum(old_counts.values())
                exact_similarities.append(Fraction(numerator, denominator))
        self.assertEqual(max(exact_similarities), Fraction(4, 13))


if __name__ == "__main__":
    unittest.main()
