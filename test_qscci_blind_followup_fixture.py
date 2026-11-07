from __future__ import annotations

import copy
import json
import unittest

from validate_qscci_blind_followup_fixture import (
    FIXTURE_PATH,
    NEAR_DUPLICATE_THRESHOLD,
    multiset_dice,
    normalize_and_tokenize,
    validate_document,
    validate_fixture,
)


class UnicodeSimilarityTests(unittest.TestCase):
    def test_nfkc_casefold_and_unicode_tokenization_are_specified(self):
        self.assertEqual(
            normalize_and_tokenize("ＣＡＦÉ—Straße １２ / co_op"),
            ("café", "strasse", "12", "co", "op"),
        )

    def test_multiset_dice_counts_repeated_tokens(self):
        self.assertAlmostEqual(multiset_dice(("a", "a", "b"), ("a", "b", "b")), 2 / 3)

    def test_threshold_value_is_exact_and_inclusive(self):
        left = ("shared",) * 43 + ("left",) * 7
        right = ("shared",) * 43 + ("right",) * 7
        self.assertEqual(multiset_dice(left, right), NEAR_DUPLICATE_THRESHOLD)


class FixtureContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))

    def mutated(self):
        return copy.deepcopy(self.document)

    def test_committed_fixture_passes(self):
        self.assertEqual(validate_fixture(FIXTURE_PATH), [])

    def test_malformed_root_is_graceful(self):
        self.assertEqual(validate_document([]), ["document root must be an object"])
        self.assertEqual(validate_document(None), ["document root must be an object"])

    def test_missing_and_unknown_root_fields_are_rejected(self):
        document = self.mutated()
        del document["status"]
        document["surprise"] = True
        errors = validate_document(document)
        self.assertIn("root: missing fields ['status']", errors)
        self.assertIn("root: unknown fields ['surprise']", errors)

    def test_missing_and_unknown_row_fields_are_rejected(self):
        document = self.mutated()
        del document["rows"][0]["logic_family"]
        document["rows"][0]["ordinal"] = 1
        errors = validate_document(document)
        self.assertIn("row 0: missing fields ['logic_family']", errors)
        self.assertIn("row 0: unknown fields ['ordinal']", errors)

    def test_wrong_row_types_report_errors_without_throwing(self):
        document = self.mutated()
        document["rows"][0]["id"] = 17
        document["rows"][0]["canonical_label"] = ["YES"]
        errors = validate_document(document)
        self.assertTrue(any("id must match" in error for error in errors))
        self.assertTrue(any("invalid canonical_label" in error for error in errors))

    def test_policy_drift_is_rejected(self):
        document = self.mutated()
        document["near_duplicate_policy"]["threshold_rule"] = "reject_greater_than"
        document["structural_policy"]["quota_per_family"]["SELECT"] = 3
        errors = validate_document(document)
        self.assertIn("near_duplicate_policy does not match the review-draft contract", errors)
        self.assertIn("structural_policy does not match the review-draft contract", errors)

    def test_partition_swap_and_id_mapping_are_rejected(self):
        document = self.mutated()
        document["rows"][0]["partition"] = "SELECT"
        errors = validate_document(document)
        self.assertTrue(any("DEVELOPMENT: opaque ID order" in error for error in errors))
        self.assertTrue(any("SELECT: opaque ID order" in error for error in errors))

    def test_label_order_swap_is_rejected_even_if_balance_survives(self):
        document = self.mutated()
        first, second = document["rows"][0], document["rows"][1]
        for field in ("canonical_label", "nuisance_label", "material_label"):
            first[field], second[field] = second[field], first[field]
        errors = validate_document(document)
        self.assertIn("DEVELOPMENT: fixed pinned label shuffle drifted", errors)

    def test_conjunctive_but_surface_leakage_is_rejected(self):
        document = self.mutated()
        row = next(row for row in document["rows"] if row["logic_family"] == "CONJUNCTIVE_RULE")
        row["canonical_text"] = row["canonical_text"].replace(" and ", " but ", 1)
        errors = validate_document(document)
        self.assertTrue(any("forbidden token 'but'" in error for error in errors))

    def test_surface_signature_must_be_derived_and_label_balanced(self):
        document = self.mutated()
        row = next(
            row for row in document["rows"]
            if row["partition"] == "SELECT"
            and row["logic_family"] == "MUTUAL_EXCLUSION"
            and row["surface_signature"] == "NEGATED_CONCLUSION"
        )
        row["surface_signature"] = "AFFIRMATIVE_CONCLUSION"
        errors = validate_document(document)
        self.assertTrue(any("does not match derived" in error for error in errors))
        self.assertTrue(any("surface_signature distribution differs by label" in error for error in errors))

    def test_cardinality_subtype_distribution_must_match_across_labels(self):
        document = self.mutated()
        row = next(
            row for row in document["rows"]
            if row["partition"] == "SELECT"
            and row["logic_family"] == "CARDINALITY"
            and row["canonical_label"] == "YES"
            and row["logic_subtype"] == "EXACT_COUNT"
        )
        row["logic_subtype"] = "THRESHOLD"
        errors = validate_document(document)
        self.assertTrue(any("logic_subtype distribution differs by label" in error for error in errors))

    def test_cardinality_subtype_requires_declared_surface_marker(self):
        document = self.mutated()
        row = next(
            row for row in document["rows"]
            if row["logic_family"] == "CARDINALITY" and row["logic_subtype"] == "EXACT_COUNT"
        )
        row["canonical_text"] = row["canonical_text"].replace("exactly", "precisely")
        errors = validate_document(document)
        self.assertTrue(any("does not match first-rule-sentence syntax" in error for error in errors))

    def test_irrelevant_later_marker_cannot_authorize_cardinality_subtype(self):
        document = self.mutated()
        row = next(
            row for row in document["rows"]
            if row["logic_family"] == "CARDINALITY" and row["logic_subtype"] == "EXACT_COUNT"
        )
        row["canonical_text"] = row["canonical_text"].replace("exactly when", "when", 1)
        row["canonical_text"] = row["canonical_text"].replace(
            ". Proposed conclusion:", ". An irrelevant note says exactly when. Proposed conclusion:", 1
        )
        errors = validate_document(document)
        self.assertTrue(any("does not match first-rule-sentence syntax" in error for error in errors))

    def test_relabel_plus_irrelevant_threshold_marker_is_rejected(self):
        document = self.mutated()
        row = next(
            row for row in document["rows"]
            if row["logic_family"] == "CARDINALITY" and row["logic_subtype"] == "EXACT_COUNT"
        )
        row["logic_subtype"] = "THRESHOLD"
        row["canonical_text"] = row["canonical_text"].replace(
            ". Proposed conclusion:", ". An irrelevant note contains at least. Proposed conclusion:", 1
        )
        errors = validate_document(document)
        self.assertTrue(any("syntax 'EXACT_COUNT'" in error for error in errors))

    def test_proposed_answer_is_not_a_conclusion_marker(self):
        document = self.mutated()
        document["rows"][0]["canonical_text"] = document["rows"][0]["canonical_text"].replace(
            "Proposed conclusion:", "Proposed answer:"
        )
        errors = validate_document(document)
        self.assertTrue(any("contiguous 'proposed conclusion' exactly once" in error for error in errors))

    def test_duplicate_or_noncontiguous_conclusion_marker_is_rejected(self):
        duplicate = self.mutated()
        duplicate["rows"][0]["canonical_text"] = (
            "A note says proposed conclusion. " + duplicate["rows"][0]["canonical_text"]
        )
        duplicate_errors = validate_document(duplicate)
        self.assertTrue(any("contiguous 'proposed conclusion' exactly once" in error for error in duplicate_errors))

        separated = self.mutated()
        separated["rows"][0]["canonical_text"] = separated["rows"][0]["canonical_text"].replace(
            "Proposed conclusion:", "Proposed final conclusion:"
        )
        separated_errors = validate_document(separated)
        self.assertTrue(any("contiguous 'proposed conclusion' exactly once" in error for error in separated_errors))

    def test_sibling_texts_must_be_distinct(self):
        document = self.mutated()
        document["rows"][0]["nuisance_text"] = document["rows"][0]["canonical_text"]
        errors = validate_document(document)
        self.assertTrue(any("canonical, nuisance, and material texts must be distinct" in error for error in errors))

    def test_family_quota_and_within_family_balance_are_enforced(self):
        document = self.mutated()
        document["rows"][0]["logic_family"] = "CARDINALITY"
        errors = validate_document(document)
        self.assertTrue(any("DEVELOPMENT/TEMPORAL_ORDER" in error for error in errors))
        self.assertTrue(any("DEVELOPMENT/CARDINALITY" in error for error in errors))

    def test_cross_row_duplicate_is_rejected(self):
        document = self.mutated()
        document["rows"][1]["canonical_text"] = document["rows"][0]["canonical_text"]
        errors = validate_document(document)
        self.assertTrue(any("near duplicate" in error for error in errors))

    def test_inclusive_boundary_is_rejected_by_validator(self):
        document = self.mutated()
        shared = [f"shared{index}" for index in range(43)]
        left = shared + [f"left{index}" for index in range(7)]
        right = shared + [f"right{index}" for index in range(7)]
        document["rows"][0]["canonical_text"] = " ".join(left)
        document["rows"][1]["canonical_text"] = " ".join(right)
        errors = validate_document(document)
        self.assertTrue(any("near duplicate at 0.860000" in error for error in errors))

    def test_material_and_nuisance_label_contracts_are_enforced(self):
        document = self.mutated()
        document["rows"][0]["nuisance_label"] = "NO"
        document["rows"][1]["material_label"] = "NO"
        errors = validate_document(document)
        self.assertTrue(any("nuisance transform must preserve" in error for error in errors))
        self.assertTrue(any("material transform must invert" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
