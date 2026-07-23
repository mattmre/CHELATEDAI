"""Contract and artifact tests for powered Obj A v2."""

from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from research.drift_recovery.artifacts import EmbeddingPack
from research.drift_recovery.estimator.features import leakage_safe_fit_indices
from research.drift_recovery.estimator.objA import block_loo_analysis
from research.drift_recovery.estimator.objA_v2 import (
    ALLOWED_VERDICTS,
    ESTIMATOR_DIR,
    EXPECTED_DATASETS,
    EXPECTED_ENCODERS,
    EXPECTED_NEW_NAMES,
    OUT_DIR,
    PREREG_HASH_PATH,
    PREREG_PATH,
    REPO_ROOT,
    analyze_objA_v2,
    collect_regimes_v2,
    run_objA_v2,
    validate_registered_regimes_v2,
    verdict_for_v2,
    verify_frozen_preregistration_v2,
)
from research.drift_recovery.harness_bridge import assert_harness_parity


V1_HASHES = {
    "research/drift_recovery/estimator/prereg_objA.md": "4d1aac9f92390e9d0298afe40a51dc5677456d8a0a765e50a778e550f53248a2",
    "research/drift_recovery/estimator/prereg_objA.json": "ab06418b5d331cdb0c19a79f021d6125a0ae58073247bd8bdc85962ca5ae658f",
    "research/drift_recovery/estimator/prereg_objA.sha256": "72e9358d1b9f66e91dbbdcaed10eda689426aea353995d1e9e36f27c404c111b",
    "research/drift_recovery/out/estimator/objA_validation.md": "6378625680410ce15294a1a753f0a321658c9a1688e034f4fa086cd56366e154",
    "research/drift_recovery/out/estimator/objA_validation.json": "4ec97cf0e4d435e51f04af12ed1dfc910d6c30c8cda2d2ba67b5cadc12c40a5c",
    "research/drift_recovery/out/estimator/objA_REPORT.md": "ecc0bc6f85a97350ac02e2ebc5cab68b6e34c9c874fac51a1509a60aca7e4d3e",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _prereg():
    return json.loads(PREREG_PATH.read_text(encoding="utf-8"))


def _synthetic_records():
    records = []
    for dataset_index, dataset in enumerate(EXPECTED_DATASETS):
        for encoder_index, encoder in enumerate(EXPECTED_ENCODERS):
            value = float(3 * dataset_index + encoder_index + 1)
            records.append(
                {
                    "name": f"{dataset}_{encoder}",
                    "dataset": dataset,
                    "encoder_family": encoder,
                    "oracle_margin_mean": value,
                    "oracle_gap": value / 10.0,
                    "R": 0.2 + value / 30.0,
                }
            )
    return records


class TestObjAV2Preregistration(unittest.TestCase):
    def test_hash_lock_is_valid_and_exact(self) -> None:
        actual = verify_frozen_preregistration_v2()
        self.assertEqual(set(actual), {"prereg_objA_v2.md", "prereg_objA_v2.json"})
        locked = {
            line.split(maxsplit=1)[1]: line.split(maxsplit=1)[0]
            for line in PREREG_HASH_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        self.assertEqual(actual, locked)

    def test_v1_preregistration_and_outputs_are_byte_unchanged(self) -> None:
        for relative, digest in V1_HASHES.items():
            self.assertEqual(_sha256(REPO_ROOT / relative), digest, relative)

    def test_registered_matrix_is_exact_4_by_3(self) -> None:
        regimes = validate_registered_regimes_v2(_prereg())
        self.assertEqual(len(regimes), 12)
        self.assertEqual(
            {(row["dataset"], row["encoder_family"]) for row in regimes},
            {
                (dataset, encoder)
                for dataset in EXPECTED_DATASETS
                for encoder in EXPECTED_ENCODERS
            },
        )
        self.assertEqual(
            {row["name"] for row in regimes if row["source"] == "build_new"},
            EXPECTED_NEW_NAMES,
        )
        self.assertEqual(sum(row["source"] == "reuse_existing" for row in regimes), 6)
        self.assertTrue(all(row["seed"] == 42 for row in regimes))
        self.assertTrue(all(row["anchor_fraction"] == 0.4 for row in regimes))

    def test_runner_verifies_freeze_before_collecting_or_building(self) -> None:
        calls = []

        def verified():
            calls.append("verify")
            return {"prereg_objA_v2.md": "x", "prereg_objA_v2.json": "y"}

        def stopped(*args, **kwargs):
            del args, kwargs
            calls.append("collect")
            raise RuntimeError("stop after ordering check")

        with patch(
            "research.drift_recovery.estimator.objA_v2.verify_frozen_preregistration_v2",
            side_effect=verified,
        ), patch(
            "research.drift_recovery.estimator.objA_v2.collect_regimes_v2",
            side_effect=stopped,
        ):
            with self.assertRaisesRegex(RuntimeError, "stop after ordering check"):
                run_objA_v2(device="cpu")
        self.assertEqual(calls, ["verify", "collect"])

    def test_preregistration_files_precede_every_new_pack_file(self) -> None:
        prereg_latest = max(
            path.stat().st_mtime
            for path in (
                ESTIMATOR_DIR / "prereg_objA_v2.md",
                ESTIMATOR_DIR / "prereg_objA_v2.json",
                PREREG_HASH_PATH,
            )
        )
        regimes = validate_registered_regimes_v2(_prereg())
        new_files = []
        for row in regimes:
            if row["source"] != "build_new":
                continue
            prefix = REPO_ROOT / row["pack_prefix"]
            new_files.extend((prefix.with_suffix(".npz"), prefix.with_suffix(".json")))
        self.assertTrue(all(path.exists() for path in new_files))
        self.assertTrue(all(prereg_latest <= path.stat().st_mtime for path in new_files))


class TestObjAV2Statistics(unittest.TestCase):
    def test_block_integrity_for_twelve_cells(self) -> None:
        records = _synthetic_records()
        by_name = {row["name"]: row for row in records}
        for field, count in (("dataset", 4), ("encoder_family", 3)):
            result = block_loo_analysis(records, field)
            self.assertEqual(result["held_out_unit_count"], count)
            self.assertEqual(result["cell_n"], 12)
            self.assertEqual({row["regime"] for row in result["rows"]}, set(by_name))
            for row in result["rows"]:
                self.assertEqual(
                    set(row["predictions"]),
                    {"oracle_margin_mean", "oracle_gap_ols", "mean_R"},
                )
                self.assertTrue(
                    all(
                        by_name[name][field] != row["held_out_block"]
                        for name in row["training_regimes"]
                    )
                )
            self.assertTrue(
                all(metric["cell_n"] == 12 for metric in result["metrics"].values())
            )

    def test_frozen_three_label_truth_table(self) -> None:
        self.assertEqual(verdict_for_v2(False, 4, 3), "NEGATIVE")
        self.assertEqual(verdict_for_v2(False, 3, 2), "NEGATIVE")
        self.assertEqual(verdict_for_v2(True, 4, 3), "POSITIVE")
        self.assertEqual(
            verdict_for_v2(True, 3, 3), "PROMISING-BUT-UNDERPOWERED"
        )
        self.assertEqual(
            verdict_for_v2(True, 4, 2), "PROMISING-BUT-UNDERPOWERED"
        )
        self.assertEqual(
            set(ALLOWED_VERDICTS),
            {"POSITIVE", "PROMISING-BUT-UNDERPOWERED", "NEGATIVE"},
        )

    def test_degenerate_new_pack_is_excluded_without_imputation(self) -> None:
        registered = {
            "name": "degenerate",
            "dataset": "ArguAna",
            "encoder_family": "e5-base-v2",
            "swap_model": "intfloat/e5-base-v2",
            "anchor_fraction": 0.4,
            "seed": 42,
            "source": "build_new",
            "pack_prefix": str(
                Path(tempfile.gettempdir()) / "objA-v2-never-written" / "pack"
            ),
        }
        with patch(
            "research.drift_recovery.estimator.objA_v2.validate_registered_regimes_v2",
            return_value=[registered],
        ), patch(
            "research.drift_recovery.estimator.objA_v2.build_regime_pack",
            side_effect=RuntimeError("oracle gap is non-positive (-0.01)"),
        ):
            records, inventory = collect_regimes_v2({}, device="cpu")
        self.assertEqual(records, [])
        self.assertEqual(inventory[0]["status"], "skipped_degenerate_gap")

    def test_non_degenerate_build_failure_is_not_softened_into_a_skip(self) -> None:
        registered = {
            "name": "broken",
            "dataset": "ArguAna",
            "encoder_family": "e5-base-v2",
            "swap_model": "intfloat/e5-base-v2",
            "anchor_fraction": 0.4,
            "seed": 42,
            "source": "build_new",
            "pack_prefix": str(
                Path(tempfile.gettempdir()) / "objA-v2-never-written-broken" / "pack"
            ),
        }
        with patch(
            "research.drift_recovery.estimator.objA_v2.validate_registered_regimes_v2",
            return_value=[registered],
        ), patch(
            "research.drift_recovery.estimator.objA_v2.build_regime_pack",
            side_effect=RuntimeError("parity failure"),
        ):
            with self.assertRaisesRegex(RuntimeError, "parity failure"):
                collect_regimes_v2({}, device="cpu")


class TestObjAV2FrozenPacksAndOutputs(unittest.TestCase):
    def test_all_six_new_packs_have_parity_frozen_scores_and_no_qrel_leakage(self) -> None:
        regimes = validate_registered_regimes_v2(_prereg())
        for row in regimes:
            if row["source"] != "build_new":
                continue
            prefix = REPO_ROOT / row["pack_prefix"]
            pack = EmbeddingPack.load(prefix)
            self.assertEqual(pack.metadata["dataset"], row["dataset"])
            self.assertEqual(pack.metadata["models"]["swap"], row["swap_model"])
            self.assertEqual(pack.metadata["seed"], 42)
            self.assertEqual(pack.metadata["anchor_fraction"], 0.4)
            self.assertEqual(pack.metadata["max_queries"], 100)
            self.assertEqual(pack.metadata["sample_docs"], 1200)
            self.assertEqual(
                np.asarray(pack.extra_arrays["leakage_safe_fit_idx"]).tolist(),
                np.asarray(pack.fit_idx).tolist(),
            )
            fit_idx = leakage_safe_fit_indices(pack)
            positive_ids = {
                str(doc_id)
                for qrels in pack.qrels.values()
                for doc_id, score in qrels.items()
                if float(score) > 0.0
            }
            fit_ids = set(np.asarray(pack.doc_ids, dtype=str)[fit_idx].tolist())
            self.assertFalse(fit_ids.intersection(positive_ids), row["name"])
            self.assertEqual(
                set(pack.per_query_scores).issuperset({"floor", "oracle", "ridge"}),
                True,
            )
            for name in ("floor", "oracle", "ridge"):
                scores = np.asarray(pack.per_query_scores[name], dtype=np.float64)
                self.assertEqual(scores.shape, (len(pack.query_ids),))
                self.assertTrue(np.all(np.isfinite(scores)))
            parity = assert_harness_parity(pack, atol=1e-12)
            self.assertLessEqual(max(parity.values()), 1e-12)
            if row["encoder_family"] == "e5-base-v2":
                self.assertEqual(pack.metadata["drift_manifest"]["swap_dim"], 768)
                self.assertEqual(
                    pack.metadata["drift_manifest"]["swap_model"],
                    "intfloat/e5-base-v2",
                )

    def test_validation_outputs_cover_both_schemes_and_novel_blocks(self) -> None:
        result_path = OUT_DIR / "objA_v2_validation.json"
        self.assertTrue(result_path.exists())
        result = json.loads(result_path.read_text(encoding="utf-8"))
        self.assertIn(result["verdict"], ALLOWED_VERDICTS)
        self.assertEqual(result["counts"]["registered_cell_n"], 12)
        self.assertEqual(result["counts"]["pseudo_replicate_cell_n"], 0)
        for scheme in ("dataset", "encoder_family"):
            analysis = result["block_loo"][scheme]
            self.assertIsNotNone(analysis)
            for model in ("oracle_margin_mean", "oracle_gap_ols", "mean_R"):
                self.assertIn("spearman", analysis["metrics"][model])
                self.assertIn("mae", analysis["metrics"][model])
                self.assertEqual(
                    analysis["metrics"][model]["cell_n"],
                    result["counts"]["cell_n"],
                )
        self.assertEqual(
            result["novel_holdouts"]["ArguAna"]["cell_n"], 3
        )
        self.assertEqual(
            result["novel_holdouts"]["e5-base-v2"]["cell_n"], 4
        )
        self.assertIn("partial_spearman_margin_R_controlling_oracle_gap", result["partial_spearman"])
        self.assertEqual(len(result["regimes"]), result["counts"]["cell_n"])
        self.assertTrue(
            result["preregistration"]["ordering_audit"][
                "preregistration_precedes_every_new_pack"
            ]
        )

    def test_analyzer_never_emits_a_legacy_verdict_label(self) -> None:
        records = _synthetic_records()
        inventory = [{**row, "status": "available"} for row in records]
        result = analyze_objA_v2(records, inventory, {})
        self.assertIn(result["verdict"], ALLOWED_VERDICTS)
        self.assertNotIn(result["verdict"], {"UNDERPOWERED-NEGATIVE"})


if __name__ == "__main__":
    unittest.main()
