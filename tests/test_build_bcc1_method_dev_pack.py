"""Tests for the non-promotional BCC-1 frozen-pack builder."""

from __future__ import annotations

import hashlib
import json
import math
import tempfile
import unittest
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import numpy as np

import scripts.build_bcc1_method_dev_pack as builder
from scripts.build_bcc1_method_dev_pack import (
    ANCHOR_STRATEGY,
    FEATURES,
    MIN_VECTOR_L2_NORM,
    REVERSE_ROUTE_ROLE_VALIDATION,
    RIDGE_REGULARIZATION,
    SCHEMA_VERSION,
    TOP_K,
    _anchor_indices,
    _binary_ndcg_at_k,
    _normalize_rows,
    _rank_order,
    build_method_dev_pack,
)


class BCC1MethodDevBuilderTests(unittest.TestCase):
    def test_binary_ndcg_idcg_uses_all_positive_qrels(self) -> None:
        score = _binary_ndcg_at_k(["a", "x"], ["a", "b"], k=2)
        expected = 1.0 / (1.0 + 1.0 / math.log2(3.0))
        self.assertAlmostEqual(score, expected, places=15)
        self.assertAlmostEqual(score, 0.6131471927654584, places=15)

    def test_binary_ndcg_rejects_duplicate_ranked_ids_and_bool_cutoff(self) -> None:
        with self.assertRaisesRegex(ValueError, "duplicate document IDs"):
            _binary_ndcg_at_k(["a", "a"], ["a"], k=2)
        with self.assertRaisesRegex(ValueError, "positive integer"):
            _binary_ndcg_at_k(["a"], ["a"], k=True)

    def test_cosine_normalization_rejects_zero_and_near_zero_rows(self) -> None:
        for vector in (
            np.asarray([[0.0, 0.0]], dtype=np.float64),
            np.asarray([[MIN_VECTOR_L2_NORM / 2.0, 0.0]], dtype=np.float64),
            np.asarray([[1.0, 0.0]], dtype=np.float64) @ np.zeros((2, 2), dtype=np.float64),
        ):
            with self.subTest(vector=vector.tolist()):
                with self.assertRaisesRegex(ValueError, "zero or near-zero"):
                    _normalize_rows(vector)

    def test_ranking_ties_use_canonical_document_id(self) -> None:
        scores = np.asarray([0.5, 0.9, 0.9, 0.5], dtype=np.float64)
        doc_ids = np.asarray(["z", "b", "a", "c"], dtype=str)
        order = _rank_order(scores, doc_ids, k=4)
        self.assertEqual(doc_ids[order].tolist(), ["a", "b", "c", "z"])

    def _source(
        self,
        root: Path,
        *,
        relevant_doc: str = "d2",
        fit_idx: Sequence[int] = (0, 1),
        pack_name: str = "toy_minilm_to_new_pack",
        dataset: str = "Toy",
        swap_model: str = "new",
        transform_delta: float = 0.0,
    ) -> Path:
        pack_dir = root / "research" / "drift_recovery" / "out" / "estimator" / "packs"
        pack_dir.mkdir(parents=True, exist_ok=True)
        prefix = pack_dir / pack_name
        do = np.asarray(
            [[1.0, 0.0], [0.0, 1.0], [0.9, 0.1], [0.1, 0.9], [-1.0, 0.0], [0.0, -1.0]],
            dtype=np.float64,
        )
        transform = np.asarray(
            [[0.8 + transform_delta, 0.3], [-0.2, 1.1 - transform_delta]],
            dtype=np.float64,
        )
        dor = do @ transform
        qd = np.asarray([[0.8, 0.2], [0.1, 0.9], [-0.9, 0.0]], dtype=np.float64)
        doc_ids = np.asarray([f"d{index}" for index in range(len(do))], dtype=str)
        query_ids = np.asarray(["q0", "q1", "q2"], dtype=str)
        inherited_fit_idx = np.asarray(fit_idx, dtype=np.int64)
        arrays_path = prefix.with_suffix(".npz")
        np.savez_compressed(
            arrays_path,
            Do=do,
            Dor=dor,
            Qd=qd,
            doc_ids=doc_ids,
            query_ids=query_ids,
            fit_idx=inherited_fit_idx,
            extra__leakage_safe_fit_idx=inherited_fit_idx,
        )
        qrels = {
            "q0": {relevant_doc: 1.0},
            "q1": {"d3": 1.0},
            "q2": {"d4": 1.0},
        }
        payload = {
            "record_type": "embedding_pack",
            "format_version": 1,
            "arrays_file": arrays_path.name,
            "arrays_sha256": hashlib.sha256(arrays_path.read_bytes()).hexdigest(),
            "qrels": qrels,
            "metadata": {
                "dataset": dataset,
                "models": {"old": "old", "swap": swap_model},
                "k": 10,
                "sample_docs": 6,
            },
        }
        prefix.with_suffix(".json").write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        return root

    def _build(self, source: Path, output: Path) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
        pack_path = output / "pack.json"
        manifest_path = output / "manifest.json"
        result = build_method_dev_pack(source, pack_path, manifest_path)
        return (
            json.loads(pack_path.read_text(encoding="utf-8")),
            json.loads(manifest_path.read_text(encoding="utf-8")),
            result,
        )

    def test_builds_strict_method_dev_schema_and_checksum(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            pack, manifest, result = self._build(self._source(root / "source"), root / "out")
            self.assertEqual(pack["evidence_mode"], "METHOD_DEV")
            self.assertTrue(pack["sampled"])
            self.assertEqual(len(pack["rows"]), 3)
            self.assertEqual(set(pack["rows"][0]["scores"]), {"reverse", "forward", "native_new", "mismatch"})
            self.assertTrue(all(block["role"] == "METHOD_DEV" for block in manifest["blocks"]))
            self.assertTrue(all(block["dataset_family_id"] == "toy" for block in manifest["blocks"]))
            self.assertTrue(all("transition_family_id" in block for block in manifest["blocks"]))
            pack_bytes = (root / "out" / "pack.json").read_bytes()
            self.assertEqual(manifest["pack_sha256"], hashlib.sha256(pack_bytes).hexdigest())
            implementation_sha256 = hashlib.sha256(Path(builder.__file__).read_bytes()).hexdigest()
            derivation = {
                "schema_version": SCHEMA_VERSION,
                "evidence_mode": "METHOD_DEV",
                "source_digest": result["source_digest"],
                "implementation_sha256": implementation_sha256,
                "anchor_strategy": ANCHOR_STRATEGY,
                "ridge_regularization": RIDGE_REGULARIZATION,
                "top_k": TOP_K,
                "minimum_vector_l2_norm": MIN_VECTOR_L2_NORM,
                "metric": {
                    "metric_name": "ndcg_at_10",
                    "gain": "binary_positive_qrel",
                    "idcg_population": "all_positive_qrels_for_query",
                    "query_inclusion": "queries_with_at_least_one_positive_qrel",
                    "ranking_tie_break": "score_desc_document_id_asc",
                },
                "features": FEATURES,
            }
            derivation_sha256 = hashlib.sha256(
                json.dumps(
                    derivation,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            self.assertEqual(result["derivation_sha256"], derivation_sha256)
            self.assertEqual(
                pack["pack_id"],
                f"bcc1-method-dev-derivation-{derivation_sha256}",
            )
            self.assertEqual(manifest["pack_id"], pack["pack_id"])
            self.assertEqual(pack["metric"]["name"], "ndcg_at_10")
            self.assertEqual(
                manifest["metric_contract"],
                {
                    "metric_name": "ndcg_at_10",
                    "implementation_sha256": implementation_sha256,
                    "gain": "binary_positive_qrel",
                    "idcg_population": "all_positive_qrels_for_query",
                    "cutoff": 10,
                    "query_inclusion": "queries_with_at_least_one_positive_qrel",
                    "qrels_sha256": manifest["metric_contract"]["qrels_sha256"],
                    "pack_qrels_binding_sha256": manifest["metric_contract"]["pack_qrels_binding_sha256"],
                    "ranking_tie_break": "score_desc_document_id_asc",
                },
            )
            expected_metric_binding = hashlib.sha256(
                json.dumps(
                    {
                        "schema_version": "chelatedai.bcc1.metric-binding.v1",
                        "pack_sha256": manifest["pack_sha256"],
                        "qrels_sha256": manifest["metric_contract"]["qrels_sha256"],
                    },
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    allow_nan=False,
                ).encode("utf-8")
            ).hexdigest()
            self.assertEqual(
                manifest["metric_contract"]["pack_qrels_binding_sha256"],
                expected_metric_binding,
            )
            self.assertEqual(
                manifest["reverse_route_role_validation"],
                REVERSE_ROUTE_ROLE_VALIDATION,
            )
            self.assertEqual(
                REVERSE_ROUTE_ROLE_VALIDATION,
                "UNVALIDATED_DOCUMENT_ROLE_REVERSE_PROXY",
            )
            self.assertEqual(
                ANCHOR_STRATEGY["anchor_representation_role"],
                "document",
            )
            self.assertEqual(
                ANCHOR_STRATEGY["reverse_application_role"],
                "query",
            )
            self.assertTrue(all(item["qrels_free"] for item in manifest["features"]))
            self.assertTrue(all("independence_group_id" in row for row in pack["rows"]))

    def test_features_and_anchors_ignore_qrels_and_inherited_fit_index(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            first, _, _ = self._build(
                self._source(
                    root / "source-a",
                    relevant_doc="d2",
                    fit_idx=(0, 1),
                ),
                root / "out-a",
            )
            second, _, _ = self._build(
                self._source(
                    root / "source-b",
                    relevant_doc="d5",
                    fit_idx=(4, 5),
                ),
                root / "out-b",
            )
            first_features = [row["features"] for row in first["rows"]]
            second_features = [row["features"] for row in second["rows"]]
            self.assertEqual(first_features, second_features)
            first_scores = [row["scores"] for row in first["rows"]]
            second_scores = [row["scores"] for row in second["rows"]]
            self.assertNotEqual(first_scores, second_scores)
            doc_ids = np.asarray([f"d{index}" for index in range(6)], dtype=str)
            first_anchors = _anchor_indices("toy", doc_ids)
            permuted_doc_ids = doc_ids[[3, 1, 5, 0, 4, 2]]
            second_anchors = _anchor_indices("toy", permuted_doc_ids)
            self.assertEqual(
                sorted(doc_ids[first_anchors].tolist()),
                sorted(permuted_doc_ids[second_anchors].tolist()),
            )
            self.assertEqual(len(first_anchors), 3)

    def test_inherited_fit_index_is_validated_but_not_used(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._source(root / "source", fit_idx=(0, 99))
            with self.assertRaisesRegex(ValueError, "out of bounds"):
                build_method_dev_pack(
                    source,
                    root / "out" / "pack.json",
                    root / "out" / "manifest.json",
                )

    def test_independence_groups_match_across_encoder_variants(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            self._source(
                source,
                pack_name="toy_old_to_new_a_pack",
                swap_model="new-a",
            )
            self._source(
                source,
                pack_name="toy_old_to_new_b_pack",
                swap_model="new-b",
                transform_delta=0.1,
            )
            pack, manifest, _ = self._build(source, root / "out")
            groups_by_raw_query: Dict[str, set] = {}
            blocks_by_raw_query: Dict[str, set] = {}
            for row in pack["rows"]:
                raw_query = str(row["query_id"]).rsplit("::", 1)[1]
                groups_by_raw_query.setdefault(raw_query, set()).add(row["independence_group_id"])
                blocks_by_raw_query.setdefault(raw_query, set()).add(row["block_id"])
            self.assertEqual(set(groups_by_raw_query), {"q0", "q1", "q2"})
            self.assertTrue(all(len(groups) == 1 for groups in groups_by_raw_query.values()))
            self.assertTrue(all(len(blocks) == 2 for blocks in blocks_by_raw_query.values()))
            self.assertEqual(
                {block["dataset_family_id"] for block in manifest["blocks"]},
                {"toy"},
            )
            self.assertEqual(
                len({block["transition_family_id"] for block in manifest["blocks"]}),
                2,
            )

    def test_transition_family_is_independent_of_dataset_family(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "source"
            self._source(
                source,
                pack_name="toy_a_old_to_new_pack",
                dataset="Toy A",
            )
            self._source(
                source,
                pack_name="toy_b_old_to_new_pack",
                dataset="Toy B",
                transform_delta=0.1,
            )
            _, manifest, _ = self._build(source, root / "out")
            self.assertEqual(
                {block["dataset_family_id"] for block in manifest["blocks"]},
                {"toy-a", "toy-b"},
            )
            self.assertEqual(
                {block["transition_family_id"] for block in manifest["blocks"]},
                {"old-to-new"},
            )

    def test_positive_qrel_outside_sampled_corpus_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._source(root / "source")
            pack_dir = source / "research" / "drift_recovery" / "out" / "estimator" / "packs"
            metadata_path = next(pack_dir.glob("*_pack.json"))
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
            payload["qrels"]["q0"]["not-in-sampled-corpus"] = 1.0
            metadata_path.write_text(
                json.dumps(payload, sort_keys=True),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                ValueError,
                "positive qrels outside the sampled corpus",
            ):
                build_method_dev_pack(
                    source,
                    root / "out" / "pack.json",
                    root / "out" / "manifest.json",
                )

    def test_feature_hashes_are_mandatory_exact_and_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._source(root / "source")
            _, first_manifest, first_result = self._build(source, root / "out-a")
            _, second_manifest, second_result = self._build(source, root / "out-b")
            self.assertEqual(first_manifest, second_manifest)
            self.assertEqual(first_result["source_digest"], second_result["source_digest"])
            implementation_sha256 = hashlib.sha256(Path(builder.__file__).read_bytes()).hexdigest()
            expected_keys = {
                "name",
                "qrels_free",
                "inference_available",
                "source",
                "availability_stage",
                "execution_cost_class",
                "implementation_sha256",
                "provenance_sha256",
            }
            for feature in first_manifest["features"]:
                self.assertEqual(set(feature), expected_keys)
                self.assertEqual(feature["implementation_sha256"], implementation_sha256)
                provenance = {
                    "feature_name": feature["name"],
                    "feature_source": feature["source"],
                    "availability_stage": feature["availability_stage"],
                    "execution_cost_class": feature["execution_cost_class"],
                    "source_digest": first_result["source_digest"],
                    "anchor_strategy": ANCHOR_STRATEGY,
                    "implementation_sha256": implementation_sha256,
                }
                expected_provenance = hashlib.sha256(
                    json.dumps(
                        provenance,
                        sort_keys=True,
                        separators=(",", ":"),
                        ensure_ascii=False,
                        allow_nan=False,
                    ).encode("utf-8")
                ).hexdigest()
                self.assertEqual(feature["provenance_sha256"], expected_provenance)
            by_name = {feature["name"]: feature for feature in first_manifest["features"]}
            self.assertEqual(
                {feature["availability_stage"] for feature in first_manifest["features"]},
                {"PRE_SEARCH", "DUAL_READ_METHOD_DEV"},
            )
            self.assertEqual(
                by_name["query_cycle_l2"]["execution_cost_class"],
                "zero_search",
            )
            self.assertEqual(
                by_name["forward_top1"]["execution_cost_class"],
                "dual_search_probe",
            )
            self.assertEqual(
                sum(feature["availability_stage"] == "PRE_SEARCH" for feature in first_manifest["features"]),
                2,
            )
            self.assertEqual(
                sum(feature["availability_stage"] == "DUAL_READ_METHOD_DEV" for feature in first_manifest["features"]),
                19,
            )

    def test_refuses_to_overwrite_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = self._source(root / "source")
            pack_path = root / "out" / "pack.json"
            manifest_path = root / "out" / "manifest.json"
            build_method_dev_pack(source, pack_path, manifest_path)
            with self.assertRaises(FileExistsError):
                build_method_dev_pack(source, pack_path, manifest_path)


if __name__ == "__main__":
    unittest.main()
