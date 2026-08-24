import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from qwen_scope_real_smoke import (
    PassObservation,
    SmokeConfig,
    SmokeFailure,
    COMMIT_RECEIPT_FILENAME,
    _atomic_json,
    _fsync_directory,
    _prepare_commit_receipt,
    _promote_directory_noreplace,
    _publish_commit_receipt,
    _verify_official_sae_state,
    run_smoke,
    verify_archived_copy_dir,
    verify_result_dir,
)


class FixtureBackend:
    def __init__(self, *, changed_digest=False, changed_indices=False, peak_rss_gib=0.05):
        self.changed_digest = changed_digest
        self.changed_indices = changed_indices
        self.peak_rss_gib = peak_rss_gib
        self.calls = 0

    def preflight(self, _config):
        return {
            "cuda_device": "fixture-cuda",
            "free_gpu_gib": 64.0,
            "total_gpu_gib": 128.0,
            "free_disk_gib": 100.0,
        }

    def load(self, config):
        return {
            "model_layer_count": 24,
            "sae_d_model": config.hidden_size,
            "sae_d_sae": config.sae_width,
            "sae_top_k": config.top_k,
            "sae_file_sha256_verified": True,
            "sae_four_tensor_contract_verified": True,
            "hook_layer_index": config.layer_index,
            "model_class": "FixtureQwen",
            "transformers_version": "4.57.3",
        }

    def run_pass(self, _prompt, config):
        self.calls += 1
        digest = "b" * 64 if self.changed_digest and self.calls == 2 else "a" * 64
        indices = tuple(range(config.top_k))
        if self.changed_indices and self.calls == 2:
            indices = tuple(range(1, config.top_k + 1))
        return PassObservation(
            residual_shape=(1, 7, config.hidden_size),
            residual_digest=digest,
            selected_indices=indices,
            selected_values=tuple(1.0 - index / 1000.0 for index in range(config.top_k)),
        )

    def measurements(self):
        return {
            "peak_rss_gib": self.peak_rss_gib,
            "peak_cuda_allocated_gib": 0.01,
            "peak_cuda_reserved_gib": 0.02,
            "backend_elapsed_seconds": 0.03,
        }


def fixture_config(output_dir):
    return SmokeConfig(output_dir=Path(output_dir))


def rewrite_canonical(path, payload):
    path.write_bytes((json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8"))


class TestQwenScopeRealSmoke(unittest.TestCase):
    def test_fixture_pass_writes_atomic_artifact_and_digest_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "result"
            config = fixture_config(target)
            artifact = run_smoke(config, FixtureBackend())
            artifact_path = target / "qwen_scope_real_smoke.json"
            manifest_path = target / "manifest.json"
            self.assertEqual(artifact["status"], "PASS")
            self.assertTrue(artifact_path.is_file())
            self.assertTrue(manifest_path.is_file())
            self.assertTrue((target / COMMIT_RECEIPT_FILENAME).is_file())
            retained = json.loads(artifact_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertFalse(retained["inputs"]["prompt_retained"])
            self.assertNotIn("prompt", retained["inputs"])
            self.assertEqual(retained["observation"]["residual_shape"], [1, 7, 2048])
            self.assertEqual(retained["observation"]["selected_feature_count"], 100)
            actual_digest = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
            self.assertEqual(manifest["artifacts"][0]["sha256"], actual_digest)
            self.assertEqual(manifest["artifacts"][0]["byte_size"], artifact_path.stat().st_size)
            self.assertEqual(verify_result_dir(target)["status"], "PASS")
            self.assertEqual(list(Path(temporary).glob(".result.stage-*")), [])

    def test_repeat_residual_change_fails_without_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "result"
            with self.assertRaisesRegex(SmokeFailure, "residual digest changed"):
                run_smoke(fixture_config(target), FixtureBackend(changed_digest=True))
            self.assertFalse(target.exists())

    def test_repeat_topk_change_fails_without_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "result"
            with self.assertRaisesRegex(SmokeFailure, "indices changed"):
                run_smoke(fixture_config(target), FixtureBackend(changed_indices=True))
            self.assertFalse(target.exists())

    def test_resource_overrun_fails_without_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "result"
            with self.assertRaisesRegex(SmokeFailure, "peak RSS"):
                run_smoke(fixture_config(target), FixtureBackend(peak_rss_gib=33.0))
            self.assertFalse(target.exists())

    def test_wrong_hidden_size_fails(self):
        class WrongShapeBackend(FixtureBackend):
            def run_pass(self, _prompt, config):
                return PassObservation(
                    (1, 3, config.hidden_size + 1),
                    "a" * 64,
                    tuple(range(config.top_k)),
                    tuple(0.2 for _ in range(config.top_k)),
                )

        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary) / "result"
            with self.assertRaisesRegex(SmokeFailure, "hidden size"):
                run_smoke(fixture_config(target), WrongShapeBackend())

    def test_real_cli_refuses_without_double_opt_in_before_backend_construction(self):
        with tempfile.TemporaryDirectory() as temporary:
            environment = os.environ.copy()
            environment.pop("CHELATED_QWEN_SCOPE_REAL_SMOKE", None)
            result = subprocess.run(
                [sys.executable, "qwen_scope_real_smoke.py", "--output-dir", temporary],
                cwd=Path(__file__).parent,
                env=environment,
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(result.returncode, 2)
            self.assertIn("REFUSED", result.stderr)
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_nonfinite_or_weakened_api_gates_are_rejected(self):
        cases = {
            "wall_zero": {"max_wall_seconds": 0.0},
            "wall_over": {"max_wall_seconds": 1800.1},
            "wall_nan": {"max_wall_seconds": math.nan},
            "wall_inf": {"max_wall_seconds": math.inf},
            "rss_zero": {"max_rss_gib": 0.0},
            "rss_over": {"max_rss_gib": 32.1},
            "rss_nan": {"max_rss_gib": math.nan},
            "rss_inf": {"max_rss_gib": math.inf},
            "disk_low": {"min_free_disk_gib": 11.9},
            "disk_nan": {"min_free_disk_gib": math.nan},
            "disk_inf": {"min_free_disk_gib": math.inf},
            "gpu_low": {"min_free_gpu_gib": 11.9},
            "gpu_nan": {"min_free_gpu_gib": math.nan},
            "gpu_inf": {"min_free_gpu_gib": math.inf},
            "atol_changed": {"feature_atol": 1e-3},
            "atol_nan": {"feature_atol": math.nan},
            "atol_inf": {"feature_atol": math.inf},
        }
        for label, changes in cases.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                target = Path(temporary) / "result"
                config = replace(fixture_config(target), **changes)
                with self.assertRaises(SmokeFailure):
                    run_smoke(config, FixtureBackend())
                self.assertFalse(target.exists())

    def test_cli_rejects_weakened_wall_gate_before_model_load(self):
        with tempfile.TemporaryDirectory() as parent:
            target = Path(parent) / "result"
            environment = os.environ.copy()
            environment["CHELATED_QWEN_SCOPE_REAL_SMOKE"] = "1"
            result = subprocess.run(
                [
                    sys.executable,
                    "qwen_scope_real_smoke.py",
                    "--allow-real-model",
                    "--output-dir",
                    str(target),
                    "--max-wall-seconds",
                    "1800.1",
                ],
                cwd=Path(__file__).parent,
                env=environment,
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("max_wall_seconds", result.stderr)
            self.assertFalse(target.exists())

    def test_small_official_four_tensor_contract_fixture(self):
        class ShapeOnly:
            def __init__(self, shape):
                self.shape = shape

        with tempfile.TemporaryDirectory() as temporary:
            config = fixture_config(temporary)
            state = {
                "W_enc": ShapeOnly((32768, 2048)),
                "W_dec": ShapeOnly((2048, 32768)),
                "b_enc": ShapeOnly((32768,)),
                "b_dec": ShapeOnly((2048,)),
            }
            _verify_official_sae_state(state, config)

    def test_official_four_tensor_contract_rejects_wrong_decoder_shape(self):
        class ShapeOnly:
            def __init__(self, shape):
                self.shape = shape

        with tempfile.TemporaryDirectory() as temporary:
            config = fixture_config(temporary)
            state = {
                "W_enc": ShapeOnly((32768, 2048)),
                "W_dec": ShapeOnly((32768, 2048)),
                "b_enc": ShapeOnly((32768,)),
                "b_dec": ShapeOnly((2048,)),
            }
            with self.assertRaisesRegex(SmokeFailure, "W_dec shape"):
                _verify_official_sae_state(state, config)

    def test_existing_target_is_rejected_without_stale_replacement(self):
        with tempfile.TemporaryDirectory() as parent:
            target = Path(parent) / "result"
            target.mkdir()
            marker = target / "old.txt"
            marker.write_text("old", encoding="utf-8")
            with self.assertRaisesRegex(SmokeFailure, "must be absent"):
                run_smoke(fixture_config(target), FixtureBackend())
            self.assertEqual(marker.read_text(encoding="utf-8"), "old")

    def test_manifest_write_interruption_leaves_no_final_or_stage(self):
        with tempfile.TemporaryDirectory() as parent:
            target = Path(parent) / "result"
            calls = 0

            def interrupt_second_write(path, payload):
                nonlocal calls
                calls += 1
                if calls == 2:
                    raise OSError("simulated manifest interruption")
                return _atomic_json(path, payload)

            with mock.patch("qwen_scope_real_smoke._atomic_json", side_effect=interrupt_second_write):
                with self.assertRaisesRegex(OSError, "manifest interruption"):
                    run_smoke(fixture_config(target), FixtureBackend())
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(parent).glob(".result.stage-*")), [])

    def test_delayed_staged_write_cannot_publish_pass_beyond_wall_ceiling(self):
        with tempfile.TemporaryDirectory() as parent:
            target = Path(parent) / "result"
            config = replace(fixture_config(target), max_wall_seconds=0.05)
            calls = 0

            def delay_first_write(path, payload):
                nonlocal calls
                calls += 1
                if calls == 1:
                    time.sleep(0.1)
                return _atomic_json(path, payload)

            with mock.patch("qwen_scope_real_smoke._atomic_json", side_effect=delay_first_write):
                with self.assertRaisesRegex(SmokeFailure, "wall-time ceiling exceeded"):
                    run_smoke(config, FixtureBackend())
            self.assertFalse(target.exists())
            self.assertEqual(list(Path(parent).glob(".result.stage-*")), [])

    def test_delayed_promotion_cannot_receive_commit_past_wall_ceiling(self):
        with tempfile.TemporaryDirectory() as parent:
            target = Path(parent) / "result"
            config = replace(fixture_config(target), max_wall_seconds=0.05)

            def delayed_promotion(stage_dir, final_dir):
                time.sleep(0.1)
                return _promote_directory_noreplace(stage_dir, final_dir)

            with mock.patch(
                "qwen_scope_real_smoke._promote_directory_noreplace", side_effect=delayed_promotion
            ):
                with self.assertRaisesRegex(SmokeFailure, "wall-time ceiling exceeded"):
                    run_smoke(config, FixtureBackend())
            self.assertTrue(target.is_dir())
            self.assertFalse((target / COMMIT_RECEIPT_FILENAME).exists())
            with self.assertRaisesRegex(SmokeFailure, "lifecycle set"):
                verify_result_dir(target)

    def test_delayed_receipt_preparation_fails_without_public_receipt(self):
        with tempfile.TemporaryDirectory() as parent:
            target = Path(parent) / "result"
            config = replace(fixture_config(target), max_wall_seconds=0.05)

            def delayed_preparation(final_dir, payload):
                time.sleep(0.1)
                return _prepare_commit_receipt(final_dir, payload)

            with mock.patch(
                "qwen_scope_real_smoke._prepare_commit_receipt", side_effect=delayed_preparation
            ):
                with self.assertRaisesRegex(SmokeFailure, "wall-time ceiling exceeded"):
                    run_smoke(config, FixtureBackend())
            self.assertTrue(target.is_dir())
            self.assertFalse((target / COMMIT_RECEIPT_FILENAME).exists())
            self.assertEqual(list(target.glob(".*.prepared")), [])
            with self.assertRaisesRegex(SmokeFailure, "lifecycle set"):
                verify_result_dir(target)

    def test_delayed_atomic_receipt_publish_is_external_supervisor_scope(self):
        with tempfile.TemporaryDirectory() as parent:
            target = Path(parent) / "result"
            config = replace(fixture_config(target), max_wall_seconds=0.05)

            def delayed_publish(prepared_path, final_path):
                time.sleep(0.1)
                return _publish_commit_receipt(prepared_path, final_path)

            with mock.patch(
                "qwen_scope_real_smoke._publish_commit_receipt", side_effect=delayed_publish
            ):
                run_smoke(config, FixtureBackend())
            verified = verify_result_dir(target)
            self.assertEqual(verified["status"], "PASS")
            receipt = json.loads((target / COMMIT_RECEIPT_FILENAME).read_text(encoding="utf-8"))
            checkpoint = receipt["deadline"]["pre_receipt_checkpoint_elapsed_seconds"]
            self.assertLessEqual(checkpoint, config.max_wall_seconds)

    def test_post_rename_parent_fsync_failure_leaves_uncommitted_result_without_cleanup(self):
        with tempfile.TemporaryDirectory() as parent:
            target = Path(parent) / "result"
            calls = 0

            def fail_post_rename_parent_fsync(path):
                nonlocal calls
                calls += 1
                if calls == 4:
                    raise OSError("simulated post-rename parent fsync failure")
                return _fsync_directory(path)

            with mock.patch(
                "qwen_scope_real_smoke._fsync_directory", side_effect=fail_post_rename_parent_fsync
            ), mock.patch("qwen_scope_real_smoke.shutil.rmtree") as cleanup:
                with self.assertRaisesRegex(OSError, "post-rename parent fsync failure"):
                    run_smoke(fixture_config(target), FixtureBackend())
                cleanup.assert_not_called()
            self.assertTrue(target.is_dir())
            self.assertFalse((target / COMMIT_RECEIPT_FILENAME).exists())
            with self.assertRaisesRegex(SmokeFailure, "lifecycle set"):
                verify_result_dir(target)

    def test_prepromotion_orphan_cannot_be_laundered_by_simple_rename(self):
        with tempfile.TemporaryDirectory() as parent:
            committed = Path(parent) / "committed"
            stage_orphan = Path(parent) / ".result.stage-orphan"
            laundered = Path(parent) / "laundered"
            run_smoke(fixture_config(committed), FixtureBackend())
            stage_orphan.mkdir()
            for name in ("qwen_scope_real_smoke.json", "manifest.json"):
                shutil.copy2(committed / name, stage_orphan / name)
            stage_orphan.rename(laundered)
            self.assertFalse(stage_orphan.exists())
            self.assertTrue(laundered.is_dir())
            with self.assertRaisesRegex(SmokeFailure, "lifecycle set"):
                verify_result_dir(laundered)

    def test_committed_result_renamed_to_another_target_breaks_receipt_binding(self):
        with tempfile.TemporaryDirectory() as parent:
            original = Path(parent) / "original"
            moved = Path(parent) / "moved"
            run_smoke(fixture_config(original), FixtureBackend())
            original.rename(moved)
            with self.assertRaisesRegex(SmokeFailure, "target identity"):
                verify_result_dir(moved)

    def test_copied_committed_result_has_distinct_archive_verification(self):
        with tempfile.TemporaryDirectory() as parent:
            original = Path(parent) / "original"
            archived = Path(parent) / "archive-copy"
            run_smoke(fixture_config(original), FixtureBackend())
            shutil.copytree(original, archived)
            with self.assertRaisesRegex(SmokeFailure, "target identity"):
                verify_result_dir(archived)
            verified = verify_archived_copy_dir(archived)
            self.assertEqual(verified["status"], "ARCHIVED_COPY_VERIFIED")
            self.assertFalse(verified["current_directory_lifecycle_pass"])
            self.assertEqual(verified["recorded_source_target"]["directory_name"], "original")
            self.assertEqual(len(verified["recorded_source_target"]["resolved_path_sha256"]), 64)
            self.assertEqual(len(verified["manifest_sha256"]), 64)
            self.assertGreater(verified["manifest_byte_size"], 0)
            self.assertEqual(len(verified["receipt_sha256"]), 64)
            self.assertGreater(verified["receipt_byte_size"], 0)

            result = subprocess.run(
                [
                    sys.executable,
                    "qwen_scope_real_smoke.py",
                    "--verify-archived-copy-dir",
                    str(archived),
                ],
                cwd=Path(__file__).parent,
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["status"], "ARCHIVED_COPY_VERIFIED")

    def test_archived_copy_verifier_rejects_artifact_manifest_and_receipt_tamper(self):
        mutations = ("artifact", "manifest", "receipt")
        for mutation in mutations:
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as parent:
                original = Path(parent) / "original"
                archived = Path(parent) / "archive-copy"
                run_smoke(fixture_config(original), FixtureBackend())
                shutil.copytree(original, archived)
                if mutation == "artifact":
                    artifact_path = archived / "qwen_scope_real_smoke.json"
                    artifact_path.write_bytes(artifact_path.read_bytes() + b" ")
                elif mutation == "manifest":
                    manifest_path = archived / "manifest.json"
                    manifest_path.write_bytes(manifest_path.read_bytes() + b" ")
                else:
                    receipt_path = archived / COMMIT_RECEIPT_FILENAME
                    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                    receipt["lifecycle_id"] = "0" * 32
                    rewrite_canonical(receipt_path, receipt)
                with self.assertRaises(SmokeFailure):
                    verify_archived_copy_dir(archived)

    def test_archived_copy_verifier_rejects_extra_member(self):
        with tempfile.TemporaryDirectory() as parent:
            original = Path(parent) / "original"
            archived = Path(parent) / "archive-copy"
            run_smoke(fixture_config(original), FixtureBackend())
            shutil.copytree(original, archived)
            (archived / "unregistered.txt").write_text("not in the frozen set", encoding="utf-8")
            with self.assertRaisesRegex(SmokeFailure, "lifecycle set"):
                verify_archived_copy_dir(archived)

    def test_rerun_refuses_valid_existing_pass(self):
        with tempfile.TemporaryDirectory() as parent:
            target = Path(parent) / "result"
            run_smoke(fixture_config(target), FixtureBackend())
            original = (target / "manifest.json").read_bytes()
            with self.assertRaisesRegex(SmokeFailure, "must be absent"):
                run_smoke(fixture_config(target), FixtureBackend())
            self.assertEqual((target / "manifest.json").read_bytes(), original)

    def test_verifier_cli_accepts_complete_canonical_result_without_opt_in(self):
        with tempfile.TemporaryDirectory() as parent:
            target = Path(parent) / "result"
            run_smoke(fixture_config(target), FixtureBackend())
            environment = os.environ.copy()
            environment.pop("CHELATED_QWEN_SCOPE_REAL_SMOKE", None)
            result = subprocess.run(
                [sys.executable, "qwen_scope_real_smoke.py", "--verify-result-dir", str(target)],
                cwd=Path(__file__).parent,
                env=environment,
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(json.loads(result.stdout)["status"], "PASS")

    def test_verifier_rejects_tamper_and_truncation(self):
        with tempfile.TemporaryDirectory() as parent:
            tampered = Path(parent) / "tampered"
            run_smoke(fixture_config(tampered), FixtureBackend())
            artifact_path = tampered / "qwen_scope_real_smoke.json"
            artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
            artifact["observation"]["residual_sha256"] = "b" * 64
            rewrite_canonical(artifact_path, artifact)
            with self.assertRaisesRegex(SmokeFailure, "digest"):
                verify_result_dir(tampered)

            truncated = Path(parent) / "truncated"
            run_smoke(fixture_config(truncated), FixtureBackend())
            (truncated / "manifest.json").write_bytes(b'{"schema_version":')
            with self.assertRaisesRegex(SmokeFailure, "strict UTF-8 JSON"):
                verify_result_dir(truncated)

    def test_verifier_rejects_schema_status_source_and_unsafe_path(self):
        mutations = {
            "schema": lambda artifact, manifest: artifact.update({"unexpected": True}),
            "status": lambda artifact, manifest: manifest.update({"status": "FAIL"}),
            "source": lambda artifact, manifest: manifest["source_contract"].update(
                {"model_revision": "0" * 40}
            ),
            "path": lambda artifact, manifest: manifest["artifacts"][0].update(
                {"path": "../qwen_scope_real_smoke.json"}
            ),
        }
        for label, mutate in mutations.items():
            with self.subTest(label=label), tempfile.TemporaryDirectory() as parent:
                target = Path(parent) / "result"
                run_smoke(fixture_config(target), FixtureBackend())
                artifact_path = target / "qwen_scope_real_smoke.json"
                manifest_path = target / "manifest.json"
                artifact = json.loads(artifact_path.read_text(encoding="utf-8"))
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                mutate(artifact, manifest)
                rewrite_canonical(artifact_path, artifact)
                rewrite_canonical(manifest_path, manifest)
                with self.assertRaises(SmokeFailure):
                    verify_result_dir(target)

    def test_verifier_rejects_oversize_artifact_before_parse(self):
        with tempfile.TemporaryDirectory() as parent:
            target = Path(parent) / "result"
            run_smoke(fixture_config(target), FixtureBackend())
            (target / "qwen_scope_real_smoke.json").write_bytes(b" " * (1024 * 1024 + 1))
            with self.assertRaisesRegex(SmokeFailure, "byte size"):
                verify_result_dir(target)

    def test_verifier_rejects_noncanonical_json_bytes(self):
        with tempfile.TemporaryDirectory() as parent:
            target = Path(parent) / "result"
            run_smoke(fixture_config(target), FixtureBackend())
            artifact_path = target / "qwen_scope_real_smoke.json"
            artifact_path.write_bytes(artifact_path.read_bytes() + b" ")
            with self.assertRaisesRegex(SmokeFailure, "canonical JSON"):
                verify_result_dir(target)

    def test_hard_stop_orphan_stage_is_not_public_pass_evidence(self):
        with tempfile.TemporaryDirectory() as parent:
            promoted = Path(parent) / "result"
            orphan_stage = Path(parent) / ".result.stage-hard-stop"
            run_smoke(fixture_config(promoted), FixtureBackend())
            promoted.rename(orphan_stage)
            self.assertFalse(promoted.exists())
            self.assertTrue((orphan_stage / "manifest.json").is_file())
            with self.assertRaisesRegex(SmokeFailure, "not promoted PASS evidence"):
                verify_result_dir(orphan_stage)
            result = subprocess.run(
                [
                    sys.executable,
                    "qwen_scope_real_smoke.py",
                    "--verify-result-dir",
                    str(orphan_stage),
                ],
                cwd=Path(__file__).parent,
                text=True,
                capture_output=True,
                timeout=10,
                check=False,
            )
            self.assertEqual(result.returncode, 1)
            self.assertIn("not promoted PASS evidence", result.stderr)


if __name__ == "__main__":
    unittest.main()
