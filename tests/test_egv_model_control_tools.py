from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.build_hf_snapshot_identity import build_identity
from scripts.run_egv_model_control_matrix import CASES, _run_case


class SnapshotIdentityTests(unittest.TestCase):
    def test_identity_is_deterministic_bound_and_path_free(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            snapshot = Path(temporary) / "private-snapshot-location"
            snapshot.mkdir()
            (snapshot / "model-00001-of-00001.safetensors").write_bytes(b"weight-bytes")
            (snapshot / "tokenizer.json").write_text('{"tokenizer":true}', encoding="utf-8")
            (snapshot / "chat_template.jinja").write_text("{{ message }}", encoding="utf-8")

            first = build_identity(snapshot, "owner/model", "a" * 40)
            second = build_identity(snapshot, "owner/model", "a" * 40)

            self.assertEqual(first, second)
            self.assertEqual(first["weight_shard_count"], 1)
            self.assertEqual(first["repository"], "owner/model")
            self.assertNotIn(str(snapshot), json.dumps(first, sort_keys=True))

            (snapshot / "tokenizer.json").write_text('{"tokenizer":false}', encoding="utf-8")
            mutated = build_identity(snapshot, "owner/model", "a" * 40)
            self.assertNotEqual(first["content_manifest_digest"], mutated["content_manifest_digest"])
            self.assertNotEqual(first["receipt_digest"], mutated["receipt_digest"])


class ModelControlMatrixTests(unittest.TestCase):
    def test_case_binds_requested_budget_and_exact_json(self) -> None:
        fixture = CASES[0]
        response = {
            "choices": [{"message": {"content": '{"answer":42}', "reasoning_content": "bounded"}}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 11},
        }

        def fake_http(url, *, body=None, timeout):
            self.assertTrue(url.endswith("/v1/chat/completions"))
            self.assertEqual(body["max_tokens"], 4096)
            self.assertEqual(body["seed"], 1729)
            self.assertEqual(timeout, 7.0)
            return 200, response

        with patch("scripts.run_egv_model_control_matrix._http_json", side_effect=fake_http):
            result = _run_case("http://operator-private", "owner/model", fixture, 7.0, 4096)

        self.assertTrue(result["exact_pass"])
        self.assertIsNone(result["response_error"])
        self.assertTrue(result["reasoning_returned_separately"])

    def test_missing_final_content_is_a_recorded_failure(self) -> None:
        fixture = CASES[0]
        response = {
            "choices": [{"message": {"content": None, "reasoning_content": "budget exhausted"}}],
            "usage": {"prompt_tokens": 9, "completion_tokens": 1024},
        }
        with patch(
            "scripts.run_egv_model_control_matrix._http_json",
            return_value=(200, response),
        ):
            result = _run_case("http://operator-private", "owner/model", fixture, 7.0, 1024)

        self.assertFalse(result["exact_pass"])
        self.assertEqual(result["response_error"], "missing-text-content")
        self.assertIsNone(result["observed"])


if __name__ == "__main__":
    unittest.main()
