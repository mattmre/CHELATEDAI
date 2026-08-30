from __future__ import annotations

import json
import unittest

from egv.recommissioning.client import (
    BoundedModelClient,
    GenerationBudget,
    ModelServiceProfile,
    RecommissioningClientError,
    TypedCandidateResponse,
    UrllibLoopbackTransport,
    candidate_response_schema,
)


MODEL_ID = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"


def _profile() -> ModelServiceProfile:
    return ModelServiceProfile(
        model_id=MODEL_ID,
        model_revision="cc84af2fe71647d87f4486c064f320e1e7535243",
        engine_image_digest="sha256:" + "a" * 64,
        max_model_len=8192,
        precision="NVFP4",
        kv_cache_dtype="fp8",
    )


class FakeTransport:
    def __init__(self, *, served_model: str = MODEL_ID, cited_evidence=None) -> None:
        self.served_model = served_model
        self.cited_evidence = cited_evidence if cited_evidence is not None else ["evidence-1"]
        self.calls = []

    def __call__(self, method, path, body, timeout):
        self.calls.append((method, path, body, timeout))
        if path == "/health":
            return 200, b""
        if path == "/v1/models":
            return 200, json.dumps({"data": [{"id": self.served_model}]}).encode("utf-8")
        if path == "/v1/chat/completions":
            request = json.loads(body.decode("utf-8"))
            self.last_request = request
            content = {
                "action": "COMMIT",
                "source": "def main(value):\n    return value + 1\n",
                "summary": "Apply the public increment rule.",
                "evidence_ids": self.cited_evidence,
            }
            response = {
                "model": self.served_model,
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": json.dumps(content)},
                    }
                ],
                "usage": {"prompt_tokens": 37, "completion_tokens": 24},
            }
            return 200, json.dumps(response).encode("utf-8")
        raise AssertionError(path)


class ModelServiceProfileTests(unittest.TestCase):
    def test_public_manifest_excludes_endpoint_and_has_stable_digest(self):
        profile = _profile()
        manifest = profile.public_manifest()
        self.assertNotIn("endpoint", manifest)
        self.assertEqual(profile.digest, profile.digest)
        self.assertEqual(manifest["model_revision"], profile.model_revision)

    def test_loopback_transport_rejects_public_or_credentialed_origins(self):
        with self.assertRaises(RecommissioningClientError):
            UrllibLoopbackTransport("http://192.0.2.10:18000")
        with self.assertRaises(RecommissioningClientError):
            UrllibLoopbackTransport("http://user@127.0.0.1:18000")
        with self.assertRaises(RecommissioningClientError):
            UrllibLoopbackTransport("http://localhost:18000")
        UrllibLoopbackTransport("http://127.0.0.1:18000")


class BoundedModelClientTests(unittest.TestCase):
    def test_attestation_and_one_typed_generation(self):
        transport = FakeTransport()
        client = BoundedModelClient(_profile(), transport)
        attestation = client.attest()
        self.assertEqual(attestation["model_id"], MODEL_ID)
        result = client.generate(
            (
                {"role": "system", "content": "Return a bounded repair."},
                {"role": "user", "content": "Implement the public increment rule."},
            ),
            visible_evidence_ids=("evidence-1",),
            budget=GenerationBudget(max_completion_tokens=128, timeout_seconds=5.0, seed=7),
        )
        self.assertEqual(result.candidate.action, "COMMIT")
        self.assertEqual(result.receipt.prompt_tokens, 37)
        self.assertEqual(result.receipt.completion_tokens, 24)
        self.assertNotIn("endpoint", result.receipt.public_record())
        self.assertEqual(transport.last_request["temperature"], 0.0)
        self.assertEqual(transport.last_request["seed"], 7)
        self.assertEqual(transport.last_request["response_format"]["type"], "json_schema")

    def test_attestation_rejects_substituted_model(self):
        client = BoundedModelClient(_profile(), FakeTransport(served_model="substituted/model"))
        with self.assertRaisesRegex(RecommissioningClientError, "served model identity"):
            client.attest()

    def test_generation_rejects_unseen_evidence_citation(self):
        client = BoundedModelClient(_profile(), FakeTransport(cited_evidence=["hidden-evidence"]))
        with self.assertRaisesRegex(RecommissioningClientError, "not model-visible"):
            client.generate(
                ({"role": "user", "content": "Repair the public task."},),
                visible_evidence_ids=("evidence-1",),
                budget=GenerationBudget(max_completion_tokens=128),
            )

    def test_candidate_response_contract_is_closed(self):
        with self.assertRaisesRegex(RecommissioningClientError, "closed field set"):
            TypedCandidateResponse.parse(
                {
                    "action": "COMMIT",
                    "source": "def main(value): return value",
                    "summary": "bounded",
                    "evidence_ids": [],
                    "unexpected": True,
                },
                visible_evidence_ids=(),
            )
        schema = candidate_response_schema()["schema"]
        self.assertFalse(schema["additionalProperties"])
        self.assertEqual(set(schema["required"]), {"action", "source", "summary", "evidence_ids"})

    def test_candidate_source_must_be_a_complete_python_module(self):
        with self.assertRaisesRegex(RecommissioningClientError, "complete Python module"):
            TypedCandidateResponse.parse(
                {
                    "action": "COMMIT",
                    "source": "def main(value):\n    return (",
                    "summary": "Malformed on purpose.",
                    "evidence_ids": [],
                },
                visible_evidence_ids=(),
            )


if __name__ == "__main__":
    unittest.main()
