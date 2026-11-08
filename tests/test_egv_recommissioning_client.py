from __future__ import annotations

from dataclasses import replace
import hashlib
import hmac
import json
import os
import tempfile
import time
import unittest
from urllib import request as urllib_request

from egv.canonical import canonical_json, digest_for
from egv.recommissioning.client import (
    AttestedLoopbackService,
    FileCoordinateTokenLedger,
    InMemoryCoordinateTokenLedger,
    ModelServiceProfile,
    PublicGenerationReceipt,
    RecommissioningClientError,
    TokenizationPreflight,
    TrustedLaunchReceipt,
    TypedCandidateResponse,
    UrllibLoopbackTransport,
    candidate_response_schema,
)
from egv.recommissioning.protocol import PilotCoordinate, TokenBudget, VisibleTaskPackage, render_prompt


MODEL_ID = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"


def _profile(**updates):
    values = {
        "model_bundle": "LIGHTNING",
        "model_id": MODEL_ID,
        "model_revision": "cc84af2fe71647d87f4486c064f320e1e7535243",
        "engine_image_digest": "sha256:" + "a" * 64,
        "max_model_len": 8192,
        "precision": "NVFP4",
        "kv_cache_dtype": "fp8",
        "tokenizer_digest": digest_for("pinned-tokenizer"),
        "chat_template_digest": digest_for("pinned-chat-template"),
    }
    values.update(updates)
    return ModelServiceProfile(**values)


def _budget(**updates):
    values = {
        "max_prompt_tokens": 128,
        "max_completion_tokens": 64,
        "max_total_tokens": 192,
        "max_coordinate_tokens": 1200,
        "hard_timeout_ms": 5000,
        "seed": 7,
    }
    values.update(updates)
    return TokenBudget(**values)


def _coordinate(**updates):
    values = {
        "campaign_id": "egv-nemotron-stage-a-v2",
        "model_bundle": "LIGHTNING",
        "protocol_id": "P1",
        "task_id": "egv-learnable-pure-function-train-001-v2",
        "task_family": "PURE_FUNCTION",
        "seed": 7,
    }
    values.update(updates)
    return PilotCoordinate(**values)


def _task(coordinate=None):
    coordinate = coordinate or _coordinate()
    return VisibleTaskPackage.for_test(
        task_id=coordinate.task_id,
        task_family=coordinate.task_family,
        public_spec={
            "schema_version": "egv-learnable-public-spec-v2",
            "family_id": coordinate.task_family,
            "task_scope": coordinate.task_id,
            "operation": "double-plus-bias",
            "parameters": {"bias": 5},
        },
        initial_source=(
            "def main(value):\n"
            "    return {'case_nonce': value.get('case_nonce'), 'task_scope': value.get('task_scope')}\n"
        ),
    )


def _envelope(profile=None, budget=None, coordinate=None):
    profile = profile or _profile()
    budget = budget or _budget()
    coordinate = coordinate or _coordinate()
    return render_prompt(
        coordinate,
        _task(coordinate),
        model_profile_digest=profile.digest,
        budget=budget,
        attempt_index=1,
    )


def _tokenization(envelope, profile=None, prompt_tokens=37):
    profile = profile or _profile()
    rendered = "<chat>" + canonical_json([message.record() for message in envelope.messages])
    return TokenizationPreflight.from_pinned_renderer(
        envelope,
        profile,
        rendered_prompt=rendered,
        token_ids=tuple(range(prompt_tokens)),
    )


class FakeTransport:
    def __init__(
        self,
        *,
        served_model=MODEL_ID,
        cited_evidence=None,
        prompt_tokens=37,
        tool_calls=False,
        source=None,
        sleep_seconds=0.0,
        bool_usage=False,
        duplicate_candidate_key=False,
    ):
        self.served_model = served_model
        self.cited_evidence = cited_evidence if cited_evidence is not None else []
        self.prompt_tokens = prompt_tokens
        self.tool_calls = tool_calls
        self.source = source or (
            "def main(value):\n"
            "    payload = value['payload']\n"
            "    return {'case_nonce': value['case_nonce'], 'task_scope': value['task_scope'], "
            "'value': payload + payload + 5}\n"
        )
        self.sleep_seconds = sleep_seconds
        self.bool_usage = bool_usage
        self.duplicate_candidate_key = duplicate_candidate_key
        self.calls = []

    def __call__(self, method, path, body, timeout):
        self.calls.append((method, path, body, timeout))
        if path == "/health":
            return 200, b""
        if path == "/v1/models":
            return 200, canonical_json({"data": [{"id": self.served_model}]}).encode("utf-8")
        if path == "/v1/chat/completions":
            if self.sleep_seconds:
                time.sleep(self.sleep_seconds)
            self.last_request = json.loads(body.decode("utf-8"))
            if self.duplicate_candidate_key:
                content = '{"action":"COMMIT","action":"EDIT","source":"def main(value):\\n return value\\n","summary":"x","evidence_ids":[]}'
            else:
                content = canonical_json(
                    {
                        "action": "COMMIT",
                        "source": self.source,
                        "summary": "Apply the complete public rule. ✓",
                        "evidence_ids": self.cited_evidence,
                    }
                )
            message = {"role": "assistant", "content": content}
            if self.tool_calls:
                message["tool_calls"] = []
            response = {
                "model": self.served_model,
                "choices": [{"finish_reason": "stop", "message": message}],
                "usage": {
                    "prompt_tokens": True if self.bool_usage else self.prompt_tokens,
                    "completion_tokens": 24,
                },
            }
            return 200, canonical_json(response).encode("utf-8")
        raise AssertionError(path)


class TestSigner:
    key_id = "test-signing-key"

    def sign_bytes(self, payload):
        return hmac.new(b"test-public-receipt-key", payload, hashlib.sha256).hexdigest()


def _service(transport=None, profile=None):
    profile = profile or _profile()
    return AttestedLoopbackService.for_test(
        profile,
        transport or FakeTransport(),
        TrustedLaunchReceipt.for_test(profile),
    )


def _request(service, ledger, *, budget=None, envelope=None, tokenization=None):
    budget = budget or _budget()
    envelope = envelope or _envelope(service.profile, budget)
    tokenization = tokenization or _tokenization(envelope, service.profile)
    return service.build_request(envelope, budget, tokenization, ledger)


class ModelServiceProfileTests(unittest.TestCase):
    def test_public_manifest_excludes_endpoint_and_binds_tokenizer(self):
        profile = _profile()
        manifest = profile.public_manifest()
        self.assertNotIn("endpoint", manifest)
        self.assertEqual(manifest["chat_template_digest"], profile.chat_template_digest)
        self.assertEqual(profile.digest, profile.digest)

    def test_loopback_transport_rejects_public_credentials_and_disables_proxies(self):
        for endpoint in (
            "http://192.0.2.10:18000",
            "http://user@127.0.0.1:18000",
            "http://localhost:18000",
            "https://127.0.0.1:18000",
        ):
            with self.subTest(endpoint=endpoint), self.assertRaises(RecommissioningClientError):
                UrllibLoopbackTransport(endpoint)
        old_proxy = os.environ.get("HTTP_PROXY")
        os.environ["HTTP_PROXY"] = "http://192.0.2.9:3128"
        try:
            transport = UrllibLoopbackTransport("http://127.0.0.1:18000")
            proxy_handlers = [handler for handler in transport._opener.handlers if isinstance(handler, urllib_request.ProxyHandler)]
            self.assertTrue(all(handler.proxies == {} for handler in proxy_handlers))
        finally:
            if old_proxy is None:
                os.environ.pop("HTTP_PROXY", None)
            else:
                os.environ["HTTP_PROXY"] = old_proxy

    def test_production_launch_authority_is_externally_supplied_and_private(self):
        profile = _profile()
        record = {
            "schema_version": "egv-trusted-launch-receipt-v1",
            "model_profile_digest": profile.digest,
            "runtime_config_digest": digest_for("closed-runtime"),
            "launch_receipt_digest": digest_for("controller-issued-receipt"),
            "service_nonce": "controller-service-nonce",
        }
        trusted = TrustedLaunchReceipt.from_controller_record(record, issuer_token=b"a" * 32)
        self.assertFalse(trusted.test_only)
        self.assertNotIn("issuer_token", trusted.public_binding_record())
        with self.assertRaises(RecommissioningClientError):
            AttestedLoopbackService(None, profile, FakeTransport(), trusted, None, test_only=False)


class RequestAndBudgetTests(unittest.TestCase):
    def test_generation_consumes_receipt_bound_request_and_returns_separated_records(self):
        transport = FakeTransport()
        service = _service(transport)
        ledger = InMemoryCoordinateTokenLedger.for_test()
        request = _request(service, ledger)
        result = service.generate(request, ledger=ledger, signer=TestSigner())
        public = result.public_receipt.public_record()
        private = result.private_record.private_record()
        self.assertEqual(public["coordinate_id"], request.envelope.coordinate_id)
        self.assertEqual(public["task_package_digest"], request.envelope.task_package_digest)
        self.assertEqual(public["prompt_digest"], request.envelope.prompt_digest)
        self.assertEqual(public["service_attestation_digest"], service.attestation.attestation_digest)
        self.assertEqual(public["prompt_tokens"], 37)
        self.assertNotIn("source", public)
        self.assertNotIn("request_json", public)
        self.assertEqual(json.loads(private["candidate_json"])["source"], result.candidate.source)
        self.assertNotIn(service.attestation.issuer_token.hex(), canonical_json(private))
        self.assertEqual(PublicGenerationReceipt.validate_record(public)["receipt_digest"], public["receipt_digest"])
        self.assertEqual(transport.last_request["response_format"]["type"], "json_schema")

    def test_prompt_tampering_is_rejected_even_with_original_digest(self):
        profile = _profile()
        budget = _budget()
        envelope = _envelope(profile, budget)
        forged = object.__new__(type(envelope))
        for field_name in envelope.__dataclass_fields__:
            object.__setattr__(forged, field_name, getattr(envelope, field_name))
        object.__setattr__(
            forged,
            "messages",
            (envelope.messages[0], replace(envelope.messages[1], content=envelope.messages[1].content + " tamper")),
        )
        service = _service(profile=profile)
        ledger = InMemoryCoordinateTokenLedger.for_test()
        tokenization = _tokenization(envelope, profile)
        with self.assertRaisesRegex(Exception, "prompt digest"):
            service.build_request(forged, budget, tokenization, ledger)

    def test_request_rejects_profile_budget_seed_and_tokenizer_substitution(self):
        service = _service()
        ledger = InMemoryCoordinateTokenLedger.for_test()
        budget = _budget()
        envelope = _envelope(service.profile, budget)
        tokenization = _tokenization(envelope, service.profile)
        wrong_profile = _profile(tokenizer_digest=digest_for("substituted-tokenizer"))
        with self.assertRaisesRegex(RecommissioningClientError, "pinned model profile"):
            tokenization.validate(envelope=envelope, profile=wrong_profile)
        with self.assertRaisesRegex(Exception, "seed"):
            _envelope(service.profile, replace(budget, seed=8), _coordinate(seed=7))
        request = service.build_request(envelope, budget, tokenization, ledger)
        forged_budget = replace(budget, max_completion_tokens=63, max_total_tokens=191)
        object.__setattr__(request, "budget", forged_budget)
        with self.assertRaisesRegex(Exception, "budget"):
            request.validate(profile=service.profile, attestation=service.attestation)

        super_coordinate = _coordinate(model_bundle="SUPER")
        super_envelope = _envelope(service.profile, budget, super_coordinate)
        super_tokenization = _tokenization(super_envelope, service.profile)
        with self.assertRaisesRegex(RecommissioningClientError, "bundle differs"):
            service.build_request(super_envelope, budget, super_tokenization, ledger)

    def test_exact_prompt_count_endpoint_match_and_bool_rejection(self):
        for transport, pattern in (
            (FakeTransport(prompt_tokens=38), "differs from pinned preflight"),
            (FakeTransport(bool_usage=True), "differs from pinned preflight"),
        ):
            with self.subTest(pattern=pattern):
                service = _service(transport)
                ledger = InMemoryCoordinateTokenLedger.for_test()
                with self.assertRaisesRegex(RecommissioningClientError, pattern):
                    service.generate(_request(service, ledger), ledger=ledger, signer=TestSigner())

    def test_hard_elapsed_budget_is_enforced_after_transport_returns(self):
        service = _service(FakeTransport(sleep_seconds=0.02))
        ledger = InMemoryCoordinateTokenLedger.for_test()
        budget = _budget(hard_timeout_ms=1)
        with self.assertRaisesRegex(RecommissioningClientError, "hard elapsed"):
            service.generate(_request(service, ledger, budget=budget), ledger=ledger, signer=TestSigner())

    def test_coordinate_budget_is_cumulatively_debited_and_durable(self):
        service = _service()
        budget = _budget(max_coordinate_tokens=201)
        with tempfile.TemporaryDirectory() as directory:
            ledger = FileCoordinateTokenLedger(directory)
            first = _request(service, ledger, budget=budget)
            service.generate(first, ledger=ledger, signer=TestSigner())
            reopened = FileCoordinateTokenLedger(directory)
            state = reopened.snapshot(first.envelope.coordinate_id, budget)
            self.assertEqual(state.cumulative_reserved_tokens, 101)
            with self.assertRaisesRegex(RecommissioningClientError, "coordinate token ceiling"):
                service.build_request(first.envelope, budget, first.tokenization, reopened)

    def test_non_request_input_and_stale_coordinate_state_are_rejected(self):
        service = _service()
        ledger = InMemoryCoordinateTokenLedger.for_test()
        request = _request(service, ledger)
        service.generate(request, ledger=ledger, signer=TestSigner())
        with self.assertRaisesRegex(RecommissioningClientError, "changed after request construction"):
            ledger.debit(request)
        with self.assertRaisesRegex(RecommissioningClientError, "typed GenerationRequest"):
            service.generate(request.envelope, ledger=ledger, signer=TestSigner())

    def test_production_boundary_rejects_test_only_task_even_for_prebuilt_request(self):
        service = _service()
        test_ledger = InMemoryCoordinateTokenLedger.for_test()
        request = _request(service, test_ledger)
        service._test_only = False
        with tempfile.TemporaryDirectory() as directory:
            durable = FileCoordinateTokenLedger(directory)
            with self.assertRaisesRegex(RecommissioningClientError, "exact learnable-v2 task adapter"):
                service.generate(request, ledger=durable, signer=TestSigner())


class ResponseBoundaryTests(unittest.TestCase):
    def test_tool_calls_are_rejected_even_when_empty(self):
        service = _service(FakeTransport(tool_calls=True))
        ledger = InMemoryCoordinateTokenLedger.for_test()
        with self.assertRaisesRegex(RecommissioningClientError, "forbidden tool call"):
            service.generate(_request(service, ledger), ledger=ledger, signer=TestSigner())

    def test_candidate_requires_exact_public_locus_and_pure_contract(self):
        bad_sources = (
            "def helper(value):\n    return value\n",
            "def main(other, value):\n    return value\n",
            "def main(value):\n    return open('secret').read()\n",
        )
        for source in bad_sources:
            with self.subTest(source=source):
                service = _service(FakeTransport(source=source))
                ledger = InMemoryCoordinateTokenLedger.for_test()
                with self.assertRaises(RecommissioningClientError):
                    service.generate(_request(service, ledger), ledger=ledger, signer=TestSigner())

    def test_strict_json_rejects_duplicate_candidate_keys(self):
        service = _service(FakeTransport(duplicate_candidate_key=True))
        ledger = InMemoryCoordinateTokenLedger.for_test()
        with self.assertRaisesRegex(RecommissioningClientError, "duplicate JSON keys"):
            service.generate(_request(service, ledger), ledger=ledger, signer=TestSigner())

    def test_candidate_schema_and_runtime_share_character_bounds(self):
        schema = candidate_response_schema()["schema"]
        self.assertEqual(schema["properties"]["source"]["maxLength"], 64 * 1024)
        self.assertEqual(schema["properties"]["summary"]["maxLength"], 1024)
        candidate = TypedCandidateResponse.parse(
            {
                "action": "COMMIT",
                "source": "def main(value):\n    return '✓'\n",
                "summary": "Unicode is counted in characters: ✓",
                "evidence_ids": [],
            },
            visible_evidence_ids=(),
            public_locus="module:main",
        )
        self.assertIn("✓", candidate.summary)

    def test_attestation_rejects_substituted_model(self):
        with self.assertRaisesRegex(RecommissioningClientError, "served model identity"):
            _service(FakeTransport(served_model="substituted/model"))

    def test_public_receipt_self_digest_rejects_tampering(self):
        service = _service()
        ledger = InMemoryCoordinateTokenLedger.for_test()
        result = service.generate(_request(service, ledger), ledger=ledger, signer=TestSigner())
        tampered = result.public_receipt.public_record()
        tampered["task_id"] = "different-task"
        with self.assertRaisesRegex(RecommissioningClientError, "digest does not bind"):
            PublicGenerationReceipt.validate_record(tampered)


if __name__ == "__main__":
    unittest.main()
