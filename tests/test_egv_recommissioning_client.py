from __future__ import annotations

from dataclasses import replace
import json
import os
import tempfile
import time
import unittest
from urllib import request as urllib_request

from egv.canonical import canonical_json, digest_bytes, digest_for
from egv.receipts import ReceiptSigner
from egv.recommissioning.client import (
    AttestedLoopbackService,
    FileCoordinateTokenLedger,
    InMemoryCoordinateTokenLedger,
    ModelServiceProfile,
    PinnedTokenizerRenderer,
    PublicGenerationReceipt,
    RecommissioningClientError,
    ServiceAttestation,
    TokenizationPreflight,
    TrustedLaunchReceipt,
    TypedCandidateResponse,
    UrllibLoopbackTransport,
    candidate_response_schema,
)
from egv.recommissioning.protocol import PilotCoordinate, TokenBudget, VisibleTaskPackage, render_prompt


MODEL_ID = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
CHAT_TEMPLATE = "<egv-chat-template-v1>"


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
        "tokenizer_implementation_digest": PinnedTokenizerRenderer.digest_tokenizer_implementation(FakeTokenizer()),
        "chat_template_digest": digest_bytes(CHAT_TEMPLATE.encode("utf-8")),
        "weights_digest": digest_for("pinned-weights"),
        "license_digest": digest_for("pinned-license"),
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
        "generation_receipt_key_id": TestSigner().key_id,
        "evaluator_key_id": ReceiptSigner(bytes.fromhex("22" * 32)).key_id,
        "evaluator_profile_digest": digest_for("frozen-evaluator-profile"),
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


class FakeTokenizer:
    chat_template = CHAT_TEMPLATE

    def __init__(self, prompt_tokens=37):
        self.prompt_tokens = prompt_tokens
        self.calls = []

    def apply_chat_template(self, conversation, *, tokenize, add_generation_prompt):
        self.calls.append((conversation, tokenize, add_generation_prompt))
        rendered = "<chat>" + canonical_json(conversation) + "<assistant>"
        if tokenize:
            return list(range(self.prompt_tokens))
        return rendered


def _tokenization(envelope, profile=None, prompt_tokens=37):
    profile = profile or _profile()
    renderer = PinnedTokenizerRenderer.for_test(profile, FakeTokenizer(prompt_tokens))
    return TokenizationPreflight.from_pinned_renderer(
        envelope,
        profile,
        renderer,
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
        attestation_sleep_seconds=0.0,
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
        self.attestation_sleep_seconds = attestation_sleep_seconds
        self.calls = []
        self.trusted = None
        self.profile = None

    def configure_trust(self, trusted, profile):
        self.trusted = trusted
        self.profile = profile

    def __call__(self, method, path, body, timeout):
        self.calls.append((method, path, body, timeout))
        if path == "/health":
            return 200, b""
        if path == "/v1/models":
            return 200, canonical_json({"data": [{"id": self.served_model}]}).encode("utf-8")
        if path == "/v1/egv/attestation":
            if self.attestation_sleep_seconds:
                time.sleep(self.attestation_sleep_seconds)
            if self.trusted is None or self.profile is None:
                raise AssertionError("test transport trust was not configured")
            return 200, canonical_json(self.trusted.build_test_service_observation(self.profile)).encode("utf-8")
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
    def __init__(self):
        self._signer = ReceiptSigner(bytes.fromhex("11" * 32))

    @property
    def key_id(self):
        return self._signer.key_id

    @property
    def public_key(self):
        return self._signer.public_key

    def sign_bytes(self, payload):
        return self._signer.sign_bytes(payload)


def _service(transport=None, profile=None):
    profile = profile or _profile()
    signer = TestSigner()
    trusted = TrustedLaunchReceipt.for_test(profile, generation_signer=signer)
    transport = transport or FakeTransport()
    transport.configure_trust(trusted, profile)
    return AttestedLoopbackService.for_test(
        profile,
        transport,
        trusted,
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
        self.assertEqual(
            manifest["tokenizer_implementation_digest"],
            profile.tokenizer_implementation_digest,
        )
        self.assertEqual(profile.digest, profile.digest)

    def test_public_model_profile_is_closed_and_scanned_before_projection(self):
        manifest = _profile().public_manifest()
        self.assertEqual(ModelServiceProfile.from_public_manifest(manifest), _profile())
        with self.assertRaisesRegex(RecommissioningClientError, "closed public record"):
            ModelServiceProfile.from_public_manifest({**manifest, "endpoint": "opaque"})
        with self.assertRaisesRegex(RecommissioningClientError, "prohibited public data"):
            _profile(model_id="https://credential.invalid/private").public_manifest()

        service = _service()
        forged_record = service.attestation.public_record()
        observation = forged_record["service_observation"]
        observation["model_profile"]["model_id"] = "https://credential.invalid/private"
        observation["model_profile_digest"] = digest_for(observation["model_profile"])
        unsigned = {key: value for key, value in observation.items() if key not in {"observation_digest", "signature"}}
        signable = {**unsigned, "observation_digest": digest_for(unsigned)}
        forged_record["service_observation"] = {
            **signable,
            "signature": service._trusted._test_service_signer.sign_bytes(canonical_json(signable).encode("utf-8")),
        }
        outer_core = {key: value for key, value in forged_record.items() if key != "attestation_digest"}
        forged_record["attestation_digest"] = digest_for(outer_core)
        forged_attestation = object.__new__(ServiceAttestation)
        object.__setattr__(forged_attestation, "record_json", canonical_json(forged_record))
        object.__setattr__(
            forged_attestation,
            "controller_public_key_raw",
            service.attestation.controller_public_key_raw,
        )
        object.__setattr__(
            forged_attestation,
            "service_public_key_raw",
            service.attestation.service_public_key_raw,
        )
        service.attestation = forged_attestation
        with self.assertRaisesRegex(RecommissioningClientError, "prohibited public data"):
            service.attest()

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
            proxy_handlers = [
                handler for handler in transport._opener.handlers if isinstance(handler, urllib_request.ProxyHandler)
            ]
            self.assertTrue(all(handler.proxies == {} for handler in proxy_handlers))
        finally:
            if old_proxy is None:
                os.environ.pop("HTTP_PROXY", None)
            else:
                os.environ["HTTP_PROXY"] = old_proxy

    def test_production_launch_authority_is_externally_supplied_and_private(self):
        profile = _profile()
        issued = TrustedLaunchReceipt.for_test(profile, generation_signer=TestSigner())
        trusted = TrustedLaunchReceipt.from_controller_record(
            issued.public_record(),
            controller_public_key=issued.controller_public_key_raw,
        )
        self.assertFalse(trusted.test_only)
        self.assertNotIn("controller_public_key_raw", trusted.public_binding_record())
        forged = issued.public_record()
        forged["runtime_config_digest"] = digest_for("attacker-runtime")
        with self.assertRaisesRegex(RecommissioningClientError, "digest does not bind"):
            TrustedLaunchReceipt.from_controller_record(
                forged,
                controller_public_key=issued.controller_public_key_raw,
            )
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
        self.assertEqual(
            public["task_ref"], "task_" + digest_for({"task_package_digest": request.envelope.task_package_digest})
        )
        self.assertNotIn(request.envelope.task_id, canonical_json(public))
        self.assertEqual(public["task_package_digest"], request.envelope.task_package_digest)
        self.assertEqual(public["prompt_digest"], request.envelope.prompt_digest)
        self.assertEqual(public["service_attestation_digest"], service.attestation.attestation_digest)
        self.assertEqual(public["prompt_tokens"], 37)
        self.assertIn("total_elapsed_ms", public)
        self.assertNotIn("elapsed_ms", public)
        self.assertEqual(public["coordinate_debit_count"], request.envelope.attempt_index)
        self.assertEqual(public["controller_key_id"], service._trusted.controller_key_id)
        self.assertEqual(public["service_key_id"], service._trusted.service_key_id)
        self.assertEqual(public["generation_receipt_key_id"], TestSigner().key_id)
        self.assertNotIn("source", public)
        self.assertNotIn("request_json", public)
        self.assertEqual(json.loads(private["candidate_json"])["source"], result.candidate.source)
        self.assertEqual(
            PublicGenerationReceipt.validate_record(
                public,
                public_key=service._trusted.generation_receipt_public_key_raw,
            )["receipt_digest"],
            public["receipt_digest"],
        )
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
        with self.assertRaisesRegex(Exception, "canonical deterministic render"):
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

    def test_attempt_sequence_and_request_replay_are_fail_closed(self):
        service = _service()
        ledger = InMemoryCoordinateTokenLedger.for_test()
        first = _request(service, ledger)
        ledger.debit(first)
        replay_bound_to_advanced_state = _request(
            service,
            ledger,
            budget=first.budget,
            envelope=first.envelope,
            tokenization=first.tokenization,
        )
        with self.assertRaisesRegex(RecommissioningClientError, "next undebited"):
            ledger.debit(replay_bound_to_advanced_state)
        with self.assertRaisesRegex(RecommissioningClientError, "changed after request construction"):
            ledger.debit(first)

    def test_generation_deadline_covers_coordinate_ledger_debit(self):
        service = _service()
        delegate = InMemoryCoordinateTokenLedger.for_test()

        class SlowLedger:
            durable = False

            def snapshot(self, coordinate_id, budget):
                return delegate.snapshot(coordinate_id, budget)

            def debit(self, request, *, deadline_ns=None):
                time.sleep(0.02)
                return delegate.debit(request, deadline_ns=deadline_ns)

        ledger = SlowLedger()
        budget = _budget(hard_timeout_ms=1)
        with self.assertRaisesRegex(RecommissioningClientError, "hard elapsed"):
            service.generate(_request(service, ledger, budget=budget), ledger=ledger, signer=TestSigner())

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

    def test_attestation_rejects_same_model_id_with_wrong_revision_launch(self):
        expected = _profile()
        wrong_revision = _profile(model_revision="d" * 40)
        trusted = TrustedLaunchReceipt.for_test(wrong_revision, generation_signer=TestSigner())
        transport = FakeTransport(served_model=expected.model_id)
        transport.configure_trust(trusted, wrong_revision)
        with self.assertRaisesRegex(RecommissioningClientError, "trusted launch receipt differs"):
            AttestedLoopbackService.for_test(expected, transport, trusted)

    def test_service_attestation_contains_signed_full_immutable_profile(self):
        service = _service()
        observation = service.attestation.service_observation
        self.assertEqual(observation["model_profile"], service.profile.public_manifest())
        self.assertEqual(observation["model_profile"]["model_revision"], service.profile.model_revision)
        self.assertEqual(observation["model_profile"]["weights_digest"], service.profile.weights_digest)
        self.assertEqual(observation["model_profile"]["license_digest"], service.profile.license_digest)
        tampered = service.attestation.public_record()
        tampered["service_observation"]["model_profile"]["model_revision"] = "e" * 40
        with self.assertRaises(RecommissioningClientError):
            type(service.attestation).from_record(
                tampered,
                controller_public_key=service._trusted.controller_public_key_raw,
            )

    def test_service_attestation_rejects_resigned_profile_outside_controller_launch(self):
        service = _service()
        record = service.attestation.public_record()
        wrong_profile = _profile(model_revision="d" * 40)
        original = record["service_observation"]
        unsigned = {
            "schema_version": original["schema_version"],
            "model_profile": wrong_profile.public_manifest(),
            "model_profile_digest": wrong_profile.digest,
            "runtime_config_digest": original["runtime_config_digest"],
            "launch_receipt_digest": original["launch_receipt_digest"],
            "service_nonce": original["service_nonce"],
            "service_key_id": original["service_key_id"],
        }
        signable = {**unsigned, "observation_digest": digest_for(unsigned)}
        record["service_observation"] = {
            **signable,
            "signature": service._trusted._test_service_signer.sign_bytes(canonical_json(signable).encode("utf-8")),
        }
        core = {key: value for key, value in record.items() if key != "attestation_digest"}
        record["attestation_digest"] = digest_for(core)
        with self.assertRaisesRegex(RecommissioningClientError, "controller-authorized model profile"):
            ServiceAttestation.from_record(
                record,
                controller_public_key=service._trusted.controller_public_key_raw,
            )

    def test_public_receipt_self_digest_rejects_tampering(self):
        service = _service()
        ledger = InMemoryCoordinateTokenLedger.for_test()
        result = service.generate(_request(service, ledger), ledger=ledger, signer=TestSigner())
        tampered = result.public_receipt.public_record()
        tampered["task_ref"] = "task_" + "0" * 64
        with self.assertRaisesRegex(RecommissioningClientError, "digest does not bind"):
            PublicGenerationReceipt.validate_record(
                tampered,
                public_key=service._trusted.generation_receipt_public_key_raw,
            )

    def test_public_receipt_rejects_forged_signature_and_unpinned_key(self):
        service = _service()
        ledger = InMemoryCoordinateTokenLedger.for_test()
        result = service.generate(_request(service, ledger), ledger=ledger, signer=TestSigner())
        forged = result.public_receipt.public_record()
        attacker = ReceiptSigner.generate()
        signable = {key: value for key, value in forged.items() if key != "signature"}
        forged["signature"] = attacker.sign_bytes(canonical_json(signable).encode("utf-8"))
        with self.assertRaisesRegex(RecommissioningClientError, "signature is invalid"):
            PublicGenerationReceipt.validate_record(
                forged,
                public_key=service._trusted.generation_receipt_public_key_raw,
            )
        with self.assertRaisesRegex(RecommissioningClientError, "controller-pinned key"):
            service.generate(
                _request(service, InMemoryCoordinateTokenLedger.for_test()),
                ledger=InMemoryCoordinateTokenLedger.for_test(),
                signer=attacker,
            )

    def test_public_receipt_rejects_resigned_limits_that_disagree_with_budget_digest(self):
        service = _service()
        ledger = InMemoryCoordinateTokenLedger.for_test()
        result = service.generate(_request(service, ledger), ledger=ledger, signer=TestSigner())
        record = result.public_receipt.public_record()
        record["max_completion_tokens"] += 1
        record["coordinate_reserved_tokens"] += 1
        unsigned = {key: value for key, value in record.items() if key not in {"receipt_digest", "signature"}}
        signable = {**unsigned, "receipt_digest": digest_for(unsigned)}
        record = {
            **signable,
            "signature": TestSigner().sign_bytes(canonical_json(signable).encode("utf-8")),
        }
        with self.assertRaisesRegex(RecommissioningClientError, "invalid frozen budget|budget digest"):
            PublicGenerationReceipt.validate_record(record, public_key=TestSigner().public_key)

    def test_public_receipt_reconstructs_every_frozen_budget_limit(self):
        service = _service()
        ledger = InMemoryCoordinateTokenLedger.for_test()
        original = service.generate(_request(service, ledger), ledger=ledger, signer=TestSigner()).public_receipt
        for field_name in (
            "max_prompt_tokens",
            "max_completion_tokens",
            "max_total_tokens",
            "max_coordinate_tokens",
            "hard_timeout_ms",
            "seed",
        ):
            with self.subTest(field_name=field_name):
                record = original.public_record()
                core = {
                    key: value
                    for key, value in record.items()
                    if key not in {"signing_key_id", "receipt_digest", "signature"}
                }
                core[field_name] += 1
                with self.assertRaisesRegex(RecommissioningClientError, "frozen budget|budget digest"):
                    PublicGenerationReceipt.build(
                        core,
                        TestSigner(),
                        pinned_public_key=TestSigner().public_key,
                    )

    def test_generate_wall_clock_budget_includes_service_revalidation(self):
        transport = FakeTransport(attestation_sleep_seconds=0.08)
        service = _service(transport)
        budget = _budget(hard_timeout_ms=1)
        ledger = InMemoryCoordinateTokenLedger.for_test()
        with self.assertRaisesRegex(RecommissioningClientError, "elapsed-time budget"):
            service.generate(
                _request(service, ledger, budget=budget, envelope=_envelope(service.profile, budget)),
                ledger=ledger,
                signer=TestSigner(),
            )

    def test_public_candidate_projection_rejects_credential_or_path_evidence_ids(self):
        candidate = TypedCandidateResponse.parse(
            {
                "action": "COMMIT",
                "source": "def main(value):\n    return value\n",
                "summary": "bounded",
                "evidence_ids": ["credential sk-secret at C:/private/host.txt"],
            },
            visible_evidence_ids=("credential sk-secret at C:/private/host.txt",),
            public_locus="module:main",
        )
        with self.assertRaisesRegex(RecommissioningClientError, "prohibited public data"):
            candidate.public_redacted_record()


class PinnedTokenizerRendererTests(unittest.TestCase):
    def test_local_adapter_reads_assets_and_invokes_exact_chat_renderer(self):
        with tempfile.TemporaryDirectory() as directory:
            asset = os.path.join(directory, "tokenizer.json")
            with open(asset, "wb") as handle:
                handle.write(b'{"version":"test"}')
            asset_names = ("tokenizer.json",)
            tokenizer_digest = PinnedTokenizerRenderer.digest_local_tokenizer_assets(directory, asset_names)
            profile = _profile(tokenizer_digest=tokenizer_digest)
            tokenizer = FakeTokenizer(prompt_tokens=11)
            renderer = PinnedTokenizerRenderer.from_local_tokenizer(
                profile,
                tokenizer,
                tokenizer_root=directory,
                asset_names=asset_names,
            )
            envelope = _envelope(profile)
            preflight = TokenizationPreflight.from_pinned_renderer(envelope, profile, renderer)
            self.assertEqual(preflight.prompt_tokens, 11)
            self.assertEqual(preflight.renderer_mode, "PINNED_LOCAL")
            self.assertEqual([call[1] for call in tokenizer.calls], [False, True])
            with open(asset, "ab") as handle:
                handle.write(b"tamper")
            with self.assertRaisesRegex(RecommissioningClientError, "assets changed"):
                renderer.render(envelope, profile)

    def test_preflight_rejects_duck_typed_renderer_and_raw_artifact_injection(self):
        profile = _profile()
        envelope = _envelope(profile)

        class ForgedRenderer:
            def render(self, _envelope, _profile):
                return b"attacker", (1,)

        with self.assertRaisesRegex(RecommissioningClientError, "concrete pinned renderer"):
            TokenizationPreflight.from_pinned_renderer(envelope, profile, ForgedRenderer())
        with self.assertRaises(TypeError):
            TokenizationPreflight.from_pinned_renderer(
                envelope,
                profile,
                rendered_prompt="attacker",
                token_ids=(1,),
            )

    def test_local_tokenizer_inventory_is_exhaustive_and_implementation_bound(self):
        with tempfile.TemporaryDirectory() as directory:
            first = os.path.join(directory, "tokenizer.json")
            second = os.path.join(directory, "tokenizer_config.json")
            with open(first, "wb") as handle:
                handle.write(b'{"version":"test"}')
            with open(second, "wb") as handle:
                handle.write(b'{"padding_side":"left"}')
            with self.assertRaisesRegex(RecommissioningClientError, "exhaustive inventory"):
                PinnedTokenizerRenderer.digest_local_tokenizer_assets(directory, ("tokenizer.json",))
            tokenizer = FakeTokenizer(prompt_tokens=11)
            asset_digest = PinnedTokenizerRenderer.digest_local_tokenizer_assets(directory)
            profile = _profile(tokenizer_digest=asset_digest)
            renderer = PinnedTokenizerRenderer.from_local_tokenizer(
                profile,
                tokenizer,
                tokenizer_root=directory,
            )
            self.assertEqual(renderer.tokenizer_implementation_digest, profile.tokenizer_implementation_digest)
            with self.assertRaisesRegex(RecommissioningClientError, "implementation differs"):
                PinnedTokenizerRenderer.from_local_tokenizer(
                    replace(profile, tokenizer_implementation_digest=digest_for("substituted-implementation")),
                    tokenizer,
                    tokenizer_root=directory,
                )

    def test_added_tokens_mutation_cannot_change_ids_while_renderer_validates(self):
        with tempfile.TemporaryDirectory() as directory:
            tokenizer_path = os.path.join(directory, "tokenizer.json")
            added_tokens_path = os.path.join(directory, "added_tokens.json")
            with open(tokenizer_path, "wb") as handle:
                handle.write(b'{"version":"test"}')
            with open(added_tokens_path, "wb") as handle:
                handle.write(b'{"<extra>":128}')
            with self.assertRaisesRegex(RecommissioningClientError, "exhaustive inventory"):
                PinnedTokenizerRenderer.digest_local_tokenizer_assets(directory, ("tokenizer.json",))

            tokenizer = FakeTokenizer(prompt_tokens=11)
            profile = _profile(tokenizer_digest=PinnedTokenizerRenderer.digest_local_tokenizer_assets(directory))
            renderer = PinnedTokenizerRenderer.from_local_tokenizer(
                profile,
                tokenizer,
                tokenizer_root=directory,
            )
            before_ids = tokenizer.apply_chat_template([], tokenize=True, add_generation_prompt=True)
            with open(added_tokens_path, "wb") as handle:
                handle.write(b'{"<extra>":128,"<attacker>":129}')
            tokenizer.prompt_tokens = 12
            after_ids = tokenizer.apply_chat_template([], tokenize=True, add_generation_prompt=True)
            self.assertNotEqual(before_ids, after_ids)
            with self.assertRaisesRegex(RecommissioningClientError, "assets changed"):
                renderer.render(_envelope(profile), profile)


class PackageExportsTests(unittest.TestCase):
    def test_package_exports_every_recommissioning_trust_boundary(self):
        import egv.recommissioning as package

        for name in (
            "PinnedTokenizerRenderer",
            "PublicGenerationReceipt",
            "ServiceAttestation",
            "SignedEvaluatorReceipt",
            "TrustedLaunchReceipt",
            "VisibleTaskPackage",
            "lineage_feedback",
            "public_task_ref",
        ):
            with self.subTest(name=name):
                self.assertIn(name, package.__all__)
                self.assertIsNotNone(getattr(package, name))


if __name__ == "__main__":
    unittest.main()
