from __future__ import annotations

from contextlib import redirect_stderr
import http.client
import io
import json
import os
from pathlib import Path
import tempfile
import threading
import unittest
from urllib import error as urllib_error
from urllib import request as urllib_request

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from egv.canonical import canonical_json, digest_for
from egv.receipts import ReceiptSigner
from egv.recommissioning.client import AttestedLoopbackService, ModelServiceProfile, TrustedLaunchReceipt
from egv.recommissioning.gateway import (
    GatewayConfig,
    LoopbackAttestationGateway,
    RecommissioningGatewayError,
    build_gateway_server,
    issue_trusted_launch_receipt,
    main,
    sign_service_observation,
)


MODEL_ID = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
SERVICE_NONCE = "nonce_" + "7a" * 32


def _profile(**updates):
    values = {
        "model_bundle": "LIGHTNING",
        "model_id": MODEL_ID,
        "model_revision": "cc84af2fe71647d87f4486c064f320e1e7535243",
        "engine_image_digest": "sha256:" + "a" * 64,
        "max_model_len": 8192,
        "precision": "NVFP4",
        "kv_cache_dtype": "fp8",
        "tokenizer_digest": digest_for("gateway-test-tokenizer"),
        "tokenizer_implementation_digest": digest_for("gateway-test-tokenizer-implementation"),
        "chat_template_digest": digest_for("gateway-test-chat-template"),
        "weights_digest": digest_for("gateway-test-weights"),
        "license_digest": digest_for("gateway-test-license"),
    }
    values.update(updates)
    return ModelServiceProfile(**values)


class _UpstreamState:
    def __init__(self, profile):
        self.health_body = canonical_json({"status": "ok"}).encode("utf-8")
        self.models_body = canonical_json({"data": [{"id": profile.model_id}]}).encode("utf-8")
        self.chat_body = canonical_json({"fixture": "non-task-control"}).encode("utf-8")
        self.chat_calls = 0
        self.redirect_health = False


class _FakeUpstreamHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    @property
    def state(self):
        return self.server.state

    def log_message(self, format, *args):  # type: ignore[no-untyped-def]
        return

    def _reply(self, status, body):
        self.send_response_only(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        self.end_headers()
        self.wfile.write(body)
        self.close_connection = True

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            if self.state.redirect_health:
                self.send_response_only(302)
                self.send_header("Location", "/redirected")
                self.send_header("Content-Length", "0")
                self.send_header("Connection", "close")
                self.end_headers()
                return
            self._reply(200, self.state.health_body)
            return
        if self.path == "/v1/models":
            self._reply(200, self.state.models_body)
            return
        self._reply(404, b"{}")

    def do_POST(self):  # noqa: N802
        if self.path != "/v1/chat/completions":
            self._reply(404, b"{}")
            return
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        self.state.chat_calls += 1
        self._reply(200, self.state.chat_body)


class _LiveServer:
    def __init__(self, server):
        self.server = server
        self.thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01})
        self.thread.daemon = True

    def __enter__(self):
        self.thread.start()
        return self.server

    def __exit__(self, exc_type, exc, traceback):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2.0)


def _fake_upstream(profile):
    state = _UpstreamState(profile)
    server = ThreadingHTTPServer(("127.0.0.1", 0), _FakeUpstreamHandler)
    server.state = state
    return state, server


def _config(profile, upstream_port, **updates):
    record = {
        "schema_version": "egv-recommissioning-gateway-config-v1",
        "bind_host": "127.0.0.1",
        "bind_port": 18443,
        "upstream_origin": "http://127.0.0.1:{}".format(upstream_port),
        "upstream_timeout_ms": 2000,
        "max_request_bytes": 4096,
        "max_response_bytes": 4096,
        "max_concurrency": 4,
        "service_nonce": SERVICE_NONCE,
        "model_profile": profile.public_manifest(),
    }
    record.update(updates)
    return GatewayConfig.from_record(record)


def _launch(config, controller, service, generation):
    record = issue_trusted_launch_receipt(
        profile=config.profile,
        runtime_config_digest=config.runtime_config_digest,
        service_nonce=config.service_nonce,
        controller_signer=controller,
        service_public_key=service.public_key,
        generation_receipt_public_key=generation.public_key,
    )
    return record, TrustedLaunchReceipt.from_controller_record(
        record,
        controller_public_key=controller.public_key,
    )


def _gateway(config, trusted, service):
    return LoopbackAttestationGateway(
        profile=config.profile,
        trusted_launch_receipt=trusted,
        service_signer=service,
        runtime_config_digest=config.runtime_config_digest,
        service_nonce=config.service_nonce,
        upstream_origin=config.upstream_origin,
        upstream_timeout_seconds=config.upstream_timeout_ms / 1000.0,
        max_request_bytes=config.max_request_bytes,
        max_response_bytes=config.max_response_bytes,
        max_concurrency=config.max_concurrency,
    )


class GatewayTrustChainTests(unittest.TestCase):
    def setUp(self):
        self.profile = _profile()
        self.controller = ReceiptSigner(b"C" * 32)
        self.service = ReceiptSigner(b"S" * 32)
        self.generation = ReceiptSigner(b"G" * 32)

    def test_issue_launch_and_service_observation_use_exact_closed_contracts(self):
        config = _config(self.profile, 18080)
        record, trusted = _launch(config, self.controller, self.service, self.generation)
        observation = sign_service_observation(
            profile=self.profile,
            trusted_launch_receipt=trusted,
            service_signer=self.service,
        )
        self.assertEqual(record["schema_version"], "egv-trusted-launch-receipt-v2")
        self.assertEqual(observation["schema_version"], "egv-recommissioning-service-observation-v1")
        self.assertEqual(observation["launch_receipt_digest"], record["launch_receipt_digest"])
        self.assertEqual(observation["model_profile"], self.profile.public_manifest())
        with self.assertRaisesRegex(RecommissioningGatewayError, "keys must be distinct"):
            issue_trusted_launch_receipt(
                profile=self.profile,
                runtime_config_digest=config.runtime_config_digest,
                service_nonce=config.service_nonce,
                controller_signer=self.controller,
                service_public_key=self.controller.public_key,
                generation_receipt_public_key=self.generation.public_key,
            )

    def test_wrong_signer_profile_runtime_and_nonce_fail_closed(self):
        config = _config(self.profile, 18080)
        record, trusted = _launch(config, self.controller, self.service, self.generation)
        wrong_service = ReceiptSigner(b"W" * 32)
        with self.assertRaisesRegex(RecommissioningGatewayError, "service signer differs"):
            sign_service_observation(
                profile=self.profile,
                trusted_launch_receipt=trusted,
                service_signer=wrong_service,
            )
        with self.assertRaisesRegex(RecommissioningGatewayError, "profile differs"):
            sign_service_observation(
                profile=_profile(model_revision="d" * 40),
                trusted_launch_receipt=trusted,
                service_signer=self.service,
            )
        with self.assertRaisesRegex(Exception, "unexpected controller key|signature is invalid"):
            TrustedLaunchReceipt.from_controller_record(
                record,
                controller_public_key=ReceiptSigner(b"X" * 32).public_key,
            )
        with self.assertRaisesRegex(RecommissioningGatewayError, "runtime configuration differs"):
            LoopbackAttestationGateway(
                profile=self.profile,
                trusted_launch_receipt=trusted,
                service_signer=self.service,
                runtime_config_digest=digest_for("wrong-runtime"),
                service_nonce=config.service_nonce,
                upstream_origin=config.upstream_origin,
                upstream_timeout_seconds=2.0,
                max_request_bytes=4096,
                max_response_bytes=4096,
                max_concurrency=1,
            )
        with self.assertRaisesRegex(RecommissioningGatewayError, "service nonce differs"):
            LoopbackAttestationGateway(
                profile=self.profile,
                trusted_launch_receipt=trusted,
                service_signer=self.service,
                runtime_config_digest=config.runtime_config_digest,
                service_nonce="nonce_" + "8b" * 32,
                upstream_origin=config.upstream_origin,
                upstream_timeout_seconds=2.0,
                max_request_bytes=4096,
                max_response_bytes=4096,
                max_concurrency=1,
            )
        with self.assertRaisesRegex(RecommissioningGatewayError, "service nonce"):
            issue_trusted_launch_receipt(
                profile=self.profile,
                runtime_config_digest=config.runtime_config_digest,
                service_nonce="predictable",
                controller_signer=self.controller,
                service_public_key=self.service.public_key,
                generation_receipt_public_key=self.generation.public_key,
            )

    def test_config_rejects_non_loopback_and_noncanonical_origins(self):
        with self.assertRaisesRegex(RecommissioningGatewayError, "numeric loopback"):
            _config(self.profile, 18080, bind_host="0.0.0.0")
        with self.assertRaisesRegex(RecommissioningGatewayError, "loopback-only"):
            _config(self.profile, 18080, upstream_origin="http://192.0.2.5:8000")
        with self.assertRaisesRegex(RecommissioningGatewayError, "normalized"):
            _config(self.profile, 18080, upstream_origin="http://127.0.0.1:18080/")
        with self.assertRaisesRegex(RecommissioningGatewayError, "numeric loopback"):
            build_gateway_server(object(), bind_host="0.0.0.0", bind_port=18081)  # type: ignore[arg-type]

    def test_gateway_surfaces_are_lazy_package_exports(self):
        from egv import recommissioning

        for name in (
            "GatewayConfig",
            "LoopbackAttestationGateway",
            "RecommissioningGatewayError",
            "build_gateway_server",
            "issue_trusted_launch_receipt",
            "sign_service_observation",
        ):
            self.assertIsNotNone(getattr(recommissioning, name))


class GatewayHTTPTests(unittest.TestCase):
    def setUp(self):
        self.profile = _profile()
        self.controller = ReceiptSigner(b"C" * 32)
        self.service = ReceiptSigner(b"S" * 32)
        self.generation = ReceiptSigner(b"G" * 32)

    def _start_pair(self):
        state, upstream = _fake_upstream(self.profile)
        upstream_live = _LiveServer(upstream)
        upstream_live.__enter__()
        config = _config(self.profile, upstream.server_address[1])
        record, trusted = _launch(config, self.controller, self.service, self.generation)
        gateway = _gateway(config, trusted, self.service)
        server = build_gateway_server(
            gateway,
            bind_host="127.0.0.1",
            bind_port=0,
            allow_ephemeral_port=True,
        )
        return state, upstream_live, config, record, trusted, gateway, server

    def test_real_loopback_connect_succeeds_without_chat_and_public_chain_has_no_topology(self):
        state, upstream_live, config, record, trusted, gateway, server = self._start_pair()
        try:
            with _LiveServer(server):
                endpoint = "http://127.0.0.1:{}".format(server.server_address[1])
                connected = AttestedLoopbackService.connect(endpoint, self.profile, trusted)
                self.assertEqual(connected.attestation.model_profile_digest, self.profile.digest)
                public = canonical_json(
                    {
                        "launch": record,
                        "observation": gateway._observation,
                        "attestation": connected.attest(),
                    }
                )
                for forbidden in (
                    "http://",
                    "127.0.0.1",
                    "::1",
                    "upstream_origin",
                    "bind_host",
                    "bind_port",
                    str(config.bind_port),
                ):
                    self.assertNotIn(forbidden, public)
                self.assertEqual(state.chat_calls, 0)
        finally:
            upstream_live.__exit__(None, None, None)

    def test_extra_routes_and_methods_fail_and_one_bounded_non_task_proxy_passes(self):
        state, upstream_live, _config_value, _record, _trusted, _gateway_value, server = self._start_pair()
        try:
            with _LiveServer(server):
                origin = "http://127.0.0.1:{}".format(server.server_address[1])
                with self.assertRaises(urllib_error.HTTPError) as missing:
                    urllib_request.urlopen(origin + "/extra", timeout=2.0)
                self.assertEqual(missing.exception.code, 404)
                head = urllib_request.Request(origin + "/health", method="HEAD")
                with self.assertRaises(urllib_error.HTTPError) as wrong_method:
                    urllib_request.urlopen(head, timeout=2.0)
                self.assertEqual(wrong_method.exception.code, 405)
                body = canonical_json({"fixture": "non-task-control", "model": self.profile.model_id}).encode("utf-8")
                request = urllib_request.Request(
                    origin + "/v1/chat/completions",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with urllib_request.urlopen(request, timeout=2.0) as response:
                    self.assertEqual(response.read(), state.chat_body)
                self.assertEqual(state.chat_calls, 1)

                noncanonical = b'{"model": "' + self.profile.model_id.encode("utf-8") + b'"}'
                request = urllib_request.Request(
                    origin + "/v1/chat/completions",
                    data=noncanonical,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with self.assertRaises(urllib_error.HTTPError) as rejected:
                    urllib_request.urlopen(request, timeout=2.0)
                self.assertEqual(rejected.exception.code, 503)
                self.assertEqual(state.chat_calls, 1)

                deeply_nested = []
                cursor = deeply_nested
                for _index in range(20):
                    child = []
                    cursor.append(child)
                    cursor = child
                body = canonical_json({"model": self.profile.model_id, "nested": deeply_nested}).encode("utf-8")
                request = urllib_request.Request(
                    origin + "/v1/chat/completions",
                    data=body,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                with self.assertRaises(urllib_error.HTTPError) as too_deep:
                    urllib_request.urlopen(request, timeout=2.0)
                self.assertEqual(too_deep.exception.code, 503)
                self.assertEqual(state.chat_calls, 1)

                connection = http.client.HTTPConnection("127.0.0.1", server.server_address[1], timeout=2.0)
                connection.request(
                    "POST",
                    "/v1/chat/completions",
                    body=b"x" * 4097,
                    headers={"Content-Type": "application/json"},
                )
                response = connection.getresponse()
                self.assertEqual(response.status, 413)
                response.read()
                connection.close()
                self.assertEqual(state.chat_calls, 1)
        finally:
            upstream_live.__exit__(None, None, None)

    def test_upstream_drift_permanently_latches_before_any_chat(self):
        state, upstream_live, _config_value, _record, _trusted, gateway, server = self._start_pair()
        try:
            frozen_models = state.models_body
            state.models_body = canonical_json({"data": [{"id": self.profile.model_id, "drift": True}]}).encode("utf-8")
            with _LiveServer(server):
                origin = "http://127.0.0.1:{}".format(server.server_address[1])
                with self.assertRaises(urllib_error.HTTPError) as drifted:
                    urllib_request.urlopen(origin + "/v1/egv/attestation", timeout=2.0)
                self.assertEqual(drifted.exception.code, 503)
                self.assertTrue(gateway.failed_closed)
                state.models_body = frozen_models
                with self.assertRaises(urllib_error.HTTPError) as still_closed:
                    urllib_request.urlopen(origin + "/health", timeout=2.0)
                self.assertEqual(still_closed.exception.code, 503)
                self.assertEqual(state.chat_calls, 0)
        finally:
            upstream_live.__exit__(None, None, None)

    def test_wrong_model_and_redirecting_upstream_cannot_start_gateway(self):
        state, upstream = _fake_upstream(self.profile)
        with _LiveServer(upstream):
            config = _config(self.profile, upstream.server_address[1])
            _record, trusted = _launch(config, self.controller, self.service, self.generation)
            state.models_body = canonical_json({"data": [{"id": "foreign-model"}]}).encode("utf-8")
            with self.assertRaisesRegex(RecommissioningGatewayError, "model identity differs"):
                _gateway(config, trusted, self.service)
            state.models_body = canonical_json({"data": [{"id": self.profile.model_id}]}).encode("utf-8")
            state.redirect_health = True
            with self.assertRaisesRegex(RecommissioningGatewayError, "bounded upstream request failed"):
                _gateway(config, trusted, self.service)


class GatewayCLITests(unittest.TestCase):
    def test_issue_launch_is_atomic_no_overwrite_and_does_not_publish_paths(self):
        profile = _profile()
        controller = ReceiptSigner(b"C" * 32)
        service = ReceiptSigner(b"S" * 32)
        generation = ReceiptSigner(b"G" * 32)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = _config(profile, 18080)
            config_path = root / "private-config.json"
            controller_path = root / "controller.key"
            service_path = root / "service.pub"
            generation_path = root / "generation.pub"
            output_path = root / "launch.json"
            config_path.write_text(canonical_json(config.private_record()), encoding="utf-8")
            controller_path.write_bytes(controller.private_key_raw)
            service_path.write_bytes(service.public_key_raw)
            generation_path.write_bytes(generation.public_key_raw)
            if os.name != "nt":
                controller_path.chmod(0o600)
            argv = [
                "issue-launch",
                "--config",
                str(config_path),
                "--controller-private-key",
                str(controller_path),
                "--service-public-key",
                str(service_path),
                "--generation-receipt-public-key",
                str(generation_path),
                "--output",
                str(output_path),
            ]
            self.assertEqual(main(argv), 0)
            first = output_path.read_bytes()
            TrustedLaunchReceipt.from_controller_record(
                json.loads(first),
                controller_public_key=controller.public_key,
            )
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                main(argv)
            self.assertEqual(output_path.read_bytes(), first)
            public = first.decode("utf-8")
            self.assertNotIn(str(root), public)
            self.assertNotIn("http://", public)
            self.assertNotIn("127.0.0.1", public)


if __name__ == "__main__":
    unittest.main()
