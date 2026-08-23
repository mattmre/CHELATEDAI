from __future__ import annotations

import base64
import hashlib
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

import torch

try:
    import peft  # noqa: F401
    from transformers import PretrainedConfig, PreTrainedModel
    from transformers.modeling_outputs import CausalLMOutput
except ImportError:  # pragma: no cover - dependency-gated production lane
    peft = None
    PretrainedConfig = object
    PreTrainedModel = torch.nn.Module
    CausalLMOutput = None

from egv.canonical import canonical_json, content_id, digest_bytes, digest_for
from egv.receipts import ReceiptSigner, key_id_for_public_key
from egv.training.contracts import LedgerCutoff, TrainingExample
from egv.training.targets import FULL_ATTENTION_LAYERS
from egv.training.trainer import SEALED_RUNTIME_DATASET_SCHEMA, run_production_training
from egv.variation.model import (
    MODEL_ARCHITECTURE,
    MODEL_CONFIG_CLASS,
    MODEL_REPOSITORY,
    MODEL_REVISION,
    LoadedPinnedModel,
    PinnedModelManifest,
    model_state_digest,
)
from egv.variation.generator import CandidateContext, ModelCandidateGenerator


class _FullAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.q_proj = torch.nn.Linear(4, 4)
        self.k_proj = torch.nn.Linear(4, 4)
        self.v_proj = torch.nn.Linear(4, 4)
        self.o_proj = torch.nn.Linear(4, 4)


class _LinearAttention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj_qkv = torch.nn.Linear(4, 4)
        self.in_proj_z = torch.nn.Linear(4, 4)
        self.in_proj_a = torch.nn.Linear(4, 4)
        self.in_proj_b = torch.nn.Linear(4, 4)
        self.out_proj = torch.nn.Linear(4, 4)


class _Layer(torch.nn.Module):
    def __init__(self, index):
        super().__init__()
        if index in FULL_ATTENTION_LAYERS:
            self.self_attn = _FullAttention()
        else:
            self.linear_attn = _LinearAttention()


class _TinyConfig(PretrainedConfig):
    model_type = "egv-tiny"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = 32
        self.hidden_size = 4


class _TinyCausalLM(PreTrainedModel):
    config_class = _TinyConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([_Layer(index) for index in range(24)])
        self.embed = torch.nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.lm_head

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        return {"input_ids": input_ids, **kwargs}

    def forward(self, input_ids=None, attention_mask=None, labels=None, **_kwargs):
        hidden = self.embed(input_ids)
        hidden = self.model.layers[3].self_attn.q_proj(hidden)
        logits = self.lm_head(hidden)
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), labels.reshape(-1), ignore_index=-100
        )
        return CausalLMOutput(loss=loss, logits=logits)


class _Tokenizer:
    eos_token_id = 1
    vocab_size = 32
    name_or_path = "egv-bounded-tokenizer"

    def __call__(self, text, *, add_special_tokens, truncation):
        if add_special_tokens or truncation:
            raise AssertionError("mutable tokenization")
        return {"input_ids": [2 + (len(item) % 29) for item in text.split()]}


@unittest.skipUnless(peft is not None and torch.cuda.is_available(), "real PEFT CUDA runtime unavailable")
class RealPeftTrainingIntegrationTests(unittest.TestCase):
    def test_candidate_generation_moves_one_prompt_to_cuda_model_device(self):
        class _GenerationModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.anchor = torch.nn.Parameter(torch.zeros(1, device="cuda"))

            def generate(self, *, input_ids, attention_mask, **_kwargs):
                self.assertions = (input_ids.device, attention_mask.device)
                suffix = torch.tensor([[9, 10]], device=self.anchor.device)
                return torch.cat((input_ids, suffix), dim=1)

        class _GenerationTokenizer:
            def apply_chat_template(self, messages, **kwargs):
                self.messages = messages
                self.kwargs = kwargs
                return {
                    "input_ids": torch.tensor([[1, 2, 3]], device="cpu"),
                    "attention_mask": torch.ones((1, 3), dtype=torch.long, device="cpu"),
                }

            def decode(self, _tokens, *, skip_special_tokens):
                self.skip_special_tokens = skip_special_tokens
                return canonical_json({
                    "source": "def solve(value):\n    return value\n",
                    "declared_locus": "module:solve",
                    "requested_authority": "EXECUTE_CANDIDATE",
                    "evidence_ids": [],
                })

        manifest = PinnedModelManifest(
            repository=MODEL_REPOSITORY,
            revision=MODEL_REVISION,
            architecture=MODEL_ARCHITECTURE,
            config_class=MODEL_CONFIG_CLASS,
            transformers_version="5.13.1",
            files={"weights.safetensors": "a" * 64},
            license={"name": "test", "source": "local"},
        )
        model = _GenerationModel()
        tokenizer = _GenerationTokenizer()
        loaded = LoadedPinnedModel(
            model=model,
            tokenizer=tokenizer,
            manifest=manifest,
            manifest_digest=manifest.digest(),
            file_hashes=dict(manifest.files),
            load_report={},
            base_state_digest=model_state_digest(model),
        )
        generator = ModelCandidateGenerator(loaded, model_digest=manifest.digest(), max_new_tokens=16)
        context = CandidateContext(
            "campaign", "run", 0, "C", "task", "PURE_FUNCTION", "module:solve", "rule",
            1, None, (), digest_for("retrieval"), manifest.digest(), None, digest_for("prompt"),
        )
        proposal = generator.propose(context)
        self.assertEqual(proposal.declared_locus, "module:solve")
        self.assertEqual(model.assertions, (model.anchor.device, model.anchor.device))
        self.assertEqual(tokenizer.kwargs["return_tensors"], "pt")
        self.assertEqual(tokenizer.messages[0]["role"], "user")
        self.assertEqual(tokenizer.kwargs["enable_thinking"], False)
        self.assertTrue(tokenizer.kwargs["tokenize"])

    def test_run_production_training_external_bridge_checkpoint_best_restore_and_reload(self):
        with tempfile.TemporaryDirectory(prefix="egv-real-peft-") as temporary:
            root = Path(temporary)
            model_root = root / "model"
            model_root.mkdir()
            weights = model_root / "weights.bin"
            weights.write_bytes(b"bounded immutable base")
            files = {"weights.bin": hashlib.sha256(weights.read_bytes()).hexdigest()}
            manifest = PinnedModelManifest(
                repository=MODEL_REPOSITORY, revision=MODEL_REVISION,
                architecture=MODEL_ARCHITECTURE, config_class=MODEL_CONFIG_CLASS,
                transformers_version="5.13.1", files=files,
                license={"name": "test", "source": "local"},
            )
            manifest_digest = manifest.digest()

            class _Loader:
                def __init__(self, supplied_root):
                    self.root = Path(supplied_root)

                def load(self, *, device="cpu", torch_dtype=None, adapter_artifact=None):
                    model = _TinyCausalLM(_TinyConfig()).to(device=device, dtype=torch_dtype)
                    adapter_digest = None
                    attestation = None
                    base_digest = model_state_digest(model)
                    if adapter_artifact is not None:
                        model = adapter_artifact.apply_to(model)
                        adapter_digest = adapter_artifact.digest
                        attestation = SimpleNamespace(
                            adapter_digest=adapter_digest,
                            base_model_manifest_digest=manifest_digest,
                            applied_model_state_digest=model_state_digest(model),
                        )
                    return LoadedPinnedModel(
                        model=model, tokenizer=_Tokenizer(), manifest=manifest,
                        manifest_digest=manifest_digest, file_hashes=files,
                        load_report={}, base_state_digest=base_digest,
                        adapter_digest=adapter_digest, adapter_attestation=attestation,
                    )

            cutoff = LedgerCutoff(
                "real-peft", 1, "event", digest_for("event"), digest_for("receipts"), 1, "key"
            )
            rows = []
            for index in range(20):
                prompt = "repair task {}".format(index)
                target = "return fixed"
                source_digest = digest_for({"source": index})
                sft = {
                    "schema_version": "egv-sft-row-v1", "task_id": "task-{:02d}".format(index),
                    "task_family": "PURE_FUNCTION", "arm": "B", "attempt_index": 1,
                    "candidate_artifact_digest": source_digest,
                    "prompt_digest": digest_for({"context": index}),
                    "diagnostic_enum": "PASS", "promotion_disposition": "PROMOTED",
                }
                values = {
                    "campaign_id": cutoff.campaign_id, "run_id": "run-{:02d}".format(index),
                    "task_id": sft["task_id"], "arm_id": "B", "seed": 1, "attempt_index": 1,
                    "candidate_id": "candidate-{:02d}".format(index),
                    "prompt_digest": digest_bytes(prompt.encode()), "target_digest": digest_bytes(target.encode()),
                    "sft_row_digest": digest_bytes(canonical_json(sft).encode()),
                    "prompt_context_digest": sft["prompt_digest"], "source_digest": source_digest,
                    "verdict_receipt_digest": digest_for({"verdict": index}),
                    "effect_receipt_digest": digest_for({"effect": index}),
                    "cutoff_digest": cutoff.digest, "input_token_count": 5,
                }
                row_id = content_id("trainrow", {"schema_version": "egv-private-training-example-v1", **values})
                rows.append(TrainingExample(
                    row_id=row_id, prompt=prompt, target=target, sft_row_json=canonical_json(sft), **values
                ))
            from egv.training.contracts import FrozenTrainingDataset

            dataset = FrozenTrainingDataset(cutoff, tuple(sorted(rows, key=lambda row: (
                row.task_id, row.arm_id, row.seed, row.attempt_index, row.candidate_id
            ))), {})
            dataset_path = root / "training.json"
            dataset_path.write_text(canonical_json({
                "schema_version": SEALED_RUNTIME_DATASET_SCHEMA, "manifest": dataset.manifest(),
                "private_rows": [row.private_record() for row in dataset.examples],
            }), encoding="utf-8")

            signer = ReceiptSigner.generate()
            public_key = root / "evaluator.pub"
            public_key.write_bytes(signer.public_key_raw)
            counter = root / "counter"
            service_path = root / "service.json"
            command = root / "evaluator.py"
            command.write_text(
                "#!/usr/bin/env python\n"
                "import json,sys\nfrom pathlib import Path\nsys.path.insert(0,{!r})\n"
                "from egv.canonical import content_id,digest_for\n"
                "from egv.receipts import GENESIS_HASH,ReceiptSigner\n"
                "request=json.loads(sys.stdin.read())\n"
                "assert 'adapter_root' not in request and request['adapter_reference']==request['adapter_digest']\n"
                "counter=Path({!r})\nindex=int(counter.read_text()) if counter.exists() else 0\ncounter.write_text(str(index+1))\n"
                "loss=[0.1,0.4,0.5][min(index,2)]\n"
                "signer=ReceiptSigner(__import__('base64').b64decode({!r}))\n"
                "service=json.loads(Path({!r}).read_text())\n"
                "out=digest_for({{'checkpoint_digest':request['checkpoint_digest'],'adapter_digest':request['adapter_digest'],'loss':loss,'sample_count':8}})\n"
                "receipt=signer.sign_receipt({{'receipt_type':'VERDICT','campaign_id':request['campaign_id'],'run_id':'development-loss','task_id':'DEVELOPMENT_LOSS','request_id':content_id('development-request',request['checkpoint_digest']),'candidate_id':content_id('development-adapter',request['adapter_digest']),'candidate_artifact_digest':request['adapter_digest'],'protocol_digest':request['protocol_digest'],'evaluator_digest':service['service_manifest_digest'],'decision':'PASS','diagnostic_enum':'PASS','resource_bucket':'UNDER_25','exit_status_class':'SUCCESS','input_digest':request['development_manifest_digest'],'output_digest':out,'effect_kind':'DEVELOPMENT_LOSS'}},sequence=1,previous_receipt_hash=GENESIS_HASH,idempotency_key=content_id('development-idempotency',request['checkpoint_digest']))\n"
                "print(json.dumps({{'schema_version':'egv-development-loss-response-v1','checkpoint_digest':request['checkpoint_digest'],'adapter_digest':request['adapter_digest'],'loss':loss,'sample_count':8,'receipt':receipt}}))\n".format(
                    str(Path.cwd()), str(counter), base64.b64encode(signer.private_key_raw).decode(), str(service_path)
                ), encoding="utf-8"
            )
            command.chmod(0o700)
            transfer_command = root / "transfer.py"
            transfer_command.write_text(
                "import base64,json,sys\n"
                "from pathlib import Path\n"
                "sys.path.insert(0,{!r})\n"
                "from egv.canonical import canonical_bytes\n"
                "from egv.receipts import ReceiptSigner\n"
                "request=json.loads(sys.stdin.read())\n"
                "signer=ReceiptSigner(base64.b64decode({!r}))\n"
                "unsigned={{'schema_version':'egv-adapter-transfer-response-v1','service_manifest_digest':request['service_manifest_digest'],'request_digest':request['request_digest'],'adapter_digest':request['adapter_digest'],'adapter_reference':request['adapter_digest'],'signing_key_id':signer.key_id}}\n"
                "print(json.dumps({{**unsigned,'signature':signer.sign_bytes(canonical_bytes(unsigned))}}))\n".format(
                    str(Path.cwd()), base64.b64encode(signer.private_key_raw).decode()
                ),
                encoding="utf-8",
            )
            from egv.training.protocol import TrainingProtocol

            protocol = TrainingProtocol()
            unsigned_service = {
                "schema_version": "egv-external-development-service-v1", "campaign_id": cutoff.campaign_id,
                "evaluator_key_id": key_id_for_public_key(signer.public_key_raw),
                "evaluator_public_key_digest": hashlib.sha256(signer.public_key_raw).hexdigest(),
                "endpoint_digest": hashlib.sha256(command.read_bytes()).hexdigest(),
                "transfer_endpoint_digest": hashlib.sha256(transfer_command.read_bytes()).hexdigest(),
                "development_manifest_digest": digest_for("private-dev"), "development_task_count": 8,
                "development_row_ids": ["dev-{:02d}".format(index) for index in range(8)],
                "model_digest": manifest_digest, "protocol_digest": protocol.digest,
            }
            service_path.write_text(canonical_json({
                **unsigned_service, "service_manifest_digest": digest_for(unsigned_service)
            }), encoding="utf-8")
            with patch("egv.variation.model.PinnedModelLoader", _Loader):
                result = run_production_training(
                    model_root=model_root, training_dataset=dataset_path,
                    evaluator_manifest=service_path, evaluator_public_key=public_key,
                    evaluator_command=command, evaluator_transfer_command=transfer_command,
                    output_root=root / "output", device="cuda",
                )
            self.assertTrue(result["real_qwen_execution_claimed"])
            self.assertEqual(result["selected_loss"], 0.1)
            self.assertEqual(result["adapter_manifest_digest"], result["development_evaluations"][0]["receipt"]["candidate_artifact_digest"])
            self.assertTrue((root / "output" / "adapter" / "adapter_model.safetensors").is_file())


if __name__ == "__main__":
    unittest.main()
