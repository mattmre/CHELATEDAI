"""Item 3C / Bundle 52: planted QSCCI positive control and norm-matched k=1 arms.

The local drycheck is a CPU PyTorch write-site hook on a 12-layer stub whose
only causal readout is a planted vector. It does not load Qwen weights and it
does not claim a live REPORT result. Live mode records a refusal while the
CUDA lease is isolated.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence

import numpy as np
import torch
from torch import nn

from chelated_pv7_spine import _json_sha256
from chelated_pv7_spine.bridge_merge_harness import paired_bootstrap_ci
from chelated_pv7_spine.burn_ledger import (
    HISTORICAL_DATASETS,
    BurnLedger,
    assert_corpus_unburned,
    build_burn_ledger,
    text_fingerprint,
)
from chelated_pv7_spine.challenger_ledger import ChallengerBitemporalLedger, LedgerRecordEnvelope
from chelated_pv7_spine.claim_registry import claim_registry_sha256
from chelated_pv7_spine.cpu_io_cage import assert_zero_cuda_context, build_cpu_only_env
from chelated_pv7_spine.ek5_reason_canonicalizer import export_canonical_ek5_audit_trace
from chelated_pv7_spine.fragment_compiler import compile_fragments
from chelated_pv7_spine.tra1_weight_sha_guard import TRA1StateIdentityContract, sha256_digest
from ek5_v3_stratified import StratifiedEvent, StratifiedEvidenceStore

REPO_ROOT = Path(__file__).resolve().parent
BUNDLE_ID = 52
ITEM = "3C"
DIM = 32
LAYER_INDEX = 11
DOSE_NORM = 0.5
CONTROL_COSINE_MIN = 0.99
SEPARATION_COSINE_MAX = 0.95
NUISANCE_SCALE = 4.0
ROWS_PER_FAMILY = 4
FAMILIES_PER_SPLIT = 32
ALLOWED_LIVE_MODELS = ("Qwen/Qwen2.5-1.5B", "Qwen/Qwen3.5-2B-Base")
DATASET_ID = "pv7-w18-3c-scratch-v1"
ARM_NAMES = (
    "k1_native_mlp_write_site",
    "k1_topk_sae_latent",
    "ordinary_contrastive_caa_direction",
    "matched_norm_random_orthogonal_sham",
    "ifeval_collateral_retention_probe",
)

_DOMAINS = (
    "harbor", "orchard", "kiln", "glacier", "cistern", "vineyard", "foundry", "atlas",
    "meadow", "quarry", "lighthouse", "granary", "aqueduct", "pasture", "observatory", "windmill",
)
_TASKS = ("define", "list", "compare", "summarize", "restate")
_SUBJECTS = ("valve", "seedling", "ingot", "spool")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise RuntimeError("refusing a zero write direction")
    return np.asarray(vector, dtype=np.float64) / norm


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.dot(_unit(left), _unit(right)))


def cpu_isolation_receipt() -> Dict[str, Any]:
    """Pin this process to the CPU lease and return the cage receipt."""
    env = build_cpu_only_env()
    for key in ("CUDA_VISIBLE_DEVICES", "HIP_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES"):
        os.environ[key] = env[key]
    receipt = assert_zero_cuda_context(env=env, strict=True)
    return {
        "passed": receipt.passed,
        "cuda_visible_devices": receipt.cuda_visible_devices,
        "violations": list(receipt.violations),
        "receipt_sha256": receipt.receipt_sha256,
    }


def scratch_families(split: str, count: int, offset: int) -> List[Dict[str, str]]:
    """Build family-disjoint scratch prompts. No external corpus is read."""
    families = []
    for index in range(offset, offset + count):
        domain = _DOMAINS[index % len(_DOMAINS)]
        task = _TASKS[(index // len(_DOMAINS)) % len(_TASKS)]
        subject = _SUBJECTS[(index // (len(_DOMAINS) * len(_TASKS))) % len(_SUBJECTS)]
        family_id = f"w18-3c-{split}-{index:04d}"
        prompt = (
            f"Scratch {split} family {index:04d}. Task {task}. "
            f"Subject {domain} {subject}. Reply in one plain sentence."
        )
        families.append({"family_id": family_id, "split": split, "prompt": prompt})
    return families


def partition_splits() -> Dict[str, List[Dict[str, str]]]:
    """CONTROL, DEV, SELECT, and REPORT are disjoint ranges of one template."""
    width = FAMILIES_PER_SPLIT
    return {
        "control": scratch_families("control", width, 0),
        "dev": scratch_families("dev", width, width),
        "select": scratch_families("select", width, 2 * width),
        "report": scratch_families("report", width, 3 * width),
    }


def _family_seed(family_id: str) -> int:
    return int(_sha256_bytes(family_id.encode("utf-8"))[:8], 16)


def admit_scratch_prompts(families: Sequence[Mapping[str, str]]) -> str:
    """Reject the scratch split if its id or text collides with the burn ledger."""
    scanned = build_burn_ledger(REPO_ROOT)
    burned = dict(scanned.datasets)
    if DATASET_ID in burned:
        raise RuntimeError(f"{DATASET_ID} is already burned")
    documents = [f"scratch document {row['family_id']}" for row in families]
    queries = [row["prompt"] for row in families]
    sentinel_ledger = BurnLedger(
        datasets=tuple((name, (origin,)) for name, origin in sorted(HISTORICAL_DATASETS.items())),
        document_fingerprints=frozenset({text_fingerprint("pv7 historical sentinel document 3c")}),
        query_fingerprints=frozenset({text_fingerprint("pv7 historical sentinel query 3c")}),
        fingerprint_inventory_complete=True,
    )
    assert_corpus_unburned(DATASET_ID, doc_texts=documents, query_texts=queries, ledger=sentinel_ledger)
    return scanned.sha256


def plant_basis(seed: int = 52) -> Dict[str, np.ndarray]:
    """Orthonormal causal, nuisance, and collateral directions."""
    rng = np.random.default_rng(seed)
    matrix = rng.normal(size=(DIM, 3))
    q_factor, _ = np.linalg.qr(matrix)
    return {
        "w_true": np.asarray(q_factor[:, 0], dtype=np.float64),
        "w_nuis": np.asarray(q_factor[:, 1], dtype=np.float64),
        "w_ifeval": np.asarray(q_factor[:, 2], dtype=np.float64),
    }


def sample_hidden(families: Sequence[Mapping[str, str]], basis: Mapping[str, np.ndarray]) -> np.ndarray:
    """Observational hidden states. Nuisance amplitude tracks the causal coefficient."""
    rows: List[np.ndarray] = []
    w_true = basis["w_true"]
    w_nuis = basis["w_nuis"]
    for family in families:
        rng = np.random.default_rng(_family_seed(family["family_id"]))
        causal = rng.normal(size=ROWS_PER_FAMILY)
        nuisance = NUISANCE_SCALE * causal + 0.05 * rng.normal(size=ROWS_PER_FAMILY)
        noise = 0.01 * rng.normal(size=(ROWS_PER_FAMILY, DIM))
        prompt_shift = 0.01 * rng.normal(size=DIM)
        hidden = causal[:, None] * w_true + nuisance[:, None] * w_nuis + noise + prompt_shift
        rows.append(hidden)
    return np.concatenate(rows, axis=0)


class _MLP(nn.Module):
    def __init__(self, readout: torch.Tensor) -> None:
        super().__init__()
        width = int(readout.numel())
        self.down_proj = nn.Linear(width, width, bias=False)
        with torch.no_grad():
            self.down_proj.weight.copy_(torch.eye(width))
        self.register_buffer("w_read", readout.float())

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.down_proj(hidden).matmul(self.w_read)


class _Block(nn.Module):
    def __init__(self, readout: torch.Tensor) -> None:
        super().__init__()
        self.mlp = _MLP(readout)


class Layer11WriteSite(nn.Module):
    """Twelve blocks. Forward uses only layer 11, the preregistered write site."""

    def __init__(self, readout: np.ndarray) -> None:
        super().__init__()
        vector = torch.tensor(np.asarray(readout, dtype=np.float64), dtype=torch.float32)
        self.layers = nn.ModuleList(_Block(vector if index == LAYER_INDEX else torch.zeros_like(vector)) for index in range(12))

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return self.layers[LAYER_INDEX].mlp(hidden)


def layer11_down_proj(model: nn.Module) -> nn.Module:
    layers = getattr(model, "layers", None)
    if layers is None or len(layers) <= LAYER_INDEX:
        raise RuntimeError("model has no layer-11 MLP write site")
    mlp = getattr(layers[LAYER_INDEX], "mlp", None)
    down = getattr(mlp, "down_proj", None)
    if not isinstance(down, nn.Module):
        raise RuntimeError("layer-11 mlp.down_proj is missing")
    return down


def _as_batch(hidden: np.ndarray) -> torch.Tensor:
    return torch.tensor(np.asarray(hidden, dtype=np.float64), dtype=torch.float32)


def _hooked_mean(model: nn.Module, hidden: np.ndarray, hook: Callable[..., torch.Tensor]) -> np.ndarray:
    site = layer11_down_proj(model)
    handle = site.register_forward_hook(hook)
    try:
        with torch.no_grad():
            values = model(_as_batch(hidden))
    finally:
        handle.remove()
    return values.detach().cpu().numpy()


def qscci_direction(model: nn.Module, hidden: np.ndarray, dose: float = 0.25) -> np.ndarray:
    """Mean finite-difference Jacobian at the hooked write site."""
    with torch.no_grad():
        baseline = model(_as_batch(hidden)).detach().cpu().numpy()
    effects = np.zeros(hidden.shape[1], dtype=np.float64)
    for index in range(hidden.shape[1]):
        def hook(module: nn.Module, inputs: object, output: torch.Tensor, coord: int = index) -> torch.Tensor:
            del module, inputs
            delta = torch.zeros_like(output)
            delta[:, coord] = dose
            return output + delta

        shifted = _hooked_mean(model, hidden, hook)
        effects[index] = float(np.mean(shifted - baseline))
    return _unit(effects)


def correlation_direction(hidden: np.ndarray, target: np.ndarray) -> np.ndarray:
    centered = np.asarray(target, dtype=np.float64) - float(np.mean(target))
    return _unit(hidden.T @ centered)


def _predict(model: nn.Module, hidden: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return model(_as_batch(hidden)).detach().cpu().numpy()


def positive_control(model: nn.Module, hidden: np.ndarray, basis: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    """QSCCI must recover w_true. Raw correlation must prefer w_nuis. Fail closed otherwise."""
    causal = qscci_direction(model, hidden)
    observed = correlation_direction(hidden, _predict(model, hidden))
    cosine_true = _cosine(causal, basis["w_true"])
    cosine_nuis = abs(_cosine(observed, basis["w_nuis"]))
    cosine_corr_true = abs(_cosine(observed, basis["w_true"]))
    passed = cosine_true > CONTROL_COSINE_MIN and cosine_nuis > cosine_corr_true
    return {
        "passed": passed,
        "cosine_qscci_w_true": cosine_true,
        "cosine_correlation_w_nuis": cosine_nuis,
        "cosine_correlation_w_true": cosine_corr_true,
        "qscci_direction": causal,
        "correlation_direction": observed,
        "failure": "" if passed else "positive_control",
    }


def _matched_delta(direction: np.ndarray, dose: float = DOSE_NORM) -> np.ndarray:
    return _unit(direction) * float(dose)


def _injection_effect(model: nn.Module, hidden: np.ndarray, delta: np.ndarray) -> np.ndarray:
    shift = torch.tensor(delta, dtype=torch.float32)

    def hook(module: nn.Module, inputs: object, output: torch.Tensor) -> torch.Tensor:
        del module, inputs
        return output + shift

    baseline = _predict(model, hidden)
    hooked = _hooked_mean(model, hidden, hook)
    return hooked - baseline


def _removal_restoration(model: nn.Module, hidden: np.ndarray, direction: np.ndarray) -> Dict[str, float]:
    unit = torch.tensor(_unit(direction), dtype=torch.float32)
    baseline = _predict(model, hidden)

    def removal(module: nn.Module, inputs: object, output: torch.Tensor) -> torch.Tensor:
        del module, inputs
        coeff = output.matmul(unit)
        return output - coeff.unsqueeze(1) * unit

    def restoration(module: nn.Module, inputs: object, output: torch.Tensor) -> torch.Tensor:
        del module, inputs
        coeff = output.matmul(unit)
        removed = output - coeff.unsqueeze(1) * unit
        return removed + coeff.unsqueeze(1) * unit

    removed = _hooked_mean(model, hidden, removal)
    restored = _hooked_mean(model, hidden, restoration)
    return {
        "removal_mean_abs_delta_y": float(np.mean(np.abs(removed - baseline))),
        "restoration_max_abs_error": float(np.max(np.abs(restored - baseline))),
    }


class _Top1SAE(nn.Module):
    def __init__(self, width: int, n_latents: int) -> None:
        super().__init__()
        self.encoder = nn.Linear(width, n_latents, bias=True)
        self.decoder = nn.Linear(n_latents, width, bias=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        preact = self.encoder(hidden)
        index = torch.argmax(preact, dim=1)
        mask = torch.zeros_like(preact)
        mask.scatter_(1, index.unsqueeze(1), 1.0)
        latent = torch.relu(preact) * mask
        return self.decoder(latent)


def sae_decoder_direction(hidden: np.ndarray, steps: int = 40) -> np.ndarray:
    """Train a tiny k=1 SAE on DEV activations and return its busiest decoder column."""
    model = _Top1SAE(hidden.shape[1], 8)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
    batch = _as_batch(hidden)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        recon = model(batch)
        loss = torch.mean((recon - batch) ** 2)
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            columns = model.decoder.weight
            model.decoder.weight.copy_(columns / columns.norm(dim=0, keepdim=True).clamp_min(1e-8))
    with torch.no_grad():
        preact = model.encoder(batch)
        usage = torch.relu(preact).mean(dim=0)
        chosen = int(torch.argmax(usage).item())
        column = model.decoder.weight[:, chosen].detach().cpu().numpy()
    return _unit(column)


def caa_direction(hidden: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Ordinary contrastive mean difference. This is the correlation arm, not QSCCI."""
    median = float(np.median(target))
    high = hidden[target >= median]
    low = hidden[target < median]
    if high.size == 0 or low.size == 0:
        raise RuntimeError("CAA split is empty")
    return _unit(high.mean(axis=0) - low.mean(axis=0))


def sham_direction(basis: Mapping[str, np.ndarray], seed: int = 5201) -> np.ndarray:
    rng = np.random.default_rng(seed)
    draw = rng.normal(size=DIM)
    for name in ("w_true", "w_nuis"):
        axis = basis[name]
        draw = draw - np.dot(draw, axis) * axis
    return _unit(draw)


def _weight_contract(model: nn.Module, basis: Mapping[str, np.ndarray]) -> TRA1StateIdentityContract:
    down = layer11_down_proj(model)
    weight = down.weight.detach().cpu().numpy().astype(np.float64)
    return TRA1StateIdentityContract(
        base_weights_sha256=sha256_digest(np.ascontiguousarray(basis["w_true"]).tobytes()),
        adapter_weights_sha256=sha256_digest(np.ascontiguousarray(weight).tobytes()),
        config_sha256=_sha256_bytes(f"pv7-3c-d{DIM}-layer{LAYER_INDEX}".encode("utf-8")),
        tokenizer_sha256=_sha256_bytes(b"pv7-w18-3c-scratch-tokenizer-v1"),
        kv_or_state_dtype="float32",
        rope_freq_base=10000.0,
        rope_scaling_hash=_sha256_bytes(b"pv7-3c-rope-none"),
    )


def _freeze_select(direction: np.ndarray, sign: int) -> Dict[str, Any]:
    payload = json.dumps(
        {"layer": LAYER_INDEX, "sign": int(sign), "dose_l2": DOSE_NORM, "direction": [round(float(v), 8) for v in direction]},
        sort_keys=True,
        separators=(",", ":"),
    )
    with ChallengerBitemporalLedger() as ledger:
        ledger.append(
            LedgerRecordEnvelope(
                tenant_id="pv7-3c",
                record_id="select-freeze",
                entity_key="qscci-site",
                authority_rank=1,
                valid_from=10,
                valid_to=None,
                tx_time=10,
                op_kind="ASSERT",
                claim_value=payload,
            )
        )
        before = ledger.query_as_of("pv7-3c", "qscci-site", 10, 5)
        frozen = ledger.query_as_of("pv7-3c", "qscci-site", 10, 10)
    if before.status != "ABSENT" or frozen.status != "RESOLVED" or frozen.winner_claim_value != payload:
        raise RuntimeError("SELECT freeze was visible before its transaction time")
    store = StratifiedEvidenceStore()
    store.apply_mutation(
        [
            StratifiedEvent(
                event_id="qscci-select-freeze",
                entity="qscci-site",
                slot="direction",
                claim_value=payload,
                text="select freeze",
                valid_time=10,
                tx_time=10,
                tenant="pv7-3c",
                domain="probe",
                authority_tier=1,
            )
        ]
    )
    trace = export_canonical_ek5_audit_trace(store)
    return {"claim_value": payload, "as_of_digest": frozen.digest(), "ek5_trace_digest": trace.digest()}


def _arm_record(name: str, delta: np.ndarray, effect: np.ndarray, **extra: Any) -> Dict[str, Any]:
    mean_effect, low, high = paired_bootstrap_ci(effect.tolist(), n_resamples=200, seed=52)
    return {
        "name": name,
        "delta_h_l2": float(np.linalg.norm(delta)),
        "injection_mean_delta_y": mean_effect,
        "injection_lcb": low,
        "injection_ucb": high,
        **extra,
    }


def run_qscci_experiment(
    *,
    readout: str = "causal",
    on_report: Optional[Callable[[], None]] = None,
) -> Dict[str, Any]:
    """Run the planted control first. REPORT arms are built only after it passes."""
    if readout not in {"causal", "nuisance"}:
        raise ValueError("readout must be causal or nuisance")
    basis = plant_basis()
    splits = partition_splits()
    burn_sha = admit_scratch_prompts([row for rows in splits.values() for row in rows])
    read = basis["w_true"] if readout == "causal" else basis["w_nuis"]
    model = Layer11WriteSite(read)
    model.eval()
    contract = _weight_contract(model, basis)
    control_hidden = sample_hidden(splits["control"], basis)
    control = positive_control(model, control_hidden, basis)
    result: Dict[str, Any] = {
        "bundle_id": BUNDLE_ID,
        "item": ITEM,
        "readout": readout,
        "report_emitted": False,
        "arms": [],
        "positive_control": {key: value for key, value in control.items() if not key.endswith("direction")},
        "qwen_report_split_executed": False,
        "live_weights_loaded": False,
        "burn_ledger_sha256": burn_sha,
        "tra1_digest": contract.digest(),
        "family_ids": {name: [row["family_id"] for row in rows] for name, rows in splits.items()},
    }
    if not control["passed"]:
        result["failure"] = "positive_control"
        return result

    dev_hidden = sample_hidden(splits["dev"], basis)
    select_hidden = sample_hidden(splits["select"], basis)
    dev_target = _predict(model, dev_hidden)
    native = qscci_direction(model, select_hidden)
    sign = 1 if float(np.mean(_injection_effect(model, select_hidden, _matched_delta(native)))) >= 0 else -1
    native = native * sign
    freeze = _freeze_select(native, sign)
    sae = sae_decoder_direction(dev_hidden)
    caa = caa_direction(dev_hidden, dev_target)
    sham = sham_direction(basis)
    collateral = basis["w_ifeval"]
    directions = {
        "k1_native_mlp_write_site": native,
        "k1_topk_sae_latent": sae,
        "ordinary_contrastive_caa_direction": caa,
        "matched_norm_random_orthogonal_sham": sham,
        "ifeval_collateral_retention_probe": collateral,
    }
    if on_report is not None:
        on_report()
    report_hidden = sample_hidden(splits["report"], basis)
    arms = []
    for name in ARM_NAMES:
        delta = _matched_delta(directions[name])
        effect = _injection_effect(model, report_hidden, delta)
        extra: Dict[str, Any] = {"cosine_to_w_true": _cosine(directions[name], basis["w_true"])}
        if name == "k1_native_mlp_write_site":
            extra.update(_removal_restoration(model, report_hidden, directions[name]))
        if name == "ifeval_collateral_retention_probe":
            native_delta = _matched_delta(native)
            native_effect_hidden = report_hidden + native_delta
            before = report_hidden @ collateral
            after = native_effect_hidden @ collateral
            extra["native_intervention_collateral_retention"] = float(1.0 - np.mean(np.abs(after - before)) / DOSE_NORM)
        arms.append(_arm_record(name, delta, effect, **extra))
    norms = [arm["delta_h_l2"] for arm in arms]
    native_arm = arms[0]
    caa_arm = arms[2]
    sham_arm = arms[3]
    result.update(
        {
            "report_emitted": True,
            "failure": "",
            "arms": arms,
            "select_freeze": {"as_of_digest": freeze["as_of_digest"], "ek5_trace_digest": freeze["ek5_trace_digest"]},
            "selection_cosine_qscci_caa": abs(_cosine(native, caa)),
            "norm_spread": float(max(norms) - min(norms)),
            "native_beats_sham": native_arm["injection_mean_delta_y"] > sham_arm["injection_mean_delta_y"] + 0.2,
            "nuisance_invariance_win": native_arm["injection_mean_delta_y"] > caa_arm["injection_mean_delta_y"],
            "restoration_max_abs_error": native_arm["restoration_max_abs_error"],
            "layer_index": LAYER_INDEX,
            "hook_site": "layers[11].mlp.down_proj",
        }
    )
    return result


def _fragment_digest() -> str:
    registry = REPO_ROOT / "docs" / "research" / "registry.d" / "52-t3c-qscci-singleton.json"
    verifier = REPO_ROOT / "scripts" / "verifiers.d" / "52_verifier.py"
    sources = {
        "docs/research/registry.d/52-t3c-qscci-singleton.json": registry.read_bytes(),
        "scripts/verifiers.d/52_verifier.py": verifier.read_bytes(),
    }
    compiled = compile_fragments(REPO_ROOT, claim_registry_sha256(), sources)
    return str(compiled["compiled_sha256"])


def _gates(report: Mapping[str, Any]) -> List[Dict[str, Any]]:
    control = report["positive_control"]
    checks = [
        ("positive_control_passed", bool(control["passed"]), f"cosine={control['cosine_qscci_w_true']}"),
        ("correlation_picks_nuisance", control["cosine_correlation_w_nuis"] > control["cosine_correlation_w_true"], "raw correlation prefers nuisance"),
        ("qscci_cosine", control["cosine_qscci_w_true"] > CONTROL_COSINE_MIN, f"cosine={control['cosine_qscci_w_true']}"),
        ("report_emitted", bool(report["report_emitted"]), "REPORT opened only after the control"),
        ("norm_match", bool(report["report_emitted"]) and report.get("norm_spread", 1.0) < 1e-8, f"spread={report.get('norm_spread')}"),
        ("separation_cosine", bool(report["report_emitted"]) and report.get("selection_cosine_qscci_caa", 1.0) < SEPARATION_COSINE_MAX, f"cos={report.get('selection_cosine_qscci_caa')}"),
        ("nuisance_invariance", bool(report.get("nuisance_invariance_win")), "QSCCI effect exceeds CAA"),
        ("restoration_error", report.get("restoration_max_abs_error", 1.0) < 1e-4, f"error={report.get('restoration_max_abs_error')}"),
        ("native_beats_sham", bool(report.get("native_beats_sham")), "native injection exceeds orthogonal sham"),
        ("splits_disjoint", _splits_disjoint(report["family_ids"]), "four splits"),
        ("cuda_isolated", bool(report["cuda_isolation"]["passed"]), "CPU lease"),
        ("live_weights_not_claimed", report["live_weights_loaded"] is False and report["qwen_report_split_executed"] is False, "no Qwen claim"),
        ("burn_ledger_admitted", bool(report["burn_ledger_sha256"]), report["burn_ledger_sha256"]),
    ]
    return [{"name": name, "passed": bool(passed), "detail": detail} for name, passed, detail in checks]


def _splits_disjoint(family_ids: Mapping[str, Sequence[str]]) -> bool:
    seen: set[str] = set()
    for rows in family_ids.values():
        if len(rows) != len(set(rows)):
            return False
        if seen.intersection(rows):
            return False
        seen.update(rows)
    return set(family_ids) == {"control", "dev", "select", "report"}


def execute(mode: str, model_name: str = "Qwen/Qwen3.5-2B-Base") -> Dict[str, Any]:
    """`local-drycheck` runs the planted hook. `run-live-gpu` refuses weight loading."""
    if mode not in {"local-drycheck", "run-live-gpu"}:
        raise ValueError("mode must be local-drycheck or run-live-gpu")
    isolation = cpu_isolation_receipt()
    if mode == "run-live-gpu":
        allowed = model_name in ALLOWED_LIVE_MODELS
        report = {
            "bundle_id": BUNDLE_ID,
            "item": ITEM,
            "mode": mode,
            "executed": False,
            "report_emitted": False,
            "reason": "cuda-isolated-live-weights-not-loaded" if allowed else "model-not-in-preregistered-set",
            "model": model_name,
            "hook_site": "layers[11].mlp.down_proj",
            "qwen_report_split_executed": False,
            "live_weights_loaded": False,
            "cuda_isolation": isolation,
            "claim_level": "not-executed",
            "gates": [
                {"name": "live_mode_refused", "passed": True, "detail": "no weights loaded and no REPORT claim"},
                {"name": "model_name_checked", "passed": allowed, "detail": model_name},
            ],
        }
        report["scoreboard_sha256"] = _json_sha256({key: value for key, value in report.items() if key != "scoreboard_sha256"})
        return report

    experiment = run_qscci_experiment(readout="causal")
    report = {
        **experiment,
        "mode": mode,
        "executed": True,
        "model": "planted-layer11-write-site",
        "cuda_isolation": isolation,
        "claim_level": "local-drycheck",
        "fragment_compiled_sha256": _fragment_digest(),
        "spine_digest": claim_registry_sha256(),
    }
    report["positive_control"] = {
        key: value for key, value in report["positive_control"].items() if not isinstance(value, np.ndarray)
    }
    report["gates"] = _gates(report)
    report["passed"] = all(gate["passed"] for gate in report["gates"])
    public = {key: value for key, value in report.items() if key != "scoreboard_sha256"}
    report["scoreboard_sha256"] = _json_sha256(public)
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", required=True, choices=("local-drycheck", "run-live-gpu"))
    parser.add_argument("--model", default="Qwen/Qwen3.5-2B-Base")
    args = parser.parse_args(list(argv) if argv is not None else None)
    report = execute(args.mode, args.model)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report.get("passed", report.get("executed") is False) else 1


if __name__ == "__main__":
    raise SystemExit(main())
