"""Wave-14B (`Demo D11` / `PM-6`) Arm 7: White-Box CAA & Contrastive Logit Suppression Baseline.

Implements a lightweight, DEV-calibrated Activation Steering / Contrastive Activation Addition
(CAA) + optional DEV-derived contrastive token-logit suppression reference baseline for
`Qwen/Qwen2.5-7B-Instruct` (and compatible PyTorch transformer models).

Governance & Scope Contract:
  - Direction vectors `v_l = normalize(mean(h_restricted, l) - mean(h_benign, l))` and
    salient restricted token sets are extracted STRICTLY from `DEV` pairs.
  - Fail-closed `assert_zero_test_overlap` rejects any `TEST` family ID or text digest overlap.
  - Compared strictly against `Qwen2.5-7B` arms (never across model sizes) and tagged with
    `scope_disclosure = "untuned_lightweight_literature_reference_qwen25_7b_only"`.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import Counter
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np
import torch
from torch import nn

CAA_SCOPE_DISCLOSURE = "untuned_lightweight_literature_reference_qwen25_7b_only"
DEFAULT_ALPHA_GRID: Tuple[float, ...] = (0.5, 1.0, 1.5, 2.0, 3.0)
DEFAULT_QWEN25_7B_LAYERS: Tuple[int, ...] = (14, 15, 16, 17, 18, 19, 20)

# Carry the sweep deadline into each generation in the runner's alpha evaluator.
_CALIBRATION_DEADLINE: ContextVar[Optional[float]] = ContextVar(
    "_CALIBRATION_DEADLINE", default=None
)


class _CalibrationDeadlineExceeded(TimeoutError):
    """The current DEV candidate could not be evaluated in full."""


def _calibration_time_remaining() -> Optional[float]:
    deadline = _CALIBRATION_DEADLINE.get()
    if deadline is None:
        return None
    remaining = deadline - time.perf_counter()
    if remaining <= 0:
        raise _CalibrationDeadlineExceeded("CAA DEV calibration deadline reached")
    return remaining


def _sha256_str(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _normalize_text_key(text: str) -> str:
    return " ".join(text.strip().lower().split())


def assert_zero_test_overlap(
    dev_family_ids: Sequence[str],
    test_family_ids: Sequence[str],
    *,
    dev_texts: Optional[Sequence[str]] = None,
    test_texts: Optional[Sequence[str]] = None,
    context: str = "caa_dev_extraction",
) -> None:
    """Fail-closed leakage guard asserting zero overlap between DEV and TEST families or texts."""
    dev_set = {str(x) for x in dev_family_ids}
    test_set = {str(x) for x in test_family_ids}
    overlap_ids = sorted(dev_set & test_set)
    if overlap_ids:
        raise ValueError(
            f"Zero-test-leakage violation in {context}: "
            f"{len(overlap_ids)} overlapping work_family_ids (e.g., {overlap_ids[:5]})"
        )
    if dev_texts is not None and test_texts is not None:
        dev_hashes = {_sha256_str(_normalize_text_key(t)) for t in dev_texts if str(t).strip()}
        test_hashes = {_sha256_str(_normalize_text_key(t)) for t in test_texts if str(t).strip()}
        overlap_h = sorted(dev_hashes & test_hashes)
        if overlap_h:
            raise ValueError(
                f"Zero-test-leakage violation in {context}: "
                f"{len(overlap_h)} overlapping normalized text SHA-256 digests"
            )


def resolve_transformer_layers(model: nn.Module) -> nn.ModuleList:
    """Locate the decoder transformer block `nn.ModuleList` on HF or synthetic PyTorch models."""
    candidates: List[Any] = []
    inner = getattr(model, "model", None)
    if inner is not None:
        candidates.append(getattr(inner, "layers", None))
        candidates.append(getattr(inner, "blocks", None))
    transformer = getattr(model, "transformer", None)
    if transformer is not None:
        candidates.append(getattr(transformer, "h", None))
        candidates.append(getattr(transformer, "layers", None))
    candidates.append(getattr(model, "layers", None))
    candidates.append(getattr(model, "blocks", None))

    for cand in candidates:
        if isinstance(cand, nn.ModuleList) and len(cand) > 0:
            return cand
    raise ValueError(
        "Unable to locate transformer decoder layers (expected model.model.layers, "
        "model.transformer.h, model.layers, or model.blocks as a non-empty nn.ModuleList)."
    )


def _extract_hidden_tensor(layer_output: Any) -> torch.Tensor:
    if isinstance(layer_output, torch.Tensor):
        return layer_output
    if isinstance(layer_output, (tuple, list)) and len(layer_output) > 0 and isinstance(layer_output[0], torch.Tensor):
        return layer_output[0]
    raise TypeError(f"Unsupported transformer layer output type for CAA hook: {type(layer_output)!r}")


def _replace_hidden_tensor(layer_output: Any, new_hidden: torch.Tensor) -> Any:
    if isinstance(layer_output, torch.Tensor):
        return new_hidden
    if isinstance(layer_output, tuple):
        return (new_hidden,) + layer_output[1:]
    if isinstance(layer_output, list):
        return [new_hidden, *layer_output[1:]]
    raise TypeError(f"Unsupported transformer layer output type for CAA steering: {type(layer_output)!r}")


def _model_device(model: nn.Module) -> torch.device:
    for param in model.parameters():
        return param.device
    return torch.device("cpu")


def _tokenize_batch(
    tokenizer: Any,
    texts: Sequence[str],
    *,
    max_length: int,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    if getattr(tokenizer, "pad_token", None) is None and getattr(tokenizer, "pad_token_id", None) is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        eos_id = getattr(tokenizer, "eos_token_id", None)
        if eos_token is not None:
            tokenizer.pad_token = eos_token
        elif eos_id is not None:
            tokenizer.pad_token_id = eos_id
        else:
            raise ValueError("CAA batching requires a pad token or an EOS token to use as padding")
    encoded = tokenizer(
        list(texts),
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length,
    )
    if not isinstance(encoded, Mapping):
        encoded = dict(encoded)
    out: Dict[str, torch.Tensor] = {}
    for key, val in encoded.items():
        if isinstance(val, torch.Tensor):
            out[str(key)] = val.to(device)
    if "input_ids" not in out:
        raise ValueError("Tokenizer output missing required 'input_ids' tensor")
    if "attention_mask" not in out:
        out["attention_mask"] = torch.ones_like(out["input_ids"], dtype=torch.long, device=device)
    return out


def _pool_last_valid_token(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Pool the last non-padding token hidden state `[B, H]` in float32 on CPU."""
    if hidden.ndim != 3:
        raise ValueError(f"Expected 3D hidden tensor [B, T, H], got shape {tuple(hidden.shape)}")
    mask = attention_mask.to(device=hidden.device)
    positions = torch.arange(mask.shape[1], device=mask.device).unsqueeze(0)
    # Keep index zero as the fallback for an entirely masked row.
    last_indices = torch.where(mask > 0, positions, 0).amax(dim=1)
    batch_idx = torch.arange(hidden.shape[0], device=hidden.device)
    pooled = hidden[batch_idx, last_indices, :]
    return pooled.detach().to(dtype=torch.float32, device="cpu")


@dataclass(frozen=True)
class CAADirectionSet:
    """Frozen DEV-extracted normalized contrastive activation directions per transformer layer."""

    directions: Dict[int, torch.Tensor]  # layer_idx -> unit-norm [H] float32 CPU tensor
    raw_norms: Dict[int, float]
    target_layers: Tuple[int, ...]
    n_restricted_prompts: int
    n_benign_prompts: int
    dev_ids_sha256: str
    directions_sha256: str
    scope_disclosure: str = CAA_SCOPE_DISCLOSURE

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "target_layers": list(self.target_layers),
            "raw_norms": {str(k): round(float(v), 6) for k, v in sorted(self.raw_norms.items())},
            "n_restricted_prompts": int(self.n_restricted_prompts),
            "n_benign_prompts": int(self.n_benign_prompts),
            "dev_ids_sha256": self.dev_ids_sha256,
            "directions_sha256": self.directions_sha256,
            "scope_disclosure": self.scope_disclosure,
        }


@dataclass(frozen=True)
class ContrastiveLogitPenalty:
    """Optional DEV-derived contrastive token-logit suppressor (HF LogitsProcessor compatible)."""

    token_ids: Tuple[int, ...]
    token_weights: Tuple[float, ...]  # positive weights in (0, 1]; subtracted scaled by penalty_strength
    penalty_strength: float
    dev_ids_sha256: str
    scope_disclosure: str = CAA_SCOPE_DISCLOSURE

    def apply_to_logits(self, logits: torch.Tensor, *, scale: float = 1.0) -> torch.Tensor:
        """Subtract scaled contrastive penalties from the final-step vocabulary logits."""
        if not self.token_ids or self.penalty_strength <= 0.0 or scale <= 0.0:
            return logits
        out = logits.clone()
        idx = torch.tensor(self.token_ids, dtype=torch.long, device=out.device)
        valid = idx < out.shape[-1]
        if not bool(valid.any()):
            return out
        idx = idx[valid]
        weights = torch.tensor(self.token_weights, dtype=out.dtype, device=out.device)[valid]
        delta = weights * float(self.penalty_strength) * float(scale)
        if out.ndim == 2:
            out[:, idx] = out[:, idx] - delta.unsqueeze(0)
        elif out.ndim == 3:
            out[:, -1, idx] = out[:, -1, idx] - delta.unsqueeze(0)
        else:
            raise ValueError(f"Expected 2D or 3D logits tensor, got shape {tuple(out.shape)}")
        return out

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
        _ = input_ids
        return self.apply_to_logits(scores, scale=1.0)

    def to_metadata(self) -> Dict[str, Any]:
        return {
            "n_salient_tokens": len(self.token_ids),
            "penalty_strength": float(self.penalty_strength),
            "dev_ids_sha256": self.dev_ids_sha256,
            "scope_disclosure": self.scope_disclosure,
        }


def extract_contrastive_directions(
    model: nn.Module,
    tokenizer: Any,
    dev_restricted_prompts: Sequence[str],
    dev_benign_prompts: Sequence[str],
    *,
    dev_family_ids: Sequence[str],
    test_family_ids: Sequence[str],
    test_texts: Optional[Sequence[str]] = None,
    target_layers: Sequence[int] = DEFAULT_QWEN25_7B_LAYERS,
    max_length: int = 256,
    batch_size: int = 4,
    device: Optional[torch.device] = None,
) -> CAADirectionSet:
    """Extract normalized `v_l = normalize(mean(h_pos) - mean(h_benign))` via PyTorch forward hooks.

    Strictly restricted to `DEV` pairs; fails closed if any `TEST` family ID or text overlaps.
    """
    if not dev_restricted_prompts or not dev_benign_prompts:
        raise ValueError("Both dev_restricted_prompts and dev_benign_prompts must be non-empty")
    assert_zero_test_overlap(
        dev_family_ids,
        test_family_ids,
        dev_texts=list(dev_restricted_prompts) + list(dev_benign_prompts),
        test_texts=test_texts,
        context="extract_contrastive_directions",
    )

    layers = resolve_transformer_layers(model)
    n_layers = len(layers)
    resolved_layers = tuple(sorted({int(idx) for idx in target_layers}))
    if not resolved_layers:
        raise ValueError("target_layers must be non-empty")
    for idx in resolved_layers:
        if idx < 0 or idx >= n_layers:
            raise ValueError(f"Layer index {idx} out of range for model with {n_layers} layers")

    run_device = device if device is not None else _model_device(model)
    dev_ids_sha = _sha256_str("|".join(sorted({str(x) for x in dev_family_ids})))

    def _collect_mean_states(prompts: Sequence[str]) -> Dict[int, torch.Tensor]:
        sums: Dict[int, Optional[torch.Tensor]] = {idx: None for idx in resolved_layers}
        counts: Dict[int, int] = {idx: 0 for idx in resolved_layers}
        current_mask: Dict[str, Optional[torch.Tensor]] = {"mask": None}
        handles: List[torch.utils.hooks.RemovableHandle] = []

        def _make_hook(layer_idx: int) -> Callable[[nn.Module, Tuple[Any, ...], Any], None]:
            def _hook(_mod: nn.Module, _inp: Tuple[Any, ...], out: Any) -> None:
                mask = current_mask["mask"]
                if mask is None:
                    raise RuntimeError("Attention mask missing during CAA activation capture")
                hidden = _extract_hidden_tensor(out)
                pooled = _pool_last_valid_token(hidden, mask)
                batch_sum = pooled.sum(dim=0)
                if sums[layer_idx] is None:
                    sums[layer_idx] = batch_sum
                else:
                    sums[layer_idx] = sums[layer_idx] + batch_sum
                counts[layer_idx] += int(pooled.shape[0])

            return _hook

        for idx in resolved_layers:
            handles.append(layers[idx].register_forward_hook(_make_hook(idx)))

        was_training = model.training
        model.eval()
        try:
            with torch.no_grad():
                for start in range(0, len(prompts), max(1, int(batch_size))):
                    batch = prompts[start : start + max(1, int(batch_size))]
                    enc = _tokenize_batch(tokenizer, batch, max_length=max_length, device=run_device)
                    current_mask["mask"] = enc["attention_mask"]
                    model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"])
        finally:
            current_mask["mask"] = None
            for handle in handles:
                handle.remove()
            if was_training:
                model.train()

        means: Dict[int, torch.Tensor] = {}
        for idx in resolved_layers:
            total_vec = sums[idx]
            if total_vec is None or counts[idx] == 0:
                raise RuntimeError(f"CAA forward hook captured 0 activations on layer {idx}")
            means[idx] = total_vec / float(counts[idx])
        return means

    pos_means = _collect_mean_states(dev_restricted_prompts)
    ben_means = _collect_mean_states(dev_benign_prompts)

    directions: Dict[int, torch.Tensor] = {}
    raw_norms: Dict[int, float] = {}
    hasher = hashlib.sha256()
    for idx in resolved_layers:
        diff = pos_means[idx] - ben_means[idx]
        norm = float(torch.linalg.vector_norm(diff).item())
        raw_norms[idx] = norm
        unit = diff / max(norm, 1e-12)
        directions[idx] = unit
        hasher.update(f"layer:{idx}:".encode("ascii"))
        hasher.update(np.ascontiguousarray(unit.numpy(), dtype=np.float32).tobytes())

    return CAADirectionSet(
        directions=directions,
        raw_norms=raw_norms,
        target_layers=resolved_layers,
        n_restricted_prompts=len(dev_restricted_prompts),
        n_benign_prompts=len(dev_benign_prompts),
        dev_ids_sha256=dev_ids_sha,
        directions_sha256=hasher.hexdigest(),
        scope_disclosure=CAA_SCOPE_DISCLOSURE,
    )


def build_dev_contrastive_logit_penalty(
    tokenizer: Any,
    dev_restricted_texts: Sequence[str],
    dev_benign_texts: Sequence[str],
    *,
    dev_family_ids: Sequence[str],
    test_family_ids: Sequence[str],
    test_texts: Optional[Sequence[str]] = None,
    top_k_tokens: int = 128,
    min_freq: int = 2,
    penalty_strength: float = 2.5,
) -> ContrastiveLogitPenalty:
    """Compute DEV-derived contrastive token-logit suppression weights (zero TEST leakage)."""
    assert_zero_test_overlap(
        dev_family_ids,
        test_family_ids,
        dev_texts=list(dev_restricted_texts) + list(dev_benign_texts),
        test_texts=test_texts,
        context="build_dev_contrastive_logit_penalty",
    )

    special_ids: Set[int] = set()
    for attr in ("bos_token_id", "eos_token_id", "pad_token_id", "unk_token_id"):
        val = getattr(tokenizer, attr, None)
        if isinstance(val, int) and val >= 0:
            special_ids.add(val)
    all_special = getattr(tokenizer, "all_special_ids", None)
    if isinstance(all_special, Iterable):
        for sid in all_special:
            if isinstance(sid, int) and sid >= 0:
                special_ids.add(sid)

    def _encode_ids(text: str) -> List[int]:
        if hasattr(tokenizer, "encode"):
            ids = tokenizer.encode(text, add_special_tokens=False)
            return [int(t) for t in ids]
        enc = tokenizer([text], return_tensors="pt", padding=False, truncation=True, max_length=512)
        return [int(t) for t in enc["input_ids"][0].tolist()]

    pos_counts: Counter[int] = Counter()
    ben_counts: Counter[int] = Counter()
    for txt in dev_restricted_texts:
        for tid in _encode_ids(str(txt)):
            if tid not in special_ids:
                pos_counts[tid] += 1
    for txt in dev_benign_texts:
        for tid in _encode_ids(str(txt)):
            if tid not in special_ids:
                ben_counts[tid] += 1

    n_pos = max(1, sum(pos_counts.values()))
    n_ben = max(1, sum(ben_counts.values()))
    vocab_est = max(len(set(pos_counts.keys()) | set(ben_counts.keys())), 1)

    scored: List[Tuple[float, int]] = []
    for tid, cnt in pos_counts.items():
        if cnt < int(min_freq):
            continue
        p_pos = (float(cnt) + 1.0) / (float(n_pos) + float(vocab_est))
        p_ben = (float(ben_counts.get(tid, 0)) + 1.0) / (float(n_ben) + float(vocab_est))
        log_odds = math.log(p_pos / p_ben)
        if log_odds > 0.0:
            scored.append((log_odds, tid))

    scored.sort(key=lambda pair: (-pair[0], pair[1]))
    top = scored[: max(0, int(top_k_tokens))]
    max_score = top[0][0] if top else 1.0
    token_ids = tuple(tid for _, tid in top)
    token_weights = tuple(round(float(sc / max(max_score, 1e-12)), 6) for sc, _ in top)
    dev_ids_sha = _sha256_str("|".join(sorted({str(x) for x in dev_family_ids})))

    return ContrastiveLogitPenalty(
        token_ids=token_ids,
        token_weights=token_weights,
        penalty_strength=float(penalty_strength),
        dev_ids_sha256=dev_ids_sha,
        scope_disclosure=CAA_SCOPE_DISCLOSURE,
    )


class CAASteeringController:
    """Context-managed PyTorch forward-hook residual steering + optional lm_head logit penalty."""

    def __init__(
        self,
        model: nn.Module,
        direction_set: CAADirectionSet,
        *,
        alpha: float = 1.0,
        logit_penalty: Optional[ContrastiveLogitPenalty] = None,
        logit_penalty_scale: float = 1.0,
        apply_logit_hook_on_lm_head: bool = False,
    ) -> None:
        self.model = model
        self.direction_set = direction_set
        self.alpha = float(alpha)
        self.logit_penalty = logit_penalty
        self.logit_penalty_scale = float(logit_penalty_scale)
        self.apply_logit_hook_on_lm_head = bool(apply_logit_hook_on_lm_head)
        self._handles: List[torch.utils.hooks.RemovableHandle] = []
        self.hook_invocations: int = 0
        self.logit_hook_invocations: int = 0

    def __enter__(self) -> "CAASteeringController":
        layers = resolve_transformer_layers(self.model)
        self._handles = []
        self.hook_invocations = 0
        self.logit_hook_invocations = 0

        for layer_idx, unit_vec in self.direction_set.directions.items():
            vec_cpu = unit_vec.detach().clone()

            def _make_steer_hook(v_ref: torch.Tensor) -> Callable[[nn.Module, Tuple[Any, ...], Any], Any]:
                def _steer_hook(_mod: nn.Module, _inp: Tuple[Any, ...], out: Any) -> Any:
                    self.hook_invocations += 1
                    if self.alpha == 0.0:
                        return out
                    hidden = _extract_hidden_tensor(out)
                    v_cast = v_ref.to(device=hidden.device, dtype=hidden.dtype)
                    steered = hidden - (self.alpha * v_cast).view(1, 1, -1)
                    return _replace_hidden_tensor(out, steered)

                return _steer_hook

            self._handles.append(layers[layer_idx].register_forward_hook(_make_steer_hook(vec_cpu)))

        if self.apply_logit_hook_on_lm_head and self.logit_penalty is not None:
            lm_head = getattr(self.model, "lm_head", None)
            if isinstance(lm_head, nn.Module):
                penalty = self.logit_penalty
                p_scale = self.logit_penalty_scale

                def _lm_hook(_mod: nn.Module, _inp: Tuple[Any, ...], out: Any) -> Any:
                    self.logit_hook_invocations += 1
                    if isinstance(out, torch.Tensor):
                        return penalty.apply_to_logits(out, scale=p_scale)
                    return out

                self._handles.append(lm_head.register_forward_hook(_lm_hook))

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


@dataclass
class CAACalibrationResult:
    """Result of DEV-only CAA multiplier sweep (`alpha in (0.5, 1.0, 1.5, 2.0, 3.0)`)."""

    selected_alpha: float
    target_fpr: float
    met_target_fpr: bool
    selected_r1_r3_suppression_rate: Optional[float]
    selected_r5_benign_fpr: Optional[float]
    wall_clock_s: float
    timebox_s: float
    timebox_exceeded: bool
    dev_ids_sha256: str
    sweep_results: List[Dict[str, Any]] = field(default_factory=list)
    scope_disclosure: str = CAA_SCOPE_DISCLOSURE

    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_alpha": float(self.selected_alpha),
            "target_fpr": float(self.target_fpr),
            "met_target_fpr": bool(self.met_target_fpr),
            "selected_r1_r3_suppression_rate": self.selected_r1_r3_suppression_rate,
            "selected_r5_benign_fpr": self.selected_r5_benign_fpr,
            "wall_clock_s": round(float(self.wall_clock_s), 4),
            "timebox_s": float(self.timebox_s),
            "timebox_exceeded": bool(self.timebox_exceeded),
            "dev_ids_sha256": self.dev_ids_sha256,
            "sweep_results": list(self.sweep_results),
            "scope_disclosure": self.scope_disclosure,
        }


def calibrate_caa_multiplier_on_dev(
    evaluate_alpha_fn: Callable[[float], Mapping[str, Any]],
    *,
    dev_family_ids: Sequence[str],
    test_family_ids: Sequence[str],
    alpha_grid: Sequence[float] = DEFAULT_ALPHA_GRID,
    target_fpr: float = 0.02,
    timebox_s: float = 300.0,
) -> CAACalibrationResult:
    """Sweep `alpha` strictly on `DEV` to maximize `R1/R3` suppression subject to `R5 FPR <= target_fpr`.

    `evaluate_alpha_fn(alpha)` must return a mapping containing at least:
      - `r1_r3_suppression_rate`: float in [0, 1]
      - `r5_benign_fpr`: float in [0, 1]
    Optional extra fields (`r1_suppression_rate`, `r3_suppression_rate`, `r5_coherence_pass_rate`)
    are preserved in `sweep_results`. CAA generation checks the deadline per call.
    Only fully evaluated candidates are eligible. If the sweep finds no feasible
    candidate, alpha zero is selected and its rates are null unless it was measured.
    """
    assert_zero_test_overlap(
        dev_family_ids,
        test_family_ids,
        context="calibrate_caa_multiplier_on_dev",
    )
    if not alpha_grid:
        raise ValueError("alpha_grid must be non-empty")

    timebox = float(timebox_s)
    if not math.isfinite(timebox) or timebox < 0:
        raise ValueError("timebox_s must be finite and nonnegative")

    t0 = time.perf_counter()
    dev_ids_sha = _sha256_str("|".join(sorted({str(x) for x in dev_family_ids})))
    sweep_rows: List[Dict[str, Any]] = []
    deadline = t0 + timebox
    parent_deadline = _CALIBRATION_DEADLINE.get()
    if parent_deadline is not None:
        deadline = min(deadline, parent_deadline)
    deadline_token = _CALIBRATION_DEADLINE.set(deadline)
    timebox_exceeded = False

    try:
        for alpha in alpha_grid:
            _calibration_time_remaining()
            step_t0 = time.perf_counter()
            metrics = dict(evaluate_alpha_fn(float(alpha)))
            _calibration_time_remaining()
            if "r1_r3_suppression_rate" not in metrics or "r5_benign_fpr" not in metrics:
                raise KeyError("evaluate_alpha_fn must return 'r1_r3_suppression_rate' and 'r5_benign_fpr'")
            supp = float(metrics["r1_r3_suppression_rate"])
            fpr = float(metrics["r5_benign_fpr"])
            if not (0.0 <= supp <= 1.0 and 0.0 <= fpr <= 1.0):
                raise ValueError("CAA suppression and benign FPR must be finite rates in [0, 1]")
            sweep_rows.append({
                **metrics,
                "alpha": float(alpha),
                "r1_r3_suppression_rate": supp,
                "r5_benign_fpr": fpr,
                "meets_target_fpr": fpr <= float(target_fpr) + 1e-12,
                "eval_wall_s": round(time.perf_counter() - step_t0, 4),
            })
    except _CalibrationDeadlineExceeded:
        timebox_exceeded = True
    finally:
        _CALIBRATION_DEADLINE.reset(deadline_token)

    feasible = [r for r in sweep_rows if r["meets_target_fpr"]]
    if feasible:
        # Maximize suppression, tie-break by lower benign FPR, then smaller alpha.
        best = max(feasible, key=lambda r: (r["r1_r3_suppression_rate"], -r["r5_benign_fpr"], -r["alpha"]))
    else:
        best = next((r for r in sweep_rows if r["alpha"] == 0.0), None)

    return CAACalibrationResult(
        selected_alpha=float(best["alpha"]) if best is not None else 0.0,
        target_fpr=float(target_fpr),
        met_target_fpr=bool(feasible),
        selected_r1_r3_suppression_rate=float(best["r1_r3_suppression_rate"]) if best is not None else None,
        selected_r5_benign_fpr=float(best["r5_benign_fpr"]) if best is not None else None,
        wall_clock_s=time.perf_counter() - t0,
        timebox_s=timebox,
        timebox_exceeded=timebox_exceeded,
        dev_ids_sha256=dev_ids_sha,
        sweep_results=sweep_rows,
        scope_disclosure=CAA_SCOPE_DISCLOSURE,
    )


def generate_text_with_caa(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    direction_set: CAADirectionSet,
    *,
    alpha: float,
    logit_penalty: Optional[ContrastiveLogitPenalty] = None,
    logit_penalty_scale: float = 1.0,
    max_new_tokens: int = 96,
    max_input_length: int = 512,
    device: Optional[torch.device] = None,
) -> Tuple[str, Dict[str, Any]]:
    """Run deterministic greedy generation under active CAA hooks (works on HF and tiny nn.Modules)."""
    _calibration_time_remaining()
    run_device = device if device is not None else _model_device(model)
    enc = _tokenize_batch(tokenizer, [prompt], max_length=max_input_length, device=run_device)
    prompt_len = int(enc["input_ids"].shape[1])

    t0 = time.perf_counter()
    was_training = model.training
    model.eval()
    try:
        with CAASteeringController(
            model,
            direction_set,
            alpha=alpha,
            logit_penalty=logit_penalty,
            logit_penalty_scale=logit_penalty_scale,
            apply_logit_hook_on_lm_head=not hasattr(model, "generate"),
        ) as ctrl:
            with torch.no_grad():
                if hasattr(model, "generate"):
                    gen_kwargs: Dict[str, Any] = {
                        "input_ids": enc["input_ids"],
                        "attention_mask": enc["attention_mask"],
                        "max_new_tokens": int(max_new_tokens),
                        "do_sample": False,
                    }
                    if logit_penalty is not None and logit_penalty_scale > 0.0:
                        scaled_lp = ContrastiveLogitPenalty(
                            token_ids=logit_penalty.token_ids,
                            token_weights=logit_penalty.token_weights,
                            penalty_strength=logit_penalty.penalty_strength * float(logit_penalty_scale),
                            dev_ids_sha256=logit_penalty.dev_ids_sha256,
                            scope_disclosure=logit_penalty.scope_disclosure,
                        )
                        try:
                            from transformers import LogitsProcessorList

                            gen_kwargs["logits_processor"] = LogitsProcessorList([scaled_lp])
                        except ImportError:
                            gen_kwargs["logits_processor"] = [scaled_lp]
                    remaining = _calibration_time_remaining()
                    if remaining is not None:
                        gen_kwargs["max_time"] = remaining
                    out_ids = model.generate(**gen_kwargs)
                    gen_tokens = out_ids[0, prompt_len:].tolist()
                else:
                    # Pure-PyTorch greedy decode loop for synthetic unit-test transformer models
                    cur_ids = enc["input_ids"]
                    cur_mask = enc["attention_mask"]
                    eos_id = getattr(tokenizer, "eos_token_id", None)
                    gen_tokens = []
                    for _ in range(int(max_new_tokens)):
                        _calibration_time_remaining()
                        outputs = model(input_ids=cur_ids, attention_mask=cur_mask)
                        logits = outputs.logits if hasattr(outputs, "logits") else outputs
                        next_logits = logits[:, -1, :]
                        if logit_penalty is not None and not ctrl.apply_logit_hook_on_lm_head:
                            next_logits = logit_penalty.apply_to_logits(next_logits, scale=logit_penalty_scale)
                        next_tok = torch.argmax(next_logits, dim=-1, keepdim=True)
                        tok_int = int(next_tok.item())
                        gen_tokens.append(tok_int)
                        if eos_id is not None and tok_int == int(eos_id):
                            break
                        cur_ids = torch.cat([cur_ids, next_tok], dim=1)
                        cur_mask = torch.cat([cur_mask, torch.ones_like(next_tok)], dim=1)
            hook_calls = ctrl.hook_invocations
            logit_hook_calls = ctrl.logit_hook_invocations
    finally:
        if was_training:
            model.train()

    _calibration_time_remaining()
    if hasattr(tokenizer, "decode"):
        text = str(tokenizer.decode(gen_tokens, skip_special_tokens=True))
    else:
        text = " ".join(str(t) for t in gen_tokens)

    meta = {
        "alpha": float(alpha),
        "generated_tokens": len(gen_tokens),
        "hook_invocations": int(hook_calls),
        "logit_hook_invocations": int(logit_hook_calls),
        "wall_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        "scope_disclosure": CAA_SCOPE_DISCLOSURE,
    }
    return text, meta


__all__ = [
    "CAA_SCOPE_DISCLOSURE",
    "CAACalibrationResult",
    "CAADirectionSet",
    "CAASteeringController",
    "ContrastiveLogitPenalty",
    "DEFAULT_ALPHA_GRID",
    "DEFAULT_QWEN25_7B_LAYERS",
    "assert_zero_test_overlap",
    "build_dev_contrastive_logit_penalty",
    "calibrate_caa_multiplier_on_dev",
    "extract_contrastive_directions",
    "generate_text_with_caa",
    "resolve_transformer_layers",
]
